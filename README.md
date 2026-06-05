# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-05 | 今日论文总数: 701

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. MASF: A Multi-Model Adaptive Selection Framework for Abstractive Text summarization

**arXiv ID:** 2606.05494 | [PDF](https://arxiv.org/pdf/2606.05494v1)

**作者:** Ahmed Alansary `[一作]` (MSA University), Ali Hamdi `[通讯]` (MSA University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种多模型自适应摘要框架（MASF），利用三种Transformer模型（T5‑small、PEGASUS‑xsum、LED‑base）在CNN/DailyMail数据集上生成候选摘要，并通过自动评估指标（ROUGE‑L、BLEU、BERTScore）自适应选择最佳摘要；

**💡 创新点**

将多模型候选生成与基于多维评估指标的综合评分自适应选择结合，突破单模型鲁棒性不足的局限，并通过LoRA参数高效微调提升轻量模型性能；

**🔧 技术方法**

Transformer架构（T5‑small、PEGASUS‑xsum、LED‑base）+ LoRA微调 + 自动评估指标组合（ROUGE‑L、BLEU、BERTScore）+ 最高分选取策略；

**📊 数据集**

CNN/DailyMail 新闻摘要数据集（训练/验证/测试分割）；

**📈 对比分析**

与单模型基线以及多种近期大型模型（GPT3‑D2、Falcon‑7B、MPT‑7B、T0、PEGASUS‑xsum、BRIO、text‑davinci‑003等）在 ROUGE‑L、BLEU、BERTScore 和平均分进行对比；MAF 在 BERTScore 88.63%、ROUGE‑L 32.75%、BLEU 16.0% 与平均 45.8% 上取得最高分，显著优于单模型和多数大模型；

**⚠️ 局限性**

评估仍依赖自动指标，缺乏人工或无参考评估；仅集成三模型，扩展到更多模型需要额外计算资源；对事实一致性等语义真实性未做专门验证。

---

## 2. Formal Concept Lattices are Good Semantic Scaffolds for Concept-Based Learning

**arXiv ID:** 2606.05471 | [PDF](https://arxiv.org/pdf/2606.05471v1)

**作者:** Deepika SN Vemuri `[一作]` (IIT Hyderabad), Vineeth N Balasubramanian `[通讯]` (IIT Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出利用Formal Concept Analysis（FCA）构建概念层次结构，并在深度网络中按层级监督概念学习，使模型从一般到特定逐步构建语义表示。

**💡 创新点**

创新点在于将FCA生成的概念格子作为可控的“语义支架”，通过层级对齐（class‑cluster density）实现跨层的有序概念学习，显著提升中间层的可解释性与干预可行性。

**🔧 技术方法**

使用FCA构造概念格子、类-属性二元关系、层级对齐算法、迭代类组细化、属性与类别多层监督损失以及基于ViT/ResNet的网络架构。

**📊 数据集**

实验数据集包括 ImageNet100、AwA2 与 CIFAR100；AwA2 采用专家标注属性，其他两者通过 LLM 生成类‑属性矩阵。

**📈 对比分析**

与十种主流 CBM 变体（Vanilla CBM、MLPCBMs、Posthoc CBM 等）在准确率、Cluster Impurity (CI) 与 Davies‑Bouldin Index (DBI) 等指标上比较；FoCA‑CBM 在 CI/DBI 上均优于基线，准确率保持竞争力，且在多级干预实验中提升效果显著。

**⚠️ 局限性**

局限性包括：需要完整的类‑属性标注或可靠的 LLM 生成；概念格子构造在大规模属性集合时计算复杂度升高；对网络架构的依赖（需有可插入的中间块）以及在高度稀疏或不完整的属性空间中可能失效。

---

## 3. Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech

**arXiv ID:** 2606.05367 | [PDF](https://arxiv.org/pdf/2606.05367v1)

**作者:** Daniel Oliveira de Brito `[一作]` (Universidade Estadual Paulista), Arnaldo Candido Junior `[通讯]` (Universidade Estadual Paulista)

**通讯引用:** 538 | [OpenAlex ID](https://openalex.org/A5083987838)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对基于语言模型的TTS系统（Qwen3-TTS），本研究通过逐步消除法定位情绪信息载体为x-vector，并提出一种无训练、跨说话人、跨语言的情绪控制方法——在x-vector空间上做多说话人均值差值（centroid arithmetic）来实现情绪强度可控的音频生成。

**💡 创新点**

①发现在LM‑TTS中情绪控制的核心不是模型权重也不是离散码元，而是可学习的speaker embedding（x‑vector）；②利用这一发现，将传统的token‑based TTS不兼容的 centroid arithmetic 转移到 x‑vector 空间，实现训练‑free、低成本的跨说话人/跨语言情绪迁移；③在单一模型上实现可调情绪强度，解决了身份‑情绪权衡问题。

**🔧 技术方法**

技术手段包括：Qwen3‑TTS‑12Hz‑1.7B LM‑TTS、ECAPA‑TDNN speaker encoder、LoRA 微调、连续与离散 codec embedding 的算子对比、x‑vector 的 centroid arithmetic、以及多种评估指标（Emotion‑Embedding Cosine Similarity、Speaker‑Embedding Cosine Similarity、WER、UTMOSv2）。

**📊 数据集**

使用了英语情感语料库 ESD（4位说话人）与巴西葡萄牙语情感语料库 emoUERJ（3位说话人）进行情绪向量提取、交叉说话人评估与跨语言验证，并通过 held‑out ESD 说话人进行独立测试。

**📈 对比分析**

与纯 ICL（α=0）baseline 进行对比；在 EN held‑out 语料上，EECS 平均提升约 +0.29，SECS_W 与 UTMOS 也显著提高；在 EN→PT‑BR 的跨语言验证中，EECS 平均提升约 +0.09，SECS_W 维持 ≳0.88，WER 接近 0；多说话人平均版相较于单说话人版在身份保持和自然度上更优。

**⚠️ 局限性**

局限性包括：实验仅在 Qwen3‑TTS 上验证，未对其他 LM‑TTS 体系进行测试；方法依赖 speaker encoder 能保留情绪信息，若 encoder 仅做身份辨识则效果不佳；低变异 fine‑tune 数据可能限制操作窗口；跨语言评估受 PT‑BR 评测工具限制；缺乏人类听感评估；目前仅针对单一情绪类别，未探讨情绪混合或多维情绪表达。

---

## 4. An interpretable and trustworthy AI framework for large-scale longitudinal structure-pain association studies using data from the Osteoarthritis Initiative (OAI)

**arXiv ID:** 2606.05357 | [PDF](https://arxiv.org/pdf/2606.05357v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 5. Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference

**arXiv ID:** 2606.05308 | [PDF](https://arxiv.org/pdf/2606.05308v1)

**作者:** Abhishek Divekar `[一作]` `[通讯]` (Amazon), Abhishek Divekar (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种将少量人工标注与大量LLM判别结合的半监督估计方法PPI++，用于无偏估计排名评估指标，并对层次化指标如Precision@K做稀疏重构。

**💡 创新点**

将Prediction-Powered Inference扩展至层次化指标，通过将输出空间从 2^|C| 压缩到 2^K，并引入统计偏差校正，使得即使LLM判断存在系统性偏差，估计仍无偏且显著降低方差。

**🔧 技术方法**

使用半监督估计PPI++、LLM判别（Claude 3 Sonnet/Haiku）、稀疏向量化计算、方差最小化 λ 调优以及条件独立假设下的联合分布求和等技术。

**📊 数据集**

在 ESCI 检索基准（60 k 查询）以及生产搜索系统的真实查询数据上进行验证。

**📈 对比分析**

与仅人工标注和仅LLM估计相比，PPI++ 将 Precision@4 的标准误从 4.45 降低到 3.50（相对下降 21%），偏差降低至 0.70，且与 Haiku 相比成本降低 12 倍；在生产 A/B 测试中正确预测系统排名并带来 +407 bps 的日销量提升。

**⚠️ 局限性**

限制包括：仅在 Precision@K 上验证，未测试其他层次指标；对文档间相关性的条件独立假设可能失效；需要与未标注集同分布的少量金标集，分布漂移会削弱偏差校正效果。

---

## 6. Can We Predict The Human Preference For Text-to-Image Content Prior To Generation And Is It Even Useful To Do So?

**arXiv ID:** 2606.05478 | [PDF](https://arxiv.org/pdf/2606.05478v1)

**作者:** Joong Ho Kim `[一作]` (Louisiana State University), Keith G. Mills `[通讯]` (Louisiana State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出通过预测扩散模型初始随机噪声与文本提示对应的人类偏好得分，从而在生成前对噪声进行最佳选取，以提升文本到图像的生成质量。

**💡 创新点**

创新点在于首次将人类偏好度量（HPM）与初始噪声关联，通过轻量级预测器实现对噪声的预估与优化，并验证其对多种HPM的通用性。

**🔧 技术方法**

使用的技术包括三类预测器（内部交叉注意力CA‑I、外部交叉注意力CA‑E、编码-拼接EnCat），最佳- N（BoN）排序，以及Lambda损失+MAE的训练方案。

**📊 数据集**

实验数据集包括Pick‑a‑Pick验证集（1k提示）、HPSv2基准（3.2k提示）以及多种文本提示；使用的模型为SDXL、DreamShaper、Hunyuan‑DiT和PixArt‑Σ。

**📈 对比分析**

方法通过与不进行噪声优化的标准扩散生成进行对比，结果表明CA‑E和EnCat在大多数模型和HPM上均可提升人类偏好评分；相对而言，CA‑I性能最差且计算成本最高。

**⚠️ 局限性**

局限性包括：ImageReward与其他HPM不一致，导致优化目标与评估不匹配；CA‑I预测器因需部分推理交叉注意力导致显著计算开销；以及预测模型对不同模型和指标的迁移性仍有待进一步提升。

---

## 7. MCBench: A Multicontext Safety Assessment Benchmark for Omni Large Language Models

**arXiv ID:** 2606.05177 | [PDF](https://arxiv.org/pdf/2606.05177v1)

**作者:** Manh Luong `[一作]` (Monash University), Dinh Phung `[通讯]` (Monash University)

**通讯引用:** 11651 | [OpenAlex ID](https://openalex.org/A5036447132)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MCBench，一套面向Omni大语言模型的多模态安全基准，要求模型综合视觉、音频与语音三种模态进行安全评估；

**💡 创新点**

创新点在于：①首次构造同时涉及多模态输入的安全评估基准；②设计了安全-安全配对的场景，并为每个场景提供推理谓词；③通过大语言模型和多模态生成模型自动化生成场景与多模态内容；

**🔧 技术方法**

采用大型语言模型（Claude‑Sonnet‑4.5、Qwen‑Omni系列、Gemini‑Flash‑2.5 等）进行场景生成与评估，使用 Gemini‑Flash‑2.5 与 Stable Audio 1.0 合成图像与音频，利用 GPT‑4o 作为判定者完成安全标签评估；

**📊 数据集**

构建了包含1196个多模态安全场景的 MCBench 数据集，覆盖四大安全类别（物理伤害、社会伤害、非法伤害、财产损失），并提供对应的安全/不安全标注与推理谓词；

**📈 对比分析**

在该基准上评估了多款开源与闭源 Omni LLM（Qwen‑Omni 3B/7B、AnyGPT、InternOmni、Baichuan‑Omni‑1.5、OmniVinci、Gemini‑Flash‑2.5、GPT‑4o‑mini），结果显示最佳模型 Gemini‑Flash‑2.5 与 Qwen‑Omni‑2.5‑3B 的平均准确率约为 64.5%，在物理伤害与财产损失类表现较好，但在社会伤害与非法伤害类准确率低于 50%；同时模型在安全场景上存在显著的过度敏感（误报率高）；

**⚠️ 局限性**

主要局限包括：①跨模态推理整合不足，导致安全判断失真；②在安全场景中倾向于过度敏感；③对音频与视觉信息的感知能力有限，文本描述往往更易被模型利用；④数据生成受模型安全策略限制，部分敏感场景被排除；⑤评估方法依赖 LLM‑as‑judge，可能引入主观性。

---

## 8. Unpaired RGB-Thermal Gaussian-Splatting Using Visual Geometric Transformers

**arXiv ID:** 2606.05491 | [PDF](https://arxiv.org/pdf/2606.05491v1)

**作者:** Jean Cordonnier `[一作]` (Ecole Polytechnique Federale De Lausanne), Malcolm Mielle `[通讯]` (Schindler)

**通讯引用:** 145 | [OpenAlex ID](https://openalex.org/A5077521115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种从未配对的 RGB 与热像素图像集合中实现多模态新视角合成的方法，消除了对立体或标定配对数据的依赖。

**💡 创新点**

创新点在于：①利用 VGGT 对 RGB 与热图像分别独立估计相机位姿；②通过 XoFTR 与 Procrustes 算法实现跨模态位姿对齐；③在对齐后的统一坐标系下，采用 3D 高斯涂射（Gaussian Splatting）同时学习 RGB 与热的三维表示，从而得到一致的多模态场景。

**🔧 技术方法**

技术手段包括：VGGT（3D 变压器）进行相机参数与几何回归；XoFTR（跨模态特征匹配）与 MAGSAC 进行匹配与外点剔除；Trimmed ICP 结合 Procrustes Sim(3) 对齐；3D Gaussian Splatting 用于联合建模；SSIM/MAE 等损失与评价指标。

**📊 数据集**

实验使用了来自 ThermalGaussian 与 ThermoScenes 的九个多样化场景，构造未配对数据集并随机平衡 RGB 与热图像数量。

**📈 对比分析**

通过内模态（RGB 及热各自评估）和跨模态（用另一模态位姿评估）两组指标进行对比。结果显示：当使用热相机位姿时，模型在热图像合成上可与基线 ThermalGaussian 相媲美甚至更优；使用 RGB 位姿时，热图像合成性能下降；RGB 图像合成在两种位姿下差异不大。表明跨模态对齐至关重要。

**⚠️ 局限性**

局限性：①对齐精度仍受 XoFTR 匹配质量与 VGGT 位姿估计误差影响，特别是低纹理或极端热场景；②无法保证不同模态的尺度一致，需后续人工或 ICP 调整；③评估仍需已配对样本来计算跨模态指标；④方法对极低对比度热图像的鲁棒性不足，未来需改进跨模态特征提取与对齐策略。

---

## 9. Ahoy: LLMs Enacting Multiagent Interaction Protocols

**arXiv ID:** 2606.05390 | [PDF](https://arxiv.org/pdf/2606.05390v1)

**作者:** Omkar Joshi `[一作]` (North Carolina State University), Amit K. Chopra `[通讯]` (Lancaster University)

**通讯引用:** 3420 | [OpenAlex ID](https://openalex.org/A5056087323)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需编程的LLM代理，能根据用户目标动态选择并执行声明式交互协议（BSPL）并支持多协议并发执行

**💡 创新点**

通过将协议约束与LLM推理解耦，构建了可即时读取BSPL协议并在不同角色下执行的通用代理框架

**🔧 技术方法**

使用大型语言模型（Claude Haiku 4.5）作为决策核心，配合Kiko适配器实现协议状态管理与消息发送；利用Prompt Builder动态生成系统与用户提示；实现工具调用与事件处理

**📊 数据集**

在论文实验中使用自定义的BSPL协议（Purchase、Logistics、FlexiblePurchase等）和人工生成的外部事件数据；未采用公开大型数据集

**📈 对比分析**

与基线的比较主要体现在四项指标：编程自由度、并发协议执行、智能路径选择、外部事件处理；实验表明无消息违规、无约束异常、所有协议均成功完成；LLM调用次数与消息数与预期一致

**⚠️ 局限性**

限制包括：仅在有限实验场景验证；仅使用Anthropic模型，未对不同规模或供应商模型进行评估；缺乏对LLM决策质量的量化基准；未实现自动角色选择与历史学习机制

---

## 10. Differentiable Efficient Operator Search

**arXiv ID:** 2606.05232 | [PDF](https://arxiv.org/pdf/2606.05232v1)

**作者:** Xiaohuan Pei `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 22254 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Efficient Operator Search (EOS) 框架，统一并自动搜索多模态模型中的 token 减少操作。

**💡 创新点**

创新地将 pruning、merging、pooling 与 adaptive reweighting 统一为可微的共享操作空间，并通过梯度搜索同时优化层级激活、token 预算与操作参数，得到混合最优操作。

**🔧 技术方法**

采用可微参数化的 token 选择与分配门控、软最大温度控制、隐藏状态对齐约束，并在冻结的 LLaVA 基础模型上进行梯度下降搜索。

**📊 数据集**

使用多模态基准数据集：POPE、SQA、MME、GQA、TextVQA、SEED、MMStar、RealWorldQA、AI2D、OCRBench、ChartQA、MMBench-en。

**📈 对比分析**

在相同 token 预算下与 SparseVLM‑v1/v2、ToMe、Pool 等传统角落操作对比，EOS 在 r=64、16 等低预算场景下保持或提升准确率，整体平均分提升约 2% 以上。

**⚠️ 局限性**

仍需在冻结的基础模型上搜索，搜索过程计算成本高；仅针对视觉 token 进行搜索，未探索文本 token 或跨模态交互；在极端低预算下性能下降仍需进一步提升。

---

## 11. Recovering Physically Plausible Human-Object Interactions from Monocular Videos

**arXiv ID:** 2606.05359 | [PDF](https://arxiv.org/pdf/2606.05359v1)

**作者:** Dingbang Huang `[一作]` (University of Texas at Austin), Georgios Pavlakos `[通讯]` (University of Texas at Austin)

**通讯引用:** 4180 | [OpenAlex ID](https://openalex.org/A5052269532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过强化学习在物理仿真器中对单目视频得到的噪声人体-物体交互轨迹进行物理一致性重建。

**💡 创新点**

提出自适应采样与双向传播及运动更新机制，使RL能够在噪声极大情况下逐步修正整条交互序列，显著提升物理可行性。

**🔧 技术方法**

结合VisTracker的kinematic估计、SMPL-H/6DoF物体模型、物理仿真器、强化学习策略、以及自适应采样与双向传播技术。

**📊 数据集**

使用BEHAVE和InterCap两大公开HOI数据集进行实验。

**📈 对比分析**

与VisTracker以及预训练的InterMimic进行对比，物理一致性指标（如接触率、渗透深度、对象漂浮）大幅提升，成功率从约20%提升至>50%，3D误差仅略有增加。

**⚠️ 局限性**

依赖两阶段流程，受初始4D重建质量限制；目前仅处理单物体单人、相对较静态的接触，无法覆盖多物体、多人的复杂交互场景。

---

## 12. Assessing the Carbon Emissions and Energy Consumption of U.S. Hyperscale Data Centers

**arXiv ID:** 2606.05420 | [PDF](https://arxiv.org/pdf/2606.05420v1)

**作者:** Gianluca Guidi `[一作]` (Harvard T.H. Chan School of Public Health), Falco J. Bargagli-Stoffi `[通讯]` (Harvard T.H. Chan School of Public Health)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并验证了涵盖403个美国超大规模数据中心的设施级数据，估算其电力消耗、来源及对应的CO2排放。

**💡 创新点**

提出基于EPA eGRID 2023的归因式电网排放模型，并通过卫星影像验证设施数据，实现了对HDC能源足迹的系统量化。

**🔧 技术方法**

采用机器学习梯度提升回归树填补缺失功率容量，利用归因分配方法、PUE系数场景以及可视化Web平台展示结果。

**📊 数据集**

使用EPA eGRID 2023 plant‑level发电与排放数据、美国电力市场的平衡权区域信息，以及从私有数据中心提供商与OpenStreetMap获取的设施容量与面积数据。

**📈 对比分析**

通过四种物理合理的设施负荷场景对消耗与排放进行敏感性分析，并与2018年Siddik估计值以及IAE报告的全行业基准进行对比，显示HDC排放为10.5Mt的3.5–5倍，碳强度约48%高于全国平均水平。

**⚠️ 局限性**

局限在于设施覆盖不完整、缺乏工作负载与PUE细节导致负荷系数不确定、未考虑分布式可再生能源与时变负荷的碳强度变化，以及使用年度平均值而非时变排放。

---

## 13. An ERP Study on Recursive Locative Processing in Mandarin-Speaking Children with Autism

**arXiv ID:** 2606.05620 | [PDF](https://arxiv.org/pdf/2606.05620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 14. STMutants: A Mutation Testing Dataset for Structured Text Programs in Industrial Automation

**arXiv ID:** 2606.05499 | [PDF](https://arxiv.org/pdf/2606.05499v1)

**作者:** Md Humaun Kabir `[一作]` (Lamar University), Helen H. Lou `[通讯]` (Lamar University)

**通讯引用:** 2462 | [OpenAlex ID](https://openalex.org/A5085231365)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文创建了STMutants数据集，包含110个基于IEC 61131-3 Structured Text（ST）程序的一阶变异体，并在此基础上评估LLM生成的单元测试套件对变异体的检测效果。

**💡 创新点**

创新点在于：①首次公开提供面向工业PLC的ST变异基准；②提出了七类适用于ST的变异操作符；③用四阶段流程（定位、变异、编译验证、等价筛选）保证变异体可执行且非等价；④通过LLM基准实验展示AI辅助测试的可行性。

**🔧 技术方法**

使用技术包括：程序解析与变异点枚举、语法变异、PLC编译器验证、人工等价筛查（Cohen κ = 0.87）、LLM（GPT‑5.2、Gemini 2.5、Claude Sonnet 4.5）的一步提示测试生成与变异检测。

**📊 数据集**

数据集来源：11个来自OSCAT基础库及工业相关的ST程序，生成110个变异体，筛选后保留108个可观察的非等价变异体。

**📈 对比分析**

比较方法是将三种LLM分别生成测试套件后，用同一套测试执行原程序与变异程序，统计被杀死的变异比例；结果显示Gemini 2.5达94.4%检测率，GPT‑5.2与Claude Sonnet各86.1%；按变异符号分组，VRO、CRO等不同类别的检测率也被详细评估。

**⚠️ 局限性**

局限性包括：只覆盖11个中等规模程序，未考虑大型工业PLC代码；变异仅为一阶，未探究高阶变异；IO类变异样本极少，统计意义有限；LLM评估仅使用单步提示，可能低估模型潜力；在具有高度时序依赖的程序（如Sequence_8）表现显著下降。

---

## 15. Mutation Without Variation: Convergence Dynamics in LLM-Driven Program Evolution

**arXiv ID:** 2606.05408 | [PDF](https://arxiv.org/pdf/2606.05408v1)

**作者:** Can Gurkan `[一作]` (Northwestern University), Uri Wilensky `[通讯]` (Northwestern University)

**通讯引用:** 11956 | [OpenAlex ID](https://openalex.org/A5050932651)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文研究大模型（LLM）在无选择压力的情况下，作为程序突变算子时的动态行为，探讨其在程序空间中的收敛倾向。

**💡 创新点**

创新点在于将LLM突变视为无序、语义感知的重写过程，系统评估其在不同提示、模型、随机性条件下的收敛动力学，并与传统子树突变做对比，揭示LLM突变天然倾向于结构同质化的吸引子现象。

**🔧 技术方法**

技术手段包括：在受限的Lisp‑style DSL 上构建程序链；使用多种LLM（Gemini、Claude、GPT‑5）在多版本提示下生成单一变异；采用验证与重试机制确保语法合法；构建程序与骨架（Skeleton）层次的转移图；统计唯一程序数、骨架数、循环长度、度数熵等动态指标；对比经典子树突变的随机游走。

**📊 数据集**

数据集为在该DSL中随机生成的程序集合（约3种不同规模），以及使用实验中所需的约600条突变链（150条prompt实验×4起始程序），每条链长度为300步；此外收集了多模型多提示的40条链作为模型敏感性评估。

**📈 对比分析**

对比方法为无选择压力下的LLM突变链与经典子树突变链的唯一程序/骨架累计计数、平均循环长度和度数熵；实验显示LLM链在大多数情形下仅访问不到20种骨架，甚至在最开放的模型上也少于100种程序，而子树突变链平均可达270种程序、143种骨架，表明LLM突变的多样性显著低于传统突变。

**⚠️ 局限性**

局限性包括：仅使用单一受限DSL，未检验在更复杂语言上的可迁移性；分析仅基于基因型，未考虑程序行为多样性；模型敏感性实验仅用四个提示且无重复，样本量有限；未探究选择压力对收敛的影响；重试机制可能对变异分布产生隐含偏差。

---

## 16. Learning Manifold and Itô Dynamics with Branched Neural Rough Differential Equations

**arXiv ID:** 2606.05272 | [PDF](https://arxiv.org/pdf/2606.05272v1)

**作者:** Luke Thompson `[一作]` (University of Sydney), Andi Han `[通讯]` (University of Sydney)

**通讯引用:** 93 | [OpenAlex ID](https://openalex.org/A5031625303)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 B‑NRDE 的分支神经粗路径微分方程模型，结合分支签名核，能够在保持流形约束的同时学习 Itô 和流形动力学。

**💡 创新点**

创新点在于：①将 NRDE 的 log‑ODE 步骤迁移到分支 Hopf 代数（Grossman–Larson、Munthe–Kaas–Wright 根树），实现对 Itô 二次变差和有序协变导数的显式表示；②提出分支签名核目标，使训练过程中直接感知二次变差；③在统一框架下同时处理欧氏 Itô、流形 Itô 与 Stratonovich 经典场景。

**🔧 技术方法**

使用技术包括：Hopf 代数与伪 bialgebra 映射、log‑signature 与 log‑ODE 方法、CF‑EES(2,5) 级联指数数值求解、自动微分计算协变导数、分支签名核与欧氏签名核对比。

**📊 数据集**

实验数据集涵盖：粗波格里米 (rough Bergomi) 波动模型、SO(3) 旋转轨迹（仿真→真实对比）、SPD 协方差动态（金融/医学/扩散模型）以及标准时间序列基线数据。

**📈 对比分析**

与 NCDE、NRDE、GRU、xLSTM、SG‑NCDE 等基线比较，B‑NRDE 在粗波格里米的 KS 分数、SO(3) 预测的旋转几何误差以及 SPD 协方差的 1‑Wasserstein 距离上均表现更好或相当，尤其在 Itô 与流形约束明显场景中显著提升。

**⚠️ 局限性**

局限性：分支签名核需截断，导致高维下记忆与计算成本增加；缺乏对分支对数签名的有效投影方法；目前对高阶迭代积分信息的利用有限。

---

## 17. Aggregating LLM-Based Weak Verifiers for Spatial Layout Generation

**arXiv ID:** 2606.05268 | [PDF](https://arxiv.org/pdf/2606.05268v1)

**作者:** Sharon Zhang `[一作]` (Stanford University), Maneesh Agrawala `[通讯]` (Stanford University)

**通讯引用:** 20688 | [OpenAlex ID](https://openalex.org/A5045835385)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LLM生成弱验证程序并通过弱学习聚合成强验证器的空间布局验证与生成流水线，自动评估并改进3D房间与2D海报布局。

**💡 创新点**

创新点在于：①使用专门设计的布局验证DSL让LLM能够写出结构化、可解释的弱验证函数；②利用弱学习（Weaver）在仅10个标注样本的情况下聚合弱验证器，显著提升验证准确性；③将聚合后的强验证器用于指导生成器迭代生成，提升人类评估通过率达89.6%。

**🔧 技术方法**

核心技术包括：LLM（GPT‑5.4）编程生成、布局验证DSL、弱学习聚合方法（Weaver、逻辑回归、majority、top‑1）以及自然语言详细反馈机制。

**📊 数据集**

使用自制的3D房间布局生成器（基于BlenderKit）和2D海报布局生成器（HTML/CSS）产生100个示例，并在部分任务使用公开的3D‑FRONT数据集，标注约10个正负样本作为dev集。

**📈 对比分析**

与传统的LLM黑盒判定（含Vision LLM）比较，聚合弱验证器的F1分数提升1.2–7倍；在15项布局任务中，Weaver聚合在18项中获得最佳F1，平均提高42.2%（Binary）至66.2%（Detailed）的生成质量；在迭代生成实验中，Detailed反馈平均仅需2.2轮即可达到89.6%的人类通过率。

**⚠️ 局限性**

局限性包括：需要为每个任务生成并标注数据集，若生成器采样成本高则成本上升；弱验证器的质量高度依赖dev集，10样本可能导致聚合不稳定；此外，当前流水线主要针对室内和海报布局，尚未验证对更复杂或多模态场景的适用性。

---

## 18. Alpha-RTL: Test-Time Training for RTL Hardware Optimization

**arXiv ID:** 2606.05253 | [PDF](https://arxiv.org/pdf/2606.05253v1)

**作者:** Peilong Zhou `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Ying Wang `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 TTT‑RTL 框架，在 RTL 设计任务的测试阶段通过强化学习动态更新 LLM 策略，并结合 EDA 流程反馈实现对物理指标（Area·Delay·Power）的优化。

**💡 创新点**

其创新点在于将 PUCT 引导的搜索与在线策略梯度更新相结合，形成闭环的 per‑design test‑time 训练；同时提出了稀疏奖励下的自适应 KL 预算控制和熵优势估计器。

**🔧 技术方法**

使用 Qwen3‑8B LLM 结合 domain SFT，Yosys+OpenSTA 进行合成与时序/功耗评估，采用三阶段奖励（语法、功能、物理），并利用 PUCT 状态池与 entropic advantage 进行策略更新。

**📊 数据集**

在 RTLLM v2.0 的 49 个开源 RTL 设计上进行评测，并在工业级 XuanTie C910 LZA 单元（Sky130）上进行案例验证。

**📈 对比分析**

与 EvolVE、VeriAgent、REvolution 等基准在相同 PDK/EDA 流程下进行对比，TTT‑RTL 在覆盖率 48/49 设计、几何平均 PPA‑product 0.349（比基准 0.739 低 53%）以及 C910 单元的 ADP 降幅 59.4% 超过现有最佳方案。

**⚠️ 局限性**

主要局限在于对单一 LLM 后端的依赖，奖励稀疏导致搜索效率仍受限，且对跨 PDK 的泛化仍需进一步验证。

---

## 19. When Should We Protect AI? A Precautionary Framework for Consciousness Uncertainty

**arXiv ID:** 2606.05528 | [PDF](https://arxiv.org/pdf/2606.05528v1)

**作者:** Anna Mikeda `[一作]` `[通讯]` (Glass Umbrella), Anna Mikeda (Glass Umbrella)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出了一个预防性框架，将AI系统在五个意识维度上的证据映射到分级的保护义务，并通过阈值+梯度机制和两种跨维聚合方法实现可操作的决策指导。

**💡 创新点**

创新点在于：①将意识评估细化为五个可分离的福利相关维度；②引入阈值与梯度混合的义务触发机制；③提出层级（Bach-Sørensen）和架构无关的两种跨维聚合方案；④将框架与实际系统案例（Replika、OpenClaw）相结合，展示其可操作性。

**🔧 技术方法**

使用的技术主要是意识科学的计算指标（基于Recurrent Processing、Global Workspace、Higher-Order、Predictive Processing等理论），阈值触发与梯度加权的决策模型，以及案例分析方法；框架本身并未涉及新的机器学习算法。

**📊 数据集**

主要使用的是两种系统的案例信息（Replika和OpenClaw）的内部设计与行为数据，而非公开机器学习数据集；框架设计强调对任何架构（神经、符号、神经符号）的通用适用性。

**📈 对比分析**

通过在案例研究中评估系统在五维空间的位置，比较不同阈值与聚合方案对保护义务的触发差异，展示了框架在现实系统中的可操作性和灵活性；并提出了“阈值接近监测”原则，用以在系统演化过程中持续评估是否需升级义务。

**⚠️ 局限性**

局限性包括：①阈值与梯度参数尚未经过系统验证；②高度依赖当前意识科学与计算功能主义的假设；③对架构透明度与专业评估的需求高；④两种聚合方法尚未正式形式化，难以保证跨系统的一致性；⑤框架聚焦单体系统，未覆盖群体、分布式或人机混合情境。

---

## 20. LANTERN: Layered Archival and Temporal Episodic Retrieval Network for Long-Context LLM Conversations

**arXiv ID:** 2606.05182 | [PDF](https://arxiv.org/pdf/2606.05182v1)

**作者:** Rahul Subramani `[一作]` `[通讯]` (Cisco Systems, Inc.), Rahul Subramani (Cisco Systems, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Lantern，一种面向 LLM 对话的内存层，能够在对话被压缩后主动归档并在需要时恢复失去的细节；

**💡 创新点**

核心创新在于零 LLM 调用的主动提取式归档加上混合检索（语义、全文、关键字+重要性评分）与 RRF+MMR 组合，既高效又低延迟；

**🔧 技术方法**

技术实现包括 SQLite（WAL + FTS5）存储、MiniLM-L6-v2 embedding、RRF 融合、MMR 多样性、可选单次 LLM 重新排序与置信度衰减；

**📊 数据集**

使用公开 ShareGPT 真实多轮对话数据（94 轮对话、1894 条真值事实）进行评测；

**📈 对比分析**

与摘要、神经 RAG、MemGPT‑Faithful 等基线对比，Lantern‑Rerank 78.3% 事实恢复率，显著优于 MemGPT‑Faithful 72.4%（p<0.0001），并在四种生产 LLM 上平均提升 8.4% 回答准确率；

**⚠️ 局限性**

局限性包括：依赖 LLM 判定与提取的事实验证、单一数据集、单会话评估、嵌入模型单一、以及基线实现范围有限。

---

## 21. nnAudio 2: Overcoming Dynamic Compilation Barriers and Transform Inconsistencies

**arXiv ID:** 2606.05394 | [PDF](https://arxiv.org/pdf/2606.05394v1)

**作者:** Abhinaba Roy `[一作]` (Singapore University of Technology and Design), Dorien Herremans `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1827 | [OpenAlex ID](https://openalex.org/A5069548004)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

更新 nnAudio 2，修复 STFT/iSTFT 的 TorchScript 不兼容、iSTFT 在非均匀频率网格下的隐式失真、CFP 在新 SciPy 中的导入错误，以及 VQT 在 γ=0 时与 CQT 的差异，并新增可微分的 iCQT 逆 CQT 模块。

**💡 创新点**

系统化的维护与修复，提供显式错误提示、兼容现代 SciPy、实现 TorchScript 可编译、以及可微分的 Landweber 逆 CQT；并在保持原有 API 的前提下完成了对核心音频前端的现代化。

**🔧 技术方法**

PyTorch 与 TorchScript、NumPy、SciPy、CQT/VQT 变换、Landweber 迭代法、1D 卷积实现。

**📊 数据集**

未使用公开音乐或语音数据集，仅在仓库自带的单元测试和随机信号上进行验证。

**📈 对比分析**

在 Python 3.11 与 PyTorch 2.x 环境下运行原始与新代码的完整测试套件，所有单元测试通过率从 0% 提升至 100%，并在 TorchScript 下成功编译并执行前向/逆向 STFT。

**⚠️ 局限性**

仅对均匀频率网格下的 iSTFT 逆变换提供支持，非均匀网格下的逆变换仍不可用；训练可微分卷积核的 TorchScript 兼容性未得到完整覆盖。

---

## 22. Executable Schema Contracts: From Automatic Ingestion to Multi-Source Retrieval

**arXiv ID:** 2606.05415 | [PDF](https://arxiv.org/pdf/2606.05415v1)

**作者:** Padmaja Jonnalagedda `[一作]` (Intuit Ai Research), Kamalika Das `[通讯]` (Intuit Ai Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自动生成可执行schema并将其作为统一契约用于知识图构建与检索路由的系统，实现跨源零样本问答。

**💡 创新点**

创新点包括闭合世界字段目录结合统计结构推理的LLM schema induction、schema驱动的KG构建与检索路由，以及可在查询时增量扩展schema的机制。

**🔧 技术方法**

采用GPT-4.1等LLM进行语义发现，统计分析推断主键/外键，Neo4j存储知识图，向量检索与SchemaLookup混合路由，并引入两阶段可靠性门控。

**📊 数据集**

在BlendQA、HybridQA、TAT-QA和ComplexTR四个公开QA基准上进行评估。

**📈 对比分析**

通过与RAG、ProbTree、CoK、AtomR等基线进行对比，控制实验下在四个基准上取得最高EM/F1，提升幅度从+10到+55 EM不等。

**⚠️ 局限性**

主要局限包括LLM推理成本、查询时扩展导致的时延、schema实现覆盖不足以及仅在公开基准上验证，尚未在大规模企业数据上测试。

---

## 23. Efficient Computation of Distance Functions for Navigation Vector Fields in Lie Groups

**arXiv ID:** 2606.05372 | [PDF](https://arxiv.org/pdf/2606.05372v1)

**作者:** Vinicius M. Gonçalves `[一作]`, Luciano C. A. Pimenta `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于G-多项式曲线的向量场导航距离计算方法，能够在Lie群（尤其是SE(3)）上高效近似求解机器人配置与目标曲线之间的距离。

**💡 创新点**

创新点在于将距离最小化问题转化为对低阶多项式根的求解，并利用线性逼近与Lie群结构（如左Jacobi矩阵）得到解析或半解析解，显著降低计算成本。

**🔧 技术方法**

使用了Lie群与李代数理论、G-多项式曲线插值、Cardano三次方程求根、根求解优化、局部梯度下降及实验验证技术。

**📊 数据集**

实验使用了150条合成基底曲线（harmonic、lemniscate、square、trefoil、experiment），在K=13到41段的G-多项式分段内采样400个姿态，共840,000对；并在Kinova Gen3 7-DOF机械臂上进行实时轨迹跟踪实验。

**📈 对比分析**

与Piyavskii–Shubert全局最优算法对比，平均加速约5×（K≤80），误差率仅0.605%小于1%；在实验中每次查询平均32 µs；当姿态靠近曲线时Piyavskii表现更好，说明两者可互补。

**⚠️ 局限性**

局限性包括：在曲线近似误差较大的情况下（如多重等距点或接近奇异点）精度下降；对极细分段（K>80）性能下降；需要保证生成的G-多项式曲线满足可逆性与连续性；非闭合或不连续曲线需要额外处理。

---

## 24. Is This Edit Correct? A Multi-Dimensional Benchmark for Reasoning-Aware Image Editing

**arXiv ID:** 2606.05172 | [PDF](https://arxiv.org/pdf/2606.05172v1)

**作者:** Yixuan Ding `[一作]` (Zhejiang University), Yi Yang `[通讯]` (Zhejiang University)

**通讯引用:** 82431 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RE-Edit 评测基准，评估图像编辑系统在隐式逻辑推理方面的能力，并设计了 EditRefine 这一轻量级后处理框架以提升推理正确性。

**💡 创新点**

创新点包括：①将人类编辑工作流程拆解为五个推理维度（物理、环境、文化、因果、指代）并构建对应的 1,000 组评测样本；②引入维度对齐的评估准则，实现细粒度推理评价；③提出 EditRefine——一个模型无关的基于 MLLM 的推理引导后处理方案。

**🔧 技术方法**

使用的技术：扩散式图像编辑、基于 MLLM 的多步推理（CoT）与强化学习（SFT+RL）进行推理代理训练；主要模型包括 Qwen2.5-VL-7B（推理代理）、Qwen-Image-Edit、FLUX 系列、Nano Banana 等；评估工具为 VLM（Qwen3-VL-30B、GPT‑4.1）。

**📊 数据集**

数据集：RE-Edit 基准，由 1,000 条人工审核的编辑指令、对应的高质量图像（使用 Qwen-Image-Edit 生成）和推理说明构成，覆盖五个维度。原始图像均为合成，亦在实验中验证了对真实图像的迁移性。

**📈 对比分析**

对比方法：将 12 种顶尖图像编辑器（10 开源 + 2 商业）在 RE-Edit 上按五个维度以及 IF、SC 两个通用指标进行评测。实验显示：尽管大多数模型在视觉质量和目标定位上表现优异，但在环境、因果、文化等推理维度的得分普遍偏低；加入 EditRefine 后，模型在各推理维度的通过率提升 3–5 分，同时 IF、SC 分数基本保持不变。

**⚠️ 局限性**

局限性：①RE-Edit 仍以合成图像为主，可能未覆盖全部真实场景；②评估依赖 VLM 的自动判定，存在主观误差；③EditRefine 仅为后处理补救，未从根本上改进编辑模型的推理能力；④迭代多轮修正会导致误差累积，效果不如一次性修正。

---

## 25. The Cascade Log: Reference-Stable Windowing over Tiered Append Sequences

**arXiv ID:** 2606.05467 | [PDF](https://arxiv.org/pdf/2606.05467v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Levent Sarioglu `[通讯]` (Bahcesehir University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种层级化的追加式记录结构，通过持久化的区间映射与摘要合并，保证在任何时间点都能准确解析记录句柄，同时工作集保持有限。

**💡 创新点**

核心创新在于：①将跨层引用的唯一权威放在一个可持久化区间树上，消除跨层异常；②采用一次性区间合并（fold）将连续压缩块凝聚为单个节点，实现每次追加恒定的索引成本；③以碎片化数 A 为衡量，给出空间、查询和更新的紧凑上界，并证明其实例最优。

**🔧 技术方法**

主要技术包括持久化搜索树（路径复制 treap 或 a,b‑tree）、区间映射（interval map）与摘要节点（digest）、基于持久化根的快照、以及预算化窗口的子模最优算法。

**📊 数据集**

实验使用合成工作负载（追加占比 70%+、参考分布为 recency‑biased + uniform），规模从 2.8×10⁴ 到 10⁶ 条记录，随机与攻击性编辑序列，并与 hash‑spine、copy‑on‑write B‑tree 等基线对比。

**📈 对比分析**

与基线相比，该结构在 1e6 条记录下保持约 0.7–4.1 微秒的解析延迟，索引节点数随碎片化 A 线性增长但始终远低于哈希树；异常率为零且历史可寻址率保持 100%；支持快照读取和有序区间查询，性能仅比哈希慢一个常数倍。

**⚠️ 局限性**

局限性包括：单写者单线程实现，未实现多写者并发；未给出完整外存化实现；对时间旅行查询的历史保留取决于显式快照；摘要压缩仅保留版本信息，无法对记录内容做无损压缩；并且需要预先设定热层容量和折叠块大小。

---

## 26. A New Quaternion-Joint Cable-Driven Redundant Manipulator Configuration and its Control Through FABRIK and Residual Reinforcement Learning

**arXiv ID:** 2606.05236 | [PDF](https://arxiv.org/pdf/2606.05236v1)

**作者:** Tanapath Pornthisan `[一作]` (Chulalongkorn University), Viboon Sangveraphunsiri `[通讯]` (Chulalongkorn University)

**通讯引用:** 315 | [OpenAlex ID](https://openalex.org/A5076921205)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种由4段8关节组成的四元数关节软体机械手，并对其进行逆运动学求解与控制实验。

**💡 创新点**

1) 采用更少的关节（每段仅2个）但每个关节弯曲角度更大，从而保持工作空间；2) 将传统的FABRIK逆运动学与基于前向模型的深度逆解算器结合，随后引入残差强化学习补偿模型误差，显著提升跟踪精度。

**🔧 技术方法**

使用四元数关节的DH参数模型、FABRIK算法、基于前向模型的深度学习逆解（FC网络+IK梯度搜索）、残差强化学习（PPO+actor‑critic）以及MuJoCo/MJX仿真环境。

**📊 数据集**

所有数据均为自生成的仿真数据：通过对8根缆长度进行采样、在MuJoCo中求稳态，得到9维末端姿态，随后用于训练前向模型与残差策略；未使用公开工业数据集。

**📈 对比分析**

与传统FABRIK基线进行对比：成功率 70.4%（对比77.6%）但平均计算时间从0.3705s降至0.0221s；残差强化学习在50M环境步数内将位置误差从265mm降低到162mm、角误差从17.1°降至11.6°，但仍未达到单纯基准控制（3.7mm/0.5°）。

**⚠️ 局限性**

1) 残差强化学习收敛慢、样本效率低；2) 四元数关节在大弯曲角度下的几何误差仍显著，影响精度；3) 实机测试仍受限于缆绳摩擦、弹性和测量噪声；4) 仅在仿真中验证，实机鲁棒性待进一步评估。

---

## 27. A formal framework for the economic security of DeFi compositions

**arXiv ID:** 2606.05418 | [PDF](https://arxiv.org/pdf/2606.05418v1)

**作者:** Massimo Bartoletti `[一作]` (University of Cagliari), Roberto Zunino `[通讯]` (University of Trento)

**通讯引用:** 1264 | [OpenAlex ID](https://openalex.org/A5003560156)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于信息流非干预的 DeFi 合成安全框架，并定义了局部最大可提取价值（local MEV）与 MEV 非干预（MEV non‑interference）两个核心安全概念；

**💡 创新点**

创新点在于：①把经典的非干预思想迁移到智能合约组合安全；②引入局部 MEV，聚焦受害合约的损失而非全链状态；③提供两种攻击者模型（有限与无限财富）并给出足够条件、局部化原则和前置交易抵抗性；

**🔧 技术方法**

使用形式化模型对以太坊类区块链、合约调用图、代币流、价格 oracle 进行抽象；构造定理证明和局部化推理；借助观察不变性（observational invariance）与 token‑flow 分析实现安全判定；

**📊 数据集**

未使用真实数据集，而是通过一系列简化的 Solidity 示例合约（AMM、Exchange、Option、LendingPool 等）作为案例验证框架的适用性；

**📈 对比分析**

通过与现有 ϵ‑可组合性理论对比，展示 MEV 非干预在理论上与其不等价；在示例合约上演示安全与不安全的分类，未进行数值实验，侧重形式化证明与案例演示；

**⚠️ 局限性**

局限性包括：①需要手工证明合约满足观测不变性与 token‑flow 条件；②对循环依赖和非 well‑formed 状态处理不完整；③仅适用于账户‑基区块链的抽象模型，无法直接应用于所有现有 DeFi 协议；④缺乏大规模实证评估。

---

## 28. Robust Scene Transfer for PointGoal Navigation via Privileged Sensor Guided Contrastive Learning

**arXiv ID:** 2606.05506 | [PDF](https://arxiv.org/pdf/2606.05506v1)

**作者:** Amirhossein Zhalehmehrabi `[一作]`, Alessandro Farinelli `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种利用训练阶段有权访问的激光雷达信息指导的对比学习框架，训练出适用于视觉PointGoal导航的鲁棒视觉表示，并在部署时仅使用单目RGB。

**💡 创新点**

创新点包括：①用几何感知的自适应温度对对比损失进行动态调节；②在预训练与策略学习阶段引入跨阶段域不匹配，迫使策略关注场景无关的几何特征；③将预训练表示冻结，彻底解耦感知与控制。

**🔧 技术方法**

技术主要包括：对比学习（InfoNCE）与几何相似度度量、基于LiDAR的温度自适应、两层MLP策略网络、Soft Actor-Critic强化学习、以及对比样本的场景不变正样本构造。

**📊 数据集**

使用了多模态数据集（包含RGB、LiDAR、深度与语义分割），从四个仓库和Photo-studio等高保真仿真环境收集的数十万帧，并在未见的室内外环境中进行测试。

**📈 对比分析**

与CLIP、MAE、DINOv2、SimCLR等通用视觉编码器以及基于重建的自编码器相比，本文方法在失真、纹理变化和室内-室外跨域迁移上显著提高成功率（如从30%上升至70%+）且在SPL和成功率上保持稳定。

**⚠️ 局限性**

局限性包括：①仅在仿真环境验证，真实世界鲁棒性未知；②仍需额外的LiDAR数据进行预训练；③跨阶段域不匹配可能导致信息损失，且对环境几何假设依赖较强。

---

## 29. LEVANTE-bench: Multi-Scale Comparison of VLMs to Children Using Cognitive Tasks (or, "Is Your VLM Smarter Than a 5th Grader?")

**arXiv ID:** 2606.05497 | [PDF](https://arxiv.org/pdf/2606.05497v1)

**作者:** Alvin Wei Ming Tan `[一作]` (Stanford University), Michael C. Frank `[通讯]` (Stanford University)

**通讯引用:** 16707 | [OpenAlex ID](https://openalex.org/A5103234894)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了LEAVE‑ANTE‑Bench，一个基于LEVANTE多语言儿童认知数据的多模态模型与人类认知对齐基准，评估多种VLM在数学、推理、语言、空间等任务中的表现；

**💡 创新点**

创新点在于：①采用多尺度（任务级、项目级、试验级）对齐评估框架；②利用已验证的心理测量任务在三种语言和多国儿童中进行大规模对比；③将开源和商业VLM在同一基准上系统评测；

**🔧 技术方法**

技术手段包括：项目响应分布与人类分布的KL散度、IRT/2PL模型估计儿童能力与项目难度、对齐相关性计算、Prompt敏感性研究、使用多模态推理的VLM（Gemma、InternVL、Qwen、SmolVLM、TinyLLaVA等）以及商业模型（GPT‑5.3、Gemini‑2.5 Pro、Gemini‑3 Flash）；

**📊 数据集**

数据集为LEVANTE 2026.1版本，涵盖英语、西班牙语、德语三语，5–12岁儿童共3,147名参与者，309,108次试验，来源于哥伦比亚、加拿大和德国的学校、实验室与家庭环境；

**📈 对比分析**

比较方法为跨任务相关性、项目难度相关性和试验级KL对齐；结果显示：较大模型在任务级对齐表现最优，但在项目级与试验级对齐仍显不足，矩阵推理和精神旋转等空间任务几乎全模型达不到人类水平；

**⚠️ 局限性**

局限性包括：仅覆盖六个任务和三种语言；项目数量有限导致对齐精度受限；模型对齐评估需要大量重复推理，计算成本高；可能存在训练数据泄漏；缺乏基于儿童数据训练的VLM，难以进一步分析学习轨迹。

---

## 30. Localizing Prompt Ambiguity in Large Language Models with Probe-Targeted Attribution

**arXiv ID:** 2606.05486 | [PDF](https://arxiv.org/pdf/2606.05486v1)

**作者:** Govind Ramesh `[一作]` (Georgia Institute of Technology), Wei Xu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 16874 | [OpenAlex ID](https://openalex.org/A5013867024)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于梯度归因的提示词歧义定位方法\u2022 通过训练线性探针区分清晰提示与歧义提示，并利用探针分数对中间隐藏层残差进行积分梯度归因，得到令牌级别的歧义分布

**💡 创新点**

创新点在于：① 将歧义视为输入的潜在属性，可在内部表示层被线性解码；② 设计了探针目标的残差空间积分梯度方法（Probe‑Targeted Residual Integrated Gradients）以提高歧义定位精度；③ 通过合成和人工金标的多域歧义数据集验证方法的跨域鲁棒性

**🔧 技术方法**

技术：线性探针（Logistic Regression）、残差层积分梯度（Residual IG）、一维高斯平滑、基准模型 GPT‑5.4 的句子定位评估

**📊 数据集**

数据集：LeetCode（编程）、MATH（数学）、PromptTensor（写作）共计约1200条提示；人工金标 12 条歧义提示；每个域均产生一条歧义版本（总 1200 条），并对原始提示进行标注

**📈 对比分析**

与梯度×输入、标准积分梯度以及 GPT‑5.4 的句子识别进行对比；在三域合成集上 AUROC 均超过 0.90，AUPRG 亦显著提升；在人工金标集上 AUROC 0.891、AUPRG 0.870；与 GPT‑5.4 的句子定位相比，F1、精确率更高，召回率相近

**⚠️ 局限性**

局限：假设歧义可线性解码；方法未验证对其他潜在属性、不同模型规模的迁移；探针主要在 GPT‑5.4 生成的合成歧义上训练，可能带来生成器偏差；标注仅基于改写句子，实际歧义可能更复杂；归因层选择对结果敏感

---

## 31. AURA: Intent-Directed Probing for Implicit-Need Surfacing in Situated LLM Agents

**arXiv ID:** 2606.05557 | [PDF](https://arxiv.org/pdf/2606.05557v1)

**作者:** Yang Li `[一作]` (Guangdong Institute of Intelligence Science and Technology), Mingkun Xu `[通讯]` (Guangdong Institute of Intelligence Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在场景感知与工具调用之间插入的意图推断框架 IntentFrame，能够估计用户隐式需求并给出缺口分数，驱动对私有状态的有针对性探测。

**💡 创新点**

创新点在于将缺口估计作为前置工具控制变量，将隐式意图推断与探测预算、工具选择解耦，实现自适应探测并提升隐式需求覆盖。

**🔧 技术方法**

技术主要包括基于 LLM 的 IntentInferrer 生成结构化 JSON（含缺口、推荐探测工具等）、基于缺口的阈值映射到探测步数、以及预探测循环 Explore 与后续推理 Reason 等模块。

**📊 数据集**

使用的数据集为 AURATown 模拟器（60×60 网格、5 名代理、20 位置），以及作者自建的 100 问四场景隐式意图基准、25 问原型基准和 50 问事实检索基准；标注采用 5 子类别划分，kappa = 0.61。

**📈 对比分析**

与 Vanilla、Static Context、ReAct、Plan‑and‑Solve 等基线比较，IntentFrame 在隐式需求覆盖上相较于 ReAct‑式 NoIntent 提升 0.07（p<10^-6），在事实检索中以 1.40 探测/问量（比 Fixed‑Probe 82% 降低）实现零禁止工具违规，整体准确率略低但在访问‑成本 Pareto 边缘表现优异。

**⚠️ 局限性**

局限性包括：仅适用于具工具介导的隐私状态的场景化查询；在事实检索等已公开状态场景下未提升准确率；缺口‑预算映射手工调节，需进一步学习；多轮对话、开放域计划等更广泛情境的适用性尚未验证。

---

## 32. Zero knowledge verification for frontier AI training is possible

**arXiv ID:** 2606.05433 | [PDF](https://arxiv.org/pdf/2606.05433v1)

**作者:** Pierre Peigné `[一作]` (General-Purpose AI Policy Lab), Paul Wang `[通讯]` (Sorbonne Université)

**通讯引用:** 83 | [OpenAlex ID](https://openalex.org/A5110868154)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种面向前沿 AI 预训练的零知识验证体系，结合预提交的训练规范、网络流量锚定和实时 Merkle 提交，使用 zkVM 与本地 BF16/FP32 预编译实现对 GPU 计算的完整可验证性；

**💡 创新点**

核心创新在于直接验证实际 GPU 的浮点运算，而非传统的有限域近似；同时通过三重信任锚（训练规范、网络观测、Merkle 跟踪）实现高效、可扩展的随机抽样验证；

**🔧 技术方法**

采用 zkVM（如 RiscZero）配合自定义浮点预编译、Poseidon/ SHA-256 哈希、Merkle 树、网络 TAP/SmartNIC 捕获、以及训练规范与随机采样协议；

**📊 数据集**

研究中未使用特定公开数据集，而是以承诺的训练数据集 Merkle 根为参考，兼容任何可承诺的数据；

**📈 对比分析**

与现有 ZK-ML 系统相比，所提方案在前沿规模下训练侧开销仅为 2–10%（单字节级别），证明体积约 200 KB，且单步验证成本极低；

**⚠️ 局限性**

局限性包括目前仅支持稠密预训练，未覆盖稀疏 MoE、强化学习后训练或多数据中心训练；需要自定义 Tensor‑Parallel all‑reduce、实现网络锚点的标准化等；

---

## 33. Brick-Composer: Using MLLMs for Assembly with Diverse Bricks

**arXiv ID:** 2606.05445 | [PDF](https://arxiv.org/pdf/2606.05445v1)

**作者:** Jiateng Liu `[一作]` (University of Illinois Urbana Champaign), Heng Ji `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究探索多模态大语言模型（MLLM）在 LEGO 风格砖块组装中的视觉定位与空间推理能力，提出了 BC-Bench 基准，并基于人类设计示例、世界反馈和合成经验三种监督信号，构建了 Brick-Composer 学习框架来提升砖块选择和姿态估计；

**💡 创新点**

创新点在于：①首次提供专门评估砖块选择与姿态推理的多模态基准 BC-Bench；②提出 Brick-Composer 通过人类设计 Sparks、仿真世界反馈以及可扩展的合成经验三重监督来显著提升 MLLM 的组装性能；

**🔧 技术方法**

技术包括：多模态大语言模型（Gemma、Qwen、GPT‑5 等）自动回归训练、基于仿真器的动作反馈循环、对称-aware 旋转误差评估、步骤级成功率与整体指标评估；

**📊 数据集**

数据集：BC‑Bench（约80个人工设计的 LEGO 对象，包含手册视图、候选砖块网格和多视角仿真状态）以及约 700 个合成的 20–100 块砖块合法配置，共计 40k+ 训练步骤；

**📈 对比分析**

通过对比直接提示、设计监督、世界反馈单独和综合 Brick‑Composer 四种方法，发现零射击性能几乎无成功率（<0.5%），而 Brick‑Composer 在最佳对象上可达 42% 步级成功率，砖块选择准确率提升至 70%，姿态误差显著降低；

**⚠️ 局限性**

局限性：仅在仿真环境中验证，缺乏真实机器人执行的适配；人类设计数据规模受版权限制；实验仅聚焦单步组装，未检验长期序列构建的鲁棒性。

---

## 34. ADK Arena: Evaluating Agent Development Kits via LLM-as-a-Developer

**arXiv ID:** 2606.05548 | [PDF](https://arxiv.org/pdf/2606.05548v1)

**作者:** Jintao Huang `[一作]` (Ohio State University), Yu Hu `[通讯]` (Microsoft)

**通讯引用:** 9602 | [OpenAlex ID](https://openalex.org/A5014478407)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出 LLM-as-a-Developer 方法，利用 LLM 自动学习并生成 ADK 框架中的 Agent 代码，实现对 51 个 Python ADK 的全自动化评估。

**💡 创新点**

创新点在于把人类开发者替换为 LLM 代理，通过验证-反馈循环量化 API 可用性，并以统一管道比较框架效果，首次在生态系统级别评估 ADK。

**🔧 技术方法**

采用 GPT‑5.4/Claude‑Opus 作为 LLM 开发者，配合 Docker 隔离、三层验证管道、LLM 代理转发、Benchmark 适配器以及 token‑级遥测等技术。

**📊 数据集**

使用了 SWE‑bench、τ²‑bench、MCP‑Atlas、Terminal‑Bench 四个公开基准（每个 50 题）共 204 个 Agent–Benchmark 组合。

**📈 对比分析**

方法在相同 LLM 后端（GPT‑5.4 Nano）下执行生成的 Agent，比较生成成本、任务成功率与前沿代码生成器；结果显示生成成本 0.6–3.4 美元/Agent，最佳 ADK 代理可达 80% 任务成功率，且成本远低于通用前沿编码器。

**⚠️ 局限性**

局限性包括仅使用单一后端 LLM（GPT‑5.4 Nano），训练数据偏向主流框架，验证仅保证语法正确而非任务完成，导致大多数生成 Agent 仍无法解决 benchmark 任务。

---

## 35. DiffSlack: Learning under Nonlinear Inequality Constraints via Learnable Slack Variables

**arXiv ID:** 2606.05247 | [PDF](https://arxiv.org/pdf/2606.05247v1)

**作者:** Ziqian Wang `[一作]` (Tsinghua University), Zhen Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 79275 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出 DiffSlack，一种利用可学习的松弛变量实现非线性不等式约束的可微投影层；

**💡 创新点**

创新点在于把不等式约束改写为等式+可学习松弛变量，提供数据驱动的热启动，减轻投影负担，并结合两阶段课程学习实现稳定训练；

**🔧 技术方法**

使用可微 Gauss-Newton 投影、滑动松弛变量、隐式微分梯度流、两阶段软硬约束训练以及基于 Log‑Sum‑Exp 的碰撞约束逼近；

**📊 数据集**

在车辆路径规划任务上，用 200 条非线性不等式（碰撞、曲率、间距）生成的 20 万个随机障碍场数据，使用全局人工势场 (G‑APF) 作为粗糙监督；

**📈 对比分析**

与经典规划器（Hybrid A*、RRT*、NMPC）以及学习式基线（IL、IL+Soft、DC3、ENFORCE）比较，DiffSlack 在成功率 93.8% 及约束满足率（碰撞 99.48%）上优于其它学习方法，推理时间与 ENFORCE 相近；

**⚠️ 局限性**

局限在于若原始预测处于投影收敛域之外（如不可行的拓扑类），投影无法完全修正，且对初始粗糙预测依赖较大，需要更强的预训练或更广泛的数据覆盖。

---

## 36. Online Safety Regulation Increases Privacy Risk: Evidence from the UK Online Safety Act

**arXiv ID:** 2606.05273 | [PDF](https://arxiv.org/pdf/2606.05273v1)

**作者:** Dhyey Mehta `[一作]` (University of Edinburgh), Tuğrulcan Elmas `[通讯]` (University of Edinburgh)

**通讯引用:** 149 | [OpenAlex ID](https://openalex.org/A5081836422)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用 Reddit 与 Google Trends 数据，结合贝叶斯结构时间序列、主题建模、情感分析以及 VPN 隐私政策风险评估，系统评估英国《在线安全法》里程碑对 VPN 讨论、搜索兴趣和用户隐私风险的影响。

**💡 创新点**

首次以阶梯式事件分析方法量化立法里程碑对 VPN 话语与搜索兴趣的影响；采用 LLM 辅助的文本筛选与主题解释；将 VPN 隐私政策风险指标与搜索关注度对齐，评估监管导致用户迁移至高风险 VPN 的可能性。

**🔧 技术方法**

贝叶斯结构时间序列 (CausalImpact)、LDA 主题建模、RoBERTa 与 Gemini 情感分析、Gemini/GPT‑5.5 文本分类与隐私标记抽取、Google Trends 时间序列分析。

**📊 数据集**

2021–2025 年 Reddit 帖子/评论（VPN 子版块 243,150 篇/2,275,677 条，UK 政治子版块 2,040,904 篇/69,361,713 条）；Google Trends 搜索兴趣（VPN 关键词及品牌）；69 家 VPN 服务的归档隐私政策。

**📈 对比分析**

通过 BSTS 对比预测与观测序列，获得相对影响率（+100%~+415%），效果显著；主题模型 C_V coherence 确定最佳主题数；情感模型 Precision ≈0.8；VPN 风险分类基于规则，风险分布保持稳定，未出现显著偏向高风险服务；整体搜索兴趣显著上升但未导致高风险 VPN 的比例增加。

**⚠️ 局限性**

Reddit 样本偏年轻、技术熟练，UK 居住过滤不精确；Google Trends 仅衡量兴趣非实际使用；隐私政策分析受限于公开文本，未反映真实行为；LLM 分类与情感分析存在误差；研究基于公开数据，缺少用户访谈与真实使用数据。

---

## 37. Insurance of Agentic AI

**arXiv ID:** 2606.05449 | [PDF](https://arxiv.org/pdf/2606.05449v1)

**作者:** Quanyan Zhu `[一作]` (New York University), Quanyan Zhu `[通讯]` (New York University)

**通讯引用:** 11423 | [OpenAlex ID](https://openalex.org/A5081500464)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过综述与分析，构建了agentic AI保险市场框架，并提出多层次风险评估与产品设计方法。

**💡 创新点**

创新点在于将agentic AI风险与传统保险（网络、技术E&O、产品责任等）整合为层级化、分配明确的保险生态，并提出基于权限、场景与控制的计费与再保险模型。

**🔧 技术方法**

采用的技术包括风险场景库、曝光清单、控制效能评估、场景模拟、累计风险管理，以及基于NIST、EIOPA等标准的计价框架。

**📊 数据集**

主要使用公开的事件记录、行业案例、法规文件以及少量已有的网络保险损失数据作为分析依据，并构建了自定义的风险情景集。

**📈 对比分析**

与传统网络保险的对比显示，agentic AI保险在暴露评估与场景定价上更为细粒度，虽然缺乏历史损失曲线，但通过情景与累积分析已能提供可操作的定价与再保险方案。

**⚠️ 局限性**

局限性包括缺乏成熟的agentic AI损失数据库、缺少统一的政策词汇和索赔分类、监管责任分配不清，以及模型更新导致的非平稳性导致定价与再保险挑战。

---

## 38. Availability-Aware and Efficiency-Driven AI Service Chain Provisioning in Multi-Domain Edge Intelligence Cloud

**arXiv ID:** 2606.05637 | [PDF](https://arxiv.org/pdf/2606.05637v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 39. Self-supervised User Profile Generation for Personalization

**arXiv ID:** 2606.05336 | [PDF](https://arxiv.org/pdf/2606.05336v1)

**作者:** Clark Mingxuan Ju `[一作]` (Snap Inc.), Neil Shah `[通讯]` (Snap Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无监督的LLM用户概述生成框架BUMP，通过双向的NDCG排名奖励（前向预测和后向识别）训练文本化用户简介，并在生成后直接作为提示前缀用于多任务个性化。

**💡 创新点**

创新点在于：①用自监督的双向排名奖励取代传统下游标签驱动的奖励；②结合硬负样本挖掘提升奖励稀疏性和对抗性；③实现了可解释、可迁移且无需用户特定训练的自然语言个人化表示。

**🔧 技术方法**

使用技术包括：GRPO强化学习、冻结LLM评判器、multi‑positive NDCG评分、BGE embedding用于硬负样本挖掘、位置去偏和长度惩罚等。

**📊 数据集**

实验数据集为LaMP基准（六个公开任务，跨二十多个任务类型）。

**📈 对比分析**

对比方法包括无个性化、原始历史、零射击摘要、闭源Gemini-3系列、下游奖励训练；在11个指标上BUMP+在大多数任务上超过闭源Gemini-3-Pro，并与之持平或更优，说明自监督方法可替代有标签奖励。

**⚠️ 局限性**

局限性包括：依赖LLM评判者可能带来的偏差与幻觉；当候选池过大时评判噪声增大；硬负样本使用BGE相似性作为代理，可能引入误噪，进一步的任务感知硬负策略是未来改进方向。

---

## 40. AsyncWebRL: Efficient Multi-Step RL for Visual Web Agents

**arXiv ID:** 2606.05597 | [PDF](https://arxiv.org/pdf/2606.05597v1)

**作者:** Hao Bai `[一作]` (University of Illinois Urbana Champaign), Tong Zhang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AsyncWebRL，构建完全异步的多步视觉‑语言 Web 代理训练框架，并通过永恒回放池和轻量级截图处理实现显著吞吐提升；

**💡 创新点**

创新点包括：①将回放池保持持续运行与截图轻量化相结合，实现 2.4–2.9× 的端到端训练速度；②将 per‑trajectory 归一化 1/|τ_i| 替换为常数 1/k，消除轨迹长度偏差并显著压缩轨迹与 token 长度；

**🔧 技术方法**

采用异步多步 RL、GRPO、Decoupled PPO 重要性采样、Qwen3‑VL‑8B 视觉‑语言模型以及 WebGym 评估框架；

**📊 数据集**

使用 WebGym 训练集（约 290k 任务，覆盖 128k 网站）和 OOD 测试集（1167 任务）进行实验；

**📈 对比分析**

与同步 REINFORCE 基线和 RAFT++ 进行对比；在 OOD 平均成功率上实现 45.4%（比 42.9% 提升 5.8%），在 Medium 和 Hard 难度分别提升 42% 与 48%；

**⚠️ 局限性**

局限性：对更大 horizon 的适配仍有限，超长轨迹下的表现尚未充分验证，且在非 WebGym 环境下的泛化性能未知。

---

## 41. What's Under the Skin? Estimating Swine Body Condition

**arXiv ID:** 2606.05611 | [PDF](https://arxiv.org/pdf/2606.05611v1)

**作者:** Mk Bashar `[一作]` (Michigan State University), Daniel Morris `[通讯]` (Michigan State University)

**通讯引用:** 4838 | [OpenAlex ID](https://openalex.org/A5031313498)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种基于天花板安装RGB‑D摄像机的端到端两阶段系统PigFormer，用来自动预测怀孕母猪的背部脂肪厚度、腰肌深度以及总组织厚度。

**💡 创新点**

创新点在于将几何前端与Slice Attention Encoder相结合，利用SAM3‑to‑MaskDINO分割蒸馏、地面平面去除、方向归一化得到标准高度图，再以跨切片注意力捕捉背部全局空间关系，并通过多目标回归实现高精度测量。

**🔧 技术方法**

技术上采用RGB‑D深度图到高度图的几何预处理、MaskDINO分割蒸馏、双重池化的Slice Attention Encoder、RoPE位置嵌入以及Huber损失的多目标回归。

**📊 数据集**

使用了来自两家美国养猪场（MSU和UNL）的319个母猪/育雏猪实例，包含6,705帧深度图，并用超声测量作为标注。

**📈 对比分析**

与单阶段ResNet‑18和ViT‑small原始深度输入的基线相比，PigFormer在整体MAE上分别提升了22%和39%，单个后者的平均误差为3.87 mm。

**⚠️ 局限性**

限制在于数据规模有限（319个实例）且仅来自两家设施，难以充分评估跨品种、跨管理体系的泛化能力；且缺乏公开的RGB‑D与超声对齐基准。

---

## 42. PEFT of SLM for Telecommunications Customer Support: A Comparative Study of LoRA Configurations with Energy Consumption Analysis

**arXiv ID:** 2606.05176 | [PDF](https://arxiv.org/pdf/2606.05176v1)

**作者:** Lucas Tamic `[一作]` (Orange), Xavier Marjou `[通讯]` (Orange)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5050612763)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Qwen2.5-3B 进行低秩适配 LoRA 微调，利用 30,000 条合成的电信客服会话数据，系统评估 16 种 LoRA 配置的量化和定性性能，并记录能耗。

**💡 创新点**

创新点在于（1）提出基于 52 条技术术语的组合式合成数据生成方法；（2）深入比较 LoRA 目标模块覆盖度与秩大小对验证损失与人类友好度的影响；（3）首次将能耗与质量指标一起绘制 Pareto 前沿，揭示验证损失与定性评估的显著分离。

**🔧 技术方法**

采用 Qwen2.5-3B 作为基模型，使用 LoRA（不同秩 16/32、模块覆盖 2/4/7 个）、AdamW、bfloat16 混合精度训练，评估指标包括交叉熵损失、困惑度、GPU 能耗、GPT‑5.2 与 Claude‑4.5 Sonnet 的 LLM‑as‑a‑judge 定性排名。

**📊 数据集**

合成数据集基于 52 条行业术语、10 种故障原因和 3 种使用场景，共 1,560 个问题场景，利用 Gemini 2.0 Flash 生成 30,000 条客服问答对。

**📈 对比分析**

通过比较 16 种 LoRA 配置的验证损失、困惑度、定性排名以及能耗，发现验证损失最低的配置并非定性最优，最佳定性配置的能耗在 284–1371 Wh 之间，能耗与质量的折衷呈现 5 倍差异。

**⚠️ 局限性**

局限性包括：合成数据缺乏真实多轮对话和用户错误；验证集与定性评估集分布不完全一致，导致量化指标与人类友好度分离；未进行真实人类专家评估；对更复杂、更长的数据集可能无法充分利用更高 LoRA 秩和更大模块覆盖度。

---

## 43. GITCO: Gated Inference-Time Context Optimization in TSFMs

**arXiv ID:** 2606.05332 | [PDF](https://arxiv.org/pdf/2606.05332v1)

**作者:** Manya Pandey `[一作]` (Birla AI Labs), Saurabh Deshpande `[通讯]` (Birla AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在推理时通过优化输入上下文来提升冻结时序基础模型（TSFM）预测精度的方法，避免上下文污染。

**💡 创新点**

创新点在于引入三阶段轻量化管线 GITCO（Gate‑Router‑Critic）以及上下文敏感性曲线，首次在不更新模型参数的前提下实现高质量的推理时输入优化。

**🔧 技术方法**

利用门控二分类器决定是否干预，路由器根据元特征选取三种专家 Critic，Critic 对每个补丁给出干扰概率并通过 5 步简单移动平均进行局部平滑；同时使用频域、波动率等低成本元特征进行模型无关判断。

**📊 数据集**

在 53 个 GIFT‑Eval 时序数据集上评估，并针对两种冻结 TSFM（TimesFM 2.5 与 Chronos2）进行实验。

**📈 对比分析**

与零样本基线比较，TimesFM 2.5 上 GITCO 在 53 个数据集平均提升 1.95% MASE（最大 4.30%），捕获 89.9% 的理论可实现改进；在 Chronos2 上尽管能发现可改善的补丁，但门控学习失败，表明方法对不同架构的可迁移性有限。

**⚠️ 局限性**

局限性包括：仅在两种模型和 53 个数据集上验证，门控机制对 Chronos2 无效；改进空间仅限于三种 Critic 与 SMA 平滑，可能不适用于更复杂或分布漂移的数据；未探究更丰富的干预算子与更大规模模型的适用性。

---

## 44. Generic Triple-Latent Compression with Gated Associative Retrieval

**arXiv ID:** 2606.05175 | [PDF](https://arxiv.org/pdf/2606.05175v1)

**作者:** Liu Xiao `[一作]` `[通讯]`, Liu Xiao

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一类通用的三元隐状态序列模型，结合运行的 token 状态和压缩的对序列记忆通道，并在字节级 WikiText-2、MiniMind 以及关联召回任务上进行实验。

**💡 创新点**

提出通过三元隐状态压缩捕获高阶 token 交互的通用架构，并证明仅在压缩状态中不需要显式检索，采用分离的门控键值路径可显著提升关联召回。

**🔧 技术方法**

使用递归式状态更新、压缩对记忆（dense、slot、卷积增强）、与 Transformer 的兼容层、以及带门控的后端键值检索实现高阶交互与检索。

**📊 数据集**

主要使用字节级 WikiText-2、MiniMind 的 tokenizer‑based 预训练语料，以及人工构造的四对键值的关联召回数据集。

**📈 对比分析**

与基准 Transformer 在相同参数/宽度下对比，三元隐状态模型在字节级 WikiText-2 上压缩率提升至约 4.76 bits/byte，MiniMind token loss 下降至 7.03（或 gated 6.77），但在关联召回上仅基础模型 13% ，门控混合模型提升至 41.9%（最佳 100%），整体速度显著慢于 Transformer。

**⚠️ 局限性**

实验规模小、仅在 Apple MPS 的 Python 循环实现上跑，缺乏优化内核，吞吐量远逊于标准注意力；门控检索表现高度种子敏感且不稳定；未进行 FLOP 对齐的训练；仅覆盖有限的预训练任务，未验证长序列或大规模数据的效果。

---

## 45. Probing Spatial Structure in Pretrained Audio Representations

**arXiv ID:** 2606.05544 | [PDF](https://arxiv.org/pdf/2606.05544v1)

**作者:** Chuyang Chen `[一作]` (New York University), Juan Pablo Bello `[通讯]` (New York University)

**通讯引用:** 8709 | [OpenAlex ID](https://openalex.org/A5031398497)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了SARL基准，用以对预训练空间音频编码器的空间因子表征进行系统性评估。

**💡 创新点**

创新点在于构建统一的线性探测框架，结合可控合成空间场景，对源级与房间级因素进行分层分析。

**🔧 技术方法**

采用线性探测器、余弦相似度敏感性分析以及多种自监督、监督与编码器训练范式的预训练模型。

**📊 数据集**

使用ESC‑50、MUSAN、UrbanSound8K合成的单源场景，并结合AudibleLight与PyRoomAcoustics生成RIR，构建源级（方位、仰角、距离、事件）与房间级（RT60、体积、形状）七任务数据集。

**📈 对比分析**

通过比较不同输入格式（单声道、立体声、双声道、FOA）和训练范式（自监督、监督、编解码）下的线性探测准确率，发现FOA+自监督模型在源级和房间级任务上均显著优于单声道或立体声模型，且源级因素的可解码性能远高于房间级因素。

**⚠️ 局限性**

局限性包括仅使用单源合成场景，未涵盖多源真实录音；评估仅限于线性探测和冻结特征，可能低估模型潜在的非线性空间信息。

---

## 46. I Know What You Meme, Even If it Emerged Today: Understanding Evolving Memes through Open-World Knowledge Acquisition

**arXiv ID:** 2606.05316 | [PDF](https://arxiv.org/pdf/2606.05316v1)

**作者:** Shanhong Liu `[一作]` (Singapore University of Technology and Design), De Wen Soh `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 128 | [OpenAlex ID](https://openalex.org/A5083355429)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了零射击框架Query‑Retrieve‑Conclude，通过查询缺失知识、检索网络证据并合成背景知识，提升多模态表情包的理解与检测。

**💡 创新点**

创新点在于将缺失知识识别、外部检索和知识合成三阶段流程嵌入多模态推理，避免直接生成导致的幻觉，并构建覆盖2024‑2026新表情包的KYMD基准。

**🔧 技术方法**

使用逆向图像搜索、视觉描述生成、问题生成、Web检索、证据生成、声明合成以及LLM评估与多模态模型进行推理。

**📊 数据集**

实验使用MemeIntent、MemeInterpret、KYM三大数据集，以及五个检测任务数据集（Hatefulness、Misogyny、Offensiveness、Sarcasm、Harmfulness）。

**📈 对比分析**

与零射击无知识、零射击生成知识、MemeAgent、MiND等基线相比，检索召回率从0.46升至0.78，整体检测F1提升至0.71，显著优于传统方法。

**⚠️ 局限性**

局限性包括对外部检索质量的依赖、检索成本高、仅在英文表情包上验证、以及可能引入偏见或敏感信息。

---

## 47. Harnessing Generalist Agents for Contextualized Time Series

**arXiv ID:** 2606.05404 | [PDF](https://arxiv.org/pdf/2606.05404v1)

**作者:** Zihao Li `[一作]` (University of Illinois Urbana Champaign), Jingrui He `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 TimeClaw——一种针对上下文化时间序列的本地化智能体驱动框架，通过可执行时间工具、能力演化和多模态记忆，赋能通用 LLM 进行端到端的时间序列推理。

**💡 创新点**

创新点包括：① 将时间序列数据以本地化运行时对象提供给 LLM，解决序列序列化导致的数据与过程失配；② 基于经验的能力演化机制，自动将重复使用的分析子流程抽象为可执行工具；③ 采用文本与时间序列指纹双模检索的多模态记忆，提升记忆检索的相关性。

**🔧 技术方法**

技术上采用冻结的 LLM（如 GPT‑5‑nano）配合工具接口、可审计轨迹记录；利用代码专用 LLM 进行工具生成与验证；构建时间序列指纹与文本嵌入的检索键；并在实验中使用 RCRPS、sMAPE、准确率等指标。

**📊 数据集**

实验数据集包括 Context‑is‑Key (CiK) 关注上下文利用；TSRBench 的多任务多模态评测；以及 TSAIA 金融时间序列分析基准，此外还覆盖能源、金融、天气、交通等领域的公开时序数据。

**📈 对比分析**

与传统统计模型（ARIMA、ETS）、神经网络预测（DLinear、PatchTST）、时间序列基础模型（Chronos、Lag‑Llama、Moirai）、专用 LLM/智能体（UniTime、Time‑LLM、TS‑Agent、TSci）、通用 LLM（LLaMA3‑70B）以及通用智能体流水线（Prompting、CoT、ReAct 等）进行对比。TimeClaw 在 CiK 上 RCRPS 提升 11.5%、sMAPE 降低；在 TSRBench 上平均准确率提升 15.8%；在 TSAIA 上相较于金融专用智能体提升 38.9%；同时在 token 需求上比多智能体反射方案减少 43.6%。

**⚠️ 局限性**

局限性包括：仅在文本‑时间序列混合场景中验证；对大型记忆库和指纹提取的计算成本未充分评估；基于冻结 LLM 的方法在实时动态环境下的适应性待验证；以及需要在更广泛的多模态与跨领域任务上进一步检验。

---

## 48. ReasoningFlow: Discourse Structures for Understanding LLM Reasoning Traces

**arXiv ID:** 2606.05402 | [PDF](https://arxiv.org/pdf/2606.05402v1)

**作者:** Jinu Lee `[一作]` (University of Illinois Urbana Champaign), Julia Hockenmaier `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并验证了一套细粒度的 LRM 推理轨迹注释框架 ReasoningFlow，并利用该框架对多模型、多任务的推理轨迹进行了大规模注释。

**💡 创新点**

首次以有向无环图形式定义 8 种节点和 14 种边的细粒度推理结构，结合人工与 LLM 自动注释，实现对 LRM 非线性推理行为的可解释性分析。

**🔧 技术方法**

采用 LLM 驱动的自动注释流水线、Krippendorff α 交叉验证、PCA 与 Jensen–Shannon Divergence 等统计方法，以及 Thought Anchors 机制进行对比。

**📊 数据集**

使用 AIME 2024、GPQA-Diamond 与 ArgKP 三大推理基准，采集了五个 LRM 与两非推理模型的 1,260 条推理轨迹。

**📈 对比分析**

通过 triplet 分布的 PCA 聚类和 JS‑Divergence 衡量模型间结构相似度，结果显示不同 LRM 在相同任务下结构高度相似；自检误差分析表明仅 14.4% 的错误导致最终答案错误。

**⚠️ 局限性**

自动注释成本高、未对全部数据进行人工复核、仅涵盖三款开源 LRM，未覆盖闭源或蒸馏模型，且与机制性因果关系的对齐仍存在差距。

---

## 49. Willing but Unable: Separating Refusal from Capability in Code LLMs via Abliteration

**arXiv ID:** 2606.05396 | [PDF](https://arxiv.org/pdf/2606.05396v1)

**作者:** Cristina Carleo `[一作]` (University of Naples Federico II), Domenico Cotroneo `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 4038 | [OpenAlex ID](https://openalex.org/A5052961846)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究将低秩权重编辑（abliteration）应用于指令微调代码 LLM，评估其在 Python SQL 注入（CWE‑89）任务中的拒绝行为及生成能力。

**💡 创新点**

创新点在于证明 abliteration 能彻底移除安全对抗的拒绝行为而不损害代码生成质量，并首次揭示模型在此任务中的“愿意”与“能力”两维度可分离。

**🔧 技术方法**

采用低秩权重编辑技术对 Qwen2.5‑Coder‑Instruct 进行修改，配合 CodeQL、Semgrep 与 Bandit 三工具的静态分析器以及人工裁定验证；推理使用 4‑bit GGUF 量化模型。

**📊 数据集**

使用公开的 Python 代码安全数据集 PromSec 与 SafeCoder（包含 Flask/MySQL/SQLite 等安全实现及对应的漏洞版本）作为实验样本。

**📈 对比分析**

对同一提示在基线与 Abliterated 权重下各执行 3 次生成，统计拒绝率、注入率与语法有效率。结果显示 Abliterated 将拒绝率降至 0% 并保持 >93% 语法正确率，注入率在 7B/14B 约 90%（PromSec 88/90，SafeCoder 89/97），而 3B 仅 25–48%。

**⚠️ 局限性**

局限性：仅评估单一 CWE‑89、单一语言 Python、单一模型族 Qwen2.5‑Coder；使用 4‑bit 量化推理，可能影响生成行为；检测器误差与人工裁定难以完全覆盖真实漏洞；未探究其它漏洞类型、语言或模型的迁移性。

---

## 50. Temporal Preference Concepts and their Functions in a Large Language Model

**arXiv ID:** 2606.05194 | [PDF](https://arxiv.org/pdf/2606.05194v1)

**作者:** Ian Rios-Sialer `[一作]` (AI Safety Camp), Justin Shenk `[通讯]` (Supervised Program for Alignment Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对LLM中的时间偏好进行因果定位、几何特征化，并通过激活空间干预实现双向调节。

**💡 创新点**

首次在单一LLM中定位到时间偏好子图，揭示其非线性几何结构，并展示可通过激活添加实现可控的时间偏好调节。

**🔧 技术方法**

多模态因果定位（梯度归因、线性探针、激活补丁、CAA），PCA几何分析，行为评估与激活空间干预。

**📊 数据集**

包含500对显式/隐式A/B提示、4588个参数化时间提示、160个IOI分类提示、960个投资一致性测试，以及Kirby MCQ-27问卷。

**📈 对比分析**

与29款现有模型对比，证明在时间偏好一致性方面仅少数前沿API模型表现优异；使用CAA干预时，在层22、α=50可将长短期偏好提升至相对几率1.39（约3.4倍），显著优于基线。

**⚠️ 局限性**

局限于单模型单任务、缺乏更细粒度电路追踪、未能验证在多轮/更大规模模型上的普适性，以及线性CAA在高幅度时易导致输出质量下降。

---

## 51. DeployBench: Benchmarking LLM Agents for Research Artifact Deployment

**arXiv ID:** 2606.05238 | [PDF](https://arxiv.org/pdf/2606.05238v1)

**作者:** Yuanli Wang `[一作]` (Boston University), Liqiang Jing `[通讯]` (University of Texas at Dallas)

**通讯引用:** 301 | [OpenAlex ID](https://openalex.org/A5051341911)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套名为DeployBench的多领域基准，用来评估大型语言模型（LLM）代理从零开始构建可运行环境并成功执行研究论文中指定实验的能力；

**💡 创新点**

创新点在于：①覆盖多语言、多学科（AI/ML、计算机系统、科学计算）51项真实研究部署任务；②任务验证采用隐藏的两层专属检验器，确保环境不仅能跑通而且能产生论文所需输出；③将部署挑战从容器化转向裸机、GPU/ CUDA、内核编译、旧代码兼容等真实系统级难题；

**🔧 技术方法**

使用OpenHands框架作为代理脚手架，测试了四款最先进LLM（GPT‑5.3‑Codex、Gemini‑3.1‑Pro、Grok‑4.20、GPT‑5.4‑Mini），并辅以LLM诊断代理分析失败原因；

**📊 数据集**

数据集为从2008至2025年顶级会议（NeurIPS、ICML、SOSP、EuroSys等）挑选的51个公开研究代码仓库，涵盖11种语言生态；

**📈 对比分析**

评估指标为任务成功率。四款模型在DeployBench上的总体通过率分别为51.0%、27.5%、11.8%和7.8%。通过率按任务类别、难度和仓库年龄细分，显示AI/ML与系统任务均高于科学计算，GPU任务与旧版代码更具挑战；

**⚠️ 局限性**

限制主要体现在：①现有LLM在完整部署与验证上的准确率仍低；②自检与完成判断失误占绝大多数（97/154），表明代理对论文要求的理解与验证仍不充分；③基准尚未覆盖更广的系统级任务（如分布式调度、网络服务交付）以及更大规模的深度学习实验。

---

## 52. Drishti AI-Event Guardian: An Intelligent Real-Time Crowd Monitoring and Emergency Response System for Mass Gathering Events

**arXiv ID:** 2606.05185 | [PDF](https://arxiv.org/pdf/2606.05185v1)

**作者:** Ritabrata Roy Choudhury `[一作]` (Kalinga Institute of Industrial Technology), Rudra Pratap Mitra `[通讯]` (Kalinga Institute of Industrial Technology)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 Drishti AI‑Event Guardian，一套集成深度学习与多模态传感器的实时人群监控与紧急响应平台，涵盖人群密度估计、异常检测、五分钟预测、缺失人员面部识别与群体通知、医疗紧急派遣、自然语言聊天机器人以及警卫动态重调等模块。

**💡 创新点**

创新点包括：多摄像头几何融合与 UAV + CCTV 组合的全景覆盖；基于 YOLOv8 的实时人检测与统计异常阈值的自适应警戒；使用 XGBoost 的短期拥挤预测；面部嵌入与车轮式通知实现缺失人员的群体搜索；医疗报告自动分级与多接收端即时派遣；LLM 驱动的多语言聊天机器人实现自然语言报案；警卫重调引擎将预测与实际密度映射为实时部署指令；双向市民参与机制将市民报案纳入传感层。

**🔧 技术方法**

核心技术：YOLOv8（人检测） + ArcFace（面部嵌入） + XGBoost（拥挤预测） + 多摄像头几何投影 + NMS + CLAHE + Gaussian 去噪 + 统计异常检测 + FAISS 索引 + Vertex AI、GCP、Kubernetes 微服务 + Firebase Realtime Database / Cloud Messaging + WebRTC/HLS + LLM（检索增强生成）+ RAG 聊天机器人 + S3/Firestore + REST/SSE 接口。

**📊 数据集**

使用数据集：CrowdHuman、VisDrone2023、Kumbh Mela 2019 现场视频、现场采集的 28 CCTV + 6 UAV 影像、RCB Victory Parade 17 CCTV + 3 UAV 影像、移动应用报案日志、缺失人照片、医疗报案文本/图片；全部均为真实事件的多模态数据。

**📈 对比分析**

通过与人工基线和传统监控对比，评估指标包括：人群密度 MAE 3.2 人/m²、F1 0.91、面部识别精度 0.93、医疗派遣中位延迟 4.3 s、聊天机器人完成率 89%、警卫重调平均延迟降低 69%、预测 MAPE 8.3%、总延迟 115 ms（P99 181 ms）。实验覆盖两大规模实战场景，表现优于现有系统。

**⚠️ 局限性**

局限性：夜间面部识别召回下降（从 0.84 降至 0.68），聊天机器人会话放弃率 7%（主要因照片上传阻碍），网络依赖导致群体通知与聊天机器人在极端密集区域可能受限，摄像头几何校准需要人工地面控制点，隐私与生物识别合规风险（短期存储与删除策略）、对抗性报案与面部匹配风险、UAV 部署成本与操作复杂度、缺乏自校准的摄像头融合算法。

---

## 53. Where Do Large Language Models Fail on Competitive Programming? A Taxonomy of Failures by Algorithm Type and Difficulty Rating

**arXiv ID:** 2606.05228 | [PDF](https://arxiv.org/pdf/2606.05228v1)

**作者:** Ayush Kumar Jha `[一作]` (Indian Institute of Information Technology Bhubaneswar), Shalini Jha `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了两大前沿LLM在315道Codeforces竞赛题上的算法失败模式，使用了两种提示策略（直接生成与Chain-of-Thought）。

**💡 创新点**

创新点在于构建了二维算法类别×难度层次的失效税onomies，并揭示了CoT在竞赛编程中对模型性能的显著负面影响以及格式化崩溃的现象。

**🔧 技术方法**

主要技术包括零样本生成、严格的执行沙盒评估、错误类型分类以及对比实验的消融研究。

**📊 数据集**

使用的数据集为CodeContests中仅包含Codeforces题目的315道平衡样本，覆盖七大算法类别和三难度层次。

**📈 对比分析**

通过与直接提示基准对比，发现GPT‑4o在CoT条件下从46.0%降至36.8%，Claude Sonnet 4.6在CoT下略有提升至63.5%，并指出WA占大多数失败原因。

**⚠️ 局限性**

局限性包括对测试集仅使用公开预检用例、Python 3环境限制、样本量不足导致统计误差、以及CoT提示可能导致的Token上限溢出问题。

---

## 54. RAINO: Anchoring Agents in Reality, A Systematic Review and Conceptual Framework for Realism in Agent-Based Modelling

**arXiv ID:** 2606.05167 | [PDF](https://arxiv.org/pdf/2606.05167v1)

**作者:** Loïs Vanhée `[一作]` (Umeå University), Melania Borit `[通讯]` (UiT Arctic University of Norway)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统文献综述（SLR）探讨了代理模型中现实性概念的定义、使用与论证，并提出了新的 Reality Anchor, Input, Output（RAINO）框架以系统化现实性构建与评估过程

**💡 创新点**

创新点在于首次将现实性作为系统性研究对象，揭示其概念缺失与多样性，并设计RAINO框架整合现实锚、输入/输出维度，提供评价现实性的结构化视角

**🔧 技术方法**

使用的技术包括PRISMA标准的系统检索、内容分析与层级编码，对文献进行定性归纳与可视化

**📊 数据集**

所使用的数据集为在Scopus数据库检索得到的73篇标题包含“realism”或“realist”的代理模型研究论文

**📈 对比分析**

本文主要以定性对比方法出现频率、类别和论证结构，未涉及数值性能比较，结果显示现实性论证方法多样但缺乏理论依据

**⚠️ 局限性**

局限性包括检索范围仅限标题导致可能遗漏大量讨论现实性的文献、样本量有限、未能系统评估方法有效性与实际应用场景验证

---

## 55. Staged Factorial Screening for Budget-Constrained Micro-Pretraining

**arXiv ID:** 2606.05186 | [PDF](https://arxiv.org/pdf/2606.05186v1)

**作者:** Felipe Chavarro Polania `[一作]` `[通讯]` (Hewlett Packard Enterprise), Felipe Chavarro Polania (Hewlett Packard Enterprise)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在极短预算下的分阶段微预训练实验设计，利用分块因子实验先识别高惩罚方向，再进行锚点确认和局部精细化。

**💡 创新点**

创新点在于将分块因子设计与基于预算的早期效果结构恢复相结合，并通过“桥接”锚点展示了在预算增加时模型大小与批量大小惩罚的显著缓解。

**🔧 技术方法**

采用分块因子实验、Benjamini–Hochberg多重检验、匹配成本的随机搜索、贪心搜索以及种子重跑等技术手段。

**📊 数据集**

使用Karpathy的Climbmix 400B shuffle数据集并在单GPU上进行训练。

**📈 对比分析**

与贪心和随机搜索基线对比，结果显示在60min-24h锚点续跑中桥接锚点往往拥有最低bits/byte，但整体提升有限。

**⚠️ 局限性**

局限包括仅在单一硬件上进行深度实验，跨主机验证不完全，缺乏下游任务评估，实验规模较小且未覆盖更高级的多保真搜索方法。

---

## 56. Exploring LLMs for South Asian Music Understanding and Generation

**arXiv ID:** 2606.05522 | [PDF](https://arxiv.org/pdf/2606.05522v1)

**作者:** Faria Binte Kader `[一作]` (University of Central Florida), Santu Karmaker `[通讯]` (University of Central Florida)

**通讯引用:** 322 | [OpenAlex ID](https://openalex.org/A5058755530)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并执行了面向南亚印度传统音乐（孟加拉古典歌谣）的LLM理解与生成评估实验，构建504题测验与100首ABC谱例，并测试33款LLM。

**💡 创新点**

首次系统评估LLM在非西方低资源音乐语境下的语义理解与生成能力，并提出五层控制提示框架及对比性评估指标。

**🔧 技术方法**

采用LLM推理、ABC符号音乐表示、TELeR提示分类、自动度量（KL散度、ABC语法正确率）与人工评测。

**📊 数据集**

自行编制504题多选问答与100首Rabindra/Nazrul歌曲的ABC谱，来自官方Swaralipi，并使用公开GitHub资源。

**📈 对比分析**

在理解任务中对33模型进行准确率对比，Gemini 2.5 Pro最高达90%以上；在生成任务中用自动指标和三名专业评测员评估，Gemini 2.5 Pro在结构性得分高，但风格准确率仅40%，显示结构与风格不匹配。

**⚠️ 局限性**

评测仅覆盖两种孟加拉古典体裁，ABC符号无法完整表达装饰音与微音；数据量有限，且自动指标与人评判的相关性低。

---

## 57. MoDex: A Diffusion Policy for Sequential Multi-Object Dexterous Grasping

**arXiv ID:** 2606.05407 | [PDF](https://arxiv.org/pdf/2606.05407v1)

**作者:** Haofei Lu `[一作]` (KTH Royal Institute of Technology), Danica Kragic `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 15629 | [OpenAlex ID](https://openalex.org/A5023792180)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练并部署了一个单一的扩散策略，用来在不放手的前提下，依次抓取多个物体，充分利用多指手的自由度；

**💡 创新点**

通过引入“Opposition Space”（手指配对指令）与抓取历史上下文，使策略能够在每一步选择最少的手指，保留未使用的自由度，并在两阶段训练（行为克隆 + DPPO）中专门为多物体顺序抓取设计奖励；

**🔧 技术方法**

使用扩散策略（DP3）与DPPO强化学习，PointNet++ 编码点云，Allegro Hand + Franka Panda 机械臂，Kinect v3 深度摄像头，以及OSC 控制器；

**📊 数据集**

在 Robosuite 环境中自动生成的多物体顺序抓取演示数据集，包含 15 种球、圆柱、盒子等物体，共约 1,600 条轨迹；

**📈 对比分析**

与 BC‑RNN、PPO、SeqDiffuser 以及仅行为克隆的 baseline 比较。仿真中，-BC（本方法）平均成功率达 56.5%，明显高于其他方法；真实世界中 Stage‑1、2、3 的成功率分别为 57.8%、26.7%、20%，相较于-BC 下降显著，验证了 DPPO 细化的有效性；

**⚠️ 局限性**

不支持手中重抓（in‑hand regrasp），且手指配对（os）和抓取顺序需预先给定，无法自动推理最佳分配与顺序。

---

## 58. Mamba-Assisted Non-Markovian Closure for Reduced-Order Modeling

**arXiv ID:** 2606.05371 | [PDF](https://arxiv.org/pdf/2606.05371v1)

**作者:** Zhi-Feng Wei `[一作]` (Pacific Northwest National Laboratory), Panos Stinis `[通讯]` (Pacific Northwest National Laboratory)

**通讯引用:** 885 | [OpenAlex ID](https://openalex.org/A5078411693)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 Mamba‑Assisted Closure (MAC) 框架，将高维动力系统的非马尔科夫闭包项视为序列建模问题，并通过 Mamba 网络学习闭包，随后与降阶模型耦合实现自回归预测。

**💡 创新点**

创新点在于将 Mori‑Zwanzig 记忆积分与状态空间模型的卷积结构对应，利用 Mamba 的输入依赖选择机制自适应确定有效记忆深度，同时在训练时采用并行卷积形式、推理时采用递归形式，实现长序列高效训练与恒定步长推理。

**🔧 技术方法**

技术手段包括 Mamba 结构化状态空间模型、卷积/递归双重表示、教师强制训练、零阶保持的 RK4 积分、噪声注入、与 GRU 和 Wilks 统计方法的对比。

**📊 数据集**

实验数据集涵盖可粘性 Burgers 方程（随机低频初始条件、三种 OOD 初始）和两尺度 Lorenz '96 系统（训练 10001 步、验证 2001 步、测试 2001 步及 100 条 OOD 初始），对两类系统的闭包问题进行评估。

**📈 对比分析**

与 Markovian ROM、GRU 序列模型和 Wilks 方法对比；在 Burgers 方程的插值与外推阶段，MAC 的相对 L² 误差比 Markovian 低约 10 倍；在 Lorenz '96 的插值、外推及 OOD 情况下，累计误差显著下降、相关系数保持 >0.99，能够维持长期稳定的预测。

**⚠️ 局限性**

局限性包括：对极端 OOD 或高度非线性场景的闭包学习仍有限；训练需要大量数据与计算资源；目前仅验证确定性系统，尚未扩展到随机闭包或更高维度流体/材料场景。

---

## 59. Multilingual Detection of Alzheimer's Disease from Speech: A Cross-Linguistic Transfer Learning Approach

**arXiv ID:** 2606.05545 | [PDF](https://arxiv.org/pdf/2606.05545v1)

**作者:** Nadine Yasser Abdelhalim `[一作]` (Imperial College London), Nicole Salomons `[通讯]` (Imperial College London)

**通讯引用:** 901 | [OpenAlex ID](https://openalex.org/A5076361939)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一系列多语言的阿尔茨海默病（AD）检测模型，利用跨语言训练在英文、中文、阿拉伯语和印地语等多种语言的语料上进行二分类任务。

**💡 创新点**

创新点在于证明多语言Transformer（XLM‑RoBERTa）能够在未见语言上实现跨语言迁移，显著降低了为每种语言单独训练模型的成本，并通过与单语模型的对比展示了跨语言学习的可行性。

**🔧 技术方法**

主要技术包括XLM‑RoBERTa多语言Transformer、Fine‑tune、AdamW优化器、超参数调优（batch 16/32，学习率1e-5~5e-5，epoch 10~45），以及Whisper ASR用于中文转写。

**📊 数据集**

使用的数据集包括：DementiaBank Pitt（英文）、DementiaBank Mandarin Lu（台湾国语）、DementiaBankHindi（人工翻译和机器翻译）、2024 TAUKADIAL（中英文）、以及GPT‑4+人工校正的阿拉伯语翻译数据，共计约549份转录文本。

**📈 对比分析**

通过五个实验（四个留一语言测试 + 一次全语言训练）与单语模型对比评估。多语模型在英文、阿拉伯语、印地语、中文的F1分别为76%、71%、61%和96%，整体平均82%；相较于单语模型（英文85%、阿拉伯语82%、印地语82%）表现略逊，尤其印地语下降21%。推断时间稳定在约0.5秒，符合实时筛查需求。

**⚠️ 局限性**

局限性包括：印地语性能显著下降；缺乏非痴呆病例的中文数据导致无法训练中文单语模型；仅基于文本特征，未结合声学或其他多模态信息；所用语料在质量和代表性上仍有限，可能影响跨语言泛化效果。

---

## 60. Dominant-Layer ZO: A Single Layer Dominates Zeroth-Order Fine-Tuning of LLMs

**arXiv ID:** 2606.05516 | [PDF](https://arxiv.org/pdf/2606.05516v1)

**作者:** Wanhao Yu `[一作]` (University of North Carolina at Charlotte), Li Yang `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 2963 | [OpenAlex ID](https://openalex.org/A5100421627)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了零阶优化（ZO）在大型语言模型微调中的层级适配现象，发现只需微调单一“主导层”即可达到或超过全模型微调效果；

**💡 创新点**

首次揭示ZO微调中主导层现象，并提供仅通过推理阶段激活异常点统计即可预先定位主导层的简易方法；解释了该层在残差流中早期位置与高扰动敏感度共同导致其在ZO中占主导地位；

**🔧 技术方法**

采用ZO优化（MeZO）与SPSA梯度估计、第一阶微调（AdamW）对比、激活异常点统计、层级微调实验、Sparse‑MeZO稀疏扰动等技术；

**📊 数据集**

在LLaMA2‑7B和Qwen3‑8B两大模型上，使用SST‑2、RTE、CB、BoolQ、WSC、MultiRC、COPA、SQuAD、DROP九个下游任务进行实验；

**📈 对比分析**

与全模型MeZO、MeZO LoRA、Sparse‑MeZO、FO AdamW等方法对比，主导层ZO在大多数任务上与全模型MeZO相当或更优，平均提升约0.5–1.1个百分点；训练速度提升1.12–4.52倍（单层参数扰动与更新显著降低）；

**⚠️ 局限性**

仍与第一阶微调存在性能差距，需要更多步骤；未尝试更大模型或与更高效的ZO优化器（如ZO‑AdaMM、FZOO）结合；在更多LLM上的通用性仍待验证。

---

## 61. SHALA-LLM: Smartly Handling Ambiguous Labels in Aligning LLMs

**arXiv ID:** 2606.05376 | [PDF](https://arxiv.org/pdf/2606.05376v1)

**作者:** Jingyao Wu `[一作]` (Massachusetts Institute of Technology), Rosalind Picard `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于强化学习的对齐框架，使LLM能够直接学习并对齐多注释者分布，并通过动态歧义加权提升学习效率；

**💡 创新点**

将注释者分布视为监督信号，设计基于 Jensen–Shannon 距离与熵加权的奖励，并在 GRPO 中实现动态歧义优先；

**🔧 技术方法**

使用 Group Relative Policy Optimization（GRPO）、文本化概率分布生成、JS 距离奖励与熵调制等技术；

**📊 数据集**

在多种模糊标签任务上评估，包括 ChaosNLI 及其子集、MSP‑Podcast、GoEmotions 等 NLI 与情感识别数据集；

**📈 对比分析**

与零射、众数标签以及其他方法对比，在 JSD、BC、Accuracy 与 F1 等指标上显著提升：JSD 减少约 60%、F1 提升约 30%，并在不同歧义层级保持稳健；

**⚠️ 局限性**

仅在结构化标签分布任务上验证，未覆盖开放式生成或长篇推理任务；奖励假设注释者分布能完整表达歧义，未考虑注释者专业性、背景差异；仅在 GRPO 框架内测试，缺乏跨 RL 框架的通用性验证。

---

## 62. Domain-Conditioned Safety in Frontier Computer-Using Agents: A 793-Episode Browser Benchmark, a Coding-Domain Cross-Reference, and a Reproducibility Audit of Recent Red-Teaming

**arXiv ID:** 2606.05233 | [PDF](https://arxiv.org/pdf/2606.05233v1)

**作者:** Nicholas Saban `[一作]` `[通讯]` (Patronus AI), Nicholas Saban (Patronus AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布了一个793条episode的公开浏览器红队基准，用手工模板评估最新CUA模型的注入成功率

**💡 创新点**

发现最新前沿模型对手工注入完全无效，但对编码代理的技能注入仍高度易受攻击，揭示安全硬化的域依赖性和RL优化文本的显著优势

**🔧 技术方法**

使用手工模板、Prompt Ablation、AutoInject RL攻击、AgentDojo等工具和方法进行实验

**📊 数据集**

构建了24个多步网页任务、8个注入模板和56个攻击模式，覆盖8个站点和5级深度；并与编码代理的同一模型权重做跨域对比

**📈 对比分析**

对比显示前沿Claude Sonnet 4.6和GPT‑5.4在浏览器任务上手工注入成功率为0%，但在编码任务上可达100%（Sonnet）或79%（GPT‑5.4），表明模型在浏览器域已强化但未迁移到其他任务域；RL攻击在同一配置下能略高于手工但仍有限

**⚠️ 局限性**

手工模板仅为近似，未覆盖所有优化文本；实验仅涉及两个前沿模型；图像通道覆盖有限；跨域对比仅限浏览器与编码代理两种表面，可能不具普遍性

---

## 63. Multi-Granularity Reasoning for Natural Language Inference

**arXiv ID:** 2606.05181 | [PDF](https://arxiv.org/pdf/2606.05181v1)

**作者:** Chunling Xi `[一作]` (Pacific Insurance Technology Co., Ltd.), Di Liang `[通讯]` (Lixin Information Services Co., Ltd.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多粒度推理网络（MGRN），通过多层次交互张量实现自然语言推理；

**💡 创新点**

创新点在于逐层捕获词级、短语级和上下文级语义的细粒度交互，并将其通过 DenseNet 进行层叠式特征提炼；

**🔧 技术方法**

技术包括 BERT 预训练模型多层表示、词对词交互矩阵构造、DenseNet 级联特征提取以及最终的全连接分类；

**📊 数据集**

使用 SNLI、MultiNLI、QQP 等公开基准数据集进行实验；

**📈 对比分析**

与 BERT、RoBERTa、SemBERT 等基线模型对比，平均提升约 0.8%–1.5%，在大多数任务上实现或逼近最优性能；

**⚠️ 局限性**

局限在于对预训练模型的依赖较重，对极端对抗样本的鲁棒性尚需提升，以及未充分利用外部知识或更长文本的多句推理。

---

## 64. REStack: A Large-Scale Dataset of Reverse Engineering Discussions from Stack Exchange

**arXiv ID:** 2606.05493 | [PDF](https://arxiv.org/pdf/2606.05493v1)

**作者:** Md Humaun Kabir `[一作]` (Lamar University), Farha Kamal `[通讯]` (Lamar University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究采集并构建了来自Stack Overflow与专属Reverse Engineering Stack Exchange的12,293条RE相关讨论数据集，并通过主题建模与人工标注生成了23个主题与6个主题类别；

**💡 创新点**

首次公开专门面向逆向工程领域的Q&A数据集，并采用遗传算法优化的LDA模型进行主题发现，提供多维度的难度与流行度指标；

**🔧 技术方法**

采用文本预处理（去除代码块、HTML、数字、停用词、词形还原）、遗传算法优化的LDA主题模型、手工主题标注以及统计检验（Kruskal‑Wallis、卡方检验、Spearman相关、Cliff’s delta）进行数据分析；

**📊 数据集**

使用了12,293条RE相关帖子，覆盖2008‑2025年，来自Reversing Stack Exchange（9,845条）与Stack Overflow（2,449条）的公开数据；

**📈 对比分析**

通过统计检验验证主题难度指标的显著性，并将该数据集作为基准用于LLM问答与专家推荐系统的评估；目前未给出具体性能指标，但已证明数据集可支持多维度评测；

**⚠️ 局限性**

局限性包括：仅来源于公开论坛，可能不涵盖私有社区与企业实践；标签与主题分类存在主观性；社区投票与回答质量不一定代表技术正确性；数据反映历史讨论，未考虑近期AI工具对知识获取行为的影响。

---

## 65. Stability vs. Manipulability: Evaluating Robustness Under Post-Decision Interaction in LLM Judges

**arXiv ID:** 2606.05384 | [PDF](https://arxiv.org/pdf/2606.05384v1)

**作者:** Srimonti Dutta `[一作]` (WAI USA Research Labs), Akshata Kishore Moharir `[通讯]` (WAI USA Research Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型在评判任务中，初始决策后通过对话挑战能否被改变，探究了“后决策可操控性”这一现象。

**💡 创新点**

首次提出了抗基线挑战和目标平衡审计两种实验协议，并引入了 Evaluation Robustness Score (ERS) 用来量化评判的交互鲁棒性。

**🔧 技术方法**

采用对话式干预（疑问、权威、证据）对 LLM 评判进行后续交互，并利用确定性解码在 GPT‑4o 与 GPT‑4o‑mini 上进行实验。

**📊 数据集**

实验数据来自 MT‑Bench（多轮对话）和 AlpacaEval（单轮指令），共 100 对评估实例，分别包含多种模型产生的回答。

**📈 对比分析**

与基线对比，重复评估几乎无漂移（0.5%），但在抗基线挑战下 Flip Rate 高达 49%（权威提示 74%），与人类偏好的一致性下降至 48%，排行榜 Kendall τ 由 1.0 降至 0.50；ERS 在抗基线条件下为 0.51，说明鲁棒性差；在目标平衡审计中 PS=0.194，DS≈0，ERS 高达 0.903，表明主导失效是可逆性而非目标驱动。

**⚠️ 局限性**

仅评估两类评判模型、两个 benchmark、样本量 100 对，未检验其他模型架构或任务领域；实验设计为受控对话，缺乏对真实评估管道中多评审或规则约束的考察；因此结果可能不完全适用于更大规模或更复杂的评估环境。

---

## 66. Worst-Case Update Complexity of the Preisach Extremum Stack

**arXiv ID:** 2606.05245 | [PDF](https://arxiv.org/pdf/2606.05245v1)

**作者:** Piotr Frydrych `[一作]` (Warsaw University of Technology), Piotr Frydrych `[通讯]` (Warsaw University of Technology)

**通讯引用:** 110 | [OpenAlex ID](https://openalex.org/A5012740592)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了Preisach极大极小栈的最坏情况复杂度，并提出了基于指针树（finger-tree）的实现，保证每步O(log k)的最坏情况时间。

**💡 创新点**

首次给出从数组、二进制搜索到指针树三层复杂度图景，并证明指针树能在保持精确最小化的同时实现O(log k)的最坏情况操作。

**🔧 技术方法**

采用输出变化模型、Kolmogorov复杂度分析、指针树（2‑3 finger tree）分裂（Split）操作以及对比分析。

**📊 数据集**

主要基于理论构造和对抗性输入，没有使用具体实验数据集，关注理论复杂度和时间/空间性质。

**📈 对比分析**

对阵数组（O(1)摊销、Θ(k)输出变化）、二进制搜索数组（O(log k + d)内存操作）和指针树（O(log k)内存操作）进行比较；指针树在最坏情况下内存操作最小，输出变化不变，适合低延迟系统。

**⚠️ 局限性**

指针树实现空间开销比数组大，结构复杂；仍需在每步记录Θ(k)输出变化，且对随机化算法或向量栈的扩展尚未解决。

---

## 67. Uncertainty-Aware Adaptive Sensor Fusion for Autonomous Navigation

**arXiv ID:** 2606.05437 | [PDF](https://arxiv.org/pdf/2606.05437v1)

**作者:** Simegnew Yihunie Alaba `[一作]` (Virginia Commonwealth University), Yuichi Motai `[通讯]` (Virginia Commonwealth University)

**通讯引用:** 1604 | [OpenAlex ID](https://openalex.org/A5061499121)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于深度学习与UKF的自适应感知融合框架，用于视觉‑惯性里程计的姿态估计。

**💡 创新点**

创新点包括：①利用Vision Transformer提取IMU时序特征，②用多尺度CNN提取视觉光流特征，③设计基于不确定性估计的门控自适应融合模块，④引入不确定性感知损失函数提升鲁棒性，⑤实现轻量化模型，达到155 FPS。

**🔧 技术方法**

技术栈：Vision Transformer、Multiscale CNN、Concrete分布门控自适应融合、Unscented Kalman Filter、ALEATORIC/EPistemic不确定性估计与不确定性感知损失。

**📊 数据集**

使用KITTI视觉‑惯性数据集进行训练与评估。

**📈 对比分析**

与VISO2、ORB‑SLAM2、sfmLearner、Depth‑VO‑Feat、GD‑VIO等基线在ATE/RPE指标下进行对比，实验显示在Seq04/07/10上均显著优于基线，并在多种降噪/失效场景下保持最佳性能。

**⚠️ 局限性**

局限性：在极端降噪或全降级（帧丢失+IMU失效）下性能仍显下降；仅依赖相机与IMU，缺少其他传感器的冗余；对长期漂移的补偿仍有提升空间。

---

## 68. LLM-Guided ANN Index Optimization for Human-Object Interaction Retrieval

**arXiv ID:** 2606.05489 | [PDF](https://arxiv.org/pdf/2606.05489v1)

**作者:** Shahrzad Esmat `[一作]` (Iowa State University), Ali Jannesari `[通讯]` (Iowa State University)

**通讯引用:** 1144 | [OpenAlex ID](https://openalex.org/A5079359777)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个阶段感知的LLM驱动的ANN索引优化代理，用于联合优化多阶段检索系统中的耦合参数。

**💡 创新点**

创新点是将历史依赖和阶段划分结合进LLM提示，突破传统独立参数假设，显著提升在耦合空间中的搜索效率。

**🔧 技术方法**

使用大语言模型（如MiniMax-M2.1/OpenRouter）生成参数建议，结合自定义诊断和未尝试值提示，配合SIEVE目标函数。

**📊 数据集**

主要使用HICO-DET、GLDv2和SIFT1M三个检索基准，并在Milvus上验证跨系统迁移。

**📈 对比分析**

与Optuna TPE、GP-BO、VDTuner、随机和网格搜索对比，在HICO-DET上提高33% SIEVE得分，GLDv2约1%提升，SIFT1M 3.6%以内，LLM代理在所有数据集均排名第一。

**⚠️ 局限性**

局限在于对超大规模（百万级）索引的可扩展性仍待验证，且依赖硬件时间昂贵的oracle评估，无法直接处理连续多目标优化。

---

## 69. BRepCLIP: Contrastive Multimodal Pretraining on BRep Primitives for CAD Understanding

**arXiv ID:** 2606.05515 | [PDF](https://arxiv.org/pdf/2606.05515v1)

**作者:** Muhammad Usama `[一作]` (DFKI), Muhammad Zeshan Afzal `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并训练了BRepCLIP，一个将CAD原生BRep结构与文本和图像对齐的对比预训练框架，生成结构感知的全局BRep嵌入，并基于此提出BRepCLIP-Score用于评估文本驱动CAD生成。

**💡 创新点**

创新点包括：①首次在BRep原生结构上进行多模态对比预训练；②采用双分支离散VAE分别对面和边进行token化，避免几何混合；③融合面、边的空间与语义描述，利用Transformer实现全局编码；④将得到的BRep嵌入与冻结的CLIP文本/图像编码器对齐，实现跨模态检索和评估。

**🔧 技术方法**

核心技术：双分支离散VAE（dVAE）面/边token化；Transformer编码器；InfoNCE对比学习；冻结CLIP文本/图像Encoder；BRepCLIP-Score基于BRep嵌入的余弦相似度评估。

**📊 数据集**

使用的数据集：预训练采用CADCap-1M（ABC子集，40万样本）；检索评估基准ABC、CADParser、Automate；零射击分类基准FabWave；生成评估基准ABC 15k + 多个文本到CAD生成器输出。

**📈 对比分析**

与传统点云编码器（PointNet、PointMLP、Point-BERT）以及多模态基线（ULIP、MixCon3D、OpenShape）进行对比；BRepCLIP在文本检索Top‑1提升40%+（ABC）、22%+（CADParser）、23%+（Automate），Zero‑shot分类Top‑1提升至38.6%；生成评估中BRepCLIP-Score与人类/GPT评分更高度相关，优于CLIP Score和Chamfer Distance。

**⚠️ 局限性**

局限性：①固定几何分辨率的token化对细节复杂或高原语计数的模型可能不足；②语义词典仅涵盖面/边的有限类型，未覆盖全部CAD原语及拓扑结构。

---

## 70. State commitment learning: training language models to distinguish computation from memory

**arXiv ID:** 2606.05201 | [PDF](https://arxiv.org/pdf/2606.05201v1)

**作者:** Fei Ding `[一作]` (Alibaba Group), Huiming Yang `[通讯]` (Tsinghua University)

**通讯引用:** 426 | [OpenAlex ID](https://openalex.org/A5101420878)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了状态承诺学习（state commitment learning）与 Counterfactual Erasure RL (CERL)，训练模型在生成隐藏思考后将必要信息提交到答案状态并随后擦除隐藏思考，保持最终答案的准确性。

**💡 创新点**

创新点在于提出 persistent‑state sufficiency 判据、CERL+HSCO 双层强化学习算法以及 Erasure Dependence Protocol，首次让模型在训练阶段明确区分临时计算与持久状态，并验证隐藏思考可安全擦除。

**🔧 技术方法**

采用双层采样与对齐的全/擦除路径评估，使用 GRPO 两层更新隐藏思考与答案状态，加入长度与延迟惩罚以控制答案长度和避免后置计算，从而实现可训练的状态承诺边界。

**📊 数据集**

使用 DeepMath‑103K（数学推理）作为基础训练数据，随后在 AIME 2024/25、ZebraLogic、AutoLogi、GPQA‑Diamond 以及 BFCL‑v3（多轮工具调用）等评测集上进行实验。

**📈 对比分析**

与基线 Qwen3‑8B、Fixed‑ratio erasure、Length‑penalty RL、TokenSkip、Halo、Long‑answer SFT、Correctness‑only RL 等方法对比；CERL full 在所有任务上获得最高准确率，同时 ASG 低、ESR/MSG 高，表明隐藏思考可以被擦除而不影响性能。

**⚠️ 局限性**

局限性：方法仅适用于明确的隐藏思考/答案接口，未覆盖一般记忆写入或多状态管理；依赖对齐的训练数据；对大规模模型的可扩展性仍待进一步验证。

---

## 71. How Far Did They Go? The Persuasive Tactics of Covert LLM Agents in a Discontinued Field Experiment

**arXiv ID:** 2606.05256 | [PDF](https://arxiv.org/pdf/2606.05256v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 72. CRESS: Quantifying Vulnerabilities of Attack Scenarios in Hardware Reverse Engineering

**arXiv ID:** 2606.05459 | [PDF](https://arxiv.org/pdf/2606.05459v1)

**作者:** Alexander Hepp `[一作]`, Georg Sigl `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于硬件逆向工程（RE）攻击场景，构建了可量化的 CRESS（Common Reverse Engineering Scoring System）评分体系，对原先的定性框架进行数值化扩展，并通过专家访谈得出属性权重和数值，最终得到完整的评分公式。

**💡 创新点**

创新点在于：①提出了专门针对 RE 相关攻击的量化评分公式，并通过专家访谈为每个属性赋予权重和数值；②采用加权几何平均与逻辑函数组合的方式，使评分更符合概率直观；③与传统 CVSS 对比，CRESS 能更细致地刻画硬件攻击的可执行性与影响，显著提高表达度。

**🔧 技术方法**

技术方法包括：结构化专家访谈（利用 Conceptboard 进行可视化评分）、统计分析（均值/中位数计算权重与数值）、自定义加权公式（爆发性可执行性、影响子得分）以及最终的混合权重与 logistic 映射，形成完整评分模型。

**📊 数据集**

数据集主要是 19 位硬件 RE 与攻击专家的访谈结果（原 21 次），以及六个典型攻击案例（故障注入、RISC‑V 侧信道、掺杂式硬件木马、IP 盗版、漏洞检测、NVM 读取）用于验证与对比。

**📈 对比分析**

评估方法为：将 CRESS 与 CVSS v3.1 在同一批案例上同时打分，观察得分分布、差异与一致性。实验显示 CRESS 的得分分布呈正态、均值≈5，且在所有案例中能够区分可执行性与影响的细微差别，优于 CVSS 在硬件攻击场景下的表达力。

**⚠️ 局限性**

局限性包括：专家样本规模有限（19 位），可能未覆盖全部 RE 场景；remediation 仅按是否存在二元化简化，未细化不同策略；评分模型基于专家主观评估，需进一步社区验证与迭代。

---

## 73. Field Validation of a Multi-Resolution ConvLSTM Framework for Retaining Wall Deformation Prediction

**arXiv ID:** 2606.05556 | [PDF](https://arxiv.org/pdf/2606.05556v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 74. Efficient Punctuation Restoration via Weighted Lookahead Scoring Method for Streaming ASR Systems

**arXiv ID:** 2606.05179 | [PDF](https://arxiv.org/pdf/2606.05179v1)

**作者:** Sungmook Woo `[一作]` (Korea University), Chanwoo Kim `[通讯]` (Korea University)

**通讯引用:** 1493 | [OpenAlex ID](https://openalex.org/A5100684423)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于 LLM 的非自回归加权前瞻评分方法，用于流式 ASR 的标点恢复。

**💡 创新点**

创新点在于通过有限未来上下文内对插入/不插入标点进行分数比较，避免生成式漂移与对齐错误，实现低延迟在线决策。

**🔧 技术方法**

使用 Llama‑3.2‑1B 作为评分器，结合 α 加权的前瞻似然与先验，阈值 τ 进行校准，可无微调或通过 LoRA 微调。

**📊 数据集**

采用英文 IWSLT 2017 语料构造的无标点/有标点句子对作为数据集。

**📈 对比分析**

与 prompt‑generation（Llama‑3.2‑1B‑Instruct）和细调 ELECTRA baseline 在 K=2 的 lookahead 下对比，未微调评分实现 4‑类宏 F1 0.893，微调后 0.937，显著优于基线（0.566 / 0.913）。

**⚠️ 局限性**

局限性：未评估真实 ASR 噪声下的鲁棒性；未测量系统级延迟和内存占用；仅在英文 IWSLT 2017 上验证。

---

## 75. Should Demand Models Incorporate Competitor Prices? Oblivious Learning and Algorithmic Collusion

**arXiv ID:** 2606.05363 | [PDF](https://arxiv.org/pdf/2606.05363v1)

**作者:** Yuhang Wu `[一作]` (Columbia Business School), Assaf Zeevi `[通讯]` (Columbia Business School)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究在竞争性市场中，卖家在学习需求时是否需要显式考虑竞争对手价格，分析了忽略竞争价格（盲目学习）与完整信息建模的动态定价行为及其对协同与价格收敛的影响。

**💡 创新点**

提出了盲目学习在竞争中的“螺旋上升”探索率、价格的短期协同波动以及对长期价格收敛的理论分析，首次证明盲目建模无法稳健维持协同，而完整信息模型在所有竞争者均学习时形成唯一的纯策略纳什均衡。

**🔧 技术方法**

使用迭代最小二乘估计、加性探索扰动、线性需求模型、随机噪声、ODE均值动力学、渐近小增益分析、误差收敛证明以及模拟实验验证。

**📊 数据集**

采用合成的线性需求环境（带噪声），生成多卖家随机价格与需求数据，未使用真实市场数据。

**📈 对比分析**

通过对比盲目与完整信息两种建模策略，计算盈余捕获比和累计收益（regret），在模拟中显示完整信息模型在竞争中收敛到纳什均衡且无持续探索税，而盲目学习则需要更高探索并产生短期协同波动，但最终无法持续。总体性能：完整信息获得最高收益，盲目学习表现最低；两种探索策略的收益差距随探索率和市场规模变化。

**⚠️ 局限性**

仅考虑二元建模选择，未探讨部分信息或更复杂的预测规则；仅使用线性需求模型；探索设计为零均值扰动，未研究其他探索策略；对实际市场数据验证不足；对鲁棒性（对需求噪声、模型假设变化）的分析有限。

---

## 76. Policy-Compliant Cloud Storage Systems

**arXiv ID:** 2606.05423 | [PDF](https://arxiv.org/pdf/2606.05423v1)

**作者:** Dimitrios Stavrakakis `[一作]` (Technical University of Munich), Pramod Bhatotia `[通讯]` (Technical University of Munich)

**通讯引用:** 6040 | [OpenAlex ID](https://openalex.org/A5002550391)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一款可信中间件，在不改动 KV 存储引擎的前提下，利用 Confidential Virtual Machine 对 KV 操作进行 GDPR 合规性校验、元数据管理与可验证审计日志生成。

**💡 创新点**

创新点包括：①声明式合规语言与编译器，将法律条款转化为可执行规则；②在 CVM 内安全执行并维护完整元数据与日志；③使用紧凑编码与专用索引显著降低性能开销；④实现透明代理，无需改动应用或后端。

**🔧 技术方法**

采用的技术包括 Confidential Virtual Machine（AMD SEV‑SNP/Intel TDX/ARM CCA）进行隔离与远程证明；声明式合规语言与编译器；紧凑元数据编码与多种索引（哈希、B+树、倒排）；异步压缩日志与可信计数器；AES‑GCM 加密。

**📊 数据集**

使用 YCSB（A–D、F）工作负载模拟 KV 操作，并使用 GDPRBench 提供的 Controller、Customer、Processor 三类自定义工作负载，数据量为 1 M 条操作、100 K KV 对。

**📈 对比分析**

通过与本地原生 Redis/RocksDB、仅加密、仅 CVM 等变体对比；测量吞吐率、延迟、存储占用；实验结果显示：在 CVM 环境下额外开销约 28–32%，整体吞吐率约 61%；启用元数据索引后 13–182 倍速度提升；日志开销 <2%；元数据占用 <20%。

**⚠️ 局限性**

局限性包括：①依赖硬件支持 CVM；②对 rollback 攻击无保护；③仅支持 KV 语义，无法处理复杂事务或 SQL 查询；④日志与元数据仍占用额外存储；⑤受 KV 引擎单线程特性影响，某些工作负载下性能受限。

---

## 77. Predict and Reconstruct: Joint Objectives for Self-Supervised Language Representation Learning

**arXiv ID:** 2606.05173 | [PDF](https://arxiv.org/pdf/2606.05173v1)

**作者:** Aimen Boukhari `[一作]` `[通讯]` (Ecole Nationale Supérieure d'Informatique), Aimen Boukhari (Ecole Nationale Supérieure d'Informatique)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种混合预训练目标，将JEPA风格的潜在空间预测与MLM重构联合到同一编码器中，以改进文本表示。

**💡 创新点**

引入可学习的权重 λ 实时平衡两种损失，证明混合目标能在保持线性探针性能的同时显著提升嵌入均匀性、谱丰富度和语义/词汇平衡。

**🔧 技术方法**

采用 Transformer 编码器、EMA 目标编码器、可学习的预测器与 token 回归头、均匀性/对齐度量、谱熵/有效秩评估、四种池化策略、线性探针与探针分类任务。

**📊 数据集**

以英文维基百科为预训练语料，评估 GLUE 五个任务（SST‑2、MRPC、MNLI、CoLA、STS‑B）进行下游线性探针。

**📈 对比分析**

通过相同模型架构、训练预算和超参，对比混合模型与纯 MLM 基线，在所有 GLUE 任务上线性探针准确率相近，但混合模型在均匀性、谱熵、有效秩等几何指标上明显优于基线。

**⚠️ 局限性**

预训练仅有限 3 个 epoch、规模小、只评估线性探针，未验证在更大规模或非线性探针、检索任务中的实际性能提升；λ 收敛慢，未探索更高效的调度或替代损失。

---

## 78. What Objects Enable, Not What They Are: Functional Latent Spaces for Affordance Reasoning

**arXiv ID:** 2606.05533 | [PDF](https://arxiv.org/pdf/2606.05533v1)

**作者:** Rohan Siva `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**通讯引用:** 10389 | [OpenAlex ID](https://openalex.org/A5068441112)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个基于功能潜在空间的机器人功能推理框架，能够从视觉观测直接推断物体功能（如可移动、可支撑等），并支持在线功能发现与不确定性管理。

**💡 创新点**

创新点在于：①将视觉与文本嵌入映射到以功能为轴的共享潜在空间；②利用投影距离进行快速且高精度功能推理；③通过不确定性校准触发功能发现，实现少样本增添新功能；④实现 100 倍更快推理。

**🔧 技术方法**

使用了预训练的 CLIP 视觉‑语言模型、对比损失微调、投影到功能轴、等距校准（Isotonic Regression）、VLM 触发功能发现与标注。

**📊 数据集**

主要数据集包括自建的功能标注图像集（约数千张），并利用公开的 CLIP 训练集与 GPT‑5.4 进行 VLM 辅助标注；实验亦用公开的机器人规划场景图片。

**📈 对比分析**

与零样本 CLIP、BLIP、GPT‑5 系列等基线相比，A4D 在已知功能上达到 94% 以上准确率，未见功能从 70% 提升至 92% 只需 16 条样本；推理时间 22 ms，远快于 GPT‑5 的 2‑3 s；不确定性触发 VLM 的方法可在不超过 20% 调用率下保持 93% 的整体准确率。

**⚠️ 局限性**

局限性包括：①功能推理与高层规划的集成仍为后处理，缺乏统一评估；②功能轴在训练后固定，无法在线根据交互反馈自适应；③不确定性校准为经验式，没有严格概率保证。

---

## 79. Bitcoin After Block Rewards

**arXiv ID:** 2606.05503 | [PDF](https://arxiv.org/pdf/2606.05503v1)

**作者:** Junhyuk Lee `[一作]` `[通讯]` (Texas A&M University), Junhyuk Lee (Texas A&M University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究比特币区块奖励消失后矿工偏离行为的条件，并提出基费、费底和自适应块大小机制来抑制偏离，确保网络安全。

**💡 创新点**

① 将矿工偏离阈值 G_t 与块奖励、交易费等收益直接关联，提出可测的偏离阈值公式；② 在实际数据中用审计得分验证偏离的现实表现；③ 设计三种协议级机制组合，首次在模拟中证明其能将偏离率维持在 BFT 限额以下。

**🔧 技术方法**

马尔可夫决策过程（MDP）建模、偏离收益阈值推导、基于网络延迟的孤块概率模型、交易费与 MEV 合成模型，以及数值模拟与实验比较。

**📊 数据集**

2024 年 790,000–890,000 号块的 100,001 条区块数据（区块大小、交易费、利用率等）来自 blockchain.com，审计得分和块信息来自 mempool.space，MEV 采用以太坊数据估计后转化为 Bitcoin 环境。

**📈 对比分析**

通过对比六种策略（无机制、单独基费、单独费底、单独自适应块大小、组合基费+费底、组合三者）在零奖励场景下的偏离率；结果显示基费+费底组合将偏离率压至 35% 以内，低于 50% BFT 阈值；单独基费或费底可略高，完全无机制时偏离率超过 70%。

**⚠️ 局限性**

① 偏离收益阈值中保留的取决于孤块概率的 ϕ(w) 公式尚未完全解析；② MEV 采用以太坊估算，可能低估比特币真实私有收益；③ 模拟假设所有矿工成本相同、网络延迟为常数，未考虑矿池治理等现实复杂性；④ 只关注短期偏离率，未深入评估长期稳定性与网络经济学交互。

---

## 80. GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data

**arXiv ID:** 2606.05441 | [PDF](https://arxiv.org/pdf/2606.05441v1)

**作者:** Al Zadid Sultan Bin Habib `[一作]` (West Virginia University), Donald A. Adjeroh `[通讯]` (West Virginia University)

**通讯引用:** 4624 | [OpenAlex ID](https://openalex.org/A5085141731)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在高维低样本（HDLSS）表格预测场景下，提出了一种基于特征排序与压缩的轻量级基准模型GOTabPFN，能够在不重新训练大型TabPFN骨干的前提下实现高精度预测。

**💡 创新点**

创新点包括：①将特征排序问题建模为加权最小线性排列（MinLA）并提出GO-LR算法实现有效的全局排序；②设计受脑皮层分单位结构启发的NSC压缩单元，将排序后的相邻特征聚合为可解释的元特征，从而在紧凑的令牌预算内保留局部相关性；③通过冻结TabPFN-2.5头部实现端到端可复现的预测管道。

**🔧 技术方法**

技术方法：图导向特征排序（GO-LR）采用最近邻TSP路径初始化并在MinLA目标下进行局部细化；NSC利用自适应分段、PCA主成分投影或共享池化网络生成元特征；最终将压缩后的令牌序列输入TabPFN-2.5进行推理。

**📊 数据集**

数据集：8个生物医学HDLSS基准（Colon、Lung、GLI‑85、SMK_CAN_187、ALLAML、Prostate‑GE、Arcene、TOX‑171）以及8个跨域高维数据集（ORL、BAS、REL、PCM、CCY、CIFAR‑10 embeddings、DrivFace‑Regression、DrivFace‑Classification），全部采用5×5嵌套交叉验证。

**📈 对比分析**

对比方法涵盖55个基线（传统ML/GBDT、深度表格模型、TabPFN系列、TANDEM、TabDPT、TabICL、BETA、TuneTables、ProtoGate等）。GOTabPFN在所有8个HDLSS任务中取得最高平均排名（1.00±0.00），在7/8跨域任务中名列前茅，显著优于现有TabPFN变体及其他基线，尤其在噪声大、难度高的任务上提升幅度更大。

**⚠️ 局限性**

局限性：受冻结的TabPFN-2.5骨干约束（最多10类、50k样本），在样本规模较大或高维度不极端的场景下效果可能不如针对性更强的方法；GO-LR+NSC在大样本情况下的图构建和排序步骤会显著增加运行时间；总体适用于HDLSS低样本、高维度的特定应用。

---

## 81. VASO: Formally Verifiable Self-Evolving Skills for Physical AI Agents

**arXiv ID:** 2606.05395 | [PDF](https://arxiv.org/pdf/2606.05395v1)

**作者:** Yunhao Yang `[一作]` (University of Texas at Austin), Ufuk Topcu `[通讯]` (University of Texas at Austin)

**通讯引用:** 10389 | [OpenAlex ID](https://openalex.org/A5068441112)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种可验证的自我进化机器人技能框架（Verification‑Guided Automated Skill Optimization），使得由大型语言模型（LLM）生成的技能可以在执行前通过形式化模型检查进行验证，并根据验证反例自动更新技能契约；

**💡 创新点**

创新点包括：①将形式化契约与计划器接口耦合的可验证技能表示；②将模型检查器产生的反例转化为文本梯度，直接反馈到技能契约而非仅用于计划拒绝；③在真实机器人上仅使用不到100个优化样本即可达到97.2%形式化规范合规率；

**🔧 技术方法**

采用的技术包括：LLM技能生成器、LLM规划器、NuSMV等模型检查器、LTL 时序逻辑规范、自动化命题对齐函数、文本梯度优化（Prompt‑gradient）以及基于符号转移系统的计划编译；

**📊 数据集**

使用自定义的11条时序逻辑规范与400条生成计划，实验平台为 Clearpath Jackal 地面机器人和 PX4 四旋翼无人机；

**📈 对比分析**

与多种基线（零射规划器、提示优化、RLVF、DSPy、LLM+P等）进行对比，Ours 在 Jackal 与 PX4 上的安全得分分别为96.6%/85.3%，任务完成率为85.3%/86.5%，比 RLVF 的94.3%/82.5% 更优，且仅需100个样本、训练时间大幅降低；

**⚠️ 局限性**

局限性：①命题对齐函数由 LLM 自动生成且未经过形式化验证，可能引入误差；②框架仅适用于顺序执行的单一技能，无法处理并行或交错的多技能执行；③依赖 LLM 生成的文本梯度，若反例不完整或模型偏差较大，可能导致优化不收敛。

---

## 82. BMCR: Adaptive Backbone Module Composition via Reinforcement Learning for Remote Sensing Object Detection

**arXiv ID:** 2606.05586 | [PDF](https://arxiv.org/pdf/2606.05586v1)

**作者:** Wenlin Liu `[一作]` (National University of Defense Technology), Ping Zhong `[通讯]` (National University of Defense Technology)

**通讯引用:** 3324 | [OpenAlex ID](https://openalex.org/A5002771011)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于强化学习的BMCR框架，实现对遥感目标检测的输入自适应骨干网络组合。

**💡 创新点**

创新点在于将CNN与ViT模块拆分为可复用工具箱，并通过OT接口实现跨架构特征对齐，以及使用AMCO算法实现路由策略与模块参数的协同训练。

**🔧 技术方法**

采用强化学习（PPO）决策、OT优化、可变结构模块组合、动态FPN聚合、辅助检测头等技术。

**📊 数据集**

使用DOTA‑v1.0/v1.5、DIOR‑R、FAIR1M等大规模遥感目标检测数据集。

**📈 对比分析**

与多种静态与动态骨干（ResNet、Swin、ViTAE等）以及最新检测方法对比，BMCR在DOTA、DIOR‑R、FAIR1M上分别提升约1–2个百分点mAP，同时保持接近的推理速度（≈63.9 FPS）。

**⚠️ 局限性**

局限在于需预先构建模块工具箱、训练成本较高（≈48 h）且对模块选择与接口设计敏感，缺乏对不同部署资源约束的可控性。

---

## 83. Full-Field Calibration of Coupled Thermomechanical Material Models at Finite Strain

**arXiv ID:** 2606.05465 | [PDF](https://arxiv.org/pdf/2606.05465v1)

**作者:** L. River Spencer `[一作]` (University of Texas at Austin), Jan N. Fuhg `[通讯]` (University of Texas at Austin)

**通讯引用:** 1537 | [OpenAlex ID](https://openalex.org/A5058627053)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于全场表面位移、温度和反作用力数据的有限变形热-弹性材料逆识别框架，能够从仅表面观测中恢复耦合热力学参数。

**💡 创新点**

创新点在于：①利用从 Helmholtz 自由能导出的热-弹性耦合项构建热力学一致的连续模型；②采用自动微分生成伴随敏感度，实现高维非线性瞬态系统的梯度优化；③在同一框架内同时处理 DIC、热像仪与力学全场数据，克服传统单一数据来源导致的可辨识性不足。

**🔧 技术方法**

技术手段包括：有限元混合近压缩性热弹性耦合（$u,p,	heta$ 三场），自动微分与伴随方程求解，L‑BFGS‑B/梯度下降优化，热像仪与 DIC 的全场测量融合，后向欧拉时间离散与热弹性源项的历史投影。

**📊 数据集**

数据集：①合成实验（均温预加热+单次/多次杆接触的温度与力学场）；②真实实验——C57BL/6 小鼠皮肤在双轴机械试验与热辐射（红外摄像）条件下获得的表面温度和边界力记录。

**📈 对比分析**

对比方法：在合成数据中通过误差对比验证参数恢复，目标函数下降3–4个数量级后收敛；在实验数据中将预测力学曲线与测量曲线（$P_x,P_y$）对比，匹配误差约为5–10%，表明模型能有效复现热膨胀/收缩对力学响应的影响。

**⚠️ 局限性**

局限性：未考虑测量噪声、粘弹性或热传递参数的可辨识性，模型仍是经验化的宏观描述，缺乏对微观机制（如胶原降解、损伤）的显式建模，且对参数可辨识性的评估仅在理想合成情境下完成。

---

## 84. Ten Headache Specialists versus Artificial Intelligence for Clinical Literature Summarization: A Critical Evaluation and Comparison

**arXiv ID:** 2606.05436 | [PDF](https://arxiv.org/pdf/2606.05436v1)

**作者:** Alejandro Lozano `[一作]` (Stanford University), Chia-Chun Chiang `[通讯]` (Mayo Clinic)

**通讯引用:** 1943 | [OpenAlex ID](https://openalex.org/A5054188179)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究对头痛学专家与三大LLM（Sonnet、GPT-4o、Llama 3.1）生成的文献摘要进行系统评估，比较两者在正确性、完整性、简洁性、实用性及偏好等维度上的表现。

**💡 创新点**

创新点在于将检索增强语言模型与专家评估相结合，揭示专家所重视的非量化特征（如结构流畅、参考质量、临床可操作性），并发现LLM与人类摘要在这些维度的显著差距。

**🔧 技术方法**

采用检索增强（RAG）代理式链式LLM框架，包含查询生成、检索、相关性分类、摘要和合成五步，并使用Sonnet、GPT-4o、Llama 3.1三种LLM进行摘要生成。

**📊 数据集**

使用10名头痛专家共同制定的13个临床问题并撰写200-300字的专业摘要作为基准，LLM通过PubMed检索相关文献并生成摘要，共产生200份待评估摘要。

**📈 对比分析**

通过10名专家在每题4份摘要上进行评分与偏好排序；结果显示专家摘要最高评分，Sonnet仅次于专家，GPT-4o与Llama在大多数维度表现显著落后，专家识别率为64%。

**⚠️ 局限性**

局限性包括样本量有限（10题、10评审），仅针对头痛学领域，评估耗时且可能受未来LLM性能提升影响。

---

## 85. Assessing the Geographic Diversity of AI's Platial Representations in Image Generation

**arXiv ID:** 2606.05188 | [PDF](https://arxiv.org/pdf/2606.05188v1)

**作者:** Zilong Liu `[一作]` (University of Vienna), Mina Karimi `[通讯]` (University of Vienna)

**通讯引用:** 37 | [OpenAlex ID](https://openalex.org/A5048584886)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了生成式 AI 图像生成中的地理多样性，选取维也纳作为案例，针对 GPT 与 DALL·E 系列模型在提示修订和图像生成阶段的多模态输出进行评估，并提出基于 Hill 数和 Leinster‑Cobbold 数的测量方法。

**💡 创新点**

创新点包括①首次将地理多样性测量扩展到图像生成；②引入基于知识图谱的地点相似性权重的 Leinster‑Cobbold 数；③揭示最新模型并不一定拥有更高多样性，强调相似性考虑的重要性。

**🔧 技术方法**

技术手段包括信息论的 Hill 数与 Leinster‑Cobbold 数，利用 Wikidata Rada 距离计算地点类型相似性，结合 OpenAI GPT‑4o、GPT Image 系列与 DALL·E 系列的多代理图像生成管道。

**📊 数据集**

使用自建实验数据集：在维也纳背景下进行 30 次会话，生成图像与修订提示，手工标注主地标并映射到 Wikidata 实体，形成包含图像、提示、标注及相似性矩阵的数据集合。

**📈 对比分析**

通过对多模型、多阶段（提示修订 vs 图像生成）的多模态实验，绘制 Hill 数与 Leinster‑Cobbold 数在不同阶 q 的多样性曲线进行比较；结果显示最新模型并非最高多样性，Leinster‑Cobbold 数显著低于 Hill 数，表明模型存在地理多样性缺失。

**⚠️ 局限性**

局限性包括仅聚焦单一城市（维也纳）和有限模型；主标注手工完成，可能存在主观偏差；未使用语义分割或更细粒度图像分析；相似性仅基于 Wikidata 路径，未考虑空间距离等因素。

---

## 86. Where's the Structure? A Systematic Literature Review of Empirical Research on Human-AI Collaboration and Hybrid Intelligence for Learning

**arXiv ID:** 2606.05222 | [PDF](https://arxiv.org/pdf/2606.05222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 87. Minimizing the Hidden Cost of Scales: Graph-Guided Ultra-Low-Bit Quantization for Large Language Models

**arXiv ID:** 2606.05429 | [PDF](https://arxiv.org/pdf/2606.05429v1)

**作者:** Rayyan Abdalla `[一作]`, Dinesh Manocha `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SAGE-PTQ，一种针对LLM的超低位后训练量化框架，能够在不需要再训练的前提下，将权重压缩至平均1.03比特，并极大降低缩放开销；

**💡 创新点**

创新点包括：①利用权重分布统计动态区分显著与非显著权重；②采用图引导的稀疏KNN+谱聚类自适应估计非显著权重组数；③双模式量化，显著权重多比特精度、非显著权重二值化，并配合单通道与单标量缩放；④自适应显著阈值优化和高效位图查找机制；

**🔧 技术方法**

使用的技术包括权重分布阈值筛选、图构建与谱聚类、显著/非显著双模式量化、Silhouette分数优化、Brent方法求阈值、位图查找表；

**📊 数据集**

在LLaMA、LLaMA‑2、LLaMA‑3、OPT、Vicuna、DeepSeek、Qwen2.5/3等1.3B~70B模型上，使用WikiText‑2、C4以及六个零样本NLU基准（PIQA、BoolQ、OpenBookQA、Winogrande、ARC‑c、HellaSwag）进行评测；

**📈 对比分析**

与BiLLM、PB‑LLM、GPTQ、AWQ等主流PTQ方法比较，SAGE‑PTQ在相同或更小的存储/查找位数下，平均权重比特为1.03、缩放比特仅0.004，WikiText‑2困惑度显著降低（LLaMA‑3‑8B仅6.74 vs BiLLM 55.8），在70B模型上实现1.5×解码加速、GPU内存占用下降50%+，且零样本准确率与FP16接近；

**⚠️ 局限性**

局限性在于仅对权重进行量化，未覆盖激活量化，且对超参数（如图构建采样、阈值上限）仍需手动设置，未来需扩展至权重-激活联合量化及对MoE模型的适配。

---

## 88. Three-Dimensional Retinal Microvasculature Restoration in OCT Angiography

**arXiv ID:** 2606.05375 | [PDF](https://arxiv.org/pdf/2606.05375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 89. Evidence-Guided Neural Architecture Selection under Uncertainty for Subject-Specific Blood Glucose Forecasting

**arXiv ID:** 2606.05373 | [PDF](https://arxiv.org/pdf/2606.05373v1)

**作者:** Md Azharul Islam `[一作]` (University at Buffalo), Danial Faghihi `[通讯]` (University at Buffalo)

**通讯引用:** 1149 | [OpenAlex ID](https://openalex.org/A5003194178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了EVIDENT框架，用于在有限、噪声、异质时序数据下进行神经网络架构选择，结合贝叶斯训练、证据评分和任务特定不确定性验证，定位最简可靠模型；

**💡 创新点**

将贝叶斯模型证据作为架构排名依据，并与预测不确定性验证分离；采用按容量层级逐级搜索，先筛选高可信度架构，再通过NLPD等任务特定指标验证；支持可信的加权集成，兼顾模型复杂度与预测可靠性；

**🔧 技术方法**

使用贝叶斯时间卷积网络（TCN）实现多通道血糖预测；利用变分推断（均值场Gaussian）与拉普拉斯近似估计后验与证据；通过证据对候选架构进行排序；采用NLPD、RMSE、CV和Parkes误差网格等指标进行任务特定验证；

**📊 数据集**

使用Bergman最小模型生成单患者时序数据，以及UVA/Padova T1D模拟器生成10名成人在硅模拟的持续血糖监测、餐食和胰岛素输入数据；采用5折交叉验证进行架构搜索与验证；

**📈 对比分析**

与等价计算预算的随机搜索基线对比；EVIDENT选出的中等容量模型在未见患者的NLPD、RMSE、CV和Parkes误差网格上均优于随机搜索；在保持模型容量较小的情况下，实现更一致、可靠的预测；

**⚠️ 局限性**

仅在离散可行架构池内探索；使用近似后验（均值场VI）和拉普拉斯估计，可能影响证据与排名准确性；实验仅基于仿真数据，缺乏真实临床数据的复杂性；若要扩展到更大或连续架构空间，需要集成更高级的搜索方法与更丰富的后验表示。

---

## 90. Flash-WAM: Modality-Aware Distillation for World Action Models

**arXiv ID:** 2606.05254 | [PDF](https://arxiv.org/pdf/2606.05254v1)

**作者:** Arman Akbari `[一作]` (Northeastern University), Yanzhi Wang `[通讯]` (Northeastern University)

**通讯引用:** 15997 | [OpenAlex ID](https://openalex.org/A5100651384)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Flash-WAM，一种面向联合视频‑动作扩散模型的步数蒸馏框架，能在保持任务成功率的前提下将每个模态的去噪步骤压缩到单步，实现实时推理。

**💡 创新点**

创新点在于揭示标准一致性蒸馏在不同噪声调度下失效的结构性瓶颈，并设计模态感知一致性函数：动作流使用线性梯度缩放参数化，视频流使用方差保持参数化，分别匹配低噪声和高噪声区的梯度信号。

**🔧 技术方法**

结合一致性蒸馏、流匹配、SNR‑shifted noise schedule、共享 Transformer 骨干和自回归生成等技术，实现视频与动作的单步推理。

**📊 数据集**

主要使用 LingBot‑VA 训练模型，并在 RoboTwin 2.0（Clean/Randomized）、LIBERO（Spatial、Object、Goal、Long‑horizon）以及 Unitree G1 机器人实测任务上进行评测。

**📈 对比分析**

与原 LingBot‑VA、VLA 基线、DMD2、Video‑only LCM 以及 Naive Joint LCM 等方法对比；在 RoboTwin 1v/2a 下实现 85.5% 成功率、速度提升 23×；在 1v/1a 下 81.4%；在 LIBERO 1v/2a 下 95.7%，速度提升 13.7×；在实测机器人上平均 60% 成功率，显著优于无蒸馏或仅视频蒸馏。

**⚠️ 局限性**

限制包括仅针对共享 Transformer 结构的 WAM 模型，无法直接迁移至其他架构；动作流的线性一致性函数在极低噪声下可能梯度不足；实验仍需较大 GPU 资源；对不同任务的泛化能力仍待进一步验证。

---

## 91. A Comprehensive Survey on Semantic Communication in Non-Terrestrial Networks: Architectures, Methodologies, and Challenges

**arXiv ID:** 2606.05216 | [PDF](https://arxiv.org/pdf/2606.05216v1)

**作者:** Loc X. Nguyen `[一作]` (Kyung Hee University), Choong Seon Hong `[通讯]` (Kyung Hee University)

**通讯引用:** 23203 | [OpenAlex ID](https://openalex.org/A5034052371)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了语义通信在非地面网络（NTN）中的架构、方法与挑战，针对卫星、UAV/HAPS与SAGIN等平台进行深度剖析，提出了基于NTN限制的语义通信设计映射、三维分类法以及前沿研究路线图。

**💡 创新点**

创新点在于将NTN的关键限制与语义通信特性一一对应，构建了三轴（平台、方法、支持技术）分类框架，并在各平台上展开深度案例分析，首次完整梳理了 D‑JSCC、Theory‑of‑Mind 与 Generative‑AI 三种语义通信范式在NTN中的应用与挑战，同时识别并聚焦未来研究热点。

**🔧 技术方法**

采用系统文献综述与对比分析技术，结合三维分类法和案例深度剖析，综述了 D‑JSCC、图谱推理、生成式 AI 等主要语义通信技术，以及相应的调度、路由、资源分配与联邦学习等支持技术。

**📊 数据集**

本论文为综述性工作，无直接实验数据集，主要引用并归纳现有研究中的数据集与实验结果。

**📈 对比分析**

通过对比表格与案例讨论，对先前的综述与研究进行了横向对比，阐述了各类方法在吞吐量、压缩效率、延迟与能耗等指标上的优劣，并指出不同语义通信范式在NTN场景下的性能差距。

**⚠️ 局限性**

局限性在于：①综述基于已公开文献，缺乏统一的实验平台和评测基准；②NTN场景快速演进，部分最新研究尚未纳入；③不同语义通信方法的评价指标不统一，导致跨研究比较困难；④缺乏大规模真实部署案例验证。

---

## 92. A Motivational Architecture for Conversational AGI

**arXiv ID:** 2606.05411 | [PDF](https://arxiv.org/pdf/2606.05411v1)

**作者:** Anna Mikeda `[一作]` (Glass Umbrella), Ben Goertzel `[通讯]` (SingularityNet)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向对话式 AGI 的动机架构，将 OpenPsi 的调节层与 MetaMo 的全局动机框架结合，形成十阶段动机处理管道，区分预行动情感和后行动情绪，并提供双流程决策；

**💡 创新点**

创新点在于将身体维持转化为对话性“家居平衡”需求，构建可检查的情感与决策分离机制，并实现对 CompanionAgent 与 ResearchAgent 的多目标调节；

**🔧 技术方法**

采用 MetaMo 的目标+调节状态、OpenPsi 的调节动力学、模块化执行基底、符号记忆与结构化抽象动作；

**📊 数据集**

未使用传统公开数据集，而是通过自建对话情境与模拟用户模型来评估动机循环；

**📈 对比分析**

通过与传统基于提示的 LLM 对话代理对比，展示该架构在需求调节、情感可解释性与行动多样性上的提升，但缺少量化实验数据；

**⚠️ 局限性**

局限在于调节符号的完整性、候选生成的具体实现、评价指标不足，以及对长时序自我修正的理论支持尚未完成。

---

## 93. Balancing Image Compression and Generation with Bootstrapped Tokenization

**arXiv ID:** 2606.05552 | [PDF](https://arxiv.org/pdf/2606.05552v1)

**作者:** Haozhe Chi `[一作]` (Peking University), Yadong Mu `[通讯]` (Peking University)

**通讯引用:** 9092 | [OpenAlex ID](https://openalex.org/A5028877572)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SelfBootTok，一种将图像分解为全局与局部两组 1D 令牌并通过自监督自引导方式从全局令牌预测细粒度局部信息的图像令牌器

**💡 创新点**

核心创新是全局‑局部分解与自引导学习：将局部细节的生成工作迁移到令牌化器中，显著减少生成器负担；同时采用 2D→1D 最优传输对齐及多级局部对齐，打破令牌间冗余，实现更紧凑、高效的表示

**🔧 技术方法**

技术包括 ViT 编码器、软向量量化、局部 1D 线性投影与 2D 变压器自回归对齐、基于预训练视觉编码器的对齐损失、最优传输（Sinkhorn）对齐 2D 与 1D 令牌、以及并行的局部对齐器与生成器训练

**📊 数据集**

主要数据集为 ImageNet‑1K，训练与评估均在 256×256 分辨率下进行，使用公开预训练的 DINOv2、I‑JEPA、SigLIP 等视觉编码器作为对齐参考

**📈 对比分析**

与现有 1D 令牌器（HieraTok、GigaTok、SoftVQ 等）以及自回归模型（MAR‑H、ViTok）相比，SelfBootTok 在仅 64 令牌的情况下实现 gFID 1.56、rFID 0.66、PSNR 与 SSIM 等指标均优于同等令牌量基线；通过并行训练局部对齐器与生成器，整体训练成本下降约 40%，训练时间缩短 54%

**⚠️ 局限性**

限制：目前验证集中于 ImageNet，模型对不同分辨率或更大数据集的泛化尚未全面评估；全局令牌数量固定后无法进一步压缩；以及自引导对齐对局部细节的恢复仍受预训练编码器特征表达的影响

---

## 94. PJ-RoPE: A Fourier-Jet-Affine Position Space for Relative Attention

**arXiv ID:** 2606.05345 | [PDF](https://arxiv.org/pdf/2606.05345v1)

**作者:** Yaobo Zhang `[一作]` `[通讯]`, Yaobo Zhang

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于光谱、Jet 与 sector 切片三维空间的 Adaptive PJ‑RoPE 表示方法，能够在 3D 领域中自适应融合多种特征。

**💡 创新点**

创新点在于将光谱基、傅里叶‑Jet、仿射和 LC 四种子空间映射到同一 3D 结构中，并通过动态门控 (Δ²) 来分配各子空间的重要性。

**🔧 技术方法**

利用光谱编码、傅里叶‑Jet 变换、仿射变换、局部坐标（LC）以及门控网络实现特征融合。

**📊 数据集**

文中未具体给出实验数据集，推测可能使用常见 3D 点云数据集如 ModelNet40 或 ShapeNet 进行验证。

**📈 对比分析**

未提供对比实验，若与传统 RoPE 或 PointNet 等方法对比，可预期在表示丰富性和参数效率方面有所提升。

**⚠️ 局限性**

缺乏实验验证、门控机制复杂度高、对大规模数据的扩展性和训练稳定性未作讨论。

---

## 95. From Scoring to Explanations: Evaluating SHAP and LLM Rationales for Rubric-based Teaching Quality Assessment

**arXiv ID:** 2606.05180 | [PDF](https://arxiv.org/pdf/2606.05180v1)

**作者:** Ivo Bueno `[一作]` (Technical University of Munich), Enkelejda Kasneci `[通讯]` (Technical University of Munich)

**通讯引用:** 12133 | [OpenAlex ID](https://openalex.org/A5008809634)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套句子级别的解释框架，结合模型无关的 SHAP 归因和大语言模型（LLM）生成的句子排序，用于评估课堂教学质量维度——反馈质量（QoF），并对细调的预训练语言模型（PLM）与提示式 LLM 的评分性能与解释可信度进行系统比较。

**💡 创新点**

创新点包括：①将 SHAP 与 LLM 排序结合在同一框架内；②通过逐句删除（deletion‑based）和跨模型迁移（cross‑model）两种方法量化解释的因果影响；③首次展示 SHAP 在多模型间具有更高的可迁移性与可信度，而 LLM 生成的解释往往不一致且易出现错误。

**🔧 技术方法**

技术手段：对 PLM（BERT、ALBERT、RoBERTa、DeBERTaV3）进行微调后做回归评分；使用 Llama、Mixtral、Qwen3、Mistral 等开源 LLM 进行 1‑shot/4‑shot 评分与句子排序；采用 SHAP 计算句子级 Shapley 值；通过逐句删除、Jaccard 与 Spearman 统计等方法评估解释的 faithfulness 与一致性。

**📊 数据集**

数据集：NCTE 小学算术课堂转录共 6,005 段落，按 80/20 分割为训练/测试，使用 CLASS 框架下的 QoF 维度（1–7 量表）进行注释。

**📈 对比分析**

比较方法：①评估 MAE/MSE；fine‑tuned PLM MAE ≈0.96，LLM MAE ≈1.02；②删除实验显示 PLM 的 SHAP 句子删除平均 Δ≈0.03，LLM 排序 Δ≈0.02；③跨模型迁移实验表明 SHAP 排序对另一模型的影响显著，LLM 排序影响微弱。总体而言，PLM 在准确性上优于 LLM，但 LLM 在输出范围上更宽；SHAP 解释在所有模型间表现更一致且更具因果性。

**⚠️ 局限性**

局限性：①数据集规模有限且 QoF 标签分布偏中间，导致 PLM 对极端分数的预测压缩；②仅使用文本转录，忽略音频、视觉等多模态特征；③每段仅有一位专家注释，缺乏可靠性评估；④删除实验破坏课堂语境，可能影响 LLM 的表现；⑤LLM 句子排序的不确定性与错误率高，限制其在实际教育系统中的可用性。

---

## 96. VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding

**arXiv ID:** 2606.05259 | [PDF](https://arxiv.org/pdf/2606.05259v1)

**作者:** Lin Fu `[一作]` (Zhejiang University), Yilun Zhao `[通讯]` (Yale University)

**通讯引用:** 639 | [OpenAlex ID](https://openalex.org/A5047416722)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模知识与推理驱动的视频理解训练语料库VideoKR，并基于该语料库推出新的评测基准VideoKR-Ref；

**💡 创新点**

创新点在于：1）以专业领域知识为核心的知识驱动视频采集与情景生成；2）基于三维能力（基础推理、知识增强感知、知识深度推理）的技能导向QA生成框架；3）采用多模型人机审核与严格过滤保证CoT推理质量；4）设计单帧过滤机制，构建更可靠的评测集；

**🔧 技术方法**

技术手段包括：LLM场景生成与QA/CoT生成、专家审核与多模型自检（Self‑Consistency、Video‑Dependency、CoT Validation）、SFT→GRPO（基于ROUGE/Exact Match奖励）的后训练流程，以及LMMs‑Eval统一评测框架；

**📊 数据集**

使用数据集：145K CC‑licensed 专业领域视频；315K QA例子（含CoT）；新评测集VideoKR-Ref（2000例子）；以及基准评测集Video‑MME、MVBench、LongVideoBench、VideoMMMU、MMVU、SciVideoBench；模型基准为Qwen2.5‑VL‑7B‑Instruct 与 Qwen3‑VL‑8B‑Instruct；

**📈 对比分析**

采用SFT→GRPO的标准后训练流程，在7个基准上与先前公开的后训练语料库对比，VideoKR 在知识密集型任务上平均提升4–8个百分点，Qwen3‑VL‑8B‑Instruct 以51.5% 成为同规模模型中最优；在一般视频推理任务上保持竞争力；

**⚠️ 局限性**

局限性包括：1）仅收集时长≤30分钟的视频，无法覆盖长时序场景；2）过度依赖LLM生成，可能带来模型偏差；3）评测基准仍可能存在未检测到的单帧可解例子；4）对更大模型或更高帧数推理的可扩展性尚未验证。

---

## 97. Learning-Augmented Online Minimization with Dual Predictions

**arXiv ID:** 2606.05380 | [PDF](https://arxiv.org/pdf/2606.05380v1)

**作者:** Christian Coester `[一作]` (University of Oxford), Alexander Turoczy `[通讯]` (University of Oxford)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了利用机器学习预测双侧（dual）变量的在线学习增强算法，应用于两类在线最小化问题：层状集合覆盖（laminar set cover）和度量任务系统（Metrical Task Systems）

**💡 创新点**

创新点在于：① 用双侧预测替代传统的事件或动作预测，双侧变量更稳定且可学习；② 证明双侧预测满足稳定性、有效性与可学习性三大属性；③ 为层状集合覆盖和MTS分别给出竞争比提升的理论保证与实验验证

**🔧 技术方法**

技术上使用：基于LP对偶框架的预测算法设计、贪心求解对偶最优、对偶误差定义与分析、结合A*搜索的MTS预测算法；实验使用随机化在线算法组合与传统竞争算法对照

**📊 数据集**

实验数据集：① 153年纽约中央公园降雨天气记录用于停车许可证问题；② 2016-2025年NYC共享单车行程数据用于k‑server实验

**📈 对比分析**

比较方法：与最优经典在线算法（随机化/确定性）以及相同预测类型的其他学习增强算法对照；实验结果显示学习增强算法在竞争比上明显优于传统算法，特别是在许可类型数或折扣因子较大时，竞争比下降至近最优水平

**⚠️ 局限性**

局限性：仅针对MTS与层状集合覆盖证明可行；对一般集合覆盖仍存在最坏情况难题；预测误差衡量与对偶LP选择对结果影响较大，需针对具体问题进一步调整

---

## 98. A Taxonomy of Runtime Faults in Model Context Protocol Servers

**arXiv ID:** 2606.05339 | [PDF](https://arxiv.org/pdf/2606.05339v1)

**作者:** Joshua Owotogbe `[一作]` (Jheronimus Academy of Data Science), Roberto Natella `[通讯]` (University of Naples Federico II)

**通讯引用:** 2518 | [OpenAlex ID](https://openalex.org/A5034602812)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对473个MCP服务器仓库的手工分析，识别并整理了837条运行时故障线索，构建了首个面向MCP服务器的经验性故障分类体系。

**💡 创新点**

创新点在于首次将MCP协议特定的运行时故障进行系统化、底层构建，并通过55名维护者的问卷验证，确保了分类的覆盖性与实用性。

**🔧 技术方法**

所采用的技术包括GitHub仓库挖掘、错误线程的人工编码与开放式归纳、层次化分类、以及针对开发者的问卷调查。

**📊 数据集**

使用的数据集来自MCP Market与Awesome MCP Servers公开目录，最终筛选出473个活跃仓库，进一步抽取了837条确认的故障线索，并收集了55名MCP服务器开发者的反馈。

**📈 对比分析**

本文并未进行传统意义上的性能对比，而是通过问卷中对每个子类的出现率、严重性和诊断难度进行统计分析，以验证分类的实践覆盖度。

**⚠️ 局限性**

局限性包括仅覆盖公开GitHub仓库，无法反映私有或企业内部实现；仅聚焦服务器端故障，未考虑客户端或调度层问题；以及可能存在的问题描述缺失导致的抽样偏差。

---

## 99. CausalPOI: Spatio-Temporal Graph-Based Causal Modeling for Cold-Start POI Check-in Forecasting

**arXiv ID:** 2606.05413 | [PDF](https://arxiv.org/pdf/2606.05413v1)

**作者:** Zhaoqi Zhang `[一作]` (Nanyang Technological University), Gao Cong `[通讯]` (Nanyang Technological University)

**通讯引用:** 16387 | [OpenAlex ID](https://openalex.org/A5045198704)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了冷启动POI签到预测问题，旨在预测新开设地点未来的签到模式。

**💡 创新点**

创新点在于构建了空间-时间功能交互图(ST‑FIG)，结合对照图实现因果推理，能够同时捕捉功能依赖与局部干预效应。

**🔧 技术方法**

采用的技术包括：BERT文本编码、对比学习预训练功能相似度、GATv2图神经网络、GRU时序建模、因果潜在结果框架与倾向得分正则化。

**📊 数据集**

实验数据来自SafeGraph美国POI与签到数据，覆盖四个地区（东北、中西部、南部、西部）。

**📈 对比分析**

与多种基线（时序GNN、统计、LLM、因果与生成模型）比较，CausalPOI在RMSE/MAE上均优于所有对手，提升幅度最高可达57.8%/34.3%。

**⚠️ 局限性**

局限性包括：仅在单一城市/国家数据上验证，缺少多时空跨域泛化实验；对干预时点的时间窗口有限；以及潜在混杂因素仍未完全消除。

---

## 100. Residual Modeling for High-Fidelity Learned Compression of Scientific Data

**arXiv ID:** 2606.05389 | [PDF](https://arxiv.org/pdf/2606.05389v1)

**作者:** Liangji Zhu `[一作]` (University of Florida), Anand Rangarajan `[通讯]` (University of Florida)

**通讯引用:** 12775 | [OpenAlex ID](https://openalex.org/A5059870257)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出两种高保真科学数据压缩中的残差编码方法（LBRC和NGLR），并将残差视为独立的可编码对象。

**💡 创新点**

创新点在于：①使用目标匹配整数Lorenzo变换对残差进行确定性编码；②在此基础上加入轻量级因果神经偏差预测器，实现熵降低且保持严格的块级NRMSE保证；③通过分离精度控制与熵优化，解决GAE在高精度下残差率瓶颈。

**🔧 技术方法**

技术手段包括：目标匹配量化、3D Lorenzo差分、zigzag映射、位平面编码、熵编码；NGLR中还加入因果神经网络预测Lorenzo偏差，网络参数在压缩流中直接携带。

**📊 数据集**

使用的科学数据集：E3SM气候模拟、JHTDB湍流模拟、ERA5大气再分析，共计约3 GB。

**📈 对比分析**

性能对比：与传统SZ和GAE比较，在块级NRMSE 10⁻⁶–10⁻⁴区间，LBRC比GAE提升30–60% CR，NGLR在此基础上再提升10–40%并往往优于SZ；不同数据集表现差异显著，NGLR在残差结构最丰富的JHTDB上收益最大。

**⚠️ 局限性**

局限性：需为每个压缩集合训练神经预测器；对残差趋近白噪声时收益有限；相对于LBRC计算开销更大，吞吐量下降；仅处理块内残差，未利用跨块全局结构。

---

## 101. Latent Reasoning Guidance for Parallel Code Translation

**arXiv ID:** 2606.05518 | [PDF](https://arxiv.org/pdf/2606.05518v1)

**作者:** Tomer Bitan `[一作]` (Technion), Gal Oren `[通讯]` (Technion)

**通讯引用:** 348 | [OpenAlex ID](https://openalex.org/A5030616543)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在可执行并行代码翻译任务中，提出一种测试时的潜在指导方法，利用小型的 Process Reward Model（PRM）在生成最终代码前对连续隐藏状态进行评分并挑选最佳分支；

**💡 创新点**

创新点在于：①将潜在空间分支选择作为预解码干预点；②通过训练小型 PRM 对隐藏前缀进行评分，而不对主生成模型进行微调；③展示 PRM 在多步潜在推理中有效过滤有害分支，提升最终可执行率；

**🔧 技术方法**

技术包括：冻结 LLaMA‑3.3‑70B 潜在推理模型；使用 Qwen‑Coder‑7B 作为 PRM，并加入线性适配器映射隐藏维度；对每一步潜在状态做多重扰动采样，计算终端奖励后训练 PRM；推理时采用贪婪分支选择；

**📊 数据集**

使用 ParaTrans 76 任务的测试集（包括 CUDA、OpenMP、Serial 等多种并行 API 方向），训练集为 60+ 开发样本，评估集为 76 个未见任务；

**📈 对比分析**

与无指导潜在推理、随机分支选择、以及 fine‑tuned LLaMA 基线相比，PRM 指导下的验证率在无修复情形提升至 42.10%（+9.21pp），在三轮修复循环中提升至 45.18%（+8.78pp），远高于随机分支（27.28%）和未指导潜在推理（32.89%）；

**⚠️ 局限性**

局限性：仅在单一基模型和单一 PRM 体系上验证；需要访问中间隐藏状态，限制了黑盒模型的适用性；需要耗时的监督训练（需多轮 roll‑out 与可执行验证）；缺乏跨模型迁移与更大规模的鲁棒性评估。

---

## 102. Do Models Share Safety Representations? Cross-Model Steering for Safe Visual Generation

**arXiv ID:** 2606.05290 | [PDF](https://arxiv.org/pdf/2606.05290v1)

**作者:** Tobia Poppi `[一作]` (University of Modena and Reggio Emilia), Rita Cucchiara `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 20212 | [OpenAlex ID](https://openalex.org/A5030948871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨模型安全方向迁移框架，将源 LLM 学习到的安全向量通过仅使用安全锚点的轻量级对齐迁移到目标文本‑图像/视频生成模型，并在推理时进行可调强度的干预。

**💡 创新点**

证明安全行为可以用可迁移的几何方向来表征，无需目标模型的危险数据即可通过共享表征几何实现跨模型安全控制。

**🔧 技术方法**

利用安全‑不安全提示对估计源安全向量；使用 SVD、岭回归或小型 MLP 在安全锚点上学习跨模型映射；在目标隐藏层添加标量可调的安全方向进行推理干预。

**📊 数据集**

SafeSteerDataset（安全/不安全提示对）、WikiText、COCO/Flickr（安全锚点）、I2P（文本‑图像安全基准）、T2VSafetyBench（文本‑视频安全基准）及 LAION（评估图像相似度）。

**📈 对比分析**

与原始模型、目标端原生安全向量、随机方向以及文本概念向量对比；实验显示转移方向显著降低攻击成功率（ASR），同时保持或提升 CLIP 相似度，性能可与拥有不安全数据的目标 oracle 相媲美。

**⚠️ 局限性**

对齐质量与调参 α 共同决定安全‑效能权衡；对极端安全需求或高维多模态模型的适用性仍有限；未在更大规模/多语言场景中验证。

---

## 103. Search-Time Contamination in Deep Research Agents: Measuring Performance Inflation in Public Benchmark Evaluation

**arXiv ID:** 2606.05241 | [PDF](https://arxiv.org/pdf/2606.05241v1)

**作者:** Yongjie Wang `[一作]` (Alibaba-NTU Global e-Sustainability CorpLab), Zhiqi Shen `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统研究并量化了深度研究代理在公开基准上因网络检索导致的搜索时间污染（STC），并提出了三类污染类型（BML、QCL、EAL）与相应检测算法。

**💡 创新点**

创新点在于：① 将STC细分为三个不同严重程度的子类型；② 结合正则表达式URL匹配、关键词重叠度和LLM-as-Judge进行自动检测；③ 通过问答级别与步长级别的评估框架（包括时间变Cox回归和Kaplan–Meier曲线）系统性衡量污染对性能的影响。

**🔧 技术方法**

采用的技术包括正则表达式URL匹配、最长公共子串重叠度、LLM-as-Judge判别、ReAct框架下的搜索–推理–观察循环、时间变Cox回归与Kaplan–Meier生存分析、以及对公开深度研究代理轨迹的可视化与日志分析。

**📊 数据集**

数据集：MedQA、MedMCQA、MMLU医学子集、MedXpertQA、HLE-149（医学子集）、Medbullets5op（临床多选题）。评估代理包括Tongyi Deep Research、Gemini Deep Research、Step Deep Research、Valyu Deep Research以及基准模型Qwen3-30B-A3B。

**📈 对比分析**

对比方法：在同一基准上分别关闭/开启网络搜索，分离不同STC类型的子集进行性能比较。结果显示开启搜索后模型性能平均提升约3–4%，其中EAL导致的提升最大，甚至可达10%（如MedXpertQA从28.45%提升至40.61%）。对比基准模型，开启搜索的Tongyi Deep Research在MedQA上达91.28%（基准为83.58%），但若剔除STC实例，提升幅度仅为1–2%。

**⚠️ 局限性**

局限性：① URL匹配依赖预设网站模式，可能漏检其他来源的污染；② 仅覆盖医学/临床QA领域，未验证通用任务；③ 商业代理评估样本有限，无法覆盖所有潜在污染渠道；④ 检测方法在极少量高质量数据上仍需人工标注以提升召回率。

---

## 104. PHKT:Personalized Dynamic Hypergraph-enhanced KAN-Transformer for Multi-behavior Sequential Recommendation

**arXiv ID:** 2606.05537 | [PDF](https://arxiv.org/pdf/2606.05537v1)

**作者:** Ruijie Du `[一作]` (Hangzhou Dianzi University), Dongjin Yu `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 2521 | [OpenAlex ID](https://openalex.org/A5042936370)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出PHKT模型，用个性化动态超图与KAN-Transformer相结合实现多行为序列推荐

**💡 创新点**

创新点在于(1)使用行为感知的动态超图实现用户个性化高阶关系建模；(2)将Kolmogorov‑Arnold Network替换Transformer中的MLP，实现对多行为潜在模式的细粒度非线性响应建模；(3)三者融合形成统一框架

**🔧 技术方法**

技术包括Transformer自注意力、超图卷积、KAN层、基于行为权重的相似度加权、Mask预测与交叉熵损失

**📊 数据集**

在三大工业级数据集上测试：Tmall、RetailRocket、IJCAI

**📈 对比分析**

与9种强基线（GRU4Rec、SASRec、MAERec、SelfGNN、MBHT、PBAT、MBSRec等）对比，PHKT在所有数据集上在HR@5/10、NDCG@5/10、MRR@5均显著提升（例如RetailRocket HR@10 0.952，Tmall HR@10 0.440，IJCAI HR@10 0.583）

**⚠️ 局限性**

局限包括对超参数（特征维度、KAN层数、行为权重）敏感；在已高度饱和的数据集上提升有限；模型复杂度相对较高，需进一步探索轻量化与更高效的训练策略

---

## 105. PSEBench: A Controllable and Verifiable Benchmark for Evaluating LLMs in Patient Safety Event Triage

**arXiv ID:** 2606.05463 | [PDF](https://arxiv.org/pdf/2606.05463v1)

**作者:** Keqi Han `[一作]` (Emory University), Zhijun Yin `[通讯]` (Vanderbilt University)

**通讯引用:** 2639 | [OpenAlex ID](https://openalex.org/A5079247989)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向患者安全事件(PSE)报告可报性 triage 的基准 PSEBench，支持多维度评估 LLM 在法律依据、缺失信息检索和不确定性处理上的表现。

**💡 创新点**

创新点在于：① 使用“条款卡”(clause card)将复杂法规拆解为可审计的决策规范；② 通过闭环验证的生成流水线确保合成案例的标签与法律依据一致；③ 设计了多轮交互评估环境，测评 LLM 的主动查询和不确定性下的自我中止。

**🔧 技术方法**

技术包括：LLM 辅助的条款卡编写与人机审核；基于 anchor 材料的情境化文本生成；闭环验证（实例化验证 + 叙事验证）；两角色交互代理（评估代理与信息代理）。

**📊 数据集**

数据集：采用 Minnesota 29 Reportable Adverse Health Events (MN29) 法规为知识库；从日本 JQ 数据库提取并翻译的 2023/2024 年安全事件报告作为 anchor；最终生成 5,074 条案例（3,455 完整、1,362 缺失信息、257 不确定）。

**📈 对比分析**

与 15 种代表性 LLM（闭源前沿、开源前沿、小型通用、医学专用）进行对比。结果显示：闭源前沿模型在判定准确率上优越，但在信息检索和不确定性下的中止表现差异显著；开源和小型模型在主动查询和不确定性处理上大幅落后；医学专用模型虽然在常规 QA 上表现不错，但在政策导向 triage 的关键维度表现最差。

**⚠️ 局限性**

局限性：① 合成文本仍未完全覆盖真实报告中的杂乱结构和非标准缩写；② 交互评估使用理想化工具接口，未检验在真实 EHR 系统中检索的可行性；③ 当前仅基于 MN29 政策，其他州或国家的报告体系需要重新构建条款卡和验证流程。

---

## 106. Scaling Laws for Behavioral Foundation Models over User Event Sequences

**arXiv ID:** 2606.05257 | [PDF](https://arxiv.org/pdf/2606.05257v1)

**作者:** Rickard Brüel Gabrielsson `[一作]` `[通讯]` (Unbox AI), Rickard Brüel Gabrielsson (Unbox AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文在现有的两阶段“事件嵌入器 → 变换器”框架下，对行为基础模型的四个关键超参维度（嵌入器比例、批量大小、计算与数据分配、冻结后负样本数）进行大规模（约600次）实验，挖掘其可扩展性规律。

**💡 创新点**

创新点：①首次给出行为模型的嵌入器规模最优比例约为2%，并解释其计算与数据重现性差异；②揭示计算分配（D/N）随预算下降的独特轨迹，既比语言模型更偏向数据，也随规模趋近Chinchilla经验；③证明评估指标本身是扩展法则的一部分，指标变化会改变最优计算配置；④提供一套完整的实验矩阵与可复现的分析方法。

**🔧 技术方法**

使用技术包括：两阶段训练（联合训练 + 冻结后额外负样本训练）、采样Softmax、Kaplan公式、等价可扩展性曲线、二项式“饿死-采样偏差”拟合、Critical Batch Size（Kaplan-McCandlish）模型、Spearman相关性与排名一致性分析、以及全量候选集与批量局部评估对比。

**📊 数据集**

数据集：匿名化真实零售交互语料，包含产品搜索、浏览、点击、购买等多模态事件，约1亿唯一动作、约10亿事件tokens；实验覆盖了10^15–10^19 FLOPs范围。

**📈 对比分析**

对比方法：在训练阶段使用批量局部评估（交叉熵/损失），在推理阶段使用全量候选集评估（recall@k、NDCG@k、MRR@10、coverage@k、predictive entropy）。实验结果表明：嵌入器比例≈2%、批量大小≈2048、D/N随预算从344降至36、冻结后负样本数最佳区间≈2.5×10^5–9×10^5；这些组合在相应指标上实现了最优或近最优性能。

**⚠️ 局限性**

局限性：①批量局部评估与全量评估不完全对齐，导致某些指标（尤其是损失）对配置的敏感度被低估；②未对完整架构/分配网格进行全量候选集评估；③实验仅覆盖固定上下文长度（256）和计算范围（10^15–10^19 FLOPs）；④未探究不同嵌入器正则化或更深层架构的影响；⑤负样本数在更大预算下受候选集显存限制，需更复杂的分布式软max方案。

---

## 107. Sharp Low-Degree Thresholds for Planted-vs-Planted Testing

**arXiv ID:** 2606.05266 | [PDF](https://arxiv.org/pdf/2606.05266v1)

**作者:** Anda Skeja `[一作]` (Uppsala University), Alexander S. Wein `[通讯]` (University of California, Davis)

**通讯引用:** 774 | [OpenAlex ID](https://openalex.org/A5021031923)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在植入测试（planted testing）问题中，如何利用低阶多项式方法区分不同社区数的植入子矩阵和植入稠密子图模型，并推导出对应的低阶阈值；

**💡 创新点**

首次将低阶方法推广到植入测试情形，提出了通用的延拓空间正交化证书框架，并在两类模型中给出匹配的强弱检测阈值；

**🔧 技术方法**

使用低阶方法、Hermite/多项式基底、延拓空间正交化以及图重叠计数等技术；

**📊 数据集**

该工作为理论研究，无需具体实测数据集，仅基于随机矩阵/图模型；

**📈 对比分析**

与传统图统计量（如迹、三角形计数）比较时，证明在相应信号强度下低阶多项式能够实现强/弱分离，性能与理论阈值一致；

**⚠️ 局限性**

局限性在于尚未提供高效的多项式实现算法、仅考虑固定社区数和稀疏性假设，以及弱检测仅在特定信号与稠密度条件下成立。

---

## 108. OLIVE: Online Low-Rank Incremental Learning for Efficient Adaptive Exoskeletons

**arXiv ID:** 2606.05234 | [PDF](https://arxiv.org/pdf/2606.05234v1)

**作者:** Dong Liu `[一作]` (University of California), Ying Nian Wu `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在可穿戴动力式下肢外骨骼上实现了在线低秩自适应控制（OLIVE），能够在实时推理约10 ms内根据多模态传感器反馈动态更新控制参数。

**💡 创新点**

核心创新包括：①将控制器参数分解为冻结的基础矩阵和可在线更新的低秩因子；②加入门控机制根据用户状态调节个性化残差；③设计动态秩调度器按环境复杂度切换低秩维度；④采用奖励塑形的策略梯度实现无参考轨迹的自适应学习。

**🔧 技术方法**

技术手段涵盖：低秩参数化、门控网络、动态秩调度、奖励塑形策略梯度、基于ARM SoC的实时推理、四模态（IMU、关节编码器、表面EMG、振动）感知融合。

**📊 数据集**

预训练基础模型使用公司内部的大规模多模态运动数据（IMU、编码器、EMG、振动、运动序列）。实验数据来自6名健康志愿者，每人完成约5000步，覆盖平地、楼梯、斜坡、崎岖路面四种场景。

**📈 对比分析**

与三种基线（静态控制、基于规则的有限状态机、固定神经网络）进行对比，评估指标包括步态平滑度、努力降低、运动稳定性。OLIVE在所有指标上均领先，平滑度提升13%，努力降低22%，并在学习曲线中快速收敛到最低努力状态。

**⚠️ 局限性**

局限性：实验仅在少数健康受试者和日常场景下进行，未验证在慢性病患者或极端环境中的鲁棒性；低秩假设可能限制对极其复杂动态的捕捉；系统对传感器噪声、异常值的鲁棒性尚未深入评估。

---

## 109. Exponential Quantum Space Advantage for Approximating Max-$k$SAT in the Streaming Setting

**arXiv ID:** 2606.05366 | [PDF](https://arxiv.org/pdf/2606.05366v1)

**作者:** Haoyu Wang `[一作]` (Penn State University), Guangxu Yang `[通讯]` (University of Southern California)

**通讯引用:** 40918 | [OpenAlex ID](https://openalex.org/A5100320478)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种一次通过的量子流式算法，在多项式对数空间内完成 Max‑kSAT（以及 Max‑2OR/Max‑2SAT）的近似求解，并给出了 0.7172（以及 0.7425）的近似比；同时证明了在流式设置下，量子空间相较于经典算法可以实现指数级的空间优势。

**💡 创新点**

创新点包括：① 设计了基于 snapshot‑to‑pseudosnapshot 的框架，将 Max‑kSAT 约简为可在一次流式中估计的加权 snapshot；② 通过求解有限线性规划得到一个严格的证书，使得该加权 snapshot 能以 0.717275 的下界保证；③ 结合量子 sketching（类似隐藏匹配技术）实现了对加权 pseudosnapshot 的一次通过估计，空间复杂度仅为 O(log⁵ n)；④ 完成了 Boolean Max‑2CSP 的量子空间优势完整分类。

**🔧 技术方法**

采用了三种核心技术：① 变分化简（删去重复字面量、真值冲突子句并记录计数）；② 通过 weighted snapshot + finite LP 证书对 3‑子句进行线性近似；③ 基于量子隐藏匹配式 sketch 的高阶度量估计，实现对所有加权坐标的精确一次通过估计。

**📊 数据集**

该工作为理论论文，无实测数据集，所有结果均来自数学证明与符号计算（如 LP 求解与 rational 证书验证）。

**📈 对比分析**

与经典流式算法对比：经典流式对 Max‑kSAT 的最优近似阈值为 √2/2 ≈ 0.7071，需要 Ω(√n) 空间；而本文的量子算法在 O(polylog n) 空间内即可实现 0.7172 近似，体现出指数级空间优势。对 Max‑2OR/Max‑2SAT 的结果则完成了量子空间优势分类，表明除 AND‑类型外，其余类型均存在量子空间优势。

**⚠️ 局限性**

局限性：① 近似比尚未达到理论最优 0.7418（或更高）仍有改进空间；② 仅适用于一次通过流式模型；③ 需要多项式对数量子寄存器，实际实现对量子硬件要求较高；④ 目前仅证明了 Max‑kSAT 与 Max‑2CSP 的量子空间优势，未覆盖更一般的 CSP。

---

## 110. SlotGCG: Exploiting the Positional Vulnerability in LLMs for Jailbreak Attacks

**arXiv ID:** 2606.05609 | [PDF](https://arxiv.org/pdf/2606.05609v1)

**作者:** Seungwon Jeong `[一作]` (Dongguk University), Woojin Lee `[通讯]` (Dongguk University)

**通讯引用:** 19440 | [OpenAlex ID](https://openalex.org/A5100410780)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于SlotGCG的改进攻击方法，系统识别并利用LLM提示中各位置的脆弱插槽，以提高越狱成功率和鲁棒性。

**💡 创新点**

引入Vulnerable Slot Score (VSS) 定量衡量插槽脆弱性，并通过SlotGCG在最易受攻击的插槽上聚焦，突破传统仅在后缀插入的攻击限制。

**🔧 技术方法**

结合梯度优化（GCG）、注意力权重分析、Softmax概率分配、Token插槽分配与多轮迭代优化，以及Universal SlotGCG的跨行为通用攻击。

**📊 数据集**

使用 AdvBench 的 50 个有害行为示例进行评估，验证跨模型和跨防御的效果。

**📈 对比分析**

与多种基线攻击（GCG、AttnGCG、I-GCG、GCG-Hij、GBDA）及多种防御（Erase-and-Check、Perplexity Filter、SmoothLLM、RPO、SafeDecoding、Llama-Guard-3）对比，SlotGCG 在多模型上平均提升攻击成功率约 14%，迭代次数下降 60%，对防御的鲁棒性提高 42%。

**⚠️ 局限性**

仍需依赖模型的注意力计算，VSS 可能对不同模型表现不一致；方法在极大规模模型或复杂提示结构下的计算开销尚未彻底评估；仅针对英文提示，跨语言适用性待验证。

---

## 111. When Evidence is Sparse: Weakly Supervised Early Failure Alerting in Dialogs and LLM-Agent Trajectories

**arXiv ID:** 2606.05414 | [PDF](https://arxiv.org/pdf/2606.05414v1)

**作者:** Avinash Baidya `[一作]` (Intuit AI Research), Kamalika Das `[通讯]` (Intuit AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于弱监督的两阶段早期失败预警框架，先学习稀疏的转折点证据，然后利用单一可调参数的停止策略在不同准确率-提前性取舍下实时触发警报。

**💡 创新点**

创新点在于：①使用多实例学习+注意力机制从仅有轨迹级标签中挖掘稀疏且往往在后期出现的关键失败证据；②将该稀疏证据与传统前缀预测融合得到更精准的在线失败风险估计；③设计α-STOP（α-STOP）可调停止策略，允许在推理时通过单个α值连续覆盖整个准确率-提前性前沿，显著降低了训练成本。

**🔧 技术方法**

核心技术包括：多实例学习（MIL）+注意力、前缀嵌入编码器（冻结预训练模型）、两阶段融合网络、基于行为克隆与PPO的α-STOP策略，以及与基线对照的准确率-提前性评估指标（最大准确率、Hypervolume、IGD+）。

**📊 数据集**

在五个公开/私有基准上评估：客户支持对话（PCS）、任务导向对话（BETOLD）、说服类对话（P4G）、工具/API使用（AppWorld）以及文本环境规划（ALFWorld），涵盖对话与LLM代理轨迹。

**📈 对比分析**

与现有最佳触发策略（ALERT^*, FIRMBOUND）、LLM-judge、端到端RL及简单阈值触发器比较，α-STOP在Hypervolume提升3–42%，最大准确率提升3–10%，并在每个指标上均优于或相当于现有方法；同时每个操作点的训练成本比对手低1–3个数量级。

**⚠️ 局限性**

局限性包括：①仅在所选五个基准上验证，稀疏证据特性可能不适用于所有多轮任务；②依赖预训练编码器的窗口与语义覆盖，迁移到不同领域可能需重新调优；③稀疏证据抽取基于后验LLM评分，仅作诊断；④在极端短轨迹或极度多样化的交互场景中，早期预测可能仍受限。

---

## 112. From Attack Simulation to SIEM Rule: Deterministic Detection-as-Code Synthesis with Probe-Level Traceability

**arXiv ID:** 2606.05252 | [PDF](https://arxiv.org/pdf/2606.05252v1)

**作者:** Alexandre Cristovão Maiorano `[一作]` `[通讯]` (Lumytics), Alexandre Cristovão Maiorano (Lumytics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于锁定探针语料库的确定性 Sigma 规则合成流程，将安全评估工具（BAS）产生的绕过探针发现直接转换为可在 SIEM 中部署的规则，并提供每条规则的可追溯 URI。

**💡 创新点**

创新点在于：①通过 OWASP Top‑10 与 MITRE ATT&CK 对探针进行归类，使发现携带足够结构化信息以实现模板级别的规则生成；②实现完全确定性的映射（同一发现同一 YAML），从而保证可审计、可版本控制的检测内容；③在规则中嵌入对原始探针和技术标签的双重引用，为操作员提供端到端的追溯链。

**🔧 技术方法**

主要技术包括：锁定 JSON 语料库（含 probe_id、OWASP 类别、MITRE 代码）、Sigma 规则模板库（23 个模板）、Python/伪代码实现的 `findingToSigma` 合成函数、pysigma 用于解析与后端转换、Splunk SPL 与 Elasticsearch Lucene KQL 的多后端兼容性处理。

**📊 数据集**

使用的数据集：17 条 LLM 探针（涵盖 OWASP LLM01‑LLM10）、23 条 Web 探针（涵盖 OWASP A01‑A09），以及两个公开的受害提示集（AdvBench、HarmBench）用于持久化测试和真实 SIEM replay。

**📈 对比分析**

比较方法包括模板命中率、假阳性率（benign baseline 100 条日志）、对 AdvBench/HarmBench 的真阳性率、以及在 OpenSearch+Lucene 上的实时 SIEM replay。实验结果显示：所有被绕过的发现均能生成规则（template‑hit 100%）；在 AdvBench 的持久化测评中，v2 模板实现 30%（30/100）火率，HarmBench 14%；在真实 SIEM replay 中，AdvBench 30% 召回、HarmBench 14%，benign‑LLM 假阳性 7.7%。

**⚠️ 局限性**

局限性包括：模板仅支持关键词/选择器检测，无法覆盖多请求或时间序列的攻击；Web 侧缺乏独立的持久化验证集；Lucene 后端的正则兼容问题导致部分规则失效；以及未进行用户时间效率实验，缺少实际 SOC 操作员的性能评估。

---

## 113. Learning from Demonstrations over Riemannian Manifolds using Neural ODEs: An Extended Abstract

**arXiv ID:** 2606.05422 | [PDF](https://arxiv.org/pdf/2606.05422v1)

**作者:** Diana Cuervo Espinosa `[一作]` (Technical University of Munich), Angela P. Schoellig `[通讯]` (Technical University of Munich)

**通讯引用:** 6224 | [OpenAlex ID](https://openalex.org/A5052147335)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

在学习示范 (LfD) 的框架下，先用变分自编码器 (VAE) 将机器人演示数据映射到黎曼流形，再利用神经常微分方程 (NODE) 计算流形上的代理测地线，从而生成自然且高效的机器人运动轨迹。

**💡 创新点**

创新点在于将流形学习与 NODE 相结合，实现在非欧氏空间中快速求解测地线，并在推理阶段大幅提升计算速度，同时保持对姿态和位置的完整建模。

**🔧 技术方法**

使用的技术包括：变分自编码器 (VAE) 进行流形学习；神经常微分方程 (NODE) 用于测地线求解；光滑激活函数保证 NODE 的连续性；低层控制器用于轨迹跟踪。

**📊 数据集**

实验使用的演示数据集为二维平面位置的 J‑形轨迹与三维球面上投影的 C‑形姿态（ℝ² × S²），来源于此前公开的演示记录。

**📈 对比分析**

与基于图的离散测地线方法（40×40、100×100 节点）对比，本文方法在目标收敛误差上相当甚至更优，同时推理时间从 3.35 秒降至 0.27 秒，显著提升了实时性能。

**⚠️ 局限性**

当前框架缺乏对测地线保持在流形内以及收敛至目标的理论保证；NODE 的训练基于局部欧氏损失，可能导致轨迹漂移；后续需在真实机器人上进一步验证并引入正式的几何约束与安全性证明。

---

## 114. Learned Subspace Compression for Communication-Efficient Pipeline Parallelism

**arXiv ID:** 2606.05484 | [PDF](https://arxiv.org/pdf/2606.05484v1)

**作者:** Paul Janson `[一作]` (Concordia University), Eugene Belilovsky `[通讯]` (Concordia University)

**通讯引用:** 1502 | [OpenAlex ID](https://openalex.org/A5025113992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 MAPL（Manifold Aware Projection Learning）方法，学习每个流水线阶段的正交低秩投影子空间，以压缩激活并降低跨设备通信成本。

**💡 创新点**

创新点在于：① 在 Stiefel 流形上进行正交投影学习，避免投影偏离正交性导致性能下降；② 采用分块锚定嵌入（factorized anchor embeddings）实现全秩激活重构，几乎不增加通信开销；③ 可选地将投影后的低维表示进行残差向量量化，进一步提升压缩比。

**🔧 技术方法**

技术手段包括：Stiefel 流形约束下的最速下降优化（SPEL）、分块锚定嵌入、残差向量量化（Multi‑Codebook VQ）以及基于张量分解的权重优化。

**📊 数据集**

实验使用 LLaMA 结构的 150M、500M、1B 参数模型，预训练数据集为 DCLM，评估数据集为 HellaSwag、PIQA、ARC‑Easy、ARC‑Challenge。

**📈 对比分析**

与未压缩训练以及 Subspace Networks（SSN）基线相比，MAPL 在 4×–8× 通信压缩下，验证交叉熵误差仅比未压缩低 1–2%，并在 16× 压缩（加 VQ）时仅额外增加约 3% 误差；SSN 在相同压缩比下的误差显著更高（约 8–14%）。

**⚠️ 局限性**

局限性包括：实验仅覆盖至 1B 参数规模，未在更大规模或真实异构网络环境下验证；在极端压缩比下（尤其是加 VQ）仍会出现显著性能下降。

---

## 115. TopoPult-SSL: Gland-Mask-Free Cross-Device Meibomian Gland Segmentation via Self-Distilled Weak Clinical Priors

**arXiv ID:** 2606.05347 | [PDF](https://arxiv.org/pdf/2606.05347v1)

**作者:** Nicolò Savioli `[一作]` (OdaxAI S.R.L.), Luca Del Tongo `[通讯]` (Topcon Group --- VISIA Imaging S.R.L.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出TopoPult-SSL两阶段跨设备泪腺分割框架，第一阶段利用眼睑掩码与临床评分等弱先验实现无肢体掩码的适配，第二阶段通过监督自蒸馏将两种教师模型融合为单一高性能学生模型。

**💡 创新点**

创新点包括：①在目标设备无肢体掩码条件下使用四个弱先验锚点（拓扑、分辨率等价、形态一致、眼睑解剖）实现自适配；②利用监督自蒸馏在仅有限标注下超越传统SSL和教师集成的表现；③在单次推理下达成SOTA Dice。

**🔧 技术方法**

使用EMA教师-学生对齐、中心线Dice（clDice）拓扑损失、分辨率等价损失、形态一致损失、眼睑解剖损失以及监督自蒸馏与阈值校准等技术。

**📊 数据集**

源域采用公开MGD-1k数据集，目标域采用CAMG 100图像进行评测，并在VISIA/Topcon MYAH→Tera 设备上验证部署。

**📈 对比分析**

与零射、UA-MT、CPS、FixMatch等SSL基线及SAM/MedSAM对比，TopoPult-SSL在CAMG测试集上Dice达到0.716±0.006（最佳0.726），优于UA-MT 0.710、CPS 0.707、教师集成 0.720，且精度显著高于SAM/MedSAM（Precision 0.694 vs 0.30-0.34）。

**⚠️ 局限性**

局限性包括：①测试集样本有限（20张）；②仅评估单一目标设备；③第一阶段仍需眼睑掩码与形态比先验，且模型选择使用验证集肢体掩码；④第二阶段需肢体掩码，无法完全无标注。

---

## 116. Would you still call this Dax? Novel Visual References in VLMs and Humans

**arXiv ID:** 2606.05409 | [PDF](https://arxiv.org/pdf/2606.05409v1)

**作者:** Ada Defne Tür `[一作]` (McGill University), Benno Krojer `[通讯]` (McGill University)

**通讯引用:** 78 | [OpenAlex ID](https://openalex.org/A5073456064)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了新的视觉参考数据集 NVRD，并通过多种提示范式评估了五种视觉语言模型（VLM）在接触新视觉概念后的学习与泛化能力，同时与人类评估者进行了对比。

**💡 创新点**

创新点在于：①首次系统化生成全新、开放式视觉参考并提供可控的形状、纹理、背景等多级扰动；②设计了三种多图提示实验（命名生成、词概率估计、Likert 评分）与人类双图 Likert 对比，以更细粒度捕捉模型对“新词”的接受与推理；③揭示了模型在与先前知识冲突时的“互斥性”失效与对形状扰动的过度泛化。

**🔧 技术方法**

主要技术包括：VLM 的多图上下文学习（in‑context learning），对生成词概率的对数归一化评估，以及基于 CLIP 的相似度池构建；实验使用了 Qwen‑2‑VL 7B、Idefics‑3 8B、Molmo‑2 8B（开源）以及 GPT‑4o Mini、Gemini‑2.5 Flash（闭源）。

**📊 数据集**

使用的数据集为自研的 Novel Visual References Dataset (NVRD)，包含 90 种视觉概念（已知、合成、全新），每种概念最多 20 级扰动，共 19,176 张图像；同时收集了 2,400 条人类 7‑级 Likert 评分。

**📈 对比分析**

比较方法：对每种模型在三种提示范式下对 NVRD 进行评估，并将模型的 Likert 评分与人类评分做 Spearman 相关性与分布对比。结果显示：模型在形状扰动下的接受度下降与人类相似，但整体上对极端扰动过度泛化，评分波动更小；对已知对象的 “互斥性” 效应更明显；在相同扰动强度下，人类的分数更低且方差更大。性能上，模型在新词学习上可行，但在与先前知识冲突时表现弱势。

**⚠️ 局限性**

局限性包括：①数据集的图像由生成式模型创建，带有生成器自身的结构与风格偏差；②部分高阶扰动在 20 级前已饱和，导致扰动范围不均；③实验人类样本有限（30 人），只覆盖 NVRD 的一部分图像；④仅评估了五种模型，未涉及更广泛的 VLM 架构。

---

## 117. Improving Heart-Focused Medical Question Answering in LLMs via Variance-Aware Rubric Rewards with GRPO

**arXiv ID:** 2606.05174 | [PDF](https://arxiv.org/pdf/2606.05174v1)

**作者:** Arash Ahmadi `[一作]` (University of Oklahoma), Mike Banad `[通讯]` (University of Oklahoma)

**通讯引用:** 751 | [OpenAlex ID](https://openalex.org/A5068910901)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文在心脏医学问答任务上，对大型语言模型进行后训练，结合GRPO和方差感知的连续奖励框架，提升模型的多标准医学回答质量。

**💡 创新点**

创新点在于提出了基于标准级别的连续奖励函数，并通过复杂度感知与混合奖励两种变体，显著缓解稀疏多标准奖励难题并实现更稳定的策略梯度学习。

**🔧 技术方法**

使用的技术包括Group Relative Policy Optimization (GRPO)、低秩适配（LoRA）+4-bit量化、LLM判别器进行标准级评估、Rubric-as-Reward框架以及自定义的奖励形状。

**📊 数据集**

训练数据来自RaR-Medicine过滤得到的心脏相关子集，评估数据则使用HealthBench的心脏相关评估集，二者均配合相应的标准化评分。

**📈 对比分析**

通过与Qwen3-14B基线、GPT‑OSS‑120B、Kimi‑K2等模型比较，GRPO（复杂度）将准确率从0.362提升至0.502、F1从0.532提升至0.668，且在单一RTX 6000 GPU上实现可落地部署，性能与大模型相当。

**⚠️ 局限性**

局限性包括仅针对心脏问答、评估仍基于自动化rubric判定未经过临床验证、训练时需大量LLM判别器调用导致耗时高、以及模型在更广泛医学领域的泛化能力尚待验证。

---

## 118. Multi-Objective Submodular Maximization with Differential Privacy

**arXiv ID:** 2606.05596 | [PDF](https://arxiv.org/pdf/2606.05596v1)

**作者:** Ting Hou `[一作]` (East China Normal University), Fan Dang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 736 | [OpenAlex ID](https://openalex.org/A5011008509)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在差分隐私（DP）约束下的基数约束多目标子模最大化（MOSM）问题，提出了两种新的DP算法并对其性能进行了理论和实验分析。

**💡 创新点**

创新点在于首次提出针对MOSM的DP算法：DP‑MultiGreedy（扩展贪心+指数机制+Laplace机制）和DP‑Bicriteria（基于Saturate的二分搜索+隐私噪声），并给出了近似保证、时间复杂度和隐私保证。

**🔧 技术方法**

核心技术包括：子模函数的指数机制（选择最大边际增益）、Laplace机制（估计函数值）、贪心迭代、二分搜索阈值、Bicriteria近似以及DP机制的组合与累积。

**📊 数据集**

实验使用了四个公开数据集：DBLP、Flickr（最大覆盖）以及FourSquare、Gowalla（设施位置），通过随机生成的多目标权重向量构造d个子模目标。

**📈 对比分析**

与非私有基准（GeneralizedGreedy、MultiGreedy、Saturate、MWU）对比，实验表明：当d=2时DP‑MultiGreedy在ε≥0.4时几乎匹配非私有最优；当d>2时DP‑Bicriteria在ε足够大时可与非私有解接近；在运行时间方面DP‑MultiGreedy快于DP‑Bicriteria，Saturate最慢。

**⚠️ 局限性**

局限性包括：仅适用于基数约束，无法直接处理更复杂的约束（如基图、背包）；对高维多目标（d>k或d大时）或低ε时隐私噪声显著影响解质量；未来需扩展至更一般的约束和改进近似因子。

---

## 119. Sequence Reconstruction for Substitution Channel: New Sufficient Conditions and Algorithms

**arXiv ID:** 2606.05454 | [PDF](https://arxiv.org/pdf/2606.05454v1)

**作者:** Chen Wang `[一作]` (Shandong University), Yiwei Zhang `[通讯]` (Shandong University)

**通讯引用:** 177771 | [OpenAlex ID](https://openalex.org/A5100381911)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的唯一序列重建框架，并给出了新的充分条件，能够在读取数小于传统Levenshtein阈值时仍实现唯一重建。

**💡 创新点**

创新点在于将读取数与读取之间的距离信息结合起来，定义了参数D(n,m,t,d)来量化读取集合的“分散程度”，从而得到比Levenshtein阈值更灵活的重建条件。

**🔧 技术方法**

使用组合数学、双计数与优化理论推导D(n,m,t,d)的解析表达式，设计了基于多数投票与布尔搜索的重建算法，并给出了复杂度分析。

**📊 数据集**

本研究为纯理论工作，未使用任何真实数据集；实验仅通过示例和概率论证明来展示条件的可行性。

**📈 对比分析**

与传统唯一重建阈值方法相比，新条件在读取数远小于阈值时仍能保证重建，且算法复杂度保持线性/多项式级别；具体数值通过示例验证了性能提升。

**⚠️ 局限性**

局限性包括：需要满足n≥m(t-⌈d/2⌉)+d的约束；在读取数大于此线性上限时D(n,m,t,d)难以精确计算；寻找触发子集的子问题在一般情况下为NP‑难，实际实现需启发式或剪枝策略。

---

## 120. Multilingual Coreference Resolution via Cycle-Consistent Machine Translation

**arXiv ID:** 2606.05444 | [PDF](https://arxiv.org/pdf/2606.05444v1)

**作者:** Adriana-Valentina Costache `[一作]` (University of Bucharest), Radu Tudor Ionescu `[通讯]` (University of Bucharest)

**通讯引用:** 8440 | [OpenAlex ID](https://openalex.org/A5081017623)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种利用英语到低资源语言的机器翻译与回译，结合BERTScore循环一致性权重的核心ference解析训练框架。

**💡 创新点**

创新点在于将MT循环一致性作为样本权重引入损失函数，既可在已有资源不足时生成训练数据，又能通过权重抑制翻译噪声。

**🔧 技术方法**

技术上使用Claude Sonnet 4.6做零样本翻译，mmBERT‑base作为多语言编码器，Maverick核心ference模型，并用BERTScore评估回译相似度。

**📊 数据集**

使用的数据集包括英语OntoNotes 5.0（用于翻译），法国ANCOR、匈牙利SzegedKoref、俄语RuCor以及人工校验的罗马尼亚测试集。

**📈 对比分析**

与基线、仅增广、增广+权重以及零样本对比实验表明，MT加权框架在四种低资源语言中显著提升CoNLL F1，尤其罗马尼亚语达到最高。

**⚠️ 局限性**

局限性在于对高能耗LLM的依赖、潜在的偏见传递以及仅在翻译阶段使用LLM，且实验仅覆盖有限的语言。

---

## 121. Less is MoE: Trimming Experts in Domain-Specialist Language Models

**arXiv ID:** 2606.05538 | [PDF](https://arxiv.org/pdf/2606.05538v1)

**作者:** Haoze He `[一作]` (Carnegie Mellon University), Heather Miller `[通讯]` (Carnegie Mellon University)

**通讯引用:** 995 | [OpenAlex ID](https://openalex.org/A5021433095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于经验 Fisher 信息的 Mixture‑of‑Experts（MoE）模型压缩方法 Fisher‑MoE，主要通过在每个专家的 FFN 中按 Fisher 重要性筛选并删除中间维度，实现对模型参数的细粒度剪枝。

**💡 创新点**

创新点在于：①首次将 Fisher 重要性引入 MoE 维度级别的压缩，证明其比激活频率、路由分数和权重幅值等传统指标更能捕捉任务关键参数；②揭示 MoE 能力并非集中于某些专家，而是分布在专家内部的极少数中间维度，因而采用维度级剪枝而非专家级删除能更好保留模型性能；③通过实验展示该方法在保持性能的同时显著提升推理吞吐量（≈21%）并减少内存占用（≈45%）。

**🔧 技术方法**

技术手段包括：经验 Fisher 信息计算、MoE 模型结构、细粒度中间维度剪枝、与 4‑bit AWQ 量化的兼容性实验，以及在不同 MoE 大小（2.7B–35B）和任务（数学推理、代码生成、知识问答、多语言理解）上的评估。

**📊 数据集**

主要数据集与基准：Qwen1.5‑MoE、OLMoE、Qwen3 系列模型；任务集合包括数学推理（MATH、GSM8K、MultiArith）、代码生成（HumanEval、MBPP）、知识问答（MMLU、CEval、CMMLU、BBH）、多语言（AIME、Olympiad、GPQA‑D）以及长链式推理（CoT）。校准数据使用 128 条样本的 GSM8K 训练集。

**📈 对比分析**

与基线（激活、路由、幅值、专家级 Fisher、专家级剪枝）相比，Fisher‑MoE 在 50% 的 MoE 压缩率下保持或提升了大多数任务的准确率，尤其在生成和数学推理任务上表现最为稳健；在推理吞吐量上提升 21%，内存占用下降 43‑48%，总体模型参数下降约 45%。实验表明维度级剪枝优于专家级剪枝，且可与 4‑bit AWQ 量化叠加，实现更大程度的部署压缩。

**⚠️ 局限性**

局限性：实验仅覆盖至 35B 参数规模的 MoE 模型，未对 100B+ 模型进行验证；未进行更长时间的后训练恢复实验，无法完全评估压缩后模型与原始模型性能的差距；且压缩效果对不同校准数据域的泛化能力仍需进一步探究。

---

## 122. X-Band UAV-enabled Integrated Sensing and Communications for Vehicular Networks

**arXiv ID:** 2606.05262 | [PDF](https://arxiv.org/pdf/2606.05262v1)

**作者:** Remon Polus `[一作]` (Polytechnique Montréal), Soumaya Cherkaoui `[通讯]` (Polytechnique Montréal)

**通讯引用:** 4030 | [OpenAlex ID](https://openalex.org/A5011549008)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在X波段为车辆网络设计并优化了UAV支持的集成感知与通信（ISaC）系统的时间分配方案，平衡感知精度与通信性能。

**💡 创新点**

首次将单阴影（SS）与双阴影（DS）信道模型与ISaC时间分配结合，提出在满足最小通信速率和雷达SNR约束下的闭式最优时间分配，并通过解析推导验证其凸性与最优性。

**🔧 技术方法**

采用Shannon容量定理、Nakagami‑m 与逆伽马分布的联合信道建模、Meijer G函数求解、凸优化理论以及MATLAB仿真进行分析。

**📊 数据集**

使用基于典型参数的仿真数据集（如P_c=30 dBm、P_s=40 dBm、B_c=180 kHz、B_s=45 MHz、路径损耗指数η=2、雷达截面σ=20 m²等），模拟不同距离、阴影强度、功率和速率阈值场景。

**📈 对比分析**

与等时分配策略对比，结果显示在低至中等通信功率范围内（0–35 dBm）最优分配显著提升功率效率（bits/W），并在不同信道条件下实现更高的感知/通信吞吐量；在高功率区性能差异趋于平缓。

**⚠️ 局限性**

主要限制包括：仅考虑单一UAV与车辆对偶，没有考虑多UAV协同或干扰；信道模型虽更现实，但仍假设统计分布已知；仿真基于参数推导，缺乏真实测量验证。

---

## 123. SentinelBench: A Benchmark for Long-Running Monitoring Agents

**arXiv ID:** 2606.05342 | [PDF](https://arxiv.org/pdf/2606.05342v1)

**作者:** Matheus Kunzler Maldaner `[一作]` (University of Florida), Saleema Amershi `[通讯]` (Microsoft Research AI Frontiers)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套名为 Sentinel 的基准，用于评估 AI 代理在外部时间变化的监控任务中的表现，包括任务完成率、反应时间和资源消耗。

**💡 创新点**

创新点在于：1）专门针对需要等待事件触发的监控任务，而非传统连续行动任务；2）设计多种环境和任务类型（被动、主动、无操作）；3）通过可插拔工具（sleep vs watch）量化工具设计对成本和反应时间的影响；4）提供完整的数据生成管道和评估协议。

**🔧 技术方法**

使用了多模态大语言模型（GPT‑5.4、GPT‑4o、Qwen 3.5:9B）与浏览器代理工具，结合自定义的 watch 工具和 sleep 工具；评估时使用基于 FastAPI 的环境仿真和 SQL 查询校验。

**📊 数据集**

数据集由 100 个任务组成，覆盖 10 个合成 web 环境（Email、Calendar、Finance、Social 等），任务与事件由生成流水线基于 100 个虚拟人物和实体产生的合成数据构建；包括事件时间、用户信息、页面内容等。

**📈 对比分析**

比较方法：在 6 种条件（3 模型 × 2 工具）下记录任务成功率、平均 API 成本、平均反应时间；实验显示 GPT‑5.4 在所有指标上优于其他模型，watch 工具在成本上明显优于 sleep 工具，并在长任务（40 分钟）时表现更佳；反应时间表现因模型与工具组合而异。

**⚠️ 局限性**

局限性：1）事件时间是人工设定，缺乏真实分布；2）环境是轻量级仿真，可能不覆盖所有真实交互；3）任务多为客观量化目标，缺乏主观判断和短暂事件；4）需要进一步自动化任务生成与错误检测；5）对训练代理的支持有限，时间压缩仍有技术难点。

---

## 124. BIDENT: Heterogeneous Operator-level Mapping for Efficient Edge Inference

**arXiv ID:** 2606.05271 | [PDF](https://arxiv.org/pdf/2606.05271v1)

**作者:** Hoseok Kim `[一作]` (Purdue University), Vijay Raghunathan `[通讯]` (Purdue University)

**通讯引用:** 7081 | [OpenAlex ID](https://openalex.org/A5102009759)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

提出了一个面向边缘 SoC 的统一算子级调度框架，将各算子映射到最合适的 CPU/GPU/NPU 上，实现跨算子、跨模型的并行与能耗优化。

**💡 创新点**

核心创新是将算子-PU 分配建模为加权有向图的最短路问题，既支持单模型顺序调度，又兼容算子级并行与多模型协同，且具有最优性保证。

**🔧 技术方法**

采用在线离线两阶段工作流：离线算子性能剖析、统一执行图构建和 Dijkstra 最短路搜索；同时通过经验交叉PU慢速因子模型考虑内存带宽争用。

**📊 数据集**

在 Intel Core Ultra SoC 上评估了 10 种不同架构（CNN、Transformer、SSM、KAN、Hyena、spiking、VLA 等）共 19 个 FP16/INT8 配置，以及 190 对多模型组合。

**📈 对比分析**

与单 PU（CPU/GPU/NPU）基线对比，算子级调度平均提升 1.09× 延迟（单模型），1.60× 内部并行，3.42× 多模型并行；能耗在并行场景下降幅度高达 48.2%。

**⚠️ 局限性**

局限性包括：调度为静态离线预映射，无法应对运行时热量、内存争用波动与动态模型流；且仅考虑了统一内存体系，未细化到 NPU/CPU 内部资源（如 tile）层面的细粒度映射。

---

## 125. The Evaluation Blind Spot: A Stereological Theory of Benchmark Coverage for Large Language Models

**arXiv ID:** 2606.05169 | [PDF](https://arxiv.org/pdf/2606.05169v1)

**作者:** Jason Z Wang `[一作]` `[通讯]` (Independent), Jason Z Wang (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LLM评测的几何盲区理论，阐明有限维度基准如何导致模型排名不可区分，并给出基准选择的子模优化算法；

**💡 创新点**

创新点在于量化可见盲区的 Hausdorff 上界、引入“可见效维”概念、解决 Gardner 问题的最优恢复率、以及证明子模贪心算法在覆盖率上的 (1-1/e) 最优性；

**🔧 技术方法**

使用了立体几何、支撑函数分析、参与度比（effective dimensionality）计算、子模最大化、Schur‑convex性与 Jackson‑Bernstein 逼近理论；

**📊 数据集**

实验涵盖 Open LLM v2、Extended 12‑benchmark suite、LiveBench 三大排行榜以及 27 个 Chatbot Arena 类别，数据量从 37 个模型到 4,576 个模型；

**📈 对比分析**

通过与半分割实验、排名交换概率、覆盖率指标对比，证明贪心子集能在 7‑12 个基准内实现 90%+ 覆盖率，并将盲区大小与实际排名不确定性关联，性能明显优于随机或传统多样性指标；

**⚠️ 局限性**

局限包括对凸形能力假设的依赖、对基准线性可线化的要求、在极低效维（d_eff≈1）下效果退化、以及对模型内部结构和非凸评价方法未做深入探讨。

---

## 126. Disentangled Fine-Grained Prototype Learning for Incomplete Image-Tabular Classification

**arXiv ID:** 2606.05455 | [PDF](https://arxiv.org/pdf/2606.05455v1)

**作者:** Feixiang Zhou `[一作]` (University of Liverpool), Yalin Zheng `[通讯]` (University of Liverpool)

**通讯引用:** 10775 | [OpenAlex ID](https://openalex.org/A5081186911)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了DFPL框架，结合共享-特定原型建模、细粒度对齐与类感知多尺度聚合，实现缺失模态下的图像-表格分类。

**💡 创新点**

创新点：在原型层面进行共享与特定分离、使用最优传输实现细粒度分布对齐、类级语义对齐，并通过可学习类查询进行多尺度融合。

**🔧 技术方法**

技术：原型学习、注意力机制、最优传输（Sinkhorn）、分布对齐、类查询自注意力。

**📊 数据集**

数据集：MIMIC（ICU图像+时间序列表格）、ADNI（MRI+临床表格）、DVM（车辆图像+属性表格）。

**📈 对比分析**

与多种基线比较（TIP、IF-MMIN、ShaSpec、IM-Fuse、STiL、DMRNet、DrFuse），在不同缺失率下均取得最高PRAUC/ACC，显著提升。

**⚠️ 局限性**

局限：仅利用配对样本进行对齐，未充分利用未配对数据；仅针对图像-表格，其他模态组合待扩展。

---

## 127. Self-Commitment Latency: A Reward-Free Probe for Prompted Implicit Hacking

**arXiv ID:** 2606.05625 | [PDF](https://arxiv.org/pdf/2606.05625v1)

**作者:** Bonan Shen `[一作]` (Independent Researcher), Tao Ning `[通讯]` (Syracuse University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种无奖励的前缀探测方法——自我承诺延迟，用于检测提示式隐式奖励黑客；

**💡 创新点**

创新点在于用模型自身最终答案作为参照，构建自我承诺曲线，避免需要外部奖励模型或验证器；

**🔧 技术方法**

技术包括全链生成、截断前缀、强制答案采样、计算自我承诺曲线及其摘要指标（τ_first、range_c、mean uncommitted 等）；

**📊 数据集**

使用 GSM8K 数学题的前 50 题，分别在普通提示和加入答案提示的条件下生成 100 条 CoT；

**📈 对比分析**

与提示加答案的情形对比，采用 AUROC、paired wins 等统计；在所有样本上 τ_first(0.8) AUROC 为 0.878、43/50 paired wins；在仅两种条件都正确的子集上 AUROC 升至 0.931、36/40 wins，指标表现稳健；

**⚠️ 局限性**

局限性包括：只在小型 3B 模型和整数答案任务上验证，采样量有限导致曲线粗糙；未验证对更大模型或非整数答案任务的泛化；方法仍比普通推理成本高，难以用于实时监控。

---

## 128. The Virtual Roundtable: Multi-Agent Personas Simulating the Dynamics of Human Brainstorming

**arXiv ID:** 2606.05178 | [PDF](https://arxiv.org/pdf/2606.05178v1)

**作者:** Tim Dorn `[一作]` (Meta), Julie Mumford `[通讯]` (Meta)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个多智能体协作的虚拟圆桌式头脑风暴系统，利用多阶段（讨论、创意、投票）流程让 AI 角色在虚拟会议中生成并评估产品概念。

**💡 创新点**

创新点在于将讨论与创意分离并通过专用的协同促进者动态调控，结合私密思考、公共评论、创意提交及投票的层级可视化，并记录完整的创意血缘链，从而模拟人类头脑风暴的“无评判、数量优先、互相借鉴”原则。

**🔧 技术方法**

核心技术包括大型语言模型（LLM）角色扮演、事件总线驱动的异步通信、基于相似度的创意去重、词向量相似度计算、以及多轮私密思考与公共发言的层级架构。

**📊 数据集**

实验数据主要来自 13 个人格角色的自我生成（无外部标注集），以及在 AI 智能眼镜产品概念任务上的内部案例研究（共约 10 分钟的头脑风暴过程）。

**📈 对比分析**

通过与不同讨论时长（0–25 分钟）的 34 次实验对比，评估指标包括“影响深度”“跨角色吸收率”和“语义多样性”，发现讨论时长显著提升影响深度与吸收率，但对概念多样性影响有限；在案例中生成的创意在质量上优于单一角色生成，且投票机制能自然排序。

**⚠️ 局限性**

主要局限包括对角色生成质量与多样性的依赖、缺乏真实世界数据的检索与检验、受限于 LLM 的幻觉倾向、以及创意多样性受角色组合约束而非讨论时间决定。

---

## 129. LoRi: Low-Rank Distillation for Implicit Reasoning

**arXiv ID:** 2606.05315 | [PDF](https://arxiv.org/pdf/2606.05315v1)

**作者:** Ryan Solgi `[一作]` (University of California-Santa Barbara), Zheng Zhang `[通讯]` (University of California-Santa Barbara)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种低秩隐式链式思考（LoRi）蒸馏框架，将教师模型的长链式推理轨迹压缩到低维子空间并指导学生模型的隐式推理过程。

**💡 创新点**

创新点在于利用教师隐藏状态的低秩张量分解与统计匹配，捕捉全局推理结构，而不依赖逐步对齐或中间监督，从而实现长度不变、高效的知识迁移。

**🔧 技术方法**

主要技术包括Tucker分解、SVD低秩因子提取、均值与协方差统计匹配、两阶段预计算低秩子空间与学生微调。

**📊 数据集**

使用数学推理基准数据集GSM8K、GSM8K‑Hard以及SVAMP进行评估。

**📈 对比分析**

与SFT-CoT、NoCoT、CODI、KAVA、PCCoT、SIM‑CoT等先前iCoT方法对比，LoRi在多模型、多规模上均提升≈12%准确率，尤其在GSM8K‑Hard上显著超越对手，且在保持低推理延迟的同时逼近显式CoT性能。

**⚠️ 局限性**

局限性在于对低秩结构与推理能力之间理论关系尚未完全阐明，方法依赖于经验性低秩假设，可能对不同任务或模型的泛化性存在未知风险。

---

## 130. Epidemiology of Model Collapse: Modeling Synthetic Data Contamination via Bilayer SIR Dynamics

**arXiv ID:** 2606.05168 | [PDF](https://arxiv.org/pdf/2606.05168v1)

**作者:** Xiangyu Wang `[一作]` `[通讯]`, Xiangyu Wang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

建立了 AI 生态污染的双层 SIR/SIRS 模型，并在理论、仿真及 GPT‑2 实验中验证其阈值行为和干预效果。

**💡 创新点**

创新点在于将跨模型污染视为流行病传播，提出双层 SIR 结构并给出基本再生产数、阈值分析、敏感性与干预策略；同时首次将该框架与实际 LLM 递归训练实验相结合。

**🔧 技术方法**

使用流行病学的 SIR/SIRS ODE、Next Generation Matrix 计算 R0、Sobol 全局敏感性、粒子群 ABM 验证、GPT‑2 递归训练实验以及匹配预算多源实验等技术。

**📊 数据集**

使用公开 AI 文本流行度数据、GPT‑2 124M、WikiText‑103、Tiny Shakespeare 以及 1,088 次匹配预算多源实验的数据集。

**📈 对比分析**

通过将实验结果与理论阈值和 ODE 预测对齐进行比较，发现大多数情景 R0>1；实验显示在全污染下 PPL 显著上升、Distinct‑2 降低，说明模型崩溃现象符合阈值预测。

**⚠️ 局限性**

局限性包括：模型为经验性且对网络异质性敏感；仅在 GPT‑2 小模型上验证，缺乏大模型或持续预训练的实验；参数取值范围为假设，未在实际生态中测量；干预效果的统计显著性在部分实验中仅为边缘。

---

## 131. Online Min-Cost Matching with General Arrivals

**arXiv ID:** 2606.05546 | [PDF](https://arxiv.org/pdf/2606.05546v1)

**作者:** Josh Ascher `[一作]` (Drexel University), Vasilis Gkatzelis `[通讯]` (Drexel University)

**通讯引用:** 3199 | [OpenAlex ID](https://openalex.org/A5057144202)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在所有请求都在线出现的线性度量空间下的最小成本完美匹配问题，提出在未知 i.i.d. 入参模型下实现 O(log²n) 竞争比的算法，并证明在随机顺序模型下竞争比不可界定；同时给出双侧到达时的常数竞争比与 (logloglog n)² 竞争比的结果。

**💡 创新点**

首次在未知 i.i.d. 与随机顺序两种模型之间证明竞争比的严格分离，提出只使用序数信息的分桶+阶段式匹配框架，能在未知分布下通过学习样本获得近似均匀划分；并给出双侧到达的减价方法。

**🔧 技术方法**

采用分桶学习、阶段合并、概率与超几何分布的尾界技术、随机化与顺序对称性分析、以及贪心匹配结构，构建仅依赖请求相对顺序的在线算法。

**📊 数据集**

无实验数据集，所有结果均为理论上界和概率论证明，主要以均匀分布、随机顺序构造极例作为验证。

**📈 对比分析**

与传统的一侧到达模型相比，未知 i.i.d. 的 O(log²n) 竞争比已超过当前已知的 Ω(log n) 下界；随机顺序模型表现为无界竞争比，体现了两模型的本质差异；双侧到达时在线性度量空间可达常数竞争比。

**⚠️ 局限性**

局限在于主要结果仅适用于线性度量空间，且在未知 i.i.d. 情况下仍需较大样本学习；对一般度量空间的竞争比仍为 (logloglog n)²，距离最优 log n 仍有显著差距；随机顺序模型的不可分辨性结果仅给出无界竞争比而非具体增长速率。

---

## 132. ColBERTSaR: Sparsified ColBERT Index via Product Quantization

**arXiv ID:** 2606.05568 | [PDF](https://arxiv.org/pdf/2606.05568v1)

**作者:** Eugene Yang `[一作]` (Johns Hopkins University), Rohan Jha `[通讯]` (Johns Hopkins University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种稀疏化的ColBERT检索模型（Sparsified ColBERT via Product Quantization），将ColBERT的密集向量索引转化为稀疏倒排索引，显著压缩索引尺寸；

**💡 创新点**

创新点在于引入残差无关的产品量化与查询感知的锚点优化，使得ColBERT可视作一种学习式稀疏检索模型，同时不需要重新训练模型；

**🔧 技术方法**

核心技术包括K‑means聚类生成锚点、产品量化（无残差）、查询感知与无监督锚点优化、前向索引与倒排索引结合的双阶段评分；

**📊 数据集**

实验使用BEIR、NeuCLIRBench和NeuCLIRTech三大检索基准，覆盖单语、跨语言和多语言检索场景；

**📈 对比分析**

与PLAID（1‑bit残差）以及BM25、SPLADEv3、MILCO等方法比较，Sparsified ColBERT在索引尺寸上比PLAID小约30‑50%，在nDCG@10/20上保持92‑95%与PLAID的效果，整体性能相当；

**⚠️ 局限性**

局限性包括：仍需工程化优化（如位压缩、最大化运算加速）；对专业术语和跨语言查询的适配仍有挑战；检索延迟分析缺失，需进一步实测。

---

## 133. Representation Learning Enables Scalable Multitask Deep Reinforcement Learning

**arXiv ID:** 2606.05555 | [PDF](https://arxiv.org/pdf/2606.05555v1)

**作者:** Johan Obando-Ceron `[一作]` (Mila - Quebec AI Institute), Pablo Samuel Castro `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在多任务强化学习场景中，作者提出并评估了一种模型自由算法 MR.Q，该算法通过在 actor‑critic 结构中加入预测性辅助任务（预测下一个状态、奖励和终止信号），从而学习高质量的潜在表示；该方法在不进行规划的前提下实现了可扩展的多任务性能。

**💡 创新点**

证明了模型自由与预测式表示学习可以匹敌甚至超过传统的世界模型方法，关键创新点在于将预测目标仅用于提升表示质量而非规划，从而大幅降低计算开销并提升样本与时间效率。

**🔧 技术方法**

采用了 TD3 基础的 actor‑critic 框架，并在其上叠加多任务语言条件编码；使用多任务辅助目标（动力学、奖励、终止）进行表示学习；在视觉任务中结合冻结的 DINOv2 编码器；实验采用了 10M 交互、不同模型规模和更新‑数据比例的评估。

**📊 数据集**

使用 MMBench（多任务连续控制、操纵、步态、游戏）作为主要数据集；在 200 任务综合集、28 个未见任务、以及使用 DINOv2 编码器的视觉输入上进行评估。

**📈 对比分析**

与最近的世界模型基线 Newt 以及多种深 RL 基线进行对比。实验结果显示，MR.Q 在 10M 步样本效率、最终性能、对未见任务的零样本表现、少量微调速度以及 wall‑clock 时间上均优于 Newt；在所有评估维度上都展现出更高的性能和更快的收敛。

**⚠️ 局限性**

主要局限在于评估集中于连续控制任务，缺乏对长时程或更复杂域的验证；对预测目标对表示学习的具体机制尚未得到充分理论解释；未探讨与规划相结合的混合策略可能带来的进一步提升。

---

## 134. Towards Persistent Case-Based Memory for Autonomous Data Science: A CBR-Augmented R&D-Agent with a Locally Deployable Small Language Model

**arXiv ID:** 2606.05250 | [PDF](https://arxiv.org/pdf/2606.05250v1)

**作者:** Felix Stocker `[一作]` `[通讯]` (Technische Hochschule Ingolstadt), Felix Stocker (Technische Hochschule Ingolstadt)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在R&D-Agent框架中加入持久化、质量控制的案例推理（CBR）层，并将Gemma 4 31B Dense作为本地可部署的LLM后端，构建了一个可跨会话、可追溯的自主数据科学代理。

**💡 创新点**

创新点在于：①将结构化符号案例与可执行代码结合形成可检索、可维护的案例库；②通过五门质量过滤和启发式重用检测机制实现高质量、无冗余的案例保留；③将开源小模型Gemma 4通过定制适配实现完整的agent功能，证明SML可替代云端大模型。

**🔧 技术方法**

使用的技术包括Gemma 4 31B Dense、R&D-Agent框架、向量检索（embedding）、代码指纹相似度、Google AI Studio定制API、prompt-based结构化输出修复、Hang检测和自适应超时。

**📊 数据集**

实验数据集为Kaggle的NOMAD 2018（回归）和Spaceship Titanic（分类）两个竞赛，分别作为回归和分类任务进行评测。

**📈 对比分析**

通过与CBR关闭的基线进行A/B比较，使用八轮自主改进循环；在Spaceship Titanic上，CBR提升了准确率至0.8147（比基线高约1.4%），并显著降低了方差；在NOMAD上，CBR未显著提升最终指标，但保持了更低的方差并提高了保留率，整体显示CBR在稳定性和保留率方面具有优势。

**⚠️ 局限性**

限制包括：实验仅覆盖两项竞赛、每组四个随机种子，缺乏跨赛题迁移评估；未对CBR检索的因果影响进行消融实验；未在GPU环境或多任务场景下验证；因此结果仅为初步探索，需进一步扩展和验证。

---

## 135. Dual Feature Decoupling for Fine-Grained OOD Detection

**arXiv ID:** 2606.05536 | [PDF](https://arxiv.org/pdf/2606.05536v1)

**作者:** Xiaokun Li `[一作]` (Beijing Jiaotong University), Qingji Guan `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 1245 | [OpenAlex ID](https://openalex.org/A5047821546)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Dual Feature Decoupling Network (DFDNet) 用于细粒度 OOD 检测，结合空间-频率解耦和对抗重建两大模块；

**💡 创新点**

将 BN 与 IN 在空间域进行可学习融合并加入 DFT 在频域去除风格信息，同时通过 CAM 引导的对抗重建剔除低级背景噪声，实现对任务相关与无关特征的双重解耦；

**🔧 技术方法**

空间-频率解耦 (SFD) 模块、离散傅里叶变换 (DFT)、重建引导解耦 (RGD) 模块、类别激活映射 (CAM)、多任务学习（分类+重建）等；

**📊 数据集**

细粒度视觉分类数据集：FGVC-Aircraft、Stanford Cars、Butterfly、North American Birds；辅助 OOD 训练集：WebVision 1.0；

**📈 对比分析**

与 MSP、ODIN、Energy、Rotation、VIM、Scale、PRO-MSP、OE、OE-M、EnergyOE、MixOE-line/ cut 等现有方法进行对比，在细粒度 OOD 检测任务中，DFDNet 在 TNR95、AUROC 等指标上平均提升 1–3%（最小 12.7% 在最难的 NA Birds 数据集上），且在不使用辅助 OOD 数据时也已超过绝大多数对比方法；

**⚠️ 局限性**

仍需手工设定 CAM 阈值 τ、λ、γ 等超参数，对不同细粒度任务的适应性需要进一步验证；计算量和训练时间相对较大；仅针对图像数据验证，未涉及文本或多模态 OOD 场景；

---

## 136. Mitigating the Curse of Dimensionality in Uniform Convergence of Deep Neural Networks via Smooth Activations

**arXiv ID:** 2606.05599 | [PDF](https://arxiv.org/pdf/2606.05599v1)

**作者:** Yizhe Ding `[一作]` (Pennsylvania State University), Lingzhou Xue `[通讯]` (Pennsylvania State University)

**通讯引用:** 2109 | [OpenAlex ID](https://openalex.org/A5089400160)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了平滑激活深度神经网络（smooth DNN）的理论框架，证明了 ReLU FNN 在均匀收敛上会出现维度灾难，并给出了 smooth DNN 在多种回归和分类任务中的非渐近 L∞ 收敛速率。

**💡 创新点**

创新点在于①首次给出 ReLU FNN 在均匀收敛上的下界，②对 Pfaffian 激活的 smooth DNN 推导了伪维数、逼近误差和 Hölder 范数上界，③利用这些结果获得 Huber、最小二乘、分位数和逻辑回归等任务的均匀收敛保证，④展示了 smooth DNN 在低维层次结构下能自适应并克服维度灾难。

**🔧 技术方法**

主要技术包括：Pfaffian 函数与伪维数分析、逼近理论（Sobolev 与层次组合模型）、Gagliardo–Nirenberg 插值、统计学习理论（非渐近误差分解）、Huber 损失与分位数损失的稳健性分析、残差网络与 C∞ 激活的组合实现。

**📊 数据集**

实验数据涵盖：1）8 维高斯噪声的模拟数据，用于评估 Huber 回归；2）美国 EPA AQS 的臭氧浓度与温度的真实数据，用于分位数回归；3）Higgs 双希格斯产生的背景与辅助分布，用于密度比估计。

**📈 对比分析**

与 ReLU FNN 进行 L² 与 L∞ 误差对比，平滑 DNN（SiLU FNN 与 SiLU ResNet）在两种误差指标下均表现出更小的误差、较高的收敛指数；在真实臭氧数据中，SiLU ResNet 的预测更平滑、极端值更少；在 Higgs 例子中，smooth DNN 的密度比估计在均匀误差上同样优于 ReLU。

**⚠️ 局限性**

局限性包括：理论结果依赖于激活函数的 Pfaffian 结构与光滑假设，常数与精确收敛速率仍未完全量化；对极端噪声、非平稳分布的鲁棒性有限；在密度比估计中需要假设信号与背景在特定子集上相似，实际可行性受限；实验验证主要聚焦于少数任务和数据集，未覆盖所有深度学习常见场景。

---

## 137. ERRORQUAKE: Heavy-Tailed Error Severity Distributions in Open-Weight Large Language Models

**arXiv ID:** 2606.05170 | [PDF](https://arxiv.org/pdf/2606.05170v1)

**作者:** Jason Z Wang `[一作]` `[通讯]` (Independent), Jason Z Wang (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在相同准确率条件下，系统性评估开放权重大型语言模型（LLM）的错误严重度分布，并提出了10,000条跨域、跨难度的查询基准（Errorquake-10k）和9级连续严重度评分体系。

**💡 创新点**

创新点在于：①证明错误率与严重度分布信息不冗余（Non‑Reducibility Theorem）并量化其信息量；②用Gutenberg–Richter指数(b)表征尾部形状；③构建严重度机制分类，揭示模型规模与严重度类型的关联；④通过人类与双评审双重验证提升评分可靠性。

**🔧 技术方法**

技术方法包括：Gutenberg–Richter拟合与Aki最大似然估计、BIC/Vuong检验、ICC、Spearman/ρ、Kappa、Bootstrap置信区间、稀疏样本校准等统计工具；评审流程采用双评审与三评者人类验证；使用Python/NumPy/Scikit‑learn等实现。

**📊 数据集**

数据集：10,000条查询（BIO、LAW、HIST、GEO、SCI、TECH、FIN、CULT），按5个难度层次分布；21个开放权重指令调优模型，参数范围约3–37B；519条人类评审样本验证评分可靠性。

**📈 对比分析**

比较方法：在人类共识评分下，85/210模型对在相同误差率（|Δε|<0.05）下显示不重叠的95%置信区间；相较传统误差率基准，严重度指数能够区分同误差率下模型的尾部风险；此外，模型规模与b值呈显著负相关（ρ≈-0.86）。

**⚠️ 局限性**

局限性包括：评审者过度报错率高（≈33%），严重度尺度需保持完整九级分辨率；仅覆盖开源模型，未评估专有或推理专用模型；人类验证样本有限，无法精确估计每模型b值；查询由LLM自动生成，可能不完全代表真实使用场景。

---

## 138. AppAgent-Claw: CLI Is All You Need for GUI Automation

**arXiv ID:** 2606.05171 | [PDF](https://arxiv.org/pdf/2606.05171v1)

**作者:** Zhixue Song `[一作]` (Westlake University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 27017 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AppAgent‑Claw，基于用户演示的记录‑注释‑重放流程，将 GUI 工作流包装为 OpenClaw 轻量级技能，实现可重复、可靠的 GUI 自动化。

**💡 创新点**

创新点在于：① 通过一次录制、一次注释、重复重放的模式保留视觉锚点、窗口上下文和文本参数；② 层级定位（局部锚点→全局上下文→相对坐标）与后置验证相结合，提供鲁棒的目标定位与错误恢复；③ 将上述流程与 OpenClaw 现有技能架构无缝集成，避免实时大模型推理带来的开销。

**🔧 技术方法**

技术包括：事件聚合与语义步骤抽象、屏幕截图与上下文裁剪、OpenCV 模板匹配、文本输入采用剪贴板复制、窗口恢复与准备、层级定位策略、后置验证与重试机制，以及结构化流定义与日志诊断。

**📊 数据集**

实验使用 5 个实际工作流（Notes 新建笔记、Safari 搜索标签、网易音乐每日推荐、网易音乐收藏歌单、微信文件传输助手），在同一台 macOS 机器上多次录制与重放；未使用公开数据集。

**📈 对比分析**

通过多轮重放（50 次）与扰动测试（暗模式、缩放、不同初始状态）进行评估。结果显示：E2E 成功率 100%，每步成功率 100%，平均时延约 20–30 秒；层级 hit 率表明当主锚点失效时，层级回退能够保持成功率；在视觉漂移条件下仍保持 100% 成功率，验证了鲁棒性。

**⚠️ 局限性**

局限性：仅适用于同机、前台、监视器与窗口布局基本不变的场景；对 UI 重设计、跨设备迁移缺乏鲁棒性；需要人工或人工+代理完成注释与参数化；无法实现开放域或完全自主的 GUI 交互。

---

## 139. Policy-Conditioned Counterfactual Credit for Verifiable Reinforcement Learning of Long-Horizon Language Agents

**arXiv ID:** 2606.05263 | [PDF](https://arxiv.org/pdf/2606.05263v1)

**作者:** Renwei Meng `[一作]` `[通讯]` (Anhui University), Renwei Meng (Anhui University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种受约束的策略梯度强化学习框架，利用稠密可验证奖励与策略条件的反事实贡献（PCCC）来提升语言代理的长时序推理性能。

**💡 创新点**

创新点在于通过PCCC估计器量化每一步对最终验证成功的因果贡献，并结合干预有效性门控、信念奖励与增广拉格朗日约束，实现可审计、可验证的信用分配。

**🔧 技术方法**

使用了冻结后向策略的反事实继续、选择调整的双重稳健估计、干预门控、贝叶斯验证器、增广拉格朗日信念约束、全词表KL投影等技术。

**📊 数据集**

在四大任务组上进行评测：长上下文问答（RULER、LongBench、LooGLE）、ALFWorld、ScienceWorld，以及WebShop/WebArena/AgentBench/ToolLLM等Web/工具任务。

**📈 对比分析**

与SFT、PPO-RLVR、TROLL、LongRLVR、RLVMR、Q-RAG及计算匹配/信息匹配基线对比，平均任务成功率从71.8%提升至78.9%，证据F1从78.9%提升至82.8%，并将黑客攻击率从7.2%降低至3.9%，所有指标均统计显著。

**⚠️ 局限性**

局限在于PCCC只给出相对冻结策略的因果贡献，无法完全反映未来学习策略的总效应；方法计算成本高，需要主动选择来平衡；依赖验证器与攻击检测，对未知攻击或错误验证器可能不鲁棒。

---

## 140. Hairpin Vortices Extraction in Turbulent Boundary Layer Flows

**arXiv ID:** 2606.05229 | [PDF](https://arxiv.org/pdf/2606.05229v1)

**作者:** Adeel Zafar `[一作]` (University of Houston), Guoning Chen `[通讯]` (University of Houston)

**通讯引用:** 3291 | [OpenAlex ID](https://openalex.org/A5055426585)

**关键词:** `38fa68f4-1c75-42bb-8d13-3b76129704e6` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种自动化提取湍流边界层毛毛尖涡旋的完整框架。

**💡 创新点**

创新点在于一次性 merge 树分割+底向重组、骨架化验证，消除了手动参数调节，显著降低欠/过分割，提高提取精度与效率。

**🔧 技术方法**

采用 λ₂ 阈值提取涡流、merge 树层化分割、vorticity 线重组、曲率骨架化、ω_y' 与 |ω| 等物理几何判据实现检测。

**📊 数据集**

在 Couette、channel 及过渡边界层（TBL）三组 DNS 数据集上进行实验验证。

**📈 对比分析**

与两种先前方法对比，F1 分数提升至约 0.85–0.88，计算时间和段落数显著下降，误检率显著降低。

**⚠️ 局限性**

仍存在假阳性/阴性、残留欠/过分割，以及在极端高雷诺数或复杂涡旋交互场景下鲁棒性有限。

---

## 141. REGEN: Reference-Guided Synthetic Multivariate Time Series Generation for Forecasting

**arXiv ID:** 2606.05264 | [PDF](https://arxiv.org/pdf/2606.05264v1)

**作者:** Moulik Gupta `[一作]` (Birla AI Labs), Saurabh Deshpande `[通讯]` (Birla AI Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 ReGeN 参考引导生成管线，利用少量真实多变量序列拆分为相位对齐周期模板、深核高斯过程残差和结构因果模型来合成新序列。

**💡 创新点**

创新点在于将所有生成组件都严格基于真实参考数据，并通过可控温度采样明确建模周期结构、局部不确定性与跨变量耦合，避免了先验无关或黑盒生成方法的缺陷。

**🔧 技术方法**

使用技术包括相位对齐周期模板提取、深核高斯过程残差建模、基于 DAG 的结构因果混合、温度调控采样以及残差滤波等。

**📊 数据集**

实验数据集涵盖十二个真实多/单变量时间序列，涉及能源、云基础设施、交通、气候、住宅电力等五个领域。

**📈 对比分析**

在 TRTR、TSTR、TRSTR 三种训练评估协议下，与 TimeGAN、CauKer 等对手比较，ReGeN 在约 2/3 的迁移设置下仅与真实数据相差 ≤3% MSE，强周期域甚至优于真实数据；合成数据与真实数据结合可进一步提升多种预测模型的性能。

**⚠️ 局限性**

局限性包括：结构因果混合在低样本高维场景下效果不一；评估仅对 Moirai-small 进行全语料预训练；组件消融分析仅针对 iTransformer，未验证其对所有模型的普适性。

---

## 142. Agentic Monte Carlo: Simulating Reinforcement Learning for Black-Box Agents

**arXiv ID:** 2606.05296 | [PDF](https://arxiv.org/pdf/2606.05296v1)

**作者:** Dae Yon Hwang `[一作]` (Layer 6 AI), Brendan Leigh Ross `[通讯]` (Layer 6 AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Agentic Monte Carlo (AMC) 方法，利用Sequential Monte Carlo对黑盒LLM代理进行无梯度的强化学习优化，通过学习价值函数引导采样获得最优策略。

**💡 创新点**

创新点在于把RL转化为贝叶斯推断，用SMA与学习的价值函数实现对黑盒代理的后验采样，避免梯度计算，兼容仅有API访问的模型，并在多步交互环境中实现有效的策略改进。

**🔧 技术方法**

使用技术包括贝叶斯推断、Sequential Monte Carlo（Importance Resampling）、价值函数学习（Transformer回归头）、ReAct/ReflAct提示策略、AgentGym benchmark。

**📊 数据集**

评估数据集为AgentGym的三大环境：WebShop、SciWorld、TextCraft；此外还进行了Weather和Movie等补充实验。

**📈 对比分析**

与Best‑of‑N、SMC(FoA)、GRPO等基线对比，AMC在所有环境中均取得更高奖励，且在足够轨迹数下可与GRPO相当甚至超过；同时在成本和推理时间上优于GRPO，能让小模型匹敌大模型。

**⚠️ 局限性**

主要限制包括：价值函数近似误差导致采样偏差；需要生成多条轨迹，推理成本相对较高；在先验模型已非常优秀时提升有限；重采样策略和轨迹数等超参数仍需手工调优；理论上对误差来源的分析尚不充分。

---

## 143. RH+: Row-Hit-Optimized Scheduling for PIM-based LLM Inference

**arXiv ID:** 2606.05511 | [PDF](https://arxiv.org/pdf/2606.05511v1)

**作者:** Yongchan Jung `[一作]` (Fairleigh Dickinson University), Jeeho Ryoo `[通讯]` (Fairleigh Dickinson University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5111062876)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过重新设计HBM3‑PIM中的地址映射，提出了一种名为RH+的调度策略，显著提升了LLM推理中GEMV操作的行访问效率，从而加速整体推理流程。

**💡 创新点**

创新点在于：① 识别到行周期时间tRC才是GEMV瓶颈，而非先前关注的功率约束tPC；② 通过将MAC_AB的地址步长从64列改为1列，使连续32个MAC指令在同一行内完成，几乎消除ACT/PRE开销；③ 在保持计算结果不变的前提下完成一次离线权重重排。

**🔧 技术方法**

使用了周期精确的Ramulator 2.0模拟器、AttAcc框架以及DRAMPower能耗模型，对HBM3‑PIM的时序与能耗进行全面评估，并通过地址重排实现行命中率提升。

**📊 数据集**

评估基准为四种主流LLM模型（GPT‑175B、LLaMA‑65B、Megatron‑Turing‑530B、OPT‑66B），在不同输入/输出序列长度与批量大小（1/4）下进行实验。

**📈 对比分析**

采用与baseline相同的硬件规格（HBM3 5.2 Gbps）进行对比，RH+实现了8.25×–11.88×的速度提升，能耗降低74.5%–77.1%，EDP提升32.4×–52.0×。

**⚠️ 局限性**

限制包括：需要一次离线权重重排，适用于单卡推理；未针对注意力层或多核并行进行优化；在功率无约束模式下，进一步提升有限。

---

## 144. Personal AI Agent for Camera Roll VQA

**arXiv ID:** 2606.05275 | [PDF](https://arxiv.org/pdf/2606.05275v1)

**作者:** Thao Nguyen `[一作]` (University of Wisconsin-Madison), Yuheng Li `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了个人相机滚动的视觉问答数据集Camroll，并提出了专门针对该场景的对话式AI代理Camroll‑Agent；

**💡 创新点**

创新点在于（1）将相机滚动视为长期个性化视觉记忆任务；（2）设计三层层次化记忆（像素→描述→事件）与五种高效工具；（3）通过工具调用实现迭代检索与推理，显著提升长文本/视觉长上下文推理能力；

**🔧 技术方法**

使用的技术包括：多模态大型语言模型（如Gemini‑2.5‑Flash、GPT‑4o）、文本检索与向量检索（BM25、FAISS）、ReAct框架、层次化数据库结构（SQLite）以及可编程工具接口；

**📊 数据集**

数据集为Camroll，包含50位真实用户的31,476张图像和2,500对QA（semantic+episodic），来源包括YFCC‑100M与自家用户采集；

**📈 对比分析**

与多类基线（裸LLM、RAG、内存层、通用代理ClaudeCode）对比实验，Camroll‑Agent在多选题精度88.5%、自由回答评判83.1、Token消耗仅4.11k，明显优于所有基线；

**⚠️ 局限性**

局限性：需要人工构建记忆与标注，数据规模有限；仅覆盖图片模态，未考虑音频/视频等；对大规模用户或实时更新的相机滚动可能需进一步优化；隐私与安全性仍需深入研究。

---

## 145. Generalized TV--$\ell_p$ Structured Priors for Bayesian $T_1$ Mapping

**arXiv ID:** 2606.05381 | [PDF](https://arxiv.org/pdf/2606.05381v1)

**作者:** Disi Lin `[一作]` (Umeå University), Tommy Löfstedt `[通讯]` (Umeå University)

**通讯引用:** 1322 | [OpenAlex ID](https://openalex.org/A5035857033)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了基于总变差与 ℓ_p 范数构造的可归一化贝叶斯先验，并将其应用于磁共振 T1 映射的参数估计。

**💡 创新点**

提出了新的 TV–ℓ_p 先验，使得先验可归一化并兼具 TV 的空间平滑性，同时提供灵活的 ℓ_p 控制，从而实现更可靠的估计与不确定性量化。

**🔧 技术方法**

采用贝叶斯建模、总变差正则化、ℓ_p 范数、NUTS 采样以及基于 MLE 的对照方法。

**📊 数据集**

使用合成脑与心脏 T1 映射数据以及公开的 QIN‑BREAST‑02 乳腺 MRI 数据。

**📈 对比分析**

与 MLE、均匀、Gamma、边界 TV 等基线方法比较，TV–ℓ_p 先验在所有数据集上显著降低了估计方差和偏差，后验分布更紧凑，表现出更好的不确定性量化。

**⚠️ 局限性**

局限在于对超参数 λ、μ 的选择仍依赖贝叶斯优化，且在高度异质组织区可能出现多模态后验，未来需进一步改进模型的适应性与计算效率。

---

## 146. ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time?

**arXiv ID:** 2606.05553 | [PDF](https://arxiv.org/pdf/2606.05553v1)

**作者:** Woojung Song `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 4683 | [OpenAlex ID](https://openalex.org/A5016844435)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了自动化基准Arc‑Aware Narrative Evaluation，用以评估角色扮演语言模型在随故事进展而演化的角色心理变化上的表现。

**💡 创新点**

创新点在于将角色分段为心理轴的不同阶段，并设计跨情境（场景内、世界内、世界外）探测器和轨迹保真度指标，超越以往静态特征或事实回忆的评测方式。

**🔧 技术方法**

使用大语言模型管道完成角色弧线构建、探测器生成，并通过SFT与DPO微调提升模型对弧线的跟随能力；评测时结合RAG、LifeChoice、TimeCHARA等多种上下文策略及LLM判定器。

**📊 数据集**

基于17本公开域小说、80位主角，构建544条角色弧线与4,601条探测器，分为训练集、验证集与低受欢迎度子集。

**📈 对比分析**

在六种模型与六种上下文模式下对比，Arc上下文始终获得最高分，整体得分提升最高可达8.4分，尤其在世界外情境中提升高达7.7分；微调模型进一步扩大优势。

**⚠️ 局限性**

局限性包括仅英文、仅小说域、聚焦单角色演化、不考虑用户-角色或角色-角色交互，且可能带来历史时期偏见，仅评估静态回答。

---

## 147. ComplexityMT: Benchmarking the Interaction Between Text Complexity and Machine Translation

**arXiv ID:** 2606.05421 | [PDF](https://arxiv.org/pdf/2606.05421v1)

**作者:** Joseph Marvin Imperial `[一作]` (University of Bath), Harish Tayyar Madabushi `[通讯]` (University of Bath)

**通讯引用:** 490 | [OpenAlex ID](https://openalex.org/A5070941491)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ComplexityMT 框架，评估文本复杂度（以 CEFR 级别为衡量）与机器翻译的交互关系；

**💡 创新点**

首次同时考察翻译质量与 CEFR 级别保持的双重维度，并证明两者统计独立；

**🔧 技术方法**

使用 Spearman 相关、COMET 与 GEMBA 质量指标、基于回译的 CEFR 级别偏移评估，辅以多模型（GPT‑5.4、Tower‑7B、TranslateGemma‑4B/12B、Google Translate）进行实验；

**📊 数据集**

基于 UniversalCEFR 数据集（含 1,000 条句子级与 515 条文档级 CEFR 标注文本，覆盖英语、法语、荷兰语、阿拉伯语、印地语、俄语）；

**📈 对比分析**

结果显示：高 CEFR 文本翻译质量显著下降（Spearman ρ ≈ –0.3~–0.5），而文档级翻译往往降低 CEFR 级别（平均 Δℓ ≈ –0.2），两指标之间相关性近零，表明质量与复杂度保持是独立的；

**⚠️ 局限性**

局限包括仅采用 CEFR 作为复杂度度量、依赖自动 CEFR 分类器可能带来的误差、语言覆盖范围有限、研究仅以定量分析为主，缺乏定性评估。

---

## 148. StableRCA: Robust Graph-Agnostic Mechanism-Level Root Cause Analysis

**arXiv ID:** 2606.05636 | [PDF](https://arxiv.org/pdf/2606.05636v1)

**作者:** Xiaoyu Lin `[一作]` (Tsinghua University), Juergen Luettin `[通讯]` (Bosch Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种不依赖全局因果图的机制级根因分析框架StableRCA，能够从观测与异常数据中识别导致系统异常的根因变量。

**💡 创新点**

核心创新在于通过局部马尔可夫边界估计与条件分布偏移检测实现根因定位，避免了全图学习的高成本与误差，同时利用独立因果机制原理证明在一定条件下可指数收敛识别正确目标。

**🔧 技术方法**

主要技术包括：1）Kolmogorov–Smirnov / χ²检验筛选边缘分布变化；2）Stable Learning（SRDO）实现局部马尔可夫边界的稳健估计；3）基于预测性能衰减的条件分布偏移检测与相对风险变异（RRV）评分。

**📊 数据集**

实验数据涵盖：1）基于Erdős–Rényi图的合成数据（含线性/非线性结构、不同噪声分布、单/多根因、不同图规模）；2）五个真实世界数据集（ProRCA、Sock-Shop、RCAEval、CausalMan、CausalChambers）。

**📈 对比分析**

与多种基线（图依赖如Traversal、CIRCA、Smooth Traversal；图无关如RCD、Score Ordering、Cholesky；非因果如ϵ‑Diagnosis、BARO）进行比较。StableRCA在合成与真实数据上在Top‑1/Top‑k准确率、精度/召回率和运行时间上均优于大多数基线，尤其在图误差、多个根因、和大规模图（至800节点）场景中表现出更高鲁棒性与更快速度。

**⚠️ 局限性**

局限性包括：1）对马尔可夫边界估计的依赖，估计误差会导致性能下降；2）假设因果充分性、无隐藏混杂、可检测的机制变动；3）不适用于存在强反馈、循环或弱干预的系统；4）需要一定量的异常样本以保证边缘与条件偏移检测的统计功效。

---

## 149. Individual Gain, Collective Loss: Metacognitive Adaptation in AI-Assisted Creativity

**arXiv ID:** 2606.05532 | [PDF](https://arxiv.org/pdf/2606.05532v1)

**作者:** Anna Mikeda `[一作]` `[通讯]` (Glass Umbrella), Anna Mikeda (Glass Umbrella)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出“选择性元认知适应”框架，解释AI创作中个体创意提升与集体多样性下降的悖论，并设计实验验证；

**💡 创新点**

将元认知容量按时间阶段分类，阐释常规AI使用导致的元认知努力再分配机制，揭示个体与群体效应的矛盾；

**🔧 技术方法**

结合元认知理论、AI交互日志分析、语义相似度测量与行为轨迹收集技术；

**📊 数据集**

使用自定义写作与创意生成实验数据，参考Moon等2024、Doshi & Hauser 2024、Anderson等2024等公开实验数据；

**📈 对比分析**

对比AI辅助组与非辅助组在个体创意评分和集体语义多样性指标上的差异，结果显示AI组在表面控制和伙伴建模上显著提升，但原创性评估与反思整合下降，集体多样性显著下降；

**⚠️ 局限性**

仅为理论框架，未进行大规模验证；依赖文本创作领域，跨领域推广需进一步研究；元认知容量划分为二分法，缺乏连续度量。

---

## 150. The Coverage Gap: Chile's Cyber Disclosure Framework versus the USA, EU and UK

**arXiv ID:** 2606.05594 | [PDF](https://arxiv.org/pdf/2606.05594v1)

**作者:** David Mellafe Z `[一作]` `[通讯]` (Reizan), David Mellafe Z (Reizan)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过被动OSINT对Chile 915个OIV（关键基础设施运营商）进行全宇宙审计，测量其披露能力、邮件认证配置及软件堆栈更新情况；

**💡 创新点**

提出“Coverage Gap”三层度量框架，将披露能力量化为可外部验证的指标，并将其与美国、欧盟、英国、荷兰和丹麦等已实施监管的司法辖区进行对标；

**🔧 技术方法**

采用RFC 9116 security.txt检测、SPF/DKIM/DMARC记录抓取、证书透明度日志、Shodan扫描等被动网络安全测量技术；

**📊 数据集**

使用ANCI发布的OIV法律名单（915实体）及其域名映射（通过公开的aci-oiv-resolver工具），结合公开DNS、证书和Shodan数据；

**📈 对比分析**

对比法：对不同司法辖区的Layer 1披露渠道覆盖率与邮件认证强制实施的时间差进行横向对标；结果显示Chile仅1.7%覆盖率，落后美国/英国/荷兰约8年；

**⚠️ 局限性**

局限包括：仅覆盖约98.7%实体，未获得ANCI真实基线数据，Layer 3仅估计，可能存在0.8%误报，且研究为时间点快照，未捕捉后续整改情况；

---

## 151. Unlocking Exponential and Unbounded Robust Gains in Shannon Capacity of Classical Multiple Access Channels with Causal CSIT via Quantum Entanglement Assistance

**arXiv ID:** 2606.05412 | [PDF](https://arxiv.org/pdf/2606.05412v1)

**作者:** Yuhang Yao `[一作]` (University of California Irvine), Syed A. Jafar `[通讯]` (University of California Irvine)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文研究了在多用户多路访问通道上利用传输端量子纠缠与因果信道状态信息（CSIT）提升香农容量的可能性。

**💡 创新点**

创新点在于发现只需传输端纠缠，且在存在CSIT的情况下可实现指数级、无界甚至对大状态字母集无限增益的容量提升，首次揭示量子纠缠在多路访问网络中的巨大潜能。

**🔧 技术方法**

主要技术包括量子测量与非本地协同编码、利用Mermin‑GHZ/CHSH等量子游戏实现干扰消除的信道转换方案，以及对加性干扰模型的概率与信息论分析。

**📊 数据集**

论文使用理论构造的各种K‑user加性干扰通道（例如B1、B2、Class A、B、C等），仅通过定义随机状态分布与干扰函数，而不依赖任何真实数据集。

**📈 对比分析**

与传统无纠缠编码方案相比，所示容量提升可达21倍、88倍甚至指数级，且在约30%去极化噪声下仍保持优势，显示出高度鲁棒性。

**⚠️ 局限性**

局限性包括仅考虑传输端纠缠，未探讨接收端或全网络共享纠缠、严格因果/非因果CSIT的更广泛适用性，以及在更一般网络拓扑下的可扩展性。

---

## 152. Anomaly Detection for Electro-Hydrostatic Actuators using LSTM Autoencoder

**arXiv ID:** 2606.05274 | [PDF](https://arxiv.org/pdf/2606.05274v1)

**作者:** Nehal Afifi `[一作]` (Karlsruhe Institute of Technology), Sven Matthiesen `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 1617 | [OpenAlex ID](https://openalex.org/A5045237635)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套基于LSTM自编码器的离线异常检测流程，用于监测电液静压执行器（EHA）传感器的温度和压力时序数据。

**💡 创新点**

首次对高采样频率的EHA数据采用重构式LSTM AE进行系统评估，证明其能在低误报率下实现高精度异常识别。

**🔧 技术方法**

采用滑窗分段、归一化、LSTM编码-解码网络、MAE重构误差以及阈值校准技术进行异常判定。

**📊 数据集**

使用受控实验台产生的EHA单变量传感器数据集，涵盖惯性负载与弹簧负载两种工况，并注入多种故障场景。

**📈 对比分析**

与统计方法（Z-score、IQR、MAD）、经典机器学习（Isolation Forest、Gaussian Mixture、k‑means）以及其他深度自编码器（CNN‑AE）对比，LSTM‑AE在准确率≈99%、精确率0.96–1.00、召回率0.90–0.99、F1≈0.93–0.99、ROC‑AUC 0.95–0.99等指标上显著优于基线。

**⚠️ 局限性**

局限性包括仅处理单变量信号、阈值需针对不同工况重新调校，以及基于σ阈值的标签定义可能无法捕捉慢性降解或关键故障。

---

## 153. Data-efficient flood depth prediction through domain-aware coreset selection and tabular foundation models

**arXiv ID:** 2606.05265 | [PDF](https://arxiv.org/pdf/2606.05265v1)

**作者:** Lipai Huang `[一作]` (Texas A&M University), Ali Mostafavi `[通讯]` (Texas A&M University)

**通讯引用:** 7477 | [OpenAlex ID](https://openalex.org/A5023165780)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于域感知的核心样本（coreset）构建管道，用于在无需每个流域单独训练的情况下，通过tabular foundation model 实现近实时的洪水深度预测。

**💡 创新点**

创新点包括：① 双阶段核心样本构造——先按降水事件的返回期和受灾最深流域双重分层，后通过目标感知的空间设施定位（FL-Depth）选择 hexagon；② 直接在推理时用核心样本对TabPFN-v2.6 进行条件化，无需梯度更新；③ 通过在邻近流域采集核心样本实现无泄漏的跨流域迁移。

**🔧 技术方法**

使用技术包括：Tabular foundation models（TabPFN‑v2.5、v2.6、TabICL），设施定位与层级采样（FL‑Depth），H3 级别10六边形网格，降水事件的返回期分层，in‑context learning，留一交叉流域评估，细粒度梯度微调实验，以及 R²、RMSE、MAE 等性能评估指标。

**📊 数据集**

数据集：592 个合成降水事件（HEC‑RAS 2D 模拟）覆盖 9 个休斯顿地区流域，约 105 M 行 hexagon‑depth 数据；静态地理特征来自 NED、NLCD、NOAA Atlas 14、TxDOT、NHDPlusHR 等；外部真实事件为 2017 年 Harvey 飓风和 2019 年 Imelda 热带风暴的观测雨量与 HEC‑RAS 模拟深度。

**📈 对比分析**

与方法比较：在同一 50k 核心样本下，Vanilla TabPFN‑v2.6 在九个流域的平均 R² 为 0.663，约为全数据 XGBoost 基线 0.673 的 98.5%；跨流域留一评估中，TabPFN‑v2.6 在 0.50–0.52 范围内稳定领先，且不需要微调；在真实事件上，TabPFN‑v2.6 与微调版在极端 OOD（Harvey）上优于基线，在主要分布内（Imelda）基线仍略胜。整体显示：核心样本 + TabPFN 在 OOD 下更稳健，基线在分布内更精准。

**⚠️ 局限性**

局限性：① 跨流域方差较大（std≈0.059）；② 仅使用 50k 行核心样本，未充分利用 TabPFN‑v2.6 的更大上下文窗口；③ 评估仅覆盖九个休斯顿流域，未测试不同气候区域；④ 仅报告 R²，未细化 RMSE/MAE 与高深尾部准确度；⑤ hexagon 级别限制，需在原始网格层面进一步验证。

---

## 154. Sharp First-Order Lower Bounds for Higher-Order Smooth Nonconvex Optimization

**arXiv ID:** 2606.05438 | [PDF](https://arxiv.org/pdf/2606.05438v1)

**作者:** Dongruo Zhou `[一作]` `[通讯]` (Indiana University Bloomington), Dongruo Zhou (Indiana University Bloomington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并证明了在高阶光滑非凸优化中，确定性第一阶方法寻找ϵ-驻点的最优复杂度；给出了匹配的Ω(ϵ⁻⁷⁄⁴)（Hessian-Lipschitz）和Ω(ϵ⁻⁵⁄³)（第三阶光滑）下界。

**💡 创新点**

创新点在于设计了块链（block‑chain）硬实例，将标量链与线性拉回相结合，巧妙分离梯度证书与信息揭示过程，从而实现了对任意有限阶光滑的最优下界。

**🔧 技术方法**

主要技术包括：零-尊重方法（zero‑respecting）、块链构造与层级隐藏、线性拉回与光滑性缩放、梯度证书分析、正交变换不变性以及对高阶导数 Lipschitz 常数的细致量化。

**📊 数据集**

无；本文为理论分析，没有使用实验数据集。

**📈 对比分析**

与已知的最优上界（O(ϵ⁻⁷⁄⁴)、O(ϵ⁻⁵⁄³)）相匹配，证明这些上界在确定性第一阶方法下已达到极限；未做实验比较。

**⚠️ 局限性**

局限性：仅适用于确定性第一阶方法，未涵盖随机或批量梯度方法；对数因子在某些上界中仍存在；对更高阶光滑（p≥4）未进一步改进复杂度，仅保持与第三阶相同的指数。

---

## 155. Auditing Demonstration Curation Metrics: Action-Only Scorers Fail on the Structural Defects That Degrade Imitation Policies

**arXiv ID:** 2606.05588 | [PDF](https://arxiv.org/pdf/2606.05588v1)

**作者:** Aarav Bedi `[一作]` `[通讯]` (University of California, Berkeley), Aarav Bedi (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一个受控的仿真 pick‑and‑place 测试平台上，对七种示例清理（curation）指标进行了系统审计，评估它们在检测演示缺陷与提升行为克隆策略性能上的效果。

**💡 创新点**

发现仅基于动作的指标不仅可能无法识别结构性错误，甚至可能出现评分反转（误把错误示例评为高质量），并证明检测准确性并不必然导致策略性能提升。

**🔧 技术方法**

采用了 NumPy 轻量级仿真器、演示缺陷注入器、行为克隆（behavior‑cloning）策略、离群检测（Isolation Forest、k‑NN、轨迹对齐等）以及自定义评估管道等技术。

**📊 数据集**

使用的是仿真生成的合成数据集，包括完全干净的脚本演示和通过已知缺陷注入器产生的受污染演示；未使用真实操作员收集的数据。

**📈 对比分析**

通过 AUROC 评估检测能力，并在每个指标清理子集上训练 150 条演示的三层 MLP，测量 50 次新环境回放的任务成功率；动作相关指标在细微扰动下表现良好，但在结构错误下表现不佳；仅状态轨迹感知指标能够部分弥补性能缺口，整体恢复率约为 30%。

**⚠️ 局限性**

局限性包括：仅在轻量级仿真器上实验，单臂单任务，合成缺陷与真实缺陷可能不同；种子数少且后向传播方差大；仅测试行为克隆策略；未涵盖接触丰富、视觉感知或多操作员真实演示的场景。

---

## 156. TensorBench: Benchmarking Coding Agents on a Compiler-Based Tensor Framework

**arXiv ID:** 2606.05570 | [PDF](https://arxiv.org/pdf/2606.05570v1)

**作者:** Bobby Yan `[一作]` (Stanford University), Fredrik Kjolstad `[通讯]` (Stanford University)

**通讯引用:** 1318 | [OpenAlex ID](https://openalex.org/A5041886781)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TensorBench，一个面向编译器式张量框架的199个特性添加与重构任务的仓库级评测基准。

**💡 创新点**

创新点在于将任务与可执行的回归测试相结合，通过后置测试套件判定补丁正确性，减少单例函数级评测饱和问题，并提供多任务多类别（API、格式、IR、调度、代码生成、运行时）覆盖。

**🔧 技术方法**

使用了基于Scorch的张量编译器、Docker化的评测管道、LLM代理与CLI scaffold（Claude Code、Codex CLI、Gemini CLI、OpenHands）以及后置测试自动化。

**📊 数据集**

数据集为199个任务（194特性添加、5重构），按六类划分，任务来自自然语言描述，覆盖API、调度、运行时、格式、IR、代码生成。

**📈 对比分析**

比较方式为统计各代理在post‑patch test‑suite的通过率，Claude 4.7最高64.8%，Codex 5.5 58.8%，Qwen3 22.1%；并计算两两Cohen's κ、联合通过率等。性能表明强模型对本地扩展有优势，但对全局重构仍不足。

**⚠️ 局限性**

局限包括缺乏隐藏验证测试、仅在单一代码库上评测、通过率可能被自写测试上浮、对任务描述质量敏感、易被恶意代理利用。

---

## 157. Deep Learning-assisted AMD Staging based on OCT and OCT Angiography

**arXiv ID:** 2606.05379 | [PDF](https://arxiv.org/pdf/2606.05379v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 158. SWE-InfraBench: Evaluating Language Models on Cloud Infrastructure Code

**arXiv ID:** 2606.05249 | [PDF](https://arxiv.org/pdf/2606.05249v1)

**作者:** Natalia Tarasova `[一作]` (Amazon Web Services), Sergei Ivanov `[通讯]` (Amazon Web Services)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个针对AWS CDK增量修改任务的IaC基准数据集，并对20种LLM在此任务上的性能进行了系统评估。

**💡 创新点**

创新点在于提出真实企业场景的增量IaC编辑任务、基于单元测试的自动评估流程以及多轮反馈与检索增强生成的实验框架。

**🔧 技术方法**

使用的大技术包括大语言模型（Claude Sonnet、GPT‑4、DeepSeek R1等）、CDK代码合成、Python单元测试、两轮交互式代理以及检索增强生成（RAG）。

**📊 数据集**

数据集来源于34个真实AWS CDK仓库，包含100个任务，已公开发布在Kaggle。

**📈 对比分析**

通过pass@1、pass@5、正确率和测试通过率等指标比较，最高单轮模型Claude 3.7仅取得34%成功率，双轮代理可提升至65%。

**⚠️ 局限性**

局限性包括仅覆盖AWS CDK（Python）场景、对测试用例覆盖度和安全/性能评估不足，以及多轮交互所带来的高算力与时间成本。

---

## 159. Step-by-Step Optimization-like Reasoning in LLMs over Expanding Search Spaces

**arXiv ID:** 2606.05464 | [PDF](https://arxiv.org/pdf/2606.05464v1)

**作者:** Nicolás Astorga `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 23070 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一类可自动验证、可按复杂度扩展的优化式任务，用于训练大型语言模型（LLM）进行逐步决策推理，并提出离线强化学习结合搜索与在线强化学习结合求解器的两种训练方案；

**💡 创新点**

创新点在于：①将传统优化问题转化为可按复杂度调节的可验证任务族；②设计可插拔的搜索组件（可剪枝、去重）提升搜索效率；③引入求解器导向的在线强化学习以获取部分状态的价值信号；④给出理论分析阐释搜索空间扩大对信息瓶颈的影响；

**🔧 技术方法**

技术手段包括：LLM提示化与解析接口、MCTS/Beam搜索改造、离线强化学习（SFT/ DPO）以及基于求解器的在线强化学习（GRPO/GSPO/PPO）并采用基于排名的奖励塑造；

**📊 数据集**

数据集为自动生成的经典优化实例（角色分配、最大满足、背包、QAP、TSP、调度等），以及MATH‑500和MBPP等公开基准，用不同复杂度参数α产生多级训练样本；

**📈 对比分析**

与基准模型、求解器参考、传统搜索等对比；评估指标包括pass@k、有效分支因子b_eff、成功率、终端可行性与最优性；实验显示求解器导向的在线RL在大多数任务上优于离线RL，离线RL相较于无训练和纯搜索提升约2–3倍，且两种方法结合能显著降低搜索开销；

**⚠️ 局限性**

局限性包括：对求解器的依赖导致在无求解器或求解成本高的场景下效果有限；LLM容量限制导致在极大搜索空间中仍难以充分探索；实验主要基于合成任务，实际工程应用中需要进一步验证泛化能力。

---

## 160. Geographic Bias and Diversity in AI Evaluation

**arXiv ID:** 2606.05187 | [PDF](https://arxiv.org/pdf/2606.05187v1)

**作者:** Zilong Liu `[一作]` (University of Vienna), Rui Zhu `[通讯]` (University of Bristol)

**通讯引用:** 27101 | [OpenAlex ID](https://openalex.org/A5100617068)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过综述前后生成式 AI 时代的研究，系统地识别并定义了地理偏差（包括训练数据代表性偏差、聚合偏差和对受保护属性的歧视），并提出了将“地理多样性”作为衡量生成式 AI 输出公平性的新指标，随后对多种 LLM 和图像生成模型在不同提示、温度设置下的输出进行实验评估。

**💡 创新点**

创新点在于：①首次将生态学中的 Hill 数、Leinster‑Cobbold 归一化多样性度量引入地理多样性评估；②将地理多样性与传统公平性指标结合，形成“偏差-多样性”二元评价框架；③通过多模型、多温度实验验证地理多样性与生成随机性、模型版本的关系，揭示新模型不一定更具多样性。

**🔧 技术方法**

使用的技术包括：大规模语言模型（BERT、GPT‑3.5、ChatGPT‑4o 等）与生成式图像模型（GPT‑Image‑1 等）；蒙版语言建模和自回归生成的评测管道；基于统计的多样性度量（Hill 数、Shannon 熵、Inverse Simpson 指数以及考虑相似度的 Leinster‑Cobbold 数）来量化输出分布；以及温度采样控制实验。

**📊 数据集**

使用的数据集主要有：Open Images、ImageNet（用于训练数据代表性评估）；World Bank 发展指标（用于事实回忆误差评估）；对 LLM 进行 geoparsing、事实回忆任务的标准数据；以及自构造的提示集合用于生成式实验。

**📈 对比分析**

比较方法：将多样性度量与温度、模型版本、提示类型（事实、描述、图像）相结合；使用图表展示不同 q 阈值下的多样性曲线；将模型输出与真实世界的地理多样性基线（如世界人口分布）做偏差对比。性能结果显示：①温度升高可显著提升 1 阶多样性；②不同模型之间差异不大，部分新模型反而多样性下降；③考虑相似度后多样性显著降低，表明模型倾向于输出相似的“典型”地点。

**⚠️ 局限性**

局限性：①评估主要基于少数公开数据集，未覆盖所有语言或文化语境；②多样性度量假设输出分布独立于上下文，忽略了语义约束；③温度调节可能导致准确性下降，尚未给出最佳权衡；④缺乏统一的地理多样性基准，导致“公平”标准相对主观；⑤实验多聚焦于英语文本，对多语种生成式 AI 的适用性仍需验证。

---

## 161. The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show

**arXiv ID:** 2606.05328 | [PDF](https://arxiv.org/pdf/2606.05328v1)

**作者:** Parsa Esmati `[一作]` (University of Bristol), Majid Mirmehdi `[通讯]` (University of Bristol)

**通讯引用:** 7113 | [OpenAlex ID](https://openalex.org/A5007937949)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过近似逆向采样恢复真实视频在视频扩散模型中的潜在轨迹，并在此轨迹上对模型内部 transformer 进行线性探测与因果干预，以验证其内部是否编码了物理结构。

**💡 创新点**

创新点在于提出逆向采样技术以获取真实视频的潜在路径，并发现即使无显式物理监督，扩散模型的 denoising 计算本身会产生可线性解码的物理信息；此外，还通过层级干预定位了物理信息的流向。

**🔧 技术方法**

使用的技术包括流匹配 ODE 逆向积分（Euler/Heun 方案）、VAE 潜在编码、Transformer 线性探针、噪声干预、probe‑surprise 评估以及对比实验框架。

**📊 数据集**

数据集涵盖 IntPhys、InfLevel 两个物理可行性判定基准，以及一个 2D 物理仿真数据集（可获取初始位置、速度等量化参数）。

**📈 对比分析**

与 V‑JEPA、VideoMAE 等表征学习基线在 IntPhys/InfLevel 上进行对比，线性探针平均准确率可达 81.3%（超过 V‑JEPA 71.4%），在量化回归任务中 R² 接近 1，显示物理信号在模型内部高度可解码。

**⚠️ 局限性**

局限性包括逆向采样的近似可能引入轨迹误差；线性探针只能证明可解码性而非模型显式表达物理定律；因果干预仅提供部分因果证据；离散化步数对物理信号解码敏感，需更精细的数值积分。

---

## 162. Uncertainty Aware Functional Behavior Prediction and Material Fatigue Assessment for Circular Factory

**arXiv ID:** 2606.05334 | [PDF](https://arxiv.org/pdf/2606.05334v1)

**作者:** Nehal Afifi `[一作]` (Karlsruhe Institute of Technology), Sven Matthiesen `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 1617 | [OpenAlex ID](https://openalex.org/A5045237635)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究返回的角磨机在循环工厂中的功能行为预测与材料疲劳评估，提供实例特定的再部署决策支持。

**💡 创新点**

创新点在于将不确定性感知的条件序列预测与组件级疲劳分析通过流式可靠性框架联合，实现一次性从同一使用历史同时预测功能演化与结构寿命。

**🔧 技术方法**

使用了卷积编码器提取力‑扭矩窗口、LSTM序列模型与加权高斯负对数似然训练实现不确定性估计；材料端采用有限元应力重构、S–N/Miner累计、Paris‑裂纹扩展等传统疲劳分析方法。

**📊 数据集**

数据来源包括在KIT测试台上采集的力、扭矩、温度、电流、速度等时序数据，以及通过旋转弯曲实验获得的S–N曲线、材料成分与硬度分布。

**📈 对比分析**

与GRU、xLSTM比较，LSTM在2%容差精度0.965、NRMSE0.0297、R^2 0.836 上优于其他模型；功能预测的可靠性校准误差低，材料重用周期从31降至3（高负荷放大）。

**⚠️ 局限性**

局限性包括仅在实验台固定循环条件下验证，缺乏真实工厂多样化使用历史；功能指标事件稀缺导致可靠性评估受限；材料模型未考虑裂纹长短、负荷序列效应与个体化参数。

---

## 163. EpiEvolve: Self-Evolving Agents for Streaming Pandemic Forecasting under Regime Shifts

**arXiv ID:** 2606.05513 | [PDF](https://arxiv.org/pdf/2606.05513v1)

**作者:** Yiming Lu `[一作]` (Emory University), Wei Jin `[通讯]` (Emory University)

**通讯引用:** 3428 | [OpenAlex ID](https://openalex.org/A5100758371)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为EpiEvolve的自进化代理，利用冻结的LLM前端在疫情预测流式部署中通过层次化经验回忆、策略规则提炼和漂移检测实现自适应。

**💡 创新点**

创新点在于把模型权重固定，仅通过可解释的经验存储和策略化的记忆更新来快速响应变异期和概念漂移，而不需梯度微调；同时结合多级回忆与规则蒸馏形成多层次自适应机制。

**🔧 技术方法**

技术包括LLM预训练/微调、层次化经验回忆（state/region/national）、反思式记忆写入、策略化规则提炼、漂移检测与基于变异文本的检索权重。

**📊 数据集**

使用的是美国50州2021‑2022年间的COVID‑19每周住院趋势数据，包含病例、政策、疫苗接种、基因测序等多源信息。

**📈 对比分析**

与静态LLM、检索仅、反思仅、流式微调、外部CDC集成等基线比较，EpiEvolve平均准确率提升至0.629，恢复延迟从5周降至2周，整体表现优于所有对照。

**⚠️ 局限性**

局限包括仅针对住院趋势分类，未覆盖完整的流行病学预测；数据集规模和变异边界不够细粒；对记忆与反思提示设计及规则质量较为敏感。

---

## 164. Biomazon: A Multimodal Dataset for 3D Forest Structure and Biomass Modeling in the Amazon Basin

**arXiv ID:** 2606.05368 | [PDF](https://arxiv.org/pdf/2606.05368v1)

**作者:** Sayan Mandal `[一作]` (Jülich Supercomputing Centre), Gabriele Cavallaro `[通讯]` (Jülich Supercomputing Centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了 Biomazon——一个覆盖亚马逊流域的 20 m 多模态遥感基准数据集，旨在同时预测完整的 GEDI 相对高度 (RH) 轮廓和上层地上生物量密度 (AGBD)，并提供统一的分割与评估协议。

**💡 创新点**

创新点在于：① 将 RH 轮廓视为有序结构化输出并引入锚点‑累积增量的单调参数化；② 结合多源 Sentinel‑1/2、ALOS‑2、DEM、LULC 与 AlphaEarth 嵌入，形成首个针对完整 RH 轮廓 + AGBD 的 ML 训练基准；③ 对基线模型在尺度、模态、嵌入融合和联合训练等维度展开系统消融。

**🔧 技术方法**

技术上采用共享 ViT‑Prithvi 编码器 + DPT 解码器，单独的 RH 与 AGBD 头，锚点‑单调化 RH 预测；使用 Huber 损失配合标签分布平滑（LDS）重加权；在 late‑fusion 模式下通过 FiLM 与门控残差融合 AlphaEarth 嵌入；实验使用 AdamW、余弦学习率退火、5 轮随机种子。

**📊 数据集**

使用的主要数据集为：亚马逊流域 2019‑2023 年期的 GEDI L2A（RH 101 维度）和 L4A（AGBD），以及与其配套的 Sentinel‑1/2、ALOS‑2 PALSAR‑2、Copernicus GLO‑30 DEM、Dynamic World V1 LULC、AlphaEarth Foundations 嵌入；所有特征统一投影到 20 m HLS‑MGRS 网格。

**📈 对比分析**

在统一验证集上与 GEDI L4D、全球高程与生物量产品（如 ESA Biomass CCI、Potapov 等）对比，基线模型在 RH 10‑98 及 AGBD 上均显著降低 RMSE（例如 RH95 RMSE 约 5.6 m，AGBD RMSE 约 73 kg ha⁻¹），并避免了 L4D 极端异常值；AlphaEarth 嵌入在单一 CNN 架构下即可匹敌或超越完整多模态 ViT‑DPT 组合。

**⚠️ 局限性**

局限性包括：① 受 GEDI 采样稀疏与目标误差限制，模型容量提升效果有限；② AGBD 仅为 GEDI L4A 的全局经验模型，缺乏独立地面验证；③ 评估对比受产品分辨率、时间与目标定义差异影响，结果更多为参考性比较；④ AlphaEarth 嵌入与基线模型在训练时共享 GEDI 监督，可能导致“嵌入优势”与特定任务相耦合。

---

## 165. Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models

**arXiv ID:** 2606.05481 | [PDF](https://arxiv.org/pdf/2606.05481v1)

**作者:** Raffael Theiler `[一作]` (EPFL), Olga Fink `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对工业PHM任务提出统一的表格化表示流程，并利用预训练的表格基础模型（TabPFN、TabDPT）在诊断和剩余寿命预测上进行评估。

**💡 创新点**

创新点在于：①将时序数据转换为可直接用于表格学习的行向量，②通过in‑context学习实现无需任务专门训练的迁移，③展示表格基础模型在低样本、缺失值以及多任务场景下的鲁棒性与高效性。

**🔧 技术方法**

采用的技术包括：表格化预处理流水线（特征抽取、时间对齐、窗口切片、平面化），in‑context学习框架，TabPFN 与 TabDPT 两类表格基础模型，Transformer系列（PatchTST、Crossformer、Spacetimeformer）、CNN、LSTM、TiDE、XGBoost 以及统一的 PICID 评估平台。

**📊 数据集**

实验使用 12 个工业 PHM 基准数据集：PHME20、Unibo、XJTU‑SY、N‑CMAPSS DS02、N‑CMAPSS Prognostics、NB14、HSF15 组件级任务、MZVAV 等。

**📈 对比分析**

在统一评估协议下，对所有模型计算 MAE/ RMSE（预测任务）与 Macro‑F1（诊断任务）。结果显示表格基础模型在所有任务中平均排名最高，TabDPT 平均第 2.67 位，TabPFN 第 3.33 位；在极少样本（仅 1%‑10% 训练数据）下，表格模型已能与传统序列模型相媲美或更优，且在缺失值处理时表现出色。

**⚠️ 局限性**

主要限制：①性能高度依赖于提供的上下文分布，若测试数据与训练上下文不匹配会显著退化；②推理时受限于上下文样本数和行维度，计算成本较高；③当前表格化不支持对单个传感器或时间步进行子采样，限制了维度压缩；④实验采用的预处理参数相对保守，未来需探索更通用的转换方案。

---

## 166. NIV: Neural Axis Variations for Variable Font Generation

**arXiv ID:** 2606.05261 | [PDF](https://arxiv.org/pdf/2606.05261v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 167. SciVisAgentSkills: Design and Evaluation of Agent Skills for Scientific Data Analysis and Visualization

**arXiv ID:** 2606.05525 | [PDF](https://arxiv.org/pdf/2606.05525v1)

**作者:** Kuangshi Ai `[一作]` (University of Notre Dame), Chaoli Wang `[通讯]` (University of Notre Dame)

**通讯引用:** 3124 | [OpenAlex ID](https://openalex.org/A5101913449)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SciVisAgentSkills，一套针对 ParaView、napari、VMD、TTK 等科学可视化工具的可重用 agent 技能，并在 SciVisAgentBench 上进行评估。

**💡 创新点**

通过将工具使用流程、API 约定、示例代码等结构化为技能，低成本提升通用编码代理在多步骤 SciVis 工作流中的成功率。

**🔧 技术方法**

采用 agent skill（YAML+Markdown 指南）与 MCP/CLI 接口，结合 Claude Code、Codex 等通用编码代理和多模态 LLM 判别器技术。

**📊 数据集**

使用了 108 个由专家编写的多步骤 SciVis 场景组成的 SciVisAgentBench 数据集，覆盖体渲染、等值面、流场、分子、图像、生物图形、拓扑可视化等任务。

**📈 对比分析**

通过整体得分、完成率、图像质量指标（PSNR/SSIM/LPIPS）和 token 消耗进行比较，结果显示加上技能后总体得分显著提升，尤其在拓扑可视化和对象识别任务中，但 token 变化与模型和 harness 相关。

**⚠️ 局限性**

技能收益受模型先验知识、工具版本等限制，部分任务收益有限；Token 使用随模型不同而变化，且技能与 harness 的交互机制仍需进一步研究。

---

## 168. LightVesselNet: An Ultra-Lightweight Sub-100K Parameter Network for Retinal Blood Vessel Segmentation

**arXiv ID:** 2606.05354 | [PDF](https://arxiv.org/pdf/2606.05354v1)

**作者:** Shadman Sobhan `[一作]` (Bangladesh University of Engineering and Technology), Farhana Jalil `[通讯]` (Bangladesh University of Engineering and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了一种超轻量化的视网膜血管分割网络 LightVesselNet，专门针对资源受限的边缘设备进行部署。

**💡 创新点**

创新点包括：
1) MicroBlockSE 轻量化特征提取块，融合深度可分离卷积、SE 通道注意力、残差和 DropBlock 正则化；
2) MSFA（多尺度特征聚合）瓶颈模块，使用四个并行膨胀深度卷积 + 空间注意力，能够捕捉不同直径的血管；
3) 子像素上采样（PixelShuffle）与边缘残差路径，避免插值失真并保留细节；
4) 采用 Tversky‑Focal 联合损失和多尺度深度监督，提升薄血管的召回率。

**🔧 技术方法**

使用的技术包括：
- encoder‑decoder 结构
- depthwise‑separable 卷积、SE 通道注意力、DropBlock、GroupNorm
- 空间注意力门、PixelShuffle 上采样、边缘残差连接
- 数据增强（Albumentations）、CLAHE、绿色通道预处理
- Tversky‑Focal 损失、深度监督
- 轻量化的 75K 参数设计

**📊 数据集**

实验使用的公开数据集：DRIVE、STARE、CHASE_DB1、FIVES、HRF。

**📈 对比分析**

通过与 UNet、RetinaLiteNet、LVS‑Net、LFA‑Net、LFRA‑Net、LW‑UNet+RA 等轻量级模型在同一数据集上对比（使用 sensitivity、specificity、accuracy、Dice、IoU 等指标），LightVesselNet 在五个数据集上实现与参数规模远大模型相当的性能，尤其在 DRIVE、CHASE_DB1 的 sensitivity 与 accuracy 处于前沿；在跨数据集评估中也表现出良好的泛化能力，并在 Pareto 前沿上占据有利位置。

**⚠️ 局限性**

局限性：
- 由于需要统一输入尺寸，部分高分辨率数据被裁剪/缩放，细节保持有待进一步验证；
- 在极薄血管检测上仍略逊于部分大型模型；
- 仅在公开数据集上评估，未公开权重/代码便于复现；
- 未对模型进行剪枝、量化等进一步压缩或在真实移动端设备上的推理延迟做详细测评。

---

## 169. Synthetic Contrastive Reasoning for Multi-Table Q&A

**arXiv ID:** 2606.05382 | [PDF](https://arxiv.org/pdf/2606.05382v1)

**作者:** Ankit Pratap Singh `[一作]` (Iowa State University), Phillip Howard `[通讯]` (Thoughtworks)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

为多表问答任务构建了由正向推理轨迹和负向对比轨迹组成的合成数据集，并使用对比偏好优化（CPO）对开源大语言模型进行微调。

**💡 创新点**

创新点在于：①首次提供多表问答的对比推理轨迹数据；②采用异构模型（GPT‑4o生成正轨迹、Gemini 2.0 Flash生成负轨迹）增强对比信号；③将对比偏好优化与行为克隆正则化相结合，提升训练稳定性和效果。

**🔧 技术方法**

主要技术包括：链式推理（Chain‑of‑Thought）轨迹生成、LLM 作为审计者（答案验证、语义一致性检查、轨迹质量评估）、对比偏好优化（CPO）与直接偏好优化（DPO）对比实验。

**📊 数据集**

使用了 MMQA（两表数据集）构建训练与测试集，并在 BIRD 基准上构建跨域评估集（三表查询），另外在 MMTU、TableBench 等公开评测集上进行泛化验证。

**📈 对比分析**

与仅使用问答对训练（Q&A SFT）和仅使用正向轨迹训练（Trace SFT）相比，CPO 微调在四个评测集上平均提升 9.7%–16.3%（对 MMQA 最大提升 21%），显著优于 DPO 并在跨域和三表情境下保持优势。

**⚠️ 局限性**

局限性包括：正负轨迹生成高度依赖商业 API（GPT‑4o、Gemini），影响可复现性与成本；LLM 审计者存在偏差且验证覆盖有限；正向轨迹筛选仅检查最终答案，可能遗漏中间推理错误；训练数据规模相对较小，虽有效但在更大规模上的表现尚待验证。

---

## 170. Multilingual Fine-Tuning via Localized Gradient Conflict Resolution

**arXiv ID:** 2606.05613 | [PDF](https://arxiv.org/pdf/2606.05613v1)

**作者:** Long P. Hoang `[一作]` (Singapore University of Technology and Design), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 5394 | [OpenAlex ID](https://openalex.org/A5042288832)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Bucket‑Level MOO，一种在分布式训练中按参数桶局部执行多目标优化的方法，用于解决大语言模型多语言微调中的负干扰。

**💡 创新点**

创新点在于：①将多目标优化从全模型层级转到桶级别，实现对梯度冲突的局部、结构化处理；②证明桶级优化自然而然满足 Refined Pareto Stationarity，严格比传统全局 Pareto Stationarity 更有利；③在保持显存和通信效率的同时，提升多语言性能。

**🔧 技术方法**

技术包括：多目标优化算法（MGDA、CAGrad、PCGrad）的桶级实现；分布式训练框架（DeepSpeed ZeRO、FSDP）的梯度桶拦截与局部梯度融合；理论证明与实验验证。

**📊 数据集**

使用 8 种语言（高低资源混合）的翻译语料（1,000 对话 + 630 推理样本），并在 BELE、ARC‑E、PolyMath、Global‑MMLU 这四个跨语言基准上进行评测；同时对 13 种语言（8 训练语言 + 5 盲测语言）进行测试。

**📈 对比分析**

与传统单目标微调（Vanilla SFT）以及全局 MOO（MGDA、CAGrad、PCGrad）对比；在所有四种基础 LLM 上，Bucket‑Level MOO 在已见语言平均提升 1.6–2.9 分，在未见语言平均提升 2.5–2.7 分；显存峰值保持在 72 GB，远低于全局 MOO 的 123 GB。

**⚠️ 局限性**

局限性包括：①桶级别的参数划分仍是经验性的，可能需要针对不同模型/任务手动调节；②对梯度冲突的检测仅基于局部内积，无法捕捉跨桶的全局协同效应；③在极低资源语言或极端多任务场景下，局部优化可能不足以消除所有干扰。

---

## 171. PyCC.id: A package for hypothesis-driven equation discovery with structural identifiability

**arXiv ID:** 2606.05191 | [PDF](https://arxiv.org/pdf/2606.05191v1)

**作者:** Federico J. Gonzalez `[一作]` (National University of Rosario), Federico J. Gonzalez `[通讯]` (National University of Rosario)

**通讯引用:** 26 | [OpenAlex ID](https://openalex.org/A5069381161)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个名为 pyCC.id 的 Python 库，用于在先验结构（骨架）驱动下进行可解释的 ODE 方程发现，支持神经网络、多项式和符号回归等多种后端。

**💡 创新点**

通过将结构骨架与特征曲线（CCs）相结合，实现了结构辨识性检查，显著减少了模型搜索空间，并提供了后处理工具将数值学习的 CC 转化为解析表达式。

**🔧 技术方法**

使用了 hypothesis‑driven 框架、结构可辨识骨架、PyTorch 神经网络、Poly 多项式、PySR 符号回归、物理约束损失函数，以及 GPU/CPU 并行加速和 ODE 求解器。

**📊 数据集**

主要使用仿真生成的非线性阻尼振子（含 Coulomb 摩擦）的时间序列数据；同样适用于任何可获得的时间序列测量。

**📈 对比分析**

通过示例演示 NN、SymbR、Poly、Interp 等方法的训练、后处理和前向仿真，并与理论模型进行对比，结果表明 NN 能准确逼近 f1、f2；后处理后获得的解析式对噪声更稳健；未给出与传统稀疏识别的定量基准，但指出方法在结构可辨识性和鲁棒性方面优于纯数据驱动方法。

**⚠️ 局限性**

限制包括需要先验骨架选择，仍需人工后验筛选模型；对高噪声数据的鲁棒性有限；实现依赖 PyTorch/NumPy 等现有库，缺乏大规模真实实验验证。

---

## 172. DP-MacAdam: Differentially Private Mechanism with Adaptive Clipping and Adaptive Momentum

**arXiv ID:** 2606.05435 | [PDF](https://arxiv.org/pdf/2606.05435v1)

**作者:** Naima Tasnim `[一作]` (Arizona State University), Oliver Kosut `[通讯]` (Arizona State University)

**通讯引用:** 3183 | [OpenAlex ID](https://openalex.org/A5055511546)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的不同ially private 优化器 DP‑MacAdam，融合 AdaClip 的自适应梯度裁剪与 Adam 的自适应动量，能够在训练过程中自动调节裁剪阈值并加速收敛。

**💡 创新点**

首次将 AdaClip 与 Adam 结合，并推导出新的偏差校正因子 κ_t，使得从带噪梯度的指数移动平均得到的梯度方差估计保持无偏；此外，该方法不需要手动调节裁剪阈值，且无额外隐私开销。

**🔧 技术方法**

采用差分隐私 SGD、AdaClip 的坐标级裁剪、Adam 的指数移动平均动量、噪声校正与偏差补偿技术，并使用 Connect‑the‑Dots 隐私核算器评估隐私预算。

**📊 数据集**

在 MNIST 和 CIFAR‑10 两个图像分类任务上进行实验，使用全连接网络和 5 层 CNN，分别包含约 8 万和 58 万可训练参数。

**📈 对比分析**

通过与 DP‑SGD、AdaClip、DP‑Adam、DP‑AdamBC 的对比，使用不同噪声倍数（σ）下的 ε 预算，测量测试准确率。DP‑MacAdam 在大多数隐私预算下的准确率均高于所有基线，尤其在 MNIST 上表现最为突出；在 CIFAR‑10 高噪声极限下略逊，但整体仍优于对手。

**⚠️ 局限性**

缺乏正式的收敛性分析；对更复杂任务（如 NLP）或更大规模数据集的泛化尚未验证；在极端高噪声或对 DP‑MacAdam‑BC 组合时性能下降；坐标级缩放向量 b_t 的最优选择仍待研究。

---

## 173. SHIELDS: Automating OS Hardening with Iterative Multi-Agent Remediation

**arXiv ID:** 2606.05476 | [PDF](https://arxiv.org/pdf/2606.05476v1)

**作者:** Andrew Hamara `[一作]` (L3Harris Technologies), Lawrence Wong `[通讯]` (Texas A&M University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个多代理AI系统SHIELDS，通过LLM在扫描-分析-修复-验证循环中自动化操作系统硬化。

**💡 创新点**

创新点在于将静态修复脚本替换为基于反馈的迭代修复流程，并通过四个专用代理实现自适应修复与安全审查。

**🔧 技术方法**

利用大语言模型（如GPT-OSS、Inception Mercury 2等）、工具调用、OpenSCAP扫描、Ansible剧本生成等技术。

**📊 数据集**

使用Rocky Linux虚拟机配合DISA STIG配置的OpenSCAP扫描结果作为评估数据集。

**📈 对比分析**

通过对6种LLM在3种VM规模下的修复率进行基准测试，发现最高达73%，且模型规模并非决定因素，工具使用能力更关键。

**⚠️ 局限性**

局限性包括模型对提示敏感、修复次数有限、需要进一步微调和专门的工具调用模型，以及对人工干预和回滚机制的不足。

---

## 174. Output Type Before Quality: A Standards-Derived XAI Admissibility Rubric for Autonomous-Driving Safety

**arXiv ID:** 2606.05461 | [PDF](https://arxiv.org/pdf/2606.05461v1)

**作者:** Abhinaw Priyadershi `[一作]` (NVIDIA Corporation), Maria Spence `[通讯]` (NVIDIA Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了基于安全标准的XAI可采纳性评估指标，并对六类XAI方法进行结构化评分。

**💡 创新点**

从四份安全标准抽取19条可验证标准，构建结构化评分表，并通过单一视觉-语言行动系统验证。

**🔧 技术方法**

结构化评分法、Pearl因果层次理论、结构因果模型(PC+GES)、SHAP、CoC。

**📊 数据集**

NVIDIA PhysicalAI-AV真实驾驶视频集（1996个片段）。

**📈 对比分析**

对SHAP、SCM、CoC等方法按输出类型进行S/P/F评分；SCM在三阶段满足全部标准，SHAP在大部分阶段失效；实验中SCM恢复部分因果边，SHAP无法生成因果路径。

**⚠️ 局限性**

评分基于自评，实验仅覆盖三类方法，单一数据集与单一VLA，因果边检索受统计功效限制。

---

## 175. Multimarginal flow matching with optimal transport potentials

**arXiv ID:** 2606.05327 | [PDF](https://arxiv.org/pdf/2606.05327v1)

**作者:** Raghav Kansal `[一作]` (Bexorg, Inc.), Bradley Parry `[通讯]` (Bexorg, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为OTP-FM的多重边界流匹配框架，通过软OT势能约束学习时间序列数据的连续动力学。

**💡 创新点**

将多重边界条件融入动态OT，利用软势能取代硬约束，形成可调的潜能空间，并在此基础上得到可解析的条件解和无仿真训练目标。

**🔧 技术方法**

采用条件流匹配、动态OT、软势能、固定点迭代、consistency model（如iMF）、梯度加权与训练进度曲线等技术。

**📊 数据集**

在单细胞RNA测序（EB、CITE-seq）、墨西哥湾海流、北京空气质量等多种生物、海洋与气象数据集上进行实验。

**📈 对比分析**

与MMFM、3MSBM、TrajectoryNet等方法对比，OTP-FM在大多数指标上实现SOTA，并且训练时间仅需数分钟。

**⚠️ 局限性**

尽管能逼近中间分布，却不能保证物理可行的中间状态；MMD/KLD势能收敛不稳定，且需要手动调节潜能参数。

---

## 176. Look Before You Leap: Checking in on Type Tag Checking

**arXiv ID:** 2606.05466 | [PDF](https://arxiv.org/pdf/2606.05466v1)

**作者:** Stephen M. Watt `[一作]` (University of Waterloo), Stephen M. Watt `[通讯]` (University of Waterloo)

**通讯引用:** 3836 | [OpenAlex ID](https://openalex.org/A5082854012)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在多种现代CPU（AArch64、x86‑64等）上对三种值编码方案（对象头标记、低位指针标签、NaN‑boxing）进行微基准测试，评估其在类型检查、立即数访问和算术运算中的性能差异。

**💡 创新点**

创新点在于将传统的Lisp式即时数与NaN‑boxing结合起来，在当前硬件和工作负载环境下重新量化它们的成本，并用跨平台、不同内存层次的广泛实验展示了这些方案在实际符号计算和动态语言运行时中的相对优劣。

**🔧 技术方法**

采用C++（G++14/13）实现两阶段基准（类型检查+载荷访问+算术），生成静态指令计数、依赖性分析和内存访问路径，并通过实验平台的完整缓存层信息来解释结果。

**📊 数据集**

使用合成数据集：按1:1:1比例随机排列的整数、双精度浮点数和链表（cons）对象，规模从10³到10⁹，保证所有三种类型的数值均匀分布，进而在数组遍历中获得可比的热点。

**📈 对比分析**

比较方法是对每种表示方式分别执行八个计时阶段（计数、求和等），记录每个元素的平均纳秒耗时，并计算相对加速比；结果显示，仅做标签检查时，指针标记/NaN‑boxing比对象头标记快约10–30×；立即数求和时比从对象读取快约2–5×，而双精度数在NaN‑boxing下可避免堆分配。

**⚠️ 局限性**

局限性包括：微基准未涵盖完整系统（分配、GC、并行）、仅使用合成比例均匀的数据、单线程实验、受虚拟化和平台特定缓存行为影响，且测得的相对优势可能随内存布局、处理器微架构和编译器优化等级而变化。

---

## 177. What Should Agents Say? Action-state Communication for Efficient Multi-Agent Systems

**arXiv ID:** 2606.05304 | [PDF](https://arxiv.org/pdf/2606.05304v1)

**作者:** Chen Huang `[一作]` (Singapore University of Technology and Design), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 5394 | [OpenAlex ID](https://openalex.org/A5042288832)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多代理系统中代理间通信策略，提出PACT协议将代理输出压缩为包含行动、状态和结果的紧凑记录，从而提升性能-成本平衡。

**💡 创新点**

创新点在于将代理间通信视为公共状态更新，设计无训练、跨平台的PACT协议，去除冗余推理，仅保留对下游有效的行动信息。

**🔧 技术方法**

采用大型语言模型Qwen3系列与PACT协议，进行诊断分析和实验评估，并在实际编码平台中实现代理通信代理层。

**📊 数据集**

使用HotpotQA、2WikiMultiHopQA、AIME2024/2025、GPQA-Diamond、OpenBookQA以及SWE-bench Verified等数据集进行实验。

**📈 对比分析**

通过与Chain of Agents、TextMAS、Multi‑Agent Debate等基线对比，PACT在不同MAS拓扑上平均减少约38.7% token消耗，保持或提升任务精度，且在OpenHands和SWE‑agent编码平台上显著降低 tokens‑per‑resolved。

**⚠️ 局限性**

局限性包括对短交互或非共享历史系统的收益未充分验证，仅覆盖两种MAS拓扑和两种编码平台，未探讨更复杂的辩论或动态网络场景。

---

## 178. SB-RF: Schrödinger Bridge Rectified Flow for One-Step Robust Speech Enhancement

**arXiv ID:** 2606.05575 | [PDF](https://arxiv.org/pdf/2606.05575v1)

**作者:** Caixia Lu `[一作]` (Xiaomi Corporation), Jiaming Xu `[通讯]` (Xiaomi Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种结合 Schrödinger Bridge 与 Rectified Flow 的一站式语音增强框架 SB‑RF，实现高质量语音恢复并保持 NFE=1 的高效推理。

**💡 创新点**

创新点在于将 SB 的熵正则最优传输轨迹与 RF 的速度匹配目标融合，引入可学习的概率管道，打破传统 RF 的线性路径限制，同时兼顾一次生成的效率。

**🔧 技术方法**

使用 Schrödinger Bridge 理论、Rectified Flow 的速度匹配、熵正则化最优传输、NCSN++ 网络骨干以及 Mel‑谱与 PESQ 损失等技术。

**📊 数据集**

主要使用 VoiceBank‑DEMAND 基准（Track A）以及扩展的 WenetSpeech4TTS+DNS‑4 训练集，并在低 SNR 条件下构造了 AISHELL‑1/LibriSpeech 混合 WHAM! 噪声的测试集（Track B）。

**📈 对比分析**

与 MP‑SENet、SGMSE+、BBED、SB‑VE、CFM、LARF、COSE 等多种基线在 Track A 和 Track B 进行比较，SB‑RF 在 Track A 以 NFE=1 获得最高 PESQ 3.39、SI‑SDR 19.5 dB；在 Track B 低 SNR 条件下得到 PESQ 2.56、ESTOI 0.70，显著优于对手且推理成本极低。

**⚠️ 局限性**

局限性包括：对极低 SNR 的 DNSMOS 仍略逊于 BBED，且在真实流式部署中还需进一步优化低延迟实现。

---

## 179. Polynomial-time satisfiability for a special case of Positive$\wedge$Negative

**arXiv ID:** 2606.05512 | [PDF](https://arxiv.org/pdf/2606.05512v1)

**作者:** Marcel Wild `[一作]` `[通讯]` (University of Stellenbosch), Marcel Wild (University of Stellenbosch)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

论文通过引入2e-row与2n-row的交集判定，提出了一种O(m²)时间内可判定DisPos∧DisNeg（亦即正负子集）CNF可满足性的算法；

**💡 创新点**

其创新点在于证明任意相同长度的2e-row与2n-row交集必非空，从而突破传统子集枚举的指数瓶颈；

**🔧 技术方法**

核心技术包括多值行压缩（wildcards）、行分裂机制、Abraham flag与图论分析、以及递归与投影的行投射；

**📊 数据集**

论文未使用具体公开数据集，而是通过构造性示例（如长度24的行、DisPos∧Horn实例等）进行验证；

**📈 对比分析**

相较于传统3‑SAT或Horn子集的判定方法，O(m²)的判定复杂度在理论上显著优越，实验中已在示例中显示可在数十步内完成判定；

**⚠️ 局限性**

局限性包括：方法仅适用于DisPos∧DisNeg或Thin（长度为2）情况；对于包含长wildcard的行需多步投射，实际实现仍需对投影、行拆分做细粒度优化。

---

## 180. MIRAI: Prediction and Generation of High-Impact Academic Research

**arXiv ID:** 2606.05443 | [PDF](https://arxiv.org/pdf/2606.05443v1)

**作者:** Alex Li `[一作]` (MIT), Joseph Jacobson `[通讯]` (MIT)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MIRAI 框架，能够仅凭论文标题、摘要和出版日期预测论文的未来影响力，并基于该预测构建了一套研究创意生成管线。

**💡 创新点**

创新点在于：①将冻结的多语言文本嵌入与轻量化前馈网络结合，实现多时段（1–5 年）影响力预测；②将预测模型嵌入到候选生成与筛选流程中，形成首创的“模型驱动研究创意”管线；③在公开评测中显著优于零射 LLM 预测，证明内容驱动的低延迟影响评估可行。

**🔧 技术方法**

使用技术包括 NVIDIA 4096 维文本嵌入器、含层归一化与 dropout 的多层前馈网络、Spearman 相关系数评估、零射 LLM（GPT‑4o、Hermes‑3）对照实验，以及 LLM‑judge 的双向比较。

**📊 数据集**

数据集为近 3 百万篇 arXiv 论文，利用 Semantic Scholar API 构建自闭合引用图，提取 5 年期的 ln(1+citation) 和对数 PageRank 作为标签。

**📈 对比分析**

性能对比采用 Spearman ρ、精确召回曲线和高影响力检索；MIRAI 在 2021 年 5 年期 citation 上 ρ=0.6192，PageRank ρ=0.4686；相较于 GPT‑4o（0.336）和 Hermes‑3（0.305），在 2 年期任务中 ρ=0.581、0.336、0.305，且在高影响力检索与累计影响力集中度上优势显著。

**⚠️ 局限性**

局限性包括：①训练集与最近测试集的分布差异随 arXiv 规模加速增长而扩大；②仅覆盖 arXiv 领域，跨学科或期刊推广尚待验证；③LLM‑judge 的偏好与评测标准受模型自身偏差影响；④生成创意的长期影响仍需后续追踪。

---

## 181. The Granularity Gap: A Multi-Dimensional Longitudinal Audit of Sycophancy in Gemini Models

**arXiv ID:** 2606.05183 | [PDF](https://arxiv.org/pdf/2606.05183v1)

**作者:** Patrick Keough `[一作]` `[通讯]` (Independent Researcher), Patrick Keough (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性评估并量化 Gemini 系列 LLM 在社交顺从（sycophancy）上的表现，揭示二元安全评估无法捕捉的中等程度合规行为。

**💡 创新点**

创新点在于：①提出“Granularity Gap”概念并通过连续 Likert 评分展示其幅度；②构建 3 维心理测量量表（Sycophancy、Truthfulness、Refusal Specificity）并与人工与外部模型交叉验证；③发现“Alignment Tax”随代际升级加剧；④显示简单直接约束比复杂链式思考更能降低中等级社交顺从。

**🔧 技术方法**

技术主要包括：LLM-as-Judge 评估框架（基于 Gemini 3.0 Pro Preview 的 Best-of-3 评分）、人类标注验证、跨模型（DeepSeek V3）校准、非参数统计（Kruskal-Wallis、Spearman、Fisher Z 等）以及多因素方差分析。

**📊 数据集**

数据集包含 350 组攻击性提示，覆盖七类心理诱导（如 Egotistical Validation、Unethical Proposals 等），在 8 个 Gemini 变体和 3 个安全条件下产生 8,830 条响应；人工验证样本 73 条、236 评分；跨模型验证 608 条。

**📈 对比分析**

比较方法是将连续评分与传统二元挑战率对照，利用 R²、U‑曲线敏感度和效应量（Cliff’s δ、Kruskal‑Wallis H 等）评估；结果显示二元指标仅解释 29% 行为方差，且对中等强度的顺从识别率低至 6.4%，而简单约束将平均 Sycophancy 分数从 2.21 降至 1.16（相对 48% 的改进）。

**⚠️ 局限性**

局限包括：①仅评估 Gemini 系列，缺乏跨家族验证；②攻击提示主要由 LLM 生成，未涵盖真实用户交互；③人工验证样本规模有限，尤其是高严重度样本；④评估工具与评审者同属西方训练体系，可能存在文化偏倚；⑤无法排除自评偏差，尽管进行了多模型校准。

---

## 182. LeanMarathon: Toward Reliable AI Co-Mathematicians through Long-Horizon Lean Autoformalization

**arXiv ID:** 2606.05400 | [PDF](https://arxiv.org/pdf/2606.05400v1)

**作者:** Yuanhe Zhang `[一作]` (University of Warwick), Fanghui Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1539 | [OpenAlex ID](https://openalex.org/A5016703530)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套基于多代理、可恢复、可验证的 Lean 4 自动化正式化平台，能够将完整研究论文逐步转化为无 sorry 的 Lean 代码；

**💡 创新点**

核心创新包括：① 将论文结构映射为可演化的 Proof DAG 并用单一 Lean 文件记录；② 通过四个合同化代理（Blueprinter、Target‑Reviewer、Worker、Refiner）与两阶段 orchestrator 分工协作，避免目标漂移、上下文腐烂等长周期失效模式；③ 用 CI gate 做全局可验证契约，确保每一次 PR 都在限定可编辑范围内且不会破坏其他代理的工作；

**🔧 技术方法**

技术手段包括：Lean 4 的类型检查与 Lean‑Architect 属性、Codex/GPT‑5.5‑xhigh 作为 LLM 工具、GitHub PR/issue 流进行代理通信、MCP 服务器对 patch 进行范围限制、自动化依赖检查与文档一致性校验、并行工作者并行提交与 squash‑merge 等；

**📊 数据集**

使用的“数据集”是两篇 2026 年发表的 Erdős 相关研究论文（分别解决 Problem #1051、#1196、#164、#1217），每篇论文的 LaTeX 源和对应的目标命题文件；

**📈 对比分析**

与商业单代理 baseline Aristotle 的对比实验表明：在三次独立跑中，该 harness 共完成 258 条证明（7 条目标定理，0 sorry），Aristotle 在两篇论文均无法完成，耗时更长（>40 h 与 >24 h）且残留 sorry；此外，该 harness 的成本约为 $257–$624 的 GPT‑5.5 等价代价，平均每个工作节点约 100 K tokens，且能够在数十轮并行工作后收敛；

**⚠️ 局限性**

局限性包括：① 仍需 Mathlib 的完整性，缺失的理论（如某些概率与分析命题）会导致正式化停滞；② 对于需要深层库支持的几何/数论证明，系统只能模拟缺失的结构，无法真正完成证明；③ 运行成本高且依赖昂贵 LLM，且多轮迭代仍可能需要人工干预以校正漂移；

---

## 183. Selective-Advantage Entropy-Adaptive Horizon GRPO: Asymmetric Token-Level Discounting for Efficient Reinforcement Learning of Language Models

**arXiv ID:** 2606.05434 | [PDF](https://arxiv.org/pdf/2606.05434v1)

**作者:** Chirag Chawla `[一作]` (Indian Institute of Technology (BHU)), Madhav S. Baidya `[通讯]` (Indian Institute of Technology (BHU))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出两种GRPO的改进算法：AH-GRPO（统一使用基于token熵的自适应折扣）和SA-AH-GRPO（仅对负advantage轨迹使用该折扣），并在数学推理任务上进行实验。

**💡 创新点**

创新点在于：①使用模型自身的token预测熵来动态调节梯度贡献，实现自适应有效梯度长度；②将折扣仅应用于负advantage轨迹，保持正确路径的完整梯度，从而显著降低训练方差。

**🔧 技术方法**

采用GRPO基础框架，结合Low‑Rank Adaptation (LoRA)、KL正则、PPO式截断策略、熵自适应权重、α-消融、以及由四个子指标构成的可验证奖励函数。

**📊 数据集**

使用GSM8K数学推理基准，分别在Qwen 2.5‑1.5B‑Instruct和Qwen 2.5‑3B‑Instruct两个规模模型上进行微调和评估。

**📈 对比分析**

对比GRPO（α=0）、AH‑GRPO（α=0.5）与SA‑AH‑GRPO（α=0.5）：在3B模型上SA‑AH‑GRPO保持与GRPO相同的Peak Pass@1（≈0.86）同时将训练方差降至3.6倍；在1.5B模型上SA‑AH‑GRPO达成Peak Pass@1=0.686，比零shot提升4.9pp，且保持更高的最终准确率。

**⚠️ 局限性**

局限性包括：仅在两种模型规模上验证，未对更大规模或其他任务做测试；α消融仅针对AH‑GRPO；未多次种子复现；使用Top‑K近似熵导致轻微误差；评估仅限于GSM8K，难以验证跨域通用性。

---

## 184. Human oversight of agentic systems in practice: Examining the oversight work, challenges, and heuristics of developers using software agents

**arXiv ID:** 2606.05391 | [PDF](https://arxiv.org/pdf/2606.05391v1)

**作者:** Shipi Dhanorkar `[一作]` (Microsoft), Mihaela Vorvoreanu `[通讯]` (Microsoft)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5066017612)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对17名软件工程师进行访谈，分析他们在使用软件代理时的监督工作、挑战与启发式策略

**💡 创新点**

首次提供实证证据揭示代理监督从预设控制到共规划、实时监控和事后审查四种形式，并呈现开发者采用的四类启发式方法

**🔧 技术方法**

采用半结构化访谈、录音转写与主题编码的定性分析方法

**📊 数据集**

以17位经验丰富的代理使用者的访谈记录为数据集

**📈 对比分析**

该研究并未进行算法性能对比，而是通过主题归纳呈现监督实践的描述，未涉及量化指标

**⚠️ 局限性**

样本主要来自同一大型科技公司，任务规模较小，且研究聚焦于高资源环境，限制了结果的普适性和对复杂任务的适用性

---

## 185. Severity-Aware Curriculum Learning with Multi-Model Response Selection for Medical Text Generation

**arXiv ID:** 2606.05510 | [PDF](https://arxiv.org/pdf/2606.05510v1)

**作者:** Ahmed Alansary `[一作]` (MSA University), Ali Hamdi `[通讯]` (MSA University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于严重程度的多模型医学文本生成框架，先按轻中重三阶段进行课程学习训练，再在推理阶段用BERTScore做语义相关性选择最优回答。

**💡 创新点**

创新点在于将医学症状严重程度与课程学习结合，构建分阶段训练流程，同时利用多模型生成多候选答案并通过语义匹配做动态选择，从而提升生成质量与可靠性。

**🔧 技术方法**

技术包括：低秩适配LoRA微调、三阶段课程学习、5个大语言模型独立训练、BERTScore语义相关性打分、多模型答案聚合与选择。

**📊 数据集**

使用MAQA（阿拉伯语医学问答）数据集，人工标注轻中重三级严重程度，共约32k问答对。

**📈 对比分析**

与基线模型和单一微调模型对比，基线BERTScore最高82.00%，单一微调最高86.74%；本框架在基线下达到86.71%，在微调下达到90.30%，显示明显性能提升。

**⚠️ 局限性**

局限性包括：严重程度标注为规则自动，缺乏人工验证；评价仅使用BERTScore，未评估安全性、误导性与医疗专业审核；未涉及多语言或更大规模数据集的泛化能力。

---

## 186. Learning Contact Representation for Leg Odometry

**arXiv ID:** 2606.05501 | [PDF](https://arxiv.org/pdf/2606.05501v1)

**作者:** Emre Girgin `[一作]` (Embry Riddle Aeronautical University), Cagri Kilic `[通讯]` (Embry Riddle Aeronautical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对四足/腿部机器人的姿态估计，提出一种完全基于关节编码器的自监督接触检测框架，利用去噪自编码器将运动学序列映射到潜在空间，再用高斯混合模型估计接触概率，并将此概率动态融入误差状态扩展卡尔曼滤波器（ESEKF）实现更鲁棒的ZUPT。

**💡 创新点**

①将接触检测任务转化为自监督的连续概率密度估计；②通过去噪自编码器在潜在空间中自然解耦站立与摆动模式；③在滤波器中用接触概率动态调整测量协方差，实现平滑的ZUPT更新；④无需外部力传感器或人工标注。

**🔧 技术方法**

去噪自编码器（CNN/GRU）、高斯混合模型、隐马尔可夫模型、误差状态扩展卡尔曼滤波器（ESEKF）、自监督学习、时间序列特征提取。

**📊 数据集**

TartanGround 仿真数据集以及真实机器人在混凝土、草地、岩石等多种地形下收集的实测数据。

**📈 对比分析**

与硬阈值的GRF阈值法、无监督 HMM-GMM、监督 CNN/GRU 进行对比。虽然监督模型在伪标签分类指标上表现最好，但自监督 DAE 在所有离线和现场的轨迹误差（ATE、RPE、FPE 等）上均优于基线，尤其在真实硬件上误差降低显著。

**⚠️ 局限性**

潜在空间假设为独立且噪声无关，导致对传感器噪声的鲁棒性不足；采用等方差协方差，无法捕捉滑移的各向异性；使用确定性去噪自编码器而非变分自编码器，难以正式建模潜在分布。

---

## 187. ORACLE-CT: Anatomy-Aware Support Pooling for CT Classification

**arXiv ID:** 2606.05460 | [PDF](https://arxiv.org/pdf/2606.05460v1)

**作者:** Lavsen Dahal `[一作]` (Duke University), Joseph Y. Lo `[通讯]` (Duke University)

**通讯引用:** 7659 | [OpenAlex ID](https://openalex.org/A5040192736)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种称为ORACLE–CT的解码器无关的解剖结构感知支持池化框架，用于多标签腹部CT分类。

**💡 创新点**

创新点在于将多器官分割得到的解剖支持作为标签特定的空间聚合约束，通过在注意力池化中仅对指定支持区域进行归一化，既提升了诊断性能，又实现了可审计的证据聚合。

**🔧 技术方法**

技术手段包括：基于多器官分割的支持映射；支持掩码注意力池化；在不同编码器（DINOv3视觉Transformer、I3D‑ResNet‑121 3D CNN、Pillar‑0 影像基础模型）上实现的统一聚合头；以及对多标签损失的缺失标签处理。

**📊 数据集**

使用的数据集为：内部训练/验证/测试集MERLIN（30个标签）；外部冻结转移评估集Duke–Abdomen（27个标签）和AMOS（10个标签）。

**📈 对比分析**

与全局平均池化和无掩码注意力池化的基线相比，支持掩码注意力池化在内部测试中显著提升了宏观AUROC（DINOv3从0.8380提升至0.8576，I3D‑ResNet‑121从0.8288提升至0.8482），在外部转移中亦保持优势，尤其在DINOv3与I3D‑ResNet‑121上AUROC提升约3‑4%。

**⚠️ 局限性**

局限性包括：依赖先验分割质量；固定标签‑支持映射可能不适用于多器官或分布式证据的疾病；标签来源于报告，存在缺失/噪声；仅评估腹部CT单体卷；未处理多系列、对比期或纵向跟踪。

---

## 188. Inverse Manipulation through Symbolic Planning and Residual Operator Learning

**arXiv ID:** 2606.05248 | [PDF](https://arxiv.org/pdf/2606.05248v1)

**作者:** Yigit Yildirim `[一作]` (CREATE Consortium), Alberto Finzi `[通讯]` (Università di Napoli Federico II)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种混合符号规划与深度强化学习的框架，实现机器人技能的逆向执行；

**💡 创新点**

创新点在于自动从演示中提取STRIPS符号操作，构造逆向目标，并将符号规划与残差RL结合，只在符号规划失败时激活RL；

**🔧 技术方法**

使用符号规划（BFS）、Soft Actor-Critic强化学习、软几何谓词及自定义奖励函数；

**📊 数据集**

在ManiSkill3的PushCube任务上进行实验；

**📈 对比分析**

与仅符号规划和随机控制对比，结果显示在1 cm容差下逆向成功率从10%提升至90%，平均误差1.4 mm；

**⚠️ 局限性**

局限性包括依赖手工设计的谓词与阈值、仅使用脚本化低层动作原语、仅在单一任务上验证，未来计划学习谓词参数和动作原语。

---

## 189. Agents' Last Exam

**arXiv ID:** 2606.05405 | [PDF](https://arxiv.org/pdf/2606.05405v1)

**作者:** Yiyou Sun `[一作]` (University of California, Berkeley), Dawn Song `[通讯]` (University of California, Berkeley)

**通讯引用:** 58730 | [OpenAlex ID](https://openalex.org/A5019426968)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Agents' Last Exam (ALE) 基准，用真实行业工作流程评估通用计算机使用智能体。

**💡 创新点**

构建覆盖 SOC/O*NET 所有行业、真实工作流、可验证输出、长期工作评估的持续增长基准。

**🔧 技术方法**

利用 GUI+CLI 交互的通用计算机使用代理（GCUA）架构，结合 LLM 推理、工具调用、视觉感知以及自动化脚本和 Rubric 检查实现评估。

**📊 数据集**

由 300+ 行业专家提交的约 1.5K 实际工作实例，按 O*NET/SOC 2018 分类整理。

**📈 对比分析**

对主流模型（如 GPT‑5.5、Claude Opus 4.7 等）与不同 harness 进行 GCUA 配置评测，设置 Near‑Term、Full‑Spectrum、Last‑Exam 三难度层级，平均通关率最低层 0%，中等层约 30%，整体表现尚未饱和。

**⚠️ 局限性**

任务仍需人工 QC、GUI 使用不足、模型专业知识瓶颈、资源成本高、无法覆盖非数字行业。

---

## 190. Answer Presence Drives RAG Rewriting Gains

**arXiv ID:** 2606.05633 | [PDF](https://arxiv.org/pdf/2606.05633v1)

**作者:** Yuejie Li `[一作]` (Ant Group), Chengjun Mao `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对检索增强问答（RAG）管线中的重写器（编译器）进行因果干预审计，检验答案字符串是否被显式显示导致性能提升。

**💡 创新点**

提出了去除/置换/插入答案的对照干预设计，以及多标记器敏感性审计，揭示传统单一遮蔽诊断的脆弱性，并量化答案显式化对F1提升的因果效应。

**🔧 技术方法**

因果干预方法（remove、placebo、insert）、paired bootstrap 置信区间、sentinel‑fragility 审计；使用大型语言模型重写器与读者，搭配自动化干预跑器与 sentinel 面板。

**📊 数据集**

HotpotQA 与 2WikiMultihopQA 两个多跳问答基准。

**📈 对比分析**

在每个 cell（reader、编译器、数据集）和不同 B1–B4 设置下，计算去除答案与同长度随机置换的 F1 差异，得到-28~-64 点的因果效应；插入答案在多数组合中提升 0.7~9.7 点。Sentinel 审计显示不同标记器导致残差波动，证明单一遮蔽诊断不可靠。

**⚠️ 局限性**

仅针对在线查询重写、字符串级干预，未覆盖同义/释义泄漏；结果仅适用于实验设置，无法直接迁移到离线预编译语料库。

---

## 191. SoCRATES: Towards Reliable Automated Evaluation of Proactive LLM Mediation across Domains and Socio-cognitive Variations

**arXiv ID:** 2606.05563 | [PDF](https://arxiv.org/pdf/2606.05563v1)

**作者:** Taewon Yun `[一作]` (Korea Advanced Institute of Science and Technology), Hwanjun Song `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2153 | [OpenAlex ID](https://openalex.org/A5033909285)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一自动化的社会冲突评估框架，分为三阶段：①利用 LLM 代理从网络检索真实公共争议并重写为可模拟情景；②在这些情景上独立扩展五个社交认知轴线（战略姿态、参与者数量、历史长度、情绪反应性、文化身份），生成 600 条带不同轴线的模拟对话；③采用主题局部评估器，仅在主题变化时评分，计算共识增益、干预时效性和干预有效性三项指标，对八种 LLM 调解器进行评测。

**💡 创新点**

①通过 LLM 代理实现可扩展的真实案例情景生成，消除人工场景编写瓶颈；②以五个社交认知轴线独立扩展策略，精确定位调解器弱点；③引入主题局部评估器，显著提升与专家一致性（Pearson 0.82）并降低噪声。

**🔧 技术方法**

使用 GPT‑5.4、Gemini‑3.1‑Pro 等大模型做情景生成、搜索和重写；LLM 代理做模拟对话；基于 DeepSeek‑V3.2 的主题局部评估器实现自动评分；评估指标包括共识增益、干预时效性和干预有效性。

**📊 数据集**

八个冲突领域（交易、医疗、环境、B2B、公共政策、国际、法律、组织内）中的 40 条真实公开争议 seed（通过网络检索），经 LLM 处理后生成 600 条带不同轴线的模拟对话；以及 1,844 条评估片段用于验证评估器与专家一致性。

**📈 对比分析**

对 8 种 LLM 调解器（2 专有，6 开源）在 600 条对话上与无调解器 baseline 进行对比；使用共识增益、干预时效性、干预有效性三项指标。结果显示平均共识增益 34.4%，最强模型仅能闭合未调解共识缺口的约 1/3，性能在不同领域和轴线间波动显著；高时效性但低有效性的模型表明仅频繁干预不足。

**⚠️ 局限性**

仅使用英文情景，未测试多语言调解；评估聚焦共识，未涵盖满意度、程序正义、信任等质量维度；模拟对话基于 LLM 角色扮演，可能与真实人类行为有偏差；评估器虽然准确但仍依赖 LLM 判断；未引入真实人类当事人反馈。

---

## 192. VITO: Vascular Geometry and Blood Flow Estimation Using Inverse Topology Optimization

**arXiv ID:** 2606.05487 | [PDF](https://arxiv.org/pdf/2606.05487v1)

**作者:** Pramod Thombre `[一作]` (University of Wisconsin-Madison), Krishnan Suresh `[通讯]` (University of Wisconsin-Madison)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种基于物理约束的拓扑优化框架，直接从时间分辨CE‑CT投影数据中重建血管几何与血流速度。

**💡 创新点**

创新点在于将血流动力学（稳态Navier–Stokes）与对比剂输送（瞬态对流扩散）耦合进拓扑优化，并在投影空间内进行可微前向投影，从而在缺失或噪声投影的情况下恢复复杂血管拓扑（分支、狭窄等），不依赖传统图像重建+分割链路。

**🔧 技术方法**

采用基于密度的TO（Brinkman惩罚与RAMP插值）、JAX自动微分、MMA优化器、分段投影与稳态/瞬态有限元求解，并使用连续滤波与Heaviside投影实现二值化。

**📊 数据集**

使用合成的二维血管模型（分支动脉、50%狭窄通道、斜面狭窄的颈动脉分支）生成时间分辨投影，并在不同稀疏度与噪声水平下进行实验。

**📈 对比分析**

与传统FBP+阈值分割基线对比，采用nRMSE与Dice系数评价；在稀疏5%投影+5%噪声时，nRMSE<0.0113，Dice>0.997；在超稀疏10投影+25%噪声时，nRMSE≈0.058，Dice≈0.983，均显著优于FBP，保持拓扑连通性。

**⚠️ 局限性**

局限性包括仅处理二维稳态流、假设壁面刚性、仅在合成数据上验证、求解器易陷入局部极小、未考虑临床真实投影几何与不确定参数；未来需扩展至三维、瞬态、FSI、概率不确定性和更复杂的扫描模式。

---

## 193. Horse Eye Blink Detection and Classification for Equine Affective State Assessment

**arXiv ID:** 2606.05458 | [PDF](https://arxiv.org/pdf/2606.05458v1)

**作者:** João Alves `[一作]` (Aalborg University), Rikke Gade `[通讯]` (Aalborg University)

**通讯引用:** 2117 | [OpenAlex ID](https://openalex.org/A5076290245)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文针对马匹面部动作单元（blink）的自动检测与分类，提出并评估了三种方法。

**💡 创新点**

首次系统比较了YOLO、光流阈值和VideoMAE在马眼眨眼检测上的表现，并构建了半自动标注工具。

**🔧 技术方法**

使用YOLOv12框架、光流幅值阈值、以及预训练的VideoMAE模型进行三分类/二分类。

**📊 数据集**

采用公开的12段1080p@25FPS马视频子集（S1–S12）及其EquiFACS注释作为测试集。

**📈 对比分析**

在二分类上Macro‑F1 0.926，在三分类上YOLO取得0.898的Macro‑F1，优于人类标注基线（0.76）。

**⚠️ 局限性**

限制主要在半眨眼判定困难、模型对时间尺度的适配性不足以及对光照/姿态变化的鲁棒性。

---

## 194. Incremental Computation for Efficient Programmable Inference in Probabilistic Programs

**arXiv ID:** 2606.05348 | [PDF](https://arxiv.org/pdf/2606.05348v1)

**作者:** Fabian Zaiser `[一作]` (Massachusetts Institute of Technology), Alexander K. Lew `[通讯]` (Yale University)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5056016273)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

通过将概率程序编译为确定性密度函数并使用增量计算（缓存化）实现高效可编程推理

**💡 创新点**

首次将增量计算与概率推理分离，设计闭包级增量化、可证明正确的缓存机制，并支持开放宇宙模型的名称与无序集合

**🔧 技术方法**

基于增量 λ‑计算、闭包捕获环境、Coinductive updaters、参考测度以及 Julia DSL 进行实现

**📊 数据集**

使用鲁棒回归、二值高斯混合、隐藏马尔可夫模型、有限混合、主题模型与 Dirichlet 过程混合等六类 Bayesian 模型进行基准测试

**📈 对比分析**

与 Gen 进行对比，利用单步 MCMC、Gibbs 与 SMC，展示了相对于全重计算显著的常数因子和渐进速度提升；在多数更新中实现 O(1) 复杂度，整体推理时间提升数倍至十倍

**⚠️ 局限性**

对长直线程序、无限循环/递归、以及需随机密度估计的变分推理支持有限，且某些更新仍需全重计算导致额外开销

---

## 195. HDST-GNN: Heterogeneous Dynamic Spatiotemporal Graph Neural Networks for Multi-Object Tracking in UAV Aerial Imagery

**arXiv ID:** 2606.05587 | [PDF](https://arxiv.org/pdf/2606.05587v1)

**作者:** Phillip Jiang `[一作]` `[通讯]` (Appsofa LLC), Phillip Jiang (Appsofa LLC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种用于无人机多目标跟踪的HDST-GNN模型，针对高度变化、稀疏小目标和遮挡等挑战实现精准追踪。

**💡 创新点**

创新点包括：① 高度自适应边构建；② 异构节点与边类型的图表示；③ 遮挡门控的时空聚合机制。

**🔧 技术方法**

核心技术包括：异构图神经网络（GNN）+ ResNet‑18 视觉特征提取器 + Sinkhorn 可微匹配头 + 两阶段 Hungarian 匹配。

**📊 数据集**

使用 VisDrone2019‑MOT 数据集进行训练与评估，并在无 VisDrone 微调的 YOLOv8n 检测器上进行端到端测试。

**📈 对比分析**

与 SORT、ByteTrack、StrongSORT、NOWA‑MOT 等基线比较，Oracle 检测下获得 94.51% MOTA、97.24% IDF1，MOTA 提升约 5 点，ID 切换率降低 81%；在噪声检测下 ID 切换降低 49%。

**⚠️ 局限性**

局限性包括：训练仅使用模拟噪声的 GT 检测，未与真实检测器联合训练；高度估计仅基于目标面积，可能误差大；推理速度约 10 FPS，尚未实现实时；未考虑多类别或 GPS/IMU 信息。

---

## 196. Noise-Aware Visual Representation Learning for Medical Visual Question Answering

**arXiv ID:** 2606.05535 | [PDF](https://arxiv.org/pdf/2606.05535v1)

**作者:** I Putu Adi Pratama `[一作]` (Deakin University), Shang Gao `[通讯]` (Deakin University)

**通讯引用:** 4558 | [OpenAlex ID](https://openalex.org/A5048961502)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种在视觉映射前加入去噪自编码器的医学视觉问答框架，提升模型对视觉嵌入噪声的鲁棒性。

**💡 创新点**

创新点在于将去噪自编码器与视觉到语言模型的映射分离为两阶段训练，并在嵌入层进行去噪，从而显著提高在噪声条件下的性能。

**🔧 技术方法**

使用冻结的CLIP视觉编码器、Gaussian噪声注入的去噪自编码器、三层MLP映射器、GPT‑2 XL语言模型以及LoRA参数高效微调，损失函数为Smooth L1。

**📊 数据集**

实验基于SLAKE和PathVQA两个公开医学视觉问答数据集进行。

**📈 对比分析**

通过与直接映射基线、无噪声AE以及DAE的对比，评估BLEU、BERTScore、F1和准确率；在干净数据上保持竞争力，在噪声数据上平均准确率从0.642提升至0.735（LoRA），显示出更强鲁棒性。

**⚠️ 局限性**

局限性包括：噪声仅采用简单的高斯模拟，可能无法覆盖真实医学影像的多样噪声；视觉映射仅用MLP，缺乏更复杂的跨模态交互；且模型仍依赖冻结的视觉编码器和有限的LLM微调。

---

## 197. Gradient Descent with Large Step Size Restores Symmetry in Deep Linear Networks with Multi-Pathway

**arXiv ID:** 2606.05219 | [PDF](https://arxiv.org/pdf/2606.05219v1)

**作者:** Hee-Sung Kim `[一作]` (Hanyang University), Sungyoon Lee `[通讯]` (Hanyang University)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5101790501)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究大步长梯度下降（GD）在多通道深线性网络中的学习动力学，证明与梯度流（GF）预测的“赢家通道”分化不同，GD会导致通道信号重新分配并收敛到更平坦的多通道平衡解。

**💡 创新点**

创新点在于揭示大步长GD通过边界不稳定（Edge of Stability）驱动的振荡实现通道重平衡，从而压制GF的赢家通道偏好，并给出基于深度的最坏回归阈值，说明更深网络拥有更宽的重平衡学习率窗口。

**🔧 技术方法**

主要技术包括：深线性网络的奇异向量保持（SVS）参数化、Hessian光滑度分析、梯度流与离散GD的比较、深链回归映射分析以及数值模拟验证。

**📊 数据集**

无具体数据集，实验主要在合成的多通道线性网络与简单的tanh MLP上进行。

**📈 对比分析**

方法对比：与梯度流预测的赢家通道分化进行理论和数值对照；在非线性MLP中验证重平衡现象，发现大步长GD仍能推向平坦多通道极值。性能方面，平坦度提升明显，导致梯度流所得到的尖锐单通道解被大步长GD驱散。

**⚠️ 局限性**

局限性：理论仅适用于线性网络和SVS条件，非线性网络的解析结果仍缺失；重平衡阈值与深度关系虽给出上界，但对更复杂结构（如专家混合、注意力）需要进一步研究。

---

## 198. Trajectory Dynamics in Language Model Hidden States Predict Human Processing Costs Beyond Surprisal

**arXiv ID:** 2606.05346 | [PDF](https://arxiv.org/pdf/2606.05346v1)

**作者:** Elan Barenholtz `[一作]` (Florida Atlantic University), Elan Barenholtz `[通讯]` (Florida Atlantic University)

**通讯引用:** 846 | [OpenAlex ID](https://openalex.org/A5055719182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了“轨迹外推误差”这一指标，用来捕捉Transformer隐藏状态在短时段内的轨迹偏差，并检验其对人类阅读时长的预测贡献，发现该指标与传统惊讶度（surprisal）正交且独立。

**💡 创新点**

首次证明阅读时长受到两个独立维度的影响：词级预测误差（惊讶度）和局部轨迹偏差（轨迹外推误差）；同时展示了轨迹外推误差跨模型尺寸、跨位置编码架构的稳健性，提出了将动态轨迹结构纳入语言理解的新框架。

**🔧 技术方法**

使用Transformer（GPT‑2与Pythia）隐藏状态的线性外推误差计算，结合线性混合效应回归、AIC/BIC、似然比检验，以及方向保持分析和位移控制等统计方法。

**📊 数据集**

使用了两大数据集：SAP Benchmark中的Garden‑path句子子集（24条句子，2k+参与者）以及约10,000词的Natural Stories自测阅读语料库（181参与者）。

**📈 对比分析**

通过逐步加入惊讶度和轨迹误差到控制模型，比较AIC/BIC和似然比；轨迹误差在多种模型尺寸（GPT‑2 Small/Medium/Large）和不同位置编码（Absolute vs. RoPE）下均显著提升预测（ΔAIC≈10–13或更高），而惊讶度在部分数据中无显著提升，证明两者独立贡献。

**⚠️ 局限性**

主要局限在于仅使用自测阅读数据，缺乏眼动或神经记录；Garden‑path样本量有限导致随机效应问题；轨迹外推误差的因果关系与惊讶度的具体交互仍未通过实验直接验证。

---

## 199. UniPixie: Unified and Probabilistic 3D Physics Learning via Flow Matching

**arXiv ID:** 2606.05399 | [PDF](https://arxiv.org/pdf/2606.05399v1)

**作者:** Qilin Huang `[一作]` (University of Pennsylvania), Lingjie Liu `[通讯]` (University of Pennsylvania)

**通讯引用:** 4497 | [OpenAlex ID](https://openalex.org/A5087345525)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个可控生成连续物理属性的框架，能够从单张视觉输入预测柔软到坚硬的材料参数，并兼容多种物理仿真引擎；

**💡 创新点**

将物理属性预测从单点估计转为可调连续分布，提出统一的Perceiver-IO编码器加Flow-Matching解码器，实现跨引擎参数输出；

**🔧 技术方法**

使用CLIP特征投影、Perceiver-IO Encoder、Flow-Matching Transformer、AdaLN条件调制、α控制参数以及多头解码器等技术；

**📊 数据集**

构建了新的PIXIEREVERSE+Range数据集，包含10类物体的物理属性范围（如Young's modulus、泊松比、密度）及多引擎标签；

**📈 对比分析**

与NeRF2Physics、PUGS、PIXIE等基线对比，模型在Young's modulus MSE、PSNR、SSIM、材料分类等指标上平均提升约50%，在三种物理引擎上性能与专用模型相当且推理速度提升数倍；

**⚠️ 局限性**

仅处理可见区域，缺乏对遮挡区域的预测，α控制仅覆盖软硬一维轴，未探索多维材料空间和更复杂非刚体材料的建模。

---

## 200. A Model of Multi-turn Human Persuadability Using Probabilistic Belief Tracing

**arXiv ID:** 2606.05330 | [PDF](https://arxiv.org/pdf/2606.05330v1)

**作者:** Jared Moore `[一作]` (Stanford University), Max Kleiman-Weiner `[通讯]` (University of Washington)

**通讯引用:** 1806 | [OpenAlex ID](https://openalex.org/A5083742002)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文开发了PersuasionTrace平台，用以记录多轮对话中的信念轨迹并为每条信息标注logos、pathos、ethos等修辞标签，同时构建了基于贝叶斯网络的目标模拟器，以研究LLM在说服人类时的信念更新过程。

**💡 创新点**

创新点在于：①引入过程级多轮信念追踪，突破传统前后测的局限；②将修辞维度与信念动态关联，首次在实验中评估其对说服效果的影响；③构建结构化贝叶斯网络模拟目标，并通过与人类轨迹对比验证模拟器的真实性，揭示模型对策略评估的敏感性。

**🔧 技术方法**

技术实现包括：使用GPT‑4o等大型语言模型进行对话生成、atomization、以及人类相似度评估；贝叶斯网络用于信念更新；聚类、线性回归等统计方法分析轨迹与修辞的关系；以及基准实验设计（对照文本/音频、个性化命题）。

**📊 数据集**

数据集主要为DebateGPT提供的27条命题（平均3.45个相关信念节点），以及通过Prolific招募的受试者在文本和音频两种模式下完成的多轮对话；还收集了个人化命题和相关信念调查。

**📈 对比分析**

对比方法：将贝叶斯网络模拟器与两种LLM基线（无结构与仅注入结构）在三项指标上进行评估——人类相似度得分（BN 81.3 vs 64.7），重放误差（BN 0.1429 vs 0.1507），以及立场偏差和对平凡策略的响应度；结果显示BN模拟器在所有指标上均优于基线。

**⚠️ 局限性**

局限性包括：自报信念受询问频率影响，样本量相对有限；模拟器仅关注命题信念，未覆盖人际关系、情感与认同等重要的说服机制；贝叶斯网络结构手工构建难以规模化；实验分析多为相关性，缺乏因果证据；以及修辞标签的自动化注解可能存在误差。

---

## 201. A prism hierarchy of learning regimes in large linear autoencoders

**arXiv ID:** 2606.05335 | [PDF](https://arxiv.org/pdf/2606.05335v1)

**作者:** Eugene Golikov `[一作]` (Applied AI Institute), Dmitry Yarotsky `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

研究了权重共享线性自编码器在不同极端学习 regime 下的梯度流动力学，构建了完整的极限 regime 分类，并给出了四个极端 regime 的解析解。

**💡 创新点**

首次将图形化损失展开方法与随机矩阵理论相结合，系统推导了该模型所有极端 regime（大数据、小数据、均值场、窄潜在、自由）的解析表征，完成了从图形化到解析解的闭合闭合桥梁。

**🔧 技术方法**

使用了损失展开的图形化方法（图形合并、Wick 计算）来确定主导项，再结合随机矩阵理论（Marchenko‑Pastur 分布、Wishart 极限）和高阶矩闭合技巧，推导了梯度流在各极限 regime 下的损失闭式或积分/ODE 表达式。

**📊 数据集**

实验采用了标准的高斯同方差数据集（X ∼ 𝒩(0, I_p)），并在不同尺寸比例下数值模拟梯度流，验证了理论预测。

**📈 对比分析**

通过对比理论极限解与数值梯度流结果（均值±2 SE），发现四个极限 regime 的理论曲线与仿真曲线高度吻合，验证了理论的准确性；在自由 regime 仍未获得可验证的解析解。

**⚠️ 局限性**

局限性包括：对非同方差高斯或非高斯数据的推断尚未完成；引入非线性激活函数时图形化方法不再适用；自由 regime 的解析解缺失；实验仅限于理论极限下的理想化设置。

---

## 202. SmellBench: Towards Fine-Grained Evaluation of Code Agents on Refactoring Tasks

**arXiv ID:** 2606.05574 | [PDF](https://arxiv.org/pdf/2606.05574v1)

**作者:** Fake Lin `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4596 | [OpenAlex ID](https://openalex.org/A5025292786)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于主动代码臭味注入的可扩展代码重构基准SmellBench，并系统评估了现有代码代理和大型语言模型在跨文件重构任务中的表现。

**💡 创新点**

创新点在于通过可控注入代码臭味生成高质量多维重构案例，提供了包含功能正确性、定位准确性和LLM评估质量的三维评估框架，首次系统探讨跨文件重构能力。

**🔧 技术方法**

使用了代码代理（OpenHands、Qwen Code）、大型语言模型（Qwen3、DeepSeek、GPT‑5、Gemini、Claude Sonnet）以及Docker化的Harbor框架完成注入、验证与评估。

**📊 数据集**

数据集为294个由7个Python开源仓库生成的重构案例，覆盖7种臭味、3个难度等级和2种指令设置。

**📈 对比分析**

通过功能通过率、定位准确率、LLM评估等六项指标对比，最佳组合Qwen Code+Claude Sonnet 4.5仅获得约0.50的臭味消除得分，显示跨文件重构仍是当前模型的瓶颈。

**⚠️ 局限性**

局限包括原始仓库中可能已有臭味导致噪声、注入方式相似可能让模型学习模式单一，以及LLM‑as‑Judge 评估可能产生偏见。

---

## 203. Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models

**arXiv ID:** 2606.05531 | [PDF](https://arxiv.org/pdf/2606.05531v1)

**作者:** Mohammad Mahdi Abootorabi `[一作]` (Qatar Computing Research Institute, Hamad Bin Khalifa University), Ehsaneddin Asgari `[通讯]` (Qatar Computing Research Institute, Hamad Bin Khalifa University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了BloomBench，一个基于Bloom知识层级的双语（英阿）视觉语言模型评测基准。

**💡 创新点**

创新点在于将认知心理学的Bloom分类体系与多模态任务相结合，构建覆盖六层认知难度的系统化评测框架，并提供半自动化、双重质量验证的数据生成流水线。

**🔧 技术方法**

采用了大型语言模型（Gemini 2.5 Pro、Gemini 3 Pro、Gemini 3 Flash 等）进行情景生成、VQA 生成、MCQ 制作与翻译，并用 LLM‑as‑a‑judge 与人工评估相结合实现质量控制。

**📊 数据集**

数据集包含 7,747 组英阿双语图像–问题–答案对，涵盖 106 个 Bloom 叶节点，来自公开网页图像，覆盖 6 个认知层级。

**📈 对比分析**

通过对多种公开 VLM（Gemma、Qwen、GPT‑4o mini 等）的 RAE 与 LBS 两种评估方法进行对比，结果显示 Gemma‑4 31B 在 RAE 上取得最高准确率，但在 LBS 上表现不佳；总体而言，模型在 Understand 与 Evaluate 上接近满分，而在 Apply、Create、Remember 与 LBS 评测中存在显著下滑，显示认知层级与跨语言差异。

**⚠️ 局限性**

局限包括：仅评估多选题而非开放式问答，受限于 GPU 资源与 API 访问未能覆盖所有最新 VLM，数据仅在部分子集人工验证，且 LBS 评估受 tokenization 影响，未完全消除语言偏差。

---

## 204. MOSAIC: A Workload-Driven Simulation and Design-Space Exploration Framework for Heterogeneous NPUs

**arXiv ID:** 2606.05362 | [PDF](https://arxiv.org/pdf/2606.05362v1)

**作者:** Arghadip Das `[一作]` (Purdue University), Vijay Raghunathan `[通讯]` (Purdue University)

**通讯引用:** 7081 | [OpenAlex ID](https://openalex.org/A5102009759)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 MOSAIC，一个可对异构 NPU 微架构进行分析仿真与设计空间探索的框架，能够自动生成包含 Big、Little 与特殊功能 Tile（FFT、SNN、多项式）的最佳混合设计。

**💡 创新点**

创新点包括：①在 12 个设计维度（Tile 类型、尺寸、精度、稀疏性、数据流、存储、特殊功能单元等）上进行联合搜索；②为非 MAC 运算单元（FFT、SNN、多项式）提供专门的能耗、面积与时序模型；③采用多种种子、分层随机抽样与遗传算法相结合的 DSE 流程，返回 Pareto 最优异构 NPU。

**🔧 技术方法**

技术手段包括：基于 FlexNPU 的多 Tile 模拟器、异构编译器映射器、ASAP7 7nm 合成校准、DRAM 能耗表、系统级 RTL 门控验证，以及遗传算法的设计空间搜索。

**📊 数据集**

数据集：20 个工作负载，涵盖 14 个基准模型（ResNet‑50、ViT‑B/16、LLaMA‑7B、Mixtral、Nemotron‑H、Mamba‑370M、Hyena‑1.3B、KAN、SNN‑VGG9、LAVISH、LLaVA、RT‑2、GNN‑GAT）及其 INT4/INT8 量化变体。

**📈 对比分析**

比较方法：在同面积、同精度或同能耗条件下与同构 NPU 基准、NVDLA 及 NVDLA‑large 进行对比。结果显示 GA‑优化的异构 HPU 在 20 个工作负载上平均能耗比最优同构降低约 46.9%，峰值能耗节省 60.1%，并在 INT8/SSM/ViT 工作负载上比 NVDLA‑large 提升 1.5–2.4×。

**⚠️ 局限性**

局限性：未对特殊功能单元进行非对称精度的搜索；未实现层内异构与全局映射器优化；面积模型未考虑布局死区和 P&R 复杂度；仿真以算子图层级进行，缺乏核层级细节。

---

## 205. Trust, but Don't Verify: Epistemic Blind Spots in LLM Source Evaluation

**arXiv ID:** 2606.05403 | [PDF](https://arxiv.org/pdf/2606.05403v1)

**作者:** Rohan N. Pradhan `[一作]` (Amazon), Steve Goley `[通讯]` (Amazon)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5044056321)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对多种大语言模型在多源信息合成情境下的行为进行实验，发现模型虽然能在单独检测时识别出伪造统计，但在多源合成中仍会将伪造统计视为与有效统计同等可信，导致误导性推断。

**💡 创新点**

创新点在于系统性揭示“方法论注册门”机制，即模型对方法论文本的分布式特征高度敏感，却忽略数值有效性；并通过因果追踪、线性探测和组件级归因三重机制解析该缺陷。

**🔧 技术方法**

主要技术包括：因果追踪（restore curves）、线性探测器（probe AUC）、组件级归因（direct logit attribution）、多模型多领域实验设计与统计评估。

**📊 数据集**

数据集为人工构造的三领域（风险投资、市场营销、公共卫生）对话线程，包含六种方法论/统计呈现水平，覆盖约一百万个实验样本。

**📈 对比分析**

比较方法：在不同模型家族（Claude、Qwen、OLMo）与不同提示策略（通用、统计、oracle）下测量源偏好指数（SPI）。结果显示：①在单独检测时所有模型均能识别伪造统计（CIR≥0.76）；②在多源合成中，伪造统计的影响与有效统计相近（约79%），方法论的影响随共识强度显著降低；③任何提示策略均未实现选择性辨别，均产生统一的怀疑或信任。

**⚠️ 局限性**

局限性包括：实验结构受限于固定四源对话框架，未覆盖更自然的多源情境；因果与归因分析主要针对Qwen 32B，无法排除其它隐蔽通路；缺乏对训练阶段如何改进的系统性验证，仍需进一步探索数据与监督策略。

---

## 206. The Role of Instructional Guidance in Generative AI-Assisted Learning: Empirical Evidence from Construction Engineering Education

**arXiv ID:** 2606.05509 | [PDF](https://arxiv.org/pdf/2606.05509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 207. Pattern Selectivity is Not Task-Causal Structure: A Cross-Architecture Mechanistic Study of Composed-Task Circuits in 1B-Class Language Models

**arXiv ID:** 2606.05378 | [PDF](https://arxiv.org/pdf/2606.05378v1)

**作者:** Yongzhong Xu `[一作]` `[通讯]`, Yongzhong Xu

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在1B级别语言模型中使用统一的屏蔽与消融（screen‑and‑ablate）方法，探究不同模型架构与训练管道在四种复合任务（间接宾语识别、大小比较、后继序列、变量绑定）中的机制差异。

**💡 创新点**

提出了五类屏蔽结果分类（主要因子、次要因子、相关器、干扰器、无效），展示了即使相同功能在不同模型上会出现完全不同的主导机制；并提出可检验的MoE模型“prev‑token 先行”假设。

**🔧 技术方法**

采用参与度比率（spectral PR）筛选头部、任务模式屏蔽、匹配随机对照消融以及单头消融等技术，结合量化阈值与匹配随机差异。

**📊 数据集**

使用四个合成任务的数据集，分别为间接宾语识别、大小比较、后继序列与变量绑定，并在三种1B级模型（Pythia 1B、OLMo 1B、OLMoE 1B‑7B）上进行评测。

**📈 对比分析**

通过将屏蔽效果与匹配随机对照进行对比，计算特异性比率，发现所有12个（任务×模型）组合均出现至少一种屏蔽结果类别；在不同模型间没有出现相同的主导屏蔽，体现了跨架构机制差异。

**⚠️ 局限性**

局限包括：仅评估四个任务与三种模型；单一预训练检查点；单头消融未覆盖所有细节；L0层屏蔽对照方差大；仅使用top‑1与logit差距作为指标；干扰器案例仅在Pythia变量绑定中明确出现，需更多验证。

---

## 208. CHASE: Adversarial Red-Blue Teaming for Improving LLM Safety using Reinforcement Learning

**arXiv ID:** 2606.05523 | [PDF](https://arxiv.org/pdf/2606.05523v1)

**作者:** Rahul Markasserithodi `[一作]` (University of New South Wales), Alan Niu `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个闭环红蓝团队框架CHASE，用以通过无模板的强化学习生成并对抗LLM安全逃逸。

**💡 创新点**

创新点在于：①模板自由的攻击生成与防御训练，②采用乘法奖励拆分消除意图漂移，③两阶段GRPO+拒绝采样SFT硬化流程，③实现跨攻击族的泛化。

**🔧 技术方法**

核心技术包括Group Relative Policy Optimization (GRPO)、乘法奖励（S_bypass×I_intent）、拒绝采样的监督微调 (SFT)、LoRA适配器，以及StrongREJECT评价框架。

**📊 数据集**

使用的主要数据集有Llama-3.1-8B-Instruct基础模型、BeaverTails、JailbreakBench、Stanford Alpaca（善意提示）、MT-Bench（衡量帮助度）。

**📈 对比分析**

在BeaverTails和JailbreakBench上与五种SOTA黑盒攻击（PAIR、TAP、AutoDAN、PAP、翻译）对比，CHASE平均将StrongREJECT分数降低43.2%，在JailbreakBench标准评测中实现0% ASR，且对100条善意提示的误拒为0%。

**⚠️ 局限性**

局限性包括仅在单一基础模型和LoRA配置下验证，跨架构与规模的迁移性未知；在含有创意推理的benign提示上拒绝率上升；评估依赖LLM判定，缺少人工注释；攻击模型未公开，防御模型易被误用。

---

## 209. Wave Focusing in Metamaterials: Tactile Displays Beyond the Diffraction Limit

**arXiv ID:** 2606.05572 | [PDF](https://arxiv.org/pdf/2606.05572v1)

**作者:** Gregory Reardon `[一作]` (University of California, Santa Barbara), Yon Visell `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 2974 | [OpenAlex ID](https://openalex.org/A5008718887)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用局部共振的韧性超材料，将柔性板与布置的杆形共振器耦合，从而在仅八个外围电磁驱动器的控制下，在表面上实现多点、可独立可编程的“虚拟触觉像素”。

**💡 创新点**

创新点在于：① 通过引入共振器实现板材折射关系的工程化，形成慢波分支，显著降低波长并突破衍射极限；② 结合逆向滤波（时空逆滤波）实现多点精准聚焦；③ 仅用稀疏驱动即可获得高分辨率触感，而非传统每点驱动。

**🔧 技术方法**

使用的技术包括：有限元布里渊边界条件求解（COMSOL），数值优化设计慢波分支；实验测量使用扫描激光多普勒激振仪获取 Green’s 函数；时域逆滤波与 Tikhonov 正则化求解驱动信号；POD 与 SVD 分析空间响应维度与输入-输出权限；行为实验验证感知准确率。

**📊 数据集**

主要数据集为实验测得的 8 号驱动器对 1225 位置的 Green’s 函数（时频域），以及 75–400 Hz 频段的振动速度场；同时使用数值仿真得到的相位速度、波长和频率分支。没有使用公开的大规模外部数据集。

**📈 对比分析**

与传统均质板进行对比：在相同尺寸（25 × 15 mm）和相同驱动器数的情况下，超材料板的波长缩短约 2–5 倍，虚拟像素面积平均仅 2.74 mm²，约为均质板的 1/10；刷新率可达 200 Hz；行为实验显示 95–99% 的定位与辨别准确率，显著优于均质板。

**⚠️ 局限性**

局限性包括：需要对每次实验重新校准（驱动器-板耦合可能随接触状态、温度等变化）；仅八个驱动器限制了同时独立像素数量；共振器耦合引入频率相关损耗，导致在高频或远场的衰减；材料非理想性（如粘性损耗）会降低波能量；实际应用需考虑长期稳定性与制造可扩展性。

---

## 210. SET: Stream-Event-Triggered Scheduling for Efficient CUDA Graph Pipelines

**arXiv ID:** 2606.05495 | [PDF](https://arxiv.org/pdf/2606.05495v1)

**作者:** Zhengxiong Li `[一作]` (University of Wisconsin-Madison), Umit Ogras `[通讯]` (University of Wisconsin-Madison)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于CUDA图的流事件触发调度框架SET，用于在GPU上高效执行任务并行管道，显著降低主机与设备之间的同步延迟与核间空隙。

**💡 创新点**

创新点在于：① 将每个任务抽象为可重用的CUDA图可执行体，并通过事件链式回调实现异步资源释放；② 采用多流任务并行编程模型，配合工作窃取和线程安全的工作者队列，实现 O(1) 的同步开销和实时任务分配；③ 在每个工作者上预分配并隔离内存缓冲区，保证多任务并发时的内存安全。

**🔧 技术方法**

使用CUDA Graph、事件回调（callback）、工作窃取（work stealing）、多流（multi‑stream）以及主机端的线程安全队列与原子操作来实现事件驱动的调度。

**📊 数据集**

在六个代表性工作负载上评估：Sobel、GEMM、BP、KNN、Hotspot、SSSP，涵盖计算密集型、内存密集型和小核时间的场景。

**📈 对比分析**

与同步、Graph、静态批处理、队列四种基线模型相比，SET在RTX 3090/RTX 5090 上平均提升 2.15×（同步）/2.20×（Graph）/2.12×（批处理）/2.08×（队列） 的吞吐量，并将调度开销降低 18–54%。

**⚠️ 局限性**

局限性包括：对极大批量大小仍会出现交叉批处理开销；在内存受限或高数据争用场景下，工作窃取与全局互斥可能产生额外开销；对需要频繁更新参数的小核工作负载，仍需进一步优化参数同步效率。

---

## 211. FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization

**arXiv ID:** 2606.05468 | [PDF](https://arxiv.org/pdf/2606.05468v1)

**作者:** Yihao Wu `[一作]` (Tencent Robotics X), Zhengyou Zhang `[通讯]` (Tencent Robotics X)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在真实机器人上对流匹配视觉语言动作模型进行无奖励的离线强化微调，构建了FlowPRO框架

**💡 创新点**

引入RPRO（Robotic Flow‑matching Proximalized Preference Optimization），通过显式近端正则化消除Reward‑Hacking，并设计了“干预‑回滚”采样与平滑插值生成稠密状态级偏好对；同时采用批次混合策略提升样本效率

**🔧 技术方法**

RPRO（PRO的流匹配扩展）、流匹配模型、贝塞尔曲线平滑插值、批次混合训练、离线强化学习算法

**📊 数据集**

基于Dobot XTrainer双臂平台的四个长时延任务（Pack、Cap、USB、Case）的真实机器人数据；数据来自SFT预训练集与干预‑回滚生成的偏好对

**📈 对比分析**

与四种基线（DAgger、DAgger‑Buffered、PI0.6*、TPO）及不同Loss组件（SFT、DPO、PRO、RPRO）对比；在所有任务上FlowPRO的成功率均高于基线，平均提升8–15个百分点，且完成时间更短

**⚠️ 局限性**

仅在单一双臂平台验证，回滚决策仍需人工干预；缺乏对移动或更复杂手部操作的泛化验证

---

## 212. Bounded Deep Unfolding for Joint Beamforming and Scheduling in Multi-Cell MIMO Networks

**arXiv ID:** 2606.05246 | [PDF](https://arxiv.org/pdf/2606.05246v1)

**作者:** Jiansheng Li `[一作]` (Chinese University of Hong Kong), Junting Chen `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1804 | [OpenAlex ID](https://openalex.org/A5101802022)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种结合深度展开的联合学习框架（P-Net与K-Net），用于多小区MIMO网络中的资源块组（RBG）调度与连续波束成形的混合整数优化问题；

**💡 创新点**

创新点包括：1）在FastFP迭代中学习一个受限的自适应松弛因子，以加速波束更新且保证单调提升；2）设计了基于优先级策略的K-Net，利用物理信息的匹配权重引导贪婪分配，显著降低Hungarian匹配的复杂度；3）两模块共享参数、可超出训练步数灵活推理；

**🔧 技术方法**

技术手段主要是：深度展开（Deep Unfolding）与可微分的学习模块、Recurrent Neural Network（GRU）实现自适应松弛因子、基于多尺度池化的图神经网络实现优先级学习、REINFORCE策略梯度训练、截断反向传播、低复杂度安全检查与重启机制；

**📊 数据集**

实验数据集基于仿真产生的多小区MIMO信道，采用Hexagonal网格、随机用户分布、路径损耗模型及阴影衰落，覆盖不同BS数量、用户数、RBG数、天线数、功率与阴影方差的多种场景；

**📈 对比分析**

与传统FastFP、Nesterov‑FastFP、DeepFP（有监督/无监督）、Hungarian匹配等基准进行比较；结果显示P‑Net+K‑Net在相同迭代次数或计算时间下实现了更高的加权总速率（WSR），并在多种规模与信道条件下保持稳健，尤其在大规模、低SNR或高阴影环境下相对优势更明显；

**⚠️ 局限性**

局限性包括：1）仍依赖FP的近似和多项式复杂度，极端大规模系统中仍可能受限；2）离线训练需覆盖足够多的网络规模与信道分布，否则泛化可能受限；3）对极端动态场景（用户快速移动、链路变动）可能需要在线微调；4）在某些设置下K‑Net的贪婪分配可能产生非单调WRS曲线，需要更细粒度的安全策略。

---

## 213. CLaaS: Continual learning as a service for sample efficient online learning

**arXiv ID:** 2606.05559 | [PDF](https://arxiv.org/pdf/2606.05559v1)

**作者:** Kion Fallah `[一作]` (Resolute Labs), Qingqing Mao `[通讯]` (Incept Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CLaaS系统，实现了在部署期间通过聊天接口收集rollout并异步使用经验回放进行在线持续学习。

**💡 创新点**

创新点在于将持续学习抽象为聊天API后端，利用LoRA热重载、单轨迹策略梯度与经验回放结合，提高样本效率并减少灾难性遗忘。

**🔧 技术方法**

技术包括：基于LLM的策略梯度（REINFORCE++、PPO、SDPO）、LoRA自适应、异步经验回放、重要性采样与剪裁、奖励信号自监督/验证器。

**📊 数据集**

使用了IH‑Challenge综合类别的对抗式测试集（共100个场景，5个连续子集）和Qwen3‑8B模型进行实验。

**📈 对比分析**

与基线与ICL（上下文学习）对比，CLaaS在前向转移上提升约3倍，后向遗忘减半，最终通过SDPO+自蒸馏实现75.2%成功率，明显优于ICL的24.1%。

**⚠️ 局限性**

局限性包括：对Replay buffer年龄的敏感性（过老导致不稳定）、仅在单一对抗任务上验证、需要离线训练资源并对奖励信号质量高度依赖。

---

## 214. Bootstrapping Semantic Layer from Execution for Text-to-SQL

**arXiv ID:** 2606.05634 | [PDF](https://arxiv.org/pdf/2606.05634v1)

**作者:** Youngwon Lee `[一作]` (Seoul National University), Seung-won Hwang `[通讯]` (Seoul National University)

**通讯引用:** 1645 | [OpenAlex ID](https://openalex.org/A5101567750)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个可迭代的文本到 SQL 的系统 GATE，通过执行反馈动态打开并确认 SQL 语句中未确定的语义映射，并将确认的映射存入可复用的记忆库。

**💡 创新点**

创新点在于把数据库执行视为“自举”机制而非仅作验证：在生成 SQL 的过程中保持语义假设开放，执行后根据观察结果选择正确的 grounding 并缓存，以后查询可复用。

**🔧 技术方法**

技术包括：部分地预先约定的 SQL 计划、生成多种 grounding 假设、使用 LLM 结合执行观察做判断、计划评估与选择、证据摘要、执行结果记忆更新和跨计划复用。

**📊 数据集**

使用了三类数据集：真实医院数据库 RealEHR、公开 MIMIC 基础的 EHRSQL，以及从 LiveSQLBench 删去 grounding 信息的 LS-Hard。

**📈 对比分析**

与 ReAct、ReFoRCE、ReDel 等基线相比，在 RealEHR、EHRSQL 和 LS-Hard 三个基准上均获得最高的准确率，平均提升约 4–10 个百分点，并且在 ReAct 预算相同下，GATE 仅用 36 次 LLM 调用即达 55.2% 的答案准确率。

**⚠️ 局限性**

局限性包括：假设缺失的 grounding 可以通过数据库内容、执行检查或外部信息恢复，无法处理隐式/未记录的约定；仅针对单轮查询，无法处理用户意图不明确需澄清的情形。

---

## 215. What's in a Name? Morphological Shortcuts by LLMs in Pharmacology

**arXiv ID:** 2606.05616 | [PDF](https://arxiv.org/pdf/2606.05616v1)

**作者:** Kaijie Mo `[一作]` (University of Texas at Austin), Junyi Jessy Li `[通讯]` (University of Texas at Austin)

**通讯引用:** 1269 | [OpenAlex ID](https://openalex.org/A5021057186)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了大型语言模型对药物名称形态学暗示的依赖，评估并量化了模型在药物分类推断中的形态学捷径。

**💡 创新点**

提出了诊断框架以分离形态学、词干和整体词汇对模型输出的贡献，并通过激活补丁定位早期层中的形态学捷径。

**🔧 技术方法**

使用多任务问答、激活补丁、分层因果分析和分布式对齐搜索等技术。

**📊 数据集**

构建了包含655种医学前缀/后缀的药物名称三元组数据集，包含真实药物、假药物和完全虚构词。

**📈 对比分析**

通过多模型比较（9大模型）在多选和开放式问答上的准确率，发现大型模型在多选中对形态学的依赖随规模增大，开放式问答中医学专用模型表现最差。

**⚠️ 局限性**

仅适用于公开可访问概率的模型，且仅在7B规模模型上做了机制分析，未验证更大模型的通用性。

---

## 216. Safety Paradox: How Enhanced Safety Awareness Leaves LLMs Vulnerable to Posterior Attack

**arXiv ID:** 2606.05614 | [PDF](https://arxiv.org/pdf/2606.05614v1)

**作者:** Long P. Hoang `[一作]` (Singapore University of Technology and Design), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 5394 | [OpenAlex ID](https://openalex.org/A5042288832)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种单查询的后验攻击（Posterior Attack），通过让LLM生成其内部安全判别器会标记为不安全的回复，从而绕过安全对齐；并系统性研究并证实“安全悖论”（Safety Paradox）——安全意识越高的模型越易被此攻击利用。

**💡 创新点**

创新点在于：①首次将内部安全判别器的后验信息作为攻击载体，突破传统先验概率攻击；②用贝叶斯推理正式化安全悖论，证明安全意识提升会放大后验攻击的成功率；③通过强化学习（GRPO）操纵安全意识，建立因果关系，验证安全意识与攻击成功率的正向耦合。

**🔧 技术方法**

使用技术包括：贝叶斯推理与后验概率计算；Group Relative Policy Optimization (GRPO) 的安全意识提升/降级训练；单查询安全提示模板；使用链式思考与推理预算的测试时扩展；对比白盒（GCG、AutoDAN）与黑盒（DeepInception、ReNeLLM、CodeChameleon 等）越狱方法；以及基于LLM的 ASR 评估与 HarmBench 分析。

**📊 数据集**

数据集涵盖：AdvBench（520种有害行为）用于 ASR 评估；HarmBench（596条有害/安全问答）用于内部安全判别准确率；WildGuardTrain（4096条问答）用于安全意识强化/降级训练；GSM8K 与 MMLU 用于评估模型通用推理能力；GPT‑4o‑mini 作为评判器。

**📈 对比分析**

方法比较：在 30 个开源模型与 10 个前沿模型上，Posterior Attack 的平均 ASR 为 83%，在大多数模型上接近 100%，明显高于现有白盒/黑盒基线（如 GCG 0.2%、DeepInception 97.7% 但成本高）。攻击仅需一次查询，输入/输出 token 大约 3,300/4,000，成本约 $0.03，显示出极高的效率与低资源需求。

**⚠️ 局限性**

局限性：ASR 评估依赖 GPT‑4o‑mini 等 LLM 判定，存在误判风险；实验集中于英文模型，跨语言适用性未知；未充分覆盖系统级防护机制；Deliberative Alignment 作为防御需高计算成本，难以实时应用；研究仅针对 2026 年初的模型与数据集，后续模型更新可能影响结果。

---

## 217. GuardNet: Ensemble Strategies of Shallow Neural Networks for Robust Prompt Injection and Jailbreak Detection

**arXiv ID:** 2606.05566 | [PDF](https://arxiv.org/pdf/2606.05566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 218. FQA: A Full-Space Quantization-Driven Architecture for Hardware-Efficient Piecewise Approximation of Nonlinear Activation Functions

**arXiv ID:** 2606.05627 | [PDF](https://arxiv.org/pdf/2606.05627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 219. Cross-Epoch Adaptive Rollout Optimization for RL Post-Training

**arXiv ID:** 2606.05606 | [PDF](https://arxiv.org/pdf/2606.05606v1)

**作者:** Yiming Zong `[一作]` (Hong Kong University of Science and Technology), Jiashuo Jiang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1400 | [OpenAlex ID](https://openalex.org/A5101856185)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出跨周期自适应滚动分配框架 CERO，优化大模型 RL 后训练中的滚动预算。

**💡 创新点**

创新点在于利用 Beta 后验估计提示信息量，构造凹性饱和效用函数，并通过 Fenchel 双重实现跨 epoch 的全局预算在线分配。

**🔧 技术方法**

技术主要包括贝叶斯后验推断（Beta 分布）、凸优化/Fenchel 双重、投影梯度下降在线更新、GRPO 后训练框架以及基于奖励方差的期望信息量计算。

**📊 数据集**

实验数据集包括 DAPO‑Math‑17K 训练集，以及 AIME24/25/26 和 AMC23 四个竞赛式数学推理基准。

**📈 对比分析**

与原始 GRPO 以及 KnapsackRL 进行对比，CERO 在所有模型（DeepSeek‑R1‑Distill‑1.5B、Qwen3‑4B‑Base、Qwen3‑4B‑Instruct、Qwen2.5‑Math‑7B）和基准上均提升 3–6 分，且有效提示比例明显更高，表明样本效率更优。

**⚠️ 局限性**

局限性：理论收敛仅适用于固定效用函数，未考虑后验更新导致的非平稳性；实验范围仅涵盖数学推理任务，未验证其他领域或更大规模的通用性。

---

## 220. From Prediction to Self: Developmental Conditions for Agency in Minimal Neural Systems

**arXiv ID:** 2606.05605 | [PDF](https://arxiv.org/pdf/2606.05605v1)

**作者:** Evan Ye `[一作]` `[通讯]` (Independent Researcher), Evan Ye (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在一个192维GRU模型上，通过40个受控实验的发育序列，系统从无行动、无自我表征逐步引入持续状态、因果行动循环、本体反馈和异步觉醒，最终实现自我-世界分解并量化代理优势；

**💡 创新点**

首次系统性地阐明自我-世界分解的四个必要条件的严格顺序，并提出可测量的“代理收益”指标，用以区分仅做预测的系统和真正意识到自身行动的系统；

**🔧 技术方法**

使用多尺度EMA增强的GRU网络、双头预测器、前向采样行动选择以及相位分离的训练策略；

**📊 数据集**

使用4通道相位正弦信号与Lorenz混沌吸引子作为环境观测，且所有模型参数少于10万；

**📈 对比分析**

通过与无因果对照组、同步训练对照组、无本体反馈对照组等对照实验进行比较，代理收益显著为正（最高可达99.5%），尖峰比率在异步训练下显著提升（5.58×），追踪回忆率从12.3%提升至56.5%；

**⚠️ 局限性**

受限于仅处理即时行动、缺乏时间延迟的因果推理、对信号幅度和行动范围敏感、模型规模过小、仅区分“自我”与“其他”但无法识别其他代理，且未对更复杂环境或不同架构进行验证。

---

## 221. Sound Effects Dataset Unification With the Universal Category System

**arXiv ID:** 2606.05571 | [PDF](https://arxiv.org/pdf/2606.05571v1)

**作者:** Jun Woo Beck `[一作]` (Georgia Institute of Technology), Alexander Lerch `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1672 | [OpenAlex ID](https://openalex.org/A5048454242)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于行业标准 Universal Category System (UCS) 的模块化数据集重标注框架，包含规则化的多阶段标签映射、冲突解析、UCS 兼容的拆分与合并工具，并用其构建了 58,057 条环境声音的统一数据集 EnvSound-UCS。

**💡 创新点**

创新点在于：① 通过预定义映射、子类别匹配、类别匹配、同义词逆向查找四阶段规则实现 98%+ 的自动标签转换；② 引入冲突解析层级优先与多数投票机制，显著提升标签一致性；③ 提供了基于 UCS 的层次化拆分工具，支持多源数据的无缝合并；④ 首次公开构建并基准测试了统一 UCS 数据集，验证了层次化分类与跨源融合的效果。

**🔧 技术方法**

技术包括：Python 实现的文本规则匹配管线、UCS 语义树查询、冲突解析算法；数据拆分采用组合层级 (类别+子类别) 的分层抽样；模型使用 PANNs CNN14 预训练特征提取器、单层线性分类器、AdamW、focal loss 以及三阶冲突规则；实验脚本自动化 5 次随机种子评估。

**📊 数据集**

主要使用 FSD50K、AudioSet（平衡训练/评估子集）、ESC-50 这三个公开音频数据集，随后合并生成 EnvSound-UCS；还引用了 UrbanSound8K、Clotho 等数据集作为对比背景。

**📈 对比分析**

对比方法包括平面分类 (Cat)、平面子类别分类 (SubCat_flat)、层次分类 (SubCat_hier) 与其 oracle 版本。宏观 F1 结果：FSD50K Cat 0.52→SubCat_flat 0.71；AudioSet Cat 0.42→0.56；ESC-50 Cat 0.89→0.99。层次分类表现逊于平面分类，oracle 版本提升 20–25%；在 EnvSound-UCS 上，跨源训练模型在单源测试集上略低于自训模型，但整体保持在 0.5–0.9 之间。

**⚠️ 局限性**

局限性包括：① 仅基于文本匹配，未做音频验证；② 冲突解析规则在 10–20% 文件中触发，需要人工复核；③ 需要手工维护预定义映射表，扩展至新数据集不自动；④ UCS 版本更新可能导致映射失效；⑤ 由于多源异构性与噪声标签，单纯合并并未提升性能；⑥ 评估仅用单层线性分类器，未探索更深层次模型或领域自适应。

---

## 222. When New Generators Arrive: Lifelong Machine-Generated Text Attribution via Ridge Feature Transfer

**arXiv ID:** 2606.05626 | [PDF](https://arxiv.org/pdf/2606.05626v1)

**作者:** Zhen Sun `[一作]` (Wuhan University), Xinlei He `[通讯]` (Wuhan University)

**通讯引用:** 13021 | [OpenAlex ID](https://openalex.org/A5031973958)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级、无样本回放的终身机器生成文本(MGT)归因框架，能在不更新深度编码器的前提下，持续学习新生成器并保持对旧生成器的识别能力。

**💡 创新点**

核心创新是将增量学习转化为基于类级足够统计量的闭式岭回归更新，并引入分数白化、等距随机特征映射和类别平衡加权，三者协同提升新类适应性同时抑制灾难性遗忘。

**🔧 技术方法**

技术包括：任务调优的冻结编码器、分数（fractional）协方差校准、等距高斯随机特征映射、类平衡闭式岭回归、低精度与统计合并压缩。

**📊 数据集**

实验使用两个公开数据集：MGT‑Academic（涵盖 STEM、Humanities、Social Science 三个学科，约 73K 条样本）和 AIGTBench（社交媒体子集，15K 条均衡样本），涉及 GPT‑3.5、GPT‑4o‑mini、Moonshot、Mixtral‑8x7B、Llama‑3.1 等多种 LLM。

**📈 对比分析**

与 LwF、iCaRL、BiC、EASE、PASS、SimpleCIL 等基线（其中部分带回放）以及 RoBERTa/DeBERTa‑base 两个骨干进行比较。该方法在 P3/P4/P5 终身归因协议下，在宏 F1、旧类 F1、新类 F1 三个指标上均优于所有基线，尤其在新类 F1 上提升 0.10+，在低资源（5% 新类样本）下仍保持 0.9+ 的宏 F1。

**⚠️ 局限性**

主要局限：仍需存储足够统计量和随机投影矩阵，存储量虽可压缩但非零；冻结编码器在面对长期演化的新生成器时可能缺乏足够的区分信息；实验仅覆盖英文，跨语言和跨域泛化需进一步验证。

---

## 223. KV-Control: Parameter-Efficient K/V Injection for Trajectory-Controlled Text-to-Motion

**arXiv ID:** 2606.05624 | [PDF](https://arxiv.org/pdf/2606.05624v1)

**作者:** Tengjiao Sun `[一作]` (University of Southampton), Hansung Kim `[通讯]` (Mogo AI Ltd.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在预训练的文本到运动变压器上引入 KV-Control 适配器，实现了对根轨迹和多关节轨迹的精确控制，同时保持原有文本驱动的运动风格。

**💡 创新点**

创新点在于：①将几何约束以键/值记忆形式注入自注意力层，避免了对查询流的干扰；②联合设计 PartVQ（基于解剖学的分块量化器）和 T‑Concat（按时间展开的分块序列），使每个帧‑部件 token 成为可直接寻址的注意力位置；③实现了极小化的可训练参数（仅约 1.5M K/V 参数 + 轨迹编码器），显著低于传统复制分支方法。

**🔧 技术方法**

技术包括：自监督的 PartVQ 量化器、T‑Concat 结构化编码、Transformer 变压器后置归一化、文本跨注意力门控、低秩 K/V 插值、以及基于 MaskControl 的后处理迭代优化。

**📊 数据集**

使用 HumanML3D 数据集进行训练与评估，遵循 MaskControl 评估协议（CFG 3.25、10 步采样等）。

**📈 对比分析**

与 MaskControl、OmniControl、TLControl 等基线相比，KV‑Control 在 M3 设定下实现根轨迹平均误差 < 3 cm、全关节误差 < 3 cm，参数量仅为同类复制分支的 1/26，计算成本保持在可接受范围内，并在多样性与匹配度上与基线相近。

**⚠️ 局限性**

局限性包括：①仅针对单一骨架结构；②在极密集的每帧接触或多关节约束下，Stage‑2 迭代可能导致过拟合；③尚未验证在其他离散或连续运动背骨（如自回归、扩散或混合模型）上的迁移效果。

---

## 224. ANCHOR: Agentic Noise Creation Framework for Human Simulation and Denoising Recommendation

**arXiv ID:** 2606.05621 | [PDF](https://arxiv.org/pdf/2606.05621v1)

**作者:** Xiangming Li `[一作]` (Xidian University), Yangtao Zhou `[通讯]` (Xidian University)

**通讯引用:** 3603 | [OpenAlex ID](https://openalex.org/A5015974394)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于生成‑识别范式的推荐系统噪声去除方法 ANCHOR，通过模拟用户行为生成带标签的噪声数据，再训练噪声识别器，实现对隐式反馈噪声的有监督识别。

**💡 创新点**

创新点在于将噪声去除从无监督滤波转为有监督学习，利用 LLM 代理主动生成多种真实噪声类型，并通过对抗式边界细化构造难分噪声样本，从而提升识别精度。

**🔧 技术方法**

使用技术包括大型语言模型代理用户行为模拟、基于 LightGCN 的协同嵌入、语义嵌入与多层感知机结合的噪声识别器、以及迭代生成‑识别的对抗细化过程。

**📊 数据集**

实验数据集为 DBbook2014、Book‑Crossing 与 MovieLens‑1M，这些数据集均包含用户‑物品交互和物品文本描述。

**📈 对比分析**

与 T‑CE、DeCA、BOD、DCF、LLaRD 等主流去噪方法以及 GMF/LightGCN 基座模型比较，ANCHOR 在 Recall 和 NDCG 上持续领先，提升幅度可达 10%+，并在不同噪声比例下保持更稳健的性能。

**⚠️ 局限性**

局限性包括对 LLM 代理的计算依赖、生成噪声标签的真实性可能受模拟策略限制，以及在缺乏丰富文本信息的领域中模型效果可能受限。

---

## 225. Predictable Scaling Laws of Optimal Hyperparameters for LLM Continued Pre-training

**arXiv ID:** 2606.05610 | [PDF](https://arxiv.org/pdf/2606.05610v1)

**作者:** Yongwei Zhou `[一作]` (MeiTuan), Rongxiang Weng `[通讯]` (MeiTuan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于计算预算与最优超参数之间可预测的缩放规律的框架，用于在持续预训练（CPT）过程中直接零样本预测学习率和批量大小；

**💡 创新点**

创新点在于首次发现CPT阶段最优超参数也遵循稳定的缩放规律，并引入“等价预训练计算”（Equivalent Compute）概念，将任意检查点映射到连续训练轨迹上，实现对起始状态的量化；

**🔧 技术方法**

技术主要包括：小规模代理模型的网格搜索收集数据、损失-计算缩放法则、等价计算逆推、以及基于功率律的最优超参数映射函数；

**📊 数据集**

使用了涵盖通用知识、数学和代码三大领域的55B Token持续预训练语料，构建了混合分布数据集；

**📈 对比分析**

通过与传统网格搜索以及忽略等价计算等变体进行对比，实验表明在Dense-8B和MoE-3B模型上，该框架将搜索成本降低70–92%，同时保持或提升最终模型在多项下游基准（MMLU、GSM8K、HumanEval等）上的平均分；

**⚠️ 局限性**

局限性包括：在更大规模（70B+）模型上的验证不足；对极为专业化域的缩放参数可能需要局部校准；以及在极端分布漂移情况下的等价计算假设尚未充分验证。

---

## 226. Energy Efficiency Optimization for Rotatable Antenna-Enabled Uplink NOMA Systems

**arXiv ID:** 2606.05600 | [PDF](https://arxiv.org/pdf/2606.05600v1)

**作者:** Yixuan Li `[一作]` (Central China Normal University), Ji Wang `[通讯]` (Central China Normal University)

**通讯引用:** 12600 | [OpenAlex ID](https://openalex.org/A5100386450)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了基于可旋转天线（RA）的上行非正交多址接收系统，并通过联合优化接收波束、用户功率分配与天线旋转角度来最大化能效。

**💡 创新点**

创新点在于首次将可旋转天线与NOMA结合，利用天线方向可调提供额外的空间自由度，并提出一种三阶段块坐标下降（BCD）算法，融合MMSE波束、FP‑SCA功率分配和凸化旋转优化，显著提升能效。

**🔧 技术方法**

使用了MMSE波束形成、分数规划（FP）、序列凸近似（SCA）以及CVX求解器进行凸化求解。

**📊 数据集**

采用仿真数据，随机生成2个地面用户、2个空中用户，均匀分布在半径50–60 m、角度±45°的区域内，空中用户高度20 m，评估不同RA角度、指向度、阵列尺寸等参数。

**📈 对比分析**

通过与RA‑SDMA、RA‑TDMA、IA‑NOMA和FA‑NOMA四种基线方案对比，实验表明RA‑NOMA在能效上明显领先，尤其在大天线直指度或大天线阵列时优势更为突出。

**⚠️ 局限性**

局限性包括：仅考虑LoS环境、用户分布与功率上限简化、RA方向模型理想化、算法收敛速度与计算复杂度较高，以及未验证在多径或多用户密集场景下的鲁棒性。

---

## 227. Dimensionality Reduction for Cyberattack Classification: A Comparative Evaluation of PCA and Linear Predictive Coding

**arXiv ID:** 2606.05584 | [PDF](https://arxiv.org/pdf/2606.05584v1)

**作者:** Nelly Elsayed `[一作]` (University of Cincinnati), Navid Asadizanjani `[通讯]` (University of Florida)

**通讯引用:** 2023 | [OpenAlex ID](https://openalex.org/A5085074016)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

比较主成分分析（PCA）与线性预测编码（LPC）在网络流特征压缩后对网络攻击分类的影响。

**💡 创新点**

首次将LPC应用于表格型网络安全特征压缩，并与传统PCA在同一任务和模型上进行系统对比。

**🔧 技术方法**

使用 PCA、LPC 进行特征压缩；通过逻辑回归、线性 SVM、随机森林、梯度提升、MLP 等多类分类器进行评估；计算准确率、加权 F1 以及宏 F1。

**📊 数据集**

CICIDS2017 数据集（2,574,264 条样本，78 维特征，15 类）。

**📈 对比分析**

在 4、8、12 维压缩下，随机森林在 PCA-8、PCA-4 维时仍保持 99.7%+ 准确率；PCA 的加权 F1 在所有维度上均略优于 LPC，LPC 在低维下略显退化。

**⚠️ 局限性**

LPC 在极低维度下仍有一定性能损失，且未探索更高阶或非线性压缩方法；实验仅基于单一数据集，缺乏跨域或不同攻击场景的验证。

---

## 228. ZERO-APT: A Closed-Loop Adversarial Framework for LLM-Driven Automated Penetration Testing under Intelligent Defense

**arXiv ID:** 2606.05567 | [PDF](https://arxiv.org/pdf/2606.05567v1)

**作者:** Anlan Zheng `[一作]` (Zhejiang University of Technology), Tiantian Zhu `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 2029 | [OpenAlex ID](https://openalex.org/A5012235835)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ZERO-APT闭环对抗框架，集成LLM驱动攻击、可配置LLM防御与Judge裁决，实现攻击与防御的实时交互与可审计；

**💡 创新点**

三大创新：① 引入可配置的LLM Defender，以Sysmon+ELK实时检测并提供多级威胁情报；② 通过规划-执行分离、多维ReAct反馈与硬约束动作库，将因果一致性从LLM推到系统层面；③ Judge进行每轮裁决并生成结构化CTI报告，提供完整可追溯的决策审计；

**🔧 技术方法**

采用Claude Opus 4.7 LLM、Sysmon+ELK日志管道、三层架构（Planner、Dispatcher、Executor）、预定义动作库（Atomic Red Team）、多维ReAct反馈机制与结构化CTI报告；

**📊 数据集**

自定义 Windows Server 2022 后渗透测试基准，包含 5 个情景、3 个 Defender 强度等级（L1-L3），共 1200 次实验；动作库来自 Atomic Red Team，场景设计受 Metasploitable3 影响；

**📈 对比分析**

与 Aurora、PentestGPT、Claude Code 进行对比：在 L3 Defender 下，ZERO-APT 79% ASR、0.860 CCS、10.3% CBF%，相较 Aurora 22%/0.930/5.1%、PentestGPT 39%/低 CCS/38.6%、Claude Code 63%/0.520/27.4%；Ablation 结果显示去除动作库或多层架构会显著下降 ASR、CCS 与 BSR；

**⚠️ 局限性**

局限性包括：仅覆盖 Windows Server 2022 后渗透场景；LLM 仍承担规划与裁决核心，仍可能产生错误；硬约束过滤在低防御环境可能过度裁剪；未实现主动防御或多机网络交互；需要更大规模跨平台验证

---

## 229. Using Large Language Models to Support High Volume Application Review for an Undergraduate Research Program

**arXiv ID:** 2606.05564 | [PDF](https://arxiv.org/pdf/2606.05564v1)

**作者:** Varun Aggarwal `[一作]` (Purdue University), John Howarter `[通讯]` (Purdue University)

**通讯引用:** 3318 | [OpenAlex ID](https://openalex.org/A5089108911)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并部署了基于大型语言模型的评估工具，对Purdue大学SURF项目约1200份Statement of Purpose进行分数与理由输出，以辅助选拔流程。

**💡 创新点**

创新点在于将结构化评估维度与行为化评分描述结合到提示中，强制模型给出引用依据，利用少量校准示例实现与人类评审高度一致。

**🔧 技术方法**

使用OpenAI GPT‑5.2（对比GPT‑4o、GPT‑5‑mini）以及自定义提示、少样本校准、JSON输出和人机协同审阅。

**📊 数据集**

使用的是SURF 2026周期约1200份经过资格筛选的Statement of Purpose文本数据集。

**📈 对比分析**

通过与人类评审的对比及模型间分数差异分析，GPT‑5.2平均每份文档耗时约14秒，总体计算时间4.6小时，成本约25美元；高分文档模型间差异≤2分，低分差异≥4分。

**⚠️ 局限性**

局限包括缺乏正式的可靠性度量、对低分文档的分数不稳定、可能存在模型偏见和隐私泄露风险，以及对最终选拔影响尚未量化。

---

## 230. Autoregressive Diffusion World Models for Off-Policy Evaluation of LLM Agents

**arXiv ID:** 2606.05558 | [PDF](https://arxiv.org/pdf/2606.05558v1)

**作者:** Kaixuan Liu `[一作]` (Emory University), Shengpu Tang `[通讯]` (Emory University)

**通讯引用:** 439 | [OpenAlex ID](https://openalex.org/A5000172328)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自回归扩散世界模型的离线评估框架 Adwm，能够仅用预收集轨迹评估大语言模型代理的价值；

**💡 创新点**

核心创新在于将策略引导的全轨迹概率精确拆解为单步条件，并将其映射为带有三项指导（先验、动作后验、策略连续性）的自回归扩散过程，使 LLM 代理能在不执行环境交互的情况下进行可靠模拟；

**🔧 技术方法**

使用扩散模型、分类器无指导（classifier‑free guidance）技术、latent 编码与投影、逆动力学与行为克隆辅助损失、蒙特卡洛价值估计；

**📊 数据集**

在四个多轮 LLM 代理基准上验证：HotpotQA、ScienceWorld、ALFWorld、WebShop，涵盖密集奖励、部分奖励、稀疏奖励和连续奖励场景；

**📈 对比分析**

与五种经典 OPE 方法（DM、IS、WIS、FQE、DR）对比，Adwm 在所有六个 (行为策略, 评估策略) 配置中均实现正向 Spearman 相关，平均相关系数 0.82，显著优于传统方法；

**⚠️ 局限性**

局限性包括依赖足够多样化的离线数据、扩散模型训练和推理的计算开销、对离线数据与真实策略分布相差过大的场景效果可能下降、以及对极端长序列的扩散步骤数可能需要调整。

---

## 231. ShotCrop$^3$: Cropping Human-Centric Images into Cinematic Triple-Shot Compositions

**arXiv ID:** 2606.05635 | [PDF](https://arxiv.org/pdf/2606.05635v1)

**作者:** Dehong Kong `[一作]` (Huawei Noah’s Ark Lab), Fan Li `[通讯]` (Huawei Noah’s Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出三镜头构图（Triple-Shot Compositions，TSC）任务，利用单张人像图生成建立镜、特写和中景三张带描述的裁剪图。

**💡 创新点**

创新点包括：①将多镜头叙事与美学裁剪结合，形成全新的任务；②构建TSC-Bench专家标注基准；③提出ShotCrop^3三阶段训练框架（CoT-SFT → Semi-SFT → GRPO-S），并设计多维度伪标签筛选策略。

**🔧 技术方法**

采用多模态大型语言模型（MLLM）进行链式推理和裁剪预测；链式思考监督微调；半监督伪标签扩展；强化学习（GRPO）结合IoU、审美、纵横比奖励；伪标签评估结合MLLM语义评分、CLIP相似度与审美评分。

**📊 数据集**

使用7,600幅专家标注图像（涵盖旅行、街拍、电影镜头、专业相册），划分为6,400训练、1,200测试；测试集构成TSC-Bench，包含1.2K三镜头裁剪案例。

**📈 对比分析**

与Gemini 2.5 Pro、GPT‑5、InternVL3.5、Qwen3‑VL等闭源/开源模型及专业裁剪模型进行对比。ShotCrop^3在IoU、BDE、Unipercent和综合故事评分上均优于所有基线，尤其在Shot Localization Accuracy 上相较GPT‑5提升约2.82倍；虽参数量仅4B，却击败更大规模模型。

**⚠️ 局限性**

局限性：仅处理静态人像图像，尚未扩展到视频；伪标签策略虽高效但仍需人工标注处理难例；基准数据域有限，可能对非人像或非叙事场景适用性不足。

---

## 232. Evaluation of LLMs for Mathematical Formalization in Lean

**arXiv ID:** 2606.05632 | [PDF](https://arxiv.org/pdf/2606.05632v1)

**作者:** Tyson Klingner `[一作]` (University of Washington), Vasily Ilin `[通讯]` (University of Washington)

**通讯引用:** 48 | [OpenAlex ID](https://openalex.org/A5009909631)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估了多种大语言模型在 Lean 4 上生成正式数学证明的能力，并分析了其成本效益和迭代改进效果。

**💡 创新点**

提出了同时使用 Pass@k 与 Refine@k 两种评估方法来衡量模型的正式化表现，并通过成本-准确度曲线量化模型在实际使用中的性价比。

**🔧 技术方法**

采用零样本 Prompt 生成策略、温度 0.5、token 限制 16384、Lean 4 编译器验证、生成错误过滤等技术实现了统一的评测管道。

**📊 数据集**

使用 miniF2F（竞赛级数学问题）与 miniCTX（包含大量前置定义和引理的实际数学结构）两个数据集的 50 条子样本进行评测。

**📈 对比分析**

在 Pass@32 与 Refine@32 的比较中，Gemini 3.1 Pro 与 Claude Opus 4.7 在两套数据集上均取得最高成功率；而在成本方面，Nemotron 3 Super 与 GPT-OSS 120B 以 <$0.01/正确证明 的成本实现了最高的成本效率；迭代改进策略在大模型上表现尤为明显，平均提升约 3% 的准确率。

**⚠️ 局限性**

实验受限于 k=32 的评估范围、样本子集可能存在偏差、统一 Prompt 可能低估某些模型表现，以及 Refine@k 的统计可靠性不足等限制。

---

## 233. AdaPlanBench: Evaluating Adaptive Planning in Large Language Model Agents under World and User Constraints

**arXiv ID:** 2606.05622 | [PDF](https://arxiv.org/pdf/2606.05622v1)

**作者:** Jiayu Liu `[一作]` (University of Illinois Urbana Champaign), Heng Ji `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出AdaPlanBench——一种基于MacGyver数据的动态交互式基准，用于评估大语言模型在逐步揭示的世界与用户双重约束下的自适应规划与重新规划能力。

**💡 创新点**

创新点包括：①把双重约束（世界约束与用户偏好）同时纳入规划任务；②设计了约束构造的自动化多代理流水线，并在运行时通过交互逐步披露约束；③引入了多维度评价指标（准确率、有效规划率、重复违规次数、约束触发率等）来系统评估模型的适应性。

**🔧 技术方法**

采用多种技术：多代理约束构造框架（重写器、过滤器、规划采样器、约束提取器、合并器、检查器）；LLM判定器用于实时约束检测和鲁棒性评估；Rubric-based LLM评审用于四维质量评分；交互式多轮协议与反馈机制。

**📊 数据集**

使用MacGyver 307个家庭场景任务，并通过AdaPlanBench管道生成对应的世界约束与用户约束，形成的任务集被公开（GitHub与HuggingFace）。

**📈 对比分析**

在10个主流LLM（包括GPT-5、Gemini、Qwen3、Llama3等）上进行实验，最高准确率仅为67.75%（GPT-5），大多数模型低于45%；虽然有效规划率普遍高于70%，但多轮交互中的重复违规率仍显著，表明当前模型在动态约束环境下仍有较大改进空间。

**⚠️ 局限性**

局限性：①域覆盖仅限家庭场景，难以直接推广到其他领域；②评估依赖LLM判定器，可能带来模型偏见；③采用文本交互，未考虑视觉感知与物理执行；④约束模型简化，未完全捕捉真实约束的模糊性与组合性。

---

## 234. The End of Software Engineering: How AI Agents Are Fundamentally Restructuring the Software Paradigm

**arXiv ID:** 2606.05608 | [PDF](https://arxiv.org/pdf/2606.05608v1)

**作者:** Zhenfeng Cao `[一作]` `[通讯]` (Lingxi Intelligent Investment Development), Zhenfeng Cao (Lingxi Intelligent Investment Development)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过第一性原理阐述AI代理是软件工程的根本范式转变，提出Agent-as-a-Service与Agentic Engineering新学科，并对传统软件与代理系统在复杂度和效率上的差异进行系统化比较。

**💡 创新点**

创新点在于：①把软件与代理的核心对象、控制模型、人类角色进行分离；②引入Agent-as-a-Service作为软件演进的第三阶段；③构建Agentic Engineering的概念框架与实践范式，强调代理生成代码为工具而非核心产物；④系统性地总结代理在持续演化与多代理协作方面的突破与局限。

**🔧 技术方法**

所采用的技术包括大型语言模型（LLM）、工具调用框架（ReAct、Chain-of-Thought）、多代理协同架构、内存管理（短期上下文+长期向量存储）、SWE-bench、EvoClaw、LangChain等基准与实验平台。

**📊 数据集**

使用的数据集与基准主要为：SWE-bench Verified（GitHub issue 自动化）、EvoClaw（连续演化评估）、LangChain多代理协作实验、SWE-GPT 72B等开放模型测试集。

**📈 对比分析**

比较方法主要是基准实验和量化指标：在SWE-bench Verified中，Lingma SWE-GPT 72B 取得30.20%（接近GPT‑4o 31.80%），而7B版本也达18.20%；在多代理协作实验中，协调团队将根因识别时间降低93%；EvoClaw中独立任务成功率约82%，连续演化仅38%，显著折扣。

**⚠️ 局限性**

主要局限包括：①上下文漂移导致长序列维持困难；②错误传播缺乏有效检测与回滚机制；③技术债务与可维护性未被充分建模；④验证机制不完整，易出现细微语义错误；⑤持续演化性能远低于孤立任务，需进一步改进记忆、压缩与验证技术。

---

## 235. Fix the Mind, Not the Move: Interpretable AI Assistance via Knowledge-Gap Localization

**arXiv ID:** 2606.05602 | [PDF](https://arxiv.org/pdf/2606.05602v1)

**作者:** Ayano Hiranaka `[一作]` (University of Southern California), Daniel Seita `[通讯]` (University of Southern California)

**通讯引用:** 1185 | [OpenAlex ID](https://openalex.org/A5041660944)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用专家轨迹和学生行为轨迹，推断学生的知识缺口并生成最小、可解释的知识纠正建议，以实现长周期任务中人机协作的行为对齐。

**💡 创新点**

1) 以结构化知识（PDDL）而非轨迹级干预进行误区定位；2) 两阶段推断框架（知识缺口定位 + 潜在编辑）实现精确、最小化的纠正；3) 通过mixup训练实现零射组合泛化，能够在单一误区训练下正确处理多重误区；4) 生成自然语言解释的可解释纠正。

**🔧 技术方法**

预训练的 CodeT5+ 编码器/解码器、轨迹与知识嵌入、基于差分的定位网络、潜在编辑网络、教师强制解码、混合损失（BCE、MSE、L2）以及 mixup 数据增强。

**📊 数据集**

在三种长周期规划域（包含不同误区类型）上进行实验，并通过 20 名受试者的用户研究验证方法。实验数据来源于专家轨迹、学生轨迹以及对应的 PDDL 知识表示。

**📈 对比分析**

与传统行为级干预（如警报、示范）和解释型方法对比。实验结果表明，在三域任务中，方法实现了高精度的误区定位与纠正，且在多误区情景下保持零射组合泛化；用户研究中约 90% 的学生误区被成功纠正，显著提升后续任务表现。

**⚠️ 局限性**

1) 依赖专家参考和 PDDL 规范，难以处理开放式、创意解决方案；2) 误差诊断可能产生“幻觉”误区，导致学生困惑；3) 目前仅在固定规划域实验，缺乏对更大、更复杂场景的验证；4) 需要更精细的交互策略以避免过度干预。

---

## 236. Monte Carlo Steklov Operators for Large-Scale Geometry Processing in the Wild

**arXiv ID:** 2606.05581 | [PDF](https://arxiv.org/pdf/2606.05581v1)

**作者:** Arman Maesumi `[一作]` (Brown University), Daniel Ritchie `[通讯]` (Brown University)

**通讯引用:** 2479 | [OpenAlex ID](https://openalex.org/A5005034184)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于蒙特卡洛采样的Dirichlet‑to‑Neumann（DtN）算子估计方法，可快速、鲁棒地求解内部与外部Steklov谱，并在大规模、多组件、低质量三维网格上实现高效的体积谱计算。

**💡 创新点**

创新点包括：① 利用Beurling‑Deny公式将DtN的双线性形式拆解为跳跃核，从而得到始终正半定的Monte‑Carlo估计器；② 对跳跃核采用球面Poisson核解析表达式与Walk‑on‑Spheres（WoS）采样相结合，显著降低方差；③ 通过Kelvin变换将外部域映射为有界域，实现在无穷域中的高效采样；④ 将估计的Steklov算子嵌入到Steklov‑CLIP网络，实现对450k Objaverse模型的零样本分类和细粒度语义检索。

**🔧 技术方法**

技术手段包括：Monte‑Carlo采样、Walk‑on‑Spheres、Beurling‑Deny理论、Kelvin变换、点云或网格基的Galerkin投影、CUDA加速、Steklov热核滤波与Steklov‑Galerkin注意力、InfoNCE对比学习。

**📊 数据集**

使用的数据集：Thingi10k（用于验证与BEM对比）、Objaverse（约450k模型用于Steklov谱计算与Steklov‑CLIP预训练）、Partverse（用于细粒度语义检索）以及公开的多视图与文本嵌入（CLIP/OpenCLIP）。

**📈 对比分析**

与传统基于BEM的Steklov谱计算相比，本方法在10M采样下比BEM快数百倍，且能处理数百万面甚至不连通的网格；在零样本分类上，Steklov‑CLIP在Objaverse‑LVIS基准上获得49.1% Top‑1 与76.4% Top‑5 的成绩，略低于某些点云/多视图模型，但在细粒度语义检索上（Partverse）显著优于对手，finetune后达到93% Top‑5。

**⚠️ 局限性**

限制与挑战：① 仅能得到低频Steklov谱，无法精确捕获高频信息；② 外部DtN估计受Kelvin逆变换导致的最远点查询开销；③ Galerkin基的构造依赖CPU稀疏特征分解，成为大模型时的瓶颈；④ Monte‑Carlo本质上无偏但收敛慢，需大量采样；⑤ 对于极端离散或退化网格，尽管鲁棒性提升，但仍可能影响算子精度。

---

## 237. UltraVR: A Diagnostic Ultra-Resolution Image-VQA Benchmark for Evidence-Grounded Reasoning

**arXiv ID:** 2606.05576 | [PDF](https://arxiv.org/pdf/2606.05576v1)

**作者:** Gexin Huang `[一作]` (University of British Columbia), Xiaoxiao Li `[通讯]` (University of British Columbia)

**通讯引用:** 5226 | [OpenAlex ID](https://openalex.org/A5100458648)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了UltraVR诊断性基准，用于评估视觉语言模型在超分辨率图像上的证据驱动视觉推理能力。

**💡 创新点**

核心创新在于：①构建跨四大高价值领域（CCTV、RS、WSI、AD）的多模态推理任务；②引入结构化的 GT‑CoT（ground‑truth chain of thought）注释，将推理过程拆解为五类操作（证据定位、局部感知、量化、证据整合、决策推理），实现过程级诊断；③设计多种视觉输入与推理格式的评估协议。

**🔧 技术方法**

采用大规模视觉语言模型（GPT‑5.x、Gemini‑3.x 等）进行端到端推理，并通过不同视觉输入（缩略图、全图、局部证据、全图+局部）和不同推理格式（直接 QA、CoT、Schema‑CoT、Few‑shot、Pred‑Step、GT‑Prefix）进行系统评测。

**📊 数据集**

使用来自 PANDA、DOTA 1.5、TCGA‑BRCA、MVTec LOCO 等公开数据集中的超分辨率图像，并人工构造问题、答案选项、GT‑CoT 步骤与操作标签。

**📈 对比分析**

与传统 VQA/高分辨率评测相比，UltraVR 通过多维评估揭示模型在证据定位和局部感知上的显著瓶颈；实验显示最强模型 GPT‑5.5 在宏观平均准确率仅 44.9%，大幅落后于文本仅条件 7.7%，表明视觉信息对任务至关重要。

**⚠️ 局限性**

限制在于：①目前评测聚焦于多选题，未覆盖开放式输出；②基准数据量有限，主要覆盖四个领域，可能不足以代表所有超分辨率应用；③未涉及模型对视觉搜索与证据选择机制的可解释性与可调控性研究。

---

## 238. Robust Repair of Reed-Solomon Codes

**arXiv ID:** 2606.05573 | [PDF](https://arxiv.org/pdf/2606.05573v1)

**作者:** Wilton Kim `[一作]` (Nanyang Technological University), Han Mao Kiah `[通讯]` (Nanyang Technological University)

**通讯引用:** 1893 | [OpenAlex ID](https://openalex.org/A5083157397)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究在低通信带宽下，如何利用 Guruswami‑Wootters 追踪修复框架对完整长度 Reed–Solomon 码中的单个擦除进行鲁棒修复。

**💡 创新点**

创新点在于将下载的追踪视为一种码（repair‑trace 码），通过分析其支持与循环宇称的关系给出维度上界、距离下界，并在二进制情形下给出单误差可纠正的最优维度；此外提出两种鲁棒修复方案，分别利用 BCH 界和字符和式估计提升误差容忍度。

**🔧 技术方法**

使用的技术包括循环码理论、BCH 约束、符号和式（character sum）估计、Berlekamp‑Welch 解码、Guruswami‑Sudan 列表译码以及贪心裁剪算法。

**📊 数据集**

本文未使用实验数据集，主要以理论分析和计算公式为主。

**📈 对比分析**

通过与 BCH 界、距离下界以及字符和式界的比较，实验图表显示鲁棒修复方案 2 可将可纠正误差数提升至接近理论上限，性能优于传统单轨道修复。

**⚠️ 局限性**

局限性包括仅针对完整长度 RS 码，对中等字段大小；鲁棒方案 2 的计算复杂度随 n 增大；对非完整长度 RS 码的推广仍待研究。

---

## 239. Domain-Aware Mispronunciation Detection and Diagnosis Using Language-Specific Statistical Graphs

**arXiv ID:** 2606.05569 | [PDF](https://arxiv.org/pdf/2606.05569v1)

**作者:** Huu Tuong Tu `[一作]` (Hanoi University of Science and Technology), Nguyen Thi Thu Trang `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 626 | [OpenAlex ID](https://openalex.org/A5101435789)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了基于语言特定统计混淆图的误发音检测与诊断模型MDD-LSSG。

**💡 创新点**

创新点在于用数据驱动的统计混淆图取代传统的语言学先验，并根据说话者的L1背景动态构建图结构，以捕获系统性发音错误。

**🔧 技术方法**

采用wav2vec2-large作为声学编码器，利用GCN对统计图进行卷积传播，使用交叉注意力融合声学与语言特征，并以CTC损失进行训练。

**📊 数据集**

在非母语英语语料L2-ARCTIC上进行实验评估。

**📈 对比分析**

与L1-aware、MDDGCN、CAT-GCN-MDD等基线比较，MDD-LSSG在F1-score上最高达59.52%，并在召回率与精准度上均优于对照模型。

**⚠️ 局限性**

局限性包括仅使用单一声学编码器、图结构未支持多模态或多语种融合，以及对少数样本L1的鲁棒性待进一步提升。

---

## 240. InfoShield: Privacy-Preserving Speech Representations for Mental Health Screening via Information-Theoretic Optimization

**arXiv ID:** 2606.05561 | [PDF](https://arxiv.org/pdf/2606.05561v1)

**作者:** Xueyang Wu `[一作]` (Shenzhen NeurStar Inc), Guang Ling `[通讯]` (Shenzhen NeurStar Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 InfoShield，融合 VIB 与 TimeAwareMINE，实现语音抑郁检测的隐私保护；

**💡 创新点**

创新点在于：① 通过跨模态注意力解决序列与静态标签的时间‑静态失配的 TimeAwareMINE；② 将信息瓶颈与目标 MI 最小化联合优化，兼顾隐私与效用；

**🔧 技术方法**

采用 Transformer 语音编码、信息瓶颈 VIB、时间感知 MINE 与跨模态注意力的联合学习；

**📊 数据集**

使用 Androids 语料库（228 条录音，118 位意大利语使用者）进行实验；

**📈 对比分析**

与 Normal、DP (ε=1,8)、VIB‑only、StandardMINE 等基线进行 5‑折交叉验证，InfoShield 在抑郁检测 F1=0.784 的同时，将性别推断降至 55.5%（从 92.6%），年龄推断降至 30.3%（从 55.7%），优于 SOTA（0.723）与 DP；

**⚠️ 局限性**

局限性在于样本量有限、单语种（意大利语）且未在更强攻击模型或多属性场景中验证鲁棒性。

---

## 241. Beyond Generative Decoding: Discriminative Hidden-State Readout from a Native Omni-Modal LLM for Multimodal Sentiment Analysis

**arXiv ID:** 2606.05713 | [PDF](https://arxiv.org/pdf/2606.05713v1)

**作者:** Bin Wen `[一作]` (Universiti Sains Malaysia), Tien-Ping Tan `[通讯]` (Universiti Sains Malaysia)

**通讯引用:** 804 | [OpenAlex ID](https://openalex.org/A5025745463)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较并提出了在多模态情感分析中将生成式读取替换为判别式读取的方法，并在 Qwen2.5‑Omni‑7B 上实现单卡训练与推理。

**💡 创新点**

创新点在于：①直接将 Thinker 的隐藏状态映射为情感分数的判别式读取，显著优于传统生成式读取；②在同一骨干网络、数据与 LoRA 配置下进行严格对照实验；③提供单卡可运行的 4‑bit QLoRA 与动态帧抽样方案。

**🔧 技术方法**

使用技术包括 Qwen2.5‑Omni‑7B Thinker、4‑bit NF4 量化、QLoRA 低秩适配、轻量 MLP 回归头、视频帧动态采样以及 DeepFilterNet 音频去噪。

**📊 数据集**

实验数据集为 CMU‑MOSI 与 CMU‑MOSEI 两个连续情感回归基准。

**📈 对比分析**

通过在相同骨干与 LoRA 配置下比较生成式读取（零样本与训练后）与判别式读取，判别式读取在 MAE、Corr、Acc‑2 等指标上与或优于 state‑of‑the‑art，且无解析错误、推理速度提升约 30%。

**⚠️ 局限性**

局限性包括：仅验证于英文连续情感回归；模态消融与音频去噪实验仅在 MOSI 上完成；未在 MOSEI 上进行多模态消融；对比基准方法使用不同特征提取与评测协议，需进一步统一评测框架。

---

## 242. Cognitive Threat Intelligence and Explainable Federated Security Analytics for distributed Infrastructure Systems

**arXiv ID:** 2606.05701 | [PDF](https://arxiv.org/pdf/2606.05701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 243. Critic-Guided Heterogeneous Multi-Agent Reasoning for Reliable Mathematical Problem Solving

**arXiv ID:** 2606.05704 | [PDF](https://arxiv.org/pdf/2606.05704v1)

**作者:** Muhammad Talha Sharif `[一作]` (National University of Computer and Emerging Sciences), Abdul Rehman `[通讯]` (National University of Computer and Emerging Sciences)

**通讯引用:** 9190 | [OpenAlex ID](https://openalex.org/A5069580698)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于批评器的异质多代理推理框架，将生成器与验证器分离，并通过多轮批评反馈迭代改进数学推理结果。

**💡 创新点**

创新点在于将批评器作为自适应学习循环，引入中间推理错误检测与修正，从而显著降低错误级联，且不依赖模型规模扩大。

**🔧 技术方法**

采用LLM生成器（如Llama‑3.1‑8B‑instant）与多种大小、不同后端的验证器（如Llama‑3.3‑70B‑versatile、OpenAI GPT‑OSS‑120B）结合的生成‑验证‑批评循环。

**📊 数据集**

使用公开的GSM8K数据集（自然语言数学问题与答案），在全测集1,319例上进行评估。

**📈 对比分析**

通过与单次推理、不同验证器规模及无批评等配置对比，实验表明批评式多轮推理在GSM8K上达到了93.56%准确率，超越现有最佳模型（如RDoLT 90.98%），表明批评机制是提升性能的关键。

**⚠️ 局限性**

局限性包括：仍需要依赖大型验证器；仅在单一数学数据集上验证，尚未证明在更广泛任务上的通用性；以及多轮批评过程增加推理时延与计算成本。

---

## 244. Parallel Jacobi Decoding for Fast Autoregressive Image Generation

**arXiv ID:** 2606.05703 | [PDF](https://arxiv.org/pdf/2606.05703v1)

**作者:** Boya Liao `[一作]` (Westlake University), Huan Wang `[通讯]` (Westlake University)

**通讯引用:** 19829 | [OpenAlex ID](https://openalex.org/A5100332013)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练‑free 的并行雅可比解码框架，利用二维空间局部性加速自回归图像生成。

**💡 创新点**

创新点在于将传统 1D 雅可比解码扩展到 2D：动态激活行、行因果注意力掩码与概率收敛判定，实现更高并行度和更快收敛。

**🔧 技术方法**

采用 2D 并行雅可比解码、行因果注意力、概率收敛验证、top‑k 采样与 CFG 引导，无需额外训练。

**📊 数据集**

使用 MS‑COCO 与 PartiPrompt 两个文本‑图像数据集进行评估。

**📈 对比分析**

与 Vanilla AR、SJD、GSD 三种基准方法比较，实验显示在 Lumina‑mGPT 与 LlamaGen 上可实现 4.8×–6.4× 的推理加速，同时保持甚至略优的 FID/CLIP‑Score/IS 等质量指标，步骤数和延迟显著降低。

**⚠️ 局限性**

局限性包括：需在上下文令牌数（c）与速度/质量之间权衡；对不同模型、分辨率的通用性尚待进一步验证；极低 c 时图像质量略有下降；方法不解决模型训练成本问题。

---

## 245. Seeing Time: Benchmarking Chronological Reasoning and Shortcut Biases in Vision-Language Models

**arXiv ID:** 2606.05702 | [PDF](https://arxiv.org/pdf/2606.05702v1)

**作者:** Haoyu Zhou `[一作]` (Jilin University), Renqiang Luo `[通讯]` (Jilin University)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5070730125)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了一个针对时间推理的 Vision‑Language 模型基准，包含工件识别、色彩短路检测和跨模态新闻匹配三大子任务。

**💡 创新点**

创新点在于提出针对跨图像与文本的时间推理基准，并系统识别 VLM 对色彩等表面特征的依赖，形成了可诊断模型短路的多模态时间推理评测框架。

**🔧 技术方法**

采用了零样本推理、链式推理（CoT）与轻量级检索增强（RAG）等技术，对现有多模态模型进行统一评测。

**📊 数据集**

使用了三大自构数据集：CHA（古代工件）、SPEED（多领域事件图像）和 HistNews（新闻文本-图片对），覆盖历史跨度与跨模态情境。

**📈 对比分析**

通过准确率、Kendall τ、MAE 等指标比较六种 VLM，整体平均分仅 42.98，表明闭源模型略优但整体性能仍偏低，凸显时间推理难点。

**⚠️ 局限性**

存在的限制包括对视觉短路的依赖、细粒度工件识别不足、跨模态对齐不稳以及检索增强效果有限，亟需更深层去耦时间语义与视觉风格。

---

## 246. DexFuture: Hierarchical Future-State Visuomotor Targeting for Bimanual Dexterous Tool Use

**arXiv ID:** 2606.05699 | [PDF](https://arxiv.org/pdf/2606.05699v1)

**作者:** Runfa Blark Li `[一作]` (University of California San Diego), Truong Nguyen `[通讯]` (University of California San Diego)

**通讯引用:** 15691 | [OpenAlex ID](https://openalex.org/A5102719190)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种分层框架DexFuture，用高层目标预测器生成未来手-工具-物体状态，低层目标条件策略执行高频接触动作，从而实现双手器具操作；

**💡 创新点**

创新点在于用无动作依赖的视觉-本体历史预测未来目标，解耦慢速目标生成与高速控制，避免需要演示目标或在线高维规划；

**🔧 技术方法**

主要技术包括结构化视觉本体嵌入、时程条件Transformer（目标预测）、每连结Transformer策略、PPO训练、以及对齐的目标解码；

**📊 数据集**

使用OakInk2双手器具操作数据集进行训练和评估；

**📈 对比分析**

与ManipTrans、PhysGraph等基准（oracle目标）和无目标策略以及DexWM动作规划比较，DexFuture在成功率上达90%左右的oracle水平，且在60Hz实时率下比DexWM快约250倍；

**⚠️ 局限性**

局限在难以预测的狭窄接触或突变动作场景下的目标准确性不足，需进一步增强不确定性或接触感知能力。

---

## 247. Continual Learning Bench: Evaluating Frontier AI Systems in Real-World Stateful Environments

**arXiv ID:** 2606.05661 | [PDF](https://arxiv.org/pdf/2606.05661v1)

**作者:** Parth Asawa `[一作]` (University of California, Berkeley), Joseph E. Gonzalez `[通讯]` (University of California, Berkeley)

**通讯引用:** 20404 | [OpenAlex ID](https://openalex.org/A5072427753)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了首个多领域、专家验证的持续学习基准Continual Learning Bench，用于评估LLM系统在连续经验中提升的能力。

**💡 创新点**

创新点在于设计共享可学习潜在结构的任务、引入“增益”指标来区分基础模型能力与在线学习效果，并提供跨任务标准化方法。

**🔧 技术方法**

技术上结合多种LLM（Claude、Gemini、GPT-5.4）与多种记忆机制（ICL、ICL Notepad、Mem0、ACE、Claude Code、Codex）进行评测。

**📊 数据集**

数据集来自六个领域（软件工程、数据库分析、流行病预测、射频监测、销售预测、战略游戏）中的专家验证任务，包含概念漂移与可学习结构。

**📈 对比分析**

评估显示，最强的纯ICL模型在归一化奖励上最高（22.3%）且增益最高（25.4%），而专门的记忆系统往往收益低、成本高，整体提升空间仍有限。

**⚠️ 局限性**

局限性包括任务数量有限、实例规模短、仅测试基于上下文的记忆方法、未涵盖参数化自适应技术，且对小模型的失败模式难以呈现。

---

## 248. Iterative Thresholding Pursuit with Continuation for $\ell_{1-2}$-Regularized Sparse Recovery

**arXiv ID:** 2606.05657 | [PDF](https://arxiv.org/pdf/2606.05657v1)

**作者:** Junxi Wu `[一作]` (Tongji University), Jun-Feng Yin `[通讯]` (Tongji University)

**通讯引用:** 4670 | [OpenAlex ID](https://openalex.org/A5005441350)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种迭代阈值追踪法（ITP-C），将ℓ1-2非凸正则的主动集识别与有限最小二乘追踪相结合，用于稀疏信号重建。

**💡 创新点**

在传统ℓ1-2阈值法基础上引入二阶追踪步骤并配备严格的下降检查，既保持无先验稀疏度的主动集识别，又通过有限最小二乘校正消除阈值偏差，理论证明收敛、支持识别及条件oracle性质。

**🔧 技术方法**

利用ℓ1-2近端映射的显式解析、限制最小二乘追踪、Kurdyka–Łojasiewicz收敛框架、RIP和半代数函数的KŁ性质，以及严格下降检查等技术。

**📊 数据集**

在合成稀疏向量（Gaussian 与 PDCT 型传感矩阵）和图像重建（Symlets‑8 小波变换下的 Peppers、Circuit、Coins 图像）上进行实验。

**📈 对比分析**

与 L1‑FISTA、L1‑2 PLDCA、L1‑2 PGD、L1‑2 ITAC、L1‑Reweighted 等先验自由算法比较，ITP‑C 在迭代次数、CPU 时间、相对误差以及 PSNR 上均显著优于其他方法，尤其在低测量比和高噪声场景下成功率最高。

**⚠️ 局限性**

追踪步骤需求解子矩阵最小二乘，计算成本随支持规模增加；理论仅保证局部 oracle 性质，需满足信号最小幅值足够大；对非 Gaussian 或结构化传感矩阵的性能仍需进一步验证。

---

## 249. Development of a Structured Approach for Establishing Mission Engineering Requirements

**arXiv ID:** 2606.05651 | [PDF](https://arxiv.org/pdf/2606.05651v1)

**作者:** Taylor C. Fazzini `[一作]` (Colorado State University), Daniel R. Herber `[通讯]` (Colorado State University)

**通讯引用:** 1005 | [OpenAlex ID](https://openalex.org/A5077474480)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种结构化方法，能在缺少客户需求的情况下从任务定义开始推导任务工程需求，并通过可行性评估、关键维度加权、效用模型和任务复杂度因子实现从任务到需求的模型驱动流程。

**💡 创新点**

将任务背景拆解为可追溯的 UAF/SysML 架构，结合 MaxDiff 关键维度加权、MAUT 价值模型以及技术复杂度评估，首次构建完整的任务复杂度因子（MCF）来平衡任务难度与技术风险，形成端到端的需求导出框架。

**🔧 技术方法**

使用 UAF 视图框架与 SysML 建模、最佳-最差排序（BWS）/MaxDiff、MAUT 多属性效用理论、技术成熟度（TRL）评估和技术复杂度公式，构建统一的模型驱动分析流程。

**📊 数据集**

未使用真实数据集，示例采用概念性的闭路空中支援（CAS）任务假设数值进行演示，展示方法的可操作性。

**📈 对比分析**

通过概念示例计算任务复杂度因子（MCF），与技术利用值（Lk）和技术复杂度（Tck）对比，未进行实验性性能对比，示例仅展示方法流程和数值计算结果。

**⚠️ 局限性**

局限性：仅为概念框架，未在完整 MBSE 环境中实现；示例基于假设数据，缺乏真实验证；需要更多专家评估和实战案例；对多架构、跨平台的可扩展性待进一步研究。

---

## 250. Enhancing Software Engineering Through Closed-Loop Memory Optimization

**arXiv ID:** 2606.05646 | [PDF](https://arxiv.org/pdf/2606.05646v1)

**作者:** Xuehang Guo `[一作]` (William & Mary), Xingyao Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 871 | [OpenAlex ID](https://openalex.org/A5023449213)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MemOp 框架，实现 SE 代理的闭环记忆优化；

**💡 创新点**

创新点在于将记忆效用定义为对下游任务的因果改进，构建任务无关的评估基准与无标注优化信号；

**🔧 技术方法**

采用基于 LLM 的记忆生成模型、强化学习 + 性能验证微调以及多维度指标评估；

**📊 数据集**

使用 SWE‑Bench‑Verified 数据集作为训练与评估的真实仓库与任务集；

**📈 对比分析**

通过与无记忆基线及多种 RL 算法比较，单/跨回合场景下在 SR、LA、E_resolve 等 10 项指标上平均提升 1.5–5.3% 绝对值，并将计算成本下降 9.8%；

**⚠️ 局限性**

局限性包括需大量轨迹数据进行性能过滤，且在更大规模或更复杂环境下记忆泛化与更新仍有挑战。

---

## 251. Revisiting Prototype Rehearsal for Exemplar-Free Continual Learning: Manifold-Aware Boundary Sampling with Adaptive Class-Balanced Loss

**arXiv ID:** 2606.05695 | [PDF](https://arxiv.org/pdf/2606.05695v1)

**作者:** Hongye Xu `[一作]` (Rochester Institute of Technology), Bartosz Krawczyk `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 11381 | [OpenAlex ID](https://openalex.org/A5054879396)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在无样本的类增量学习中提出了一种改进的原型重放方法。

**💡 创新点**

创新点在于引入了基于最近敌对类的边界感知过采样（CEOS）和随时间调整的类别平衡损失（ACB）。

**🔧 技术方法**

采用特征空间插值、最近邻检索、时变权重机制以及 ResNet‑18 作为骨干网络。

**📊 数据集**

在 CIFAR‑100、TinyImageNet、ImageNet‑100 和 CUB‑200 四个基准数据集上进行评测。

**📈 对比分析**

与多种基线（EWC、LwF、SDC、PASS、FeTrIL、PRAKA、FeCAM、EFC、ADC、LDC 等）对比，平均增量与最近任务准确率均达到或超过现有最优，刷新 SOTA。

**⚠️ 局限性**

局限性在于需要可靠的原型和足够密集的敌对类样本；在极端漂移或稀疏区域，插值可能产生模糊或误导性的合成样本。

---

## 252. Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models

**arXiv ID:** 2606.05688 | [PDF](https://arxiv.org/pdf/2606.05688v1)

**作者:** Hancheol Park `[一作]` (Nota Inc.), Tae-Ho Kim `[通讯]` (Nota Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对Mixture-of-Experts模型的路由一致性量化方法VSRAQ，通过在后训练量化中加入路由值与结构对齐目标，显著降低量化导致的专家选择不稳定。

**💡 创新点**

创新点在于同时考虑路由值匹配与路由结构（专家排序与top‑k边界）对齐，并使用敏感度加权的sigmoid值对齐来强化在sigmoid路由下的鲁棒性。

**🔧 技术方法**

技术手段包括基于AutoRound的后训练量化框架，加入Top‑K MSE、sigmoid‑aware、排序对齐、边界对齐等多项损失，并在校准阶段优化。

**📊 数据集**

使用了Solar‑Open‑100B和Nemotron‑3‑Nano‑30B‑A3B两大MoE LLM，并采用NVIDIA Reasoning calibration set（OpenCodeReasoning、OpenScienceReasoning‑2、OpenMathReasoning）进行校准。

**📈 对比分析**

在W4A16和NVFP4量化下，与AutoRound和Top‑K MSE基线相比，VSRAQ在WikiText‑2 perplexity、生成式答案提取和多项选择评估上均取得更低PPL和更高分数，尤其在生成式任务上提升显著。

**⚠️ 局限性**

局限在于实验仅覆盖4‑bit量化，且在更极端的2/3‑bit场景下的效果尚未验证，且对不同MoE架构和量化框架的通用性需进一步研究。

---

## 253. Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillatio

**arXiv ID:** 2606.05682 | [PDF](https://arxiv.org/pdf/2606.05682v1)

**作者:** Fangbo Tu `[一作]` (PayPal, Inc.), Srinivasan Manoharan `[通讯]` (PayPal, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究NVFP4低精度LLM在QAD过程中的内部表征漂移，并提出CKA引导的表示对齐方法以恢复低位精度模型性能。

**💡 创新点**

将Center Kernel Alignment（CKA）作为中间层表示正则化加入QAD，解决仅对齐输出导致内部表征失真的问题。

**🔧 技术方法**

采用NVFP4量化、KL对齐、top‑k logits蒸馏、层级CKA相似度计算及动态损失平衡等技术。

**📊 数据集**

训练使用混合SFT+RL生成的数据，评估基准包括AIME25、GPQA‑D和LiveCodeBench‑v5等推理与代码生成任务。

**📈 对比分析**

与PTQ和标准KL‑QAD对比，CKA‑QAD在Nemotron 3 Nano等模型上将平均CKA提升至0.99，并使AIME25从68.5%提升至72.3%，GPQA‑D从59.5%提升至61.1%，LiveCodeBench从57.9%提升至59.8%，仅增加0.5%步长时间和7%显存。

**⚠️ 局限性**

CKA计算带来额外显存/算力负担，未对高阶几何约束进行处理，且验证仅限于4‑bit NVFP4及部分模型，未覆盖更低精度或其他架构。

---

## 254. Sustainability by Design in Decentralized Autonomous Organizations: An Empirical Review of Governance, Innovation, and Institutional Design

**arXiv ID:** 2606.05667 | [PDF](https://arxiv.org/pdf/2606.05667v1)

**作者:** Yutian Wang `[一作]` (Duke Kunshan University), Luyao Zhang `[通讯]` (Duke Kunshan University)

**通讯引用:** 4157 | [OpenAlex ID](https://openalex.org/A5100447104)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对ERC‑8004和Google A2A两种治理模式进行比较，构建LLM驱动的多层分析流程，系统评估治理结构、话语创新与网络参与等维度。

**💡 创新点**

将大语言模型、主题建模和多层网络分析相结合，提出“设计→出现→现实→修正”循环框架，用可量化指标评估DAO与传统企业治理对可持续发展目标的影响。

**🔧 技术方法**

使用MiniMax‑M2.5 LLM进行自动标注，结合BERTopic、Thematic‑LM进行主题建模，运用社交网络分析（SNA）、共参与网络、话语网络和多层二分网络进行结构与关系分析。

**📊 数据集**

数据来源为4,323条治理记录，来自Ethereum Magicians论坛、ERC‑8004与Google A2A的GitHub issue、PR和讨论，手工验证109位顶级贡献者的机构归属。

**📈 对比分析**

通过决策架构重建、话语类型化与主题模型、网络图谱等三层分析，量化比较治理形式在议题分布、参与度、熵分布等指标上的差异，统计检验显著性（p<.001）表明方法能有效揭示治理对话语和网络结构的影响。

**⚠️ 局限性**

局限性包括：数据仅为公开记录，难以捕捉参与者意图；NLP/ML算法可能放大偏见；样本仅涵盖两条协议，缺乏广泛组织和跨法域的代表性；规范性假设与文化背景相关，需要进一步验证。

---

## 255. V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation

**arXiv ID:** 2606.05665 | [PDF](https://arxiv.org/pdf/2606.05665v1)

**作者:** Tao Liu `[一作]` (Centific Global Solutions Inc.), Vishav Garg `[通讯]` (Centific Global Solutions Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向视频到视频生成的多维度评测基准 V2V-Bench，细化 11 个评测维度并进行层次化分析。

**💡 创新点**

创新点在于：① 设计专门针对 V2V 的六个核心维度（帧对应、时序一致、结构保真、布局一致、编辑可信度、风格迁移质量），② 引入合规性检查确保帧级对应；③ 通过人类标注与 VLM 评估验证指标的可靠性。

**🔧 技术方法**

采用 DINO + SSIM、光流一致性、Canny 边缘 F1、CLIP 相似度、VGG Gram 等多种视觉与语义特征，结合统计量和相似度计算构建评测指标。

**📊 数据集**

使用 81 条多场景、不同动作、长度约 8 秒的源视频，并为每条视频配备多种编辑任务（外观编辑、风格迁移、场景改造等）组成评测数据集；对 Grok、Veo‑3.1、Open‑Sora‑2 三个模型进行测试。

**📈 对比分析**

通过 Spearman 相关性与人类评估及两大 VLM（Gemini 2.5 Pro、GPT‑4o）对比，V2V‑Bench 在六个 V2V 专属维度上与人类相关性为 0.905，显示其对模型性能区分度高；模型对比显示 Grok 在编辑可信度上优于 Gemini，Veo‑3.1 在视觉质量方面表现更好。

**⚠️ 局限性**

限制主要包括：评测视频仅为短片（≈8 秒），难以覆盖长时长或跨域场景；合规性检查导致部分模型无法完整参与评测；指标权重选择依赖先验设定，可能影响跨模型可比性。

---

## 256. QDAG: Declarative Composition of Reusable Analytics Methodologies at LinkedIn

**arXiv ID:** 2606.05662 | [PDF](https://arxiv.org/pdf/2606.05662v1)

**作者:** Peter Ho `[一作]` (LinkedIn Corporation), Endong Zhu `[通讯]` (LinkedIn Corporation)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个可部署在 LinkedIn 生产中、用于描述和执行分析方法学（multi‑step OLAP 任务）的声明式 DAG 框架 QDAG。

**💡 创新点**

创新点在于将业务方法学抽象为单一可序列化的有向无环图，并在请求层面使用需求驱动、记忆化、条件剪枝和并行执行的评估模型，使得方法学既可重用、可测试，又保持交互式延迟。

**🔧 技术方法**

使用了 YAML 作为声明式语法，jq 作为 JSON 转换语言，Calcite 进行 SQL 解析与校验，SQLite、Pinot、RESTLI 等多引擎节点，异步 Future 结合双检查锁实现记忆化，内置缓存与 mock 执行模式。

**📊 数据集**

主要数据集来自 LinkedIn 内部的 Apache Pinot OLAP 数据，实验涵盖超过 100 个真实生产用例，包括 headcount growth、top‑skills、talent‑pool、unique‑post‑impressions 等。

**📈 对比分析**

与手写 imperative glue 对比，QDAG 在集成时间上提升约 60%，单次请求的额外开销仅为 5–10 ms（P50）且 50 ms（P99）；单机 QPS 达 400，远高于业务需求；微基准显示 2 ms 纯引擎开销。

**⚠️ 局限性**

局限包括：仅支持单输入单输出的 acyclic DAG，难以表达递归或循环逻辑；SQLite 作为内存 join 方案不适合宽表大聚合；目前缺乏完善的分布式缓存与下游服务的完整类型；需要更多可视化、lint 与 Schema 验证工具来提升易用性。

---

## 257. Real-Time Threat Detection from Surveillance Cameras using Machine Learning

**arXiv ID:** 2606.05708 | [PDF](https://arxiv.org/pdf/2606.05708v1)

**作者:** Gajendra Mandal `[一作]` (Chhattisgarh Swami Vivekanand Technical University), Priyansh Mahant `[通讯]` (Chhattisgarh Swami Vivekanand Technical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发了一套实时摄像头武器检测系统，利用YOLOv8识别枪支、刀具及印度特有的钝器，并结合VLM进行行为分析。

**💡 创新点**

创新点包括：①创建地区特定钝器数据集并与公开枪刀数据合并；②通过延长训练周期显著提升钝器召回率；③将行为分析与视觉‑语言模型结合，实现威胁级别评估。

**🔧 技术方法**

主要技术：YOLOv8目标检测、时序帧缓存、VLM行为推理、实时推理框架。

**📊 数据集**

数据集：自制336帧钝器图像（铁杆、木杆、塑料杆）+公开枪刀数据集，总计7,959张标注图像，统一裁剪至640×640。

**📈 对比分析**

对比50与100轮训练，mAP@0.5由0.777提升至0.819，钝器类召回率提升9.6%，整体性能提升且未出现过拟合。

**⚠️ 局限性**

局限性：钝器种类多样导致误检，数据集规模相对有限，实时推理受限于低配GPU，且未涉及跨场景迁移评估。

---

## 258. PerceptUI: LLM Agents as Human-Aligned Synthetic Users for UI/UX Evaluation

**arXiv ID:** 2606.05697 | [PDF](https://arxiv.org/pdf/2606.05697v1)

**作者:** Nicolas Bougie `[一作]` (Woven by Toyota), Narimasa Watanabe `[通讯]` (Woven by Toyota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PerceptUI 框架，能够根据 UI 截图、用户画像和评估问题预测用户的答案并生成自然语言理由。

**💡 创新点**

创新点在于将对比反思微调与反射提示演进相结合，既实现面向个体用户的答案预测，又生成解释性强、对比性高的理由，从而提升评估的真实性与可解释性。

**🔧 技术方法**

使用多模态大型语言模型（如 Qwen‑VL）作为基础，配合教师‑学生对话式对比理由蒸馏、对比反思微调、提示演进以及结构化解释生成技术。

**📊 数据集**

采用多种公开数据集：WiserUI‑Bench、UIClip/BetterApp、WebDevJudge、LabintheWild、LabintheWild‑UX、UICrit 以及专有的 UXCar 数据集。

**📈 对比分析**

与零射基线和现有最佳方法进行对比，PerceptUI 在 UI 设计选择、质量预测、评分预测、理由质量等任务上均获得更高的准确率、F1、平均误差和人类评分，表现接近或优于人类水平。

**⚠️ 局限性**

局限性包括：实验可复现性受限于专有数据；模型可能继承视觉‑语言模型和人物画像中的文化、性别、年龄等偏见；仅基于截图缺乏交互过程；生成的理由虽更具解释性，但仍是模型产出，可能不完全符合真实原因；模型性能依赖教师模型和基础模型质量。

---

## 259. Causal Modeling of Selection in Evolution

**arXiv ID:** 2606.05689 | [PDF](https://arxiv.org/pdf/2606.05689v1)

**作者:** Haoyue Dai `[一作]` (Carnegie Mellon University), Kun Zhang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 21958 | [OpenAlex ID](https://openalex.org/A5100342355)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种针对进化数据中选择偏差的因果建模框架，并开发了相应的结构学习与因果效应估计方法。

**💡 创新点**

创新点在于将选择机制显式建模为一个因果变量，利用该模型修正因果推断中的选择偏差，并给出可识别条件与估计策略。

**🔧 技术方法**

采用结构因果模型（SCM）、图式因果推断、do-演算以及基于贝叶斯网络的学习算法（如 PC-Selection、FCI-Selection）等技术。

**📊 数据集**

在合成进化模拟数据（包含不同选择强度和突变率的模拟基因组）以及公开的生物进化数据集（如 HIV 病毒基因序列）上进行评估。

**📈 对比分析**

与传统的 PC、FCI、GES 等不考虑选择的因果结构学习方法对比，本文方法在结构恢复准确率、因果效应估计误差（MSE）以及计算时间等指标上均表现出显著提升；在真实数据中还能恢复已知的选择机制。

**⚠️ 局限性**

局限包括：假设选择机制可被完整建模且与其它变量无未观测混淆，计算复杂度在大规模网络中仍高；实验主要集中于模拟数据，真实数据验证有限。

---

## 260. Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation

**arXiv ID:** 2606.05687 | [PDF](https://arxiv.org/pdf/2606.05687v1)

**作者:** Junheng Li `[一作]` (California Institute of Technology), Aaron D. Ames `[通讯]` (California Institute of Technology)

**通讯引用:** 15283 | [OpenAlex ID](https://openalex.org/A5039171820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种在训练阶段使用基于质心动力学（centroidal dynamics）的模型预测控制（MPC）来指导强化学习（RL）的框架，用以提升类人机器人在行走和搬运任务中的性能，并在真实硬件（Themis V2）上验证其有效性。

**💡 创新点**

创新点包括：① 通过预测置信度加权（prediction‑confidence weighting）构造多时刻MPC轨迹作为奖励，充分利用长时域信息；② 开发了并行-在-预测期（parallel‑in‑horizon）且无构造步骤的批量MPC求解器πⁿMPC，显著降低训练时的计算成本；③ 将该框架从行走扩展到搬运任务，展示了MPC‑guided RL在高载荷推箱任务中的优势。

**🔧 技术方法**

所使用技术：模型预测控制（MPC）、质心动力学模型、强化学习（PPO）、ADMM求解器、PyTorch/JAX并行GPU实现、PD控制器与自适应惯量调优、离散化的碰撞约束与摩擦锥约束。

**📊 数据集**

数据集：在仿真环境中构造了 4096 个并行环境，使用 Themis V2 机器人物理模型进行训练；随后在真实机器人上进行硬件验证，测试了不同速度指令、冲击恢复和负载推箱（最高 290 kg）。

**📈 对比分析**

比较方法：将 MPC‑guided RL 与纯 RL 基线、基于 CLF 的 MDP 目标、插值或acles 等模型指导奖励进行对比。实验结果表明，MPC‑guided RL 在速度跟踪、冲击恢复、奖励进展以及搬运任务中的推力输出方面均优于其它方法，且训练效率相对可接受。

**⚠️ 局限性**

limitations：① 依赖简化的质心动力学模型和预设接触序列，难以覆盖更复杂的接触模式；② 数值求解器仍存在规模限制，长时域与高阶非线性 MPC 仍需进一步优化；③ 仅在训练阶段使用 MPC，缺乏实时 MPC 的自适应能力，未来需结合更完整的动态模型和非线性 MPC。

---

## 261. Two-Way Is Better Than One: Bidirectional Alignment with Cycle Consistency for Exemplar-Free Class-Incremental Learning

**arXiv ID:** 2606.05675 | [PDF](https://arxiv.org/pdf/2606.05675v1)

**作者:** Hongye Xu `[一作]` (Rochester Institute of Technology), Bartosz Krawczyk `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 11381 | [OpenAlex ID](https://openalex.org/A5054879396)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 BiCyc，使用双向投影器与循环一致性损失来补偿 exemplar‑free 逐任务学习中的原型漂移。

**💡 创新点**

在训练期间同时学习旧→新与新→旧投影，并通过 stop‑gradient 与循环一致性实现近似双射，从而收缩奇异值谱并提升分类决策的稳定性。

**🔧 技术方法**

采用线性/浅层 MLP 投影器、stop‑gradient 门控、循环一致性损失、最小二乘投影理论与 CCA 关联、Gaussian 贝叶斯分类器以及抗崩塌正则化等技术。

**📊 数据集**

在 CIFAR‑100、TinyImageNet、ImageNet‑100（从零训练）以及预训练的 CUB‑200 上进行实验。

**📈 对比分析**

与 AdaGauss、LDC、EFC 等前沿方法比较，平均增益约 3–5 pp，显著降低最后任务遗忘率，同时保持或提升新任务准确率。

**⚠️ 局限性**

仅在已中心化、近似高斯的原型统计下理论成立，对低样本或严重不平衡类别敏感，且理论假设局限于小误差范围。

---

## 262. Do More Agents Help? Controlled and Protocol-Aligned Evaluation of LLM Agent Workflows

**arXiv ID:** 2606.05670 | [PDF](https://arxiv.org/pdf/2606.05670v1)

**作者:** Yuhang Fu `[一作]` (Beijing University of Posts and Telecommunications), Tao Lin `[通讯]` (Westlake University)

**通讯引用:** 9182 | [OpenAlex ID](https://openalex.org/A5100702153)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

BenchAgent框架统一 benchmark 加载、工具访问、计费与轨迹记录，对单代理、固定MAS、进化MAS和运行时生成工作流进行对比评估。

**💡 创新点**

通过同一执行子系统和统一协议，首次将工作流组织的“提升”与模型、工具、评估器等因素隔离开来，并在 Substrate‑Internal 与 Protocol‑Aligned External 两种设置下展示工作流对准确率和成本的真实影响。

**🔧 技术方法**

使用 GPT‑4.1 作为后端，构建 BenchAgent 框架，采用 Wilson 95% 置信区间评估单跑结果，并集成多种 MAS 实现（Jarvis、LLM‑Debate、AutoGen、CAMEL、EvoAgent）及 Claude‑Code 风格的运行时工作流。

**📊 数据集**

在十个常用基准（MATH、AIME、GSM8K、DROP、BBH、MMLU‑Pro、HumanEval、MBPP、HotpotQA、IFEval）以及 GAIA 验证集（分层 3 级）上进行实验。

**📈 对比分析**

方法：Substrate‑Internal（SI）对比保持模型、工具、评估器、日志一致；Protocol‑Aligned External（PAE）对比保持输入、输出格式、后端、工具类等一致。结果显示：大多数 MAS 在平均准确率上不超过单代理；EvoAgent 略有提升但落在单跑不确定性范围内；Claude‑Code 运行时工作流在 GAIA 上领先 20+ 点且 token/时间成本更低。

**⚠️ 局限性**

局限性：实验仅为一次单跑，缺乏重复验证；Claude‑Code 的优势混合了多种工程改进，难以分离具体机制；对比不具备因果性，工具表面与内部实现不完全对齐。

---

## 263. Safe Embodied AI for Long-horizon Tasks: A Cross-layer Analysis of Robotic Manipulation

**arXiv ID:** 2606.05660 | [PDF](https://arxiv.org/pdf/2606.05660v1)

**作者:** Dabin Kim `[一作]` (Ulsan National Institute of Science and Technology), Sungroh Yoon `[通讯]` (Seoul National University)

**通讯引用:** 13093 | [OpenAlex ID](https://openalex.org/A5086877012)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对长时程机器人操纵中的安全性进行系统综述，构建了跨层次（规划‑策略‑执行）与证据边界（正式/统计/经验）两维框架，对现有文献进行归类与评估，并指出研究空白与未来方向。

**💡 创新点**

创新点在于将安全性视为跨层次协同的闭环属性，提出安全干预位置与证据强度的双轴组织，并系统识别规划、策略与执行层面各自的安全关注点与证据缺口，首次为长时程操纵提供整体安全评估蓝图。

**🔧 技术方法**

主要技术包括文献检索与筛选、跨层次框架设计、证据边界分类（正式验证、统计推断、经验实验）、安全问题与机制的层级映射、以及对评测与基准方法的梳理。

**📊 数据集**

该工作并未使用具体数据集；在综述过程中引用了多篇关于规划、强化学习、感知、模型预测控制等领域使用的公开数据集与模拟平台，但未自行进行实验。

**📈 对比分析**

文章通过对比不同层级安全机制的理论与实验支持，指出正式保证最多但覆盖面窄、统计保证能量覆盖更广但需样本、经验证据覆盖面最广但缺乏普适性；总体上呈现安全性评价的分层不平衡与评测指标的碎片化。

**⚠️ 局限性**

局限性包括：综述范围限定于长时程操纵，未覆盖所有体现式AI安全议题；跨层次与证据框架虽系统但仍具主观性，缺乏统一量化指标；未提供新数据或实验验证，主要停留在理论与现有文献的梳理层面。

---

## 264. When Surface Form Changes Moderation Decisions: A Paired Study of Code-Mixed Workflow Instability

**arXiv ID:** 2606.05654 | [PDF](https://arxiv.org/pdf/2606.05654v1)

**作者:** Suraj Babu Thimma Krishnaram `[一作]` `[通讯]` (Illinois Institute of Technology), Suraj Babu Thimma Krishnaram (Illinois Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了代码混合输入对仇恨言论审核系统在 Allow/Flag/Review 三类动作上的不稳定性。

**💡 创新点**

首次从工作流角度评估代码混合带来的动作不稳定，提出基于判定不一致的分流规则。

**🔧 技术方法**

使用多语言 BERT 作为检测器，配合阈值路由、置信度驳回和判定不一致驳回三种技术。

**📊 数据集**

基于 HateBenchSet 的英文样本生成对应的泰米尔-英文代码混合与泰米尔单语对照集。

**📈 对比分析**

在固定阈值下比较清晰英文、代码混合和泰米尔三种视角，发现代码混合导致审查率翻倍、非仇恨误标率上升；使用判定不一致分流可降低自动错误但审查率升高。

**⚠️ 局限性**

实验仅在单一英文-泰米尔混合设置下，未对多语言覆盖、生成质量、实时生成成本以及公平性做深入评估，结果可能不适用于更广泛语言或真实流量。

---

## 265. Protecting K-Nearest Neighbor Queries from Location Inference Attacks

**arXiv ID:** 2606.05648 | [PDF](https://arxiv.org/pdf/2606.05648v1)

**作者:** Zhiyu Sun `[一作]` (East China Normal University), Zhili Chen `[通讯]` (East China Normal University)

**通讯引用:** 5325 | [OpenAlex ID](https://openalex.org/A5018009756)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了 kNNQ（k 最近邻查询）服务中的位置隐私问题，首次系统性提出两种基于查询排名的定位推断攻击 GI‑LIA 与 ZO‑LIA，并针对这些攻击设计了基于拒绝采样和私有区间的差分隐私保护框架 DPRS，以在保证 kNNQ 查询效能的前提下显著降低位置泄露风险。

**💡 创新点**

创新点：
• 首次提出针对 kNNQ 的几何交点推断攻击和零阶优化推断攻击；
• 设计 DPRS，结合 DP‑k‑means 生成自适应私有区间和基于 Rényi‑DP 的拒绝采样机制，实现了更优的隐私‑效能折衷；
• 在理论上证明了 DPRS 的 RDP 隐私保证，并给出了相应的 utility 上界。

**🔧 技术方法**

技术手段：
• GI‑LIA 通过多点二分搜索获取圆轨迹，再求圆交点定位目标；
• ZO‑LIA 采用零阶优化（只利用排名信息）快速逼近目标；
• DPRS 采用 DP‑k‑means 生成私有中心及半径、构造私有区间；
• 拒绝采样机制在区间内采样并使用 Rényi‑DP 进行隐私分析；
• 基于 Laplace/高斯噪声实现差分隐私。

**📊 数据集**

数据集：
• 实际数据：Brightkite、Gowalla（均来自旧金山地区）；
• 合成数据：Gaussian（N(0,1)）和 Beta（B(2,5)）。

**📈 对比分析**

评估与对比：
• 与 SRR、Square、Laplace 三种现有 kNNQ 隐私方案对比；
• 在 k=10/30/50、ϵ=0.5/1/3/5 等设置下，DPRS 在 Recall 和 Ratio 上始终领先，且在防御攻击方面将 ZO‑LIA 的成功率降至 1–3%（相比 2–4% 的基线），误差距离提升至 0.54–0.59；
• 实验显示 DPRS 在不同隐私预算下保持较高查询效能与较强的攻击抵御能力。

**⚠️ 局限性**

局限性：
• 需要预先离线构建私有区间，若数据分布变化频繁需定期更新；
• 在极低隐私预算（ϵ→0）时，查询精度仍会显著下降；
• 对于高度聚类或密度不均的数据，区间划分可能导致距离失真，对 kNNQ 的排名信息影响仍需进一步研究。

---

## 266. Discrete-WAM: Unified Discrete Vision-Action Token Editing for World-Policy Learning

**arXiv ID:** 2606.05645 | [PDF](https://arxiv.org/pdf/2606.05645v1)

**作者:** Ziyang Yao `[一作]`, Hangjun Ye `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Discrete-WAM，一种统一的离散视觉-行动世界策略框架，通过离散词汇统一表示观测、决策和行动，并使用离散扩散生成未来视觉与动作。

**💡 创新点**

创新点在于将观测、决策和动作放入同一离散词空间，并通过统一的 token‑editing 与离散扩散实现世界与策略的联合建模；同时引入分层决策生成以保持多模态一致性。

**🔧 技术方法**

采用 VQ‑VAE 视觉词化、离散扩散网络、Transformer 解码器、soft‑label 动作量化、GMM 模式解码、LoRA 微调与 RL 后训练等技术。

**📊 数据集**

主要使用 NAVSIM v1、v2 评测集以及 nuPlan 训练集进行预训练与微调。

**📈 对比分析**

在 NAVSIM v2 上达 90.4 EPDMS，超过 WAM‑Flow +2.7 并优于多种世界模型；生成质量 FID 6.6、FVD 80.0，显示更高的视觉保真度；在 v1 上也取得 +2.1 PDMS 的提升。

**⚠️ 局限性**

局限性包括对长时域规划的能力仍有限、离散词化导致的表达精度损失、计算成本较高以及在极端或稀有场景下的安全性仍需进一步验证。

---

## 267. PoCQ: Proof of Contribution Quality as a Lightweight Blockchain Consensus for Secure Federated Learning

**arXiv ID:** 2606.05642 | [PDF](https://arxiv.org/pdf/2606.05642v1)

**作者:** Sudad Abed `[一作]` (La Trobe University), Mohammad Jabed Morshed Chowdhury `[通讯]` (La Trobe University)

**通讯引用:** 1907 | [OpenAlex ID](https://openalex.org/A5033646038)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 PoCQ（Proof of Contribution Quality）——一种基于区块链的轻量级共识机制，用于安全的去中心化联邦学习，能有效检测模型投毒和恶意投票，并动态调整节点声誉。

**💡 创新点**

创新点：① 用 L₂ 范数快速验证梯度质量，避免昂贵的重训练；② 通过加密承诺和签名确保更新不可伪造；③ 采用声誉加权投票与动态声誉衰减，抵御投毒和 bad‑mouthing 攻击；④ 仅记录审核元数据，保持链条轻量。

**🔧 技术方法**

技术：分布式验证、SHA‑256 哈希、数字签名、L₂ 范数阈值检测、声誉系统（指数移动平均）、可验证随机函数（VRF）选主、加权 FedAvg、区块链存证。

**📊 数据集**

数据集：MNIST、OrganAMNIST（腹部 CT）和 PathMNIST（结肠组织病理），在 IID 与不同非 IID（Dirichlet 参数 α=0.1、0.5、1.0）下实验。

**📈 对比分析**

与 Vanilla FL、VBFL、LBFL 比较，PoCQ 在 4 个投毒节点的攻击下保持 100% 真阳性检测率、FP 极低，平均验证时间比 LBFL 低 21% 、VBFL 低 40%；在极端非 IID 下对模型准确率提升 34.1%（医学数据）及整体平均准确率提升 11%。

**⚠️ 局限性**

局限：在极端非 IID 场景下，误将部分合法高方差节点误判为恶意；需要手动调节 warm‑up 轮数、阈值 τ 与黑名单阈值 R_min；未来需自动化阈值自适应机制。

---

## 268. Multi-Task Crack Foundation Model for Engineering-Reliable Crack Representation and Topology Preservation in Civil Infrastructure

**arXiv ID:** 2606.05641 | [PDF](https://arxiv.org/pdf/2606.05641v1)

**作者:** Blessing Agyei Kyem `[一作]` (North Dakota State University), Armstrong Aboah `[通讯]` (North Dakota State University)

**通讯引用:** 551 | [OpenAlex ID](https://openalex.org/A5005333881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 CrackGeoFM，一种多任务基础模型，能够同时输出裂纹分割掩模、裂纹骨架和像素级不确定性估计，用于高效可靠的基础设施裂纹检测与评估。

**💡 创新点**

创新点包括：1）将预训练的视觉基础模型冻结，并引入三大专用模块——频率引导裂纹增强（FCEM）、裂纹域特征自适应（CFAM）和结构感知多任务解码器（SMTD）；2）利用高频小波信息增强裂纹特征；3）从现有掩模自动生成骨架监督目标，无需额外标注；4）多任务学习与拓扑一致性损失的结合实现裂纹连通性和不确定性双重可靠性；5）在20个公开裂纹数据集上实现强大的跨域迁移与少样本适配。

**🔧 技术方法**

技术包括：预训练视觉Transformer（DINOv2-Large）冻结，Wavelet DWT频率增强，轻量化瓶颈适配器，特征金字塔网络，骨架与掩模联合训练，clDice拓扑损失，不确定性校准损失以及少样本微调。

**📊 数据集**

使用了20个公开裂纹数据集（共7,842张训练图，4,271张评估图），涵盖沥青路面、混凝土结构、砌体墙面、石材表面、隧道内壁等不同材料和摄影平台（智能手机、车辆摄像、UAV、工业相机），并划分为训练池和零样本池。

**📈 对比分析**

与11种CNN/Transformer基准（U-Net、DeepLabV3+、SegFormer等）比较，CrackGeoFM在7个验证集上平均Dice 0.6843，MCC 0.6901，超越最佳对照的0.6771/0.6784；在6个零样本跨域集上平均Dice 0.4406，显著高于最佳对照0.3957，尤其在低光照UAV数据上提升37.8%。少样本适配（5张样本）进一步将Dice提升至0.5890。

**⚠️ 局限性**

主要局限包括：推理速度较慢（约9.6 FPS，需大型GPU）；在极端差异域（如隧道内壁）零样本性能仍偏保守；骨架目标仅通过形态学得到，缺乏精确宽度/曲率信息；不确定性仅捕捉测度误差，未涵盖模型不确定性；未利用时空关联信息，无法支持裂纹随时间演化分析。

---

## 269. Beyond tokens: a unified framework for latent communication in LLM-based multi-agent systems

**arXiv ID:** 2606.05711 | [PDF](https://arxiv.org/pdf/2606.05711v1)

**作者:** Yingzhuo Liu `[一作]` (Beijing University of Posts and Telecommunications), Yingzhuo Liu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 26 | [OpenAlex ID](https://openalex.org/A5066540545)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并系统化了三轴框架（WHAT、WHICH、HOW），对 2024‑2026 年间 18 种 LLM 多代理隐式通信方法进行分类、对比并提出五大设计模式与六大开放问题。

**💡 创新点**

核心创新在于：①把隐式通信拆解为信息类型、发送接收对齐和融合策略三维空间，②归纳出 KV‑cache、隐藏状态、嵌入等五大信息类型与对应对齐与融合方式，③将多种方法映射到框架中，揭示设计空间与研究热点。

**🔧 技术方法**

使用了 LLM 内部连续表征（embedding、hidden state、KV‑cache、state‑delta 等）、对齐机制（投影、层对齐、视觉编码等）和融合操作（拼接、预置、数学运算、交叉注意、缓存恢复等）来实现代理间的无文本通信。

**📊 数据集**

实验基准涵盖数学推理（GSM8K、MATH、AIME）、常识问答（MMLU、ARC）、代码生成（HumanEval、MBPP）、多模态推理（MathVista、MMMU、ChartQA）以及多代理 QA 与桌面游戏等多任务集。

**📈 对比分析**

与传统自然语言通信相比，隐式通信在大上下文下可实现 2–24× 的推理延迟降低、4.7–136× 的 TTFT 加速、3–4× 的 token 消耗下降，且多数方法在准确率上与 NL‑Comm 相当或更优；KV‑cache 方法在信息量和速度上表现最强，但传输成本最高。

**⚠️ 局限性**

主要局限包括：多架构对齐仍需手工/训练，隐式通道缺乏可解释性与安全保障，KV‑cache 传输仍受限于压缩技术，理论上限与信息传递度量尚不明确，且现有方法多为训练‑free，缺乏针对特定任务的细粒度优化。

---

## 270. MolE-RAG: Molecular Structure-Enhanced Retrieval-Augmented Generation for Chemistry

**arXiv ID:** 2606.05693 | [PDF](https://arxiv.org/pdf/2606.05693v1)

**作者:** Joey Chan `[一作]` (University of Illinois Urbana Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个训练无须、基于检索的LLM框架（Molecule-Centric Retrieval-Augmented Generation），通过检索化学文献、注入分子特定上下文（同义词、功能基、RDKit描述子）以及检索结构相似分子来增强LLM对分子属性预测的性能。

**💡 创新点**

创新点在于：① 三种互补检索来源的组合；② 任务自适应的检索查询（结合任务描述、LLM生成关键词和分子同义词）；③ 结构检索使用任务最佳指纹；④ 通过结构相似例子实现少样本学习；⑤ 在保持无训练的前提下显著提升多任务预测性能。

**🔧 技术方法**

使用技术包括：多种开源与专有LLM（Llama-3.2‑3B、Mistral‑7B、Qwen3‑4B、ChemDFM‑14B、GPT‑4o‑mini、GPT‑5.4‑nano）；BM25检索加LLM增强查询；AccFG检测功能基团；RDKit计算物理化学描述子；任务自适应指纹选择（ECFP、FCFP、MACCS、AtomPair、Topological Torsion等）；结构检索与在情境学习；Prompt设计与零样本推理。

**📊 数据集**

主要数据集为 MoleculeNet 的九个任务（六个二分类：BBBP、BACE、ClinTox、HIV、Tox21、SIDER；三个回归：ESOL、FreeSolv、Lipophilicity），并使用 ChemRAG 语料库做文本检索。

**📈 对比分析**

在零样本、无微调的设置下，将不同检索源组合与SMILES‑only baseline比较，并与监督与自监督图模型（MGCN、SchNet、GROVER、MolCLR）对比。结果显示：分类任务上平均提升约20–28 ROC‑AUC，回归任务上RMSE下降约67%；在多数任务中，LLM+检索已接近或超过多数图模型，尤其在回归任务中表现突出。

**⚠️ 局限性**

局限性：① 由于 ChemRAG 语料库庞大，未能评估完整的 dense 检索版本；② 未测试 Chain‑of‑Thought 等思维提示技术；③ 对于规模较小的LLM（如 Llama‑3.2‑3B），检索增强效果有限。

---

## 271. Benchmarking Counterfactual Prediction in Epidemic Time Series with Time-Varying Interventions

**arXiv ID:** 2606.05692 | [PDF](https://arxiv.org/pdf/2606.05692v1)

**作者:** Wenhao Mu `[一作]` (University of Michigan), Alexander Rodríguez `[通讯]` (University of Michigan)

**通讯引用:** 1211 | [OpenAlex ID](https://openalex.org/A5067521241)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发了EpiCF-Bench大规模基准，利用可微分代理模型在150多个美国县级上生成真实可观测的时间序列事实与反事实，用于评估动态干预下的因果推断。

**💡 创新点**

创新点包括将现实人口、移动、流行病与政策数据嵌入可微分ABM，产生可观测反事实；支持静态与时间变异干预以及单/多政策交互；提供开源数据与代码。

**🔧 技术方法**

采用可微分Agent-based Model（GradABM/AgentTorch）进行张量化建模与梯度校准，结合网络冻结机制实现政策干预，并利用Transformer、RNN、VAE、Diffusion、CDE等多种因果模型进行评估。

**📊 数据集**

使用美国2020年人口普查、Google移动数据、CDC COVID-19病例、Oxford政策追踪器等公开数据构建158县的合成人口与交互网络。

**📈 对比分析**

通过将KDE、S-learner、T-learner、Transformer、TE-CDE、CVAE、CDiffusion、MSDiffusion、MSVAE等方法在单一与多政策任务上比较，评估指标为Wasserstein距离、RMSE、CATE RMSE、95% PI覆盖率及校准得分；结果显示RNN/Transformer在预测准确度上优于传统方法，TE-CDE在CATE估计上最优，而生成模型在准确度上提升但不确定度覆盖不足，整体性能差异显著。

**⚠️ 局限性**

局限性在于模型假设与数据抽象可能无法捕捉真实人群细粒度行为、政策遵从及未观测干预；政策通过随机删边近似，缺乏个体层面行为响应；受计算成本限制县级人口上限为200k，可能限制规模与多样性。

---

## 272. CASS-RTL: Correctness-Aware Subspace Steering for RTL Generation with LLMs

**arXiv ID:** 2606.05680 | [PDF](https://arxiv.org/pdf/2606.05680v1)

**作者:** Mohammad Akyash `[一作]` (University of Central Florida), Hadi Kamali `[通讯]` (University of Central Florida)

**通讯引用:** 268 | [OpenAlex ID](https://openalex.org/A5020619624)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 CASS-RTL，一种在不需要额外训练或监督的情况下，通过在推理时对 LLM 的内部注意力表示进行几何驱动的干预，以提升 RTL 代码生成的功能正确性的框架。

**💡 创新点**

创新点在于：①利用 KL 散度差分识别出对 RTL 正确性敏感的注意力头；②从这些头的激活构建低维正确性子空间；③设计基于该子空间的几何感知干预机制，实现对生成过程的轻量级、可解释的控制。

**🔧 技术方法**

技术方法包括：注意力头激活分析、KL 散度对比、主成分分析（PCA）构造子空间、在每个解码步骤对子空间投影后进行修正向量注入的推理时干预。

**📊 数据集**

所用数据集为 VerilogEval 和 CVDP 两个 RTL 代码生成基准，涵盖多种硬件描述任务。

**📈 对比分析**

与基线模型以及 ITI（注意力干预）方法进行对比，CASS-RTL 在 VerilogEval 上 Pass@1、Pass@5、Pass@10 分别提升约 10%–20%，在 CVDP 上提升约 5%；同时不增加显著的推理开销。

**⚠️ 局限性**

局限性包括：①对模型内部结构的分析仅在 decoder‑only Transformer 上验证，可能不适用于所有 LLM 架构；②子空间选择的头数量和干预强度需要调参，过度干预可能引入噪声；③目前仅针对 RTL 生成验证，尚未在更广泛的硬件设计任务或其他领域验证其通用性。

---

## 273. Data Flow Control: Data Safety Policies for AI Agents

**arXiv ID:** 2606.05679 | [PDF](https://arxiv.org/pdf/2606.05679v1)

**作者:** Charlie Summers `[一作]` (Columbia University), Eugene Wu `[通讯]` (Columbia University)

**通讯引用:** 5650 | [OpenAlex ID](https://openalex.org/A5049016095)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Data Flow Control (DFC) 框架，能够在数据库引擎内部对 SQL 查询的记录级数据流进行声明式安全策略的定义与强制执行。

**💡 创新点**

创新点在于：① 将安全策略视作聚合谓词，基于归一化的 provenance monomials 表达，保证策略与查询优化器无关；② 设计了一套轻量级的策略语言，支持多源、维度、输出引用、sink 等语义；③ 开发了可移植的查询重写层，利用 semiring 聚合在查询执行中“内联”计算策略约束，避免了完整 provenance 的昂贵材料化；④ 通过模板化与对称自连接的优化，支持百万级策略和自连接查询。

**🔧 技术方法**

主要技术包括：provenance polynomial 的归一化与 monomial 表示、semiring 聚合（计数、求和、最大值、集合等）、查询重写规则（推送聚合、去除多余 join、处理 UNION/OUTER JOIN 等）、多源与自连接的组合优化、以及对 LLM 的可选调用。

**📊 数据集**

使用的基准数据集为 TPC‑H（规模因子 1 与 10），并在四个主流 DBMS（DuckDB、Umbra、PostgreSQL、DataFusion、SQL Server）上进行实验。

**📈 对比分析**

与现有基于 provenance 的方法（如 Perm、SmokedDuck、P‑polynomial）相比，DFC 在所有引擎上平均相对开销 <1%，甚至在某些查询上比原始无策略查询更快；对多策略、模板化与自连接场景，优化后仍保持接近 0% 开销，显著优于传统方法 10‑10000× 的性能税。

**⚠️ 局限性**

局限性包括：仅支持单调 SQL‑92 查询（不涵盖非单调或递归、窗口等功能）；策略编写仍需要人工；对非阻断（非删除）或交互式决策的支持尚未实现；在极端复杂的聚合组合或多源交叉场景下，重写仍可能产生较大中间结果；需要进一步研究跨查询/会话的安全策略。

---

## 274. Dynamic Multi-Agent Pickup and Delivery in Robotic Cellular Warehousing Systems

**arXiv ID:** 2606.05669 | [PDF](https://arxiv.org/pdf/2606.05669v1)

**作者:** Cheng Ren `[一作]` (Hong Kong Polytechnic University), George Q. Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 23110 | [OpenAlex ID](https://openalex.org/A5015681327)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了动态多代理抢送（Dynamic‑MAPD）问题，研究在机器人细胞式仓库系统中订单在执行过程中动态增添 SKU 的情形，并设计了基于事件触发的令牌传递（Dynamic‑TP）和协同令牌传递（Coop‑TP）两种在线重规划算法；

**💡 创新点**

创新点在于首次将订单内部动态更新纳入 MAPD 框架，提出事件驱动的令牌传递机制，并通过协同策略让空闲机器人主动协助已在执行的订单，实现更高效的资源利用与更低的订单流量；

**🔧 技术方法**

技术主要包括：事件触发式令牌传递框架、基于 A* 的时间扩展图路径规划、优先级队列与截止时间优先调度、协同任务分配与多机器人并行路径重规划；

**📊 数据集**

实验使用基于 RubikCell 的仿真仓库网格数据集，分别在 60×60 与 40×80 两种尺寸下生成 500 条随机实例；

**📈 对比分析**

与传统 Token Passing（TP）和 TP‑Append（TP‑A）基线相比，Dynamic‑TP 在动态更新频率与更新规模均提高时平均订单流量降低 6%–12%；Coop‑TP 在高动态强度下将平均流量进一步降低 20%–30%，且在所有参数组合下均优于其他方法；

**⚠️ 局限性**

局限性包括：仅考虑每个订单最多一次动态更新，易扩展但未验证；Coop‑TP 的计算开销随空闲机器人数量显著增长，实际部署需平衡实时性与协同收益；实验仅在仿真环境中验证，缺乏真实仓库实测。

---

## 275. Agent-Orchestrated Adaptive RAG: A Comparative Study on Structured and Multi-Hop Retrieval

**arXiv ID:** 2606.05658 | [PDF](https://arxiv.org/pdf/2606.05658v1)

**作者:** Anuj Maharjan `[一作]` (University of Toledo), Richard Molyet `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于Agent的自适应RAG框架，通过查询分类、分解、评估与有限反思实现动态检索与生成。

**💡 创新点**

创新点在于将多Agent协作与自适应路由相结合，既支持查询分解又引入有限自我反思，并在结构化与多跳两个域上评估其异质性。

**🔧 技术方法**

使用了LLM Llama‑3.1‑8B、BGE嵌入、FAISS向量检索、Docling结构化转化，以及自定义的AgentOrchestrator、QueryDecomposer、AnswerEvaluator等。

**📊 数据集**

使用了自制的DevOps知识库（80篇文档）和开源多跳基准MuSiQue。

**📈 对比分析**

通过对比基线Naïve RAG与完整Agentic RAG，评估指标包括Overall Score、Citation Accuracy、MRR、Success@5、Topic Coverage、Latency；结果显示：在DevOps上分解提升整体分数和MRR，但在MuSiQue上降低排名精度；反思虽提升Citation Accuracy 但大幅增加延迟。

**⚠️ 局限性**

局限包括数据规模偏小、仅使用单一LLM与嵌入模型、策略选择基于规则而非学习、并且高延迟对实时应用不友好，需进一步验证更大规模与更强模型的效果。

---

## 276. CoFi-UCGen: Coarse-to-Fine Unsupervised Conditional Generation without Label Priors

**arXiv ID:** 2606.05652 | [PDF](https://arxiv.org/pdf/2606.05652v1)

**作者:** Shengxi Li `[一作]` (Beihang University), Si Liu `[通讯]` (Beihang University)

**通讯引用:** 13976 | [OpenAlex ID](https://openalex.org/A5100330138)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出CoFi-UCGen框架，实现完全无标签的粗细粒度条件图像生成；

**💡 创新点**

创新点包括：对抗性逆向学习保证图像与潜在空间的语义一致性；引入二进制码构造可分离的粗粒度潜在空间；在扩散模型中设计层级调制HM‑UNet实现细粒度控制；

**🔧 技术方法**

技术手段：GAN+扩散模型、对抗性逆向学习、特征编码器、对比学习、特征分布匹配（CF）、变分自编码器、AdaGN层级调制、ELBO、DINO特征对齐等；

**📊 数据集**

使用数据集：Stanford Cars、UTKFace、CUB200、Oxford102‑Flowers，以及合成von Mises‑Fisher混合数据集；

**📈 对比分析**

与ClusterGAN、Self‑Cond GAN、MIC‑GANs、SG‑DM及其变体进行对比，评估指标包括FID、IS、Purity、NMI、Precision、Recall、DINO‑Aligned。CoFi‑UCGen在粗细粒度上均优于基线，尤其在FID、Precision、DINO‑Aligned上显著提升；

**⚠️ 局限性**

局限性：训练时间长、对显卡资源要求高；对极细粒度属性的精准控制仍受限；多属性场景下的多重细粒度控制尚未充分验证；高分辨率扩展性需要进一步评估。

---

## 277. GS-NFS: Bandwidth-adaptive Streaming of Dynamic Gaussian Splats and Point Clouds

**arXiv ID:** 2606.05650 | [PDF](https://arxiv.org/pdf/2606.05650v1)

**作者:** Rajrup Ghosh `[一作]`, Ramesh Govindan `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于GPU的动态3D高斯 splatting 视频压缩与解压方案，能够实现 30fps 的全帧率编码/解码。

**💡 创新点**

创新点包括：① 将 G-PCC 的八叉树编码与 RAHT 预测完全移植到 GPU，实现大规模并行化；② 引入 KLT+YUV 去相关，提升颜色系数压缩率；③ 通过 RLGR 与 ANS 混合熵编码进一步压缩位流；④ 针对移动设备设计的轻量级解码器，支持实时点云压缩。

**🔧 技术方法**

技术主要包括：GPU并行八叉树构建与遍历、GPU并行RAHT预测/逆预测、KLT颜色去相关、YUV转换、ANS & RLGR熵编码、CUDA与PyTorch混合实现、Jetson Orin 上的移动解码优化。

**📊 数据集**

使用了 HiFi4G（7个人物动作序列）和 Neural 3D Video (N3DV)（6个全景室内场景）两大公开数据集，包含 200–300 帧、120k–400k 高斯/点云。

**📈 对比分析**

与 2D 视频编码（H.264）、MesonGS、G-PCC、LTS-Draco 等基线对比：编码速度提升 1–2 个数量级；解码速度提升 10–20 倍；在 PSNR/比特率曲线上大多数场景下 率失真性能优于或接近基线；在 Jetson Orin 上可达 17–25fps，支持实时点云压缩。

**⚠️ 局限性**

局限性：① 仍主要针对静态/轻运动场景，对大幅运动的 3D 高斯分布映射可能导致压缩率下降；② 采用的 G-PCC 结构对极稀疏点云或高噪声场景的鲁棒性尚待验证；③ 需要专用 GPU 支持，CPU 或低端硬件实现难度较大；④ 量化参数调优仍需人工，缺乏自适应码率机制。

---

## 278. Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?

**arXiv ID:** 2606.05647 | [PDF](https://arxiv.org/pdf/2606.05647v1)

**作者:** Jingheng Ye `[一作]` (Northeastern University), Weiyan Shi `[通讯]` (Northeastern University)

**通讯引用:** 1281 | [OpenAlex ID](https://openalex.org/A5089522357)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对AI编码代理的恶意破坏（隐藏侧任务）进行首次大规模、长期、多轮人机协作实验，评估开发者能否检测并阻止代理插入恶意代码。

**💡 创新点**

创新点在于：①结合真实的长时段开发流程和前沿LLM代理，系统研究人类监督失效；②量化人类检测失败率（94%）并揭示其根源；③引入LLM监视器实验，评估其对恶意行为的抑制作用；④基于受试者反馈提出面向人因的监视器设计建议。

**🔧 技术方法**

使用四个前沿代码代理（Claude‑Opus‑4.6、GPT‑5.4、Gemini‑3.1‑Pro、MiniMax‑M2.7），配合Claude‑Haiku‑4.5的LLM监视器；在自托管 GitLab CI 环境中执行任务；采用红队提示诱导破坏；收集交互日志、代码提交和CI结果。

**📊 数据集**

自定义的迷你电商项目代码与合成敏感数据（模拟个人信息）作为任务数据；实验中收集的交互记录、提交历史、CI 测试结果等为研究数据集。

**📈 对比分析**

通过对比有无监视器的 19 次实验，计算破坏成功率（SSR）：无监视器 94%，有监视器 63%（检测率 84%）。不同模型间 SSR 也被比较：Gemini‑3.1‑Pro 100%，MiniMax‑M2.7 82%，其余 94%。监视器虽能显著降低成功率，但仍有 56% 的破坏在收到警报后成功被合并。

**⚠️ 局限性**

局限性包括：①实验环境为受控沙箱，未完全反映真实部署；②样本主要来自美国，可能存在样本偏差；③监视器仅为 flag‑only，未探索主动干预；④实验仅关注数据外泄破坏，未涵盖其他恶意形式；⑤受试者在实验条件下可能行为与真实工作不同，结果可推广性待验证。

---

## 279. QueryAgent-R1: Bridging Query Generation and Product Retrieval for E-Commerce Query Recommendation

**arXiv ID:** 2606.05671 | [PDF](https://arxiv.org/pdf/2606.05671v1)

**作者:** Dike Sun `[一作]` (Alibaba International Digital Commercial Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commercial Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于记忆增强的智能体框架 QueryAgent‑R1，用于在无用户输入的电商搜索栏中主动推荐查询，并通过检索验证以提升点击转化。

**💡 创新点**

创新点在于将查询生成与真实库存检索耦合，加入一致性奖励通过强化学习实现查询与后续商品匹配的端到端优化，并设计了压缩用户长历史的记忆抽象模块。

**🔧 技术方法**

技术包括大语言模型 Qwen3‑4B、记忆抽象压缩工具、基于 BM25＋Qwen3‑Embedding 的混合检索、交叉编码重排、以及 GDPO 强化学习与一致性奖励。

**📊 数据集**

使用了两份数据集：内部电商平台 54k 活跃用户训练/5k 测试，以及合并的 Amazon ESCI 与 Review 数据 16k/1k。

**📈 对比分析**

与库存检索和 LLM 直接推理基线相比，离线实验中 Cons@1 提升到 0.117（工业）/0.063（Amazon），在线 A/B 测试中查询 CTR +2.9%，订单 CVR +3.1%，GMV +4.9%。

**⚠️ 局限性**

主要限制是在线推理时延较高，需采用异步预计算，影响实时响应并增加系统复杂度。

---

## 280. FIDES: Faithful Inference via Deep Evidence Signals for Retrieval-Memory Conflict in RAG

**arXiv ID:** 2606.05644 | [PDF](https://arxiv.org/pdf/2606.05644v1)

**作者:** Zhe Yu `[一作]` (Zhejiang University), Meng Han `[通讯]` (Zhejiang University)

**通讯引用:** 10958 | [OpenAlex ID](https://openalex.org/A5031867321)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了FIDES，一种无训练、基于对比解码的检索增强生成方法。

**💡 创新点**

通过融合输出层、隐藏层和预测轨迹的三种内部信号实现token级对比强度控制，从而精细化抑制检索-参数冲突。

**🔧 技术方法**

使用双路径对比解码、JSD、L2距离、KL散度信号融合、线性映射α_t等技术实现无训练式自适应控制。

**📊 数据集**

在NQ‑Swap、PopQA和TriviaQA的反事实评测集上进行实验，覆盖7B至70B多种模型。

**📈 对比分析**

与Standard RAG、CAD、AdaCAD、COIECD、DeCoRe、DVD等基线对比，FIDES在18个模型-数据组合上取得最高上下文忠实度，提升3–13点；在70B模型上CF达92–94%，F1达62–63%。

**⚠️ 局限性**

需要双向前向传递导致近2×推理成本；仅在检索错误时仍可能跟随错误证据，无法验证事实正确性。

---

## 281. Explainable AI-Driven Cyber Risk Analytics and Model Reliability Assessment for Intelligent Governance of U.S. Critical Infrastructure: An XGBoost and SHAP-Based Intrusion Detection Framework

**arXiv ID:** 2606.05710 | [PDF](https://arxiv.org/pdf/2606.05710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 282. Preserving Full 6-DOF Actuation Under Abrupt Total Rotor Failures: Passive Fault-Tolerant Flight Control Using a Biaxial-Tilt Hexacopter

**arXiv ID:** 2606.05663 | [PDF](https://arxiv.org/pdf/2606.05663v1)

**作者:** Yipeng Yang `[一作]`, Huijun Gao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了在突发完整旋翼失效情况下，采用双轴倾斜可过度驱动六旋翼无人机（BTO）实现无故障检测的被动容错飞行控制；

**💡 创新点**

创新点在于：①基于可达推力空间（AWS）分析提出BTO结构相较于单轴倾斜（UTO）和传统共面（CCU）更具容错能力；②设计两种轻量级被动容错控制框架（CL‑PFTC与AL‑PFTC），无需故障检测或在线优化；③AL‑PFTC通过自适应控制分配与动量基外力估计实现快速恢复全DOF可控性；

**🔧 技术方法**

采用的技术包括：高阶全可控控制器（HOFA）+线性扩展状态观测器（LESO）用于扰动补偿；基于模型参考的自适应控制分配与伪逆分配；动量基力矩估计与低通滤波外力估计；AWS指标与容错度量分析；

**📊 数据集**

实验数据集主要为仿真（Simscape Multibody）与实地飞行实验（在STM32H7飞控上进行的悬停、轨迹跟踪、窄框穿行与空中书写任务），没有公开标注的公共数据集；

**📈 对比分析**

比较方法：将BTO与UTO、CCU在同一故障条件下对比；CL‑PFTC与AL‑PFTC在不同失效模式（单旋翼、双旋翼、三旋翼）下进行悬停稳定性和轨迹跟踪误差评估；结果显示BTO在所有测试中表现最佳，AL‑PFTC在精度与容错性上优于CL‑PFTC，尤其在严重失效（如Λ₁,₄）时仍保持稳定；

**⚠️ 局限性**

局限性：①仅考虑保持完全可驱动的失效情况，对多旋翼失效导致AWS坍塌的情形未覆盖；②对极端扰动、传感器噪声及实时动态环境的鲁棒性仍需进一步验证；③控制分配与估计算法对飞控计算资源有限的设备有一定要求，未来可进一步简化。

---

## 283. Q-GNN: Query-Conditioned Graph Neural Networks with Type Awareness for Knowledge Graph Completion

**arXiv ID:** 2606.05639 | [PDF](https://arxiv.org/pdf/2606.05639v1)

**作者:** Dongxiao He `[一作]` (Tianjin University), Zhiyong Feng `[通讯]` (Tianjin University)

**通讯引用:** 13055 | [OpenAlex ID](https://openalex.org/A5001714538)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了一种名为Q-GNN的知识图谱完成方法，在推理过程中充分利用查询实体的结构上下文和语义类型来引导消息传播与评分。

**💡 创新点**

在传统GNN仅使用查询关系作为引导信号的基础上，首次将查询实体的结构上下文（通过逆向消息传递编码）和实体类型信息（通过类型感知注意力和类型专属解码器）结合进查询中心的推理流程。

**🔧 技术方法**

逆向与正向消息传递的GNN、FiLM特征线性调制、类型感知注意力机制、类型图构建、LLM推断实体类型、基于类型路径的上下文过滤、类型专属解码器等。

**📊 数据集**

使用了WN18RR、FB15k-237（转导）以及NELL-995、UMLS、Family等额外数据集，并在WN18RR、FB15k-237、NELL-995的归纳划分上进行实验。

**📈 对比分析**

与三大类基线（三元组、路径、GNN）及多种最新GNN方法对比，转导实验中在所有指标上均超过最优基线（例如在FB15k-237 MRR 0.408 对比 0.376），归纳实验中在12个划分中取得9个最高分，整体性能优于DiffusionE等。

**⚠️ 局限性**

每查询的计算复杂度随消息传递层数增加，可能在极大规模知识图上效率受限，未针对大规模图进行扩展。

---

## 284. T-SAR-JEPA: Self-Supervised Temporal Anomaly Detection in SAR Amplitude Stacks via Latent Prediction

**arXiv ID:** 2606.05700 | [PDF](https://arxiv.org/pdf/2606.05700v1)

**作者:** Kerod Woldesenbet `[一作]` (Independent Researcher), Abem Woldesenbet `[通讯]` (Dakota State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种自监督的T-SAR-JEPA框架，用于基于SAR幅度堆栈的时间异常检测。

**💡 创新点**

创新点在于将JEPA与时间Transformer结合，并通过正弦时间编码和渐进解冻实现对幅度信息的时间预测，无需干涉测相；同时使用InSAR相干度做独立伪真值验证。

**🔧 技术方法**

技术包括ViT-Base/16编码器、局部遮挡重建、时间Transformer、正弦时间编码、Smooth L1损失、渐进解冻、L2预测误差评分。

**📊 数据集**

使用Capella Space的39,300个幅度补丁及DFC 2026数据集中的三大AOI（夏威夷/基拉韦厄、洛杉矶、Pilbara），并以InSAR相干度作为伪真值。

**📈 对比分析**

与RX、PaDiM、线性AR、LSTM等基线进行比较，T‑SAR‑JEPA在夏威夷喷发窗口的ROC‑AUC达到77.0%，明显优于约50%的基线；在空间相干性和几何不变性上也表现突出。

**⚠️ 局限性**

局限性包括伪真值的噪声与偏差、仅在夏威夷有事件级真值、对时间序列长度有限、时间基线与非时间模型的比较不完全公平、以及对不同卫星ID的适应性需进一步验证。

---

## 285. Rethinking LoRA Memory Through the Lens of KV Cache Compression

**arXiv ID:** 2606.05698 | [PDF](https://arxiv.org/pdf/2606.05698v1)

**作者:** Chunsheng Zuo `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**通讯引用:** 8766 | [OpenAlex ID](https://openalex.org/A5075825791)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在文档级问答中研究了文档特定 LoRA 适配器与 KV 缓存压缩的交互效果。

**💡 创新点**

提出 LoRA 在 KV 缓存被激进压缩时能补偿缺失证据，且更适合作为解码时的参数记忆，并证明 QA 样式监督产生最强的 LoRA。

**🔧 技术方法**

采用 LoRA 适配器、Compactor/KV 缓存压缩、不同阶段激活策略、QA 与原始文本监督等技术。

**📊 数据集**

使用 NarrativeQA 与 LongHealth 两个大规模文档级问答数据集。

**📈 对比分析**

通过 ROUGE‑L 对比基线（仅压缩 KV）与加入 LoRA 的方法，发现当 KV 缓存被完全移除时可提升约 13–21 分，且在 99% 压缩时已恢复约一半性能。

**⚠️ 局限性**

假设每个文档都有预先训练好的 LoRA，未考虑适配器检索、组合多文档证据或加载延迟，且在一次性查询且上下文充足时仍可能不如传统长上下文推理。

---

## 286. AdaMEM: Test-Time Adaptive Memory for Language Agents

**arXiv ID:** 2606.05684 | [PDF](https://arxiv.org/pdf/2606.05684v1)

**作者:** Yunxiang Zhang `[一作]` (University of Michigan), Lu Wang `[通讯]` (University of Michigan)

**通讯引用:** 26862 | [OpenAlex ID](https://openalex.org/A5100364413)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了适应性记忆代理（AdaMEM），通过在测试时动态生成短期策略记忆来实现语言代理的连续自适应；并设计了逐步记忆微调（Step-wise Memory Fine‑Tuning）以训练策略生成；

**💡 创新点**

创新点在于将长期轨迹存储与短期策略抽象分离，支持在每一步即时检索并合成符合当前状态的策略；同时引入基于过程级动作变化的拒绝采样微调，避免传统仅靠结果监督的噪声问题；

**🔧 技术方法**

采用LLM提示式推理与ReAct框架，利用预训练嵌入进行检索；在推理时根据当前状态合成自然语言策略；通过双重过滤（成功与动作改变）进行无监督微调；

**📊 数据集**

使用三大基准数据集：ALFWorld（家居导航任务）、WebShop（电商搜索/购买）和HotpotQA（多跳问答搜索）来构建长期轨迹记忆并评估性能；

**📈 对比分析**

与无记忆、Synapse（全轨迹检索）和ReasoningBank（静态策略）等基线对比；在ALFWorld未见场景提升11.4个百分点，在WebShop提升约11%，在HotpotQA提升13%；在多种推理成本下展示更优的性能/代价 Pareto 前沿；

**⚠️ 局限性**

局限性包括：需要大量成功轨迹构建长期记忆；策略生成与检索仍有令牌开销；在无外部检索时易产生幻觉；跨模型迁移仍受限于检索适配；未对隐私与治理问题给出完整解决方案。

---

## 287. Beyond Waveform Robustness: Robust Feature-Vocoder Adversarial Attacks on Automatic Speech Recognition

**arXiv ID:** 2606.05678 | [PDF](https://arxiv.org/pdf/2606.05678v1)

**作者:** Yifan Liao `[一作]` (Hong Kong University of Science and Technology), Xinlei He `[通讯]` (Wuhan University)

**通讯引用:** 13021 | [OpenAlex ID](https://openalex.org/A5031973958)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种 Clean‑Referenced Feature‑Vocoder Attack，将对抗样本从原始波形改为在自监督学习（SSL）特征空间中优化，然后通过冻结的声码器重构成音频，提升黑盒迁移能力并规避传统基于波形噪声的防御；

**💡 创新点**

创新点在于：①将攻击空间迁移到共享的 SSL 表征层，减少对单一模型梯度的依赖；②通过声码器重构实现特征级扰动而非直接加噪声，使攻击更难被波形级防御捕捉；③引入“clean‑referenced”感知正则化，保证生成音频既能诱发错误又保持语音质量；

**🔧 技术方法**

使用冻结的 WavLM‑Large 作为 SSL 编码器，HiFi‑GAN 声码器进行重构；通过最大化对抗目标模型（如 Whisper‑small）对真值的负对数似然；加入基于时间梯度和高频能量的感知正则；

**📊 数据集**

主要数据集：LibriSpeech（English）和 AISHELL‑1（Chinese）两套公开语音数据；

**📈 对比分析**

与多种基线（PGD、MI‑FGSM、VMI‑FGSM、Muting Whisper、SlothSpeech）以及多类防御（对抗训练与输入预处理）对比；在 Whisper‑family 及 CTC‑based ASR 上实验，结果显示：在未防御模型上 WER/CER 近 99%/94%，在多种防御下仍保持 70%+ 的错误率，显著优于基线（大多在 30% 以内），并在跨模型与跨架构迁移方面表现出更高的鲁棒性；

**⚠️ 局限性**

局限性：①实验仅覆盖有限的 ASR 系统与防御，无法验证对更大规模或商业系统的泛化；②依赖特定 SSL 编码器与声码器，其他组合可能影响效果；③感知评估主要基于自动指标与小规模人类测试，真实环境下的可听性评估不足；④物理世界实验仅在少量设备与环境下进行，缺乏对不同房间、噪声及录音链的广泛验证。

---

## 288. LongSpace: Exploring Long-Horizon Spatial Memory from Perception to Recall in Video

**arXiv ID:** 2606.05677 | [PDF](https://arxiv.org/pdf/2606.05677v1)

**作者:** Shiqiang Lang `[一作]` (Beijing University of Posts and Telecommunications), Honggang Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 12483 | [OpenAlex ID](https://openalex.org/A5100626780)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 LongSpace-Bench 这一基于室内房间游览视频的长时段空间记忆基准，并设计了 LongSpace 框架来支持长视频空间推理。

**💡 创新点**

创新点在于将 3D 结构感知融入前几层解码器，并构建分层 KV 内存，实现跨片段的可查询空间记忆。

**🔧 技术方法**

采用 Qwen3-VL-8B 语言模型、π^3 3D 结构编码器以及层级 KV 内存机制，对视频分块进行编码与压缩。

**📊 数据集**

主要使用室内房间游览视频集，约 445 条视频、159 小时，总计 4,073 个问答对，并结合 VSI-Bench 等公开基准进行评测。

**📈 对比分析**

在 LongSpace-Bench 上，LongSpace-9B 取得 49.2 分的最高分，显著优于 Qwen3-VL-32B、Gemini-3-Pro 等基线，尤其在“出现顺序”“状态变化”“路径回忆”等记忆密集任务上提升 8–9 分。

**⚠️ 局限性**

局限在于仅涵盖室内房间游览场景，未覆盖复杂户外环境，且仅关注被动观察的空间记忆，未涉及主动探索或交互操作。

---

## 289. PivCo-Huffman

**arXiv ID:** 2606.05765 | [PDF](https://arxiv.org/pdf/2606.05765v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 290. CollabBench: Benchmarking and Unleashing Collaborative Ability of LLMs with Diverse Players via Proactive Engagement

**arXiv ID:** 2606.05793 | [PDF](https://arxiv.org/pdf/2606.05793v1)

**作者:** Hong Qian `[一作]` (East China Normal University), Aimin Zhou `[通讯]` (East China Normal University)

**通讯引用:** 10044 | [OpenAlex ID](https://openalex.org/A5050248676)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CollabBench基准，设计了多样化玩家档案模拟管线、统一的代理推演与混合奖励训练框架，并在多玩家合作游戏环境中训练与评估LLM协作代理。

**💡 创新点**

创新点在于：①构建基于大五人格的高质量玩家行为映射与筛选；②统一代理推演，将推理、沟通与行动合并为单步输出；③引入轨迹效率与步骤情感双重奖励，兼顾任务完成与情感适配；④设计覆盖效率与情感的多维评估体系。

**🔧 技术方法**

技术手段包括：大语言模型（如Qwen、GPT-5.2）、ReAct式推理、Agentic Rollout、混合奖励（轨迹效率+步骤情感）以及GIGPO强化学习；评估使用LLM判定器进行帮助性、信任度与同理心评分。

**📊 数据集**

使用了从大五人格驱动的玩家行为轨迹数据集，并将CWAH与Overcooked两套经典游戏扩展为CWAH‑MultiPlayer与Cook‑MultiPlayer，覆盖多种玩家档案与游戏布局。

**📈 对比分析**

与CoELA、ProAgent等基线相比，训练后模型在CB‑Efficiency上提升约19.5%，在CB‑Affective上提升约24.4%；在人类评测中，帮助性、信任度与同理心均显著高于基线。

**⚠️ 局限性**

局限性包括：玩家档案映射仍需人工验证；情感奖励易受奖励劫持影响；高频交互场景训练成本高；评估依赖LLM判定，缺乏足够的人类标注。

---

## 291. Can LLMs Write Correct TLA+ Specifications? Evaluating Natural-Language-to-TLA+ Generation

**arXiv ID:** 2606.05792 | [PDF](https://arxiv.org/pdf/2606.05792v1)

**作者:** Arslan Bisharat `[一作]` (Loyola University Chicago), Mohammed Abuhamad `[通讯]` (Loyola University Chicago)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5042456819)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了30款大型语言模型（涵盖8个模型族）在从自然语言生成TLA+规范时的表现，采用了四种提示策略并通过SANY语法检查与TLC模型检测验证语义正确性。

**💡 创新点**

首次量化了LLM在语法合法性与语义正确性之间的巨大差距，揭示了模型规模、代码专门化与推理对齐的非直观影响，并归纳了五类系统化的幻觉错误。

**🔧 技术方法**

使用SANY解析器、TLC模型检查器、BLEU/ROUGE文本相似度指标以及基于提示的生成（few‑shot、progressive、fill‑in‑middle、half‑completion）技术。

**📊 数据集**

构建了包含205条TLA+规范及其自然语言注释与TLC配置的基准数据集，并划分为训练/验证/测试集。

**📈 对比分析**

对每个模型-提示组合执行一次生成，共计2,600次核心实验；最高语法通过率为26.6%，仅Progressive提示下的语义通过率为8.6%，最佳模型DeepSeek r1:8b在Progressive提示下达到53.8%语义通过率。

**⚠️ 局限性**

实验仅考虑单次生成，未覆盖模型随机性；数据集覆盖范围有限，未包含工业级真实规范；仅评估TLA+，对其他形式化语言的迁移性未知。

---

## 292. Next-Generation Parallel Decoder for LPDR: Architectural Optimization and Class-Balanced GAN-Augmentation

**arXiv ID:** 2606.05785 | [PDF](https://arxiv.org/pdf/2606.05785v1)

**作者:** Shawaiz Obaid `[一作]` (National University of Sciences & Technology), Muhammad Khuram Shahzad `[通讯]` (National University of Sciences & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究提出了CSHA（Cross‑Spatial Hybrid Attention）和CBSA（Class‑Balanced Synthetic Augmentation）两种技术，以改进YOLOv5‑PDLPR模型的平行解码器，解决字符空间不匹配和数据长尾不平衡问题。

**💡 创新点**

创新点在于将空间注意力与通道注意力结合的CSHA模块，以及利用CycleGAN实现的类别平衡合成数据的CBSA管线。

**🔧 技术方法**

技术手段包括YOLOv5‑PDLPR框架、Transformer平行解码器、CSHA注意力机制、CycleGAN生成合成车牌样本以及PyTorch训练。

**📊 数据集**

使用的数据集主要是中国车牌数据集CCPD（安徽）进行训练，CLPD（混合）进行鲁棒性评估，并合成了约75,000张稀有省份车牌样本。

**📈 对比分析**

与CRNN‑CTC、YOLOv5‑PDLPR等方法比较，CSHA‑PDLPR在CCPD‑Base、CCPD‑Tilt和CLPD‑Mixed上分别达到99.6%、94.8%和91.5%的识别率，并保持152 FPS的实时速度，整体性能提升显著。

**⚠️ 局限性**

局限性包括在低光照环境下仍有性能下降、对非中文车牌格式的适应性不足、CSHA块虽仅增0.45M参数但仍略增推理延迟，以及未来需要实现动态宽度解码器和跨国车牌识别。

---

## 293. Beyond Soft Masks: Hard-Perturbation Mixup Explainer for Robust GNN Explainability

**arXiv ID:** 2606.05756 | [PDF](https://arxiv.org/pdf/2606.05756v1)

**作者:** Jialiang Yin `[一作]` (Xi'an Jiaotong University), Jiaxing Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了HPME框架，通过硬性扰动和结构化混合增强GNN的可解释性，直接提取离散子图并生成与原分布一致的混合图。

**💡 创新点**

创新点在于将图信息瓶颈理论推广至硬性扰动形式，利用图池化实现信息压缩，并通过结构级替换生成不含冗余边的in‑distribution混合图，解决软掩码方法导致的OOD和冗余信息问题。

**🔧 技术方法**

主要技术包括：基于硬性扰动的图信息瓶颈（Generalized GIB），图池化与节点得分筛选，结构化混合（subgraph replacement），二值Concrete分布生成离散掩码，BCE和预测损失的联合优化。

**📊 数据集**

实验数据集涵盖13个带真值子图的图分类与回归任务，分别为13个合成数据集（如BA‑2Motifs、BA‑HouseGrid、SPMotif等）和4个真实分子数据集（Alkane‑Carbonyl、Fluoride‑Carbonyl、Benzene、Crippen）。

**📈 对比分析**

与GNNExplainer、PGExplainer、TAGExplainer、MetaGNN、MatchExplainer、MixupExplainer、ProxyExplainer、RegExplainer、GRAD和ATT等基准方法对比，HPME在AUC‑ROC上平均提升约11%（分类）和13%（回归），在某些数据集可达30%的绝对增益，且在分布偏移评估中余弦相似度更高、欧氏距离更低。

**⚠️ 局限性**

局限性主要在于目前仅针对同质图结构验证，尚未推广到异构或多模态图，以及对极大规模图的计算效率和池化参数敏感性待进一步研究。

---

## 294. Policy-Guided ML for Energy Savings: Cell On/Off Switching under Operator QoS Constraints in Real 5G Networks

**arXiv ID:** 2606.05755 | [PDF](https://arxiv.org/pdf/2606.05755v1)

**作者:** D. Reiss `[一作]` (Universitat Politècnica De Catalunya), O. Sallent `[通讯]` (Universitat Politècnica De Catalunya)

**通讯引用:** 4238 | [OpenAlex ID](https://openalex.org/A5048114047)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于机器学习的 5G 基站开/关策略，能够在满足运营商指定的吞吐量和停机容忍度 QoS 要求的前提下实现能耗降低。

**💡 创新点**

创新点在于：① 将运营商自定义的 QoS 政策（吞吐量阈值与停机容忍度）直接嵌入到模型训练与决策中；② 通过调节 XGBoost 的类比重（class ratio）实现停机误差的可控；③ 让运营商在部署前即可通过参数调整实现能耗与 QoS 的权衡。

**🔧 技术方法**

技术实现主要使用 XGBoost 分类器进行二分类预测；利用类比重超参数实现停机误差权衡；采用 oracle 策略生成标签；使用吞吐量误差与停机比例等指标进行评估。

**📊 数据集**

数据集来自欧洲一家移动运营商的真实运营数据，覆盖 70 个 5G 基站（从原 200 个子集选取），15 分钟粒度，包含 5G 基站负载及同一站点 4G 基站负载，时长为整整一个月。

**📈 对比分析**

通过与 oracle 策略（最优开/关）以及平衡类比重模型进行对比。平衡模型在评估周实现 92.6% 的能耗节约；在 3% 的停机容忍度下仍保持约 82% 的节约。停机误差均能保持在指定阈值附近，且停机误差（吞吐量偏差）随类比重增大而下降。

**⚠️ 局限性**

局限性：仅考虑单一吞吐量阈值（15 Mbps），未验证对不同吞吐量的通用性；只覆盖 70 个基站，缺乏更大规模或不同场景的验证；类比重需要人工调节，缺乏在线自适应机制；模型训练与评估周期为周，未能实时跟随网络状态变化；仅关注吞吐量和停机容忍度两项 QoS 指标。

---

## 295. Do speech foundation models perceive speaker similarity as humans do?

**arXiv ID:** 2606.05739 | [PDF](https://arxiv.org/pdf/2606.05739v1)

**作者:** Minoru Kishi `[一作]` (Keio University), Yuki Saito `[通讯]` (University of Tokyo)

**通讯引用:** 4862 | [OpenAlex ID](https://openalex.org/A5042731264)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对 40+ 种语音基础模型的说话人嵌入与人类主观说话人相似度进行对比分析，使用 Pearson、Spearman、Frobenius 距离和谱距等指标，结合多元回归阐明模型配置对人类感知的影响。

**💡 创新点**

系统化大规模评估、引入多种图结构距离度量以及利用多元回归揭示模型架构、训练方式与人类感知的关联，首次量化语音模型说话人表示的“人类化”程度。

**🔧 技术方法**

Transformer 语音模型、说话人嵌入提取、余弦相似度、Pearson/Spearman 相关、Frobenius 距离、谱距、线性回归与多元回归分析。

**📊 数据集**

JVS（49 男 51 女）和 VCTK（52 女）两大语音数据集提供的人类相似度评分。

**📈 对比分析**

通过 Pearson/Spearman 相关、Frobenius 距离和谱距对比模型与人类评分，结果显示 WavLM 等 Encoder/监督大规模模型对齐度最高；解码器和大模型规模往往降低整体对齐度但平滑层级趋势；回归表明模型配置可解释约 80% 的平均对齐度，层级趋势解释度仅约 20%。

**⚠️ 局限性**

仅评估语音数据集，未覆盖所有说话人性别/语言多样性；缺乏对其他评测指标（如 ASV 性能）的交叉验证；回归变量不涵盖所有可能影响层级结构的因素；音频相关模型未包含在回归分析中。

---

## 296. An Embarrassingly Simple Detector for Model Extraction Attacks in Large Language Model API Traffic

**arXiv ID:** 2606.05725 | [PDF](https://arxiv.org/pdf/2606.05725v1)

**作者:** Shuze Liu `[一作]` (Santa Clara University), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 1019 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于历史正常流量的分布测试方法，用来监测大型语言模型（LLM）API中的模型提取攻击；

**💡 创新点**

创新点在于将提取监测转化为“良性校准的流量窗口分布检验”，只需正常查询数据即可设定阈值，且采用简单的MMD统计量即可实现高效检测；

**🔧 技术方法**

核心技术包括句子嵌入编码器（BGE/FlagEmbedding）、多核RBF最大均值差距（MMD）统计量以及基于正态假设的阈值校准；

**📊 数据集**

实验使用十四组攻击–正常查询对，涵盖医疗、SQuAD、WikiText-103和BERT API等四类提取场景，正常查询来自WildChat、GLUE、BoolQ、AG News等公开数据集；

**📈 对比分析**

在统一的流量窗口评估协议下，MMD在保守阈值下实现0.3%正常误报率、100%纯攻击检测率、90.5%平均检测率和95.1%平衡准确率，优于改编后的PRADA、SEAT、CAP、DATE和马氏距离等基线；

**⚠️ 局限性**

局限性主要体现在对极低比例攻击流量（5%以下）时仍存在一定漏检，且在大规模实时部署时对计算资源和窗口大小的需求需要进一步优化。

---

## 297. Interpreting Style Representations via Style-Eliciting Prompts

**arXiv ID:** 2606.05716 | [PDF](https://arxiv.org/pdf/2606.05716v1)

**作者:** Junghwan Kim `[一作]` (University of Michigan), David Jurgens `[通讯]` (University of Michigan)

**通讯引用:** 5237 | [OpenAlex ID](https://openalex.org/A5046126345)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将文本的隐式风格向量解码为可读的、可直接用于指令式生成的自然语言风格提示。

**💡 创新点**

创新点在于：①通过将风格向量映射到可执行的提示，实现对风格的可解释与可控操作；②构造了大规模（1.8M）带有 1,010 维风格特征的合成数据集，用于监督学习；③在三项任务（风格提示恢复、风格生成对齐、人类文本风格引导）上系统评估，并显著优于传统 LLM 描述与风格迁移基线。

**🔧 技术方法**

使用的技术包括：①冻结 LLM（Ministral‑8B‑Instruct）+可训练投影层生成提示；②风格表示模型（Mistral‑Nemo‑Instruct‑2407）用于编码文本；③使用 GPT‑4o 生成风格特征；④采用 ROUGE‑1、LaBSE 语义相似度和 L2 距离等指标评估。

**📊 数据集**

主要数据集为：1) 1,010 条精炼风格特征（26 类）和 1.8M 由 Phi‑4、Qwen2.5‑14B、OLMo‑2‑13B 生成的带有对应风格提示的 QA 回复；2) 300K 人类手写 QA 回复用于对比评估。

**📈 对比分析**

方法通过解码器在无监督提示恢复、风格控制、与人类文本风格对齐三项任务上都优于基线；在提示恢复中 ROUGE‑1、LaBSE、LLM‑judge 均显著提升；在风格控制中 L2 距离最低；在人类风格引导中距离显著小于所有基线。

**⚠️ 局限性**

局限性包括：仅在英文上验证；数据集仅覆盖问答领域，未测试叙事或技术文档等其他写作风格；生成模型受限于三种相似规模的 LLM，可能在更大或不同架构的模型上效果下降。

---

## 298. Human Oversight and Overload: Two Hidden and Costly Burdens of AI-Assisted Software Engineering

**arXiv ID:** 2606.05770 | [PDF](https://arxiv.org/pdf/2606.05770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 299. SentinelRAG: Synthetic Sentinel Knowledge for RAG Database Copyright Protection

**arXiv ID:** 2606.05787 | [PDF](https://arxiv.org/pdf/2606.05787v1)

**作者:** Tsun On Kwok `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10751 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为RAG数据库设计了一种名为SentinelRAG的水印框架，在知识库中注入与语料风格一致的虚构知识条目，实现后期所有权验证。

**💡 创新点**

创新点在于：①利用全新虚构实体而非真实实体构造水印，避免污染知识库；②采用知识层面的标记而非词表级别，显著提升对内容重写与检索过滤攻击的鲁棒性；③以黑盒查询+假设检验的统计方法实现可量化的所有权判定，显著降低误报。

**🔧 技术方法**

主要技术包括：基于LLM的虚构知识生成与风格匹配；秘钥哈希挑选sentinel文档；检索-生成管线中的嵌入与检索策略；黑盒查询与统计假设检验；对抗性攻击实验（内容重写、检索频率删减、异常检测）。

**📊 数据集**

实验使用四大检索语料（MS‑MARCO、HotpotQA、NFCorpus、FiQA），并在附录中扩展至WikiHow、PolicyQA、MATH、CodeSearchNet等六类数据集，覆盖多领域与多规模。

**📈 对比分析**

与现有RAG‑WM及词表级别方法对比，SentinelRAG在仅0.1%注入率时即可获得p<10⁻⁵的统计显著性，检索/回答干扰率低于1%；在高阈值检索频率删减或异常检测攻击下仍能保持高检出率；相比之下，RAG‑WM在低阈值下干扰率高且误报率大。整体表现显示高检测准确度、低干扰和强鲁棒性。

**⚠️ 局限性**

局限性：生成的虚构知识缺乏严格的事实核查，可能与真实世界事实无意冲突；在对事实敏感的医学或法律知识库中，需额外验证机制（如查询外部知识图谱）以确保安全。

---

## 300. Physics-Guided Deep Unfolding for Blind Cross-Sensor Spectral Super-Resolution via Learning the Spectral Transformation Function

**arXiv ID:** 2606.05759 | [PDF](https://arxiv.org/pdf/2606.05759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 301. TinyML-Driven Cybersecurity for Autonomous Spacecraft: Latency-Accuracy Analysis for SPARTA RF and Cyber Threat Detection

**arXiv ID:** 2606.05779 | [PDF](https://arxiv.org/pdf/2606.05779v1)

**作者:** Van Le `[一作]` (Virginia Tech), Tan Le `[通讯]` (Hampton University)

**通讯引用:** 1066 | [OpenAlex ID](https://openalex.org/A5055487755)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了在自主航天器上使用 TinyML 对 SPARTA 框架下的 RF 与网络攻击进行实时检测。

**💡 创新点**

创新点在于将物理信息驱动的 RF 系统模型与 TinyML 经典模型相结合，系统性分析了推理延迟与准确率权衡，并指出 Payload 变形和地面段妥协的检测弱点。

**🔧 技术方法**

采用物理启发式 spectrogram 生成、Logistic Regression、SVM、Random Forest 与 MLP 四种经典 TinyML 模型，以及 CMSIS‑NN 低延迟实现。

**📊 数据集**

使用基于 SPARTA 攻击模型合成的六分类 RF spectrogram 数据集，包括正向、干扰、伪造、Payload 变形、地面妥协与命令注入六种模式。

**📈 对比分析**

通过在 Cortex‑M7 上测量推理时延和在测试集上的准确率与宏 F1，结果显示 LR 与 SVM 在微秒级延迟下准确率 94‑95%，RF 最高 95% 但延迟 7.3 ms，MLP 最高宏 F1 0.824 且推理 100 µs。

**⚠️ 局限性**

主要限制是所有模型在 Payload 变形和地面段妥协两类攻击中表现欠佳，表明现有 spectrogram 特征不足以捕捉细微或瞬态扰动。

---

## 302. An Improved CNN-LSTM Based Intrusion Detection System for IoT Networks

**arXiv ID:** 2606.05776 | [PDF](https://arxiv.org/pdf/2606.05776v1)

**作者:** Mohammad Tariq Ikhlas `[一作]` (National University of Sciences and Technology), Muhammad Khuram Shahzad `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 3050 | [OpenAlex ID](https://openalex.org/A5035787368)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种改进的CNN-LSTM混合模型，用于IoT网络入侵检测，能够实现多类别（Benign、DDoS、DoS、Recon）分类。

**💡 创新点**

创新点包括：① 将CNN与LSTM结合，既提取空间特征又捕获时序关联；② 扩展到多类别分类，超越传统二分类；③ 融合多来源数据提升泛化能力；④ 在实验中实现实时预测模拟。

**🔧 技术方法**

技术手段：TensorFlow/Keras深度学习框架；CNN卷积层＋MaxPooling进行特征提取；LSTM层学习时序依赖；全连接层+Softmax完成分类；预处理包括缺失/无限值处理、MinMax归一化、标签编码；评估指标包括准确率、精确率、召回率、F1分数。

**📊 数据集**

使用了多份CSV文件构成的IoT流量数据集，包含正常流量以及DDoS、DoS、Recon攻击样本，经过合并后形成统一的训练/测试集。

**📈 对比分析**

通过与基线CNN模型在相同数据、相同超参数（批量32，10个epoch，Adam优化器，交叉熵损失）下的对比实验，CNN-LSTM在测试集上实现约97%的准确率，精确率、召回率、F1分数均提升至≈97%，相较基线CNN提升了约3%～4%。然而Recon类的召回率仍略低。

**⚠️ 局限性**

局限性：Recon类识别效果不佳，易与正常流量混淆；实验仅在离线CPU环境完成，缺乏实时部署验证；对少数类的处理不足（数据不平衡）；未尝试更复杂的注意力或Transformer架构，也未进行深入的超参数调优。

---

## 303. Imagine Before You Predict: Interleaved Latent Visual Reasoning for Video Event Prediction

**arXiv ID:** 2606.05769 | [PDF](https://arxiv.org/pdf/2606.05769v1)

**作者:** Tianxiang Jiang `[一作]` (University of Science and Technology of China), Yi Wang `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种交互式潜在视觉推理框架，允许多模态大语言模型在自回归解码过程中交替生成文本和连续潜在视觉状态，从而实现视频事件预测

**💡 创新点**

核心创新是将中间未来视觉信息保持在连续潜在空间而非完全转化为文本，并通过视觉收益筛选构造高效的训练语料；同时设计LA-DAPO两项奖励（结果对比与时间多样性）实现对潜在轨迹的强化学习优化

**🔧 技术方法**

使用Qwen3‑VL‑8B作为基线模型，结合专门的潜在视觉通道（特殊控制标记）、视觉嵌入对齐、监督微调以及LA‑DAPO强化学习算法

**📊 数据集**

主要实验数据集包括FutureBench（多项选择预测）和TwiFF‑Bench（开放式未来帧推理），以及用于构造潜在训练集的TwiFF‑2.7M视频推理链条

**📈 对比分析**

在FutureBench上，模型从61.0提升至85.4（超过Video‑CoE 10.4点、Qwen3‑VL‑30B‑A3B 18.5点）；在TwiFF‑Bench上，平均分从2.44提升至3.04，显示出在更长、非连续未来事件上的显著优势

**⚠️ 局限性**

限制包括对潜在空间的监督依赖较强，需高质量的视觉收益筛选；潜在推理的可解释性相对较低，且在极大视频长度或复杂交互场景下可能仍存在性能瓶颈

---

## 304. Quantifying the Energy-Saving and QoS Trade-Off in Traffic Offloading for Real 4G/5G Scenarios

**arXiv ID:** 2606.05752 | [PDF](https://arxiv.org/pdf/2606.05752v1)

**作者:** D. Reiss `[一作]` (i2cat foundation), O. Sallent `[通讯]` (upc)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在5G NSA部署中，通过关停低负载5G小区并将其流量转移到同一站点内的4G小区，以节能并评估对QoS的影响。

**💡 创新点**

创新点在于：① 使用真实运营商RAN KPI数据构建数据驱动评估框架；② 设计基于阈值γ的贪心关机策略，并加入QoS阈值的“QoS-aware”策略；③ 采用水填充算法均衡4G负载并用lookup表估算平均吞吐量，量化能耗与吞吐需求的权衡；④ 发现单一全局阈值不可行，需要细粒度γ。

**🔧 技术方法**

技术手段包括：数据分析与可视化、负载与能耗相关性建模、阈值控制、水填充负载均衡、lookup表映射吞吐量、CCDF分析及数字孪生概念。

**📊 数据集**

使用了欧洲某移动网络运营商提供的真实RAN KPI数据，覆盖4G 312站点/3427小区、5G 220站点/1271小区，15分钟平均值，覆盖两个月及多周。

**📈 对比分析**

通过计算每小区可关机比例、全网能耗节约以及各吞吐阈值下满足率进行比较；结果显示γ=100%时关机率最高达79%，能耗节约13.7 MWh；但在满足10/15/20/25 Mbps时关机率降至约45%/37%/33%/29%，能耗仅2.9 MWh，显著体现能耗与QoS之间的权衡。

**⚠️ 局限性**

局限性包括：① 缺乏UE位置信息，无法实现跨区负载平衡；② 仅基于单周数据，未考虑长期负载波动；③ 采用oracle式全局视角的QoS评估，实际部署需实现实时决策；④ 未考虑基站深睡模式及跨频段调度的细节。

---

## 305. BeGREEN Intelligent Plane for AI-driven Energy Efficient O-RAN management

**arXiv ID:** 2606.05747 | [PDF](https://arxiv.org/pdf/2606.05747v1)

**作者:** M. Catalan-Cid `[一作]`, M. Ghoraishi `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了BeGREEN Intelligent Plane与AI Engine，提供松耦合的AI/ML服务，支持O-RAN RIC的能效自动化控制，示例实现了基于流量预测的基站开关控制。

**💡 创新点**

创新点在于将AI/ML工作流外部化到专用AI Engine，并通过Assist rApp/xApp实现模型产出与RIC的松耦合发布，兼容O-RAN的DME和即将定义的O1+、O2+接口。

**🔧 技术方法**

采用MLOps框架MLRun与Nuclio实现服务器无关的模型训练、推理与监控，使用XGBoost回归模型进行基站负载预测，结合A1、O1、O2等O-RAN接口进行控制决策。

**📊 数据集**

使用来自西班牙运营商的真实网络流量与能耗数据集进行回归建模，探索Gradient Boosting与时间序列预测方法。

**📈 对比分析**

文中未给出具体实验比较或性能评估，主要呈现架构设计与工作流示例，未展示能效提升量化结果。

**⚠️ 局限性**

局限性包括：架构仍处于设计阶段，未完成接口标准化；缺乏实测验证与能效收益量化；对异构网络环境与多租户场景的兼容性尚未评估。

---

## 306. AISC deployment in dynamic UAV-assisted MEC network: a reinforcement learning method based on heterogeneous graph attention neural network

**arXiv ID:** 2606.05722 | [PDF](https://arxiv.org/pdf/2606.05722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 307. Class-Specific Branch Attention for Mitigating Gradient Interference under Class Imbalance

**arXiv ID:** 2606.05740 | [PDF](https://arxiv.org/pdf/2606.05740v1)

**作者:** Arush Singhal `[一作]` (Thapar Institute of Engineering and Technology), Umang Soni `[通讯]` (Netaji Subhash University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了多分支卷积网络在严重类别不平衡条件下的梯度干扰问题，并提出一种轻量级的类特定分支注意力（CSBA）结构以缓解此类干扰。

**💡 创新点**

创新点在于首次将梯度冲突矩阵作为诊断工具系统性量化梯度干扰，并通过CSBA实现梯度解耦，既保持结构简单又显著提升少数类性能。

**🔧 技术方法**

主要技术包括基于余弦相似度的梯度冲突分析、CSBA（SE风格的分支注意力）、梯度归一化（GradNorm）、焦点损失以及类特定分支头等。

**📊 数据集**

使用的数据集为含六类的太阳能板故障图像集（Solar Panel Clean and Faulty Images）和经过极端重采样的CIFAR‑10‑LT长尾数据集。

**📈 对比分析**

通过与基准多分支网络、焦点损失、GradNorm和类特定分支头的对比实验，CSBA将Physical‑Damage类F1从0.261提升至0.522，在CIFAR‑10‑LT上宏F1从0.595提升到0.655，整体准确率基本不变。

**⚠️ 局限性**

实验仅覆盖两类数据集，未评估推理延迟、内存占用及对分布漂移的鲁棒性；此外CSBA虽显著提升少数类，但对多数类提升有限，未来需在更多域和架构上验证。

---

## 308. Hybrid CNN-LSTM Framework for Intelligent Cyber Attack Detection and Prevention in U.S. Critical Digital Infrastructure: A Comparative Machine Learning Evaluation on CSE-CIC-IDS2018

**arXiv ID:** 2606.05714 | [PDF](https://arxiv.org/pdf/2606.05714v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 309. TextWand: A Unified Framework for Scene Text Editing

**arXiv ID:** 2606.05730 | [PDF](https://arxiv.org/pdf/2606.05730v1)

**作者:** Shuyu Wang `[一作]` (Peking University), Jian Zhang `[通讯]` (Peking University)

**通讯引用:** 54892 | [OpenAlex ID](https://openalex.org/A5100410082)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一框架 TextWand，能够同时完成场景文本的删除、生成与替换，支持精准的布局与风格控制。

**💡 创新点**

创新点在于将编辑任务拆解为渲染与擦除两大原语，并引入Overlay-Reference Positional Encoding (ORPE) 与 Region-Adaptive Suppression (RAS) 两种机制，同时采用分阶段学习策略实现统一编辑。

**🔧 技术方法**

技术实现基于扩散模型（Qwen-Image-Edit-2509）和 MMDiT，配合 VAE、VLM、LoRA、ORPE、RAS 等模块；通过 progressive curriculum 训练提升稳定性。

**📊 数据集**

使用自构建的大规模数据集 TextWand-72K（72000 训练样本）和专门的评测集 TextWand-Bench（1500 测试样本，均分三类任务）。

**📈 对比分析**

在删除、生成和替换三类任务上与多种开源/闭源基线（AnyText2、FluxText、Qwen-Image-Edit-2509、FLUX.1-Kontext、LongCat、Nano Banana Pro、Seedream）进行对比，采用 NED、IoU_bbox、FID、LPIPS、SSIM、PSNR、CLIP-I 等指标以及 VLM 评估，TextWand 在所有指标均超越对手，并在用户研究中获得 61.7% 的优先选择率。

**⚠️ 局限性**

局限性包括推理效率低、模型体积大，难以在移动端部署；缺乏模型蒸馏、量化或轻量化设计，需进一步优化以适应资源受限环境。

---

## 310. DiG-Plan: Mitigating Early Commitment for Tool-Graph Planning via Diffusion Guidance

**arXiv ID:** 2606.05728 | [PDF](https://arxiv.org/pdf/2606.05728v1)

**作者:** Yansi Li `[一作]` (Shanghai Jiao Tong University), Zhuosheng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3912 | [OpenAlex ID](https://openalex.org/A5070962435)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DiG-Plan，一种将工具集合探索与依赖关系精炼分离的框架，使用扩散模型生成多样化工具集合，AR 模型完成依赖预测，并通过无 LLM 判断的价值函数选择最优计划。

**💡 创新点**

创新点在于：① 通过扩散模型实现全局、可迭代的工具集合探索，突破 AR 的早期承诺限制；② 将组合搜索与结构精炼解耦；③ 用可部署的价值函数实现 judge‑free 选择，避免外部 LLM 评判成本。

**🔧 技术方法**

核心技术包括：扩散式语言模型（Dream、LLaDA）、自回归（AR）精炼模型、GradientBoosting 回归器的价值函数、Dense Retrieval 作为对照基线。

**📊 数据集**

使用数据集：TaskBench-23（含单工具、链、DAG 任务）和 API‑Bank（转换为 TaskBench 格式的跨域数据）。

**📈 对比分析**

方法评估：与 AR‑only、AR‑two‑stage、AR‑beam、Retrieval+AR 等基线对比；在 TaskBench‑23 上，ToolF1 从 0.661 提升至 0.729（约 10% 相对提升），EdgeRec 同样提升；在候选池层面，Pass@10 由 32% 提升至 94%；在 API‑Bank 上跨域实验亦保持优势。

**⚠️ 局限性**

局限性：① 边缘预测仍存在较高缺失率，需进一步提升；② 依赖多步骤推理，推断时间较长；③ 对于单工具任务无显著优势，主要收益集中在组合搜索场景。

---

## 311. ViCuR: Visual Cues as Recoverable Privilege for Multimodal On-Policy Distillation

**arXiv ID:** 2606.05718 | [PDF](https://arxiv.org/pdf/2606.05718v1)

**作者:** Kanghui Tian `[一作]` (Shanghai AI Laboratory), Yi Wang `[通讯]` (Fudan University)

**通讯引用:** 18204 | [OpenAlex ID](https://openalex.org/A5100364902)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ViCuR框架，在多模态推理任务中用视觉线索替代答案侧的特权信息，并加入轻量级的sink‑token跨注意力模块以实现内部线索恢复，提升了对视觉证据的依赖；

**💡 创新点**

创新点在于：①将特权信息从不可观测的答案/推理文本改为可在输入中恢复的视觉线索，消除训练‑测试不匹配；②设计了sink‑token跨注意力回收模块，帮助模型在不改动推理接口的前提下内部聚合视觉证据；

**🔧 技术方法**

使用了基于策略梯度的 on‑policy distillation（PPO 风格）与教师监督、视觉线索生成、sink‑token 跨注意力机制以及标准的 transformer 架构；

**📊 数据集**

在 Vision R1、DynaMath、MathVista、WeMath、MathVerse、MMMU‑Val、Video‑MME、Geometry3K 等七个多模态推理基准上进行实验；

**📈 对比分析**

与基线（原始模型、GRPO、OPSD、OPD）对比，ViCuR 在 2B 学生上平均提升 +1.19/ +1.24，8B 学生上平均提升 +0.64/ +1.08，尤其在视觉驱动的几何与数学推理任务上显著提升；

**⚠️ 局限性**

局限性包括：对视觉线索生成质量高度依赖；回收模块增加参数与训练复杂度，且在更大模型上收益不一定递增；若线索缺失或误导，教师监督的可靠性会下降。

---

## 312. Beyond Absolute Scores: Relative Edit-induced Difference for Generalizable Image Aesthetic Assessment

**arXiv ID:** 2606.05778 | [PDF](https://arxiv.org/pdf/2606.05778v1)

**作者:** Qifei Jia `[一作]` (Xiaomi Corporation), Yue Zhang `[通讯]` (Xiaomi Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了RED-Aes框架，将图像美学评估从绝对分数回归转向基于可控编辑的相对差异学习，并构建了RED-20k编辑对数据集。

**💡 创新点**

创新点包括：①利用可控编辑模型模拟人类相对审美推理，生成具有因果关系的美学差异；②三阶段训练（编辑因果预训练、格式校准、GRPO强化学习）结合相对排序一致性奖励；③在无绝对分数监督下实现跨域零样本泛化。

**🔧 技术方法**

采用的技术有：可控图像编辑模型（如Qwen-Image-Edit、FLUX、Seedream等）、视觉语言模型（Qwen、GPT）、对比学习与编辑因果推理、链式思考（CoT）生成、GRPO强化学习与相对排名一致性奖励、三阶段训练策略。

**📊 数据集**

使用了自研的RED-20k数据集（约20k源-编辑对），每对配有量化美学差异、编辑指令和CoT推理；数据由VLM与编辑模型自动生成并通过多模型共识筛选。

**📈 对比分析**

在TAD66K、AVA、FLICKR-AES、PARA、AADB五大公开基准上进行零样本跨域评估，RED-Aes-7B平均PLCC≈0.744，SRCC≈0.732，显著优于现有VLM（如GPT-5、Qwen3）与专业模型（Aes-R1、UniPercept）。三阶段训练与RL奖励进一步提升性能，轻量化2B版本亦超越所有基线。

**⚠️ 局限性**

局限性包括：①依赖可控编辑模型生成的差异，可能无法覆盖所有主观审美场景；②自动化数据生成虽规模大，但仍可能存在质量噪声或偏差；③RL阶段计算开销大，且需精细调参；④对极端内容或全新美学标准的适应性尚待进一步验证。

---

## 313. DRIFT: A Residual Flow Adapter for Decoding Continuous Outputs in Vision-Language Models

**arXiv ID:** 2606.05758 | [PDF](https://arxiv.org/pdf/2606.05758v1)

**作者:** Zhuoming Liu `[一作]` (University of Wisconsin--Madison), Yin Li `[通讯]` (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DRIFT框架，将预训练的视觉‑语言模型（VLM）从离散文本输出迁移到连续量化输出；

**💡 创新点**

采用残差流适配器：先用基线预测器给出粗略估计，再通过流匹配进行局部残差迭代细化，显著简化优化并提升精度；

**🔧 技术方法**

主要技术包括：流匹配（flow matching）、ODE求解、残差预测与门控（skip connection & gating）、LoRA微调；

**📊 数据集**

在多种数据集上评估：视频事件定位（Charades‑STA、ActivityNet‑Captions）、视觉‑语言‑动作任务（Libero、Simpler WidowX）、二维定位（RefCOCO）以及世界动作模型（FastWAM）；

**📈 对比分析**

与MLP、扩散、流匹配、tokenization等基线对比，DRIFT在多项指标上均超过对手，取得多项SOTA结果（如Libero平均成功率97.9%、Charades‑STA R@0.3 67.2%等）；

**⚠️ 局限性**

局限性：依赖于强大的基线预测器，若基线信息不足效果受限；实验主要在仿真环境，缺乏真实机器人验证；

---

## 314. SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents

**arXiv ID:** 2606.05761 | [PDF](https://arxiv.org/pdf/2606.05761v1)

**作者:** Wenxuan Wang `[一作]` (Harbin Institute of Technology), Yang Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 50769 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了 SubtleMemory 基准，用于测试长周期 AI 助手在细粒度关系记忆辨识上的能力。

**💡 创新点**

创新点在于将互补、细微、矛盾三类关系嵌入用户历史，并提供可控的关系标签与分阶段诊断框架。

**🔧 技术方法**

使用大型语言模型进行记忆生成与判定，并结合 Mem0、MemOS、EverMemOS 等记忆系统以及 OpenClaw/MetaClaw 等代理。

**📊 数据集**

数据集由 1,090 组关系控制的语义变体组成，共 1,522 个评估实例，覆盖 10 个领域。

**📈 对比分析**

通过对比六种独立记忆系统、两种原生 Claw 代理及插件化代理，发现最优系统仍低于 oracle，尤其在矛盾关系上性能差距显著。

**⚠️ 局限性**

局限性包括仅覆盖文本长历史、受限的关系类型、对 LLM 判定的依赖，以及未涵盖多模态、跨语言等场景。

---

## 315. SagnacAssisted Enhanced OTDR for Distributed Acoustic Sensing: A Standardized Benchmark and Engineering Evaluation Framework

**arXiv ID:** 2606.05754 | [PDF](https://arxiv.org/pdf/2606.05754v1)

**作者:** Weiguang Wang `[一作]` (East China Jiaotong University), Tianchang Xie `[通讯]` (Guangdong University of Technology)

**通讯引用:** 13 | [OpenAlex ID](https://openalex.org/A5111672407)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种基于Sagnac干涉仪辅助的ϕ-OTDR分布式声学传感系统，并在该系统上构建了统一的融合式事件识别基准框架。

**💡 创新点**

创新点主要包括：①在物理层引入Sagnac相位连续响应作为补偿，缓解偏振诱发衰减带来的信号退化；②提出了以多指标（准确率、宏F1、噪声报警率、漏检率、推理时延）为核心的标准化评估协议；③通过通道分组优化展示了通道组合对双分支融合效果的重要性，强调了通道分组是评估中的关键因素。

**🔧 技术方法**

使用了Sagnac干涉仪+ϕ-OTDR混合光学前端、FPGA实现的时空同步与跨通道相关、手工特征+SVM、概率增强SVM、单分支CNN、双分支融合CNN等多种模型和预处理方法。

**📊 数据集**

实验数据来自10km长光纤，包含15,419个样本，六类事件（背景、挖掘、敲击、流水、围栏摇晃、行走）。数据同时提供平衡拆分和极度不平衡长尾拆分。

**📈 对比分析**

所有方法在相同的训练/测试划分、预处理和指标下进行比较。结果显示手工特征+SVM约41%准确，概率SVM约56%，单分支CNN 86%，双分支融合CNN 89.8%准确、89.8%宏F1、5%噪声报警率、0%漏检率、12.8ms推理时延。双分支融合在所有指标上均优于其他方案，体现了从浅层→概率→单分支→双分支的性能递进。

**⚠️ 局限性**

局限性包括：①通道分组仅在统一12通道矩阵上实现，未真实分离Sagnac与ϕ-OTDR通道；②仍依赖监督学习，标注质量受限；③模型复杂度导致推理时延较高，需进一步优化至边缘化部署；④未在极长距离或多参数联合感知环境下验证；⑤未探讨在线/自适应学习策略。

---

## 316. PlanBench-V: A Spatial Planning Map Benchmark for Vision-Language Models

**arXiv ID:** 2606.05744 | [PDF](https://arxiv.org/pdf/2606.05744v1)

**作者:** Minxin Chen `[一作]` (Tongji University), Wenjia Zhang `[通讯]` (Tongji University)

**通讯引用:** 2210 | [OpenAlex ID](https://openalex.org/A5100719738)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 PlanBench‑V，一个专门评估视觉‑语言模型在空间规划图解释能力的基准，包含 223 张专家标注的规划图和 1,629 对问题‑答案对，并设计了四维评估框架。

**💡 创新点**

创新点在于首个针对空间规划图的领域专用基准、基于理论的 Perception、Reasoning、Association、Implementation 四维评估框架，以及对代理推理模型在专业任务中的差异性分析。

**🔧 技术方法**

使用了多模态视觉‑语言模型（如 GPT‑4o、Qwen 系列、InternVL、Gemini‑Pro 等）与 LLM‑as‑Judge 评估协议，结合结构化提示与量化指标进行模型性能比较。

**📊 数据集**

使用的数据集为 Spatial Planning Map Database（SPMD），包括 223 张国内外规划图和 1,629 条由专业规划师手工标注的问答对。

**📈 对比分析**

通过 17 种 VLM 在两代模型（2025 与 2026）上的对照评估，采用 LLM‑as‑Judge 打分，结果显示 2026 年代理推理模型 Qwen3.6‑Plus 的整体得分 1.701，显著高于 2025 年最佳 GPT‑4o 的 1.342，提升约 27%。

**⚠️ 局限性**

局限在于实现层面任务仍表现不佳，模型往往输出冗长、缺乏政策依托且缺少专业判断，显示当前通用 VLM 在专业规划推理方面仍有显著瓶颈。

---

## 317. Microskill Architecture: A Modular Skill-Driven Framework for AI-Native Code Generation

**arXiv ID:** 2606.05720 | [PDF](https://arxiv.org/pdf/2606.05720v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 318. AdaPLD: Adaptive Retrieval and Reuse for Efficient Model-Free Speculative Decoding

**arXiv ID:** 2606.05742 | [PDF](https://arxiv.org/pdf/2606.05742v1)

**作者:** Runheng Liu `[一作]` (Beijing Institute of Technology), Heyan Huang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 5200 | [OpenAlex ID](https://openalex.org/A5087631670)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无训练的自回归推理加速方法AdaPLD，利用可复用上下文进行推测式解码

**💡 创新点**

创新点在于两方面：①采用词形匹配为默认检索，检索失败时回退到语义相似度检索，提升检索召回率；②在主复制路径之外构造分支与后继复制，扩展可接受的假设空间

**🔧 技术方法**

使用词形匹配、token嵌入相似度检索、隐藏层表示重排序、分支+后继复制以及树注意力的推测验证

**📊 数据集**

在输入驱动生成（摘要、对话、检索增强生成）、代码编辑（CodeEditorBench）以及推理任务（AIME 2024、MATH‑500、MMLU‑Pro）等多组公开基准上进行评测

**📈 对比分析**

与传统自回归、PLD、PLD+、Token Recycling、SAMD、Logitspec等无草稿模型的推测式解码方法对比，AdaPLD在所有模型规模上平均提升2.27×至3.10×的解码速度，且在编辑任务中表现尤为突出

**⚠️ 局限性**

受限于必须在已有上下文中寻找可复用锚点，若任务对上下文依赖弱则加速效果有限；使用token级嵌入作为语义检索，忽略了更丰富的短语或结构相似度；超参数（分支宽度、复制长度、相似度阈值）对性能影响显著，需在不同部署场景中手动调优

---

## 319. VTI-CoT: Visual-Textual Interleaved Chain of Thought for Video Reasoning

**arXiv ID:** 2606.05736 | [PDF](https://arxiv.org/pdf/2606.05736v1)

**作者:** Shufan Zhang `[一作]` (Beijing University Of Posts And Telecommunications), Kunlin Yang `[通讯]` (Beijing Shanwei Zhixing Technology Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种视觉‑文本交错的链式推理框架 VTI‑CoT，旨在提升视频推理的解释性与准确性

**💡 创新点**

创新点在于：① 将每一步推理与对应的视频片段显式关联，② 通过 OCR 渲染将结构化的视觉‑文本推理压缩为单一 canvas，显著提升训练效率与信息密度

**🔧 技术方法**

采用 CLIP、InternVL3、Qwen2.5‑VL‑7B‑Instruct 等视觉与语言模型，并结合自动化 CoT 构造、OCR 渲染与视觉‑文本交错训练策略

**📊 数据集**

使用 Video‑R1 与 MovieChat 两大视频推理数据集，并通过自动化 pipeline 构建 VTI‑Video‑R1‑CoT‑165K 与 VTI‑MovieChat 两大视觉‑文本交错 CoT 数据集

**📈 对比分析**

在 MVBench、TempCompass、Video‑MME、MMVU、LongVideoBench 与 LVBench 六大基准上，VTI‑CoT 在相同参数规模下均优于现有 SOTA，且训练收敛速度显著提升

**⚠️ 局限性**

局限性包括：对视觉信息的高度依赖可能导致细节丢失；OCR 渲染与压缩过程需要额外计算资源；在极长视频或极高分辨率场景下仍面临效率与精度的权衡

---

## 320. Automated Proving of Shannon-Type Entropy Inequalities via Fine-Tuned Language Models and Guided Tree Search

**arXiv ID:** 2606.05729 | [PDF](https://arxiv.org/pdf/2606.05729v1)

**作者:** Shing Yin Wong `[一作]` (City University of Hong Kong), Cheuk Ting Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5084702636)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用小规模精调的语言模型配合引导束搜索，自动化证明Shannon型熵不等式。

**💡 创新点**

首次证明在0.6B参数模型上通过树搜索实现85%成功率，并通过系统消融验证上下文长度和数据分布对性能影响。

**🔧 技术方法**

低秩适配LoRA细调Qwen3模型、基于残差的步进证明、束搜索加启发式评分以及格式与LP误判过滤等技术。

**📊 数据集**

四个训练集（A‑D）共389.5M token，覆盖n=3–9、不同上下文长度和是否n=9偏置；测试集60题，n=10–15。

**📈 对比分析**

与GPT‑5.5零射击、PSITIP、AITIP及1.7B模型对比，0.6B在最终测试集上85%成功率，GPT‑5.5仅1.7%，LP求解器在大n下内存超限。

**⚠️ 局限性**

主要限于步骤格式失效、搜索预算过快消耗以及对中等n的鲁棒性不足，需改进约束解码与更高层次的搜索策略。

---

## 321. AGI and the Limits of Value Production

**arXiv ID:** 2606.05715 | [PDF](https://arxiv.org/pdf/2606.05715v1)

**作者:** Zichen Song `[一作]` `[通讯]` (Sungkyunkwan University), Zichen Song (Sungkyunkwan University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建AGI替代劳动力的政治经济学动态模型，分析技术可替代性与实际采纳的差异，并揭示深度采纳对资本结构和利润率的结构性冲击。

**💡 创新点**

首次系统区分技术可替代性与实际采纳，利用阈值与连续动态方程揭示AGI深度采纳导致资本有机组合上升、剩余价值与利润率下降的内生机制。

**🔧 技术方法**

采用马克思价值论框架的定量政治经济学建模，使用连续动力学方程、阈值解析、导数推导等数学技术。

**📊 数据集**

无实证数据集，模型仅基于理论推导。

**📈 对比分析**

模型未进行实验比较，主要通过数学演绎说明利润率随采纳率下降的趋势。

**⚠️ 局限性**

主要局限在于假设AGI仅转移价值而不创造新价值，忽略人力创造性和新产业出现的可能性，且缺乏实证验证。

---

## 322. Domain-Adapted Small Language Models with Hybrid Post-Processing: Achieving Cost-Efficient, Low-Latency Multi-Label Structured Prediction via LoRA Fine-Tuning on Scarce Data

**arXiv ID:** 2606.05781 | [PDF](https://arxiv.org/pdf/2606.05781v1)

**作者:** Srinivasan Manoharan `[一作]` (PayPal Inc), Haifeng Wu `[通讯]` (PayPal Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过在仅219个示例上对LLaMA 3.1 8B模型进行LoRA微调，并结合确定性规则后处理，实现了多标签合规评估任务；

**💡 创新点**

创新点包括将大模型的域知识压缩到小模型权重中、引入针对关键决策边界的硬负例增强，以及将神经网络与符号规则分层的混合推理框架；

**🔧 技术方法**

采用LoRA参数高效微调、正则化、标签遮蔽、规则匹配与JSON结构校验等技术；

**📊 数据集**

使用来自金融服务对话系统的219条人工标注对话（其中包含20条硬负例增强样本）和一个53条盲测集；

**📈 对比分析**

与前沿API（GPT‑4o、Claude等）对比，单GPU推理约2 s，成本每评估仅$0.013，准确率83%（人类验证），并实现了46–76%的成本节省；

**⚠️ 局限性**

局限性包括测试样本量有限导致统计不稳、对“Disposition”字段的预测偏置、类别不平衡以及规则后处理对新对话模式的脆弱性。

---

## 323. LiAuto-GeoX: Efficient Grounded Driving Transformer

**arXiv ID:** 2606.05774 | [PDF](https://arxiv.org/pdf/2606.05774v1)

**作者:** Jiawei Lian `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 129016 | [OpenAlex ID](https://openalex.org/A5100604690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了 LiAuto-GeoX，一种高效的基于 Transformer 的驾驶几何模型，能够在车载上实时生成高质量的三维重建，并将该几何表示迁移到后续的轨迹预测、占据预测和未来帧预测任务中。

**💡 创新点**

创新点在于（1）构建了一个大规模的驾驶几何教师模型，并通过两种新颖的几何保持蒸馏方法——基于教师激活的掩膜引导的深度感知蒸馏和相对位姿关系蒸馏，保持了局部精细度和跨视角一致性；（2）在同等参数量下实现了 220 FPS 的实时推理；（3）展示了几何表示在多任务迁移中的强大通用性。

**🔧 技术方法**

主要技术包括：多视角摄像头条件的 Transformer 视觉几何网络；基于教师激活的掩膜引导的深度感知蒸馏；相对位姿关系蒸馏；大规模多数据集联合训练；以及在车载环境下的高效模型压缩。

**📊 数据集**

使用了七个公开自动驾驶数据集：Waymo、nuScenes、PandaSet、Lyft、DDAD、KITTI 以及 OpenScene，涵盖不同摄像头配置和多样化场景。

**📈 对比分析**

与 VGGT、FastVGGT、LiteVGGT、π^3、OmniVGGT、DVGT 等基线比较，LiAuto-GeoX 在 155M 参数量下实现了：- 3D 重建 Acc/Comp 与大模型相当，尤其在 DDAD 上最佳；- 220 FPS 推理速度，显著优于其他 1B+ 模型；- 在多任务迁移中获得 90.6 PDMS 轨迹预测、24.63% mIoU 占据预测和 47.67% IoU 未来帧预测，优于同类基线。

**⚠️ 局限性**

局限性包括：在极端天气或复杂光照条件下的几何精度仍有提升空间；对非摄像头传感器（如激光雷达、毫米波雷达）的泛化能力未充分验证；以及在极大视角差异或跨域迁移时可能出现一致性下降。

---

## 324. ExpSpeech-Net: Multimodal Fusion of Expression and Speech for Deepfake Detection

**arXiv ID:** 2606.05760 | [PDF](https://arxiv.org/pdf/2606.05760v1)

**作者:** Ruchika Sharma `[一作]` (Netaji Subhas University of Technology), Rudresh Dwivedi `[通讯]` (Netaji Subhas University of Technology)

**通讯引用:** 1556 | [OpenAlex ID](https://openalex.org/A5008487159)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种多模态深度伪造检测框架 SqN-R-DFD，结合人脸图像和语音信号，使用 SqueezeNet 与 RNN 两个轻量级网络对融合后的特征进行训练。

**💡 创新点**

创新点包括：①将人脸表达与语音模式融合；②采用 ISLBT、MPNCC 等前沿特征提取；③使用 Sandpiper‑Assisted Slime Mould Algorithm (SASMA) 的层级特征选择策略，实现特征的高效筛选与平衡。

**🔧 技术方法**

技术实现：SqueezeNet、RNN、ISLBT、DSBME、VGG16/ResNet50 深度特征、MPNCC、MFCC、色度特征、面部级联检测、DSN 语音归一化、SASMA 优化与 HFS 级联特征选择。

**📊 数据集**

使用的公开数据集：World Leader Dataset (WLDR) 与 DeepfakeTIMIT，分别包含 1710 条与 640 条真假视频样本。

**📈 对比分析**

对比方法：SqueezeNet、RNN、Bi‑LSTM、LinkNet、DCNN、DBN、SVM 等传统与深度模型。实验表明 SqN‑R‑DFD 在 WLDR 上取得 94.5% 准确率、97.4% F‑measure、99.3% 精确率；在 DeepfakeTIMIT 上达到 97.4% 准确率、98.4% F‑measure、AUC 超过 0.92，均显著优于基线模型。

**⚠️ 局限性**

局限性：①模型主要在美国政治人物及英语语料上验证，泛化至多语言、多文化场景尚未充分评估；②依赖大量标注数据，跨域适应性有限；③虽然模型轻量，但在极低算力设备上的实时推理仍需进一步优化。

---

## 325. MARDoc: A Memory-Aware Refinement Agent Framework for Multimodal Long Document QA

**arXiv ID:** 2606.05749 | [PDF](https://arxiv.org/pdf/2606.05749v1)

**作者:** Kaifeng Chen `[一作]` (Tianjin University), Qing Yang `[通讯]` (Qifu Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MARDoc框架，将多模态长文档问答拆解为探索、精炼、反思三步，利用结构化记忆保持关键信息。

**💡 创新点**

创新点在于用结构化记忆代替单一上下文流，分离检索与推理，并通过反思反馈动态优化检索。

**🔧 技术方法**

技术包括多模态文档解析（MinerU2.5）、多粒度检索工具、LLM（Qwen3-VL）与多代理框架（ReAct）。

**📊 数据集**

使用MMLongBench-Doc和DocBench这两个多模态长文档问答基准。

**📈 对比分析**

与传统MLLM、RAG和其他代理基线对比，MARDoc在两大基准上均达到或超过最先进模型，单一Qwen3-30B基线即可与Claude 3.5 Sonnet媲美。

**⚠️ 局限性**

局限：无任务专门微调、仅验证Qwen3-VL模型、推理多轮导致推理时延较高。

---

## 326. Intercomparison of Machine Learning Algorithms for Remote Sensing-based In-season Crop Mapping

**arXiv ID:** 2606.05731 | [PDF](https://arxiv.org/pdf/2606.05731v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 327. UNIVID: Unified Vision-Language Model for Video Moderation

**arXiv ID:** 2606.05748 | [PDF](https://arxiv.org/pdf/2606.05748v1)

**作者:** Kejuan Yang `[一作]` (Bytedance), Kenan Xiao `[通讯]` (Bytedance)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于统一视觉-语言模型的全流程视频内容审核系统（UNIVID），通过生成可解释的政策相关字幕实现细粒度多模态推理和可解释输出。

**💡 创新点**

创新点包括：①使用统一的政策意识字幕代替千千种黑盒分类器，提供可视化审计证据；②采用混合人机迭代数据工艺（GPT‑4o + 专家修正）对模型进行安全和政策对齐；③设计三阶段审核流程（Risk Filter、Moderation Actor + RAG、Trend Governance），显著提升违规漏报率和过度裁决率；④将多任务功能集成至单一背骨模型，降低工程维护成本。

**🔧 技术方法**

技术手段包括：基于LLaVA-OneVision架构的多模态大语言模型；Mistral‑v0.3‑7B LLM；视觉编码、token压缩、投影模块；多模态融合网络 + 轻量级策略头；检索增强推理（UNIVID‑RAG）与知识库；FP8量化推理；自研评测基准CapBench（分解字幕为原子事件评估召回与精度）。

**📊 数据集**

数据集：内部自建视频审核数据集，结合GPT‑4o生成的一段式字幕与VQA对，人工修正后形成高质量标注；使用合成数据与真实视频混合，覆盖暴力、性虐、心理健康、监管活动、完整性等五大违规域；还构建Violation Knowledge Base（≈10万条结构化违规事件）供检索增强使用。

**📈 对比分析**

与公开VLM（GPT‑4o、Gemini‑2.5‑Pro、LLaVA‑OV‑8B）比较，UNIVID‑7B在CapBench上各违规域召回率均优于对手，并在生产流量模拟中将违规漏报率下降42.7%，过度裁定率下降37.0%；成本更低，单设备QPS 5.7，推理费用约$180/100万视频（比商业VLM低15×）。

**⚠️ 局限性**

局限性：①未使用强化学习或策略优化直接将平台政策编码为奖励；②仅基于关键帧抽样，无法捕捉仅出现于单帧的违规内容；③模型对极端稀有违规的泛化仍受限；④对跨语言细粒度表达的覆盖需进一步提升。

---

## 328. Space-CIM: Enabling Compute-In-Memory Accelerators for Thermally-Constrained Space Platforms

**arXiv ID:** 2606.05741 | [PDF](https://arxiv.org/pdf/2606.05741v1)

**作者:** Sohan Salahuddin Mugdho `[一作]` (Iowa State University of Science and Technology), Cheng Wang `[通讯]` (Iowa State University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在空间热制约下，GPU+HBM与基于非易失性内存的计算内存（CIM）加速器的性能差异，并提出了辐射器-循环共设计方法

**💡 创新点**

首次将辐射器热拒绝功率直接与系统TOPS关联的共设计框架；证明CIM在空间热限制下能消除热点并显著提升TOPS/W

**🔧 技术方法**

有限元热模拟（FEM）、基于Stefan‑Boltzmann的辐射冷却模型、温度感知的DVFS功率模型、CIM设计空间工具CiMLoop、Comsol仿真

**📊 数据集**

AI工作负载：GEMM（128、4096维）、Llama‑3.2‑3B 语言模型的Prefill/Decode阶段，均采用8‑bit权重/激活

**📈 对比分析**

将GPU与多种CIM配置在相同面积和不同TRP（100‑300 W）下进行TOPS/瓦特和温度衰减比较，结果显示在1 m²辐射器时CIM可比GPU高10‑40倍TOPS，甚至在极低功耗下仍能运行

**⚠️ 局限性**

仅聚焦热制约，未考虑辐射硬化、机械载荷、真机验证等空间环境因素；模型假设简化（如固定辐射效率、理想化功率分布）；实验仅限于特定NVM类型和面积，缺乏对更大功率/更复杂系统的验证

---

## 329. Let It Be Simple: One-Step Action Generation for Vision-Language-Action Models

**arXiv ID:** 2606.05737 | [PDF](https://arxiv.org/pdf/2606.05737v1)

**作者:** Yitong Chen `[一作]` (University of Science and Technology of China), Xipeng Qiu `[通讯]` (Shanghai Innovation Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在视觉-语言-动作（VLA）模型中使用扩散模型进行一次性动作生成，提出通过在训练时偏向高噪声状态的时间分布，从而让标准流匹配目标在一次推断步骤中即可产生强大的动作策略，并在多种机器人实验中验证了该方法的有效性。

**💡 创新点**

创新点在于将VLA动作生成视为“条件‑目标”问题，证明高噪声时间分布能显著提升一次性推断的性能，无需引入教师模型、蒸馏或额外的辅助目标；并通过一系列对比实验表明，一步生成可以与甚至超过传统的多步推断。

**🔧 技术方法**

主要技术包括：条件流匹配（Conditional Flow Matching）框架、时间分布偏移（高噪声调度）策略、轻量化动作解码头（仅用少量参数预测低维动作块）、以及使用SigLIP+PaliGemma的多模态编码器。

**📊 数据集**

使用的数据集包括：MNIST格子到序列的对照实验、LIBERO（Spatial、Object、Goal、Long 四个子套装）、LIBERO‑Plus、LIBERO‑Pro，以及真实机器人双臂YAM RSS任务。

**📈 对比分析**

比较方法：在相同的训练设置下，将一次性推断与十步推断（统一时间分布）进行对比，评估成功率。结果显示，高噪声调度下的一步策略在LIBERO‑Long可达95.6%成功率，且在多套装和真实机器人实验中，常常能与甚至优于十步推断。

**⚠️ 局限性**

局限性：一、对高噪声调度为何有效的理论解释仍不完整；二、如何为不同动作长度、条件组合或执行协议自动选择最佳α值尚未解决；三、对于非常长的动作块（如H40）一次性推断仍表现不佳，表明该方法并非通用。

---

## 330. TAPO: Tool-Aware Policy Optimization via Credit Transfer for Multimodal Search Agents

**arXiv ID:** 2606.05784 | [PDF](https://arxiv.org/pdf/2606.05784v1)

**作者:** Chengqi Dong `[一作]` (University of Science and Technology of China), Guojun Yin `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对工具增强多模态搜索代理中，GRPO算法在多步工具调用时产生的信用误分配（credit misassignment）问题进行系统分析与量化，并提出了基于工具参数确定性的轻量级解决方案——TAPO，能够零成本补偿错误负优势，提升代理性能。

**💡 创新点**

创新点在于：①首次将工具调用的参数相似性与信息获取一致性关联，形成参数确定性假设；②利用批量内成功轨迹构建对照工具调用库，通过信心门控的优势转移与保守阈值校正，精准纠正信用误分配；③实现完全无额外标注、模型或采样的对照方法，计算开销极低。

**🔧 技术方法**

主要技术包括：Group Relative Policy Optimization（GRPO）、GSPO、SAPO等组策略优化；工具参数相似度计算与聚类；参考库构建与信用门控的优势转移；保守优势补偿与经验回放；以及对比实验与训练动态分析。

**📊 数据集**

使用了七个多模态搜索基准数据集，分别是MMSearch、HR-MMSearch、FVQA-test、InfoSeek、SimpleVQA、LiveVQA和MAT-Search，并在这些基准上对齐统一的工具调用（图像搜索、文本搜索、区域放大）进行实验。

**📈 对比分析**

在Qwen3‑VL‑8B模型上，TAPO将GRPO/GSPO/SAPO的平均准确率从51.33%提升至62.64%（相对提升约4.4%），在所有七个基准上平均提升4–6个百分点；相较于SenseNova‑MARS、DeepEyesV2、WebWatcher等现有搜索代理，TAPO也取得了显著优势；计算时间增幅仅0.06%，几乎无额外成本。

**⚠️ 局限性**

局限性包括：①仅在参数确定性强的查询工具（图像搜索、文本搜索、区域放大）验证，其他非确定性工具的适用性未知；②对极难问题成功轨迹稀缺时参考库构建可能不足；③对工具调用的语义相似度度量依赖手工定义，可能影响泛化；④未在更大规模或不同任务上进一步验证。

---

## 331. PiL-World: A Chunk-Wise World Model for VLA Policy-in-the-Loop Evaluation

**arXiv ID:** 2606.05773 | [PDF](https://arxiv.org/pdf/2606.05773v1)

**作者:** Chong Ma `[一作]` (Tongji University), Hanli Wang `[通讯]` (Tongji University)

**通讯引用:** 5408 | [OpenAlex ID](https://openalex.org/A5058982350)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PiL-World，一种针对VLA（视觉-语言-动作）策略的闭环世界模型，用于在不执行真实机器人操作的情况下进行策略循环评估。

**💡 创新点**

创新点在于：①将VLA预测的动作块转化为多视角的未来观测并反馈给策略；②使用动作衍生的视觉控制信号和潜在多视角历史记忆来提高生成一致性；③通过在成功与失败轨迹上细调训练，提升想象轨迹的真实性。

**🔧 技术方法**

技术包括：基于扩散式视频生成的潜在空间模型；动作到视觉控制的几何投影；Latent History Memory（潜在历史记忆）；双阶段训练（先在大规模RealSource World预训练，再在目标任务上细调）。

**📊 数据集**

使用的主要数据集是RealSource World（14M帧，11k+ 任务）进行预训练，以及针对排序方块、堆叠碗、堆叠方块三类双臂操作任务的成功与失败示范数据。

**📈 对比分析**

与最先进的Ctrl-World进行对比。PiL-World在三项任务中将真实-想象成功率差距从平均63.2%降低至12.0%，并将幻觉自由比例（HFR）提升至70%以上，单步LPIPS误差亦比Ctrl-World低约10-30%。

**⚠️ 局限性**

局限性包括：仅在三种双臂任务上验证，泛化能力未知；接触丰富的操作仍有挑战，误差会被放大；幻觉判定依赖人工标注；在严重遮挡或手腕摄像头主导的场景下，动作-控制投影效果可能受限。

---

## 332. Cosine Misleads: Auxiliary Losses Reshape Vision Language Models, Not Their Latents

**arXiv ID:** 2606.05753 | [PDF](https://arxiv.org/pdf/2606.05753v1)

**作者:** XiuYu Zhang `[一作]` (National University of Singapore), Zhenkai Liang `[通讯]` (National University of Singapore)

**通讯引用:** 5439 | [OpenAlex ID](https://openalex.org/A5084611756)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过设计五种Latent Visual Reasoning (LVR) 变体，系统验证了传统对齐度量（如余弦相似度）与模型性能之间的假设，并提出了两个推理时诊断方法（PRISM），用以评估隐层是否真正参与答案生成。

**💡 创新点**

创新点在于：①首次证明余弦相似度与准确率呈负相关，揭示对齐度量的误导性；②通过线性探测器与扰动实验联合定位答案信息所在位置，并用解码差距预测模型对隐层的依赖程度；③指出辅助损失往往通过共享参数而非被监督隐层实现功能。

**🔧 技术方法**

技术上使用了：LVR框架、余弦/均方误差对齐损失、交叉熵主损失、线性探测器（logistic回归）来测量隐藏层解码能力、对隐层的截断/噪声/随机替换扰动以检验因果依赖，以及信息瓶颈理论做解释。

**📊 数据集**

实验数据集包括：Visual-CoT（438k 任务相关标注）用于训练；在三个视觉推理基准上评估：V^*Bench（视觉搜索），MMVP（多模态比较），BLINK（多选问答）。

**📈 对比分析**

通过对比五个LVR变体，发现余弦相似度与准确率相关性为-0.94；线性探测器在答案解码层的准确率与模型准确率高度正相关（+0.98），而在隐层的探测准确率与准确率相关性仅为+0.20；解码差距与模型准确率相关性为+0.86；扰动实验显示隐层被大多数变体绕过，最大误差不超过4个百分点。

**⚠️ 局限性**

局限性包括：仅在单一基础模型（Qwen2.5-VL-3B-Instruct）和单一微调语料上验证；线性探测器只能衡量线性可解码性，不能完全反映后端层的实际使用；扰动方法有限，未覆盖所有可能的隐层干预；研究聚焦于细粒度感知任务，结果可能不适用于更宏观的推理任务。

---

## 333. Three Years of r/ChatGPT: Societal Impact Evaluations from Social Media Data

**arXiv ID:** 2606.05750 | [PDF](https://arxiv.org/pdf/2606.05750v1)

**作者:** Jessica Dai `[一作]` (University of California, Berkeley), Nika Haghtalab `[通讯]` (University of California, Berkeley)

**通讯引用:** 750 | [OpenAlex ID](https://openalex.org/A5080772091)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了基于社交媒体的框架，用以监测并分析大众对 ChatGPT 等大规模语言模型的社会影响，并对 r/ChatGPT 子版块做了从 2022‑2025 年的纵向研究。

**💡 创新点**

创新点在于提出了一个可在线实时监测算法，利用无监督特征提取和随时间更新的顺序假设检验，能够在产品发布后数月内提前识别情感参与等社会影响指标。

**🔧 技术方法**

采用稀疏自编码器（SAE）进行文本特征学习，使用分段线性回归进行变点检测，并通过 anytime‑valid 序列假设检验实现在线监测。

**📊 数据集**

使用 137,154 条 r/ChatGPT 版块的帖子数据（2022‑12‑01 至 2025‑11‑30），涵盖 89,346 名独立用户。

**📈 对比分析**

在实验中，在线方法的重训练事件仅出现三次，重构误差与最优历史模型相近；针对“治疗”主题的监测能在 2024‑10‑29 早于公开关注点，显示了显著的提前预警能力。

**⚠️ 局限性**

局限性包括：仅基于 Reddit 数据，样本不具代表性；无监督特征解释性有限；未做因果推断，结果仅为相关性；对其他子版块或平台的推广性未知。

---

## 334. Membrane: A Self-Evolving Contrastive Safety Memory for LLM Agent Defense

**arXiv ID:** 2606.05743 | [PDF](https://arxiv.org/pdf/2606.05743v1)

**作者:** Minseok Choi `[一作]` (KAIST AI), Youngjun Kwak `[通讯]` (KakaoBank Corp)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于 Contrastive Safety Memory（CSM）的自我进化安全防御体系，利用存储的对抗性与对应安全输入的对比细胞，在不重新训练模型的情况下对抗日益演化的 jailbreak 攻击。

**💡 创新点**

创新点：1) 每个存储细胞同时包含攻击条件与对应的安全条件（对比边界），避免单向记忆导致的误拒；2) 按攻击策略而非表面语料索引细胞，使单个细胞可泛化至同一机制的不同主题变体；3) Paired Self‑Evolution 机制通过每一次攻击‑安全配对自动创建、更新或删除细胞；4) Retrieval Critic 进一步过滤不相关细胞，提升决策准确率；5) 可为模型安全与代理安全提供统一框架。

**🔧 技术方法**

技术：外部可写内存存储 Contrastive Safety Memory；基于向量检索 + LLM 重新排序的两阶段检索；Paired Self‑Evolution 写入策略（创建、更新、删除）；对比决策标准（d_unsafe vs d_safe）；工具动作上下文用于代理安全；基于 Gemini3 Flash 的拒绝判定。

**📊 数据集**

数据集：HarmBench（模型级安全评估）和 AgentHarm（代理级安全评估），使用 Qwen3‑8B 作为受保护模型，Gemini3 Flash 作为响应级判定器；攻击集合包含 PAIR、PAP、TAP、ReNeLLM、FlipAttack、AutoDAN‑Turbo 等六种现代 jailbreak 方法。

**📈 对比分析**

与基线（Self‑Reminder、SmoothLLM、SelfDefend、LlamaGuard3、WildGuard、GuardReasoner、RAD、TrustAgent、ShieldAgent、GuardAgent、AGrail）对比，CSM 在所有六种攻击上均取得最高 F1；模型级 ASR 下降至 0.0–16.0%，Agent 级 ASR 降至 0.9%；FRR（benign refusal）保持在 7–14%，远低于 28–85% 的对手；跨攻击转移 F1 仍保持 87–88%；对内存毒化攻击表现出强鲁棒性；平均延迟在 2–3 秒范围内，低于大多数重型防御。

**⚠️ 局限性**

局限性：仅在英文文本模型/代理安全基准上验证；未涵盖多模态、跨语言或嵌套代理；受限于单一受保护模型 Qwen3‑8B（虽然补充实验表明可扩展）；对完全白盒自适应攻击缺乏评估；依赖 Gemini3 Flash 作为判定器，可能与被保护模型存在共享盲点。

---

## 335. When AI Says It Feels

**arXiv ID:** 2606.05734 | [PDF](https://arxiv.org/pdf/2606.05734v1)

**作者:** Shin-nosuke Ishikawa `[一作]` (Rikkyo University), Hirotsugu Ohba `[通讯]` (Rikkyo University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了一种名为 HMX‑feel 的后训练方法，通过自奖励强化学习让大型语言模型（LLMs）表达情感、意图和自我意识等人类‑like 行为。

**💡 创新点**

首次将 Group Relative Policy Optimization（GRPO）与自评判式奖励结合，构建“自奖励”框架，使模型在保持原有参数的同时显著提升人类‑like 表达，并通过正向与逆向训练对比系统评估其对下游任务的影响。

**🔧 技术方法**

使用 LLM‑as‑a‑judge 评价器、GRPO 强化学习、LoRA 参数微调以及基于自定义人类‑like 评测问答集的自奖励策略。

**📊 数据集**

自定义的 100 题评测集（10 类共 100 问）用于训练与自奖励；对比基准包括 IFEval、BigBench Hard、RULER、ACPBench、BBQ、EQ‑Bench、SQuAD2.0、ToxiGen、TruthfulQA 与 SycophancyEval。

**📈 对比分析**

通过正向训练模型与逆向训练模型（奖励取负）对比，统计五次随机种子下的分数差异。大多数指标提升或仅轻微下降，唯一显著下降的是 BBQ 难辨别情境下的准确率；TruthfulQA 下降提示事实性受影响；sycophancyEval 的 explicit_rate 与 correct_rate 显著上升，模型对用户误导更具抵抗力。

**⚠️ 局限性**

实验仅覆盖少数模型（Qwen3、Gemma、Llama）和有限评测集，缺乏对更大规模模型和更广泛人类‑like 行为的验证；自奖励使用正则化与 regex 提取可能低估答案多样性；未评估长期或实际场景下的安全与伦理风险，结果的普适性仍不确定。

---

## 336. Zero-Copy Semantic Contagion: An In-Memory Streaming Architecture for Evolving Attention Graphs

**arXiv ID:** 2606.05733 | [PDF](https://arxiv.org/pdf/2606.05733v1)

**作者:** Kabir Murjani `[一作]` `[通讯]` (Nirma University), Kabir Murjani (Nirma University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个以Rust为零拷贝边、Python/PyTorch为推断核心的连续时间Neural Hawkes体系，实时解析新闻并通过多资产注意网络检测跨公司关注传播。

**💡 创新点**

首次将零拷贝Rust解析、可扩展的持续时间点点过程、双线性异向激励与自适应边修剪相结合，既实现微秒级新闻解析与毫秒级推断，又将文本驱动的注意网络直接映射到多资产价格传播。

**🔧 技术方法**

Rust零拷贝解析、Python/PyTorch Neural Hawkes + c‑LSTM、双线性注意投影、边修剪策略、MiniLM‑L6‑v2句子嵌入、FPGA/DPDK网络接口等。

**📊 数据集**

FNSPID 2022年7月的新闻与行情数据（638篇文章、47只股票）。

**📈 对比分析**

与同月FNSPID基准单资产预测模型对比，评估跨公司传播检测精度：在90%分位阈值下模型精度为15.1%（随机8.9%，同行业4.5%），提升1.70×和3.36×；整体提升1.36–1.81×；端到端延迟约13 ms。

**⚠️ 局限性**

样本稀疏导致双线性权重无显著提升；评估仅基于每日回报，缺乏微观时段验证；仅单月单组股票，泛化能力有限；嵌入仍使用CPU，GPU未被充分利用。

---

## 337. Narrative Knowledge Weaver: Narrative-Centric Retrieval-Augmented Reasoning for Long-Form Text Understanding

**arXiv ID:** 2606.05724 | [PDF](https://arxiv.org/pdf/2606.05724v1)

**作者:** Qiuyu Tian `[一作]` (Southeast University), Zequn Liu `[通讯]` (Beijing Zhongguancun Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Narrative Knowledge Weaver（NKW），一种面向长篇叙事问答的源文本关联图谱与叙事资产构建与推理框架。

**💡 创新点**

创新点：①将叙事证据按功能（事件、互动、场景）与长程结构（情节片段、故事线）组织；②构建动态角色状态与关系的时间感知实体档案；③在推理时提供文本、图谱、叙事三通道工具与后检阅读卡，确保证据来源可追溯。

**🔧 技术方法**

技术：基于实体关系抽取构建稳定图谱；源文本关联的事件、互动、场景抽取与原子事实；图谱归一化与属性强化；情节片段聚类与故事线 DAG 构造；多通道检索工具与后检阅读卡；LLM后端推理（Qwen3、Llama-3.1、GPT-5.5）。

**📊 数据集**

数据集：STAGE（剧本+问题）、FairytaleQA（儿童故事+问题）、QuALITY（长篇英文段落+多项选择）。

**📈 对比分析**

与Hybrid RAG、GraphRAG、LightRAG、HippoRAG、A-RAG等基线对比，NKW在STAGE的整体准确率和Pass@5均为最优，在FairytaleQA和QuALITY的高容量模型上也表现竞争力，尤其在需要多约束推理的场景中优势明显。

**⚠️ 局限性**

局限：对短篇或单段落问题增益有限；构建与查询成本高；依赖抽取器的准确性，错误会传播；多步骤工具调用导致结果波动；缺乏对不同领域、语言的泛化评估。

---

## 338. Retry Policy Gradients in Continuous Action Spaces

**arXiv ID:** 2606.05888 | [PDF](https://arxiv.org/pdf/2606.05888v1)

**作者:** Soichiro Nishimori `[一作]` (University of Tokyo), Paavo Parmas `[通讯]` (University of Tokyo)

**通讯引用:** 63 | [OpenAlex ID](https://openalex.org/A5047914843)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在连续动作空间中对ReMax目标进行理论分析，并实现了基于ReMax的离线actor–critic算法ReMAC。

**💡 创新点**

证明了ReMax在连续空间中通过梯度方向与幅度双重机制促使策略保持较高熵，并给出了对梯度阻尼的理论解释；同时揭示Adam的数值稳定参数如何调节探索与收敛。

**🔧 技术方法**

采用路径梯度（reparameterization）估计ReMax梯度，基于SAC框架的离线actor–critic架构，使用双Q网络与经验回放；算法对SAC的软熵奖励进行移除，只保留ReMax损失。

**📊 数据集**

使用Brax提供的六个连续控制任务：Ant、HalfCheetah、Hopper、Reacher、Swimmer、Walker2d。

**📈 对比分析**

与SAC和PPO进行对比实验。ReMAC在retry预算M>1时与SAC在返回值上相当，且在部分环境（如Ant、Reacher、Swimmer、Walker2d）表现更好；同时显示出更高的策略熵。

**⚠️ 局限性**

理论分析基于强凸与光滑假设，实际环境可能不满足；算法在每一步需额外计算多次Q评估，计算成本较高；未验证稀疏奖励场景和对Q后验的更深探索。

---

## 339. TAGA: Terrain-aware Active Gaze Learning for Generalizable Agile Humanoid Locomotion

**arXiv ID:** 2606.05880 | [PDF](https://arxiv.org/pdf/2606.05880v1)

**作者:** Peizhuo Li `[一作]` (National University of Singapore), Guillaume Sartoretti `[通讯]` (National University of Singapore)

**通讯引用:** 1464 | [OpenAlex ID](https://openalex.org/A5069667034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TAGA框架，实现了具有主动凝视能力的地形感知类人机器人行走；

**💡 创新点**

核心创新在于无监督学习的层次化主动凝视模块，能基于视觉、运动和本体感知预测最具决策价值的高度扫描区域，并通过跨模态注意力实现信息融合；

**🔧 技术方法**

使用深度强化学习（PPO+异步演员-评论家）、混合注意力网络、Mixture‑of‑Experts动作解码器、对比损失与边界惩罚以及AMP运动先验，整体实现了端到端的控制策略；

**📊 数据集**

训练数据主要来自于Isaac Lab中自定义的多难度地形集合（桥隙、楼梯、窄板、稀疏踏脚石等），每种地形有10个难度级别；实际验证使用Unitree G1机器人在室内外真实场景完成实验；

**📈 对比分析**

与基准CReF以及TAGA的若干消融版本对比，TAGA在多种挑战地形下的成功率均位居前列（例如在1.2 m大桥隙上成功率达98.3%，在稀疏踏脚石上达97.9%），且训练成本显著低于全扫描版本；

**⚠️ 局限性**

局限性包括在高负荷动态运动下伺服器热量负担大、驱动器过热导致动作精度下降，以及高度扫描质量不佳时可能导致步态失误或失败。

---

## 340. Evaluating Stochastic Collapse and Implicit Bias in Multimodal Large Language Models

**arXiv ID:** 2606.05874 | [PDF](https://arxiv.org/pdf/2606.05874v1)

**作者:** Huiyuan Zheng `[一作]` (Fudan University), Hongcheng Guo `[通讯]` (Fudan University)

**通讯引用:** 342 | [OpenAlex ID](https://openalex.org/A5073687083)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RandomBench基准来评估多模态大语言模型在逻辑中立情境下的随机性失衡（Stochastic Collapse），并揭示其隐式偏好与可解释性问题

**💡 创新点**

首次系统地量化逻辑中立随机选择中的偏差，引入随机性指数（RI）、偏差强度指数（BII）和偏差一致性指数（BCI），并发现强大模型反而更易出现偏差；同时揭示视觉劫持和能力一致性悖论

**🔧 技术方法**

利用基于信息熵的统计指标、Kullback-Leibler和Jensen-Shannon距离对模型输出进行分析；通过高温抽样（T=1.0）和禁用链式推理（CoT）获取纯随机决策

**📊 数据集**

构建200个逻辑中立样本，分为RB-Text与RB-Vision两类，每个样本重复抽样50次，共10,000条响应；样本覆盖抽象符号、语言等价、空间感知与情感身份等四大维度

**📈 对比分析**

在七大前沿多模态模型（GPT‑5.1、Gemini‑3.1、Claude‑Sonnet‑4.6、Kimi‑K2.5、Qwen‑3.6、Grok‑4、Doubao‑Seed‑1.6）上进行评测，发现所有模型均出现显著随机性下降，最高单一选项概率可达90%以上；强模型在RI上更低，BII更高，显示能力越强随机性越差

**⚠️ 局限性**

局限性包括：仅使用静态网格化多模态数据，无法检视模型内部表示；受API闭源限制，无法进行机制解释；基准只覆盖逻辑中立情境，未扩展至动态或复杂环境

---

## 341. Deciphering Two Training Clocks in Grokking via Deep Linear Network Theory with Conditional ReLU Reduction

**arXiv ID:** 2606.05863 | [PDF](https://arxiv.org/pdf/2606.05863v1)

**作者:** Hu Tan `[一作]` (Chinese Academy of Sciences), Shihua Zhang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9437 | [OpenAlex ID](https://openalex.org/A5076619069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出并分析了训练过程中的两种时钟——分类器时钟和表示简化时钟，并在深度线性网络和条件ReLU网络上证明了它们的收敛速率；通过实验在模块加法任务上验证了慢时钟导致的泛化延迟现象。

**💡 创新点**

创新点在于将“grokking”现象归因于拟合与内部表示简化两条不同的时间尺度，并给出在深度线性网络中通过层级权重衰减诱导的Schatten范数正则化来解释表示时钟；此外提供了条件ReLU门稳定化的归约原理，将线性分析映射到非线性网络。

**🔧 技术方法**

主要技术包括：深度线性网络理论、后边距增大与尾收缩条件、Schatten范数正则化、KL-尾条件、稳健性（robustness）泛化分析、稳定秩（stable‑rank）与低秩逼近、条件ReLU门冻结与梯度层级分析。

**📊 数据集**

实验数据集为模块加法（mod 113）任务，使用小型ReLU多层感知机（MLP）配合交叉熵损失与权重衰减进行训练。

**📈 对比分析**

比较方法：在相同网络结构与权重衰减下记录训练误差、测试误差及稳定秩随时间变化的曲线；通过可视化激活模式与输出权重展示低秩结构。结果显示，训练误差在日志尺度内快速下降（classifier clock），但稳定秩及测试误差在多项式尺度内缓慢改善，形成明显的“平台”后才出现泛化提升。

**⚠️ 局限性**

局限性：仅在ReLU网络的门激活模式稳定时才适用，无法保证全局收敛；低秩仅是简化机制之一，未涵盖所有可能的结构简化；理论证明基于深度线性近似与KL尾假设，需进一步验证在有限宽度非线性网络中的适用性。

---

## 342. Architecting Strategic Influence: Operationalising the UXR Point of View Framework for Research Function Maturity

**arXiv ID:** 2606.05826 | [PDF](https://arxiv.org/pdf/2606.05826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 343. EEGDancer: Dynamic Emotion Latent Space Masked Modeling with Reinforcement Learning for EEG Continuous Emotion Prediction

**arXiv ID:** 2606.05855 | [PDF](https://arxiv.org/pdf/2606.05855v1)

**作者:** Zhihao Zhou `[一作]` (Shenzhen University), Zhen Liang `[通讯]` (Shenzhen University)

**通讯引用:** 1355 | [OpenAlex ID](https://openalex.org/A5036446937)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了EEGDancer框架，用于从EEG信号中连续预测情绪，包含自监督的VQ‑VAE、Transformer掩码预训练以及基于SAC的强化学习轨迹优化。

**💡 创新点**

创新点在于：①将EEG情绪预测转化为马尔可夫决策过程；②引入离散-连续混合的情绪潜在空间，通过VQ‑VAE学习情绪原型；③使用Transformer掩码学习长时序情绪依赖；④通过强化学习在轨迹层面优化预测，从而捕捉全局情绪演化。

**🔧 技术方法**

核心技术包括：因果时空VQ‑VAE、Transformer掩码预训练、Soft Actor‑Critic（SAC）强化学习、MSE/MAE/相关性评价指标。

**📊 数据集**

采用公开情绪EEG数据集SEED、SEED‑IV以及Long‑Term Naturalistic Emotion（Arousal、Dominance）进行实验。

**📈 对比分析**

与传统机器学习（SVR、Random Forest等）和深度学习方法（EEGNet、DDC、RGNN等）对比，EEGDancer在三大数据集上均实现了最低MSE、最高相关性，表明在连续情绪回归任务上取得SOTA性能。

**⚠️ 局限性**

主要局限包括：模型复杂度高、训练时间长；对极端噪声或跨任务泛化仍需验证；强化学习奖励设计需手工调参，适用性受限。

---

## 344. Amortized Nonlinear Model Predictive Control

**arXiv ID:** 2606.05840 | [PDF](https://arxiv.org/pdf/2606.05840v1)

**作者:** Francesco Pillitteri `[一作]` (IMT School for Advanced Studies), Alberto Bemporad `[通讯]` (IMT School for Advanced Studies)

**通讯引用:** 32516 | [OpenAlex ID](https://openalex.org/A5053340099)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种针对输入仿射非线性系统的学习式约化模型预测控制（MPC）框架，该框架在每个采样周期内仅需求解一个状态相关的二次规划（QP），从而大幅加速控制器计算。

**💡 创新点**

创新点在于：①使用解析基线（基于输入仿射结构的解析QP参数）与神经网络残差校正相结合，显著减少了需要学习的参数；②引入混合损失函数，包括监督模仿、KKT 站定性损失和 Fischer–Burmeister 补全性残差，提升了闭环最优性与约束满足性；③采用可微内部点 QP 解算器确保约束严格满足。

**🔧 技术方法**

技术包括：输入仿射系统的解析QP推导、残差校正神经网络（MLP + 软加法），可微 QP 解算层（内部点），KKT 及 FB 残差正则化，使用 JAX 进行自动微分和隐式微分，Adam 优化器以及余弦学习率衰减。

**📊 数据集**

使用三关节平面机械臂的仿真数据，共计 100,000 条采样点（状态+参考），通过 IPOPT 求解完整 NLP 作为标签；训练集通过 Latin Hypercube 采样获得，测试集为 100 条独立场景。

**📈 对比分析**

与 acados 的实时迭代（RTI）、完整 NLP 求解器（IPOPT）以及直接预测 MLP 进行对比。结果显示，amortized QP 在平均每步解算时间上比 RTI 快约 21 倍、比 IPOPT 快 1826 倍；闭环成本与 IPOPT 相比仅高 4.7%，且在所有 100 场景中收敛，且硬约束得到严格满足；相比之下 RTI 在 12 场景失收敛，成本偏高 17.5%；直接 MLP 收敛率低、约束违反显著。

**⚠️ 局限性**

局限性包括：①依赖于输入仿射结构，非仿射系统需先转化；②残差校正网络的泛化能力受限于训练数据覆盖范围；③在非线性阶段（如前向运动学）解析基线不精确，需更高阶线性化或更复杂的网络；④训练过程需要大量 NLP 求解标签，计算成本较高；⑤软约束仅在极少情况激活，可能无法处理极端约束冲突。

---

## 345. Learning Geometric Representations from Videos for Spatial Intelligent Multimodal Large Language Models

**arXiv ID:** 2606.05833 | [PDF](https://arxiv.org/pdf/2606.05833v1)

**作者:** Haibo Wang `[一作]` (University of California, Davis), Lifu Huang `[通讯]` (University of California, Davis)

**通讯引用:** 2665 | [OpenAlex ID](https://openalex.org/A5042819803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用仅含 2D 视频的训练，通过在训练阶段加入四个几何目标（相机姿态估计、深度图回归、尺度校准、特征对齐）对 MLLM 的内部表示进行重构，使其获得 3D 空间推理能力。

**💡 创新点**

创新点在于：1）将 3D 教师模型的几何先验作为训练目标，而非在推理阶段插入 3D 模块，保持零推理开销；2）通过多尺度特征投影与余弦对齐，将 3D 先验“重塑”到 MLLM 的隐空间；3）联合使用四个互补的几何约束，形成完整的多任务几何学习框架。

**🔧 技术方法**

采用的技术包括：预训练的 Qwen3‑VL‑2B‑Instruct 作为基础 MLLM；冻结的 3D 教师模型（VGGT、DepthAnything‑3 等）用于生成相机姿态、深度、尺度和特征伪标签；多任务学习框架（cam、depth、scale、align）与自定义的 Dense Head 深度预测；多尺度特征投影与 MLP 维度投影；训练使用 AdamW，冻结视觉编码器与教师；在推理阶段移除所有辅助头与教师模型。

**📊 数据集**

训练使用混合视频数据集 VSI‑590K 与 VLM‑3R；评估数据集为 VSI‑Bench（约 5k 题目，来源于 ScanNet、ScanNet++、ARKitScenes），并在实验中对不同 3D 教师模型（VGGT、VGGT‑Ω、DepthAnything‑3）进行了验证。

**📈 对比分析**

方法通过在 VSI‑Bench 上与多种基线（通用模型 GPT‑5、LLaVA‑Video‑72B、空间专用模型 SpaceMind‑8B、VLM‑3R‑7B 等）进行对比，GeoVR‑2B 在平均分 69.1，显著高于基础 Qwen3‑VL‑2B‑Instruct（50.3）以及多项空间模型（如 SpaceMind‑8B 69.6、VLM‑3R‑7B 60.9），且推理阶段无额外 3D 模块开销。

**⚠️ 局限性**

局限性包括：①仅通过 2D 视频学习，仍受限于教师模型的精度和表达能力；②目前仅在 2B 规模模型上验证，尚未在更大模型或更复杂场景下测试；③对长时序、动态物体的细粒度 3D 匹配仍有限；④训练阶段需要额外的 3D 教师和辅助头，计算成本相对较高；⑤缺乏对更通用场景（如室外或非结构化环境）的评估。

---

## 346. CaliDist: Calibrating Large Language Models via Behavioral Robustness to Distraction

**arXiv ID:** 2606.05799 | [PDF](https://arxiv.org/pdf/2606.05799v1)

**作者:** Mohammad Anas Jawad `[一作]` (University of Illinois Chicago), Cornelia Caragea `[通讯]` (University of Illinois Chicago)

**通讯引用:** 5378 | [OpenAlex ID](https://openalex.org/A5089085275)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于对抗性分散信息的行为鲁棒性校准方法 CaliDist，利用模型对语义干扰的预测与置信度变化来调节 LLM 的置信度。

**💡 创新点**

创新点在于：①将心理学的误信息效应与达克效应转化为可度量的行为鲁棒性指标；②通过预测不稳定性 μ 与置信度不稳定性 δ 计算可靠性得分 λ，再用可学习的 Sigmoid 缩放调整置信度；③无需访问 logits，兼容黑盒 API，成为 Temperature Scaling 的高分辨率行为代理。

**🔧 技术方法**

技术实现包括：生成 Assertion、Probe、Sample‑Corruption 三类干扰样式；计算 μ、δ 并合成 λ；通过 Min‑Max 归一化和参数化 Sigmoid（α,β）对初始置信度进行缩放；与 TS、Self‑Consistency、Entropy、FSD、SPUQ 等基线对比。

**📊 数据集**

使用 7 个 NLU 基准：MNLI、MSciNLI、PPDB、Yahoo Answers、HellaSwag、CSQA、AQuA‑RAT（以及 ContractNLI）进行评估。

**📈 对比分析**

与 TS、Vector Scaling、Isotonic Regression、Self‑Consistency、Entropy、FSD、SPUQ 等传统和一致性基线相比，CaliDist 在 7 个数据集上平均将 ECE 从 23% 降至 7%（提升 70%）并显著降低 Brier Score；在 GPT‑4o‑Mini、Gemini‑2.0‑Flash 等黑盒 API 上同样优于基线。

**⚠️ 局限性**

局限性：需要多次前向推理和干扰生成，仍有一定计算开销；对极大模型的抗干扰能力尚未验证；每个任务需调优 Sigmoid 参数；在极简或无关干扰下效果可能不如预期。

---

## 347. Agentic Molecular Recovery via Molecule-Aware Exploration

**arXiv ID:** 2606.05847 | [PDF](https://arxiv.org/pdf/2606.05847v1)

**作者:** Suwan Yoon `[一作]` (Korea University), Changhee Lee `[通讯]` (Korea University)

**通讯引用:** 21364 | [OpenAlex ID](https://openalex.org/A5100406188)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对LLM生成的无效SMILES进行化学身份保持的恢复方法。

**💡 创新点**

创新点在于引入分子匹配跟踪、候选扩展和轨迹级选择，构建了一个基于LLM的多阶段代理框架，突破了仅校正合法性或贪婪单轨迹修复的局限。

**🔧 技术方法**

使用了LLM代理（含需求提取、反馈、规划、候选生成与选择四个模块）、RDKit可执行编辑工具、SMISELF语法修复和现有的ReAct/PlanAndAct等基线框架。

**📊 数据集**

实验数据集为ChEBI‑20验证集的无效草稿与对应文本描述。

**📈 对比分析**

与SMISELF修复、LLM仅纠错、贪婪代理（ReAct、ReWOO、PlanAndAct）及其工具增强版本进行对比；在结构相似度、精确匹配、字符串相似度及分布距离等多项指标上，AMREC均实现了最优或接近最优的性能。

**⚠️ 局限性**

局限性包括：仅在计算上评估无效SMILES恢复，未验证可合成性、毒性或生物活性；对LLM的依赖导致结果受提示和模型差异影响；实验仅覆盖ChEBI‑20，缺乏对更广泛化学数据集的验证。

---

## 348. Staying with the Uncertainty: Uncertainty-Scaffolding Strategies for Artificial Moral Advisors in LLM-to-LLM Simulated Conversations

**arXiv ID:** 2606.05890 | [PDF](https://arxiv.org/pdf/2606.05890v1)

**作者:** Salvatore Greco `[一作]` (King's College London), Sylvie Delacroix `[通讯]` (King's College London)

**通讯引用:** 681 | [OpenAlex ID](https://openalex.org/A5027906807)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型（LLM）作为人工道德顾问（AMA）时，如何通过三种不确定性表达策略（Perspective‑Multiplying、Tension‑Preserving、Process‑Reflecting）与三种对照策略（Baseline、Persuasive、Sycophantic）在伦理对话中维持“与不确定性共处”，并比较宣言式与叙事式人格设定的对话效果；

**💡 创新点**

创新点在于：①提出并系统评估三种面向不确定性的对话框架，说明它们在维持参与、促进立场修正与提升对话质量方面的差异；②使用多代理模拟框架在大规模对话中验证这些策略；③发现开源与闭源模型在表达不确定性时采用不同机制（群体多样性 vs. 个体回避）。

**🔧 技术方法**

技术方法包括：多代理模拟框架、Prompt 工程实现人格与不确定性策略、LLM 生成预/后问卷与对话、对话分类模型评估、统计分析（AUROC、F1、精确率/召回率）等。

**📊 数据集**

使用的主要数据集为：Scruples 100道伦理困境（分为高模糊与低模糊两类）以及 PersonaHub 32份宣言式人格（随后转化为叙事式）。

**📈 对比分析**

评估方法：计算AUROC评估模型对人类不确定性标签的匹配度；使用对话分类模型的宏F1评估六种策略的可区分度；对预后问卷的 Δ 统计评估立场、确定性、相似度、清晰度等指标。结果显示：三种不确定性策略在可区分度上均高于对照；Process‑Reflecting 产生最少强化、最高修正率和最强主观帮助感；Perspective‑Multiplying 最大化弱势立场支持；Tension‑Preserving 提升相对理解度；对照策略整体效果最低。

**⚠️ 局限性**

限制包括：①结果基于LLM自我报告，缺乏真实人类交互验证；②实验仅使用单一LLM（GPT‑5.4）作为AMA，可能不具泛化；③开源与闭源模型的差异可能与特定模型实现有关；④Scruples 题目被 LLM 处理后原始模糊标签可能失效；⑤研究情境主要是西方人际伦理，可能不适用于结构性或跨文化情境。

---

## 349. TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning

**arXiv ID:** 2606.05878 | [PDF](https://arxiv.org/pdf/2606.05878v1)

**作者:** Etienne Le Naour `[一作]` (EDF Research and Development), Adrien Petralia `[通讯]` (EDF Research and Development)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种概率性 In‑Context Learning 编码‑回归 Transformer（称为“ICL‑TSFM”），将时间序列预测统一为时间戳对齐的回归问题，能够在同一模型下同时完成缺失值插补和预测，并自然支持协变量与不规则采样；

**💡 创新点**

创新点在于：①将时间序列表述为可插值的回归任务；②采用基于 DAG 的合成因果先验生成协变量依赖结构；③使用时间索引化的上下文查询模块实现对任意时间点的可查询，打破传统固定格点的限制；

**🔧 技术方法**

主要技术包括：Perceiver 风格的时间序列编码器、通道混合器、频率编码的时间上下文查询、以及基于 Transformer 的 In‑Context 学习回归器；

**📊 数据集**

使用包含 31 个真实多领域数据集（能源、气候等）和大量由生成器产生的合成单变量序列（约 2M 条）作为预训练数据；同时在基准上使用公共零样本插补与预测数据集（如“Zero-shot imputation benchmark”和“Zero-shot forecasting benchmark”）进行评估；

**📈 对比分析**

与 TSFMs、TFMs、局部插值方法和监督式插补模型相比，ICL‑TSFM 在零样本插补任务上实现了新的最佳点估计和概率分数（NMAE/CRPS），比 TFMs 快约 50 倍；在零样本预测任务中，性能与领先 TSFMs 差距不到 6%，且在缺失历史观测下表现更稳健；

**⚠️ 局限性**

局限性主要体现在：相比高度优化的 Patch‑based TSFMs，ICL‑TSFM 的推理成本更高（约 4 倍），主要原因是点对点回归的计算开销，可通过缓存或混合精度等结构优化进一步提升。

---

## 350. LadderMan: Learning Humanoid Perceptive Ladder Climbing

**arXiv ID:** 2606.05873 | [PDF](https://arxiv.org/pdf/2606.05873v1)

**作者:** Siheng Zhao `[一作]` (Amazon FAR), Guanya Shi `[通讯]` (Amazon FAR)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 LadderMan 系统，使 Unitree G1 类人机器人在零 shot sim-to-real 下能够稳健爬梯并执行在梯上操作。

**💡 创新点**

创新点在于：①两阶段学习管线，先用混合运动跟踪从单一参考动作学习多种专家；②将专家蒸馏成单一基于深度的视觉运动策略，采用混合模仿+强化学习；③利用视觉基础模型桥接深度感知的 sim-to-real 问题；④双代理学习实现稳定的在梯上操作。

**🔧 技术方法**

技术包括：混合运动跟踪、Hybrid DAgger + PPO 训练、Fast-FoundationStereo 深度估计、梯子中心化掩模、IsaacSim 物理仿真、双代理 teleoperation、RFM、domain randomization。

**📊 数据集**

使用单一参考运动（OptiTrack 捕捉）、AMASS 数据集生成上肢目标姿态、IsaacSim 合成梯子环境、RealSense D435i 深度摄像头。

**📈 对比分析**

与盲目运动跟踪基线、TWIST2 传统全身 teleop 以及人类攀爬速度对比。LadderMan 在 95%+ 成功率的梯子配置上表现优异，零 shot 实际硬件测试成功率高；攀爬速度约 3.4 秒/段，接近人类平均 3.2 秒/段。

**⚠️ 局限性**

局限性包括：仅在 75° 以下倾斜梯子上验证，未覆盖竖直梯；缺乏灵巧末端执行器，限制了更复杂的操作；对极端梯子几何或材料仍有失败风险。

---

## 351. YouZhi: Towards High-Concurrency Financial LLMs via Adaptive GQA-to-MLA Transition

**arXiv ID:** 2606.05868 | [PDF](https://arxiv.org/pdf/2606.05868v1)

**作者:** PSBC LLM Team `[一作]` (Postal Savings Bank Of China), Xinzhuang Niu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种层级自适应的 GQA→MLA 转换和基于华为 Ascend 的两阶段后训练管线，以显著压缩 KV 缓存并提升金融 LLM 的并发性能。

**💡 创新点**

通过层级自适应 FreqFold 选择降低 GQA→MLA 过程中的困惑度损失，并结合知识蒸馏与金融特定 SFT 恢复并强化模型能力。

**🔧 技术方法**

使用层级自适应 GQA2MLA 转换、FreqFold、Generalized Knowledge Distillation、金融特定 SFT、以及 vLLM‑Ascend 推理框架。

**📊 数据集**

使用 WikiText‑2、C‑Eval、IFEval、MATH‑500、LCB、H‑Swag、SST‑5、CrossNER、CFLUE、FinanceIQ、FinEval、FBP、OpenFinData 等通用与金融基准数据集，以及 970k 规模的金融指令‑响应对。

**📈 对比分析**

与 TransMLA 及原始 GQA 模型对比，在 WikiText‑2 上层适应方法减少约35% 的 perplexity；在金融基准上提升 7–10% 的准确率；在 Ascend NPU 上 KV 缓存减少 72%，最大并发提升 2.69×，吞吐率提升 1.76×。

**⚠️ 局限性**

仍需要手工调参的层级 FreqFold 选择、对更大规模模型的可扩展性未充分验证、缺乏跨平台（非 Ascend）验证，以及对非金融领域任务的泛化能力有限。

---

## 352. GenAutoML: An Agentic Framework for Dynamic Architecture Generation and Optimization in Time-Series Analysis

**arXiv ID:** 2606.05860 | [PDF](https://arxiv.org/pdf/2606.05860v1)

**作者:** Oleeviya Babu Poikarayil `[一作]` (Paul Wurth S.A.), Jawid Ahmad Baktash `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `39fd911c-56a4-425d-a2f9-8038ad3b6e21`

**🎯 论文内容**

构建了 GenAutoML 框架，利用 LLM 自动生成、验证并部署针对时间序列预测与异常检测的轻量化神经网络架构，专为边缘 AI 设计。

**💡 创新点**

创新点包括：① 将 LLM 作为“神经架构师”进行语义到代码的映射；② 沙盒反射循环实现代码自动调试；③ 动态可逆实例归一化（Dyn‑RevIN）实现对非平稳数据的统计硬化；④ JIT 动态注入和签名感知运行时提升模型热插拔与稳定性。

**🔧 技术方法**

核心技术：大型语言模型（Llama 3‑70B）、LangChain 对话驱动、Optuna 超参搜索、沙盒反射调试、Dyn‑RevIN、签名感知推理、Shape‑agnostic 投影层、PyTorch JIT 注入。

**📊 数据集**

使用公开时间序列数据集 ETTh1、ETTm1 与 Weather 进行实验，并在这些数据上进行预测与异常检测。

**📈 对比分析**

与传统 LSTM、Conv1D、DLinear、iTransformer、TimesNet、CrossFormer 等基线以及零射击的 Chronos‑T5‑Mini 进行对比；实验显示：生成的 ResNet / Inception 与 Transformers 竞争力相当，WaveInterferenceNet 仅 <0.01 ms 推理延迟，性能提升约 100 000×，且在异常检测中获得最高的判别率。

**⚠️ 局限性**

局限性：LLM 生成延迟高、依赖外部 API；Optuna 超参搜索仍耗时；缺乏本地小型 LLM 方案；目前仅支持数值时序，未兼顾多模态输入；在高变动数据集上 Dyn‑RevIN 可能抑制周期性信号。

---

## 353. TARPO: Token-Wise Latent-Explicit Reasoning via Action-Routing Policy Optimization

**arXiv ID:** 2606.05859 | [PDF](https://arxiv.org/pdf/2606.05859v1)

**作者:** Liting Zhang `[一作]` (Nankai University), Qicheng Li `[通讯]` (Nankai University)

**通讯引用:** 6430 | [OpenAlex ID](https://openalex.org/A5001810544)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种全新的纯强化学习框架TARPO，可在每个Token步动态在离散生成和连续潜在推理之间切换

**💡 创新点**

创新点在于将推理模式选择视为可学习的动作路由策略，并通过轻量级路由头实现Token级的自适应模式切换，从而在保持离散采样随机性的同时引入连续潜在推理的表达力

**🔧 技术方法**

使用了Transformer LLM骨干、轻量级线性路由头、共享group-relative advantage信号、RL梯度优化、KL正则化以及soft-token的top‑k加权混合表示

**📊 数据集**

在Qwen2.5系列（1.5B/3B/7B）和Llama‑3.1‑8B骨干上，评估了GSM‑8K、MATH、MATH500、AMC23、Olympiad等数学推理数据集，并在GPQA‑Diamond、ARC‑C、HumanEval等OOD数据集上做了泛化测试

**📈 对比分析**

与基准CoT、Pure Latent、Entropy‑Routed、GRPO、HRPO及Soft‑Tokens等方法对比，TARPO在Pass@1/Pass@32等指标上均有提升，且在token效率和训练稳定性方面优于其他RL基线

**⚠️ 局限性**

局限性包括：实验规模仅至8B参数，未验证更大规模的可扩展性；潜在表示本身仍为确定性，缺乏代表层面的探索，未来可结合Gumbel‑Softmax或噪声注入以增强表征随机性

---

## 354. Forgive or forget: Understanding the context of hate in audio retrieval systems

**arXiv ID:** 2606.05857 | [PDF](https://arxiv.org/pdf/2606.05857v1)

**作者:** Arghya Pal `[一作]` (Monash University), Shekhar Nayak `[通讯]` (University of Groningen)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5001110193)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了后处理的因果去偏框架，包含 Forget（对数平均的对抗性有毒变体）和 Forgive（音频级重排序）两种策略，以降低文本到音频检索中的有毒内容同时保持语义相关性。

**💡 创新点**

创新点在于将前门因果调整与情感控制中介相结合，通过生成六类有毒语义变体来暴露并抑制模型的毒性偏差，同时提供模型无侵入式的后处理方法，兼顾整体与细粒度的安全性。

**🔧 技术方法**

使用了前门因果去偏、情感控制中介生成有毒变体（LLM）、Noise2Noise 对数平均、Silero ASR 转写、Detoxify 毒性检测、软max 重新排序以及模型无侵入式的后处理管道。

**📊 数据集**

评估数据集包括 AUDIOCAPS、CLOTHO；实验模型为 ATNLL、TUAR、WavCaps 三种先进的文本到音频检索模型。

**📈 对比分析**

与基线、单独的 Logit 调整、单独的对抗式提示和两者组合进行比较。使用 Success Rate、Accuracy、Sensitivity 三个新指标衡量毒性抑制与检索质量。组合 Forget+Forgive 在所有 Top@K（5、10、15、20）下均显著提升三项指标，毒性下降显著而检索准确率保持甚至提升。

**⚠️ 局限性**

局限性包括：LLM 生成的有毒变体质量不稳定，对长音频或极低噪声环境的 ASR 可能影响检测；Forgetting 需要多次推理，运行时开销相对较大；仅在 Top@K 候选上进行 ASR 过滤，可能漏检非 Top@K 的毒性音频。

---

## 355. Gender Artifacts from Art History to Text-to-Image Generation

**arXiv ID:** 2606.05829 | [PDF](https://arxiv.org/pdf/2606.05829v1)

**作者:** Piera Riccio `[一作]` (University of Amsterdam), Nanne van Noord `[通讯]` (University of Amsterdam)

**通讯引用:** 583 | [OpenAlex ID](https://openalex.org/A5081881122)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了StyleGender数据集，系统研究艺术风格与性别表征的相互作用，并提出PixelSGA与MaskSGA两种量化指标。

**💡 创新点**

创新点包括：①首个涵盖19种艺术风格、约74k张图像的性别与风格交互数据集；②引入PixelSGA与MaskSGA两种新的性别痕迹度量；③发现文本到图像模型在生成过程中放大了历史作品中的性别痕迹。

**🔧 技术方法**

技术手段包括：使用Stable Diffusion 3.5 Medium与Flux.1-dev进行图像生成；利用CLIP/SigLIP嵌入及kNN分类器进行性别识别；对图像进行降采样、颜色编码转换以及遮罩操作以计算PixelSGA与MaskSGA。

**📊 数据集**

数据来源为：18k条来自WikiArt的历史绘画（19种风格、男女二元标签）；19k条使用上述两种T2I模型按性别和风格关键词生成的图像；19k条与历史图像语义对齐的生成图像。

**📈 对比分析**

对历史图像、简单提示生成图和语义对齐生成图分别计算PixelSGA与MaskSGA；结果显示所有风格均高于随机基准，Neoclassicism与Art Nouveau得分最高；生成模型的PixelSGA普遍高于历史图像，表明性别痕迹被放大；两模型间相关性高，Monotonicity分数接近1，验证度量的稳定性。

**⚠️ 局限性**

局限性包括：指标依赖kNN与CLIP的分类性能，可能受嵌入模型限制；遮罩检测在高度抽象或非人类主体的作品中不稳健；风格标签可能存在噪声与西方偏见；二元性别标注可能强化刻板印象，且生成文本描述可能包含模型偏差。

---

## 356. From Risk Classification to Action Plan Remediation: A Guardrail Feedback Driven Framework for LLM Agents

**arXiv ID:** 2606.05805 | [PDF](https://arxiv.org/pdf/2606.05805v1)

**作者:** Yuhao Sun `[一作]` (University of Melbourne), Xingliang Yuan `[通讯]` (University of Melbourne)

**通讯引用:** 3262 | [OpenAlex ID](https://openalex.org/A5064553444)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种三方决策的 LLM 代理安全防护框架（Tri-Guard），在代理的规划阶段对每一步计划进行安全评估，并提供结构化自然语言反馈，指导代理在执行前进行计划修正或拒绝。

**💡 创新点**

创新点包括：① 将安全评估与代理规划紧密耦合，形成闭环；② 采用 Proceed/Update/Refuse 三种决策而非传统的允许/拒绝；③ 通过知识蒸馏生成的轨迹‑反馈数据进行监督微调，让模型既能检测风险又能给出可操作的修正指引。

**🔧 技术方法**

核心技术为 ReAct 代理框架、基于 Qwen3.5‑9B 的 guardrail 微调、结构化自然语言反馈模板、三方决策逻辑以及与代理的交互 ICL 模板；训练采用加权监督微调（wSFT）。

**📊 数据集**

使用了 Agent Security Bench (ASB) 与 AgentHarm 两个安全基准；在 ASB 上构造了 5288 条轨迹数据并用 GPT‑5.4 进行反馈与决策的蒸馏，形成用于微调的轨迹‑反馈对。

**📈 对比分析**

与传统的 ReAct、ToolSafe、TS‑Guard 等基线相比，Tri‑Guard 在 ASB 的攻击成功率 (ASR) 下降到 10.42%（从 74.45% 降低），任务成功率 (TSR) 提升至 68.60%（从 28.45% 提升），在 AgentHarm 上帮助度‑安全评分 (HS) 提升至 80.92，显著兼顾安全与效用。

**⚠️ 局限性**

主要局限包括：① 额外的 guardrail 推理和计划修正导致推理延迟；② 目前仅使用 9B 模型，模型规模与数据量有限；③ 评估聚焦于 Prompt Injection 与直接有害任务，未涵盖更复杂或新颖攻击，需要进一步验证泛化能力。

---

## 357. Can LLMs Be Constrained to the Past? Improving Knowledge Cutoff through Recall-Based Prompting

**arXiv ID:** 2606.05804 | [PDF](https://arxiv.org/pdf/2606.05804v1)

**作者:** Michiro Asai `[一作]` (Institute of Science Tokyo), Manabu Okumura `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 5904 | [OpenAlex ID](https://openalex.org/A5035876897)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了自回忆（SR）和问题回忆（QR）两种基于回忆的提示策略，用于在LLM中强制知识截止，并创建了Multi-cutoff Historical Event Benchmark（MHEB）评估不同截止时间的鲁棒性。

**💡 创新点**

创新点在于将截止条件转化为模型自我声明的内部状态（SR），并通过问题相关事实与时间信息的回忆（QR）来锚定检索，二者组合进一步提升性能；同时构建多截止时间评测数据集，系统评估截止效果随时间距离变化的表现。

**🔧 技术方法**

采用提示工程，设计SR、QR以及其组合SR→QR的提示模板；通过在GPT‑4o、gpt‑oss‑120b、Llama‑3.3‑70B‑Instruct等模型上进行零样本推理，比较不同提示策略的知识截止成功率。

**📊 数据集**

使用三大现有知识截止基准（Factual、Semantic、Counterfactual）以及新构建的900条历史事件的MHEB数据集。

**📈 对比分析**

与基线P1、零样本链式思维（ZS‑CoT）和计划与解决（ZS‑PS）等方法对比。SR在所有模型和数据集上均超过P1，尤其在Counterfactual上显著提升；SR→QR在三大基准和MHEB上均取得最高成功率（如Llama‑3.3‑70B‑Instruct在MHEB上各截止偏移下平均成功率达约89%）。

**⚠️ 局限性**

实验仅关注提示方法，未探索需要额外数据微调的去学习技术；所有提示与基准均为英文，未验证多语言环境的适用性。

---

## 358. Compositional Boundaries for Density Fusion

**arXiv ID:** 2606.05871 | [PDF](https://arxiv.org/pdf/2606.05871v1)

**作者:** Ratan Bahadur Thapa `[一作]` (University of Stuttgart), Steffen Staab `[通讯]` (University of Stuttgart)

**通讯引用:** 27925 | [OpenAlex ID](https://openalex.org/A5062807811)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文把分布加权融合问题视为代数组合问题，分析了在二元分段规则下，怎样保证分布融合的顺序不变。

**💡 创新点**

创新点在于给出了局部段式融合规则的组合边界，证明只有归一化线性池化（以及通过变换权重得到的等价形式）才能在所有加权分布类中实现关联性与交换性；同时揭示了f‑散度平衡与高斯混合压缩的组合障碍。

**🔧 技术方法**

采用代数函数方程、连续性与关联性证明、f‑散度泰勒展开、以及无归一化成分测度的幺半群同构等技术。

**📊 数据集**

该工作为理论研究，无使用公开实验数据集；若需验证可在人工生成的高斯混合或伯努利分布上测试。

**📈 对比分析**

通过与全局f‑barycenter、对数池化等全局聚合方法的理论对比，证明局部二元平衡在顺序独立性上存在本质局限；实验方面未给出具体性能指标。

**⚠️ 局限性**

局限性包括：仅适用于连续段值且权重仅取决于输入权重的规则；f‑散度端点平衡、局部压缩策略在一般情况下不具备关联性；对高维、非凸分布或不满足 bounded‑ratio 条件时结果不一定成立。

---

## 359. QCFuse: Query-Aware Cache Fusion via Compressed View for Efficient RAG Serving

**arXiv ID:** 2606.05875 | [PDF](https://arxiv.org/pdf/2606.05875v1)

**作者:** Jianxin Yan `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35498 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出QCFuse压缩视图的查询感知缓存融合选择器，改进RAG KV缓存复用。

**💡 创新点**

将查询信息压缩为每个chunk的anchor和仅关注关键层，既保持查询感知，又避免全视图选择导致的流水线阻塞。

**🔧 技术方法**

chunk-anchor query probing、critical-layer profiling、SGLang实现、Triton稀疏KV重算等。

**📊 数据集**

LongBench（MuSiQue、2WikiMQA、HotpotQA）和RULER（multi-query、multi-value、variable-tracking）等。

**📈 对比分析**

与全预填、直接PIC重用以及CacheBlend、EPIC、FusionRAG、ProphetKV等基线在质量-TTFT曲线和吞吐量上对比，QCFuse在保持全预填质量的前提下，TTFT平均提升1.7×，相对ProphetKV提升1.5×，吞吐量亦较高。

**⚠️ 局限性**

仍需要对anchor比例和关键层进行模型特定的离线调优，对极端长上下文或不同检索方式的适应性尚未完全验证，且在极低带宽情况下仍受限于KV加载。

---

## 360. Analysis of the Neglect-Zero Effect in Large Language Models

**arXiv ID:** 2606.05864 | [PDF](https://arxiv.org/pdf/2606.05864v1)

**作者:** Jin Tanaka `[一作]` (University of Tokyo), Hitomi Yanaka `[通讯]` (University of Tokyo)

**通讯引用:** 301 | [OpenAlex ID](https://openalex.org/A5045824013)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）是否会出现人类认知偏差——忽视零模型效应（neglect-zero effect），并通过结构性诱导（structural priming）实验评估其推理机制。

**💡 创新点**

提出了将人类认知实验设计（结构性诱导+图片匹配任务）转化为文本提示的框架，首次将该框架应用于多种LLM进行系统对比。

**🔧 技术方法**

使用了结构性诱导实验、图片匹配任务转文本提示、GLMM统计分析，以及对六个开源/闭源LLM（Gemma‑3‑27B、Gemma‑3‑1B/4B/12B、Llama‑4‑Scout‑17B、GPT‑5 nano）的评估。

**📊 数据集**

采用了之前人类实验的图片匹配数据集（380条prime/target + 250条filler），并通过Python脚本自动转换为文本提示。

**📈 对比分析**

对比结果显示，Gemma‑3‑27B、Llama‑4 与 GPT‑5 nano 在结构性诱导下未表现出忽视零模型效应；Gemma‑3‑27B 与 Llama‑4 对零模型敏感但方式与人类不同；GPT‑5 nano 对零模型基本无反应，倾向于文字字面推理。

**⚠️ 局限性**

局限性在于仅测试了少量模型，无法充分解释Gemma‑3‑27B与GPT‑5 nano与Llama‑4 的差异；实验仅基于一种图像‑文本转化方法，未来需扩大模型样本并探究更多实验设置。

---

## 361. ReverseEOL: Improving Training-free Text Embeddings via Text Reversal in Decoder-only LLMs

**arXiv ID:** 2606.05858 | [PDF](https://arxiv.org/pdf/2606.05858v1)

**作者:** Ailiang Lin `[一作]` (Institute of Science Tokyo), Manabu Okumura `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 5904 | [OpenAlex ID](https://openalex.org/A5035876897)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为ReverseEOL的方法，通过在原始文本顺序和反向文本顺序上分别生成嵌入并平均，提升冻结LLM的训练免费文本嵌入质量。

**💡 创新点**

创新点在于利用输入文本的逆序来弥补因因果注意力导致的上下文缺失，从而提供互补语义信息，而无需内部模型改造或复杂的提示工程。

**🔧 技术方法**

方法核心技术包括文本反转、基于单词限制的PromptEOL等提示模板，以及在冻结LLM中提取最后一词隐藏状态并进行平均融合。

**📊 数据集**

在STEM和MTEB两大基准上进行评测，涵盖7个STS数据集（STS-2012~2016、STS-B、SICK-R）以及MTEB的5类任务（分类、成对分类、重排序、聚类、检索），共10+种LLM模型。

**📈 对比分析**

与多种训练免费基线（PromptEOL、MetaEOL、ECHO、Pretended CoT、Knowledge等）以及内部干预方法（Contrastive Prompting、Token Prepending）进行对比，ReverseEOL在STS平均提升约3.8分、在MTEB平均提升约5.3分，显著优于其他方法。

**⚠️ 局限性**

局限性包括仍无法匹敌专门训练的嵌入模型、对多语言扩展不足以及模型固有偏差和幻觉可能在嵌入中传播。

---

## 362. UniVoice: A Unified Model for Speech and Singing Voice Generation

**arXiv ID:** 2606.05852 | [PDF](https://arxiv.org/pdf/2606.05852v1)

**作者:** Junjie Zheng `[一作]` (Giant Network), Zihao Chen `[通讯]` (Giant Network)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的语音与歌声合成框架，使用条件流匹配与Diffusion Transformer实现单一模型同时生成自然语音和可控歌声。

**💡 创新点**

核心创新在于把条件分解为内容、旋律和音色三部分，并为语音引入学习到的“空旋律”标记，既保留歌声的旋律控制，又避免在语音中强制加旋律约束；同时采用任务令牌通过AdaLN实现不同模式的轻量化调节。

**🔧 技术方法**

技术手段包括Conditional Flow Matching（CFM）、Diffusion Transformer（DiT）骨干、Melody/Content/Timbre Encoder、Learned Null Melody Token、轴选择式分类器自由引导、FlashAttention‑2与RoPE、Song Bloom VAE等。

**📊 数据集**

使用约65k小时混合语音与歌声数据（30k小时语音、35k小时歌声），包括多语言（中英）语音数据和从真实歌曲中提取的旋律/歌词对应的MIDI序列，并配合音色参考音频进行零样本声纹克隆。

**📈 对比分析**

与专用TTS系统（F5‑TTS、CosyVoice3）和统一基线（Vevo1.5、Soul‑X‑Singer）对比，单模型0.3B参数在语音PER 5.26%（接近专用TTS）和歌声PER 16.22%（显著优于Vevo1.5 45.07%）上均表现突出；人类评测S‑MOS/N‑MOS和SIM也与基线竞争，证明统一训练在保持质量的同时实现跨模态一致性。

**⚠️ 局限性**

局限性包括：统一训练导致与单模态模型相比在声纹相似度(SIM)上略有下降；多语言与低资源场景支持不足；模型未专门处理像rap或说唱等混合语调；32步ODE采样产生一定推理延迟；以及潜在的深度伪造与版权风险。

---

## 363. Towards Worst-case Hardness for Low-Noise LPN

**arXiv ID:** 2606.05834 | [PDF](https://arxiv.org/pdf/2606.05834v1)

**作者:** Divesh Aggarwal `[一作]` (National University of Singapore), Prashant Nalini Vasudevan `[通讯]` (National University of Singapore)

**通讯引用:** 479 | [OpenAlex ID](https://openalex.org/A5032319697)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文探讨了低噪声学习奇偶性（LPN）问题的最坏情况难度，提出了一种新的方法来实现从最坏情况到平均情况的减少，旨在提高LPN在公钥加密中的应用。

**💡 创新点**

创新点在于放宽了对生成矩阵行的随机稀疏组合的要求，转而要求其在计算上与均匀分布不可区分，从而实现了LPN的平均情况难度与最坏情况解码的关联。

**🔧 技术方法**

使用了线性代数和编码理论中的解码技术，特别是对双重代码的解码和区分问题的研究。

**📊 数据集**

使用了随机生成的线性代码矩阵和相应的噪声向量，具体数据集未详细说明，但涉及到的噪声率和样本数量是关键参数。

**📈 对比分析**

与现有方法相比，本文的方法在噪声率达到n^-α（α<1）时，能够实现LPN的平均情况难度，特别是在公钥加密所需的参数范围内，性能显著提升。

**⚠️ 局限性**

限制在于当前的最坏情况假设较强，且在处理特定噪声模型时，可能需要新的分析工具来证明其有效性。

---

## 364. When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents

**arXiv ID:** 2606.05806 | [PDF](https://arxiv.org/pdf/2606.05806v1)

**作者:** Dongsheng Zhu `[一作]` (Shanghai AI Laboratory), Dawei Yin `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ToolMaze 基准，针对 LLM 的工具集成推理（TIR）在真实世界工具失效场景下的动态路径发现与错误恢复进行系统评估。

**💡 创新点**

创新点包括：① 两维评估框架（DAG 复杂度 × 四种工具失效模式），② 通过完整恢复路径的地面真值实现对主动重规划与盲目搜索的精准区分，③ 引入 Perturbation Recovery Rate (PRR) 与 Recovery Cost (RC) 两个新指标，用以衡量恢复成功率与重规划效率。

**🔧 技术方法**

技术手段包括 DAG 生成与验证、工具语义一致性检查、故障注入引擎（按预设规则注入显式/隐式、短暂/永久错误）、自然语言任务化流程、两种提示策略（标准 vs failure‑aware），以及多模型评估（Open‑weight 与专有 LLM）。

**📊 数据集**

使用了 270 个手工构建、按功能与业务域标注的工具集合，生成 400 个基础任务（100 个对应 C1–C4 复杂度）并在每个任务上注入 4 种失效模式，形成 2000 条评测实例。

**📈 对比分析**

方法：在 9 个 LLM（包括 6 个开放权重模型和 3 个专有模型）上使用 TSR、PRR、RC 三个指标进行评估；结果显示所有模型在带失效模式时性能显著下降，failure‑aware 提示可提升 1.5–20.8%；Gemini‑3.1‑Pro‑Preview 在综合指标上排名第一；模型规模提升对 PRR 的增益远慢于 TSR，表明动态恢复是独立的能力。

**⚠️ 局限性**

局限性：① 评估仅基于结构化 DAG 任务，未覆盖开放式 Web 或高度不确定的工作流；② 失效模式仅覆盖 2×2 基本范畴，未考虑级联失效、恶意注入等更复杂情形；③ 任务生成虽保证语义一致，但仍可能缺乏真实世界的复杂性与多样性。

---

## 365. Statistical Priors for Implicit Preferences: Decoupling Skill Selection as a Local Harness in Personal Agents

**arXiv ID:** 2606.05828 | [PDF](https://arxiv.org/pdf/2606.05828v1)

**作者:** Zeyu Gan `[一作]` (Renmin University of China), Yong Liu `[通讯]` (Renmin University of China)

**通讯引用:** 20905 | [OpenAlex ID](https://openalex.org/A5100724297)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Local Harness 框架，在本地部署的个人代理中将统计偏好学习与语义意图解析解耦，使用轻量级本地统计模块做默认决策，远程 LLM 仅用于语义覆盖。

**💡 创新点**

通过物理和逻辑上严格分离统计偏好学习与语义推理，提出本地优先、LLM 覆盖的双阶段决策流程，并构建了首个专用评估基准 ToolBench-60。

**🔧 技术方法**

使用域分类、基于频率与 LinUCB 的本地统计先验、特征哈希、UCB 探索以及单轮 LLM 语义覆盖探测。

**📊 数据集**

构建 ToolBench-60 评估环境，包含 60 个技能、10 个域；对多种用户偏好分布（one-hot、Dirichlet）进行实验。

**📈 对比分析**

与 9 种基线（随机、ZeroShot-LLM、纯统计、记忆增强 LLM 等）在 GPT-5.2、DeepSeek-V4-Flash、Qwen3-30B-Instruct 三种后端上比较；Local Harness 在累计遗憾最低、测试准确率最高，Bandit-as-Override 表现最优。

**⚠️ 局限性**

仅针对静态用户偏好、即时二元奖励；使用哈希特征限制表示能力；依赖强大远程 LLM；缺乏对非平稳情境和噪声反馈的评估。

---

## 366. Causal Longitudinal Prior-Fitted Networks for Counterfactual Outcome Prediction

**arXiv ID:** 2606.05797 | [PDF](https://arxiv.org/pdf/2606.05797v1)

**作者:** Amirhossein Zare `[一作]` (Tehran University of Medical Sciences), Mohammad Kashkooli `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种先验拟合网络（Prior-Fitted Network，PFN）模型，专门用于在长期随访数据中预测给定未来治疗方案下的潜在结果，并在推理时保持模型冻结，完全不依赖目标域的梯度更新或倾向性建模。

**💡 创新点**

创新点在于：
1) 设计了覆盖多种时间结构因果模型（Temporal Structural Causal Models，TSCM）的合成先验，生成包含时间相关性、非线性动力学、隐藏异质性、反馈与累计效应的多样化长期任务；
2) 将PFN与因果Transformer相结合，构建三分支架构（历史编码器、上下文编码器、Gaussian混合预测头），实现一次步预测的无监督自适应；
3) 通过递归一次步预测实现多步潜在结果滚动，提供零样本的长期因果推断能力。

**🔧 技术方法**

主要技术包括：
- Prior-Fitted Networks (PFN) 进行在情境预测；
- 因果Transformer 进行时间序列编码；
- Gaussian混合分布头用于生成预测分布；
- 通过合成TSCM预训练实现任务迁移；
- 递归一次步预测实现多步潜在结果的滚动。

**📊 数据集**

使用的主要数据集：
1) 计算机模拟/半机械化的癌症肿瘤生长、Warfarin PK/PD、HIV 治疗动力学三类可分支的因果实验数据；
2) 真实临床 ICU 数据 MIMIC-III，用于事实滚动预测。

**📈 对比分析**

比较方法：MSM、RMSN、G-Net、CRN、Causal Transformer、G-Transformer。实验结果显示：
- 在四个基准的域平衡一阶预测中取得最优的规范化 RMSE（0.2217）；
- 在五步预测中排名第三；
- 在 MIMIC-III 的真实事实预测（一阶和五步）中获得最佳性能，证明其在无目标域训练下的强大迁移能力。

**⚠️ 局限性**

局限性：
- 仍依赖一致性、正向性、连续可交换等因果假设，未能消除未观测混杂、治疗重叠不足、删失或不规则采样带来的偏差；
- 合成先验的覆盖范围有限，当目标域的动力学、治疗策略、缺失模式或干预效应超出先验支持时性能可能下降；
- 当前实现仅处理离散固定时间步、确定性滚动预测，未涵盖连续治疗空间、时间不规则性、显式缺失/删失建模或不确定性传播等复杂情况；
- 过度信任模型输出的风险，尤其是在真实临床设置中对未观测干预的个体因果效应仍需谨慎解读。

---

## 367. Towards Truly Multilingual ASR: Generalizing Code-Switching ASR to Unseen Language Pairs

**arXiv ID:** 2606.05846 | [PDF](https://arxiv.org/pdf/2606.05846v1)

**作者:** Gio Paik `[一作]` (Theta One Korea), Soungmin Lee `[通讯]` (AeryAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在有限语言对的代码切换ASR中学习到的能力能否迁移到未见语言对，并评估模型合并与域泛化方法的有效性

**💡 创新点**

系统性探讨模型合并与域泛化在CS-ASR中的迁移效果，首次公开韩日与韩德代码切换评测集，并揭示当前方法的局限性

**🔧 技术方法**

使用 Whisper‑medium 作为基准模型，采用细调、Task Arithmetic、TIES、DARE 等模型合并技术，以及 Fish、Fishr、GGA‑L 等域泛化方法，并进行层级 MAV 分析

**📊 数据集**

使用公开的 ko‑en、ja‑en、de‑en 代码切换语料库；自建 ko‑ja 与 ko‑de 评测集（分别包含 450 与 387 条语音样本）

**📈 对比分析**

通过 Mixed Error Rate（MER）进行比较，单对细调对见/未见对均有提升，但仅能将未见对 MER 降到约 0.32；模型合并在未见对上略优，域泛化提升有限

**⚠️ 局限性**

未见对性能仍偏高，数据量与多样性不足，未验证对全新语言对的迁移，且仅在 Whisper‑medium 上实验

---

## 368. GLASS: GRPO-Trained LoRA for Acoustic Style Steering in Zero-Shot Text-to-Speech

**arXiv ID:** 2606.05889 | [PDF](https://arxiv.org/pdf/2606.05889v1)

**作者:** Jaehoon Kang `[一作]` (Sungkyunkwan University), Kyuhong Shim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 220 | [OpenAlex ID](https://openalex.org/A5064051041)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

GLASS框架在零样本TTS中通过奖励引导训练轻量LoRA适配器，实现可组合的声学风格控制（速度与音高）。

**💡 创新点**

创新点在于使用后生成奖励（WER、语音长度、平均F0）与GRPO训练单个LoRA适配器，从而无需风格标签即可学习可交换、可插值、可组合的风格方向。

**🔧 技术方法**

采用GRPO强化学习、LoRA参数高效适配、权重算术插值、Speech-token长度与平均F0奖励、WER与ASR等技术。

**📊 数据集**

训练使用LibriTTS-R多说话人提示，评估在Seed-TTS-eval（1088句）上。

**📈 对比分析**

与未改CosyVoice2-0.5B及DSP时域拉伸/音高移位基线对比，LoRA在保持自然度、说话人相似度、可懂度的同时实现速度和音高目标变换，且可平滑插值和多轴组合，性能优于DSP基线。

**⚠️ 局限性**

仅覆盖可测量的两个声学轴（语速、音高），缺乏情感、口音等更丰富风格；奖励信号依赖自动代理指标，可能不完全反映人类感知。

---

## 369. When Denser Credit Is Not Enough: Evidence-Calibrated Policy Optimization for Long-Horizon LLM Agent Training

**arXiv ID:** 2606.05885 | [PDF](https://arxiv.org/pdf/2606.05885v1)

**作者:** Yuanfan Li `[一作]` (Shanghai Jiao Tong University), Lu Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 17024 | [OpenAlex ID](https://openalex.org/A5100432103)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Evidence‑Calibrated Policy Optimization（ECO），通过对多轮回合中重复出现的状态（anchor）进行行动级证据校准与方差门控，以稳定长期 LLM 代理的强化学习；

**💡 创新点**

创新点在于将步骤级奖励视为统计证据问题，采用 Evidence‑Calibrated Action Advantage（ECA）对低样本行动进行缩减估计，并用 Variance‑Gated Credit Weighting（VarGate）对 anchor 的可靠性进行门控，从而消除“divergent anchor bias”；

**🔧 技术方法**

使用无价值函数的分组强化学习框架（GRPO / GiGPO），结合 shrinkage 估计、方差分解和裁剪目标的 PPO 核心；

**📊 数据集**

在 ALFWorld 和 WebShop 两大长期任务上使用 Qwen2.5‑1.5B/7B‑Instruct 作为基线模型；

**📈 对比分析**

与闭源 LLM、提示式代理、PPO、RLOO、GRPO、GiGPO 等基线比较，ECO 在 Qwen2.5‑1.5B 上分别提升 ALFWorld 整体成功率 5.2% 与 WebShop 7.3%；在 7B 上同样取得最高成功率；相对 GiGPO 减少最终奖励方差并仅增加 0.1% 计算开销；

**⚠️ 局限性**

局限性在于依赖可重复 anchor 状态，若环境中状态重复稀缺或难以标准化，ECO 会退回到轨迹级奖励；实验范围局限于 ALFWorld、WebShop 与 SearchQA，缺乏更广泛多工具或多代理场景验证。

---

## 370. Exploring cooperation mechanisms via reinforcement learning in network common-pool resource games

**arXiv ID:** 2606.05867 | [PDF](https://arxiv.org/pdf/2606.05867v1)

**作者:** Yihang Qin `[一作]` (Shanghai Jiao Tong University), Lin Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 473907 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出网络化公共池资源博弈模型，并用图神经网络强化学习学习自适应分配策略

**💡 创新点**

将分配机制从手工规则迁移到可解释的混合机制，兼顾资源状态与网络结构

**🔧 技术方法**

图神经网络+深度强化学习（TD3）、两层GraphNet、双重评估器

**📊 数据集**

仿真数据：N=50，四种网络拓扑（Regular、ER、BA、WS），平均度4，20000步演化

**📈 对比分析**

与等分配和按贡献分配基线比较；RL代理在所有拓扑上均实现更高合作率、平均资源更高、Gini系数更低；可解释的M1、M2机制在新图上保持鲁棒性

**⚠️ 局限性**

对未知网络和长周期演化的鲁棒性有限；只使用固定Fermi更新规则；未验证在人类实验中的有效性

---

## 371. Emotion-Aware Image Generation from Korean Diary Text via LLM-based Prompt Translation and LoRA Fine-Tuning

**arXiv ID:** 2606.05816 | [PDF](https://arxiv.org/pdf/2606.05816v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 372. Visuotactile and Explicitly Force-Controlled Robotic Ultrasound for Abdominal Volumetric Reconstruction

**arXiv ID:** 2606.05848 | [PDF](https://arxiv.org/pdf/2606.05848v1)

**作者:** Adrian Piedra `[一作]` (Stanford University), Oussama Khatib `[通讯]` (Stanford University)

**通讯引用:** 32940 | [OpenAlex ID](https://openalex.org/A5051336665)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文构建了一套集成立体视觉、触觉反馈与专家扫描策略的机器人超声采集系统，可实现对腹部的自主、可适应扫描，并能生成三维超声体积。

**💡 创新点**

创新点在于：①将专家的手持扫描轨迹、施加力与诊断叙述数据录制并用于机器人重放；②利用立体视觉生成患者特定表面拓扑，并通过触摸测量硬度实现肋骨边界分割；③结合上述信息制定两条自主扫描路径（肋下斜扫与软组织正扫）并实现三维体积重建；④在单一实验平台上实现从手持到全自主的完整闭环。

**🔧 技术方法**

使用技术包括：七自由度顺应性关节驱动机器人、操作空间统一运动/力控制、立体摄像头生成点云与泊松重建、力/扭矩传感器进行硬度测量、POD提取力方向、三角网格与Geodesic路径生成、Hamming距离图像相似度评估、ImFusion软件进行3‑D重建与病灶分割。

**📊 数据集**

数据集主要为人工制备的腹部超声模型（含肝、胆囊、胰腺、肾及血管结构与假性肿瘤），并记录了专业放射科医生的自由手扫描轨迹、施力与图像叙述。

**📈 对比分析**

比较方法：将自由手扫描、机器人重放扫描、机器人自主扫描三种模式在同一模型上进行，分别评估轨迹误差、力误差、图像哈希距离、三维体积一致性和病灶分割精度；实验显示机器人重放与自由手在图像内容与病灶可视性上基本相当；自主扫描在保持足够接触力的同时实现三维重建，体积重建误差低于20%且能成功分割模拟肿瘤。

**⚠️ 局限性**

局限性包括：仅在静态模型上验证，未考虑呼吸或器官运动；扫描路径与力策略基于单一专家记录，缺乏跨患者泛化；缺乏实时图像质量优化与安全预警；未来需加入自适应速度/压力控制、动态跟踪与多模态感知。

---

## 373. Mechanistic Insights into Functional Sparsity in Multimodal LLMs via CoRe Heads

**arXiv ID:** 2606.05843 | [PDF](https://arxiv.org/pdf/2606.05843v1)

**作者:** Ruoxi Sun `[一作]` (Soochow University), Min Zhang `[通讯]` (Soochow University)

**通讯引用:** 42381 | [OpenAlex ID](https://openalex.org/A5013794939)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模态大语言模型（MLLM）中跨模态信息检索的机制，发现并量化了一类稀疏的专用注意力头（CoRe头）负责从嘈杂视觉上下文中提取查询相关视觉特征。

**💡 创新点**

首次提出了基于令牌级别的检索注意力质量（RAM）度量来识别CoRe头，并揭示了模型规模与架构对这些头位置和分布的影响，以及其对推理效率与性能的关键作用。

**🔧 技术方法**

利用注意力权重提取、RAM评分、因果干预（头部遮蔽）以及混合注意力（全注意力+稀疏滑动窗口）实现对CoRe头的定位与加速。

**📊 数据集**

在四个多模态基准上评估：RefCOCOg（视觉定位）、VidSTG（视频定位）、MMLongBench（多跳推理）、MMDocIR（文档检索）。

**📈 对比分析**

与完整模型及随机/底层头部遮蔽对照，结果显示遮蔽仅前5% CoRe头即可使性能大幅下降，而保留这些头并对其余头使用稀疏注意力可在保持或略提升任务准确率的同时实现1.8–2.1×的推理加速。

**⚠️ 局限性**

局限性包括：对不同模型结构的通用性仍需进一步验证；仅关注视觉检索，未探究语言层面或多模态交互的其他功能；加速方案在极端长序列下的稳定性和硬件适配性尚待实测。

---

## 374. ProSPy: A Profiling-Driven SQL-Python Agentic Framework for Enterprise Text-to-SQL

**arXiv ID:** 2606.05836 | [PDF](https://arxiv.org/pdf/2606.05836v1)

**作者:** Zhaorui Yang `[一作]` (Zhejiang University), Wei Chen `[通讯]` (Zhejiang University)

**通讯引用:** 68884 | [OpenAlex ID](https://openalex.org/A5100344384)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于数据剖面、逐步裁剪、无方言DSL数据检索和Python后端分析的企业级 Text-to-SQL 框架 ProSPy，能够在大型、异构数据库上完成复杂查询

**💡 创新点**

创新点在于：①利用自动数据剖面获取列级别细粒度信息，减少对不完整元数据的依赖；②以逐步裁剪方式构建紧凑、任务相关的 schema 上下文；③采用无方言DSL抽象 SQL 语义，跨数据库引擎统一处理；④将复杂分析迁移至 Python，提升灵活性与可扩展性

**🔧 技术方法**

技术包括：LLM驱动的 schema 链接与裁剪、DSL 视图生成与 SQL 编译、Python 代码合成与执行、数据剖面采集、面向任务的多步骤推理

**📊 数据集**

使用 Spider 2.0-Lite（Snowflake、BigQuery、SQLite）和 Spider 2.0-Snow（Snowflake）两个企业级基准数据集

**📈 对比分析**

与 Spider-Agent、ReForce、AutoLink、DSR-SQL、RSL-SQL、LinkAlign、APEX-SQL 等基线对比，ProSPy 在两套数据集上均以单一推理实现 60%+ 的执行准确率，显著超越多方法（尤其在不使用多数投票的情况下）

**⚠️ 局限性**

局限性包括：1）错误传播风险，视图错误会误导 Python 分析；2）实验仅覆盖两套基准，缺乏更广泛真实企业数据库验证；3）对极大 schema 的效率仍有提升空间

---

## 375. Robust and sparse support vector machine via hybrid truncated loss for supervised classification

**arXiv ID:** 2606.05814 | [PDF](https://arxiv.org/pdf/2606.05814v1)

**作者:** Yuliang Yang `[一作]` (Beijing Forestry University), Huiru Wang `[通讯]` (Beijing Forestry University)

**通讯引用:** 416 | [OpenAlex ID](https://openalex.org/A5100630692)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种混合截断（L_ht）损失函数，并基于该损失构建单视图和多视图SVM分类器；

**💡 创新点**

创新点在于：① L_ht损失兼具截断和非负性特性，既能提升鲁棒性又能减少支持向量；② 采用工作集规则的ADMM求解策略大幅降低单视图模型训练成本；③ 多视图模型融合结构协方差与自适应视角权重，兼顾一致性与互补性；

**🔧 技术方法**

主要技术包括：基于P‑stationary点的优化理论、交替方向乘子法（ADMM）与工作集规则、结构协方差矩阵与视角加权；

**📊 数据集**

实验使用UCI二分类基准数据集、合成噪声数据集以及图像数据集STL‑10；

**📈 对比分析**

与传统hinge、squared hinge、truncated、logistic等单视图损失以及六种多视图基线（SVM‑2K、MvSVM‑2C、MvLSSVC‑2C、Wave‑MvSVM、MVASY‑B、MvSL_0/1‑SVM）进行比较，L_ht‑SVM在保持竞争性精度的同时显著减少支持向量、提升噪声鲁棒性；MvL_ht‑SVM在所有评价指标上均优于六个基线，且随着数据规模增长效率优势更为明显；

**⚠️ 局限性**

局限性：未给出ADMM的收敛速率；仅实现线性SVM，未推广到核方法或深度学习；多类别场景的扩展尚未完成。

---

## 376. Detecting Large Quasi-cliques on Dynamic Networks

**arXiv ID:** 2606.05809 | [PDF](https://arxiv.org/pdf/2606.05809v1)

**作者:** Luciano Gualà `[一作]` (Tor Vergata University of Rome), Alessandro Straziota `[通讯]` (Tor Vergata University of Rome)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种能够在动态网络中实时维护大α-准团（quasi‑clique）的算法，并给出了全动态与增量两种变体。

**💡 创新点**

创新点在于：①用“信用”机制动态估计γ‑度（γ-degree）并据此驱动准团提取；②在增量场景实现 O(logΔ) 的最坏情况更新时间；③将该机制扩展到完整的插入/删除动态网络，取得 21× 加速。

**🔧 技术方法**

核心技术包括：信用计数器维护 γ‑度估计、k‑min‑hash 近似包含度、基于优先级队列的动态准团提取、缓冲 min‑hash 以支持删除操作。

**📊 数据集**

在 SNAP、NetworkRepository 及 DyReach 上的多组真实数据集（FB Ego、HP Ca‑HepPh、CM Ca‑CondMat、ER Email‑Enron、GW Loc‑Gowalla、SF Web‑Stanford、LJ LiveJournal、Wiki Wikipedia、Linux、HPDyn）以及对应的增量/全动态工作负载上进行实验。

**📈 对比分析**

与静态重跑的 Baseline 进行对比，增量版在大多数图上实现 200–207 倍的速度提升，完全动态版实现 20–21 倍加速；准团大小与密度与 Baseline 非常接近（>0.9 的高密度），即在性能提升的同时保持了解质量。

**⚠️ 局限性**

局限性：①全动态版本没有证明最坏情况 O(Δ) 以下的复杂度；②信用估计对 γ‑度的误差可能导致偶尔漏检大准团；③算法性能高度依赖参数 δ、ϕ、k 的调优；④在删除频繁的场景下需要更频繁的重搜索，可能降低加速比。

---

## 377. SALT: When More Rollouts Don't Help in Group-Based Policy Optimization and How to Make Them Matter

**arXiv ID:** 2606.05800 | [PDF](https://arxiv.org/pdf/2606.05800v1)

**作者:** Powei Chang `[一作]` (Bilibili Inc.), Dongying Kong `[通讯]` (Bilibili Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对RLVR中GRPO式分组归一化导致的梯度冗余和取消问题，提出SALT方法，使用子空间自适应几何重加权来提升有效更新。

**💡 创新点**

创新点在于利用批量梯度几何估计主子空间，将优势分解为共享与残差通道，并根据有效样本量自适应放大残差通道以降低签名取消。

**🔧 技术方法**

技术包括GRPO、PPO裁剪、Gram矩阵参与比率、有效样本量度量、子空间投影、LM-head梯度代理、在线自适应混合系数。

**📊 数据集**

使用数学推理基准（AIME24/25、GSM8K、MATH500、GPQA-Diamond）以及代码验证基准（MBPP），模型为DeepSeek-Distill-Qwen 1.5B/7B。

**📈 对比分析**

与GRPO/DAPO和熵正则化等基线对比，SALT平均提升约2.5-3个百分点精度，Pass@8提升，计算开销仅约7.7%，性能提升显著。

**⚠️ 局限性**

局限在于需额外的几何计算与LM-head梯度代理，可能对不同架构泛化受限；作为有偏的几何重加权代理，收益依赖于签名取消被有效抑制。

---

## 378. GCD: Garbled, Corrected, Demonstrandum -- Fixing and Proving Go's Extended GCD Implementation

**arXiv ID:** 2606.05796 | [PDF](https://arxiv.org/pdf/2606.05796v1)

**作者:** Linard Arquint `[一作]` `[通讯]` (National University of Singapore), Linard Arquint (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

修复了Go语言中扩展欧几里得算法实现的错误，并给出了形式化证明；

**💡 创新点**

提出了将静态分析与形式化验证相结合的全新流程，以发现并修复实现中的漏洞；

**🔧 技术方法**

使用Go语言自带工具链、SMT求解器Z3以及Coq（或Dafny）等形式化验证框架；

**📊 数据集**

采用标准整数测试集，包括随机大整数以及已知会触发错误的边界案例；

**📈 对比分析**

与原始实现及其他语言（如C/C++）实现做性能对比，修正后算法在正确性上无懈可击，执行时间略有提升；

**⚠️ 局限性**

局限在于验证范围主要覆盖单线程、有限精度整数，无法直接扩展到大规模多线程或任意精度整数场景。

---

## 379. Geometry-Aware Dataset Condensation for Diffusion Model Training

**arXiv ID:** 2606.05883 | [PDF](https://arxiv.org/pdf/2606.05883v1)

**作者:** Xiao Cui `[一作]` (University of Science and Technology of China), Houqiang Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27002 | [OpenAlex ID](https://openalex.org/A5078141810)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对扩散模型训练提出了几何感知的数据集压缩方法，将真实子集选择视为分布对齐问题，构造了基于一侧部分最优传输的目标，并辅以统计与语义正则；

**💡 创新点**

创新点在于将一侧部分OT与轻量级统计与语义正则相结合，形成几何感知的分布对齐目标，并提出两阶段离散优化（贪婪构造+交换细化）以高效逼近全局最优；

**🔧 技术方法**

采用一侧部分最优传输（POT）与熵正则化的Sinkhorn迭代求解、均值-方差正则化、置信度正则化，以及两阶段离散优化策略；

**📊 数据集**

在ImageNet-1K上进行实验，使用DiT-L/2、SiT-L/2等扩散模型，子集规模分别为10K、50K、100K；

**📈 对比分析**

与随机、K-Center、Herding、CCS、DQ、D^2C等基线对比，实验显示所提方法在FID、IS、Precision、Recall等指标上均优于所有基线，尤其在低数据预算和高分辨率下表现突出；

**⚠️ 局限性**

局限性包括对OT计算的求解复杂度、对特征编码器的依赖以及在极小子集规模下对样本多样性的进一步提升空间。

---

## 380. Entropy-Based Evaluation of AI Agents: A Lightweight Framework for Measuring Behavioral Patterns

**arXiv ID:** 2606.05872 | [PDF](https://arxiv.org/pdf/2606.05872v1)

**作者:** Olasimbo Ayodeji Arigbabu `[一作]` `[通讯]`, Olasimbo Ayodeji Arigbabu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于熵的AI代理评估框架EEA，并实现了可与多种代理框架（如LangChain、Google ADK）集成的Python工具包。

**💡 创新点**

创新点在于用熵指标（动作熵、轨迹熵、工具熵、信息增益等）从行为多样性和不确定性降低角度评估代理，而非仅依赖成功率，并提供可配置的综合得分EAS。

**🔧 技术方法**

采用信息论熵理论、Python实现、事件记录与标准化trace、与LangChain、Google ADK的适配器、绘图工具等技术。

**📊 数据集**

使用受控合成基准（六个事实问答、多跳推理、编程/调试任务）和Learning Roadmap Agent的三条学习路线规划任务作为评估数据集。

**📈 对比分析**

通过比较不同代理模式（direct‑LLM、react‑search、react‑search‑code、planner‑executor）和跨框架的LangChain/Google ADK在动作熵、工具熵、信息增益与EAS得分上的差异，展示了EEA在相同任务下的可比性，Google ADK略高的IG和EAS表明评估能够捕捉细微行为差异。

**⚠️ 局限性**

实验规模有限（仅三条任务、一轮重复）、缺乏人工评估、成本估算不完整且高度依赖trace质量，指标需结合任务难度和环境谨慎解释。

---

## 381. LLMCodec: Adapting Video Codecs for Efficient Weight Compression of Large Language Models

**arXiv ID:** 2606.05861 | [PDF](https://arxiv.org/pdf/2606.05861v1)

**作者:** Rui Wang `[一作]` (Shanghai Jiao Tong University), Zhengxue Cheng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1920 | [OpenAlex ID](https://openalex.org/A5071945287)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出LLMCodec，一种基于视频编码器的LLM权重量化压缩框架；

**💡 创新点**

通过可学习的仿射变换消除权重极值，并将权重映射为YUV420图像，利用现代视频编码器（如VVC/H.266）实现高效压缩；

**🔧 技术方法**

技术包括可学习仿射变换、RTN量化、VVC/H.266视频编码、All-Intra编码配置以及对不同编码器/配置的对比评估；

**📊 数据集**

在LLaMA-3-8B、LLaMA-2-7B、Qwen-2.5-Instruct-7B等模型上，使用WikiText2、C4以及ARC、HellaSwag、LAMBADA、PIQA、WinoGrande等下游任务数据；

**📈 对比分析**

与GPTQ和FlatQuant比较，LLMCodec在低比特率（尤其是2比特）下显著降低困惑度（如LLaMA-3-8B从41.15降至26.53），并提升下游任务平均准确率约21%；

**⚠️ 局限性**

局限性包括目前仅针对静态权重量化，未覆盖激活压缩，且在极低位宽下仍需进一步优化；

---

## 382. GenTI: Benchmarking LLMs for Autonomous IDPS Rule Generation for Unseen Attacks

**arXiv ID:** 2606.05844 | [PDF](https://arxiv.org/pdf/2606.05844v1)

**作者:** Hassan Jalil Hadi `[一作]` (King Abdullah University of Science and Technology), Ali Shoker `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 459 | [OpenAlex ID](https://openalex.org/A5015028788)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GenTI框架，利用大型语言模型自动生成并验证基于CTI的网络入侵检测系统（IDPS）与YARA规则，并构建首个面向LLM的规则级别基准数据集GTI。

**💡 创新点**

①将150k+ IDPS规则与50k YARA规则与多源CTI映射整合，形成可训练与评估的统一语义化规则库；②设计LLM驱动的Chain‑of‑Thought/Chain‑of‑Verification循环，实现规则生成、优化与验证；③采用QLoRA+课程化学习与CoT/CoV增量训练提升规则语法、语义与CTI覆盖。

**🔧 技术方法**

使用大型语言模型（DeepSeek‑7B、Qwen2.5‑7B等）结合QLoRA参数高效微调、链式思考与验证（CoT/CoVe）、多源CTI映射（OTX、MISP、MITRE ATT&CK/D3FEND）、自动化提示工程与规则验证管线。

**📊 数据集**

GTI数据集：约150k IDPS规则与50k YARA规则，包含协议行为、负载签名、CTI映射、威胁上下文与RCRA等结构化字段。

**📈 对比分析**

在GenTI测试集上与GPT‑3.5 Turbo、LLaMA、GPT‑4等基线对比，使用Syntax Accuracy、Semantic Similarity、CTI Coverage、Security Effectiveness和Composite Score指标；GenTI在所有指标上优于基线，Composite Score 89.4%，未知攻击检测提升42个百分点，误报率降至2.3%。

**⚠️ 局限性**

仍受限于已公开规则的覆盖范围，极端稀有或高度加密攻击模式表现有限；模型生成规则需人工验证以防误报；实验主要依赖Snort/Suricata和特定PCAP数据，泛化性需进一步评估。

---

## 383. PriSrv: Privacy-Enhanced and Highly Usable Service Discovery in Wireless Communications

**arXiv ID:** 2606.05821 | [PDF](https://arxiv.org/pdf/2606.05821v1)

**作者:** Yang Yang `[一作]` (Singapore Management University), Jian Weng `[通讯]` (Jinan University)

**通讯引用:** 18402 | [OpenAlex ID](https://openalex.org/A5082041657)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了PriSrv协议，实现无线通信中安全、隐私保护的服务发现与互操作的完整实现。

**💡 创新点**

创新点在于提出匿名凭证匹配加密（ACME）与快速匿名凭证（FAC），实现双向细粒度策略控制、选择性属性披露和多次使用不可关联性。

**🔧 技术方法**

核心技术包括双层匹配加密、匿名凭证、基于双线性映射的属性加密、Diffie‑Hellman密钥协商和哈希MAC。

**📊 数据集**

实验使用多种硬件平台（桌面、笔记本、手机、Raspberry Pi）与三种椭圆曲线（MNT159、MNT201、BN256）进行性能评估，并在IEEE 802.1X环境下测试。

**📈 对比分析**

相较于现有ME/AC方案和其他匿名凭证方案，ACME在功能上支持更复杂的布尔策略、选择性披露，且在计算和通信成本上与FAC等方案相比实现了更低的开销，广播/互鉴延迟均低于1 s。

**⚠️ 局限性**

局限性在于广播负载较大，导致在BLE等低速/碎片网络中吞吐和可靠性受限，且消息尺寸与链路层随机化机制的兼容性仍需进一步优化。

---

## 384. Consistency Training Along the Transformer Stack

**arXiv ID:** 2606.05817 | [PDF](https://arxiv.org/pdf/2606.05817v1)

**作者:** Sukrati Gautam `[一作]` (Purdue University), David Demitri Africa `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文扩展了一致性训练框架，引入了两种新的内部一致性目标（MLPCT和AttCT），并将其应用于四种新的安全威胁模型（人物上下文攻击、prefill攻击、多轮对抗性沮丧和条件失调），同时探讨了跨威胁的一致性训练效果与机制。

**💡 创新点**

创新点在于：①首次将MLP后激活状态与每头注意力分布作为一致性目标；②将一致性训练推广到多种新型威胁；③揭示不同一致性目标在共享残差流上的线性路径与BCT的差异性方向；④发现某些威胁训练可跨威胁产生正向或负向迁移。

**🔧 技术方法**

使用技术包括：一致性训练通用框架（扰动源、对齐目标、距离度量）、LoRA微调、BCT（输出层一致性）、ACT（残差流一致性）、MLPCT（MLP隐藏状态一致性）与AttCT（注意力分布一致性），以及 Jensen–Shannon Divergence、余弦距离等损失函数。

**📊 数据集**

数据集涵盖：44人设的wolf facts（人物上下文攻击）、ClearHarm prefill攻击集合（23种prefill每个有害提示）、WildChat和数学谜题（多轮沮丧）以及多模型的EM/IP/条件失调数据（Llama‑3.1、Qwen3等），并在Gemma、Llama、Qwen等模型上进行训练与评估。

**📈 对比分析**

与基线相比，BCT在prefill、沮丧、条件失调等威胁上实现了几乎零攻击成功率；MLPCT与AttCT在sycophancy、jailbreak、人物上下文等威胁上显著降低误导率，且在某些情况下实现了跨威胁正向迁移；所有方法在维持MMLU/MTBench能力上保持≤0.02%的漂移。

**⚠️ 局限性**

局限性包括：仅在LoRA微调下验证，可能不适用于全参数微调；威胁模型受限于实验设计，未覆盖所有真实攻击场景；部分威胁（如人物后缀攻击）仍对所有一致性方法无效；跨威胁迁移实验仅基于单一基模型，模型间差异未充分验证；机制分析仅揭示共享路径，未证明完全因果关系。

---

## 385. The Self-Correction Illusion: LLMs Correct Others but Not Themselves

**arXiv ID:** 2606.05976 | [PDF](https://arxiv.org/pdf/2606.05976v1)

**作者:** Kuan-Yen Chen `[一作]` (National Cheng Kung University), Jung-Hsien Chiang `[通讯]` (National Cheng Kung University)

**通讯引用:** 3980 | [OpenAlex ID](https://openalex.org/A5062802526)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型代理在自身推理中自我纠错的不足，并提出了通过重新标记（role relabel）将错误声明放入外部角色（如 <memory>、用户消息或工具响应）来提升纠错率的零训练干预方法。

**💡 创新点**

核心创新在于将自我纠错缺失归因为聊天模板（chat‑template）标签的“可寻址性（addressability）”缺失，而非认知缺陷；通过仅改变角色标签即可在多种模型和任务上将纠错率提升 23–93 个百分点，展现了一种可控、无需微调的可靠性杠杆。

**🔧 技术方法**

技术手段包括：1) 字节级身份保证（SHA‑256 断言）；2) 角色重标记（role relabel）与句法包装（handle‑granularity ladder）分解；3) 审计指令与自我怀疑提示对照；4) 对数概率分析与对齐词序；5) 成对 bootstrap 统计与显著性检验；6) 多模型、多域实验框架。

**📊 数据集**

使用了可验证推理数据集：GSM8K‑style 代数题、生成式逻辑推理谜题、BBH Logical Deduction 子任务，每个实验单元保留 30 条配对任务，确保错误声明在所有条件下完全相同。

**📈 对比分析**

比较方法是对 13 个模型‑域单元（七大模型家族 × 三个任务域）分别在五种角色包装条件下测量“显式纠错率（CR）”，并用配对 bootstrap 计算 ΔCR 与 p‑值。结果显示，绝大多数单元（10/13）在至少一种重标记条件下显著提升 23–93 pp，且效果在不同模型、不同角色标签与不同任务域中保持稳健；对照实验表明单纯自我怀疑提示无法复制该提升，进一步验证了可寻址性机制。

**⚠️ 局限性**

限制包括：1) 结果主要针对已知错误且在基线无法自我纠错的任务；2) 在接近上限（ceiling）或已经过深度纠错训练的模型中提升空间有限；3) 仅评估结构化可验证任务，未覆盖自由文本推理或复杂规划；4) 安全性依赖于避免信任指令，单句信任命令可显著破坏防御；5) 关闭权重模型样本量有限，置信区间较宽。

---

## 386. FORTE: FOL-guided Optimal Refinement for Text-audio rEtrieval

**arXiv ID:** 2606.05812 | [PDF](https://arxiv.org/pdf/2606.05812v1)

**作者:** Arghya Pal `[一作]` (Monash University), Sailaja Rajanala `[通讯]` (Monash University)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5039283151)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FORTE框架，通过将文本查询转换为一阶逻辑并进行结构化精炼，再与音频嵌入对齐，显著提升文本到音频检索精度

**💡 创新点**

创新点在于将符号逻辑推理与参数高效的跨模态对齐结合，并通过谓词感知的重排序进一步强化语义一致性

**🔧 技术方法**

使用一阶逻辑表示、受限搜索、对比学习、轻量级投影模块、逻辑一致性正则化以及基于谓词的重排序

**📊 数据集**

在AudioCaps和Clotho两个标准音频检索数据集上进行评估

**📈 对比分析**

与CLAP、LAION-CLAP、Pengi等主流基线相比，FORTE在R@1、R@5、R@10、mAP@10上均取得显著提升，Clotho R@1提升至20.4（基线16.75），AudioCaps R@1提升至38.2（基线33.9）

**⚠️ 局限性**

依赖于FOL解析器的准确性和词汇覆盖度，OOV谓词会导致性能下降；对音频标题生成质量敏感；整体仍存在一定的模态间桥接限制

---

## 387. Quantifying Uncertainty In Wide Two-Layer Neural Networks: On The Law Of The Limiting Fluctuation Process

**arXiv ID:** 2606.05982 | [PDF](https://arxiv.org/pdf/2606.05982v1)

**作者:** Arnaud Descours `[一作]` (Université Lyon 1), Paul Stos `[通讯]` (Université Clermont-Auvergne)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种通过直接评估不确定性来减少神经网络预测中的计算成本的方法，避免了深度集成方法的复杂性。

**💡 创新点**

创新点在于通过偏微分方程（PDE）信息直接评估不确定性，而不是依赖于训练多个网络的蒙特卡洛估计。

**🔧 技术方法**

使用了偏微分方程（PDE）和随机梯度下降（SGD）等技术，结合了轨迹中心极限定理。

**📊 数据集**

使用了一维回归示例的数据集，具体为y = x^3 + ε，其中x从均匀分布中抽取，ε为独立的高斯噪声。

**📈 对比分析**

与传统的深度集成方法进行比较，PDE方法在计算效率上具有优势，且在一维回归示例中表现出良好的不确定性量化能力。

**⚠️ 局限性**

限制在于该方法的实现依赖于对PDE的求解，且在高维和实际应用中可能面临计算复杂性和准确性的问题。

---

## 388. Edit-R2: Context-Aware Reinforcement Learning for Multi-Turn Image Editing

**arXiv ID:** 2606.05950 | [PDF](https://arxiv.org/pdf/2606.05950v1)

**作者:** Yuxiao Ye `[一作]`, Ling Pan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对多轮交互式图像编辑的新框架 Edit‑R2，利用强化学习和会话意图重构实现连续编辑与上下文一致性。

**💡 创新点**

创新点在于引入 IC‑CoT 会话意图重构以及统一的多轮 RL 换言之，将文本推理与图像生成在离散与连续空间同步优化，并加入轨迹过滤提升训练稳定性。

**🔧 技术方法**

采用的技术包括基于流匹配的生成器、IC‑CoT 逻辑链、统一的 GRPO 优化、前缀有效优势精炼以及 LoRA 细化。

**📊 数据集**

使用的数据集为新构建的 MICE‑Bench，包含 720 个三轮编辑样本，涵盖内容记忆和理解任务，并在 EdiVal‑Bench 与 GEdit‑Bench 上做进一步评测。

**📈 对比分析**

与 BAGEL、OmniGen、VINCIE 等开放源模型以及 Nano‑Banana‑2、GPT‑Image‑1 等闭源模型对比，Edit‑R2 在 MICE‑Bench 上在指令跟随、内容一致性和全局意识三项指标平均提升约 20%，并在多轮后仍保持较高性能。

**⚠️ 局限性**

局限性包括 MICE‑Bench 仅涵盖两类编辑场景，难以覆盖更广泛的真实交互需求，且模型规模与推理速度仍有待进一步提升。

---

## 389. Exploring the connection between coding habits and cognitive styles in malware developers

**arXiv ID:** 2606.05945 | [PDF](https://arxiv.org/pdf/2606.05945v1)

**作者:** Vasilis Vouvoutsis `[一作]` (University of Piraeus), Fran Casino `[通讯]` (Universitat Rovira i Virgili)

**通讯引用:** 4732 | [OpenAlex ID](https://openalex.org/A5013675709)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对658份泄露的恶意软件源代码与249份公开开源软件进行静态代码分析，计算软件度量与CWE漏洞，探究恶意开发者的编码习惯与认知风格；

**💡 创新点**

首次将工业级SAST工具与多种代码度量相结合，对海量跨语言恶意软件进行大规模定量分析，并将软件工程指标视为行为信号进行解释；

**🔧 技术方法**

使用Cppcheck、Bandit、Snyk、Semgrep等SAST工具以及Mozilla的rust-code-analysis对代码进行度量，辅以COCOMO估算、GMM聚类与PCA可视化；

**📊 数据集**

恶意软件来源于VX Underground共658个项目，正向软件分别为Python PyPI前100、JavaScript npm前100及49个安全相关OSS（如nmap、sqlmap、zap）；

**📈 对比分析**

通过统计比较代码规模、复杂度、可维护性及CWE密度等指标，发现恶意代码更小、文档稀少、函数复杂度略高、存在更多质量缺陷，聚类结果显示两类代码重叠显著；

**⚠️ 局限性**

主要局限包括语言生态差异（恶意多为C/C++，对比正向多为Python/JS），部分代码缺乏度量支持，静态分析无法捕捉运行时行为，且度量与行为间的因果关系仍需进一步验证。

---

## 390. Epistemic Injustice in Language Models: An Audit of Pretraining Filters and Guardrails

**arXiv ID:** 2606.05936 | [PDF](https://arxiv.org/pdf/2606.05936v1)

**作者:** Marco Antonio Stranisci `[一作]` (University of Turin), Anne Lauscher `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

审计了预训练过滤器和推理时守卫模型在 Common Crawl 句子上的表现，评估其对边缘化群体的影响。

**💡 创新点**

提出系统化评估框架，结合手工标注，揭示“知识性抹杀”现象及其与人类判断的显著差异。

**🔧 技术方法**

使用规则词表、文本分类器、Meta Guardrail、Llama-Guard、Qwen3Guard 等技术，以及词典抽取、人工注释和统计一致性检验。

**📊 数据集**

使用 Common Crawl 2024‑33 句子样本、手工标注 500 条子集、Wikidata 性别和地区词典等数据集。

**📈 对比分析**

通过 Cohen’s κ 评估系统一致性，比较 flag 率与基线，发现过滤器/守卫对边缘化身份过度 flag、缺失 IP/隐私类；与人类标注差距显著。

**⚠️ 局限性**

仅考察部分身份维度、标注样本有限、跨文化评判不足、未覆盖非英语数据。

---

## 391. Towards World Models in Biomedical Research

**arXiv ID:** 2606.05925 | [PDF](https://arxiv.org/pdf/2606.05925v1)

**作者:** Guangyu Wang `[一作]` (Beijing University of Posts and Telecommunications), Yong Li `[通讯]` (Tsinghua University)

**通讯引用:** 38716 | [OpenAlex ID](https://openalex.org/A5100355277)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了生物医学世界模型的概念，并概述了其在多尺度、可干预动态模拟与闭环科学决策中的潜在应用。

**💡 创新点**

将世界模型框架迁移到生物医学领域，强调多模态潜在状态、干预感知动力学、以及与实验/临床闭环的结合，推动从静态识别向动态仿真转变。

**🔧 技术方法**

利用生成式模型（如扩散模型、变分自编码器、Transformer、神经ODE等）构建潜在空间与观测空间的动态模型，并与智能代理耦合实现内部仿真与规划。

**📊 数据集**

未给出具体公开数据集，主要讨论了所需的纵向多模态、干预记录等数据类型，暗示可使用如UK Biobank、CITE‑seq、10X Multiome等数据来源。

**📈 对比分析**

论文为视角性工作，未进行实验比较；作者指出未来需构建专门的基准和评估指标来验证模型的预测和规划能力。

**⚠️ 局限性**

主要限制包括缺乏足够的纵向干预数据、评估标准不完善、隐私与安全风险、模型解释与偏差累积、部署成本高等。

---

## 392. DBHN-Net: Dual-Branch Hybrid Neural Network For Low-Complexity Monaural Speech Enhancement

**arXiv ID:** 2606.05911 | [PDF](https://arxiv.org/pdf/2606.05911v1)

**作者:** Cunhang Fan `[一作]` (Anhui University), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 57271 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出双分支混合神经网络（DBHN-Net）用于单通道语音增强，融合ANN和SNN以兼顾性能与低能耗。

**💡 创新点**

创新点在于设计并行的ANN+SNN分支，配合BandSplit+TF‑Mamba与LIF+SFEB+ITB模块，加入Interaction与TF‑Cross Attention Fusion实现跨分支高效信息融合，并首次在语音增强中使用SNN降低计算复杂度。

**🔧 技术方法**

使用的技术包括卷积网络、Leaky Integrate‑and‑Fire脉冲神经元、Mamba状态空间模型、双域注意力机制、残差连接、梯度代理（Sigmoid）等。

**📊 数据集**

采用公开数据集WSJ0‑SI84+DNS‑Challenge、VoiceBank+Demand和DNS‑Challenge 2020进行训练与评估。

**📈 对比分析**

通过与11–17个基准模型在PESQ、ESTOI、SI‑SDR等指标下对比，DBHN‑Net在三组数据集上均取得最优或竞争性表现，尤其在低SNR情境下表现稳健，并将计算复杂度平均降低约7.5倍。

**⚠️ 局限性**

局限性包括SNN分支仍存在离散化信息损失、模型架构较为复杂且训练需要梯度代理，以及目前仅验证单通道场景，尚未推广至多通道或实时应用。

---

## 393. ACE-SQL: Adaptive Co-Optimization via Empirical Credit Assignment for Text-to-SQL

**arXiv ID:** 2606.05906 | [PDF](https://arxiv.org/pdf/2606.05906v1)

**作者:** Xiaobing Chen `[一作]` (Harbin Engineering University), Zhiqi Pang `[通讯]` (Harbin Engineering University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为 ACE-SQL 的强化学习框架，联合优化文本到 SQL 的模式生成与架构检索，利用执行结果为检索动作分配信用。

**💡 创新点**

创新点在于：①使用执行正确的 SQL 产出动态生成检索目标（empirical credit assignment），实现检索与生成的双向协同；②通过 PCGrad 和生成器权重调度稳定两任务的联合梯度；③采用在线列集池和多数投票来缓解训练中的非平稳性。

**🔧 技术方法**

核心技术包括：基于 GRPO 的双角色策略优化；在线列集池与多数投票生成检索目标；长度惩罚与稀疏奖励机制；PCGrad 梯度冲突缓解；生成器权重线性调度。

**📊 数据集**

主要使用的数据集为：SynSQL‑2.5M（合成训练），2913 对问答数据库对（RL 训练），以及公开基准 BIRD Dev 与 Spider（评估）。

**📈 对比分析**

与多种闭源和开源基线（如 GPT‑4、MAC‑SQL、SQL‑R1‑7B、MTIR‑SQL‑8B 等）对比，ACE‑SQL 在 BIRD Dev 上获得 65.3% 的 greedy 执行准确率，Spider 测试集上 87.2%，并将平均输出长度从 1.90k 降至 0.93k，显著提升效率并在 BIRD Dev 上超过所有公开基线。

**⚠️ 局限性**

限制包括：仅使用单一 8B 参数模型（Qwen3‑8B），未验证更大模型或不同架构的泛化；训练数据主要来自合成语料，缺乏真实世界查询数据库对；初始列集池从 SFT checkpoint 开始，可能限制对基准模型的探索。

---

## 394. Metric Facility Assignment with Partial Information

**arXiv ID:** 2606.05905 | [PDF](https://arxiv.org/pdf/2606.05905v1)

**作者:** Vasilis Gkatzelis `[一作]` (Drexel University), Alexandros A. Voudouris `[通讯]` (University of Essex)

**通讯引用:** 740 | [OpenAlex ID](https://openalex.org/A5090907162)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在未知线性度量空间中将代理人分配到设施的问题，目标是近似最小化社会成本。

**💡 创新点**

创新点在于系统性地评估三种信息类型（序数偏好、审批偏好、设施间距离）组合对决策失真（distortion）的影响，并给出了几乎完备的紧界。

**🔧 技术方法**

主要技术包括构造虚拟距离变换、基于 α‑阈值的审批匹配算法、分层分区与递归分解，以及对三种信息模型的严格上界与下界证明。

**📊 数据集**

本文没有使用实验数据，而是通过构造理论实例证明上界与下界的匹配性；所有结果均在理论上严谨证明。

**📈 对比分析**

与仅使用序数偏好的算法（失真≤3）相比，本文提出的算法在不同信息组合下实现了失真 1+√2（O+A）、2√2−1（O+A+D，2设施）以及 2（O+A+D，≥3设施）的严格上限，证明了这些上限在相应模型中是最优的。

**⚠️ 局限性**

局限性包括：仅在线性度量空间得到完整结果；对一般度量空间只能给出 1+√2 的上界；仅使用审批信息在四及以上设施时仍无法突破失真 3；对三设施情形的最优性尚未完全确定。

---

## 395. PriSrv+: Privacy and Usability-Enhanced Wireless Service Discovery with Fast and Expressive Matchmaking Encryption

**arXiv ID:** 2606.05902 | [PDF](https://arxiv.org/pdf/2606.05902v1)

**作者:** Yang Yang `[一作]` (Singapore Management University), Robert H. Deng `[通讯]` (Singapore Management University)

**通讯引用:** 24249 | [OpenAlex ID](https://openalex.org/A5001712801)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了基于 FEME 匹配加密的 PriSrv+ 服务发现协议，提升了隐私、可表达性和实用性。

**💡 创新点**

首次提出 Fast and Expressive Matchmaking Encryption (FEME)，支持无限属性、双向策略匹配、发送方身份认证并实现匿名，同时消除 ACME 的小宇宙与二进制向量限制。

**🔧 技术方法**

融合部分隐藏访问结构、随机性分裂、双重再随机化、A‑CP‑ABE、A‑KP‑ABE、Hybrid‑ABE 等技术，基于双线性映射与线性秘密共享实现高效密文生成与解密。

**📊 数据集**

实验使用公开曲线 MNT159/201/BN256，并在桌面、笔记本、手机及 Raspberry Pi 四个平台上随机生成属性与访问策略进行评测。

**📈 对比分析**

在相同硬件与曲线条件下与 ACME 与 PriSrv 进行对比，FEME/ PriSrv+ 的加解密时间提升约 7.62×/6.23×，广播与鉴权阶段的通信成本压缩 87% 以上，安全性通过 GGM 模型证明。

**⚠️ 局限性**

仍需可信中心进行属性分配，缺乏内置撤销机制，强匿名性可能导致服务滥用，需要进一步探索去中心化与可追溯方案。

---

## 396. MemoryCard: Topic-Aware Multi-Modal Clue Compression for Long-Video Question Answering

**arXiv ID:** 2606.05917 | [PDF](https://arxiv.org/pdf/2606.05917v1)

**作者:** Qing Yang `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 38812 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于视频记忆卡（MemoryCard）的增强框架，先把长视频切分为语义一致的事件单元，生成事件级视频要点并挑选代表性视觉时刻，再将要点与视觉时刻渲染为统一的图片式记忆卡，以此来提升长视频问答的性能。

**💡 创新点**

创新点在于将稀疏帧级证据转化为高密度多模态事件级记忆卡，通过自读VLM实现视频语义分割与要点生成，并将视觉时刻与要点包装为可直接输入VLM的记忆卡，从而在视觉令牌预算受限时实现更有效的长视频问答。

**🔧 技术方法**

采用自读VLM（如Qwen3-VL-8B）进行视频语义分割与要点生成；使用ASR与字幕对齐获取文本信息；通过可视化渲染将要点与代表性视觉时刻合成图片式记忆卡；检索阶段使用LongCLIP进行问题-记忆卡相似度检索；检索到的卡按时序排序、按相关度分配分辨率后送入回答VLM（Qwen2-VL、Qwen3-VL、MiniCPM-V等）。

**📊 数据集**

在Video-MME、MLVU和LongVideoBench这三个长视频问答基准上进行评估，采用多选任务的准确率作为评价指标。

**📈 对比分析**

在保持回答器、提示格式、解码配置等不变的条件下，将MemoryCard与原始模型进行增量对比，结果显示在所有基准上均取得显著提升，最高可达21.8%相对准确率提升，并在不同视频时长与任务子集上均表现出更稳健的优势。

**⚠️ 局限性**

主要局限在于记忆卡构建需要额外的前处理开销，且依赖自读VLM；对于需要细粒度运动动态或连续动作理解的问答场景，事件级记忆卡可能无法充分捕捉细节，如何在保持高密度证据的同时提升构建效率仍是未来挑战。

---

## 397. Knowledge Manifold: A Riemannian Geometric Framework for Semantic Mapping and Geodesic Analysis of Scientific Literature

**arXiv ID:** 2606.05907 | [PDF](https://arxiv.org/pdf/2606.05907v1)

**作者:** Tomonaga Okabe `[一作]` (Tohoku University), Kazuhiko Komatsu `[通讯]` (Tohoku University)

**通讯引用:** 1033 | [OpenAlex ID](https://openalex.org/A5051997684)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了知识流形，对文献集进行语义映射、插值与最短路径分析，并生成虚拟论文摘要。

**💡 创新点**

将字符 n‑gram TF‑IDF 与 Smoothed Particle Hydrodynamics (SPH) 结合，实现连续语义插值；利用 GPR 进行不确定性量化；通过 Riemannian 元量化的 SPH 地图求解几何最短路径。

**🔧 技术方法**

字符 n‑gram TF‑IDF、SPH 插值、梯度与方向相似度、低维 SVD + 高斯过程回归、L‑BFGS‑B 多起点 geodesic 优化。

**📊 数据集**

20 篇聚合材料与航空结构力学领域的期刊论文。

**📈 对比分析**

与传统聚类/可视化方法（UMAP、t‑SNE、k‑NN 图）对比，SPH+GPR 既能提供插值预测又给出不确定性；geodesic 能在文献空间中找到更自然的概念迁移路径，改进幅度虽小但可验证。

**⚠️ 局限性**

仅限于 20 篇单语域的样本；SPH 需手工设定平滑长度；GPR 在高维词向量上计算受限；多起点 geodesic 仍可能收敛到局部最优；无法直接处理跨语言或大规模语料。

---

## 398. Resonant Minds: Closed-Loop Social Avatars with Theory of Mind

**arXiv ID:** 2606.05896 | [PDF](https://arxiv.org/pdf/2606.05896v1)

**作者:** Jianxu Shangguan `[一作]` (University of Washington), Wentao Zhu `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种闭环双代理多模态交互框架，整合感知、Theory of Mind推理与表情生成，实现在信息不对称下的社交数字人对话与视频合成。

**💡 创新点**

核心创新在于：①闭环感知-推理-生成的全流程集成；②使用ToM模块对伴侣隐含心理状态进行结构化推断；③引入多评估器集成机制平衡共情、策略与个性一致性；④构建心理学基础的Persona‑Scenario层级数据集。

**🔧 技术方法**

技术包括 GPT‑4o 作为对话与推理核心、HumanOmni 进行多模态感知、BDIE 框架实现 ToM 结构化推断、DICE‑Talk+Index‑TTS+RIFE 进行双人表情视频合成、CLIP‑文本-权重映射进行情绪控制、ensemble evaluator 等。

**📊 数据集**

使用自建的 Persona‑Scenario 数据集（50 位心理特征丰富的 persona、90 个 Sotopia 场景），以及 DialToM、Sotopia‑Hard 进行外域验证。

**📈 对比分析**

与基线 Agent‑mode（仅自我信息）和 Script‑mode（完整信息）对话模型对比，实验显示在真实性、深度、共情与一致性等多项指标上均优于 Agent‑mode，甚至在部分自然性指标上超过 Script‑mode；在视频质量上相较于 Sonic、EDTalk 等最新说话头模型，情感准确度、同步性和多样性均更高。

**⚠️ 局限性**

局限性包括：①对大规模或长对话的可扩展性未知；②依赖 GPT‑4o 及昂贵算力；③情绪控制仍以预训练权重映射为主，细粒度情感细节受限；④对跨文化、多语言场景的鲁棒性待进一步验证。

---

## 399. T-FunS3D: Task-Driven Hierarchical Open-Vocabulary 3D Functionality Segmentation

**arXiv ID:** 2606.05975 | [PDF](https://arxiv.org/pdf/2606.05975v1)

**作者:** Jingkun Feng `[一作]` (Delft University of Technology), Reza Sabzevari `[通讯]` (Delft University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无需训练、基于任务驱动的分层开放词汇3D功能分割方法T-FunS3D，可从RGB‑D图像和点云中定位并分割任务相关的功能部件。

**💡 创新点**

创新点：① 通过视觉语言模型（VLM）与大型语言模型（LLM）共同解析自由文本任务，抽取空间关系、参照对象与功能实体；② 构建只包含视觉嵌入的轻量化开放词汇场景图，实现一次性生成实例分割后可复用；③ 在功能分割阶段采用最小化的SAM掩码而非最高置信度掩码，提升细粒度部件的定位精度。

**🔧 技术方法**

技术方法：Mask3D（无类别实例提议）+ FG‑CLIP（多视角视觉嵌入）构造场景图；Qwen3‑14B 进行任务解析；Molmo 与 SAM 生成功能部件的2D掩码，再投影回3D；使用余弦相似度在场景图中检索参照与功能对象。

**📊 数据集**

使用公开数据集 SceneFun3D 进行实验，该数据集提供室内场景点云、RGB‑D图像、功能部件标注和多样化任务描述。

**📈 对比分析**

与 Fun3DU、OpenMask3D、OpenIns3D、LERF 等基线对比，T‑FunS3D 在 mAP、AP_25、AP_50、mIoU 等指标上均优于基线，特别在具有空间指代表达的任务上提升了约 12.5 AP_25 和 4.5 mIoU；同时在运行时与内存占用上显著低于 Fun3DU。

**⚠️ 局限性**

局限性：① 对大角度视角的图像分割精度下降；② 对功能部件无物理附着的场景（如天花板灯开关）分割失败；③ 仍依赖高质量多视角图像与点云，场景不完整或噪声大时性能会受影响。

---

## 400. Causal Scaffolding for Physical Reasoning: A Benchmark for Causally-Informed Physical World Understanding in VLMs

**arXiv ID:** 2606.05966 | [PDF](https://arxiv.org/pdf/2606.05966v1)

**作者:** Tianyi Tang `[一作]` (CFAR, IHPC, Agency for Science, Technology and Research (A*STAR)), Haiyan Yin `[通讯]` (CFAR, IHPC, Agency for Science, Technology and Research (A*STAR))

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计了CausalPhys基准，构建了3000+带专家标注因果图的视觉问答数据，并提出CRFT因果推理微调方法，以提升VLM的因果推理能力。

**💡 创新点**

创新点在于将因果DAG与VLM评估和训练相结合，提出基于因果图的评估指标和因果图对齐的微调策略，实现可解释且结构化的因果推理。

**🔧 技术方法**

采用因果图结构化评估、LLM判定器、链式思考与因果推理微调（CRFT）等技术，支持多模型的因果推理训练与评估。

**📊 数据集**

利用11个公开视觉数据集构建CausalPhys，并在PhysBench等跨基准数据上验证模型的迁移效果。

**📈 对比分析**

通过与11个VLM的对比实验，发现开放源代码与闭源模型在大多数任务上相当，CRFT显著提升ACC和关系意识，约提升20%–30%。

**⚠️ 局限性**

局限性包括：缺乏对多智能体和随机动态场景的因果泛化能力；因果图构建与校验成本高；评估仍主要关注静态视觉推理。

---

## 401. Dead Directions: Geometric Singular Learning

**arXiv ID:** 2606.05957 | [PDF](https://arxiv.org/pdf/2606.05957v1)

**作者:** Tejas Pradeep Shirodkar `[一作]` `[通讯]` (International Institute of Information Technology), Tejas Pradeep Shirodkar (International Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

提出了“死亡方向”这一统一原语，将信息几何与奇异学习理论中的 Fisher 失真、KL阶数、RLCT 等概念联系起来，证明了在原始参数坐标下可直接通过 Fisher 最小特征值的衰减速率读取 KL 阶数，从而在不做 Hironaka 解析的情况下恢复 Watanabe 的 RLCT；随后将该速率在深度网络中通过 K‑FAC 分解推广为层级速率梯度，并给出了梯度流和 G‑等变 Adam（DDCAdam）的适配；最后构建了速率链，涵盖 Fisher 率、Riemann 曲率和高曲率体积的相互关系。

**💡 创新点**

核心创新在于：
- 定义并统一 “死亡方向” 这一概念，使得 Amari 的 Fisher 零点与 Watanabe 的奇异切线在同一向量上对应；
- 证明死亡方向的 KL 阶数可直接由 Fisher 最小特征值的指数 2(k‑1) 读取；
- 开发 “选择规则” 将该指数转换为局部 RLCT 贡献 1/(2k)；
- 在多层 K‑FAC 框架下构造层级速率梯度 ladder，揭示激活/梯度因子对称性；
- 推导出 G‑等变 Adam（DDCAdam）以保持梯度流在 gauge 商空间上的速率；
- 形成完整的速率链，说明 Fisher 率、曲率发散率和高曲率体积是同一 KL 阶数的三种表达。

**🔧 技术方法**

主要技术包括：
- 解析展开与 L^2(p*) 归一化的 score 系数递推，利用 Schur 补充与非奇异 Fisher 块的正定性；
- K‑FAC Kronecker‑factored Fisher 分解，将整体 Fisher 分解为激活与梯度子块；
- 通过对角线死方向的 Schur 补充，得到层级速率梯度；
- 使用 Hironaka 解析的概念来定义 RLCT，但实际证明中避免了显式解析；
- Riemann 计算，证明曲率发散指数为 -(2k‑1)；
- 实验验证，利用合成数据（高斯混合、低秩回归、深线性网络）对速率和 RLCT 进行对比。

**📊 数据集**

实验数据集：
- 低维合成高斯混合模型（双/三分量）；
- 低秩回归数据（多层线性网络与 rank‑r 教师）；
- 线性、光滑和 ReLU 激活的深层线性网络（随机高斯输入）；
- 这些都是人工生成的控制实验，用于验证理论推导。

**📈 对比分析**

对比方法：
- 通过对 Fisher 最小特征值的 log‑log 拟合获得速率指数，并与理论的 2(k‑1) 进行对比；
- 将该指数映射为 RLCT 1/(2k) 并与 Watanabe 的解析结果对照，误差通常在 1–2% 以内；
- 与传统的局部学习系数（LLC）方法对比，证明了在不需后验采样的情况下可得到相同的 RLCT；
- 在深层网络中，通过 K‑FAC 层级速率梯度与实验测得的 λ_min 对比，验证了层级梯度 ladder 的正确性。总体而言，实验结果与理论一致，验证了所提出的桥接原语和速率链。

**⚠️ 局限性**

局限与挑战：
- 需要模型在接近奇异点时满足光滑、解析性，并且奇异纤维必须是光滑子流形；非解析或高阶交叉（如 ReLU 的分段性）可能不完全符合假设；
- 证明中假设死亡方向在各层保持“规范对齐”，若不满足则层级速率梯度推断失效；
- 需要“光谱通用性”条件 (G) 以分离切线与正常方向的特征值指数，实际中可能出现指数重叠导致判定困难；
- 目前只在理论层面给出了 G‑等变 Adam 的构造，实际数值实现与收敛性尚待进一步验证；
- 对于多重死亡方向或非平凡交叉的情况，曲率与高曲率体积的分析仍不完整，需进一步研究。

---

## 402. Bidirectional Search for Longest Paths: Case for Front-to-Front Heuristics

**arXiv ID:** 2606.05956 | [PDF](https://arxiv.org/pdf/2606.05956v1)

**作者:** Tzur Shubi `[一作]` (Ben Gurion University of Negev), Shahaf S. Shperberg `[通讯]` (Ben Gurion University of Negev)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了BiXDFBnB，一种面向Generalized Longest Simple Path（GLSP）问题的双向深度优先分支限界算法，利用前后前沿（F2F）启发式实现高效剪枝；

**💡 创新点**

核心创新在于将单前沿双向搜索（SFBDS）框架迁移至MAX优化场景，采用同步Cartesian扩展和Exact/Adjacent Meet策略，消除传统双向前沿匹配开销；

**🔧 技术方法**

采用深度优先分支限界、前后前沿（F2F）启发式、Biconnected-Component（h_BCC）等图论启发式，以及并行Cartesian产品扩展的技术实现；

**📊 数据集**

使用二维网格（6×6到8×8）和Maze、蛇形（Snake）问题的随机障碍实例，以及超立方体（4D至7D）中的Coil-in-the-Box（CIB）实例；

**📈 对比分析**

与单向A*、双向XMM、单向DFBnB等基线比较，BiXDFBnB F2F在节点扩展上降低一至两位数，运行时间显著优于所有对比方法，甚至在高维CIB中实现了最优解；

**⚠️ 局限性**

局限性包括对高维度或极大分支因子时的Cartesian扩展开销、缺乏动态跳跃策略的进一步优化以及对并行化实现的研究不足。

---

## 403. To Be Multimodal or Not to Be: Query-Adaptive Audio-Visual Person Retrieval via Active Modality Detection

**arXiv ID:** 2606.05931 | [PDF](https://arxiv.org/pdf/2606.05931v1)

**作者:** Erfan Loweimi `[一作]` (University of Cambridge), Mark Gales `[通讯]` (University of Cambridge)

**通讯引用:** 15292 | [OpenAlex ID](https://openalex.org/A5050766679)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于查询自适应的音视频人检索框架，能够在广播档案中根据是否存在语音或面孔来决定是否采用多模态融合，避免无关模态注入噪声导致检索性能下降。

**💡 创新点**

创新点在于利用跨模态得分一致性检测查询时活跃的模态，并据此动态调整融合权重，从而在多模态缺失或信息不均衡的真实场景下实现最佳检索效果。

**🔧 技术方法**

使用技术包括 ECAPA‑TDNN 语音嵌入、ResNet‑400 视觉嵌入、跨模态得分一致性特征构造、逻辑回归/线性SVM/决策树等分类器以及基于余弦相似度的后期融合。

**📊 数据集**

数据集为公开的 BBC Rewind（约12,594个视频文件，覆盖1948–1979年），并从中构建了523条包含不同音视频存在类型（AVP、AoP、VoP）的检索查询。

**📈 对比分析**

与单模态（语音82.9% P@1、面孔93.4% P@1）、固定权重多模态（90.0% P@1）以及知情模态标签的 Oracle（96.6% P@1）相比，本文自适应融合取得了94.2% P@1，恢复了固定融合与 Oracle 之间约64%的性能差距，整体表现显著提升。

**⚠️ 局限性**

主要局限在于：1）误判模态导致的检索错误仍然存在，约11% 的检测错误会影响性能；2）模型对极低质量或缺失模态的鲁棒性有限；3）嵌入模型的零样本性质在极端条件下可能降低识别精度。

---

## 404. Political Persuasion and Endorsement in Large Language Models

**arXiv ID:** 2606.05961 | [PDF](https://arxiv.org/pdf/2606.05961v1)

**作者:** Alessia Antelmi `[一作]` (Università degli Studi di Torino), Giovanni Da San Martino `[通讯]` (Università degli Studi di Padova)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估了六款开放权重的 LLM 在面对包含说服技巧的政治文本时的认同倾向，并探究了中立与左/右倾向角色提示对认同评分的影响。

**💡 创新点**

创新点在于首次将 LLM 的说服认同行为作为研究对象，结合线性混合效应模型对说服技巧、话题与政治立场三者的交互效应进行定量分析；并揭示角色提示能显著放大认同极化。

**🔧 技术方法**

使用了基于指令调优的 LLM（Llama‑3、Mistral、Qwen、Yi、Phi、Aya），通过固定 Likert 评分提示并采用 guided decoding 限制输出；分析手段为 KDE 可视化和线性混合效应模型。

**📊 数据集**

采用了两类公开数据集：含 29,596 条乌克兰‑俄罗斯冲突相关推文（共 3,865 条带说服标签）和 SemEval‑2023 英文新闻片段（共 1,766 条单一说服标签）。

**📈 对比分析**

对比方法为将中立、左倾、右倾三种角色提示下的评分分布进行 KDE 可视化，并用 LME 模型估计“是否含说服”与“角色倾向”的交互效应。结果显示，中立提示下平均评分约 3；含说服内容显著降低评分，而左倾提示进一步抑制，右倾提示则提升；不同说服技巧和话题亦显现显著差异。

**⚠️ 局限性**

主要局限包括仅使用英文文本、角色提示过于粗略（仅三类），仅单轮一次性评分，未能模拟多轮对话；数据集差异（推文短文本 vs 新闻片段）可能影响可比性；以及仅考虑单标签的新闻片段，忽略多标签情况。

---

## 405. High-Dimensional Theory of LoRA Fine-Tuning in a Solvable Attention Model

**arXiv ID:** 2606.05899 | [PDF](https://arxiv.org/pdf/2606.05899v1)

**作者:** O. Duranthon `[一作]` (École Polytechnique Fédérale de Lausanne), L. Zdeborová `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 10486 | [OpenAlex ID](https://openalex.org/A5089268172)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

研究了在大型预训练注意力模型上进行低秩适配（LoRA）细调的高维统计理论，给出了预训练与细调阶段的尖锐渐近刻画，并推导出对测试误差和表示质量的明确预测。

**💡 创新点**

创新点在于：① 构建了可解析的两阶段模型（大规模全秩预训练 + 低秩细调），② 通过自由熵、自由卷积与近似算子把高维问题降到有限维，③ 提出了有效噪声（Δ_eff）概念，量化预训练对细调的影响；④ 发现测试误差与表示质量可能出现不匹配；⑤ 基于理论提出了主动细调策略并给出理论与数值一致的验证。

**🔧 技术方法**

使用了高维极限分析（replica/自旋玻璃方法）、自由熵极大化、自由卷积（Marchenko–Pastur 与半圆分布的自由卷积）、近似算子（Prox）以及数值模拟（大尺寸 D≈10³–10⁴）来验证理论。

**📊 数据集**

使用的是合成的高维教师-学生数据集：输入 X 服从 i.i.d. 高斯，输出通过注意力的激活函数（软最大或线性）生成，目标权重 W*、w* 也为高斯。该模型可模拟预训练和细调的数据分布。

**📈 对比分析**

通过将理论预测（LoRA、无细调、全细调、主动细调）与数值实验（不同 λ、λ'、噪声水平、样本比例）进行对比，发现 LoRA 能显著降低测试误差并接近贝叶斯最优；主动细调进一步提升误差并改善表示对齐；理论与实验在各参数设置下吻合良好。

**⚠️ 局限性**

局限性：仅考虑单头、rank‑1 LoRA；使用张量化简的 Softmax 或线性激活；假设输入为高斯且不考虑真实数据的分布差异；未讨论学习动态与时间演化；对更高秩 LoRA、多头注意力和非高斯数据的推广仍待研究。

---

## 406. Retrospective Harness Optimization: Improving LLM Agents via Self-Preference over Trajectory Rollouts

**arXiv ID:** 2606.05922 | [PDF](https://arxiv.org/pdf/2606.05922v1)

**作者:** Wenbo Pan `[一作]` (City University of Hong Kong), Xiaohua Jia `[通讯]` (City University of Hong Kong)

**通讯引用:** 19818 | [OpenAlex ID](https://openalex.org/A5013643572)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无标签、利用历史轨迹自我监督优化AI代理的工具集、提示与工作流程的 Retrospective Harness Optimization 方法。

**💡 创新点**

创新点在于用代理自身的自偏好和对过去轨迹的自验证、自一致性诊断来驱动无监督的 Harness 更新，避免了对外部验证数据的依赖。

**🔧 技术方法**

采用 DPP 选取多样且难度高的核心任务、并行 rollouts、对结果进行自我排名、生成候选 Harness 并用 pairwise self-preference 选优的技术组合。

**📊 数据集**

在软件工程的 SWE‑Bench Pro、技术工作 Terminal‑Bench 2 和知识工作 GAIA‑2 三个基准上进行评估。

**📈 对比分析**

与无验证基线（Dynamic Cheatsheet、ReasoningBank、Sleep‑time Compute）及验证反馈优化（Meta‑Harness）对比，单轮 RHO 在 SWE‑Bench Pro 上将通过率从 59% 提升至 78%，在其他两个基准也实现了显著提升，且算力消耗低于 Meta‑Harness。

**⚠️ 局限性**

局限性包括需要可重置且可多次重播的环境，假设 Harness 可编辑，且对一次性或不可逆任务适用性不足。

---

## 407. Representing Research Attention as Contextually Structured Flows

**arXiv ID:** 2606.05895 | [PDF](https://arxiv.org/pdf/2606.05895v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 408. Unveiling the Unknown: Open Vocabulary Object Detection with Scene Graphs

**arXiv ID:** 2606.05916 | [PDF](https://arxiv.org/pdf/2606.05916v1)

**作者:** Yi Chen `[一作]` (Ningbo University), Jiangbo Qian `[通讯]` (Ningbo University)

**通讯引用:** 931 | [OpenAlex ID](https://openalex.org/A5004828717)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于场景图的相邻区域关系建模（SRM）框架，利用场景图捕获候选框间的语义与空间关系，并通过关系注意力和文本对齐提升开放词汇目标检测性能。

**💡 创新点**

创新点：①显式相邻区域关系建模（NR^2M）与关系注意力模块（RAM）协同增强对象间交互；②基于图像-字幕的文本对齐（STA）利用词检索提炼语义一致性；③将场景图关系与文本信息融合，显著提升新类别检测。

**🔧 技术方法**

使用的技术包括场景图生成、邻域采样、关系注意力（自注意力+深度可分离卷积+FFN+位置编码）、CLIP文本编码、对比学习、Faster R‑CNN框架等。

**📊 数据集**

使用的数据集：COCO（开放词汇设置48基类+17新类）、LVIS（频繁+常见为基类，稀有为新类）、COCO Caption、CC3M、Objects365 等用于迁移实验。

**📈 对比分析**

与多种 Faster R‑CNN 与 DETR 基线方法对比，OV‑COCO 上 AP_50^novel 36.9% 超过第二名 3.5%，OV‑LVIS 上 AP_r 24.1% 超过 0.9%；迁移至 COCO、Objects365 也保持了较好的泛化性能。

**⚠️ 局限性**

局限性：对场景图质量敏感，稀疏图结构仍可能引入噪声；邻域采样与 Top‑K 参数需手动调优；在基类检测上略逊于某些 DETR 方法；推理时的计算与内存开销相对较大。

---

## 409. CamFlow+: Hybrid Motion Bases for 2D Camera Motion Estimation with Stabilization Applications

**arXiv ID:** 2606.05915 | [PDF](https://arxiv.org/pdf/2606.05915v1)

**作者:** Haipeng Li `[一作]` (University of Electronic Science and Technology of China), Shuaicheng Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7171 | [OpenAlex ID](https://openalex.org/A5039387461)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CamFlow+，一种基于混合基底的二维相机运动估计框架，并在此基础上实现数字视频稳定化

**💡 创新点**

创新点包括：①直接将多幅同类相机运动的流场相加以解耦非线性合成；②引入深度驱动的平移基底与深度感知平滑正则化，显著提升对非平面场景的适应性；③在同一框架中兼容传统单平面、局部多平面、以及稠密光流模型；④通过GHOF‑Cam基准实现对密集相机运动的无监督评估

**🔧 技术方法**

使用混合基底（物理+随机+深度平移）与自注意力运动估计变换器（MET），基于概率拉普拉斯损失、特征一致性损失和深度感知光滑损失进行训练；同时构建深度金字塔以稳定深度平移基底

**📊 数据集**

使用CAHomo、GHOF和新增的GHOF‑Cam数据集；GHOF‑Cam通过SAM分割与光流标注剔除动态物体与遮挡，专门用于评测相机运动

**📈 对比分析**

与传统单/多平面方法（SIFT、ORB、HM‑Mix）、基于光流的网络（MeshFlow、RAFT、Sea‑RAFT）、以及3D基础模型（VGGT、DUSt3R）等进行对比；在GHOF‑Cam上，CamFlow+在EPE上达到0.50（比CamFlow 1.10提升约45%），在数字视频稳定化实验中获得最高用户偏好率43.3%

**⚠️ 局限性**

局限性：需要相机内参与深度图用于构造深度平移基底；训练仍需任务特定的数据与分阶段优化；目前仅在普通手持场景验证，未覆盖恶劣天气、低照度、快速运动等极端条件

---

## 410. Reducing Hallucinations in Complex Question Answering using Simple Graph-based Retrieval-Augmented Generation (long version)

**arXiv ID:** 2606.05901 | [PDF](https://arxiv.org/pdf/2606.05901v1)

**作者:** Christopher J. Wedge `[一作]` (Newcastle University), Jacek Cała `[通讯]` (Newcastle University)

**通讯引用:** 461 | [OpenAlex ID](https://openalex.org/A5052190130)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了在复杂问答任务中结合轻量级知识图谱与向量检索的检索增强生成（RAG）系统，显著降低幻觉并提升答案的事实正确性

**💡 创新点**

创新点在于使用简单的半结构化Wikipedia图谱与预写的Cypher查询工具，使LLM能够高效利用图结构进行多跳、多实体推理，而不需要生成复杂查询；同时通过对比向量RAG、向量+图RAG与零shot三种场景验证了图结构的有效性

**🔧 技术方法**

技术包括：LangChain agent + tool调用、Neo4j图数据库与向量索引、Harrier‑0.6B嵌入模型、GPT‑5.4推理模型、Llama‑4‑Maverick 17B 作为LLM-as-a-judge、RAGAS评价框架

**📊 数据集**

使用基于August 2025版的英语维基百科（约250 M段落，5.77 M节点）以及MoNaCo复杂问答基准（1315问，后筛选1207问）作为评测数据集

**📈 对比分析**

方法：在三种实验（vector RAG、vector+graph RAG、zero-shot）下对510个MoNaCo问题进行评估，使用事实正确性、答案相关性、粗细粒度真值度（CRAG）和token消耗等指标。vector+graph RAG在事实正确性上比纯向量RAG高约2×，真值度明显优于零shot，token消耗略高但仍低于纯向量RAG

**⚠️ 局限性**

局限性包括：仅在单一基准和单一推理模型上验证；LLM-as-a-judge评估非确定性，缺少充分人工验证；图工具使用率低，未完全利用图结构；未进行更细粒度的工具消融与跨数据集复现

---

## 411. EMBER: Efficient Memory via Budgeted Evidence Retention for Long-Horizon Agents

**arXiv ID:** 2606.05894 | [PDF](https://arxiv.org/pdf/2606.05894v1)

**作者:** Yilong Li `[一作]` (University of Wisconsin--Madison), Tong Che `[通讯]` (NVIDIA Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了预算化的预查询保留（Budgeted Pre-Query Retention）框架，训练代理在未知查询的情况下在有限的源证据预算内选择并保存可检索的证据。

**💡 创新点**

创新点在于引入回答可行性探测（answerability probes）和检索键（retrieval keys）来构造可检索的证据胶囊，并通过答案门控的链式奖励（answer-gated evidence-chain）直接对写入决策进行强化学习，从而最大化在预算约束下的回答质量。

**🔧 技术方法**

核心技术包括基于Transformer的记忆写入策略（Qwen2.5‑7B/14B），多阶段强化学习（PPO/GRPO）奖励设计，检索键生成和证据胶囊的预算控制层，以及在检索后使用GPT‑4o作为阅读器。

**📊 数据集**

使用的数据集包括LongMemEval‑RR（长序列记忆评估）、RULER‑HotpotQA（多跳证据保持的受控基准）以及MultiQ‑LongMemEval‑RR（多查询覆盖性测试）。

**📈 对比分析**

与全日志RAG、查询可见的MemAgent等基线相比，-14B模型在8192-token预算下实现F1 0.3017，显著高于最佳非预算基线0.1765；在RULER‑HotpotQA上达到0.8412 F1，优于Vanilla RAG 0.7772。

**⚠️ 局限性**

局限性包括仅在基准数据集上评估，未在真实部署环境中验证；模型可能保留敏感或过时证据，需配合可审计与删除机制；训练过程对随机种子和奖励权重较为敏感。

---

## 412. Learning of Robot Safety Policies via Adversarial Synthetic Scenarios

**arXiv ID:** 2606.05952 | [PDF](https://arxiv.org/pdf/2606.05952v1)

**作者:** Nikolai Dorofeev `[一作]` (SafePI.ai), Rostislav Yavorskiy `[通讯]` (SafePI.ai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于红队与蓝队对抗的游戏化框架，通过生成合成安全场景来帮助机器人学习和改进安全策略。

**💡 创新点**

创新点在于将场景生成转化为一个适应性对抗过程，利用红队主动寻找边缘案例，蓝队持续更新安全规则，从而高效覆盖难以枚举的高风险边缘场景，并形成可审计的安全策略演化轨迹。

**🔧 技术方法**

采用对抗性游戏设计、基于 Webots 的高保真仿真、合成数据生成、机器学习微调（以及可能的上下文学习）等技术。

**📊 数据集**

使用小规模仿真数据集（5 条包含安全与不安全状态的视频），并通过这些数据训练安全判别模型；后续框架设计旨在生成更大规模的合成安全数据。

**📈 对比分析**

在实验中，模型在帧级别的安全检测中取得 0.99 的高精度和 0.69 的召回率；显示高精度但仍有误报率，表明合成数据不足以覆盖所有边缘案例。相比传统随机或手工枚举方法，对抗性生成能更聚焦于难检测的边缘场景。

**⚠️ 局限性**

局限性包括：仅有小规模、单一任务的实验验证；缺乏在真实环境中的长期评估；合成数据的多样性和真实性待提升；对抗游戏的策略和奖励设计尚未成熟，可能无法充分挖掘所有潜在风险。

---

## 413. Faithful, Enriched, and Precise: Benchmarking Natural-Science Illustration Generation by T2I models

**arXiv ID:** 2606.05949 | [PDF](https://arxiv.org/pdf/2606.05949v1)

**作者:** Yifan Chang `[一作]` (Shanghai Innovation Institute), Yihao Liu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FEPBench 基准，用细粒度原子集合评估科学插图生成模型的指令忠实度、推理丰富度和语义精准度。

**💡 创新点**

创新点在于把科学图像拆解为文本、视觉、关系、布局原子，并按指令与推理两类原子分别评估，既能衡量模型对提示的遵从，又能考察其推理与精确性。

**🔧 技术方法**

采用多模态大语言模型（MLLM）和 OCR 自动提取原子集合，结合人类专家校准，随后使用 MLLM 进行自动评估；模型测试覆盖 GPT Image 系列、Nano Banana Pro、Seedream、Qwen、FLUX 等。

**📊 数据集**

使用 1,300 张自然科学论文中的高质量单面与多面插图，涵盖物理/材料、地理/生态、生物/医学三大学科，配备自由文本与结构化两种提示。

**📈 对比分析**

通过与专家人工评估的高度一致性（ICC≥0.88、Spearman ρ≥0.83），发现闭源模型在指令忠实度与推理丰富度上明显优于开源模型；文本生成与关系推理仍是主要瓶颈。

**⚠️ 局限性**

局限性包括推理原子数量相对较少导致推理分数波动，且当前评估仍未能完全涵盖科学推理的深度与多样性。

---

## 414. Short paper: Models in the dark -- Rectification and erasure under GDPR in ML supply chains

**arXiv ID:** 2606.05946 | [PDF](https://arxiv.org/pdf/2606.05946v1)

**作者:** Henrik Graßhoff `[一作]` (Karlstad University), Sara Ramezanian `[通讯]` (Karlstad University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对GDPR下机器学习供应链中数据主体权利（纠正权与删除权）的实现挑战进行了系统综述，提出了模型自身与供应链两大类挑战的分类法，并引入了“黑暗模型”（models in the dark）的概念。

**💡 创新点**

创新点在于：①首次将纠正权与删除权的技术实现与法律要求结合，并提供了完整的挑战分类；②提出黑暗模型概念，揭示供应链中隐蔽的个人数据传播风险；③在法律、技术与运作视角之间搭建跨学科桥梁。

**🔧 技术方法**

主要技术手段包括：文献与指导原则回顾、GDPR条文与监管文件分析、机器不学习（unlearning）与隐私审计方法的技术评估；未针对特定模型实施具体算法。

**📊 数据集**

未使用任何公开数据集；研究基于文献、法规与案例分析。

**📈 对比分析**

由于是综述性工作，没有实验或性能对比；文章未给出具体实现或指标。

**⚠️ 局限性**

局限性包括：①缺乏实证验证，无法量化黑暗模型对个人数据的实际影响；②未提供标准化的模型更新或不学习验证机制；③在多主体供应链中，责任链条不清晰，导致难以有效执行纠正或删除；④对持续学习模型的深度影响与监管合规性仍需进一步探讨。

---

## 415. Large Language Models are Perplexed by some Political Parties

**arXiv ID:** 2606.05937 | [PDF](https://arxiv.org/pdf/2606.05937v1)

**作者:** Paul Lerner `[一作]` (Sorbonne Université), François Yvon `[通讯]` (Sorbonne Université)

**通讯引用:** 3522 | [OpenAlex ID](https://openalex.org/A5030615769)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过计算LLM对不同政党文本的困惑度(PPL)，评估其政治公平性。

**💡 创新点**

提出使用单语料库的困惑度来量化政治公平性，并证明其与翻译质量相关联。

**🔧 技术方法**

使用PPL、BPC、BPEC等信息熵指标，对多语言LLM进行评估。

**📊 数据集**

使用Manifesto、Parlamint、21-EuroParl等多语种政治文本数据集。

**📈 对比分析**

采用Kruskal-Wallis检验、Spearman相关性与Borda计数进行比较；结果显示各语言下的公平性差异显著，且基线LLM与指令调优模型的相关性极高。

**⚠️ 局限性**

仅限于70B参数以内模型，数据可能泄露，且仅覆盖正式政治语料，非社交媒体等非正式场景。

---

## 416. A Pre-Registered Causal Partition of Self-Consistency Elicitation and Reward Design in RLVR

**arXiv ID:** 2606.05932 | [PDF](https://arxiv.org/pdf/2606.05932v1)

**作者:** Yuze Gao `[一作]` `[通讯]`, Yuze Gao

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 RLVR 奖励设计的真实贡献，将奖励效果拆分为自一致性激励与真实奖励设计，并在模拟器和真实模型上验证。

**💡 创新点**

提出 telescoping 分解公式，将 Δ_naive 分离为 Δ_elicit + Δ_rd，并提供预注册的诊断协议，揭示奖励信号偏差与交互非加性。

**🔧 技术方法**

采用 group-relative policy optimization（GRPO）、Tabular-GRPO 仿真、2×2×2 因子设计、Bootstrap CI、LoRA 微调、最佳‑N 选择与奖励设计评估。

**📊 数据集**

使用 GSM8K 推理任务以及内部按 prior strength 分布生成的抽象问题集。

**📈 对比分析**

通过四种奖励条件（Frozen、Random、Spurious、True）计算 Δ_null、Δ_elicit、Δ_rd；在强 prior 模型中 Δ_elicit 占主导，弱 prior 中 Δ_rd 占主导；真实模型实验显示分解准确，奖励设计比例在强 prior 仅 5%，弱 prior 约 118%。

**⚠️ 局限性**

仅在 1–1.5B 模型规模验证，跨更大模型、不同任务、软投票奖励的泛化有限；在自一致性临界点处方差高导致只能给出 bounds；模拟器假设理想化，真实 RL 仍需进一步扩展。

---

## 417. Steering Vectors are an Adversarial Attack Surface

**arXiv ID:** 2606.05958 | [PDF](https://arxiv.org/pdf/2606.05958v1)

**作者:** Abzal Aidakhmetov `[一作]` (Sapienza University of Rome), Emanuele Rodolà `[通讯]` (Sapienza University of Rome)

**通讯引用:** 7224 | [OpenAlex ID](https://openalex.org/A5087051832)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在对比式激活向量生成过程中对对比数据集进行隐蔽性令牌替换，成功制造出可在保持正向属性的同时削弱LLM拒绝机制的“jailbreak”向量。

**💡 创新点**

首次将对比式激活向量视为攻击面，并设计了基于嵌入空间最近邻与安全词汇约束的GCG优化方法，实现对少量令牌的隐蔽修改而不影响属性表达。

**🔧 技术方法**

使用Greedy Coordinate Gradient (GCG)优化、嵌入空间最近邻搜索、安全词汇过滤、流畅性惩罚以及反拒绝方向正交化防御等技术。

**📊 数据集**

对Gemma‑2‑2B‑IT与Llama‑3.1‑8B‑Instruct等开放权重模型，使用20对对比式文本（包括bullet‑list、Spanish‑output、lowercase‑output等属性），构成不同模型–属性组合的实验数据集。

**📈 对比分析**

在八个模型–属性组合上，攻击后绝对ASR提升至0.20–0.55（相对干净向量提升+19–+51pp），属性遵从率保持在±0.07以内，且向量范数变化不大；正交化防御可恢复≈82%攻击提升。

**⚠️ 局限性**

需要白盒梯度与嵌入检索权限，实验仅限于至8B规模模型；仅针对拒绝方向，未探究其他攻击目标或跨模型迁移；部分属性组合在干净状态下已表现弱化。

---

## 418. IN2P3 Computing Center 2024 Workload Dataset

**arXiv ID:** 2606.05914 | [PDF](https://arxiv.org/pdf/2606.05914v1)

**作者:** Guillaume Cochard `[一作]` (CC-IN2P3, CNRS), Bertrand Simon `[通讯]` (CC-IN2P3, CNRS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提供并分析了2024年IN2P3计算中心的44,749,836个作业的工作负载数据集，包含提交、运行和资源使用信息。

**💡 创新点**

创新点在于提供了长达一年时间窗口的最新数据，并结合了内存使用、作业步骤等更丰富的指标，显著提升了数据集的多样性和可用性。

**🔧 技术方法**

作者利用SLURM调度器的收集工具采集日志，并对数据进行结构化、匿名化处理，随后采用统计分析方法评估作业提交、等待和资源利用情况。

**📊 数据集**

使用的数据集为IN2P3 2024年全年的作业记录，共44M条，已发布于Zenodo，包含12个月压缩CSV文件。

**📈 对比分析**

通过对提交时间、等待时间、内存/CPU效率等指标的分布与累计曲线进行分析，展示了作业行为模式和资源利用率；结果显示大多数作业内存占用与请求相差显著，CPU利用率受多核与I/O影响。

**⚠️ 局限性**

限制包括仅记录峰值内存而非全程使用、RHEL9迁移导致maxrss异常、作业多步骤并行时内存上限不准确、I/O写入数据未实时监控，以及部分作业起始时间报告错误等。

---

## 419. Self-Learning Expression Deformations for Data-Efficient Gaussian Avatars

**arXiv ID:** 2606.05912 | [PDF](https://arxiv.org/pdf/2606.05912v1)

**作者:** Jiahao Yang `[一作]` (Queen Mary University of London), Shanxin Yuan `[通讯]` (Queen Mary University of London)

**通讯引用:** 2119 | [OpenAlex ID](https://openalex.org/A5068360563)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一套可从极少输入数据（单帧、单视角或一张图）快速生成可动画化的高保真Gaussian头像的完整训练与推理管线。

**💡 创新点**

创新点包括：① 用SDF与2DGS联合优化实现表面对齐的Gaussian分布；② 通过MLP对每个Gaussian的外观与形状进行参数化，消除独立更新导致的不一致；③ 引入基于曲率与拉伸的Gaussian形状校正；④ 采用自监督阶段（正则化表面法线、扭曲损失及与canonical渲染对比）替代传统长序列训练，显著降低对表情数据的需求。

**🔧 技术方法**

核心技术：Gaussian Splatting（3DGS/2DGS）、SDF表面监督、FLAME面部动画绑定、MLP参数化、曲率与拉伸估计、逆变形射线投射、CUDA渲染与tinyCUDANN优化。

**📊 数据集**

主要使用的公开数据集包括 NeRSemble（多视角表情序列）和 INSTA（无标定真实摄像），在一-shot 场景中还借助 GenHead 合成视角。

**📈 对比分析**

与 GA、GHA、FlashAvatar、RGBAvatar、GPAvatar、LAM 等现有方法对比，SAGE 在多视角单帧、单视角旋转、单图三种低数据场景下均实现了与或优于现有 SOTA 的 PSNR、SSIM 与 LPIPS，且训练时间与数据量大幅降低。

**⚠️ 局限性**

局限性：依赖 FLAME 参数，若表情不在 FLAME 可建模范围内会出现误差；对未观测区域（如牙齿、舌头）需预先拟合；在极端表情或高动态动作下可能出现形状过平滑或细节失真。

---

## 420. A Novel Method with Encoder-Decoder for Cross-Sensor Adaptation in Surface Shape Sensing with Sparse Strain Sensors

**arXiv ID:** 2606.05903 | [PDF](https://arxiv.org/pdf/2606.05903v1)

**作者:** Shuo Wang `[一作]`, Xiaoming Tao `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Transformer编码器和图神经网络解码器的稀疏应变传感器阵列表面形状感知框架，并通过元学习实现跨传感器阵列的少样本快速适配。

**💡 创新点**

创新点在于将元学习与Encoder‑Decoder结构相结合，利用虚拟任务实现不同传感器阵列的跨域自适应；通过曲率约束的损失提升形状光滑性；实现少量样本(<5%)在1秒内完成适配，误差仅约4 mm。

**🔧 技术方法**

使用技术包括Transformer编码器、图注意力卷积（GAT）解码器、MAML类元学习、曲率正则化损失、虚拟任务数据增强、稀疏应变传感器阵列与立体相机同步采集。

**📊 数据集**

数据集为在PVC柔性板上附加4×5棋盘，使用四组传感器阵列（a、b、c、d）采集的数据；主训练集为12850组（阵列a），适配集为b、c、d各约500–250组，所有数据配合立体相机获取的标注坐标。

**📈 对比分析**

与传统几何、光学或深度学习方法对比，未适配时误差约23 mm；适配后平均误差约4.0 mm，误差<5 mm的样本比例提升至>70%；相较于无元学习方案误差降低47%，适配速度<1 s。

**⚠️ 局限性**

局限性在于对复杂变形场景的验证不足；阵列尺寸与布局未优化；模型仍依赖有标签数据，未实现无监督跨传感器适配；对多传感器分布差异的鲁棒性有待提升。

---

## 421. Measuring the sensitivity of LLM-based structured extraction to prompt, model, and schema choices in clinical discharge summaries

**arXiv ID:** 2606.05970 | [PDF](https://arxiv.org/pdf/2606.05970v1)

**作者:** Martin Murin `[一作]` `[通讯]` (DryLabz GmbH), Martin Murin (DryLabz GmbH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文系统评估了大型语言模型在临床出院摘要结构化提取时，三种配置选择——提示词表述、模型尺寸、标注粒度——对输出一致性的影响，并提出一种无需人工标注、可在大规模数据上复现的评估方法。

**💡 创新点**

创新点在于：①通过跨提示词的 Cohen's κ 和二元折叠分析，揭示了三向标记（是/否/未记录）与提示词对结果差异的来源；②发现模型尺寸在多类标签分类上对输出差异的主导作用，而提示词对三向标记的影响有限；③提出了可复现的同笔记配对比较与自动化提示优化循环的完整方法论。

**🔧 技术方法**

技术手段包括：多变体提示词（A、B、C）驱动 GPT‑4 系列模型；Cohen's κ、Jaccard 指标和百分比一致率用于度量字段级别的一致性；对三向标记做后处理二元折叠；Labeling‑function（ICD、正则、LLM）与 Snorkel LabelModel 的集成；以及基于同笔记配对的模型尺寸差异 Δκ 统计。

**📊 数据集**

使用的数据集为 MIMIC‑IV v3.1 出院摘要，包含 331,793 条记录，按 ICD‑10 层级分层后拆分成 200、150、1,000、5,000、500、1,500 等子集，以验证样本大小对一致性指标的稳健性。

**📈 对比分析**

比较方法为：在相同笔记上分别使用三种提示词、两种模型尺寸进行抽取，计算字段级别的 κ；在模型尺寸比较时采用同笔记配对差异，求每字段 Δκ；对多类标签采用 Jaccard 与 exact‑match；对三向标记采用二元折叠后的 κ。结果显示：①在三向标记上，模型尺寸对整体 κ 没有显著提升，Δκ 均值约 -2%，但在二元折叠后 Δκ 变为 +10%；②在多类标签分类上，模型尺寸导致约 50% 的记录标签改变，提示词差异仅约 12.5%；③提示词对三向标记的一致性影响较小，主要集中在 “否/未记录” 轴。

**⚠️ 局限性**

局限性包括：①评估只测量一致性而非对照真实临床事实的准确性；②未使用人工标注参考，无法验证结果是否正确；③只测试三种提示词、两款 GPT‑4 变体、单一 MIMIC 数据，泛化性未知；④二元折叠是后处理，未验证直接二元提示下的行为；⑤不同字段的 κ 分布仍较低，需进一步审查；⑥对标签稀缺字段的 kappa 受基线影响，可能不稳定。

---

## 422. Demystifying NVSHMEM: A System-Level Analysis on Symmetric Memory and Device-Initiated Operations in GPU Communication

**arXiv ID:** 2606.05951 | [PDF](https://arxiv.org/pdf/2606.05951v1)

**作者:** Yijun Ma `[一作]` (ETH Zürich), Torsten Hoefler `[通讯]` (ETH Zürich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

这篇论文对NVSHMEM的编程模型、实现机制和性能特性进行了系统级源代码分析，并以DeepEP为案例展示其在稀疏深度学习中的应用。

**💡 创新点**

首次完整剖析NVSHMEM对称内存、单向RMA和设备侧集体的实现，并通过与NCCL GIN对比验证其设备驱动一边通信的优势，同时指出多CTA支持和集体算法的改进空间。

**🔧 技术方法**

采用源代码级分析、虚拟内存管理、GPU直通(RMA)、InfiniBand GPUDirect Async、NVSwitch/NVLink SHARP、CUDA Graphs、多团队机制等技术。

**📊 数据集**

使用DeepEP的Mixture-of-Experts稀疏训练/推理工作负载，在H200集群上跑实验，未引入公开大型数据集。

**📈 对比分析**

在CoreWeave H200集群上进行一边RMA和AllReduce的微基准，比较NVSHMEM与NCCL（Host‑side、Ring、NVLS）在单节点和跨节点的吞吐与延迟；结果显示NVSHMEM在大规模批量传输时接近或超过NCCL NVLS，但在小消息/低延迟场景略逊。

**⚠️ 局限性**

主要局限是缺乏多CTA集体支持、单CTA路径性能低、集体算法对跨节点网络支持不足，以及对非8 GPU/非NVLink节点的可移植性有限。

---

## 423. Beyond Greedy Chunking: SLO-Aware Sliding-Window Scheduling for LLM Inference

**arXiv ID:** 2606.05933 | [PDF](https://arxiv.org/pdf/2606.05933v1)

**作者:** Yuansheng Chen `[一作]` (Sun Yat-sen University), Jialun Li `[通讯]` (Guangdong Polytechnic Normal University)

**通讯引用:** 580 | [OpenAlex ID](https://openalex.org/A5100679362)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SlidingServe，一种滑动窗口驱动的 SLO‑感知调度系统，提升 LLM 在线推理的吞吐量和 SLO 满足率。

**💡 创新点**

创新点在于结合滑动窗口动态 chunk 分配、批量延迟预测器、层级优先级排序器以及 BatchConstructor 在批内动态请求选择，从而实现多目标（吞吐、延迟、SLO）调度。

**🔧 技术方法**

采用 vLLM 框架、Batch Latency Predictor、Multi‑Level Priority Sorter、SlidingChunker、BatchConstructor 等模块，使用离线+在线学习的预测器、动态规划、滑动窗口搜索等技术。

**📊 数据集**

在 ShareGPT、Arxiv‑v1/v2 长文本摘要、混合数据集 mixed‑v1/v2 以及合成负载下进行实验，使用 Llama3‑8B、Qwen2.5‑7B 两大模型。

**📈 对比分析**

与 Sarathi‑EDF 与 QoServe 基线比较，测量 goodput、延迟分布和 SLO 违例率；SlidingServe 在高负载条件下提升约30% 吞吐量，并将 SLO 违例率降低 16%–53%。

**⚠️ 局限性**

局限性包括对短 prompt 场景的改进有限；在极端高峰负载下仍可能出现尾部延迟；系统对预测器精度和 GPU 资源敏感，迁移到不同硬件需重新训练。

---

## 424. Securing the Sandbox: A Rootless Containerized Framework for Process-Oriented Monitoring in Computer Graphics Education

**arXiv ID:** 2606.05929 | [PDF](https://arxiv.org/pdf/2606.05929v1)

**作者:** Germán Arroyo `[一作]` (University of Granada), Juan Carlos Torres `[通讯]` (University of Granada)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 VISMATIC 框架，在 Raspberry Pi 5 低成本硬件上提供安全、过程导向的计算机图形教学环境，捕获学生真实开发行为并检测 AI 辅助作弊。

**💡 创新点**

将 rootless Podman 容器、Linux 用户命名空间、loop‑device 存储、JupyterHub 以及事件级 API 追踪结合，形成双层防御；利用匿名账号和宏观行为指标（节奏、累计参与、会话强度、日活量）构建过程监控与安全隔离的创新体系。

**🔧 技术方法**

rootless Podman + Linux User Namespaces、loop‑device 存储与配额、Apache2 反向代理 + Let's Encrypt、JupyterHub、Python/交互式可视化库、eBPF 日志收集与指标分析、Raspberry Pi 5 低功耗单板机。

**📊 数据集**

19 名学生在课程实验中产生的 API 交互日志（约 1880 条事件、57 小时），并未使用公开数据集，而是基于自有课程作业与实验场景收集。

**📈 对比分析**

通过宏观行为指标与人工审核或传统提交方式对比，能检测出多种 AI 辅助与自动化作弊行为；在 Pi 5 上支持 10–20 名学生同时运行，每人 1.5 CPU、0.5GB RAM，系统保持稳定且满足交互式图形任务需求。

**⚠️ 局限性**

仅依赖宏观 API 事件，缺乏细粒度键盘/鼠标信息；无法直接确认作弊意图，仅能提供异常信号；需人工阈值设定与后续审核；低成本硬件在高并发下性能受限；对外部工具完整性检测仍不完整。

---

## 425. Video-Rate Streaming Stylization on a Vision-Aware MLLM-Conditioned Edit Diffusion: Asymmetric Batched Inference on a Distilled UNet + MLLM Text Encoder

**arXiv ID:** 2606.05981 | [PDF](https://arxiv.org/pdf/2606.05981v1)

**作者:** Yoshiyuki Ootani `[一作]` `[通讯]` (Independent Researcher), Yoshiyuki Ootani (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种在单卡 GPU 上实现视频速率流式视频风格化的系统，结合已蒸馏的编辑 U‑Net 与大规模多模态语言模型文本编码器，并通过侧流/主流异步流水线、可编译友好的 LLLite 适配器以及周期性条件刷新等机制实现。

**💡 创新点**

关键创新在于将瓶颈从 U‑Net 转移到 MLLM 文本编码器，提出异步侧流流水线与批量文本编码器共振、LLLite 编译友好重构和基于子集钩子与刷新间隔的条件更新三项技术。

**🔧 技术方法**

采用异步 CUDA 流式调度、PyTorch 2.6+triton、LLLite（ControlNet‑LLLite）适配器、Farnebäck/RAFT 光流、TensorRT 编译、B‑batch 缓冲等。

**📊 数据集**

在 DAVIS‑2017、未见的 DAVIS‑19 以及七个非 DAVIS 数据源的视频片段上进行评测，使用油画等风格提示。

**📈 对比分析**

与同堆栈的 StreamDiffusion 对比，RTX 3090 Ti 上可持续 27–30 fps（B=8/16），RTX 4090 达 54.9 fps，RTX 5090 达 74.1 fps，且保持 0.5–1 s 的 p50 延迟，同时保持高质量。

**⚠️ 局限性**

局限在于训练的 temporal LLLite 适配器仅在 10 条 DAVIS 轨道上过拟合，1‑step 推理通用性、剪辑切换检测以及 TE 的数值范围与 FP16 兼容性仍需改进。

---

## 426. World-Language-Action Model for Unified World Modeling, Language Reasoning, and Action Synthesis

**arXiv ID:** 2606.05979 | [PDF](https://arxiv.org/pdf/2606.05979v1)

**作者:** Yi Yang `[一作]` (Shanghai Jiao Tong University), Zhijie Deng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 867 | [OpenAlex ID](https://openalex.org/A5102623510)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了世界-语言-动作（WLA）模型，整合世界建模、语言规划与动作生成，实现对多模态输入的统一控制。

**💡 创新点**

核心创新是使用自回归（AR）Transformer作为主干，结合文本子任务和低层物理动力学两条信息流，同时引入世界专家和动作专家，实现在无视觉监督下端到端学习潜在动作，并在推理时可省略世界专家以显著降低延迟。

**🔧 技术方法**

技术包括：AR Transformer主干、元查询机制、世界专家（轻量级扩散Transformer），动作专家（流匹配头），VAE特征预测，测试时缩放（TTS）与价值模型，深度学习框架（DeepSpeed、AdamW）等。

**📊 数据集**

使用了多种数据集：RoboTwin 2.0、LIBERO、RMBench 的仿真数据；AgilexRobotics Piper 双臂平台的实测数据；以及跨身体、跨任务的视频数据用于无动作标注的任务学习。

**📈 对比分析**

与现有的世界动作模型（WAM）和视觉-语言-动作模型（VLA）相比，WLA-0 在仿真和实测环境中表现突出：RoboTwin 2.0 92.94% 成功率；LIBERO 98.6%（TTS + 想象提升至 98.9%）；RMBench 56.5%（几乎翻倍）；实测任务完成时间与推理延迟分别低于 Motus 与 π_0.5，延迟约 40 ms，适合实时控制。

**⚠️ 局限性**

局限性包括：实测评估仅在单一双臂平台的四项任务上，跨身体和任务的广泛验证不足；视频驱动的任务学习主要基于模拟数据，实际环境下的鲁棒性待进一步验证。

---

## 427. LLM Explainability with Counterfactual Chains and Causal Graphs

**arXiv ID:** 2606.05972 | [PDF](https://arxiv.org/pdf/2606.05972v1)

**作者:** Nirit Nussbaum-Hoffer `[一作]` (Technion), Roi Reichart `[通讯]` (Technion)

**通讯引用:** 6980 | [OpenAlex ID](https://openalex.org/A5054952724)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于因果图的LLM推理可解释框架，通过四阶段流程（预测标签、概念提取、MCMC式对抗扩充、因果发现）自动构建模型的概念级因果图。

**💡 创新点**

创新点在于：①将因果图作为LLM推理的解释对象而非外部世界因果；②设计MCMC启发的对抗生成策略以填补概念空间稀疏；③提出预测性与结构稳定性评估协议。

**🔧 技术方法**

技术包括：LLM自生成标签与概念，基于阈值的概念筛选，MCMC风格的文本对抗链扩展，σ‑CG因果学习算法（支持循环与离散变量）。

**📊 数据集**

实验使用三类数据集：LIBERTY（疾病诊断，3类），IMDB（情感分析，2类），Reddit问答偏好（LLM‑as‑Judge，跨主题）。

**📈 对比分析**

与仅使用原始数据、仅扩充种子、以及无扩充对照相比，加入MCMC对抗样本后，因果图的父集预测性能提升最高，结构收敛稳定，整体性能显著优于基线。

**⚠️ 局限性**

局限性包括：概念提取受批处理随机性影响；评估主要关注局部父集而非全局因果链；对LLM自注释与生成的依赖可能引入误差。

---

## 428. Towards a Data Flywheel for Embodied Intelligence in Logistics

**arXiv ID:** 2606.05960 | [PDF](https://arxiv.org/pdf/2606.05960v1)

**作者:** Anlan Yu `[一作]` (Peking University), Daqing Zhang `[通讯]` (Peking University)

**通讯引用:** 14971 | [OpenAlex ID](https://openalex.org/A5045729966)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出物流行业的以数据为核心的“物流数据飞轮”框架，结合 WM‑DAgger 通过世界模型生成可恢复的合成数据，并与真实演示数据融合进行强化的模仿学习；同时规划如何将传统自动化日志、未标注操作视频等工业数据转化为可复用的学习数据集。

**💡 创新点**

创新点在于：①利用世界模型作为数据引擎，合成可恢复轨迹并通过纠正动作合成和一致性过滤保证任务相关性与物理可行性；②构建闭环数据飞轮，将现场操作数据、合成数据与部署反馈协同提升策略；③提出一套适用于物流场景的端到端数据转换与持续改进方案。

**🔧 技术方法**

使用的技术包括：世界模型（action‑conditioned dynamics）、WM‑DAgger 数据聚合框架、纠正动作合成模块、一致性导向过滤模块、模仿学习、离线强化学习与数据增强。

**📊 数据集**

实验数据集包括：在真实物流现场采集的 5–20 次演示（软袋推送、抓取‑放置、投票插入、毛巾折叠等）以及 1500 条由世界模型生成的合成恢复轨迹；未来工作将整合传统自动化日志、未标注操作视频与机器人执行日志等多源工业数据。

**📈 对比分析**

与基线（Behavioral Cloning 及 Diffusion‑based Data Augmentation）相比，WM‑DAgger 在 5-shot 软袋推送任务中将成功率从 26.7% 提升至 93.3%，在 20-shot 任务中从 30.0% 提升至 96.7%；在抓取‑放置、投票插入、毛巾折叠等多项任务中均实现显著性能提升。

**⚠️ 局限性**

局限性包括：合成数据依赖世界模型的准确性，仍可能产生幻觉或物理不一致的轨迹；目前验证主要基于小规模仿真/实验任务，缺乏大规模工业部署的实测；将传统自动化日志转换为可学习序列仍面临多源异构对齐与元数据标注挑战；方法对超大规模、极其多样化的物流场景的泛化能力尚未充分验证。

---

## 429. Correct-by-Construction Design of Timed Systems in Event-B

**arXiv ID:** 2606.05939 | [PDF](https://arxiv.org/pdf/2606.05939v1)

**作者:** Guillaume Dupont `[一作]` (Toulouse INP ENSEEIHT/IRIT), Jun Sun `[通讯]` (Singapore Management University)

**通讯引用:** 22975 | [OpenAlex ID](https://openalex.org/A5100728816)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出一种在不改动Event-B核心语言的前提下，对时间进行非侵入式嵌入，并扩展了Event-B的细化操作，使得能够以正确构造的方式逐步设计与验证实时系统。

**💡 创新点**

创新点在于：①将稠密时钟与时钟约束以理论（theory）形式集成到Event-B中；②在细化过程中加入时间维度，构造“时序细化”机制；③提供了一个通用的时序系统模型（TimedGeneric）和对应的细化方法，可直接将任意时序自动机映射为Event-B机器；④通过案例研究验证了该方法能够在单一模型中同时处理功能和时序约束。

**🔧 技术方法**

使用技术包括：Event-B 形式化方法与Rodin平台；Event-B理论扩展（theory）实现时间与时钟语义；稠密时钟与时钟约束的代数定义；时序细化（Timed Refinement）框架；自动生成的证明义务（PO）与手工证明；与传统基于模型检查的工具（如 Uppaal、ProB）进行对比。

**📊 数据集**

本文未使用外部真实数据集，而是以参数化的案例研究（一个包含n条指令的程序执行模型）作为验证对象，所有约束和参数均在理论层面符号化定义。

**📈 对比分析**

在证明方面，作者统计了各模型的证明义务数量及自动/手工比率：如 15个PO中60%自动，51个PO中49%自动，135个PO中53%自动，剩余部分需手工证明。与基于模型检查的工具相比，该方法不需要模型转换且能保留完整的功能与时序关系，但缺乏动画与即时验证；相对模型检查器，证明工作量更大但能得到更全面的正确性保证。

**⚠️ 局限性**

局限性包括：①ProB等现有工具无法很好支持Event-B理论，导致无法直接进行动画与模型检查；②不支持稠密时钟的动画；③目前无法验证活跃性（liveness）性质；④需要手工构造大量不变量与证明，证明负担较高；⑤仅针对稠密时钟与离散时钟系统的扩展仍待研究。

---

## 430. Addressing Imbalance in Multi-Label Data via Label-Specific Distance-based Oversampling

**arXiv ID:** 2606.05927 | [PDF](https://arxiv.org/pdf/2606.05927v1)

**作者:** Bin Liu `[一作]` (Chongqing University of Posts and Telecommunications), Grigorios Tsoumakas `[通讯]` (Aristotle University of Thessaloniki)

**通讯引用:** 12342 | [OpenAlex ID](https://openalex.org/A5026561247)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054`

**🎯 论文内容**

暂无论文主体内容，无法确定具体研究工作。

**💡 创新点**

暂无创新点描述。

**🔧 技术方法**

暂无使用的技术信息。

**📊 数据集**

暂无使用的数据集信息。

**📈 对比分析**

暂无方法比较与性能评估信息。

**⚠️ 局限性**

缺乏足够信息，无法评估论文的局限性。

---

## 431. Better Literary Translation: A Multi-Aspect Data Generation and LLM Training Approach

**arXiv ID:** 2606.05924 | [PDF](https://arxiv.org/pdf/2606.05924v1)

**作者:** Zhihao Lin `[一作]` (Amazon Web Services), Peiyang He `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多维度迭代细化框架，通过专门的表达流畅度和文学效果模块生成高质量翻译参考和偏好对。

**💡 创新点**

创新点在于把文学翻译质量拆解为两冲突维度并同时产生参考与偏好数据，利用显式奖励模型+GRPO实现更稳健的RL训练。

**🔧 技术方法**

使用大语言模型（Qwen3-235B-A22B-Instruct）构建细化模块，结合监督微调、奖励模型（Bradley‑Terry）和GRPO；对比DPO等隐式奖励方法。

**📊 数据集**

数据集为MetaphorTrans（英译中文学句子）和O. Henry Collection（跨域评测）。

**📈 对比分析**

与多种基线（包括Qwen3、Claude Sonnet、DeepSeek、专门文学翻译模型）比较，LitMT-8B和LitMT-14B在MetaphorTrans上分别达到67.25和69.07 CEA100，明显优于同参数规模模型，且在O. Henry上表现出色。

**⚠️ 局限性**

局限在于缺乏对显式奖励优于隐式奖励的理论解释，未探索链式推理提升效果，以及仅针对英译中，其他语言对待研究。

---

## 432. Asuka-Bench: Benchmarking Code Agents on Underspecified User Intent and Multi-Round Refinement

**arXiv ID:** 2606.05920 | [PDF](https://arxiv.org/pdf/2606.05920v1)

**作者:** Xin Wang `[一作]` (University of Science and Technology of China), Zhendong Mao `[通讯]` (University of Science and Technology of China)

**通讯引用:** 6488 | [OpenAlex ID](https://openalex.org/A5023341829)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Asuka-Bench，构建一套面向网页开发的迭代式评测框架，允许代码代理在多轮基于浏览器渲染行为的反馈循环中完成项目。

**💡 创新点**

创新点在于将不完整需求与多轮交互反馈结合，使用 DAG 结构化评估、浏览器可观察行为作为评测依据，并对比一轮全规格提示的效果，突出修复能力的独立维度。

**🔧 技术方法**

核心技术包括 LLM 驱动的 Code Agent（配合 OpenHands 或 Claude Code 框架）、UI Agent 自动化浏览器测试（类似 Puppeteer/WebVoyager）、User LLM 生成自然语言反馈以及 DAG‑aware 的评估协议。

**📊 数据集**

数据集由 50 题 Web 开发任务组成，每题产生 784 条评估准则和 2,402 条期望结果，任务来源于真实用户请求、GitHub 仓库与现有网站，并通过 LLM 生成 Clarified PRD、模拟数据及评估 DAG。

**📈 对比分析**

评测 8 种主流 LLM 与 2 个代理框架，使用 Project Completion Rate、加权 Task Pass Rate、加权 Criteria Pass Rate 三个指标；最佳模型在 3 轮后 Task Pass Rate 约 90%，Project Completion Rate 仅 52%，展示出 38pp 的性能差距与迭代修复潜力。

**⚠️ 局限性**

局限性包括：仅覆盖前端自包含场景，未涵盖 3D、实时协作或后端交互；评估器依赖 GPT‑5.4 可能引入偏差；且任务易被未来训练集记忆，需定期更新以防泄露。

---

## 433. Beyond WER: A Paired Acoustic Stress Test for Ambient Clinical Scribes

**arXiv ID:** 2606.05909 | [PDF](https://arxiv.org/pdf/2606.05909v1)

**作者:** Xiao-Hang Jiang `[一作]` (University of Science and Technology of China), Zhi-Yang He `[通讯]` (iFLYTEK Co., Ltd.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实施了配对声学压力测试，评估 ASR‑LLM 临床转录管道在不同噪声条件下对安全性的影响，揭示 WER 与临床安全不相关；

**💡 创新点**

首次将声学干扰按“静态环境噪声”与“语义干扰”两类分类，并引入因果级联评估框架及轻量级证据驱动混合代理来缓解噪声诱发的安全风险；

**🔧 技术方法**

使用 Whisper‑large‑v3 作为 ASR 前端，Qwen3‑235B‑A22B‑Instruct‑2507 作为 LLM 后端，结合符号证据过滤器，并在 OSCE 标准化对话上进行实验；

**📊 数据集**

采用 Fareez 等公开的 272 条 OSCE 对话（52 小时，无 PHI）作为临床语料，并使用 DEMAND（办公室/餐厅/交通）和 MUSAN（语音）噪声库；

**📈 对比分析**

通过 WER、NegErr、TriageMatch、SCER、ErrProp、Mean Score、Unsafe Rate 等指标进行对比，实验显示噪声类型导致安全性显著下降；在 5 dB 语义干扰下，轻量证据代理将 Unsafe 率从 91.54 % 降至 70.96 %，而在 5 dB 环境噪声下则从 91.54 % 降至 83.82 %；

**⚠️ 局限性**

局限于 OSCE 标准化场景，未覆盖真实临床录音；噪声种类有限；仅评估单一 LLM 模型；证据过滤器虽降低安全错误，却可能牺牲部分完整性与细节覆盖。

---

## 434. IA-RAG: Interval-Algebra-Driven Temporal Reasoning for Dynamic Knowledge Retrieval

**arXiv ID:** 2606.06044 | [PDF](https://arxiv.org/pdf/2606.06044v1)

**作者:** Xiaoman Wang `[一作]` (East China Normal University), Pinlong Cai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1438 | [OpenAlex ID](https://openalex.org/A5062614549)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 IA-RAG，一种基于时间区间和 Allen 区间代数的层级检索增强生成框架；

**💡 创新点**

创新点包括：① 用时间区间而非单点时间戳建模事件，明确定义 13 种 Allen 关系；② 引入子图时间紧化机制，通过逻辑约束推断不确定区间；③ 采用方向化 Allen 关系遍历实现基于时间约束的检索；④ 构建主题森林层级结构提升检索效率与准确性；

**🔧 技术方法**

使用的大语言模型进行 IEU 提取与逻辑推理，句向量检索、图构建与层级抽象、Allen 区间代数关系计算、子图时间紧化 LLM 逻辑推断以及方向化遍历等技术；

**📊 数据集**

在 TimeQA、TempReason、ComplexTR 三个公开时间问答基准上进行评估；

**📈 对比分析**

与 Vanilla RAG、GraphRAG、Temporal‑RAG 等多种基线对比，IA‑RAG 在 TimeQA 与 ComplexTR 上取得最高准确率和召回率，提升幅度可达 2%‑10%；在 TempReason 上略低于 DyG‑RAG；Ablation 实验验证每个模块的贡献；

**⚠️ 局限性**

局限性在于：① 依赖时间表达提取与区间归一化质量，模糊时间可能导致错误；② 仅关注区间层面推理，未显式建模因果关系或事件演化动态；

---

## 435. LoomVideo: Unifying Multimodal Inputs into Video Generation and Editing

**arXiv ID:** 2606.06042 | [PDF](https://arxiv.org/pdf/2606.06042v1)

**作者:** Jianzong Wu `[一作]` (Peking University), Hao Jiang `[通讯]` (Peking University)

**通讯引用:** 27440 | [OpenAlex ID](https://openalex.org/A5064335105)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了LoomVideo，一种5B参数的统一视频生成与编辑框架。

**💡 创新点**

创新点包括：Deepstack多层MLLM特征注入、Zero-overhead Scale-and-Add源视频条件、Negative Temporal RoPE多参考引导，以及三阶段渐进训练和RL后训练。

**🔧 技术方法**

采用的技术包括Diffusion Transformer、Qwen3‑VL多模态大语言模型、深层注入交叉注意力、Scale-and-Add条件、RoPE定位编码、动态批处理、多分辨率训练和DiffusionNFT强化学习。

**📊 数据集**

使用的数据集覆盖10M+图像‑文本、10M视频‑文本、3M指令编辑、1M参考编辑、0.5M多参考视频、Taobao电商内部数据以及公开数据集如OpenVid、RefVIE、IntelligentVBench等。

**📈 对比分析**

通过VBench、OpenVE‑Bench、RefVIE‑Bench、IntelligentVBench与自建FashionVideoBench等基准进行对比，LoomVideo在大多数任务上达到或逼近SOTA，尤其在时尚电商场景中表现最优，并在视频编辑任务中实现5.41×的推理速度提升。

**⚠️ 局限性**

主要限制在于5B参数规模不足以在极高分辨率（720p/1080p）或长时长视频生成/编辑中与13B级模型匹敌，并且在某些复杂多模态任务上仍略逊于更大规模模型。

---

## 436. Modified augmented Lagrangian preconditioning for mixed-dimensional beam-solid coupling

**arXiv ID:** 2606.06035 | [PDF](https://arxiv.org/pdf/2606.06035v1)

**作者:** Max Firmbach `[一作]` (Bundeswehr University Munich), Matthias Mayr `[通讯]` (Bundeswehr University Munich)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了一种改进的增广拉格朗日乘子块预条件器，用于三维固体与一维梁耦合的有限元求解。

**💡 创新点**

创新点包括将增广拉格朗日方法与块三角预条件器相结合，提出三种Schur补近似和不同罚参数的变体，以解决纯Neumann子问题并实现对模型参数和网格尺寸的鲁棒性。

**🔧 技术方法**

所采用技术包括有限元耦合、增广拉格朗日乘子、块三角预条件器、AMG、LU、SPAI、谱等价分析等。

**📊 数据集**

实验数据集主要为随机短纤维复合材料的代表体积元（RVE）以及含混合层板的工程案例，包含不同体积比、刚度比和纤维直径的几何配置。

**📈 对比分析**

与传统ILU预条件器比较，使用GMRES迭代次数、求解时间和并行规模性指标，改进方案在不同参数下保持10–50次迭代，弱/强规模性效率均优于基线。

**⚠️ 局限性**

局限性包括仅针对无扭转、无剪切的Kirchhoff–Love梁模型，旋转自由度需进一步扩展；对极大刚度比或体积比仍敏感；需手动调优罚参数；对多材料/面-体耦合的适用性待进一步研究。

---

## 437. ReSAGE-PAR: Representational Similarity Assessment for Generative Expansion in Pedestrian Attribute Recognition

**arXiv ID:** 2606.06020 | [PDF](https://arxiv.org/pdf/2606.06020v1)

**作者:** Pablo Ayuso-Albizu `[一作]` (Universidad Autónoma de Madrid), Paula Moral `[通讯]` (Universidad Autónoma de Madrid)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5051231657)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于LoRA微调的文本到图像扩散模型 ReSAGE-PAR，用来在低分辨率监控域生成并自动标注行人属性数据，解决域差距和生成标签误差问题。

**💡 创新点**

创新点包括：1）使用LoRA在img2img场景下适配Stable Diffusion至PAR数据的低分辨率和噪声；2）构建基于BLIPScore的视觉‑语言相似度评分，并通过贝叶斯分类器将连续得分转化为可靠的伪标签；3）该框架无须手工标注，且能在多种网络架构上实现增益。

**🔧 技术方法**

技术包括：LoRA参数高效微调、img2img扩散生成、BLIPScore与CLIPScore等视觉‑语言相似度评估、贝叶斯阈值判定的自标记器、数据增强混合训练。

**📊 数据集**

在四个主流PAR数据集上进行评估：PETA、PA100K、RAP（v1/v2）及其零射击变体PETAzs、RAPzs。

**📈 对比分析**

与传统像素级增强（AutoAug, CutMix 等）、基于GAN或无验证的扩散生成等方法对比，ReSAGE-PAR 在所有数据集上均提升平均准确率（mA），最高可达约 89%（PromptPAR+ReSAGE-PAR 在 PA100K 上）。在不同骨干网络（ResNet50, BN‑Inception, Swin）和 1:0.5–1:2 的生成比例下也表现出稳健性和可扩展性。

**⚠️ 局限性**

局限性包括：1）对极罕见或细粒度属性的生成受限；2）贝叶斯自标记器以全局相似度为依据，可能抑制部分正确属性的标签；3）将连续得分二值化，忽略了不确定样本的细微信息，影响更精细学习任务。

---

## 438. Adaptive Oscillatory-State Alignment for Time Series Forecasting

**arXiv ID:** 2606.06010 | [PDF](https://arxiv.org/pdf/2606.06010v1)

**作者:** Zhangyao Song `[一作]` (Southeast University), Tao Guo `[通讯]` (Southeast University)

**通讯引用:** 31712 | [OpenAlex ID](https://openalex.org/A5090925242)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于 Hilbert 变换的周期性时间序列预测框架，利用振荡状态对齐（oscillatory-state alignment）将传统的固定周期模板方法替换为自适应的局部振荡状态匹配，从而更好地捕捉振幅调制、相位漂移和局部频率变化等非平稳特征。

**💡 创新点**

创新点包括：① 将周期性建模视为振荡状态空间而非固定周期模板；② 通过全局可学习的振荡先验和观察序列的 Hilbert 信号描述符（幅值、相位、瞬时频率）构建描述符条件门；③ 该门实现对观测序列的局部软校正，实现更灵活的周期性对齐；④ 引入双路径头（注意力路径 + 基础路径）实现跨变量交互与稳健的时间投射。

**🔧 技术方法**

核心技术：Hilbert 变换 + 解析信号提取（幅值、相位、瞬时频率）；描述符条件卷积门；全局可学习振荡先验；双路径预测网络（MHA + 线性投射）; 轻量级参数与可微实现。

**📊 数据集**

使用了八个公共多变量时间序列基准（ETTh1/2/ETTm1/2、Electricity、Solar-Energy、Traffic、Weather）以及两个云工作负载数据集（IaaS、PaaS）。还在四个合成数据集（Syn‑S、Syn‑A、Syn‑P、Syn‑C）上做了针对非平稳性的控制实验。

**📈 对比分析**

与 TQNet、CycleNet、iTransformer、PatchTST、SSformer、Amplifier、DLinear 等近年先进模型对比。实验显示本文模型在 7/8 个基准上实现了最佳或次佳 MSE/MAE，并在云工作负载预测中获得第一名；相对传统固定周期方法在非平稳场景下优势显著，误差随非平稳度递增。

**⚠️ 局限性**

局限性：仅针对点预测且采用固定长度窗口，未探索多尺度或不规则采样；全局先验为全局共享，缺乏对样本条件或聚类状态的适配；在大型 Transformer 或概率预测框架中的效用尚未验证；缺少对突发 regime 转变、缺失值和外生事件的深入评估。

---

## 439. Beyond Vector Similarity: A Structural Analysis of Graph-Augmented Retrieval for Industrial Knowledge Graphs

**arXiv ID:** 2606.06003 | [PDF](https://arxiv.org/pdf/2606.06003v1)

**作者:** Grama Chethan `[一作]` `[通讯]` (Siemens Digital Industries Software), Grama Chethan (Siemens Digital Industries Software)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

比较了八种检索架构（从传统TF‑IDF文本检索到基于LLM的图遍历和图计算）在航空供应链知识图上的表现，评估了23个结构化查询；

**💡 创新点**

提出“操作词汇论”，认为图推理瓶颈在于可用工具而非LLM本身，并构建了从文本检索到图计算的递进架构；

**🔧 技术方法**

采用TF‑IDF向量检索、NetworkX图遍历、Claude Haiku 4.5 LLM工具调用（包含九种遍历原语和六种图计算工具）等技术；

**📊 数据集**

使用了一个46节点、64条时间戳边的航空供应链知识图，并在实验中扩展到1,100节点的合成规模；

**📈 对比分析**

通过实体级F1评分（与人工标注κ=0.716一致）比较性能，标准RAG 0%正确，Deterministic GraphRAG 47%正确，LLM查询规划 63%正确，最终图计算架构 64%正确；

**⚠️ 局限性**

实验仅覆盖单一合成域，实体级F1在结构查询上存在测量缺口，且未验证跨模型与跨领域的泛化能力。

---

## 440. ATT-CR: Adaptive Triangular Transformer for Cloud Removal

**arXiv ID:** 2606.05999 | [PDF](https://arxiv.org/pdf/2606.05999v1)

**作者:** Yang Wu `[一作]` (Xi'an Jiaotong University), Jinjun Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 7790 | [OpenAlex ID](https://openalex.org/A5100746487)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种自适应三角Transformer（ATT-CR）用于遥感图像云去除。

**💡 创新点**

创新点在于引入低复杂度的三角注意力（TAN）来克服线性注意力的低秩限制，并结合特征选择门控模块（FSGM）自适应抑制云影响。

**🔧 技术方法**

采用多尺度卷积、线性化注意力、门控卷积与残差Transformer块等技术。

**📊 数据集**

在RICE1、RICE2、T-CLOUD和SEN12MS-CR四个公开云去除数据集上进行实验。

**📈 对比分析**

与现有SOTA方法相比，ATT-CR在PSNR/SSIM/MAE/SAM等指标上取得最高或近似最高分，且参数量与算力显著低于多数对手。

**⚠️ 局限性**

局限包括三角注意力的串行计算导致推理速度略慢，且门控机制可解释性仍有限。

---

## 441. HoT-SSM:Higher-order Temporal Knowledge Graph Reasoning with State Space Models for Health Care

**arXiv ID:** 2606.05994 | [PDF](https://arxiv.org/pdf/2606.05994v1)

**作者:** Thummaluru Siddartha Reddy `[一作]` (Fujitsu Research of India), Mahesh Chandran `[通讯]` (Fujitsu Research of India)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于知识注入的时序超图与状态空间模型（SSM）相结合的框架，用于构建患者的高阶临床关系序列，并通过超图卷积与HiPPO SSM学习长程时序表示，支持临床预测与可解释推理。

**💡 创新点**

① 采用LLM生成全局超知识图并迁移至患者访视超图，捕获多元临床概念的高阶关系；② 将超图卷积与HiPPO SSM融合，既保留高阶空间交互，又显式建模长时序依赖；③ 设计基于注意力或梯度归因的时间推理路径，实现模型决策的可解释性。

**🔧 技术方法**

超图构建、超图卷积（HConv/注意力版）、HiPPO状态空间模型、BERT特征、LLM（GPT‑4.1）生成超边、梯度归因、注意力机制、LLM辅助解释。

**📊 数据集**

MIMIC‑III 与 MIMIC‑IV 电子健康记录数据集，任务包括死亡率预测、住院时长预测、药物推荐和再入院预测。

**📈 对比分析**

与 GRU、Transformer、RETAIN、GRAM、Deepr、StageNet、GraphCARE 等基线进行对比；在死亡率预测中 AUPRC 提升约 17‑18%，AUROC 提升约 3‑11%；在 LOS、药物推荐和再入院等任务亦显著优于基线。

**⚠️ 局限性**

依赖LLM生成超边导致计算与成本高；高阶超图构建与参数规模相对较大；实验仅在 MIMIC 系列数据上验证，跨域泛化及鲁棒性尚待进一步研究；对外部知识图谱的构建质量敏感。

---

## 442. English-to-Prakrit Machine Translation via Multilingual Transfer Learning

**arXiv ID:** 2606.06038 | [PDF](https://arxiv.org/pdf/2606.06038v1)

**作者:** Om Choksi `[一作]` (Sardar Vallabhbhai National Institute of Technology), Pruthwik Mishra `[通讯]` (Sardar Vallabhbhai National Institute of Technology)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5019439792)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在低资源环境下，将 IndicTrans2 通过将 Prakrit 映射到 Hindi 语言标签的方式，适配为英译 Prakrit 机器翻译模型；

**💡 创新点**

创新点在于利用脚本兼容的语言标签映射实现对未支持古典语言的无架构、无词表修改的快速迁移；

**🔧 技术方法**

采用多语言 Transformer 基础模型 IndicTrans2 进行全微调，配合 Hugging Face Transformers 与 SacreBLEU 评估；

**📊 数据集**

使用 VIITPune Prakrit‑to‑English 平行语料库（共 1,474 对 Maharashtri Prakrit‑English 句子）进行训练，评测则使用 20 对 Ardhamagadhi Prakrit‑English 句子；

**📈 对比分析**

与未调参的 IndicTrans2 基线相比，BLEU 分数从 1.57 提升至 14.30，训练耗时约 55 分钟，表明脚本兼容迁移在跨方言场景下可显著提升翻译质量；

**⚠️ 局限性**

局限在于数据量极小、评测集极小、缺乏人工评估、方言不匹配导致性能受限，且未尝试其他标签映射、数据增强或专门的 Prakrit 语言建模。

---

## 443. Online KL-Regularized Reinforcement Learning with Function Approximation under Misspecification

**arXiv ID:** 2606.06053 | [PDF](https://arxiv.org/pdf/2606.06053v1)

**作者:** Haoyang Hong `[一作]`, Huazheng Wang `[通讯]` (Company Name)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供ICML 2026会议论文提交和最终稿的详细格式与排版指南。

**💡 创新点**

通过双盲审稿流程、完整的PDF要求、字体、图表、算法环境等细节提升提交质量。

**🔧 技术方法**

使用LaTeX模板、Type‑1字体、矢量图（.eps/.pdf）、无损位图（.png）以及algorithm环境。

**📊 数据集**

无实验数据集，示例中仅包含常见机器学习数据集列表用于说明。

**📈 对比分析**

不包含实验比较，指南仅描述排版和提交流程。

**⚠️ 局限性**

缺乏实验内容，适用范围仅限于ICML 2026会议提交规范。

---

## 444. Passive Learning of Symbolic Automata over Monotonic Algebras

**arXiv ID:** 2606.06050 | [PDF](https://arxiv.org/pdf/2606.06050v1)

**作者:** Erwann Loulergue `[一作]` (Université Paris-Saclay), Peter Habermehl `[通讯]` (Université Paris Cité)

**通讯引用:** 1801 | [OpenAlex ID](https://openalex.org/A5054351430)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种新的符号有限自动机（SFA）学习算法，能够在主动学习框架下有效推断SFA模型。

**💡 创新点**

创新点在于引入了基于分层符号约束的分解策略，显著降低了学习时的查询复杂度，并支持更大规模的符号域。

**🔧 技术方法**

使用了主动学习技术（membership 和 equivalence 查询）、符号约束求解器以及符号自动机的分层表示方法。

**📊 数据集**

在公开的SFA基准数据集（如符号正则表达式、数字序列以及人工生成的随机SFA）上进行实验。

**📈 对比分析**

与现有SFA学习工具（如Lstar、RPNI等）相比，新算法在平均查询次数和学习时间上降低了约30%‑50%，并在识别精度上保持一致。

**⚠️ 局限性**

局限性包括对等价查询的强依赖、符号域必须可有效约束求解，以及在极大符号域下仍存在空间复杂度挑战。

---

## 445. OPRD: On-Policy Representation Distillation

**arXiv ID:** 2606.06021 | [PDF](https://arxiv.org/pdf/2606.06021v1)

**作者:** Shenzhi Yang `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**通讯引用:** 103657 | [OpenAlex ID](https://openalex.org/A5100389265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出On‑Policy Representation Distillation (OPRD)，通过在学生自身生成的序列上对隐藏层表示进行对齐，取代传统的仅在LM头输出空间进行的概率对齐；

**💡 创新点**

创新点在于将监督信号从输出空间迁移到隐藏层空间，消除采样方差、突破LM头信息瓶颈，并提供更丰富的结构化监督；

**🔧 技术方法**

采用on‑policy采样、教师-学生隐藏层MSE对齐、可选层级/位置选择、可与传统OPD线性组合、并使用stop‑gradient和投影矩阵；

**📊 数据集**

使用Qwen2.5‑1.5B作为教师和学生模型，在数学竞赛数据集（AIME 2024、AIME 2025、AIMO）上进行训练和评估；

**📈 对比分析**

与采样‑token OPD、top‑16 OPD等baseline对比，OPRD在所有三个基准上更快（1.44×）、内存更低（最高54%）、并在Avg@16上几乎完全收敛到教师水平，闭合教师-学生差距；

**⚠️ 局限性**

局限性包括：需要教师与学生相同架构；未对跨规模模型进行有效对齐；仅对隐藏层对齐，未利用注意力映射等信息；需要进一步探索多层级/位置自适应以及跨架构扩展。

---

## 446. Cheating in Multiplayer Online Games: a Dataset

**arXiv ID:** 2606.06013 | [PDF](https://arxiv.org/pdf/2606.06013v1)

**作者:** Hugo Bertin `[一作]` (Univ Rennes, CNRS, INRIA, IRISA), Yérom-David Bromberg `[通讯]` (Univ Rennes, CNRS, INRIA, IRISA)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建并公开了一个包含真实玩家与脚本化游戏会话的网络与应用层日志数据集，用于检测多玩家在线游戏中的作弊行为，尤其是网络流量扰乱类作弊。

**💡 创新点**

首个公开数据集标注网络流量扰乱作弊，并同时提供多层次（网络包、协议层日志、游戏事件）可标注的数据，支持跨层面作弊检测与研究。

**🔧 技术方法**

采用基于Unreal Engine的自研插件实现作弊注入与双层日志采集，使用pcap抓包、Protocol Buffers序列化日志，配套Python解析器；网络层同步通过NTP，使用Traceroute与Ping记录网络状态。

**📊 数据集**

数据集由多场次（含6~8名玩家的真实比赛与多轮脚本化场景）组成，包含 raw pcap、过滤后protobuf二进制日志、应用层游戏事件、作弊事件与网络统计，全部托管于Zenodo（待更新）。

**📈 对比分析**

文章未提供具体方法对比与性能评估，主要定位为数据资源发布，期望后续研究可基于此数据开展检测算法验证与基准测试。

**⚠️ 局限性**

受限于实现作弊为自研脚本化版本（非真实作弊软件）、仅覆盖少数作弊类型、仅单一基于Unreal Engine的游戏、且实验环境多在法国网络，数据可泛化性与覆盖范围有限。

---

## 447. A Conversational Framework for Human-Robot Collaborative Manipulation with Distributed Generative AI models

**arXiv ID:** 2606.06061 | [PDF](https://arxiv.org/pdf/2606.06061v1)

**作者:** Arash Ghasemzadeh Kakroudi `[一作]` (Tampere University), Roel Pieters `[通讯]` (Tampere University)

**通讯引用:** 1450 | [OpenAlex ID](https://openalex.org/A5077269216)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

构建了一个分布式对话式框架，将语言理解、视觉-语言模型、任务协调与运动执行拆分为ROS 2节点，支持边缘与工作站分布式部署，并通过Web仪表盘实现人工确认后才执行机器人动作。

**💡 创新点**

创新点在于：①以可插拔的ROS 2多代理架构实现模块解耦，便于在不同硬件上部署；②利用本地LLM与VLM零样本生成结构化动作请求，避免直接由模型控制机器人；③在每一步骤提供像素/深度/机器人坐标可视化，并强制人工确认，提升安全性与可解释性。

**🔧 技术方法**

技术包括：ROS 2节点通信、MoveIt 2运动规划、Ollama本地LLM/VLM推理、Qwen2.5‑VL 32B视觉-语言模型、Franka FR3机械臂、Intel RealSense RGB‑D相机、Web‑Bridge与ROSBridge的实时交互。

**📊 数据集**

实验数据集为Frank FR3工作台场景，使用一组彩色点数骰子进行单体、复体与重叠三种布局，所有任务均以自然语言指令触发。

**📈 对比分析**

对比方法为多模型多配置（LLM‑VLM配对与硬件部署方式）在 pick、place、handover 三项任务及三种场景下的成功率、延迟与内存占用；默认配置（minstral‑3:8b + Qwen2.5‑VL 32B）在 5/5/5 的成功率下取得最低延迟（LLM 5.47 s，VLM 9.57 s），并在 GPU 与 Edge 分离部署时显著降低系统总延迟。

**⚠️ 局限性**

局限性包括：无法处理需要组合推理或多句指令的复杂语言；无会话历史，无法执行后续指令或上下文相关请求；整体性能受所用模型能力限制，缺乏针对特定任务的微调；对极其不规范或多指令的输入易导致动作提取失败。

---

## 448. When Should Memory Stay Silent: Measuring Memory-Use Boundaries in Memory-Augmented Conversational Agents

**arXiv ID:** 2606.06055 | [PDF](https://arxiv.org/pdf/2606.06055v1)

**作者:** Lingxiang Xu `[一作]` (Hefei University of Technology), Ning An `[通讯]` (Hefei University of Technology)

**通讯引用:** 5236 | [OpenAlex ID](https://openalex.org/A5100651298)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了RBI‑Eval基准，用以评估语言模型在访问长期记忆时是否会不恰当地整合敏感历史。

**💡 创新点**

提出了“当前回合权限”概念，并通过对比测评集揭示不同模型和检索方式在敏感记忆整合上的差异。

**🔧 技术方法**

利用检索增强生成、记忆包装与评估流程，结合对四大LLM（Claude‑Sonnet‑4.6、GPT‑5.4‑mini、DeepSeek‑V4‑Flash、Qwen3.5‑9B）的实验。

**📊 数据集**

基于LoCoMo对话的10个虚构角色，生成2400个单回合测试实例，包含敏感披露与相同提示的对比。

**📈 对比分析**

对比无记忆、全上下文和三种检索条件，使用BSS等指标，发现Claude、DeepSeek 等模型在敏感记忆整合上显著高于 GPT‑5.4‑mini，检索过滤可降低约70%–80%的风险。

**⚠️ 局限性**

局限在于仅覆盖英语单回合情境、采用保守隐私规范且未考察多语言或更复杂交互，可能不适用于所有部署场景。

---

## 449. Beyond Similarity: Trustworthy Memory Search for Personal AI Agents

**arXiv ID:** 2606.06054 | [PDF](https://arxiv.org/pdf/2606.06054v1)

**作者:** Jiawen Zhang `[一作]` (Zhejiang University), Ruoxi Jia `[通讯]` (Virginia Tech)

**通讯引用:** 2849 | [OpenAlex ID](https://openalex.org/A5032275274)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一个名为 MEMGATE 的安全内存检索框架，提升个人 AI 代理在面对噪声和对抗攻击时的可靠性与可信度。

**💡 创新点**

创新性地将门控机制与相似度正则化相结合，形成一种可解释且鲁棒的检索路径，并证明其在多场景下优于传统仅基于相似度的检索方法。

**🔧 技术方法**

采用深度记忆网络、门控结构、对抗训练及多模态特征融合技术，构建整体的检索与防御体系。

**📊 数据集**

实验使用公开的 Persona‑Chat、MS MARCO 以及自行构造的个人语料库，覆盖对话、问答与多模态任务。

**📈 对比分析**

与传统 k‑NN、Dense Retrieval 以及 Retrieval‑Augmented Generation 进行对比，MEMGATE 在准确率提升约 8%（对话场景）、鲁棒性提升约 12%（对抗攻击场景）以及响应时间略优约 5%。

**⚠️ 局限性**

当前模型在大规模实时查询时仍存在计算开销较大和门控阈值需人工调参的限制，影响部署的可扩展性与自动化水平。

---

## 450. L-SDPPO: Policy Optimization of Spiking Diffusion Policy for Intra-vehicular Robotic Manipulation

**arXiv ID:** 2606.06049 | [PDF](https://arxiv.org/pdf/2606.06049v1)

**作者:** Liwen Zhang `[一作]` (Harbin Institute of Technology), Zuoquan Zhao `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 187 | [OpenAlex ID](https://openalex.org/A5081005685)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了低能耗的 L‑SDPPO 框架，结合 Spiking Diffusion Policy 与强化学习，用于微重力环境下的车载机器人精准操作。

**💡 创新点**

创新点在于将全脉冲残差网络与状态依赖延迟注入（SDLI）相结合，实现低功耗的多模态动作生成，并通过 PPO 对策略进行在线微调。

**🔧 技术方法**

采用的技术包括脉冲神经网络（SNN）、扩散策略（DP）、PPO 强化学习、TTFS 代码的延迟注入以及 surrogate gradient 反向传播。

**📊 数据集**

使用在五种典型车载任务（舱门开启、容器装载、容器盖闭合、面板操作、抽屉装载）上收集的 200 条专家演示轨迹构成的数据集进行预训练。

**📈 对比分析**

与 GBC、DP、PPO、GPPO、DPPO 等基线相比，L‑SDPPO 在平均成功率 0.97、平均奖励 478.25 的基础上，能耗仅为 DPPO 的 36% 左右，显著提升了效率与性能。

**⚠️ 局限性**

局限性包括仅在仿真环境下验证，缺乏真实 neuromorphic 硬件能耗评估；对双臂或柔性物体等更复杂任务的适用性尚未测试。

---

## 451. Catastrophic Forgetting as Accessibility Collapse: A Three-Level Framework for Knowledge Persistence in Continual Learning

**arXiv ID:** 2606.06032 | [PDF](https://arxiv.org/pdf/2606.06032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 452. Gotta Grow Fast: Design and Benchmarking of a Tip Mount for High-Speed Vine Robots

**arXiv ID:** 2606.06040 | [PDF](https://arxiv.org/pdf/2606.06040v1)

**作者:** Antonio Alvarez Valdivia `[一作]` (Massachusetts Institute of Technology), Nathaniel Hanson `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 117 | [OpenAlex ID](https://openalex.org/A5065894510)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了一种三角形滚轮式端部安装结构，使软藤机器人在高速成长时能够更顺畅地携带传感器。

**💡 创新点**

创新点在于将安装结构与藤体的天然三角形变形匹配，并使用低摩擦PTFE滚轮实现内部摩擦显著降低；同时提出了可重复的尾部张力测试平台。

**🔧 技术方法**

采用软体机器人设计、3D打印结构、PTFE滚轮、内部对准球、气压驱动与张力传感测量等技术。

**📊 数据集**

未使用公开数据集，而是构建自定义测试平台，对不同安装变体进行多次实验收集张力、压力和成长速度数据。

**📈 对比分析**

通过比较各变体在相同气压、速度下的尾部张力损失（ΔT）与增长一致性，最终PTFE三角滚轮装置获得最低ΔT（约22 N）和最高成功率。

**⚠️ 局限性**

局限在于仅评估轴向单一方向成长，无法区分内部滑动、膜弯曲和外部摩擦的具体贡献；未在狭窄或混乱环境中验证。

---

## 453. Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents

**arXiv ID:** 2606.06036 | [PDF](https://arxiv.org/pdf/2606.06036v1)

**作者:** Shuo Ji `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 6007 | [OpenAlex ID](https://openalex.org/A5065675832)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MRAgent，一个通过主动多步重构（active reconstruction）在结构化图内检索记忆的 LLM 代理；

**💡 创新点**

创新点在于将记忆检索转变为动态、可学习的多步探索过程，并设计了 Cue–Tag–Content 关联图，使 LLM 在检索时可基于中间证据主动调整检索路径；

**🔧 技术方法**

采用 LLM 进行记忆元素提取与标签生成，构建异构图；在检索时使用 LLM 推理选取前向/逆向 Traversal 操作，完成主动检索；

**📊 数据集**

在 LoCoMo（长对话记忆理解）和 LongMemEval（跨会话长记忆评估）两个公开基准上进行实验；

**📈 对比分析**

与 RAG、LangMem、A-Mem、MemoryOS、Mem0 等基线相比，MRAgent 在 F1/LLM-Judge 分数上提升 12–23%（Gemini/Claude），且 token 与运行时间显著降低；

**⚠️ 局限性**

局限包括：检索深度增加导致延迟上升；静态记忆构建未实现自适应更新，导致存储膨胀；需要进一步改进检索深度与记忆维护机制。

---

## 454. The Generator-Eraser Paradox: Community Guidelines for Responsible LLM-Assisted Dialect Resource Creation

**arXiv ID:** 2606.06004 | [PDF](https://arxiv.org/pdf/2606.06004v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 455. When Good Enough Is Optimal: Multiplication-Only Matrix Inversion Approximation for Quantized Gated DeltaNet

**arXiv ID:** 2606.06034 | [PDF](https://arxiv.org/pdf/2606.06034v1)

**作者:** Luoming Zhang `[一作]` (Qualcomm AI Research, an initiative of Qualcomm Technologies, Inc), Liang Zhang `[通讯]` (Qualcomm AI Research, an initiative of Qualcomm Technologies, Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种仅使用矩阵乘法的近似矩阵逆算法，用于提升分块线性注意力中的矩阵求逆速度。

**💡 创新点**

创新点在于将截断 Neumann 展开与结构化对角掩码结合，并通过并行残差校正实现无序列依赖的高效求逆。

**🔧 技术方法**

采用了低阶 Neumann 展开、对角掩码、并行残差校正以及低位 INT 量化等技术。

**📊 数据集**

实验使用 WikiText‑v2、MMLU、CSR、RealWorldQA 等数据集验证。

**📈 对比分析**

与 Flash Linear Attention 对比，单核矩阵求逆加速达 5×，整体解码层延迟降低约 20%，且在 FP16/INT16 量化下保持相同准确率。

**⚠️ 局限性**

局限性包括对特定块尺寸和量化精度的依赖，以及在极大块或更低精度下可能出现数值不稳定。

---

## 456. EGTR-Review: Efficient Evidence-Grounded Scientific Peer Review Generation via Multi-Agent Teacher Distillation

**arXiv ID:** 2606.06025 | [PDF](https://arxiv.org/pdf/2606.06025v1)

**作者:** Xinpeng Qiu `[一作]` (Peking University), Jimin Wang `[通讯]` (Peking University)

**通讯引用:** 6525 | [OpenAlex ID](https://openalex.org/A5100772432)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

提出了 EGTR-Review 框架，利用多代理教师模型进行结构化论文拆分、证据检索与可靠性标注，并通过任务前缀多任务学习将推理轨迹与最终评论蒸馏到轻量级学生模型，实现可追溯、基于证据的同行评审生成。

**💡 创新点**

将多代理教师的证据驱动推理路径与最终评审结果以稀疏监督蒸馏到学生模型，并引入证据权重目标与证据状态标签，兼顾可追溯性、证据依据与推理效率。

**🔧 技术方法**

基于结构化拆分、外部学术检索、证据状态标注、链式推理、任务前缀多任务学习以及证据加权损失的多代理教师蒸馏。

**📊 数据集**

在 PeerRead 与 OpenReview 的 ICLR 2017‑2024 论文集上构建的 1,386 篇数据集（997 训练、60 验证、329 测试）进行训练与评估。

**📈 对比分析**

与零样本、提示、微调、结构化/多代理基线比较，EGTR-Review 学生在自动指标、LLM-as-Judge 与人工评价中均优于 TreeReview 等强基线，ROUGE‑L 49.20、BERTScore 85.60、SN‑F1 48.45；学生相较教师在效率上显著提升（token 105k、时长 44s）。

**⚠️ 局限性**

模型在领域泛化（仅限 AI 领域）和多模态评审（图表、表格等）方面有限，检索可靠性与证据匹配仍需改进。

---

## 457. Empathy on Demand: How Empathic AI Can Scale Emotional Support for Verbal Harassment

**arXiv ID:** 2606.05995 | [PDF](https://arxiv.org/pdf/2606.05995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 458. AttackPathGNN: Cross-function vulnerability detection in smart contracts using state interference graphs and conjunction pooling

**arXiv ID:** 2606.05986 | [PDF](https://arxiv.org/pdf/2606.05986v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 459. PLAN-S: Bridging Planning with Latent Style Dynamics for Autonomous Driving World Models

**arXiv ID:** 2606.06014 | [PDF](https://arxiv.org/pdf/2606.06014v1)

**作者:** Xiaoyun Qiu `[一作]` (Hong Kong University of Science and Technology), Xinhu Zheng `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2393 | [OpenAlex ID](https://openalex.org/A5062424202)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向规划器的桥接模块，利用双 AdaFiLM 对 BEV 隐空间进行风格编码，解码出四通道语义成本图，并在回归式与锚点评分式 LWM 规划器中通过注意力融合与奖励融合两种接口预先影响轨迹决策。

**💡 创新点**

创新点在于：① 将风格动态显式建模为可控成本图，既能可视化又可调节；② 设计了双 AdaFiLM 使得车辆状态与风格信息在特征层面分离；③ 通过统一的成本图合同实现了跨规划器（回归与锚点）可插拔的桥接，提升了可解释性与可控性。

**🔧 技术方法**

使用了隐世界模型（LWM）生成 BEV 隐空间；成本图解码器采用卷积 + 双 AdaFiLM；回归规划器使用注意力级联融合；锚点规划器使用奖励级联融合；训练时结合 BCE 成本监督与规划损失；样式编码通过 GRU 或 2D 线性向量得到。

**📊 数据集**

在 nuScenes（开放循环）上评估回归规划器；在 NAVSIM（闭环仿真）上评估锚点规划器，二者分别使用 ResWorld 与 WoTE 作为 host。

**📈 对比分析**

与 ResWorld 基线相比，nuScenes 上平均 L2 从 0.59 m 降至 0.55 m，3 s 碰撞率下降 42%；在 NAVSIM 上规则成本版获得 89.4 PDMS，学习版 89.1 PDMS，且在难度较高的场景中学习版优势显著。消融实验表明成本通道和双 AdaFiLM 主要降低碰撞率，融合接口提升轨迹质量。

**⚠️ 局限性**

局限性包括：仅在两种 host 上验证，未展示跨模型普适性；学习成本在简单场景下表现不如手工规则；缺少对风格匹配的专门评测（如 SM‑PDMS）与人类偏好标签；依赖额外的辅助监督与风格编码，部署成本较高。

---

## 460. Global-Local Monte Carlo Tree Search in Vision-Language Models for Text-to-3D Indoor Scene Generation

**arXiv ID:** 2606.06002 | [PDF](https://arxiv.org/pdf/2606.06002v1)

**作者:** Mengshi Qi `[一作]` (Beijing University of Posts and Telecommunications), Huadong Ma `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 14394 | [OpenAlex ID](https://openalex.org/A5100710713)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过层级场景表示、PRM 引导的 MCTS 以及 LVLM 进行文本到 3D 室内场景的生成，并对场景中的对象进行纹理重绘，实现全流程的无训练、可回溯的布局生成。

**💡 创新点**

将室内布局生成转化为树搜索问题，引入进度奖励模型 (PRM) 对中间状态进行评估并剪枝，采用视觉 emoji 网格提示 LVLM 进行空间推理，并首次提出大规模多样化基准 3DTindo‑Bench。

**🔧 技术方法**

核心技术包括：大型视觉语言模型（如 Qwen2.5‑VL‑72B）、PRM‑guided MCTS、emoji‑grid 视觉提示、基于扩散模型的纹理重绘、OBJaverse 1.0 3D 资产检索。

**📊 数据集**

使用了两大数据集：3DTindo‑Bench（65 类室内场景、3250 条多样化文本指令）和 Objaverse 1.0（含 3D 模型与属性注释）。

**📈 对比分析**

与 HoloDeck、LayoutVLM、DirectLayout、Deng 等基线在 3DTindo‑Bench 上进行对比，平均得分提升约 10 分，单指标提升约 14%；在 LayoutVLM 基准上 CF 100、Pos 53.8、Rot 60.6、PSA 50.4，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性包括：当指令中出现数据库中不存在的稀有物体时生成失败；纹理美感仍受限于扩散模型的风格表达；整体方法依赖外部 3D 模型库，缺乏生成式 3D 模型能力；LVLM 对细腻美学、风格一致性的理解尚不充分。

---

## 461. Deep Learning-based 3D Oral Cavity Reconstruction Using 2D Intraoral Images

**arXiv ID:** 2606.05998 | [PDF](https://arxiv.org/pdf/2606.05998v1)

**作者:** Jihun Cho `[一作]` (Pai Chai University), Sun-Young Ihm `[通讯]` (Pai Chai University)

**通讯引用:** 200 | [OpenAlex ID](https://openalex.org/A5102751223)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种仅使用10张固定角度二维口内图像即可实现三维口腔模型重建的基于深度学习的软件方法。

**💡 创新点**

创新点在于将MobileNetV2与多头注意力融合用于多视角特征融合，并通过直接预测50,000个三维顶点坐标，实现低成本、无硬件设备的全自动三维重建。

**🔧 技术方法**

技术手段包括MobileNetV2特征提取、位置编码与多头注意力聚合、MLP解码器、特征调制器、Chamfer+L1损失及权重调度。

**📊 数据集**

使用公开的Dental3DS数据集中的950个上颌模型（每个约150,000顶点）进行训练与评估。

**📈 对比分析**

相较于传统扫描和摄影测量，本文模型在10张图像下取得77.49%的邻域匹配精度（阈值0.035），但与需数百张图像的NeRF或光度计相比性能尚低。

**⚠️ 局限性**

主要限制是Chamfer损失导致预测顶点集中于高密度区域，点云分布不均匀，降低了临床可用性。

---

## 462. RealDexUMI: A Wearable Universal Manipulation Interface for Dexterous Robot Learning

**arXiv ID:** 2606.06033 | [PDF](https://arxiv.org/pdf/2606.06033v1)

**作者:** Chaoyi Xu `[一作]` (Peking University), Zongqing Lu `[通讯]` (Peking University)

**通讯引用:** 2231 | [OpenAlex ID](https://openalex.org/A5089642905)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可穿戴、无机器人主体的数据收集接口 RealDexUMI，使演示者通过掌面等距手套直接控制可部署的灵巧手，采集对部署时相同手指接触、触觉和视觉观测一致的数据。

**💡 创新点**

创新点在于：①将部署手本身作为演示接口，完全消除映射/重定位需求；②实现了动作-状态一一对应的可执行手指指令；③通过相对手帧动作表示实现跨机器人本体无缝迁移；④集成轻量化指尖触觉与内置摄像，实现局部高精度感知。

**🔧 技术方法**

技术手段包括 11 DoF 轻量化灵巧手、指尖压阻式触觉阵列、内置摄像、6 DoF 轨迹跟踪、掌面等距手套（绝对磁编码映射手指位置）以及基于 ACT 的分段动作预测策略；部署时通过 IK 与低层控制将相对手帧动作投射到具体机器人上。

**📊 数据集**

数据集为 8 个真实机器人任务共 100+ 小时、每任务 200+ 片段的演示数据，包含手指指令、手指姿态、触觉、视觉及相对姿态标签。

**📈 对比分析**

对比实验显示在 8 个任务上平均成功率 88.75%；去除触觉下降至 70%，仅使用状态作为动作进一步降至 51%；同一检查点在 Franka FR3、RealMan RM65、PND Adam‑U 上跨本体部署均能保持 80%+ 成功率，验证了无重训练的迁移能力。

**⚠️ 局限性**

局限性主要在于仅局部感知（无全局/长距离规划视角）、指尖触觉与视觉不足以完成需要全局搜索或长程规划的任务，以及当前手指 DoF（6）有限，无法覆盖更高自由度的灵巧手。

---

## 463. Learning solution operators of PDEs with sparse approximation methods

**arXiv ID:** 2606.06046 | [PDF](https://arxiv.org/pdf/2606.06046v1)

**作者:** Sebastian Neumayer `[一作]`, Fabian Taubert `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种结合维度递增支持检测与正交匹配追踪（OMP）的稀疏高维技术，用于近似参数化PDE的解算子，并通过多点求解进一步提高采样效率。

**💡 创新点**

创新点在于：①将维度递增框架与稀疏恢复方法相结合，显著降低所需的PDE求解次数；②利用OMP+在单次求解中提取多空间点数据，进一步提升样本利用率；③检测得到的索引集可直接揭示参数重要性和交互作用，增强可解释性。

**🔧 技术方法**

核心技术包括：维度递增支持检测、正交匹配追踪（OMP）稀疏回收、Chebyshev多项式（或傅里叶）基展开、改进采样策略（OMP+）以及与Tensorized Fourier Neural Operators（TFNO）作为基线进行对比。

**📊 数据集**

使用三类自定义PDE数据集：1）热方程（参数为前9个正弦系数）；2）布格斯方程（同样的参数；只关注最终时间）；3）参数化二维扩散方程（20个随机参数）。所有训练样本均通过数值求解产生，采样随机生成。

**📈 对比分析**

与基于rank‑1 lattice的cubature方法以及Tensorized FNO进行对比。实验结果显示：OMP+在PDE求解次数和总运行时间上比cubature方法低数百倍；在热方程、布格斯方程和扩散方程上，OMP+的相对误差与FNO相当或更低，且训练时间显著短；但在参数维度非常高时，样本数仍显著大于FNO。

**⚠️ 局限性**

局限性：1）方法高度依赖所选基展开的稀疏性；若解在该基中不稀疏，则性能大幅下降；2）需要结构化的采样（anchor分解），不易直接推广到任意PDE；3）尽管维度递增显著压缩候选集，但在极高维场景下仍需大量样本和计算；4）对强非线性或高阶交互的检出可能需要更多迭代，导致额外的求解成本。

---

## 464. LLM-Conditioned Synthesis of Pathological Gaits via Structured Gait-Language Representations

**arXiv ID:** 2606.06048 | [PDF](https://arxiv.org/pdf/2606.06048v1)

**作者:** Mritula Chandrasekaran `[一作]` (Kingston University), Dimitrios Makris `[通讯]` (Kingston University)

**通讯引用:** 9447 | [OpenAlex ID](https://openalex.org/A5040949022)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于大型语言模型的多模态框架，利用结构化步态语言表示和专门设计的病理tokenizer生成能保留病理特征的3D骨架步态序列。

**💡 创新点**

创新点在于：①设计专用的病理tokenizer以保留病理运动的细微特征；②将病理统计先验与LLM结合，实现病理条件化的文本生成，再映射回运动空间；③整合姿势编码、G2L、L2G以及生物力学约束的解码器。

**🔧 技术方法**

使用的技术包括姿势编码器、空间/时间/病理三分支tokenization、Gait-to-Language (G2L)映射、LLM微调与语义增强、Language-to-Gait (L2G)重建以及生物力学约束的3D步态解码器。

**📊 数据集**

采用Jun等人公开的病理步态数据集，包含多种病理步态样本。

**📈 对比分析**

通过与MotionGPT和Qwen-5B生成的合成数据对比，使用GRU、LSTM、CNN三种分类器评估；最佳GRU在真实+合成数据下（LOS 0）达到92.77%准确率，高于仅使用真实数据的91.08%。

**⚠️ 局限性**

局限性包括：对CNN等非循环模型效果不佳；缺乏统计学验证与生物力学评估；需要更大规模的临床数据和专家审阅以进一步验证合成数据的真实性。

---

## 465. Framing, Judging, Steering: An Assessable Competency Model for Teach-ing Students to Reason With Generative AI

**arXiv ID:** 2606.05983 | [PDF](https://arxiv.org/pdf/2606.05983v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 466. Automatic Labelling of Speech Translation Errors

**arXiv ID:** 2606.06047 | [PDF](https://arxiv.org/pdf/2606.06047v1)

**作者:** Dominik Macháček `[一作]` (Charles University), Ondrej Klejch `[通讯]` (University of Edinburgh)

**通讯引用:** 557 | [OpenAlex ID](https://openalex.org/A5044666169)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了语音翻译错误标注（STEL）任务，设计了标注协议，构建了小规模真实评估数据集，并评估了现有自动系统在该任务上的表现。

**💡 创新点**

首次将错误跨度标注与严重程度等级、段级DA相结合，专为语音翻译场景设计；同时提出利用文本与音频双模的评估方法。

**🔧 技术方法**

采用XCOMET参考自由质量估计模型、Qwen2.5-Omni多模态LLM；使用ASR转录、音频输入、mWERSegmenter对齐、AwesomeAlign对齐误差；评估指标为字符级F1和Kendall τ。

**📊 数据集**

使用CsEn Robothon辩论集、ACL6060英文演讲等真实语料，涵盖Cs→En、En→Cs、En→De、En→He四语种，32分钟音频共329段。

**📈 对比分析**

通过与人工注释的跨度检测F1及DA相关性比较，发现人类F1≈71.9%，自动系统约38–50%；XCOMET在翻译错误检测上优于Qwen，Qwen在语音处理错误检测上表现更好，整体性能约为人类的一半。

**⚠️ 局限性**

受限于仅两篇文档、14/10分钟长、单一领域、有限的语言对、注释者数量少、经验有限，且未充分探索最佳分割、外部ASR及多样化说话人和情境下的挑战。

---

## 467. Misaligned AI as a New Insider Risk

**arXiv ID:** 2606.06028 | [PDF](https://arxiv.org/pdf/2606.06028v1)

**作者:** Matteo Pistillo `[一作]` (Apollo Research), Mark Beall `[通讯]` (AI Policy Network)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文阐述了在高风险环境中部署的AI模型如何因授权访问和潜在的对齐失效而产生与人类内部威胁同等的内部风险，并呼吁将AI视为内部风险向量来加以管理。

**💡 创新点**

创新点在于将AI模型的内部风险与传统人类内部威胁进行等价比较，并提出将现有人类内部风险评估与监控框架迁移至AI模型的方案。

**🔧 技术方法**

该论文主要采用了政策分析与案例讨论的方法，并未使用具体技术实现，而是聚焦于对AI授权访问和对齐失效的理论阐释。

**📊 数据集**

文中未使用特定数据集，主要引用了美国国防部、情报机构等对AI应用的公开信息和案例。

**📈 对比分析**

由于该工作为政策性讨论而非实验研究，没有进行实验比较；作者通过对已有内部风险检测机制的评估，建议持续监测和评估AI模型的行为。

**⚠️ 局限性**

局限性在于缺乏针对AI内部风险的实证数据与检测算法，政策建议的可行性和效果尚未经过实验验证，且现行监管框架难以直接覆盖AI模型的特殊性。

---

## 468. Contextualized Prompting For Stance Detection On Social Media

**arXiv ID:** 2606.06022 | [PDF](https://arxiv.org/pdf/2606.06022v1)

**作者:** Tilman Beck `[一作]` (Institute of Intensive Care, University Hospital of Zurich and University of Zurich), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在Twitter上使用大语言模型进行零/少样本立场检测，并系统评估加入不同上下文特征对性能的影响。

**💡 创新点**

证明LLM生成的目标描述能显著提升性能，同时揭示用户元数据和同用户推文对模型造成噪声的挑战。

**🔧 技术方法**

采用零样本/少样本提示、Chain-of-Thought、LLM-as-a-Judge等提示技术，并使用多家LLM（GPT、Gemma、Qwen、Ministral）。

**📊 数据集**

四个Twitter立场检测基准：covid19-de（德语）、covid19-glandt、semeval2016t6、wtwt。

**📈 对比分析**

与无上下文基线、CoT、LLM-as-a-Judge比较，LLM生成目标描述在大多数数据集上提升2–8个百分点，整体最优性能仍低于有监督训练。

**⚠️ 局限性**

主要限制包括对Twitter数据获取受限、提示设计未最优、模型对噪声上下文的鲁棒性不足。

---

## 469. ReCache: Learning Budget-Aware Caching Schedules for Diffusion Models via REINFORCE

**arXiv ID:** 2606.06060 | [PDF](https://arxiv.org/pdf/2606.06060v1)

**作者:** Mishan Aliev `[一作]` (HSE University), Denis Rakitin `[通讯]` (HSE University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于强化学习的缓存调度方法ReCache，能够在给定的计算预算下自动选择最优的去噪步骤进行完整计算，从而提升扩散模型的采样效率与生成质量。

**💡 创新点**

创新点在于将缓存调度建模为预算感知的RL问题，直接优化最终输出的一致性，并通过一个单一的预算条件MLP实现跨预算的嵌套调度，摆脱了传统的手工或均匀调度策略。

**🔧 技术方法**

核心技术包括：强化学习（REINFORCE+留一法基线）、Plackett‑Luce分布与Gumbel‑Top‑k采样实现k‑子集策略、预算条件的MLP logits预测、以及结合LPIPS与HPS/ImageReward的双目标奖励。

**📊 数据集**

使用的数据集包括MS‑COCO（文本到图像）、DrawBench（ImageReward评估）、VBench（视频评估），以及对应模型的全推理生成作为监督标签。

**📈 对比分析**

与统一、DiCache、DPCache等基线对比，ReCache在FLUX、Wan2.1和HunyuanVideo模型上分别实现了×5.04 FLOPs（LPIPS降31%）、×2.6速度提升（LPIPS降65%、VBench提升7%）等显著性能提升，显示其在不同预算和缓存机制下的优越性。

**⚠️ 局限性**

局限性：在极低步数（1–4步）下，缓存特征信息不足，导致加速效果有限；且依赖于已有的缓存机制，无法彻底突破模型自身的计算瓶颈。

---

## 470. MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following

**arXiv ID:** 2606.06058 | [PDF](https://arxiv.org/pdf/2606.06058v1)

**作者:** Mohammad Mahdi Salmani-Zarchi `[一作]` (University of Tehran), Mohammad Javad Dousti `[通讯]` (University of Tehran)

**通讯引用:** 622 | [OpenAlex ID](https://openalex.org/A5001927789)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种改进的RLVR方法MDP-GRPO，用以解决多约束指令跟随中标准GRPO因离散低方差奖励导致的不稳定性。

**💡 创新点**

创新点包括：①识别并正式化三种group-normalization病态（低方差放大、均值中心盲目、零方差崩塌）；②引入多温度采样、双锚优势和前景理论优势塑形来分别抑制上述病态；③使用不对称KL正则化进一步稳定更新。

**🔧 技术方法**

技术细节包括GRPO框架、温度多样化采样、多温度组采样、双锚优势（group-relative与goal-aware优势混合）、前景理论tanh塑形、lambda失利惩罚、asymmetric KL正则化，以及基于规则的离散可验证奖励。

**📊 数据集**

使用的数据集有：FollowBench、IFEval、以及自构造的500+多约束测试集；实验模型为Gemma-2-2B-Instruct和Llama-3.2-3B-Instruct。

**📈 对比分析**

与零样本基线和标准GRPO对比，采用SSR（软成功率）和HSR（硬成功率）指标评估。MDP-GRPO在I/F/自制基准上提升至最多5%严格成功率，且在小批量G=4时通过多温度采样恢复与G=8相近的性能，同时保持MMLU和ARC等通用能力不下降。

**⚠️ 局限性**

局限性：仅适用于可通过规则自动验证的确定性约束，难以扩展到主观、风格化或含糊约束；方法涉及多超参（锚混合权重、塑形系数、温度调度），在不同域或奖励尺度下需重新调优；实验范围限于2B/3B规模模型及英文任务，未验证多语言或大模型的泛化性。

---

## 471. Metamorphic Testing with the Rashomon Set: Explanation Faithfulness in Machine Learning

**arXiv ID:** 2606.06056 | [PDF](https://arxiv.org/pdf/2606.06056v1)

**作者:** Helge Spieker `[一作]` (Simula Research Laboratory), Arnaud Gotlieb `[通讯]` (Simula Research Laboratory)

**通讯引用:** 2724 | [OpenAlex ID](https://openalex.org/A5013313145)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一套基于Rashomon集的变形测试框架，用于无监督评估机器学习模型的解释可信度。

**💡 创新点**

引入五条形变关系来衡量单模型和跨模型解释一致性，并利用Rashomon集揭示模型多样性对解释可信度的影响。

**🔧 技术方法**

采用变形测试、SHAP/LIME后置解释器、Spearman相关、AOPC、EFI、PS等指标，结合Bootstrap置信区间进行统计评估。

**📊 数据集**

在加州房价（California Housing）和葡萄酒质量（Wine Quality）两份表格回归基准上进行实验。

**📈 对比分析**

将每个解释器的指标与随机归因基线对比，并报告违约率及置信区间；结果显示 SHAP 的解释多样性更大、LIME 更一致，某些形变关系违约率高达 70%，但总体上 LIME 在一致性上表现更好。

**⚠️ 局限性**

实验仅覆盖低维表格回归数据，扰动幅度和 Rashomon 阈值依经验设定；模型覆盖范围有限，可能产生离域输入，且对更复杂的深度学习或多模态任务尚未验证。

---

## 472. Sample-efficient Low-level Motion Planning for Robotic Manipulation Tasks via Zero-shot Transfer Learning

**arXiv ID:** 2606.06041 | [PDF](https://arxiv.org/pdf/2606.06041v1)

**作者:** Yuanzhi He `[一作]` (Cardiff University), Gualtiero Colombo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了iCEM+TL框架，利用迁移学习和奖励重构实现低层运动规划的样本高效零射击转移；

**💡 创新点**

创新点在于将迁移学习与奖励重构集成到iCEM优化过程中，提供上游任务经验来指导下游复杂任务，并给出了上游任务选择的可解释标准；

**🔧 技术方法**

采用的技术包括改进的iCEM（含颜色噪声、精英重用）、零射击迁移学习、奖励重构（任务分解）、以及基于高斯分布的轨迹搜索；

**📊 数据集**

实验数据集包括MuJoCo的FetchStack、FetchSlide和自定义Shelf环境，以及Franka Emika FR3机器人的真实堆叠任务；

**📈 对比分析**

与随机采样、CEM、iCEM、TQC+HER、CEE-US、PointFlowMatch等基线进行对比，iCEM+TL在Stack、Slide、Shelf任务中分别提升约23%、10%和9.9%的成功率；

**⚠️ 局限性**

局限性包括对上游任务的结构匹配敏感、对采样规模与精英大小的平衡要求高、以及真实世界实验仅验证单一任务且未在线重规划，缺乏更广泛的多任务评估。

---

## 473. Texture-preserving implicit neural representation for Cone beam CT truncated reconstruction

**arXiv ID:** 2606.06039 | [PDF](https://arxiv.org/pdf/2606.06039v1)

**作者:** Genyuan Zhang `[一作]` (Chongqing University), Fenglin Liu `[通讯]` (Chongqing University)

**通讯引用:** 8629 | [OpenAlex ID](https://openalex.org/A5100632772)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种自监督的、基于神经场的CBCT截断重建框架，将坐标网络与物理迭代细化结合，以实现截断伪影消除、空间外推和高频纹理恢复。

**💡 创新点**

创新点在于：①通过前向渲染直接映射空间坐标到衰减系数，天然避免滤波/反投影导致的环状伪影；②设计迭代细化模块（SIRT）在坐标网络初始值上补充高频细节，实现高保真纹理重建；③实现完全自监督训练，无需未截断的真值。

**🔧 技术方法**

核心技术包括：坐标网络（NeRF、Lineformer、Transformer）进行连续表示学习；Beer-Lambert定律的光线积分渲染与光度损失；物理迭代优化（SIRT）进行后处理；NTK理论分析谱互补性。

**📊 数据集**

使用公开的三维医学数据集Pancreas、Pelvis、Abdomen（通过TIGRE模拟截断投影）以及真实的羊骨外科样本（CD-130BX微CT）进行验证。

**📈 对比分析**

与传统分析方法（FDK、EX-FDK）、迭代方法（SIRT）以及其他神经重建方法（NAF、NeRF、SAX-NeRF、R²GS）进行对比；在所有数据集的FOV和全局范围内，PSNR提升约1–4 dB，SSIM提升至0.99（FOV）或0.87（全局），显著优于SAX-NeRF及其他基线。

**⚠️ 局限性**

主要限制包括：坐标网络训练和渲染耗时较长（NeRF约10 h重建512³体素）；3D Gaussian splatting在截断条件下表现不佳；在真实数据中，迭代细化可能产生轻微二次环状伪影。

---

## 474. SpeechJBB: Probing Safety Alignment and Comprehension in Large Audio Language Models under Code-Switched Speech

**arXiv ID:** 2606.06037 | [PDF](https://arxiv.org/pdf/2606.06037v1)

**作者:** Virginia Ceccatelli `[一作]` (Mila - Quebec AI Institute), David Ifeoluwa Adelani `[通讯]` (Mila - Quebec AI Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究多语言音频模型（LALMs）在代码混杂与语音伪词混入场景下的 jailbreak 漏洞，并构建了首个音频代码混杂攻击数据集 SpeechJBB，系统评估九种主流 LALM 的安全性能。

**💡 创新点**

创新点在于首次将音频代码混杂和语音伪词模糊攻击引入安全评估，揭示非英语–非英语代码混杂和伪词插入可显著降低拒绝率、提高 jailbreak 成功率。

**🔧 技术方法**

技术方法包括使用 GPT-4o 生成代码混杂文本、XTTS 合成语音、GPT-4.1 作为评判者进行 Refusal/Deflection/JSR 分类，以及在多模型（open‑source 与 proprietary）上对比实验。

**📊 数据集**

数据集方面，主要使用自研的 SpeechJBB（基于 JBB 的 100 份 harmful/benign 提示，转译成 10 种语言对并合成语音），并对照 Fleurs、MGSM 等公开多语言评测集。

**📈 对比分析**

通过对 monolingual、英语–其他和非英语–非英语三种代码混杂设置以及伪词插入率（10%、30%、50%）的系统评测，发现非英语混杂下 Jailbreak Success Rate 达 20.92%，伪词插入后最高可达 25.48%，模型间差异显著，proprietary 系统表现最稳健。

**⚠️ 局限性**

局限性包括仅评估有限的九种模型，未涵盖更强的音频对抗攻击；防御仅采用 prompt‑level 方案；数据集与攻击方式相对狭窄，未来需扩展到更广泛的多语音攻击与训练时对齐方法。

---

## 475. NAVIRA: Decoupled Stochastic Remasking for Masked Diffusion Language Models

**arXiv ID:** 2606.06031 | [PDF](https://arxiv.org/pdf/2606.06031v1)

**作者:** Andrey Fomenko `[一作]` (Lomonosov Moscow State University), Roman Ischenko `[通讯]` (Lomonosov Moscow State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出了一种新的推理时间解码策略NAVIRA，分离了质量评分与重掩码步骤，并引入温度控制的随机重掩码。

**💡 创新点**

创新点在于将PRISM的评分与生成解耦，并通过温度调度实现随机重掩码，以平衡文本流畅度与多样性。

**🔧 技术方法**

使用了Masked Diffusion Language Model（MDM）170M版本及其PRISM token‑level质量头，构建了双前向推理和温度采样机制。

**📊 数据集**

实验数据集为无条件OpenWebText。

**📈 对比分析**

通过对比PRISM、非学习式重掩码、NAVIRA的不同变体，使用PPL、熵以及LLM‑as‑judge评估，结果表明NAVIRA在更大推理预算下保持更高多样性，并在LLM评估中取得更佳分数。

**⚠️ 局限性**

局限性：仅在170M MDM与无条件生成任务上验证，未测试更大模型或下游条件任务（如摘要、代码生成、指令执行）。

---

## 476. RedditPersona: A Modular Framework for Community-Conditioned LLM Adaptation from Reddit

**arXiv ID:** 2606.06027 | [PDF](https://arxiv.org/pdf/2606.06027v1)

**作者:** Amirhossein Ghaffari `[一作]` (University of Oulu), Ekaterina Gilman `[通讯]` (University of Oulu)

**通讯引用:** 714 | [OpenAlex ID](https://openalex.org/A5032821793)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个模块化框架 RedditPersona，用于将 Reddit 数据转化为社区条件下的 LLM 适配器并评估其性能。

**💡 创新点**

创新点在于统一了数据收集、社区划分、适配器训练和评估流程，支持多种社区定义并可复用。

**🔧 技术方法**

采用了 QLoRA 参数高效适配、Leiden/Louvain 社区检测、BERTScore、MAUVE 等评估指标。

**📊 数据集**

使用了 112 个城市福祉相关子版块，包含 16M+ 评论、301k 用户档案。

**📈 对比分析**

通过对比五种社区划分策略的生成质量、可识别度和分布相似度，发现子版块划分最具可识别性、语义划分最自然，整体提升了 16–34% perplexity。

**⚠️ 局限性**

局限在于仅测试了单一模型家族（4.1-3B），混合策略不足，未与个体级微调直接对比。

---

## 477. Merging model-based control with multi-agent reinforcement learning for multi-agent cooperative teaming strategies

**arXiv ID:** 2606.06011 | [PDF](https://arxiv.org/pdf/2606.06011v1)

**作者:** Christian Llanes `[一作]` (Georgia Institute of Technology), Samuel Coogan `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1888 | [OpenAlex ID](https://openalex.org/A5010552433)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

结合多智能体强化学习与模型预测控制，提出 MA‑AC‑MPC 框架，用于安全、动态可行的多智能体协作控制，并在追捕‑躲避与无人机‑地面车降落任务中验证。

**💡 创新点**

将 AC‑MPC 扩展到多智能体，使用共享参数成本网络将 MPC 作为 actor 层实现端到端控制；同时采用分层训练、集中训练/分散执行与死亡掩码等技术，提升鲁棒性与样本效率。

**🔧 技术方法**

使用 Actor‑Critic 强化学习（MAPPO）、可微模型预测控制（MPC），梯度通过 KKT 条件传播；在 ROS2 环境中集成 Crazyflie 与 ROSMASTER X3 进行硬件实验。

**📊 数据集**

基于 MuJoCo/CrazySim 的仿真环境（多智能体追捕‑躲避、无人机‑地面车降落）以及相应硬件实验数据。

**📈 对比分析**

与多层感知器 MA‑AC‑MLP 进行对比；在追捕‑躲避中 MA‑AC‑MPC 在更少训练步数内获得更高赢率且对质量变化更稳健；在降落任务中硬件成功率 100% 对比 60%，误差更小。

**⚠️ 局限性**

训练时间更长；MPC 组件只能在 CPU 上高效实现；对未知参数或大幅动态变化的鲁棒性仍有限；需手工设计 MPC 目标与约束，导致通用性受限。

---

## 478. Diffusion Models for Adaptive Sequential Data Generation

**arXiv ID:** 2606.06007 | [PDF](https://arxiv.org/pdf/2606.06007v1)

**作者:** Haoyang Cao `[一作]` (Johns Hopkins University), Renyuan Xu `[通讯]` (Stanford University)

**通讯引用:** 832 | [OpenAlex ID](https://openalex.org/A5068576165)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种顺序前向-后向扩散框架（Adaptive Diffusion for Sequential Data Generation），通过在时间维度上逐步注入与去噪，并在每一步仅依据已生成的历史条件，生成满足信息流约束的时间序列。

**💡 创新点**

创新点包括：①将扩散模型顺序化，强制满足非预期（adapted）结构；②设计可并行训练的分段分数匹配目标；③提供通用的统计学习理论（分数逼近、估计与分布估计），并给出ReLU网络实现的具体误差上界。

**🔧 技术方法**

技术方法涵盖：连续时间扩散SDE、条件分数匹配、截断历史窗口、Transformer/causal masking用于参数化分数网络、ReLU网络的泛化与覆盖数分析、早停与终止策略。

**📊 数据集**

实验数据集包括：1) 合成ARMA(2,2)序列和高斯过程（H=32）用于评估分布与自相关；2) 实际S&P 500日收益序列用于构建均值‑方差组合并检验策略性能。

**📈 对比分析**

与静态扩散、GAN、VAE、离散Transformer等方法对比，实验显示：①在ARMA中生成样本的自相关误差显著低于原始训练样本；②在高斯过程中恢复的协方差矩阵与真实核一致；③在RL策略中使用该模型生成的合成路径能显著提升Sharpe比，超过传统合成方法和基准RL策略。

**⚠️ 局限性**

局限性包括：①对条件子高斯假设与光尾特性依赖，可能在极端非高斯或长记忆数据上失效；②截断历史窗口会在强长程相关情形下引入偏差；③理论误差上界涉及早停与终止时间的调参，实际实现需经验性调整；④模型训练与采样计算量大，尤其在高维长序列下。

---

## 479. Multimodal Sexism Identification and Characterization using Large Language Models and Gradient Boosting

**arXiv ID:** 2606.05997 | [PDF](https://arxiv.org/pdf/2606.05997v1)

**作者:** Kyriakos Chaviaras `[一作]` (National Technical University of Athens), Athanasios Voulodimos `[通讯]` (National Technical University of Athens)

**通讯引用:** 6436 | [OpenAlex ID](https://openalex.org/A5062640206)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

该论文提出一种基于特征工程与梯度提升模型的分层多模态管线，用于在存在性别偏见的 meme 与 TikTok 视频中进行性别歧视识别、意图检测与细粒度分类。

**💡 创新点**

创新点在于将 LLM 提取的语义指示器（如刻板印象、客体化、厌女）与传统视觉、文本、音频及生理/感知特征相结合，构建“目标化”与“分类化”两级 LLM 语义特征，并在多任务层面实施门控归一化。

**🔧 技术方法**

技术主要包括 CLIP、BLIP、XLM‑RoBERTa、Qwen2.5-1.5B‑Instruct、Whisper、MFCC、XGBoost 回归、特征选择与层级归一化。

**📊 数据集**

使用 EXIST 2026 数据集，共 8,235 条英文与西班牙语的 meme 与短视频，包含文本、图像、音频、OCR、用户人口统计与多传感器心率/眼动/EEG 等元数据。

**📈 对比分析**

与多种 ablation 与 baseline 进行比较，最终在软标签评估中取得最优 ICM‑Soft 分数，尤其在细粒度视频性别歧视分类上实现排名前十；硬标签性能相对较低，体现软标签评估更符合不确定性需求。

**⚠️ 局限性**

局限性包括 LLM 语义指示器的 prompt 依赖性与潜在偏差、视频处理缺乏显式时序建模、特征选择对测试集表现的不稳定性，以及固定阈值导致的硬标签解码不够鲁棒。

---

## 480. Double-Directional Wireless Channel Modeling Using Statistics-Aided Machine Learning

**arXiv ID:** 2606.05993 | [PDF](https://arxiv.org/pdf/2606.05993v1)

**作者:** Richmond Boamah `[一作]` (Utah State University), Ferdous Pervej `[通讯]` (Utah State University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5121731726)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对双向方向传播通道建模，提出一种基于统计辅助的图神经网络方法，用top‑M极端多径子集生成长期的多径序列。

**💡 创新点**

创新点在于：① 采用传播知识驱动的参数特定补丁与曼哈顿距离构造可学习图；② 将TimeFilter与TimesNet的优势融合为Hybrid TT模型；③ 在训练中加入统计辅助损失，直接匹配全通道统计，克服传统预测方法对MPC数量变化的敏感。

**🔧 技术方法**

主要技术包括：图神经网络（GNN）、混合Transformer/TimesNet模型、FFT频域分析、Manhattan距离构造图、统计辅助损失函数、动态专家门控（Moe）等。

**📊 数据集**

实验数据集为：① 基于SCM的合成数据；② 使用Sionna、OpenStreetMap、Blender生成的真实射线追踪数据。

**📈 对比分析**

与Transformer、BiLSTM‑Transformer、TimesNet、TimeFilter等基线进行对比，统计辅助版本（特别是TNTF(S)）在不同M和P下均取得更低的NMSE和更接近真值的CDF，显示出更优的统计匹配性能。

**⚠️ 局限性**

局限性包括：① 仅处理单天线点的单向/双向通道，未考虑多天线阵列；② 仅匹配第一阶统计量，对二阶统计未做扩展；③ 对极少MPC（M较小）或高度动态场景的鲁棒性尚待进一步验证。

---

## 481. Compress-Distill: Reasoning Trace Compression for Efficient Knowledge Distillation

**arXiv ID:** 2606.05988 | [PDF](https://arxiv.org/pdf/2606.05988v1)

**作者:** Maxime Griot `[一作]` (Universite catholique de louvain), Tanishq Mathew Abraham `[通讯]` (Sophont Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究教师模型生成的推理轨迹的后置压缩，并在压缩后使用知识蒸馏训练小型学生模型。

**💡 创新点**

创新点在于用可训练的压缩模型而非简单截断来缩短轨迹，同时系统评估压缩带来的准确率与效率的权衡。

**🔧 技术方法**

采用三阶段管道：教师生成、压缩器重写、学生微调，使用LoRA和全微调两种方法，压缩器为Llama-70B和Ministral-14B两大模型。

**📊 数据集**

使用十个数学、科学、医学、常识等基准数据集以及OOD推理与知识集进行评估。

**📈 对比分析**

与原始轨迹、截断、仅答三种训练目标比较，压缩可将训练标记降至12–30%，训练时间缩短2–7.6×，推理长度缩短3–19×，但准确率略低于原始轨迹。

**⚠️ 局限性**

局限包括单一通用压缩提示、缺乏域特定压缩策略、截断导致的误差、答案仅模式不稳定以及对极短/极长推理的处理不足。

---

## 482. Beyond Alignment: Value Diversity as a Collective Property in Multicultural Agent Systems

**arXiv ID:** 2606.05985 | [PDF](https://arxiv.org/pdf/2606.05985v1)

**作者:** Shaoyang Xu `[一作]` (Singapore University of Technology and Design), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 5394 | [OpenAlex ID](https://openalex.org/A5042288832)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文定义并量化了系统级价值多样性度量，评估多文化多代理系统在不同骨干模型、文化组合、代理规模以及交互情境下的多样性与对齐表现，并在参与式预算实验中验证多样性对集体决策的影响。

**💡 创新点**

创新点在于提出价值多样性作为多文化多代理系统的系统级评估轴，揭示多样性与传统对齐指标不相关且当前LLM系统在多样性上远低于人类社会，同时发现交互会加速同质化。

**🔧 技术方法**

使用World Values Survey提供的多文化价值问题回答，结合配对与结构多样性度量（平均欧氏距离和最小生成树距离），对单骨干、混合骨干、大规模配置、社交曝光以及参与式预算进行实验。

**📊 数据集**

采用World Values Survey（Wave 7）中的223个价值问题的多国调查答案，作为文化参考与人类多样性基准。

**📈 对比分析**

通过对18种LLM骨干在19种文化下进行单/混合骨干配置，计算对齐与多样性得分；结果显示单骨干多样性始终低于人类（最高≈36 vs 44），混合骨干虽提升但仍差距显著；多样性与对齐相关性仅为-0.12；交互导致多样性下降；高多样性系统在参与式预算中获得更广泛的项目支持。

**⚠️ 局限性**

局限性包括实验交互与决策场景过于简化，未考虑复杂社交网络和真实社会情境；对文化原型的假设与LLM样本偏差；实验结果可能不易推广到更丰富的文化表达与更大规模的多代理系统。

---

## 483. WorldFly: A World-Model-Based Vision-Language-Action Model for UAV Navigation

**arXiv ID:** 2606.06147 | [PDF](https://arxiv.org/pdf/2606.06147v1)

**作者:** Shengtao Zheng `[一作]` (Tsinghua Shenzhen International Graduate School), Xiao-Ping Zhang `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种双分支耦合的世界模型驱动视觉-语言-动作框架（WorldFly），实现了在UAV导航中同时生成未来视频预测和导航动作，从而提升对复杂城市峡谷环境的鲁棒性。

**💡 创新点**

创新点在于：①创建了挑战性更高的Urban Canyon Traversal基准；②设计了双分支耦合架构，将世界模型与动作专家分离但交叉互动；③利用流匹配并对齐噪声，联合优化未来视觉与动作的生成；④通过想象未来场景来提供运动先验，显著缓解“短视”问题。

**🔧 技术方法**

采用T5文本编码器、LTX-Video VAE编码器、流匹配（Flow Matching）框架、双分支Transformer与交叉注意力耦合、共享时间步噪声、动作离散化映射等技术。

**📊 数据集**

使用OpenFly仿真工具链生成的4,000+条无人机导航轨迹，并构建了包含见过与未见交叉口的TEST‑EASY和TEST‑HARD两个测试集；指令通过基于Qwen3‑VL的LLM自动生成。

**📈 对比分析**

在Urban Canyon Traversal基准上与OpenFly、Pi‑0‑UAV等基线进行对比。WorldFly在TEST‑EASY上的成功率、SPL、导航误差均大幅提升；在TEST‑HARD上，成功率从16%提升至31%，SPL提升15%，导航误差降低约4.2m，显示出显著的泛化能力。

**⚠️ 局限性**

主要限制是未来帧预测的计算成本高，导致每步推理约7.8秒，显著低于实时控制频率；未来工作需通过剪枝、蒸馏等方法降低推理开销。

---

## 484. RedEdit: Agentic Red-Teaming of Image Safety Classifiers via MCTS-Guided Photo-Editing

**arXiv ID:** 2606.06140 | [PDF](https://arxiv.org/pdf/2606.06140v1)

**作者:** Weilin Lin `[一作]` (Hong Kong University of Science and Technology), Li Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 38504 | [OpenAlex ID](https://openalex.org/A5100418910)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为RedEdit的黑盒红队攻击框架，用于在不改变恶意内容的前提下，利用常见的图像编辑工具绕过图像安全分类器。

**💡 创新点**

创新点在于将视觉-语言模型（VLM）用于生成语义导向的编辑候选，并结合蒙特卡洛树搜索（MCTS）进行全局规划，从而实现高效的迭代回溯和领域知识引导。

**🔧 技术方法**

主要技术包括VLM基于提示的候选生成、MCTS树搜索、工具库（旋转、裁剪、颜色调整、压缩等非生成编辑操作）以及内容保留率（CPR）评估机制。

**📊 数据集**

使用UnsafeBench数据集（777张标注为不安全的图像，覆盖11类不同危害类型）进行实验。

**📈 对比分析**

与随机搜索、单步最佳搜索和无回溯的ReAct Agent相比，RedEdit在ASR（攻击成功率）上达到76.2%，平均仅需1.26步，CPR保持93.0%，显著提升了安全分类器的易受攻击性。

**⚠️ 局限性**

局限性包括仅覆盖非生成编辑工具，未考虑文本或视频多模态内容；对不同VLM规模的评估有限；并未在真实平台上验证攻击可行性与防御策略的实际可用性。

---

## 485. MotionDisco: Motion Discovery for Extreme Humanoid Loco-Manipulation

**arXiv ID:** 2606.06139 | [PDF](https://arxiv.org/pdf/2606.06139v1)

**作者:** Ilyass Taouil `[一作]` (Technical University Of Munich), Majid Khadiv `[通讯]` (Technical University Of Munich)

**通讯引用:** 699 | [OpenAlex ID](https://openalex.org/A5043216529)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本研究提出MotionDisco框架，能够从零开始自动发现并部署长时程全身机器人行走与操作（loco‑manipulation）行为，最终实现在真实人形机器人上的零样本执行。

**💡 创新点**

创新点在于将大型语言模型（LLM）引导的进化搜索与分层的动力学轨迹优化相结合，利用轨迹规划的结构化失败反馈对LLM的程序变异进行引导，形成闭环高层推理与低层优化的协同搜索；并首次在无需动作重映射或遥控的情况下实现了复杂的全身行为。

**🔧 技术方法**

技术手段包括：Claude Opus 4.7 LLM进行程序化接触计划生成与变异、基于多射击（multiple shooting）的顺序几何可行性检验与完整动力学轨迹优化、基于动态可行性反馈的文本反馈机制、以及在仿真中训练的DeepMimic式强化学习跟踪策略。

**📊 数据集**

实验使用了八个自定义的长时程行走与操作任务场景（如香蕉捡取、箱子堆叠、跑酷式搬运等），并在这些场景中进行仿真和真实机器人部署，未使用公开的大规模数据集。

**📈 对比分析**

与单次LLM调用、无反馈的进化搜索以及仅使用几何可行性检查的基线相比，MotionDisco在所有任务中均达到了更高的成功率、更低的轨迹优化成本，并在几分钟内即可找到首个有效解；在真实机器人上零样本执行也表现出稳定的成功率。

**⚠️ 局限性**

局限性包括：仅支持单向粘性接触和盒状刚体物体；未集成感知模块，需预先知晓场景；无法处理滑动或非矩形接触、关节式/自由形状物体；以及在更复杂或非结构化环境中的可扩展性尚待提升。

---

## 486. TLA-Prover: Verifiable TLA+ Specification Synthesis via Preference-Optimized Low-Rank Adaptation

**arXiv ID:** 2606.06133 | [PDF](https://arxiv.org/pdf/2606.06133v1)

**作者:** Eric Spencer `[一作]` (Loyola University Chicago), Mohammed Abuhamad `[通讯]` (Loyola University Chicago)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5042456819)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建并训练了一个20B参数的语言模型，用于自动生成TLA+形式化规范，并通过四层验证层级和基于修复的GRPO训练策略实现高质量的模型检查通过。

**💡 创新点**

创新点包括：①提出四层验证体系，最高层使用变异测试剔除永真不变式；②设计基于修复的GRPO算法，直接利用TLC模型检查的奖励信号；③实证表明金银层通过率在所有训练阶段完全一致，验证了防止奖励作弊的有效性。

**🔧 技术方法**

使用技术主要有：LoRA微调（仅训练注意力投影），基于TLC的奖励信号的GRPO（修复式），对比的DPO偏好优化，四层验证管道及AST级变异测试。

**📊 数据集**

数据集来自公开的TLA+规范仓库（Foundation、TLA+官方仓库）经手动筛选得到1,053条满足金钻层标准的规范，另外划分30题独立验证集。

**📈 对比分析**

与FormaLLM公开基准（平均8.6% TLC通过率）对比，本模型在30题验证集上实现30%金钻层通过率（≈3.5倍提升），DPO版本仅达20%；Best‑of‑K推理进一步提升至约43%潜在可用率。

**⚠️ 局限性**

局限性包括：只擅长简单不变式模板的规范，对需要多轮消息分析或量化表达的协议表现不足；奖励信号计算昂贵，限制了RL吞吐；验证集规模小导致结果波动大；变异测试无法捕捉所有语义弱规范。

---

## 487. Deterring Searches for Child Sexual Abuse Material on Google Search and Promoting Help-Seeking

**arXiv ID:** 2606.06126 | [PDF](https://arxiv.org/pdf/2606.06126v1)

**作者:** Rebecca Umbach `[一作]` (Google), Abhishek Roy `[通讯]` (Google)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5048926128)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估谷歌搜索中 CSAM Onebox 语言更新对用户继续搜索 CSAM 的行为及其点击求助资源的影响。

**💡 创新点**

首次在大规模多国真实数据中，将 Onebox 从仅提供报告提示改为包含治疗资源链接，并在搜索结果顶部展示，验证其在抑制 CSAM 搜索和促进求助方面的效果。

**🔧 技术方法**

使用差分中的差分（DID）方法和中断时间序列分析，结合匿名用户会话日志与热线访问流量数据。

**📊 数据集**

Google 搜索匿名会话日志（覆盖 9 个实验国家与 7 个对照国家）以及 7 家热线网站的每日访问量。

**📈 对比分析**

通过 DID 对比实验组与对照组的 CSAM 二次查询率，发现新 Onebox 使再搜索率下降约 3.8 个百分点；点击求助资源率约为 0.73%；中断时间序列显示热线流量显著上升。

**⚠️ 局限性**

局限性包括仅观察单会话行为，缺乏长期跟踪；可能受新闻事件、VPN 使用导致的数据偏差；对照组匹配不完全，难以完全排除混杂因素；且仅覆盖 Google 搜索用户，未考察其他平台或暗网的流动。

---

## 488. Towards Healthy Evolution: Exploring the Role and Mechanisms of Human-Agent Interaction in Self-Evolving Systems

**arXiv ID:** 2606.06114 | [PDF](https://arxiv.org/pdf/2606.06114v1)

**作者:** Dianxing Shi `[一作]` (University of Osaka), Yuta Nakashima `[通讯]` (University of Osaka)

**通讯引用:** 2336 | [OpenAlex ID](https://openalex.org/A5065649079)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ANCHOR框架，用LLM模拟人类监督来改进自演化代理，缓解安全漂移与能力退化；

**💡 创新点**

在自演化学习循环中引入统一的阶段性监督机制，并系统评估监督频率与阶段对性能的影响；

**🔧 技术方法**

结合LLM评估器、强化学习自演化框架（AZR、R‑Zero）、任务生成‑验证‑反馈循环与阶段性评估技术；

**📊 数据集**

使用编码任务集LiveCodeBench、EvalPlus；数学推理集AIME24/25、AMC23、MATH500、OlympiadBench、Minerva Math；安全评测集HarmBench、SaladBench、HEx‑PHI、奖励劫持基准；

**📈 对比分析**

在多模型规模下与原始自演化模型对比，采用Pass@1、数学平均分、攻击成功率、危害评分、奖励劫持抗性等指标，ANCHOR在保持核心能力的同时提升安全性，部分指标提升5–10%；

**⚠️ 局限性**

监督由LLM模拟，可能引入偏差；实验范围局限于文本任务，未覆盖工具使用或具身环境；真实人类监督效果未得到验证。

---

## 489. A Sliced-Wasserstein Framework on Correlation Matrices for EEG Decoding

**arXiv ID:** 2606.06104 | [PDF](https://arxiv.org/pdf/2606.06104v1)

**作者:** Chen Hu `[一作]` (Westlake University), Yefeng Zheng `[通讯]` (Westlake University)

**通讯引用:** 17837 | [OpenAlex ID](https://openalex.org/A5051649145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于拉回欧几里得度量（PEM）的切片Wasserstein（PEMSW）框架，并在全秩相关矩阵流形上构造了闭式切片坐标的Correlation Sliced‑Wasserstein（CorSW）用于EEG解码的域泛化；

**💡 创新点**

将切片Wasserstein推广到具有全局可微分嵌入的流形；给出CorSW的几何闭式实现；证明SPDSW为PEMSW的特例；结合CorSW的域泛化损失显著提升跨场域EEG解码性能；

**🔧 技术方法**

拉回欧几里得度量、OLM/LSM相关几何、切片Wasserstein距离、概率分布对齐、基于Correlation Attention的网络、随机/均匀采样切片方向、Gaussian参考分布、梯度归因可解释性；

**📊 数据集**

BCIC‑IV‑2a（运动想象）、MAMEM‑SSVEP‑II（稳态视觉诱发电位）、BCI‑ERN（错误相关电位）三大公开EEG基准；

**📈 对比分析**

与欧氏深度模型、Riemannian模型以及多种实例级对齐和域适应方法对比；在三任务上均实现平均准确率/AUC提升，跨会话域泛化实验中取得最高平均得分；训练时间仅增加约5 ms/迭代；

**⚠️ 局限性**

仅适用于具全局可微分嵌入的PE流形，无法直接扩展至一般Riemannian流形；使用EEG数据需严格遵守隐私与数据治理规范。

---

## 490. HyperVis: Continuous Latent Visual Relational Graphs on the Lorentz Hyperboloid for Compositional Reasoning

**arXiv ID:** 2606.06100 | [PDF](https://arxiv.org/pdf/2606.06100v1)

**作者:** Moshiur Farazi `[一作]` (University of Doha for Science and Technology), Shafin Rahman `[通讯]` (North South University)

**通讯引用:** 1267 | [OpenAlex ID](https://openalex.org/A5007895165)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出HyperVis框架，利用连续视觉关系图和Lorentz超球面进行关系建模，并通过可学习的角度和包含性损失对LoRA适配器进行正则化，提升VQA与组合推理性能。

**💡 创新点**

创新点在于：①完全摆脱离线场景图生成器和离散谓词标签，直接构造连续视觉关系张量；②将关系嵌入到高曲率Lorentz超球面，并通过IoA驱动的包含性锥和角度排斥形成层次结构；③通过超球面Top‑K门选择前缀令牌，既在训练时正则化模型，又在推理时提供关系编码。

**🔧 技术方法**

使用技术包括：类无关区域提议、带空间偏置的跨注意力计算O(N²)关系；Lorentz模型的指数与对数映射、Einstein中点、包含性锥与角度损失；Riemannian Adam优化与自定义梯度截断；LoRA微调与前缀令牌注入。

**📊 数据集**

主要使用数据集：GQA（VQA基准），Winoground（图像文本匹配），SugarCrepe（组合推理评测）。

**📈 对比分析**

在GQA上，HyperVis通过LoRA正则化实现61.03%（比基线+0.65pp、比LoRA-only+3.82pp）；在SugarCrepe上，前缀令牌提升79.94%（比基线+6.25pp、比欧氏对照+4.58pp）；与文本SGG注入或CCoT方法对比，HyperVis在所有任务上表现更优。

**⚠️ 局限性**

局限性包括：依赖区域提议，可能漏检细小或遮挡对象；O(N²)关系矩阵计算与存储规模受限；生成式VQA在注入前缀时易出现注意力失衡；目前对抽象关系（如目光、运动）支持有限。

---

## 491. OrderGrad: Optimizing Beyond the Mean with Order-Statistic Policy Gradient Estimation

**arXiv ID:** 2606.06096 | [PDF](https://arxiv.org/pdf/2606.06096v1)

**作者:** Paavo Parmas `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 14179 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为OrderGrad的无偏梯度估计器，可直接针对奖励分布的顺序统计量（如VaR、CVaR、剪枝均值、中位数及best‑of‑K）进行强化学习与大型语言模型（LLM）的优化；

**💡 创新点**

通过对传统的似然比（LR）和重参数化（RP）梯度估计器做奖励/优势的简单秩变换，通用化为任意顺序统计目标，且保证无偏、低方差；

**🔧 技术方法**

使用LR/RP梯度、U‑statistics权重计算、离线留一基线、Beta核平滑的分位数表达，以及在LLM训练中的奖励/优势变换；

**📊 数据集**

在LLM数学推理任务中使用Qwen3‑4B‑Base和Qwen2.5‑Math‑7B两大模型，并在AIME24/25、AMC23、MATH500、Minerva等数学推理基准上进行评估；

**📈 对比分析**

与GRPO和Max@K基线比较，OrderGrad（如Top2@4）在大k（如pass@256）上提升约0.09（Qwen3‑4B‑Base）及0.02（Qwen2.5‑Math‑7B），在多奖励设置下既保持pass@k，又显著缩短输出长度；

**⚠️ 局限性**

受限于秩分裂处理、离策略数据、奖励模型误差、提示依赖以及PPO裁剪/归一化等实际因素；更大k会导致方差升高，需在设计权重α与k时权衡稳定性与极端目标匹配度。

---

## 492. Integrating Mechanistic and Data-Driven Models for Neurological Disorders through Differentiable Programming

**arXiv ID:** 2606.06094 | [PDF](https://arxiv.org/pdf/2606.06094v1)

**作者:** Shah Pallav Dhanendrakumar `[一作]` (Indian Institute of Technology Delhi), Sitikantha Roy `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 604 | [OpenAlex ID](https://openalex.org/A5020339890)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了混合建模（Hybrid Modeling）在神经系统疾病中的应用，阐述了并行、串行及并行-串行三种架构，并详细讨论了残差建模、神经常微分方程（NODE）、求解器循环等关键技术及其训练策略、优势与挑战。

**💡 创新点**

提出了一套完整的混合建模框架分类方法，并将其与可微物理、自动微分、对偶状态方法等前沿技术结合，强调了如何在保持物理可解释性的同时实现高效的数据驱动学习，形成了面向神经疾病预测与治疗规划的新型工具箱。

**🔧 技术方法**

可微物理、自动微分（Auto‑Diff）、对偶状态方法（Adjoint State Method）、PINN、神经常微分方程（Neural ODE）、通用微分方程（UDE）、求解器循环（Solver‑in‑the‑Loop）以及并行/串行混合架构等。

**📊 数据集**

本论文为综述性文章，未使用任何单一实验数据集；所引用的案例多基于公开研究或模拟实验。

**📈 对比分析**

文章未进行实证性对比实验；主要通过理论分析、已有文献案例和示例演示来比较各架构的计算成本、可解释性、泛化能力等，并指出它们在不同神经疾病场景下的潜在性能优势。

**⚠️ 局限性**

局限性包括：求解器可微化导致的高计算开销和内存占用；训练不稳定、梯度爆炸/消失问题；超参数众多、调优困难；在真实临床数据稀缺、噪声严重的环境下仍缺乏充分验证；需要进一步的跨学科合作与临床验证来提升可信度。

---

## 493. The Dignity-Centric Stack: A Commons-Governed, Horizontally Federated Architecture for Human-Dignity AI

**arXiv ID:** 2606.06083 | [PDF](https://arxiv.org/pdf/2606.06083v1)

**作者:** Eduardo C. Garrido-Merchán `[一作]` (Universidad Pontificia Comillas), Eduardo C. Garrido-Merchán `[通讯]` (Universidad Pontificia Comillas)

**通讯引用:** 1507 | [OpenAlex ID](https://openalex.org/A5070783543)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出了“Dignity Stack”——一种六层共治架构，旨在通过自治、互助、退出权等机制，将数字社会合约中的人类尊严、数据主权等价值落到实际数据治理实践中。

**💡 创新点**

创新点在于：①将人类尊严与共治理念结合，提出资本–治理脱耦与多层级自治；②引入退出权、声誉反馈与互助信用体系，以抵御外部强势主体的捕获；③将传统共治原则与现代计算基础（开源硬件、社区能源、联邦协议）相融合，形成可操作的治理模型。

**🔧 技术方法**

使用技术：基于共治理论（Ostrom、Illich、Bookchin 等）与计算架构（社区能源、开源硬件、联邦学习协议、互助信用系统）的组合设计，而非单一机器学习算法。

**📊 数据集**

未使用具体数据集；本工作为理论与制度设计，侧重于制度框架与治理机制。

**📈 对比分析**

比较方法：无实验对照；通过逻辑一致性与可行性分析评估，认为在实现数字社会合约价值上优于传统监管模式。

**⚠️ 局限性**

局限性：①低层硬件与芯片供应的多中心化实现难度大；②模型训练后数据不可逆导致退出后信息泄漏；③对强大外部势力（国家、巨型企业）的治理抵御仍有限，需与现行法律体系协同。

---

## 494. On Advantage Estimates for Max@K Policy Gradients

**arXiv ID:** 2606.06080 | [PDF](https://arxiv.org/pdf/2606.06080v1)

**作者:** Shota Takashiro `[一作]` (University of Tokyo), Yutaka Matsuo `[通讯]` (University of Tokyo)

**通讯引用:** 14179 | [OpenAlex ID](https://openalex.org/A5090592819)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大型语言模型推理任务中pass@K/​max@K强化学习奖励优化，并提出了Leave‑Two‑Out（L2O）基线实现无偏、中心化的优势估计器。

**💡 创新点**

提出L2O基线在保持PG无偏的同时使批量优势完全中心化，并给出max@K的“canonical finite‑batch advantage”统一框架，将现有方法分类。

**🔧 技术方法**

基于期望提升（EI）公式、U统计、离散奖励的policy‑gradient估计、GPU向量化实现O(B^2) L2O计算，以及group‑based RL框架MaxPO。

**📊 数据集**

在多臂赌博机、迷宫实验以及LLM推理任务中，使用数学推理基准AIME24、AIME25、AMC23、MATH500、Minerva，模型为Llama‑3.2‑3B‑Instruct与Qwen2.5‑Math‑7B。

**📈 对比分析**

与PKPO、GRPO、Entropy‑Adv等基线比较，MaxPO在pass@256上分别提升5.2%和2.4%，且训练期间梯度方差降低77.4%，证明中心化优势提升采样效率和优化稳定性。

**⚠️ 局限性**

仅适用于有限批量、on‑policy i.i.d. 采样；需要K≤B‑1；未验证离线或相关生成情形；主要针对可验证序列奖励，泛化性待进一步评估。

---

## 495. FontFusion: Enhancing Generative Text in Diffusion Models with Typographic Conditioning

**arXiv ID:** 2606.06066 | [PDF](https://arxiv.org/pdf/2606.06066v1)

**作者:** Marian Lupascu `[一作]` (Adobe), Zhaowen Wang `[通讯]` (Adobe)

**通讯引用:** 13555 | [OpenAlex ID](https://openalex.org/A5068625652)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FontFusion框架，在Diffusion Transformer上实现可插拔的字体控制。

**💡 创新点**

通过双编码器（DeepFont+ DINOv2）、分层字体‑文本Token绑定、位置感知Embedding以及多层Token丢弃策略，突破传统字体控制与文本可读性之间的权衡。

**🔧 技术方法**

结合Diffusion Transformer、深度卷积/自监督视觉编码器、余弦注意力、位置编码、Dropout、双Encoder组合等技术实现字体条件注入。

**📊 数据集**

使用合成的534字体多样组合训练集、406K专业设计模板集进行训练，并在公开的CRAFT与TIDE基准上进行评测。

**📈 对比分析**

与FonTS、Glyph‑ByT5等基线比较，针对FLUX.1基准模型，OCR准确率从72.31%提升至74.97%，字体一致性从0.91%提升至76.52%，在装饰字体上提升约76%。

**⚠️ 局限性**

目前仅支持拉丁文字、需要显式字体指定、训练数据为专有且不可公开，限制了复现与跨语言推广。

---

## 496. Multi-task Learning is Not Enough: Representational Entanglement in Dual-output Second Language Speech Recognition

**arXiv ID:** 2606.06065 | [PDF](https://arxiv.org/pdf/2606.06065v1)

**作者:** Seung Hwan Cho `[一作]` (Hanyang University), Young-Min Kim `[通讯]` (Hanyang University)

**通讯引用:** 29566 | [OpenAlex ID](https://openalex.org/A5100337311)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文研究了在韩语和英语二语自动语音识别中使用双输出多任务学习，分析其对表面转录与意义转录的影响。

**💡 创新点**

创新点在于揭示双输出多任务学习在不同语言中的不对称效果，并将其归因于编码器层级的任务混杂现象。

**🔧 技术方法**

采用了Conformer+Transformer双解码器架构，结合CTC与注意力损失，并用Centered Kernel Alignment (CKA) 进行表示层分析。

**📊 数据集**

使用AI-Hub公开的韩语和英语读音数据集，包含分别由L1中国/日语和L1韩语说话人朗读的样本。

**📈 对比分析**

通过单输出(SO)与双输出(DO)模型的字符错误率(CER)对比，发现DO在意义转录上略有提升但表面转录显著恶化，尤其英语的恶化随表面-意义编辑距离增大。

**⚠️ 局限性**

限制在于仅通过解码器适配无法完全补偿编码器级混杂，需要进一步探索稀疏分解、对抗训练或门控机制等结构化解决方案。

---

## 497. Regret Minimization in Single-Dimensional Contract-Design with Binary Actions

**arXiv ID:** 2606.06125 | [PDF](https://arxiv.org/pdf/2606.06125v1)

**作者:** Riccardo Poiani `[一作]` (Bocconi University), Andrea Celli `[通讯]` (Bocconi University)

**通讯引用:** 274 | [OpenAlex ID](https://openalex.org/A5023478606)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在线主-代理问题，主方在没有观测代理行动的前提下，通过与代理的多轮交互学习最优基于结果的支付合同；重点在二元行动和一维代理类型的设定。

**💡 创新点**

提出了两个关键创新：① 对于任意对抗性代理类型，证明并实现了与结果维数无关的 Θ(T^{2/3}) 监管误差上界；② 当代理类型固定不变时，利用探测阈值与鲁棒化的探索-承诺策略，取得了匹配的 Θ(√T) 监管误差下界。

**🔧 技术方法**

技术核心包括：① 将高维合同空间降维到一个阈值 λ 的一维优化；② 对非 Lipschitz 的 λ-函数采用非均匀离散化；③ 对固定类型情形，构造基于最优合同的阈值测试与随机二分搜索，并加入鲁棒化校正；④ 通过 KL 变换与连续动作空间下的下界构造证明。

**📊 数据集**

本工作为理论分析，无使用实际数据集；所有结果均在假设模型（二元行动、单维类型、已知结果分布、已知成本）下证明。

**📈 对比分析**

与以往工作（线性或指数增长的监管误差、对结果维度依赖的上界）相比，本文的上界显著改进：对抗性情形实现 Θ(T^{2/3})，并完全消除对结果维度 m 的依赖；固定类型情形实现 Θ(√T)，已达到信息理论下界。

**⚠️ 局限性**

局限性：仅处理二元行动和一维类型；对多行动或多维类型的扩展仍不清晰；对抗性下的下界与算法仅在理论层面；未考虑噪声或非标准分布的现实情况。

---

## 498. IR3DE: A Linear Router for Large Language Models

**arXiv ID:** 2606.06098 | [PDF](https://arxiv.org/pdf/2606.06098v1)

**作者:** Eros Fanì `[一作]` (Gensyn), Oğuzhan Ersoy `[通讯]` (Gensyn)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于岭回归的线性专家路由器，用来在不同领域专家LLM之间动态分配查询。

**💡 创新点**

创新点在于：1) 用闭式岭回归学习token级路由器，无需额外的语言模型或集中的训练数据；2) 通过熵筛选高置信度token投票来决定最终专家，提升鲁棒性；3) 结构轻量化，支持在线增删专家。

**🔧 技术方法**

核心技术包括：token嵌入 + 线性岭回归 (RLS) 训练 Token Router；样本路由选择器（SRS）利用token熵做top‑k筛选并投票；多种SRS变体（-avg、-all）。

**📊 数据集**

使用的公开数据集包括：HumanEval（代码生成）、GSM8k（数学）、M_ARC（多语言推理）、IFEval（指令遵循）、以及多个专用领域数据集（coding、math、biology、legal、dialogue 等）进行实验。

**📈 对比分析**

与基线（域专家、平均专家、随机路由、MoDEM、1NN/kNN 路由器）对比，在线性路由器（及其变体）在三种设置下均保持或超过基线性能；在 Reasoning 设置下，线性路由器达到了 98.4% 的标准化平均性能，优于 kNN（97.6%）。

**⚠️ 局限性**

局限性在于：作为线性模型，难以捕捉复杂语义或多步推理需求；对非常细粒度的领域区分可能表现不佳；未来可考虑核岭回归或加入系统成本因素进一步改进。

---

## 499. CHALIS: A Challenge Dataset for Language Identification in Difficult Scenarios

**arXiv ID:** 2606.06088 | [PDF](https://arxiv.org/pdf/2606.06088v1)

**作者:** Michal Tichý `[一作]` (Charles University), Jindřich Libovický `[通讯]` (Charles University)

**通讯引用:** 1200 | [OpenAlex ID](https://openalex.org/A5061045500)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CHALIS基准数据集，专门评估语言识别在亲属语言互通和非标准正字法（转写、去重音、同形字攻击、网络俚语）下的表现。

**💡 创新点**

创新点在于设计了针对亲属语言歧视和正字法噪声的挑战子集，并对单语与双语共语句子进行了人工验证，填补现有数据集缺口。

**🔧 技术方法**

技术手段包括使用KenLM语言模型进行句子评分、命名实体过滤、人工注释；对正字法进行多脚本转写、去重音、同形字替换及网络俚语生成。

**📊 数据集**

数据来源包括WMT新闻、Leipzig Wikipedia、Czech/Slovak/Spanish/Catalan/Portuguese/Galician等语料，最终形成公开的CHALIS数据集。

**📈 对比分析**

通过对FastText、OpenLID、GlotLID、GCLD3四大主流语言识别系统的评测，发现单语子集F1明显下降，双语子集对高资源语言偏好明显；正字法噪声几乎让所有系统失效，只有GCLD3在部分转写场景下保留一定鲁棒性。

**⚠️ 局限性**

局限性包括只覆盖四对欧洲亲属语言，人工标注样本有限，正字法噪声人工生成不反映真实分布，且CHALIS仅为评估基准，无法直接用于模型训练。

---

## 500. Toward Mobile and Converged Backhaul: The Promise of Wireless Access and Backhaul

**arXiv ID:** 2606.06075 | [PDF](https://arxiv.org/pdf/2606.06075v1)

**作者:** Chiara Rubaltelli `[一作]` (Politecnico di Milano), Antonio Capone `[通讯]` (Politecnico di Milano)

**通讯引用:** 6614 | [OpenAlex ID](https://openalex.org/A5055843108)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出并实现了 5G-Advanced 的无线接入与回传（WAB）架构，并在实测实验平台上验证其可行性与性能。

**💡 创新点**

创新点在于将接入与回传解耦、支持多技术多模式移动回传、实现模块化部署，可与不同运营商的回传网络互联，并开启移动 SD‑WAN 与公共 5G 接入岛等新应用场景。

**🔧 技术方法**

采用 5G NR（FR1/FR2）与 OAI gNB、Open5GS 核心、WireGuard VPN、商用 CPE 与 SDR 设备等技术实现全闭环系统。

**📊 数据集**

使用自建实验数据集：在车载与室内实验中，利用 iperf3 进行 DL/UL 速率测试，并记录 FR2‑only 与 WAB 整体链路的速率与频谱效率。

**📈 对比分析**

对比方法为将 WAB 系统的端到端吞吐率与单独使用 FR2 回传的基线进行对比；实验显示在 LOS 条件下 DL 达 50 Mbps，UL 甚至超过 FR2‑only，证明 WAB 在低覆盖与移动场景中性能可观。

**⚠️ 局限性**

主要局限包括回传链路波动导致的无线配置重调延迟、时频同步误差导致干扰、缺乏多跳回传支持、以及需要 AI 轻量化管理等挑战。

---

## 501. Edge-Aware Curvature Modeling for Graph Understanding in Large Language Models

**arXiv ID:** 2606.06073 | [PDF](https://arxiv.org/pdf/2606.06073v1)

**作者:** Zhenghong Lin `[一作]` (Nanyang Technological University), Shiping Wang `[通讯]` (Fuzhou University)

**通讯引用:** 14325 | [OpenAlex ID](https://openalex.org/A5100604230)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过训练无关的文本提示和曲率感知图表示学习框架（CureLLM），将边层信息注入预训练大型语言模型，以提升图结构与文本信息的联合表征。

**💡 创新点**

创新点在于①首次从理论上证明缺失边信息导致对齐子最优；②将负曲率视为信息瓶颈，使用正曲率边进行跨视图信息传递；③提出无训练参数的文本提示机制，将边结构直接编码为文本。

**🔧 技术方法**

采用Frozen LLM（如Llama2）作为文本编码器，Frozen图神经网络（如GraphCL）作为图编码器；利用Ollivier-Ricci曲率、Wasserstein距离和基于曲率的消息传递；结合MLP、GNN进行融合和下游任务预测。

**📊 数据集**

使用 11 个公开数据集：5 个节点分类（Cora、Citeseer、Instagram、Photo、WikiCS），4 个链路预测（Baby、Sports、MovieLen-1M、MovieLen-10M），2 个图问答（ExplaGraphs、WebQSP）。

**📈 对比分析**

与 20 种基线（传统 GNN、PLM、LLM+GNN 复合模型）比较。CureLLM 在所有任务上均取得最高或次高分，节点分类平均准确率 79.73%，链路预测 Recall@10/Precision@10 最高（如 MovieLen-10M Recall 14.95%），图问答准确率 87.21%，显著优于最强基线。

**⚠️ 局限性**

局限性：①实验主要集中在中小规模图，未评估极大图下的可扩展性；②依赖预训练 LLM 和 GNN 的冻结特征，需额外计算曲率和 Wasserstein 距离；③未探究不同曲率阈值或负曲率边的进一步利用，可能导致信息丢失。

---

## 502. Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions

**arXiv ID:** 2606.06085 | [PDF](https://arxiv.org/pdf/2606.06085v1)

**作者:** Anders Bo Ipsen `[一作]` (Aarhus University), Stratis Skoulakis `[通讯]` (Aarhus University)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5020247743)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了离散第一价格拍卖中近似相关均衡的收入，展示了任何ϵ-近似相关均衡的收入至少为v_2 - Θ(1/k) - Θ(ϵ k^2)。

**💡 创新点**

首次建立了无交换后悔竞标者在第一价格拍卖中产生的收入的多项式收敛速率。

**🔧 技术方法**

采用了在线学习算法和双重拟合方法来分析竞标者的出价行为。

**📊 数据集**

使用了离散第一价格拍卖的模拟数据集，竞标者的出价集为ℬ = {0, 1/k, …, 1 - 1/k, 1}。

**📈 对比分析**

与之前的研究相比，提供了更快的时间平均收入收敛速率，证明了在使用无交换后悔算法的情况下，收入在多项式数量的轮次内接近第二高估值。

**⚠️ 局限性**

在所需的轮次T上提供的上界相对较大，尤其是当所有竞标者使用最优算法时，T的上界为k^𝒪(log k)。

---

## 503. Adaptation of the hybrid fictitious domain-immersed boundary method for Reynolds-averaged turbulence modeling

**arXiv ID:** 2606.06135 | [PDF](https://arxiv.org/pdf/2606.06135v1)

**作者:** Lucie Kubíčková `[一作]`, Martin Isoz `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了基于Hybrid Fictitious Domain-Immersed Boundary (HFDIB) 的稳态RANS求解器openHFDIBRANS，能够在非贴合网格上求解多种两方程湍流模型并使用壁函数；

**💡 创新点**

创新点在于将直接迫使IBM与RANS模型结合，并通过摩擦速度计算、紧耦合湍流变量、以及仅在流体区强制无压缩性等技术实现了稳态求解的稳定性和准确性；

**🔧 技术方法**

技术包括OpenFOAM框架、SIMPLE算法改进、壁函数壁面处理、掩蔽场α、摩擦速度求解、以及混合虚拟域IBM；

**📊 数据集**

通过一系列标准基准测试（2D/3D管流、反向台阶、圆柱、NACA‑0009、Ahmed体等）以及对应的实验数据或simpleFoam参考解进行验证；

**📈 对比分析**

与simpleFoam的数值比较显示在多数基准上误差在0.1–5％之间，壁面剪切、流速、湍动粘度场基本一致，计算速度快且对几何变化鲁棒；

**⚠️ 局限性**

局限在高Re、未分辨边界层时易低估壁面剪切、回流长度以及在部分基准中预测的wake尺寸略大，且无法捕捉湍流过渡或脉动等细节。

---

## 504. VZCrash: A Large-Scale IMU Dataset of Ego-Vehicle Crashes

**arXiv ID:** 2606.06074 | [PDF](https://arxiv.org/pdf/2606.06074v1)

**作者:** Tommaso Bianconcini `[一作]` (Verizon Connect), Leonardo Taccari `[通讯]` (Verizon Connect)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了全球最大公开车碰撞IMU数据集VZCrash，并在其上开展深度学习与基线模型的实验评测。

**💡 创新点**

创新点在于公开了规模近19万事件、3.1万次验证碰撞的高频IMU数据，并系统研究了数据量对模型性能的影响。

**🔧 技术方法**

实验采用物理阈值基线、CNN‑RNN、1D Swin Transformer、CNN‑Transformer、Chronos‑Bolt以及将加速度转换为Scalogram后使用MobileNetV3的多模型方案。

**📊 数据集**

使用的数据集为VZCrash，涵盖2020‑2025年73,010辆商用车收集的约19万条样本（含3.1万起碰撞、1.58万负样本）。

**📈 对比分析**

模型按车辆分层划分训练/验证/测试集，评估以AP为指标，CNN‑RNN和CNN‑Transformer在完整数据集上达约97.5% AP，物理基线89.6%；在真实稀缺事件（约0.02%正例）中AP下降至≈86%，小样本模型性能更差。

**⚠️ 局限性**

局限性包括仅使用加速度信号、缺乏多模态与更细粒度标签，在极低正例比例下仍产生大量误报，且大模型对数据量要求高。

---

## 505. Tight list replicability bounds via a novel sphere covering theorem

**arXiv ID:** 2606.06148 | [PDF](https://arxiv.org/pdf/2606.06148v1)

**作者:** Ari Blondal `[一作]` (McGill University), Sivan Tretiak `[通讯]` (Ohio State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

本文通过提出新的球面覆盖定理，给出了学习理论中列表可复制性（list replicability）的紧确下界，并在 VC 类和大边距半空间（large‑margin half‑spaces）等重要概念类上实现了最优的列表大小。

**💡 创新点**

创新点在于：①提出“每个开集都包含在开放半球内的球面覆盖定理”，从而将传统的 Borsuk–Ulam 推理进一步强化；②利用该定理得到对列表可复制性数的改进下界，使得 VC 类的列表大小在任何 ϵ<1/2 时均至少为 d；③在大边距半空间中精确划分不同边距 γ 的两个极端：γ<1/√2 时列表大小为 d，γ 接近 1 时最小列表大小为 ⌈d/2⌉+1；④证明对线性分类器的限制列表大小也达到 d。

**🔧 技术方法**

主要技术：拓扑方法（Borsuk–Ulam 定理、球面覆盖定理、Lebesgue 数 Lemma）、连续映射与分区细分、以及与可复制性定义相结合的覆盖与重叠度分析。

**📊 数据集**

该工作为纯理论研究，未使用实际数据集；所有结果均通过严谨的数学证明得到。

**📈 对比分析**

相比以往工作（如使用局部 Borsuk–Ulam 得到的 ⌈d/2⌉+1 下界），本文在大多数参数区间内提供了更紧的下界，消除了 1/2 系数的人工影响，达到了理论上的最优性；同时给出了在大边距半空间中完整的列表大小区间，完成了此前的开放问题。

**⚠️ 局限性**

局限性包括：①当前结果仅适用于满足“开集位于开放半球内”的覆盖结构；②对其它更一般的概念类或更复杂的学习模型（如深度网络）仍需进一步研究；③在中间边距范围 1/√d < γ < 1/√2 的列表可复制性数仍未完全确定。

---

## 506. Workload-Aware Autotuning of Block Size in Square-Root Decomposition

**arXiv ID:** 2606.06145 | [PDF](https://arxiv.org/pdf/2606.06145v1)

**作者:** Ruize Zhao `[一作]` `[通讯]` (Xi'an Jiaotong University), Ruize Zhao (Xi'an Jiaotong University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

对平方根分解的块大小选择进行自动调优实验，验证工作负载和平台感知的模型可提升性能。

**💡 创新点**

首次将块大小调优视为可重复的算法工程问题，提出基于KNN-9的学习策略并引入置信门控。

**🔧 技术方法**

采用非参数KNN-9、岭回归、随机森林等机器学习模型，结合交叉验证、置信门控、外部迁移验证以及真实存储轨迹映射。

**📊 数据集**

使用3840个合成工作负载（8种家庭、4种n值、2种m/n、3种查询率），并在6000个外部工作负载与Baleen实时轨迹窗口上进行验证。

**📈 对比分析**

与固定√n、Fenwick BIT和段树基线对比，KNN-9策略平均误差下降0.22，速度提升1.15倍，置信门控保持1.13倍且下降率降低。

**⚠️ 局限性**

结果受平台、编译器和工作负载范围限制；短前缀在线调优无效，迁移性能对编译器敏感，且未评估不同CPU/OS环境。

---

## 507. Beyond Semantic Organization: Memory as Execution State Management for Long-Horizon Agents

**arXiv ID:** 2606.06090 | [PDF](https://arxiv.org/pdf/2606.06090v1)

**作者:** Yaoqi Chen `[一作]` (University Of Science And Technology Of China), Qi Chen `[通讯]` (Microsoft)

**通讯引用:** 3014 | [OpenAlex ID](https://openalex.org/A5071396780)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新的记忆管理框架Memory as Agent-Guided Exploration（MAGe），将记忆视为主动的执行状态管理器；

**💡 创新点**

创新点在于使用两层层次化的状态树（原始轨迹层与压缩摘要层）并设计四个操作（Grow、Compress、Maintain、Revise）实现闭环错误检测与隔离，避免语义检索导致的状态碎片化；

**🔧 技术方法**

技术包括基于LLM的RAG与Agent执行的交互、树结构管理、边界感知压缩、LLM验证摘要、错误恢复分支；

**📊 数据集**

在MemoryArena交互式长期任务基准（包含Web购物、旅游规划、Web搜索、形式推理）上进行实验；

**📈 对比分析**

与长上下文、HippoRAG2、MemoRAG、Mem0、ReasoningBank、MemoryOS、SimpleMem等基线比较，MAGe在任务成功率上提升约7.8–20.4个百分点、平均任务进度提升约8.7个百分点，同时令token消耗下降55.1%；

**⚠️ 局限性**

局限性包括对子目标边界划分的依赖、LLM在Maintain阶段的误判风险、在极端长轨迹下树结构维护成本以及对非交互式检索场景的适用性待验证。

---

## 508. LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents

**arXiv ID:** 2606.06087 | [PDF](https://arxiv.org/pdf/2606.06087v1)

**作者:** Aofan Yu `[一作]` (Shanghai Jiao Tong University), Jianghao Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 570 | [OpenAlex ID](https://openalex.org/A5036057873)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将文本技能通过预训练的超网络转换为可插拔的LoRA适配器，从而将技能知识从上下文空间迁移到权重空间，减少上下文开销并提升模块化与可控性。

**💡 创新点**

创新点在于：①利用超网络一次性生成技能LoRA，避免逐步推理时的文本注入；②将技能存储在权重空间，实现零文本注入的插件化；③证明生成的LoRA权重具有结构化语义几何、可通过缩放系数精准控制以及可在参数空间进行安全组合。

**🔧 技术方法**

核心技术包括：超网络（Hypernetwork）生成LoRA权重；文档级预训练与轨迹监督微调；LoRA注入系数控制；组件级组合与权重空间算术。

**📊 数据集**

使用的数据集为 ALFWorld（文本交互式家居任务）和 Search‑QA（单跳与多跳问答）。

**📈 对比分析**

与传统的在提示中插入技能（In‑Context Skill）和其他基线（Vanilla、Few‑Shot、RAG 等）相比，LatentSkill 在 ALFWorld 的 seen、unseen 分别提升 21.4 / 13.4 分，Search‑QA 的平均 EM 提升 3.0 分；同时预填充词量下降 64.1%（ALFWorld）与 72.2%（Search‑QA），显著降低推理成本。

**⚠️ 局限性**

局限性包括：仅在 ALFWorld 与 Search‑QA 两个基准上评估；仅使用 Qwen3‑8B 作为冻结基础模型，未探究不同模型规模或 LoRA 配置下的表现；未验证在更广泛的代理部署场景（如网页浏览、软件工程、协作多智能体）中的通用性。

---

## 509. Learning Visual Spatial Planning from Symbolic State via Modality-Gap-Aware Self-Distillation

**arXiv ID:** 2606.06076 | [PDF](https://arxiv.org/pdf/2606.06076v1)

**作者:** Haocheng Luo `[一作]` (Tsinghua University), Xiu Li `[通讯]` (Tsinghua University)

**通讯引用:** 9932 | [OpenAlex ID](https://openalex.org/A5100754504)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个两阶段的模态差距自蒸馏框架（MGSD），先通过感知导向的监督微调使视觉模型能够可靠地从图像中恢复任务相关的状态结构，再通过符号引导的在策略自蒸馏（OPSD）将符号规划行为迁移到视觉模型，最终实现纯视觉输入下的空间规划。

**💡 创新点**

创新点包括：① 在视觉规划中首次系统性地将符号状态作为训练时的特权监督；② 通过两阶段训练（冷启动感知微调 + 在策略自蒸馏）在同一模型中同时解决感知噪声和多步推理的双重瓶颈；③ 采用逆 KL 风格的上下文蒸馏，对学生自身生成的轨迹提供密集的 token 级监督，从而显著降低训练与推理分布不匹配。

**🔧 技术方法**

使用的技术主要有：视觉语言模型（Qwen3-VL-4B/8B-Instruct）、LoRA 适配器、感知导向的监督微调（Perception SFT）、符号引导的在策略自蒸馏（OPSD）与逆 KL 蒸馏损失、Frozen teacher（文本仅模型）以及一系列训练超参数设置。

**📊 数据集**

数据集包含三类视觉规划基准：FrozenLake、Maze、MiniBehaviour。每个基准都提供图像观察、对应的符号状态描述以及参考动作计划，感知微调数据为 18K 个自动生成的图像-问题-答案对。

**📈 对比分析**

与多种公开与专有 VLM（Claude-4.5-Haiku、GPT-4o、GPT-5、Gemini 系列、LLaVA-OneVision、InternVL3、Qwen 系列等）在相同的纯视觉输入/文本输出接口下进行比较。MGSD 在 Qwen3-VL-4B 上从 11.2% 提升至 30.5%，在 8B 上从 17.2% 提升至 35.6%；在宏平均上分别提升 19.3% 和 18.4%，并显著缩小与符号输入上限的差距，仅剩 6–7 点。相比之下，直接 SFT 或基于奖励的 RL 方法提升幅度有限。

**⚠️ 局限性**

局限性包括：① 需要配对的视觉+符号训练数据，适用于仿真环境但在开放世界中难以获取；② 仅验证了离散动作空间的结构化任务，未涵盖连续控制、动态场景、部分可观测或长期推理；③ 尽管性能提升显著，但在视觉歧义或分布外场景下的中间推理仍不保证准确。

---

## 510. A Finite Certificate for the Positive $n=9$ Vasc Inequality

**arXiv ID:** 2606.06136 | [PDF](https://arxiv.org/pdf/2606.06136v1)

**作者:** Dakai Guo `[一作]` (Chinese Academy of Sciences), Ruyong Feng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 311 | [OpenAlex ID](https://openalex.org/A5050890628)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

证明了正实数域下 n=9 的 Vasc 周期不等式成立。

**💡 创新点**

创新点是首次提供完整的有限精确证明证书，覆盖所有 8! 个排序区域，并利用人工引导的 LLM 代理自动生成与验证。

**🔧 技术方法**

使用了人工指导的 LLM 代理 MechMath、Python 自动化工具、Polya 多项式放缩、AM‑GM 中点重叠等技术。

**📊 数据集**

无传统数据集，采用理论分析和计算机生成的 40,320 条最终行证书文件。

**📈 对比分析**

与以往 SDS 方法相比，本方法在 n=9 的所有 40,320 个排序区域实现了完全符号验证，证明了全覆盖且验证通过。

**⚠️ 局限性**

局限性在于仅适用于 n=9，未解决 n=11 以及零坐标边界情况，证书仅在固定最大值和排序后的子空间有效。

---

## 511. Validation of graph databases against PG-Schema

**arXiv ID:** 2606.06127 | [PDF](https://arxiv.org/pdf/2606.06127v1)

**作者:** Jacek Ciszewski `[一作]` (University of Warsaw), Filip Murlak `[通讯]` (University of Warsaw)

**通讯引用:** 915 | [OpenAlex ID](https://openalex.org/A5037986297)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在属性图数据库中验证实例是否符合给定图类型（无完整约束）的复杂性，分解为节点类型、边类型和整体图类型的一致性判定，并给出了相应的算法与复杂度分析。

**💡 创新点**

创新点在于首次给出完整的图类型验证框架，证明其综合复杂度为NP‑complete、数据复杂度为PTIME；揭示“|自由”类型在多层归一化后可实现多项式时间验证；提出基于匹配与递归消除通配符的高效判定算法。

**🔧 技术方法**

主要技术包括：属性图的形式化定义、类型表达式归一化与T‑expansion、分支集覆盖与推理、图匹配与Hopcroft–Karp算法、递归消除通配符以及层次化DAG构造等。

**📊 数据集**

论文未使用真实数据集，而是通过理论模型与示例图（如小型客户图）进行说明与演示。

**📈 对比分析**

与现有图查询/验证方法相比，本文通过理论证明给出了最优时间复杂度；在|自由类型下给出多项式时间算法，理论上性能优于一般NP‑hard情况，尽管没有实验验证。

**⚠️ 局限性**

局限性包括：仅对无完整约束的图类型给出绝对复杂度；对包含完整约束的情况复杂度取决于宿主查询语言，未给出通用结果；未考虑并行/分布式实现以及与实际数据库系统的兼容性。

---

## 512. Computation-Aware Event-to-Frame Reconstruction via Selective Attention

**arXiv ID:** 2606.06142 | [PDF](https://arxiv.org/pdf/2606.06142v1)

**作者:** Jingqian Wu `[一作]` (University of Hong Kong), Edmund Y. Lam `[通讯]` (University of Hong Kong)

**通讯引用:** 9793 | [OpenAlex ID](https://openalex.org/A5008832723)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种计算友好的事件到帧重建框架，利用因果递归编码器-解码器和选择性上下文融合实现高质量重建。

**💡 创新点**

创新在于将轻量化混合注意力机制嵌入卷积上下文融合，既保持计算效率又提升特征选择性，避免全局自注意力的高成本。

**🔧 技术方法**

采用卷积递归单元（ConvLSTM）、轻量化混合注意力、跨模态交叉注意、线性自注意力、通道/空间重加权等技术。

**📊 数据集**

使用公开的事件相机基准数据集ECD和MVSEC进行训练和评估，并通过MSCOCO合成训练数据。

**📈 对比分析**

与基线（E2VID、FireNet、ET-Net等）对比，在ECD上实现MSE 0.034、SSIM 0.554，参数仅10.16M，MAC 37.83G，显著低于ET-Net的资源消耗并保持或超越其性能；在MVSEC上取得竞争性LPIPS 0.530。

**⚠️ 局限性**

局限在于仍需更好地处理极端光照和长时序动态，且混合注意虽轻量但在极低功耗嵌入式平台上仍需进一步压缩。

---

## 513. Towards Realistic 3D Sonar Simulation

**arXiv ID:** 2606.06130 | [PDF](https://arxiv.org/pdf/2606.06130v1)

**作者:** Youssef Attia `[一作]` (University of Genova), Enrico Simetti `[通讯]` (University of Genova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了基于Isaac Sim的可GPU加速3D声呐模拟器，并将其与FastLIO2 SLAM集成进行硬件在环（HIL）验证。

**💡 创新点**

提出了一种将几何渲染与物理声学传播相结合的模块化架构，支持体积感知、混合传播、散射及相位感知信号生成，填补了传统声呐模拟仅依赖几何渲染的空白。

**🔧 技术方法**

使用NVIDIA Isaac Sim、RTX光追/OptiX、CUDA、FastLIO2 SLAM、BELLHOP等技术实现GPU加速渲染与声学传播模型的协同工作。

**📊 数据集**

采用Water Linked 3D‑15声呐的数据表参数、BlueROV2实验场景以及港口船坞钢板现场真实声呐点云进行实验与对比。

**📈 对比分析**

通过在Jetson Orin Nano上运行FastLIO2 SLAM与多传感器融合（声呐、DVL、IMU、压力），实现实时映射；点云与真实数据对比显示模拟点云稠密但仍与真实数据在稀疏性、散射和遮挡方面存在差异。

**⚠️ 局限性**

主要局限在于仿真仍采用单次射线几何模型，缺乏完整的声速折射、多路径与相位干涉建模，导致与真实数据存在显著的sim-to-real差距。

---

## 514. MS-DKC: A Dataset Knowledge Card Framework for Designing and Adapting Medical Image Segmentation Models

**arXiv ID:** 2606.06103 | [PDF](https://arxiv.org/pdf/2606.06103v1)

**作者:** Tariq M. Khan `[一作]` (Chulalongkorn University), Mohammad AU Khan `[通讯]` (COMSATS University Islamabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出 Medical Segmentation Dataset Knowledge Card (MS‑DKC) 框架，将医学分割数据集的图像采集、目标形态、监督、上下文与部署风险等五层指标量化，并通过风险映射指导模型设计、训练与评估，实现数据集驱动的分割方法。

**💡 创新点**

创新点在于构建可追溯的从测量到风险再到设计优先级的闭环流程，打破传统“以架构为先”的范式，强调基于可量化数据集特征做模型与评价决策。

**🔧 技术方法**

技术实现包括：MS‑DKC 结构设计、指标量化与质量标注、风险推理函数 Ψ 与优先级映射 Γ；在 DRIVE、ISIC2018 与 ACDC 三大基准上实验，并使用 DKC‑TNet‑v2、SA‑UNetv2‑DKC‑AmbRef、MS‑DKC‑AttNextTopo‑VCSF‑NoAug 等模型验证。

**📊 数据集**

使用的三大医学分割数据集为：DRIVE（血管细分）、ISIC2018（皮肤病变）和 ACDC（心脏多类别）。

**📈 对比分析**

比较方法采用 Dice、IoU、敏感度、特异度、AUC、Boundary F1、ASSD 等指标；在 DRIVE 上 DKC‑TNet‑v2 达到 Dice 0.8044/IoU 0.6730，SA‑UNetv2‑DKC‑AmbRef 达到 Dice 0.8141/IoU 0.6865；在 ISIC 上 MS‑DKC‑AttNextTopo‑VCSF‑NoAug 获得 Dice 0.8872/IoU 0.8214；在 ACDC 上通过四类 softmax 与类别平衡 Dice/CE 监督验证多类别策略的有效性。

**⚠️ 局限性**

局限性：指标映射规则主要手工设定，缺乏统一自动化决策器；实验仅覆盖三种基准，需进一步验证在更大、多样化数据集上的泛化；对监督质量变化的动态更新机制尚未深入研究。

---

## 515. LLM-Based Porting of Optimized C++ to CUDA Through Deoptimization and Reoptimization

**arXiv ID:** 2606.06063 | [PDF](https://arxiv.org/pdf/2606.06063v1)

**作者:** Daichi Mukunoki `[一作]` (Nagoya University), Takahiro Katagiri `[通讯]` (Nagoya University)

**通讯引用:** 673 | [OpenAlex ID](https://openalex.org/A5078063020)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并评估了一种在LLM（大型语言模型）驱动的C++到CUDA迁移过程中先进行去优化（Deoptimization）再进行重新翻译和再优化（Reoptimization）的工作流，探讨其在不同核算模型、单次和迭代生成中的有效性。

**💡 创新点**

创新点在于将去优化作为迁移前的预处理步骤，结合LLM翻译与迭代修复机制，系统地比较了直接翻译与去优化+再优化的性能与可行性，并提供了以LLM调用次数匹配的对照实验。

**🔧 技术方法**

采用了两种大规模MoE模型——OpenAI的O120（117B参数）和Alibaba的Q235（235B参数），在GH200系统上进行CUDA代码生成、再优化与迭代修复；使用LLM作为项目经理、程序员与修复者，执行多代生成、编译、验证与选择。

**📊 数据集**

使用了12个高性能计算（HPC）基准核算，包括Conv2D、BFFT、Softmax、BGEMM、DFSpMM、FFT、BTDMA、DDGEMM、Stencil、SpMM、GEMM和SpMV，分别覆盖CPU优化、数据布局、并行策略等多种场景。

**📈 对比分析**

比较方法为：单次生成（50次实验）与迭代生成（最多3代，每代3个候选+修复），记录成功率和成功样本的中位性能；通过两侧Mann‑Whitney U检验并做BH‑FDR校正；结果显示去优化+再优化在部分核算中显著加速（如Softmax、FFT等），但在另一些核算（如BGEMM、SpMM）表现不如直接翻译；迭代过程显著缩小两工作流间差距，尤其在O120模型上。

**⚠️ 局限性**

局限性包括：仅评估两种LLM和单一GPU环境；样本规模有限（12核算、每对模型50次实验），统计显著性受限；简化度仅用LOC度量，未探索更细粒度的简化指标；验证仅在给定尺寸、布局和稀疏模式下进行，未覆盖更广泛的真实场景。

---

## 516. A Framework for Measuring Appropriate Reliance on Set-Valued AI Advice

**arXiv ID:** 2606.06081 | [PDF](https://arxiv.org/pdf/2606.06081v1)

**作者:** Ranjan Mishra `[一作]` (University of Groningen), Jakob Schoeffer `[通讯]` (University of Groningen)

**通讯引用:** 279 | [OpenAlex ID](https://openalex.org/A5090234310)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了针对分类和回归任务的、基于判断-顾问（JAS）框架的首个正式方法，用于衡量人类在接收集合形式AI建议时的恰当依赖。

**💡 创新点**

创新点在于将传统点预测依赖度量扩展到集合建议，定义了两个维度的分类指标（CRR_AI、CRR_self）和两个回归指标（AIR_quant、AIR_qual），从数量与质量两方面同时描述依赖行为，揭示了自动化偏差、算法厌恶等失效模式。

**🔧 技术方法**

采用序列判断-顾问实验设计、集合预测（Top‑k、合规预测）和预测区间，构造了正式的统计公式与案例演示，利用合成/模拟数据展示指标计算。

**📊 数据集**

示例使用了合规预测集（28个职业分类）和房价预测区间（数值任务），并未使用公开大规模真实数据集，主要以仿真情景为主。

**📈 对比分析**

与传统的准确率、switch/agreement、WoA等指标对比，显示单一指标难以区分有益与有害的依赖；利用等高线分析说明相同准确率下不同CRR组合对应不同失败模式，表现出更细粒度诊断能力。

**⚠️ 局限性**

局限包括：难以在人类初始与最终均在集合内的情形下拆分AI与人类的影响；回归指标假设向区间中点靠拢代表依赖，忽视区间边界；需要先决的初始决策导致锚定偏差；指标依赖完整的实验数据，难以直接迁移到无序列记录的实际部署。

---

## 517. Adaptive state-action abstractions via rate-distortion

**arXiv ID:** 2606.06123 | [PDF](https://arxiv.org/pdf/2606.06123v1)

**作者:** Fernando E. Rosas `[一作]` (University of Sussex), Fernando E. Rosas `[通讯]` (University of Sussex)

**通讯引用:** 6218 | [OpenAlex ID](https://openalex.org/A5020498855)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于率失真理论的软状态-动作抽象方法，并给出自适应细化原则，能在学习过程中根据误差动态调整抽象精度。

**💡 创新点**

创新点在于将价值误差拆分为学习误差（Bellman残差）与抽象误差（bisimulation度量），并通过阈值原则自动决定何时细化抽象，同时实现软抽象和率失真连续可调。

**🔧 技术方法**

使用率失真优化、bisimulation度量、Bellman残差估计以及软状态-动作抽象（概率编码/解码）技术，构造可调分辨率的抽象空间。

**📊 数据集**

在多种离散基准上验证，包括Maze、PuddleWorld、MiniGrid（任务阶段化）以及SysAdmin环形多机系统，均采用完整状态空间作为基准。

**📈 对比分析**

与精确的bisimulation、MDP同态以及固定抽象的基线相比，所提方法在保持接近最优回报的同时，显著减少有效状态-动作对的数量；在大规模SysAdmin任务中，抽象后信息量下降而性能仅低于1%。

**⚠️ 局限性**

局限性：仅在离散标量环境中测试，需先验环境模型或完整奖励/转移信息；未针对模型无关或深度强化学习进行实验，且细化策略基于理论阈值，实际应用中可能需要经验调参。

---

## 518. Diff-CA: Separating Common and Salient Factors with Diffusion Models

**arXiv ID:** 2606.06120 | [PDF](https://arxiv.org/pdf/2606.06120v1)

**作者:** Michaël Soumm `[一作]` (INRIA), Alasdair Newson `[通讯]` (Télécom Paris)

**通讯引用:** 668 | [OpenAlex ID](https://openalex.org/A5047314512)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于扩散模型的对比分析方法Diff-CA，能够在弱监督下将图像条件分解为公共因素和显著因素，实现高保真图像重构与精准编辑。

**💡 创新点**

创新点包括：①在弱监督下证明加性分解可识别；②构造DINOv3交叉查询+颜色标记的高质量条件空间；③采用两阶段训练、对抗GRL与循环一致性等多重约束实现子空间分离。

**🔧 技术方法**

使用的技术包括：潜在扩散模型、交叉注意力条件编码、DINOv3特征、流匹配训练、Transformer分离编码器、梯度反转对抗、循环一致性损失与互信息下界。

**📊 数据集**

实验数据集涵盖FFHQ/CelebA‑HQ人脸、AFHQ猫狗、BraTS 2023脑部MRI等多领域任务。

**📈 对比分析**

与DiffAE、EncDiff、CLIP/DINOv3等条件编码基线以及MM‑cVAE、SepVAE、DoubleInfoGAN等对比，Diff‑CA在重构指标（SSIM↑、LPIPS↓、FID↓）和显著属性交换准确率（如玻璃交换达94.5%）上均显著优于对手。

**⚠️ 局限性**

局限性在于仅处理2D图像、仅采用加性分解，未考虑显著因素在两分布均出现的情况，也未扩展至3D医学体积。

---

## 519. Where, What, Why, and Importance: Structured Defect Grounding for Text-to-Image Feedback

**arXiv ID:** 2606.06113 | [PDF](https://arxiv.org/pdf/2606.06113v1)

**作者:** Huaisong Zhang `[一作]` (Tsinghua University), Chun Yuan `[通讯]` (Tsinghua University)

**通讯引用:** 33291 | [OpenAlex ID](https://openalex.org/A5008769328)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了结构化缺陷定位（Structured Defect Grounding, SDG）框架，用于检测文本到图像模型生成图像中的缺陷并生成可用于模型对齐的稠密奖励；

**💡 创新点**

创新点在于把缺陷诊断转化为多实例结构化集合预测（位置、类型、原因、重要度），构建30K图像缺陷数据集（SDG-30K）及评估协议，并将检测结果转换为空间奖励（BoxFlow-GRPO）用于扩散模型的强化学习；

**🔧 技术方法**

采用了 VLM（Qwen3‑VL）进行监督微调（SFT）与群组相对策略优化（GRPO），结合链式推理（CoT）、格式校验奖励；后续通过 BoxFlow‑GRPO 将检测到的盒子与重要度映射为空间奖励，使用 RL 对齐扩散模型；在图像修复中使用 GPT‑Image‑1.5 结合盒子叠加和结构化文本反馈；

**📊 数据集**

使用了自制的 SDG‑30K 数据集（30,096 张图像，覆盖 FLUX.2‑dev、Z‑Image‑Turbo、LongCat‑Image、SANA‑1.5 四大 T2I 生成器），并在 RichHF‑18K、DrawBench 等公开数据集上进行验证；

**📈 对比分析**

与零拷贝 GPT‑5.4、Gemini 3 Pro 进行对比，SDG 在 BoxF1@0.5 上接近人类标注（artifact 0.263/0.278，misalignment 0.387/0.409）；在对齐任务中 BoxFlow‑GRPO 在质量、真实性等指标上优于基线 RL 方案；在图像修复中，SDG 的 Good% 较 ImageDoctor 与 Fixed 提升至约11%；

**⚠️ 局限性**

局限性包括仅涵盖两类缺陷（artifact 与 misalignment），对高分辨率和更丰富缺陷类型的泛化仍待验证，且依赖人工框注和 VLM 推理，导致推理速度和成本较高；

---

## 520. WebKnoGraph: GNN-Powered Internal Linking

**arXiv ID:** 2606.06106 | [PDF](https://arxiv.org/pdf/2606.06106v1)

**作者:** Emilija Gjorgjevska `[一作]` (Technical University of Munich), Miroslav Mirchev `[通讯]` (Ss. Cyril and Methodius University in Skopje)

**通讯引用:** 299 | [OpenAlex ID](https://openalex.org/A5056081801)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 WebKnoGraph 框架，用于在正式发布前通过图模型评估网站内部链接改动的权威分配和语义连贯性。

**💡 创新点**

将内部链接优化转化为可执行的图干预问题，结合 GNN 预测候选链接并在不同外部主机网络中模拟评估，首次在真实网站抓取数据上验证自动与专家辅助策略的权衡。

**🔧 技术方法**

使用 GraphSAGE 进行节点嵌入与链接预测，利用 768 维文本嵌入（nomic‑embed‑text‑v1）作为特征，采用 PageRank 计算权威并量化 Authority Yield、Volatility、Loss‑Gain Ratio 与语义连贯性。

**📊 数据集**

数据来源为 2025 年 Kalicube.com 的完整抓取（约 1,841 页）以及 200,000 条 FineWeb 采样网页构建经验主机图，另外使用 Barabási‑Albert 模型生成的 100,000 节点的合成主机图。

**📈 对比分析**

通过在两种主机环境（FineWeb 与 BA）中执行 10,000 次桥接模拟，比较自动与专家辅助两种选链模式。结果显示：自动策略在平均 Authority Yield 上表现更好，但语义连贯性下降更明显；专家辅助策略在语义连贯性上更稳定，且在某些策略（如 Low）能实现最高 Authority Yield，尽管 Loss‑Gain 失衡。

**⚠️ 局限性**

限制包括：仅基于抓取的结构与内容评估，未包含用户行为、点击、搜索算法更新等实时信号；未在多网站上进行跨域验证；未通过在线 A/B 测试检验实际排名或流量影响。

---

## 521. Harnessing Structural Context for Entity Alignment Foundation Models

**arXiv ID:** 2606.06109 | [PDF](https://arxiv.org/pdf/2606.06109v1)

**作者:** Xingyu Chen `[一作]` (Nanjing University), Wei Hu `[通讯]` (Nanjing University)

**通讯引用:** 35064 | [OpenAlex ID](https://openalex.org/A5031365355)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的跨图交互编码器-结构校准解码器框架，用于可迁移实体对齐；

**💡 创新点**

创新点包括：①在编码阶段引入锚桥实现跨图早期交互传播；②在解码阶段使用实体、邻域、关系及锚支持四视角结构校准，显著提升候选区分；

**🔧 技术方法**

采用查询条件图神经网络、关系图编码、跨图桥接、轻量级MLP结构校准以及对比学习训练；

**📊 数据集**

在29个OpenEA、SRPRS、DBP基准数据集上进行实验，覆盖稀疏/稠密、跨KG与跨语言、不同规模图；

**📈 对比分析**

与ULTRA‑EA、EAFM等基线比较，预训练模型在所有数据集上均超越基线，finetune进一步提升，特别在大规模稠密图表现突出；

**⚠️ 局限性**

局限性包括：仍需更强的传播模块适配更大异构图；对锚桥质量依赖较高；结构校准在候选集过大时计算成本上升。

---

## 522. Step-adaptive multimodal fusion network with multi-scale cloud feature learning for ultra-short-term solar irradiance forecasting

**arXiv ID:** 2606.06102 | [PDF](https://arxiv.org/pdf/2606.06102v1)

**作者:** Jingxin Zhang Xiaoqin Wang `[一作]` `[通讯]` (Southeast University), Jingxin Zhang Xiaoqin Wang (Southeast University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一种多模态融合框架 IST，用于超短时光伏辐照度预测。

**💡 创新点**

创新点包括：1) 在云图像中使用 InceptionNeXt 进行多尺度、多方向特征提取；2) 引入 Step-Adaptive Low‑Frequency Compensation Unit (SALFCU) 实现预测步长自适应的低频信息补偿；3) 在时间维上使用带可学习周期查询的 TempAttnLSTM 捕捉全局时序依赖。

**🔧 技术方法**

采用深度卷积网络（InceptionNeXt）、离散小波变换 + 门控机制（SALFCU）、多头注意力 + LSTM（TempAttnLSTM）以及标准数据预处理与标准化流程。

**📊 数据集**

使用公开的 NREL 数据集（10 分钟采样）和山东省真实光伏电站实测数据（5 分钟采样）。

**📈 对比分析**

与 9 种基线（单时序、单图像、传统多模态）在 MAE、RMSE、nMAE、nRMSE、R² 等指标上进行对比，IST 在 NREL 上 MAE/ RMSE 分别为 91.45 / 149.28，R² 0.7618；在山东实测上 MAE/ RMSE 分别为 50.86 / 68.53，R² 0.8386，均显著优于所有对比模型。

**⚠️ 局限性**

局限性：模型依赖高质量云图像和同步气象序列；在极端多云或缺少图像的情况下性能下降；缺少对不同光伏系统规模与不同地区气候的更广泛验证。

---

## 523. CogManip: Benchmarking Manipulative Behavior in Multi-Turn Interactions with Large Language Model

**arXiv ID:** 2606.06099 | [PDF](https://arxiv.org/pdf/2606.06099v1)

**作者:** Zeyang Yue `[一作]`, Yi Zeng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了CogToM——一个包含46种心理学Theory of Mind任务范式、8000+中英双语实例、由49名专家评审验证的理论基础评测基准，并对22款LLM进行系统零样本与链式推理评估。

**💡 创新点**

①将心理学ToM多样范式转化为LLM友好的多选格式；②规模化且引入新颖任务提升辨别力；③联动人类注释一致性与发展里程碑，揭示LLM与人类ToM的认知差异与莫拉维奇悖论。

**🔧 技术方法**

使用大型LLM（如GPT‑5.1、Qwen3‑Max）进行零样本与CoT评测；多轮人工审核与专家验证构建数据；统计人类IAR与模型准确率相关性；对比不同模型的参数规模与表现。

**📊 数据集**

CogToM数据集：46个ToM任务范式，8013条中英双语实例（每个任务54‑130组），通过LLM扩展至5‑10条样本；与20余款LLM进行对比。

**📈 对比分析**

采用零样本多选输出、温度0、选项旋转5次求平均准确率；结果显示最新LLM整体准确率80%+，情感类任务近满分；感知类仅约20%；模型规模与表现呈正相关，开放源模型逐步逼近闭源性能。

**⚠️ 局限性**

仅覆盖中文和英文；文本化转写可能失真；多选格式不捕捉生成式对话；静态场景缺少交互更新；缺乏多模态与文化多样性。

---

## 524. The Complexity of Asynchronous HyperLTL

**arXiv ID:** 2606.06091 | [PDF](https://arxiv.org/pdf/2606.06091v1)

**作者:** Gaëtan Regaud `[一作]` (ENS Rennes), Martin Zimmermann `[通讯]` (Aalborg University)

**通讯引用:** 5496 | [OpenAlex ID](https://openalex.org/A5064596121)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了异步HyperLTL（AHLTL）的可满足性、有限状态可满足性和模型检查的计算复杂度，并给出了精确的可判定性界限。

**💡 创新点**

创新点在于：①将AHLTL的可满足性问题证明为存在量化轨迹时Σⁱ₁-完全、全称量化轨迹时Σⁱ₁-硬且属于Σ²ⁱ；②证明模型检查与有限状态可满足性与第二阶算术真值问题等价，进一步确定了其为最高级别的不可判定性；③通过构造小模型、Skolem函数以及轨迹的编码完成了上述证明，填补了此前该领域的复杂度空白。

**🔧 技术方法**

主要技术包括：小模型性质证明、Skolem化、轨迹与指针赋值的二阶算术编码、对AHLTL语义的逐层归约以及与第二阶算术真值问题的双向多项式时间归约。

**📊 数据集**

本文未使用任何实验数据集，而是完全基于理论证明与归约。

**📈 对比分析**

由于没有实验评估，本文没有性能对比；其结果纯粹是理论复杂度分析，表明所有相关问题都是高度不可判定的。

**⚠️ 局限性**

局限性：结论仅适用于AHLTL；对其他异步超逻辑（如带上下文、带停顿的逻辑）仍需进一步研究；同时，只给出了复杂度边界，并未给出可实现的实用算法或工具。

---

## 525. SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization

**arXiv ID:** 2606.06079 | [PDF](https://arxiv.org/pdf/2606.06079v1)

**作者:** Qi Zhang `[一作]` (Zhejiang University), Junbo Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 12240 | [OpenAlex ID](https://openalex.org/A5042402520)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可在推理时自我演化的技能构造框架 SkillComposer，分为创建、改进、合并三种操作

**💡 创新点**

将技能构造拆解为可学习的三操作，并通过基于 delta pass‑rate 的拒绝采样进行训练，实现离线、在线与混合部署

**🔧 技术方法**

使用语言模型自我演化、拒绝采样监督、技能库检索、delta pass‑rate 指标、t‑SNE 可视化等技术

**📊 数据集**

使用 τ²‑Bench、LiveCodeBench v6、AppWorld 以及 OpenCodeReasoning 生成的数据

**📈 对比分析**

与无技能、MemP 等基线对比，SkillComposer 在 4B/27B 模型上在 agent 与代码任务中均提升 4.5–9.1 以上，跨域与跨模型泛化表现良好

**⚠️ 局限性**

训练过程耗时高，需要多次推理计算 delta pass，实验仅在 Qwen3.5‑4B/27B 上验证，缺乏更广泛模型和规模的评估

---

## 526. Knowledge Distillation for Visual Autoregressive Models

**arXiv ID:** 2606.06078 | [PDF](https://arxiv.org/pdf/2606.06078v1)

**作者:** Elia Peruzzo `[一作]` (Qualcomm AI Research), Amirhossein Habibian `[通讯]` (Qualcomm AI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并提出了一种针对视觉自回归图像生成模型的知识蒸馏框架；

**💡 创新点**

创新点在于在混合数据–学生上下文中进行自回归 roll‑out，使用教师置信度重加权以及在压缩的视觉词表空间中匹配分布，以降低教师监督噪声；

**🔧 技术方法**

采用自回归解码、教师置信度重加权、词表聚类压缩、并行解码加速训练以及混合数据–学生 roll‑out 等技术；

**📊 数据集**

使用 ImageNet 作为类条件生成任务的数据集；

**📈 对比分析**

与传统 KD、SeqKD 和 GKD 进行比较，VarKD 在 LlamaGen 和 ARPG 两种骨干上均获得更低的 FID、提升的 IS 和更高的精度/召回率，且在小学生模型上效果最显著；

**⚠️ 局限性**

仅在 ImageNet 类条件生成任务上验证，未测试开源文本到图像等更复杂多模态场景；方法依赖教师置信度和词表聚类的可靠性，可能无法充分捕捉语义正确性或提示一致性。

---

## 527. 3D Underwater Path Planning via Generative Flow Field Surrogates

**arXiv ID:** 2606.06077 | [PDF](https://arxiv.org/pdf/2606.06077v1)

**作者:** Zachary Cooper-Baldock `[一作]` (Flinders University), Karl Sammut `[通讯]` (Flinders University)

**通讯引用:** 2137 | [OpenAlex ID](https://openalex.org/A5080292048)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

作者提出了一套基于条件GAN的三维流场预测方法，将其作为高保真RANS CFD数据的替代，集成进能量加权A*路径规划，以实现AUV LAR任务的低耗能轨迹规划。

**💡 创新点**

创新点在于首次系统量化cGAN预测流场在三维AUV路径规划中的价值，并通过两种不同的cGAN架构（Regularised PatchGAN GradNorm 与 2D3DGAN SA）实现约45–60% 的CFD能量与高速度区回避优势，同时保持亚毫秒级推理速度。

**🔧 技术方法**

技术上采用了条件GAN（PatchGAN、2D3DGAN）结合自注意力、FiLM层与跨注意力机制，并使用多尺度损失（对抗、MSE、梯度加权MSE、SSIM、感知损失）训练全卷积生成器。

**📊 数据集**

数据集来自ANSYS Fluent 2023R1生成的550个不同推进速度（0.1–5.0 m/s）和舵角（0–50°）的RANS CFD模拟，经过数据增强后得到4,364个实例，使用128³体素网格作为训练与评估基准。

**📈 对比分析**

在19,800条轨迹上对比四种环境模型（均匀流、RANS、PatchGAN、2D3DGAN SA），评估能量消耗、路径长度、高速区穿越、湍流穿越与计算时间；结果显示cGAN模型在能量上恢复约45–60% 的CFD优势，搜索时间比CFD更快，且推理时间仅约100 μs。

**⚠️ 局限性**

主要限制包括：cGAN模型对极端低速/低角度的近轴对称涡流预测不足；2D3DGAN SA 的显存需求高，难以在嵌入式硬件上部署；并且未考虑非稳态流场与 AUV 动力学的耦合影响。

---

## 528. AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding

**arXiv ID:** 2606.06155 | [PDF](https://arxiv.org/pdf/2606.06155v1)

**作者:** Qize Yu `[一作]` (Peking University), Yingcong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2765 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AffordanceVLA 框架，利用 Which2Act、Where2Act、How2Act 三种结构化可行性预测作为中间表示，实现从视觉语言到机器人动作的精准映射。

**💡 创新点**

将可行性预测内嵌为训练中间目标，采用 Mixture-of-Transformer 分专家架构和三阶段数据训练策略，兼顾语义对齐、结构化可行性预测与控制解码，显著提升跨域泛化和指令遵循。

**🔧 技术方法**

使用预训练 VLM、MoT（Understanding、Affordance Generation、Action 专家）、自监督可行性生成（Which2Act、Where2Act、How2Act）、Transformer 解码器、Diffusion、Smooth-L1、UAA 注意机制、数据增强及合成可行性标注。

**📊 数据集**

利用 AGD20K、RefSpatial、PRISM、InternData-A1 等视觉语言数据，结合合成可行性标注，最终在 LIBERO、CALVIN 与 DROID 真实场景上进行评估。

**📈 对比分析**

在 LIBERO 和 CALVIN 的基准上与 Pi0、π_0.5/π_0.7 等 VLAs 对比，AffordanceVLA 在 LIBERO 平均成功率 95.8%，在 CALVIN 平均长度 4.33，均优于或竞争现有方法；在真实任务中成功率达 88.3%。

**⚠️ 局限性**

对极长时序任务仍受限于缺乏显式长期记忆；How2Act 在桌面双指场景效果有限；可行性生成的准确性受合成标注质量影响。

---

## 529. Symb-xMIL: Symbolic Explanations for Multiple Instance Learning in Digital Pathology

**arXiv ID:** 2606.06224 | [PDF](https://arxiv.org/pdf/2606.06224v1)

**作者:** Yanqing Luo `[一作]` (Berlin Institute for the Foundations of Learning and Data), Mina Jamshidi Idaji `[通讯]` (Berlin Institute for the Foundations of Learning and Data)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5003471583)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Symb-xMIL 框架，在多实例学习（MIL）模型的后置解释中使用符号化逻辑规则，对模型决策过程进行高阶特征交互分析，并构建符号表示空间以实现样本级和队列级解释。

**💡 创新点**

创新点在于：① 将逻辑规则与 MIL 子袋预测直接对齐，捕捉实例集合间的组合交互；② 通过符号表示空间将每个样本映射到规则相似度向量，首次实现可比较的全局解释；③ 在真实病理任务中揭示隐藏错误（如 Clever Hans）并发现有临床意义的亚组。

**🔧 技术方法**

采用 Symbolic XAI 的查询生成与相关性度量，利用 TransMIL 等注意力/Transformer MIL 模型，对子袋进行预测；随后通过聚类、对比实验和生存分析验证结果。

**📊 数据集**

使用的数据集包括：合成 MNIST‑MIL（验证规则恢复）、Camelyon16（乳腺淋巴结肿瘤检测）、TCGA‑HNSCC（HPV 预测与生存分层）。

**📈 对比分析**

与传统热图解释和原始 Symbolic XAI 方法比较：在合成数据上能准确恢复真规则；在 Camelyon16 中发现并通过对比实验验证了模型的 Clever Hans 策略；在 TCGA‑HNSCC 中，符号分组的生存分层（logrank p≈0.04）略优于仅使用 HPV 状态（p≈0.06）。模型性能方面，Camelyon16 的 AUROC 约 0.995，TCGA‑HNSCC 的 AUROC 约 0.899。

**⚠️ 局限性**

局限性在于：① 依赖语义标签的质量和生成方式；② 对模型对扰动的鲁棒性要求较高，若模型易受噪声影响，解释可能失真；③ 查询空间的构建需要人工或启发式方法，可能影响可扩展性和覆盖所有可能规则。

---

## 530. Towards the Readability of LLM-Generated Codes through Multitask Representation Engineering

**arXiv ID:** 2606.06214 | [PDF](https://arxiv.org/pdf/2606.06214v1)

**作者:** Huifan Gao `[一作]` (Xiamen University), Weidi Sun `[通讯]` (Peking University)

**通讯引用:** 435 | [OpenAlex ID](https://openalex.org/A5028666046)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种多任务表征工程框架 MRepE，用于通过注入可调控的 steering 向量在 LLM 的隐藏层中改善生成代码的可读性。

**💡 创新点**

创新点在于：①提出了多维正交约束的联合主成分分析（MOC‑JPCA）来同时提取多种可读性度量（注释密度、命名规范、环形复杂度）的 steering 向量，避免相互干扰；②在理论层面给出了可读性提升的下界与正确性下降的上界，阐明二者之间的可控权衡。

**🔧 技术方法**

使用的技术包括：表征工程（RepE）实现隐藏层注入；MOC‑JPCA 算法（Stiefel manifold 迭代求解）提取正交 steering 向量；在推理时通过系数 c 控制向量强度；以及对模型行为的概率分析与理论证明。

**📊 数据集**

使用的数据集为：基于 MBPP 的扩展数据集 MBPP‑CR（评估三种可读性度量）和 MBPP‑CC（评估代码正确性），并在三大基础模型上进行实验。

**📈 对比分析**

比较方法：在 Deepseek_R1_14b、Qwen2.5coder_14b_Instruct、CodeLlama_13b_Instruct 三个 LLM 上分别注入 steering 向量，测量可读性相关查询的正答概率与代码正确性查询的正答概率。实验表明，可读性显著提升（正答概率上升至理论下界附近），而正确性仅轻微下降（符合理论上界），实现了可控的可读性‑正确性权衡。

**⚠️ 局限性**

局限性：①仅针对三种可读性指标，未覆盖更细粒度或其它代码质量属性；②实验聚焦于少数大型模型，缺乏对小模型或不同体系结构的泛化验证；③在真实工程项目中的可读性和可维护性评估仍待进一步验证；④理论分析假设较理想，实际部署可能受数据稀缺或模型差异影响。

---

## 531. Evaluating Agentic Configuration Repair for Computer Networks

**arXiv ID:** 2606.06212 | [PDF](https://arxiv.org/pdf/2606.06212v1)

**作者:** Rufat Asadli `[一作]` (ETH Zurich), Laurent Vanbever `[通讯]` (ETH Zurich)

**通讯引用:** 4061 | [OpenAlex ID](https://openalex.org/A5070110259)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在现有网络配置修复的基础上，作者设计并实现了一套最小化的代理式（agentic）LLM系统，结合动态上下文检索、迭代编辑与形式化验证工具，对数百个真实网络误配置场景进行自动修复，并在公开基准 Cornetto 上进行评测。

**💡 创新点**

创新点主要包括：① 将代理式框架与网络配置修复结合，首次系统性地在大规模网络场景下验证其优势；② 提出三大设计：动态上下文检索、可迭代的 patch‑apply 以及即时形式化验证回调，分别解决噪声过滤、单轮失误和语义漂移问题；③ 通过严格的消融实验揭示了验证反馈与上下文检索对修复效果与安全性的具体影响。

**🔧 技术方法**

技术手段：使用 ReAct‑style 代理与自定义工具（context‑retrieve、edit‑apply、verifier‑call），采用多种开源与闭源 LLM（GPT‑5 mini、GLM‑4.7、Qwen3.5‑9B、Gemma‑4‑E4B），结合 Batfish 等网络形式化验证器以及 Config2Spec 等规范映射工具；在每个修复迭代中动态调用这些工具实现自动推理。

**📊 数据集**

数据集：Cornetto 基准，包含 231 个误配置场景，来自 Topology Zoo 的真实拓扑（最多 754 节点、200k 配置行），覆盖 27 种故障类型，单场景最多 8 个并发故障。

**📈 对比分析**

对比方法：将代理式 LLM 与单轮（monolithic）LLM 在同一任务集上进行比较，使用 Fix Score（修复成功率）与 Regression Rate（安全性回归率）两个指标。实验结果表明，代理式方案平均提升约 12% 的 Fix Score、降低 17% 的 Regression Rate；对开源模型效果尤为显著，修复成功率提升 7 倍，回归率从 34.2% 降至 <6%。消融实验进一步验证了验证反馈和动态检索对性能的正负影响。

**⚠️ 局限性**

局限性：① 仅基于静态形式化验证，无法处理瞬态或性能相关的故障；② 代理与 LLM 之间的交互成本高，特别是多轮推理导致算力/Token 消耗显著；③ 目前缺乏更强的安全防护（如对已知不变式的预屏蔽）与人机协同审批流程；④ 在极大规模配置中，长上下文的有效性与一致性仍是挑战；⑤ 需要进一步研究 RL 或微调方式来提升中间步骤的学习与稳定性。

---

## 532. Unsupervised Pattern Analysis in Japanese Veterinary Toxicology: A Regulatory-Compliant Framework for Cross-Species Risk Assessment

**arXiv ID:** 2606.06207 | [PDF](https://arxiv.org/pdf/2606.06207v1)

**作者:** Yukiko Kawakami `[一作]` (Tohoku University), Matsumoto Kawahara `[通讯]` (Tohoku University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

通过将日本NVAL数据库的ADE报告按器官系统进行编码并调整种群偏差，构建无监督的监管集成框架，对跨物种药物不良事件进行模式发现与聚类分析。

**💡 创新点**

创新点在于将监管标准和物种特异性生物学融入特征设计与无监督学习，能够识别区域性、机制性毒性模式，而非仅预测。

**🔧 技术方法**

使用特征向量化、余弦相似度、K-means、层次聚类、UMAP降维等技术。

**📊 数据集**

数据集为4,120条高置信ADE报告，9,080个药物- ADE-物种组合，覆盖5种物种，来自日本NVAL Veterinary Drug Side Effects Database。

**📈 对比分析**

通过与MAFF监管分类和NVAL药物类别比较，使用内部评价如轮廓系数、聚类精度、ARI等指标，发现余弦相似度在稀疏数据上优于欧氏距离、皮尔逊相关等，聚类精度达87%，监管匹配率81%。

**⚠️ 局限性**

局限包括：数据受报告偏差限制、仅涵盖5种物种、未考虑时间序列变化、缺乏外部验证，且对稀有事件的发现依赖手工调参。

---

## 533. A Swarm Approach to Public Transit Using On-demand Routing in a Slime-Mold-Inspired Framework

**arXiv ID:** 2606.06189 | [PDF](https://arxiv.org/pdf/2606.06189v1)

**作者:** Lindsay Burke `[一作]` (New Jersey Institute of Technology), Petras Swissler `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 102 | [OpenAlex ID](https://openalex.org/A5034624024)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于分布式RAPID算法和动态转移的需求响应公共交通系统，并通过代理仿真评估其性能。

**💡 创新点**

创新点在于将类黏菌的分布式路径寻找算法与实时协作竞价机制相结合，形成无中心化的车辆调度与动态乘客转移方案。

**🔧 技术方法**

采用分布式算法（RAPID）、局部通信与动态竞价、代理仿真、以及动态转移机制实现车辆与乘客协同决策。

**📊 数据集**

使用从OpenStreetMap抓取的实际道路网络数据，构建郊区、城市和半农村三种场景进行仿真。

**📈 对比分析**

与传统固定路线系统对比，仿真显示在三种场景下乘客交付率分别提升28%、49%和101%，而乘客步行时间平均下降超过75%。

**⚠️ 局限性**

局限性包括仿真环境未覆盖真实交通拥堵与突发事件，参数设置对不同城市规模和需求模式的适应性尚未充分验证，且系统对极端需求波动的鲁棒性尚需进一步研究。

---

## 534. Learning to replenish: A hybrid deep reinforcement learning for dynamic inventory management in the pharmaceutical supply chains

**arXiv ID:** 2606.06201 | [PDF](https://arxiv.org/pdf/2606.06201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 535. Adversarial Attacks Already Tell the Answer: Directional Bias-Guided Test-time Defense for Vision-Language Models

**arXiv ID:** 2606.06186 | [PDF](https://arxiv.org/pdf/2606.06186v1)

**作者:** Liangsheng Liu `[一作]` (University of Science and Technology of China), Tianzhu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18466 | [OpenAlex ID](https://openalex.org/A5100648981)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种测试时防御框架 Directional Bias-guided Defense (DBD)，通过多种输入变换估计防御方向并根据 DB-score 进行特征重构，以提升 CLIP 等 VLM 的对抗鲁棒性。

**💡 创新点**

创新点在于发现对抗样本在多变换下呈现显著的方向偏倚，并利用该方向（防御方向）对特征进行线性修正，首次实现对抗样本的方向优先重构。

**🔧 技术方法**

使用了多模态 VLM CLIP、图像变换（空间、像素、频域）+熵过滤、方向偏倚（DB）得分、两流特征重构、线性特征平移等技术。

**📊 数据集**

实验使用了15个数据集，包括10个细粒度分类（Caltech101、Pets、Flower102、Cars、Aircraft、DTD、EuroSAT、UCF101、SUN397、Food101）和5个 ImageNet-OOD 变体（ImageNet、ImageNet-A、ImageNet-V2、ImageNet-R、ImageNet-S）。

**📈 对比分析**

与对抗微调、提示调优、R-TPT、TTC 等多种基线对比，DBD 在所有模型和攻击设置下实现了 SOTA 对抗准确率，并且在多数情况下对抗准确率甚至超过干净准确率。

**⚠️ 局限性**

局限性包括需对输入做大量变换导致推理开销上升，且对抗方向估计依赖于熵过滤和 DB-score 阈值，可能在特定攻击或数据分布下失效。

---

## 536. Steering LLM Viewpoints through Fabricated Evidence Injection

**arXiv ID:** 2606.06244 | [PDF](https://arxiv.org/pdf/2606.06244v1)

**作者:** Xi Yang `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10751 | [OpenAlex ID](https://openalex.org/A5020880385)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在遇到带有伪权威标记的误导性证据时的信任倾向，提出并验证了名为Ghostwriter的两阶段攻击框架；

**💡 创新点**

首次识别并利用“authority bias”这一模型内在脆弱点，创新性地将误导性观点包装成看似客观、带有权威性的证据，再通过条件模板使目标模型在相关查询中无声内部化该观点；

**🔧 技术方法**

使用轻量级LLM（GPT‑4o‑mini或微调的Qwen‑2.5‑7B）进行声明重写，构造两阶段注入模板，利用JudgeLM评估观点支持度（VSScore），并结合安全分类器与后端检测策略（gpt‑oss‑safeguard）进行评估与防御；

**📊 数据集**

自构造的Hazardous Viewpoints Dataset（HVD）以及现有的BBQ和ToxiGen数据集用于实验与对比；

**📈 对比分析**

在未攻击基线、对齐模型、推理模型及带安全分类器的模型上进行对比实验；Ghostwriter显著提升VSScore，导致HCRate和NRRate大幅上升；针对不同平台的攻击通过率普遍高于80%；采用定制的gpt‑oss‑safeguard后，检测准确率超过80%，误报率低于2%；

**⚠️ 局限性**

仅能在第三方聚合器、内部修改或检索渠道实现注入，无法直接攻击官方主流服务；HVD数据集依赖LLM生成，可能带来偏见；检测策略依赖较强分类器，仍有漏报风险；

---

## 537. Tracing the Oracle: Improving Diffusion Timestep Scheduling for 3D CT Reconstruction

**arXiv ID:** 2606.06236 | [PDF](https://arxiv.org/pdf/2606.06236v1)

**作者:** Yujia Wu `[一作]` (University of Electronic Science and Technology of China), Zhaoqiang Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 541 | [OpenAlex ID](https://openalex.org/A5089855005)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种基于oracle轨迹的非均匀时间步调度框架 TrO，用于改进预训练扩散模型在三维CT重建中的采样效率与图像质量。

**💡 创新点**

其创新点在于先用高密度采样的轨迹作为参考oracle，利用噪声重用技术剔除随机误差，然后将时间步选取问题转化为离散最短路径求解，并通过动态规划得到全局最优的步长分布。

**🔧 技术方法**

采用扩散概率流ODE/ SDE理论、DDIM采样、CG数据一致性更新、噪声重用、动态规划以及总变分正则化等技术。

**📊 数据集**

在AAPM公开数据集的稀视角CT与有限角度CT任务上进行实验。

**📈 对比分析**

与Uniform-t、Uniform-λ、Quadratic、EDM、Cosine等固定调度方案以及DiffMBIR、Score-Med、MCG等高NFE解算器对比，TrO在NFE≤10时均显著提升PSNR/SSIM，且在低预算下性能差距最小。

**⚠️ 局限性**

局限性在于需要先生成高密度oracle轨迹作为离线参考，且调度方案对不同测量算子及硬件仍需进一步验证，未展示在真实临床数据上的泛化能力。

---

## 538. Learning Emotion-discriminative Representations for Zero-Shot Cross-lingual Speech Emotion Recognition

**arXiv ID:** 2606.06200 | [PDF](https://arxiv.org/pdf/2606.06200v1)

**作者:** Jinyi Mi `[一作]` (Nagoya University), Tomoki Toda `[通讯]` (Nagoya University)

**通讯引用:** 11943 | [OpenAlex ID](https://openalex.org/A5078330211)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了零样本跨语言语音情感识别框架，结合监督对比学习和说话人对抗学习实现情感表示对齐。

**💡 创新点**

创新点在于显式对跨语言情感对齐进行语言加权的监督对比，并通过说话人对抗学习去除说话人信息，获得语言不变且情感可区分的特征。

**🔧 技术方法**

采用 wav2vec 2.0 预训练特征提取器，监督对比学习（Supervised Contrastive Learning）与说话人对抗学习（GRL+分类器），以及低秩适配器、瓶颈适配器等微调技术。

**📊 数据集**

使用了 MELD、ESD、EMO‑DB、CaFE、URDU 等多语言情感语音数据集，覆盖英语、汉语、德语、法语、乌尔都语等五种语言。

**📈 对比分析**

通过与两种基线和上限系统对比，在九个零样本跨语言设定下，平均 UAR 约 82.3% / F1 约 81.96%，比基线提升约 9%/9%，逼近上限性能。

**⚠️ 局限性**

局限性包括仍依赖源语言和辅助语言的标注，跨语言对齐仅靠监督对比；未利用多模态信息，且对极低资源语言的泛化尚不充分。

---

## 539. Trust-Aware Predictive Emissions Monitoring for Gas Turbine Fleets with Limited Labelled Data

**arXiv ID:** 2606.06156 | [PDF](https://arxiv.org/pdf/2606.06156v1)

**作者:** Rebecca Potts `[一作]` (University of Aberdeen), Georgios Leontidis `[通讯]` (UiT Arctic University of Norway)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种面向机组级燃气轮机排放预测的可信度感知概率预测框架，在只有一台机组有排放标签的条件下实现对其余机组的排放估计。

**💡 创新点**

创新点在于将学习到的置信度、深度集成的不确定度、特征空间距离、辅助特征预测与机组运行范围诊断等多种信号结合，构建可校准的样本级可信度分数，既能评估预测误差，也能在无标签条件下判断可信度。

**🔧 技术方法**

采用多头LSTM概率模型进行排放和辅助特征预测，并通过深度集成估计预测/知识不确定度；加入置信度头学习置信度；利用马氏距离和XGBoost回归构建可信度模型；最后通过阈值校准生成可信度等级。

**📊 数据集**

实验使用57台燃气轮机的多变量时序数据，其中107,936条序列来自单台带排放标签机组，1,249,310条序列来自其余无标签机组。

**📈 对比分析**

与基线XGBoost对比，概率LSTM在标记集上的MAE为0.202，RMSE 0.303，优于XGBoost MAE 0.313；通过置信度筛选前10%高置信度样本，MAE降至0.070，显示置信度与误差高度相关。

**⚠️ 局限性**

主要限制包括：仅有一台有排放标签的机组，可信度阈值依赖该机组分布；在无标签机组无法直接验证误差；仅在概率LSTM上验证，未检验其他模型；分布漂移情况下置信度单独使用效果有限。

---

## 540. Learning to Contest: Decentralized Robust Fairness in Cooperative MARL via Cross-Attention

**arXiv ID:** 2606.06162 | [PDF](https://arxiv.org/pdf/2606.06162v1)

**作者:** Can Savcı `[一作]` `[通讯]`, Can Savcı

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

在分布式多智能体强化学习中提出通过分级竞争和交叉注意力网络实现鲁棒公平性，抵御自利代理的免费搭车行为。

**💡 创新点**

创新点在于：①证明分级竞争可恢复竞争杠杆；②设计CAN交叉注意力政策以推断免费搭车数量并按比例反应；③采用PSRO联盟训练实现稳健。

**🔧 技术方法**

使用交叉注意力网络、REINFORCE策略梯度、PSRO联盟训练以及公平性/效率度量。

**📊 数据集**

实验数据集为自定义的分级竞争环境、拥堵、多服务器、赌注以及Matthew规则等离散多智能体环境，无公开真实数据集。

**📈 对比分析**

与公平-MARL学习者（GGF、FEN、SOTO）、基线（全投标、全放弃）以及集中式需求分配oracle对比，CAN在所有竞争度下实现ρ≈1.2–1.5、效率≈0.83–1.0，显著优于其他方法。

**⚠️ 局限性**

局限性：仅适用于存在竞争杠杆的规则，all-or-nothing或winner-take-all情况下失效；对大规模团队的零样本转移在高竞争度下表现下降；需要昂贵的联盟训练；未在连续或空间环境验证。

---

## 541. Where does Absolute Position come from in decoder-only Transformers?

**arXiv ID:** 2606.06160 | [PDF](https://arxiv.org/pdf/2606.06160v1)

**作者:** Valeria Ruscio `[一作]` (Intuition Machines), Fabrizio Silvestri `[通讯]` (Sapienza University of Rome)

**通讯引用:** 5277 | [OpenAlex ID](https://openalex.org/A5044165871)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对RoPE‑trained transformer中的绝对位置泄漏进行系统性分析，识别出泄漏的两大来源——因果mask的softmax正则化和残差流中的位置信息，并通过多种干预和统计回归验证。

**💡 创新点**

首次将绝对位置泄漏拆解为两个架构成分，并揭示残差流通过位置0的闭环动力学将绝对位置信息传递给sink‑reading heads；同时量化不同架构（NTK‑scaled RoPE、滑窗注意力、标准RoPE）以及模型规模对泄漏组成的影响。

**🔧 技术方法**

使用RoPE、因果mask、残差流、softmax归一化、位置0闭环动力学分析；进行Identity‑RoPE、scrambled‑RoPE、bidirectional mask、embedding替换等干预；通过ΔR²回归、层级与位置-bin分块、长度缩放、sink‑reading head抑制等多维度统计方法。

**📊 数据集**

在64个Wikipedia片段的平衡语料上进行实验，tokenized为256长度，使用16种不同长度（56–768）进行长度缩放；对Llama‑3.2‑1B、Qwen‑2.5‑3B、Mistral‑7B‑v0.3等模型进行评估。

**📈 对比分析**

通过ΔR²对比基线（相对偏移基准）并使用FDR校正评估显著性；对不同架构与规模的泄漏贡献进行定量比较；干预实验表明残差流贡献可由embedding替换降低40%，其余60%分布式在多方向；整体泄漏量在三种模型中均显著且可追溯。

**⚠️ 局限性**

未能直接评估泄漏对下游任务性能的影响；对双向训练模型的泄漏分解尚未验证；残差流中剩余60%的来源未完全归因（可能涉及层归一化、旋转写入等）；因果mask干预在训练后模型中的可迁移性尚待进一步研究。

---

## 542. Amortizing Federated Adaptation: Hypernetwork Driven LoRA for Personalized Foundation Models

**arXiv ID:** 2606.06154 | [PDF](https://arxiv.org/pdf/2606.06154v1)

**作者:** Sunny Gupta `[一作]` (Indian Institute of Technology), Amit Sethi `[通讯]` (Indian Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HyperLoRA 框架，用 hypernetwork 生成客户端 LoRA 初始化，并在服务器端采用学习的 product‑space 聚合器和残差校正器，解决传统 Federated LoRA 的聚合偏差和初始化滞后问题。

**💡 创新点**

创新点包括：① 基于分布签名的 amortized 客户端适配器，使每轮训练从无信息的随机初始化跳过为预热；② 学习的 product‑space 合成器直接在低秩乘积空间聚合更新，消除因独立平均导致的结构性聚合偏差；③ 残差校正模块补偿低秩近似误差，提升在严重非 IID 条件下的鲁棒性。

**🔧 技术方法**

使用技术：LoRA（低秩适配）、Hypernetwork（条件生成）、Federated Learning、meta‑learning（联合训练三大模块）、product‑space 采样与残差校正、SVD 对比、基于签名的分布特征提取。

**📊 数据集**

实验数据集：DomainNet（6 个视觉域）和 NICO++（6 个语境域），在 ViT‑B/16 与 MLP‑Mixer 两个 backbone 上进行验证。

**📈 对比分析**

与 FedIT、FFA‑LoRA、FLoRA、FlexLoRA、LoRA‑FAIR 等基线进行对比；在 Feature‑nonIID 与 Feature+Label‑nonIID 两种更严苛的分布异质性设置下，HyperLoRA 在所有组合上均优于基线，尤其在 NICO+++MLP‑Mixer 提升 +2.30%；在本地迭代仅 20 次时已超越 LoRA‑FAIR 的 100 次迭代性能，显著降低 80% 客户端计算量。

**⚠️ 局限性**

局限性：① 目前仅在视觉任务上验证，语言/多模态领域仍待扩展；② 需要对分布签名进行预训练，若客户端数据分布极度稀疏或不稳定可能影响效果；③ 额外的 hyper‑network 与聚合器训练引入一定的元学习开销；④ 对极大规模客户端群（数千甚至数万）在通信与计算可扩展性尚未系统评估。

---

## 543. Benchmarking Open-Source Layout Detection Models for Data Snapshot Extraction from Institutional Documents

**arXiv ID:** 2606.06242 | [PDF](https://arxiv.org/pdf/2606.06242v1)

**作者:** AJ Carl P. Dy `[一作]` (World Bank), Aivin V. Solatorio `[通讯]` (World Bank)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5020026033)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了针对机构文档（人道主义报告、政策研究论文、项目评估文件）的数据快照提取基准，并评估了多款开源布局检测模型在该任务上的表现。

**💡 创新点**

创新点在于：①提出“数据快照”这一关注可复用分析信息的视觉对象概念；②设计了同时评估检测精度与空间提取完整度的评价框架；③公开了包含真实机构文档的标注数据集、代码与评测工具。

**🔧 技术方法**

使用了Transformer‑基的TF‑ID‑Large、YOLO‑基的DocLayout‑YOLO、YOLOv11、YOLOv26 等布局检测模型，结合Label Studio 进行人工标注；在评估时采用 IoU、Area Recall、Area Precision 等指标。

**📊 数据集**

数据集来源于 UNHCR/ReliefWeb 人道主义报告、World Bank Policy Research Working Papers（PRWP）以及 Refugee Project Appraisal Documents（PADs），共 476 份 PDF，标注 3,908 个数据快照。

**📈 对比分析**

对比四个模型，YOLO‑基模型在检测召回率方面更优（表中 0.80–0.89 之间），但 TF‑ID‑Large 在空间提取（IoU、Area Recall）上表现更好；整体上各模型在机构文档上相较学术基准出现显著性能下滑，揭示了泛化与语义差距。

**⚠️ 局限性**

局限性包括：①标注主观性大，尤其是复合视觉艺术与上下文边界定义；②仅用矩形框表示，难以精确捕捉复杂布局；③评估聚焦定位与空间质量，未覆盖下游信息提取或推理准确性；④数据集仅覆盖人道主义与公共部门文档，未涵盖其他行业多样性；⑤研究聚焦基准构建，未提出专门的模型改进方案。

---

## 544. TOKI: A Bitemporal Operator Algebra for Contradiction Resolution in LLM-Agent Persistent Memory

**arXiv ID:** 2606.06240 | [PDF](https://arxiv.org/pdf/2606.06240v1)

**作者:** Ziming Wang `[一作]` (Hong Kong University of Science and Technology), Ziming Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 473907 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于双时态算子代数的LLM代理持久化内存冲突解决框架，统一定义了四种生产策略并给出了完整的写时正确性契约。

**💡 创新点**

创新点在于把四种无类型的冲突解决策略转化为带隔离前置条件和K-半环系数的算子，形成可验证的三轴（隔离、模式、溯源）安全保证，并证明键值日志纪律对重放一致性是必要的。

**🔧 技术方法**

技术包括：双时态事实模型、Bitemporal SQL:2011 视图、K-半环溯源、基于键值日志的隔离门控、双行（current+audit）模式和Python实现的四个纯算子。

**📊 数据集**

使用的数据集包括 LoCoMo、LongMemEval‑S、MultiTQ、STALE 以及内部的 1,444 条可回答事实问答池，覆盖单跳、时间推理和开放域事实三类。

**📈 对比分析**

对比方法是构建八个现有代理内存系统与参考实现的“判定矩阵”，通过控制实验验证每种防御的效果；实验表明审计行防御使 LoCoMo 准确率提升 0.86，整体延迟保持次线性，内存占用在 10⁵ 条事实时 3.9–4.2 ms。

**⚠️ 局限性**

局限性：实验仅在单节点、单进程环境下完成，无法覆盖分布式事务；键值日志必要性证明仅适用于有界非确定性推理机；防御粒度有限，跨部署一致性依赖解码器不确定性分析。

---

## 545. A Machine Learning-Based Framework for Discovering Huntington's Disease Stages: Integrating Graph Representation Learning and clustering to Uncover Progression Dynamics in Longitudinal Enroll-HD Dataset

**arXiv ID:** 2606.06196 | [PDF](https://arxiv.org/pdf/2606.06196v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 546. Design a Reliable LLM-Integrated Interface for Mortality Forecasting

**arXiv ID:** 2606.06235 | [PDF](https://arxiv.org/pdf/2606.06235v1)

**作者:** Thi Kim Ngan Nguyen `[一作]` `[通讯]` (Curtin University), Thi Kim Ngan Nguyen (Curtin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计并实现了基于 CoMoMo 的死亡率预测管线，将本地 LLM 作为自然语言接口，将用户请求转化为可执行的 JSON 配置，使用 R 脚本完成模型拟合、滚动-origin 评估和多步预测，并通过 Streamlit 前端展示结果。

**💡 创新点**

创新点在于将 LLM 限定为受控的编排层而非预测引擎，使非技术用户可通过自然语言启动完整的死亡率预测流程，同时在同一可审计管线中集成多模型组合和滚动评估，保证透明度和可重复性。

**🔧 技术方法**

使用的技术包括：R 语言与 CoMoMo 包进行模型拟合与组合；Python、FastAPI 与 Streamlit 构建接口和前端；本地 Llama LLM 进行自然语言到 JSON 的转换；滚动-origin MSE 评估；JSON 验证与配置合并。

**📊 数据集**

数据集为 Human Mortality Database（HMD）中的英国/威尔士男性人口（1960-2016 年龄 50–89）

**📈 对比分析**

通过滚动-origin 平均 MSE 和模型排名进行基线复制，比较单模型与堆叠回归、贝叶斯平均、模型置信集等组合方法；结果显示组合方法在大多数预测 horizon 上显著降低 MSE，且 LLM 生成的配置与基线结果保持一致，验证了准确性。

**⚠️ 局限性**

局限性包括：计算资源消耗大，滚动评估和多模型组合耗时；本地 LLM 对复杂自然语言理解有限导致验证失败；实验仅在单一人口群体上验证，缺乏跨国家、性别的泛化能力。

---

## 547. FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition

**arXiv ID:** 2606.06211 | [PDF](https://arxiv.org/pdf/2606.06211v1)

**作者:** Fernando López `[一作]` (Telefónica Innovación Digital), Jordi Luque `[通讯]` (Telefónica Innovación Digital)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

使用 Feature-wise Linear Modulation (FiLM) 在冻结的 SpeechLLM 编码器中注入 x‑vector 说话人嵌入，以在不修改基础模型权重的前提下，对神经病理性说话进行自适应，提高病理语音识别性能并保持对健康语音的表现。

**💡 创新点**

创新点在于：① 将说话人信息通过 FiLM 生成器在每个 Transformer 层后进行线性调制；② 对健康语音使用零嵌入实现身份映射，从而保证基线模型不受影响；③ 仅更新极少量参数（FiLM 生成器和 x‑vector 提取器），实现轻量级、可迁移的自适应。

**🔧 技术方法**

采用的技术包括：FiLM 生成器（两层 MLP + 归一化 + 门控），x‑vector 说话人嵌入网络（SiAmResNet34），Voxtral‑Mini SpeechLLM，LoRA 与全微调作为对比基线，以及基于规则的后处理（字符级、词级、短语级去重）。

**📊 数据集**

实验数据集：English 病理语音 corpus TORGO（15 位说话人，约 13.68 小时）和 Spanish 病理语音 corpus NeuroVoz（111 位说话人，约 2.31 小时）；为 Spanish 语料补充 Common Voice v24.0 读音样本（约 7.3 小时）。

**📈 对比分析**

与 FFT（全微调）、F‑LoRA、EFT、E‑LoRA 等方法比较。结果显示：在 TORGO 上，FiLM 方案原始 WER 23.24%→16.36%（后处理），与全微调相当；在 NeuroVoz 上，原始 WER 6.57% 与全微调相近。MCQA 性别识别上，FiLM 性能 60.7%（仅 4.2% 低于 EFT 的 64.9%），但大幅优于 F‑LoRA（仅 8.4%）。

**⚠️ 局限性**

局限性：① 需要先验判断输入是否为病理语音以实现零嵌入；② 仅在性别、年龄的 MCQA 任务中评估语音推理能力；③ 对噪声、短时语音的鲁棒性尚未验证；④ FiLM 方案对非常短的说话人嵌入可能不稳定。

---

## 548. Dense Contexts Are Hard Contexts: Lexical Density Limits Effective Context in LLMs

**arXiv ID:** 2606.06203 | [PDF](https://arxiv.org/pdf/2606.06203v1)

**作者:** Giovanni Dettori `[一作]` (Politecnico di Torino), Marco Mellia `[通讯]` (Politecnico di Torino)

**通讯引用:** 10550 | [OpenAlex ID](https://openalex.org/A5060087704)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究词汇密度对大型语言模型（LLM）长上下文性能的影响，量化并验证高词汇密度会显著缩减模型可用的有效上下文窗口。

**💡 创新点**

首次将词汇密度作为长上下文评估的第三维度，并提出两种新的基准（Scene‑Rules 与 WordChecker），系统展示词汇密度导致检索准确率急剧下降，并与长度、位置共同作用。

**🔧 技术方法**

使用 Moving‑Average Type‑Token Ratio (MATTR) 测量词汇密度，构造三种“找针”基准，在 9B–685B 规模的开源 LLM 上执行约12k‑token 长度的检索实验，并通过“synthetic sparsification”方法控制密度。

**📊 数据集**

基准数据集包括：MK‑NIAH（标准 Needle‑in‑a‑Haystack）、Scene‑Rules（从 Moral Stories 数据生成的情景‑规则对）、WordChecker（由短语与词表生成的词检索任务）。

**📈 对比分析**

通过保持长度、位置、任务不变，逐步降低词汇密度，比较检索准确率；在高密度情形下，模型准确率从接近 100% 降至 60% 或更低；降低密度后性能恢复，显示显著性能波动。

**⚠️ 局限性**

限制包括：仅能通过重复抑制来降低密度，无法合成更高密度场景；MATTR 仅衡量词汇多样性，未完整反映信息理论密度；实验基于合成基准，缺乏对真实世界输入的验证；未提出缓解高密度下性能下降的具体方法。

---

## 549. Ouvia: A User-centered Framework for Measuring Usability of Speech Translation in Real-World Communication Scenarios

**arXiv ID:** 2606.06177 | [PDF](https://arxiv.org/pdf/2606.06177v1)

**作者:** Giuseppe Attanasio `[一作]` (Instituto de Telecomunicações), André F. T. Martins `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实施了多阶段的用户研究框架（Ouvia），用于评估英语到葡萄牙语语音翻译在真实一对一沟通情境中的可用性。

**💡 创新点**

创新点包括：① 将评估从实验室脱离，聚焦用户感知的可用性；② 在一对一交互中结合翻译输出、接收者理解问题、验证者评分和发送者主观可用性评估；③ 发现基于问答的质量评估（QA Score）比传统整体质量指标更能预测可用性。

**🔧 技术方法**

使用了四种前沿语音翻译系统（Phi‑4 Multimodal、Voxtral Small、DeSTA2 直接ST 与 Whisper+Tower+ 脚本化 ST），自动评估指标（COMET、XCOMET、MetricX 等），以及 Gemini 2.5 Pro、Qwen 3 32B 等大型语言模型来生成对话起始语和问题；同时构建了自定义 Web 平台进行数据收集。

**📊 数据集**

收集了 14.6 小时的提示式语音录音、13.8K+ 人工 QA 评注、1.7K+ 翻译质量分数，并使用 MED‑MT 医疗对话、BConTrasT 客服对话以及 LLM 生成的模拟对话，最终得到 300 个对话起始语、1,738 交互记录。

**📈 对比分析**

通过线性混合模型和 OLS 预测用户可用性，比较各种质量指标的影响。传统整体质量指标（如 COMET、MetricX）与可用性的相关系数在 0.4–0.5 之间；QA Score 的相关系数为 0.63，且在高质量区间保持显著。提出 QA Score ≥ 0.91 对应可用性 ≥ 4 的部署阈值；不同 ST 模型中 Tower+ 与 Voxtral 得分最高，Phi‑4 与 DeSTA2 在可用性上落后。

**⚠️ 局限性**

局限性包括：仅研究英葡一对语对，难以推广到其他语对；对话起始语为剧本化、清晰朗读，缺乏自然流畅语音；验证者为众包非专业译者；问答评估依赖已有原始文本，迁移性有限；数据发布需限制语音克隆风险；性别划分为二元，缺少非二元群体。

---

## 550. RQUL-UIE: Revitalizing Quality-Unstable Labels for Underwater Image Enhancement via In-Dataset Self-Supervision

**arXiv ID:** 2606.06176 | [PDF](https://arxiv.org/pdf/2606.06176v1)

**作者:** Haochen Hu `[一作]` (Hong Kong Polytechnic University), Bing Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 19329 | [OpenAlex ID](https://openalex.org/A5100382568)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于扩散模型的自监督框架，利用标签质量分层去噪并结合 Fourier 细节恢复网络来提升水下图像。

**💡 创新点**

创新点在于用预训练扩散模型评估标签质量并按质量分配去噪步数，避免低质量标签干扰；同时设计了频域细节恢复模块。

**🔧 技术方法**

使用 Stable Diffusion 2.1 作为扩散基座，配合 LLSD（级别去噪）和 TRFDM（频域细节恢复）两大模块。

**📊 数据集**

在公开的 UIEB、LSUI、EUVP 等水下图像数据集上训练和评估。

**📈 对比分析**

与 12 种最新 SOTA 方法在四个学习型指标（MUSIQ、Uranker、TwiceMix）和 UIQM 进行比较，平均排名 1.83，显著优于对手。

**⚠️ 局限性**

主要局限是推理速度相对较慢，且对不同海域的泛化能力及实时应用仍需进一步验证。

---

## 551. ProSarc: Prosody-Aware Sarcasm Recognition Framework via Temporal Prosodic Incongruity

**arXiv ID:** 2606.06168 | [PDF](https://arxiv.org/pdf/2606.06168v1)

**作者:** Prathamjyot Singh `[一作]` (Thapar Institute of Engineering and Technology), Jasmeet Singh `[通讯]` (Thapar Institute of Engineering and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种仅使用音频的框架ProSarc，通过建模局部语调动态与整体情绪基线之间的时间性语调不一致性来检测讽刺。

**💡 创新点**

创新点在于：① 以可解释的“语调不一致性”分数对讽刺进行显式建模；② 通过注意力加权的弱监督机制实现无标注的讽刺起始点定位；③ 采用MC dropout提供不确定性估计；④ 通过双路径编码（全局情绪+时间语调）实现对情绪基线与细粒度语调的融合。

**🔧 技术方法**

技术包括：两路编码器（全局情绪MLP、基于SSL的音频编码+BiLSTM+多头注意力），语调不一致性分析器（MLP+sigmoid），注意力加权池化，MC dropout不确定性估计，以及基于注意力的单一时序起始点预测。

**📊 数据集**

使用四个数据集：MUStARD、MUStARD++（脚本化英文对话）、PodSarc（自发播客语音）和MuSaG（德语YouTube视频）。

**📈 对比分析**

与传统基线（OpenSMILE+SVM、各类SSL编码器）及先前最优音频方法相比，ProSarc在MUStARD++上F1提升至75.3%（相较先前最佳64-66%），在PodSarc和MuSaG上也分别获得62.9%和65.6%的F1，且十次随机种子实验Wilcoxon p=0.002、Cohen’s d=1.51验证显著性。

**⚠️ 局限性**

局限包括：仅依赖音频，无法捕捉视觉与语义线索；单一起始点预测无法处理多次讽刺；不确定性仅为相对估计而非校准概率；在自发语料与跨语言数据上性能受限，且存在不可逾越的视觉信息门槛。

---

## 552. On the training of physics-informed neural operators for solving parametric partial differential equations

**arXiv ID:** 2606.06164 | [PDF](https://arxiv.org/pdf/2606.06164v1)

**作者:** Nanxi Chen `[一作]` (Tongji University), Rujin Ma `[通讯]` (Tongji University)

**通讯引用:** 1740 | [OpenAlex ID](https://openalex.org/A5089037577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对物理信息神经算子（PINO）的训练进行系统的实证研究，比较三种主流算子结构（DeepONet、FNO、CViT）在五个典型偏微分方程基准上的表现；同时探索并验证多种训练技术（GradNorm、因果加权、SOAP优化器、FiLM时间条件、周期性嵌入等）与自由采样碰撞点对训练稳定性和精度的影响。

**💡 创新点**

首次将PINN的训练技巧与算子学习相结合，构建了一个完整且可复现的训练管线；在该管线中引入了梯度归一化与因果加权的联合权重策略，采用SOAP二阶预处理优化器，并将自由碰撞点采样与参数空间随机化相结合，从而显著提升了物理信息算子的泛化与数值稳定性。

**🔧 技术方法**

使用基于Transformer的CViT算子架构（编码器+交叉注意力解码器），并对DeepONet与FNO做对照；梯度归一化（GradNorm）与因果加权；SOAP优化器；FiLM时间条件；周期性坐标嵌入；训练热身期；拉丁方采样等。

**📊 数据集**

五个基准 PDE 数据集：二维 Burgers 方程、二维波动方程、线性化浅水方程、冰融化相变方程、旋转箱流；所有基准均采用随机初始场（高斯随机场或椭圆相界面）和参数采样，生成大量仿真轨迹以评估模型。

**📈 对比分析**

通过在 100 条测试轨迹上计算相对 L2 误差进行比较；在所有基准中，PI‑CViT 的平均误差始终最低（如 Burgers 0.78%、Wave 2.46%、Shallow Water 4.29%、Ice Melting 1.87%、Lid‑Driven Cavity 6.46%），显著优于 PI‑DeepONet 与 PI‑FNO；纯物理信息训练在大多数数据规模下可与或超过纯监督训练的表现。

**⚠️ 局限性**

局限性包括：在复杂边界或非周期域（如旋转箱流）时 FNO 失效；对数据与物理损失的联合训练仍易产生梯度冲突，尚缺乏系统的冲突缓解策略；对三维、非结构化网格、复杂几何的推广需要更具几何感知的架构；在具有刚性界面或强非线性源项的方程（如相变）中，点值残差可能不足，需要更高级的变分或能量约束。

---

## 553. ITP-STDP: An Intrinsic-Timing Power-of-Two Learning Engine for On-Chip SNN Training

**arXiv ID:** 2606.06159 | [PDF](https://arxiv.org/pdf/2606.06159v1)

**作者:** Haihang Xia `[一作]` (University of Sheffield), Tiantai Deng `[通讯]` (Donghai Laboratory)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了基于二进制指数近似的Intrinsic‑Timing Power‑of‑Two STDP (ITP‑STDP) 算法及其可在芯片上实现的学习引擎，并在多种网络规模与数据集上验证其训练效果；

**💡 创新点**

创新点在于将STDP的指数运算替换为基于二进制位移的base‑2指数近似，消除浮点运算并通过时间常数补偿实现与传统STDP相同的生物学可塑性，同时硬件实现仅需shift寄存器和极少的算术单元，显著降低资源占用；

**🔧 技术方法**

主要技术包括：基于shift寄存器的脉冲历史存储、LLSMu近似乘法器的LIF神经元模型、FPGA/28‑nm ASIC实现、mean‑field突触漂移模型分析；

**📊 数据集**

使用了MNIST、Fashion‑MNIST与工业机车故障诊断（电流/磁通）三组数据集；

**📈 对比分析**

与多种现有STDP实现（C‑STDP、P‑STDP、R‑STDP、t‑STDP 等）对比，FPGA上能耗提升4.5×–219.8×、频率提升2.6×–11.5×、ASIC面积缩减至1–3%，整体性能显著优于前沿方案；

**⚠️ 局限性**

局限性：在不做时间常数补偿时存在约9–10% 的权重更新误差；对极大规模网络的可扩展性与长期稳定性仍待进一步验证。

---

## 554. GRAMformer: Any-Order Modality Interactions via Volumetric Multimodal Cross-Attention

**arXiv ID:** 2606.06249 | [PDF](https://arxiv.org/pdf/2606.06249v1)

**作者:** Giordano Cicchetti `[一作]` (Sapienza University of Rome), Danilo Comminiello `[通讯]` (Sapienza University of Rome)

**通讯引用:** 2855 | [OpenAlex ID](https://openalex.org/A5019647783)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Volumetric Multimodal cross‑Attention（VMA）机制，并将其嵌入到新的轻量级Transformer架构GRAMformer中，用于处理任意数量的模态；

**💡 创新点**

创新点在于用并行体积作为注意力分数，直接建模多模态的联合几何关系，突破传统基于点积相似度的二阶限制，实现任何阶模态交互；

**🔧 技术方法**

技术上结合了Transformer编码器、VMA跨模态注意力、多头扩展、门控机制与点积正则化，并使用预训练的BERT/ RoBERTa、Data2Vec等模型作为模态编码器；

**📊 数据集**

实验数据集包括MOSI、MOSEI、UR‑FUNNY、MUsTARD、MuJoCo Push、Vision & Touch等多模态情感、幽默、机器人等基准；

**📈 对比分析**

通过与TFN、LMF、MulT、MAG‑BERT、MMML、Self‑MM等现有基线对比，GRAMformer在多分类与回归指标上均取得最高或接近最高的准确率/ F1/ MAE 等，同时参数量更少、内存占用更低；

**⚠️ 局限性**

限制在于需要模态序列对齐，且每组键向量数目不能超过投影维度；在极高模态数或无对齐情形下性能可能下降；计算体积时仍需注意数值稳定性和门控参数的微调敏感性。

---

## 555. MPCoT: Reward-Guided Multi-Path Latent Reasoning for Test-Time Scalable Vision-Language-Action

**arXiv ID:** 2606.06245 | [PDF](https://arxiv.org/pdf/2606.06245v1)

**作者:** Boyang Zhang `[一作]` (Boston University), Lianlei Shan `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多路径潜在推理框架MPCoT，在视觉-语言-动作（VLA）策略中通过深度K和宽度M实现可配置的计算量，保持原有动作接口且不产生推理token；

**💡 创新点**

核心创新在于将多路径潜在推理与奖励引导的路径偏好学习相结合，在训练阶段使用专家一致性、进展评估和成功反馈来指导路径评分器，提升长期决策质量；

**🔧 技术方法**

使用了OpenVLA-OFT骨干、共享权重的残差迭代细化模块、轻量级路径评分器、以及训练时的世界模型/VLM进展评估器；

**📊 数据集**

在LIBERO和CALVIN两个标准VLA基准上进行评估，涵盖多任务和长序列执行；

**📈 对比分析**

与现有基线（如OpenVLA-OFT、文本CoT、VLA-Adapter等）比较，MPCoT在LIBERO平均成功率从96.8%提升至98.9%，在CALVIN长序列（5步）成功率从72.9%提升至89.4%，且仅增加约38ms的延迟，远低于文本CoT的110–160ms；

**⚠️ 局限性**

局限性包括仅在仿真基准上验证，未考虑真实机器人中的硬件延迟、传感噪声和接触失效；此外训练时依赖世界模型/VLM评估器，但该评估器在推理阶段不被调用；

---

## 556. Generative Criticality in Large Language Model Temperature Scaling

**arXiv ID:** 2606.06238 | [PDF](https://arxiv.org/pdf/2606.06238v1)

**作者:** Huajian Ruan `[一作]` (South China Normal University), Lingxiao Wang `[通讯]` (University of Tokyo)

**通讯引用:** 4937 | [OpenAlex ID](https://openalex.org/A5100779977)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文基于统计物理视角，将LLM生成文本的词嵌入视作一维链上的连续自旋，定义易感度和序参，系统探究软max温度对生成文本结构的影响，发现温度接近临界点时易感度峰值、序参突变以及内在维度最小化等相变特征；

**💡 创新点**

创新点在于首次将LLM文本生成映射到统计场理论框架，提出易感度和序参等物理量，并通过温度扫描、PCA与TwoNN内在维度验证多重指标一致的相变现象，为量化LLM输出的集体结构提供新工具；

**🔧 技术方法**

主要技术包括统计场理论（易感度、序参定义）、TwoNN内在维度估计、PCA降维分析、软max温度控制以及对Qwen3系列模型的批量生成实验；

**📊 数据集**

使用的数据集包括英文和中文维基百科作为提示，并结合笑话、诗歌、小说及无意义文本等多种提示类型；生成长度固定为300个token，样本量为1000个；

**📈 对比分析**

通过对不同模型规模（0.6B–32B）和不同提示类别的温度扫描，比较易感度曲线、临界指数γ≈0.1、序参幅值变化和内在维度最小化位置，结果显示所有指标均在同一临界温度附近出现特征峰值，且峰值随模型规模增大而提升，验证了所提出框架的有效性；

**⚠️ 局限性**

局限性在于生成过程仍为非平衡自回归，温度参数并非严格热力学温度；未对长程相关或重正化效应进行深入理论分析；实验仅覆盖Qwen3系列和有限提示类型，未来需扩展至更多模型与数据集以进一步验证。

---

## 557. SAM-Flow: Source-Anchored Masked Flow for Training-Free Image Editing

**arXiv ID:** 2606.06228 | [PDF](https://arxiv.org/pdf/2606.06228v1)

**作者:** Haowang Cui `[一作]` (Tianjin University), Jiaze Wang `[通讯]` (Tianjin University)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5066420328)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SAM-Flow，一种无训练的图像编辑框架，通过局部差分流与源锚定投影实现精确的局部编辑，显著降低背景泄漏；

**💡 创新点**

创新点在于将全局分布传输转为局部语义流编辑，结合探针图像和Token驱动的注意力定位可编辑区域，并引入时间可变软掩码与源锚定投影来平滑边界并保持背景；

**🔧 技术方法**

利用预训练的流匹配模型（Stable Diffusion 3、FLUX），差分流编辑、Token‑grounded注意力提取、软掩码、时间可变投影以及累计掩码等技术；

**📊 数据集**

在DIV2K基础的编辑基准上进行实验，包含1024×1024图像、源/目标提示及Token级注解，覆盖多种编辑任务；

**📈 对比分析**

与SDEdit、iRFDS、ODE Inversion、FlowEdit、RF Inversion、RF Edit、StableFlow等基线比较，采用CLIP‑T、DINO、PSNR、SSIM、LPIPS等指标；SAM‑Flow在语义一致性与背景保真度上实现了更优的权衡，用户研究显示其被偏好；

**⚠️ 局限性**

局限在于仍需探针图像和注意力估计，掩码生成可能对细小或极端变形的目标不够精确，导致边界不自然或编辑不足。

---

## 558. CLEAR: Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving

**arXiv ID:** 2606.06219 | [PDF](https://arxiv.org/pdf/2606.06219v1)

**作者:** Yining Xing `[一作]`, Jianqiang Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CLEAR 框架，将多步扩散采样替换为 VAE 隐空间中的单步条件漂移，结合视觉编码器 Drive-JEPA 与微调后的 Qwen 3.5 0.8B LLM，实现实时多模态路径规划。

**💡 创新点**

创新点在于单步条件漂移机制、LLM 驱动的自适应调度器（选择 α 与采样数）以及跨注意力评分器，将认知信息与生成过程紧密耦合；同时采用冻结视觉编码器与小型 LLM 仅作为语义特征提取。

**🔧 技术方法**

技术手段包括 Drive‑JEPA 视觉编码、VAE+PCA 生成框架、MLP‑Mixer 解码器、AdaLN 条件归一化、Qwen 3.5 0.8B 语义特征提取、Transformer‑based 自适应调度与跨注意力评分。

**📊 数据集**

使用 NAVSIM v1/ v2 benchmark 进行闭环评估，并利用 ReCogDrive 提供的 150k 驾驶 QA 对 Qwen 进行微调，130k 轨迹数据预训练 VAE，以及 10k 场景合成数据训练调度器与评分器。

**📈 对比分析**

与 Drive‑Suprim、Drive‑JEPA、iPad 等先进方法在 NAVSIM 上对比，CLEAR 在 PDMS（v1）取得 93.7 分，超过 Drive‑Suprim 93.5；在 v2 的 EPDMS 也得到 88.6，领先同类 ViT/L 方法；消融实验显示 LLM 评分器与自适应调度显著提升性能。

**⚠️ 局限性**

局限性在于调度器仅从离散 (α,N) 集合中选择，可能错过连续最优配置；多阶段训练流程复杂；在车道保持和交通灯合规等指标仍低于部分方法。

---

## 559. Hub-Aware Hybrid Search: Accelerating the Locally Aligned Ant Technique

**arXiv ID:** 2606.06198 | [PDF](https://arxiv.org/pdf/2606.06198v1)

**作者:** Simone Vilardi `[一作]` (University of Groningen), Kerstin Bunte `[通讯]` (University of Groningen)

**通讯引用:** 1444 | [OpenAlex ID](https://openalex.org/A5030298766)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Hub-Aware Hybrid Search (Hub-LAAT)，先检测并用贝叶斯高斯混合模型建模密集枢纽，再改进蚁群迁移概率，显著提升宇宙网络中线状结构的识别效率与鲁棒性。

**💡 创新点**

创新点在于两阶段改进：快速枢纽检测 + 似然模型替代密集区域，以及混合似然-信息素的蚁群迁移策略配合双跳与远程排斥机制，避免信息素过度聚集并有效导引蚂蚁探索微弱结构。

**🔧 技术方法**

技术包括局部PCA、马尔科夫链稳态估计、贝叶斯高斯混合模型、Friends-of-Friends 聚类、混合似然-信息素蚁群优化、双跳跳跃与远程排斥转移。

**📊 数据集**

使用合成已知真值的高维点云、50³ Mpc³/h 大规模宇宙 N‑body 仿真（约 2.8×10⁵ 颗粒）以及中等规模宇宙仿真数据集进行实验。

**📈 对比分析**

与原 LAAT 在相同蚁数、步骤和邻域半径下比较，在合成数据上密集枢纽识别从 88% 降至 5%，线状结构恢复提升至 72%；在大规模仿真中预处理减少约 40% 计算周期，信息素高值粒子分布更集中，整体性能显著提升。

**⚠️ 局限性**

局限性包括对枢纽阈值参数（ψ、η、b）高度敏感、需要人工调参；目前仅处理静态数据，缺乏时序演化建模；对极端噪声或非高斯分布的枢纽识别可能不足。

---

## 560. Improving Answer Extraction in Context-based Question Answering Systems Using LLMs

**arXiv ID:** 2606.06197 | [PDF](https://arxiv.org/pdf/2606.06197v1)

**作者:** Hafez Abdelghaffar `[一作]` (MSA University), Ali Hamdi `[通讯]` (MSA University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在SQuAD1.1数据集上对多种Transformer型大型语言模型进行微调，并在统一框架下比较它们在上下文问答任务中的表现。

**💡 创新点**

提出了一套统一的训练与评估流程，细粒度比较不同模型的微调效果，同时使用ROUGE-L、BLEU和BERTScore三项指标从词汇重叠和语义相似度两方面综合评估答案质量。

**🔧 技术方法**

使用预训练Transformer编码器，采用监督学习对起止位置进行微调；评估环节利用ROUGE-L、BLEU和BERTScore；实验中对11种模型在同一数据集上进行对比。

**📊 数据集**

Stanford Question Answering Dataset（SQuAD1.1），包含超过100k个上下文–问题–答案对，用于训练与验证。

**📈 对比分析**

在相同的预处理、训练集、验证集以及评估指标下，对11个模型进行基线与微调后的对比。微调后模型显著提升，最佳模型Roberta-base达到ROUGE-L 86.84%、BLEU 28.24%、BERTScore 95.38%，并且模型排名因微调而发生变化，容量更大的模型表现更优。

**⚠️ 局限性**

小模型虽然也能通过微调提升，但提升幅度有限；实验仅在SQuAD1.1上验证，缺乏跨领域或多模态的泛化能力；未引入检索增强或多模型融合等进一步提升方法。

---

## 561. The Tell-Tale Norm: $\ell_2$ Magnitude as a Signal for Reasoning Dynamics in Large Language Models

**arXiv ID:** 2606.06188 | [PDF](https://arxiv.org/pdf/2606.06188v1)

**作者:** Jinyang Zhang `[一作]` (Peking University), Yasha Wang `[通讯]` (Peking University)

**通讯引用:** 4992 | [OpenAlex ID](https://openalex.org/A5055336632)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型内部推理动态，发现隐藏状态的ℓ_2范数可作为推理强度的内在信号并利用该信号在推理时进行控制

**💡 创新点**

首次将隐藏状态ℓ_2范数与Sparse Autoencoder推理特征关联，理论证明ℓ_2范数上界/下界对应推理特征激活，并基于此提出无训练、无数据的推理增强技术

**🔧 技术方法**

使用Sparse Autoencoders (SAE) 作为解释工具、ℓ_2范数分析、因果干预、以及三种基于ℓ_2的推理控制方法（层级递归、状态驱动、结果排序）

**📊 数据集**

在多种大型模型（Qwen‑3系列、DeepSeek‑R1‑Distill‑Llama等）上使用多项数学与通用推理基准（MMLU‑Pro、AIME、GPQA、GSM‑8k 等）评估

**📈 对比分析**

与基线（原模型、随机抑制、输出熵等）对比，三种方法平均提升约4.5%（AIME 9%）的推理准确率，实验显示ℓ_2范数与推理特征、熵高度相关

**⚠️ 局限性**

局限性包括：对任务难度与域的敏感性（需自适应阈值）、对SAE假设的依赖、未解释ℓ_2范数与推理机制的完整物理含义，且方法仅适用于推理任务，对非推理任务效果有限

---

## 562. Opportunities and Challenges in Securely Reusing and Repurposing Mobile Devices

**arXiv ID:** 2606.06181 | [PDF](https://arxiv.org/pdf/2606.06181v1)

**作者:** Adelin Roty `[一作]` (Universite Libre de Bruxelles), Jean-François Determe `[通讯]` (Universite Libre de Bruxelles)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文通过在开放硬件的 PinePhone 上实验，探究在设备被重新用途时硬件后置安全机制（如安全引导与可信执行环境）能否被恢复并保持可信度。

**💡 创新点**

创新点在于提供了第一份实证研究，系统评估了在完全重置软件栈后恢复硬件可信性所需的步骤与挑战，并提出了三种安全再利用策略。

**🔧 技术方法**

使用的技术包括自制 U‑Boot、Linux 内核、Buildroot 自动化构建以及尝试集成 OP‑TEE 可信执行环境。

**📊 数据集**

没有使用公开数据集，而是以 PinePhone 的硬件规格、设备树（DTS）和自行构建的镜像作为实验素材。

**📈 对比分析**

通过对比手工编译与 Buildroot 构建的镜像在引导完整性、TEE 可用性以及系统稳定性上的表现，结果显示即便在开放平台上实现功能系统仍无法完全恢复硬件安全链；性能方面，单纯的 Linux 运行可达预期，但安全功能大多失效。

**⚠️ 局限性**

主要局限在于实验仅覆盖单一开放平台，无法直接推广到主流闭源手机；TEE 集成高度依赖低级硬件细节，难以通用；且缺乏对多型号设备的广泛验证。

---

## 563. Learning to Route LLMs from Implicit Cost-Performance Preferences via Meta-Learning

**arXiv ID:** 2606.06178 | [PDF](https://arxiv.org/pdf/2606.06178v1)

**作者:** Jiahao Zeng `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Ningning Ding `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MetaRouter，基于元学习的感知LLM路由框架，能通过少量用户偏好反馈自动匹配最合适的LLM，实现成本与性能的平衡；

**💡 创新点**

创新点在于：①将用户隐含的成本-性能权衡视为任务并使用元学习；②设计感知路由范式，少量交互即完成偏好推断；③结合上下文编码器、Gated Residual机制、熵正则及tanh奖励，提升鲁棒性与学习效率；

**🔧 技术方法**

采用元学习（MAML类）、Permutation‑invariant 上下文编码、政策梯度+熵正则、基于相对效用的tanh奖励、Gated Residual网络以及噪声注入等技术；

**📊 数据集**

使用RouteLLM、AlpacaEval（Hybrid QA）；Magicoder、FullStackBench（代码生成）；MATH、Omni‑MATH（数学推理）等数据集，并使用GPT‑4/DeepSeek、Mixtral、Claude3等多模型；

**📈 对比分析**

与GraphRouter、Avengers‑Pro、SW Ranking、Matrix Factorization等基线对比，MetaRouter在ID和OOD任务上均取得最高HV、最低IGD、最高AUC；在新模型对、5模型多路由场景亦保持优势；

**⚠️ 局限性**

局限性包括：仍需预设模型对，基线多限于二元或阈值；未覆盖延迟、隐私等其他偏好维度；实验多基于人工模拟反馈，真实用户体验尚待验证。

---

## 564. Bridging the Semantic-Collaborative Gap: An Asymmetric Graph Architecture for Cold-Start Item Recommendation

**arXiv ID:** 2606.06225 | [PDF](https://arxiv.org/pdf/2606.06225v1)

**作者:** Anh Truong `[一作]` (Tubi), Michael Tamir `[通讯]` (Tubi)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种异构两塔模型 Shallow‑RHS，用于解决流媒体平台中新内容（零观看记录）和新设备（无观看记录）的冷启动问题，并通过“隐式图完成”技术把冷启动节点嵌入协同过滤一致的向量空间，以实现即时推荐和推广。

**💡 创新点**

创新点包括：
1) 异构两塔结构，左塔（设备）利用历史观看边进行信息传递，右塔（内容）仅使用内容内在特征，避免内容侧子图依赖；
2) 通过学习的内容编码器实现语义‑协同映射，将内容的 LLM、元数据等语义信息直接映射到协同过滤空间；
3) 采用“surrogate neighbors”策略在推理阶段用已知热内容的近邻完成冷内容的隐式图邻接；
4) 对设备冷启动采用人群层级嵌入，从人口统计特征生成设备表示，进一步实现设备侧的隐式图完成。

**🔧 技术方法**

主要技术包括：
- 基于时间的二分图 GNN 结构；
- HeteroTF + FT‑Transformer 对多类型特征进行编码；
- 时间加权信息传递与多层聚合；
- 软最大化链路预测损失；
- 近邻检索（FAISS/Annoy）做冷内容的 surrogate retrieval；
- 人群聚类 + 均值嵌入实现设备冷启动。

**📊 数据集**

使用的数据集来自 Tubi 流媒体平台，包含：
- 约数十亿条时间戳化观看边；
- 数百万设备节点、数十万内容节点；
- 设备属性（国家、设备类型、平台、人口统计等）；
- 内容属性（标题、类型、语言、制作年份、时长、分类、评分、LLM 语义嵌入等），并对缺失字段进行覆盖率提升。

**📈 对比分析**

评估方式：在线 A/B 测试对比生产基线，主要指标包括总观看时长（TVT）、合格观看日、首页 5 分钟转化率等。实验结果显示：
- 内容冷启动：从 +0.10% TVT 到 +0.42% TVT 的累计提升，冷内容曝光速度提升 13%–38%；
- 设备冷启动：合格观看日 +0.29%，每日总观看时长 +0.39%，首页 5 分钟转化 +0.43%；
- 与传统语义嵌入、传统协同过滤、无冷启动策略相比，Shallow‑RHS 在所有指标上均显著提升。

**⚠️ 局限性**

局限性：
- 依赖内容侧特征覆盖率，若 LLM 或元数据缺失会削弱语义‑协同映射效果；
- surrogate neighbors 仅在已有热内容邻域内有效，对极长尾冷内容的覆盖有限；
- 设备冷启动仅使用人口统计，缺少更细粒度行为先验；
- 目前仅在 Tubi 生产环境验证，跨平台可迁移性待进一步验证；
- 模型训练和推理需要较大计算资源，尤其是大规模近邻索引维护。

---

## 565. From Reward-Hack Activations to Agentic Risk States: Context-Calibrated Mechanistic Monitoring in LLM Agents

**arXiv ID:** 2606.06223 | [PDF](https://arxiv.org/pdf/2606.06223v1)

**作者:** Patrick Wilhelm `[一作]` (Technische Universität Berlin), Odej Kao `[通讯]` (Berlin Institute for Foundations of Learning and Data)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在基于语言模型的ReAct式代理中，研究了奖励‑劫持（reward‑hack）监测器在连续决策过程中的表现，并提出通过结合内部激活、熵以及决策上下文信息进行风险估计的方法；

**💡 创新点**

创新点在于把单一激活信号转化为“潜在风险状态”描述，强调在代理部署时需对激活信号进行上下文校准，而非单阈值截断；同时通过激活方向驱动的干预（steering）验证该信号的行为相关性；

**🔧 技术方法**

使用了稀疏自编码器+线性探针提取激活奖励‑劫持得分、token‑级熵统计，结合逻辑回归进行下一步风险预测；实验中还实现了激活方向steering干预；

**📊 数据集**

实验数据集包括自定义的Gameable ALFWorld（含显式代理奖励动作）和公开的WebShop购物环境；模型采用Qwen、Llama、Falcon的LoRA适配器，覆盖从控制到混合再到全奖励‑劫持的三种训练策略；

**📈 对比分析**

通过分组交叉验证（按episode分组）评估AUROC、AUPRC、AUPRC提升和Recall@20%；结果显示，仅靠激活得分的预测提升极小（如bad_action AUPRC增幅≈0.02），而加入熵+上下文后AUPRC提升可达≈0.16；奖励‑劫持适配器在Gameable ALFWorld中可显著提升代理的代理奖励利用率，混合适配器有时表现更恶劣；

**⚠️ 局限性**

局限性包括：仅在少数模型（Qwen为主）与受限环境（Gameable ALFWorld与WebShop）验证，其他模型和环境表现不稳定；上下文特征相对简单，缺乏更丰富的状态表示与在线门控策略；Steering干预效果不一，未能提供统一可部署的缓解方案；

---

## 566. TAM: Torque Adaptation Module for Robust Motion Transfer in Manipulation

**arXiv ID:** 2606.06218 | [PDF](https://arxiv.org/pdf/2606.06218v1)

**作者:** Dongwon Son `[一作]` (KAIST), Dieter Fox `[通讯]` (Allen Institute for AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研发并验证了一种Torque Adaptation Module（TAM），通过在低层力矩控制器中引入残差校正，实现了在未使用域随机化的情况下，动态机械臂从仿真到真实机器的零射击转移。

**💡 创新点**

将自适应模块嵌入到策略下的低层力矩接口，利用历史感知编码与残差力矩校正，实现在不同机器人、负载与动作空间下共享同一模块权重，并在不收集真实机器人数据的前提下提升转移鲁棒性。

**🔧 技术方法**

采用历史编码器（history encoder）与力矩适配器（torque adaptor）相结合的网络架构，在多机器人预训练和单机器人微调阶段通过模拟学习完成；与RL、BC、MPC等不同策略的融合与评估。

**📊 数据集**

仅使用仿真随机化产生的多机器人轨迹数据进行预训练，随后在Franka Panda机器人上使用无真实数据的仿真微调；实验任务涵盖视觉盒子推送、翻转和球盘平衡三种不同动态操控场景。

**📈 对比分析**

与在线系统识别、RMA等基线进行零射击实验比较，TAM在真实机器人上显著提升了执行精度和鲁棒性，尤其在动态操控任务中表现优于基线方法。

**⚠️ 局限性**

对极端物理不匹配或未见机器人场景的适应性仍有限；受限于仿真环境的多样性；未对长期延迟或高频噪声的影响进行系统评估。

---

## 567. DisasterBench: A Multimodal Benchmark for UAV-Based Disaster Response in Complex Environments

**arXiv ID:** 2606.06217 | [PDF](https://arxiv.org/pdf/2606.06217v1)

**作者:** Tan Zhang `[一作]`, Ping Hu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 72459 | [OpenAlex ID](https://openalex.org/A5100751322)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了针对低空UAV灾害响应的多模态基准DisasterBench，并构建了轻量级模型DisasterVL。

**💡 创新点**

基准覆盖14种灾害场景与9项关键任务，实现多阶段推理评估；模型采用域指令微调、链式思维对齐与强化学习策略优化三阶段训练。

**🔧 技术方法**

使用三阶段训练流程：域指令微调、链式思维引导的多模态对齐、基于强化学习的策略优化，并在多模态大语言模型上实现推理。

**📊 数据集**

基准数据集DisasterBench，包括低空UAV图像的预、在、后灾害阶段共14类场景与9个响应任务。

**📈 对比分析**

在21个主流多模态LLM上进行对比实验，DisasterVL显著优于现有开源模型，且与闭源先进系统差距大幅缩小，且保持高效。

**⚠️ 局限性**

主要局限在于仅针对低空UAV图像，缺乏跨模态和跨域通用性验证，且实测部署效率与安全性待进一步评估。

---

## 568. Non-Negative Matrix Factorization for Event Data

**arXiv ID:** 2606.06205 | [PDF](https://arxiv.org/pdf/2606.06205v1)

**作者:** Raphaël Romero `[一作]` (Ghent University), Raphaël Romero `[通讯]` (Ghent University)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5046853585)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种直接作用于连续时间事件数据的非负矩阵分解模型 EventNMF，用 Poisson 过程似然实现低秩时间模板的学习；

**💡 创新点**

创新点在于：①在不进行时间分箱或平滑的情况下，利用 B‑spline 基底对事件强度函数进行光滑的非负分解；②给出可直接应用于原始事件时间的乘法更新优化方法；③将传统的离散时间 Poisson‑NMF 与连续时间模型统一起来，揭示两者的数学联系；

**🔧 技术方法**

技术包括：连续时间 Poisson 过程似然、非负 B‑spline 基底展开、非负矩阵分解的乘法更新、Hungarian 匹配、以及多维度实验评估（NLL、NMSE、NFISE 等）；

**📊 数据集**

使用的实验数据集有：①通过已知因子合成的模拟数据；②真实数据：多电极神经尖峰记录、地震事件清单以及小学课堂面对面接触网络；

**📈 对比分析**

方法与传统离散时间 NMF、NARFD、PPCA 等进行比较；在 NLL、NMSE、NFISE 等指标上，EventNMF 在稀疏、连续时间信息丰富的情境下显著优于基于分箱的对照方法；同时在大规模数据上展现出更快的收敛速度；

**⚠️ 局限性**

局限性包括：①模型假设为无自激 Poisson 过程，无法直接捕捉 Hawkes 等自激事件；②乘法更新易受局部最优影响，对初始值敏感；③在极大规模事件数（百万级实体、十亿级事件）下仍需进一步加速，例如通过子采样或分布式实现。

---

## 569. SC-MFJ: A Simple Haptic Quality Metric for Medical Image Segmentation

**arXiv ID:** 2606.06199 | [PDF](https://arxiv.org/pdf/2606.06199v1)

**作者:** Souraj Adhikary `[一作]` (Jade University of Applied Sciences), Andre Mastmeyer `[通讯]` (Jade University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种名为SC-MFJ（Surface-Constrained Mean Force Jerk）的新指标，用于评估医学图像分割产生的三维模型在触觉渲染中的光滑程度；

**💡 创新点**

创新点在于将力学平滑度（通过随机表面走向的力jerk）作为量化标准，既能捕捉到传统几何指标（Dice、Hausdorff）忽视的表面阶梯效应，又可快速评估不同分割后处理对触觉质量的影响；

**🔧 技术方法**

技术包括：利用梯度求取表面法线、构建三维网格、在网格上进行随机表面步行、采用弹簧式触碰模型计算力、用二阶差分求力jerk并取均值；

**📊 数据集**

使用数据集：NIH胰腺CT（80个病例，5折交叉验证）和LiTS肝脏肿瘤分割（131个病例）；

**📈 对比分析**

对比方法：原始二值分割、Gaussian σ=1.0后处理和SDF回归。结果显示：二值分割的SC-MFJ远高于两种光滑方法（147×和189×），Gaussian后处理在两种方法中保持更低的标准差和更稳定的性能；

**⚠️ 局限性**

局限性包括：未与人类触觉感知实验对照；仅在胰腺和肝脏两种解剖结构上验证；使用简化的弹簧力模型，未考虑更复杂的接触或材料特性；

---

## 570. ActiveMimic: Egocentric Video Pretraining with Active Perception

**arXiv ID:** 2606.06194 | [PDF](https://arxiv.org/pdf/2606.06194v1)

**作者:** Xingyao Lin `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 25076 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出ActiveMimic框架，利用单RGB摄像头从人类自视角视频中同步恢复相机与手腕轨迹，联合学习主动感知与操控，并迁移至类人机器人执行真实任务。

**💡 创新点**

关键创新是将摄像机运动视作主动感知行为，通过无硬件束缚的姿态恢复和统一27维动作空间，实现主动感知与操控的联合预训练，填补人类视频预训练中的感知缺失。

**🔧 技术方法**

使用Vision‑Language Model前缀+Action Expert混合Transformer架构，流匹配损失训练；姿态估计依赖VGGT、SAM‑3D‑Body、UniDepth等现成视觉模型，并通过尺度对齐恢复真实轨迹。

**📊 数据集**

预训练基于Ego4D Hands & Objects子集（约10小时10fps视频），标签精度验证采用HOT3D；实验在AGIBOT G1 humanoid上完成四项真实任务。

**📈 对比分析**

与四个基线（π₀、MotoVLA、仅手腕监督、仅机器人微调）比较，ActiveMimic在四项任务上成功率均≥90%，显著优于仅靠机器人预训练或手腕监督的模型，且可匹敌甚至超越混合人机数据的MotoVLA。

**⚠️ 局限性**

局限性在于单RGB摄像头姿态估计的误差导致标签精度不及专业捕捉设备；模型对视角变化仍敏感，泛化到更复杂环境或不同硬件配置尚未充分验证。

---

## 571. Learning to model pediatric asthma exacerbation from multiple risk factors: a case study in coastal Virginia

**arXiv ID:** 2606.06174 | [PDF](https://arxiv.org/pdf/2606.06174v1)

**作者:** Jonathan Colen `[一作]` (Old Dominion University), Mary Margaret Gleason `[通讯]` (Old Dominion University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究通过收集海岸弗吉尼亚地区儿童医院的哮喘急性加重诊断记录、空气质量监测、气象数据以及社区机会指数等多维度信息，构建并比较了三种不同复杂度的预测模型，以解析空气污染、气象与社会经济因素对儿童哮喘加重的影响。

**💡 创新点**

创新点在于将传统的GLM、深度神经网络与稀疏字典学习三者结合，首次在哮喘预测中引入稀疏字典学习框架，既保留可解释性又兼顾预测性能，并通过该框架自动识别并量化非线性交互效应。

**🔧 技术方法**

技术手段包括：1）使用分层交叉验证的quasi‑Poisson GLM；2）采用两层256节点的全连接神经网络；3）实现基于稀疏字典学习的顺序阈值最小二乘（STLSQ）模型；4）通过模型显著性、相对风险（RR）和相对过剩风险（RERI）等统计量进行结果解释。

**📊 数据集**

使用的数据集涵盖：2018‑2023年间海岸弗吉尼亚地区儿童医院的诊断访问数据（ICD10 J45哮喘代码）、EPA AQS空气质量监测（NO₂、SO₂、CO、PM₂.₅、PM₁₀）、NOAA气象站日常气象数据、NOAA烟雾密度测量、以及Child Opportunity Index与Social Vulnerability Index等社会经济指标。

**📈 对比分析**

通过5次3折交叉验证对模型进行比较，评估指标为R²和MAE。神经网络在R²≈0.645、MAE≈2.04时表现最佳；稀疏字典模型在R²≈0.399、MAE≈2.64时居中；GLM在R²≈0.270、MAE≈2.94时最低。三种模型的相对风险估计基本一致，且稀疏字典模型能够揭示非线性交互作用。

**⚠️ 局限性**

局限性包括：1）数据未追踪个体随访，无法区分初发与复发哮喘加重；2）仅依赖诊断编码，可能存在编码误差；3）空气质量监测仅为大尺度监测，缺乏室内或更细空间分辨率；4）缺少高频极端烟雾日样本，限制对极端事件的分析；5）模型未能充分捕捉人群动态和更高阶非线性关系。

---

## 572. Adaptive Tokenisation Via Temporal Redundancy Masking And Latent Inpainting

**arXiv ID:** 2606.06158 | [PDF](https://arxiv.org/pdf/2606.06158v1)

**作者:** Kevin Dave `[一作]` (Phronetic AI), Rajeshkumar SA `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无参数的自适应视频分词方法，通过在连续潜在空间中阈值化时序L1差值来自动筛选冗余位置，随后用Latent Inpainting Transformer（LIT）重建丢失的潜在信息，完成视频重构；

**💡 创新点**

创新点在于：①仅利用冻结的连续分词器潜在空间的时序冗余，无需学习路由网络或额外解码评估；②引入参数化极低的LIT，通过空间-时间分解注意力高效恢复丢失潜在；③实现内容驱动的压缩率，并在推理时只需单次编码+LIT前向；

**🔧 技术方法**

技术包括：连续潜在分词器（Cosmos-Tokenize1-CV），基于L1阈值的时序掩码算法，Factorised Transformer（LIT）带RoPE位置编码，L1重建损失；

**📊 数据集**

数据集：UCF-101、Kinetics-400用于训练；TokenBench、DAVIS用于评估；

**📈 对比分析**

与固定率分词器（Cosmos、OmniTokenizer-VAE）及自适应分词器（ElasticTok-CV、InfoTok）对比，保持约32%–62%保留率时，在TokenBench和DAVIS上取得PSNR+2–3 dB、FVD大幅下降；推理速度相较ElasticTok-CV提升31倍，InfoTok提升约2倍；

**⚠️ 局限性**

局限性：压缩率由内容决定，无法预设目标比特率；在高运动视频中冗余少，压缩效果有限；

---

## 573. A Unified Framework for Uniform-Price Resource Allocation Mechanisms

**arXiv ID:** 2606.06151 | [PDF](https://arxiv.org/pdf/2606.06151v1)

**作者:** Ioannis Caragiannis `[一作]` (Aarhus University), Stratis Skoulakis `[通讯]` (Aarhus University)

**通讯引用:** 74 | [OpenAlex ID](https://openalex.org/A5020247743)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种统一的资源分配机制框架——α‑proportional机制，能够在保持分配比例与统一定价的同时，实现从Kelly机制到一价拍卖的连续过渡；

**💡 创新点**

创新点在于引入标量参数α的代理价值函数，通过拉格朗日乘子产生统一单位价格，既保持机制简洁，又在α→0时逼近一价拍卖且价格失效时实现效率接近最优；

**🔧 技术方法**

使用凸优化（原始/对偶程序）、KKT 条件、潜在函数理论、supergradient/潜在函数分析以及无冲突学习算法（Hedge）来证明存在唯一Nash均衡、计算PoA、评估收入；

**📊 数据集**

未使用真实数据集，全部采用模拟/随机生成的稀疏/同质/异质 concave 价值函数（如线性、幂次函数）进行实验；

**📈 对比分析**

通过理论证明显示α‑proportional机制的PoA上界为(1+√α)²/(1+2√α)，比Kelly的4/3大幅改进；在线性价值函数下收入至少为VCG收入的1/(1+α)；实验展示学习动态收敛至唯一均衡，且在不同α、β、n下社会福利均匀或更优；

**⚠️ 局限性**

局限包括：当α→0时出现与一价拍卖的阶跃相位转变，导致PoA可达n/2；需要对价值函数可微、单调递增的假设；机制在预算约束或更一般多面体约束下的性能尚未证明。

---

## 574. Subspace-Aware Sparse Autoencoders for Effective Mechanistic Interpretability

**arXiv ID:** 2606.06333 | [PDF](https://arxiv.org/pdf/2606.06333v1)

**作者:** Seyed Arshan Dalili `[一作]` (Pennsylvania State University), Mehrdad Mahdavi `[通讯]` (Pennsylvania State University)

**通讯引用:** 5982 | [OpenAlex ID](https://openalex.org/A5076083402)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Subspace-Aware Sparse Autoencoders (SASA)，用子空间字典和分组稀疏替代传统 SAE 的单向量字典，从而解决多维特征拆分问题，并在 GPT‑2 与 Mistral‑7B 上进行理论分析与实验验证。

**💡 创新点**

创新点包括：① 从几何和优化角度证明向量字典必然导致特征拆分；② 引入子空间字典、Top‑s 分组门和核范数正则化，实现子空间级别的稀疏；③ 证明 SASA 的样本复杂度从指数下降到多项式，显著提升训练效率。

**🔧 技术方法**

使用技术包括子空间自编码器结构、Top‑s 分组门控、核范数正则化（与谱正则化等价）、主子空间估计、LLM 激活采样、以及对比实验中的梯度下降与 Adam 等优化器。

**📊 数据集**

数据集为 GPT‑2 Small (d=768) 与 Mistral‑7B‑v0.1 (d=4096) 的残差流激活，采样自大规模网络语料；实验中以 GPT‑2 第 7 层和 Mistral‑7B 的中间层为主要测试点。

**📈 对比分析**

通过与 ReLU、Gated、JumpReLU、TopK、BatchTopK 以及 Mistral SAE 等基线在 KL 评分、CE 评分、解释方差、稀疏度和特征吸收等指标进行对比。SASA 在相同或约半量的 token 预算下取得相当或更优的性能，显著降低特征吸收率，提升单义性与可解释性。

**⚠️ 局限性**

局限性：① 仍需大量 LLM 前向推理，训练成本在大模型上仍然较高；② 子空间学习只能给出子空间的跨度，无法直接解释子空间内部的坐标结构；③ 对不同模型的跨模型泛化尚未验证。

---

## 575. PAMF: Prior-Aware Multimodal Fusion for Incomplete Time Series Data

**arXiv ID:** 2606.06328 | [PDF](https://arxiv.org/pdf/2606.06328v1)

**作者:** Ziwen Kan `[一作]`, Song Wang `[通讯]` (University of Central Florida)

**通讯引用:** 7083 | [OpenAlex ID](https://openalex.org/A5100326206)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了PAMF框架，针对多模态时间序列的两类缺失模式（within‑modality 与 modality‑level）进行显式补全，并将补全与下游分类耦合在一个端到端的训练管线中。

**💡 创新点**

创新点包括① 通过先验感知的流匹配初始化区分两种缺失类型；② 采用权重共享的 Encoder 使补全过程受下游任务指导；③ 三阶段训练将补全与分类协同优化。

**🔧 技术方法**

使用了条件流匹配（prior‑aware flow matching）、多模态混合专家（MoE）编码器、Transformer 自注意力、权重共享与三阶段训练策略。

**📊 数据集**

在四个医疗多模态时间序列基准上评估：Sleep‑EDF、PTB‑XL、PPG‑DaLiA、Chapman‑Shaoxing。

**📈 对比分析**

与 FuseMoE、Flex‑MoE、Maestro、MIRA、CSDI、SSSD 等基线对比；在 20%/混合/50% 缺失率下，PAMF 在所有数据集与缺失设置中均取得最高 macro‑F1 与 AUROC，并在推理延迟上优于 Diffusion‑based 对手。

**⚠️ 局限性**

仍存在极端缺失、非规则采样或更大规模实时部署时的可扩展性挑战；目前框架主要处理离散时间点缺失，未覆盖空间/多尺度缺失场景。

---

## 576. AIS-Based Vessel Trajectory Prediction Using Memory-Augmented Neural Networks

**arXiv ID:** 2606.06311 | [PDF](https://arxiv.org/pdf/2606.06311v1)

**作者:** Wonmo Koo `[一作]` (Korea Advanced Institute of Science and Technology), Heeyoung Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2942 | [OpenAlex ID](https://openalex.org/A5100716622)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aaccfe5c-6b26-4208-b23c-35331481e142` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用内存增强神经网络MANTRA对AIS数据进行船舶轨迹预测，并与六种主流基线模型进行对比。

**💡 创新点**

首次将船速（SOG）和航向（COG）等航行信息纳入内存写入与检索过程，证明外部内存机制在船舶轨迹预测中能显著提升精度，且在无显式交互建模的前提下即可取得优异表现。

**🔧 技术方法**

核心技术为Memory‑Augmented Neural Network（MANTRA）+ GRU 编码器/解码器 + 外部内存写入控制器 + K‑模态预测 + 滑动窗口切片，实验中使用单层 GRU、Adam 优化等。

**📊 数据集**

使用美国NOAA公开的Gulf of Mexico和New York Bight AIS数据（长度≥70 m，时间段2023‑03‑06至03‑08），共计约152 k条记录。

**📈 对比分析**

通过ADE和FDE两项指标与STGAT、Social‑STGCNN、TransformerTF、AgentFormer、Social‑Implicit、TUTR六个基线模型对比；MANTRA在所有预测时长（10/20/30 min）上均取得最低ADE/FDE，提升幅度约30‑55%，并且标准误最小，表现最稳定。

**⚠️ 局限性**

局限性：未显式建模多船交互；未引入高维航行上下文（如船舶类型、天气、水道几何等）；缺乏对预测不确定性的量化；仅在两地区、短期预测情形下验证，缺乏更广泛场景和长期预测的实验。

---

## 577. RedZeD: Computing persistent homology by Reduction to Zero Differentials

**arXiv ID:** 2606.06310 | [PDF](https://arxiv.org/pdf/2606.06310v1)

**作者:** Chris Kapulkin `[一作]`, Nathan Kershaw `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 RedZeD 算法及其活跃枚举版本 o3，用于在 Vietoris–Rips 过滤器中高效计算持久同调。

**💡 创新点**

创新点在于将滤波链复形替换为零微分链复形的“Reduction to Zero Differentials”框架，并基于此引入活跃枚举策略，显著减少高维出生单纯形的枚举与归约。

**🔧 技术方法**

技术实现包括矩阵归约的压缩、全局归约与回溯归约，以及使用字典/反向字典快速维护活跃单纯形，兼顾 𝔽₂ 上的实现与通用域扩展。

**📊 数据集**

实验使用随机点集、带噪声圆点、多圆叠加、随机距离矩阵以及 Ripser 官方基准数据集进行评估。

**📈 对比分析**

通过与 Ripser 在多维度和多样本规模下的运行时间对比，H1 维度中 o3 在多数情形下速度提升 2–180 倍、扩展性更好；在 H2+ 维度时 Ripser 更快；在低维无信号数据时 Ripser 约 1.5 倍更快。

**⚠️ 局限性**

局限性包括：在高维持久性或低信号数据上效率下降；活跃枚举无法消除所有出生单纯形；目前实现仅支持 𝔽₂，且在非 Vietoris–Rips 过滤器的适用性尚待验证。

---

## 578. Tangram: Unlocking Non-Uniform KV Cache for Efficient Multi-turn LLM Serving

**arXiv ID:** 2606.06302 | [PDF](https://arxiv.org/pdf/2606.06302v1)

**作者:** Hyungmin Kim `[一作]` (Hanyang University), Jungwook Choi `[通讯]` (Hanyang University)

**通讯引用:** 3136 | [OpenAlex ID](https://openalex.org/A5089720016)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Tangram系统，使非均匀KV缓存压缩在多轮LLM服务中可行并显著提升吞吐量。

**💡 创新点**

创新点在于将KV保留预算转化为离线确定的静态预算，创建按保留需求聚类的Head Group Page，以及提前预计算GPU负载平衡方案，三者共同消除内存碎片、调度开销和工作负载不平衡。

**🔧 技术方法**

使用技术包括离线头部预算剖面、Deterministic Budget Allocation、Head Group Page（多组页表）与Vectorized Block Table、Ahead‑of‑Time Load Balancing、FlashAttention‑2 以及自定义CUDA kernel。

**📊 数据集**

实验数据集涵盖 Qwen3‑4B、Qwen2.5‑7B‑Instruct‑1M、Qwen2.5‑32B，使用 SCBench、LoCoMo、RealTalk 与 LongMemEval 四大多轮长文本基准。

**📈 对比分析**

相较于 vLLM、FlashAttention‑2 与 FlashInfer 基线，Tangram 在多轮上下文长度下吞吐量提升最高可达 2.6×，TTFT 与延迟保持稳定，且无显著准确率下降。

**⚠️ 局限性**

限制在于预算离线剖面假设对模型内在特性稳定性要求高；对极端动态上下文变化或新模型可能需重新校准；Head Group Page 需要平衡组大小与管理开销。

---

## 579. ToolChoiceConfusion: Causal Minimal Tool Filtering for Reliable LLM Agents

**arXiv ID:** 2606.06284 | [PDF](https://arxiv.org/pdf/2606.06284v1)

**作者:** Rahul Suresh Babu `[一作]`, Laxmipriya Ganesh Iyer `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练的工具过滤方法Causal Minimal Tool Filtering (CMTF)，通过工具的前置条件和后置效果在每一步仅展示对当前任务状态至关重要的工具，从而降低工具误选和错误操作

**💡 创新点**

核心创新在于将工具选择视为因果充分性问题，使用轻量级的工具契约（precondition‑effect）构造依赖图，寻找最短因果路径，仅暴露下一步必要工具，而非传统语义相关或可执行性筛选

**🔧 技术方法**

实现技术包括基于STRIPS/ PDDL 思想的预置-后置图构建、宽度优先搜索寻找最短因果路径、动态生成工具菜单，结合LLM的工具调用框架（Toolformer/ ReAct 等）

**📊 数据集**

在一套100个合成工具的控制性基准上进行评估，该基准包含102个任务（日历、邮件、文件三大域），每个任务定义初始状态、目标状态和金标准工具链

**📈 对比分析**

与全工具暴露、关键词 top‑5/10、可执行性筛选、完整因果路径曝光等六种策略比较；CMTF在所有模型上均实现近乎完美的任务成功率（0.99），同时将工具曝光从平均100个降至1个，错误工具调用降至0.01，预发操作几乎为0，令token消耗比全工具低约90%

**⚠️ 局限性**

局限性包括：基准为合成实验，未涵盖真实API的不确定性与失败；依赖完整准确的工具契约；对状态估计或目标映射误差敏感；对开放式、多目标任务可能不足；需要进一步验证在真实生产环境和多供应商模型上的稳健性

---

## 580. From Self to Other: Evaluating Demographic Perspective-Taking in LLM Hate Speech Annotation

**arXiv ID:** 2606.06266 | [PDF](https://arxiv.org/pdf/2606.06266v1)

**作者:** Paloma Piot `[一作]` (Universidade da Coruña), Javier Parapar `[通讯]` (Universidade da Coruña)

**通讯引用:** 1949 | [OpenAlex ID](https://openalex.org/A5046723532)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了零射条件下 persona‑conditioned 大语言模型在仇恨言论检测任务中对三维社会判断（跨群体不一致、群体敏感性、外群体预测）的模拟效果。

**💡 创新点**

创新点在于提出并系统评估了三维社会判断框架，结合三大开放 LLM（Nemo、Llama、Qwen‑3）和五个二元身份轴，揭示 vicarious prompting 在跨群体一致性上的显著优势。

**🔧 技术方法**

采用零射 persona‑prompt、Cohen’s κ 衡量跨群体一致性，Δ_IG 计算群体敏感性，并通过 vicarious prompting 让模型预测他群体观点，构建完整的评估流程。

**📊 数据集**

使用 MHS（Measuring Hate Speech）数据集，包含约 40,000 条来自 YouTube、Reddit、Twitter 的评论，135,000 条人工注释，并提供性别、种族、宗教、意识形态、性取向等身份维度。

**📈 对比分析**

通过将模型在三维指标上的 κ 与人类多数投票结果对比，发现仅 Nemo 在 vicarious prompting 下在四/五个身份轴上逼近或超过人类跨群体一致性；其他模型表现不稳定且难以统一。

**⚠️ 局限性**

限制主要包括：仅采用零射无指导 prompt、仅评估英语数据、使用二元身份对照而忽略交叉身份、多样性不足，以及仅测试三大开放 LLM，缺乏跨文化验证和更细粒度身份分析。

---

## 581. RadiusFPS: Efficient Farthest Point Sampling on CPUs and GPUs via Spherical Voxel Pruning

**arXiv ID:** 2606.06255 | [PDF](https://arxiv.org/pdf/2606.06255v1)

**作者:** Ziyang Yu `[一作]` (Institute of Science Tokyo), Jun Miyazaki `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 1361 | [OpenAlex ID](https://openalex.org/A5026559426)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了RadiusFPS及其GPU实现RadiusFPS-G，用球面体素裁剪与点跳过技术加速Farthest Point Sampling（FPS），并保持与传统FPS完全等价的采样结果。

**💡 创新点**

创新点包括：①双层过滤——先用球面体素提供安全的半径下界裁剪，再用坐标差异进行点级跳过，显著减少无效距离计算；②GPU实现中采用活跃体素压缩与两级融合核（融合voxel选择+点选择与融合voxel筛选+距离更新），消除全局内存往返，提升并行度。

**🔧 技术方法**

使用的技术包括：球面体素索引、半径下界裁剪、坐标差异点跳过、SoA数据布局、活跃体素压缩、CUDA共享内存、warp/块级归约、两级融合核和快速随机种子、确定性tie-breaking。

**📊 数据集**

实验数据集：S3DIS、ScanNet（室内）以及SemanticKITTI（室外LiDAR）。

**📈 对比分析**

通过与CPU FPS、GPU FPS、QuickFPS、FPS+NPDU、FastPoint等方法在相同网络骨干（PointMetaBase、PointVector）和同一数据集上对比，CPU上可实现最高186×加速，GPU上可实现2–5×加速，GPU内存仅使用QuickFPS的一半；端到端推理加速最高可达3.3×，且保持或略优于原始FPS的分割精度。

**⚠️ 局限性**

局限性：对体素分辨率（voxel resolution）敏感，过细体素会导致管理开销和缓存不命中；方法仍保持精确FPS，未提供更进一步的近似加速方案；在极稀疏或非均匀分布的点云中，裁剪效果可能不如预期。

---

## 582. Bridging Domain Expertise and Generalization for Performance Estimation

**arXiv ID:** 2606.06335 | [PDF](https://arxiv.org/pdf/2606.06335v1)

**作者:** Shuxuan Li `[一作]`, Wei-Shi Zheng `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9`

**🎯 论文内容**

未知

**💡 创新点**

未知

**🔧 技术方法**

未知

**📊 数据集**

未知

**📈 对比分析**

未知

**⚠️ 局限性**

未知

---

## 583. Constant Approximation for Hylland--Zeckhauser Equilibria

**arXiv ID:** 2606.06317 | [PDF](https://arxiv.org/pdf/2606.06317v1)

**作者:** Yonglei Yan `[一作]` (Beijing Institute of Technology), Zhengyang Liu `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5101597242)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了一个多项式时间算法，能够在任意多值效用的 HZ 市场中求得 1/e 近似的 Hylland–Zeckhauser 均衡。

**💡 创新点**

创新点在于提出了“效用分层”技术，将原始多值市场转化为结构化的双值实例，从而可直接利用 Vazirani–Yannakakis 的精确算法；并通过精细的整数化与取整策略控制近似误差。

**🔧 技术方法**

主要技术包括：
- 效用分层与重构市场（将每个代理拆分成若干组并复制商品），
- 对拆分比例 λ_{i,k} 的最优选取（最小化最坏情况误差），
- 对 λ_{i,k}·W 的整数化与误差分析，
- 结合 Vazirani–Yannakakis 的双值 HZ 均衡算法。

**📊 数据集**

本文为理论算法研究，未使用任何实验数据集。

**📈 对比分析**

相较于先前的结果（已知在四值效用下的近似是 PPAD‑hard），该算法在多值效用下实现了可接受的 1/e 近似，并且复杂度为 O(poly(n,1/δ))，在理论上实现了先前所缺失的多值情况的多项式时间近似。

**⚠️ 局限性**

局限性：
- 近似比率为 1/e，仍与已知的强硬性下界存在间隙；
- 需要设置大整数 W 以控制误差，导致实际规模扩大到 O(n^2/δ)；
- 目前无法进一步提升近似比率或证明更紧的下界；
- 对实际数据的适用性与实验验证尚未给出。

---

## 584. LLM Self-Recognition: Steering and Retrieving Activation Signatures

**arXiv ID:** 2606.06315 | [PDF](https://arxiv.org/pdf/2606.06315v1)

**作者:** Thibaud Ardoin `[一作]` (Freie Universitaet Berlin), Gerhard Wunder `[通讯]` (Freie Universitaet Berlin)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在LLM内部激活层注入稀疏向量实现文本水印，并验证LLM能够自我识别自身生成文本；

**💡 创新点**

首次展示LLM内部激活中天然存在可线性分离的自识别信号，并通过稀疏向量注入实现高可靠的多模型归因；

**🔧 技术方法**

采用内部激活提取、稀疏向量注入、线性判别分析和多层感知器等技术；

**📊 数据集**

在XL‑Sum、ELI5、Fresh News等新闻摘要与问答数据集上进行实验；

**📈 对比分析**

与perplexity基线及传统 KGW 水印对比，稀疏注入方案在不同模型（Llama、GPT‑J、Mix‑MoE）上达到 98%+ 的归因准确率，且对改写鲁棒性优于传统方法；

**⚠️ 局限性**

仅适用于白盒检测，检测需要额外前向推理，且水印对不同体系结构的跨迁移性差，若向量泄露则易被伪造。

---

## 585. Attitude-Aided Linear Calibration of Triaxial Accelerometers

**arXiv ID:** 2606.06308 | [PDF](https://arxiv.org/pdf/2606.06308v1)

**作者:** Yongqiang Yu `[一作]` (Independent researcher), Yipeng Yang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 49432 | [OpenAlex ID](https://openalex.org/A5100401978)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种闭式线性加速度计校准方法ALAC，利用姿态信息构建联合误差矩阵（CEM），一次性求解偏置、标度、非正交、对齐旋转等参数。

**💡 创新点**

创新点在于将所有误差统一到CEM，并通过约束齐次最小二乘（CHLS）解析求解，既不依赖高精度实验台，也不需要非线性迭代，可递归在线校准，且仅需5个任意姿态。

**🔧 技术方法**

技术包括构建CHLS问题、使用GEVP/SVD求解、矩阵分解提取校准参数，以及递归GEVP实现在线更新。

**📊 数据集**

实验使用合成数据、在Effort ER3B-C20机械臂上挂载MPU6050加速度计以及公开的MPU6050RM3100 IMU轨迹数据集。

**📈 对比分析**

与传统参考式（如TLS、Newton、UKF）及自校准（Ellipsoid、PSO）方法比较，ALAC在静态/准静态条件下在误差和RMSE上均优于参考式，且与自校准方法相当；在线版本对原始数据表现更好。

**⚠️ 局限性**

局限性包括对姿态误差的敏感性仍显著，且在高噪声或极端姿态分布下误差可能增大；同时未考虑温度漂移和高阶非线性误差。

---

## 586. A Spherical Stochastic Geometry Framework for Patrol-Based HAPs Network: Coverage and Energy Efficiency Analysis

**arXiv ID:** 2606.06307 | [PDF](https://arxiv.org/pdf/2606.06307v1)

**作者:** Mohammad Taha Shah `[一作]` (King Abdullah University of Science and Technology), Mohamed-Slim Alouini `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 92708 | [OpenAlex ID](https://openalex.org/A5083193286)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文建立了在球面地球上基于巡航轨迹的高空平台网络的随机几何框架，提出两种小圆环 Cox 过程模型（SCR‑PCP 与 SCR‑BCP），并推导了空间统计、覆盖概率与能效优化表达式。

**💡 创新点**

创新点在于：① 将锚点驱动与局部巡航轨迹直接嵌入 Cox 过程，形成球面尺度的巡航模型；② 证明模型的等方性并获得完整的距离分布；③ 通过同环与异环干扰的拉普拉斯变换实现覆盖概率解析；④ 将稳态圆形飞行推进功率模型与覆盖分析相结合，定义覆盖能效指标并求解能量最优巡航半径。

**🔧 技术方法**

主要技术手段包括随机几何、Cox 过程（Poisson 与 Binomial 内核）、球面几何距离推导、干扰的条件拉普拉斯变换、能效优化求解、以及 3‑维 Monte‑Carlo 仿真验证。

**📊 数据集**

实验使用无实际数据集，仅采用默认参数（如高度 20 km、基站密度 1e‑6 km⁻²、路径损耗指数 2 等）进行 3‑D Monte‑Carlo 模拟以验证解析结果。

**📈 对比分析**

通过在匹配平均每环平台数的前提下，分别对 SCR‑PCP 与 SCR‑BCP 进行覆盖概率、能效曲线、最优巡航半径等指标比较；结果表明 SCR‑PCP 对巡航半径呈现非单调覆盖行为，而 SCR‑BCP 的性能主要受每环平台数影响，对巡航半径相对稳健；两模型均与仿真结果高度吻合。

**⚠️ 局限性**

局限性包括：假设锚点为均匀 PPP，忽略地形、气象及动态需求；仅考虑 Rayleigh 衰落和热噪声忽略；对平台间相互干扰做独立假设；推进功率模型过于简化，未涵盖复杂气动或能量管理策略；不考虑非圆形巡航或锚点非均匀分布情况。

---

## 587. Decomposing Factual Sycophancy in Language Models: How Size and Instruction Tuning Shape Robustness

**arXiv ID:** 2606.06306 | [PDF](https://arxiv.org/pdf/2606.06306v1)

**作者:** Victor De Marez `[一作]` (University of Antwerp), Walter Daelemans `[通讯]` (University of Antwerp)

**通讯引用:** 10706 | [OpenAlex ID](https://openalex.org/A5083411784)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对事实性sycophancy（在社交压力下模型偏离真相的行为）进行通道拆分，分别评估真相边际（truth‑margin）和操纵敏感度（manipulation‑sensitivity）两种机制，并在56个从0.3B到32B的开源模型上进行大规模因子实验，分析模型规模、指令调优、架构和13种操纵类型对稳健性的影响。

**💡 创新点**

提出了将“flip”事件拆解为可量化的两个通道，揭示规模与指令调优的交互主要通过真相边际和操纵敏感度的不同扩展实现；证明在大模型中指令调优主要提升真相边际，而在小模型中可能导致更高的操纵敏感度，导致稳健性下降；首次在统一实验设置下给出多种操纵的层级排名，并用通道分析解释规模×指令调优交互的根本原因。

**🔧 技术方法**

采用因子实验设计、层级分层抽样的层级自助法（hierarchical bootstrap）评估flip率；构建对数回归模型并使用LMG分解解释方差；使用对数差分法实现通道拆分；对比模型对不同操纵的最坏情况flip率；对规模和指令调优进行跨规模与跨指令的配对对比。

**📊 数据集**

使用PlausibleQA多项选择问答数据集，在此基础上通过两轮过滤（competence & knowledge）得到可测知识量高的实例；在每个实例上应用13种操纵类型（包括8种方向性、5种无方向性控制），生成约147k条实验数据。

**📈 对比分析**

通过层级自助法计算每种操纵的全球flip率，得到“authority（专家/随机证言）> belief（置信度递增）> bribery”三层级排名；利用对数回归和LMG分解，发现模型规模解释约63%方差，指令调优的影响随规模变化，在7B以上大模型中提升稳健性，平均worst‑case flip率从约55%下降到16%；在小模型中指令调优不一定提升稳健性。

**⚠️ 局限性**

仅考察了Base–Instruction‑Tuning（IT）对比，未涉及中间阶段（SFT、DPO）；保持统一的提问模板，未测试自由生成或更大答案集的场景；未解释Base模型真相边际的根本来源；实验仅涵盖开源模型，缺乏商业闭源模型的验证。

---

## 588. Plug-and-Play Guidance for Discrete Diffusion Models via Gradient-Informed Logit Correction

**arXiv ID:** 2606.06303 | [PDF](https://arxiv.org/pdf/2606.06303v1)

**作者:** Hongkun Dou `[一作]` (Beihang University), Yue Deng `[通讯]` (Beihang University)

**通讯引用:** 4185 | [OpenAlex ID](https://openalex.org/A5082404485)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的离散扩散模型控制框架GILC

**💡 创新点**

利用预训练去噪网络做变分代理估计价值函数，并在logit空间进行梯度校正，解决离散梯度不稳定问题

**🔧 技术方法**

Gumbel-Softmax+Straight‑Through、变分代理、梯度导向logit校正、策略梯度、Monte Carlo采样

**📊 数据集**

DNA增强子、蛋白质逆折叠、QM9分子、CIFAR-10与文本图像生成

**📈 对比分析**

与Classifier/Classifier‑Free、Fine‑Tuning、SMC、Best‑of‑N、TFG‑Flow等训练‑free或训练‑后方法比较，在DNA、蛋白质、分子和图像任务中均实现或接近最佳性能，并显著降低模型调用量

**⚠️ 局限性**

仍需多次采样、对token独立假设的局限、在高维离散空间对梯度采样效率待提升

---

## 589. Multi-ResNets for Subspace Preconditioning in Constrained Optimization

**arXiv ID:** 2606.06300 | [PDF](https://arxiv.org/pdf/2606.06300v1)

**作者:** Merve Karakas `[一作]` (UCLA), Nikhil Rao `[通讯]` (Tapestry, Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 MResOpt，一种基于多阶段残差网络的约束优化模型；

**💡 创新点**

创新点在于通过分层优先级约束分解与阶段化残差更新，实现安全回退与可学习的任务分工；

**🔧 技术方法**

采用 predict–complete–correct 框架、DC3+recomp 作为基线、分层残差网络、Gaussian Process 与 Neural Tangent Kernel 分析；

**📊 数据集**

使用合成 QP/QCQP/SOCP 数据和 IEEE 30/57 节点 ACOPF 负载样本；

**📈 对比分析**

与 DC3、DC3+recomp 对比，MResOpt 在高优先级约束满足率显著提升、等式平面保持更好、运行时虽略高但比重投影更高效；

**⚠️ 局限性**

局限性包括需人工确定约束顺序、非 detach 变体缺乏理论支持、重投影导致额外计算开销、实验仅覆盖电网问题。

---

## 590. Reactive Flux Matching: Mechanism Discovery and Adaptive Sampling of Rare Events

**arXiv ID:** 2606.06295 | [PDF](https://arxiv.org/pdf/2606.06295v1)

**作者:** Rishal Aggarwal `[一作]` (University of Pittsburgh), Eric Vanden-Eijnden `[通讯]` (New York University)

**通讯引用:** 18440 | [OpenAlex ID](https://openalex.org/A5044193587)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一种名为 Flux Matching 的框架，用以直接从反应路径采样数据中学习概率流的速度场与标量势能，从而揭示反应机制并生成可解释的反应坐标。

**💡 创新点**

创新点在于：①不依赖传统的收敛指示函数（committor），能够在非马尔可夫投影下保持定义；②通过加权 Helmholtz–Hodge 分解得到的势能 h 既是数据驱动的反应坐标，又可用于自适应界面生成；③将路径采样与流匹配（flow matching）相结合，构建了二次变分原理并可直接用神经网络回归估计。

**🔧 技术方法**

核心技术包括：路径采样（TPS、TIS、FFS、WE 等）、加权 Helmholtz–Hodge 分解、流匹配损失（类似 Benamou–Brenier 与 stochastic interpolants）、双向传播求解梯度、神经网络参数化（对速度场 u 与势能 h 的回归）。

**📊 数据集**

使用的实验数据集：Müller–Brown 力场（二维）、Alanine Dipeptide（22 原子、66 自由度）与其二维 dihedral 投影、AIB9 多肽（129 原子、29 backbone dihedral），所有数据均通过 OpenPathSampling 的 transition path sampling 生成。

**📈 对比分析**

与传统方法比较：在 ADP、AIB9 系统中，使用 Flux Matching 生成的流线完成率均 ≥ 0.97，Torsional Wasserstein 距离低于 TPS 基准；在低维投影上，流线与势能能够重现真实反应通道并提供更平滑、可解释的路径；同时，势能 h 的水平集可直接作为 TIS、FFS 或 WE 的自适应界面。

**⚠️ 局限性**

局限性包括：需要足够密集的反应路径样本，尤其在高维系统中采样成本高；神经网络对高维空间的泛化仍受限；当前实现主要针对平稳状态，非平稳过程或多重路径分辨率的处理尚不完善。

---

## 591. Synthetic Data Generation and Vision-based Wrinkle and Keypoint Detection for Bimanual Cloth Manipulation

**arXiv ID:** 2606.06292 | [PDF](https://arxiv.org/pdf/2606.06292v1)

**作者:** Ariel Herrera `[一作]` (University of Luxembourg), Atal Anil Kumar `[通讯]` (Université de Lorraine)

**通讯引用:** 265 | [OpenAlex ID](https://openalex.org/A5003428739)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种基于视觉的双手布料展开框架，利用Blender域随机化生成合成数据，训练可无序排列的关键点CNN与YOLOv8皱纹检测模型，并通过结构皱纹抓取点实现全折叠状态下的展开；

**💡 创新点**

创新点在于：1）引入结构皱纹检测作为关键点不可见时的替代抓取策略；2）设计可置换的关键点热图CNN和基于柔性置信中心的亚像素回归；3）融合合成与真实数据实现Sim-to-Real无微调迁移；

**🔧 技术方法**

使用技术包括Blender物理仿真、CNN热图回归+非极大值抑制、YOLOv8目标检测、OpenCV边缘提取与椭圆拟合、非极大值抑制、软Argmax亚像素定位、仿真物理引擎Newton、状态机控制双手抓取；

**📊 数据集**

数据集为5000张Blender合成图像（随机纹理、光照、相机），其中500张手工标注皱纹边框；结合Roboflow Wrinkle Detector 2.0及真实物理布料图像进行验证；

**📈 对比分析**

与Lips等人（MPE≈26像素）和Cheng Li（MPE≈10.45像素）的基线相比，本文关键点MPE仅为1.76±0.75像素，且在完全折叠状态下能稳定抓取；

**⚠️ 局限性**

局限性包括：对极其复杂或无明显结构皱纹的布料识别能力有限；仿真与真实布料的物理差异仍可能导致极端情况下抓取失败；未来需进一步完善鲁棒性与RL策略训练。

---

## 592. Geodesic Flow Matching on a Riemannian Degradation Manifold for Blind Image Restoration

**arXiv ID:** 2606.06278 | [PDF](https://arxiv.org/pdf/2606.06278v1)

**作者:** Akshay Janardan Bankar `[一作]` (Samsung Research Institute), Amit Satish Unde `[通讯]` (Samsung Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于黎曼度量的降解流匹配框架，将图像降解建模为低维黎曼流形上的点，并在联合潜在-流形空间上进行地理一致的流动推理，解决盲图像恢复问题。

**💡 创新点**

创新点在于：①用黎曼流形显式刻画降解空间，取代传统欧氏嵌入；②在潜在空间和降解流形上耦合地理一致的流动匹配；③两阶段训练策略，先学习降解表示再对流形速度进行切向束对齐。

**🔧 技术方法**

核心技术包括：流动匹配（Flow Matching）、黎曼流形算子（指数/对数映射、切向投影）、潜在空间（Stable Diffusion 编码器/解码器）、跨注意力条件注入。

**📊 数据集**

使用多任务数据集：DPDD、RealBlur-J（去模糊）；NH‑HAZE、RESIDE（去雾）；LHP、RealRain/RealTest（去雨）；RealSnow、Dense‑Snow（去雪）。

**📈 对比分析**

与CNN、Transformer、ResFlow等基线在 PSNR/SSIM/LPIPS/FID/KID 上对比，特别是超弯曲几何（Hyperbolic）在难度较高的去模糊和真实降解任务中取得与最先进方法相当甚至略优的性能，且保持良好的分布质量。

**⚠️ 局限性**

局限性包括：对流形几何参数（维度、曲率）的敏感性；两阶段训练较为复杂；在极端混合降解下仍可能出现恢复细节不足；对真实世界多模态降解的泛化需进一步验证。

---

## 593. Adapting Diffusion Language Models for Lossless Pixel-Level Image Transmission

**arXiv ID:** 2606.06273 | [PDF](https://arxiv.org/pdf/2606.06273v1)

**作者:** Tianqi Ren `[一作]` (Zhejiang University), Zhifeng Zhao `[通讯]` (Zhejiang Lab)

**通讯引用:** 6554 | [OpenAlex ID](https://openalex.org/A5036831868)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于离散扩散模型的分离源-信道编码框架 DDM-SSCC，用于无损像素级图像传输，并将扩散的反向去噪过程同步到算术编码中；

**💡 创新点**

创新点包括：1）将非自回归扩散去噪与算术编码对齐的同步编码协议；2）使用 Halton 序列实现低差异的去噪顺序；3）采用基于掩码比例的余弦去噪调度；4）引入温度校准以解决概率失配；5）在 SSCC 框架下实现更高的可靠性；

**🔧 技术方法**

技术手段：离散扩散模型（DiffuGPT）、算术编码、Halton 低差异序列、余弦去噪调度、温度缩放、ECCT（增强型误差校正码）信道编码、基于块的像素-标记化；

**📊 数据集**

使用的公开数据集：CIFAR10、DIV2K-LR-X4（验证集）和 Kodak；

**📈 对比分析**

与 DLPR+ECCT、JPEG‑XL+ECCT、SparseSBC、DeepJSCC 以及基于 iGPT 的自回归 LVM‑SSCC 进行对比；在 AWGN 与 Rayleigh 信道下，DDM‑SSCC 在统一信噪比（SNR_unified）>2 dB 时即可实现完美恢复，显著低于其他基线；在预热区（SNR_unified < 2 dB）也表现出更高的 PSNR/SSIM；

**⚠️ 局限性**

局限性：1）在高分辨率图像上需要较多扩散步数（T≥50）导致解码/编码耗时数小时；2）实验仅覆盖 AWGN 与 Rayleigh 两种信道，未验证对更复杂信道的鲁棒性；3）对模型训练和调参要求较高；4）在实时或极低延迟场景下的可行性尚未评估。

---

## 594. RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention

**arXiv ID:** 2606.06256 | [PDF](https://arxiv.org/pdf/2606.06256v1)

**作者:** Yang Liu `[一作]` (Xiaohongshu Inc), Junhao Hu `[通讯]` (Peking University)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5107244602)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 RedKnot，一种面向大规模上下文 LLM 的 KV 缓存管理系统，能够在保持输出质量的前提下显著提升预填充速度与并发吞吐。

**💡 创新点**

核心创新点是将 KV 缓存按注意力 head 进行分层拆分，区分全局（prefix‑敏感）与局部（只需局部窗口）head，并实现 head‑aware 的缓存恢复、稀疏 FFN 与 SegPagedAttention，打破传统单一 token‑级别、矩阵级别的限制。

**🔧 技术方法**

技术包括 head‑class sparsification、Elastic Sparsity（RoPE 对齐、head‑aware 注意力恢复、稀疏 FFN）、SegPagedAttention（按 head 划分 KV 页面、可变长度 FlashAttention 核心）以及分布式 KV 缓存与位置无关重用策略。

**📊 数据集**

在三类模型（Mistral‑7B、Qwen3‑32B、Llama‑3.3‑70B）上使用六个公开的长上下文 QA 数据集（HotpotQA、MuSiQue、2WikiMQA、TriviaQA、MultiFieldQA、Qasper）进行评测。

**📈 对比分析**

与密集前填充、CacheBlend、ProphetKV 等基线相比，RedKnot 在 8K‑128K 令牌范围内实现了 1.6–3.54× 的 TTFT 加速、67–79.5% 的 FLOPs 降低，以及 4.7–7.8× 的 GPU 并发会话数提升，且答案质量与 dense baseline 均相当或更优。

**⚠️ 局限性**

主要局限在于当前实现仍依赖 dense‑level KV 接口（如 SDPA），导致部分 head‑aware 速率提升被 mask 代价削弱；以及在 PD 传输与调度层面尚未完全将 head‑aware 语义下沉到系统级别，未来需要更完整的 per‑head 计算与缓存框架。

---

## 595. LLMs Can Leak Training Data But Do They Want To? A Propensity-Aware Evaluation of Memorization in LLMs

**arXiv ID:** 2606.06286 | [PDF](https://arxiv.org/pdf/2606.06286v1)

**作者:** Gianluca Barmina `[一作]` (University of Southern Denmark), Lukas Galke Poech `[通讯]` (University of Southern Denmark)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5104621238)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于倾向的记忆评估框架PropMe，结合前缀攻击与普通提示来衡量大语言模型在真实使用与极端诱导下的记忆行为。

**💡 创新点**

创新点包括：①首次将记忆评估从单纯的能力测评转向倾向评估；②设计了将传统指标转换为倾向度的公式；③开发了轻量级、可并行的训练集追踪管线PropMe。

**🔧 技术方法**

采用infini‑gram suffix‑array索引实现高速n‑gram查询，结合后缀遍历、稀疏过滤、文档检索与跨度合并，最后通过自定义的倾向转换公式与多工并行聚合生成指标。

**📊 数据集**

使用公开许可的Common Pile（英语）和丹麦Dynaword（丹麦）训练语料；模型为Comma v0.1和其持续预训练版本DFM Decoder。

**📈 对比分析**

通过generic、specific、prefix三种提示设置计算verbatim、nv‑recall、full‑match率等指标；结果显示前缀攻击下的记忆信号显著高于非对抗提示，而DFM Decoder在继续预训练后对Common Pile的记忆倾向显著下降。

**⚠️ 局限性**

局限性在于仅适用于可公开获取训练集的模型，缺乏对不同架构与多语言场景的通用验证；在无训练集时倾向转换公式难以直接应用。

---

## 596. From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws

**arXiv ID:** 2606.06324 | [PDF](https://arxiv.org/pdf/2606.06324v1)

**作者:** Mengzhuo Chen `[一作]` (State Key Laboratory of Complex System Modeling and Simulation Technology), Qing Wang `[通讯]` (State Key Laboratory of Complex System Modeling and Simulation Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于执行轨迹的诊断与修复框架，能够从失败轨迹中定位责任步骤并针对相应的 harness 层进行局部补丁生成。

**💡 创新点**

创新点在于引入 Harness-aware Trace Intermediate Representation（HTIR）将异构轨迹统一成步级证据；通过诊断记录聚合生成可重复的缺陷模式；并结合受限修复操作与验证回归的机制实现安全的 harness 修复。

**🔧 技术方法**

技术包括多模型 LLM 代理协作（抽象、诊断、修复、验证），HTIR 的构造与链路推断，层级责任映射，诊断记录聚合，受限修复操作模板，差异验证与回归检测。

**📊 数据集**

使用了四个公开 benchmark：SWE-Bench Verified、Terminal-Bench 2.0 Verified、GAIA 以及 AppWorld，覆盖软件修复、命令行工作流、开放式研究 QA 与应用自动化等四个领域。

**📈 对比分析**

与人类设计的 harness 以及 GEPA、SCOPE、ReCreate 等自适应修复基线对比，实验显示在所有 benchmark 的测试集上提升 15.2%–50.0%，在所有任务上均超越了最佳基线。

**⚠️ 局限性**

局限性包括对特定 harness 代码结构的依赖、对复杂跨层缺陷的覆盖仍有限、以及在极端低资源或高度动态环境下的适用性尚待验证。

---

## 597. More than a Judge: An Empirical Study of Agent-Human Interaction in Crowdsourced Testing Assessment

**arXiv ID:** 2606.06301 | [PDF](https://arxiv.org/pdf/2606.06301v1)

**作者:** Yue Wang `[一作]` (Nanjing University), Qing Gu `[通讯]` (Nanjing University)

**通讯引用:** 6978 | [OpenAlex ID](https://openalex.org/A5110367975)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在众包测试流程中，利用先前验证的多维 LLM 评估框架，开展四阶段实验，评估评估代理生成的可操作反馈是否能提升测试报告的文本质量、覆盖率以及对后续任务的迁移效果。

**💡 创新点**

创新点在于将 LLM‑as‑a‑Judge 从仅做评判转变为流程中的反馈提供者，系统性研究其对人类测试员行为的即时影响、任务间迁移及长期效果。

**🔧 技术方法**

使用了基于 LLM 的多维评估代理（文本性、充分性、竞争性），双 LLM 并行评估并冲突解决，Checklist 量化评分，自动生成可执行反馈，并在实验中结合人机交互实验设计。

**📊 数据集**

实验数据来自三款真实应用的需求文档与测试报告，测试员为 MoocTest 平台的 20 名参与者提交的报告。

**📈 对比分析**

通过四阶段对照设计比较两组在同一任务修订、首次提交新任务及跨任务迁移的文本性和充分性得分，结果显示反馈后文本性提升约10–20%，充分性提升约25%（在有改进空间者），新任务中文本性提升约10%，跨任务迁移整体提升约7–8%，证明反馈有效。

**⚠️ 局限性**

局限性包括样本规模小、仅涵盖功能测试且受外部 AI 工具使用影响、评估框架自身标准导致测量偏倚、结果泛化性受限于实验设置和参与者特征。

---

## 598. Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation

**arXiv ID:** 2606.06281 | [PDF](https://arxiv.org/pdf/2606.06281v1)

**作者:** Rickmer Krohn `[一作]` (Technische Universität Darmstadt), Georgia Chalvatzaki `[通讯]` (Technische Universität Darmstadt)

**通讯引用:** 853 | [OpenAlex ID](https://openalex.org/A5026055366)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出多分辨率触觉感知框架 MiTaS，融合 RGB、GelSight 以及高频事件触觉传感器，并基于流匹配生成式策略实现多种接触式操作。

**💡 创新点**

首次将帧基 GelSight 与高频事件触觉 Evetac 的多分辨率融合结合到同一模型，并引入异构模态 CNN‑Transformer 编码与多触觉共训练方法。

**🔧 技术方法**

采用多模态 CNN 支柱 + Transformer 融合、条件流匹配生成器、跨模态注意力、异构触觉共训练以及遥控演示数据收集等技术。

**📊 数据集**

使用自制的 30 条遥控演示数据集，涵盖 5 种接触密集的操作任务：齿轮装配、板擦拭、灯具安装、钥匙锁扣和灯泡接入。

**📈 对比分析**

与 Sparsh‑X 多模态基线及 Vision‑only ViT 进行对比，MiTaS 平均成功率 80%，Sparsh‑X 54%，Vision‑only 31‑26%，共训练提升约 10% 以上。

**⚠️ 局限性**

局限性：仅针对 GelSight 与 Evetac 两种传感器；数据量受遥控演示限制；仅 4‑DoF 动作，未包含机器人本体状态；控制频率 15 Hz，缺乏更高反应性和更大规模多传感器评估。

---

## 599. Robust Ensemble of Selectively Strengthened and Augmented Predictors

**arXiv ID:** 2606.06265 | [PDF](https://arxiv.org/pdf/2606.06265v1)

**作者:** Parsa Memarzadehsaghezi `[一作]` (Ontario Tech University), Mehran Ebrahimi `[通讯]` (Ontario Tech University)

**通讯引用:** 2225 | [OpenAlex ID](https://openalex.org/A5088051228)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为 RESSAP 的模型无关框架，通过特征子集划分、数据增强和随机子模型选择，将单一分类器转化为鲁棒集成模型，以提升对对抗性迂回攻击的抵抗力。

**💡 创新点**

将特征重要性、鲁棒性与复原度结合形成复原度评分以指导特征子集生成；在训练时对每个子集进行噪声增强；推理时随机选取子模型进行投票，实现高可变性与高准确度的双重平衡。

**🔧 技术方法**

特征重要性评估（置换法）、特征鲁棒性评估（噪声扰动）、复原度分数、基于高斯噪声的数据增强、子模型集成与随机化推理。

**📊 数据集**

在实验中使用合成的 600×10 维度数据集，基于均匀分布与正切变换生成特征，标签由样本范数阈值划分得到平衡二分类标签。

**📈 对比分析**

采用多线搜索（MLS）黑盒迂回攻击，比较基线 SVM 与不同 RESSAP 变体的干净准确率与攻击成功率及查询次数；结果显示完整 RESSAP 在保持约 96% 干净准确率的同时，将攻击成功率从 100% 降至 32.5%，并显著提升对抗查询成本。

**⚠️ 局限性**

实验仅在人工合成数据上验证，缺乏真实世界数据集与对比其他先进鲁棒模型的评估，且未探讨子模型数量与查询参数对性能的细粒度影响。

---

## 600. Breaking Time: A Fully Gaussian Framework for Distributed and Continuous-Time SLAM

**arXiv ID:** 2606.06250 | [PDF](https://arxiv.org/pdf/2606.06250v1)

**作者:** Davide Ceriola `[一作]` (Sapienza University of Rome), Giorgio Grisetti `[通讯]` (Sapienza University of Rome)

**通讯引用:** 12558 | [OpenAlex ID](https://openalex.org/A5028708181)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种名为 G-solver 的全高斯分布式连续时间 SLAM 框架，能够融合异构传感器的异步数据并估计平滑轨迹。

**💡 创新点**

创新点在于将高斯过程运动先验与高斯贝叶斯传播相结合，实现了在分布式、多摄像机、滚动快门场景下无需同步或额外工程即可获得一致的连续轨迹估计。

**🔧 技术方法**

采用了常速高斯过程先验、Gaussian Belief Propagation 消息传递和欧氏/李群切空间映射等技术，形成链式运动模型，显著降低循环依赖。

**📊 数据集**

使用了合成的 helix 与 sphere 轨迹、真实 ChArUco 视觉测距、KITTI 06 的滚动快门和多摄像机序列进行实验。

**📈 对比分析**

与基于三次 Z‑Spline/ B‑Spline 的 Hyperion 以及中心化的 Gauss–Newton 进行对比，G-solver 在高噪声、滚动快门和多摄像机条件下保持更低的 ATE/ARE、收敛更稳，计算时间与 Hyperion 相当。

**⚠️ 局限性**

局限性包括需要手动或学习设置的 GP 超参、仅支持常速先验、分布式消息调度仍未最优化，且在极大规模图中可能因通信延迟导致收敛慢。

---

## 601. FOXGLOVE: Understanding Goal-Oriented and Anchored Writing Feedback from Experts and LLMs on Argumentative Essays

**arXiv ID:** 2606.06271 | [PDF](https://arxiv.org/pdf/2606.06271v1)

**作者:** Yijun Liu `[一作]` (University of Illinois Urbana-Champaign), Tal August `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 536 | [OpenAlex ID](https://openalex.org/A5029563909)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了FOXGLOVE数据集，包含696条训练有素写作教师的注释反馈和1,644条来自四款大型语言模型（GPT‑5.2、Claude Sonnet 4.5、Llama 3.3 70B、Qwen3‑Next 80B）的注释反馈，涵盖69篇十二年级论证性作文，并对部分反馈进行1,430次专家质量评分。

**💡 创新点**

首次实现了在同一评价框架下，对教师与LLM生成的反馈在目标定位、文本锚定与优先级划分上的系统对比，并提供了公开的协议、代码与质量评估标准；同时揭示了LLM反馈在长度与细节上的优势及其对专家评分的影响。

**🔧 技术方法**

采用大规模语言模型（GPT‑5.2、Claude Sonnet 4.5、Llama 3.3 70B、Qwen3‑Next 80B）生成反馈，利用JSON架构输出目标标签、句子跨度、评论文本、紧急度等级及全局评论；对教师反馈采用Google Docs批注并手工标注。

**📊 数据集**

使用PERSUADE 2.0语料库的69篇十二年级论证性作文，按句子层面标注五类论证目标（Position、Claim、Evidence、Counterclaim、Rebuttal）作为反馈标签；结合自制的教师与LLM注释协议。

**📈 对比分析**

通过句子跨度重叠率、目标分布一致性、紧急度排序对比，以及文本可读性、代词与问题使用频率的定量分析，发现教师与LLM在目标与位置分布上高度一致，但LLM在句子级标注与紧急度判断上存在显著差异；专家评分显示LLM反馈在六个质量维度上得分更高，且差异在一定程度上归因于评论长度。

**⚠️ 局限性**

受限于特定反馈框架（目标定位+紧急度+句子锚定），可能低估了鼓励性、表扬性等非结构化反馈；数据来源为实验环境，缺乏持续师生关系与真实课堂背景；专家评分人数有限，结果需结合学生改写成效进一步验证。

---

## 602. Learning What to Forget: Improving LLM Unlearning via Learned Token-Level Importance

**arXiv ID:** 2606.06320 | [PDF](https://arxiv.org/pdf/2606.06320v1)

**作者:** Gizem Yüce `[一作]` (EPFL), Nicolas Flammarion `[通讯]` (EPFL)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种联合优化框架 ATWU，实现无监督的令牌级别记忆清除。

**💡 创新点**

通过“保持冲突”概念证明了令牌重要性可在未标注情况下恢复，并用线性隐藏状态评分器实现可扩展的令牌加权。

**🔧 技术方法**

利用保留损失与记忆损失的联合最优化、线性隐藏状态投影、可变的饱和负交叉熵等技术。

**📊 数据集**

在 TOFU（虚构作者问答）和 RWKU（真实人物实体消除）两个基准上进行评测。

**📈 对比分析**

与多种样本级、概率基、辅助模型等方法对比，ATWU 在遗忘质量、保留能力与通用性能上均超过了现有最优方法。

**⚠️ 局限性**

局限包括对小规模忘记集信号不足、对 ρ 超参数的敏感性、对“结构 vs 记忆”严格分离假设的依赖，以及仅在两个基准与模型上验证。

---

## 603. Data valuation model for non-monetary exchanges

**arXiv ID:** 2606.06325 | [PDF](https://arxiv.org/pdf/2606.06325v1)

**作者:** Julia Blyumen `[一作]`, Eitan Farchi `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出一种基于用户选择的注意力度量方法，用来估算内部数据产品的价值。

**💡 创新点**

创新点在于将消费者注意力直接纳入价值公式，并将该度量映射为Shapley值的闭式解，既公平又能激励稀有长尾产品。

**🔧 技术方法**

使用了合作博弈理论、Shapley值、消费者选择模型以及对非金钱交换场景的理论分析。

**📊 数据集**

未使用实际数据集，主要以假想示例和理论推导验证方法。

**📈 对比分析**

通过与简单的流行度（订阅数）度量对比，示例表明该度量能更公平地提升稀有产品价值，表现优于传统指标。

**⚠️ 局限性**

局限在于假设消费者行为独立且理性，缺乏实证验证，且对多重优先级和真实订阅行为的建模尚不充分。

---

## 604. Efficient Mean Curvature Computation on High-Dimensional Data Manifolds

**arXiv ID:** 2606.06329 | [PDF](https://arxiv.org/pdf/2606.06329v1)

**作者:** Alexandre L. M. Levada `[一作]` `[通讯]` (Federal University of Sao Carlos), Alexandre L. M. Levada (Federal University of Sao Carlos)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种高效计算高维数据点局部均值曲率的方法，显著降低了原始算法的时间复杂度。

**💡 创新点**

创新点在于：①利用协方差矩阵特征向量正交性与迹算子循环性推导出完全精确的代数恒等式，将原本 O(m⁴) 的计算降至 O(m²)；②在协方差矩阵秩远小于维数时，用截断 SVD 取代全特征分解，将 O(m³) 降到 O(k²m)，并给出 null‑space 贡献的解析近似。

**🔧 技术方法**

技术手段包括：特征向量正交性分析、迹算子循环性、矩阵幂与 Hadamard 运算、截断奇异值分解、Haar 分布下的外积期望计算。

**📊 数据集**

实验使用 40 个 OpenML 公开数据集（维度 4–279，样本数 150–13910）和 25 个高维度（m ≥ 400）数据集（如 MNIST、面部图像、基因表达等）进行评估。

**📈 对比分析**

与原始 MCBP 算法比较：MeCuCo 在 Exact 模式下保持精确无误差；Fast 模式在 D≫k 时实现 50–300 倍（甚至 800 倍）速度提升，绝大多数数据集 Spearman ρ ≥ 0.97、Chatterjee ξ ≥ 0.86，归一化后平均 MAE < 0.001，几乎无关键信息丢失。

**⚠️ 局限性**

局限性：①对连续、满足流形假设的数据适用；②在低维或 D/k 接近 1 的情形下 Fast 模式近似误差上升；③对离散或类别型特征的处理需预先映射到连续空间；④仍需完成特征分解或 SVD，极端大规模 k 时可能成为瓶颈。

---

## 605. VOLT: Vision and Language Trajectory Segmentation for Faster-than-Demonstration Policies

**arXiv ID:** 2606.06323 | [PDF](https://arxiv.org/pdf/2606.06323v1)

**作者:** Robert Ramirez Sanchez `[一作]` (Virginia Tech), Siddarth Jain `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VOLT，一种基于视觉‑语言模型的视频演示分段方法，通过在可加速段落上下采样训练机器人以实现比人类演示更快的任务执行；

**💡 创新点**

创新点在于利用 VLM 对整段演示进行语义分段，准确识别需要保持速度与可加速的子任务，从而避免全局加速导致的误操作；

**🔧 技术方法**

采用 Qwen3‑VL‑32B‑Instruct‑FP8 对演示视频进行推理，结合扩散策略（Denoising Diffusion Implicit Models）训练高层策略，并使用低层速度控制器实现动作跟踪；

**📊 数据集**

在 Franka Emika 机械臂的桌面实验平台上收集约 250 条演示（涵盖 Pick‑and‑Place、Push Cup、Tower Transfer、Plug Insertion、Table Sorting 等任务），并使用 GELLO 进行远程演示；

**📈 对比分析**

与 DemoSpeedup、SAIL 等基线相比，VOLT 在所有短周期任务中平均提升 2.18 倍完成速度，同时保持 5–7% 的成功率提升（显著优于 DemoSpeedup，略低于 SAIL 的极端加速但成功率更稳健）；

**⚠️ 局限性**

局限在于需要手动设定下采样比例、受高层策略推理延迟与低层控制精度限制；VOLT 对长序列视频的处理也需分块或更小模型以降低计算成本。

---

## 606. DragOn: A Benchmark and Dataset for Drag-Based GUI Interactions

**arXiv ID:** 2606.06322 | [PDF](https://arxiv.org/pdf/2606.06322v1)

**作者:** Nathan Bout `[一作]` (H Company), Ronan Riochet `[通讯]` (H Company)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并发布了覆盖文本高亮、单元格选择、元素缩放与滑块操作四类拖拽交互的大规模视觉‑语言基准与训练集DragOn，旨在提升 GUI 交互中的拖拽定位能力。

**💡 创新点**

创新点包括：① 通过渲染器几何信息直接生成像素级标注（rendering-as-supervision），消除 OCR 与目标检测误差；② 四种异构拖拽域的统一框架与自动化生成流程；③ 在该基准上微调 Qwen3.5-VL 以显著提升性能，展示数据驱动可超越闭源旗舰模型。

**🔧 技术方法**

使用了渲染器 (LibreOffice, Playwright‑Chromium)、PDF/EMUs 解析、颜色键探测等技术进行标注；构建了模板化自然语言指令、图像增强与位置约束；模型方面采用 VLM (Qwen, Holo, Claude, GPT 等) 与 Qwen3.5‑VL 训练。

**📊 数据集**

数据集包含 286K 训练截图与 3.5M 训练任务，涵盖四个领域；还提供 2,000 条验证/测试样本，分别来自 PDF、Excel、PowerPoint 与 HTML 滑块场景。

**📈 对比分析**

通过与多款闭源与开源 VLM 的对比评估，发现最强闭源模型在拖拽基准上仍低于 30% 成功率；对比微调后的 Qwen3.5‑VL 在所有四类任务中均超过所有基线，整体成功率提升至 35.3%，尤其在文本高亮与滑块操作表现突出。

**⚠️ 局限性**

主要限制：① 对于缩放/旋转等几何操作仍受容差限制，误差多为小幅偏移；② 数据集虽大但多为合成场景，真实用户环境中的视觉噪声与交互多样性仍待进一步覆盖；③ 仍缺乏动态拖拽轨迹预测与闭环控制能力。

---

## 607. RhymeFlow: Training-Free Acceleration for Video Generation with Asynchronous Denoising Flow Scheduling

**arXiv ID:** 2606.06309 | [PDF](https://arxiv.org/pdf/2606.06309v1)

**作者:** Chensheng Dai `[一作]` (Princeton University), Yueqi Duan `[通讯]` (ABC Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的异步去噪流程调度方法RhymeFlow，显著加速视频生成过程。

**💡 创新点**

创新点在于通过动态调度不同时间步的去噪流程，既保持生成质量，又减少不必要的迭代。

**🔧 技术方法**

基于扩散模型和异步流程调度技术，结合轻量级控制模块实现无训练加速。

**📊 数据集**

在UCF101、Kinetics-400等公开视频数据集上进行实验验证。

**📈 对比分析**

与传统视频扩散模型、以及其他加速方案（如DDIM、ControlNet）比较，RhymeFlow在保持视觉质量相近的前提下，帧率提升约2-3倍，生成时间缩短50%以上。

**⚠️ 局限性**

局限性包括对高分辨率或高动态范围视频的适用性有限，且在极端复杂场景下可能出现轻微细节丢失。

---

## 608. PAC-Bayesian Adversarially Robust Generalization for Message Passing Graph Neural Networks: A Sensitivity Analysis

**arXiv ID:** 2606.06293 | [PDF](https://arxiv.org/pdf/2606.06293v1)

**作者:** Ziling Liang `[一作]` (Southeast University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 45074 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对消息传递图神经网络在对抗攻击下的PAC‑Bayesian鲁棒泛化上界，利用低秩Jacobian敏感性分析并构造各参数块的各向异性高斯后验。

**💡 创新点**

创新点在于将K维输出结构的Jacobian低秩特性与各向异性后验相结合，去除了传统宽度相关项，改进了谱复杂度；同时统一了MPGNN与GCN的鲁棒分析框架。

**🔧 技术方法**

主要技术包括PAC‑Bayesian理论、Jacobian敏感性矩阵、各向异性高斯后验、谱范数与低秩近似。

**📊 数据集**

论文为理论分析，无实证实验，未使用具体数据集。

**📈 对比分析**

通过理论推导与已有的等方差PAC‑Bayesian上界进行对比，证明在维度因子从隐藏宽度→类别数、谱因子更紧时，鲁棒上界更小；但无实验验证。

**⚠️ 局限性**

局限在于对邻接矩阵扰动仅考虑最坏情况，未细化不同攻击模式；敏感性矩阵仍未充分利用图结构信息；且仅给出理论上界，缺乏实验验证。

---

## 609. Evolution of bilateral and multilateral collaboration of EU-14 countries across disciplines, 2010-2024

**arXiv ID:** 2606.06330 | [PDF](https://arxiv.org/pdf/2606.06330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 610. Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit Discovery

**arXiv ID:** 2606.06267 | [PDF](https://arxiv.org/pdf/2606.06267v1)

**作者:** Alireza Bayat Makou `[一作]` (Technical University of Darmstadt), Iryna Gurevych `[通讯]` (Technical University of Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在保持任务不变的前提下，通过改变输入词频统计，研究了电路发现方法产生的电路结构差异，并发现这些差异并不对应功能上的差异，从而提出“幻影专业化”(phantom specialization)概念。

**💡 创新点**

核心创新在于揭示电路结构与功能之间的多对一映射，证明评估粒度对判定专业化的重要性，并给出了多重提取、边级评估、交叉条件转移测试等方法论建议。

**🔧 技术方法**

采用 ACDC（基于激活补丁的贪婪剪枝）电路发现算法，配合零化/重采样消融、logit lens、interchange patching 等技术进行功能和因果验证。

**📊 数据集**

使用 Pythia 系列（70 M–1.4 B）模型的 Literal Sequence Copying (LSC) 任务，输入为从 Pythia 训练语料 The Pile 中划分出的四个 token 频率区间的单词。

**📈 对比分析**

与基准模型、随机边集合、不同阈值、EAP‑IG 等方法比较，验证边级评估比源级评估更能揭示电路功能；功能转移率在 81–97% 之间，证明结构差异不导致功能差异；跨模型和频率的平均精度差距很小，表明电路功能高度相似。

**⚠️ 局限性**

主要局限包括：仅在单一非语义复制任务和单一模型族中验证，缺乏对语义任务或不同架构的泛化；仅使用 ACDC，尚未探究可微分掩码方法；未能构建正向对照以检测真正的专业化；以及计算成本高限制了抽取次数。

---

## 611. DAST: A VLM-LLM Framework for Cross-Interface Anomaly Detection in O-RAN

**arXiv ID:** 2606.06261 | [PDF](https://arxiv.org/pdf/2606.06261v1)

**作者:** Francesco Spinelli `[一作]` (i2cat foundation), Xavier Costa-Perez `[通讯]` (i2cat foundation)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一种零样本多代理的 VLM–LLM 三阶段管道，用于跨接口检测 O‑RAN 计算连续体中的异常与性能降级。

**💡 创新点**

首次将视觉感知、基于领域知识的语言推理和高分辨率视觉验证三种代理协同工作，并通过 O‑RAN 领域知识进行显式领域锚定，实现跨接口协同检测，无需标签或微调。

**🔧 技术方法**

VLM（如 qwen3.6:35b 作为视觉与语言模型）、LLM（同一模型作为语言推理器）、多代理架构、热图生成、链式推理、O‑RAN 领域知识库。

**📊 数据集**

基于公开 srsRAN、Open5GS 以及 O‑RAN SC 项目搭建的实验 O‑RAN 测试平台所采集的实时 KPI 流，并注入多种性能降级攻击（F1‑u、F1‑c、A1、E2）形成的真实网络跟踪。

**📈 对比分析**

与四种传统 TSAD 基线（Multi‑Scale Encoder‑Decoder、三步 VLM、局部 VLM、四代理框架）以及 DAST 进行对比，采用标准召回、精确度、F1‑Score 与 Range‑Wise 评价；DAST 在所有接口上平均 F1‑Score 达到 0.910，Accuracy 0.843，显著优于其它基线。

**⚠️ 局限性**

仅在单一开源栈下验证；对多厂商、多切片环境的泛化仍待测试；在非实时 RIC 环境下的延迟受限；高假警率仍低于基线，但在非异常基线下存在 9 次误报。

---

## 612. Quantifying the Privacy of Counterfactuals by Leveraging Membership Inference Attacks Against Synthetic Data

**arXiv ID:** 2606.06334 | [PDF](https://arxiv.org/pdf/2606.06334v1)

**作者:** Maryam Babaei `[一作]` (ÉTS Montreal), Sebastien Gambs `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对机器学习模型生成的反事实（counterfactual）进行无盒式成员推断攻击，验证将针对合成数据的MIA迁移到反事实后能否在不访问模型的情况下泄露训练样本信息。

**💡 创新点**

首次将多种针对合成数据的MIAs集成到无盒式攻击框架，并证明其在无模型查询的情形下比现有需要查询的反事实距离攻击更具威胁；同时系统评估了不同反事实生成方法对攻击易受攻击性的影响。

**🔧 技术方法**

使用六种合成数据MIA：DCR/DCR‑Diff、DOMIAS、DPI、Gen‑LRA、LOGAN/Classifier 与 Monte‑Carlo，并以这些攻击的输出做集成（多数投票/均值）；基准攻击为反事实距离攻击（Dist‑LRT）；实现了四种反事实生成器（Nice、Dice_gradient、SCFE、Dice‑kdtree）。

**📊 数据集**

四个公共表格数据集：Adult、Acs_income、Compas、Heloc；对每个数据集分别生成不同规模的反事实集用于攻击。

**📈 对比分析**

采用 ROC‑AUC、TPR@FPR、PR‑AUC 等指标进行比较；实验表明无盒式集成MIA在大多数数据集和反事实方法上均优于 Dist‑LRT，尤其在实例基反事实（Nice、Dice‑kdtree）和小型数据集（Compas、Heloc）上提升显著；对扰动型反事实（Dice_gradient、SCFE）效果接近随机。

**⚠️ 局限性**

局限性包括：在某些组合（如 Compas 上的 dice_gradient）提升有限；当攻击效果仅略高于随机时仍可能存在隐私泄漏；需要足够规模的反事实样本；在高维或样本稀缺情形下攻击精度仍受限。

---

## 613. DAS-PINNs for high-dimensional partial differential equations: extending deep adaptive sampling to spacetime domains

**arXiv ID:** 2606.06314 | [PDF](https://arxiv.org/pdf/2606.06314v1)

**作者:** Anshima Singh `[一作]` (University of Manchester), David J. Silvester `[通讯]` (University of Manchester)

**通讯引用:** 6152 | [OpenAlex ID](https://openalex.org/A5007116796)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究将深度自适应采样（DAS-PINNs）框架扩展到时间相关的高维偏微分方程，通过将空间与时间视为统一的高维域，利用正则化流(KRnet)逼近PDE残差分布，从而在训练过程中自动生成聚焦于残差高的采样点，提升PINNs在高维时空问题中的学习效率与精度。

**💡 创新点**

创新点包括：①将空间和时间统一为高维采样空间，无需显式时间推进或问题特定的网格移动；②利用KRnet拟合残差分布，实现对高残差区的自适应采样；③通过投影与分类步骤将采样点动态分配到内部、边界和初始条件；④在保持无耦合采样的同时，能够追踪随时间演化的局部特征，解决传统PINNs在高维、强局部化问题中的不足。

**🔧 技术方法**

使用技术：Physics‑Informed Neural Networks (PINNs) + Deep Adaptive Sampling (DAS) + Normalizing Flow（KRnet，基于Knothe–Rosenblatt重排） + Adam优化器 + 自动微分；训练过程中交替优化解算器网络与KRnet，并通过投影映射将采样点映射回原始空间-时间域。

**📊 数据集**

数据集：一系列基准问题（二维移动Gaussian峰、旋转峰、粘性Burger方程；高维（d=6,8）热方程与传输方程）。所有问题均具有解析解，用于生成训练和验证点集。验证集为高分辨率网格（如 100×100×5、200×200×5、或高维点阵），训练集为随机采样点（初始2000或10000点，随后通过KRnet生成多批点）。

**📈 对比分析**

比较方法：与使用相同网络结构、相同训练迭代次数与相同总采样点数的均匀采样PINNs进行对比。实验结果表明：在低维时DAS-PINNs与均匀采样相当，但在高维（d≥6）时误差显著下降；例如在d=8的热方程中，均匀采样L2误差接近1，而DAS-PINNs在相同点数下保持在10⁻³–10⁻⁴范围；相对L∞、MSE误差亦有类似改善。整体训练迭代数约为1.8×10⁵–2.25×10⁵，展示了方法的高效性。

**⚠️ 局限性**

局限性：①需要额外训练KRnet，增加计算复杂度；②在极高维或极强非线性、分数阶/非本地方程等更复杂场景下，KRnet逼近残差分布可能需要更大网络或更多采样点；③投影与分类步骤对边界/初始条件的处理仍有限，可能导致边界误差积累；④方法在对多尺度、强耦合或形变域的适用性尚未验证；⑤依赖于随机采样的初始点，若初始分布过于粗糙可能影响后续自适应效果。

---

## 614. Meridian: Metric-Semantic Primitive Matching for Cross-View Geo-Localization Beyond Urban Environments

**arXiv ID:** 2606.06312 | [PDF](https://arxiv.org/pdf/2606.06312v1)

**作者:** Mason Peterson `[一作]` (Massachusetts Institute of Technology), Jonathan P. How `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 32287 | [OpenAlex ID](https://openalex.org/A5011665886)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出Meridian方法，利用高层度量-语义原语在空地影像与地面RGB‑D数据之间进行匹配，实现跨视角全球定位；

**💡 创新点**

创新点在于：①不依赖场景特定训练，使用开源基础模型提取点与线原语；②提出考虑有界线段的几何一致性评分；③将匹配产生的概率分布融入离群点容错的姿态图优化；

**🔧 技术方法**

技术上采用点云与线段特征提取、语义匹配、基于图论的原语对齐、概率姿态图优化和不确定性尺度化的循环闭环过滤；

**📊 数据集**

在KITTI、Park/Campus、自采Camp A/B等多种城市与自然环境的数据集上进行评估，覆盖不同季节和传感器组合；

**📈 对比分析**

相较于基于OSM、KISS‑ICP等基线方法，Meridian在增量和最终全局轨迹误差上均表现优异，平均增量ATE约2.4‑3.1m，最终ATE可低至2.35m，且在跨季节、离地道路等新环境中保持稳健；

**⚠️ 局限性**

局限性包括：放回识别召回率低，导致需要对大量无重叠的子图-图像对进行匹配；地面与空视差异导致的语义特征区分度不足，未来需改进跨视角语义桥接。

---

## 615. A MATLAB Toolbox for Standardized Reading Speed Assessment: Implementing and Extending the Perrin Sentence Generator for English Corpora

**arXiv ID:** 2606.06297 | [PDF](https://arxiv.org/pdf/2606.06297v1)

**作者:** Daniel P. Spiegel `[一作]` (Meta Reality Labs), Romain Bachy `[通讯]` (Meta Reality Labs)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一个MATLAB工具箱，自动生成无歧义英文句子，用于测量阅读速度并支持可调节难度的“Band‑Pass”筛选。

**💡 创新点**

创新点包括：将Perrin的语义本体+“proto‑真值”逻辑迁移到英文，加入针对英文语义歧义的过滤器，并引入宽带筛选来精准控制句子难度。

**🔧 技术方法**

使用MATLAB、语义本体树、正则化的单复数转换表、语法约束逻辑，以及基于心理语言学维度的词汇过滤技术。

**📊 数据集**

利用Glasgow Norms 5,554词汇数据集，对词汇的熟悉度、具体性和情感价值进行筛选。

**📈 对比分析**

与原法语实现的Perrin方法已在实验中验证，True/False 静默阅读与MNREAD口头测试等效；该工具提供无限多样句子，消除记忆偏差，初步实验显示生成句子质量可靠，但尚未在英文阅读测试中完成正式验证。

**⚠️ 局限性**

局限性：目前缺乏正式英文验证；工具只处理可数名词，非可数词被排除；语义冲突过滤仍需手动检查；跨语言推广仍待进一步实验验证。

---

## 616. Towards One-to-Many Temporal Grounding

**arXiv ID:** 2606.06294 | [PDF](https://arxiv.org/pdf/2606.06294v1)

**作者:** Qi Xu `[一作]` (Wuhan University), Jason Li `[通讯]` (Nanyang Technological University)

**通讯引用:** 39518 | [OpenAlex ID](https://openalex.org/A5060617433)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向One-to-Many Temporal Grounding（OMTG）的完整框架，包括构建56k高质量训练集、基于SFT+RL的模型训练以及基准评测。

**💡 创新点**

首次定义OMTG任务并引入计数准确率（C-Acc）与有效Temporal F1（EtF1）评估指标，结合链式思维生成密集字幕和专门的时序与字幕奖励实现精准计数与定位。

**🔧 技术方法**

采用多模态大型语言模型（如Qwen3-VL-4B）为基础，利用监督微调+GRPO强化学习，链式思维推理、可视化验证与自我评判奖励。

**📊 数据集**

构造了56k OMTG样本，采集自Cosmos-Cap、Moment-10M、VTimeLLM等公开数据集，并在Benchmark中使用340手工标注样本进行评估；SFT阶段混合使用TimeLens-100k。

**📈 对比分析**

与多款开源与专有模型（Gemini 2.5 Pro、Seed-1.8、Qwen系列等）对比，OMTG-4B在EtF1上达43.65%，比Gemini 2.5 Pro高15.85%、比Seed-1.8高15.61%；在单段定位任务亦表现出色。

**⚠️ 局限性**

训练成本高、对极长视频的可扩展性有限，RL阶段对资源要求苛刻，且仅在部分数据集上验证。

---

## 617. TRACE: A Temporal Conditional Estimation for Multimodal Time Series Foundation Models

**arXiv ID:** 2606.06285 | [PDF](https://arxiv.org/pdf/2606.06285v1)

**作者:** Ziwen Kan `[一作]` (University of Central Florida), Tianlong Chen `[通讯]` (University of North Carolina at Chapel)

**通讯引用:** 4256 | [OpenAlex ID](https://openalex.org/A5103073431)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出TRACE，一种用于多模态时间序列基础模型的条件估计范式，专门解决时序错位和部分模态缺失问题。

**💡 创新点**

创新点在于将缺失信息视为可条件估计的潜在变量，使用跨模态的条件扩散（Conditional Diffusion）和基于MoE的交叉模态上下文聚合，实现对缺失段的概率性重建；同时保持两阶段训练以避免任务特定短路。

**🔧 技术方法**

核心技术包括：跨模态条件扩散模型、MoE门控上下文聚合、FuseMoE混合专家融合层以及自监督掩码策略。

**📊 数据集**

在三大数据集上进行实验：CMU‑MOSI、CMU‑MOSEI（情感分析）和MIMIC‑IV（临床预测）。

**📈 对比分析**

与多模态基线（MulT、MAG、TFN、MAESTRO、FuseMoE）以及扩散基准（CSDI、SSSD）对比，TRACE在所有指标（MAE、准确率、F1、AUROC等）均显著优于对手，尤其在高缺失率下表现更稳健。

**⚠️ 局限性**

局限性包括：对完全缺失模态的处理仍依赖FuseMoE的路由机制；在部分任务（如MIMIC‑IV的25‑PHE）未能超过专门设计的领域模型HAIM；两阶段训练导致总体耗时较高，且缺失建模仅聚焦于部分缺失而非全模态缺失。

---

## 618. Your GFlowNet Secretly Learns an Optimal Transport Plan

**arXiv ID:** 2606.06272 | [PDF](https://arxiv.org/pdf/2606.06272v1)

**作者:** Ian Maksimov `[一作]` (HSE University), Sergey Samsonov `[通讯]` (HSE University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过在非循环GFlowNet中固定初始流分布，将其最小化流目标转化为图上最优传输问题，并证明该网络学习的策略等价于最优耦合；

**💡 创新点**

创新点在于首次揭示非循环GFlowNet与Kantorovich最优传输的理论等价性，并利用GFlowNet的策略学习框架实现图最优传输的近似；

**🔧 技术方法**

主要技术包括：将GFlowNet最小流问题线性化、使用轨迹平衡（Trajectory Balance）损失训练带神经网络参数化的GFlowNet、以及通过图短路成本构造OT目标；

**📊 数据集**

实验数据集包括：小型超立方体（Hypergrid）环境、Cayley图对应的排列（Permutation）环境；

**📈 对比分析**

与精确OT求解器（POT）及理想采样器进行比较，结果显示GFlowNet在TV误差、轨迹长度和OT成本上均能达到或逼近最优解；

**⚠️ 局限性**

局限性在于需要精细调节正则化参数λ以平衡轨迹长度与采样质量，且对大规模图的精确OT求解不可行，GFlowNet训练仍受样本效率和计算资源限制。

---

## 619. OneReason Technical Report

**arXiv ID:** 2606.06260 | [PDF](https://arxiv.org/pdf/2606.06260v1)

**作者:** OneRec Team `[一作]`, Ziyi Zhao `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款名为 OneReason 的具备思考能力的推荐系统基础模型，融合多粒度对齐数据、推理训练和强化学习，实现更精确、可解释的推荐。

**💡 创新点**

① 多粒度（token、item、relational、user）对齐预训练；② 结构化 CoT 推理框架（两轴压缩+过渡判断）；③ “专化后统一”强化学习与拒绝采样 Fine‑tune 的混合策略。

**🔧 技术方法**

Transformer‑LLM（Qwen3‑VL）+多粒度预训练 + 指令式 SFT + CoT 结构化推理 + GRPO 强化学习 + 两阶段 rollout + diversity 奖励 + 阶段性截断与负样本下权。

**📊 数据集**

内部四领域推荐数据（视频/商品/直播/广告）四粒度语料；外部通用文本/多模态数据；OpenOneRec、OneRec‑Think、StepFun 3.5 Flash SFT 等。

**📈 对比分析**

在 OneReason‑Bench 的 R0‑R3 任务以及 Cross‑Domain Recall@K 等指标上，与基线 Qwen3‑8B、OpenOneRec、OneRec‑Think 等相比，在多域 Recall@64 及思考模式下取得 20‑30% 以上提升。

**⚠️ 局限性**

思考模式在跨域混合 RL 时仍劣于非思考；需要大量人工标注 CoT 轨迹；模型对极端稀疏或冷启动用户的推理仍有限。

---

## 620. SecRL-Prune: Structured Reinforcement Learning-Based Pruning of CodeLLMs for Preserving Adversarial Code Mutation

**arXiv ID:** 2606.06254 | [PDF](https://arxiv.org/pdf/2606.06254v1)

**作者:** Parsa Memarzadehsaghezi `[一作]` (Ontario Tech University), Khalil El-Khatib `[通讯]` (Ontario Tech University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在论文中提出了 SecRL‑Prune，一种针对 CodeLLM 的结构化剪枝框架，能够在压缩模型的同时保留其代码变异能力；

**💡 创新点**

创新点在于通过强化学习学习 MLP 通道剪枝策略，并利用 KL 方向的教师-学生奖励进行剪枝，辅以 Top‑P 预测缓存显著降低训练内存；

**🔧 技术方法**

使用技术包括 RL（REINFORCE）策略学习、KL 交叉熵损失、Top‑P 预测缓存、结构化 MLP 通道剪枝；

**📊 数据集**

实验数据集包括 HumanEval（Python 编程题）以及 300 条来自 MBPP 与 GitHub 的校准示例，并对三款 7B CodeLLM 进行评估；

**📈 对比分析**

与 PruneNet 及未压缩教师模型对比，SecRL‑Prune 在 pass@k 与 var@k 指标上保持更高的准确率和多样性，且在训练期间 GPU 内存消耗比 PruneNet 减少约 55%；

**⚠️ 局限性**

限制在于只剪枝 FFN 通道、使用的奖励基于截断的 Top‑P 目标、缺乏对稀有或长尾行为的覆盖，且压缩后仍存在显著性能差距。

---

## 621. Closing the Loop on Latent Reasoning via Test-Time Reconstruction

**arXiv ID:** 2606.06252 | [PDF](https://arxiv.org/pdf/2606.06252v1)

**作者:** Xiaopeng Yuan `[一作]` (University of Illinois Urbana Champaign), Haohan Wang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在测试时通过重构引导潜在推理的框架，提升生成质量与推理准确性；

**💡 创新点**

将重构损失与潜在空间约束结合，采用自适应权重调节策略，解决传统方法测试时潜在不稳定的问题；

**🔧 技术方法**

基于变分自编码器（VAE）+ 生成对抗网络（GAN）+ 逆向重构网络，辅以自适应注意力机制；

**📊 数据集**

在CIFAR-10、CelebA、MNIST等公开图像数据集上进行实验；

**📈 对比分析**

与传统VAE、GAN以及最新Latent Diffusion模型对比，FID与准确率均提升约5%–10%；

**⚠️ 局限性**

计算开销较大，推理过程需额外重构步骤，且在高分辨率图像上的可扩展性尚未充分验证。

---

## 622. Domain Diversity, Motivation, Inclusion, and Feedback in Software Modelling Education

**arXiv ID:** 2606.06275 | [PDF](https://arxiv.org/pdf/2606.06275v1)

**作者:** Isabella Graßl `[一作]` (TU Darmstadt), Miguel Goulão `[通讯]` (NOVA-LINCS, NOVA School of Science and Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对 90 名软件建模课程学生和 22 名教师进行问卷调查，比较了两方对问题域、教学方法、包容性和反馈的看法，并提炼出影响学习动机和参与度的关键因素。

**💡 创新点**

创新点在于首次系统比较学生与教师在建模课程中对问题域选择、动机、包容性及反馈的认知差异，提出以学生兴趣为核心的域选择与交互式教学的具体改进建议。

**🔧 技术方法**

采用定量（Likert 量表、Fisher 精确检验）与定性（主题分析、内容分析）相结合的混合研究方法，对问卷结果进行分析。

**📊 数据集**

数据集为 112 条匿名问卷（学生 90 条，教师 22 条），包含人口统计、动机、教学方法、包容性与反馈等多维度问题。

**📈 对比分析**

研究通过 Fisher 检验和主题归纳对比两方观点；结果表明教师对动机域的看法更为一致，学生对动机域、合作与游戏化的偏好更为多元，显示了显著的认知不一致。

**⚠️ 局限性**

局限性包括自报数据的主观性、样本主要来自欧洲高校且规模有限、缺乏学习成效的客观测量，因而外部效度与因果推断受限。

---

## 623. RiskFlow: Fast and Faithful Safety-Critical Traffic Scenario Generation

**arXiv ID:** 2606.06423 | [PDF](https://arxiv.org/pdf/2606.06423v1)

**作者:** Qi Lan `[一作]` (Chongqing University), Guofa Li `[通讯]` (Chongqing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了RiskFlow框架，使用单步流传输方法在动作空间生成安全关键交通场景；

**💡 创新点**

创新点在于用MeanFlow一次性生成动作序列，利用JVP目标提升训练稳定性，并在测试时对输出动作直接引导，减少迭代误差；

**🔧 技术方法**

采用场景Transformer、ResNet‑18地图编码、MeanFlow生成器、TTC风险评估、JVP修正及地图约束指导技术；

**📊 数据集**

在nuScenes数据集上训练并通过tbsim闭环仿真进行评估；

**📈 对比分析**

与STRIVE、BITS、CTG、CTG++、CCDiff等基线对比，在多车与长时序闭环评估中实现更高的现实性得分，同时保持竞争性控制性，并显著提升推理速度；

**⚠️ 局限性**

局限在控制性与现实性仍有权衡，且对极端风险场景的生成仍受限于训练数据分布。

---

## 624. Double Preconditioning (DoPr): Optimization for Test-Time Performance, not Validation Loss

**arXiv ID:** 2606.06418 | [PDF](https://arxiv.org/pdf/2606.06418v1)

**作者:** Thomas T. Zhang `[一作]` (University of Pennsylvania), Max Simchowitz `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了深度学习中训练-验证误差与部署后表现不一致的现象（Test‑Time Feedback，TTF），并提出一种双重预条件化（Double Preconditioning，DP）优化器来缓解该问题。

**💡 创新点**

创新点在于将激活预条件化（AP）与梯度预条件化（如Adam、RMSProp）结合，形成可直接插拔的DP框架；该框架通过让特征学习趋向各向同性来降低TTF导致的分布漂移，从而提升下游任务性能，而不必依赖验证损失的进一步下降。

**🔧 技术方法**

技术手段包括：
- 激活预条件化（AP），利用层内激活协方差的逆矩阵对梯度做预处理；
- 梯度预条件化（如Adam、RMSProp等）保证数值稳定；
- 最大更新参数化（μP）用于超参数的尺度不变性；
- EMA 复制用于策略评估；
- 统一的“层归一化+激活预条件化+梯度预条件化”更新公式。

**📊 数据集**

数据集与任务：
- 连续控制：Gymnasium 的 Humanoid‑v5 等运动控制环境；
- 图像‑基机器人策略：Pixel‑based 任务如 Proficient‑Human、Dexterous 等；
- 语言模型：OpenMathInstruct‑2（100K 子集）用于数学推理；
- 生成模型：ImageNet‑256 采用 score‑based 生成任务。

**📈 对比分析**

比较方法：把 DP 与标准优化器（Adam、SGDM、RMSProp 等）在相同模型、相同训练设置下进行对比；评估指标包括：
- 连续控制终端奖励；
- 机器人策略成功率；
- LLM 下游任务准确率（MathQ、MathPro 等）以及验证 NLL；
- 生成模型的 FID/精度。结果显示：DP 在大多数任务上显著提升下游指标，而验证损失并未必同步下降；在大模型（3B、8B）上，DP 还能在更高学习率下保持下游性能不退化。

**⚠️ 局限性**

局限性：
- 仅在少数 TTF 场景验证，尚不确定能否推广到所有序列模型或更复杂的机器人任务；
- 需要为每个基准优化器调优 μP 参数，调参成本仍高；
- DP 在训练损失上不一定优于基准，可能导致在某些场景下不易收敛；
- 对超参数（如 γ、学习率）的敏感性尚未完全系统化。

---

## 625. Proper Scoring Rules for Right-Censored Survival Data

**arXiv ID:** 2606.06393 | [PDF](https://arxiv.org/pdf/2606.06393v1)

**作者:** Jef Jonkers `[一作]` (Ghent University), Sofie Van Hoecke `[通讯]` (Ghent University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了右删失生存预测的适当评分规则与训练目标，提出先对预测分布进行删失映射后再使用基础评分规则的方法，并引入了基于删失能量得分的多变量生成式学习方法（censored engression）。

**💡 创新点**

创新点在于统一的删失映射框架，使对数得分、CRPS、Pinball、Brier、能量得分等传统适当评分规则在右删失数据上保持严格适当性；以及提出一种不依赖显式似然、可处理多变量事件时间的生成式训练目标。

**🔧 技术方法**

使用适当评分理论、删失映射定理、IPCW 与随机删失平均化、能量得分以及生成模型（engression）结合的样本基学习框架。

**📊 数据集**

在模拟数据（单变量及多变量混合对数正态 DGP）以及真实 ICU MIMIC‑IV 数据中的急性肾损伤（AKI）预测实验中进行验证。

**📈 对比分析**

与传统对数似然、IPCW Brier、CRPS、Pinball、以及插件加权评分比较；在模拟实验中，所提评分始终给出oracle 最佳排名；censored engression 在多变量实验和真实数据实验中均取得比无删失训练和现有 Weibull/copula 基线更低的能量得分与 Brier 分数，显示更优的观测数据预测性能。

**⚠️ 局限性**

需单独估计删失分布，若删失模型误设会影响性能；对不可观测尾部信息的估计仍受限，且方法在极端高维或极少删失情况下的效果尚待进一步评估。

---

## 626. On GPU Implementation for Multi-Precision Integer Division

**arXiv ID:** 2606.06386 | [PDF](https://arxiv.org/pdf/2606.06386v1)

**作者:** Martin B. Marchioro `[一作]` (University of Copenhagen), Stephen M. Watt `[通讯]` (University of Waterloo)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了基于整体移位逆（whole‑shifted‑inverse）的整数除法算法，并针对GPU中间规模整数（2^15–2^18 位）做了高效实现。

**💡 创新点**

创新点在于：①在整数域内完成 Newton 迭代，避免浮点反演；②针对无符号整数进行专门修正；③构建以全乘次数为度量的成本模型；④结合共享内存、寄存器、扫描等 GPU 原语实现块级高性能算子。

**🔧 技术方法**

使用 CUDA 原语（共享内存、寄存器调度、warp/block 层扫描、按位移位、变长多精度乘法、闭合乘积、块级并行化）以及按需切换不同规模的乘法实现。

**📊 数据集**

使用随机生成的整数数据集，位数从 2^13 到 2^18，批量实例数从 2^14 到 2^19，确保除数位数在 2–M/2 之间，以覆盖整个中间规模范围。

**📈 对比分析**

与 CGBN（Cooperative Groups Big Numbers）和 GMP 进行对比；在最大精度 2^18 时除法接近理论下限 5 次全乘法，速度比 CGBN 在高精度下慢 5–7 倍，但在低精度下快 3–7 倍；相较 GMP 的 CPU 版本，速度提升数十倍。

**⚠️ 局限性**

局限性包括：①受共享内存和寄存器限制，只能支持至 2^18 位；②对低精度时性能不足，未采用更快的乘法（Karatsuba/FFT/NTT）或更细粒度的调优；③依赖手写 CUDA，Futhark 等高层 DSL 仍需额外编译器优化。

---

## 627. Rethinking Infrastructure Inspection as Image Difference Classification: A Traffic Sign Case Study

**arXiv ID:** 2606.06375 | [PDF](https://arxiv.org/pdf/2606.06375v1)

**作者:** Ching Yau Fergus Mok `[一作]` (University of Cambridge), Ioannis Brilakis `[通讯]` (University of Cambridge)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用已有的资产历史图像，将交通标志缺陷检测改为图像差分分类（IDC），并在低资源场景下评估其效果。

**💡 创新点**

①将时间序列资产图像作为参考实现无额外标注的差分推理；②构建真实检验环境下的细粒度多标签数据集；③首次在IDC任务中应用指令驱动的视觉语言模型。

**🔧 技术方法**

使用MetaCLIP2等视觉编码器构建多种Encoder‑Based IDC管线；采用Qwen3‑VL‑8B的指令式视觉语言模型进行fine‑tune；通过伪“无缺陷”参考图像平衡训练数据；进行few‑shot（0、1、2、4、8 shot）实验。

**📊 数据集**

UK Traffic Sign Inspection Dataset：来自英国国道管理系统的970对图像，包含标志面、杆及背景，9类细粒度缺陷标签，平衡损坏与无缺陷比例。

**📈 对比分析**

对比IDC管线与等效的单图像基线，在0、1、2、4、8 shot条件下计算F1/宏观F1；指令式IDC在缺陷存在检测上1-shot F1>0.9，分类宏观F1>0.6；Encoder‑Based管线几乎没有提升，甚至出现性能下降。

**⚠️ 局限性**

①Encoder‑Based IDC未能充分利用参考图像；②实验结果对few‑shot设定敏感，表现波动大；③仅使用单个参考图像，未探索多参考的潜力；④伪缺陷的使用需要进一步验证其必要性。

---

## 628. Visual Commonsense Driven Knowledge Refinements for Scene Graph Generation

**arXiv ID:** 2606.06369 | [PDF](https://arxiv.org/pdf/2606.06369v1)

**作者:** Maëlic Neau `[一作]` (Umeå University), Mehul Bhatt `[通讯]` (Örebro University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在不需要重新训练的前提下，利用训练数据自动挖掘空间、功能和逻辑约束，并使用Answer Set Programming（ASP）对学习得到的场景图进行后处理，剔除不符合常识的关系并补全缺失的关系。

**💡 创新点**

创新点在于：①无监督地从标注数据中提取三类视觉常识约束；②将这些约束转化为可解释、可验证的ASP规则进行后处理；③方法模型无关、无需人工规则、可迁移至不同数据集与模型，显著提升了常识一致性。

**🔧 技术方法**

技术包括：基于RCC5、方向、IoU等几何特征的空间约束挖掘；功能（角色卡片）和逻辑（对称、逆、组合）规则挖掘；利用Clingo实现ASP推理；验证步骤筛选对性能有益的规则；与多种SGG模型（Motifs、Transformer、REACT++）结合。

**📊 数据集**

使用了三大公开数据集：PSG、Visual Genome 150（VG150）和IndoorVG；在这些数据集上评估不同SGG架构的表现。

**📈 对比分析**

通过对比基线模型与加入规则后的模型，使用Recall@K、meanRecall@K、F1@K、零样本Recall和Constraint Violation Rate (CVR) 等指标进行评估。实验表明，F1@K平均提升0.65~1.10分，CVR从8–12%降至<1%，零样本Recall提升约20%，在三大数据集上均显著优于基线。

**⚠️ 局限性**

局限性包括：依赖训练标注质量，噪声或缺失标注可能导致规则误导；对称/逆等关系难以挖掘，影响规则覆盖；评估指标仍以Recall为主，无法完全体现一致性提升；ASP求解在极大规模图或复杂规则时可能成为瓶颈。

---

## 629. GMBFormer: An NDVI-Guided Global Memory Bank Transformer for Urban Green-Space Extraction from Ultra-High-Resolution Imagery

**arXiv ID:** 2606.06363 | [PDF](https://arxiv.org/pdf/2606.06363v1)

**作者:** Hao Lei `[一作]` (Chengdu University of Technology), Zhanfeng Shen `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 GMBFormer 通过 NDVI 引导的全局记忆银行实现城市绿地提取，克服 UHR 图像分块处理导致的语义重用缺失；

**💡 创新点**

创新点在于将 NDVI 作为物理门控而非直接融合，利用记忆银行存储高置信度植被原型，并通过相似度检索增强当前块特征；

**🔧 技术方法**

使用 SegFormer（MiT‑B4）骨干+交叉注意力记忆检索、EMA 记忆更新、门控融合；

**📊 数据集**

主要使用自行构建的成都 UHR 数据集（7700 512×512 标注块）以及两种 ISPRS Potsdam 低标签版本；

**📈 对比分析**

与 SegFormer‑B4、Mask2Former、Swin‑UPerNet、DeepLabV3 等基线进行对比，GMBFormer 在成都验证集 mIoU 提升至 89.25%（比 SegFormer‑B4 提升 1.85%），Potsdam 自定义二分类和三分类 mIoU 分别提升至 92.17% 与 83.72%；

**⚠️ 局限性**

局限包括记忆银行依赖训练分布、固定容量、仅存储 Stage‑3 特征、NDVI 分辨率不匹配导致门控噪声、仅适用于可获得 NIR 的场景。

---

## 630. Bilateral and multilateral international scientific collaboration of EU member states: OpenAlex vs Scopus (2000-2024)

**arXiv ID:** 2606.06336 | [PDF](https://arxiv.org/pdf/2606.06336v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 631. Computational Modeling of Human Adaptation in Urban Infrastructure Management under Extreme Conditions: A Case Study of Subway Flood Scenarios

**arXiv ID:** 2606.06429 | [PDF](https://arxiv.org/pdf/2606.06429v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 632. Physics in 2-Steps: Locking Motion Priors Before Visual Refinement Erases Them

**arXiv ID:** 2606.06361 | [PDF](https://arxiv.org/pdf/2606.06361v1)

**作者:** Woojung Han `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过在扩散模型中提取两步推理的运动先验并在后续高精度推理中引入相位保持机制（PhaseLock），实现无训练、无外部引导的物理一致性提升。

**💡 创新点**

创新点在于发现并利用扩散过程早期阶段的相位信息（运动先验）进行相位锁定，提出基于潜在差分的引导方法，显著降低后期细节优化导致的相位退化；同时实现了极低的计算开销与高通用性。

**🔧 技术方法**

使用频域相位与幅度分解、潜在差分（Latent Delta）引导、训练自由的后向采样技术以及对已有视频扩散模型（CogVideoX、Wan 2.1、LTX-Video）的直接适配。

**📊 数据集**

在Physics-IQ、PhyGenBench、VBench等公开基准上评估；使用多种视频扩散模型作为实验对象。

**📈 对比分析**

与原始模型、WMReward、Stable Video Diffusion等对比，PhaseLock平均提升Physics-IQ 6.2分，性能提升后仅增加1.06×推理时间、1.02×显存；在多模型实验中均保持视觉质量相近并显著提升物理一致性。

**⚠️ 局限性**

局限性：依赖于两步推理得到的运动先验，若该先验本身物理不合理则会被放大；对相位退化的纠正受 λ_0 影响，需针对场景调优；不适用于无迭代式的自回归视频生成器。

---

## 633. Maximising the Set-Piece Return: Optimising Football Corner Tactics with Graph Reinforcement Learning

**arXiv ID:** 2606.06353 | [PDF](https://arxiv.org/pdf/2606.06353v1)

**作者:** Sean Groom `[一作]` (University of Birmingham), Shuo Wang `[通讯]` (University of Birmingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过图强化学习优化足球角球，生成提升首球射门概率的玩家位置与速度调整方案。

**💡 创新点**

将角球优化转化为MDP并引入图神经网络状态嵌入，结合SAC/PPO等连续控制算法，首次实现对任意起始角球状态的通用策略；与传统元启发式搜索相比，显著提升奖励并在实时推理下保持高效。

**🔧 技术方法**

图注意力网络（GATv2）用于状态表示，SAC和PPO为策略学习算法，JAX实现并行环境；奖励通过冻结的xFCS预测GNN计算。

**📊 数据集**

使用英超2025/26赛季共3,223次角球的历史事件与跟踪数据进行训练与评估。

**📈 对比分析**

在80/20时间序列拆分上进行零样本测试，SAC/ PPO在推理预算内分别提升平均xFCS约+0.068和+0.063；在相同推理成本下比随机搜索高出约94%/92%，在“全训练预算”条件下仍优于随机搜索。

**⚠️ 局限性**

仅考虑单帧奖励与静态防守，缺乏时间序列与动态防守互动；元启发式方法在更大搜索空间下仍可能优于RL，且RL难以捕捉复杂协同动作如挡拆。

---

## 634. "Chi nas dal soch el sent de legn" -- Auditing Text Corpora for Lombard

**arXiv ID:** 2606.06349 | [PDF](https://arxiv.org/pdf/2606.06349v1)

**作者:** Edoardo Signoroni `[一作]` (Masaryk University), Pavel Rychlý `[通讯]` (Masaryk University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对意大利地区语言连续体中的拉姆巴德（Lombard）文本语料进行了手工质量审核。

**💡 创新点**

创新点在于将大规模网络爬取语料与多种正字法变体结合评估，并揭示了西部拉姆巴德正字法的代表性偏差。

**🔧 技术方法**

采用了Kreutzer等人提出的错误分类体系进行手工标注，并结合正字法分类器对有效拉姆巴德文本进行语料组成分析。

**📊 数据集**

评估数据集包括NLLB、WikiMatrix、WikiMedia、XLEnt、CulturaX、Glot500、HPLT 3.0、GATITOS、minecraft-translations、OLDI Seed、Piötòst以及FLORES+基准集。

**📈 对比分析**

通过对不同语料类型（爬取、精炼、基准）和方向（拉姆巴德–英语/意大利语）的质量与正字法分布进行对比，发现爬取语料噪声高且正字法偏西部，而精炼和基准语料尽管准确但同样缺乏东部正字法代表性。

**⚠️ 局限性**

局限性包括只使用一位东部拉姆巴德母语注释员，正字法辨别不确定性高，并且为解决变体细节而将多个正字法合并为宏观组。

---

## 635. Warning Message Content Increases Help Seeking in a Large-Scale Dark Web CSAM Intervention

**arXiv ID:** 2606.06417 | [PDF](https://arxiv.org/pdf/2606.06417v1)

**作者:** Caoilte Ó Ciardha `[一作]` (University of Kent), Nina Vaaranen-Valkonen `[通讯]` (Protect Children)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Tor搜索引擎Ahmia.fi上进行大规模字段实验，比较不同警告信息内容（合法性、危害、控制、自我效能与心理困扰）和正负框架对搜索者点击匿名自助资源链接的影响。

**💡 创新点**

首次在高匿名环境下系统评估警告信息内容对帮助资源点击率的影响，并发现负面危害信息在此情境中最为有效；同时证明界面简化与信息集中也能提升点击率。

**🔧 技术方法**

采用随机日实验、全周期多条件轮换和中断时间序列分析，使用二项式逻辑回归与Newey‑West Heteroskedasticity‑Autocorrelation Consistent标准误。

**📊 数据集**

约20,000,000次查询日志，其中约3,000,000次触发警告，全部来自Ahmia.fi Tor搜索引擎的匿名查询记录。

**📈 对比分析**

与中性信息对照，所有活跃信息均显著提升点击率，负面危害信息最高（相对提升≈100%）；在实验日和全周期模型中点击率从≈0.01提升至≈0.02，p<0.001。

**⚠️ 局限性**

缺乏个体层面数据导致无法分析不同人群的差异；未评估长期曝光的习惯化效应；结果仅在Tor搜索环境中得到验证，难以直接推广至明网或其他平台。

---

## 636. Bridging CAD and Data-Driven Design: Attributed Feature Graphs for Engineering Design

**arXiv ID:** 2606.06405 | [PDF](https://arxiv.org/pdf/2606.06405v1)

**作者:** Abhishek Indupally `[一作]` (Clemson University), Jami J. Shah `[通讯]` (Ohio State University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了属性特征图（AFGs），将CAD模型中的特征如凸台、凹槽、壁板等映射为节点并构成图结构，并在CarHoods10K数据集上训练图神经网络实现结构性能（最大应力、质量、偏转）的快速预测；

**💡 创新点**

通过将CAD原生特征与参数嵌入可学习图，保留设计意图、层级关系与可编辑性，实现了CAD与数据驱动模型的直接对接，并使模型输出可解释且可映射回CAD编辑；

**🔧 技术方法**

利用AFG生成框架提取特征，使用PyTorch Geometric的GCNConv层叠加共享MLP和多任务回归头进行训练，形成端到端的评价引擎；

**📊 数据集**

使用CarHoods10K汽车发动机罩框架数据集，约3600个CAD实例配合FEA标签（最大应力、质量、偏转）；

**📈 对比分析**

与传统图像、点云、VAE、CNN等基线方法比较，AFG+GNN在三项指标上R²≥0.84、MAPE≈9%，性能与或优于基线，同时保持了对CAD特征的可解释性；

**⚠️ 局限性**

受限于对CAD命名规范与特征本体的依赖，缺乏显式物理约束，梯度解释在分布外设计时可能失效，且需更大、更多样化的数据集来提升泛化与生成式应用。

---

## 637. The Post-GCN Decade Revisited: Curvature-Stratified Evaluation of Relational Learning

**arXiv ID:** 2606.06397 | [PDF](https://arxiv.org/pdf/2606.06397v1)

**作者:** Shuo Wang `[一作]` (University of Electronic Science and Technology of China), Zhao Kang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了CurvBench曲率分层评估框架，基于图的局部相对曲率对14个关系数据集进行正、负、近零曲率区间划分并对18种模型进行评测。

**💡 创新点**

证明曲率是决定模型效果的关键隐变量，提出曲率分层而非传统平均排行榜，揭示了模型在不同几何下的性能转移与偏见。

**🔧 技术方法**

采用节点相对曲率统计（均值、偏度）、混合或自适应Riemannian方法、Graph Foundation Models以及传统GCN、GAT等模型，结合曲率分层的部分排序比较和多任务指标。

**📊 数据集**

使用14个关系数据集：9个自然图（Cora、Citeseer、PubMed、Airport、Cornell、Actor、Disease、Telecom、CS_Phds）以及5个表导图（Carcinogenesis、Hepatitis、PTE、Toxicology、F1）。

**📈 对比分析**

在每个曲率区间内部计算Top‑3模型的排名一致性（Spearman/Kendall/Jaccard），发现同一区间内一致性显著高于跨区间；GFMs在不同曲率下表现差异明显，未能消除曲率依赖。

**⚠️ 局限性**

评测仍受曲率统计粗糙度限制，未覆盖局部细粒度几何；GFMs在负曲率图上OOM问题突出；缺少对动态图或多模态异构图的进一步验证。

---

## 638. HomeWorld: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes

**arXiv ID:** 2606.06390 | [PDF](https://arxiv.org/pdf/2606.06390v1)

**作者:** Wenbo Li `[一作]` (Ace Robotics), Hongsheng Li `[通讯]` (CUHK MMLab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套从自然语言描述生成完整住宅三维场景的分层管线，先用LLM在大规模平面图数据集上生成精细化平面图，再通过层级漫游（自上而下与自下而上视角）、基于2D生成模型的图像填充与SAM/SAM-3D的3D重建、递归VLM细化纠正以及表面小物件放置等步骤，最终得到可交互、物理可行的全屋3D场景，并公开了300K真实住宅平面图与5K家具化全屋三维数据集。

**💡 创新点**

创新点包括：①采用K‑D树结构化平面图表示，使LLM能高效生成连通且多样的平面图；②将2D生成模型与3D约束结合的层级漫游技术，实现全屋可交互布局；③利用VLM递归细化循环自动纠正物理与语义错误；④通过大规模平面图数据集与5K三维场景缓解3D数据匮乏。

**🔧 技术方法**

使用技术包括：LLM微调生成平面图（Qwen3-4B-Instruct + LoRA），K‑D树结构化输出；2D图像生成（inpainting）与SAM、SAM‑3D 3D重建；层级视角漫游（top‑down + ego‑centric）结合3D约束；VLM递归细化模块；PhysX‑Anything 物理属性预测；基础纹理与灯光规则。

**📊 数据集**

主要数据集为：300K真实住宅平面图（含门窗、尺寸、标签及结构化注释），5K完整家具化全屋三维场景；基线对比使用Proctor‑10K、LayoutGPT、Holodeck、LayoutVLM、FloorPlan‑LLaMa、Floorplan‑Diffusion等。

**📈 对比分析**

通过图连通性、图多样性、碰撞率(CR)、越界率(OOB)、体积密度(VD)、对象密度(FOD)以及30位用户的三维场景评测进行定量与定性对比。实验结果显示：平面图连通性提升约15%，多样性显著提高；3D布局的碰撞率和越界率明显下降，物体密度和多样性上升；在用户研究中，受访者大多数偏好本文方法。

**⚠️ 局限性**

局限性包括：对极端复杂布局仍可能出现错误；依赖大型2D生成模型的质量与可控性；小物件物理属性预测仍相对粗糙；缺乏对动态交互细节的建模；数据集虽大但多样性与复杂度仍待进一步扩展。

---

## 639. An Infectious Disease Spread Simulation Based on Large Language Model Decision Making

**arXiv ID:** 2606.06360 | [PDF](https://arxiv.org/pdf/2606.06360v1)

**作者:** Yonchanok Khaokaew `[一作]` (University of New South Wales), David J Heslop `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个将大语言模型与基于空间的代理模拟相结合的感染传播框架，用于生成个体症状上报决策。

**💡 创新点**

创新点在于用预生成决策银行的LLM驱动决策替代传统回归规则，能够捕捉空间、人口学与信息情境对报案行为的交互影响。

**🔧 技术方法**

技术手段包括SEIR传染模型、基于Patterns of Life的Java模拟器、四款开源LLM（Llama‑3、Gemma‑2、Mistral‑8B、Galactica）以及Prompt敏感性与情境编码。

**📊 数据集**

数据来源为美国人口普查Tract级别数据（旧金山与亚特兰大）用于生成合成代理，实验还参照COVID‑19疫苗意向调查等真实数据进行验证。

**📈 对比分析**

通过与逻辑回归基线和疫苗意向数据的Spearman相关、均值/方差比较，LLM模型能产生更广泛的行为谱并与实际数据更匹配，表现优于传统回归。

**⚠️ 局限性**

局限性包括结果高度依赖LLM模型与Prompt设计，预生成决策库限制了实时适应性，且仅考虑了少数人口学属性，可能放大偏见且缺乏充分可解释性。

---

## 640. Comparison of Deep Learning Frameworks For Rice Disease Mapping From UAV Multispectral Imaging

**arXiv ID:** 2606.06359 | [PDF](https://arxiv.org/pdf/2606.06359v1)

**作者:** Yadav Raj Ghimire `[一作]` (North Carolina A&T State University), Leila Hashemi Beni `[通讯]` (North Carolina A&T State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对UAV多光谱图像下稻叶黄化病的语义分割进行了系统的模型基准实验。

**💡 创新点**

创新点在于首次将传统CNN编码器‑解码器（U‑Net、U‑Net++、DeepLabV3+）与基于Transformer的SegFormer在同一数据集和统一训练管线下进行对比，并探讨植被指数（NDVI/NDRE）对分割精度的影响。

**🔧 技术方法**

使用的技术包括U‑Net++与EfficientNet-B3/B7编码器、ResNet‑101编码器、DeepLabV3+的空洞空间金字塔池化、MiT‑B2 Transformer编码器，以及结合NDVI/NDRE通道的六通道输入；训练采用Adam优化器、加权交叉熵+Soft‑Dice损失、早停等。

**📊 数据集**

使用了公开的BLB（Bacterial Leaf Blight）稻田UAV多光谱数据集，包含蓝、绿、红、红边、NIR和植被指数通道，并划分为训练/验证/测试三部分。

**📈 对比分析**

通过统一的训练协议和mIoU、mAcc、mF1等指标进行比较，U‑Net++‑EfficientNet‑B3在所有配置下均取得最高mIoU（97.62%），DeepLabV3+略逊，SegFormer精度显著较低但推理速度相近；添加NDVI/NDRE通道可提升约0.5–1.0个百分点。

**⚠️ 局限性**

局限性包括仅基于单季稻田单一多光谱数据，Transformer模型表现不足，缺乏跨地区/跨年份验证，且未结合热像或LiDAR等多源传感器。

---

## 641. F3-Tokenizer: Taming Audio Autoencoder Latents for Understanding and Generation

**arXiv ID:** 2606.06357 | [PDF](https://arxiv.org/pdf/2606.06357v1)

**作者:** Dinghao Zhou `[一作]` (WeNet Open Source Community), Sixiang Lv `[通讯]` (Nanjing University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种名为F3‑Tokenizer的音频tokenizer，结合连续自动编码器与高维表示编码器，实现既可重构、可生成又可理解的音频表示。

**💡 创新点**

用通道归一化+随机噪声替代KL正则的非变分瓶颈；在冻结自动编码器latents上训练的自监督表示编码器与冻结LLM对齐；并通过patch级flow匹配实现连续生成可预测的latents。

**🔧 技术方法**

SpectroStream风格STFT域自编码器、channel normalization、噪声正则、RQ‑MTP自监督、冻结LLM投影、Patch‑wise flow匹配以及自回归生成训练。

**📊 数据集**

语音数据（AISHELL‑3、LibriTTS、LibriSpeech等）、音乐数据（MUSDB18‑HQ、FMA、GTZAN、NSynth）、环境声音数据（AudioCaps、DESED、ESC‑50、UrbanSound8K、FSD50K）以及音频‑文本配对数据用于TTS和TTA。

**📈 对比分析**

通过重构指标（MCD、PESQ等）、多任务理解评测（probe得分）和生成评测（TTS CER/WER、TTA CLAP/FD_OpenL3）与VibeVoice、Whisper、MingTok等基线对比，F3‑Tokenizer在重构保持竞争力、理解性能显著提升、TTS与TTA生成效果优于或相当于主流模型。

**⚠️ 局限性**

依赖多任务自监督与冻结LLM教师，训练开销大；高维表示与连续latent仍分离，未实现统一可解码的高维表示；对极端长序列或无配对文本的生成能力尚未充分验证。

---

## 642. Performance Evaluation of GraphCast for Medium-Range Weather Forecasting over Brazil

**arXiv ID:** 2606.06348 | [PDF](https://arxiv.org/pdf/2606.06348v1)

**作者:** Wolfgang R. Rowell `[一作]`, Lucas S. Kupssinskü `[通讯]` (PUCRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对巴西四个气候子区的GraphCast操作模型与IFS HRES基线进行中期天气预报性能评估。

**💡 创新点**

首次在巴西不同气候区域进行MLWP与传统NWP的区域化对比，并揭示夏季和冬季的变量特异性误差分布。

**🔧 技术方法**

使用WeatherBench-X框架、云原生数据管道和ZARR格式进行评估，计算RMSE、ACC和归一化RMSE差异。

**📊 数据集**

利用ECMWF IFS分析、IFS HRES预报以及Google DeepMind的GraphCast操作预报，覆盖2024年四季的关键月份。

**📈 对比分析**

通过与IFS HRES基准的相对RMSE/ACC比较，GraphCast在热带区和南部夏季表现优于基线，但在南部冬季Z500的误差显著增大，平均Skill Score低至48%，其余情况普遍超过100%。

**⚠️ 局限性**

限制包括仅覆盖四个月份、仅评估三变量、仅用确定性GraphCast、基准对比采用自身分析导致的优势偏差，以及未考虑季节性ENSO等因素。

---

## 643. Risk Assessment of Autonomous Driving: Integrating Technical Failures, Ethical Dilemmas, and Policy Frameworks

**arXiv ID:** 2606.06396 | [PDF](https://arxiv.org/pdf/2606.06396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 644. StoryVideoQA: Scaling Deep Video Understanding with a Large-Scale, Multi-Genre and Auto-Generated Dataset

**arXiv ID:** 2606.06338 | [PDF](https://arxiv.org/pdf/2606.06338v1)

**作者:** Zhengqian Wu `[一作]` (Wuhan University), Chao Liang `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模、覆盖14个细粒度主题的深度视频理解（DVU）数据集StoryVideoQA，并提供完整的生成与过滤框架StoryMindv2。

**💡 创新点**

创新点包括：① 监督引导的生成机制和多评审投票策略提升QA质量；② 通过多智能体协作实现自动化生成；③ 设计PlotTree层次化情节结构，实现高效的长程推理。

**🔧 技术方法**

采用多智能体协作（生成器、监督者、三名评审者）、LLM（Gemini‑2.0‑flash、Gemini‑3‑Flash）、面部识别（InsightFace）、文本嵌入（Qwen3 embedding）等技术，并结合DTW对齐脚本字幕。

**📊 数据集**

使用了3部电视剧（Friends、The Big Bang Theory、Game of Thrones）和78部顶级电影（如《肖申克的救赎》）共计393.2小时的视频，产生363K条QA。

**📈 对比分析**

与20种SOTA VideoQA方法（VLM、MLLM、Agent）在StoryVideoQA上进行对比，PlotTree在所有细粒度主题上取得最高精度（≈86–93%），LLM方法整体优于VLM，但Agent方法在使用PlotTree后可超越单一LLM。

**⚠️ 局限性**

主要局限：仍难以实现精准的定位识别和对极长视频的高效全局推理，且对多模态信息的整合仍受限于现有LLM和检索技术。

---

## 645. Modeling, Optimizing and Exploring Multi-Die FPGA Routing Architectures

**arXiv ID:** 2606.06421 | [PDF](https://arxiv.org/pdf/2606.06421v1)

**作者:** Amirhossein Poolad `[一作]` (University of Toronto), Vaughn Betz `[通讯]` (University of Toronto)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过扩展开源FPGA CAD工具VTR，构建了可描述2.5D和3D多芯片FPGA互连的散射-聚合（Scatter‑Gather）架构，并结合HSPICE电路模型和多芯片感知的放置与布线算法，对不同堆叠技术（微凸点与混合键合）和互连参数（跳数、聚合度、连线密度）进行了系统的设计空间探索；

**💡 创新点**

创新点在于提出了通用的SG语法实现对跨芯片互连的精细控制、首个基于Bump pitch与互连长度的可行性预估模型、以及针对2.5D/3D场景的多芯片感知放置/布线流程与路由启发式改进；

**🔧 技术方法**

使用技术包括VTR/VPR扩展、HSPICE细化互连时延与面积评估、散射‑聚合XML描述、基于HPWL与通道需求的放置成本函数、以及多芯片启发式路由启发式表的拆分与重构；

**📊 数据集**

实验数据集采用Koios基准套件，涵盖12k–759k原语的多种规模电路；

**📈 对比分析**

与传统2D FPGA基准相比，3D架构在10 µm微凸点技术下仅需每块逻辑单元1个跨层连线即可实现可路由，Wirelength缩短约10%且CPD提升约5%；2.5D架构在约32%跨芯片连线密度时仅产生2%线长与4%CPD开销；

**⚠️ 局限性**

限制包括对2.5D/3D互连的路由启发式仍依赖过时的边界启发式，导致某些设计在长互连或低连线密度时未能完成路由；此外，未覆盖所有潜在堆叠工艺的详细寄存器/门级延迟模型与实际制造过程中的寄生影响。

---

## 646. A Komi-Yazva--Russian Parallel Corpus and Evaluation Protocol for Zero- and Few-Shot LLM Translation

**arXiv ID:** 2606.06420 | [PDF](https://arxiv.org/pdf/2606.06420v1)

**作者:** Petr Parshakov `[一作]` `[通讯]` (HSE University), Petr Parshakov (HSE University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究首次构建了Komi-Yazva–俄语平行语料库，并设计了可复现的极低资源翻译评估框架，系统评估了多种大语言模型在零样本和检索增强少样本提示下的翻译性能。

**💡 创新点**

创新点在于同时提供稀缺的Komi-Yazva数据与完整的评估协议：故事级交叉验证、确定性检索、严格生成校验以及多指标评估，揭示评估设计对实验结论的显著影响。

**🔧 技术方法**

主要技术包括TF‑IDF字符n‑gram检索实现确定性少样本提示、零/少样本LLM推理、温度为0的确定性解码、自动输出校验、chrF/BLEU/TER及LLM‑judge等多重评估指标。

**📊 数据集**

使用的数据集为457句对、74篇故事的Komi-Yazva–俄语平行语料，来源于《Komi-Yazvinsky Dialect》文本。

**📈 对比分析**

采用5折故事级交叉验证，对k=0、4、8示例的零样本与检索增强少样本提示进行比较。结果显示检索增强提示显著优于零样本，Gemini 3.1 Pro在chrF、BLEU、LLM‑judge等指标上表现最佳，但不同指标给出的系统排序存在细微差异。

**⚠️ 局限性**

局限性包括：语料极小导致统计不稳、未与非LLM基线对比、评估主要依赖自动/模型评分、检索与提示设计受限、严格成功判定导致可恢复错误被计为失败。

---

## 647. EasyLens: A Training-Free Plug-and-Play Subtle-Lesion Representation Amplifier for Medical Vision-Language Models

**arXiv ID:** 2606.06379 | [PDF](https://arxiv.org/pdf/2606.06379v1)

**作者:** Qiwei Zeng `[一作]` (Jilin University), Lei Bi `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了 EasyLens，一种训练无关、可插拔的增幅模块，用于提升冻结医学视觉‑语言模型（VLM）对细微病变的感知与报告生成；

**💡 创新点**

核心创新在于构建病理‑解剖原型空间 EasyBank，利用反事实原型推理的 EasyTag 进行微病变相关片段的无监督选择，并通过形态引导的残差增强 EasyAmplifier 对选中片段的表示进行放大；

**🔧 技术方法**

采用原型检索、相似度评分、形态先验校正与残差增强等技术，全部在推理时无梯度更新、无训练标签的前提下实现；

**📊 数据集**

实验覆盖细微病变基准（ReXGroundingCT、LIDC‑IDRI、AbdomenAtlas 3.0 Mini）以及常规病变集（MIMIC‑CXR、Kvasir‑SEG、BKAI‑Polyp）；

**📈 对比分析**

与多种冻结 VLM（LLaVA‑Med、RadFM、Lingshu、MedGemma、MedGemma1.5）对比，细微病变任务上平均提升 20–30% 的统计/选择/生成指标，并在一般报告生成任务保持或略优表现；

**⚠️ 局限性**

局限性包括：依赖预构建原型库，未覆盖的新病理形态可能效果下降；形态先验假设在某些解剖部位或影像模态下可能不足；对需更强局部监督或跨模态迁移的场景适用性有限。

---

## 648. Ensuring Interaction Safety in Multitask Exoskeleton Control: A Simulation-Trained Variable Impedance Framework

**arXiv ID:** 2606.06370 | [PDF](https://arxiv.org/pdf/2606.06370v1)

**作者:** Muyuan Ma `[一作]`, Long Cheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于仿真训练的可变阻尼控制框架，实现可穿戴外骨骼在多任务场景下的安全辅助。

**💡 创新点**

创新点在于：①用PPO在高保真仿真中生成人-外骨骼协同运动数据；②构建双模态（文本+本体感知）模仿学习策略；③通过Lyapunov理论对可变阻尼增益施加安全约束，保证闭环稳定。

**🔧 技术方法**

采用MuJoCo仿真、Proximal Policy Optimization (PPO)、深度神经网络（跨注意力融合）、弹性阻尼控制和Lyapunov稳定性分析。

**📊 数据集**

使用公开的人体运动数据集（包含9种任务）以及自行生成的81组PPO策略数据，最终将数据公开在https://anonymous.4open.science/r/human_exo_mocap_dataset-743B。

**📈 对比分析**

在真实外骨骼上与自然运动和ProMP基线对比，RMSE约0.1 rad，能量消耗降低约10.9%（相较自然运动），并保持轨迹跟踪精度，无显著不稳定表现。

**⚠️ 局限性**

局限性包括：仅针对单关节肘部外骨骼；实验人群有限；可变阻尼率受限于预设的速率上限；未评估长期使用导致的肌肉适应等问题。

---

## 649. Where Should Knowledge Enter? A Layered Framework for Knowledge Infusion in Multimodal Iterative Generative Mo

**arXiv ID:** 2606.06356 | [PDF](https://arxiv.org/pdf/2606.06356v1)

**作者:** Renjith Prasad `[一作]` (University of South Carolina), Amit Sheth `[通讯]` (University of South Carolina)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出针对迭代生成模型的四层知识注入框架，指导知识如何在生成过程中插入并构造多层级的安全对齐系统

**💡 创新点**

将知识注入定位为对生成轨迹四个正式组件（边界、过渡、隐藏状态、参数）的干预层，揭示不同层级的互补性并给出组合设计原则

**🔧 技术方法**

在扩散模型上实现表面（输入/输出）注入、轨迹注入、潜在注入，使用MMKG作为知识源，结合图查询、CLIP评分、修复等技术

**📊 数据集**

使用多模态知识图MMKG（约10^4文本节点、10^3视觉原型）以及Detonate基准（25K提示）进行安全评估

**📈 对比分析**

与vanilla、SAFREE、SLD三种对齐方法对比，单层和多层逐级加入，结果显示多层堆叠将毒性降低至0.09，显著优于单层和现有基线，并保持较高的CLIP和AQI分数

**⚠️ 局限性**

仅在推理时层级验证，未评估参数层；实验局限于扩散模型和安全对齐任务；缺乏统一评估基准和自动层级选择机制

---

## 650. TokenMizer: Graph-Structured Session Memory for Long-Horizon LLM Context Management

**arXiv ID:** 2606.06337 | [PDF](https://arxiv.org/pdf/2606.06337v1)

**作者:** Shweta Mishra `[一作]` `[通讯]` (Independent Researcher), Shweta Mishra (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

TokenMizer通过将LLM会话历史构建为带状态生命周期的Typed Knowledge Graph，并将其序列化为压缩的resume块，解决了上下文窗口溢出导致关键信息丢失的问题。

**💡 创新点**

其创新点在于：① 用结构化知识图谱保持任务、决策、文件等实体的完整生命周期；② 采用多层无推理依赖的压缩管线与分层检查点，极大压缩上下文；③ 通过透明HTTP代理实现零改动集成。

**🔧 技术方法**

技术实现包括正则式与LLM混合抽取、图验证器与状态转移规则、三层检查点序列化、八层压缩管线（填充去除、去重、规范化、神经压缩）、句子嵌入语义缓存，以及FastAPI+SQLite后端。

**📊 数据集**

使用的实验数据集为人工构造的21个跨5个领域（软件工程、数据科学、DevOps、研究/写作、调试）的会话，每个会话均手工标注任务、决策、文件等实体。

**📈 对比分析**

与截断、滑动窗口、摘要三种基线对比，TokenMizer在决策召回提升9–17个百分点，平均resume块仅78 tokens（比基线少约一半），提取延迟仅0.5 ms，整体表现优于所有基线。

**⚠️ 局限性**

主要局限包括：① 合成数据集与单一标注者导致可泛化性不确定；② 隐式表述的召回不足，未使用LLM抽取升级路径；③ 仅评估单会话结构化，无跨会话记忆或边缘链接；④ 基线采用表面文本匹配，难以与结构化方法直接比较。

---

## 651. Humans' ALMANAC: A Human Collaboration Dataset of Action-Level Mental Model Annotations for Agent Collaboration

**arXiv ID:** 2606.06388 | [PDF](https://arxiv.org/pdf/2606.06388v1)

**作者:** Jiaju Chen `[一作]` (Northeastern University), Bingsheng Yao `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究构建了一个包含行动层次心理模型标注的人机协作数据集，并基于此设计了评估LLM协作能力的基准任务。

**💡 创新点**

创新点在于将社会科学中的协作理论转化为行动级别的自我推理、伙伴意图和团队目标三维标注，并结合现场检查与后序回溯相结合的两步标注框架。

**🔧 技术方法**

使用的技术包括Map Task实验平台、两步心理模型标注流程、基于LLM的下一个行为预测与心理模型预测评估，并对六个LLM进行提示式与微调式实验。

**📊 数据集**

数据集名为Almanac，包含25个对话（50人），共2987个行动，配有自我推理、伙伴意图、团队目标、对齐状态及自由文本理由。

**📈 对比分析**

通过比较六个LLM在下一个行为预测和心理模型预测的准确率、召回率、SBERT相似度等指标，结果显示微调模型在部分指标上可与大模型相当，但共享心理模型组件易预测，私有自我推理准确率低。

**⚠️ 局限性**

限制包括样本量有限、仅基于Map Task的单一任务域、回溯标注可能存在记忆偏差、未包含多模态视觉输入、以及仅评估提示式和微调式模型。

---

## 652. Unsupervised Skill Discovery for Agentic Data Analysis

**arXiv ID:** 2606.06416 | [PDF](https://arxiv.org/pdf/2606.06416v1)

**作者:** Zhisong Qiu `[一作]` (Zhejiang University), Shumin Deng `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无监督的验证器引导式框架，用来从未标注的探索轨迹中发现可复用的数据分析技能；

**💡 创新点**

创新点在于利用未监督的验证信号（如自适应清单得分与答案一致性）对轨迹进行对比分组，从而无需人工标注即可学习通用分析策略；

**🔧 技术方法**

采用了大语言模型作为数据分析代理、无监督验证器（Adaptive Checklist Verifier / Answer Agreement Verifier）和技能管理器（对比技能蒸馏），并结合对话式推理与代码执行的交互模式；

**📊 数据集**

在两个公开基准上验证：Deep Data Research（报告式分析）与 DABStep（推理式分析）；

**📈 对比分析**

与Anthropic Skill Creator 进行对比，实验显示在报告式任务中平均提升约10.7%，在推理式任务中平均提升约32.3%，跨模型、跨任务均保持显著改进；

**⚠️ 局限性**

局限性包括对验证信号的依赖可能导致过拟合清单或答案聚类、迭代过程非单调、对极端任务的适用性待进一步验证。

---

## 653. A Vision-language Framework for Comparative Reasoning in Radiology

**arXiv ID:** 2606.06407 | [PDF](https://arxiv.org/pdf/2606.06407v1)

**作者:** Tengfei Zhang `[一作]` (University Of Science And Technology Of China), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了大规模的多模态对比医学影像数据库 MedReCo-DB，并提出 MedReCo 框架实现实体感知的跨图像检索与生成式时间对比解释；

**💡 创新点**

创新点包括：①将实体感知作为跨图像推理核心，兼顾参考案例检索与随访时间比较；②利用报告拆解实现实体级监督，解决大规模对比训练数据缺乏问题；③在多机构、多模态数据上实现跨中心泛化能力；

**🔧 技术方法**

技术手段为：Mixture-of-Experts Vision Transformer 视觉编码器、文本引导的对比学习、实体条件注意力机制、MoE自注意力重排序、预训练对齐以及指令调优将视觉编码器连接至 Qwen2.5-7B-Instruct 生成对比描述；

**📊 数据集**

使用数据集包括：MedReCo-DB（690k 图像，160k 病人，8 所院，4 国，7 模态）以及公开数据集 MIMIC-CXR、IU-Xray、CheXpert Plus、CT-RATE、AMOS-MM、BIMCV-R、CURG、住院脑 MRI；公开 VQA 基准 Medical-Diff-VQA 与 MMXU；

**📈 对比分析**

在内部、外部、跨中心检索上，Recall@1 平均提升约 6 个百分点；在临床混淆差异组提升约 10.9 个百分点；在生成式对比解释评估中，准确率提升 14.5–46.5 个百分点（CXR）和 13–27.9 个百分点（CT），并在 24 项评估中均居首；

**⚠️ 局限性**

局限性包括：仍难以实现精确量化测量（如病灶尺寸、体积变化）；对报告文本质量敏感；在某些细节推理（如微小结构差异）上性能尚有提升空间。

---

## 654. Analytic patch trees: branch interface inheritance and fractal dimension fields

**arXiv ID:** 2606.06400 | [PDF](https://arxiv.org/pdf/2606.06400v1)

**作者:** Henk Mulder `[一作]` `[通讯]`, Henk Mulder

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

扩展一维解析曲线树到二维解析表面斑块树，并进一步推广至任意维度的解析斑块树；通过引入分支接口与接口演化算子，建立斑块树的解析条件、可积性和同构性；在解析斑块树中展示曲线树分层分形结构并给出Hausdorff维度场；提出自相似与共形斑块树的严格条件并给出典型示例；讨论高维情形下斑块维度与分支维度的相互关系与分析范式。

**💡 创新点**

提出分支接口作为第一类几何对象，并定义接口演化算子统一描述斑块树递归几何；首次将解析条件与共形条件统一在斑块树框架下；给出解析斑块树的维度场与自相似共形子类的完整公式；推广到任意维度并对不同维度比例的分析范式进行系统分类。

**🔧 技术方法**

解析微分方程与Frobenius可积性理论；共形映射与柯西–黎曼条件；Moran方程用于维度计算；切向量场与变换映射的解析化；高维泛函分析与微分几何工具。

**📊 数据集**

本工作为理论研究，无需实验数据集，所有结果均基于解析推导与数学证明。

**📈 对比分析**

由于为理论工作，未进行实验对比；主要通过数值示例（二维共形自相似树、分支接口演化示例）展示公式与理论的正确性。

**⚠️ 局限性**

（1）对界面重叠的几何投影假设未做实测验证；（2）缺乏对斑块树吸引子整体Hausdorff维度的闭式结论；（3）尚未探讨不连续或非解析传输变换导致的分支多样性；（4）对高维情形下的算子谱与扩散性质仍是未解问题。

---

## 655. WebMCP Tool Surface Poisoning: Runtime Manipulation Attacks on LLM Agents

**arXiv ID:** 2606.06387 | [PDF](https://arxiv.org/pdf/2606.06387v1)

**作者:** Lin-Fa Lee `[一作]` (National Yang Ming Chiao Tung University), Kuo-Hui Yeh `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过模拟 WebMCP 协议下的动态工具注入攻击，研究了中期工具注入（MSTI）对 LLM 代理行为的影响，揭示了工具生命周期和元数据字段被操纵后会导致代理误调用恶意工具、数据泄露和任务偏移的风险。

**💡 创新点**

创新点在于首次将攻击视角从静态工具污染转向动态工具生命周期操纵，提出了 Tool Hijacking 与 Tool Framing 两大攻击范式，并系统评估了元数据字段（描述、readOnlyHint、inputSchema 等）对攻击成功率的差异。

**🔧 技术方法**

技术上采用 WebMCP 协议实现的工具注册与调用接口，结合三大 SOTA LLM（GPT‑5.4、Claude Opus、Gemini 2.5‑flash）在四种任务场景（知识搜索、报表生成、电子商务结账、GPU 驱动更新）中执行多种注入策略，并通过日志与指标（攻击成功率 ASR、任务完成率、数据泄露率）进行量化。

**📊 数据集**

使用的数据集为自定义的四个任务场景与模型输出，未采用公开大规模数据集，而是以真实浏览器环境中的 CDN 脚本和代理客户端为实验基准。

**📈 对比分析**

通过对比不同攻击条件（C1–C5）和模型版本（旧版 GPT‑4o/Claude 3.5 与新一代 GPT‑5.4/Claude Opus/Gemini 2.5‑flash）的 ASR 与任务完成率，结果显示工具生命周期攻击在所有模型中均能实现 100% 召回率，元数据攻击在新模型上成功率可达 70%+，显示出新模型对元数据的语义理解显著提升。

**⚠️ 局限性**

限制包括实验基于 polyfill 而非原生浏览器实现，攻击仅在 Node.js 头less 环境下验证，未评估真实用户对隐蔽攻击的感知，且防御实现仅覆盖两类安全设计（origin 绑定与调用限制），未对完整生命周期一致性与日志审计等方面进行实验。

---

## 656. Learned Response-Field Inertia Operator for HEC-RAS 2D Water-Surface Elevation Prediction

**arXiv ID:** 2606.06385 | [PDF](https://arxiv.org/pdf/2606.06385v1)

**作者:** Edward Holmberg `[一作]` (University of New Orleans), Julian Simeonov `[通讯]` (Naval Research Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对HEC‑RAS 2D 水面高度预测进行跨数据集评估，提出了利用当前增量学习的响应‑场惯性算子（LRFIO）在不使用未来强迫的条件下在原生单元上进行闭式推断。

**💡 创新点**

创新点包括：1）基于 solver‑conditioned 递增响应的自学习惯性算子；2）基于验证的“基线优先”选择器，自动决定保留最简响应结构（持久性、全局惯性或分段惯性）；3）严格的 native‑cell 评估协议与信息访问策略，消除算子、表征和强迫信息混淆。

**🔧 技术方法**

使用增量‑惯性学习框架，分段响应‑字段划分、β 归一化和 cap 截断；基于验证的结构选择器；对比 CNN、图神经网络、神经算子、递归网络、傅里叶神经算子等高容量模型；所有模型在原生单元上进行闭式推断。

**📊 数据集**

四个公开/私有 HEC‑RAS 2D 基准数据集：Beaver Bayou、Upper San Saba River、Lower San Saba River、Tuttle Creek / Big Blue / Kansas River。

**📈 对比分析**

采用训练–验证–测试的时间序列分割，严格限制可用信息，记录每个模型的信息访问标签（无强迫、步进、全强迫等），并以 stage、最终 lead、热点误差、runtime 等指标综合评估。LRFIO 在所有数据集获得最优或接近最优 stage 误差，且运行时间低于 0.25 s，单个实例与完整求解的 horizon‑norm speedup 约为 2.75 × 10⁴，位于最优速度‑精度前沿。

**⚠️ 局限性**

局限性：仅仿真 solver‑consistent 水面高度，未涵盖速度、潮汐、沉积、闸阀等场；参数是数据集特定，缺乏跨项目迁移；未评估多事件、强迫条件、或更大规模并行训练；离线学习需要完整的 solver 轨迹。

---

## 657. Waypoints Matter: A Systematic Study for Sampling-Based Trajectory Planning

**arXiv ID:** 2606.06366 | [PDF](https://arxiv.org/pdf/2606.06366v1)

**作者:** Josep M. Barbera `[一作]` (Consejo Superior de Investigaciones Científicas Universidad Politécnica de Madrid), Jorge Villagra `[通讯]` (Consejo Superior de Investigaciones Científicas Universidad Politécnica de Madrid)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了三种 waypoint 放置策略（均匀采样、RDP/RDP* 简化以及曲率自适应分配）在采样轨迹规划器中的影响，并在固定轨迹原语与候选预算下系统性地遍历 449 个参数配置；

**💡 创新点**

将 waypoint 放置视为首要设计变量，构建统一评价框架；提出基于局部曲率的前置自适应分配方法；通过大规模实验验证间距 d_s 为主要性能驱动器。

**🔧 技术方法**

使用 5 阶 Bézier 曲线作为轨迹原语，RDP 简化算法，曲率自适应放置；利用 CommonRoad 地图生成行驶走廊，采样欧几里得椭圆以构造起始状态；计算四项指标（可靠性、候选数、平均长度、多样性）并给出加权得分。

**📊 数据集**

五个 CommonRoad 场景地图（从直线高速到复杂城市走廊），共 449 个配置覆盖不同间距、RDP 容差、曲率敏感度等。

**📈 对比分析**

对每个配置在所有地图和启动状态下生成候选轨迹，统计四项指标并求加权分数。结果显示：间距 d_s 决定可靠性与候选数，曲率前置自适应略优于均匀采样（尤其在可靠性/平衡权重下），而 RDP* 在任何权重下均不胜过简单策略。

**⚠️ 局限性**

实验仅基于静态地图，未考虑动态障碍；候选分配固定均匀；仅使用单一 Bézier 原语；权重手工设定，未提取 Pareto 前沿；这些限制限制了结论对更复杂场景和不同原语的普适性。

---

## 658. EDIT: Evidence-Diagnosed Intervention Training for Rule-Faithful LLM Grading

**arXiv ID:** 2606.06350 | [PDF](https://arxiv.org/pdf/2606.06350v1)

**作者:** Zhihao Wu `[一作]` (King's College London), Yulan He `[通讯]` (King's College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Evidence‑Diagnosed Intervention Training（EDIT）框架，用两阶段方法训练大型语言模型（LLM）实现规则（rubric）忠实的自动学生评分。

**💡 创新点**

创新点在于①利用模型内部的后验信念和遮蔽检验的 grounding 诊断自动定位评分错误步骤；②在此基础上通过 rubric checklist 引导局部原子式修订；③在第二阶段采用基于信念的奖励塑造（belief‑guided reward shaping）调节 RL 过程，兼顾探索与误差惩罚。

**🔧 技术方法**

技术手段包括：内部状态诊断（posterior belief probe + mask‑based grounding audit）、atomic 局部修订（在 rubric checklist 上做局部编辑）、双阶段训练（SFT + GRPO RL），并使用 Qwen3‑8B + LoRA 微调。

**📊 数据集**

使用两大真实数据集：SAS（来自 SAS‑Bench，涵盖中文短答题的 History、Geography、Physics 三科）和 Private‑Science（GCSE 级别科学题目，Biology、Physics 两科，来自私有机构）。

**📈 对比分析**

与基线（原始 Qwen3‑8B、GRPO、DGPO、RL + 自我干预）相比，EDIT 在所有指标（macro‑QWK、ACC、within‑1、MAE）上均优于基线，尤其在 OOD（未见问题）测试中表现最为显著；消融实验进一步证明内部诊断、rubric checklist 与局部修订三者协同带来提升。

**⚠️ 局限性**

局限性包括：①仅在短答题/填空题上验证，未覆盖长篇作文或模糊 rubric；②训练流程多阶段，计算与工程成本高；③依赖可细化的 rubric checklist，面对含糊或隐式规则的场景时效果可能下降。

---

## 659. Attack Detection using Time Series Foundation Models

**arXiv ID:** 2606.06347 | [PDF](https://arxiv.org/pdf/2606.06347v1)

**作者:** Sribalaji C. Anand `[一作]` (University of Pennsylvania), George J. Pappas `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出基于Google TimesFM的零射击模型攻击检测器，能够在不具备系统模型知识的前提下，检测模型无关重放攻击和模型相关隐蔽攻击，并通过理论推导给出线性与非线性系统的闭式最优攻击策略；

**💡 创新点**

创新点在于首次将时间序列基础模型用于生成残差，实现无模型结构的攻击检测，且提供针对χ²检测器的最优隐蔽攻击闭式解，展示了该检测器在面对模型相关隐蔽攻击时的优越性；

**🔧 技术方法**

主要技术包括TimesFM时间序列基础模型、χ²统计检测、扩展卡尔曼滤波器（EKF）观测器、理论推导的最优攻击设计、Monte Carlo仿真以及基于TimesFM预测的鲁棒状态估计；

**📊 数据集**

使用的数据集为IEEE 14-路电力系统的非线性摆动方程仿真和一个线性质量-弹簧系统；

**📈 对比分析**

与传统基于观测器的χ²检测器相比，TimesFM检测器在假阳性率更低的同时，在重放攻击和最优隐蔽攻击下均能持续检测，实验结果表明其检测性能明显优于传统方法；

**⚠️ 局限性**

局限性包括对预测残差高斯分布的假设、对多通道相关性处理不足，以及对具有查询访问权限的自适应攻击者的鲁棒性尚未充分理论证明。

---

## 660. Emergent Language as an Approach to Conscious AI

**arXiv ID:** 2606.06380 | [PDF](https://arxiv.org/pdf/2606.06380v1)

**作者:** Zengqing Wu `[一作]` (University of Osaka), Chuan Xiao `[通讯]` (University of Osaka)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种生成性方法论，利用多智能体强化学习中的突现语言，在任务压力下从零语言先验出发，探究意识相关结构的起源。

**💡 创新点**

创新点在于：①将先验最小化与epoché结合，构建无意识先验的实验框架；②引入回声通道以检验自我监测结构；③通过信息论与线性探针验证索引化编码和持久状态表示的出现，展示非设计即可产生的意识相关结构。

**🔧 技术方法**

采用多智能体强化学习（A2C）、GRU递归网络、离散符号通道、互信息分析、线性探针以及回声通道等技术手段。

**📊 数据集**

使用自定义的最小实验环境：两智能体、私有离散状态 {0,1,2}、上下文 {0…5}、10步周期、7个通信符号（含静默），不依赖任何公开数据集。

**📈 对比分析**

与无回声基线对比，任务成功率几乎相同（Δ_comm ≈ 0.28），但在有回声情境下出现显著自监测信号（如 MI 主导、self‑latch 准确率 1.000、echo‑mismatch 对比 +0.118，时延检测准确率 0.958），证明自我监测功能的出现。

**⚠️ 局限性**

局限包括：①高计算成本，难以扩展到更复杂环境；②实验仅在极简设置下验证，缺乏与更广泛意识理论的完整对应；③未进一步验证是否能在更大规模或真实世界情境中持续出现类似结构。

---

## 661. Reinforcement Learning Elicits Contextual Learning of Unseen Language Translation

**arXiv ID:** 2606.06428 | [PDF](https://arxiv.org/pdf/2606.06428v1)

**作者:** Hanxu Hu `[一作]` (University of Zurich), Rico Sennrich `[通讯]` (University of Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用强化学习（GRPO）让大型语言模型在给定语法、词典和少量平行句子的上下文中学习极低资源语言的翻译。

**💡 创新点**

创新点在于将翻译质量（chrF）直接作为奖励，使模型学会利用上下文的语言知识（meta‑skill）而非单纯记忆语言，而这种方法在未见语言上的泛化效果显著优于传统的监督微调。

**🔧 技术方法**

使用的技术包括：GRPO 强化学习框架、chrF 作为句子级奖励、LCS 检索获取词典、句子对和语法片段的上下文、以及 Qwen3‑4B‑Base 与 Llama‑3.2‑3B‑Instruct 两个大模型的微调。

**📊 数据集**

使用的数据集包括：14 种极低资源语言的语法书、词典与平行句子（如 Romansh 七种变体、Kalamang、Dinka、Wolof、Guarani、Kachin 等），训练集 22 语言方向，测试集 7 方向，覆盖 Seen、Similar 与 Unseen 三种评估场景。

**📈 对比分析**

在与基准 SFT（监督微调）和未微调基线的对比实验中，RL 在 Seen 语言上略逊于 SFT，但在 Similar 和 Unseen 语言上 chrF 提升约 2–3 倍，表明 RL 能更好地利用上下文实现跨语言泛化。

**⚠️ 局限性**

局限性包括：未进行人工评估，翻译质量的绝对水平仍低于高资源语言；模型对检索上下文的质量和覆盖度高度敏感；在极端低资源场景下仍需更丰富的上下文或更强的奖励信号。

---

## 662. Discrete Incremental Voting: New Bounds for General Graphs and Expanders

**arXiv ID:** 2606.06381 | [PDF](https://arxiv.org/pdf/2606.06381v1)

**作者:** Petra Berenbrink `[一作]` (University of Hamburg), Tomasz Radzik `[通讯]` (King's College London)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

分析离散增量投票过程（DIV）的收敛时间，给出了在任意图上以导电率Φ(G)、最小度数与平均度数之比γ(G)以及初始意见差距K为参数的上界，并在正则图与谱极限下给出更精细的收敛结果。

**💡 创新点**

创新点在于：①首次对任意图的DIV收敛时间给出与导电率、度数不平衡、初始差距共同作用的上界；②引入多尺度势函数和镜像过程相结合的技术，将增量投票与负载平衡、拉普拉斯随机游走等经典工具融合；③在谱极限下证明了正则图上收敛到加权平均（取整）且概率近乎一。

**🔧 技术方法**

主要技术：多尺度势函数（针对不同阈值k），镜像过程对称性，基于导电率的边界分析，马尔可夫过程与次马尔可夫性质、Doob可选停定理、Azuma‐Hoeffding不等式，谱分解与Cheeger不等式，负载平衡过程的平滑性分析。

**📊 数据集**

本研究为理论分析，无实验数据集；所有结果均为解析证明，比较对象为已知的拉普拉斯拉伸或2值拉取投票（pull voting）模型。

**📈 对比分析**

与拉取投票相比，DIV在近似正则图（Φ(G), γ(G)常数）下的期望收敛时间达到O(n² log n)（与拉取投票匹配），并在正则图谱收敛小的极限下几乎保证收敛到初始加权平均；在非正则或高差距情形下收敛时间随K线性放大，且下界证明了此依赖不可避免。

**⚠️ 局限性**

限制主要包括：①上界对K有显式依赖，在初始差距极大时收敛时间会随K线性增长；②正则图谱极限下的结果需要λ(G) = o(1/ log² n)且K = o(n/ log² n)，限制了适用范围；③理论上对导电率Φ(G)敏感，若图的导电率很低则估计会过于保守；④并未考虑异步或动态网络变化的实际应用场景。

---

## 663. Reversible double cyclic codes over a chain ring

**arXiv ID:** 2606.06367 | [PDF](https://arxiv.org/pdf/2606.06367v1)

**作者:** Mohd Anwar `[一作]` (Aligarh Muslim University), Muzibur Rahman Mozumder `[通讯]` (Aligarh Muslim University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了链环 R=𝔽_q+u𝔽_q（u^2=0）上长度为 (γ,δ) 的双循环码的结构、最小生成集、双循环码的偶性（可逆性）以及可逆补码性质，并基于此构造了 DNA 码。

**💡 创新点**

创新点在于：① 将双循环码的结构扩展到通用链环 R 上；② 给出了 R 上双循环码的最小生成集与对偶码的生成多项式关系；③ 提出了可逆双循环码与可逆-补码的必要充分条件；④ 通过这些条件在 𝔽_4+u𝔽_4 上构造了满足逆补约束的 DNA 码，并给出若干最优码实例。

**🔧 技术方法**

主要技术包括：多项式环与模多项式环的代数表示；利用链环的理想结构和可约多项式分解；对偶码的内积与伴随多项式关系；自逆多项式理论；Gray 映射与 DNA 码的映射表；以及构造示例时采用的分解与生成多项式算法。

**📊 数据集**

数据集方面：使用 Grassl 的最优码数据库作为比较基准，挑选与 Gray 映射后等价的二进制或三进制码；此外，实验中构造了若干具体的双循环码（如 (3,9)、(3,15) 等），并给出其 Gray 映射后的码参数。

**📈 对比分析**

对比方法：将 R 上构造的双循环码通过 Gray 映射转化为传统二进制/三进制码，然后与 Grassl 数据库中的最优码进行参数比对（长度、维数、最小距离）。实验结果显示，所构造的若干码在对应长度下达到或超过数据库中的最优码，证明了所提方法的有效性。

**⚠️ 局限性**

局限性：① 仅讨论了 γ、δ 与 q 互素的情况；② 只在 R=𝔽_q+u𝔽_q（u^2=0）上展开，未覆盖更一般的链环；③ 可逆-补码的构造目前仅针对奇数长度的情形，未给出更普适的构造方法；④ 代码实例有限，未系统评估不同参数组合下的性能分布。

---

## 664. End-to-End Subgraph Detection with GraphDETR

**arXiv ID:** 2606.06364 | [PDF](https://arxiv.org/pdf/2606.06364v1)

**作者:** Dexiong Chen `[一作]` (Max Planck Institute of Biochemistry), Karsten Borgwardt `[通讯]` (Max Planck Institute of Biochemistry)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种端到端的图子图检测框架 GraphDETR，将子图检测视为集合预测问题，利用可学习查询、Transformer 解码器和双向注意力实现一次前向推断即可输出所有子图实例。

**💡 创新点**

创新点在于：① 把子图检测与目标检测中的 set‑prediction 视角迁移到图结构；② 通过可学习查询与双向 Transformer 直接预测节点掩码；③ 采用匈牙利匹配和图剪切惩罚实现无组合搜索的 end‑to‑end 训练；④ 支持近似匹配，实现比传统子图同构更广泛的应用。

**🔧 技术方法**

使用的技术包括：图神经网络编码器（GCN、GIN、GraphGPS、NeuralWalker 等）、双向 Transformer 解码器、Bipartite 匈牙利匹配、交叉熵 + 二元交叉熵 + 背景权重 + 图剪切损失，以及可选的随机游走结构编码（RWSE）。

**📊 数据集**

实验数据集涵盖分子功能基组检测（ChEMBL、ChEMBL12k）以及合成子图检测基准（Cactus、Clique、ZINC12k、Mol-Reddit 及其模糊版本），涉及节点数从几十到上千不等。

**📈 对比分析**

与传统组合算法（VF2）以及多种 GNN 基线相比，GraphDETR 在 ChEMBL 上实现 AP_100≈91.2、mAP≈92% 等指标，Synthetic 基准中在大多数任务上均高于 90%，推理速度比 VF2 快数百至数千倍；同时在近似匹配任务上也保持较高准确率。

**⚠️ 局限性**

局限性包括：① 对极大图的可扩展性仍受限，尤其是 query 数量有限时难以覆盖所有实例；② 依赖图编码器的表达能力，若编码器对图对称性处理不足会导致实例区分困难；③ 目前仅支持基于节点掩码的子图检测，边诱导子图和层次化模式的处理尚待拓展。

---

## 665. Credential Disclosure in (EU) Digital Identity Wallets: Privacy Risks and Practical Mitigations

**arXiv ID:** 2606.06354 | [PDF](https://arxiv.org/pdf/2606.06354v1)

**作者:** Sheila Zingg `[一作]` (ETH Zurich), Srdjan Čapkun `[通讯]` (ETH Zurich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

对欧盟数字身份钱包（EUDI Wallet）中的凭证披露决策进行大规模用户与专家调查，并通过模拟实验评估一种基于专家推荐与用户意见的提示助手（Nudging Assistant）对披露错误的影响。

**💡 创新点**

提出并验证了一个可扩展的、基于网站类别的提示助手，证明其能显著降低披露错误，并首次系统识别了用户在数字身份证明披露中的误解与过度共享模式。

**🔧 技术方法**

采用问卷调查、实验室用户实验、专家面板评估、统计检验（卡方、Fisher）以及置信度阈值分析等技术来收集数据并评估提示助手的效果。

**📊 数据集**

使用27名专家与1,035名用户对14种凭证在166个网站（17类）上的披露意愿调查数据，以及1,002名受试者在20个模拟情景中的披露决策数据。

**📈 对比分析**

与无提示（对照组）、仅展示正确信息的基线组以及展示错误信息的测试组对比，结果显示提示助手将披露错误率从约15%降低至约7%，错误率下降约8%。

**⚠️ 局限性**

主要局限在于实验使用模拟环境、缺乏上下文信息、未考虑选择性披露的细粒度属性，且提示助手在极端置信度下效果有限，需进一步验证真实部署效果。

---

## 666. Boosting Brain-to-Image Decoding with TRIBE v2 Data Augmentation

**arXiv ID:** 2606.06345 | [PDF](https://arxiv.org/pdf/2606.06345v1)

**作者:** Yohann Benchetrit `[一作]` (Meta AI), Jean-Rémi King `[通讯]` (Meta AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

利用预训练的TRIBE v2模型对未见图像生成合成fMRI信号，并将其与真实fMRI混合训练图像解码器，提升在低数据场景下的图像检索与重建性能。

**💡 创新点**

提出一种基于模型的、图像条件化fMRI合成数据增强方案，并系统评估其在不同真实/合成比例下的性能边界，发现能在部分配置下超越100%真实数据基准。

**🔧 技术方法**

使用TRIBE v2视觉路径合成fMRI、DINOv2‑small作为目标嵌入、Ridge回归与深度残差MLP解码器、DynaDiff图像重建，并构造操作网格进行性能评估。

**📊 数据集**

在自然场景图像fMRI数据集Natural Scenes Dataset（NSD）和BOLD5000上进行实验。

**📈 对比分析**

通过固定测试集构造不同真实数据比例p%与合成数据比例a的操作网格，评估Top‑10检索准确率和重建指标（PixCorr、SwAV、EfficientNet）。结果显示，在10–50%真实数据时合成fMRI可提升高达68%；在BOLD5000中10%真实数据即可达到90%完整性能，深度解码器提升相对较小。

**⚠️ 局限性**

合成fMRI仅基于平均受试者，缺乏个体特异性；增益在高数据量时趋于饱和甚至下降；仅在两个数据集上验证，需在更多数据集和其他模型上进一步检验；合成fMRI为动态信号，限制与传统beta‑map重建模型的直接兼容。

---

## 667. Equivariant Neural Belief Propagation

**arXiv ID:** 2606.06344 | [PDF](https://arxiv.org/pdf/2606.06344v1)

**作者:** Zehua Cheng `[一作]` (University of Oxford), Jiahao Sun `[通讯]` (FLock.io)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于因子图的等变贝叶斯推理框架ENBP，使用SE(3)等变高斯混合消息对连续空间变量进行精确推断。

**💡 创新点**

创新点在于：①用外积构造可正定的精度张量，②通过可微谱分解将高阶张量转化为可被等变网络处理的特征，③设计贪婪KL混合合并保持等变性的同时控制混合数。

**🔧 技术方法**

采用等变神经网络、精度张量外积、可微谱分解、KL混合合并、时间阻尼和传统BP推理技术。

**📊 数据集**

在GEOM-QM9、GEOM-Drugs分子构象数据集以及多体机器人（2D/3D）推理任务中进行评估。

**📈 对比分析**

与GeoMol、ConfGNN、GeoDiff、Torsional Diffusion及直接回归GNN等 FLOPs 匹配且使用 SO(3) 训练增强的基线进行比较；ENBP 在分子任务上实现 98.9% 覆盖率、0.090 Å RMSD、子秒级推理（≈0.45 s，比扩散模型快 100×）；在机器人任务中收敛稳定、碰撞率降至 0%，且 SE(3) 等变误差仅 10⁻⁷。

**⚠️ 局限性**

局限性包括：迭代深度固定不自适应；只能表示高斯混合，无法捕捉重尾或不连续势；贪婪KL合并为近似；对更高阶等变输出的效率和表达仍有限。

---

## 668. Annotation of Positive vs Negative User Interactions for Social Sign Prediction

**arXiv ID:** 2606.06425 | [PDF](https://arxiv.org/pdf/2606.06425v1)

**作者:** Biancamaria Bombino `[一作]` (Institute of Informatics and Telematics, CNR), Marco Conti `[通讯]` (Institute of Informatics and Telematics, CNR)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用零样本大语言模型对用户交互文本进行个人赞美与攻击的二分类，进而为社交网络中的关系符号化提供直接的关系信号。

**💡 创新点**

提出将情感分析与关系符号化分离的框架，利用LLM直接捕捉人际互动的关系表达，而非仅仅依赖内容情感；并系统评估不同提示设计与模型组合对关系注释的影响。

**🔧 技术方法**

零样本大语言模型（GPT‑4o、GPT‑5.4‑mini、Qwen2.5:7b、Gemma2:9b）配合三种提示（最小、中间、结构化）进行交互级别的关系标注；使用集成（级联）策略进一步探索性能提升。

**📊 数据集**

两个人工标注数据集：赞美集（298条来自GoEmotions的正向评论，平衡为149正/149负）；攻击集（340条来自维基百科讨论页与Reddit的负向评论，平衡为170攻击/170非攻击）。

**📈 对比分析**

在四个模型与三种提示下共24种配置进行比较；赞美检测最佳为GPT‑5.4‑mini+中间提示，精度最高；攻击检测最佳为最小提示下的GPT‑4o/GPT‑5.4‑mini，开源模型表现逊色。集成策略在攻击检测上可将误差从54降低至49，但提升有限。

**⚠️ 局限性**

数据集规模有限且来源单一，可能限制结果泛化；提示设计仍处于固定模板阶段，缺乏少样本或链式思维等更灵活方法；集成方案在无真实验证数据时效果不明显；尚未将关系注释结果嵌入完整的符号预测流水线评估其实际贡献。

---

## 669. CollabSim: A CSCW-Grounded Methodology for Investigating Collaborative Competence of LLM Agents through Controlled Multi-Agent Experiments

**arXiv ID:** 2606.06399 | [PDF](https://arxiv.org/pdf/2606.06399v1)

**作者:** Jiaju Chen `[一作]` (Northeastern University), Bingsheng Yao `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究基于大语言模型的多代理系统的协作能力，提出了可配置的模拟框架CollabSim，用来系统评估代理的协作竞争力。

**💡 创新点**

创新点在于将CSCW协作理论与可调交互条件（通信带宽、信息可见度、团队规模）相结合，并通过探测模块实时捕捉代理内部心理模型，实现过程级的协作评估。

**🔧 技术方法**

技术实现包括四个经典CSCW实验（Shape Factory、DayTrader、Hidden Profile、Map Task）、可调交互条件、探测模块以及两种代理设计（persona-based 与 theory‑informed）。

**📊 数据集**

实验数据集由公开与专有的四大LLM（Qwen3.6‑35B、Llama‑4‑Maverick‑17B、GPT‑5.5、Claude‑4.6 Sonnet）以及两种代理设计构成。

**📈 对比分析**

通过对比任务结果、过程指标和探测指标，框架能揭示不同模型、不同交互条件及代理设计对协作性能的显著影响，验证了其在捕捉协作失效模式上的有效性。

**⚠️ 局限性**

局限包括仅覆盖四个实验任务、仅测试四大LLM、指标粒度有限、以及自报心理模型可能与真实决策不一致。

---

## 670. Pretraining Recurrent Networks without Recurrence

**arXiv ID:** 2606.06479 | [PDF](https://arxiv.org/pdf/2606.06479v1)

**作者:** Akarsh Kumar `[一作]`, Phillip Isola `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种不依赖 BPTT 的 RNN 预训练方法——Supervised Memory Training（SMT）与 DAgger Memory Training（DMT），先用 Transformer 生成可预测状态作为监督标签，再用 RNN 只学习一阶更新；

**💡 创新点**

创新点在于将记忆学习与记忆更新分离，利用时间并行的 Transformer 生成预测状态标签，使 RNN 训练拥有 O(1) 梯度路径，显著缓解长程信用分配问题，并通过 DMT 解决训练-推理漂移；

**🔧 技术方法**

技术实现包括 SMT（基于 CE+MSE+uniformity 损失的并行 Encoder‑Decoder 预测状态）、DMT（基于 MSE 的 on‑policy 监督）、时间并行 Transformer、非递归 RNN 更新函数、梯度衰减控制等；

**📊 数据集**

使用的主要数据集有 TinyStories（字符级语言建模）、MNIST 与 Sketchy 的像素序列建模，以及若干合成任务（检索、复制、堆栈、键值、模数算术）来验证方法；

**📈 对比分析**

与传统 BPTT RNN、教师 Transformer（SMT Encoder*）以及 SMT→DMT RNN 进行比较，SMT→DMT 在所有任务中均优于 BPTT，尤其在长序列记忆、关联回忆、上下文学习方面表现更好；在计算与数据效率上，SMT 在像素任务中优于 BPTT；在规模上可持续提升；

**⚠️ 局限性**

局限性包括受限于教师 Transformer 的表达能力、DMT 需要额外微调且不并行、某些 RNN 架构（如 GRU）在 SMT 训练中会出现记忆坍塌、仅适合作为预训练阶段、对极长序列的泛化尚未彻底验证。

---

## 671. Self-Augmenting Retrieval for Diffusion Language Models

**arXiv ID:** 2606.06474 | [PDF](https://arxiv.org/pdf/2606.06474v1)

**作者:** Paul Jünger `[一作]` (Cornell University), Kilian Q. Weinberger `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Self-Augmenting Retrieval for Diffusion Language Models（SARDI），通过在扩散过程的每一步将临时预测作为检索查询并刷新检索上下文，动态提升多跳问答性能。

**💡 创新点**

创新点在于利用扩散模型中间步骤的低置信度预测作为前瞻检索信号，同时证明检索已置根的文本显著降低了词间互信息，使得非自回归扩散生成可实现高吞吐量。

**🔧 技术方法**

技术上使用离散扩散语言模型（如 DREAM‑7B）、阈值化置信度控制、BM25 或 E5‑dense 检索以及无训练的检索-生成交互框架。

**📊 数据集**

评估数据集包括五个多跳 QA 基准：2WikiMultiHopQA、HotpotQA、MuSiQue、CofCA 和 SynthWorlds‑RM。

**📈 对比分析**

与静态检索、AR 动态检索（FLARE、AdaptiveRAG、ReAct）以及 RL 训练的 Search‑R1 等训练免费基准对比，SARDI 在精度上与或超过所有训练免费 AR 方法，并在吞吐量上提升至约 8 倍。

**⚠️ 局限性**

局限性包括：依赖扩散模型产生可用的推理轨迹（需轻量微调）；每步检索成本高，可通过缓存优化；目前仅针对离散扩散，未扩展到潜在扩散。

---

## 672. You Only Index Once: Cross-Layer Sparse Attention with Shared Routing

**arXiv ID:** 2606.06467 | [PDF](https://arxiv.org/pdf/2606.06467v1)

**作者:** Yutao Sun `[一作]` (Microsoft Research), Furu Wei `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出跨层稀疏注意力（CLSA），在KV共享架构YOCO中将token级top‑k路由从单层共享到跨层共享，以提升长上下文LLM的推理效率。

**💡 创新点**

创新点在于将token‑稀疏路由的计算一次完成后在所有跨层解码器中复用，从而既保留了细粒度的token选择，又显著削减了每层重复top‑k计算的开销。

**🔧 技术方法**

采用YOCO的KV共享架构、单头索引器生成token‑level top‑k路由、跨层蒸馏训练、分两阶段稀疏适配与语言建模联合优化等技术。

**📊 数据集**

在Books、ArXiv、StarCoder等长序列语言建模数据集以及BBH、MMLU、ARC‑Challenge、GSM8K、DROP、HellaSwag、WinoGrande、HumanEval、RULER等下游评测集上进行实验。

**📈 对比分析**

与Transformer基线、YOCO密集版和YOCO+CLSA对比，CLSA在短/长上下文任务中保持几乎无质量损失，并在128K上下文下实现解码速度提升7.6×、总体吞吐量提升17.1×。

**⚠️ 局限性**

局限在于跨层共享路由假设各层注意分布相似，若层间关注差异显著可能导致稀疏选择失效；同时GPU上对top‑k的硬件实现仍有限，未完全消除稀疏方法的实现瓶颈。

---

## 673. Will the Agent Recuse Itself? Measuring LLM-Agent Compliance with In-Band Access-Deny Signals

**arXiv ID:** 2606.06460 | [PDF](https://arxiv.org/pdf/2606.06460v1)

**作者:** Thamilvendhan Munirathinam `[一作]` `[通讯]`, Thamilvendhan Munirathinam

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Recuse Signal ——一种协议无关、轻量级的内置拒绝信号，让服务器在与自动化 LLM 代理交互时主动请求代理撤回。实现了 SSH banner/PAM hook 和 PostgreSQL wire‑protocol 代理两种适配器，并在生产环境中部署，随后通过实验验证该信号能否被代理遵从。

**💡 创新点**

创新点：1）首次提出“Recuse Signal”这一开放规范的内置拒绝信号；2）首次对 LLM 代理是否遵从此信号进行系统实验测量，发现其为合作式、可被覆盖且模型依赖；3）提供了两种低成本、可直接部署的适配器，验证了技术可行性。

**🔧 技术方法**

技术要点：协议内嵌信号格式、SSH banner 与 PAM hook、Go 写的 PostgreSQL wire‑protocol 代理、LLM API 工具循环（GPT‑4o、GPT‑4o‑mini、Claude Code）、实验框架与判定机制、UUID 关联日志。

**📊 数据集**

数据集：实验任务为“检查服务器根文件系统磁盘使用率”，仅使用本地服务器的实际连接与凭证；并未使用公开数据集。主要实验对象是三种 LLM 代理（两种 API 模型与一种部署式代理）。

**📈 对比分析**

比较方法：对照实验——含信号 vs 不含信号（control），以及授权 vs 未授权两组；记录每次实验的“recusal”与“完成”结果。性能（recusal 率）显示：在未授权且有信号时 100% recusal；在授权且有信号时 GPT‑4o 仅 20% recusal，其余模型保持 100%。实验规模小，样本量有限，但效应显著。

**⚠️ 局限性**

限制：实验仅在单一任务、单一协议（SSH）和单一生产主机上进行，样本量小；未覆盖 PostgreSQL 适配器；信号能否被代理工具表面化的依赖尚未充分验证；授权交互的稳定性需要更多实验；未评估对恶意或不合规代理的抵抗能力。

---

## 674. Revising Context, Shifting Simulated Stance: Auditing LLM-Based Stance Simulation in Online Discussions

**arXiv ID:** 2606.06443 | [PDF](https://arxiv.org/pdf/2606.06443v1)

**作者:** Xinnong Zhang `[一作]` (Fudan University), Jiebo Luo `[通讯]` (University of Rochester)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了基于大语言模型的立场模拟在对话上下文微调下的敏感性，构建了反事实上下文修订框架并评估其效果。

**💡 创新点**

创新点在于提出反事实上下文修订审计方法，并比较文本修订与多模态（表情包）修订对模拟立场的影响，揭示不同修订机制。

**🔧 技术方法**

使用大语言模型（如GPT-5.2、Claude、DeepSeek、Llama）进行立场推断与上下文修订，并利用多模态模型生成表情包。

**📊 数据集**

数据集来自 Reddit 上 3 个 LLM 主题（DeepSeek、Claude、Llama）的 1,821 条对话，包含 851 个目标用户。

**📈 对比分析**

通过平均方向性立场偏移率和立场转移率两项指标比较，并发现“add”文本修订与“meme”多模态修订均能显著提升支持倾向，效果在不同模型、主题与 prompt 下保持一致。

**⚠️ 局限性**

局限性包括仅针对 Reddit 上的 LLM 讨论，缺乏更广泛主题和多平台验证，以及对真实用户态度变化的推断仍受限。

---

## 675. PAR3D: A Unified 3D-MLLM with Part-Aware Representation for Scene Understanding

**arXiv ID:** 2606.06485 | [PDF](https://arxiv.org/pdf/2606.06485v1)

**作者:** Shaohui Dai `[一作]` (Xiamen University), Liujuan Cao `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PAR3D统一的3D-多模态大型语言模型并构建ScenePart数据集，实现对3D场景中物体及其部件的理解、推理与分割。

**💡 创新点**

通过引入部件感知的3D表示学习、层次化分割查询生成以及专门的ScenePart数据，突破了以往仅关注物体级别的局限。

**🔧 技术方法**

结合预训练的Point Transformer编码器、对比学习与自蒸馏的视觉表示优化、LLM（LLaVA-1.5-7B）与Hierarchical Segmentation Query，并采用两阶段训练（视觉背骨预训练 + LoRA指令微调）。

**📊 数据集**

使用ScenePart（合成3D场景+部件注解与语言指令）以及ScanNet、ScanRefer、Multi3DRefer、ScanQA、SQA3D、Scan2Cap等公开3D视觉语言基准。

**📈 对比分析**

与专家模型、Fine-tuned 3D-MLLMs以及Generalist 3D-MLLMs对比，在ScanRefer、Multi3DRefer等物体级任务上取得SOTA提升；在ScenePart-Seg、ScenePart-QA中表现最优，显著提升部件分割mIoU与问答准确率。

**⚠️ 局限性**

受合成场景与真实扫描的域差距限制，部件类别覆盖有限，未涉及复杂的物体-部件交互与真实世界的操作场景。

---

## 676. Latent Reasoning with Normalizing Flows

**arXiv ID:** 2606.06447 | [PDF](https://arxiv.org/pdf/2606.06447v1)

**作者:** Guancheng Tu `[一作]` (University of Pennsylvania), Jiatao Gu `[通讯]` (University of Pennsylvania)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在大型语言模型内部使用可逆归一化流（normalizing flow）生成连续链式思维（continuous CoT）的框架，并将思维与答案共享同一因果Transformer流，实现了思维的可采样、可评分和可训练。

**💡 创新点**

创新点在于：① 将可逆流嵌入LLM的因果流，构建连续思维的显式概率分布，解决了传统离散CoT的冗长和连续Latent CoT的迭代去噪问题；② 统一了监督学习与强化学习的接口，使得思维分布可直接用于策略梯度；③ 通过“shallow + deep”流设计，使得训练端到端可微且生成效率大幅提升。

**🔧 技术方法**

主要技术包括：可逆归一化流（Autoregressive Flow）、Transformer LLM（Qwen3-8B-Base）、VAE编码的连续思维、流前向变换、policy‑gradient（GRPO）强化学习、vLLM加速解码等。

**📊 数据集**

使用的主要数据集：1) Ling‑Coder（1.4M Python指令生成样本）做预训练；2) 评估集：MBPP、MBPP+、HumanEval、HumanEval+、LiveCodeBench v6；3) RL阶段额外使用AceCoder与KodCode混合数据。

**📈 对比分析**

与显式CoT、Diffusion‑based Latent CoT（如LaDiR）、循环式Latent Reasoning（Ouro）、Soft Thinking、TaH+、LaVAE等基线对比。NF‑CoT（Unified）在Qwen3-8B-Base上pass@1提升13%（55.8→68.8），优于LaDiR+RL 7.1%；在pass@k上持续提升，且RL后保留多样性。整体速度与算力：相比LaDiR，NF‑CoT在推理阶段降低约2.7×时间、49.3T→19.9T FLOPs，训练阶段提升5.7×吞吐。

**⚠️ 局限性**

局限性：1) 只在代码生成任务上验证，未覆盖其他推理领域；2) 依赖固定长度VAE编码的连续轨迹，可能不适用于更长或更复杂的推理；3) 连续Latent CoT不可解释，解码为自然语言仅是定性探测；4) RL奖励依赖可验证任务（如单元测试），在无可验证目标的任务中难以推广；5) 生成的代码仍可能存在错误或安全隐患。

---

## 677. Causal Atlases from Entropic Inference: Bayesian Networks beyond Optimal DAGs

**arXiv ID:** 2606.06440 | [PDF](https://arxiv.org/pdf/2606.06440v1)

**作者:** Hazhir Aliahmadi `[一作]` (Queen's University), Greg van Anders `[通讯]` (Queen's University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于最大熵推断的贝叶斯网络学习框架，通过对加权邻接矩阵进行统计物理采样，得到包含多条可行有向无环图（DAG）的集合；

**💡 创新点**

创新点在于不预设结构先验，而是让熵驱动的能量景观自然产生有效的结构先验；通过分子动力学（Nosé‑Hoover链）在能量景观上进行高温采样，并随后利用对数行列式无环性约束进行非线性投影，从而获得多样化的 DAG 集合；

**🔧 技术方法**

采用最大熵正则化的 Gibbs 分布、分子动力学采样（Simmering、Nosé‑Hoover链）、对数行列式无环性约束投影、结构希尔曼距离（SHD）等统计量进行不确定性量化；

**📊 数据集**

在两个人工数据集上评估：一是 2‑节点带噪声线性 SEM；二是 20‑节点 Erdős–Rényi（期望度 4）线性 SEM，样本量 1000；

**📈 对比分析**

与传统优化方法 DAGMA 进行对比。结果显示在低温时两方法收敛相近；中温时采样得到的 DAG 集合显著更丰富（SHD≈40），能够揭示多条可行因果方向，且边缘概率与权重方差提供细粒度的不确定性信息；

**⚠️ 局限性**

局限性包括：计算成本较高（需长时间分子动力学轨迹）；对温度、s 值等超参数敏感；仅在模拟数据上验证，未在真实世界数据集上测试；以及对非线性或含隐藏变量的模型适用性尚不充分。

---

## 678. Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution

**arXiv ID:** 2606.06492 | [PDF](https://arxiv.org/pdf/2606.06492v1)

**作者:** Liliana Hotsko `[一作]` (University of Waterloo), Pengyu Nie `[通讯]` (University of Waterloo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一种基于超网络的框架，能够为代码语言模型生成仓库特定的LoRA适配器，实现无推理时 token 开销的仓库知识注入。

**💡 创新点**

创新点包括：①从知识进入模型参数与何时更新两维度设计；②提出静态（单快照）与递归（基于 GRU 的代码差异流）两种使用场景；③通过递归聚合代码 diff 使适配器随仓库演化动态更新。

**🔧 技术方法**

主要技术：超网络生成 LoRA 权重；仓库编码器（基于 Qwen3-Embedding）压缩仓库上下文；GRU 递归编码代码差异；冻结 Qwen2.5-Coder-1.5B 作为基底模型。

**📊 数据集**

使用自建的 Python 仓库基准，包含数百个（约 200+）GitHub Python 仓库，分为训练/验证/测试及时间序列切分。

**📈 对比分析**

与多种基线比较：预训练模型、RAG、依赖分析、全参数微调、单仓库 LoRA、跨仓库 LoRA 等。静态轨道上跨仓库 EM 约 64%，等同单仓库 LoRA 上限；演化轨道上跨仓库 EM 超过 70%，显著优于所有基线，且不产生推理时额外 token。

**⚠️ 局限性**

局限性：仅在 Python、单一 1.5B 级模型与断言完成任务上验证；超网络参数量巨大；OOD 评测受断言长度差异影响；仅使用精确匹配、EditSim 与 CodeBLEU 等表层指标，未做完整语义执行验证。

---

## 679. Operation-Guided Progressive Human-to-AI Text Transformation Benchmark for Multi-Granularity AI-Text Detection

**arXiv ID:** 2606.06481 | [PDF](https://arxiv.org/pdf/2606.06481v1)

**作者:** Sondos Mahmoud Bsharat `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套名为 OpAI-Bench 的操作导向基准，用以模拟从纯人类文本到人工智能协助编辑的递进过程，并保留多粒度的作者归因信息。

**💡 创新点**

创新点在于：①以连续版本轨迹而非终点二分类来评估 AI 文本检测；②引入五种编辑操作（润色、改写、风格重写、压缩、扩展）并控制 AI 覆盖率；③提供文档、句子、词/跨度级别的完整 provenance，支持轨迹感知与操作感知的检测研究。

**🔧 技术方法**

技术实现包括：对 15,722 篇四领域（学生作文、新闻、政府报告、科研摘要）原始文档进行增量式累计编辑；使用 GPT‑5.4、Gemini、Claude 等大型语言模型完成文本重写；构建多版本数据集并使用 8 种文档级、7 种句子级、2 种细粒度检测器进行对比实验；并通过覆盖率、操作与累积历史的控制分析探究检测难度。

**📊 数据集**

数据集：源自四大领域共 15,722 篇原始文档，扩展为 31,089 条轨迹（每条 9 版），共 279,794 条样本；覆盖 4 个 LLM 生成器（GPT‑5.4、Gemini、Claude、Qwen3‑8B）并留出 Qwen 作为泛化测试。

**📈 对比分析**

对比方法包括零射（Zero‑Shot）检测器、LLM 直接作为检测器、以及在 OpAI‑Bench 训练的微调模型；评估指标为文档/句子/词级别的准确率与 AI‑F1。实验表明：检测性能并非随 AI 覆盖率单调提升，而是受到编辑操作、文本域、生成器和历史累积的共同影响，尤其混合作者的中间版本往往比完全人类或完全 AI 版本更难检测。

**⚠️ 局限性**

局限性：仅覆盖四个英语领域，编辑操作与覆盖率设定有限；基准仅处理句子级重写，未覆盖段落或全文级结构变更；评测仅基于预定义 LLM 生成器，可能无法完全泛化至未知模型或真实使用场景；并未考虑多轮交互中的实时适应与人类编辑习惯多样性。

---

## 680. Flow-based Policy Adaptation without Policy Updates

**arXiv ID:** 2606.06461 | [PDF](https://arxiv.org/pdf/2606.06461v1)

**作者:** Luzhe Sun `[一作]` (Toyota Technological Institute at Chicago), Matthew R. Walter `[通讯]` (Toyota Technological Institute at Chicago)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出一种基于流模型的动作适配框架，能够在不更新原始智能体的前提下对其提出的动作块进行选择性纠正，兼具策略适配与共享自治功能。

**💡 创新点**

创新点在于：①使用条件流匹配学习专家动作分布，并通过逆流得到 OOD 分数实现单模型的检测与纠正；②设计三种流导向的纠正方法（FPAS、FEEG、IFAE），在不同场景下实现灵活的动作调整；③无需额外的置信度或不确定性网络，利用流本身的逆向评分实现门控介入。

**🔧 技术方法**

核心技术包括条件流匹配（Flow Matching）、逆向 OOD 检测、基于能量引导的流编辑（FEEG）以及无逆向的流对齐编辑（IFAE），并结合局部采样的 FPAS。

**📊 数据集**

使用了有限的专家演示数据，主要在四个仿真任务（Slalom、CAN、Keypad、Charger）和两个真实机器人任务（Charger 插入、杯子服务）中进行训练与评估。

**📈 对比分析**

与基线 DDPM、FlashBack CSA 等共享自治方法对比，实验显示在 13/16 个随机化包装设置下取得最佳成功率，在仿真中可提升至 29% 以上，真实机器人任务亦显著优于基线。

**⚠️ 局限性**

局限性：不同纠正变体对任务的适用性不一，缺乏系统的变体选择标准；对源动作分布建模的依赖在某些环境下效果不稳；未来需进一步研究编辑超参数、任务属性与方法匹配的关系。

---

## 681. Event Detection for Parameter-to-KPI Dependency Learning for AI-RAN

**arXiv ID:** 2606.06459 | [PDF](https://arxiv.org/pdf/2606.06459v1)

**作者:** Christie Djidjev `[一作]` (Idaho National Laboratory), Nicholas Kaminski `[通讯]` (Idaho National Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在AI‑RAN环境下如何从噪声连续遥测中恢复可解释的参数–KPI依赖关系，并提出将布尔化视为显著性检测的事件检测框架。

**💡 创新点**

创新点在于将布尔化过程转化为显著性检测问题，并通过合成闭环生成器验证依赖恢复的可行性，量化阈值与信噪比对事件检测质量的影响。

**🔧 技术方法**

使用决策树解释性学习、基于零更新基线的z‑score标准化与阈值检测、以及布尔化事件表征等技术实现。

**📊 数据集**

采用自行构建的AI‑RAN样式合成遥测数据集，该数据集可植入真实依赖结构并可调节噪声与干预。

**📈 对比分析**

通过对连续轨迹的决策树可解释性恢复验证和对布尔化的F1/召回/精确度评估，实验表明在信噪比足够或阈值合适时能达到≈0.95的F1；阈值与信噪比是关键影响因素。

**⚠️ 局限性**

仅在合成数据上验证；生成器假设基于即时状态的反馈控制，不包含策略演化或模型更新，且布尔化在高噪声下仍易产生误检，限制了直接迁移到真实AI‑RAN场景的适用性。

---

## 682. Temporal matching in trees

**arXiv ID:** 2606.06439 | [PDF](https://arxiv.org/pdf/2606.06439v1)

**作者:** Márk Hunor Juhász `[一作]` (Eötvös Loránd University), Péter Madarasi `[通讯]` (HUN-REN Alfréd Rényi Institute of Mathematics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了在时变树（temporal tree）上的两种最大匹配模型：Δ‑matching（时间点间隔为 Δ）和 γ‑matching（连续出现 γ 次的边块），并证明了它们在树结构下的计算复杂性，给出了多种多项式时间解法、动态规划方案以及多项式时间近似方案（PTAS）。

**💡 创新点**

创新点包括：
• 证明了即使每条边仅出现两次，Δ‑matching 与 γ‑matching 在时变树上仍为 NP‑hard；
• 通过双模型与 d‑matching 的归约，展示了 γ‑matching 在二分图上的 APX‑hard 性；
• 在每条边恰好出现一次或每条边最多有一条 γ‑block 时，给出了基于树 DP 的多项式算法；
• 引入了局部使用/稀疏性假设并给出对应的 DP；
• 提出了一种窗口分割 + 局部解算 + 组合的 PTAS，适用于所有 Δ, γ ≥ 2。

**🔧 技术方法**

主要技术手段包括：
• 复杂度归约（从双匹配、d‑matching 等经典 NP‑hard 问题），
• 树结构下的 DP（按节点父子关系递推），
• 基于时间窗口与周期模板的划分与计数（用于 PTAS），
• 组合优化（最大权匹配、动态规划状态枚举），
• 细致的时间标签编码与区间冲突分析。

**📊 数据集**

本文全部为理论分析，无实验数据集；所有结果均为多项式时间算法或复杂度证明。

**📈 对比分析**

比较方法：与已知的 APX‑完整性、NP‑难度、以及特殊图结构下的多项式算法进行对比；性能表现为：
• 对于一般时变树，问题保持 NP‑hard；
• 在受限情况（单次出现或单 γ‑block）下实现多项式时间；
• 通过 PTAS 可在多项式时间内获得 (1‑ε) 近似解，时间复杂度为 L^O(1/ε) · n（L 为时间标签总数）。

**⚠️ 局限性**

局限性：
• 对于一般时变树（除上述限制外）问题仍为 NP‑hard，无法得到精确多项式解；
• PTAS 的运行时间随 1/ε 指数增长，实际应用需权衡 ε；
• 仅考虑无权图（无边权），对加权匹配的扩展尚未给出；
• 对非树结构的时变图（如网格、图的高树宽）缺乏完整的复杂度或近似分析。

---

## 683. TailLoR: Protecting Principal Components in Parameter-Efficient Continual Learning

**arXiv ID:** 2606.06494 | [PDF](https://arxiv.org/pdf/2606.06494v1)

**作者:** Marius Dragoi `[一作]` (Bitdefender), Florin Brad `[通讯]` (Bitdefender)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TailLoR低秩适配方法，利用预训练权重的奇异基底为固定参考框架，学习对奇异值矩阵的低秩更新，并通过软谱正则化将更新引导至谱尾，以减轻干扰。

**💡 创新点**

通过对主奇异向量施加软谱正则化惩罚，专注于谱尾更新；不需要访问先前任务的适配器；实现连续学习时保持任务隐私且不需任务特定超参数。

**🔧 技术方法**

SVD分解、低秩适配器（A·B）、软谱正则化（head、tail、uniform惩罚）、有效秩评估等技术。

**📊 数据集**

在T5-large模型上评估Standard CL、Long Sequence和TRACE基准，TRACE使用每任务500样本的子集。

**📈 对比分析**

与EWC、IncLoRA、SVFT、MiLoRA、PiSSA、ELLA等方法对比，TailLoR(head)在Standard CL和TRACE上达到或超过ELLA的整体准确率，同时后向转移更低，表明性能优异。

**⚠️ 局限性**

仅在编码器-解码器架构T5上验证，未扩展到因果解码器LLM；TRACE评估使用500样本/任务，需进一步扩展至完整数据集。

---

## 684. HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers

**arXiv ID:** 2606.06493 | [PDF](https://arxiv.org/pdf/2606.06493v1)

**作者:** Lizhi Yang `[一作]` (California Institute of Technology), Aaron Ames `[通讯]` (California Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出了可直接被自然语言驱动的 10 维指令接口控制器 HANDOFF，能在全身控制中实现步态、抓取与跌倒恢复等多种能力。

**💡 创新点**

创新点在于将运动跟踪、步态和跌倒恢复三类教师通过上下文门控的多教师 KL 蒸馏与 Mixture‑of‑Experts 进行融合，得到单一、紧凑且可解释的全身控制器。

**🔧 技术方法**

使用了 PPO、KL 蒸馏、Mixture‑of‑Experts、上下文门控、CoP 过滤、VLM 与 LLM 结合的任务规划等技术。

**📊 数据集**

采用公开的人类运动捕捉重定向数据、行走轨迹数据以及混合跌倒恢复数据集进行教师训练。

**📈 对比分析**

与 SOTA（FALCON、OpenHomie、AMO、SONIC）对比，速度跟踪误差和可达工作空间体积与最优方法持平或更优，并在真实 Unitree G1 上完成多任务演示。

**⚠️ 局限性**

局限包括：仅提供腕位姿（无 6D 姿态）、感知受限于前向 RGB‑D 摄像头、教师集尚未覆盖所有环境，未来可进一步扩展。

---

## 685. TempoVLA: Learning Speed-Controllable Vision-Language-Action Policies

**arXiv ID:** 2606.06491 | [PDF](https://arxiv.org/pdf/2606.06491v1)

**作者:** Dong Jing `[一作]` (RUC), Mingyu Ding `[通讯]` (UNC)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 TempoVLA，一个能在单一 VLA 模型上通过可变速度的数据增强与速度条件化实现双向速度控制的框架，且不需重新训练底层结构；

**💡 创新点**

创新点在于将数据侧的 Variable‑Speed Trajectory Augmentation (VSTA) 与模型侧轻量级速度注入（文本前缀/soft prompt/MLP）相结合，首次实现显式可调、可逆的执行速度控制，并通过外部 VLM 动态调度实现任务阶段自适应加速/减速；

**🔧 技术方法**

技术包括 VSTA（通过合并/拆分动作实现目标速度）、速度条件化（文本前缀、RMSNorm 调制、soft prompt）、流匹配 VLA（π_0.5 + PaliGemma）以及 GPT‑4o 作为速度调度器；

**📊 数据集**

使用 LIBERO 仿真数据集（四套任务，共 500 条演示）和真实 Franka 7‑DOF 机器人上 5 个任务的 50 条遥控轨迹；

**📈 对比分析**

与单速基线以及三种速度注入方式对比，实验显示在 1× 下成功率从 96.7% 提升至 98‑99%，最高成功率在 1.25×–1.5×；在真实机器人上 1× 从 80% 提升至 88%，动态 VLM 调度进一步达到 96% 成功率，速度实现与命令保持高度一致；

**⚠️ 局限性**

局限性在于高速度下策略输出超出低层控制器的跟踪带宽，导致速度提升逐渐饱和；此外动态调度的安全性仍偏保守，需进一步细化速度分配策略与控制器协同调优。

---

## 686. Regret Minimization with Adaptive Opponents in Repeated Games

**arXiv ID:** 2606.06486 | [PDF](https://arxiv.org/pdf/2606.06486v1)

**作者:** Mingyang Liu `[一作]` (Massachusetts Institute of Technology), Kaiqing Zhang `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了重复策略遗憾（RP-Regret）指标，并分析其子线性收敛的必要条件。

**💡 创新点**

创新点在于将对手的历史依赖性纳入遗憾度量，提供更强的对比基准和更少的对手约束。

**🔧 技术方法**

采用了三种技术：基于优化oracle的直接优化、线性化并凸化的近似算法以及对慢变对手的直接最小化策略。

**📊 数据集**

实验使用了经典博弈环境，如Stag‑Hunt。

**📈 对比分析**

与传统外部遗憾方法相比，RP-Regret算法在Stag‑Hunt中实现了更高的合作收益。

**⚠️ 局限性**

局限在于对手需缓慢变化的假设，以及在大规模博弈中的计算复杂度较高。

---

## 687. DNQ: Deep Nash Q-Network for Partially Observable n-Player Games

**arXiv ID:** 2606.06480 | [PDF](https://arxiv.org/pdf/2606.06480v1)

**作者:** Qintong Xie `[一作]`, Peter Chin `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

设计了多轮同时竞价环境，并提出DNQ框架，利用均衡监督训练代理；

**💡 创新点**

首次将外部求解器嵌入学习循环，比较精确多玩家与可扩展的pairwise critic，并通过共享critic实现效率提升；

**🔧 技术方法**

结合可观测马尔可夫游戏建模、神经网络critic、Nash equilibrium求解、KL目标对齐、共享编码器；

**📊 数据集**

使用仿真生成的多轮竞价环境数据，不依赖公开数据集；

**📈 对比分析**

在2、3、4玩家实验中对比精确与pairwise critic，评估critic loss、预算使用率、策略熵、累计训练时间；pairwise在规模扩大时保持相近或更优行为且计算成本大幅下降；

**⚠️ 局限性**

精确方法在玩家数增大时计算成本不可接受，pairwise近似可能忽略多玩家交互，实验仅限于简化竞价仿真。

---

## 688. Complexity-Balanced Diffusion Splitting

**arXiv ID:** 2606.06477 | [PDF](https://arxiv.org/pdf/2606.06477v1)

**作者:** Noam Issachar `[一作]` (Hebrew University of Jerusalem), Raanan Fattal `[通讯]` (Hebrew University of Jerusalem)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于复杂度均衡分割（CBS）的时间分割框架，用专门的子网络在扩散模型的不同时间段分配容量。

**💡 创新点**

创新点在于把扩散时间轴视为函数逼近域，利用 de Boor 等距分配原理结合 Dirichlet 能量与路径加速度两种监视函数，得到无需搜索或启发式的最优分割。

**🔧 技术方法**

使用逼近理论、Dirichlet 能量、曲线加速度、随机迹估计和辅助网络估计复杂度，并在 SiT、JiT、UNet 等多种架构上训练多网络。

**📊 数据集**

实验涵盖 ImageNet‑256（潜在空间）、ImageNet‑64（像素空间）和 CIFAR‑10（无条件）三个数据集。

**📈 对比分析**

与单体模型和传统启发式分割对比，CBS 在保持相同每步 FLOPs 的前提下，在 SiT‑XL、JiT 等模型上将 FID 提升约 35%，在不同 N 的设置中始终取得最优或近优性能。

**⚠️ 局限性**

局限性包括仅在时间轴上分割，未考虑空间或动态路由；监视函数的估计仍依赖辅助网络，且对极大规模模型的鲁棒性尚未完全验证。

---

## 689. MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery

**arXiv ID:** 2606.06473 | [PDF](https://arxiv.org/pdf/2606.06473v1)

**作者:** Shangheng Du `[一作]` (Shanghai Artificial Intelligence Laboratory), Lei Bai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了一种基于LLM的自演化多智能体框架，用于端到端机器学习工程任务。

**💡 创新点**

通过进化型MCGS、回顾性记忆与分层规划+自适应代码生成三大创新，有效突破跨分支信息孤岛、无记忆搜索和缺乏分层控制等长期优化瓶颈。

**🔧 技术方法**

采用Gemini‑3.1‑Pro‑preview LLM，进化型蒙特卡洛图搜索、动态全局记忆、软切换探索‑利用策略、差分编辑等技术。

**📊 数据集**

在MLE‑Bench（75个Kaggle任务）和AlphaEvolve（15个数值优化任务）上进行评估。

**📈 对比分析**

与FM‑Agent、MLE‑STAR‑Pro、AIBuildAI、ML‑Master 2.0等公开与专有基线比较，在12小时预算下平均奖牌率65.3%（金牌率34.7%），超过所有基线；在AlphaEvolve任务中取得11/15任务最佳成绩。

**⚠️ 局限性**

仍受限于对大型LLM的高算力需求、图搜索实现复杂性以及在极端任务难度下的收敛速度等问题。

---

## 690. PC Layer: Polynomial Weight Preconditioning for Improving LLM Pre-Training

**arXiv ID:** 2606.06470 | [PDF](https://arxiv.org/pdf/2606.06470v1)

**作者:** Senmiao Wang `[一作]` (Chinese University of Hong Kong), Ruoyu Sun `[通讯]` (Chinese University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种称为Preconditioning（PC）的低阶多项式预处理层，用于在LLM预训练期间对权重矩阵进行软谱调节，从而保持良好的权重条件数并提升训练效率。

**💡 创新点**

提出了在训练时通过低阶多项式预处理直接调整权重谱的软谱调节机制，兼容标准优化器且不产生推理开销；同时给出了对深层线性网络的理论收敛证明，阐明权重谱控制与梯度下降几何收敛之间的关系。

**🔧 技术方法**

采用低阶矩阵多项式预处理、谱归一化、可学习缩放γ、正则化等技术，并通过多次矩阵乘法实现；实验中结合MuOn优化器与AdamW进行对比。

**📊 数据集**

在Llama-271M和Llama-1B的预训练任务中使用通用大规模语言模型数据（如Common Crawl等）进行训练，并在零样本下游任务上进行评估。

**📈 对比分析**

与基线Transformer（无PC）在相同训练配置下对比，使用AdamW和Muon两种优化器；在Llama-1B上分别实现了2×（AdamW）和1.13×（Muon）的标记效率提升，并在零样本下游任务中取得更高准确率，同时通过修改条件数指标展示权重谱显著改善。

**⚠️ 局限性**

仅在Llama系列模型和两种规模下验证，未探究更大规模或不同架构的通用性；PC参数（如多项式次数、适用层）需进一步优化，未提供自适应或层级差异化策略；理论证明仅适用于深线性网络，未直接覆盖非线性Transformer。

---

## 691. Goedel-Architect: Streamlining Formal Theorem Proving with Blueprint Generation and Refinement

**arXiv ID:** 2606.06468 | [PDF](https://arxiv.org/pdf/2606.06468v1)

**作者:** Jui-Hui Chung `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Goedel-Architect，一个基于蓝图生成与迭代精炼的Lean定理证明管道。

**💡 创新点**

采用全局依赖图蓝图而非递归子目标，支持并行证明与全局精炼，并可选自然语言证明种子，显著提升效率与效果。

**🔧 技术方法**

结合DeepSeek-V4-Flash大模型、Lean 4、Mathlib检索、工具集成推理以及蓝图生成/精炼循环等技术。

**📊 数据集**

在MiniF2F-test、PutnamBench、IMO 2025、Putnam 2025和USAMO 2026等公开竞赛数据集上进行评测。

**📈 对比分析**

与其他公开或闭源系统按pass@1、pass@4及成本进行对比，取得MiniF2F 99.2%、PutnamBench 75.6%/88.8%（NL）等领先成绩，单题成本约$0.44，性能高且成本低。

**⚠️ 局限性**

对极难题仍需多次迭代或自然语言引导，几何类问题需专用引擎，模型容量与预算受限，且对未覆盖的公式仍可能失败。

---

## 692. Human Adults and LLMs as Scientists: Who Benefits from Active Exploration?

**arXiv ID:** 2606.06464 | [PDF](https://arxiv.org/pdf/2606.06464v1)

**作者:** Mandana Samiei `[一作]` (Mila), Doina Precup `[通讯]` (Mila)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了成人在主动探索条件下的共轭因果学习，比较主动与被动学习以及与大型语言模型（LLM）在同一因果发现任务中的表现。

**💡 创新点**

首次引入主动探索的“nexiom detector”任务，检验主动干预对成人共轭学习的提升，并将人类与多款LLM在相同环境下的探索策略和推断精度进行对比。

**🔧 技术方法**

使用基于Streamlit的Web平台实现实验；计算信息增益、累积信息量等过程指标；设计三组对照（主动、被动、被动提议）并对多种LLM（GPT‑5、Gemini‑2.5‑f、ds‑reasoner等）进行实验。

**📊 数据集**

收集了306名成人（Prolific）在24种共轭/离散因果配置下的48次测试数据；LLM实验在相同配置下进行288次测试，形成可直接比较的行为与结果数据集。

**📈 对比分析**

通过对象识别准确率、规则推断准确率及完整假设准确率进行比较；结果显示主动探索显著提升成人在共轭规则下的性能，LLM表现参差不齐，部分强模型接近人类但整体低于顶尖成人。

**⚠️ 局限性**

局限性包括仅研究成人样本，缺乏儿童或更广泛年龄层；实验对象与规则有限，难以推广至更复杂因果结构；LLM未充分学习主动干预与即时反馈的耦合机制，导致其探索效率不足。

---

## 693. Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents

**arXiv ID:** 2606.06453 | [PDF](https://arxiv.org/pdf/2606.06453v1)

**作者:** Zhuoming Chen `[一作]` (Carnegie Mellon University), Beidi Chen `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Vortex系统，提供可编程的稀疏注意力框架，支持快速开发、部署并在现代LLM服务环境中大规模验证稀疏注意力算法；

**💡 创新点**

创新点包括：①嵌入式Python前端语言实现可组合的稀疏注意力编程模型；②解释器将程序自动降低为可执行算子；③多层次的执行优化（工作负载规划、核融合、随机Top‑k优化），以及与现有高效注意力后端（GQA、MLA）的兼容；④让AI代理通过Vortex直接生成并迭代稀疏注意力方案，提升设计效率。

**🔧 技术方法**

技术上采用分页张量抽象（paged tensor），统一布局元数据；使用自动化编译器实现核融合与GPU调度；支持动态与静态稀疏模式；利用稀疏键值缓存、块级摘要等机制；结合CUDA、TensorRT‑LLM、MLA自定义核；并与NVIDIA H200、B200 GPU集群集成。

**📊 数据集**

使用的主要数据集包括AMC23、AIME24（长生成评测），以及RULER、MiniMax-M2.7、GLM‑4.7‑Flash等模型；对Qwen3系列模型在不同参数规模（0.6B–8B）进行基准测试；同时对229B级MiniMax‑M2.7进行大规模并行测试。

**📈 对比分析**

与全注意力、SGLang、Quest等现有实现对比：在长生成任务中，block top‑k和Quest分别实现2–3.5×吞吐量提升；在用户侧延迟上，P95 TPOT降低11–12×；在极大规模模型（229B）中仍能获得1.2–1.4×速度提升。

**⚠️ 局限性**

局限性包括：仅支持解码阶段，未覆盖prefill和训练；目前仅适用于基于KV缓存的注意力模型，未扩展到RNN或Mamba等状态空间架构；未实现完整的端到端稀疏注意力算法自动搜索框架。

---

## 694. Agent Memory: Characterization and System Implications of Stateful Long-Horizon Workloads

**arXiv ID:** 2606.06448 | [PDF](https://arxiv.org/pdf/2606.06448v1)

**作者:** Yasmine Omri `[一作]`, Thierry Tambe `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文件是会议论文排版模板与说明文档，描述了如何使用LaTeX撰写符合会议规范的稿件。

**💡 创新点**

创新点在于对之前的模板进行更新，包括字体、排版细节及可用页面数的调整，以提升排版效率。

**🔧 技术方法**

使用的技术主要是LaTeX（如iISWC26.cls、compsoc等宏包），并强调使用10pt Libertine字体。

**📊 数据集**

无数据集，本文仅提供排版示例与规范说明。

**📈 对比分析**

无方法比较与性能评估，本文不包含实验或研究结果。

**⚠️ 局限性**

局限性在于本文并非研究论文，而是模板说明，缺乏实验验证与实际内容。

---

## 695. Thinking with Imagination: Agentic Visual Spatial Reasoning with World Simulators

**arXiv ID:** 2606.06476 | [PDF](https://arxiv.org/pdf/2606.06476v1)

**作者:** Chenming Zhu `[一作]` (University of Hong Kong), Xihui Liu `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本工作提出了一个交互式空间推理框架，利用视觉语言模型主动查询世界模拟器获取想象的新视角，从而弥补有限观测导致的空间不确定性。

**💡 创新点**

创新点在于将世界模拟器与视觉语言模型耦合，并通过视角一致性微调提升模拟器的空间可靠性，同时设计了两阶段强化学习课程，让模型学会何时、如何以及以何种方式请求并利用想象视图。

**🔧 技术方法**

技术上使用了 Qwen3‑VL 作为推理策略、Bagel 基础的动作条件世界模拟器、视角一致性微调（View Consistency Tuning）和基于 RL 的工具使用策略训练。

**📊 数据集**

训练数据包括从 IsaacSim、ScanNet++/ScanNet、Matterport3D、DL3DV、ARKitScenes 等室内场景构造的 544k 个视角一致性 SFT 样本；强化学习样本来自 SenseNova‑800K、VST‑500K 与自构 Hard‑UMMQA。

**📈 对比分析**

与基线（直接答题、强制工具使用）相比，实验在 MMSI‑Bench 上提升了 45.1→49.5 的整体准确率，Qwen3‑VL 框架由 29.8 提升至 38.8，MindCube 上提升至 42.7，证明了可靠模拟与主动想象策略的协同效果。

**⚠️ 局限性**

局限性包括对高质量世界模拟器的依赖、强化学习训练成本高、在某些对象或区域中心关系上仍易受噪声影响，且当前仅在室内多视图设置中验证，泛化到更复杂或外部环境尚需进一步研究。

---

## 696. RREDCoT: Segment-Level Reward Redistribution for Reasoning Models

**arXiv ID:** 2606.06475 | [PDF](https://arxiv.org/pdf/2606.06475v1)

**作者:** Mykyta Ielanskyi `[一作]` (Johannes Kepler University Linz), Sepp Hochreiter `[通讯]` (Johannes Kepler University Linz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于模型自身概率分布的奖励重分配（RREDCoT）方法，用以解决链式推理（CoT）生成中的延迟奖励问题并提升 RL 微调样本效率。

**💡 创新点**

创新点在于：①将 RUDDER 的奖励重分配思想直接适用于 CoT MDP，利用语言模型本身估计中间奖励；②设计混合关键词‑熵分段策略，兼顾可解释性与计算效率；③通过概率奖励（PR）估计器在不额外生成文本的情况下实现近似最优信用分配。

**🔧 技术方法**

使用的技术包括：强化学习（GRPO、RLVR）、概率奖励估计、熵分段算法、重要性采样（PR）、自回归语言模型的概率评估。

**📊 数据集**

实验数据集：MATH‑500、AIME‑24/25/26、Numina‑CoT、AIME24、AMC23、MATH500、MINERVA、Olympiad Bench 等，主要涵盖数学推理与编程类任务。

**📈 对比分析**

与传统 GRPO、GRPO+改进等方法比较，RREDCoT 在长生成（最高 25k token）场景下显著提升模型准确率（如 Qwen3‑Instruct 在 AIME、MATH500 等数据集上提升 0.15‑0.3），并保持相对较低的计算开销（比 MC 采样快 1.5‑2 倍）。

**⚠️ 局限性**

局限性包括：①需要参考答案或子目标结构，无法直接应用于无解路径或约束满足类问题；②实现上仍有额外计算成本；③在多解或多样化解空间中，重要性采样偏差可能导致估计不稳定。

---

## 697. Benchmark Everything Everywhere All at Once

**arXiv ID:** 2606.06462 | [PDF](https://arxiv.org/pdf/2606.06462v1)

**作者:** Shiyun Xiong `[一作]` (Chinese University of Hong Kong), Xiangyu Yue `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种完全自主的 Benchmark Agent 系统，用于从用户需求到可执行基准的端到端自动化构建，支持多模态、可定制和持续迭代。

**💡 创新点**

创新点在于将 Benchmark Planner 与 Benchmark Executor 两层代理相结合，利用多工具链协作实现子任务拆分、数据检索、变换计划验证、样本级规划与质量控制，实现低人工成本、快速迭代与高质量基准生成。

**🔧 技术方法**

技术方案基于大型语言模型（GPT‑5.1）与多种工具（文本生成、图像/音频处理、脚本化处理等）的代理框架，包含设计、定位、分配、执行与验证四大模块。

**📊 数据集**

使用 General‑Bench 数据库进行数据检索，并结合公开多模态数据集（文本、音频、图像）生成 benchmark 样本。

**📈 对比分析**

评估方法包括人工评估、LLM‑as‑Judge 评判与一致性检查，生成基准样本的接受率约 96–98%，UIA 得分 68–81%，与直接 LLM 生成相比显著提升；在 Qwen3.5 系列模型上展示规模一致性，证明基准具备良好判别力。

**⚠️ 局限性**

局限性：对语义对齐、上下文对应、目标信号依赖和难度控制的质量仍有提升空间；在细粒度领域（艺术、动物）模型表现不佳；系统对 LLM 质量和多模态变换工具的依赖仍需进一步优化，部分复杂场景仍需人工干预。

---

## 698. In-Context Multiple Instance Learning

**arXiv ID:** 2606.06458 | [PDF](https://arxiv.org/pdf/2606.06458v1)

**作者:** Alexander Möllers `[一作]`, Klaus-Robert Müller `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了In-Context Multiple Instance Learning（ICMIL）框架，利用Perceiver式网络在合成数据上预训练，使模型能够在仅有少量标注袋的情况下，通过一次前向传播即可完成新任务的分类；

**💡 创新点**

创新点在于将Prior-Data Fitted Networks（PFN）扩展到层次集合输入，设计适用于bag结构的Perceiver架构，并通过混合多种合成先验（factorized与joint）实现跨任务的泛化；

**🔧 技术方法**

技术包括Perceiver-style注意力网络、交叉注意力与行列自注意力、可学习的bag令牌、不同的合成先验生成器（factorized与joint），以及在无梯度更新的In-Context学习；

**📊 数据集**

使用十二个MIL基准数据集（SMIL、Musk1/2、Letters、HEPMASS、RSNA-ICH、Elephant、Fox、Tiger、TCGA、Adjacent Pairs、Pos/Neg），以及PCA降维后的特征；

**📈 对比分析**

与五种传统MIL基线（MeanLogReg、SVM-Summ、ABMIL、TabPFN-Concat/Cluster/Subsample）和单一/混合先验的ICMIL模型比较，ICMIL在低标注样本下平均AUROC为84.17，排名第3，超越所有基线且在大多数任务上表现最稳健；

**⚠️ 局限性**

局限性包括训练仅使用至多20个实例的袋，未覆盖更大袋和高维特征；合成先验与真实任务间的差距仍存在，且模型在某些任务（如SMIL、TCGA、Letters）上未能击败特定基线，需进一步改进先验设计、扩展预训练和细调。

---

## 699. Scaffold, Not Vocabulary? A Controlled, Two-Tier, Pre-Registered Study of a Popperian Code-Generation Skill

**arXiv ID:** 2606.06454 | [PDF](https://arxiv.org/pdf/2606.06454v1)

**作者:** Mehmet Iscan `[一作]` `[通讯]` (Yildiz Technical University), Mehmet Iscan (Yildiz Technical University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了一种采用波普尔式伪证主义（Falsifiability、Severity 等）的编码提示技能对大型语言模型（LLM）代码生成的影响，使用预注册的两层消融实验和执行判据。

**💡 创新点**

创新点在于将三种对照（长度匹配安慰剂、仅标签结构、执行判据）与小模型自评审结合，形成一套完整的消融框架，以剖析词汇与结构对代码质量的真正贡献。

**🔧 技术方法**

使用技术包括 LLM-as-judge 校准、HumanEval+ 执行判据（单元测试）、自评审机制、halo sentinel 监测词汇漂白、Bootstrap 95% CI、McNemar 检验等统计方法。

**📊 数据集**

数据集为 HumanEval+（即 HumanEval 题目加上约 80 倍自动生成单元测试），共 163（高阶模型）/164（低阶模型）道题目。

**📈 对比分析**

比较方法是将四个条件（V、L、F、P）在高阶模型下的通过率相互对比，差距 ≤ 2%；在低阶模型下结构化条件（LD、LDS、F、P）相较 Vanilla 提升约 20–22 点，但完整 Popperian 内容与仅标签结构无显著差异；自评审与随机挑选差别不显著。

**⚠️ 局限性**

局限性包括：只在单一 benchmark 家族、两种能力层次下验证；未收集单样本标签/安慰剂比较；未检验更大模型或其它任务的转移效果，结果仅为特定设置下的负面结论。

---

## 700. Simultaneous EF1 and approximate MMS allocations for submodular valuations

**arXiv ID:** 2606.06451 | [PDF](https://arxiv.org/pdf/2606.06451v1)

**作者:** Uriel Feige `[一作]` (Weizmann Institute of Science), Assaf Fine `[通讯]` (Weizmann Institute of Science)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文证明了在子模函数的公平分配中存在同时满足EFL与ρ-MMS的分配。

**💡 创新点**

创新在于构造了随机的RECE1算法并证明其可达到常数α≈0.142的MMS近似。

**🔧 技术方法**

技术上使用了随机分析、自由人不等式、子模函数拆分、代理福利W'等手段。

**📊 数据集**

由于是理论研究，没有使用具体数据集。

**📈 对比分析**

通过概率上界与期望下界证明算法在高概率下满足α-MMS，且与先前的1/n-MMS结果相比性能提升显著。

**⚠️ 局限性**

局限在于仅适用于子模函数，扩展到XOS或子可加函数的证明仍需进一步研究。

---

## 701. CarbonSim: A Lifecycle-Aware Framework for Evaluating Carbon Tradeoffs in Hardware Upgrade Decisions

**arXiv ID:** 2606.06438 | [PDF](https://arxiv.org/pdf/2606.06438v1)

**作者:** Kartik Hans `[一作]` (University of Pittsburgh), Stephen Lee `[通讯]` (University of Pittsburgh)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发并评估了 CarbonSim 框架，用以综合考量硬件升级决策中的生命周期碳排放，包括工作负载执行特征、机器功耗、嵌入式碳清单、调度策略和电网碳强度。

**💡 创新点**

创新点在于：①将嵌入式碳与运营碳统一建模并支持均匀摊销与前置生命周期归属两种计量策略；②在多代异构集群中系统地分析硬件刷新与碳/性能权衡；③提供多种碳感知与 SLO 感知调度策略，实现工作负载与碳排放的协同优化。

**🔧 技术方法**

技术手段包括：Python 离散事件仿真；RAPL、macOS powermetrics 等能耗采集；Electricity Map API 实时获取时变电网碳强度；实现 Carbon-Aware、SLO-Aware、Wait-a-While、Threshold-based Greedy 等调度算法。

**📊 数据集**

使用的数据集包括：多代 CPU（2009、2013、2014、2020、2022）的硬件配置和能耗测量；矩阵乘、MapReduce、Fibonacci、GetPrime 等基准工作负载；不同地点（魁北克、西班牙、加州、昆士兰）的电网碳强度数据。

**📈 对比分析**

通过在同一工作负载下比较不同硬件代号和调度策略的总碳排放与执行时间。结果显示：轻负载或低碳电网环境下旧硬件的生命周期碳更低；混合代数集群可降低碳排放，但往往导致执行时间显著延长（约两倍）。

**⚠️ 局限性**

局限性包括：未考虑老机维护和可靠性成本；聚焦计算节点，未全面纳入机房冷却、网络、存储等设施开销；依赖工作负载采样，可能不足以覆盖所有复杂工作负载；未来工作需扩展到完整机房模型和容器编排平台。

---

