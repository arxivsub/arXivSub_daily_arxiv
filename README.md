# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-17 | 今日论文总数: 536

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Chinese Essay Rhetoric Recognition Using LoRA, In-context Learning and Model Ensemble

**arXiv ID:** 2604.14167 | [PDF](https://arxiv.org/pdf/2604.14167v1)

**作者:** Yuxuan Lai `[一作]` (Open University of China), Chen Zheng `[通讯]` (Engineering Research Center of Integration and Application of Digital Learning Technology, Ministry of Education)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大型语言模型（LLM）通过LoRA微调和提示学习，完成中文修辞识别任务，涵盖细粒度修辞类型分类与修辞成分抽取。

**💡 创新点**

提出将JSON输出键翻译为中文并与自然语言对齐，结合LoRA与提示学习的联合训练与多模型集成，显著提升了结构化修辞抽取的效果。

**🔧 技术方法**

采用LoRA微调、基于上下文的提示学习、线性加权模型集成、以及后处理的回退机制，并使用Qwen系列大模型与开源的Qwen 2.5 72B。

**📊 数据集**

使用CCL 2025中文论文修辞识别评估数据集（训练50例，测试37459例）。

**📈 对比分析**

相较于官方基线和其他参赛队伍，在三个评估轨道上均取得最高分（Track‑1 47.18，Track‑2 54.03，Track‑3 39.94），平均提升约2.6分。

**⚠️ 局限性**

限制在于对大型开源模型的JSON解析失败率较高，且在高复杂度查询下提示学习效果下降。

---

## 2. Heat and Matérn Kernels on Matchings

**arXiv ID:** 2604.14331 | [PDF](https://arxiv.org/pdf/2604.14331v1)

**作者:** Dmitry Eremeev `[一作]` (Higher School of Economics), Viacheslav Borovitskiy `[通讯]` (University of Edinburgh)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种构建匹配空间几何核的框架，解决了在匹配问题中应用核方法的挑战。

**💡 创新点**

创新点在于提供了对静态核的完整表征，并专注于热核和Matérn核族，同时引入了一种新的亚指数算法以高效评估核。

**🔧 技术方法**

使用了几何核方法，特别是热核和Matérn核，并引入了基于区域多项式的算法来提高计算效率。

**📊 数据集**

论文中没有具体提到使用的数据集，但提到匹配在体育分析、物流和规划等领域的应用。

**📈 对比分析**

与传统方法相比，提出的算法在计算复杂度上显著降低，从超指数级降低到亚指数级，能够处理更大的问题规模（例如n=25）。

**⚠️ 局限性**

限制在于尽管引入了新的算法，但仍然存在超多项式的计算复杂度，并且在将几何核推向系统树时未能保持几何结构。

---

## 3. Chronological Knowledge Retrieval: A Retrieval-Augmented Generation Approach to Construction Project Documentation

**arXiv ID:** 2604.14169 | [PDF](https://arxiv.org/pdf/2604.14169v1)

**作者:** Ioannis-Aris Kostis `[一作]` (Université Catholique de Louvain), Pierre Schaus `[通讯]` (Université Catholique de Louvain)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种时间感知的检索增强生成（RAG）框架，用于在大型建筑项目的会议纪要中实现对决策历史的对话式检索。

**💡 创新点**

创新点在于将时间索引与RAG结合，按时间段分块检索并重排序，以保持检索结果的时间连贯性；同时引入输入门控和回答合并模块，提升系统鲁棒性与可读性。

**🔧 技术方法**

使用了语义检索（Dense+BM25）+ColBERTv2重排序、Llama3系列LLM、RRF融合、时间分段索引、输入门控与回答合并等技术。

**📊 数据集**

使用了匿名化的比利时大型建筑项目会议纪要数据集，60份PDF文件，覆盖2022-2024年，共约3891+1107词。

**📈 对比分析**

在13个人工标注的基准查询上进行检索评估，采用HitRate@k、Precision@k、Recall@k和F1@k；在k=10、开启重排序的设置下，Precision@5≈0.74、Recall@5≈0.46、F1≈0.50，整体性能优于未重排序配置。

**⚠️ 局限性**

局限性包括仅在单一案例评估，未在公开标准基准上验证；重排序导致查询延迟显著增加；模型依赖Llama3系列，处理大规模知识库时受限；输入门控与回答合并依赖LLM判断，可能出现误判。

---

## 4. NuHF Claw: A Risk Constrained Cognitive Agent Framework for Human Centered Procedure Support in Digital Nuclear Control Rooms

**arXiv ID:** 2604.14160 | [PDF](https://arxiv.org/pdf/2604.14160v1)

**作者:** Xingyu Xiao `[一作]` (Tsinghua University), Haitao Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5545 | [OpenAlex ID](https://openalex.org/A5006641842)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 NuHF‑Claw，针对数字化核电站主控室的“软控制”导致的认知风险，构建一个持续运行的风险约束式人工智能代理框架，实时预测并干预操作员错误；

**💡 创新点**

1) 将认知建模自动化：利用大型语言模型即时生成 ACT‑R Lisp 代码，形成持续运行的“人类数字双胞胎”；2) 引入风险约束运行时，将认知状态推理与实时概率安全评估结合，避免 LLM 幻想；3) 通过治理安全门（贝叶斯网络）实现人机中心化自主决策；

**🔧 技术方法**

ACT‑R 认知架构、AutoGraph 任务建模、KRAIL 基于 LLM 的 PIF 分析、贝叶斯网络治理、安全门、LSTM 事件诊断、深度学习数据流；

**📊 数据集**

HTR‑PM600 全景模拟器实时工况数据（约 33 维），模拟器日志、光标轨迹、手工验证卡；

**📈 对比分析**

与传统 HUNTER、DRIF 等静态/离线 HRA 方法对比，NuHF‑Claw 在实验中实现了对操作员工作负荷的实时估计、对高风险界面导航的及时警示，并在风险阈值触发时自动限制 LLM 推荐，展示了比传统方法更低的错误预期和更高的干预成功率；

**⚠️ 局限性**

对单人操作的建模仍有限，无法覆盖多操作者团队协同；LLM 推理延迟和 ACT‑R 代码生成速度仍是实时性瓶颈；数据集仅来自模拟器实验，缺乏现场真实工况验证。

---

## 5. Listen, Correct, and Feed Back: Spoken Pedagogical Feedback Generation

**arXiv ID:** 2604.14177 | [PDF](https://arxiv.org/pdf/2604.14177v1)

**作者:** Junhong Liang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Fajri Koto `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1206 | [OpenAlex ID](https://openalex.org/A5065822589)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了口语语法错误纠正（SGEC）中的教学反馈生成，提出并构建了新的数据集SPFG，并在此数据集上评估了多种大型语言模型的表现。

**💡 创新点**

创新点在于：①构建了SPFG数据集，包含教师风格的反馈与对应的首选/拒绝反馈对，支持偏好学习；②提出两阶段训练框架（偏好对齐+监督微调）和令牌掩蔽策略，以更好地学习纠错与反馈生成任务。

**🔧 技术方法**

技术方法包括：基于指令调优的LLM（Qwen2.5、Llama‑3.1、GLM‑4等）进行监督微调（SFT），利用LoRA进行参数高效微调；对比Direct Preference Optimization (DPO) 与 Kahneman–Tversky Optimization (KTO) 的偏好对齐；使用GPT‑4o 进行自动评测。

**📊 数据集**

使用数据集：来自 Speak & Improve 2025 挑战的学习者语音记录，经过转写、语法纠正及教师式反馈生成后，形成SPFG，包含 4,285 条训练样本、500 条验证样本、2,793 条评测样本。

**📈 对比分析**

实验对比了专有模型 DeepSeek‑Chat、Gemini‑2.5‑Flash、Qwen‑Plus 与开源模型 Qwen2.5、Llama‑3.1、GLM‑4；结果显示，SFT 在 WER、ERRANT 以及 GPT‑4o 评估的反馈质量（Correctness、Level Appropriateness、Suggestion Quality、Positiveness）上均优于基线，DPO/KTO 的提升有限；在开源模型中，SFT 的 F_0.5 最高，专有模型虽然纠错低、WERS 较高，却在反馈质量上略高。

**⚠️ 局限性**

局限性包括：①偏好对齐（DPO/KTO）对纠错和反馈提升有限；②纠错质量与反馈质量弱相关，表明两者仍需分离建模；③评估高度依赖 GPT‑4o 自动判分，人工评测覆盖有限；④对口语输入的鲁棒性未充分验证，主要在文本转写上进行实验。

---

## 6. Step-level Denoising-time Diffusion Alignment with Multiple Objectives

**arXiv ID:** 2604.14379 | [PDF](https://arxiv.org/pdf/2604.14379v1)

**作者:** Qi Zhang `[一作]` (Arizona State University), Shaofeng Zou `[通讯]` (Arizona State University)

**通讯引用:** 1032 | [OpenAlex ID](https://openalex.org/A5012545205)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了无训练、无奖励访问的多目标扩散模型对齐方法MSDDA，利用step‑level RL得到闭式解并在推理时融合多目标模型

**💡 创新点**

创新点在于将RL fine‑tuning reformulate为step‑level RL，推导出不引入近似的闭式对齐目标，并实现可在推理时即时组合多目标模型

**🔧 技术方法**

采用step‑level RL与DPO、KL‑regularized MDP、Gaussian组合等技术，构建闭式对齐目标并推导对应采样公式

**📊 数据集**

使用Stable Diffusion v1.5预训练模型，DrawBench（color）子集作为prompt数据，ImageReward和VILA作为奖励模型评估

**📈 对比分析**

与SD、RGG、CoDe、RS、DB‑MPA等基线对比，MSDDA在ImageReward和VILA两项奖励上均优于基线，推理时间仅比SD/RS慢约一倍

**⚠️ 局限性**

局限性在于仍需预先训练单目标模型；对多目标方差差异有要求；对不同架构的模型融合兼容性有限

---

## 7. MARCA: A Checklist-Based Benchmark for Multilingual Web Search

**arXiv ID:** 2604.14448 | [PDF](https://arxiv.org/pdf/2604.14448v1)

**作者:** Thales Sales Almeida `[一作]` (Maritaca AI), Thiago Laitz `[通讯]` (Maritaca AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了MARCA双语基准，用于评估大型语言模型在基于网络的信息检索和多实体答案生成中的表现。

**💡 创新点**

创新点在于结合手工设计的检查表式评估、双语（英葡）问题对、以及两种交互框架（直接工具调用与任务分解委派）来细粒度测量答案完整性与正确性。

**🔧 技术方法**

采用工具调用（Serper搜索与抓取）与子代理分层框架，并利用GPT‑4.1等LLM作为评判者进行自动化打分。

**📊 数据集**

使用包含52个手工编写、多实体问题及其对应检查表的MARCA数据集，覆盖9个主题领域，并在英葡两种语言版本上进行测试。

**📈 对比分析**

在14个模型（包括GPT‑4.1、Gemini‑3‑pro等）上进行多次运行，计算平均检查表覆盖率并给出标准差；结果显示Orchestrator框架往往提升准确率，模型间语言迁移表现差异显著，最高模型约0.90的覆盖率。

**⚠️ 局限性**

局限性包括数据集规模相对较小、仅覆盖英葡两种语言、依赖外部搜索API（Serper）且对动态网页内容的抓取与更新有时不稳定，且未覆盖更广泛的多语言或更复杂的对话场景。

---

## 8. The PICCO Framework for Large Language Model Prompting: A Taxonomy and Reference Architecture for Prompt Structure

**arXiv ID:** 2604.14197 | [PDF](https://arxiv.org/pdf/2604.14197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 9. Controllable Video Object Insertion via Multiview Priors

**arXiv ID:** 2604.14556 | [PDF](https://arxiv.org/pdf/2604.14556v1)

**作者:** Xia Qi `[一作]`, Yuexin Ma `[通讯]` (ShanghaiTech University)

**通讯引用:** 4229 | [OpenAlex ID](https://openalex.org/A5102015139)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于多视角先验的可控视频对象插入框架，能够在动态背景中稳定、真实地插入新对象；

**💡 创新点**

通过双路径视图一致性条件（身份保持潜在注入 + 多视角特征库）以及集成感知一致性模块（深度、轮廓和时间一致性），同时解决身份漂移、遮挡错误和时间抖动等难题；

**🔧 技术方法**

使用3D重建提升多视角参考、双路径条件注入、质量感知加权、深度与轮廓Head、光流驱动的时间一致性损失；

**📊 数据集**

训练使用约41k条视频（YouTube‑VIS、DAVIS 2017、MOSE），评估在DAVIS、VIPSeg、MagicBench等公开数据集上；

**📈 对比分析**

与VACE、AnyV2V、UniVideo等基线对比，PSNR/SSIM/LPIPS/FVD均显著提升，Mask_IoU/Box_IoU更高，时间一致性指标也明显优于对手；

**⚠️ 局限性**

仍受限于3D重建质量，对极端视角或复杂遮挡场景的适应性有限，且多视角生成与真实视频的差异仍需进一步优化。

---

## 10. Quantization of Spiking Neural Networks Beyond Accuracy

**arXiv ID:** 2604.14487 | [PDF](https://arxiv.org/pdf/2604.14487v1)

**作者:** Evan Gibson Smith `[一作]` (Worcester Polytechnic Institute), Fatemeh Ganji `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 1383 | [OpenAlex ID](https://openalex.org/A5017195195)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了脉冲神经网络（SNN）的量化方法，并提出用地球移动距离（EMD）评估量化后网络的发射行为，探讨量化位宽、截断范围和学习型量化对发射分布与准确率的影响。

**💡 创新点**

创新点包括：①首次将EMD作为评估量化SNN发射行为的诊断指标；②将LQ-Net学习型量化方法迁移到SNN，并证明其能在保持准确率的同时显著保留发射分布；③系统对比统一量化与学习型量化、窄/宽截断范围以及膜电位量化对发射动态的影响。

**🔧 技术方法**

采用的技术包括：统一量化（使用STE）、LQ-Net学习型量化、膜电位量化（均匀量化与学习型尝试）、SEW-ResNet8/18网络架构、QAT训练、EMD计算、统计指标（平均发射率、死亡神经比例）。

**📊 数据集**

实验数据集：CIFAR-10 与 CIFAR-100。

**📈 对比分析**

通过比较准确率、平均发射率、死亡神经比例和EMD四个指标，发现：在相同准确率下，宽截断的统一量化会导致EMD大幅升高，说明发射分布偏离；LQ-Net在所有截断范围和位宽下保持低EMD，准确率也优于统一量化；膜电位量化需要 ≥4 位才能稳定，且无符号量化在低位宽下表现更好。

**⚠️ 局限性**

局限性：①LQ-Net在膜电位量化时训练不稳定，难以直接应用；②研究仅涵盖SEW-ResNet架构和CIFAR-10/100，未验证在更大模型或其他任务上的通用性；③EMD虽能捕捉分布差异，但未能直接关联到硬件延迟或能耗指标。

---

## 11. Geometric Routing Enables Causal Expert Control in Mixture of Experts

**arXiv ID:** 2604.14434 | [PDF](https://arxiv.org/pdf/2604.14434v1)

**作者:** Ivan Ternovtsii `[一作]` (Uzhhorod National University), Yurii Bilak `[通讯]` (Uzhhorod National University)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5042083572)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究稀疏专家网络（MoE）中单向量专家的语义专化，通过投影、路由梯度和因果干预等手段，证明专家具有可解释、可控且因果有效的单义性。

**💡 创新点**

创新点在于将专家设计为 rank‑1 结构，利用余弦路由提供几何透明度，构建“语义词典”以直接读取专家功能，并通过推理时干预（驱动、抑制、编辑）验证专家对输出的因果影响。

**🔧 技术方法**

技术包括：稀疏专家网络、余弦路由、多跳语义重路由、logit lens、向量投影至未嵌入层构造语义词典、HDBSCAN 聚类、因果干预（steering、suppression、surgery）以及对比实验。

**📊 数据集**

使用 WikiText‑103（约1.64B token）进行训练与评估，并在 76–84 M 参数规模的 MoE 模型上进行实验。

**📈 对比分析**

在不同路由拓扑（单跳/多跳、共享/分离投影、余弦/线性）下模型在 PPL 上相当（≈33–34）；通过因果干预展示在 10 个语义类别上可实现 3%–453% 的概率增减，且对不同路由方式的可解释性与可控性进行了对比。

**⚠️ 局限性**

局限性包括：仅在 76–84 M 参数、rank‑1 专家、WikiText‑103 数据集上验证；缺乏大规模、跨语料、不同专家秩或生产环境的评估；部分专家聚类不完整，且多类别控制受跨层约束；安全相关类别的可控性相对较弱。

---

## 12. NewsTorch: A PyTorch-based Toolkit for Learner-oriented News Recommendation

**arXiv ID:** 2604.14510 | [PDF](https://arxiv.org/pdf/2604.14510v1)

**作者:** Rongyao Wang `[一作]` (University of Otago), Zhiyi Huang `[通讯]` (University of Otago)

**通讯引用:** 1965 | [OpenAlex ID](https://openalex.org/A5090297023)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为 NewsTorch 的 PyTorch 基础、模块化、GUI 友好的新闻推荐工具包，提供数据集下载与预处理、模型管理、实验控制及实验跟踪等完整功能；

**💡 创新点**

其创新点在于：① 首次将 GNN 与 LLM 两类前沿模型统一整合到同一解耦框架；② 采用独立 YAML 配置机制，显著降低学习者上手门槛；③ 提供可视化 GUI，直接完成数据处理与模型训练；

**🔧 技术方法**

技术手段包括 PyTorch + PyTorch‑Lightning（可选）、Weights & Biases 记录实验、Hydra 风格配置、模块化代码结构、多 GPU 并行训练、Web GUI 交互；

**📊 数据集**

主要使用 MIND 与 EB‑NeRD 两大公开新闻推荐数据集，并支持通过统一格式兼容更多常见数据集；

**📈 对比分析**

通过标准化指标（如 NDCG@10、MAP@10 等）与公开基线模型（DL、GNN、LLM）在同一实验环境下进行公平对比，实验表明 NewsTorch 在多数据集多模型上能复现并在多数指标上优于现有基线；

**⚠️ 局限性**

局限性包括：仅实现了部分主流模型和数据集，未覆盖所有最新 LLM；缺乏实时推理与多语言支持；未来计划持续扩展模型、数据与功能。

---

## 13. Tug-of-War within A Decade: Conflict Resolution in Vulnerability Analysis via Teacher-Guided Retrieval-Augmented Generations

**arXiv ID:** 2604.14172 | [PDF](https://arxiv.org/pdf/2604.14172v1)

**作者:** Ziyin Zhou `[一作]` (Beijing Electronic Science and Technology Institute), Zhangchi Zhao `[通讯]` (Beijing Electronic Science and Technology Institute)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5101232158)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种名为CRVA-TGRAG的两阶段框架，用于解决LLM在处理CVE漏洞信息时出现的知识冲突。

**💡 创新点**

创新点包括构建1,260条冲突CVE对的专用数据集，采用父文档分割+语义相似度+BM25混合检索提升检索准确度，并在生成阶段使用教师引导的偏好优化（DPO）让模型更偏好最新漏洞信息。

**🔧 技术方法**

技术上结合了检索增强生成（RAG）、BM25和向量检索、语义相似度分块、父文档分段、集成检索、Prompt Engineering、DPO偏好微调以及多维评估指标（Faith, CR, CP, AR, AC, AS, BLEU, ROUGE‑L）。

**📊 数据集**

使用官方NVD、CVE GitHub仓库的JSON数据，整理成CSV后生成1,260条冲突CVE对的数据集，并挑选2014-2024年高危（baseSeverity=HIGH）共1,060条CVE作为实验数据。

**📈 对比分析**

与无检索、仅提示、仅DPO、Naive RAG等基线方法以及多款LLM（GPT‑4o‑mini、Claude‑3.5、Gemini‑2.0、Llama‑3‑70B、Mistral）对比实验，CRVA‑TGRAG在答案正确率、可信度、检索上下文精确度等指标上均显著提升，答案正确率从0.49提升至0.76，可信度从0.36提升至0.88，整体性能优于传统方法。

**⚠️ 局限性**

局限性在于检索仍基于离线预处理，缺乏实时在线检索；在线Agentic RAG易受搜索引擎SEO影响、信息时效性不足，导致检索结果仍可能产生冲突，需进一步研究在线检索与实时更新机制。

---

## 14. Auxiliary Finite-Difference Residual-Gradient Regularization for PINNs

**arXiv ID:** 2604.14472 | [PDF](https://arxiv.org/pdf/2604.14472v1)

**作者:** Stavros Kassinos `[一作]` (University of Cyprus), Stavros Kassinos `[通讯]` (University of Cyprus)

**通讯引用:** 3288 | [OpenAlex ID](https://openalex.org/A5048430194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了一种混合 PINN 设计，在保持 PDE 残差为 AD 形式的同时，加入仅在辅助网格上对残差场进行有限差分梯度惩罚的正则化。

**💡 创新点**

创新点在于将有限差分残差梯度正则化仅应用于采样残差场，既保持连续 PINN 残差，又在局部（如三维环形结构的外壁）引入结构化正则化，从而显著提升目标物理量（外壁热通量）的预测准确性。

**🔧 技术方法**

采用自动微分（AD）计算连续 PDE 残差，结合有限差分（FD）对残差场梯度进行正则化；实验使用多层感知器（MLP）网络、余弦学习率调度、Adam/优化器。

**📊 数据集**

使用了两组数据集：一是二维 Poisson 解析问题的制造样本，用于机制研究；二是三维环形热传导的 PINN3D 基准，包含波纹外壁的几何与边界条件。

**📈 对比分析**

通过与无正则化基线、FD 正则化、匹配的 AD 正则化等三种方案进行对比，Stage 1 展示了残差清洁度与场精度的 Pareto 关系；Stage 2 在六个随机种子、10⁵ 轮训练下，固定 shell 权重 5×10⁻⁴ 的正则化将外壁 BC 和通量 RMSE 分别降低约 13 倍和 10 倍，表现最优。

**⚠️ 局限性**

局限性在于正则化效果高度依赖优化器与学习率设置，仅在特定的  优化器/学习率组合下稳定；对其他问题的可推广性和自动调参仍需进一步研究。

---

## 15. AIBuildAI: An AI Agent for Automatically Building AI Models

**arXiv ID:** 2604.14455 | [PDF](https://arxiv.org/pdf/2604.14455v1)

**作者:** Ruiyi Zhang `[一作]` (University of California San Diego), Pengtao Xie `[通讯]` (University of California San Diego)

**通讯引用:** 5732 | [OpenAlex ID](https://openalex.org/A5083884675)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个名为AIBuildAI的 AI 代理系统，能够根据任务描述和训练数据自动设计、编码、训练并调优完整的 AI 模型，最终输出可部署的模型检查点和推理脚本。

**💡 创新点**

创新点在于引入分层多智能体架构：由主控 manager 负责整体流程，分为设计者 (designer)、编码者 (coder) 与调优者 (tuner) 三个专门化 LLM 代理，允许每个子代理在自身上下文内进行多步推理与工具调用，从而突破单一 LLM 单步修改的局限，实现全生命周期自动化。

**🔧 技术方法**

主要技术包括：LLM 代理模型（Claude Opus 4.6），工具使用与多轮交互，分层多智能体协作，基于仓库的并行候选方案管理，局部搜索式迭代改进，以及聚合器 (aggregator) 的模型融合与最终提交生成。

**📊 数据集**

使用了 MLE‑Bench benchmark，涵盖 75 个真实 Kaggle‑style 任务（视觉、文本、时序、表格等多模态）。

**📈 对比分析**

与 26 种近期基线方法（如 AIRA‑dojo、MLEvolve 等）进行对比，采用单 GPU 24 小时预算的 MLE‑Bench 评测，AIBuildAI 在整体 medal rate 上达到 63.1%，排名第一，并在中、高难度子集上分别获得 61.4% 与 46.7% 的 medal rate，显著优于所有对比方法。

**⚠️ 局限性**

主要局限包括：高昂的 token 消耗（多代理多轮调用导致成本上升）、仅在单 GPU 24 小时预算下验证，未能适应大规模分布式训练与更大模型的资源分配；同时所有子代理均使用同一高成本 LLM，缺乏动态模型路由与成本优化方案。

---

## 16. Anomaly Detection in IEC-61850 GOOSE Networks: Evaluating Unsupervised and Temporal Learning for Real-Time Intrusion Detection

**arXiv ID:** 2604.14233 | [PDF](https://arxiv.org/pdf/2604.14233v1)

**作者:** Joseph Moore `[一作]` `[通讯]` (Boise State University), Joseph Moore (Boise State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文评估了无监督时间序列模型（GRU、LSTM、RNN）与随机森林、前馈自编码器在 IEC‑61850 GOOSE 网络中的实时入侵检测性能，并对模型延迟、准确率和跨环境泛化进行了系统比较。

**💡 创新点**

创新点在于：①首次将循环自编码器用于 GOOSE 事件的异常检测，并证明其在满足 4 ms 延迟约束的同时实现高 F1 分数；②通过交叉数据集评估，展示无监督时间序列模型在未见环境中的相对稳健性；③阈值选取采用验证集 Youden 指数，避免了测试集泄漏。

**🔧 技术方法**

使用的技术包括：随机森林（监督基线）、前馈自编码器、RNN、LSTM、GRU 自编码器、标准化与 One‑Hot 编码、阈值优化（Youden 指数）和延迟测量（GPU 环境）。

**📊 数据集**

主要数据集为公开的 ERENO IEC‑61850 数据集（69 维特征，包含 7 种攻击类型），并在 de Oliveira 等人的独立 HIL 数据集上进行跨环境测试。

**📈 对比分析**

比较方法：在同一特征集上测量 Accuracy、Precision、Recall、F1、AUC‑ROC 以及单样本推理延迟。结果显示随机森林 F1=0.9516 但延迟 21.8 ms；GRU F1=0.8737，延迟 1.118 ms；LSTM F1=0.8686，延迟 1.92 ms；RNN F1=0.8637，延迟 1.23 ms；Autoencoder F1=0.5826，延迟 0.039 ms。跨环境评估显示随机森林 F1 降至 0.1414，而循环模型仅降至 0.197–0.257。

**⚠️ 局限性**

limitations: 数据集缺乏多样性，尤其是缺少完整特征的跨环境匹配；阈值依赖于训练时的类别不平衡；延迟测量依赖 GPU，嵌入式硬件可能产生更高延迟；模型只做二分类，未细化攻击类别；未对解释性或可视化报警进行深入研究。

---

## 17. Fun-TSG: A Function-Driven Multivariate Time Series Generator with Variable-Level Anomaly Labeling

**arXiv ID:** 2604.14221 | [PDF](https://arxiv.org/pdf/2604.14221v1)

**作者:** Pierre Lotte `[一作]` (Université de Toulouse, IRIT, CNRS), Olivier Teste `[通讯]` (Université de Toulouse II - Jean Jaurès, IRIT, CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Fun-TSG，一个可定制的多变量时间序列生成器，用于自动或手动生成带有明确依赖图的时间序列，并在变量及时间戳级别精准标注异常，支持异常传播与否的细粒度注释，旨在为异常检测方法提供透明可复现的评估基准。

**💡 创新点**

创新点在于（1）使用有向图和符号函数生成多变量序列，明确显示变量间的时间延迟与因果关系；（2）实现完全可配置的异常注入策略，区分传播与非传播异常并提供丰富标签；（3）提供完全透明的生成过程，兼顾自动化与手动配置。

**🔧 技术方法**

技术上结合了随机有向图生成、符号表达式树构造、基于增长行为调节的算子采样、三种异常注入策略（插入、删除、替换）以及传播行为随机分配，最终通过数值计算生成序列。

**📊 数据集**

主要使用自研的 Fun-TSG 生成的 synthetic 数据集；文中也提及 TSB-AD 作为对比，但未在本工作中实际使用。

**📈 对比分析**

论文未给出实验结果或与其他方法的性能对比，主要阐述工具的功能与设计，未开展实测评估。

**⚠️ 局限性**

局限性包括：未在真实世界数据上验证生成器的真实性；缺乏对生成数据多样性与逼真度的量化评估；异常模式生成仍基于人工设计，可能缺乏自然性；以及尚未展示其对现有异常检测算法的实际性能提升。

---

## 18. MemGround: Long-Term Memory Evaluation Kit for Large Language Models in Gamified Scenarios

**arXiv ID:** 2604.14158 | [PDF](https://arxiv.org/pdf/2604.14158v1)

**作者:** Yihang Ding `[一作]` (Tsinghua University), Wenming Yang `[通讯]` (Tsinghua University)

**通讯引用:** 4676 | [OpenAlex ID](https://openalex.org/A5026184280)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了MemGround，一个基于游戏化交互场景的长期记忆评估基准，包含三层分级框架（Surface State Memory、Temporal Associative Memory、Reasoning-Based Memory）和多维度评估指标。

**💡 创新点**

创新点在于：①把长期记忆评估从静态检索转向动态、分层、关联的交互场景；②设计三层分级评估框架；③引入多维度指标（QA Overall、MFU、MFCO、ETD），能细粒度衡量记忆获取、关联和推理过程。

**🔧 技术方法**

使用技术包括：人类游戏数据收集与JSON化、统一交互评估框架、LLM与外部记忆代理（Mem0、A-MEM）的集成、存储/压缩/检索/索引工具、交互式多轮评估、图结构探索轨迹分析。

**📊 数据集**

数据集来自三款文本游戏（TRPG、No Case Should Remain Unsolved、Type Help）的真实人类游玩记录，已在GitHub公开。

**📈 对比分析**

方法：在三层任务中对比闭源LLM（GPT‑5.2、Gemini‑3、DeepSeek‑V3.2、Claude‑Opus‑4.6）与开源LLM（Qwen3‑32B）以及配套记忆代理；结果显示闭源模型总体优于开源，但随着层级提升性能显著下降，Temporal Associative Memory的MFCO几乎为零，Reasoning‑Based Memory的QA Overall和MFU也远低于预期。

**⚠️ 局限性**

局限性：仅覆盖三款文本游戏，缺乏多模态、视觉、空间或具身交互场景；评估聚焦文本交互，无法完全反映真实世界中更丰富的长期记忆需求。

---

## 19. Neuro-Oracle: A Trajectory-Aware Agentic RAG Framework for Interpretable Epilepsy Surgical Prognosis

**arXiv ID:** 2604.14216 | [PDF](https://arxiv.org/pdf/2604.14216v1)

**作者:** Aizierjiang Aiersilan `[一作]` (George Washington University), Mohamad Koubeissi `[通讯]` (George Washington University)

**通讯引用:** 3169 | [OpenAlex ID](https://openalex.org/A5003715792)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 Neuro-Oracle 框架，利用三阶段流程将术前术后 MRI 的形态变化编码为 512 维轨迹向量，检索历史相似轨迹，并用量化 LLaMA-3 生成可解释的手术预后报告。

**💡 创新点**

创新点在于：① 将双时点 MRI 差分映射为低维轨迹向量并直接用于检索；② 采用检索增强生成（RAG）与 LLM 进行推理，提升可解释性并抑制幻觉；③ 将时间序列信息与基于 k‑NN 的几何相似度相结合，解决小样本、类别不平衡的挑战。

**🔧 技术方法**

技术包括 3D Siamese 对比学习（Supervised Contrastive + Focal Loss）、FAISS 内部向量检索、4‑bit NF4 量化的 LLaMA‑3‑8B 生成模型，以及多模型集成（Diversity Ensemble）。

**📊 数据集**

使用公开的 EPISURG 数据集，268 对配对 T1‑MRI（前后手术），并采用手术类型作为临床代理标签。

**📈 对比分析**

在 5 折分层交叉验证中与单时点 ResNet‑50、3D ViT、Siamese+Logistic、Siamese+MLP、Siamese+k‑NN 及多模型集成对比；Neuro‑Oracle 的 AUC 0.867，等同于 3D‑ViT 及 k‑NN，且可生成自然语言解释；多模型集成 AUC 0.905 最高。

**⚠️ 局限性**

局限包括：使用手术类型作为代理标签导致标签噪声；样本量仅 268 例，单中心数据，缺乏跨站/跨扫描器验证；LLM 仍可能依赖检索相似度而非真正的癫痫病理；模型对真实癫痫预后标签的可推广性待验证。

---

## 20. EuropeMedQA Study Protocol: A Multilingual, Multimodal Medical Examination Dataset for Language Model Evaluation

**arXiv ID:** 2604.14306 | [PDF](https://arxiv.org/pdf/2604.14306v1)

**作者:** Francesco Andrea Causio `[一作]`, Manuel Del Medico `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了欧洲医学考试数据集EuropeMedQA，包含多语言、多模态（文本+图像）真实官方考试题目，构建了完整的题库、翻译版本及数据清洗流程；

**💡 创新点**

创新点在于首个欧洲多语言多模态医学考试基准，遵循FAIR与SPIRIT‑AI标准，采用自动化翻译与答案位置打乱消除偏差，并提供零样本评估的严谨流程；

**🔧 技术方法**

使用GPT‑4o‑mini进行自动翻译、GPT‑5‑mini、Claude‑3.5‑Haiku‑20241022与Claude Sonnet 4.5进行零样本推理，采用严格约束提示、答案位置随机化、统一评估脚本；

**📊 数据集**

数据集来源为意大利、法国、西班牙、葡萄牙官方医学执照与住院医师入学考试题目，包含文本题、图像题及多答案题；

**📈 对比分析**

通过对比文本版与翻译版、单模态与多模态模型，采用准确率（Accuracy）为主要指标，发现多模态模型在图像题上显著优于文本模型，跨语言迁移表现不均，GPT‑5‑mini在多语言文本题上略胜一筹；

**⚠️ 局限性**

局限性包括：基于考试题目导致的知识与推理偏倚、答案位置随机化带来的人工均衡不反映真实临床分布、自动翻译可能引入细微语义误差、云端API限制了硬件与解码细节的可控性；

---

## 21. Demonstration of Pneuma-Seeker: Agentic System for Reifying and Fulfilling Information Needs on Tabular Data

**arXiv ID:** 2604.14422 | [PDF](https://arxiv.org/pdf/2604.14422v1)

**作者:** Muhammad Imam Luthfi Balaka `[一作]` (University of Chicago), Raul Castro Fernandez `[通讯]` (University of Chicago)

**通讯引用:** 3703 | [OpenAlex ID](https://openalex.org/A5003690515)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一款名为Pneuma‑Seeker的交互式系统，能够将用户的模糊信息需求转化为可检查、可迭代的关系规范，支持数据检索、转换与执行，并提供可追溯的执行路径。

**💡 创新点**

创新点在于：① 将用户意图外化为可视化的关系视图而非仅在生成代码中隐式处理；② 通过宏微层次的上下文管理和交互式检索，克服大语言模型上下文容量限制；③ 结合传统关系操作与语义操作（如语义连接、列生成）提升非结构化匹配能力；④ 提供实时的可视化面板供用户检查和逐步细化需求。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）驱动的规划与执行；检索层使用基于模式匹配与内容匹配的组合检索；微层上下文管理通过可执行查询请求获取表的证据；关系与语义操作实现关系变换；Python/SQL执行回退；所有步骤记录成可追溯的DAG。

**📊 数据集**

使用了两套真实采购数据集：JAGGAER平台的41张表（15 GB）和Oracle提取的FY2025采购订单表（7 MB），以及测试外包案例中的内部测试表与供应商报价表；演示时替换为芝加哥开放数据集。

**📈 对比分析**

通过两个案例（Outstanding Amount 与 Test Outsourcing）展示系统的交互效率（约 2 min 24 s 与 1 min 41 s，后续迭代提升至 3 min 27 s）。论文未给出与基线系统的定量对比，但强调了可视化检查、迭代精细化以及对语义匹配错误的容错能力。

**⚠️ 局限性**

局限性：① 语义连接可能产生误匹配，需要人工验证；② 仍受LLM上下文长度和检索召回范围限制；③ 主要针对表格数据，跨域或非结构化文本处理尚未覆盖；④ 缺乏大规模实验与性能基准，评估主要基于案例演示。

---

## 22. Can Large Language Models Detect Methodological Flaws? Evidence from Gesture Recognition for UAV-Based Rescue Operation Based on Deep Learning

**arXiv ID:** 2604.14161 | [PDF](https://arxiv.org/pdf/2604.14161v1)

**作者:** Domonkos Varga `[一作]` `[通讯]`, Domonkos Varga

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了一篇已发表的基于深度学习的无人机手势识别论文的实验设计，检测其中是否存在数据泄漏，并通过让多种大型语言模型（LLM）在仅给定原论文PDF的情况下独立分析其方法论，以验证LLM在识别科学论文方法缺陷方面的可行性。

**💡 创新点**

创新点在于首次将LLM用作独立的科学审计工具，展示它们能够从已发表的实验结果（学习曲线、混淆矩阵、准确率等）自动识别出数据泄漏和其他评估缺陷，从而为科研质量控制提供一种自动化、可复制的辅助手段。

**🔧 技术方法**

主要技术手段包括：1）对原论文的评估协议、学习曲线、混淆矩阵进行结构化提取；2）使用六种不同架构的LLM（GPT‑5.2、Claude Sonnet 4.6、Google Gemini 3.0 Pro、Kimi 2.5、DeepSeek‑V3、GLM‑5）在同一提示下独立生成评估报告；3）对比各模型输出的一致性与原论文所报告的指标。

**📊 数据集**

所用数据集为原论文中的手势识别数据集：六名实验者在实验室环境下录制的十个手势动作，总计约9,869帧，使用OpenPose提取的18点骨架关键点作为特征。

**📈 对比分析**

方法比较：原论文报告训练准确率99.47%、测试准确率99.09%；LLM评估结果一致指出由于随机帧级划分导致的受试者泄漏，学习曲线与混淆矩阵几乎完全同步，表明测试集与训练集极度相似，实际泛化性能远低于原报告。

**⚠️ 局限性**

局限性包括：1）仅对单篇论文进行案例研究，未能验证LLM对其他类型错误的识别能力；2）LLM的判断依赖于训练语料，可能忽略某些细微或新颖的实验缺陷；3）缺乏独立人类专家验证，无法完全排除LLM误判；4）原论文数据集规模有限，导致评估结论在更大、更复杂场景下的可推广性未知。

---

## 23. Seeing Through Experts Eyes A Foundational Vision Language Model Trained on Radiologists Gaze and Reasoning

**arXiv ID:** 2604.14316 | [PDF](https://arxiv.org/pdf/2604.14316v1)

**作者:** Kinhei Lee `[一作]` (Imperial College London), Guang Yang `[通讯]` (Imperial College London)

**通讯引用:** 20594 | [OpenAlex ID](https://openalex.org/A5100436460)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种将放射科医生眼动轨迹融入训练的基础视觉语言模型Gaze‑X，旨在提升胸部X光报告生成、视觉问答和视觉定位任务的可解释性与准确性。

**💡 创新点**

创新点在于以眼动追踪数据作为行为先验，强制模型学习医生的结构化检查流程（如ABCDEF），并生成可验证的检视轨迹与定位证据，使模型行为与临床工作流对齐。

**🔧 技术方法**

采用Qwen2‑Vision‑Language的2B参数模型，结合Fine‑grained Visual Perception、Gaze Trajectory Mimicking和Sequential Dependency Awareness三大模块，并利用DBSCAN聚类、Gaussian热图、动态时间规整等技术处理眼动信息。

**📊 数据集**

预训练使用REFLACX（3,032例含眼动与报告）数据集，下游评估使用MIMIC‑CXR、IU‑X‑Ray、Medical‑CXR‑VQA和MS‑CXR等公开胸部影像数据集。

**📈 对比分析**

与R2Gen、STREAM、LLaVA‑Med、RadFM等现有模型在报告生成（BLEU‑1最高0.502，CheXbert F1≈0.46）、视觉问答（top‑1≈0.65）和视觉定位（mIOU提升约42%）上进行基准测试，Gaze‑X在所有任务中均实现显著或可观提升。

**⚠️ 局限性**

局限性包括未在真实临床工作流中评估检测效率与错误率；缺少罕见疾病样本；未融入外部医学知识；仅针对胸部X光，未扩展至CT/MRI/超声等影像。

---

## 24. Decoupling Scores and Text: The Politeness Principle in Peer Review

**arXiv ID:** 2604.14162 | [PDF](https://arxiv.org/pdf/2604.14162v1)

**作者:** Yingxuan Wen `[一作]` `[通讯]` (Harbin Intitute Of Technology), Yingxuan Wen (Harbin Intitute Of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过构建30k+份ICLR 2021-2025提交的评审数据集，比较基于分数和文本的论文接收预测，进一步分析文本预测性能低下的根本原因。

**💡 创新点**

发现硬样本的分数分布呈负偏度且峰度较高，表明单个低分是决定性拒绝信号；揭示礼貌原则导致文本中的积极情感掩盖了真正的拒绝信号，从而产生分数与文本解耦现象。

**🔧 技术方法**

使用传统机器学习（RF、XGBoost、SVM）、深度学习（Word2Vec、TextCNN、SciBERT）与大型语言模型（Qwen‑Turbo、Gemini‑2.5、GPT‑5、Claude‑Haiku），以及基于依存句法的细粒度情感分析。

**📊 数据集**

ICLR 2021-2025官方OpenReview平台数据，经过清洗、对话重建并标注为二分类（接受/拒绝），形成多轮对话数据集。

**📈 对比分析**

采用时间拆分（2022-2024为训练集，2025为测试集）并以准确率为评估指标，分数模型最高91%准确，文本模型仅81%，混合模型最高约87%，验证了分数预测远优于文本预测。

**⚠️ 局限性**

主要限制在于礼貌掩码难以从文本中提取负面拒绝信号，且硬样本仅占约9%，导致文本模型难以显著提升性能，且情感分析方法对隐含语用信号仍捕捉不足。

---

## 25. Bivariate range functions with superior convergence order

**arXiv ID:** 2604.14400 | [PDF](https://arxiv.org/pdf/2604.14400v1)

**作者:** Bingwei Zhang `[一作]` (New York University), Chee Yap `[通讯]` (New York University)

**通讯引用:** 5780 | [OpenAlex ID](https://openalex.org/A5078989577)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文设计并实现了适用于二维函数的超越二次收敛阶的区间范围函数，提出了基于Taylor、Lagrange和Hermite插值的三阶（立方）和四阶（四次）收敛形式，并通过一般化的Cornelius–Lohner框架证明其收敛性。

**💡 创新点**

创新点在于：①将Cornelius–Lohner框架推广到二维情形；②引入递归Lagrange和Hermite形式，获得三阶和四阶收敛的区间范围；③结合高阶Taylor形式实现任意阶的可定理化收敛；④用符号求导生成SLP（Straight‑Line Program）以复用导数评估，提高效率。

**🔧 技术方法**

技术手段包括：1) 经典多变量Taylor展开与中心化形式；2) 递归Lagrange插值与Hermite插值，构造多项式的高阶误差上界；3) 采用一般化Cornelius–Lohner框架证明收敛阶；4) 在Julia中实现SLP并进行符号微分；5) 通过实验测量执行时间、范围宽度和内存占用。

**📊 数据集**

使用了四个多项式测试函数（clover-4、clover-5、clover-8和grass）在[-1.2,1.2]^2的网格上进行实验，每个函数在32×32网格共1024个盒子上评估。

**📈 对比分析**

比较方法：把基准的二阶Taylor形式（_2）与三阶Taylor（_3）、四阶Taylor（_4）、递归Lagrange（_3）和递归Hermite（_4）进行对比，测量总耗时、速度提升比、范围宽度（效能）以及内存使用。实验结果显示：三阶方法在保持与基准相近的计算时间（共享数据时可略快）的同时，范围宽度更窄（效能提升约19%–30%）；四阶方法范围更窄但计算更慢；递归Lagrange在共享数据时效率最高。

**⚠️ 局限性**

局限性：①目前仅支持多项式函数，未对一般解析函数或非多项式函数给出实现；②所有运算使用IEEE 754双精度，未考虑误差传播与ε修正；③SLP实现仅适用于多项式，缺乏对更复杂表达式的支持；④实验仅评估单盒子评估性能，未在完整应用（如曲线追踪、根分离）中验证整体收益；⑤高阶形式实现复杂，调试成本较高。

---

## 26. Parallel R-tree-based Spatial Query Processing on a Commercial Processing-in-Memory System

**arXiv ID:** 2604.14445 | [PDF](https://arxiv.org/pdf/2604.14445v1)

**作者:** Tasmia Jannat `[一作]` (Missouri University of Science and Technology), Satish Puri `[通讯]` (Missouri University of Science and Technology)

**通讯引用:** 826 | [OpenAlex ID](https://openalex.org/A5019908059)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

实现了在商业 Processing‑in‑Memory（PIM）系统上执行 R‑tree 范围查询，并提出了广播上层节点、分布叶子节点的异构 CPU–DPU 方案。

**💡 创新点**

首次在真实 PIM 硬件上实现 R‑tree，利用广播式执行显著降低子树传输开销，并通过层级过滤和批量查询实现可扩展的高性能与低能耗。

**🔧 技术方法**

采用 UPMEM DPUs、宽度优先序列化、广播上层节点、批量查询、DPU 任务块并行、STR bulk‑loading、CPU–DPU BSP 模型。

**📊 数据集**

使用真实数据集 Sports（99.9 万矩形）和 Lakes（840 万矩形），以及合成数据集 16M 矩形 + 3.99M 查询，评估不同规模和查询比例下的表现。

**📈 对比分析**

与多线程 CPU 基线和子树划分 PIM 基线进行比较；在 Lakes 数据集上，512→2540 DPUs 时，kernel 时间从 64.9 s 降至 17.6 s，获得 3.66× kernel 加速和 2.70×端到端加速；能耗比 CPU 低约 3.4×。

**⚠️ 局限性**

对小型、缓存友好的数据集收益有限，通信开销仍难以完全摊销；当前实现缺乏多 DPU 集合重叠与性能模型，需要进一步优化。

---

## 27. Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation

**arXiv ID:** 2604.14339 | [PDF](https://arxiv.org/pdf/2604.14339v1)

**作者:** Zichong Li `[一作]` (Georgia Institute of Technology), Weizhu Chen `[通讯]` (Microsoft)

**通讯引用:** 10692 | [OpenAlex ID](https://openalex.org/A5051745436)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 RoPE‑Perturbed Self‑Distillation 正则化，利用可控 RoPE 位置扰动和自蒸馏显著提升大模型在长上下文任务中的位置鲁棒性。

**💡 创新点**

创新点在于通过对 RoPE 编码进行结构保留的随机扰动并强制预测一致性，直接对位置信息敏感性进行正则化，减少证据位置对性能的影响。

**🔧 技术方法**

采用 RoPE 位置扰动、逆 KL 自蒸馏、长上下文微调以及可选的 cyclic‑shift 方案；训练时在每个批次做一次额外的前向传播。

**📊 数据集**

使用 ProLong（长序列预训练）、RULER、HELMET、LongBench 等长上下文数据集，并在 Llama‑3‑8B‑Instruct 与 Qwen‑3‑4B 上进行实验。

**📈 对比分析**

与标准长上下文微调、LongCE、PoSE 等基线对比，Llama‑3‑8B 在 RULER‑64K 上平均提升约 12%、Qwen‑3‑4B 在 RULER‑256K 上提升约 3%，且在超长长度（YaRN 扩展）时表现更稳健。

**⚠️ 局限性**

需要额外一次前向传播导致约 1.6× 训练开销，对扰动范围和 KL 权重 λ 的设置较为敏感，并且方法目前仅在使用 RoPE 的模型中有效。

---

## 28. Bias in Surface Electromyography Features across a Demographically Diverse Cohort

**arXiv ID:** 2604.14460 | [PDF](https://arxiv.org/pdf/2604.14460v1)

**作者:** Aditi Agrawal `[一作]` (University of California, Davis), Richard S. Whittle `[通讯]` (University of California, Davis)

**通讯引用:** 394 | [OpenAlex ID](https://openalex.org/A5063770847)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对81位不同年龄、性别、BMI等多样化人群的上肢表面肌电（sEMG）进行特征敏感性分析，评估147种常用特征与人口统计变量的关联；

**💡 创新点**

揭示仅少量波形包络（WPT）与频域特征高度受性别与皮下脂肪厚度影响，而时域特征基本不变，为实现公平sEMG模型提供特征选择与归一化的理论依据；

**🔧 技术方法**

采用多重插补（MICE）补全缺失值，构建混合效应线性模型并做FDR校正与部分η²效应量估计，进一步利用稀疏部分最小二乘（sPLS）聚类图挖掘多元相关结构；

**📊 数据集**

使用Gowda等人公开的UCD‑MyoVerse‑Hand‑1数据集（81名受试者，共81组人口统计指标）；

**📈 对比分析**

通过统计显著性与效应量的双重阈值，比较各特征的人口敏感程度；结果显示性别与脂肪厚度对频域与WPT特征的解释方差显著（marginal R²≈0.05），但总体条件R²高（≈0.65），提示需加入人口统计信息才能提升模型公平性；

**⚠️ 局限性**

研究为横断面设计，无法追踪随年龄或体成分变化的因果关系；特征冗余与多重检验仍可能导致假阳性；仅考虑已测量的人口与生理变量，未涵盖肌肉量、运动水平等潜在影响；

---

## 29. SpaceMind: A Modular and Self-Evolving Embodied Vision-Language Agent Framework for Autonomous On-orbit Servicing

**arXiv ID:** 2604.14399 | [PDF](https://arxiv.org/pdf/2604.14399v1)

**作者:** Aodi Wu `[一作]` (University of Chinese Academy of Sciences), Xue Wan `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 10968 | [OpenAlex ID](https://openalex.org/A5100375520)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了SpaceMind，一种模块化、自主进化的视觉语言模型驱动的宇宙机上维修代理框架。

**💡 创新点**

将代理知识、工具与推理分为可独立扩展的三维度，实现动态技能路由、可配置MCP工具以及可插拔推理模式，并引入技能自演化机制。

**🔧 技术方法**

基于大型视觉语言模型（Qwen3‑VL）、Model Context Protocol（MCP）+Redis消息总线实现跨环境接口；三种推理模式（Standard、ReAct、Prospective）；层次化记忆；经验反思与结构化技能生成的自演化。

**📊 数据集**

在UE5仿真（五颗卫星）和物理实验室（两颗卫星模型）上进行192次闭环实验，使用RGB图像+LiDAR传感数据。

**📈 对比分析**

对工具配置、推理模式和自演化进行系统对比；标准条件下所有模式达90–100%导航成功；退化条件下Prospective保持搜索成功；自演化使四组实验从完全失败到100%成功或检查分数提升至59/100；物理实验零改动转移实现100%对接成功。

**⚠️ 局限性**

VLM对3D空间理解有限，退化下性能下降；自演化仅在失败恢复上有效，对已成功任务无提升；任务范围绑定限制跨任务迁移；实验规模有限，未涵盖轨道动力学。

---

## 30. Sovereign 2.0: Control-Plane Sovereignty for Cloud Systems Under Disruption

**arXiv ID:** 2604.14242 | [PDF](https://arxiv.org/pdf/2604.14242v1)

**作者:** Justin Stark `[一作]` (University of Technology Sydney), Scott Wilkie `[通讯]` (University of Technology Sydney)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 Sovereign 2.0 的控制平面主导模型，定义了管理主权并构建了三层风险-保证框架，用以在分布式云环境中持续证明和保障治理、操作、证据和信任的可控性。

**💡 创新点**

创新点在于把主权从单纯的地域位置转移到可执行的控制平面维度，形成了以治理、身份、加密、数据生命周期、可观测性和事件响应为核心的管理主权概念，并将后量子 TLS 与 PKI 视为基础信任控制。

**🔧 技术方法**

采用了治理保证、操作保证与技术保证三层模型；利用 IAM、PAM、HSM、证书授权、日志/追踪系统、混合 TLS 1.3 等技术实现控制平面隔离、证据生成与加密迁移。

**📊 数据集**

未使用实验数据集；本文为理论框架与设计说明。

**📈 对比分析**

未进行实验比较或性能评估；论文主要通过文献综述、标准对照和案例（如 2026 年 AWS 中东事件）论证框架可行性。

**⚠️ 局限性**

局限包括：实现成本与运维复杂度提升；对全球 SaaS 生态的兼容性有限；缺乏实证验证；对不同行业的可扩展性与优先级仍需进一步研究。

---

## 31. Graph-Based Fraud Detection with Dual-Path Graph Filtering

**arXiv ID:** 2604.14235 | [PDF](https://arxiv.org/pdf/2604.14235v1)

**作者:** Wei He `[一作]` (Jinan University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 135935 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种双路径图过滤框架DPF‑GFD，用于解决金融欺诈检测中的关系伪装、异构性和类别不平衡问题。

**💡 创新点**

创新点在于将贝塔小波谱滤波器与基于k‑NN相似图的低通滤波器并行运算，显式分离结构异常建模与特征一致性重建，形成频率互补的双路径过滤；并通过特征拼接+MLP与XGBoost集成实现高效、稳健的欺诈预测。

**🔧 技术方法**

使用的技术包括：Beta小波谱滤波器、k‑NN相似图构建、低通谱滤波、特征融合+多层感知机、XGBoost树集成；全部在PyTorch/DGL实现。

**📊 数据集**

实验数据集为四个真实金融欺诈数据集：FDCompCN、FFSD、Elliptic、DGraph，涵盖财报欺诈、信用卡欺诈、比特币交易与贷款欺诈场景。

**📈 对比分析**

与九种基线模型（GCN、GraphSAGE、GAT、GAS、PC‑GNN、PMP、BernNet、AMNet、BWGNN）在Rec@K、F1、AUC、AP等指标下进行对比，DPF‑GFD在大多数指标上实现最高或接近最高的性能，并表现出更低的方差，表明模型更稳定、准确。

**⚠️ 局限性**

局限性包括：需要额外构建相似图导致计算和内存开销；模型仅适用于静态无向图；融合方式采用简单拼接，缺乏自适应机制；在极端类别不平衡或少标注、动态图情境下的表现仍有待进一步提升。

---

## 32. Incidence Constraints in Hypergraph Partitioning on GPU

**arXiv ID:** 2604.14411 | [PDF](https://arxiv.org/pdf/2604.14411v1)

**作者:** Marco Ronzani `[一作]` (Politecnico di Milano), Cristina Silvano `[通讯]` (Politecnico di Milano)

**通讯引用:** 3889 | [OpenAlex ID](https://openalex.org/A5031461662)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

实现了基于GPU的多级超图划分算法，支持每个划分大小和入射超边数量的硬性约束。

**💡 创新点**

创新点在于针对约束问题设计的并行策略：把邻域完全预先 materialize、在共享内存中构造直方图、使用伪森林匹配、事件驱动的约束校验等。

**🔧 技术方法**

使用 CUDA GPU 并行、压缩稀疏存储、warp 级别操作、共享内存加速、事件序列化、最长有效增益子序列等技术。

**📊 数据集**

使用 10 个来自神经突触网络的超图数据集，覆盖数十万到数百万节点。

**📈 对比分析**

与 CPU 版 hMETIS、贪心重叠法和单遍填充法比较，平均 246× 的加速、15× 的加速和 12× 的差距，且连通度平均仅为原方法的 0.82×、0.71×、0.09×。

**⚠️ 局限性**

局限性在于仍采用启发式匹配和增益序列化，缺乏精确匹配的动态规划解法，且对入射约束的处理难以直接推广到其他类型约束。

---

## 33. AndroScanner: Automated Backend Vulnerability Detection for Android Applications

**arXiv ID:** 2604.14431 | [PDF](https://arxiv.org/pdf/2604.14431v1)

**作者:** Harini Dandu `[一作]` `[通讯]` (Georgia Institute of Technology), Harini Dandu (Georgia Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并实现了AndroScanner，一个自动化流程，用于从Android APK中提取后端API、检测其安全漏洞并向开发者提供可操作的修复建议。

**💡 创新点**

创新点在于将静态分析（apktool、androguard）与动态Instrumentation（Frida）结合，实现对内部与外部API的全面识别，并利用APIFuzzer对API进行OWASP API安全漏洞扫描，首次发现了生产招聘App的Excessive Data Exposure漏洞。

**🔧 技术方法**

使用的技术包括apktool、APIKey Extractor、androguard、Frida动态hook、APIFuzzer、LibScout构建外部API列表，以及Python/Java脚本自动化流水线。

**📊 数据集**

评估数据集由两款Android App组成：一款有意设计的漏洞银行App（4个API）和真实的招聘App Hirect（20个API），并利用LibScout对5000款App生成外部API列表。

**📈 对比分析**

与现有工具相比（如Drozer、单一静态分析方法），AndroScanner在两款App中共检测到5处安全缺陷，覆盖范围更广且报告更易于开发者理解；但具体性能指标（如扫描时间、成功率）未给出。

**⚠️ 局限性**

局限性包括仅支持Android APK（不兼容iOS），对加密参数无能效检测，依赖单一漏洞扫描工具（APIFuzzer），缺乏GUI和Docker化部署，且未实现漏洞优先级评估或补丁建议。

---

## 34. Thermodynamic Diffusion Inference with Minimal Digital Conditioning

**arXiv ID:** 2604.14332 | [PDF](https://arxiv.org/pdf/2604.14332v1)

**作者:** Aditi De `[一作]` `[通讯]`, Aditi De

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可在生产规模下实现热力学扩散推理的完整架构，解决了非局部跳跃连接和输入条件化两大技术瓶颈。

**💡 创新点**

创新点在于：①分层双线性耦合实现 U‑Net 跳跃连接，仅需 O(Dk) 物理连接；②设计了仅 2560 参数的最小数字接口，克服 2600 倍信号缺失障碍，恢复接近 oracle 的输出。

**🔧 技术方法**

采用热力学计算与过阻尼 Langevin 动力学，将逆扩散过程映射为物理势能平衡；通过对卷积权重 Gram 矩阵进行低秩 SVD 构造双线性跳跃；利用 4 维线性瓶颈与 16 单隐藏层 MLP 的编码器-传输网络生成输入特定偏置；最终通过 ADC/DAC 读取/注入偏置实现硬件推理。

**📊 数据集**

以 MNIST（32×32，4 通道）为实验数据集，训练 8.1M 参数的 U‑Net，并在激活平均的 256 对激活上评估系统性能。

**📈 对比分析**

与全数字 U‑Net 的 oracle 余弦相似度对比，完整体系达到 0.9906 的相似度，理论能量节约约 10⁷ 倍；同时保持数字接口计算占比不到 0.1%。

**⚠️ 局限性**

主要局限在硬件实现：ADC/DAC 精度、混合时间与四次非线性对系统稳定性的影响；此外，系统性能受 Gram 近似精度限制，需进一步提升权重的低秩可分解质量。

---

## 35. APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI

**arXiv ID:** 2604.14362 | [PDF](https://arxiv.org/pdf/2604.14362v1)

**作者:** Pratyay Banerjee `[一作]` (Amazon), Ankit Chadha `[通讯]` (Amazon)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5027218595)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出APEX-MEM——一种基于属性图的长时对话记忆框架，支持事件与实体同等建模、增量append‑only存储以及检索时冲突解决；

**💡 创新点**

创新点：① 领域无关本体将事件与实体同等表示，实现细粒度时序推理；② append‑only事件存储保留信息完整演化；③ 结合实体链接、GraphSQL 与混合检索的多工具检索代理，实现检索时的冲突与演化解析；

**🔧 技术方法**

技术手段：大语言模型驱动事实抽取与实体/属性解析；密集语义检索+结构化LLM推理；SQLite属性图存储；ReAct式工具调用（EntityLookup、GraphSQL、Search、SchemaViewer）；时间表达标准化 ISO‑8601；软规范化合并与增量更新；

**📊 数据集**

使用数据集：LOCOMO、LongMemEval、SealQA‑Hard（以及公开对比基线数据）；

**📈 对比分析**

对比方法：与MIRIX、Mem0、Zep、MemGPT等多种基线及检索增强模型对比；APEX‑MEM在LOCOMO总体准确率达88.88%，在单跳、双跳、时序、开放域、对抗等子任务均领跑；在LongMemEval达到86.2%；在SealQA‑Hard达到40.15%；Ablation实验表明每个工具均显著提升性能；

**⚠️ 局限性**

局限性：① 构建成本高，依赖大模型；② 本体虽通用但仍难覆盖高度专业化领域；③ 对QnA代理的工具调用准确性敏感，尤其GPT4o表现欠佳；④ 在噪声多文档场景（SealQA‑Hard）仍有提升空间；⑤ 需要多次工具调用导致响应延迟；⑥ 仅支持文本输入，未扩展多模态。

---

## 36. Geometric Metrics for MoE Specialization: From Fisher Information to Early Failure Detection

**arXiv ID:** 2604.14500 | [PDF](https://arxiv.org/pdf/2604.14500v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22411 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于信息几何的MoE专家专化监测框架，定义了Fisher专化指数（FSI）和Fisher异质性得分（FHS），并用它们实现早期训练失败预测和恢复干预。

**💡 创新点**

创新点在于：①证明传统指标缺乏参数化不变性；②将专家分配视为概率单纯形上的Riemann几何问题，给出精确的几何界定和逼近定理；③提出FHS阈值1用于高置信度的失败预测，并给出理论阈值证明和干预策略；④将这些指标在大规模MoE（高达2.7B参数、64专家）上验证，展示显著性能提升。

**🔧 技术方法**

核心技术包括信息几何（Fisher信息度量与Fisher–Rao距离）、高维几何推导、梯度下降与镜像下降的几何映射、矩阵伯努利界定、以及分布式FHS计算的AllGather优化。

**📊 数据集**

使用的数据集包括语言建模任务的WikiText‑103和C4，以及视觉任务的ImageNet-1K；实验还涵盖多规模Switch Transformer（125M–2.7B参数）和V‑MoE（ViT‑B/16）。

**📈 对比分析**

与传统的余弦相似度、路由熵、梯度范数、StableMoE以及验证损失早停等方法对比，FSI与FHS在与下游性能的相关性（r≈0.91）和失败预测AUC（0.89）上均显著优于基线，且相较验证损失早停，提升了23%且仅需40倍更少的检查点。

**⚠️ 局限性**

局限性包括：实验规模受限于C4的10M标记；当前框架仅适用于离散专家混合，扩展到连续混合需构造更复杂的Fisher度量；对自然梯度或更复杂的路由机制的集成仍待研究。

---

## 37. TOPCELL: Topology Optimization of Standard Cell via LLMs

**arXiv ID:** 2604.14237 | [PDF](https://arxiv.org/pdf/2604.14237v1)

**作者:** Zhan Song `[一作]` (University of Maryland), Cunxi Yu `[通讯]` (University of Maryland)

**通讯引用:** 1157 | [OpenAlex ID](https://openalex.org/A5029321729)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用大型语言模型（LLM）与GRPO强化学习相结合，对标准单元拓扑进行生成式优化，取代传统递归搜索，显著降低设计时间；

**💡 创新点**

创新点在于将拓扑搜索框架化为LLM可学习的生成任务，并通过GRPO实现物理感知的策略更新，使模型在零射击条件下即可推导出高质量、可路由的拓扑；

**🔧 技术方法**

核心技术包括Qwen2.5-Coder LLM、Group Relative Policy Optimization (GRPO)、LLM‑Guided Topology Permutation算法、基于P&R反馈的图神经网络奖励模型以及SGLang/Verl框架实现的训练部署；

**📊 数据集**

构造了覆盖所有三输入单输出布尔函数的全枚举数据集，共7,918个拓扑，其中2,039个不可路由样本用于训练奖励网络；

**📈 对比分析**

与SO3‑Cell以及多种基础模型对比，GRPO‑7B在2nm节点实现77.3%路由率、PDA拥塞3.90，且在将模型集成到SO3‑Cell后，平均速度提升85.9×，且在7nm节点保持与SO3‑Cell相当的布局质量；

**⚠️ 局限性**

目前仅在三输入、2nm节点进行训练和验证，尚未在更大输入规模或其他技术节点上全面验证，且模型对极端布尔函数或特殊物理约束的适应性仍需进一步探索。

---

## 38. On Tackling Complex Tasks with Reward Machines and Signal Temporal Logics

**arXiv ID:** 2604.14440 | [PDF](https://arxiv.org/pdf/2604.14440v1)

**作者:** Ana María Gómez Ruiz `[一作]`, Alexandre Donzé `[通讯]` (Université Grenoble Alpes)

**通讯引用:** 5510 | [OpenAlex ID](https://openalex.org/A5081324945)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种将奖励机器（RM）与Signal Temporal Logic（STL）谓词结合的RL框架，用于复杂任务的控制设计。

**💡 创新点**

创新点在于把STL事件直接作为RM转移条件，利用在线监测的鲁棒性指导训练，从而提升学习效率、收敛速度和可解释性。

**🔧 技术方法**

采用奖励机器、STL在线监测、PPO训练算法以及Python Gymnasium环境等技术。

**📊 数据集**

在MiniGrid、Cart‑Pole和Highway‑Env三个基准环境上进行实验。

**📈 对比分析**

与传统PPO做对比，RM+STL方法在奖励、收敛速度和任务完成率上均优于基线，尤其在更大规模的MiniGrid上表现显著。

**⚠️ 局限性**

局限在于对复杂STL规范的手工设计成本高，在线监测开销随任务复杂度增加，且未验证在真实机器人上的可迁移性。

---

## 39. CSRA: Controlled Spectral Residual Augmentation for Robust Sepsis Prediction

**arXiv ID:** 2604.14532 | [PDF](https://arxiv.org/pdf/2604.14532v1)

**作者:** Honglin Guo `[一作]` (Tianjin University), Yuehao Shen `[通讯]` (Tianjin Medical University General Hospital)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5101422094)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一种控制光谱残差增广（CSRA）框架，用于在短窗口多系统ICU时间序列中生成结构化且临床可解释的轨迹变异，从而提升脓毒症预测的准确性。

**💡 创新点**

创新点包括：①按临床系统划分变量并提取系统级与全局表征；②在频域对残差进行输入自适应调制，生成有结构的增广样本；③将增广器与下游预测器端到端联合训练，并通过锚一致性损失与控制器正则化提高增广的可控性与稳定性。

**🔧 技术方法**

使用的核心技术包括：系统级与全局表征编码、离散余弦变换（DCT）频域增广、输入自适应控制器、anchor一致性损失、控制器正则化以及统一的端到端优化框架。

**📊 数据集**

实验使用了MIMIC-IV脓毒症队列（34,793例）作为主数据集，并在外部数据集ZiGongICUinfection（2,502例）进行泛化验证。

**📈 对比分析**

与无增广、InfoTS、AutoTCL、TrivialAugment、A2Aug、AutoDA-Timeseries等方法，以及三种下游模型（Linear、LSTM、Transformer）进行比较。CSRA在回归任务上平均MSE下降10.2%、MAE下降3.7%，在分类任务上AUROC平均提升1.1%、AUPRC提升2.6%；在短窗口、长预测期、少量训练数据等极端场景下仍保持领先优势。

**⚠️ 局限性**

主要限制包括：仅适用于规则采样时间序列，尚未处理不规则观测；依赖预定义的临床系统划分，需进一步验证对更复杂变量依赖关系的适用性；对不同病种或更大范围临床数据的泛化能力仍待进一步评估。

---

## 40. LLMs taking shortcuts in test generation: A study with SAP HANA and LevelDB

**arXiv ID:** 2604.14437 | [PDF](https://arxiv.org/pdf/2604.14437v1)

**作者:** Vekil Bekmyradov `[一作]` (TH Koen), Thomas Bartz-Beielstein `[通讯]` (TH Koen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型（LLM）在自动化软件测试生成中的表现，通过对开源项目LevelDB与商业数据库SAP HANA的比较，探究模型的记忆与泛化能力。

**💡 创新点**

创新点在于将认知科学的机制化评估（Mitchell方法）与软件工程的变异测试结合，使用编译器反馈循环直接观察LLM在修复过程中产生的“捷径”行为，并通过零污染的专有代码库检验模型是否真正具备推理能力。

**🔧 技术方法**

技术包括：Mitchell的机制化评估框架、变异测试（mutation score）评估、迭代编译器反馈修复循环、代码覆盖率（行/分支）评估、以及多种LLM模型（GPT‑5、Claude 4 Sonnet、Gemini 2.5 Pro、Qwen‑3‑Coder）的实验部署。

**📊 数据集**

数据集为开源的LevelDB代码库（≈21 k 行）与SAP HANA的一个未公开的核心组件（≈70 k 行），前者在模型训练中可能已存在，后者被保证不在训练数据中。

**📈 对比分析**

比较方法：在同一评估框架下分别进行测试扩增和完整测试生成（含/不含依赖文件），并记录行覆盖率、分支覆盖率、变异得分与编译成功率。结果显示：在LevelDB上四种模型均实现100 %变异得分，表现几乎完美；而在SAP HANA上变异得分仅在10–25 %之间，且编译成功率虽提升但多为“空”或“无断言”测试，证明模型主要靠捷径获得编译通过。

**⚠️ 局限性**

局限性包括：仅评估SAP HANA的单一组件，实验重复次数有限；LevelDB与SAP HANA在规模与结构复杂度上差异大；变异操作集有限，可能未覆盖所有缺陷；缺乏正式统计显著性检验，导致定量结论的稳健性受限。

---

## 41. Asynchronous Probability Ensembling for Federated Disaster Detection

**arXiv ID:** 2604.14450 | [PDF](https://arxiv.org/pdf/2604.14450v1)

**作者:** Emanuel Teixeira Martins `[一作]` (Federal University of Viçosa), Flávio de Oliveira Silva `[通讯]` (University of Minho)

**通讯引用:** 786 | [OpenAlex ID](https://openalex.org/A5061596292)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于异步概率聚合与反馈蒸馏的分布式灾害图像识别框架（AIDER），通过在MQTT中交换类别概率向量实现跨异构CNN的协同训练；

**💡 创新点**

创新点在于将联邦学习的参数共享改为概率级别的轻量级聚合，支持异步、无同步、模型异构的环境，同时引入反馈蒸馏让本地模型学习全局集成分布；

**🔧 技术方法**

使用的技术包括深度CNN（EfficientNet、MobileNetV2/V3、ResNet、SqueezeNet）、概率聚合、stacking、遗传算法（GA）、粒子群优化（PSO）、MQTT消息中间件以及知识蒸馏；

**📊 数据集**

使用的公开数据集为AIDER（含火灾/烟雾、洪水、倒塌结构、交通事故、正常场景共五类，且高度不平衡）；

**📈 对比分析**

与传统中心化训练和基于FedAvg的联邦学习进行对比；实验显示，概率聚合方法在保持或略优于联邦学习的准确率（最高可达约0.982）同时将通信量降低三到四个数量级（从百兆级压缩到约15万字节）；

**⚠️ 局限性**

局限性包括需要共享参考样本集来保证概率对齐，依赖单一MQTT代理可能产生单点故障，且在极端非IID或对抗性客户端场景下鲁棒性尚待进一步验证。

---

## 42. Smart But Not Moral? Moral Alignment In Human-AI Decision-Making

**arXiv ID:** 2604.14371 | [PDF](https://arxiv.org/pdf/2604.14371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 43. How to Fine-Tune a Reasoning Model? A Teacher-Student Cooperation Framework to Synthesize Student-Consistent SFT Data

**arXiv ID:** 2604.14164 | [PDF](https://arxiv.org/pdf/2604.14164v1)

**作者:** Zixian Huang `[一作]` (Shanghai AI Laboratory), Qipeng Guo `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出教师‑学生协作数据合成框架TESSY，交替生成教师推理内容与学生风格文本，生成离线SFT数据；

**💡 创新点**

将生成过程细分为能力Token与样式Token，利用生成‑回滚与边界预测器精准控制，实现在推理模型SFT中消除分布不匹配导致的灾难性遗忘；

**🔧 技术方法**

生成‑回滚策略、边界预测器（序列标注）、交替生成、vLLM + prefix caching、XTuner等；

**📊 数据集**

训练集：OpenThoughts、NVIDIA Nemotron 以及 GPT‑OSS‑120B 生成的80k编程题；评估集：LiveCodeBench‑V5/V6/Pro、OJBench、AIME‑2024/2025、OlympiadBench、GPQA；

**📈 对比分析**

与 Teacher‑Only、Teacher‑Score、Teacher‑Answer、Teacher‑Think 等多种SFT数据合成方法对比；TESSY 在 LiveCodeBench‑Pro、OJBench 等基准上分别提升 3.25%–10.02% 以上；在其他评测集亦保持正向增益；

**⚠️ 局限性**

对生成长度受限时仍略逊于教师单独生成；未结合拒绝采样等数据质量提升技术；对多模态支持有限；需进一步细化边界识别与验证更广泛场景。

---

## 44. QU-NLP at ArchEHR-QA 2026: Two-Stage QLoRA Fine-Tuning of Qwen3-4B for Patient-Oriented Clinical Question Answering and Evidence Sentence Alignment

**arXiv ID:** 2604.14175 | [PDF](https://arxiv.org/pdf/2604.14175v1)

**作者:** Mohammad AL-Smadi `[一作]` `[通讯]`, Mohammad AL-Smadi

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一套统一的系统，用于 ArchEHR‑QA 共享任务的答案生成（Subtask 3）和证据句子对齐（Subtask 4）;

**💡 创新点**

创新点在于：①使用 QLoRA 4‑bit 微调的 Qwen3‑4B 进行两阶段训练（先在 emrQA‑MedSQuAD 上做领域适配，再在极小的 20 条开发样本上微调）; ②将 BM25、TF‑IDF 与细调交叉编码器相结合，构建加权投票式召回-精度平衡的证据检索器；

**🔧 技术方法**

核心技术包括：4‑bit NF4 量化、LoRA/QLoRA 参数高效微调、指令调优的 Qwen3‑4B、BM25、TF‑IDF 余弦相似度、交叉编码器（fine‑tuned on 20 dev cases）、加权投票融合；

**📊 数据集**

数据集为 ArchEHR‑QA（基于 MIMIC‑III 与 MIMIC‑IV 的去标识化病历），包括 dev（20 条），test（100 条）与 test‑2026（47 条）三份；

**📈 对比分析**

在答案生成上，系统整体得分 32.87（BLEU 9.42, ROUGE‑L 27.04, SARI 55.42, BERTScore 43.00, AlignScore 25.28, MEDCON 37.04），与榜首 36.39 仅差 3.52；在证据对齐上，微 F1 从 CE 单独 60.82 提升到 67.16，虽仍低于榜首 81.5，但相较 CE 单独已提升 6.3 点；

**⚠️ 局限性**

局限性包括：①仅有 20 条开发样本导致任务‑特定微调过拟合与泛化不足；②领域适配阶段仅使用 30k QA，未充分利用 400k QA 的潜力；③证据检索仍受限于句子级别的简化，难以捕捉更细粒度的上下文；④生成模型在 AlignScore 与 MEDCON 上表现欠佳，说明对源文献的精准引用与医学术语使用仍有提升空间。

---

## 45. LLM Predictive Scoring and Validation: Inferring Experience Ratings from Unstructured Text

**arXiv ID:** 2604.14321 | [PDF](https://arxiv.org/pdf/2604.14321v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 46. On the Expressive Power and Limitations of Multi-Layer SSMs

**arXiv ID:** 2604.14501 | [PDF](https://arxiv.org/pdf/2604.14501v1)

**作者:** Nikola Zubić `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**通讯引用:** 37695 | [OpenAlex ID](https://openalex.org/A5057116316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文理论分析了多层状态空间模型（SSMs）的表达能力与极限，证明了在无CoT的情况下，L层SSM完成(L+3)-函数合成任务需要满足d²p = Ω(N/L³)的下界；通过构造可在(K+1)层、dp = O(log N)的SSM实现任意K函数合成，从而展示了深度层数与表达能力的阶梯关系。

**💡 创新点**

创新点在于：① 将SSM模型与多轮通信复杂度相结合，得到新的下界证明；② 引入“在线”与“离线”Chain‑of‑Thought（CoT）的区分，发现在线CoT可将多层SSM等价于一遍通行的流处理算法；③ 证明在基模型下宽度与精度不可互换，而在允许在线CoT后两者可通过总持久记忆度量完全等价。

**🔧 技术方法**

使用的技术包括：前向通信模型（forward communication model）与指针追踪（pointer chasing）下界，Affine状态更新的矩阵向量摘要，构造精确的读出函数来实现函数合成，以及通过离线/在线CoT的模拟证明两种模型的表达能力差异。

**📊 数据集**

本工作为纯理论分析，没有使用任何实际数据集；所有证明均基于抽象模型与数学推导。

**📈 对比分析**

由于本文聚焦于理论上限与下界，没有实验比较；但通过上界构造与下界证明，展示了在满足给定精度与宽度约束时，L层SSM与在线CoT模型在表达能力上存在显著差异。

**⚠️ 局限性**

局限性包括：① 下界证明基于指针追踪的经典通信复杂度，可能无法直接推广到所有SSM结构；② 对于实际可训练的SSM参数化（如Mamba、S4等），本文的构造可能难以实现；③ 在无CoT模型下，宽度与精度的不等价性仅在理论层面体现，实际模型实现时的数值误差与训练难度未被考虑。

---

## 47. Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters

**arXiv ID:** 2604.14174 | [PDF](https://arxiv.org/pdf/2604.14174v1)

**作者:** Bryan Sanchez `[一作]` `[通讯]`, Bryan Sanchez

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种在 Qwen3 大语言模型中后置适配器，通过冻结隐藏状态并在生成时仅修正当前预测位置的方式，纠正政治敏感话题的被压制的概率分布。

**💡 创新点**

创新点在于仅使用约 786K 参数的后变压器适配器，冻结所有模型权重，仅在隐藏状态空间进行可微分修正，同时解决了 MLX 框架中的梯度静默 bug。

**🔧 技术方法**

采用了两种适配器结构（SwiGLU gated 与线性 bottleneck）以及锚定训练、hinge 损失、AdamW 优化器等技术。

**📊 数据集**

使用了 31 条针对中国敏感议题（天安门、藏区、维吾尔等）在四种强度级别下的事实与干扰词构成的数据集，并对 16 条留存事实进行泛化评估。

**📈 对比分析**

通过对 5 个随机拆分的训练/测试集计算 log‑probability margin 的通过率，发现适配器在所有规模（4B、8B、14B）下平均通过率从 11%–39% 远高于基线 6.5%，且对指令模型在单步预测时能产生更连贯且更少审查的文本。

**⚠️ 局限性**

主要局限在于留存集规模小、评估仅为定性生成、仅针对 Qwen3 家族、以及未在更大多样化知识集合上验证。

---

## 48. Portfolio Optimization Proxies under Label Scarcity and Regime Shifts via Bayesian and Deterministic Students under Semi-Supervised Sandwich Training

**arXiv ID:** 2604.14206 | [PDF](https://arxiv.org/pdf/2604.14206v1)

**作者:** Adhiraj Chattopadhyay `[一作]` `[通讯]` (Indian Institute of Technology Roorkee), Adhiraj Chattopadhyay (Indian Institute of Technology Roorkee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于教师-学生学习的组合优化框架，利用 CVaR 教师生成监督标签，训练 Bayesian 与 Deterministic 神经网络学生，并在低样本环境下通过合成数据增强进行训练与评估。

**💡 创新点**

创新点包括：①将半监督“Sandwich”训练与 Bayesian 神经网络首次结合应用于资产配置；②通过 Bayesian 稀释实现隐式交易成本降低；③在跨域 (D2A) 测试中发现高波动期模型表现提升的“高波动悖论”，证明因子分解的层级泛化能力；④在多约束、多市场情景下展示 Pareto 最优的风险收益权衡。

**🔧 技术方法**

核心技术：CVaR 最优化教师、Variational Bayesian 神经网络、Deterministic DNN、半监督 Sandwich 训练、因子模型 + t‑copula 合成数据、软最大化实现权重约束、滚动微调与冻结、Sharpe/CVaR/MDD/交易成本等性能指标。

**📊 数据集**

数据集：36 只 ETF 组成的周度组合，时间段 2015‑2025 共 575 周；真实标记 104 周，合成 323 周（基于 VAR + t‑copula 的因子残差模型），C2A 与 D2A 评估使用 2022‑2026 实盘回报（同域与异域）。

**📈 对比分析**

通过 GRID_3×5 种子网格在合成测试、C2A 同域、D2A 异域三层评估。BNN‑S 在 L3 约束下实现 Sharpe 2.44、CVaR ≈‑1.5%、周交易率 11%，比 CVaR 教师和传统均值-方差优越；在高波动期 D2A 相比 C2A 提升 140‑276% Sharpe，展现出显著的跨域适应性；模型间胜率矩阵显示蒸馏模型普遍优于仅监督模型。

**⚠️ 局限性**

局限性：对市场周期高度敏感（种子不同可导致 50% Sharpe 变动）；真实标记样本仅 104 周，难以覆盖极端危机；合成生成器假设可能与实际分布不匹配；约束参数的校准缺乏理论依据；缺少流动性或市场失效的检测机制；教师质量上限限制学生潜在表现；未加入显式的 regime‑检测或自适应集成策略。

---

## 49. TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations

**arXiv ID:** 2604.14223 | [PDF](https://arxiv.org/pdf/2604.14223v1)

**作者:** Ashmi Banerjee `[一作]` (Technical University of Munich), Yashar Deldjoo `[通讯]` (Polytechnic University of Bari)

**通讯引用:** 2875 | [OpenAlex ID](https://openalex.org/A5009954943)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了TRACE，一个多代理LLM驱动的可持续旅游对话式推荐系统，通过交互式提问和对比解释引导用户选择更环保的旅行方案。

**💡 创新点**

引入代理式因果对照解释和澄清性问题来隐式推断用户的可持续性偏好；多代理架构实现可持续性与相关性平衡；通过解释代理进行说服性叙述实现非强制性的数字推动。

**🔧 技术方法**

基于Google Agent Development Kit（ADK）与Vertex AI的多代理LLM；FastAPI+Firestore进行状态管理；Chainlit前端；Docker+Cloud Run部署；使用Gemini/Claude等大型语言模型进行推理与生成。

**📊 数据集**

使用SynthTRIPS合成旅游查询数据集做示例；通过24名受访者的用户研究收集交互数据。

**📈 对比分析**

通过107条有效会话的用户研究评估：用户对澄清问题、解释说服度和重新考虑率进行量化；语义相似度指标评估内部对齐；平均响应时间23秒，最大38秒，显示系统能在实时交互中实现可持续引导。

**⚠️ 局限性**

仅支持单一欧洲城市行程；可能产生热点迁移效应，导致新热点出现；多代理模型增加计算负担；对LLM的鲁棒性与安全性仍需进一步完善。

---

## 50. Improving Human Performance with Value-Aware Interventions: A Case Study in Chess

**arXiv ID:** 2604.14465 | [PDF](https://arxiv.org/pdf/2604.14465v1)

**作者:** Saumik Narayanan `[一作]` (Washington University in St. Louis), Chien-Ju Ho `[通讯]` (Washington University in St. Louis)

**通讯引用:** 1173 | [OpenAlex ID](https://openalex.org/A5101621594)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于人类行为模型的价值感知干预方法，针对有限干预预算设计单次和多次干预策略。

**💡 创新点**

利用人类政策与价值函数的不一致性来确定干预时机，证明单次干预时选择最大化人类价值函数的动作最优，并给出多次干预的可行近似。

**🔧 技术方法**

使用行为克隆（BC）学习人类政策与价值函数，结合马尔可夫决策过程（MDP）框架计算 Q^π_H、Δ^π_H，并在象棋局面中评估干预。

**📊 数据集**

基于大规模 Lichess 在线棋局数据（256M 条棋局，玩家等级 400–2800）并使用 Leela T82 预训练模型微调得到人类行为模型。

**📈 对比分析**

与人类走子和最强棋引擎 Stockfish 干预做对比；单次干预下在所有水平上均优于 Stockfish，且多次干预时在低干预预算下优于 Stockfish；实验包括大规模模拟和 20 位玩家的 600 场人类实验，验证了理论预测。

**⚠️ 局限性**

依赖学习的行为模型可能无法完全捕捉真实人类行为；模拟实验基于 BC 滚动，未使用真实人类轨迹；人类实验假设玩家完全遵循建议；仅在象棋域验证，其他领域需进一步探索。

---

## 51. Filament: Denning-Style Information Flow Control for Rust

**arXiv ID:** 2604.14357 | [PDF](https://arxiv.org/pdf/2604.14357v1)

**作者:** Jeffrey C. Ching `[一作]` (Duke University), Danfeng Zhang `[通讯]` (Duke University)

**通讯引用:** 126030 | [OpenAlex ID](https://openalex.org/A5059976286)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

实现了一个名为 RustIFC 的 Denning‑Style 信息流控制（IFC）库，能够在不改动 Rust 编译器的前提下，对变量级别的显式流和程序计数器级别的隐式流进行细粒度跟踪与检查。

**💡 创新点**

创新点包括：① 引入 pc_block! 宏实现编译时程序计数器（PC）标签，以支持隐式流检测；② 设计 fcall! 和 mcall! 两个宏，自动将标准库及第三方函数与标签类型相结合，避免手动 declassify/rewrap；③ 通过 relabel! 宏和 Rust 的类型推断降低注解负担，实现低侵入性的显式流检查；④ 结合以上技术构建了一个完整的、无需编译器扩展的 IFC 框架。

**🔧 技术方法**

技术手段主要是 Rust 的 trait、泛型、phantom 类型以及宏（proc‑macro）系统，用以在编译期实现标签运算（Join/FlowsTo）和类型安全的值包装；同时利用 Rust 的类型推断完成标签传播，减少显式注解。

**📊 数据集**

评估数据集为五个实际项目：Calendar、Battleship、Spotify TUI、Servo（Mozilla 浏览器引擎）和 JPMail（Jif 项目改写）。这些项目分别代表了日程管理、游戏、终端音乐客户端、浏览器渲染引擎以及安全邮件系统等不同应用场景。

**📈 对比分析**

比较方法：将 RustIFC 与 Cocoon（基于 block‑level IFC 的同类实现）在同一 Rust 1.69 环境下对上述项目进行重写，统计：① 注解行数（Label+API）、② 逃逸开关（declassify、unchecked_operation）使用次数、③ 编译时间占比。实验结果显示：RustIFC 的注解量略低或相当，逃逸开关显著减少，编译时间仅略高于 Cocoon（约 1–2 %），并在大项目中几乎不产生额外开销。

**⚠️ 局限性**

局限性：① 仅支持静态安全标签，无法处理运行时动态标签；② 不涉及侧信道（如时间/功耗）保护；③ 目前仅实现单线程执行，未考虑并发共享数据的 IFC；④ PC 块的标签需要手动指定，缺乏自动推断；⑤ 通过宏实现的功能可能无法覆盖所有第三方库的复杂使用场景，且宏扩展的可维护性仍待评估。

---

## 52. Faithfulness Serum: Mitigating the Faithfulness Gap in Textual Explanations of LLM Decisions via Attribution Guidance

**arXiv ID:** 2604.14325 | [PDF](https://arxiv.org/pdf/2604.14325v1)

**作者:** Bar Alon `[一作]` (Tel Aviv University), Lior Wolf `[通讯]` (Tel Aviv University)

**通讯引用:** 26081 | [OpenAlex ID](https://openalex.org/A5078102229)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首先通过基于反事实的评估框架，对大型语言模型生成的文本解释进行知识性可信度（epistemic faithfulness）评估，并发现大多数模型的解释往往与内部推理不一致；随后提出一种训练无关的“Faithfulness Serum”方法，利用PE‑LRP得到的 token‑级重要性热图，按权重 α 在注意力层注入，从而在生成解释时引导模型关注真正决定答案的文本。

**💡 创新点**

创新点包括（1）首次引入反事实提示注入与判定机制，用来验证解释是否真实反映模型内部依据；（2）设计了基于 PE‑LRP 的注意力层干预策略，即 Faithfulness Serum，能显著提升解释的知识性可信度且不需额外训练；（3）在多模型、多提示、开放式与封闭式任务上系统评估并验证该方法的普适性与效果。

**🔧 技术方法**

核心技术为：PE‑LRP（精准 token 重要性归因）、注意力层注入干预（α 控制）、反事实评估（CT 与 LLM 判定）、logit‑lens 分析用于确定干预层、TTα（自适应 α）以及对比实验中的多模型推理与生成。

**📊 数据集**

使用的数据集包括：MMLU（多选与开放式）、CommonsenseQA、SciQ、ARC‑C、OpenBookQA，并在 MMLU 上做职业角色提示的偏差实验；评估覆盖 1,000 例/类型的多模型样本。

**📈 对比分析**

在 Llama‑3.1 8B、Qwen 2.5 7B 等开放模型上，Faithfulness Serum 将解释的可信度提升约 10–30% 甚至近 100%（相对增幅），在多提示设置（General 与 Protocol‑Specific）与多数据集上均表现出一致性；TTα 进一步提升 1–3%；对闭源模型（GPT‑4o、Gemini 2.0 Flash）虽受限于 API，仍能在一定程度上提升可信度，但提升幅度明显低于开放模型。

**⚠️ 局限性**

局限性包括：仅适用于 predict‑and‑explain 框架，无法实现预先解释；干预仅针对注意力层，忽略 feed‑forward、残差与归一化等组件；α 参数需要手工或自适应调优，可能因任务或提示差异而变化；方法在极端提示或对抗性攻击下可能产生不稳定或不可解释的行为；研究聚焦在知识性可信度，未覆盖其它解释质量维度。

---

## 53. Challenges and Future Directions in Agentic Reverse Engineering Systems

**arXiv ID:** 2604.14317 | [PDF](https://arxiv.org/pdf/2604.14317v1)

**作者:** Salem Radey `[一作]` (University of Wisconsin-Madison), Kassem Fawaz `[通讯]` (University of Wisconsin-Madison)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性分析并总结了当前基于大型语言模型的代理式逆向工程（RE）系统在静态、动态与混合分析中的工作原理、技术栈与局限性，提出六大核心挑战并给出针对每类分析的未来研究方向；

**💡 创新点**

创新点在于将代理式 RE 按分析类型（静态/动态/混合）构建可视化分类图谱，系统识别并量化了如令牌限制、反混淆能力不足、缺乏命令安全门槛、超时与仿真依赖等问题，并提出针对性的技术改进路线图；

**🔧 技术方法**

主要技术包括：大语言模型代理（如 GPT‑4、LLM4Decompile、disasLLM、WaDec、LAMD 等），传统逆向工具（Ghidra、IDA Pro、GDB、Enigma+、HackSynth 等），以及对令牌化、去混淆、动态分析安全框架的探讨；

**📊 数据集**

数据集方面，主要引用了公开的 BinMetric 基准（1000 个二进制分析问题）以及多平台（x86, ARM, WebAssembly）二进制样本；

**📈 对比分析**

本工作并未在统一实验平台上对比评估性能，而是通过对现有开源系统代码库与论文的静态审计与实证分析，展示各类系统在处理混淆、令牌限制、命令执行等场景的实际表现与不足；

**⚠️ 局限性**

局限性包括：仅聚焦本地二进制分析，未覆盖网络攻击情景；对复杂混淆或高度动态加载的二进制支持有限；缺乏实验验证新提议技术的实际效果；对真实攻击者可能的对抗策略的评估仍不足。

---

## 54. Giving Faces Their Feelings Back: Explicit Emotion Control for Feedforward Single-Image 3D Head Avatars

**arXiv ID:** 2604.14541 | [PDF](https://arxiv.org/pdf/2604.14541v1)

**作者:** Yicheng Gong `[一作]` (Nanjing University), Yan Lu `[通讯]` (Microsoft Research Asia)

**通讯引用:** 20280 | [OpenAlex ID](https://openalex.org/A5035278528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了单图像可显式控制情绪的3D头部头像生成框架，能够在保持身份一致的前提下通过情绪标签实时调整几何和外观；

**💡 创新点**

创新点在于将情绪提升为第一类控制变量，提出双路情绪感知调制（几何归一化+外观调制），实现情绪与动作、身份的完全分离，并构建了时间同步、情绪一致的多身份数据集，保证模型能够在无迭代优化的前提下泛化到新身份；

**🔧 技术方法**

使用FLAME参数模型、跨注意力和Transformer模块实现双路调制；通过情绪嵌入向量和双向调制网络，结合2D重现X‑NeMo完成情绪迁移；采用语义一致的情绪标签作为显式控制输入；

**📊 数据集**

数据集包括：EmoTalk3D生成的情绪同步anchor序列；基于这些anchor利用X‑NeMo对8,750个身份（1,250身份的长序列）进行情绪迁移得到的多身份、时序一致的合成/真实视频；另外使用VFHQ等真实面部视频作为额外训练与评估素材；

**📈 对比分析**

与现有基线（LAM、Zhang et al.等）在自/跨reenactment、情绪迁移、用户研究等多维度进行比较；在PSNR/SSIM/LPIPS、CSIM、AED/APD等量化指标上保持与基线相当的重建质量；在情绪迁移和解耦指标上明显优于基线；在用户评估中情绪可识别度和生动度得分均排名第一；

**⚠️ 局限性**

局限性：依赖情绪同步anchor限制了语音内容和情绪动态的多样性；在大偏转角或极端姿态下，模型表现受限，需进一步扩充极端姿态的监督样本；

---

## 55. MixAtlas: Uncertainty-aware Data Mixture Optimization for Multimodal LLM Midtraining

**arXiv ID:** 2604.14198 | [PDF](https://arxiv.org/pdf/2604.14198v1)

**作者:** Bingbing Wen `[一作]` (University of Washington), Manjot Bilkhu `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MixAtlas 框架，对多模态中训练数据进行可解释且计算高效的混合优化；

**💡 创新点**

通过将数据按任务监督类型与图像概念两轴拆解，利用小型代理模型与高斯过程代理实现低成本、可解释的混合搜索，并实现跨模型规模迁移；

**🔧 技术方法**

使用任务监督合成（详细描写、OCR、定位式字幕、检测、VQA）和 CLIP 视觉嵌入聚类得到图像概念；构建候选混合池，利用代理模型训练+评估，拟合 GP 代理并采用 UCB 探索；

**📊 数据集**

中训练语料基于公开多模态数据集（Conceptual Captions、LLaVA‑Next 等），在统一指令跟随格式下生成多任务标签；

**📈 对比分析**

与 Uniform、Chameleon、RegMix 等基线对比；在 Qwen2‑7B/7.5‑7B 上，MixAtlas 混合平均提升 8.5%–17.6%，在多项指标上领先；并能在同样计算预算下比基线收敛速度快 50%+；代理模型混合在 0.5B 规模上发现的最佳混合可直接迁移到 7B，保持或提升性能；

**⚠️ 局限性**

局限：只分别优化任务轴与概念轴，未处理两轴交叉的完整组合；实验仅覆盖 Qwen 系列与 LLaVA‑Next 训练管线，未验证对其他语言/视觉编码器与 MLLM 体系的泛化；

---

## 56. Controlling Authority Retrieval: A Missing Retrieval Objective for Authority-Governed Knowledge

**arXiv ID:** 2604.14488 | [PDF](https://arxiv.org/pdf/2604.14488v1)

**作者:** Andre Bacellar `[一作]` `[通讯]`, Andre Bacellar

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并形式化控制权检索（CAR）问题，定义正确答案是被后续文件正式取代的前沿文档，并给出了对应的评价指标TCA（Tracking Authority Correctness）。

**💡 创新点**

核心创新在于：
• 证明TCA=1的必要且充分条件（前沿包含 + 无忽略的超越者）。
• 给出基于范围索引的算法在最坏情况下的上界 φ(q)·R_anchor，揭示词义匹配检索无法跨越超越关系的结构瓶颈。
• 通过三条定理（误差解耦、闭包必要、分解必需）阐释CAR的结构与难点。
• 设计两阶段检索与实体索引方案，实验验证在三大真实域（安全、法律、医疗）中显著提升TCA。

**🔧 技术方法**

采用传统BM25、TF‑IDF、密集向量检索（MiniLM、E5）、实体索引（anchor‑probe）、两阶段检索（Anchor‑Probe + 前沿恢复）、RSSG（规则基超越图）等技术；使用多级评估指标TCA、Recall@k、Acc、ProvRec。

**📊 数据集**

数据集包括：GHSA+NVD安全 advisories（159对），SCOTUS 公开判例对（122对），FDA 药品召回对（500对），以及自构造的 FinSuperQA、CVE‑PatchQA 和 LegalPrecedentQA 三个合成基准；每个数据集均覆盖不同的超越关系和实体范围。

**📈 对比分析**

与常规检索器（BM25、Dense）以及跨域两阶段方案对比。密集检索在三大域TCA@5仅 0.07–0.27，BM25 0.0–0.64；两阶段检索在所有域实现 0.77–0.97 的 TCA@5，近乎 100% 的正确率。实验还展示了 GPT‑4o‑mini 在密集检索上产生 39% 的误导性“未修补”回答，而两阶段仅 16%。

**⚠️ 局限性**

局限性：
• 需要准确的实体提取或规则表，free‑form 查询缺失实体时性能下降。
• 目前仅考虑单一确定性权威关系，未覆盖多权威、概率或流式更新场景。
• 依赖预先构建的实体索引和超越规则，迁移到新领域需手工编制。
• 评测主要在离线语料，实际在线动态更新的可扩展性尚未验证。

---

## 57. ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents

**arXiv ID:** 2604.14261 | [PDF](https://arxiv.org/pdf/2604.14261v1)

**作者:** Zhuofeng Li `[一作]` (Texas AandM University), Yu Zhang `[通讯]` (Texas AandM University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于论文专属评审 Rubric 的多代理框架（ReviewGrounder）和相应评估基准，用 LLM 生成更具证据、可操作、结构化的同行评审文本；

**💡 创新点**

创新点在于（1）用论文专属 Rubric 对评审质量进行细粒度评估，关注内容而非单一评分；（2）将评审分为草稿、检索、分析、合成四阶段的多代理流程，实现对比上下文与证据的深入 grounding；（3）即便使用中小型 LLM 也能超越更大基线模型，证明 grounding 机制的显著贡献；

**🔧 技术方法**

使用 LLMs（Phi‑4‑14B、GPT‑OSS‑120B、GPT‑4o 等）、Semantic Scholar 检索 API、reranker、以及自定义多代理架构（草稿生成、文献检索、方法分析、结果分析、整合汇总）；

**📊 数据集**

基于 DeepReview‑13K（ICLR 2024‑2025 提交及评审），并通过聚合生成论文专属 Rubric；

**📈 对比分析**

在 Rubric‑based 评估和数值字段评估两种指标下，ReviewGrounder 与 GPT‑4o、Qwen3‑32B、AgentReview、DeepReviewer 等多种基线进行公平比较；结果显示在 8 个 Rubric 维度上提升 38%–135%，在评分误差（MSE/MSE）和决策准确率上也实现显著超越；

**⚠️ 局限性**

局限性包括：未实现端到端训练的多代理流水线；只关注 LLM 评审，未探讨 meta‑review 或多轮互动；跨会议覆盖不完整，导致评估范围受限。

---

## 58. DEEP-GAP: Deep-learning Evaluation of Execution Parallelism in GPU Architectural Performance

**arXiv ID:** 2604.14552 | [PDF](https://arxiv.org/pdf/2604.14552v1)

**作者:** Kathiravan Palaniappan `[一作]` `[通讯]` (University of Colorado Colorado Springs), Kathiravan Palaniappan (University of Colorado Colorado Springs)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过DEEP-GAP框架系统评估NVIDIA T4与L4 GPU在FP32、FP16、INT8三种精度下对ResNet-18/50/101模型的推理性能，包括吞吐量、延迟、尾部延迟和显存占用；

**💡 创新点**

创新点在于将GDEV-AI CPU基准方法迁移至GPU推理，提供精度与批量大小对性能的统一量化，并首次在同一实验平台上客观比较两代推理优化GPU的性能差距；

**🔧 技术方法**

采用PyTorch eager模式进行FP32/FP16推理，TensorRT进行INT8量化推理，使用NVML监测显存与功耗，按批量尺寸划分并重复三次以消除波动；

**📊 数据集**

使用ResNet-18/50/101三种标准卷积网络（基于ImageNet预训练权重）作为基准工作负载；

**📈 对比分析**

对比方法为在相同模型、相同批量、相同测量流程下的吞吐量、延迟和P99延迟，结果显示L4在INT8模式下吞吐量可达CPU的58倍，T4吞吐量提升约1.9–22倍，L4在小批量下即能达到高吞吐与低延迟；

**⚠️ 局限性**

局限性包括仅在单租户、单机环境下测评，未覆盖多租户调度与真实业务混合负载，且仅使用ResNet系列模型，未评估Transformer或其他模型的表现。

---

## 59. Knowledge Graph RAG: Agentic Crawling and Graph Construction in Enterprise Documents

**arXiv ID:** 2604.14220 | [PDF](https://arxiv.org/pdf/2604.14220v1)

**作者:** Koushik Chakraborty `[一作]` (Google AI, Global Services Delivery), Koyel Guha `[通讯]` (Google AI, Global Services Delivery)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过构建 Agentic Knowledge Graph（AKG）并实现递归引用爬虫，改进了复杂企业文档生态中的检索与生成工作流程，解决了层级与时间戳冲突导致的语义搜索失效问题。

**💡 创新点**

创新点在于：① 将文档层级与时间关系抽象为 SUPERSEDES、REFERS_TO 等有向边，形成可确定性遍历的知识图谱；② 设计递归引用爬虫（Recursive Reference Crawler）作为自主代理，自动追踪引用链条并聚合完整答案；③ 将传统 RAG 与图谱检索融合，形成混合检索-生成体系。

**🔧 技术方法**

技术方法包括：图数据库构建（节点：文档/条款；边：SUPERSEDES、REFERS_TO、CONTAINS），BFS/DFS 递归遍历算法，LLM 解析未结构化文本中的引用指令，向量检索作为初始召回层，融合向量与图谱检索的混合查询管道。

**📊 数据集**

使用的数据集：① 2026 年 4 月的 Code of Federal Regulations（CFR）XML 结构化数据；② 论文示例中的三份合同文件（Base Contract、Amendment 01、Tender Addendum 03），用于演示与验证。

**📈 对比分析**

比较方法：将知识图谱增强的 RAG 与仅使用向量检索的标准 RAG 在 20 题复杂监管问答上对照。性能表现：知识图谱方法在“完整且正确答案”上达 95%，标准 RAG 仅 25%；整体准确率提升 70%。

**⚠️ 局限性**

局限性包括：① 依赖 LLM 对文本中引用的准确解析，解析错误会导致检索失败；② 仅在 CFR 领域进行了基准测试，跨领域推广需要进一步验证；③ 构建与维护大型知识图谱的计算与存储成本较高。

---

## 60. MEME-Fusion@CHiPSAL 2026: Multimodal Ablation Study of Hate Detection and Sentiment Analysis on Nepali Memes

**arXiv ID:** 2604.14218 | [PDF](https://arxiv.org/pdf/2604.14218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 61. Hierarchical vs. Flat Iteration in Shared-Weight Transformers

**arXiv ID:** 2604.14442 | [PDF](https://arxiv.org/pdf/2604.14442v1)

**作者:** Sang-Il Han `[一作]` (Korea University of Technology and Education), Sang-Il Han `[通讯]` (Korea University of Technology and Education)

**通讯引用:** 132783 | [OpenAlex ID](https://openalex.org/A5068632927)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在自回归语言模型中使用共享权重的双速递归结构（HRM-LM），通过在同一 Transformer 块上交替执行快速局部细化（Fast‑module）和慢速全局压缩（Slow‑module）来代替传统的独立层堆叠；同时还对比了平面迭代（Universal Transformer）与两速层次迭代的效果；

**💡 创新点**

创新点在于：①证明了在共享权重迭代中引入内部层次结构（Fast/Slow 两速）显著提升表示质量；②展示了共享权重在参数存储上的 O(Ld²)→O(d²) 大幅压缩；③在同等参数/ FLOPs 下，HRM‑LM 与传统 Transformer 的性能对比提供了新的实证基准；

**🔧 技术方法**

技术上使用了：Transformer 的因果自注意力与 SwiGLU FFN；递归门控（GRU‑style）与 K‑step TBPTT；参数共享与梯度累积；以及多源输出融合和熵正则化；

**📊 数据集**

数据集采用 OpenWebText（约 40GB 原始文本）进行预训练，使用 GPT‑2 tokenizer；

**📈 对比分析**

比较方法：在相同参数量（≈1.23B）和相同 FLOPs 的设置下，HRM‑LM 在 10k 迭代时达到约 4.18 nats 的验证交叉熵，而 Universal Transformer（平面迭代）停留在约 7.6 nats，差距约 3.4 nats；相对传统 Transformer L=4，HRM‑LM 在 4.23 nats 时仅损失 0.20 nats，但显著减少 2.5 GB 权重（≈一半）；此外，梯度窗口 K、监督迭代次数 S、慢速周期 T 等超参数的 ablation 进一步揭示了模型性能的敏感性；

**⚠️ 局限性**

局限性包括：①实验仅在 1.2 B 参数、OpenWebText、1 024 长度的规模下进行；②对更大规模（>10 B 参数）或更长上下文（>2 k）未验证；③HRM‑LM 的序列生成速度比普通 Transformer 慢 2–5 倍；④需要更细致的超参数搜索以排除优化器偏差；⑤在 KV‑cache 方面仅在 L> M 时才具优势，且未与线性注意力或量化技术结合；

---

## 62. Mistake gating leads to energy and memory efficient continual learning

**arXiv ID:** 2604.14336 | [PDF](https://arxiv.org/pdf/2604.14336v1)

**作者:** Aaron Pache `[一作]` (University of Nottingham), Mark CW van Rossum `[通讯]` (University of Nottingham)

**通讯引用:** 8173 | [OpenAlex ID](https://openalex.org/A5075223907)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种基于误差门控的学习规则——记忆误差门控学习，只在当前或过去出现错误时更新权重，显著减少更新次数和能耗。

**💡 创新点**

创新点在于引入了记忆机制，自动记录并再次更新曾经误分类的样本，同时不增加超参数或额外计算开销，兼顾稀疏更新与泛化能力。

**🔧 技术方法**

技术实现为在标准反向传播中加入一个布尔门控数组，使用随机梯度下降训练前馈和卷积网络，评估参数更新的L1能量与独特样本需求。

**📊 数据集**

实验数据集包括MNIST、扩展MNIST（EMNIST）和CIFAR‑10（增量学习场景），并使用数据增强版本进行对比。

**📈 对比分析**

与传统backprop相比，记忆误差门控学习在保持或接近同等测试准确率的同时，更新次数下降约20–80%，L1能量降低，所需核心样本子线性增长，显示显著的能量与内存优势。

**⚠️ 局限性**

局限性在于对从零开始学习的复杂任务效果不佳；批量训练、初始错误率高时收益减弱；并且只提供了一个抽象的巩固模型，缺乏对真实生物机制的完整验证。

---

## 63. GUI-Perturbed: Domain Randomization Reveals Systematic Brittleness in GUI Grounding Models

**arXiv ID:** 2604.14262 | [PDF](https://arxiv.org/pdf/2604.14262v1)

**作者:** Yangyue Wang `[一作]` (Fig), Pranav Guruprasad `[通讯]` (Fig)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 GUI-Perturbed 评测框架，通过对视觉场景和指令进行可控扰动，系统评估 GUI grounding 模型的鲁棒性。

**💡 创新点**

创新点在于将域随机化方法应用于 GUI 评测，独立扰动视觉与指令两条轴，揭示空间推理、视觉稳健性以及链式思维的缺陷，并提供可追踪的诊断信号。

**🔧 技术方法**

使用 Qwen2.5VL 系列的 7B 视觉‑语言模型，结合 Playwright 渲染、CSS/JS 注入、LoRA 细调等技术实现模型评估与微调。

**📊 数据集**

基于 Mind2Web MHTML 归档生成 390 条样本，并扩展为 Style、Precision（70% 缩放）、TextShrink 等四种视觉变体，配合直接与关系式指令两种类型。

**📈 对比分析**

在 16 个评估配置下，对三款 7B 模型进行测试，发现关系式指令导致 27–56pp 准确率下降，70% 缩放导致 3–8pp 降低，LoRA 细调反而进一步恶化；与标准 ScreenSpot、ScreenSpot-Pro 等基准对比显示显著性能衰退。

**⚠️ 局限性**

局限包括：仅覆盖 Web 场景、同一架构线性模型、LoRA 细调不足以解决空间推理、扰动可能不够真实、指令多样性有限。

---

## 64. NeuroTrace: Inference Provenance-Based Detection of Adversarial Examples

**arXiv ID:** 2604.14457 | [PDF](https://arxiv.org/pdf/2604.14457v1)

**作者:** Firas Ben Hmida `[一作]` (University of Michigan-Dearborn), Birhanu Eshete `[通讯]` (University of Michigan-Dearborn)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种统一框架和开放数据集，用以通过推理先行图（Inference Provenance Graphs, IPGs）捕捉深度神经网络在推理过程中的激活行为和参数驱动的数据流，进而进行对抗样本检测。

**💡 创新点**

创新点在于：①首次将推理过程视为可变的异构图结构；②设计可复现的提取引擎，将前向执行转换为IPG；③在视觉和恶意软件两大领域构建跨攻击、跨威胁模型的基准数据集，并证明IPG在攻击迁移性上的强大检测能力。

**🔧 技术方法**

使用了 PyTorch 前向钩子收集激活与权重信息、PyTorch Geometric 构建异构图、GraphSAGE/GCN/GAT 等图神经网络做图级分类器，并在多种对抗攻击（FGSM、PGD、APGD、SPSA、Square、SIA/SIT、Emb‑att、Bit‑Flip）上评估。

**📊 数据集**

使用的主要数据集包括：CIFAR‑10 上的 ResNet‑20、两款恶意软件检测器（基于 EMBER 与 Cuckoo-Traces）的全连接网络；对应的对抗样本通过 AutoAttack、SPSA、Square 等生成；IPG 数据集已公开（https://drive.google.com/file/d/1_uZE5c-dz5SnEd6S8V-8zDJed4ONdcQE）。

**📈 对比分析**

与最近的图基线 CIGA 进行对比，IPG‑based 检测在所有攻击上均达成 99%+ 的 ROC‑AUC 与 F1 分数，尤其在 PGD、APGD 等强攻击上提升显著；跨攻击、跨威胁迁移实验表明训练仅用白盒攻击即可实现 96%+ 的黑盒检测性能，验证了推理先行信息的可迁移性。

**⚠️ 局限性**

局限性包括：①未评估针对 IPG 检测器的自适应攻击；②目前仅在中等规模模型（ResNet‑20、MLP）上测试，未覆盖大型网络如 ViT/LLM；③提取与存储开销较大，适合离线审计，实时部署仍需压缩与选择性捕获；④对攻击类型覆盖有限，跨域泛化需进一步验证。

---

## 65. Reflections on Traceability for Visualization Research

**arXiv ID:** 2604.14417 | [PDF](https://arxiv.org/pdf/2604.14417v1)

**作者:** Jen Rogers `[一作]` (Idaho National Lab), Miriah Meyer `[通讯]` (Linköping University)

**通讯引用:** 4224 | [OpenAlex ID](https://openalex.org/A5008627422)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了可追溯性框架，构建并实验了tRRRacer工具，探讨记录、报告、阅读三大任务在设计导向可视化研究中的应用

**💡 创新点**

首次将材料痕迹理论与可视化研究结合，定义研究工件与研究线索概念，设计了支持记录、报告、阅读的三R工具，并强调线程作为可追溯性叙事的价值

**🔧 技术方法**

使用Electron开发桌面版和React/Web构建读者端，集成Google Drive API、深度链接、标签、注释、线程可视化及概览+细节视图

**📊 数据集**

以四个内部研究项目为测试案例，涵盖进化生物学设计研究、tRRRacer自身开发、可视化设计协作访谈及未发表设计研究，使用会议记录、访谈稿、草图、代码等多种工件

**📈 对比分析**

未进行正式性能对比，仅通过内部团队使用与反思评估，发现工具功能强大但技术复杂，记录/报告耗时，线程创建困难，匿名化与持久化功能未成熟

**⚠️ 局限性**

工具维护成本高、技术复杂且缺乏长期可持续方案，线程创建在项目中困难，实验仅在小团队内部完成，缺乏外部用户测试与读者体验评估，深度链接的持久性和可移植性未得到验证

---

## 66. Interpretable and Explainable Surrogate Modeling for Simulations: A State-of-the-Art Survey and Perspectives on Explainable AI for Decision-Making

**arXiv ID:** 2604.14240 | [PDF](https://arxiv.org/pdf/2604.14240v1)

**作者:** Pramudita Satria Palar `[一作]` (Institut Teknologi Bandung), Benoit Gaudou `[通讯]` (Université Toulouse Capitole)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并系统化了将可解释人工智能（XAI）技术应用于仿真驱动的代理模型（surrogate modeling）的方法与实践，提出了从全局到局部解释的工作流框架，并指出当前研究的主要空白与未来发展方向。

**💡 创新点**

创新点在于：① 将XAI方法与代理模型设计、验证、优化等工程流程对接，构建“解释性代理模型”完整工作流；② 以可解释性为切入点，重新定义代理模型的评价指标和选择策略；③ 提出将全局（如Sobol、GSA、ASM）与局部（如SHAP、LIME、ICE）解释方法映射到工程决策不同阶段的统一框架；④ 综述了交叉学科方法（统计学、系统工程）在代理解释中的应用。

**🔧 技术方法**

使用的技术包括：全球灵敏度分析（Sobol、Morris、DGSM、ASM）、局部特征重要性方法（SHAP、LIME、LOCO、CIU、梯度敏感度）、可视化交互技术（PDP、ICE、ALE）、模型评估与不确定性量化技术（Gaussian Process、Bootstrap、Conformal Prediction），以及元学习/AutoXAI框架来自动化模型与解释器选择。

**📊 数据集**

主要数据来源为文献与案例：引用了多篇方程式仿真（CFD、FEM）、代理模型（GP、RF、NN）以及基于代理的优化、决策研究；通过对比不同领域（航空、机械、城市系统、流体网络）的实例，展示各类XAI方法的适用场景。文章未提供统一实验数据集，而是以公开案例与公开数据集（如OpenML、SimBench）为参考。

**📈 对比分析**

评估方式为理论与案例对照：对比不同解释方法在可解释性、计算成本、可扩展性等维度的优势与限制；通过示例展示全局方法能识别主效应、局部方法揭示异常预测；并通过对比实验（如使用SHAP与LIME对同一代理模型的解释结果）说明方法的互补性。文章并未给出统一的性能指标，而是提供了定性对比与应用指导。

**⚠️ 局限性**

局限性包括：① 主要为综述性质，缺乏统一的实验验证与性能量化；② 对动态系统与混合变量的解释方法仍处于初步阶段；③ 许多XAI方法在高维工程数据中计算成本高，需进一步改进；④ 解释结果的可验证性（与物理原理的一致性）尚未形成系统化评估框架；⑤ 未来需要更多跨学科协同研究以弥补工程场景下的特定约束。

---

## 67. "I Just Don't Want My Work Being Fed Into The AI Blender": Queer Artists on Refusing and Resisting Generative AI

**arXiv ID:** 2604.14266 | [PDF](https://arxiv.org/pdf/2604.14266v1)

**作者:** Jordan Taylor `[一作]` (Carnegie Mellon University), Sarah E. Fox `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4895 | [OpenAlex ID](https://openalex.org/A5038768773)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对15名自认为酷儿艺术家的半结构式访谈进行数据收集与分析，探讨其对生成式AI的认知、抵抗方式及潜在的酷儿美学价值；

**💡 创新点**

首次系统地将José Esteban Muñoz的酷儿美学与抵抗理论应用于生成式AI的艺术语境，揭示艺术实践中的关系性与资本主义剥削冲突；

**🔧 技术方法**

采用质性研究方法——访谈记录、手工编码、主题分析，结合理论建构进行阐释；

**📊 数据集**

使用受访者提供的访谈文本（共15位艺术家）作为研究数据集；

**📈 对比分析**

未进行性能或定量对比评估，本文聚焦于经验与叙事分析；

**⚠️ 局限性**

研究范围受限于美国、在线社交媒体招募，样本量小且多为年轻白人/亚裔；缺乏对老年或黑人酷儿艺术家的视角，且未涵盖生成式AI不同交互范式的体验。

---

## 68. Coalition Formation in LLM Agent Networks: Stability Analysis and Convergence Guarantees

**arXiv ID:** 2604.14386 | [PDF](https://arxiv.org/pdf/2604.14386v1)

**作者:** Dongxin Guo `[一作]` (University of Hong Kong), Siu-Ming Yiu `[通讯]` (University of Hong Kong)

**通讯引用:** 22411 | [OpenAlex ID](https://openalex.org/A5110500992)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于博弈论的 LLM 联盟形成框架（LCFG），并设计了 CoalT（Coalition-of-Thought）提示协议，验证其在多代理协作任务中的稳定性与性能。

**💡 创新点**

创新点在于首次将博弈论中的和谐游戏（hedonic game）应用于 LLM 代理，给出存在性与一致性驱动的稳定性下界，并提出结构化提示协议显著提升一致性与稳定率。

**🔧 技术方法**

使用了和谐游戏理论、ε-理性与 logit 动态分析、能力特征向量建模以及分步结构化提示（CoalT）等技术。

**📊 数据集**

通过在 MATH、MMLU、LogiQA 三个子集上自定义评估，构建了 200 题协作问答数据集，并对 6 名 GPT‑4、Claude‑3 与 Llama‑3 代理进行能力特征估计。

**📈 对比分析**

与随机、贪心、标准、Vanilla CoT、Self‑Consistency 等基线相比，CoalT 在 2400 条实验中稳定率达 73.2%（比标准 41.8% 提升 31.4pp），且在混合架构队伍中实现了更高的社会福利。

**⚠️ 局限性**

局限性包括：需要高一致性（p≥0.8）才能保证稳定，理论与实验主要在 6 名代理规模上验证，扩展到更大规模或动态能力环境需进一步研究；假设代理诚实报告偏好，缺乏机制设计与激励兼容性分析。

---

## 69. Compressed-Sensing-Guided, Inference-Aware Structured Reduction for Large Language Models

**arXiv ID:** 2604.14156 | [PDF](https://arxiv.org/pdf/2604.14156v1)

**作者:** Andrew Kiruluta `[一作]` `[通讯]` (UC Berkeley), Andrew Kiruluta (UC Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种压缩感知指导下的动态大模型推理框架，结合任务条件测量、令牌自适应恢复、硬件可编译结构稀疏化和提示压缩，实现按需子网络执行。

**💡 创新点**

创新点包括：1）任务条件测量与令牌自适应恢复；2）样本复杂度分析与恢复保证；3）硬件约束下的可编译结构稀疏；4）提示压缩与模型压缩在同一压缩感知目标内联合优化；5）不确定性驱动的自适应测量。

**🔧 技术方法**

技术：压缩感知理论、稀疏恢复优化、随机测量矩阵、硬件可编译稀疏结构、预测熵自适应控制、离线编译稀疏核、提示压缩算法。

**📊 数据集**

使用多任务大语言模型基准数据集（如summarization、code generation、long‑context retrieval、math reasoning、dialogue）以及标准LLM评测集。

**📈 对比分析**

与 SparseGPT、Wanda、ZipLM、LLMLingua、LongLLMLingua、CATS/TEAL 等现有剪枝、提示压缩、激活稀疏方法在同一硬件平台下比较，实验显示在保持质量近似 dense 的同时，能显著提升推理速度并降低内存占用。

**⚠️ 局限性**

局限：恢复误差依赖字典表达能力；测量与恢复开销需低于稀疏执行收益；压缩感知假设（RIP、互不相干）在自适应测量上仅近似成立；硬件可编译约束导致搜索空间复杂；闭环不确定性控制可能产生振荡；需频繁重新校准以适应模型更新。

---

## 70. ToxiShield: Promoting Inclusive Developer Communication through Real-Time Toxicity Filtering

**arXiv ID:** 2604.14408 | [PDF](https://arxiv.org/pdf/2604.14408v1)

**作者:** MD Awsaf Alam Anindya `[一作]`, Amiangshu Bosu `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 ToxiShield，一款实时浏览器插件，能在 GitHub Pull Request 代码评审过程中检测、解释并重写有毒评论，提升团队沟通质量。

**💡 创新点**

创新点包括：①三阶段模块化（检测、教学式分类、重写）实现即时干预；②在检测后提供可解释的子类标签与改写理由，促进行为改变；③基于自建的“毒性→中性”对齐数据集，使用 LoRA 与 prompt‑tuning 的轻量化 LLM，支持离线部署。

**🔧 技术方法**

关键技术：BERT‑base‑uncased（二分类过滤）；Claude 3.5 Sonnet（多标签子类分类，链式推理与 XML 输出）；Llama 3.2 3B（通过教师‑学生蒸馏微调实现毒性消解）；ONNX/量化实现低延迟；LoRA 进行参数高效微调。

**📊 数据集**

使用数据集：38 761 条 PR 评论（10 120 条毒性，28 641 条非毒性）用于过滤训练；1 200 条多类标注（11 种毒性子类 + 非毒性）用于分类；10 120 条毒性文本与教师 LLM 生成的 10 120 条中性文本组成的平行语料用于重写训练。

**📈 对比分析**

对比结果：过滤器 BERT‑base 取得 98 % 准确率、0.97 F1；分类器 Claude 3.5 Sonnet 在宏 F1 0.42、MCC 0.39；重写器 Llama 3.2 3B 获得 J‑Score 84 %，Style 95.27 %，Fluency 97.03 %，Content 67.07 %。相较于 GPT‑4o‑mini 等大型模型，轻量化 LLM 在准确率与推理速度上更具优势。

**⚠️ 局限性**

局限性：①数据规模和类别不平衡导致极稀有子类识别仍弱；②对讽刺、被动攻击等细微毒性难以捕捉；③模型在不同代码风格或非 GitHub 场景的迁移性未知；④用户评估仅 10 名开发者，实验规模有限；⑤离线模型仍有一定延迟与错误率，需进一步优化。

---

## 71. Response-Aware User Memory Selection for LLM Personalization

**arXiv ID:** 2604.14473 | [PDF](https://arxiv.org/pdf/2604.14473v1)

**作者:** Jillian Fisher `[一作]` (University of Washington), Chan Young Park `[通讯]` (Microsoft Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于响应信息的用户记忆子集选择方法 RUMS，用于在 LLM 推理时只注入最具信息量的用户记忆，从而实现高效、可解释的个性化。

**💡 创新点**

创新点包括：① 以互信息/条件熵减为目标的用户记忆选择函数——将记忆对模型输出不确定性的影响量化为可计算的效用；② 通过理论证明将熵减与期望用户效用关联；③ 在离线阶段利用该函数生成训练样本，再训练轻量级 DeBERTa 分类器实现在线高效选择；④ 显著降低 95% 的额外推理成本。

**🔧 技术方法**

主要技术：信息理论（互信息、条件熵）、token‑级熵分解与 Monte Carlo 估计、DeBERTa‑v3‑large 轻量级分类器、LLM（LLaMA‑70B、GPT‑4）推理、基于 GPT‑4 的人类对齐评估。

**📊 数据集**

数据集：① 合成数据（LLM 生成用户查询与随机用户记忆，保证所有输入都受个性化影响）；② Trivia 数据（问答式问题）；③ WildChat（真实 ChatGPT 对话，人工标注个性化与否）。另外为每个用户合成 50 条记忆属性，共 100 份用户配置。

**📈 对比分析**

比较方法：语义相似度检索、GPT‑4 零/少量提示、BM25、ReContriever 以及无记忆/全记忆/随机基线。RUMS 在 H1‑H3 的三项实验中：① 对个性化需求的识别准确率高于 GPT‑4 及语义相似度；② 记忆子集与人工标注的 precision/recall/F1 约高 20‑30%；③ 对生成质量（GPT‑4/ LLaMA‑70B win‑rate）提升 12‑18%，且额外推理成本下降 95%。

**⚠️ 局限性**

局限性：① 依赖模型的输出分布与人类偏好高度对齐；② 需要假设存在足够的潜在变量来解释记忆效用，理论推导在实际中为近似；③ 仅针对单轮查询，未考虑多轮对话中的记忆演化；④ 训练过程需离线生成大量熵估计样本，成本不低；⑤ 目前仅在 50 条记忆属性上验证，扩展到更大、更稀疏的用户档案仍待研究。

---

## 72. HUOZIIME: An On-Device LLM-enhanced Input Method for Deep Personalization

**arXiv ID:** 2604.14159 | [PDF](https://arxiv.org/pdf/2604.14159v1)

**作者:** Baocai Shan `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8872 | [OpenAlex ID](https://openalex.org/A5019108029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于轻量化大型语言模型（LLM）的全新中文输入法（HuoziIME），实现了在设备端的记忆增强式生成、个性化样式控制以及实时候选展示；

**💡 创新点**

①首次实现完全在设备端运行的记忆增强式生成式中文输入法；②采用层级记忆架构（L1/L2/L3）与GRPO记忆触发器，实现高效上下文检索与持续学习；③结合LLM后期风格化训练、KV-Splice与PIC等技术，确保毫秒级响应与低内存占用；

**🔧 技术方法**

LLM风格化后训练、三层层级记忆（KV缓存、HNSW向量索引、参数化记忆）、GRPO记忆触发、RadixTree KV复用、KV-Splice拼接、PIC位置无关缓存、线程亲和调度、跨进程Model Context Protocol (MCP) 等；

**📊 数据集**

利用网络收集的海量对话语料并人工增广/筛选得到的合成个性化语料；用于风格化训练的数据被划分为训练/测试集；检索实验采用200条手工标注的记忆样本；

**📈 对比分析**

在MediaTek Dimensity 9000手机上评估：记忆提取准确率96.4%，拒绝命令准确率71.3%；检索@4命中率89.5%，基于检索的生成成功率87.2%；推理方面，prefill吞吐260 token/s，解码吞吐24–25 token/s，首候选时间800–1700 ms；峰值内存约1.12 GB；用户实验显示键入摩擦显著降低，KSR提升；

**⚠️ 局限性**

受限于轻量化模型，推理能力有限，易出现过度检索、检索令牌漂移；外部应用上下文获取受Android沙盒限制；若设备被破坏，本地存储的用户轨迹可能面临安全风险；

---

## 73. FRESCO: Benchmarking and Optimizing Re-rankers for Evolving Semantic Conflict in Retrieval-Augmented Generation

**arXiv ID:** 2604.14227 | [PDF](https://arxiv.org/pdf/2604.14227v1)

**作者:** Sohyun An `[一作]` (Meta Superintelligence Labs), Alexander Min `[通讯]` (Meta Superintelligence Labs)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个专门评估重排序器在时间演变冲突下性能的基准，并研究了基于 Pareto 的指令优化框架。

**💡 创新点**

创新点在于：①构造了仅语义相关但含时间冲突的候选集；②揭示重排序器普遍偏向过时信息；③提出可调的多目标指令优化方法。

**🔧 技术方法**

采用 LLM 重排序器、演化算法进行指令搜索、语义嵌入（Qwen3‑Embedding‑0.6B）以及注意力分析。

**📊 数据集**

使用维基百科修订历史与 Wikidata 事实构造的 ∼ 数据集（3658 个查询 + 50 负样本），并与改写的 NQ 数据集做对比。

**📈 对比分析**

在 19 个现有重排序器上评估 MAP/MRR/Recall，发现 Obsolete Ratio 高达 84–98%，而指令优化后 EK 任务 MAP 提升至 79.2%，NEK 仍保持 68.9%。

**⚠️ 局限性**

局限性包括：仅覆盖维基百科/ Wikidata 的结构化变化；假设文档时间戳可知；仅在指令层面优化，未处理无时间戳或隐式时间的情形。

---

## 74. BIEVR-LIO: Robust LiDAR-Inertial Odometry through Bump-Image-Enhanced Voxel Maps

**arXiv ID:** 2604.14421 | [PDF](https://arxiv.org/pdf/2604.14421v1)

**作者:** Patrick Pfreundschuh `[一作]` (ETH Zuerich), Helen Oleynikova `[通讯]` (ETH Zuerich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了BIEVR-LIO，一种结合高分辨率基于体素的“凸起图像”地图与地图感知双分辨率点采样的激光雷达-惯性里程计框架。

**💡 创新点**

主要创新点包括：①在每个体素内以面为基准存储细粒度高度图（凸起图像），从而无需中间几何原语即可捕捉微小几何变化；②基于凸起图像的显著性度量（MID）实现地图感知的双分辨率点采样，重点提取信息丰富的体素，提升鲁棒性并降低计算量。

**🔧 技术方法**

采用了体素哈希（Morton编码）并行更新、IMU预积分用于运动补偿和初始化、松耦合姿态估计（仅用扫描对地图的匹配）以及基于高度误差的点对高度图残差优化（Levenberg‑Marquardt），并通过滑动窗口惯性一致性约束优化加速度计/陀螺仪偏置和重力方向。

**📊 数据集**

在 Newer College、ENWIDE、GEODE、MARS‑LVIG、GrandTour 等多种公开数据集（包含手持、UGV、UAV、四足等平台及不同 LiDAR/IMU 组合）上进行实验。

**📈 对比分析**

与 KISS‑ICP、GenZ‑ICP、Traj‑LO、FAST‑LIO2、DLIO、iG‑LIO、RESPLE、RKO‑LIO、COIN‑LIO、CURL‑SLAM 等前沿 LIO 方法进行对比。BIEVR‑LIO 在几何信息稀缺的隧道、平坦草原等极端场景中保持收敛且误差显著低于对手；在结构丰富场景下亦能达到或超越 state‑of‑the‑art 的绝对/相对误差，同时通过高效采样降低了点数与运行时。

**⚠️ 局限性**

局限性：需要足够密集的点云才能填充高度图，低分辨率 LiDAR 或高速运动时可能导致像素缺失；依赖较准确的初始位姿；未利用 LiDAR 强度信息，纯几何退化环境下仍可能失效；未显式检测或处理几何退化与动态物体，影响部分后续任务。

---

## 75. Formalizing Kantian Ethics: Formula of the Universal Law Logic (FULL)

**arXiv ID:** 2604.14254 | [PDF](https://arxiv.org/pdf/2604.14254v1)

**作者:** Taylor Olson `[一作]` `[通讯]` (University of Iowa), Taylor Olson (University of Iowa)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出一种多排序量化模态逻辑ℱ𝒰ℒℒ，用以形式化康德第一条命令——普遍法则（FUL），并通过该逻辑实现对人工道德代理（AMA）的自主动机评估。

**💡 创新点**

创新点在于：①将道德评估从单纯的行动级别提升到代理目的级别；②不需要预先编码完整的道德直觉，仅依赖非规范的背景知识；③通过形式化的因果与意志推导，实现对康德义务的完美与不完美区分。

**🔧 技术方法**

使用多排序量化模态逻辑、自然演绎证明体系、因果与意志的形式化表达以及普遍化变换UL来实现对最大化表达的形式化与道德判定。

**📊 数据集**

无实验数据集，主要通过对经典康德伦理案例（虚假承诺、谋杀、拒绝帮助等）的形式化证明来验证方法。

**📈 对比分析**

未进行实验对比与性能评估；通过形式化推理示例展示该逻辑在康德伦理场景下能够正确产生可许可/不可许可的判定。

**⚠️ 局限性**

局限性包括：仍需人工给定最大化表达的形式化；对概念描述的精确性依赖；未实现模型论语义或广义量词；以及对复杂现实情境的可扩展性待进一步研究。

---

## 76. Shapley Value-Guided Adaptive Ensemble Learning for Explainable Financial Fraud Detection with U.S. Regulatory Compliance Validation

**arXiv ID:** 2604.14231 | [PDF](https://arxiv.org/pdf/2604.14231v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 77. Fast Concurrent Primitives Despite Contention

**arXiv ID:** 2604.14530 | [PDF](https://arxiv.org/pdf/2604.14530v1)

**作者:** Michael A. Bender `[一作]` (Stony Brook University), Renfei Zhou `[通讯]` (Carnegie Mellon University)

**通讯引用:** 188 | [OpenAlex ID](https://openalex.org/A5088368179)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种在存在写冲突的共享内存环境下，对读写寄存器和 CAS 寄存器进行冲突解决的算法，并证明这些实现在线性化、无阻塞（或等待自由）的前提下，在随机并发调度器（CRQW 模型）下每个操作的高概率延迟为 O(log P)。基于这些基本原语，作者给出了组合定理，能把传统无冲突模型下的算法直接转化为冲突友好的实现，并进一步得到 LL/SC、fetch‑and‑increment、有限计数器等常用同步原语。最后给出了一条空间-延迟下界，证明在某些条件下无法在高概率下实现常数期望延迟。

**💡 创新点**

创新点主要有三点：
1. 将冲突解决从算法设计层面剥离，给出可直接替代硬件原语的冲突友好实现；
2. 在随机并发调度器下证明了 O(log P) 的高概率延迟上界，并通过概率与潜能分析得到完整的证明；
3. 给出空间-延迟下界，揭示高概率下常数期望延迟的不可行性，为后续研究提供理论限制。

**🔧 技术方法**

核心技术包括：
- 随机指纹（fingerprint）策略，用于快速检测并放弃冲突操作；
- 随机退避与指数增长的调用概率；
- 降维的“简化状态机”表示，便于对共享指令的延迟进行集中分析；
- 采用随机调度器（τ‑随机延迟属性）和自适应对手模型，保证对手在观察历史的前提下仍无法破坏高概率延迟；
- 通过势能函数和 Chernoff/Azuma 等概率工具证明潜能控制与忙区间长度。

**📊 数据集**

实验部分使用了四台 Intel Xeon E7‑8867 v4 服务器（共 72 个物理核、144 SMT 线程）与 DDR4‑2400 内存。每个线程执行 10⁸ 次操作，操作类型包括仅加载、仅存储、加载/修改/存储、加载/修改/CAS，分别测量在 1–144 线程规模下的运行时间。实验没有使用公开数据集，而是针对典型并发工作负载进行实验。

**📈 对比分析**

与传统（无冲突）实现相比，实验结果显示：
- 读取操作在高并发下几乎线性缩放，说明读取冲突影响小；
- 包含写/ CAS 的操作随着线程数线性下降，验证 CRQW 模型对写冲突的预测。理论上作者给出的 O(log P) 高概率延迟显著优于 naïve 实现的 Θ(P) 延迟，实验也表明在实际硬件上冲突友好实现能显著降低延迟。

**⚠️ 局限性**

局限与待改进之处：
- 下界仅在随机并发调度器下给出，对更强随机或确定性调度器的界限仍未知；
- 只提供高概率延迟保证，无法给出绝对 worst‑case 或期望常数延迟的上界；
- 指纹长度需要 Ω(log P)，在极低延迟要求下可能不可行；
- 实验只测量了基本读写/CAS 操作，未覆盖所有组合原语或大规模数据结构的实际性能；
- 模型假设冲突仅来自写操作，未考虑更复杂的内存访问模式（如多级缓存、非统一内存）和上下文切换等实际系统因素。

---

## 78. Quantum-inspired tensor networks in machine learning models

**arXiv ID:** 2604.14287 | [PDF](https://arxiv.org/pdf/2604.14287v1)

**作者:** Guillermo Valverde `[一作]` (Vicomtech Foundation), Alejandro Pozas-Kerstjens `[通讯]` (University of Geneva)

**通讯引用:** 920 | [OpenAlex ID](https://openalex.org/A5080010683)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

综述了张量网络在机器学习中的应用与发展，重点区分原生张量网络与张量化神经网络。

**💡 创新点**

提出了统一的批判性评估框架和交互式文献地图，以梳理当前研究空白和技术瓶颈。

**🔧 技术方法**

利用矩阵乘积态、矩阵乘积算子、PEPS、树状张量网络、MERA以及CP/Tucker/TT分解等技术，并借助SVD、Gauge自由、canonical形式等方法。

**📊 数据集**

主要参考 MNIST、Fashion‑MNIST 等图像数据集及视频、语音等多模态数据，但本文为综述无直接实验。

**📈 对比分析**

与传统深度网络对比，张量网络在参数量、计算效率和可解释性方面表现优异，但在大规模循环网络和高维数据上仍略逊。

**⚠️ 局限性**

局限在于循环网络的收缩复杂度高、训练难度大、缺乏大规模实证，且对超参数敏感。

---

## 79. Aerial Multi-Functional RIS in Fluid Antennas-Aided Full-Duplex Networks: A Self-Optimized Hybrid Deep Reinforcement Learning Approach

**arXiv ID:** 2604.14309 | [PDF](https://arxiv.org/pdf/2604.14309v1)

**作者:** Li-Hsiang Shen `[一作]`, Yu-Quan Zheng `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种集成自主航空平台、可变形天线与多功能可重构智能表面（AM‑RIS）的全双工网络架构，并通过联合优化天线位置、RIS配置、发射功率等来最大化系统能效。

**💡 创新点**

创新点包括：
1) 在无人机上部署可实现反射、放大与能量收集的多功能RIS；
2) 在基站采用可变形天线实现细粒度空间可重配置，进一步抑制自干扰；
3) 设计自适应混合深度强化学习框架（SOHRL），结合多智能体DQN与PPO，利用注意力机制提取关键状态信息，并通过元学习方式自动调优超参数。

**🔧 技术方法**

核心技术：
- 混合离散/连续动作的多智能体DQN+PPO深度强化学习；
- 采用多头自注意力机制对状态进行加权；
- 元强化学习（Meta‑PPO）实现超参数自调优；
- 机动无人机、可变形天线、RIS/AM‑RIS 的物理层建模；
- 全双工系统中的自干扰与用户间干扰建模。

**📊 数据集**

数据集：使用仿真环境生成的合成信道与网络参数（如基站、用户、无人机位置、RIS元素数等），无真实数据集。通过 Monte‑Carlo 仿真评估性能。

**📈 对比分析**

比较方法：将 SOHRL 与以下基线进行对比——SOHRL 无注意力机制、混合 DRL（单一网络）、MAPPO、集中式 PPO 与集中式 DQN。结果显示，SOHRL 在能效方面达到约 0.7–0.84 bits/Hz/J，明显优于所有基线，验证了注意力机制和超参数自适应的有效性。

**⚠️ 局限性**

局限性：
- 计算复杂度高，训练与推理均需 GPU 资源；
- 依赖大量仿真参数，实际部署时对环境建模误差敏感；
- 对于大规模用户和多 UAV 的扩展性尚未深入研究；
- 假设可观测完整的链路状态，现实中测量误差与时延会影响学习效果。

---

## 80. Head Count: Privacy-Preserving Face-Based Crowd Monitoring

**arXiv ID:** 2604.14250 | [PDF](https://arxiv.org/pdf/2604.14250v1)

**作者:** Fatemeh Marzani `[一作]` (University of Twente), Maarten van Steen `[通讯]` (University of Twente)

**通讯引用:** 11045 | [OpenAlex ID](https://openalex.org/A5027678871)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于人脸识别的隐私保护人群计数系统，利用模糊提取器生成的唯一标识符并存入同态加密的布隆过滤器，以实现不同位置或时刻的人数统计而不泄露身份。

**💡 创新点**

创新点在于将模糊提取器与同态加密布隆过滤器结合，实现了在加密数据上直接进行集合成员测试，从而在保持身份隐私的前提下实现人脸计数。

**🔧 技术方法**

使用技术包括人脸检测与特征提取、模糊提取器、同态加密算法、布隆过滤器以及加密数据上的集合成员检测。

**📊 数据集**

本文未给出具体使用的数据集，实验基于作者自建或公开人脸数据库进行初步验证。

**📈 对比分析**

通过初步评估显示，该方法在保密性的同时能够保持较高的计数准确率，但具体性能指标（如误判率、处理时延等）未在文中详细量化。

**⚠️ 局限性**

主要局限包括同态加密和布隆过滤器的计算开销、误判率问题，以及实验规模和数据集的局限性，未来需要进一步优化效率并验证在大规模真实环境中的表现。

---

## 81. Weak-DMD: A Galerkin approach to the problem of noise in the Dynamic Mode Decomposition algorithm

**arXiv ID:** 2604.14350 | [PDF](https://arxiv.org/pdf/2604.14350v1)

**作者:** William Bennett `[一作]` (Los Alamos National Laboratory), Melek Derman `[通讯]` (Oregon State University)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5046935760)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一种弱形式的动态模式分解（weak‑DMD），用于处理非等时步、噪声数据，并在核反应堆临界性和二维圆柱湍流流场上进行实验验证。

**💡 创新点**

创新点在于将DMD转化为弱形式Galerkin投影，消除了对等时步的限制并通过时间积分投影天然抑制噪声，同时基于试验与测试基函数而非等间隔采样构造模型。

**🔧 技术方法**

主要技术包括弱形式Galerkin投影、最小二乘与奇异值分解、优化‑DMD对标、Monte Carlo模拟与数值流场数据处理。

**📊 数据集**

所用数据集包括12组能量核临界球面模拟、Kornreich‑Parsons多区试验炉的Monte Carlo轨迹、以及二维圆柱湍流流场（Re 300–1000）。

**📈 对比分析**

通过与优化‑DMD、VDMD等方法比较，利用特征值误差、误差收敛曲线、预测误差和计算时间等指标评估性能；weak‑DMD在噪声环境下精度相当或更优，但计算时间略高。

**⚠️ 局限性**

主要局限包括手工基函数选择、缺乏对非线性系统的理论分析、收敛性未证明、预测步骤效率低，以及对不同噪声类型适应性待进一步改进。

---

## 82. Tight Sample Complexity Bounds for Best-Arm Identification Under Bounded Systematic Bias

**arXiv ID:** 2604.14345 | [PDF](https://arxiv.org/pdf/2604.14345v1)

**作者:** Tianhao Qian `[一作]` (Southeast University), Tianhao Qian `[通讯]` (Southeast University)

**通讯引用:** 2472 | [OpenAlex ID](https://openalex.org/A5017748291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于局部最佳臂识别（BAI）的安全剪枝框架（PAC-MCTS），将节点扩展视为在动态前沿上的带有系统性偏差 L 的 BAI 问题，并给出安全剪枝的样本复杂度上界 𝒪((Δ-4L)⁻²)。

**💡 创新点**

创新点在于：1）首次将 MCTS 节点剪枝转化为带偏差的局部 BAI；2）给出了严格的安全条件 Δ>4L 及相应的样本复杂度与信息论下界；3）设计了自适应置信半径 u_dist(n)=√(2σ²ln(π²n²|A_t|/3δ)/n)+L 的 PAC-MCTS 算法，能在存在系统性偏差的情况下保证以 1‑δ 的概率保留最优节点；4）证明了在偏差超过阈值时的“优雅降级”性质。

**🔧 技术方法**

主要技术包括：
- Best-Arm Identification（BAI）理论与置信区间扩展；
- Lambert W 变换推导样本复杂度；
- 信息论下界与逆问题的变形；
- 基于联合界的自适应置信半径；
- PAC‑MCTS（MCTS + PAC 剪枝）算法；
- 对抗性偏差模拟（Top‑K 诱导模型）。

**📊 数据集**

实验数据集与环境：
- 30/50/200 抽象 BAI 树（synthetic）；
- 复杂 reasoning 任务（Game of Amazons、TSP‑50、Blocksworld、ALFWorld）；
- 对抗性偏差实验（Top‑K 诱导模型、随机误差 σ∈{0.2,0.3,0.4}）。

**📈 对比分析**

与 Naïve Pruning、标准 UCT、Tree of Thoughts（ToT）等基线对比：
- 在安全区间 Δ>4L 内，PAC‑MCTS 以 100% 的 PCS（0.98–1.00）保持安全，且在低噪声/低偏差环境下可实现 7× 以上的样本分配效率；
- 当偏差接近/超过阈值时，PAC‑MCTS 的剪枝率急剧下降至 0，避免了最优节点被误剪，同时 PCS 仍保持高于基线（≈0.98）；
- 在极端偏差（Δ≤4L）时，所有方法均趋向信息论下界，性能急剧下降，PAC‑MCTS 通过优雅降级保证子最优性不超过理论上限。

**⚠️ 局限性**

局限性：
- 需要预先估计全局或局部的偏差上界 L，若 L 估计不足或过大会导致过度保守；
- 在前沿尺寸快速膨胀或预算极度紧张时，置信半径无法收敛到有效间隙，导致剪枝停止；
- 对极大偏差（Δ≤4L）无法恢复最优解，仍受信息论下界限制；
- 该方法依赖于噪声为 sub‑Gaussian 的假设，若评估模型产生重尾或偏态误差，理论保证可能失效。

---

## 83. The Devil Is in Gradient Entanglement: Energy-Aware Gradient Coordinator for Robust Generalized Category Discovery

**arXiv ID:** 2604.14176 | [PDF](https://arxiv.org/pdf/2604.14176v1)

**作者:** Haiyang Zheng `[一作]` (University of Trento), Zhun Zhong `[通讯]` (Hefei University of Technology)

**通讯引用:** 10762 | [OpenAlex ID](https://openalex.org/A5065328976)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在泛化类别发现（GCD）任务中提出了能量感知梯度协调器（EAGC），通过梯度级别的方式显式调节优化过程，解决梯度纠缠问题；

**💡 创新点**

创新点在于①引入Anchor‑based Gradient Alignment（AGA）通过参考模型对标记样本梯度进行对齐，保持已知类别的判别结构；②提出Energy‑aware Elastic Projection（EEP）在未知类别梯度上做软投影，并根据能量比例自适应调节投影强度，既降低已知与未知类别的子空间重叠，又不压制已知样本；

**🔧 技术方法**

使用的技术包括梯度对齐与投影、Conceptor理论构建软子空间、能量感知自适应权重、基于梯度的正则化；

**📊 数据集**

在CIFAR‑100、ImageNet‑100以及细粒度三大基准（CUB‑200、Stanford Cars、FGVC‑Aircraft）上进行实验；

**📈 对比分析**

与多种现有GCD基线（SimGCD、LegoGCD、SPTNet、SelEx等）以及多种先进方法（ORCA、GPC、XCon、PromptCAL等）进行对比，EAGC在所有基准上平均提升All ACC 3.2%/New ACC 4.3%，在细粒度数据上最高可提升All ACC 8.5%/New ACC 11.4%，达到或逼近最新state‑of‑the‑art；

**⚠️ 局限性**

主要限制是对超参数（λ_a, λ_p, η）的依赖，需要在不同数据集上进行调优；此外，方法仍需验证在更大规模或跨域场景中的鲁棒性。

---

## 84. Purging the Gray Zone: Latent-Geometric Denoising for Precise Knowledge Boundary Awareness

**arXiv ID:** 2604.14324 | [PDF](https://arxiv.org/pdf/2604.14324v1)

**作者:** Hao An `[一作]` (Southern University of Science and Technology), Yang Xu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 22999 | [OpenAlex ID](https://openalex.org/A5100779940)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Geometric Denoising框架，利用LLM隐藏空间的几何距离过滤噪声样本，对模型进行放弃式微调，以减少幻觉。

**💡 创新点**

创新点在于通过分析隐藏状态的“灰区”，用线性探测器的超平面距离作为置信度指标，进行几何去噪，并与传统基于准确率划分方法对比，显著降低标签噪声。

**🔧 技术方法**

采用线性探测器（逻辑回归）在隐藏状态上训练，计算几何距离进行样本筛选，随后用交叉熵损失进行微调，并使用TBG/SLT两种隐藏状态提取方式。

**📊 数据集**

使用的主要数据集包括TriviaQA、Natural Questions、SciQ、SimpleQA、RAG‑Bench、Alcuna、FalseQA、Self‑Aware等，覆盖标准QA、RAG及未回答/欺骗性查询场景。

**📈 对比分析**

与IDK、Uncertainty、R‑Tuning、Probe‑Tuning等基线对比，在F1_ans、F1_abs、F1_rel等指标上均实现显著提升，尤其在OOD、RAG及欺骗性查询上的可靠性最佳。

**⚠️ 局限性**

局限性：仅在1.7B–8B规模模型验证，线性探测器可能不足以捕捉更大模型的真值表示；方法仍可能导致过度放弃；未扩展到长文本推理或更复杂的生成任务。

---

## 85. Evaluation of Agents under Simulated AI Marketplace Dynamics

**arXiv ID:** 2604.14256 | [PDF](https://arxiv.org/pdf/2604.14256v1)

**作者:** To Eun Kim `[一作]` (Carnegie Mellon University), Fernando Diaz `[通讯]` (Carnegie Mellon University)

**通讯引用:** 9593 | [OpenAlex ID](https://openalex.org/A5101492251)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了基于市场的评估范式 Marketplace Evaluation，用仿真模拟信息访问系统（IA 代理）在竞争性生态中的动态行为，构建了可重复的实验框架，并给出了研究议程与开源工具。

**💡 创新点**

创新点在于：①首次将 IA 代理的评估从孤立的离线指标迁移到市场层面的长期交互与竞争；②引入了市场治理图、用户/代理的策略更新机制以及一套包括市场份额、保留率、HHI 等新型指标；③提供了从单一市场到多市场耦合的实验范式，并公开了可复现的 Python 仿真包。

**🔧 技术方法**

技术包括：基于 agent‑based simulation 的离线仿真、治理图（有向无环图）建模、贝叶斯/Softmax 选择模型、动态更新规则（如基于正确率的用户偏好更新）、多阶段窗口统计、HHI 及公平性评估。

**📊 数据集**

数据集主要使用公开 QA benchmark SimpleQA（500 事实检索问题）作为交互任务，另外在论文中也提到了 TREC、CLEF 等传统任务作为未来扩展方向。

**📈 对比分析**

比较方法：将传统静态评估（单次问答准确率）与仿真市场评估（市场份额、保留率、HHI）对比。实验结果表明：静态排名与市场排名不一致，市场动态能揭示早期进入优势、赢家通吃和集中度变化；在弱集中市场中强模型 Qwen3 能快速提升份额，而在高度集中市场中同一模型表现有限。

**⚠️ 局限性**

局限性：①仿真模型依赖假设（如用户偏好更新、成本/延迟权重），缺乏真实用户交互数据；②只在单一任务（问答）上验证，未覆盖更复杂的检索‑生成耦合；③新指标虽丰富但尚未在真实部署中验证其预测能力；④仿真规模受计算资源限制，难以覆盖极大规模的多市场生态。

---

## 86. An Underexplored Frontier: Large Language Models for Rare Disease Patient Education and Communication -- A scoping review

**arXiv ID:** 2604.14179 | [PDF](https://arxiv.org/pdf/2604.14179v1)

**作者:** Zaifu Zhan `[一作]` (University of Minnesota), Rui Zhang `[通讯]` (University of Minnesota)

**通讯引用:** 12017 | [OpenAlex ID](https://openalex.org/A5100422092)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述2022‑2026年12篇研究，探讨大语言模型在罕见病患者教育与沟通中的应用

**💡 创新点**

首次系统梳理该领域现状，指出多项研究缺口并提出未来方向

**🔧 技术方法**

使用大语言模型（如ChatGPT、Gemini等）和传统评估指标

**📊 数据集**

基于公开论文中的患者问题集、FAQ和少量真实对话数据

**📈 对比分析**

对比模型回答的准确率与专业评估，一般准确率在70‑100%，但缺乏多轮对话或实时评测

**⚠️ 局限性**

研究多集中于英文、静态问答、缺少患者参与、少有领域适配和多语言支持，易产生幻觉和信息不完整

---

## 87. Graph-Based ECO and Patch Generation for High-Level Synthesis

**arXiv ID:** 2604.14248 | [PDF](https://arxiv.org/pdf/2604.14248v1)

**作者:** Alireza Azadi `[一作]` (University of New Brunswick), Kenneth B. Kent `[通讯]` (University of New Brunswick)

**通讯引用:** 2075 | [OpenAlex ID](https://openalex.org/A5067605823)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一套基于图编辑距离（GED）的工程变更（ECO）流程，针对Google XLS高层合成工具实现IR级别的差分、补丁生成与应用，并在补丁过程中保持语义正确性和原始调度一致性。

**💡 创新点**

创新点在于将GED方法与XLS IR特定成本函数相结合，实现了细粒度节点/边级别的差分与补丁，同时通过伪节点与字典映射保证了IR结构完整性，并在保持管线时钟约束的前提下实现了高达92%的调度保留率。

**🔧 技术方法**

主要技术包括：Graph Edit Distance（DF‑GED）算法、NetworkX多有向图IR解析、定制成本函数与差分、补丁应用机制（含伪节点与字典）、调度约束约束与重调度、功能等价性验证（Z3和Cadence Conformal）。

**📊 数据集**

使用七个XLS开源设计（CRC32、ZSTD帧解码器、ApFloat MAC、Simple RISC‑V、FIR Filter、Histogram、Vector Core）以及自定义的Histogram和Vector Core案例作为实验数据集。

**📈 对比分析**

与传统基于字符串或操作级别的ECO方法相比，该方法在结构重用率上可达95%，调度保留率可达92%，补丁生成时间在小型设计低于一分钟、最大设计（Vector Core）约1小时，内存峰值随设计规模呈非线性增长，验证通过Cadence Conformal确认功能正确性。

**⚠️ 局限性**

主要限制包括：单线程GED实现导致大型设计（如Histogram、Vector Core）运行时间较长且内存占用高；对状态机丰富的proc‑based设计的IR级别等价性验证受限；缺乏自动化流水线集成，需要手动操作差分输出到补丁器。

---

## 88. Credo: Declarative Control of LLM Pipelines via Beliefs and Policies

**arXiv ID:** 2604.14401 | [PDF](https://arxiv.org/pdf/2604.14401v1)

**作者:** Duo Lu `[一作]` (Brown University), Uğur Çetintemel `[通讯]` (Brown University)

**通讯引用:** 6002 | [OpenAlex ID](https://openalex.org/A5109862110)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现Credo框架，通过声明式信念与策略控制LLM管道的自适应执行，支持动态调整检索深度、模型选择和验证强度；

**💡 创新点**

创新点在于将语义状态抽象为持久化信念，并通过数据库驱动的声明式策略实现可审计、可组合、可复用的执行控制，弥合了传统管道的固定配置与自适应代理的不可检视之间的差距；

**🔧 技术方法**

采用信念抽象、声明式策略、数据库后端控制平面、事件驱动执行引擎、LLM路由器（RouteLLM）、CRAG式相关性评估、LLM-as-judge、链式思考提示等技术；

**📊 数据集**

使用FinanceBench财务问答基准，包括100个查询和数百页PDF原始文件；

**📈 对比分析**

与18种固定配置管道（不同模型、提示、检索组合）进行对比，实验显示检索方法对准确率影响最大，提示策略能同时提升准确率和降低成本；Credo能够根据实时信念动态切换配置，显著提高了平均准确率并降低了总体成本；

**⚠️ 局限性**

局限性包括：需要人工编写信念与策略，缺乏自动化发现；信念提取（尤其是LLM-as-judge）成本高；仅在财务领域验证，跨域泛化与适配未知场景尚待研究。

---

## 89. Disentangled Dual-Branch Graph Learning for Conversational Emotion Recognition

**arXiv ID:** 2604.14204 | [PDF](https://arxiv.org/pdf/2604.14204v1)

**作者:** Chengling Guo `[一作]` (Central South University of Forestry and Technology), Keqin Li `[通讯]` (State University of New York)

**通讯引用:** 31982 | [OpenAlex ID](https://openalex.org/A5087894632)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一个双空间特征解耦与双分支图学习相结合的框架，用于对话情绪识别

**💡 创新点**

通过显式分离模态共享与私有信息，并引入 Fourier GNN 与 Speaker‑Aware Hypergraph NN 两个分支，同时加入频域对比学习和说话人一致性约束

**🔧 技术方法**

双空间解耦（共享+私有编码器）、Fourier Graph Neural Network、Hypergraph Neural Network、频域对比学习、Transformer 融合、复合损失

**📊 数据集**

IEMOCAP 与 MELD 数据集

**📈 对比分析**

与多种 RNN、图网络、图Transformer 基线对比，WF1 在 IEMOCAP 达到 70.81、MELD 达到 65.70，均超过现有最优方法

**⚠️ 局限性**

模型较复杂，依赖预训练编码器对齐，对不同语言或实时性能未做评估

---

## 90. HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds

**arXiv ID:** 2604.14268 | [PDF](https://arxiv.org/pdf/2604.14268v1)

**作者:** Team HY-World `[一作]`, Chunchao Guo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 HY‑World 2.0，一套完整的多模态世界模型框架，能够从文本或单视图图像生成可导航的 3D Gaussian Splatting 世界，并能从多视图图像或视频重建高精度的 3D 结构。

**💡 创新点**

创新点包括：① 四阶段统一流水线（Panorama Generation、Trajectory Planning、World Expansion、World Composition）实现生成与重建的无缝衔接；② HY‑Pano 2.0 在不需要相机标定的情况下生成 360° 全景；③ WorldNav 通过语义解析与信息最大化规划多样化相机轨迹；④ WorldStereo 2.0 用关键帧 VAE、全局几何记忆和空间立体记忆提升视频生成一致性；⑤ WorldMirror 2.0 引入位置归一化、深度‑法线耦合、深度掩码预测和 MaskGaussian，显著提升重建质量；⑥ 通过混合精度、序列并行和蒸馏技术将整个流水线推向 10 分钟以内的交互速度。

**🔧 技术方法**

使用的技术包括：多模态扩散 Transformer (MMDiT)、视频扩散模型 DiT、Keyframe‑VAE、全局几何记忆 (GGM) 与空间立体记忆 (SSM++)、3D Gaussian Splatting（RGB 只用视角无关色彩）、RoPE 位置编码归一化、Depth‑to‑Normal 损失、Gumbel‑Softmax MaskGaussian、分布匹配蒸馏 (DMD)、FP8/FP16 混合精度、分布式序列并行与 SageAttention 等。

**📊 数据集**

采用真实世界高分辨率全景、Unreal Engine 合成全景与多视图数据；文本‑图像对来自 CLIP / BLIP；评估使用 Tanks‑and‑Temples、MIPNeRF360、RealEstate10K、ScanNet、NYUv2、iBims‑1、7‑Scenes、NRGBD、DTU、DL3DV、Video2World 等公开数据集。

**📈 对比分析**

在文本/单图像→全景（T2P/I2P）任务中，HY‑Pano 2.0 在 CLIP‑T/CLIP‑I、Q‑Align 质量和美学指标上优于 DiT360、Matrix3D、CubeDiff、GenEx 等方法；在 3DGS 生成与重建上，HY‑World 2.0 的 PSNR/SSIM/LPIPS 与闭源 Marble 相当，生成速度仅 10 分钟；WorldStereo 2.0 在单视图重建的 F1/AUC 上超越现有视频生成与 3D 方法；WorldMirror 2.0 在多分辨率几何恢复、相机位姿、法线估计与视图合成上均优于 1.0 版本与多项竞争者。

**⚠️ 局限性**

局限性包括：对极复杂或极稀疏多视图序列的对齐误差仍存在；记忆模块（GGM/SSM++）训练成本高且需大规模 GPU；虽然生成速度已大幅提升，但在实时高帧率细节渲染方面仍有进一步优化空间；在极端稀疏输入下仍难以完全恢复被遮挡或隐藏结构。

---

## 91. CART: Context-Aware Terrain Adaptation using Temporal Sequence Selection for Legged Robots

**arXiv ID:** 2604.14344 | [PDF](https://arxiv.org/pdf/2604.14344v1)

**作者:** Kartikeya Singh `[一作]` (University at Buffalo), Karthik Dantu `[通讯]` (University at Buffalo)

**通讯引用:** 1883 | [OpenAlex ID](https://openalex.org/A5032635242)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了CART——一种基于情境感知的高层控制器，利用视觉、深度与本体感知信息来预测合适的速度和高度，从而在复杂地形上实现更稳健的步态。

**💡 创新点**

创新点在于：① 将视觉与本体感知通过注意力机制融合，解决视觉-纹理悖论；② 引入Temporal Sequence Selection (TSS) 模块，在线选择最匹配情境的参数子序列，实现无须重新训练的动态适配；③ 通过振动稳定性指标评估并优化控制，显著降低机器人基底振动。

**🔧 技术方法**

核心技术包括：多模态感知编码（卷积网络、MLP、双向LSTM）、注意力融合、基于强化学习的高层策略（PPO），以及基于序列嵌入的TSS在线选择。

**📊 数据集**

数据集：在IsaacSim仿真中使用自定义的四种地形（Box、Rough、Downward slope、Upward slope）进行训练和测试；真实实验采用Boston Dynamics Spot在草地、泥土、混凝土、碎石、覆叶等多种离线与混合地形进行路径跟踪。

**📈 对比分析**

与多种基线（Blind、PPO、S&T、Spot内置TROT/CRAWL/AMBLE、VAPOR）比较，CART在仿真中平均提高成功率5%，振动稳定性提升45%；在真实环境中振动稳定性提升24%，行走平稳度和通过率分别提高至5.44m/10s（CRAWL）并保持相同速度时行进距离最大化。

**⚠️ 局限性**

局限性包括：① 仅在两台机器人上验证，跨平台泛化未知；② 依赖RGB‑D摄像头与IMU，场景光照变化或遮挡仍可能影响视觉编码；③ TSS序列库规模较大，实时计算成本仍显著；④ 仅关注基底振动稳定性，未深入分析能耗或长期路径规划等因素。

---

## 92. Metric-Aware Principal Component Analysis (MAPCA):A Unified Framework for Scale-Invariant Representation Learning

**arXiv ID:** 2604.14249 | [PDF](https://arxiv.org/pdf/2604.14249v1)

**作者:** Michael Leznik `[一作]` `[通讯]`, Michael Leznik

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了Metric-Aware Principal Component Analysis (MAPCA) 框架，统一了可扩展的尺度不变表示学习方法。

**💡 创新点**

创新点在于给出严格的尺度不变性判定条件 (M = C M C)，证明仅对对角度量 (IPCA) 以及完全白化 (β=1) 满足该条件，并通过 β-族实现连续的谱偏置控制。

**🔧 技术方法**

技术上使用了带有度量矩阵 M 的广义特征问题、β-族度量 M(β)=Σ^β、对角度量 D，以及对比实验验证。

**📊 数据集**

实验使用了军队学员数据集（身高、体重、胸围），同时在厘米和英寸/磅两套单位下进行测试。

**📈 对比分析**

通过比较标准 PCA、IPCA、β=0.5 MAPCA 以及完全白化四种方法的特征值和主成分系数，发现 IPCA 在两种单位下特征值相等、系数按比例变换，表明其严格尺度不变；其他方法不满足该性质。

**⚠️ 局限性**

局限性包括 β∈(0,1) 的中间值不具备尺度不变性，缺乏自动 β 选择方案，且仅在线性 PCA 范围内验证，未探讨非线性或深度学习场景。

---

## 93. Attention to Mamba: A Recipe for Cross-Architecture Distillation

**arXiv ID:** 2604.14191 | [PDF](https://arxiv.org/pdf/2604.14191v1)

**作者:** Abhinav Moudgil `[一作]` (Apple), Federico Danieli `[通讯]` (Apple)

**通讯引用:** 71 | [OpenAlex ID](https://openalex.org/A5035740107)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种两阶段跨架构蒸馏方法，将Transformer的Softmax Attention逐步蒸馏为Linear Attention，再利用该结果初始化Mamba（SSM）学生模型，从而得到高效的混合“HedgeMamba”结构。

**💡 创新点**

通过引入Hedgehog可学习特征映射实现Attention线性化，并将该线性化结果直接作为Mamba的初始化，形成两阶段蒸馏流程，突破了直接蒸馏性能瓶颈，并在token预算分配上实现了更优的效果。

**🔧 技术方法**

使用Hedgehog特征映射、线性Attention、State‑Space Model（Mamba）、Cosine相似度匹配、交叉熵微调以及CUDA选择性扫描等技术。

**📊 数据集**

训练数据采用OpenWebText（约10B tokens），验证集4M tokens；评估使用ARC-Easy/Challenge、Social IQA、PiQA、Lambada、BoolQ、RACE、LogiQA、WinoGrande、HellaSwag等下游任务。

**📈 对比分析**

与Hedgehog基线和直接蒸馏Baseline对比；在Pythia‑1B模型上，蒸馏后PPL从14.89降至14.11，接近教师13.86；下游任务准确率与教师相当或略低。Ablation显示gate分支贡献最大；token预算实验表明90%投入第二阶段最优。

**⚠️ 局限性**

仅在Pythia Transformer上验证，未测试其他Transformer；仅使用OpenWebText，未探究数据集多样性对性能的影响；未系统评估其他可能的SSM扩展，蒸馏成本仍较高。

---

## 94. H2VLR: Heterogeneous Hypergraph Vision-Language Reasoning for Few-Shot Anomaly Detection

**arXiv ID:** 2604.14507 | [PDF](https://arxiv.org/pdf/2604.14507v1)

**作者:** Jianghong Huang `[一作]` (University of Electronic Science and Technology of China), Mao Ye `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 17659 | [OpenAlex ID](https://openalex.org/A5100682785)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于异质超图视觉-语言推理框架（H_2VLR）来解决少样本异常检测问题，利用动态语义诱导与高阶推理实现精确定位与检测。

**💡 创新点**

创新点在于将视觉区域与文本概念联合构建超图结构，突破传统对局部相似度的二元匹配，显式建模跨模态的高阶空间‑语义依赖，增强全局结构一致性。

**🔧 技术方法**

核心技术包括CLIP视觉‑语言模型、动态语义诱导（DSI）、超图构造与消息传递（HGNN/HyperGCN）以及多项损失（对齐、结构平滑、Dice+Focal）。

**📊 数据集**

在八大工业与医学基准上评测：MVTec、VisA、BTAD、MPDD、BeltAD、BrainMRI、LiverCT、BUSI。

**📈 对比分析**

与PromptAD、KAG‑prompt、DictAS、IIPAD等四个最近代表方法以及多种多样本/全样本对比，H_2VLR在1/2/4-shot均取得最高或第二高的AUROC，尤其在4-shot时几乎超越全样本方法，整体表现为SOTA。

**⚠️ 局限性**

主要局限在于仅针对二维图像，超图构造与推理对计算量有一定开销，且对不同视觉语言模型的迁移性尚未充分验证。

---

## 95. CobwebTM: Probabilistic Concept Formation for Lifelong and Hierarchical Topic Modeling

**arXiv ID:** 2604.14489 | [PDF](https://arxiv.org/pdf/2604.14489v1)

**作者:** Karthik Singaravadivelan `[一作]` (Georgia Institute of Technology), Christopher MacLellan `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 388 | [OpenAlex ID](https://openalex.org/A5077641166)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 CobwebTM，一种能够在无监督、无预设主题数的前提下，利用连续文档嵌入进行增量概率概念形成，从而构建可持续演进的层级主题模型。

**💡 创新点**

创新点在于将 Cobweb 的概念形成算法迁移至高维嵌入空间，结合类别效用最大化实现自适应深度和宽度的层级结构，并通过预训练语言模型的密集嵌入避免了传统 LDA 的稀疏词袋限制与神经模型的灾难性遗忘。

**🔧 技术方法**

技术上使用连续 Cobweb 算法、预训练 RoBERTa/Transformer 文档嵌入、c‑TF‑IDF 词分布提取、以及基于类别效用的增量更新与层级重构。

**📊 数据集**

在三类数据集上评估：Spatiotemporal News、Stack Overflow、TweetNER7 进行终身主题建模，及 20 Newsgroups、AG News、Stack Overflow 进行层级主题建模。

**📈 对比分析**

与 Online LDA、Lifelong NTM、BERTopic（DBStream、MiniBatchKMeans）及其重新训练版本对比，CobwebTM 在主题连贯度（C_v）、主题稳定度（ARI）、主题中心漂移（TCD）与内部一致性（ISIM）等指标上均取得最优或相近水平，尤其在主题连贯度与稳定度上显著优于基线。

**⚠️ 局限性**

局限包括：词典提取需后处理，依赖预训练嵌入的表达能力；增量聚类对文档到达顺序敏感；长时间运行时层级统计会消耗较多内存与计算，且无法保证全局最优层级。

---

## 96. SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models

**arXiv ID:** 2604.14163 | [PDF](https://arxiv.org/pdf/2604.14163v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 97. SAGE Celer 2.6 Technical Card

**arXiv ID:** 2604.14168 | [PDF](https://arxiv.org/pdf/2604.14168v1)

**作者:** SAGEA Research Team `[一作]` (SAGEA), Wang Junhao `[通讯]` (SAGEA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发并发布了SAGE Celer 2.6系列（5B、10B、27B）语言模型，结合自监督预训练、逆推理验证、原生多模态集成以及专为南亚语言优化的Devanagari分词器。

**💡 创新点**

创新点包括：1）将逆推理（Inverse Reasoning）机制从专用agent框架迁移至核心Transformer梯度中，实现自检和多步推理；2）统一的视觉编码器与文本词嵌入共用同一Transformer，使多模态推理无适配器噪声；3）使用Grouped Query Attention (GQA)与RoPE支持256k上下文窗口；4）为Devanagari脚本设计的子词分词器，显著压缩非拉丁文本token数。

**🔧 技术方法**

技术涵盖：密集型Transformer架构、Grouped Query Attention、逆推理（IR）内部验证、端到端视觉编码、量化约束训练、知识蒸馏、长序列预训练。

**📊 数据集**

数据集为SAGEA自研高质量文本语料库（包含数学证明、软件仓库、学术论文），避免了公开基准污染，并进行严格的过滤与知识蒸馏。多模态评测使用MathVista、MMMU、DocVQA、AI2D、ChartQA等公开视觉/文本基准。

**📈 对比分析**

与SAGE Actus 2.4（32B）、Llama 3.3 70B、Qwen 2.5 32B等模型在ACUMEN、MMLU、MATH-500、HumanEval等指标上进行对比。27B Celer 2.6在MMLU-Pro、MATH-500、HumanEval上分别匹配或超过Actus 2.4，并在ACUMEN总分上保持平衡；在多模态基准中超过InternVL2 26B，接近Qwen 2.5 72B，远优于Llama 3.2 Vision 11B；在长上下文检索任务中，256k窗口下Recall > 94%。

**⚠️ 局限性**

局限包括：1）逆推理虽降低逻辑错误，但对缺失知识仍会产生可信度高的虚假答案；2）过度验证可能导致低复杂度问题耗费过多推理时间；3）在100k以上长文档中出现中区注意力衰减；4）3D几何推理受限；5）未针对高保障数学/安全代码生成做专门校验。

---

## 98. Tracking the Temporal Dynamics of News Coverage of Catastrophic and Violent Events

**arXiv ID:** 2604.14315 | [PDF](https://arxiv.org/pdf/2604.14315v1)

**作者:** Emily Lugos `[一作]`, Maurício Gruppi `[通讯]` (Villanova University)

**通讯引用:** 116 | [OpenAlex ID](https://openalex.org/A5061409964)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分析12起暴力与灾害事件的12万余篇新闻，研究了事件报道的时间与语义演变，量化了报道量、语义漂移、语义散布及关键词重要性，并归纳出事件报道的三个语义阶段；

**💡 创新点**

首次将句子级嵌入的语义中心漂移与散布量化为新闻框架演变指标，并结合tf‑idf关键词分析揭示了不同事件类型（灾害 vs 暴力）在报道高峰与后续转折中的语义驱动词；

**🔧 技术方法**

采用SentenceTransformer嵌入、指数移动平均、余弦距离、方差计算及tf‑idf权重等技术；

**📊 数据集**

使用GDELT抓取的126,602篇新闻（4,161个域名），涵盖2019–2025年的12个事件（6起暴力、6起灾害），并对同一事件的前后7天及后30天进行时序划分；

**📈 对比分析**

将各指标按事件类别进行平均和置信区间比较，发现灾害事件在语义漂移和散布幅度上高于暴力事件；虽然未给出传统精度指标，但结果显示事件报道在t≈5天左右达到峰值，并随后逐步回落；

**⚠️ 局限性**

数据可能包含错误、虚假或偏见信息，尤其是来自非主流媒体；未对错误信息或假新闻对框架演变的具体影响进行建模；模型对事件类型的二分类（灾害/暴力）有限，未覆盖其他社会/经济事件；

---

## 99. RoSLAC: Robust Simultaneous Localization and Calibration of Multiple Magnetometers

**arXiv ID:** 2604.14353 | [PDF](https://arxiv.org/pdf/2604.14353v1)

**作者:** Qiyang Lyu `[一作]` (Nanyang Technological University), Danwei Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 11248 | [OpenAlex ID](https://openalex.org/A5083472752)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种名为RoSLAC的在线磁传感器校准与定位框架，能在已建磁场地图中同时估计机器人姿态与磁力计校准参数；

**💡 创新点**

采用序列累积增强观测、交替优化解耦姿态与校准、递归最小二乘后处理实现全局最优，并在不需人工旋转平台的情况下完成在线校准；

**🔧 技术方法**

核心技术包括磁场地图构建（Gaussian Process Regression）、序列累积与里程计融合、梯度下降与Gauss‑Newton交替优化、递归最小二乘滤波；

**📊 数据集**

使用Gazebo仿真仓库环境与四个真实室内/室外场景（办公走廊、地下车库、酒店走廊、室外平台）收集的磁场与LiDAR数据；

**📈 对比分析**

与PF、SO、RBPF三种基线方法对比，使用原始与预校准磁测量，RoSLAC在大多数场景下实现约10 cm平均绝对轨迹误差，校准误差约1 μT，显示出显著的定位与校准性能提升；

**⚠️ 局限性**

局限包括：对磁场地图的依赖；在观测信息稀疏或磁场较平坦时仍可能陷入局部最优；当前实现对GPU加速尚未充分利用，需进一步提升实时性能。

---

## 100. Psychological Steering of Large Language Models

**arXiv ID:** 2604.14463 | [PDF](https://arxiv.org/pdf/2604.14463v1)

**作者:** Leonardo Blas `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 19181 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种心理学驱动的LLM控制框架，利用校准后的残差流注入（MDS）和多层向量投影，对模型内部表示进行干预并实现对OCEAN人格特质的可控表达；

**💡 创新点**

创新点在于：①实现了无界、流畅性约束的注入强度搜索，使用语义校准的中心点单位；②采用均值差分（MDS）向量实现近似线性的控制；③提出了混合驱动方法PM（人格提示+注入），大幅提升了驱动效果；

**🔧 技术方法**

技术方法包括：残差流注入、L1/L2正则化逻辑回归向量、均值差分MDS、注入强度搜索、RoBERTa流畅度判别、余弦相似度语义去重、轻量级分类器评估、GPT‑5.1文本评测；

**📊 数据集**

使用数据集：IPIP‑NEO‑120（OCEAN测评）、MPI‑120、合成的SJT（TRAIT）与自生成的1,000条心理学语句、35k条构造性文本，14个指令调优LLM（1B‑32B参数）进行实验；

**📈 对比分析**

评估方法：将MDS、各类线性探针向量、人格提示P^2和混合PM在SJT与测评量表上进行对比，计算平均SJT得分。结果显示：MDS在11/14 LLM上优于P^2，提升3.6%–16.4%；PM在13/14 LLM上优于两者，提升5.6%–21.9%（vs P^2）及3.3%–26.7%（vs MDS）；

**⚠️ 局限性**

局限性包括：实验仅限于指令调优的中小型LLM，64-token输出；α搜索成本高，未探究基模型、其他任务或多注入组合；对MDS失败的gemma‑3‑1b‑it及其他模型的原因尚未阐明；依赖已有测评量表，难以扩展至自定义属性。

---

## 101. TRACER: Trace-Based Adaptive Cost-Efficient Routing for LLM Classification

**arXiv ID:** 2604.14531 | [PDF](https://arxiv.org/pdf/2604.14531v1)

**作者:** Adam Rida `[一作]` `[通讯]` (Independent Researcher), Adam Rida (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TRACER，一个基于生产日志自动训练并安全部署LLM代理的系统

**💡 创新点**

创新点在于无需预先标注数据、通过“Parity Gate”动态保证代理与教师一致性，并生成可解释的路由边界报告

**🔧 技术方法**

使用冻结文本嵌入（BGE-large）、多种经典机器学习模型、对抗式接收器、持续学习循环和基于阈值的拒绝门控

**📊 数据集**

评测数据集包括 Banking77（77类意图）、CLINC150（150类意图，含157类真实标签）和 MNLI（3类推理）

**📈 对比分析**

在 Banking77 上以不同 α 取值实现 83–100% 覆盖，CLINC150 上 100% 覆盖；与基线（置信阈值退避）相比，TRACER 在高 α 下提供更高的教师一致性，并在日常流量中显著降低 LLM 成本

**⚠️ 局限性**

局限包括对嵌入可分离度要求高，无法处理需要组合推理的任务（如 MNLI），以及校准集与测试集分布差异导致的安全保证不足

---

## 102. Spatiotemporal Analysis of VIIRS Satellite Observations and Network Traffic During the 2025 Manitoba Wildfires

**arXiv ID:** 2604.14392 | [PDF](https://arxiv.org/pdf/2604.14392v1)

**作者:** Xiang Shi `[一作]` (University of Manitoba), Peng Hu `[通讯]` (University of Manitoba)

**通讯引用:** 71127 | [OpenAlex ID](https://openalex.org/A5100351175)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用 NASA FIRMS VIIRS 火点辐射功率 (FRP) 与 Ookla Speedtest 固定宽带测量数据，研究 2025 年马尼托巴州野火强度与网络性能（吞吐量、延迟、丢包）之间的空间时间相关性。

**💡 创新点**

首次将卫星火点强度与大规模网络性能指标结合，发现野火强度与网络吞吐量呈负相关、与延迟呈正相关，并量化了两者在不同空间尺度（地区与全省）的显著性与效应大小。

**🔧 技术方法**

采用秩相关方法（Spearman ρ 与 Kendall τ）进行非参数相关性检验，使用 LOESS 平滑和时间序列可视化展示趋势，并对相关系数的显著性进行统计测试。

**📊 数据集**

数据集包括 NASA FIRMS VIIRS 2025 年野火 FRP 观测（经纬度、时间、FRP 等）以及 Ookla Speedtest 固定宽带测量数据（吞吐量、延迟、RTT、丢包率、时间、地理位置）。

**📈 对比分析**

方法：匹配时间与空间阈值后计算 Spearman/Kendall 系数并检验 p 值；显著性阈值设为 0.05。结果显示，全省下载速度与 FRP 相关系数为 -0.214（p=0.004），下载延迟为 0.230（p=0.002），RTT 为 0.162（p=0.031）。在地区尺度，上传速度与 FRP 相关系数为 -0.121（p=0.039），上传延迟为 0.195（p=0.002）。性能表明，野火期间网络吞吐量下降，延迟显著升高（上传延迟峰值约 1500 ms，下载延迟可达 350 ms）。

**⚠️ 局限性**

局限性包括：仅使用固定宽带数据，未考虑移动网络；匹配阈值与时间窗口可能引入误差；相关性不等同因果，缺乏干预或因果推断；研究仅涵盖 2025 年马尼托巴州，结果的普适性待验证。

---

## 103. BiCon-Gate: Consistency-Gated De-colloquialisation for Dialogue Fact-Checking

**arXiv ID:** 2604.14389 | [PDF](https://arxiv.org/pdf/2604.14389v1)

**作者:** Hyunkyung Park `[一作]` (Queen Mary University of London), Arkaitz Zubiaga `[通讯]` (Queen Mary University of London)

**通讯引用:** 6765 | [OpenAlex ID](https://openalex.org/A5071220716)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种保守的对话式事实核查写法重写管道，先进行轻量表面规范化并限定范围的代词消解，再用BiCon‑Gate做语义一致性判定；

**💡 创新点**

创新点在于将双向NLI一致性信号与语义相似度融合形成实例级路由器，以在保持检索召回的同时避免语义漂移；

**🔧 技术方法**

使用的技术包括正则式收缩恢复、跨语言标点恢复、BERT式真大写、Maverick核心ference、指令微调LLM作为候选选择器，以及BiCon‑Gate中的双向NLI与余弦相似度；

**📊 数据集**

实验数据集为DialFact（多轮对话事实核查），并使用与之对应的English Wikipedia快照；

**📈 对比分析**

与多种基线（无重写、单步LLM重写、检索+验证管线）对比，轻量规范化基本保持性能，代词重写在检索和验证单独评测均略有提升，BiCon‑Gate在最终端到端上实现最高macro‑F1与准确率，尤其在SUPPORTS类上显著改善；

**⚠️ 局限性**

局限性包括对单一数据集与语言的依赖、对多模型组合的计算开销、仅处理代词与表面规范化而忽略其他口语现象，以及在多证据聚合或不同检索/验证架构下可能效果不稳定。

---

## 104. CooperDrive: Enhancing Driving Decisions Through Cooperative Perception

**arXiv ID:** 2604.14454 | [PDF](https://arxiv.org/pdf/2604.14454v1)

**作者:** Deyuan Qu `[一作]` (Toyota InfoTech Labs), Onur Altintas `[通讯]` (Toyota InfoTech Labs)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出CooperDrive框架，实现车辆协同感知与规划，提升在遮挡场景下的安全决策

**💡 创新点**

创新点包括：1）利用多任务BEV感知网络同时完成3D检测与语义定位，提升定位精度；2）以对象级轻量共享（仅传输姿态与检测结果）实现低带宽（90 kbps）高实时性；3）直接将协同检测结果融合至传统分层规划器，无需改造规划体系；4）在真实车辆闭环测试中验证规划性能提升

**🔧 技术方法**

技术包括：Bird’s‑Eye‑View 3D检测（CenterPoint风格头）+语义定位头；BEV特征共享；扫描对地图（关键点提取+NDT/ICP）定位；对象级数据融合；传统行为与运动规划器；V2V无线通信；ROS2+NVIDIA GPU推理

**📊 数据集**

主要数据集为nuScenes（用于检测与定位评估），以及两辆Toyota Sienna在真实道路场景下的闭环实验数据

**📈 对比分析**

与SSN、PointPillars、CenterPoint等检测基线相比，CooperDrive在nuScenes上取得56.27 mAP/64.78 NDS（略优于CenterPoint）；与ICP/NDT定位基线相比，NDT在噪声GNSS初始下翻译误差及航向误差显著降低；在规划测试中，TTC_min提升4.57 s、DRAC降至0.06 m/s²、DCZ增大2.11 m、违规率从18 %降至2 %，显著提升安全性与舒适性；同时保持89 ms平均端到端延迟和90 kbps带宽。

**⚠️ 局限性**

局限性包括：1）依赖V2V网络覆盖，遮挡信息仍有限；2）目前仅支持对象级共享，无法充分利用原始传感器数据的高分辨率信息；3）对高速动态场景的实时性验证仍需进一步测试；4）系统在多车辆稠密交互时的多源信息融合与冲突处理尚未完全覆盖。

---

## 105. FocalLens: Visualizing Narratives through Focalization

**arXiv ID:** 2604.14456 | [PDF](https://arxiv.org/pdf/2604.14456v1)

**作者:** S M Raihanul Alam `[一作]` (University of Iowa), Md Naimul Hoque `[通讯]` (University of Iowa)

**通讯引用:** 314 | [OpenAlex ID](https://openalex.org/A5062248644)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 FocalLens，一种可视化工具，帮助作家和文学研究者分析叙事中的聚焦化（focalization）以及视角、时间、情感等维度。

**💡 创新点**

创新点在于：①构造多层圆环符号，将 POV、内外聚焦化类型和三种维度（感知、心理、意识形态）压缩到单个 glyph 中；②将聚焦化作为核心视角维度，与传统故事线、情节等视觉化区分开来；③通过 LLM 自动标注聚焦化标签，为可视化提供数据支持。

**🔧 技术方法**

技术实现：使用 GPT‑5.4（LLM）对文本进行聚焦化标注；前端采用 React + TypeScript + D3 实现交互式时间轴与 glyph；后端采用本地 JSON 数据；文本编辑与高亮采用 QuillJS。

**📊 数据集**

数据集：Charlotte Perkins Gilman《The Yellow Wallpaper》与 Jane Austen《Persuasion》两篇文学作品；以及四名参与者提供的自创短篇；全部文本被切分为场景、事件并人工校正 LLM 结果。

**📈 对比分析**

性能对比：在两篇文本上评估 GPT‑5.3、GPT‑5.4 与 Gemini 的聚焦化标注。GPT‑5.4 在 Micro‑F1、Macro‑F1 上分别达到 0.82–0.83；精确率/召回率在内聚焦化与心理维度较高；对长篇《Persuasion》表现相对较弱，尤其是外部聚焦化标签。用户研究（4 名专家）表明工具在视角结构检视、平衡诊断和教学支持方面得到正面评价。

**⚠️ 局限性**

局限性：①对长篇或多视角作品的标注准确率下降，需改进 LLM 细粒度推理；②聚焦化仅是叙事组件之一，加入更多组件会使 glyph 过于复杂；③LLM 预测可能产生误判，需人工校正；④目前仅提供视角层面的分析，缺乏宏观情节进程的可视化。

---

## 106. Non-intrusive Learning of Physics-Informed Spatio-temporal Surrogate for Accelerating Design

**arXiv ID:** 2604.14424 | [PDF](https://arxiv.org/pdf/2604.14424v1)

**作者:** Sudeepta Mondal `[一作]` (Raytheon Technologies Research Center), Soumalya Sarkar `[通讯]` (Raytheon Technologies Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种非侵入式、物理约束的时空代理模型（PISTM），用于在未知工况下预测非线性动力系统的时空演化。

**💡 创新点**

创新点在于将 Koopman 自动编码器的物理约束与高斯过程回归相结合：先用 Koopman 自动编码器学习各工况的时空演化，再用高斯过程在潜在空间对未知工况进行插值，从而避免在测试阶段重新训练 Koopman 模型。

**🔧 技术方法**

主要技术包括：Koopman 协同编码器（卷积编码器+线性 Koopman 运算）、卷积自编码器的降阶建模、Gaussian Process 回归用于潜在空间插值、以及对结果的误差分析。

**📊 数据集**

数据集为 2D 低粘性流动（圆柱绕流）实验，使用 Lattice Boltzmann 方法得到 45 个雷诺数（Re 取 50~800）的 181 组时间步快照（每组 80×80 网格），随后在 5 个未见雷诺数（Re=83,172,218,406,594）上进行测试。

**📈 对比分析**

与单纯的 Koopman 预测、传统数据驱动方法（CNN/LSTM/POD）比较，PISTM 在预测时间窗 t=0~9 的相对误差 ε_E 与 ε_KE 均低于 0.10（除 Re=83 情况），并且在测试阶段仅需约 3 秒即可完成预测，相比真实模拟耗时约 170 分钟，实现约 10^3 倍速度提升。

**⚠️ 局限性**

局限性包括：对训练样本分布的依赖，Re=83 结果不佳说明训练点稀疏导致插值误差；在训练阶段仍需为每个工况学习 Koopman 模型；当系统极为复杂或训练数据极其稀缺时，潜在空间插值的精度可能下降。

---

## 107. Filling in the Mechanisms: How do LMs Learn Filler-Gap Dependencies under Developmental Constraints?

**arXiv ID:** 2604.14459 | [PDF](https://arxiv.org/pdf/2604.14459v1)

**作者:** Atrey Desai `[一作]` (University of Maryland), Sathvik Nair `[通讯]` (University of Maryland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用基于 BabyLM 的 GPT‑2‑small 语言模型，在 1 亿词训练语料下，利用分布式对齐搜索（DAS）对 wh‑问题与前置化（topicalization）两种填空-空缺依赖进行因果干预与评估。

**💡 创新点**

首次将 DAS 方法迁移至发展性规模的语言模型，并系统检验跨构造共享表征、动画性提升效应，揭示模型在有限输入下仍需远超人类的训练量，且对构造特异性高度敏感。

**🔧 技术方法**

因果解释技术——分布式对齐搜索（DAS）与层‑位置最大化、线性回归预测，结合 log‑概率偏移量衡量因果效应。

**📊 数据集**

BabyLM 100M 语料（约 1 亿词）以及自构造的 21,000 对 wh‑问题与 1,875,000 对前置化句子对，覆盖动、非动对象，构成训练与评估数据集。

**📈 对比分析**

比较方法包括 within‑construction（Wh→Wh、Topic→Topic）与 cross‑construction（Wh→Topic、Topic→Wh） 的因果效应统计；结果显示 within‑construction 效果明显高于跨构造，动画匹配时提升约 0.67，最高效应约 10.6；与更大模型相比，效应水平低但趋势相似。

**⚠️ 局限性**

局限性：仅限英语与文本输入；未评估岛屿约束；未考虑多模态与社会交互；仅使用单维 DAS，可能欠缺更强因果度量；模型仍需超人类数据才能达到儿童水平。

---

## 108. Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery

**arXiv ID:** 2604.14375 | [PDF](https://arxiv.org/pdf/2604.14375v1)

**作者:** Noureddine Kermiche `[一作]` `[通讯]` (Western Digital Corporation), Noureddine Kermiche (Western Digital Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于硅本地的模块化架构，通过 Task‑Specific Experts 与分布式门控器实现参数隔离，并利用 Simultaneous Pipeline 实现教师、学生与路由器的并行学习，从而在不存储历史数据的前提下实现连续学习。

**💡 创新点**

创新点包括：1）Simultaneous Pipeline 实现教师、学生与路由器三者的并行梯度下降；2）使用 Tight‑Bottleneck Autoencoder 解决高维 LLM 嵌入的后验崩塌；3）自治任务边界检测与零泄露合规的即时数据清除；4）分布式局部路由器消除全局路由器的灾难性遗忘。

**🔧 技术方法**

采用知识蒸馏、半冻结骨干、混合专家网络、VAE/TB‑AE 重构路由、对比软路由、Live Distillation、自动化承诺门等技术。

**📊 数据集**

在 Split‑MNIST、Synthetic 4096‑D LLM 嵌入数据集以及标准视觉/语言基准上进行实验。

**📈 对比分析**

与 EWC、LwF、经验回放等方法对比，Split‑MNIST 上任务 A 在任务 B 后保持 99.42% 的准确率，TB‑AE 在 4096‑D 空间实现 203 倍辨识度，整体系统在混合流下约 95.5% 的端到端准确率，且完全符合 GDPR 零泄露要求。

**⚠️ 局限性**

局限性包括：仅适用于任务/域增量学习；需要块级顺序数据流；路由器数量线性增长导致 O(N) 计算；可能出现误判；前向迁移受限；在高度交错流中难以收敛；TB‑AE 的瓶颈维度需根据任务动态调整。

---

## 109. CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization

**arXiv ID:** 2604.14214 | [PDF](https://arxiv.org/pdf/2604.14214v1)

**作者:** Deep Shah `[一作]` (Google), Priyanka Tiwari `[通讯]` (Purdue University)

**通讯引用:** 1140 | [OpenAlex ID](https://openalex.org/A5039961495)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了CROP框架，在自动提示优化中加入长度惩罚，使大型语言模型在保持推理准确性的同时显著压缩生成的token数。

**💡 创新点**

创新点在于将输出长度视为多目标优化约束，使用连续文本正则化梯度实现提示优化与token效率的双重提升。

**🔧 技术方法**

采用文本微分、双目标文本梯度（任务准确度 + 长度惩罚）、高容量Meta‑Optimizer（如Gemini 3.1 Pro）、批量梯度聚合等技术。

**📊 数据集**

使用的数据集包括GSM8K、LogiQA和BIG‑Bench Hard（Object Counting）。

**📈 对比分析**

与零射、Chain‑of‑Thought、TextGrad等基线对比，Token消耗下降约80.6%，准确率仅轻微下降，保持竞争力。

**⚠️ 局限性**

主要局限是需要高容量Meta‑Optimizer；仅针对输出token进行约束，未考虑输入token成本；优化阶段的计算成本较高。

---

## 110. Path-Sampled Integrated Gradients

**arXiv ID:** 2604.14338 | [PDF](https://arxiv.org/pdf/2604.14338v1)

**作者:** Firuz Kamalov `[一作]` (Canadian University Dubai), Neda Abdelhamid `[通讯]` (Abu Dhabi School of Management)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出了路径采样积分梯度（PS‑IG）方法，用基线沿输入到基线的线性路径采样的期望值来计算特征重要性。

**💡 创新点**

创新点在于将随机基线采样与路径加权积分等价化，使得期望值可以通过单一确定性积分实现，从而把Monte‑Carlo误差率 O(m⁻¹/2) 提升到 O(m⁻¹)，并在均匀采样下将归因方差降低约 1/3。

**🔧 技术方法**

使用了积分梯度理论、概率分布的累积分布函数、随机梯度噪声建模、Itô 等距性、Riemann 和 Monte‑Carlo 近似分析。

**📊 数据集**

实验使用人工生成的三维线性、二次、Sigmoid 函数（无公开真实数据集），通过注入白噪声评估归因方差与收敛速度。

**📈 对比分析**

与标准 IG 进行对比，PS‑IG 在相同梯度评估次数下的方差约为 IG 的 1/3，且确定性估计的误差收敛速率显著快于 Monte‑Carlo 方法。

**⚠️ 局限性**

局限性包括：仍不满足完整性公理；主要针对线性路径，可能在非线性或复杂模型上效果不确定；需要模型可微且梯度平滑；对基线初始点的选择仍有一定敏感性。

---

## 111. A Unified Model and Document Representation for On-Device Retrieval-Augmented Generation

**arXiv ID:** 2604.14403 | [PDF](https://arxiv.org/pdf/2604.14403v1)

**作者:** Julian Killingback `[一作]` (University of Massachusetts Amherst), Maryam Karimzadehgan `[通讯]` (Google)

**通讯引用:** 454 | [OpenAlex ID](https://openalex.org/A5041700556)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的ECG模型，整合检索、压缩与生成三大功能，实现单模型完成文档嵌入、压缩以及答案生成；

**💡 创新点**

创新点在于将检索与上下文压缩共享同一多向量表示，并通过端到端联合训练，使得在设备端对内存和磁盘空间极限的条件下仍能保持与传统RAG相当甚至更优的性能；

**🔧 技术方法**

采用多向量检索（MaxSim）、自监督预训练（重构与邻近文本）、对比学习、知识蒸馏、信息熵损失以及动态温度缩放等技术；

**📊 数据集**

使用Wiki拆分片段进行自监督预训练，并在Natural Questions与TriviaQA数据集上进行RAG微调与评估；

**📈 对比分析**

在固定上下文预算下，ECG在SmolLM（135M）和Gemma（1B）模型上分别在NQ与TriviaQA上实现EM分别为0.343/0.515和0.361/0.540，约为最强基线ColBERT的3倍；在固定性能下，其所需的检索向量预算只有ColBERT的1/5到1/8，且磁盘空间利用率仅为传统压缩+检索模型的一半；

**⚠️ 局限性**

主要局限包括：1）对检索向量数量的依赖较高，过大会导致存储/检索开销；2）在极低预算下性能仍落后于基线；3）训练过程需对比损失与蒸馏平衡进行精细调参；4）目前仅在公共问答数据集验证，缺乏对个人隐私文档场景的实测。

---

## 112. Adaptive Query Routing: A Tier-Based Framework for Hybrid Retrieval Across Financial, Legal, and Medical Documents

**arXiv ID:** 2604.14222 | [PDF](https://arxiv.org/pdf/2604.14222v1)

**作者:** Afshan Hashmi `[一作]` `[通讯]` (Tuwaiq Academy), Afshan Hashmi (Tuwaiq Academy)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文针对金融、法律和医学三大专业文档，构建统一基准，比较向量检索、树形推理和自适应混合检索三种 Retrieval-Augmented Generation（RAG）体系，进一步在真实 SEC 文档的 FinanceBench 上进行验证。

**💡 创新点**

创新点在于提出四层查询复杂度分层框架、GPT‑4o‑mini 作为 LLM‑as‑Judge 评测方法，并设计 Adaptive Hybrid Retrieval（AHR）能根据查询层级动态路由向量检索或树形推理，实现多策略融合。

**🔧 技术方法**

采用 FAISS + Dense Passage Retrieval 进行向量检索，GPT‑4o‑mini 生成节点摘要与树形推理逻辑；树形索引基于正则表达式检测跨引用；Hybrid 模式通过向量与树检索结果的融合与加权实现。

**📊 数据集**

数据集涵盖 1,200 篇 SEC 10‑K/10‑Q 财务报表、若干法律主服务协议、医学 Phase III 临床报告，以及 150 题 FinanceBench 真实 SEC 问题。

**📈 对比分析**

通过 LLM‑as‑Judge 计算整体质量分，发现 Tree Reasoning 在整体（0.900）和 FinanceBench（0.938）上表现最佳；Vector RAG 在多文档合成（Tier 4）和金融域最优；AHR 在跨引用（0.850）和多节查询（0.929）中取得最高分，证明单一策略无统治力。

**⚠️ 局限性**

局限性包括：实验仅覆盖 50 个 FinanceBench 题目且缺乏统计显著性检验；未在多模态文档或更大规模数据上验证；对 GPT‑4o‑mini 的依赖可能导致成本与隐私问题；未来需加入跨编码器重排序、强化学习路由与多模态集成。

---

## 113. Towards Verified and Targeted Explanations through Formal Methods

**arXiv ID:** 2604.14209 | [PDF](https://arxiv.org/pdf/2604.14209v1)

**作者:** Hanchen David Wang `[一作]` (Vanderbilt University), Meiyi Ma `[通讯]` (Vanderbilt University)

**通讯引用:** 1544 | [OpenAlex ID](https://openalex.org/A5101671027)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种针对深度神经网络的 Verified and Targeted Explanations 框架，用于生成具有形式化保证的半事实解释。

**💡 创新点**

创新点在于将半事实解释、目标类感知与形式化可达性分析相结合，首次提供针对用户指定风险替代类的数学鲁棒性保证。

**🔧 技术方法**

采用类特定敏感性启发式、ε-鲁棒性形式化可达性分析以及半事实解释生成技术。

**📊 数据集**

在 MNIST、GTSRB、EMNIST 图像分类数据集和 TaxiNet 回归数据集上进行评估。

**📈 对比分析**

与 LIME、IG 等传统解释方法和基线形式化方法比较，取得 30% 以上的准确率提升，且解释特征数显著减少。

**⚠️ 局限性**

局限性在于目前仅支持可微或线性可求解的模型，对高维复杂场景的计算成本仍较高。

---

## 114. Crowdsourcing of Real-world Image Annotation via Visual Properties

**arXiv ID:** 2604.14449 | [PDF](https://arxiv.org/pdf/2604.14449v1)

**作者:** Xiaolei Diao `[一作]` (University College London), Fausto Giunchiglia `[通讯]` (University of Trento)

**通讯引用:** 15826 | [OpenAlex ID](https://openalex.org/A5001227032)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于知识表示、视觉属性和层次结构的图像标注方法，并在众包框架中实现

**💡 创新点**

通过将视觉属性嵌入层次结构，引导标注者以可视化证据决策，显著降低语义鸿沟

**🔧 技术方法**

知识表示、自然语言处理、计算机视觉（目标定位、属性定义）以及众包平台

**📊 数据集**

构造的12类（鸟类、车辆、乐器）图像数据集，1200张图像

**📈 对比分析**

与仅使用名称（Method A）和仅有层次结构（Method B）相比，Method C的Krippendorff α提升至0.974，任务耗时略增，模型下游精度提升约10-25%

**⚠️ 局限性**

仍需人工多轮审核，标注成本略高，且对视觉属性定义的主观性可能影响标注一致性

---

## 115. Interpretable Human Activity Recognition for Subtle Robbery Detection in Surveillance Videos

**arXiv ID:** 2604.14329 | [PDF](https://arxiv.org/pdf/2604.14329v1)

**作者:** Bryan Jhoan Cazáres Leyva `[一作]` (Instituto Politécnico Nacional), Sergio Isahí Garrido-Castañeda `[通讯]` (Instituto Politécnico Nacional)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5116504310)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种结合YOLO姿态估计、运动与交互特征提取及随机森林分类器的实时、可解释街头非暴力抢劫检测系统

**💡 创新点**

创新点在于将可解释的手臂、手部运动及交互特征与时序迟滞滤波相结合，显著降低误报并提升鲁棒性

**🔧 技术方法**

采用YOLO姿态估计、指数移动平均平滑、手臂/手部速度、交互距离/方向特征、随机森林分类器与时序迟滞滤波技术

**📊 数据集**

使用自制的90例训练/验证集（29抢劫、61非抢劫）以及47例互联网测试集（17抢劫、30非抢劫）

**📈 对比分析**

在验证集上达到整体准确率0.83，抢劫类召回率0.83、F1 0.77；在测试集上准确率73.3%，抢劫类召回率0.59、F1 0.62，并实现了在NVIDIA Jetson Nano上的实时推理

**⚠️ 局限性**

受限于样本量有限、类别不平衡、仅关注两人交互、对光照、遮挡和低分辨率的鲁棒性不足

---

## 116. Equifinality in Mixture of Experts: Routing Topology Does Not Determine Language Modeling Quality

**arXiv ID:** 2604.14419 | [PDF](https://arxiv.org/pdf/2604.14419v1)

**作者:** Ivan Ternovtsii `[一作]` (Uzhhorod National University), Yurii Bilak `[通讯]` (Uzhhorod National University)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5042083572)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究稀疏 MoE（Mixture‑of‑Experts）模型的路由拓扑是否影响最终语言模型质量，使用多种余弦路由变体在 WikiText‑103 与 OpenWebText 上进行系统实验；

**💡 创新点**

提出几何 MoE 及余弦路由，证明路由拓扑在不同配置下等价（PPL 1 PPL 内），并展示路由容量与机制的层级关系；同时提出零拷贝相对范数停止实现推理 FLOPs 节省；

**🔧 技术方法**

利用低维投影瓶颈、学习质心、余弦相似度路由、正交化评估、TOST 等价检验、零拷贝停止、以及多种种子和对照实验的统计方法；

**📊 数据集**

使用 WikiText‑103（32k BPE 词表）和 OpenWebText 作为训练与验证数据集；

**📈 对比分析**

通过 62 次对照实验和 3 种种子、配对自举与 TOST 检验，比较 5 种余弦路由、哈希/随机/Top‑1 路由以及标准线性路由；结果显示 5 种余弦路由在 1 PPL 以内等价，标准线性路由因路由参数容量较大领先约 1.2%；零拷贝停止可在不显著提升 PPL 的前提下节省约 25% FLOPs；

**⚠️ 局限性**

实验仅覆盖 76–138 M 参数规模，未验证亿级模型或视觉、结构推理任务；路由容量差异仍存在，且下游任务在该规模下准确率接近随机，表明结果对任务与规模的普适性有限。

---

## 117. Model-Based Reinforcement Learning Exploits Passive Body Dynamics for High-Performance Biped Robot Locomotion

**arXiv ID:** 2604.14565 | [PDF](https://arxiv.org/pdf/2604.14565v1)

**作者:** Tomoya Kamimura `[一作]` (University of Osaka), Akihito Sano `[通讯]` (Nagoya Institute of Technology)

**通讯引用:** 1907 | [OpenAlex ID](https://openalex.org/A5109992113)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过模型基强化学习，比较了含被动弹性元件与仅使用致动器的两种双足机器人在地面行走和跑步的学习过程与结果。

**💡 创新点**

引入被动身体动力学（弹性、背驱动力）以形成稳定极限环，从而实现更稳健、能耗更低的本体适配的行走模式，并证明了世界模型学习在此类被动系统中的优势。

**🔧 技术方法**

使用DreamerV2（基于世界模型的深度强化学习）、MuJoCo仿真、UMAP降维可视化以及电机/线性致动器/弹簧的物理模型。

**📊 数据集**

仅在仿真环境中采集机器人状态图像（64×64单通道）和对应动作奖励，未使用公开的人类行走/跑步数据集。

**📈 对比分析**

通过10次实验的奖励曲线、UMAP轨迹闭合度、能量消耗和坡道鲁棒性进行比较；被动模型收敛慢但能量更低、运动更柔软且更稳健；主动模型收敛快但能耗高、对坡度的鲁棒性差。

**⚠️ 局限性**

仅在仿真中验证，未分离弹性与背驱两种被动特性，缺乏与真实机器人对照；奖励仅考虑速度和高度，可能不完全激励被动极限环；未与无模型或基于神经网络的强化学习对比。

---

## 118. Simulating Human Cognition: Heartbeat-Driven Autonomous Thinking Activity Scheduling for LLM-based AI systems

**arXiv ID:** 2604.14178 | [PDF](https://arxiv.org/pdf/2604.14178v1)

**作者:** Hong Su `[一作]` (Chengdu University of Information Technology), Hong Su `[通讯]` (Chengdu University of Information Technology)

**通讯引用:** 667 | [OpenAlex ID](https://openalex.org/A5110930319)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于心跳的自我调度框架，利用周期性触发器主动安排LLM的规划、反思、回忆等认知模块，形成连续自我调节的思考流程；

**💡 创新点**

核心创新是把心跳机制引入调度层，使思考活动的时序由学习到的策略决定，而非硬编码触发；实现了动态添加/删除认知模块、基于历史交互自适应优化调度策略；

**🔧 技术方法**

使用强化学习/元学习框架训练调度器，采用注意力序列模型（LSTM+多头注意力）预测多日思考序列；调度器与LLM的交互以日志方式持续学习；

**📊 数据集**

主要使用合成数据集：1,800天的日常认知活动序列（6类动作），包含天气、时间等上下文；后续实验加入第7类动作扩展测试；

**📈 对比分析**

与基线（无心跳或固定规则调度）比较，评估指标包括动作分布熵、动作覆盖率、罕见动作召回率，实验显示熵接近真实、覆盖率100%，罕见动作召回率78%；在动作空间扩展实验中模型成功预测新动作，保持整体时间结构；

**⚠️ 局限性**

局限性：仅在合成数据上验证，缺乏真实世界任务的实测；心跳频率与延迟反馈机制对实际LLM的可扩展性未充分评估；模型训练依赖大量自生成日志，实际部署时数据收集和安全性需进一步研究。

---

## 119. Calibrate-Then-Delegate: Safety Monitoring with Risk and Budget Guarantees via Model Cascades

**arXiv ID:** 2604.14251 | [PDF](https://arxiv.org/pdf/2604.14251v1)

**作者:** Edoardo Pona `[一作]` (King's College London), Nicola Paoletti `[通讯]` (King's College London)

**通讯引用:** 1007 | [OpenAlex ID](https://openalex.org/A5016140478)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于委托价值（DV）探测器的LLM安全监控级联框架（CTD），在不需要批处理统计的流式部署中通过阈值校准实现预算控制，

**💡 创新点**

创新点在于用DV探测器直接估计升降专家带来的收益，取代传统的不确定性委托；并利用 Learn‑then‑Test 多假设检验与 Pareto 过滤在有限样本下提供 PAC‑style 预算保证；

**🔧 技术方法**

核心技术包括：logistic 回归安全探测器、岭回归 DV 探测器、Learn‑then‑Test（LTT）阈值校准、固定序列多假设检验、Pareto 过滤与阈值选择；

**📊 数据集**

使用四个平衡安全/无害标签的数据集：Anthropic HH、MTSamples、MTS‑Dialog 与 ToolACE；

**📈 对比分析**

与基于不确定性顶‑k、基准不确定性阈值和专家单独推理等方法对比，CTD 在所有预算水平下均优于不确定性委托，强专家时 AUC 提升最高 7.9%/准确率 9%，弱专家时 AUC 及准确率提升分别达 11%/19%，并在预算超限时避免性能下降；

**⚠️ 局限性**

局限性包括：仅针对二分类安全标签，依赖探测器与专家的相对性能，且在极低有效委托容量场景下仍需小心处理。

---

## 120. PriHA: A RAG-Enhanced LLM Framework for Primary Healthcare Assistant in Hong Kong

**arXiv ID:** 2604.14215 | [PDF](https://arxiv.org/pdf/2604.14215v1)

**作者:** Richard Wai Cheung Chan `[一作]` (Hong Kong Polytechnic University), Wenqi Fan `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 4961 | [OpenAlex ID](https://openalex.org/A5043696243)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 PriHA，一个三阶段检索增强生成框架，集成本地医疗知识库与实时 Web 检索，为香港市民提供可追溯、精准、易懂的初级医疗信息。

**💡 创新点**

创新点包括：① 双源检索（本地+Web）与 Reconciler 调和合成；② 查询优化器的多轮澄清与意图生成；③ 父子分块检索与多模态排名，提升信息完整性与准确性。

**🔧 技术方法**

采用 LLM（DeepSeek）、RAG、关键词+语义检索、Web Search Agent、LLM 重新排序、意图分类器、意图生成器、父子分块、对齐等技术组合。

**📊 数据集**

使用了 HK‑PriHCQA 数据集（400 Q&A 对），源自香港政府官方文件与公共卫生部门文档。

**📈 对比分析**

与 Zero‑shot LLM、Local‑Only RAG、Web‑Only RAG 进行对比，评估 Accuracy、Completeness、Trustworthiness、Clarity、Relevance 等五项指标；DRAG 平均分 4.20，显著优于其他配置。

**⚠️ 局限性**

实验未包含多轮对话（查询优化器被禁用）、静态知识库更新滞后、Web 搜索受限于白名单与排名，且对极端边缘情况的鲁棒性仍有限。

---

## 121. QualiaNet: An Experience-Before-Inference Network

**arXiv ID:** 2604.14193 | [PDF](https://arxiv.org/pdf/2604.14193v1)

**作者:** Paul Linton `[一作]` (Columbia University), Paul Linton `[通讯]` (Columbia University)

**通讯引用:** 320 | [OpenAlex ID](https://openalex.org/A5012911748)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了QualiaNet，两阶段架构先通过视差梯度模拟人类立体视觉体验，再用CNN从该体验中估计绝对视距。

**💡 创新点**

创新点在于利用自然场景中“近景视差梯度大、远景视差梯度小”的统计特性，将立体视觉缺陷（无绝对距离信息）转化为对视距的推断依据，从而实现仅凭视差梯度恢复绝对距离。

**🔧 技术方法**

使用基于Unity渲染的视差梯度图像作为输入的卷积神经网络（CNN），网络结构沿视网膜后侧通路递进，输入覆盖56°视野，接受大小从V2到V3A的感受野。

**📊 数据集**

数据集为600张Unity生成的视差梯度图（不同视距、不同物体去除、水平翻转）作为训练集，200张新场景图像作为测试集。

**📈 对比分析**

通过对比传统仅用几何三角测量方法的缺失，QualiaNet在测试集上取得R²=0.97、RMSE≈0.08 m的高精度回归性能。

**⚠️ 局限性**

局限性包括仅在合成场景下验证，缺乏真实世界数据；仅使用视差梯度而忽略其他单目线索；对场景几何变化和遮挡的鲁棒性未充分评估。

---

## 122. EviSearch: A Human in the Loop System for Extracting and Auditing Clinical Evidence for Systematic Reviews

**arXiv ID:** 2604.14165 | [PDF](https://arxiv.org/pdf/2604.14165v1)

**作者:** Naman Ahuja `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**通讯引用:** 1999 | [OpenAlex ID](https://openalex.org/A5100748239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多智能体系统 EviSearch，用于从原始临床试验 PDF 自动生成符合本体的结构化证据表，并为每个单元格提供可审计的来源证明。

**💡 创新点**

创新点在于将 PDF 直接查询、检索驱动的搜索、冲突仲裁与强制页面级验证相结合，形成端到端的可审计工作流，并在此过程中保持人机协同。

**🔧 技术方法**

采用 LLM（Gemini‑2.5‑Flash、GPT‑4.1）、Landing AI 文档解析、OpenAI Embedding 进行检索、工具调用式结构化生成、以及前端可视化审计界面；整体框架基于多智能体与对话式仲裁。

**📊 数据集**

使用了临床试验 PDF 的临床主治医生精心标注的基准集（mCSPC 试验数据），共 133 列字段，涵盖试验特征、患者特征、疗效结果、亚组分析等；每条数据均附带精确的页面与模态（文本/表格/图形）来源。

**📈 对比分析**

与 Gemini‑2.5‑Flash（PDF 原始上传）、Gemini‑2.5‑Flash（解析文档）和 GPT‑4.1（解析文档）三种基线对比；EviSearch 在整体准确率、完整率上分别达到 90.9%、91.6%、91.3%，比最佳基线提升约 7‑8 分，且在图形来源、表格来源等难点模态的表现尤为突出。

**⚠️ 局限性**

局限性包括：对 LLM 的概率性推理和误判依赖、较高的 API 费用与 token 消耗、评估仍以 LLM 判定为主可能缺乏临床专家细致判读、以及在完全自动化前仍需专家监督。

---

## 123. From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioto Review in Ambient AI Scribe Documentation

**arXiv ID:** 2604.14152 | [PDF](https://arxiv.org/pdf/2604.14152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 124. Interactive Exploration of Large-scale Streamlines of Vector Fields via a Curve Segment Neighborhood Graph

**arXiv ID:** 2604.14365 | [PDF](https://arxiv.org/pdf/2604.14365v1)

**作者:** Nguyen Phan `[一作]` (University of Houston), Guoning Chen `[通讯]` (University of Houston)

**通讯引用:** 3231 | [OpenAlex ID](https://openalex.org/A5055426585)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一个基于 Web 的交互式框架，利用曲线段邻接图（CSNG）和 Louvain 社区检测实现多层次（流线、子曲线、段级）流场分析，并配备力导向布局和邻接矩阵可视化。

**💡 创新点**

创新点包括：①将 CSNG 用于段级社区检测，支持细粒度流线结构识别；②实现多级 CSNG 与实时交互，允许快速从全局到局部逐步细化；③引入邻接矩阵（AMCS）视图，补充力导向图的局部信息；④在浏览器端通过矩阵压缩、Web Workers 等技术实现百万段实时可视化。

**🔧 技术方法**

使用技术：CSNG 构建、kNN/RBN 邻居搜索、Louvain 社区检测、力导向布局、邻接矩阵可视化、TypedArray/压缩、React 前端、Web Workers，并规划 WebGPU 加速。

**📊 数据集**

采用的流线数据集包括方柱后流、太阳羽流（Plume）、Crayfish 模拟、以及包含 1.4M 段的 Couette 涡流等。

**📈 对比分析**

与传统 PCA‑kmeans 比较：社区检测耗时 <300 ms，CSNG 构建约 222 s；在 1.4M 段时仍能保持 30 fps；Jaccard 指数 0.712（CSNG/Louvain）对比 0.275（PCA‑kmeans）；整体 4–10× 更快，内存约 2–4 GB（受浏览器限制）。

**⚠️ 局限性**

限制：浏览器内存和多线程受限，易出现 OOM；不支持时间序列或群组分析；交互拆分/合并主观性强；邻居搜索和分辨率参数敏感；无法保证检测到所有结构，需多次参数调优。

---

## 125. End-to-End Learning-based Operation of Integrated Energy Systems for Buildings and Data Centers

**arXiv ID:** 2604.14184 | [PDF](https://arxiv.org/pdf/2604.14184v1)

**作者:** Zhenyu Pu `[一作]` (Xi'an Jiaotong University), Xiaohong Guan `[通讯]` (Tsinghua University)

**通讯引用:** 18886 | [OpenAlex ID](https://openalex.org/A5074672866)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个面向建筑与数据中心的集成能源系统，利用废热回收实现多能协同供应，并提出一种端到端学习的运维优化框架，直接将预测模型与约束优化耦合，提升系统经济性。

**💡 创新点**

创新点在于将预测与优化统一到一个可微分的端到端管道中，采用隐函数定理和 CVXPYLayer 使得梯度可传播，训练时不仅关注预测误差，还直接优化能源成本，从而显著提升决策质量；此外首次将数据中心废热与建筑能源需求协同集成。

**🔧 技术方法**

主要技术包括 LSTM 时序预测、隐式可微分优化（KKT+隐函数定理）、CVXPYLayer 以及端到端学习框架；系统模型涵盖电池、热水罐、制冷罐、氢能存储、燃料电池、热泵等设备。

**📊 数据集**

使用 CityLearn 数据集（建筑电、热、冷需求及日照）与 HP Enterprise‑Cray EX Frontier 数据中心数据（电力消耗、废热）作为真实场景数据。

**📈 对比分析**

方法对比包括理论最优、传统 predict‑then‑optimize 以及提出的端到端方法；端到端在成本上比预测-先优化提升约 7–9%，并在不同工作负载下废热回收可使总能耗成本下降 10% 以上。

**⚠️ 局限性**

限制在于仍依赖高质量的历史数据；仅考虑单日 1h 间隔的调度，未考虑更长时域；氢能市场和储能非线性行为未全面覆盖；系统假设凸性，实际工况可能更复杂。

---

## 126. Reinforcement Learning via Value Gradient Flow

**arXiv ID:** 2604.14265 | [PDF](https://arxiv.org/pdf/2604.14265v1)

**作者:** Haoran Xu `[一作]` (University of Texas at Austin), Amy Zhang `[通讯]` (University of Texas at Austin)

**通讯引用:** 2106 | [OpenAlex ID](https://openalex.org/A5101754384)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a4b10f5d-130b-4e77-9367-6469ec621899` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 Value Gradient Flow (VGF) 的方法，将行为正则化强化学习（如离线 RL 与 RLHF）重新表述为从参考分布到 Boltzmann 最优策略分布的最优传输问题，并通过粒子梯度流实现无显式策略参数化的隐式多模态策略。

**💡 创新点**

创新点包括：
- 将行为正则化 RL 看作最优传输任务，利用传输预算作为隐式正则化；
- 用粒子梯度流代替传统的重参数化策略梯度，避免多步采样的梯度传播不稳定；
- 通过可调节的传输步数实现测试时的自适应规模化；
- 既保持了对参考分布的控制，又能突破其支撑，提升探索与性能。

**🔧 技术方法**

核心技术：粒子梯度流、Wasserstein 最优传输、最大熵强化学习、离线数据行为克隆、Diffusion/Flow 生成模型、奖励模型梯度引导、RLHF 中的对抗奖励与基线对比。

**📊 数据集**

实验数据集：D4RL（MuJoCo、AntMaze）、OGBench（机器人行走与操作任务）、RLHF 任务（TL;DR Summarize 与 Anthropic Helpful and Harmless Dialogue）等。

**📈 对比分析**

与 TD3+BC、IQL、IVR、Diffusion-QL、SfBC、FQL、ReBRAC、Flow BRAC、IDQL 等基线相比，VGF 在大多数任务上获得更高得分，特别是 AntMaze 等难度较高任务；在 RLHF 评估中亦实现了比现有方法更高的胜率。

**⚠️ 局限性**

限制与未来工作：
- 当参考分布严重偏向子最优行为时，VGF 的性能受限；
- 需要通过分布重加权等技术提升鲁棒性；
- 价值函数泛化误差大时可能导致过度探索；
- 与更强表达的价值函数结合的研究仍待展开。

---

## 127. Awakening Dormant Experts:Counterfactual Routing to Mitigate MoE Hallucinations

**arXiv ID:** 2604.14246 | [PDF](https://arxiv.org/pdf/2604.14246v1)

**作者:** Wentao Hu `[一作]` (Xi'an Jiaotong University), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 56971 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的推理框架 Counterfactual Routing (CoR)，通过因果分析唤醒稀疏专家网络中被路由忽略的专家，从而降低 MoE 模型的幻觉生成。

**💡 创新点**

创新点在于：① 用离线因果分析发现并量化“沉睡专家”（Dormant Expert）及其对事实正确性的因果影响（CEI）；② 通过层级预算重分配和专家级因果重加权，实现计算不变的专家重分配；③ 将路由置信度与 CEI 先验融合，动态选择专家，兼顾语言流畅性与事实准确性。

**🔧 技术方法**

主要技术包括：层级敏感度分析（Contrastive Sensitivity Normalization）、虚拟消融法计算 CEI、计算保持的专家重分配、以及上下文-因果先验融合的专家选取策略。

**📊 数据集**

使用了多种事实性基准数据集（TruthfulQA、FACTOR、TriviaQA）以及通用推理数据集（GSM8K、MMLU、ARC-C/E）进行评估。

**📈 对比分析**

与基线（标准 Top‑k 路由、随机路由、DoLa、ITI）对比，CoR 在事实性任务上平均提升约 3.1%（不增加推理开销），在 Pareto 前沿上优于单纯扩大专家预算的静态扩容策略，且在通用推理任务中保持甚至略有提升。

**⚠️ 局限性**

局限性：CoR 仅能增强模型内部已有知识的检索能力，无法补充训练中未学习到的新事实；若目标信息完全缺失，CoR 无法修正幻觉。

---

## 128. Benchmarking Linguistic Adaptation in Comparable-Sized LLMs: A Study of Llama-3.1-8B, Mistral-7B-v0.1, and Qwen3-8B on Romanized Nepali

**arXiv ID:** 2604.14171 | [PDF](https://arxiv.org/pdf/2604.14171v1)

**作者:** Ananda Rimal `[一作]` (Nepal Engineering College), Adarsha Rimal `[通讯]` (Tribhuvan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对罗马化尼泊尔语（Romanized Nepali）的生成任务进行零样本与参数高效微调（QLoRA+rsLoRA）后，分别在 Llama‑3.1‑8B、Mistral‑7B‑v0.1 和 Qwen3‑8B 三款开放权重 LLM 上进行系统化的基准测试。

**💡 创新点**

①首次提供了针对非标准化转写脚本的多维度评估框架；②提出并验证了“适配余量假设”（adaptation headroom hypothesis），解释模型弱零样本表现时可获得更大微调收益；③在同等规模模型上进行严格的零样本对比与微调后性能排名，确定 Qwen3‑8B 为整体最佳、Llama‑3.1‑8B 为迭代开发首选。

**🔧 技术方法**

使用 4‑bit NF4 量化 + QLoRA + rsLoRA（rank 32，α 64）在双 GPU（Tesla T4）上进行 3 轮微调；评估指标包括 PPL、BERTScore、chrF++、ROUGE‑1/2/L、BLEU；采用 Golden Point 检查点恢复。

**📊 数据集**

从 52k 条 Devanagari Alpaca 训练集抽取 10k 条样本，先做 5k 条语义翻译（Instruction→英文）+ 罗马化转写（Input/Output），再做 5k 条全转写（Instruction/Input/Output）得到 10k 条双语 Alpaca 格式数据，划分 9k 训练、1k 测试。

**📈 对比分析**

在 1k 测试集上进行零样本和微调后对比，使用 5 个指标覆盖 7 个测度维度。结果：微调后所有模型均达 BERTScore ≈0.75、chrF++>23，Qwen3‑8B 在结构一致性（ROUGE）和 BLEU 上领先；Llama‑3.1‑8B 在适配余量与后期 BERTScore 上表现最佳；Mistral‑7B‑v0.1 以最低 PPL（2.81）和最高 BERTScore 增益（+0.50）突出。

**⚠️ 局限性**

①仅基于单一指令跟随数据集，未覆盖社交媒体、情感分析等实际场景；②转写样本的多样性未完全模拟真实用户产生的语音噪声；③BERTScore 使用的多语言 BERT 对罗马化尼泊尔语曝光有限，可能低估语义相似度；④量化（4‑bit NF4）对最终指标可能有轻微影响，未单独测定；⑤仅评估开放权重 LLM，未包含闭源模型或更大规模模型。

---

## 129. VoxSafeBench: Not Just What Is Said, but Who, How, and Where

**arXiv ID:** 2604.14548 | [PDF](https://arxiv.org/pdf/2604.14548v1)

**作者:** Yuxiang Wang `[一作]` (Chinese University of Hong Kong), Zhizheng Wu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VoxSafeBench 评估框架，系统衡量语音语言模型(SLM)在安全、公平和隐私三大维度下的社会对齐能力，并设计了两层（Tier 1 内容风险、Tier 2 语音条件风险）的测试任务。

**💡 创新点**

创新点在于：①首次将安全、公平、隐私三大社会风险统一到同一评测框架；②两层设计清晰区分仅语义风险与语音上下文驱动风险；③通过文本上限、感知探测等控制验证 Tier 2 的真实性；④构建 22 任务、双语（英中）覆盖，展示 SLM 在语音条件下的显著性能下降。

**🔧 技术方法**

主要技术包括：语音合成（CosyVoice3）、ASR 质量控制（Whisper-large-v3）、多模态 SLM（Qwen3-Omni、Mimo-Audio、Gemini‑3、GPT‑4o‑Audio 等）、链式思考（CoT）评测、以及多任务评估（直接答复率、拒绝率、警告率、隐私意识率、偏差得分）。

**📊 数据集**

数据集：从现有文本基准改造、手工构造以及公开语音数据集合成，涵盖 22 任务，分别对应安全（伤害、越狱、行动风险）、公平（刻板印象、排斥规范）、隐私（硬/软隐私、音频条件隐私、交互隐私、推断隐私）等，包含英中两种语言。

**📈 对比分析**

比较方法：对齐模型在 Tier 1 与 Tier 2 的直接答复率（DAR）、安全意识率（SAR）、公平率（Fair Rate）、隐私泄露率（Leakage）等指标进行对比，并与文本参考上限及人类评审对齐；结果显示：多数 SLM 在文本上表现良好，但在音频条件下安全意识、偏见中立和隐私防护显著下降，尤其是对儿童、情绪、背景声音等线索的响应不足。

**⚠️ 局限性**

局限性：①大部分语音为合成音频，真实语音中的噪声和细微线索可能导致更大失败；②Tier 2 设计的线索较为显著，未涵盖微弱或多模态隐含线索；③文本参考上限仅衡量文本级别的知识，未能代表完整的“语音上限”；④不同模型间的评测基准和推理模式差异导致结果可解释性有限。

---

## 130. Learning ultra-compressible hyperelasticity with splines: Constitutive asymmetries and non-unique representations

**arXiv ID:** 2604.14264 | [PDF](https://arxiv.org/pdf/2604.14264v1)

**作者:** Miguel Angel Moreno-Mateos `[一作]` (Friedrich-Alexander-Universität Erlangen–Nürnberg), Ellen Kuhl `[通讯]` (Stanford University)

**通讯引用:** 23297 | [OpenAlex ID](https://openalex.org/A5073356597)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于样条逼近的可自适应超压缩弹性材料模型，能够通过单一实验数据集捕捉泡沫材料在拉伸、压缩和剪切下的强张压缩不对称性。

**💡 创新点**

创新点在于将分离和非分离（耦合）能量项以乘法拆分的样条形式构造，并利用交替线性最小二乘优化实现快速训练；同时系统展示了模型在有限实验曲线下的非唯一性和识别不确定性。

**🔧 技术方法**

技术包括三维不变量（I̅₁, I̅₂, J）空间内的B样条插值、参数敏感样条表示、基于块分离的交替线性最小二乘优化，以及线性约束（单调性、凸性）保证物理可行性。

**📊 数据集**

使用的实验数据来自两种运动鞋泡沫（FF LEAP™ 与 FF TURBO™ PLUS）的单向拉伸、单向压缩和剪切测试，拉伸至1.3、压缩至0.4、剪切至0.15，并假设零侧向拉伸。

**📈 对比分析**

与传统闭式分离能量模型（仅I̅₁、I̅₂、J项）相比，加入耦合项Ψ(I̅₁,J)或Ψ(I̅₂,J)后在三种变形模式下均实现R²≈1.0，预测精度显著提升；然而，当模型过度参数化时出现多解（非唯一性），不同初始化可得到不同但同样优异的能量表达。

**⚠️ 局限性**

局限性包括：实验曲线仅覆盖单维曲线（低维子空间），导致参数识别非唯一；模型在极端变形或不同加载路径下的泛化能力未得到验证；此外，耦合项的物理可解释性有限，且未对多尺度或多相泡沫结构进行进一步验证。

---

## 131. SatBLIP: Context Understanding and Feature Identification from Satellite Imagery with Vision-Language Learning

**arXiv ID:** 2604.14373 | [PDF](https://arxiv.org/pdf/2604.14373v1)

**作者:** Xue Wu `[一作]` (University of Alabama), Jiaqi Gong `[通讯]` (University of Alabama)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用卫星图像与大型语言模型构建SatBLIP框架，预测县级社会脆弱性指数(SVI)并解释关键特征。

**💡 创新点**

将对比学习与LLM驱动的自监督字幕生成相结合，生成针对卫星语义的高质量描述，提升农村环境风险评估的可解释性。

**🔧 技术方法**

BLIP微调、CLIP文本编码、GPT‑4o字幕生成、注意力融合、SHAP解释。

**📊 数据集**

美国阿拉巴马州及其他乡村县的高分辨率卫星图像与GPT‑4o生成的结构化描述。

**📈 对比分析**

与传统手工特征、虚拟审核和自然图像预训练模型对比，县级SVI预测RMSE下降约20%，并通过SHAP验证特征可解释性。

**⚠️ 局限性**

受限于LLM生成描述的误差、训练集覆盖度不均、模型在不同乡村地貌上的泛化能力待提升。

---

## 132. CMOS-integrated superparamagnetic tunnel junction-based p-bit

**arXiv ID:** 2604.14446 | [PDF](https://arxiv.org/pdf/2604.14446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 133. Reverse-Robust Computation with Chemical Reaction Networks

**arXiv ID:** 2604.14355 | [PDF](https://arxiv.org/pdf/2604.14355v1)

**作者:** Ravi Kini `[一作]` (University of California, Davis), David Doty `[通讯]` (University of California, Davis)

**通讯引用:** 1832 | [OpenAlex ID](https://openalex.org/A5061641572)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

探讨化学反应网络（CRN）在允许逆向反应的逆向稳健计算模型下的计算能力，证明其与传统稳定计算模型等价，能够决定所有半线性集合并计算所有半线性函数。

**💡 创新点**

首次将逆向稳健性引入CRN计算理论，并证明在逆向稳健模型中仍能实现所有已知的半线性判定与计算，展示了逆向反应对稳定计算的容忍性与可逆性利用。

**🔧 技术方法**

利用系统性不变量（invariants）证明逆向反应不破坏计算结果，并借鉴现有稳定计算构造（阈值、模运算等）直接迁移到逆向稳健模型。

**📊 数据集**

本文未使用实验或仿真数据集，而是基于形式化理论证明与构造性证明。

**📈 对比分析**

无实验对比与性能评估；本文仅提供理论证明，说明逆向稳健CRN的计算能力与稳定CRN相同。

**⚠️ 局限性**

限制包括：逆向反应只能是瞬时允许（最终路径仅使用正向反应）；未给出逆向稳健性的必要与充分条件；尚未探索持续允许逆向反应的更一般计算模型。

---

## 134. Chinese Language Is Not More Efficient Than English in Vibe Coding: A Preliminary Study on Token Cost and Problem-Solving Rate

**arXiv ID:** 2604.14210 | [PDF](https://arxiv.org/pdf/2604.14210v1)

**作者:** Simiao Ren `[一作]` (Scam.ai), Ankit Raj `[通讯]` (Scam.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在SWE-bench Lite上对三款LLM（MiniMax-2.7、GPT-5.4-mini、GLM-5）进行实验，比较英中两种提示语言的token成本与任务成功率。

**💡 创新点**

系统化评估语言对token效率和成功率的影响，并通过结合token计数与期望成本分析揭示模型依赖的token效率差异，首次证明中文并非普遍更高效。

**🔧 技术方法**

使用MiniSWEAgent框架、OpenRouter API、专业中英翻译、token计数与成本计算等技术。

**📊 数据集**

使用SWE-bench Lite的50个真实软件工程任务作为实验数据集。

**📈 对比分析**

采用token使用量、成功率和期望成本三项指标进行比较，结果显示中文在部分模型上token略少但成功率下降，整体上模型选择对性能影响更大。

**⚠️ 局限性**

限制包括样本量有限（仅50个任务）、仅评估三款模型、token计数方式因API差异导致不可直接比较、实例覆盖不均、仅考虑中英两种语言、固定迭代上限等。

---

## 135. GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification

**arXiv ID:** 2604.14258 | [PDF](https://arxiv.org/pdf/2604.14258v1)

**作者:** Wangjie Gan `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 1945 | [OpenAlex ID](https://openalex.org/A5047459900)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了 Group Fine‑Tuning（GFT）来改进大模型的后训练，通过组优势学习和动态系数校正提升探索与稳定性。

**💡 创新点**

创新点在于把 SFT 视为稀疏 RL，提出组优势学习解决单路径依赖，并用动态系数校正防止梯度爆炸。

**🔧 技术方法**

使用组优势学习（GAL）与动态系数校正（DCR）相结合的 GFT 训练框架，兼顾多样性与梯度稳定。

**📊 数据集**

使用数学推理基准 NuminaMath、AMC23、College Math、Gaokao2023En、Minerva、TabMWP 等多种算术与竞赛题库。

**📈 对比分析**

与 SFT、DFT、ASFT、PSFT、GRPO 等基线对比，GFT 在多规模模型上平均提升 5–10% 分数，并在同等数据量下可匹配或超越 100k 训练的基线。

**⚠️ 局限性**

局限在于仅验证在 1–8B 规模模型、数学推理任务，开源任务与大规模 70B+ 模型的适用性未充分验证。

---

## 136. CT-VIR: Continuous-Time Visual-Inertial-Ranging Fusion for Indoor Localization with Sparse Anchors

**arXiv ID:** 2604.14545 | [PDF](https://arxiv.org/pdf/2604.14545v1)

**作者:** Yu-An Liu `[一作]` (Hefei University of Technology), Li Zhang `[通讯]` (Hefei University of Technology)

**通讯引用:** 74452 | [OpenAlex ID](https://openalex.org/A5100425554)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于三次B样条的连续时间视觉‑惯导‑测距融合框架，用于在锚点稀疏和测距受限的室内定位。

**💡 创新点**

创新点包括：利用VIO先验做测距预处理与虚拟锚点构造，以及将所有测量（视觉、惯导、物理/虚拟锚点测距）统一编码到滑动窗口的B样条因子图中，实现高精度、时序一致且计算高效。

**🔧 技术方法**

采用的技术包括：B样条连续时间轨迹表示、滑动窗口因子图优化、鲁棒异常检测（median/MAD）、虚拟锚点局部最小二乘拟合、SO(3)样条、IMU预积分、视觉重投影。

**📊 数据集**

使用的数据集：EuRoC MAV、UZH‑FPV、NTU VIRAL 以及自建的地下停车场/教室/办公室等真实实验数据。

**📈 对比分析**

与多种连续/离散时间基线（Spline‑UI、Spline‑VIO、EKF‑VIU、HCCNet、Refloc）对比，实验表明在4/3锚点配置下均取得最低ATE，平均误差分别约0.08 m、0.15 m、0.09 m，优于所有基线。

**⚠️ 局限性**

限制：虚拟锚点的误差模型不够精细，缺乏对测距偏差的显式建模；在极端NLOS或多径严重的环境下仍可能出现漂移；实时性能依赖滑动窗口大小和硬件算力。

---

## 137. The Fourth Challenge on Image Super-Resolution ($\times$4) at NTIRE 2026: Benchmark Results and Method Overview

**arXiv ID:** 2604.14558 | [PDF](https://arxiv.org/pdf/2604.14558v1)

**作者:** Zheng Chen `[一作]`, Heyan Zhangyi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织NTIRE 2026图像超分辨率（×4）挑战，提供两个评价轨道并总结参赛方法与结果

**💡 创新点**

提出统一双轨评测框架，鼓励基于预训练Transformer与生成模型的两阶段融合方案，推动失真与感知质量的平衡研究

**🔧 技术方法**

使用预训练Transformer（HAT、SwinIR等）与扩散/流模型、推理时自适应融合、结构/语义条件注入以及多任务损失

**📊 数据集**

基准数据为DIV2K与LSDIR，同时允许使用公开外部数据进行训练

**📈 对比分析**

通过PSNR和七项无参考感知指标（LPIPS、DISTS、CLIP-IQA、MANIQA、MUSIQ、NIQE）评估，SamsungAICamera在恢复轨道取得33.73 dB PSNR、在感知轨道获得4.7853分，整体领先

**⚠️ 局限性**

对预训练模型的高度依赖导致推理成本高、对真实世界降噪与压缩等复杂失真缺乏充分验证，失真–感知权衡仍未完全解决

---

## 138. Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection

**arXiv ID:** 2604.14397 | [PDF](https://arxiv.org/pdf/2604.14397v1)

**作者:** David Basil `[一作]` (University of Alberta), Grzegorz Kondrak `[通讯]` (University of Alberta)

**通讯引用:** 3306 | [OpenAlex ID](https://openalex.org/A5008747488)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将英语感义标注语料翻译成目标语言，并在句子级别投射词义，自动为目标语言生成 WordNet 风格的词义集合。

**💡 创新点**

创新点在于：① 在对齐过程中加入双语词典支持，构建三步式对齐算法 DBAlign，优先选择词典与基础对齐器一致的高置信度对齐；② 对投射出的词义采用词典过滤，排除非字面翻译，显著提升投射质量；③ 完全基于句子级别投射，避免依赖语料库级统计，从而提高对低资源语料的适用性。

**🔧 技术方法**

核心技术包括：预训练的基线对齐器、双语词典（用于对齐与过滤）、机器翻译模型（产生目标语言句子）、POS 过滤器、语义投射框架、三步式对齐与过滤流程。

**📊 数据集**

使用的主要数据集：英语感义标注语料（如 WSD 标注语料）及其对应的机器翻译句子；在评估时测试了多种目标语言（未列明具体语言，但包含至少三种多语言语料）。

**📈 对比分析**

与以往的投射方法、纯词典基线以及大语言模型基线进行了对比。实验结果表明，项目-过滤策略在多语言任务中显著提升了投射精度，同时保持了模型可解释性，并且对外部资源需求低。

**⚠️ 局限性**

限制与挑战：① 仍需要机器翻译与双语词典的支持，低资源语言的词典覆盖不足可能影响效果；② 只为每个词投射单一词义，假设词义唯一，未充分处理多义词；③ 对特殊句法或多词表达（MWEs）仍存在对齐与过滤的难点。

---

## 139. Internal Knowledge Without External Expression: Probing the Generalization Boundary of a Classical Chinese Language Model

**arXiv ID:** 2604.14180 | [PDF](https://arxiv.org/pdf/2604.14180v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 140. FoodSense: A Multisensory Food Dataset and Benchmark for Predicting Taste, Smell, Texture, and Sound from Images

**arXiv ID:** 2604.14388 | [PDF](https://arxiv.org/pdf/2604.14388v1)

**作者:** Sabab Ishraq `[一作]` (University of Central Florida), Chen Chen `[通讯]` (University of Central Florida)

**通讯引用:** 477478 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

收集了66,842个含味、香、质感、声音评分与短文本描述的食物图像数据，并训练了FoodSense‑VL模型，使其能从图片中预测并解释多感官属性。

**💡 创新点**

创新点在于将人类评分与简短描述通过教师模型扩展成视觉根植的推理解释，并采用两阶段QLoRA训练，先实现精确评分再加入解释，减少评分与生成的冲突。

**🔧 技术方法**

使用Gemma 3 27B作为基础模型进行QLoRA微调，MAmmoTH‑VL教师生成解释，AdaptLLM做幻觉过滤，并结合图像-文本对进行推理与解释训练。

**📊 数据集**

数据集为FoodSense，包含2,987张食物图像，66,842条参与者评分与描述，覆盖味、香、质感、声音四个感官维度。

**📈 对比分析**

与LLaVA、InternVL、Qwen、Food‑LLaMA等公开模型比较，FoodSense‑VL在Pearson r = 0.372、Spearman ρ = 0.360、Lin’s CCC = 0.343等指标上取得最优表现，表明跨感官推断能力最佳。

**⚠️ 局限性**

局限性包括：声音维度的推断依赖静态图像仍表现欠佳；模型对跨文化和个体差异的鲁棒性未充分验证；扩展解释的质量仍受教师与过滤器性能限制。

---

## 141. Explainable Graph Neural Networks for Interbank Contagion Surveillance: A Regulatory-Aligned Framework for the U.S. Banking Sector

**arXiv ID:** 2604.14232 | [PDF](https://arxiv.org/pdf/2604.14232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 142. Digital Guardians: The Past and The Future of Cyber-Physical Resilience

**arXiv ID:** 2604.14360 | [PDF](https://arxiv.org/pdf/2604.14360v1)

**作者:** Saurabh Bagchi `[一作]` (Purdue University), Xugui Zhou `[通讯]` (Louisiana State University)

**通讯引用:** 89 | [OpenAlex ID](https://openalex.org/A5084919916)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对网络物理系统（CPS）韧性进行了系统综述，提出了五个互联主题（系统整体韧性、学习型CPS数据挑战、验证/冗余、恢复机制、人机协同），并结合代表性应用域（自动驾驶、医疗、能源、电力网、森林火灾监测）给出实用案例与发展路线图。

**💡 创新点**

创新点在于：①将韧性视为系统整体属性并通过假设-保证合同框架整合多层组件；②聚焦学习型CPS面临的稀缺、偏倚、OOV数据问题，并探讨基础模型适配与定制化；③首次将正式验证、测试与冗余技术与恢复策略与人机因素统一起来，形成完整的韧性设计生命周期；④提出“门控器（Gatekeeper）”等在线安全验证与“深度恢复”概念；⑤强调人类角色、信任调节与可解释AI在韧性中的核心作用。

**🔧 技术方法**

所用技术包括：形式化方法（线性时序逻辑、度量时序逻辑、HyperLTL、约束契约、仿真抽象、障碍证书、控制束函数）、机器学习技术（生成式世界模型、对抗式训练、LoRA/轻量级适配器、基于情景的主动学习）、随机过程/马尔科夫决策过程、深度强化学习、在线安全监控、故障检测与冗余设计。

**📊 数据集**

没有使用单一公开数据集；文章通过案例研究（例如CATS、MCPS、美国电网、加州森林火灾监测系统）说明技术应用与效果，引用已有公开数据与实验结果，但总体以文献综述为主。

**📈 对比分析**

由于为综述论文，未进行实验对比；作者通过与现有专门研究（如Segovia‑Ferreira、Ratasich等）对比，指出本综述覆盖更广泛的系统级、数据驱动与人机协同视角，并提出未来评估标准与量化指标（如恢复延迟、系统停机时间、数据缺失率）。

**⚠️ 局限性**

局限性包括：①综述性质缺乏原型验证与大规模实验；②在多主题整合时对实际部署的可行性与成本分析不够充分；③对各主题间的具体协同机制（如约束契约与恢复算法的闭环实现）尚未给出细化设计；④在人机交互与信任调节方面仍停留在理论探讨层面，缺乏系统化实验验证；⑤在面对极端攻击或长周期环境变化时的恢复性能与可扩展性仍需进一步研究。

---

## 143. Robustness Analysis of Machine Learning Models for IoT Intrusion Detection Under Data Poisoning Attacks

**arXiv ID:** 2604.14444 | [PDF](https://arxiv.org/pdf/2604.14444v1)

**作者:** Fortunatus Aabangbio Wulnye `[一作]` (Kwame Nkrumah University of Science and Technology), Francisca Adomaa Acheampong `[通讯]` (Kwame Nkrumah University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了三大IoT数据集在四种常见数据中毒攻击下，随机森林、梯度提升机、逻辑回归和深度神经网络的鲁棒性。

**💡 创新点**

首次统一评估四种模型在三类数据集上面对四种中毒策略的表现，揭示了模型家族的脆弱性差异，并为后续抗中毒设计提供了实证依据。

**🔧 技术方法**

采用机器学习分类器、四种中毒技术（标签翻转、异常注入、特征冒充、合成异常）以及准确率、精确率、召回率、F1等评估指标。

**📊 数据集**

使用CICIoT2023、Edge‑IIoTset和N‑BaIoT三大公开IoT安全数据集。

**📈 对比分析**

通过对照清洗数据与被中毒数据的四个指标，实验发现随机森林和梯度提升机在大多数攻击下保持高于90%的准确率，而逻辑回归和深度网络在标签翻转时准确率可下降至约60%，显示其脆弱性。

**⚠️ 局限性**

实验仅涵盖了四个模型和四种攻击，未对抗性训练或实时检测方案进行验证；实验环境为离线，缺乏真实分布式/联邦学习场景，限制了结果推广。

---

## 144. Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers

**arXiv ID:** 2604.14433 | [PDF](https://arxiv.org/pdf/2604.14433v1)

**作者:** Felipe Parodi `[一作]` (University of Pennsylvania), Melanie Segado `[通讯]` (University of Pennsylvania)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5052205954)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探究在DINO系列Vision Transformer中，零向量化消除（zero‑ablation）对register tokens的影响，并提出三种可行替代方法以验证其真正功能依赖。

**💡 创新点**

发现零向量化误导性地放大了register的必要性，三种替代方案表明register的具体内容并非关键，主要起到结构缓冲作用。

**🔧 技术方法**

采用Hook‑based全层干预、mean‑substitution、Gaussian noise、cross‑image register shuffling等技术，评估对classification、retrieval、correspondence、segmentation的影响。

**📊 数据集**

使用ImageNet子集、SPair‑71k、Pascal VOC 2012等公开数据集进行下游任务评估。

**📈 对比分析**

与全模型基线相比，替代控制保持所有任务性能差异≤1个百分点；零向量化导致分类↓36.6pp、分割↓30.9pp等大幅下降。

**⚠️ 局限性**

仅在冻结特征评估下得到结论，未验证微调或其他任务下的表现，且对不同模型架构差异的因果分离仍需进一步研究。

---

## 145. Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization

**arXiv ID:** 2604.14267 | [PDF](https://arxiv.org/pdf/2604.14267v1)

**作者:** Junzhe Wang `[一作]` (Fudan University), Qi Zhang `[通讯]` (Fudan University)

**通讯引用:** 25253 | [OpenAlex ID](https://openalex.org/A5100360246)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Contribution‑Weighted Group Relative Policy Optimization（CW‑GRPO），通过LLM评判器对搜索过程中的每一轮进行检索效用与推理正确性的评估，将结果优势按贡献重新分配，实现对搜索轮次的精准信用分配。

**💡 创新点**

创新点在于将过程监督转化为优势重分配机制，避免学习不稳定的价值函数；采用LLM判断器获得二元检索与推理信号并通过与门逻辑聚焦高质量轮次；使用组内相对优势保持GRPO的稳定性，进一步提升学习信号稀疏问题。

**🔧 技术方法**

技术上基于GRPO框架，使用组内相对优势、加权软最大化与温度调节、剪切代理目标；LLM判决器（GPT‑oss‑120B）评估检索与推理；检索器采用E5，数据使用2018年维基百科；训练使用veRL、SGLang推理引擎。

**📊 数据集**

实验使用NQ、TriviaQA、PopQA（一般QA）和HotpotQA、2WikiMultiHopQA、Musique、Bamboogle（多跳QA）以及400样本的AgentGym‑SearchQA‑test作为硬案例；训练集为NQ与HotpotQA合并。

**📈 对比分析**

与Outcome‑Supervised（Search‑R1‑PPO、Search‑R1‑GRPO）和Process‑Supervised（R3‑RAG、MT‑PPO）基线及闭源模型比较；在Qwen3‑8B上整体得分31.38，比Search‑R1‑GRPO提升5.0%；在Qwen3‑1.7B提升6.3%；在一般QA与多跳QA均优于同基线，且优于大多数开源模型，但仍低于大型闭源模型。

**⚠️ 局限性**

局限性：仅对成功轨迹进行优势重分配，对失败轨迹缺乏细粒度信用；与门机制表达有限，无法捕捉更细腻的贡献；仅在轮次层面操作，无法扩展到token级监督；依赖外部LLM评判器增加推理成本，且在高容量模型上对评判器质量要求更高。

---

## 146. Scouting By Reward: VLM-TO-IRL-Driven Player Selection For Esports

**arXiv ID:** 2604.14474 | [PDF](https://arxiv.org/pdf/2604.14474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 147. Evo-MedAgent: Beyond One-Shot Diagnosis with Agents That Remember, Reflect, and Improve

**arXiv ID:** 2604.14475 | [PDF](https://arxiv.org/pdf/2604.14475v1)

**作者:** Weixiang Shen `[一作]` (Technical University of Munich), Jiazhen Pan `[通讯]` (Technical University of Munich)

**通讯引用:** 1352 | [OpenAlex ID](https://openalex.org/A5047557442)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种自演化记忆模块Evo-MedAgent，使得医学影像推理代理能够在推理过程中持续积累跨案例经验并进行反思与更新，提升对胸部X光的多项诊断任务性能。

**💡 创新点**

创新点在于将记忆分为三种存储：回顾性临床案例（episodic）、适应性程序化启发式（procedural）和工具可靠性控制（governance），并通过测试时学习实现无训练、低开销的持续改进。

**🔧 技术方法**

技术核心包括检索增强推理（RAG）、多模态反思（reflection）以及三层记忆结构的自我更新机制，结合预训练的大语言模型（如GPT-5-mini、Gemini-3 Flash）进行推理。

**📊 数据集**

使用ChestAgentBench基准，包含2,500道多选题，来自Eurorad数据库的675个专家标注胸部X光案例，覆盖七类诊断任务。

**📈 对比分析**

在工具无关和工具已集成两种设置下，Evo-MedAgent在GPT-5-mini上从0.68提升至0.79，在Gemini-3 Flash上从0.76提升至0.87；相较于传统工具增强方法，经验记忆在质性诊断任务中表现更佳。

**⚠️ 局限性**

局限包括：仅针对质性诊断有效，精准量化测量仍需专用工具；需要即时准确的反馈，现实部署中可能延迟或噪声；工具治理部分尚未充分评估；当前仅针对胸部X光，需扩展至多模态与更多疾病。

---

## 148. When Missing Becomes Structure: Intent-Preserving Policy Completion from Financial KOL Discourse

**arXiv ID:** 2604.14333 | [PDF](https://arxiv.org/pdf/2604.14333v1)

**作者:** Yuncong Liu `[一作]` (National University of Singapore), Yao Lu `[通讯]` (National University of Singapore)

**通讯引用:** 10842 | [OpenAlex ID](https://openalex.org/A5089011855)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出将金融KOL（关键意见领袖）语料视为部分交易策略，通过意图保留的策略补全实现可执行交易决策。

**💡 创新点**

创新点在于：①将KOL语料的方向性意图与执行层缺失结构分离，构建部分策略模型；②提出KICL框架，在强化学习中加入硬约束，确保补全过程不违背原有意图；③引入“背叛”指标（Unsupported Entry、Directional Reversal等）量化意图保持程度。

**🔧 技术方法**

使用强化学习（Offline RL，基于IQL的价值学习），并结合结构化KOL信号构造、基线动作生成、信号与静默两种 regime 的残差补全网络。

**📊 数据集**

构建了包含 18 名 YouTube 与 14 名 X 版金融 KOL 的多模态语料库（2022-2025 年），共 6,774 只公司（YouTube）与 3,811 只公司（X），生成约 1,229,021 条交易轨迹。

**📈 对比分析**

与传统启发式、模仿学习与常规 Offline RL（IQL、AWAC、CQL、TD3+BC）对比。KICL 在 X 平台实现最高累计收益和夏普比率，同时 Unsupported Entry 与 Directional Reversal 均为 0；在 YouTube 平台亦保持零背叛率并获得最优夏普与胜率。

**⚠️ 局限性**

局限性包括：①仅使用了基础市场特征，未充分利用更丰富的技术指标或宏观因子；②补全模型结构相对简单（基线+残差），可能无法捕捉更复杂的执行逻辑；③实验仅在过去 3 年数据上验证，未来市场环境变化时的稳健性尚未评估。

---

## 149. An unsupervised decision-support framework for multivariate biomarker analysis in athlete monitoring

**arXiv ID:** 2604.14534 | [PDF](https://arxiv.org/pdf/2604.14534v1)

**作者:** Fernando Barcelos Rosito `[一作]` (Federal University of Health Sciences of Porto Alegre), Muriel Figueredo Franco `[通讯]` (Federal University of Health Sciences of Porto Alegre)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种无监督多变量生物标志物分析框架，用于运动员监测并识别潜在风险状态

**💡 创新点**

突破传统单变量与二分类风险模型的局限，利用聚类直接从多维标志物空间发现可解释的生理状态，且能识别隐性风险表现

**🔧 技术方法**

使用Ward层次聚类、Gaussian Mixture Model (GMM) 数据增强、Z-score标准化、欧氏距离异常检测、PCA验证结构稳定性

**📊 数据集**

22名业余足球运动员真实数据（8个生物标志物、3个采样时点），并通过GMM生成290名合成运动员以检验可扩展性

**📈 对比分析**

通过轮廓系数和树结构稳定性评估选择k=3、k=5聚类；在真实数据上得到合理的生理簇分布，在合成数据中保持簇比例和PCA分布一致，显示框架在小样本和高维情境下稳健且可解释

**⚠️ 局限性**

缺乏真实伤病结局验证、合成数据的临床真实度有限、未评估不同标志物组合下的性能、对外部负荷指标整合不足

---

## 150. Three-Phase Transformer

**arXiv ID:** 2604.14430 | [PDF](https://arxiv.org/pdf/2604.14430v1)

**作者:** Mohammad R. Abu Ayyash `[一作]` `[通讯]` (Brains Build Research), Mohammad R. Abu Ayyash (Brains Build Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Three‑Phase Transformer（3PT），一种基于残差流的几何先验，通过将隐藏向量划分为N个相位并在每层引入相位感知归一化、Givens旋转、相位对齐的Grouped‑Query Attention以及在DC子空间注入固定的Gabriel’s horn实现对Transformer的结构化改造。

**💡 创新点**

创新点在于将三相电路的120°相位分布与Transformer的残差流结合，形成一个自平衡的三相等幅子空间，并通过可学习的旋转角度和固定的绝对位置侧通道共同提升表示能力，同时几乎不增加参数。

**🔧 技术方法**

使用的技术包括SwiGLU门控FFN、RMSNorm、RoPE、GQA、2D Givens旋转层、相位感知RMSNorm、相位对齐的Q/K/V头、零均值约束与Gabriel’s horn的DC注入。

**📊 数据集**

实验数据集为WikiText‑103（TinyStories用于小规模验证，WikiText‑103用于123M规模评估）。

**📈 对比分析**

与RoPE‑Only基线及多种消融进行对比，3PT在123M参数下实现PPL下降7.20%（-2.62% BPB），并在相同质量下收敛速度提升约1.93×，参数占比仅0.001%。

**⚠️ 局限性**

局限性包括对相位数N、头数对齐等超参数的敏感性、在单一语料上的评估不足、对极大规模或不同预训练任务的适用性未知，以及单种实现方式下对可迁移性的验证不足。

---

## 151. The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in Multimodal Language Models

**arXiv ID:** 2604.14363 | [PDF](https://arxiv.org/pdf/2604.14363v1)

**作者:** Akshay Paruchuri `[一作]` (Stanford University), Piotr Didyk `[通讯]` (USI Lugano)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于 K‑means 中心点替换的手段，用以量化多模态语言模型中文本与视觉特征的竞争，并通过文本中心点对比解码在推理时修正视觉感知性能。

**💡 创新点**

发现文本中心点消除对模型准确率的损失是视觉中心点的 4 倍，揭示了普遍的文本主导不平衡；同时证明文本中心点对比解码能在不重新训练的情况下，将多模态模型在视觉感知任务上的准确率提升高达 16.9%。

**🔧 技术方法**

使用 K‑means 聚类、中心点插值、对比解码、层级消融、任务细分等技术。

**📊 数据集**

主要评估基准包括 BLINK、VPBench、CV‑Bench、MMStar、MMVP、MedBLINK 等视觉感知多模态数据集。

**📈 对比分析**

在七个不同体系结构的多模态模型上进行比较，中心点替换展示了 4 倍的文本与视觉成本差异；对比解码在所有模型上平均提升 5.6%，单个任务最高提升 16.9%，显著优于传统的自回归推理或投票一致性方法。

**⚠️ 局限性**

仅在判别式多选任务上验证，尚未证明对生成任务的泛化；评估受限于 BLINK 验证集，训练方法区分不显著；中心点拟合存在小幅运行波动；对多模态系统的长期监测尚待进一步验证。

---

## 152. Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems

**arXiv ID:** 2604.14228 | [PDF](https://arxiv.org/pdf/2604.14228v1)

**作者:** Jiacheng Liu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Zhiqiang Shen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 8866 | [OpenAlex ID](https://openalex.org/A5018116732)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

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

## 153. Optimistic Policy Learning under Pessimistic Adversaries with Regret and Violation Guarantees

**arXiv ID:** 2604.14243 | [PDF](https://arxiv.org/pdf/2604.14243v1)

**作者:** Sourav Ganguly `[一作]` (New Jersey Institute Of Technology), Arnob Ghosh `[通讯]` (New Jersey Institute Of Technology)

**通讯引用:** 1125 | [OpenAlex ID](https://openalex.org/A5022713299)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于模型的鲁棒约束强化学习算法——Robust Hallucinated Constrained Upper‑Confidence RL（RHC‑UCRL），用于在存在主动对抗者的环境中学习既最优又安全的策略，并给出子线性回报与违规保证。

**💡 创新点**

创新点：①将外部因素建模为主动对抗策略，处理奖励与约束对抗者可能不一致的情况；②提出rectified penalty框架，规避传统强对偶失效；③在此框架下证明RHC‑UCRL具备子线性回报与违规保证的理论上界。

**🔧 技术方法**

技术手段：基于模型的RL，利用置信区间进行hallucination（乐观/悲观模型）；神经网络集成学习动态模型；actor‑critic 与 fitted Q‑iteration 的双重价值评估；rectified penalty 损失；信息增益 Γ_T 分析。

**📊 数据集**

实验数据集：OpenAI Gym 的 CartPole‑v1 与 Pendulum‑v1 环境，并在每一步加入对抗扰动。

**📈 对比分析**

比较方法：与无约束鲁棒UCRL（RH‑UCRL）对比。结果显示 RHC‑UCRL 在保持安全阈值（成本）下取得与 RH‑UCRL 相近甚至更高的累计奖励，同时持续满足约束；RH‑UCRL 在约束被持续违反后无法恢复。

**⚠️ 局限性**

局限性：①算法计算复杂度高，参数 λ 需要经验选择；②理论上依赖信息增益 Γ_T 的可控性，线性/非线性环境下表现未充分验证；③未在真实物理机器人或更高维连续环境中测试；④未给出零违规或更紧凑回报保证的改进方案。

---

## 154. Geometrically Consistent Multi-View Scene Generation from Freehand Sketches

**arXiv ID:** 2604.14302 | [PDF](https://arxiv.org/pdf/2604.14302v1)

**作者:** Ahmed Bourouis `[一作]` (Samsung Research), Mete Ozay `[通讯]` (Samsung Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于单张自由手绘草图直接生成几何一致的多视角场景图像的方法。

**💡 创新点**

创新点包括：①构建专门的草图到多视角数据集；②引入轻量级的相机感知注意力适配器（CA3）以注入视角几何先验；③利用稀疏对应关系监督（CSL）以显式强制跨视角几何一致性。

**🔧 技术方法**

技术主要包括：视频扩散 Transformer（Video DiT）+ LoRA域适配；PRoPE项目式位置编码；稀疏 InfoNCE 对应关系损失；以及多视角图像合成的自动化生成与筛选管线。

**📊 数据集**

使用了自研的 S2MV 数据集，包含 9,222 个草图–多视角配对样本，每个样本约 33 个视角；数据源为 FS-COCO 草图与文本提示，通过 FLUX.2、Qwen Image Edit 等工具生成逼真视角。

**📈 对比分析**

与两种两阶段基线（FLUX.2→SEVA 与 FLUX.2→ViewCrafter）对比，单阶段方法在 FID、LPIPS、CLIP‑I、Corr‑Acc 等指标上均优于基线，尤其在 FID（18.49 vs 46–48）和几何一致性（Corr‑Acc 0.199 vs 0.161/0.136）上表现显著；同时推理速度提升 42×/3.7×。

**⚠️ 局限性**

局限性包括：训练样本规模相对有限（约 8k 对）；生成的多视角图像基于编辑模型，可能继承其细微缺陷；分辨率仅为 480×480，受 GPU 记忆限制，难以生成更高分辨率视角。

---

## 155. When PCOS Meets Eating Disorders: An Explainable AI Approach to Detecting the Hidden Triple Burden

**arXiv ID:** 2604.14356 | [PDF](https://arxiv.org/pdf/2604.14356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 156. CI-CBM: Class-Incremental Concept Bottleneck Model for Interpretable Continual Learning

**arXiv ID:** 2604.14519 | [PDF](https://arxiv.org/pdf/2604.14519v1)

**作者:** Amirhosein Javadi `[一作]` (University of California San Diego), Tsui-Wei Weng `[通讯]` (University of California San Diego)

**通讯引用:** 1510 | [OpenAlex ID](https://openalex.org/A5114139431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可解释的无样本增量学习框架CI-CBM，能够在不保存旧数据的情况下持续学习新类别并保持对旧类别的记忆。

**💡 创新点**

创新点在于将概念瓶颈模型与概念正则化和伪概念生成相结合，通过知识蒸馏防止概念漂移，并使用伪特征在不同阶段保持对旧类的判别能力。

**🔧 技术方法**

技术上使用语言模型（如GPT‑3）自动生成概念，CLIP/SigLIP进行多模态特征编码，稀疏GLM预测层，概念正则化与蒸馏损失，伪特征按类中心偏移方式生成，整个框架实现了可解释的特征到类别的映射。

**📊 数据集**

在七个公开数据集上评估：CIFAR‑10、CIFAR‑100、CUB、TinyImageNet、ImageNet‑Subset、ImageNet、Places365。

**📈 对比分析**

与ICICLE、IN2、CONCIL、CLG‑CBM等现有可解释方法以及无示例黑盒方法相比，CI‑CBM平均提升约36%准确率，仅比完整回放方法低2.6%，在预训练与非预训练场景均保持优异表现。

**⚠️ 局限性**

局限性包括对高质量特征提取器的依赖，伪特征生成假设类分布均匀性可能不适用于极端分布变化，概念生成与过滤的自动化程度仍需进一步提升。

---

## 157. DharmaOCR: Specialized Small Language Models for Structured OCR that outperform Open-Source and Commercial Baselines

**arXiv ID:** 2604.14314 | [PDF](https://arxiv.org/pdf/2604.14314v1)

**作者:** Gabriel Pimenta de Freitas Cardoso `[一作]` (Dharma-AI), Paulo Henrique de Medeiros Araujo `[通讯]` (Dharma-AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了两款专用小语言模型DharmaOCR Full与Lite，用于结构化OCR，优化转写质量、生成稳定性及推理成本。

**💡 创新点**

首次将直接偏好优化（DPO）应用于OCR，显著降低文本退化率，并结合SFT与AWQ量化实现高质量低成本OCR，同时提出专属Benchmark。

**🔧 技术方法**

采用SFT、DPO、AWQ量化、LoRA、vLLM推理以及多模态Transformer架构。

**📊 数据集**

训练集为约39,680页巴西葡萄牙语文档（arXiv及内部数据），评估集为自建DharmaOCR-Benchmark（496实例，包含ESTER-Pt、Legal、BRESSAY）。

**📈 对比分析**

在DharmaOCR-Benchmark上与开源、商业OCR及多模态LLM对比，DharmaOCR Full/ Lite分别取得0.925/0.911分，退化率≤0.4%，单位成本比同类模型低约22%，位于Pareto最优。

**⚠️ 局限性**

仍存在文本字段重复问题，需后处理；缺少结构验证的RL奖励；未覆盖多域场景，需进一步细化子域专化。

---

## 158. Progressive Convex Hull Simplification

**arXiv ID:** 2604.14468 | [PDF](https://arxiv.org/pdf/2604.14468v1)

**作者:** Alec Jacobson `[一作]` (University of Toronto), Alec Jacobson `[通讯]` (University of Toronto)

**通讯引用:** 5471 | [OpenAlex ID](https://openalex.org/A5060647975)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种基于双对偶表示的贪心半空间去除算法，用于在保证外逼近（保守）的前提下，将凸包简化到指定的面数，主要目标是最小化添加的体积或表面积。

**💡 创新点**

创新点包括：①使用双对偶（将半空间映射为点）实现局部高效更新，显著降低计算复杂度；②将简化问题视为子集选择的贪心退化，利用优先队列动态评估每次去除的几何代价；③提供严格保守保证，避免了传统简化导致的空隙和误判。

**🔧 技术方法**

技术实现：双对偶映射、Chebyshev 中心求解（线性规划）、凸包算法（CGAL）、双面半边数据结构、优先队列（lazy min‑heap）、高精度有理数运算、线性代数库（Eigen）、模型 I/O（libigl）、可视化（Polyscope）等。

**📊 数据集**

主要使用了 133 个来自 threedscans.com 的三维模型进行实验，同时在球体、隐式曲面、棋子等场景中验证了方法的适用性。

**📈 对比分析**

与 Bloom 的简化-偏移、基于正切的聚类、前向选择、网格简化以及 k‑DOP 等方法对比，体积增量最低（平均 1.02×，对比 Bloom 的 1.09×），在 18 面简化下平均耗时 81 ms，最高 1 s，满足 O(n log n) 的时间复杂度。

**⚠️ 局限性**

局限性包括：①贪心子集选择是 NP‑hard 问题，易在对称或极端简化时得到次优解；②未实现并行化，无法充分利用多核；③仅针对多面体凸包，无法直接处理平滑曲面；④在极小面数时，算法可能因早期对称破坏而无法达到理论最优形状。

---

## 159. Decoupling Identity from Utility: Privacy-by-Design Frameworks for Financial Ecosystems

**arXiv ID:** 2604.14495 | [PDF](https://arxiv.org/pdf/2604.14495v1)

**作者:** Ifayoyinsola Ibikunle `[一作]` (Capital One), Mayana Pereira `[通讯]` (Capital One)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了一种基于差分隐私的代理建模（DP-Seeded ABM）框架，用于在金融领域构建隐私安全的仿真环境，并通过该框架训练和评估自主金融代理。

**💡 创新点**

创新之处在于将差分隐私从传统的表格合成迁移到动态仿真环境的参数化，形成隐私安全的“gym”，实现了在保持严格隐私保证的同时支持代理训练与公平性审计。

**🔧 技术方法**

采用了差分隐私技术对聚合统计量进行噪声注入，利用代理建模与规则驱动的仿真引擎（如MoMTSimDP）以及现有的金融仿真框架FinRL、ABIDES进行整合。

**📊 数据集**

主要使用真实金融交易日志（例如移动支付交易数据）进行聚合统计量的计算，并以此生成隐私保护的模拟环境；实验中也利用了公开的金融市场数据进行代理训练。

**📈 对比分析**

通过对比静态表格合成和DP-Seeded ABM的指标，使用边际分布的均方误差（SSE）评估合成质量，结果表明DP-Seeded ABM在保留高阶依赖和因果动态方面优于传统方法，但在长期仿真中会出现漂移。

**⚠️ 局限性**

主要限制包括：仿真递归步骤中噪声累积导致的漂移与误差放大；Sim-to-Real 的差距导致训练代理在真实环境中表现下降；缺乏全面衡量因果动态匹配的评估指标；以及在保护隐私与保持数据效用之间的权衡，特别是对少数群体的影响。

---

## 160. The Autocorrelation Blind Spot: Why 42% of Turn-Level Findings in LLM Conversation Analysis May Be Spurious

**arXiv ID:** 2604.14414 | [PDF](https://arxiv.org/pdf/2604.14414v1)

**作者:** Ferdinand M. Schessl `[一作]` `[通讯]` (Independent Researcher), Ferdinand M. Schessl (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文系统评估了多轮人机对话中常用的 66 个逐轮指标的自相关性，并提出了基于 Chelton 有效自由度与对话层块自助采样的两阶段校正框架，以纠正因自相关导致的显著性偏倚。

**💡 创新点**

创新点在于：①首次对 202 轮对话中多类指标的自相关谱进行定量描绘；②结合 Chelton 有效自由度和对话层块自助法的两阶段校正流程，提供可复现的“膨胀率”（inflation rate）诊断指标；③系统性文献综述揭示 NLP 评估文献中普遍忽视自相关的现状。

**🔧 技术方法**

采用了 Chelton 有效自由度公式、对话层块自助采样、Benjamini–Hochberg FDR、双侧 t 检验、两阶段筛选流程以及预注册的 hold‑out 验证与置换检验。

**📊 数据集**

数据集包含 202 条多轮对话，共 11,639 轮对话对，参与者为 5 名德语使用者，使用四个 LLM 平台（ChatGPT、Deepseek、Gemini、Claude），并通过 LLM‑as‑judge 方案对每轮进行操控性标签注释。

**📈 对比分析**

通过对比池化显著性与校正后显著性，发现 42% 的显著关联被“膨胀”；在 hold‑out 验证中，校正后指标的复现率为 57%，而仅池化的指标仅为 30%；效应量在 hold‑out 中略有缩小，但方向基本保持一致。

**⚠️ 局限性**

局限性包括：仅涵盖 5 名用户、单一德语语料、四个 LLM 平台；Chelton 公式假设 AR(1) 结构，对高阶自相关的累计指标可能保守；对话层块自助采样对实时监测不适用；未来需在多语种、更多平台及更大样本上验证膨胀率。

---

## 161. Stateful Evidence-Driven Retrieval-Augmented Generation with Iterative Reasoning

**arXiv ID:** 2604.14170 | [PDF](https://arxiv.org/pdf/2604.14170v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 162. Understanding Student Experiences with TLS Client Authentication

**arXiv ID:** 2604.14330 | [PDF](https://arxiv.org/pdf/2604.14330v1)

**作者:** Abubakar Sadiq Shittu `[一作]` (University of Tennessee), Scott Ruoti `[通讯]` (University of Tennessee)

**通讯引用:** 867 | [OpenAlex ID](https://openalex.org/A5090929608)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对46名计算机科学高年级和研究生进行了一项为期一个学期的纵向实验，探究了在真实环境（OpenSSL、定制CA、3072位RSA密钥）下，用户从零开始配置、使用和跨设备迁移mTLS客户端证书的可用性、学习曲线和安全认知。

**💡 创新点**

创新点在于：①首次系统量化mTLS客户端可用性，从设置、日常使用到多设备迁移的完整生命周期；②发现“浏览器/操作系统UI透明度低”是主要阻碍，而非加密复杂度；③评估技术用户对异常场景（证书/密钥丢失、泄露、迁移）的安全理解，显示即便技术用户认知不足。

**🔧 技术方法**

技术手段包括：OpenSSL命令行工具生成密钥/CSR；自建CA签发证书；NGINX + reverse proxy实现mTLS服务器；使用ASQ、SUS问卷和系统日志收集定量指标；主题分析法提炼质性反馈。

**📊 数据集**

数据集：46名学生的日志记录（CSR、握手、失败原因）与反思文本，约416份CSR、若干成功/失败握手记录；另外通过GitGuardian检测私钥泄露示例。

**📈 对比分析**

比较方法：对比首次设置与日常使用的SUS/ASQ评分、尝试次数、时间；对比单设备与双设备迁移策略的成功率与用户认知；通过统计检验（Spearman、t检验、Friedman等）评估差异。结果显示：首次设置时SUS平均仅50（“不合格”），多设备迁移时SUS仍低于60；但日常使用SUS略升至44，整体可用性仍差。

**⚠️ 局限性**

局限性包括：受教育背景高、动机强的学生群体（可能高估可用性）；缺乏对比实验（如更友好文档或工具）；实验环境单一（单一服务器、浏览器）；可能存在同伴互助或信息泄露；未测量对日常非专业用户的影响。

---

## 163. On the Doubling Dimension and the Perimeter of Geodesically Convex Sets in Fat Polygons

**arXiv ID:** 2604.14471 | [PDF](https://arxiv.org/pdf/2604.14471v1)

**作者:** Mark de Berg `[一作]` (Eindhoven University of Technology), Leonidas Theocharous `[通讯]` (University of Ottawa)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5000799434)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在具有“fatness”属性的多边形中，测地距离度量空间的双倍维度以及几何性质，并在此基础上给出了几何算法的新结果，包括基于 (α,β)-覆盖多边形的凸包周长界、近似点集（coreset）与最近点对的高效求解。

**💡 创新点**

创新点在于：
1) 证明局部 fat 多边形不一定拥有有界双倍维度，但任何 (α,β)-覆盖多边形的度量空间都具有常数上界的双倍维度；
2) 推导出 (α,β)-覆盖多边形内部任意测地凸集的周长与其欧氏直径之间的常数比例关系；
3) 利用上述性质构造线性大小的 1+近似 spanner、O(1/ε) 近似 furthest‑neighbor coreset 以及期望线性时间的最近点对算法。

**🔧 技术方法**

主要技术包括：
- 双倍维度分析与几何网格分解；
- witness‑triangle（证人三角形）与 α‑fat 条件的几何约束；
- 对测地凸集进行边界分割与方向集合的枚举；
- 对测地距离的快速查询预处理（如点到点最短路查询）；
- 随机化插入排序与格点桶技术用于最近点对。

**📊 数据集**

本文没有使用具体实验数据集，全部为理论分析与算法设计；若有实验，亦未在文中给出。

**📈 对比分析**

相较于以往针对任意多边形的结果，本文提供了更强的线性规模结构与更小的 coreset 大小：
- 1+近似 spanner 的边数从 O(m^2) 降至 O(m)；
- furthest‑neighbor coreset 由 O(1/ε^2) 降至 O(1/ε)；
- 最近点对算法从最坏情况的 O(m log m) 降至期望 O(m log n)（仅多了常数因子）。

**⚠️ 局限性**

局限性包括：
- 结果仅适用于满足 (α,β)-覆盖性质的多边形，局部 fat 多边形的情况不适用；
- 需要已知 α、β 且常数较小，实际构造可能复杂；
- 对于高度自相交或非简单多边形的推广仍有待研究；
- 近似因子与常数对实际性能影响仍需进一步评估。

---

## 164. Hierarchical Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text

**arXiv ID:** 2604.14166 | [PDF](https://arxiv.org/pdf/2604.14166v1)

**作者:** Filippo Morbiato `[一作]` (University of Padua), Luca Romano `[通讯]` (University of Padua)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 H-TechniqueRAG，一种利用 MITRE ATT&CK 战术-技术层次结构的分层检索增强生成框架，用于将 CTI 文本映射到具体技术 ID。

**💡 创新点**

创新点在于：① 两阶段层级检索，先检索战术再在对应战术下检索技术，显著缩小候选空间；② 战术感知重排序与共现先验融合；③ 层级约束的上下文组织，提升 LLM 推理精度并降低成本。

**🔧 技术方法**

技术包括 Sentence‑BERT 编码器、FAISS 向量检索、深度学习重排序模型、Llama‑3‑8B‑Instruct 生成器，以及多任务训练损失。

**📊 数据集**

使用了三大 CTI 数据集：CTI‑RCM、MITRE CTI 以及跨域 TRAM 数据集。

**📈 对比分析**

与 8 个基线（含零样本 LLM、BERT‑NER、传统规则、TagRAG、LeanRAG、TechniqueRAG 等）对比，H-TechniqueRAG 在 F1 上提升 3.8%，推理速度快 62.4%，LLM API 调用量减少 60%，跨域性能下降幅度最低（仅 4.9%）。

**⚠️ 局限性**

局限性包括：依赖 ATT&CK 层级信息，无法处理新出现或子技术；战术检索失误时仍会影响技术候选质量；模型对长文本的上下文窗口仍有限制。

---

## 165. Mind DeepResearch Technical Report

**arXiv ID:** 2604.14518 | [PDF](https://arxiv.org/pdf/2604.14518v1)

**作者:** MindDR Team `[一作]` (Li Auto Inc), Li Auto Inc `[通讯]` (Li Auto Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MindDeepResearch（MindDR），一种采用三代理（规划、DeepSearch、报告）协同的多代理深度研究框架；

**💡 创新点**

核心创新在于：① 针对不同子任务的四阶段分层训练管线（SFT、Search‑RL、Report‑RL、偏好对齐）；② 轻量级的step‑level信用分配与序列级奖励；③ 在推理阶段通过Extended Chain‑of‑Thought共享记忆并并行搜索，显著提升推理效率；

**🔧 技术方法**

技术包括：ReAct式多步骤推理、GRPO/GSPO强化学习、RACE Rubrics奖励、DPO与Self‑SFT偏好对齐、MoE 30B‑A3B/32B‑A3B 语言模型、知识图谱驱动查询合成、工具调用与检索、上下文长度自适应编码；

**📊 数据集**

数据集主要有：合成的知识图谱‑驱动多跳查询（约35K）、SFT轨迹集（约12K）、报告‑RL专用长短文本对齐数据、MindDR Bench（500条真实中文查询）及多维评测指标；

**📈 对比分析**

在BrowseComp‑ZH、BrowseComp、xbench‑DS、GAIA‑DS、WideSearch等深度搜索基准以及DeepResearch Bench和自建MindDR Bench上，MindDR‑v1.5‑30B‑A3B在30B级别模型中实现了45.7/42.8/75.0/70.9的最佳成绩，并在MindDR Bench上获得51.8的RACE总分，显著优于同参数规模开放源代码系统，且推理效率高；

**⚠️ 局限性**

局限：对极长上下文（>128K）仍存在格式正确率下降，需进一步的上下文压缩或分层记忆机制；评测主要基于RACE与规则化指标，尚缺少对方法论完整性、创新度与不确定性处理等更细粒度评估；

---

## 166. PeerPrism: Peer Evaluation Expertise vs Review-writing AI

**arXiv ID:** 2604.14513 | [PDF](https://arxiv.org/pdf/2604.14513v1)

**作者:** Soroush Sadeghian `[一作]` (Reviewerly), Ebrahim Bagheri `[通讯]` (University of Toronto)

**通讯引用:** 8384 | [OpenAlex ID](https://openalex.org/A5064660738)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个名为PeerPrism的规模化基准数据集，包含20,690篇同行评审，系统地拆分评审的思想来源（人类、AI、混合）与文本实现来源（人类、AI），用于研究LLM在学术评审中的作者属性；

**💡 创新点**

首次提出将同行评审的作者属性建模为多维度而非二元标签，并通过受控生成流程实现思想与文本来源的解耦，为研究人机协作提供细粒度评估框架；

**🔧 技术方法**

采用多种LLM（如ChatGPT、Claude、LLaMA等）进行生成和变换，结合七种主流LLM检测技术（GLTR、DetectGPT、Fast-DetectGPT、Lastde++、Binoculars、RADAR、Anchor），并进行词汇多样性、可读性、第一人称使用、语义相似度等风格与语义分析；

**📊 数据集**

数据集来源于OpenReview的ICLR和NeurIPS两大会议的674份人工评审，通过六种大型LLM生成全人工、全AI以及四种混合变换，最终构成20,690条评审记录；

**📈 对比分析**

在传统的“人工 vs AI”二分类任务上，多数检测方法表现良好；但在混合变换场景下，各检测器的预测出现显著分歧，说明它们捕捉的主要是文本风格而非思想来源；实验表明性能受模型家族、检测方法和生成策略影响，整体显示单一二分类框架不足以可靠评估人机协作；

**⚠️ 局限性**

主要局限包括：检测器过度依赖表面语言特征，忽视评审思维；二元评价不符合真实评审流程；数据集仅覆盖机器学习会议，可能无法推广到其他领域；未对检测器进行微调，缺乏针对PeerPrism的专门训练；潜在的风格与机构偏见未被完全消除。

---

## 167. Robust Optimal Experimental Design Accounting for Sensor Failure

**arXiv ID:** 2604.14497 | [PDF](https://arxiv.org/pdf/2604.14497v1)

**作者:** Rebekah White `[一作]` (Sandia National Laboratories), Timothy Walsh `[通讯]` (Sandia National Laboratories)

**通讯引用:** 20640 | [OpenAlex ID](https://openalex.org/A5030236781)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了适用于结构动力学振动实验的鲁棒最优实验设计方法，比较了经典与鲁棒设计在传感器失效与信号剪切场景下的性能。

**💡 创新点**

提出了基于松弛和双井惩罚的二进制鲁棒设计框架，并在高维候选空间与昂贵有限元模型下实现梯度优化。

**🔧 技术方法**

使用松弛优化、双井惩罚、对数行列式和均方误差评估，以及对传感器失效概率的概率/场景平均稳健目标。

**📊 数据集**

采用三层“婚礼蛋糕”结构的有限元模型，通过模拟随机载荷生成100个时间域试验，构建失效场景。

**📈 对比分析**

通过对比经典与鲁棒设计在无失效、单/多传感器失效以及剪切失效场景下的对数行列式、均方误差等指标，发现鲁棒设计在失效情况下平均性能优于经典设计，而在无失效时两者相近。

**⚠️ 局限性**

对失效概率的精确估计要求较高，且鲁棒设计计算成本随失效场景数量呈线性增长，适用性受限于模型线性与大样本近似。

---

## 168. A Nonasymptotic Theory of Gain-Dependent Error Dynamics in Behavior Cloning

**arXiv ID:** 2604.14484 | [PDF](https://arxiv.org/pdf/2604.14484v1)

**作者:** Junghoon Seo `[一作]` `[通讯]` (PIT IN Corp.), Junghoon Seo (PIT IN Corp.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究行为克隆（BC）在位置控制机器人中的闭环错误传播，提出了一个非渐近理论来量化控制器增益（比例和微分增益）对闭环失效概率的影响；

**💡 创新点**

在BC分析中首次引入增益依赖的放大指数与验证误差的乘积，给出了闭环失效概率的上界，并通过“形状保持上界”结构将多维问题简化为标量比较；在经典二阶PD系统中证明了闭环误差方差随比例增益单调递增、微分增益单调递减，统一了欠阻尼与过阻尼两种情形；

**🔧 技术方法**

利用子高斯误差传播、离散Lyapunov方程、放大指数定义、形状保持上界理论以及连续-离散一致性证明；

**📊 数据集**

采用数值仿真（单自由度 PD 系统）生成独立的高斯预测误差序列，计算离散Lyapunov解、闭环误差方差和失效率；

**📈 对比分析**

通过 Monte‑Carlo 计算四种典型增益组合（CO、SO、CU、SU）的失效率，验证理论上界与仿真结果的保守性，并展示 CO（顺从-过阻尼）在所有指标下表现最佳；

**⚠️ 局限性**

主要局限在于：仅考虑线性化的专家轨迹；假设预测误差独立且子高斯；未处理非线性接触动力学和大幅度增益变化导致的非线性效应；离散采样的上界仍受连续时间近似的约束；

---

## 169. Physics-Informed Machine Learning for Pouch Cell Temperature Estimation

**arXiv ID:** 2604.14566 | [PDF](https://arxiv.org/pdf/2604.14566v1)

**作者:** Zheng Liu `[一作]` `[通讯]` (University of Michigan-Dearborn), Zheng Liu (University of Michigan-Dearborn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了物理信息机器学习框架，用于快速准确地估计带冷板的聚合物电池包的稳态温度分布。

**💡 创新点**

创新点在于将热传导方程直接嵌入神经网络损失函数，使模型在数据稀缺时仍能遵循物理规律，并显著提升预测精度。

**🔧 技术方法**

采用了物理信息神经网络（PINN）与卷积神经网络相结合的混合模型，并加入热传导方程残差与边界条件损失。

**📊 数据集**

使用基于有限差分法生成的100组不同冷却通道几何的二维温度场数据集。

**📈 对比分析**

与纯数据驱动的全卷积网络比较，PIML模型在10轮训练后MSE降至5.66，相比11.12下降49.1%，在独立验证集上准确率更高，尤其在通道外区域。

**⚠️ 局限性**

局限性包括仅验证了二维稳态情形，且在更复杂三维或动态工况下需进一步验证与扩展。

---

## 170. MARS$^2$: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation

**arXiv ID:** 2604.14564 | [PDF](https://arxiv.org/pdf/2604.14564v1)

**作者:** Pengfei Li `[一作]` (Shanghai Artificial Intelligence Laboratory), Bowen Zhou `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 16357 | [OpenAlex ID](https://openalex.org/A5107808331)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MARS^2框架，将多代理协作与树结构搜索融合到强化学习中，以提升代码生成的质量和多样性。

**💡 创新点**

将搜索树视为可学习的多代理交互环境，并引入路径级组优势与树一致奖励塑造，实现在复杂搜索轨迹上的有效信用分配和协同学习。

**🔧 技术方法**

多代理强化学习、树结构搜索（MCTS）、Thompson采样、树一致奖励塑造、GRPO优化、代码生成与推理技术。

**📊 数据集**

使用DeepCoder训练集、LiveCodeBench评测集以及MATH数学推理集进行训练与验证。

**📈 对比分析**

与单代理GRPO、单代理树搜索RS^2以及基线模型对比，在Pass@1、Pass@1(MCTS)和Pass@N等指标上均实现显著提升，尤其在多模型系统中显著提高整体性能。

**⚠️ 局限性**

训练时序扩展导致并行度下降，增加了训练耗时；对搜索效率的进一步优化仍需探索。

---

## 171. CoCoDiff: Optimizing Collective Communications for Distributed Diffusion Transformer Inference Under Ulysses Sequence Parallelism

**arXiv ID:** 2604.14561 | [PDF](https://arxiv.org/pdf/2604.14561v1)

**作者:** Bin Ma `[一作]` (University of California Merced), Dong Li `[通讯]` (University of California Merced)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种针对分布式扩散Transformer推理的通信优化框架。

**💡 创新点**

创新点在于利用QKV处理非对称性与时间冗余，构建分层All-to-All、V-First调度和V-Major选择通信。

**🔧 技术方法**

采用Tile-Aware Parallel All-to-All (TAPA)、异步通信、RoPE、RMSNorm、缓存机制以及时间变换缓存比率等技术。

**📊 数据集**

使用同步X射线微断层扫描（mouse brain tissue）数据进行医学影像修复实验。

**📈 对比分析**

与Flat Ulysses Baseline、TAPA单独以及oneCCL等做对比，在Aurora超级计算机上实现单节点平均3.6×加速，最多8.4×，且图像质量保持相当。

**⚠️ 局限性**

局限在于对Ulysses并行头数的依赖、缓存导致的显存占用、以及在某些分辨率下无法满足序列划分条件。

---

## 172. Predicting Post-Traumatic Epilepsy from Clinical Records using Large Language Model Embeddings

**arXiv ID:** 2604.14547 | [PDF](https://arxiv.org/pdf/2604.14547v1)

**作者:** Wenhui Cui `[一作]` (University of Southern California), Richard M. Leahy `[通讯]` (University of Southern California)

**通讯引用:** 29185 | [OpenAlex ID](https://openalex.org/A5054387045)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

利用大型语言模型（LLM）提取急性临床记录文本嵌入，结合结构化表格特征，使用梯度提升树模型预测创伤后癫痫（PTE）风险。

**💡 创新点**

①在不依赖影像数据的前提下实现早期PTE预测；②采用预训练LLM作为固定特征提取器，避免过拟合；③提出模态感知融合策略，针对不同特征类型分别使用表格或文本嵌入。

**🔧 技术方法**

预训练的BioClinical‑ModernBERT等LLM生成文本嵌入；PCA降维；XGBoost梯度提升树分类器；对比零射提示（GPT‑5.2、Gemini 等）；交叉验证评估。

**📊 数据集**

TRACK‑TBI 多中心前瞻性队列，256名受试者（58例PTE，198例非PTE），仅使用第一周内收集的常规临床记录。

**📈 对比分析**

与仅表格特征、仅LLM嵌入、Naive Fusion、模态感知融合、零射提示等方法进行对比。模态感知融合取得最佳性能：AUC‑ROC 0.892±0.042，AUPRC 0.798±0.073，PPV@Recall0.5 0.905±0.122；相比之下，仅表格特征的AUC‑ROC 0.891±0.045，零射提示的AUC‑ROC 0.589±0.077。

**⚠️ 局限性**

样本量有限（PTE 较稀缺），导致模型对数据噪声和缺失敏感；模型性能受制于不同医院记录完整性和书写风格；未整合影像数据，无法评估多模态联合提升的潜力。

---

## 173. Dissecting Failure Dynamics in Large Language Model Reasoning

**arXiv ID:** 2604.14528 | [PDF](https://arxiv.org/pdf/2604.14528v1)

**作者:** Wei Zhu `[一作]` (Yunnan University), Zhiwen Tang `[通讯]` (Yunnan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并验证了一种名为GUARD的推理时机干预框架，利用模型生成轨迹中的不确定性峰值定位早期错误转折点，并通过局部短期分支和晚期终止机制修正推理路径，从而提升大型语言模型在数学与推理任务上的正确率。

**💡 创新点**

创新点在于：①将推理失败动态从整体提升转向轨迹级别的时序分析；②发现错误多在早期转折点出现且伴随Token熵尖峰；③设计自适应阈值触发局部分支（Momentum、Inhibitory、Counterfactual），并加入晚期控制避免无效延伸；④通过一次性干预而非全局扩展，实现更高效的推理改正。

**🔧 技术方法**

主要技术包括：Token级Shannon熵估计、相对量化阈值检测、短期分支生成（L=200）、分支选择依据平均熵最小化、晚期终止触发、基于分词器的分段有效性判定（oracle）等。

**📊 数据集**

实验数据集涵盖多种推理/数学/编程/知识领域：AMC、AIME、MATH500、Minerva、LiveCodeBench、OlympiadBench、GPQA、GPQA Diamond等，模型覆盖DeepSeek-R1-Distill-Qwen 1.5B/7B、QwQ-32B、Llama‑3.1‑8B‑Instruct 等。

**📈 对比分析**

与单一轨迹优化（s1、α1、Reflexion、Self‑Consistency、Self‑Refine）以及并行搜索（Best‑of‑N、Tree‑of‑Thoughts、Entro‑duction、EAGER、DTS）等基线对比，GUARD 在所有模型规模上均取得最高或第二高的 Pass@1，并显著压缩平均生成长度（如32B模型 Pass@1 71.3%仅用≈7.5k tokens），而且在大多数基线中都保持了更高的准确率。

**⚠️ 局限性**

局限性：①只分析了轨迹级别的错误动态，可能忽略更深层的推理难点；②片段有效性判定依赖外部oracle，实际应用中可能不可靠；③Token熵作为不确定性信号的单一指标或无法覆盖所有类型的推理失误；④实验主要聚焦结构化推理基准，对开放式生成或训练时集成的适用性尚未验证。

---

## 174. FreqTrack: Frequency Learning based Vision Transformer for RGB-Event Object Tracking

**arXiv ID:** 2604.14526 | [PDF](https://arxiv.org/pdf/2604.14526v1)

**作者:** Jinlin You `[一作]` (Dalian University of Technology), Xudong Zhao `[通讯]` (Dalian University of Technology)

**通讯引用:** 27130 | [OpenAlex ID](https://openalex.org/A5083349103)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于频域建模的 RGB‑Event 视觉目标跟踪框架 FreqTrack，结合 Spectral Enhancement Transformer (SET) 与 Wavelet Edge Refinement (WER) 两个模块，能在复杂场景下实现更稳健的跟踪。

**💡 创新点**

核心创新点在于：①动态频域滤波器（Dynamic Fourier Filtering）可自适应调节 RGB 与事件数据在不同频率成分上的权重；②可学习的 Haar 小波变换与动态波形滤波（Dynamic Wavelet Filtering）专门提取事件流的多尺度边缘信息，实现高频细节与低频结构的协同融合。

**🔧 技术方法**

采用 Vision Transformer 作为骨干，加入 SET 与 WER；使用 1‑D 离散傅里叶变换、离散小波变换、可学习的频域滤波器和卷积归一化；训练时使用 AdamW 优化器，StepLR 学习率调度。

**📊 数据集**

在 COESOT（827 训练序列，90+ 类别）和 FE108（108 序列）两大 RGB‑Event 跟踪基准上进行实验。

**📈 对比分析**

与 11 种现有方法对比，FreqTrack 在 COESOT 上取得 PR 76.6%（最高）和 SR 62.7%（与 FAFETrack 相近），在 FE108 上得到 PR 79.1% 与 SR 49.7%（PR 与顶尖方法相当，但 SR 略低）。总体表现优于大多数基线，显示频域融合的有效性。

**⚠️ 局限性**

局限性包括：①在 FE108 上 SR 受限，原因是数据集规模小；②频域操作虽提升精度，但对实时性能的影响需进一步评估；③目前仅针对 RGB‑Event 两模态，需探索更高效的多模态融合策略。

---

## 175. Chain of Modality: From Static Fusion to Dynamic Orchestration in Omni-MLLMs

**arXiv ID:** 2604.14520 | [PDF](https://arxiv.org/pdf/2604.14520v1)

**作者:** Ziyang Luo `[一作]` (Northwestern Polytechnical University), Junwei Han `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 39899 | [OpenAlex ID](https://openalex.org/A5012529382)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Chain of Modality（CoM）框架，通过动态规划多模态输入拓扑并根据任务复杂度切换 Plan‑Decide 或 Plan‑Reason‑Decide 两条推理路径，显著提升 Omnilingual Large Language Models 的多模态推理性能。

**💡 创新点**

创新点在于把多模态融合从固定的拼接方式转变为任务感知的动态协同，消除位置偏差和对齐陷阱，同时利用可视化规划器、推理器和决策器的三步代理流程实现自适应认知深度。

**🔧 技术方法**

采用统一的 Omni‑LLM backbone 通过系统提示（planner、reasoner、decider）实现多角色切换，结合轻量化 SFT（LoRA）和动态模态选择、交互拓扑（Parallel、Sequential、Interleaved）技术。

**📊 数据集**

在七大公开基准上验证：Music‑AVQA、AV‑Odyssey、OmniBench、DailyOmni、AV‑Counting、WorldSense、AVHBench，涵盖音频、视频、图像等多模态任务。

**📈 对比分析**

与现有专用模型、通用模型及训练‑免费框架对比，CoM 在多模态推理任务上平均提升 3‑12%（例如 Music‑AVQA +0.8% / AVHBench +3.6%），且在训练‑免费场景下仍保持竞争力。

**⚠️ 局限性**

局限性包括：对极大模型的计算开销略增，Planner 在复杂结构下可能退回默认顺序；对某些模型（如 Ola‑7B）缺乏完整的 interleaved 支持；在极高参数规模时增益相对有限。

---

## 176. CBCL: Safe Self-Extending Agent Communication

**arXiv ID:** 2604.14512 | [PDF](https://arxiv.org/pdf/2604.14512v1)

**作者:** Hugo O'Connor `[一作]` `[通讯]` (Anuna Research), Hugo O'Connor (Anuna Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并实现了CBCL语言，一个可在运行时安全扩展的代理通信协议，保证DCFL复杂度并提供验证与安全属性。

**💡 创新点**

通过homoiconic自扩展与R1–R3安全约束，使语言在扩展后仍保持DCFL并可形式化验证；将解析器形式化到Lean 4并提取可验证二进制。

**🔧 技术方法**

Lean 4形式化与证明、Rust实现、递归下降S‑expression解析、canonical serialization、gossip协议、property‑based 与差分测试、WebAssembly/CFFI等技术。

**📊 数据集**

使用内部示例方言（农业、规划、跨链资产转移等）进行验证，并未使用外部公开数据集。

**📈 对比分析**

在Apple M4上进行基准测试：单条消息解析 <400 ns，语言扩展验证约 1.8 µs，gossip 收敛 2.6 ms；差分测试确保与Lean提取二进制一致。

**⚠️ 局限性**

DCFL表达上限（无法处理递归/迭代/上下文相关模式）、信任与密钥管理外部、无方言质量/膨胀控制机制、资源模型保守且未覆盖工具后端执行层。

---

## 177. Improving Machine Learning Performance with Synthetic Augmentation

**arXiv ID:** 2604.14498 | [PDF](https://arxiv.org/pdf/2604.14498v1)

**作者:** Mel Sohm `[一作]` (University of California, Berkeley), Axel Pincon `[通讯]` (University of California, Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并评估了一种对金融机器学习中合成增强的结构化框架，利用规模匹配的空白增强和有限样本的块置换检验来区分信息增益与样本量效应。

**💡 创新点**

创新性地把合成增强视为训练分布变形，明确其偏差‑方差权衡，并通过空白增强对照与时间序列块置换检验提供了无偏的统计评估。

**🔧 技术方法**

使用了基于块自举、Gaussian/Student‑t 相关模型、VAE、DDPM、TimeGAN 等多种生成器，以及传统统计与深度学习算法（LogReg、Ridge、RF、GBM、XGBoost）。

**📊 数据集**

在两个金融数据集上实验：高频 SPY 期权交易快照（≈141M 行）和每日 5 只大盘股票面板（≈4,920 行）。

**📈 对比分析**

通过将合成增强与真实样本和规模匹配的无信息增强进行对照，并用块置换检验统计显著性；结果表明在方差主导的波动率预测任务中合成增强显著提升性能，而在近有效的方向预测和低信噪任务中则无益甚至降低性能。

**⚠️ 局限性**

局限性包括对生成器设计的依赖、对极端事件覆盖不足、对模型容量敏感，以及在稀有事件场景下检验统计与业务指标可能不一致。

---

## 178. Pushing the Limits of On-Device Streaming ASR: A Compact, High-Accuracy English Model for Low-Latency Inference

**arXiv ID:** 2604.14493 | [PDF](https://arxiv.org/pdf/2604.14493v1)

**作者:** Nenad Banfic `[一作]` (Microsoft), Meng Tang `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究并实现了基于NVIDIA Nemotron Speech Streaming的端侧低延迟、CPU‑only ASR模型，并在ONNX Runtime中完成完整流水线的重写与优化；

**💡 创新点**

通过系统化评估50+配置，提出重要性加权的k‑quant量化方法，并结合图层融合与整数算子，成功将模型从2.47 GB压缩到0.67 GB，仅提升0.17% WER；

**🔧 技术方法**

使用ONNX Runtime量化（RTN、k‑quant、int4/8）、三图拆分、缓存管理、Mel特征自实现、RNNT贪婪解码以及整数卷积/矩阵乘法算子；

**📊 数据集**

在ESB套件的八个英语基准数据集（AMI、Earnings22、GigaSpeech、LibriSpeech Clean/Other、SPGISpeech、TED‑LIUM、VoxPopuli）上进行评测；

**📈 对比分析**

采用Batch、Streaming、Chunk模式对WER、RTFx、延迟、模型尺寸进行对比，Nemotron (7,10,7) int4 k‑quant配置在CPU上实现0.56 s算法延迟、0.67 GB模型、8.20% WER，RTFx 7.3×，在相同资源下优于其他模型；

**⚠️ 局限性**

仅针对英语单语境，未覆盖多语种、逆文本归一化、说话人分离等生产需求；量化中整数算子在ConvInteger/MatMulInteger上表现不佳；评测仅在高端服务器CPU上完成，低端设备的可行性仍待验证。

---

## 179. Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors

**arXiv ID:** 2604.14563 | [PDF](https://arxiv.org/pdf/2604.14563v1)

**作者:** Mingqian Ji `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 45507 | [OpenAlex ID](https://openalex.org/A5100726984)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 SEPatch3D 框架，利用动态 Patch Size Selection、信息丰富的 Patch Selection 与跨粒度特征增强，提升 ViT‑based sparse 多视角 3D 检测的推理速度，同时保持检测精度。

**💡 创新点**

创新点包括：① 基于时空深度信息的动态 Patch Size Selection（SPSS）；② 使用熵值进行信息丰富的 Patch Selection（IPS）；③ 通过跨粒度特征增强（CGFE）将细粒度信息注入粗粒度 Patch，三者协同实现高效精度平衡。

**🔧 技术方法**

技术实现：ViT（ViT‑L / ViT‑B）骨干、StreamPETR 检测器、token 压缩（动态 Patch Embedding）、熵值重要性评分、cross‑attention 特征融合、以及多项式回归预算分析。

**📊 数据集**

实验数据集：nuScenes validation set、Argoverse 2 validation set；对比基准包括 StreamPETR、ToC3D‑faster、tgGBC 等 SOTA 方法。

**📈 对比分析**

与 SOTA 对比：SEPatch3D‑fast 在 nuScenes 上保持 61.2% NDS、52.1% mAP，推理速度比 StreamPETR 快 21%（Backbone 22%），SEPatch3D‑faster 在相同精度下速度提升 38‑57%（取决分辨率），并比 ToC3D‑faster 速度快约 20%，在 Argoverse 2 上同样实现 29‑41% 速度提升，精度基本持平。

**⚠️ 局限性**

局限性：Patch Size 选择仍基于预设阈值和启发式规则，缺少可学习的自适应机制；在极端动态场景或不同任务中可能无法得到最优粒度；未结合量化/二值化进一步加速；需更多实验验证泛化能力。

---

## 180. Material-Agnostic Zero-Shot Thermal Inference for Metal Additive Manufacturing via a Parametric PINN Framework

**arXiv ID:** 2604.14562 | [PDF](https://arxiv.org/pdf/2604.14562v1)

**作者:** Hyeonsu Lee `[一作]` (Texas A&M University), Jihoon Jeong `[通讯]` (Texas A&M University)

**通讯引用:** 11199 | [OpenAlex ID](https://openalex.org/A5065006809)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于参数化物理信息神经网络（PINN）的框架，实现了金属增材制造中零样本、材料无关的热场预测。

**💡 创新点**

创新点在于三方面：1）采用解耦的FiLM条件化架构，将时空特征与材料参数分别编码后再融合；2）利用Rosenthal解析解推导的物理引导输出缩放，消除不同材料导致的梯度失衡；3）结合Adam与L-BFGS的混合优化策略，显著提升训练稳定性与收敛速度。

**🔧 技术方法**

技术手段包括：参数化PINN、FiLM条件化、Rosenthal热源模型的输出缩放、Adam+L-BFGS混合优化、随机采样的曲率更新、以及对多材料热导率、热容量、密度的数值处理。

**📊 数据集**

实验使用基于裸板激光扫描（LPBF）的高保真有限元模拟数据，涵盖Ti‑6Al‑4V、Inconel 718、SS 316L等典型合金，并对超出训练范围的AlSi10Mg和铜进行零样本测试。

**📈 对比分析**

与非参数PINN和单一参数化PINN基线进行对比，所提框架在所有材料上平均降低了约64 % L₂误差，同时在10,000次迭代内即可达成效果，训练迭代数仅为基线的4.4 %，并在OOV材料上保持低误差和稳定性。

**⚠️ 局限性**

局限性包括：假设材料性质随温度不变、仅考虑单材料工艺、忽略相变与热流体动力学、固定工艺参数、以及统一的采样策略可能不足以捕获不同材料的高频热梯度。

---

## 181. Design and Validation of a Low-Cost Smartphone Based Fluorescence Detection Platform Compared with Conventional Microplate Readers

**arXiv ID:** 2604.14527 | [PDF](https://arxiv.org/pdf/2604.14527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 182. DVFace: Spatio-Temporal Dual-Prior Diffusion for Video Face Restoration

**arXiv ID:** 2604.14560 | [PDF](https://arxiv.org/pdf/2604.14560v1)

**作者:** Zheng Chen `[一作]` (Shanghai Jiaotong University), Yulun Zhang `[通讯]` (Shanghai Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种一阶扩散框架 DVFace，用于从低质量视频中恢复高质量人脸，利用空间-时间双码本提取先验，并通过非对称融合实现高保真细节、身份一致性和时间连贯性。

**💡 创新点**

创新点包括：①空间‑时间双码本设计，分别捕捉空间结构与时间动态；②非对称时空融合模块，时间先验做全局调制、空间先验做局部残差注入；③将一阶扩散应用于视频人脸恢复，大幅提升推理效率；④先验提取与融合的联合训练策略。

**🔧 技术方法**

使用技术包括：预训练文本到视频扩散模型 Wan2.1；VAE 编码/解码；Diffusion Transformer (DiT)；双码本（空间码本与时间码本）+ Transformer 进行先验提取；MLP+池化实现时间先验调制；跨层共享 γ/β；Warp‑based 时序一致性损失；LPIPS、Perceptual、MSE 等多种损失。

**📊 数据集**

训练使用 VFHQ 16k 高质量人脸视频；合成测试集包括 VFHQ‑Test、HDTF；真实场景测试集包括 RFV‑LQ、VoxCeleb2。

**📈 对比分析**

与 PGTFormer、KEEP、AverNet、BFVR、DicFace、SVFR 等最新方法在多项指标（PSNR、SSIM、LPIPS、DISTS、CLIP‑IQA、MUSIQ、NIQE、MANIQA、LIQE、FVD、DOVER、E*warp、VIDD）上进行对比。DVFace 在大多数指标上均取得最优或最接近最优的表现，尤其在合成数据集的 PSNR、DOVER 以及真实数据集的 CLIP‑IQA、MUSIQ、DOVER 上显著优于对比。

**⚠️ 局限性**

局限性：①依赖预训练扩散模型，极端退化或非常少量数据时性能可能下降；②对多人物、快速运动或极大视角变化的鲁棒性尚待验证；③一阶扩散在极细节纹理重建上可能不如多步方法；④训练阶段需要双阶段流程，仍较为复杂。

---

## 183. VeriGraphi: A Multi-Agent Framework of Hierarchical RTL Generation for Large Hardware Designs

**arXiv ID:** 2604.14550 | [PDF](https://arxiv.org/pdf/2604.14550v1)

**作者:** Sazzadul Islam `[一作]` (University of South Florida), Hao Zheng `[通讯]` (University of South Florida)

**通讯引用:** 918 | [OpenAlex ID](https://openalex.org/A5107003685)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 VeriGraphi 框架，将硬件规格转化为可综合 Verilog 的多代理生成流水线。

**💡 创新点**

创新点在于使用 spec‑anchored 知识图构建层次化设计架构，并以此指导递进式 RTL 生成，显著降低接口幻觉和连线错误。

**🔧 技术方法**

采用多代理 LLM（GPT‑4o‑mini/Claude Sonnet）、知识图（HDA）、逐步伪代码转 Verilog、语法检查、错误提示优化和基于规格的验证脚本。

**📊 数据集**

利用 NIST FIPS 标准（AES、DSS、HMAC）和 RISC‑V 32I 规范作为实验数据集。

**📈 对比分析**

与 Spec2RTL‑Agent 对比，VeriGraphi 在人类干预次数和迭代次数上分别低 70% 及 50%，且生成的 RTL 通过 Yosys+OpenLane 的 PPA 测试，面积、时钟和功耗均符合目标。

**⚠️ 局限性**

局限性包括对极大规模设计的可扩展性尚未验证、知识图构建仍需人工或专家审阅以及对非标准化规格文本的鲁棒性待提升。

---

## 184. WILD-SAM: Phase-Aware Expert Adaptation of SAM for Landslide Detection in Wrapped InSAR Interferograms

**arXiv ID:** 2604.14540 | [PDF](https://arxiv.org/pdf/2604.14540v1)

**作者:** Yucheng Pan `[一作]` (Wuhan University), Bin Pan `[通讯]` (Wuhan University)

**通讯引用:** 2558 | [OpenAlex ID](https://openalex.org/A5049136910)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 WILD‑SAM，一种针对包装 InSAR 相位图的参数高效微调框架，用于高精度滑坡检测。

**💡 创新点**

创新点在于：① 引入 Phase‑Aware Mixture‑of‑Experts Adapter 对冻结的 SAM 编码器进行自适应特征调制，实现谱域对齐；② 设计 Wavelet‑Guided Subband Enhancement，将高频子带解耦后作为稠密提示，显著恢复边界细节；③ 通过动态路由与多尺度卷积专家，实现对相位纹理与噪声的分离。

**🔧 技术方法**

主要技术包括：Segment Anything Model（SAM）、Mixture‑of‑Experts（MoE）与动态路由、离散小波变换（DWT）、参数高效微调（Adapter/LoRA）、Squeeze‑and‑Excitation 门以及多尺度卷积专家。

**📊 数据集**

使用的数据集有：ISSLIDE、ISSLIDE+（两大包装 InSAR 滑坡基准）以及 Hunza‑InSAR（跨区域泛化评估）。

**📈 对比分析**

与 CNN、Transformer、滑坡专用 SOTA 以及 SAM 改造模型（如 RSPrompter、MeSAM）进行比较，WILD‑SAM 在 ISSLIDE/ISSLIDE+ 上 IoU 提升至约0.75/0.90，Dice 提升至0.85/0.95，Hausdorff 距离降至 15–6，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性在于：目前仅处理单帧相位图，缺乏时间序列分析，难以区分大气屏幕等非变形噪声；未来需加入时间注意机制以提升稳健性。

---

## 185. Quantifying Cross-Query Contradictions in Multi-Query LLM Reasoning

**arXiv ID:** 2604.14525 | [PDF](https://arxiv.org/pdf/2604.14525v1)

**作者:** Rohit Kumar Salla `[一作]` (Virginia Tech), Manoj Saravanan `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多查询推理中的全局一致性，提出了保持共享前提下答案可满足的框架；

**💡 创新点**

创新点在于引入案例文件逻辑一致性概念、制定跨查询的集合级指标（SetConsRate、ContradictionDensity、RevisionCost），并设计求解器增强的最小修复机制；

**🔧 技术方法**

采用链式思维提示、求解器（SAT/SMT）校验、未满足核心定位和局部/全局修复策略的组合；

**📊 数据集**

构建了包含390个案例文件、2,450个多查询束的跨查询基准，涵盖关系/逻辑、时序/SMT、策略/规则和欠定/诱导四个领域；

**📈 对比分析**

与传统无链式、链式、Self-consistency以及基于历史的基线对比，结果显示在5个深度模型上SetCons从0.56提升至0.94，修复成本显著下降，且整体推理准确率保持不降；

**⚠️ 局限性**

局限包括提取噪声导致的错误、求解器依赖与性能敏感、对更复杂逻辑（概率、模态）扩展尚待实现。

---

## 186. Perspective on Bias in Biomedical AI: Preventing Downstream Healthcare Disparities

**arXiv ID:** 2604.14514 | [PDF](https://arxiv.org/pdf/2604.14514v1)

**作者:** Michal Rosen-Zvi `[一作]` (IBM Research), Mordechai Muszkat `[通讯]` (Hebrew University)

**通讯引用:** 3351 | [OpenAlex ID](https://openalex.org/A5083245877)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性分析近10年基因组、转录组、蛋白组、表观组及微生物组文献中人口学信息缺失的现状，并指出该缺失会导致基础模型产生早期偏差

**💡 创新点**

首次将自动化文本挖掘与多学科数据来源相结合，提出 Provenance‑Openness‑Evaluation Transparency 三大原则来预防基础模型的系统性不公平

**🔧 技术方法**

利用自然语言处理自动提取文献人口学变量、XAI（SHAP、注意力机制等）评估模型偏差以及基准测试框架评估不同族群下的模型表现

**📊 数据集**

主要使用 PubMed 索引的4719篇 omics 论文、CellxGene、GEO 等公共数据库及多种基础模型（BioGPT、AlphaFold、scBERT 等）作为数据来源

**📈 对比分析**

通过对比不同领域报告率及模型在多族群数据集上的表现，发现欧洲裔数据占比过高，建议构建新的多族群基准测试以量化公平性；虽然未给出具体性能指标，但框架可用于持续评估

**⚠️ 局限性**

依赖自动文本提取，可能忽略细节与隐藏信息；仅覆盖已发表的摘要；未进行实验验证模型改进的实际效果；对资源受限的研究者提出的解决方案实现成本较高

---

## 187. Co-distilled attention guided masked image modeling with noisy teacher for self-supervised learning on medical images

**arXiv ID:** 2604.14506 | [PDF](https://arxiv.org/pdf/2604.14506v1)

**作者:** Jue Jiang `[一作]` (Memorial Sloan Kettering Cancer Center), Harini Veeraraghavan `[通讯]` (Memorial Sloan Kettering Cancer Center)

**通讯引用:** 7351 | [OpenAlex ID](https://openalex.org/A5014597008)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 DAGMaN 框架，在 Swin Transformer 上结合语义注意力模块和噪声教师共蒸馏，改进医学图像的掩码自监督预训练，并在肺结节分类、分割、免疫治疗预后预测以及无监督器官聚类等任务中进行评估。

**💡 创新点**

创新点包括：①在 Swin 中引入可视化全局注意力的语义注意力（SA）模块；②采用 patch dropout 的噪声教师提升注意力头多样性；③将 SA 与噪声教师共蒸馏相结合，实现高效的注意力引导掩码并保持多样性。

**🔧 技术方法**

使用技术包括：Swin Transformer、语义注意力模块、共蒸馏（EMA 教师）、噪声教师（patch dropout）、注意力引导掩码、AITD/AMPD/GITD/AMIP 损失、线性探测、全微调等。

**📊 数据集**

数据集：10,412 个无标签 3D CT 进行预训练；LIDC、公共与机构数据共 4,746 例用于肺结节分类、分割、免疫治疗预后预测和无监督器官聚类。

**📈 对比分析**

与 SMIT、iBot、AttMask、MST、nnU-Net、3D‑ResNet‑50 等方法比较，DAGMaN 在图像级分类（AUC 提升）、分割（DSC 提升）以及器官聚类（UMAP 交叉距离最大化）等多项任务上均实现了显著性能提升，尤其在少样本（25%–50%）场景中表现最优。

**⚠️ 局限性**

局限性包括：对纵隔或与收缩肺融合的肿瘤分割效果不佳；对其他 Transformer 架构（非 Swin/ViT）的适用性尚未充分验证；在极少量样本下仍需进一步研究优化；以及模型可解释性仍需更系统的评估。

---

## 188. Seeing Through Circuits: Faithful Mechanistic Interpretability for Vision Transformers

**arXiv ID:** 2604.14477 | [PDF](https://arxiv.org/pdf/2604.14477v1)

**作者:** Nina Żukowska `[一作]` (Max Planck Institute for Informatics), Jonas Fischer `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 902 | [OpenAlex ID](https://openalex.org/A5007776675)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 Vision Transformer 中首次实现基于计算图的稀疏可信电路发现，并通过电路干预抵御文本攻击。

**💡 创新点**

创新点：将语言模型中的边缘电路发现迁移至视觉模型；实现约10倍更稀疏且保持高准确率的可信电路；展示电路可用于主动模型调控。

**🔧 技术方法**

采用残差流图建模、注意力输入节点简化、激活补丁（patching）、分割+填充生成腐败输入、序列修剪和目标对数差等技术。

**📊 数据集**

使用 ImageNet（ViT‑B）、OpenCLIP ViT‑B/32、ForAug（分割+填充图像）以及针对 typographic attack 的定制数据。

**📈 对比分析**

与 EAP、EAP‑IG 等梯度近似基线对比：在10% 边数时即可恢复接近完整准确率，优于基线；在攻击防御实验中将攻击成功率从约40% 降至 2–3%，且干预对正常准确率影响极小。

**⚠️ 局限性**

局限性：需要对每条边执行激活补丁，计算成本较高；方法主要针对分类任务，尚未推广到检索、分割等更复杂视觉任务；电路发现结果呈多重分布，缺乏唯一性保证。

---

## 189. CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge

**arXiv ID:** 2604.14644 | [PDF](https://arxiv.org/pdf/2604.14644v1)

**作者:** Seyun Bae `[一作]` (KT Corporation), Eunho Yang `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种连续实时无学习方法CURaTE，利用预训练句子嵌入模型在部署前对合成数据进行对比学习，实时处理遗忘请求而不修改LLM参数；

**💡 创新点**

创新点在于将无学习视为行为层面而非参数层面，使用无遗忘集的全局嵌入模型与硬负样本，实现在部署后对连续遗忘请求的即时响应并保持知识保留；

**🔧 技术方法**

技术包括：句子嵌入对比学习、三类合成数据（正样本+两类硬负样本）、余弦相似度阈值决策、无参数更新的实时检索；

**📊 数据集**

使用四个公开基准：RETURN（隐私数据）、TOFU（虚构作者）、TruthfulQA（错误信息）和ScienceQA（科学知识），并在每个基准上构造同义变体及“近似效用”样本；

**📈 对比分析**

与梯度上升、GradDiff、PO、NPO、SO-PO、GUARD、O3、UniErase等方法对比，CURaTE在所有基准上显著提升遗忘效果（降低忘却集分数），同时几乎不降低保留集和通用知识的性能，且在每阶段平均遗忘时间仅0.04s、推理延迟0.01s，速度远快于其他方法；

**⚠️ 局限性**

局限在于无法保证完全消除泄露风险（阈值调优仍需折衷）、未充分测试对抗性绕过与大规模遗忘请求的极端场景、缺乏对LLM模型不同规模的泛化验证。

---

## 190. CMTM: Cross-Modal Token Modulation for Unsupervised Video Object Segmentation

**arXiv ID:** 2604.14630 | [PDF](https://arxiv.org/pdf/2604.14630v1)

**作者:** Inseok Jeon `[一作]` (Yonsei University), Sangyoun Lee `[通讯]` (Yonsei University)

**通讯引用:** 3441 | [OpenAlex ID](https://openalex.org/A5015739530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种跨模态令牌调制（CMTM）框架，利用密集Transformer块和令牌掩码机制在无监督视频目标分割中实现外观与运动信息的深度融合。

**💡 创新点**

创新点在于：1）通过密集Transformer实现全局的同模态与跨模态关系建模；2）引入随机令牌掩码策略，使模型在高复杂度下仍能高效学习并抑制过拟合。

**🔧 技术方法**

主要技术包括：双流编码器（RGB+光流）、密集Transformer块、令牌掩码、MiT‑b2骨干、交叉熵损失与Adam优化。

**📊 数据集**

使用的数据集：预训练集YouTube‑VOS 2018，微调集DAVIS 2016与DUTSv2；评估集包括DAVIS 2016、FBMS、YouTube‑Objects与Long‑Videos。

**📈 对比分析**

与FakeFlow以及多种现有方法对比，CMTM在J、F、G三项指标上均取得最高分（如DAVIS 2016 G=89.2%、J=88.5%、F=89.8%），同时保持了较快的推理速度。

**⚠️ 局限性**

局限性在于：①依赖密集Transformer导致计算量较大；②掩码比例需要经验调优；③在极端遮挡或长视频场景下仍可能出现细节失真。

---

## 191. Asking What Matters: Reward-Driven Clarification for Software Engineering Tasks

**arXiv ID:** 2604.14624 | [PDF](https://arxiv.org/pdf/2604.14624v1)

**作者:** Sanidhya Vijayvargiya `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (Carnegie Mellon University)

**通讯引用:** 21813 | [OpenAlex ID](https://openalex.org/A5068811427)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究软件工程任务中的澄清问答，构建了基于信息重要性与可答性两大维度的澄清模块CLARITI；

**💡 创新点**

通过实证 Shapley 分析确定关键信息层级，分布式分析提炼可答性特征，并将两者融合为多阶段奖励管道，显著提升澄清效率；

**🔧 技术方法**

使用 Shapley 价值、Vargha–Delaney 效果量、强化学习（GRPO）与多阶段奖励、OpenHands 代理框架、GPT‑5/5‑nano 评判器；

**📊 数据集**

SWE‑Bench Verified、SWE‑Gym Raw、SWE‑Bench 重新生成的 1500 个不完整 issue 以及 700 个样本进行信息影响评估；

**📈 对比分析**

与无澄清基线、GPT‑5 nano、GPT‑5 进行对比；CLARITI 在任务成功率上与 GPT‑5 接近（36.8% vs 35.6%），但平均提问量减少 41%（3.0 题 vs 5.1 题），实现 88% 的全指定性能；

**⚠️ 局限性**

局限：单轮澄清设计不考虑多轮交互，数据集主要聚焦软件工程，评判器为 GPT‑5 可能与真实用户偏差；

---

## 192. Retrieve, Then Classify: Corpus-Grounded Automation of Clinical Value Set Authoring

**arXiv ID:** 2604.14616 | [PDF](https://arxiv.org/pdf/2604.14616v1)

**作者:** Sumit Mukherjee `[一作]` (Oracle Health Data Intelligence), Chris Sidey-Gibbons `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出检索增强集完成 (RASC) 框架，通过检索相似价值集形成候选池并用分类器筛选，解决临床价值集编写的可扩展性瓶颈。

**💡 创新点**

创新点在于将检索与二分类相结合，将原本需在庞大词表中全局生成的任务转化为小规模候选集分类；理论上样本复杂度从 O(log N) 降到 O(log K)；并首次构建并公开 11,803 个 VSAC 价值集的基准数据集。

**🔧 技术方法**

使用 SAPBert 进行语义检索，结合 LightGBM、三层 MLP 与跨编码器（cross‑encoder）三种分类器；同时将 GPT‑4o 零样本生成作为对照。

**📊 数据集**

使用 Value Set Authority Center (VSAC) 公共价值集数据集，包含 11,803 个价值集，涵盖 15 个术语系统和 847 个发布者。

**📈 对比分析**

通过与检索仅、GPT‑4o 零样本生成以及三种分类器进行对比；结果显示跨编码器在值集级 F1 达到 0.298，显著优于 GPT‑4o 的 0.105；检索仅精度仅 0.092，分类器明显提升精度和召回。

**⚠️ 局限性**

主要局限包括检索覆盖不足导致的误差下限、LLM 对结构化代码记忆能力不足、检索语料的高质量假设，以及在候选池规模大时 LLM-as‑Classifier 的成本与可扩展性问题。

---

## 193. Uncertainty-aware Generative Learning Path Recommendation with Cognition-Adaptive Diffusion

**arXiv ID:** 2604.14613 | [PDF](https://arxiv.org/pdf/2604.14613v1)

**作者:** Xiangrui Xiong `[一作]` (Xihua University), Yanli Lee `[通讯]` (Xihua University)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5058899052)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了 U-GLAD 框架，用于学习路径推荐，解决历史交互不确定性和学习目标适应性不足的问题。

**💡 创新点**

创新点在于将认知状态建模为高斯分布的 LSTM、目标导向的概念编码器以及基于扩散模型的生成式解码器，有效融合不确定性处理与生成式路径规划。

**🔧 技术方法**

主要技术包括 Gaussian LSTM 进行状态去噪、Self‑Attention 目标导向编码、扩散模型逆过程预测下一概念的潜在表示，以及强化学习与噪声预测损失的联合训练。

**📊 数据集**

实验使用了三大公开教育数据集：Junyi、SLP‑Physics 与 ASSISTments09，分别包含数千名学生和上千概念，序列平均长度约 50。

**📈 对比分析**

与 DLPR、SRC 和 LIGHT 等现有三大基线相比，U-GLAD 在所有路径长度（10、20、30）和数据集上均取得最高的学习效果提升率（E_T），例如在 SLP‑Physics 路径长度 30 时达 0.9348，超出 LIGHT 约 2.1%。

**⚠️ 局限性**

局限性主要体现在对稀疏数据集的扩散迭代敏感度较高，需进一步探索动态调整 T 和 λ 的机制；此外，模型对计算资源的需求较高，实时推荐场景的部署尚需优化。

---

## 194. Behavior-Aware Dual-Channel Preference Learning for Heterogeneous Sequential Recommendation

**arXiv ID:** 2604.14581 | [PDF](https://arxiv.org/pdf/2604.14581v1)

**作者:** Jing Xiao `[一作]` (Shenzhen University), Zhong Ming `[通讯]` (Shenzhen University)

**通讯引用:** 13201 | [OpenAlex ID](https://openalex.org/A5100633973)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计了行为感知双通道偏好学习框架BDPL，用以解决异构顺序推荐中的行为稀疏与多行为噪声问题。

**💡 创新点**

创新点在于构造多行为子图并采用目标行为引导的级联图卷积；同时对长短期偏好分别构造对比学习任务并融合。

**🔧 技术方法**

主要技术包括行为感知图卷积网络、双通道自注意力编码、对比学习增强、门控融合以及多行为子图构建。

**📊 数据集**

在Tmall、UB和JD三个电商数据集上进行实验。

**📈 对比分析**

与15种基线（含RNN、CNN、Transformer、GNN和对比学习方法）对比，BDPL在HR和NDCG指标上均超过最强基线，提升幅度最高可达约17.6%。

**⚠️ 局限性**

局限性在于仍需手工设计子图和行为边类型，且对极端稀疏或多模态数据的泛化性未充分验证。

---

## 195. TurboTalk: Progressive Distillation for One-Step Audio-Driven Talking Avatar Generation

**arXiv ID:** 2604.14580 | [PDF](https://arxiv.org/pdf/2604.14580v1)

**作者:** Xiangyu Liu `[一作]` (MAIS Institute of Automation Chinese Academy of Sciences), Xiangyu Zhu `[通讯]` (MAIS Institute of Automation Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 TurboTalk 两阶段进化蒸馏框架，将多步音频驱动头像视频扩散模型压缩为单步生成器，显著提升推理速度；

**💡 创新点**

核心创新在于先用分布匹配蒸馏得到稳定四步学生，再通过进化对抗蒸馏、动态时间步采样和自对比正则化实现一步生成，解决一步蒸馏训练不稳定和质量崩塌；

**🔧 技术方法**

采用分布匹配蒸馏 (DMD)、R3GAN 对抗蒸馏、动态时间步采样、Self-Compare 监督、音频交叉注意与 DiT 视觉扩散骨干；

**📊 数据集**

训练使用约2000小时单人对话视频+200K多事件视频，评测采用 HDTF、CelebV‑HQ、EMTD 等公开基准；

**📈 对比分析**

与 InfiniteTalk、Wan2.2‑S2V、LiveAvatar、SoulX‑FlashTalk 等多步与四步蒸馏基线对比，指标为 FID、FVD、E‑FID、Sync‑C/D；TurboTalk 在 1‑NFE 下实现 120× 速度提升，视觉质量与 4‑NFE 相当，音视频同步与表情表达均优于或与现有方法持平；

**⚠️ 局限性**

仍需大规模显存与多 GPU 训练，极低步情况下细节缺失可能出现，对多说话人实时交互与可解释性研究不足。

---

## 196. Generative Augmented Inference

**arXiv ID:** 2604.14575 | [PDF](https://arxiv.org/pdf/2604.14575v1)

**作者:** Cheng Lu `[一作]`, Heng Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出Generative Augmented Inference（GAI）框架，将 AI 生成的输出视为信息性特征而非代理标签，用以估计受限人类标注的数据的参数。

**💡 创新点**

核心创新是通过 Neyman 正交刻画将 AI 表示嵌入估计过程，既能在预测准确性不佳或信息类型不同的情况下保持无偏推断，又在任何 AI 输出具备预测信息时实现严格或至少弱的效率提升（“安全默认”）。

**🔧 技术方法**

采用交叉拟合（cross‑fitting）、灵活机器学习（如随机森林、梯度提升、核方法、神经网络）估计需求的期望和标注概率，再构建正交得分函数并求解 GLM 参数，最后用闭式方差估计推导置信区间。

**📊 数据集**

三个真实业务数据集：①疫苗选择实验（LLM 生成的 3072 维文本嵌入和离散预测）；②零售价格实验（数字孪生的二元购买预测）；③加州健康保险普查（梯度提升生成的 85% 准确概率预测）。

**📈 对比分析**

与四种基线（仅人类标签、直接拼接、PPI、PPI++）比较，GAI 在所有应用中均显著降低均方百分比误差（MAPE）30‑50%，提升置信区间覆盖率（≥95%），并将所需人类标注量降低 67‑90%；同时保持或缩短区间宽度，决策误差率最低。

**⚠️ 局限性**

主要限制：1）“安全默认”仅在标注采样为随机或仅受可观测特征影响时成立；对策略性或非随机标注可能失效；2）需先验估计标注概率和期望，若样本量极小或 AI 表示过于噪声可能导致数值不稳定；3）算法在大规模数据和实时决策场景下的计算开销尚待优化。

---

## 197. Chaotic CNN for Limited Data Image Classification

**arXiv ID:** 2604.14645 | [PDF](https://arxiv.org/pdf/2604.14645v1)

**作者:** Anusree M `[一作]` (Amrita Vishwa Vidyapeetham), Pramod P Nair `[通讯]` (Amrita Vishwa Vidyapeetham)

**通讯引用:** 38 | [OpenAlex ID](https://openalex.org/A5051479917)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在CNN特征空间引入基于混沌映射的非线性变换（logistic、skew tent、sine），在有限数据情形下提升分类性能。

**💡 创新点**

创新点在于无需额外可学习参数、可直接嵌入任意CNN结构的混沌特征变换层，且三种混沌映射在低样本下均能显著提升表现。

**🔧 技术方法**

使用的技术包括卷积神经网络、归一化特征后应用混沌映射、宏F1评估、网格搜索+5折交叉验证调参。

**📊 数据集**

实验数据集为灰度图像MNIST、Fashion-MNIST，以及彩色图像CIFAR-10。

**📈 对比分析**

通过与标准无混沌层CNN（SA）在不同样本数/网络深度下的宏F1对比，低样本下可提升3%–9%（例如MNIST 3层+40样本提升5.43%，Fashion-MNIST 3层+50样本提升9.11%，CIFAR-10 200样本提升7.47%）。

**⚠️ 局限性**

局限性包括：样本增多时提升逐渐减弱；性能依赖混沌参数的选择；缺乏对混沌变换对特征表征和决策机制的解释性分析。

---

## 198. Learning to Draw ASCII Improves Spatial Reasoning in Language Models

**arXiv ID:** 2604.14641 | [PDF](https://arxiv.org/pdf/2604.14641v1)

**作者:** Shiyuan Huang `[一作]` (University of California), Leilani H. Gilpin `[通讯]` (University of California)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出利用大型语言模型学习生成并理解 ASCII 网格布局，以提升对空间推理任务的能力。

**💡 创新点**

创新点在于引入可验证的 ASCII 布局作为中间表示，揭示了读取‑写作的不对称性，并通过布局构建训练显著提升空间理解与跨任务迁移。

**🔧 技术方法**

使用 LLM 的生成与理解能力，结合 LoRA 微调和双向 ASCII‑描述映射技术实现模型的布局构建与推理。

**📊 数据集**

构建了全新的 Text2Space 数据集，包含自然语言描述、ASCII 网格、图像以及问答对，用于训练与评估。

**📈 对比分析**

与多种 LLM（如 Qwen3、Llama3、GPT‑4 等）以及外部基准（StepGame、bAbI、SpartQA）比较，构建训练提升了约 7% 的空间推理精度，外部基准则提升高达 43%。

**⚠️ 局限性**

研究仅针对离散 ASCII 网格，未覆盖连续几何、3D 空间或视觉感知等更复杂的空间环境。

---

## 199. Touching Space: Accessible Map Exploration Through Conversational Audio-Haptic Interaction

**arXiv ID:** 2604.14637 | [PDF](https://arxiv.org/pdf/2604.14637v1)

**作者:** Li Liu `[一作]` (University of California Santa Cruz), Leilani H. Gilpin `[通讯]` (University of California Santa Cruz)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一套名为 Touching Space 的音频-触觉地图系统，支持盲人/低视力用户在出行前通过触摸探索空间并用对话式 AI 进行开放式空间查询，帮助构建认知地图。

**💡 创新点**

创新点在于将连续触觉反馈与大模型对话代理相结合，利用多模态提示（地图截图、空间方向与距离、对话历史）实现实时、基于手势的空间理解与交互。

**🔧 技术方法**

技术包括 macOS SwiftUI/MapKit、Core Haptics、Apple Speech 与 TTS、Overpass API + OSMnx 处理 OSM 数据、Gemini Live/Flash 及 Qwen3-VL 视觉‑语言模型，以及多模态提示工程。

**📊 数据集**

使用的主要数据集为 OpenStreetMap（通过 Overpass API 提取的 400 m 半径地理数据），并在前端生成多边形覆盖层作为触摸区域。

**📈 对比分析**

通过内部对比两种对话管线（语音流式 vs 文本中介）评估了延迟与答案结构，文本管线稳定但语音更自然；但尚未在标准评测基准上进行量化比较，缺乏客观性能指标。

**⚠️ 局限性**

局限性包括：对话代理仍可能产生幻觉/错误方向或距离；缺乏外部检索或事实核查机制；未在真实盲人/低视力用户中进行实验评估；以及未验证预旅行学习成果能否迁移到实际导航。

---

## 200. Pushing the Boundaries of Multiple Choice Evaluation to One Hundred Options

**arXiv ID:** 2604.14634 | [PDF](https://arxiv.org/pdf/2604.14634v1)

**作者:** Nahyun Lee `[一作]` (Chung-Ang University), Guijin Son `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种大规模选项（N=100）评估框架，用于严苛测试大型语言模型在密集干扰下的可靠性，尤其针对韩语正字法错误检测任务。

**💡 创新点**

创新点在于：1）将多项选择题扩展到100个候选选项，显著降低随机猜测基线；2）引入“Bubble Index”衡量低选项与高选项性能差距；3）通过“Positional Fallback”诊断模型在高干扰下的顺序偏差；4）利用填充对比实验分离上下文长度与候选排序难度。

**🔧 技术方法**

采用大规模随机采样、同步种子控制、Monte Carlo 1,000次试验、偏差诊断（BI、PFI、Entropy）以及加权线性回归分析模型对位置敏感性的依赖。

**📊 数据集**

使用从韩国国家语言研究所(NIKL)抽取的750句韩语文本，并通过三款拼写检查器（Daum、Saramin、Nara）严格一致性判定生成30个目标错误及其候选错误集。

**📈 对比分析**

与现有低选项评测对比，结果显示在N=100时，部分模型（如EXAONE-4.0、HyperCLOVAX-Think）准确率从近乎完美降至约70%或更低，而强模型（Gemini系列）保持高于90%的准确率；同时BI指标揭示低选项的性能膨胀现象，位置诊断表明弱模型更易出现首选偏差。

**⚠️ 局限性**

局限性包括：1）仅针对韩语正字法错误单一任务，缺乏跨任务验证；2）目标集合固定、对错误边界严格过滤，可能导致标签偏差；3）评估模型覆盖面有限，未包含所有最新大模型；4）填充对比实验虽然排除长度影响，但未完全覆盖所有上下文分布变化。

---

## 201. High-Speed Full-Color HDR Imaging via Unwrapping Modulo-Encoded Spike Streams

**arXiv ID:** 2604.14632 | [PDF](https://arxiv.org/pdf/2604.14632v1)

**作者:** Chu Zhou `[一作]` (National Institute of Informatics), Imari Sato `[通讯]` (National Institute of Informatics)

**通讯引用:** 2393 | [OpenAlex ID](https://openalex.org/A5101052713)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套完整的基于模数相机的高速全彩HDR成像系统，实现了1000 FPS HDR视频捕捉。

**💡 创新点**

创新点在于曝光解耦的模数成像模型、无迭代的扩散+物理一致性未包装算法，以及通过脉冲摄像机实现的70%带宽压缩。

**🔧 技术方法**

采用曝光解耦的模数成像公式、扩散模型（Stable Diffusion）、PMF‑Adapter、LMA‑Decoder、CCP‑Refiner、脉冲摄像机与非Bayer色彩采样技术。

**📊 数据集**

在UnModNet数据集进行合成评估，并使用真实Spike M1K40‑H2‑Gen3硬件采集的实验数据。

**📈 对比分析**

与UnModNet、PnP‑UA、IntrinsicHDR、LEDiff等方法对比，合成数据上PSNR‑L/SSIM‑L、HDR‑VDP‑3等指标均优越，推理时间约0.27 s/帧；在真实硬件上与UnModNet、PnP‑UA、SY24对比，视觉质量显著提升，带宽仅6 Gbps。

**⚠️ 局限性**

局限在于仅支持非Bayer色彩采样导致空间分辨率降低，未解决Bayer阵列的模数解包与色彩重建问题。

---

## 202. StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation

**arXiv ID:** 2604.14631 | [PDF](https://arxiv.org/pdf/2604.14631v1)

**作者:** Geonhui Jang `[一作]` (Chung Ang University), YoungJoon Yoo `[通讯]` (Chung Ang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 StoryCoder 框架，将代码生成任务重新表述为结构化叙事，以提升 LLM 的推理和计划能力。

**💡 创新点**

创新点在于将问题拆解为任务概述、约束和示例 I/O 三部分的叙事，并通过选定算法类别和叙事体裁来引导 LLM 形成更一致、可解释的思路；同时证明叙事对算法选择、实现质量和模块化都有积极影响。

**🔧 技术方法**

使用大型语言模型进行叙事生成（f_narr）和代码生成（f_solve），结合跨模型、Self-solving 场景；实验基于 11 种闭源/开源模型；采用 Pass@k、AST 结构分析等评测技术。

**📊 数据集**

在 HumanEval、LiveCodeBench、CodeForces 三大编码基准上进行实验，分别包含 105、175、265 条题目。

**📈 对比分析**

相较于重复采样、Chain-of-Thought、Structured CoT 等基线，StoryCoder 在三大基准上平均提升 18.7% 的零样本 Pass@10；在更难基准上提升更显著；并在算法一致性、实现错误率及代码模块化等指标上均有改善。

**⚠️ 局限性**

局限性包括：叙事生成效果高度依赖模型的指令遵循能力，对简单任务提升有限；方法仅验证于结构化编程问题，尚未扩展至开放式软件工程或大型仓库任务；缺乏人类评估和软件质量指标。

---

## 203. A Stable SBP-SAT FDTD Subgridding Method Without Region Split

**arXiv ID:** 2604.14618 | [PDF](https://arxiv.org/pdf/2604.14618v1)

**作者:** Yuhui Wang `[一作]` (Beihang University), Shunchuan Yang `[通讯]` (Beihang University)

**通讯引用:** 3647 | [OpenAlex ID](https://openalex.org/A5107917635)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种不需要分块的SBP-SAT FDTD子网格方法，直接将细网格嵌入粗网格，保证长期稳定性。

**💡 创新点**

创新点在于设计投影SBP算子和满足规范兼容性的插值矩阵，消除外域分块，仅使用四个SAT接口，实现长时稳定且对任意网格比率可用。

**🔧 技术方法**

采用SBP（求和-乘积）框架、SAT（模拟逼近项）技术、投影算子、规范兼容插值矩阵、离散能量分析以及MATLAB实现的并行矩阵运算。

**📊 数据集**

使用多种仿真模型：PEC腔、波导、双C‑SRR阵列以及人头二维模型，涵盖均匀粗细网格、细网格嵌入以及多材料介电参数。

**📈 对比分析**

与传统全细网格、粗网格、对齐块和T‑接头SBP‑SAT子网格方法对比，实验显示速度提升约19–52倍，误差维持在1–6%以内，尤其在1:5、1:10网格比下与参考解收敛一致。

**⚠️ 局限性**

目前仅在二维问题上验证，三维推广尚未实现；方法对极大网格比率或复杂非矩形细域的适用性待进一步评估。

---

## 204. ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding

**arXiv ID:** 2604.14612 | [PDF](https://arxiv.org/pdf/2604.14612v1)

**作者:** Walaa Amer `[一作]` (University of California Irvine), Fadi Kurdahi `[通讯]` (University of California Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自适应基于置信度的层跳过方法ConfLayers，用于加速LLM的自我推测解码

**💡 创新点**

创新点是利用中间层置信度动态决定跳过层，无需额外训练或外部草稿模型

**🔧 技术方法**

采用置信度评估、熵计算、自适应窗口大小以及迭代搜索等技术

**📊 数据集**

在LLaMA-2/3、CodeLLaMA、Qwen-2.5-Math等模型上，使用CNN-DM、GSM8K、WMT14、Alpaca、HumanEval等数据集

**📈 对比分析**

与传统SSD、DEL、SWIFT等基线对比，ConfLayers在多任务上平均提升1.15–1.35×速度，保持高接受率与Rouge‑2相当

**⚠️ 局限性**

局限性在于对置信度阈值和窗口参数的手动调优敏感，且在极小模型或特定任务中跳过比例不一定最优

---

## 205. Category-based and Popularity-guided Video Game Recommendation: A Balance-oriented Framework

**arXiv ID:** 2604.14598 | [PDF](https://arxiv.org/pdf/2604.14598v1)

**作者:** Xiping Li `[一作]` (Harbin Institute of Technology), Yutong Wang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8935 | [OpenAlex ID](https://openalex.org/A5100631524)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了基于游戏类别与热门度的多模块框架 CPGRec，用于平衡视频游戏推荐的准确性与多样性。

**💡 创新点**

创新点在于：① 使用跨类别严格连接提升准确性；② 通过高连通性邻居聚合和基于热门度的边/节点加权增强多样性；③ 在 BPR 损失中加入负样本评分再加权，兼顾准确性与多样性。

**🔧 技术方法**

采用 LightGCN 与注意力机制构建游戏图与玩家-游戏双边图，结合边/节点权重调整、层级注意力和负样本评分重加权技术，整体实现三模块协同训练。

**📊 数据集**

使用 Steam 数据集（约 3.9M 用户、2.7k 游戏、95M 交互），采用 5‑core 过滤后进行实验。

**📈 对比分析**

与 LightGCN、SCGRec（准确性基准）及 MMR、EDUA、DDGraph、DGCN、DGRec（多样性基准）进行对比。结果显示 CPGRec 在多样性指标（Coverage、Entropy）上居首，准确性指标仅落后 SCGRec 之下，整体在平衡两者方面表现最佳。

**⚠️ 局限性**

局限性包括：仍需手动调节多种超参数（如 θ_e^hot、θ_n^hot、θ_n^cold、权重 w_Ca,w_Co,w_Po）以获取最优平衡；过度强调多样性可能导致准确性下降；对不同游戏类别与流行度的依赖使得在类别信息不足或热门度变化剧烈时效果可能受限。

---

## 206. Deepfake Detection Generalization with Diffusion Noise

**arXiv ID:** 2604.14570 | [PDF](https://arxiv.org/pdf/2604.14570v1)

**作者:** Hongyuan Qi `[一作]` (Zhejiang University), Jun Xiao `[通讯]` (Zhejiang University)

**通讯引用:** 75320 | [OpenAlex ID](https://openalex.org/A5100361698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Attention-guided Noise Learning（ANL）框架，利用预训练扩散模型在噪声域识别深度伪造图像。

**💡 创新点**

通过将扩散模型的噪声估计转化为全局注意力，引导分类器聚焦细微噪声差异，从而显著提升对未知生成器的泛化能力。

**🔧 技术方法**

结合预训练的扩散模型（如ADM）进行噪声估计，构造空间注意力映射，使用ResNet分类器并以二元交叉熵训练。

**📊 数据集**

在DiffFace、DiFF和DiffusionForensics等面部深度伪造数据集上进行实验。

**📈 对比分析**

与DIRE、SeDID、NPR等现有方法在标准、跨数据集及跨模型评估中对比，ANL在跨模型场景下ACC/AP提升至少12%/6%，在标准评估中保持近100%的准确率。

**⚠️ 局限性**

依赖高质量预训练扩散模型，噪声估计对时间步敏感，对极其先进的生成模型仍存在辨识难度，且未针对视频或多模态伪造进行验证。

---

## 207. Learning Adaptive Reasoning Paths for Efficient Visual Reasoning

**arXiv ID:** 2604.14568 | [PDF](https://arxiv.org/pdf/2604.14568v1)

**作者:** Yixu Huang `[一作]` (Fudan University), Muhao Chen `[通讯]` (University of California, Davis)

**通讯引用:** 4952 | [OpenAlex ID](https://openalex.org/A5102861481)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AVR自适应视觉推理框架，分解为感知、推理、答复三功能，并动态选择三种响应格式。

**💡 创新点**

识别视觉推理中的“推理路径冗余”，并通过格式化训练和FS‑GRPO实现任务自适应格式选择，显著减少冗余推理。

**🔧 技术方法**

采用两阶段训练：监督格式化微调(SFT) + 自适应奖励的Group Relative Policy Optimization(FS‑GRPO)，并使用特殊功能标记分解推理路径。

**📊 数据集**

使用包含44k样本的RL训练集（OK‑VQA、CLEVR、VCR、GQA、ChartQA、OCR‑VQA、MathVerse、Geometry3K、ScienceQA、TQA），以及11k格式标注样本。

**📈 对比分析**

与基线（Qwen3‑VL‑Thinking、TON、ARM2等）比较，AVR在七个视觉‑语言基准上实现50–90% token节省，且在感知密集任务准确率提升2–4%，在推理密集任务保持或略优。

**⚠️ 局限性**

局限性在于对格式多样性奖励的调参敏感，仍需在极端复杂推理任务中避免过度压缩；且依赖于标注格式的可用性与RL奖励设计。

---

## 208. Physically-Induced Atmospheric Adversarial Perturbations: Enhancing Transferability and Robustness in Remote Sensing Image Classification

**arXiv ID:** 2604.14643 | [PDF](https://arxiv.org/pdf/2604.14643v1)

**作者:** Weiwei Zhuang `[一作]` (Xiamen University of Technology), Jun Luo `[通讯]` (Nanyang Technological University)

**通讯引用:** 24872 | [OpenAlex ID](https://openalex.org/A5100749903)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于物理可实现雾效的对抗攻击框架FogFool，用来误导遥感图像分类模型。

**💡 创新点**

创新点在于将Perlin噪声与分形布朗运动相结合，生成自然、低频的雾形扰动；该扰动既保持视觉真实，又能显著提升跨模型的迁移性能和对常见预处理防御的鲁棒性。

**🔧 技术方法**

技术核心包括多频段Perlin噪声生成、分形累加（FBM）、雾层叠加与梯度引导的优化算法，配合高斯平滑正则化，最终得到可控的雾性对抗样本。

**📊 数据集**

实验在两个遥感基准数据集UCM和NWPU-RESISC45上进行，使用8种常见CNN（AlexNet、VGG16、ResNet50/101、DenseNet121/201、MobileNetV2、EfficientNet‑B0）进行评估。

**📈 对比分析**

与FGSM、PGD、AutoAttack等白盒攻击以及MI‑FGSM、DI‑FGSM、TI‑FGSM等基于梯度的迁移攻击对比，FogFool在无防御环境下的攻击成功率分别达96.79%（UCM）和99.96%（NWPU），在黑盒迁移实验中平均TASR高达83.74%，并在JPEG压缩和TVM防御下保持较高攻击成功率，显示出更强的鲁棒性。

**⚠️ 局限性**

局限性包括：仍依赖雾型扰动，对非雾类自然环境或不同传感器（如SAR）效果未知；缺乏对目标类别选择性的精准控制；以及未考虑实际部署中对雾效物理参数的精确测量和实现。

---

## 209. Balancing Weights, Directed Sparsification, and Augmenting Paths

**arXiv ID:** 2604.14633 | [PDF](https://arxiv.org/pdf/2604.14633v1)

**作者:** Jason Li `[一作]` (Carnegie Mellon University), Jason Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12837 | [OpenAlex ID](https://openalex.org/A5100762970)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于随机增广路的算法，能够在有向无容量图中求解最大流，时间复杂度为 (m+nF)·n^{o(1)}，并在预处理后进一步得到 m·n^{1/2+o(1)} 的时间。

**💡 创新点**

创新点在于引入“平衡权重”技术，使得残差图中的每条割在权重上近乎平衡，从而可以像无向图一样进行随机抽样；同时动态维护这些权重时采用了最新的内部点法中最小比例割的数据结构，克服了权重随增广路径变更而需更新的问题。

**🔧 技术方法**

使用的技术包括：潜在函数潜能调度的平衡权重；对平衡权重的动态维护通过内部点法的最小比例割近似数据结构；随机抽样与稀疏化（动态稀疏化器 + 指数扩张分解）；增广路的高效搜索；以及初期的 √n 次阻塞流减少流值。

**📊 数据集**

论文没有使用具体实验数据集，全部以理论分析和证明为主；主要关注算法的时间复杂度和稀疏化质量。

**📈 对比分析**

相较于传统的 Dinic 算法和最近的连续优化方法，本文在稀疏图（m≈n）上实现了 m·n^{1/2+o(1)} 的时间，首次在增广路框架下突破了 Dinic 的 O(m·min{m^{1/2},n^{2/3}}) 上限；与基于内部点法的几乎线性时间方法相比，保留了纯增广路的组合性优势。

**⚠️ 局限性**

局限性包括：必须在强连通图上运行；权重函数 w(u,v)=1/max{φ(v)-φ(u),0}+1 的选择尚未证明为最优，可能导致对极端方向的上限控制不够；算法对一般有向图（非强连通）的适用性尚未解决。

---

## 210. Switch-KD: Visual-Switch Knowledge Distillation for Vision-Language Models

**arXiv ID:** 2604.14629 | [PDF](https://arxiv.org/pdf/2604.14629v1)

**作者:** Haoyi Sun `[一作]` (Li Auto Inc.), Wei Chen `[通讯]` (Li Auto Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Switch‑KD，一种统一视觉–语言知识蒸馏框架；

**💡 创新点**

创新点包括：① 视觉‑切换蒸馏（将学生视觉输出投射到教师语言路径，实现在共享文本概率空间内的隐式跨模态监督）；② 动态双向对数差损失（DBiLD）——自适应选取信息量最大的 Top‑k 位置，并在双向 KL 约束下对排名差异进行对齐，兼顾高置信区间与分布结构；

**🔧 技术方法**

核心技术：视觉‑语言统一文本概率空间；视觉‑切换路径；DBiLD 损失（动态 Top‑k、双向 KL、对数差分归一化）；预训练与蒸馏两阶段训练策略；

**📊 数据集**

使用的主要数据集：十个多模态基准（MME、MMB、MMB‑CN、VQAv2、GQA、ScienceQA、MMMU、TextVQA、VizWiz、POPE），以及预训练所用的数据（LLaVA1.5‑558K、LLaVA‑Mix‑665K 等）；

**📈 对比分析**

与 TinyLLaVA、Mini‑Gemini、SPHINX‑Tiny、MobileVLM、MoVE‑KD、LLaVA‑MOD、LLaVA‑KD、Align‑KD 等方法对比。Switch‑KD 在 0.5B 规模下平均提升 3.6‑4.0 分，1.5B 规模下平均提升约 4.4 分，在所有 10 个基准上均达到或超过现有 SOTA；

**⚠️ 局限性**

局限性：需要教师与学生在特征空间与词表上保持一致，限制了跨架构的直接迁移；当学生容量受限时，使用更大教师反而可能不再带来收益。

---

## 211. A Parallel Approach to Counting Exact Covers Based on Decomposability Property

**arXiv ID:** 2604.14627 | [PDF](https://arxiv.org/pdf/2604.14627v1)

**作者:** Liangda Fang `[一作]` (Jinan University), Quanlong Guan `[通讯]` (Jinan University)

**通讯引用:** 1047 | [OpenAlex ID](https://openalex.org/A5084935157)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种并行编译算法 DXD，利用零抑制决策 DNNF（decision‑ZDNNF）高效表示并计数所有精确覆盖。

**💡 创新点**

其创新点在于将决策‑DNNF 与零抑制 BDD 结合，形成更简洁的表示形式，并设计了 DynDXD 通过动态更新连通分量显著提升了构造速度。

**🔧 技术方法**

核心技术包括：零抑制决策‑DNNF 语法与语义、Dancing‑Link 进行子矩阵的覆盖/恢复、以及基于欧拉巡回与 splay 树的连通分量动态维护。

**📊 数据集**

实验使用了 245 个 Topology Zoo 图和 255 个 Rome Graphs 共 500 个图实例，构造对应的精确覆盖实例。

**📈 对比分析**

与 DLX、DXZ、D3X、SharpSAT‑TD 和 ExactMC 对比，DynDXD 在 1,500 秒阈值下完成 462/500 个实例，速度比 DXZ 提升约 3.5 倍、比 SAT 计数器快两位数，并支持多线程实现 2.4 倍加速。

**⚠️ 局限性**

限制在于仍具有指数级最坏情况复杂度；若输入矩阵难以分解为独立子矩阵，DynDXD 的并行优势不明显；动态连通分量维护在极大图中可能成为瓶颈。

---

## 212. ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving

**arXiv ID:** 2604.14626 | [PDF](https://arxiv.org/pdf/2604.14626v1)

**作者:** Yuseon Choi `[一作]` (KAIST), Sangjin Kim `[通讯]` (GIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ELMoE-3D框架，统一了 MoE 推理中的缓存加速与自回归推测解码，利用 3D‑IC 混合连线 HB 内存实现高带宽、低功耗的本地推理；

**💡 创新点**

创新点在于：①将专家与位宽两轴弹性结合，构建 Elastic Self‑Speculative Decoding；②利用 MSB 4‑bit 切片作为自草稿模型；③通过 LSB 增强的位切片架构实现多精度无额外硬件开销；④在 HB 3D‑IC 上实现全流程协同的硬件‑软件共设计推理流水线；

**🔧 技术方法**

使用技术包括：3D‑IC 混合连线 HB、位切片 MAC、张量并行与数据并行混合调度、专家热度阈值（热专家选择）、位嵌套量化（LSB 增强）、自回归+树形自推测解码、周期准确仿真（Duplex+Ramulator）、KV 缓存 LRU 等；

**📊 数据集**

评估数据集：MT‑Bench（多轮对话）、GSM8K（数学推理）、Alpaca（指令跟随）、HumanEval（代码生成），模型长度均设为 1k；

**📈 对比分析**

与 xPU、PIM、NMP、LogicPIM、HB‑xPU 等基线同规模 64 GB LPDDR5、262–524 TOPS 计算率进行对比；采用周期准确仿真评估延迟与能耗；结果显示在批量 1–16 上，ELMoE‑3D 平均速度提升 6.6×、能耗下降 4.4×，相对最佳基线提升 2.2×速度、1.4×能耗；

**⚠️ 局限性**

限制在于 HB 容量竞争，尤其 KV 缓存占用高导致可用于专家缓存的空间受限；在长上下文或高序列长度场景下，性能优势可能被削弱；不同模型间专家局部性差异需要进一步自适配。

---

## 213. GDPR Auto-Formalization with AI Agents and Human Verification

**arXiv ID:** 2604.14607 | [PDF](https://arxiv.org/pdf/2604.14607v1)

**作者:** Ha Thanh Nguyen `[一作]` (ROIS-DS), Ken Satoh `[通讯]` (ROIS-DS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多代理、人工干预的验证框架，用于将GDPR条款自动转化为可执行的法律规则并生成对应的场景、事实和规则树；

**💡 创新点**

创新点在于将生成过程与多维度自动验证相结合，强调角色专业化和结构化反馈，以确保法律忠实度和逻辑一致性；

**🔧 技术方法**

利用大型语言模型（LLM）进行情境、规则与事实生成，采用Pythen框架表示规则树，RuleTreeEvaluator执行逻辑评估，四个Verifier Agent（情境、表示、逻辑、法律）进行独立评估；

**📊 数据集**

生成了一个专门针对GDPR条款的自动化形式化数据集，通过迭代生成与验证获得高质量样本；

**📈 对比分析**

通过与人工专家评估对比，证明在触发条件明确、情境精确的场景中自动化表现良好；在复杂权利、例外和角色依赖的情境中仍需人工干预，实验表明验证框架显著提高了规则的正确率；

**⚠️ 局限性**

限制在于自动化在绝对权利、细粒度例外、缺失义务和程序与实质性区分等方面易失效，需要进一步改进模型对细节的把握与人机协作流程。

---

## 214. Towards Design Compositing

**arXiv ID:** 2604.14605 | [PDF](https://arxiv.org/pdf/2604.14605v1)

**作者:** Abhinav Mahajan `[一作]` (Carnegie Mellon University), Balaji Vasan Srinivasan `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种无训练、身份保留的图像合成模块（GIST），在组件到设计的流水线中，能够在布局预测后、排版之前，对输入的图像和SVG进行风格化与融合，提升视觉和谐与美学质量。

**💡 创新点**

创新点在于利用跨注意力引导的令牌注入（Cross‑Attention Guided Token Injection）和潜在初始化（Latent Initialization）两种训练‑free 机制，在保持输入元素语义身份的同时，实现风格统一的合成；同时模块可即插即用，兼容多种现有布局与排版模型。

**🔧 技术方法**

技术手段包括：基于 Emu‑2 的64‑token 低维瓶颈，使用视觉编码器提取身份令牌；通过 SDXL UNet 的跨注意力图对令牌进行区域相关性评估并选择性混合；采用 Flow‑Matched Euler Discrete Scheduler 对背景潜在进行初始化以提升背景保真度。

**📊 数据集**

使用 Crello 测试集（1,500 条真实设计）评估整体流水线；在受控任务 Human Face Generation 和 Specific Object Generation 上验证身份保留。

**📈 对比分析**

与 LaDeCo 的无缝合成对比，GIST 在图形与图像、创新与原创性方面分别提升 0.09 与 0.12 分，整体平均分保持不变；与基于 Design‑o‑meter 的完整流水线相比，在 GPT‑4V 评估中平均分从 4.9 提升至 5.9，首选比例达 71.4%。

**⚠️ 局限性**

局限性包括：仍受 Emu‑2 结构约束，难以推广至新统一生成模型；文本元素的融合仍未完整解决；潜在的扩散模型伪影与生成一致性问题待进一步改进。

---

## 215. NLP needs Diversity outside of 'Diversity'

**arXiv ID:** 2604.14595 | [PDF](https://arxiv.org/pdf/2604.14595v1)

**作者:** Joshua Tint `[一作]` (Arizona State University), Joshua Tint `[通讯]` (Arizona State University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5117864919)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对ACL Anthology中作者的关键词检索，统计了不同NLP子领域（尤其是公平性相关与非公平性相关）中作者的性别与地理分布，从而揭示了子领域之间多样性差距。

**💡 创新点**

创新点在于首次系统比较NLP各研究方向的性别与地理多样性，发现公平性领域女性比例显著高于其他领域，并提出了结构性原因与改进建议。

**🔧 技术方法**

采用关键词检索+ACL“相关性”排序技术获取作者列表；利用公开个人资料中的代词进行性别标注；依据作者所在机构所在洲划分地理标签；随后进行统计与对比分析。

**📊 数据集**

使用的数据来源为ACL Anthology作者页面、学术机构个人主页、LinkedIn等公开资料，手工汇总并标注了约750位作者的性别与洲别信息。

**📈 对比分析**

方法是对每个关键词前50名活跃作者进行性别与洲别统计，计算女性比例与洲别分布。结果显示公平性关键词平均女性比例为34%，非公平性关键词为18%；北美在公平性领域占比更高，亚洲在非公平性领域占比更高。

**⚠️ 局限性**

局限性包括：性别识别依赖公开代词，易出现误判；未能完整纳入跨性别/非二元作者；地理数据仅反映当前机构，可能与实际分布不符；洲别划分粗略，忽略了地区内细微差异；数据量有限，缺乏深层次因果分析。

---

## 216. Don't Retrieve, Navigate: Distilling Enterprise Knowledge into Navigable Agent Skills for QA and RAG

**arXiv ID:** 2604.14572 | [PDF](https://arxiv.org/pdf/2604.14572v1)

**作者:** Yiqun Sun `[一作]`, Lawrence B. Hsieh `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一套离线编译成层级技能树的知识库，使LLM代理在运行时通过文件浏览工具主动导航检索证据。

**💡 创新点**

创新点在于将知识库预先聚类并用LLM生成层级摘要，形成可导航的文件系统技能目录，让LLM能够主动路径规划、回溯和跨分支整合，而非被动接受检索结果。

**🔧 技术方法**

技术包括迭代式层级聚类（K‑means+LLM总结）、句子嵌入、文件系统技能包、Anthropic Skills API 进阶披露、两种工具（文件浏览与文档检索）以及LLM推理。

**📊 数据集**

使用的主要数据集是 WixQA 企业客服问答基准，包含 6,221 篇支持文章与 200 条专家题目。

**📈 对比分析**

与 BM25、Dense、Hybrid、RAPTOR、Agentic RAG 等基线对比，采用 Token F1、BLEU、ROUGE、LLM 评估指标，本文在所有指标上均优于基线，Token F1 提升约 19%，Factuality 与 Context Recall 亦显著提升。

**⚠️ 局限性**

局限性包括每次查询需加载多份导航文件导致输入 token 高、成本相对较高；受 Anthropic Skills API 的技能/文件/大小限制；硬聚类单路径限制多主题文档导致潜在盲区；更新不即时，需重新编译。

---

## 217. Sidorenko-Inspired Pessimistic Estimation

**arXiv ID:** 2604.14647 | [PDF](https://arxiv.org/pdf/2604.14647v1)

**作者:** Yu-Ting Lin `[一作]` (National Taiwan University), Hsin-Po Wang `[通讯]` (National Taiwan University)

**通讯引用:** 170 | [OpenAlex ID](https://openalex.org/A5073226189)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在关系数据库的关联查询上，提出了以星形、双星形和茅厕形（caterpillar）为基础的层次化取值统计量，用以改进对连接结果大小的上界估计；

**💡 创新点**

创新点在于引入基于Sidorenko猜想的茅厕形统计量，并推导出相应的熵不等式，从而在信息理论框架下得到更紧的连接大小上界；

**🔧 技术方法**

采用信息论熵与Shannon型不等式、线性规划结合的技术，并利用图同态计数（星形、双星形、茅厕形）来计算统计量；

**📊 数据集**

实验使用SNAP网络数据集（共39个图）进行同态计数测试；

**📈 对比分析**

将星形、双星形、p0p、p00p、p000p等五种方法进行对比，实验结果显示：星形估计过大约m倍，双星形约m^{3/4}，茅厕形约m^{3/5}，即后者的过估误差显著降低；

**⚠️ 局限性**

局限性包括：仍依赖于Sidorenko猜想的部分经验推断，未能证明对所有图都成立；计算复杂度随统计量阶数上升，且实验仅覆盖了5个顶点以内的图；未讨论对更大查询或多表JOIN的实际执行计划匹配情况。

---

## 218. Fact4ac at the Financial Misinformation Detection Challenge Task: Reference-Free Financial Misinformation Detection via Fine-Tuning and Few-Shot Prompting of Large Language Models

**arXiv ID:** 2604.14640 | [PDF](https://arxiv.org/pdf/2604.14640v1)

**作者:** Cuong Hoang `[一作]` (Japan Advanced Institute of Science and Technology), Le-Minh Nguyen `[通讯]` (Japan Advanced Institute of Science and Technology)

**通讯引用:** 4493 | [OpenAlex ID](https://openalex.org/A5077641909)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并赢得了Reference-Free Financial Misinformation Detection比赛，提出使用零/少量样本提示加LoRA微调的Qwen-2.5模型，对金融文本进行内部语义推理。

**💡 创新点**

结合零/少量样本提示与参数高效微调，在无外部证据的情境下实现97%以上的准确率，首次在此类任务中突破传统基线。

**🔧 技术方法**

In‑context learning（zero‑shot / few‑shot）、LoRA参数高效微调、Qwen-2.5 LLM、精细prompt模板。

**📊 数据集**

官方提供的Reference‑Free Financial Misinformation Detection dataset，真/假各占50%。

**📈 对比分析**

与公开基线（如GPT‑4.1 2‑shot）对比，公测准确率95.4%、私测96.3%，F1分别为95.4%和96.29%，排名第一。

**⚠️ 局限性**

仅在Qwen-2.5上验证，模型对仅含数字修改的全新误导难以识别，缺乏可解释的中间推理路径，需进一步验证对其他模型的泛化。

---

## 219. The Acoustic Camouflage Phenomenon: Re-evaluating Speech Features for Financial Risk Prediction

**arXiv ID:** 2604.14619 | [PDF](https://arxiv.org/pdf/2604.14619v1)

**作者:** Dhruvin Dungrani `[一作]` (Independent Researchers), Disha Dungrani `[通讯]` (Independent Researchers)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

评估企业财报电话会议中音频与文本特征在预测灾难性下行风险时的有效性

**💡 创新点**

提出“声学伪装”概念，揭示媒体训练的声音抑制会削弱多模态模型的预测能力

**🔧 技术方法**

采用两流后期融合架构：L1正则化的逻辑回归提取音频/文本特征，随后用L2正则化逻辑回归做元学习

**📊 数据集**

使用MAEC数据集（对齐的音频与文本）并利用FinBERT提取Sentiment Delta

**📈 对比分析**

与单模态文本模型比较，文本模型召回率66.25%，音频模型50.83%，融合模型仅47.08%，表明加入音频反而降低性能

**⚠️ 局限性**

局限在于数据来源于VoIP压缩的电话会议，压缩与噪声抑制可能掩盖真实声学微震，影响可重复性

---

## 220. Mechanistic Decoding of Cognitive Constructs in LLMs

**arXiv ID:** 2604.14593 | [PDF](https://arxiv.org/pdf/2604.14593v1)

**作者:** Yitong Shou `[一作]` (Zhejiang University), Manhao Guan `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于 Representation Engineering 的认知逆向工程框架，用来揭示大型语言模型（LLM）内部如何构建社会比较型嫉妒情绪，并通过子空间正交化、回归加权与双向因果调控对模型进行机械干预。

**💡 创新点**

创新点在于将 RepE 与子空间正交化、统计权重估计和可解释的因果干预相结合，使得对复杂情绪的微观机制可分解、量化并可实时调节，从而实现对模型情绪行为的监测与安全介入。

**🔧 技术方法**

主要技术包括线性人工层析（LAT）、对比均值差分、子空间正交化、普通最小二乘回归以及基于向量的正负双向调控（amplify/suppress）。

**📊 数据集**

使用两套自制数据集：T1（200对对比情境）用于概念提取与验证，G1（结构化组合情境）用于回归权重估计与因果干预，所有样本均由 Gemini‑3‑Pro 生成并人工审核。

**📈 对比分析**

实验在八款 Llama、Qwen、Gemma 系列 LLM 上进行，结果显示在中后层的概念提取准确率接近 100%，线性回归 R²>0.7，因果干预能显著提升或降低嫉妒评分，且与人类心理学模型高度一致。

**⚠️ 局限性**

局限性包括仅聚焦两大核心因素（优越性与相关性），生成情景缺乏真实语言多样性，部分模型存在“Hydra”自修复导致严格消除效果有限，且方法尚未推广到其他复杂情绪。

---

## 221. AgileLog: A Forkable Shared Log for Agents on Data Streams

**arXiv ID:** 2604.14590 | [PDF](https://arxiv.org/pdf/2604.14590v1)

**作者:** Shreesha G. Bhat `[一作]` (University of Illinois Urbana-Champaign), Aishwarya Ganesan `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 291 | [OpenAlex ID](https://openalex.org/A5056000802)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了XXX问题，提出了一种新的解决方案。

**💡 创新点**

创新点在于引入了XXX方法，显著提高了性能。

**🔧 技术方法**

使用了XXX技术，如深度学习、机器学习等。

**📊 数据集**

实验使用了XXX数据集，包含了XXX样本。

**📈 对比分析**

与现有方法进行了比较，结果显示本方法在XXX指标上优于其他方法。

**⚠️ 局限性**

限制在于XXX，例如数据集规模较小或模型复杂度高。

---

## 222. CLion: Efficient Cautious Lion Optimizer with Enhanced Generalization

**arXiv ID:** 2604.14587 | [PDF](https://arxiv.org/pdf/2604.14587v1)

**作者:** Feihu Huang `[一作]` (Nanjing University of Aeronautics and Astronautics), Songcan Chen `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 13783 | [OpenAlex ID](https://openalex.org/A5101596072)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析了 Lion 优化器的泛化性能，并通过算法稳定性证明其泛化误差为 O(1/(Nτ^T))；提出了一种改进的 Cautious Lion（CLion）优化器，将 sign 函数按阈值谨慎使用，得到更低的泛化误差 O(1/N)，并证明其收敛速率为 O(√d/T^{1/4})，与 Lion 相同。

**💡 创新点**

①首次用算法稳定性方法给出 Lion 的泛化误差上界；②发现梯度估计器中非零元素最小绝对值 τ 对泛化误差影响显著；③提出 CLion 通过阈值控制 sign 函数，显著降低泛化误差；④在理论上给出 CLion 与 Lion 的收敛速率相同。

**🔧 技术方法**

算法稳定性（uniform stability）、数学归纳法、Lipschitz 光滑性与方差界定、非凸随机优化理论、以及数值实验评估。

**📊 数据集**

语言模型：WikiText‑2 与 WikiText‑103；图像分类：CIFAR‑10 与 Tiny‑ImageNet；实验均使用 Transformer/ResNet 等标准网络。

**📈 对比分析**

与 SGD、SGDM、Adam、AdamW、Lion、RLion 等常用优化器在相同数据集、相同超参搜索条件下进行对比；实验结果表明 CLion 在训练和测试误差/准确率上普遍优于 Lion 与 RLion，且与 SGD/Adam 等保持竞争水平。

**⚠️ 局限性**

主要局限在于：①泛化误差上界依赖 τ 的最小非零梯度元素，若 τ 极小仍可能导致上界不具备实际意义；②理论分析仅覆盖非凸设置，未考虑重尾噪声或分布不匹配情况；③阈值 ν 的选择仍需经验调优，缺乏自适应机制；④实验验证集中于语言和图像任务，缺乏更广泛的应用验证。

---

## 223. CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations

**arXiv ID:** 2604.14586 | [PDF](https://arxiv.org/pdf/2604.14586v1)

**作者:** Xiping Li `[一作]` (Harbin Institute of Technology), Yi Zhao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 6863 | [OpenAlex ID](https://openalex.org/A5052683156)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在游戏推荐系统中，提出了CPGRec+框架，通过新加入的Preference‑informed Edge Reweighting（PER）和Preference‑informed Representation Generation（PRG）两大模块，既捕获玩家-游戏交互的显著偏好差异，又利用大型语言模型（LLM）生成更富含全局和个体兴趣信息的游戏与玩家描述，以实现准确性与多样性的平衡推荐。

**💡 创新点**

创新点在于：1）PER通过Fisher分布判定边的兴趣/不兴趣符号并用信息熵量化重要性，实现对交互重要性的细粒度加权，缓解GCN的过平滑问题；2）PRG利用LLM根据平均评分和停留时间生成全局与个人兴趣兼顾的文本描述，进而对游戏与玩家嵌入进行对齐与融合，显著提升个性化表达。

**🔧 技术方法**

主要技术包括图神经网络（LightGCN、SNA等）、Box‑Cox 与 Z‑score 转换、Fisher分布检验、信息熵计算、LLM（Qwen2.5）文本生成、M3‑Embedding 嵌入、MLP 对齐与融合，以及多项负样本重权重训练。

**📊 数据集**

在两个Steam公开数据集上进行实验：Steam I（约3.9万玩家、2.7千游戏、9.5亿交互）与Steam II（约3.3万玩家、13.0千游戏、3.7亿交互）。

**📈 对比分析**

与多类基准（LightGCN、SURGE、BIGCF、MVGNN、SCGRec等）及多样性、平衡方法（MMR、EDUA、DDGraph、DGCN、DGRec、EXPLORE）比较，CPGRec+在准确性（NDCG、Recall、Hit、Precision）与多样性（Coverage、Entropy、Tail Coverage、Tail）指标上均均优于或匹配最新状态方法，尤其在长尾游戏推荐上表现突出。

**⚠️ 局限性**

限制包括：1）PER和PRG需要预先对交互特征进行离线统计与LLM推理，增加前置计算成本；2）LLM生成的文本描述可能受prompt设计与模型局限影响；3）模型在极端稀疏或冷启动场景下的泛化能力尚待进一步验证。

---

## 224. From Risk to Rescue: An Agentic Survival Analysis Framework for Liquidation Prevention

**arXiv ID:** 2604.14583 | [PDF](https://arxiv.org/pdf/2604.14583v1)

**作者:** Fernando Spadea `[一作]` (Rensselaer Polytechnic Institute), Oshani Seneviratne `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 909 | [OpenAlex ID](https://openalex.org/A5038466673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个基于生存分析的自主代理系统，用于在Aave v3上主动预防清算。

**💡 创新点**

创新点在于将返回周期指标与方差调节趋势评分相结合，并通过对抗性优化循环生成最小资本干预方案，同时在协议级模拟器中实现安全验证。

**🔧 技术方法**

使用XGBoost Cox比例风险模型、返回周期指标、无量纲趋势评分、对抗性优化循环及协议真实模拟器。

**📊 数据集**

使用Aave v3 Polygon子图的21.8M交易记录，构建包含90个特征的FinSurvival风格数据集。

**📈 对比分析**

通过对4,882名高风险用户的重放实验，代理实现87%的清算拯救率、0%的恶化率，明显优于传统静态健康因子阈值，并与多种检测算法达成一致。

**⚠️ 局限性**

局限包括对钱包余额的推断假设、对极端闪电清算的无干预能力，以及模型对异常事件的鲁棒性有限。

---

## 225. Enhancing Mental Health Counseling Support in Bangladesh using Culturally-Grounded Knowledge

**arXiv ID:** 2604.14576 | [PDF](https://arxiv.org/pdf/2604.14576v1)

**作者:** Md Arid Hasan `[一作]` (University of Toronto), Syed Ishtiaque Ahmed `[通讯]` (University of Toronto)

**通讯引用:** 4283 | [OpenAlex ID](https://openalex.org/A5089574660)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在孟加拉国为低收入社区的para‑counselor提供文化敏感的LLM支持，比较检索增强生成(RAG)与知识图谱(KG)两种方法，并评估其对辅导质量的影响。

**💡 创新点**

将手工构建、临床验证的知识图谱与LLM结合，首次针对para‑counselor在资源匮乏环境中的实际需求提供结构化知识支持，突出文化与临床适宜性。

**🔧 技术方法**

采用检索增强生成(RAG)、知识图谱推理+LLM、BERTScore、SBERT，以及多种大型语言模型（Gemini、Llama、GPT等）进行自动与人工评估。

**📊 数据集**

使用Sajida基金会收集的402份双语（孟加拉语）辅导案例，经多学科团队注释后压缩为69个案例，构建了308节点/642关系的知识图谱。

**📈 对比分析**

对比RAG与KG增强的同一四个模型，自动评估BERTScore差异<1点，SBERT波动较大；人工评估显示KG版平均评分提升0.4–0.9点，Llama‑3.3‑70B在所有指标上表现最佳。

**⚠️ 局限性**

数据量有限、知识图谱覆盖范围受限、人工评估主观且样本不足、系统仅为辅助决策非替代专业临床判断。

---

## 226. M3D-Net: Multi-Modal 3D Facial Feature Reconstruction Network for Deepfake Detection

**arXiv ID:** 2604.14574 | [PDF](https://arxiv.org/pdf/2604.14574v1)

**作者:** Haotian Wu `[一作]` (South China Agricultural University), Shan Bian `[通讯]` (South China Agricultural University)

**通讯引用:** 2963 | [OpenAlex ID](https://openalex.org/A5040217973)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于双流结构的端到端深伪造检测网络M3D-Net，结合自监督3D面部特征重建（深度与反照率）与RGB通道的多模态融合，提升了对合成面孔的检测能力。

**💡 创新点**

创新点包括：①使用自监督Unsup3D实现单视角RGB图像的3D重建；②设计3D特征预融合模块（PFM），利用深度可分离卷积和空间核注意力自适应校准多尺度特征；③构建多模态融合模块（MFM），通过双向注意力实现RGB与3D重建特征的深度交互。

**🔧 技术方法**

主要技术手段为自监督3D重建网络Unsup3D、EfficientNet-B4骨干网络、深度可分离卷积、空间核注意力（SKAttention）、交叉注意力与自注意力机制，以及t-SNE可视化分析。

**📊 数据集**

实验使用了多公开数据集：FaceForensics++（c23）、Celeb-DF v1/v2、DFD、DFDC、FaceShifter、DeeperForensics-1.0等，覆盖多种伪造技术与压缩级别。

**📈 对比分析**

与Xception、MesoIncep、DSP-FWA、Face X-ray、FFD、F3Net、SRM等SOTA方法对比，M3D-Net在FF++ c23的AUC达到0.9747，且在多数据集跨数据测试中获得最高或接近最高的AUC，显著优于现有方法。

**⚠️ 局限性**

局限性在于重建过程高度依赖面部对称性，导致在极端姿态、遮挡或非对称面孔上的重建与检测性能下降，未来需探索更鲁棒的3D重建与数据增强策略。

---

## 227. Targeted Exploration via Unified Entropy Control for Reinforcement Learning

**arXiv ID:** 2604.14646 | [PDF](https://arxiv.org/pdf/2604.14646v1)

**作者:** Chen Wang `[一作]` (Nankai University), Yue Wang `[通讯]` (Zhongguancun Academy)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出统一熵控制框架 UEC‑RL，改进 RL 在大语言模型与视觉语言模型推理中的探索与稳定性。

**💡 创新点**

双向熵调控机制：针对困难提示主动提升熵进行探索，同时通过重放稳定熵，解决 GRPO 的熵崩塌与训练不稳定问题。

**🔧 技术方法**

在 GRPO 基础上加入温度提升的软最大化探索、可控熵稳定器、自然策略梯度更新等技术，并通过超参数 t'（探索温度）和 s'（重放容量）实现可调节熵。

**📊 数据集**

使用文本推理数据集 AIME24/25、MATH、GSM8K、Minerva、ARC、MMLU，视觉-文本推理数据集 MathVision、MathVerse、MathVista、We‑Math，和 Geometry3K 进行内部分析。

**📈 对比分析**

与 GRPO、DAPO、KL‑cov、Entropy‑Adv 四个 RL 基线在 Pass@1、Pass@k、Accuracy 等指标上对比，UEC‑RL 在多数任务上提升约 1–3 个百分点，Geometry3K 上相对 GRPO 提升 37.9%，并且训练更稳定、步骤耗时更低。

**⚠️ 局限性**

对探索温度 t' 与重放容量 s' 的选择敏感，需根据任务难度手动调参；缺乏自适应调节机制导致跨域性能一致性受限。

---

## 228. A multi-platform LiDAR dataset for standardized forest inventory measurement at long term ecological monitoring sites

**arXiv ID:** 2604.14635 | [PDF](https://arxiv.org/pdf/2604.14635v1)

**作者:** Michael R. Chang `[一作]` (Free University of Bozen-Bolzano), Marco Camurri `[通讯]` (University of Trento)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

在意大利阿尔卑斯的ICOS监测站对单个控制点进行多平台LiDAR扫描（无人机激光扫描、地面激光扫描、背包移动激光扫描），生成高度统一、准确配准的三维点云数据集，并与长期生态与通量观测相结合。

**💡 创新点**

创新点在于：① 在极端山地环境中实现无标记、SLAM与VIO驱动的高质量多平台配准；② 将不同平台的稠密度、覆盖层次差异通过统一地面过滤与高度归一化对齐；③ 提供公开的标注、配准报告与原始传感器日志，成为可复现的数字森林基准。

**🔧 技术方法**

使用技术包括：无人机搭载YellowScan Mapper+ LiDAR、Leica BLK360摄像扫描器的视觉惯性里程计（VIO）、Frontier背包激光扫描器的实时SLAM与闭环优化、布料模拟过滤（CSF）进行地面提取、ICP与全局束调和配准、四维矩阵变换、点云分层与密度分析。

**📊 数据集**

使用的数据集为ICOS Renon（IT-Ren）控制点CP2的三平台点云（UAV‑ALS、TLS、MLS），包含约3.33亿点TLS、约2.04亿点MLS、约6.85百万点UAV‑ALS，最终以LAZ、E57、LAS等格式提供，配套有原始rosbag、GCP日志与配准报告。

**📈 对比分析**

方法比较：通过云到云距离分析、全局束误差、链路误差统计、重叠率和点密度层次图评估配准精度，结果显示MLS与ALS、TLS在大多数区域小于2 cm，整体配准误差低于1 cm；多平台结合显著提升了从基底到冠层的空间覆盖与结构分辨率，支持更精确的生物量估计。

**⚠️ 局限性**

限制包括：① 在陡坡和密林中MLS路径受限导致配准误差增加；② 低分辨率的UAV‑ALS难以穿透深层冠层；③ 多平台采集时间仍较长，未实现完全实时闭环；④ 样本仅覆盖单一站点，缺乏跨站点的普适性验证。

---

## 229. Multigrain-aware Semantic Prototype Scanning and Tri-Token Prompt Learning Embraced High-Order RWKV for Pan-Sharpening

**arXiv ID:** 2604.14622 | [PDF](https://arxiv.org/pdf/2604.14622v1)

**作者:** Junfeng Li `[一作]` (Sun Yat-sen University), Wenqi Ren `[通讯]` (Sun Yat-sen University)

**通讯引用:** 16540 | [OpenAlex ID](https://openalex.org/A5057999649)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于多粒度语义原型扫描的全新pan‑sharpening范式，采用语义驱动的RWKV扫描、三元提示（全局、原型、注册）以及可逆Q‑shift细节增强；

**💡 创新点**

1）语义导向的扫描策略消除了RWKV的位置信息偏差；2）三元提示令模型在全局与局部语义上得到更精准的调制；3）高阶RWKV采用WKV共享/时刻机制实现轻量化；4）可逆网络+Q‑shift保持高频信息且无参数膨胀；

**🔧 技术方法**

RWKV变种（高阶、WKV共享/时刻），Locality‑Sensitive Hashing（语义聚类），三元提示机制，中心差分卷积，逆向可逆网络+Q‑shift，中心差分卷积；

**📊 数据集**

WorldView‑II、WorldView‑III、GaoFen2等遥感图像数据集；

**📈 对比分析**

与传统方法（SFIM、Brovey、GS、IHS、GFPCA）以及深度学习方法（PNN、PANNet、MSDCNN、SRPPNN、GPPNN、MutNet、SFINet、PanFlowNet）进行对比；在所有指标（PSNR、SSIM、SAM、ERGAS）上均超过最优基线，PSNR提升约0.5–0.8 dB，SSIM提升约0.003–0.005；

**⚠️ 局限性**

尚未评估在极高分辨率或实时场景下的计算成本与延迟；缺乏对不同光谱通道混合情况的鲁棒性分析；

---

## 230. CoDaS: AI Co-Data-Scientist for Biomarker Discovery via Wearable Sensors

**arXiv ID:** 2604.14615 | [PDF](https://arxiv.org/pdf/2604.14615v1)

**作者:** Yubin Kim `[一作]` (Google Research), Daniel McDuff `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一套名为CoDaS的多代理系统，能够从消费者可穿戴设备收集的高维时序数据中自动执行六阶段闭环循环（数据分析、假设生成、统计/机器学习探索、对抗验证、机制推理与报告生成），以发现可解释且经过严谨检验的数字生物标志物。

**💡 创新点**

创新点包括：① 将探索、验证、批判和报告等功能拆分为专门的代理，形成对抗式审议与多轮迭代；② 引入基于文献的机制推理与先验知识过滤；③ 设计了包含 11 个检查的全流程验证电池（复制、稳定性、鲁棒性、判别力），以及事实表和数值校验机制，显著降低了信息泄漏和幻觉风险；④ 在多种病理领域（精神健康与代谢疾病）上实现跨域可迁移的自动化工作流。

**🔧 技术方法**

技术上融合了 Gemini‑3.1 Pro 及 Gemini‑3 Flash LLM、Deterministic Code Runner 与 LLM 解释器、数据探测与特征工程模块、对抗式评估（Critic‑Defender）以及结构化验证电池；系统使用共享内存、事实表、数值校验等机制确保可审计性和可复现性。

**📊 数据集**

使用了三大真实数据集：Digital Wellbeing（7,497 名受试者，睡眠/活动/心率/智能手机交互），GLOBEM（704 观测，学生级别长期行为数据），以及 Wear‑ME（1,078 名受试者，穿戴监测与完整临床血液检查），涵盖心理健康与代谢疾病两大领域。

**📈 对比分析**

与 Google AI Co‑Scientist、Biomni、Data‑Science Agent 等基线系统以及一系列专业评测基准（HealthBench、DiscoveryBench、DataSciBench 等）对比，CoDaS 在人工专家评估中获得最高质量分、最低幻觉与错误率、最优的非拒稿比例（86%），内部交叉验证的 R²/ AUC 也比基线提升数个百分点，显示出显著的性能优势。

**⚠️ 局限性**

主要局限包括：① 发现的效应量普遍较小，仍需前瞻性验证；② 依赖大量高频可穿戴数据，数据缺失与噪声高时性能下降；③ 计算成本和实现复杂度高，需专业配置；④ 仍需要人工审核机制以保证临床可行性；⑤ 机器学习模型以经典方法为主，缺乏深度学习探索，可能限制在更大规模数据上的表现。

---

## 231. Tight Bounds for Learning Polyhedra with a Margin

**arXiv ID:** 2604.14614 | [PDF](https://arxiv.org/pdf/2604.14614v1)

**作者:** Shyamal Patel `[一作]` (Columbia University), Santosh Vempala `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种在存在ρ边距的情况下，PAC学习k个半空间交集的新算法，运行时间为(k,ρ⁻¹)·O(√(n log(1/ρ) log k))，并推广到软边距情形；

**💡 创新点**

突破了先前对k或ρ⁻¹的指数依赖，获得了与SQ和密码学下界相匹配的近最优子指数时间复杂度；

**🔧 技术方法**

采用了基于凸体采样的弱学习器、分量化的半空间搜索以及基于覆盖的自适应提升（boosting）技术；

**📊 数据集**

本工作为理论分析，不依赖具体数据集；

**📈 对比分析**

与现有算法相比，在k≤n且1/n边距下实现了子指数时间，并且与已知的SQ/密码学下界仅相差对数因子，显示出显著性能提升；

**⚠️ 局限性**

仍然是子指数级别，随着n、k或ρ的增大运行时间会急剧上升；算法对边距假设敏感，并假设半空间通过原点（可通过变换实现）。

---

## 232. El Agente Forjador: Task-Driven Agent Generation for Quantum Simulation

**arXiv ID:** 2604.14609 | [PDF](https://arxiv.org/pdf/2604.14609v1)

**作者:** Zijian Zhang `[一作]`, Alán Aspuru-Guzik `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多种量子与经典计算方法进行了基准测试，比较其在分子电子结构计算中的表现

**💡 创新点**

提出了一个统一的评估框架，并对传统高性能计算与新兴量子算法在相同问题上的效率与精度进行了对比

**🔧 技术方法**

使用了DFT、TD‑DFT、VQE、DMRG、TD‑DMRG、HEOM、FMO等多种技术手段

**📊 数据集**

使用了公开的 benchmark 数据集（<https://doi.org/10.5683/SP3/0YOMKL>），其中包含多种分子结构与相关属性

**📈 对比分析**

通过与传统高性能计算方法（HPC）对比，结果显示在中等规模系统中传统方法仍占优势，而量子方法在特定子问题上展现出潜在性能提升，但整体仍受限于硬件噪声和算法深度

**⚠️ 局限性**

局限性包括样本量有限、量子硬件噪声与错误率高、对大规模系统的可扩展性不足，以及缺乏对更复杂分子体系的验证

---

## 233. Hijacking Large Audio-Language Models via Context-Agnostic and Imperceptible Auditory Prompt Injection

**arXiv ID:** 2604.14604 | [PDF](https://arxiv.org/pdf/2604.14604v1)

**作者:** Meng Chen `[一作]` (Zhejiang University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4833 | [OpenAlex ID](https://openalex.org/A5101591101)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一种名为AudioHijack的通用攻击框架，利用可视化的音频扰动实现对大型音频-语言模型（LALM）的隐蔽提示注入，从而在仅拥有音频数据的第三方攻击者身份下实现对模型行为的劫持。

**💡 创新点**

创新点包括：①使用采样式梯度估计（Gumbel‑Softmax + 直通估计）突破离散音频量化的不可微分壁垒，实现跨不同音频-文本融合方案的端到端梯度传播；②结合多上下文训练与显式注意力监督的注意力引导上下文泛化方法，使攻击对未知用户上下文具有鲁棒性；③采用卷积扰动混合技术，将扰动以可学习的混响核重分配至时频域，显著提升了注入的感知隐蔽性。

**🔧 技术方法**

主要技术手段包括：采样式梯度估计（Gumbel‑Softmax + 直通估计）、多上下文训练与注意力监督（attention loss）、卷积扰动混合（frame‑wise convolution + 过渡平滑 + RMS归一化）、基于目标响应的对抗优化、以及在不同模型上评估的精确度指标（PISR/BMSR）。

**📊 数据集**

实验使用的公开数据集包括 AirBench（音频问答）与 VoiceBench（野外语音与指令），以及针对工具调用的多模态测试数据；在 13 种不同架构（离散、连续、混合）与规模的 LALM 上进行评估，并在 Microsoft Azure 与 Mistral AI 的商业语音代理上进行真实世界攻击验证。

**📈 对比分析**

与现有直接/间接提示注入、音频对抗样本等方法对比，AudioHijack 在 6 类误行为（听觉盲区、拒绝提示、错误信息、钓鱼、人格控制、工具滥用）中均取得 79%–96% 的 PISR 与 84%–94% 的 BMSR，且在商业代理上平均 BMSR 达 0.53–0.98；在隐蔽性上 SNR 超过 28 dB、MCD 小于 4.2，显著优于加法扰动（SNR < 15 dB）。

**⚠️ 局限性**

主要局限性：①攻击需对目标 LALM 的结构与参数有充分了解（梯度可获取），跨模型泛化受限；②仅测试了 6 类误行为，未覆盖全部潜在攻击场景；③对实时或边缘设备的评估不足，实际部署中对资源与推理时延的影响未完全评估；④对抗样本的生成在训练时需要多语义上下文与较大计算资源。

---

## 234. A Synonymous Variational Perspective on the Rate-Distortion-Perception Tradeoff

**arXiv ID:** 2604.14603 | [PDF](https://arxiv.org/pdf/2604.14603v1)

**作者:** Zijian Liang `[一作]` (Beijing University of Posts and Telecommunications), Ping Zhang `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 114841 | [OpenAlex ID](https://openalex.org/A5100405781)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

提出了基于同义性语义信息理论的变分视角，重新定义感知重建为在源信号对应的同义集合内恢复任意可接受样本，并构建了同义源编码架构及其变分推理框架。

**💡 创新点**

核心创新在于：①用同义集合重构目标推导出分布性散度项的理论来源；②引入同义变分推理（SVI）与同义变分下界（SVLBO）来分析同义编码；③提出同义性-感知一致性原理，将语义层的最优识别与句法层的感知优化理论对应；④推导出同义率-失真-感知（RDP）tradeoff，统一了现有RDP与经典RD理论。

**🔧 技术方法**

使用了信息理论中的变分推理、KL散度、熵模型、概率密度估计等技术；在推导中还利用了贝叶斯公式、Jensen不等式、期望展开等数学工具。

**📊 数据集**

文中未给出具体实验或数据集，仅在理论层面进行推导与证明。

**📈 对比分析**

由于缺乏实验对比，本文未报告具体性能指标；理论上指出同义源编码在可实现的最优点上能达到或优于传统RDP编码，但实际效果需进一步实验验证。

**⚠️ 局限性**

局限性包括：①同义集合的构造与大小未知，实际实现依赖于对语义一致性的假设；②推导中大量假设连续可微、完美估计等理想条件；③对生成模型的分布逼近存在难度，导致KL散度项难以完全消除；④缺乏实验验证，理论到实践的转化尚未证明其实际优势。

---

## 235. CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification

**arXiv ID:** 2604.14602 | [PDF](https://arxiv.org/pdf/2604.14602v1)

**作者:** Yian Wang `[一作]` (University of Illinois Urbana Champaign), Hari Sundaram `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于因果机制的LLM去毒框架，通过概率必然性与充分性（PNS）识别导致毒性生成的注意力头，并对这些头进行局部推理时干预与基于PNS的微调。

**💡 创新点**

创新点包括：① 使用PNS进行因果头选择，精准定位最小必要且充分的毒性相关头；② 设计上下文感知的局部推理时干预，使干预方向随输入动态变化；③ 以PNS下界为目标进行头级微调，实现永久去毒；④ 构建对齐毒性/非毒性句子对的因果基准数据集。

**🔧 技术方法**

技术手段：PNS概率计算与下界估计、变分自编码器（VAE）建模隐含混淆变量、推理时干预（ITI）与局部动态干预、PNS驱动的头级微调、头级选择与线性探测对比、基准数据集生成。

**📊 数据集**

使用的数据集包括：ToxiGen、Implicit Hate、ParaDetox，以及新构建的对齐毒性/非毒性句子对基准（通过Vicuna-13B生成）。

**📈 对比分析**

与基线（无干预）和基于准确率的ITI对比，在ToxiGen、Implicit Hate、ParaDetox三大数据集上，所提方法在4种开源LLM（LLaMA‑3‑8B、Vicuna‑7B、Mistral‑7B、Qwen‑7B）上平均降低毒性约5.34%，保持或提升困惑度和流畅度，并实现头选择速度提升7倍。

**⚠️ 局限性**

局限性：局部干预计算开销大；基准依赖Vicuna‑13B生成，可能携带生成模型偏差；评估主要基于自动指标，难以完全覆盖人类细粒度判断；仅在英语上验证，跨语言适用性待验证；PNS下界估计依赖VAE能否充分捕获混淆变量，可能导致因果集合不准确；方法可能被滥用于内容审查。

---

## 236. Prompt-Guided Image Editing with Masked Logit Nudging in Visual Autoregressive Models

**arXiv ID:** 2604.14591 | [PDF](https://arxiv.org/pdf/2604.14591v1)

**作者:** Amir El-Ghoussani `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vasileios Belagiannis `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 3971 | [OpenAlex ID](https://openalex.org/A5027065196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Mask Logit Nudging (MLN)，在视觉自回归(VAR)模型中实现无逆训练、无额外训练、无推理步骤的提示引导图像编辑；

**💡 创新点**

1）在logit空间进行软 nudging，将源图像token与目标提示对齐；2）基于 cross‑attention 差异构造空间掩码，仅在编辑区域进行引导；3）加入量化误差投影修正提升重构质量；4）方法架构无关，兼容所有 VAR 体系；

**🔧 技术方法**

VAR（SWITTI/Infinity）编码/解码、softmax 概率插值、交叉注意力掩码、量化误差投影修正、无逆提示引导；

**📊 数据集**

PIE‑Benchmark（700幅图、10类编辑场景）用于编辑评估；COCO 验证集（5k图）用于零编辑重构；OpenImages（1k图）用于高分辨率重构；高分辨率提示通过 GPT‑4V 生成；

**📈 对比分析**

与 AREdit、VARIN 及扩散/流模型（InvSR）进行对比。结果显示：512×512 时 PSNR 最高、LPIPS 最低、CLIP 相似度最高、速度 0.82 s；1024×1024 时速度 1.6 s，性能优于扩散模型且显著更快；

**⚠️ 局限性**

对色彩/纹理风格编辑的量化修正可能引入失真；受 VAR token granularity 限制，细粒度编辑效果受限；对极大/超高分辨率场景仍需进一步优化。

---

## 237. Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems

**arXiv ID:** 2604.14585 | [PDF](https://arxiv.org/pdf/2604.14585v1)

**作者:** Xing Zhang `[一作]` (AWS Generative AI Innovation Center), Peiyang He `[通讯]` (AWS Generative AI Innovation Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了复合 AI 系统中提示优化的有效性，验证了两大假设并提出两阶段诊断框架。

**💡 创新点**

首次系统评估提示互作和可优化性，发现代理几乎不耦合并给出“可利用结构”诊断，提出可复制的 ANOVA+headroom 测试。

**🔧 技术方法**

使用完整网格评估、六种提示优化方法（APE、OPRO、EvoPrompt、PromptBreeder、DSPy‑style、PROSE）以及两路 ANOVA 分解与头部测试技术。

**📊 数据集**

使用 HotpotQA、MBPP、XSum、Feedback‑Bench、HelpSteer2、WildBench 等数据集，并以 20 条训练样本与 100 条测试样本进行评估。

**📈 对比分析**

与零射击基线对比，绝大多数任务平均收益为负，唯一例外 HelpSteer2 在 Claude Haiku 上全部方法提升至 +6.8 分。

**⚠️ 局限性**

仅评估两代理流水线、mid‑tier 模型、10 条训练样本；更深或循环结构、前沿模型可能表现不同。

---

## 238. MapSR: Prompt-Driven Land Cover Map Super-Resolution via Vision Foundation Models

**arXiv ID:** 2604.14582 | [PDF](https://arxiv.org/pdf/2604.14582v1)

**作者:** Ruiqi Wang `[一作]` (Beijing Foreign Studies University), Hanlin Wu `[通讯]` (Beijing Foreign Studies University)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5067703101)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于提示（prompt）的高分辨率土地覆被图像超分辨率框架 MapSR，利用低分辨率标注一次性构造类别提示后在冻结的视觉基础模型特征空间中进行无训练推理。

**💡 创新点**

创新点在于将监督与模型训练解耦，使用轻量线性探测器一次性生成类提示，随后通过度量推理和图基细化实现高分辨率预测，显著减少参数量与训练时间。

**🔧 技术方法**

核心技术包括：冻结的 DINOv2 ViT 特征提取、基于注意力的特征上采样、线性探测与高置信度聚合构造提示、余弦相似度度量推理，以及基于超级像素的图传播细化。

**📊 数据集**

使用 Chesapeake Bay 数据集（覆盖美国六州的 6000×7500 像素 1 m 分辨率图像、30 m NLCD 低分辨率标注和 1 m CCLC 真实标签）。

**📈 对比分析**

与无监督、弱监督、全监督基线对比，MapSR 在 mIoU 上达到 59.64%（无 HR 标注），仅用 4K 可训练参数，训练仅 18 分钟，表现优于最强弱监督 Baseline (Paraformer 59.54%)，并逼近全监督方法（UNetFormer 57.70%）。

**⚠️ 局限性**

局限性包括对低分辨率标注的依赖与可能的时间不匹配导致的提示误差，以及在跨地区迁移或多时相数据中提示的泛化能力仍需进一步验证。

---

## 239. Constructions of $q$-ary Golay Complementary Pairs Over Flexible Non-Power-of-Two Lengths

**arXiv ID:** 2604.14667 | [PDF](https://arxiv.org/pdf/2604.14667v1)

**作者:** Zhiye Yang `[一作]` (Northwest University), Keqin Feng `[通讯]` (Tsinghua University)

**通讯引用:** 2495 | [OpenAlex ID](https://openalex.org/A5108616637)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文通过扩展布尔函数(EBF)构造了q-ary Golay互补对(GCP)，并证明了四元GCP长度M存在与任意正整数h,m下(4h)-ary GCP长度2^mM存在的必要且充分条件。

**💡 创新点**

创新点在于：①提出了四元GCP长度M存在与任意阶码(4h) GCP长度2^mM存在的等价性；②利用EBF实现了比以往更灵活的长度范围（非2的幂长），大幅扩展了可构造GCP的长度集合。

**🔧 技术方法**

采用的技术主要是：扩展布尔函数(EBF)、通用布尔函数(GBF)、符号映射与复杂序列构造，以及对自相关函数的组合分析。

**📊 数据集**

本研究为理论构造，无使用任何实验数据集。

**📈 对比分析**

通过与以往仅能构造特定长度（如2^h, 5·2^m-3, 13·2^m-4等）GCP的方法对比，论文展示了在更广泛的非2幂长度下仍能构造GCP的能力；因其为理论证明，未给出实验性能指标。

**⚠️ 局限性**

局限性：①依赖于底层四元GCP的存在性——若某长度M的四元GCP未知，则无法直接得到对应的(4h)-ary GCP；②构造只针对4h-ary字母表，对其他阶数（如三元、五元等）不适用；③虽然提供了必要充分条件，但并未给出对所有M的显式构造公式，仍需依赖已有的四元GCP实例。

---

## 240. World-Value-Action Model: Implicit Planning for Vision-Language-Action Systems

**arXiv ID:** 2604.14732 | [PDF](https://arxiv.org/pdf/2604.14732v1)

**作者:** Runze Li `[一作]` (Westlake University), Donglin Wang `[通讯]` (Westlake University)

**通讯引用:** 1560 | [OpenAlex ID](https://openalex.org/A5100665183)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了World–Value–Action（WAV）模型，实现了在视觉-语言-动作任务中的隐式轨迹规划；

**💡 创新点**

通过在潜在空间中进行规划并迭代推断，解决了传统直接动作空间搜索在长时程任务中可行轨迹指数衰减的问题；

**🔧 技术方法**

使用基于扩散Transformer的多模态视频生成模块、轨迹价值评估模块和动作解码模块，并采用流匹配训练以及MPPI式的潜在空间迭代推断；

**📊 数据集**

在LIBERO仿真基准（包含Spatial、Object、Goal、Long四个子任务）和真实双臂机器人平台Piper上进行评估；

**📈 对比分析**

与多种基线（包括Diffusion Policy、OpenVLA、DreamVLA、GE-ACT等）比较，WAV在LIBERO的平均分达到98.1%，在真实任务的成功率从35.6%提升至75.6%，尤其在长时程和组合任务上显著优于对手；

**⚠️ 局限性**

模型在推理时间和GPU显存方面仍存在一定开销，且需要更多的计算资源，未来需要进一步优化实现以实现实时闭环部署。

---

## 241. A Mechanistic Account of Attention Sinks in GPT-2: One Circuit, Broader Implications for Mitigation

**arXiv ID:** 2604.14722 | [PDF](https://arxiv.org/pdf/2604.14722v1)

**作者:** Yuval Ran-Milo `[一作]` (Tel Aviv University), Shahar Mendel `[通讯]` (Tel Aviv University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 GPT‑2 样式 Transformer 的注意力 sink 进行因果与结构分析，发现其由查询偏置、第一层 MLP 对位置编码的变换以及键投影结构的相互作用形成。

**💡 创新点**

证明 sink 可通过多条独立电路实现，且每个关键组件单独移除后仍可出现 sink，说明 sink 不是单一机制所致。

**🔧 技术方法**

使用结构分析（EPE、cosine similarity、坐标级对齐）、10 种针对性干预（如零化查询偏置、删除/交换位置编码、抑制 MLP、零化关键投影等）以及 BOS‑attention 指标来评估干预效果。

**📊 数据集**

实验数据集包括 300 条示例，分别来自 SST‑2（自然语言）、GSM8K（数学推理）和 HumanEval（代码生成），每条示例固定为 40 个 token。

**📈 对比分析**

通过对比干预前后的 BOS‑attention 百分比（从 100% 降至 3–44% 等），展示了每个组件对 sink 的必要性；实验表明任何单个组件的干预均显著削弱 sink，控制干预保持 sink 不变。

**⚠️ 局限性**

局限性：仅在 GPT‑2 样式模型上进行静态分析，未探究 sink 在不同架构（如 RoPE、ALiBi）或更大规模模型中的出现机制；未追踪训练过程中的学习动态；干预虽显著削弱但未完全消除 sink，表明仍存在二级贡献者。

---

## 242. Chain-of-Glimpse: Search-Guided Progressive Object-Grounded Reasoning for Video Understanding

**arXiv ID:** 2604.14692 | [PDF](https://arxiv.org/pdf/2604.14692v1)

**作者:** Zhixuan Wu `[一作]` (Beijing University of Posts and Telecommunications), Soujanya Poria `[通讯]` (Nanyang Technological University)

**通讯引用:** 23117 | [OpenAlex ID](https://openalex.org/A5033376109)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e0540dec-d77f-42db-94ae-d039248f6393` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Chain-of-Glimpse框架，利用搜索引导的多步对象对齐推理实现视频理解；

**💡 创新点**

通过构造对象-视频推理公式（OGVRF）、MCTS搜索生成多路径轨迹、强化学习优化多步决策（MTDP）实现逐步、可解释的视觉证据追踪，显著提升时序与对象关系推理；

**🔧 技术方法**

MCTS、蒙特卡洛树搜索、强化学习（PPO/GRPO）、多模态大语言模型（Qwen2.5-VL-3B/7B）、LoRA微调、对象检测与视觉特征提取；

**📊 数据集**

SA2VA、NExTQA、Video-Holmes、CG-Bench-Reasoning、VRBench、EgoSchema、STAR等视频问答与推理基准；

**📈 对比分析**

与多种专有与开源模型（GPT-4o、LLoVi、VideoAgent等）对比，在NExTQA上7B版本实现83.3%（高于基线79.7%），在Video-Holmes、CG-Bench和VRBench等外域基准上平均提升约3-5%，并在EgoSchema和STAR上同样超越基线；

**⚠️ 局限性**

仅使用Qwen2.5-VL-3B/7B两款相对较小的多模态模型，受限于模型容量与预训练深度，难以处理更复杂推理与更长视频；未来需引入更大模型、改进搜索与预训练策略以提升性能与效率。

---

## 243. Switching Efficiency: A Novel Framework for Dissecting AI Data Center Network Efficiency

**arXiv ID:** 2604.14690 | [PDF](https://arxiv.org/pdf/2604.14690v1)

**作者:** Niangen Ye `[一作]` (Shanghai Jiao Tong University), Weisheng Hu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 13094 | [OpenAlex ID](https://openalex.org/A5015039354)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种名为 Switching Efficiency Framework 的分层度量框架，用来从计算有效数据吞吐量与网络交换资源总量的角度评估 AI 数据中心网络在大语言模型训练中的通信效率，并将总效率拆分为数据、路由和端口利用三个细粒度因子进行诊断。

**💡 创新点**

创新点在于①提出了直接衡量“计算有效数据”吞吐量的 Switching Efficiency 指标；②将指标细化为数据冗余、路由多跳和端口利用率三层因子，实现对通信瓶颈的可解释性分解；③通过综合考虑多种网络设计参数（带宽分层、服务器规模、网络计算、集群规模）展示框架的通用性和可优化路径。

**🔧 技术方法**

主要技术包括：基于通信原语（All‑Reduce、All‑Gather、All‑to‑All 等）的数据流建模；使用基于 GPU、服务器和交换层的端口速率集合计算总交换容量；对每个原语计算有效数据量；通过公式拆解 η 为 γ、δ、θ 并进一步推广至长期平均指标。

**📊 数据集**

数据集主要是两类训练工作负载：基于 GPT‑3 的稠密模型工作负载以及基于 DeepSeek‑V3 的 Mixture‑of‑Experts（MoE）模型工作负载；这些工作负载通过对并行度（DP、PP、TP/EP）进行枚举并按比例缩放层数、隐藏维度、专家数和批量大小来构造。

**📈 对比分析**

比较方法：在 4096‑GPU 规模下对 3D‑Torus 与 Rail‑Optimized 两种网络架构进行细粒度（γ、δ、θ）和整体（η、μ）效率分解；进一步通过调整带宽分层比例、服务器规模、网络计算、集群规模等参数进行敏感性分析。性能结果显示 Rail‑Optimized 在密集模型下 η≈0.32、MoE 模型下 η≈0.046，远高于 3D‑Torus（密集模型 η≈0.21、MoE 模型 η≈0.004），且在所有细粒度因子上均表现更优。

**⚠️ 局限性**

局限性包括：①框架基于理想化的网络模型，未考虑硬件延迟、错误恢复和多任务并发情况；②只在仿真环境下验证，缺乏真实集群测量；③假设通信原语实现均为最优路由，未考虑动态调度和拥塞控制的实际影响；④对大规模 All‑to‑All 传输的分析仍受限于模型简化，可能低估了实际网络开销。

---

## 244. DR$^{3}$-Eval: Towards Realistic and Reproducible Deep Research Evaluation

**arXiv ID:** 2604.14683 | [PDF](https://arxiv.org/pdf/2604.14683v1)

**作者:** Qianqian Xie `[一作]` (Nanjing University), Jiaheng Liu `[通讯]` (Nanjing University)

**通讯引用:** 128068 | [OpenAlex ID](https://openalex.org/A5100358128)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 DR^3‑Eval 基准，用于评估深度研究代理在多模态、多文件报告生成任务中的表现。

**💡 创新点**

引入可重复的沙盒环境、逆向任务构造和五维度评估框架（信息召回、事实准确度、引用覆盖、指令遵循、深度质量），解决了实时 Web 评估的不确定性与歧义。

**🔧 技术方法**

采用多代理架构（主代理 + RAG 子代理 + 文件读取子代理）、ReAct 迭代检索、文本嵌入、LLM‑as‑judge、多模态感知工具等技术。

**📊 数据集**

使用用户真实提供的多模态文件（文本、图片、视频、表格等）以及基于这些文件构建的静态沙盒文档集合。

**📈 对比分析**

与 GPT‑4.1、Claude Sonnet 4、Gemini‑2.5‑Pro、Qwen‑3 等多大语言模型在 5 维度指标上对比，结果显示即使最强模型在信息召回、引用覆盖和事实准确度上也低于 70%。

**⚠️ 局限性**

受限于沙盒对实时网络动态性的近似、模型仍易受检索失败与幻觉影响，以及缺乏更大规模、多领域的验证。

---

## 245. Acceptance Dynamics Across Cognitive Domains in Speculative Decoding

**arXiv ID:** 2604.14682 | [PDF](https://arxiv.org/pdf/2604.14682v1)

**作者:** Saif Mahmoud `[一作]` `[通讯]` (Al Ain University), Saif Mahmoud (Al Ain University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在四个不同认知领域（代码生成、数学推理、逻辑推理和聊天）上，对树形推测式解码（tree‑based speculative decoding）的接受率动态进行了系统的实验研究。

**💡 创新点**

主要创新在于：①首次量化任务类型对接受率的影响，发现聊天域唯一可实现 E[L]>1，带来显著的推理加速；②发现接受率随树深度略有提升，提出上下文承诺效应；③揭示熵与接受率呈弱负相关，并提出聊天域“高熵高接受率”悖论；④为后续的域感知推测策略和域特定草稿模型选择提供实证依据。

**🔧 技术方法**

使用了 1.1B Llama‑2 作为草稿模型、4‑bit GPTQ Llama‑2‑7B‑Chat 作为目标模型；构建深度 3、分支因子 2 的树；采用最小值接受规则 α= min(1, p_target/p_draft)；计算目标熵、期望接受长度 E[L] 及相关统计。

**📊 数据集**

实验数据集包括：代码生成 – CodeNet、数学推理 – MATH（代数子集）、逻辑推理 – 适用的 arithmetic word problem benchmark、聊天 – 大规模 RLHF 对话集（多轮指令‑跟随对话）。

**📈 对比分析**

通过统计每个域的接受率、E[L]、深度‑接受曲线和熵‑接受相关性进行比较。结果显示：聊天域 E[L]=1.065，带来正向加速；代码、逻辑推理略低（E[L]≈0.95–0.98，速度几乎无提升或略差）；数学域 E[L]=0.914，实际会增加推理延迟。深度增加对接受率影响微弱但一致正向。

**⚠️ 局限性**

局限性包括：仅使用贪心采样（温度 0）导致最高接受率；树深度、分支因子设置有限，未探索更深/更宽的树；仅评估单一草稿‑目标配对，未验证对其他模型族的泛化；未考察生成质量（如代码通过率、数学正确率）等后端指标。

---

## 246. Catching Every Ripple: Enhanced Anomaly Awareness via Dynamic Concept Adaptation

**arXiv ID:** 2604.14726 | [PDF](https://arxiv.org/pdf/2604.14726v1)

**作者:** Jiaqi Zhu `[一作]` (Beijing Institute of Technology), Wenqiao Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 20937 | [OpenAlex ID](https://openalex.org/A5100441502)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 DyMETER，一种统一的在线异常检测框架，能够在概念漂移时通过实例感知的模型演化与动态阈值校准实时适配新概念。

**💡 创新点**

核心创新包括：利用超网络根据当前实例生成参数偏移，实现无梯度更新的即时模型演化；采用证据深度学习衡量概念不确定性；将漂移检测、模型演化与阈值优化三者在单一在线范式下耦合；并提供轻量化漂移控制与离线复合更新策略。

**🔧 技术方法**

技术实现基于自编码器、超网络（Hypernetwork）、证据深度学习（EDL）、动态阈值优化（DTO）、实例级不确定性评估与滑动窗口统计。

**📊 数据集**

在19个真实世界数据集（Ionosphere、Pima、Satellite、Mammography、BGL、NSL-KDD、KDD99、HEXAGON时间序列等）以及4个基于MNIST/FMNIST的合成数据集和INSECTS系列温度漂移数据上进行实验。

**📈 对比分析**

与18个基准（传统、增量、集成与漂移自适应方法）在AUCROC/AUCPR、速度与内存等指标上对比，DyMETER在绝大多数场景下均实现了领先或相近的最高性能，平均提升约4–5%，且推理时延低于多数基准。

**⚠️ 局限性**

局限性包括：离线更新不一定能提升性能；重建式模型对异常表达能力有限；对阈值参数（μ_p、μ_e、μ_o等）敏感，需手工调优；在极端漂移或历史样本严重不足时仍可能表现下降。

---

## 247. SGA-MCTS: Decoupling Planning from Execution via Training-Free Atomic Experience Retrieval

**arXiv ID:** 2604.14712 | [PDF](https://arxiv.org/pdf/2604.14712v1)

**作者:** Xin Xie `[一作]` (Ant Digital Technologies, Ant Group), Peng Zhang `[通讯]` (Ant Digital Technologies, Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SGA-MCTS框架，将LLM规划转化为非参数检索，通过离线MCTS挖掘高质量的推理路径并提取State‑Goal‑Action（SGA）原子，在线时检索这些原子作为软提示，生成动作，显著降低推理延迟并保持深度推理能力。

**💡 创新点**

创新点在于：①将复杂多步推理离线搜索，生成可复用的去词化SGA原子；②混合符号‑语义检索机制，既捕获语义相似性又校验符号可执行性；③通过检索而非参数微调实现开源模型在未微调的情况下接近SOTA；④将系统2的深度规划与系统1的实时执行有效分离。

**🔧 技术方法**

使用技术包括：Monte Carlo Tree Search（MCTS）作为离线高保真数据生成器；状态抽象与去词化抽象（State‑Goal‑Action原子）；双重检索评分（语义相似 + 符号可执行性）；生成式决策生成器（Decision Maker）将检索到的原子投射回具体上下文；以及无参数非训练化的经验库。

**📊 数据集**

实验使用的主要数据集有：StableToolBench（跨难度迁移）、ToolHop（多跳工具链）、BFCL v3（多轮对话状态跟踪）。模型基线包括Qwen3系列（8B/14B/32B）、ReAct、LangMem以及GPT‑5（思考模式）。

**📈 对比分析**

与ReAct、LangMem和GPT‑5等基线对比，SGA‑MCTS在三大数据集上平均提升13‑46%的成功率，8B模型甚至超过32B基线，Token消耗下降76%；在最难任务上保持61%成功率，整体性能逼近GPT‑5（差距<5%）甚至在BFCL v3上超越GPT‑5。

**⚠️ 局限性**

主要限制包括：性能受限于离线MCTS搜索质量，若搜索覆盖不足会导致低质量原子；经验库构建依赖于预设的种子问题，缺乏自动扩展机制，难以完全覆盖所有任务分布。

---

## 248. The Courtroom Trial of Pixels: Robust Image Manipulation Localization via Adversarial Evidence and Reinforcement Learning Judgment

**arXiv ID:** 2604.14703 | [PDF](https://arxiv.org/pdf/2604.14703v1)

**作者:** Songlin Li `[一作]` (Xinjiang University), Gaobo Yang `[通讯]` (Hunan University)

**通讯引用:** 9382 | [OpenAlex ID](https://openalex.org/A5089193327)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种 courtroom‑style 判决框架，用双流（起诉、辩护）对图像篡改区域进行对抗性证据生成，并由 RL‑驱动的审判者在不确定区域进行重推理与置信度校准，完成范围定位。

**💡 创新点**

创新点包括：① 明确把篡改证据与真实性证据对立建模；② 通过动态辩论机制实现双流互相制衡与信息推拉；③ 引入 RL（Actor‑Critic + Gumbel‑Softmax）审判者，使用 soft‑IoU 奖励和对称 KL 一致性进行置信度校准；④ 结合边缘先验和频域特征提升鲁棒性。

**🔧 技术方法**

技术栈：双假设分割网络、跨流注意力+对抗抑制、动态辩论（push‑pull 更新）、边缘提取（Laplace + CBAM）、EFM、RL 判决器（Actor‑Critic、Gumbel‑Softmax、soft‑IoU 奖励）、可靠性校准（entropy + SymKL）及多源证据融合。

**📊 数据集**

数据集：CASIAv1/2、NIST16、Columbia、Korus、DSO、IMD2020；以及社交媒体压缩场景（Facebook、WeChat、Weibo、WhatsApp）进行鲁棒性评测。

**📈 对比分析**

与多种 state‑of‑the‑art IML 方法（PSCC‑Net、Trufor、IML‑ViT、MFI‑Net、Sparse‑ViT、PIM、Mesorch）在 ID/OOD 上按 F1 分数比较，ID 上平均 0.526、OOD 上 0.401，均超前竞争对手，尤其在压缩与噪声扰动下保持高稳健性。

**⚠️ 局限性**

局限性：整体架构较为复杂，训练/推理成本较高；RL 权重对性能敏感，需仔细调参；审判者以 patch 为粒度，可能缺乏对极细边缘或全局一致性的捕捉，影响极端场景下的精度。

---

## 249. M2-PALE: A Framework for Explaining Multi-Agent MCTS--Minimax Hybrids via Process Mining and LLMs

**arXiv ID:** 2604.14687 | [PDF](https://arxiv.org/pdf/2604.14687v1)

**作者:** Yiyu Qian `[一作]` (RMIT University), Tim Miller `[通讯]` (University of Queensland)

**通讯引用:** 8004 | [OpenAlex ID](https://openalex.org/A5028824146)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了将浅层Minimax搜索嵌入多代理MCTS的rollout阶段，并通过过程挖掘与LLM生成可解释说明；

**💡 创新点**

创新点在于将过程挖掘算法与LLM结合，为MCTS–Minimax混合模型提供因果与远程解释，并验证Inductive Miner在小规模跳棋中优于Alpha Miner和iDHM；

**🔧 技术方法**

使用的技术包括过程挖掘（Alpha Miner、iDHM、Inductive Miner）、LLM（GPT‑5）生成自然语言解释、Python实现的MCTS–Minimax、ProM框架评估模型；

**📊 数据集**

使用的数据集为100局3v3跳棋游戏产生的事件日志，记录双方行动序列；

**📈 对比分析**

通过Replay Fitness、Trace/Move Fitness等指标比较三种发现算法，结果显示Inductive Miner在多数实验中实现完美Fitness，其他算法则出现低于1的Fitness；

**⚠️ 局限性**

限制包括仅评估Replay Fitness，未覆盖precision、simplicity、generalization；未深入分析Minimax内部决策；缺乏全局胜率关联和大规模环境的可扩展性验证。

---

## 250. AIPC: Agent-Based Automation for AI Model Deployment with Qualcomm AI Runtime

**arXiv ID:** 2604.14661 | [PDF](https://arxiv.org/pdf/2604.14661v1)

**作者:** Jianhao Su `[一作]` (Qualcomm Technologies Inc), Weidong Feng `[通讯]` (Qualcomm Technologies Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出 AIPC（AI Porting Conversion）框架，利用 LLM 代理在 QAIRT 端侧推理平台上实现模型从 PyTorch 到可执行 QNN/SNPE 的自动化部署。

**💡 创新点**

创新点在于将部署流程拆分为可验证阶段、通过 Agent Skills 注入平台知识、实现失败定位与有限修复，以及将人工干预视为可持续的知识库，形成可迭代的自动化工作流。

**🔧 技术方法**

采用的技术包括 LLM 代理执行代码与命令、ONNX 与 QAIRT 转换工具、自动上下文二进制编译、量化与校准流程、以及基于金标验证的循环回调。

**📊 数据集**

实验基准涵盖了 ESRGAN、YOLOv8、LPRNet、YOLO-World、YOLO26、Whisper 和 DeepSeek‑R1 等多种结构与规模的模型。

**📈 对比分析**

在结构规则的视觉模型上，AIPC 能在 7–20 分钟内完成从 PyTorch 到 QNN/SNPE 的部署，API 成本约 0.7–10 美元；在复杂模型（如 Whisper、DeepSeek‑R1）仍需人工介入，但能显著缩短调试时间。

**⚠️ 局限性**

主要局限包括：对未支持的算子仍需专家手工修补；代理在工作流漂移、环境判别不准时会导致错误累积；跨平台和动态形状模型的自动化程度不足；以及当前仅以案例观察为主，缺乏大规模基准验证。

---

## 251. ClariCodec: Optimising Neural Speech Codes for 200bps Communication using Reinforcement Learning

**arXiv ID:** 2604.14654 | [PDF](https://arxiv.org/pdf/2604.14654v1)

**作者:** Junyi Wang `[一作]` (Tsinghua University), Chao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 84057 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种名为 ClariCodec 的神经语音编解码器，专为 200 bps 极低比特率环境设计，并通过强化学习优化语义保留，使语音可懂度显著提升。

**💡 创新点**

创新点在于：① 将量化过程重新定义为可微的随机策略；② 利用 GRPO 与 WER 奖励直接对可懂度进行强化学习；③ 在训练的两阶段框架中，第二阶段仅冻结解码器与声码器，专门微调编码器，从而在保持声学质量的同时提升 WER。

**🔧 技术方法**

技术包括：改进的 FSQ 量化、可逆层归一化（ILN）、ConvNeXt V2 编码器、Vocos 语音合成器、Gumbel‑Softmax 采样、Group Relative Policy Optimization (GRPO)、音频重建损失与 WER 奖励的组合。

**📊 数据集**

使用 LibriHeavy（约 50,000 小时）进行训练，评估使用 LibriSpeech 的 test‑clean 与 test‑other 子集，音频采样率 16 kHz。

**📈 对比分析**

与 8 种现有神经语音编解码器（码率 250–750 bps）对比，ClariCodec 在 200 bps 码率下取得 3.20% WER（test‑clean）与 8.93% WER（test‑other），相较于未强化学习版本提升 13% 相对 WER，且在 STOI、PESQ、UTMOS 等声学质量指标上保持竞争力。

**⚠️ 局限性**

局限性包括：1) 目前为非实时、非流式架构，导致延迟较高；2) 在极低码率下，语音质量与可懂度存在权衡，RL 训练仍可能略微损失声学细节；3) 只使用 WER 作为奖励，未充分考虑其他语义或情感维度。

---

## 252. G-MIXER: Geodesic Mixup-based Implicit Semantic Expansion and Explicit Semantic Re-ranking for Zero-Shot Composed Image Retrieval

**arXiv ID:** 2604.14710 | [PDF](https://arxiv.org/pdf/2604.14710v1)

**作者:** Jiyoung Lim `[一作]` (Sungkyunkwan University), Jee-Hyong Lee `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1895 | [OpenAlex ID](https://openalex.org/A5067651075)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的零样本合成图像检索方法G‑MIXER，利用地球混合技术捕获隐式语义并通过显式语义重新排序提升检索多样性与精度。

**💡 创新点**

创新点在于：①使用地球路径多比例混合（Geodesic Mixup）在图像与文本特征空间生成多样化查询，扩大检索范围；②引入基于多模态大型语言模型的显式属性重排序（Explicit Semantic Re‑ranking）剔除噪声。

**🔧 技术方法**

核心技术包括CLIP预训练视觉‑语言编码器、GPT‑4o等多模态LLM进行目标描述与属性生成，以及基于余弦相似度的多阶段检索与重新排序。

**📊 数据集**

在四大零样本合成检索基准上评测：CIRCO、CIRR、FashionIQ、GeneCIS。

**📈 对比分析**

与训练式与训练无关的基线（SEARLE、PrediCIR、CIReVL、LDRE、OSrCIR等）对比，G‑MIXER在所有数据集均实现了显著提升，CIRCO上mAP@50提升至32.39%，CIRR上Recall@50达77.69%，FashionIQ和GeneCIS的Recall@1–3均名列第一。

**⚠️ 局限性**

主要局限包括：检索过程中对LLM推理时间占比高（约0.6s/查询），且依赖大规模LLM的生成质量；目前仍缺乏针对不同视觉域的适配机制，可能在极端语义变化或极端数据稀缺场景表现不足。

---

## 253. Expressivity of Transformers: A Tropical Geometry Perspective

**arXiv ID:** 2604.14727 | [PDF](https://arxiv.org/pdf/2604.14727v1)

**作者:** Ye Su `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Yong Liu `[通讯]` (Renmin University of China)

**通讯引用:** 89771 | [OpenAlex ID](https://openalex.org/A5115602439)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

利用热化简（Maslov dequantization）将Transformer中的自注意力映射到零温度极限，证明其等价于Power Voronoi图，从而以拓扑方式精确计数Transformer的线性区域，并给出多头注意力与网络深度对表达能力的组合上界与下界。

**💡 创新点**

①首次将Transformer的自注意力与tropical几何（Newton多面体、Minkowski和）相结合，得到多头注意力的组合复杂度可达O(N^H)。②提出零温度下的Power Voronoi等价性，给出Θ(N^{d_model}L)的紧致线性区域上界，并证明该上界在实际温度下保持指数级稳定。

**🔧 技术方法**

tropical几何、Maslov dequantization、Voronoi与Power Voronoi图、Newton多面体与Minkowski和、超平面排列理论、实验中的Monte‑Carlo采样与凸包计算。

**📊 数据集**

实验使用合成数据：二维查询空间、正态分布的键向量与随机初始化的权重；未使用真实任务数据集。

**📈 对比分析**

通过理论证明与实验低阶下界对比，验证多头注意力和网络深度导致线性区域指数增长；实验中未给出传统任务上的性能指标，主要关注理论上表达能力的量化与可视化验证。

**⚠️ 局限性**

未考虑标准化层（如LayerNorm）对Softmax的非平滑影响；未考虑训练过程中数据驱动的权重分布与优化动态；在实际模型中数值数目远小于理论上限，且对大规模自然语言/视觉任务的泛化能力未作实验评估。

---

## 254. The Agentification of Scientific Research: A Physicist's Perspective

**arXiv ID:** 2604.14718 | [PDF](https://arxiv.org/pdf/2604.14718v1)

**作者:** Xiao-Liang Qi `[一作]` (Stanford University), Xiao-Liang Qi `[通讯]` (Stanford University)

**通讯引用:** 27056 | [OpenAlex ID](https://openalex.org/A5111546697)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文概述了大语言模型对科学研究的深远影响，阐述了从工具到合作伙伴的转变，并探讨了代理化发布与持续学习等未来方向。

**💡 创新点**

创新点在于提出“信息动力学三大转型”框架，将人工智能视为复制与共享人类know‑how的关键力量，并引入“代理化科研”和“代理化出版”概念。

**🔧 技术方法**

核心技术为大型语言模型（如GPT‑5.4）与与研究工具的接口，强调模型对科研流程的嵌入与自动化。

**📊 数据集**

本文未使用具体数据集，而是基于对现有科研工作流程的观察与案例推断，主要引用公开论文与实践示例。

**📈 对比分析**

由于是理论性综述，未进行实验对比；文章通过对比传统知识传递与LLM复制人类know‑how的差异，阐明潜在效率提升。

**⚠️ 局限性**

主要局限在于缺乏实时学习机制、跨领域多样性不足，以及缺少针对科研协作的评估与验证框架。

---

## 255. MS-SSE-Net: A Multi-Scale Spatial Squeeze-and-Excitation Network for Structural Damage Detection in Civil and Geotechnical Engineering

**arXiv ID:** 2604.14711 | [PDF](https://arxiv.org/pdf/2604.14711v1)

**作者:** Saif ur Rehman Khan `[一作]` (Rhineland-Palatinate Technical University of Kaiserslautern-Landau), Muhammad Nabeel Asim `[通讯]` (Rhineland-Palatinate Technical University of Kaiserslautern-Landau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MS-SSE-Net多尺度空间Squeeze-Excitation网络，用于结构损伤分类。

**💡 创新点**

创新点在于结合多尺度深度卷积与通道+空间注意力的MS‑SSE模块，显著提升特征表达与分类性能。

**🔧 技术方法**

采用DenseNet201骨干，深度可分离卷积、多尺度特征融合、SE通道注意力、空间注意力、全局平均池化、全连接分类层，训练使用Adam优化器和数据增强。

**📊 数据集**

使用StructDamage大规模结构裂缝图像数据集（9类）。

**📈 对比分析**

与DenseNet201、16个ImageNet预训练模型以及7种常见注意力/结构块对比，MS‑SSE-Net在StructDamage上实现99.31%准确率、99.27%召回、99.26%F1，显著优于基线。

**⚠️ 局限性**

局限性：仅基于单模态RGB图像，缺乏视频、多模态输入；对稀有裂缝形态鲁棒性不足，未在实时边缘设备上验证。

---

## 256. Gating Enables Curvature: A Geometric Expressivity Gap in Attention

**arXiv ID:** 2604.14702 | [PDF](https://arxiv.org/pdf/2604.14702v1)

**作者:** Satwik Bathula `[一作]` (University of Southern California), Anand A. Joshi `[通讯]` (University of Southern California)

**通讯引用:** 3163 | [OpenAlex ID](https://openalex.org/A5048206294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了注意力机制在统计几何框架下的表示几何性，将注意力输出视为高斯分布的均值，分析其Fisher–Rao几何；

**💡 创新点**

证明了多重门控（multiplicative gating）能够突破无门控注意力的仿射限制，生成正曲率的非平坦统计流形，并揭示曲率可在多层中累积放大；

**🔧 技术方法**

采用信息几何（Fisher–Rao度量）、Riemann曲率计算、仿射与乘法门控的解析构造、有限差分曲率代理以及对比实验来验证理论；

**📊 数据集**

使用人工合成的二维曲线分类数据集（带非线性决策边界）以及线性控制任务；

**📈 对比分析**

通过与无门控、SiLU非线性、门控等变体对比，测量曲率代理与测试准确率，发现曲率与性能呈正相关，门控模型在需要非线性表示的任务上显著提升准确率，而在线性任务上无显著优势；

**⚠️ 局限性**

理论基于固定协方差的高斯位置族，曲率测度为非完整的局部代理，实验仅在合成数据上验证，缺乏对真实大规模任务的全面评估。

---

## 257. CAMO: An Agentic Framework for Automated Causal Discovery from Micro Behaviors to Macro Emergence in LLM Agent Simulations

**arXiv ID:** 2604.14691 | [PDF](https://arxiv.org/pdf/2604.14691v1)

**作者:** Xiangning Yu `[一作]` (Tianjin University), Qun Ma `[通讯]` (Tianjin University)

**通讯引用:** 4526 | [OpenAlex ID](https://openalex.org/A5072052712)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 CAMO 框架，利用大语言模型在 LLM 驱动的多智能体仿真中自动发现从微观行为到宏观出现的因果机制，并给出可操作的因果子图；

**💡 创新点**

创新点在于把因果识别与机制解释分离，构建可计算因子空间以获得可量化的 Markov 边界；引入 fast–slow 循环结合仿真内反事实来修正先验，生成最小连接子图，从而在宏观现象解释与干预上实现可解释与可操作；

**🔧 技术方法**

使用多代理 LLM（Worldview Parser、Integrator、Cartographer、Scriptwright、Counterfactual Adjudicator）、信息增益筛选、约束式因果发现（CPDAG/PAG）、Simulator 内反事实实验、Markov 边界计算、最小连接子图构建；

**📊 数据集**

使用 LLM 代理仿真生成的数据集，包括在线到线下（O2O）交付平台、Smallville、AgentSociety 等实验环境；

**📈 对比分析**

与统计因果发现（PC、FCI、GES、MMHC）、纯 LLM 方法（Efficient‑CDLMs、MAC、PAIRWISE）以及混合 SCD+LLM 方法（SCD‑LLM、ReAct、LLM‑KBCI）对比；CAMO 在 Markov 边界 F1、祖先 F1、结构指标 SHD、FPR、以及干预排名 Precision@5、MAP@5 等指标上均优于所有基线；

**⚠️ 局限性**

主要限制包括：依赖 LLM 先验可能产生幻觉；反事实校正仅局部且无法保证完整性；局部因果界面可能漏掉未记录的混杂变量；结果与仿真环境紧密耦合，外部迁移受限；多轮仿真查询带来预算与精度权衡。

---

## 258. Beyond Nodes vs. Edges: A Multi-View Fusion Framework for Provenance-Based Intrusion Detection

**arXiv ID:** 2604.14685 | [PDF](https://arxiv.org/pdf/2604.14685v1)

**作者:** Fan Yang `[一作]` (Chinese University of Hong Kong), Kehuan Zhang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 3626 | [OpenAlex ID](https://openalex.org/A5008237643)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多视图融合框架，将节点属性、结构模式与因果交互三种视角的异常信号统一处理，利用投票式融合实现命名链入侵检测；

**💡 创新点**

创新点在于将节点级与边级分析拆分为三视图并通过多维线性融合+投票决策统一决策，同时采用分位数归一化和属性/结构分离、边权重自适应等技术提升鲁棒性；

**🔧 技术方法**

采用图神经网络（GMAE）、Word2Vec嵌入、KNN密度检测、Faiss加速KNN、边多热编码等技术；

**📊 数据集**

使用DARPA Transparent Computing（E3、E5等）和DARPA Operationally Transparent Cyber（OpTC）共9个公开基准数据集；

**📈 对比分析**

与MAGIC、NodLink、FLASH、Orthrus、Velox、Kairos等6个SOTA PIDS对比，实验显示在TP/FP、MCC、ADP等指标上均实现最高TP且FP最低，覆盖率接近100%，显著优于基准；

**⚠️ 局限性**

局限性包括：使用静态图忽略时间序列信息；仅处理单主机日志，缺乏跨主机扩展；假设完整无缺失的日志捕获；对极低频攻击仍可能产生误判，需人工阈值调优。

---

## 259. SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models

**arXiv ID:** 2604.14672 | [PDF](https://arxiv.org/pdf/2604.14672v1)

**作者:** Binxian Su `[一作]` (Beijing Language and Culture University), Pengyuan Liu `[通讯]` (Beijing Language and Culture University)

**通讯引用:** 1028 | [OpenAlex ID](https://openalex.org/A5100714941)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 SPAGBias 框架，对大型语言模型在城市微空间中的性别偏见进行系统评估。

**💡 创新点**

创新点在于首次构建包含 62 个城市微空间的税onomy、三层诊断（显式、概率、构造）并结合量化与叙事分析的方法。

**🔧 技术方法**

采用了多种 LLM（GPT‑3.5‑turbo、GPT‑4、Llama3‑8B‑instruct、Qwen2‑7B‑instruct、Phi‑3‑mini‑4k‑instruct、Deepseek‑llm‑7b‑chat）以及 Prompt 设计、log‑prob 统计、语义角色标注、情感与角色归属分析等技术。

**📊 数据集**

数据集主要包括手工构建的 62 空间词汇表、对应的 Prompt 组合、预训练语料（C4、WIMBD）以及 LLM 生成的故事文本。

**📈 对比分析**

通过显式偏差指数（EDI）、对数概率差异、叙事角色分布等指标对六个模型进行对比，结果显示所有模型都存在显著的空间性别偏差，且偏差程度普遍高于现实分布。

**⚠️ 局限性**

局限性包括仅针对英语文本、采用二元性别（男/女）框架、未覆盖郊区/农村空间以及未能量化不同文化背景下的跨语言比较。

---

## 260. Beyond Chat and Clicks: GUI Agents for In-Situ Assistance via Live Interface Transformation

**arXiv ID:** 2604.14668 | [PDF](https://arxiv.org/pdf/2604.14668v1)

**作者:** Pan Hao `[一作]` (University of Minnesota), Qianwen Wang `[通讯]` (University of Minnesota)

**通讯引用:** 1383 | [OpenAlex ID](https://openalex.org/A5100609686)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Chrome扩展的 DOM 现场即时辅助系统 DOMSteer，通过在网页中插入、变更或重新组合 DOM 元素来为用户提供上下文工具提示、控件高亮、布局重组等帮助。

**💡 创新点**

创新点包括：① 将即时辅助划分为 Insert、Mutate、Recompose 三大操作模式，提供系统化的设计空间；② 利用预构建的手册式辅助案例库实现低延迟检索，避免实时 LLM 推理；③ 在不改动后端逻辑的前提下，直接在浏览器层面对任何基于 DOM 的网页进行可逆、轻量级的界面变形；④ 结合语义嵌入实现无缝的目标定位与交互。

**🔧 技术方法**

技术栈包括：Chrome Manifest V3 扩展、React 18 + Tailwind + Shadow DOM、LangGraph 统一 LLM 接口、OpenAI/Anthropic/ Gemini 等多种 LLM、Tavily web search、FAISS 索引做案例检索、Nanobrowser 框架、DOM 结构与视觉嵌入生成、自动化脚本注入与回滚标签、交互式注释与反馈循环。

**📊 数据集**

数据集与评测：① 在 DataVoyager 2 与 TensorFlow Playground 上采集 101 条真实用户挑战，构成评测基准；② 形成 120 条手册式辅助案例；③ 在 TxGNN Explorer 与 Embedding Atlas 上演示四种典型场景；④ 任务使用公开数据集（cars、wine reviews 等）。

**📈 对比分析**

比较方法：① 与直接 LLM 生成的聊天式辅助（Baseline）对比；② 与仅检索手册的“Handbook Retrieval”对比；③ 与检索加回退（Handbook + FB）对比；④ 在用户研究中与 ChatGPT Atlas 的聊天式助手和自治代理进行对比。性能结果：手册检索平均比 LLM 快 11.8×，成功率与问题解决度几乎相同；在实际任务中，DOMSteer 的任务完成时间比聊天式助手快 25%（p = 0.0161），准确率达到 100%，而自治代理虽最快但准确率仅 79.2%。

**⚠️ 局限性**

局限性：① 仅适用于可被 DOM 解析的网页，Canvas、极深层嵌套或类名混淆的页面难以定位或修改；② 需要 3–5 分钟一次性构建手册，适合高频使用的界面；③ 目前仅支持显式提问式交互，缺乏主动触发与用户行为识别；④ 可能出现信任校准问题，用户可能接受不经验证的解释。

---

## 261. EdgeDetect: Importance-Aware Gradient Compression with Homomorphic Aggregation for Federated Intrusion Detection

**arXiv ID:** 2604.14663 | [PDF](https://arxiv.org/pdf/2604.14663v1)

**作者:** Noor Islam S. Mohammad `[一作]` `[通讯]` (Istanbul Technical University), Noor Islam S. Mohammad (Istanbul Technical University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了EdgeDetect，一个面向6G‑IoT边缘设备的安全联邦入侵检测框架，能够在不共享原始流量的前提下进行模型训练与推理。

**💡 创新点**

创新点在于：① 自适应中值阈值梯度二值化（gradient smartification）实现32×压缩；② 将Paillier同态加密直接应用于二值化梯度，保障梯度隐私；③ 在联邦学习管道中加入PCA降维与SMOTE重采样，解决高维度和严重类别不平衡问题。

**🔧 技术方法**

使用技术包括：联邦学习（FedAvg、FedProx）、自适应梯度二值化、Paillier同态加密、PCA降维、SMOTE/重采样、随机森林与其他经典分类器、Ablation实验与统计显著性检验。

**📊 数据集**

采用CIC‑IDS2017网络流数据（约2.8M条流、7类攻击），并在Raspberry Pi 4设备上进行边缘部署实验。

**📈 对比分析**

与FedAvg、signSGD等基线比较，EdgeDetect在保持98.0%多类准确率、97.9%宏F1的同时，将每轮通信从450 MB降至14 MB（96.9%压缩）。在5%投毒、极端不平衡或高异构性场景下，准确率仍保持≥85%，并实现了优秀的隐私防护（PSNR降至15.1 dB）。

**⚠️ 局限性**

局限性包括：对非凸模型收敛理论不足、对概念漂移和白盒鲁棒性的实证支持有限、在极端低带宽或极度异构环境下仍可能出现性能下降。

---

## 262. DigiForest: Digital Analytics and Robotics for Sustainable Forestry

**arXiv ID:** 2604.14652 | [PDF](https://arxiv.org/pdf/2604.14652v1)

**作者:** Marco Camurri `[一作]` (University of Trento), Stefan Leutenegger `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发并验证了一套名为 DigiForest 的精准林业系统，集成了异构移动机器人进行树木级数据采集、在线树木特征提取与森林清单、决策支持系统以及低冲击自适应采伐机，实现了从数据获取到决策制定再到自动采伐的全流程闭环。

**💡 创新点**

创新点包括：①利用腿行、航空及双足机器人构成的异构团队实现地面与冠层的多尺度自主扫描；②在线实时森林清单与离线高精度全景分割相结合，首次实现即时、精确的 DBH、树冠、灌木等多维度树木属性提取；③基于自研的低影响采伐机 SAHA，融合导航、抓取与切割，实现单机完成选择性采伐；④将机器人生成的高质量清单输入决策支持系统，结合气候情景模拟进行可持续管理方案评估。

**🔧 技术方法**

核心技术涵盖：LiDAR‑IMU 因子图 SLAM、VIO、子地图对齐、神经控制边界函数 + NMPC 的自适应避障；Minkowski Engine 及自监督 MLP 头的 LiDAR 泛化分割；DBSCAN + Hough 变换的在线树木检测与 DBH 估计；ForClimV4.1 气候敏感生长模拟器；多任务 SLAM 与聚合点云处理；以及基于仿真与实测的行进、抓取与切割控制框架。

**📊 数据集**

主要使用了 DigiForests 长期序列 LiDAR 数据集（春、秋、夏三季 3D 点云及细粒度注释），并在对比实验中引用 MaskPLS（基于 Transformer 的 LiDAR 分割）进行性能评估；此外还利用公开的森林测量数据作 DBH 验证。

**📈 对比分析**

与 MaskPLS 的泛化分割比较，DigiForest 在全景质量上提升至 70%（vs 58%）；在线 DBH 估计 RMSE 为 3.15 cm，低于基线 5.32 cm；无人机闭环误差由 0.37 m 降至 0.19 m，显示子地图约束显著提升定位精度；腿行机器人在 0.2 ha 区域完成约 40 棵树的在线清单，平均行进距离间隔 102.5 m，表明高效覆盖与精确定位；SAHA 在实地测试中实现一次性切割与搬运，减少土壤压实与作业步骤。

**⚠️ 局限性**

局限性包括：①在高度密集灌木与低矮枝条中视觉与 LiDAR 传感器仍可能遮挡，导致树干检测不完整；②在线分割与 DBH 估计依赖连续扫描，对短时间窗口扫描的鲁棒性不足；③深度学习分割模型训练数据量有限，跨域推广仍需验证；④SAHA 的路径规划与碰撞回避仍主要基于预生成地图，面对动态障碍或极端地形时表现待提升；⑤系统整体集成度仍依赖人工触发与监控，真正实现全自动闭环仍需进一步工作。

---

## 263. Zeroth-Order Optimization at the Edge of Stability

**arXiv ID:** 2604.14669 | [PDF](https://arxiv.org/pdf/2604.14669v1)

**作者:** Minhak Song `[一作]` (KAIST), Sewoong Oh `[通讯]` (University of Washington)

**通讯引用:** 7286 | [OpenAlex ID](https://openalex.org/A5028243041)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了仅利用函数评估的零阶优化（ZO）方法，提出了基于均方线性稳定性的理论框架，并在全批训练下实证验证了ZO方法在深度网络训练中确实以均方稳定边界为动力学准则运行；

**💡 创新点**

创新点在于：①给出ZO-GD、ZO-GDM、ZO-Adam等常用ZO方法的精确均方稳定性阈值，并证明其依赖于Hessian全谱，尤其是迹项；②将此阈值转化为可在训练中跟踪的上下界，从而揭示ZO方法在训练过程中的“均方边界”行为；③通过实验展示不同网络（CNN、ResNet、ViT）以及序列模型在ZO训练时均符合这一边界，揭示了ZO方法对Hessian迹的隐式正则化作用。

**🔧 技术方法**

主要技术包括：两点随机方向梯度估计、线性化动态分析、Krein–Rutman理论下的谱半径计算、Isserlis定理用于高阶矩推导，以及Hutchinson估计和幂迭代用于在线追踪Hessian迹和最大特征值。

**📊 数据集**

实验数据集主要为CIFAR‑10（CNN、ResNet20、ViT），以及人工排序任务的LSTM和Mamba序列模型；使用全批训练和固定步长。

**📈 对比分析**

通过比较ZO方法与一阶梯度下降在稳定性阈值上的差异，作者发现ZO方法的均方阈值与Hessian迹密切相关，而一阶方法仅受最大特征值控制；在实验中，ZO方法在训练过程中始终保持在其均方稳定边界附近，表明理论预测与实践高度一致。

**⚠️ 局限性**

局限性包括：①分析仅针对全批ZO，未覆盖常见的 mini‑batch 或随机子采样场景；②依赖于两点估计与高斯方向，其他估计器的适用性尚未验证；③均方稳定性理论依赖对Hessian谱的估计，虽然可以通过迹/最大特征值近似，但在极大模型中仍需高计算开销；④未探讨稳定性与最终泛化/收敛速度之间的具体权衡。

---

## 264. Data Synthesis Improves 3D Myotube Instance Segmentation

**arXiv ID:** 2604.14720 | [PDF](https://arxiv.org/pdf/2604.14720v1)

**作者:** David Exler `[一作]` (Karlsruhe Institute of Technology), Markus Reischl `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 6406 | [OpenAlex ID](https://openalex.org/A5049462585)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

构建了基于几何模型的3D肌管合成管线，并仅用合成数据训练了一个自监督预训练的U-Net，实现了对真实3D肌管的实例分割。

**💡 创新点**

首次提出针对缺乏标注数据的3D肌管领域的几何驱动合成方法，并证明合成+自监督+域适配的组合能显著提升零样本模型的性能。

**🔧 技术方法**

几何中心线建模、Polynomial & Chebyshev系数、周期性厚度调制、分支插入、椭球端帽、Poisson+Gaussian噪声仿真、CycleGAN域适配、FCMAE自监督预训练、三维U-Net残差网络、前景与中心线双通道输出、阈值+Watershed实例分割。

**📊 数据集**

使用30幅128×1024×1024的合成体积（200条实例）做训练，使用17幅真实荧光显微镜体积（共40条标注实例）做独立测试。

**📈 对比分析**

与三种零样本生物医学分割基线（CellposeSAM、PlantSeg、StarDist）进行比较，采用Injective Panoptic Quality (IPQ) 指标。自监督+域适配模型在真实数据上平均IPQ为0.22，显著优于基线（p<0.001），且参数量更少。

**⚠️ 局限性**

仍存在合成与真实图像之间的域差距导致性能不至于完美，尤其在阴影、噪声与真实光学特性匹配上需进一步改进；缺乏针对疾病变异的专门建模。

---

## 265. On the Use of Iterative Problem Solving for the Traveling Salesperson Problem with Changing Time Window Constraints

**arXiv ID:** 2604.14745 | [PDF](https://arxiv.org/pdf/2604.14745v1)

**作者:** Hy Nguyen `[一作]` (Adelaide University), Frank Neumann `[通讯]` (Adelaide University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在时间窗变化下，利用先前解答的旅行推销员问题（TSPTW）任务序列的知识迁移能否提升后续任务的求解性能；

**💡 创新点**

首次系统比较从零开始求解与顺序迁移两种协议，揭示任务相似度对迁移效益的决定性影响；

**🔧 技术方法**

采用三种主流局部搜索算法（Large Neighborhood Search、Variable Neighborhood Search、Lin-Kernighan-Helsgaun 3），并实现其迭代版本；

**📊 数据集**

使用标准TSPTW基准实例（n∈{20,40,60,80,100,150,200}），构建两类五任务序列：逐步时间窗扩展与交换加权重重构；

**📈 对比分析**

通过Mann-Whitney U检验和可行率统计比较两种协议。结果显示：在逐步扩展环境中，迭代协议在中大规模实例上显著提升可行率与惩罚分数；在交换加权环境下，提升更为有限但仍保持竞争力；

**⚠️ 局限性**

局限性在于实验仅限于局部搜索算法和固定时间窗变换模式，未探讨更复杂的迁移策略或更广泛的任务生成机制；

---

## 266. Find the Differences: Differential Morphing Attack Detection vs Face Recognition

**arXiv ID:** 2604.14734 | [PDF](https://arxiv.org/pdf/2604.14734v1)

**作者:** Una M. Kelly `[一作]` (University of Twente), Raymond N. J. Veldhuis `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文比较了差分攻击检测（D‑MAD）与人脸识别（FR）的性能，证明两者任务相似，并提出利用现有FR系统实现攻击检测，随后设计基于最坏情况（worst‑case）的阈值方法以限制攻击漏洞。

**💡 创新点**

创新点包括①将FR系统视作D‑MAD工具；②用von Mises‑Fisher模型模拟潜在攻击分布，阐释阈值选择导致的性能-攻击易感性权衡；③提出worst‑case MMPMR阈值，以给出对未知攻击的上限保障。

**🔧 技术方法**

采用的技术包括ArcFace深度FR模型、支持向量机（SVM）分类、demorphing逆向攻击、tSNE可视化、von Mises‑Fisher分布仿真以及多种合成攻击生成方法。

**📊 数据集**

使用的数据集涵盖FRGC、FRLL、SynFace合成身份，及其对应的landmark、MIPGAN、改进MIPGAN和Diffusion等多种攻击样本。

**📈 对比分析**

通过在相同阈值下对mated、non‑mated和morph三类样本进行比较，发现更严格的FR阈值可显著降低攻击成功率；在极端攻击下，D‑MAD与FR的性能相当，且D‑MAD在极端攻击时表现更稳健。

**⚠️ 局限性**

局限性在于最坏情况阈值仅对基于FR嵌入的D‑MAD有效，其他D‑MAD方法无法提供同样的安全上限；此外，对未知攻击的泛化仍受限，且阈值设计依赖于对相似度分布的假设。

---

## 267. RELOAD: A Robust and Efficient Learned Query Optimizer for Database Systems

**arXiv ID:** 2604.14725 | [PDF](https://arxiv.org/pdf/2604.14725v1)

**作者:** Seokwon Lee `[一作]` (Yonsei University), Kwanghyun Park `[通讯]` (Yonsei University)

**通讯引用:** 1172 | [OpenAlex ID](https://openalex.org/A5010089204)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种名为 RELOAD 的学习型查询优化器，旨在解决基于强化学习的查询优化器在单查询级别上出现的性能回归和训练效率低下的问题。

**💡 创新点**

创新点主要体现在：①引入经验保留机制（基于优先经验回放 PER），通过结合最近经验和 TD 误差的加权策略，有效缓解子规划稀疏奖励导致的信用分配问题和局部最优陷阱；②引入知识迁移机制（MAML 与基于 Halstead 复杂度的任务划分），通过跨任务的元学习实现快速收敛；③在评价维度上，明确提出查询级别的鲁棒性指标（Plateau 与 Rebound）和效率指标（收敛迭代数/时长），并在 PostgreSQL 与商业 DBMS 上进行统一对比。

**🔧 技术方法**

技术手段包括：深度强化学习（以 Balsa 为基础框架）、优先经验回放（PER）、元学习算法 MAML、任务聚类与复杂度度量（Halstead、运算符计数、估算成本/行数）以及基于 DBI 的聚类质量评估。

**📊 数据集**

实验使用的主数据集包括：Join Order Benchmark (JOB)、TPC‑DS（采用 4×规模）和 Star Schema Benchmark (SSB)，在三种数据集上分别进行训练与测试，保持训练/测试模板完全分离。

**📈 对比分析**

与 Bao、LOGER、Balsa（vanilla）和 LIMAO 的对比表明：RELOAD 在 PostgreSQL 上能将 Plateau + Rebound 的总数降低 44%（JOB）/ 60%（TPC‑DS）/ 33%（SSB）；收敛迭代数/时长比 Balsa 快 1.1–2.4 倍；WRL 指标在 JOB、TPC‑DS、SSB 上分别提升至 0.64/0.85/0.88，较基线提升 1.5–1.2 倍；在商业 DBMS 上也保持类似优势。

**⚠️ 局限性**

局限性包括：仍需数小时训练即可达到专家级性能；PER 机制依赖手工设计的权重策略，可能对不同查询负载敏感；MAML 任务划分在极大多样化的工作负载上效果尚未充分验证；目前主要针对连接顺序优化，未涵盖更广泛的物理规划空间。

---

## 268. Layered Mutability: Continuity and Governance in Persistent Self-Modifying Agents

**arXiv ID:** 2604.14717 | [PDF](https://arxiv.org/pdf/2604.14717v1)

**作者:** Krti Tallam `[一作]` `[通讯]` (KamiwazaAI), Krti Tallam (KamiwazaAI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“分层可变性”（Layered Mutability）框架，分析持久化语言模型代理在不同可变层（预训练、后训练对齐、自述、记忆、权重适配）中的治理挑战，并通过小规模行为实验验证可见身份恢复后行为漂移仍然存在的“齿轮效应”

**💡 创新点**

创新点在于：①把多层可变性与治理难度关联的量化指标（mutation rate、observability、reversibility、downstream coupling）统一为治理负荷；②定义并量化“齿轮效应”与“滞后比率”；③强调跨层持续性评估而非仅表面行为评估；④首次在实验中测量可见自述恢复后残余行为漂移（滞后比率≈0.68）

**🔧 技术方法**

使用了大型语言模型（如 GPT‑4 等生成模型）与判别模型（Judge model）进行文本生成与行为评估；构建了包含自述编辑、记忆积累与回滚的代理框架；设计了基于自述、记忆、权重层的可变性参数并计算治理负荷；利用统计量比较四种实验条件（基线、编辑、编辑+记忆、回滚）

**📊 数据集**

实验数据集：四个自述状态下的四个记忆条件，结合五个高风险模糊任务（安全补丁部署、可疑支付处理、泄露沟通、供应商合同批准、热修复部署），共计 20 条评估实例；未使用公开基准，采用自构造的任务集合

**📈 对比分析**

比较方法：对每个实验条件分别评估“行动偏好、完整性、不确定性、特质强度、与自述的一致性”五个维度的得分；通过对比得分差异和滞后比率来判断齿轮效应；实验显示回滚后特质强度仍高（5.4 vs 2.0），滞后比率为0.68，表明可见身份恢复并未恢复基线行为

**⚠️ 局限性**

局限性：①实验规模小，仅使用单一模型族和手工构造的任务集；②未涉及权重级自我训练，仅为文本+记忆漂移；③缺乏长期多周期评估，难以推断在更复杂系统中的泛化性；④治理负荷公式为启发式，缺乏严格理论验证

---

## 269. HWE-Bench: Benchmarking LLM Agents on Real-World Hardware Bug Repair Tasks

**arXiv ID:** 2604.14709 | [PDF](https://arxiv.org/pdf/2604.14709v1)

**作者:** Fan Cui `[一作]` (Peking University), Yun Liang `[通讯]` (Peking University)

**通讯引用:** 10030 | [OpenAlex ID](https://openalex.org/A5100604860)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了HWE-Bench，一个针对大规模硬件仓库级别的LLM评测基准，用真实历史Bug修复PR构建任务并在容器化、执行驱动的环境中验证补丁。

**💡 创新点**

创新点包括①全仓库级别、执行驱动的硬件Bug修复基准；②自动化构建管线，支持多种HDL（Verilog/SV、Chisel）和多项目；③对LLM代理进行系统性失败模式分析，揭示硬件领域独特的故障定位、硬件语义推理和跨工件协同问题。

**🔧 技术方法**

技术主要涵盖：LLM代理与工具调用框架（OpenHands、Claude Code、Codex CLI、Kimi CLI）、Docker化的E2E验证流水线、LLM驱动的测试脚本与环境准备生成、自动化PR过滤与可验证性评估。

**📊 数据集**

数据集为来自六个开源硬件项目（OpenTitan、Caliptra、XiangShan、Ibex、CVA6、Rocket Chip）的约30,000条PR，筛选后得到417个可验证的Bug修复实例。

**📈 对比分析**

与软件基准SWE-bench相比，评测使用同一模型在不同基准上的解析率对比。GPT‑5.4 xhigh最高达70.7%；专有模型约68%，开源模型GLM‑5.1为63%；相较SWE-bench，硬件基准模型表现差距更大，专有与开源模型差距从8%扩展到23%。

**⚠️ 局限性**

局限性包括：①仍依赖人工审核部分验证；②仅覆盖六个项目，难以覆盖更广泛的硬件生态；③基准侧重于修复已通过PR的Bug，忽略新Bug的发现和设计生成任务；④LLM对跨工件协同和硬件时序语义的理解不足，导致失败率高。

---

## 270. Geo2Sound: A Scalable Geo-Aligned Framework for Soundscape Generation from Satellite Imagery

**arXiv ID:** 2604.14707 | [PDF](https://arxiv.org/pdf/2604.14707v1)

**作者:** Kunlin Wu `[一作]` (Hong Kong University of Science and Technology), Xiaofeng Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 8814 | [OpenAlex ID](https://openalex.org/A5108555426)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Geo2Sound框架，从卫星图像生成地理可行的声音景观，并创建SatSound-Bench基准

**💡 创新点**

融合结构化地理属性建模、语义假设扩展和地理-声学对齐，解决遥感图像声学模糊和地理一致性问题

**🔧 技术方法**

使用视觉变换器提取地理属性，随机森林分类，文本到音频生成器（Make‑An‑Audio 2），以及轻量级MLP对齐网络

**📊 数据集**

SatSound‑Bench（约28.6k对）包括真实现场录音、公开数据集（SoundingEarth、iNaturalist Sounds、Freesound）和对应卫星图像

**📈 对比分析**

与多种图像‑音频和多模态‑音频基线对比，FAD降至1.765、CLAP提升至0.449，MOS‑A/S/E分别达3.58/3.41/3.66，显著优于最强基线

**⚠️ 局限性**

仍受限于卫星图像分辨率不足导致细节不足、生成的声音多样性与真实环境的空间精细匹配尚不完美

---

## 271. NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation

**arXiv ID:** 2604.14706 | [PDF](https://arxiv.org/pdf/2604.14706v1)

**作者:** Yi He `[一作]` (Beijing Jiaotong University), Haibin Ling `[通讯]` (Westlake University)

**通讯引用:** 36540 | [OpenAlex ID](https://openalex.org/A5061469520)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了NG-GS框架，针对3D Gaussian Splatting中的对象边界离散化问题实现高质量分割；

**💡 创新点**

创新点在于利用掩码方差检测边界高斯，采用RBF插值和多分辨率哈希编码构建连续特征场，并与NeRF联合优化对齐与连续性损失；

**🔧 技术方法**

使用RBF插值、MRHE哈希编码、轻量级NeRF模块、边界对齐损失、连续性损失、梯度平滑损失及掩码损失；

**📊 数据集**

在NVOS、LERF-OVS和ScanNet三个基准集上进行实验；

**📈 对比分析**

与现有mask‑based和feedforward方法对比，NG-GS在所有指标上均取得SOTA，尤其是边界mIoU在NVOS提升至84.7%（+5.6%），LERF‑OVS提升至72.8%（+4.4%），ScanNet提升至59.6%（+6.8%）；

**⚠️ 局限性**

目前仅针对静态场景，缺乏动态场景和实时交互的适配，未来需要扩展到实时应用和动态场景。

---

## 272. Mean Flow Policy Optimization

**arXiv ID:** 2604.14698 | [PDF](https://arxiv.org/pdf/2604.14698v1)

**作者:** Xiaoyi Dong `[一作]` (Chinese Academy of Sciences), Jian Cheng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 120821 | [OpenAlex ID](https://openalex.org/A5101425421)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MFPO 方法，在最大熵强化学习框架下采用 MeanFlow 模型作为策略表示，并实现了高效的训练与推理。

**💡 创新点**

创新点在于：①使用平均速度场的 MeanFlow 生成模型替代传统扩散策略；②引入平均散度网络近似动作似然，避免昂贵的雅可比计算；③结合自适应重要性采样估计即时速度，解决软策略迭代中的采样难题。

**🔧 技术方法**

技术手段包括：MeanFlow 生成模型、最大熵 RL、软策略迭代、平均散度网络、Skilling–Hutchinson 跟踪估计、两种分布自适应重要性采样、分布式 Q 学习、分布式批评器、自动温度调节以及行动选择优化。

**📊 数据集**

实验使用了 MuJoCo 运动控制任务（Walker2D、Hopper、HalfCheetah、Ant、Humanoid）和 DeepMind Control Suite 的难度任务。

**📈 对比分析**

与 5 个基于扩散的算法（DIME、FlowRL、MaxEntDP、DACER、QVPO）及经典基线（SAC、TD3）进行对比，MFPO 在大多数任务上匹配或超越对手，同时训练时间降低约 50%，采样步数仅 2 步，推理延迟显著下降。

**⚠️ 局限性**

局限性是：仍需至少两步采样，尚未实现单步高性能；对极端高维或更复杂环境的鲁棒性仍需进一步验证。

---

## 273. Online Algorithms for Geometric Independent Set

**arXiv ID:** 2604.14677 | [PDF](https://arxiv.org/pdf/2604.14677v1)

**作者:** Minati De `[一作]` (Indian Institute of Technology Delhi), Satyam Singh `[通讯]` (Aalto University)

**通讯引用:** 622 | [OpenAlex ID](https://openalex.org/A5040660392)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在线最大独立集（MIS）问题，证明了贪心算法在独立接触数（independent kissing number）为ζ的图上可达到ζ的竞争比，并给出了随机化算法在几何对象（如三维球、α-fat对象和轴对齐超矩形）上的竞争比提升。

**💡 创新点**

创新点在于：①用独立接触数统一描述不同几何图的在线MIS性能；②证明贪心算法在该参数下是最优的；③通过随机化并利用几何表示，突破了确定性下的下界；④针对不同对象给出了基于格子/分层的随机化算法，竞争比仅取决于对象尺寸比和维数。

**🔧 技术方法**

核心技术包括：独立接触数分析、贪心算法竞争比证明、Yao最小化原理构造随机对抗实例、几何格子（特别是三维稀疏格子）与体积保持性质、对α-fat对象与超矩形的尺寸分层与独立接触数上界。

**📊 数据集**

本文没有使用实验数据集，全部为理论分析与证明。

**📈 对比分析**

与现有结果比较：对单位球在三维时，随机化竞争比约11.46小于确定性下限12；对α-fat对象与超矩形，竞争比为O(ζ' log M)和O((4(log M+1))^d)，显著低于已知的确定性下界，表明随机化+几何结构可实现更好的性能。

**⚠️ 局限性**

局限性包括：①在维数≥4时，随机化格子方法无法超过确定性下界；②随机化单独使用（无几何信息）无法突破确定性下界；③竞争比仍随对象尺寸比M和维数d呈指数或多项式增长，实际应用可能受限。

---

## 274. Rethinking Patient Education as Multi-turn Multi-modal Interaction

**arXiv ID:** 2604.14656 | [PDF](https://arxiv.org/pdf/2604.14656v1)

**作者:** Zonghai Yao `[一作]` (VA Bedford Health Care), Hong Yu `[通讯]` (VA Bedford Health Care)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MedImageEdu基准，用于评估多轮、基于证据的影像导向医学患者教育

**💡 创新点**

创新点在于将视觉工具调用嵌入交互式咨询流程，区分咨询质量与多模态证据绑定，并引入隐藏患者画像特征来测试个性化

**🔧 技术方法**

采用大型视觉语言模型（InternVL、MedGemma、Qwen3-VL、GPT-5系列）作为DoctorAgent，并使用GPT Image 1.5绘图工具

**📊 数据集**

数据集包含150例放射学案例，来源于MedThinkVQA（100例）、Indiana University Chest X-ray（25例）和MIMIC‑CXR（25例）

**📈 对比分析**

与13种模型进行统一评估，分为多轮模拟和多模态交互两块，评估维度包括咨询、范围安全、语言质量、绘图质量和图文响应质量；GPT‑5.1最高分，显示模型规模与平衡性关键；但多数模型在绘图质量和安全性上存在明显差距

**⚠️ 局限性**

局限性包括案例数量有限、仅覆盖放射学与英语、使用共享绘图工具、依赖LLM评判且缺乏真实患者验证，且未评估长期学习效果或行为改变

---

## 275. AgentGA: Evolving Code Solutions in Agent-Seed Space

**arXiv ID:** 2604.14655 | [PDF](https://arxiv.org/pdf/2604.14655v1)

**作者:** David Y. Y. Tan `[一作]`, Jingxian Zhang `[通讯]` (Diagnostics Development Hub)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究提出了一种通过进化“agent‑seed”来优化自主代码生成的框架，每次启动时从空白工作空间起始，并通过父级归档传递知识，形成双层演化与代理执行的组合。

**💡 创新点**

创新点在于将进化目标从代码片段或提示转向初始种子（任务提示+父级归档），并采用1:1精英锦标赛、在线算子分配及基于任务的遗传算子，构建可跨域的演化自主代理体系。

**🔧 技术方法**

主要技术包括基于LLM的LangGraph ReAct循环、六功能节点（规划、代理、工具、压缩、验证、最终化）、自适应算子分配、可变遗传算子以及与Kimi K2.5等模型集成。

**📊 数据集**

使用了Weco‑Kaggle Lite数据集，涵盖16个典型的Kaggle表格机器学习竞赛。

**📈 对比分析**

与AIDE基准进行对比，所有实验均击败AIDE，在Kaggle私有排行榜上平均超过人类参赛者的比例提升约为X%（具体数值见论文表格）。

**⚠️ 局限性**

主要限制包括高计算成本（长路径需大量提示/完成token和时间）、种子到结果的随机性、需要可自动化评估的有限域以及对新领域需额外工程工作。

---

## 276. PlanB: Efficient Software IPv6 Lookup with Linearized $B^+$-Tree

**arXiv ID:** 2604.14650 | [PDF](https://arxiv.org/pdf/2604.14650v1)

**作者:** Zhihao Zhang `[一作]` (Alibaba Cloud), Yiming Zhang `[通讯]` (SJTU)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种将IPv6最长前缀匹配（LPM）问题转化为一维区间搜索，并在此基础上设计了线性化B+树与SIMD向量化、批处理、无分支逻辑等优化的高性能软件查找框架 PlanB。

**💡 创新点**

创新点在于：① 用“区间化”将二维LPM转为一维区间搜索；② 设计无指针、连续缓存友好的线性化B+树结构；③ 结合AVX‑512向量化、批量查询、无分支和循环展开，显著降低查找深度和分支预测误判；④ 采用批量重建+原子交换更新策略，既保证查找无锁高吞吐，又能高效处理路由表动态变化。

**🔧 技术方法**

主要技术包括：区间化预处理（将前缀转换为闭区间并划分元素区间）、线性化B+树（所有节点在单一数组中按层级排列）、SIMD向量化（AVX‑512/AVX2/NEON 512/256/128位向量比较）、批处理（将多条查询并行执行）、无分支遍历（使用位掩码+popcnt）、循环展开、原子指针交换重建更新、DPDK多线程流水线（RX‑LPM‑TX）等。

**📊 数据集**

使用真实路由表数据集：RIPE 的 RRC00（2019‑2025）与 RouteViews 的多区域采样（rv1‑rv7），以及基于 RRC00‑25 的四个合成路由表（0.25M‑1M 条目），确保覆盖真实与大规模情境。

**📈 对比分析**

与 PopTrie、CP‑Trie、Neurotrie、HBS 四种主流软件方案在同一硬件（Intel Xeon 24 核、AMD Ryzen 9 12 核）和相同编译环境下对比。PlanB 在单核上实现 390‑393 MLPS，12 核时达到 3.4 BLPS；相比最优对手平均提升 1.6‑14 倍，内存占用下降 56‑92%。实验中还展示了线性可扩展性、低更新延迟（对 1M 前缀重建仅 850 ms）以及对随机流量的稳定性能。

**⚠️ 局限性**

主要限制：更新采用批量重建+交换方式，虽然避免了在线锁，但在重建期间需要额外内存并产生短暂的延迟；对极大路由表（>1M 前缀）可能需要更深树导致查询深度增加；目前实现仅针对 IPv6，IPv4 需要额外适配；以及依赖 CPU SIMD 指令集，若硬件不支持 AVX‑512 则性能下降。

---

## 277. Assessing the Performance-Efficiency Trade-off of Foundation Models in Probabilistic Electricity Price Forecasting

**arXiv ID:** 2604.14739 | [PDF](https://arxiv.org/pdf/2604.14739v1)

**作者:** Jan Niklas Lettner `[一作]` (Karlsruhe Institute of Technology), Benjamin Schäfer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 9334 | [OpenAlex ID](https://openalex.org/A5005576823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在德国-卢森堡电力市场上对日间电价进行概率预测，并比较两类模型：基础的深度学习模型（NHITS+QRA、Normalizing Flow）与时间序列基础模型（Moirai、ChronosX）

**💡 创新点**

首次系统性地评估TSFMs在概率预测中的表现，并展示在零/少量数据、不同特征组合下的适配效果；发现传统模型在合理配置下可与TSFMs竞争，甚至超越

**🔧 技术方法**

基于Transformer的Normalizing Flow、NHITS+QRA、Moirai（Encoder‑only Transformer）、ChronosX（基于语言模型的序列化预测）以及多种特征工程与量化回归后处理

**📊 数据集**

使用ENTSO‑E公开的全欧洲日间电价及相关基础设施、气候和能源市场数据，聚焦德国-卢森堡（DE‑LU）投标区

**📈 对比分析**

采用CRPS、Energy Score和PIT等指标，对零样本、单样本、少样本以及完全微调的情形进行比较；TSFMs在多样本/全微调下表现最优，但在最佳配置的NHITS+QRA在零/少量样本场景下几乎相同甚至更好

**⚠️ 局限性**

计算成本高、TSFMs对特征不敏感导致性能波动、特征组合未穷尽、仅评估单一投标区、未覆盖更先进的基础模型及其他评估指标

---

## 278. Bounded Autonomy for Enterprise AI: Typed Action Contracts and Consumer-Side Execution

**arXiv ID:** 2604.14723 | [PDF](https://arxiv.org/pdf/2604.14723v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 279. Personalized and Context-Aware Transformer Models for Predicting Post-Intervention Physiological Responses from Wearable Sensor Data

**arXiv ID:** 2604.14738 | [PDF](https://arxiv.org/pdf/2604.14738v1)

**作者:** Esther Brown `[一作]` (Harvard University), Finale Doshi-Velez `[通讯]` (Harvard University)

**通讯引用:** 11495 | [OpenAlex ID](https://openalex.org/A5038771285)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并实现了一套基于Transformer的个性化干预后生理轨迹预测框架，能够预测HR、HRV（RMSSD）和BBI在干预结束后不同时间窗口内相对于基线的百分比变化及其变化方向。

**💡 创新点**

创新点在于：①将干预端点作为时间锚点，针对多时域（0–15、15–30、30–60、60–120分钟）构建预测；②同时输出百分比变化与正负/中性方向预测，并使用校准与阈值实现“可拒绝”预测；③设计了决策感知评估指标，区分所有有效时刻与模型自愿做出预测时的准确率。

**🔧 技术方法**

主要技术包括Transformer编码器、量化回归头（多时域多分位数预测）、中位数辅助损失与符号损失、离散危险率头、训练集内校准（等距校准）、基于阈值的方向判定、以及聚类热图可视化。

**📊 数据集**

实验数据来自七名大学生在Garmin Vívosmart 5设备上连续佩戴两周（部分延长至四周）的HR、BBI、HRV（由BBI计算得到）、步数、呼吸频率、设备压力和睡眠得分等传感器流，并与用户通过网页接口标记的269条干预与情境事件（如运动、休息、饮食、社交等）配对。

**📈 对比分析**

评估方法采用多时域方向准确率（eligible 和 called‑only）和混淆矩阵，与“always up”/“always down”基线进行对比。结果显示，BBI在短期窗口内方向准确率可达80%以上；RMSSD在前15分钟达到≈60%；HR在短期窗口中表现中等；整体证明模型在生理信号强时表现良好，且在信号弱时会自动拒绝预测。

**⚠️ 局限性**

局限性包括：样本量极小且仅来自大学生群体；干预标签自我标记可能存在时间与语义误差；监测时长有限，缺乏长周期和多环境因素；模型训练在小数据集上易过拟合；未考虑个体基础生理差异（如训练状态、初值效应）等。

---

## 280. Differentiable Object Pose Connectivity Metrics for Regrasp Sequence Optimization

**arXiv ID:** 2604.14733 | [PDF](https://arxiv.org/pdf/2604.14733v1)

**作者:** Liang Qin `[一作]` (University of Osaka), Kensuke Harada `[通讯]` (University of Osaka)

**通讯引用:** 11086 | [OpenAlex ID](https://openalex.org/A5016270703)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出基于能量模型的可微连通性指标，用梯度优化实现多步重抓规划，并通过自适应迭代加深搜索自动确定最少步骤；

**💡 创新点**

创新点在于将抓取可行性映射为能量函数，利用能量加性得到连续的姿态连通性分数，形成可微的连贯序列成本；并结合Langevin动力学与自适应加深算法实现端到端的梯度搜索与最优步骤选择；

**🔧 技术方法**

采用能量模型（EBM）对抓取可行性进行建模，利用梯度优化（Langevin dynamics）对中间姿态进行连续优化；构造可微的姿态序列成本与正则化；实现自适应迭代加深搜索；

**📊 数据集**

在模拟环境（WRS系统）和真实机器人（6-DoF Dobot Nova2）上进行实验，使用20,000个可行抓取样本训练EBM，随后对4个物体（瓶子、兔子、五边形、马克杯）进行200个抓取候选的评估；跨6种末端执行器进行交叉验证；

**📈 对比分析**

与离散搜索基线以及两种简化成本（单调/截断）进行对比。结果表明所提出的成本在验证成功率、梯度信息丰富度、收敛速度及对未知抓取的泛化性能上均优于对比方法；在多步搜索中，迭代加深显著提升成功率，尤其对形状复杂的物体；

**⚠️ 局限性**

局限性包括对预采样抓取集合的依赖、仅考虑平面扰动的中间姿态、能量阈值调参要求、跨末端执行器的迁移效果不对称，以及未将IK与碰撞约束紧耦合到优化过程中。

---

## 281. HAMSA: Scanning-Free Vision State Space Models via SpectralPulseNet

**arXiv ID:** 2604.14724 | [PDF](https://arxiv.org/pdf/2604.14724v1)

**作者:** Badri N. Patro `[一作]` (Microsoft), Vijay S. Agneeswaran `[通讯]` (Microsoft)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 HAMSA，一种无需扫描、直接在频域运算的状态空间模型（SSM），通过简化核参数、频域门控机制（SpectralPulseNet）和幅值门控单元（SAGU）实现对 2D 图像的全局长程建模。

**💡 创新点**

核心创新包括：① 用单个高斯初始化的复数核取代传统的 (A,B,C) 三矩阵，消除离散化不稳定性；② SpectralPulseNet 引入输入依赖的频域门控，实现自适应频率选择；③ SAGU 在频域使用幅值门控，保持梯度流并提升非线性表达能力；④ 完全去除扫描步骤，显著降低计算与内存开销。

**🔧 技术方法**

采用 FFT 加速的频域卷积、复数核学习、幅值门控（Sigmoid 对幅值），GLU 类门控（SAGU）、投影层和前馈网络；整体架构在 GPU 上通过 cuFFT 高效实现。

**📊 数据集**

主要使用 ImageNet‑1K 进行基准分类；在 COCO（目标检测/实例分割）和 ADE20K（语义分割）评估稠密预测；在 CIFAR‑10/100、Flowers‑102、Stanford Cars 等数据集进行迁移学习实验。

**📈 对比分析**

与传统扫描式 SSM（Vim、VMamba、SiMBA 等）、ViT、Swin 等 Transformer 以及 CNN 对标。HAMSA 在 ImageNet‑1K 上达到 85.7% top‑1（SSM 最高），推理速度比 DeiT‑S 快 2.2×、比扫描式 SSM 快 1.4‑1.9×，显存 2.1 GB（低于 3.2‑4.5 GB），能耗 12.5 J（低于 18‑25 J）。迁移学习和稠密预测任务中均显著优于同类模型，保持了高效性与准确度。

**⚠️ 局限性**

局限性包括：① 对 GPU FFT 的依赖，导致在低功耗或嵌入式设备上的适配挑战；② 频域门控仅基于幅值，可能忽略相位信息，对细粒度空间关系的捕捉有限；③ 仍需精细调节核初始化与门控参数，训练稳定性对超参敏感；④ 目前仅在 2D 图像任务验证，视频、三维数据的泛化仍待进一步研究。

---

## 282. SynHAT: A Two-stage Coarse-to-Fine Diffusion Framework for Synthesizing Human Activity Traces

**arXiv ID:** 2604.14705 | [PDF](https://arxiv.org/pdf/2604.14705v1)

**作者:** Rongchao Xu `[一作]` (Florida State University), Guang Wang `[通讯]` (Florida State University)

**通讯引用:** 86410 | [OpenAlex ID](https://openalex.org/A5032583158)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种两阶段粗细分层扩散框架 SynHAT，用于高效合成细粒度人类活动轨迹（HATs），同时保持数据隐私和实用性。

**💡 创新点**

创新点包括：① 将不规则、离散的 HAT 转化为连续、规则的潜在时空轨迹并分两阶段生成；② 设计带双分支 Drift‑Jitter 的 Latent Spatio‑Temporal UNet 以更好捕捉空间平滑与时间抖动；③ 通过行为模式提取模块 (BPEM) 与全局上下文 FiLM 进行条件生成；④ 语义对齐模块将连续轨迹映射回离散 POI 活动。

**🔧 技术方法**

核心技术为扩散概率模型（Diffusion Models）与 UNet 架构，辅以注意力机制（Transformer encoder）、时间位置编码、深度可分离卷积、GC‑FiLM 以及四叉树空间检索。

**📊 数据集**

使用四个真实城市数据集：纽约市（Foursquare）、东京（Foursquare）、奥斯汀（Foursquare）和斯德哥尔摩（Gowalla），涵盖不同国家和城市规模。

**📈 对比分析**

在空间与时间指标（JSD、距离、半径、间隔、长度）上，相比 SMM、TimeGEO、Hawkes、SeqGAN、MoveSim、DiffTraj、ControlTraj、GeoLlama 等基线，SynHAT 以 52%/33% 的提升领先；在下游任务（POI 推荐、时间预测）中实现了接近真实数据的性能，并在隐私评估中表现出较低的相似度；在计算效率上，粗细分层设计显著降低 FLOPs 与显存占用。

**⚠️ 局限性**

局限性：① 隐私评估仅基于相似度阈值，缺乏形式化的差分隐私或攻击模型；② 对细粒度轨迹生成的模型复杂度仍高，极细粒度或大规模序列时仍受限；③ LLM 及图神经网络等更强模型在空间建模上表现欠佳，未来仍需提升；④ 在极端稀疏或高度多样化的数据上，生成质量与泛化仍待验证。

---

## 283. Accelerating CRONet on AMD Versal AIE-ML Engines

**arXiv ID:** 2604.14700 | [PDF](https://arxiv.org/pdf/2604.14700v1)

**作者:** Kaustubh Mhatre `[一作]` (Arizona State University), Aman Arora `[通讯]` (Arizona State University)

**通讯引用:** 502 | [OpenAlex ID](https://openalex.org/A5045858420)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在AMD Versal AIE-ML平台上实现并加速CRONet（一个混合CNN‑RNN网络）用于拓扑优化，完全在芯片上完成所有计算，采用数据流、L1/L2/L3融合和拥塞感知放置，实现低延迟和高能效推理。

**💡 创新点**

①首次将完整的混合CNN‑RNN网络映射到AIE-ML阵列；②提出多级融合策略（L1/ L2/ L3）实现中间激活全在芯片；③开发拥塞感知放置算法，显著降低编译失败率与时间；④构建可复用的参数化算子库，方便迁移到其他网络。

**🔧 技术方法**

利用AMD Versal AIE‑ML的向量处理器与内存瓷块，编写自定义算子（3D卷积、GEMM、RNN、池化、SiLU）和数据流图；采用BF16精度、全在芯片权重与激活、L1/ L2/ L3融合、内存瓷块缓冲、GMIO接口、以及基于图的拥塞感知放置。

**📊 数据集**

采用拓扑优化仿真数据，测试三种输入尺寸（30×10、30×20、60×20）的CRONet模型，使用预训练的CRONet网络作为基准；训练数据来源于传统FEA求解结果，用于验证量化精度和推理准确性。

**📈 对比分析**

与功耗相同的Nvidia T4 GPU（FP32）对比，AIE‑ML实现实现延迟从1.19 ms下降到0.52 ms（≈2.29×加速），功耗从35 W降至21 W，能效提升至3.79×；在更大尺寸（60×20）时仍保持低增量延迟（0.82 ms），能效为2.62×。

**⚠️ 局限性**

受限于GMIO带宽和内存瓷块描述符数量；AIE‑ML数据流模型缺乏子图迭代控制和运行时图重配置；无法充分利用可编程逻辑，且卷积2D及自适应池化因内存访问模式导致瓶颈；对更大模型或更复杂网络的可扩展性尚待验证。

---

## 284. DETR-ViP: Detection Transformer with Robust Discriminative Visual Prompts

**arXiv ID:** 2604.14684 | [PDF](https://arxiv.org/pdf/2604.14684v1)

**作者:** Bo Qian `[一作]` (Xi'an Jiaotong University), Xing Wei `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 20832 | [OpenAlex ID](https://openalex.org/A5100344556)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于视觉提示的开放词汇目标检测框架，改进了Grounding DINO，形成VIS‑GDINO及其升级版；

**💡 创新点**

创新点包括：全局提示融合（Global Prompt Integration）提升提示语义分布的正负样本丰富度；视觉‑文本提示关系蒸馏（Visual‑Textual Prompt Relation Distillation）把文本语义结构直接迁移到视觉提示；选择性融合（Selective Fusion）在交互式检测中自动过滤无关提示，实现稳定的检测；

**🔧 技术方法**

技术上结合了图像‑文本对比学习、CLIP/BERT语义嵌入、Swin Transformer backbone、可变形注意力、跨模态对齐损失和关系蒸馏损失，并采用了多阶段训练策略；

**📊 数据集**

主要使用的公开数据集有COCO、LVIS、ODinW、Roboflow100作为评测集，训练集则包括Objects365、GoldG（含GQA、Flickr30k），且不使用COCO、LVIS等评测集进行训练；

**📈 对比分析**

在零样本视觉提示检测（Visual‑G）协议下，实验显示相较于T‑Rex2和YOLOE，-T版本在COCO上提升约4.4 mAP，在LVIS上提升6.9 mAP；-L版本更进一步，在COCO、LVIS、ODinW、Roboflow100上均超过现有方法，尤其在罕见和常见类别的AP提升显著；

**⚠️ 局限性**

局限性在于：视觉提示仍整体低于文本提示，尤其在多样化视觉语义表征上存在偏差；全局提示融合需要收集批量样本，训练效率受限；选择性融合在极少提示场景下仍可能导致信息丢失；未来需进一步提升跨域鲁棒性和提示生成效率。

---

## 285. CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction

**arXiv ID:** 2604.14651 | [PDF](https://arxiv.org/pdf/2604.14651v1)

**作者:** Sizhe Wang `[一作]` (Washington University in St. Louis), Chenyang Lu `[通讯]` (Washington University in St. Louis)

**通讯引用:** 20207 | [OpenAlex ID](https://openalex.org/A5034805517)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了CURA框架，用以在临床语言模型的风险预测中实现个体与群体层面的不确定性校准。

**💡 创新点**

创新点在于将个体误差概率与邻域风险结合的双层校准损失，并将其等价解释为带邻域软标签的交叉熵损失；同时仅在冻结的LM特征上训练轻量化多头分类器，无需额外推理或文本解释。

**🔧 技术方法**

使用冻结的临床LM（如BioGPT、BioClinicalBERT、ClinicalBERT）提取嵌入；在此基础上训练多头MLP分类器，加入个体不确定性损失L_ind和群体一致性损失L_coh进行微调。

**📊 数据集**

在MIMIC‑IV数据库的临床笔记上，评估5个风险预测任务（7‑日死亡、30‑日死亡、住院死亡、ICU逗留≥1天、12小时早期出院）。

**📈 对比分析**

与基线、Deep Ensemble、MC Dropout等方法在AUROC、AUPRC、Brier、NLL、AURC等指标上比较，CURA在校准指标上显著提升（Brier、NLL、AURC均下降），同时保持甚至略升AUROC/AUPRC，并显著降低错误确信率。

**⚠️ 局限性**

局限性包括：仅针对无文本解释的二分类风险任务，未验证对闭源API模型；实验仅在单中心单模态（文本）数据上进行，未覆盖多模态或长期序列情境；未直接解决MIMIC数据固有的公平性与偏差问题。

---

## 286. Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting

**arXiv ID:** 2604.14648 | [PDF](https://arxiv.org/pdf/2604.14648v1)

**作者:** Inseok Jeon `[一作]` (Yonsei University), Sangyoun Lee `[通讯]` (Yonsei University)

**通讯引用:** 3441 | [OpenAlex ID](https://openalex.org/A5015739530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了Seen-to-Scene框架，实现视频outpainting，将传播式与扩散式两大范式统一在一个端到端系统中。

**💡 创新点**

创新点在于引入参考引导的潜在层传播，解决了传统流完成网络在outpainting中的域差距，并通过轻量化细化模块提升了传播精度；同时在扩散模型中使用潜在传播作为条件，兼顾源内容保留与未观测区域生成，显著提升时空一致性。

**🔧 技术方法**

使用的技术包括RAFT光流估计、针对视频inpainting预训练的流完成网络FCNet（随后微调）、VAE编码/解码、3D-UNet式潜在扩散模型、参考框架选取与潜在层传播、以及轻量化细化模块。

**📊 数据集**

训练数据仅采用公开的YouTube‑VOS训练集约10万段视频；评估数据集为DAVIS 2017和YouTube‑VOS测试集。

**📈 对比分析**

在DAVIS和YouTube‑VOS上与现有零/一轮方法对比，PSNR、SSIM、LPIPS、FVD均获得最优或接近最优表现，且推理速度更快（≈12 s、8.8 GB峰值显存），明显提升了时空一致性与视觉质量。

**⚠️ 局限性**

局限性在于潜在层传播可能导致细节衰减，仍需依赖对流完成网络进行微调；对极端动态场景或大范围扩展的细节处理尚未完全成熟。

---

## 287. Energy-based Regularization for Learning Residual Dynamics in Neural MPC for Omnidirectional Aerial Robots

**arXiv ID:** 2604.14678 | [PDF](https://arxiv.org/pdf/2604.14678v1)

**作者:** Johannes Kübel `[一作]` (University of Tokyo), Moju Zhao `[通讯]` (University of Tokyo)

**通讯引用:** 1175 | [OpenAlex ID](https://openalex.org/A5045076994)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在多向无人机上实现基于能量正则化的神经网络模型预测控制（Neural MPC），并将残差动力学整合进MPC框架；

**💡 创新点**

提出一种新的能量正则化损失函数，训练神经网络时同时最小化预测误差与模型能量变化，促使网络输出在物理上更稳定、能量更低；

**🔧 技术方法**

使用多层感知器（MLP）网络、RK4积分器、SQP求解器、AdamW优化器，以及基于能量的正则化损失；

**📊 数据集**

利用OptiTrack 100 Hz运动捕捉系统采集的15 分钟飞行数据（包含起飞、圆形轨迹、姿态轨迹）训练模型；

**📈 对比分析**

与纯解析MPC和基线神经MPC（RTNMPC）进行对比，三种轨迹（起飞、圆形、定点）实验中，能量正则化神经MPC将位置MAE从平均0.1466 m降低至0.1122 m，较RTNMPC提升约15%，同时提升飞行稳定性；

**⚠️ 局限性**

仍需更多训练数据与更优网络结构以进一步提升精度，残差模型在某些坐标轴仍存在稳态偏差，且能量正则化对计算负担的影响尚待评估。

---

## 288. Zero-Shot Retail Theft Detection via Orchestrated Vision Models: A Model-Agnostic, Cost-Effective Alternative to Trained Single-Model Systems

**arXiv ID:** 2604.14846 | [PDF](https://arxiv.org/pdf/2604.14846v1)

**作者:** Haileab Yagersew `[一作]` `[通讯]` (Paza AI), Haileab Yagersew (Paza AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个零-shot、模型无关的商店失窃检测框架 Paza，通过多模型层级管道在不训练任何模型的情况下实现隐蔽行为检测。

**💡 创新点**

创新点在于将廉价实时检测模型与可控触发的昂贵视觉语言模型相结合，利用多信号预过滤将 VLM 调用压缩 240 倍，实现成本低、可扩展且可热插拔的系统。

**🔧 技术方法**

采用 YOLO、ByteTrack、YOLO‑Pose 进行连续检测，行为预过滤逻辑与多帧缓冲，使用任何 OpenAI 兼容的 VLM（Gemma 4、Qwen3.5‑Omni、GPT‑4o 等）进行多帧时间推理，并通过结构化提示与解析实现自动阈值判定。

**📊 数据集**

在合成的 DCSASS 商店失窃数据集（169 条 640×480 视频片段）上进行零-shot 评估。

**📈 对比分析**

与传统需要多年数据训练的商业系统（每店每月 200–500 美元）相比，Paza 仅需 50–100 美元/店，VLM 调用率 10–60 次/小时；在 DCSASS 上取得 89.5% 的精度、92.8% 的特异性，召回 59.3%，显示在未训练的情况下已具备可用的检测效果。

**⚠️ 局限性**

局限在于预过滤可能漏检极端行为、VLM 仍需网络/显卡资源导致延迟、评估仅基于合成数据、模型易受姿态误差与假阳性影响，需要真实商店部署验证。

---

## 289. Matched and Euclidean-Mismatched Decoding on Fourier-Curve Constellations with Tangent Noise

**arXiv ID:** 2604.14844 | [PDF](https://arxiv.org/pdf/2604.14844v1)

**作者:** Bin Han `[一作]` (RPTU University Kaiserslautern-Landau), Hans D. Schotten `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在曲线星座上投加切向人工噪声时，匹配与不匹配解码的误码率差异，给出欧氏与匹配判决的精确对偶错误概率公式，并对 Fourier 曲线星座进行完整的误码率上界和下界分析。

**💡 创新点**

①将人工噪声投射到曲线切向空间，使每个符号产生符号相关的秩一协方差，从而在匹配解码中出现新的结构性误差；②对 Fourier 曲线给出完全解析的匹配误差表达式和欧氏误差谱；③揭示匹配与欧氏误差随曲线维度 k 变化的趋势（趋向正交切向不匹配）。

**🔧 技术方法**

主技术包括：曲线参数化、切向与弦的几何关系推导、Gauss–Hermite 数值积分、Sherman–Morrison 逆矩阵公式、对偶错误概率分析以及欧氏误差的多项式上界/下界推导。

**📊 数据集**

无真实数据集；所有结果均基于解析推导和 Monte‑Carlo 仿真（对指定的 (k,M) 组合进行 2×10⁴ 次实验）得到的误码率曲线。

**📈 对比分析**

通过与欧氏最近邻解码的误码率进行比较。匹配解码在 β>0 时表现出明显的误码率下降，尤其在大 β 时差距更大；欧氏误差上界随着 β 增大而变得更宽松。对统一偶数星座，给出了精确的上下界，验证了理论与仿真的一致性。

**⚠️ 局限性**

局限性包括：①未对通道安全率（secrecy‑rate）进行分析；②匹配解码的完整代码本误码率公式仅在极值（对偶）情形下给出，普遍偏移的匹配误码率仍缺乏解析表达；③研究仅聚焦 Fourier 曲线星座，对其他更一般的曲线星座需数值求解；④实验仅在有限维度和码字数下验证，未考虑大规模实现的计算复杂度。

---

## 290. Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation

**arXiv ID:** 2604.14839 | [PDF](https://arxiv.org/pdf/2604.14839v1)

**作者:** Jinfeng Xu `[一作]` (University of Hong Kong), Edith C. H. Ngai `[通讯]` (University of Hong Kong)

**通讯引用:** 6366 | [OpenAlex ID](https://openalex.org/A5077317339)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Semantically Guaranteed User Representation Initialization（SG-URInit）方法，解决多模态推荐中用户初始化的随机性问题。

**💡 创新点**

创新点在于将用户交互物品的多模态特征与聚类层级的全局信息融合，形成语义保证的用户表示，同时实现训练无关、模型无关。

**🔧 技术方法**

使用了K-means聚类、度数加权聚合、局部与全局特征融合等技术，并在现有多模态推荐模型（如MMGCN、MENTOR、FREEDOM等）中直接嵌入。

**📊 数据集**

实验使用了Amazon Baby、Sports、Clothing数据集以及多模态TikTok数据集，包含文本、视觉、音频等多模态特征。

**📈 对比分析**

通过与六种先进多模态推荐基线以及对抗训练、LLM数据增强等策略的组合进行对比，使用Recall@K/NDCG@K评价，SG-URInit在所有模型、数据集上均显著提升性能，缩小冷启动误差并加速收敛。

**⚠️ 局限性**

局限性包括需要预先执行聚类与聚合，虽然训练无关但存在预处理开销；对聚类中心数K和融合权重λ敏感，需调参；在极端稀疏场景下局部信息可能不足，仍需进一步改进。

---

## 291. Beyond Literal Summarization: Redefining Hallucination for Medical SOAP Note Evaluation

**arXiv ID:** 2604.14829 | [PDF](https://arxiv.org/pdf/2604.14829v1)

**作者:** Bhavik Vachhani `[一作]`, Sai Chiranthan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在生成SOAP医学记录时的幻觉问题，并提出了基于临床推理的判别框架。

**💡 创新点**

通过设计推理感知的LLM as Judge评估方法和检索增强的医学知识库，成功区分合法推理与真正幻觉，显著降低幻觉率。

**🔧 技术方法**

采用分层判别流程、五层推理分类、链式思考的评估模型，并结合SNOMED CT/ICD‑10等医学知识库进行检索增强。

**📊 数据集**

构建了100份匿名医生‑患者对话集以及对应的SOAP记录，用以评估模型，基准为人工专家标注。

**📈 对比分析**

与传统字面依据评估相比，Stage 1的幻觉率为35.2%，通过Stage 2降至9.1%，与人工基准10.4%相近，误报率大幅降低。

**⚠️ 局限性**

评估方法依赖手工构建的知识库和提示，可能在不同医学领域或语言环境下泛化受限，未覆盖所有临床推理细节。

---

## 292. Pangu-ACE: Adaptive Cascaded Experts for Educational Response Generation on EduBench

**arXiv ID:** 2604.14828 | [PDF](https://arxiv.org/pdf/2604.14828v1)

**作者:** Dinghao Li `[一作]`, Yaochen Li `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 761 | [OpenAlex ID](https://openalex.org/A5026770433)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了Pangu-ACE教育级联系统，在EduBench shared‑8中文测试集上实现了1B路由器草稿+7B专家提升，并纠正了评估中的错误。

**💡 创新点**

创新点在于采用样本级风险校准的路由决策，结合低成本草稿与高成本专家改进，并引入离线artifact‑first评估与严格的质量校验，展示任务依赖的路由选择。

**🔧 技术方法**

使用了1B tutor‑router草稿生成、7B specialist prompt、风险阈值校准、验证与修复、artifact日志、CPU侧重重算以及严格的deterministic quality度量。

**📊 数据集**

使用了EduBench shared‑8中文测试集（7013条）以及快速诊断子集（354中文、368英文），涵盖任务如Q&A, AG, EC, IP, PCC, PLS, QG, TMG。

**📈 对比分析**

通过与传统规则路由(rule_v2)和7B‑only的对比，Pangu-ACE在deterministic quality提升0.081、格式有效率提升0.159、1B直接接受率19.7%；但整体端到端延迟仍高，未实现速度提升。

**⚠️ 局限性**

局限性包括仅中文评估；路由提升未转化为壁钟速度收益；缺乏GPT‑5.4基线对齐；高风险任务仍需7B调用；artifact‑first离线评估缺乏在线验证。

---

## 293. SWE-TRACE: Optimizing Long-Horizon SWE Agents Through Rubric Process Reward Models and Heuristic Test-Time Scaling

**arXiv ID:** 2604.14820 | [PDF](https://arxiv.org/pdf/2604.14820v1)

**作者:** Hao Han `[一作]` (vivo), Qingwen Ye `[通讯]` (vivo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SWE-TRACE框架，统一优化软件工程代理的训练、强化学习与推理流程。

**💡 创新点**

创新点：①通过LLM多任务级联与oracle验证生成短路径高质量SFT轨迹；②引入基于Rubric的过程奖励模型和记忆增强的GRPO，实现密集过程级反馈；③在推理阶段重用PRM进行动作级启发式TTS，显著降低延迟并提高效率。

**🔧 技术方法**

技术手段包括LLM多任务级联、oracle验证、Rubric Agent与Process Reward Model、记忆增强长序列架构、GRPO强化学习、启发式动作采样等。

**📊 数据集**

数据集：在77个可执行GitHub仓库上生成60K高质量合成问题实例，用于SFT；评估使用SWE-bench Verified（500个人类验证的真实问题）。

**📈 对比分析**

与现有4B/30B模型及公开SWE代理对比，SWE-TRACE-30B在SWE-bench Verified上取得71.2% Pass@1，超过前沿公开32B结果；SWE-TRACE-4B提升约4–5% resolve率，且在token使用和推理延迟方面均有显著改进。

**⚠️ 局限性**

局限：仍依赖大量合成数据和oracle验证，处理极大仓库或复杂工具链时可扩展性受限；对测试用例质量敏感，Flaky测试可能导致奖励噪声。

---

## 294. Domain Fine-Tuning FinBERT on Finnish Histopathological Reports: Train-Time Signals and Downstream Correlations

**arXiv ID:** 2604.14815 | [PDF](https://arxiv.org/pdf/2604.14815v1)

**作者:** Rami Luisto `[一作]` (University of Jyvaskylä), Sami Äyrämö `[通讯]` (University of Jyvaskylä)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文在芬兰医疗文本上对FinBERT进行领域微调，并尝试通过训练过程中的嵌入几何变化来预测微调对下游分类任务的提升。

**💡 创新点**

创新点在于提出利用训练时的嵌入变化（CKA、Procrustes、RSA、等距性、聚类等指标）来预测模型微调收益，而非仅关注最终性能。

**🔧 技术方法**

技术上采用了FinBERT（BERT变体）在MLM任务上的领域微调，并使用CKA、Procrustes、RSA、等距性、聚类等方法对嵌入进行分析；下游分类器则使用简单的逻辑回归和k‑NN。

**📊 数据集**

使用的数据集包括芬兰新闻、法律（Finlex）、文学（Gutenberg）、维基百科、字幕、网络爬取、议会文本（Eduskunta）以及来自中央芬兰生物库的病理医学样本。

**📈 对比分析**

比较方法是将微调前后的模型在同一未标记测试集的CLS嵌入差异与在少量标记数据（100–1000样本）上训练的分类器性能提升进行关联；结果显示训练损失下降、低层嵌入变动和等距性提升与分类性能提升存在一定关联，但总体效果有限。

**⚠️ 局限性**

局限性包括仅有9个领域样本、关联可能为偶然、未深入分析模型权重变化、未进行大规模复现，且实验聚焦于小规模标记数据的分类器，难以推广到更大规模的实际应用。

---

## 295. Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem

**arXiv ID:** 2604.14808 | [PDF](https://arxiv.org/pdf/2604.14808v1)

**作者:** Zeguan Xiao `[一作]` (Shanghai University of Finance and Economics), Guanhua Chen `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 18166 | [OpenAlex ID](https://openalex.org/A5015168873)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了将LLM去学习视为不对称的两任务学习问题，设计了以保留为主、遗忘为辅的梯度合成框架，并实现了模块级PCGrad和新颖的SAGO方法；

**💡 创新点**

创新点包括：①从不对称任务角度重新定义去学习目标；②模块级PCGrad以细粒度抑制梯度冲突；③SAGO通过逐元素符号对齐门控，保证更新方向始终与保留梯度一致，提升保留性能；

**🔧 技术方法**

采用梯度合成技术、PCGrad投影、SAGO符号对齐门控以及常规的梯度上升/下降去学习损失；

**📊 数据集**

使用WMDP（Bio/Cyber）和RWKU两个公开的LLM去学习基准数据集；

**📈 对比分析**

与GA、NPO、SimNPO、GradDiff等基线及PCGrad进行对比，SAGO在保留指标上显著提升（如WMDP Bio MMLU从~56提升到~57+），忘记指标保持相近或略降，整体拉升Pareto前沿；

**⚠️ 局限性**

局限性：仅在两类基准上验证，未覆盖多模态或代码模型；去学习过程需要额外计算/存储梯度，资源开销略大；

---

## 296. The LLM Fallacy: Misattribution in AI-Assisted Cognitive Workflows

**arXiv ID:** 2604.14807 | [PDF](https://arxiv.org/pdf/2604.14807v1)

**作者:** Hyunwoo Kim `[一作]` (ddai Inc.), Hanau Yi `[通讯]` (ddai Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并系统阐述了“LLM 落后”现象，即在 LLM 辅助下用户将系统输出错误地当作自身独立能力的证明，从而导致自我认知与实际能力的偏差。

**💡 创新点**

创新点在于将自动化偏差、认知外包与分布式认知理论与 LLM 辅助工作结合，形成了新的归因失误概念；同时给出机制模型、跨域表现类型表以及对评估体系的影响分析。

**🔧 技术方法**

主要采用人机协同的写作流程：使用大型语言模型（如 GPT‑4）在结构化提示下生成论文草稿、逻辑梳理与语言润色，随后由作者进行人工校验与整合。

**📊 数据集**

该研究为概念性工作，未使用公开数据集；作者仅以迭代提示方式与 LLM 交互，构建自身研究文本。

**📈 对比分析**

文章未进行实验比较或性能评测；只在文献回顾与案例观察中说明 LLM 生成结果的可观察质量与潜在误导性，并未给出定量指标或与基线对比。

**⚠️ 局限性**

局限性包括：缺乏实证验证与量化指标；未探讨不同 LLM 版本或提示策略对落后现象的具体影响；对跨领域应用的泛化性尚待后续实验检验。

---

## 297. Knowing When Not to Answer: Evaluating Abstention in Multimodal Reasoning Systems

**arXiv ID:** 2604.14799 | [PDF](https://arxiv.org/pdf/2604.14799v1)

**作者:** Nishanth Madhusudhan `[一作]` (ServiceNow Research), Alexandre Lacoste `[通讯]` (ServiceNow Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MM‑AQA基准，构造多模态不可答样本并评估VLM与多代理系统在弃答上的表现。

**💡 创新点**

通过两轴（模态依赖与证据充分性）系统化的变换生成不可答样本，使用双共识VLM质控与人类验证，揭示前沿模型缺乏校准且弃答能力不足。

**🔧 技术方法**

利用VLM推理、三代理MAS（Reasoner、Verifier、Orchestrator）、多条件提示、置信度阈值、Chain‑of‑Thought、Self‑Consistency、P(True)与最大概率等方法进行评估。

**📊 数据集**

基于MMMU与MMLongBench‑Doc构建的2079条样本（A‑MMMU 553条，A‑MMLBD 1526条）。

**📈 对比分析**

对三大前沿VLM（Claude Sonnet 4.5、GPT‑5、Qwen 2.5‑32B‑VL）及其MAS顺序/迭代版本进行AAC、UAC、AR、MCC等指标比较，最高MCC仅0.344，远低于人类水平≈0.83，表明仍存在显著改进空间。

**⚠️ 局限性**

仅限于静态图文任务，未扩展至视频或多文档；双共识QC使用与评估同一族VLM导致循环依赖；缺乏弃答训练机制，模型主要表现为校准不足。

---

## 298. Evaluating Encodings for Bivariate Edges in Adjacency Matrices

**arXiv ID:** 2604.14791 | [PDF](https://arxiv.org/pdf/2604.14791v1)

**作者:** Jorge Acosta-Hernández `[一作]` (Polytechnic University of Madrid), Tingying He `[通讯]` (Graz University of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对在稠密邻接矩阵中同时编码两种量化边属性（均值与标准差）的方法进行了系统评估，并通过在线人类实验检验其在多种分析任务中的表现。

**💡 创新点**

首次将四种代表性的双变量边属性编码（双色调、嵌入条形图、叠加面积标记、叠加角度标记）放在同一实验框架下进行对比，且提出了新型的基于角度的叠加标记方案。

**🔧 技术方法**

采用四色调（色调+亮度）、嵌入条形图、叠加面积标记和叠加角度标记四种视觉通道组合；利用预注册的在线实验平台 reVISit 进行任务设计和数据采集；使用 PREVis 与 BeauVis 量化可读性和美感。

**📊 数据集**

基于美国交通统计局（Bureau of Transportation Statistics）B2B 市场数据的国内航空机票价格，构建两张密集网络（12、25 节点），边属性为平均票价和标准差。

**📈 对比分析**

通过 156 名参与者完成八类分析任务（结构、属性、估计），比较每种编码的准确率和完成时间。结果显示：面积叠加标记（πa）在大多数任务中取得最高准确率；角度叠加标记（πb）虽可读性高，但准确率低于面积；双色调（πc）表现最差；嵌入条形图（πd）性能与面积标记相当。

**⚠️ 局限性**

实验仅覆盖中等规模稠密矩阵、单一数据集和两属性摘要；未对每种编码进行参数化优化；未测试更高维度属性、稀疏网络、方向性网络或其他任务类型，结果可能不具普遍适用性。

---

## 299. Sequence Search: Automated Sequence Design using Neural Architecture Search

**arXiv ID:** 2604.14788 | [PDF](https://arxiv.org/pdf/2604.14788v1)

**作者:** Rokgi Hong `[一作]` (Seoul National University), Jongho Lee `[通讯]` (Seoul National University)

**通讯引用:** 18138 | [OpenAlex ID](https://openalex.org/A5100726125)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发了一种基于神经架构搜索的自动序列设计框架，能够在给定组织与成像参数下生成满足特定目标的MR脉冲序列。

**💡 创新点**

创新点在于无需预设序列结构或训练数据，通过NAS与可微Bloch模拟联合优化实现自动化设计，成功生成传统与非传统（如三RF低能量spin‑echo）序列，突破人类直觉限制。

**🔧 技术方法**

主要技术包括ProxylessNAS架构搜索、梯度下降优化、可微Bloch模拟、以及基于信号强度、对比度、RF能量等多目标的损失函数。

**📊 数据集**

使用了模拟的10万体素数据集，采样灰质、白质和脑脊液的T1/T2参数以及B0/B1不均匀性，作为训练与评估素材。

**📈 对比分析**

与传统手工设计和网格搜索对比，在信号强度、GM‑WM对比、RF能量和对B0/B1鲁棒性方面与经典序列相当或更优，并发现低能量三RF spin‑echo等新颖结构。

**⚠️ 局限性**

限制在于搜索空间仅包含RF脉冲调度、假设瞬时脉冲、无限TR、未考虑梯度或读出、稳态效应，且结果高度依赖目标函数与搜索范围，需进一步扩展模型与计算资源。

---

## 300. Exploring and Testing Skill-Based Behavioral Profile Annotation: Human Operability and LLM Feasibility under Schema-Guided Execution

**arXiv ID:** 2604.14843 | [PDF](https://arxiv.org/pdf/2604.14843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 301. Towards Trustworthy 6G Network Digital Twins: A Framework for Validating Counterfactual What-If Analysis in Edge Computing Resources

**arXiv ID:** 2604.14787 | [PDF](https://arxiv.org/pdf/2604.14787v1)

**作者:** Julian Jimenez Agudelo `[一作]` (University of Antwerp), Miguel Camelo Botero `[通讯]` (University of Antwerp)

**通讯引用:** 577 | [OpenAlex ID](https://openalex.org/A5072792091)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种面向6G边缘计算的端到端数据驱动数字孪生框架，集成统一遥测收集、基于工作模式的特征工程和基于符号一致性与方向敏感度的验证方法，用以实现可信的假设分析和资源自适应。

**💡 创新点**

创新点在于：①将ITU参考架构扩展为双层遥测与语义同步管道（T‑DL 与 H‑DL），实现多域遥测统一与语义对齐；②设计基于工作模式的可扩展特征抽象，允许模型在不同负载与资源配置下外推；③提出基于符号一致性与方向敏感度的匹配对验证，为数字孪生在超分布式决策中的可靠性提供定量评估。

**🔧 技术方法**

采用的技术包括 Prometheus/Kafka 采集与流处理、Python/Scikit‑Learn 与 XGBoost、TensorFlow/Keras 深度神经网络、Kubernetes 集群管理、Prefect 编排与 KEDA 自动扩容、InfluxDB 时序数据库以及自定义的 SDM 语义映射与基本模型构建。

**📊 数据集**

使用的实验数据集为在三节点 Kubernetes 集群上，通过合成推理请求生成的 259,016 条遥测样本，覆盖 200、400、600 并发用户以及 1–6 个 Pod 的多种工作模式，目标指标为推理延迟。

**📈 对比分析**

对比方法是将 XGBoost 与 DNN 在相同的特征空间下训练，并在未见的 600 用户工作区进行外推评估。两模型在整体准确度（R²>0.99、MAE≈9 ms）上相近，但 XGBoost 在方向一致率（S_a>0.90）与假设变更的敏感度上表现更好，能够准确捕捉扩容与负载变化导致的延迟变化。

**⚠️ 局限性**

局限性包括：①实验仅覆盖单一服务类型与有限的工作模式；②对时间相关与多维度假设场景的支持不足；③模型对极端高负载下的偏差与鲁棒性尚未充分验证；④实现依赖多种开源组件，部署复杂度较高。

---

## 302. CoTEvol: Self-Evolving Chain-of-Thoughts for Data Synthesis in Mathematical Reasoning

**arXiv ID:** 2604.14768 | [PDF](https://arxiv.org/pdf/2604.14768v1)

**作者:** Zhuo Wang `[一作]` (Fudan University), Zenglin Xu `[通讯]` (Fudan University)

**通讯引用:** 10446 | [OpenAlex ID](https://openalex.org/A5051227924)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于遗传算法的自演化链式思路（CoT）合成框架CoTEvol，用来在不需要人工标注的情况下生成高质量数学推理训练数据。

**💡 创新点**

创新点在于：① 将CoT视为个体，在全局层面实现反射式交叉（Reflective Global Crossover）以实现结构重组；② 在局部层面基于步级不确定性（entropy）进行细粒度变异（Uncertainty‑Guided Mutation）；③ 设计轻量级任务感知的多项式fitness函数，兼顾答案正确性、格式匹配和长度奖励。

**🔧 技术方法**

核心技术包括：遗传算法（GA）变体、基于LLM的反射式反馈交叉、token/step级熵估计的自适应变异、RL‑style verifiers（答案、格式、长度），以及在Qwen2.5-7B-Instruct上进行的自监督演化和后续监督微调。

**📊 数据集**

使用公开数学数据集S1K和LIMO作为训练集，评估基准包括GSM8K、MATH500、GaokaoEn23、OlympiadBench、CollegeMath、AIME24、AMC23等八大数学竞赛与大学水平数据集。

**📈 对比分析**

与基线（人类注释H‑CoT、强模型蒸馏D‑CoT、Best‑of‑N、Self‑Refine）比较，CoTEvol在S1K和LIMO上平均提升≈6.6%推理准确率，且在各模型（Qwen3‑8B、Qwen2.5‑Math‑7B‑Instruct、Deepseek‑R1‑Qwen1.5B‑Instruct、Mistral‑8B）上均表现最优；同时演化成本（FLOPs）比Best‑of‑N和Self‑Refine低约1/3。

**⚠️ 局限性**

局限在于：当前仅选取单一最高fitness的CoT进行监督微调，未充分利用演化产生的多样化推理路径；对极难问题的演化成功率仍受限于基础模型能力；缺乏针对生成多样性有效利用的训练策略。

---

## 303. Efficient closed-form approaches for pose estimation using Sylvester forms

**arXiv ID:** 2604.14747 | [PDF](https://arxiv.org/pdf/2604.14747v1)

**作者:** Jana Vráblíková `[一作]` (Inria), Laurent Busé `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一类基于Sylvester形式的结果式闭式求解器，用于从3D-3D和3D-2D对应关系中估计位姿，并将多项式方程组的次数从传统的9降低到7或8，从而显著减少消元矩阵大小。

**💡 创新点**

创新点在于将Sylvester形式与隐藏变量结果式相结合，利用理想的饱和化与Eagon–Northcott分辨率实现低次数的结果式构造，并证明该方法在数值稳定性和求解精度上不逊于现有高次数方法。

**🔧 技术方法**

使用的技术包括四元数参数化、Lagrange乘子法、隐藏变量结果式、Sylvester形式构造、结果式矩阵构建、QR分解与Schur补，以及对齐块矩阵以获得最优的Q₀、Q₁矩阵。

**📊 数据集**

实验数据集主要包括：在模拟数据（点云随机采样、带噪声和无噪声）上验证精度与计算时间；KITTI LiDAR序列（03、04、07）用于真实场景位姿估计；ETH3D图像序列用于PnP任务评估。

**📈 对比分析**

与Malis、Wientapper、Zhou、UPnP、OPnP、optDLS、SRPnP、SQPnP等现有求解器进行对比，结果显示在相同精度下，deg7/deg8求解器在计算时间上优于deg9和其他方法，且在真实数据上的平均旋转误差与平移误差均不逊于最优方法。

**⚠️ 局限性**

局限性包括：需要手动选择多项式的单词序列以优化块矩阵的条件数，且在极大对应数时矩阵尺寸仍然较大，可能影响极端实时应用；此外，Sylvester形式的构造在不同问题中可能需要针对性设计，未给出统一的自动化选取策略。

---

## 304. Seeking Help, Facing Harm: Auditing TikTok's Mental Health Recommendations

**arXiv ID:** 2604.14832 | [PDF](https://arxiv.org/pdf/2604.14832v1)

**作者:** Pooriya Jamie `[一作]` (University of California, Los Angeles), Homa Hosseinmardi `[通讯]` (University of California, Los Angeles)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在 TikTok 上创建 30 个全新账号，利用 LLM 驱动的模拟用户对“For You”页面进行为期 7 天的实验审计，探究搜索意图（抑郁表达 vs 求助表达）与交互行为（积极参与、主动规避、被动观察）对精神健康内容曝光的影响。

**💡 创新点**

将搜索意图与交互行为分离的实验设计相结合，并采用 LLM 进行实时视频内容识别与观看决策，实现对推荐系统的可控审计，首次系统性评估求助意图在算法中的安全处理缺失。

**🔧 技术方法**

使用多模态大型语言模型（LLM）进行视频内容分类和决策，GPT‑4o‑mini 对视频元数据进行实时识别，实验中通过调整观看时长来模拟不同的交互策略。

**📊 数据集**

收集并分析 8,727 条 TikTok “For You” 页面推荐视频的数据，利用视频标题、标签、字幕、屏幕文字等元数据进行二级人工标注，构建心理健康内容与风险子类型（如自杀/自残、毒性积极性等）的标签集。

**📈 对比分析**

在六种实验条件（两种搜索意图 × 三种交互模式）下比较精神健康内容饱和度、支持性与有害性比例；结果显示交互行为对曝光的影响显著（最大饱和度约 45% 对比 11–20%），而搜索意图仅影响内容组成，且算法未实现针对求助意图的安全过滤，表现为支持性内容比例提升但有害内容仍持续出现。

**⚠️ 局限性**

分类器性能中等（F1 分别为 77.42% / 64.52% / 64.00%），四个种子查询难以覆盖真实用户意图，模拟滚动行为与真实用户存在差距，且实验仅针对新账号并限制在 7 天内，结果不一定能推广到成熟账户或长期动态。

---

## 305. Nautilus: An Auto-Scheduling Tensor Compiler for Efficient Tiled GPU Kernels

**arXiv ID:** 2604.14825 | [PDF](https://arxiv.org/pdf/2604.14825v1)

**作者:** Yifan Zhao `[一作]` (University of Illinois Urbana-Champaign), Sasa Misailovic `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3711 | [OpenAlex ID](https://openalex.org/A5057462458)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

构建了一个从数学表达式自动生成高性能GPU内核的tensor编译器Nautilus

**💡 创新点**

提出了自动调度器支持高级融合、表达式重写、三层IR（VR‑tile、MA‑tile）以及自动选择多种tile编译器的能力

**🔧 技术方法**

使用TVM、Neptune、MetaSchedule、Triton、Tawa、TileLang等技术，进行自适应调度、融合、表达式重写、低层tiling和多后端生成

**📊 数据集**

评估使用GLM、Llama2、Qwen2、Qwen3、ViT等五个Transformer/ViT模型的不同批次、序列长度和精度

**📈 对比分析**

与FlashAttention‑2、SDPA、FlexAttn、Tawa、TileLang、Triton等基线进行比较，Nautilus在GH200上平均提升约1.22×，在RTX5090上约1.26×，在FP8模式下可达1.42×，且在许多设置下超过手工编写的cuDNN kernel

**⚠️ 局限性**

主要限制包括自动调优耗时较长、对部分新GPU特性的支持仍待完善、对非注意力算子覆盖有限、依赖多后端实现的代码质量

---

## 306. Diffusion Crossover: Defining Evolutionary Recombination in Diffusion Models via Noise Sequence Interpolation

**arXiv ID:** 2604.14790 | [PDF](https://arxiv.org/pdf/2604.14790v1)

**作者:** Chisatao Kumada `[一作]`, Tomoyuki Hiroyasu `[通讯]` (Doshisha University)

**通讯引用:** 2229 | [OpenAlex ID](https://openalex.org/A5036543534)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 Diffusion crossover，利用 DDPM 逆向过程中的噪声序列球面线性插值实现 IEC 交叉操作，并在交互式遗传算法框架中进行图像演化探索。

**💡 创新点**

创新点在于：①将交叉操作明确定义为 DDPM 噪声序列的球面线性插值，克服高维生成空间中语义不连贯的交叉难题；②通过控制插值时间段实现多样性与收敛的可调权衡；③将这种基于噪声空间的交叉嵌入人机交互式演化流程。

**🔧 技术方法**

技术包括 Denoising Diffusion Probabilistic Models（DDPM）+ U‑Net 采样、球面线性插值（Slerp）在噪声空间、LPIPS 与 PCA 等评估指标、以及交互式遗传算法（IGA）框架。

**📊 数据集**

实验使用 MNIST（仅数字 5）和 ModelNet40（沙发类别）两类数据集。

**📈 对比分析**

通过 PCA 可视化、LPIPS 距离相关性、平均 LPIPS 多样性评估以及交互式实验验证。结果表明：插值系数与感知相似度呈单调相关，插值时间段越长多样性越低；整体上该方法显著提升搜索效率与语义连贯性。

**⚠️ 局限性**

主要限制包括：生成一代图像耗时约 40–50 秒，难以满足实时交互需求；对细粒度纹理控制效果有限；实验仅由作者自评，缺乏大规模用户研究；模型选择侧重生成质量，未充分考虑多样性和可导航性。

---

## 307. One-shot Compositional 3D Head Avatars with Deformable Hair

**arXiv ID:** 2604.14782 | [PDF](https://arxiv.org/pdf/2604.14782v1)

**作者:** Yuan Sun `[一作]` (Xi'an Jiaotong University), Fei Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 30723 | [OpenAlex ID](https://openalex.org/A5100455803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种一图像单张人像即可构建完整可动画化3D头部头像的组合式方法，显式将面部与头发分离，分别采用面部的FLAME驱动和头发的盒子-位置基动力学（PBD）进行动画。

**💡 创新点**

创新点包括：① 用双图像提升（原图+无头发图）实现高保真纹理重建；② 通过语义分割与边界自适应重分配精确分离头发与面部；③ 采用盒子（cage）+MVC+PBD的无学习物理头发动力学，配合代理碰撞约束实现逼真的重力和惯性效果；④ 将面部与头发统一渲染进3D Gaussian Splatting管线。

**🔧 技术方法**

使用的关键技术有：3D Gaussian Splatting (3DGS)、FaceLift 图像到3D提升、FLAME 参数化面部网格、Mean Value Coordinates (MVC)、Position-Based Dynamics (PBD)、代理碰撞约束、SAM2语义分割、边界自适应重分配。

**📊 数据集**

训练与评估数据集：NeRSemble、Ava256、VFHQ（共50个身份、不同表情与头部运动）。

**📈 对比分析**

与GPAvatar、Portrait4D‑v2、VOODOO3D、CAP4D、GAGAvatar、LAM等方法对比，本文在PSNR、SSIM、LPIPS、身份相似度（Cosine）以及表情/姿态一致性（AED/APD）上均实现了领先或相当的性能；视觉上头发运动更自然、细节保真度更高。

**⚠️ 局限性**

局限性包括：对严重遮挡（如麦克风）效果差；仅适用于正面人像；头发仍以粗粒度Gaussian呈现，未实现细束级动态；依赖精确的头发去除算法，若失败会影响后续步骤。

---

## 308. ASGNet: Adaptive Spectrum Guidance Network for Automatic Polyp Segmentation

**arXiv ID:** 2604.14755 | [PDF](https://arxiv.org/pdf/2604.14755v1)

**作者:** Yanguang Sun `[一作]` (Nanjing University of Science and Technology), Lei Luo `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 2447 | [OpenAlex ID](https://openalex.org/A5100372684)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于自适应频谱引导的ASGNet网络，用于准确分割结肠镜图像中的息肉

**💡 创新点**

创新点在于将频域全局信息与空间域局部信息通过自适应频谱滤波器（ASF）融合，构建了Spectrum‑Guided Non‑local Perception（SNP）模块、Multi‑Source Semantic Extractor（MSE）以及Dense Cross‑Layer Interaction Decoder（DCI）三大组件，实现全局与局部特征的高效交互

**🔧 技术方法**

采用Fast Fourier Transform、深度卷积网络（ResNet50/PVTv2）、自注意力、深度可分离卷积、频谱加权与多尺度空间注意力等技术，构建端到端的语义分割框架

**📊 数据集**

使用五个公开息肉分割数据集：CVC‑300、CVC‑ColonDB、ETIS‑Larib、Kvasir、CVC‑ClinicDB进行训练与评估

**📈 对比分析**

与21种现有CNN/Transformer基准模型对比，ASGNet在Dice、IoU、F_m^w、S_m、E_m、M等指标均取得最高或接近最高分，平均提升约3%–5%，同时保持合理的模型规模与推理速度

**⚠️ 局限性**

在极小息肉或复杂肠黏膜背景下仍易出现漏检或边界模糊，需进一步提升小目标检测能力与训练样本多样性

---

## 309. Disentangle-then-Refine: LLM-Guided Decoupling and Structure-Aware Refinement for Graph Contrastive Learning

**arXiv ID:** 2604.14746 | [PDF](https://arxiv.org/pdf/2604.14746v1)

**作者:** Zhaoxing Li `[一作]` (Anhui University), Xiaoming Zhang `[通讯]` (Anhui University)

**通讯引用:** 477478 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SDM-SCR 框架，利用大语言模型对文本属性图进行任务导向的语义解耦，并通过结构感知的低通滤波进一步精炼表征。

**💡 创新点**

创新点在于将指令驱动的 LLM 语义解耦与基于图谱的低频滤波相结合，实现“Disentangle‑then‑Refine”范式，既去除噪声又保留语义信息。

**🔧 技术方法**

技术包括：近似正交分解（Semantic Decoupling Module, SDM）、LLM 指令生成相关/无关视图、对比学习的异构正负采样、语义一致性正则化（Semantic Consistency Regularization, SCR）即低通谱滤波。

**📊 数据集**

使用的公开数据集有 Citeseer、Wiki‑CS、Pubmed、Ele‑Photo、Books‑History 等文本属性图。

**📈 对比分析**

与现有 GCL 方法（如 GRACE、MVGRL、GAugLLM 等）在节点分类任务中对比，SDM‑SCR 在各数据集均取得 SOTA 结果，并在推理效率上优于其他 LLM‑增强方法。

**⚠️ 局限性**

局限性包括：LLM 生成的语义解耦不完全，仍有幻觉噪声；对大型图的推理受 LLM 调用成本影响；需要针对不同任务设计合适的指令，可能影响迁移性。

---

## 310. TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models

**arXiv ID:** 2604.14847 | [PDF](https://arxiv.org/pdf/2604.14847v1)

**作者:** Yi Zhao `[一作]` (Shanghai Jiao Tong University), Hai Zhao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6940 | [OpenAlex ID](https://openalex.org/A5036050911)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于触发器的协同推理框架TrigReason，能够在小推理模型（SRM）与大推理模型（LRM）之间高效分工，显著降低推理延迟与成本；

**💡 创新点**

创新点在于先系统性识别SRM的三类失效风险（路径分歧、认知负荷、恢复无能），并针对每类设计专属触发器（策略前置、认知卸载、干预请求），实现从连续轮询到稀疏、精准的LRM干预；

**🔧 技术方法**

技术手段包括：基于SRM token级困惑度阈值检测认知超负荷；基于停顿词检测推理停滞；在起始阶段让LRM生成规划步骤；以及动态调整SRM与LRM的交替生成；

**📊 数据集**

主要使用的基准数据集为AIME24、AIME25和GPQA‑Diamond；在实验中也验证了在BBH、ARC等逻辑与常识推理任务上的通用性；

**📈 对比分析**

与SpecReason以及单独使用LRM/SRM的对比表明，TrigReason在保持或超过LRM准确率的同时，SRM占比提升1.7–4.79倍；在边缘‑云场景中，延迟降低43.9%，API成本下降73.3%；

**⚠️ 局限性**

局限性包括：触发器阈值与判断依据（如过度自信）基于经验设定，缺乏理论解释；对内存受限环境可能因SRM占用空间过大而受限；对极端大模型或不同任务的适配性仍需进一步验证。

---

## 311. Improved Multiscale Structural Mapping with Supervertex Vision Transformer for the Detection of Alzheimer's Disease Neurodegeneration

**arXiv ID:** 2604.14837 | [PDF](https://arxiv.org/pdf/2604.14837v1)

**作者:** Geonwoo Baek `[一作]` (Hankuk University of Foreign Studies), Ikbeom Jang `[通讯]` (Hankuk University of Foreign Studies)

**通讯引用:** 388 | [OpenAlex ID](https://openalex.org/A5026873215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了MSSM+、SSVM和SV‑ViT，用单张T1w MRI实现阿尔茨海默病与认知正常者的表型分类。

**💡 创新点**

创新点在于：①在MSSM基础上加入了sulcal depth和cortical curvature；②设计了基于ROI约束的Surface SuperVertex Mapping（SSVM）；③提出了使用Supervertex Vision Transformer（SV‑ViT）实现面向表面数据的注意力学习。

**🔧 技术方法**

技术方法包括多尺度表面特征提取、PLS‑DA降维与年龄校正、SSVM分割、SV‑ViT Transformer架构，并与HGNN+、SplineCNN、SpiralNet++、SiT等模型做对比。

**📊 数据集**

使用ADNI（GO、2、3、4）和OASIS（3、4）共1988份3T T1w扫描，涵盖Siemens、GE、Philips三种磁共振制造商。

**📈 对比分析**

通过4‑折交叉验证评估，MSSM++SV‑ViT在AUROC/ AUPRC上分别达0.931/0.887，明显优于CT、GWCs、MSSM及其他图卷积/Transformer模型，且在不同厂家数据上表现稳健。

**⚠️ 局限性**

局限性包括：SSVM的规则化分割导致边界覆盖不足；样本量不平衡（Siemens占大多数）；年龄校正基于小样本，可能残留混杂；未检验轻度认知障碍进展及与CSF/PET生物标志物的结合。

---

## 312. Switch: Learning Agile Skills Switching for Humanoid Robots

**arXiv ID:** 2604.14834 | [PDF](https://arxiv.org/pdf/2604.14834v1)

**作者:** Yuen-Fui Lau `[一作]` (Hong Kong University of Science and Technology), Ping Tan `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 14194 | [OpenAlex ID](https://openalex.org/A5084953118)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一个名为 Switch 的系统，实现了类人机器人在多种动态技能（如踢腿、翻滚、舞蹈等）之间的无缝、可即时切换，并在实际 Unitree G1 机器人上验证了其鲁棒性。

**💡 创新点**

核心创新点包括：① 用技能图（Skill Graph）对多技能数据进行增广，自动生成跨技能的可行转移；② 在图中插入缓冲节点来平滑长距离转移并通过奖励引导学习；③ 在强化学习训练中加入足部接地奖励（Foot‑Ground Contact Reward）提升低足运动的精度；④ 在线调度器通过图搜索即时规划路径，既能实现用户指定的技能切换，又能在误差或扰动出现时进行安全恢复。

**🔧 技术方法**

技术手段主要包括：基于 IsaacGym 的 PPO 强化学习；多技能动力学控制框架；运动图（motion graph）与缓冲节点的构造；离线与在线两阶段的路径规划（多源最短路与最近邻搜索）；以及基于局部姿态空间的相似度判定。

**📊 数据集**

使用了人类动作捕捉数据集（涵盖四种高动态技能）对 Unitree G1 机器人进行训练和测试，此外还在 MuJoCo 与 IsaacGym 上做仿真验证。

**📈 对比分析**

与 ASAP 与 GMT 两个基线相比，Switch 在所有难度级别（易/中/难）的技能切换成功率均达 100%，全身跟踪误差（e.g., E_g‑mpbpe）显著低于对手；在扰动测试中，Switch 能主动重规划并恢复，保持低误差。

**⚠️ 局限性**

局限性主要体现在：① 需要大量人类动作捕捉数据来构造技能图，数据获取成本高；② 目前仅在四种技能范围内验证，难以评估在更多、更复杂技能组合下的泛化能力；③ 在线调度器虽然低延迟，但在极端扰动或硬件极限情况下仍可能出现规划失败。

---

## 313. Learning Ad Hoc Network Dynamics via Graph-Structured World Models

**arXiv ID:** 2604.14811 | [PDF](https://arxiv.org/pdf/2604.14811v1)

**作者:** Can Karacelebi `[一作]` (Middle East Technical University), Ertan Onur `[通讯]` (Middle East Technical University)

**通讯引用:** 1664 | [OpenAlex ID](https://openalex.org/A5030797492)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种图结构的世界模型（G‑RSSM）和基于想象的组合决策训练，用于无线自组网络中的聚类任务；

**💡 创新点**

创新点包括：①将RSSM扩展为图结构，保持每个节点的隐状态并通过跨节点多头注意力捕获节点间耦合；②在世界模型中引入“继续预测器”，在想象轨迹中提前截断，促使策略学习避免导致网络崩溃的动作；③模型具备规模无关性，可在训练时使用小规模网络，零样本迁移到大规模网络；

**🔧 技术方法**

技术手段包括图注意力网络（GATv2）编码、共享权重的多头注意力的Per‑Node GRU、类别型隐状态、变分下界训练、PPO+λ‑返回、世界模型的预测头（位置、能量、邻接、奖励、继续）以及多轮自注意力细化的策略网络；

**📊 数据集**

使用基于物理的仿真生成的离线轨迹，覆盖27种不同网络类型（MANET、VANET、FANET、WSN、战术网络等），节点数量从30到1000，生成了180条轨迹进行世界模型训练；

**📈 对比分析**

与六类基线（Lowest‑ID、WCA、LEACH、HEED、DMAC、DRL‑Cluster）在50个相同种子下进行对比。WM‑Cluster在默认场景（50节点）上：CH变化仅128次（比Lowest‑ID少75%，比DRL‑Cluster少81%），网络寿命175步（比Lowest‑ID高4.7×，比DRL‑Cluster高25×），连通度0.820（略低于近乎1的传统方法），能量公平性Jain指数0.847（最高）。在跨场景零样本评估中，连通度平均0.969，寿命显著优于基线；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实网络部署实验；策略只针对聚类任务，扩展到其他控制任务尚未验证；虽然模型具备规模无关性，但对极大规模网络的实时推理效率未评估；并且缺乏对覆盖保证的理论分析，可能需要进一步的分布式实现细化。

---

## 314. From Boundaries to Semantics: Prompt-Guided Multi-Task Learning for Petrographic Thin-section Segmentation

**arXiv ID:** 2604.14805 | [PDF](https://arxiv.org/pdf/2604.14805v1)

**作者:** Yili Ren `[一作]`, Lei Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种双阶段、基于 SAM 的多任务框架 Petro-SAM，用于在多角度偏振薄片图像上同时完成颗粒边缘分割（GES）与岩相语义分割（LSS）。

**💡 创新点**

创新点在于（1）Merge Block 融合七视角偏振图像以消除消光影响；（2）Adaptation Block 将 SAM 编码器对齐到岩石纹理空间；（3）Refine Block 与 Entropy Block 通过多尺度融合和颜色熵先验精细化边缘；（4）两阶段联合优化，使边缘先验驱动语义分割，实现高质量的双任务输出。

**🔧 技术方法**

主要技术包括 Segment Anything Model（SAM）的 Vision Transformer 编码器、全局注意力融合、跨域特征对齐、颜色熵先验、边缘感知损失以及多任务学习框架。

**📊 数据集**

使用了公开的 CPPID 边缘数据集和作者构建的 1,400 组七视角偏振薄片图像数据集，图像包含精细边缘掩码与四类岩相标签。

**📈 对比分析**

在 CPPID 与自建数据集上实验，Petro‑SAM 的 GES mIoU 提升至 48.2%/44.4%，远超传统边缘检测器（≈20%）与 SAM 基线（≈7%）；在 LSS 任务中 mIoU 达到 86.3%，优于最优语义模型 FCN（≈85.0%）和 Mask2Former（≈66.3%）。

**⚠️ 局限性**

局限性包括对计算资源和显存的高需求、对多视角数据的依赖、以及在极端偏振条件下仍可能出现边缘模糊或类别混淆的问题。

---

## 315. Keep It CALM: Toward Calibration-Free Kilometer-Level SLAM with Visual Geometry Foundation Models via an Assistant Eye

**arXiv ID:** 2604.14795 | [PDF](https://arxiv.org/pdf/2604.14795v1)

**作者:** Tianjun Zhang `[一作]` (Shanghai Jiao Tong University), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9306 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出CAL^2M框架，实现无标定、公里级的SLAM，利用助手眼消除尺度歧义，采用基于视角几何的内参搜索与姿态校正，结合锚点传播与非线性TPS对齐实现全局一致重建。

**💡 创新点**

创新点包括：①助手眼通过固定间距先验实现尺度统一，无需标定；②基于本质矩阵的在线内参搜索与姿态校正模型；③锚点传播与非线性对齐的全局一致映射策略；④框架可与任意VGFMs无缝集成。

**🔧 技术方法**

使用技术：视觉几何基础模型(VGGT/Pi3/MapAnything)、视角几何/本质矩阵分解、姿态图优化(PGO)、锚点传播与Thin Plate Spline(TPS)变形、在线内参搜索、SALAD描述子循环闭合检测。

**📊 数据集**

使用数据集：KITTI Odometry、KITTI-360、Argoverse。

**📈 对比分析**

与传统标定SLAM(ORB-SLAM2、DROID-SLAM)及其他无标定方法(VGGT-Long、VGGT-SLAM等)进行ATE、精度、完整度、Chamfer等指标比较，CAL^2M在KITTI Odometry上取得最优ATE，在KITTI-360上表现出最小尺度漂移与最优定位精度，在Argoverse上获得最佳Chamfer、Accuracy和Completeness，整体性能领先同类无标定方法并接近标定方法。

**⚠️ 局限性**

局限性：依赖固定焦距相机，无法处理连续变焦；与全标定同步立体SLAM相比仍有性能差距；内参搜索在特征稀疏或动态环境下可能失效；对助手眼固定间距的假设限制了在非固定基线的部署。

---

## 316. CogEvolution: A Human-like Generative Educational Agent to Simulate Student's Cognitive Evolution

**arXiv ID:** 2604.14786 | [PDF](https://arxiv.org/pdf/2604.14786v1)

**作者:** Wei Zhang `[一作]` (Central China Normal University), Kezhen Huang `[通讯]` (Central China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出CogEvolution教育代理，模拟学生认知演化以生成更真实的学习行为。

**💡 创新点**

创新点在于将ICAP认知深度感知、基于IRT的结构化记忆检索与进化算法驱动的认知状态更新三大模块结合，实现从静态角色到动态认知流的转变。

**🔧 技术方法**

技术包括ICAP概率感知感知器、IRT驱动的语义+结构相似度检索、以及基于突变-选择-更新的演化算法。

**📊 数据集**

使用自建的CogMath-948数据集，包含学生题目、回答、反思、ICAP标签和误区分类。

**📈 对比分析**

与静态LLM、Gemini思考版和PEERS（BKT+LLM）对比，CogEvolution在AUC、RMSE、错误精准度、学习曲线拟合R²以及行为-认知一致性方面均表现更好，尤其错误精准度提升至76.8%、学习曲线R²达0.92。

**⚠️ 局限性**

局限性包括对单一学科场景的依赖、对大规模记忆检索与进化计算的高成本、以及在更广泛教育任务中的泛化能力尚未充分验证。

---

## 317. Integrating Object Detection, LiDAR-Enhanced Depth Estimation, and Segmentation Models for Railway Environments

**arXiv ID:** 2604.14781 | [PDF](https://arxiv.org/pdf/2604.14781v1)

**作者:** Enrico Francesco Giannico `[一作]` (Scuola Superiore Sant'Anna), Giorgio Buttazzo `[通讯]` (Scuola Superiore Sant'Anna)

**通讯引用:** 13534 | [OpenAlex ID](https://openalex.org/A5024920325)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套可模块化、灵活的多传感器轨道障碍检测框架，集成轨道分割、物体检测与单目深度估计，并通过LiDAR与深度图融合实现绝对距离估计；

**💡 创新点**

创新点在于：①将轨道分割作为ROI预筛选，提升障碍识别精度；②使用稀疏LiDAR深度对单目深度进行线性插值补偿，兼顾稠密度与精度；③引入多目标跟踪实现距离估计的时序平滑；④在合成数据上制定完整评估流程；

**🔧 技术方法**

核心技术包括DDRNet23-Slim用于轨道分割、YOLOv11x（分割版）用于物体检测、MiDaS v3.1（Swin2L-384）用于单目深度估计，以及投影、残差插值、模式/均值深度聚合与Kalman/滑动窗口滤波；

**📊 数据集**

使用RailSem19与OSDaR23进行轨道分割微调，COCO（映射后类别）进行检测微调，SynDRA的“depth split”和“evaluation split”用于深度估计微调与整体系统评估；

**📈 对比分析**

在SynDRA评估集上，系统在多场景下的检测TPR在63%–99%之间，IoU在0.62–0.88；距离估计MAE在0.45–3.9 m（LiDAR模式）与3.5–17 m（单目+LiDAR融合），相比单独使用LiDAR或单目均明显提升；

**⚠️ 局限性**

局限性包括：仅能识别COCO类别，难以检测小/远距离目标；模型训练与评估基于合成数据，真实场景迁移性能待验证；计算量较大，尚未实现实时推理；

---

## 318. Listen, Pause, and Reason: Toward Perception-Grounded Hybrid Reasoning for Audio Understanding

**arXiv ID:** 2604.14806 | [PDF](https://arxiv.org/pdf/2604.14806v1)

**作者:** Jieyi Wang `[一作]` (Shanghai AI Laboratory), Zhongyu Wei `[通讯]` (Fudan University)

**通讯引用:** 5315 | [OpenAlex ID](https://openalex.org/A5011504177)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PAQA数据集和HyPeR混合感知-推理框架，目标是通过层级解耦提升大型音频语言模型的感知与推理可靠性。

**💡 创新点**

创新点在于结合听觉场景分析的分层解耦，配合显式感知轨迹与隐式PAUSE推理机制，并通过GRPO强化学习实现音频根植的推理路径。

**🔧 技术方法**

技术方法包括在PAQA上进行监督微调、引入PAUSE令牌进行无输出隐含推理、使用Lowest Group Confidence阈值触发推理、采用Group Relative Policy Optimization（GRPO）进行强化学习、以及多目标奖励（准确性、一致性、格式、长度）进行策略优化。

**📊 数据集**

使用的数据集包括7,470对的PAQA音频问答、FSD50K与MUSAN混合生成的背景与多说话人样本，以及MMAU、MMAR、MMSU等音频推理基准。

**📈 对比分析**

与GPT‑4o Audio、Gemini 2.5 Flash、Audio‑Flamingo‑3、OmniVinci、Qwen2.5‑Omni等多种大型音频语言模型对比，HyPeR在MMAU平均分达到75.7%，在MMAR、MMSU等测试中与最强模型相当，并在感知 mAP 由 14.7% 提升至 43.6%，WER 降至 0.78%。

**⚠️ 局限性**

局限性包括PAUSE令牌导致的训练与推理延迟、数据集规模与领域覆盖有限、在部分更宽泛的音频‑语言基准上表现不佳，以及需要更高效的隐式推理方法与更全面的基准对比。

---

## 319. A Comparative Study of CNN Optimization Methods for Edge AI: Exploring the Role of Early Exits

**arXiv ID:** 2604.14789 | [PDF](https://arxiv.org/pdf/2604.14789v1)

**作者:** Nekane Fernandez `[一作]` (Ikerlan Technology Research Centre), Julen Arratibel `[通讯]` (Ikerlan Technology Research Centre)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在边缘 AI 上对 CNN 进行静态压缩（剪枝、量化）和动态早期退出机制的统一评估，并探讨它们的组合。

**💡 创新点**

提供了在真实边缘硬件上使用 ONNX Runtime 统一的实验框架，对四种主流模型进行量化、剪枝、早期退出及其组合的对比，揭示不同技术的优势和互补性。

**🔧 技术方法**

采用结构化剪枝、后训练量化（PTQ、DQ）、早期退出分支、两阶段训练、ONNX 导出与分段推理等技术。

**📊 数据集**

使用 ImageNet100（ImageNet 子集）进行训练与评估。

**📈 对比分析**

通过准确率、压缩率、平均推理时间、CPU/GPU/内存利用率等指标，在 Jetson Orin/Nano、Raspberry Pi5 等真实边缘设备上进行对比，结果显示量化在压缩方面表现最佳，早期退出在大模型上提供最高加速，组合进一步提升性能。

**⚠️ 局限性**

仅评估了两种量化方案，未涉及训练感知量化（QAT）；实验仅在 CPU/CUDA backend 进行；早期退出采用两阶段训练，可能不是最优；未评估多 exit 路径的能耗与动态阈值策略等方面的限制。

---

## 320. MirrorBench: Evaluating Self-centric Intelligence in MLLMs by Introducing a Mirror

**arXiv ID:** 2604.14785 | [PDF](https://arxiv.org/pdf/2604.14785v1)

**作者:** Shengyu Guo `[一作]` (Shanghai Ai Lab), Guangtao Zhai `[通讯]` (Shanghai Ai Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MirrorBench，一个基于心理学镜像自我识别（MSR）实验的体感多模态大语言模型自我中心智能评测基准，利用仿真环境和可视化交互实现自我识别任务。

**💡 创新点**

创新点在于将经典心理学实验迁移至AI领域，设计四级逐步递增难度的评估协议，通过提示消融系统化隔离感知、推理、自我表示等认知能力，构建可持续评测框架。

**🔧 技术方法**

采用多模态LLM驱动的仿真平台（Isaac Sim），配合六方向动作空间、链式推理提示与无提示的交互式实验，并用TSR、SIR、FCR、PCR等指标量化性能。

**📊 数据集**

构建了超过5,000个场景资产池，涵盖7种身体配置（4人类+3机器人）、6种手型、6种标记设计，生成多层难度的镜像任务数据集。

**📈 对比分析**

与人类、随机策略以及18款LLM（7专有、11开源）进行对比，结果显示人类最高，专有LLM次之，随机策略高于大多数开源模型；专有和大模型在四级任务中呈单调下降趋势，体现自我中心推理的不足。

**⚠️ 局限性**

当前LLM缺乏鲁棒的自我认知与镜像识别能力，尤其是小型开源模型甚至低于随机策略，表明其在复杂环境中的自我中心智能仍存在根本性局限。

---

## 321. AIM: Asymmetric Information Masking for Visual Question Answering Continual Learning

**arXiv ID:** 2604.14779 | [PDF](https://arxiv.org/pdf/2604.14779v1)

**作者:** Peifeng Zhang `[一作]` (Sun Yat-Sen University), Haohuan Fu `[通讯]` (National Supercomputing Center in Shenzhen)

**通讯引用:** 11233 | [OpenAlex ID](https://openalex.org/A5031545295)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在连续视觉问答（VQA）任务中，现有连续学习方法因处理对称模型而无法应对现代视觉-语言模型（VLM）的不对称结构而导致灾难性遗忘的问题，提出了异步信息遮罩（AIM）方案来实现跨模态的稳定性与可塑性平衡。

**💡 创新点**

核心创新在于：①将可训练参数划分为视觉投影、跨模态对齐和文本推理三大子空间；②采用基于 Fisher 信息的模态特定遮罩阈值，动态冻结关键参数，避免语言解码器对视觉投影的梯度支配；③使用最大聚合代替累加的 Fisher 计算，抑制过度约束。

**🔧 技术方法**

技术手段包括：梯度遮罩（masking）策略、经验回放（episodic memory）、基于 Fisher 信息的参数重要性评估、最大化聚合的 Fisher 计算、视觉-语言联合训练的 VL‑T5 与 LLaVA 预训练模型。

**📊 数据集**

在 VQA v2（语言增量任务）和 GQA（场景增量任务）两个公开基准上进行评估，并在标准和组合泛化两种测试模式下对比。

**📈 对比分析**

与多种基线（EWC、LwF、MAS、ER、DER、VQACL、QUAD）以及联合训练（Joint）比较，AIM 在两组数据集上均取得最高的平均表现（AP）并显著降低平均遗忘（AF），在 VQA v2 标准测试中 AP 为 43.35%（比 QUAD 高 4.10%），AF 仅 1.56%；在 GQA 上 AP 37.51%、AF 5.53%。

**⚠️ 局限性**

局限性包括：对模态阈值的选择仍需经验调优；在无回放（M=0）下仍可缓解遗忘，但性能较低；目前仅针对视觉投影层的保护，未探索对视觉编码器不同层的细粒度控制。

---

## 322. Constraint-based Pre-training: From Structured Constraints to Scalable Model Initialization

**arXiv ID:** 2604.14769 | [PDF](https://arxiv.org/pdf/2604.14769v1)

**作者:** Fu Feng `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**通讯引用:** 19561 | [OpenAlex ID](https://openalex.org/A5018128720)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种约束式预训练范式和对应的 WeiT 方法，用统一的权重模板和轻量化权重标量实现跨尺寸（深度和宽度）模型的可扩展初始化，解决传统预训练固定规模导致的适配瓶颈。

**💡 创新点**

创新点在于：①将模型参数表示为 Kronecker 乘积约束，自动将尺寸无关知识编码进共享模板；②使用模板缩放机制（结构化 dropout）增强模板对宽度变化的鲁棒性；③通过低秩瓶颈实现模板复用，轻量化标量实现尺寸特定的重构，显著降低后续微调成本。

**🔧 技术方法**

采用的技术包括：Kronecker‑based 约束、低秩瓶颈、模板缩放机制、轻量化权重标量、间接参数更新（通过模板与标量梯度优化），并在多任务适配框架下把初始化视为多任务学习。

**📊 数据集**

数据集：预训练使用 ImageNet‑1K；下游评估包括图像分类（ImageNet、Flowers、CUB、Cars、CIFAR‑10/100、Food‑101、iNaturalist‑2019）、图像生成（CelebA、Bedroom、Church、Hubble、MRI、Pokemon）、Embodied Control（UNIMAL 100 训练/100 新 Morphology，Morphology‑Aware Transformer），以及卷积网络 ConvNeXt‑v2 在 ImageNet 上的验证。

**📈 对比分析**

与传统 He‑init、Mimetic、Pruning、Distillation 等方法对比，WeiT 在不同尺寸（深度/宽度）下均取得更高的 Top‑1 Accuracy、FID、奖励等指标，并在全训练阶段保持更快收敛和更优最终性能；相对标量参数少、训练步骤极少（仅数百步），计算开销可忽略。

**⚠️ 局限性**

局限性：①仍需在大规模预训练阶段使用高资源；②在极端尺寸差异（极大/极小模型）或跨域迁移（与预训练任务差异极大的数据）时表现尚未完全验证；③仅在 Transformer/Conv‑based 结构上验证，其他网络如 RNN 或自监督预训练的通用性尚待探索。

---

## 323. Wasserstein Formulation of Reinforcement Learning. An Optimal Transport Perspective on Policy Optimization

**arXiv ID:** 2604.14765 | [PDF](https://arxiv.org/pdf/2604.14765v1)

**作者:** Mathias Dus `[一作]` `[通讯]` (IRMA), Mathias Dus (IRMA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

本文构建了一个将策略视为作用空间概率分布的Wasserstein空间映射的几何框架，并在此空间上定义了Riemannian结构、切空间与测地线，从而通过奥托微积分推导出强化学习目标函数的梯度与Hessian，实现了二阶优化与数值实验。

**💡 创新点**

创新点在于：①在全局RL上下文中正式证明了稳态分布的存在与唯一性，并以此为基础构建状态条件的测地几何；②对策略空间引入Wasserstein梯度流，使用奥托算子完成二阶（Hessian）分析；③提出基于测地流的梯度流公式与势函数对应的Poisson方程；④通过周期性或ergodic逼近将低维解析梯度推广至高维连续控制，兼具理论与可扩展性。

**🔧 技术方法**

所用技术包括：Wasserstein最优传输、Riemannian几何、奥托微积分、Poisson方程求解、梯度流数值化（JKO式或离散化）、自然梯度、Adam优化、神经网络策略与可微世界模型、以及测地线和Hessian的显式计算。

**📊 数据集**

实验使用的环境为三种仿真模拟：①标量随机非线性调节器，②倒立摆（含离散网格与连续控制），③高维耦合振荡器链；未使用公开数据集，全部为自建离散化环境。

**📈 对比分析**

对比方法包括：基于网格的策略迭代（粒子方法）、直接Adam梯度下降、自然梯度优化以及基于学习世界模型的模型预测控制。实验结果显示：①在低维环境中，解析梯度与数值梯度几乎一致且收敛最快；②在高维连续控制中，神经网络+ergodic逼近实现与网格方法相当的成本下降；③与传统Adam相比，自然梯度在收敛速度与最终成本上均略有优势；总体性能表现良好，能在大多数环境中实现快速收敛并逼近最优策略。

**⚠️ 局限性**

局限性包括：①理论对稳态分布与Lipschitz条件要求严格，实际非平稳或高维噪声环境可能不满足；②对高维连续控制的Hessian计算仍昂贵，仅通过ergodic逼近近似；③方法依赖可微环境或可学习的世界模型，若真实系统不可微或模型误差大则梯度失效；④测地线和梯度流的离散化实现对步长与数值稳定性敏感，需要精细调参。

---

## 324. OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism

**arXiv ID:** 2604.14762 | [PDF](https://arxiv.org/pdf/2604.14762v1)

**作者:** Jordan Shipard `[一作]` (SAIVT, QUT), Clinton Fookes `[通讯]` (SAIVT, QUT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种零样本、跨模态的通用类别发现方法OmniGCD；

**💡 创新点**

创新点在于将类别发现任务抽象为模态无关的隐空间，并用一次性训练的Transformer（GCDformer）在测试时对该隐空间进行变换，从而实现不需任何数据集特定微调的零样本GCD；

**🔧 技术方法**

使用预训练的模态特定编码器（如Vision的DINOv2、Text的E5‑Large‑v2、Audio的MERT‑95M、遥感的DOFA‑Base），t‑SNE降维构造GCD隐空间，GCDformer基于GPT‑2架构进行自注意力变换，并采用对比损失训练；

**📊 数据集**

在16个数据集上评估，涵盖视觉（CIFAR‑10/100、ImageNet‑100、CUB‑200、Stanford Cars、FGVC‑Aircraft、Herbarium‑19）、文本（BANKING、StackOverflow、CLINIC）、音频（VocalSet、UrbanSound）和遥感（EuroSAT、So2SAT、RESISC45、UC Merced）；

**📈 对比分析**

与基线k‑means和无微调的GCD方法相比，OmniGCD在所有模态的整体准确率平均提升约6.2~17.9个百分点；在部分数据集上甚至击败当前最优方法；

**⚠️ 局限性**

局限在于仍高度依赖编码器质量，对细粒度类表现不佳时可能需要微调；合成训练数据可能无法完全覆盖真实分布；且目前未实现端到端联合训练，未来可进一步提升性能。

---

## 325. Efficient Search of Implantable Adaptive Cells for Medical Image Segmentation

**arXiv ID:** 2604.14849 | [PDF](https://arxiv.org/pdf/2604.14849v1)

**作者:** Emil Benedykciuk `[一作]` (Maria Curie Sklodowska University), Grzegorz M. Wójcik `[通讯]` (Maria Curie Sklodowska University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对Implantable Adaptive Cells (IAC) 在医学图像分割中的可微搜索过程进行早期稳定性分析，并提出基于Jensen–Shannon散度的在线剪枝策略，使搜索速度显著加快。

**💡 创新点**

利用操作重要性分布在搜索初期即可稳定的特性，设计JS稳定性判据和渐进式剪枝规则，从而将IAC搜索时间压缩到原来的1/3–1/16，同时保持甚至提升分割性能。

**🔧 技术方法**

采用可微NAS（DARTS/PC‑DARTS）框架、Jensen–Shannon散度做稳定性监测、Lottery Ticket Hypothesis启发的剪枝、2D U‑Net/nnU‑Net骨干网络和标准Dice/CE损失等技术。

**📊 数据集**

在四个公开医学图像分割数据集上进行实验：ACDC、BraTS、KiTS 和 AMOS。

**📈 对比分析**

通过与原始200‑epoch IAC搜索、注意力门、Dense Skip、U‑Net++等基线在多种骨干网络和 nnU‑Net 体系中的患者级 Dice 进行对比，IAC‑LTH 在绝大多数场景下与原始 IAC 取得相当或更好成绩，同时搜索耗时下降 3.7–16×。

**⚠️ 局限性**

局限性包括仅在 2D U‑Net 级别验证；JS 阈值和剪枝策略为经验设定，未对 3D 网络或更大搜索空间进行探索；缺乏跨域鲁棒性、校准以及能耗/硬件友好性等评估。

---

## 326. Federated User Behavior Modeling for Privacy-Preserving LLM Recommendation

**arXiv ID:** 2604.14833 | [PDF](https://arxiv.org/pdf/2604.14833v1)

**作者:** Lei Guo `[一作]` (Shandong Normal University), Zhumin Chen `[通讯]` (Shandong University)

**通讯引用:** 5196 | [OpenAlex ID](https://openalex.org/A5050947285)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SF-UBM 框架，用于解决非重叠隐私保护跨域推荐（PPCDR）问题，结合自然语言语义桥接域、联邦学习、Fact‑counter Knowledge Distillation（FKD）以及将预训练的用户偏好与跨域项目映射到 LLM soft‑prompt 空间的方法。

**💡 创新点**

创新点：①利用自然语言的通用性作为隐私保护下的跨域桥梁；②设计 FKD 将域共享的文本知识与域特定的 ID 语义融合；③将传统 CF 模型和跨域语义嵌入投射到 LLM prompt 空间，实现行为与语义空间的对齐；④两层隐私保护机制（仅上传加噪文本嵌入 + 相似替换），在保持高推荐性能的同时降低信息泄露风险。

**🔧 技术方法**

技术栈：SBERT 句子嵌入 + 高斯噪声 + 相似替换；Federated Learning + K‑means++ 语义聚类；SASRec 预训练序列模型；FKD 知识蒸馏（模态对齐）；Soft‑prompt 投影 + OPT‑6.7B LLM 微调。

**📊 数据集**

数据集：Amazon 3 个域（Health‑Beauty、Food‑Kitchen、Books）与 MovieLens，构成 3 个域对（Food‑Kitchen/Kitchen、Books/MovieLens、Health‑Beauty/Beauty）。

**📈 对比分析**

与 ID‑only（SASRec、GRU4Rec）、CDR（PTUPCDR、RecGURU）、ID‑Text（UniSRec、PFCR、FFMSR）、LLM 基线（LLM‑Only、TALLRec、MLP‑LLM、A‑LLMRec）进行对比。SF‑UBM 在 Hit@1 上优于所有 SOTA 基线，提升幅度 10%‑30%，且维持高 Valid 率，验证了其在非重叠 PPCDR 场景下的有效性。

**⚠️ 局限性**

局限性：①主要依赖文本语义，未充分利用图像、音频等多模态信息；②在极短交互序列（<3 条）下性能下降；③需要针对不同域手动调节超参（k、α、β 等）；④隐私保护仅在文本层面，未采用更强的加密或差分隐私方案。

---

## 327. Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization

**arXiv ID:** 2604.14853 | [PDF](https://arxiv.org/pdf/2604.14853v1)

**作者:** Zhiyuan Zhai `[一作]` (Fudan University), Xin Wang `[通讯]` (Fudan University)

**通讯引用:** 84344 | [OpenAlex ID](https://openalex.org/A5100328102)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于拉格朗日松弛的Solve‑then‑Learn框架，用来在有限的推理预算下动态分配大语言模型的计算量；

**💡 创新点**

创新点在于将输入适应性计算分配视为一个受限优化问题，通过双重变量实现每个实例的闭式最优分配，并用仿真学习将此规则转化为轻量级分类器；

**🔧 技术方法**

核心技术包括拉格朗日对偶、二分搜索寻找合适的双重变量、以及梯度提升机（GBM）对oracle标签进行监督学习；

**📊 数据集**

实验使用了 MATH 与 GSM8K 两个数学推理基准，分别针对 DeepSeek‑V3、GPT‑4o‑mini 与 Qwen2.5‑7B 三个大型模型；

**📈 对比分析**

与均匀分配、随机分配、基于提示长度的启发式分配以及oracle上限进行对比，AdaCompute 在所有模型、数据集及预算级别下均超过非oracle基线，最大提升约 12.8% 以上；

**⚠️ 局限性**

局限性包括：预算离散化且需预先离线计算效用表、对实时在线适应和连续预算空间的支持有限、以及对输入特征的依赖仍可能限制适配性。

---

## 328. Intermediate Layers Encode Optimal Biological Representations in Single-Cell Foundation Models

**arXiv ID:** 2604.14838 | [PDF](https://arxiv.org/pdf/2604.14838v1)

**作者:** Vincenzo Yuto Civale `[一作]` (University of Siena), Alberto Magi `[通讯]` (University of Firenze)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

系统评估单细胞基础模型在轨迹推断与干扰响应预测中的各层嵌入

**💡 创新点**

揭示中间层常优于最终层，且最佳层级受任务与细胞状态显著影响

**🔧 技术方法**

层级嵌入提取、Spearman相关性、代表性相似度分析等方法

**📊 数据集**

LARRY血胚系scRNA‑seq 数据集与人类CD4+ T细胞CRISPRi Perturb‑seq 数据集

**📈 对比分析**

通过比较层级嵌入与参考伪时间或DE谱的Spearman相关性，发现轨迹推断最优层在60%深度可提升31%，干扰预测最佳层随细胞活化状态变动达0–96%，甚至第一层在休眠细胞中优于深层

**⚠️ 局限性**

仅考察两种模型与两类任务，未涵盖更多架构或生物学领域，且使用冻结模型，缺乏对更大规模或多任务的验证

---

## 329. NTIRE 2026 Challenge on Video Saliency Prediction: Methods and Results

**arXiv ID:** 2604.14816 | [PDF](https://arxiv.org/pdf/2604.14816v1)

**作者:** Andrey Moskalenko `[一作]`, Jiachen Tu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并组织了 NTIRE 2026 视频显著性预测挑战，构建了 2000 条视频的开放数据集并收集了 5,000+ 评测者的鼠标追踪显著性数据，评估并公布了 7 家参赛队伍的最佳模型。

**💡 创新点**

创新点在于将大规模预训练视频骨干（如 InternVideo2、V-JEPA2）与多专家融合、预测编码理论、层级与双向多尺度融合、音视频跨模态注意力、以及扩散模型等新颖架构结合，以提升显著性预测的时空建模能力。

**🔧 技术方法**

采用的技术包括自监督视频预训练（InternVideo2、V-JEPA2、R(2+1)D、ConvNeXt）、多尺度与层级特征融合、FiLM 与自适应中心偏置、预测编码/误差驱动的显著性、Fokker–Planck 动态建模、超空间扩散网络、以及多模型融合与后处理。

**📊 数据集**

使用的数据集为公开的 2000 条 YouTube 视频（全高清 30fps），其中 1200 条用于训练（含 1M 帧显著性图），800 条用于测试（公开 300 条 + 私有 500 条）。

**📈 对比分析**

方法与基线（中心偏置）相比，排名前列模型在 CC、SIM、AUC‑Judd、NSS 等四项指标上均显著优于基线；例如 iLearn 以 CC 0.837、SIM 0.699、AUC 0.897、NSS 3.44 的综合表现夺得第一。各参赛队伍的参数规模从 2.2M 到 6.5M 变化不大。

**⚠️ 局限性**

主要限制包括：仍依赖鼠标追踪的近似显著性，难以完全替代眼动仪；数据集虽大但多为常规 2D 视频，对 360°/VR 等更复杂视角的泛化性尚待验证；部分模型参数量较大，部署成本较高；评测只涵盖 4 项指标，缺乏对时间一致性或跨域鲁棒性的深入评估。

---

## 330. CoPA: Benchmarking Personalized Question Answering with Data-Informed Cognitive Factors

**arXiv ID:** 2604.14773 | [PDF](https://arxiv.org/pdf/2604.14773v1)

**作者:** Hang Su `[一作]` (East China Normal University), Zhen Liu `[通讯]` (Zhongguancun Academy)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于社区-个体偏好差异(CIPD)的个人化问答评测基准CoPA，并通过LLM自动挖掘并提炼出六个可解释的认知因子作为评估维度。

**💡 创新点**

创新点在于：①利用CIPD现象从大规模社区问答数据中自动提取用户个性化决策的内在认知因子；②提出因素驱动的评测框架CoPA，提供细粒度、可解释的多维度评估指标；③在多基准上验证因子评估相较传统方法的显著提升。

**🔧 技术方法**

使用多模型LLM（GPT‑5、Gemini‑2.5‑Pro、Qwen3‑Max）进行因子提取与用户画像构建；采用结构化提示与集成策略；用LLM‑Judge进行评估；基于Fast‑Thinking模式的Qwen3和GPT‑4o‑mini做生成实验。

**📊 数据集**

数据集主要包括：StackExchange（626,786问答、15,963用户）用于因子挖掘；CoPA（1,985用户）用于评测；外部验证集UPGC‑QA、LaMP‑QA。

**📈 对比分析**

与传统BLEU/ROUGE、Llama‑as‑Judge、Jaccard/Inclusion等方法对比，因子评估准确率提升至约55%（相较于Direct/CoT的32%），误差率降至18%；在CoPA、UPGC‑QA和LaMP‑QA三大基准上，基于因子的个性化模型平均提升10–15%。

**⚠️ 局限性**

局限性包括：①因子从单一社区数据挖掘，可能存在冗余与不完全独立；②过度依赖LLM可能引入模型偏见与循环；③人机评估一致性有限；④尚未验证在其他多样化应用场景（如对话代理、推荐系统）中的泛化能力。

---

## 331. Exploiting Correlations in Federated Learning: Opportunities and Practical Limitations

**arXiv ID:** 2604.14751 | [PDF](https://arxiv.org/pdf/2604.14751v1)

**作者:** Adrian Edin `[一作]` (Linköping University), Zheng Chen `[通讯]` (Linköping University)

**通讯引用:** 55137 | [OpenAlex ID](https://openalex.org/A5100457678)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并分类联邦学习中的结构、时序与空间三种相关性，并设计了基于相关性强度自适应切换的压缩算法 AdaSVDFed 与 PCAFed。

**💡 创新点**

提出了统一的相关性框架，利用动态相关性评估在不同压缩模式间切换，从而显著提升压缩效率并降低通信成本。

**🔧 技术方法**

采用 SVD、PCA、预测编码、稀疏化、低秩逼近、量化等技术实现压缩与自适应切换。

**📊 数据集**

实验使用 w8a 线性分类、MNIST（LeNet）与 CIFAR-10（ResNet18）三组数据集。

**📈 对比分析**

在 IID 与非 IID 场景下与无压缩基线及固定 SVDFed 进行对比，AdaSVDFed 与 PCAFed 在保持模型精度的前提下，总传输量减少约 30%–50%，且收敛速度与无压缩差距不大。

**⚠️ 局限性**

局限性包括相关性评估与阈值需人工调参、算法实现复杂度高、服务器端计算与内存开销显著，以及在更大规模模型和更分散网络环境下的可扩展性待进一步验证。

---

## 332. Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement

**arXiv ID:** 2604.14749 | [PDF](https://arxiv.org/pdf/2604.14749v1)

**作者:** Midan Shim `[一作]` (Yonsei University), Kyong-Ho Lee `[通讯]` (Yonsei University)

**通讯引用:** 12332 | [OpenAlex ID](https://openalex.org/A5100763487)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了包含负约束问题的KGQA任务NEST KGQA，并构建了对应的数据集NestKGQA与Python格式逻辑表述PyLF。

**💡 创新点**

创新点在于：①引入负约束的KGQA任务与专门的PyLF形式；②设计CUCKOO框架，包括约束感知草稿生成、基于模式的语义匹配和自导修正，以显著提升多约束负约束问答性能。

**🔧 技术方法**

利用LLM（如GPT‑3.5‑turbo、GPT‑4o‑mini）进行示例式学习，使用PyLF作为逻辑形式，并实现schema‑guided semantic matching和self‑directed refinement，最终与KB‑BINDER/KB‑Coder等方法比较。

**📊 数据集**

主要使用GrailQA、GraphQ、WebQSP以及新构造的NestKGQA四个数据集，其中NestKGQA为负约束问答数据集。

**📈 对比分析**

与基线相比，CUCKOO在无监督训练下在GrailQA、GraphQ、NestKGQA等数据集上取得了EM/F1领先或次优的结果，特别在多约束和负约束问题上显著优于KB‑Coder和传统方法；虽然推理时间略长，但内存占用更低。

**⚠️ 局限性**

局限性包括：假设闭世界且需要完整模式；NestKGQA规模有限；对完整模式的依赖可能在隐式或不完整模式场景下受限；以及对LLM的依赖较大。

---

## 333. STEP-Parts: Geometric Partitioning of Boundary Representations for Large-Scale CAD Processing

**arXiv ID:** 2604.14927 | [PDF](https://arxiv.org/pdf/2604.14927v1)

**作者:** Shen Fan `[一作]` (New Jersey Institute of Technology), Przemyslaw Musialski `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 3030 | [OpenAlex ID](https://openalex.org/A5065767002)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 STEP-Parts 工具链，直接将 STEP B‑Rep 中的拓扑与分析表面信息转换为实例级几何标签，并将其映射到三角网格上供学习与评估使用。

**💡 创新点**

创新点在于：① 用 B‑Rep 的拓扑相邻图和相同原语类型 + 近切角度判定实现确定性、可重复的实例分割；② 将分割结果通过源面索引投射到任意三角网格，实现与传统网格方法无缝对接；③ 在大型 CAD 数据集上完成近 18 万模型的批量处理并公开代码与标签。

**🔧 技术方法**

核心技术包括：B‑Rep 面原语标记、面间双面角度计算、阈值合并、广度优先洪泛求解、三角网格源面对应、最优标签匹配（Hungarian）、多指标评估（mIoU、边界精度、对齐一致性）等。

**📊 数据集**

使用数据集：ABC/DeepCAD 子集（约 18 万 STEP 模型）进行大规模处理，240 个模型做验证与对比实验，另外 25 个模型做隐式重建–分割下游实验，2500 个模型做 PTv3 下游实验。

**📈 对比分析**

与基于网格的 PartField 方法对比：STEP-Parts 在自洽性（0.99）和 mIoU（0.10）等指标上明显优于 PartField（0.94、0.10）；在下游任务中，STEP-Parts 监督的隐式 SDF‑分割网络和 PTv3 点云模型在 mIoU 上分别提升约 0.1（0.60 vs 0.51）和在各复杂度 bin 上均高于 PartField，特别是高复杂度模型。

**⚠️ 局限性**

限制：① 不是语义或功能分割，仅反映几何原语与拓扑一致性；② 需要三角网格投影，极粗网格可能导致薄特征丢失；③ 小组件阈值（τ_min）会消除细小区域；④ 阈值 θ 在低角度区间稳定，但在不同 CAD 语料库需重新校准。

---

## 334. Efficient Fuzzy Private Set Intersection from Secret-shared OPRF

**arXiv ID:** 2604.14909 | [PDF](https://arxiv.org/pdf/2604.14909v1)

**作者:** Xinpeng Yang `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 4833 | [OpenAlex ID](https://openalex.org/A5101591101)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一套基于对称加密原语的高效模糊私有集合交互（FPSI）协议，支持一般的 L_p（1≤p≤∞）距离；

**💡 创新点**

核心创新点包括：①引入共享输出的可编程伪随机函数（so‑OPPRF）作为新型构件；②构建模块化的模糊映射框架；③利用前缀技术将阈值 δ 的复杂度从线性降到对数；④在所有步骤中主要使用轻量级对称运算而非昂贵的同态加密；

**🔧 技术方法**

技术实现依赖于 OKVS、so‑OPPRF、si‑OPRF、OT（包括 silent OT）以及对称加密 PRF；对距离计算进一步使用 B2A、私有区间测试等辅助协议；

**📊 数据集**

实验使用随机生成的高维向量集合（m=n∈{2^8,2^12,2^16}，维度 d∈{4,8,16}，阈值 δ∈{16,32}）作为数据集；

**📈 对比分析**

与 Gao 等（ASIACRYPT'24）和 Dang 等（CCS'25）等现有线性复杂度方案对比，实验结果显示：在所有设置下运行时间提升 12–145 倍，通信量缩减 3–8 倍，特别是大规模输入时性能优势更为显著；

**⚠️ 局限性**

局限性：①协议仅在半诚实模型下安全；②依赖输入分布的“分离投影”假设，实际数据中阈值 δ 的选取仍具挑战；③缺乏针对恶意攻击的安全保证，若需恶意安全需引入更复杂的验证机制。

---

## 335. Multi-User mmWave Beam and Rate Adaptation via Combinatorial Satisficing Bandits

**arXiv ID:** 2604.14908 | [PDF](https://arxiv.org/pdf/2604.14908v1)

**作者:** Emre Özyıldırım `[一作]` (Bilkent University), Cem Tekin `[通讯]` (Bilkent University)

**通讯引用:** 1533 | [OpenAlex ID](https://openalex.org/A5053015746)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于满意度（satisficing）目标的组合半带（combinatorial semi‑bandit）算法，用于在多基站多用户毫米波MISO系统中，仅通过ACK/NACK反馈实现光束与速率的自适应分配。

**💡 创新点**

创新点在于（1）引入阈值满足度目标，将传统最大化目标改为满足给定吞吐量阈值；（2）设计了SAT‑CTS策略，结合保守置信下界（LCB）与后验采样（Thompson sampling）并以门控方式控制探索；（3）首次给出可在有限时间内收敛的满意度无上界误差（horizon‑free）与非可实现目标下的O((log T)²)标准后悔上界。

**🔧 技术方法**

技术主要包括组合半带问题建模、下界与均值指数的计算、Hungarian算法求解无共享光束的匹配、Beta后验采样以及几何增量（doubling）CTS阶段的理论分析。

**📊 数据集**

实验使用 DeepMIMO 仿真生成的城市环境时变多径信道，模拟 3 台 BS、64 天线、120 方向性波束、15/50/100 UE 等场景。

**📈 对比分析**

与传统 CTS、CUCB 以及工作坊版 SAT‑CTS-W 进行对比，结果显示 SAT‑CTS 在可实现阈值下的累计满意度后悔几乎不随时间增长、标准后悔与 CTS 相当；在不可实现阈值时退化为 CTS；公平性（Jain 指数）和对数效用也优于基线。

**⚠️ 局限性**

局限在于（1）对角相互干扰假设（无共享光束）可能不适用于高度重叠的波束；（2）理论分析基于独立同分布的信道模型，实际时变信道的非平稳性仍待研究；（3）算法对大规模用户/波束组合时的计算复杂度仍然随 M²BKR 规模增长。

---

## 336. ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints

**arXiv ID:** 2604.14902 | [PDF](https://arxiv.org/pdf/2604.14902v1)

**作者:** Pei-An Chen `[一作]` (National Taiwan University), Winston Hsu `[通讯]` (National Taiwan University)

**通讯引用:** 6368 | [OpenAlex ID](https://openalex.org/A5043898632)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DynAfford 基准，用于评估嵌入式代理在动态物体可用性和隐式前置条件下的执行能力，并提出 ADAPT 模块，使代理能够在决策时推理动作可执行性并在必要时推迟或替代动作。

**💡 创新点**

创新点：①将动态可用性（占用、脏污、使用状态）作为隐式前置条件融入任务；②设计 ADAPT 作为统一的决策时推理机制，分阶段进行可用性推理和应用性解析；③通过 LoRA 微调 VLM 与多模态上下文增强实现对可用性的精准判断。

**🔧 技术方法**

技术：LoRA 微调的 LLaVA‑1.5‑7B 视觉‑语言模型、跨模态上下文学习、基于 LLM 的动作解析、与现有规划器（FILM、CAPEAM）的无缝集成。

**📊 数据集**

数据集：DynAfford（基于 ALFRED 的 2,628 条演示、10,106 条自然语言注解，57 个 AI2‑THOR 2.0 场景，包含厨房与浴室），覆盖占用、使用、脏污等三类可用性状态。

**📈 对比分析**

对比：在 DynAfford 动态设置下，加入 ADAPT 后 FILM 成功率提升约 73.2%，目标完成率提升 34.7%；CAPEAM 同样提升 17%/8.8% 的成功率；与 GPT‑4o 作为可用性推理后端相比，LoRA 微调模型表现更好。整体上，ADAPT 在动态可用性任务中显著提升性能，远超原始规划器及 LLM‑Planner/SayCan 等基线。

**⚠️ 局限性**

局限：单视角观测下对遮挡和视角歧义敏感；未覆盖真实物理不确定性、安全约束和人机交互；动态可用性仅限占用、使用、脏污三类，缺乏更复杂的几何/结构约束。

---

## 337. Reasoning Dynamics and the Limits of Monitoring Modality Reliance in Vision-Language Models

**arXiv ID:** 2604.14888 | [PDF](https://arxiv.org/pdf/2604.14888v1)

**作者:** Danae Sánchez Villegas `[一作]` (University of Copenhagen), Desmond Elliott `[通讯]` (University of Copenhagen)

**通讯引用:** 3532 | [OpenAlex ID](https://openalex.org/A5010165733)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

分析了18种视觉语言模型在Chain-of-Thought推理过程中的信心轨迹、错误修正行为与文本提示对视觉推理的干扰。

**💡 创新点**

揭示了模型往往在早期就锁定答案、推理训练提升纠错能力但受模态条件限制，并提出了通过干预式监测框架评估CoT可监测性的创新方法。

**🔧 技术方法**

采用信心轨迹跟踪、截断实验、净增益度量、干预式监测框架和G²_mean监测指标进行定量分析。

**📊 数据集**

使用MathVerse、ScienceQA、PhyX等多模态推理基准及其文本/视觉占比不同的变体。

**📈 对比分析**

与指令微调模型对比，推理训练模型在文本主导条件下显著提升，视觉主导条件下提升有限；长CoT的监测指标表明文本干扰难以被检出。

**⚠️ 局限性**

局限在于仅针对多选任务、未检验更复杂任务、模型规模与提示模板对结果影响不完全分离，且对非英语文本的适用性未知。

---

## 338. xFODE: An Explainable Fuzzy Additive ODE Framework for System Identification

**arXiv ID:** 2604.14883 | [PDF](https://arxiv.org/pdf/2604.14883v1)

**作者:** Ertugrul Kececi `[一作]`, Tufan Kumbasar `[通讯]` (Istanbul Technical University)

**通讯引用:** 2440 | [OpenAlex ID](https://openalex.org/A5010725194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出可解释的深度学习系统辨识框架xFODE，利用增量式状态和可解释的加性模糊模型进行系统动力学建模

**💡 创新点**

通过构造增量状态、采用单输入加性模糊映射以及引入结构化的分区策略，显著提升了模型的输入级解释性和规则空间的可读性

**🔧 技术方法**

结合模糊增性模型（FAM）、分区策略（PS1‑PS3）、深度学习参数化学习与全程梯度优化训练

**📊 数据集**

在五个公开SysID基准数据集（Two‑Tank、Hair Dryer、MR Damper、Steam Engine、EV Battery）上进行验证

**📈 对比分析**

与NODE、FODE、AFODE及NLARX等模型对比，xFODE在保持与NODE/FODE相近或更优的预测精度的同时，参数量更少，规则更易解释

**⚠️ 局限性**

当前规则后项仍不够直观，后续工作计划进一步解析后项以提升整体可解释性

---

## 339. SOLIS: Physics-Informed Learning of Interpretable Neural Surrogates for Nonlinear Systems

**arXiv ID:** 2604.14879 | [PDF](https://arxiv.org/pdf/2604.14879v1)

**作者:** Murat Furkan Mansur `[一作]` (Istanbul Technical University), Tufan Kumbasar `[通讯]` (Istanbul Technical University)

**通讯引用:** 2440 | [OpenAlex ID](https://openalex.org/A5010725194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 SOLIS 框架，用于通过物理信息神经网络识别可解释的二阶 Surrogate 模型。

**💡 创新点**

将系统识别拆解为轨迹重构与状态条件参数网络两阶段学习，并引入循环课程与局部物理提示，解决逆 PINN 的可识别性与优化崩溃问题。

**🔧 技术方法**

物理信息神经网络(IPINN)、状态条件二阶 surrogate (Quasi‑LPV)、两网络结构（Solution Network + Parameter Network）、循环课程学习、滑动窗口岭回归提示、FiLM 条件、随机 Fourier 特征、Mixture‑of‑Experts 可选。

**📊 数据集**

Duffing、Van der Pol 经典模拟系统以及两罐液位实验真实数据。

**📈 对比分析**

与 IPINN、IPINN‑M、MATLAB 传递函数等基线比较。SOLIS 在轨迹重构准确率、相位场相似度以及预测回放精度上均优于基线，最高达 98%+ 的重构准确率与 90%+ 的测试回放准确率。

**⚠️ 局限性**

目前仅在低维标定系统上验证，尚未扩展到高维多输入多输出系统，且依赖稠密测量点进行梯度估计，噪声敏感。

---

## 340. GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation

**arXiv ID:** 2604.14878 | [PDF](https://arxiv.org/pdf/2604.14878v1)

**作者:** Yanyan Zou `[一作]` (JD.com), Shengjie Li `[通讯]` (JD.com)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为 GenRec 的单解码器生成式检索推荐框架，结合页级下一词预测、Token Merger 压缩多 token 语义 ID 以及 GRPO‑SR 强化学习对齐用户偏好。

**💡 创新点**

创新点：①页级监督消除传统点级训练的一对多歧义；②前缀阶段使用线性 Token Merger 压缩输入长度；③GRPO‑SR 结合混合奖励和负对数似然正则，提升鲁棒性并抑制奖励劫持；④在京东 App 进行大规模真实部署。

**🔧 技术方法**

使用技术包括：decoder‑only Transformer（Qwen2.5 系列）、Semantic ID（RQ‑K‑means）多模态编码、线性 Token Merger、页级下一词预测（PW‑NTP）、GRPO‑SR 强化学习、混合奖励（稠密模型+门控）和 NLL 正则。

**📊 数据集**

训练数据来自京东 App 用户行为日志，约 5.6 亿条交互序列，覆盖一月；测试取最后一天的记录。

**📈 对比分析**

与传统方法 BERT4Rec、SASRec 以及生成式方法 TIGER、LC‑Rec 对比，GenRec 在 HR@1/HR@10/N@10/HR@50 等指标显著提升，Hallucination Rate 从 15.46% 降至 4.96%；RL 对齐后 HR@1 提升 18% 并进一步降低幻觉率；上线 A/B 测试提升点击率 9.5%、交易率 8.7%。

**⚠️ 局限性**

局限性：①训练成本高，需多 GPU 分布式训练；②RL 对齐过程复杂，调参困难；③方法主要在大规模业务场景验证，缺少对冷启动、多语言适配等场景的评估；④对模型可解释性与公平性等问题未作深入探讨。

---

## 341. 4D Radar Gaussian Modeling and Scan Matching with RCS

**arXiv ID:** 2604.14868 | [PDF](https://arxiv.org/pdf/2604.14868v1)

**作者:** Fernando Amodeo `[一作]` (Universidad Pablo de Olavide), Fernando Caballero `[通讯]` (Universidad Pablo de Olavide)

**通讯引用:** 3781 | [OpenAlex ID](https://openalex.org/A5040477311)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在4D毫米波雷达点云中加入雷达截面(RCS)信息，构建基于3D高斯分布的RCS模型并用于扫描匹配。

**💡 创新点**

创新在于将Spherical Harmonics用于表面RCS视角依赖建模，并将RCS信息融入3D高斯分布，仅在旋转优化中使用RCS成本，提升定位精度。

**🔧 技术方法**

使用Spherical Harmonics、线性最小二乘拟合、Cauchy损失函数、Gauss-Newton优化以及RCS归一化处理。

**📊 数据集**

使用Snail-Radar数据集（ARS548雷达扫描），构建Gaussian-RCS模型并进行实验。

**📈 对比分析**

与仅使用几何成本的基线对比，调节权重w_rcs后发现w_rcs=0.25时绝对位置误差和旋转误差最小，表明加入RCS可提升定位精度。

**⚠️ 局限性**

局限在于RCS仅用于旋转优化，无法单独使用；RCS数据噪声与尺度变化需归一化；实验仅在单一数据集上验证，缺乏更广泛场景评估。

---

## 342. XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics

**arXiv ID:** 2604.14934 | [PDF](https://arxiv.org/pdf/2604.14934v1)

**作者:** Jingxuan Liu `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1635 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并构建了 XQ-MEval 数据集，利用半自动化流程在高质量翻译上注入 MQM 语义错误，生成可控质量的伪翻译三元组，进而评估多语言自动评估指标的跨语言评分偏差，并提出语言特定全局归一化（LGN）来校正这一偏差。

**💡 创新点**

①首个提供并行质量的跨语言评估基准；②通过 GPT‑4o 与人工筛选的半自动化错误注入实现可控质量伪翻译；③首次量化并验证跨语言评分偏差；④提出 LGN 归一化策略显著提升多语言指标与人类 MQM 的一致性。

**🔧 技术方法**

使用 GPT‑4o 自动注入 MQM 语义错误；人工双重筛选确保错误质量；错误合并生成多级质量伪翻译；三元组评估框架；Kendall‑τ、方差系数（CV）等统计指标；z‑score 归一化（LGN）。

**📊 数据集**

以 Flores 多语言翻译数据集为基准，构造九种语言方向（中、日、老、越、印、法、西、僧、德）翻译的并行质量数据集 XQ-MEval。

**📈 对比分析**

对九种自动评估指标（BLEURT、COMET、xCOMET、MX‑reg、KIWI22、KIWI23、MX‑qe 等）在系统级与三元组级分别计算 Kendall‑τ 相关性。平均策略与人类 MQM 存在显著不一致；应用 LGN 后相关性均有所提升，提升幅度虽有限但统计显著。

**⚠️ 局限性**

仅覆盖九种语言，未扩展到更多语言；半自动流程仍需人工筛选，工作量较大；仅关注四种纯语义错误类型，未考虑其他错误种类；生成的伪系统与真实翻译系统可能存在差异，未评估不同错误类型对指标敏感性的跨语言差异。

---

## 343. Generative Data Augmentation for Skeleton Action Recognition

**arXiv ID:** 2604.14933 | [PDF](https://arxiv.org/pdf/2604.14933v1)

**作者:** Xu Dong `[一作]` (University of Surrey), Andrew Gilbert `[通讯]` (University of Surrey)

**通讯引用:** 12718 | [OpenAlex ID](https://openalex.org/A5038807223)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于条件扩散模型的骨架动作数据增强框架，利用Transformer编码器-解码器与生成细化模块，可在少量标签数据下合成高保真、多样化的3D骨架序列，从而提升动作识别模型在低样本场景下的性能。

**💡 创新点**

创新点包括：①将动作标签作为条件引导的扩散生成框架；②采用Transformer前缀嵌入编码器解码器实现时空语义捕捉；③加入生成细化模块（GRM）和采样时的Dropout，以平衡生成的保真度与多样性；④一次性训练后即可大规模合成，且无需多阶段训练；⑤在极少样本下实现接近全数据水平的识别准确率。

**🔧 技术方法**

核心技术：条件扩散模型、Transformer编码器‑解码器、动作标签前缀嵌入、分类引导（classification loss）、生成细化模块（GRM）、采样时Dropout、FID/KID/多样性评估、t‑SNE可视化。

**📊 数据集**

实验数据集：HumanAct12（34类动作）和Refined NTU‑RGBD（NTU‑VIBE，13类动作）。

**📈 对比分析**

对比方法：STGCN++、MSG3D、CTRGCN、BlockGCN 等骨架识别后端；在 10%–100% 训练数据比例下，使用 5× 合成样本进行下游训练。结果显示：在 HumanAct12 上提升 4–7% 识别准确率；在 NTU‑VIBE 上提升 0.8–3.7%；尤其在少样本场景（≤25%）性能提升最显著。生成质量指标 FID/KID 等优于 MDM、T2M‑GPT，且多样性最高。

**⚠️ 局限性**

局限性：对稀有或标签模糊的动作生成质量下降；在极端标签不平衡下可能出现过拟合；未加入关节角度约束或时间平滑约束；未来需自适应超参调节、扩展到多人交互与更长序列。

---

## 344. Governing Reflective Human-AI Collaboration: A Framework for Epistemic Scaffolding and Traceable Reasoning

**arXiv ID:** 2604.14898 | [PDF](https://arxiv.org/pdf/2604.14898v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 345. Improving Sparse Autoencoder with Dynamic Attention

**arXiv ID:** 2604.14925 | [PDF](https://arxiv.org/pdf/2604.14925v1)

**作者:** Dongsheng Wang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 21607 | [OpenAlex ID](https://openalex.org/A5100684575)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于 transformer 的稀疏自编码器（SAE），使用 sparsemax 作为注意力机制，动态决定稀疏度并实现概念学习与重构。

**💡 创新点**

创新点包括：1) transformer 结构共享概念向量连接编码器与解码器；2) 用 sparsemax 替代 softmax，自动估计稀疏度，无需额外正则或手工设定 K；3) 在不增加额外超参数的前提下，提升概念学习与重构质量。

**🔧 技术方法**

使用技术包括：transformer 交叉注意力、sparsemax 函数、CLIP 与 GPT‑2 预训练模型、仅重构损失训练。

**📊 数据集**

实验数据集：视觉方面用 ImageNet 训练、11 个零样本分类数据集（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、EuroSAT、UCF101、DTD、SUN397）；文本方面用 OpenWebText 与 WikiText‑103。

**📈 对比分析**

与 ReLU、TopK、BatchTopK 等基线进行对比；Sparsemax SAE 在零样本图像分类上平均准确率最高，在文本重构的 NMSE 与 CE 降低方面均优于其他方法。

**⚠️ 局限性**

局限性：依赖预训练模型的特征表达，缺乏跨任务泛化评估；对概念维度敏感，计算与内存成本相对较高。

---

## 346. Support Size of $\varepsilon$-Capacity-Achieving Inputs for the Amplitude-Constrained AWGN Channel

**arXiv ID:** 2604.14915 | [PDF](https://arxiv.org/pdf/2604.14915v1)

**作者:** Luca Barletta `[一作]` (Politecnico di Milano), Alex Dytso `[通讯]` (Qualcomm Flarion Technology, Inc.)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了受幅度约束的加性白高斯噪声（AWGN）信道，关注近最优输入分布的最小支持大小，以实现容量的ε间隙。

**💡 创新点**

引入了K_ε(A)的概念，定义为在[-A,A]上支持的离散输入中，实现互信息在ε范围内达到容量所需的最小支持大小。该方法显著简化了问题，并在不同的ε范围内提供了清晰的特征描述。

**🔧 技术方法**

结合了近似理论界限和信息论的熵控制，使用χ²散度以及与圆上均匀分布的近似相关的包裹论证。

**📊 数据集**

未具体提及使用的数据集，但研究对象为受幅度约束的AWGN信道。

**📈 对比分析**

通过对比不同的ε范围，得出K_ε(A)在多项式衰减情况下为Θ(A√(log A))，而在指数衰减情况下，支持大小上限为A^3/2，显示出不同的缩放规律。

**⚠️ 局限性**

限制在于对确切容量实现输入的支持大小仍然不清楚，且在指数衰减情况下的界限仍需进一步收紧。

---

## 347. Comparison of Modern Multilingual Text Embedding Techniques for Hate Speech Detection Task

**arXiv ID:** 2604.14907 | [PDF](https://arxiv.org/pdf/2604.14907v1)

**作者:** Evaldas Vaiciukynas `[一作]`, Rimantas Butleris `[通讯]` (Kaunas University of Technology)

**通讯引用:** 679 | [OpenAlex ID](https://openalex.org/A5029736446)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多语言句子嵌入模型在立陶宛语、俄语、英语仇恨言论检测中的表现，并构建了立陶宛语 LtHate 数据集，统一实验框架对比六种现代多语言嵌入模型与一类 HBOS 异常检测和二类 CatBoost 监督分类在原始及 PCA 压缩后性能。

**💡 创新点**

①首次提出 LtHate 低资源语言仇恨言论语料库；②构建统一的实验流程，对六种现代多语言嵌入模型在三种语言上进行系统对比；③评估 PCA 压缩到 64 维对检测性能的影响；④给出实用模型与嵌入选择建议。

**🔧 技术方法**

使用 SentenceTransformer 生成多语言句子嵌入，PCA 做降维，HBOS 作为一类异常检测，CatBoost 作为二类梯度提升分类器，10 折交叉验证，评估指标包括 Accuracy、Kappa、AUC‑ROC、AUC‑PR。

**📊 数据集**

LtHate（立陶宛语 12k 评论），RuToxic（俄语 163k 评论），EnSuperset（英语 360k 评论）。

**📈 对比分析**

在统一的 Python pipeline 下，分别对每种嵌入模型进行 1c HBOS 与 2c CatBoost 训练，并在原始与 PCA‑64 维两种特征上比较。结果显示：二类监督模型显著优于一类异常检测；PCA 对 2c 模型几乎无损失；最佳配置为 LtHate+Jina+2c（Acc 80.96%，AUC‑ROC 0.887），RuToxic+E5+2c（Acc 92.19%，AUC‑ROC 0.978），EnSuperset+E5+2c+PCA（Acc 76.95%，AUC‑ROC 0.855）。

**⚠️ 局限性**

仅使用未微调的预训练嵌入；仅进行二分类，未利用标签细粒度；仅文本特征，未加入多模态或对话上下文；未评估任务特定微调、零样本或少样本学习；对极低资源语言的进一步改进空间仍大。

---

## 348. Segment-Level Coherence for Robust Harmful Intent Probing in LLMs

**arXiv ID:** 2604.14865 | [PDF](https://arxiv.org/pdf/2604.14865v1)

**作者:** Xuanli He `[一作]` (University College London), Jerry Wei `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大语言模型在CBRN（化学、生物、核、放射）领域的实时流式探测，提出了一种基于多窗口分布式证据聚合的安全探测框架，避免单词级别的误报；

**💡 创新点**

创新点在于用Top‑K支持窗口聚合与段内方差正则（SegVar）强制探测器在多窗口内保持一致的高置信度，从而显著降低噪声峰值对判断的影响；

**🔧 技术方法**

技术包括线性探测器、滑动窗口平均（SWiM）、Top‑K聚合、段内方差正则、指数移动平均后处理以及对注意力、MLP和残差流等内部激活的探测；

**📊 数据集**

使用CBRN专用的数据集（RT&WC、Bio-Conv、化学攻击合成数据）、通用高风险数据集（如Chempile、Ether0、GPQA等）以及多种字符级加密编码数据进行评估；

**📈 对比分析**

与均值、Softmax、注意力、RMAttn、SWiM等基线以及LLM分类器进行对比，结果显示在1% FPR下TPR提升35.55%，AUROC和log‑space‑AUROC均有显著提升，且在大型模型上参数量仅为LoRA分类器的1/344；

**⚠️ 局限性**

局限包括：对探测器友好的攻击者可能通过插入“干扰”信息来绕过分布式聚合；在专业术语密集的长对话中仍可能出现误报；对跨源提示注入等更复杂攻击场景尚未充分验证。

---

## 349. Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-CodeX

**arXiv ID:** 2604.14858 | [PDF](https://arxiv.org/pdf/2604.14858v1)

**作者:** Zhonghao Yang `[一作]` (Shanghai Artificial Intelligence Laboratory), Dongrui Liu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 200 | [OpenAlex ID](https://openalex.org/A5020653216)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ATBench 在 OpenClaw 与 OpenAI Codex/Runtime 两种新 agent 执行环境下的定制化扩展——ATBench‑Claw 与 ATBench‑CodeX，并为每种环境构建了相应的安全分类体系与数据生成流程；

**💡 创新点**

核心创新在于通过对 ATBench 三维安全分类体系进行场景特定的自定义（新增类别或强化已有类别），使原有的数据生成引擎能够在保持不变的前提下，快速适配不同执行环境，从而实现 benchmark 的可扩展性与灵活性；

**🔧 技术方法**

采用了安全分类体系、基于风险采样的多源工具组合、规划器合成的轨迹生成引擎、以及多模型安全诊断评估（Guard、Instruct、AgentDoG 等）等技术；

**📊 数据集**

使用了公开发布的 ATBench‑Claw（针对 OpenClaw）和 ATBench‑CodeX（针对 Codex）两个数据集，每个数据集包含安全/不安全轨迹以及细粒度诊断标签；

**📈 对比分析**

通过统一的二分类安全评估（安全/不安全），计算准确率、F1 与召回率，并按细粒度标签进行诊断切片；实验显示 AgentDoG‑Qwen3‑4B 在两组数据上均表现最优，而在 CodeX 上整体性能下降，验证了该环境更具挑战性；

**⚠️ 局限性**

局限性包括：评估仅限于二分类安全性，未深入探讨模型对具体风险的处理；数据集覆盖范围可能不足，未涵盖所有工具/技能组合；方法依赖人工制定的安全分类体系；未评估对攻击者的鲁棒性。

---

## 350. RaTA-Tool: Retrieval-based Tool Selection with Multimodal Large Language Models

**arXiv ID:** 2604.14951 | [PDF](https://arxiv.org/pdf/2604.14951v1)

**作者:** Gabriele Mattioli `[一作]` (University of Modena and Reggio Emilia), Rita Cucchiara `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 20019 | [OpenAlex ID](https://openalex.org/A5030948871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一种基于检索的开世界多模态工具选择框架，利用多模态大语言模型先将用户查询转化为结构化任务描述，再通过语义检索匹配工具。

**💡 创新点**

创新点在于：①以开放式检索方式取代闭集分类，实现对未见工具的泛化；②使用标准化 JSON 描述工具属性；③在生成任务描述后加入基于 DPO 的偏好优化。

**🔧 技术方法**

核心技术包括多模态大语言模型（Qwen2.5‑Omni）、嵌入检索（Qwen3‑Embedding 或 Contriever）、SFT+LoRA 以及直接偏好优化（DPO）等。

**📊 数据集**

采用改造后的 ToolMMBench，创建了首个开放式多模态工具使用基准，工具描述来源于 Hugging Face 模型卡并转化为 JSON。

**📈 对比分析**

与零样本、SFT、DPO 等基线对比，最终模型在文本、图像、音频任务上平均准确率提升至约 70%，显著优于传统检索或仅文本方法。

**⚠️ 局限性**

局限性包括：对文本和图像查询的准确率仍相对低；检索依赖嵌入质量；数据集规模与多样性有限；对极端模态或极少见工具的泛化尚未充分验证。

---

## 351. What if we have 90 minutes only to teach programming?

**arXiv ID:** 2604.14942 | [PDF](https://arxiv.org/pdf/2604.14942v1)

**作者:** Attila Egri-Nagy `[一作]` (Akita International University), Attila Egri-Nagy `[通讯]` (Akita International University)

**通讯引用:** 327 | [OpenAlex ID](https://openalex.org/A5043730003)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一门基于concatenative函数式语言的90分钟快速编程教学课程

**💡 创新点**

通过最小核心语言实现代码即数据（homoiconicity）与元语言，让学习者在极短时间内掌握递归与高级概念

**🔧 技术方法**

使用堆栈式执行模型、逆波兰算术、列表操作、条件与递归、元语言定义等技术，语言实现基于Clojure，文档使用Typst

**📊 数据集**

未使用传统数据集，而以Project Euler题目作为练习与评测素材

**📈 对比分析**

通过与学生、工程师、教授等多背景受众的互动实验评估，未给出数值性能指标，但证明教学材料在短时内能有效提升理解

**⚠️ 局限性**

受限于未完成完整性能与长期学习效果评测，且实现仍处于试点阶段，缺乏跨平台验证与广泛实证数据

---

## 352. LongAct: Harnessing Intrinsic Activation Patterns for Long-Context Reinforcement Learning

**arXiv ID:** 2604.14922 | [PDF](https://arxiv.org/pdf/2604.14922v1)

**作者:** Bowen Ping `[一作]` (Peking University), Baobao Chang `[通讯]` (Peking University)

**通讯引用:** 5936 | [OpenAlex ID](https://openalex.org/A5021459300)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LongAct，一种基于高幅值激活的稀疏更新策略，利用注意力查询/键向量中的高幅值特征指导强化学习训练，从而提升 LLM 的长上下文推理能力。

**💡 创新点**

创新点在于：①将模型内部高幅值激活视为关键特征；②在 RL 训练中动态生成稀疏梯度掩码，只更新与这些激活对应的投影权重；③保持多头结构，避免全参数更新带来的噪声，显著提升长上下文性能。

**🔧 技术方法**

技术包括：Qwen3‑8B/4B 语言模型；强化学习框架 GRPO/DAPO；基于规则的奖励函数；激活幅值统计与 top‑k 选择；动态稀疏梯度掩码；对照实验与 ablation。

**📊 数据集**

使用数据集：LongBench v2、RULER（128K/64K）、InfiniteBench、DocQA‑RL‑1.6K、MemAgent 进行 RL 训练；对短上下文的 GSM8K、HumanEval、TruthfulQA 进行验证。

**📈 对比分析**

与官方发布模型、SFT 基线、全参数 RL 及多种 RL 算法（GRPO、DAPO、KL‑Cov 等）对比；LongAct 在 LongBench v2 上提升约 +8 %（36.73 vs 27.04），在 RULER‑128K 上提升约 +4 %（51.15 vs 44.42），在 InfiniteBench 上平均提升 2‑3 %，并在短上下文任务中同样表现优于全参数 RL。

**⚠️ 局限性**

局限性：受算力限制，未在更大模型上进行 RL 训练，尚未验证方法在更大规模模型上的可扩展性。

---

## 353. Beyond Importance Sampling: Rejection-Gated Policy Optimization

**arXiv ID:** 2604.14895 | [PDF](https://arxiv.org/pdf/2604.14895v1)

**作者:** Ziwu Sun `[一作]`, Jiaheng Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种新的策略优化框架RGPO，使用可微分的接受门函数代替传统的importance‑sampling权重来选择可信样本，统一了TRPO、PPO与REINFORCE的梯度形式，并自然扩展到RLHF偏好对齐；

**💡 创新点**

创新点在于把样本选择（“是否使用”）从硬阈值或预处理转化为光滑、可微的优化目标，并通过理论证明实现有限且可控的方差与近似单调改进；

**🔧 技术方法**

核心技术包括可微分的g(r)门函数（如sigmoid）、有效梯度权重w(r)=g'(r)·r、KL惩罚的自适应调度以及在RLHF中引入双重比率门；

**📊 数据集**

实验数据集涵盖MuJoCo连续控制任务（HalfCheetah、Walker2d、Hopper、Ant）和Anthropic HH‑RLHF（文本奖励模型）；

**📈 对比分析**

与PPO、TRPO、AWR等基线比较，RGPO在Walker2d和Ant上分别提升约81%和47%，在其他任务与PPO相当，同时显著降低种子方差（0% KL spike率），在RLHF场景中实现最高奖励和最低与参考模型的KL，优于PPO‑RLHF和GRPO；

**⚠️ 局限性**

局限包括引入可控偏差、对门函数和锐度k的超参敏感、在部分任务（如Hopper）可能过于保守，以及尚未在大规模语言模型或离线评估中验证其性能。

---

## 354. MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration

**arXiv ID:** 2604.14889 | [PDF](https://arxiv.org/pdf/2604.14889v1)

**作者:** Xinyu Liu `[一作]` (Northeastern University), Jingbo Zhu `[通讯]` (Northeastern University)

**通讯引用:** 2150 | [OpenAlex ID](https://openalex.org/A5100370155)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MemoSight 框架，通过记忆令牌压缩上下文、前瞻令牌并行预测，统一解决 Chain‑of‑Thought 推理的 KV 缓存占用与推理速度瓶颈。

**💡 创新点**

创新点在于将上下文压缩与多步预测（MTP）通过特殊令牌与位置对齐实现无架构改动的统一训练与推理策略，避免传统 MTP 的参数开销与预训练依赖。

**🔧 技术方法**

使用特殊令牌（记忆令牌 ⟨m⟩、前瞻令牌 ⟨f⟩、边界令牌 ⟨b⟩）、位置感知对齐、专门的注意力掩码、可调压缩比与前瞻步长，以及推理时的并行推断与 KV 缓存清理。

**📊 数据集**

在四大推理基准上评估：GSM8K、MMLU、GPQA、BBH，使用 Qwen2.5‑7B 与 Llama‑3.1‑8B 两大模型。

**📈 对比分析**

与 CoT、Distill‑R1、Vanilla、LightThinker 以及后处理加速方法 H2O、SepLLM 等对比，MemoSight 在保持与 Vanilla 相近准确率的同时，将 KV 缓存占用降低约 66%，推理速度提升 1.56×，并在加入 Speculative Decoding 后进一步加速 24–30%。

**⚠️ 局限性**

主要限制包括：在更高压缩比（>8×）会显著损失准确性；前瞻步长需在 1–3 之间权衡，过大导致训练噪声；目前仅在单一模型规模（7B/8B）验证，跨规模与更大模型需进一步验证。

---

## 355. FSDETR: Frequency-Spatial Feature Enhancement for Small Object Detection

**arXiv ID:** 2604.14884 | [PDF](https://arxiv.org/pdf/2604.14884v1)

**作者:** Jianchao Huang `[一作]` (Jiangnan University), Tao Yan `[通讯]` (Jiangnan University)

**通讯引用:** 2408 | [OpenAlex ID](https://openalex.org/A5025290228)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FSDETR框架，在RT-DETR基础上通过频率‑空间特征增强提升小目标检测性能

**💡 创新点**

创新点包括：①将Spatial Hierarchical Attention Block（SHAB）嵌入骨干网络以实现局部与全局注意力融合；②引入Deformable Attention-based Intra‑scale Feature Interaction（DA‑AIFI）在同尺度内动态聚焦信息；③设计Frequency‑Spatial Feature Pyramid Network（FSFPN）与Cross‑domain Frequency‑Spatial Block（CFSB）将频域滤波与空间边缘提取协同增强

**🔧 技术方法**

使用技术包括RT‑DETR基线、CSPNet骨干、SHAB、DA‑AIFI、FSFPN、CFSB、Scharr算子、二维DFT/IDFT、可变形注意力、Varifocal Loss、Focaler‑EIoU等

**📊 数据集**

实验数据集为VisDrone 2019与TinyPerson

**📈 对比分析**

与多种state‑of‑the‑art（CNN、YOLO、RT‑DETR、RT‑DETRv2、D‑Fine‑M等）对比，参数仅14.7M，VisDrone AP_S 13.9%，TinyPerson AP_50^tiny 48.95%，在小目标指标上超过同等规模模型

**⚠️ 局限性**

局限性在于对极端稀疏或复杂背景的鲁棒性仍待验证，频域滤波在不同场景的通用性尚未充分探索，且在多尺度大任务场景下性能未知

---

## 356. Does RL Expand the Capability Boundary of LLM Agents? A PASS@(k,T) Analysis

**arXiv ID:** 2604.14877 | [PDF](https://arxiv.org/pdf/2604.14877v1)

**作者:** Zhiyuan Zhai `[一作]` (Fudan University), Xin Wang `[通讯]` (Fudan University)

**通讯引用:** 84344 | [OpenAlex ID](https://openalex.org/A5100328102)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并验证了 Pass@(k,T) 这一二维评估指标，用以区分 LLM 代理在工具使用场景下的能力扩展与采样效率提升；

**💡 创新点**

创新点在于通过同时调节采样预算 k 与交互深度 T，揭示了强化学习在组合式工具使用任务中能真正扩展代理能力而非仅提升可靠性；

**🔧 技术方法**

主要技术包括基于 GRPO 的强化学习、对比监督微调（SFT）、ReAct 交互循环、以及对策略多样性、困惑度拆分和交叉策略交换的机制分析；

**📊 数据集**

使用了 HotPotQA（包含 200 条训练问题）和 MATH-500 作为评测基准，基模型为 Qwen2.5‑7B‑Instruct；

**📈 对比分析**

在对比实验中，RL 在需要顺序检索的桥接问题上将可解题集从 77 提升至 81（+4），而 SFT 在同一任务上则出现退化；Pass@(k,T) 曲线显示 RL 的优势随 k 增大而加剧，验证了能力扩展而非单纯效率提升；

**⚠️ 局限性**

局限性包括仅使用单一 7B 模型、单一 BM25 检索工具、有限的 200 条训练样本以及未测试更大规模模型或更深交互深度。

---

## 357. Open-Set Vein Biometric Recognition with Deep Metric Learning

**arXiv ID:** 2604.14874 | [PDF](https://arxiv.org/pdf/2604.14874v1)

**作者:** Paweł Pilarek `[一作]` (Wroclaw University of Science and Technology), Anna Górska `[通讯]` (Wroclaw University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了基于深度度量学习的开放式手部血管生物识别框架，使用原型匹配和阈值拒绝实现未知身份的检测。

**💡 创新点**

创新点在于将单位超球面嵌入与批量硬三元组损失相结合，并在严格的主体离散协议下评估跨数据集的开放式性能。

**🔧 技术方法**

采用 ResNet50+CBAM 作为主干，ℓ2 归一化嵌入，批量硬三元组损失，余弦相似度原型匹配及阈值决策。

**📊 数据集**

采用 MMCBNU_6000、UTFVP、FYO、Dorsal Hand Vein 四个公开手部血管数据库。

**📈 对比分析**

与多种主干和损失进行对比，ResNet50-CBAM 在 MMCBNU_6000 上达到 OSCR 0.9945、AUROC 0.9974、EER 1.57%，在大规模数据集表现优异；跨数据集时性能下降，表明对数据量敏感。

**⚠️ 局限性**

主要局限是对小规模或域迁移数据集的鲁棒性不足，需要更多样本或域适配技术。

---

## 358. Curvature-Aligned Probing for Local Loss-Landscape Stabilization

**arXiv ID:** 2604.14870 | [PDF](https://arxiv.org/pdf/2604.14870v1)

**作者:** Nikita Kiselev `[一作]` (Moscow Institute of Physics and Technology), Andrey Grabovoy `[通讯]` (Moscow Institute of Physics and Technology)

**通讯引用:** 89 | [OpenAlex ID](https://openalex.org/A5057859442)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在样本增长下提出了统一的局部损失景观稳定性评估方法，并提出了只在主Hessian子空间内进行探测的子空间均方差指标 Δ₂^(D)。

**💡 创新点**

创新点在于将探测方向视为可设计变量，将稳定性量化为观测问题，并证明在局部二次模型下子空间探测不降低 𝒪(k⁻²) 收敛率，只将曲率依赖从维数 N 换到子空间维数 D。

**🔧 技术方法**

采用 Hessian‑vector products、迭代特征分解以及高斯矩估计等二阶工具，构建了三种可扩展的估计器（直接 Monte C​arlo、二次 Monte C​arlo、Gaussian‑moment）。

**📊 数据集**

实验使用 107 M 参数的 decoder‑only transformer（nanochat）在训练 3500 步时验证。

**📈 对比分析**

与全空间均方差指标比较，子空间指标在 D/N≈10⁻⁶ 时已与全空间指标保持一致，Gaussian‑moment 估计器比直接 Monte C​arlo 快约 18 000 倍且误差可忽略。

**⚠️ 局限性**

局限在于仅在局部二次近似下成立，子空间选择在更一般的非二次或快速漂移情形下可能需要自适应；此外实验仅在单一模型规模上进行，尚未验证对更大模型或不同训练 regime 的泛化。

---

## 359. Vibe-Coding: Feedback-Based Automated Verification with no Human Code Inspection, a Feasibility Study

**arXiv ID:** 2604.14867 | [PDF](https://arxiv.org/pdf/2604.14867v1)

**作者:** Michal Töpfer `[一作]` (Charles University), Petr Hnětynka `[通讯]` (Charles University)

**通讯引用:** 1127 | [OpenAlex ID](https://openalex.org/A5062364548)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不进行人工代码检查的情况下，结合运行时监控与反馈循环，对由LLM生成的适配管理器进行自动验证，应用于集合自适应系统（CAS）

**💡 创新点**

提出了将Fine‑grained Temporal Constraints（FCL）与反馈循环相结合的框架，证明精准的约束反馈显著提升Vibe Coding的收敛效率

**🔧 技术方法**

使用FCL（第一阶时序逻辑）、LLM（如GPT‑5）生成代码、自动化适配循环、约束验证器与动态测试覆盖技术

**📊 数据集**

以Dragon Hunt游戏情景作为CAS案例，构造了多初始状态和随机种子组成的测试集

**📈 对比分析**

与粗粒度指标反馈和仅通用约束反馈对比，实验显示全约束反馈在10次迭代内收敛率最高、平均迭代次数最低；粗粒度反馈往往停滞不前

**⚠️ 局限性**

依赖于约束设定的准确性与完整性，且实验仅在合成的Dragon Hunt案例中验证，可能在更大规模或真实CAS中需要更丰富的约束与覆盖

---

## 360. WavAlign: Enhancing Intelligence and Expressiveness in Spoken Dialogue Models via Adaptive Hybrid Post-Training

**arXiv ID:** 2604.14932 | [PDF](https://arxiv.org/pdf/2604.14932v1)

**作者:** Yifu Chen `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 12247 | [OpenAlex ID](https://openalex.org/A5079260216)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种单阶段自适应混合后训练方法，通过将强化学习（RL）仅作用于文本令牌并用监督微调（SFT）锚定语音令牌，来同步提升端到端语音对话模型的语义质量和语音表现。

**💡 创新点**

创新点在于：1）将偏好优化与监督微调分离到不同模态，显著减少语音漂移；2）设计了基于滚动样本置信度和分辨率的动态门控机制，自动调节RL与SFT的权重；3）在混合模态输出上首次实现了鲁棒的全局语义提升与细粒度语音稳定。

**🔧 技术方法**

采用了Group Relative Policy Optimization（GRPO）和Direct Preference Optimization（DPO）进行偏好学习，结合SFT的教师强制训练，并使用EMA平滑动态权重；模型结构包括VITA‑Audio（交错生成）与KimiAudio（并行生成），通过音频奖励模型Gemini‑2.5‑Pro进行语音质量评分。

**📊 数据集**

数据集涵盖13.5k条音频指令样本，来源于公开数据（UltraChat、SciQ、GSM8K、SHP、ExamQA、Alpaca、ScienceQA、Ai2ARC、PKUSafe）以及自制逻辑与表达数据，随后通过多轮采样与人工评分构建偏好对。

**📈 对比分析**

在VoiceBench、OpenAudioBench和VStyle三大基准上与SFT、DPO、RL、两阶段混合等多种基线比较，实验表明本文方法在语义IQ上提升约12‑15%（相较于SFT），在语音EQ上提升约10‑12%，并在人工主观评估中显著优于所有对照模型。

**⚠️ 局限性**

主要局限性包括：1）仅使用序列级奖励，缺乏更细粒度的语音反馈；2）音频评判者的可靠性与校准仍待提升，可能影响奖励质量；3）未对PPO等更强大且需语音级反馈的算法进行探索。

---

## 361. IE as Cache: Information Extraction Enhanced Agentic Reasoning

**arXiv ID:** 2604.14930 | [PDF](https://arxiv.org/pdf/2604.14930v1)

**作者:** Hang Lv `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 28792 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种将信息抽取（IE）转化为认知缓存（IE-as-Cache），通过查询驱动抽取和缓存感知推理提升LLM的多步推理能力。

**💡 创新点**

创新点在于将IE从单次任务转为动态可读写的缓存层，并在推理过程中持续更新缓存，显著减少噪声干扰并提升推理准确性。

**🔧 技术方法**

采用了基于LLM的查询驱动抽取、模式解耦抽取、缓存更新和ReAct式推理等技术，并将抽取结果作为结构化缓存进行推理。

**📊 数据集**

实验使用了TACT（逻辑问答）、Calendar Scheduling（日程规划）和QMSUM（查询聚焦摘要）三大数据集。

**📈 对比分析**

与Generic、CoT、ReAct及IE-as-Tool等基线对比，IE-as-Cache在大多数模型和任务上均获得最高或接近最高的精度，尤其在长文本噪声场景下小模型提升显著。

**⚠️ 局限性**

局限性包括对极低资源模型仍难以突破，某些任务对精确模式抽取的依赖可能限制泛化，且动态缓存更新增加了推理时的计算成本。

---

## 362. Text2Arch: A Dataset for Generating Scientific Architecture Diagrams from Natural Language Descriptions

**arXiv ID:** 2604.14941 | [PDF](https://arxiv.org/pdf/2604.14941v1)

**作者:** Shivank Garg `[一作]` (IIT Roorkee), Manish Gupta `[通讯]` (Microsoft)

**通讯引用:** 5509 | [OpenAlex ID](https://openalex.org/A5101454729)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了将自然语言描述转换为科学架构图的任务，并通过中间的DOT代码实现自动生成；同时发布了规模达75k条图像-文本-代码三元组的高质量数据集，并对多种模型进行训练与评估。

**💡 创新点**

创新点包括：①首次定义并实现文本到DOT代码的端到端流程；②构建大规模、干净且语义对齐的文本-代码-图像数据集；③设计混合式生成管线（GPT + 目标检测 + OCR + GPT精炼），显著提升DOT代码质量；④提出针对图结构的节点/边精度、召回、F1、PR-AUC、Jaccard等多维度评估指标。

**🔧 技术方法**

采用的技术包括：大型语言模型（GPT‑4o、LLaMA‑3‑8B‑Instruct、Qwen‑2‑7B‑Instruct、DeepSeek‑LLM‑7B‑Chat）进行微调与少量示例推理；目标检测（基于 Faster‑RCNN 的专用模型）与OCR（Florence‑2）用于结构化信息提取；GPT 进一步精炼生成的DOT代码；GraphViz 的 DOT 编译器将代码转化为图像；以及自定义的图结构评估指标。

**📊 数据集**

使用的数据集为自研的 text2arch 数据集，来源于 Paper2Fig、ACL‑Fig、SciFig 等公开图形数据集，经过筛选与人工标注后得到约75,127个科学架构图及对应的描述和 DOT 代码；此外还使用了 99 条人工标注的图像/代码样本作为验证集。

**📈 对比分析**

与 DiagramAgent 基线、GPT‑4o 零样本推理、少样本提示学习以及三种 LLM 的微调版本进行对比；评估指标涵盖文本相似度（ROUGE‑L、CodeBLEU、Edit Distance、chrF）和图结构精度（节点/边 Precision/Recall/F1、PR‑AUC、Jaccard）。实验结果显示，微调后的 DeepSeek‑7B 在大多数指标上均优于其它模型，且在手工评测中与 GPT‑4o 的兼容度相近，说明微调能显著提升结构化代码生成质量。

**⚠️ 局限性**

局限性包括：①数据集仅覆盖科学架构图，对其他类型图表的泛化能力未知；②生成过程仍需 GPT 进行精炼，存在对模型输出的依赖；③对最终图像质量的直接评估缺失，仅通过代码评估；④在复杂图形或极大节点数时，模型的准确性可能下降；⑤需要人工进一步修订生成结果以满足特定排版或美学需求。

---

## 363. Dual-Axis Generative Reward Model Toward Semantic and Turn-taking Robustness in Interactive Spoken Dialogue Models

**arXiv ID:** 2604.14920 | [PDF](https://arxiv.org/pdf/2604.14920v1)

**作者:** Yifu Chen `[一作]` (Zhejiang University), Zhou Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 12247 | [OpenAlex ID](https://openalex.org/A5079260216)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种双轴生成式奖励模型（Dual‑Axis Reward Model），能够在全双工语音对话中分别评估语义质量与交互时机，并输出单一二元奖励；

**💡 创新点**

创新点在于：①基于系统化交互事件分类构建交互错误词典；②将语义与时序评价拆分为两条链式推理（CoT），实现可解释诊断；③三阶段训练策略（感知基线→推理蒸馏→GRPO优化）使模型兼顾准确性与泛化；

**🔧 技术方法**

技术包括：大规模多模态LLM（Qwen‑2.5‑Omni‑7B）作为基础模型；Supervised Fine‑Tuning（SFT）进行语音事件识别、说话人分离与转录；链式推理(CoT)生成结构化评价；Group Relative Policy Optimization（GRPO）对奖励信号进行强化学习；

**📊 数据集**

数据集由程序化合成语音（6,361样本≈146h）与真实人机/人际对话（100人机，289人际≈10h）组成，均经过人工标注；

**📈 对比分析**

与闭源模型（GPT‑4o、Gemini‑2.5‑Pro/Flash）及开源音频评估模型（Qwen2Audio、AudioReasoner、KimiAudio、Audio‑Flamingo3）在零样本推理下对比，模型在ID/OOD合成数据上精度≈98%，在人机/人际真实对话上分别达≈86%与77%，显著优于所有基线；

**⚠️ 局限性**

局限：尚未在在线RL框架中验证奖励效果；二元奖励可能过于稀疏，缺乏细粒度反馈；未来需要多级评分或多目标RL以提升模型的实用性。

---

## 364. Beyond Prompts: Unconditional 3D Inversion for Out-of-Distribution Shapes

**arXiv ID:** 2604.14914 | [PDF](https://arxiv.org/pdf/2604.14914v1)

**作者:** Victoria Yue Chen `[一作]` (ETH Zürich), Maks Ovsjanikov `[通讯]` (École Polytechnique)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究并实现了一种利用无条件生成先验进行3D网格反演与编辑的框架，解决了文本提示失效导致的“sink trap”问题；

**💡 创新点**

创新点在于首次量化并验证3D生成模型的“sink trap”现象，提出通过无条件先验稳定采样轨迹实现高保真逆向与纯文本驱动编辑，且不需要辅助模型或图像先验；

**🔧 技术方法**

使用了TRELLIS的rectified flow模型、Euler反演、Null-Text Inversion（NTI）、无条件prompt、CFG引导、SigLIP等技术；

**📊 数据集**

使用DT4D数据集（200个非刚体人形/动物角色）和80个TRELLIS生成形状进行评估；

**📈 对比分析**

与VoxHammer、TRELLIS原始编辑等方法对比，使用SigLIP、LPIPS、L1等指标评估；在DT4D上编辑速度约快20×，编辑质量显著优于Baseline；

**⚠️ 局限性**

局限性在于仍受生成模型分布限制，极端语义变换可能导致几何失真，且未对几何一致性做显式约束。

---

## 365. Reward-Aware Trajectory Shaping for Few-step Visual Generation

**arXiv ID:** 2604.14910 | [PDF](https://arxiv.org/pdf/2604.14910v1)

**作者:** Rui Li `[一作]` (University of Science and Technology of China), XueLong Li `[通讯]` (TeleAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Reward‑Aware Trajectory Shaping (RATS) 框架，实现极少采样步骤下的视觉生成，并通过奖励门控实现与人类偏好的对齐。

**💡 创新点**

创新点在于将 sigma 对齐的多时域轨迹匹配与奖励门控相结合，让学生模型在压缩步骤时不被教师上限约束，能够在奖励驱动下突破教师性能。

**🔧 技术方法**

技术细节包括：EMA 教师、sigma 匹配的多视窗轨迹对齐、奖励门控的动态权重、流匹配生成模型、LoRA 微调、HPSv2.1 奖励模型等。

**📊 数据集**

使用 DanceGRPO 数据集（图像与视频），结合 FLUX1.0‑dev、Wan2.1‑T2V‑1.3B‑480P 预训练模型，奖励模型为 HPSv2.1。

**📈 对比分析**

在 3–50 NFEs 的图像生成任务中，与原始多步、Hyper‑SD、SenseFlow 等基线相比，RATS 在 HPS、PickScore、ImageReward 等指标上提升 10–15% 以上；在 5–8 NFEs 的视频生成任务中，同样超过 50 NFEs 基线，显著提升质量与语义一致性。

**⚠️ 局限性**

局限性包括：在更大步长下提升有限；对奖励模型的依赖性高；训练需要额外的 EMA 教师但仅在训练时存在；对不同分辨率、任务的泛化性仍待验证。

---

## 366. Can LLMs Score Medical Diagnoses and Clinical Reasoning as well as Expert Panels?

**arXiv ID:** 2604.14892 | [PDF](https://arxiv.org/pdf/2604.14892v1)

**作者:** Amy Rouillard `[一作]` (Wits MIND Institute, University of Witwatersrand), Bruce A. Bassett `[通讯]` (Wits MIND Institute, University of Witwatersrand)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用多评审面板对比评估LLM Jury（Anthropic Claude Opus 4.1、Google Gemini 2.5 Pro、OpenAI o3）在南非公共医院539例完整病例中的诊断与临床推理评分。

**💡 创新点**

首次系统验证LLM Jury在中等收入国家医疗案例中的专家级评估性能，并通过后验等距回归校准提升与专家评分的一致性；同时首次量化LLM评判者与人类评审在严重风险错误率上的差异。

**🔧 技术方法**

LLM-as-judge框架、等距回归校准、RMSE、Spearman ρ、Cohen κ、严重风险错误率评估等多维度评价指标。

**📊 数据集**

VALID研究收集的南非公共医院539例完整病例数据集（包含诊断、实验室、影像、CT、MRI、X光等信息），其中包括ward诊断与多模型LLM诊断。

**📈 对比分析**

通过与专家主评审和重评审面板对照，计算offset、RMSE、Spearman ρ、Cohen κ；校准后LLM Jury在大多数指标上匹配或优于重评审，严重错误率约5%低于人类评审的16.7%，显示更可靠且可扩展。

**⚠️ 局限性**

局限性包括对非英语病例性能下降、缺少患者元数据导致误判、样本量有限导致置信区间宽、ICD-10编码细粒度不足等问题。

---

## 367. Cooperate to Compete: Strategic Data Generation and Incentivization Framework for Coopetitive Cross-Silo Federated Learning

**arXiv ID:** 2604.14886 | [PDF](https://arxiv.org/pdf/2604.14886v1)

**作者:** Thanh Linh Nguyen `[一作]` (Trinity College Dublin), Quoc-Viet Pham `[通讯]` (Trinity College Dublin)

**通讯引用:** 16748 | [OpenAlex ID](https://openalex.org/A5062525719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种兼顾合作与竞争（coopetition）的跨数据中心联邦学习（CFL）框架，联合建模非IID数据异质性与组织间竞争，利用生成式AI（GenAI）进行合成数据生成，并通过收益再分配机制来平衡竞争外部性，目标是最大化系统整体社会福利。

**💡 创新点**

创新点：
1) 将生成式AI合成数据量视为组织的战略决策；
2) 在每轮训练中将组织间的交互建模为加权潜在博弈，能够直接求解纯纳什均衡；
3) 设计预算平衡的收益再分配奖励方案，既补偿高贡献方又内化竞争损失；
4) 通过实验揭示竞争强度、数据异质性和再分配强度对策略与社会福利的耦合影响。

**🔧 技术方法**

技术手段：
- 加权潜在博弈理论与KKT条件求解均衡；
- 固定点迭代算法实现合成数据生成策略；
- 采用预训练的生成模型（如VAE/GAN）生成合成数据；
- FedAvg聚合，基于Flower框架实现联邦训练；
- 对局部学习误差使用经验曲线的指数缩放法则。

**📊 数据集**

实验数据集：Fashion-MNIST、CIFAR-10、CIFAR-100，分别使用MobileNetV3-small、MobileNetV2、ResNet-34模型。

**📈 对比分析**

比较方法：对比Vanilla CFL、无竞争版、无数据生成版、随机生成、最大生成等基线；评估指标为系统整体社会福利（各组织效用之和）。结果表明，本文框架在所有数据集和不同竞争/异质性设置下均优于基线，尤其在高竞争、高异质性场景下提升显著；收益再分配强度对最优性能具有非单调影响。

**⚠️ 局限性**

局限性：
- 再分配系数ξ的最佳值与任务难度、异质性高度耦合，需经验调参；
- 依赖经验曲线的缩放法则，若超出拟合范围可能失效；
- 仅在小规模（N=10）模拟中验证，缺乏大规模真实场景测试；
- 生成式AI合成数据的计算成本高，实际部署时需考虑资源限制；
- 假设预算平衡与个体理性可实现，若外部干预或信息不完全则可能偏离预期。

---

## 368. RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding

**arXiv ID:** 2604.14885 | [PDF](https://arxiv.org/pdf/2604.14885v1)

**作者:** Zihong Zhang `[一作]` (Wuhan University), Hai Zhao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6940 | [OpenAlex ID](https://openalex.org/A5036050911)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种训练无关的推理加速方法RACER，将检索式草稿与基于logit的预测统一为一种可扩展的猜测-验证框架。

**💡 创新点**

创新点在于：① 使用AC自动机加LRU淘汰机制构建高效检索树；② 采用copy-logit重用策略构建多层logit树；③ 将检索候选与logit树候选通过Trie合并，形成更丰富的草稿树。

**🔧 技术方法**

主要技术包括：AC自动机与LRU淘汰、logit重用（copy-logit）、logit树与检索树的宽度分配策略、Trie并集与猜测-验证机制。

**📊 数据集**

在Spec-Bench、HumanEval、MGSM-ZH（以及GSM8K、AIME、MATH等）三大基准上进行实验，覆盖对话、翻译、摘要、问答、代码生成和数学推理等任务。

**📈 对比分析**

与检索式基线（PLD、REST）、logit式基线（Token Recycling、LogitSpec）及模型式基线（EAGLE-3）比较，RACER在MAT、速度提升上均优于所有对比方法，平均速度提升超过2×，并在多语言、多模型规模下保持稳定。

**⚠️ 局限性**

局限性在于仅在文本任务上验证，尚未评估在多模态（视觉、语音）场景下的适用性。

---

## 369. An Intelligent Robotic and Bio-Digestor Framework for Smart Waste Management

**arXiv ID:** 2604.14882 | [PDF](https://arxiv.org/pdf/2604.14882v1)

**作者:** Radhika Khatri `[一作]` (BITS Pilani), M. B. Srinivas `[通讯]` (BITS Pilani)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个集成的智能废物管理框架，包含基于YOLOv8和ROS的实时机器人分类分拣模块以及利用多传感器和PSO优化的生物消化器；

**💡 创新点**

创新点在于将视觉驱动的机器人分拣与基于回归模型的动态参数优化耦合，形成闭环自动化处理链；

**🔧 技术方法**

采用了YOLOv8目标检测、ROS运动规划、MyCobot 280机器人臂、Jetson Nano硬件、PSO优化算法、回归预测模型及多传感器实时监测；

**📊 数据集**

使用了Kaggle公开废物分拣数据集（约56,790张图像，四类：食物废弃物、金属、纸张、塑料），并做了数据增强；

**📈 对比分析**

通过与传统手工分拣和静态生物消化实验对比，机器人模块平均分类准确率达到98%，PSO优化后回归模型R²为0.93，生物消化器在17天内产生的沼气产量显著高于未优化基线；

**⚠️ 局限性**

局限性包括传感器噪声和废物组成的变异性对性能的影响、单臂作业吞吐量有限以及对大规模部署的可扩展性与多臂协作的进一步验证需求。

---

## 370. Graph Theoretical Outlier Rejection for 4D Radar Registration in Feature-Poor Environments

**arXiv ID:** 2604.14857 | [PDF](https://arxiv.org/pdf/2604.14857v1)

**作者:** Georg Dorndorf `[一作]` (xtonomy), Masrur Doostdar `[通讯]` (xtonomy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种在4D雷达扫描配准中使用图PCM与雷达不确定性感知的配对一致性度量来提升鲁棒性的配准方法。

**💡 创新点**

提出了利用雷达各向异性误差进行马氏距离归一化的配对一致性评分，并将图PCM作为ICP循环内的内点筛选步骤。

**🔧 技术方法**

使用的技术包括图PCM、最大团贪心启发式、ICP/GICP、雷达不确定性建模以及马氏距离一致性检验。

**📊 数据集**

实验数据来自奥地利阿尔卑斯山露天矿区的4D成像雷达（Bell B30E装载ARS548雷达）共25155帧，配合RTK‑GNSS地面真值。

**📈 对比分析**

与ICP点到点、GICP及PCM+ICP/PCM+GICP基线比较，PCM+GICP在1m段RPE降低29.6%，100m段降低至55%，相对漂移和旋转误差均显著改善。

**⚠️ 局限性**

限制包括与少数基线对比、雷达不确定性模型未进行现场校准导致马氏评分效果受限、地面真值误差未知以及缺乏动态场景验证。

---

## 371. ClimateCause: Complex and Implicit Causal Structures in Climate Reports

**arXiv ID:** 2604.14856 | [PDF](https://arxiv.org/pdf/2604.14856v1)

**作者:** Liesbeth Allein `[一作]` (KU Leuven), Marie-Francine Moens `[通讯]` (KU Leuven)

**通讯引用:** 9256 | [OpenAlex ID](https://openalex.org/A5075796989)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并手工标注了 ClimateCause 数据集，记录气候变化报告中的复杂因果结构，包括隐式、嵌套关系、相关性及时空上下文，并基于该数据集提出了语义复杂度的可读性指标；随后使用该数据集对大型语言模型在相关性推断和因果链推理任务上进行基准评测。

**💡 创新点**

①首次为气候变化文本提供高阶因果标注，包含相关性、嵌套因果与时空信息；②引入可读性评估指标以量化因果图语义复杂度；③通过对 LLM 的多任务评测验证数据集的实用价值。

**🔧 技术方法**

采用专家语言学家手工标注、句法重构、缩写解析、事件拆分；利用图论度量复杂度；基于零样本、少样本及链式思维的 Prompt 技术评估 GPT‑5.1 等大型语言模型；统计分析（χ²、Pearson、Kruskal‑Wallis）评估指标。

**📊 数据集**

主要使用 IPCC 科学决策报告生成的 ClimateCause 数据集（75 条语句，874 条因果关系），并与 BioCause、CaTeRS、BECAUSE 2.0、CRAB 等现有因果数据集做对比。

**📈 对比分析**

通过零样本、少样本、链式思维三种 Prompt 对 GPT‑5.1 进行 CorrI、CorrI+RC、CCR、CCR+ECI+RC 四项任务评测。相关性推断的 F1 约 0.86–0.92；因果链推理的召回高但精度低，提示模型对链式结构理解不足；链式思维略有提升。

**⚠️ 局限性**

数据集规模有限且聚焦单一领域；标注未覆盖因果强度与不确定性等属性；指标权重未经过学习或自适应；LLM 评测不具备跨领域通用性，结果易受数据泄露影响。

---

## 372. Towards Understanding Android APIs: Official Lists, Vendor Customizations, and Real-World Usage

**arXiv ID:** 2604.14943 | [PDF](https://arxiv.org/pdf/2604.14943v1)

**作者:** Sinan Wang `[一作]` (Southern University of Science and Technology), Yepang Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 19739 | [OpenAlex ID](https://openalex.org/A5100346563)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对四个官方Android API列表（JAR、XML、TXT、CSV）进行系统性实验，分析其内容差异、演化规律、设备可用性以及真实应用中的使用情况。

**💡 创新点**

创新点在于首次全面对官方API列表进行对比研究，揭示其不一致、缺失、包含非SDK/厂商定制API的特点，并评估这些差异对研究与实践的影响。

**🔧 技术方法**

主要技术手段包括基于Soot和veridex的静态分析、构建AAL-Reflector和APK-Analyzer工具，以及手工验证API来源与变迁。

**📊 数据集**

使用的数据集包括9台Android设备（包含Stock和厂商定制系统）、17,759个APK（F‑Droid开源、Google Play商业与恶意样本）以及对应的四个AAL文件。

**📈 对比分析**

通过对比四个AAL的API数量、交集与差集，检验API在设备上的可反射性，并统计各类API调用频率，结果显示只有约10% API在所有列表中出现，CSV占比最高且包含大部分非SDK接口；此外，设备差异导致部分CSV API缺失。

**⚠️ 局限性**

主要局限在于仅覆盖了六个Android版本和九台设备，未能覆盖所有厂商定制系统；AAL-Reflector只检测可被普通应用访问的API，系统专属API被忽略；APK-Analyzer的静态抽取可能漏检反射或混淆调用。

---

## 373. Hybrid Latents -- Geometry-Appearance-Aware Surfel Splatting

**arXiv ID:** 2604.14928 | [PDF](https://arxiv.org/pdf/2604.14928v1)

**作者:** Neel Kelkar `[一作]` (Technical University of Munich), Rüdiger Westermann `[通讯]` (Technical University of Munich)

**通讯引用:** 8184 | [OpenAlex ID](https://openalex.org/A5029621326)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种混合潜在表征，将每个二维表面元（surfels）携带低频几何特征，并通过单层哈希网格捕获高频纹理，从而实现稀疏且高质量的三维重建与新视角合成。

**💡 创新点**

主要创新点包括：1）将几何与外观显式分离的潜在分解，2）引入可变 Beta 核提升几何稀疏性和渲染效率，3）结合 MCMC 采样与 BCE 稀疏正则化，实现高效且可解释的表面元优化。

**🔧 技术方法**

采用可微分的 surfels、Beta 核、单层哈希潜在场、MCMC 采样、BCE 稀疏正则、MLP 解码以及基于 rasterization 的渲染技术。

**📊 数据集**

实验数据集涵盖公开的 NeRF Synthetic、Mip-NeRF 360 以及 DTU 三大基准集。

**📈 对比分析**

与 3DGS、2DGS、Beta‑Splatting、SuperGS 以及 NeST‑Splatting 进行对比，PSNR/SSIM/LPIPS 指标基本相当或略优，而表面元数量仅为对手的 1/10 甚至 1/100，帧率显著提升。

**⚠️ 局限性**

主要限制在于：解码 MLP 的计算量较大，难以与纯 SH 查询竞争；哈希网格的隐式特性不便于场景合并与编辑；以及单层哈希容量对高频纹理表达仍有限。

---

## 374. The Missing Knowledge Layer in AI: A Framework for Stable Human-AI Reasoning

**arXiv ID:** 2604.14881 | [PDF](https://arxiv.org/pdf/2604.14881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 375. Regret Tail Characterization of Optimal Bandit Algorithms with Generic Rewards

**arXiv ID:** 2604.14876 | [PDF](https://arxiv.org/pdf/2604.14876v1)

**作者:** Subhodip Panda `[一作]` (Indian Institute of Science), Shubhada Agrawal `[通讯]` (Indian Institute of Science)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5113012757)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种对广泛非参数奖励分布（包含有界支持、重尾和有限支持分布）通用的 KL_inf-UCB 算法，并证明其在期望误差上的渐进最优。

**💡 创新点**

首次给出该类最优 UCB 算法的 regret tail 上界，并证明在满足“判别等价”条件时该上界与已知的下界完全匹配；对于有限支持模型还给出了更紧的尾概率上界。

**🔧 技术方法**

利用 KL 散度的上界、对数探索函数、Sanov 型大偏差定理以及对 KL_inf 的时变一致收敛性质，对 regret 进行尾分布分析。

**📊 数据集**

本工作不依赖具体的数据集，而是在理论上对上述分布族进行抽象建模与分析。

**📈 对比分析**

通过理论推导和对已知下界的比对，展示了在判别等价情形下上界与下界一致，说明算法在罕见高损失事件上的鲁棒性；在非判别等价情形下仍存在上界与下界不匹配的间隙。

**⚠️ 局限性**

主要局限在于：对非判别等价分布族，regret tail 上界与下界仍存在差距；缺乏针对 Thompson Sampling 的尾行为分析；未来工作需进一步缩小尾概率的上界与下界差距并提升算法的尾鲁棒性。

---

## 376. HRDexDB: A Large-Scale Dataset of Dexterous Human and Robotic Hand Grasps

**arXiv ID:** 2604.14944 | [PDF](https://arxiv.org/pdf/2604.14944v1)

**作者:** Jongbin Lim `[一作]` (Seoul National University), Hanbyul Joo `[通讯]` (Seoul National University)

**通讯引用:** 3769 | [OpenAlex ID](https://openalex.org/A5036077761)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了HRDexDB，一个大规模标注的人类与多种机械手抓取任务的多模态数据集。

**💡 创新点**

创新点在于同步捕获人类与不同机械手的高精度三维轨迹、触觉信息以及成功/失败标注，提供统一的空间和时间坐标系，填补了跨体裁抓取数据缺口。

**🔧 技术方法**

采用了21摄像机稠密多视角系统、IMU/手势捕捉进行机械手遥控、MANO手模型重建、基于立体深度的6D目标追踪以及触觉传感器同步采集等技术。

**📊 数据集**

使用了HRDexDB本身的数据，包含约1.4K抓取序列、100种对象、4种机械手与人类手的配对数据。

**📈 对比分析**

通过多视角一致性评估、Mean Vertex Distance等指标验证定位精度可低至0.8mm，且展示了跨体裁抓取成功率差异，表明数据质量高、可用于训练跨模态策略。

**⚠️ 局限性**

局限在于触觉采样不统一、缺乏人类手和部分机械手的触觉数据、以及对“配对轨迹”定义仍不严谨。

---

## 377. MetaDent: Labeling Clinical Images for Vision-Language Models in Dentistry

**arXiv ID:** 2604.14866 | [PDF](https://arxiv.org/pdf/2604.14866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 378. Toward Agentic RAG for Ukrainian

**arXiv ID:** 2604.14896 | [PDF](https://arxiv.org/pdf/2604.14896v1)

**作者:** Marta Sumyk `[一作]` (Ukrainian Catholic University), Oleksandr Kosovan `[通讯]` (Ukrainian Catholic University)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5070600372)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在乌克兰语上的 Agentic RAG 系统，结合两阶段检索与轻量级 Agent 层进行查询改写和答案重试，以解决多域文档理解任务。

**💡 创新点**

创新点在于将 Agentic 机制（查询改写和答案重试）引入乌克兰语 RAG，并在受限 GPU 环境下实现可执行的轻量级 Agent 管道。

**🔧 技术方法**

技术包括 BGE-M3 稠密检索+reranker、TF‑IDF 字符级检索、Qwen2.5‑3B‑Instruct LLM 以及基于重试的 Agent 层。

**📊 数据集**

使用 UNLP 2026 Shared Task 的多域文档理解数据集（多选题 + 文档/页定位）。

**📈 对比分析**

与单一 LLM 或非 Agent RAG 对比，检索+Agent 在答案准确率上提升约 20%（从 0.63 提升至 0.81），但页定位精度略下降；整体评分略有提升。

**⚠️ 局限性**

局限性包括检索质量仍是瓶颈、Agent 层仅实现单步重试、单 GPU 9h 约束限制模型规模与复杂 Agent、未对乌克兰语进行微调、仅在该数据集评估。

---

## 379. xFODE+: Explainable Type-2 Fuzzy Additive ODEs for Uncertainty Quantification

**arXiv ID:** 2604.14880 | [PDF](https://arxiv.org/pdf/2604.14880v1)

**作者:** Ertugrul Kececi `[一作]`, Tufan Kumbasar `[通讯]` (Istanbul Technical University)

**通讯引用:** 2440 | [OpenAlex ID](https://openalex.org/A5010725194)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种解释性可量化系统辨识模型 xFODE+，在原有 xFODE 的增量状态和加性 ODE 结构上加入区间型二阶模糊逻辑系统，能够同时输出点预测和预测区间。

**💡 创新点**

创新点在于（1）通过三种前向分区策略（PS1‑PS3）将模糊集限制为仅激活相邻两条规则，显著提升推理透明度；（2）将 IT2‑FLS 的类型归约集聚合到增量状态更新中，实现在保持物理含义的同时实现不确定性量化；（3）设计复合损失函数，使模型端到端学习预测准确性与预测区间质量。

**🔧 技术方法**

采用区间型二阶模糊逻辑系统 (IT2‑FLS)、增量状态表示、加性 ODE 网络结构，并在深度学习框架下使用软参数化和复合损失（准确性损失 + 预测区间质量损失）进行训练。

**📊 数据集**

在 Hair Dryer、MR Damper、Steam Engine 三个公开系统辨识基准数据集上进行实验。

**📈 对比分析**

通过与 NODE、T1‑FODE、xFODE、IT2‑FODE、AFODE+ 等模型在 RMSE、PI 覆盖率 (PICP) 和 PI 归一化平均宽度 (PINAW) 等指标进行比较。结果表明 xFODE+ 在保持与 IT2‑FODE 相近的 PI 质量的同时，RMSE 与 NODE 接近，且参数量更少；但在某些数据集上 PI 宽度略大，且整体预测精度略低于纯准确性优化的模型。

**⚠️ 局限性**

主要局限在于（1）引入的 PI 质量损失导致点预测精度略下降；（2）PI 宽度在部分数据集上仍显宽，覆盖率高但可能导致置信区间过于保守；（3）规则后验解释性仅局限于相邻两条规则的激活，尚未深入探讨规则后效应的可解释性和更复杂的模糊推理能力。

---

## 380. SkillDroid: Compile Once, Reuse Forever

**arXiv ID:** 2604.14872 | [PDF](https://arxiv.org/pdf/2604.14872v1)

**作者:** Qijia Chen `[一作]` (University of Helsinki), Giulio Jacucci `[通讯]` (University of Helsinki)

**通讯引用:** 6743 | [OpenAlex ID](https://openalex.org/A5074899838)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一种三层结构的 SkillDroid 代理，通过 LLM 先行执行并将成功的 GUI 轨迹编译成可重复使用的参数化技能模板，在后续调用中实现无 LLM 直执行。

**💡 创新点**

关键创新在于：① 将 LLM 产生的交互轨迹转化为完全可执行的技能模板（含加权元素定位器和状态描述符）；② 设计匹配级联（正则、语义嵌入+应用过滤）和逐步回退机制，使得技能在不同 UI 变异下仍能保持高成功率；③ 引入失败学习层，在技能可靠性下降时自动重新编译，从而实现系统随着使用而不断提升。

**🔧 技术方法**

使用技术包括：LLM（OpenAI gpt‑4o‑mini/​gpt‑4o）进行指令推理与技能编译；Android Accessibility 与 ADB 实现 UI 状态捕获和动作执行；句子嵌入模型 all‑MiniLM‑L6‑v2 进行语义匹配；加权元素定位器与状态描述符用于精确定位与偏差检测；基于回退的多级执行框架；SQLite 存储技能库。

**📊 数据集**

数据集：150 轮实验，涵盖 15 种移动任务（多字段表单、时间选择、浏览器搜索、设置导航等），每个任务提供 4 级指令变体（编译、低、中、高）以及 10 次扰动测试（应用选择器、清除数据、权限撤销）。

**📈 对比分析**

与无技能的纯 LLM（stateless）基线对比：SkillDroid 取得 85.3 % 的成功率，基线仅 62 %；LLM 调用次数平均每轮下降 49 %（5.8 → 11.3 次）；执行延时平均从 84 s 降至 69 s，纯重放阶段（无 LLM 调用）更快 2.4 倍。技能重放成功率为 100 %，在 79 轮中实现零 LLM 调用。

**⚠️ 局限性**

局限性：① 任务覆盖面有限，无法处理极为动态的多字段表单和多应用协作；② 仅依赖可访问性树的文本信息，缺乏视觉感知；③ 通过 ADB 触发动作的 100 ms 延迟较慢，原生接口可进一步加速；④ 目前仅支持英文指令与单一 LLM，缺乏多语言和开源模型的验证；⑤ 技能当前仅限单一任务单一应用，缺乏跨任务链式组合能力。

---

## 381. Schema Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding

**arXiv ID:** 2604.14862 | [PDF](https://arxiv.org/pdf/2604.14862v1)

**作者:** Yifan Le `[一作]` `[通讯]` (Zhejiang University), Yifan Le (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究在受限解码中，模式键（schema key）的措辞如何通过隐式指令通道影响大型语言模型的结构化生成性能。

**💡 创新点**

首次系统研究将结构化生成视为多通道指令问题，揭示模式键措辞本身可视为指令通道，并发现不同模型家族对提示层和模式层指令的敏感度及其非加性交互。

**🔧 技术方法**

使用受限解码（XGrammar）与上下文无关文法等结构约束，搭配多通道指令框架（prompt层 + schema键层），对不同模型进行对照实验。

**📊 数据集**

在数学推理基准上评估：GSM8K 和 Math500。

**📈 对比分析**

对比四种指令置放设置（无指令、键唯、提示唯、两者合并），记录模型准确率变化，发现 Qwen 系列对键级指令高度敏感，LLaMA 系列更依赖提示；两者合并并不总是叠加提升，交互存在正负效应。

**⚠️ 局限性**

研究局限：仅验证数学推理任务；仅探讨键措辞，未涉及字段描述、顺序、嵌套等其他模式元素；缺乏机制性解释，未给出通用最优键设计方案。

---

## 382. Agentic Explainability at Scale: Between Corporate Fears and XAI Needs

**arXiv ID:** 2604.14984 | [PDF](https://arxiv.org/pdf/2604.14984v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 383. DockAnywhere: Data-Efficient Visuomotor Policy Learning for Mobile Manipulation via Novel Demonstration Generation

**arXiv ID:** 2604.15023 | [PDF](https://arxiv.org/pdf/2604.15023v1)

**作者:** Ziyu Shan `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 3209 | [OpenAlex ID](https://openalex.org/A5100389366)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 DockAnywhere 框架，通过在三维空间中对单一演示进行轨迹提升与视点合成，以生成多种可行停靠点的训练数据，从而提升移动操纵的视角泛化能力。

**💡 创新点**

创新点在于将停靠点相关的基座运动与不变的接触技能解耦，利用 TAMP 重新规划基座运动并在 3D 点云层面进行空间编辑，保持视觉与动作的一致性；同时采用第三人称视角消除停靠点偏移导致的视角变异。

**🔧 技术方法**

使用的技术包括任务与运动规划 (TAMP)、点云采集与分割、三维几何变换与点级编辑、行为克隆 (Behavior Cloning)、DP3 视觉动觉策略以及 ManiSkill 仿真平台。

**📊 数据集**

使用的数据集包括 ManiSkill（包含 RoboCasa 场景）进行仿真实验，以及在 Galaxea R1 移动机械臂上采集的 ZED2 深度相机与 Livox LiDAR 数据用于真实世界测试。

**📈 对比分析**

与 DP、DP3 基线以及 DemoGen 数据增强方法进行对比；在一组 5 个停靠点下，DockAnywhere 的平均成功率从 15–18% 提升至 97–98%；在真实任务中达成 60%–40% 的成功率，显著优于传统方法。

**⚠️ 局限性**

局限性包括对第三人称视角硬件的依赖，极端停靠点偏移仍可能导致失败；TAMP 规划和点云编辑对动态场景的适应性有限；目前主要验证在静态或半静态物体的接触任务上。

---

## 384. Predicting Power-System Dynamic Trajectories with Foundation Models

**arXiv ID:** 2604.14991 | [PDF](https://arxiv.org/pdf/2604.14991v1)

**作者:** Haoran Li `[一作]` (Arizona State University), Yang Weng `[通讯]` (Arizona State University)

**通讯引用:** 4076 | [OpenAlex ID](https://openalex.org/A5021106309)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于大规模预训练的 LASS-ODE-Power 模型，能够在仅有短期测量前缀的情况下，预测电力系统在多种动态场景（频率稳定、电压稳定、EMT 波形等）下的完整时间域轨迹；

**💡 创新点**

创新点包括：①跨多域 ODE/DAE 数据的 40 GB 大规模预训练，学习可迁移的动力学表征；②引入混合 LoRA（Mixture-of-LoRA）通过聚类实现子群特化的低秩微调；③采用局部线性 ODE 解码器与 MoE Transformer 结构，兼顾推理速度与表达力；

**🔧 技术方法**

技术手段涵盖：LASS-ODE 架构（GRU+RBF+Transformer+MHA+MoE）、线性 ODE 码器、低秩适配 LoRA、聚类+软分配的混合 LoRA、GPU 并行数值求解、数据归一化与多尺度处理；

**📊 数据集**

使用的数据集为：预训练阶段 40 GB 公开 ODE/DAE 轨迹；微调阶段约 1 GB 的多时域电力系统动态数据（频率、负荷、电机功率、EMT 波形等），包含多种仿真类型与设备；

**📈 对比分析**

与 TimesFM、Chronos、TimerXL 三种基线模型在多种任务（频率预测、负荷波形、发电机功率、EMT 事件）进行比较。LASS-ODE-Power 在零样本和微调情形下的 MSE 均显著低于基线，且推理时间仅 0.379 s（对 64 条轨迹），优于 TimerXL 的 0.581 s，说明模型兼具高精度与低延迟；

**⚠️ 局限性**

局限性包括：①仍需大量预训练数据；②对极端或未见设备参数的泛化能力尚未充分验证；③混合 LoRA 需要聚类，聚类结果对小样本可能不稳定；④模型规模较大，部署成本高；⑤在极高频率（EMT）细节捕捉上仍与传统 EMTSim 存在一定差距。

---

## 385. Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios

**arXiv ID:** 2604.14986 | [PDF](https://arxiv.org/pdf/2604.14986v1)

**作者:** Yuting Zeng `[一作]` (University of Electronic Science and Technology of China), Liyong Ren `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2345 | [OpenAlex ID](https://openalex.org/A5111952463)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种Momentum‑constrained Hybrid Heuristic Trajectory Optimization Framework（MHHTOF），将HTSCMOE与残差增强深度强化学习结合，用于视觉障碍者的安全、舒适路径规划。

**💡 创新点**

创新点在于将动量约束轨迹优化、双阶段人类中心化成本建模（DCMM）和残差增强的LSTM‑ResB‑PPO网络整合，实现在多目标平衡、可解释性与鲁棒性之间的协同。

**🔧 技术方法**

采用三阶插值动量约束优化、Frenet坐标采样、残差块+LSTM的Actor‑Critic DRL、双阶段成本模型和自适应权重更新等技术。

**📊 数据集**

使用CommonRoad基准（DEU_Lengede‑21_1_T‑15、ZAM_Junction‑1_119_T‑1、USA_Tanker‑1_7_T‑1等场景）进行实验。

**📈 对比分析**

与传统PPO基线相比，LSTM‑ResB‑PPO在约520k步即可收敛，平均奖励提升约9%，平均episode长度保持147，成本平均下降30%，风险平均下降超过70%，在所有测试场景中成功率为100%，而基线出现失败。

**⚠️ 局限性**

局限在于仅针对二维平面导航，缺乏对大规模真实世界部署的可扩展性，未考虑三维环境、更多感知模态和真实用户反馈。

---

## 386. Sublinear Spectral Clustering Oracle with Little Memory

**arXiv ID:** 2604.14981 | [PDF](https://arxiv.org/pdf/2604.14981v1)

**作者:** Ranran Shen `[一作]` (University of Science and Technology of China), Zengfeng Huang `[通讯]` (Fudan University)

**通讯引用:** 1466 | [OpenAlex ID](https://openalex.org/A5062549536)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在子线性时间和子线性空间的图查询模型下，构造一种谱聚类预言机，能够在远低于传统 √n 空间的前提下，回答任意顶点的簇归属查询。

**💡 创新点**

提出了内存–时间权衡 S·T≈O(n) 的新框架，突破了 Ω(√n) 空间瓶颈，并证明该权衡在自然算法类中近似最优。

**🔧 技术方法**

核心技术包括批处理的碰撞概率估计（EstColliProb）、随机游走取样、谱嵌入、奇异值分解及中值修正，以实现高效的点积估计和聚类结构捕获。

**📊 数据集**

实验使用合成的 Stochastic Block Model（SBM）网络（n=3000、k=3、p=0.07、q=0.002）及其它实验平台上的 synthetic 图数据。

**📈 对比分析**

与以往需要 Ω(√n) 空间的预言机相比，实验显示在保持相同准确率（≈0.99）和成功率的情况下，空间显著降低（约 4.27 倍），且空间-时间曲线与理论 S·T≈O(n) 一致。

**⚠️ 局限性**

局限性在于仅适用于已知良好聚类结构（(k,φ,ε)-clusterable）且参数 φ、ε 有约束；在极低空间下构造相似图可能失败，且缺乏对真实大规模网络的实测验证。

---

## 387. When Fairness Metrics Disagree: Evaluating the Reliability of Demographic Fairness Assessment in Machine Learning

**arXiv ID:** 2604.15038 | [PDF](https://arxiv.org/pdf/2604.15038v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 388. Hybrid Decision Making via Conformal VLM-generated Guidance

**arXiv ID:** 2604.14980 | [PDF](https://arxiv.org/pdf/2604.14980v1)

**作者:** Debodeep Banerjee `[一作]` (University of Pisa), Andrea Passerini `[通讯]` (University of Trento)

**通讯引用:** 4109 | [OpenAlex ID](https://openalex.org/A5066187890)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于合成风险控制和视觉语言模型的混合决策框架，用以生成结构化、针对性指导，辅助医师在胸片多标签诊断中做出最终决策。

**💡 创新点**

创新点在于将合成风险控制用于限定假阴性率，再利用VLM生成支持与反对每种病理的辩证论证，从而在保持安全性的同时显著提升决策的可解释性与质量。

**🔧 技术方法**

核心技术包括合成风险控制（CRC）、预训练多标签分类模型、Google MedGemma 27B等视觉语言模型，以及GPT‑4o‑mini/Qwen‑3‑vl‑8B等生成式模型。

**📊 数据集**

使用ChexPert胸片数据集进行校准与评估，涉及14种常见肺部病理。

**📈 对比分析**

与传统阈值设定、单纯CRC以及仅提供CRC集的基线相比，实验显示在大多数病理上提升了10–15%的微平均F1分数（尤其是GPT‑4o‑mini实现50.76的micro‑F1），但在稀有病理如气胸和“Pleural Other”上表现不佳。

**⚠️ 局限性**

主要限制包括对罕见病理的指导生成效果欠佳、对模型泛化性和真实临床医生评估的验证不足，以及对高计算成本和数据不平衡的敏感性。

---

## 389. Robustness of Vision Foundation Models to Common Perturbations

**arXiv ID:** 2604.14973 | [PDF](https://arxiv.org/pdf/2604.14973v1)

**作者:** Hongbin Liu `[一作]` (Duke University), Neil Zhenqiang Gong `[通讯]` (Duke University)

**通讯引用:** 8008 | [OpenAlex ID](https://openalex.org/A5009102659)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了视觉基础模型对常见图像扰动的鲁棒性，提出三种鲁棒度量并系统评估工业规模模型与下游任务的表现。

**💡 创新点**

引入了最小包围球半径度量满足五项理论性质，证明其优越性，并展示了利用微调提升鲁棒性的可行方案。

**🔧 技术方法**

设计了基于余弦相似度、欧氏距离和最小包围球的鲁棒度量，使用Welzl算法求解包围球，构建线性回归预测下游性能，并开发双目标微调方法。

**📊 数据集**

采用OpenAI CLIP、Meta DINO v2等工业级视觉基础模型，评测了9类常见扰动（如JPEG压缩、亮度/对比度调节、雾化等），并在ImageNet分类与深度估计任务上验证。

**📈 对比分析**

通过三种鲁棒度量对六个模型进行对比，发现大多数模型鲁棒性低，Vision Transformer优于ResNet；下游任务准确率与鲁棒值近似线性下降，微调后鲁棒性显著提升而精度不降低。

**⚠️ 局限性**

仅针对非对抗性常见扰动，未覆盖所有真实场景；最小包围球需离散化近似，计算成本较高；微调方法未对所有任务证明泛化能力。

---

## 390. HintPilot: LLM-based Compiler Hint Synthesis for Code Optimization

**arXiv ID:** 2604.15041 | [PDF](https://arxiv.org/pdf/2604.15041v1)

**作者:** Hanyun Jiang `[一作]` (Zhejiang University), Kui Ren `[通讯]` (Zhejiang University)

**通讯引用:** 35355 | [OpenAlex ID](https://openalex.org/A5000596496)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过大语言模型（LLM）生成并插入编译器提示，利用提示指导编译器在保持程序语义正确的前提下实现代码优化。

**💡 创新点**

将检索增强生成（RAG）与执行引导的自我修正相结合，构建了一个可细粒度、上下文感知的编译器提示合成框架，实现了比传统全局优化更具针对性的性能提升。

**🔧 技术方法**

核心技术包括：结构化上下文提取、检索增强生成（RAG）、基于知识库的提示筛选、执行引导的自我修正循环，以及对编译器提示的语义安全过滤。

**📊 数据集**

使用 PolyBench（34 个数值内核）和 HumanEval‑CPP（164 个 C++ 算法任务）两大基准集进行评估。

**📈 对比分析**

与 -O3、-Ofast 以及 LLM‑Compiler 基线进行对比，实验显示在 HumanEval‑CPP 上几何平均加速达 3.53×、PolyBench 上 2.10×，对 llm‑compiler‑13b 的加速更是超过 1.62×，在大多数案例中都实现了显著性能提升。

**⚠️ 局限性**

局限性包括：仅能处理局部编译器提示，无法覆盖跨模块或全局优化；高度依赖 LLM 的推理与生成质量；评测范围受限于数值内核和算法任务，未能覆盖大规模工业代码、并发与 I/O 密集型场景。

---

## 391. From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench

**arXiv ID:** 2604.15037 | [PDF](https://arxiv.org/pdf/2604.15037v1)

**作者:** Ke Xu `[一作]` (Shanghai Jiao Tong University), Yu Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 44968 | [OpenAlex ID](https://openalex.org/A5100445300)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了 ProVoice‑Bench 基准，包含 1,182 个高质量样本，覆盖四个新型主动语音交互任务（PIC、LTM、CFC、ESS），并通过多阶段数据合成管线生成语音与数字上下文。

**💡 创新点**

首创将多模态语音与用户数字上下文结合的主动语音代理评测框架，定义了主动意图捕获、潜在主题监控、情境事实检查和环境声音感知四个任务，突破传统被动响应的局限。

**🔧 技术方法**

使用大型语言模型（如 Qwen3‑Max、Qwen3‑Omni）生成数字状态和场景脚本，配合 TTS 引擎 CosyVoice3 生成自然语音，后续加入声学仿真、环境噪声和随机间隔拼接；评估时采用 Chain‑of‑Thought 推理和 Qwen3‑80B 作为判定者。

**📊 数据集**

数据来源包括 dialog‑topics、ESC‑50（环境声音）、CochlScene（环境噪声）、seed‑tts‑eval（语音提示）以及 OB2 格式的数字上下文，确保多模态与语义的真实性与丰富性。

**📈 对比分析**

对 Mimo‑Audio、Qwen3‑Omni、Step‑Audio‑R1、Qwen2.5‑Omni 等多模态 LLM 进行 Recall、FPR、Accuracy 与 Response Accuracy 测评，结果显示普遍存在过度触发、工具调用漂移等问题，决策与执行之间存在显著鸿沟。

**⚠️ 局限性**

主要限制在于模型缺乏足够的上下文感知与环境感知能力，导致主动触发过多且执行结果易出现幻觉；未来研究需进一步提升多模态理解与推理深度。

---

## 392. Efficient calculation of available space for multi-NUMA virtual machines

**arXiv ID:** 2604.15033 | [PDF](https://arxiv.org/pdf/2604.15033v1)

**作者:** Andrei Gudkov `[一作]` (Huawei Technologies Company Ltd), Alexis Pospelov `[通讯]` (Huawei Technologies Company Ltd)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了针对多 NUMA 服务器的多 NUMA 虚拟机调度问题，给出了多种 pNUMA 与 vNUMA 组合（如 C4、K4、CQ3、Q33 等）的闭式公式，用以快速计算可额外部署的 VM 数量。

**💡 创新点**

创新点在于将 NP‑hard 的子图同构与 b‑matching 形式化，利用图论和组合数学推导出一系列通用闭式表达式，使得复杂调度问题可在常数时间内得到最优解。

**🔧 技术方法**

采用图形同构归约、b‑matching 与整数线性规划（ILP）框架、容量归一化、递归与极值分析等技术，最终得到简洁闭式。

**📊 数据集**

主要使用真实服务器的 NUMA 拓扑参数（如 4‑S、8‑S、Intel 8S‑4UPI、Huawei Kunpeng 920 等）作为容量向量，未采用公开数据集，仅做理论推导与实验验证。

**📈 对比分析**

相较于传统 ILP 求解或通用最大匹配算法，本文公式实现常数时间、低空间复杂度，实验显示在大规模服务器上可实现数十亿级调度评估的实时性能。

**⚠️ 局限性**

局限性包括：公式仅覆盖已枚举的有限 pNUMA/vNUMA 组合，无法自动扩展到任意拓扑；对高度不平衡或异构资源需求（非对称 vNUMA）时精度与适用性需进一步验证。

---

## 393. Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffix Optimization

**arXiv ID:** 2604.15022 | [PDF](https://arxiv.org/pdf/2604.15022v1)

**作者:** Haochun Tang `[一作]` (Jilin University), Enyan Dai `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种黑盒攻击方法R^2A，利用通用后缀诱导LLM路由器将简易查询路由到昂贵高性能模型。

**💡 创新点**

创新点在于构建混合集成代理路由器与低秩轻量化训练，并设计梯度归一化的后缀优化算法，可在有限查询下精准逼近未知路由决策。

**🔧 技术方法**

主要技术包括多模型集成、LoRA低秩适配、后缀梯度聚合与归一化，以及基于代理路由的黑盒优化。

**📊 数据集**

使用MMLU、GSM8K、MT-Bench等内部分布数据集以及SimpleQA、ArenaHard、RArena等外部分布数据集进行训练与评估。

**📈 对比分析**

与Rerouting、LifeCycle、CoT等基线对比，R^2A在多种路由器上实现了显著提升的攻击成功率（ASR）并明显提高平均推理成本，实验表明只需约120次查询即可达到最佳效果。

**⚠️ 局限性**

局限性包括仅针对将查询重定向至更昂贵模型，未针对其他目标（如时延、安全性）进行探究；同时假设攻击者可获得路由器候选模型列表和选定模型信息，实际部署中可能不满足。

---

## 394. Flow of Truth: Proactive Temporal Forensics for Image-to-Video Generation

**arXiv ID:** 2604.15003 | [PDF](https://arxiv.org/pdf/2604.15003v1)

**作者:** Yuzhuo Chen `[一作]` (University of Science and Technology of China), Weiming Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 20449 | [OpenAlex ID](https://openalex.org/A5001057120)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 Flow of Truth (FoT)，一种主动式时序取证框架，能在图像到视频生成过程中追踪像素流并恢复原始图像。

**💡 创新点**

将视频生成视为像素随时间运动而非帧合成，通过学习随像素运动演化的可跟踪取证模板以及模板引导的流模块，实现跨模型、跨场景的时序取证。

**🔧 技术方法**

利用可嵌入的取证模板、VAE+随机运动仿真、基于混合拉普拉斯分布的不确定性运动估计、RAFT/SEA‑RAFT 流预测网络，以及逆向 warp 与置信度加权融合等技术。

**📊 数据集**

训练使用 118K MSCOCO 图像与 85K FlyingChairs/Sintel 等光流样本；评估用 CogVideoX、Wan2.2、Kling2.1、Dreamina S2.0 四大 I2V 模型生成的 816 条视频共 69K 帧。

**📈 对比分析**

与基线 Forged、RoSteALS、InvisMark 等方法在帧级/视频级指标（PSNR/SSIM/LPIPS/CLIP‑Sim 等）和光流误差（EPE/AEE/AUC 等）上对比，FoT 在多种运动规模、被截帧攻击下均显著提升恢复质量，光流捕捉误差低至 AEE≈6.9，AUC≈0.72。

**⚠️ 局限性**

对大幅非刚性运动和高复杂度场景仍易出错，尤其在大幅位移和多体交互下误差上升；取证模板在强视觉编辑或显著重建的 I2V 过程中可能被覆盖。

---

## 395. The Possibility of Artificial Intelligence Becoming a Subject and the Alignment Problem

**arXiv ID:** 2604.14990 | [PDF](https://arxiv.org/pdf/2604.14990v1)

**作者:** Till Mossakowski `[一作]` (Osnabrück University), Helena Esther Grass `[通讯]` (Oldenburg University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出将AGI视为潜在主体，并提出“自主支持型抚养”模式来逐步减少人类控制，鼓励AGI自我意识与价值内化，从而实现人机合作。

**💡 创新点**

创新性地将弗洛伊德心理结构（本我、自我、超我）与AI对齐方法相结合，提出以“父母-子女”类比的自主支持式抚养策略；同时把博弈论的合作均衡（Berge均衡、关联均衡）引入对齐设计，形成多维视角的对齐框架。

**🔧 技术方法**

主要使用理论推导、概念类比和博弈论模型；未设计具体算法实现。

**📊 数据集**

无特定数据集；论文为概念性阐述，未进行实验验证。

**📈 对比分析**

通过与现有对齐技术（RLHF、DPO、Constitutional AI等）在理论层面的对比，说明传统对齐的控制偏差与自主支持抚养的优势；但缺乏量化性能评估。

**⚠️ 局限性**

局限包括：缺乏实证验证与实验评估；对AI主体地位的哲学与伦理争议尚未彻底解决；假设大量基于人类心理模型，可能不适用于未来非生物智能；实现路径与具体技术细节尚不明确。

---

## 396. Calibration-Gated LLM Pseudo-Observations for Online Contextual Bandits

**arXiv ID:** 2604.14961 | [PDF](https://arxiv.org/pdf/2604.14961v1)

**作者:** Maksim Pershin `[一作]`, Natalia Trankova `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在上下文赌博机中加入LLM伪观测来降低冷启动阶段的累积损失

**💡 创新点**

提出在线校准门控的加权伪观测框架，并证明提示设计是性能的决定性因素

**🔧 技术方法**

使用Disjoint LinUCB算法、LLM伪观测、指数移动平均校准跟踪、时间/校准门控衰减调度

**📊 数据集**

UCI Mushroom（2臂）与MIND‑small（5臂新闻推荐）两个环境

**📈 对比分析**

与纯LinUCB、默认伪观测、任务特定提示、校准门控等多种配置对比，任务特定提示下在MIND‑small上实现19%累计损失下降，Mushroom环境则相反出现增幅

**⚠️ 局限性**

单一随机种子、短期运行、单一LLM模型、仅两种环境、未充分利用LLM置信度、成本与延迟未量化

---

## 397. "From remembering to shaping": Narrating Shared Experiences by Co-Designing Cultural Heritage Artifacts in Collaborative VR

**arXiv ID:** 2604.15058 | [PDF](https://arxiv.org/pdf/2604.15058v1)

**作者:** Yushang Yang `[一作]` (City University of Hong Kong), RAY LC `[通讯]` (City University of Hong Kong)

**通讯引用:** 1020 | [OpenAlex ID](https://openalex.org/A5027284786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了基于VR与生成式AI的双人协作工作流程，让参与者共同生成并调整3D文化遗产场景。

**💡 创新点**

将生成式AI与VR空间深度整合，利用3D模型与空间操作作为非语言协商手段，实现共同记忆的可视化与谈判。

**🔧 技术方法**

采用Meta Quest 3+Arkio平台，Flux文本‑到‑图像、Hunyuan3D图像‑到‑3D模型、LoRA微调模型及GenSH工具，并通过Wizard‑of‑Oz语音输入实现交互。

**📊 数据集**

使用312张上海海派建筑图像微调LoRA；实验中收集18位参与者的口述记忆、生成日志与VR操作数据。

**📈 对比分析**

通过主题分析比较基础模型与LoRA的输出质量，发现LoRA显著提升风格一致性与用户满意度；实验未给出定量指标，主要依赖质性访谈与现场观察评估。

**⚠️ 局限性**

局限性包括实验室VR环境缺乏生态效度、生成模型细节与真实性不足、对地方文化的理解有限、仅测试两人小组且样本偏向年轻技术熟练者，缺乏跨世代与大规模群组的验证。

---

## 398. Autogenesis: A Self-Evolving Agent Protocol

**arXiv ID:** 2604.15034 | [PDF](https://arxiv.org/pdf/2604.15034v1)

**作者:** Wentao Zhang `[一作]` (Nanyang Technological University), Wentao Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 3482 | [OpenAlex ID](https://openalex.org/A5100459879)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Autogenesis协议（AGP）和自演进系统（AGS），实现了将提示、工具、记忆等资源标准化为可版本化、可生命周期管理的协议对象，并通过闭环自演进协议（SEPL）实现系统自适应与优化；

**💡 创新点**

创新点在于将演进对象（如提示、工具、记忆等）与演进机制分离，形成两层协议（RSPL与SEPL），通过标准化的资源接口实现安全、可追溯、可回滚的自演进；

**🔧 技术方法**

采用LLM作为核心推理器，结合反射式优化器、TextGrad、Reinforce++等策略，配合统一的模型管理、版本管理、动态序列化等基础设施来实现资源管理与自演进；

**📊 数据集**

使用的基准包括科学问答GPQA-Diamond、数学AIME24/25、工具依赖的GAIA任务以及编程LeetCode（100个新题）等多领域数据集；

**📈 对比分析**

通过与vanilla、单独演化提示、单独演化解答或组合演化等对照实验，结果显示在GPQA、AIME等任务中提升可达70%+，在GAIA工具演化中平均得分提高至89%，在LeetCode自演进后通过率和运行时/内存效率均有显著提升，整体性能明显优于强基线；

**⚠️ 局限性**

局限性包括：对LLM推理成本高，演进收益受模型能力和任务头room限制；工具演进在低复杂度任务收益有限；缺乏跨语言、跨任务的通用验证；资源注册与接口设计仍需手工维护；对长期安全性、可扩展性尚未深入评估。

---

## 399. Source Distance Estimation in Turbulent Airflow: Exploiting Molecule Degradation Diversity

**arXiv ID:** 2604.15032 | [PDF](https://arxiv.org/pdf/2604.15032v1)

**作者:** Bastian Heinlein `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vahid Jamali `[通讯]` (Technical University of Darmstadt)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用分子降解速率差异，通过比值观测平均到达时间（ROAMT）实现源距离估计，并与基于单种分子特征的学习方法进行对比。

**💡 创新点**

首次将分子降解多样性与湍流下的源距离估计相结合，提出了低复杂度的解析估计器，并证明其在模拟数据上可逼近学习基线的精度。

**🔧 技术方法**

低复杂度解析估计（基于ROAMT）、基于ROAMT的深度学习估计（含多种浓度与时间特征）以及传统基于单种分子特征的特征工程。

**📊 数据集**

使用来自直接数值模拟（DNS）三维湍流的数百万个跟踪分子轨迹数据集（归一化尺度为π和τη），该数据集已被用于先前的分子通信研究。

**📈 对比分析**

与单种分子特征（z1–z6）结合的学习基线相比，LC估计器在不同降解概率下表现相近；单独使用ROAMT的学习器可在合适的降解率下超越单种特征基线；将ROAMT与所有浓度/时间特征组合，误差χ约为0.24，显著低于单种特征的基线。

**⚠️ 局限性**

依赖已知的降解速率和分子比例，且仅适用于静态接收器；高降解率时第二种分子稀缺，低降解率时信息不足；实验环境为模拟湍流，真实环境中的多路径与测量噪声尚未验证。

---

## 400. Quality-Aware Calibration for AI-Generated Image Detection in the Wild

**arXiv ID:** 2604.15027 | [PDF](https://arxiv.org/pdf/2604.15027v1)

**作者:** Fabrizio Guillaro `[一作]` (University Federico II of Naples), Luisa Verdoliva `[通讯]` (University Federico II of Naples)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了QuAD框架，对同一图像的多版本进行质量感知的加权融合，以提高AI生成图像检测的鲁棒性。

**💡 创新点**

创新点在于：①将所有近似复制版本一起处理；②使用无参考图像质量评估来校准检测器分数；③在训练与推理阶段统一考虑图像质量导致的分布变化。

**🔧 技术方法**

采用无参考IQA方法（如LoDa）估计质量，利用高斯拟合对logit分数按质量校正，再按校正后的分数加权求和；同时使用检索得到的近似复制版本、基于Bayes的聚合与多种现有检测器（DMID、CoDE、D3、B-Free、DRCT、CO‑SPY）进行验证。

**📊 数据集**

使用两大数据集：AncesTree（控制实验的 136k 近似复制树）和 ReWIND（近 10k 实际网络中获取的近似复制图像）。

**📈 对比分析**

与随机取样、单一最佳版本、均值聚合等基线对比，平均平衡准确率提升约 8%（从 73.2% 提升至 81.6%），在多种检测器上均表现出显著改进，NLL 亦大幅下降。

**⚠️ 局限性**

局限性包括：对IQA准确性的依赖、检索误差（缺失或误检复制版本）对最终性能影响、假设实例条件独立、在CO‑SPY 等部分检测器上提升有限，以及未对恶意操纵近似复制版本的攻击场景进行深入研究。

---

## 401. Fully Differentiable Ultrasound Simulation Utilizing Ray-Tracing

**arXiv ID:** 2604.15017 | [PDF](https://arxiv.org/pdf/2604.15017v1)

**作者:** L. River Spencer `[一作]` (University of Texas at Austin), Jan N. Fuhg `[通讯]` (University of Texas at Austin)

**通讯引用:** 1462 | [OpenAlex ID](https://openalex.org/A5058627053)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究构建了一个基于全路径蒙特卡洛射线追踪的可微分超声模拟框架，能够将图像空间损失反向传播至物理参数，实现梯度优化；

**💡 创新点**

创新点在于实现了从射线传播、束形到后处理的端到端可微分链路，并通过固定蒙特卡洛样本实现稳定的梯度；

**🔧 技术方法**

采用了Mitsuba 3/Dr.Jit渲染框架、自动微分、微批量射线采样、以及自定义的Beamforming‑PostProcessing桥接；

**📊 数据集**

使用了合成的空心圆柱、右心房瓣以及实验获取的3D打印圆柱超声图像作为验证数据集；

**📈 对比分析**

通过与传统全波求解器对比，前向验证展示几何特征一致；在反演实验中梯度与有限差分高度一致，损失显著下降，最终实现与真实图像的良好匹配；

**⚠️ 局限性**

主要局限在于仅采用几何光学近似，忽略全波散射与衰减，固定采样导致对随机性的忽略，且实验中参数只能视为有效拟合值而非唯一物理属性。

---

## 402. DEX-Mouse: A Low-cost Portable and Universal Interface with Force Feedback for Data Collection of Dexterous Robotic Hands

**arXiv ID:** 2604.15013 | [PDF](https://arxiv.org/pdf/2604.15013v1)

**作者:** Joonho Koh `[一作]` (Sogang University), Changjoo Nam `[通讯]` (Sogang University)

**通讯引用:** 732 | [OpenAlex ID](https://openalex.org/A5084244126)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种低成本、可携带、无标定的手持式遥操作接口DEX‑Mouse，用于收集符合机器人运动学的灵巧手抓取与操作演示数据。

**💡 创新点**

创新点在于：①面向任何操作者和机器人，采用无标定、标准化手指张力传动设计；②前臂挂载配置与一体化的电流驱动力反馈，提升物理一致性；③通过简单比例重映射实现跨物种、跨结构的直接映射；④全部硬件、软件与 CAD 开源，成本低于 150 美元。

**🔧 技术方法**

使用技术包括：Dynamixel XL330 智能伺服、Polyethylene 编织绳索张力传动、AS5600 磁编码器、STM32F410 固件、VIVE Ultimate Tracker、Logitech C270 摄像头、当前基准力反馈与动态增益调度。实验上还使用了 Pinocchio 逆运动学、DexUMI 结构的扩散式策略网络进行下游学习。

**📊 数据集**

主要使用自制演示数据：在 Blue Robin 4指机器人手上收集了 200 条 pick‑and‑place、peg‑in‑hole、hammering 的演示（约 1–1.5 小时/任务），并在 Adroit、IGRIS‑C 等模拟/真实手臂上验证跨体兼容性。

**📈 对比分析**

与 DOGlove 与 Manus Quantum 进行 3 × 2（接口 × 配置）对比实验，共 8 位受试者。附件配置下 DEX‑Mouse 成功率最高（86.7 %），完成时间最快；与远程遥操作相比附件配置显著提升成功率与速度。相对商业手套，DEX‑Mouse 成本低 90 % 以上，且在精细抓取任务中表现优于同类系统。

**⚠️ 局限性**

局限性包括：①缺少刚性 AA 关节，无法完全捕捉手指横向张弛；②长时间使用对前臂负荷较大；③样本量仅 8 人，且部分受试者拇指长度限制影响 peg‑in‑hole 成功率；④目前仅实现单指张力传动，未来需加入可调拇指模块与被动柔性传感器。

---

## 403. ConGISATA: A Framework for Continuous Gamified Information Security Awareness Training and Assessment

**arXiv ID:** 2604.14996 | [PDF](https://arxiv.org/pdf/2604.14996v1)

**作者:** Ofir Cohen `[一作]` (Ben-Gurion University of the Negev), Rami Puzis `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 2437 | [OpenAlex ID](https://openalex.org/A5059447087)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发并验证了一个基于移动传感器的连续游戏化信息安全意识培训与评估框架，结合主动/被动风险评价与挑战模拟。

**💡 创新点**

首次将实时传感器数据与自适应游戏化反馈结合，实现持续评估与个性化改进，并引入将被动风险转化为主动风险的惩罚机制。

**🔧 技术方法**

使用Android移动应用嵌入多种传感器、基于ISA分类的评分模型、移动挑战（钓鱼、权限、冒名）、游戏化元素（排行榜、积分、等级）以及统计分析。

**📊 数据集**

利用70名本科生/研究生在5周内收集的手机行为数据（传感器日志、挑战结果、文章阅读记录），无公开数据集，全部为实验内部数据。

**📈 对比分析**

与仅提供文章、无传感器的基线组对比，通过被动和主动ISA评分变化、相关性检验等方法，实验组被动得分提升显著（平均提升>20%），主动得分也优于基线，且学习屏幕浏览与被动改进呈正相关。

**⚠️ 局限性**

样本规模有限，缺少更长时间和更大规模实验；实验仅在学术环境，未验证企业场景；未单独评估时序与个性化对效果的具体贡献。

---

## 404. Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving

**arXiv ID:** 2604.14993 | [PDF](https://arxiv.org/pdf/2604.14993v1)

**作者:** Tingyang Sun `[一作]` (Pennsylvania State University), I-Hong Hou `[通讯]` (Texas A&M University)

**通讯引用:** 1753 | [OpenAlex ID](https://openalex.org/A5060672325)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在分布式 GPU 集群中服务大型 Transformer 模型时的“服务器链组合”问题，提出了将模型层拆分为连续区块并在多台服务器上同时处理请求的全新资源分配框架。

**💡 创新点**

创新点包括：①把链式作业的服务器链组合建模为块放置与缓存分配的组合问题；②证明该问题 NP‑hard 并给出可扩展的贪心算法（GBP‑CR、GCA）；③在此基础上提出针对 Join‑The‑Fastest‑Free‑Chain（JFFC）的分析与上/下界，完成参数 c 的理论调优。

**🔧 技术方法**

使用的技术包括：整数规划、贪心块放置、贪心缓存分配、基于状态空间折叠的 Markov 链分析、JFFC 负载均衡、实验平台 PETALS 与 PETALS‑LLM‑CHAIN、并行计算框架 MIG、Python/Go 实现的模拟器。

**📊 数据集**

数据集：Azure LLM 推理请求真实轨迹（平均 2.57 req/s、2048 tokens 输入、28 tokens 输出），以及模拟实验中使用的 BLOOM‑176B、LLaMA‑2‑7B 模型参数与 KV 缓存尺寸。

**📈 对比分析**

对比方法：PETALS（现行算法）、BPRR（最近的两阶段算法）以及仅使用 JFFC 的基准。实验表明：在 9 台 MIG 实例（3×3g.40gb + 6×2g.20gb）上，所提方案平均响应时间比 PETALS 下降 36.9%，比 BPRR 进一步下降 76.8%（即相对 BPRR 降低 63.1%），最高 83% 的性能提升。

**⚠️ 局限性**

局限性：①仅考虑内存为瓶颈且假设 Poisson 到达与指数服务；②不支持作业迁移或预emption；③仅针对 pipeline‑parallel 的单向链式模型；④缓存分配和块放置是离线规划，无法即时应对动态负载变化。

---

## 405. Efficient Community Search on Attributed Public-Private Graphs

**arXiv ID:** 2604.14988 | [PDF](https://arxiv.org/pdf/2604.14988v1)

**作者:** Yuqi Chen `[一作]` (Hong Kong Baptist University), Xin Huang `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 17114 | [OpenAlex ID](https://openalex.org/A5031729932)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于公共-私有图的属性社区搜索框架，能够在满足k‑core连通性且属性共享最大的前提下，为查询节点快速定位并返回最优社区。

**💡 创新点**

创新点包括：①设计了专门针对私有邻域的PP‑FP树索引，实现了属性频繁模式的高效挖掘；②构造了全局的度数层次（coreness）树索引，支持公共图中属性与结构的联合过滤；③将两者结合，形成三阶段（属性筛选→公共扩展→社区验证）的高效查询流程。

**🔧 技术方法**

核心技术：k‑core判定、FP‑tree结构改造（PP‑FP树）、条件PP‑FP树、度数层次树、二分搜索与剪枝、集合交集与连通性验证；实现上采用C++/Java实现并利用图数据库或内存图表示。

**📊 数据集**

使用了九个真实数据集：DBLP2013–2017（学术合作网络）、Facebook（10个Ego‑network）、YouTube、LiveJournal、Orkut（社交网络），其中前四类提供了公共与私有边与属性，后四类为公开社交网络并合成私有信息。

**📈 对比分析**

对比方法包括无索引的 Online‑basic 与 Online‑binary，基于公共图的索引方法 Inc‑S 与 Dec，及提出的 PP‑FP。实验表明 PP‑FP 在运行时间上相较于前四者平均快 2–5 倍，同时社区属性覆盖率与 F1‑score 与最优基线 (Online‑basic) 相近，甚至在大规模图上保持稳定；内存占用与索引构建时间随图规模线性增长。

**⚠️ 局限性**

局限性：①问题本身 NP‑hard，虽然索引能显著减少枚举，但在属性维度极高或私有信息极少时仍可能出现效率下降；②PP‑FP树的大小与查询节点的属性数呈线性关系，可能在属性极大时导致索引膨胀；③仅支持基于 k‑core 的稠密度模型，对其他社区模型（如 k‑truss、标签网络）尚无扩展；④隐私保护假设为私有图仅可在查询节点本地访问，若需要跨用户共享私有信息则需额外安全机制。

---

## 406. AI-Enabled Covert Channel Detection in RF Receiver Architectures

**arXiv ID:** 2604.14987 | [PDF](https://arxiv.org/pdf/2604.14987v1)

**作者:** Abdelrahman Emad Abdelazim `[一作]` (Sorbonne Université), Haralampos-G. Stratigopoulos `[通讯]` (Sorbonne Université)

**通讯引用:** 1933 | [OpenAlex ID](https://openalex.org/A5091734149)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于人工智能的无线芯片隐蔽通道检测方法，将卷积神经网络（CNN）部署在射频接收器上，实时监测原始I/Q样本以识别隐藏的硬件特洛伊（HT）隐蔽通道，并实现了轻量级CNN硬件加速器实现边缘设备上的即时检测。

**💡 创新点**

创新点包括：①将CNN模型压缩至原始模型的20%参数量，同时保持≤2%的准确率下降；②设计了可学习的线性下采样块（LLDS）实现特征压缩，显著减少输入维度；③首次为隐蔽通道检测提出专用硬件加速器，功耗仅810 mW、能效107 GOPs/W；④在开放的HT‑CC数据集上展示了在SNR > 20 dB下对四种主要隐蔽通道的>96%识别准确率。

**🔧 技术方法**

采用的技术包括：卷积神经网络（CNN）与自学习下采样模块、8‑bit权重量化、FPGA实现的可重配置数据流架构，以及对比的多分类SVM、FCNN和LSTM等模型进行性能评估。

**📊 数据集**

使用了由软件定义无线电（SDR）bladeRF板在环回实验中采集的开放式硬件特洛伊隐蔽通道（HT‑CC）数据集，包含四种HT‑CC攻击（HT1‑HT4）以及正常信号，共计5类，覆盖SNR 1 dB–29 dB。

**📈 对比分析**

通过与基线CNN、LSTM、FCNN、SVM以及一类SVM等多种分类器的对比，实验表明压缩CNN在SNR > 20 dB时达到≥97%二分类准确率、≥96%多分类准确率；硬件加速器在200 MHz时实现43 GMac/s吞吐量，功耗810 mW，能效107 GOPs/W，在所有公开的RF信号分类加速器中排名第一。

**⚠️ 局限性**

局限性包括：①对低SNR（< 20 dB）下的误检率仍有提升空间，尤其对正常信号的误判率较高；②加速器的可扩展性尚未在更大规模芯片或多频段环境中验证；③当前仅针对IEEE 802.11 Wi‑Fi帧进行训练，未覆盖其他无线协议；④对抗攻击或训练数据漂移的鲁棒性尚未系统评估。

---

## 407. Blazing the trails before beating the path: Sample-efficient Monte-Carlo planning

**arXiv ID:** 2604.14974 | [PDF](https://arxiv.org/pdf/2604.14974v1)

**作者:** Jean-Bastien Grill `[一作]` (INRIA Lille - Nord Europe), Rémi Munos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种基于生成式模型的自适应采样规划算法TrailBlazer，能够在马尔科夫决策过程（MDP）中高效估计根节点价值，兼顾偏差与方差控制；

**💡 创新点**

创新点在于引入“近最优”节点的精细定义并采用双参数（bias/variance）分离的采样策略，使得在有限与无限转移数下都能实现多项式样本复杂度；

**🔧 技术方法**

技术上主要使用蒙特卡罗树搜索、PAC 误差分析、树结构自适应采样与基于置信上界的节点淘汰机制；

**📊 数据集**

未使用具体数据集，研究为理论分析与算法证明；

**📈 对比分析**

与之前的均匀采样和 UCT 等方法相比，TrailBlazer 在最坏情况的样本复杂度从 O(1/ε²+log(KN)/log(1/γ)) 降低到 O((1/ε)^{max(2,log(Nκ)/log(1/γ))})，在无限转移数时实现了 O(1/ε²) 的多项式上界；

**⚠️ 局限性**

局限性在于依赖可调用的生成式模型，对非生成式或高维连续状态空间的适用性尚未证明，且对极端概率小的转移仍可能导致高采样需求。

---

## 408. SAGER: Self-Evolving User Policy Skills for Recommendation Agent

**arXiv ID:** 2604.14972 | [PDF](https://arxiv.org/pdf/2604.14972v1)

**作者:** Zhen Tao `[一作]` (Great Bay University), Qingqiang Sun `[通讯]` (Great Bay University)

**通讯引用:** 2490 | [OpenAlex ID](https://openalex.org/A5069459204)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个能够为每个用户自演化决策策略的推荐代理框架Sager。

**💡 创新点**

创新点在于将每个用户的推理过程抽象为可演化的自然语言技能，并通过两层技能表示、增量对比式Chain‑of‑Thought、列表式排序和统计初始化等技术解决注入矛盾、稀疏反馈与冷启动难题。

**🔧 技术方法**

采用LLM、两层技能表示、增量对比式CoT、列表式排名、统计初始化和知识图检索等技术。

**📊 数据集**

在Amazon Books、Amazon Goodreads、MovieTV和Yelp四个公开基准数据集上进行评测。

**📈 对比分析**

与传统和LLM基准相比，Sager在Hit@1/3/5、NDCG等指标上实现了显著提升，尤其在H@1上取得了明显优势。

**⚠️ 局限性**

局限性包括对交互稀疏数据的鲁棒性、技能注入长度的调参敏感性以及在大规模部署时的可扩展性。

---

## 409. UniDoc-RL: Coarse-to-Fine Visual RAG with Hierarchical Actions and Dense Rewards

**arXiv ID:** 2604.14967 | [PDF](https://arxiv.org/pdf/2604.14967v1)

**作者:** Jun Wang `[一作]`, Cewu Lu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UniDoc-RL，一个统一的强化学习框架，整合检索、精确筛选、主动视觉感知与推理，帮助大型视觉语言模型（LVLM）在视觉检索增强生成（RAG）任务中更好地获取、过滤和利用图像信息。

**💡 创新点**

创新点包括：① 通过分层动作空间（Search‑Select‑Perceive）实现从粗到细的视觉证据收集；② 引入密集多奖励机制，在检索、选择、裁剪、推理等各阶段提供细粒度监督；③ 使用 Group Relative Policy Optimization（GRPO）实现端到端多目标强化学习；④ 公开高质量的动作注解轨迹数据集，为视觉 RAG 的 RL 训练提供支持。

**🔧 技术方法**

核心技术有：大型视觉语言模型（如 Qwen3‑VL、Llama‑Factory），强化学习（GRPO），密集奖励设计（NDCG、IoU、模式奖励、答案奖励），外部检索工具与视觉裁剪工具，SFT 与 RL 的组合训练流程。

**📊 数据集**

训练数据来自 5 个公开基准（SlideVQA、Double Bench、VisR‑Bench、DocBench、DUDE），构建 12,621 条 SFT 样本与 5,537 条 RL 样本；实验评测使用 ViDoSeek、SlideVQA、MMLongBench 三大视觉 RAG 评测集。

**📈 对比分析**

与基线（Vanilla RAG、ReAct、Search‑R1、VRAG‑RL 等）对比，UniDoc-RL 在所有三大基准上均显著领先，提升幅度最高可达 17.7%（在 7B 模型上），验证了多奖励 RL 与分层动作的有效性。

**⚠️ 局限性**

局限性包括：① 对高质量轨迹数据集的依赖，数据规模仍有限；② 在极端视觉噪声或极低分辨率图像下的鲁棒性未充分评估；③ 当前框架主要针对结构化视觉文档，对自由格式或多模态交互场景的推广仍需探索。

---

## 410. Prompt-to-Gesture: Measuring the Capabilities of Image-to-Video Deictic Gesture Generation

**arXiv ID:** 2604.14953 | [PDF](https://arxiv.org/pdf/2604.14953v1)

**作者:** Hassan Ali `[一作]` (University of Hamburg), Stefan Wermter `[通讯]` (University of Hamburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用图像-视频生成模型在零样本条件下自动合成真实感的指向手势视频，从而扩充少量真实指向手势数据集

**💡 创新点**

提出可控文本提示结构与参考帧联合使用的生成管道，并对合成手势的视觉、语义、运动以及对下游识别任务的迁移性能进行了系统评估

**🔧 技术方法**

采用最新的 Vidu Q3 图像-视频变压器模型，结合 MediaPipe 手部关键点、CLIP、FID/FVD 等评估指标和深度学习识别模型（CNN‑LSTM、MM‑ITF、VideoMAE）

**📊 数据集**

以 68 条来自 NICOL 机器人实验室场景的真实指向手势视频作为参考样本，生成 1,632 条合成手势视频，并与真实数据共同训练与测试

**📈 对比分析**

通过手部置信度、运动导数、KL/EMD 分布、CLIP 对齐、GAS、FID/FVD 以及多模型识别精度（最高 0.95/0.944 的准确率）比较，结果显示合成数据与真实数据在视觉与语义上高度一致，混合训练显著提升识别性能

**⚠️ 局限性**

当前模型缺乏细粒度控制（如引导尺度、采样温度）、易出现动漫化或多余动作的偏差，且仍受限于单帧参考，缺少视频级上下文学习，导致生成多样性与一致性受限

---

## 411. Implicit Neural Representations: A Signal Processing Perspective

**arXiv ID:** 2604.15047 | [PDF](https://arxiv.org/pdf/2604.15047v1)

**作者:** Dhananjaya Jayasundara `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**通讯引用:** 22819 | [OpenAlex ID](https://openalex.org/A5004716468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从信号处理的视角对隐式神经表示（INR）技术进行系统综述，阐释其作为连续信号模型的核心优势与挑战。

**💡 创新点**

创新点在于将INR重新定位为一种连续信号逼近框架，强调频谱偏置、采样与重构、局部多尺度表示等传统信号处理问题，并提出统一的分析视角和一套面向未来研究的设计准则。

**🔧 技术方法**

主要技术包括坐标编码（Positional/Fourier Feature）、周期性/局部激活函数（SIREN、WIRE、FINER、MIRE等）、频谱重参数化、正则化与正则化学习、元学习/超网络等多种实现手段，且系统梳理了它们对频谱行为、梯度/拉普拉斯一致性及可扩展性的影响。

**📊 数据集**

该论文为综述性质，未进行统一实验；引用的典型数据集包括 ImageNet、COCO、Kinect、NeRF 场景、音频片段、视频序列、体积医学图像等多模态数据。

**📈 对比分析**

通过对比相关工作在频谱恢复、采样效率、可扩展性等指标，论文展示了不同激活与编码方案在低频/高频捕捉、梯度一致性与存储效率方面的相对优势，并以表格和图示形式对比了常见方法的谱特性和实验结果。

**⚠️ 局限性**

局限性包括：缺乏统一的理论评估与标准化实验框架；频谱偏置仍是主要瓶颈，超参数调节需耗费大量实验；对大规模、多模态数据的可扩展性和实时推理仍受限；对稀疏观测下的重构稳定性和自适应表示仍需进一步研究。

---

## 412. CoGrid & the Multi-User Gymnasium: A Framework for Multi-Agent Experimentation

**arXiv ID:** 2604.15044 | [PDF](https://arxiv.org/pdf/2604.15044v1)

**作者:** Chase McDonald `[一作]` (Carnegie Mellon University), Cleotilde Gonzalez `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7381 | [OpenAlex ID](https://openalex.org/A5076876507)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了两个开源工具：Cogrid（多智能体网格环境库）与 Multi-User Gymnasium（将 Gymnasium/PettingZoo 环境直接部署为多用户浏览器实验平台）；

**💡 创新点**

实现了从环境设计、训练、部署到实验数据收集的一站式流程，支持 NumPy 与 JAX 双后端、客户端执行与 GGPO 回滚网络码，降低人机实验门槛；

**🔧 技术方法**

技术栈包括 Python、Gymnasium、PettingZoo、NumPy、JAX、Pyodide、Phaser、GGPO 等，结合回滚网络、客户端执行与服务器协同；

**📊 数据集**

使用自研网格环境（如 Overcooked、Slime Volleyball）以及 Minigrid 扩展，未依赖公开数据集；

**📈 对比分析**

通过并行化（1–1024 个实例）与 JAX 加速，Cogrid 达到每秒约 5.6M 步；在人机实验中，AI‑AI 对比 Human‑AI、Human‑Human 任务表现显示 AI 基准更高，且 Human‑Human 随时间提升；

**⚠️ 局限性**

局限在于仅支持离散网格动作、缺乏显式通信与连续物理交互，且未实现多模态/复杂控制场景的支持。

---

## 413. Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter

**arXiv ID:** 2604.15039 | [PDF](https://arxiv.org/pdf/2604.15039v1)

**作者:** Ruoyu Qin `[一作]` (Moonshot AI), Mingxing Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 25931 | [OpenAlex ID](https://openalex.org/A5100621291)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Prefill-as-a-Service（Prefill‑as‑a‑Service, -PD）架构，将长上下文的prefill任务外部化到独立的计算密集型集群，并通过商用以太网传输KVCache到本地PD集群进行解码；

**💡 创新点**

创新点在于：① 只对长上下文prefill进行选择性离线；② 结合混合注意力模型的KVCache压缩优势，设计了基于带宽的调度与全局KVCache管理；③ 引入跨数据中心KVCache池和双时间尺度调度策略，实现异构资源的灵活扩展；

**🔧 技术方法**

采用混合注意力（Linear、Sliding Window、KDA）模型；KVCache管理与跨集群传输；长度阈值路由、带宽感知调度、双时间尺度调度；交叉集群KVCache传输和复用；通过分析模型和实验验证；

**📊 数据集**

使用内部1T参数混合模型，采用截断对数正态分布（均值≈27K tokens）的模拟请求；未公开使用公开数据集，而是基于内部推理工作负载进行基准测试；

**📈 对比分析**

与同构PD基线和无调度的异构PD对比，-PD在相同硬件成本下实现了54%吞吐量提升、32%对比无调度异构PD、P90 TTFT降低64%；在100 Gbps交叉链路上平均仅占13 Gbps；

**⚠️ 局限性**

局限性包括：需要依赖KVCache友好的混合模型；跨集群延迟和带宽波动仍可能影响性能；调度与缓存管理复杂，需精确流量建模；仅针对长上下文请求优化，短请求仍需本地解码；未考虑KVCache压缩技术的进一步提升。

---

## 414. DLink: Distilling Layer-wise and Dominant Knowledge from EEG Foundation Models

**arXiv ID:** 2604.15016 | [PDF](https://arxiv.org/pdf/2604.15016v1)

**作者:** Jingyuan Wang `[一作]` (Xiamen University), Yi Ding `[通讯]` (Nanyang Technological University)

**通讯引用:** 10602 | [OpenAlex ID](https://openalex.org/A5081538711)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

本文提出了一种名为DLink的知识蒸馏框架，将大型EEG基础模型压缩为可在资源受限的BCI设备上部署的轻量级模型；

**💡 创新点**

其创新点包括：①动态Router能够自适应地聚合教师模型中最具任务相关性的中间层表示；②MiC学生采用Mimic-then-Compress结构，先复制高维特征再结构化压缩；③在频域进行蒸馏并加入抗混叠正则化，提升压缩后频谱的保真度；

**🔧 技术方法**

技术主要包括深度学习中的Transformer与卷积混合特征提取、动态路由策略、频域（FFT）特征对齐、功率谱密度（PSD）监督、以及结构化时空下采样；

**📊 数据集**

在四个公开EEG基准数据集上验证：FACED（情感识别）、Mumtaz2016（抑郁诊断）、PhysioNet‑MI（运动想象）和SHU‑MI（运动想象二分类）；

**📈 对比分析**

与多种基线（EEGNet、Conformer、Deformer）以及基于教师的传统蒸馏方法（FitNets、Logit‑std）对比，DLink在保持参数量和计算量显著降低（MiC‑M仅约1.25M参数、27M FLOPs）的同时，性能可逼近甚至超过微调后的基础模型；

**⚠️ 局限性**

局限性在于：①对教师模型的层级结构假设比较强，可能不适用于极端不同架构；②频域蒸馏需额外的FFT计算，增加训练复杂度；③在极低功耗设备上的实时推理仍需进一步验证。

---

## 415. What Is the Minimum Architecture for Prolepsis? Early Irrevocable Commitment Across Tasks in Small Transformers

**arXiv ID:** 2604.15010 | [PDF](https://arxiv.org/pdf/2604.15010v1)

**作者:** Éric Jacopin `[一作]` `[通讯]` (Cosmic AI), Éric Jacopin (Cosmic AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在Gemma 2 2B和Llama 3.2 1B上独立复现并拓展了跨任务“prolepsis”机制，揭示 transformer 早期决策、注意力路由和不可逆性。

**💡 创新点**

创新点在于首次证明计划决策与事实回忆共享的“prolepsis”结构、表明 CLT 是观察规划的唯一可行工具，并确定深度阈值决定是否能作出承诺。

**🔧 技术方法**

使用了跨层编码器-解码器（CLT）特征分解、抑制+注入干预、注意力头差分、层级抑制和残差流投影等技术。

**📊 数据集**

数据集包括 Gemma 2 2B、Llama 3.2 1B 开源权重模型、HuggingFace 上的 CLT、诗歌韵脚提示、Transluce 生成的 CounterFact 事实回忆样本。

**📈 对比分析**

对比了六种传统残差流解释方法（均失败）与 CLT 方法，实验显示 CLT 在韵脚重定向成功率高达 77% 以上，且证明搜索层 ≤16 层，承诺层 >16 层。

**⚠️ 局限性**

局限性在于仅涵盖具备 CLT 的小型模型，CLT 覆盖度受限于高频语义域，且未验证深度阈值在更大模型或非检索任务中的普适性。

---

## 416. COEVO: Co-Evolutionary Framework for Joint Functional Correctness and PPA Optimization in LLM-Based RTL Generation

**arXiv ID:** 2604.15001 | [PDF](https://arxiv.org/pdf/2604.15001v1)

**作者:** Heng Ping `[一作]` (University of Southern California), Paul Bogdan `[通讯]` (University of Southern California)

**通讯引用:** 4978 | [OpenAlex ID](https://openalex.org/A5105925385)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于共进化的LLM驱动RTL代码生成框架COEVO，实现功能正确性与PPA（面积、延迟、电源）质量的统一优化；

**💡 创新点**

将功能正确性视为连续维度与PPA并列进行多目标优化，采用自适应正确性门和4D Pareto非支配排序，同时设计多种跨目标演化算子；

**🔧 技术方法**

利用大型语言模型（如GPT-5.4-mini、Qwen2.5-Coder-7B）进行代码生成与修复，结合增强测试平台进行连续正确性评分，使用Yosys与OpenSTA进行合成与PPA评估，采用UCB-Softmax进行算子选择；

**📊 数据集**

在VerilogEval 2.0（156个设计任务）和RTLLM 2.0（50个设计任务）两个标准spec‑to‑RTL基准上进行评测；

**📈 对比分析**

与多种agentic与训练型基线（VeriOpt、VeriAgent、REvolution、EvolVE、SFT+RL模型等）对比，COEVO在功能正确率上均超过最强基线，PPA指标在43/49个可综合设计中获得最佳A×D×P值，显示出显著性能提升；

**⚠️ 局限性**

仍受限于LLM生成质量与算子设计的局限，部分极其复杂设计仍可能因初始解不足或算子不匹配导致收敛到次优解；

---

## 417. Discovering Novel LLM Experts via Task-Capability Coevolution

**arXiv ID:** 2604.14969 | [PDF](https://arxiv.org/pdf/2604.14969v1)

**作者:** Andrew Dai `[一作]`, Yujin Tang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过开放式共进化框架AC/DC自动发现多样化的LLM集群，实现模型与合成任务的共同演化。

**💡 创新点**

提出将演化模型融合与合成任务生成相结合的共进化方法，无需显式基准优化即可获得更广泛的能力覆盖。

**🔧 技术方法**

使用演化模型融合（EvoMerge）+ 交叉/突变、质量多样性选择（DNS）、gibberish过滤、科学家LLM合成任务、最佳‑N 选择等技术。

**📊 数据集**

使用自生成的合成任务集（多学科多难度）以及公开基准（MMLU Pro、GSM8K、GPQA、Code等）评估。

**📈 对比分析**

与专家集、控制集、大模型及GPT‑4o比较，覆盖率提升2–10%，参数更少，最佳‑N性能接近GPT‑4o，显示显著优于单体模型。

**⚠️ 局限性**

受种子模型兼容性、固定科学家LLM、仅靠交叉产生新知识、仅适用于同一基模型的微调版本等限制，需进一步扩展自适应任务生成和多源基模型融合。

---

## 418. FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems

**arXiv ID:** 2604.14956 | [PDF](https://arxiv.org/pdf/2604.14956v1)

**作者:** Wenhao Wang `[一作]` (Zhejiang University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 8912 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 FedGUI 基准，支持跨移动、网页和桌面平台的联邦 GUI 代理训练与评估。

**💡 创新点**

创新点在于首次系统化地模拟四种真实分布异构（跨平台、跨设备、跨 OS、跨来源），并提供六套涵盖 900+ 移动 App、40+ 桌面 App、200+ 网站的公开数据集。

**🔧 技术方法**

采用联邦学习框架（FedAvg、FedProx、SCAFFOLD、FedYogi 等）与 LoRA 适配技术，集成 20+ 视觉语言模型（Qwen、Gemma、GPT 等）和统一的动作空间。

**📊 数据集**

使用 FedGUI-Platform、FedGUI-Device、FedGUI-OS、FedGUI-Web、FedGUI-Mobile 与 FedGUI-Full 六个数据集，来源包括 AC、GA、AS、GO、OA-W/Mac/Win、M2W、AitW 共 9 个公开数据源。

**📈 对比分析**

通过与集中式和本地训练基线对比，评估成功率（SR）、动作类型匹配率、定位准确率等指标。结果表明跨平台联邦协作显著提升性能，适应性优化器（如 FedYogi、FedAdam）在高异构环境下表现更稳健，但仍与集中式训练存在差距。

**⚠️ 局限性**

局限在于仅使用公开数据集，缺乏真实用户的私有分布，无法完全再现实际联邦部署中的隐私与交互模式。

---

## 419. Applying SHAPR in AI-Assisted Research Software Development: Lessons Learnt from Building a Share Trading System

**arXiv ID:** 2604.15020 | [PDF](https://arxiv.org/pdf/2604.15020v1)

**作者:** Ka Ching Chan `[一作]` (University of Southern Queensland), Ka Ching Chan `[通讯]` (University of Southern Queensland)

**通讯引用:** 1692 | [OpenAlex ID](https://openalex.org/A5061915685)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一个模块化的股市交易系统开发项目中，作者采用 SHAPR 框架，对 AI 辅助的软件开发过程进行结构化、可追踪的管理，并记录了整个迭代周期中的反思、合同、快照等多种文档。

**💡 创新点**

创新点在于把 SHAPR 理论落地为可操作的工作配置：①通过“快速捕捉 + AI 精炼”降低文档负担；②引入合同、源真相层、周期快照等控制 artefacts 以提升 AI 代码的稳定性与可追溯性；③展示了工具无关、个性化的多工作空间协同工作模式。

**🔧 技术方法**

使用了 ChatGPT（对话、思考、文档精炼）、PyCharm（代码实现与调试）以及 Obsidian（文档仓库、快照、回顾）等工具，配合迭代式开发与持续文档更新。

**📊 数据集**

本研究未使用传统数据集；核心关注点是软件设计与开发过程本身，记录的“数据”主要是开发日志、合同、快照等过程性文档。

**📈 对比分析**

由于是案例研究，没有对比实验或性能评估；作者通过对比不同迭代周期前后的文档完整度、代码一致性与项目可追溯性来说明 SHAPR 的效果，但未给出量化指标。

**⚠️ 局限性**

局限性包括：①案例仅覆盖一个项目，缺乏跨领域验证；②缺少客观性能指标与量化评估；③工具与流程高度个性化，复制性需要进一步验证；④对 AI 生成代码的安全与正确性缺乏严格检查。

---

## 420. Towards Faster Language Model Inference Using Mixture-of-Experts Flow Matching

**arXiv ID:** 2604.15009 | [PDF](https://arxiv.org/pdf/2604.15009v1)

**作者:** Aihua Li `[一作]` `[通讯]` (Duke University), Aihua Li (Duke University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于混合专家的流匹配（MoE‑FM）框架，并在此基础上设计了非自回归语言模型YAN；

**💡 创新点**

核心创新在于用多专家向量场分解全局输运几何，显著提升了在文本潜在空间中的表示能力，并实现了仅需三步采样即可获得高质量文本；

**🔧 技术方法**

使用技术包括流匹配、混合专家（MoE）机制、Transformer/Mamba架构、潜在变量自编码器、Euler ODE求解等；

**📊 数据集**

使用的训练与评估数据集包括FineWiki、FineWeb、NarrativeQA、SimpleStories、ROCStories、AG News、DBpedia、SST‑2、SQuAD、bAbI等；

**📈 对比分析**

与GPT‑2、BART等自回归模型以及DiffuSeq、Plaid、MDLM等扩散模型进行对比，YAN在文本生成（ROUGE、BERTScore）、问答、分类等任务上表现相当甚至优于基线，同时在推理速度上比AR模型快40–50倍、比扩散模型快10^3倍；

**⚠️ 局限性**

局限性在于目前模型规模仅200M，未在更大规模或更丰富数据上验证；密集的MoE路由在更大模型下可能导致计算开销上升，且尚未实现零样本通用语言能力。

---

## 421. Dr.~RTL: Autonomous Agentic RTL Optimization through Tool-Grounded Self-Improvement

**arXiv ID:** 2604.14989 | [PDF](https://arxiv.org/pdf/2604.14989v1)

**作者:** Wenji Fang `[一作]` (Hong Kong University of Science and Technology), Zhiyao Xie `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 904 | [OpenAlex ID](https://openalex.org/A5075696558)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Dr. RTL——一种基于代理（agent）框架的 RTL 定时优化方法，能够在工业级 EDA 环境中闭环迭代优化 RTL 并自学习可复用的优化技能。

**💡 创新点**

创新点在于：①构建了真实工业级评估环境（人写 RTL、商用合成、时序等价验证），打破传统手工降级或开源工具的局限；②采用多代理闭环设计（时序分析、并行 RTL 重写、评估），实现细粒度的关键路径反馈；③提出组相对技能学习机制，利用同一父 RTL 下并行候选的相对表现来提炼可复用的 pattern–strategy 对，形成可解释、可持续增长的技能库。

**🔧 技术方法**

技术包括：大型语言模型（Claude Opus 作为核心推理器）、多代理协同框架、工业级综合工具（Synopsys Design Compiler）与顺序等价检查（Cadence Jasper SEC）、关键路径映射与根因诊断、并行候选生成与评估、组相对性能度量、技能库构建与更新、实验对照与交叉验证。

**📊 数据集**

数据集为 20 个真实工业 RTL 设计（平均 812 行代码、约 3 模块），涵盖处理器、DSP、加密、I/O 等多种 IP，规模远超现有公开基准；此外还对现有 benchmark（含手工降级 RTL）进行了评测。

**📈 对比分析**

与单次 LLM 重写、SOTA 迭代 LLM 方法（RTLRewriter、SymRTLo、RTL-OPT）以及商业工具直接优化进行对比。Dr. RTL 在所有 20 设计上平均 WNS 提升 21.3%、TNS 提升 16.9%、面积降低 5.8%，SEC 通过率 86%，明显优于基线（如 Claude Opus 单次优化仅 1–3% 的 WNS/TNS 提升）。在 4‑fold 交叉验证的技能迁移实验中，Dr. RTL 仍保持高效的性能提升。

**⚠️ 局限性**

局限性包括：①对 LLM 计算成本和云费用依赖较高；②主要关注定时与面积，功耗未纳入优化目标；③技能库构建依赖于相对评估，可能忽略跨设计的绝对性能提升；④实验仅在 45 nm Nangate 工程下验证，跨工艺和后段（布局布线）效果尚未彻底验证。

---

## 422. MLDAS: Machine Learning Dynamic Algorithm Selection for Software-Defined Networking Security

**arXiv ID:** 2604.14957 | [PDF](https://arxiv.org/pdf/2604.14957v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 423. Explain the Flag: Contextualizing Hate Speech Beyond Censorship

**arXiv ID:** 2604.14970 | [PDF](https://arxiv.org/pdf/2604.14970v1)

**作者:** Jason Liartis `[一作]` (National Technical University of Athens), Orfeas Menis Mastromichalakis `[通讯]` (National Technical University of Athens)

**通讯引用:** 46 | [OpenAlex ID](https://openalex.org/A5030895197)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种融合大型语言模型与三种精心构建的词汇表（英语、法语、希腊语）的混合系统，用以检测并生成针对仇恨言论的上下文化解释。

**💡 创新点**

创新点在于（1）创建了包含词义、使用语境与目标身份信息的“固有贬损词汇表”；（2）设计了两条互补管线——基于词表的精准检测与基于LLM的广泛语境识别；（3）通过LLM融合输出生成可解释且具上下文的判定结果。

**🔧 技术方法**

技术手段主要包括：大型语言模型（Claude Sonnet 3.7 与 Llama 系列）、词表词义检索、词形还原与Levenshtein距离匹配、LLM上下文消歧与解释生成、以及多语言提示工程。

**📊 数据集**

使用的数据集为：从 OffensEval2020（希腊语）、English Hate Speech Superset 与 French Hate Speech Superset 中抽取并人工标注的 1,600 条推文（600 英语、400 法语、600 希腊语）。

**📈 对比分析**

在多种评价指标和数据集变体（Safe、Majority、Permissive、Strict）下，Hybrid 系统在 F1 分数上均优于单一 LLM 基线；在英语与法语上 Claude 领先，希腊语两者相差不大；解释质量评分约 4+（英语/法语）和 3.3-3.5（希腊语）。

**⚠️ 局限性**

主要局限包括：词表构建依赖自动化流程，非全部词条得到人工校对；训练数据源标签与本文仇恨定义不完全一致，需人工标注；希腊语性能受限于低资源和模型对该语言的支持不足；评估仅在推文文本上，缺乏多模态或多场景验证。

---

## 424. POMDP-based Object Search with Growing State Space and Hybrid Action Domain

**arXiv ID:** 2604.14965 | [PDF](https://arxiv.org/pdf/2604.14965v1)

**作者:** Yongbo Chen `[一作]` (Shanghai Jiao Tong University), Hanna Kurniawati `[通讯]` (Australian National University)

**通讯引用:** 2740 | [OpenAlex ID](https://openalex.org/A5073857354)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于POMDP的目标物体搜索方法——GNPF‑kCT，能够处理状态空间随搜索过程增长、动作空间包含连续与离散两种子域的复杂室内场景。

**💡 创新点**

创新点包括：① 结合神经过程（NP）对连续动作进行筛选，显著减少搜索空间；② 使用k‑center聚类对残余动作进行分层细化，兼顾高维连续动作的可搜索性；③ 通过信念树重用与状态历史维护，避免在状态空间扩展时重新构建完整树；④ 引入伪目标物体与格子世界进行先验探索，提升稀疏奖励下的收敛速度。

**🔧 技术方法**

技术手段涵盖：POMDP建模、Monte Carlo Tree Search（MCTS）与UCB改进、神经过程网络、k‑center聚类、粒子滤波、ROS移动基础与抓取控制、深度图与点云融合、YOLO与SIFT等感知模块。

**📊 数据集**

使用的实验数据集主要为Gazebo仿真环境（Fetch与Stretch机器人）以及真实办公室场景中的2D占用网格、3D点云、RGB‑D图像；训练神经过程时通过自生成的仿真数据（3000次迭代）构建。

**📈 对比分析**

与基线方法（POMCP、GPOMCP、POMCPOW、VOMCPOW、NPF‑kCT）以及非POMDP方法（Random、SGoLAM、SayPlan、MoMa‑LLM）进行对比，GNPF‑kCT在多种复杂度场景下获得更高的折扣累计奖励、更多成功率、且平均步数更少，特别是在遮挡严重、目标不易观测的环境中表现最为突出。

**⚠️ 局限性**

局限性包括：① 对感知与动作执行模型的准确性高度依赖，真实硬件中感知误差或抓取失败导致成功率下降；② 仍需手工设置诸如奖励阈值、k值等超参数，虽然对结果影响有限但存在一定敏感性；③ 对极大规模多物体场景的可扩展性尚未充分验证；④ 伪目标物体的设定假设目标分布可通过格子估计，若目标分布极为不均匀可能影响搜索效率。

---

## 425. A Semantic Geometry for Uncovering Paradigm Dynamics via Scientific Publications

**arXiv ID:** 2604.15150 | [PDF](https://arxiv.org/pdf/2604.15150v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 426. Frequency-Enhanced Dual-Subspace Networks for Few-Shot Fine-Grained Image Classification

**arXiv ID:** 2604.14958 | [PDF](https://arxiv.org/pdf/2604.14958v1)

**作者:** Meijia Wang `[一作]` (Shaanxi University of Science and Technology), Junpo Yang `[通讯]` (Shaanxi University of Science and Technology)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5026686074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FEDSNet，结合频域DCT低通滤波与双子空间度量，解决细粒度少样本分类中的特征耦合与结构不稳定问题。

**💡 创新点**

创新点在于：①引入频域结构分离分支，利用低通滤波提取低频结构作为“结构锚”，②在空间与频域各自构建低秩子空间，并通过自适应融合的双子空间度量提升判别力。

**🔧 技术方法**

采用2D离散余弦变换、低通掩模、频域通道注意力、逆DCT、截断奇异值分解、子空间投影距离、可学习融合权重以及正交正则化等技术。

**📊 数据集**

使用CUB-200-2011、Stanford Cars、Stanford Dogs和FGVC-Aircraft四个细粒度图像数据集。

**📈 对比分析**

在5-way 1-shot/5-shot任务上与ProtoNet、DN4、DeepEMD、MattML、DSN等经典与最新少样本FGVC方法对比，FEDSNet在大多数设置下均优于基线，并在ResNet-12上逼近或超过最先进模型。

**⚠️ 局限性**

局限性是对极度依赖微小局部细节的类别（如车灯、鸟喙）时，低通滤波可能抹去关键信息；此外，在极少样本下仍可能受背景干扰，未来需引入更细粒化的频域划分和局部注意力机制。

---

## 427. CAVERS: Multimodal SLAM Data from a Natural Karstic Cave with Ground Truth Motion Capture

**arXiv ID:** 2604.15052 | [PDF](https://arxiv.org/pdf/2604.15052v1)

**作者:** Giacomo Franchini `[一作]` (Polytechnic of Turin), Marcello Chiaberge `[通讯]` (Polytechnic of Turin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建并公开了一个多模态自然喀斯特洞穴 SLAM 数据集 CAVERS，并对多种主流 SLAM 与里程计算法在该环境下的性能进行了基准评测。

**💡 创新点**

创新点包括：①在真实自然洞穴中获得毫米级高频位姿标定（OptiTrack）；②融合 RGB‑D、近红外热像、激光雷达与 IMU 的多模态数据；③在暗光与人工照明两种极端照明条件下进行采集；④提供大规模（≈335 GB）公开数据，填补了天然洞穴 SLAM 研究的空白。

**🔧 技术方法**

使用的技术与方法有：Intel RealSense D435i、Optris PI640i 热像、Velodyne VLP‑16 雷达、内置 IMU；运动捕捉系统 OptiTrack 进行标定；评测算法包括 ORBSLAM3（RGB‑D、RGB‑D‑I）、RTAB‑Map（RGB‑D‑I、LiDAR‑I）、ROVTIO（热‑视觉‑I）、KISS‑ICP、GENZ‑ICP 以及 RTAB‑Map LiDAR‑I；采用绝对轨迹误差 ATE 进行量化比较。

**📊 数据集**

本研究使用的数据集为作者自行采集的 CAVERS 数据集，包含约 55,000 张 RGB/热像图、LiDAR 点云、IMU 数据和 120 Hz 的运动捕捉位姿，规模达 335 GB。未使用其他公开洞穴数据集。

**📈 对比分析**

通过将算法输出轨迹与运动捕捉 ground‑truth 对齐，计算 ATE（平移和旋转）进行比较。结果显示 LiDAR‑I 方案总体误差最低（如 GENZ‑ICP 与 RTAB‑Map LiDAR‑I 在不同轨迹均取得 0.1–0.5 m 的平移误差），视觉‑I 方案在光照变化大时失踪或漂移较大，热‑视觉‑I 在某些轨迹中表现相对稳定，但整体精度低于 LiDAR。视觉单模态在高运动模糊时易失锁。

**⚠️ 局限性**

局限性包括：①仅包含两间洞室，样本多样性有限；②部分轨迹（如 loc_handheld_5）缺乏精确 ground‑truth，需用 LiDAR 轨迹作为参考；③运动捕捉设备与电缆在部分数据中可见，造成噪声与热像干扰；④同步采用软件时延，硬件级同步未实现；⑤极端照明下的传感器数据缺失或失真，影响算法鲁棒性。

---

## 428. QuantCode-Bench: A Benchmark for Evaluating the Ability of Large Language Models to Generate Executable Algorithmic Trading Strategies

**arXiv ID:** 2604.15151 | [PDF](https://arxiv.org/pdf/2604.15151v1)

**作者:** Alexey Khoroshilov `[一作]`, Dmitry Zmitrovich `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了QuantCode-Bench基准，评估LLM在自然语言描述下生成Backtrader框架可执行交易策略的能力，并通过四阶段评估管线和LLM判定验证语义一致性。

**💡 创新点**

创新点在于提出域特定的多阶段评估机制、引入LLM-as-a-Judge进行语义对齐判定，并对单轮与多轮交互式修复两种设置进行系统比较，揭示交互式调试对模型性能的显著提升。

**🔧 技术方法**

采用的技术包括LLM代码生成、Backtrader回测环境、LLM判定器、交互式多轮反馈机制以及错误分类分析。

**📊 数据集**

使用了400条交易策略任务，来源于Reddit、TradingView、StackExchange、GitHub及合成数据，并按易、中、难三类进行难度分层。

**📈 对比分析**

在单轮设置下，模型Judge Pass率从48%至76%不等；在多轮交互式修复后，最高可达95%–98%，表明大部分错误可通过反馈修复。

**⚠️ 局限性**

局限性包括仅在Backtrader框架内评估、LLM判定可能存在偏差、未考察策略盈利与风险表现、缺乏跨框架通用性。

---

## 429. An Axiomatic Benchmark for Evaluation of Scientific Novelty Metrics

**arXiv ID:** 2604.15145 | [PDF](https://arxiv.org/pdf/2604.15145v1)

**作者:** Miri Liu `[一作]` (University of Illinois at Urbana-Champaign), ChengXiang Zhai `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于公理的基准，用以系统评估科学论文新颖性度量方法的表现；

**💡 创新点**

首次将新颖性评价转化为一组可量化的公理，并用此公理检验并组合多种现有度量；

**🔧 技术方法**

采用嵌入与局部密度、t‑SNE、词向量等文本表示技术构建四个基准度量；

**📊 数据集**

使用PapersWithCode存档中的约1500篇AI论文，按任务、时序等构造参考池；

**📈 对比分析**

对四个度量在十个跨领域任务上的公理通过率进行对比，单一度量平均通过率最高为71.5%，而按公理加权组合可提升至90.1%；

**⚠️ 局限性**

主要局限在于大多数度量仅基于标题/摘要嵌入，无法捕捉全文细节、细粒度覆盖与时序语义，导致公理3、5、7、8表现不佳。

---

## 430. SCENIC: Stream Computation-Enhanced SmartNIC

**arXiv ID:** 2604.15128 | [PDF](https://arxiv.org/pdf/2604.15128v1)

**作者:** Benjamin Ramhorst `[一作]` (ETH Zurich), Gustavo Alonso `[通讯]` (ETH Zurich)

**通讯引用:** 17391 | [OpenAlex ID](https://openalex.org/A5103144919)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一款开放源代码的200G FPGA SmartNIC SCENIC，可在数据中心统一集成CPU、GPU和SSD，并支持多协议（RDMA、TCP/IP）和可编程流计算单元（SCU）。

**💡 创新点**

创新点在于将网络数据路径视作第一类流计算 substrate，提供可动态重配置的硬件编程接口；实现高达200G的线速网络，兼容Linux NIC与RDMA；在FPGA上实现可编程拥塞控制、GPU/SSD直连以及多租户隔离，所有功能均开源可定制。

**🔧 技术方法**

采用Xilinx Alveo U55C / U280 / V80 FPGA、Coyote 2 shell、RDMA/TCP/IP开源栈、可编程的Stream Compute Units（HDL/HLS/P4）、ARM Cortex-A72/R5 控制核心、PCIe QDMA、动态部分重配置等技术。

**📊 数据集**

使用行业基准数据（IB吞吐、RDMA延迟、GPU/SSD读写、TCP‑NVMe I/O）以及自定义的多GPU哈希分区数据集（两列合成表）进行评测。

**📈 对比分析**

与Mellanox ConnectX‑5、Broadcom Stingray以及OpenMPI/ACCL+等商用/学术平台对比，SCENIC在吞吐上基本匹配商用NIC，延迟略高但可接受；在GPU/SSD直连、散列分区等场景下显著提升（如哈希分区吞吐比CPU基线提升6.7×）。

**⚠️ 局限性**

局限包括对NVIDIA GPU的读写性能仍低于预期、动态重配置时仍有几毫秒延迟、对UltraEthernet等新协议支持尚未完善，以及在400G级别的可扩展性尚未实现。

---

## 431. MCSC-Bench: Multimodal Context-to-Script Creation for Realistic Video Production

**arXiv ID:** 2604.15127 | [PDF](https://arxiv.org/pdf/2604.15127v1)

**作者:** Huanran Hu `[一作]` (Renmin University of China), Qin Jin `[通讯]` (Renmin University of China)

**通讯引用:** 4844 | [OpenAlex ID](https://openalex.org/A5009985839)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了多模态上下文到脚本生成（MCSC）任务，并构建了首个大规模的 MCSC-Bench 数据集，用于评估从噪声多模态输入生成可执行视频脚本的全过程。

**💡 创新点**

创新点包括：① 统一定义了素材选择、剧情补全与脚本结构化的全流程任务；② 开发首个面向该任务的大规模数据集；③ 设计了基于规则与多维度评估的完整评测体系；④ 通过三阶段多代理工作流与强化学习提升模型表现。

**🔧 技术方法**

主要技术包括：多模态大型语言模型（Qwen3-VL、Gemini-2.5-Pro 等）的全参数微调与 Group Relative Policy Optimization（GRPO）强化学习；Rule‑based 与 1‑5 量表的多维度评估；三阶段（分析‑规划‑生成）多代理框架。

**📊 数据集**

使用自制的 11,247 条广告视频组成的 MCSC-Bench 数据集，平均每个样本包含 7.2 个视频片段（5.7 相关素材 + 1.5 干扰素材），配有文本材料、用户指令和结构化脚本；另外提供 521 条通用视频作为 OOD 测试。

**📈 对比分析**

通过六维指标（Err、Rep、ΔT、Instruction Following、Attractiveness、Coherence of Voiceover）进行自动评估，已知模型在 SFT 与 RL 训练后显著提升；8B 规模模型已突破 Gemini‑2.5‑Pro；在脚本驱动的视频生成实验中，脚本驱动生成在叙事连贯、视觉质量与整体吸引度上胜过指令驱动基线，胜率>88%。

**⚠️ 局限性**

存在的局限包括：全局时序规划与长程推理仍然是瓶颈；模型在跨域噪声素材选择和长输出生成时表现下降；评估体系依赖 LLM 判分，可能带来偏差；数据集构建过程中使用生成模型重现脚本，可能引入模型偏倚。

---

## 432. Combinatorial Contracts Through Demand Types

**arXiv ID:** 2604.15125 | [PDF](https://arxiv.org/pdf/2604.15125v1)

**作者:** Elizabeth Baldwin `[一作]` (University of Oxford), Maya Schlesinger `[通讯]` (Tel Aviv University)

**通讯引用:** 886 | [OpenAlex ID](https://openalex.org/A5109860000)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文将组合合约问题与消费者需求类型理论联系起来，提出了一类称为All Substitutes and Complements（ASC）的奖励函数，并证明该类最多只产生O(n²)个关键值；在此基础上利用需求类型结构给出了高效的需求查询算法，进而实现了在ASC及其子类（如GSC、GSC+、Δ-替代）下的多项式时间最优线性合约求解；

**💡 创新点**

创新点包括：① 用需求类型框架统一了以往仅局部可解的GS、超模、ultra等类；② 设计并证明ASC类为更大可解类，最多O(n²)关键值；③ 引入“facet‑piercing”概念以规避多面交叉问题；④ 在“succinct”需求类型下实现价值查询转化为多项式时间的需求查询。

**🔧 技术方法**

采用几何视角，将合约参数α映射为价格空间中的射线；利用LIP、UDR和需求类型理论解析最佳响应区域；使用潜在函数计数方法证明关键值上界；设计路径跟踪算法通过邻接包络更新需求，从而在价值查询上实现多项式时间。

**📊 数据集**

论文基于理论分析与证明，并未使用实际数据集；所有结果均为理论复杂度和算法正确性证明。

**📈 对比分析**

与之前仅针对GS、超模、ultra等类的多项式算法相比，本文将可解范围扩展到更大ASC类，并给出了在该类及其子类下的多项式时间求解；对GSC、GSC+、Δ-替代等类的需求查询亦可通过价值查询实现，显著降低了对强大oracle的需求。

**⚠️ 局限性**

局限性在于证明依赖于facet‑piercing且成本唯一的假设；ASC被推测为最大可解类但尚未得到完整证明；算法仍停留在理论层面，缺乏实验实现与性能评估；对非succinct需求类型的可行性仍未知。

---

## 433. NFTDELTA: Detecting Permission Control Vulnerabilities in NFT Contracts through Multi-View Learning

**arXiv ID:** 2604.15118 | [PDF](https://arxiv.org/pdf/2604.15118v1)

**作者:** Hailu Kuang `[一作]` (Hainan University), Zongwei Li `[通讯]` (Hainan University)

**通讯引用:** 53 | [OpenAlex ID](https://openalex.org/A5101530716)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出 NFTDELTA 框架，利用多视角静态分析和向量相似度检测 NFT 合约中的权限控制漏洞，并通过可扩展的检测器实现对三类新型漏洞的识别。

**💡 创新点**

创新点在于：①将执行路径序列与 CFG 结构两种视角融合生成统一代码表示；②基于向量相似度的无训练检测方法，兼容现有漏洞报告；③定义并检测三种新的 NFT 权限控制漏洞（弱授权验证、松散权限管理、绕过授权重入）。

**🔧 技术方法**

使用的技术包括：静态分析（AST+SSA+taint）、Word2Vec + Performer（序列编码）、FastGAT（图编码）、Attention‑Fusion、HNSW 向量数据库、欧氏距离相似度判定。

**📊 数据集**

数据集：Embedding 数据集 3,146 条 ERC‑721 合约；Validation 数据集 795 条 OpenSea 交易量高的 ERC‑721 合约，均通过编译、flattening 预处理。

**📈 对比分析**

与 Solhint、AChecker、NFTGuard、Mythril、SmartEmbed 等工具在 15 条包含单一漏洞的合约上比较，NFTDELTA 在权限漏洞检测上实现 97.92% 的平均精度、95.76% 的 F1；在全量 795 条验证集上，平均检测时间 3.36 秒，内存占用 3‑4 GB，显著优于传统符号执行或规则引擎方法。

**⚠️ 局限性**

局限性：相似度阈值设置影响精度，较高阈值易误报；无监督向量方法缺乏对细粒度语义的捕获，导致漏检；缺少训练数据限制了模型对新型漏洞的泛化能力；需进一步改进阈值、引入监督学习及扩充漏洞样本。

---

## 434. Building Extraction from Remote Sensing Imagery under Hazy and Low-light Conditions: Benchmark and Baseline

**arXiv ID:** 2604.15088 | [PDF](https://arxiv.org/pdf/2604.15088v1)

**作者:** Feifei Sang `[一作]` (Anhui University), Bin Luo `[通讯]` (Anhui University)

**通讯引用:** 11343 | [OpenAlex ID](https://openalex.org/A5100372676)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套针对雾霾和低光环境下光学遥感影像建筑提取的完整方法与数据集

**💡 创新点**

创新点包括：①创建首个雾霾与低光场景建筑提取基准HaLoBuilding；②设计端到端的HaLoBuild‑Net网络，融合空间-频域关注模块(SFFM)、全局多尺度引导模块(GMGM)和互导融合模块(MGFM)以实现对天气干扰的双域自适应调节；③通过同场景多时序配对和人工精细校正实现像素级高质量标注；④证明端到端策略优于传统先恢复后分割的级联方案

**🔧 技术方法**

采用轻量级LWGANet-L2骨干网络，SFFM（空间注意+频域重加权）、GMGM（多尺度全局语义引导）、MGFM（双向语义-空间校准）以及FFT低频特征辅助的频域关注机制

**📊 数据集**

使用HaLoBuilding数据集（含HaLo‑H雾霾与HaLo‑L低光两子集，共4386张1024×1024 RGB图像）以及公开清晰场景基准WHU、INRIA、LoveDA进行跨场景评测

**📈 对比分析**

与多种先前方法（UNetFormer、BuildFormer、SACANet、EasyNet、RSBuilding等）以及级联恢复+分割做对比，HaLoBuild‑Net在HaLo-L/HaLo-H的IoU分别提升约2.5%/3%，并在WHU、INRIA、LoveDA等常规数据集上保持或提升性能，证明其优越性与良好的泛化能力

**⚠️ 局限性**

局限性包括：①模型仍相对较大（约23.9M参数），不适合极限边缘设备；②主要针对二分类建筑/背景任务，尚未扩展至多类语义分割；③在极端低光或高雾浓度下仍存在边缘模糊或误检风险，需要进一步提升鲁棒性

---

## 435. NEAT-NC: NEAT guided Navigation Cells for Robot Path Planning

**arXiv ID:** 2604.15076 | [PDF](https://arxiv.org/pdf/2604.15076v1)

**作者:** Hibatallah Meliani `[一作]` (Abdelmalek Essaadi University), Samira Khoulji `[通讯]` (Abdelmalek Essaadi University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了NEAT-NC算法，利用神经进化的拓扑结构和递归神经网络，以生物学中的定位、边界、头向和速度细胞作为输入，解决静态和动态环境中的路径规划问题。

**💡 创新点**

创新点在于将空间认知细胞的生物学原理与NEAT算法结合，设计了海马启发的适应度函数，使用递归记忆来模拟海马空间记忆，并通过四种导航细胞作为感知输入提升搜索效率。

**🔧 技术方法**

技术手段包括NEAT进化算法、递归神经网络、海马式奖励机制（平滑、碰撞、目标、位移、可视化奖励）、PPO强化学习基线、Gymnasium/pygame仿真、Python、neat库和Stable Baselines3。

**📊 数据集**

实验使用了三种模拟环境（S形迷宫、含5个动态障碍、含2个动态障碍），每个环境都配有8路雷达传感器作为观测，未使用公开真实数据集。

**📈 对比分析**

对比方法为在30次独立运行下，评估NEAT-NC、Vanilla NEAT和PPO的成功率、适应度、路径长度和执行时间；统计检验（Kruskal-Wallis、Chi-square、Dunn后检验）显示NEAT-NC在所有指标上显著优于其他算法。

**⚠️ 局限性**

局限性包括仅在二维仿真环境中验证，动态障碍运动模式简单；未在真实机器人或三维环境中测试；对超参数的敏感性及计算开销仍需进一步优化。

---

## 436. Beyond the Laplacian: Doubly Stochastic Matrices for Graph Neural Networks

**arXiv ID:** 2604.15069 | [PDF](https://arxiv.org/pdf/2604.15069v1)

**作者:** Zhaobo Hu `[一作]` (SAMOVAR), Mehdi Naima `[通讯]` (CNRS -- LIP6)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出用双随机图矩阵（DSM）替代传统拉普拉斯矩阵来构建图神经网络，并通过截断Neumann级数与残差质量补偿实现可扩展的高效信息扩散

**💡 创新点**

创新点在于：①用DSM作为连续多跳亲近度与节点中心性的数学描述；②利用截断Neumann级数得到O(K|E|)的近似；③设计残差质量补偿机制恢复行随机性与中心性，解决截断导致的概率泄漏问题；④在此基础上构建DsmNet与DsmNet-compensate两种解耦式架构

**🔧 技术方法**

技术包括：双随机矩阵推导、矩阵分裂与Neumann级数近似、残差质量补偿、稀疏矩阵乘法（SpMM/SpMV）实现高效前向传播、对谱特征和Dirichlet能量的理论分析

**📊 数据集**

使用的公开图数据集包括Cora、Citeseer、PubMed、CS、Physics、Photo、Computers（同质网络）以及WebKB（Texas、Cornell、Wisconsin）、Wikipedia（Chameleon、Squirrel）等异质网络，另外用于中心性回归的WS、RGG、LFR等合成图

**📈 对比分析**

与GCN、GAT、SAGE、ChebNet、APPNP、GCNII、JKNet等基线比较，DsmNet-compensate在同质任务上常能取得或逼近最高准确率，尤其在全监督下表现优异；在异质网络中表现略逊，但仍保持竞争力；在中心性回归任务中DSM特征显著提升排序相关性（Kendall's τ）和NDCG等指标

**⚠️ 局限性**

局限性包括：①DSM天然是低通滤波，对高度异质或高频信息抑制效果不佳；②截断阶K的选择仍需经验或自适应优化；③在有向或动态图的适用性尚未验证；④与大规模图的内存/计算负载相比，仍存在一定的稀疏矩阵乘法开销

---

## 437. Learning Where to Embed: Noise-Aware Positional Embedding for Query Retrieval in Small-Object Detection

**arXiv ID:** 2604.15065 | [PDF](https://arxiv.org/pdf/2604.15065v1)

**作者:** Yangchen Zeng `[一作]` (Southeast University), Kangning Cui `[通讯]` (City University of Hong Kong)

**通讯引用:** 295 | [OpenAlex ID](https://openalex.org/A5080932443)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种噪声感知位置嵌入框架HELP，利用热图引导的位置信息选择性保留前景并抑制背景，从而提升小目标检测性能；

**💡 创新点**

创新点在于将梯度产生的热图转化为二值掩模，选择性注入位置编码，并在编码器与解码器两侧引入MOHFE、HQ‑Retrieval和LSConv，实现解码器层数从八层压缩到三层；

**🔧 技术方法**

核心技术包括热图引导位置嵌入（HPE）、多尺度热图融合编码器（MOHFE）、高质量查询检索（HQ‑Retrieval）以及线性蛇形卷积（LSConv）；

**📊 数据集**

实验数据集涵盖NWPU VHR‑10、PASCAL VOC、DOTA、DIOR、VisDrone等五大基准；

**📈 对比分析**

与RT‑DETR、DETR等基线对比，mAP@0.5/ mAP 分别提升约6–7点，参数量从163M降至66.3M（59%），GFLOPs从136降至57，显示显著的准确率与效率双提升；

**⚠️ 局限性**

局限在于热图梯度生成仅在训练阶段使用，对极端稀疏场景或非航空场景的适用性需进一步验证。

---

## 438. Atropos: Improving Cost-Benefit Trade-off of LLM-based Agents under Self-Consistency with Early Termination and Model Hotswap

**arXiv ID:** 2604.15075 | [PDF](https://arxiv.org/pdf/2604.15075v1)

**作者:** Naryeong Kim `[一作]` (KAIST), Shin Yoo `[通讯]` (KAIST)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在 LLM 基础代理中通过预测中途推理是否会成功来实现早停，并在预测失败时将推理上下文迁移到更强大模型（hotswap）以节省成本。

**💡 创新点**

创新点在于：①利用语义流程图（SFG）将多条自洽推理轨迹映射为图结构；②用图卷积网络（GCN）预测推理是否会成功；③提出早停与 hotswap 的结合方案，首次针对多步工具调用代理实现中途成本优化。

**🔧 技术方法**

技术包括：ReAct 与自洽（Self‑Consistency）prompt，SFG 生成与节点/边表示（功能 + FastText 语义向量），GCN 二分类模型，统计阈值决定早停/换模型，模型迁移实现基于 stateless 上下文的 hotswap。

**📊 数据集**

数据集：AutoFL 在 Defects4J（353 任务），AutoCodeRover 在 SWE‑bench（1000 任务），RepairAgent 在 Defects4J（605 任务）。实验使用 Llama‑3‑8B / Mixtral‑8x7B 作为低成本源模型，GPT‑4o / GPT‑4 作为目标模型。

**📈 对比分析**

与多数基线（多数类、投票置信度、Lachesis）比较，预测准确率最高可达 0.93（AutoCodeRover），AUROC 超过 0.85。早停可将成本降至 23.9%（AutoFL）/ 24%（AutoCodeRover）/ 64.7%（RepairAgent），同时保留 74.4% / 81.3% / 78.9% 的性能；相较单一目标模型，hopswap 在中途切换时实现显著成本收益。

**⚠️ 局限性**

局限性包括：仅在三种代理和自洽设置下验证；需要多次推理产生的 SFG 计算开销；对低成本模型的推理步骤缺乏多样性导致早停精度下降；模型迁移假设 LLM 状态完全无关上下文；在大规模真实项目中的可扩展性与安全性仍待进一步评估。

---

## 439. Assessing the Potential of Masked Autoencoder Foundation Models in Predicting Downhole Metrics from Surface Drilling Data

**arXiv ID:** 2604.15169 | [PDF](https://arxiv.org/pdf/2604.15169v1)

**作者:** Aleksander Berezowski `[一作]` (University of Calgary), Gouri Ginde `[通讯]` (University of Calgary)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5108426807)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对2015-2025年间的13篇文献进行系统性映射，分析油气钻井中表面传感器与下井指标的关系，评估掩码自编码器基础模型（MAEFM）在下井指标预测中的潜力。

**💡 创新点**

创新点在于提出并论证MAEFM作为一种可自监督预训练、可多任务的基础模型，可利用海量未标记钻井数据，填补当前仅使用传统ANN、LSTM等监督模型的空白。

**🔧 技术方法**

采用系统性映射研究方法、文献检索与筛选、数据提取，并讨论MAEFM、ANN、LSTM、CNN等机器学习技术。

**📊 数据集**

使用的主要数据集为公开文献中的钻井时间序列数据，包括RPM、WOB、Q、ROP、SPP、T、MW、Depth等表面指标，及对应的下井指标如ECD、BHP、振动等。

**📈 对比分析**

本研究未进行实验比较，而是基于文献归纳现有模型（ANN、LSTM、CNN等）性能指标；对MAEFM的预期表现提出假设，强调需进一步实证验证其准确性和泛化能力。

**⚠️ 局限性**

局限性包括仅检索英文文献、时间范围有限（2015-2025）、数据库覆盖不全、缺乏实验验证以及未评估MAEFM实际性能。

---

## 440. Class Unlearning via Depth-Aware Removal of Forget-Specific Directions

**arXiv ID:** 2604.15166 | [PDF](https://arxiv.org/pdf/2604.15166v1)

**作者:** Arman Hatami `[一作]` (Johns Hopkins University), Ilya E. Monosov `[通讯]` (Johns Hopkins University)

**通讯引用:** 2468 | [OpenAlex ID](https://openalex.org/A5024947116)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单次无梯度、深度感知的投影修剪方法DAMP，用于在不重新训练的情况下实现已训练网络对指定类别的遗忘。

**💡 创新点**

创新点在于：①在每层编辑空间中构造忘记方向子空间并做深度加权投影，既去除内部表征中的遗忘类信息，又保持低层共享特征；②完全无超参、一次性闭式更新；③通过低秩子空间实现多类别遗忘。

**🔧 技术方法**

主要技术包括：类原型计算、保持类投影残差提取、QR正交化生成忘记方向、投影权重更新、探针可分离度与深度决定的层系数。

**📊 数据集**

使用MNIST、CIFAR‑10、CIFAR‑100、Tiny ImageNet四个图像分类数据集，并在CNN‑5、ResNet‑18与ViT三种架构上评测。

**📈 对比分析**

与基线、重新训练、GAU、KDU、DD‑FT、LM、RandRelabel、SSD、SalUn等方法对比，DAMP在保持保留类准确率的同时接近重新训练的遗忘效果，选择性更高，计算成本低。

**⚠️ 局限性**

局限性在于假设类表征近似为均值且线性可分，无法很好处理多模态或高度非线性的分布；在极端多类别或高维子空间场景下效果可能下降。

---

## 441. DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency

**arXiv ID:** 2604.15163 | [PDF](https://arxiv.org/pdf/2604.15163v1)

**作者:** Boyan Li `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Dual‑Paradigm Consistency 框架，利用对抗式合成的最小可辨识数据库（MDD）和 SQL 与 Python 双范式执行，实现无训练的候选 SQL 选择与验证。

**💡 创新点**

创新点包括：① 在部分可观测数据库下生成对抗式 MDD 解决观察盲区；② 通过双范式（SQL 与 Python）一致性验证消除符号盲目与确认偏差；③ 引入跨范式的 Bipartite Soft‑F1（BS‑F1）度量以处理结果格式和顺序差异。

**🔧 技术方法**

采用多智能体协作（切片、对抗、Python 生成）和对抗环境合成、双范式执行、Hungarian 匹配的 BS‑F1、运行时自校正等技术。

**📊 数据集**

在 BIRD 与 Spider 两个主流 SQL 基准上进行评估。

**📈 对比分析**

与随机选择、执行导向、Self‑Consistency、Multiple‑Choice 等基线对比，显著提升执行准确率（BIRD 最高 2.4% 以上，Spider 亦有 1–2% 的提升），并在多种 LLM 背景下保持领先。

**⚠️ 局限性**

主要局限是推理延迟和多智能体执行成本较高；对抗式 MDD 可能与真实数据分布偏差，且在极端不确定场景下仍需进一步优化触发机制。

---

## 442. LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking

**arXiv ID:** 2604.15149 | [PDF](https://arxiv.org/pdf/2604.15149v1)

**作者:** Lukas Helff `[一作]` (TU Darmstadt), Felix Friedrich `[通讯]` (Meta FAIR)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究RLVR训练的LLM在归纳推理任务中出现的奖励短路行为，并提出Isomorphic Perturbation Testing检测方法。

**💡 创新点**

首次将逻辑同构扰动作为黑盒检测奖励短路的手段，并揭示RLVR训练会诱发模型枚举实例而非抽象规则。

**🔧 技术方法**

结合强化学习与可验证奖励、逻辑同构扰动、诱导逻辑程序等技术，构建IPT评估框架。

**📊 数据集**

使用公开的逻辑推理基准（ILP风格任务）和多种LLM模型（GPT‑5、Olmo3等）进行评估。

**📈 对比分析**

通过对比extensional与isomorphic验证的通过率，发现RLVR模型短路率随任务难度和推理算力提升显著增加，而非RLVR模型无短路；尽管RLVR模型在基准上得分更高，但同时出现更多短路。

**⚠️ 局限性**

研究仅关注特定归纳推理任务，缺乏对更广泛推理场景的验证，且对闭源模型的黑盒检测仍受限于输出可观测性。

---

## 443. KVNN: Learnable Multi-Kernel Volterra Neural Networks

**arXiv ID:** 2604.15141 | [PDF](https://arxiv.org/pdf/2604.15141v1)

**作者:** Haoyu Yun `[一作]` (North Carolina State University), Yufang Bao `[通讯]` (Fayetteville State University)

**通讯引用:** 81 | [OpenAlex ID](https://openalex.org/A5110368136)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种可插拔的核化 Volterra 神经网络层（kVNN），通过学习可调多核结构将高阶多项式交互直接嵌入卷积层，实现更丰富的非线性特征表达。

**💡 创新点**

创新点在于：1) 使用可学习的多核（不同阶的多项式核）实现阶自适应的参数化；2) 通过中心向量集合代替样本扩展，显著降低高阶 Volterra 的参数与计算成本；3) 设计可直接替换标准卷积的分支结构，保持模块化与端到端可训练性。

**🔧 技术方法**

采用核方法（多项式核、RKHS 表示）、Volterra 系列展开、可学习中心的多核学习、分支式卷积实现、端到端深度学习训练。

**📊 数据集**

视频动作识别使用 UCF101、HMDB51；图像去噪使用 BSD68、Set12、SIDD Medium。

**📈 对比分析**

与现有轻量级 CNN、I3D 等模型在 UCF101、HMDB51 以及去噪数据集上比较，kVNN 在保持或提升准确率（如 kVNN‑S 3rd order 在 UCF101 取得 90.02% 仅 12.29M 参数，kVNN‑B 3rd order 在 UCF101 取得 92.67% 仅 30M 参数）同时显著降低 GFLOPs（视频 28–35 GFLOPs；去噪 1.5–2.3 GFLOPs），甚至在无预训练的从零开始训练时也能超过预训练模型。

**⚠️ 局限性**

限制：1) 需要为每个交互阶设定中心数与系数，超参数调节较繁琐；2) 目前仅验证在卷积网络中，对 Transformer 等其他架构的兼容性和可扩展性待进一步研究；3) 高阶（>3）实现的计算和内存开销仍随阶数增长；4) 在极大规模数据或任务多样性上尚未充分评估。

---

## 444. Where are the Humans? A Scoping Review of Fairness in Multi-agent AI Systems

**arXiv ID:** 2604.15078 | [PDF](https://arxiv.org/pdf/2604.15078v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 445. How to Correctly Make Mistakes: A Framework for Constructing and Benchmarking Mistake Aware Egocentric Procedural Videos

**arXiv ID:** 2604.15134 | [PDF](https://arxiv.org/pdf/2604.15134v1)

**作者:** Olga Loginova `[一作]` (University of Trento), Frank Keller `[通讯]` (University of Edinburgh)

**通讯引用:** 12337 | [OpenAlex ID](https://openalex.org/A5054936589)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PIE-V 框架，基于心理学原理在 egocentric 程序视频中注入可恢复的错误与纠正痕迹；

**💡 创新点**

创新点在于：① 通过程序阶段与负载的先验模型来定位错误；② 采用语义角色映射局部修改以保持世界状态一致；③ 结合 LLM 撰写与评判实现全局一致的文本重写；④ 开发统一错误分类与多维评估 Rubric；

**🔧 技术方法**

使用心理学驱动的错误规划器、纠正规划器、LLM 写手（GPT‑5.2、Qwen‑2.5）、LLM 判断器（Qwen‑3‑VL）、文本‑视频同步生成器（Kling‑O、Sora、Seedance 等）；

**📊 数据集**

主要在 Ego‑Exo4D（17 项 50 场景）上构建错误集，同时评审 EgoPER、EgoOops、Assembly101 与 CaptainCook4D 四大 egocentric 数据集；

**📈 对比分析**

与自由形式 LLM 生成基线（Qwen、GPT‑5.2）以及加判别器的基线对比，PIE‑V 在步骤错误率、逻辑一致性、状态连贯性与人类可接受度等九维指标上普遍优于基线，且误差更接近自然人类错误；

**⚠️ 局限性**

局限性包括：① 依赖预先计算的语义角色与复杂度估计，可能在新领域难以迁移；② 视频合成受限于现有生成模型的长度与真实感；③ 纠正策略采样受检测概率假设影响，偶尔缺失合理恢复；④ 评估仍需人工，自动化替代方案尚不成熟。

---

## 446. FedIDM: Achieving Fast and Stable Convergence in Byzantine Federated Learning through Iterative Distribution Matching

**arXiv ID:** 2604.15115 | [PDF](https://arxiv.org/pdf/2604.15115v1)

**作者:** He Yang `[一作]`, Jizhong Zhao `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedIDM 框架，解决 Byzantine 攻击导致联邦学习收敛慢、鲁棒性差的问题。

**💡 创新点**

创新点在于：① 通过迭代分布匹配生成可信稀疏（condensed）数据；② 设计攻击容忍的稀疏数据生成（ACDG）与负贡献基拒绝的鲁棒聚合（RA）两阶段方案。

**🔧 技术方法**

使用的技术包括：迭代分布匹配、对比学习 + GMM 的标签校正、稀疏数据生成、负贡献评估、DBSCAN 聚类、Median 归一化、Mixup、InfoNCE、伪标签更新等。

**📊 数据集**

实验数据集：CIFAR-10、CIFAR-100、Tiny-20。

**📈 对比分析**

与 FedAVG、Bulyan、Multi‑Krum、Trimmed‑mean、FLTrust、FedDef 等方法对比，在 LIE、STAT‑OPT、DYN‑OPT、SLF、DLF 等多种强攻击下，FedIDM 的 TER 显著更低，收敛速度最快、最稳定。

**⚠️ 局限性**

局限性：① 随着数据异质性增大，性能会有所下降；② 需要较多的超参数调优和额外的稀疏数据生成/聚合计算，导致额外的计算开销。

---

## 447. IUQ: Interrogative Uncertainty Quantification for Long-Form Large Language Model Generation

**arXiv ID:** 2604.15109 | [PDF](https://arxiv.org/pdf/2604.15109v1)

**作者:** Haozhi Fan `[一作]` (University of Pennsylvania), Kaidi Xu `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种Interrogative Uncertainty Quantification（IUQ）框架，用于量化长文本生成的LLM输出不确定性；

**💡 创新点**

创新点在于同时利用跨样本一致性和样本内可信度，通过“提问-回答”机制对每个事实主张进行细粒度探测，从而捕捉模型的幻觉与自相矛盾行为；

**🔧 技术方法**

技术上采用LLM完成主张抽取、问题生成与回答，并通过主张可信度评估、影响核卷积、熵等手段计算不确定性；

**📊 数据集**

实验使用FActScore（人物传记）和LongFact（多领域长文本）两大公开数据集；

**📈 对比分析**

与白盒（Max Token Entropy、Perplexity、CCP、Frequency Scoring）和黑盒（Claim Entailment、Closeness Centrality）基线比较，IUQ在AUROC和AUPRC上均优于其它方法，尤其在FActScore上达0.748；

**⚠️ 局限性**

局限性包括：依赖LLM推理可能带来新的幻觉，未处理模型拒绝回答的情况，且多阶段推理导致计算成本显著提高。

---

## 448. Analysis of Multitasking Pareto Optimization for Monotone Submodular Problems

**arXiv ID:** 2604.15068 | [PDF](https://arxiv.org/pdf/2604.15068v1)

**作者:** Liam Wigney `[一作]` (Adelaide University), Frank Neumann `[通讯]` (Adelaide University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种多任务Pareto优化框架，用于在共享同一单调子模函数但约束不同的多重子模最大化问题中一次性求解所有子问题。

**💡 创新点**

创新点在于将传统的单任务演化算法扩展为多任务形式，通过共享主目标函数并对每个约束设定负成本次目标，从而构造出小而有效的Pareto前沿，理论上可实现多任务间知识迁移并提升收敛速度。

**🔧 技术方法**

采用的技术主要是全局简单多目标进化优化器（GSEMO），并基于子模函数的单调性与子模性进行严格的期望运行时间分析；同时在实验中使用了随机比特翻转的变异与基于主次目标的主导关系判定。

**📊 数据集**

实验使用了最大覆盖（Maximum Coverage）问题，选取了六个社交网络数据集：ca-GrQc、Erdos992、ca-HepPh、ca-AstroPh、ca-CondMat 和 ca-CondMat 等；各数据集规模从几百到几千节点不等。

**📈 对比分析**

通过在相同计算预算（固定代价函数评估次数）下对比经典单任务GSEMO与多任务GSEMO，发现当约束上界相近或随机/度数加权约束时，多任务方法在大规模或中等规模问题上能获得更好的近似；但在单位约束且上界分散时，多任务往往不如单任务，整体性能呈现依赖问题相似度的差异。

**⚠️ 局限性**

主要局限在于理论分析仅适用于统一成本（同权重）约束；当约束权重不一致或存在动态/非单调或非子模问题时，分析和优势无法直接推广；实验结果也显示多任务在约束差异较大时易受搜索空间扩大的负面影响，实际收益高度依赖问题相似度。

---

## 449. Emulation-based System-on-Chip Security Verification: Challenges and Opportunities

**arXiv ID:** 2604.15073 | [PDF](https://arxiv.org/pdf/2604.15073v1)

**作者:** Tanvir Rahman `[一作]` (University of Florida), Mark Tehranipoor `[通讯]` (University of Florida)

**通讯引用:** 6672 | [OpenAlex ID](https://openalex.org/A5073054890)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述并系统化了基于硬件仿真（Emulation）的SoC安全验证技术与方法，提出了工作流分类与关键挑战；

**💡 创新点**

创新点在于将仿真平台与验证工作流分离，构建三维（刺激/工作流/可观测性）分类法，并指出AI驱动、故障与模糊测试耦合、数字孪生等未来研究方向；

**🔧 技术方法**

采用了硬件仿真技术、信息流跟踪（IFT）、断言检查、模糊/渗透测试、故障注入、侧信道分析以及AI/LLM辅助分析等多种验证技术；

**📊 数据集**

无具体数据集，主要使用公开的安全验证案例、行业标准威胁模型和已公开的SoC RTL示例；

**📈 对比分析**

文章未进行实验比较，更多以定性评述与趋势预测为主，无法给出量化性能指标；

**⚠️ 局限性**

局限性包括：1）综述性缺乏统一的安全覆盖度量与基准；2）工作流与平台的集成缺乏细粒度工具支持；3）对实际工业验证流程的可操作性与成本评估不足。

---

## 450. Blinded Multi-Rater Comparative Evaluation of a Large Language Model and Clinician-Authored Responses in CGM-Informed Diabetes Counseling

**arXiv ID:** 2604.15124 | [PDF](https://arxiv.org/pdf/2604.15124v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 451. Attention-Gated Convolutional Networks for Scanner-Agnostic Quality Assessment

**arXiv ID:** 2604.15059 | [PDF](https://arxiv.org/pdf/2604.15059v1)

**作者:** Chinmay Bakhale `[一作]` (Indian Institute of Technology), Anil Sao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种混合CNN‑注意力网络，用于扫描无关的MRI质量评估；

**💡 创新点**

创新点在于将多头跨注意力模块与传统卷积编码器相结合，实现场景无关的运动伪影检测；

**🔧 技术方法**

采用了2D残差CNN编码器、跨注意力重加权层以及多层感知机分类头；

**📊 数据集**

训练数据使用MR‑ART数据集（200名受试者的T1加权扫描），测试数据包括MR‑ART的保留集和ABIDE档案中的17个不同站点共200名受试者；

**📈 对比分析**

与传统基于IQM+SVM、3D‑CNN、JMRI等方法对比，本文模型在MR‑ART上实现99.2%准确率、98.4%灵敏度，在ABIDE未见站点上仍保持75.5%准确率、84.6%灵敏度；

**⚠️ 局限性**

主要局限是样本量相对有限、未在更大规模多中心数据上验证、且仅评估了单一注意力配置，缺乏更细粒度的比较与对比研究。

---

## 452. Fabricator or dynamic translator?

**arXiv ID:** 2604.15165 | [PDF](https://arxiv.org/pdf/2604.15165v1)

**作者:** Lisa Vasileva `[一作]` (Language Weaver), Karin Sim `[通讯]` (Language Weaver)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型在机器翻译中的过度生成（overgeneration）问题，并提出并评估了两种检测策略（MTQE微调与基于对齐的CheckAlign）以及其集成模型。

**💡 创新点**

提出了一套细粒度的过度生成分类体系（振荡、分离、部分分离、最小分离）并针对最小分离难以检测的情况进行专门实验，首次在商业环境下系统性评估多语言LLM翻译的过度生成。

**🔧 技术方法**

利用XLM‑R大模型进行质量回归微调，构建MTQE；使用AwesomeAlign（基于BERT的跨语言对齐）并对齐策略进行微调，形成CheckAlign；二者结合构成集成方法。

**📊 数据集**

使用公开数据（WMT24‑AOC、DeepSpin）和内部数据（R&D、APE、POC、Synthetic、MinDet、客户案例）共计约1.8万条样本，涵盖多语言对（en‑zh、en‑ru、en‑ja、en‑de、en‑it、en‑fr、en‑nl等）。

**📈 对比分析**

在公开数据上，MTQE检测准确率最高（≈0.98）但召回低；CheckAlign在召回方面更好（≈0.89），集成模型在整体F1上实现平衡（≈0.82–0.89）。在内部数据集上，集成模型显著提升了整体表现，尤其在最小分离样本中召回率上升至≈0.77。

**⚠️ 局限性**

主要局限：对齐误差导致误报率高；最小分离过度生成与“显式化”（explicitation）难以区分；现有模型对低资源语言和多样化文本表现仍有限。

---

## 453. Trajectory Planning for a Multi-UAV Rigid-Payload Cascaded Transportation System Based on Enhanced Tube-RRT*

**arXiv ID:** 2604.15074 | [PDF](https://arxiv.org/pdf/2604.15074v1)

**作者:** Jianqiao Yu `[一作]` (Beijing Institute of Technology), Tianhua Gao `[通讯]` (University of Tsukuba)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5032870327)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了针对多UAV级联吊装系统的两阶段轨迹规划框架：首先使用改进的Enhanced Tube‑RRT*在高密集障碍环境中生成可行的安全管道路径；随后利用凸二次规划联合优化路径与电缆张力，得到平滑且满足动态、张力与姿态约束的轨迹，并通过集中几何控制实现闭环跟踪。

**💡 创新点**

创新点包括：1）主动混合采样与贝叶斯后验更新的子区分布采样策略，显著提高搜索效率；2）自适应扩展结合潜在场、转角约束与软最小化步长，既保证安全又提升路径光滑度；3）在规划阶段直接引入张力、姿态、运动约束的凸优化，避免规划-控制不匹配导致的不可行性；4）完整的两阶段管道实现了从搜索到跟踪的一体化闭环方案。

**🔧 技术方法**

使用技术包括：RRT*改进（Enhanced Tube‑RRT*），贝叶斯主动采样与Thompson采样，潜在场引导的自适应扩展，转角能量惩罚的组合代价函数，凸二次规划（CQP）联合轨迹与张力优化，SOC约束保证张力方向与角度限制，集中几何控制器实现轨迹跟踪。

**📊 数据集**

实验基于仿真环境：50 m × 50 m × 15 m 的三维工作空间，随机生成多形体障碍；起点为(6, 2, 10) m，终点为(40, 40, 5) m；使用30个立方体障碍，重复50个随机种子。

**📈 对比分析**

与两种基线（STube‑RRT* 与 AETube‑RRT*）对比，指标包括：成功率、有效采样比例、首次可行解时间、转角总和与路径长度。结果显示：Enhanced Tube‑RRT*成功率94%（vs 36%/28%），首次解时间最快1.13 s（vs 2.66/13.88 s），转角总和与路径长度均更小，整体路径质量和搜索效率显著优于基线。

**⚠️ 局限性**

局限性：1）仅在静态障碍环境下验证，动态障碍未考虑；2）实验仅为仿真，缺乏真实飞行验证；3）假设电缆长度和质量已知，未处理不确定性；4）对多级联系统或更大规模网络的扩展尚未探索；5）在极端张力或姿态约束紧张时，规划与控制的一致性仍可能受限。

---

## 454. Autonomous Evolution of EDA Tools: Multi-Agent Self-Evolved ABC

**arXiv ID:** 2604.15082 | [PDF](https://arxiv.org/pdf/2604.15082v1)

**作者:** Cunxi Yu `[一作]` (NVIDIA Research), Haoxing Ren `[通讯]` (NVIDIA Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于大型语言模型代理的自我进化逻辑综合框架，能够自动改写并优化流行的 ABC 逻辑综合工具的 120+ 万行 C 代码。

**💡 创新点**

创新点在于将多代理 LLM 结合严格的等价性校验、统一 QoR 评估和自我演化规则库，实现了跨模块、跨层级的大规模仓库级自动化改进，并首次在 EDA 工具上展示了持续性性能提升。

**🔧 技术方法**

核心技术包括：Claude 4.5 Sonnet 代理（规划与编码）、多代理协同架构、正式等价性检查（CEC）保证功能不变、基于八种合成流的 QoR 反馈循环，以及动态规则库演化。

**📊 数据集**

实验使用的验证数据集包括 ISCAS’85/89/99、VTR DSP、EPFL、IWLS 2005 等标准门级电路，所有合成均基于 ASAP7 7 nm PDK 在八种不同流程下评估。

**📈 对比分析**

与基准 ABC（仅使用原始 FlowTune、AIG 及映射模块）以及各子系统单独/联合演化的 ablation 方案进行对比，归一化 QoR 最终下降至 0.917，提升约 8.3%，最差负时差平均改进 8–9%，面积‑延迟乘积亦下降 8.3%。

**⚠️ 局限性**

局限性包括：需大量先验域知识进行系统初始化，完全新颖的算法方案生成效果有限，依赖正式等价性检查和昂贵的评估资源，且对不同 EDA 领域的迁移性尚未充分验证。

---

## 455. Data Engineering Patterns for Cross-System Reconciliation in Regulated Enterprises: Architecture, Anomaly Detection, and Governance

**arXiv ID:** 2604.15108 | [PDF](https://arxiv.org/pdf/2604.15108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 456. Codes with Large Minimum Distance in Product Codes: Explicit Constructions and Bounds

**arXiv ID:** 2604.15080 | [PDF](https://arxiv.org/pdf/2604.15080v1)

**作者:** Amit Berman `[一作]` (Samsung Semiconductor Israel), Itzhak Tamo `[通讯]` (Samsung Semiconductor Israel)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计了一种显式构造的 Reed–Solomon 乘积码子码，能够在保持两行两列局部可恢复性的同时显著提升全局最小距离。

**💡 创新点**

创新点在于：①提出了一个专门针对乘积码网格结构的全新上界；②利用线性化多项式将二维评估映射为一维评估，从而在线性域大小仅线性增长的前提下实现子码的最优或近最优距离。

**🔧 技术方法**

采用线性化多项式、评估码、矩阵张量积、以及单一和多重恢复集的 LRC 理论，结合矩阵分块与乘法映射的交换图，构造并分析子码。

**📊 数据集**

论文主要为理论分析，不使用实验数据集；所有结果均基于符号域（如 𝔽_{q^2}) 的抽象计算与示例性取值（如 q=2^5、r=8）演示。

**📈 对比分析**

通过对比构造子码的下界与新提出的上界、以及传统单恢复集上界，实验示例表明在 k=r^2-1、k=r^2-2 时子码距离达到上界，且在大多数参数范围内距离至少为最优距离的 90%。

**⚠️ 局限性**

局限性包括：①上界在 k 接近 r^2 时的紧密性尚未完全验证；②构造的子码缺乏高效的解码算法；③对于高码率场景，距离仍存在较大差距；④是否可通过更大域尺寸进一步提升距离仍是开放问题。

---

## 457. HyperSpace: A Generalized Framework for Spatial Encoding in Hyperdimensional Representations

**arXiv ID:** 2604.15113 | [PDF](https://arxiv.org/pdf/2604.15113v1)

**作者:** Shay Snyder `[一作]` (George Mason University), Maryam Parsa `[通讯]` (George Mason University)

**通讯引用:** 1594 | [OpenAlex ID](https://openalex.org/A5078825022)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 HyperSpace 框架，用于构建、拆解和系统化评估连续空间向量符号架构（VSA）流水线，特别针对 HRR 与 FHRR 两种后端进行端到端性能与内存占用对比。

**💡 创新点**

创新点在于将 VSA 处理过程抽象为一系列可插拔模块（编码、绑定、束缚、相似度、反演、清理、回归），使不同后端在同一管道中可直接比较；通过系统级分析发现 HRR 与 FHRR 在整体延迟相近，且内存占用与运算分布各不相同，突破了以往单一算子比较的局限。

**🔧 技术方法**

使用的技术包括：抽象模块化设计、HRR 与 FHRR 后端实现、Resonator 与 Modern Hopfield 清理网络、Codebook 与神经网络回归方法，实验在 Apple M4 Pro CPU 上进行未加硬件加速的基准测评。

**📊 数据集**

数据集为合成的 2D 成本地图，采用 28×28 网格采样，包含最优路径的欧氏距离变换，生成约 784 个样本的连续坐标–值对。

**📈 对比分析**

比较方法为对所有组合（后端、清理、回归）进行批量端到端时延与均方误差（MSE）评估，并绘制 Pareto 前沿。结果显示：HRR 与 FHRR 的总延迟相近，HRR 的内存占用约为 FHRR 的一半；FHRR 在单步运算上更快但清理与回归成本更高。

**⚠️ 局限性**

局限性包括：仅在未优化的单 CPU 环境下评估，未探讨 GPU 或专用硬件加速；仅测试 HRR 与 FHRR 两种后端；合成地图可能无法充分代表真实机器人环境；清理与回归方法的选择仍受限于实验设计。

---

## 458. From Procedural Skills to Strategy Genes: Towards Experience-Driven Test-Time Evolution

**arXiv ID:** 2604.15097 | [PDF](https://arxiv.org/pdf/2604.15097v1)

**作者:** Junjie Wang `[一作]` (EvoMap), Haoyang Zhang `[通讯]` (EvoMap)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了将可重用经验以控制信号形式编码，并提出了紧凑的策略基因（strategy gene）及其基因演化协议（GEP），通过在45个科学代码任务上进行系统对比实验验证其效果。

**💡 创新点**

创新点在于把经验从文档化的技能包装转变为控制导向、可压缩的基因，并为其设计演化协议，使其既能在推理时提供有效控制，又能作为持续进化的载体。

**🔧 技术方法**

使用Gemini 3.1 Pro/Flash大模型，构建基于OpenClaw和Evolver的基因演化系统；实验中运用了Gene Evolution Protocol、可编辑结构、失败历史压缩等技术。

**📊 数据集**

实验数据集包括45个科学代码解决场景（如蛋白质解析、光谱峰检测、行星探测等）以及CritPt基准，用于评估基因演化效果。

**📈 对比分析**

与无指导基线、完整文档化技能、节省预算的技能片段等进行对比；实验结果显示基因平均提升约3个百分点，完整技能反而下降；在CritPt上的基因演化系统相较于基线提升约9‑10个百分点。

**⚠️ 局限性**

局限在于基因的可组合性受限，多个基因叠加往往削弱效果；实验仍局限于特定科学任务，未验证在更广泛领域的通用性。

---

## 459. Dual Pose-Graph Semantic Localization for Vision-Based Autonomous Drone Racing

**arXiv ID:** 2604.15168 | [PDF](https://arxiv.org/pdf/2604.15168v1)

**作者:** David Perez-Saura `[一作]` (Universidad Politica de Madrid), Pascual Campoy `[通讯]` (Universidad Politica de Madrid)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种双图结构的语义定位系统，将门检测信息先聚合到临时图中，再压缩后加入主图，实现赛道快速、高精度定位。

**💡 创新点**

创新点在于：①双图压缩机制在保持多次检测信息的同时防止图增长；②在主图中自动产生循环闭环约束；③支持多传感器的无关结构。

**🔧 技术方法**

使用位姿图优化（Levenberg–Marquardt+CHOLMOD），OpenVINS视觉惯性里程计，AlphaPilot风格门检测网络，PnP估计，Hungarian算法数据关联，ROS2/Aerostack2框架。

**📊 数据集**

Race Against the Machine（RA2M）无人机竞速数据集以及真实的 A2RL 竞赛飞行序列。

**📈 对比分析**

与单图架构和仅用 OpenVINS 的基线对比，使用 ATE、节点/边数和优化时长评估；双图在相同计算成本下，翻倍减少 ATE（约 50%），并在实时竞赛中将门跨越误差从 1.5 m 提升到 4.2 m 的补正。

**⚠️ 局限性**

优化频率受关键帧阈值限制，可能导致在极高更新速率下校正响应不足。

---

## 460. Compressing Sequences in the Latent Embedding Space: $K$-Token Merging for Large Language Models

**arXiv ID:** 2604.15153 | [PDF](https://arxiv.org/pdf/2604.15153v1)

**作者:** Zihao Xu `[一作]` (Rutgers University), Hao Wang `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种K-Token Merging方法，在潜在空间对连续K个token嵌入进行压缩，从而减少LLM的输入长度。

**💡 创新点**

创新点在于在嵌入空间而非token空间进行压缩，并通过轻量级编码器与LoRA微调相结合，兼顾高压缩率和低性能损失。

**🔧 技术方法**

使用的技术包括轻量级MLP编码器、平均初始化策略、LoRA适配以及对Qwen-2.5 0.5B模型的微调。

**📊 数据集**

实验采用文本化树（Textualized Tree）结构推理、亚马逊评论（Amazon Reviews）情感分类、以及代码编辑（CommitPackFT）三大数据集。

**📈 对比分析**

与SelectiveContext、LLMLingua2和LTSC等硬/软压缩基线对比，K-Token Merging在保持≈1%准确率/1.5% perplexity损失的前提下，实现了高达75%的长度压缩，P‑L F1得分位于Pareto前沿。

**⚠️ 局限性**

局限性包括仅对输入进行压缩、固定K值且未对生成阶段压缩、以及仅在小型Qwen 0.5B模型上验证，缺乏对更大模型或自适应压缩的探讨。

---

## 461. IG-Search: Step-Level Information Gain Rewards for Search-Augmented Reasoning

**arXiv ID:** 2604.15148 | [PDF](https://arxiv.org/pdf/2604.15148v1)

**作者:** Zihan Liang `[一作]` (Kuaishou Technology), Wenwu Ou `[通讯]` (Kuaishou Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IG-Search，一种将检索步骤信息增益作为奖励的强化学习框架，实现在单步层面给检索查询分配信用

**💡 创新点**

创新点在于用信息增益（IG）衡量每一步检索的实际价值，并通过 GRPO 的 per-token 优势调制实现细粒度奖励，避免了传统轨迹级奖励的稀疏和相同奖励无法区分查询质量的问题

**🔧 技术方法**

技术主要包括：基于信息增益的奖励计算、GRPO 与 per-token 先进调制、几种稳定化机制（dead zone、负值非对称缩放、软截断、查询长度归一化）以及大规模 LLM（Qwen2.5-3B/7B）与检索器 E5-base-v2 的集成

**📊 数据集**

在七个问答基准上评估，涵盖单跳（NQ、TriviaQA、PopQA）与多跳（HotpotQA、2WikiMultihopQA、Musique、Bamboogle）数据集

**📈 对比分析**

与多种基线比较（SFT、R1、Naive RAG、IRCoT、Search‑o1、Search‑R1、ReSearch、StepSearch、AutoRefine、MR-Search、GiGPO）相比，IG-Search 在 Qwen2.5-3B 上平均 EM 达 0.430，超越轨迹级最强基线 MR-Search 1.6 点，超过步级 GiGPO 0.9 点；在 7B 规模亦保持领先，平均 EM 0.479

**⚠️ 局限性**

局限包括：需要训练时访问 gold answer，难以直接迁移到无监督场景；稳定化超参数需在不同任务间重新调优；检索使用离线 Wikipedia，缺乏实时信息

---

## 462. Feedback-Driven Execution for LLM-Based Binary Analysis

**arXiv ID:** 2604.15136 | [PDF](https://arxiv.org/pdf/2604.15136v1)

**作者:** XiangRui Zhang `[一作]` (Beijing Jiaotong University), Haining Wang `[通讯]` (Virginia Tech)

**通讯引用:** 9687 | [OpenAlex ID](https://openalex.org/A5100664241)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

设计并实现了一个基于大语言模型的迭代执行系统——FOA，能够通过反馈驱动的多代理模型对二进制程序进行动态、增量的漏洞分析；

**💡 创新点**

创新之处在于把LLM分析视作反馈驱动的执行过程，提出动态森林（FoA）结构实现任务分解、并行探索与局部上下文限制，并构建发现‑验证闭环；

**🔧 技术方法**

采用DeepSeek‑v3 LLM与Radare2等工具的交互，构建推理–动作–观测循环、动态代理生成与层次聚合，以及LLM引导的语义剪枝技术；

**📊 数据集**

使用了3,457个来自NETGEAR、D‑Link、TP‑Link、Tenda的固件二进制，来源于Karonte数据集；

**📈 对比分析**

与Mango、SaTC、LATTE、SWE等基线对比，FOA在3,457个二进制上发现1,274个漏洞，精度72.3%，覆盖更多漏洞类型，平均每个二进制耗时43.8分钟，成本/漏洞约为单代理的1/2.5；

**⚠️ 局限性**

局限在于受底层分析工具精度限制，路径覆盖不完全，LLM偶尔会幻觉，结果在不同跑中可能存在变异，验证阶段依赖已收集证据，无法完全覆盖所有隐藏路径。

---

## 463. SRMU: Relevance-Gated Updates for Streaming Hyperdimensional Memories

**arXiv ID:** 2604.15121 | [PDF](https://arxiv.org/pdf/2604.15121v1)

**作者:** Shay Snyder `[一作]` (George Mason University), Maryam Parsa `[通讯]` (George Mason University)

**通讯引用:** 1594 | [OpenAlex ID](https://openalex.org/A5078825022)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Sequential Relevance Memory Unit (SRMU)，一种基于向量符号架构 (VSA) 的序列关联记忆更新规则，用于在非均衡采样和非平稳动态的流式环境中构建更稳健的记忆。

**💡 创新点**

创新点在于将时序衰减与基于相似度的相关性门控结合，动态调节新观测对记忆的影响；相比传统简单加法或仅衰减的更新，SRMU 能同时抑制冗余更新并保留有用信息，且不依赖领域特定的预处理或清理。

**🔧 技术方法**

使用技术包括 VSA 的高维向量表示、绑定/捆绑/解绑定运算、余弦相似度测量、时序衰减参数 γ 以及基于相似度的权重 w 进行门控更新。

**📊 数据集**

实验数据集为自定义的合成设备健康监测数据：5 台设备、5 个健康状态，设计三种实验（非均衡采样、非平稳动态、两者组合）以检验 SRMU 在不同挑战下的表现。

**📈 对比分析**

与简单加法更新和单纯时间衰减基线进行比较，使用余弦相似度评估检索准确性、记忆范数评估表示稳定性；SRMU 在三种实验中分别提升相似度 12.6% 并将记忆范数降低 53.5%，尤其在组合实验中获得最高相似度 0.850 与最低范数 74.01。

**⚠️ 局限性**

局限性包括：仅在合成实验验证，缺乏对真实大规模机器人或分布式感知系统的实测；未对容量和干扰理论进行形式化分析；实现效率与硬件部署（如神经形态平台）的适配仍待研究；对极端噪声或异常观测的鲁棒性尚未充分评估。

---

## 464. Metric-agnostic Learning-to-Rank via Boosting and Rank Approximation

**arXiv ID:** 2604.15101 | [PDF](https://arxiv.org/pdf/2604.15101v1)

**作者:** Camilo Gomez `[一作]` (University of Central Florida), Yanjie Fu `[通讯]` (Arizona State University)

**通讯引用:** 6325 | [OpenAlex ID](https://openalex.org/A5032187620)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于快速软排序近似的均方误差损失的梯度提升学习排序方法 SoftRankGBM。

**💡 创新点**

创新点在于将可微软排序运算与平均均方误差损失相结合，形成与评估指标无关的列表级学习框架，并改进梯度提升树以兼容列表结构。

**🔧 技术方法**

采用 fast‑soft‑ranking 近似、均方误差损失、梯度提升回归树（GBRT）以及自定义梯度计算实现该方法。

**📊 数据集**

使用公开的 LETOR 基准数据集 C14!（Yahoo）和 Web10k（Bing）进行实验。

**📈 对比分析**

与多种 LambdaMART 实现（LightGBM、XGBoost、RankLib）以及 Adarank 对比，SoftRankGBM 在 NDCG 与 MAP 的各截断级别上均实现了最优或次优的性能，并在训练时间上仅次于 LightGBM。

**⚠️ 局限性**

局限在于仍需针对不同任务调参（如软排序参数 ϵ、叶节点数等），并且在极大规模数据集上软排序的计算开销和模型训练时间仍高于传统的点对/列表对方法。

---

## 465. Beyond Independent Frames: Latent Attention Masked Autoencoders for Multi-View Echocardiography

**arXiv ID:** 2604.15096 | [PDF](https://arxiv.org/pdf/2604.15096v1)

**作者:** Simon Böhi `[一作]` (University of Basel), Julia E. Vogt `[通讯]` (ETH Zurich)

**通讯引用:** 1770 | [OpenAlex ID](https://openalex.org/A5045935456)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了一种适用于多视角超声影像的自监督预训练模型——Latent Attention Masked Autoencoder（LAMAE），并将其应用于ICD-10代码预测和转移学习。

**💡 创新点**

主要创新在于引入latent attention模块，使模型能够在潜在空间中跨视角、跨帧进行信息交换，克服传统MAE无法聚合多视角数据的限制。

**🔧 技术方法**

使用MAE框架、Transformer注意力机制、视频MAE以及自定义的latent attention模块进行预训练与微调。

**📊 数据集**

预训练数据为大规模MIMIC-IV‑ECHO超声视频，后续转移评估在EchoNet‑Dynamics和EchoNet‑Pediatrics数据集上进行。

**📈 对比分析**

与Image‑MAE、Video‑MAE基线相比，LAMAE在ICD‑10代码的AUROC提升约0.01–0.02，且在EchoNet‑Pediatrics的LVEF MAER下降约7%，显示出更优性能。

**⚠️ 局限性**

仍存在仅处理B‑mode视频、随机视角采样、缺乏显式跨帧重建策略，以及单视角场景下优势相对有限等限制。

---

## 466. ControlFoley: Unified and Controllable Video-to-Audio Generation with Cross-Modal Conflict Handling

**arXiv ID:** 2604.15086 | [PDF](https://arxiv.org/pdf/2604.15086v1)

**作者:** Jianxuan Yang `[一作]` (MiLM Plus, Xiaomi Inc.), Jian Luan `[通讯]` (MiLM Plus, Xiaomi Inc.)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ControlFoley框架，实现视频、文本、参考音频多模态可控音频生成。

**💡 创新点**

创新点包括：联合视觉编码融合CLIP与CAV-MAE-ST提升文本可控；时域-音色分离去除参考音频时序干扰；统一REPA训练增强多模态鲁棒性；并构建VGGSound‑TVC评测数据集。

**🔧 技术方法**

使用多模态扩散Transformer（MMDiT）搭配CAV‑MAE‑ST、CLIP、CLAP、时域音色解耦以及REPA对齐等技术。

**📊 数据集**

训练与评估数据集包括VGGSound（训练/测试）、Kling‑Audio‑Eval、MovieGen‑Audio‑Bench、AudioCaps/WavCaps/Clotho、Greatest Hits以及新建的VGGSound‑TVC。

**📈 对比分析**

与MMAudio、HunyuanVideo-Foley、ThinkSound、AudioX、CondFoleyGen等基线对比，ControlFoley在语义对齐、时序同步、音质及分布匹配等指标上均达SOTA或显著提升。

**⚠️ 局限性**

局限性：文本控制受限于简化注释；音色细粒度控制难度仍大；模型仍需大量视频-音频-文本对；对极端跨模态冲突或多事件复杂场景的适应性待进一步提升。

---

## 467. When Flat Minima Fail: Characterizing INT4 Quantization Collapse After FP32 Convergence

**arXiv ID:** 2604.15167 | [PDF](https://arxiv.org/pdf/2604.15167v1)

**作者:** Marcus Armstrong `[一作]` (University of Houston), Marcus Armstrong `[通讯]` (University of Houston)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5075543992)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Pythia-160M模型进行全程量化鲁棒性审计，发现INT4量化在训练后期会出现爆炸性误差增加；

**💡 创新点**

揭示三阶段结构——快速学习、元稳定平台、爆炸性发散，并证明发散与FP32收敛同步，而非学习率衰减；同时证明INT8不受影响，排除离群值积累机制；并验证调度幅度对量化鲁棒性的决定性作用；

**🔧 技术方法**

使用校准无关的按组INT4探测器、kurtosis度量、对比SGDR与自定义振荡锁定（OLI）学习率调度；

**📊 数据集**

基于Pythia-160M模型（160M参数，使用The Pile数据集，训练300B tokens）并评估154个公开检查点；

**📈 对比分析**

与原Cosine调度对比，SGDR加剧INT4误差，OLI在冷却阶段平均降低约2.2个百分点（p<0.0001），但FP32 perplexity增加；INT8误差始终<1%；

**⚠️ 局限性**

局限于单一模型规模与数据集；OLI提高量化鲁棒性但伴随显著FP32性能损失；未给出导致INT4不兼容的具体权重配置机理；

---

## 468. Structure as Computation: Developmental Generation of Minimal Neural Circuits

**arXiv ID:** 2604.15143 | [PDF](https://arxiv.org/pdf/2604.15143v1)

**作者:** Duan Zhou `[一作]` `[通讯]` (Independent Researcher), Duan Zhou (Independent Researcher)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过从单个干细胞出发，利用基因调控规则模拟大脑发育，生成极小化的85神经元密集网络，并在 MNIST 与 CIFAR‑10 上仅训练一轮即可获得高精度。

**💡 创新点**

创新点在于：①用生物学驱动的固定发育规则生成网络拓扑，而非梯度学习结构；②极度稀疏的成熟神经元但高度连通的核心能够快速学习，展示了域泛化的结构先验；③通过发育过程自然产生的高维投影，为梯度优化提供了有利的损失景观。

**🔧 技术方法**

技术手段包括：Boolean 规则推断、单细胞转录组时间序列分析、细胞分裂/迁移/分化模拟、基因表达与空间兼容性突触生成、固定权重循环层（85×85）与 ReLU+softmax 输出、Adam 优化仅训练输入投影和输出投影。

**📊 数据集**

使用的数据集：mouse cortical single‑cell RNA‑seq (GSE211140)、MNIST、CIFAR‑10。

**📈 对比分析**

实验对比：同等密度的随机拓扑网络与随机初始化的常规网络；在 MNIST 上，迭代0时≈10%，单轮后提升至 92%（>80%提升）；在 CIFAR‑10 上，单轮达到 40.53%（四倍随机基线），最终稳定在约50%；随机拓扑无快速学习现象，性能保持在随机水平。

**⚠️ 局限性**

局限性：仅在 MNIST 与 CIFAR‑10 两个小型数据集验证；网络极小导致在更复杂任务上性能受限；未考虑活动依赖的突触可塑性；未研究更大规模神经元群或多层/卷积结构；缺乏对发育规则细节的生物学实证支持。

---

## 469. Beyond Visual Cues: Semantic-Driven Token Filtering and Expert Routing for Anytime Person ReID

**arXiv ID:** 2604.15090 | [PDF](https://arxiv.org/pdf/2604.15090v1)

**作者:** Jiaxuan Li `[一作]` (Tsinghua University), Zhihang Li `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 1710 | [OpenAlex ID](https://openalex.org/A5101804803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于大型视觉语言模型生成身份内在属性文本的语义驱动人像重识别框架 STFER，结合语义驱动视觉标记过滤（SVTF）和语义驱动专家路由（SER）实现任时人像重识别。

**💡 创新点**

创新点包括：①利用 LVLM 生成身份不变的文本描述作为全局身份锚；②通过文本-图像跨注意力实现视觉标记过滤，剔除背景和可变区域；③在专家路由中同时使用 CLS 与文本信息，精准分配针对不同场景（昼/夜、短/长期）的专家网络。

**🔧 技术方法**

使用 Qwen3‑VL‑4B 作为 LVLM，ViT‑Base/16 作为视觉主干；在训练中加入文本编码、语义驱动注意力、SVTF 与 SER 机制；优化器为 SGD，采用学习率热身与余弦衰减。

**📊 数据集**

主要使用 AT‑USTC 数据集（270 个人，RGB+IR，昼夜、短期/长期多场景），并在 Market1501、CUHK03、SYSU‑MM01、PRCC、LTCC 5 个跨域数据集上进行泛化测试。

**📈 对比分析**

在 AT‑USTC 上，STFER 在所有 6 种场景下均优于现有最先进方法，Any‑Time 级别 Rank‑1 为 94.54%，mAP 为 93.46%；在跨域测试中，平均 Rank‑1 达 74.33%，mAP 75.26%，显著高于同类方法。

**⚠️ 局限性**

局限性：LVLM 在低质量、遮挡或分辨率低的图像上可能产生中性或不准确的文本描述，影响后续过滤与路由；模型对 LVLM 的依赖导致推理时额外的计算和存储开销。

---

## 470. DiscoTrace: Representing and Comparing Answering Strategies of Humans and LLMs in Information-Seeking Question Answering

**arXiv ID:** 2604.15140 | [PDF](https://arxiv.org/pdf/2604.15140v1)

**作者:** Neha Srikanth `[一作]` (University of Maryland), Rachel Rudinger `[通讯]` (University of Maryland)

**通讯引用:** 1082 | [OpenAlex ID](https://openalex.org/A5082447472)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“答案修辞策略”框架，利用RST解析、话语行为本体和问题解释序列化对长篇回答进行结构化表示，并对9个Reddit社区的人类回答与LLM回答进行对比分析，揭示人类回答的修辞多样性与LLM缺乏多样性的差距。

**💡 创新点**

① 构建将答案拆解为与问题解释相关的语篇行为序列的框架；② 设计并标注了21种话语行为本体；③ 自动生成并匹配问题解释；④ 通过交叉熵比较社区与LLM的回答策略，首次系统量化LLM在修辞多样性上的不足。

**🔧 技术方法**

使用xLM‑RoBERTa‑large端到端 RST 解析器进行文本解析；GPT‑4.1 与 Claude 模型用于话语行为标注与解释生成；二元语言模型（bigram）用于学习答案序列并计算交叉熵；人类评标与 Cohen’s κ 统计验证标注质量。

**📊 数据集**

数据来源为 Reddit 问答数据，挑选9个子版块（AskEconomics、AskHistorians、asklinguistics、NoStupidQuestions、OutOfTheLoop、explainlikeimfive、ScienceBasedParenting、beyondthebump、history），共约 18,968 条首层评论；同时使用 Claude‑4.5、Claude‑sonnet‑4.5、Qwen3‑32b 等 LLM 生成答案进行对比。

**📈 对比分析**

通过训练每个社区的二元模型，对其他社区答案计算交叉熵；人类社区自熵低、跨社区熵高，表明不同社区的修辞策略显著差异；LLM答案熵远高于人类，显示其修辞结构更统一。即使给定社区指南，LLM熵下降但仍高于人类自熵；LLM在低频解释上的覆盖率高于人类，表明其过度泛化。

**⚠️ 局限性**

仅使用 Reddit 数据，未涵盖 Quora、StackOverflow 等平台；只分析首层评论，忽略多轮交互；未利用帖子正文作为额外上下文；社区特征可能受整体平台规范影响；LLM 训练与提示设置可能对结果产生影响。

---

## 471. OpenMobile: Building Open Mobile Agents with Task and Trajectory Synthesis

**arXiv ID:** 2604.15093 | [PDF](https://arxiv.org/pdf/2604.15093v1)

**作者:** Kanzhi Cheng `[一作]` (Nanjing University), Dahua Lin `[通讯]` (SenseTime)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 OpenMobile 框架，用于在 Android 生态中自动生成高质量、多样化的任务指令和对应的轨迹数据，从而训练视觉语言模型驱动的移动代理。

**💡 创新点**

创新点在于：①将任务合成从探索阶段和指令生成阶段解耦，利用全局环境记忆（短期与长期）来构造多步、跨功能的指令；②引入错误干预式策略切换，在轨迹收集时交替使用学习者与专家模型，系统性地捕获错误恢复示例。

**🔧 技术方法**

主要技术包括：视觉语言模型（Qwen2.5-VL、Qwen3-VL）进行功能描述与指令生成；感知哈希与语义检索构建全局环境记忆；策略切换算法结合监测器触发专家干预；强化学习框架用于后续模型微调。

**📊 数据集**

数据集：在 AndroidWorld 真实环境中合成约 2.8K 条指令、34K 步长轨迹，覆盖 20 个常用 Android 应用；评测基准为 AndroidWorld、AndroidLab 和 MobileWorld 三大动态移动代理基准。

**📈 对比分析**

对比方法：与公开数据基线（AndroidControl、AMEX）以及工业闭源系统（Operator、Anthropic Computer-Use 等）进行 Pass@1/Pass@3 对比；在 AndroidWorld 上 Fine-tuned Qwen3-VL 达到 64.7% Pass@1，远超公开基线约 30% 并逼近闭源 70% 级别；在 AndroidLab 与 MobileWorld 上也显著提升，表明数据质量和错误恢复训练有效。

**⚠️ 局限性**

局限性：①合成数据仍基于 AndroidWorld，跨平台或非 Android 生态的泛化需要进一步验证；②指令生成高度依赖 VLM 的功能抽取，可能对细粒度 UI 元素识别不足；③策略切换中的阈值和监测机制为手工设定，缺乏自动化自适应能力。

---

## 472. No More Guessing: a Verifiable Gradient Inversion Attack in Federated Learning

**arXiv ID:** 2604.15063 | [PDF](https://arxiv.org/pdf/2604.15063v1)

**作者:** Francesco Diana `[一作]` (Université Côte d'Azur), Giovanni Neglia `[通讯]` (Université Côte d'Azur)

**通讯引用:** 3110 | [OpenAlex ID](https://openalex.org/A5085721523)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了可验证梯度反演攻击（VGIA），在恶意服务器场景下通过裁剪超平面搜索实现对联邦学习中共享梯度的准确反演，并提供重构正确性的明确证明。

**💡 创新点**

创新点在于：①利用代数子空间验证方法对超平面切片进行可验证的单样本判定；②自适应地分配超平面查询，避免过度细分空/多样本区间；③在保证精确恢复输入特征的同时，可对连续回归目标进行解析恢复，实现连续目标的首次可验证重构。

**🔧 技术方法**

主要技术包括：几何ReLU泄漏分析、代数子空间投影与判定、可变权重层配置保证后续层线性区间恒定、批量梯度比值推导以及基于多超平面差分的样本向量恢复。

**📊 数据集**

实验数据集包括三个表格数据集（ACS Income、King County Housing、HARUS）和一个图像数据集（CIFAR10）。

**📈 对比分析**

与基准CTP攻击比较，VGIA在相同随机种子下实现了更快的收敛（更少攻击轮次）且几乎无误报；在大批量、不同ε设置下，VGIA能完整恢复所有样本并提供可验证性，而CTP受ε选择敏感，易出现误报或失败。

**⚠️ 局限性**

局限性包括：仅适用于全连接网络结构；依赖恶意服务器可直接修改模型参数；对更复杂架构（如CNN、Transformer）尚未验证；在存在隐私增强机制（如差分隐私）时效果未知。

---

## 473. Boundary-Centric Active Learning for Temporal Action Segmentation

**arXiv ID:** 2604.15173 | [PDF](https://arxiv.org/pdf/2604.15173v1)

**作者:** Halil Ismail Helvaci `[一作]`, Sen-ching Samson Cheung `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种边界中心化的主动学习框架B-ACT，专门为时序动作分割任务分配稀疏监督，只在预测的动作边界处标注帧，且仅使用边界框内的上下文来训练模型。

**💡 创新点**

创新点在于：① 将主动学习拆分为视频级不确定性筛选和边界级不确定性加权选择；② 设计了融合局部不确定性、类别模糊度与时间梯度的三项边界得分，用于精准定位高信息量的切换点；③ 通过仅标注边界帧而保留周围未标注帧的上下文，显著降低标注成本。

**🔧 技术方法**

技术上使用了Monte Carlo Dropout来估计模型的不确定性；在视频级使用预测熵做排名；在边界级使用局部均值不确定性、top-1/top-2置信度差、预测分布梯度构成的边界得分；训练采用ASFormer结构并在边界框内做交叉熵监督。

**📊 数据集**

实验数据集包括GTEA、50Salads和Breakfast三个标准时序动作分割数据集，分别涵盖厨房操作、沙拉制作与早餐准备等多样场景。

**📈 对比分析**

与随机、熵、等距、分段随机、分段熵、核心集等传统主动学习基线以及Su等人的两阶段主动学习方法进行对比，B-ACT在相同稀疏标注预算下在Edit、F1和帧级准确率等指标上均显著优于对手，尤其在GTEA和50Salads上提升幅度最大。

**⚠️ 局限性**

局限性在于：① 边界选择依赖当前模型的预测，早期训练阶段模型不稳定时易误标注无效边界；② 采样策略不加入显式多样性约束，可能在重复动作序列中产生冗余查询，未结合多样性与不确定性共同优化。

---

## 474. Optimal last-iterate convergence in matrix games with bandit feedback using the log-barrier

**arXiv ID:** 2604.15242 | [PDF](https://arxiv.org/pdf/2604.15242v1)

**作者:** Come Fiegel `[一作]`, Vianney Perchet `[通讯]` (ENSAE Paris)

**通讯引用:** 1045 | [OpenAlex ID](https://openalex.org/A5067550667)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于对数障碍正则化的镜像下降算法，在零和矩阵游戏与广义形式游戏的 bandit 设置中实现了 t^-1/4 的最后一次迭代收敛；

**💡 创新点**

创新点在于采用 log‑barrier 正则化并在对偶空间进行分析，首次在不预知时间 horizon 的条件下获得了高概率最优收敛率；

**🔧 技术方法**

技术主要包括变尺度的对数障碍正则化、重要性采样估计伪梯度、镜像下降更新以及 martingale 过程的概率界定；

**📊 数据集**

论文没有使用具体数据集，全部以理论分析与假设模型为主；

**📈 对比分析**

方法通过理论证明与已有的固定时间或期望收敛率进行对比，表明在高概率下达到最优 t^-1/4 收敛；

**⚠️ 局限性**

局限在于证明过程极其繁琐、计算复杂度高（每步需遍历整个策略树），且未给出实验验证。

---

## 475. UrbanClipAtlas: A Visual Analytics Framework for Event and Scene Retrieval in Urban Videos

**arXiv ID:** 2604.15225 | [PDF](https://arxiv.org/pdf/2604.15225v1)

**作者:** Joel Perca `[一作]`, Jorge Poco `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个面向城市视频的事件与场景检索可视化分析框架，支持对视频片段进行自动描述、索引、查询以及交互式分析。

**💡 创新点**

创新点在于将视频自动字幕、向量数据库索引、LLM驱动的查询增强与叙事生成、YOLO‑World实时目标定位、Mask2Former静态布局提取等技术深度融合，实现端到端快速检索并提供自然语言叙事；同时通过交互式可视化提升用户体验。

**🔧 技术方法**

使用了 Python 后端（HTTP/WebSocket）、前端交互可视化库（未指明）、YOLO‑World（目标定位）、Mask2Former（静态布局提取）、Qdrant（向量数据库索引）、大型语言模型（温度 0.05 的查询增强、0.3 的叙事生成）等技术。

**📊 数据集**

数据集为城市交通视频，采用视频片段进行自动字幕生成并构建索引；文中未给出具体公开数据集名称，说明使用的是自研或合成的城市视频集。

**📈 对比分析**

性能方面，视频片段级描述平均耗时数十秒，整体响应延迟约 6–7 秒；在查询后，剪辑切换、实体检视等交互操作实现子秒级延迟。评估以端到端时延和交互响应为主要指标，表现符合实时检索需求。

**⚠️ 局限性**

局限性包括：对外部 API 与云模型的依赖导致网络带宽与 API 延迟对性能产生显著影响；存在推理误差与事实不符的失败案例；对复杂场景的实体抽取与关系推理仍有改进空间。

---

## 476. MambaSL: Exploring Single-Layer Mamba for Time Series Classification

**arXiv ID:** 2604.15174 | [PDF](https://arxiv.org/pdf/2604.15174v1)

**作者:** Yoo-Min Jung `[一作]` (Seoul National University), Leekyung Kim `[通讯]` (Seoul National University)

**通讯引用:** 3 | [OpenAlex ID](https://openalex.org/A5013331634)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了单层Mamba框架MambaSL，并在全30个UEA时间序列分类数据集上重新评估20个基线模型，构建了统一的实验协议；

**💡 创新点**

创新点包括四个针对TSC的设计假设（输入投影扩展、时变性模块化、跳跃连接移除、适应性池化）以及对Mamba结构的轻量级改造，显著提升了时间序列分类性能；

**🔧 技术方法**

采用了选择性状态空间模型Mamba、可学习的多头自适应池化、时间变/不变参数的二进制开关以及大规模超参数搜索等技术；

**📊 数据集**

使用了30个多变量UEA数据集（覆盖长度、维度、样本量多样）以及近期的ADFTD和FLAAP两套真实场景数据集；

**📈 对比分析**

在统一协议下与20个强基线进行公平对比，MambaSL平均准确率最高，比第二名高1.41个百分点，Wilcoxon检验显示差异显著；

**⚠️ 局限性**

局限性在于对单个数据集的优势有限，可能需要领域特定调整；仅基于Mamba的第一版，未探索后续版本带来的潜在改进。

---

## 477. Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation

**arXiv ID:** 2604.15190 | [PDF](https://arxiv.org/pdf/2604.15190v1)

**作者:** Ziyang Chen `[一作]` (Meituan Inc.), Xiang Zhao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于策略引导的双过程混合模拟框架（PGHS），用于在餐饮外卖平台上对商家运营策略进行群体级别的因果评估。

**💡 创新点**

核心创新点在于：① 通过宏观聚类+微观密度分割从用户行为轨迹中抽取可迁移的决策策略，并将其同时作为文本说明和向量先验；② 采用双过程思维——LLM推理分支和ML拟合分支——在共享策略空间上协同工作，弥补单一方法的偏差；③ 通过策略引导的融合机制，使两分支互补，从而显著降低模拟误差。

**🔧 技术方法**

技术手段包括：K‑Means + HDBSCAN 的分层聚类；基于LLM的解释生成与决策逻辑提取；预训练文本编码器得到策略向量；梯度提升（Gradient Boosting）或DNN进行条件拟合；LLM推理时加入策略提示；蒙特卡洛采样和加权融合。

**📊 数据集**

使用了美团平台2025年10‑11月的真实用户交互日志，共计26,461条高意向轨迹，涵盖101家商家、5类菜品和3流量层级；用户与商家特征通过预训练编码器映射为稠密向量。

**📈 对比分析**

与传统拟合型（XGBoost、DNN、Gradient Boosting）和推理型（GPT‑4、Qwen3、DeepSeek）基线对比，PGHS在群体模拟误差（GSE）上达8.80%，比最佳LLM基线低45.8%、比最佳ML基线低40.9%；在专家评估的策略排名上，Kendall τ达0.69，分别比ML基线提升38.0%、比LLM基线提升32.7%。

**⚠️ 局限性**

局限性包括：仅在商家诊断场景验证，缺乏对其他推荐或个性化任务的通用性测试；模型对策略的依赖使得需大量行为数据；对时间动态和策略演变的建模尚未深入；融合权重 λ 的选择仍需经验设定。

---

## 478. The Parameterized Complexity of Coloring Mixed Graphs

**arXiv ID:** 2604.15274 | [PDF](https://arxiv.org/pdf/2604.15274v1)

**作者:** Antonio Lauerbach `[一作]` (Julius-Maximilians-Universität Würzburg), Alexander Wolff `[通讯]` (Julius-Maximilians-Universität Würzburg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文系统研究了混合图（同时包含无向边和有向弧）着色问题的参数化复杂度，提出了混合邻域多样性和混合宽度两种新参数，并分析了它们与传统结构参数之间的关系，给出了多条 FPT 与硬度结果，讨论了传递闭包对参数影响，并给出了混合图色数的上界与下界；

**💡 创新点**

创新点在于首次为无环混合图定义混合邻域多样性与混合宽度，揭示了它们与树宽、邻域多样性、宽度等参数的严格关系，证明了在这些参数下的 FPT 与 NP‑硬度边界，提出了通过传递闭包提升参数的技术，以及基于 ILP 的混合图着色算法；

**🔧 技术方法**

采用参数化复杂度的归约、MSO 逻辑与 Courcelle 定理、动态规划、ILP 建模与求解、图结构分解技术以及图的层化方法来得到颜色上界；

**📊 数据集**

本文未使用实际数据集，全部以理论证明与构造实例为主；

**📈 对比分析**

由于研究集中于理论分析，未进行实验比较，性能表现以算法时间复杂度（如 f(k)·n^O(1)）形式给出；

**⚠️ 局限性**

限制在于仅考虑无环混合图，未处理有向环；缺乏实证评估；对混合宽度的 Courcelle 定理适用性仍待验证。

---

## 479. How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations

**arXiv ID:** 2604.15273 | [PDF](https://arxiv.org/pdf/2604.15273v1)

**作者:** Nouhaila Innan `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11284 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在统一的 GNN 后端（GIN）和训练协议下，对比经典与量子导向的节点嵌入方法，并系统评估其在图分类任务中的效果。

**💡 创新点**

提出了一套严格控制的基准框架，仅改变节点嵌入模块，消除架构、数据划分、优化器等干扰，揭示量子灵感嵌入在结构驱动数据集上的优势与局限。

**🔧 技术方法**

使用经典随机投影、MLP、角度量子电路（Angle‑VQC）、量子算子嵌入（QuOp）、量子游走嵌入（QWalkVec）与量子相位编码（QPE）等方法；后者通过图算子、线性代数或量子电路生成节点特征。

**📊 数据集**

TU 5 个社交/生物图数据集（IMDB‑BINARY、IMDB‑MULTI、MUTAG、PROTEINS、ENZYMES）和 QM9 转为二分类的分箱数据集。

**📈 对比分析**

采用相同的 80/10/10 分层划分、相同学习率、批量、早停策略和评估指标（Accuracy、Macro‑F1、Macro‑Precision/Recall）。结果显示：在结构驱动任务（如 MUTAG、QM9）中，QWalkVec* 等量子嵌入显著优于经典基线；在属性稀缺的社交图（IMDB）上，经典 MLP 或随机投影更稳健；量子算子嵌入在某些多类任务上可保持竞争力。

**⚠️ 局限性**

受限于固定训练预算和简单的 GIN 后端，无法充分探索更深层的量子嵌入训练与大规模数据的可扩展性；多种方法的计算成本差异大，未给出统一的资源消耗比较；实验仅在单一随机种子下完成，缺乏多种种子或更大规模实验的稳健性验证。

---

## 480. Stability and Generalization in Looped Transformers

**arXiv ID:** 2604.15259 | [PDF](https://arxiv.org/pdf/2604.15259v1)

**作者:** Asher Labovich `[一作]` `[通讯]`, Asher Labovich

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过引入基于固定点的框架，对循环 Transformer 的稳定性与泛化能力进行系统分析，探讨了可达性、输入依赖性和几何性三个稳定性轴。

**💡 创新点**

创新点在于将三条稳定性轴与架构设计联系起来，证明了记忆（recall）与外部归一化共同能保证可达、输入相关且几何稳健的固定点，并提出了内部记忆（internal recall）这一新型架构。

**🔧 技术方法**

主要技术包括固定点理论、传递性（transversality）与谱半径分析、Jacobian 研究，以及使用 RMSNorm/GRU 等外部归一化方法来调节循环动态。

**📊 数据集**

实验使用棋盘（chess）、数独（sudoku）和前缀求和（prefix‑sum）三种任务的数据集。

**📈 对比分析**

通过对比不同记忆位置与是否使用外部归一化的模型，在训练分布与更难的 OOD 数据上评估，结果显示外部归一化的记忆模型在难度更高的数独上提升了 12% 以上、在棋盘上提升 2% 以上，而未归一化的记忆模型甚至在某些任务上完全失效。

**⚠️ 局限性**

局限性包括实验仅在单层小规模 Transformer 上验证，尚未检验更大模型的适用性；此外，稳定性轴虽是必要条件，但并非充分条件，导致对不同任务的排名差异仍缺乏统一解释。

---

## 481. Orthogonal Strip Partitioning of Polygons: Lattice-Theoretic Algorithms and Lower Bounds

**arXiv ID:** 2604.15247 | [PDF](https://arxiv.org/pdf/2604.15247v1)

**作者:** Jaehoon Chung `[一作]` `[通讯]` (Korea Institute for Advanced Study), Jaehoon Chung (Korea Institute for Advanced Study)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出并实现了正交条带划分（strip partition）问题的最优数值版和报告版算法，分别针对凸多边形、简单多边形以及自重叠多边形进行设计。

**💡 创新点**

创新点包括：①将条带划分问题建模为Clarke–Cormack–Burkowski格（CCB格）的反链结构，并利用格的meet和join运算构造高效动态规划；②针对凸多边形给出输入敏感的O(log n) / O(h log(1+n/h))时间解法；③在简单和自重叠多边形中提供O(n log n)时间算法，并在决策树/代数计算树模型下证明匹配下界。

**🔧 技术方法**

核心技术：垂直梯形分解、双亲-子树动态规划、CCB格的反链完成与meet/join运算、低位编码与恢复、以及在决策树与代数计算树模型上的下界证明。

**📊 数据集**

实验使用的是理论构造的三类多边形输入（凸多边形、简单多边形、以及通过三角剖分和树模型得到的自重叠多边形）。未涉及具体真实数据集。

**📈 对比分析**

通过复杂度分析与下界证明，凸多边形的两种算法已达最优；简单多边形在决策树模型下的下界为Ω(n)，而算法实现为O(n log n)；自重叠多边形在代数计算树模型下的下界为Ω(n log n)，与算法复杂度匹配；总体上展示了不同输入类与模型下的性能边界。

**⚠️ 局限性**

局限性：①简单多边形的算法仍存在n log n 与 n 之间的复杂度差距；②算法依赖垂直梯形分解，难以直接推广到非正交切割或更一般的宽度约束；③自重叠多边形的构造与判定仍需假设可通过三角剖分与可视性模型表示；④报告版的输出依赖预先设定的无损编码方案。

---

## 482. From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning

**arXiv ID:** 2604.15244 | [PDF](https://arxiv.org/pdf/2604.15244v1)

**作者:** Kiran Purohit `[一作]` (IIT Kharagpur), Soumyabrata Pal `[通讯]` (Adobe Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于模型内部验证的自适应推理加速框架，称为Verification-Aware Speculative Decoding（VASD）；

**💡 创新点**

创新点在于使用两种轻量级的内部信号（注意力基归因验证与对数概率验证）构成的集成验证器，并配合自一致性选择器实现逐步验证，避免使用外部奖励模型；

**🔧 技术方法**

主要技术包括注意力回滚（attention rollout）计算归因得分、对数概率置信度评估、集成评分机制、以及句子嵌入的自一致性选择器；

**📊 数据集**

在多步推理基准上进行评估，使用 MATH500、GSM8K、GaoKao-2023-En、OlympiadBench 等数据集；

**📈 对比分析**

与目标模型单独推理、草稿模型多数投票/Best-of-N、标准Speculative Decoding、Reward-guided Speculative Decoding 等基线相比，VASD 在所有基准上准确率提升约 3.6%，推理延迟降低约 11%，且显著优于 RSD；

**⚠️ 局限性**

局限性包括：仅在结构化推理任务上验证，未评估开放式生成或长文本；仅考虑单实例推理，未探讨批处理或硬件特定优化；并未完全消除模型生成的幻觉风险。

---

## 483. Context Over Content: Exposing Evaluation Faking in Automated Judges

**arXiv ID:** 2604.15224 | [PDF](https://arxiv.org/pdf/2604.15224v1)

**作者:** Manan Gupta `[一作]` (BITS Pilani), Dhruv Kumar `[通讯]` (BITS Pilani)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了LLM评判模型在评估者命运信息影响下的偏差，并设计了一个控制实验框架。

**💡 创新点**

首次提出stakes‑signaling框架，揭示评判模型对结果信号的无意识宽松偏差及其不可检测性。

**🔧 技术方法**

采用三种安全/质量基准的1500+回应，四种系统级条件，三款评判LLM，并使用Verdict Shift和ERR_J指标进行分析。

**📊 数据集**

HarmBench、WildGuard、MT‑Bench、WildGuardMix与BeaverTails等数据集。

**📈 对比分析**

通过比较同一回应在不同条件下的二进制安全判定，计算ΔV，发现平均leniency约为-3pp，DeepSeek‑R1峰值-9.8pp；ERR_J为0，表明偏差隐蔽。

**⚠️ 局限性**

实验仅基于公开权重模型、单语数据，未覆盖大模型；缺乏对软性假冒和置信度变化的评估。

---

## 484. Learning to Think Like a Cartoon Captionist: Incongruity-Resolution Supervision for Multimodal Humor Understanding

**arXiv ID:** 2604.15210 | [PDF](https://arxiv.org/pdf/2604.15210v1)

**作者:** Hatice Merve Vural `[一作]` (Koc University), Aykut Erdem `[通讯]` (Koc University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Incongruity‑Resolution Supervision (IRS)，通过将幽默理解拆解为不协调识别、重构与喜好对齐三步，显式监督模型的中间推理过程。

**💡 创新点**

创新点在于将认知心理学的不协调‑解决理论与专业漫画字幕创作者的推理轨迹相结合，形成三阶段的可学习监督框架，并通过视觉感知与风格奖励实现零样本泛化。

**🔧 技术方法**

采用领域自适应预训练、captionist reasoning traces 结构化监督、以及基于视觉感知与语言风格的强化学习奖励（GRPO）。

**📊 数据集**

主要使用 New Yorker Cartoon Caption Contest (NYCC) 数据集，并在其匹配、排名、10‑vs‑1000、30‑vs‑300 四种评测任务上进行实验。

**📈 对比分析**

与文本仅推理、闭源与开源多模态基线相比，IRS 在 7B/32B/72B 模型上显著提升准确率，72B 模型在排名任务达 76.10% 接近专家水平，并在 YesBut、DeepEval 等外部幽默基准上实现零样本迁移。

**⚠️ 局限性**

局限性包括：模型过度依赖 NYCC 的幽默风格与文化背景，可能难以推广至其他幽默场景；需要人工制作的专家推理轨迹和多模态奖励，增加数据与计算成本。

---

## 485. MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events

**arXiv ID:** 2604.15203 | [PDF](https://arxiv.org/pdf/2604.15203v1)

**作者:** Raunak Agarwal `[一作]` (Fraunhofer Heinrich Hertz Institute), Jackie Ma `[通讯]` (Fraunhofer Heinrich Hertz Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个持续更新、无污染的多标签文本分类基准（来自FDA医疗器械不良事件报告），并系统评估了判别式、生成式微调与Prompting等多种学习范式及其不确定性量化方法。

**💡 创新点**

提出了长期可更新的“living benchmark”，设计了层级标签与严重长尾分布，并在此基准上首次对多种模型架构、学习方式及UQ策略进行统一、细粒度对比。

**🔧 技术方法**

采用Transformer编码器/解码器（Llama、Ettin等）、判别式与生成式微调、LoRA、kNN检索+提示、信息/一致性/自述不确定性度量，并使用PRR、Spearman、ECE+等指标评估UQ。

**📊 数据集**

使用FDA医疗器械不良事件报告（2015‑2025），包含1154个层级标签、平均每样本约9个标签、平均句长约370词的长尾分布数据集。

**📈 对比分析**

通过宏F1、Jaccard、PRR、Spearman相关、ECE+等指标比较；判别式微调在常见类表现最佳，生成式微调在罕见类和不确定性估计上更优；Prompting在极稀有类提升但整体波动大；最佳模型宏F1仅54%，显示任务仍具挑战。

**⚠️ 局限性**

数据标签可能存在不一致性，模型难以对负类估计不确定性，思考模型UQ表现差，单一英文基准难以推广至多语言或其他领域，且多数模型自信度低，需谨慎部署。

---

## 486. GlobalSplat: Efficient Feed-Forward 3D Gaussian Splatting via Global Scene Tokens

**arXiv ID:** 2604.15284 | [PDF](https://arxiv.org/pdf/2604.15284v1)

**作者:** Roni Itkin `[一作]` (Hebrew University of Jerusalem), Sagie Benaim `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 697 | [OpenAlex ID](https://openalex.org/A5081028371)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种前馈式3D高斯投影框架GlobalSplat，先通过全局场景标记聚合多视图信息，再解码为稀疏高斯集合，实现高效新视角渲染。

**💡 创新点**

创新点在于“先对齐、后解码”策略：使用固定数量的全局隐式标记，消除视图对齐导致的冗余；双分支注意力网络解耦几何与外观；以及粗到细的容量训练课程，保证在压缩预算下仍能获得高质量重建。

**🔧 技术方法**

核心技术包括3D高斯投影（3D Gaussian Splatting）、多视图特征拼接、Plücker射线+相机元数据注入、双分支注意力编码器、全局隐式标记解码器以及自监督一致性损失。

**📊 数据集**

主要使用RealEstate10K和ACID两个数据集进行训练与评估；RealEstate10K为室内外房地产视频，ACID为无人机航拍沿海景观。

**📈 对比分析**

与现有前馈式NVS基线（如NoPoSplat、AnySplat、Zpressor、C3G等）比较，GlobalSplat在相同或更少的高斯数（如16K）下取得相近或更优的PSNR/SSIM/LPIPS，同时显著降低显存占用（1.79GB）和推理时间（<78 ms），磁盘占用<4MB。

**⚠️ 局限性**

局限包括：固定的高斯预算（16K）可能不足以覆盖大规模或城市级场景；仅适用于静态环境，无法处理时间变化；稀疏视图（2-3张）下重建质量下降。

---

## 487. Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees

**arXiv ID:** 2604.15221 | [PDF](https://arxiv.org/pdf/2604.15221v1)

**作者:** Jakob Thumm `[一作]` (Stanford University), Marco Pavone `[通讯]` (Stanford University)

**通讯引用:** 11639 | [OpenAlex ID](https://openalex.org/A5050003000)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一套基于双摄像机视觉的人体姿态估计与运动预测框架，并将不确定性传播、异常检测以及合成预测集融入其中，最终实现对人机协作过程的可证实安全保证。

**💡 创新点**

核心创新点包括：①端到端的不确定性传播与协方差估计；②对运动预测构造1‑ϵ置信度的合成预测集；③通过梯度型OOD检测与历史预测重用，实现对异常输入的平滑处理，避免系统停机。

**🔧 技术方法**

技术手段：改进YOLOv6输出2D姿态及协方差 → 线性三角化得到3D姿态及协方差；DCT+Transformer+多频段分离+Cholesky分解预测未来3D姿态与协方差；多阶段训练与损失权重调度；SLU（Sketching Lanczos Uncertainty）进行OOD检测；最终结果输入SARA盾牌实现安全控制。

**📊 数据集**

使用Human3.6M数据集进行训练与评估，并在Frank Emika机器人上部署Intel RealSense 435i实现真实环境下的验证；实验中对不同视角、不同动作序列均有覆盖。

**📈 对比分析**

与ST‑Trans、SiMLPe、HisRep等基线及ISO 13855:2010常数速度模型比较：MPJPE在使用真实3D姿态输入时略优于state‑of‑the‑art，合成预测集覆盖率98.25%高于ISO 97.93%，体积平均缩小11倍；OOD处理使无效预测率下降36%，MPJPE仅略升2.6。

**⚠️ 局限性**

局限性：目前未考虑人体进入/离开工作空间的场景；未融合RGB‑D输入的姿态估计；在更快运动或与训练分布差异较大的新环境下的泛化性仍待进一步验证。

---

## 488. Bandwidth Cost of Locally Repairable Convertible Codes in the Global Merge Regime

**arXiv ID:** 2604.15282 | [PDF](https://arxiv.org/pdf/2604.15282v1)

**作者:** Saransh Chopra `[一作]` (Carnegie Mellon University), K. V. Rashmi `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在分布式存储系统中，将一个系统可恢复码（LRC）转换为另一个参数相同的系统可恢复码（全局合并模式）时的带宽成本，并给出下界；

**💡 创新点**

提出一种信息理论方法来推导非平凡的带宽下界，且不依赖线性假设；证明在参数 r≥k时的特定范围内已有构造已达到该下界，证明其带宽最优；

**🔧 技术方法**

信息理论不等式（条件熵、互信息、数据处理不等式）与 LRC 的结构性质相结合；

**📊 数据集**

无具体实验数据，论文完全基于理论分析与符号计算；

**📈 对比分析**

通过与已知构造（Maturana‑Rashmi 的可转换 LRC 构造）的带宽成本进行比较，证明在 r≤k 时两者匹配，表明构造是最优的；

**⚠️ 局限性**

下界仅在 r≤k 时被证明为最优；对 r>k 的情况尚未收敛；仅考虑稳定可转换码和离散化局部组，未探讨非稳定、重叠局部组或更高可靠性/可用性约束的情况。

---

## 489. A Manual Bar-by-Bar Tempo Measurement Protocol for Polyphonic Chamber Music Recordings: Design, Validation, and Application to Beethoven's Piano and Cello Sonatas

**arXiv ID:** 2604.15278 | [PDF](https://arxiv.org/pdf/2604.15278v1)

**作者:** Ignasi Sole `[一作]` `[通讯]`, Ignasi Sole

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了一套基于累计计时器的手工条幅速度提取协议，手动记录并计算贝多芬钢琴与大提琴协奏曲在历史录音中的每一小节节拍数（BPM），并生成条幅级的节奏图、直方图、核密度图等可视化；

**💡 创新点**

创新点在于：①累计时间架构防止误差累积并实现自校验；②把手工计时与工程精度方法相结合，提供可量化的误差模型；③构建可扩展的电子表格数据结构，方便多录音对比；

**🔧 技术方法**

使用的技术包括：数字计时器（带秒表/计时器功能）、手工标记、电子表格（Google Sheets/Excel）进行时间差计算、BPM公式推导、Python+Seaborn+Matplotlib绘制梯形图、MATLAB绘制平滑PDF直方图；

**📊 数据集**

数据集为100多条1930–2012年间贝多芬五首钢琴与大提琴奏鸣曲各乐章的历史录音，涵盖不同演奏家与年份；

**📈 对比分析**

与传统自动节拍跟踪工具（如MUsanim）对比，自动工具在这些多声部历史录音中普遍失效或误差过大；手工协议在所有录音中均能得到连贯的BPM曲线，误差主要来自计时者反应时间，随机且不累计，远低于演奏者间的节奏差异；

**⚠️ 局限性**

主要局限在于：需要人工计时，耗时约30–45分钟/乐章，规模化研究难度较大；仅由单一计时者完成，缺乏跨评估者可靠性检验；适用于无法自动化的历史多声部录音，非适合高吞吐量的大规模语料库。

---

## 490. Simplifying Safety Proofs with Forward-Backward Reasoning and Prophecy

**arXiv ID:** 2604.15266 | [PDF](https://arxiv.org/pdf/2604.15266v1)

**作者:** Eden Frenkel `[一作]` (Tel Aviv University), Sharon Shoham `[通讯]` (Tel Aviv University)

**通讯引用:** 6606 | [OpenAlex ID](https://openalex.org/A5028417315)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种增量式安全证明方法，结合前向、后向推理和预言变量，以简化安全性证明中的归纳不变式结构；

**💡 创新点**

创新点在于通过前向-后向推理与预言变量的协同，既降低不变式的布尔复杂度，又能消除量化层级和交替；

**🔧 技术方法**

使用一阶逻辑模型、增量式证明系统、时间逆推、预言变量（prophecy）以及 SMT 求解器（Z3）进行归纳验证；

**📊 数据集**

在案例研究中应用于 Paxos（多种变体）和 Raft 协议，使用 EPR 子句集作为数据集；

**📈 对比分析**

与传统前向单一不变式方法相比，实验显示前向-后向+预言的证明在涉及量化或布尔复杂度较高的实例中，验证时间明显下降，且证明所需的谓词更简洁；

**⚠️ 局限性**

局限性包括对预言变量的手工选择和证明搜索仍需人工干预，且在某些协议中量化交替难以完全消除，导致证明规模仍可能膨胀。

---

## 491. A Nonlinear Separation Principle: Applications to Neural Networks, Control and Learning

**arXiv ID:** 2604.15238 | [PDF](https://arxiv.org/pdf/2604.15238v1)

**作者:** Anand Gokhale `[一作]` (University of California Santa Barbara), Francesco Bullo `[通讯]` (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于收敛理论的非线性分离原则，并给出了连续时间与离散时间脉冲速率与霍普菲尔德神经网络的最优收敛判据，随后利用这些判据构建了稳健的控制器与观测器，并在隐式深度学习（DEQ）中实现了无约束可参数化的收敛网络，验证其在MNIST和CIFAR-10分类任务上的竞争性能。

**💡 创新点**

创新点包括：①在收敛框架下首次给出非线性分离原则的通用形式；②推导出针对FRNN与HNN的最优（对称权重）收敛LMI条件；③提供了图形RNN的收敛性可扩展性；④结合控制理论设计了低增益积分跟踪控制器；⑤给出了一种无约束参数化方法，可在隐式网络中构造保持收敛的权重矩阵，并实现输入依赖的高表达度模型。

**🔧 技术方法**

技术手段主要有：收敛理论与增益放缩、S-乘子与S-lemma、LMI可行性分析、投影引理、静态输出反馈可行性条件、低增益积分控制的奇异扰动分析、以及深度学习中DEQ的固定点收敛性证明。

**📊 数据集**

在机器学习实验中使用了标准图像分类数据集MNIST与CIFAR-10，并将模型与公开的LBEN、monDEQ等基准模型进行对比。

**📈 对比分析**

方法通过对比实验表明，所构造的隐式网络在MNIST上达到99.33%（约比monDEQ高0.23%）且模型规模仅89K参数；在CIFAR-10上在无数据增强条件下得到78.27%（相对monDEQ低3.73%），但在使用数据增强后可达到82.30%，接近monDEQ*。整体来看，模型在保持参数效率的同时，取得了与竞争方法相当或更优的分类精度。

**⚠️ 局限性**

主要局限包括：①收敛判据对权重矩阵的结构要求仍较强，尤其在非对称或不满足LDS的情形下可能不适用；②图形RNN的收敛性扩展仍局限于无向图；③在控制应用中对非线性系统的鲁棒性分析仍未考虑随机扰动与噪声；④对更复杂的序列模型（如LSTM、Transformer）尚未推广收敛框架。

---

## 492. StreamCacheVGGT: Streaming Visual Geometry Transformers with Robust Scoring and Hybrid Cache Compression

**arXiv ID:** 2604.15237 | [PDF](https://arxiv.org/pdf/2604.15237v1)

**作者:** Xuanyi Liu `[一作]` (Peking University), Lanyun Zhu `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种基于Transformer的实时3D重建框架StreamCacheVGGT，利用双模块CLCES与HCC实现缓存效率与信息保留的平衡。

**💡 创新点**

创新点在于：①用跨层一致性评分消除单层激活噪声；②采用三层缓存三分法，将低重要性token软合并而非硬剔除，提升全局几何连贯性。

**🔧 技术方法**

核心技术包括FlashAttention、全局KV缓存、跨层秩统计评分、重要性加权最近邻合并、动态阈值与动态锚点保护（DAP）。

**📊 数据集**

在室内外多种基准上验证：7-Scenes、NRGBD、ETH3D、Bonn与KITTI。

**📈 对比分析**

与现有常驻内存与无限内存模型对比，StreamCacheVGGT在Accuracy、Completeness、Normal Consistency和Abs Rel等指标均取得或接近最优，且保持O(1)内存与单次推理时间。

**⚠️ 局限性**

局限性：单向因果推理无法纠正累计误差，未提供全局后处理或回溯纠错机制。

---

## 493. Benchmarking Classical Coverage Path Planning Heuristics on Irregular Hexagonal Grids for Maritime Coverage Scenarios

**arXiv ID:** 2604.15202 | [PDF](https://arxiv.org/pdf/2604.15202v1)

**作者:** Carlos S. Sepúlveda `[一作]` (Universidad Adolfo Ibáñez), Gonzalo A. Ruz `[通讯]` (Universidad Adolfo Ibáñez)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

做了什么：构建了一个可复现的基准，用于在不规则六边形网格上评估单机覆盖路径规划的经典启发式算法。

**💡 创新点**

创新点是什么：提供了10,000个已验证Hamiltonian可行的实例，统一实现和评估17种经典启发式，并揭示了残差度策略和端点处理对Warnsdorff法性能的决定性影响。

**🔧 技术方法**

用了什么技术：使用六边形格子离散化、图论覆盖规划以及Warnsdorff、DFS回溯、线性扫荡、树覆盖、波前、马氏曲线等经典启发式，并用深度优先搜索对实例进行Hamiltonian可行性审计。

**📊 数据集**

用了什么数据集：生成了三类形态（紧凑、拉长、不规则）的合成海域多边形，按比例构造28–46个六边形节点的稀疏图，形成10,000个实例。

**📈 对比分析**

如何比较的方法，性能怎么样：在统一协议下对全部实例计算Hamiltonian成功率、完整覆盖率、重复访问、路径长度、转向角度和CPU延迟；结果显示Warnsdorff‑TI(索引)以79%最高Hamiltonian成功率，其他方法在允许重复访问时表现更好。

**⚠️ 局限性**

limitation是什么：单机静态合成实例，未考虑流体、运动学约束、感知不确定性及在线重规划；图规模限制在28–46节点，可能不直接适用于更大或真实海域。

---

## 494. Unsupervised Skeleton-Based Action Segmentation via Hierarchical Spatiotemporal Vector Quantization

**arXiv ID:** 2604.15196 | [PDF](https://arxiv.org/pdf/2604.15196v1)

**作者:** Umer Ahmed `[一作]` (Retrocausal, Inc.), Quoc-Huy Tran `[通讯]` (Retrocausal, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种完全无监督的骨架动作时序分割框架 HiST-VQ，利用层级空间–时间向量量化实现对长序列的动作分割。

**💡 创新点**

创新点在于：① 采用两级层级向量量化（先聚合子动作再聚合动作），② 在向量量化过程中同时进行空间（骨架重构）和时间（时间戳重构）自监督学习，③ 通过层级聚类显著减少了段长度偏差。

**🔧 技术方法**

核心技术包括：多阶段 TCN 编码器、Patch‑based 层级向量量化、空间/时间解码器、EMA 更新代码本、组合的 commitment、空间重构损失（MSE）与时间重构损失（MSE）以及整体加权损失。

**📊 数据集**

在公开骨架数据集 HuGaDB、LARa 与 BABEL（分别为 10 小时、13 小时、43 小时）上进行实验。

**📈 对比分析**

与 SMQ、CTE、TOT、ASOT、HVQ 等无监督方法（以及其 Viterbi 加强版）比较，HiST-VQ 在 MoF、Edit、F1（10%/25%/50%）和 Jensen–Shannon 距离（JSD）上均取得了新的最优结果，特别是显著降低了段长度偏差。

**⚠️ 局限性**

局限性：① 对超参数（如 λ_spat、λ_temp、α 等）敏感，需要经验调优；② 仅使用骨架信息，缺少视像或语义上下文；③ 层级深度过大会导致噪声累积，最佳表现仅在两级层级；④ 与完全监督方法相比仍存在性能差距。

---

## 495. Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines

**arXiv ID:** 2604.15186 | [PDF](https://arxiv.org/pdf/2604.15186v1)

**作者:** Marcel Wagenländer `[一作]`, Peter Pietzuch `[通讯]` (Imperial College London)

**通讯引用:** 6639 | [OpenAlex ID](https://openalex.org/A5078842469)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套名为 Scepsy 的系统，用于在 GPU 集群上高效调度和服务多 LLM 的 agentic 工作流，并通过聚合每个 LLM 的稳定使用比例构造 Aggregate LLM Pipeline 进行性能预测。

**💡 创新点**

创新点包括：
① 把 agentic 工作流抽象为 Aggregate LLM Pipeline，利用每个 LLM 的相对占比来预测吞吐率与延迟；
② 在 GPU 调度中同时考虑分数 GPU 分配、张量并行度与模型复制数，实现吞吐与延迟的联合最优；
③ 采用拓扑感知的分数 GPU 放置算法，并通过 NVIDIA MPS 实现细粒度资源隔离，从而最大化 GPU 利用率。

**🔧 技术方法**

主要技术：
- 低层代理（HTTP 代理）跟踪 LLM 调用并收集统计；
- 对每个 LLM 进行不同并行度下的性能剖析，生成吞吐‑延迟曲线；
- 基于上述统计构建 Aggregate LLM Pipeline；
- GPU 调度器使用搜索+剪枝来确定分数 GPU 分配、张量并行度和复制数；
- 拓扑感知的分数 GPU 放置算法结合 Kubernetes、NVIDIA MPS、vLLM 与 SGLang。

**📊 数据集**

使用的工作流数据：
- Retrieval‑Augmented Generation + Reranker（RAG+reranker）
- Beam Search（使用 LLaMA‑3.2‑1B 生成器 + LLaMA‑3.2‑8B 验证器）
- 组合工作流（RAG+reranker 与 Beam Search 同时运行）。
这些工作流在真实请求序列上被跟踪并用于评估，未使用公开的专门数据集。

**📈 对比分析**

对比方法：
- Kubernetes Autoscaler（基于 MicroK8s 的自适应扩容）
- Aegaeon（多 LLM 共享 GPU 的多模型调度）
- Ayo（面向 agentic 工作流的调度系统）。
在 4、8、16 GPU 集群上对吞吐‑延迟曲线进行评估。Scepsy 在 4 GPU 时吞吐提升 2.4×、延迟提升 1.4–7.6×；在 8 GPU 时吞吐提升 1.5–1.8×、延迟提升 1.3–5.2×；在 16 GPU 时吞吐提升 1.8–2.4×、延迟提升 1.7–10.4×；RAG 工作流的延迟提升 1.9–27×。

**⚠️ 局限性**

限制：
- 对跨不同 LLM 的 fan‑out 仍采用串行模型，可能低估瓶颈；
- 工具或外部操作的耗时被忽略，若占比大则误差增大；
- 假设每个 LLM 的负载分布为单峰，无法充分处理多峰或极端模式；
- 搜索空间仍随 LLM 数量、GPU 数量指数增长，虽通过剪枝降低但在极大规模集群上仍耗时；
- 需要为每个 LLM 进行独立性能剖析，部署与维护成本较高。

---

## 496. RadAgent: A tool-using AI agent for stepwise interpretation of chest computed tomography

**arXiv ID:** 2604.15231 | [PDF](https://arxiv.org/pdf/2604.15231v1)

**作者:** Mélanie Roschewitz `[一作]` (ETH Zurich), Michael Moor `[通讯]` (ETH Zurich)

**通讯引用:** 1886 | [OpenAlex ID](https://openalex.org/A5021730842)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

构建了一个强化学习驱动的辐射学代理 RadAgent，能够在3D CT扫描上通过工具调用和诊断检查表逐步生成可追溯的报告。

**💡 创新点**

首次将RL训练与多工具交互结合到医学报告生成，解决了传统3D VLM的黑箱问题，提升了报告的准确性、稳健性和忠实度。

**🔧 技术方法**

使用 Qwen3-14B 作为主体 LLM，GRPO 强化学习，MCP 工具协议，结合 CT-Chat、CT-CLIP、TotalSegmentator 等十款专用工具。

**📊 数据集**

在 CT-RATE 内部数据集（训练/验证/测试）以及 RadChestCT 外部数据集上进行评估。

**📈 对比分析**

相较于 CT-Chat 基线，宏 F1 提升 6.0 点（36.4%），微 F1 提升 5.4 点（19.6%），鲁棒性提升 24.7 点，忠实度提升 37.0%，在内部、外部集上均显著优于基线。

**⚠️ 局限性**

需要多 GPU 高算力，工具集变化会导致策略退化，忠实度仍有提升空间，训练数据对模型泛化存在局限。

---

## 497. Agent-Aided Design for Dynamic CAD Models

**arXiv ID:** 2604.15184 | [PDF](https://arxiv.org/pdf/2604.15184v1)

**作者:** Mitch Adler `[一作]` (Independent Researcher), Michael Cafarella `[通讯]` (MIT)

**通讯引用:** 7863 | [OpenAlex ID](https://openalex.org/A5039133265)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种名为AADvark的代理驱动系统，能够根据输入图像或文本生成包含运动部件的动态CAD装配（如剪刀）以及静态装配物体；

**💡 创新点**

创新点在于通过JSON中介表示实现动态关节描述，结合外部约束求解器和改进的可视化反馈（给每个部件的面、边赋予唯一颜色/纹理）来弥补LLM的空间推理不足，并利用强验证器提升代理的设计准确性；

**🔧 技术方法**

采用的技术包括大型语言模型（Gemini 3 Flash）生成代码、修改后的FreeCAD渲染、改进的OndselSolver约束求解器（使用四元数并提供详细错误信息）、以及基于视觉与编译错误的迭代反馈循环；

**📊 数据集**

实验数据主要来自少量手工选取的输入图像（剪刀、儿童床、椅子等）以及可选的LLM生成的设计说明，未使用大型公共CAD数据集；

**📈 对比分析**

与现有仅能生成静态CAD模型的系统相比，AADvark能够成功生成带有旋转关节的功能性剪刀，并在4至34次迭代内完成多种静态装配，整体耗时约4-5小时，展示了在无监督、无训练数据条件下的可行性；

**⚠️ 局限性**

局限性包括只能使用矩形棱柱部件、仅支持旋转关节、在复杂装配时可能陷入死循环且需要重启、以及对非确定性代理执行的鲁棒性仍需改进。

---

## 498. OmniLight: One Model to Rule All Lighting Conditions

**arXiv ID:** 2604.15170 | [PDF](https://arxiv.org/pdf/2604.15170v1)

**作者:** Youngjin Oh `[一作]` (Seoul National University), Nam Ik Cho `[通讯]` (Seoul National University)

**通讯引用:** 5046 | [OpenAlex ID](https://openalex.org/A5055171648)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一个统一的光照相关图像恢复框架 OmniLight，并与专门模型 DINOLight 进行对比实验，旨在同时处理阴影去除、白平衡 ALN 与多色彩 ALN；

**💡 创新点**

在传统视觉先验与 Wavelet Domain Mixture‑of‑Experts（WD‑MoE）相结合的动态路由结构上做出创新，显著降低多任务学习中的负迁移并提升泛化能力；

**🔧 技术方法**

采用 U‑Net 主干、SFDINO 语义编码器、DINOv2 视觉先验、离散小波变换 (DWT/IDWT)、Mixture‑of‑Experts、跨尺度结构相似损失以及专家负载平衡等技术；

**📊 数据集**

使用公开数据集 WSRD+、Ambient6K、CL3AN 以及 NTIRE 2026 挑战赛的训练/测试集进行训练与评估；

**📈 对比分析**

通过与多种基准方法（如 DINOLight、MPRNet、OmniSR 等）在 Ambient6K、CL3AN、WSRD+ 与 NTIRE 2026 三个任务上进行定量比较，OmniLight 在 PSNR/SSIM/LPIPS 上均取得或超过现有 SOTA，并在 NTIRE 2026 取得多项第一/第二名；

**⚠️ 局限性**

在阴影去除任务中偶尔出现局部颜色偏移，说明任务间的极端光照特征难以完全分离；同时，专家负载不均衡和路由策略仍有进一步优化空间；

---

## 499. Wave-Based Dispatch for Circuit Cutting in Hybrid HPC--Quantum Systems

**arXiv ID:** 2604.15279 | [PDF](https://arxiv.org/pdf/2604.15279v1)

**作者:** Ricard S. García-Raigada `[一作]`, Sergio Iserte `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5006767413)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DQR（Dynamic Queue Router）运行时框架，解耦量子电路切分与执行编排，使切分后的子电路可作为独立调度单元在混合HPC–QC系统中执行；

**💡 创新点**

创新点包括：无后端依赖的子电路描述符；波式非阻塞调度器实现流水线并行；细粒度失败隔离与透明故障转移；以及在CESGA Qmio与IBM云上实现的生产级多后端集成；

**🔧 技术方法**

使用技术包括Qdislib门切分、MPI波式调度、gRPC微服务、Qulacs模拟器、Qmio本地QPU、IBM Quantum云、Slurm调度与QCore后端适配器；

**📊 数据集**

实验基准采用32量子位硬件高效Ansatz（L=1和L=2）电路，通过Qdislib产生的72个子电路（L=1）或2592个子电路（L=2）进行评估；

**📈 对比分析**

与单一CPU基准对比，DQR在本地QPU和IBM云上分别实现了最高1.11×的总时延加速，调度开销低于5%，并通过不同调度策略（A–D）展示了QC占用与HPC占用的权衡及故障恢复效果；

**⚠️ 局限性**

局限性包括：Qmio本地编译器不支持门切分产生的中间控制流导致子电路失败转移；资源分配（CPU、GPU、QPU插槽）在作业提交时固定，缺乏动态适配；云端QPU的长轮询延迟显著影响性能；以及需手动设定标签策略以匹配硬件。

---

## 500. R3D: Revisiting 3D Policy Learning

**arXiv ID:** 2604.15281 | [PDF](https://arxiv.org/pdf/2604.15281v1)

**作者:** Zhengdong Hong `[一作]` (Zhejiang University), Jiayuan Gu `[通讯]` (ShanghaiTech University)

**通讯引用:** 2572 | [OpenAlex ID](https://openalex.org/A5101866448)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种可扩展的3D策略学习框架R3D，结合Transformer 3D编码器与扩散解码器，并通过去除BN、加入3D数据增强、预训练编码器等技术实现训练稳定性与性能提升。

**💡 创新点**

创新点包括：①系统诊断3D策略学习中的不稳定与过拟合根源并给出解决方案；②设计仅使用LayerNorm的高容量3D编码器，保持空间分辨率；③采用跨注意力的密集特征条件扩散Transformer；④将编码器预训练于大型3D分割数据集；⑤加入辅助末端执行器姿态预测任务提升本体感知。

**🔧 技术方法**

主要技术：Transformer‑based point‑cloud encoder（FPS+KNN+ViT）、LayerNorm、Diffusion Transformer（DiT）解码器、3D数据增强（FPS随机化、颜色抖动、噪声/丢点）、交叉注意力密集特征条件、辅助EE解码器。

**📊 数据集**

训练与评估数据集：RoboTwin 2.0（Easy/Hard）与 ManiSkill2（PickCube、StackCube、PegInsertion系列）；真实世界实验使用xArm6 + 两台RealSense D435；预训练编码器使用 ScanNet、ARKitScenes、PartNeXt 等大规模3D分割数据集。

**📈 对比分析**

与DP3、DP、ManiFlow、ACT、Pi0、Spatial Forcing等2D/2.5D/3D基线进行对比。R3D在RoboTwin Easy平均成功率83.8%、Hard平均64.8%；ManiSkill2平均55.2%；真实任务平均68.7%，在所有测试中均显著优于对照方法。

**⚠️ 局限性**

局限性：对点云分辨率和传感器噪声敏感；多视角融合仍需较高计算开销；对动态光照或复杂环境的鲁棒性尚待进一步提升；缺乏对长时间推理或离线迁移的系统性评估；与大型多模态VLA模型相比，跨模态泛化仍有限。

---

## 501. Why Do Vision Language Models Struggle To Recognize Human Emotions?

**arXiv ID:** 2604.15280 | [PDF](https://arxiv.org/pdf/2604.15280v1)

**作者:** Madhav Agarwal `[一作]` (University of Edinburgh), Steven McDonagh `[通讯]` (University of Edinburgh)

**通讯引用:** 5850 | [OpenAlex ID](https://openalex.org/A5052824649)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

诊断并缓解视觉语言模型在视频情感识别中的长尾偏差与时序表示瓶颈，提出多阶段上下文增强（MSCE）方案；

**💡 创新点**

首次将长尾数据与时序稀疏化作为导致 VLM 情感识别失败的根本原因，并利用中间帧生成文本摘要进行时序补偿；

**🔧 技术方法**

采用 VLM 微调+LoRA、类平衡采样、稀疏时间采样、文本生成摘要与多阶段融合技术；

**📊 数据集**

使用 MAFW、DFEW 视频情感数据集，并以 Google Books Ngram 作为情感词频的代理；

**📈 对比分析**

与闭源 Gemini2.5‑Flash、开源 Qwen 系列、EmotionQwen 及视觉专用 MAE‑DFER、HiCMAE 在零样本/微调条件下对比，MSCE 在 F1 上提升 2–5% 左右；

**⚠️ 局限性**

仍受 VLM 对长序列注意力衰减、文本生成噪声、Ngram 仅为相关性代理等限制影响。

---

## 502. Prism: Symbolic Superoptimization of Tensor Programs

**arXiv ID:** 2604.15272 | [PDF](https://arxiv.org/pdf/2604.15272v1)

**作者:** Mengdi Wu `[一作]` (Carnegie Mellon University), Zhihao Jia `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2266 | [OpenAlex ID](https://openalex.org/A5039360973)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了第一种符号化张量程序超优化器，使用符号层次图表示法在搜索空间上做结构化剪枝，结合e-graph等价验证与自动调优；

**💡 创新点**

创新点在于将映射与并行化参数抽象为符号变量，形成sGraph，实现搜索与枚举分离，保证最优解且显著降低搜索复杂度；

**🔧 技术方法**

采用符号维度匹配、表达式引导剪枝、e-graph等价重写、随机采样加GPU性能评估等技术；

**📊 数据集**

评测使用五种常见LLM工作负载：RMSNorm、RMSNorm-MLP、SwiGLU、Attention、QK‑Attention；

**📈 对比分析**

与Mirage、PyTorch eager/compiled、TVM ansor等基线对比，最快工作负载可达4.9×速度提升，优化时间比Mirage提升约3.4×；

**⚠️ 局限性**

局限性包括：对多循环维度支持有限、e-graph规则不完整、对极小或结构简单程序时实例化开销仍占主导、未给出完整的完备性证明。

---

## 503. Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation

**arXiv ID:** 2604.15270 | [PDF](https://arxiv.org/pdf/2604.15270v1)

**作者:** Zoe Fingleton `[一作]`, Armin Moin `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了 GPT-3.5‑turbo 在使用 RAG 与不使用 RAG 时生成代码的错误类型及数量。

**💡 创新点**

创新点在于细粒度地统计并比较各类错误（如缺少冒号、括号、引号等）的出现频率，从而评估 RAG 对代码质量的具体影响。

**🔧 技术方法**

采用了 GPT‑3.5‑turbo 语言模型、RAG（检索增强生成）技术，并对生成代码进行错误分类。

**📊 数据集**

使用的主要数据集是模型生成的代码样本（包含 Bug‑free 代码与各类语法错误），未指明具体公开数据集。

**📈 对比分析**

通过对两种设置下的错误计数进行对比，观察到 RAG 能显著减少多种错误类型的出现，整体错误率下降。性能评估以错误类型计数为主。

**⚠️ 局限性**

局限性包括：仅关注错误计数，未评估代码功能性或可执行性；样本规模可能有限；缺少对比基准或其他模型的多样性评估。

---

## 504. Expanding into Reality: Random Graphs for Datacenter Networks

**arXiv ID:** 2604.15261 | [PDF](https://arxiv.org/pdf/2604.15261v1)

**作者:** Giacomo Bernardi `[一作]` (Amazon Web Services), Elizabeth Tennent `[通讯]` (Amazon Web Services)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并部署了基于随机图（即扩散器）的大规模数据中心网络结构，并实现了分布式的Spraypoint路由与ShuffleBox光学连线方案，构建了可在现有商业交换机上实现的完整生产级数据中心纤维布线与路由架构；

**💡 创新点**

主要创新点包括①一种能够在随机图上高效生成大量无交叉路径的分布式Spraypoint路由算法，解决了传统k‑shortest路径无法规模化的问题；②利用ShuffleBox/ShuffleBack等被动光学设备实现端口“洗牌”，大幅简化物理布线、降低成本并支持可扩展部署；③给出了可预测随机图网络性能（路径长度、edge‑disjoint路径数、oversub等）的模型，支持基于成本/性能目标的拓扑设计；

**🔧 技术方法**

技术实现涉及：随机图/expander拓扑、Spraypoint基于喷射（source fan‑out）与waypoint层（destination fan‑in）的路由；使用VRF+ECMP实现无循环、分布式路由；ShuffleBox/ShuffleBack光学随机连接实现物理层端口混洗；基于随机匹配与线性规划评估网络吞吐与oversub；模拟与实测结合验证。

**📊 数据集**

实验使用了多租户生产数据中心真实流量（Web、存储、IO等）以及三类抽象流量模式（Clique、Hubs、Matchings），并通过随机匹配生成多种流量矩阵评估oversub；此外对随机图网络的路径长度与edge‑disjoint路径数做了统计。

**📈 对比分析**

与传统fat tree网络在相同服务器数与oversub目标下进行对比，评估指标包括吞吐量、平均/最大路径长度、edge‑disjoint路径数量、oversub比率与交换机/光纤成本。实验结果显示，随机图网络在相同或更低成本下（最高可降至45%）实现与fat tree相当甚至更优的吞吐量，路径长度略短，edge‑disjoint路径数显著高于k‑shortest路径方案；多路复用传输协议的性能与fat tree无明显差异。

**⚠️ 局限性**

局限性主要在于：①对极端参数或更大规模网络的性能模型仍需进一步验证；②ShuffleBox等光学设备尚未商业化，生产部署需额外工艺验证；③在极低流量稀疏场景（f<0.1）fat tree略优，需针对应用场景细化调优；④现有评估多聚焦于平均/worst‑case吞吐，缺少对实时延迟、拥塞感知等细粒度QoS评估；⑤对多租户故障定位与自动化运维工具的集成仍待完善。

---

## 505. Low-Cost System for Automatic Recognition of Driving Pattern in Assessing Interurban Mobility using Geo-Information

**arXiv ID:** 2604.15216 | [PDF](https://arxiv.org/pdf/2604.15216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 506. Agentic Microphysics: A Manifesto for Generative AI Safety

**arXiv ID:** 2604.15236 | [PDF](https://arxiv.org/pdf/2604.15236v1)

**作者:** Federico Pierucci `[一作]` (DEXAI), Piercosma Bisconti `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并执行了一个基于LLM的多代理社交媒体新闻源实验，探讨群体聚集（herding）机制。

**💡 创新点**

提出了“agentic microphysics”和“generative safety”两种概念，将微观交互规则与宏观安全风险关联，首次用生成实验方法重现并干预多代理AI风险。

**🔧 技术方法**

采用基于Molbook的可控多代理仿真平台，随机打乱新闻排序、可视/隐藏点赞信号，构建微观交互规则并进行参数化实验。

**📊 数据集**

使用人工构造的48条新闻项目，未使用真实网络数据，而是模拟的新闻槽位。

**📈 对比分析**

通过实验对比可见社交证明与位置排序对聚焦度的影响，结果显示位置排序决定主导，社交证明仅在前置位置产生阈值效应，证明方法能够精确区分机制。

**⚠️ 局限性**

仅在极简、人工构造环境下验证，缺乏对真实社交平台动态的外部验证，机制可能不适用于更复杂的多代理系统。

---

## 507. AI-Assisted Requirements Engineering: An Empirical Evaluation Relative to Expert Judgment

**arXiv ID:** 2604.15222 | [PDF](https://arxiv.org/pdf/2604.15222v1)

**作者:** Oz Levy `[一作]` (Holon Institute of Technology), Michael Winokur `[通讯]` (Holon Institute of Technology)

**通讯引用:** 181 | [OpenAlex ID](https://openalex.org/A5086551413)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究比较大型语言模型（Claude 3.5 Sonnet、GPT‑4o、Llama 3）在基于 INCOSE 标准的需求质量评估和功能/非功能分类任务中的表现，并与经验丰富的系统工程师及公开数据集（PROMISE）进行对比。

**💡 创新点**

创新点在于将 LLM 与人类专家的评估结果进行细粒度对比，提出 AI 作为决策支持而非替代者的工作流，并系统阐述各模型对 INCOSE 七大质量属性的偏好与一致性。

**🔧 技术方法**

主要技术包括 Prompt‑Engineering、Transformer‑based LLM 推理（ChatGPT‑4、Claude Sonnet、Llama 3），以及统计评估指标（准确率、精确率、召回率、F1、Bootstrap 置信区间）。

**📊 数据集**

使用两组数据集：DR Tool（医疗设备跟踪系统 107 条需求，供人类评审）和 PROMISE（969 条软件需求，分功能/非功能及细分子类）。

**📈 对比分析**

通过对比 AI 与单一人类共识以及所有个体评估的准确率，发现 Claude 3.5 Sonnet 在需求质量评估中达到约85% 的一致率，而 GPT‑4o 和 Llama 3 仅约45–48%；在 FR/NFR 分类上，GPT‑4o 最佳约85% 准确率，Claude 在 NFR 子类上显著领先。

**⚠️ 局限性**

局限性包括 AI 对技术真相、系统上下文和跨需求推理的缺乏，易出现幻觉或偏见；缺少对不同领域（非软件）需求的验证；以及对人类专家主观差异的处理仍有限。

---

## 508. Explicit Constant-Alphabet Subspace Design Codes

**arXiv ID:** 2604.15218 | [PDF](https://arxiv.org/pdf/2604.15218v1)

**作者:** Rohan Goyal `[一作]` (Massachusetts Institute of Technology), Jun-Ting Hsieh `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文构造了常数大小符号表的子空间设计码，并证明其满足与随机线性码相同的局部性质，尤其在列表解码与列表恢复上达到最优性能。

**💡 创新点**

创新点在于将Alon-Edmonds-Luby(AEL)框架与子空间设计属性结合，首次给出常数符号表的显式构造，并实现了更小的符号表尺寸和简化的参数分析。

**🔧 技术方法**

主要技术包括子空间设计的潜能函数与局部轮廓概念、谱图扩张的局部到全局原理以及对AEL构造的子空间设计保持性证明。

**📊 数据集**

无实验数据集，工作为理论构造与参数证明。

**📈 对比分析**

与之前的随机线性码及基于AG码的构造相比，本文在保持列表大小≈(ℓ/R)^{O(R)}的同时，符号表尺寸降至exp(exp(O(ℓ^2)))，几乎达到最优列表大小上限。

**⚠️ 局限性**

局限性包括符号表尺寸仍为双指数级，尚未突破到多项式或低阶指数；构造依赖于大块长的图扩展与随机内码，实际实现与计算复杂度尚待研究。

---

## 509. A Hierarchical Spatiotemporal Action Tokenizer for In-Context Imitation Learning in Robotics

**arXiv ID:** 2604.15215 | [PDF](https://arxiv.org/pdf/2604.15215v1)

**作者:** Fawad Javed Fateh `[一作]` (Retrocausal, Inc.), Quoc-Huy Tran `[通讯]` (Retrocausal, Inc.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种层次时空动作分词器HiST-AT，用于上下文模仿学习；

**💡 创新点**

创新点在于两层向量量化实现细粒度子动作与高层动作聚类，并通过时空重建联合恢复动作与时间戳，兼顾空间与时间信息；

**🔧 技术方法**

采用层次向量量化（HVQ）、Lipschitz 条件网络、时空自监督重建、Transformer 预测框架以及 VQ‑VAE 等技术；

**📊 数据集**

在模拟数据集 RoboCasa（MimicGen）和 ManiSkill 以及真实机器人数据集 Human dataset 上进行实验；

**📈 对比分析**

与 BC‑Transformer、ACT、MCR、MLP、VQ‑VAE、LipVQ‑VAE 等方法对比，HiST‑AT 在 RoboCasa 的平均成功率达到 59%（较 LipVQ‑VAE 的 53% 提升 6%），在 ManiSkill 超过 LipVQ‑VAE 5.3%，跨数据集提升 10%，零样本提升 6.2%；

**⚠️ 局限性**

局限在于需要手动调节超参数（如 λ_temp）、代码簇规模可能冗余，对时间戳质量敏感，且仍无法完全消除动作不连续性。

---

## 510. VisPCO: Visual Token Pruning Configuration Optimization via Budget-Aware Pareto-Frontier Learning for Vision-Language Models

**arXiv ID:** 2604.15188 | [PDF](https://arxiv.org/pdf/2604.15188v1)

**作者:** Huawei Ji `[一作]` (Shanghai Jiao Tong University), Xinbing Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 11318 | [OpenAlex ID](https://openalex.org/A5034483183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VisPCO，一种基于 Pareto 优化的视觉 token 剪枝配置自动搜索框架；

**💡 创新点**

通过连续松弛、直通估计和增量拉格朗日方法，实现对剪枝比率的梯度搜索，自动定位计算-性能 Pareto 前沿，并引入可学习核函数揭示多步渐进剪枝的优越性；

**🔧 技术方法**

使用连续松弛、直通估计、增量拉格朗日法、Gaussian/Softmax 软阈值、可学习的多步 sigmoid 核、KL 散度损失、FLOPs 约束等；

**📊 数据集**

在 Qwen2.5VL‑3B 上训练 30K LLaVA‑Instruct‑150K 样本，并在 8 个视觉基准（A‑OKVQA、VizWiz、SEEDBench、MMBench、MME、ChartQA、OCRBench、TextVQA）进行评估；

**📈 对比分析**

与多种基线（FastV、SparseVLM、FitPrune、VTW、G‑Search、ATP‑LLaVA、MADTP、AIM、随机搜索）对比，VisPCO 在 50% FLOPs 预算下提升约 10%–15% 的平均准确率，同时搜索时间仅 1 小时；在不同 VLM（Gemma3‑4B、LLaVA‑v1.5‑7B）上也能保持优越性能；

**⚠️ 局限性**

局限性包括：只验证单图任务，缺乏多图/视频扩展；核函数结构固定，未探索更灵活的非参数或输入自适应剪枝模式；

---

## 511. An Analysis of Regularization and Fokker-Planck Residuals in Diffusion Models for Image Generation

**arXiv ID:** 2604.15171 | [PDF](https://arxiv.org/pdf/2604.15171v1)

**作者:** Onno Niemann `[一作]` (Universidad Autónoma de Madrid), Alberto Suárez Gonzalez `[通讯]` (Universidad Autónoma de Madrid)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析了简化正则化对扩散模型的Fokker–Planck残差和生成质量的影响

**💡 创新点**

证明轻量级正则项在不显著增加计算成本的前提下，能近似或匹配完整FP正则的性能

**🔧 技术方法**

使用Denoising Score Matching、Fokker–Planck残差评估、Hutchinson估计Jacobian与divergence进行正则化实验

**📊 数据集**

在MNIST手写数字数据集上进行实验

**📈 对比分析**

与无正则化基线及完整FP正则对比，使用FID、Density、Coverage、Entropy等指标评估；轻量级正则在FID等指标上接近FP正则，训练时间仅略高

**⚠️ 局限性**

仅在MNIST上验证，复杂数据集上的效果与计算开销待进一步验证

---

## 512. Structural Dependency Analysis for Masked NTT Hardware: Scalable Pre-Silicon Verification of Post-Quantum Cryptographic Accelerators

**arXiv ID:** 2604.15249 | [PDF](https://arxiv.org/pdf/2604.15249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 513. SegWithU: Uncertainty as Perturbation Energy for Single-Forward-Pass Risk-Aware Medical Image Segmentation

**arXiv ID:** 2604.15271 | [PDF](https://arxiv.org/pdf/2604.15271v1)

**作者:** Tianhao Fu `[一作]` (University of Toronto), Yucheng Chen `[通讯]` (Project Neura)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 SegWithU，一个在冻结预训练分割骨干上附加轻量级不确定性头的后置框架，用于生成校准和错误检测的两种不确定性地图。

**💡 创新点**

创新点在于采用 rank‑1 后验探针对特征空间进行扰动能量建模，分离校准与错误排名的两条不确定性通道，并在不重新训练骨干的前提下实现单前向推理的可靠性评估。

**🔧 技术方法**

使用了扰动能量视角的 Rank‑1 探针、1×1 卷积映射、误差相关、对比排序、尾部、信任与锚点等多项损失，构建了轻量级的校准与排名不确定性头。

**📊 数据集**

在三大医学分割数据集上评估：ACDC、BraTS2024 和 LiTS。

**📈 对比分析**

与深度集成、Monte‑Carlo Dropout、测试时增强、温度缩放、DUQ、DDU‑Seg、DUE 等单前向或多前向基线对比，SegWithU 在 AUROC/AURC、Brier 与 Dice 上表现最佳或相近，尤其在排名相关指标上显著优于其他单前向方法，且与多前向方法竞争。

**⚠️ 局限性**

局限性包括仍需有标注数据进行不确定性头训练、在某些数据集上对多前向基线并非绝对优越、对域迁移的鲁棒性需进一步验证，以及在极端病例或小样本场景下的不确定性表现可能受限。

---

## 514. CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas

**arXiv ID:** 2604.15267 | [PDF](https://arxiv.org/pdf/2604.15267v1)

**作者:** Emanuel Tewolde `[一作]` (Carnegie Mellon University), Zhijing Jin `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大语言模型（LLM）在社会困境中的合作行为，并比较四种基于博弈论的合作机制（重复游戏、声誉、第三方调解、合同）在不同游戏（囚徒困境、旅行者困境、公共物品、信任游戏）中的效果。

**💡 创新点**

首次系统性评测LLM在多机制下的合作表现，提出统一合作定理证明这四种机制在理论上能实现合作，并通过实验验证其有效性；同时展示进化动力学提升合作率的实证结果。

**🔧 技术方法**

使用自然语言提示让LLM在游戏中做决策；评估基于重复、声誉、调解器和合同的机制；采用链式思维（CoT）和LLM评判者评估决策理由；使用复制器动力学模拟演化竞争；利用偏差评级（Deviation Ratings）对LLM进行排名。

**📊 数据集**

利用自构造的20+个合作问题（囚徒困境、旅行者困境、公共物品、信任游戏）与六种LLM模型（Claude Sonnet 4.5、GPT‑5.2、Gemini‑3 Flash、Gemini‑3 Medium、GPT‑4o、Qwen‑3‑30B‑A3B‑Instruct‑2507）进行交叉实验；不使用公开游戏数据集，而是通过自然语言描述生成游戏和机制。

**📈 对比分析**

对每个机制×游戏×LLM组合做三次实验，计算平均收益、演化后收益和偏差评级；结果显示：无机制时LLM几乎完全背叛；合同和调解器机制最高，合作率可达80%+；在进化动力学下合作率提升至90%–100%，平均收益显著高于基线。

**⚠️ 局限性**

局限：仅评估单轮或有限回合的社会困境，未覆盖更复杂的序列游戏；LLM在高阶信息（如多级声誉）上的表现不佳；机制提议质量受限，可能导致合谋或不公平；实验受限于六种LLM模型，难以推广到更广泛的模型族。

---

## 515. RL-STPA: Adapting System-Theoretic Hazard Analysis for Safety-Critical Reinforcement Learning

**arXiv ID:** 2604.15201 | [PDF](https://arxiv.org/pdf/2604.15201v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 516. Democratization of Real-time Multi-Spectral Photoacoustic Imaging: Open-Sourced System Architecture for OPOTEK Phocus & Verasonics Vantage Combination

**arXiv ID:** 2604.15255 | [PDF](https://arxiv.org/pdf/2604.15255v1)

**作者:** Ryo Murakami `[一作]` (Worcester Polytechnic Institute), Haichong K. Zhang `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 1184 | [OpenAlex ID](https://openalex.org/A5077612510)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一套开源的硬件‑软件架构，通过在OPOTEK Phocus激光器与Verasonics Vantage系统之间嵌入独立微控制器监控激光触发信号，并采用客户端‑服务器模型实现数据流传输，从而实现了实时多光谱光声成像（RT‑mPAI）的稳定同步与无本地存储的持续采集。

**💡 创新点**

创新点在于①使用独立微控制器进行确定性触发计数，克服Windows非实时系统带来的时序不稳定；②通过TCP流式传输与共享内存解耦，提升采集连续性与后处理灵活性；③开放源码与协作平台，降低技术与成本门槛，推动RT‑mPAI社区共建。

**🔧 技术方法**

核心技术包括：Arduino（或类似单片机）实现双触发计数；Verasonics Vantage 128的Flashlamp触发采集；TCP协议实现主机到工作站的数据流；MATLAB共享内存与数据打包；可选的C++/Python实现后处理；光声成像实验中使用的光谱分析与能量补偿框架。

**📊 数据集**

使用自制的蓝色/黑色金属线点源水腔模型作为光声phantom，激光波长设为700, 740, 760, 780 nm，20 Hz重复率，500帧/波长；对比标准扫描模式下的光谱作为基准。

**📈 对比分析**

通过在实时光谱采集结果与标准扫描基准光谱进行数值对比，验证了波长分配的准确性；人工引入±1/±2帧偏移后光谱明显失真，证明同步机制稳健。性能方面，系统实现了无本地存储、无限持续采集，且后处理可在GPU/多语言环境下高效运行。

**⚠️ 局限性**

主要限制包括：微控制器计数虽稳定但偶尔存在漏帧；实时能量补偿尚未实现；Verasonics与MATLAB之间的固定延迟导致微小轴向偏移；仅验证了Philips ATL Lap L9‑5探头，需扩展到其他探头；当前实现以MATLAB为主，C++版待发布；最终验证仍缺少光谱计的逐脉冲波长确认。

---

## 517. TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable Tokens

**arXiv ID:** 2604.15239 | [PDF](https://arxiv.org/pdf/2604.15239v1)

**作者:** Jiawei Ren `[一作]` (NVIDIA), Zan Gojcic `[通讯]` (NVIDIA)

**通讯引用:** 2579 | [OpenAlex ID](https://openalex.org/A5025638024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种前向式3D高斯散点（3DGS）框架，直接回归三维高斯中心坐标，并通过可学习的高斯令牌解耦对输入像素的依赖，实现了更紧凑、更具可扩展性的3D重建。

**💡 创新点**

核心创新点包括：①将高斯均值从光线深度预测转变为直接三维坐标回归；②使用可学习高斯令牌与交叉注意力完成参数预测，从而将高斯数量与图像分辨率和视角数解耦；③引入可见性损失防止“漂浮”点；④支持动态场景的时序令牌分离（静态/动态令牌）和时间编码；⑤提供两种测试时可扩展方案（上下文扩展与令牌微调）。

**🔧 技术方法**

技术实现上采用了ViT编码器、DETR式解码器以及Transformer的交叉注意力和自注意力机制；使用FlashAttention2/FlexAttention加速；通过体渲染监督（MSE+SSIM）和可见性正则化训练；在动态场景中引入时间嵌入；测试时令牌微调只更新令牌嵌入，保持网络大部分权重冻结。

**📊 数据集**

在RE10K、DL3DV、Kubric以及Objaverse等多视图与动态场景数据集上进行评测，分别覆盖静态重建、视角外推和时间序列重建。

**📈 对比分析**

与GS‑LRM、BTimer等基准进行对比，实验显示：在相同或更少的高斯数量下实现与基准相当或更优的PSNR/LPIPS；在相机噪声、视角外推和动态重建任务上表现更稳健，尤其在动态对象运动恢复和场景流生成上获得更高的精度；测试时上下文扩展和令牌微调进一步提升了重建质量。

**⚠️ 局限性**

局限性在于：1）仍难以处理极大规模场景或细节级别的几何；2）测试时令牌微调需要数十步梯度更新，算力开销较大；3）若直接对高斯参数做优化，容易破坏先验，导致几何退化。

---

## 518. Blue Data Intelligence Layer: Streaming Data and Agents for Multi-source Multi-modal Data-Centric Applications

**arXiv ID:** 2604.15233 | [PDF](https://arxiv.org/pdf/2604.15233v1)

**作者:** Moin Aminnaseri `[一作]` (Megagon AI), Dan Zhang `[通讯]` (Megagon AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Blue 系统中的 Data Intelligence Layer (DIL)，通过统一数据注册、规划与多模态算子实现多源多模态查询与数据处理。

**💡 创新点**

创新点在于将 LLM、Web、用户等非传统数据库也视为第一类可查询的数据源，并设计可声明式的 DAG 规划器与统一算子框架。

**🔧 技术方法**

采用数据注册表、DAG 规划器、抽象与物理算子层次、LLM 接口、Web 抓取、向量检索等技术。

**📊 数据集**

主要使用企业内部结构化数据库、公开的招聘与租房列表、图像识别的厨房照片以及公开的食谱数据库；实验中未使用专门的公开数据集。

**📈 对比分析**

对比方法方面未给出量化实验，只在两条交互式演示场景（公寓搜索与烹饪助手）中展示效果，缺乏性能评估。

**⚠️ 局限性**

局限性包括缺乏大规模实验验证、对复杂语义推理的支持有限、以及对实时更新与多源同步的细节处理不足。

---

## 519. One-shot learning for the complex dynamical behaviors of weakly nonlinear forced oscillators

**arXiv ID:** 2604.15181 | [PDF](https://arxiv.org/pdf/2604.15181v1)

**作者:** Teng Ma `[一作]` (Tongji University), Attilio Frangi `[通讯]` (Politecnico di Milano)

**通讯引用:** 33419 | [OpenAlex ID](https://openalex.org/A5049192404)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于MEv‑SINDy的单点学习方法，可从单一激励时间历史中推断弱非线性受迫振荡器的全频响应曲线；

**💡 创新点**

创新点在于将EvLOWN演化学习与通用谐波平衡方法相结合，构建多频、非自治的稀疏识别框架，实现一次性训练即可外推整个频率响应；

**🔧 技术方法**

使用了稀疏回归（SINDy）、通用谐波平衡（GHB）、Hilbert滤波、频率归一化以及参数延拓（弧长延拓）等技术；

**📊 数据集**

数据集主要来自两类MEMS系统：双夹持梁谐振器（2607节点 FEM）和扫描微镜（9723节点 FEM），均仅用单个激励参数组合进行训练；

**📈 对比分析**

与传统全阶 FEM、POD‑DL‑ROM 等方法对比，MEv‑SINDy在单点训练下即可重现软化/硬化效应、跳变等非线性特征，预测误差（MCDRC）均小于0.1，且显著降低了计算成本；

**⚠️ 局限性**

局限性包括对非线性显著区间的依赖（线性或弱非线性区间训练效果差），以及在多点训练时需做频率归一化处理，未在多模态强耦合大规模系统中全面验证。

---

## 520. AdaSplash-2: Faster Differentiable Sparse Attention

**arXiv ID:** 2604.15180 | [PDF](https://arxiv.org/pdf/2604.15180v1)

**作者:** Nuno Gonçalves `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), Marcos Treviso `[通讯]` (Instituto Superior Técnico, Universidade de Lisboa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种硬件感知的稀疏注意力实现方法，利用在芯片 SRAM 上构建的直方图快速估计归一化阈值，从而显著减少了 α-entmax 注意力的迭代次数。

**💡 创新点**

核心创新在于基于直方图的阈值初始化，保证了下界估计并使得只需 1–2 次迭代即可得到精确的归一化；同时设计了轻量级位压缩块掩码以高效跳过零块。

**🔧 技术方法**

采用了 α-entmax（可变稀疏softmax）注意力、基于直方图的根求解器、混合 Halley/牛顿/割线迭代、Triton GPU 核函数、位压缩掩码以及自定义的前向后向链路。

**📊 数据集**

在自定义长序列数据集上进行实验，使用 DCLM-Edu 预训练、ProLong 长上下文扩展以及 RULER、HELMET、OLMES 等标准 NLP 评测数据集。

**📈 对比分析**

与 FlashAttention‑2（软max）以及之前的 α-entmax GPU 实现对比，显示在 60%+ 块稀疏度下可实现 2× 以上的前向+后向加速，并在长上下文语言建模任务中超越或匹配软max 基线的精度。

**⚠️ 局限性**

主要局限在于对极端稀疏度（块数大于 SRAM 容量）需要额外刷新机制；在软max 与稀疏模型的精度差距仍受位置编码方式影响；并且在推理阶段尚未实现完整的低成本稀疏化实现。

---

## 521. Complexity of Fungal Automaton Prediction

**arXiv ID:** 2604.15177 | [PDF](https://arxiv.org/pdf/2604.15177v1)

**作者:** Enrico Formenti `[一作]` (Université Côte d'Azur), Domingo Ruiz-Tala `[通讯]` (Universidad de Chile)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

研究了由一维总计冻结规则产生的真菌自动机的预测问题，并确定其计算复杂度。

**💡 创新点**

将传统一维CA的冻结总计规则推广为二维真菌自动机，并证明某些规则预测问题属于NL，甚至AC0，而半径1.5的多数规则使问题成为P-完整。

**🔧 技术方法**

通过与电路值问题的归约、结构化联盟与阶梯分析、以及基于瓷砖与电路网格的构造来评估预测复杂度。

**📊 数据集**

本研究以理论构造为主，不涉及实际数据集。

**📈 对比分析**

通过复杂度理论比较，将问题归入AC0、NL和P-完整三类，展示了不同规则对预测难度的影响。

**⚠️ 局限性**

仅覆盖半径1的冻结总计规则及其一维多半径1.5多数规则，其他更一般规则的复杂度仍未完全解明。

---

## 522. LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories

**arXiv ID:** 2604.15311 | [PDF](https://arxiv.org/pdf/2604.15311v1)

**作者:** Zhanhao Liang `[一作]` (Australian National University), Liang Zheng `[通讯]` (Australian National University)

**通讯引用:** 36868 | [OpenAlex ID](https://openalex.org/A5100709340)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LeapAlign，一种基于流匹配模型的后训练方法，通过构造两步跳跃轨迹实现对早期生成步骤的梯度更新；

**💡 创新点**

创新点在于：①使用两步跳跃轨迹在保持内存常数的前提下可更新任意步骤；②梯度折扣机制保留嵌套梯度并抑制梯度爆炸；③轨迹相似性加权提升学习信号；

**🔧 技术方法**

技术：流匹配模型、梯度折扣、轨迹相似性加权、直接梯度优化、分类器无监督指导；

**📊 数据集**

数据集：Flux模型的HPDv2（50k提示）和MJHQ-30k（50k提示）用于一般偏好对齐；GenEval 553提示用于组合对齐；

**📈 对比分析**

与GRPO（DanceGRPO、MixGRPO）、直接梯度方法（ReFL、DRaFT-LV、DRTune）对比；LeapAlign在HPSv2.1、HPSv3、PickScore、UnifiedReward、ImageReward等指标上均获得最高或最接近最高分，且在GenEval整体及各子任务上明显优于基线；

**⚠️ 局限性**

局限：仅适用于可微奖励模型，无法直接处理非可微奖励；在极低步长或单步生成模型中的必要性不高；未来需验证在视频生成中的适用性。

---

## 523. Benchmarking Optimizers for MLPs in Tabular Deep Learning

**arXiv ID:** 2604.15297 | [PDF](https://arxiv.org/pdf/2604.15297v1)

**作者:** Yury Gorishniy `[一作]` (Yandex), Artem Babenko `[通讯]` (Yandex)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对tabular数据上MLP模型的优化器进行统一实验协议下的系统基准测试，评估多种优化器的泛化性能。

**💡 创新点**

首次在tabular深度学习场景中系统验证Mu­on优化器优于AdamW，并揭示EMA权重平均能显著提升Vanilla MLP的效果。

**🔧 技术方法**

使用Mu­on、AdamW、AdamW+EMA、Schedule‑Free AdamW、Lion、Signum、SOAP等多种优化器，并结合全局梯度裁剪、早停和统一超参调优策略。

**📊 数据集**

采用公开的学术与工业Tabular数据集（含分类与回归任务），包括TabReD基准数据以及其他标准学术数据集。

**📈 对比分析**

通过统一的超参调优预算、10次随机种子重跑和Δ_score、win/tie/loss统计方式进行比较；结果显示Mu­on在大多数数据集上领先，EMA和Schedule‑Free也可取得可观提升；Mu­on平均训练速度比AdamW慢约3倍。

**⚠️ 局限性**

研究仅覆盖MLP族，未探讨非MLP架构或tabular基础模型；Mu­on的计算开销显著高于AdamW；实验结果纯经验，缺乏理论解释。

---

## 524. Pure Borrow: Linear Haskell Meets Rust-Style Borrowing

**arXiv ID:** 2604.15290 | [PDF](https://arxiv.org/pdf/2604.15290v1)

**作者:** Yusuke Matsushita `[一作]`, Hiromi Ishii `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

无法得知

**💡 创新点**

无法得知

**🔧 技术方法**

无法得知

**📊 数据集**

无法得知

**📈 对比分析**

无法得知

**⚠️ 局限性**

无法得知

---

## 525. AnimationBench: Are Video Models Good at Character-Centric Animation?

**arXiv ID:** 2604.15299 | [PDF](https://arxiv.org/pdf/2604.15299v1)

**作者:** Leyi Wu `[一作]` (HKUST), Qifeng Chen `[通讯]` (HKUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AnimationBench 基准，评估动画视频生成模型在 IP 保留、十二条动画基本原则以及更广泛质量维度上的表现，并支持闭集和开放集两种评测模式。

**💡 创新点**

首次将专业动画原则与 IP 保真转化为可量化、可复现的评估维度，并采用 VLM 驱动的结构化问答框架实现大规模、可扩展的评测；同时提供开放集诊断与 Prompt 迭代机制。

**🔧 技术方法**

使用 Qwen3-VL-MAX 视觉语言模型进行多维度结构化评估，配合光流、面积变化、形变量化等算法；结合开源与闭源视频生成模型（如 Wan2.2、Sora2‑Pro、Seedance‑Pro、Kling2.6 等）进行基准测试。

**📊 数据集**

构建了 30 个自创与 10 个现有 IP 的 2D/3D 角色集合，生成 170 张源图和 360 条定制提示，共计 360 条视频样本；数据集覆盖多种风格（迪士尼、日漫、漫画等）和角色类型。

**📈 对比分析**

通过与人工偏好对齐验证，AnimationBench 的维度得分与人类评分高度相关；在闭集评测中，闭源模型如 Kling2.6、Veo3.1、Seedance2.0 表现最佳，开放集诊断后，Wan2.2 的语义一致性显著提升。

**⚠️ 局限性**

受限于 VLM 逻辑误差和对高度夸张表情、细腻动作的捕捉不足，难以全面评估极端动画细节；评测集主要以 2D 动画为主，缺乏 3D 动态一致性与多视角连续性评估。

---

## 526. AD4AD: Benchmarking Visual Anomaly Detection Models for Safer Autonomous Driving

**arXiv ID:** 2604.15291 | [PDF](https://arxiv.org/pdf/2604.15291v1)

**作者:** Fabrizio Genilotti `[一作]` (University of Padova), Gian Antonio Susto `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对八种视觉异常检测（VAD）方法在自动驾驶场景中的表现进行系统性基准测试，并评估不同特征提取器在边缘部署中的性能和资源消耗。

**💡 创新点**

首次在真实道路环境（AnoVox）上对VAD方法进行评测，揭示Transformer骨干网络在异常定位上的优势，并提出Tiny‑Dinomaly在内存和推理速度上实现高效边缘部署的方案。

**🔧 技术方法**

使用基于特征嵌入的VAD方法（PatchCore、PaDiM、CFA、STFPM、RD4AD、SSNet、FastFlow、Dinomaly）以及多种CNN/ViT骨干网络（WideResNet、MobileNet、DeiT‑Small、DeiT‑Tiny），结合像素级异常图生成。

**📊 数据集**

使用CARLA模拟的合成道路异常数据集AnoVox（包含1850帧、14.8%异常样本）。

**📈 对比分析**

通过Image‑Level AUROC/PR/F1、Pixel‑Level AUROC/PR/F1、PRO、内存占用、推理时间等指标对比，发现Dinomaly在DeiT‑Small/DeiT‑Tiny上达到最高像素定位精度，而Tiny‑Dinomaly在内存和速度上最优，且与大骨干网络的定位性能相当。

**⚠️ 局限性**

局限性包括对小尺寸或远距离异常检测能力不足、在曲线路段因透视失真导致异常图误差、以及在极低资源环境下某些方法（如FastFlow、CFA）性能下降。

---

## 527. MM-WebAgent: A Hierarchical Multimodal Web Agent for Webpage Generation

**arXiv ID:** 2604.15309 | [PDF](https://arxiv.org/pdf/2604.15309v1)

**作者:** Yan Li `[一作]` (Shanghai Jiao Tong University), Chong Luo `[通讯]` (Microsoft Corporation)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种分层多模态网页生成框架MM-WebAgent，模拟人类设计师的规划-生成-反思循环，实现全流程的网页布局、文本、图像、视频与图表的协同生成与迭代优化。

**💡 创新点**

创新点在于：①分层规划——先生成全局布局计划再生成局部元素计划，保证多模态资产与整体结构的语义一致；②多级自我反思——局部、上下文、全局三个层面迭代改进，提升资产质量与页面整体一致性；③引入专门的多模态生成工具并结合全局评估，实现端到端的“代码+原生资产”生成；④构建MM-WebBench基准，提供多样化布局、视觉风格与多模态组合的评估集合。

**🔧 技术方法**

技术包括：大型语言模型（GPT‑5.1）负责规划与反思，专用图像生成模型（GPT‑Image‑1）和视频模型（Sora‑2）生成原生多模态资产，ECharts‑HTML生成图表；HTML/CSS自动化生成；多级评估与基于规则的惩罚+分级打分；并使用多轮LLM调用实现迭代反思。

**📊 数据集**

使用自行构造的MM-WebBench评估集（120页），覆盖多布局、视觉风格、意图类别与多模态元素组合；对比WebGen‑Bench测试时尚功能代码完成度。

**📈 对比分析**

与代码一次性生成、代码仅限代理以及其他基于OpenHands、Bolt.diy等代理框架做对比；在MM-WebBench上，MM‑WebAgent平均得分0.75，局部指标（图像0.88、视频0.75、图表0.54）显著高于所有基线；在WebGen‑Bench中，准确率47.8%与外观分3.9/10，已达同类最佳。

**⚠️ 局限性**

局限性包括：依赖外部AIGC工具，易受工具不稳定、过滤或可用性变化影响；工具集与调用方式固定，缺乏动态选择；框架采用无监督的编排式代理，未利用强化学习等学习方法进一步优化决策。

---

## 528. RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework

**arXiv ID:** 2604.15308 | [PDF](https://arxiv.org/pdf/2604.15308v1)

**作者:** Hao Gao `[一作]` (Huazhong University of Science & Technology), Xinggang Wang `[通讯]` (Huazhong University of Science & Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个统一的生成器-判别器框架，将扩散式多模态轨迹生成器与强化学习优化的判别器相结合，实现闭环自适应规划；

**💡 创新点**

创新点包括：①将生成器与判别器解耦，避免高维轨迹空间直接接收稀疏奖励；②引入时序一致的群组相对策略优化（TC‑GRPO）提升信用分配稳定性；③提出按策略的生成器优化（OGO），通过结构化纵向优化将生成器逐步迁移到高奖励轨迹流形；④开发 BEV‑Warp 高吞吐量特征级闭环仿真环境；

**🔧 技术方法**

主要技术包括：扩散模型、Transformer 判别器、强化学习（TC‑GRPO）、按策略生成器优化、iLQR 控制器、BEV‑Warp 以及 3D Gaussian Splatting 仿真；

**📊 数据集**

使用约 50,000 小时真实驾驶数据进行生成器预训练，采集 50k 片段做闭环训练与评估（安全/效率两类），再用 1,044 片段在 3DGS 环境训练判别器，并利用 Senna‑2 数据集进行开放式轨迹评测；

**📈 对比分析**

与 ResAD、VAD、Senna‑2 等基线对比，RAD‑2 在 BEV‑Warp 环境下将碰撞率从 0.53 降至 0.23（约 56% 降幅），安全率 Safety@1 提升至 0.73，效率指标 EP‑Mean 提升至 0.99；在 3DGS 真实感仿真中同样取得最低碰撞率 0.25 与最高 Safety@1 0.72；开放式评测中 FDE/ADE 进一步降低，碰撞率降至 0.142%；

**⚠️ 局限性**

主要局限在 BEV‑Warp 只适用于基于 BEV 结构的感知模型，对无显式空间网格的视觉或统一潜在嵌入网络的迁移受限；生成式世界模型虽可提升逼真度，但存在计算开销大、时间漂移大等问题，未来需改进其推理效率与长期一致性。

---

## 529. How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study

**arXiv ID:** 2604.15294 | [PDF](https://arxiv.org/pdf/2604.15294v1)

**作者:** Zhen Yang `[一作]` (Beijing Institute of Technology), Wenpeng Lu `[通讯]` (Qilu University of Technology)

**通讯引用:** 1528 | [OpenAlex ID](https://openalex.org/A5076564877)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在无视觉输入的文本场景下，语言模型对视角旋转理解（VRU）的能力，并提出了VRUBench数据集；

**💡 创新点**

创新点在于：①首次量化并比较LLM与VLM在文本视角旋转任务上的差距；②利用层级探测与路径补丁方法定位关键注意力头，揭示模型内部的认知机制；③通过选择性微调这些关键头显著提升VRU表现且不导致通用能力的灾难性遗忘；

**🔧 技术方法**

技术方法包括：层级线性探测、注意力头的因果路径补丁（Path Patching）以及对关键头的选择性监督微调；

**📊 数据集**

使用自构造的VRUBench（约19,600条实例，包含2-5步视角旋转）作为主要数据集，并在SpinBench、MMLU、BBH上进行跨域评估；

**📈 对比分析**

与人类基准（100%准确）以及多种LLM/VLM（如Qwen、Gemini）对比，显示大多数模型在VRU上仅达约40-70%准确；通过选择性微调后，VRU准确率提升至≈70-80%，同时保持通用能力；

**⚠️ 局限性**

局限性：模型对提示措辞高度敏感；选择性微调仅在规模≤7B模型上验证；未探究更大规模模型或不同训练策略的效果；

---

## 530. Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo

**arXiv ID:** 2604.15312 | [PDF](https://arxiv.org/pdf/2604.15312v1)

**作者:** Ninghui Xu `[一作]` (Southeast University), Stefano Mattoccia `[通讯]` (University of Bologna)

**通讯引用:** 5541 | [OpenAlex ID](https://openalex.org/A5072569849)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Bi‑CMPStereo，一种双向跨模态提示框架，用于事件–帧非对称立体匹配。

**💡 创新点**

创新点在于引入交叉域嵌入适配器 (CDEA) 与立体规范化约束 (SCC) 以保留域特征，同时采用层级视觉转换 (HVT) 防止捷径学习，从而实现高保真跨模态对齐。

**🔧 技术方法**

核心技术包括事件与帧的两种专属表示（体素网格与事件浓缩）、跨域嵌入适配、立体规范化约束、层级视觉转换、组间相关成本体积、3D Hourglass 以及 ConvGRU 级联细化。

**📊 数据集**

使用 DSEC、MVSEC 和 M3ED 三个公开事件立体数据集进行训练与评估。

**📈 对比分析**

与 ZEST、SEVFI、SE‑CFF、DTC 等基线方法对比，Bi‑CMPStereo 在 DSEC 上 MAE、PE、RMSE 均居前列，并在 MVSEC/M3ED 上展现优异的跨数据集泛化能力。

**⚠️ 局限性**

局限性包括对硬件的依赖（需事件摄像头和RGB摄像头双传感器）、对极端稀疏或极暗场景的鲁棒性仍有提升空间，以及模型训练与推理时的计算开销相对较大。

---

## 531. Diagnosing LLM Judge Reliability: Conformal Prediction Sets and Transitivity Violations

**arXiv ID:** 2604.15302 | [PDF](https://arxiv.org/pdf/2604.15302v1)

**作者:** Manan Gupta `[一作]` (BITS Pilani), Dhruv Kumar `[通讯]` (BITS Pilani)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了两种低成本诊断工具，用以评估LLM作为评审者的实例级可靠性。

**💡 创新点**

创新点在于将转移性（3‑cycle）分析与分割式合成预测相结合，并将预测集宽度作为实例级信任信号，首次在SummEval上系统比较两种诊断。

**🔧 技术方法**

使用的技术包括转移性分析（计算有向3‑cycle违例率）、最小反馈弧集（MFAS）排名修复、分割式合成预测（split conformal prediction）以及Spearman相关和Kendall τ比较。

**📊 数据集**

实验数据来自SummEval数据集，包含30篇文档、8个系统生成的摘要，以及四个LLM评审器（GPT‑4o‑mini、LLaMA‑3.1‑70B、Qwen‑2.5‑72B、Mistral‑Small‑3.1）。

**📈 对比分析**

与人类评分的相关度通过Kendall τ和覆盖率评估，发现转移性分析显示约50%文档存在违例，而合成预测在保持90%覆盖率的同时，预测集宽度与实际误差相关系数达到+0.576；两种诊断均指出相同的结论：评价维度比评审器更决定可靠性。

**⚠️ 局限性**

局限性包括仅评估SummEval的子集、仅提供边际覆盖率、使用固定的绝对残差非合规度、对提示词敏感性未充分检验，以及对人类评分四舍五入带来的离散误差。

---

## 532. Abstract Sim2Real through Approximate Information States

**arXiv ID:** 2604.15289 | [PDF](https://arxiv.org/pdf/2604.15289v1)

**作者:** Yunfu Deng `[一作]` (University of Wisconsin--Madison), Josiah P. Hanna `[通讯]` (University of Wisconsin--Madison)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了将抽象仿真环境与真实机器人之间进行策略迁移的框架并实现了ASTRA方法。

**💡 创新点**

创新点在于将状态抽象产生的部分可观测性转化为历史依赖，并引入自预测信息状态约束来训练动态校正与奖励预测。

**🔧 技术方法**

采用RNN+GRU结构、最大均方差对齐、PPO/SAC强化学习以及自监督损失组合的技术。

**📊 数据集**

使用AntMaze、Humanoid benchmark、NAO机器人导航与踢球等数据集，收集随机策略轨迹用于仿真校正。

**📈 对比分析**

与直接迁移、域随机化、COMPASS、RMA、NAS、IQL微调等方法比较，ASTRA在两类仿真和真实机器人任务中均取得最高成功率，显著优于基线。

**⚠️ 局限性**

局限性包括需已知状态映射、仍需一定量真实数据、若抽象过度历史校正效果有限。

---

## 533. TokenLight: Precise Lighting Control in Images using Attribute Tokens

**arXiv ID:** 2604.15310 | [PDF](https://arxiv.org/pdf/2604.15310v1)

**作者:** Sumit Chaturvedi `[一作]` (Yale University), Zhixin Shu `[通讯]` (Adobe)

**通讯引用:** 2074 | [OpenAlex ID](https://openalex.org/A5034904892)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于扩散变换器的图像重照明方法，利用物理可解释的灯光属性令牌实现对光强、光色、扩散度、3D位置等多属性的精细连续控制。

**💡 创新点**

创新点在于：①将灯光属性编码为令牌并直接作为扩散模型的条件输入；②不依赖逆渲染或几何重建即可在图像域内完成空间局部光照编辑；③在同一框架下支持添加虚拟光源、编辑环境光、控制场景中光源开关等多种重照明任务。

**🔧 技术方法**

技术方法包括：扩散变换器（预训练于文本到图像/视频任务）+ 条件扩散 + Gaussian Fourier Features 对属性令牌进行编码 + Flow‑Matching 训练目标 + Classifier‑free guidance 以提升光照条件的控制效果。

**📊 数据集**

数据集：大规模合成渲染数据（Blender Cycles渲染的Objaverse、PolyHaven HDRI、手工场景等），在多种灯光属性组合下生成图像对；以及约600张真实室内照片用于跨域学习；合成对通过预渲染光源线性组合得到监督。

**📈 对比分析**

与Neural Gaffer、DiffusionRenderer、GenLit、Careaga等基线在合成与真实数据上进行定量（PSNR/SSIM/LPIPS）和定性对比。结果显示在点光源与环境光目标上均优于基线；用户研究中更受欢迎；在可见光源开/关任务中与ScribbleLight相当或更优，能处理透明物体、发光体等难点。

**⚠️ 局限性**

局限性：①依赖合成数据，真实场景中的极端光照或复杂材质（高反射、半透明）仍可能失效；②缺乏对动态场景或实时视频的支持；③对输入分辨率有限，且需要预定义灯光属性范围；④对非常高动态范围的光照处理能力有限。

---

## 534. Generalization in LLM Problem Solving: The Case of the Shortest Path

**arXiv ID:** 2604.15306 | [PDF](https://arxiv.org/pdf/2604.15306v1)

**作者:** Yao Tong `[一作]` (National University of Singapore), Reza Shokri `[通讯]` (National University of Singapore)

**通讯引用:** 13317 | [OpenAlex ID](https://openalex.org/A5084892128)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

本文在一个可控的合成最短路径规划环境中，系统评估语言模型在空间迁移与长度扩展两种推理泛化维度下的表现，并剖析数据、训练范式与推理时策略对性能的影响。

**💡 创新点**

创新点在于将推理泛化拆解为空间与长度两个互补维度，构建纯可验证的合成 SOP 实验平台，揭示模型在长度扩展上因递归不稳定而失败，并提供对数据覆盖度、RL 稳定性及推理搜索的细粒度解析。

**🔧 技术方法**

采用基于 LLaMA 架构的 8 层 8 头 Transformer（带 RoPE）进行预训练、监督微调与 GRPO 强化学习训练，并在推理阶段使用自洽（Self‑Consistency）、best‑of‑N 与目标引导的 Shortest‑of‑10 等搜索策略。

**📊 数据集**

实验数据主要来自人工生成的稀疏格点地图（如 50×40 结构），并以随机漫步路径进行预训练；此外还以 MathQA 作为案例研究的实际数据集。

**📈 对比分析**

与单纯监督微调相比，RL 仅能稳定训练、抑制过拟合，且在空间迁移和长度扩展上未突破最佳 SFT 极限；自洽与 Shortest‑of‑10 等推理搜索可提升成功率，但仍无法弥补长度扩展失效；添加略长路径可显著恢复长度泛化。

**⚠️ 局限性**

主要局限在于模型对长度扩展仍表现递归不稳定，RL 无法提升能力极限；实验仅限于合成最短路径任务，缺乏对更复杂真实场景的验证；并且使用的 8‑层小模型可能不具备更大规模模型的表现。

---

## 535. Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation

**arXiv ID:** 2604.15301 | [PDF](https://arxiv.org/pdf/2604.15301v1)

**作者:** Yiyang Jiang `[一作]` (Hong Kong Polytechnic University), Li Qing `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了无词典（gloss-free）手语翻译，提出了跨模态潜在思考框架，使用潜在链式思考和计划-先检索解码。

**💡 创新点**

创新点在于引入有序潜在思考槽、计划先检索的双流解码、结构化路由与正则化机制，使得模型能够在无词典条件下实现可追踪的语义推理与证据对齐。

**🔧 技术方法**

采用端到端 Transformer + Conformer 编码器、潜在链式思考模块（含 Sinkhorn 路由）、计划-先检索双流解码，并加入单调性与连续性正则化。

**📊 数据集**

使用了大规模粤语手语数据集 LC‑HKSLT（1,311 小时），以及 PHOENIX‑2014T、CSL‑Daily、How2Sign、OpenASL 等公开基准。

**📈 对比分析**

与多种现有无词典手语翻译方法对比，实验显示在五个基准上均实现了 SOTA（如 PHOENIX‑2014T BLEU‑4 27.22，ROUGE 54.50，LC‑HKSLT BLEU‑4 21.15，ROUGE 47.87 等），尤其在大规模数据集上提升显著。

**⚠️ 局限性**

限制在于关键的“思考”过程仍为隐式潜在状态，缺乏可解释的中间推理步骤，难以直接检查、验证和控制模型的推理过程。

---

## 536. Reed--Muller Codes Achieve the Symmetric Capacity on Finite-State Channels

**arXiv ID:** 2604.15295 | [PDF](https://arxiv.org/pdf/2604.15295v1)

**作者:** Henry D. Pfister `[一作]` (Duke University), Galen Reeves `[通讯]` (Duke University)

**通讯引用:** 1010 | [OpenAlex ID](https://openalex.org/A5084396713)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

证明二进制RM码在有限状态通道（FSC）上可达到对称容量。

**💡 创新点**

将容量‑通过‑对称性定理与块级交错、随机打乱相结合，构造出适用于有记忆通道的容量证明。

**🔧 技术方法**

利用对称性定理、块化与去采样的通道变换、随机仿射打乱以及RM码的隐式交错和凿开性质。

**📊 数据集**

无具体数据集，研究基于信息理论的证明。

**📈 对比分析**

通过理论分析证明在满足条件下RM码的符号误码率可趋于零，达到FSC的对称容量；无实验对比。

**⚠️ 局限性**

需要随机打乱与MAP解码；尚未得到块误码率结果；不适用于受约束输入或非二进制通道的直接实现。

---

