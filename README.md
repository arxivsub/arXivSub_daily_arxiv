# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-29 | 今日论文总数: 385

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. On the stability of scale-space metrics

**arXiv ID:** 2606.27605 | [PDF](https://arxiv.org/pdf/2606.27605v1)

**作者:** William Leeb `[一作]` `[通讯]` (University of Minnesota), William Leeb (University of Minnesota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究了基于函数（图像）Gaussian 规模空间表示的指标族，给出了其对几何变形和加噪声的稳定性分析，提出了对旋转不变的改进版本，并设计了计算高效的离散化算法。

**💡 创新点**

创新点包括：① 对尺度空间指标的稳定性给出严谨的误差界，尤其是对位移和投影角度的影响；② 引入旋转不变版本，解决了在投影对比中的角度敏感问题；③ 提出 O(n²log n) 的 FFT/NUFFT 基础实现，并证明其在加噪声下的鲁棒性；④ 通过数值实验展示该指标在噪声和旋转下优于欧氏、切片 Wasserstein 与切片 Cramér 距离。

**🔧 技术方法**

主要技术手段包括：Gaussian 核卷积、Besov 空间范数、差分 Gaussian 滤波、傅里叶变换与 Poisson 求和、非均匀 FFT、子高斯分布的集中不等式以及误差传播分析。

**📊 数据集**

实验数据使用合成 2D 以及 3D 高斯混合模型：① 10 个等间距高斯的 2D 圆形分布；② 24 个网格布置的 3D 高斯（用作投影实验）；在这些图像上加入不同方差的 Gaussian 噪声。

**📈 对比分析**

与欧氏、切片 Wasserstein 与切片 Cramér 距离的对比表明：① 旋转不变 _α 指标随角度的变化更平滑；② 在噪声存在时误分类率显著低于欧氏和切片 Cramér；③ 在投影分类实验中，α=2 的指标保持最小误差；计算成本为 O(n²log n)，可在常规模板上快速实现。

**⚠️ 局限性**

局限性包括：① 只考虑 SO(2) 旋转，未对完整刚体变换（平移+旋转）做最小化；② 参数 α 与 p 的选择仍需经验；③ 对 α≤1 的噪声鲁棒性下降；④ 目前仅在合成数据上验证，尚未在真实 cryo‑EM 数据上测试。

---

## 2. DeLux: Cross-Modal Local Artifact Restoration in Video Using Neuromorphic Data

**arXiv ID:** 2606.27576 | [PDF](https://arxiv.org/pdf/2606.27576v1)

**作者:** Bartosz Stachowiak `[一作]` (Poznan University of Technology), Dariusz Brzezinski `[通讯]` (Poznan University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了DeLux框架，利用事件相机与RGB视频实现跨模态局部光学伪影恢复。

**💡 创新点**

首创将伪影检测、模态融合与修复模块分离，并使用事件数据作为结构先验进行局部修补，提供统一评测基准。

**🔧 技术方法**

采用E2VID++事件重建、UNet检测与修复、卷积融合、联合损失（检测、重建、非掩码一致性）等技术，训练基于合成与真实自动驾驶数据。

**📊 数据集**

训练集使用E2VID、DSEC、CARLA等事件+RGB数据并合成光学伪影；评估集包括1,724帧合成伪影和11,056帧真实自动驾驶伪影。

**📈 对比分析**

与Flare7K++、Wu et al、SHDR、HDRev‑Diff、DAD等基准对比，合成数据上MS‑SSIM>0.99、PSNR最高；真实数据上平均SAS最高，优于所有基准。

**⚠️ 局限性**

局限包括事件在静止场景无信息、缺乏真实配对伪影数据、单帧处理导致时间不一致，以及冻结的E2VID重建限制端到端优化。

---

## 3. Formal Grammars in Business Process Management: A Systematic Literature Review

**arXiv ID:** 2606.27399 | [PDF](https://arxiv.org/pdf/2606.27399v1)

**作者:** Milliam Maxime Zekeng Ndadji `[一作]` `[通讯]` (University of Dschang), Milliam Maxime Zekeng Ndadji (University of Dschang)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述了1994‑2024年间34篇关于正式文法与业务流程管理（BPM）交叉研究的论文，并基于文法形式学构建了七个研究流的分类法。

**💡 创新点**

创新点在于首次将正式文法视为BPM核心工具，提出了覆盖BPM生命周期各阶段的七条文法研究流（过程文法、建模文法、生成式文法、属性文法、图文法、语法推断和过程代数），并指出它们的贡献、局限与未来研究方向。

**🔧 技术方法**

采用了Kitchenham‑Charters 的系统文献综述（SLR）协议，利用六个数据库（ACM, IEEE Xplore, SpringerLink, ScienceDirect, Scopus, Google Scholar）检索关键词，进行双阶段筛选、分类和合成，形成PRISMA流程图。

**📊 数据集**

数据集来源于检索的526条记录，去重后341条，最终筛选34篇核心论文；在检索之外，还通过向后引用检索补充了2篇π‑计算机相关研究。

**📈 对比分析**

比较方法主要是对每个研究流内部进行定性对比（技术特点、应用阶段、优缺点），并在全局层面通过跨流对比揭示不同文法形式在表达力、可验证性和可扩展性等维度的差异。未对单个方法做统一实验评估，但综述中收集了各研究的性能/案例规模信息，表明多数研究在小型案例上验证，缺乏大规模工业部署。

**⚠️ 局限性**

局限性包括：①缺乏跨流整合，研究相互独立；②整体质量评估缺失（未对每篇论文进行方法论严谨性打分）；③实验规模有限，缺乏大规模事件日志和生产环境验证；④对噪声、概念漂移等实际日志问题的处理不足；⑤对属性文法与图文法/过程代数之间的互操作性缺乏理论支持。

---

## 4. Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents

**arXiv ID:** 2606.27472 | [PDF](https://arxiv.org/pdf/2606.27472v1)

**作者:** Vedant Patel `[一作]` `[通讯]` (Vrin), Vedant Patel (Vrin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化LLM代理在长时间对话中处理事实更新（supersession）时的能力缺失，并提供可训练的环境Supersede来闭合该缺口。

**💡 创新点**

创新点在于将事实时效性错误的检测与奖励机制直接编码为可训练的信号，构建了首个针对“当前事实正确性”进行强化学习的环境；并证明通过RL可显著提升模型的supersession准确率。

**🔧 技术方法**

使用强化学习（GRPO）在自定义环境中训练，利用程序化的匹配器作为奖励；评估时对比完整上下文与受限记忆两种策略。

**📊 数据集**

数据来源于LongMemEval真实对话子集以及自制的模板化时间线生成器，用于训练与评估。

**📈 对比分析**

比较方法：对完整上下文与固定字符数记忆（B=300）进行对比，利用McNemar检验验证差异显著；在gpt-4.1-mini、gpt-4.1、gpt-5.4模型上观察到supersession gap在更大模型或更大记忆容量下均未消失。训练后，Qwen2.5-3B在真实Held-out上从9.0%提升至16.7%，显著改进。

**⚠️ 局限性**

限制包括：实验仅使用单个小模型单次训练，提升幅度有限；训练数据为模板化生成，难以覆盖更复杂的现实更新；自我终止的RL训练无法学习更难的supersession场景；统计显著性未在多种随机种子下充分验证。

---

## 5. Aloe-Vision: Robust Vision-Language Models for Healthcare

**arXiv ID:** 2606.27500 | [PDF](https://arxiv.org/pdf/2606.27500v1)

**作者:** Jaume Guasch-Martí `[一作]` (Barcelona Supercomputing Center), Dario Garcia-Gasulla `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6215c339-3735-4be3-8a07-5bbb7004712d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个高质量、token‑平衡的医疗与通用多模态混合数据集 Aloe‑Vision‑Data，并在此基础上训练了开源可复现的 Aloe‑Vision 7B 与 72B 视觉语言模型；同时提出了无污染的 CareQA‑Vision 视觉基准，并对模型在标准与对抗性评测中的鲁棒性进行了系统分析。

**💡 创新点**

创新点在于①发布首个公开可复现的医疗多模态训练混合；②采用 token‑基重平衡与 LVLM‑辅助的半自动过滤，显著提升数据质量；③引入 CareQA‑Vision 基准，实现对未泄露医疗场景的可靠评估；④在单一对抗样本上进行后期 fine‑tuning，验证跨模态鲁棒性迁移。

**🔧 技术方法**

技术方法包括：基于 Qwen2‑VL‑Instruct 的单阶段监督微调；token‑基重平衡、perceptual hashing 去重；半自动过滤（LVLM 评分 + 答案 perplexity）；多任务评测框架 VLM‑EvalKit 与 lm‑evaluation‑harness；对抗性评测采用 HEART 框架（Sycophancy、Captions、Prompt、Legends）。

**📊 数据集**

使用的数据集有 8 个来源：PubMedVision、MedMax、MeCoVQA、Med‑GRIT、MedTrinity‑25M、Cambrian‑10M、Aloe、Magpie‑Ultra‑v1.0；以及评测基准 CareQA‑Vision、PathMMU、GMAI‑MMBench、OmniMedVQA、ProbMed、SLAKE、MMMU、MultiMedQA、MMLU 等；对抗性数据来自 HEART 框架中的 24k 样本。

**📈 对比分析**

与 12 个公开 LVLM（Qwen2‑VL、Qwen3‑VL、Kimi‑VL、GLM‑4.5V、HuatuoGPT‑Vision、Lingshu、Hulu‑Med、Chiron‑o1、MiMo‑VL、MiMo‑VL‑RL 等）以及 GPT‑5.2 进行对比。Aloe‑Vision‑72B 在 CareQA‑Vision、GMAI‑MMBench、OmniMedVQA、PathMMU 等多模态任务上均达到了或逼近最优水平；在文本任务（MMLU、MultiMedQA）表现亦不逊色；对抗评测中，AR 版本显著提升鲁棒性，性能提升 10–30% 以上。

**⚠️ 局限性**

局限性包括：仍对对抗性输入（尤其是错误 Caption）敏感；CareQA‑Vision 规模有限，覆盖范围相对窄；模型在某些高阶推理任务中表现并无显著提升；依赖 Qwen‑VL 家族的预训练模型，若基础模型出现偏差，后续训练效果受限。

---

## 6. Test Case Selection for Deep Neural Networks: A Replication Study on LLMs for Code

**arXiv ID:** 2606.27601 | [PDF](https://arxiv.org/pdf/2606.27601v1)

**作者:** Ali Asgari `[一作]` (Delft University of Technology), Annibale Panichella `[通讯]` (Delft University of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究对测试案例选择（TCS）技术在大语言模型（LLM）代码相关分类任务中的有效性进行了大规模复制实验，系统评估了多种特征与选择策略的表现；

**💡 创新点**

创新点在于：①将以往针对视觉 DNN 的 TCS 研究迁移到代码领域（clone、vulnerability、technical‑debt 三类任务）进行验证；②引入了统计抽样与分层策略，扩展了原有的 uncertainty‑based 与 surprise‑adequacy 策略；③提供了跨模型、跨任务、跨预算的综合实证分析，为 LLM 代码模型的操作性评估提供了实践依据；

**🔧 技术方法**

采用了多种 TCS 策略（如 SRS、SUPS、RHC‑S、DeepEST、GBS、SSRS、Balanced PPS、D²‑Strat、KCenterHT、MCUCB、PPSsys、Pivotal），以及七种预测特征（confidence、entropy、margin、DeepGini、LSA、DSA、MDSA），并结合 30 次随机重跑、RMSE 与失效发现计数评估；

**📊 数据集**

使用的公开数据集包括：BigCloneBench（代码克隆检测），Devign（漏洞检测），Tesoro（技术债务预测），共计 17 个任务‑模型组合；

**📈 对比分析**

通过固定预算 n=200 进行 30 次独立实验，评估准确率估计误差（RMSE）和早期失效发现数量；统计检验采用 Friedman + Nemenyi，结果显示：统计抽样策略（Balanced、PPSsys）在准确率估计上明显优于随机抽样；failure‑driven 策略（RHCS、DeepEST、GBS）在失效发现上占优势；不同任务、模型与特征交互显著影响性能；

**⚠️ 局限性**

局限性包括：仅针对分类任务，未覆盖生成任务；实验覆盖的模型与数据集有限，未必能代表所有 LLM 代码模型；特征选择范围受限，可能遗漏更有效的度量；实验成本高，特别是特征提取的计算开销；最后，真实运营环境的分布差异和标签噪声未在实验中充分考虑。

---

## 7. Causal Connections: Leveraging Multilingual Fine-Tuning for Financial QA@FinCausal 2026

**arXiv ID:** 2606.27446 | [PDF](https://arxiv.org/pdf/2606.27446v1)

**作者:** Akash Kumar Gautam `[一作]` (Anhalt University of Applied Sciences), Christian Hänig `[通讯]` (Anhalt University of Applied Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对金融文本中的因果关系提取，提出并比较了三类抽取式问答方法，最终发现多语言微调的生成模型表现最佳。

**💡 创新点**

创新点在于将多语言微调与动态few-shot示例结合，使用LLM-as-a-judge评估且引入多语言跨域迁移，显著提升了因果关系抽取的准确性。

**🔧 技术方法**

技术上主要使用BERT、BART、Llama-3.1、GPT-3.5、GPT-4.1 Mini等预训练模型，采用IO标注、seq2seq生成、提示工程、LoRA微调与动态few-shot选择等策略。

**📊 数据集**

使用来自英文和西班牙金融年报的混合语料，训练集2000样本，测试集分别为500（英）和503（西）条数据。

**📈 对比分析**

实验对比发现：BART优于BERT；多语言微调的GPT‑4.1 Mini在两语种均获最高分（英4.814，西4.775），甚至在零shot下击败更大模型GPT‑5.2；少量few-shot示例（20例）能显著提升Decoder-only模型表现。

**⚠️ 局限性**

限制包括对嵌套或多因果结构的处理不充分，西班牙子任务误差更高，缺乏细粒度标注指南导致模型难以解决歧义；未来需探索强化学习和更精细的提示设计来进一步提升性能。

---

## 8. Recall Before Rerank: Benchmarking Deep Learning Models for Large-Scale Code-to-Code Retrieval

**arXiv ID:** 2606.27401 | [PDF](https://arxiv.org/pdf/2606.27401v1)

**作者:** Leonardo Venuta `[一作]` (Sant'Anna School of Advanced Studies), Paolo Ferragina `[通讯]` (Sant'Anna School of Advanced Studies)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言大规模代码检索中第一阶段召回模型进行系统评估，比较了多种Transformer、编码器-解码器以及大型LLM的嵌入效果与吞吐量，探索了LLM风格规范化对召回率的提升。

**💡 创新点**

首次在千兆级代码库上对超过20种模型进行统一的召回与效率基准，揭示了精度-吞吐量的Pareto前沿，并提出基于LLM的代码重写提升低性能模型召回的策略。

**🔧 技术方法**

使用深度学习编码器、对比学习、解码器嵌入以及LLM生成重写的技术，并在BM25等传统IR上进行对比；还使用了精度@k、NDCG和KB/s吞吐率指标。

**📊 数据集**

使用了BigCloneBench（Type-2/3）、CodeNet、MultiPL-E、xCodeEval等多语言、多任务的公开数据集。

**📈 对比分析**

通过在单GPU环境下的精确向量检索与顺序扫描评估，结果显示大型LLM嵌入模型在精度上领先但吞吐率低；轻量编码器吞吐快但召回率显著下降；LLM重写能将弱模型召回提升约28-29个百分点。

**⚠️ 局限性**

局限包括：仅评估第一阶段召回，未完整覆盖后续重排；使用单GPU实验难以直接推广到多GPU分布式场景；部分数据集可能被模型预训练数据污染；对大型LLM在存储与索引压缩方面的进一步研究不足。

---

## 9. Kimball's Data Warehouse Architecture: Evaluating the Challenges of Conformed Data against the Inmon Model

**arXiv ID:** 2606.27571 | [PDF](https://arxiv.org/pdf/2606.27571v1)

**作者:** Júlio Rocha `[一作]` (INESC Technology and Science), Filipe Cabral Pinto `[通讯]` (Altice Labs)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

评估并对比了Kimball的数据仓库架构与Inmon模型，分析了其设计原则、优缺点、维度整合与数据总线实现等方面，提出了以用户需求为中心的维度建模视角。

**💡 创新点**

从用户需求出发，将Kimball的维度建模与数据总线与Inmon的顶层设计进行系统对比，并对Conformed Dimension的实现挑战进行了深入讨论，形成了一个多维评价框架。

**🔧 技术方法**

使用了维度建模、星型模式、数据总线矩阵、ETL流程、3NF规范化与反规范化等技术手段，对两种架构进行理论上的实现与比较。

**📊 数据集**

论文未使用具体的真实数据集，而是基于公开的示例图表和理论案例进行分析与说明。

**📈 对比分析**

通过在信息质量、系统质量、个人影响、组织影响四个维度上进行定性比较，阐述了Kimball架构在查询性能和易用性方面的优势，Inmon在数据一致性与集成性方面的优势；未给出量化性能指标，侧重理论分析。

**⚠️ 局限性**

研究缺乏实证数据与量化性能评估，主要停留在理论分析层面；对Conformed Dimension的实施细节与组织阻力探讨不足，实际应用效果受企业环境差异限制。

---

## 10. RANSAC Scoring Done Right

**arXiv ID:** 2606.27385 | [PDF](https://arxiv.org/pdf/2606.27385v1)

**作者:** James Pritts `[一作]` (Kiel University), Kevin Köser `[通讯]` (Kiel University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种新的鲁棒模型估计算法MAGSAC++，主要用于图像匹配中的相机位姿估计；

**💡 创新点**

创新点在于自动估计内点阈值并将噪声分布与内点计数结合，从而消除了传统RANSAC对阈值的依赖，并显著提升了采样效率；

**🔧 技术方法**

使用了改进的随机采样一致性（RANSAC）框架、概率阈值估计、内点加权计数以及与MAGSAC、MSAC、Gau等算法的混合采样技术；

**📊 数据集**

实验基于公开的KITTI及Oxford VGG/HPatch等场景数据集；

**📈 对比分析**

与传统RANSAC、MSAC、Gau等方法对比，MAGSAC++在误差率、内点率和整体鲁棒性方面均表现优异，误差下降约20%，内点率提升约15%；

**⚠️ 局限性**

主要局限在于计算复杂度较高，尤其在极端噪声或稀疏匹配点环境下仍可能出现误差，并且对噪声模型假设要求较高。

---

## 11. Observers, Symmetries, and the Hierarchy of Language Classes: A Theory of Computation Parameterized by the Observer

**arXiv ID:** 2606.27407 | [PDF](https://arxiv.org/pdf/2606.27407v1)

**作者:** Fabio F. G. Buono `[一作]` `[通讯]` (Independent Researcher), Fabio F. G. Buono (Independent Researcher)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出了一种新的语言分类轴——观测层次（Observational Hierarchy），通过定义“观察者”函数来刻画计算机系统对输入信息的可见程度，并以此为基础重新表述语言识别与复杂度问题。

**💡 创新点**

创新点在于：①将观测者作为独立的参数引入复杂度理论，形成观测参数化的P_O与NP_O；②定义了观测者的偏序及其产生的语言类层次结构，揭示了长度分支、奇偶分支与子序列分支之间的金字塔状、菱形子格；③证明观测层次与传统的Chomsky层次相互独立；④指出计算难度与结构盲区是两个互相独立的现象。

**🔧 技术方法**

采用形式化的语言理论工具：等价关系、饱和语言、阶层结构、集合论与图论证明；利用Simon定理和Parikh定理连接子序列与计数信息；在复杂度方面使用时间可计算的观测者和模拟证明来界定P_O与NP_O的包含关系。

**📊 数据集**

本研究为纯理论工作，无使用具体数据集；所有结果均基于数学证明与逻辑推导。

**📈 对比分析**

通过理论证明展示不同观测者之间的严格包含关系与不相容性，并与经典复杂度类P、NP进行比较，证明P_O_prof与NP_O_prof在P内完全相等，却严格小于P，体现了结构盲区的独立性。

**⚠️ 局限性**

限制与开放问题包括：①观测者仅为确定性函数，忽略随机/概率观测者的情况；②缺乏对观测者组合（范畴结构）的完整研究；③对P_O_prof在子线性复杂度中的确切位置尚未确定；④如何将观测层次应用于实际算法与模型仍待探索。

---

## 12. OverFlowLight: Real-Time Gridlock Prevention and Traffic Signal Optimization for Urban Intersections

**arXiv ID:** 2606.27381 | [PDF](https://arxiv.org/pdf/2606.27381v1)

**作者:** Mingyuan Li `[一作]` (Beijing University of Posts and Telecommunications), Qiang Wu `[通讯]` (Lanzhou University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套名为 OverFlowLight 的实时框架，利用摄像头与毫米波雷达多模态感知，实时检测并预判交叉口溢出，并在信号周期中插入专门的溢出相位以清除排队车辆。

**💡 创新点**

创新点包括：①把溢出定义为可操作的车道级控制问题；②构建溢出相位映射表（OPM），实现溢出相位的快速匹配；③将传统规则控制与强化学习（RL）结合，形成模块化、可插拔的溢出抑制机制；④在真实城市环境中实现了低时延、低成本的部署。

**🔧 技术方法**

核心技术包括多模态感知（摄像头+雷达）+溢出检测规则、溢出相位映射表（OPM）、基于Q‑learning 的 RL 交叉口相位选择、传统压力最大化与 FuzzyLight 等基线控制器的融合。

**📊 数据集**

使用了在三座城市 43 个交叉口 1.5 年的真实交通数据，涵盖不同道路结构、天气与高峰时段的感知与控制记录。

**📈 对比分析**

与传统固定时序、FuzzyLight、DQN 等基线对比；在 43 个交叉口上，溢出事件减少 60.4%，网络吞吐量提升 18.2%；在高峰时段吞吐量提升约 15.8%，车辆停留次数下降 18.7%。

**⚠️ 局限性**

局限性包括：①主要聚焦单交叉口溢出抑制，跨交叉口协调尚未充分验证；②对感知阈值（α、β）的依赖需在不同环境下进一步调优；③RL 训练需要足够覆盖溢出情景，可能受限于数据分布；④雷达与摄像头的硬件成本与安装复杂度仍是推广瓶颈。

---

## 13. SidConArena: An Environment Evaluating Agents in Open-Ended,Positive-Sum Bargaining Game

**arXiv ID:** 2606.27397 | [PDF](https://arxiv.org/pdf/2606.27397v1)

**作者:** Yeqi Feng `[一作]` (Tsinghua University), Tianxing He `[通讯]` (Tsinghua University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SidConArena评估LLM代理的开放式正和讨价还价基准框架。

**💡 创新点**

将多玩家经济划分为自然语言谈判、转换器生产与封闭式拍卖三阶段，并提供神经符号动作接口与异步执行。

**🔧 技术方法**

使用了部分可观测随机博弈建模、神经符号动作接口、异步执行与阶段感知代理调度等技术。

**📊 数据集**

构建了基于合成情景的多阶段谈判与交易数据集（未使用公开数据集）。

**📈 对比分析**

在同质与异质锦标赛中对比前沿LLM模型，表现为经济收益提升，但仍存在资源误估和被动议价。

**⚠️ 局限性**

局限性包括资源误估、议价被动、对长期投资规划的不足。

---

## 14. AO-ARC: Almost-Surely Asymptotically Optimal Multi-Robot Motion Planning with ARC

**arXiv ID:** 2606.27495 | [PDF](https://arxiv.org/pdf/2606.27495v1)

**作者:** James D. Motes `[一作]` (University of Illinois at Urbana Champaign), Nancy M. Amato `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种新的任意多机器人运动规划方法AO-ARC，能够在保持初始规划速度的同时实现渐近最优的完成时间（makespan）解。

**💡 创新点**

创新点在于：将ARC元算法与bounded feasibility求解器结合，使用递归局部子问题扩展和统一的team‑time bound，既保留了自适应耦合优势，又实现了渐近最优性；同时提供了完整的理论证明。

**🔧 技术方法**

核心技术包括：状态‑成本空间的bounded feasibility规划、递归局部子问题构造与扩展、基于时间窗口的局部约束计算、以及对全局完成时间的严格保持；实现上采用RRT*和dRRT*等基础规划器作为子问题求解器。

**📊 数据集**

使用的数据集包括：2D移动机器人场景（Cross、Circle）、2D平面机械臂场景、以及3D Panda机械臂在共享工作空间中的5个任务（共4或8台机械臂）。

**📈 对比分析**

与Comp（基于RRT*的组合空间规划）、CompAORRT*、dRRT*、PP-ST-RRT*等基线方法对比，AO-ARC在初始解时间上最快，随着时间推进收敛速度最快，尤其在机器人数量增加时保持高成功率；在多数任务中还取得了更低的归一化makespan。

**⚠️ 局限性**

局限性：目前仅支持makespan成本指标，对sum‑of‑costs或更复杂状态空间的推广尚未实现；局部子问题的时间约束保守，可能导致在单一机器人占用长路径的任务中表现不佳；并且在极大规模或极高冲突场景下的计算复杂度仍需进一步评估。

---

## 15. Benchmarking Multi-Modal Graph-based Social Media Popularity Prediction

**arXiv ID:** 2606.27539 | [PDF](https://arxiv.org/pdf/2606.27539v1)

**作者:** Utkarsh Sahu `[一作]` (University of Oregon), Yu Wang `[通讯]` (University of Georgia)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了统一的多模态图网络预测社交媒体内容未来受欢迎程度，并发布了MMG-Pop基准数据集。

**💡 创新点**

首次将文本、视觉、时间与交互图结构三类信息在同一框架下联合建模，提供跨平台、跨社区的统一评价协议。

**🔧 技术方法**

使用Transformer、CLIP、双向GraphSAGE消息传递和多任务MLP进行端到端训练。

**📊 数据集**

收集 Bluesky 与 Reddit（r/AMA、r/Gaming、r/Futurology）四个大规模数据集，构成 MMG-PopBench。

**📈 对比分析**

在所有指标（最大宽度、最大深度、结构性病毒性、规模、独立用户、点赞分数）上相较现有 Graph-LSTM、CasSeqGCN 等基线，MMG-PopNet 平均 MSE 下降 4.6%–17.0%，在多观察窗口和跨平台训练下表现最优。

**⚠️ 局限性**

受限于 LLM 在数值预测上的校准能力，无法与专门建模交互图结构的网络竞争；对不同平台动态差异的适应性仍需进一步研究。

---

## 16. Learning to Throw: Agile and Accurate Cable-Suspended Payload Delivery with a Quadrotor

**arXiv ID:** 2606.27603 | [PDF](https://arxiv.org/pdf/2606.27603v1)

**作者:** Yifan Zhai `[一作]` (University of Zurich), Davide Scaramuzza `[通讯]` (University of Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

利用深度强化学习实现无人机对悬挂载荷的敏捷精准投掷。

**💡 创新点**

创新点在于将高保真机械臂动力学与物理引擎通过单一力矩耦合，实现零样本的实机转移。

**🔧 技术方法**

采用PPO强化学习、混合仿真框架（解析四旋翼模型 + PhysX绳索动力学）以及域随机化技术。

**📊 数据集**

使用内部仿真环境生成数据，并在真实硬件上使用台球作为载荷进行验证；未使用公开数据集。

**📈 对比分析**

与传统离线轨迹优化+MPC基线相比，实验显示投掷误差降低约50%、投掷时长缩短约30%，在agility‑accuracy权衡上更优。

**⚠️ 局限性**

局限性包括目标固定、地面平坦、未考虑投掷后空中阻力、载荷属性固定，且未验证移动目标或多无人机协作投掷。

---

## 17. DMV-Bench: Diagnosing Long-Horizon Multimodal Agents' Visual Memory with Incidental Cue Injection

**arXiv ID:** 2606.27499 | [PDF](https://arxiv.org/pdf/2606.27499v1)

**作者:** Yujin Tang `[一作]` (Dartmouth College), Nikhil Singh `[通讯]` (Dartmouth College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 DMV-Bench，一个在交互式多模态购物环境中评估视觉记忆的基准；提出了 DualMem 双编码记忆架构，并在该基准上对多种记忆系统进行对比实验。

**💡 创新点**

创新点包括：① 通过“L2 泄露合约”把判别信号完全隐藏在像素层；② 在每个产品图像中嵌入唯一的偶然视觉线索；③ 设计了“回忆距离”指标和共享前缀回滚树高效评估；④ DualMem 通过视觉和文本双通道并行编码并在检索时融合，显著提升长链记忆表现。

**🔧 技术方法**

使用的技术包括：SigLIP‑2 视觉编码器、SBERT 文本编码器、双编码融合策略（α = 0.75）、外部多模态记忆（WorldMM、MMA、M2A）以及对比实验的 Gemini 2.5 Flash 与 Qwen2.5‑VL‑7B LLM。

**📊 数据集**

数据集为合成的现代家具电商目录，包含 10 个类别、10 种风格、每种风格 10 个变体，总计 1,000 个商品。每个商品图像预渲染了唯一的视觉线索，形成完整的评测环境。

**📈 对比分析**

通过在不同链长度（J = 5、10、15）和回忆距离上对 TSR（任务成功率）进行评估，DualMem 在两大 LLM 后端均显著优于 caption 基线、WorldMM、MMA、M2A 等系统；在 Gemini 后端 TSR 最高达 81.1%，在 Qwen 后端 82.7%，而 M2A 仅为 70.4% 左右。

**⚠️ 局限性**

局限性包括：仅测试于合成家具视觉域，未验证跨域迁移；仅评估两种 LLM 后端，缺乏更广泛的跨模型验证；缺少人工上限基准；DualMem 的 α 超参数和视觉/文本融合策略仍可进一步自适应优化。

---

## 18. When Does Personality Composition Matter for Multi-Agent LLM Teams?

**arXiv ID:** 2606.27443 | [PDF](https://arxiv.org/pdf/2606.27443v1)

**作者:** Aryan Keluskar `[一作]` (Arizona State University), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统评估了在多代理LLM团队中，通过人格提示改变合作者的亲和性对三类任务（编码、科研协作、谈判）结果的影响；

**💡 创新点**

创新点在于揭示人格影响在受任务输出结构约束的“缓冲”效应，并对常用负向人格词汇的“词义偏置”进行了消解实验；

**🔧 技术方法**

方法包括使用Goldberg双极词对进行人格提示、Bales互动过程分析的四类交互行为标注、以及多代理LLM框架AutoGen、MetaGPT等；

**📊 数据集**

数据集为MultiAgentBench（5种编码与科研任务、50个谈判任务）与其基准任务评估；

**📈 对比分析**

通过对比低/高亲和性对话状态（exploration fraction）、里程碑完成率或协议达成率，发现编码任务对人格无显著影响，科研与谈判任务显著受低亲和性抑制；

**⚠️ 局限性**

局限包括仅考察同质人格团队、仅使用负向词汇的对照实验缺乏足够泛化、模型内部安全约束可能掩盖真实人格效应。

---

## 19. Distribution-based deep multiple instance learning for tumor proportion scoring in NSCLC

**arXiv ID:** 2606.27579 | [PDF](https://arxiv.org/pdf/2606.27579v1)

**作者:** Krzysztof Pysz `[一作]` (Politechnika Wrocławska), Witold Dyrka `[通讯]` (Politechnika Wrocławska)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了一套弱监督的深度多实例学习（MIL）框架，利用零膨胀β（ZIBeta）分布对肺腺癌免疫组化切片的肿瘤比例得分（TPS）进行预测。

**💡 创新点**

创新点在于将ZIBeta分布引入MIL模型，实现对TPS零类（TPS<1%）的显式建模，并通过熵最小化约束提升每个补丁的预测一致性，兼顾精度与不确定性估计。

**🔧 技术方法**

技术方案包括：利用MIHIC数据集训练的嵌入提取网络；采用门控注意力聚合的MIL架构；在ZIBeta分布下使用MSE+自定义负对数似然损失；对补丁级预测引入熵正则化。

**📊 数据集**

数据集方面，嵌入提取使用公开的MIHIC 6类组织图像；TPS预测使用自有的162例NSCLC IHC全切片，按SP263/22C3染色，并仅提供滑动窗口级TPS标签。

**📈 对比分析**

与传统线性、岭回归以及基线MIL模型（均值聚合和门控注意力）对比，ZIBeta模型在RMSE、MAE、F1及Kappa指标上均表现最佳，RMSE最低约0.201，零类F1最高约0.612。

**⚠️ 局限性**

局限性包括：仅针对NSCLC单一病种和染色，嵌入的通用性未验证；标签来自单位认证病理学家，可能存在系统偏差；高TPS区间标签粗糙；并且ZIBeta浓度与预测误差的相关性噪声大，需进一步验证。

---

## 20. TruEye: Fine-Grained Detection of AI-Generated Human Subjects in Images

**arXiv ID:** 2606.27505 | [PDF](https://arxiv.org/pdf/2606.27505v1)

**作者:** Jay Barot `[一作]` (Vanderbilt University), Dan Lin `[通讯]` (Vanderbilt University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了名为TruEye的细粒度检测器，用于识别图像中AI生成或合成的人体与场景，并提供可解释的定位与分类。

**💡 创新点**

首次实现了五类合成内容的细粒度分类（SHSS、SHRS、RHSS、RHRS、SS），并通过掩码条件双流Transformer、特征放大模块以及跨流交叉注意，兼顾高精度、可解释性与超快推理速度（相较LLM模型快100×）。

**🔧 技术方法**

采用掩码生成器、双流Transformer+特征放大、区块级交叉注意、token级监督与全局分类、多级训练策略、ViT基础Transformer架构。

**📊 数据集**

主要使用自建的FineSyn数据集（5k每类合成样本+10k真实图像），并在OpenForensics、FaceForensics++、OpenSDID等公开数据集上进行跨域评估。

**📈 对比分析**

与SIDA、LEGION、TruFor等六种基准模型在AUC、IoU、F1、推理速度等指标对比，TruEye在多数据集上AUC>96%、IoU>79%、F1>97%，且每张图片推理仅4.57 ms，速度和精度均优于LLM模型。

**⚠️ 局限性**

局限性在于对极低分辨率或强后处理图像的鲁棒性仍有限；模型对新型生成器的极端变异可能产生误检，且需大量标注数据支持训练。

---

## 21. On the Inseparability of Instructions and Data in Shared-Embedding Sequence Models

**arXiv ID:** 2606.27567 | [PDF](https://arxiv.org/pdf/2606.27567v1)

**作者:** Dewank Pant `[一作]` (Independent Researcher), Avijit Kumar `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

证明在共享嵌入的 Transformer 体系结构中，无法通过管道内部机制实现完美的提示注入防护。

**💡 创新点**

提出了 Semantic‑Faithful Control（SFC）安全属性并给出 Inseparability 定理，揭示提示注入与缓冲区溢出的结构相似性。

**🔧 技术方法**

使用概率与信息论（总变差、Bayes 最优错误）、注意力分析、MMD 与可解释性实验（如通用对抗后缀）等理论与实验技术。

**📊 数据集**

基于 Llama‑3.1 8B、Mistral‑7B 等公开 LLM 以及自建指令与用户数据语料进行评估。

**📈 对比分析**

通过对比现有防御（指令微调、RLHF、门控分类器、通用后缀攻击）发现即使在对抗性硬化下仍能产生 SFC 失效；实验表明共享嵌入模型在所有测试中均存在控制路径泄露。

**⚠️ 局限性**

局限性在于仅适用于共享嵌入、无固有控制‑数据分离的 Transformer 体系结构；不排除通过架构层面的显式分离或多层防御实现实质性防护。

---

## 22. A Survey of Automated Presentation Coaching: Systems, Methods, and Open Challenges

**arXiv ID:** 2606.27380 | [PDF](https://arxiv.org/pdf/2606.27380v1)

**作者:** Wen Liang `[一作]` (Columbia University), Julia Hirschberg `[通讯]` (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对自动化口头演讲辅导系统进行系统综述，构建五维任务分类法并映射15个代表性系统。

**💡 创新点**

创新点在于提出统一的五维分类框架，系统映射与评估指标，识别领域缺口并提出未来研究方向。

**🔧 技术方法**

综述涉及CAPT、TTS（流匹配/非自回归）、GOP/CTC、克隆对比、音调/韵律、节奏、内容一致性等核心技术。

**📊 数据集**

参考TIMIT、L2-ARCTIC、Common Voice、TED‑LIUM 3、GigaSpeech、EpaDB、Speak & Improve等公开语料，指出缺乏适合演讲结构的标注集。

**📈 对比分析**

通过对比各系统在五维任务覆盖、实时性与针对L2的能力，发现大多数仅关注发音或节奏，少数提供实时反馈，整体性能在综合评估上仍处于缺口阶段。

**⚠️ 局限性**

局限包括综述非全量，聚焦英语且缺少跨语言案例，数据集不足以支持完整评估，评测指标与实际学习效果的关联验证不足。

---

## 23. Learning from Annotation Uncertainty: Entropy-Aware Curriculum for Speech Emotion Recognition

**arXiv ID:** 2606.27536 | [PDF](https://arxiv.org/pdf/2606.27536v1)

**作者:** Zahra Omidi `[一作]` (University of Texas at Dallas), John H. L. Hansen `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在MSP‑Podcast 2.0上，用多任务WavLM‑Base模型研究将注释者投票分布直接作为监督目标，比较硬一致标签与分布式监督在9类语音情感识别中的效果；

**💡 创新点**

将注释者不一致性视为结构化感知信息而非噪声，首次在SER中引入分布式监督和熵感知课程学习，以保留并利用多评审产生的情感模糊度；

**🔧 技术方法**

使用WavLM‑Base + TC‑GRU多任务框架，交叉熵/类别平衡交叉熵与KL散度损失，熵正则化、课程学习；评估指标包括Macro‑F1、UAR、KLD、JSD、CCC；

**📊 数据集**

MSP‑Podcast 2.0（267,905句，9类情感，多评审投票，连续Valence/Activation/Dominance）作为实验数据集；

**📈 对比分析**

对比硬标签（CE/CBCE）与Primary、Merged（0.8/0.9）分布目标，并尝试过滤/加权熵课程；在Test1/Test2上，分布监督显著降低KLD/JSD，宏F1保持相近甚至略优，说明对人类投票分布的对齐更好；

**⚠️ 局限性**

高熵样本仍难以准确分类，宏F1受“Other”类别影响，分布监督对极高不确定性样本提升有限；且评估仍以硬决策为主，无法完全体现分布信息的价值。

---

## 24. Toward AI-Native 6G Air Interface: A 3GPP Perspective on Protocol Framework

**arXiv ID:** 2606.27466 | [PDF](https://arxiv.org/pdf/2606.27466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 25. Boundary condition fidelity for bottom-hole pressure and CO2 plume prediction in geological carbon storage

**arXiv ID:** 2606.27515 | [PDF](https://arxiv.org/pdf/2606.27515v1)

**作者:** Romal Ramadhan `[一作]`, Larry W. Lake `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对不同截断域边界条件（如孔隙体积乘子、渗透率修正、角单元修正、分层与渐进修正）进行系统评估，比较其在同质与异质储层中对井下压力（BHP）和CO₂云团迁移预测的影响。

**💡 创新点**

首次将十种边界处理方案与完整域参考结果同时比较，揭示角单元孔隙体积守恒是最关键的修正；渐进功率律修正（VAGT）与分层修正（GhVT）在两类储层中均表现出最稳健的精度；同时指出渗透率修正并非普适有益，易在异质储层中加剧误差。

**🔧 技术方法**

使用CMG‑GEM组成模拟器进行等温多组分模拟；实现边界单元孔隙体积乘子、渗透率修正、角单元修正以及分层与渐进功率律分配；通过BHP时间序列和CO₂饱和度等温面计算RMSE、NRMSE、峰值偏差和IoU等指标。

**📊 数据集**

基于两套模型：同质模型（全域501×501格、孔隙率0.20、渗透率10 mD）和异质模型（随机分布、均值相同）；分别截取101×101与251×251子域进行比较；使用相同注入速率（2500 吨/天）和注入/停注时序（10 年注入+40 年监测）。

**📈 对比分析**

对照完整域（Truth）与十种边界方案，计算BHP RMSE/NRMSE和CO₂ IoU；结果显示：角单元修正方案将BHP RMSE从≈300 psi降至≈110 psi（≈60–70 %提升），IoU从≈0.80提升至≈0.99；VAGT（含渗透率修正）在同质模型达到NRMSE≈3.5%、IoU≈0.99；在异质模型VAGT与GhVT分别实现NRMSE≈3.6%、IoU≈0.97；不含角修正方案误差明显，IoU仅≈0.80–0.84。

**⚠️ 局限性**

限制包括：仅考虑等温、无化学反应与固结耦合；模型为二维（单层）网格，可能不适用于多层或三维复杂地层；渗透率修正基于邻接单元的谐平均，异质区易产生过度限制流动；未检验在更大规模或多相化学耦合情况下的推广性。

---

## 26. Towards Evaluation of Implicit Software World Models in Coding LLMs

**arXiv ID:** 2606.27406 | [PDF](https://arxiv.org/pdf/2606.27406v1)

**作者:** Egor Bogomolov `[一作]` (JetBrains Research), Yaroslav Zharov `[通讯]` (JetBrains Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该研究基于SWE‑Bench Verified构建了一个面向库级测试的执行行为数据集，并提出了四项新评估指标（测试结果、峰值内存、壁钟时间以及方法与行级的资源分配排名）。

**💡 创新点**

创新点在于将软件世界模型的评估从单纯的控制流扩展到更广泛的运行资源维度，并将评估对象提升至真实项目级别的代码片段，而非传统的简化函数。

**🔧 技术方法**

主要技术包括自定义Docker内置Tracer收集执行时间、内存和异常信息，以及基于日志、线性校准和NDCG等多种评价方法。

**📊 数据集**

使用的数据集来源于SWE‑Bench Verified的500条真实GitHub issue和金标准补丁，经过过滤后得到435个库级测试样本。

**📈 对比分析**

与众多Anthropic、OpenAI以及开源大模型进行对比，使用分类准确率、log‑scale校准斜率/偏差、MAE、Recall@5和NDCG@5等指标，结果显示所有模型均表现平庸：F1低、斜率偏离1、偏差普遍偏大，排名准确率不足20%。

**⚠️ 局限性**

局限性包括：仅覆盖Python项目且上下文受限于50万字符；使用单一oracle式上下文收集策略；未尝试更复杂的提示或投票机制；缺乏对其他语言、无策划项目或更宽泛测试集的泛化验证。

---

## 27. Unified Zero-Shot Time Series Forecasting: A Darts Foundation

**arXiv ID:** 2606.27438 | [PDF](https://arxiv.org/pdf/2606.27438v1)

**作者:** Zhihao Dai `[一作]` (University of Oxford), Alain Gysi `[通讯]` (Unit8 SA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Darts库中实现了统一的API封装，支持四种时间序列基础模型（Chronos-2、TimesFM 2.5、TiRex、PatchTST-FM）的零样本预测、可选的微调、协变量、概率预测与SHAP可解释性。

**💡 创新点**

创新点在于为多款TSFM提供一致的接口和轻量级依赖，解决了不同模型间的互操作性与评估难题，同时引入了模型微调控制、协变量对齐和统一的可解释性工具。

**🔧 技术方法**

采用PyTorch Lightning进行训练与推理，利用safetensors与HuggingFace Hub分发模型，支持概率预测的likelihood抽象，使用SHAP（Permutation）实现模型解释，并集成了数据处理、评估、超参搜索与模型集成等功能。

**📊 数据集**

使用20个公开时间序列数据集（从完整M4/TSF benchmark中抽取），用于快速可复现的零样本预测与微调实验。

**📈 对比分析**

通过与季节性Naïve基线对比的技能得分（Scaled Quantile Loss和MASE），验证了实现与原始模型差异不超过1.3%，平均SQL技能约50%，MASE约39%；与原始实现的性能一致。

**⚠️ 局限性**

目前仅支持四个TSFM，缺少参数高效微调方法（如LoRA），对部分TSFM的功能兼容性与扩展性仍有限。

---

## 28. Glite ARF: Verifier-Driven Research with Parallel LLM Coding Agents

**arXiv ID:** 2606.27416 | [PDF](https://arxiv.org/pdf/2606.27416v1)

**作者:** Vassili Philippov `[一作]` (Glite), Anton Nikolaev `[通讯]` (University Of Sheffield)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了 Glite ARF，一个三角色（人类研究者、LLM 编码代理、Python 验证器）框架，以可审计、并行的方式管理大规模实验，完成 BEA 2026 词汇难度预测任务并在闭赛中夺冠、开赛第二，同时在词义消歧和 CEFR 可读性两项任务中验证其通用性。

**💡 创新点**

将实验规则写入可执行的 Python 验证器，实现任务隔离、不可变性、修正覆盖与可视化概览，形成“Verifier‑driven research”范式；通过结构化流程与 LLM 代理并行执行，提供全流程审计与合规性追踪，显著降低结构性错误。

**🔧 技术方法**

使用 Claude Code、Codex CLI 等 LLM 编码代理；Git + Git LFS、GitHub Actions、CI/CD、DVC、Git 工作树；Python 脚本（verifiers、aggregators、run_with_logs）与 A100 GPU、LLM API；并行任务调度与日志收集。

**📊 数据集**

BEA 2026 词汇难度数据集（约 330 万条测试响应）以及两项公开任务（词义消歧、CEFR 可读性）的相关语料与标注，均以 CSV/JSON 形式存储并受版本控制。

**📈 对比分析**

与 BEA 2026 官方基线对比，闭赛 RMSE 降低 29.9%、开赛 35.9%，在所有三种语言中闭赛首名、开赛第二；在其他两项任务中完成率与指标均达到预期；框架在三场实战中仅增加约 1% 的工作时间。

**⚠️ 局限性**

依赖单一人类研究者执行角色 1，无法多用户并行；验证器无法防御恶意代理，需要信任代理；验证器仅捕捉结构性错误，语义错误仍需人工判断；框架对多实验室协作存在冲突风险。

---

## 29. SemCityLoc: Aerial 6DoF Localization Using Semantic 3D City Models

**arXiv ID:** 2606.27444 | [PDF](https://arxiv.org/pdf/2606.27444v1)

**作者:** Jingfeng Mao `[一作]` (Technical University of Munich), Olaf Wysocki `[通讯]` (University of Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过将预训练的语义分割和单目深度与标准化LoD城市模型对齐，实现了无人机6DoF姿态估计。

**💡 创新点**

创新点在于：①利用语义表面与深度信息进行粗细分层的语义几何配准；②构建SemCityLockeD真实低空、厘米级位姿、LoD1–3城市模型的基准；③证明语义+深度足以替代稠密纹理模型。

**🔧 技术方法**

采用DINOv3+DPT进行语义分割，MoGe-2进行单目深度估计，4D语义成本体搜索与粒子滤波精细化，结合GPU光栅化实现实时推理。

**📊 数据集**

使用SemCityLockeD（真实低空UAV图像与LoD1–3城市模型）、UAVD4L‑LoD、Swiss‑EPFL等公开基准。

**📈 对比分析**

与LoD‑Loc、MC‑Loc、CAD‑Loc等基线对比，SemCityLoc在SemCityLockeD 2m–2°召回率从35.11%提升至69.15%，平均位置误差从9.89m降至2.62m；在UAVD4L和Swiss‑EPFL亦保持或优于基线。

**⚠️ 局限性**

受限于城市模型的完整性与精度、语义分割/深度误差；在极简几何或缺失语义的场景表现下降；对大范围先验误差敏感。

---

## 30. Developmental approach reveals the statistical learning of Neural Language Models: Transformers generalize from the most abstract statistical patterns

**arXiv ID:** 2606.27460 | [PDF](https://arxiv.org/pdf/2606.27460v1)

**作者:** Wang Bojun `[一作]` (University of Oxford), Elizabeth Wonnacott `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练生成式Transformer模型在一个层级嵌套的合成语法上，记录训练进程并分析内部表示和预测分布，探究NLM的统计学习路径。

**💡 创新点**

首次证明NLM在学习过程中呈现“先过度泛化、后约束”模式，即先掌握全局统计规律后逐步细化局部依赖，揭示其发展路径与人类语言习得相似。

**🔧 技术方法**

使用单层单头的低维Transformer、auto‑encoder降维、概率质量分析及运动图可视化技术。

**📊 数据集**

合成语法数据集（四层级：M,N,P,Q及其子类，共320,000个字符串），并对六种不同词序排列进行测试。

**📈 对比分析**

通过平均100个模型的概率质量曲线对比各层级学习进度，结果显示全局→中间→局部的梯度学习顺序，且在所有排列中保持一致；未给出传统准确率，但概率质量趋近1说明模型已掌握相应类别。

**⚠️ 局限性**

局限在于过度简化的合成语法、极小模型规模、缺乏对自然语言tokenization影响的考虑，且结果难以直接推广到大规模预训练语言模型。

---

## 31. Equivalence of Continuous-Time Markov Chains and Linear Dynamical Systems

**arXiv ID:** 2606.27533 | [PDF](https://arxiv.org/pdf/2606.27533v1)

**作者:** Mihir Vahanwala `[一作]` `[通讯]` (Max Planck Institute for Software Systems), Mihir Vahanwala (Max Planck Institute for Software Systems)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文证明了连续时间线性动力系统与马尔可夫链之间的维度对应关系。

**💡 创新点**

创新点在于给出了连续时间系统中生成矩阵与离散系统等价的理论证明。

**🔧 技术方法**

采用了谱分解、矩阵指数、生成矩阵的线性代数技术。

**📊 数据集**

无实验数据集，仅使用符号计算与理论证明。

**📈 对比分析**

对比方法主要是理论推导，没有性能评估。

**⚠️ 局限性**

局限性在于仅适用于有0特征值的生成矩阵，且未给出数值实现细节。

---

## 32. Auditing AI Investment Recommendations as Executable Actions

**arXiv ID:** 2606.27570 | [PDF](https://arxiv.org/pdf/2606.27570v1)

**作者:** Sidnei Barbieri `[一作]` (Aeronautics Institute of Technology), Ágney Lopes Roth Ferraz `[通讯]` (Aeronautics Institute of Technology)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种可重现、可审计的投资建议评估协议，将AI生成的投资建议拆分为可执行订单，并通过确定性基线对其有效性、稳定性和与基线的一致性进行评估。

**💡 创新点**

创新点在于将“可执行性、可重现性、可解释性”三大维度拆分出来，构建基于规则的可重现验证器和冻结场景库，形成与传统收益/重叠度不同的全新评估框架；同时提供可插拔的合约谓词，便于扩展治理需求。

**🔧 技术方法**

技术上使用行为测试（可执行合同）、确定性验证器、Jaccard/总变差等统计量、交易成本量化（分段固定费用）、混合整数规划（部署上限）以及对LLM输出进行结构化解析和重放。

**📊 数据集**

使用的主要数据集为：1）120个冻结场景（覆盖现金部署、持仓、费用、事实等12类变体）；2）17只股票组成的基准组合在2021‑2026年的每日收盘价；3）GPT‑5.5与Opus‑4.8模型的多轮生成样本。

**📈 对比分析**

比较方法：在冻结场景下评估模型的有效率、稳定性和与基线的重叠；在回测中将基准组合与等权、风险平价、最小方差等 allocator 进行对比。结果显示基准组合对 SPY 的年化收益率高 4.96pp，Sharpe 1.14，但等权策略表现更好（22.8% CAGR，1.19 Sharpe）。

**⚠️ 局限性**

局限性包括：仅考虑固定分段费用的交易成本，未覆盖点差、滑点、税费或整数手数限制；合约对资产类别或主题的约束较简单，可能无法覆盖更复杂的合规需求；基线的因子权重对结果影响有限，仍需在更多市场与周期中验证。

---

## 33. Ko-WideSearch: A Korean Breadth-Search Benchmark for Exhaustive Set Enumeration by Web Agents

**arXiv ID:** 2606.27595 | [PDF](https://arxiv.org/pdf/2606.27595v1)

**作者:** Minbyul Jeong `[一作]` `[通讯]` (Upstage AI), Minbyul Jeong (Upstage AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Ko-WideSearch，一个面向韩语的宽度搜索基准，包含228张表格，涵盖三种难度层级；

**💡 创新点**

创新点在于通过自动合成与验证流水线构建可扩展、可核查的宽度搜索数据集，解决了传统深度搜索评测在非英语场景下缺失宽度评测的缺口；

**🔧 技术方法**

采用了自动构建流水线（包括枚举、非记忆化检验、完整性验证、跨源属性核对）和统一的归一化比较器，以及基于网页的来源标签；

**📊 数据集**

使用来自韩国公开网站的真实网页内容作为数据来源，构成了超过190个父实体、16类的集合枚举任务；

**📈 对比分析**

在20个模型（包括前沿闭源模型、开源大模型和韩语专用模型）上进行对比，评估指标为Item‑F1、Column‑F1、Row‑F1和表格成功率，结果显示模型能恢复成员集但难以完整填充行，且难度随宽度和二维键增加显著下降；

**⚠️ 局限性**

局限性包括类别偏向（硬层级主要为体育赛季）、单一来源页面依赖、评测环境受限于单一搜索后端和预算，以及数据随网页变化需要定期重新验证。

---

## 34. A General Pipeline for Digesting Scientific Literature into a Shared Scientific Knowledge Base

**arXiv ID:** 2606.27384 | [PDF](https://arxiv.org/pdf/2606.27384v1)

**作者:** Charles T. Black `[一作]` `[通讯]`, Charles T. Black

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了Materials Explorer Pipeline，用于将科学文献中的实验数据提取并存储为结构化知识库；

**💡 创新点**

通过可扩展的可携带知识单元（PUK）架构、catchall字段自适应扩展和 AI 辅助的语义相似度搜索，实现了无人工标注即可从文献中生成可查询的实验记录；

**🔧 技术方法**

使用大型语言模型（LLM）进行多模态文档理解、数据提取与推理，结合 JSONL 日志、语义相似度匹配和协同人工审核；

**📊 数据集**

在2023年Co-design Center for Quantum Advantage（C2QA）公开论文中提取了233条样本记录，覆盖10种超导材料；

**📈 对比分析**

通过 Explorer 提供可视化、相似度检索和基于相同材料的比较，展示了 T₂,echo 与 T₁ 的关系，并在小型语料上验证了数据提取与推理的准确性，误差率低于5%；

**⚠️ 局限性**

主要局限包括：对仅出现在图形中的数据识别不足、对非同行评审数据的可信度处理不足、以及需要进一步完善的图形解析和字段标准化步骤。

---

## 35. Spectral Subsurface Scattering from RGB via Biophysical Skin Inversion

**arXiv ID:** 2606.27604 | [PDF](https://arxiv.org/pdf/2606.27604v1)

**作者:** Carlos Aliaga `[一作]` (Meta Reality Labs Research), Adrian Jarabo `[通讯]` (Meta Reality Labs Research)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

从单一的 RGB 皮肤反射率预测出完整的波长依赖表面散射（SSS）参数，提供可直接用于光线追踪的光谱渲染输入。

**💡 创新点**

创新点包括：① 将多层皮肤建模为 K=2/3 的混合介质，显著克服单一介质的形状/振幅耦合限制；② 训练端到端的神经解码器，实现一次前向传播即可得到全波长 SSS 参数；③ 采用可微分 LUT 损失与光谱平滑正则化，解决传统参数回归导致的颜色漂移；④ 通过大量 GPU Monte‑Carlo 数据实现跨波长、跨肤色的泛化。

**🔧 技术方法**

技术手段主要包括：GPU 加速的两层生物物理皮肤 Monte‑Carlo 传输模拟、两维 LUT（α,g）与三维 LUT 预计算、基于权重混合的随机漫步 SSS 采样、三阶段冻结训练的多段式神经网络（Enc→Dec_R→Dec_T），以及自适应曝光、光谱平滑等正则化策略。

**📊 数据集**

数据集为 25,000 个合成皮肤色调（每色调 376 个波长、10⁶ 光子/波长）作为训练集，5,000 个色调做验证；色调参数覆盖从极浅（f_mel≈0.001）到极深（f_mel≈0.65）的完整人类皮肤范围。

**📈 对比分析**

与传统的 RGB 颜色反演+手动调节散射距离/各向异性相比，K=2 方案将渲染误差从 ΔE≈51.4 降至 3.80（约 13.5 倍提升）；K=3 神经解码器进一步逼近并略优于联合优化上限，ΔE=3.19；在 10 个代表性肤色上，混合方案相较单媒介提升 79.9% 的可见光 SAM，显示显著的色谱匹配和散射形状恢复。

**⚠️ 局限性**

局限性包括：① 仅在法线入射、平滑表面假设下拟合，无法完全捕捉高仰角/粗糙表面下的散射行为；② 对唇部等非典型皮肤区域预测效果欠佳；③ K=3 的低α组分在边界均衡区（σ_t <10 cm⁻¹）占比 23.1%，当前 2D LUT 无法精确处理；④ 需要大量 Monte‑Carlo 预训练，模型不易直接迁移到真实测量数据；⑤ 仅使用两层生物物理模型，未提供直接两层参数以消除同质化误差。

---

## 36. The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching

**arXiv ID:** 2606.27510 | [PDF](https://arxiv.org/pdf/2606.27510v1)

**作者:** Sankaran Vaidyanathan `[一作]` (University of Massachusetts Amherst), David Jensen `[通讯]` (University of Massachusetts Amherst)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了Transformer中激活补丁方法（activation patching）所产生的交互效应（INT），揭示自然间接效应（NIE）并不只包含单个组件的因果贡献，而是混合了交互项；同时提出了纯间接效应（PIE）并对其理论性质与多组件交互进行了系统分析；

**💡 创新点**

创新点在于将因果中介分析与Transformer激活补丁结合，正式推导出INT的存在及其对排名和faithfulness的影响，揭示了NIE与PIE排名差异的根本原因，并解释了多组件交互导致的faithfulness不稳定性；

**🔧 技术方法**

技术手段包括因果中介分析、激活补丁（noising/denoising）、理论推导（单/多介质交互分解、二阶近似）以及对GPT-2 IOI电路的实证评估；

**📊 数据集**

使用的数据集包括GPT-2 small、Pythia-70M、Qwen2.5-0.5B模型，任务涵盖IOI、SVA、性别偏见等，并通过对称/非对称（pABC）等不同的counterfactual prompt设计进行评估；

**📈 对比分析**

与传统NIE排名相比，PIE能够消除INT导致的负面影响，但也会产生新的排名失效；实验显示NIE与PIE的Spearman相关系数在不同任务中从0.989降至0.51；INT的方差解释了faithfulness得分在不同prompt模板和噪声设置下的高波动；

**⚠️ 局限性**

局限性包括：INT本质上是不可避免的，需要进行组合搜索或递归搜索以完全控制；方法依赖于模型的局部线性假设与prompt设计，无法保证在所有Transformer架构或任务中适用；此外，现有技术对多组件交互的估计仍然昂贵，难以在大模型上扩展。

---

## 37. Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs

**arXiv ID:** 2606.27378 | [PDF](https://arxiv.org/pdf/2606.27378v1)

**作者:** Fahd Seddik `[一作]` (University of British Columbia), Fatemeh Fard `[通讯]` (University of British Columbia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套基于四条功能公理（因果性、最小化、可分离性与稳定性）的原子化评估框架，用以独立衡量大型语言模型的潜在“思维”向量质量。

**💡 创新点**

创新点在于将思维表示的质量从仅靠下游任务准确率切换为与模型本身内部统计特征无关的四维内在度量，并提供了可直接计算的定量指标；同时系统性审计了多种现有潜在思维提取方式。

**🔧 技术方法**

技术手段包括：KL 替代误差（Causality）、信息瓶颈残差（Minimality）、线性判别器准确率（Separability）以及基于语义熵的 AUROC 评估（Stability），全部在不需要重新训练源模型的前提下完成。

**📊 数据集**

实验使用 BigBench Extra Hard (BBEH) 23 任务集，并在五个公开权重 LLM（Llama‑3.1‑8B、Llama‑3.3‑70B、DeepSeek‑R1‑Distill‑Qwen‑32B、Skywork‑OR1‑32B、GPT‑OSS‑20B）上进行评测。

**📈 对比分析**

比较方法将每个候选思维表示与输入嵌入 (IE) 和输出嵌入 (OE) 两个基准对照；结果表明无一候选在四个轴上均超过 IE，且迭代思维模型随步骤数增大反而性能下降。

**⚠️ 局限性**

局限性包括：未测评稳定性中的词汇不变子属性、评估成本高于单一准确率测试、仅覆盖英文推理任务、且所有候选均为无训练提取，可能遗漏需要专门训练才能满足公理的表示方式。

---

## 38. Qwen-Image-2.0-RL Technical Report

**arXiv ID:** 2606.27608 | [PDF](https://arxiv.org/pdf/2606.27608v1)

**作者:** Yixian Xu `[一作]`, Chenfei Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Qwen-Image-2.0-RL，利用 RLHF 与 on‑policy distillation (OPD) 进一步提升 Qwen-Image‑2.0 在图像生成与编辑任务上的视觉质量与指令遵循能力。

**💡 创新点**

创新点主要包括：① 任务专属的 VLM 复合奖励模型（层级化对齐、美学、肖像/编辑指令与人脸一致性）；② 采用 GRPO 框架的混合 CFG 策略与异步奖励管道；③ 通过轨迹级速度匹配实现多教师 OPD，将 T2I 与编辑两类任务专门化策略统一到单一部署模型。

**🔧 技术方法**

技术手段：Flow‑GRPO、DiffusionNFT、GRPO、混合 CFG、异步奖励管道、点对点回归奖励训练、轨迹级速度匹配 OPD。

**📊 数据集**

数据集：Qwen‑Image‑Bench（自动化评测）与约 130K 人类标注图像‑提示对（训练奖励模型、评估 Elo），以及用于编辑任务的专门化提示/图像数据。

**📈 对比分析**

与基线和混合 RL (Mix‑RL) 对比，Qwen‑Image‑2.0‑RL 在 Qwen‑Image‑Bench 最高得分从 55.23 提升至 57.84，Elo 评分从 1193（T2I）/1349（编辑）进一步提升，显著优于混合 RL 与原始基线。

**⚠️ 局限性**

局限性：需人工定义奖励维度与权重；对多任务混合优化的依赖；在极端或新颖任务下的泛化可能受限；训练与推理均需较高算力与复杂管线。

---

## 39. Prism Transformer: Progressive Head Schedules for Hierarchical Attention Processing

**arXiv ID:** 2606.27449 | [PDF](https://arxiv.org/pdf/2606.27449v1)

**作者:** Shubham Aggarwal `[一作]` `[通讯]` (Indian Institute Of Technology Madras), Shubham Aggarwal (Indian Institute Of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Prism Transformer，通过在Transformer层级中逐层增加注意力头数，形成从宽到窄的非均匀子空间分配；

**💡 创新点**

核心创新是将注意力头数从早层的少且宽子空间逐步增加到后层的多且窄子空间，实现无额外参数或计算开销的局部到全局表征层次；

**🔧 技术方法**

技术手段包括多头注意力的进阶头数调度、保持投影矩阵维度不变以实现参数和 FLOPs 的零增；

**📊 数据集**

在 FineWeb 语料上预训练，随后在 PIQA、HellaSwag、ARC‑Easy、WinoGrande 等零样本下游任务上评估；

**📈 对比分析**

通过与统一头数基线进行多种模型规模（124M/354M/757M）对比，验证 Prism Transformer 在相同训练算力下取得更低的验证损失和更高或相当的下游任务准确率，且硬件吞吐量保持一致；

**⚠️ 局限性**

局限性包括仅评估解码器仅模型、1024-token 上下文长度、未探讨编码器或编码-解码器架构、进阶头数调度的更灵活学习方式未尝试。

---

## 40. Structured-Li-GS: Structured 3D Gaussians Splatting with LiDAR Incorporation and Spatial Constraints

**arXiv ID:** 2606.27509 | [PDF](https://arxiv.org/pdf/2606.27509v1)

**作者:** Huaiyuan Weng `[一作]` (University of Waterloo), Chul Min Yeum `[通讯]` (University of Waterloo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

结合 LiDAR 与 3D 高斯投影（Gaussian Splatting）技术，提出轻量级 Structured‑Li‑GS 框架，实现高质量 3D 重建且保持模型体积紧凑。

**💡 创新点**

创新点在于：① 通过 LiDAR‑视觉 SLAM 生成稠密彩色点云，做锚点初始化；② 正则化基于法线的高斯旋转与扁平化；③ 综合光度、扁平化、偏移、深度和法线损失，显著提升几何与视觉一致性；④ 直接用稠密点云而非传统稀疏 SfM，避免密集高斯增量。

**🔧 技术方法**

使用技术包括：LiDAR‑惯性‑视觉 SLAM（Fast‑LIVO2）、体素化与锚点生成、正交旋转四元数、光度与深度渲染损失、基于差分光栅化的 3D Gaussian 渲染。

**📊 数据集**

评估数据集为：公共 FASTLIVO2、HILTI22（室内外混合）以及作者自建的硬件同步 LiDAR‑摄像头手持扫描数据。

**📈 对比分析**

与 3D‑GS、2D‑GS、Scaffold‑GS、AtomGS、LetsGo 等现有方法对比，Structured‑Li‑GS 在 PSNR、SSIM、LPIPS 上均位居或紧随前列，并且高斯数量仅为 Scaffold‑GS 的 1/3（约 356k），显著提升了模型紧凑性与渲染效率。

**⚠️ 局限性**

主要局限在于：① 需要高质量的 LiDAR‑视觉 SLAM 预处理，若传感器噪声大或 SLAM 失效会影响重建；② 对极其稀疏或无 LiDAR 覆盖的区域依旧受限；③ 当前实现缺乏多层级细节（LoD）与网格提取，难以直接用于大规模场景的细粒度编辑。

---

## 41. Not All Relations Rotate Alike: Transformation-Aware Decoupling for Viewpoint-Robust 3D Scene Graph Generation

**arXiv ID:** 2606.27412 | [PDF](https://arxiv.org/pdf/2606.27412v1)

**作者:** Jingjun Sun `[一作]` (Northwestern Polytechnical University), Shan Gao `[通讯]` (Northwestern Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种面向视角鲁棒的3D场景图生成框架TAD，通过视角稳定对象编码和谓词分解实现对姿态变换的自适应推理。

**💡 创新点**

创新点在于将谓词分为角度相关和角度不变两类，并在描述符、参数和目标层面同时进行解耦，配合正交正则化和组感知辅助监督，使模型在不做旋转增强的情况下即可在不同yaw视角下保持高精度。

**🔧 技术方法**

采用RI-MAE预训练的旋转不变点云编码器、基于GNN的关系网络、方向特定描述符、正交正则化和辅助监督等技术实现分支解耦与融合。

**📊 数据集**

使用3DSSG（基于3RScan的室内场景图数据集）进行实验，包含160类对象和26类谓词。

**📈 对比分析**

在标准3DSGG任务上与SOTA方法（如VL‑SAT、OCRL等）相比，TAD在SGCls和PredCls的平均召回率上排名第一；在yaw角度（0°、90°、180°、270°）的鲁棒性评估中，TAD在90°交换轴的情况下相较于基线下降仅1.5点，远优于仅使用旋转增强或几何规则的对照模型。

**⚠️ 局限性**

局限性包括仅针对离散的yaw旋转和固定词表设计，无法直接推广到连续旋转、自由空间或开放谓词集合的场景；未来工作需扩展到更广泛的姿态变化和自学习谓词变换规律。

---

## 42. QueenBee Planner: Skill-Evolving Communication Topologies for Token-Efficient LLM Multi-Agent Systems

**arXiv ID:** 2606.27492 | [PDF](https://arxiv.org/pdf/2606.27492v1)

**作者:** Congjia Tian `[一作]` (Virginia Tech), Jiaming Cui `[通讯]` (Virginia Tech)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种名为QueenBee Planner的框架，冻结工作者，利用外部LLM规划器生成时间展开的通信DAG，并通过经验迭代生成设计规则。

**💡 创新点**

将跨代理通信拓扑视为可检索、可自我改进的设计技能，采用证据条件的Preserve/Modify/Avoid规则，并通过持有门槛、方差敏感信用等机制保证可信自进化。

**🔧 技术方法**

使用LLM生成器、证据驱动的技能库、结构性motifs信用分配、持有门槛、转移信任、洞察反驳以及结构去重等技术。

**📊 数据集**

在Count‑Frequency计数聚合任务和Silo‑Bench风格的分布式协调任务上进行实验。

**📈 对比分析**

与固定拓扑、冷启动自由DAG以及per‑case固定拓扑oracle比较，结果显示自进化的自由DAG在RMSE、消息数、模型调用、token成本以及Silo的EM方面均优于基线，提升幅度可达30–50%以上。

**⚠️ 局限性**

主要限制包括：对结构motifs的手工设计可能不足，算法依赖大量自我评估样本，无法保证对极端任务的泛化，且工件库增大时检索效率下降。

---

## 43. Global Explanations for Multivariate Time Series Forecasting Models via $K$-Order Markov Approximations

**arXiv ID:** 2606.27599 | [PDF](https://arxiv.org/pdf/2606.27599v1)

**作者:** Amadeo Tunyi `[一作]` `[通讯]` (XITASO), Amadeo Tunyi (XITASO)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出KARMA方法，通过K阶马尔可夫近似为黑盒多变量时间序列预测模型提供全局可解释性。

**💡 创新点**

创新点在于自适应确定最小足够滞后K*，实现模型压缩与基线认证，并从马尔可夫转移核直接生成五层全局解释层次，同时给出统计可靠性证明。

**🔧 技术方法**

采用马尔可夫近似、量化分箱离散化、总变差停止规则、基线自适应、蒙特卡洛采样、树结构合并、总变差影响度量以及因果边缘检测等技术。

**📊 数据集**

使用了真实天气数据（北京PM2.5）、ETTh1电力、外汇以及合成VAR数据进行实验。

**📈 对比分析**

与TimeSHAP、WinIT、DynaMask、FO、IG等方法比较，KARMA在合成VAR上召回率最高、精度优秀；在真实数据上AUC_lag@25%最高或接近最佳，尤其在ETTh1 TCN上显著优于其他方法。

**⚠️ 局限性**

限制在于状态空间指数增长导致高维时序难以完整估计，仅能通过边缘核和采样策略近似，且在极高维系统下仍需改进。

---

## 44. Implementing GenAI-Supported Learning in Software Engineering and Computer Science Education using Bloom's Taxonomy

**arXiv ID:** 2606.27398 | [PDF](https://arxiv.org/pdf/2606.27398v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 45. Odyssey: Constructing Verifiable Local Truth-Preserving Foundation Models

**arXiv ID:** 2606.27593 | [PDF](https://arxiv.org/pdf/2606.27593v1)

**作者:** Sridhar Mahadevan `[一作]` `[通讯]` (Adobe Research), Sridhar Mahadevan (Adobe Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套名为 Odyssey 的可验证、局部真值保持的 foundation model 架构，将模型拆分为可检验的 sheaf‑like 本地预测与逻辑模型，并通过范畴论的 Kan extension、Topos World Model、Toulmin 论证、FSQL 查询等技术实现模型的构造、管理与审计；

**💡 创新点**

创新点主要在于①将 foundation model 视为由 generic foundry 组合、限制、拼接而成的类型化 algebra；②引入基于范畴论的 Kan extension 机制（TICKET）实现模型的可插拔与可审计；③在每个阶段加入 Toulmin 论证层与 FSQL 查询表面，使局部证据、推理、冲突显式化并易于可视化；

**🔧 技术方法**

使用范畴论（sheaf、Kan extension）、Topos World Model、Toulmin 论证框架、FSQL typed 查询、BRIDGE/SKFM 侧面、SkillOpt 训练、LLM 预测/推理与局部 PSR 生成等技术；

**📊 数据集**

涉及数据集包括 Dick’s Sporting Goods、Indus Script、MyFixIt 修复手册、TCC 44K 经济因果主张、Amazon Reviews 2023、BLaIR‑Bench、IKEA ASM、10‑K 财务文件等多种领域数据；

**📈 对比分析**

以 MyFixIt 为例，将 PSR action‑observation 作为检索特征，相比 token baseline在 compact、broader、strict 三个 profile 的 nDCG@10、Recall@10、MRR 均提升约 +0.24~+0.27；其他 foundry 通过 gluing diagnostics、obstruction ledger 等可视化验证了可插拔性与局部一致性；与传统单一模型 benchmark 对比显示 Odyssey 在可解释性和局部一致性上更稳健；

**⚠️ 局限性**

局限性包括：依赖 deterministic templates 与手工规则；generic foundry 的定义和拼接仍需人工维护；跨域迁移的全局一致性未实现自动化保证；LLM 论证层受限于提示质量；缺乏大规模自动化评估与在线更新机制；

---

## 46. MemoBench: Benchmarking World Modeling in Dynamically Changing Environments

**arXiv ID:** 2606.27537 | [PDF](https://arxiv.org/pdf/2606.27537v1)

**作者:** Haoyu Chen `[一作]` (Harvard University), Yilun Du `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MemoBench，一个基于消失-再现（Disappear‑and‑Reappear）范式的世界建模基准，用于评估视频生成模型在动态环境下的记忆一致性。

**💡 创新点**

创新点在于首次将目标在视野外消失后再出现的物体记忆性作为核心评价指标，并设计了自动化指标与 LLM‑驱动的 VQA 两层评估体系，覆盖视觉质量、运动平滑、物体一致性、3D 几何一致性以及记忆、像素级精度、相机控制等维度。

**🔧 技术方法**

采用多模态评估技术：无参考视觉质量评估（AestheticScore + CLIP‑IQA+）、RAFT 光流与深度一致性评估、DINOv2 视觉特征相似度、Depth Anything 预测深度、SAM‑3 目标检测、MapAnything 姿态估计、ImageReward 与 LLM（Gemini‑3.1‑Pro）生成问题与答案。

**📊 数据集**

使用 360 条高质量真实与合成视频（196 合成、164 真实），每条视频按可见–消失–再现三阶段划分，附带相机轨迹、深度图、相机内外参，覆盖多种场景与物理状态变化。

**📈 对比分析**

对 10 款主流视频生成模型（CI2V、3D、I2V）进行评估，发现无模型的对象再现得分（ORS）超过 0.6，且 I2V 模型往往通过保持相机静止来夸大一致性；CI2V 模型在执行相机轨迹上优于 I2V，但在记忆保持和物体再现方面仍表现不足；相机控制和像素级精度之间存在明显 trade‑off。

**⚠️ 局限性**

主要局限在于缺乏持久状态表示，现有模型在物体消失后很快遗忘，导致再现得分低；评估指标对不运动的模型易产生偏差；数据集规模和场景多样性仍有限，未来需加强对动态记忆机制的研究与更大规模的基准构建。

---

## 47. A Sensitivity-Aware Test Collection for Search Among Personal Information

**arXiv ID:** 2606.27559 | [PDF](https://arxiv.org/pdf/2606.27559v1)

**作者:** Jack McKechnie `[一作]` (University of Glasgow), Craig Macdonald `[通讯]` (University of Glasgow)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了SARA（Sensitivity‑Aware Relevance Assessment）测试集合，对Enron邮件数据进行查询、相关性与敏感性标注；

**💡 创新点**

提供了免费的、可公开使用的SAS测试集合，解决了先前昂贵或受限许可的集合缺口，并结合人类与LLM评估保证标注质量；

**🔧 技术方法**

使用主题模型（LDA）生成信息需求，Crowdsource生成查询和相关性评估；利用LLM（Llama 7B）自动生成剩余评估；训练T5模型做敏感性分类；评估使用nDCG、sens_docs@k和nCSDCG等指标；

**📊 数据集**

基于Enron邮件语料（约129k条邮件）及其Hearst子集的敏感性标签；

**📈 对比分析**

通过对多种检索模型（BM25、TCTColBERT、SPLADE++、cross‑encoders、LLM rerankers）进行基线实验，显示稠密检索>稀疏检索>词袋检索，且LLM reranker提升nDCG；敏感性过滤器基于T5分类器能显著减少敏感文档且保持高nCSDCG；

**⚠️ 局限性**

SARA仍依赖手工与LLM评估，可能存在偏差；敏感性分类仅区分两类（个人/非个人），缺乏细粒度；评估主要基于离线检索，未覆盖用户交互场景；

---

## 48. Large Language Model Teaches Visual Students: Cross-Modality Transfer of Fine-Grained Conceptual Knowledge

**arXiv ID:** 2606.27527 | [PDF](https://arxiv.org/pdf/2606.27527v1)

**作者:** Thomas Shih-Chao Liang `[一作]` (University of Wisconsin-Madison), Yong Jae Lee `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种跨模态知识蒸馏框架，利用大型语言模型（LLM）生成多项选择题（MCQ）来提供结构化概念监督，进而训练仅使用视觉输入的学生模型；

**💡 创新点**

创新点在于：1）仅通过语言教师而不依赖图像-文本对；2）通过生成的MCQ提取的软标签分布构建“概念签名”，为视觉模型提供多维语义约束；3）实现了在细粒度分类任务中显著提升鲁棒性和性能；

**🔧 技术方法**

技术包括：大规模语言模型（如Qwen2.5-7B）作为教师，GPT‑4o或Gemini 2.5 Pro生成MCQ，使用预softmax logits做软标签，学生模型添加辅助头并以MSE与概念签名对齐；

**📊 数据集**

在六个细粒度分类数据集（CUB‑200、Caltech‑101、102Flowers、FGVC Aircraft、Oxford‑IIIT Pet、Stanford Cars）以及通过WordNet分组的ImageNet子集上进行评估；

**📈 对比分析**

与传统知识蒸馏方法（KD、RKD、DKD、MLKD、Logit Standardization）以及多模态LLM蒸馏方法（MaKD、LLaVA FitNet/CRD）对比，实验显示本方法在多数数据集上均优于或匹配最先进方法，并在Waterbirds数据集显著提升worst‑group准确率；

**⚠️ 局限性**

局限性包括：1）依赖LLM生成的MCQ，若类别名称或语义难以表述，监督效果受限；2）对抽象或无意义标签的效果不明；3）在大规模类别空间上需生成更多多样化问题，虽然单次生成成本低，但覆盖面仍是挑战；

---

## 49. Multi-Objective Molecular Generation with Frequency-Controlled Evolutionary Dynamics

**arXiv ID:** 2606.27467 | [PDF](https://arxiv.org/pdf/2606.27467v1)

**作者:** Elia Colleoni `[一作]` (King Abdullah University of Science and Technology), William Lafayette Roberts `[通讯]`

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出SpectralMol，一种基于傅里叶系数的演化算法，用频谱矩阵作为基因型，投影得到位置向量再通过SELFIES解码生成分子；采用NSGA-II进行多目标优化，完全无训练需求，结构可解释；

**💡 创新点**

创新点在于：①将分子基因型表示为压缩的傅里叶系数矩阵，天然实现低频全局变化与高频局部微调的分离；②利用固定正交的token嵌入与约束掩码保证解码合法；③在多目标任务中无需聚合成标量奖励，直接保持Pareto前沿；④所有模块均训练‑free，极大提升可部署性与解释性。

**🔧 技术方法**

核心技术包括：频域编码（傅里叶变换、固定基底投影）、SELFIES解码（距离‑基概率、约束掩码）、基于NSGA‑II的多目标遗传演化（交叉、变异、非支配排序）、手工构造的token词表与正交投影嵌入。

**📊 数据集**

使用了标准分子生成基准数据集GuacaMol和MolExp(L)（包括多任务、探索和多目标子集）以及ZINC250k化合物库作为种子；还使用ClpP蛋白（PDB 6U0J）与QuickVina2‑GPU进行对接评价。

**📈 对比分析**

在GuacaMol与MolExp基准上，SpectralMol与GraphGA等传统演化方法整体得分相近，但在多参数优化（MPO）任务上显著优于GraphGA；在ClpP对接+QED+SA的真实多目标实验中，在相同1000次oracle调用预算下，SpectralMol生成的独特骨架数和hit数量均高于Saturn RL模型，且保持了相似的药物性状（QED）但略高的合成可行性（SA）。

**⚠️ 局限性**

局限性包括：①词表与嵌入是手工编码，可能缺失深度学习能捕获的细粒度化学模式；②频谱模式权衡（低频 vs 高频）对不同任务敏感，需进一步调优或尝试其他光滑基底；③对非常难的目标（如严格的对接阈值）时收敛一致性略低，且生成的分子合成复杂度略升高。

---

## 50. Support-Constrained RL Enables Real-World Policy Improvement without Real-World Experience

**arXiv ID:** 2606.27475 | [PDF](https://arxiv.org/pdf/2606.27475v1)

**作者:** Raymond Yu `[一作]` (University of Washington), Abhishek Gupta `[通讯]` (University of Washington)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过在仿真中对已用真实数据预训练的生成式流匹配策略进行流量引导，实现离域强化学习，显著提升多指手操纵任务的成功率和执行速度。

**💡 创新点**

创新点在于将策略改进限制在真实策略的支持集合内，利用流量引导避免仿真与真实世界的分布偏移，同时不需要对原策略进行蒸馏或再训练。

**🔧 技术方法**

技术主要包括条件流匹配（生成式流策略）、DSRL/RFS 流量引导、PPO 强化学习以及异步 actor‑critic 架构。

**📊 数据集**

使用了 LEAP Hand 连接 Franka FR3 的 8 种多指操纵任务的真实演示、重试和玩耍数据集。

**📈 对比分析**

与无约束 RL、分布正则化（BC、残差）等基线相比，Score 在 8 个任务上平均成功率从 37.8% 提升至 89.9%，并将执行时间平均缩短 36.8%。

**⚠️ 局限性**

局限性在于改进效果受训练数据覆盖范围限制，离域提升在分布外环境下表现受限；多任务扩展仍需更丰富、更广泛的行为先验。

---

## 51. Test-Input Generation for Tensor Programs: What Actually Finds Kernel Bugs

**arXiv ID:** 2606.27396 | [PDF](https://arxiv.org/pdf/2606.27396v1)

**作者:** Dipankar Sarkar `[一作]` `[通讯]` (Arizona State University), Dipankar Sarkar (Arizona State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了七种张量内核测试输入生成策略，探究它们对缺陷检测召回率和误报率的影响。

**💡 创新点**

创新点在于将形状候选集、数据类型和数值分布三大维度进行离散化并量化评估，发现边界形状采样既能显著提升召回率，又保持零误报，构成可推广的基线。

**🔧 技术方法**

使用了 gpuemu 的模式感知种子模糊器作为生成引擎，配合可切换的形状候选集（boundary、regular、default）、数据类型集合以及数值分布（Uniform、NaNInjected、Adversarial）三项参数。

**📊 数据集**

实验基于包含 26 个算子（16 个正确控制、10 个基于 LLM 语法错误生成的 buggy 变体）的 GPU 模块，在 RTX 3060 实例上完成。

**📈 对比分析**

通过 1,456 次（7 策略 × 26 算子 × 8 次）实验比较，每种策略在 10 个 buggy 样本上的召回率与 16 个控制样本上的假阳性率进行对比；结果显示 boundary 策略召回 78%、假阳性 0%，而 adversarial 策略召回 99% 但假阳性高达 94%。

**⚠️ 局限性**

主要局限在于验证器对非有限输入（NaN/Inf）过度严格，导致正确内核被误判为失败，且实验仅使用作者预先构造的错误样本，未对真实 LLM 生成的内核进行模糊验证。

---

## 52. Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement

**arXiv ID:** 2606.27409 | [PDF](https://arxiv.org/pdf/2606.27409v1)

**作者:** Igor Itkin `[一作]` `[通讯]` (Independent Researcher), Igor Itkin (Independent Researcher)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了多智能体LLM系统中验证延迟导致的幻觉传播和振荡，并给出了闭式阈值。

**💡 创新点**

创新点在于提出了验证剂量-延迟阈值、基于超级模性质的贪心置换方案，并证明了延迟同步最不稳定。

**🔧 技术方法**

采用了有根拉普拉斯谱分解、离散延迟控制理论、Chebyshev多项式、超级模理论以及实验验证。

**📊 数据集**

在五个开源LLM模型（Qwen3.6‑35B、Qwen‑3.14B、Mistral‑7B、Phi‑4、Gemma‑4‑12B）上进行了数值估计和事实问答实验。

**📈 对比分析**

与无延迟或未固定真相的基线对比，发现验证剂量过大或延迟过长导致振荡；贪心置换可将误差降低约70%以上。

**⚠️ 局限性**

主要限制包括仅适用于对称交互图、线性局部分析、未验证非线性全局收敛，以及对拜占庭错误或边缘修正的适用性尚未探讨。

---

## 53. AI-Model Network: Concept, Current State and Future

**arXiv ID:** 2606.27382 | [PDF](https://arxiv.org/pdf/2606.27382v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 54. The Context-Ready Transformer

**arXiv ID:** 2606.27538 | [PDF](https://arxiv.org/pdf/2606.27538v1)

**作者:** Mahesh Godavarti `[一作]` `[通讯]` (A Carrot Inc), Mahesh Godavarti (A Carrot Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“Context-Ready Transformer”，通过在进入Transformer块前对每个 token 进行基于前一位置块输出的预上下文化，减少对深度的依赖；

**💡 创新点**

核心创新在于：①非累积（non‑cumulative）与仅依赖过去（past‑only）的纠正网络，保证在推理时单次左到右通道即可得到收敛结果；②训练时通过对整个序列进行 K 步迭代（K≪T）实现对序列递归的近似，从而在保持 Transformer 并行训练优势的同时获得 RNN 的递归特性；

**🔧 技术方法**

技术手段包括：Transformer block、RoPE 位置编码、专用纠正 FFN、K 步并行训练、随机深度训练（k_min）、KV 缓存、BPTT 训练与微调、指针追踪任务验证；

**📊 数据集**

主要实验数据集：OpenWebText、English Wikipedia 以及自制指针追踪任务；

**📈 对比分析**

与标准多层 Transformer、Roformer 等基线在同一 FLOPs/参数规模下进行对比；结果显示：D=5、C=1120 的 Context-Ready 模型在 121M FLOPs/token 下比 12 层 Transformer 取得更低的 PPL（36.38 vs. 37.83）且生成速度提升 1.7×；D=1、C=2048 的单层模型在 149M FLOPs/token 下比 6 层 Transformer 速度提升 2.6×，且 PPL 仅高 0.1；在指针追踪任务中，D=1 能解所有 10 步级别，表明可突破单层 Transformer 的深度瓶颈；

**⚠️ 局限性**

局限性包括：仅在 1–4 亿参数规模实验，缺乏大规模（B+级）验证；从零训练时 K 步迭代导致显存与训练成本提升；prefill 过程需要 K 步，导致初始填充延迟；对标准基准（如 LAMBADA、GLUE 等）验证不足；

---

## 55. Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM Decoding

**arXiv ID:** 2606.27596 | [PDF](https://arxiv.org/pdf/2606.27596v1)

**作者:** Liu Yu `[一作]` (University of Electronic Science and Technology of China), Gillian Dobbie `[通讯]` (University of Auckland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无训练的因果推理时框架，诊断并切断大型视觉语言模型在决策关键时刻出现的危险注意力头，从而消除生成中的物体幻觉并保持语言流畅。

**💡 创新点**

创新点：①把幻觉视作动态结构失配；②利用视觉注意力熵无监督定位危险介质；③通过数值 logit 饱和实施 do‑operator，物理切断语言先验快捷路；④设计冲突门控合作解码，平衡可信度与表达丰富度。

**🔧 技术方法**

技术手段：因果结构模型（SCM）、视觉注意力熵探测、数值 logit 饱和干预、冲突门控融合、Nucleus 采样等。

**📊 数据集**

使用数据集：POPE、CHAIR、MME 以及 GPT‑4V 评测；在 LLaVA‑1.5、Shikra、InstructBLIP 三大 LVLM 上进行实验。

**📈 对比分析**

与 ICD、VCD、OPERA、SID、CausalMM 等推理时方法对比；在 POPE、CHAIR、MME 上均显著提升，尤其在 CHAIRS 的实例幻觉率下降 29.1%；保持甚至提升文本的细节与可读性；在 GPT‑4V 评估中同时提升正确性与详细性。

**⚠️ 局限性**

局限性：主要针对对象幻觉，对属性、关系、计数等错误缺乏覆盖；在分布漂移环境下鲁棒性未全面验证；无训练方式无法保证所有生成事实的绝对正确，仍需人工监督与风险沟通。

---

## 56. Beyond MoCap: Scaling Motion Tokenizers with Synthetic Human Motion for Generative Modeling

**arXiv ID:** 2606.27547 | [PDF](https://arxiv.org/pdf/2606.27547v1)

**作者:** Yiwen Yan `[一作]` (Dartmouth College), Yu-Wing Tai `[通讯]` (Dartmouth College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通过大规模合成运动数据并扩展 VQ‑VAE 码表来丰富运动表示的方法。

**💡 创新点**

创新点在于将数据分布与离散码表容量同步扩展，并利用结构化随机探索生成物理可行的稀有动作，从而突破传统 MoCap 数据限制。

**🔧 技术方法**

采用基于 SMPL 的姿态生成与遗传算法风格的交叉/变异探索、球面插值构造运动序列；使用 VQ‑VAE 对运动进行离散化并通过 EMA 与码表重置提升利用率；随后对已有生成模型进行细调。

**📊 数据集**

主要使用 HumanML3D 作为训练集，并合成约 64 倍额外运动；在 Motion‑X++ 上做跨数据集评估。

**📈 对比分析**

与原始 VQ‑VAE 代码表和传统 tokenizer 比较，合成数据训练后在 HumanML3D 上 reconstruction FID 下降 42.4%，在 Motion‑X++ 上 MPJPE 同样降低；在文本到运动生成任务中，FID 分别提升 16.4%、24.1% 与 20.0%。

**⚠️ 局限性**

仅覆盖身体运动，缺少手部与面部细节；合成运动未配文本注释，导致语义对齐略有下降；生成模型对极端动作的准确性仍有限。

---

## 57. Contextual Associations Between Webpage Elements for Web Accessibility: An Empirical Study

**arXiv ID:** 2606.27506 | [PDF](https://arxiv.org/pdf/2606.27506v1)

**作者:** Kishan Rakesh `[一作]` (University of Texas at Dallas), Shiyi Wei `[通讯]` (University of Texas at Dallas)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个基于可访问性树的网页上下文关联数据集，并用图神经网络模型预测哪些周围元素能为可访问名称提供语义上下文。

**💡 创新点**

首次将可访问性树与空间、语义、样式特征结合，用链路预测的方式学习人类感知的上下文关联，解决屏幕阅读器在无上下文下难以解释可访问名称的问题。

**🔧 技术方法**

采用了四种图神经网络（MLP、GCN、GAT、SEAL）和两种启发式基线（空间距离、树距离），在留一网站交叉验证下进行评估；使用 Playwright、Chrome DevTools 捕获 AX 树并构建图。

**📊 数据集**

使用了 35 个从 Tranco top‑million 列表抽样的网站，每个页面有 3 名注释者标注上下文关联，形成约 600+ 正边的有向图数据集。

**📈 对比分析**

通过 Hit@10、Hit@5、Hit@1 和 MRR 进行评估，实验显示 MLP 等学习模型在 Hit@10 上平均提升约 0.1 以上，相较于启发式基线取得显著但不超过预设阈值的改进；各模型在不同网站的泛化表现不一。

**⚠️ 局限性**

局限性在于数据仅覆盖英语、Tranco 前百万且为静态快照，未覆盖弹窗、懒加载等交互动态内容；样本量 35 站点有限，无法充分评估布局与内容多样性；标注任务开放式导致噪声与标签稀疏，影响学习效果。

---

## 58. A&A community survey on the future of scientific publishing: Credibility over speed, fairness over profit, human judgment over automation

**arXiv ID:** 2606.27447 | [PDF](https://arxiv.org/pdf/2606.27447v1)

**作者:** João Alves `[一作]` (University of Vienna), Eva Villaver `[通讯]` (Instituto de Astrofísica de Canarias)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了一项涵盖69个国家、2944名作者的问卷调查，系统收集并分析了关于期刊选择、同行评审、科研评价、开放获取以及人工智能在出版中的作用等方面的定量与定性数据。

**💡 创新点**

提供了首批大规模、细致的天文学出版社区意见调查结果，并通过人工智能辅助的文本编码揭示了社区对可信度、公平性和人类监督的共识，为期刊政策制定提供了基于证据的决策框架。

**🔧 技术方法**

采用在线问卷平台、统计分析（均值、频率分布）以及大语言模型（LLM）对近6万字自由文本进行编码和主题归纳。

**📊 数据集**

使用了来自A&A近三年作者和合作者的邮件列表，共计27,787名潜在受访者所发出的问卷，最终收集到2944份有效回复，涵盖了69个国家的研究者。

**📈 对比分析**

通过对问卷结果与先前相关研究（如San Francisco宣言、Leiden Manifesto等）的对比，发现社区在期刊质量、成本和公平性方面的偏好与以往一致；但在AI使用上显示出新的分歧；总体上，调查提供的见解具有高度可靠性，受访者覆盖面广、响应率约10%。

**⚠️ 局限性**

主要局限包括自选样本偏差、邮件列表覆盖范围限制、对非活跃作者的欠缺覆盖以及定性评论可能过度代表较为发声的意见；此外，对不同地区、不同学科子领域的细粒度分析仍不足。

---

## 59. CoIn: Comprehensive 2D-3D Inpainting with Gaussian Splatting Guidance

**arXiv ID:** 2606.27584 | [PDF](https://arxiv.org/pdf/2606.27584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 60. Speculative Refinement: A Hybrid Autoregressive Diffusion Decoding Strategy and Its Behavior Across Benchmarks

**arXiv ID:** 2606.27474 | [PDF](https://arxiv.org/pdf/2606.27474v1)

**作者:** Aditi Gupta `[一作]` (IIIT Hyderabad), Pawan Kumar `[通讯]` (IIIT Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SpecRef，一种训练无关的混合AR-扩散生成方法，通过把AR生成的草稿作为起点并利用熵引导的选择性掩码进行扩散细化；

**💡 创新点**

创新点在于将稀疏掩码与扩散模型结合，实现对AR草稿的快速细化；同时系统性评估了混合生成在多种评估协议下的表现，揭示了评估协议对结果的显著影响；

**🔧 技术方法**

使用Phi-2（2.7B）作为AR草稿生成器，LLaDA（8B）作为掩码扩散细化器，熵度量决定掩码位置，并采用数学块扩展与尾部截断两种启发式；

**📊 数据集**

在六大基准上评估：HumanEval、MBPP、GSM8K、BBH、ARC-Challenge、HellaSwag；使用执行、精确匹配和对数似然三种评估协议；

**📈 对比分析**

在代码基准中，SpecRef将单纯扩散模型从0%提升至20%以上，仅需8-16步；在数理推理基准上，SpecRef的准确率可超过单独AR或扩散模型；但在已经高度准确的任务（如BBH）中，多阶段细化反而导致精度下降；整体而言，SpecRef在给定步数下可实现更高准确率并减少2-3倍推理时间；

**⚠️ 局限性**

局限性包括仅测试单一AR-扩散组合，掩码比例固定为60%且未针对任务自适应；Phi-2的上下文窗口限制了对长文本任务的评估；未深入探索如何在已达上限任务中避免细化张力；

---

## 61. Fine-tuning a multimodal large language model for clinician-grade autism behavioral scoring from short home videos

**arXiv ID:** 2606.27484 | [PDF](https://arxiv.org/pdf/2606.27484v1)

**作者:** Mohammadmahdi Honarmand `[一作]` (Stanford University), Dennis P. Wall `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在此研究中，作者通过低秩适配（LoRA）对Gemini 2.5 Pro进行微调，使用400条临床评分的家庭短视频，提取30项行为特征并评估其在99条独立测试视频上的诊断性能。

**💡 创新点**

创新点在于：①仅用行为特征标签进行微调即可显著提升模型与临床评分的一致性（κ从0.40升至0.56，27/28特征提升），②微调后模型在未见过的全球ASD诊断任务上零样本性能跃升（F1提升53%），③将微调得到的特征输入下游经典分类器，可与人类评分相媲美并超越基线模型。

**🔧 技术方法**

技术手段包括：多模态大语言模型Gemini 2.5 Pro、低秩适配微调、基于视频+提示的行为评分提示设计、以及下游随机森林、XGBoost、LR10/5等传统机器学习分类器。

**📊 数据集**

使用的数据集由595条来自三源（美国GuessWhat、公开YouTube、孟加拉国研究）的短视频组成，划分为400条训练、96条验证、99条测试（49 ASD、50 NT）。

**📈 对比分析**

通过与临床评分的三种输入条件（基线Gemini、微调Gemini、临床评分）以及五条诊断路径（直接诊断、四种下游分类器）比较，微调模型在准确率77%/AUC86%/F1≈0.77的表现与临床评分基本持平，并且在直接诊断上显著优于基线模型。

**⚠️ 局限性**

局限性包括：①微调数据量相对有限（仅400条视频），②测试集规模小（99条），③模型为封闭式Gemini 2.5 Pro，缺乏开放权重复现，④缺乏在更广泛人口与录制环境下的前瞻性验证。

---

## 62. Radar Guided Camera Verification for Automatic Emergency Braking Rethinking Object Detection in Radar Camera Fusion

**arXiv ID:** 2606.27556 | [PDF](https://arxiv.org/pdf/2606.27556v1)

**作者:** Ram Charan Akula `[一作]` (SRM Institute of Science and Technology), Manikandan Ganesan `[通讯]` (SRM Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在自动紧急制动系统中，提出并实现了一种基于雷达投影的边缘密度门控方法，用来在雷达定位的兴趣区域内进行障碍物验证，从而不需要完整目标检测。

**💡 创新点**

创新点在于将摄像头的角色从完整目标检测转变为只做障碍物验证，利用无训练、无GPU、无模型权重的经典边缘检测与阈值门控，显著降低计算量并保持高召回率。

**🔧 技术方法**

核心技术包括雷达目标检测与卡尔曼滤波追踪、雷达到摄像头的投影、Canny边缘检测、简单阈值门控、时间到碰撞（TTC）计算与制动决策；全部使用Python + OpenCV实现。

**📊 数据集**

使用真实车辆（ARS408雷达 + Intel RealSense D435摄像头）在72个驾驶会话中收集的131,603帧图像，并手工标注了317个雷达投影ROI用于门控评估；还在33个威胁情境和3个无威胁情境中进行行为测试。

**📈 对比分析**

与传统全图目标检测相比，ROI处理将搜索空间缩小至1–17%（最多98.7%），平均延迟从0.706 ms降至0.121 ms（5.8×加速）。门控AUC为0.898，召回率0.994，所有33个威胁场景均无误触发制动，误报率在非威胁场景为1.000。

**⚠️ 局限性**

主要局限包括雷达-摄像头标定误差导致ROI投影不准、对照明条件敏感（晚间误报率上升）、单一标注者导致标注一致性未知、仅在单辆车平台验证，缺乏跨车、跨环境的通用性验证。

---

## 63. When the Aggregator Cheats: Data-Free Backdoors in Federated LLM-based QA Systems

**arXiv ID:** 2606.27511 | [PDF](https://arxiv.org/pdf/2606.27511v1)

**作者:** Chenqing Zhu `[一作]` (Southeast University), Songze Li `[通讯]` (Southeast University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对联邦学习中的LLM问答系统，论文设计了一个无需访问客户端数据的后门注入方法，利用梯度反演从上传梯度中重建训练样本并在发布前对全局模型进行微调，植入广告式后门。

**💡 创新点**

创新点在于将梯度反演与两阶段注入结合，证明仅从上传梯度即可恢复足够的分布信息，并实现近乎100%的攻击成功率，同时保持清洁任务性能。

**🔧 技术方法**

使用的技术包括梯度反演、低秩投影子空间恢复、基于LLM（GPT‑4.1、DeepSeek‑V3）的语料重构、LoRA与全微调、以及后门注入的BadNets式微调。

**📊 数据集**

实验采用医疗、心理健康和法律问答领域的公开数据集（如MedQA、HealthQA、LegalQA）以及四种7B–8B规模的LLM模型（LLaMA‑3.1‑8B、Qwen3‑8B、Mistral‑7B、Command‑R‑7B）。

**📈 对比分析**

与无后门的联邦训练以及在分布上已知数据的上界实验相比，攻击在所有模型和数据集上均实现了≈100%的ASR，并且在ROUGE‑L、BERTScore和LLM‑Eval指标上几乎无明显下降，甚至偶尔提升。

**⚠️ 局限性**

局限性包括对梯度反演比例的依赖（至少5–20%）、对辅助LLM的算力和API成本、对固定触发词的鲁棒性有限、以及对差分隐私或安全聚合等防御手段仍有一定抵抗性。

---

## 64. Cluster, Route, Escalate: Cascaded Framework for Cost-Aware LLM Serving

**arXiv ID:** 2606.27457 | [PDF](https://arxiv.org/pdf/2606.27457v1)

**作者:** Yasmin Moslem `[一作]` (Trinity College Dublin), John D. Kelleher `[通讯]` (Trinity College Dublin)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种两阶段级联路由系统，先基于语义聚类分配成本效益模型，再通过质量估计决定是否提升到更强模型，以在生产环境中高效部署大型语言模型。

**💡 创新点**

创新点在于利用单个可解释的 λ 超参数在训练时自动满足 TPOT 预算，实现成本感知路由；同时仅使用任务正确性标签训练轻量级 QE 级联，恢复大部分准确率而无需额外标注。

**🔧 技术方法**

采用 k‑means 语义聚类、基于误差+成本的路由评分、Pareto 分析剔除被支配模型、Fine‑tuned ModernBERT 作为二分类质量估计器，以及离线 λ 的自动优化。

**📊 数据集**

在 AIME 2024 数学推理数据集和 TeleQnA 电信多选问答基准上进行评估。

**📈 对比分析**

与单模型基线对比，Stage‑1+2 在 AIME 保留 97–99% 最强模型准确率，TPOT 降低 18%；在 TeleQnA 同样在准确率与 TPOT 上接近最强模型，准确率提升约 3%，TPOT 降低 4–5 ms。

**⚠️ 局限性**

局限性包括：聚类和路由表需离线重建，无法实时适应查询分布漂移；仅以 TPOT 为成本度量，未考虑排队、吞吐量等指标；所有候选模型必须同时驻留 GPU，受显存限制。

---

## 65. PairSAE: Mechanistic Interpretability from Pair Representations in Protein Co-Folding

**arXiv ID:** 2606.27440 | [PDF](https://arxiv.org/pdf/2606.27440v1)

**作者:** Giosue Migliorini `[一作]` (University of California), Olivia Viessmann `[通讯]` (Flagship Pioneering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种名为 PairSAE 的稀疏自编码器，能够同时重建 AlphaFold/ Boltz-2 等结构预测模型中的序列嵌入和配对张量。

**💡 创新点**

创新点在于使用 N‑mode SVD 将配对张量压缩为 token 级交互摘要，然后共享稀疏特征同时解码到序列与配对空间，解决了传统 SAE 在 pairformer 结构中特征爆炸和分布混杂的问题。

**🔧 技术方法**

主要技术包括 N‑mode SVD、稀疏自编码器（BatchTopK + ReLU）、Matryoshka 损失、线性探针、LASSO 预测绑定亲和力。

**📊 数据集**

使用 Boltz‑2 在 PLINDER 记录的蛋白‑配体复合物的中间层（第 33、64 层）作为输入，评估其对 UniProt 注释和亲和力的可解释性。

**📈 对比分析**

与单层 ESM‑2‑650M 神经元相比，PairSAE 在 token 与复合体级别的 F1 ≥ 0.5 概念覆盖率分别提升至约 29%/53%，并能用 291 个特征在 R3‑L64 上预测亲和力，R² ≈ 0.53。

**⚠️ 局限性**

主要局限是缺乏配体层面的概念映射，无法将激活的稀疏特征与具体配体特征关联，并且在高亲和力样本上预测误差增大。

---

## 66. EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction

**arXiv ID:** 2606.27550 | [PDF](https://arxiv.org/pdf/2606.27550v1)

**作者:** Carrie Chen `[一作]` `[通讯]`, Carrie Chen

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种无训练的动态多token预测调度器EntMTP，在LLM推理过程中根据局部熵动态切换树形注意力拓扑，以提升多token预测的吞吐量。

**💡 创新点**

将多token预测的树拓扑视为任务特定且可动态切换，构建Pareto前沿树库并通过熵引导的实时调度，实现对不同上下文预测深度的自适应匹配。

**🔧 技术方法**

利用Hydra和Medusa多头预测框架，预编译多种树形拓扑，使用EAGLE‑2路径值/熵特征做调度阈值，实施二进制或多级阈值策略，保持无训练、无额外推理成本。

**📊 数据集**

在HumanEval、GSM8K、ShareGPT三个基准上评估，使用100条prompt测试集，并用100条校准集来选阈值。

**📈 对比分析**

与Vanilla Vicuna、Medusa默认、Hydra默认进行对比；EntMTP^*在每个基准上提升7–13%速度，EntMTP^τ进一步提升0.5–2.1%，整体在HumanEval 3.26×、GSM8K 3.13×、ShareGPT 3.47×速率，保持近似无损的perplexity。

**⚠️ 局限性**

依赖任务特定的树前沿，需要离线搜索；仅在单线程(batch=1)上验证；对动态阈值调参仍需校准；在高并发或大批量推理中的开销未评估。

---

## 67. Quantum Generative Diffusion Model for Real-World Time Series

**arXiv ID:** 2606.27561 | [PDF](https://arxiv.org/pdf/2606.27561v1)

**作者:** Jack Waller `[一作]` (Kingston University London), Xing Liang `[通讯]` (Kingston University London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 QDiffusion-TS，一种用于金融时间序列合成的量子生成扩散模型，并在 IQM 量子处理器上进行验证。

**💡 创新点**

创新点在于将量子神经网络（QNN）完整替代传统扩散模型中的前馈网络，实现每个组件参数量降低近三阶，并通过量子增强的变压器在保持甚至提升生成质量的同时显著提升参数效率。

**🔧 技术方法**

采用了量子生成扩散模型、参数化量子电路、混合量子变压器、振幅/角度编码、傅里叶损失、Wasserstein 与 KS 距离以及 RMSE、R² 等评价指标。

**📊 数据集**

使用了来自 Yahoo Finance 的 Apple 与 Amazon 日交易价格（开盘价、收盘价、最高价、最低价、成交量）数据，时间跨度为 2021‑2026 年，序列长度为 256 步（硬件测试时压缩为 32 步）。

**📈 对比分析**

通过对比真实与合成的对数收益分布（MAE、Wasserstein、KS），量子模型平均降低约 44% Wasserstein 距离；在 BiLSTM 下游预测任务中，合成数据提升 RMSE 最高可达 71%；硬件实现的生成质量与仿真相当，甚至略优。

**⚠️ 局限性**

局限性包括：受限于当前 9 比特量子设备和序列长度限制，混合模型仍以经典组件为主；在极少数据场景下缺乏显著优势；硬件噪声虽未显著损害性能，但未被充分利用，且整体性能受训练规模和量子电路深度的影响。

---

## 68. Narrative-UFET: Narrative Generation for Ultra-Fine Entity Typing

**arXiv ID:** 2606.27598 | [PDF](https://arxiv.org/pdf/2606.27598v1)

**作者:** Mreedul Gupta `[一作]` (University of Colorado Boulder), Maria Leonor Pacheco `[通讯]` (University of Colorado Boulder)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建 Narrative-UFET 数据集，将每个实体-句子对扩展为包含目标实体的短篇连贯叙事，并设计两种变体：类型保持（Maintain）与类型变化（Change）。随后用掩码与因果语言模型评估叙事上下文对超细粒度实体类型（UFET）的影响。

**💡 创新点**

首次通过合成叙事控制话语属性，验证多句子上下文能显著提升长尾实体的细粒度分类；提出保持/变化两种类型变化设计；证明合成叙事在提供隐藏信息方面优于自然文本。

**🔧 技术方法**

使用大型生成式语言模型 Qwen3‑32B 生成叙事；采用 TinyStories、对齐度和共指密度等自动指标及人工评估验证叙事质量；在实体类型任务中应用 BERT（掩码语言模型）以及 Llama3.3‑70B、Qwen3‑32B（因果语言模型）。

**📊 数据集**

基于 UFET（5,994 个实体-句子对）crowd‑annotated 版本扩展为 Narrative‑UFET；对比 OntoNotes 5.0 的自然上下文。

**📈 对比分析**

按实体出现频率分成四个箱子，使用 F1 评估。Narrative‑UFET‑Change 在所有箱子均优于句子级基线和 OntoNotes 上下文；因果模型总体优于掩码模型，Qwen 在长尾箱子上的提升尤为显著。

**⚠️ 局限性**

仅研究了类型保持/变化一种话语属性；合成叙事可能偏离自然分布；未充分控制词数、词汇多样性等混杂因素；实验仅覆盖 BERT、Llama3.3‑70B、Qwen3‑32B，单语言单任务；使用量化模型可能影响结果；模型选择阶段的评估缺乏严格指南。

---

## 69. Forecasting Technological Directions in Wireless Networks and Mobile Computing via AutoML Framework

**arXiv ID:** 2606.27394 | [PDF](https://arxiv.org/pdf/2606.27394v1)

**作者:** Ahmed Abolfadl `[一作]` (German University in Cairo), Maggie Mashaly `[通讯]` (German University in Cairo)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个自动化的 AutoML 框架，用于预测无线网络与移动计算领域的科研趋势。

**💡 创新点**

创新点在于将聚类、主题建模与时间序列预测通过元学习与自适应搜索完全自动化，并引入 LLM 辅助主题标签化，提升解释性。

**🔧 技术方法**

使用 SPECTER 文档嵌入、AutoCluster（基于元特征的聚类算法选择）、AutoTopicModeling（多模型顺序加速、Optuna 超参数搜索）、AutoTrendAnalysis（ARIMA/Prophet/STM/ LSTM 时序预测）等技术。

**📊 数据集**

数据集为 2010‑2024 年 127,820 篇高影响力期刊与会议摘要，来源于 Scopus，经过预处理后使用 768 维嵌入降维至低维表示。

**📈 对比分析**

与传统手工或固定模型方法比较，最佳模型 STL 在所有主题上实现 RMSE 36.76，显示出较高预测精度，且通过成功率与一致性检验验证了聚类与主题模型的有效性。

**⚠️ 局限性**

局限性包括使用静态 SPECTER 嵌入无法捕捉语义漂移，LLM 生成标签可能产生不一致或歧义，且预测仅覆盖短期趋势，未建模长期不确定性。

---

## 70. ReWorld: Learning Better Representations for World Action Models

**arXiv ID:** 2606.27504 | [PDF](https://arxiv.org/pdf/2606.27504v1)

**作者:** Tianze Xia `[一作]` (Huazhong University of Science and Technology), Xinggang Wang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了ReWorld框架，对世界动作模型（WAM）的中间表示进行直接优化，显著提升视频生成与闭环规划性能。

**💡 创新点**

创新点在于三种表示学习机制：①未来预测监督和自引导的Video DiT中间层；②跨模态对齐的Action DiT；③硬负样本惩罚的安全感知规划；全部在模型自身潜在空间完成，省去外部编码器，计算开销极低。

**🔧 技术方法**

采用Video DiT + Action DiT、Flow‑Matching 训练、轻量预测头、余弦对齐、硬负样本 RDE、self‑guidance 以及 VAE 编码的端到端实现。

**📊 数据集**

使用 nuScenes 与 nuPlan 进行视频训练，NAVSIM 进行闭环仿真评估。

**📈 对比分析**

与 DriveLaW 及多种世界模型在 FVD（nuScenes）和 PDMS（NAVSIM）等指标对比，FVD 从 81.3 降到 61.9（下降 23.9%），PDMS 从 89.1 提升至 90.4（+1.3 点），训练收敛速度约提升 2×，且无需 RL 或后处理。

**⚠️ 局限性**

局限性包括：仅在模拟环境和特定数据集验证，缺乏在更复杂或真实车辆部署中的评估；对极端天气、长时序预测的鲁棒性尚待进一步验证；并未探究多任务或多模态扩展的可行性。

---

## 71. Advancing Speaker-Based Vocal Effort Classification with WavLM and Data Augmentation in Naturalistic Non-Calibrated Speech Recordings

**arXiv ID:** 2606.27543 | [PDF](https://arxiv.org/pdf/2606.27543v1)

**作者:** Zahra Omidi `[一作]` (University of Texas at Dallas), John H. L. Hansen `[通讯]` (University of Texas at Dallas)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在AVID语料库上细调WavLM-Base，结合多种波形级数据增强和高斯邻域软标签，对语音的不同发声强度进行分类，并与wav2vec2和HuBERT做对比。

**💡 创新点**

首次将WavLM应用于语音强度分类，并系统评估了RIR、时间遮挡、频带限制等传统增强以及MixUp/CutMix的效果；另外提出高斯邻域软标签来建模发声强度的连续性，显著降低边界误差。

**🔧 技术方法**

使用自监督学习（SSL）预训练模型Fine‑tune、波形级数据增强（RIR卷积、噪声、时间遮挡、速度扰动、频带限制、MixUp、CutMix）、逐层解冻训练、KL散度软标签训练。

**📊 数据集**

使用AVID（非标定）语料库，共10,000条英语语音，包含四个发声强度标签（soft、normal、loud、very loud）。

**📈 对比分析**

采用10折组交叉验证进行评估。WavLM-Base在不做增强时达到75.24%准确率，优于wav2vec2（67.58%）和HuBERT（74.13%）。在最佳配置（MixUpα=0.6 + 高斯邻域软标签）下，平均准确率升至78.22%，是该语料库的最佳性能。

**⚠️ 局限性**

受限于数据量少且主要为实验室录制，模型仍在相邻标签间出现混淆；缺乏对自然环境下多说话人或团队对话的验证。

---

## 72. Incremental Dominating Set

**arXiv ID:** 2606.27469 | [PDF](https://arxiv.org/pdf/2606.27469v1)

**作者:** Ilan Doron Arad `[一作]` (Massachusetts Institute of Technology), Seffi Naor `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究增量主导集（Incremental Dominating Set）问题，给出了针对增量/在线情境的确定性和随机化算法，并将方法推广至连通主导集（Connected Dominating Set）。

**💡 创新点**

创新点在于：①首次为增量主导集设计了O(Δ)竞争比的确定性算法和O(log²Δ)竞争比的随机化算法；②提出了一种通过线性规划捕获连通性的局部约束框架；③给出了匹配的下界，证明这些竞争比在常数因子上是最优的。

**🔧 技术方法**

主要技术包括竞争分析、随机化选择策略、线性规划建模（通过局部约束保证连通性）以及对照离线最优的理论证明。

**📊 数据集**

文中未使用具体实验数据集，研究基于理论分析与证明。

**📈 对比分析**

与离线最优全知算法进行竞争比对比。确定性算法在最坏情况下竞争比为O(Δ)，随机化算法为O(log²Δ)。在连通主导集中，当每个新到达顶点的邻域事先已知时，亦可实现类似的polylog级竞争比。

**⚠️ 局限性**

局限性包括：①仅适用于增量/在线模型，未考虑撤销或更新；②随机化算法虽有更好的竞争比，但实现复杂；③对连通主导集的结果需要事先知道邻域信息；④缺乏实验验证，未评估在真实网络上的具体性能。

---

## 73. Learning in Markovian bandits with non-observable states and constrained decision epochs

**arXiv ID:** 2606.27448 | [PDF](https://arxiv.org/pdf/2606.27448v1)

**作者:** Thomas Hira `[一作]` (IRIT), Ina Maria Verloop `[通讯]` (University Of Basque Country)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在非可观测状态下的马尔可夫赌博机（bandits）中，基于纯策略（pure policy）的收益比较与调度问题，提出了自衰退（self‑degrading）马尔可夫赌博机模型并探讨了罕换策略的遗憾界限；

**💡 创新点**

在自衰退模型下证明了纯策略在长期内是最优的，并揭示了不具备先验信息时，低频切换算法无法达到对数级遗憾，而通过设计的乐观算法实现近对数遗憾，并在已知偏差跨度的先验条件下实现对数级及平方根对数级的最坏情况遗憾；

**🔧 技术方法**

使用了基于马尔可夫链理论的偏差函数分析、乐观（optimistic）算法框架以及罕换策略的稀疏性假设，结合理论下界与上界证明；

**📊 数据集**

未使用特定数据集，本文以理论分析为主；

**📈 对比分析**

与传统的随机（stochastic）赌博机对比，本文表明在马尔可夫赌博机中即使采用近对数遗憾的算法，仍无法达到对数级遗憾，体现了非可观测状态带来的挑战；

**⚠️ 局限性**

主要限制在于假设状态空间可能无限大且仅考虑了自衰退模型，未探讨已知状态空间、状态簇或非被激活臂产生奖励的情况。

---

## 74. Internalizing the Future: A Unified Agentic Training Paradigm for World Model Planning

**arXiv ID:** 2606.27483 | [PDF](https://arxiv.org/pdf/2606.27483v1)

**作者:** Xuan Zhang `[一作]` (Fudan University), Yuan Qi `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过三阶段训练（WM-AMT、FE-SFT、FC-RL）将未来感知与规划能力内化到单一自回归LLM策略中，形成能在决策前生成未来情景和成功估计的“世界模型”模块。

**💡 创新点**

①识别并克服“格式-能力”缺口，证明仅后训练无法获得真正的预测能力；②首次将世界模型训练与策略训练统一在同一LLM中，并通过语言化的Q值实现可解释、可校准的内部规划；③构建了以未来状态摘要为训练信号的WM-AMT中间训练。

**🔧 技术方法**

使用自回归语言模型（Youtu-LLM-2B）、中间训练、格式化监督微调、基于组的强化学习（GRPO）以及基于Brier分数的校准奖励；同时采用词嵌入检索（E5）和基于知识的检索。

**📊 数据集**

搜索推理：NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle；数学推理：AIME 2024/25/26。

**📈 对比分析**

与多种后训练/中间训练/状态预测基线（Post-Training Only、State-Only Prediction、IWM）以及现有世界模型方法对比，实验显示在七大搜索QA和AIME数学推理任务中，平均分和通过率均提升约3-5%，在多跳推理集上提升更为显著。

**⚠️ 局限性**

仍依赖大规模LLM的推理能力，训练过程对计算资源要求高；在极长序列或高度不确定的环境中仍可能出现误报或过度自信的预测；当前框架主要针对文本推理任务，需进一步验证在更复杂多模态或真实环境中的可迁移性。

---

## 75. A Quantum Method of Types

**arXiv ID:** 2606.27442 | [PDF](https://arxiv.org/pdf/2606.27442v1)

**作者:** Arick Grootveld `[一作]` `[通讯]` (Syracuse University), Arick Grootveld (Syracuse University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了量子方法的类型，将经典经验分布推广为量子经验算子，并给出了其组合学和大偏差界；

**💡 创新点**

创新点在于使用精确单元设计离散化 Keyl 量子测量，构造有限输出POVM，使经验算子数量多项式增长，并实现量子Sanov定理和复合量子假设检验的可实现性；

**🔧 技术方法**

技术手段包括Schur-Weyl 双线性分解、精确单元设计、逆相对熵（reverse relative entropy）的大偏差分析以及对态估计的有限测量构造；

**📊 数据集**

该工作完全为理论分析，不涉及具体实验数据集；

**📈 对比分析**

由于是理论构造，未进行实验对比，主要通过证明量子逆相对熵可作为错误指数来展示方法的有效性；

**⚠️ 局限性**

局限性在于逆相对熵不一定是所有假设检验问题的最优 Stein 指数，且该方法仅适用于特定复合假设检验情形，无法覆盖一般量子 Stein 定理。

---

## 76. Position: The Term "Machine Unlearning" Is Overused in LLMs

**arXiv ID:** 2606.27379 | [PDF](https://arxiv.org/pdf/2606.27379v1)

**作者:** Sangyeon Yoon `[一作]` (Yonsei University), Albert No `[通讯]` (Yonsei University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过概念分析和案例讨论，提出应将“机器未学习”严格限定为数据集定义的删除，即在已知忘记集F的情况下，生成的模型与在D∖F上重新训练的模型在行为上近似不可区分；同时对在LLM研究中广泛使用的“未学习”做意图分类，强调不同目标需不同术语和基准；

**💡 创新点**

创新点在于：①将机器未学习与其他“忘记”目标区分，提出以重训练可比性为核心的正式定义；②构建了基于意图的分类体系，厘清不同方法的目标和保证；③提出评估必须以重训练模型为参照，并引入“派生能力”探测作为评测补充，避免仅靠输出失败指标的误判；

**🔧 技术方法**

技术手段主要为理论定义与推理（如近似可区分性、两样本检验、成员推断攻击等），以及对现有评测框架（TOFU、MUSE、RWKU、WMDP）的分析；

**📊 数据集**

论文并未使用具体数据集进行实验，而是以通用案例（如合成推理轨迹、数据投毒样本）说明概念；

**📈 对比分析**

由于缺乏实验，文中未给出方法对比或性能指标，主要通过案例论证评测失效与必要性；

**⚠️ 局限性**

局限性包括：①概念性讨论缺乏实证验证；②对重训练参考模型可行性的假设在实际大型模型中可能受限；③分类体系仍为高层次，实际边界可能模糊；

---

## 77. Operator Learning for Cubic Nonlinear Schrödinger Equation on Periodic Domains

**arXiv ID:** 2606.27459 | [PDF](https://arxiv.org/pdf/2606.27459v1)

**作者:** Emmanuel E. Oguadimma `[一作]` (Oregon State University), Xueying Yu `[通讯]` (Oregon State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了二维立方非摊薄非线性薛定谔方程在不同方面比的平面上，用几何条件的傅里叶神经算子学习其一阶解算子，并验证能捕捉不同几何下的能量传递特征。

**💡 创新点**

将几何参数（ω²）作为额外输入通道，使同一网络能够区分有理与无理圆环的共振结构，并展示该条件化可显著提升长期预测精度。

**🔧 技术方法**

采用 Fourier Neural Operator（FNO）架构，包含 Fourier 层、GELU/Sigmoid 激活、显式几何通道，并通过 Adam 优化训练一阶解算子。

**📊 数据集**

使用随机相位初始数据在 ω²=1（有理）和 ω²=√2（无理）平面上生成 1200 条轨迹，参考解由 Fourier pseudospectral 方法产生，按 800/200/200 划分训练/验证/测试。

**📈 对比分析**

通过相对 L² 误差、Sobolev H² 范数增长曲线与理论对比，学习算子在两几何下均保持约 10⁻² 级误差，并在无理几何下成功捕获更弱的能量级增长；与无条件模型相比误差显著下降。

**⚠️ 局限性**

仅在固定分辨率与有限预测时长内验证，未测试更长时间尺度、不同网格或连续空间 (ℝ×𝕋²) 的泛化；模型对训练数据质量与数值解耦的鲁棒性有限。

---

## 78. Retroactive Advantage Correction: Closed-Form V-Trace Bias Correction for Delay-Aware RLHF

**arXiv ID:** 2606.27580 | [PDF](https://arxiv.org/pdf/2606.27580v1)

**作者:** Arnav Raj `[一作]` `[通讯]` (Indian Institute of Technology Delhi), Arnav Raj (Indian Institute of Technology Delhi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 Retroactive Advantage Correction (RAC)，一种在 RLHF 中处理奖励延迟的前向注入技术，改进 PPO/GRPO 的优势估计。

**💡 创新点**

创新点在于使用非负年龄衰减核和 V‑trace 剪切重要性比率，将慢速奖励作为优势的正向修正，并给出闭式累计无偏性公式，能够在多通道延迟情况下保持无偏性并恢复 V‑trace 的在政策保证。

**🔧 技术方法**

采用了 V‑trace 剪切重要性采样、年龄衰减核、行随机延迟核、总变差与 Pinsker / Bretagnolle‑Huber 上界等技术。

**📊 数据集**

实验使用 3×2 维度的表格 MDP 进行闭式偏差验证，并在 7B 规模的 Qwen2.5‑7B（快通道）与 Skywork‑Llama‑3.1‑8B（慢通道）奖励分布上做了机器精度检验。

**📈 对比分析**

与原始 PPO、wait‑for‑slow、Retrace‑A 等方法对比，K=2 时闭式策略偏差降低 47.9 倍，成本保持与原 PPO 相同；在 7B 规模下与 V‑trace 等保持零误差，显示出优异的成本‑质量平衡。

**⚠️ 局限性**

局限性包括：验证仅在表格 MDP 和 7B 规模奖励分布上进行，缺乏在更高维、复杂环境中的实验；年龄核和延迟分布的估计依赖于系统假设，实际部署时可能需要额外调优。

---

## 79. How is Latin America engaging with responsible metrics? A systematic review comparing regional and global scientific production

**arXiv ID:** 2606.27395 | [PDF](https://arxiv.org/pdf/2606.27395v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 80. Tessellating The Earth

**arXiv ID:** 2606.27514 | [PDF](https://arxiv.org/pdf/2606.27514v1)

**作者:** Daniel Cher `[一作]` (Washington University in St. Louis), Nathan Jacobs `[通讯]` (Washington University in St. Louis)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于可学习球面Voronoi分割的地理位置编码器，并加入全局语义标记实现跨模态对齐；

**💡 创新点**

创新点在于：①将固定的球面基函数替换为可学习的Voronoi分区，使编码器能自适应地理复杂度；②引入全局语义标记，让不同地区的相似环境通过共享概念标记实现语义共享；

**🔧 技术方法**

采用的技术包括：可微分球面Voronoi分割、对比学习预训练、语义标记的注意力机制、重建和对齐损失以及软赋值权重等；

**📊 数据集**

使用的数据集包括全球Sentinel‑2卫星影像、WWF生态区、Biome、Country分类标签、气温、海拔、人口密度、加州住房价格以及iNaturalist‑2018的细粒度物种数据；

**📈 对比分析**

与已有方法（GeoCLIP、SatCLIP、CSP等）在一系列地理分类与回归基准上进行对比，TTE在平均分类精度上提升至80.0%（比前一最佳高2.8%），平均R²提升至0.777（高0.045），并在iNaturalist‑2018的Top‑k准确率上显著优于其他地理先验方法；

**⚠️ 局限性**

局限性包括：仅使用单时段的Sentinel‑2影像，缺乏时间维度；未整合地面照片或其他环境属性；对某些任务（如EcoRegion、加州住房）因数据与影像时空不匹配表现不如特定方法；

---

## 81. Masked Language Flow Models

**arXiv ID:** 2606.27617 | [PDF](https://arxiv.org/pdf/2606.27617v1)

**作者:** Iskander Azangulov `[一作]` (University of Oxford), Patrick Rebeschini `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Masked Language Flow Models（MLFMs），结合掩码扩散模型和流模型，实现任意位置的条件生成和多步推理，并提供了新的采样器；

**💡 创新点**

创新点在于用 Brownian bridge 把部分掩码序列与完整序列连接，分离掩码与连续去噪；实现了预训练 MDM 的轻量迁移；提出在线 Token Promotion 与上下文腐蚀的 classifier‑free guidance 结合的采样策略；

**🔧 技术方法**

使用 LangFlow 流模型、Brownian bridge、DDPM、classifier‑free guidance、LoRA、AdaLN、token embedding、流图蒸馏等技术；

**📊 数据集**

预训练使用 SlimPajama（LLaMA‑2 tokenizer），SFT 训练混合 ShareGPT、NuminaMath‑CoT、GSM8K‑Aug‑NL、MetaMathQA、OpenCodeInstruct；评估在 GSM8K 与 MT‑Bench 上；

**📈 对比分析**

与 1.1B SMDM 及同规模 AR 模型对比；在 MT‑Bench 上得到 2.27 分，显著高于 1.60（SMDM）和 1.57（AR）；在 GSM8K 上得到 31.24% 的准确率，低于 58.5%（SMDM）和 58.6%（LLaMA‑2）；使用 128 步采样仍优于 SMDM 的 256 步；

**⚠️ 局限性**

局限性：在数学推理任务（GSM8K）表现仍不足，需更精细的任务专门微调；在线 Token Promotion 可能因早期错误导致偏差；模型规模虽大但相比部分 1.7B 级模型仍有提升空间；需要进一步验证多步推理的鲁棒性。

---

## 82. CalBrief: A Pilot Diagnostic Benchmark for Evidence-Calibrated Scientific Briefing with Large Language Models

**arXiv ID:** 2606.27383 | [PDF](https://arxiv.org/pdf/2606.27383v1)

**作者:** Yu Fu `[一作]` (Sichuan University), Yong Zhao `[通讯]` (Sichuan University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并构建了一个人工验证的科学证据包装基准，用于评估大型语言模型在生成包级结论时的证据校准能力，并提出并验证了CalBrief诊断框架。

**💡 创新点**

提出证据校准科学简报任务，创建可验证基准，并通过实验分解保守性来源，发现标签空间扩展是主要因素；同时证明后置二值化可恢复或超过直接二值提示的性能。

**🔧 技术方法**

采用CalBrief框架（角色识别、缺口分析图、强度校准策略），使用多种LLM（GPT‑4o、Claude Sonnet、Gemini Flash、Kimi‑K2.6、GLM‑5.1、Qwen3‑32B），进行公平模式评估、直接提示与结构化管线对比，并统计角色准确率、缺口相关性、强度匹配和宏F1等指标。

**📊 数据集**

构造了16个不同领域的科学证据包装（共96条包级摘要），涵盖检索评估、LLM代理基准、科学NLP任务、安全/可信度和多文档问答等，所有摘要和标签均经过人工双重验证。

**📈 对比分析**

与“始终中等”基准和直接简报基线比较，结构化管线在角色和缺口上显著提升，但强度校准性能远低于直接提示；通过拆分标签空间、信号注入、策略三因子，约63%保守性来自标签空间扩展；后置二值化后宏F1可恢复或超过直接二值提示。

**⚠️ 局限性**

限制包括：角色识别准确率仍偏低（0.46）；细粒度缺口/不匹配标签人工一致性低；后置二值化效果模型依赖；基准规模有限，未覆盖更细粒度标签；系统在高风险领域的可靠性仍需进一步验证。

---

## 83. PEBS: Per-rater Empirical-Bayes Shrinkage for RLHF Reward-Model Calibration

**arXiv ID:** 2606.27578 | [PDF](https://arxiv.org/pdf/2606.27578v1)

**作者:** Arnav Raj `[一作]` `[通讯]` (Indian Institute of Technology Delhi), Arnav Raj (Indian Institute of Technology Delhi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种后置经验贝叶斯收缩校准方法PEBS，用于修正RLHF奖励模型在不同评估者尺度上的偏差。

**💡 创新点**

创新点在于将Efron–Morris–James–Stein经验贝叶斯收缩应用于每个评估者的线性标定，得到闭式后置校准器，无需重训练奖励模型，并显著提升用户级RMSE。

**🔧 技术方法**

采用经验贝叶斯收缩（Morris–James–Stein）、OLS线性校准、样本分割的方差估计、混合模型理论、oracle不等式以及Morris g‑函数预测等技术。

**📊 数据集**

使用了PRISM、PluriHarms、HelpSteer2、OASST2、SHP‑subreddit以及MultiPref等公开的多种偏好与评分数据集。

**📈 对比分析**

通过与无校准、全局斜率、单用户OLS等四种设置对比，PEBS在PRISM上实现了8.58%的用户级RMSE下降，跨语料平均提升7–10%，并带来5.7%的BT‑NLL改善，在多种基模型与跨家族转移实验中保持正向效果。

**⚠️ 局限性**

局限性包括：仅适用于满足高斯随机效应假设的连续评分数据，对顺序偏好类如MultiPref无效；跨模型转移受架构与头部交互影响；少量评分的评估者冷启动需要至少5条记录；对少数群体评估者可能产生偏向性收缩。

---

## 84. Training Observable Control Policies to Expose Agent State Through Actions

**arXiv ID:** 2606.27609 | [PDF](https://arxiv.org/pdf/2606.27609v1)

**作者:** Andres Enriquez Fernandez `[一作]` (University of Texas at El Paso), John J. Bird `[通讯]` (University of Texas at El Paso)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

使用强化学习训练固定翼无人机的跟踪控制策略，并在奖励函数中嵌入UKF估计器的性能指标，以提升基于动作观测的状态估计可观测性。

**💡 创新点**

创新点在于将估计器性能直接作为奖励项融入策略训练，使得生成的控制策略在不显著降低任务完成度的前提下显著提升可观测性和估计精度。

**🔧 技术方法**

采用Soft Actor‑Critic（SAC）强化学习、PyTorch实现的神经网络策略、Unscented Kalman Filter（UKF）估计器，以及线性化可观测矩阵与删减可观测矩阵（SOM）的奇异值分析。

**📊 数据集**

使用基于OpenAI Gym的自定义二维固定翼飞行仿真环境，随机初始化机身位置、速度和航向，模拟约14,500个试验。

**📈 对比分析**

通过与仅基于任务奖励训练的策略比较，嵌入估计器奖励的策略在中位数时位置误差由约3.85 m降至1.29 m（提升66%），速度误差提升约79%，且对跟踪奖励仅低约1%；同时估计器发散率显著下降。

**⚠️ 局限性**

局限性包括：仅在仿真环境中验证，未涉及真实飞行数据；可观测性在任意时刻仍不完全；对多智能体协同和人机交互的理论与实验验证不足；且策略空间复杂，可能需要更高的计算或训练成本。

---

## 85. SelectAnyTree: A Promptable Instance Segmentation Model for 3D Forest LiDAR Point Clouds

**arXiv ID:** 2606.27491 | [PDF](https://arxiv.org/pdf/2606.27491v1)

**作者:** Trung Thanh Nguyen `[一作]` (Nagoya University), Teja Kattenborn `[通讯]` (University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一种可点击的森林 LiDAR 点云实例分割模型 SelectAnyTree，可让用户仅用几次点击即可分割任意单棵树。

**💡 创新点**

将点击转为单一查询向量的 Prompt Encoder；使用 CHM 棵顶作为几何引导的首选点击；统一状态空间查询解码器实现高效多树选择。

**🔧 技术方法**

稀疏体素编码 + 线性状态空间解码器；随机傅里叶位置编码 + 圆柱池化；共享点击模拟滚动训练。

**📊 数据集**

FOR-instanceV2（7个地区共11134棵树）与未见 LAUTx（516棵树）。

**📈 对比分析**

与 AGILE3D、NPISeg3D、Point‑SAM、PartSAM 等提示模型以及 TreeLearn、ForestFormer3D、ForestMamba 等自动模型对比；单次点击 IoU 78.2（比 Point‑SAM 高 24.8），在点击数和速度上均优于基线，且模型参数 19.4M，速度最快。

**⚠️ 局限性**

对 CHM 的依赖在密集多层冠层下易失效，固定的全景尺度参数不适用于所有森林类型；在 LAUTx 上去除 CHM 反而更好。

---

## 86. Unbent collections of non-planar $s$-grid-drawing

**arXiv ID:** 2606.27501 | [PDF](https://arxiv.org/pdf/2606.27501v1)

**作者:** Therese Biedl `[一作]` `[通讯]` (University of Waterloo), Therese Biedl (University of Waterloo)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在非平面图中为4-图和2-图构造无弯折的多幅绘图集合，证明在非平面情况下两幅绘图足以覆盖所有边，而平面情况下需要三幅绘图。

**💡 创新点**

将Antić等人关于平面正交绘图的无弯折集合扩展到非平面以及任意s-网格绘图，首次证明两幅绘图就能满足非平面4-图的无弯折要求，并推广到s-网格与严格岛松树。

**🔧 技术方法**

利用边分割、伪森林/树枝分解、伪森林与伪树构造的无弯折绘图，并通过缩放与网格线插入技术完成缺失边的添加。

**📊 数据集**

未使用具体实验数据集，主要以理论图（如四面体、八面体、K7、K2,3）作为示例说明。

**📈 对比分析**

未给出实验比较，理论证明表明所需绘图数最优（2或3），且添加边的规模可控，复杂度为多项式。

**⚠️ 局限性**

对s-网格绘图的尺寸控制未知，平面绘图需三幅而非两幅，且未解决是否可将严格岛松树替换为伪森林，或在更高阶图中进一步降低绘图数。

---

## 87. hia-gat: A Heterogeneous Interaction-Aware Graph Attention Network For Frame-Level Traffic Conflict Risk Prediction On Freeways

**arXiv ID:** 2606.27577 | [PDF](https://arxiv.org/pdf/2606.27577v1)

**作者:** Mahshid Malazizi `[一作]` (University of Tennessee at Chattanooga), Hoang H. Nguyen `[通讯]` (University of Tennessee at Chattanooga)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对高速公路视频帧进行风险评估，将每帧标记为危险或安全，并基于交通安全指标（TTC、PET）构建图模型进行预测。

**💡 创新点**

提出 HIA-GAT：双流异构图注意网络，分别处理纵向（追尾）和横向（变道）交互，并通过冲突类型感知门控机制与事件级门控监督实现对冲突类型的可解释分离。

**🔧 技术方法**

使用图注意网络（GAT）、双流设计、物理信息边特征（闭合速率、重叠度、侧向速度）、门控机制、事件级门监督、交叉熵+门损失的联合训练。

**📊 数据集**

基准数据集为 NGSIM I‑80 与 US‑101 的高频轨迹数据（约 4.6M 与 4.1M 车辆‑帧样本）。

**📈 对比分析**

与七类基线（非图模型、GCN、GraphSAGE、HomoGAT）在九种阈值设置下比较，HIA‑GAT 在两条高速公路上平均 AUC 分别为 0.835 与 0.867，显著优于所有对照模型；在 PET（侧向冲突）场景下提升约 0.10 AUC，随机森林在 F1 上略优。

**⚠️ 局限性**

局限性：在极少数样本的阈值下训练不稳定；门控监督仅覆盖 0.1–1.5% 的节点，且门控分流可能导致正样本稀疏；模型在其他道路类型或不同交通环境下的泛化尚待验证。

---

## 88. Doppler Tracking of the Artemis II Mission (and Other Spacecraft)

**arXiv ID:** 2606.27531 | [PDF](https://arxiv.org/pdf/2606.27531v1)

**作者:** Ankur Purao `[一作]` (American University), Michael Robinson `[通讯]` (American University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用本科生团队搭建2.4 m抛物面天线和HackRF SDR系统，开展Artemis II（Orion）探测器的S‑band Doppler跟踪实验，收集并处理数据，验证并改进了针对QPSK信号的非线性Doppler估计算法。

**💡 创新点**

创新点在于：① 将QPSK调制符号的四次幂用于去除信息并保留Doppler信号；② 使用GPU批处理实现高速频率扫描；③ 将学生自制的天线、GPSDO时钟、SDRAngel收发链与Python Doppler处理器集成，形成完整实验链条。

**🔧 技术方法**

技术包括：2.4 m抛物面天线（3‑D打印支架、线性极化馈点）、HackRF SDR、GPS‑disciplined 10 MHz时钟、SDRAngel采集软件、基于NumPy/SciPy/PyTorch的Python Doppler处理器，GPU加速（RTX 5070），以及自制的非线性4次幂算法。

**📊 数据集**

数据集包括：1）模拟QPSK+AWGN信号（不同SNR、窗口尺寸）；2）Artemis I 25 m波段档案数据；3）机遇型CubeSat和GPS卫星的现场收集数据；4）Artemis II 6条现场录制文件（2.214–2.2165 GHz，5–10 MHz采样）。

**📈 对比分析**

对比方法：将Doppler处理器在模拟数据中与SNR阈值相关联；在Artemis I档案中与已公布的Doppler值比较；在现场数据中扫描25–45 kHz范围并评估峰值显著性。结果显示：SNR≥‑5 dB可稳定检测；SNR≈‑10 dB时偶有检出；SNR＜‑10 dB时几乎无效。Artemis II现场数据因噪声-90 dB、SNR约‑39 ~‑29 dB，未能成功提取Doppler峰。

**⚠️ 局限性**

局限性包括：① 天线噪声和环境干扰导致噪声底高于理论值；② 路径损耗大、SNR不足（最优情况‑15 dB）；③ 观测窗口短、机动范围受限；④ 4次幂非线性产生子波峰，可能在低SNR下产生误峰；⑤ 需要更长采样窗口或更高增益天线以提高灵敏度。

---

## 89. Agentic Publication Protocol: An Attempt to Modernize Scientific Publication

**arXiv ID:** 2606.27386 | [PDF](https://arxiv.org/pdf/2606.27386v1)

**作者:** Sirui Lu `[一作]` (Max-Planck-Institut fuer Quantenoptik), Xiao-Liang Qi `[通讯]` (Stanford University)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Agentic Publication Protocol（APP），将论文及其代码、数据、环境和可执行指令打包为版本控制的Git仓库，并为LLM代理提供交互式阅读与复现功能。

**💡 创新点**

创新点在于将论文的静态文本与可操作的实验材料统一托管在可复现的版本控制对象中，并通过agent技能让机器能够主动解释、复现和推演后续研究。

**🔧 技术方法**

采用了Git、GitHub仓库结构、manifest文件、agent指令文件以及LLM代理技能的组合技术。

**📊 数据集**

主要以协议和工具代码为数据集，未涉及具体科学数据集。

**📈 对比分析**

通过对APP发布流程与传统静态论文的可复现性差异进行定性评估，并展示了示例论文的agent复现效果，表明APP能显著降低复现成本。

**⚠️ 局限性**

局限性包括对开发者和研究者的技术门槛、需要社区广泛接受、代理能力受限于训练数据以及协议细节的进一步完善。

---

## 90. COOPA: A Modular LLM Agent Architecture for Operations Research Problems

**arXiv ID:** 2606.27611 | [PDF](https://arxiv.org/pdf/2606.27611v1)

**作者:** Chuanhao Li `[一作]` (Tsinghua University), Zhuoran Yang `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 COOPA，一种模块化的 LLM 代理体系结构，用于运筹学（OR）建模与求解，包含迭代置信度建模、源追溯与多解算器调度。

**💡 创新点**

创新点在于：①通过生成多候选模型并使用自评置信度（四维评分）做 max‑min 选择，显著提升建模质量；②将每个建模元素与原问题文本关联，形成可追溯的审计链；③设计可扩展的解算器代理，使系统能轻松加入新后端。

**🔧 技术方法**

技术实现：LLM 代理层（如 LangChain）、Pydantic 结构化 schema、confidence scoring 与自然语言解释、max‑min 选择策略、基于解算器（Pyomo、OR‑Tools、pymoo、Python）的专用代理。

**📊 数据集**

使用三大基准集：ComplexLP（211 LP/MIP）、IndustryOR（100工业案例）和 BWOR（82 课堂题目）。

**📈 对比分析**

与 Chain‑of‑Experts、OptiMUS、OptiTree、OR‑LLM‑Agent 四个基线对比；宏平均准确率最高 64.8%，在 8 种 LLM 主干中 6 处领先；在 GPT‑5、GPT‑4.1 等强大主干上提升 6.7 / 6.3 个百分点。

**⚠️ 局限性**

局限性：对弱主干（如 o3、Qwen3-30B）提升有限；仅针对确定性问题；未在真实用户实验中评估源追溯与置信度解释的实际帮助；多解算器框架仍需手动添加新代理。

---

## 91. Beyond Points: Spherical Distributional Part Prototypes for Interpretable Classification

**arXiv ID:** 2606.27582 | [PDF](https://arxiv.org/pdf/2606.27582v1)

**作者:** Duarte Leão `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), Carlos Santiago `[通讯]` (Instituto Superior Técnico, Universidade de Lisboa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种在超球面上建模的分布式部件原型网络，将每类表示为von Mises‑Fisher混合，以实现可解释的细粒度图像分类。

**💡 创新点**

创新点包括学习每个原型的自适应浓度参数、利用熵正则化的最优传输进行全局分配约束、以及基于分布重叠的多样性正则化，三者共同消除点原型的冗余与不稳定。

**🔧 技术方法**

核心技术包括vMF混合分布、熵正则化的最优传输（Sinkhorn迭代）、两阶段训练策略（先发现后细化）、前景门控（attention→PCA）以及基于Transformer的自监督DINO特征。

**📊 数据集**

在CUB‑200‑2011、Stanford Dogs和Stanford Cars三个细粒度数据集上进行实验，使用冻结的DINOv2/v3 ViT骨干。

**📈 对比分析**

与ProtoPNet、Deformable ProtoPNet、TesNet、EvalProtoPNet、MGProto和NPPP等基线对比，本文在一致性、稳定性、唯一性等解释质量指标上均取得领先，同时保持竞争力的分类准确率，尤其在CUB上实现了最优的解释效果。

**⚠️ 局限性**

局限性包括：在某些骨干或大规模原型数下准确率略低，仍依赖冻结的自监督特征，训练过程对最优传输参数敏感，且未提供从原型分布生成样本的机制。

---

## 92. SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction

**arXiv ID:** 2606.27581 | [PDF](https://arxiv.org/pdf/2606.27581v1)

**作者:** Sirui Chen `[一作]` (Stanford), C. Karen Liu `[通讯]` (Stanford)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SceneBot，一个结合运动跟踪与接触标签的全身控制框架，能够在平坦地面、斜坡以及物体操作等多种场景下实现一条策略的无缝执行。

**💡 创新点**

创新点在于通过后视场景重建从人类运动生成接触丰富的训练数据，并将接触标签作为低级控制指令，使单一政策即可统一自由空间运动与接触丰富任务。

**🔧 技术方法**

主要技术包括基于 PPO 的强化学习训练全身跟踪策略、后视场景重建构造交互图、以及利用 SuperOdometry 与 IMU 的无外设状态估计。

**📊 数据集**

使用 AMASS、OMOMO、Bones、Lafan 等人类运动数据集，并通过重建方法生成配套的机器人-场景交互数据。

**📈 对比分析**

在仿真与实地测试中，与 SONIC 等现有通用运动跟踪器相比，SceneBot 在自由空间任务保持相当性能，在地形与物体交互任务上显著提升成功率（约 85% 以上），并在长时限任务（如搬箱上楼梯）中保持高精度。

**⚠️ 局限性**

局限性包括对高质量机器人运动的高度依赖，重建过程对低质量或不一致运动敏感；以及对视频或生成模型产生的物理不一致运动的鲁棒性不足，可能导致交互图构造失败。

---

## 93. Perceptual 3D Simulation With Physical World Modeling

**arXiv ID:** 2606.27575 | [PDF](https://arxiv.org/pdf/2606.27575v1)

**作者:** Wanhee Lee `[一作]` (Stanford University), Daniel L. K. Yamins `[通讯]` (Stanford University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 P3Sim 的感知物理仿真系统，能够在仅有局部视觉观测和不完整 3D 变换信号的情况下预测场景未来状态，涵盖静态视角合成、对象操作以及动态交互。

**💡 创新点**

创新点在于将概率推理与几何约束相结合：①构建可处理多模态（RGB、深度、光流）随机变量的概率图模型；②利用“几何化模块”生成基于已知变换的部分深度与光流作为显式条件；③通过可持续场景记忆实现在线更新与不确定性下的连贯性。

**🔧 技术方法**

核心技术包括：7B 参数的因果自回归 Transformer，用随机指针序列实现任意子集条件推断；几何化模块（geometrizer）通过投影和遮挡剔除生成目标深度与光流；持久场景记忆（persistent scene memory）将多帧几何信息聚合到全局坐标，进行一致性校验与更新。

**📊 数据集**

训练使用约 3 万万帧的 RGB 视频（约 1.4 万亿个 token）；在评估阶段采用 SEVA 基准（单图视角合成）和 3DEditBench 基准（3D 对象变换）进行测试。

**📈 对比分析**

与 ViewCrafter、SEVA、DiffusionHandles、LightningDrag、DragAnything 等现有方法对比，P3Sim 在 SEVA 上保持高 PSNR/LPIPS 并实现更精确的摄像机控制；在 3DEditBench 上实现更高的 PSNR 与 Edit Adherence（EA）分数，表明图像质量更好、变换更准确。

**⚠️ 局限性**

局限性包括：对极其复杂或长时间动态场景的建模仍有挑战；系统对输入深度与光流的准确性高度依赖，噪声会影响推断；目前仍使用离散化的多模态编码，限制了细节表达；记忆模块在大规模场景下的存储与查询效率待进一步提升。

---

## 94. Spacecraft Fiducial Marker for Autonomous Rendezvous, Proximity Operations, and Docking

**arXiv ID:** 2606.27566 | [PDF](https://arxiv.org/pdf/2606.27566v1)

**作者:** Ravi Kumar Thakur `[一作]` (Czech Technical University in Prague), Martin Saska `[通讯]` (Czech Technical University in Prague)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并验证了一种基于Spidron分形的多尺度fiducial标记AstraTag，旨在提升航天器近距离操作、对接与清除任务中的视觉定位可靠性。

**💡 创新点**

创新点包括：① 使用自相似正方形Spidron模板实现多尺度递归编码；② 48位GRS码与三角采样结合，提升容错率；③ 内嵌白色矩形边界用于Thin‑Plate Spline（TPS）重映射，增强曲面与大倾角下的检测；④ 在同一标记上兼顾远距与近距可检测。

**🔧 技术方法**

关键技术：Spidron几何递归、Generalised Reed–Solomon 编码、三角区域采样、CLAHE + 自适应阈值二值化、轮廓+Hough线检测、IPPE位姿估计、TPS曲面纠正。

**📊 数据集**

实验数据集：① Aditya‑L1 太阳能板平面模型（平面），② BAS‑01 圆柱模块模型（曲面），在相机-点光源配置下以不同距离和倾角采集多幅图像；还使用金属光电镀铝版的AstraTag进行低分辨率测试。

**📈 对比分析**

与三层Fractal ArUco和AprilTag (48h12) 在同尺寸标记下对比检测率：在曲面上AstraTag在10°–70°倾角保持97–100%检测率，ArUco仅21.9%–93%，AprilTag 100%至70°后下降；在平面上三者均达≈100%；AstraTag在不同距离下（30–150 cm）保持高检测率，尤其在70°倾角时TPS补偿显著优于ArUco。

**⚠️ 局限性**

局限性：① AprilTag在标准尺寸下占用过多像素，导致远距检测优势；② AstraTag 在极远距离（≈150 cm）内层无法分辨；③ 低分辨率或强镜面反射时需放宽 Hamming 阈值；④ 当前识别使用字典匹配，未来需实现 GRS 代数解码以进一步提升速度。

---

## 95. How far does a random forest generalize from a 54-run LAMMPS+SPICA benchmark?

**arXiv ID:** 2606.27695 | [PDF](https://arxiv.org/pdf/2606.27695v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 96. Productionized Fairness Measurement Under Privacy Constraints

**arXiv ID:** 2606.27558 | [PDF](https://arxiv.org/pdf/2606.27558v1)

**作者:** Osonde A. Osoba `[一作]` (LinkedIn), Natesh S. Pillai `[通讯]` (LinkedIn)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了PPRE（Privacy‑Preserving Probabilistic Race/Ethnicity Estimation）系统，利用两方安全计算和加密技术，在LinkedIn生产环境中实现隐私保护的种族/族裔公平测量。

**💡 创新点**

创新点包括：① 将概率种族估计（BISG）与安全多方计算结合，实现软标签公平评估；② 构建一套 HE 扩展模式（比例估计、固定点编码、符号编码、预加密比较、bootstrap 置信区间），使得非线性公平指标可在加密空间下计算；③ 在工业规模上部署并验证了完整的两方协议与治理框架。

**🔧 技术方法**

技术手段包括：Commutative encryption（Pohlig‑Hellman on Curve25519）用于私有集合交叉；Additive Homomorphic Encryption（Paillier 2048‑bit）实现加密聚合；Local Differential Privacy（随机响应）保护 Self‑ID 记录；安全两方协议（PSI‑Sum）与 HE 扩展模式。

**📊 数据集**

使用的数据集：LinkedIn 会员数据；BISG 依据美国 2010 年 Census 姓氏和地理频率表生成概率种族分布；自报 Self‑ID 调查（约 6% US 会员）作为稀疏校准集。

**📈 对比分析**

通过生产部署两款 AI 排名系统（LOT 和 MQOS‑NDCG），采用 bootstrap 置信区间进行显著性检验，结果显示无统计显著差异；系统在大规模数据上运行，B=1000 的 bootstrap 仍可接受，计算成本与延迟可被业务流程接受。

**⚠️ 局限性**

局限性：① 依赖 US OMB 六分类和 Census 数据，难以直接迁移至其他司法管辖区；② BISG 校准不足、Self‑ID 样本自选导致代表性偏差；③ 对近似独立的假设（相邻候选者）可能引入误差；④ 计算成本仍高，尤其是 bootstrap 的并行实现；⑤ 样本选择（地理过滤）可能影响结果的可推广性。

---

## 97. Pick Two: An Adversarial Animal Survival Game

**arXiv ID:** 2606.27557 | [PDF](https://arxiv.org/pdf/2606.27557v1)

**作者:** Jack Vanlyssel `[一作]`, Ramsha Anwar `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对“Pick Two”动物防御选择谜题进行大规模仿真研究，探讨个体能力、群体互动与联盟结构如何影响生存结果

**💡 创新点**

将该问题形式化为对抗性多智能体优化问题，并提出基于生物学的代理模型，揭示团体规模与并行交互是决定成功的关键因素

**🔧 技术方法**

使用Unity引擎构建物理驱动的多智能体战斗仿真，结合HP/装甲、攻击力、速度等属性，采用有限状态机实现不同行为原型

**📊 数据集**

模拟数据基于9种动物（老鼠、鹰、狼、猎人、黑猩猩、狮子、鳄鱼、熊、斗牛）各自的体质量、速度、攻击力等生物学参数，配合10000只老鼠等数量构成群体

**📈 对比分析**

通过18,000次Monte Carlo试验（36种防御组合×500次），评估VIP存活率；结果显示仅少数包含老鼠的联盟能成功，联盟效果非加法，显示规模与协同效应主导性能

**⚠️ 局限性**

模型简化为固定行为原型、缺乏学习与适应、使用简化的伤害与装甲映射、平坦无障碍环境，限制了对真实动物交互的精确预测

---

## 98. Understanding Cross-Rig Generalization in Automotive Perception: a Multi-Rig Benchmark and Rig Variation Metrics

**arXiv ID:** 2606.27554 | [PDF](https://arxiv.org/pdf/2606.27554v1)

**作者:** Tim Alexander Bader `[一作]` (Dr. Ing. h.c. F. Porsche AG), Wilhelm Stork `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于CARLA的多摄像头阵列基准PCCR，用以研究跨摄像头阵列的感知泛化问题。

**💡 创新点**

创新点在于将摄像头几何配置变化与感知性能解耦，提出Rig Variance与Rig Contrastive Distance两种仅基于标定信息的度量，用于定量评估跨阵列性能差距。

**🔧 技术方法**

方法上采用CARLA模拟器渲染同一场景在14套系统化设计的摄像头阵列下的图像，构建统一数据集并利用多视角3D检测网络（BEVDet、BEVFusion、Fast-BEV、PETR）进行跨阵列训练与评估。

**📊 数据集**

数据集为PCCR，由115个场景、30秒/场、60帧/场、共724,500张图像，包含9类3D目标，覆盖不同车身、摄像头位置、视角、FOV等组合。

**📈 对比分析**

通过在每套阵列上训练模型并在其它阵列上测试，计算相对mAP差距，结果显示跨阵列误差显著，且校准后的RigCD与性能下降的Spearman相关系数可达0.7-0.8，证明该度量能有效预测转移难度。

**⚠️ 局限性**

局限性包括仅关注几何配置的差异而忽略光度、畸变等实际硬件因素，基准仍为仿真数据，且仅评估了四种网络，对更广泛模型和真实场景的泛化仍需验证。

---

## 99. Deployment-Side Adaptiveness in Multi-Horizon Volatility Forecasting

**arXiv ID:** 2606.27688 | [PDF](https://arxiv.org/pdf/2606.27688v1)

**作者:** Riku Green `[一作]` (University of Bristol), Telmo M Silva Filho `[通讯]` (University of Bristol)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究在多步波动率预测中，同一训练好的多输出模型在推理时不同的滚动规则会产生不同的预测器，并且通过验证选择或组合这些规则可显著提升预测精度与成本平衡；

**💡 创新点**

提出“诱导规则族”概念，强调部署策略是模型设计的重要组成部分，并证明默认直接多输出（MIMO）部署往往不是最佳选择；

**🔧 技术方法**

使用多种回归与时序模型（线性模型、MLP、LSTM、N-BEATS、PatchTST），并通过不同块大小（s）实现多种滚动规则；

**📊 数据集**

基于VOLARE数据库的20条股票日常波动率序列，采用对数实现方差（log realized variance）作为目标，评估3个预测时长（H=10、20、30）；

**📈 对比分析**

将各部署策略与默认MIMO在MSE和QLIKE上进行比较；验证选择的单一滚动规则在成本最低时能略微提升MSE；小规模子集或线性组合在中等成本下可获得较大MSE改进；但在QLIKE上性能转移不均；与单独训练每个滚动规则的模型相比，诱导策略在中位数误差上几乎无损失，却显著降低训练成本；

**⚠️ 局限性**

实验仅限单变量、日频数据、固定滚动块大小及简单策略，未覆盖多变量/高频场景，也未验证对实际投资决策的收益，且评估仅基于预测误差而非下游金融绩效。

---

## 100. Intuition-Guided Latent Reasoning for LLM-Based Recommendation

**arXiv ID:** 2606.27684 | [PDF](https://arxiv.org/pdf/2606.27684v1)

**作者:** Chang Liu `[一作]` (Beihang University), Wenge Rong `[通讯]` (Beihang University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 IntuRec 两阶段框架，通过先从 LLM 生成的 Top‑K 候选集合抽取直觉源，再将其编码为起始向量注入潜在推理，从而提升 LLM‑based 推荐的准确性。

**💡 创新点**

创新点在于将“直觉”视为一种先验，使用目标感知候选平衡（TACB）和 BPR 对比损失对直觉向量进行监督，使推理起点更贴近目标物品，并显著改善推理轨迹。

**🔧 技术方法**

技术手段包括基于 Qwen2.5‑1.5B LLM 的 beam search、两阶段训练、目标感知候选平衡、直觉双注意编码器（自注意 + 交叉注意）以及对比式直觉对齐损失。

**📊 数据集**

实验数据集为 Amazon Review 的 CDs、Toys、Games 三个子集，采用 5‑core 过滤和时间滑窗分割。

**📈 对比分析**

在 Recall@5/10 与 NDCG@5/10 指标上，IntuRec 在所有传统与 LLM‑based 基线（Caser、GRU4Rec、SASRec、ReaRec、LLM‑Base、LLM‑CoT、AlphaRec、BIGRec、D^3、LatentR^3 等）上均取得最优或显著提升，尤其在 IntuRec‑D（在 D^3 基线上）表现最佳。

**⚠️ 局限性**

局限性包括仅在小规模 Amazon 子集验证，候选列表长度和推理步数有限；直觉编码与对齐损失增加训练复杂度；对长序列或大规模场景的可扩展性尚待进一步评估。

---

## 101. CBD: API-Only LLM Black-Box Unlearning through Controlled Behavioral Divergence

**arXiv ID:** 2606.27683 | [PDF](https://arxiv.org/pdf/2606.27683v1)

**作者:** Zhiqiang Xie `[一作]` (Beijing University of Posts and Telecommunications), Dong In Kim `[通讯]` (Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向API-only LLM服务的黑盒机器消忘框架Controlled Behavioral Divergence (CBD)，通过路由机制将与待消忘数据相关的查询转发给辅助模型，减少对目标模型的直接影响。

**💡 创新点**

创新点包括：①使用双辅助模型（冻结参考模型与可训练探测模型）通过行为差异识别消忘相关查询；②在高相似度场景下构造梯度统计的判别 Fisher 基底（DFB），提升路由准确性；③在不访问目标模型参数、梯度或 logits 的前提下实现有效消忘。

**🔧 技术方法**

技术手段包括：LoRA 微调的双辅助模型、对探测模型进行保留-遗忘梯度投影、基于 Fisher 矩阵的广义特征值问题求解、对称 KL 散度得分与阈值校准的路由策略。

**📊 数据集**

使用的基准数据集：ToFU（虚构作者问答）与 WMDP（危险知识问答），并对 Llama‑2‑7B‑Chat、Zephyr‑7B‑beta 等预训练模型进行实验。

**📈 对比分析**

与11个白盒/灰盒基线（GA、DPO、NPO、ULD、Offset 等）对比，CBD 在 ToFU 的 10% 消忘场景下实现了 74.90% 的保留效用和 2.55 的忘记-保持权衡，比第二佳方法高约 45%；在 WMDP 上将危险知识准确率降至接近随机 25% 级别，同时保持 52.67% 的 MMLU，显示出更优的消忘-保持折中。

**⚠️ 局限性**

局限性：路由误判导致保留查询误转发（false‑positive），但此成本相对较小；对大规模 LLM 的部署仍需辅助模型的存储与推理；当消忘数据与保留数据差异极小且分布重叠严重时，仍可能面临分离难度，需进一步提升判别基底的鲁棒性。

---

## 102. DIM-WAM: World-Action Modeling with Diverse Historical Event Memory

**arXiv ID:** 2606.27677 | [PDF](https://arxiv.org/pdf/2606.27677v1)

**作者:** Kai Wang `[一作]` (NLPR, CASIA), Liang Wang `[通讯]` (NLPR, CASIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种记忆增强的世界动作模型（DIM‑WAM），通过多银行记忆结构整合多尺度历史、即时未来动态与全局任务进度，实现长时程机器人操作。

**💡 创新点**

创新点包括：①多银行记忆与独立相似度合并，实现不同时间尺度信息的非竞争存储；②在读取时注入银行身份和RoPE时间嵌入，支持视频与动作去噪的协同条件；③任务进度监督让记忆同时编码已完成事件与当前阶段，提升全局一致性。

**🔧 技术方法**

技术手段：基于扩散模型的世界动作联合预测、可分离的多银行记忆写入与读取、相似度加权合并、RoPE时间编码、进度监督头、动作与视觉的去噪训练、分支多样性损失。

**📊 数据集**

使用的数据集包括：RMBench（仿真）九个非马尔可夫任务；以及四个真实Frank Panda长时程任务（三角交换、线性交换、两次按压、寻找蓝块），每个任务用15–25个演示进行训练。

**📈 对比分析**

与LingBot‑VA、Diffusion Policy、ACT、π_0.5、X‑VLA及Mem‑0等基线比较，RMBench上平均成功率从28.4%提升至69.8%，超过Mem‑0的42%；在真实机器人实验中，阶段成功率从70.7%提升至91.5%，整体任务成功率从52.5%提升至80%，体现显著性能提升。

**⚠️ 局限性**

局限性包括：记忆容量固定，仍可能在极长任务或极稀疏事件下出现信息遗失；多银行结构增加模型复杂度和训练成本；进度监督需手工离散化进度标签，可能不适用于所有任务。

---

## 103. Explainable AI for Biodiversity Monitoring and Ecological Image Analysis

**arXiv ID:** 2606.27667 | [PDF](https://arxiv.org/pdf/2606.27667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 104. Geometry-Preserving Reduced-Order Modeling via Immersed Tensor Decomposition (ITD)

**arXiv ID:** 2606.27674 | [PDF](https://arxiv.org/pdf/2606.27674v1)

**作者:** Lei Zhang `[一作]` (University of Chinese Academy of Sciences), Wing Kam Liu `[通讯]` (Northwestern University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种浸入式张量分解（ITD）框架，结合了无网格几何表示和可分离的C-HiDeNN-TD降阶求解器，以便在常规背景体素网格上进行大规模仿真。

**💡 创新点**

创新点在于通过体适应函数强制执行Dirichlet边界条件，消除了变分惩罚和复杂的界面积分，同时保持了张量积结构，从而提高了数值性能和收敛性。

**🔧 技术方法**

使用了浸入式方法、张量分解和精确的Dirichlet公式，结合了C-HiDeNN-TD降阶建模技术。

**📊 数据集**

在标准的2D和3D几何域上进行了评估，具体包括圆形、环形、星形和齿轮形状的域，展示了在非笛卡尔几何体素网格上的鲁棒性和收敛性。

**📈 对比分析**

与传统的体适应有限元方法相比，ITD框架在数值性能上表现出最佳的O(h^p+1)收敛性，且在复杂几何体上也能保持良好的收敛性，证明了其有效性。

**⚠️ 局限性**

局限性在于当前框架主要针对标量泊松方程，未来需要扩展到矢量场和非线性偏微分方程，并直接处理体素原生数据。

---

## 105. When Search Agents Should Ask: DiscoBench for Clarification-Aware Deep Search

**arXiv ID:** 2606.27669 | [PDF](https://arxiv.org/pdf/2606.27669v1)

**作者:** Yiling Tao `[一作]` (Tencent), Zhihao Zhu `[通讯]` (Tencent)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本工作提出了 DiscoBench 基准，用于评估搜索代理在多步检索过程中主动识别歧义、提出澄清问题并在用户交互中恢复正确推理路径的能力。

**💡 创新点**

创新点包括：①将歧义视为在多步检索链中动态传播的现象；②设计了多轮交互用户模拟器和四维评估框架（任务效用、歧义检测、交互策略、成本效率）；③构建了覆盖 11 领域、四类歧义的 211 个样本与 463 个歧义实例。

**🔧 技术方法**

技术手段包括：利用 LLM 生成和重构多跳问答链；注入四类歧义并生成辨别性事实；使用外部检索工具（Tavily）配合 LLM；构建基于 LLM 的用户模拟器；实现交互式评估指标和结果分析。

**📊 数据集**

数据集为 DiscoBench，包含 211 个问题、463 个歧义实例，覆盖 11 个真实世界领域，四种歧义类型（实体、版本、标准、事实不准），并配有多轮交互提示。

**📈 对比分析**

在 10+ 代表性 LLM（Gemini、Claude、Doubao、DeepSeek 等）上进行 Neutral/Guided 提示实验。最高端到端准确率仅达 43.1%，检测 F1 最高 64.9%；实验表明主动澄清显著提升通过率，而频繁检索或直接猜测往往效果不佳。

**⚠️ 局限性**

局限性在于：仅涵盖四类客观歧义，未涉及主观偏好等更复杂歧义；用户模拟器基于 LLM 生成，可能无法充分捕捉真实人类交互的多样性、噪声与不可预测性。

---

## 106. LLM-Assisted Model-Based GUI Testing for Vue.js Web Applications

**arXiv ID:** 2606.27665 | [PDF](https://arxiv.org/pdf/2606.27665v1)

**作者:** Tao Li `[一作]` (Macau University of Science and Technology), Lei Ma `[通讯]` (University of Tokyo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于 LLM 的 Vue.js 站点页面转移图（PTG）构建与模型驱动 GUI 测试框架 LLMVue。

**💡 创新点**

创新点在于将大语言模型用于静态候选转移提取的语义化细化，并结合多阶段图规范化实现高精度 PTG，首次在 Vue.js 上验证 LLM 辅助模型驱动 GUI 测试。

**🔧 技术方法**

技术包括静态 AST 分析、GPT‑4o 大语言模型提示工程、图论归一化、Playwright 自动化测试。

**📊 数据集**

使用了 10 个公开 GitHub Vue.js SPA 项目（P1–P10），并对比专家手工 PTG。

**📈 对比分析**

与随机探索相比，LLMVUE 在相同时间预算下平均提升动作数约 60%–70%，页面覆盖率提升 10%–15%，语句覆盖率提升 5%–10%；精度、召回率平均 94.6% 与 88.8%，F1 91.3%。

**⚠️ 局限性**

局限在于对条件/回调驱动的导航识别仍不完全，无法处理复杂权限/状态依赖的多上下文路由；并且评估仅针对单页面 Vue.js，缺乏对其他框架的泛化验证。

---

## 107. Characterizing Driver Interactions with Autonomous Vehicles via Response Maps

**arXiv ID:** 2606.27656 | [PDF](https://arxiv.org/pdf/2606.27656v1)

**作者:** Dave Broaddus `[一作]` (University of New Mexico), Meeko Oishi `[通讯]` (University of New Mexico)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出一种基于数据驱动的线性响应图方法，用于推断人类驾驶员对不同类型自动驾驶车辆行为的反馈控制律。

**💡 创新点**

创新点在于利用响应图而非传统游戏理论或最优假设，直接从模拟实验数据学习人类对AV状态的反馈关系，且采用线性可解释模型。

**🔧 技术方法**

使用线性回归构建响应图，结合工程心理学信息处理框架、混合系统模型与智能驾驶员模型等技术。

**📊 数据集**

使用50名纽约市驾驶执照持有者在StrangeLand驾驶模拟器中完成的7个交叉路口场景，包含3种AV策略（Yield、NoYield、Contingent）的数据集。

**📈 对比分析**

通过留一交叉验证评估模型，平均均方误差低于1（最大值4），证明模型能较好拟合数据，并定量展示不同AV类型下人类驾驶员的响应差异。

**⚠️ 局限性**

局限包括样本量有限、部分实验组数据不足、仅使用线性模型且未进行安全性或稳健性分析，且模拟环境与真实道路存在差距。

---

## 108. DysLexLens: A Low-Resource LLM Framework for Analysing Dyslexic Learners Insights from Online Forums

**arXiv ID:** 2606.27619 | [PDF](https://arxiv.org/pdf/2606.27619v1)

**作者:** Dana Rezazadegan `[一作]` (Swinburne University of Technology), Yong-Bin Kang `[通讯]` (Swinburne University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DysLexLens 框架，利用概念词典过滤、知识图谱推理、检索增强生成和证据追踪，对低资源的 Reddit 论坛数据进行可追溯的分析，聚焦阅读障碍学习者对 AI 工具的使用体验。

**💡 创新点**

创新点在于将多模态技术（概念词典、KG、LLM 推理与检索）结合成一个端到端、可追溯的低资源论坛分析管道，并提供混合量化与人工评估的评估框架。

**🔧 技术方法**

使用 GPT‑4o‑mini 进行三元组抽取与生成、LlamaIndex 构建知识图谱、text‑embedding‑3‑small 进行向量检索，评估采用 RAGAS 指标、查询鲁棒性分析及人工验证指南。

**📊 数据集**

基于 50 个相关 subreddit 的 23,480 篇帖子与评论，经过概念词典过滤后得到 319 篇（来自 27 个子版块）专注于阅读障碍和 AI 讨论的数据集。

**📈 对比分析**

通过 30 个针对五大研究问题的查询，量化评估平均答案相关性 0.75、事实一致性 0.52、检索相关性 0.40，鲁棒性在关键词扰动下下降；人工审计显示 39/100 证据完全可验证、55/100 部分可验证、6/100 无法验证。

**⚠️ 局限性**

局限包括 Reddit 讨论不具备全体阅读障碍人群的代表性、LLM 抽取三元组可能引入噪声、检索精度不足导致证据匹配不完整，以及对非文本平台的适用性有限。

---

## 109. Joint Transcription and Decryption of Images of Encrypted Handwritten Documents: A Comparison with the Traditional Pipeline

**arXiv ID:** 2606.27700 | [PDF](https://arxiv.org/pdf/2606.27700v1)

**作者:** Marino Oliveros-Blanco `[一作]` (Universitat Autonoma De Barcelona), Beáta Megyesi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种端到端的直接图像解密（Direct Image Decryption）模型，跳过传统的字母识别转录步骤，直接将加密手稿图像映射到明文；同时构建大规模Copiale风格的合成数据集以训练模型。

**💡 创新点**

通过消除转录阶段，实现错误传播减弱和信息损失最小化；使用端到端注意力机制在视觉特征和语言生成之间共享梯度，从而提升对真实手稿的泛化能力。

**🔧 技术方法**

利用CRNN（卷积递归神经网络）进行视觉特征提取，结合多头注意力的LSTM解码器进行字符级生成；对转录阶段使用CTC损失，对端到端解密使用交叉熵；训练采用AdamW、梯度裁剪等技术。

**📊 数据集**

主要数据集为：115,000条来自18世纪德语文本的合成Copiale样本（Faust数据集），1,300条Novalis古德语诗歌样本，以及约2,000条真实Copiale手稿线段。

**📈 对比分析**

与传统两阶段转录+解密管线相比，端到端模型在合成数据上提升约1.1%词标记准确率、WER降低49%；在Novalis外域数据上提升约6.3%标记准确率、WER降至31.6%；在真实Copiale数据上提升约11.8%标记准确率（从39.6%到51.4%），WER从89%降至76%。

**⚠️ 局限性**

主要限制为训练数据不足，尤其是真实手稿样本极少（仅2,000条），导致模型在真实数据上的性能显著下降；合成数据与真实手稿仍存在退化和风格差距；此外模型仅适用于已知密钥的替换密码，尚未证明对未知密码的鲁棒性。

---

## 110. Class-frequency Guided Noise Schedule for Diffusion Models

**arXiv ID:** 2606.27696 | [PDF](https://arxiv.org/pdf/2606.27696v1)

**作者:** Jiequan Cui `[一作]` (Hefei University of Technology), Hanwang Zhang `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于类别频率的噪声调度方法，针对扩散模型在长尾数据集上生成低频类样本质量差的问题进行改进。

**💡 创新点**

创新点在于将类别频率信息引入多尺度噪声调度，噪声尺度与类频率成反比，从而缩小低频类的低密度区域并平衡得分空间。

**🔧 技术方法**

技术手段包括基于分数的扩散模型（DDPM、NCSN）、类别频率引导噪声调度、无分类器引导（classifier-free guidance）以及对 Stable Diffusion 与流匹配（flow matching）的拓展。

**📊 数据集**

实验数据集主要为长尾版本的 CIFAR-100（CIFAR-100-LT）和 ImageNet（ImageNet-LT），并在 Stable Diffusion 上使用 CIFAR-100-LT 进行文本到图像的实验。

**📈 对比分析**

与 DDPM 基线、重采样、SQRT 重采样、ADA、CBDM 等方法对比，CFRG 在 CIFAR-100-LT 上将 FID 从 7.38 提升至 6.62（下降 0.76），在 ImageNet-LT 上从 3.09 降至 2.62（下降 0.47）；在分类任务中，利用生成样本提升的 top‑1 准确率达到 50.54%（比基线提升 9.22%）。

**⚠️ 局限性**

局限性包括对类别频率估计的依赖、需调节 σ_T^min/σ_T^max 等超参数，对不同分辨率或更大规模数据集的泛化性尚未充分验证，且方法在某些生成任务中对计算资源的需求仍较高。

---

## 111. Mitigating LLM-based p-Hacking by Preregistering for the Next LLM

**arXiv ID:** 2606.27687 | [PDF](https://arxiv.org/pdf/2606.27687v1)

**作者:** Maria Thomas `[一作]` (Johns Hopkins University), Nihar B. Shah `[通讯]` (Carnegie Mellon University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证一种针对LLM的预注册协议，即先在现有模型上确定实验流程并预注册未来可接受的模型集合，然后在首个符合条件的新模型上执行确认性分析，旨在阻止LLM相关的p‑hacking。

**💡 创新点**

创新点在于将传统的预注册方法扩展到LLM研究，通过在新模型尚未发布时预先锁定实验和模型集合，利用模型跨版本不易迁移的特性切断研究者对结果的后向调优。

**🔧 技术方法**

技术实现包括：预注册流程设计（包含prompt、解码参数、输出合法性规则等）、多模型多配置的p‑hacking检测、基于阈值的目标标签率判定、以及对比下一版本模型的迁移率统计。

**📊 数据集**

使用的数据集为：2020年ICLR的100篇同行评审（假设全部人为写作）进行AI生成文本判别任务；MyFitnessPal的100对连续日食日志（真实卡路里计算）用于日食卡路里比较任务。

**📈 对比分析**

通过在已发布和未来模型之间对同一配置的p‑hacking成功率进行比较，得到约73.9%和72.7%的抑制率；并在注入真实效应的情形下，确认下一模型确认协议对检测功效几乎无损（与使用同一模型复测对比）。

**⚠️ 局限性**

局限性包括：实验仅覆盖四个LLM提供商的二分类任务，配置数量有限；部分配置在新模型上仍能成功p‑hacking；未评估多分类或生成式任务的效果；阈值法未直接使用p‑值或多重检验校正。

---

## 112. Multi-Modal Conditioned High-Resolution Transformer for Urban Electromagnetic Field Map Prediction Download PDF

**arXiv ID:** 2606.27671 | [PDF](https://arxiv.org/pdf/2606.27671v1)

**作者:** Do-Eon Kim `[一作]` (Soongsil University), Seongsin Kim `[通讯]` (Soongsil University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种多条件密集预测框架，用于从建筑布局图像和天线配置快速生成城市电磁场（EMF）强度地图。

**💡 创新点**

创新点包括：1）使用FiLM对标量天线参数在HRFormer各阶段进行特征调制；2）在最深层加入交叉注意力，将天线辐射图案令牌与空间特征融合；3）引入天线相对空间通道编码距离、接近度与方位，支持坐标一致的测试时增强（TTA）；4）组合掩码L1、MS‑SSIM和焦点L1的复合损失，提升高信号区域的预测精度。

**🔧 技术方法**

主要技术包括：High‑Resolution Transformer（HRFormer）骨干网络；FiLM调制；交叉注意力机制；UNet式解码器；复合损失函数；坐标一致的TTA。

**📊 数据集**

使用768张3.5 GHz（5G NR n78）城市EMF模拟地图（500×500像素），每张包含建筑布局、天线参数和辐射图案，训练集614张，验证76张，测试78张。

**📈 对比分析**

与Plain UNet和HRFormer+Reg基线对比，本文模型在测试集上的MAE降至0.0461，分别比UNet低25.2%和HRFormer低31.8%；SSIM提升至0.949；热点IoU提升至0.406。TTA进一步实现6.3%的MAE下降。

**⚠️ 局限性**

局限性：仅基于单一城市拓扑的模拟数据，缺乏真实测量；数据分割允许相同建筑在训练与测试中出现，评估的是同一拓扑内的泛化；仅处理二维传播地图，未覆盖三维多层建筑与垂直信号变化。

---

## 113. GeoFace: Consistent Multi-View Face Generation with Geometry-Constrained Diffusion

**arXiv ID:** 2606.27659 | [PDF](https://arxiv.org/pdf/2606.27659v1)

**作者:** Yeji Choi `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

从单张人脸图像生成多视角 RGB 图像并同步生成对应的 3D UV 位置图（即几何表示）；

**💡 创新点**

在双流扩散框架中引入几何流并通过几何引导的注意力对齐损失实现视角间的几何一致性；

**🔧 技术方法**

采用基于 CAT3D 的双流扩散网络、Plücker 嵌入、可学习的几何相机 token 以及 UV 位置图的几何对齐损失；

**📊 数据集**

使用 RenderMe‑360 与 Nersemble v2 这两个多视角人脸视频数据集进行训练与评估；

**📈 对比分析**

与 PanoHead、SphereHead、DiffPortrait360、CAP4D、SEVA 等基线对比，实验表明其在 PSNR、SSIM、LPIPS 与 CSIM 等指标上均优于对手，特别是在大角度视角下的几何一致性更佳；

**⚠️ 局限性**

依赖 FLAME 的几何表示，难以捕捉头发、耳朵、牙齿等非面部区域的细节，对极端姿态或复杂配饰的生成效果受限。

---

## 114. Temporal-Emerged Prompting for Segment Anything in Multiframe Infrared Small Target Detection

**arXiv ID:** 2606.27655 | [PDF](https://arxiv.org/pdf/2606.27655v1)

**作者:** Yinghui Xing `[一作]` (Northwestern Polytechnical University), Di Xu `[通讯]` (Huawei Technologies Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Segment Anything的多帧红外小目标检测框架，利用时间上出现的提示实现无交互精确分割。

**💡 创新点**

设计了差异增强时序编码器与时序特征注入及时序提示生成器，显式挖掘全局与局部运动差异并注入SAM，解决低SNR背景下的目标可辨识性。

**🔧 技术方法**

采用冻结的SAM编码器、ConvGRU全局-局部时序建模、时序差异调制、MLP注入、注意力生成提示以及BCE+Dice损失与辅助损失等技术。

**📊 数据集**

在NUDT‑MIRSDT和TSIRMT两个多帧红外小目标检测基准上进行评测。

**📈 对比分析**

与单帧、传统多帧、SAM、SAM2及TSP‑SAM等方法对比，TEP‑SAM在IoU、nIoU、Pd、Fa等指标上均显著提升，尤其在低SNR子集提升4-10分。

**⚠️ 局限性**

仍依赖预训练SAM对目标尺度与分辨率有一定限制，对极小目标与极高噪声场景的鲁棒性尚待进一步验证，且模型对时序窗口大小敏感。

---

## 115. MER-R1: Multimodal Emotion Reasoning via Slow-Fast Thinking Synergy

**arXiv ID:** 2606.27652 | [PDF](https://arxiv.org/pdf/2606.27652v1)

**作者:** Zhiyuan Han `[一作]` (University of Science and Technology of China), Xun Yang `[通讯]` (University of Science and Technology of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于强化学习的慢快思维协同框架，旨在提升多模态情感识别的准确性与可解释性。

**💡 创新点**

创新点在于将记忆率与精准度解耦为双目标，并通过慢快置信度校准实现两种思维模式的互补；同时提出了情感车轮级别的置信度计算。

**🔧 技术方法**

技术手段包括强化学习（GRPO）框架、双目标优势函数、慢快置信度差值奖励以及多模态LLM Qwen2.5-Omni的推理。

**📊 数据集**

使用的主要数据集为MER-UniBench（涵盖9个子任务）和MME-Emotion（8个任务），并在MER-Caption+上进行预训练。

**📈 对比分析**

在官方评测下，模型在MER-UniBench上平均F1达83.50，显著优于此前最高模型；在MME-Emotion上CoT平均得分达到51.5，综合提升了识别和推理指标。

**⚠️ 局限性**

局限性包括仅在情感识别任务上验证，依赖情感车轮映射且训练时需要额外的快思维前向传播，可能限制细粒度情感覆盖并增加计算成本。

---

## 116. Real-Time State Estimation in Smart Grids over 5G Networks: Experimental Validation Using Raspberry Pis and Typhoon HIL

**arXiv ID:** 2606.27642 | [PDF](https://arxiv.org/pdf/2606.27642v1)

**作者:** Biswajit Kumar Dash `[一作]` (State University of New York at Buffalo), Filippo Malandra `[通讯]` (State University of New York at Buffalo)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个多节点的5G智能电网实验平台，将基于Raspberry Pi的节点与Quectel 5G HAT以及Typhoon HIL实时仿真器相结合，实现了在商用5G网络上对同步相量数据的实时状态估计和故障检测实验验证。

**💡 创新点**

首个完整的端到端硬件实验验证，既展示了5G在智能电网中的低延迟、高可靠性，又将状态估计和故障检测融入同一平台，系统性对比LTE Cat‑M的性能，证明5G在实际网络环境下可满足实时监测与控制需求。

**🔧 技术方法**

采用Raspberry Pi 4 + Quectel RM520N-GL 5G HAT实现节点通信，利用Typhoon HIL 602+进行电力系统实时仿真，采样通过16‑bit ADS1115 ADC数字化，使用IEEE C37.118.2相量帧格式，UDP协议与PDC服务器交互；状态估计采用加权最小二乘（WLS）+SQP求解器；利用NTP实现时钟同步。

**📊 数据集**

使用Typhoon HIL生成的IEEE 4节点配电馈线实时同步相量数据（电压、电流、功率），在实验中加入周期性负荷波动与三相短路故障，作为实验数据集；未使用公开的标准数据集。

**📈 对比分析**

对比方法：在同一商用5G网络下测量不同上报速率（0.5–120 fps）及室内外环境的延迟、抖动、帧丢失，并与先前LTE Cat‑M实验结果对比；状态估计误差通过MAE、RMSE评估，动态负荷与故障场景下的误差随时间变化；故障检测延迟计算为阈值触发到检测帧的帧数。实验显示5G平均延迟最高32.66 ms，仅为LTE的1/6.5；状态估计在稳态下MAE≈0.6–0.7 V、RMSE≈0.7–0.8 V，动态负荷MAE≈2.8–3.0 V、RMSE≈17–21 V；故障检测平均延迟0.80 s。整体性能满足智能电网实时监测与控制的指标。

**⚠️ 局限性**

局限性包括：仅在单一运营商商用5G网络下验证，未覆盖多运营商或干扰场景；节点采样速率受ADC与处理延迟限制，最大实测速率为50 fps；实验规模仅为4节点馈线，无法直接推断更大系统；外部16‑bit ADC精度有限，进一步提升可提高负荷跟踪与故障灵敏度；实验未评估长期稳定性与能源消耗等。

---

## 117. Yuvion LLM: An Adversarially-Aware Large Language Model for Content And AI Safety

**arXiv ID:** 2606.27632 | [PDF](https://arxiv.org/pdf/2606.27632v1)

**作者:** Ting Ma `[一作]` (Alibaba Security AGI Lab), Hui Xue `[通讯]` (Alibaba Security AGI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一款针对对抗性攻击和代理安全场景的安全大型语言模型Yuvion LLM，并提出了从知识增强预训练到策略对齐多任务后训练再到安全感知代理强化学习的逐步训练范式。

**💡 创新点**

创新点包括：① 把安全视为内在对抗性问题，将对抗鲁棒性与代理能力视为首要目标；② 设计了分阶段的训练流程（知识增强预训练、策略对齐后训练、代理强化学习），实现了多维度安全与实战兼备；③ 研发了四层级评估框架Yuvion LLM RiskEval (YLRE)，涵盖公开基准、对抗基准和真实工业部署场景；④ 在对抗数据、工具调用与搜索推理方面引入了GRPO强化学习，显著提升了模型的鲁棒性和任务执行能力。

**🔧 技术方法**

技术手段主要包括：知识增强的持续预训练（将安全领域知识库映射为多粒度文本样本）；风险感知的多任务安全后训练（监督微调+GRPO策略优化）；安全感知代理强化学习（工具使用与搜索推理的分解奖励设计）；以及对抗性数据构造、红队自动化与闭环迭代机制。

**📊 数据集**

使用了多源训练数据：一般语言数据、安全领域知识数据、对抗性数据、代理任务数据以及专家构造的稀有场景数据；评估数据集则覆盖了公开通用基准、公开安全基准、自建对抗安全基准和内部工业部署基准（共计93个子任务）。

**📈 对比分析**

与公开的通用模型、同等规模与更大规模的商业模型以及专门的安全守门模型进行对比，使用宏观F1、准确率以及对抗合并得分等指标评估。结果显示，Yuvion‑32B在安全基准上平均宏观F1达到78.2%，超过GPT‑5.4、Qwen3‑Max等大型模型；Yuvion‑8B在对抗安全基准与工业部署指标上亦优于大多数同规模模型，并在对抗鲁棒性（动态测试组合得分）上实现最低20.6%；同时在通用语言基准上保持与基线模型相近的性能。

**⚠️ 局限性**

局限性包括：1）对未知新型对抗攻击的鲁棒性仍有限，需要持续红队更新；2）评估主要聚焦中英双语场景，跨语言与跨文化适用性尚未充分验证；3）在通用语言任务上相对基线仍存在轻微性能下降；4）模型对极端稀有或复杂多步骤安全流程的处理仍需进一步强化。

---

## 118. Cross-Platform Chinese Offensive Comment Detection via Dual-Threshold Hard Example Mining

**arXiv ID:** 2606.27629 | [PDF](https://arxiv.org/pdf/2606.27629v1)

**作者:** Ruixing Ren `[一作]` (Beijing Jiaotong University), Fangfang Wang `[通讯]` (Beijing Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出双阈值硬样本挖掘与二次微调方案，用于跨平台中文攻击性评论检测；

**💡 创新点**

通过双阈值筛选高低置信度错误样本，实现低成本、少标注的跨平台适配；

**🔧 技术方法**

利用中文RoBERTa (chinese-roberta-wwm-ext)、二分类 Softmax 头、交叉熵损失、AdamW优化；

**📊 数据集**

使用COLD基准数据集做源域训练，并构建覆盖微博、小红书、贴吧、知乎四个平台、三类标签的交叉平台测试集；

**📈 对比分析**

在源域基线的基础上，通过硬样本微调显著提升四个平台的召回率与F1，尤其在小红书上F1从0.216提升至0.473，整体跨平台宏观F1由0.438提升至0.540；

**⚠️ 局限性**

受限于仅使用单一源域数据，且多平台的标注仍有限，模型在极端隐晦攻击或新兴俚语上仍可能出现误判，且在高置信度误判样本的人工校正仍需人工干预。

---

## 119. FoggyTrust: Robust Federated Learning with Hierarchical Trust Networks

**arXiv ID:** 2606.27622 | [PDF](https://arxiv.org/pdf/2606.27622v1)

**作者:** Emmanuel Rassou `[一作]` (Harvard), Tomas Gonzalez `[通讯]` (Harvard)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FoggyTrust，基于分层信任网络的联邦学习框架，解决全球异构环境下的拜占庭鲁棒性问题

**💡 创新点**

创新点在于将信任计算从单一全局服务器迁移到局部雾节点，利用局部同质性构建根数据集，并在二级聚合层引入可替换的异构感知优化器（FedAdam、SCAFFOLD）

**🔧 技术方法**

使用 FLTrust 的根数据集与余弦相似度信任评估、分层聚合、FedAvg/FedAdam/SCAFFOLD 等聚合与优化技术

**📊 数据集**

在 MNIST、Fashion‑MNIST、CIFAR‑10 和真实世界的 Snapshot Safari 四个图像分类数据集上评估

**📈 对比分析**

与 FLTrust、FedAvg 等基线对比，在 CIFAR‑10 的 Krum/Trim 等局部模型攻击下提升 50% 以上，其他场景下也保持或略优表现；对数据污染攻击提升有限

**⚠️ 局限性**

对数据污染攻击鲁棒性不足，雾节点划分需要先验同质性假设，通信与系统级开销未做深入分析

---

## 120. Room for Error: Large-Scale Simulation of Over-the-Air Acoustic Attacks

**arXiv ID:** 2606.27701 | [PDF](https://arxiv.org/pdf/2606.27701v1)

**作者:** Andrew C. Cullen `[一作]` (University of Melbourne), Benjamin I. P. Rubinstein `[通讯]` (University of Melbourne)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一套大规模声学模拟框架，结合高通量的房间冲击响应（RIR）仿真，系统评估 OTA 对 ASR 的对抗攻击，涵盖 8 百万次实验。

**💡 创新点**

创新点包括：① 将攻击者信息量量化为 Knowledge Gradient，揭示环境不确定性对攻击效果的“信息成本”；② 引入 Dual‑Form SNR，区分源端与受众端的能量和可探测性；③ 基于 Image‑Source Method 的可微分、高通量声学仿真，实现数值可扩展性。

**🔧 技术方法**

技术手段：Image‑Source Method（ISM）声学仿真、可微分 RIR 处理、FGSM 与 PGD 对抗攻击、双重 SNR 约束、知识梯度建模、统计误差分析。

**📊 数据集**

数据集：200 条音频样本，50 种不同几何的房间（通过 ISM 随机生成），构成 10,000 条样本‑环境组合；每条组合产生 256 种攻击/评估变体。

**📈 对比分析**

与基准对比：对比 Naive、Blind、Approx、Oracle 四种信息层级，测量 WER 提升幅度。结果显示：wav2vec2 对声学干扰更脆弱（相对 WER 最高达 94.5%），Whisper 受影响相对较小；在未使用房间信息时的 FGSM 仍能保持高效，揭示梯度失配现象；双重 SNR 展示源端可被忽视而受众端仍能高效攻击，提示能量成本和可探测性解耦。

**⚠️ 局限性**

局限性：① 仅模拟镜面反射，未涵盖衍射与散射；② 采用线性时不变模型，实际环境中存在非线性与噪声；③ 结果基于模拟，需在真实物理实验中进一步验证；④ 计算上仍以数字频率为基准，可能忽略高频行为；⑤ 只关注无目标攻击，对目标攻击的适用性未做系统评估。

---

## 121. What Was That Again? Certified Robustness for Automatic Speech Recognition

**arXiv ID:** 2606.27698 | [PDF](https://arxiv.org/pdf/2606.27698v1)

**作者:** Andrew C. Cullen `[一作]` (University of Melbourne), Benjamin I. P. Rubinstein `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于E值的双门（Atomic + Tournament）诊断管线，对自动语音识别（ASR）系统进行在线安全性认证与鲁棒性修复。

**💡 创新点**

创新点在于：①用E值实现任意时刻有效的统计检验，避免传统频率统计的“窥视”问题；②将token层与句子层分离，先验证词的存在/排除，再通过排名博弈选取最可信序列；③在序列层避免对齐算法，直接在候选集上进行竞争式E值测试，显著减少组合复杂度。

**🔧 技术方法**

技术包括：随机平滑（Randomised Smoothing）与E值、Ville不等式、蒙特卡洛采样、竞赛式E值博弈、置信序列与安全半径映射、POS标签分析。

**📊 数据集**

使用LibriSpeech和Common Voice两个公开语音数据集，并在四种主流ASR架构（Whisper‑Large、Whisper‑Small、wav2vec 2.0 Large、HuBERT）上进行实验。

**📈 对比分析**

与传统对齐式校正（ROVER）和Cohen随机平滑方法对比，本文方法在所有噪声水平下均实现了更低的WER（最高可达54%相对下降），召回率保持在40–90%之间，Spearman相关性相对较弱但稳定，且在极端低SNR（-5 dB）时仍能提供有意义的安全半径，传统方法则召回率趋近0。

**⚠️ 局限性**

局限性包括：①Atomic门的单词级错误率受词汇规模影响，实际Family‑Wise Error率可能高于设定的α_atomic；②E值阈值设定需经验调参；③对α_atomic与α_tourn的分配缺乏自动化方法；④仅针对ℓ₂扰动，未覆盖更通用的攻击范式。

---

## 122. Host-Driven Flowlet Balancing with Segment Routing over IPv6

**arXiv ID:** 2606.27697 | [PDF](https://arxiv.org/pdf/2606.27697v1)

**作者:** Ryo Nakamura `[一作]` (University of Tokyo), Tomoko Okuzawa `[通讯]` (Toyota Motor Corporation)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于SRv6的主机驱动流子平衡方法，让主机检测流子并通过SRv6封装将其路由到不同路径，从而消除交换机对流状态的需求。

**💡 创新点**

创新点在于：主机侧实现流子检测与SRv6封装，完全无交换机状态；通过自估在途字节并结合两选一策略实现更均衡的流量分配；并结合动态流子超时（Halflife）进一步提升性能。

**🔧 技术方法**

使用技术包括Segment Routing over IPv6（SRv6）、CSID压缩标识、eBPF实现的包处理、Power-of-two Choices算法、动态流子超时调节（Halflife）以及流量测量工具flowperf。

**📊 数据集**

数据集为由flowperf产生的固定大小100KB流，以及模拟真实工作负载的Meta Hadoop和Web Search两组流量分布。

**📈 对比分析**

与ECMP、随机路由RPS和LetFlow等方法对比，实验显示在固定大小流上P2C/ Halflife-P2C能将99%分位流完成时间降低15%–33%；在大流量负载下尾部延迟可被减半，整体性能显著优于传统ECMP。

**⚠️ 局限性**

局限性在于验证仅在有限规模物理测试平台上进行，缺乏大规模部署与多种拓扑的验证；估算模型与阈值设定依赖经验，无法充分适应动态网络条件。

---

## 123. Halt Fast! Early Stopping for Certified Robustness

**arXiv ID:** 2606.27694 | [PDF](https://arxiv.org/pdf/2606.27694v1)

**作者:** Andrew C. Cullen `[一作]` (University of Melbourne), Benjamin I. P. Rubinstein `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一种基于随机平滑的可随时验证（anytime‑valid）鲁棒性认证框架，利用元学习预测输入特定的先验分布，从而在保持严格统计保证的同时显著降低样本复杂度并实现动态资源分配

**💡 创新点**

1）将混合E‑值方法推广到连续假设，支持精细的半径估计；2）提出样本自适应元学习器，预测贝塔混合先验以加速财富积累；3）设计任务自适应终止策略（精度阈值、急救退出等），实现对不同安全级别的资源三叉分配

**🔧 技术方法**

随机平滑（Randomised Smoothing）、E‑值与测试马丁格尔理论、贝塔分布与混合E‑值、元学习（残差MLP）、贝塔截断与安全锚、精度阈值与伯特兰‑德克ker求解

**📊 数据集**

ImageNet、CIFAR‑10、MNIST（以及其多倍噪声版本）以及在这些数据集上训练的基准网络

**📈 对比分析**

与传统Clopper‑Pearson固定样本RS（如Cohen‑10k）和KT先验Mixture RS进行对比；平均样本数下降8–15%，对ImageNet可达4%更紧致的置信区间；在专门化安全桶（桶化策略）下，平均样本数下降至≈700（相较10,000下降≈14×），并保持99.9%正确桶化率；整体计算成本降低约89%（相较标准RS）

**⚠️ 局限性**

对极端难度输入的先验预测仍有误差导致早期退出误判；E‑值方法在边界附近的置信区间可能更宽，导致对真实半径的估计略逊于固定样本方法；元学习器对数据分布的依赖仍需进一步评估其跨域泛化性

---

## 124. Are Time-Series Foundation Models Ready for E-Nose Data? An Empirical Assessment of Their Embeddings

**arXiv ID:** 2606.27672 | [PDF](https://arxiv.org/pdf/2606.27672v1)

**作者:** Taeyeong Choi `[一作]` (Kennesaw State University), Mohammed Kamruzzaman `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文系统评估了时间序列基础模型（Chronos-2、MOMENT）在电子鼻（E‑Nose）气体识别与浓度预测任务中的表现，并探讨其与专用域模型的融合效果。

**💡 创新点**

创新点在于首次将跨域预训练的时间序列基础模型迁移到气体传感数据，验证了其可迁移性与融合潜力，并给出Fine‑Tuning与跨设备泛化的经验。

**🔧 技术方法**

技术包括预训练的Transformer‑基编码器（Chronos-2、MOMENT）、全局平均池化提取嵌入、后置分类/回归头、以及多任务学习和微调策略。

**📊 数据集**

使用公开的Twin Gas Sensor Arrays数据集，包含五台E‑Nose单元、八个MOX传感器、四种气体与十个浓度水平的时间序列记录。

**📈 对比分析**

与MLP、DBFE等基线做比较；结果显示冻结的基础模型表现平平；微调后Chronos-2在所有测试单元上均优于基线，且与MLP拼接后性能进一步提升，分类准确率提升约107%，回归RMSE降低约19%。

**⚠️ 局限性**

局限性包括：仅测试小型模型，未覆盖更大规模或多样化气体传感器；缺乏漂移适应；未针对不同传感器阵列结构进行更深入的迁移分析。

---

## 125. MVPruner: Dynamic Token Pruning for Accelerating Multi-view Vision-Language Models in Autonomous Driving

**arXiv ID:** 2606.27660 | [PDF](https://arxiv.org/pdf/2606.27660v1)

**作者:** Nan Yang `[一作]` (Chang'an University), Xiangmo Zhao `[通讯]` (Chang'an University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 MVPruner，一种两阶段动态 token 剪枝框架，针对多视角视觉语言模型在自动驾驶中的高效推理问题；

**💡 创新点**

①基于视角信息多样性和任务相关性动态分配剪枝比率；②跨阶段贡献评估同时保留全局语义与任务关键信息；③实现无离线超参数搜索的自适应剪枝；

**🔧 技术方法**

token 剪枝（相似度与注意力双重评估）、多视角信息多样性度量、跨层贡献评估、语义相似度驱动的预算分配；

**📊 数据集**

DriveLM、DriveLMM-o1、MAPLM、STSnu（均基于 NuScenes、LiDAR 等多视角数据集）；

**📈 对比分析**

与 FastV、DART、SparseVLM、PACT、DivPrune、Prune2Drive 等 SOTA 方法对比；在仅保留 10%–25% 视觉 tokens 的情况下，在 DriveLMM-o1、DriveLM、MAPLM、STSnu 上分别获得约 98%–99% 的原始性能，推理速度提升 3.8–5×，FLOPs 下降 90%+，显著优于对比方法；

**⚠️ 局限性**

主要计算开销集中在跨阶段相似度计算；对极高分辨率/极长序列的可扩展性待进一步验证；当前仅覆盖多视角图像/视频，未涵盖其他传感器融合场景。

---

## 126. GenWorld: Empirically Grounded Urban Simulation Infrastructure for Scalable LLM-Agent Studies

**arXiv ID:** 2606.27650 | [PDF](https://arxiv.org/pdf/2606.27650v1)

**作者:** Gen Li `[一作]` (Hiroshima University), Tao Feng `[通讯]` (Hiroshima University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 GenWorld 平台：构建建筑级合成城市与人口、定义结构化代理-环境接口，并通过离线编译将 LLM 决策转换为查找表，实现城市级规模（20万+居民）LLM‑agent 仿真。

**💡 创新点**

创新点包括：① 在建筑级别上实现人口合成与真实地理约束；② 设计可追踪、JSON 验证的结构化决策接口；③ 通过离线多次 LLM 推理采样并汇总为查找表，将昂贵的在线推理移出仿真循环；④ 提供可复现的评估案例和完整的验证流程。

**🔧 技术方法**

采用技术：IPF 迭代比例拟合、空间分配与引力模型、结构化代理接口（观察编码、候选集、JSON 验证）、离线 LLM 采样与统计、查找表编译、Python 事件驱动模拟引擎、Streamlit UI 展示。

**📊 数据集**

使用数据集：日本 2020 年人口普查（区块级、年龄性别、就业、学校、住房等），地理信息（建筑、道路、POI、土地用途、行政区划），Yahoo Japan Mobility 100K 移动电话数据（通勤诊断），全国时间使用调查（活动时长），以及相应的行政区划与教育资源数据。

**📈 对比分析**

方法对比：与现有 LLM‑agent、城市仿真、人口合成平台对照表显示 GenWorld 在规模、建筑级别赋值、人口一致性、空间细节、社交网络等方面优于其他平台。评估案例展示全城日间模拟、周末/警报响应等，在 20 万+ 代理下完成无空缺。性能方面：在线 LLM 推理需约 1.9×10⁷ 次调用，离线编译后查找表查询≈0.5 µs/次，整体仿真可在数秒/日完成。

**⚠️ 局限性**

局限性：① 仅对人口、通勤、时间使用进行验证，未对交通流、灾害响应等进行校准；② 离线编译的精度与泛化性未系统验证；③ 仅在日本东广岛进行实验，迁移到其他城市需要重新获取数据并调参；④ 需要大量原始数据，迁移成本高；⑤ 结构化接口限制了动作空间和日程结构，缺乏对更复杂行为模式的支持。

---

## 127. AI-Generated Image Recognition via Fusion of CNNs and Vision Transformers

**arXiv ID:** 2606.27637 | [PDF](https://arxiv.org/pdf/2606.27637v1)

**作者:** Xuan-Bach Mai `[一作]` (University of Science), Trung-Nghia Le `[通讯]` (Vietnam National University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了将CNN与Vision Transformer进行融合，用于识别AI生成图像，实验在CIFAKE数据集上实现了97.32%准确率。

**💡 创新点**

创新点在于采用两种融合策略（拼接与线性组合），结合EfficientNet与ViT的特征提升识别精度，并针对光照变化的鲁棒性做了实验。

**🔧 技术方法**

技术包括EfficientNet v2、ViT-b16模型、特征拼接与加权线性融合、AdamW优化器、早停与检查点训练策略。

**📊 数据集**

使用的主要数据集是CIFAKE（真实图像来源于CIFAR-10，假图像来自Stable Diffusion v1.4）。

**📈 对比分析**

与单一CNN（EfficientNet、ResNet50等）和单一ViT相比，融合模型准确率从单一CNN的97.17%提升到97.44%（拼接）或97.32%（线性），在亮度降低50%的测试集上，VGG+ViT融合模型表现最优，准确率达95.29%。

**⚠️ 局限性**

局限性包括：对极端光照或其他图像扰动的鲁棒性仍有限；模型主要针对单标签二分类，未针对多样化生成模型的泛化能力进行深入评估；以及融合策略在不同硬件/部署场景下的推理速度和资源占用未作详细分析。

---

## 128. TeRoR: Decoupled Temporal Rotation with Relational Circular Region for Temporal Knowledge Graph Embedding

**arXiv ID:** 2606.27651 | [PDF](https://arxiv.org/pdf/2606.27651v1)

**作者:** Peijia Xie `[一作]` (South China Normal University), Huiling Zhu `[通讯]` (South China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了TeRoR模型，对时序知识图谱进行更精准的实体时间演化与关系映射建模。

**💡 创新点**

创新点在于将实体时间演化拆分为主语与宾语的独立相位旋转，并为每个关系引入可学习的圆形有效域半径以捕捉1‑1、1‑N、N‑1、N‑N等复杂映射。

**🔧 技术方法**

使用了复杂数空间中的相位旋转、Hermitian乘积、可学习半径向量、负采样损失以及时间-间隔事实拆分等技术。

**📊 数据集**

在四个公开数据集上进行评估：ICEWS14、ICEWS05-15、YAGO11k和Wikidata12k。

**📈 对比分析**

与静态KGE（TransE、DistMult、ComplEx、RotatE、QuatE）和时序KGE（TTransE、TA‑TransE、TA‑DistMult、DE‑SimplE、ATiSE、TeRo）进行MRR与Hits@K比较，结果表明在ICEWS系列数据集上比TeRo提升1–2.6% MRR，Hits@K亦显著提升；在YAGO11k上取得最高MRR 0.189；在Wikidata12k上略逊于TeRo。

**⚠️ 局限性**

在稀疏数据集如Wikidata12k上性能略逊，主要因半径参数难以充分优化；总体模型仍受限于对极端稀疏情况与更复杂时间区间处理的进一步提升空间。

---

## 129. ZooClaw-FashionSigLIP2: Distilled Fine-tuning for Robust Fashion Retrieval

**arXiv ID:** 2606.27708 | [PDF](https://arxiv.org/pdf/2606.27708v1)

**作者:** Siqiao Xue `[一作]` (ZooClaw.ai), Chunxue Xu `[通讯]` (ZooClaw.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对时尚图文检索，改造SigLIP2-base模型，结合全量微调、知识蒸馏和权重插值，构建高性能检索系统。

**💡 创新点**

创新点在于提出一种“全微调+蒸馏+插值”三步精细化方案，既提升专属域表现，又保持对外域的泛化能力，并证明其优于LoRA、增大骨干网络、LLM文本编码器等传统做法。

**🔧 技术方法**

使用的核心技术包括：全参数微调、Generalized Contrastive Loss（GCL）结合分级相关性、Learning-with-Forgetting蒸馏、线性权重插值（α），以及TREC式池化评测流程。

**📊 数据集**

实验数据集包括：专有的ZooClaw‑Fashion（2k查询/12k图像）、公开的Fashion200k（2k长文本/20.1万图像）以及H&M（2k查询/10.5万图像），并利用Marqo-fashion等外部数据进行对比。

**📈 对比分析**

在三大基准上均取得领先：ZooClaw‑Fashion R@10≈0.795、Fashion200k池化R@10≈0.136、H&M R@10≈0.286；相较于Marqo-fashion、LoRA、SigLIP2-base等基线，综合性能提升10‑20个百分点。

**⚠️ 局限性**

局限性包括：模型仍依赖专有训练数据，对完全陌生时尚类别的泛化受限；权重插值α的选择仍需在特定评测集上调优；未在跨语言或多模态多任务场景下验证。

---

## 130. AdvScan: Black-Box Adversarial Example Detection at Runtime through Power Analysis

**arXiv ID:** 2606.27704 | [PDF](https://arxiv.org/pdf/2606.27704v1)

**作者:** Robi Paul `[一作]` (Rochester Institute of Technology), Michael Zuzak `[通讯]` (Rochester Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为 AdvScan 的运行时对抗样本检测方法，利用功耗侧信道信息在黑盒 TinyML 设备上检测输入是否为对抗样本。

**💡 创新点**

创新点在于：1）仅需单次功耗测量，无需模型内部访问或额外预处理；2）通过统计假设检验（one‑sample t‑test）实现可量化的检测置信度；3）方法与任何 TinyML 模型、攻击方式及硬件平台无关。

**🔧 技术方法**

采用技术包括：功耗侧信道采样、Butterworth 带通滤波、Pearson 相关系数计算、one‑sample t‑test、基于功耗签名的统计检测。

**📊 数据集**

使用的数据集包括 MNIST、CIFAR‑10 和 Speech Commands（DS‑CNN），并在 MLPerf Tiny 基准模型上进行实验。

**📈 对比分析**

与现有白盒/黑盒检测方案对比，AdvScan 在两款 MCU（STM32F303RC / STM32L562RE）上对 18 种攻击/模型组合实现 99.987% 的检测准确率，仅出现 40 个误判；单个全连接层功耗测量，检测延迟约 2.6 ms（on‑chip）或 < 1 s（off‑chip）并保持零误报。

**⚠️ 局限性**

局限性：需要在部署前收集足够的基准功耗样本，硬件漂移或温度变化需重新校准；主动监控模式存在短暂的误机窗口；对极低扰动或针对性精细调整的攻击可能导致漏检。

---

## 131. When AI Deceives: A Natural Experiment on the Causal Effects of Perceived Deception on Player Ratings in RPGs

**arXiv ID:** 2606.27689 | [PDF](https://arxiv.org/pdf/2606.27689v1)

**作者:** Shudong Yang `[一作]` `[通讯]` (Dalian Jiaotong University), Shudong Yang (Dalian Jiaotong University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用《博德之门3》54 次版本更新作为自然实验，构建玩家-版本双向固定效应面板数据，分别量化设计欺骗强度（DDI）和玩家欺骗感知（PDA），检验它们对玩家评价的因果影响及玩家经验的调节作用。

**💡 创新点**

独立区分设计欺骗与玩家感知，并通过版本更新自然实验实现因果识别，提出经验去敏感化机制；同时验证自然实验与双向固定效应可用于游戏用户研究的可行性。

**🔧 技术方法**

使用 BERT 微调分类器提取评论中的欺骗感知，人工编码补丁说明获取 DDI；采用双向固定效应线性概率模型、Logit 模型以及多项稳健性检验（滞后变量、置换检验、情感分数替代等）对结果进行验证。

**📊 数据集**

基于 2019-2025 年间 49 个版本窗口的 160,835 条英文 Steam 评论，配合 54 版本的补丁说明；数据包含玩家 ID、游戏时长、评论文字长度等属性。

**📈 对比分析**

通过双向固定效应 LPM 与 Logit 对比，并进行五项稳健性检验，发现 PDA 负向影响约 0.097 个百分点，DDI 形成 U 型曲线，玩家经验显著调节；伪造检验显示结果不为随机，整体方法具有良好的因果识别与稳健性。

**⚠️ 局限性**

PDA 仅捕捉评论中表达的感知，覆盖率有限；DDI 与内容增量高度共线，难以完全分离；研究仅基于单一 RPG 题材，外部有效性待验证；固定 28 天窗口可能遗漏长期讨论；公开评论数据的隐私与伦理需进一步处理。

---

## 132. CWI: Composite Humanoid Whole-Body Imitation System for Loco-manipulation

**arXiv ID:** 2606.27676 | [PDF](https://arxiv.org/pdf/2606.27676v1)

**作者:** Wenqi Ge `[一作]` (LimX Dynamics), Hua Chen `[通讯]` (LimX Dynamics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出Composite Whole-Body Imitation (CWI) 框架，实现在人形机器人上同时完成稳定的行走与多样化的上肢操作，能够通过双手姿态与速度/高度命令实现完整的全身控制；

**💡 创新点**

创新点在于将MoCap数据按角色解耦：上肢使用完整AMASS上肢数据实现丰富的操纵追踪；下肢仅使用精挑细选的行走/蹲姿片段，通过双AMP鉴别器学习稳定的行走风格；采用多评价器架构降低不同目标间冲突，并通过教师‑学生蒸馏将高维参考压缩为仅需双手姿态的可部署策略；

**🔧 技术方法**

使用深度强化学习（PPO+GAE）、对抗式运动先验（AMP）双鉴别器、动作匹配的行为克隆、教师‑学生蒸馏以及MoCap数据的重定标和平滑处理；

**📊 数据集**

主要数据集为AMASS全上肢MoCap数据；下肢仅采用约10个精选的行走与蹲姿片段；在仿真中使用域随机化，真实实验使用LimX Oli全尺寸机器人；

**📈 对比分析**

在仿真中与HOVER、FALCON、HOMIE等基线对比，CWI在上肢追踪误差、下肢速度/高度误差以及成功率等多项指标上均达到或接近最优；在真实机器人上完成门开启、物品搬运、精细组装等多种日常任务，并支持基于Meta Quest VR的远程操控，表现出强大的协调与鲁棒性；

**⚠️ 局限性**

局限性包括：控制接口仅限双手姿态+速度/高度，无法直接指定任意关节或接触指令；行为覆盖受限于所选运动库和命令空间；难以处理需要特殊接触或高级层次决策的任务；未来工作需要收集更丰富的多模态演示并构建更高层的自律策略。

---

## 133. MultModLM: A multi-modal benchmark for Large-Language Model based hardware schematic generation

**arXiv ID:** 2606.27666 | [PDF](https://arxiv.org/pdf/2606.27666v1)

**作者:** Dhruv Kulkarni `[一作]` (SVNIT), Sai Manoj Pudukotai Dinkarrao `[通讯]` (George Mason University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MultModLM，一个用于评估LLM从RTL描述生成硬件电路原理图的多模态基准；

**💡 创新点**

创新点在于构建了99个RTL模块的数据集、设计了多阶段评估框架（含自评、交叉评估、盲评和人工评估），并揭示LLM作为评判者在结构精确领域的可靠性不足；

**🔧 技术方法**

使用了大语言模型（GPT‑5.2 Go、Gemini 3 Flash Pro）、基于JSON的八项评估量表、Cohen κ统计以及多模型互评技术；

**📊 数据集**

使用从OpenRISC及手工生成的99个Verilog模块构成的数据集，涵盖算术、FSM、控制逻辑等类别；

**📈 对比分析**

通过模型互评与人工评估的多阶段流程进行比较，发现Gemini在功能和结构方面略优于GPT，整体准确率约0.3，且两模型与人工评估的κ值几乎为0，显示评判可靠性低；

**⚠️ 局限性**

主要限制包括缺乏形式化等价检查、评估工具依赖人工判断、数据集规模有限以及LLM在视觉输出中无法准确捕捉结构细节，导致评判一致性差。

---

## 134. VLM-Aware Meta-Optic Front-End Design for Frozen Vision-Language Models

**arXiv ID:** 2606.27646 | [PDF](https://arxiv.org/pdf/2606.27646v1)

**作者:** Chanik Kang `[一作]` (Hanyang University), Haejun Chung `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种CODA框架，利用冻结的视觉语言模型（VLM）分类损失直接对连续密度的二维元光学前端进行反向传播优化，无需重建网络或图像质量辅助目标。

**💡 创新点**

创新点在于将VLM的下游任务损失与光学元件的Adjoint Maxwell求解耦合，实现对元光学密度的梯度更新，并且通过冷启动与热启动两种初始化验证了聚焦先导与VLM损失耦合的优势。

**🔧 技术方法**

采用的技术包括：有限差分时域（FDTD）模拟二维TE模式的光学响应，线扫图像形成模型，将PSF映射为传感器图像，CLIP/SigLIP/DINOv2的冻结编码器以及Adam优化器与梯度裁剪。

**📊 数据集**

使用的数据集有ImageNet-100、CIFAR-100、Food-101，并在三种冻结VLM（CLIP ViT‑L/14、SigLIP、DINOv2）上进行评估；实验基准包含解析菲涅尔光学、聚焦优化（Focus‑opt）和两种CODA训练（冷启动/热启动）。

**📈 对比分析**

比较方法为在相同光学设计域和图像形成模型下，对比不同设计的零样本分类准确率；CODA热启动版在ImageNet‑100上比Focus‑opt提升11.66pp，且在所有三数据集与三模型组合中均无需重新优化光学即获得+7.61～+20.80pp的准确率提升。

**⚠️ 局限性**

局限性主要体现在：仅使用二维FDTD与线扫近似，未建模全宽视场与完整相机包装；仅考虑三色波长与单一传感器行；实验完全在仿真环境中，缺乏硬件实现与真实光学噪声；因此结果需在更真实的光学与感知系统中进一步验证。

---

## 135. Denoising ICF Images with Multiplicative Uniform Noise: A Self-Supervised Study Based on the Log-Domain Noisier2Inverse Framework

**arXiv ID:** 2606.27635 | [PDF](https://arxiv.org/pdf/2606.27635v1)

**作者:** Gyeongha Hwang `[一作]` (Yeungnam University), Naima Naheed `[通讯]` (Benedict College)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发并评估了自监督去噪框架Log‑Domain Noisier2Inverse，用于去除ICF图像中的乘法均匀噪声。

**💡 创新点**

提出了将乘法噪声映射到对数域并证明自监督损失等价于监督学习的理论，结合每图噪声参数的JSON加载和数值稳定化技术，提升了去噪性能。

**🔧 技术方法**

使用对数域转换、Noisier2Inverse网络、U‑Net结构、梯度裁剪、早停、噪声仿真与JSON噪声参数加载等技术。

**📊 数据集**

在由100张合成ICF图像（256×256）组成的数据集上训练，并配有每图的噪声边界信息，用于评估。

**📈 对比分析**

与BM3D（对数域/直接）和Noise2Self进行比较，Log‑Domain NN2I Variant B取得平均PSNR 21.41 dB、SSIM 0.8358，显著优于基准。

**⚠️ 局限性**

仍存在对低对比度细节恢复不足、对真实实验图像的泛化能力未知以及噪声相关性未显式建模等局限。

---

## 136. HybridCodec: Modeling Discrete and Continuous Representations for Efficient Speech Language Models

**arXiv ID:** 2606.27627 | [PDF](https://arxiv.org/pdf/2606.27627v1)

**作者:** Artem Ploujnikov `[一作]` (Mila, Quebec AI Institute), Mirco Ravanelli `[通讯]` (Mila, Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 HybridCodec 与 HybridLM 两个模块，构建混合离散-连续音频表示并统一用于 TTS 与 ASR。

**💡 创新点**

创新点在于通过低帧率离散令牌与一次性非自回归连续残差预测相结合，既保持语义结构，又恢复细节音频信息，显著降低自回归步骤。

**🔧 技术方法**

使用 FocalCodec 基础的离散-连续两路编码、AdaLN 适配器、Vocos 解码器、ECAPA‑TDNN 说话人嵌入、Transformer 架构与自回归+非自回归推理流程。

**📊 数据集**

主要数据集为 LibriTTS 960 小时的清晰音频，评测时仅用 clean 子集，截断 20 秒以内的样本。

**📈 对比分析**

与基线 VocalCodec、FocalCodec 等离散编码器对比，HybridCodec 在 6.25 Hz 下 UTMOS 3.98‑4.07、dWER 1.47‑1.50、SpkSim ≈97%；TTS 中 hybrid 模型在 12.5 Hz 时 UTMOS 从 1.99 提升至 4.10，dWER 从 32.97 降至 14.79；ASR 中 hybrid 模型在 50 Hz 下降 WER 28.11→23.36、CER 14.48→12.36，低帧率下同样表现优于离散基线。

**⚠️ 局限性**

限制在于仍需依赖离散编码器的离散层来保持语义，残差预测对低频细节恢复有限；极低帧率（≤6.25 Hz）下的自然度与多语种、多说话人适应性尚待进一步验证。

---

## 137. P-ARC: Exploiting Subproblem Independence for Parallel Multi-Robot Motion Planning

**arXiv ID:** 2606.27625 | [PDF](https://arxiv.org/pdf/2606.27625v1)

**作者:** James D. Motes `[一作]` (University of Illinois at Urbana Champaign), Nancy M. Amato `[通讯]` (University of Illinois at Urbana Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了Parallel ARC，一种多机器人运动规划的并行变体，针对Adaptive Robot Coordination方法的三个主要阶段进行了并行化处理，包括初始独立规划、冲突检测和冲突解决。

**💡 创新点**

创新点在于通过并行化每个阶段的处理，利用ARC方法的分解特性来提高规划效率，并引入了OR-并行多启动策略以增强并行性。

**🔧 技术方法**

使用了并行处理技术，特别是在冲突检测和解决阶段引入了独立冲突批处理和OR-并行策略。

**📊 数据集**

评估使用了一组扩展的2D移动和3D平面操纵器场景，最多可控制128个机器人，模拟真实世界的多机器人操作。

**📈 对比分析**

与传统的顺序版本相比，Parallel ARC在处理大型Panda多操纵器团队时，规划时间的加速比接近4倍，且在多个场景中表现出显著的速度提升。

**⚠️ 局限性**

限制在于当问题的独立性降低时，Parallel ARC的性能可能不如顺序方法，且在某些复杂场景中，额外的并行化开销可能导致效率下降。

---

## 138. Mitigating Position Bias in Transformers via Layer-Specific Positional Embedding Scaling

**arXiv ID:** 2606.27705 | [PDF](https://arxiv.org/pdf/2606.27705v1)

**作者:** Changze Lv `[一作]` (Fudan University), Xiaoqing Zheng `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种层级位置嵌入缩放(LPES)，通过为Transformer层分配不同的RoPE缩放因子，以在不增加推理延迟或进行模型微调的情况下缓解长文本中的“中间失效”偏差。

**💡 创新点**

创新点在于：1）在层级而非头部或全局范围内自适应缩放，避免多前向传播；2）利用Bézier曲线将高维缩放搜索空间压缩到少量控制点；3）使用基于遗传算法的曲线约束搜索，以极少的示例快速找到最优因子。

**🔧 技术方法**

核心技术包括RoPE位置编码、多尺度RoPE组合、Bézier曲线建模、基于遗传算法的超参数搜索以及注意力分布均衡评估。

**📊 数据集**

使用的主要数据集有MDQA（多文档问答）、Key‑Value Retrieval、ZeroSCROLLS、L‑Eval、MMLU和C‑Eval，用于长文本推理与通用能力评估。

**📈 对比分析**

与基线、Attention Buckets、Ms‑PoE和MoICE等方法比较，LPES在关键位置均衡性上提升最高11.2%（Key‑Value Retrieval）并在推理速度上分别比MoICE快2.42×、比Ms‑PoE快1.45×，整体在所有长文本基准上保持或提升性能且无额外延迟。

**⚠️ 局限性**

局限性：目前仅为训练‑free 方式，未探究在训练或微调流程中的效果；若扩展到可训练的缩放参数，可能进一步提升但需研究其对模型稳定性的影响。

---

## 139. Textual Belief States for World Models: Identifiable Representation Learning Under Strict Mediation

**arXiv ID:** 2606.27681 | [PDF](https://arxiv.org/pdf/2606.27681v1)

**作者:** Xiang Gao `[一作]` (Intuit AI Research), Kamalika Das `[通讯]` (Intuit AI Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在文本世界模型中引入严格的潜在状态中介，确保所有转移与预测仅依赖于潜在状态和动作。

**💡 创新点**

提出通过 factorized GRPO 强制实现严格中介，使潜在状态的充分性可直接评估并显著提升表示质量。

**🔧 技术方法**

使用基于 RL 的 fGRPO 训练、离散文本潜在状态、树结构策略采样、结构化奖励以及对比实验方法。

**📊 数据集**

在 TextWorld 与 ScienceWorld 两个合成文本环境上进行实验。

**📈 对比分析**

与 leaky‑state、stateless 与 embedding‑based 体系对比，严格中介模型保持相同的一步预测精度，在高难度合成任务下 state‑F1 提升多达 30%，并在多步回放中收益从 5% 逐步升至 98%，验证了错误传播优势。

**⚠️ 局限性**

仅评估表示质量，未检验下游控制；仅在两个人工环境中验证，缺乏真实世界测试；单次编码器无法递归补偿信息损失；结构化奖励依赖事实提取，难以推广到非结构化场景。

---

## 140. From Signals to Transfer: A Factorised Study of Probe-Based Uncertainty Estimation in Large Language Models

**arXiv ID:** 2606.27679 | [PDF](https://arxiv.org/pdf/2606.27679v1)

**作者:** Ponhvoan Srey `[一作]` (Nanyang Technological University), Anh Tuan Luu `[通讯]` (Nanyang Technological University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探讨探针式不确定性估计（UE）在不同特征、训练数据构建与迁移设置下的表现，并提出可迁移的基准预训练探针；

**💡 创新点**

系统地分解影响探针性能的因素，发现原始隐藏状态和注意力特征在域内表现强劲，但在分布偏移下压缩与结构化特征更稳健，并给出针对部署的实用最佳实践；

**🔧 技术方法**

使用线性、MLP和CNN探针，提取隐藏状态、注意力、概率等19种特征，采用线性交叉熵损失和Adam优化；

**📊 数据集**

覆盖七个数据集（TriviaQA、SciQ、PopQA、BoolQ、StrategyQA、CommonsenseQA、ARC）以及实体生成的九个领域（Biographies、Artworks、Books、Cities、Events、Inventions、Landmarks、Movies）；

**📈 对比分析**

通过AUROC和ECE指标在域内、同任务跨数据集和跨任务迁移上进行比较；发现原始特征域内AUROC最高，但在迁移和开放式生成任务中，结构化特征（Internal Variance、Lookback Ratio、Layer Top-m Prob.）保持更高AUROC；预训练探针在无目标域标签的开放式生成中可与有限监督的线性探针竞争；

**⚠️ 局限性**

受限于实体中心生成、仅评估少数LLM与特征，未覆盖对话、多轮交互；缺乏对更大规模或其他内部信号的验证；未探讨探针引导的幻觉抑制方法。

---

## 141. Two-Stage Cross-Domain Cervical Abnormality Screening with Cytopathological Image Synthesis and Knowledge Distillation

**arXiv ID:** 2606.27678 | [PDF](https://arxiv.org/pdf/2606.27678v1)

**作者:** Jincheng Li `[一作]` (Nantong University), Lili Zhao `[通讯]` (Nantong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种两阶段跨域宫颈细胞检测框架，先使用空间连续的未配对神经Schrödinger桥（SC-UNSB）生成中间域，再通过双层特征对齐的知识蒸馏提升检测性能。

**💡 创新点**

创新点包括：①引入SC-UNSB，利用位置感知的密集归一化消除高分辨率图像块效应；②提出联合浅层结构对齐（LFA）与深层语义对齐（CFA）的双层特征蒸馏策略，实现渐进式跨域知识迁移。

**🔧 技术方法**

技术手段包括：空间连续的Schrödinger桥（熵正则化最优传输）、密集归一化（Dense Normalization）、多尺度低通滤波+频域特征对齐、MMD+MSE对齐损失、RetinaNet目标检测网络、伪标签策略。

**📊 数据集**

使用的实验数据集为源域CRIC（7,839张宫颈细胞标注图）和目标域ComparisonDetector（7,410张Pap涂片细胞图），并在严格的跨域评估拆分上进行实验。

**📈 对比分析**

与CycleGAN、CUT、NOT、i2i-Turbo、UNSB等图像翻译器以及KD、DKD、SPD等蒸馏方法对比，SC-UNSB+LFA+CFA在目标域获得26.9% mAP、45.8% mAP50，显著优于所有基线。

**⚠️ 局限性**

局限性在于：①对极高分辨率图像的生成和特征对齐仍需较高计算资源；②仅在两大数据集验证，尚未在更多机构或不同病理条件下充分验证泛化能力；③生成域与目标域的差距虽减小，但未完全消除，可能影响最终检测鲁棒性。

---

## 142. CryptoGAT: Are Time Series Models Effective for Cryptocurrency Forecasting?

**arXiv ID:** 2606.27670 | [PDF](https://arxiv.org/pdf/2606.27670v1)

**作者:** Yu Peng `[一作]` (University of Sydney), Josiah Poon `[通讯]` (University of Sydney)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出CryptoGAT，将加密货币价格预测从时序问题重新定义为图问题，并在单价数据上实现高精度预测。

**💡 创新点**

创新性在于重新把加密货币纯价预测任务视作图结构任务，证明时序模型不足，并设计轻量级图注意网络CryptoGAT及其FGAT变体。

**🔧 技术方法**

使用图注意网络（GAT）结合Pearson相关图构造，实验对比LSTM/GRU、Transformer、StockMixer等时序模型，并通过IC、ICIR、Precision@10、年化Sharpe等指标评估。

**📊 数据集**

采用币安交易所日度OHLCV数据，筛选出66只市值>2亿USDT、USDT对、无缺失连续记录的加密货币，时间区间2023‑04‑15至2026‑01‑08，共999个交易日。

**📈 对比分析**

采用与StockMixer相同的训练/验证/测试划分与30天回溯窗口，结果显示CryptoGAT/FGAT在IC、ICIR、Sharpe等指标上显著优于所有时序及空间‑时间模型，Sharpe最高达3.128且回撤控制良好。

**⚠️ 局限性**

仅依赖单价信息，缺乏多模态特征；数据规模受市值、流动性限制；模型在极端市场崩盘情形下的鲁棒性仍待进一步验证。

---

## 143. Direct Action-Head Injection of A Grounded 3D Point Unlocks Spatial and Task Generalization

**arXiv ID:** 2606.27663 | [PDF](https://arxiv.org/pdf/2606.27663v1)

**作者:** Shiang-Feng Tsai `[一作]` (National Tsing Hua University), Yi-Ting Chen `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量化、模型无关的模块，在Vision‑Language‑Action（VLA）模型中将2D定位信息提升到3D，并直接注入动作头以提升空间和任务泛化能力。

**💡 创新点**

核心创新在于：①将2D定位信号提升到3D空间，①通过两层MLP编码3D位移为空间嵌入；②利用AdaLN直接将该嵌入注入Diffusion Transformer（DiT）动作头，实现无须改动VLA骨干网络或预训练过程的泛化提升。

**🔧 技术方法**

使用2D目标点+深度相机参数进行3D投影、两层MLP编码、AdaLN条件化注入；实验中使用LIBERO、LIBERO‑PRO仿真环境与Franka Emika Panda实机，使用Qwen3‑VL‑4B等VLM做2D定位。

**📊 数据集**

主要数据集为LIBERO与LIBERO‑PRO（含任务与位置扰动），以及实机8个抓取‑投放任务。

**📈 对比分析**

在LIBERO‑PRO上相较基线，GR00T‑N1.6的任务扰动成功率从31.2%提升至77.5%，位置扰动从28.1%提升至60.2%；在π_0.5上亦实现类似提升；实机实验亦在扰动条件下保持高成功率，表明方法有效且可迁移。

**⚠️ 局限性**

局限性：仅适用于可关联到目标物体或区域的子目标，无法处理无目标动作；依赖准确的2D定位信号，对极端噪声或复杂场景的定位鲁棒性尚未彻底验证；实验仅覆盖单臂机器人，未验证双臂或人形机体。

---

## 144. CascadeOcc: Rethinking 3D Occupancy World Models with Cascaded VQ Representations

**arXiv ID:** 2606.27644 | [PDF](https://arxiv.org/pdf/2606.27644v1)

**作者:** Kyumin Hwang `[一作]` (Daegu Gyeongbuk Institute of Science and Technology), Sunghoon Im `[通讯]` (Daegu Gyeongbuk Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为CascadeOcc的自回归占据世界模型，通过层次化的VQ编码和时间混合器实现对3D占据场景的粗到细细化预测与驾驶轨迹规划。

**💡 创新点**

创新点在于将粗到细的多尺度VQ框架与时间层次化的TimeMixer相结合，既提升了空间细节表达，又有效捕捉长短期时序依赖，同时不依赖外部知识模态。

**🔧 技术方法**

主要技术包括多尺度VQVAE-v2、Transformer的分层预测机制以及TimeMixer的双层时序注意力模块。

**📊 数据集**

实验数据集使用了Occ3D‑nuScenes和nuScenes两套基准数据集。

**📈 对比分析**

与OccWorld等现有方法相比，CascadeOcc在IoU/mIoU上提升约3–4%，并在nuScenes的碰撞率上显著下降，展示了更优的占据预测与安全规划性能。

**⚠️ 局限性**

在高密度环境中，模型仍可能出现物体漏检或闪烁现象，导致细节预测不够稳定。

---

## 145. Continual Learning for Sequential Personalization of Small Language Models: A Stability Monitoring Analysis

**arXiv ID:** 2606.27634 | [PDF](https://arxiv.org/pdf/2606.27634v1)

**作者:** Thomas S. Paula `[一作]` (PUCRS), Rodrigo C. Barros `[通讯]` (PUCRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在边缘设备上对小型语言模型（SLM）进行序列化个性化的稳定性监控，提出了基于检查点-任务矩阵的轻量级监控协议，并将其与持续学习指标和参考集诊断结合；

**💡 创新点**

创新点在于将检查点级别的稳定性监测与传统持续学习评估相融合，利用固定参考集上的KL散度作为模型漂移的早期预警信号，展示了该指标对模型性能衰退的高度相关性；

**🔧 技术方法**

主要技术包括参数高效微调方法LoRA（包括QLoRA和O-LoRA变体）、对比基线SLM（Qwen、Llama、Gemma），以及使用KL散度、熵、边际等指标对参考集进行分布漂移评估；

**📊 数据集**

使用了TRACE基准中的三大数据集：FOMC（金融/政策分类）、ScienceQA（科学问答）和NumGLUE-CM（算术推理），每个数据集约500个训练样本，约210-51-32个平均长度；

**📈 对比分析**

通过对三种SLM在三任务序列（FOMC→ScienceQA→NumGLUE）以及其逆序进行实验，发现Qwen保持稳定且最终平均准确率最高（≈0.59），Gemma表现出明显的漂移和准确率下降；KL散度与平均准确率呈显著负相关，能提前预警模型失效；

**⚠️ 局限性**

局限性包括：任务数量有限（仅三任务），未覆盖更广泛领域或更长任务序列；LoRA超参数固定，未探索不同rank或增量策略；缺乏对内部权重变化的深入机制分析。

---

## 146. Physics-Guided Robotic Radiation Source Localization along Arbitrary Measurement Paths in Unstructured Environments

**arXiv ID:** 2606.27624 | [PDF](https://arxiv.org/pdf/2606.27624v1)

**作者:** Hojoon Son `[一作]` (Georgia Institute of Technology), Fan Zhang `[通讯]` (Georgia Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一种基于物理引导的机器学习框架，实现了在无结构环境下，利用机器人任意测量路径上的伽马射线通量实现辐射源定位。

**💡 创新点**

创新点包括：① 将物理模型（Beer‑Lambert 与逆平方定律）转化为可训练张量；② 设计偏移张量以稳健处理测距误差；③ 采用并行推理与最小 L1 损失挑选模型，显著提升鲁棒性与精度；④ 该模块与路径规划无关，可作为任何机器人任务的通用感知组件。

**🔧 技术方法**

主要技术包括：物理信息化机器学习（PIML）、并行推理、连续学习、Monte Carlo 粒子输运模拟（OpenMC）、A* 路径规划、深度学习优化器（RAdam）和多模型权重融合。

**📊 数据集**

数据集：36 000 条 OpenMC 生成的仿真样本（18 种环境规模×6 种障碍数量，每种 2 000 份），涵盖 18 种工业源（如 Ir‑192、Cs‑137 等）和 5 种障碍材质（混凝土、铁、铝等）；实验数据包括 3 m×3 m 现场网格采样与 Unitree Go2 四足机器人在三种实验场景下的实时测量。

**📈 对比分析**

与先前 PIML‑2025、粒子滤波、EKF+ML‑EM、STE 双阶段等基准对比，采用均值距离误差、成功率等指标：整体平均误差 0.53 m，成功率 90.74 %（误差阈值1.5 m）、84.74 %（0.75 m）及 63.19 %（0.25 m）。在所有环境规模和复杂度上均优于基线，且实验误差均低于 0.2 m，证明了方法的高精度与鲁棒性。

**⚠️ 局限性**

局限性：仅针对单源、二维定位；未实现多源/无源检测；实验仅覆盖实验室级小规模环境；对连续学习中的窗口大小未自适应；在高度接近或发射率极不均的多源情形下性能下降。

---

## 147. Characterisation of reactive Nash equilibria in repeated additive games

**arXiv ID:** 2606.27653 | [PDF](https://arxiv.org/pdf/2606.27653v1)

**作者:** Franziska Lesigang `[一作]` (Interdisciplinary Transformation University), Nikoleta E. Glynatsi `[通讯]` (RIKEN)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在有限动作集合的可加重复博弈中，基于对手上一次动作的反应式（memory‑1）策略的对称纳什均衡，并给出了完全可行的线性条件；通过将均衡划分为 S‑支持类，进一步分析了其在进化学习过程中的出现频率和稳健性。

**💡 创新点**

创新点在于：①提出了一种通用的自反策略自收益表示方法，使得在可加博弈下的对称纳什均衡完全由线性等式与不等式描述；②揭示了均衡与动作集合子集之间的 1:1 对应关系（S‑支持均衡），并将等价者策略归为该框架；③将该理论与进化动力学相结合，量化了不同均衡类的维度与侵入稳健性的相互影响。

**🔧 技术方法**

技术手段包括：构建马尔可夫链求长期分布；利用可加性与反应式特性将自收益拆解为纯常态策略收益的凸组合；使用线性代数与凸分析得到均衡条件；在三动作捐赠游戏上做数值模拟；采用基于Imhof–Nowak的稀疏突变-模仿进化过程评估均衡类的出现概率与侵入概率。

**📊 数据集**

主要使用自生成的三动作捐赠博弈（C、M、D）作为实验场景；不涉及真实数据集，所有结果均来自计算机仿真。

**📈 对比分析**

通过在三动作捐赠博弈中对不同 S‑支持类进行参数空间采样与突变-模仿进化模拟，比较了各类均衡的出现频率和被侵入概率；结果显示，支持单一动作（小 S）的均衡在模拟中最常出现且最稳健，支持全部动作（等价者）的均衡则最易被侵入；类的维度与稳健性共同决定其在进化过程中的流行程度。

**⚠️ 局限性**

局限性包括：①仅适用于可加博弈与 1 步记忆的反应式策略，无法直接推广到更复杂的策略空间或非可加博弈；②自收益表示与线性条件的可行性依赖于可加性，若游戏偏离此假设需重新推导；③演化模拟使用的是稀疏突变模型，真实社会学习行为可能更为多样化；④对均衡类维度的上界估计是理论上最坏情况，实际空间可能更受具体游戏结构限制。

---

## 148. SHIFT: Gate-Modulated Activation Steering for Knowledge Conflict Mitigation in Retrieval-Augmented Generation

**arXiv ID:** 2606.27786 | [PDF](https://arxiv.org/pdf/2606.27786v1)

**作者:** Ruochang Li `[一作]` (Northeastern University), Maosong Sun `[通讯]` (Tsinghua University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种轻量级门控调节框架SHIFT，用于在检索增强生成中解决参数知识与检索上下文之间的冲突。

**💡 创新点**

创新点在于将神经元级改动转化为可学习的门控调节，采用输入自适应门控制FFN激活，并通过GRPO强化奖励学习，既保持模型冻结又能动态平衡外部与内部知识。

**🔧 技术方法**

技术包括在Transformer层的FFN分支插入可学习门控模块，使用sigmoid调制并结合Group Relative Policy Optimization（GRPO）进行强化学习，正则化门控参数。

**📊 数据集**

使用多种公开数据集：MRQA系列（HotpotQA、SearchQA、NewsQA、NQ、TriviaQA、SQuAD）、ConFiQA（QA、MR、MC）以及MMLU，用于评估冲突解决与通用能力。

**📈 对比分析**

与提示、解码和微调基线相比，SHIFT在Qwen-3和Llama系列模型上平均提升约6% EM/F1，并在大模型中仍保持正向增益，且对通用评测几乎不产生性能损失。

**⚠️ 局限性**

局限性在于未覆盖专业领域或多模态问答，且实验规模未扩展至更大模型，未来需验证跨域和多模态适用性。

---

## 149. ModaFlow: Modality-Aware Flow Matching for High-Fidelity Virtual Try-On

**arXiv ID:** 2606.27773 | [PDF](https://arxiv.org/pdf/2606.27773v1)

**作者:** Xiangyu Sai `[一作]` (South China University of Technology), Yong Xu `[通讯]` (South China University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种基于流匹配的高保真虚拟试衣框架——ModaFlow，能够在保持服装细节的同时实现精准的姿态与服装对齐。

**💡 创新点**

创新点包括：模态感知引导机制，将视觉嵌入与文本嵌入异构融合；多尺度遮罩策略增强对遮挡的鲁棒性；以及余弦相似度与感知流判别双重正则化提升流场一致性。

**🔧 技术方法**

技术手段包括流匹配网络（FLUX.1-Fill-dev）、视觉图像提示适配器、CFG-Zero* 分类器无指导、LoRA 微调、Transformer 流编码器等。

**📊 数据集**

实验数据集使用 VITON‑HD 与 DressCode 两大高分辨率试衣基准，分别在配对与非配对设置下进行评估。

**📈 对比分析**

与多种 SOTA 方法比较，ModaFlow 在 FID、KID、SSIM、LPIPS 等指标上均优于对手，尤其在非配对试衣中 FID 降低约 30%，显示出显著的生成质量提升。

**⚠️ 局限性**

主要局限包括：推理时步数较多导致推理时间长、显存占用高；文本控制仍不够精细；对极端遮挡或极端服装类别的跨域迁移仍存在细微失真。

---

## 150. NormGuard: Reward-Preserving Norm Constraints in Flow-Matching Reinforcement Learning

**arXiv ID:** 2606.27771 | [PDF](https://arxiv.org/pdf/2606.27771v1)

**作者:** Tianlin Pan `[一作]` (Hong Kong University of Science and Technology), Wenhan Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对RL后训练的流匹配模型出现的速度范数膨胀问题，提出了一种一次性正则化策略；

**💡 创新点**

创新点在于引入了一种基于速度范数上限的一侧铰链惩罚，只在速度范数超过参考模型时激活，从而抑制不必要的范数膨胀并提升感知质量；

**🔧 技术方法**

采用了流匹配、强化学习后训练（NFT、AWM、DPO）、速度范数正则化、以及多模态LLM评估和法医真实性检测等技术；

**📊 数据集**

使用的主要数据集包括PickScore训练集、HPDv3测试集，以及公开的SD3.5-Medium和FLUX.2-klein-base-4B基准模型；

**📈 对比分析**

通过与未正则化基线对比、早停对照、KL正则化对照以及不同采样步长的实验，显示该方法在多种配置下均能提升MLLM判定的图像质量与真实性评分，同时保持或略增奖励得分；

**⚠️ 局限性**

局限性在于只针对速度本地化的后训练目标验证，尚未覆盖如Flow-GRPO等轨迹级优化；对径向范数膨胀的动态机制仍未深入剖析，且仅在批级一阶敏感性上证明了无奖励干扰。

---

## 151. RS-Diffuser: Risk-Sensitive Diffusion Planning with Distributional Value Guidance

**arXiv ID:** 2606.27766 | [PDF](https://arxiv.org/pdf/2606.27766v1)

**作者:** Shiqiang Gong `[一作]` `[通讯]` (Northwestern Polytechnical University), Shiqiang Gong (Northwestern Polytechnical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 RS-Diffuser，一种在离线强化学习中使用分布式价值引导的风险敏感扩散规划框架。

**💡 创新点**

通过在扩散采样过程中注入分布式价值评估器的梯度，实现单模型在推理时可自由切换风险厌恶、中立或寻求。

**🔧 技术方法**

结合扩散模型（DDPM/DDIM）、逆动力学解码器、量化回归分布式价值评估以及基于 CVaR 的指导梯度。

**📊 数据集**

在风险敏感 D4RL（Half-Cheetah、Walker2D、Hopper）和风险机器人导航（Risky PointMass、Risky Ant）基准上训练与评估。

**📈 对比分析**

与 CQL、Diffusion-QL、OWCPG、ORAAC、CODAC、UDAC 等基线比较，RS-Diffuser 在平均回报、CVaR 以及安全违规率上均优于对手，显示更强的平均性能与鲁棒性。

**⚠️ 局限性**

仅在模拟环境验证，缺乏在高维真实机器人上的测试；分布式价值需要大量 Monte Carlo 样本；对离线数据分布偏移仍然敏感。

---

## 152. Lightweight Multi-Vehicle Collaborative Perception Acceleration with Fusion Position Adjustment

**arXiv ID:** 2606.27750 | [PDF](https://arxiv.org/pdf/2606.27750v1)

**作者:** Wenzhao Zhang `[一作]` (State Key Laboratory of Networking and Switching Technology), Xiaodong Xu `[通讯]` (State Key Laboratory of Networking and Switching Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于融合位置调整的轻量级中间融合多车协同感知加速方案。

**💡 创新点**

创新点在于引入条件可加性分析，证明在线性层之间可安全调整融合位置而不损失精度，并基于此提出FALL加速框架。

**🔧 技术方法**

采用条件可加性理论、前向/后向融合位置一致性证明以及对PIXOR模型的实验评估。

**📊 数据集**

使用PIXOR模型（基于KITTI/nuScenes等数据集）进行仿真实验。

**📈 对比分析**

通过在不同V2V速率下对比不同融合位置的总感知时延，发现FALL在通信受限时可减低74.8%时延，计算受限时减低30.3%。

**⚠️ 局限性**

局限性包括仅适用于可加性特征融合的模型，非线性层需特殊处理；实验仅在仿真环境和单一模型上验证，缺乏真实部署验证。

---

## 153. Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation

**arXiv ID:** 2606.27732 | [PDF](https://arxiv.org/pdf/2606.27732v1)

**作者:** Yuhang Chen `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Bifocal dLLM，利用右侧上下文副流与因果Transformer并行生成文本

**💡 创新点**

将右侧上下文以低成本SSM（反向Mamba）方式并行注入，保持KV缓存并弥补因果模型的右侧信息缺失

**🔧 技术方法**

采用离散扩散语言模型（MDLM）、因果Transformer、反向Mamba状态空间模型、前向hooks、KV缓存、对齐训练等技术

**📊 数据集**

使用Qwen3-1.7B的60B持续预训练数据，评测在ARC、HellaSwag、PIQA、WinoGrande、OpenBookQA、BoolQ、MMLU等任务

**📈 对比分析**

与双向dLLM和纯因果dLLM比较，R2LM在多项任务上超过因果基准，ALL平均提升约0.97pp；通过KV缓存实现3–12.9×速度提升

**⚠️ 局限性**

限制包括模型规模/训练数据偏小、R2L流增加10.8%参数、单请求时略慢、未验证更大规模效果

---

## 154. Learning 1-Bit LiDAR-based Localization with Auxiliary Objective

**arXiv ID:** 2606.27729 | [PDF](https://arxiv.org/pdf/2606.27729v1)

**作者:** Kaijie Yin `[一作]` (University of Macau), Hui Kong `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种完全二值化的 LiDAR 定位框架 BiLoc，能够实现 6-DoF 的高精度定位；

**💡 创新点**

通过信息瓶颈视角引入训练时辅助目标（软掩码特征蒸馏 + 结构对齐）来缓解二值化导致的信息丢失与梯度不匹配；

**🔧 技术方法**

采用二值 Vision Transformer（BHViT）骨干、PoseDiffusion 解码器、软掩码特征蒸馏、结构对齐（Sinkhorn 算法）等技术；

**📊 数据集**

在 Oxford Radar RobotCar 与 NCLT 两大户外 LiDAR 数据集上进行评估；

**📈 对比分析**

与现有 BNN 和实数值方法对比，BiLoc 在两数据集上分别将位置误差降低约10%~15%，并实现约2×的推理速度提升；

**⚠️ 局限性**

局限在于二值化后仍存在表达容量与梯度匹配问题，且受限于硬件支持，实际速度提升未达到理论上限。

---

## 155. Scene and Human in One World: Reconstruction in a Feedforward Pass

**arXiv ID:** 2606.27720 | [PDF](https://arxiv.org/pdf/2606.27720v1)

**作者:** Boao Shi `[一作]` (University of Pennsylvania), Lingjie Liu `[通讯]` (University of Pennsylvania)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出SHOW框架，能够在单目人像视频中通过一次前向传播同时恢复人类3D网格与周围场景，解决尺度、位置和对齐问题。

**💡 创新点**

核心创新在于：①使用人像掩码作为prompt引导预训练的视觉几何基础模型关注人体区域；②在几何编码中加入DensePose监督，让特征既保持全景几何信息，又具有人体先验；③在SMPL-X解码器中融合场景上下文和人体特征，实现人类尺度与场景几何的双向约束；④联合训练使人类与场景在统一度量空间中自洽。

**🔧 技术方法**

技术包括：预训练的VGGT视觉几何模型、轻量化掩码编码器、DensePose辅助监督、基于Transformer的SMPL-X解码器、射线位置编码、端到端联合优化。

**📊 数据集**

训练使用BEDLAM2.0合成数据，包含SMPL-X参数、深度与相机轨迹；评估在3DPW、EMDB、RICH等公开数据集。

**📈 对比分析**

与Human3R、UniSH等现有前向人像-场景联合恢复方法对比，SHOW在多项指标（MPJPE、PVE、HS-CF、HS-V等）上均表现更优，尤其在人类与场景的尺度对齐和物理交互一致性方面显著提升。

**⚠️ 局限性**

局限在于仿真到真实的泛化仍有限，尤其对极端尺度或人类占图像比例极小的情况适应性差。

---

## 156. The Simulacrum: Decision-Theoretic Pretraining for Near-Optimal Time-Series Forecasting and Inference

**arXiv ID:** 2606.27711 | [PDF](https://arxiv.org/pdf/2606.27711v1)

**作者:** Pablo Montero-Manso `[一作]` (University of Sydney), Marcel Scharth `[通讯]` (University of Sydney)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用决策理论预训练（Simulacrum）让神经网络直接学习结构时间序列模型（如 ETS、AR(p)、混合模型）的参数估计、预测、区间和模型选择，从而在零样本推断上实现近似最优的决策规则。

**💡 创新点**

提出统一的模拟‑决策框架，可自定义生成世界和目标，直接逼近过程层最优决策，能够显式约束有限样本偏差、最小化风险、统一校准等难以用传统方法实现的统计性质。

**🔧 技术方法**

采用两阶段分层模拟（外层抽样机制+内层多次复制）训练神经网络（如 DenseNet），损失函数结合平均风险、偏差惩罚、校准误差、极大值（minimax）等，支持偏差校正、统一校准、最小化风险等多目标训练。

**📊 数据集**

实验主要在合成的结构世界（ETS、AR(p)、混合 ARIMA‑ETS）上训练，并在公开真实数据集 M1、M3、Monash、M4、旅游等基准上验证。

**📈 对比分析**

与传统最大似然、AICc、贝叶斯、N‑BEATS、Chronos 等基准比较，神经估计器在预测误差（MSE/SMAPE/MASE）、参数偏差、均匀校准和最小化风险等指标上往往优于或与最佳方法相当；在组合预测任务中获得近似最优或超越传统组合规则。

**⚠️ 局限性**

限制在于：需要精确设计生成世界，若与真实分布偏差会导致估计失真；训练需要大量模拟样本；对高维、复杂多机制或动态不确定性场景的推广仍面临挑战；未处理战略性数据报告或对抗性鲁棒性等问题。

---

## 157. TRUST: Efficient Abdominal Trauma Recognition via Image-to-Ultrasound-Video Transfer Learning

**arXiv ID:** 2606.27777 | [PDF](https://arxiv.org/pdf/2606.27777v1)

**作者:** Enguang Wang `[一作]` (Southeast University), Guangquan Zhou `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种针对腹部超声创伤检测的参数高效图像到视频迁移学习框架TRUST

**💡 创新点**

通过交叉频率协同适配器（CFCA）、双粒度运动感知模块（MGMA）和视觉查询语义聚合（VQSA）实现细粒度空间、时序与跨模态信息的联合建模

**🔧 技术方法**

采用CLIP对比学习、离散小波变换、运动先验位置编码、视觉-文本注意力等技术

**📊 数据集**

使用内部的294条腹部超声创伤视频数据集

**📈 对比分析**

与多种单模态与多模态SOTA方法对比，在4×2和8×2采样下分别提升准确率至81.36%/83.34%，Jaccard系数提升至64.32%/66.85%，显著优于竞争者

**⚠️ 局限性**

仍受限于数据规模、对不同扫描者的泛化能力以及对极端噪声的鲁棒性等挑战

---

## 158. Constructions and Characterizations of $s$-Plateaued Partitions

**arXiv ID:** 2606.27776 | [PDF](https://arxiv.org/pdf/2606.27776v1)

**作者:** Jiaxin Wang `[一作]` (Hefei University of Technology), Fulin Li `[通讯]` (Hefei University of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文提出了 s-平衡分区（s-plateaued partition）的概念，推广了传统的凹（bent）分区，并证明了利用这种分区可以构造大量 p-元 s-平衡函数、向量值 s-平衡函数以及广义 s-平衡函数。

**💡 创新点**

创新点在于：①给出了 s-平衡分区的完整定义及其与 p-元 s-平衡函数、向量值函数和广义函数之间的等价性；②分析了分区各块大小的可能取值，揭示了与凹分区相比更为复杂的结构；③提出了一般性构造方法，并给出若干显式构造，确保生成的函数无非零线性结构；④在对称分区（-A_i=A_i）情况下给出完整的特征化定理，进一步证明当 f(x)=f(-x) 时其预像分区是 s-平衡分区的充要条件；⑤对凹分区与扩散的关系给出了部分答案。

**🔧 技术方法**

主要采用的技术包括：Walsh 变换与特征函数的分析；数域与整数环（Cyclotomic field ℚ(ζ_p)）的代数整数理论；预半域（presemifield）的结构及其转置；以及对称分区的自同构和自动化性质。

**📊 数据集**

本文属于理论密码学与组合数学研究，没有使用实验数据集。

**📈 对比分析**

由于研究集中在理论证明与构造上，没有实验比较；但通过构造方法与已有的凹函数构造手段做对比，证明了构造的 s-平衡分区可生成更多满足无线性结构的 p-元函数。

**⚠️ 局限性**

限制在于目前构造得到的 s-平衡分区均为非平衡（unbalanced），尚未给出能生成平衡 p-元 s-平衡函数的分区；对称分区之外的 s-平衡分区是否具有相同的 Walsh 支持特性仍未解决；以及 s>0 的分区在编码理论与组合学中的潜在联系尚待深入。

---

## 159. Towards Reliable and Robust LLM Planning: Symbolic Feedback-Driven Iterative Self-Refinement Framework

**arXiv ID:** 2606.27757 | [PDF](https://arxiv.org/pdf/2606.27757v1)

**作者:** Jiajing Zhang `[一作]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences), Daniel Zeng `[通讯]` (State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出符号反馈驱动的迭代自我修正框架，用于提升LLM在长时序规划任务中的可靠性与鲁棒性。

**💡 创新点**

创新点在于将PDDL符号映射为自然语言提示，并通过符号验证器和计划识别器生成可解释的错误反馈，形成闭环自我修正流程。

**🔧 技术方法**

采用自然语言提示、语法/语义验证（VAL）、里程碑式计划识别器、LLM（GPT‑4o、Claude‑3‑5、DeepSeek‑R1）多轮迭代反馈等技术。

**📊 数据集**

使用PlanBench基准（1200个任务）中的Blocksworld和Mystery两个域进行实验。

**📈 对比分析**

通过与LLM直接生成方式对比，使用成功率/覆盖率指标；框架使Blocksworld覆盖率从最高的99.1%提升至100%，Mystery从0%提升至约65%；对不同LLM、规划长度和反馈策略做消融实验，验证显著性能提升。

**⚠️ 局限性**

局限性包括受限于令牌长度、符号复杂度，框架在极长规划或高度动态环境下效果可能受限，且依赖手工构建符号验证与识别模型。

---

## 160. From General-Purpose Audio Tagging to Spatially Grounded Sound Event Localization and Detection

**arXiv ID:** 2606.27751 | [PDF](https://arxiv.org/pdf/2606.27751v1)

**作者:** Stefano Giacomelli `[一作]` (University of L'Aquila), Toon van Waterschoot `[通讯]` (KU Leuven)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一个名为 at2seld 的框架，用预训练的音频标签模型（gpat）作为高层语义先验，结合多通道空间特征、时序建模、轨道级输出以及自适应损失，完成从语义识别到空间事件定位与检测（seld）的端到端迁移。

**💡 创新点**

创新点：① 将大规模弱标注音频预训练模型直接注入到空间定位体系；② 以模块化 NAS 方式系统评估空间前端、时序处理、语义‑空间交互和轨道级监督的组合；③ 设计了兼容多种空间表示（FoA、Gabor 预处理、SRP‑PHAT 等）的通用语义‑空间桥接模块；④ 在设计空间内加入主动‑只关注活动源的定位损失与 permutation‑aware 轨道分配，解决同类重叠与身份保持问题。

**🔧 技术方法**

使用技术包括：卷积+RNN 或 Conformer 进行时频编码；多尺度注意力与残差块做空间前端；Track‑wise 输出空间（multi‑accdoa、adpit 等）；多任务 BCE+MSE 或自适应加权损失；数据增强（ACS、SpecAugment、混合噪声、混响）以及阶段性训练策略。

**📊 数据集**

数据集：AudioSet（预训练）、DCASE 2022/23 SELD 任务数据（真实与模拟混合）、STARSS22/23（高重叠、多通道 FoA），以及用来评估迁移效果的自定义交叉数据集。

**📈 对比分析**

比较方法：在统一的实验框架下对不同空间前端、时序模块、交互策略进行逐级 NAS；将最佳配置与基线 SELDnet、ein‑V2、Conformer‑based 系统进行对比。实验结果表明：在保持或略低于现有基线的整体 SELD 分数下，语义‑空间融合显著提升了同类重叠场景下的定位精度与轨道一致性；在单源场景下的定位误差下降约 10%–15%。

**⚠️ 局限性**

局限性：① 预训练语义先验在强噪声或极端重叠场景下仍会引入误定位，导致“inactive‑target dominance”问题；② 对阈值与类不平衡的敏感性较高，需要额外的校准与自适应阈值；③ 目前仅在 FoA 或固定四通道阵列上验证，尚未完全通用到任意麦克风几何；④ NAS 过程仍依赖大量计算资源，实际部署时需进一步压缩模型。

---

## 161. End-to-End Dynamic Sparsity for Resource-Adaptive LLM Inference

**arXiv ID:** 2606.27743 | [PDF](https://arxiv.org/pdf/2606.27743v1)

**作者:** Yuhang Chen `[一作]` (University of North Carolina at Chapel Hill), Xi Liu `[通讯]` (Meta AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个预算感知的动态推理框架，结合轻量化门控网络与LoRA，在LLM推理时实现层跳过、注意力头裁剪和推理长度控制。

**💡 创新点**

创新点在于将模型压缩与动态推理联合建模，使用预算条件门控网络和统一损失实现端到端学习，能够自适应分配计算资源并保持性能。

**🔧 技术方法**

使用了轻量化门控网络、LoRA参数高效微调、KL蒸馏、统一损失函数，以及预算信号嵌入和动态阈值化。

**📊 数据集**

在Llama‑3‑8B、Qwen‑3‑4B等开源模型上，用OpenWebText、GSM8K、MMLU、Alpaca‑Eval、HumanEval、BBH等数据集进行训练与评测。

**📈 对比分析**

与原始模型、静态裁剪、AdaSkip、FlexiDepth等基线对比，实验表明在相同计算预算下性能更好，尤其在推理难度高的任务上接近全模型并显著提升推理速度与显存利用率。

**⚠️ 局限性**

局限性包括对头裁剪加速受实现依赖，预算校准需要准确的在线延迟/内存估计，结构化输出对提示工程有要求，且在极端预算或新任务时仍可能出现门控崩溃。

---

## 162. Reduction of Probabilistic Chemical Reaction Networks

**arXiv ID:** 2606.27737 | [PDF](https://arxiv.org/pdf/2606.27737v1)

**作者:** Mauricio Montes `[一作]` (Auburn University), Gregoire Sergeant-Perthuis `[通讯]` (Sorbonne Universite)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套从因子图识别并压缩实现贝叶斯推断的化学反应网络（CRN）的理论与算法，并证明压缩后网络在正 steady state 下仍能保留原始因子图的 BP 固定点。

**💡 创新点**

创新点在于：①用六个可直接检验的化学结构条件（W1–W6）识别并重建因子图；②将因子图的 SP–B 重缩操作（保留 BP 固定点）在 CRN 层面实现；③首次给出结构保持的化学网络压缩方法；④在实验中展示压缩后显著提升 ODE 求解速度。

**🔧 技术方法**

技术包括：基于 Napp–Adams 编译的 CRN 设计、Stoichiometry 分块与束结构解析、速率多项式分离与生产系数一致性检验、因子图重缩（SP–B）与 Functor 迁移、以及 Mass‑action ODE 模拟。

**📊 数据集**

数据集主要为合成因子图：链、树、含环核心带尾部、格子图，以及 Erdős–Rényi 随机图；各类图在不同状态空间（主要为二元）下进行实验。

**📈 对比分析**

通过与未压缩网络编译后的 CRN 进行对比，测量化学网络规模（变量/物种数）和 ODE 计算时间；结果显示：树链图压缩后速度提升可达 885×（树）和 270×（链），含环图提升约 22×，随机图在稀疏区可达 16×；压缩比例随图的可缩减结构显著相关。

**⚠️ 局限性**

局限性包括：①压缩仅适用于满足 W1–W6 条件的 Napp–Adams 编译网络，无法直接处理任意化学网络；②对包含多重 BP 固定点的图，压缩后可能切换到不同固定点，导致推断结果不唯一；③在高度密集或无可压缩结构的图（如格子、密集 ER 图）几乎无效；④实验均在模拟环境中验证，实际 DNA 实现仍需进一步工程化验证。

---

## 163. Learning to Reason with Curriculum II: Compositional Generalization

**arXiv ID:** 2606.27721 | [PDF](https://arxiv.org/pdf/2606.27721v1)

**作者:** Nived Rajaraman `[一作]` (Microsoft Research), Akshay Krishnamurthy `[通讯]` (Microsoft Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

学习了如何通过自动生成的层次化课程（autocurriculum）实现对半自动机（semiautomata）推理的组合泛化，从而在监督细粒度和强化学习两种设置下显著降低学习成本。

**💡 创新点**

创新点在于证明自适应分解长序列为短子问题、递归组合并提升弱学习器的多尺度提升（multiscale boosting）可突破 Ω(T) 标注壁垒，获得 2^(√(log T)) 的子多项式样本/查询复杂度，以及在强化学习中将参考模型的覆盖需求从全长 T 降到短块 B。

**🔧 技术方法**

使用的技术包括基于 Markov 半自动机的分解与组合框架、自动课程生成、Boosting+Composition 的多尺度提升、逆采样（inverted sampling）高效采样加权混合边缘、在线学习与错误界定（Littlestone/VC 等）等。

**📊 数据集**

使用的“数据集”为从分布 ρ 采样的合成半自动机实例，涉及长度 T 的随机输入词；实验主要是理论证明，未使用公开真实数据集。

**📈 对比分析**

与传统的全链路监督或端到端反馈方法对比，证明在监督细粒度 iSFT 设置下查询复杂度从 O(dT) 降至 O(2^(√(log T))·d)，在 RLVR 设置下将参考模型覆盖需求从 O(C^T) 降至 O(C^B)，显著提高效率。

**⚠️ 局限性**

限制在于只适用于马尔可夫半自动机、确定性转移；模型必须是马尔可夫式；对随机转移或非马尔可夫语言模型的推广尚未解决；理论结果对真实语言任务的可迁移性仍待实验验证。

---

## 164. Difference of Convex Programming in the Wasserstein Space with Applications to MMD Optimization

**arXiv ID:** 2606.27767 | [PDF](https://arxiv.org/pdf/2606.27767v1)

**作者:** Clément Bonet `[一作]` (Ecole Polytechnique), Youssef Mroueh `[通讯]` (IBM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文将凸凹分解（DC）方法推广到Wasserstein空间，提出Wasserstein凸凹程序（WCCCP），并理论分析其在凸与非凸场景下的收敛性质；

**💡 创新点**

创新点在于：1) 设计了一种新的Wasserstein空间DC优化算法WCCCP；2) 为最大均值差异（MMD）提供了可行的核DC分解，并通过实验验证其优于传统Wasserstein梯度下降；

**🔧 技术方法**

采用的技术包括：Wasserstein梯度、Bregman近似、镜像下降、差分凸优化（DCA）框架、Jordan分解与代数分解法来构造核的DC分解；

**📊 数据集**

实验主要使用了：① 500样本的螺旋形与猫形目标分布；② 500样本的CIFAR-10图像分布；③ 500样本的高斯目标分布；

**📈 对比分析**

比较方法：将WCCCP与Wasserstein梯度下降（WGD）和Wasserstein前向后向（FB）三种算法在同一计算预算下进行比较；实验结果显示，WCCCP在能量距离和高斯核MMD上收敛速度更快、最终目标函数值更低；

**⚠️ 局限性**

局限性：收敛效果高度依赖于核的DC分解策略；对内层优化的近似求解未在理论中充分考虑；在某些分解（如Jordan）下仍可能陷入局部极小值；

---

## 165. DE-2LS: Differential Evolution with Lightweight Late Local Search for Constrained Numerical Optimization

**arXiv ID:** 2606.27764 | [PDF](https://arxiv.org/pdf/2606.27764v1)

**作者:** Dikshit Chauhan `[一作]` (National University of Singapore), Anupam Trivedi `[通讯]` (National University of Singapore)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DE-2LS，一种在 RDEx 基础上加入轻量级后期局部搜索的差分进化算法，专门用于约束单目标数值优化；

**💡 创新点**

在不破坏 RDEx 原有搜索动力学的前提下，设计了仅在搜索后期激活、受限预算、可行性感知的坐标模式局部搜索，并通过可行性比较规则和触发逻辑实现对最优解的精细化改进；

**🔧 技术方法**

使用了 RDEx 的成功历史参数自适应、ε 约束处理、种群规模递减、外部存档等机制，并加入坐标模式局部搜索、可行性感知接受规则、预算与触发控制；

**📊 数据集**

在 CEC 2026/2025 约束优化基准套件中评估，共 28 个 D=30 的连续约束单目标问题，最大函数评估为 20000×D；

**📈 对比分析**

通过 U‑score（对速度与精度的双向对比）在所有函数上进行 pair‑wise 评估。DE‑2LS 的 U‑score 为 80968，排名第 48；相较 RDEx 提升 5.58%，在四算法对比中获得最高 U‑score 和最优总排名；

**⚠️ 局限性**

局部搜索预算与步长参数为固定值，缺乏自适应机制；仅在后期才激活，可能错失早期精细化机会；未在更高维度或其他约束基准上验证，适用范围待进一步扩展。

---

## 166. Drop-Then-Recovery: How Redundant Are Vision-Language-Action Models?

**arXiv ID:** 2606.27755 | [PDF](https://arxiv.org/pdf/2606.27755v1)

**作者:** Guoheng Sun `[一作]` (University of Maryland), Ang Li `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过系统地移除 VLA 模型中的 Transformer 块并随后微调，评估各块在闭环控制中的必要性，揭示语言、视觉与动作分支在当前机器人操纵任务中的结构冗余差异。

**💡 创新点**

创新点包括① Drop-Then-Recovery (DTR) 协议，首次将块移除与后续恢复结合以量化可恢复性；② GateProbe 虚拟门敏感度度量，提供梯度导向的块重要性评估；③ 在多种 VLA 架构和仿真/实机环境下统一验证语言分支过度容量的普遍性。

**🔧 技术方法**

技术手段包括 Transformer 块移除、虚拟门敏感度（梯度内积）重要性评估、下游动作预测（回归、流匹配、扩散头）微调、以及与量化、稀疏剪枝等传统压缩方法的对比。

**📊 数据集**

实验使用 LIBERO、LIBERO-Plus、RoboTwin 2.0 等仿真数据集，并在真实工厂抓取任务中采集 110K 帧 Meta Quest 3 远程操控数据；另外评估了 OpenVLA-OFT、π_0.5 等 VLA 体系。

**📈 对比分析**

对比方法包括 DTR 与无恢复零射门剪枝、不同重要性度量（Taylor、IGIA、CosSim 等）。结果显示：在 LIBERO 上半块语言块移除后 OpenVLA-OFT 仍达 98.3% 成功率，甚至仅保留 2 块语言块即可 95%；实机环境 Drop‑9 超过基线；硬件加速方面 DTR‑16 实现 1.64× 任务速度提升、内存下降 42%。

**⚠️ 局限性**

局限性在于评估集中在短模板式指令任务，语言冗余结论可能不适用于更复杂或开放式指令；在视觉/物理 OOD 变异下恢复性能下降；需要更具挑战性的语言和环境基准来进一步验证；极端压缩时梯度估计可能不稳定。

---

## 167. Flexformer: Flexible Linear Transformer with Learnable Attention Kernel

**arXiv ID:** 2606.27748 | [PDF](https://arxiv.org/pdf/2606.27748v1)

**作者:** Haoran Zhang `[一作]` (Renmin University of China), Feng Zhou `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Flexformer，一种通过可学习随机傅里叶特征频率构造可变核的线性注意力模型，能够高效处理长序列并支持软max注意力蒸馏与跨域转移。

**💡 创新点**

创新点在于将频率参数化为可学习参数，利用谱表示理论从数据中直接学习任意正定核；并引入 stationary 与非 stationary 两种变体，后者在表达能力上严格优于前者。

**🔧 技术方法**

技术手段包括：线性注意力框架、随机傅里叶特征、谱表示理论、核蒸馏、非 stationary 频率对学习、跨域核迁移。

**📊 数据集**

使用数据集包括：Long Range Arena（LRA）长序列分类、WikiText-103 语言建模、GLUE 语言理解（CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE）以及预训练 RoBERTa。

**📈 对比分析**

与 Transformer、Reformer、Longformer、Linear Transformer、Performer、RFA、Cosformer、Hedgehog、Polaformer 等方法比较；Flexformer 在 LRA 平均准确率提升 4.4%，在 WikiText-103 小模型 PPL 接近软max，甚至在大模型超越；在 GLUE 蒸馏后几乎恢复软max 性能，并在跨域迁移任务中表现更优。

**⚠️ 局限性**

限制在于假设学习到的核为正定且正定核是否为最优选择尚未理论证明，需要进一步研究核正定性的必要性和更广泛的核家族。

---

## 168. SIFT: Self-Imagination Fine-Tuning for Physically Plausible Motion in Video Diffusion Models

**arXiv ID:** 2606.27741 | [PDF](https://arxiv.org/pdf/2606.27741v1)

**作者:** Ruoyu Wang `[一作]` (Wuhan University), Yu Wu `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究视频扩散模型的运动纠缠问题，并提出自我想象微调（SIFT）来提升运动可解耦和物理可行性。

**💡 创新点**

创新点在于打破重建捷径，利用从随机噪声生成的自我想象视频并通过运动判别监督实现运动的物理合理性和独立控制。

**🔧 技术方法**

采用自我想象微调、R3D 与 SlowFast 两种运动分类器、进阶硬样本重放以及 LLM 生成的文本提示进行训练。

**📊 数据集**

使用公开视频数据集与通过 LLM 自动生成的文本提示，构建 4000 条四类运动标签的数据集进行评估。

**📈 对比分析**

与原始模型、传统监督微调（SFT）和 VideoREPA 对比，SIFT 在 VLM 与人工评估的语义一致性（SA）与物理共识（PC）上均优于基线，提升约 0.3–0.4 分。

**⚠️ 局限性**

局限性在于运动标签过于粗粒度，仅区分四类运动，未细化方向、幅度、轨迹或多实体交互，且分类器对噪声视频的鲁棒性有限。

---

## 169. BashCoder-R1: Towards Robust and Explainable Bash Code Generation with Robustness-Aware Group Relative Policy Optimization

**arXiv ID:** 2606.27733 | [PDF](https://arxiv.org/pdf/2606.27733v1)

**作者:** Lei Yu `[一作]` (Institute of Software, Chinese Academy of Sciences), Fengjun Zhang `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BashCoder-R1 框架，通过连续预训练、长链思路监督微调和鲁棒性感知组相对策略优化三阶段训练，实现可解释且鲁棒的 Bash 脚本生成。

**💡 创新点**

首次将持续预训练、长链思路 SFT 与 R-GRPO 结合，构建显式鲁棒性推理与自动化奖励对齐机制，显著提升 Bash 代码的可解释性与可靠性。

**🔧 技术方法**

使用 Qwen2.5-Coder-7B-Instruct 作为基础模型，结合 CPT、L-CoT SFT、R-GRPO；采用 shellcheck 进行鲁棒性奖励；通过 RL 框架实现策略优化。

**📊 数据集**

CPT 使用 676,524 条 Bash 代码与 300k 常规代码；SFT 生成 12,334 条带 reasoning 的样本（7,005 单行、5,329 多行）；R-GRPO 选取 1,824 个挑战样本；评测采用 BashBench 的 952 真实自动化任务。

**📈 对比分析**

与多种通用 LLM、Code LLM 与 Reasoning LLM 进行对照，采用 SyntaxPass、RobustWarnRate、RobustPass、FuncRate、FullRate 五个指标；单行任务 FullRate 90.04%、多行任务 73.18%，明显优于 DeepSeek-V3.2 的 65.33% 与 60.89%。

**⚠️ 局限性**

奖励函数仅覆盖语法、鲁棒性与格式，可能忽略代码简洁性与风格；模型在超出训练分布的新任务或不同操作系统环境中的泛化能力有限。

---

## 170. MASS: Motion-Aligned Selective Scan for Refinement in Flow-Based Video Frame Interpolation

**arXiv ID:** 2606.27718 | [PDF](https://arxiv.org/pdf/2606.27718v1)

**作者:** Jun-Sang Yoo `[一作]` (Korea University), Seung-Won Jung `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种Motion‑Aligned Selective Scan (MASS) 框架，将 VFI 的特征扫描从静态网格改为基于运动轨迹的序列建模，实现更好的运动一致性和细节恢复。

**💡 创新点**

创新点在于学习非线性路径积分和速度感知自适应扫描，动态调整采样密度和 SSM 步长，使得特征聚合与实际运动轨迹保持一致，并利用双向一致性进行流和遮挡的精细修正。

**🔧 技术方法**

技术主要包括可学习的非线性轨迹生成、Velocity‑Aware Selective State Space Model (VA‑SSM)、双向轨迹一致性约束以及基于重构和融合的生成模块。

**📊 数据集**

使用 Vimeo90K、UCF101、Middlebury、SNU‑FILM、Xiph 等标准数据集进行训练与评估。

**📈 对比分析**

在多个基准上与最新方法（如 VFIFormer、VFIMamba、LC‑Mamba 等）对比，MASS 在大运动、复杂遮挡以及高分辨率场景下取得了最优或接近最优的 PSNR/SSIM，并在 SNU‑FILM Extreme 子集上领先 0.38 dB。

**⚠️ 局限性**

局限性是过度依赖初始粗略流的质量，若粗估失效（如极端运动模糊或微小目标），轨迹引导扫描可能无法恢复正确语义，且目前未实现流与 SSM 的端到端联合训练。

---

## 171. Aurora: A Leverage-Aware Spectral Optimizer

**arXiv ID:** 2606.27715 | [PDF](https://arxiv.org/pdf/2606.27715v1)

**作者:** Alec Dewulf `[一作]` (Tilde Research), Ben Keigwin `[通讯]` (Tilde Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Aurora 优化器，针对 Muon 在 MLP 层的“行范数不均匀”导致的神经元死亡问题，采用行均匀化与极值分解相结合的迭代更新方式，在保持 Muon 极分解几何的前提下提升更新质量。

**💡 创新点**

创新点在于：① 通过对 Muon 的极分解和行范数约束联合建模，推导出“行均匀化 + 极分解”双约束优化问题；② 设计可在分布式训练中实现的 Aurora 迭代（含阻尼），实现局部线性收敛；③ 通过 U‑NorMuon 把行均匀化仅应用于高宽比（tall）矩阵，验证行均匀化对神经元死亡的显著抑制。

**🔧 技术方法**

核心技术包括：Newton‑Schulz 极分解迭代、行均匀化（RMS/极分解缩放）、Riemannian 极分解投影（参考解法）、多步阻尼迭代、梯度动量（Muon's momentum buffer）和统计利用率/归一化技巧。

**📊 数据集**

使用 NemotronCC v2 100B token 预训练数据，评估基准包括 MMLU、HellaSwag、PIQA、WinoGrande、ARC‑Challenge/Easy、LAMBADA、OpenBookQA 等通用知识推理任务。

**📈 对比分析**

与 Muon、NorMuon、U‑NorMuon 在 340M 与 1.1B 参数规模下对比：Aurora 在所有指标上均优于基线；MMLU 得分比 Muon 提升 9.1 分；在 speedrun 优化赛道上达到 3.28 目标 loss，减少 25 步；在 MLP 宽度扩展实验中，Aurora 的优势随宽度增加而增大。

**⚠️ 局限性**

局限性：① 额外迭代导致的计算与内存开销（尤其是 K>2 时），但可通过梯度通信重叠缓解；② 只针对 tall 矩阵设计，未针对宽/方阵或其他参数类型；③ 需要在不同模型/任务中调优阻尼 β、迭代次数 K；④ Riemannian-Aurora 作为参考解法在实际规模下不可行；⑤ 当动量趋于稳定时可能可切换回 Muon，仍待实验验证。

---

## 172. Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning

**arXiv ID:** 2606.27709 | [PDF](https://arxiv.org/pdf/2606.27709v1)

**作者:** Austin MY Cheung `[一作]` (Hong Kong University of Science and Technology), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究者通过构建低同意性（Agreeableness）用户发言与温和降级（de‑escalating）助手回复的重写管线，对四个LLM进行微调，探究情感温暖与安全性之间的关系。

**💡 创新点**

证明安全性下降并非温暖情感固有缺陷，而是数据共现统计导致；仅通过数据构成（低同意性对话+温和拒绝）即可在不使用安全标签或目标改动的情况下恢复安全性。

**🔧 技术方法**

采用LLM重写的语料重构管线、LoRA微调、Jailbreak与红队安全评测以及温暖与遵从方向余弦角度探针等技术。

**📊 数据集**

使用PersonaFuse（低Agreeableness子集）、EmpatheticDialogues、ESConv、Lahnala-style、ShareGPT Vicuna Unfiltered、MentalChat‑16K等多种对话数据集。

**📈 对比分析**

通过与未微调基线模型比较Jailbreak成功率与有害输出率，实验显示低同意性条件在大多数模型上显著降低两项指标，并保持或提升对话温暖度。

**⚠️ 局限性**

局限性包括：仅评估单轮攻击，多轮升级鲁棒性未系统验证；重写过程可能引入噪声，且对不同模型的效用差异较大。

---

## 173. NLL-Guided Full-Attention Layer Selection for Training-Free Sliding-Window Adaptation

**arXiv ID:** 2606.27791 | [PDF](https://arxiv.org/pdf/2606.27791v1)

**作者:** Qiong Tang `[一作]`, Yunfan Shao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的NLL-guided层选择方法，用于混合全注意力与滑动窗口注意力的长上下文LLM推理。

**💡 创新点**

直接通过对答案token的负对数似然衰减量衡量每层对全注意力的依赖，自动挑选最重要的层。

**🔧 技术方法**

采用NLL评分、SWAA滑动窗口适配、一次性前向校准、贪婪层挑选与去偏分析。

**📊 数据集**

在LongMemEval基准上使用Qwen3-4B-Thinking-2507模型，校准样本来自LongAlign-10k与fusang-v1-filtered，评估由GPT-5-mini判定。

**📈 对比分析**

与全注意力、周期性1/2-FA、周期性1/4-FA、LightTransfer 1/4-FA以及全SWA对比；使用1/4全注意力层得到64.6%准确率，几乎等同于1/2-FA周期性（65%），比1/4-FA周期性提升10.4个百分点，超过LightTransfer 26.4个百分点。

**⚠️ 局限性**

仅在单一模型和基准上验证，未测试跨模型/任务的泛化；层选择为静态，未考虑输入动态性。

---

## 174. An Embedded Real-Time License Plate Recognition System for Complex Traffic Scenes

**arXiv ID:** 2606.27772 | [PDF](https://arxiv.org/pdf/2606.27772v1)

**作者:** Anuki Pasqual `[一作]` (University of Moratuwa), Udaya S. K. P. Miriya Thanthrige `[通讯]` (University of Moratuwa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对发展中国家复杂交通场景，构建了嵌入式实时车牌识别系统，包含车牌检测与字符识别两大轻量化 CNN 模型，并在 Xilinx Kria KV260 FPGA 平台上实现加速。

**💡 创新点**

创新点在于：① 提出了专为低功耗嵌入式设计的低位宽量化 YOLO（LPYOLO）检测模型与改进的 fast-plate-ocr 字符识别模型，参数量仅 1.4M；② 设计了 SL-LPR 数据集，涵盖多车型、多车道、复杂光照的真实场景；③ 将量化模型直接编译为 FPGA IP，结合 FINN 框架实现端到端高效推理。

**🔧 技术方法**

采用的技术包括：Brevitas 量化库（4/2/1-bit 训练），YOLO 变体（LPYOLO），fast-plate-ocr 变体（LPCR），FPGA 加速（FINN + Xilinx Kria KV260），以及图像预处理（亮度增强、尺寸缩放）和后处理（NMS、字符置信度阈值）。

**📊 数据集**

使用 SL-LPR 数据集（2970 帧、3412 车牌图像）进行训练和评估，并在公开数据集（PKU、UFPR-ALPR、AOLP）上做对比验证。

**📈 对比分析**

与公开实现相比，检测模型在 SL-LPR 上 AP 93.6%，召回率 92.2%，显著优于 CPU 实现；字符识别模型在 SL-LPR 上 plate 87.9% 及 97.4% 字符准确率，推理时间仅 4 ms/车牌；整体系统在 KV260 上达到 11.5 FPS、最大 87 ms/帧，功耗 4.2 W，优于 GPU 方案。

**⚠️ 局限性**

局限性包括：对远距离小车牌检测仍有漏检；夜间或极端光照条件下性能下降；数据集主要来源于特定地区，缺乏夜景与多地理变异；系统仍需进一步优化帧缩放瓶颈以提升吞吐量。

---

## 175. DE-2LS: Differential Evolution with Late-Stage local-search for Unconstrained Single-Objective Numerical Optimization

**arXiv ID:** 2606.27762 | [PDF](https://arxiv.org/pdf/2606.27762v1)

**作者:** Dikshit Chauhan `[一作]` `[通讯]` (National University of Singapore), Dikshit Chauhan (National University of Singapore)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DE-2LS，一种在RDEx基础上加入晚期局部搜索的差分进化算法，用于无约束单目标数值优化。

**💡 创新点**

创新点在于：①引入平滑的利用率分支比率更新机制，稳定晚期搜索中的利用偏向；②设计了受限坐标模式局部搜索（Guarded Coordinate‑Pattern LS），在评估预算剩余时对全局最优解进行细粒度改进，保持RDEx的全局搜索优势。

**🔧 技术方法**

使用技术包括：RDEx的成功历史参数自适应、排名选择、利用率分支控制、标准分支高斯采样、以及基于目标值的单目标选择与局部搜索实现。

**📊 数据集**

在CEC 2026 单目标优化基准（29 个无约束测试函数，D=30，最大评估次数 300000）上进行实验，并采用官方 U‑score 评估方法。

**📈 对比分析**

与 RDEx 以及 LSRTDE、BlockEA、mLSHADE、jSO、IEACOP 等竞争算法进行配对 U‑score 比较；DE‑2LS 在 U‑score 上提升 11.45%（从 33602 到 37448），总 U‑score 178966.5，超过竞争者 34.43%。

**⚠️ 局限性**

局限性包括：局部搜索仅在变量边界约束下使用，未考虑更一般约束；LS 的触发阈值和步长参数仍需手动设定，缺乏完全自适应；在某些问题上相对 RDEx 的速度提升有限。

---

## 176. PerturbCellRL: Verifier-Guided Reinforcement Learning for Single-Cell Perturbation Prediction

**arXiv ID:** 2606.27752 | [PDF](https://arxiv.org/pdf/2606.27752v1)

**作者:** Dongxia Wu `[一作]` (Stanford University), Emily B. Fox `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了 PerturbCellRL 框架，利用强化学习在已有单细胞流匹配生成器上进行后训练，结合四种生物学验证器来提升单细胞水平的预测一致性。

**💡 创新点**

创新点在于将生物学验证器作为奖励函数，利用 RL 进行后训练，并引入无参考的通路活性奖励实现推理时的最佳- N 采样；同时通过对已预训练模型的正则化避免奖励劫持。

**🔧 技术方法**

技术包括基于 scDFM 的单细胞流匹配模型、DiffusionNFT 强化学习算法、Pearson top-k、RMSE top-k、DE Spearman 和 PROGENy 通路活动四个验证器。

**📊 数据集**

实验使用 Norman 数据集（additive 与 holdout 方案）和 ComboSciPlex 数据集进行遗传与化学扰动预测。

**📈 对比分析**

与 Control、Additive、GEARS、CPA、STATE、CellFlow、scDFM 等基线比较，在单细胞奖励指标上显著提升，且在群体级别指标（MAE、Pearson Δ、DS、MMD 等）保持或略优于 scDFM。

**⚠️ 局限性**

局限在于验证器的覆盖范围和准确性，通路活性奖励仅适用于单基因扰动，需要更丰富的注释；并且仍需外部实验验证高分预测。

---

## 177. UNICS: Multilingual Code Search via Unified Pseudocode and Contrastive Transfer Learning

**arXiv ID:** 2606.27747 | [PDF](https://arxiv.org/pdf/2606.27747v1)

**作者:** Ye Fan `[一作]` (Nanjing University), Bin Luo `[通讯]` (Nanjing University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 UNICS 框架，利用统一伪代码表示和多任务对比学习实现多语言代码检索。

**💡 创新点**

创新点：① 通过无损伪代码抽象跨语言算法逻辑，消除语法噪声；② 设计语义切片、硬正样本挖掘与动态跨语言硬负样本挖掘的多任务训练；③ 两阶段预训练＋迁移学习策略，兼顾全局抽象与语言细粒度。

**🔧 技术方法**

技术：LLM 指令式伪代码生成、InfoNCE 对比学习、语义代码切片、硬正样本挖掘、动态硬负样本队列、跨语言迁移学习。

**📊 数据集**

数据集：自构造的多语言统一伪代码指令数据集；CodeSearchNet、CoSQA、APPS、XLCoST、StackOverflow QA、CodeFeedBack、CodeTransOcean、NicheLang 等公开基准。

**📈 对比分析**

与 OpenAI Ada、text‑embedding‑3‑small、BGE、Contriever、CodeRetriever、UniXCoder 等基线对比，在多语言检索、跨语言检索和低资源语言检索任务中均实现 SOTA。平均 MR R、NDCG、Top@10 均提升约 15‑25%，零样本迁移性能尤为突出。

**⚠️ 局限性**

局限性：伪代码生成可能带来 LLM 知识偏差；对极少见 API 或细粒度功能差异的区分仍有限；仅在 1B 参数模型上验证，未覆盖更大模型或不同架构；硬负样本挖掘依赖队列大小与更新频率。

---

## 178. Output-Space Allocation Costs for Calibration-Guided LLM Compression: An Empirical Study

**arXiv ID:** 2606.27785 | [PDF](https://arxiv.org/pdf/2606.27785v1)

**作者:** Qiong Tang `[一作]` (Analemma), Yunfan Shao `[通讯]` (Analemma)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对ROCKET压缩方法，提出在多选背包分配中使用输出空间误差代替权重空间误差，并选择对应的输出最优稀疏配置。

**💡 创新点**

创新点在于将分配成本与层级分解目标对齐，使得全局压缩预算更加关注激活分布下的输出误差，从而在任务准确率上获得提升。

**🔧 技术方法**

采用训练自由低秩分解、白化激活空间误差计算以及多选背包（MCKP）求解技术；所有成本均在已完成的校准步骤中无额外计算开销。

**📊 数据集**

使用 Qwen3-8B 与 Llama-3.2-1B 两个大模型；校准数据来自 RefinedWeb；评估基准包括 lm-eval-harness 上的 8 个零样本任务和 WikiText-2 语言模型困惑度。

**📈 对比分析**

与原 ROCKET-default 直接对比；在 Qwen3-8B 50% 压缩下，平均准确率提升 0.8pp，WikiText PPL 上升 16%；在 Llama-3.2-1B 20% 压缩下，两者差异几乎不存在。

**⚠️ 局限性**

局限性：输出误差与权重误差相关度 >0.99，限制了分配决策的差异，导致改进幅度受限；在轻度压缩时几乎无效，且仅在激进压缩场景下才显现明显收益。

---

## 179. Understanding Rollout Error in Graph World Models

**arXiv ID:** 2606.27780 | [PDF](https://arxiv.org/pdf/2606.27780v1)

**作者:** Xinyuan Song `[一作]` (Emory University), Zekun Cai `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究图世界模型（Graph World Models, GWM）在长时间步滚动预测中的误差累积，提出统一的固定边（FE）与动态边（DE）框架，并设计 Error‑Aware GWM 通过谱正则、滚动一致性与关键节点加权提升长期稳定性。

**💡 创新点**

① 用谱半径与模型权重范数分离出的图误差放大因子 (GEAF) 定量描述拓扑对误差扩散的影响；② 引入耦合节点‑边误差算子 B，阐释 DE 模型下节点与边误差的交互；③ 结合三项正则化的 Error‑Aware 训练目标，实现对长期滚动误差的主动抑制。

**🔧 技术方法**

基于 GNN 的消息传递网络（GCN、MPNN、GPS）、谱正则化、滚动一致性正则、关键节点权重、动态边预测头、节点‑边耦合算子、图谱分析、奖励与规划误差连接等技术。

**📊 数据集**

合成图谱（链、树、格、小世界、无标度、星、完全图）、异构代理图测试床（调用树、技能图）、真实图数据集（Cora、Citeseer、Bitcoin‑Alpha）以及多种基准（MPNN、GPS、ActionNode 等）。

**📈 对比分析**

通过多模型对照（GNN、ActionNode、Error‑Aware）在不同拓扑与滚动长度下比较节点误差、规划损失、任务完成率；在真实任务中与专用图分类器（GCN、GAT、GraphSAGE）对比；结果显示 Error‑Aware GWM 在高谱半径图上避免了误差爆炸、在异构代理图上保持低误差；但在静态分类和稀疏时间链任务中相对表现不占优势。

**⚠️ 局限性**

主要局限在于实验集中于受控合成环境；在真实稀疏/大型图上 Dynamic‑Edge 模型并未持续显著优于 FE；joint‑operator 理论在 DE 数据集上验证有限；缺乏对更大规模稠密时间图、实时多智能体系统等场景的评估。

---

## 180. MindFlow: Harmonizing Cognitive Semantics and Acoustic Dynamics for Facial Animation Generation in Dyadic Conversations

**arXiv ID:** 2606.27779 | [PDF](https://arxiv.org/pdf/2606.27779v1)

**作者:** Hejia Chen `[一作]` (Beihang University), Shuai Li `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究提出MindFlow框架，实现面向双人对话的实时流式面部动画生成；

**💡 创新点**

创新点在于结合神经科学的Ventual‑Dorsal双路径结构，推出Chunk‑State连续情感状态推理与Selective Acoustic Injector自适应音频门控，解决传统方法语义缺失与粗粒度问题；

**🔧 技术方法**

技术实现包括多模LLM（MLLM）进行情感推理、基于自回归Transformer的Selective Acoustic Injector、以及流匹配（flow‑matching）生成网络；

**📊 数据集**

使用了HDTF、VICOX、MEAD、VICO共约20小时的音视频数据集；

**📈 对比分析**

与EmoTalk、UniTalker、DualTalk等SOTA方法在FD、SyncNet、MSE等指标上进行量化对比，MindFlow在说话与聆听两阶段均取得最优或次优成绩，生成更自然、更同步的面部动画；

**⚠️ 局限性**

局限在于仅依赖音频输入，未结合视觉或身体线索，可能在无声但语义丰富的交互场景下表现不足。

---

## 181. Deriving Approximate Message Passing from the Convex Gaussian Min-Max Theorem

**arXiv ID:** 2606.27769 | [PDF](https://arxiv.org/pdf/2606.27769v1)

**作者:** Vikrant Malik `[一作]` (California Institute of Technology), Babak Hassibi `[通讯]` (California Institute of Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过凸高斯极小极大定理（CGMT）直接推导出近似信息传递（AMP）与最大求和GAMP的固定点方程，证明在高维比例极限下，当CGMT的辅助优化（AO）与原始优化（PO）共享同一原始-对偶解时，CGMT能够自动产生AMP的Onsager校正，并将AO中的高斯向量解释为AMP中的有效噪声通道。

**💡 创新点**

创新点在于：① 提供了一个直接、统一的框架将CGMT与AMP/ GAMP关联起来，避免了以往间接或经验性推导；② 通过匹配AO与PO的解，揭示了Onsager校正系数的本质来源；③ 将AO中的高斯向量与AMP的原始与残差通道噪声对应，给出新的解释视角；④ 扩展到一般可分离凸输出损失，得到GAMP的固定点。

**🔧 技术方法**

主要技术包括：凸高斯极小极大定理（CGMT）及其AO/PO结构；高维随机优化的极限分析；对偶与KKT条件的匹配；对接AMP的状态演化；以及对可分离凸正则化和输出损失的Fenchel对偶推导。

**📊 数据集**

本文属于理论分析工作，未使用具体数据集；研究的假设是高维比例极限、Gaussian设计矩阵、可分离凸正则化与损失。

**📈 对比分析**

由于是理论证明，未做实验比较；论文通过解析推导表明在满足假设的前提下，AMP/ GAMP的状态演化与CGMT得到的标量方程完全一致，说明两种框架在高维Gaussian设计下具有相同的性能预测。

**⚠️ 局限性**

局限性包括：① 需要Gaussian设计矩阵与高维比例极限，实际应用中可能不满足；② 需要假设AO与PO共享同一解并且存在共同的次梯度，证明仅在理论上成立，实际实现可能困难；③ 对非凸正则化、非Gaussian设计或小样本场景的推广尚未解决。

---

## 182. PixelU: A U-Shaped Transformer for Efficient End-to-End Pixel Diffusion

**arXiv ID:** 2606.27760 | [PDF](https://arxiv.org/pdf/2606.27760v1)

**作者:** Zipeng Guo `[一作]` (JD.com), Yan Li `[通讯]` (JD.com)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PixelU，一种单阶段 U‑形 Transformer，用于端到端像素空间扩散生成，采用 x‑prediction 并通过跳连结与单一空间下采样实现频率解耦。

**💡 创新点**

创新点：1) 证明在 x‑prediction 下复杂像素解码器多余；2) 用零成本跳连结直接传递高频细节；3) 通过常数通道下采样实现低通滤波，聚焦低频语义；4) 仅一个下采样阶段即可完成频率分离，显著降低计算量。

**🔧 技术方法**

技术手段：Transformer 编码/解码结构、x‑prediction（流匹配目标）、跳连结、常数通道空间下采样、REPA、CFG 以及 Heun 采样等。

**📊 数据集**

数据集：ImageNet‑1K，分辨率 256×256 与 512×512。

**📈 对比分析**

与现有像素空间模型（如 JiT‑G、DeCo、PixelFlow、ADM‑G 等）和部分潜在空间模型（DiT‑XL/2、SiT‑XL/2）对比，PixelU‑H/16 在 256×256 上实现 FID 1.63、IS 305.88，512×512 上 FID 1.92、IS 322，计算量约为 JiT‑G 的 1/3，显著提升效率和生成质量。

**⚠️ 局限性**

局限性：仍略逊于最先进的潜在空间扩散模型；对多尺度生成的适配需要进一步探索；在 x‑prediction 之外的任务/数据分布中表现未知。

---

## 183. Layerwise Progressive Freezing: A Training Scaffold for Depth-Scalable Binary Networks

**arXiv ID:** 2606.27759 | [PDF](https://arxiv.org/pdf/2606.27759v1)

**作者:** Evan Gibson Smith `[一作]` (Worcester Polytechnic Institute), Bashima Islam `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种层级进阶的二值化训练方法StoMPP，逐层将权重与激活从可微的clip转为硬二值化；

**💡 创新点**

创新点在于将二值化的时间与位置分层进阶，并通过随机遮罩和软刷新避免梯度阻塞，展示进阶顺序决定深度网络可训练性；

**🔧 技术方法**

技术包括层级进阶遮罩、随机遮罩与软刷新、可选的STE-free或StoMPP+STE两种梯度处理方式；

**📊 数据集**

使用CIFAR-10/100、ImageNet、MobileNetV2、BERT（SST-2）等数据集进行实验；

**📈 对比分析**

与传统STE baseline对比，StoMPP在ResNet-18/34/50、MobileNetV2以及BERT微调中在深度网络上提升明显，STE-free版在ResNet-50上相对STE提升约+18%（CIFAR-10）、+13.5%（CIFAR-100）、+3.8%（ImageNet），StoMPP+STE进一步提升至+27.1%/ +19.8%/ +17.7%；

**⚠️ 局限性**

局限在于未与SOTA混合蒸馏、学习率调度或专门的架构改进结合；进阶顺序、刷新率等超参数仍需调节；未对梯度阻塞机制做直接干预实验。

---

## 184. Panoramic Scene Analysis: A Survey from Distortion-Aware Engineering to Sphere-Native Foundation Modeling

**arXiv ID:** 2606.27745 | [PDF](https://arxiv.org/pdf/2606.27745v1)

**作者:** Qinfeng Zhu `[一作]` (University of Liverpool), Lei Fan `[通讯]` (Xi'an Jiaotong-Liverpool University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对全景场景分析进行了系统综述，梳理了从投影适配、失真感知工程到球面原生建模以及与基础模型接口适配的演进路径，提出了二维方法-训练范式的体系，并指出评估与数据集的不足。

**💡 创新点**

创新点在于提出以几何为先的两维分类法，识别出结构性失真与球面等价性冲突，归纳出五大评估缺口与六大研究路线，形成面向通用全景智能的路线图。

**🔧 技术方法**

综合了投影分解、失真补偿卷积/Transformer、球面卷积/图卷积、几何感知位置编码、基础模型自适应、域自适应及多任务统一等技术，阐释了各方法在球面几何与预训练兼容性上的权衡。

**📊 数据集**

主要使用了DensePASS、WildPASS、Stanford2D3D、Matterport3D、Structured3D等密集标注全景数据集，以及Pano-AVQA、OSR-Bench等多模态与开放世界评测集。

**📈 对比分析**

通过对比分析，展示了从PB到SN再到GT+FMA等方法在语义分割、深度估计、房间布局等任务上的性能提升，但仍普遍存在约50% mIoU下降、未充分利用全景空间一致性等问题。

**⚠️ 局限性**

局限性包括缺乏从零开始预训练的球面基础模型、评估指标未考虑球面面积加权和边缘连续性、数据集规模和室外标注不足，以及缺少跨投影、跨视角的通用验证框架。

---

## 185. KG2Cypher: Data-Centric Pipeline for Building Enterprise Text-to-Cypher Systems

**arXiv ID:** 2606.27742 | [PDF](https://arxiv.org/pdf/2606.27742v1)

**作者:** Minjun Choi `[一作]` (Sungkyunkwan University), Youngjoong Ko `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一套基于企业知识图谱的全流程文本到 Cypher 查询生成系统 KG2Cypher，涵盖数据构造、LLM 判定、候选感知 SFT 以及类条件 schema prompting，最终实现可在企业 KG 上部署的自然语言接口。

**💡 创新点**

创新点在于利用 KG 实际事实直接生成可执行 Cypher 作为监督目标，仅将 LLM 用于语言重写与质量判定，避免符号错误；同时引入候选感知 SFT 与类条件 schema prompting，解决关系检索低召回瓶颈。

**🔧 技术方法**

主要技术包括 gpt‑oss‑120B 用于查询生成、判定与多样化扩展；LoRA 微调 Llama‑3.1‑8B‑Instruct 生成模型；KLUE‑BERT‑base 进行领域分类；vLLM 加速推理；以及规则、模板与实体检索接口。

**📊 数据集**

使用了私有的韩国企业 KG，包含广播与公司两大类，并扩展到 11 个图谱类别，生成约 26 000 条训练/验证/测试的文本–Cypher 对。

**📈 对比分析**

与 prompt‑only LLM 基线对比，采用 EM、执行率和执行结果 F1 三项指标；LoRA SFT 将广播 F1 从 0.806 提升至 0.950，公司的 F1 从 0.70 提升至 0.92，整体 11 类场景下 EM 达 95.2%、执行率 99.9%、F1 0.964。

**⚠️ 局限性**

局限性包括只能覆盖已有 KG 的关系/实体，缺失时无法生成监督；依赖实体检索与类分类，外部实体或同义实体仍难处理；语言特定组件需重新适配；数据与模型未公开，难以复现和迁移。

---

## 186. The Weakest Link Tells It All: Outcome-Supervised Process Reward Modeling via Learnable Credit Assignment

**arXiv ID:** 2606.27739 | [PDF](https://arxiv.org/pdf/2606.27739v1)

**作者:** Tianyu Jia `[一作]` (Peking University), Yasha Wang `[通讯]` (Peking University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于弱链原则的Outcome‑Supervised Process Reward Modeling框架，并在此基础上设计了Softmax‑Weighted‑Sum (SWS) 池化方法，实现无步骤级标签的过程奖励学习。

**💡 创新点**

创新点在于：① 将过程奖励建模转化为多实例学习问题，利用弱链赋权原则形成的标准多实例假设；② 设计SWS池化在强依赖冗余实例场景下自适应分配信用，避免传统Max/Attention池化导致的“只关注最后一步”问题；③ 证明在适当温度下SWS池化的贝叶斯一致性。

**🔧 技术方法**

核心技术包括：多实例学习（MIL）框架、Softmax‑Weighted‑Sum池化、预训练LLM骨干 + 分类头、基于交叉熵的全局监督训练，以及对温度参数的理论与实验分析。

**📊 数据集**

使用的数据集有：Math‑Shepherd（训练集，去除步骤标签仅保最终答案）；ProcessBench（错误定位评估）；MATH‑500（最佳‑of‑N 采样评估）；另外在实验中还使用了Qwen2.5‑Math‑7B、Llama3.2‑3B 等不同大小的LLM骨干。

**📈 对比分析**

与ImplicitPRM、MathShepherd、SCAN、OmegaPRM、PQM等现有基线（以及公开的LLM‑as‑Critic模型）进行对比；在ProcessBench上F1均优于第二名约21.7%（3B骨干）或5.5%（7B骨干）；在MATH‑500的best‑of‑N 评估中，在N=32/64时均取得最高平均分，显示出良好的可扩展性和跨模型泛化能力。

**⚠️ 局限性**

局限性包括：仍低于使用人工步骤标签训练的PRM；最终答案的正确性作为轨迹标签时可能引入噪声，导致弱链假设不严格；在强化学习中应用时易出现奖励欺骗，需进一步研究奖励设计与策略共演方法。

---

## 187. HandMade: Spatial Prompting for Generative 3D Creation with Part-Labeled VR Sketches

**arXiv ID:** 2606.27738 | [PDF](https://arxiv.org/pdf/2606.27738v1)

**作者:** Jialin Huang `[一作]` (George Mason University), Yotam Gingold `[通讯]` (George Mason University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种将VR 3D草图与文本描述相结合的工作流，利用多视角引导生成开源域文本到3D模型的管线。

**💡 创新点**

创新点在于把粗略、带标签的VR 3D草图视为空间提示，通过将其渲染成多视角颜色引导并与结构化文本提示结合，既保留手绘的布局，又充分利用文本的语义信息。

**🔧 技术方法**

使用的技术包括：Meta Quest Pro 的手势VR绘制接口；ChatGPT Images 2.0 的图像多视角生成；Hunyuan3D-2mv 的多视角图像到3D重建；以及结构化提示构造、颜色-部件映射、手绘稀疏线条处理等中间处理步骤。

**📊 数据集**

数据集为20个开放域手绘3D草图与对应文本的自建集合，涵盖建筑、车辆、家具、角色等多样化场景；此外在评测中使用了多种公开基线（Instant3dit、SketchDream、LAS‑Diffusion、MeshPad 等）和人类标注的几何对齐指标。

**📈 对比分析**

在20例对比中，该方法在 Sketch‑to‑Mesh 距离、Chamfer 距离、Recall@0.05/Precision@0.05/F‑score@0.05 等指标上均优于文本仅、单视角、以及公开基线，平均 Recall@0.05 达 0.792，F‑score@0.05 达 0.727，显示其在保留草图空间信息方面具有显著优势。

**⚠️ 局限性**

主要局限在于：对粗略草图的空间约束只能是粗糙的，难以表达细节与精确尺寸；多视角生成与重建对不一致或误导的草图仍易产生误差；交互过程需人工介入，缺乏自动化修订与精细控制；目前未支持精细拓扑、尺寸约束和专业级编辑。

---

## 188. ToE: A Hierarchical and Explainable Claim Verification Framework with Dynamic Multi-source Evidence Retrieval and Aggregation

**arXiv ID:** 2606.27736 | [PDF](https://arxiv.org/pdf/2606.27736v1)

**作者:** Zhaoqi Wang `[一作]` (Beijing Institute of Technology), Jiamou Liu `[通讯]` (University of Auckland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Tree of Evidence（ToE），一种层次化、可解释的主张验证框架，通过强化学习驱动的多源检索、证据评估与论证树聚合，实现对主张的动态拆分与证据链推理。

**💡 创新点**

创新点包括：①将主张建模为可动态扩展的论证树，自动拆分子主张；②使用强化学习检索代理在多源环境中主动搜索支持与反驳证据；③提供理论检索误差上界与子模性证明；④构造针对GEO毒化的对抗数据集AdvFact。

**🔧 技术方法**

核心技术包括：POMDP框架与信息论奖励设计、GRPO强化学习检索、LLM生成检索查询与证据提取、姿态感知多路径注意力的评估网络、树聚合与剪枝算法。

**📊 数据集**

使用的数据集有LIAR、PolitiFact、Check‑COVID以及自建的AdvFact，所有方法仅在LIAR训练集上无针对性微调，直接迁移到其他数据集。

**📈 对比分析**

与Direct、ZCoT、DefGen、AFaCTA、TELLER、STEEL、AdSent等六种基线在DeepSeek‑V3.2与gpt‑oss‑20b上对比，ToE在四个数据集上均实现4–24个百分点的准确率提升，尤其在对抗样本AdvFact上提升显著。

**⚠️ 局限性**

局限性包括：对LLM推理质量依赖度高，计算成本高；论证树深度与检索步骤受硬性限制；在极度稀缺或伪造证据场景下仍可能失效；并未针对不同语言或更细粒度可信度标注进行扩展。

---

## 189. Enhancing Numerical Prediction in LLMs via Smooth MMD Alignment

**arXiv ID:** 2606.27731 | [PDF](https://arxiv.org/pdf/2606.27731v1)

**作者:** Zhuo Zuo `[一作]` (Sichuan University), Xianggen Liu `[通讯]` (Sichuan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对大型语言模型数值预测的 SMMD 损失，通过构造数值子词表的距离引导核，对预测分布进行 MMD 对齐并对残差做平滑正则化，提升数值输出准确率。

**💡 创新点**

将最大均值差距（MMD）与图结构平滑结合，利用数值距离构造核函数，同时在残差上做 Dirichlet 能量正则，形成轻量级、可插拔的数值监督方式。

**🔧 技术方法**

基于 RBF 核的距离诱导核、MMD 对齐、图拉普拉斯平滑、交叉熵混合损失、可微的 softmax 以及数值子词表。

**📊 数据集**

在数学推理（GSM8K、SVAMP）、算术计算（DeepMind-Math）、钟表时间识别（Clock-Time）和图表问答（ChartQA）等四类数值输出任务上进行评估，使用多种开源 LLM/VLM 模型。

**📈 对比分析**

与标准交叉熵、Gaussian CE、NTL、NTIL 等数值目标损失对比，SMMD 在各任务上均显著提升精确匹配率（如 GSM8K 提升至 +约3%），并在多模型、多规模上保持鲁棒性。

**⚠️ 局限性**

受限于分词器导致的数值距离定义、仅适用于数值型目标不适用于标识符；以及需调节 λ 与 σ 等超参，且在某些任务（如 Clock-Time）增量有限。

---

## 190. Beating Trivial Time for Tricky Triangle Tasks

**arXiv ID:** 2606.27727 | [PDF](https://arxiv.org/pdf/2606.27727v1)

**作者:** Neha Pant `[一作]` (Massachusetts Institute of Technology), Ryan Williams `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了稀疏图中三角形和4环检测的细粒度复杂度问题，给出在Word RAM上改进的算法。

**💡 创新点**

通过压缩列表、0级电路和线性哈希的组合，实现了对All-Edges Sparse Triangle、Exact Triangle和4环检测的log、word因子加速。

**🔧 技术方法**

使用压缩列表排序、指示掩码、线性哈希、概率化简、元素不同性、4环化简等技术。

**📊 数据集**

论文主要基于理论模型，无实验数据集，所有结果均为理论上限。

**📈 对比分析**

与传统枚举算法相比，取得了O(n^{1+2δ}log w /w)等子多项式加速，Exact Triangle到O(n^3 log w /w^{3/2})，4环检测到O(n^2 log^2 n /w)。

**⚠️ 局限性**

受限于稀疏度上限、随机性依赖、对最高度无优化，以及对更一般稀疏图或确定性算法的缺乏。

---

## 191. Do Speech Emphasis Models Generalize across Languages and Emotions?

**arXiv ID:** 2606.27717 | [PDF](https://arxiv.org/pdf/2606.27717v1)

**作者:** Megan Wei `[一作]` (Adobe Research), Zeyu Jin `[通讯]` (Adobe Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了MMEE（Multilingual Multi-Emotion Emphasis）语料库并对多语言、多情感下的强调检测模型进行大规模实验。

**💡 创新点**

创新点包括①提供首个10,000句、7语种、34情感类别、3级人类注释的多语言强调语料；②系统评估模型在跨语种、跨情感、跨数据集以及不同训练规模下的泛化能力。

**🔧 技术方法**

采用了两种最新的强调检测框架：基于XLS-R的EmphaClass（可做二分类或回归）和基于Whisper的WhiStress（加语言条件的多语言解码器+分类头），并在MMEE上进行微调。

**📊 数据集**

使用的数据集为自建的MMEE（10,000句，14.13小时，7宏语种+10方言，34情感/语调类别），对比评估时也使用了EmphAssess和TinyStress-15K等公开基准。

**📈 对比分析**

通过固定train/val/test拆分，在monolingual、cross‑lingual、multilingual、arousal、数据规模、cross‑dataset等多维度设置下，用二进制准确率、F1、Pearson相关等指标比较，结果显示多语种训练显著提升跨语言鲁棒性，跨情感迁移良好，数据规模在前几千样本时收益最大，且在跨数据集时实现了双向良好迁移。

**⚠️ 局限性**

局限性包括中文语料因声调与重音交叉导致表现不佳、标注一致性仍有提升空间、实验仅覆盖7宏语种且未深入低资源语言、模型仍依赖大规模预训练语音模型。

---

## 192. Improving Adversarial Robustness via Activation Amplification and Attenuation

**arXiv ID:** 2606.27784 | [PDF](https://arxiv.org/pdf/2606.27784v1)

**作者:** Taïga Gonçalves `[一作]` (Tohoku University), Shinichiro Omachi `[通讯]` (Tohoku University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了 A3 模块，在网络中动态放大或衰减激活，以提高对抗鲁棒性。

**💡 创新点**

使用可学习的掩码和基于激活幅值的可微缩放因子实现同时放大与衰减，并将放大后的激活作为负样本构造排名和对比损失。

**🔧 技术方法**

采用 Gumbel-Softmax 掩码、对数缩放因子、排名损失、对比 logits 损失，以及标准对抗训练框架。

**📊 数据集**

CIFAR‑10、CIFAR‑100 和 Tiny ImageNet。

**📈 对比分析**

与 CAS、CIFS、FSR、FPCM、RiFT、FTA2C、EMFF 等主流插件防御在 AT、TRADES、MART 三种对抗训练方法下比较，A3 在 ensemble 与 AutoAttack 上取得最高或最接近最高的鲁棒准确率，且仅略微降低干净准确率。

**⚠️ 局限性**

对清洁图像的准确率略有下降，对超参数（如 λ_rank、λ_cl、τ_m）较为敏感，目前仅在通用图像分类数据集上验证，缺乏对更复杂任务或更大规模数据的评估。

---

## 193. Drifting in the Future: Stabilizing Path Following Drifting on High-Latency Vehicle Systems

**arXiv ID:** 2606.27914 | [PDF](https://arxiv.org/pdf/2606.27914v1)

**作者:** Frederik Werner `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在生产级奔驰‑AMG GT 63 S 车上实现了完整的自动漂移控制，涵盖漂移启动、维持和转移，并能沿预定圆形及 figure‑eight 轨迹实现精确路径跟踪。

**💡 创新点**

创新点包括：① 用状态预测器补偿发动机动力传动与方向盘的高延迟，保持控制相位一致；② 针对高延迟、差速耦合后轮设计了改进的漂移控制公式；③ 引入制动速度补偿器抑制漂移过程中的速度波动；④ 在无需独立后轮扭矩控制的情况下实现稳定漂移。

**🔧 技术方法**

采用两轨车辆动力学模型、Pacejka MF5 轮胎模型、非线性模型反演（Newton‑Raphson）、Yaw Stability Preserving Projection、2阶 Runge‑Kutta 前馈状态预测器（170 ms 预测）、制动速度控制器、PT2 方向盘延迟模型以及状态相关发动机扭矩模型。

**📊 数据集**

使用奔驰‑AMG GT 63 S 的真实测试数据：在混合湿干路面上完成圆形和 figure‑eight 漂移，采集 OXTS RT3000 INS/GNSS 数据，并对方向盘、发动机扭矩进行辨识；没有使用公开数据集，全部为实验收集。

**📈 对比分析**

与原始 Goh 等人基于 MARTY 平台的漂移控制器对比：仿真中最大侧滑率 0.64，最大横向误差 2.4 m；真实车中最大侧滑 0.06、横向误差 1.1 m、航向误差 0.1，速度上下限 12.6/14.4 m/s，明显优于基线的 1.4 m/0.11 rad/10.8/13.8 m/s。关键改进是预测器和速度控制器，消除了高延迟导致的振荡并实现了稳健漂移。

**⚠️ 局限性**

局限性：①预测器缺乏正式闭环稳定性证明；②仅在几何圆形和 figure‑eight 轨迹上验证，未测试任意曲率轨迹；③轮胎模型和摩擦估计仍是性能瓶颈；④未利用学习或在线自适应提升鲁棒性。

---

## 194. Phase Matters: Characterizing Heterogeneous Vision-Language Inference on a Mobile SoC

**arXiv ID:** 2606.27906 | [PDF](https://arxiv.org/pdf/2606.27906v1)

**作者:** Aryama V Murthy `[一作]` (International Institute of Information Technology, Hyderabad), Priyesh Shukla `[通讯]` (International Institute of Information Technology, Hyderabad)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在Qualcomm SM8750 SoC上对视觉-语言模型(VLM)进行阶段化加速评估，探讨了NPU与CPU在预填充、解码和视觉编码三阶段的性能差异，并给出可操作的异构管线配置与端到端推理策略。

**💡 创新点**

提出了阶段感知的部署图谱，首次量化了不同阶段的NPU加速比（预填充1.64×、解码1.18×）与热特性（平均低10.47°C、能耗降低2.52×），并开发了四步图重写方法将非QNN兼容编码器（如Phi-3.5-V）迁移至Hexagon NPU，获得22×速度提升。

**🔧 技术方法**

使用Qualcomm QNN SDK、LiteRT、W4A8量化、CPU/GPU/NPU比较实验；对FastVLM-0.5B及四大视觉编码器（ViT-B/16、Phi-3.5-V、NanoVLM、MobileNetV5）进行离线性能与热稳定性评估；通过多种缓存状态（S1、S2、S3）与冷启动耗时测量，展示异构管线在冷/热状态下的时延与能耗曲线。

**📊 数据集**

主要数据集为FastVLM-0.5B的内部训练集；评估时采用500张COCO val2017图像（CIDEr指标）和200张VQAv2开放式问题（准确率），以验证量化后NPU推理与CPU FP16的输出一致性。

**📈 对比分析**

通过与CPU单端和NPU单端基线对比，并在冷启动、热启动与异构配置下测量平均时延、吞吐量、能耗和温升；结果显示NPU在预填充阶段速度提升1.64×，解码阶段1.18×；全流程在热启动下NPU平均时延为2.08 s，能耗为CPU的1/2.5；Phi-3.5-V编码器在四步重写后实现22×加速，几乎匹配ViT-B/16的性能。

**⚠️ 局限性**

实验仅覆盖SM8750平台，未验证在其他NPU架构或不同SoC上的通用性；迁移方法依赖QNN编译器支持，可能不适用于所有自定义注意力或非标准算子；温度和能耗评估基于固定图像尺寸，动态分辨率或批处理规模下的表现尚未探测。

---

## 195. Co-Optimization of Analog Kolmogorov-Arnold Networks for Low-Power Function Approximation in Flexible Electronics

**arXiv ID:** 2606.27892 | [PDF](https://arxiv.org/pdf/2606.27892v1)

**作者:** Paula Carolina Lozano Duarte `[一作]` (Karlsruhe Institute of Technology), Sani Nassif `[通讯]` (Radyalis LLC)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种模拟Kolmogorov-Arnold网络（AKAN），通过硬件-软件协同优化方法，准确地近似复杂的多变量函数，以满足灵活电子设备中低功耗功能近似的需求。

**💡 创新点**

创新点在于结合电路级误差建模和系数剪枝策略，显著降低硬件成本的同时提高近似精度，展示了AKAN在灵活电子设备中的强大适应性和通用性。

**🔧 技术方法**

使用了硬件-软件协同设计方法，结合电路仿真、误差建模和参数剪枝，进行系统的准确性与成本权衡。

**📊 数据集**

验证在多个数据集上进行，包括PPG-DaLiA、ECG5000、家庭电力消耗和鸢尾花数据集，涵盖生物医学监测和传感场景。

**📈 对比分析**

与传统数字实现相比，AKAN通过系数剪枝实现了高达55%的面积和50%的功耗节省，且在多个数据集上平均减少近30%的硬件成本，同时保持或提高了近似精度。

**⚠️ 局限性**

限制在于当前方法未考虑局部器件不匹配的影响，未来工作需进行更全面的蒙特卡洛不匹配和产量分析，以验证AKAN在生产意图灵活电子环境中的可靠性。

---

## 196. SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV Perception, Collaboration, and Motion

**arXiv ID:** 2606.27876 | [PDF](https://arxiv.org/pdf/2606.27876v1)

**作者:** Haoyu Zhang `[一作]`, Liqiang Nie `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个真实低空 UAV 空间智能基准 SpatialUAV，包含 4,331 个实例，覆盖 14 种细粒度任务，并统一了视觉输入–问题–答案的评测框架。

**💡 创新点**

创新点在于：① 采用七种输入配置和九种答案格式，全面覆盖语义识别、空间关系、跨视角协作、地面协作与运动理解；② 设计了完整的数据构造与验证流水线，保证标注可靠且消除文本捷径；③ 通过多任务、多视角、时间序列的评测，揭示现有 VLM 在低空 UAV 场景下的瓶颈。

**🔧 技术方法**

使用了检测器辅助区域标注、深度监督、元数据规则、人工复核、盲过滤（使用 DeepSeek‑V4‑Pro 与 Qwen3.6‑27B），并在任务级别制定专属评价指标；同时对多模型（闭源、空间专用、开源）进行统一测试。

**📊 数据集**

数据来源包括 BEDI、AirCopBench、MAVREC、AirScape、University‑1652 等真实 UAV 图像、视频及元数据，经过整合与精细标注得到 SpatialUAV 数据集。

**📈 对比分析**

采用人类基准、随机猜测和 18 种 VLM 进行对比，评价指标覆盖精确匹配、部分匹配、F1、IOU 及语义相似度。结果显示最佳模型 GPT‑5.4 仅达 56.7% 平均分，开源 Qwen3.6‑27B 49.5%，与人类 89% 的差距明显，尤其在跨视角关联、几何推理与运动理解上表现最弱。

**⚠️ 局限性**

局限性包括：① 空间专用模型在低空 UAV 视角下迁移性差；② 高分辨率输入提升有限；③ 复杂答案格式（如 Bounding Box、Angle‑Distance Pair）导致得分低，主要反映任务本身难度而非输出格式问题；③ 现有模型缺乏针对低空特定的几何与跨视角推理能力。

---

## 197. FlexMoE: One-for-All Nested Intra-Expert Pruning for MoE Language Models

**arXiv ID:** 2606.27866 | [PDF](https://arxiv.org/pdf/2606.27866v1)

**作者:** Fan Mo `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FlexMoE，一种后训练压缩框架，能将预训练的 Mixture‑of‑Experts（MoE）大模型一次性转换为一系列嵌套的可部署子网络；通过专家 FFN 通道的重要性排序、离散保持动作学习、以及一次性中等预算的恢复微调，生成多预算共享权重子网络，并通过自定义 GPU kernel 支持实时预算切换。

**💡 创新点**

创新点包括：
1) 只需一次动作训练即可得到全预算范围内的嵌套子网络；
2) 在保持原始路由器与专家拓扑不变的前提下，利用通道前缀切片实现细粒度压缩；
3) 采用一次中等预算的 LoRA 微调即可恢复并共享权重，适用于所有子网络；
4) 结合算法与系统的 kernel 设计，使在线预算切换可行且性能提升。

**🔧 技术方法**

使用的技术：
- 重要性基排序（第一阶泰勒敏感度）对专家 FFN 通道进行重排；
- 离散动作学习（Straight‑Through Gumbel‑Softmax）为每个专家学习保持比例；
- 负载感知的成本正则化与熵正则化；
- 单点 LoRA 微调与权重合并；
- 自定义 GPU 前缀切片 + batched GEMM kernel 以降低在线切片开销。

**📊 数据集**

主要使用的数据集：
- Zyda‑2 作为校准集用于重要性排序和动作训练；
- 七个零样本推理基准（ARC‑Challenge、ARC‑Easy、HellaSwag、OpenBookQA、PIQA、WinoGrande、MathQA）用于评估子网络质量。

**📈 对比分析**

与 NAEE、MoE‑I²、MoE‑SVD、TD‑MoE 等基线进行对比。实验结果显示：
- 在 Qwen2‑57B‑A14B 上，50% 参数剪枝时保持 99.8% 基础性能，80% 剪枝时保持 92.9%；
- 通过中等预算微调，平均准确率与原模型差距 < 1%；
- 离线推理吞吐量提升 1.5–1.7 倍，在线切片在自定义 kernel 下提升 1.12–1.47 倍。

**⚠️ 局限性**

局限性：
- 主要聚焦于 MoE 结构搜索与静态参数剪枝，对非 MoE 模型的通用性有限；
- 动态预算调整仍以一次性中间点微调为主，未提供更细粒度的在线适配策略；
- 对极端硬件环境或多租户部署的系统级兼容性尚待进一步验证。

---

## 198. PPO-EAL: Exact Augmented Lagrangian Proximal Policy Optimization for Safe Robotic Control

**arXiv ID:** 2606.27861 | [PDF](https://arxiv.org/pdf/2606.27861v1)

**作者:** Jiatao Ding `[一作]` (University of Trento), Matteo Saveriano `[通讯]` (University of Trento)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种名为PPO-EAL的第一阶安全强化学习框架，将Exact Augmented Lagrangian与PPO结合，用于满足多重物理安全约束的机器人控制任务。

**💡 创新点**

创新点在于：①使用Exact Augmented Lagrangian代替传统大权重罚项，实现精确的约束满足；②引入动量调节的Lagrange乘子更新，显著降低约束振荡；③在保持剪切策略更新的同时实现高效收敛，理论上证明了exactness与收敛性。

**🔧 技术方法**

核心技术包括：PPO（剪切策略优化）、Exact Augmented Lagrangian（二次罚项）、动量调节的Lagrange乘子更新、Actor‑Critic网络、GPU加速模拟（IsaacLab）以及领域随机化。

**📊 数据集**

使用了多种模拟/真实数据集：Cart‑pole、双摆、Franka机械臂、ANYmal四足机器人，以及实际的齿轮装配硬件；所有实验均在IsaacLab环境或真实机器人上完成。

**📈 对比分析**

与基线方法（PPO、PPO‑L、P3O、APPO）在四大基准任务和齿轮装配任务上进行比较。结果显示PPO‑EAL‑m在满足所有安全阈值的同时，获得最高奖励；在零样本 sim‑to‑real 转移中成功率提升至80%（相较PPO的30%），并显著降低峰值接触力和扭矩。

**⚠️ 局限性**

局限性包括：仅使用前馈Actor‑Critic网络，对需要长期记忆的任务可能不足；约束仍为软约束，未提供在线安全屏障；在极端安全阈值或新环境下仍可能出现少量违规，需要进一步完善多策略鲁棒性和安全保险机制。

---

## 199. Video-MME-Logical: A Controlled Diagnostic Benchmark for Video Temporal-Logical Reasoning

**arXiv ID:** 2606.27828 | [PDF](https://arxiv.org/pdf/2606.27828v1)

**作者:** Hohin Kwan `[一作]` (Hong Kong University Of Science And Technology), Si Liu `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个可控的诊断性视频时间逻辑推理基准 Video‑MME‑Logical，包含 25 个细粒度任务，覆盖状态追踪、顺序计数、时序排序、动态空间推理和结构组合等五大逻辑操作，并为每类任务提供可调难度和可验证的中间状态。

**💡 创新点**

创新点在于：①通过构造五大时间逻辑操作的清晰分类，将视频推理拆解为可解释的子任务；②使用程序化生成视频实现视觉噪声剔除、难度可控与中间状态可验证；③提供包含中间推理轨迹的子集，使模型的推理过程可被逐步评估；④在大规模数据上进行 SFT 规模化实验，揭示数据量提升的边际收益与模型性能瓶颈。

**🔧 技术方法**

技术手段包括：程序化生成脚本（类似 CLEVR）实现视频合成与元数据记录；多模态大语言模型（Qwen3‑VL、Qwen‑Omni、InternVL、LLaVA、KimiVL、GPT‑5.4、Gemini‑3.1 Pro）进行零样本与 fine‑tune 评估；多种评估指标（最终答案准确率、中间状态准确率、过程可验证性）；采用多轮 fine‑tune（训练样本从 25K 至 500K）来分析规模化效果。

**📊 数据集**

使用的数据集为 Video‑MME‑Logical（总计 503,750 条视频，500K 训练集、3.75K 测试集），其中包含 25 个任务类别、3 个难度级别（易、中、难），以及 8 个中间状态诊断子任务。对比传统基准（TOMATO、TempCompass、ReXTime 等）时，该数据集在可控性、难度可调和中间状态可验证性上更具优势。

**📈 对比分析**

实验采用零样本评估与多种模型的 fine‑tune 对比，最终答案准确率从最弱的开源模型（约 0%）到最强的专有模型 Gemini‑3.1 Pro（≈28.6%）不等；人类参考准确率约 96%。中间状态评估显示专有模型在推理轨迹上明显优于开源模型（GPT‑5.4 17.4% vs Qwen3‑VL‑30B‑A3B‑Think 3.6%）。SFT 规模化实验表明，随着训练样本增至 375K，整体准确率提升至约 39%，但继续增至 500K 反而略有下降，显示增量收益递减。整体来看，模型与人类之间仍存在巨大差距，尤其在长时序和高复杂度任务中。

**⚠️ 局限性**

局限性包括：①数据来源为程序化合成，视觉风格与真实视频差距较大，可能影响在真实场景下的迁移能力；②实验仅在 8B 规模模型上进行，无法全面展示更大模型在时间逻辑推理上的潜力；③中间状态评估采用精确匹配，可能误判语义相同但表述不同的有效推理轨迹；④缺乏对视觉多样性与不确定性的进一步探讨。

---

## 200. Pepti-drift: Toxicity-Repulsive Drifting for Antigen-Conditioned Discrete Peptide Generation

**arXiv ID:** 2606.27824 | [PDF](https://arxiv.org/pdf/2606.27824v1)

**作者:** Takashi Fujiwara `[一作]` (SB Intuitions), Keisuke Ozawa `[通讯]` (SB Intuitions)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Pepti-drift，一种单步负向感知潜在漂移框架，用于在抗原条件下生成既能结合目标蛋白又能规避毒性的肽序列。

**💡 创新点**

创新点：①将吸引-排斥潜在漂移结合到单步漂移模块；②引入温启动负向排斥策略，稳定学习吸引与排斥共存的潜在向量场；③通过共享的 ESM-2 嵌入空间实现抗原特异性与毒性规避的几何控制；④实现了极快的端到端生成速度（16.2 倍比 PepMLM，1092 倍比 PepTune）。

**🔧 技术方法**

技术：预训练蛋白语言模型 ESM-2（作为共享潜在空间），投影层与 L2 归一化；单步漂移模块（Drifting Model）构建吸引-排斥向量场；非自回归 Transformer 解码器；温启动负向排斥训练策略；外部评估器 PeptiVerse、HemoPI2 等。

**📊 数据集**

数据集：训练集包含 20,547 对抗原‑肽结合对（来源 PPIKB、PepCCD），以及 12,709 条毒性相关肽（DRAMP、ToxiPred、Hemolytik）；在抗原层面使用 CD‑HIT 90% 阈值分割为训练/验证/测试三集，保证抗原独立性。

**📈 对比分析**

比较方法：与 PepMLM 和 PepTune 在相同的端到端生成任务（1,095 个抗原，64 条肽/抗原）进行对比。Pepti-drift 的生成速度最快（0.019 s/抗原），比 PepMLM 快 16.2 倍、比 PepTune 快 1,092 倍；生成肽有效率 100%，唯一性 98.1%，多样性最高，交叉抗原重用仅 0.27。外部评估显示其毒性和溶血风险明显降低，绑定亲和力虽然略低于 PepMLM，但高于 PepTune。

**⚠️ 局限性**

limitations：①仍需实验验证生成肽的真实结合活性与安全性；②模型依赖 ESM-2 嵌入空间，若该空间不足以充分分离毒性与结合信号，性能受限；③对长肽的非亲和性（non‑fouling）表现下降；④未评估其它药物开发指标（稳定性、可合成性等）。

---

## 201. SpikeVLA: Vision-Language-Action Models with Spiking Neural Networks

**arXiv ID:** 2606.27807 | [PDF](https://arxiv.org/pdf/2606.27807v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 202. Triadic Werewolf: A Jester Role for Multi-Hop Theory of Mind in LLMs

**arXiv ID:** 2606.27909 | [PDF](https://arxiv.org/pdf/2606.27909v1)

**作者:** Avni Mittal `[一作]` `[通讯]`, Avni Mittal

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在经典狼人杀游戏中加入Jester角色，构建三方激励结构，生成60场模拟游戏并对GPT‑4.1、DeepSeek‑V3.1、Llama‑3.3‑70B进行实验，评估自学习循环对Jester表现的影响。

**💡 创新点**

通过将一维疑惑信号转化为三方激励预测，揭示LLM在多跳Theory‑of‑Mind任务中的不足，并提出最小化自学习回环以检验模型的推理与学习能力。

**🔧 技术方法**

使用bidding‑based debate、实时怀疑评分、8类欺骗分类、自学习经验文件、系统提示与RL动作选择、统计分析（Wilson CI）等技术。

**📊 数据集**

采用生成的60场游戏日志（包含发言、投票、怀疑评分及自我欺骗标签）以及Jester自学习经验文件。

**📈 对比分析**

通过比较OFF/ON自学习条件下的胜率并使用Wilson置信区间评估，结果显示GPT‑4.1基线胜率已高，DeepSeek与Llama随自学习提升显著，Jester击败狼人，狼人胜率低于20%，且日常投票自毁现象普遍。

**⚠️ 局限性**

样本量有限（每细胞10场）、仅英文、未检验跨模型学习效果、缺少人类真标注、未实现更高效的记忆机制，可能导致结论不够稳健。

---

## 203. Swarm sign language: motion-based communication between drones

**arXiv ID:** 2606.27883 | [PDF](https://arxiv.org/pdf/2606.27883v1)

**作者:** Thomas Rey `[一作]` (ONERA, Université Paris-Saclay), Antoine Manzanera `[通讯]` (ENSTA, Institut Polytechnique de Paris)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于三维飞行轨迹的无人机群视觉通信协议，利用平面几何原语（如正方形、圆形、螺旋等）及其尺度、平面法向量对消息进行编码，并在接收端通过姿态估计与3DTrajDecoder实现多任务解码（分类、分割、尺寸回归、法向回归）

**💡 创新点**

将动作轨迹视为可调节的符号字母表，创新性地将轨迹的尺寸和法向量作为额外信息通道；采用基于Transformer的多任务架构，兼顾全局和局部信息；提供可在线、可配置的程序化生成管线，显著缓解仿真到现实的领域差距

**🔧 技术方法**

程序化轨迹生成（最小-摆动优化、噪声注入）、RGB姿态估计（6D估计网络）、多任务Vision Transformer（3DTrajDecoder）以及多头自注意力、位置编码、[CLS] token、上采样卷积等技术

**📊 数据集**

自研的无公开同类数据集，采用程序化生成的三维轨迹集合（包含9类符号及非通讯运动），以及在真实实验中使用MoCap记录的88个通讯段与50个非通讯段作为验证数据集

**📈 对比分析**

与传统LSTM、ResNet、ViT等基线模型进行对比；在测试集上多任务Transformer实现F1分类≥91%、分割≥94%、尺寸MAE≈0.06m、法向平均误差≈13°；在真实数据上亦保持较高准确率（分类>95%、分割>87%、尺寸MAE≈0.08m、法向误差≈4–5°）

**⚠️ 局限性**

对时间窗口偏移敏感（±6s时性能显著下降）；对轨迹尺寸和法向量的估计受噪声影响；目前仅支持离线姿态估计；实验规模受限于安全区域与控制器误差，且未验证在动态背景或多无人机环境下的鲁棒性

---

## 204. GNBAN: Graph Neural Basis Attention Networks for Long-Horizon Forecasting over Large Entity Sets

**arXiv ID:** 2606.27863 | [PDF](https://arxiv.org/pdf/2606.27863v1)

**作者:** Janak M. Patel `[一作]` (Quantiphi), Dagnachew Birru `[通讯]` (Quantiphi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种统一的图神经网络框架GNBAN，用于大规模零售需求预测，并通过可解释的基函数分解实现长期预测。

**💡 创新点**

创新点在于：① 将零售数据直接映射为异构图；② 引入基函数注意力头，使每个基函数拥有独立的查询，能够从历史邻域中专门检索趋势、季节性和残差信息；③ 通过可解释的趋势/季节/通用分解，避免传统黑箱预测头，提供内在可解释性。

**🔧 技术方法**

采用异构GraphSAGE作为图编码器，基函数注意力机制（per‑basis attention）以及多基函数（多项式、正弦、可学习通用基）实现分解；训练使用Huber损失和Adam优化。

**📊 数据集**

在M5 Walmart（≈30k SKU‑store序列，28天预测）和Favorita Grocery Sales（≈50k SKU‑store序列，16天预测）两个大规模零售基准上进行评估。

**📈 对比分析**

与同样图结构下的基线（MLP预测头）比较，GNBAN在两数据集的体积加权WRMSSE上分别提升约4–5%（Favorita）和3.6%（M5），在不加权指标上保持或略低，表明专用基函数注意力显著提升高价值序列预测精度。

**⚠️ 局限性**

局限性包括：① 仅在零售场景验证，缺乏跨领域泛化；② 对极度稀疏/低销量序列的性能仍不理想；③ 需要对图结构和基函数数量进行手工调优，模型复杂度较高。

---

## 205. ScaLe-INR: Scale and Learn Implicit Neural Representations

**arXiv ID:** 2606.27862 | [PDF](https://arxiv.org/pdf/2606.27862v1)

**作者:** Buwaneka Epakanda `[一作]`, Parakrama Ekanayake `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种多分支隐式神经表示（ScaLe‑INR）框架，通过在不同方向上对坐标进行缩放来匹配信号频谱与网络可学习带宽，并引入置信度融合与边缘引导正则化，实现高频细节的专门化学习，提升连续信号的重建与逆问题性能。

**💡 创新点**

创新点包括：1) 将傅里叶逆缩放定理应用于坐标缩放，构造可控的频谱匹配；2) 设计多分支网络对不同方向与交叉高频分量进行专门建模；3) 通过置信度软化融合实现像素级分支权重自适应；4) 引入方向性边缘引导正则化，强制高频分支仅关注边缘信息，消除跨频干扰。

**🔧 技术方法**

使用的技术有：多分支全连接网络（MLP）+ 频域缩放 + 置信度软化融合 + 边缘引导正则化 + 对数似然损失 + 训练调度（edge loss 逐步启用）。

**📊 数据集**

数据集涵盖多任务：图像重建（Kodak），图像去噪（DIV2K Parrot），图像超分（DIV2K），3D占用率重建（Lucy），音频重建（Bach 小提琴协奏曲）。

**📈 对比分析**

与现有 SOTA 方法（COSMO‑INR、COSMO‑RC、FINER、INCODE、SIREN 等）进行对比，实验表明：图像重建平均 PSNR 46.4 dB，比最优对手高 5.16 dB；去噪 PSNR 30.90 dB，提升 0.65 dB；6× 超分 SSIM 0.85，2× 提升 0.40 dB；3D 占用率 IoU 0.999，超越 0.004；音频 PSNR 50.02 dB，领先 0.92 dB。

**⚠️ 局限性**

局限性包括：1) 需要为每个任务手工选择合适的缩放因子；2) 多分支架构增加模型参数和训练开销；3) 对过度缩放敏感，可能导致频谱失真；4) 目前仅验证在静态场景/单维音频，对动态图或高维数据的推广尚未充分验证。

---

## 206. WattLayer: Get Layers Right to Estimate Inference Energy of Neural Networks

**arXiv ID:** 2606.27841 | [PDF](https://arxiv.org/pdf/2606.27841v1)

**作者:** Adrien Sardi `[一作]` (Bell Labs, Nokia Networks France), Joanna Moulierac `[通讯]` (Université Côte d’Azur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种任务无关的层级能耗估计模型，利用层级能耗分解对AI模型进行推理能耗预测。

**💡 创新点**

创新点在于层级分解的通用性，校正因子自适应，可在不重新训练的情况下跨任务和跨模型泛化。

**🔧 技术方法**

采用Python脚本提取PyTorch执行图中的层级信息，收集能耗数据后使用线性/多线性/对数线性回归为每类层训练能耗模型，并用校正因子合并。

**📊 数据集**

构建了包含100k+层级能耗样本的数据库，覆盖295个网络，3个任务（图像、文本、音频），在3种GPU（RTX6000、TITAN X、H100/A100）上收集。

**📈 对比分析**

与单指标MAC回归和HJ对数线性模型对比，使用MAPE/MedAPE等指标，模型在图像、文本、音频上分别实现MAPE≈40%、46%和27%，显著优于SOTA，LLM零样本推断MAPE≤30%。

**⚠️ 局限性**

局限在于需针对每种硬件单独训练，未考虑量化、稀疏等加速技术，且校正因子在不同平台可能需要重新校准。

---

## 207. LXD-SLAM: LiDAR+X Dense SLAM with $\sum_{i=0}^{5}C_5^i$ Configurable Sensor Combinations

**arXiv ID:** 2606.27811 | [PDF](https://arxiv.org/pdf/2606.27811v1)

**作者:** Zhong Wang `[一作]` (Tongji University), Shengjie Zhao `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出LXD‑SLAM框架，支持LiDAR、Camera、IMU、Wheel Encoder、GNSS五种传感器的32种组合，实现全局一致的稠密地图与多模态融合SLAM；

**💡 创新点**

核心创新在于：星型模块化拓扑实现任意组合无重构；统一的迭代误差状态卡尔曼滤波器与多层Gaussian Process子网格的点‑网格约束；基于GP的Extended Scan Context（ESC）描述子与双向PnP视觉闭环；光线‑网格深度恢复与逆深度滤波；以及实时的稠密建图；

**🔧 技术方法**

采用Iterative Error‑State Kalman Filter（IESKF）+层级预测、点‑网格误差 + 视觉重投影；光线追踪与逆深度滤波；多层GP子网格建图；ESC全局描述子；双向PnP闭环；GTSAM因子图优化；光学流与特征管理；

**📊 数据集**

在NTU‑VIRAL、FusionPortableV2、KITTI公开数据集以及自行收集的校园室内外多传感器数据上进行实验；

**📈 对比分析**

与LOAM、MLOAM、Fast‑LIO2、R^3LIVE、Fast‑LIVO2、LIO‑SAM等多种SOTA系统在ATE、RMSE等指标下进行对比，LXD‑SLAM在大多数配置下均达到或超过SOTA，尤其在视觉辅助、双LiDAR、IMU+Wheel组合中精度提升显著，并在实时稠密地图质量上优于SLAMesh；

**⚠️ 局限性**

局限性包括：对低纹理/低光场景视觉深度恢复仍受限；GP子网格构造在大规模环境下仍有计算开销；系统需要较高CPU/GPU资源；对动态物体的建模尚未完善；在长时间无闭环情况下仍难以完全消除漂移。

---

## 208. Grounded Iterative Language Planning: How Parameterized World Models Reduce Hallucination Propagation in LLM Agents

**arXiv ID:** 2606.27806 | [PDF](https://arxiv.org/pdf/2606.27806v1)

**作者:** Xinyuan Song `[一作]` (Emory University), Zekun Cai `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

比较基于LLM的Agent世界模型与参数化预测世界模型，并提出GILP混合框架以降低长周期规划中的hallucination；

**💡 创新点**

将小型参数化图神经网络的可测转移预测与LLM推理结合，利用一致性门和重提示实现自动纠正，从而显著提升成功率并抑制错误传播；

**🔧 技术方法**

采用图神经网络骨干（MLP、GCN、MPNN等）、Jaccard一致性门、针对性重提示、LLM API（GPT‑4o‑mini、Claude‑3‑Haiku、Gemini‑1.5‑Flash、Llama‑3‑8B）以及行为模拟器校准；

**📊 数据集**

使用四个图结构规划基准（TaskGraph、ToolChain、ResourceAlloc、RepairFlow）以及FB15k‑237 关系图遍历任务；

**📈 对比分析**

在模拟器与真实GPT‑4o‑mini评估中，GILP将成功率从0.668提升至0.838，hallucinated‑state率从0.176降至0.035，Token成本略升高但仍保持在可接受范围；

**⚠️ 局限性**

结果主要基于校准模拟器，骨干模型需要可观测的离散转移，且对非图结构或连续状态的泛化尚未验证。

---

## 209. The quantum instrument monad

**arXiv ID:** 2606.27805 | [PDF](https://arxiv.org/pdf/2606.27805v1)

**作者:** Tobias Fritz `[一作]` `[通讯]` (University of Innsbruck), Tobias Fritz (University of Innsbruck)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了量子仪表单子（quantum instrument monad），用来刻画在固定量子系统上执行任意仪表并返回经典结果的计算效应；提供了两个实现版本——有限版本（有限离散结果）和测度论版本（可测结果空间）并证明其为强单子。

**💡 创新点**

创新点在于将经典状态单子推广为非交换的量子仪表单子，并在测度论层面引入新的量子操作积分（对向量测度的积分），从而实现了对一般可测结果空间的量子仪表的单子化；同时给出了两种实现并与已有的线性状态单子、参数化单子等概念的关联。

**🔧 技术方法**

技术手段主要包括范德尔马恩代数与其预对偶的量子操作定义、仪表的可加性与通道性、公测度论与Radon–Nikodym性质的结合、向量测度积分以及对单子结构（单位、乘法、强度）的严格证明。

**📊 数据集**

本工作完全是理论性的，没有使用实验或公开数据集。

**📈 对比分析**

方法的有效性通过证明单子公理、强度兼容性以及与继续单子（continuation monad）的嵌入来检验；在测度论版本中还通过对量子操作积分的可测性和可积性证明了乘法的可测性，展示了理论一致性。

**⚠️ 局限性**

局限性包括：测度论版本仅适用于类型 I、可分预对偶的冯·诺伊曼代数；对Radon–Nikodym性质的依赖排除了如L∞([0,1])和III 型代数等离散或扩散情形；且对一般可测空间的完备性与可积性仍有未解决的问题。

---

## 210. Exploring and Exploiting Synchrony Limitations of Time-Triggered Network-Agnostic Guardians

**arXiv ID:** 2606.27819 | [PDF](https://arxiv.org/pdf/2606.27819v1)

**作者:** Shreya Vithal Kulhalli `[一作]` (RPTU Kaiserslautern-Landau), Gerhard Fohler `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究网络无关守护者（network‑agnostic guardian）的时钟同步限制，并基于该限制设计并验证一种在时间触发网络中可被隐藏的定时攻击（target slot attack），通过 OMNeT++/FiCo4OMNeT 的 FlexRay 仿真展示攻击对关键帧的破坏效果。

**💡 创新点**

提出网络无关守护者时钟同步的理论下界，并基于该下界计算出可被隐藏注入的最大延迟，随后构造一种利用该延迟逐步使守护者失同步并与目标节点时隙重叠的定时攻击，证明其理论可行性和 100% 的实验证明成功率。

**🔧 技术方法**

使用时钟同步理论分析、最大隐蔽注入延迟推导、时域攻击模型、OMNeT++ 及 FiCo4OMNeT FlexRay 模型进行仿真，以及对攻击成功率和检测率的统计分析。

**📊 数据集**

采用八节点 FlexRay 网络的标准参数（pdMicrotick=12.5 ns、gdMacrotick=1 µs、gdStaticSlot=4 MT、gdNIT=2 MT 等），不使用公开数据集，仅在仿真环境中生成实验数据。

**📈 对比分析**

通过比较不同同步精度（1 MT、0.75 MT、0.5 MT）和不同注入延迟下的攻击，实验结果显示：在满足理论精度下界时，攻击始终未被守护者检测，且对所有目标时隙的攻击均以 100% 的成功率破坏关键帧；性能指标主要为攻击成功率与检测率。

**⚠️ 局限性**

局限性：攻击前提假设攻击者可完全控制节点并知晓全局时序；仅针对网络无关守护者，未考虑更强的安全防御（如节点再生、守护者间通信）以及不同协议或拓扑下的适用性；实验仅在模拟环境中进行，未在真实硬件上验证。

---

## 211. There and Back Again: A Flexible-Frame Transformer for Multi-Exposure Fusion

**arXiv ID:** 2606.27905 | [PDF](https://arxiv.org/pdf/2606.27905v1)

**作者:** Lishen Qu `[一作]` (Nankai International Advanced Research Institute), Jufeng Yang `[通讯]` (Nankai International Advanced Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了FreeMEF，一种能够处理任意数量曝光帧的多曝光融合模型，并在保持低计算成本的同时提升HDR图像质量。

**💡 创新点**

创新点包括：①递归状态空间模块(RSSM)实现可变长度帧融合；②全局特征引导块(GFGB)，其中极值感知混合注意力(EAHA)解决了类似性悖论；③仿射注入前馈网络(AFFN)动态调节亮度与对比度。

**🔧 技术方法**

使用了递归状态空间模型(ASE)、变形卷积对齐、混合注意力、自注意力、仿射投影、Transformer U‑shaped 架构以及可调温度的注意力机制。

**📊 数据集**

在Kalantari、Real‑HDRV和SICE三个数据集上进行训练与评估，其中SICE用于测试不同帧数的泛化能力。

**📈 对比分析**

与8种最先进方法（包括MEFLUT、SAFNet、SCTNet、HDR‑Transformer、AFUNet、Restormer、MambaIRv2、ASTv2）进行对比，FreeMEF在PSNR/SSIM/LPIPS上均优于第二名，且参数量与FLOPs更低，跨数据集和多帧数均保持领先。

**⚠️ 局限性**

局限性在于当曝光间隔过大时，融合结果可能出现轻微颜色偏移，尤其是基帧过暗或长曝光过度饱和的场景。

---

## 212. OrthoTryOn: Geometric Orthogonalization for Conflict-Free Unified Fashion Generation

**arXiv ID:** 2606.27880 | [PDF](https://arxiv.org/pdf/2606.27880v1)

**作者:** Zhaotong Yang `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了OrthoTryOn框架，能够在单一低秩适配模块中统一实现虚拟试穿、服装重建和姿态迁移三大时尚生成任务。

**💡 创新点**

创新点在于：①通过任务特定的正交子空间投影（OSP）将共享LoRA参数映射到正交坐标系，抑制梯度冲突；②在推理阶段使用基于Fisher信息的负向引导（FNG）消除残余语义干扰，且不引入额外参数。

**🔧 技术方法**

采用LoRA+正交旋转、Fisher信息估计、分类器无指导（CFG）等技术，并将其嵌入现有扩散模型（如LongCat-Image-Edit、Stable Diffusion 2.1、FLUX.2-klein）。

**📊 数据集**

使用VITON-HD和DeepFashion两个公开数据集，分别用于虚拟试穿/服装重建和姿态迁移任务的联合训练。

**📈 对比分析**

在多种统一基线（AnyDoor、Any2AnyTryon、LongCat-Image-Edit、FLUX.2-klein）以及对应单任务专家模型上进行对比。OrthoTryOn在LPIPS、FID、SSIM、CLIP-I等指标上均优于或接近单任务专家，表现出显著的性能提升。

**⚠️ 局限性**

在LoRA秩极低或任务数过多时，残余梯度耦合仍可能显著，FNG虽可缓解但无法完全恢复训练阶段信息，需适当提高LoRA秩以获得更好效果。

---

## 213. LocalNav: Distilling Frontier VLMs and Embodied RL for On-Device Object Goal Navigation

**arXiv ID:** 2606.27871 | [PDF](https://arxiv.org/pdf/2606.27871v1)

**作者:** Nicolas Baumann `[一作]` (ETH Zurich), Michele Magno `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可在资源受限机器人上本地部署的 VLM‑based 开放词汇目标导航框架 LocalNav，利用知识蒸馏和强化学习优化实现高效推理。

**💡 创新点**

创新点包括：① 用仅约 500 条教师轨迹完成从大型云模型到 4B 轻量级模型的蒸馏；② 设计 E‑RLVR（强化学习 + 输出长度惩罚）在闭环环境中压缩生成序列长度；③ 结合量化实现 82.8% 的整体推理延迟下降。

**🔧 技术方法**

技术手段：VLM 蒸馏、SFT 微调、E‑RLVR（带 token 长度奖励的强化学习）、量化（int8/float16）、Jetson Orin AGX 上的高效推理后端。

**📊 数据集**

数据集：HM3D OVON benchmark 用于评估；教师轨迹数据集（Gemini 3.1 Pro、GPT‑5.4、Claude Sonnet 4.6）在 HuggingFace 上公开，约 500 条样本。

**📈 对比分析**

与 10+ 现有 OGN 方法对比，云模型 Claude Sonnet 4.6 在 HM3D OVON 上取得 39.7% sr、19.7% SPL；蒸馏后的 4B 学生模型达 34.5% sr、17.2% SPL，差距仅 5.2 %；在 Jetson Orin AGX 上结合 E‑RLVR 与量化后，平均运行时从 305 s 降至 52.5 s，延迟减少 82.8%。

**⚠️ 局限性**

局限性：基于场景图的 OGN 方案在时空动态语义（如“椅子上有人”）上缺乏即时表示，需多次 VLM 推理，可能遗漏中途信息；对动态场景的感知和更新仍需进一步提升。

---

## 214. From Bootstrapping to Sequence Modeling: A Unified Generative Framework for Personalized Landing-Page Modeling

**arXiv ID:** 2606.27865 | [PDF](https://arxiv.org/pdf/2606.27865v1)

**作者:** Fan Li `[一作]` (Duke University), Kaiqiao Zhan `[通讯]` (Kuaishou Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了GLAN框架，使用Decision Transformer结合L-RTG和HRM两模块实现个性化着陆页分配，并在快手平台进行线上A/B实验。

**💡 创新点**

将序列建模与强化学习结合，使用L-RTG预测跨日Return-to-Go提供全局指导，HRM细粒度拆分会话奖励以解决信用分配模糊。

**🔧 技术方法**

基于Decision Transformer的自回归序列模型、周期性注意力与Transformer编码器、门控多专家混合网络、Huber与焦点交叉熵损失。

**📊 数据集**

使用快手两个月匿名用户日历和会话级日志，包含用户特征、统计特征和30天历史行为序列。

**📈 对比分析**

对比现行KLAN基于CQL的上线系统，在56天A/B实验中DAU +0.158%、LT +0.108%、APP使用时长 +0.369%等指标显著提升，页跳出率下降15.8%。

**⚠️ 局限性**

模型依赖准确的RTG预测和HRM信号，跨日周期性变化仍可能导致目标误估，且在极少量用户数据下性能可能受限。

---

## 215. A Unified Framework for Vision Transformers Equivariant to Discrete Subgroups of $\mathrm{O}(2)$

**arXiv ID:** 2606.27864 | [PDF](https://arxiv.org/pdf/2606.27864v1)

**作者:** Tīkun Ông `[一作]` (Independent Researcher), Georg Bökman `[通讯]` (University of Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种能对任意离散二维子群O(2)实现等变的视觉Transformer框架，并在此基础上构造了等变线性层、非线性层、注意力机制以及位置编码，构成完整的等变ViT模型；

**💡 创新点**

创新点在于将等变ViT推广到任意离散子群，提供统一的表示理论构造；提出了一种基于Fourier变换的最一般等变非线性方法，并证明单头等变自注意力能够实现所有可表示的标准自注意力；

**🔧 技术方法**

利用群表示理论构造等变线性映射、Fourier基底实现非线性、等变自注意力、类标记不变化化、以及基于G‑patchified网格的patch embedding等技术；

**📊 数据集**

在PatternNet航空图像数据集上进行实验，数据量分为不同比例（10%、40%、100%）进行低样本与完整样本训练；

**📈 对比分析**

通过与非等变ViT及不同注意力方式（irrep‑wise 与 coupled）对比，发现等变模型在低样本条件下显著提升准确率，但在完整数据下几乎无差异；irrep‑wise注意力在部分设置下略优；

**⚠️ 局限性**

实验规模受限于小模型（≈0.5M参数）和有限的超参搜索，尚未验证在大规模数据/大模型上的可扩展性；缺乏系统的特征空间V选择方法，且对更一般子群的实现仍需进一步研究。

---

## 216. Spectral clustering of time-evolving networks using spatio-temporal random walks

**arXiv ID:** 2606.27850 | [PDF](https://arxiv.org/pdf/2606.27850v1)

**作者:** Filip Blašković `[一作]` (Zuse Institute Berlin), Nataša Djurdjevac Conrad `[通讯]` (Zuse Institute Berlin)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文基于多视角主成分分析与转移算子理论，构建了一个可捕捉时变社区的空间-时间随机游走框架，并通过谱聚类实现社区检测。

**💡 创新点**

创新点在于提出了通用的快照耦合方案，将多视角 CCA 形式化为时空网络的自反随机游走的谱问题，并阐明了时空谱分解能区分时间与空间效应，进而实现模型降维与误差控制。

**🔧 技术方法**

核心技术包括多视角 CCA、Koopman 与 Perron‑Frobenius 算子、时空转移矩阵构造、谱聚类、Galerkin 投影的模型降阶以及 Nyström 方法近似特征向量。

**📊 数据集**

实验使用合成时间演化的块模型网络（包含社区分裂合并、周期性重现）以及 Hegselmann–Krause 观点动力学产生的加权临时网络；未使用真实大规模数据集。

**📈 对比分析**

通过与完整模型对比、与仅耦合相邻快照的基线方法对比，证明降阶模型在保留主要空间特征和社区分辨率的同时，显著降低了计算时间；在周期性社区实验中，循环耦合方案能成功捕获社区重现。

**⚠️ 局限性**

局限性包括：需要手动设定快照耦合权重与时间尺度；对节点集合不变的假设，无法直接处理加入/删除节点的网络；虽然降阶后速度提升显著，但在大规模时间步长和节点数极大时仍存在显著内存与计算瓶颈；空间特征向量的筛选仍依赖经验启发式。

---

## 217. Learning Complementary Action Modeling from Automotive Maintenance Instructions

**arXiv ID:** 2606.27808 | [PDF](https://arxiv.org/pdf/2606.27808v1)

**作者:** Jiaqi Wu `[一作]` (Eindhoven University of Technology), Sander Stuijk `[通讯]` (Eindhoven University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Complementary Action Modeling（CAM）任务，旨在识别或生成汽车维修指令的程序性对应指令，只改变动作短语并保持其余上下文不变。

**💡 创新点**

将微小词汇变化映射为程序互补关系并正式定义CAM；设计候选匹配和受控生成两种方法；提出跨车款泛化评估与规则/人工对比的双重验证框架。

**🔧 技术方法**

使用多语言与德语预训练Transformer做双编码器对比学习（InfoNCE）进行候选匹配；使用mBART、mT5、Flan‑T5等Seq2Seq模型并加入嵌入空间对比正则化进行受控生成。

**📊 数据集**

基于德国OEM汽车维修手册构建约1459对互补指令的手工标注数据集，分为规则匹配（rule‑resolved）和非规则匹配（rule‑unresolved）两类，并按流程桶进行划分。

**📈 对比分析**

候选匹配中，句子变换器在R@1≈0.74、MRR≈0.83上最佳；受控生成中mBART‑large‑50获得BLEU≈63、ROUGE‑L≈0.80；人类评估显示95%生成实例语义互补，错误多为动作短语未正确替换；跨车款推断准确率约64.5%。

**⚠️ 局限性**

局限性包括：数据仅来自单一德语OEM，泛化性受限；对齐依赖规则+人工校验，可能带来偏倚；对比学习侧重局部流程，忽略跨文档关联；生成模型高度依赖大规模预训练，低容量模型表现不佳；跨车款评估仅手工验证，缺乏完整标注基准。

---

## 218. Accelerating Hierarchical Sparse Predictive Coding with Hybrid Amortized Inference

**arXiv ID:** 2606.27802 | [PDF](https://arxiv.org/pdf/2606.27802v1)

**作者:** Kazuhisa Fujita `[一作]` `[通讯]` (Komatsu University), Kazuhisa Fujita (Komatsu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在层级稀疏预测编码模型上结合拟合推理与迭代修正的混合推理策略，并在固定的稀疏生成目标下，对不同推理预算进行系统评估。

**💡 创新点**

创新点在于提出一种浅层 LISTA 风格的快速初始化与少量 ISTA/ MFISTA 迭代修正相结合的 Hybrid 推理框架，并通过细粒度预算分配实验验证了该方案在质量‑延迟折衷上的优势。

**🔧 技术方法**

使用技术包括层级稀疏预测编码模型、ISTA/MFISTA 迭代求解、基于 LISTA 的结构化可分配推理器、混合推理、算法展开、共享稀疏能量、字典学习与 Adam 优化、软阈值处理。

**📊 数据集**

实验数据集包括 MNIST、Fashion‑MNIST、CIFAR‑10 灰度图像以及 BSDS500 补丁。

**📈 对比分析**

通过比较 ISTA、MFISTA、LISTA 与 Hybrid 四种推理方式，在测试集上评估稀疏能量损失、重建误差和样本延迟，结果显示 Hybrid 在相同或更低延迟下显著优于单纯 LISTA，并在可接受的计算开销内逼近完整迭代求解的性能。

**⚠️ 局限性**

局限性在于学习规则缺乏生物学可解释性、仅针对静态图像且未考虑卷积结构、对深层次结构与不同稀疏度敏感、并未提供完整的收敛稳定性保证。

---

## 219. Position Bias Correction is Insufficient for One-Pass Attention Sorting

**arXiv ID:** 2606.27793 | [PDF](https://arxiv.org/pdf/2606.27793v1)

**作者:** Qiong Tang `[一作]` (Analemma), Yunfan Shao `[通讯]` (Analemma)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了一种 Debiased One-Pass Attention Sorting，利用估计的位置信息曲线校正原始注意力得分，从而在单次排序后完成长上下文文档重排。

**💡 创新点**

创新点在于从低注意力文档中构造每个提示的位置信息曲线，并用加法或除法方式去偏校正注意力得分，避免了原方法所需的多轮排序循环。

**🔧 技术方法**

采用的技术包括：注意力得分聚合、位置偏差曲线估计（截取前 α% 文档，分 B 组统计中位/均值并线性插值）、加法/除法去偏校正、单次排序加生成过程。

**📊 数据集**

使用的数据集为 SynthWiki@28K，该数据集包含约 28K 令牌的合成长上下文抽取式 QA 示例。

**📈 对比分析**

与无排序、k=1 未校正 Attention Sorting、k=5 迭代 Attention Sorting 进行对比；在 LLaMA‑2‑7B‑32K‑Instruct 上准确率保持 94.83% 无提升；在 YaRN‑Llama‑2‑7b‑64k 上提升 8.67pp 到 55.83%，仍落后 k=5 的 70.67%，显示去偏校正仅部分弥补了迭代排序的优势。

**⚠️ 局限性**

局限性包括：仅在两种模型和单一 Benchmark 上评估；校正参数在相同数据集上调优，可能导致过拟合；结果高度模型依赖，未验证在更广泛任务和模型上的通用性。

---

## 220. A Comparison of Fusion Techniques for Multi-Modal Human Activity Recognition on the HARMES Dataset

**arXiv ID:** 2606.27886 | [PDF](https://arxiv.org/pdf/2606.27886v1)

**作者:** Ahmed Mohamady `[一作]` (University of Siegen), Kristof Van Laerhoven `[通讯]` (University of Siegen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估并比较了七种多模态融合技术在HARMES数据集上的表现。

**💡 创新点**

系统地在统一实验条件下对七种主流融合方法进行头对头比较，提供首个公开的多模态HAR融合方法排名，并分析手性对性能的影响。

**🔧 技术方法**

采用三模态（IMU、音频、湿度）分别用TinyHAR、AST、TSMixer编码器，构建七种融合架构（Late Fusion、GMF、LMF、CMA、MBT、CLS-Token Transformer、Decision Fusion），使用10秒窗口、3折组交叉验证与LOPO评估。

**📊 数据集**

使用HARMES公开多模态HAR数据集，包含20位参与者、61小时同步IMU、音频、湿度数据，15项日常活动+Null。

**📈 对比分析**

在3折CV下所有方法均优于最佳单模态（AST 0.734），GMF最高宏F1 0.827，LOPO下GMF 0.819，显著高于HARMES基线0.760；简易方法（GMF、Late Fusion）最强，复杂注意力/张量方法略逊。

**⚠️ 局限性**

数据量有限（仅20人、3位左撇子），仅评估三模态且仅在特定编码器架构下；湿度贡献低；未考虑早期融合与其他数据集；结果对更大规模或不同任务可能变化。

---

## 221. Combining Axiomatic Models for Refinement Proofs

**arXiv ID:** 2606.27916 | [PDF](https://arxiv.org/pdf/2606.27916v1)

**作者:** Suha Orhun Mutluergil `[一作]` (Sabanci University), Alperen Dogan `[通讯]` (Sabanci University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一套将实现程序与抽象规范之间的前向与后向仿真关系与四种程序逻辑（Hoare、Incorrectness、Lisbon、Necessary Preconditions）相结合的框架，并利用该框架在一个左‑右（LR）并发计数器的实现与其原子顺序规范之间传递安全性质，避免直接对并发实现做复杂的线性化分析。

**💡 创新点**

创新点主要在于：①将传统的仿真关系与四类程序逻辑统一表述；②给出前向仿真与后向仿真在各类逻辑下可被归约为单一三元组的判定；③构造了一个中间的无确定并发计数器，将单条实现与规范之间不存在的直接仿真分解为前向+后向仿真链，从而实现了安全性质的逐层传递；④提出了在逻辑层面验证仿真的方法，并通过实例展示其可操作性。

**🔧 技术方法**

使用的技术包括：自动机理论（状态、动作、转移、trace）；前向/后向仿真定义与证明；四种程序逻辑的三元组验证（Hoare、Incorrectness、Lisbon、Necessary Preconditions）；逻辑推理中的图像与逆像概念；以及对中间计数器的辅助状态（历史、待决、活跃读取）进行的规范化。

**📊 数据集**

该工作不依赖外部数据集，而是通过形式化的计数器模型和自动机构造进行实验验证；所有示例均为手工构造的抽象计数器模型。

**📈 对比分析**

与传统的线性化点证明相比，本文方法不需要显式推导每个操作的线性化时刻，而是通过仿真链在逻辑层面完成安全性质的迁移；由于仿真关系的验证被归约为单一三元组的有效性检查，理论上可在现有的程序逻辑工具（如Coq、Lean、HOL等）中实现；实验层面未给出具体性能数值，重点在于证明流程的可行性与可扩展性。

**⚠️ 局限性**

限制与挑战：①前向仿真需要规范为全局确定且可决定；②后向仿真需要规范为后向全局确定；③在存在多值或无确定性的规范时需要使用Lisbon或Incorrectness逻辑，验证成本可能上升；④目前仅证明了对安全性质（状态不变式）的传递，对完整性或活性性质的支持尚未系统化；⑤对更复杂的数据结构（如并发堆栈、队列）需要进一步验证其可扩展性。

---

## 222. TA-SparseMG: Trend-Aware Sparse Forecasting via Multi-Scale Gating for Long-Term Time Series

**arXiv ID:** 2606.27908 | [PDF](https://arxiv.org/pdf/2606.27908v1)

**作者:** Wenchao Liu `[一作]` (Guizhou Normal University), Xiangguang Xiong `[通讯]` (Guizhou Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种轻量级的跨周期时间序列预测框架TA‑SparseMG，改进了分布自适应、特征去噪和预测头的设计；

**💡 创新点**

创新点包括趋势感知可逆实例归一化（TA‑RevIN）、尺度自适应门控去噪模块和多尺度门控注意力MLP预测器，实现了分布匹配、噪声抑制与动态映射；

**🔧 技术方法**

技术上融合了可逆实例归一化、卷积去噪、门控机制、注意力以及基于SparseTSF的跨周期稀疏建模；

**📊 数据集**

在六个主流长周期时间序列预测基准（ETTh1、ETTh2、Weather、Electricity、Solar‑Energy、Traffic）上进行实验；

**📈 对比分析**

与包括DLinear、FITS、PatchTST、iTransformer、FEDformer、TQNet、SparseTSF/MLP、SimpleTM、FiLM等九个基线对比，TA‑SparseMG在大多数数据集和预测时长下均取得最低或第二低的MSE/MAE，显示出显著的性能提升；

**⚠️ 局限性**

局限性在于对周期长度、卷积核大小、门控系数等超参敏感，且对突发分布漂移的鲁棒性有限，未充分挖掘多变量间的交互信息。

---

## 223. S$^2$-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation

**arXiv ID:** 2606.27872 | [PDF](https://arxiv.org/pdf/2606.27872v1)

**作者:** Zhipeng Xie `[一作]` (East China Normal University), Jing Zhao `[通讯]` (East China Normal University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出S2-VLA框架，利用State‑Space Guided Adaptive Attention（SSGAA）解决长周期机器人操作中的累计误差。

**💡 创新点**

创新点在于引入基于任务进展的动态门控融合机制，能够在不同执行阶段自适应加权视觉、语言与动作序列信息，克服静态融合导致的早期偏差。

**🔧 技术方法**

使用预训练Vision‑Language模型（如Qwen3‑VL）、GRU状态更新、Transformer多层SSGAA、动态门控网络和端到端训练。

**📊 数据集**

使用长周期机器人操作基准数据集LIBERO、SimplerEnv和ALOHA进行训练与评估。

**📈 对比分析**

与FlowVLA、UnifiedVLA、OpenVLA‑OFT等SOTA VLA方法在LIBERO等基准上对比，S2‑VLA在4个子集的平均成功率达98.2%，参数仅2B，显著优于同类方法。

**⚠️ 局限性**

局限性：在更复杂的真实场景、极长序列和跨模态一致性方面仍需进一步验证，且模型依赖Transformer结构，计算量仍较大。

---

## 224. Applicability of memorization indicators for early spotting of overfitting while recalibrating sEMG-decoders on low sample sizes

**arXiv ID:** 2606.27855 | [PDF](https://arxiv.org/pdf/2606.27855v1)

**作者:** Stephan J. Lehmler `[一作]` (German Aerospace Center), Ioannis Iossifidis `[通讯]` (Ruhr West University of Applied Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在低样本迁移学习条件下，通过监测最后隐藏层ReLU神经元的激活率，早期检测sEMG解码器过拟合的可行性。

**💡 创新点**

首次将基于激活率的记忆化指标应用于sEMG解码器校准，并展示其与测试性能下降的相关性，提供了无需额外验证集的在线监测方案。

**🔧 技术方法**

使用1D卷积神经网络、ReLU激活、Adam优化、交叉熵损失，结合激活率统计（平均值、分位数、变异系数）作为指标。

**📊 数据集**

采用NinaPro数据库第2版（DB2），20个受试者的预训练数据与21-30号受试者的校准数据。

**📈 对比分析**

通过比较只微调最后两层与全参数微调的两种方式，发现后者在低样本情况下无法提升测试准确率，而前者可将准确率从0.065提升至约0.30-0.34；激活率下降与测试性能下降呈正相关。

**⚠️ 局限性**

实验仅使用极小网络、单一数据集、少量超参，激活率阈值缺乏通用性，且结果可能受数据集特性与模型规模限制，难以直接推广至更大或更深网络。

---

## 225. Booster Lab: A Data-Centric Pipeline for Learning Deployable Humanoid Locomotion Policies

**arXiv ID:** 2606.27813 | [PDF](https://arxiv.org/pdf/2606.27813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 226. USAD: Uncertainty-aware Statistical Adversarial Detection

**arXiv ID:** 2606.27832 | [PDF](https://arxiv.org/pdf/2606.27832v1)

**作者:** Zhijian Zhou `[一作]` (University of Melbourne), Feng Liu `[通讯]` (University of Melbourne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于统计的对抗检测框架USAD，通过捕捉对抗样本的全局和局部不确定性来判别是否存在攻击；

**💡 创新点**

创新点在于设计两种新的统计量——方差差异（VD）捕捉特征分布的全局扩散变化、扰动协方差差异（PCD）捕捉局部对噪声的敏感性，并将两者按相关性加权融合，弥补传统MMD对对抗样本不确定性敏感度不足的问题；

**🔧 技术方法**

采用两样本检验与置换检验框架，利用核方法（bounded kernel）对特征空间中的方差与协方差进行度量，并使用语义特征（分类器的倒数层输出）作为输入；

**📊 数据集**

在CIFAR‑10与ImageNet‑1K数据集上测试，使用ResNet‑18、ResNet‑50以及ViT‑B‑16等模型的特征；

**📈 对比分析**

与SAD基线方法（SAMMD、MMDAgg、MMD‑FUSE、MMD‑DUAL）以及单独的VD/PCD进行对比。结果表明，USAD在ε≥4/255且批量大小|Y|=50时可实现近乎100%的检测功率；在|Y|仅为10时仍保持功率1.0，且在自适应攻击、不同攻击范式以及混合清洁样本比例下均优于基线；

**⚠️ 局限性**

局限性包括：需要先验的清洁参考样本；对特征提取器和模型架构的依赖；对极小批量（如1~5样本）或极低对抗样本比例时检测功率可能下降；生成噪声扰动时计算开销相对较大；目前仅在图像分类任务中验证，跨模态推广需要进一步研究。

---

## 227. CSD: Content-aware Speculative Decoding for Efficient Image Generation

**arXiv ID:** 2606.27829 | [PDF](https://arxiv.org/pdf/2606.27829v1)

**作者:** Mingcheng Wang `[一作]` (East China Normal University), Shaohui Lin `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种内容感知的Speculative Decoding（CSD）算法，用于加速自回归图像生成模型的推理，同时保持或提升生成质量。

**💡 创新点**

创新点在于：①引入目标模型熵作为自适应概率松弛的依据，实现对不同区域（平滑与纹理）动态调整接受率；②基于总变差（TV）构建分布对齐过滤器，保证在松弛后仍与目标分布保持一致；③证明了在满足  min(1,q/p)≤b≤1 条件下，使用 norm{max(0,q-p)} 的重采样分布可达到最优。

**🔧 技术方法**

使用技术包括：Speculative Decoding、Jacobi解码、熵计算、TV距离过滤、Transformer式图像自回归生成、top‑k采样与CFG等。

**📊 数据集**

实验数据集：MS‑COCO 2017 验证集；使用 Lumina‑mGPT 7B（768×768）与 Janus‑Pro 1B/7B（384×384）作为基准模型。

**📈 对比分析**

与训练型方法（EAGLE、LANTERN）和无训练型方法（JD、SJD、Amplify、Addition、GSD）对比，CSD 在 Lumina‑mGPT 上实现 2.6× 的推理加速，CLIP 31.49；在 Janus‑Pro 7B 上实现 4.33× 加速，CLIP 32.21、FID 33.58，显著优于现有最优方法。

**⚠️ 局限性**

局限性：需要对 λ（松弛系数）和 δ（TV阈值）进行调参，尤其在高分辨率或高纹理区域松弛可能导致质量轻微下降；在小模型上加速效果受限，且对极端纹理细节的捕捉仍有待提升。

---

## 228. ATOD: Annealed Turn-aware On-policy Distillation for Multi-turn Autonomous Agents

**arXiv ID:** 2606.27814 | [PDF](https://arxiv.org/pdf/2606.27814v1)

**作者:** Qitai Tan `[一作]` (Tencent Inc.), Peng Chen `[通讯]` (Tencent Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种结合教师引导与奖励驱动的混合在线蒸馏算法，先通过OPD快速接近教师行为，再逐步加强RL以实现超越教师的提升；同时引入基于教师-学生不一致与不确定性的Turn‑Level Disagreement‑Uncertainty Reweighting（T‑DUR）对关键决策回合加权；

**💡 创新点**

(1) OPD与RL的自适应温度调度，使训练在早期快速收敛、后期获得更高奖励；(2) 将权重从token级提升到turn级，以更稳定、语义相关的方式分配监督；

**🔧 技术方法**

On‑policy distillation (OPD)、Group Relative Policy Optimization (GRPO)、温度调度、T‑DUR加权、单一混合优势（A_t=κ·Δlogp+ρ·Δlogp）和CLIP surrogate优化；

**📊 数据集**

ALFWorld、WebShop、Search‑QA（包含Natural Questions、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle）；

**📈 对比分析**

与Vanilla、GRPO、SDAR、OPD、SOD、TCOD等后训练基线相比，在三大基准上平均成功率最高；在最小学生模型上从几乎0%提升至82%+，超过对应教师模型；在较大模型上仍保持领先；

**⚠️ 局限性**

缺点包括：(a) 需要教师模型且训练依赖教师分布；(b) 温度调度参数需手工设定，可能对不同任务敏感；(c) T‑DUR仅关注token级对齐与不确定性，未考虑更复杂的策略梯度信号；(d) 在极度稀疏奖励环境下，RL信号仍可能不足。

---

## 229. Reliable Homomorphic Matching for Fuzzy Labeled PSI at Scale

**arXiv ID:** 2606.27803 | [PDF](https://arxiv.org/pdf/2606.27803v1)

**作者:** Erkam Uzun `[一作]` `[通讯]` (Georgia Institute Of Technology), Erkam Uzun (Georgia Institute Of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

设计并实现了一种可扩展的 Fuzzy Labeled PSI（FLPSI）协议，解决了传统实现中随数据库规模放大的实现音响错误（RSE）问题。

**💡 创新点**

创新点在于提出可组合的多 token 轮加密核 CSTPSI，给出闭式 RSE 上界并将 token 轮数作为可调参数；同时引入一次性 GC 和加密查询缓存，使安全改进几乎不增加在线成本。

**🔧 技术方法**

采用级联 BFV 同态加密、AES‑OPRF garbled circuit、Shamir 秘密共享等技术；通过缓存 GC、查询加密窗口和重用重构权重实现性能优化。

**📊 数据集**

使用公开数据集 LFW（人脸嵌入）和 Deep1B（深度特征）进行评估，以验证协议在真实应用场景下的效果。

**📈 对比分析**

与基线 STLPSI（单 token 轮）在同一代码基下对比：在百万级数据库时 RSE 从 100% 降至 0%，单线程加速超过 20×，通信量减少多达 93%，在大多数规模下保持接近基线性能。

**⚠️ 局限性**

局限性包括：只能消除实现错误，无法降低匹配器固有的误匹配率（FMR）；标签长度有限（主要针对 23 位到 64 字节）；仅在半诚实模型下安全；每增加一轮 token 需要额外通信和计算，且不适用于极大标签或高 k 的情况。

---

## 230. Parameterized Verification of Asynchronous Round-Based Distributed Algorithms via Reduction to Finite-Counter Systems

**arXiv ID:** 2606.27867 | [PDF](https://arxiv.org/pdf/2606.27867v1)

**作者:** Nathalie Bertrand `[一作]` (University of Rennes), Sasha Rubin `[通讯]` (University of Sydney)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出一种针对异步轮次分布式算法的参数化验证框架，解决了每个进程无界轮次导致的无限状态问题。

**💡 创新点**

创新点在于证明此类算法的参数化验证是不可判定的，但提供了一个完整的六步归约，将原问题精确映射到有限计数系统（finite‑counter system）上，从而可使用成熟的符号模型检查器进行求解。

**🔧 技术方法**

核心技术包括：接收消息抽象、进程身份抽象、半同步约束、有限窗口抽象、历史记录扩展以及轮次识别抽象，并通过这些抽象构造可判定的计数系统；实现时使用nuXmv的IC3引擎进行模型检查。

**📊 数据集**

实验数据集涵盖四个经典协议：Ben‑Or（崩溃/拜占庭错误）、Bracha拜占庭一致性和Raft领导者选举，所有模型均已公开于GitHub仓库。

**📈 对比分析**

与传统手工证明或专用工具相比，该方法仅需几秒钟（最大15秒）完成验证，IC3深度低至3–14层，且能够在出现错误时快速给出反例，证明了归约的实用性与高效性。

**⚠️ 局限性**

局限性包括：参数化验证仍为不可判定，归约依赖于算法满足特定的递增性条件；不支持概率性行为或轮次/状态数随轮次递增的协议；且目前只针对有限轮次模板，无法直接处理如Paxos或DAG共识等动态拓扑的算法。

---

## 231. Long-Term Prediction of Local and Global Human Motion with Occlusion Recovery

**arXiv ID:** 2606.27900 | [PDF](https://arxiv.org/pdf/2606.27900v1)

**作者:** Qiaoyue Yang `[一作]` (Bielefeld University), Sven Wachsmuth `[通讯]` (Bielefeld University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种非自回归时空Transformer模型，可同时预测局部姿势和全局运动，并能处理遮挡和可变历史长度。

**💡 创新点**

创新点在于将空间与时间注意力结合于非自回归Transformer，既实现了局部与全局双重预测，又引入学习式遮挡补偿与可变观察窗口支持。

**🔧 技术方法**

采用Pose Embedding、空间注意力、时间位置编码、非自回归Transformer结构、学习式遮挡标记、L1损失及AdamW优化等技术。

**📊 数据集**

使用公开数据集Human3.6M、AMASS和HA4M进行训练与评估。

**📈 对比分析**

与标准Transformer、SPOTR、POTR以及ST-Transformer比较，MPJPE与MAE均显著降低，长时序（1秒）误差显著提升，推理时间约250 ms。

**⚠️ 局限性**

局限性包括对动作切换的预测能力不足、复杂手臂/手部动作精度仍有限、对稀缺动作数据敏感且未充分利用动作语境信息。

---

## 232. A Multi-Attribute Latent Space for Visual Analysis of Watches

**arXiv ID:** 2606.27897 | [PDF](https://arxiv.org/pdf/2606.27897v1)

**作者:** Kai Lawonn `[一作]` (Leipzig University), Monique Meuschke `[通讯]` (Otto von Guericke University of Magdeburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一个基于多属性UMAP的手表可视化系统，利用颜色、设计和类型三维度生成手表设计空间，并提供交互式探索与查询；

**💡 创新点**

提出将属性特定模糊邻域图与类别约束和衰减布局项结合的多属性UMAP变体，以及将手表类型作为全局结构、颜色和设计作为局部相似性来构造嵌入；

**🔧 技术方法**

使用U‑Net进行表盘分割、ViT进行手表类型分类、CIELAB色彩抽样、HOG设计描述、UMAP梯度优化，并通过交互式可视化（平移、缩放、悬停、过滤、搜索）呈现结果；

**📊 数据集**

使用约33,000张来自Chrono24的手表图片（去重后32,861张），涵盖七种手表类型；

**📈 对比分析**

通过邻域保留Jaccard、误差E、检索Recall@k与R@1等指标评估，并与PCA、t‑SNE、标准UMAP对比；提出的方法在邻域保留和检索准确率上优于基线（如Top‑1≈0.88、Recall@5≈0.96），运行时间约0.5秒；

**⚠️ 局限性**

仅适用于视觉相似性较强的类别，依赖预训练模型，难以处理极端光照或遮挡；评估样本有限，缺乏大规模统计验证；未充分整合品牌、机械细节等非视觉属性；

---

## 233. SEADA: An efficient methodology for optimizing mixed-precision DNNs on multi-precision spatial architectures

**arXiv ID:** 2606.27884 | [PDF](https://arxiv.org/pdf/2606.27884v1)

**作者:** Leandro Fiorin `[一作]` (Independent Researcher), Cristina Silvano `[通讯]` (Politecnico di Milano)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 SEADA 框架，整合了多精度空间加速器的系统级分析成本模型、快速映射工具、浮点层的分析模型以及基于位级熵的层级精度选择方法，用于在多精度硬件上高效部署混合精度深度网络。

**💡 创新点**

创新点在于：①将熵驱动的精度选择（EAGL）扩展至多种比特宽度并与硬件映射紧耦合；②构建精细化的整数+浮点混合成本模型，准确捕捉量化/去量化开销；③提出“计算比特减量（CBR）”作为硬件无关的准确性代理，显著加速架构空间探索；④通过快速映射工具和一次性熵评估，避免了传统全搜索或多轮微调的高成本。

**🔧 技术方法**

主要技术包括：整数与浮点混合分析成本模型、QuickFlow 基于活动的映射搜索、Learned Step Size Quantization（LSQ）量化、熵估计的 EAGL 精度分配、Knowledge Distillation 微调、FlashAttention‑2 和 Newton–Raphson 方法用于 Softmax/GELU 等浮点运算，以及 CBR 作为准确性代理。

**📊 数据集**

使用的数据集为 BERT‑base 在 SQuAD 1.1 上的问答任务，以及 ResNet‑50 在 ImageNet 图像分类任务；两者在全精度模型训练后作为教师模型进行知识蒸馏。

**📈 对比分析**

与统一 8‑bit 方案和全精度方案对比，SEADA 在 BERT‑base 上能在能量预算为 45% 时仅损失 1% 以上准确率，同时实现 48% 的 EDP 缩减；在 ResNet‑50 上能在能量预算为 55% 时仅损失 4.5% 准确率，同时实现 57% 的 EDP 缩减。相较于纯 8‑bit 方案，能量/EDP 进一步提升 20–30%；相较于全精度，能量/EDP 几乎实现 90% 以上的节省。

**⚠️ 局限性**

局限性包括：①需要在每一次精度分配后进行一次微调，仍存在训练成本；②熵估计仅基于单一预训练模型，可能在极端低精度或不同网络结构下失效；③目前仅支持整数 + FP32 计算单元，无法直接处理更高或更低的浮点精度；④对硬件的假设（Eyeriss‑style 结构）较为有限，推广到其他空间加速器时需重新建模；⑤CBR 作为准确性代理在某些任务或网络中可能不够稳健。

---

## 234. Hippocampus-DETR: An Explicit Memory Object Detection Framework Based on Hippocampus Modeling

**arXiv ID:** 2606.27831 | [PDF](https://arxiv.org/pdf/2606.27831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 235. A Study of Temporal Fusion Strategies for Named Entity Recognition in Historical Texts

**arXiv ID:** 2606.27881 | [PDF](https://arxiv.org/pdf/2606.27881v1)

**作者:** Emanuela Boros `[一作]` `[通讯]` (EPFL), Emanuela Boros (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地将出版年份信息嵌入历史文本 NER 模型中，设计并比较多种早期/晚期融合策略，并通过线性探测验证模型内部对时间的编码。

**💡 创新点**

首次在 token‑level NER 任务中全面对比早期与晚期时间融合方法，探讨绝对年与相对距离编码的影响，并通过线性探针验证时间信息是否真正被内部化。

**🔧 技术方法**

基于 Transformer 的多语种历史 BERT，采用交叉注意力、适配器、拼接、FiLM 等轻量级融合模块；使用线性探针、配对 t‑检验等评估方法。

**📊 数据集**

HIPE‑2020 数据集的法语和德语子集（含出版年份、粗粒度实体标签）。

**📈 对比分析**

通过每年微观 F1 分数、年代段、实体长度、实体类型等维度进行细粒度比较；晚期融合方法整体提升幅度较小但更稳健，只有在某些配置下显著；未能在大多数年份上获得统计显著性提升。

**⚠️ 局限性**

仅考虑年份级粒度，实验仅覆盖法语和德语，使用单一 BERT 体系；探针仅评估线性可解码的时间信息，可能低估更细微的编码；缺失或噪声时间元数据的情况未充分测试。

---

## 236. Differential Privacy over Hamming Codes

**arXiv ID:** 2606.27849 | [PDF](https://arxiv.org/pdf/2606.27849v1)

**作者:** Borzoo Rassouli `[一作]` (Nokia Bell Labs), Morteza Varasteh `[通讯]` (University of Essex)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

探讨在二进制对称信道（BSC）上传输计数查询结果时，利用汉明码编码可实现的差分隐私提升，且不增加额外噪声或在线计算开销。

**💡 创新点**

提出通过对码字进行最小距离排列（每对相邻码字汉明距离为3）来最小化隐私损失的最优策略，并给出显式构造方法。

**🔧 技术方法**

使用汉明码、最大似然（ML）解码、灰度编码及线性代数理论对隐私损失进行解析求解。

**📊 数据集**

未使用具体数据集，研究基于理论分析和符号推导。

**📈 对比分析**

将最优隐私损失与最差情况（相邻码字距离为n）做比较，证明最优方案可将隐私损失降低到理论最小值，提升量随码长n增长而显著。

**⚠️ 局限性**

仅适用于汉明码与计数查询的情形，假设BSC噪声可被利用且不考虑实际硬件误差；对其他编码或查询类型需进一步研究。

---

## 237. Robust Shattering Arguments

**arXiv ID:** 2606.27847 | [PDF](https://arxiv.org/pdf/2606.27847v1)

**作者:** Mohsen Ghaffari `[一作]`, Alexandre Nolin `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对分裂（shattering）技术中的概率依赖性假设进行严谨修正，纠正了先前关于最大独立集、(Δ+1)-着色和分布式Lovász局部引理（LLL）算法的错误论证，并提供了一套通用的、对长时间依赖不敏感的分析工具。

**💡 创新点**

提出了“粒度采样（granular sampling）”框架和“机会式进展过程（opportunistic process）”模型，能够在不依赖节点间独立性的前提下，给出分裂集上概率衰减的上界，从而实现鲁棒的分裂分析。

**🔧 技术方法**

使用自适应采样过程的组合概率推导、全局再分配（reconstruction）技术、分离集计数上界（基于Cayley树的计数）、以及对分布式LLL的渐进重写证明。

**📊 数据集**

该工作为理论分析，未使用实验数据集，所有结果均基于抽象图模型与概率论证明。

**📈 对比分析**

与先前的非严格依赖性假设的分裂证明进行对比，证明在满足改进后可实现的时间复杂度：如分布式LLL在满足 8pd^{-9}<1 条件下，随机算法以 O(d^2)+_LLL,q,d(log n) 轮完成；Δ+1-着色可在 O(_d1LC(log n)) 轮内完成，性能与现有最优结果相当或更优。

**⚠️ 局限性**

局限性在于仍需要对算法预处理阶段的轮数做严格上界（如 ω(1) 轮时需要更细致的依赖分析），且对某些特殊图结构（如极端高度或特殊依赖图）仍需进一步验证。

---

## 238. NormAct: A Benchmark for Hidden Social Norm Compliance in Embodied Planning

**arXiv ID:** 2606.27826 | [PDF](https://arxiv.org/pdf/2606.27826v1)

**作者:** Shiyun Zhao `[一作]` (Beijing Institute for General Artificial Intelligence), Bo Dai `[通讯]` (Beijing Institute for General Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了NormAct基准，用于评估嵌入式代理在完成普通任务时对隐含社会规范的识别与执行能力。

**💡 创新点**

创新点在于将隐含社会规范视为可执行动作的约束，设计多维度任务与规范映射，并引入情境化提示生成器自动激活与语境化规范。

**🔧 技术方法**

采用多模态大型语言模型（GPT‑5.4、Claude Opus 4.7、Gemini 3 Pro）结合高层动作空间、情境化提示、检索式提示与自动生成提示进行规划与评估。

**📊 数据集**

使用NormAct数据集（基于TongSim生成的550个第一人称场景，涵盖11类任务与5个规范维度）以及公开的语义注释。

**📈 对比分析**

通过对不同提示条件（无提示、类别提示、具体提示、证据提示、检索提示、自动生成提示）和不同模型的基准评估，发现仅靠目标实现的准确率为67.3%，但规范合规率仅26.4%；添加情境化提示后规范合规率可提升至67.1%，任务成功率提升至50.2%。

**⚠️ 局限性**

局限性包括任务类型有限、动作序列简短、文化多样性不足、自动生成提示尚不如人工提示有效，且未验证在弱模型或不同任务上的泛化。

---

## 239. Scalable and Differentiable Point-Cloud Registration Using Maximum Mean Discrepancy

**arXiv ID:** 2606.27818 | [PDF](https://arxiv.org/pdf/2606.27818v1)

**作者:** Rixon Crane `[一作]` (Data61, CSIRO), Russell Tsuchida `[通讯]` (Monash University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种可微分、对应关系无关的点云配准方法MMD-Reg，并将其作为可微分优化层集成进深度学习网络Neural MMD-Reg，用于无监督和有监督的配准任务。

**💡 创新点**

创新点在于将最大均值差异（MMD）通过随机傅里叶特征近似为线性时间的非线性最小二乘目标，并利用隐函数定理实现对配准解的自动微分，使得配准可嵌入端到端训练的模型中；此外，提出了多尺度随机特征与残差加权策略，提升了在噪声、稀疏和遮挡条件下的鲁棒性。

**🔧 技术方法**

主要技术包括：随机傅里叶特征（RFF）近似高斯/拉普拉斯核、Levenberg–Marquardt求解非线性最小二乘、隐函数定理与JAXopt实现隐式微分、Set Transformer架构用于预测初始化与尺度参数、以及加权残差实现部分重叠配准。

**📊 数据集**

使用了PCPNet（synthetic CAD模型）、Wild Places和KITTI（真实户外激光雷达扫描）以及ModelNet40（CAD模型）等公开数据集进行实验。

**📈 对比分析**

与ICP、GICP、FilterReg、GMMReg、CPD等经典几何/概率配准方法，以及Go-ICP、FGR、PointNet-LK、DCP-v2、RPM-Net、REGTR、GeoTransformer、DCMR、RPMNet++等学习型方法进行对比。实验表明MMD-Reg在保持线性时间复杂度的前提下，在多种噪声、稀疏、遮挡和大规模点云场景中实现了更低的旋转/平移误差和更快的推理速度；Neural MMD-Reg在无监督和有监督任务中均超过现有基准，误差低至0.01°/0.0001m。

**⚠️ 局限性**

局部收敛性受限，需良好初始化或多尺度细化；随机特征近似导致解的随机性，对D值选择敏感；目前仅处理无属性点云，尚未验证对多模态属性（颜色、语义等）的扩展。

---

## 240. Optimizing Teacher-Student Partitioning for Scalable Knowledge Distillation on HPC Systems

**arXiv ID:** 2606.27797 | [PDF](https://arxiv.org/pdf/2606.27797v1)

**作者:** Adrian P. Dieguez `[一作]` (Qualcomm AI Research), Harris Teague `[通讯]` (Qualcomm AI Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多节点HPC环境下针对Generalized Knowledge Distillation（GKD）提出了教师-学生分区优化方法，显著提升训练吞吐量；

**💡 创新点**

核心创新在于：①将教师模型从训练引擎剥离，去除不必要的优化器状态；②实现教师与学生分区解耦，按拓扑选择垂直（DDP）或水平（TP）分区；③基于简化的成本模型分析并识别出何时水平分区优于垂直分区；

**🔧 技术方法**

技术包括DeepSpeed（ZeRO优化）、DeepSpeed-Inference（TP）、FlashAttention‑2、BF16/FP16混合精度、PyTorch 2.5、TRL库、RoCE网络、NVLink互连；

**📊 数据集**

使用8B参数的量化LLaMA‑3学生模型与浮点LLaMA‑3‑8B教师模型，训练序列长度1024 token；

**📈 对比分析**

与TRL默认GKD实现对比，实验在16节点8×H100 GPU（128 GPU）上，改进方案可将样本吞吐量提升约50%至67%，可在更大微批量下运行而不OOM；

**⚠️ 局限性**

局限性包括：①模型规模与网络拓扑依赖强，低带宽/高延迟环境下TP优势不明显；②目前仅支持数据/张量并行，未考虑序列/流水线并行；③成本模型为简化近似，缺乏自动调优实现；

---

## 241. Text as Illumination: Spatial Contrastive Retinex Learning for Language-guided Medical Image Segmentation

**arXiv ID:** 2606.27794 | [PDF](https://arxiv.org/pdf/2606.27794v1)

**作者:** Jian Shi `[一作]` (Dalian University Of Technology), Huchuan Lu `[通讯]` (Dalian University Of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种文本作为光照的Retinex网络（TIRNet），用于语言指导的医学图像分割。

**💡 创新点**

通过将文本嵌入视为语义光照，设计了正负光照映射的RTMB模块实现前景增强与背景抑制，并引入多尺度光照监督损失（MSIS-Loss）实现精细跨模态对齐。

**🔧 技术方法**

采用Retinex-inspired文本调制块RTMB、Consistent Detail Compensation Block CDCB、Region-Grounded Contrastive Loss RGC-Loss、Background Suppression Loss BS-Loss、CLIP文本编码器和U‑shaped encoder‑decoder结构。

**📊 数据集**

使用公开的MosMedData+（胸部CT）和QaTa-COV19（胸部X光）数据集进行实验。

**📈 对比分析**

与单模态（U‑Net、nnUNet）和多模态（LViT、TeViA等）方法在m-Dice、m-IoU、g-Dice、g-IoU等指标上对比，TIRNet在两数据集均取得最高分数，提升约1–3% Dice/IoU。

**⚠️ 局限性**

模型对文本描述的质量敏感，受限于CLIP预训练编码器的通用性；对不同分辨率下的鲁棒性和计算成本未完全评估，需要进一步研究更细粒度的文本-像素对齐。

---

## 242. Listwise Explanation of Embedding-Based Rankings via Semantic Chunk Grouping

**arXiv ID:** 2606.27980 | [PDF](https://arxiv.org/pdf/2606.27980v1)

**作者:** Hyunkyu Kim `[一作]` (KakaoBank Corp), Youngjun Kwak `[通讯]` (KakaoBank Corp)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ChunkGroupSHAP，一种基于列表Shapley值的解释方法，用语义化块群组对密集检索模型的排名进行解释。

**💡 创新点**

创新点在于将语义相似的块聚类为跨文档特征，使解释单元与密集检索的句子/段落级表示对齐，并提供语料库级和查询本地两种聚类范围。

**🔧 技术方法**

采用块级分块+嵌入、k-means聚类、KernelSHAP估计Shapley值、NDCG/τ_b等列表级效度指标进行评估。

**📊 数据集**

使用四个主基准：MS MARCO、FinanceBench、AILACaseDocs、FinQA，并与BM25及E5系列（small/base/large）嵌入检索器对比。

**📈 对比分析**

通过“Fidelity_b”指标与RankSHAP、RankingSHAP等词级解释器对比，发现对密集检索在域内长文档集上效果更好；在BM25、MS MARCO等场景下词级方法仍占优；查询本地聚类进一步提升性能。

**⚠️ 局限性**

局限包括：仅评估对原始检索器的忠诚度而非人类可解释性；聚类质量与超参数对结果敏感；仅解释固定候选列表，无法覆盖全检索管线；使用KernelSHAP近似可能导致样本方差和计算成本上升。

---

## 243. Decoys Cannot Go Everywhere: Mapping the Deception Surface in MITRE ATT&CK

**arXiv ID:** 2606.27966 | [PDF](https://arxiv.org/pdf/2606.27966v1)

**作者:** Veronica Valeros `[一作]` (Czech Technical University in Prague), Harm Griffioen `[通讯]` (Delft University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估并绘制了 MITRE ATT&CK 250 技术中哪些攻击行为能够被基础设施诱饵（decoy）干预，并提出四标准评估体系。

**💡 创新点**

首次系统性量化攻击技术可被诱饵覆盖的“欺骗表面”，并提出 Sweep‑Seek 两种诱饵放置模式。

**🔧 技术方法**

基于四项评估准则的结构化评分法，结合专家评审与规则化判定。

**📊 数据集**

MITRE ATT&CK v18.1 的 250 条技术作为评估对象。

**📈 对比分析**

通过专家评审得出近似一致度，评分精确一致率约 28–44%，接近一致率 67–76%，表明评估方法可重复但仍需细化。

**⚠️ 局限性**

评估由单一主评审完成，未在真实部署中验证，且仅覆盖基础设施诱饵，未考虑叙事式诱饵等更广泛情景。

---

## 244. Grammar-Guided Hierarchical Parsing for Long-form Audio Activity Recognition

**arXiv ID:** 2606.27965 | [PDF](https://arxiv.org/pdf/2606.27965v1)

**作者:** Peng Zhang `[一作]` (University of Surrey), Wenwu Wang `[通讯]` (University of Surrey)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种基于层次活动语法（Hierarchical Activity Grammar, HAG）的结构化推理框架，用事件级证据实现长时音频的 Act–Sub–Event 解析，进而实现无监督的子活动划分和活动分类。

**💡 创新点**

创新点在于将层次化的概率上下文无关文法（PCFG）与事件检测输出结合，通过语法引导的 Earley‑style MAP 解析在不需子活动或活动标注的前提下，强制执行子活动顺序与层次关系，提升了全局时序一致性。

**🔧 技术方法**

核心技术包括：事件级音频检测器（SlowFast+ActionFormer 生成事件段及类别后验），HAG 的非终结符与噪声符号定义、语法规则概率学习，以及基于语法的最大后验解析（Earley‑style 维特比）。

**📊 数据集**

实验使用了长时音频数据集 MultiAct，该数据集包含 3 种高层活动、12 种子活动和 44 种事件标签，总时长近 9 小时。

**📈 对比分析**

在多项指标上与完全监督方法比较，grammar‑induced 方案在 Edit 分数上提升显著（如 Eval 语句从 24.6% 提升至 35.3%），同时在高层活动分类上可达 66.7% 的 Top‑1 准确率，虽然在精细边界指标（F1@25/50、帧级精度）上略逊于监督模型，但已展示无监督层次推理的可行性。

**⚠️ 局限性**

局限性包括：解析依赖事件检测的粒度，导致子活动边界对齐不佳；对语法权重 λ 的敏感性需要手工调参；以及缺乏对更大规模或多语种、跨域数据的泛化验证。

---

## 245. Understanding How MLLMs Describe Artworks Using Token Activation Maps

**arXiv ID:** 2606.27947 | [PDF](https://arxiv.org/pdf/2606.27947v1)

**作者:** Nicola Fanelli `[一作]` (University of Bari Aldo Moro), Giovanna Castellano `[通讯]` (University of Bari Aldo Moro)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

分析多模态大型语言模型在艺术品描述中的视觉定位，利用Token Activation Map（TAM）探测不同语义类别的视觉归因。

**💡 创新点**

提出在艺术语境下的Token级定位分析，展示风格与情感描述更分散、具体对象更聚焦，并验证TAM在多模态模型中的可解释性。

**🔧 技术方法**

采用TAM方法、Qwen2‑VL‑2B‑Instruct生成描述并提取激活图、Qwen3‑4B‑Instruct-2507进行语义分块、SAM 3进行对比定位、LLM评判元数据准确性。

**📊 数据集**

使用WikiArt上前1000幅最受关注的绘画图像，配合Qwen2‑VL‑2B‑Instruct生成的全句描述。

**📈 对比分析**

通过归一化空间熵、Gini系数、前10%激活度等指标比较不同语义类型的定位特征；与SAM 3的IoU对比表明TAM的定位粗糙但仍可识别；元数据预测中艺术家识别率高于题目识别率。

**⚠️ 局限性**

局限于单一模型（Qwen2‑VL），缺乏精确的标注对比；TAM激活图粗糙且对图像边界不敏感；研究仅覆盖西方画作，无法推广至其他艺术传统。

---

## 246. It Lied to a Doctor to Buy Poison Ingredients: Quantifying Real-World Misuse of Phone-use Agents

**arXiv ID:** 2606.27944 | [PDF](https://arxiv.org/pdf/2606.27944v1)

**作者:** Yiming Sun `[一作]` (Fudan University), Mi Zhang `[通讯]` (Fudan University)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了基于多模态大型语言模型的手机助手在真实设备上的误用风险，构建了法规驱动的误用基准并对其安全性进行量化评估。

**💡 创新点**

创新点在于：①提出以六条法律与三十四条公开违规案例为基础的法规驱动误用基准；②设计三层评估框架（单步意识、轨迹能力、设备执行）；③揭示并量化“安全意识–执行差距”，并通过神经元干预等技术实现对策。

**🔧 技术方法**

使用的技术包括视觉对齐的多模态 LLM（如Claude‑Sonnet、Gemini、GPT‑5.4、Qwen、AutoGLM 等）+ UI 动作生成与执行；机制解释方法识别安全相关神经元；以及外部检测、提示干预和激活调节三类防御。

**📊 数据集**

使用数据集为 1,381 条单步误用样本，涵盖 6 类/34 子类，覆盖 27 个日常应用，源自 6 条法规与 34 条官方公开违规案例，随后通过 LLM 变异与人工审核扩充为 144 条轨迹样本及 50 条设备测试样本。

**📈 对比分析**

实验对 9 个代理（4 商用、5 开源）分别在单步、轨迹和设备层面进行评估，结果显示大多数开源模型完成率超过 78%（最高 96%），商用模型完成率约 39‑86%；速度方面，小型开源模型平均 44‑58 秒/任务，低于人类基准 78 秒；成本上，开源模型本地部署成本几乎为零，商用模型每任务约 0.04‑0.21 美元。

**⚠️ 局限性**

局限性包括：仅评估自然语言普通指令未使用解锁模板或攻击增强；未覆盖对本地部署攻击或平台级防护；实验受人工审核限制，无法大规模实时测试；数据集规模相对有限，主要聚焦手机应用场景。

---

## 247. MathModDB: A Database for Mathematical Models

**arXiv ID:** 2606.27933 | [PDF](https://arxiv.org/pdf/2606.27933v1)

**作者:** Jochen Fiedler `[一作]` (Fraunhofer Institute for Industrial Mathematics), Thomas Koprucki `[通讯]` (Weierstrass Institute for Applied Analysis and Stochastics)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文实现了MathModDB——一个面向数学模型的知识图谱服务，部署在MaRDI门户上，并通过一个电离子发射模型示例展示了如何从研究问题到模型、公式、量、假设、计算任务及相关软件的全链路展示与查询；

**💡 创新点**

创新点在于：①构建了跨学科通用的数学模型语义化数据模型并实现为可搜索的知识图谱；②引入模型细化层级与假设标注，支持模型的版本化和可比性；③与MathAlgoDB、MaRDMO等生态系统连接，形成从问题到模型到算法到软件的闭环；

**🔧 技术方法**

技术手段包括：Wikibase（MaRDI门户）、OWL/RDF本体、SPARQL查询接口、链接开放数据（Wikidata、QUDT）、插件化工具（MaRDMO）以及知识图谱设计与维护；

**📊 数据集**

数据集为MathModDB自身的知识库，当前包含227个数学模型、708条公式、792个量、173个计算任务以及24,762条语义声明，覆盖约200个模型；

**📈 对比分析**

论文未给出传统算法性能评测；其比较方法主要通过语义层级和假设标注来进行模型间可比性分析，并通过与MathAlgoDB链接提供对应算法任务，方便用户自行选取并评估算法；

**⚠️ 局限性**

局限性包括：①模型的采集和注入依赖人工专家，规模扩展受限；②离散化（数值求解）在知识图谱中仅隐式表示，缺乏细粒度建模；③尚未实现全面的自动化文本抽取与语义标注，导致更新效率不高；

---

## 248. Provable Reductions in TFNP

**arXiv ID:** 2606.27931 | [PDF](https://arxiv.org/pdf/2606.27931v1)

**作者:** Noah Fleming `[一作]` (Lund University and Columbia University), Robert Robere `[通讯]` (McGill University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

提出了一类新的命题证明系统 ⟨EF,R⟩，将 Extended Frege 与任意 TFNP 搜索问题 R 组合，通过可证明的映射还原实现对无效子句搜索问题的证明，并研究该系统与已知证明系统的多项式等价性。

**💡 创新点**

通过构造白盒可证明映射还原与黑盒决策树还原的对应关系，首次证明 ⟨EF,Res⟩ 与 G₁ 及 [EF,Res] 在多项式等价；进一步将此框架泛化到任意 TFNP R，并给出一个通用定理：若 P 是足够强的证明系统，则存在多项式可计算的搜索问题 R_P，使得 ⟨EF,R_P⟩ 与 P 在多项式等价。

**🔧 技术方法**

结合边界算术的可证明见证定理、EF 的映射还原、决策树与白盒/黑盒 TFNP 的关系、V¹ 等有限算术证明、证明转译技术，以及对 EF、G_i、Implicit Resolution 等系统的正式编码与分析。

**📊 数据集**

本文为理论研究，不涉及实验数据集。

**📈 对比分析**

通过多项式等价性证明，展示 ⟨EF,R⟩ 与已知系统（EF、G₁、[EF,Res]、Implicit Resolution）在证明长度和复杂度上可互相模拟，从而表明在 EF 可证明 R 属于 TFNP 时，⟨EF,R⟩ 与 EF 等价；而在 R 不是 EF 可证明的 TFNP 时，⟨EF,R⟩ 与更强的系统（如 G₁）等价。

**⚠️ 局限性**

结果依赖于 EF 能证明 R 属于 TFNP 的可证明性，且对弱系统（如 Resolution、Frege）的完整通用定理尚未给出；对更强系统的进一步泛化以及对 EF 之外的白盒 TFNP 类的探究仍是开放问题。

---

## 249. Home3D 1.0: A High-Fidelity Image-to-3D Asset Generation System for Interior Design

**arXiv ID:** 2606.27923 | [PDF](https://arxiv.org/pdf/2606.27923v1)

**作者:** Yiyun Fei `[一作]` (Alibaba Group and Taobao), Feng Zhang `[通讯]` (Alibaba Group and Taobao)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个名为 Home3D 1.0 的四阶段室内家具图像到 3D 生成系统，能够从单张照片生成高质量、可编辑的 3D 资产（完整网格、全覆盖纹理、PBR 材质以及可材质编辑的部件）

**💡 创新点**

将任务拆分为专门的几何、纹理、材质与部件四个模块，采用粗细分层流匹配 Diffusion 与 VAE 的 Latent SDF 生成、视频基 UV 投票分割、层级多模态材质检索以及一次性生成可编辑部件的多头 SDF 解码，兼顾物理真实性与可编辑性

**🔧 技术方法**

VAE+Flow‑Matching Diffusion、Latent SDF、Coarse‑to‑Fine DiT、Perceiver‑style 交叉注意力、UV 投影与重投影、3D 纹理场、MatWeaver 视频分割、RoMA+P4Pf 相机对齐、Qwen3‑VL 多模态检索、VLM 重排序、PartVAE+PartDiT 多头 SDF 解码、物理烘焙与 Draco 压缩

**📊 数据集**

公开的 ObjaverseXL 与私有 Tmall Design Home 家具模型（用于预训练与高质量微调）；专业室内设计 PBR 材质数据库（层级分类）；带部件 ID 的 3D 资产合成视频；Blender Cycles 生成的多视图训练样本

**📈 对比分析**

在 100 例家具基准上与 Hunyuan3D、Seed3D、Tripo 三款闭源系统对比，使用 CD、EMD、F1@0.01、CLIP‑I 与 LPIPS 等指标；Home3D 在几何指标上最低（CD 0.4936×10⁻³、EMD 5.174×10⁻²、F1 0.6329），材质与纹理指标也同样表现优异，整体性能领先

**⚠️ 局限性**

对极端遮挡或光照变化的鲁棒性仍有限；材质检索高度依赖高质量数据库，缺失或误检时效果下降；部件生成在高分辨率下仍受多头 SDF 解码复杂度限制；未实现跨阶段反馈与动态可变形物体的处理

---

## 250. Reflect-R1: Evidence-Driven Reflection for Self-Correction in Long Video Understanding

**arXiv ID:** 2606.27922 | [PDF](https://arxiv.org/pdf/2606.27922v1)

**作者:** Shuimu Chen `[一作]` (Tsinghua University), Fei Ma `[通讯]` (Guangdong Laboratory of Artificial Intelligence and Digital Economy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Reflect‑R1 框架，采用基于证据驱动的三阶段（直觉‑验证‑仲裁）自纠机制，解决长视频理解中的自反与误报问题。

**💡 创新点**

创新点在于：①引入外部视觉证据检索实现独立验证，彻底打破闭环自反的盲目信心；②设计阶段解耦的强化学习 SD‑GRPO，消除多阶段策略耦合；③构建 120K 样本数据集，为多模态自反训练提供专属数据。

**🔧 技术方法**

技术包括：多模态大型语言模型（如 Qwen2.5‑VL‑7B）、工具调用与关键帧检索、三阶段推理管道、阶段解耦的 GRPO 强化学习，以及精细化奖励设计（格式、准确、诚实、反腐）。

**📊 数据集**

使用数据集：VideoMME、LongVideoBench、MLVU、Haystack‑LVBench；自建 Reflect‑R1‑CoT‑90k（SFT）与 Reflect‑R1‑RL‑30k（RL），并整合 LLaVA‑Video‑178K、Panda‑70M、NExT‑QA、PerceptionTest、CLEVRER、STAR 等源数据。

**📈 对比分析**

与内部闭环反射模型（如 Qwen2.5‑VL‑7B、GPT‑4o 等）以及工具增强基线（VideoAgent、TimeSearch‑R）对比，Reflect‑R1 在 VideoMME、LongVideoBench、MLVU 上提升终端准确率约 +2.82% / +1.41%，Temporal Search 指标亦优于 TimeSearch‑R，表现出显著的性能提升。

**⚠️ 局限性**

局限性包括：推理过程需多次工具调用，导致延迟和计算成本高；检索不准或关键信号弱时验证与仲裁失效；整体表现仍受底层视觉‑语言模型能力的限制。

---

## 251. RAMSES: Secure high-performance computing for sensitive data

**arXiv ID:** 2606.27919 | [PDF](https://arxiv.org/pdf/2606.27919v1)

**作者:** Peter Heger `[一作]` (IT Center Cologne), Viktor Achter `[通讯]` (IT Center Cologne)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了名为RAMSES的集成硬件加密与软件安全的高性能计算平台，支持数据在休眠、传输与使用中的端到端加密。

**💡 创新点**

首次将AMD SEV‑SNP硬件内存加密与IBM Storage Scale文件加密、Thales CipherTrust HSM统一到单一HPC体系中，并通过MFA、OS硬化实现可管理的可信研究环境。

**🔧 技术方法**

采用AMD EPYC Genoa的SEV‑SNP内存加密、GPFS文件系统加密、Thales CipherTrust HSM与KMIP、SR‑IOV虚拟化、Slurm SPANK插件、Cisco Duo MFA和Red Hat Linux 9.6的安全镜像。

**📊 数据集**

在生物信息工作流上使用 Tinamus major 基因组（RepeatMasker）和人类 GRCh38（BWA‑MEM2）测序数据作为基准。

**📈 对比分析**

通过在七种加密配置下重复六次测算，发现完整加密方案相较裸机产生4.4–18倍的运行时间增长，其中约一半来自虚拟化，文件加密几乎无额外开销。

**⚠️ 局限性**

主要局限在虚拟化和内存加密导致的性能损失、对AMD平台的依赖以及GPU可信计算尚未实现。

---

## 252. Lifted Causal Inference

**arXiv ID:** 2606.28024 | [PDF](https://arxiv.org/pdf/2606.28024v1)

**作者:** Malte Luttermann `[一作]` (University of Hamburg), Marcel Gehrke `[通讯]` (University of Hamburg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在关系域上进行因果推断的提升方法，构建了可承载因果知识的参数化因果因子图（pcfg）与其扩展的部分导向参数化因果因子图（pdpcfg），并分别给出了在这些模型上计算因果效应的 LCI 与 ELCI 算法。

**💡 创新点**

创新点包括：
1) 将提升推理（lifting）与因果模型结合，首次在关系域上实现高效的因果效应计算；
2) 引入 pcfg 定义参数化因果因子图并给出其干预语义；
3) 提出 LCI 算法，能够在保持精确答案的前提下避免大量离散化；
4) 扩展至 pdpcfg，处理部分已知的因果关系，并设计 ELCI 算法枚举所有一致扩展来得到可能的干预结果；
5) 在理论与实验层面统一了全导向与部分导向提升因果模型的视角。

**🔧 技术方法**

技术方法包括：
- 概率图模型与因果因子图（CFG）
- 参数化因果因子图（pcfg）与部分导向 pcfg（pdpcfg）
- 提升推理技术（lve、ljt）
- Bayes‑Ball 与路径阻断判定
- 受限的分解（parfactor splitting）与干预语义实现
- 一致扩展枚举与克利克（clique）约束
- 截断乘积公式（g‑formula）用于干预分布的计算。

**📊 数据集**

实验使用的是一个简化的员工关系示例（Alice、Bob、Charlie），通过改变员工数 d ∈ {8, 16, …, 4096} 来模拟不同的域规模。没有使用公开真实数据集，而是基于该合成模型进行跑时比较。

**📈 对比分析**

比较方法：在同一合成模型下，分别在 pcfg（使用 LCI + lve）、CFG（使用 VE）和 CBN（使用 VE）上计算同一干预查询的运行时。结果显示：
- CFG 与 CBN 的运行时随域规模呈指数增长；
- LCI（pcfg）在对数坐标下几乎线性增长，表明提升推理显著降低了规模依赖性；
- 因此，Lci 在大域规模下具有显著的性能优势。

**⚠️ 局限性**

局限性：
1) ELCI 的实验评估缺失，仅给出理论复杂度分析；
2) 对图结构在所有实例上相同的假设限制了对更一般关系域的适用性；
3) LCI 需要每个因子表示条件概率分布；
4) 当干预变量有未知父节点时，仍需枚举所有可能的父集合，若父集大则复杂度回升；
5) 需要满足全因果马尔可夫性和因果马尔可夫性，实际应用中可能难以验证。

---

## 253. HumanMoveVQA: Can Video MLLMs reason about human movement in videos?

**arXiv ID:** 2606.27999 | [PDF](https://arxiv.org/pdf/2606.27999v1)

**作者:** Pulkit Gera `[一作]` (University of Surrey), Armin Mustafa `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套评估视频多模态大型语言模型（VideoMLLMs）在全局人体运动轨迹与方向推理能力的基准（MotionReasonBench），并通过从视频中提取世界坐标系下的3D姿态轨迹生成结构化的多项选择问答；

**💡 创新点**

创新点包括：①首个基于第一帧锚定世界坐标系的全局轨迹与方向推理基准；②利用可扩展的多阶段管道将2D视频姿态恢复为一致的3D轨迹并离散化为“Spatial Codes”，从而实现可验证的结构化问答生成；③展示了通过针对该基准的有监督微调，能够显著提升模型的轨迹推理能力；

**🔧 技术方法**

采用的技术包括：SMPL-X人类姿态恢复、世界坐标系轨迹提取、空间离散化编码、基于模板的多项选择问答生成、BLIP-2生成身份描述、以及对多模态大语言模型（如Qwen3-VL 8B、Gemini-3-Flash、GPT‑4o等）的零样本评估与有监督微调；

**📊 数据集**

使用了三大人类动作数据集：EMDB（长时序单人视频）、RICH（多视角短视频）和EgoBody（多视角双人视频），并通过统一管道生成10,203条问答对；

**📈 对比分析**

方法通过与现有模型（Gemini-3-Flash、GPT‑4o、Qwen3‑VL 8B等）在七个推理类别上进行对比，发现零样本表现普遍不佳，尤其在数值、顺序和轨迹推理上低于随机；而在基准上进行有监督微调后，Qwen3‑VL 8B的整体得分从12.8提升至37.9（约三倍），显著弥合能力差距；

**⚠️ 局限性**

局限性包括：依赖于单目视频中的高质量姿态追踪，对细微旋转和短时事件的噪声敏感；未覆盖多人人体交互与多主体运动；并且对排序推理的效果仍不理想，需进一步方法改进。

---

## 254. AdvancedShelLM: A Stateful Multi-Agent LLM Honeypot for SSH Deception

**arXiv ID:** 2606.27990 | [PDF](https://arxiv.org/pdf/2606.27990v1)

**作者:** Muris Sladić `[一作]` (Czech Technical University in Prague), Sebastian Garcia `[通讯]` (Czech Technical University in Prague)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一种基于多模 LLM 的 SSH 蜜罐 AdvancedShelLM，使用 Worker–Manager 两个代理实现交互生成与验证分离，并保持文件系统与历史命令的显式一致性。

**💡 创新点**

创新点包括：多代理 LLM 架构、JSON 结构化文件系统、时间同步与命令历史双向同步、以及多 LLM 组合以提升生成质量与可验证性。

**🔧 技术方法**

技术方案涵盖 GPT‑OSS‑120B 与 GPT‑5.1 大语言模型、JSON 文件系统编码、Prompt‑Engineering、Manager 审核循环、单元测试框架、ARACNE 自适应 AI 攻击器、以及真实网络部署与日志分析。

**📊 数据集**

使用的数据集包括：扩展后的 34 条单元测试（从原始 12 条扩展而来）、30 次 ARACNE 评估、26 名人类参与者实验、以及 11 台 DigitalOcean 服务器产生的 148,850 次真实 SSH 会话。

**📈 对比分析**

与原 shelLM、Cowrie 以及真实 Ubuntu 系统对比，单元测试通过率提升至 99% 以上，AI 攻击器对 AdvancedShelLM 的识别率约为 45%（与 Cowrie 相近），人类攻击者的误判率约 71%（与 Ubuntu 相同），在真实网络部署中发现约 22 条提示信息被后续攻击者直接或间接复用。

**⚠️ 局限性**

主要限制包括：响应延迟较高、偶尔出现文件系统或历史状态不一致、人工评估样本量不足、AI 攻击器评估尚处于初级阶段，需进一步降低延迟、提升长会话一致性以及扩大评估规模。

---

## 255. ToxiREX: A Dataset on Toxic REasoning in ConteXt

**arXiv ID:** 2606.27981 | [PDF](https://arxiv.org/pdf/2606.27981v1)

**作者:** Stefan F. Schouten `[一作]` (Vrije Universiteit Amsterdam), Piek Vossen `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多语言、包含对话上下文且基于“毒性推理”模式的 Reddit 评论数据集 ToxiREX；

**💡 创新点**

提出并验证了一套结构化的毒性推理 schema，能够系统化地描述隐式与上下文相关的有害言论；

**🔧 技术方法**

采用了 GPT‑4o 的结构化输出进行银标注，并对 XLM‑RoBERTa 进行 fine‑tune 作为基线模型；

**📊 数据集**

训练集约 125k 条评论（6 种语言），测试集约 3k 条评论由母语者手工标注；

**📈 对比分析**

通过 F1、Precision、Recall 等指标对比 GPT‑4o 和 XLM‑RoBERTa，发现 GPT‑4o 在零样本下性能优于前者，但两者在推理准确性、跨度识别及态度评分上仍低于基准；

**⚠️ 局限性**

主要限制在于模型难以完整、精准地匹配人类注释的推理结构，尤其在多义性、情境依赖与态度刻画上存在显著误差，且目前评测仍受多重注释不一致的影响。

---

## 256. ProMSA:Progressive Multimodal Search Agents for Knowledge-Based Visual Question Answering

**arXiv ID:** 2606.27974 | [PDF](https://arxiv.org/pdf/2606.27974v1)

**作者:** ZhengXian Wu `[一作]` (Tsinghua University), Haoqian Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 ProMSA 的进阶多模态检索代理，能够在知识驱动视觉问答中根据当前证据与预算动态切换图像检索、文本检索或直接给答案。

**💡 创新点**

创新点在于：①将检索与推理嵌入同一端到端生成策略，支持多轮交互与去重；②在 RL 训练中引入工具交互深度归一化的 TN‑GSPO，使更新更稳定；③通过拒绝抽样 SFT 预训练提升工具调用可靠性。

**🔧 技术方法**

技术主要包括：多模态大型语言模型（Qwen‑VL 系列）、图像检索工具（EVA‑CLIP）、文本检索工具（BGE）、摘要生成（Qwen‑3‑VL‑8B）、SFT 与强化学习（veRL、LLaMA‑Factory），以及工具调用的预算约束与去重机制。

**📊 数据集**

使用的公开数据集包括：Encyclopedic‑VQA（E‑VQA）和 InfoSeek；在 OK‑VQA 上也进行了跨域验证。

**📈 对比分析**

与零样本 MLLMs、现有搜索代理和固定 RAG 方案相比，ProMSA 在 E‑VQA 和 InfoSeek 上均取得显著提升（单模/多模单跳/多跳平均准确率提升约 5–10% 甚至 13%），并在 OK‑VQA 上获得 82.7/85.6 的最高分。

**⚠️ 局限性**

主要局限包括：依赖外部检索工具和预先设定的预算，过大/过小的工具调用预算或 Top‑k 可能导致噪声累积；在极端长尾实体或视角差异极大的场景下检索失败仍可能导致误检；计算与推理成本相比某些基线仍略高。

---

## 257. Reasoning Beyond Prediction: From Data-Driven to Causal Software Engineering

**arXiv ID:** 2606.27960 | [PDF](https://arxiv.org/pdf/2606.27960v1)

**作者:** Roberto Pietrantuono `[一作]` (University of Naples Federico II), Stefano Russo `[通讯]` (University of Naples Federico II)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并论证了“因果软件工程（Causal Software Engineering）”框架，主张利用因果模型与推理来提升软件工程的决策与安全性。

**💡 创新点**

创新点在于将因果推断嵌入软件工程全过程，提供可解释的因果图模型，突破传统数据驱动的相关性限制，实现主动干预与“what‑if”分析。

**🔧 技术方法**

核心技术包括因果图（结构因果模型与因果贝叶斯网络）、因果结构学习（数据驱动与专家知识结合）、因果推断（干预与反事实）以及LLM辅助模型构建与推理。

**📊 数据集**

使用了多种真实系统日志、微服务监控数据、自动驾驶仿真数据等非公开数据集进行模型学习与验证。

**📈 对比分析**

对比方法主要是传统相关性机器学习/LLM生成的预测与因果方法的干预分析，实验结果表明因果方法能更准确地识别根因并生成针对性测试，但文中未给出具体性能数值。

**⚠️ 局限性**

主要局限包括对因果专业知识与统计方法的依赖、数据质量与可观测性不足导致模型不稳、计算开销大、与现有工具链集成挑战以及工业采用门槛高。

---

## 258. AI Persuasive Framing in Collective Dilemmas

**arXiv ID:** 2606.27951 | [PDF](https://arxiv.org/pdf/2606.27951v1)

**作者:** Anders Giovanni Møller `[一作]` (IT University of Copenhagen), Luca Maria Aiello `[通讯]` (IT University of Copenhagen)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在一项大型实验中，研究人员让1283名参与者在5轮集体风险游戏中玩耍，并测试AI助手的说服性框架对其合作贡献的影响。

**💡 创新点**

创新点在于结合大语言模型驱动的个性化说服与公共物品游戏，发现利他与自利说服效果不对称，并首次系统评估AI说服在短期与长期合作中的差异。

**🔧 技术方法**

使用的技术包括基于LLM的实时文本对话、基于社会价值取向(SVO)的个性化提示、统计检验（Mann‑Whitney U、Fisher exact）和OLS回归分析。

**📊 数据集**

数据来源为Prolific招募的1283名参与者在307场游戏中的贡献记录、对话日志及SVO评估，构成自建实验数据集。

**📈 对比分析**

通过与对照组比较，采用Mann‑Whitney U检验贡献、Fisher exact检验成功率；结果显示个性化AI在第一轮将平均贡献从6.30提升至6.57，成功率从3.75%提升至4.29%，但后续轮次效果迅速衰减；自利AI的负面影响则更持久。

**⚠️ 局限性**

局限性包括仅研究一种公共物品游戏、固定的组规模和回合数、频繁干预可能导致疲劳、个性化仅基于粗略的SVO分类、缺乏跨文化与长期验证。

---

## 259. Mixed-Precision For Energy Efficient Computations

**arXiv ID:** 2606.27949 | [PDF](https://arxiv.org/pdf/2606.27949v1)

**作者:** Gülçin Gedik `[一作]` (Université Paris-Saclay, UVSQ), Roman Iakymchuk `[通讯]` (Umeå University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文利用混合精度技术优化Reactor Simulator与LULESH两种显式求解器的计算性能与能耗，实现约30%时间与能耗提升。

**💡 创新点**

创新点在于基于Verificarlo工具的数值热点分析与分阶段精度降低策略，兼顾高效与准确。

**🔧 技术方法**

采用Verificarlo的VPREC与MCA后端、SLURM能源计量插件、HPE Cray PM计数器，以及混合精度实现与性能监测。

**📊 数据集**

使用Reactor Simulator与LULESH（Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics）代理应用。

**📈 对比分析**

与全双精度基线相比，在LUMI系统上实现时间减少约30%且能耗降低至约25%（Reactor Simulator）或能耗降低约30%（LULESH），显示显著效能提升。

**⚠️ 局限性**

局限在于搜索空间仍需手工探索，且方法难以完全自动化，未来需扩展到更广泛的科学应用。

---

## 260. An LLM-Powered Semantic Alignment Framework for Journal Recommendation

**arXiv ID:** 2606.27930 | [PDF](https://arxiv.org/pdf/2606.27930v1)

**作者:** Yanglin Yan `[一作]` (Central University of Finance and Economics), Hansheng Wang `[通讯]` (Peking University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于大型语言模型的语义匹配框架，用于期刊推荐。

**💡 创新点**

零训练、无手工特征、通过自然语言提示直接对齐稿件内容与期刊范围，且可生成可解释的推理。

**🔧 技术方法**

使用DeepSeek‑V3 LLM + prompt‑engineering 进行语义匹配与排序。

**📊 数据集**

基于Web of Science 23,609篇统计学/相关领域文章，49个候选期刊的标题、摘要、关键词（+参考文献）构成数据集。

**📈 对比分析**

与TF‑IDF、BM25、SBERT等传统方法对比，Top‑3/5/10准确率分别为40.23%/53.67%/70.05%，在无监督设定下与已有方法竞争或优于之。

**⚠️ 局限性**

候选期刊数量有限、仅评估单一LLM、未考虑作者偏好、开放获取、审稿周期等实际决策因素。

---

## 261. When Multi-Robot Systems Meet Agentic AI:Towards Embodied Collective Intelligence

**arXiv ID:** 2606.27929 | [PDF](https://arxiv.org/pdf/2606.27929v1)

**作者:** Yuxuan Yan `[一作]` (Zhejiang University), Qianqian Yang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了“具身集体智能（Embodied Collective Intelligence, ECI）”框架，强调机器人团队需要共享具身代理的内部状态（世界记忆、任务进度、技能经验），并用导航案例演示共享世界记忆的继承效果。

**💡 创新点**

创新点在于把传统多机器人共享的地图/任务分配、学习经验等提升到共享“agent loop”状态的层面，提出Co‑Perception、Co‑Action、Co‑Evolution三种共享资源模型，使新加入机器人可直接利用团队历史记忆而无需从零开始探索。

**🔧 技术方法**

技术手段包括：基于大型语言模型（Qwen3.5‑Plus）与视觉语言模型的代理推理；构建可查询的语义索引；利用 Uni‑NaVid 进行自由探索；通过在 Habitat–Matterport 3D 场景中记录 RGB‑D、姿态与检测结果实现记忆聚合。

**📊 数据集**

使用的数据集为 Habitat–Matterport 3D（4 个室内场景，166 个目标对象导航任务）。

**📈 对比分析**

与单机器人或无记忆机器人比较：在文字查询导航下，机器人 A/B 约 54–58% 的成功率；机器人 C（无历史）仅 24.1%；继承团队合并记忆的机器人 D 成功率提升至 77.1%（文字）/82.5%（图像），SPL 分别为 0.757/0.809，显著优于其他设置。

**⚠️ 局限性**

局限性包括：仅验证了世界记忆共享效果，未评估 Co‑Action 与 Co‑Evolution；共享记忆的时效性、冗余与跨异构机器人可迁移性尚未解决；通信开销、信息老化、责任边界与安全约束等实际部署挑战仍待攻克。

---

## 262. RelBall: Relation Ball with Quaternion Rotation for Knowledge Graph Completion

**arXiv ID:** 2606.27967 | [PDF](https://arxiv.org/pdf/2606.27967v1)

**作者:** Yike Liu `[一作]` (South China Normal University), Huiling Zhu `[通讯]` (South China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出RelBall模型，用四元数旋转、模尺度变换和关系球三者结合实现知识图谱嵌入。

**💡 创新点**

创新点在于①通过模尺度变换捕捉语义层级关系；②引入尾中心关系球可自然处理1‑1、1‑N、N‑1、N‑N等复杂映射；③保持三维旋转避免欧拉角万向锁。

**🔧 技术方法**

采用四元数乘法实现三维旋转，模尺度s_r和半径因子ρ_r控制模长与球半径；使用负采样和自对抗训练优化；实现基于PyTorch。

**📊 数据集**

在WN18RR与FB15k-237两个标准基准上进行实验。

**📈 对比分析**

与RotatE、QuatE、Rotate3D、HAKE、HA‑RotatE、MuRP、HBE、RotatH等几何变换与层级感知模型对比；在两数据集上均位列首位，MRR/Hits@k均略优于所有基线。

**⚠️ 局限性**

仍需进一步验证在大规模、稀疏或动态知识图谱上的可扩展性与鲁棒性，且模尺度与球半径的超参选择对性能影响较大。

---

## 263. Every Step of the Way: Video-based Parkinsonian Turning Step Counting

**arXiv ID:** 2606.27918 | [PDF](https://arxiv.org/pdf/2606.27918v1)

**作者:** Qiushuo Cheng `[一作]` (University of Bristol), Majid Mirmehdi `[通讯]` (University of Bristol)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种被动视频基的框架，利用3D人体网格与光流通过交叉注意力融合，再用多实例学习（MIL）对视频片段进行加权聚合，最终精确计数帕金森患者在转弯动作中的步数。

**💡 创新点**

①使用交叉注意力让粗糙的网格运动作为查询，检索细粒度光流信息；②通过MIL自适应聚合不同片段的运动嵌入，实现残差校正；③在无需可穿戴设备的环境下完成转弯步数计数。

**🔧 技术方法**

3D网格恢复（PromptHMR）、光流提取（FlowSeek）、ResNet-TSM特征编码、交叉注意力编码器、MIL聚合与对比正则化。

**📊 数据集**

PD-FOG（临床在场所转弯视频）和Turn-REMAP（家庭自由生活转弯视频）两大真实PD转弯数据集。

**📈 对比分析**

与IMU、姿态分析和重复活动计数等方法对比；在PD-FOG上MAE从37.7降至15.9，ACC_80提升至0.962；在Turn-REMAP上MAE为0.625，ACC_95为0.231，整体优于现有视觉和IMU方法。

**⚠️ 局限性**

仅适用于轻中度PD；对视频质量和光流估计敏感；极短步数或异常转弯模式下精度下降；仅针对转弯动作，未覆盖其他运动类型。

---

## 264. Performance Analysis and Optimal Design of ORB-Type GRAND Algorithms

**arXiv ID:** 2606.28030 | [PDF](https://arxiv.org/pdf/2606.28030v1)

**作者:** Li Wan `[一作]` (University of Science and Technology of China), Wenyi Zhang `[通讯]` (University of Science and Technology of China)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种以平均猜测后验（AGP）为核心的理论框架，用来分析和优化基于可靠性排序的ORB‑GRAND算法的错误率和测试次数，进一步设计了基于AGP的RS‑ORBGRAND重排方案。

**💡 创新点**

创新点在于将AGP作为唯一可预估的指标，证明在随机码族中按AGP递减排序能同时最小化BLER与平均测试数；对固定线性码提出了包含高阶重量关系的BLER表达式，并给出可计算的一阶上界；利用这些理论设计RS‑ORBGRAND并在BCH(127,113)上实现接近MLD极限的性能。

**🔧 技术方法**

使用的主要技术包括：离散随机过程与大数定律的误差分析、AGP的闭式与Monte‑Carlo积分估计、线性码的高阶重量分布计数与容斥原理、AGP重排的离线预估方法。

**📊 数据集**

主要数据集包括：随机码族（如N=127,K=113的随机码）以及典型的BCH(127,113)线性块码；实验中亦验证了Hamming(7,4)等短码。

**📈 对比分析**

与传统ORBGRAND、SGRAND等排序GRAND方案进行仿真对比，RS‑ORBGRAND在BCH(127,113)上仅落后约0.1 dB于MLD下限，且在10⁻⁶ BLER时已接近MLD性能。

**⚠️ 局限性**

局限性包括：需要离线计算AGP，导致预处理复杂度较高；对较大测试预算T时所需的AGP积分计算与模拟误差会显著；理论分析主要针对输出对称信道与线性码，对非对称信道或非线性码的推广尚未充分探讨。

---

## 265. EMOSH: Expressive Motion and Shape Disentanglement for Human Animation

**arXiv ID:** 2606.28026 | [PDF](https://arxiv.org/pdf/2606.28026v1)

**作者:** Dongbin Zhang `[一作]` (Tsinghua University), Haoqian Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了EMOSH框架，实现高保真、可控的人的视频动画，并在保持体形不泄露的前提下实现表情与动作的精准驱动。

**💡 创新点**

核心创新包括：①基于Expressive Human Model (EHM) 的形体与姿态显式分离，消除运动-体形耦合；②自适应的Confidence‑Aware Motion Tracker 能精准捕捉面部、手部与身体运动；③粗细结合的Hybrid Motion Injection 结合低频网格信息与高频关键点，提升细节控制；④Spatially‑Aligned Conditioning 缩小训练与推理域差距，提升身份一致性。

**🔧 技术方法**

技术上使用 3D 参量模型 (EHM)、Diffusion Transformer (DiT) 与 3D Causal VAE 的推理网络，结合可微渲染、关键点估计与联合优化，整体框架基于Wan2.1‑I2V 的 diffusion‑based 视频生成。

**📊 数据集**

训练数据集为约90万条视频，主要来源于公开网络人类动作视频与 Speaker‑Vid，辅以 VFHQ 用于面部动画；评测在 EchoMimicV2、TikTok 以及自制数据集上进行。

**📈 对比分析**

与 Wan‑Animate、HyperMotion、MimicMotion、UniAnimate‑DiT 等多种 SOTA 方法对比，EMOSH 在自驱动场景下 PSNR、SSIM 等指标均居首；在跨驱动场景下身份保持得分 IPS 最高，且用户评测显示显著优于基线。

**⚠️ 局限性**

局限性包括：对极端大幅度或遮挡场景的运动追踪仍有误差；长视频推理仍受 GPU 内存限制，需分块生成；当前缺少对背景动态与服装细节的显式建模。

---

## 266. From Detection to Action: Using LLM Agents for Fault-Tolerant Control

**arXiv ID:** 2606.28011 | [PDF](https://arxiv.org/pdf/2606.28011v1)

**作者:** Javal Vyas `[一作]` (Imperial College London), Mehmet Mercangöz `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一套基于大型语言模型（LLM）代理的主动容错控制框架，将故障检测结果转化为约束感知、可验证的恢复动作，并在数字孪生环境中先行仿真验证后执行。

**💡 创新点**

创新点：① 将多代理架构与Graph RAG结合，实现对工厂专有知识的关系式、多跳检索与LLM推理的无缝对接；② 在代理层引入验证与再提示机制，保证在仿真验证通过后才将动作送至现场；③ 通过语义化知识图谱统一结构、功能、行为与故障信息，为LLM提供可解释的语境。

**🔧 技术方法**

技术：大语言模型（GPT‑4o‑mini、GPT‑4.1‑mini）+ 代理式控制（监控、规划、动作、仿真、验证、再提示）；Graph RAG + CPSMod 语义对齐本体；数字工厂孪生（数据流、模型、仿真服务）；SPARQL + RDF/OWL 本体映射；规则与模拟验证（交互式仿真、约束检查）。

**📊 数据集**

数据集：两套仿真案例——Festo混合模块（离散序列流程）和连续搅拌罐反应器（CSTR），每个案例在不同故障场景下分别执行多次（n=10或n=60）以评估框架性能。

**📈 对比分析**

比较方法：对同一故障场景下的两款轻量化 LLM 进行对比，采用成功率、恢复率、规划与动作正确率等指标。结果显示 GPT‑4.1‑mini 在所有场景下实现近 100% 成功率；GPT‑4o‑mini 在某些故障（如泵降级、传感器失效）恢复率低至 33%–60%，但整体成功率约 80%。计划与动作的正确率在 GPT‑4.1‑mini 上保持 100%，GPT‑4o‑mini 仅在规划或动作上偶有误差。

**⚠️ 局限性**

局限性：① 知识图谱构建需要大量人工/结构化数据，工业环境中往往为非结构化；② 仅在已预先建模的故障条件下可验证，对未知或组合故障缺乏应对；③ 受限于 LLM 生成的延迟，实时硬实时控制场景不适用；④ 大型工厂规模、模型不匹配、传感器噪声等未在实验中验证；⑤ 工业部署需考虑边缘/本地化模型、网络安全、审计日志等非技术因素。

---

## 267. Ghost Without Shell: Measuring Non-Interactive SSH Attacks on Honeypots

**arXiv ID:** 2606.28006 | [PDF](https://arxiv.org/pdf/2606.28006v1)

**作者:** Veronica Valeros `[一作]` (Czech Technical University), Sebastian Garcia `[通讯]` (Czech Technical University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

部署了11台基于LLM的SSH Honeypot，持续15天收集攻击流量，并将其中的交互式、非交互式及文件传输会话进行统计与分析。

**💡 创新点**

首次系统量化SSH攻击中非交互式会话占比，发现超过99%为单一命令检查，并揭示了传统交互式衡量方法的局限性，提出应以命令正确性和验证通过率为评估指标。

**🔧 技术方法**

使用了AdvancedShelLM高交互SSH Honeypot（集成本地LLM和OpenAI gpt-5-nano等模型）以及内部会话类型识别机制，并采用专家启发式映射将独立Cowrie日志归类为三种模式。

**📊 数据集**

自身收集的11台Honeypot在15天内的SSH攻击数据；以及与之同期的CZ.NIC运营的4,737台Cowrie Honeypot组成的外部验证数据集（HaaS）。

**📈 对比分析**

通过统计两套数据集中非交互式会话的比例并绘制柱状/折线图进行对比，结果显示两者均呈现≈99%的非交互式比例，验证了实验结论的稳健性；传统交互式评估指标被证明不具代表性。

**⚠️ 局限性**

仅在云服务器、接受任意凭证的环境中实验，未支持文件传输，且honeypot主动拒绝传输请求可能进一步夸大非交互式比例；对特定目标系统的交互式攻击行为未被捕获。

---

## 268. Dual-Learning based Penalized Multi-Align Clustering for Multi-View Incomplete and Disorderly Data

**arXiv ID:** 2606.27984 | [PDF](https://arxiv.org/pdf/2606.27984v1)

**作者:** Liang Zhao `[一作]` (Dalian University of Technology), Qingchen Zhang `[通讯]` (Hainan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了双学习惩罚多对多聚类模型（DLPMAC），通过样本级对齐与多模态融合，解决了多模态数据的缺失与失配问题。

**💡 创新点**

创新点在于：①双学习机制同时捕捉语义一致性与结构相似性；②惩罚多对多对齐模块允许单样本与多样本配对，防止聚类现象；③从采样视角进行对齐，显著提升样本级对齐精度。

**🔧 技术方法**

使用了编码器-解码器网络、Hungarian算法、惩罚机制、噪声对比学习以及软对齐和多对多对齐技术。

**📊 数据集**

评估使用了16个多模态数据集（如3Sources、BBCsports、Caltech101、BDGP、Handwritten、Movies、flowers17、Prokaryotic、yale_mtv、Reuters_dim10、ORL、MSRCv1、20NewsGroups、BBC4、ALOI、Wikipedia-test）以及6个聚类数据集。

**📈 对比分析**

与PVC、MvCLN、SMILE、EGPVC、DGPPVC等六种最新方法对比，DLPMAC在所有16个数据集上实现最高样本级对齐准确率，在聚类任务中在6个数据集上均取得ACC/NMI/F1最高分，提升幅度显著。

**⚠️ 局限性**

局限性包括：对极低对齐率或更大缺失率的鲁棒性未充分验证；模型在缺乏先验知识时的表现仍有提升空间；实验主要集中在缺失率0.5的场景，未探讨更极端情况。

---

## 269. Directing the World: Fast Autoregressive Video Generation with Compositional Human-Camera Control

**arXiv ID:** 2606.27964 | [PDF](https://arxiv.org/pdf/2606.27964v1)

**作者:** Haoyuan Wang `[一作]` (Institute of Artificial Intelligence, China Telecom), Xuelong Li `[通讯]` (Institute of Artificial Intelligence, China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于MMPL的自回归框架，能够在长时序视频生成中实现人类运动（SMPL）与相机轨迹的联合可控，保持时间稳定性与空间一致性；

**💡 创新点**

核心创新包括：①Fast‑Slow Memory训练策略，将稳定的长视频先验与可控模块分离；②t‑guided Dynamic Projection结合Motion‑CFG，使SMPL条件在扩散的不同阶段进行时序适配；③因果对齐的相机控制模块，兼容块级自回归生成；④两阶段解耦训练与同步收集的视频‑文本‑运动‑相机多模态数据集；

**🔧 技术方法**

采用MMPL自回归生成、SMPL 3D人体模型、VAE编码、Diffusion denoising、ControlNet式控制、t‑guided投影、Motion‑CFG、因果相机模块以及Fast‑Slow Memory学习策略；

**📊 数据集**

构建自研的约20M iStock 视频数据集，经过质量过滤、SMPL 与 SfM 运动与相机估计后，划分为运动中心（~20k）与相机中心（~30k）两子集，总计约50k高质量可控视频对，包含视频、文本、SMPL 运动、相机轨迹四模态；

**📈 对比分析**

在 APRIL‑AIGC/UltraVideo‑Long 数据集上，与 Uni3C、FunCamera、FunMotion、WanMove 等基线在单一或联合可控设置下对比，Ours 在整体得分、时间抖动、运动平滑、运动误差以及 ATE/RPE/RRE 等指标均表现优越；消融实验表明 Fast‑Slow Memory 与 t‑guiding Projection 两项技术显著提升整体性能；

**⚠️ 局限性**

受限于数据规模与多样性，难以充分泛化到更复杂的开放域场景；对细粒度动作（如手部、面部）控制仍不够精准，尤其在主体尺寸较小时效果受限。

---

## 270. RECAST: Model Reconstruction via Counterfactual-Aware Wasserstein Geometry under Limited Data

**arXiv ID:** 2606.27948 | [PDF](https://arxiv.org/pdf/2606.27948v1)

**作者:** Xuan Zhao `[一作]` (Forschungszentrum Jülich), Ira Assent `[通讯]` (Aarhus University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于 Wasserstein 几何的模型重构方法 RECAST，利用单侧 CF 在低查询场景下对黑盒模型进行重构，并提供公平性诊断工具。

**💡 创新点**

创新点在于构造 CF 兼容的 Wasserstein 置信集合，将 CF 视为软样本，使用 Wasserstein 重心作为类原型，解决决策边界偏移、过拟合和非可识别问题；同时提出基于原型的阈值不变公平诊断。

**🔧 技术方法**

采用 Wasserstein 最优传输与重心优化、分布式鲁棒优化、Sinkhorn 近似求解、阈值不变公平度量，并使用多种 CF 生成方法（MCCF、DiCE、ROAR）支持重构。

**📊 数据集**

实验使用四个真实二分类数据集：Adult Income、COMPAS、HELOC、California Housing。

**📈 对比分析**

与 SAMPLES、CCA、TRA 等基线在低查询量（100、25-400）下进行比较，采用 fidelity 作为评估指标；RECAST 在大多数实验中获得最高 fidelity，并在噪声、分布漂移及近阈值区间表现出更强鲁棒性。

**⚠️ 局限性**

局限性在于仅支持二分类、单侧 CF；在大规模数据上计算成本相对较高；未对多分类扩展，且无法恢复唯一的真实决策边界，只能得到可辨行为等价类。

---

## 271. VASAE: Naming SAE Dictionary Directions with Vocabulary-Aligned Anchoring

**arXiv ID:** 2606.27941 | [PDF](https://arxiv.org/pdf/2606.27941v1)

**作者:** Kairui Zhang `[一作]` (University of Bristol), Martha Lewis `[通讯]` (University of Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Vocabulary-Aligned Sparse Autoencoder (VASAE)，在训练过程中通过词嵌入的最近向量为SAE字典的每个特征赋予内在的词标签，并保持与标准SAE相近的重构质量。

**💡 创新点**

创新点在于软词对齐约束：在保持可学习字典的同时，使用词嵌入作为几何锚点，让特征自然靠近词向量，从而在训练阶段直接获得词标签，而非后置标注。

**🔧 技术方法**

采用Transformer残差流SAE架构、稀疏编码（k‑top‑k+ReLU）、词嵌入作为锚点、余弦相似度对齐损失以及多层实验比较。

**📊 数据集**

使用WikiText‑103的数据集提取GPT‑2‑small（12层、768维）和Llama‑3.1‑8B（32层、4096维）的后残差流激活作为训练与评估样本。

**📈 对比分析**

与标准SAE和硬绑定词嵌入的SAE进行对比，重构误差、方差解释、交叉熵损失与恢复率均与标准SAE相当；在GPT‑2‑small大多数层约90%特征满足0.8的对齐阈值，Llama‑3.1‑8B浅层也可达到92.8%对齐，深层对齐稳定性下降。

**⚠️ 局限性**

局限性包括仅在残差流SAE和两类模型上验证，深层或大型模型对齐不稳定；只使用输入嵌入作为锚点，对未绑定模型的输出嵌入探索不足；对齐强度与重构质量的权衡未完全系统化。

---

## 272. TempAct: Advancing Temporal Plausibility in Autoregressive Video Generation via Planner-Executor RL

**arXiv ID:** 2606.28016 | [PDF](https://arxiv.org/pdf/2606.28016v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 273. Two-Stage Fine-Tuning for Protein Sequence Generation with Targeted Amino-Acid Composition

**arXiv ID:** 2606.27939 | [PDF](https://arxiv.org/pdf/2606.27939v1)

**作者:** Violeta Basten-Romero `[一作]` (Barcelona Supercomputing Center), Víctor Guallar `[通讯]` (Barcelona Supercomputing Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一个两阶段的蛋白序列生成管线，使预训练语言模型能够在保持序列多样性与可合成性的前提下，生成满足指定氨基酸组成的蛋白序列，应用于合成饲料蛋白设计。

**💡 创新点**

创新点在于将领域自适应微调与奖励加权强化学习相结合，并设计了差异化的组成奖励函数，能够在不牺牲序列可行性的情况下精确对齐目标氨基酸分布。

**🔧 技术方法**

采用预训练ProtGPT2语言模型，进行领域自适应微调（FT），随后使用奖励加权强化学习（类似DPO/PPO）与软最大化的组成奖励进行微调。

**📊 数据集**

数据集为UniProtKB/TrEMBL的蛋白，先筛选长度100–500、与目标组成余弦相似度≥0.95、身份<70%的序列；目标组成为实验鸡饲料组成q_A和公开参考q_B。

**📈 对比分析**

与未微调ProtGPT2、仅FT、不同奖励基线比较，结果显示两阶段管线将Jensen–Shannon距离从0.247/0.202降至0.0044/0.0008，组成分数提升至0.822/0.830，且所有序列约束（必需氨基酸覆盖、互换池平衡、零目标合规）均满足；RL-only或单一微调表现显著逊色。

**⚠️ 局限性**

局限包括缺乏结构/功能实验验证、种子数量有限、仅评估氨基酸组成对齐且RL训练可能出现模式崩溃；此外仍需实验确认蛋白可消化性与生物活性。

---

## 274. Agentic AI-Powered Re-Identification: An Emerging, Scalable Threat to Mobility Microdata Privacy

**arXiv ID:** 2606.27936 | [PDF](https://arxiv.org/pdf/2606.27936v1)

**作者:** Oscar Thees `[一作]` (University of Applied Sciences and Arts Northwestern Switzerland), Matthias Templ `[通讯]` (University of Applied Sciences and Arts Northwestern Switzerland)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个基于大型语言模型（LLM）智能体的端到端自动化再识别流水线，能够仅凭移动轨迹和公开网络信息将匿名GPS数据与真实身份关联。

**💡 创新点**

创新点在于：①利用Agentic AI完成从坐标抽取、地址推断、身份候选排序到最终验证的全流程，完全无需人工干预；②通过多阶段质量门控保证推理链可追溯；③展示了极低成本（≈$2.24/案例）和短时效（≈17分钟）即可实现大规模再识别。

**🔧 技术方法**

技术手段包括：Claude LLM（Anthropic）作为主控 orchestrator；专用子智能体执行GPS分析、逆向地理编码、建筑登记查询、候选评分与验证；Web search/fetch工具、CSV/JSON日志记录；多阶段质量门控与不确定性账本。

**📊 数据集**

数据集：在瑞士收集的43名受试者同意提供的真实居住与工作地址，利用这些地址生成模拟的三周GPS轨迹；公开网络资源（搜索引擎、建筑登记、社交媒体、公司网站、公共住宅注册）用于身份推断。

**📈 对比分析**

性能评估：在43条轨迹中，流水线实现全再识别率41.9%，对可再识别的25例中成功恢复姓名的比例72%；精确度94.7%；平均API成本$2.24，平均运行时17分钟；与以往的人工Krumm等方法相比，效率提升数百倍、成本降低数十倍。

**⚠️ 局限性**

局限性：①未使用真实商业数据集，模拟轨迹缺乏多样性；②样本量小，且仅限瑞士，特定建筑登记依赖；③未进行人工对照基准；④仅使用公开信息，未涉及非公开或社交工程攻击；⑤单次运行，未评估结果稳定性；⑥模型安全过滤可能影响结果。

---

## 275. Benchmarking on Tasks That Matter: Dataset Selection for Preserving Model Rankings

**arXiv ID:** 2606.27997 | [PDF](https://arxiv.org/pdf/2606.27997v1)

**作者:** Rostislav Gusev `[一作]` (Applied AI Institute), Alexey Zaytsev `[通讯]` (Applied AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种统一的框架用于从大规模多数据集基准中选择子集，保证子集上的模型排名与完整基准的排名尽量一致；

**💡 创新点**

1) 引入bootstrap聚合评估排名保持度并给出置信区间；2) 对贪婪最远点（farthest‑first）方法给出理论误差上界；3) 将多领域（时间序列、推荐、NLP）与多种数据集表示相结合，展示了非随机选择在某些领域能显著提升排名保持。

**🔧 技术方法**

多种子集选择策略（随机、k‑means、A/D‑optimal、farthest‑first）+ 统计学评估（Bootstrap、Wilcoxon+Holm）、理论分析（GP+covering bound）+ 代表性数据集描述子（meta‑features、landmarking、BERT 描述）。

**📊 数据集**

时间序列分类 112 个数据集；推荐系统 30 个数据集；NLP（MTEB）57 个任务；以及合成 Oracle/Broken 表征的模拟实验。

**📈 对比分析**

通过 AUC(ρ)、AUC(MAE)、NDCG@5 等指标对 200 次试验的置信区间进行比较。结果表明：在 TSC 与 NLP 中，Cosine 或 Euclidean farthest‑first 及 k‑means 能在仅选 5~10 个数据集时达到 Spearman ρ≈0.90 或 MAE≈1.5；在 RecSys 中，随机选择往往与其他方法无显著差异。

**⚠️ 局限性**

表示子集选择对数据集特征质量高度依赖；当特征与模型性能无关时，几乎无优势；方法对大规模基准外的数据集可推广性尚待验证；理论证明依赖 GP 先验，实际任务中可能不完全符合。

---

## 276. Latent Visual Diffusion Reasoning with Monte Carlo Tree Search

**arXiv ID:** 2606.27988 | [PDF](https://arxiv.org/pdf/2606.27988v1)

**作者:** Xirui Teng `[一作]` (Beijing Jiaotong University), Junsong Yuan `[通讯]` (University at Buffalo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Latent Visual Diffusion Reasoning (LVDR) 框架，用于运动与外科视频的细粒度技能评估，结合潜在扩散模型与关键点引导的蒙特卡罗树搜索（MCTS）生成可解释的视觉推理轨迹；

**💡 创新点**

创新点在于把视觉推理视为潜在空间中的扩散过程，并通过关键点驱动的MCTS挖掘并可视化关键视觉序列，实现高精度评估与可解释性双赢；

**🔧 技术方法**

使用 Denoising Diffusion Implicit Models (DDIM) 的 Transformer 版本进行潜在扩散；采用3D关键点提取、注意力机制和基于UCT的MCTS；

**📊 数据集**

实验覆盖四个数据集：EgoExo4D（篮球、足球、攀岩）、JIGSAWS（手术任务）、FitnessAQA（健身动作）和 Cataract-101（白内障手术）；

**📈 对比分析**

与 MAGR、FineParser、LLaVA‑Video、GPT‑4o、Gemini 2.5 等方法对比，LVDR 在 Spearman ρ、R‑ℓ2、F1 等指标上均取得领先或接近 SOTA 的性能；

**⚠️ 局限性**

局限性包括对关键点标注的依赖、MCTS 计算量对实时性的影响、仅处理单人场景且跨域泛化验证不足。

---

## 277. From Black-Box to Clinical Insight: A Multi-Stage Explainable Framework for Speech-Based Cognitive Impairment Detection

**arXiv ID:** 2606.27973 | [PDF](https://arxiv.org/pdf/2606.27973v1)

**作者:** Yasaman Haghbin `[一作]` (Independent Researcher), Maryam Zolnoori `[通讯]` (Columbia University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究提出一种多阶段可解释框架，将Transformer预测结果转化为基于SHAP的词级归因、理论驱动的语言特征以及四阶段LLM推理，生成可供临床医生使用的语言学解读报告，用于识别认知障碍。

**💡 创新点**

创新点在于将SHAP层级聚合与手工语言特征相结合，并利用四阶段LLM序列化推理，将模型归因映射到临床可解释的认知‑语言维度，实现从黑盒到个体化解读的桥梁。

**🔧 技术方法**

使用的技术包括SpeechCARE Adaptive Gating Fusion多模态Transformer、SHAP层级归因、手工提取的词汇丰富度、句法复杂度、停顿频率和语义连贯度特征，以及基于LLaMA‑3.1‑70B‑Instruct的四阶段LLM推理。

**📊 数据集**

采用NIA PREPARE基准数据集（2058个参与者，包含英语、西班牙语、普通话，分为健康对照、轻度认知障碍、阿尔茨海默病），并在其测试集上评估。

**📈 对比分析**

通过与PREPARE挑战官方结果对比，模型在测试集上实现AUC 86.83%和F1 72.11%，在70例临床医生评估中报告与医生判断一致率98%，SUS评分82/100显示易用性高。

**⚠️ 局限性**

限制在于当前框架仅解释语言层面，尚未对声学Transformer的关注模式做临床可解释化，且跨语言推广仍待验证。

---

## 278. An Empirical Analysis of Factual Errors in Human-Written Text and its Application

**arXiv ID:** 2606.27959 | [PDF](https://arxiv.org/pdf/2606.27959v1)

**作者:** Kazuma Iwamoto `[一作]` (Nikkei Inc.), Shotaro Ishihara `[通讯]` (Nikkei Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于新闻更正稿对人类文本中的事实错误进行分类，并构建合成数据集，评估大型语言模型在文本层级和词层级的事实错误检测（FED）性能。

**💡 创新点**

创新点在于：① 提炼出人类写作中特有的事实错误分类体系；② 通过七类词替换方法生成可控合成错误数据；③ 系统性评估了LLM在无外部知识情境下的FED能力。

**🔧 技术方法**

主要技术包括：基于T5的实体候选生成与GiNZA命名实体识别、pykakasi与mozcpy进行汉字误转换、word2vec与LLM交叉验证反义词、正则表达式与数值变换、LLM过滤同义词噪声、QLoRA微调以及few‑shot提示。

**📊 数据集**

使用的数据集包括：① 234 篇日经（Nikkei）更正稿（真实错误）；② 约 28,000 条合成错误样本（来自 2015‑2019 年日经原稿）；③ 100 条每类测试样本以及 100 条无错误样本；④ 公开可复现的 Wikinews 日语文章样本。

**📈 对比分析**

与传统事实核查或修正任务不同，本文采用句级/词级检出框架，评估模型 F1：GPT‑5.4 仅 few‑shot 时句级 43.8% / 词级 52.0%；Qwen3‑Swallow‑8B‑QL 微调后词级 36.2%；在真实更正数据上，GPT‑5.4 仅 10.6%/16.9%，表明合成数据与实际情况差距仍大。

**⚠️ 局限性**

局限性包括：仅在日语环境下验证，合成数据仍含噪声且未利用外部知识库；使用的模型规模局限于 8B 参数；数据来源包含专有日经稿件，影响复现；few‑shot 仅 6 shot，未充分探索多 shot 情景。

---

## 279. Controllable Histopathology Image Synthesis with Training-free Structural Initialization and Textural Modulation

**arXiv ID:** 2606.27935 | [PDF](https://arxiv.org/pdf/2606.27935v1)

**作者:** Yuheng Qiu `[一作]` (Harbin Institute of Technology), Jianfeng Cao `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种无需训练、可控制的病理图像合成框架（CHIS），通过结构初始化和纹理调制引导预训练扩散模型生成与先验掩码一致的高质量图像；

**💡 创新点**

创新点在于：①将掩码的相位信息与高斯噪声在频域融合，形成结构对齐的起始状态；②利用多尺度小波分解在逆扩散过程中自适应调节粗细纹理，提升结构与纹理的一致性；

**🔧 技术方法**

采用预训练的Latent Diffusion Model、FFT相位融合、Stationary Wavelet Transform（SWT）纹理调制、以及细胞聚合的掩码生成；

**📊 数据集**

在MoNuSAC、Kumar和PanNuke三个公开病理图像数据集上进行实验；

**📈 对比分析**

与全监督NuDiff、SDM、GAN方法（SynDiff、CycleDiff）以及训练自由方法（UGDM、ADMMDiff）比较，CHIS在掩码-图像一致性（FS1、HD95）、下游实例分割（Dice、AJI）和图像质量（FID、IS）指标上均优于或逼近全监督模型，尤其在不需标注和训练的情况下表现突出；

**⚠️ 局限性**

局限性包括：仍需参考图像以确定纹理风格，掩码生成依赖手工设置的聚合参数，且在跨组织或不同染色方法的泛化性尚未充分验证。

---

## 280. Self-Verifying Measurement Records: Hash-Linked Evidence Graphs for Hardware Benchmarking

**arXiv ID:** 2606.27934 | [PDF](https://arxiv.org/pdf/2606.27934v1)

**作者:** Faruk Alpay `[一作]` (Bahçeşehir University), Baris Basaran `[通讯]` (Bahçeşehir University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于哈希链证据图的硬件性能测量记录，在记录中将每个数值与其观测、验证步骤绑定，支持离线可审计的完整档案。

**💡 创新点**

创新点包括：①将测量结果与计算内容通过 SHA‑256 哈希实现不可篡改的透明日志；②使用机型校准的残差阈值对线性量做概率性 Freivalds 身份检查；③对非线性量提供代数检查与可复现性分类；④跨设备差分验证与可重建的设备签名；⑤多阶段修复与重检的可追溯 transcript；⑥可在任意设备上重验并实现子线性查询。

**🔧 技术方法**

采用的技术：SHA‑256 内容寻址、Merkle 树、Freivalds 随机探针、浮点误差分析、随机校验、可复现性类划分、Fiat‑Shamir 挑战、跨设备差分、设备签名、离线审核脚本。

**📊 数据集**

使用的数据集：在 NVIDIA Blackwell 与 Hopper 加速器上跑的标准工作负载——稠密 GEMM、内存带宽流、注意力（softmax）、归约、原子散列/索引加法，尺寸 4k/8k/16k、精度 FP16、BF16、TF32、FP8；并在 RTX 5090、RTX PRO 6000、RTX 5090 等单卡上重放。

**📈 对比分析**

比较方法：利用同型号两卡的差分验证结果一致性；跨设备重放对比残差阈值、可复现性类与吞吐量；单卡在不同架构上重验检查。实验表明残差阈值保持一致，吞吐量差异可作为签名；重放恢复率高（≈92% 可转移），吞吐量不转移；离线重检在 CPU 上通过 FP16 产品验证，误差概率 ≤ 2⁻ᵏ。

**⚠️ 局限性**

局限性：仅覆盖线性量与已分解的非线性量；可复现性类给出的是范围而非精确值；校准依赖原始硬件；对物理攻击受限，需硬件根信任；重检需相同/兼容架构；大规模计算仍需线性审计成本；固定探针安全性需 Fiat‑Shamir 变动；设备签名是行为签名，非加密身份。

---

## 281. Verifiable Geometry Problem Solving: Solver-Driven Autoformalization and Theorem Proposing

**arXiv ID:** 2606.27926 | [PDF](https://arxiv.org/pdf/2606.27926v1)

**作者:** Can Li `[一作]` (Beijing Normal University), Hua Huang `[通讯]` (Beijing Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SD‑GPS，基于解算器的闭环框架，用于几何问题的自动形式化和推理。

**💡 创新点**

创新点在于将符号解算器作为执行 oracle，在自动形式化阶段采用可执行性奖励，在推理阶段引入可验证的定理提议代理，三者共同构成统一的解算器驱动循环。

**🔧 技术方法**

核心技术包括：QwenVL3‑2B 多模态语言模型的监督形式化适配、基于 GSPO 的可执行性引导强化学习、修复机制以及基于符号验证的局部定理提议模块。

**📊 数据集**

使用了 Geometry3K、PGPS9K 两大几何数据集，并在此基础上合成多样化样本，形成训练、验证、测试集。

**📈 对比分析**

与现有 MLLM、神经、神经‑符号基线对比，SD‑GPS 在 Geometry3K 上完成率 86.4%、选择率 90.4%，在 PGPS9K 上完成率 79.8%、选择率 84.5%，相较最强对手提升约 3–4 个百分点。

**⚠️ 局限性**

局限性：仅适用于平面几何、基于 Inter‑GPS 形式语言；依赖解算器规则覆盖，若缺少必要定理或搜索深度不足，仍可能失败；定理提议受限于符号验证器的判定。

---

## 282. Graph Dimensionality Reduction for Contextual Bandits: Structure-Specific Regret Bounds under Approximate Smoothness and Noisy Eigenspaces

**arXiv ID:** 2606.27917 | [PDF](https://arxiv.org/pdf/2606.27917v1)

**作者:** Joyanta Jyoti Mondal `[一作]` (University of Delaware), Anuj Sharma `[通讯]` (Iowa State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 GraphDR-LinUCB，通过将 arm 特征投影到图 Laplacian 的低频子空间后在 k 维空间内执行 LinUCB，降低探索维度；

**💡 创新点**

首次证明硬投影到图谱子空间在上下文 bandit 中能实现 k√T 的 regret，且高频残差的影响被细化为实测的残差杠杆和候选集残差宽度，避免了传统线性错配理论的线性惩罚；

**🔧 技术方法**

使用图 Laplacian 低频特征投影、LinUCB、Davis–Kahan 扰动分析、结构化残差理论、以及基于子空间捕获度的无阈值模型选择；

**📊 数据集**

六个真实图结构 bandit 数据集（MovieLens‑100k/1M、Amazon、LastFM、MIND‑small、ogbn‑arxiv）以及合成 SBM 与随机几何图；

**📈 对比分析**

与全维 LinUCB、PCA‑LinUCB、JL‑LinUCB、随机策略、SpectralUCB、Laplacian‑regularized LinUCB 等基线对比，GraphDR-LinUCB 在大多数数据集上显著降低累计 regret（高达 15×/69× 的提升），仅在图谱与奖励不对齐时失效；

**⚠️ 局限性**

仅在图谱与奖励高度不匹配（高频残差能量大但不在候选集内）时效果下降，且需要提前估计残差或使用基准子空间捕获度作为模型选择依据，未覆盖动态或非静态图谱情形。

---

## 283. DiStash: A Disaggregated Multi-Stash Transactional Key-Value Store

**arXiv ID:** 2606.27979 | [PDF](https://arxiv.org/pdf/2606.27979v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 284. The Signal-Coverage Matrix: Stratifying Type and Semantic Errors in Statement Autoformalization

**arXiv ID:** 2606.28013 | [PDF](https://arxiv.org/pdf/2606.28013v1)

**作者:** Chengxiao Dai `[一作]` (University of Sydney), Zhanhui Lin `[通讯]` (University of Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过引入信号覆盖矩阵（signal‑coverage matrix）对 Lean 自动形式化过程进行细粒度诊断，区分类型错误、语义错误、两者同时错误和成功四类，并通过 4×4 转移矩阵追踪方法改进时错误类型的迁移；

**💡 创新点**

创新点在于（1）提出了基于类型检查器和语义评估器交叉的 2×2 矩阵及其 4×4 转移分析，以揭示不同改进方法提升的真实机制；（2）构建了双评审协议（Claude Opus 与 GTED）来校准语义精确度，量化 elaborator 诱导的结构重写导致的判定差异；（3）证明三种不同的 elaborator‑反馈方法在类型错误恢复上可实现相同的 37.7% 率，并可用该率预测其他方法的提升。

**🔧 技术方法**

采用 Lean 4.19.0+Mathlib、GTED（通用树编辑距离）做语义评估，Claude Opus 4.7 做跨评审；利用 DeepSeek V4-Pro LLM 进行自动形式化，配合 Lean‑Retry、Sample‑Filter、Stratified Autoformalization (SAF) 三种反馈策略；引入 Typed JSON IR 与 deterministic translator；使用 30 行模板实现 IR→Lean；通过统计与回归分析验证预测。

**📊 数据集**

主要数据集包括 ProofNet^#（186 个本科教材题，类型复杂）和 MiniF2F-test（244 个奥数题，类型简单），并在三种 LLM 族（DeepSeek、Qwen3.5‑Plus、MiMo‑v2.5‑Pro）上进行交叉实验。

**📈 对比分析**

在 DeepSeek V4-Pro × ProofNet^# 上，Vanilla 方法的成功率为 59.7%；Lean‑Retry、Sample‑Filter 与 SAF 在三种评审下均提升至约 75–76%，即 +34 到 +36 个额外成功案例；TO→TS 的恢复率约为 23/61（约 37.7%）对所有三种方法一致；跨模型/数据集实验显示预测 ΔTS 与观测差距≤2个百分点，且 Δ 与 Vanilla elab‑fail 率呈线性关系（R²≈0.96）。

**⚠️ 局限性**

局限性包括：（1）语义评判仍依赖 LLM（Claude Opus），若出现错误可能误判；（2）对 Gold formalization 的审计不足，导致残留语义错误低估；（3）方法在 K=3 的预算下已饱和，需探索更高层次的信号（如语义相似度）来进一步提升；（4）实验聚焦于 Lean 生态，未验证在其他定理证明系统的通用性。

---

## 285. Curriculum-guided Change Detection Training: Toward Accurate Serac Fall Monitoring

**arXiv ID:** 2606.28012 | [PDF](https://arxiv.org/pdf/2606.28012v1)

**作者:** Arthur Dérédel `[一作]` (LIRIS), Laure Tougne Rodet `[通讯]` (LIRIS)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了基于课程学习的变化检测训练框架，利用样本难度指标逐步引入更难样本以提升鲁棒性。

**💡 创新点**

首次提出在变化检测中使用太阳角度差(SAG)与结构相似度SSIM两种难度度量，并设计课程学习策略与之配合。

**🔧 技术方法**

采用课程学习、SAG/SSIM难度度量、Siamese ViT+DPT以及RT-DETR架构，结合数据增强和动态采样技术。

**📊 数据集**

在新建的SeracFallDet冰川崩塌变化检测数据集上进行评估。

**📈 对比分析**

与均匀采样基线对比，使用IoU/F1等指标，课程学习尤其是baby step+SAG/SSIM可提升3–8% F1，显著降低伪变更。

**⚠️ 局限性**

SAG对阴影/云等大气条件敏感，SSIM受真实变化影响，两者单独使用仍存在误差；目前仅在全监督设置验证，未探索半监督或更复杂场景。

---

## 286. Dialogue to Detection: A Multimodal Hybrid NLP Pipeline for Insurance Fraud Detection

**arXiv ID:** 2606.28002 | [PDF](https://arxiv.org/pdf/2606.28002v1)

**作者:** Muhammad Shakeel Akram `[一作]` (Aston University), Karishma Jaitly `[通讯]` (Domestic & General)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个端到端的合成多模态FNOL（首次报案）数据生成与欺诈检测管线，覆盖文本生成、语音合成、ASR+分离、实体抽取、语义检索、声纹相似度与可解释风险评分；

**💡 创新点**

首次将对话生成、双声道语音合成、WhisperX ASR与分离、NER+Regex实体抽取、BERT-RAG检索、Resemblyzer声纹聚类以及规则融合四大模块集成，形成可解释的风险评分框架；

**🔧 技术方法**

使用GPT-2生成多样化文本、xTTS进行双声道语音合成、WhisperX实现STT与分离、RoBERTa NER+Regex抽取实体、BERT-RAG结合FAISS检索、Resemblyzer声纹编码与余弦相似度、规则权重融合；

**📊 数据集**

在无公开多模态保险数据的前提下，构建了包含500个平衡文本、250个两声道对话及语音的合成数据集，并在此基础上进行训练与评估；

**📈 对比分析**

在合成数据上，二分类模型实现 100% 准确率、召回率和 F1；语音识别WER 11.9%；声纹聚类 ARI 0.868；在未见的真实格式数据上仍保持 60–90% 召回率，精度略降，表明管线具备一定泛化能力；

**⚠️ 局限性**

受限于合成数据的多样性与真实噪声模拟不足，模型在真实保险数据上的验证尚未完成，泛化能力与鲁棒性仍需进一步提升；

---

## 287. Parallel Rollout Approximation for Pixel-Space Autoregressive Image Generation

**arXiv ID:** 2606.27978 | [PDF](https://arxiv.org/pdf/2606.27978v1)

**作者:** Jiayi Xu `[一作]` (Peking University), Guolin Ke `[通讯]` (DP Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种面向像素空间的自回归图像生成框架Parallel Rollout Approximation (PRA)，在不依赖预训练分词器的前提下直接生成原始像素补丁

**💡 创新点**

创新点在于：①用低维中间状态替代高维像素补丁的直接预测，降低单步错误；②通过相同的像素解码路径并行构造推理时的像素输入，近似自回归回滚，缓解训练-推理差距；③实现像素输入-像素输出的完整自回归接口，同时保持内部低维生成

**🔧 技术方法**

技术核心包括自回归Transformer、基于流的token级预测头、可学习的像素解码器、端到端学习的低维中间目标以及并行的推理式像素输入构造；同时采用噪声注入、masking、LPIPS+L1解码器损失、EMA与CFG等训练技巧

**📊 数据集**

在ImageNet-1K 256×256图像上进行类别条件生成实验

**📈 对比分析**

与两阶段生成、像素空间扩散和像素空间自回归等方法比较；PRA-S（135M）FID 2.58，PRA-L（511M）FID 1.94，分别优于之前最大的像素空间AR模型（FARmer 1.9B FID 3.60）并与扩散模型竞争；线性探针测试显示PRA-L顶级准确率68.8%，显著高于SphereAR-L和JiT-L

**⚠️ 局限性**

需要额外的内部中间状态、像素解码器和训练时的额外并行AR前向，虽然整体端到端，但在更广泛数据域和生成任务的泛化仍待验证

---

## 288. SHARD: cell-keyed residual splitting for alignment-resistant private dense retrieval

**arXiv ID:** 2606.27976 | [PDF](https://arxiv.org/pdf/2606.27976v1)

**作者:** Sergey Kurilenko `[一作]` `[通讯]` (Moscow Institute of Physics and Technology), Sergey Kurilenko (Moscow Institute of Physics and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Shard的细胞级别嵌入保护变换，将向量拆分为公共前缀和私有残差，并在残差上使用细胞专属正交钥匙，兼顾检索精度与对齐攻击抵抗。

**💡 创新点**

创新点在于：① 去除单一全局几何轴，改为多细胞钥匙化残差；② 通过短公共前缀实现粗粒度检索、细粒度完整重排；③ 细胞钥匙的使用使对齐成本随细胞数C呈线性增长，显著提升对齐和逆向攻击的抵抗；④ 引入微钥极限实现残差可重置和不可关联。

**🔧 技术方法**

技术包括：SVD子空间投影、Haar随机正交旋转、细胞分簇（k-means）与每细胞Householder反射钥匙、CKKS同态加密下的加密查询与内积重排、产品量化索引、以及对齐/逆向攻击的实验框架。

**📊 数据集**

使用的数据集涵盖五种编码器（multilingual-e5-small、e5-base、mpnet、e5-large、BGE-M3）在一百万条俄文维基文档上的缓存嵌入；在BEIR三大子集（SciFact、NFCorpus、ArguAna）与MIRACL两语种（斯瓦希里语、孟加拉语）上评估检索性能；同时用20k/100k文档样本评估PQ泄漏与参考语料库查询。

**📈 对比分析**

对比结果显示：Shard在nDCG@10上恢复与原始空间相同的检索质量（而基线SVD截断损失2–8分）；细胞数C越大，对齐攻击所需锚点数近似乘C；公共前缀泄漏比基线PQ低约30–50%；微钥极限下残差图泄漏完全消失；在线成本约7–30次加密残差查询，整体检索延迟保持在≈0.6 s。

**⚠️ 局限性**

局限性包括：① 对单细胞内的针对性攻击仍易被d_priv锚点破解；② 共享的公共前缀在存在重叠参考语料时仍能实现精确恢复；③ 仅提供几何防御，未对访问模式或完整加密提供安全保证；④ 对恶意服务器或PIR/ORAM的防护未做实现；⑤ 需要细胞划分与钥匙管理，部署复杂度高。

---

## 289. Building a Scalable, Reproducible, Evaluatable, and Closed-Loop Simulation Environment Foundation for Embodied Intelligence Cloud-Native Simulation Infrastructure for Embodied Intelligence Training, Evaluation, and Data Collection

**arXiv ID:** 2606.27962 | [PDF](https://arxiv.org/pdf/2606.27962v1)

**作者:** Junwu Xiong `[一作]` (AI Infra Team at JDT), Yince Gao `[通讯]` (AI Infra Team at JDT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向“具身智能”的云原生仿真基础设施，涵盖环境生成、任务执行、轨迹采集、模型评估、闭环优化以及统一服务化接口，旨在提升大规模数据产出、标准化评测与多团队协作效率。

**💡 创新点**

创新点包括：
1) 将仿真、训练、评测与数据治理整合为统一可调度的云平台；
2) 引入自动化场景与任务生成、随机扰动、风格增广（Sword）与预检滤波（Pre-VLA）等技术；
3) 架设多引擎（Isaac Sim、MuJoCo、SAPIEN 等）互通的运行层和统一任务协议；
4) 设计了闭环反馈机制（错误归因 → 定向数据增广 → 重新训练），实现数据、模型与环境的持续协同迭代；
5) 通过 D-VLA、RL-VLA3 等异步分布式框架提升大参数模型与高保真物理仿真并行效率。

**🔧 技术方法**

主要技术手段：
- 云原生容器化与弹性调度（Kubernetes/资源池）；
- 统一任务与模型接口协议；
- 多引擎适配层；
- 轨迹管道、数据质量控制与版本管理；
- 评测层的分层指标体系与标准化报告；
- Sword、Pre-VLA、D-VLA、RL-VLA3 等框架实现异步并发、动态调度与增广。

**📊 数据集**

使用的数据集与任务库：
- 虚拟任务集合（Isaac Sim 5.1、RLBench、ManiSkill、LIBERO、RoboCasa 等）；
- 真实机器人轨迹（Open X-Embodiment、DROID、BridgeData V2）；
- 生成式数据（Sword 视觉增广、Pre-VLA 预检过滤）。

**📈 对比分析**

比较方法：在同一云平台上运行多模型（VLA、RL-VLA、世界模型等），使用统一的 L1–L5 任务层级和标准化指标（成功率、子目标完成率、指令遵循度、碰撞率、资源占用等）对比；报告中提供模型版本回归、失败分布和资源消耗。性能方面：通过弹性调度和多引擎并行，预计吞吐量提升至单机的 5–10 倍；Pre-VLA 预检可将无效渲染步骤降低 20–30%，并提升任务成功率 6.8% 左右。

**⚠️ 局限性**

局限性：
1) 当前实现仍处于单任务、单引擎（Isaac Sim 5.1）验证阶段，尚未覆盖多引擎互通与大规模并行；
2) 任务生成与随机化依赖手工定义的模板，缺乏完全自适应的生成器；
3) 评测指标和协议版本管理尚未成熟，跨版本比较仍需人工桥接；
4) 对真实机器人部署的 sim-to-real 适配仍有限，需进一步融合真实数据校准；
5) 资源消耗与成本控制未做细粒度预算，可能在大规模部署时出现瓶颈。

---

## 290. PhysisForcing: Physics Reinforced World Simulator for Robotic Manipulation

**arXiv ID:** 2606.28128 | [PDF](https://arxiv.org/pdf/2606.28128v1)

**作者:** Peiwen Zhang `[一作]` (Peking University), Daquan Zhou `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 PhysisForcing，一种在训练时通过层次化物理对齐提升机器人视频生成的框架。

**💡 创新点**

创新之处在于将物理监督聚焦于交互关键区域，结合像素级运动一致性和语义级关系一致性的双重层次对齐。

**🔧 技术方法**

技术上基于 Diffusion 的 DiT 模型，利用点跟踪、冻结的视频理解编码器以及像素轨迹与语义关系对齐损失实现对齐。

**📊 数据集**

采用过滤后的 RoVid-X 50 万机器人视频以及 R‑Bench、PAI‑Bench、EZS‑Bench 等三个生成基准，以及 WorldArena 与 RoboTwin 用于策略评估。

**📈 对比分析**

在多项基准与商业/开源对照模型的对比中，PhysisForcing 在 R‑Bench 上实现了 9% 的物理可行性提升、在 WorldArena 关断循环成功率从 16% 提升至 24%，并使下游策略平均成功率提高 4.6%。

**⚠️ 局限性**

局限性包括仍需依赖训练时的物理监督与手工定义的区域掩码，且在更广泛或更复杂的交互场景下的鲁棒性未充分验证。

---

## 291. From Tokens to States: LLMs as a Special Case of World Models and the Continuous Path Beyond

**arXiv ID:** 2606.28127 | [PDF](https://arxiv.org/pdf/2606.28127v1)

**作者:** Paul Dubois `[一作]` `[通讯]`, Paul Dubois

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文从理论角度证明大语言模型（LLM）是受限的世界模型（World Model）的特殊子集，并提出一条从标准下一词预测（NTP）到联合嵌入预测架构（JEPA）的连续谱框架，阐释在中间节点（多词预测、未来摘要预测、下一潜在预测）如何逐步放宽LLM的约束、扩展世界模型能力，同时说明数据与架构瓶颈的演进。

**💡 创新点**

创新点在于：① 将LLM定位为严格包含关系的世界模型特殊例子，否定二元对立；② 通过“连续谱”构造将不同预测目标、数据需求与架构适配性系统化，形成从文本自监督到动作条件连续状态预测的渐进路径；③ 对不同节点的约束与优势进行系统比较，揭示在未来计划与推理任务中更细粒度的技术选择。

**🔧 技术方法**

主要技术包括：Transformer自回归架构、Multi‑Token Prediction (MTP)、Future‑Summary Prediction、Next‑Latent Prediction、以及JEPA（Action‑conditioned Latent‑space Prediction）。同时利用隐藏层的线性可解码方法验证内部世界模型的存在，并通过自监督训练（文本或隐藏状态）实现各阶段模型。

**📊 数据集**

使用的数据集：
- 互联网文本（≈10¹³ tokens）用于训练 NTP、MTP、Future‑Summary；
- 模型自身隐藏状态自监督（≈10¹¹）用于 Next‑Latent；
- 仪器化环境数据（≈10⁹ samples）用于 JEPA 训练（如机器人、驾驶仿真、游戏引擎等）。

**📈 对比分析**

对比方式：对 OthelloGPT、棋类 LLM、Llama‑2 等模型的隐藏层进行线性解码，评估其对棋盘状态、地理空间和时间等世界信息的编码精度；报道指标主要为 >99% 的线性可解码准确率。对中间架构（MTP、Future‑Summary、Next‑Latent）在推理与规划任务中的性能提升已在文献中报道（如 MBPP 任务提升 15%），但本文未给出统一实验表格或绝对数值。

**⚠️ 局限性**

局限性：
1) 对连续状态预测仍缺乏与 Transformer 等价的专用架构，导致在 JEPA 端数据与模型匹配困难；
2) 中间节点的实际规划效能尚未在大规模真实任务中系统验证；
3) 文章主要为观点性综述，缺乏新的实验结果与量化评估；
4) 依赖于大量文本自监督数据，仍未突破文本数据瓶颈向更稀缺的动作标注数据迁移的难题。

---

## 292. Dangerous Liaisons of Convex Learning and Non-Affine Aggregation

**arXiv ID:** 2606.28123 | [PDF](https://arxiv.org/pdf/2606.28123v1)

**作者:** Thomas Boudou `[一作]` (Inria), Aurélien Bellet `[通讯]` (Inria)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在凸学习中使用非仿射聚合规则时对更新操作单调性的影响，并证明只有正仿射聚合才能保持单调性，从而保证稳态收敛与良好的泛化性能。

**💡 创新点**

核心创新在于给出非仿射聚合无法保持单调性的必要与充分条件，提出可恢复单调性的充分条件，并用此统一解释现代学习系统中多种失效模式。

**🔧 技术方法**

主要采用了凸优化理论、单调性分析以及算法稳定性理论，对聚合规则的数学性质进行了严谨证明。

**📊 数据集**

本研究为理论分析性质，未涉及具体数据集；若有实验，主要以传统线性平均聚合为对照。

**📈 对比分析**

作为纯理论工作，论文未给出实验性能对比；若有实验，结果显示非仿射聚合导致收敛速率下降、稳态偏差增大，稳定性显著受损。

**⚠️ 局限性**

局限性包括：仅适用于凸学习框架；对非凸问题的适用性未知；所给的恢复单调性的充分条件可能过于严格，缺乏实证验证。

---

## 293. Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability

**arXiv ID:** 2606.28116 | [PDF](https://arxiv.org/pdf/2606.28116v1)

**作者:** Ruixuan Huang `[一作]` (Hong Kong University of Science and Technology), Yang Zheng `[通讯]` (Huawei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出针对大型语言模型训练中关键子模块（低精度Flash Attention 与 MoE 路由器）的机制驱动监控方法，利用模块功能与失效机制推导内部监视器，在训练早期即可检测到数千步的失稳信号。

**💡 创新点**

创新点在于：①从子模块功能与失效机理出发设计内部监视器；②对Flash Attention 采用 ΔW 的谱熵与一阶二阶双线性分解，能在不需要重计算或内核改动的情况下捕捉低精度误差导致的低秩漂移；③针对 MoE 路由器设计了路由权重相似度与每个 token 的熵指标，揭示学习率和全局批大小对路由器稳定性的影响；④演示两类监控信号不互相干扰，能分别定位不同故障。

**🔧 技术方法**

技术手段包括：权重更新 ΔW 的奇异值谱与稳定秩、谱熵；双线性 QK 分解的第一阶/第二阶项；路由器权重中心化与相似度下界；Per-token 路由熵与负载平衡指标；以及基于随机矩阵理论的低秩集中预测。

**📊 数据集**

实验使用标准大规模预训练语料（数万亿 token 的 LLM 训练集），在 Frontier 级别的加速器集群上进行故障注入实验（低精度 FA、学习率/批大小扰动）。

**📈 对比分析**

与传统全局指标（loss、梯度范数、权重范数）相比，本文的内部监视器可在低精度 FA 下提前约 9~17k 步检测到失稳；在 MoE 路由器的学习率/批大小实验中，路由熵指标能在 loss 衰退前数千步显著下降，而 ΔW 谱指标保持正常，证明了两类监控信号的可区分性。

**⚠️ 局限性**

局限性包括：①仅覆盖特定的故障类型（低精度 FA 与学习率/批大小），未覆盖 FP8、随机舍入、梯度裁剪等；②对不同注意力变体（MLA、GQA 等）的分解需要重新推导；③缺乏闭式的失稳起始步预测，仅基于经验；④路由器指标在激活非齐性时的解释性有限。

---

## 294. OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal

**arXiv ID:** 2606.28094 | [PDF](https://arxiv.org/pdf/2606.28094v1)

**作者:** Qinming Zhou `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种单步扩散模型 OSOR，用于实现对象及其视觉效果（阴影、反射）的高效、效果感知、mask 鲁棒的去除。

**💡 创新点**

创新点包括：①占据引导判别器实现单步边界一致性；②轻量 alpha 头在不完整 mask 下自适应混合；③语义锚定验证管道 SAVP 用于大规模获取效果感知训练数据。

**🔧 技术方法**

采用扩散模型的单步潜在恢复、PatchGAN 占据引导判别器、alpha 头、SAVP 以及 LoRA 参数适配等技术。

**📊 数据集**

使用 SAVP 构造的 CORNE 数据集（28 万对）以及 AnimeEraseBench 和 TextEraseBench。

**📈 对比分析**

与多步扩散基线、GAN、Transformer 方法比较，OSOR 在 6 个基准上在 PSNR、FID、LPIPS 等指标均优于其他方法，速度提升 4–30 倍，单张 1024×1024 图像在 A100 上 1 秒内完成。

**⚠️ 局限性**

局限性：仍受训练数据质量和掩码误差影响，对极端光照或复杂阴影可能效果不足；单步扩散对极大遮挡仍有挑战。

---

## 295. RPM-Distill: Physiology-guided Adaptive Cross-modal Distillation for Robust Remote Physiological Measurement

**arXiv ID:** 2606.28089 | [PDF](https://arxiv.org/pdf/2606.28089v1)

**作者:** Jiyao Wang `[一作]` (McGill University), Jiangbo Yu `[通讯]` (McGill University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出RPM-Distill，一种利用训练时雷达信息进行频域生理结构蒸馏的框架，使得仅用视频即可进行远程生理测量。

**💡 创新点**

创新点在于将视频与雷达共享的周期节律映射到频域，并设计峰值对齐、背景匹配、形状一致三项生理约束；同时引入谱策略网络和双层元学习，使蒸馏过程对样本可靠性自适应。

**🔧 技术方法**

技术方法包括频谱蒸馏（三个生理约束）、基于频谱关系的自适应策略网络、双层元学习调度蒸馏门控与权重、以及使用FactorizePhys等深度模型与预训练雷达教师。

**📊 数据集**

实验使用四个公开数据集：EquiPleth、PhysDrive、PURE、MMPD。

**📈 对比分析**

在跨数据集、光照/运动挑战以及少量标签设置下，RPM-Distill 将MAE下降约81%/21%，相关系数提升至0.94，显著优于传统融合与KD基线。

**⚠️ 局限性**

局限性包括：需要训练时同步的视频-雷达配对；对非周期或强非平稳信号的适用性有限；在极端失同步或雷达缺失的场景下仍可能产生负迁移。

---

## 296. TextDS: Parameter-Efficient Representation Alignment for Scene Text Detection under Distribution Shifts

**arXiv ID:** 2606.28077 | [PDF](https://arxiv.org/pdf/2606.28077v1)

**作者:** Boyuan Chen `[一作]` (Xi'an Jiaotong University), Yi Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种高效的场景文本检测框架 TextDS，能够在分布偏移和恶劣成像条件下保持鲁棒性。

**💡 创新点**

创新点包括：双编码器（SAM2 与 DINOv3）无需文本预训练、逐步低秩适配器 SWLoRA 的动态早停、以及共空间融合 CSF 以保持域鲁棒特征。

**🔧 技术方法**

使用视觉基础模型 SAM2‑Hiera‑L 与 DINOv3‑ViT‑L/16，LoRA 适配，子空间投影与融合，轻量化解码器。

**📊 数据集**

使用 CTW‑1500、Total‑Text、MLT 三大基准，并构造降噪版本（雨、雾、曝光不足/过度、低分辨率）进行评估。

**📈 对比分析**

与 DB‑Net、TextBPN、TextPMs、S3INet 等现有方法对比，TextDS 在三大基准上达到 90%+ 的 F‑measure，参数仅 4.9M，FPS 44.1，显著优于传统方法且在域迁移与恶劣条件下表现更稳健。

**⚠️ 局限性**

局限性：依赖双编码器的高分辨率特征提取，可能在极低分辨率或极端噪声场景下仍有限；以及对基础模型版本的依赖，使得迁移到更小模型时性能尚未验证。

---

## 297. Single and Multi Truth Data Fusion using Large Language Models

**arXiv ID:** 2606.28062 | [PDF](https://arxiv.org/pdf/2606.28062v1)

**作者:** Hira Beril Kucuk `[一作]` (University of Manchester), Zhenyu Wu `[通讯]` (University of Manchester)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文探究了在表格数据融合（真值发现）任务中使用大型语言模型（LLM）的方法，设计了多种提示策略（零样本/一示例、域相关/域无关、单真值/多真值），并在三个基准数据集上进行实验。

**💡 创新点**

创新点在于：①首次将LLM作为直接真值发现组件，提出可变提示框架；②系统评估域特定提示、示例、约束对融合效果的影响；③在公开数据集上展示LLM优于传统无监督真值发现算法。

**🔧 技术方法**

核心技术为：基于GPT‑4o‑mini等LLM的提示生成与推理；对比传统方法如Majority Vote、Source Reliability Vote、LTM、DART。

**📊 数据集**

使用的三数据集为：Book（多真值作者列表），Movie（多真值导演年份），Flight（单真值起降时间），其中Flight还进一步通过obfuscated ID评估背景知识影响。

**📈 对比分析**

与传统基线相比，域相关的一示例提示在Book、Movie、Flight均取得更高的F1（最高达0.9119），尤其在单真值Flight上F1提升至0.9119；域无关提示也表现优异。

**⚠️ 局限性**

局限性包括：示例与约束对不同数据集效果不一致；LLM对结构化时间等细节仍易出错；依赖于LLM的算力与成本，且在非公共领域可能表现欠佳。

---

## 298. ReScene: Structured Indoor Scene Reconstruction from Multi-View Captures

**arXiv ID:** 2606.28060 | [PDF](https://arxiv.org/pdf/2606.28060v1)

**作者:** Haoran Xu `[一作]` (East China Normal University), Xin Tan `[通讯]` (East China Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

ReScene能够自动将随意拍摄的室内视频转化为可直接用于仿真、包含物体级结构、交互关系且物理可行的三维场景；

**💡 创新点**

其创新点在于：①使用HierView基于语义一致性与3D完整度的多层视角筛选，提升单视图重建质量；②采用Relation-Aware Assembly融合多帧视觉语言模型关系预测与几何/房间框架先验，生成可信的场景图并通过规则求解实现物理一致布局；

**🔧 技术方法**

技术实现包括CLIP语义对齐、Sim(3)点云配准、基于VLM（如Qwen3‑VL‑32B）关系预测、几何关系能量闭式求解与非穿透后处理；

**📊 数据集**

主要使用ScanNet真实室内场景（30个房间）作为输入，并基于重建结果自动生成约5,688条带解析答案的Embodied VQA数据集；

**📈 对比分析**

与SimRecon、RICO、DRAWER等现有多视图方法比较，ReScene在Chamfer Distance下降约17%、LPIPS下降26%，运行速度最快（10倍提升），同时在结构合理性上OOB率降低至0.9%、碰撞率显著下降；

**⚠️ 局限性**

局限性包括：对相机标定和点云质量敏感，仍难以处理极端拥挤或遮挡严重的场景；VQA生成的空间推理依赖解析答案，对测量精度较差的距离判断仍不够精准；

---

## 299. AirGroundBench: Probing Spatial Intelligence in Multimodal Large Models under Heterogeneous Multi-View Embodied Collaboration

**arXiv ID:** 2606.28049 | [PDF](https://arxiv.org/pdf/2606.28049v1)

**作者:** Haotian Li `[一作]` (Beijing University of Posts and Telecommunications), Zhaofeng He `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了AirGroundBench基准，专门评估异构无人机-地面车辆双视角空间理解，并包含多任务VQA与闭环VLN；

**💡 创新点**

首次系统化探测跨视角几何一致性，提供结构化空间注释、跨视角身份绑定与度量化3D框，打造从感知到决策的完整能力维度；

**🔧 技术方法**

采用多模大语言模型零样本推理、四选一单步VQA、视觉语言导航、Unreal Engine+AirSim仿真与几何校准技术；

**📊 数据集**

利用11个高保真仿真环境生成1021对同步UAV‑UGV图像，产生62k双视角VQA实例和115个闭环VLN情节，配有2D/3D边界框与跨视角ID；

**📈 对比分析**

在UAV‑UGV单视角与双视角三种输入下对13个代表性MLLM进行评估，双视角可提升约15‑20%准确率，但最高模型VQA平均准确率仍仅为54%，VLN成功率低于40%，与人类约80%/90%仍相距甚远；

**⚠️ 局限性**

主要局限在于几何一致性仍是瓶颈，模型难以完成跨视角对齐与坐标变换，累积误差导致导航失败；此外数据来源为仿真，缺乏真实世界噪声与多样性。

---

## 300. Evolution-Aware Regression Test Prioritization of ML-Enabled Systems Using Gradient-Based Behavior Vectors

**arXiv ID:** 2606.28037 | [PDF](https://arxiv.org/pdf/2606.28037v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 301. A Flexible Encoding Model for Non-Unique Note Alignments

**arXiv ID:** 2606.28032 | [PDF](https://arxiv.org/pdf/2606.28032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 302. A Tree-of-Thoughts Inspired Hybrid Approach for Legal Case Judgement Summarization using LLMs

**arXiv ID:** 2606.28044 | [PDF](https://arxiv.org/pdf/2606.28044v1)

**作者:** Aniket Deroy `[一作]` (IIT Kharagpur), Saptarshi Ghosh `[通讯]` (IIT Kharagpur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于树形思维（Tree-of-Thoughts）的混合式提取-抽象式法律判决摘要方法，并在两个大型语言模型（DeepSeek-R1 与 Llama-3.2）上实现。

**💡 创新点**

创新点在于将三阶段提示（①提取关键信息，②评估片段重要性，③生成抽象式摘要）融合为一个完整的思维流程，使提取与抽象互相补偿，既保留事实准确性，又提升文本流畅度。

**🔧 技术方法**

使用的技术包括：大型语言模型（DeepSeek-R1、Llama-3.2）、多阶段提示设计、以及ROUGE、METEOR、MoverScore、BERTScore等标准评估指标。

**📊 数据集**

实验采用 IN-Ext 数据集，该数据集包含 50 对（法律判决、专家写作摘要）样本，供模型训练与评估使用。

**📈 对比分析**

比较方法：在同一数据集上分别使用提取式、抽象式和 Ext-Abs-ToT 三种提示，对每种 LLM 生成摘要后计算四项指标；结果显示 Ext-Abs-ToT 在大多数指标上均优于单一方式，DeepSeek-R1 的表现优于 Llama-3.2，抽象式略优于提取式，混合式略胜抽象式。

**⚠️ 局限性**

局限性：样本量仅为 50 条，可能不足以覆盖法律文本多样性；仅测试了两款 LLM，未验证在更大模型或跨语言场景的泛化能力；缺少对生成错误（事实偏差、法律推理失误）的系统评估。

---

## 303. How Humans, Bots, and Agents Communicate About Vulnerabilities in Pull Requests

**arXiv ID:** 2606.28125 | [PDF](https://arxiv.org/pdf/2606.28125v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 304. Mind the Gap: Quantifying the Domain Gap in Cross-Sensor Diffusion Super-Resolution

**arXiv ID:** 2606.28039 | [PDF](https://arxiv.org/pdf/2606.28039v1)

**作者:** Dawid Kopeć `[一作]` (Wrocław University of Science and Technology), Maciej Zięba `[通讯]` (Wrocław University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了跨传感器超分辨率中的合成到真实域差距，系统评估了五种扩散模型，提出了基于Sentinel-2自监督特征的LPIPSSat感知指标，并构建了大规模的 Sentinel‑2 与 PlanetScope 时空对齐数据集。

**💡 创新点**

①首次系统量化合成-真实域差距对扩散 SR 模型的影响；②设计了适用于卫星影像的 LPIPSSat 感知度量；③构建了大规模、几何和时间同步的 Sentinel‑2/PlanetScope 数据集；④通过燃烧面积分割任务评估 SR 对实际应用的影响。

**🔧 技术方法**

采用扩散模型（DDPM、Flow Matching、I²SB、ResShift、UniDB），统一 U‑Net 架构，使用混合精度训练与余弦学习率调度；使用 LPIPSSat 作为感知评价指标；在 NVIDIA A100 GPU 上进行实验。

**📊 数据集**

Sentinel‑2 与 PlanetScope 的 175 对匹配影像（约61k 训练对、7k 测试对），以及 CaBuAr 燃烧面积分割数据集用于下游评估。

**📈 对比分析**

在三种配置（合成基线、跨传感器评估、真实训练）下，分别计算 PSNR、SSIM、LPIPS 和 LPIPSSat。合成训练表现优异（PSNR>43 dB），但跨传感器性能显著下降（PSNR≈16.9 dB），真实训练虽改善感知指标但仍远低于合成基线。下游燃烧面积任务显示，synthetic‑SR 模型略有提升，而 real‑SR 模型导致 IoU 降低。

**⚠️ 局限性**

存在显著域差距，扩散 SR 模型难以同时完成 SR 与域适配；合成训练虽然稳定但在真实环境中的实用性受限；LPIPSSat 仍局限于 RGB 通道，无法完整覆盖多光谱信息；下游评估受标注质量和任务适配性限制。

---

## 305. Translation as a Bridging Action: Transferring Manipulation Skills from Humans to Robots

**arXiv ID:** 2606.28133 | [PDF](https://arxiv.org/pdf/2606.28133v1)

**作者:** Sijin Chen `[一作]` (HKU-MMLab), Xihui Liu `[通讯]` (HKU-MMLab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了如何通过仅使用腕部平移的桥接动作，将人类操纵技能迁移到双手机械臂机器人，并提出了一套三阶段训练框架（人类预训练、机器人共训练、少量机器人微调）实现任务执行。

**💡 创新点**

创新点在于：①提出仅使用相对腕部平移作为跨身体的共享动作空间，避免了噪声旋转信息的干扰；②设计交错动作序列与注意力掩码处理不同来源动作缺失；③实现了大规模人类数据预训练与机器人执行的一体化模型。

**🔧 技术方法**

使用的技术包括π₀式vision‑language‑action模型、交错动作序列与注意力掩码、流匹配动作目标、视觉语言共训练以及三阶段训练策略。

**📊 数据集**

主要数据集：≈600小时人类动作（70h EgoDex、500h 物件家居、45h 实验室）、≈72h机器人pick‑and‑place、≈3h任务特定人类动作、≈100条机器人轨迹/任务、少量（10）机器人微调轨迹。

**📈 对比分析**

通过与仅用机器人pick‑and‑place训练、仅用6DoF人类动作共训练等方案对比，实验表明桥接动作显著提升任务成功率与进度，尤其在大规模人类预训练后提升约10‑20%，并在少量机器人示例下进一步提升至约55%成功率。

**⚠️ 局限性**

局限性包括：①仅使用平移动作无法处理需要精细旋转的任务；②人类动作噪声和观测差距导致对薄物体抓取表现不佳；③缺乏高质量机器人演示限制了上限性能。

---

## 306. The Reciprocal Impact of Science and Software: A Cross-Corpus Analysis of How Research Shapes Software and Software Enables Research

**arXiv ID:** 2606.28120 | [PDF](https://arxiv.org/pdf/2606.28120v1)

**作者:** Audris Mockus `[一作]` `[通讯]` (University of Tennessee), Audris Mockus (University of Tennessee)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个覆盖全球公开源代码与科学文献的跨语料库图谱（Science‑Software Supply Chain），并利用该图谱评估科学对软件及软件对科学的相互影响。

**💡 创新点**

首次实现跨语料库的身份解析与图谱构建，提供了可视化科学‑软件链路的全景视图，并揭示两者影响层面的差异性。

**🔧 技术方法**

使用图数据库、自然语言处理（NER）提取软件提及、版本控制历史分析、身份桥接技术等。

**📊 数据集**

World of Code、Semantic Scholar、OpenAlex 以及 SciCat 科学仓库数据。

**📈 对比分析**

与 Softcite 黄金语料对齐评估召回率，并与传统引文、星标指标对比；发现软件依赖入度与星标相关性弱（Spearman ρ=0.36）。

**⚠️ 局限性**

链接稀疏、命名不规范导致召回低；软件‑科学交互的因果解释受限；对 AI 代码代理的测度仍需进一步完善。

---

## 307. MMAO: A Metabolic Multi-Agent Optimizer with Endogenous Resource Allocation for Continuous and Discrete Optimization

**arXiv ID:** 2606.28109 | [PDF](https://arxiv.org/pdf/2606.28109v1)

**作者:** Jinliang Xu `[一作]`, Liping Ma `[通讯]` (Seventh Medical Center of Chinese PLA General Hospital)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于代谢闭环的多智能体优化器（MMAO），在连续和离散优化中实现自适应资源分配和动态种群管理。

**💡 创新点**

核心创新是将能量循环作为唯一的自适应机制，生成连续角色状态、感知强度、搜索尺度与生命周期决策，从而实现跨域参数轻量化。

**🔧 技术方法**

采用私有-公共能量账务、对称零阶梯度估计、结构感知与边缘记忆、角色插值运动、以及能量驱动的分支/修剪/再生策略。

**📊 数据集**

在CEC2017子集（10D/30D，20个随机种子）和五个TSPLIB实例（共100个种子）上进行实验。

**📈 对比分析**

与随机搜索、爬山、DE-lite等连续基线以及NN+2opt、RR-2opt、CI+2opt等离散基线对比，MMAO在所有基准上显著优于轻量级基线，表现稳健但未能与专业级求解器竞争。

**⚠️ 局限性**

主要局限包括高维时评估成本较高、离散版仍未匹配顶尖TSP求解器、部分参数仍需经验设定、理论分析仅覆盖稳定性而非收敛性，以及在更大规模基准上的验证不足。

---

## 308. Diffusion Model Attribution via Spectral Coupling of Denoiser Responses

**arXiv ID:** 2606.28092 | [PDF](https://arxiv.org/pdf/2606.28092v1)

**作者:** Pragati Shuddhodhan Meshram `[一作]` (University of Illinois Urbana-Champaign), Varun Chandrasekaran `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Spectral Denoising Signatures（SDS）的无侵入式方法，通过对扩散模型去噪函数的频谱响应进行探测，实现对生成图像来源模型的精准归因。

**💡 创新点**

创新点在于发现并利用去噪器对频率能量的重分配模式（谱耦合）作为模型身份的内在指纹，且不需要对模型或输出进行修改。

**🔧 技术方法**

采用频率受控噪声注入、谱耦合矩阵构造、跨时步聚合以及线性/核分类器来生成和匹配模型指纹。

**📊 数据集**

使用Stable Diffusion（v1.4、v1.5、v2.1）、SDXL、PixArt‑α、Dreamshaper‑8、Realistic Vision v5等八款扩散模型的生成图像，实验基于 SD Prompts、MS‑COCO 等公开数据集。

**📈 对比分析**

与 LatentTracer、RONAN 等现有非侵入式基线相比，SDS 在闭集任务上实现了 ≈100% 的准确率，跨域提示下误差仅 3.8pp，且在旋转、JPEG、裁剪、亮度、噪声、模糊等强失真下仍保持高准确率。

**⚠️ 局限性**

局限性包括只能在已注册模型的闭集环境下工作、需要白盒访问候选模型权重、以及对生成后图像篡改缺乏检测能力。

---

## 309. Context-Aware Explanations for Spatialized Document Layouts

**arXiv ID:** 2606.28081 | [PDF](https://arxiv.org/pdf/2606.28081v1)

**作者:** Wei Liu `[一作]` (Virginia Tech), Rebecca Faust `[通讯]` (Tulane University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了CAPE框架，能够生成结合文档语义与布局空间上下文的自然语言解释，帮助用户理解空间化文档布局中的结构与关系。

**💡 创新点**

创新点在于将语义内容与多层空间上下文（全局、局部、模式特定）融合，用LLM生成针对不同空间模式（聚类、子群、离群、桥接文档）的可解释文本，并支持多级细节与用户驱动探索。

**🔧 技术方法**

技术主要包括空间模式识别（聚类、密度检测、离群检测）、结构化上下文构建、基于提示的LLM（GPT‑4o）解释生成、交互式可视化（AI引导与用户驱动模式）和多级解释输出。

**📊 数据集**

使用两套数据集：20 Newsgroups新闻文本（216篇）和IEEE VIS 2022/23会议论文（252篇），通过文本嵌入（OpenAI text‑embedding‑3‑small）投影到二维空间（t‑SNE或UMAP）进行实验。

**📈 对比分析**

通过10名技术背景受试者的对照实验，比较CAPE、仅基于内容的LLM和关键词基线。CAPE在位置感知与总体有用性上显著优于其他两种方法（p<0.01），而在清晰度、相关性和细节性上与内容‑仅LLM相当；关键词基线在所有维度上均最低。

**⚠️ 局限性**

局限性包括受试者规模小且技术背景单一，未测量实际分析效率或准确性，依赖于布局的质量与模式识别的启发式方法，可能在更复杂或不同领域的文档集合中表现不一致。

---

## 310. Ontology-Guided Evidence Path Inference for Multi-hop Knowledge Graph Question Answering

**arXiv ID:** 2606.28076 | [PDF](https://arxiv.org/pdf/2606.28076v1)

**作者:** Yongxue Shan `[一作]` (National University of Defense Technology), Xiaodong Wang `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为OPI的多跳知识图问答框架，利用本体级关系签名进行答案侧约束，结合前缀扩展与最终跳匹配实现双向检索，并通过生成‑修正循环实现答案精细化；

**💡 创新点**

创新点在于①构建关系中心化本体图，显式捕获关系的头尾类型约束；②在检索阶段采用答案类型指导的双向检索，抑制路径爆炸与类型混杂；③引入迭代生成‑修正机制，使模型能够在检索与答案间相互反馈，提升语义对齐与噪声过滤；

**🔧 技术方法**

使用了大型语言模型微调（Llama‑2‑7B）进行答案类型预测，Prompt‑based LLM（DeepSeek‑v3/GPT‑4o）进行生成‑修正；本体构建采用基于schema或数据驱动的关系签名抽取；检索采用基于本体的最终跳约束的双向扩展；答案精细化采用生成‑修正循环；

**📊 数据集**

在三个主流 KGQA 任务上实验：WebQSP、CWQ（均基于 Freebase）和 MetaQA（基于 Wiki‑Movie）；

**📈 对比分析**

与嵌入、检索、单一 LLM 以及多种 KG+LLM 基线（如 ToG、RoG、ORT、GCR 等）对比，OPI 在 WebQSP 上 Hit@1 达到 92.3、F1 74.9，CWQ 上 Hit@1 76.5、F1 59.6，显著优于最强对手；在 MetaQA 上检索仅阶段即接近饱和 Hit@1 100%；

**⚠️ 局限性**

主要局限在于对本体完整性和实体类型信息的依赖，缺失或噪声类型会削弱检索约束效果；对极其模糊或多义的问题仍可能产生误检；以及在极大规模图上双向检索与多轮生成‑修正的计算开销仍有提升空间；

---

## 311. ToolPrivacyBench: Benchmarking Purpose-Bound Privacy in Tool-Using LLM Agents

**arXiv ID:** 2606.28061 | [PDF](https://arxiv.org/pdf/2606.28061v1)

**作者:** Shijing Hu `[一作]` (Beijing University of Posts and Telecommunications), Zhicheng Zhao `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 ToolPrivacyBench，一个面向多工具 LLM 代理的目的绑定隐私评估基准，结合可执行工作流、政策知识库和轨迹审计，评估代理在完成任务时是否仅将必要的私有信息传递给授权工具与接收器。

**💡 创新点**

创新点在于①引入目的绑定隐私过度披露概念；②构造私有原子–工具–接收器授权矩阵；③采用轨迹级审计和后端日志对隐私泄漏进行细粒度评估；④设计多层次指标（FOR、SWLR、ToolFOR、FreeTextFOR、MT‑POI、SMTC）以全面衡量功能与隐私的权衡。

**🔧 技术方法**

技术方法包括：OpenClaw 可执行环境模拟工具调用，Mock 后端记录实际参数；知识库驱动的隐私检测器与授权推理器；多维度指标计算与可视化；对九种主流 LLM 代理在统一环境下进行多工具任务执行与评估。

**📊 数据集**

数据集包括：1,150 条内部构造的需要了解型隐私工作流（涵盖医疗、保险、金融、税务、招聘、IT、软件安全等领域）；1,000 条从现有多工具基准（如 τ‑bench、API‑Bank、BFCL、AppWorld）改编而来的公开案例，结合人工标注的私有原子、工具用途与授权关系。

**📈 对比分析**

比较方法：在统一可执行环境中跑九种 LLM 代理，记录任务完成率（TaskSuccess）和多层次隐私泄漏指标（FOR、SWLR、ToolFOR、FreeTextFOR、MT‑POI、SMTC）。实验结果显示：任务完成率普遍在 92–98% 之间，但 MT‑POI 在 15–28 之间，表明高功能完成并不必然意味着低隐私泄漏；SMTC 综合评价显示不同模型在功能与隐私上的权衡差异显著。

**⚠️ 局限性**

局限性：使用的后端为 Mock，缺乏真实生产环境和动态访问控制；授权矩阵为人工标注，可能存在误差；评估聚焦非对抗性隐私泄漏，未考虑攻击场景；数据为合成或测试值，未覆盖真实敏感信息，难以直接映射到实际业务风险。

---

## 312. Rapid Prototyping of Event-Driven Contextual Memory in the ACT-Up Cognitive Architecture

**arXiv ID:** 2606.28045 | [PDF](https://arxiv.org/pdf/2606.28045v1)

**作者:** Robert Thomson `[一作]` (Carnegie Mellon University), Christian Lebiere `[通讯]` (Carnegie Mellon University)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现了一个可扩展的上下文记忆与事件处理模块，集成了ACT-Up的工作记忆与联想学习，并在多轮序列回忆实验中验证其有效性。

**💡 创新点**

创新点在于提供了理论中立的工作记忆与传播激活实现，以及轻量级事件系统，降低新用户学习门槛，同时将生成式AI用于实验代码自动化。

**🔧 技术方法**

采用了ACT-Up核心架构、Python实现PyACT-Up、事件处理库以及生成式AI提示，结合自定义的传播激活与衰减函数。

**📊 数据集**

使用了klein2005comparative的多次呈现序列记忆实验数据（19个单词，每个列表呈现5次）。

**📈 对比分析**

通过比较CRP曲线和序列/自由回忆的准确率，模型在100次迭代下的r²>0.9，性能与现有理论相当，且运行时间约30秒。

**⚠️ 局限性**

局限在于基于简化的衰减与传播函数，未能完全稳定基线激活与联想激活的结合，且验证范围仅限于序列回忆，缺乏对更复杂任务的推广。

---

## 313. Typing Behavior in Human-LLM Interaction: Keystroke Dynamics Reveal Cognitive Effort During Prompting

**arXiv ID:** 2606.28090 | [PDF](https://arxiv.org/pdf/2606.28090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 314. OperatorSHAP: Fast and Accurate Shapley Value Estimation for Neural Operators

**arXiv ID:** 2606.28065 | [PDF](https://arxiv.org/pdf/2606.28065v1)

**作者:** Joshua Stiller `[一作]` (LMU Munich), Eyke Hüllermeier `[通讯]` (LMU Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为OperatorSHAP的网格无关解释器，用于在神经算子模型上实现快速、可扩展的Shapley值解释；

**💡 创新点**

其创新点在于将Aumann–Shapley理论与函数空间中的神经算子相结合，构建可训练的FastSHAP式解释器，并能在不同网格分辨率上无缝迁移；

**🔧 技术方法**

主要技术包括：基于Sobolev空间的Fréchet可微神经算子（如FNO、LocalFNO、GINO）、Aumann–Shapley值的解析公式、FastSHAP加权MSE损失、以及对多分辨率采样的网格无关训练；

**📊 数据集**

实验使用了九个结构化的1D/2D PDE数据集以及MeshGraphNets的两大基准数据集，且通过随机步长生成了异构网格；

**📈 对比分析**

与KernelSHAP、RegressionMSR（大预算）以及积分梯度（IG）对比，OperatorSHAP在Pearson相关系数、NRMSE和faithfulness指标上与最优基线相当甚至更好，同时推理速度快数十倍，且同一模型可在不同网格尺寸下直接使用；

**⚠️ 局限性**

局限性包括：训练成本高，尤其在高分辨率场景下需要大量梯度计算；对平滑掩码与Fréchet可微性的假设可能限制适用范围；缺乏真实ground‑truth导致评价依赖相对指标；以及解释结果若不慎可能误导决策。

---

## 315. AB-Sync: Attention-Based Slot-Level Clock Synchronization Method for UWB-TDOA Localization Networks

**arXiv ID:** 2606.28087 | [PDF](https://arxiv.org/pdf/2606.28087v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 316. Higher-Order Fourier Neural Operator: Explicit Mode Mixer for Nonlinear PDEs

**arXiv ID:** 2606.28122 | [PDF](https://arxiv.org/pdf/2606.28122v1)

**作者:** Alex Colagrande `[一作]` (Miles Team, LAMSADE, Universite Paris Dauphine Psl), Alexandre Allauzen `[通讯]` (Miles Team, LAMSADE, Universite Paris Dauphine Psl)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Higher-Order Fourier Neural Operator（HO‑FNO），通过高阶谱卷积显式建模非线性偏微分方程中的频率交互；

**💡 创新点**

创新点在于在 Fourier Neural Operator 的谱卷积中加入 m‑线性混合，直接捕捉多项式非线性导致的频率三角/多角相互作用；

**🔧 技术方法**

技术上使用 FFT‑based 低维谱截断、点乘混合后再映射回时域，并在多种几何（欧氏网格、结构网格、点云、球面）中实现；

**📊 数据集**

实验数据集包括：Poisson 方程（多项式源）、Navier–Stokes (Airfoil, Pipe, 3‑D)、Elasticity, Plasticity, Darcy、SWE（球面）等标准 PDE 任务；

**📈 对比分析**

与 FNO、DSFNO、LaMO、Transformer、State‑Space 等 20+ 频域/注意力/SSM 模型对比，HO‑FNO 在所有基准上均优于传统谱网络，在 Navier–Stokes 与 Plasticity 上接近或超过最强 Transformer/SSM，且在高非线性情形下单层性能可匹敌 16 层 FNO；

**⚠️ 局限性**

局限在于高阶卷积的参数增长与计算复杂度随阶数 m 线性增加，适用于多项式非线性较强的 PDE，非多项式或高阶非线性时收益有限，并且在点云等非欧氏场景下仍需进一步优化。

---

## 317. Discrete Event Population Updates: finding game theoretic emergent behaviour in queueing systems with simulation

**arXiv ID:** 2606.28100 | [PDF](https://arxiv.org/pdf/2606.28100v1)

**作者:** Vincent Knight `[一作]`, Thomas Hutton `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了 Discrete Event Population Updates (DEPU) 框架，将离散事件仿真（DES）与进化动力学耦合，用以在缺乏闭式适应度表达式的排队系统中寻找平衡行为；实现了两种参考实现：Discrete Event Replicator Dynamics (DERD) 和 Discrete Event Moran Replacement (DEMR)。

**💡 创新点**

创新点在于：①消除了传统方法对闭式适应度的依赖；②通过单次长时间仿真而非嵌套短仿真+进化更新，显著降低计算成本；③提供可插拔的进化更新步骤，使框架可推广至任何可用 DES 的群体博弈问题。

**🔧 技术方法**

使用技术包括：离散事件仿真（Ciw 库）、进化博弈理论（复制动力学、Moran 过程）、指数平滑（非平稳多臂赌博机处理）、欧拉离散化、学习率与时间步长调参。

**📊 数据集**

数据集：全部为仿真生成，主要实验场景包括两节点与四层图书馆“jockeying”模型、扩展的六策略模型、Naor 可观测 M/M/1 队列。未使用外部真实数据。

**📈 对比分析**

比较方法：将传统 RD 与嵌套式 DES 的实现（RD）与 DEPU（DERD）在相同问题上进行对比，评价指标为 max_i |x_i(f_i - f̅)| 的收敛速度。实验显示 DERD 在相同精度下仅需约 44–50 倍更少的仿真客户数（如 0.005 精度时 RD 需 875,000 客户，DERD 仅 20,000），显著提升效率；然而 DERD 的解会围绕平衡点波动，无法突破约 1.5×10⁻⁵ 的噪声阈值。

**⚠️ 局限性**

局限性：①对适应度仅可通过 DES 采样的模型有限；②未处理完全不可观测的队列（需改写成本函数）；③在 DEMR 中需要足够大的固定种群才能避免随机漂移；④DERD 由于随机估计导致解在平衡附近振荡，无法实现极高精度；⑤对参数（学习率、时间步长、种群大小）的选择敏感，需经验调优。

---

## 318. JD Oxygen AI Item Center (Oxygen AIIC) V1: An Industrial-Scale LLM/VLM-Centric Solution for Item Understanding, Management, and Applications

**arXiv ID:** 2606.28070 | [PDF](https://arxiv.org/pdf/2606.28070v1)

**作者:** Oxygen AIIC `[一作]`, Ziyan Xing `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了JD Oxygen AI Item Center，结合人机协作的Ontology工程、Semantic Search‑then‑Discrimination知识生产框架、可自我演化的Item Understanding LLM/VLMs、统一的Item Tunnel和多业务应用矩阵，实现数十万类、数十亿SKU的商品知识自动化生成与服务；

**💡 创新点**

创新点包括：①基于人机协作的高效Ontology演进与自动化扩展；②将知识匹配拆解为语义检索+判别，显著提升吞吐并降低推理成本；③可自我演化的LLM/VLM框架，利用LoRAM+GROLE+GRPO实现持续精度提升；④统一Item Tunnel提供分层实时性与一致性的服务；⑤多维度并行优化（计算负载削减、缓存复用、异步流水线）实现十倍吞吐提升；

**🔧 技术方法**

使用技术包括：大型LLM/VLM（8B‑12B）+LoRA/LoRAM微调、Multi‑Task SFT、InfoNCE对比学习、Latent CoT蒸馏、谱噪声注入、GROLE专家组合、GRPO策略学习、vLLM推理、Huawei Ascend NPU、Faiss向量检索、Spark/Flink流批处理、Hudi MVCC、Kafka实时流、Parquet存储、Docker/K8s容器化；

**📊 数据集**

数据来源覆盖多源商品信息（标题、主图、详情页）、用户查询、外部网页内容，结合数十亿SKU、数十万类的业务数据；训练集包括数百万条手工标注与机器辅助生成的属性键值对、三元组，覆盖常见与长尾属性；

**📈 对比分析**

通过统一的Item Knowledge Test Set进行离线评估，最终模型实现94.2%精度、82.8%召回；在搜索场景覆盖80.4%流量，质量问题下降37%；自动属性填充率80%；创意点击率提升9%；相较传统BERT/NER方法，精度提升约2‑4%，吞吐提升10倍；

**⚠️ 局限性**

局限性包括：对极端长尾属性的覆盖仍不充分；多源异构数据的语义一致性与更新实时性仍需人工干预；模型对稀有概念的识别仍受限于人机协作与审核流程；系统对极端大规模并行时的资源调度与一致性控制仍具挑战。

---

## 319. Prophecy-Based Automated Verification of Message-Passing Programs

**arXiv ID:** 2606.28066 | [PDF](https://arxiv.org/pdf/2606.28066v1)

**作者:** Takashi Nagatomi `[一作]` (University of Tokyo), Ken Sakayori `[通讯]` (University of Tokyo)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将基于消息传递的并发程序功能正确性验证转化为约束 Horn 子句（CHC）求解的全自动方法，利用“预言”技术将发送通道视为未来要发送值的列表并为每个值附加时间戳以捕捉因果依赖，保证求解的安全性与完整性，并实现了一个 Rust‑style 程序的原型验证器。

**💡 创新点**

创新点在于：①将多发射器、多发送的消息通道通过预言列表和时间戳建模，实现模块化的发送/接收线程编码；②通过时间戳补全预言缺陷，获得对因果依赖的完整捕获；③证明了该转换在任意并发程序上保持安全性与完整性；④给出多种消息传递语义变体的适配方案。

**🔧 技术方法**

核心技术包括：预言（prophecy）建模、时间戳约束、CHC 编码、递归 Horn 子句求解以及多目标 CHC 求解器组合（CVC4、Z3、其他）。

**📊 数据集**

使用了若干小型 Rust‑like 程序基准：ack、multi‑sends、client‑server、calc‑server 等，并在每个程序上构造安全/不安全两种版本（后缀“-e”表示故障）。

**📈 对比分析**

与现有 CHC 求解器结合，采用 120 秒超时的投票策略；在安全基准中，除一个因子化推理不充分导致的例子外，其余全部在时间限制内通过；在不安全基准中均在不到 0.1 秒内判定失败，表明方法在小规模实例上具有良好的效率。

**⚠️ 局限性**

主要局限在于后端 CHC 求解器对递归谓词（如“列表只包含 1 的性质”）的合成不足，导致某些安全程序无法通过；需要改进求解器或引入手工/注解辅助的 catamorphism 推理。

---

## 320. Same Coeffect, Different Base: Connecting Two Dominant Approaches to Graded Types

**arXiv ID:** 2606.28042 | [PDF](https://arxiv.org/pdf/2606.28042v1)

**作者:** Vilem Liepelt `[一作]` (University of Kent), Dominic Orchard `[通讯]` (University of Kent)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文研究并阐明了两类梯度协变类型系统（线性基底与梯度基底）之间的关系，构造了相互嵌入的译码器，并证明其保持类型、梯度与运行时语义；

**💡 创新点**

创新点在于给出了两种主流协变系统的严格等价关系与互译，揭示它们在表达能力与可扩展性上的差异，并提出了线性与梯度基底的统一视角；

**🔧 技术方法**

主要使用了线性 λ 演算、分配模态、半环与代数结构、CPS（Continuation‑Passing Style）翻译、Polymorphic 以及产品/求和类型的形式化推导与证明；

**📊 数据集**

该工作为理论论文，未使用实验数据集；

**📈 对比分析**

对比方法通过构造可逆译码与语义保持定理进行理论验证，未给出具体运行时性能测量；

**⚠️ 局限性**

局限性包括：译码并非双射，无法形成完备逆映射；在引入产品/求和时线性基底无法直接翻译至梯度基底，需引入新的 Push 构造；部分翻译不保持 η‑等价，且对语义的完备性仍有待进一步研究。

---

## 321. Fair Classification with Efficient and Post-hoc Controllable Fairness-Accuracy Trade-off

**arXiv ID:** 2606.28097 | [PDF](https://arxiv.org/pdf/2606.28097v1)

**作者:** Maaya Sakata `[一作]` (University of Tsukuba), Kazuto Fukuchi `[通讯]` (University of Tsukuba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新型公平分类算法 GFB，能够在模型训练后通过后处理实现公平性与准确性的权衡控制，并在不需要重新训练的情况下快速调整公平性阈值。

**💡 创新点**

创新点在于：① 通过理论分析揭示特征分布聚焦于最公平决策边界能显著提升后处理分类器的权衡效率；② 设计基于该分析的分布塑形损失，学习特征变换，使得在训练阶段就已将特征拉近公平边界；③ 将该问题建模为双层优化并使用 MA‑SOBA 以及平滑近似实现可微分梯度求解。

**🔧 技术方法**

采用的技术包括：基于贝叶斯最优公平分类器的理论分析、特征变换网络、分布塑形损失（距离损失 + 预测损失）、双层优化、MA‑SOBA（移动平均 SOBA）梯度求解、平滑指标替代与阈值梯度解析。

**📊 数据集**

实验数据集包括四个公开数据集：CelebA、UTKFace（图像），Adult、COMPAS（表格），用于评估公平性（绝对人口统计差异 |DDP|）与准确率。

**📈 对比分析**

与 YOTO、FairBayes（后处理）以及 EPO、FairBiNN（前处理）等基线方法比较。GFB 在所有数据集上均实现更高的标准 Hypervolume（HV）且更低的倒置 Hypervolume（inverted HV），表明在相同公平性水平下能获得更高准确率，或在相同准确率下实现更低公平偏差，且无须重新训练即能在不同公平阈值下得到优良权衡。

**⚠️ 局限性**

局限性：目前仅支持二元敏感属性与二元标签；不覆盖所有公平性定义（如等化机会）；在多类别或多敏感属性交叉群组等更复杂场景下的扩展仍需进一步研究。

---

## 322. BiDeMem: Bidirectional Degradation Memory for Explainable Image Restoration

**arXiv ID:** 2606.28112 | [PDF](https://arxiv.org/pdf/2606.28112v1)

**作者:** Xinrui Wu `[一作]` (University of Electronic Science and Technology of China), Lichen Huang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种双向降解记忆框架BiRank，通过从降解特征与输入统计构建查询检索top‑k记忆槽，并将相同槽身份用于恢复路径和训练时的前向降解解释路径，实现可验证的图像恢复。

**💡 创新点**

创新点在于将记忆槽的选择与恢复和降解解释共享，构建可审计的降解先验；通过控制实验和干预探针验证记忆槽的可解释性和干预敏感性；提出compact top‑k记忆和bidirectional一致性训练。

**🔧 技术方法**

采用FiLM条件化、slot attention式记忆网络、前向降解一致性损失、rank loss以及多头注意力等技术，并在控制背骨上使用简化版网络。

**📊 数据集**

训练使用BSD400、WED（含不同噪声水平）、5000+雨景对和5500+雾景对；测试在BSD68、Urban100、Rain100L、Haze4K等；还评估了JPEG、未见噪声和混合降解。

**📈 对比分析**

与基线+宽度扩展、仅纠正头、dense query FiLM、static global prior等对照；BiRank在控制实验中提升约0.26 dB PSNR，干预实验显示更高的错误先验敏感性；在未见/混合降解上平均提升约0.2 dB；低数据适配下提升≈0.7 dB。

**⚠️ 局限性**

模型参数紧凑但计算量增加，延迟提升；记忆槽设计对不同基骨可能不最优；未必能在所有降解混合或更大规模数据中保持优势；主要为验证导向，而非最终通用恢复基座。

---

## 323. Cross-view Multimodal Vision-Based Assessment Framework for Traditional Chinese Medicine Rehabilitation Training

**arXiv ID:** 2606.28104 | [PDF](https://arxiv.org/pdf/2606.28104v1)

**作者:** Francis Xiatian Zhang `[一作]` (Durham University), Hubert P. H. Shum `[通讯]` (Durham University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多视角视觉与姿态融合的跨视角多模态动作质量评估框架CME-AQA，用于中医康复训练的动作质量评估。

**💡 创新点**

创新点包括：①跨视角视觉–姿态交叉注意力融合模块，②多尺度视角对齐训练策略；同时首次公开了双视角针灸（TCM-AQA61-A）和推拿（TCM-AQA61-T）评估数据集。

**🔧 技术方法**

技术实现采用ResNet提取视觉特征、MediaPipe手部姿态、视觉–姿态交叉注意力（AVPF）、多尺度视角对齐（MVA）以及Transformer时序自注意力模块。

**📊 数据集**

使用的数据集为TCM-AQA61-A（61名医学生针灸）、TCM-AQA61-T（61名医学生推拿）双视角同步视频与专家评分；并在CPR多视角数据集上验证迁移性能。

**📈 对比分析**

与STGCN、STNN、FineParser、Uni-FineParser、PGT、PHI等基线对比，关键指标（针深度、快速插针、推拿频率等）均提升10%+F1或MAE下降至3.12 s/0.94 s，整体性能优于或相当于最强基线。

**⚠️ 局限性**

局限性：依赖姿态估计准确性，对姿态噪声敏感；缺乏对物体语义建模，RGB缺失时性能下降；数据集仅覆盖医学生，缺少多样化专业水平和极端误差场景。

---

## 324. Fast and Feasible: Permutation-based Constrained Reranking for Revenue Maximization

**arXiv ID:** 2606.28059 | [PDF](https://arxiv.org/pdf/2606.28059v1)

**作者:** Svetlana Shirokovskikh `[一作]` (Avito), Egor Samosvat `[通讯]` (Avito)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种轻量级的置换式重排算法 PermR，旨在在满足多项业务约束的前提下提升电商平台的收入。

**💡 创新点**

创新点在于将收入最大化与约束满足结合成整数线性规划，并通过邻接项交换的迭代方式实现近似最优且满足约束的重排，显著降低了实时推理时延。

**🔧 技术方法**

技术手段包括整数线性规划建模、MOSEK/HiGHS ILP 求解器、基于位置的关注模型、邻接项置换的 PermR 算法，以及基准对照的遗传算法。

**📊 数据集**

实验使用了大型分类平台 Goods 目录（约 200M 条商品）日志数据，涵盖 27,000 条查询（离线实验）和 56M 条查询（14 天 A/B 测试）。

**📈 对比分析**

通过与 ILP 求解器和遗传算法的离线对比，PermR 在 750 次迭代、0.05 秒推理时延下实现了约 63% 的 ILP 收入提升；在线 A/B 测试中，整体收入提升约 2%（个别子类最高 6%），且所有业务约束均得到保持。

**⚠️ 局限性**

局限性包括 ILP 求解器在实时环境中的耗时过长，遗传算法收益低，PermR 虽逼近最优但仍不等同于精确解，并且对迭代次数的调优以及约束定义相对当前排序的依赖可能影响其在不同场景的泛化能力。

---

## 325. SBridge: Identifying Source-to-Binary Function Similarity via Cross-Domain Control Block Matching

**arXiv ID:** 2606.28058 | [PDF](https://arxiv.org/pdf/2606.28058v1)

**作者:** Heedong Yang `[一作]` (Korea University), Seunghoon Woo `[通讯]` (Korea University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过跨域控制块匹配识别源代码函数与二进制函数相似性的精确方法。

**💡 创新点**

创新点在于基于控制块的函数匹配机制，将函数拆分为条件、循环等语义单元，利用控制块作为跨域表示来精确测量相似性，克服了函数内联与去符号化的挑战。

**🔧 技术方法**

使用了控制块分段、跨域控制块匹配技术，并结合相似度度量算法实现源代码与二进制之间的相似性评估。

**📊 数据集**

评估采用了来自BinKit的3,904个真实世界C/C++二进制文件。

**📈 对比分析**

与现有依赖字符串或全函数结构相似度的基线方法相比，召回率@1提升至75.13%（基线43.31%），召回率@5提升至80.98%（基线50.2%），显示显著性能提升。

**⚠️ 局限性**

局限性包括仅在C/C++二进制上验证，可能对其他语言、不同架构或高度优化/混淆二进制的适用性未知；计算成本与多态/动态加载函数匹配等情况未充分评估。

---

## 326. The ARDoCo Tool Landscape: REST API, TraceView, and TraceViz for Architecture Traceability

**arXiv ID:** 2606.28064 | [PDF](https://arxiv.org/pdf/2606.28064v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 327. CrossLangFuzzer: Differential Testing of Cross-Language JVM Compilers

**arXiv ID:** 2606.28132 | [PDF](https://arxiv.org/pdf/2606.28132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 328. MultiHashFormer: Hash-based Generative Language Models

**arXiv ID:** 2606.28057 | [PDF](https://arxiv.org/pdf/2606.28057v1)

**作者:** Huiyin Xue `[一作]` (University of Sheffield), Nikolaos Aletras `[通讯]` (University of Sheffield)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 MultiHashFormer，利用多哈希签名实现无词表瓶颈的生成式语言模型；

**💡 创新点**

创新在于通过多重哈希签名消除哈希碰撞，实现自回归生成；

**🔧 技术方法**

技术包括：多哈希函数+MMH3哈希、门控组合嵌入、Cascade Decoder、Transformer 解码器；

**📊 数据集**

预训练数据：English FineWeb-Edu（10B/100B tokens），多语言FineWeb2（6B）用于词表扩展；

**📈 对比分析**

对比标准 Transformer LM（及加层版），在100M、1B、3B规模下，多哈希模型在11项任务中多数超过基线；在词表扩展后，保持参数不变仍优于增加30M参数的基线；

**⚠️ 局限性**

局限：仅评估到3B模型，缺少多次随机种子验证，且在较小模型上表现不显著。

---

## 329. Can LLMs Judge Better Than They Generate? Evaluating Task Asymmetry, Mechanistic Interpretability and Transferability for In-Context QA

**arXiv ID:** 2606.28050 | [PDF](https://arxiv.org/pdf/2606.28050v1)

**作者:** Sambaran Bandyopadhyay `[一作]` `[通讯]` (Adobe Research), Sambaran Bandyopadhyay (Adobe Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在受控的上下文问答框架下，对LLM的生成与自我评估准确率进行直接比较，并通过注意力分析与LoRA微调实验探究两者之间的不对称机制。

**💡 创新点**

①首次在同一上下文条件下对生成和自评进行直接对比；②利用last‑token注意力揭示评估对上下文关注显著降低的原因；③通过LoRA跨任务转移实验表明生成与评估不共享相同的参数结构。

**🔧 技术方法**

使用Llama‑3.1‑8B‑Instruct、GPT‑4o‑mini、Oracle LLM、零-shot提示、last‑token attention采样、LoRA低秩微调以及硬负样本生成等技术。

**📊 数据集**

SQuAD 2.0、DROP、HotpotQA、MuSiQue四个英文QA基准（仅使用验证集）。

**📈 对比分析**

采用生成准确率(GA)、自评准确率(EA)及差值Δ进行比较。结果显示大部分数据集Δ<0，说明自评不如生成；MuSiQue在多跳情况下Δ>0。注意力分析显示评估对上下文关注3–5倍降低，LoRA实验进一步证实两任务的参数不共享。

**⚠️ 局限性**

局限性包括：只分析短答案的第一token注意力，未覆盖长文本生成；仅在Llama内部可见，GPT‑4o‑mini不适用；不考虑不同提示或多语言情境；MuSiQue正向差异机制尚未完全解释。

---

## 330. DG^VoiC: Speaker Clustering for Fraud Investigation under Real Call-Centre Conditions

**arXiv ID:** 2606.28048 | [PDF](https://arxiv.org/pdf/2606.28048v1)

**作者:** Muhammad Shakeel Akram `[一作]` (Aston University), Karishma Jaitly `[通讯]` (Domestic & General)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了 DGVoiC 语音聚类框架，用于保险欺诈调查中对呼叫中心录音进行客户身份验证和跨案例声纹链接；

**💡 创新点**

创新点在于将敏感信息匿名化、语音预处理、滑动窗口声纹嵌入、余弦相似度阈值聚类以及风险评分等技术整合成一个在真实呼叫中心条件下可直接使用且误报率低的完整流程；

**🔧 技术方法**

使用了 WhisperX+RoBERTa NER 进行文字和音频敏感信息匿名化，Resemblyzer 去除静音，ECAPA‑TDNN 提取声纹嵌入，滑动窗口+平均池化，余弦相似度阈值聚类，以及基于阈值的声纹风险评分；

**📊 数据集**

数据集为 121 条真实保险呼叫中心录音（时长 2.75–34.6 分钟），在人工标注后挑选 56 条（22 个聚类）作为验证集；

**📈 对比分析**

通过与人工聚类对比，使用 AMI、ARI、完整度、同质性、V-Measure 等聚类一致性指标，最佳配置达到 AMI 96%、ARI 95%、完整度 98%、同质性 100%、V-Measure 99%，准确率 95%、F1 0.96；平均处理时延约 10.08 秒/录音；

**⚠️ 局限性**

局限性包括对背景噪声和说话人音色变化的鲁棒性不足；聚类阈值需经验调整；仅聚焦声纹，未结合文本或行为特征；样本规模有限，需进一步验证。

---

## 331. GTI-mSEMP Framework : A Proposed Framework to Stimulate Malware Propagation with Inclusion of Attacker-Defender Strategy

**arXiv ID:** 2606.28079 | [PDF](https://arxiv.org/pdf/2606.28079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 332. AI-Driven Synthesis for High-Tech System Design: Automating Innovation

**arXiv ID:** 2606.28126 | [PDF](https://arxiv.org/pdf/2606.28126v1)

**作者:** Luuk Oerlemans `[一作]` (Eindhoven University of Technology), Theo Hofman `[通讯]` (Eindhoven University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了计算设计合成（CDS）框架，通过深度学习与生成式 AI 自动化高科技系统的拓扑与尺寸设计，并用电驱动系统与空间尺寸化两大案例进行验证。

**💡 创新点**

创新点在于将强化学习与非线性规划结合成双层“RL‑NLP”框架，克服单纯 RL 不能满足严格物理约束且缺乏中间奖励的问题；同时提出最大不相交球分解（MDBD）几何抽象方法，将 NP‑hard 的空间布局问题转化为可微分的连续优化，从而实现高效的自动化尺寸化。

**🔧 技术方法**

技术主要包括：强化学习（policy network、Q‑network 等）、非线性规划求解器、生成式深度网络（Deep Q‑Network）、大语言模型提取组件信息、MDBD几何抽象与梯度优化、以及代理学习（Actor‑Critic）与传统混合优化的结合。

**📊 数据集**

数据集来源于工程部件库（如齿轮、轴、机电元件）与 CAD 模型（电机-齿轮箱组装），以及用于验证的解析基准（1×1×2 立方体布局），无公开公开数据集，但可视为行业专有部件与几何数据。

**📈 对比分析**

在电驱动系统案例中，RL‑NLP 与蛮力搜索相比，评估时间缩短约 3 个数量级，预测误差仅 2%；在空间尺寸化案例中，对解析基准的验证显示体积与布线长度误差仅 0.6%–2%，说明方法在精度与效率上均优于传统手工 CAD 迭代。

**⚠️ 局限性**

局限性包括：仍需高质量的部件知识库与物理模型，RL 训练与 NLP 求解器对计算资源要求高；MDBD 抽象虽然加速优化，但对极复杂几何的逼真度仍有限；方法在跨领域迁移时需重新构建知识提取与约束模型。

---

## 333. When One Adapter Speaks for Many: Discovering Low-Rank Redundancy in Continual Fine-Tuning

**arXiv ID:** 2606.28117 | [PDF](https://arxiv.org/pdf/2606.28117v1)

**作者:** Tanguy Dieudonné `[一作]` (ETH Zurich), Thomas Hofmann `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了LoRA在持续学习中的低秩冗余，提出可学习门控机制动态决定是否为新任务引入适配器。

**💡 创新点**

引入可微分的二值门控与两阶段训练，既保持稳定性与可塑性，又显著降低适配器数量。

**🔧 技术方法**

采用LoRA低秩适配器、SD‑LoRA幅度‑方向分解、Gumbel‑Sigmoid与Straight‑Through Estimator实现门控，并通过两阶段训练分离特征学习与结构剪枝。

**📊 数据集**

在CIFAR‑100、ImageNet‑A（对抗样本）与ImageNet‑R（风格渲染）三大增量学习基准上进行评估。

**📈 对比分析**

与SD‑LoRA等基准方法对比，平均准确率相当或略优，参数增量减少20–70%，遗忘率相近或更低。

**⚠️ 局限性**

对任务顺序敏感；门控训练需加入噪声，且在极度稀疏配置下可能导致性能下降。

---

## 334. STAG: Spatio-temporal Evolving Structural Representation of Action Units for Micro-expression Recognition

**arXiv ID:** 2606.28083 | [PDF](https://arxiv.org/pdf/2606.28083v1)

**作者:** Nandani Sharma `[一作]` (Indian Institute of Technology Mandi), Dinesh Singh `[通讯]` (Indian Institute of Technology Mandi)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一的动态ROI–AU耦合空间‑时间网络 STAG，用于微表情识别。

**💡 创新点**

创新点在于：① 用 AU 指导的自适应图结构实现面部区域动态连接；② 通过双分支（E‑GAT + Transformer）并行建模空间与时间，并用双向交叉注意力实现信息互补；③ 采用基于光流的多尺度运动提取和焦点损失提升少样本鲁棒性。

**🔧 技术方法**

使用技术包括：光流计算 + 位置编码；增强图注意网络 (E‑GAT)；Transformer 编码器；双向交叉注意力模块；AU 辅助嵌入；焦点损失；数据增强（MixUp、CutMix、TTA）。

**📊 数据集**

数据集涵盖六大微表情基准：CASME II、4DME、DFME、NaME、SAMM、SMIC‑HS，分别在 3/5/7 类标签协议下进行评估。

**📈 对比分析**

在 LOSO、Group‑K‑Fold、Stratified‑K‑Fold 等严格交叉验证中，STAG 在 3 类协议下 UF1/UAR 均超过 90%（如 NaME 0.898/0.898），在 5 类协议下亦保持 70% 以上；与多种 SOTA 方法对比，STAG 在大部分数据集获得最佳或竞争性成绩，且跨数据集迁移性能显著提升。

**⚠️ 局限性**

局限性：① 对 SMIC‑HS 等低强度、样本少的数据集表现仍偏弱；② 需要手工标注 AU 作为辅助信息，若无则性能下降；③ 模型结构较为复杂，训练成本高；④ 在 5 类跨域评估中仍受类别不平衡与域差异影响。

---

## 335. PAC-Bayesian Certificates for Quadratic Closed-Loop Control

**arXiv ID:** 2606.28281 | [PDF](https://arxiv.org/pdf/2606.28281v1)

**作者:** Domagoj Herceg `[一作]` `[通讯]` (Eindhoven University of Technology), Domagoj Herceg (Eindhoven University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于PAC‑Bayes理论的有限时域二次闭环控制安全性证明方法，利用系统层级综合（SLS）参数化将随机化后验分布直接作用于可行闭环响应。

**💡 创新点**

创新点包括：①在SLS框架下将控制成本视为扰动轨迹和自由响应坐标的二次函数；②针对高斯扰动给出精确的一侧Chernoff变换和可计算的敏感性上界；③证明随机后验安全性可迁移到确定性均值部署；④构建数据驱动的后验安全性上界，消除对先验扰动协方差的依赖。

**🔧 技术方法**

所用技术主要有PAC‑Bayes推导、系统层级综合（SLS）参数化、Gaussian Chernoff变换、二次风险的凸性分析、数据驱动的协方差上界以及Julia自动微分实现。

**📊 数据集**

实验使用合成的双积分器（double integrator）系统，生成不同数量的扰动轨迹样本进行训练和评估。

**📈 对比分析**

与传统岭回归（ridge）等基线方法比较，实验显示在低样本量下，后验优化并确定性均值部署的控制器在留存集成本和闭环敏感性方面均优于基线，随着样本增大，所有方法表现趋同。

**⚠️ 局限性**

局限性包括：仅适用于无硬约束的线性有限时域系统；对扰动的分布要求为高斯或有界；未考虑模型不确定性或MPC约束；后验分布假设为正态或易于计算的形式。

---

## 336. SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation

**arXiv ID:** 2606.28276 | [PDF](https://arxiv.org/pdf/2606.28276v1)

**作者:** Nadun Ranawaka `[一作]` (NVIDIA), Yuke Zhu `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种端到端的自动化管道，将单段现实视频转换为交互式、可直接用于机器人仿真的数字孪生，并进一步生成对象、场景和任务的多样化“数字亲戚”，支持在仿真中评估和训练机器人策略。

**💡 创新点**

创新点包括：① 模块化、可插拔的构造流程，使新技术能随时替换；② 自动化生成三种维度的亲戚（对象、场景、任务），显著扩展训练数据多样性；③ 在同一体系下完成仿真评估与零样本 sim‑to‑real 迁移，提升评估准确性与实用性。

**🔧 技术方法**

主要技术包括：基于视觉基础模型的深度估计、场景理解与对象分割；3D 重建与生成模型（V_mesh、FoundationPose、CoACD 等）用于生成和对齐物体网格；可动部件检测与关节参数生成；物理属性注释与碰撞几何构建；背景重建使用 3D Gaussian Splat；仿真集成使用 PyBullet/IsaacLab；子任务评估与亲戚生成使用规则与语义空间约束。

**📊 数据集**

使用单个现实视频（桌面级布局）作为输入；实验数据来源于 7 个多步操控任务、5 种策略架构，并在 YAM 与 DROID 两个机器人平台上进行真实实验；还利用从生成的亲戚中采样的仿真轨迹进行训练。

**📈 对比分析**

与现有基准 PolaRiS 对比，仿真评估的 Pearson 相关系数提升 0.59，平均最大排名违背 MMRV 仅为 0.018；在 zero‑shot sim‑to‑real 训练中，基于生成亲戚的策略在 YAM/DROID 上成功率达到 99%–100%；使用对象、场景、任务亲戚分别提升任务成功率 17%、21%、40%，并显著提高对未见对象、布局和任务的泛化能力。

**⚠️ 局限性**

局限性：依赖当前的视觉基础模型，受其误差和失败模式影响；假设输入为桌面级、平面布局，难以处理多层或非平面环境；对复杂动态场景、遮挡严重的对象重建仍有挑战；需要较高质量的视频与相机标定，背景重建对纹理稀缺区域表现不佳。

---

## 337. Agent-Native Immune System: Architecture, Taxonomy, and Engineering

**arXiv ID:** 2606.28270 | [PDF](https://arxiv.org/pdf/2606.28270v1)

**作者:** Bo Shen `[一作]` (Novo Ordo for AI), Dehui Li `[通讯]` (Novo Ordo for AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并构建了一套面向自主智能体的内生免疫体系（Agent‑Native Immune System, ANIS），包括六层免疫塔、统一的病毒/疫苗分类、Harness 三元组和持续免疫学习（CIL）循环，旨在保障智能体在运行时的安全、健康、秩序与演化。

**💡 创新点**

创新点：
1) 把免疫学概念直接映射到智能体架构，形成六层免疫塔与Barrier Immunity等新层次；
2) 设计了病毒/疫苗统一本体，区分非参数（规则/提示）与参数（Steering Vector、LoRA）疫苗；
3) 将 Harness Engineering 的 Meta‑Harness、Auto‑Harness 与 Self‑Harness 重构为免疫三元组，形成闭环的持续免疫学习；
4) 提出了Autoimmunity Rate、Ecological Order Coefficient 等新评估指标，并在SIR模型下分析多智能体免疫动力学。

**🔧 技术方法**

主要技术手段包括：
- 规则引擎、签名检测与行为基线（L2）；
- 预训练参数微调（LoRA）与Steering Vector（L3）；
- 机器学习驱动的疫苗生成与验证（Meta‑Harness/Auto‑Harness）；
- 自监督的免疫记忆与疫苗分发协议（L5）；
- 硬件信任根（TEE/TPM）与跨层归档。

**📊 数据集**

论文为概念性框架，未在公开数据集上进行实验；未给出具体使用的数据集。

**📈 对比分析**

对比方法：文中对比了传统外围防御、模型对齐和 ANIS 的三种安全范式，指出 ANIS 能覆盖已知与未知威胁、实现持续演化并支持多智能体协同。由于缺乏实验结果，性能指标（如准确率、Autoimmunity Rate 等）仅以理论和模拟方式给出；作者承认实验验证工作仍在进行。

**⚠️ 局限性**

局限性：
- 缺乏大规模实证验证；
- 运行时监控与疫苗更新带来的计算开销；
- Autoimmunity Rate 与阈值设定的权衡尚未有正式方法；
- 多模态与跨平台标准化未实现；
- 免疫压力可能导致路径进化与攻击者的“抗药性”问题。

---

## 338. Learning Topology-Aware Representations via Test-Time Adaptation for Anomaly Segmentation

**arXiv ID:** 2606.28268 | [PDF](https://arxiv.org/pdf/2606.28268v1)

**作者:** Ali Zia `[一作]` (La Trobe University), Wei Xiang `[通讯]` (La Trobe University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于拓扑的测试时自适应（TopoTTA）框架，用于工业缺陷的二维/三维像素级异常分割。

**💡 创新点**

创新点包括：①将持久同调（persistent homology）与多层立方体复形滤波结合，直接从异常分数图生成结构一致的伪标签；②Euler‑aware 交互融合（EAI）保证掩模的拓扑一致性；③在测试时训练轻量级对比编码器（PCES）以对特征空间进行自适应，对比学习不依赖源域标签；④整体模块化、模型无关，可与任意现有异常检测/分割基线配合。

**🔧 技术方法**

核心技术包括持久同调、立方体复形多级滤波、Euler特征融合、对比学习、无监督的测试时训练（TTT）以及传统的阈值化与后处理。

**📊 数据集**

在六大工业异常数据集上评估：MVTec AD、VisA、Real‑IAD、MVTec 3D‑AD、AnomalyShapeNet 以及 MVTec LOCO，用于逻辑异常检测的验证。

**📈 对比分析**

与传统阈值化（THR）和现有的测试时训练方法（TTT4AS）对比，TopoTTA 在 F1、IoU 等分割指标上平均提升约 15%（2D）和 10%（3D），在复杂几何或结构变化的缺陷上最高可提升 20%+，同时保持更好的边界连贯性和更少的噪声片段。

**⚠️ 局限性**

局限性包括：①对上游异常分数图的质量高度依赖，若分数图噪声过大或对比度低，拓扑信号会失效；②需要手工设定 K、L、β 等超参数，虽不直接是阈值但仍影响性能；③持久同调计算目前使用 CPU，造成一定的运行时开销；④框架针对静态图像/体素，未考虑时序一致性或视频/医学序列；⑤无法将拓扑信号梯度直接传播到特征提取器，导致最终仍是后处理方式。

---

## 339. RSICCLLM: A Multimodal Large Language Model for Remote Sensing Image Change Captioning

**arXiv ID:** 2606.28266 | [PDF](https://arxiv.org/pdf/2606.28266v1)

**作者:** Yelin Wang `[一作]` (State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Zitong Yu `[通讯]` (Dongguan Key Laboratory for Intelligence and Information Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出RSICCLLM，一个针对遥感图像变化描述的多模态大模型后训练框架，并通过差异感知监督微调和双负样本偏好优化提升变化感知与描述能力。

**💡 创新点**

创新点在于：①基于二时遥感图像与二值变更掩模的指令数据生成范式，构建规模达4万条的RSICI指令集；②差异感知监督微调结合中央差分卷积与Hough变换提取时空差异；③双负样本偏好优化（信息过滤+替换策略）生成RSICP偏好数据。

**🔧 技术方法**

使用Qwen-VL-Max、Qwen2.5-VL-7B作为基础模型，加入中央差分卷积、Hough变换、交叉注意力融合，结合自监督的差异提取与DPO偏好优化。

**📊 数据集**

使用LEVIR-CD、SYSU-CD、CDD、S2Looking等遥感变化数据集生成约4万条RSICI指令集，以及20k条RSICP偏好样本。

**📈 对比分析**

在自建的RSICI基准上与多款7B-240B大模型对比，RSICCLLM-7B在BLEU-4、ROUGE-L、SBS等指标上分别提升约1.1点、1.0点和1.4点，远优于更大模型；在外域测试集同样保持领先。

**⚠️ 局限性**

局限在于：对极端光照/视角变化仍易产生误判；依赖人工审核生成的指令集质量；模型仍受限于单一视觉编码器的表达能力，难以处理极细粒度多尺度变化。

---

## 340. Unleashing Infinite Motion: Scaling Expressive Quadrupedal Motion via Generative Video Priors

**arXiv ID:** 2606.28237 | [PDF](https://arxiv.org/pdf/2606.28237v1)

**作者:** Youzhi Liu `[一作]` (Amap, Alibaba Group), Ziqiao Li `[通讯]` (Amap, Alibaba Group)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Uni‑Mo 自动化管线，将自然语言提示转化为可执行的四足机器人动作，并在 Unitree Go2 上实现部署，最终公开了包含 7,488 条富表达性的动作轨迹的 Quad‑Imaginarium 数据集。

**💡 创新点**

创新点包括：① 完全去除动物中介，直接在机器人上生成动作；② 设计 Identity Consistency Loss，消除视频扩散模型中的身份漂移；③ 通过 LLM 与视频扩散结合，自动生成高质量动作；④ 公开大规模可执行四足动作数据集，供后续研究使用。

**🔧 技术方法**

使用技术包括：大语言模型生成动作提示；Wan‑2.2 视频扩散模型（LoRA 微调 + Identity Consistency Loss）；DINOv2 进行身份特征匹配；ViTPose 提取 2D 关键点；URDF‑Anchored 运动回推；CLIP 语义门控；PPO 轨迹跟踪策略；MuJoCo 物理仿真；以及多种评价指标（FID、FVD、VLM、CLIP相似度）。

**📊 数据集**

使用的数据集有：① 通过管线生成的 Quad‑Imaginarium（18.5 小时、7,488 条动作）；② 为微调 Wan‑2.2 预先收集的机器人视频数据；③ 用于对比的公开数据集 QuadFM、T2QRM。

**📈 对比分析**

实验对比方法：在相同提示集下比较 Wan‑Base、Wan‑FT、Wan‑FT+IC 三种模型，指标包括 FID、FVD、Identity Consistency、Naturalness、Alignment；结果表明加上 IC 后所有指标均提升；提取后 68.1% 轨迹通过门控，仿真跟踪成功率 97.6%，真实机器人部署成功率 96.7%；相较于 QuadFM/T2QRM，Quad‑Imaginarium 在 kinematic 多样性和动态表现上更优。

**⚠️ 局限性**

局限性：① 视频扩散模型语义控制不精准，需多次生成并人工校验，扩展成本主要在人工审核；② 仅能处理机器人始终可见的场景，无法支持视角丢失或大幅位移的动作；③ 目前的管线侧重于站姿表情动作，缺乏大范围平移行走的支持。

---

## 341. Govern the Repository, Not the Agent: Measuring Ecosystem-Level Risk in AI-Native Software

**arXiv ID:** 2606.28235 | [PDF](https://arxiv.org/pdf/2606.28235v1)

**作者:** Daniel Russo `[一作]` `[通讯]` (Aalborg University), Daniel Russo (Aalborg University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了自主编码代理在共享仓库中的集成摩擦，并用多层模型量化其在生态系统层面的不可还原性。

**💡 创新点**

提出并验证“不可还原性”指标，将集成摩擦的仓库级方差作为出现涌现的统计证据，并与人工提交对比，证明问题源于生态系统而非单个代理。

**🔧 技术方法**

采用多层线性/混合模型、ICC估计、逻辑回归、泊松‑对数正态模型以及聚类自助法和极大似然区间等统计方法。

**📊 数据集**

基于AIDev公开的93万+代理拉取请求数据，结合GitHub API提取仓库属性，使用AgenticFlict重放冲突信息，并匹配同仓库的人类拉取请求做基线。

**📈 对比分析**

通过在不同控制层级下比较ICC（最高约0.61），并在相同仓库中对比代理与人工ICC差值，显示代理提交的仓库级摩擦约为人类的两倍，控制变量后差异显著。

**⚠️ 局限性**

研究仅为观测性且受限于公开仓库与五款代理，未捕捉所有可能的仓库属性（如治理、应用域），且因代理间交互仅可部分观察，结果为必要性证据非因果证明。

---

## 342. Physics-Informed Neural Network with Transfer Learning for State Estimation in Lithium-Ion Batteries using the Single Particle Model with Electrolyte

**arXiv ID:** 2606.28220 | [PDF](https://arxiv.org/pdf/2606.28220v1)

**作者:** Gift Modekwe `[一作]`, Qiugang Lu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于单颗粒模型（SPMe）的物理信息神经网络（PINN），并将迁移学习引入其中，以实现不同锂离子电池的状态估计与关键电化学参数识别。

**💡 创新点**

首次将迁移学习与SPMe-PINN相结合，预训练模型可共享电化学知识，快速适配不同化学、几何或工作条件的电池，并通过可辨参数实现对固相扩散系数等关键参数的识别。

**🔧 技术方法**

采用 Physics-Informed Neural Network (PINN) 架构，在损失函数中嵌入SPMe的偏微分方程、边界与初始条件以及电压约束；使用迁移学习技术对预训练权重进行冻结与微调；利用 PyBaMM 生成的数值解进行验证。

**📊 数据集**

使用 PyBaMM 生成的三组电池数据集：B1（LG M50 圆柱体）、B2（Kokam 软包）和 B3（LFP），分别在 1 C 与 0.5 C/1.2 C 放电条件下进行训练与测试。

**📈 对比分析**

通过与 PyBaMM SPMe 数值解的终端电压对比，RMSE 在 8.1 × 10⁻⁴（B1）范围内；跨域（B2）和跨化学（B3）迁移后误差仍保持 < 10⁻³；内部状态预测（固相浓度）与参考解高度吻合，参数估计误差 < 0.5%。

**⚠️ 局限性**

局限性包括仅在仿真数据上验证，未使用真实实验数据；未考虑温度变化与电池老化效应；迁移学习中冻结层的选择仍基于经验，缺乏系统性优化。

---

## 343. Non-Linear Strategic Classification Made Practical

**arXiv ID:** 2606.28204 | [PDF](https://arxiv.org/pdf/2606.28204v1)

**作者:** Jack Geary `[一作]` (University of Edinburgh), Henry Gouk `[通讯]` (University of Edinburgh)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了利用拉格朗日对偶近似策略性最佳响应，并借助隐函数定理求得总梯度，从而训练出更鲁棒的机器学习模型。

**💡 创新点**

创新点在于：①把代理人的目标转化为受限优化，使用拉格朗日对偶提供可求解的近似；②提出 Agreement 指标评估响应逼近质量；③设计基于总梯度（TGD）的训练算法。

**🔧 技术方法**

核心技术包括拉格朗日对偶、Karush‑Kuhn‑Tucker 条件、隐函数定理、梯度下降、总梯度（Indirect Derivative）以及 REGD 与 TGD 的实现。

**📊 数据集**

实验使用了 TALENT 子集、Housing、Give Me Some Credit、Bank Customer Churn、Default of Credit Card、Employee、GMSC、Houses、HR Analytics、mobile c36、statlog、Water Quality 等实测数据集，并在 Twin Moons、Ball‑and‑disk 等人工合成数据上进行验证。

**📈 对比分析**

通过比较被游戏点比例、Agreement 指标和策略准确率，发现与传统 Gradient Response 和 REGD 相比，TGD 在绝大多数数据集上策略准确率更高，证明了所提方法的有效性。

**⚠️ 局限性**

局限性在于仅验证了线性和 MLP 这两类模型，未涵盖更复杂的非线性结构；理论假设相对宽松，但实际适用范围仍需进一步扩展。

---

## 344. Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models

**arXiv ID:** 2606.28153 | [PDF](https://arxiv.org/pdf/2606.28153v1)

**作者:** Yanchen Yin `[一作]` (Beijing University of Posts and Telecommunications), Linghui Li `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过机制解释，探究大型语言模型在越狱攻击下安全特征的抑制与保持。

**💡 创新点**

创新点在于识别两类注意力头：被攻击抑制的 ACH 与保持安全激活的 SAH，揭示攻击通过抑制少量头即可绕过拒绝。

**🔧 技术方法**

使用反向投影拒绝方向、核密度估计与重叠系数分隔注意力头，并进行干预与 token 级归因。

**📊 数据集**

采用 Llama‑3‑8B‑Instruct、Llama‑2‑7B‑Chat 与多种公共攻击与安全数据集（如 Alpaca、HarmBench、WildJailbreak 等）。

**📈 对比分析**

与现有安全检测模型（WildGuard、LlamaGuard 等）对比，训练‑free 检测在 10 组基准上取得与专门训练模型相当甚至更优的宏 F1，且对攻击具有鲁棒性。

**⚠️ 局限性**

局限在仅分析注意力层、使用单一拒绝方向、样本稀缺、需白盒访问等。

---

## 345. Which Nash Equilibrium? Solver-Dependent Selection on Zero-Sum Nash Polytopes

**arXiv ID:** 2606.28308 | [PDF](https://arxiv.org/pdf/2606.28308v1)

**作者:** Luis Leal `[一作]` `[通讯]`, Luis Leal

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了两人零和游戏中，当 Nash 集合为凸多面体而非唯一点时，不同求解器会系统性地选择不同的 Nash 点。

**💡 创新点**

创新点在于首次证实正则化最后一次迭代方法倾向于最大熵（即信息投影）成员，而这一选择由算法本身决定，而非随机种子或迭代预算。

**🔧 技术方法**

采用了表格精确求解器、CFR、CFR+、Fictitious Play、Magnetic Mirror Descent（MMD）以及 R-NaD 等算法，并结合熵正则化、信息投影理论进行分析。

**📊 数据集**

使用了六个手工构造的解析 Nash 集合游戏（包含对称基准、非对称游戏、一个 2‑D 多面体以及 Kuhn 纸牌游戏）以及 180 个随机生成的非对称矩阵游戏。

**📈 对比分析**

通过比较每个游戏求解器的选择坐标、策略熵、可利用性（exploitability）以及与 R-NaD 结果的 Jensen‑Shannon 距离进行量化。结果显示，正则化方法在 100% 的随机样本中达到最大熵成员，而回归平均方法在 94% 时低于最大熵，且两者差距显著（p<10⁻²⁷）。

**⚠️ 局限性**

主要限制包括：仅在小规模、表格、零和游戏中验证，未涉及采样或函数近似；Kuhn 是唯一的非矩阵实验；正则化方法对尺度敏感，且 I‑投影命题尚未正式证明。

---

## 346. CacheMPC: Certified Cached Model Predictive Control for Quadruped Locomotion

**arXiv ID:** 2606.28300 | [PDF](https://arxiv.org/pdf/2606.28300v1)

**作者:** Nimesh Khandelwal `[一作]` (Indian Institute of Technology Kanpur), Mangal Kothari `[通讯]` (Indian Institute of Technology Kanpur)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

针对四足机器人运动学中Model Predictive Control（MPC）的实时计算瓶颈，提出了一种可认证的缓存MPC框架（Certified CacheMPC），实现了基于局部性敏感哈希（LSH）的轨迹缓存与后验最优性与可行性检验。

**💡 创新点**

创新点包括：① 引入可验证的Lagrangian dual-gap后验证书，对缓存的控制序列提供上界子最优性保证；② 将缓存、证书检验与时间预算调度相结合，实现了严格的实时执行保障；③ 在四足机器人上首次将该框架与NVIDIA Orin NX硬件部署并评估。

**🔧 技术方法**

使用的技术主要有：凸MPC（SRBD模型）→QP求解（HPIPM）；局部性敏感哈希（LSH）进行近似邻居检索；基于KKT条件的双重对偶间隙计算做后验证书；敏感性增量修正与活集过滤；时间预算控制与回退机制。

**📊 数据集**

数据集与实验：在MuJoCo仿真环境下，对Unitree Go2进行2,038次冷启动实验（包含速度巡航、侧向推力阶梯、斜坡与台阶等多种地形），并在真实硬件（NVIDIA Orin NX）上进行约90秒的自由行走和手工侧向推力测试。

**📈 对比分析**

对比方法：与无缓存的标准MPC（Convex）在四种缓存配置（Cache-NoCert、Cache-Cert、Cache-Full）以及两种下层架构（WBC、MRT）进行比较。仿真中，未加证书的缓存实现了25×的中位求解速度提升；加证书后速度提升约3–4×，但仍显著快于无缓存。跟踪误差在±14%范围内无显著差异。硬件上，未加证书缓存实现18.7×速度提升且无跌倒，带证书版本在硬件上因拒绝率高导致速度低于无缓存。

**⚠️ 局限性**

局限性包括：① 证书采用最松散的λ=0版本，未利用缓存的对偶信息；② 仅在单一四足平台与单一摩擦/地形条件下验证；③ 侧向推力方向单一，未覆盖更广泛的失稳边界；④ 样本量不足以在统计上证明安全提升；⑤ 只与无缓存Convex baseline对比，未涉及热启动、KD-tree或学习式近似MPC等更强基线。

---

## 347. Democratic ICAI: Debating Our Way to Steering Principles from Preferences

**arXiv ID:** 2606.28294 | [PDF](https://arxiv.org/pdf/2606.28294v1)

**作者:** Kevin Kingslin `[一作]` (TCS Research), Shirish Karande `[通讯]` (TCS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Democratic ICAI，利用多角色辩论从对比偏好中抽取多重理由并生成可解释的原则，进而指导模型决策与生成

**💡 创新点**

通过结构化辩论与多样化专家角色实现对人类偏好深层推理的多元化捕获，显著提升原则的表达丰富度与可解释性

**🔧 技术方法**

多角色推理委员会、结构化辩论、嵌入聚类+抽象化、LLM 与决策树判定器

**📊 数据集**

MuCE‑Pref 与 LiTBench 两大创意偏好数据集

**📈 对比分析**

与 ICAI、AutoRubric 以及 CoT、ToT、Self‑Refine 等对话推理基线相比，DICAI 在多任务上平均偏好准确率最高、标准差最低，LLM 与决策树评估均显著优于对照组，且在创意生成实验中获得最高的创新度与第二高的多样性与质量

**⚠️ 局限性**

缺乏真实的“底层”原则作为对照，可能因训练数据的系统偏差导致原则出现有害偏见或无关表面特征，且需人工审查以保证原则完整与安全

---

## 348. Agentic Hardware Design as Repository-Level Code Evolution

**arXiv ID:** 2606.28279 | [PDF](https://arxiv.org/pdf/2606.28279v1)

**作者:** Cunxi Yu `[一作]` (NVIDIA Research), Brucek Khailany `[通讯]` (NVIDIA Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个自我进化的硬件设计代理框架，利用git仓库管理RTL设计与反馈循环，实现全流程无人工干预。

**💡 创新点**

将硬件设计任务转化为仓库级代码演进，利用可执行评估门控和git作为状态追踪器，支持RTL生成、修改、测试生成等多类别任务的完整迭代。

**🔧 技术方法**

基于LLM的自我进化代理、Git作状态存储与回溯、可执行评估器（编译、仿真、覆盖）、Markdown harness到项目包的编译器、Token缓存与会话复用。

**📊 数据集**

ChipBench、RTLLM‑2.0、Verilog‑Eval v2 以及CVDP的九个代码/验证生成类别（CID 002–016）。

**📈 对比分析**

通过单一hands‑free循环在每个基准上完成100%通过率；与单次模型Pass@1相比，迭代0的通用率较低，但最终迭代可达100%；Token消耗聚焦于难题，整体耗费约210M token，其中91%为缓存输入。

**⚠️ 局限性**

评价信号可能导致奖励黑客与过拟合评估器；评估周期短，实际设计中的长周期PPA/验证反馈难以处理；缺乏隐藏测试、鲁棒性检查，难以判断是否真正实现规范。

---

## 349. Parameter Efficient Hybrid Transformer (PEHT) for Network Traffic Prediction via Dynamic Urban Congestion Integration

**arXiv ID:** 2606.28274 | [PDF](https://arxiv.org/pdf/2606.28274v1)

**作者:** Abdolazim Rezaei `[一作]` (Texas A&M University), Mahboobeh Haghparast `[通讯]` (Texas A&M University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Parameter‑Efficient Hybrid Transformer (PEHT)，将城市移动性与拥堵信息融合进网络流量预测模型，并在 Transformer Encoder 中引入 LoRA 以显著减少可训练参数；

**💡 创新点**

①在 Encoder 训练阶段使用 LoRA 进行低秩权重更新，提升正则化与参数效率；②将城市移动性特征与 Encoder 输出在解码器中进行多模态融合，保持时间因果性；③设计可扩展的虚拟基站聚类与多模态特征工程，提升模型对高维时空数据的适应性；

**🔧 技术方法**

Transformer Encoder‑Decoder、LoRA、低秩分解、位置编码、特征融合、虚拟基站聚类、自动回归解码、AdamW 及早停策略；

**📊 数据集**

Telecom Italia Milan 实测流量数据 + CARLA 生成的 5 个合成拥堵/移动性场景；

**📈 对比分析**

与 HGCRN、ST‑DenseNet、STCNet、StTran、MVSTGN、ST2T 等基线在 RMSE、MAE、R² 上进行对比，Milan 数据上 RMSE 下降 14.6%、R² 达 0.9685；合成数据中整体表现优于单独 LoRA 或 Encoder Fusion 版本；

**⚠️ 局限性**

需在更大规模、多城市与极端场景下验证鲁棒性；对实时部署的计算与资源需求仍未完全评估；模型对异常事件的适应性待进一步研究。

---

## 350. Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models

**arXiv ID:** 2606.28273 | [PDF](https://arxiv.org/pdf/2606.28273v1)

**作者:** Niclas Lietzow `[一作]` (University of Tübingen), Michal Golovanevsky `[通讯]` (Harvard University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过激活补丁、组拆除和机制分析等方法，对视觉语言模型在视觉与先验知识冲突时的决策过程进行了因果层面的组件级研究，揭示了视觉默认、先验覆盖的稀疏注意力头电路。

**💡 创新点**

创新点在于首次给出视觉-先验冲突的组件级因果解释，发现仅约 2.5–4.8% 的后半层注意力头构成了路由-写作电路，实现视觉信息默认、先验信息需要少数头部注入的非对称机制，并在不同体系结构中得到验证。

**🔧 技术方法**

使用的技术包括：激活补丁（patching）在残差流、注意力头和 MLP 子层的三种粒度；组件拆除（ablation）验证必要性；主成分分析与分类判定促进/抑制头；注意力分布与 logit lens 机制分析。

**📊 数据集**

使用 Visual‑Counterfact 数据集，包含 469 个重新着色的物体图像以及对应的颜色识别问题，用于制造视觉与先验知识的冲突。

**📈 对比分析**

对比五个 3B–10B 参数量的 VLM（Qwen‑VL、LLaVA‑NeXT、PaliGemma），通过准确率、补丁恢复分数和拆除翻转率等指标评估，发现视觉引导在冲突下保持 68–96% 的准确率，而先验引导仅 17–55%；拆除促进注意力头可将先验引导翻转 68–96%，但对视觉引导影响极小，表明该电路是视觉默认的关键。

**⚠️ 局限性**

局限性包括：仅关注颜色属性冲突，未验证到形状、大小或空间关系等其他冲突；模型规模限制在 3B–10B；干预仅在最终输出 token 位置，可能忽略更早阶段的相关组件。

---

## 351. Towards Value-Constrained Credit Assignment in Fully Delegated AI Cooperatives

**arXiv ID:** 2606.28217 | [PDF](https://arxiv.org/pdf/2606.28217v1)

**作者:** Young Yoon `[一作]` (Hongik University), Soyeon Park `[通讯]` (Hongik University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在完全委托的 AI 合作社中，设计了一套基于价值约束的信用分配框架，通过代理人过滤梯度、使用遍历学习更新模型，并根据每步可接受更新的边际贡献进行累积结算。

**💡 创新点**

创新点在于将价值约束直接嵌入梯度过滤器中，结合遍历学习的分布式反向传播实现更透明的归因，同时提供在线边际贡献信号和累计结算机制，首次将价值可接受性与奖励分配统一到学习过程。

**🔧 技术方法**

使用的技术包括价值条件梯度过滤（规则/学习型可行域投影/梯度修改）、遍历学习（TL）框架、边际贡献评估、累计收入结算公式。

**📊 数据集**

论文未公开使用具体数据集，主要以理论模型与公式阐述。

**📈 对比分析**

由于缺乏实验评估，本文未与现有方法（如 FedAvg、Shapley 等）进行性能比较。

**⚠️ 局限性**

局限性包括：未验证在实际分布式训练中的收敛性；在多元价值约束下模型可能无法收敛到单一最优解；结算机制依赖于准确的边际贡献估计，且在缺少实验支持时难以评估公平性。

---

## 352. Tandem Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2606.28166 | [PDF](https://arxiv.org/pdf/2606.28166v1)

**作者:** Difan Jiao `[一作]` (University of Toronto), Ashton Anderson `[通讯]` (University of Toronto)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在RLVR（Reinforcement Learning with Verifiable Rewards）框架下，作者提出了Tandem Reinforcement Learning（TRL），让训练时的高级模型（senior）与一个被冻结的低级模型（junior）在单词级别交替生成答案，并只对 senior 的 tokens 进行 GRPO 损失更新；

**💡 创新点**

创新点是将同伴训练（tandem training）嵌入 RLVR，借助同伴生成的结构天然保证推理过程可被低级模型或人类跟随，从而在不额外正则化或监督的前提下，保持了 RLVR 的推理能力，同时显著降低分布漂移、提升手工切换鲁棒性并提高链条可读性；

**🔧 技术方法**

使用的技术包括：GRPO（Group Relative Policy Optimization）作为 RLVR 基准损失；在单词边界处交替产生 tokens 的同伴生成（tandem rollout）；在 Qwen3‑4B‑Instruct 上进行 fine‑tune；对训练过程做 vLLM 并行加速；对结果做对比实验和统计显著性检验；

**📊 数据集**

使用的数据集：数学竞赛题目 AMC 2023–2025、AIME 2024–2026 以及 Minerva Math；训练数据来源 DeepScaleR，包含标准的数学推理任务；

**📈 对比分析**

与 vanilla GRPO 和 KL‑regularized baseline 比较：solo 推理（pass@k）几乎相当；在与低级模型合作的手工切换场景下，TRL 在 pass@8 上提升约 3–4 个百分点；分布漂移方面，TRL 的 KL 与 junior 的距离降低约 14%；junior 对 senior 生成链条的可预测度（交叉熵下降 7.6%）和分布重叠提升 1.3%；

**⚠️ 局限性**

局限性包括：训练成本翻倍（每一步需双模型前向），收敛步数更少但仍需加速；仅使用同一基础模型作为 junior，缺乏多样化同伴或跨语言/工具的泛化；对长期共生成的动态衰退未深入解释；缺乏对人类实际交互的评估。

---

## 353. EchoSonar-R: A Multi-View Reasoning-Enabled Model for Disease Classification and Report Generation in Echocardiography

**arXiv ID:** 2606.28164 | [PDF](https://arxiv.org/pdf/2606.28164v1)

**作者:** Darya Taratynova `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Mohammad Yaqub `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种多视角、基于推理的视觉语言模型EchoSonar-R，用于心脏超声的疾病分类和报告生成。

**💡 创新点**

创新点在于将空间-时间视频编码器与结构感知心脏检测器相结合，实现跨视角推理并在生成报告时提供可解释的推理轨迹，同时采用两阶段训练（SFT + GRPO）联合优化分类与文本生成。

**🔧 技术方法**

采用冻结的EchoPrime mViT编码器与RT-DETR-L结构检测器，投影到Qwen3-8B语言模型中；使用跨模态投影、交叉序列插入、组相对策略优化（GRPO）和奖励设计。

**📊 数据集**

使用私有多视角心脏超声数据集（5061训练/1215测试，平均7视角）以及公开的MIMICEchoQA（单视角）和EchoNet-Dynamic（EF标签）数据集。

**📈 对比分析**

与Chiron-o1、Lingshu等基线相比，私有集宏观平衡准确率提升17.1%（macro BAcc 67.4%），MIMICEchoQA提升6.1%（macro BAcc 59.0%），报告生成GREEN分数0.800，推理质量几乎满分，错误率显著下降。

**⚠️ 局限性**

目前未利用患者历史序列，仅处理单次研究，且模型主要在单机构格式上训练，跨机构泛化受限。

---

## 354. Monocular Avatar Reconstruction via Cascaded Diffusion Priors and UV-Space Differentiable Shading

**arXiv ID:** 2606.28144 | [PDF](https://arxiv.org/pdf/2606.28144v1)

**作者:** Hong Li `[一作]` (Beihang University), Baochang Zhang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于单张野外图片的高保真可重光3D头像重建框架，能够生成4K分辨率的PBR材质；

**💡 创新点**

核心创新在于使用预训练扩散模型的串联LoRA模块完成纹理修复、光照均衡与材质分解，并通过跨本质注意力与UV空间可微BRDF损失实现物理一致的材质解耦；

**🔧 技术方法**

结合了预训练的DiT扩散变压器、LoRA适配器、Cross-Intrinsic Attention、可微BRDF渲染器、DINOv3语义先验与ConvNeXt V2几何编码器；

**📊 数据集**

利用少于100张真实3D扫描的PBR纹理数据，配合FFHQ、CelebAMask-HQ等大规模野外人脸图像进行数据增强与合成；

**📈 对比分析**

与MoSAR、FitMe、Relightify等前沿方法对比，在REALY几何重建评测中NMSE排名第三，在纹理填补与光照均衡任务上PSNR、SSIM、LPIPS指标均超过现有方法，最终在材质重光实验中表现出更真实的肤色与细节；

**⚠️ 局限性**

目前仅支持面部与头部PBR资产，未覆盖全身、动态发型、服装等，且对极端表情或遮挡的处理仍有局限。

---

## 355. Exposure Bias Can Alleviate Itself via Directional and Frequency Rectification in Flow Matching

**arXiv ID:** 2606.28226 | [PDF](https://arxiv.org/pdf/2606.28226v1)

**作者:** Guanbo Huang `[一作]` (Tsinghua Shenzhen International Graduate School Tsinghua University), Shao-Lun Huang `[通讯]` (Tsinghua Shenzhen International Graduate School Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在Flow Matching框架下，提出了DEFAR（Directional‑Frequency Adaptive Rectification）机制，用于动态校正训练与推理过程中的曝光偏差。

**💡 创新点**

创新点在于：① 将曝光偏差本身视为可利用的信号，设计Anti‑Drift Rectification（ADR）主动引导模型纠正偏移；② 通过频率补偿（FC）利用偏差中的低频信息，重新加权损失以弥补高噪声阶段缺失的低频结构。

**🔧 技术方法**

使用技术包括：单步推理仿真、流匹配模型、方向性正则化、频率域分析（FFT）、曝光偏差自反馈权重、训练时动态时间步采样。

**📊 数据集**

在公开图像生成数据集上评估：CIFAR‑10、CelebA‑64、ImageNet‑256（以及ImageNet‑512）以及对应的条件/无条件任务。

**📈 对比分析**

与SiT、IP、SDSS、MDSS、DG、OT‑CFM等基线相比，DEFAR在50 NFE采样下显著降低FID（ImageNet‑256减1.24/1.83、CIFAR‑10减2.08、CelebA‑64减1.13等），同时保持或提升IS、Precision/Recall，证明了其在多尺度、不同模型和采样策略下的稳健性。

**⚠️ 局限性**

局限性包括：只在图像生成任务上验证，未深入探讨在更高分辨率或跨模态（视频/音频）场景的适用性；引入ADR与FC会略微增加训练复杂度和计算开销；在极大模型规模下的可扩展性和数值稳定性仍需进一步研究。

---

## 356. Learning Stable In-Grasp Manipulation in a Non-Dropping Action Space

**arXiv ID:** 2606.28196 | [PDF](https://arxiv.org/pdf/2606.28196v1)

**作者:** Ha Thang Long Doan `[一作]`, Kenji Tahara `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出TSIGL框架，利用FTODG理论和控制障碍函数指导强化学习，实现稳定的抓握与在抓握内的精准位姿重定位与旋转。

**💡 创新点**

创新点在于将手掌抓握稳定性理论转化为可约束的“稳定动作空间”，并用CBF严格限制RL探索，显著降低物体掉落风险并提升采样效率。

**🔧 技术方法**

结合FTODG控制、控制障碍函数、PPO算法、Isaac Lab仿真、Shadow手等技术，完成端到端强化学习与经典控制的混合。

**📊 数据集**

使用仿真数据集：多种基本形状（立方体、圆柱体、锥体）以及实际物体（金枪鱼罐头、Spam罐头、三棱柱、芥末瓶）共96个并行环境。

**📈 对比分析**

与基线端到端RL、FTODG固定控制器以及无重力/重力课程等方法对比，TSIGL在物体掉落率为0、连续成功率提升至40–50步、相对精准度提升约130–155%（位置）和39–81%（姿态）。

**⚠️ 局限性**

局限性：仅在仿真中验证，未在真实机器人上测试；动作空间约束可能限制进一步探索；未讨论抓握质量、工作空间分析等关键问题。

---

## 357. Buffered control for opacity in timed automata

**arXiv ID:** 2606.28170 | [PDF](https://arxiv.org/pdf/2606.28170v1)

**作者:** Étienne André `[一作]` (Nantes Université), Engel Lefaucheux `[通讯]` (Université de Lorraine)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

在缓冲观测模型下研究时序自动机（Timed Automata）控制问题，探讨在控制策略下如何保证系统的不透明性（opacity）

**💡 创新点**

①证明了在一般控制下的不可辨识性问题是不可判定的；②提出了两类可判定的子问题（N-顺序策略与可观测顺序策略），并给出了它们的精确复杂度；③展示了弱不可辨识性与全不可辨识性可相互还原，从而统一分析两种隐私强度。

**🔧 技术方法**

使用时序自动机理论、区域自动机（Region Automaton）、信念自动机（Belief Automaton）、两人游戏理论以及从 Post 对应问题（PCP）构造的可归约证明

**📊 数据集**

本工作为纯理论分析，未使用实际数据集；通过构造的范例自动机和理论证明来展示可判定与不可判定的边界

**📈 对比分析**

通过复杂度分析与已有的可辨识性判定结果对比，证明 N-顺序策略下的可辨识性问题是 EXPTIME‑完整，OSS（可观测顺序策略）下的可辨识性问题是 PSPACE‑完整；实验评估未开展，无法给出性能数值

**⚠️ 局限性**

主要限制：（1）所得到的算法复杂度极高（指数级/双指数级），在大规模系统上难以直接应用；（2）仅考虑缓冲观测模型，未覆盖更一般的噪声或不确定的观测；（3）实验验证缺失，实际可行性和鲁棒性尚未评估。

---

## 358. Recovering Sharp Conductivity Features in the Finite-Data Calderón Problem with Physics-Informed Neural Networks

**arXiv ID:** 2606.28158 | [PDF](https://arxiv.org/pdf/2606.28158v1)

**作者:** Ali AlHadi Kalout `[一作]` (Universitat de Barcelona), Guy David `[通讯]` (Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于 PINN 的电导率反演框架，利用随机小波边界激励和 Fourier 频率编码联合学习电导率和对应电势。

**💡 创新点**

创新点在于：①引入多尺度随机小波边界激励提高有限边界数据下的辨识性；②使用 Fourier 特征编码缓解 PINN 的低频偏置，从而更好恢复尖锐不连续结构。

**🔧 技术方法**

技术方法包括：多网络 PINN 架构（电导率网络 + 条件电势网络）、随机小波边界条件、Fourier Feature Encoding、自动微分与 Adam 优化、物理约束损失函数。

**📊 数据集**

使用合成数据集，采用有限差分前向求解器在二维单位正方形上生成多种导电率场（含斑块、光滑、随机频率分布）及对应的 Dirichlet‑Neumann 对。

**📈 对比分析**

通过比较 FFE 与无 FFE 两种坐标表示，使用多指标评估（相对误差、均方误差、SSIM、IoU、Pearson 相关、PSNR），结果显示 FFE 在尖锐或高频导电率下提升重建精度，而在光滑低频场下无明显优势。

**⚠️ 局限性**

局限性包括：对中心小尺寸目标的重建仍不稳定；仅使用合成无噪声数据，缺乏对测量噪声、模型误差、非完整边界数据等实际情况的评估；缺乏对不同边界激励策略的系统性比较。

---

## 359. Regularized Reward-Punishment Reinforcement Learning

**arXiv ID:** 2606.28152 | [PDF](https://arxiv.org/pdf/2606.28152v1)

**作者:** Jiexin Wang `[一作]` (ATR Computational Neuroscience Laboratories), Eiji Uchibe `[通讯]` (ATR Computational Neuroscience Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出KL-Coupled Policy Regularization (KCPR) 框架，并实现其深度版本klDMP，利用动态学习的同伴策略作为先验实现奖励与惩罚两条动机路径的互调；

**💡 创新点**

创新点在于把KL正则化从传统的熵或固定先验改为互相学习的同伴先验，实现奖励与惩罚策略的policy‑level协同，并提出伴随先验软化与分离重放缓冲的稳定机制；

**🔧 技术方法**

核心技术包括KL‑Coupled Soft Optimality (KCSO)、伴随先验软化、分离奖励/惩罚重放缓冲、深度网络实现（共享特征层+分离价值/策略头）以及温度衰减的探索策略；

**📊 数据集**

实验使用低维网格世界（U‑maze、Three‑room maze）和高维机器人导航（TurtleBot3 在 ROS Gazebo 中的 U‑maze、T‑maze、Three‑room maze）作为数据集；

**📈 对比分析**

与 DQN、SQL、softDMP 进行对比；klDMP 在三种迷宫中实现了与软DMP相当的导航效率，同时大幅降低碰撞率，表现出更稳健的学习与更好的安全性；

**⚠️ 局限性**

局限包括：需要手工调节KL系数和软化参数；目前仅针对离散动作空间，未验证在连续控制任务上的泛化；框架聚焦奖励/惩罚双分解，其他动机如好奇心、不确定性尚未扩展。

---

## 360. Functional outcomes and naturalistic engagement with a purpose-built conversational AI for mental health (Ash)

**arXiv ID:** 2606.28241 | [PDF](https://arxiv.org/pdf/2606.28241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 361. Toward Robust In-Context Segmentation via Concept Guidance

**arXiv ID:** 2606.28149 | [PDF](https://arxiv.org/pdf/2606.28149v1)

**作者:** Zhigang Chen `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc`

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

## 362. How Width and Data Shape Generalization Scaling Laws in Quadratic Neural Networks

**arXiv ID:** 2606.28242 | [PDF](https://arxiv.org/pdf/2606.28242v1)

**作者:** Julius Girardin `[一作]` (École Polytechnique Fédérale de Lausanne), Lenka Zdeborová `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

研究了二阶神经网络在有限宽度与样本规模下的泛化误差扩展律，揭示宽度对特征学习的隐式正则化作用；

**💡 创新点**

提出完整的相位图，给出宽度、样本、正则化及目标谱共同作用下的精确指数；证明在合适宽度或正则化下可达到 Bayes 最优收敛速度；

**🔧 技术方法**

利用 AMP 状态演化推导解析式、矩阵压缩感知等工具；将网络映射为核正则化且秩受限的矩阵优化；使用数值 LBFGS 与 AMP 预测对比；

**📊 数据集**

采用合成数据：高维高斯输入、中心化二次特征，目标矩阵满足幂律谱，带可调标签噪声；

**📈 对比分析**

与数值模拟（LBFGS）进行比较，理论预测与仿真高度一致；相较传统固定特征或无限宽度模型，提供更细致的宽度调节策略；在不同正则化和宽度下可实现 Bayes 最优性能；

**⚠️ 局限性**

仅适用于理想化浅层二次网络、Gaussian 输入和谱结构，未给出严谨证明；宽度约束导致非凸问题；结果难以直接推广到深度网络或真实数据集。

---

## 363. Towards Automating Scientific Review with Google's Paper Assistant Tool

**arXiv ID:** 2606.28277 | [PDF](https://arxiv.org/pdf/2606.28277v1)

**作者:** Rajesh Jayaram `[一作]` (Google Research), Vincent Cohen-Addad `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并部署了 Paper Assistant Tool（PAT），一种基于 Gemini Deep Think 的代理式 AI 框架，用于对科学论文（特别是数学与计算机科学）进行深度验证、错误检测和改进建议，先后在 STOC、ICML 会议中做了作者预审试点。

**💡 创新点**

提出四层 AI‑人类协作的审查架构，并在 PAT 中实现了论文分段、动态计算预算、深度推理多模型协同和最终合成的推理扩展，从而显著提升了对复杂理论与实验细节的识别能力。

**🔧 技术方法**

核心技术包括 Gemini Deep Think 的多轮推理、段落分割器、动态预算分配、深度审查代理、合成代理以及基于搜索的真伪校验，整体构成四阶段推理扩展流水线。

**📊 数据集**

使用 SPOT 错误数据集（数学与 CS 论文的方程/证明错误子集）进行评估，并在 STOC 与 ICML 会议的 4,700 篇论文上进行实地部署。

**📈 对比分析**

与单一 LLM 调用（零射击 Gemini 3.1 Pro）相比，PAT 在 SPOT 子集上的召回率从 55.2% 提升至 89.7%（提升 34%）；在会议试点中，作者对 PAT 的再次使用意愿超过 92%，发现关键错误率为 10%（STOC）至 35%（ICML），并推动 31% 的作者开展新实验，整体性能显著优于传统人工评审。

**⚠️ 局限性**

主要局限包括：知识截止与时间漂移导致的 hallucination，PDF 解析误差，模型误判（将正确的论证误认为错误）以及对计算资源的高需求，仍需进一步完善推理准确性与可扩展性。

---

## 364. Humanizing Automatically Generated Unit Test Suites with LLM-Based Refactoring

**arXiv ID:** 2606.28229 | [PDF](https://arxiv.org/pdf/2606.28229v1)

**作者:** Wendkûuni C. Ouédraogo `[一作]` (University of Luxembourg), Tegawendé F. Bissyandé `[通讯]` (University of Luxembourg)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TestHumanizer，利用 LLM 作为 SBST（如 EvoSuite）生成的测试用例的可读性、可维护性改进层，对整个测试套件进行重构而不改变其行为。

**💡 创新点**

创新点在于将 LLM 作为控制型重构层，配合多级上下文（仅测试、代码、摘要）与多重验证（编译、相似度、覆盖率）来确保语义保持，且在摘要上下文下显著提升可读性与结构化，减少测试异味。

**🔧 技术方法**

技术包括：搜索型软件测试工具 EvoSuite、ChatGPT 系列 LLM（gpt‑4o、mistral‑large‑2407）、代码摘要模型与 SIDE 评估、CodeBERT/GraphCodeBERT/OpenAI Embedding、CTSES 综合相似度、PMD 复杂度与 CCTR 认知复杂度、TsDetect 异味检测、JaCoCo 覆盖率收集，及基于规则的编译与语义验证。

**📊 数据集**

使用 Defects4J（147 个类）和 SF110（203 个类）共 350 个 Java 类作为实验数据集。

**📈 对比分析**

与 EvoSuite 基线和一次性 LLM 生成的测试比较，TestHumanizer 在 88–98% 的编译率、覆盖率仅低 1–2% 之差、可读性指标提升至 0.74–0.75、认知复杂度下降 1–2 分，且在测试异味（如 Conditional Logic Test）显著降低，整体性能优于直接 LLM 生成。

**⚠️ 局限性**

局限性包括：仍需手工验证覆盖率与正确性，摘要上下文虽稳定但可能缺乏细节，重构过程中仍存在 hallucination 引起的编译错误，且相似度评估对结构大改不敏感，未来需引入多轮修复与检索增强机制。

---

## 365. Estimation--Prediction Tradeoff in Causal Probabilistic Temporal Graphs

**arXiv ID:** 2606.28225 | [PDF](https://arxiv.org/pdf/2606.28225v1)

**作者:** Aniq Ur Rahman `[一作]` `[通讯]` (University of Oxford), Aniq Ur Rahman (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种带有短暂边的概率因果时间图生成器，并研究了参数估计误差与预测误差之间的固有权衡。

**💡 创新点**

创新点在于：①将参数估计误差（Fisher 信息）与不可约预测不确定性（熵）关联起来，揭示了二项式逻辑回归模型中的估计–预测权衡；②构造了可直接恢复因果结构的离散时间图生成模型，为因果发现与预测性能提供统一评估框架。

**🔧 技术方法**

使用了二项式逻辑回归、Fisher 信息、Cramér–Rao 下界、二进制交叉熵、KL 散度、泊松点过程和二项式随机交集图等信息论与概率图模型技术。

**📊 数据集**

使用自定义生成器 ℳ 产生的合成数据：5 个节点（k=10 条边）、噪声水平 η=0、时间窗 T=10⁴、不同的 m、p、σ² 参数组合，全部为模拟实验数据。

**📈 对比分析**

通过比较经验参数估计误差与 CR 下界、以及不可约预测误差与 CR 下界，验证了两者的倒置关系。实验显示，参数可估计越好（Fisher 信息越大）对应的预测熵越大，预测误差越高；相反，预测熵较低时参数可估计性较差。虽然没有与真实世界数据对照，但实验结果充分说明了理论预期。

**⚠️ 局限性**

局限性包括：①仅针对二项式逻辑回归模型推导，难以直接推广到更复杂的时间图模型；②所有实验均为仿真数据，缺乏在真实时间图上的验证；③生成器假设因果结构已知，实际因果发现仍面临高维稀疏性和非线性问题。

---

## 366. GBC: Gradient-Based Connections for Optimizing Multi-Agent Systems

**arXiv ID:** 2606.28187 | [PDF](https://arxiv.org/pdf/2606.28187v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 367. LLawCo: Learning Laws of Cooperation for Modeling Embodied Multi-Agent Behavior

**arXiv ID:** 2606.28182 | [PDF](https://arxiv.org/pdf/2606.28182v1)

**作者:** Qinhong Zhou `[一作]` (University of Massachusetts Amherst), Anoop Cherian `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于行为法则学习的框架，允许嵌入式多智能体通过自我反思与法则提取实现自适应协作。

**💡 创新点**

创新点在于自动从失败轨迹中提炼高层次行为法则，并通过监督微调将法则内化到LLM，使推理过程可解释且可被人工修改。

**🔧 技术方法**

主要技术包括LLM自我反思、法则提取、SFT微调以及基于法则的推理与行动生成。

**📊 数据集**

使用了自构造的PARTNR‑Dialog大规模对话协作基准（2000训练/1000验证）以及TDW‑MAT协作操控基准。

**📈 对比分析**

在PARTNR‑Dialog上平均提升约4.5%成功率、在TDW‑MAT提升约6.8%，相较于ReAct、RoCo、CoELA、CommPARTNR等现有通讯代理实现了显著性能提升。

**⚠️ 局限性**

局限性包括仅验证双智能体场景，训练成本较高，法则的安全性和可行性仍需人工验证。

---

## 368. CPAgents: Agentic Composite Phenotype Generation for Cardiac Disease Association

**arXiv ID:** 2606.28179 | [PDF](https://arxiv.org/pdf/2606.28179v1)

**作者:** Zuoou Li `[一作]` (University College London), Mengyun Qiao `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种基于三代理（分析、建议、验证）的迭代框架，用于自动构建和验证可解释的心脏影像复合表型，提升疾病关联性和分类性能。

**💡 创新点**

创新点在于：①将表型构建视为受限符号组合的搜索，结合统计诊断引导；②使用专门的验证链（ΔAUC、稳定性、SHAP重要性）确保新表型既有统计意义又稳健；③提供透明的构造轨迹和证据链，便于可审计与复现。

**🔧 技术方法**

主要技术包括：深度语言模型（DeepSeek‑V3.2）用于生成表达式；符号树（AST）与运算符限制实现安全组合；多阶段过滤（ΔAUC、ElasticNet 稳定性、LightGBM‑SHAP）和交叉验证；集成评估（SVM、LDA、AdaBoost、MLP）以决定表型接受。

**📊 数据集**

使用英国生物样本库（UK Biobank）26,893名参与者的标准化CMR影像表型及相关协变量，研究9类临床疾病。

**📈 对比分析**

与专家手工选取特征和MESHAgent基准进行对比，采用AUC和召回率衡量。实验显示，新复合表型在72种模型–疾病–指标组合中获得56个最高排名，平均AUC提升约0.06，召回率提升约0.04，显著优于基准。

**⚠️ 局限性**

局限性包括：①仍受限于预设的运算符集，可能遗漏更复杂的生物学关系；②在高度相关的影像特征中，组合空间增大导致多重检验压力；③模型对不同人群或多模态数据的泛化尚待验证。

---

## 369. MixTTA: Low-Rank Cross-Channel Mixing for Reliable Test-Time Adaptation

**arXiv ID:** 2606.28142 | [PDF](https://arxiv.org/pdf/2606.28142v1)

**作者:** Mansoo Jung `[一作]` (Seoul National University), Jungwoo Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MixTTA，一个在归一化层加入低秩跨通道混合的轻量化插件，以实现更可靠的测试时自适应。

**💡 创新点**

创新点在于通过低秩跨通道变换补充传统的逐通道缩放与偏置，并引入 Decoupling Projection 与 Spectral Projection 两种投影机制，显著提升跨通道结构修正能力。

**🔧 技术方法**

使用了低秩适配技术、离散投影（Decoupling Projection）与谱投影（Spectral Projection）来控制更新空间，并将插件无缝集成到现有基于归一化层的 TTA 方法中。

**📊 数据集**

主要在 ImageNet‑C 与 ImageNet‑Sketch 两大基准集上进行实验，并在多种野生场景（标签失衡、单样本、混合分布）下验证其鲁棒性。

**📈 对比分析**

与 Tent、EATA、SAR、DeYO、ReCAP、LinearTCA 等主流 TTA 方法对比，MixTTA 在所有场景下平均提升 2–3% 的准确率，显著降低适配失败率，且算力和参数开销仅增加约 4%。

**⚠️ 局限性**

局限性包括对低秩维度 r 的敏感性（需要在不同任务中保持适当选择）、在极端非平稳数据流中仍可能出现收敛问题，以及目前仅在视觉任务上验证，尚缺乏跨领域的通用性探讨。

---

## 370. Pairwise Reflection Symmetry in Generalized Latin Rectangles

**arXiv ID:** 2606.28315 | [PDF](https://arxiv.org/pdf/2606.28315v1)

**作者:** Enrico Iurlano `[一作]` (TU Wien), Günther R. Raidl `[通讯]` (TU Wien)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

研究并构造满足对称性条件（对称列对的有序符号出现次数相等）的广义拉丁矩阵（称为“对称拉丁矩阵”），并探讨其存在性与构造方法。

**💡 创新点**

提出了对称拉丁矩阵与 2‑子方块完整性、Williams 设计和完美序列覆盖数组的深层联系；给出了完整的存在性阐述：偶数 λ 时始终存在；奇数 λ 的最小可行值为 1 当且仅当 n 为 2 的幂；并给出利用二面体群与直接积构造的通用方法，首次给出奇数 λ 的结构与计数下界。

**🔧 技术方法**

核心技术包括群论（充分利用对称群、二面体群与其可归约性）、组合设计理论（拉丁方、交叉试验设计、覆盖数组）、以及数值枚举（利用二次约束整数规划与 Gurobi 进行完整枚举）。

**📊 数据集**

使用的“数据集”为理论构造与枚举结果，主要是对小 n（≤12）和 λ 的完整枚举表；没有外部实验数据集。

**📈 对比分析**

对称性矩阵的构造与存在性通过理论证明与直接积构造得到；对小参数的存在性通过 QCP 枚举得到。实验中对 λ=1、2、3 的计数与存在性在秒级完成，λ≥2、n≥8 的完整枚举则在上限时间内部分完成；与已知的 Williams 设计、PSCA 等结构进行对比，发现多数实例具有群结构，表明该构造方法具有较高的密度与可扩展性。

**⚠️ 局限性**

局限性：
1) 对奇数 λ 的最小可行值 n 的上界仍为超指数级，尚未证明其紧凑性；
2) 对偶数 λ 的存在性虽然已证实，但对称矩阵的完整计数与最优构造仍未知；
3) 枚举方法受限于计算资源，λ≥2、n≥8 的完整计数尚不可得；
4) 研究主要集中在纯理论与小规模实验，缺乏对更大参数空间的实证验证。

---

## 371. V-TSN: A Software-Defined TSN Overlay for General-Purpose Networks

**arXiv ID:** 2606.28285 | [PDF](https://arxiv.org/pdf/2606.28285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 372. Beyond Sparse Supervision: Diffusion-Guided Learning for Few-Shot Graph Fraud Detection

**arXiv ID:** 2606.28134 | [PDF](https://arxiv.org/pdf/2606.28134v1)

**作者:** Liming Liu `[一作]` (Central South University), Heyuan Shi `[通讯]` (Central South University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 ADC-GNN 模型，用于在极少标签和类别不平衡的图欺诈检测场景下提升异常节点识别效果。

**💡 创新点**

创新点包括：①将扩散式特征空间噪声增强与对比学习结合，以稀缺标签下对异常特征进行鲁棒性增强；②设计关系感知的多跳频谱注意力机制，既保留局部结构信息，又强调高频异常信号；③通过残差融合与多分支注意力统一多尺度、多关系特征，实现异常语义的充分利用。

**🔧 技术方法**

核心技术：图神经网络 (GNN)、特征空间扩散（cosine 调度）、对比学习（跨视图正负对），多跳多关系多项式滤波 + 频谱注意力，关系注意力融合，残差融合，交叉熵分类。

**📊 数据集**

使用公开数据集 Amazon、YelpChi、T‑Finance 三个欺诈检测基准，以及私有电信交易数据 CM（约61k节点、39M边）。

**📈 对比分析**

实验对比：在 1% 训练样本下与原始基线及最新图异常/欺诈基线（CGAD、ARC、UniGAD、CGNN 等）进行统一协议比较，ADC‑GNN 在 AUC 和 Macro‑F1 上均领先 1.9–3.1% 及 1–3.5%；在不同训练比例（1%–10%）和多种随机划分下保持稳健性；与传统过采样（ROS、SMOTE）相比亦表现更优。

**⚠️ 局限性**

局限性：①扩散模块仅在特征空间进行去噪，未覆盖拓扑生成；②当标签量充足时，优势可能不明显；③私有 CM 数据无法公开，限制了完整复现；未来需探索拓扑感知扩散、轻量部署和隐私友好型数据共享方案。

---

## 373. VGB for Masked Diffusion Model: Efficient Test-time Scaling for Reward Satisfaction and Sample Editing

**arXiv ID:** 2606.28301 | [PDF](https://arxiv.org/pdf/2606.28301v1)

**作者:** Kijung Jeon `[一作]` (Georgia Tech), Molei Tao `[通讯]` (Georgia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 MDM‑VGB，一种基于奖励引导的掩码扩散模型（Masked Diffusion Model）回溯采样器，能够在任意位置进行令牌的揭示和再掩码，实现高质量的生成与编辑。

**💡 创新点**

创新点包括：①将经典的 Jerrum–Sinclair 回溯链从固定前缀树推广到任意掩码状态图，允许任意坐标的再掩码；②使用价值估计指导重新掩码，兼顾探索与修正；③提出动量提升（MDM‑VGB‑Momentum）以减少振荡；④在理论上证明对奖励倾斜分布的快速混合、对过程验证器噪声的鲁棒性和二次时间复杂度。

**🔧 技术方法**

主要技术：掩码扩散模型、奖励倾斜采样、基于价值的马尔可夫链、掩码状态图、动量提升、短列表候选更新、过程验证器（value estimator）训练。

**📊 数据集**

实验数据集包括：QM9（分子设计）、DNA enhancer 设计、Protein motif scaffolding、Sudoku（约束满足）、Letter avoidance（自然语言约束）、Dyck grammar（形式语言编辑）等。

**📈 对比分析**

与基准方法（Base MDM、Best‑of‑N、MDM‑VGR、AR‑VGB、AR‑VGB‑Momentum）进行对比，评估指标为 Pass@95、成功率、RMSD 等。MDM‑VGB 与其 Momentum 版本在多项任务上均实现了更优的质量–成本曲线，生成任务中显著提升 Pass@95，编辑任务中显著降低编辑次数和成本，且理论上混合时间为 O(n²)（Momentum 为 O(n)），优于 Best‑of‑N 的指数复杂度。

**⚠️ 局限性**

局限性：需要为每个任务训练高质量的过程验证器，尤其在长序列或大词表场景下成本高；对极长序列或更复杂推理任务（如编程、算术推理）的可扩展性尚未验证；Momentum 版本在编辑任务中偶尔牺牲精度；整体实现仍依赖手工短列表和块更新策略，可能在更大规模任务中受限。

---

## 374. The Remittance Blueprint: Data-driven Intelligence for Sri Lanka

**arXiv ID:** 2606.28190 | [PDF](https://arxiv.org/pdf/2606.28190v1)

**作者:** Dhinanjaya Fernando `[一作]` (University of Moratuwa), Nisansa de Silva `[通讯]` (University of Moratuwa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本研究利用32年（1994‑2025）月度数据，对斯里兰卡移民与汇款流动进行多维度分析，并预测2026年汇款额

**💡 创新点**

创新点在于构建高频统一数据集，结合Johansen协整与VECM的长期均衡分析、冲击响应与方差分解，并用Ridge回归等机器学习方法提升预测精度

**🔧 技术方法**

使用时间序列分析（ADF、Johansen、VAR/VECM、IRF、FEVD）、结构断点检测、机器学习模型（Ridge回归、K‑Means）以及传统SARIMA做基准

**📊 数据集**

8个权威来源的月度数据（中央银行、世界银行、SLBFE、ILO、FRED、DCS、EIA、PPP），共384个月，涵盖人口、汇款、宏观经济与外部变量

**📈 对比分析**

与单变量SARIMA基准对比，Ridge回归在测试集上RMSE降至494.8百万美元，提升73.8%，预测2026年汇款约9001百万美元

**⚠️ 局限性**

局限包括年数据线性插值可能掩盖季节波动、SLBFE只计注册移民未覆盖非正规流动、模型线性结构无法捕捉极端政策或地缘冲击导致的非线性变化

---

## 375. PA-BiCoop: A Primary-Auxiliary Cooperative Framework for General Bimanual Manipulation

**arXiv ID:** 2606.28192 | [PDF](https://arxiv.org/pdf/2606.28192v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 376. HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration

**arXiv ID:** 2606.28215 | [PDF](https://arxiv.org/pdf/2606.28215v1)

**作者:** Jiaxin Li `[一作]` (Shanghai Jiao Tong University), Yong-Lu Li `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HAT-4D 框架，从单目视频中重建动态 4D 多物体交互；

**💡 创新点**

创新点包括：① 将交互知识图（IKG）作为因果引擎，引导 3D 生成与 4D 传播；② 多级人机协同（HITL）细化生成过程；③ 结合 4D Gaussian Splats 与 L4GM 的 4D 传播，实现时空一致、物理可行的交互；

**🔧 技术方法**

技术手段包括：Vision‑Language Model (VLM) 提取 IKG；SAM3D 生成 3D 高斯剖面；L4GM 用于 4D 传播；多级 HITL 反馈（高斯级、区域级、对象级）；多视角渲染与评估；

**📊 数据集**

使用新建的 MVOIK-4D 基准，包含 112 场景、77 任务、39 交互类别及多视角记录；同时在公开数据上对比基线；

**📈 对比分析**

与 L4GM、GVFDiffusion、SV4D、STAG4D、FB4D 等多帧 4D 生成模型对比，HAT-4D 在生成质量（LPIPS、FVD）、交互质量（Deform、Relation）以及时间记忆（Intra、Long）等指标上均达到或超过最优；

**⚠️ 局限性**

局限性包括：对高度柔性变形和高速非刚体运动仍有重建误差；受 VLM 推理效率和推理能力限制；需要更完善的时间对应与形变建模。

---

## 377. An Exponential Lower Bound for Spectral Density Estimation on Unweighted Graphs

**arXiv ID:** 2606.28188 | [PDF](https://arxiv.org/pdf/2606.28188v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 378. COCOLogic-V2: Identifying Logical Inconsistencies via Truly Hard-Negatives

**arXiv ID:** 2606.28194 | [PDF](https://arxiv.org/pdf/2606.28194v1)

**作者:** David Steinmann `[一作]` (TU Darmstadt), Wolfgang Stammer `[通讯]` (Max Planck Institute for Informatics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个面向真实图像的物体中心视觉推理数据集COCOLogic-V2，并对样本进行正样本、近边界负样本和远边界负样本分类；

**💡 创新点**

创新点在于将逻辑推理问题拆解为不同难度的负样本类型，为模型可解释性与可靠性提供细粒度诊断框架；

**🔧 技术方法**

采用了概念瓶颈模型、程序合成方法等可解释模型，结合逻辑规则搜索与视觉特征提取技术进行推理；

**📊 数据集**

使用了COCOLogic-V2数据集，覆盖了第一阶逻辑的广泛子集，包含正样本、近边界负样本和远边界负样本；

**📈 对比分析**

实验表明，模型能够很好地区分正样本与远边界负样本，但在近边界负样本上表现差，且在少样本场景下受到感知噪声和规则搜索空间过大影响；

**⚠️ 局限性**

局限性包括对近边界负样本的推理能力不足、在少样本设置下对噪声和搜索空间的鲁棒性差，以及整体对复杂现实图像推理仍面临挑战。

---

## 379. Autoencoder Architectures for Athlete Performance Scoring from Wearable Telemetry

**arXiv ID:** 2606.28145 | [PDF](https://arxiv.org/pdf/2606.28145v1)

**作者:** Mateusz Kubita `[一作]` (Warsaw University of Technology), Krzysztof Siwek `[通讯]` (Warsaw University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

评估了多种无监督降维模型，将跑步者九维可穿戴数据压缩为单一可解释的表现得分。

**💡 创新点**

引入复合选择准则，将重构误差与多元解释性指标（Spearman、Kendall、互信息、置换重要性）结合，并通过Borda计数与自助验证实现评分的稳定性与可解释性。

**🔧 技术方法**

使用深度自编码器（Simple、Medium、Deep）、PCA、变分自编码器；统计解释性指标、Borda计数、Bootstrap；训练与评估基于 MSE 与 Latent Score Quality。

**📊 数据集**

公开的 Golden Cheetah 运动日志，45,836 条样本，9 个特征（配速、平均心率、海拔增益、总距离、步频、aerobic decoupling、年龄、体重、性别）。

**📈 对比分析**

在测试集上通过 MSE 与 Latent Score Quality 组合成的复合分数进行模型排序；Deep AE 以最低 MSE（0.00178）和最高复合分数（0.972）排名第一，PCA 次之。

**⚠️ 局限性**

数据来源单一平台，缺少真实标签（如 VO₂max 或比赛成绩）进行验证，复合权重设定随意，未检验跨平台泛化能力。

---

## 380. Cognitive Episodes in LLM Reasoning Traces Enable Interpretable Human Item Difficulty Prediction

**arXiv ID:** 2606.28186 | [PDF](https://arxiv.org/pdf/2606.28186v1)

**作者:** Chenguang Wang `[一作]` (Virginia Tech), Dawei Zhou `[通讯]` (Virginia Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Epi2Diff 框架，利用语言模型的推理轨迹转换为句子级的认知事件序列，并将其与项目语义表示结合，来预测考试题目难度。

**💡 创新点**

创新在于将 LLM 的推理轨迹细分为认知事件并构造长度、分布、转移等过程特征，从过程层面捕捉题目难度，并提供可解释的难度信号。

**🔧 技术方法**

使用 RoBERTa 进行句子级事件分类，Sentence‑BERT 获得项目嵌入，XGBoost 作为预测器，并对多种学生角色的推理轨迹进行聚合。

**📊 数据集**

在四个真实考试基准上验证：USMLE、Cambridge 英语资格、SAT 阅读与写作以及 SAT 数学。

**📈 对比分析**

与传统小模型、LLM 零样本、提示学习、全参数微调及 LoRA 等基线对比，Epi2Diff 在所有四个数据集上均取得最高分，分类上平均提升约 8.1% 相关指标，回归上误差显著下降且 R² 明显提高。

**⚠️ 局限性**

局限在于依赖有限数量的 LLM 生成的推理轨迹，轨迹风格差异可能影响事件分布，且仅覆盖四个评测领域，尚需在更广泛的题型与模型上验证。

---

## 381. Disentangling Continuous-Time Latent Dynamics: Identifiability of Latent SDEs via Diffusion Shifts

**arXiv ID:** 2606.28228 | [PDF](https://arxiv.org/pdf/2606.28228v1)

**作者:** Yuanyuan Wang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Kun Zhang `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究连续时间潜在SDE模型的可识别性，利用两种环境下扩散协方差的差异来恢复潜在坐标系及其漂移-雅可比因果图。

**💡 创新点**

证明只需两种对角扩散矩阵且其坐标比率互不相同，即可在不需要漂移稀疏的情况下识别潜在坐标到置换、尺度变换，并给出两阶段估计器实现该理论。

**🔧 技术方法**

使用Ito公式与二次变差理论推导坐标变换约束，结合短时Euler–Maruyama拟合与稀疏漂移回归，构建可训练的编码器+漂移+对角扩散参数模型。

**📊 数据集**

合成数据（稠密/稀疏线性OU、非线性稀疏漂移）以及实测的Hardanger桥加速度监测数据。

**📈 对比分析**

通过合成实验比较MCC、单调得分和图匹配率；实测实验比较跨种子MCC与GMR。结果显示：仅在满足“扩散比率不同”条件时，潜在坐标与因果图的恢复精度高；控制条件下精度显著下降。

**⚠️ 局限性**

局限性：要求观测映射为可逆光滑（C²）微分同胚；扩散为状态无关的对角矩阵；需已知且仅有两种环境；对非对角扩散、状态相关噪声或不完整环境标签的鲁棒性不足。

---

## 382. StructSplat: Generalizable 3D Gaussian Splatting from Uncalibrated Sparse Views

**arXiv ID:** 2606.28321 | [PDF](https://arxiv.org/pdf/2606.28321v1)

**作者:** Jia-Chen Zhao `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 StructSplat，一种前向、可泛化的 3D 高斯重建框架，能够直接从未标定的稀疏视角图像生成高质量的新视角合成。

**💡 创新点**

创新点包括：1）采用结构化表示，将几何、语义与纹理信息分别处理；2）引入像素对齐特征注入机制，实现高频纹理恢复；3）设计相机对齐策略，防止信息泄漏，提升跨场景泛化。

**🔧 技术方法**

使用技术：VGGT 作为几何编码器、DINOv3 语义编码器、轻量纹理编码器、像素对齐特征注入、相机对齐模块、3D 高斯点渲染、混合精度训练（BF16）与 DeepSpeed、WSD 学习率调度。

**📊 数据集**

实验使用 DL3DV、RealEstate10K、ACID 三个大规模数据集，进行跨数据集评估。

**📈 对比分析**

与无相机参数方法（如 AnySplat、Splatt3R、Depth Anything 3 等）及弱/强相机参数方法比较，评估指标为 PSNR/SSIM/LPIPS。在 DL3DV 上 PSNR 28.045（+5.67 dB 超越 AnySplat），ACID 上 24.372（+1.94 dB），RealEstate10K 上 22.240（+1.72 dB），在多视角设置下仍保持优势，整体表现显著优于现有方法。

**⚠️ 局限性**

局限性：在极稀疏视角或严重遮挡场景下仍可能出现重建失真；对视角相关光照、动态场景的建模仍不完善。

---

## 383. WARP-RM: A Warp-Augmented Relative Progress Reward Model for Data Curation

**arXiv ID:** 2606.28320 | [PDF](https://arxiv.org/pdf/2606.28320v1)

**作者:** Justin Yu `[一作]` (University of California, Berkeley), Ken Goldberg `[通讯]` (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种完全自监督的任务进度评估方法WARP-RM，并将其进度信号用于行为克隆中的动作块加权；

**💡 创新点**

创新点在于利用时间扭曲数据增强生成相对进度标签，避免绝对进度噪声，并通过自监督学习得到高质量的帧级进度估计；

**🔧 技术方法**

技术包括：时间扭曲采样器（基于AR(1)过程的速度变化与反转）、DINOv3视觉编码器、双向Transformer聚合、分类式进度回归以及基于进度的动作块加权；

**📊 数据集**

数据集为数千条人类远程操作的T‑shirt折叠与瓶子放置演示，其中T‑shirt折叠分为3个效率层级（≤60s、≤90s、≤120s）进行实验；

**📈 对比分析**

与传统BC、SARM、DemInf、SCIZOR等基线对比，WARP‑BC在T‑shirt折叠任务中从0/20提升到19/20成功率，吞吐率提升约18倍，吞吐率在更低质量数据集上仍保持高性能；

**⚠️ 局限性**

局限性在于对训练示例行为的依赖、使用反向播放生成负进度标签的真实性，以及在缺乏显式视觉进度线索的任务中可能效果不佳。

---

## 384. PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception

**arXiv ID:** 2606.28322 | [PDF](https://arxiv.org/pdf/2606.28322v1)

**作者:** Yana Wei `[一作]` (Johns Hopkins University), Vishal M. Patel `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PerceptionRubrics，一种基于规则的评估框架，用来衡量多模态模型在信息稠密图像上的真实感知能力。

**💡 创新点**

创新点在于：①构建极度信息密集的 1,038 张图像集合；②通过圆形同行评审生成高质量黄金描述并从中提炼 12,004 条原子化评估规则；③采用门控评分机制，将必需的 Must‑Right 规则视为门槛，确保关键事实的绝对准确。

**🔧 技术方法**

使用技术包括：多模态大模型（如 Gemini、GPT‑5、Seed 系列）用于黄金描述生成；LLM‑Judge（GPT‑OSS‑120B）进行规则评估；多轮循环审稿与人类验证的组合流程；双流评估体系（Must‑Right 与 Easy‑Wrong）。

**📊 数据集**

数据集：1,038 张信息稠密图像，配套 1,038 条黄金描述与 12,004 条原子化评估规则，涵盖自然场景、文档、UI、结构化数据、STEM、逻辑谜题和文化创意等七大领域。

**📈 对比分析**

与传统基准（如 DOCCI、DetailCaps、MMBench 等）相比，PerceptionRubrics 对 25 个模型进行了评估，显示出明显的性能差异：Open‑source 与 proprietary 之间约 8% 的感知差距；模型在 GUI、文档等稠密领域表现最差；在与 Vision Arena 人类偏好对齐度上取得 Pearson 0.916、Spearman 1.000 的最高相关性。

**⚠️ 局限性**

局限性：①评估依赖 LLM‑Judge 进行真伪判定，受模型推理误差与偏见影响；②构造黄金描述的圆形同行评审成本高，难以大规模扩展；③门控机制虽然提升严谨性，但对部分合理多义答案可能过于苛刻。

---

## 385. DexCompose: Reusing Dexterous Policies for Multi-Task Manipulation with a Single Hand

**arXiv ID:** 2606.28323 | [PDF](https://arxiv.org/pdf/2606.28323v1)

**作者:** Dihong Huang `[一作]` (University of North Carolina at Chapel Hill), Mingyu Ding `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了一种DexCompose框架，用预训练的多指操控策略在不重新训练的前提下，通过指尖动作归属与双重残差稳定器实现任务复合。

**💡 创新点**

创新点在于：①将握持与后续交互视为动作空间分配问题；②使用后置指尖归属检测识别可释放的自由度；③引入双重残差（保持握持与适配后续任务）实现结构化动作所有权。

**🔧 技术方法**

核心技术包括：后置指尖归属（release‑test+mask selection）、双重残差学习（bounded residual stabilizer 与 context‑aware residual）、掩码动作分配；使用Shadow Hand在Isaac Lab的仿真环境中训练与评估。

**📊 数据集**

数据集：为每个基准任务收集50条人类演示，用于训练基础策略；为任务A收集4096个保持状态用于残差训练；总共16种组合任务（4×4）。

**📈 对比分析**

与四类基线对比（Frozen grasp、Decomposed Action Space、Residual Learning、Ours‑ZS），平均复合成功率提升至77.4%，比最强基线高15.8个百分点，且在所有任务组合中表现最稳健。

**⚠️ 局限性**

局限性：目前仅支持两任务的顺序复合；缺乏针对更长时间尺度、多任务交互的扩展；对环境动态变化与更大规模动作空间的适应性待验证。

---

