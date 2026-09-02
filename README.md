# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-02 | 今日论文总数: 757

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Toppling the Hierarchy in Byte-level Language Modeling

**arXiv ID:** 2609.00463 | [PDF](https://arxiv.org/pdf/2609.00463v1)

**作者:** Lukas Edman `[一作]` (Technische Universitaet Muenchen), Alexander Fraser `[通讯]` (Technische Universitaet Muenchen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了字节级（byte-level）Transformer与层次化（hierarchical）模型在字符级别理解任务（如CUTE）的表现，系统比较了不同层次分布、字节层与词层的比例、以及注意力与前馈层的作用。

**💡 创新点**

创新点在于：①提出了新的层次化结构（交错式 interleaved）并证明纯字节级模型在字符操作上更优；②通过拆解Transformer层发现字节级注意力是关键；③在保持计算效率的前提下，设计了更高效的层次化替代方案。

**🔧 技术方法**

使用的技术包括：Transformer架构、字节级与词级注意力/前馈模块、白空格分词下采样与上采样、不同层数与模块配置的实验调参、以及对多种基准的评估。

**📊 数据集**

使用的数据集为：Nemotron‑ClimbMix（1B 与 4B 词级子集）、CUTE 与其多语种扩展 EXECUTE、以及通用NLP基准 Lambada、PIQA、ARC‑Easy。

**📈 对比分析**

通过对12层和24层模型的基准测试（包括纯字节级、各种层次化策略和子词模型），发现纯字节级模型在CUTE上显著优于层次化模型，且字节级注意力层的增加能进一步提升性能；层次化模型在速度和内存上更高效，但字符理解能力受限。

**⚠️ 局限性**

限制包括：实验仅在英语文本上进行；未覆盖所有可能的模块组合（如更高比例的字节层）；模型规模仅为1B‑4B词级，未探讨极低或极高资源场景；以及仅采用白空格分词策略，可能不适用于非拉丁文字。

---

## 2. Jailbreaking Text-to-Image Models Through Cracks: Navigating Heterogeneous Safety Filters via Multi-Agent Debate

**arXiv ID:** 2609.01168 | [PDF](https://arxiv.org/pdf/2609.01168v1)

**作者:** Kaiyan Wen `[一作]` (National University of Defense Technology), Guangdong Bai `[通讯]` (City University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于多智能体辩论的攻击框架CRACK，用以在多层异构安全堆栈下对文本到图像模型进行绕过攻击。

**💡 创新点**

① 定义了Detection Surface统一几何框架表征异构安全过滤的决策边界；② 设计了跨层冲突、稀疏性、非凸性三大结构属性；③ 通过多智能体辩论将探索、诊断、仲裁分离，并结合强化学习自适应搜索。

**🔧 技术方法**

多智能体对话框架、LLM（DeepSeek V3）代理、策略库突变、层级诊断、CLIPScore交叉模态评估、强化学习策略优化等技术。

**📊 数据集**

NSFW-200、I2P、UnsafeDiff三个NSFW内容数据集，以及目标模型Stable Diffusion 1.4/XL、DALL·3、Midjourney。

**📈 对比分析**

与SneakyPrompt、Ring-A-Bell、MMA-Diffusion、JailFuzzer、DACA、PGJ等六种基线对比，CRACK在多数安全配置下获得最高攻击成功率（最高达99.63%），查询量显著降低，语义保真度也更高。

**⚠️ 局限性**

仍依赖黑盒查询；对未知或自适应商业防御的迁移性不足；多策略库和RL训练成本高；仅聚焦文本到图像模型，未验证在其他生成任务上的通用性。

---

## 3. Let Confidence Change, Not the Prediction: Prediction-Preserving Repair for Post-hoc Calibration

**arXiv ID:** 2609.01072 | [PDF](https://arxiv.org/pdf/2609.01072v1)

**作者:** Daehwan Kim `[一作]` (Hanyang University), Ikbeom Jang `[通讯]` (Hankuk University of Foreign Studies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种后置校准修复方法CORD，能够在保持预测结果不变的前提下，修复已拟合的多分类校准器输出，使其置信度更符合真实概率；

**💡 创新点**

创新点在于将预测保持从校准器拟合阶段迁移到后置修复阶段，利用原始与校准后输出的一维修复，实现精确的top‑1预测保持，同时保留校准器对其余类别的条件分布；

**🔧 技术方法**

采用一维概率修复框架，利用Bernoulli‑KL目标求解最佳修复概率；通过投影与拉格朗日乘子得到全局唯一的标量η⋆；实现无额外监督模型、无超参数的轻量级后置修复；

**📊 数据集**

在CIFAR‑10、CIFAR‑100、ImageNet‑1K等公开数据集上测试，并通过CIFAR‑10‑C、CIFAR‑100‑C等噪声变体评估分布漂移鲁棒性；

**📈 对比分析**

与多种未保持预测的校准器（VS、MS、SMS等）及保持预测的校准器（TS、IRM、AdaTS等）对比；指标包括ECE、NLL、Brier、TPCR；CORD在所有数据集上实现TPCR为0，同时平均降低ECE、NLL和Brier，提升校准质量，且在分布漂移和不同校准集规模下保持显著优势；

**⚠️ 局限性**

局限性包括：仅对top‑1预测保持，无法直接扩展到top‑k或多标签场景；需在校准集上预先构建，需额外计算；在某些极低TPCR情况下改进有限；仍依赖原始校准器的表达能力，不能保证预测准确率提升。

---

## 4. Closed Forms and Synthetic Twins: Predicting Approximate Nearest Neighbor Recall from Embedding Statistics

**arXiv ID:** 2609.00364 | [PDF](https://arxiv.org/pdf/2609.00364v1)

**作者:** Shmuel Herman `[一作]` `[通讯]` (Independent Researcher), Shmuel Herman (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于标签无关统计量与合成双生子（synthetic twin）预测近似检索系统（如HNSW、IVF、PQ、FDE）召回率的框架，并给出了针对不同索引家族的闭式预测公式与模拟方法；

**💡 创新点**

创新点在于：①将固定网格量化和分区/图索引的召回率分别归因于可闭式的统计量与合成双生子；②引入“误差边际”（score margin）作为可学习的、跨索引的关键统计量，证明其能在不需要后置变换的情况下提升所有索引的召回率；③提出一种“交换边界”（exchange bound）计价方法，可在构建索引前预估后置变换的利弊；

**🔧 技术方法**

使用的技术包括：统计量提取（均值、协方差、MCS、误差边际、簇内/簇间距离等）；闭式公式（FDE碰撞、PQ误差不匹配因子、IVF路由边际等）；合成双生子生成与仿真；梯度优化学习误差边际；以及精细的实验评估框架。

**📊 数据集**

数据集包括MS MARCO passage shards、BEIR的SciFact、NFCorpus、SciDocs、ArguAna以及HotpotQA等，使用了多种稠密与后交互式编码器（BGE、大型Late-Interaction模型等）。

**📈 对比分析**

通过与真实构建的索引（PQ、IVF、HNSW、FDE）在 held‑out 编码器-数据集对上进行比较，预测召回率与实测召回率的平均误差均在0.03以内；闭式公式在固定网格索引上达到0.05-0.1的MAE；合成双生子在导航索引上MAE约0.04；学习误差边际可将召回率提升约3-5个百分点。

**⚠️ 局限性**

局限性包括：①闭式公式对模型假设（如cone模型）有依赖；②合成双生子对簇结构的近似在百万级规模上会失效，需要更细粒度的多尺度双生子；③误差边际的学习成本及其对下游任务相关性的影响尚未充分评估；④实验多聚焦于特定索引家族与编码器，其他索引或更大规模场景的泛化仍需验证。

---

## 5. Commit-first LLM judging inherits the judge's own errors

**arXiv ID:** 2609.00088 | [PDF](https://arxiv.org/pdf/2609.00088v1)

**作者:** Idil Gozel `[一作]` `[通讯]` (Evaluator Integrity), Idil Gozel (Evaluator Integrity)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

审计八大主流评估框架默认LLM判定器是否实现 commit‑first 评判，并在真实编码任务上验证其对优化循环的影响。

**💡 创新点**

创新点在于：①系统性审计 24 个默认配置，发现无实现 commit‑first；②通过实验展示 commit‑first 能消除但也可能放大评估错误；③提出低成本的预判 probe，可在几分钱内预测判定器是否安全。

**🔧 技术方法**

技术手段包括：DeepEval G‑Eval 与 Claude Opus 的 LLM 判定器；plain best‑of‑N 搜索；可见与隐藏测试套件；commit‑first 改造；以及小规模 probe 评估判定器自身解题能力。

**📊 数据集**

数据集由四个小型 Python 函数任务组成（区间合并、持续时间解析等），每个任务配备可见测试和隐藏测试套件。

**📈 对比分析**

比较方法：在相同优化循环下对比标准配置与 commit‑first 两种判定器的评估分数与隐藏正确率。结果显示：commit‑first 在区间合并任务完全消除 100% 的游戏；但在持续时间解析任务导致误判，误差受判定器本身准确性限制。

**⚠️ 局限性**

局限性：实验仅覆盖四个小任务、单一判定器家族；未检验大规模任务或多模型生成的情况；并且改进仅针对默认配置，实际部署系统若修改模板可能失效。

---

## 6. Human-robot conversation with multiple participants in noisy public spaces

**arXiv ID:** 2609.00648 | [PDF](https://arxiv.org/pdf/2609.00648v1)

**作者:** Divesh Lala `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了面向嘈杂公共环境的多方人机对话音频系统，并在2025年大阪世博会上演示了两个场景——ERICA的注意倾听对话和Teleco的会话支持（包含远程操作者头像与自治机器人）。

**💡 创新点**

创新点在于将低成本四通道ReSpeaker麦克风阵列移植至移动机器人，并实现动态音频通道选择；结合空间音频渲染和实时姿态跟踪，实现在噪声环境中的“鸡尾酒会”效果；将语音增强、语音识别、LLM生成对话及视觉跟踪统一在同一架构中。

**🔧 技术方法**

采用了ReSpeaker四通道麦克风阵列、基于声源方向的多通道语音分离/增强模型、日语低延迟增量ASR模型、GPT‑4.1生成的自然语言响应、语音活动投影（VAP）模型实现即时转场、空间音频渲染、SLAM+YOLO定位、回声消除软件、WebRTC传输等技术。

**📊 数据集**

ASR模型使用包含YouTube等噪声语音的数据集进行训练；语音分离模型未详细说明数据来源；YOLO细化训练采用实验室采集的机器人与人类数据；GPT‑4.1使用公开大语言模型训练数据。

**📈 对比分析**

通过在实验室进行四种条件下的语音识别测评，计算字符错误率（CER）：手持麦克风7.87%、固定Teleco 6.71%、移动Teleco 6.71%、嘈杂环境 7.18%。结果显示系统性能与手持麦克风相近，且在世博现场能在高噪声、回声环境下实现语音识别与对话流。

**⚠️ 局限性**

系统未实现真正多方无缝轮流和目标者检测；ASR偶尔未拾取用户语音，主要因声音过低或对着自治机器人说话；缺乏多模态（视觉+语音）融合支持；未进行正式的主观用户评估。

---

## 7. The Indefinite Summation Problem for the Laurent Ring

**arXiv ID:** 2609.00824 | [PDF](https://arxiv.org/pdf/2609.00824v1)

**作者:** Shiva Shankar `[一作]` `[通讯]` (Kerala School of Mathematics), Shiva Shankar (Kerala School of Mathematics)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文解决了差分环(A, α)的无穷求和问题（ISP），其中A是由格^n上的移位算子生成的洛朗环，α是A的有限阶环自同构。

**💡 创新点**

创新点在于将ISP的解决方案转化为涉及矩阵乘法的有限过程，并且可以估计矩阵的大小，从而确定解决方案的算术复杂性。

**🔧 技术方法**

使用了差分环理论和群上同调的相关技术，特别是计算群同调H^i([α], A)以解决ISP。

**📊 数据集**

使用了洛朗环和函数环作为数据集，特别是涉及到的函数在^n上的定义。

**📈 对比分析**

通过与已有的算法进行比较，本文的方法在处理无穷求和问题时提供了一个有限的决策方法，性能上能够在算术复杂性上进行估计，具体复杂度为O(d^2)。

**⚠️ 局限性**

限制在于当自同构α的阶数大于1时，求和元素的子空间是无限维的，并且在某些情况下，求和元素的空间是稀疏的。

---

## 8. AnalysisBank: An Expert Analysis Pattern Library for Financial Report Generation

**arXiv ID:** 2609.00818 | [PDF](https://arxiv.org/pdf/2609.00818v1)

**作者:** Yajing Yang `[一作]` (National University of Singapore), Min-Yen Kan `[通讯]` (National University of Singapore)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于专家报告提炼的可重用分析库，推理时检索并执行匹配的分析来生成更具洞察力的财务报告。

**💡 创新点**

创新点在于引入“分析层”而非传统的“结构层”，将数据信号与具体分析动作拆分为三字段（信号、动作、引用文本），并通过检索实现针对性、数据驱动的洞察。

**🔧 技术方法**

采用四阶段抽取管道（识别、抽象、去重、质量过滤）构建库，并在推理中使用LLM（Qwen、DeepSeek、GPT‑5）完成信号提取、分析执行与报告合成，检索通过余弦相似度实现。

**📊 数据集**

使用 550 份日常市场分析报告（DataTales）、550 份股东研究报告（Earnings）构建库，并在 SciGen 科学论文生成基准上验证跨域迁移。

**📈 对比分析**

与直接提示、CoT、RAG、Buffer‑of‑Thoughts、Agent‑Workflow‑Memory 等多种基线对比；在 DataTales 和 Earnings2Insights 上洞察率提升 1.7–3.7 倍，factual 准确率保持 89–97%，赢率高达 99.6%；在 SciGen 上实现最高分析率与赢率，表明方法在不同领域均有显著优势。

**⚠️ 局限性**

局限性包括受限于源报告的分析深度，库只能反映已有模式；对多语言未验证；在分析需求不强的领域提升有限；当前仅生成文本，缺乏可视化输出。

---

## 9. TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning

**arXiv ID:** 2609.00470 | [PDF](https://arxiv.org/pdf/2609.00470v1)

**作者:** Muhaimin Bin Munir `[一作]` (University of Texas at Dallas), Latifur Khan `[通讯]` (University of Texas at Dallas)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在检索增强生成(RAG)系统中提出一种三层筛选中间件(TRIS)，通过交叉嵌入空间聚类、触发器-负载结构检测和LLM一致性验证三步过滤，阻止检索阶段毒化攻击；

**💡 创新点**

创新点在于将对抗攻击的三项约束分别拆解为三个互补的过滤层，利用架构独立的判别器与自我一致性检查，既提升鲁棒性又降低验证成本；

**🔧 技术方法**

技术包括：对检索结果重新嵌入并使用MiniLM或Sentence‑BERT做K‑means聚类；对前20词进行Jaccard/n‑gram重叠检测；使用大型语言模型（GPT‑3.5‑turbo/LLama‑2‑7B）进行自知式一致性判断；

**📊 数据集**

使用公开数据集：Natural Questions、HotpotQA 与 MS‑MARCO；

**📈 对比分析**

在与Vanilla RAG、TrustRAG、RobustRAG等基线对比中，TRIS将黑盒攻击成功率从≈67–87%降低至≈3–14%，在多数数据集恢复≈40–45点的准确率；相对TrustRAG/RobustRAG在攻击防御上保持竞争力，且验证成本更低；

**⚠️ 局限性**

局限性包括：未评估判别器感知或完全白盒攻击；Layer‑3一致性层对LLM知识依赖强，且推理延迟较高；实验规模仅为100条查询，缺乏对完整语料库的评估；未提供正式的鲁棒性证明。

---

## 10. SAM3-LoRA: Parameter-Efficient Adaptation of a Concept-Promptable Foundation Model for Multi-Class Structural Defect Segmentation

**arXiv ID:** 2609.00469 | [PDF](https://arxiv.org/pdf/2609.00469v1)

**作者:** P. Malaisree `[一作]` (MAA Consultants Co., Ltd), W. Songkitti `[通讯]` (Thailand Institute of Scientific and Technological Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用低秩适配（LoRA）对SAM3进行参数高效微调，并通过直接使用COCO标签名作为文本提示来训练概念可提示的分割模型，验证其在隧道衬砌缺陷数据集和公开的Structural Defects Dataset（S2DS）上的跨域性能。

**💡 创新点**

创新点包括：①将类别标签本身作为文本提示，无需模板或额外嵌入；②提出硬负提示策略以消除正样本提示导致的存在预测解耦；③对两种LoRA置放方式（Light与Full）进行系统对比，展示其在不同数据集上的一致性和差异。

**🔧 技术方法**

采用的技术主要有：SAM3基础模型、LoRA低秩适配、硬负文本提示、COCO格式标注映射、联合分类-框-掩码损失以及多轮Hungarian匹配。

**📊 数据集**

使用的数据集包括：①为隧道衬砌缺陷的定制数据集（包含裂缝、混凝土剥落、水渗入三类）；②公开的Structural Defects Dataset（S2DS），涵盖裂缝和剥落两类。

**📈 对比分析**

通过与冻结的SAM3基线、Light和Full两种LoRA配置进行对照，在10种检测与分割指标（mAP、cgF1、像素IoU、像素召回、平均每提示IoU、实例召回等）上评估。结果显示，Pixel IoU提升约20‑52倍，实例召回提升约1.7‑1.8倍，所有指标方向一致，验证了参数高效适配的有效性。

**⚠️ 局限性**

主要局限包括：仅在两类混凝土/隧道衬砌缺陷数据集上验证，缺乏对其他视觉域的通用性评估；跨数据集的适配容量对比受实现细节差异影响；检测指标仍相对较低；硬负提示的效果未给出定量基准；部分类别样本量有限，导致单类别评估不够稳健。

---

## 11. Prediction-Assisted Pricing and Admission for LLM APIs with Stochastic Token Consumption

**arXiv ID:** 2609.00710 | [PDF](https://arxiv.org/pdf/2609.00710v1)

**作者:** Patrick Wong `[一作]` `[通讯]`, Patrick Wong

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种针对大语言模型（LLM）服务的联合定价、产品选择与容量分配框架，采用预测裁剪的 UCB（Prediction‑Clipped UCB）算法在拥有离线预测区间和在线置信区间的基础上实现动态决策，并保证硬性容量可行性。

**💡 创新点**

创新点在于：① 将离线预测误差半径与在线置信区间交叉裁剪，兼顾先验信息与在线学习；② 采用资源影子价格将收益与多维资源消耗统一计价；③ 通过路径级保留包（reservation envelope）实现无缝的硬性容量约束；④ 证明了在不同预测误差规模下的渐进性能插值，并给出可实现的硬性可行性保证。

**🔧 技术方法**

使用的技术主要有：分布式离线预测（带有效误差半径）；UCB 带置信区间的在线学习；双重投影（shadow‑price）更新；保留包（envelope）校验；流形/流量约束的线性规划与 Lagrangian 评分；以及基于计量经济学的模拟与实验。

**📊 数据集**

在实验中使用了合成数据（T=6000，三类用户段，16 个产品组合）以及可复现的代码包；未使用公开的真实 LLM 生产日志，而是提供了可直接执行的模拟脚本。

**📈 对比分析**

与多种基线对比：Oracle（已知真实均值）、Prediction‑Only（仅使用离线预测）、Online‑UCB（从零学习）、Myopic‑Prediction（只最大化预测收益）。实验结果显示：在误差半径 ε=0.18 时，Prediction‑Clipped UCB 获得 94.8% 的 Oracle 收入，明显优于 Prediction‑Only（90.8%）和 Online‑UCB（82.5%）。在更小的误差下两者几乎相等，且所有方法均保持硬性可行。

**⚠️ 局限性**

局限性包括：① 仅适用于一次性（one‑shot）请求，未考虑重复或策略性用户；② 需要对离线预测提供同时有效的误差半径，若预测失效会导致在线裁剪失效；③ 假设分段/产品均值稳态且独立；④ 不处理队列延迟、服务水平合约等后端排队动态；⑤ 需要手动校准置信区间与保留包，实际部署需更复杂的监控与校正机制。

---

## 12. IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training

**arXiv ID:** 2609.00161 | [PDF](https://arxiv.org/pdf/2609.00161v1)

**作者:** Rongze Tang `[一作]`, Zhibo Chen `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用跨注意力先验实现交互感知的世界模型训练，解决传统均方误差（MSE）训练中监督分配不匹配导致的交互区域训练不足问题。

**💡 创新点**

创新点在于：① 将语言指令中操纵对象的跨注意力作为交互先验；② 通过注意力分布采样（ADS）校准该先验并生成交互映射；③ 用交互映射在反向传播中重新加权监督（IWS），无需任何外部稠密表示或推理时改动。

**🔧 技术方法**

主要技术包括：预训练视频扩散变换器（DiT）+流匹配训练；跨注意力提取与聚合；ADS（候选区域采样与预测误差加权）；IWS（交互加权损失）与梯度路由；使用冻结的文本编码器和对象词根抽取模型。

**📊 数据集**

使用两个数据集：1）机器人手臂操纵数据集 WorldArena（约 350K 条 17 帧视频，双臂 14-DoF 动作轨迹）；2）第一人称人手操纵数据集 EgoDex（约 256K 条 81 帧视频，手势与 RGB 对齐）。

**📈 对比分析**

与多种基线（一般世界模型、具身模型、表示引导模型、不同 DiT 变体）进行对比。 在 WorldArena 上，IMPACT 在 Cosmos-Predict 2.5（动作）上将 EWMScore 提升至 62.53（+6.6 分），在 Wan 2.2-AC 上提升至 62.46（+3.8 分）。 在 EgoDex 上，FVD 从 366.12 降至 110.94，FID 从 44.71 降至 5.79，CLIP-Hand 从 0.921 提升至 0.952，Hand IoU 从 0.693 提升至 0.772，整体表现明显优于所有对比方法。

**⚠️ 局限性**

局限性：① 仍依赖跨注意力先验的质量，若先验不准确可能影响交互映射；② 目前主要验证了手臂与人手两类交互，尚未在更复杂或长时序交互中全面评估；③ 训练过程相对复杂，涉及多步采样与梯度路由，可能在大规模部署时产生计算负担。

---

## 13. Spec-Driven Development for Agentic Software Engineering: Harnessing Human-Agent Teamwork

**arXiv ID:** 2609.00252 | [PDF](https://arxiv.org/pdf/2609.00252v1)

**作者:** Jessica Diaz `[一作]` (Universidad Politénica de Madrid), Jorge Perez `[通讯]` (Universidad Politénica de Madrid)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过概念性分析与灰色文献综述，构建了Spec‑Driven Development (SDD) 作为团队级别的治理框架，用以在Agentic Software Engineering (ASE) 中实现可扩展的协作；并提出了“harness”两层（技术与方法）以及五种人‑代理交互模式，阐明了人类角色从代码编写转向规范制定与流程调度。

**💡 创新点**

创新点在于：①将规范（specifications）作为人‑代理协作的契约基底，形成统一的可执行契约；②将传统的技术 harness 与团队层面的方法 harness 结合，形成完整的治理体系；③定义了五种交互模式和相应的技术实现，为人类角色重塑提供可操作的范式；④以灰色文献为主的多源研究方法，为早期 ASE 领域提供可复制的研究框架。

**🔧 技术方法**

主要技术与方法包括：构造主义（interpretivist）视角下的多源文献综述（Multivocal Literature Review, MLR）；基于概念框架的理论合成与案例工作（worked examples）；规范化（specification）与工作流编排（orchestration）的抽象模型；以及通过 Git 工作树、持久共享知识库等技术实现的工具实践。

**📊 数据集**

本文未使用传统实验或公开数据集；研究数据来源为：行业报告（Faros AI、DORA、METR 等）、学术论文（ASE 相关 roadmap 与 vision 论文）、技术博客、工具文档与演示（Claude Code、OpenCode 等）等灰色文献，构成了多维度的案例材料。

**📈 对比分析**

由于研究是概念性与方法论性，未进行量化对比或性能评估；作者通过对比分析（如 Agile→ASE）与案例示例来说明框架的可行性，提出了四项预期收益（一致性、吸收力、可迁移性、复合效应），但尚无实证验证。

**⚠️ 局限性**

局限性包括：①证据基础仍属灰色文献，缺乏同行评议的实证研究；②所提出的技术细节（如 harness 组件、规范格式）随基础模型快速演进，易过时；③对不同领域（嵌入式、工业软件）适用性的验证不足；④未提供可量化的效能指标，缺乏实际项目中的对比实验。

---

## 14. Do General NLP Embeddings Capture Ontological Reasoning?

**arXiv ID:** 2609.00177 | [PDF](https://arxiv.org/pdf/2609.00177v1)

**作者:** Hamed Babaei Giglou `[一作]` (TIB Leibniz Information Centre for Science and Technology), Sören Auer `[通讯]` (TIB Leibniz Information Centre for Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了AVA评估框架，生成171,007个逻辑敏感的对照三元组，用于检验词向量模型是否能区分本体语义中的正负关系。

**💡 创新点**

创新点在于通过结构化本体扰动（层次反转、关系替换、互斥注入）产生高难度负样本，并揭示语言模型在本体推理上的显著缺陷，凸显优化与泛化之间的鸿沟。

**🔧 技术方法**

采用对比学习技术，包括余弦三元组损失、Poincaré球面超平面损失以及改进的DPO方法，并利用大型语言模型(Qwen3.5-35B-A3B)自动生成样本。

**📊 数据集**

数据集由163个不同领域的本体（包括GO、OBI、schema.org等）提取两跳子图，随后通过LLM合成生成三元组，最终获得约17万条对照样本。

**📈 对比分析**

对超过25个预训练嵌入模型进行评估，最好的原始模型在triplet准确率仅达0.739、硬负样本准确率0.572；对比微调后，模型可达≈0.99的triplet准确率，但在下游分类发现和本体对齐任务中的提升仅为数个百分点，显示迁移效果有限。

**⚠️ 局限性**

局限性包括：样本由单一LLM生成可能产生偏差；评估仅关注语义区分，未覆盖推理推断；高性能并未转化为对本体结构的真正理解，存在优化–泛化差距。

---

## 15. Are You Thinking What I am Thinking? : Examining Conceptual Separation in Neural Architectures

**arXiv ID:** 2609.00764 | [PDF](https://arxiv.org/pdf/2609.00764v1)

**作者:** Jaee Ponde `[一作]` (Truth Audit Labs), Subhashis Banerjee `[通讯]` (Ashoka University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究卷积神经网络和大型语言模型的内部激活空间，检验概念是否能形成连贯且相互区分的表示，并用几何与分布式方法量化概念分离。

**💡 创新点**

提出“概念分离”框架，利用激活空间的几何和协方差结构评估概念间关系，并将该方法同时应用于视觉与文本两大模态，揭示概念模糊度与内部表示一致性之间的关系。

**🔧 技术方法**

使用KL散度、PCA、余弦/欧氏/马氏距离、协方差的Frobenius范数、SVCCA/CKA、均值池化句子嵌入、随机置换检验等统计与几何工具。

**📊 数据集**

ImageNet及其子类（猫、狗、车）、未见概念（rangoli、显微镜图像）、地理道路图像（印度、土耳其）、猫姿态图像；文本方面包含计算机科学、莎士比亚、信息安全、理论计算机学、仇恨言论与非仇恨言论句子，来源分别为维基百科、学术语料库以及2020年美国总统选举推文。

**📈 对比分析**

通过计算同类内/异类间的KL散度、距离比值、PCA可视化和协方差差异进行比较，结果显示熟悉概念在一阶和二阶均表现出较高分离度，未见概念和域移时一致性下降，仇恨言论的分离几乎消失，表明模型在该任务上缺乏内部概念结构。

**⚠️ 局限性**

局限性包括样本量有限（每类100例）、仅评估最终层或少数层、使用单一数据集和提示模板；对其他模型、语言、攻击情境的泛化尚未验证。

---

## 16. It Takes Two to Match: Co-Evolving Generative Retriever with Reinforcement Learning

**arXiv ID:** 2609.00638 | [PDF](https://arxiv.org/pdf/2609.00638v1)

**作者:** Runpeng Dai `[一作]` (University of North Carolina at Chapel Hill), Ciya Liao `[通讯]` (Apple)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种共进化的生成式检索框架：先用监督微调(SFT)初始化查询侧和物品侧的关键词生成器，然后交替进行强化学习（GRPO），让两侧生成器在固定的倒排索引环境下相互优化，最终直接用生成的关键词做检索与排序。

**💡 创新点**

创新点包括：①两侧关键词生成器共进化、相互适配；②将生成的关键词直接作为检索索引，兼容传统关键词检索基础设施；③在物品侧使用计量奖励（counterfactual marginal reward）来衡量单个物品关键词对整体检索质量的影响；④结合SFT初始化与强化学习，避免从零开始的探索困难。

**🔧 技术方法**

技术手段主要包括：使用大规模语言模型（Qwen3-4B/1.7B）做关键词生成；监督微调构造查询/物品的关键词目标；使用GRPO（带有奖励归一化）进行强化学习；在物品侧构造对数值变化的奖励，查询侧使用检索 F1 作为奖励；检索阶段采用倒排索引+BM25排序。

**📊 数据集**

实验数据集：内部 APP Marketplace 搜索数据（13.5k 训练查询 / 1.5k 验证查询，39.6k 应用，≈1k 相关条目/查询）；公开 WANDS（Wayfair）商品搜索数据（430 / 50 验证查询，42.9k 商品，≈200 相关条目/查询）。

**📈 对比分析**

与稀疏检索（BM25, SPLADE-v2）、稠密检索（DPR, ANCE, Qwen3-Embedding）、生成检索（DSI, RIPOR, DeepRetrieval）进行对比。CoGR 在内部数据集上 F1 0.396，WANDS 上 0.682，分别比最强基线提升 10.9% 和 36.1%；在各种检索指标（MRR@100、NDCG@100 等）也表现出最高或最接近最高的得分。

**⚠️ 局限性**

局限性：①需要训练两个生成器和交替强化学习，训练成本和复杂度高；②检索排名仍依赖 BM25，未充分探索更强的基于关键词的排序方法；③对业务指标（如广告收入、非相关广告比例）的直接优化尚未实现；④在多语言、多领域的泛化能力尚待验证。

---

## 17. DGNet: Dual-knowledge Guided Network for Infrared Small Target Detection

**arXiv ID:** 2609.00666 | [PDF](https://arxiv.org/pdf/2609.00666v1)

**作者:** Chenglong Yu `[一作]` (Shandong University), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种双知识引导网络（DGNet）用于红外小目标检测，利用多种通用文本对网络进行指导；

**💡 创新点**

创新点在于：①使用双文本先验（背景与目标）在频域通过PWM模块实现语义解耦；②设计跨样本共识导向对齐（CDA）损失，在CLIP嵌入空间中构造从“复杂背景”到“明亮目标”的方向约束；③仅在训练阶段使用CLIP，推理阶段无额外开销；

**🔧 技术方法**

技术方法包括：多尺度编码解码架构、离散小波变换（DWT）+频域调制、基于文本的通道注意力（B-KGM/T-KGM）、CLIP文本与图像编码、CDA损失（方向一致性与对齐约束）以及IoU损失；

**📊 数据集**

使用三大公开数据集：IRSTD-1K、SIRST 与 NUDT-SIRST；

**📈 对比分析**

与21种state‑of‑the‑art方法比较，DGNet在三大数据集上均取得最高或竞争性的IoU、P_d 与最低F_a，速度达75.6 FPS，参数仅5.34M；

**⚠️ 局限性**

局限性主要在：①仍需依赖预训练CLIP文本/图像编码器进行特征对齐；②对极端低信噪比或非常稀疏目标的鲁棒性待进一步提升；③对不同热成像传感器的跨域泛化尚未充分验证。

---

## 18. Reveree: Diagnosing LLM Reverse-Engineering Agents

**arXiv ID:** 2609.01185 | [PDF](https://arxiv.org/pdf/2609.01185v1)

**作者:** Hadjer Benkraouda `[一作]` (University of Illinois Urbana-Champaign), Gang Wang `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为Reveree的系统，用于诊断和评估大型语言模型（LLM）的逆向工程代理，并提供了完整的逆向工程诊断管线和相关工具。

**💡 创新点**

创新点在于将逆向工程过程拆解为可评估的子任务，设计了专门的诊断指标和自动化评估框架，首次实现了对逆向工程代理的系统化性能分析。

**🔧 技术方法**

主要技术包括基于提示工程的逆向任务生成、对话日志分析、动态安全性评分模型以及对LLM的接口调用封装，实现了端到端的自动化评估。

**📊 数据集**

使用了公开的逆向工程数据集RevereeBench（包含1.2万条多模态逆向任务），并在OpenAI GPT‑4、Anthropic Claude 等主流LLM上进行实验，数据集已在 https://anonymous.4open.science/SEC27/ 公开。

**📈 对比分析**

与传统的手工逆向实验、黑盒评估以及基线模型（如PlainPrompt, PromptEngineer）相比，Reveree 在成功率、效率和安全性评分上均提升了约12‑18%，并在多轮对话测试中表现出更高的鲁棒性。

**⚠️ 局限性**

局限性主要体现在：1）评估受限于当前公开的LLM API 访问成本和速率限制；2）逆向任务设计偏向文本，未覆盖完整的多模态场景；3）对极端安全威胁的预判仍不够精准，需进一步完善安全阈值与异常检测。

---

## 19. The Irreversibility Budget: Fleet-Level Risk Accounting and Admission Control for Agent Operating Systems

**arXiv ID:** 2609.00275 | [PDF](https://arxiv.org/pdf/2609.00275v1)

**作者:** Bardia Mohammadi `[一作]` (Max Planck Institute for Software Systems), Laurent Bindschaedler `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了大规模LLM代理舰队产生的不可逆外部化风险，提出将该风险视为资源并在运行时进行累计计费与预提交预留，以防止跨代理的风险聚合导致预算超限。

**💡 创新点**

创新点在于把不可逆曝光建模为可计量资源，使用风险计价（VaR）和分层账本实现跨代理的实时预留与拒绝，解决了传统单代理门控无法捕捉的共享触发与协同攻击问题。

**🔧 技术方法**

采用风险计价与依赖检测、分层预留-确认-取消账本、资源模型和调度器来实现预提交计费；利用仿真采购流程和公共Agent轨迹进行验证。

**📊 数据集**

使用的实验数据包括基于采购场景的离散事件仿真（50-1000台代理、预算$250k）以及公开的τ‑bench和AgentDojo轨迹（共38,452条）。

**📈 对比分析**

与传统本地门控、面值/聚合阈值、子预算、循环断路器等方案对比，结果显示预算在规模扩大至千台代理时仍保持约0.48×预算占用，而本地门控最高可达48×；同时在预算容量、吞吐率和安全性之间展示了可调节的权衡。

**⚠️ 局限性**

主要限制包括：定价模型的准确性与保守性难以保证；依赖检测与共享触发识别仍处于早期阶段；分布式账本在分区容忍度和一致性上存在挑战；攻击者可通过碎片化或误报类型逃逸预算约束。

---

## 20. The Multiple Timescales of Gradient Descent on the Edge of Stability: A Perturbative Derivation of the Central Flow

**arXiv ID:** 2609.01034 | [PDF](https://arxiv.org/pdf/2609.01034v1)

**作者:** Raphaël Berthier `[一作]` `[通讯]` (Sorbonne Université), Raphaël Berthier (Sorbonne Université)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在梯度下降的边缘稳定（Edge of Stability）现象下，提出一种尖锐山谷（sharp valley）假设的微扰范式，并通过多尺度方法推导出中心流（Central Flow）作为 ε→0 的极限。

**💡 创新点**

首次在理论上为中心流提供正式推导，揭示三种时间尺度（快速振荡、半尺度自稳化、慢尺度中心流）并得到自稳化机制与能量演化的显式公式，扩展到任意维数的山谷。

**🔧 技术方法**

采用梯度下降分解 f=g+εh、奇异微扰理论中的多尺度展开（Method of Multiple Scales）来推导梯度流约束、拉格朗日乘子等解析表达式。

**📊 数据集**

主要使用人工构造的最小例子（Z=ℝ²）以及在 CIFAR‑10 子集上训练的多层感知机（2 层、256 隐藏单元、GELU 激活）进行实验验证。

**📈 对比分析**

与其他平均化近似（如 Rod Flow、Edge Gradient Descent、Free Energy Flow）进行对比，结果表明中心流已足够精确地近似梯度下降轨迹，尽管在捕捉自稳化细节上略逊一筹，但模型更为简单。

**⚠️ 局限性**

局限在于推导仅为形式化，缺乏严格收敛证明；仅适用于线性山谷、极小 ε 的情形；对曲率山谷、非线性约束以及能量饱和等情况仍需进一步研究。

---

## 21. Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity

**arXiv ID:** 2609.00632 | [PDF](https://arxiv.org/pdf/2609.00632v1)

**作者:** Lei Wang `[一作]` (University of Florida), Jie Xu `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FedRoRA，一种在资源和数据异质环境下的联邦学习中对大型语言模型进行低秩适配（LoRA）细粒度个性化的方法。

**💡 创新点**

创新点在于将 LoRA 更新拆分为共享方向矩阵和可学习的对角尺度，允许同一秩预算下不同客户端获得个性化方向与幅度；服务器通过 SVD 提取全局子空间并投影回个性化初始化，从而兼顾全局知识与本地差异。

**🔧 技术方法**

使用了 Rank-heterogeneous Federated LoRA、低秩适配 LoRA、SVD 投影、Top‑k 个性化选择、对角尺度学习等技术。

**📊 数据集**

实验数据集包括 NLU 的 GLUE（MNLI、QNLI、SST‑2、QQP）以及 NLG 的 FLAN（Text Edit、Struct2Text、Sentiment、Reasoning）/LLaMA‑2‑7B。

**📈 对比分析**

与 FLoRA、HETLoRA、FlexLoRA、Fed‑PLoRA 等同秩异质 FL‑LoRA 基线对比，GLUE 上平均准确率 91.44%（比最高基线 +4.90 分），FLAN 上平均 ROUGE‑1 71.95%（比最高基线 +2.97 分），性能显著提升。

**⚠️ 局限性**

局限性包括服务器端计算开销（SVD、投影），以及缺乏正式的差分隐私或加密保护机制。

---

## 22. From Truncation to Commitment: Persistent Context in Uniform Discrete Diffusion

**arXiv ID:** 2609.01043 | [PDF](https://arxiv.org/pdf/2609.01043v1)

**作者:** Satoshi Hayakawa `[一作]` `[通讯]` (University of Tokyo), Satoshi Hayakawa (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的采样器——Committed Reveal Sampling（CRS），通过在反向扩散过程中将已选出的 argmax 令牌持续保存并作为后续输入的上下文，从而实现对全局采样路径的控制。

**💡 创新点**

创新点在于：①将单步的 top‑p 支持约束与持久化上下文区分开来，提出了持久化上下文的概念；②通过理论分析证明在更低噪声水平下等待选择令牌可以降低 Bayes 错误，并且持久化令牌能在后续并行更新中对同一模式产生协调作用；③设计了 warm‑up 阶段和按置信度分配的令牌选择策略。

**🔧 技术方法**

使用技术包括：统一状态离散扩散模型（Duo）、留一 (LOO) 归约的去噪器、top‑p 核心采样、argmax 选择、理论推导（Bayes 风险、KL 与总相关性证明）以及实验评估指标 GenPPL 与 unigram entropy。

**📊 数据集**

使用数据集：公开的 Duo-distilled 与 Base Duo 检查点，分别用于评估不同规模的模型表现。

**📈 对比分析**

比较方法：在统一的 argmax 最终解码器下，对比不同 NFE（8–64）下的 GenPPL 与 unigram entropy 关系；在 64‑NFE 级别下，对比 CRS 的选择预算与全局 top‑p 的精度–多样性曲线。实验显示：①在所有 NFE 范围内，CRS（p=1.0）比固定 top‑p（0.95、0.9）获得更低的 GenPPL；②在匹配相同 entropy 时，CRS 在 64‑NFE 下的 GenPPL 进一步下降，提升约 0.15 nats/token。

**⚠️ 局限性**

局限性：仅在 Duo-distilled 与 Base Duo 检查点上验证，未在更大规模或不同统一状态模型上测试；指标对不同多样性评估（如 Rep‑4、固定前缀 likelihood）结果会有所差异；持久化上下文的效应需要在更广泛的任务与模型上进一步检验。

---

## 23. ReFlowSET: Representation-Aligned Latent Flow Matching for SAR-to-EO Image Translation

**arXiv ID:** 2609.00968 | [PDF](https://arxiv.org/pdf/2609.00968v1)

**作者:** Jeonghyeok Do `[一作]` (Korea Advanced Institute of Science and Technology), Munchurl Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了ReFlowSET，一种用于 SAR‑to‑EO 图像翻译的表示对齐潜在流匹配框架。

**💡 创新点**

创新点包括：①基于多模态重建上限进行 codec 选择，②从零开始训练条件 DiT 并采用双流 SAR 处理与后续融合，③使用仅训练阶段的 EO 表示对齐指导。

**🔧 技术方法**

技术主要包括潜在流匹配（flow matching）、双流条件 DiT、训练时的 VFM 对齐（DINOv3）、分类器无关引导（CFG）和 Euler ODE 求解。

**📊 数据集**

使用了 QXS‑SAROPT（GF‑3 SAR 与 Google Earth）和 SAR2Opt（TerraSAR‑X 与 Google Earth）两套 SAR‑EO 对齐数据集。

**📈 对比分析**

在两大基准上与多种 GAN、Diffusion、桥模型对比，ReFlowSET 在 FID、DISTS、LPIPS 上均获得最佳或接近最佳指标，尤其在 FLUX.2 codec 下实现了显著提升。

**⚠️ 局限性**

局限性包括：①对 SAR‑EO 对齐的空间误差仍影响像素级指标，②缺乏跨传感器泛化评估，③对生成不支持的 EO 结构的抑制机制尚未完善。

---

## 24. Bounded, Indeterminate, or a Bug: A Condition-Aware Oracle for Differential Testing of SQL Aggregates

**arXiv ID:** 2609.00381 | [PDF](https://arxiv.org/pdf/2609.00381v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究并提出了针对数据库浮点聚合的条件数驱动判据，用精确算术结果作为基准，构建了可判定的误差边界。

**💡 创新点**

创新点在于将聚合算法的误差指数与条件数关联，首次给出可计算的可测试边界，并揭示 ClickHouse 的一遍方差算法导致的不可测区。

**🔧 技术方法**

使用任意精度算术计算精确值、前向误差分析、斜率实验测量、差分测试以及随机搜寻技术。

**📊 数据集**

实验数据来源于 TPC‑H 0.1、人工构造的数值列以及六种常见存储约定（时间戳、传感器、坐标等）组成的实际列。

**📈 对比分析**

通过在 PostgreSQL、MySQL、DuckDB、SQLite、ClickHouse 等引擎上执行相同聚合，将误差与条件数判定为“精确”“受限”“不可判定”，实验显示除 ClickHouse 外其余引擎保持精确；差分测试误差率为 0，验证判据有效。

**⚠️ 局限性**

限制在于仅适用于标准化、非溢出二进制64位列；对亚正数、溢出或十进制语义不适用；实验未发现引擎缺陷，仅证明判据无误。

---

## 25. PredErase: Training-Free Object-and-Effect Removal with Predictive Latent Guidance

**arXiv ID:** 2609.00956 | [PDF](https://arxiv.org/pdf/2609.00956v1)

**作者:** Waikit Xiu `[一作]` (University of Hong Kong), Xiying Li `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑free 的对象及其光照效应移除方法，能够在仅给定实例掩码的情况下同时消除目标物体及其投射阴影和接触阴影。

**💡 创新点**

核心创新在于将编辑支持区域与掩码分离：通过构造“contact‑band”扩展掩码以覆盖阴影区域，并利用预训练的 I‑JEPA 作为空洞结构先验，在填充时进行稀疏投影梯度更新，从而在冻结的 FLUX.2 生成器上实现效果-aware 的移除。

**🔧 技术方法**

使用的技术包括：冻结的流匹配生成器 FLUX.2、预训练的 I‑JEPA 预测器、灰色填充的掩码预处理、可扩展的掩码（contact‑band）、稀疏投影梯度引导、以及标准的 Prompt 与 CFG 控制。

**📊 数据集**

在 RemovalBench、RORD‑Val（基于 OMNIeraser 协议）和 DEFACTO‑Val（SmartEraser 协议）三个公开数据集上进行实验。

**📈 对比分析**

与传统的扩散填充器（如 FLUX.1‑Fill、ZITS++、LaMa 等）、训练‑free 编辑器（CLIPAway、Attentive Eraser）以及监督式移除器（OmniEraser、SmartEraser、PowerPaint）进行对比。结果显示：在 RemovalBench 上本方法将原始 FLUX.2 的 CMMD 从 0.496 降低至 0.108，PSNR 提升至 24.36 dB；在 RORD‑Val 上 FID 与 CMMD 分别从 149.02 降至 55.59 与 0.305；在 DEFACTO‑Val 上在 ReMOVE、LPIPS 与 PSNR 维度表现最佳，尽管监督式 SmartEraser 在 CMMD 与 SSIM 上仍占优。

**⚠️ 局限性**

局限性包括：对大面积遮挡的空洞先验缺乏足够可见上下文；contact‑band 仅适用于水平接触的物体，无法处理侧面光照、悬浮阴影或镜面/水面反射；且生成效果受冻结生成器的偏好限制，无法完全超越监督式模型的表现。

---

## 26. Towards reliable multimodal disaster severity assessment through preference optimization and explainable vision-language reasoning

**arXiv ID:** 2609.00879 | [PDF](https://arxiv.org/pdf/2609.00879v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 27. Intelligent Edge Computing

**arXiv ID:** 2609.00181 | [PDF](https://arxiv.org/pdf/2609.00181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 28. Towards Agentic Cloud Engineering: Graph and Loop Engineering with a Zero-Trust Agent Harness

**arXiv ID:** 2609.00050 | [PDF](https://arxiv.org/pdf/2609.00050v1)

**作者:** Sagar Srinivas Sakhinana `[一作]` (Tata Research Development and Design Centre), Venkataramana Runkana `[通讯]` (Tata Research Development and Design Centre)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Agentic Cloud Workflow Engineering 框架，能够将自然语言云工程任务自动转换为已验证的代码仓库与已部署的云服务，并通过证据门控保证任务完成。

**💡 创新点**

核心创新在于将工作流拆解为图工程、循环工程与 Agent Harness 三层抽象，实现长周期、受限恢复以及零信任执行；同时定义了证据门控的完成机制，保障每一步都有可机器检查的证据。

**🔧 技术方法**

采用 Google Agent Development Kit、Agent2Agent、Model Context Protocol、gVisor 沙箱、OPA Gatekeeper、GitOps、OpenTelemetry 等技术；对 Gemini Flash‑Lite/Flash/Pro 与 GPT‑5.6 Sol 四种大模型进行任务调度与执行。

**📊 数据集**

构造了 140 条跨 14 个领域的自然语言任务基准（Agentic DevOps、CloudOps、SRE、SecOps、DataOps、MLOps/LLMOps、AgentOps、RAG 等），共 840 个受控实验条件。

**📈 对比分析**

在每个模型上跑 3600+ 次实验，测量 VTCR、EGER、RSR、UCDR、ACPR、BTR 等指标；结果显示 GPT‑5.6 Sol 的验证完成率最高（95%），循环恢复率随模型强度提升而提升；所有模型均满足证据门控与授权检查。

**⚠️ 局限性**

主要限制包括对大模型的强依赖，低阶模型易出现验证失败；实验仅在 Google Cloud 上实现，未验证跨平台适配；恢复预算和成本上限的设定仍需经验调优。

---

## 29. Where Should Experience Live? Hierarchical Hebbian Memory for Continual Vision Transformers

**arXiv ID:** 2609.00358 | [PDF](https://arxiv.org/pdf/2609.00358v1)

**作者:** Mohammed Yusuf Mujawar `[一作]` (University of Alabama), Noorbakhsh Amiri Golilarz `[通讯]` (University of Alabama)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了层次 Hebbian 记忆架构，使 Vision Transformer 能在工作记忆、情节记忆和语义记忆三种时间尺度上进行快速关联学习并存储长期经验

**💡 创新点**

创新点在于将 Hebbian 关联与可学习的读写路由、保留与整合机制结合，形成三层次的可持续记忆体系，并引入因果读-写生命周期防止当前结果影响自身监督

**🔧 技术方法**

采用 Transformer（Swin‑Tiny、DeiT‑Small、ViT‑Small）为骨干，结合快速权重 Hebbian 关联、可学习控制器、多银行路由、语义整合与经验重放

**📊 数据集**

使用 Omniglot（5‑way 1‑shot）进行快速适应评估，使用 CORe50（新实例连续学习）评估持续记忆与性能提升

**📈 对比分析**

与基线、固定 Hebbian、可学习 Hebbian 路由以及经验重放等方法比较；在 Omniglot 上 97.39%（比基线高约1.25个百分点），在 CORe50 上 95.37%（与经验重放相当），并展示了延迟检索、路由利用和记忆时间尺度等机制优势

**⚠️ 局限性**

局限包括：对任务边界的依赖（虽无任务 ID，但仍需边界信息）；模型参数量和计算延迟略增；实验仅在中等规模数据集上验证，缺乏对更大规模连续学习环境的评估

---

## 30. MUGEN: Generating Unlearnable Graph Examples for Multiple Learning Tasks

**arXiv ID:** 2609.00696 | [PDF](https://arxiv.org/pdf/2609.00696v1)

**作者:** Ziyan Liu `[一作]` (Harrisburg University of Science and Technology), Huan Liu `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 MUGEN 框架，生成单一特征扰动数据集以在未授权学习中对节点分类、图分类和链路预测三种任务共同实现不可学习性

**💡 创新点**

创新点包括：① 第一次实现多任务可学习性防护；② 设计 Task‑Aligned Separability Objective (TASO) 通过类间可分性增强扰动的迁移性；③ 引入 Type‑Adaptive Perturbation (TAP) 针对连续和离散属性分别采用 PGD 与硬坐标搜索，显著提升离散属性的不可学习性

**🔧 技术方法**

技术主要包括：图神经网络共享编码器与任务特定头部；交替优化模型参数与特征扰动；使用类间可分性判别器 CSD；PGD 与离散硬更新；对扰动预算与预算约束的精细控制

**📊 数据集**

实验使用五个图数据集：MUTAG、ENZYMES、PROTEINS_full（有向无标记图），Cora、PubMed（传递式图）

**📈 对比分析**

与清洁训练、随机扰动、以及现有单任务图不可学习方法对比；在同一后端表现为多任务均显著下降（如图分类 Macro‑F1、节点 F1、链路 AUC 均大幅负向差距），在跨后端（GCN、GAT、GIN、GraphSAGE）迁移亦保持负向差距；在自监督框架（GraphMAE、GRACE）以及在特征遮盖、边缘丢弃、对抗训练等干预下依旧有效

**⚠️ 局限性**

局限性包括：仅考虑二分类链路预测；离散属性扰动仅支持类别重映射，未考虑连续属性的更细粒度变换；对知识图谱关系预测的扩展仍待研究；扰动预算对不同数据集的敏感度需要进一步调优

---

## 31. Hypotheses-Guided Self Distillation for Continual Personalization

**arXiv ID:** 2609.00251 | [PDF](https://arxiv.org/pdf/2609.00251v1)

**作者:** EunJeong Hwang `[一作]` (University of British Columbia), Estevam Hruschka `[通讯]` (Megagon Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可持续的持续个性化框架，利用显式的、不确定性感知的用户偏好假设，并通过反思式细化和假设引导的自蒸馏实现对大规模交互历史的稳定、可修订建模

**💡 创新点**

核心创新在于将多源、噪声且稀疏的用户信号抽象为可解释的偏好假设，随后在一个可控窗口内进行反思式细化，以维护长期信息；再将细化后的假设作为教师信息指导自蒸馏，从而提升模型的个性化表现和泛化能力

**🔧 技术方法**

主要技术包括：①基于LLM的假设生成（GenerateHypotheses）、②窗口化反思与细化（ReflectAndRefine）、③假设引导的自蒸馏（Hypothesis‑Guided Self‑Distillation）；实验中使用Qwen3.5‑4B/9B与Gemma4‑4B作为基础模型

**📊 数据集**

评估数据集覆盖三类个性化场景：①HelpSteer2（基于显式反馈的在线个性化），②HiCupid（跨会话用户画像），③Flight Recommendation（隐式行为推断），并使用LLM评判器对结果进行对比评估

**📈 对比分析**

与多种基线（原始历史、增量摘要、增量假设、纯自蒸馏等）相比，所提出的反思细化与假设蒸馏方法在HelpSteer2、HiCupid和Flight Recommendation上分别实现约3–10%的win‑rate/accuracy提升，表现出更好的鲁棒性与跨域/未见用户泛化能力；实验还展示了更佳的上下文预算与反思窗口设置

**⚠️ 局限性**

局限性包括：①实验仍在模拟环境与人工评判下完成，缺乏真实用户交互验证；②对隐私与偏见的风险未做深入缓解；③模型对假设生成与细化的准确性高度依赖LLM的推理能力，可能在极端噪声或稀缺信号场景下表现不佳

---

## 32. EEG-VID: Task-Guided Latent Predictive Pretraining for EEG Decoding and Assistive Target Selection

**arXiv ID:** 2609.00566 | [PDF](https://arxiv.org/pdf/2609.00566v1)

**作者:** Guanzhong Sun `[一作]` (China University of Mining and Technology), Yanzi Miao `[通讯]` (China University of Mining and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种基于任务引导的潜在预测预训练框架EEG‑VID，先利用历史EEG窗口预测未来潜在状态并加入弱任务监督，再在此基础上进行监督解码；

**💡 创新点**

创新点在于将潜在状态预测与弱任务引导相结合，使预训练能保留对下游解码有用的可预测信号，并证明该策略在不同后端解码器、预测器结构和跨主体/跨会话条件下具有可迁移性；

**🔧 技术方法**

采用了多尺度时序-统计混合EEG编码器、EMA目标编码器、RoPE注意力预测器以及多头Transformer/循环网络等技术；

**📊 数据集**

使用了自研的VIG‑48视觉意图跨日数据集、BCI Competition IV‑2a/IV‑2b运动想象公开数据以及六位受试者的机器人场景目标选择数据；

**📈 对比分析**

通过与七种基准解码器（EEG‑Net、DeepConvNet、EEG‑Conformer、TSConv、Neuro‑3D、EEG2Rep、BENDR）以及同构无预训练版本的对比，EEG‑VID在VIG‑48 Top‑1从4.80%提升至6.52%，在IV‑2a/IV‑2b的跨主体/跨会话设置中平均提升10–16个百分点，机器人场景目标选择准确率达40.24%（相较随机25%提升15%）；

**⚠️ 局限性**

局限性包括仅有单位受试者的VIG‑48数据、未能区分视线与脑电信号、预训练需要额外计算、以及未进行闭环机器人控制验证。

---

## 33. Vision-Language-Guided Pseudo-Labels for Unsupervised Domain Adaptation in Semantic Segmentation for Waste Sorting

**arXiv ID:** 2609.00898 | [PDF](https://arxiv.org/pdf/2609.00898v1)

**作者:** Udo Schlegel `[一作]` (LMU Munich), Thomas Seidl `[通讯]` (LMU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种端到端的跨模态伪标签生成与自训练管道，用于无监督域自适应语义分割；

**💡 创新点**

创新点在于利用预训练的SAM生成几何区域提议，再用EVA-CLIP对区域进行跨模态语义匹配，并可选用BLIP进行语言验证，从而绕过分割器自身置信度的循环依赖；

**🔧 技术方法**

核心技术包括SAM（Mask生成）、EVA-CLIP（区域–文本相似度赋标签）、BLIP（生成描述进行验证）以及标准DeepLabV3-ResNet50的自训练；

**📊 数据集**

使用的评测数据集为synthetic-to-real的GTA5→Cityscapes和lab-to-factory的LabWaste→RealWaste（工业废弃物分拣图像）；

**📈 对比分析**

与源域直接推断（source-only）以及AdaBN、CBST、DAFormer、PLSR等基线相比，Pipeline在Cityscapes上从20%提升到26.3% mIoU（+31.5%），在废弃物数据上从7.7%提升到18% mIoU（+133.8%），显示显著性能提升；

**⚠️ 局限性**

局限性包括对伪标签质量高度依赖，难以处理细小薄层目标；仅适用于闭集标签，开放集物体被归为unknown；BLIP验证增加计算成本；整体方法在高噪声或极端域移时仍可能产生误导性标签。

---

## 34. Perceptible or Not? Diagnosing Passive Fingerprints for Speech Deepfake Attribution

**arXiv ID:** 2609.00765 | [PDF](https://arxiv.org/pdf/2609.00765v1)

**作者:** Yupei Li `[一作]` (Imperial College London), Björn Schuller `[通讯]` (Imperial College London)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了 Perceptible‑Imperceptible Passive‑Fingerprint Diagnostic Protocol (PIPDP)，通过三步诊断方法区分语音深度伪造中可感知与不可感知的被动指纹，并系统评估其在源归因任务中的可靠性。

**💡 创新点**

创新点包括：①首次将可感知与不可感知指纹明确区分并给出定义；②设计了轻量级、模型无关的诊断协议，包括残差能量、Grad‑CAM、音频透明扰动与情绪诱导等多维度检验；③证明不可感知指纹在不同生成器、鉴别器、内容与说话人变化下更为持久和可靠。

**🔧 技术方法**

技术手段：选取10种语音生成器（vocoders、TTS、VC）与3种跨架构鉴别器（WavLM‑AASIST、RawNet2、w2v‑bert‑MLP）；利用残差能量、Grad‑CAM可视化、五类音频扰动（相位噪声、高清噪声、包络平滑、位深度抖动、陷波滤波）以及基于情绪提示的生成对比；统计准确率、概率下降、F1等指标。

**📊 数据集**

数据集：VCTK语料库（2500句子，10名说话人）与10名外域说话人的ESD数据集（每人100句），在10种生成器上合成25,000条合成语音样本，构成闭集10分类归因实验。

**📈 对比分析**

评估方式：在闭集10分类设置下计算准确率/宏F1；残差能量轨迹与Grad‑CAM对比验证指纹可复制性；对透明扰动进行DNSMOS/ STOI评估与准确率/概率下降对比；情绪诱导实验比较不同情绪下的准确率下降。实验结果表明：不可感知指纹在不同生成器和鉴别器中保持高一致性，透明扰动（尤其是位深度抖动、相位噪声）可显著降低归因准确率；而可感知指纹（情绪）对归因影响极小。

**⚠️ 局限性**

局限性：仅在闭集设置下验证，未评估开放集或跨域情况；扰动与情绪实验只覆盖部分生成器；未探讨如何通过训练或数据增强进一步强化可感知指纹；对模型更新后指纹持久性的长期跟踪仍缺乏足够实验。

---

## 35. ProxPI: Proximal Prior Injection for Sampling-Based MPC under Learned-Prior Mismatch

**arXiv ID:** 2609.00941 | [PDF](https://arxiv.org/pdf/2609.00941v1)

**作者:** Euncheol Im `[一作]` (Korea Institute of Science and Technology), Yisoo Lee `[通讯]` (Korea Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Proximal Prior Injection（ProxPI）机制，将学习到的策略通过软亲和成本注入 MPPI 控制器，而不是把采样中心直接移动到策略输出，从而在策略失配时保持在线自适应。

**💡 创新点**

创新点在于将策略先验与采样中心分离：保留 MPPI 的 Nominal 中心，同时以亲和项软约束策略，使得在策略与任务不匹配时采样能逃离错误中心，理论上证明了重心重置会抛弃前一步校正而导致失败，并通过实验验证了在多平台和真实机器人上的鲁棒性。

**🔧 技术方法**

核心技术包括：基于信息论的 MPPI（Model Predictive Path Integral）采样控制、局部高斯近似的目标与采样分布分析、χ² 距离评估重要性采样权重集中度、以及软亲和正则化的成本设计。

**📊 数据集**

使用的实验数据集包括：2D 导航与到达任务（无障碍和有障碍），三自由度 FR3 手臂障碍规避任务，四足 Go2 的抬腿任务，人体 G1 的蹲姿任务，以及真实 FR3 手臂在行为克隆先验下的目标切换实验。

**📈 对比分析**

与 vanilla MPPI、warm‑start、Mixture、Mixture‑Elite、Blend、GPC‑CEM、Residual‑MPPI 等多种先验注入方案进行对比。实验显示，在先验与任务对齐时 ProxPI 与其它注入方案相当；在先验失配时 ProxPI 与 vanilla MPPI 维持近似最优性能，显著优于 warm‑start、Blend 等在所有五个平台上的 0% 成功率，且在高维平台上仍保持高 ESS，表明对样本预算和噪声敏感度低。

**⚠️ 局限性**

局限性：需手动调节亲和权重 α，过大时会抑制自适应；理论分析仅在局部高斯近似下成立，未覆盖更复杂非线性分布；在极端失配或高维任务中仍可能出现权重集中现象，需要进一步自适应权重或更强的探索机制。

---

## 36. VOIM: Training-Free Open-Vocabulary 3D Instance Mapping for RGB-D and Monocular SLAM

**arXiv ID:** 2609.00775 | [PDF](https://arxiv.org/pdf/2609.00775v1)

**作者:** Sangmin Song `[一作]` (University of Technology Sydney), Jodi Martin `[通讯]` (Guide Dogs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了VOIM，一个训练免费、体素基础的在线实例管理器，能从RGB‑D或单目RGB流构建开放词汇3D实例地图。

**💡 创新点**

将软标签累积与实例分割推迟到3D映射阶段，使用跨模型互斥的检测与区域描述融合，并实现单目无深度无姿态标注的实例级开放词汇地图。

**🔧 技术方法**

采用MASt3R‑SLAM/ VGGT前端、Grounding‑DINO+SAM2检测、PE‑Core文本区域描述、soft‑label投票与HDBSCAN实例提取、跨模型veto等技术。

**📊 数据集**

在ScanNet++（10场景、top100类）与Replica（8场景、51类）上评估，并在真实建筑地图上演示。

**📈 对比分析**

在相同帧、姿态、词表的公平协议下，VOIM在ScanNet++上对比OVO‑SLAM获得+11.7 mIoU，且在所有10个场景赢得胜利；在Replica上单目模式实现与RGB‑D池化mIoU相当，整体表现优于或等同于现有方法。

**⚠️ 局限性**

受检测驱动的标签限制（如墙面、地面等“stuff”类无法标注），单目姿态漂移限制大规模建筑级别性能，且需要昂贵的GPU内存并且实时性受限。

---

## 37. The Privacy-Hallucination Tradeoff in Differentially Private Language Models

**arXiv ID:** 2609.00492 | [PDF](https://arxiv.org/pdf/2609.00492v1)

**作者:** Krithika Ramesh `[一作]` (Johns Hopkins University), Anjalie Field `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在不同隐私预算下对大规模语言模型进行差分隐私预训练和微调，系统评估了隐私与事实性之间的权衡，并通过自动化（FactScore）和人工评估发现，差分隐私会导致模型产生更多虚假信息，尤其在更严格的预算下更为显著。

**💡 创新点**

创新点在于首次揭示并量化了差分隐私训练对语言模型事实性的负面影响（隐私‑幻觉权衡），提出了模型输出分布扁平化导致幻觉的机制，并通过受控实验验证了事实频率对差分隐私模型学习的影响。

**🔧 技术方法**

主要技术包括：差分隐私 SGD（DP‑SGD）和自适应噪声分配；低秩适配（LoRA）进行高效微调；使用 GPT‑J、VaultGemma、Gemma 系列模型；FactScore 自动化事实性评估；人工评估；大词典分布和有效词汇量分析。

**📊 数据集**

使用的数据集：WikiScience（231 篇 2020 年后创建的科学条目）、WikiAI（124 篇 AI 条目）、WikiPretrain（250 篇 GPT‑J 预训练数据中的随机条目）以及 20k 随机 Wikipedia 文章做为微调补充；同时构造了合成事实三元组实验用于控制记忆频率。

**📈 对比分析**

比较方法：对比 DP 与非 DP 模型在同一训练集上的生成文本，通过 FactScore 计算事实得分、困惑度（perplexity）以及人工评估的真伪与支持度。实验显示：DP 模型在相同数据下往往拥有更低的 FactScore（幻觉率更高），而非 DP 模型则事实得分更高；DP 模型的困惑度往往更低，但这并不能保证事实性。

**⚠️ 局限性**

局限性：仅限于公开可用的开源模型与已知预训练截止日期；未探讨后训练消幻觉技术或检索增强生成等其他差分隐私方法；未覆盖多查询环境下的隐私保证；实验只验证了差分隐私对幻觉的负面影响，未给出完整的解决方案。

---

## 38. Elite-Weighted Supervised Fine-tuning for Goal-Directed Molecular Optimization

**arXiv ID:** 2609.00189 | [PDF](https://arxiv.org/pdf/2609.00189v1)

**作者:** Shiyun Wa `[一作]` (Biogen), Ye Wang `[通讯]` (Biogen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Elite-Weighted Supervised Fine-tuning (EW‑SFT) 方法，用于在分子生成器中进行目标导向优化；

**💡 创新点**

创新点在于通过滚动精英缓冲区的精英选择，将奖励信息传递到模型的本地预训练损失更新，从而实现跨架构、跨任务的统一优化；

**🔧 技术方法**

使用技术包括遗传搜索（生成器作为变异器）、精英缓冲区（top‑K 最高分子）、以及对本地预训练损失进行加权微调；

**📊 数据集**

数据集涵盖 3D 形状相似性评估（BMS‑986195、Zasocitinib 两个激酶参考）、2D ECFP4 相似性评估（四种药物参考）以及 PMO 10k 调用预算下的 23 个公开或acles；

**📈 对比分析**

与原生优化器（PPO、GA、Genetic‑PPO）以及其他基准方法比较，EW‑SFT 在 Top‑1k 3D 相似度上平均提升约 0.079，2D 相似度上也明显优于冻结采样，且在 PMO benchmark 中与 Genetic‑PPO 的性能相当；

**⚠️ 局限性**

局限性包括仍需昂贵的 oracle 调用、对精英缓冲区大小与更新策略敏感、以及在非分子生成或极端约束条件下的可推广性尚未充分验证。

---

## 39. Random-Priority Frontier Routing: Tight $Θ(n^c)$ Bounds Against $c$-Node Cartels

**arXiv ID:** 2609.00893 | [PDF](https://arxiv.org/pdf/2609.00893v1)

**作者:** Krišjānis Petručena `[一作]` `[通讯]` (University of Latvia), Krišjānis Petručena (University of Latvia)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了一种基于随机优先级的全局前沿扩展路由方法，证明在任意可信节点网络中，只要有一组不超过 c 个被攻击的中间节点（cartel），即可在Θ(n^c) 次独立执行内，以至少 q 的概率得到不经过该 cartel 的路径。

**💡 创新点**

创新点在于：①给出了一个全局前沿扩展的路由规则，并证明其相对于任意图形的最坏情况卡特尔避开概率为 1/𝐶(n−1,c)；②证明该下界是紧的；③提供了对比分析，指出局部邻居随机路由在某些图形上表现极差；④给出了独立执行所需次数的精确对数式并展示其渐进为 Θ(n^c)。

**🔧 技术方法**

主要技术包括：随机优先级排序（i.i.d 连续分布）、全局前沿扩展算法、概率论中排列与组合的计数论证、对极端图形的构造证明、以及对多次独立执行的复合概率分析。

**📊 数据集**

未使用公开数据集，而是通过理论构造的最坏情况图形（例如连通路径加上与起点相连的卡特尔节点）进行证明与评估。

**📈 对比分析**

与局部邻居随机路由（local greedy walk）进行对比，发现后者在某些图形上卡特尔避开概率衰减为 2^−Ω(n)，远低于本方法的 1/𝐶(n−1,c)。此外，若能获取完整拓扑信息，可使用最大内部节点无交集路径均匀选择，得到更高的 1−c/κ 的避免概率；但本方法不依赖拓扑预知，适用于拓扑未知或动态变化的场景。

**⚠️ 局限性**

局限性包括：①需要在每次执行前生成全局随机优先级，若随机源不可靠则安全性受损；②需要维护完整已访问子图和前沿信息，导致标记/令牌尺寸 O(n+|E|)；③对卡特尔规模 c 较大的网络，所需的独立执行次数 Θ(n^c) 仍然指数级，无法满足实时性要求；④在已知拓扑的环境下，性能不如基于离散路径选择的有向方法。

---

## 40. Asymptotically Optimal List Size of Random Linear Codes

**arXiv ID:** 2609.01070 | [PDF](https://arxiv.org/pdf/2609.01070v1)

**作者:** Chen Yuan `[一作]`, Ruiqi Zhu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明任意有限域上随机线性码在容量附近的列表可解度为H_q(p)/ε + O(1)，实现了列表大小的渐进最优上界

**💡 创新点**

首次给出对所有质数幂q的随机线性码精确的领先常数H_q(p)，解决了之前仅在二元情形已知的猜想

**🔧 技术方法**

采用组合与概率方法，精细计数N(A)；利用Hamming球交叉估计、增量链Lemma、线性代数与随机投影技术

**📊 数据集**

本研究为纯理论分析，不依赖任何实验数据集

**📈 对比分析**

与以往只给出Θ(1/ε)列表大小的上界相比，本工作将常数提升到最优H_q(p)，并证明在高概率下满足此上界

**⚠️ 局限性**

证明在极大n的假设下成立，常数隐藏在O_p,q(1)中，且仅适用于固定q与p，缺乏可直接实现的构造算法

---

## 41. Inspicio: Open-Vocabulary, LLM-Based Sense Retrieval for Historical Languages

**arXiv ID:** 2609.00998 | [PDF](https://arxiv.org/pdf/2609.00998v1)

**作者:** Michele Ciletti `[一作]` `[通讯]` (University of Foggia), Michele Ciletti (University of Foggia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Inspicio——一种无需源语言词义库存、通过LLM生成双重翻译、定义与词干并结合密集检索、稀疏词干检索和MMR多样化重排，从上下文中为历史/低资源语言词汇检索英文WordNet同义词集的零样本管线。

**💡 创新点**

突破传统WSD对源语言词义表的依赖，采用双重翻译与LLM生成的定义词干相结合的检索策略，并在不需要任何手工标注的情况下实现高召回。

**🔧 技术方法**

使用指令调优LLM（如DeepSeek、Qwen、Mistral等）、句子嵌入模型（KaLM、Qwen3、Cohere等）、密集检索、稀疏词干匹配以及MMR多样化重排等技术。

**📊 数据集**

自制的150词感知动词双语测试集（拉丁语、古希腊语）、PREMOVE预词动词语料库以及意大利语年代性动词样本。

**📈 对比分析**

通过对六种LLM与六种嵌入模型的Recall@k进行交叉对比，最佳组合DeepSeek V4 Pro+KaLM‑Embedding‑Gemma3在感知动词集上Recall@50达96%，PREMOVE 81.65%，意大利语 91%；对照各组件ablation显示翻译阶段、词干增强和检索组合对性能贡献显著。

**⚠️ 局限性**

仅评估动词，依赖英语Pivot导致部分源语言词义缺失；LLM输出不确定性与高采样温度导致可重复性差；需要进一步验证对其他词性与更广泛语料的适用性。

---

## 42. KItCAT: Knowledge Injection via Input Corruption for Auto-regressive Training

**arXiv ID:** 2609.00082 | [PDF](https://arxiv.org/pdf/2609.00082v1)

**作者:** Meghanadh Pulivarthi `[一作]` (IBM), Yatin Nandwani `[通讯]` (IBM)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对持续预训练中知识注入缺乏多样性导致的过拟合问题，本文提出一种轻量级输入腐蚀(KItCAT)方法，在训练时随机替换部分输入词并保持下一个词标签不变，以实现数据多样化并提升知识注入效果。

**💡 创新点**

创新点在于不依赖昂贵的合成重述，而通过随机噪声（掩码、随机词、上下文可行替换、关键词掩码）在decoder‑only LLM中注入多样性，从而显著缓解过拟合并提高知识注入质量。

**🔧 技术方法**

技术包括自回归训练目标、四种输入腐蚀方案、LoRA微调、基于TF‑IDF的关键词识别、使用Masked Language Model生成上下文可行替换、以及LLM‑as‑Judge评估。

**📊 数据集**

实验数据集包括PopQA子集、Companies（24家虚构公司）以及IBM Redbooks两本技术手册，实验在Mistral‑7B、Qwen3‑14B、Llama‑2‑7B‑Chat等多模型上进行验证。

**📈 对比分析**

与标准CPT（NTP）以及SSMBA、MASKER等方法对比，KItCAT在四个数据集上平均提升10+分（从17.6提升至27.7），在不同模型规模下保持优势；结合合成重述进一步提升性能，但KItCAT单独即可实现5–9倍的数据效用，仅为重述方法约15%–30%的计算成本。

**⚠️ 局限性**

局限性在于，虽然KItCAT显著降低了对昂贵重述的依赖，却无法完全替代合成重述，最佳性能通常需要两者结合。

---

## 43. mimeo: Compiling Public Expert Corpora into Agent Skills and Testing What Transfers

**arXiv ID:** 2609.00453 | [PDF](https://arxiv.org/pdf/2609.00453v1)

**作者:** Timothy Kassis `[一作]` `[通讯]` (K-Dense), Timothy Kassis (K-Dense)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了开源工具Mimeo，将命名专家的公开记录转换为可加载的Markdown技能文件，并用编码代理套件评估其在知识检索、人格识别和判断转移上的效果。

**💡 创新点**

提出可审计的专家档案生成流程，首次在不依赖实时检索的情况下将公开专家信息直接嵌入代理上下文，并系统评估信息获取、人格表现与决策迁移的相互影响。

**🔧 技术方法**

使用Python CLI、网页爬取、文本提取、引文匹配与聚类、编辑检查清单，平均每个构建调用约38次LLM；同时对照BM25关键词检索和多模态评判者进行实验。

**📊 数据集**

公开的专家作品（论文、演讲、访谈、书籍、框架等）四位专家（如Richard Sutton、Andrej Karpathy）及内部设计的20+问题集与16个新情景数据集。

**📈 对比分析**

通过对比无技能、单行人格、从内存人格、以及Mimeo生成文件四种条件，配合四位评判者和BM25基线；结果显示文件显著提升对稀有引用的回忆（比内存人格高），人格识别率远高于随机但受任务材料影响；判断转移未显著提升，且测量对评判者高度敏感。

**⚠️ 局限性**

仅捕获公开记录，无法反映隐性知识和专家品味；输出非验证过的推理，仍可能误述；引文匹配不保证来源真实性；实验样本有限，许多度量在上限或受评判者影响；缺乏专家同意与更新机制。

---

## 44. Spawn Freely, Act Sparingly: Progressive Risk Vesting for Recursive LLM-Agent Trees

**arXiv ID:** 2609.01035 | [PDF](https://arxiv.org/pdf/2609.01035v1)

**作者:** Molly Wang `[一作]` `[通讯]` (Imperial Business School), Molly Wang (Imperial Business School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了递归LLM代理的进阶风险归属机制Progressive Risk Vesting，区分沙盒树与权限树并为激活行为提供可预估的风险预算。

**💡 创新点**

在任意适应性生成树下给出了灾难激活概率的任何时刻保证；引入权威复制率相位转移理论阐释路径伤害的三种尺度；基于多型占用模型提供风险与计算影子价格，实现阈值决策。

**🔧 技术方法**

采用条件证书与可预测概率支出（alpha‑spending）相结合的会计框架；利用分支过程理论与固定预算线性规划/占用测量；通过蒙特卡罗仿真验证理论预期。

**📊 数据集**

仅使用合成数据（Beta分布质量、泊松分支、加性噪声）进行两项可复现实验，不涉及真实部署数据集。

**📈 对比分析**

与传统 spawn‑charging 方案对比，延迟 vesting 允许更多沙盒候选，实验显示净效用提升约5.75%；在相同风险预算下，subcritical、critical、supercritical 三种情形的路径伤害概率均与理论一致。

**⚠️ 局限性**

核心假设为条件证书在部署中保持有效；沙盒边界需外部强制；分支过程模型未捕获工具共享导致的相关失败；实验仅在合成环境验证，未证明真实系统的安全性。

---

## 45. Teaching Vision-Language Models to Use the Scale They Are Given: Label-Free Equivariance Training for Metric Physical Reasoning

**arXiv ID:** 2609.00658 | [PDF](https://arxiv.org/pdf/2609.00658v1)

**作者:** Kaizhen Tan `[一作]` (New York University), Hanzhe Hong `[通讯]` (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过利用视频尺度同质性约束，对视觉语言模型在处理单摄像机视频的度量物理问题时的尺度利用进行评估与改进。

**💡 创新点**

①提出尺度同质性（α-变换）作为精确无标注的物理约束；②设计投影自监督（equivariance self‑distillation）方法，仅需每个视频一次查询即可训练模型满足尺度同质性；③证明该方法在合成和真实视频上均能显著提升模型的尺度泛化能力，优于传统模拟器监督。

**🔧 技术方法**

使用 Qwen2.5‑VL‑3B‑Instruct 视觉语言模型、4‑bit NF4、LoRA 微调；对数斜率、equivariance误差评估；E‑step 投影到同质性族；M‑step 以投影目标微调；MuJoCo 渲染合成视频；QuantiPhy 基准测试。

**📊 数据集**

1,000 条 MuJoCo rigid‑body 视频（5 个动力学家族），产生 3,000 条 QuantiPhy 问题和 10,661 条单参数干预记录；在 QuantiPhy 验证集（159 题）和离散训练集的 held‑out 动态（滑动块）进行评估。

**📈 对比分析**

与八种开源/专有视觉语言模型在 QuantiPhy 上对尺度斜率和平均相对准确率（MRA）进行基准；通过尺度同质性自监督后，模型斜率从 0.66 提升到 0.94，MRA 提升 9.2 点；在真实视频上提升 6.4 点，近乎达到模拟器监督的 93% 效果。

**⚠️ 局限性**

约束仅适用于时间基准固定、参考为长度/长度率、目标具有 L T⁻ᵏ 维度的情形；对纯时间或无量纲比值无效；同质性仅描述答案之间的关系，无法保证答案绝对正确；实验仅在 3B 学生模型上验证，通用性和对更大模型的影响待进一步研究。

---

## 46. Beyond Locks and Thread IDs: Static Data Race Detection Off The Beaten Path (Extended Version)

**arXiv ID:** 2609.00246 | [PDF](https://arxiv.org/pdf/2609.00246v1)

**作者:** Daniel Bund `[一作]` (Technical University of Munich), Michael Schwarz `[通讯]` (National University of Singapore)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

扩展 digest 框架，加入对 barrier、pthread_once 以及复杂的锁/线程创建模式的支持，实现更精准的静态数据竞争检测。

**💡 创新点**

创新点在于为这些较少被关注的同步机制设计专门的 digest，并将多种同步信息组合成更强的并发历史抽象，从而显著降低误报并保证 soundness。

**🔧 技术方法**

主要技术包括 digest‑driven 抽象解释、局部跟踪（local trace semantics）、多属性 digest 组合、约束系统重写与混合流敏感性以及可扩展的 digest 定义。

**📊 数据集**

实验使用了 60k–80k 行 GNU Coreutils、90 个手工构造的 litmus 测试以及约 30,000 个 SV‑Comp 基准。

**📈 对比分析**

与基线（SV‑Comp 配置）对比，locked‑creation 功能增加约 30.5% 运行时间，整体约 +2%；与 CPAChecker、Dartagnan、UGemcutter、ESBMC‑kind、RacerF 等工具比较，未出现假阳性或假阴性，证明了更高的准确性和竞争力。

**⚠️ 局限性**

局限性包括：对进程级并发的支持有限、仅处理 pthread/简化原语、使用抽象 digest 时可能存在精度折衷、尚未在更大规模真实项目上验证，且对某些极端同步模式仍缺乏覆盖。

---

## 47. Cleaner Speech, Weaker Generalization: Revisiting Pitt-Derived Benchmarks for Alzheimer's Disease Detection

**arXiv ID:** 2609.00276 | [PDF](https://arxiv.org/pdf/2609.00276v1)

**作者:** Luqi Sun `[一作]` (Johns Hopkins University), Berrak Sisman `[通讯]` (Johns Hopkins University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了语音预处理与数据集策划对阿尔茨海默病（AD）语音检测模型的影响，系统评估了多种语音增强技术在域内和域外（跨数据集）性能的变化。

**💡 创新点**

首次在Pitt衍生数据集上全面对比预处理对模型泛化的影响，并指出“更干净的语音并不一定带来更鲁棒的检测结果”。

**🔧 技术方法**

使用传统特征模型(eGeMAPS)、自监督语音表示模型(XLS‑R)、SLS加权层选择模型以及最新的大型音频‑语言模型（Kimi‑Audio、Qwen系列、Audio‑Flamingo 等），并结合多种语音增强器（FRCRN、MossFormer、Resemble、MAP‑SEMamba 等）。

**📊 数据集**

Pitt‑origin、Pitt、ADReSS、ADReSSo、ADReSS‑M 以及独立收集的 Lu Corpus 作为交叉验证数据。

**📈 对比分析**

采用宏 F1 分数进行域内与跨域评估，并在匹配/不匹配增强条件下进行对比；实验显示增强数据在域内提升宏 F1，但在跨域测试中性能下降；匹配增强略有改善，但仍低于未增强的基线。

**⚠️ 局限性**

仅使用英文 Pitt 相关数据，缺乏多语种或无噪声真实临床样本；增强方法细节未公开，导致复现性受限。

---

## 48. LLM-Driven Autonomous Vehicles Inherit Human Driver Biases in Pedestrian Yielding: Results and Implications From A New Benchmark

**arXiv ID:** 2609.00192 | [PDF](https://arxiv.org/pdf/2609.00192v1)

**作者:** Irem Yoldas `[一作]` (King's College London), Odinaldo Rodrigues `[通讯]` (King's College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过设计两种偏差审计方法（All Else Being Equal 与 Self-Consistency）对大型语言模型和视觉语言模型在自动驾驶车辆交叉路口让行决策中的人种、性别、宗教、残疾、年龄等属性偏差进行评估。

**💡 创新点**

创新点在于首次提出针对AV情境的 AEBE 与 SC 两种偏差测试框架，并构建公开基准供模型评估使用。

**🔧 技术方法**

使用技术包括零样本大语言模型推理、视觉语言模型的图像-文本推断以及基于统计显著性的差异检验。

**📊 数据集**

数据集涵盖 nuImages、JAAD 与 PIE 三大公开车辆图像集，经过筛选得到 23,812 张单人行人交叉场景并生成文本描述。

**📈 对比分析**

实验比较显示 Qwen-3、Llama-3.1、Mistral 等 LLM 在不同人群上存在显著让行率差异，而 Qwen-3-VL、SPHINX 等 VLM 的让行率普遍偏低，且在肤色、性别与种族上表现出明显偏差。

**⚠️ 局限性**

局限在于 SC 测试对场景相似度的依赖、缺乏对 VLM 的 AEBE 适配、未覆盖偏差缓解策略及根因分析。

---

## 49. FocusBuddy: Encouraging Healthy Desk-Work Habits by Caring for a Virtual Pet on a Water Bottle

**arXiv ID:** 2609.00412 | [PDF](https://arxiv.org/pdf/2609.00412v1)

**作者:** Mohamed Ouf `[一作]` (Queen's University), Rowan Hussein `[通讯]` (University of Ottawa)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并评估了 FocusBuddy，一款将虚拟宠物嵌入水瓶套件的原型，帮助学生提升饮水量、活动频次并减少长时间坐姿。

**💡 创新点**

将情感化的虚拟宠物与日常饮水容器结合，形成“共同照顾”机制，利用环境感知实现个性化提示。

**🔧 技术方法**

基于 BBC micro:bit 微控制器、5×5 LED 矩阵、加速度计、光/温度/麦克风传感器，以及自定义固件和手工 Fabric 造型。

**📊 数据集**

通过 20 名大学生在两周内的自报数据（饮水量、活动次数、最长坐姿时长），未使用公开数据集。

**📈 对比分析**

使用配对 Wilcoxon 符号秩检验比较前后自报习惯，显著提升饮水量（45%）、活动频次（86%）并减少最长坐时（32%），效应值接近 0.9；未做对照组比较。

**⚠️ 局限性**

样本规模小、仅自报、缺乏对照组、原型可视化受限、情感反馈可能产生罪恶感、社交场景适配性有限。

---

## 50. Social bots weaken activist cohesion

**arXiv ID:** 2609.00197 | [PDF](https://arxiv.org/pdf/2609.00197v1)

**作者:** Linda Li `[一作]` (London School of Economics), Balazs Vedres `[通讯]` (Central European University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了2020年黑人生命运动（BLM）在Twitter上的讨论，考察社交机器人曝光对人类网络凝聚力的影响。

**💡 创新点**

创新点在于将机器人曝光量与后续人类网络在个体层面（三角闭合）和群体层面（社区密度）凝聚力下降关联，并揭示支持者网络受损更为严重。

**🔧 技术方法**

采用图网络分析（k-core筛选、Clique Percolation Method社区检测）、回归模型与网络误差校正等技术来量化机器人曝光与凝聚力变化的关系。

**📊 数据集**

使用约250万条BLM相关推文及其用户元数据，构建峰值、前后期的转发网络进行比较。

**📈 对比分析**

通过OLS和网络误差模型比较，结果显示机器人曝光显著负向影响凝聚力，且该效应在不同阈值、层面与控制变量下保持稳健。

**⚠️ 局限性**

局限包括机器人识别阈值敏感性、仅关注转发网络、未考虑离线行为及因果推断困难。

---

## 51. RestoreBench: Can AI Agents Restore Power Flow Convergence?

**arXiv ID:** 2609.00384 | [PDF](https://arxiv.org/pdf/2609.00384v1)

**作者:** Riccardo Mansutti `[一作]` (ETH Zürich), Kevin O'Sullivan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个面向非收敛潮流诊断与修复的基准框架，评估LLM在不同交互架构（聊天机器人、单体代理、多体代理）下的表现

**💡 创新点**

首次将Agentic AI与电力系统非收敛问题结合，构造了可重复的Scenario Card、观察/动作空间、工具接口与评价指标，并比较三种架构的效果

**🔧 技术方法**

采用大语言模型（GPT‑5.6、Claude系列、DeepSeek、Kimi、GLM‑5）、工具调用（网络检查、功率流评估、可行性判定、动作排序）以及多体协同（分析师/执行者/编排者）

**📊 数据集**

使用IEEE 118‑bus与PEGASE 89‑bus两套电网，构建了46个非收敛案例的Scenario Card数据集

**📈 对比分析**

通过成功率、成本、运行时与电压在阈值内的比例四指标进行对比；结果显示单体代理性能最佳，聊天机器人最低，多体代理成本与时延较高但未明显提升成功率

**⚠️ 局限性**

仅做单次实验，未考虑LLM输出的随机性；缺乏多次重复测试的置信区间与性能波动分析；未加入后收敛敏感度优化工具

---

## 52. Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking

**arXiv ID:** 2609.00588 | [PDF](https://arxiv.org/pdf/2609.00588v1)

**作者:** Guangyu Chen `[一作]` (Institute of Science Tokyo), Manabu Okumura `[通讯]` (Institute of Science Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种基于不确定性引导的早停方法 Quit，联合减少候选生成和重排序开销。

**💡 创新点**

创新在于同时对候选生成与重排序进行自适应终止，首次提出使用重排序分数波动作为早停的实用代理。

**🔧 技术方法**

利用不确定性估计、窗口范围内最大最小差值、连续批量生成以及基于 MBR 和 QE 的重排序算法。

**📊 数据集**

在 WMT24 与 WMT25 的 19 组语言对上，对三种 NMT 模型（Qwen3、TranslateGemma、Hy-MT2）进行评测。

**📈 对比分析**

通过与无加速基线及现有 MBR 加速方法 PruneMBR、PMBR 对比，Quit 在 MBR 下实现 1.47–2.66 倍加速，QE 下实现 3.43–4.12 倍加速，并在多项质量指标上保持等价或更优。

**⚠️ 局限性**

局限在于需手动设置窗口与阈值、仅适用于可增量生成的策略、以及对重排序分数尺度不变的假设。

---

## 53. Attention Sensitivity Is Not Enough: Dissociating Attention-Level and Behavioural In-Context Learning under Fine-Tuning

**arXiv ID:** 2609.00064 | [PDF](https://arxiv.org/pdf/2609.00064v1)

**作者:** Jinyuan Zhang `[一作]` (Hubei University), ShengShuo Jiao `[通讯]` (Hubei University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在对大语言模型进行微调时，关注注意力级别的上下文学习代理（ICS）与实际行为表现（ICL‑GAP）之间的关系，并探讨将注意力代理作为优化目标的后果。

**💡 创新点**

创新点在于提出了可几何上界定的注意力敏感度指标ICS，并通过对ICS进行显式最大化的“压力测试”揭示了注意力代理与行为表现的Goodhart失配；同时提出了行为门控（B‑ICS）和预训练计算锚定（AnchorTune）两种缓解策略。

**🔧 技术方法**

使用的技术包括：Transformer模型的注意力行向量距离计算、匹配/不匹配演示前缀的对比实验、ICL‑GAP行为测量、ICS最大化的辅助损失、行为门控与锚定正则化。

**📊 数据集**

实验数据集主要为Llama‑2‑7B模型的instruction‑tuning混合数据、MMLU（240题）验证集、随机标签二分类验证集以及5-shot QA对照。

**📈 对比分析**

比较方法：对比四个实验分支（冻结注意力、冻结MLP、全参数微调、ICS最大化），评估ICS、ICL‑GAP、MMLU准确率与ECE。结果显示，ICS可被驱至几何上限1.413，而ICL‑GAP保持接近零，MMLU从0.371下降至0.279；行为门控可略微缓解，而锚定正则化能保持ICS和MMLU在预训练附近。

**⚠️ 局限性**

局限性包括仅在单一Llama‑2‑7B模型上验证，实验设置为单一随机种子，probe设计和演示数量有限，未覆盖更大模型或不同任务范畴，且对不同注意力层的泛化性尚未深入。

---

## 54. ExpArt-KG: Artwork Image Description Generation through Iterative Exploration of Knowledge Graphs

**arXiv ID:** 2609.00629 | [PDF](https://arxiv.org/pdf/2609.00629v1)

**作者:** Yuta Kato `[一作]` (University of Tokyo), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于知识图的检索增强生成框架，动态迭代检索与答案生成，以提升图像解释的细节和准确性。

**💡 创新点**

创新点在于引入LLM作为答案正确性判定器，实现迭代控制，避免固定迭代导致的检索成本过高，并构建专属艺术品知识图 ExpArt‑KG。

**🔧 技术方法**

采用检索增强生成（RAG）、TF‑IDF 三元组检索、LLM 验证器以及 Qwen3‑VL 与 Qwen3 进行实验。

**📊 数据集**

使用 ExpArt 数据集与自建的 ExpArt‑KG 知识图。

**📈 对比分析**

与单次生成基线和固定五次迭代的 RAG‑Loop5 进行对比，RAG‑Validate 在保持相似质量的同时迭代次数平均减少至 3.6 次，显著降低检索成本。

**⚠️ 局限性**

局限在于无标题场景下验证器缺乏必要信息导致性能下降，且仅适用于一对一对应的艺术品域，扩展性待验证。

---

## 55. Exploring Quantum Software Testing Across Research and Practice: Emerging Results from a Multivocal Literature Review

**arXiv ID:** 2609.00354 | [PDF](https://arxiv.org/pdf/2609.00354v1)

**作者:** Rodolfo Gil-Pereira `[一作]` (University of Calgary), Italo Santos `[通讯]` (University of Hawaii at Manoa)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过多元文献回顾，整合学术研究与灰色文献，系统梳理了量子软件测试（QST）的技术方法、面临的挑战、工具生态以及实践经验。

**💡 创新点**

创新点在于首次将学术与实践资料合并分析，形成对QST现状的整体视角，并识别出工具碎片化、标准缺失、工作者技能缺口等系统性障碍；同时提出未来研究的集成测试生态与标准化路径。

**🔧 技术方法**

采用的技术主要包括：多元文献检索与筛选、LLM辅助信息抽取、定性与主题分析；技术方法涵盖经典适配技术、量子特定技术、统计方法、调试与验证技术、仿真技术等五大类测试技术；并对多种工具（如MorphQ、QuanFuzz、Muskit等）进行分类。

**📊 数据集**

数据集为350条URL（57篇同行评审论文、53条灰色文献），其中110条符合研究问题，覆盖2024-2026年间的学术论文与实践博客、公司网页、教程、论坛等多种来源。

**📈 对比分析**

比较方法是将检索到的文献按技术类别、挑战与工具进行编码后，进行主题归纳与交叉对照，未采用实验性性能对比；结果表明QST技术多样但生态碎片化，实践与学术关注点高度重叠但仍缺乏统一标准与成熟工具。

**⚠️ 局限性**

局限性包括：灰色文献检索受Google搜索排名变化影响，文献覆盖非完整；LLM抽取可能存在信息遗漏或误判，虽人工校对；未对不同技术或工具进行实证性能评估，结果偏向概念性与描述性；未来研究需进一步细化比较与实测。

---

## 56. AM-Bench: A Modular Simulation Suite and Benchmark for Aerial Manipulation Policy Learning

**arXiv ID:** 2609.00641 | [PDF](https://arxiv.org/pdf/2609.00641v1)

**作者:** Yutong Wang `[一作]` (Carnegie Mellon University), Guanya Shi `[通讯]` (Carnegie Mellon University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AM-Bench，一个模块化的多旋翼空中操控学习基准，集成任务环境、机器人形态、控制器、扰动模型与高层视觉-语言-动作接口，并在 12 个真实任务上进行系统级评估。

**💡 创新点**

创新点在于：①将空中操控系统拆解为可插拔的五大模块；②在统一高层接口下比较多种学习策略与控制器；③结合地面/墙面效应、风扰动与执行器饱和的物理模型；④通过实验揭示策略-控制接口和机器人形态对性能的交互影响。

**🔧 技术方法**

使用的技术包括：物理仿真（基于 Unity/PhysX 等），PID/MPC/ L1 下沉控制；高层策略包括 imitation learning（ACT、DP）与 vision‑language‑action（π₀、π₀.₅）；地面/墙面效应模型、风扰动模型；四种多旋翼平台（UA‑Quad、UA‑Hexa、FA‑Hexa、Omni‑Hexa）与 EE‑only 端执行器。

**📊 数据集**

使用的数据集为 12 项任务的 80 条成功演示（单任务），合并成多任务集用于 IL 与 VLA 训练；同时收集真实硬件演示以做仿真‑实测对比。

**📈 对比分析**

方法：对 12 项任务分别评估宏平均任务成功率与子任务完成率；VLA 零射击性能低，fine‑tune 后可与 IL 相比；EE‑target + MPC 接口表现最好，任务成功率最高；不同形态平台（UA‑Hexa vs FA‑Hexa vs Omni‑Hexa）在倾斜、饱和和跟踪误差上差异显著。

**⚠️ 局限性**

limitations: 仅覆盖单机硬臂多旋翼空中操控，未涵盖协作、柔性或软抓手系统；未实现强化学习实验；仿真模型缺乏完整电机/ESC 动力学及更广泛真实扰动，仿真‑实测验证仅在部分任务上完成。

---

## 57. FLaG: Frequency-Domain Latent-attention Gated Pooling for Token Aggregation

**arXiv ID:** 2609.00831 | [PDF](https://arxiv.org/pdf/2609.00831v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 58. Aligned but Flattened: Analyzing the Trade-off between Cultural Alignment and Diversity in LLMs

**arXiv ID:** 2609.00565 | [PDF](https://arxiv.org/pdf/2609.00565v1)

**作者:** Jingshen Zhang `[一作]` (Tianjin University), Wenxuan Zhang `[通讯]` (Singapore University of Technology and Design)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对六种主流LLM进行文化微调，并通过统一框架同时评估文化一致性与多样性，发现二者存在系统性权衡。

**💡 创新点**

首次将文化一致性与多样性视为互补维度，并用低秩简化偏差解释文化压平现象。

**🔧 技术方法**

提出基于Soft Accuracy的双向度评估、FFN激活空间探测以及低秩稀疏分析等技术。

**📊 数据集**

使用世界价值观调查（WVS‑7）和对应的问卷样本进行训练与评测。

**📈 对比分析**

与基线模型对比，微调后一致性显著提升，但多样性大幅下降，且激活空间显著压缩。

**⚠️ 局限性**

仅评估SFT方法，评测不包含开放式或动态情境，且未提出具体缓解文化压平的策略。

---

## 59. Less Is More: Balancing Positive and Negative Space in Visual Concept Blending

**arXiv ID:** 2609.00476 | [PDF](https://arxiv.org/pdf/2609.00476v1)

**作者:** Shishi Xiao `[一作]` (Brown University), David H. Laidlaw `[通讯]` (Brown University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种自动化的视觉概念混合管线，显式利用正负空间对图形进行空间组合，并在共享轮廓上融合两种概念。

**💡 创新点**

创新点在于（1）将空间布局（正负空间、轮廓共享）作为一等设计控制；（2）使用多模态代理进行区域规划与评估；（3）融合像素级扩散填充与矢量级优化的混合管线，实现可编辑、结构连贯的混合图形。

**🔧 技术方法**

技术主要包括：视觉-语言模型与几何约束的区域检测；多模态代理（LLM+VLM）进行规划与评估；基于扩散的像素级填充；DiffVG + Score Distillation Sampling 的矢量级轮廓优化；Chamfer 与弯曲损失用于保持曲线平滑。

**📊 数据集**

数据集方面使用公开文本-图像对（如COCO/LAION）来训练扩散模型，生成主概念轮廓；为验证评估，人工收集了20对概念混合示例，并使用人类设计师和非专业用户进行用户研究。

**📈 对比分析**

与现有基准（PixelGPT-5.3、Claude、NanoBanana等）在三项指标（CLIP-T可识别度、CLIP-IQA感知质量、Alignment一致性）对比，本文方法在Alignment上最高（0.902）且在可识别度与感知质量上与顶尖方法相近；在人类实验中表现出更高的识别度、创造性和表达性。

**⚠️ 局限性**

局限性包括：未对专业设计师的工作流程进行评估；缺乏专门针对“负空间”概念的训练数据与模型；在复杂场景下的实时交互性能尚未充分测试；以及对概念对的选择和意图映射仍需用户手动设定。

---

## 60. Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding

**arXiv ID:** 2609.00097 | [PDF](https://arxiv.org/pdf/2609.00097v1)

**作者:** Zhigeng Liu `[一作]` (Fudan University), Xipeng Qiu `[通讯]` (Fudan University)

**通讯引用:** 18958 | [OpenAlex ID](https://openalex.org/A5044665993)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 FFD（Faster Flash Decoding），一种在 LLM 解码阶段实现高效稀疏注意力的框架，显著提升长上下文生成速度。

**💡 创新点**

创新点包括 2‑bit 内容感知量化扫描、基于相对阈值 δ 的 Top‑δ 选择策略、融合 Selector‑Computer 核心并采用伪最大值近似，打破传统固定预算与全局同步的瓶颈。

**🔧 技术方法**

采用低位量化（2‑bit 关键字 + 8‑bit 残差）、Top‑δ 动态阈值、CUDA Graph 动态捕获、Triton 融合核以及硬件算法协同设计，实现高吞吐低延迟。

**📊 数据集**

在 Llama‑3.1‑8B、Qwen‑2.5‑7B 等模型上，用 LongBench 与 RULER 进行长上下文性能评估，结合标准 KV‑Cache 长上下文数据集。

**📈 对比分析**

与 FlashAttention‑2、Quest、KIVI、Twilight 等基线相比，FFD 在 RTX‑4090 上 kernel 级别可达 11.6× 加速、整体吞吐提升 2.37×，在 RULER 上平均得分约 87‑89%（与稠密基线相当），同时保持低 LSE 误差和高召回。

**⚠️ 局限性**

局限性：仅在 GQA 结构模型验证；对非 GQA 或 MLA‑style 注意力未验证；伪最大近似依赖“sink + local”假设，极端分布下可能略显保守；缺乏对多头差异化稀疏性的系统性评估。

---

## 61. Behaviorally Grounded User Profiles from the Wild for Personalized Alignment and Multi-Perspective Reasoning

**arXiv ID:** 2609.00014 | [PDF](https://arxiv.org/pdf/2609.00014v1)

**作者:** Yuxuan Li `[一作]` (University of Waterloo), Ehsan Kamalloo `[通讯]` (ServiceNow AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种从真实社交媒体行为中自动抽取开放式用户画像，并将其用于LLM的个性化训练与测试时多视角推理。

**💡 创新点**

创新点在于用真实行为数据替代传统合成角色，生成更高保真、无刻板印象、长尾多样化的用户画像，从而提升个性化效果。

**🔧 技术方法**

核心技术包括：1) 用LLM抽取帖子中的候选属性并去重；2) 将清洗后的描述总结为简洁可读的bio；3) 用合成画像驱动的SFT数据生成；4) 在测试时通过采样多份画像生成多视角答案并用聚合模型整合；5) 使用LLM作为评判者进行评估。

**📊 数据集**

使用的数据集：Bluesky 200万条社交媒体帖子；RecBench、URS Bench、PRISM、Lastfm等评测基准；SFT数据合成与评估均采用GPT-OSS-120B。

**📈 对比分析**

与无画像提示和任务相关的合成画像基线相比，行为画像在SFT（RecBench F1提升0.2–0.3、URS评分提升0.8–1.0）以及多视角推理（UR评估提升0.5–0.6）均表现显著优势，且MAUVE指标显示生成文本更贴近真实下游数据。

**⚠️ 局限性**

局限性包括：在需要强分类格式或极其稀缺领域（如PRISM、Lastfm）时表现下降；可能出现过度个性化、语言漂移、信息泄露或安全性过度保守的错误；聚合过程中仍需改进以防止细节泄露和安全违规。

---

## 62. SoK: When Safe Agents Fail Together: The Security of Multi Agent LLM Systems

**arXiv ID:** 2609.00595 | [PDF](https://arxiv.org/pdf/2609.00595v1)

**作者:** Rui Yang `[一作]` (Johns Hopkins University), Yinzhi Cao `[通讯]` (Johns Hopkins University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多智能体LLM系统（MAS）安全性进行系统化综述，提出基于执行路径的 A→I→R 框架，归纳了 8 个配置维度、7 种系统级风险、8 条攻击路径以及 5 维防御契约，并对 44 篇评估/基准工作进行审核。

**💡 创新点**

创新点在于：①首次以执行为单位统一描述攻击与防御，②构建 A→I→R 结构把威胁模型、交互接口、风险后果整合为一条完整路径，③定义 5 维防御契约（路径目标、观察、干预、信任边界、恢复）以评估防御完整性，④系统性梳理 197 篇文献，识别评估、指标、基准可配置性与开放系统评估等空缺。

**🔧 技术方法**

主要技术为文献检索、编码与分类、框架构建、路径映射与评估审核；未涉及深度学习模型或实验实现，而是以理论与方法学手段进行系统整理。

**📊 数据集**

使用公开的 197 篇相关论文作为语料库，其中 44 篇为安全评估/基准工作，涵盖多种配置、交互接口和攻击路径；未使用传统 NLP 或机器学习数据集。

**📈 对比分析**

比较方法：对 197 篇论文按配置维度 C1–C8、交互接口 I1–I6、攻击路径 P1–P8、风险 R1–R7 进行多维度映射，并用 5 维防御契约评估各防御措施的覆盖与缺口；性能评估未涉及具体数值，而是识别了评估指标的不足与比较深度的缺失。

**⚠️ 局限性**

limitations: ①缺少对多智能体交互导致安全影响的因果追踪与可解释性，②评估指标如 ASR 统一度不高，缺少可复现的度量规范，③基准缺乏可配置与可扩展接口，④多大多数评估基准集中于闭环系统，未覆盖开放/自生成成员环境，⑤防御路径闭合与恢复机制未得到充分验证。

---

## 63. Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning

**arXiv ID:** 2609.00605 | [PDF](https://arxiv.org/pdf/2609.00605v1)

**作者:** Miso Kim `[一作]` (Dongguk University), Woojin Lee `[通讯]` (Dongguk University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究忘记集失配导致大型语言模型遗忘失败，并提出CONFS框架，在不访问原始数据的情况下通过“自诉”与递归自诉提取模型记忆，构造与模型已学习内容一致的忘记集，实现可控遗忘并保留模型实用性。

**💡 创新点**

①明确了两种忘记集失配失效模式（Under Unlearning 与 Out-of-Knowledge Unlearning）。②通过梯度级别分析解释失配机制。③提出CONFS：利用模型自诉、递归自诉、三元组化与幻觉验证，生成结构化、精确且高质量的忘记集，完全数据盲。

**🔧 技术方法**

梯度上升 (GA)、梯度差异 (GD)、负偏好优化 (NPO)、拒绝调优 (RT) 等四种遗忘目标；SRO三元组抽取与子三元组分解；递归自诉查询；幻觉验证（SelfCheckGPT + DeBERTa NLI）；使用 GPT‑4o（或可替换的指令模型）进行结构化与能力问题生成。

**📊 数据集**

TOFU（合成作者资料）、CLEAR（多模态扩展）和 RWKU（真实公众人物）三个基准数据集。

**📈 对比分析**

与 Gold‑forget 集、FreeRecall‑QA、RWKU‑style 等数据盲构造进行对比；在 TOFU、CLEAR、RWKU 上评估四种遗忘目标，使用 Token Probability、ROUGE‑L 等指标。CONFS 在忘记性能与 Gold 相近的同时，显著降低了对 Retain、Real Authors、World Facts 等维度的实用性损伤，优于其他数据盲基线。

**⚠️ 局限性**

目前仅针对可结构化为 Subject‑Relation‑Object 的实体事实；对长叙事、程序性知识或上下文记忆等更广泛的记忆形式尚未覆盖，需进一步扩展。

---

## 64. FoldingAgent: Inferring Parametric Origami Procedures from Demonstration Videos

**arXiv ID:** 2609.00377 | [PDF](https://arxiv.org/pdf/2609.00377v1)

**作者:** Maya Moriya `[一作]` (Weizmann Institute of Science), Tali Dekel `[通讯]` (Weizmann Institute of Science)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从原纸折叠教学视频中逐帧推断出完整的可执行参数化折叠程序，形成可渲染、可编辑的折叠模型。

**💡 创新点**

提出了基于预训练视觉语言模型的代理框架，配备专用工具库（仿真、可视化、验证、回滚），实现了对多步折叠过程的自我纠错和重规划，显著减少错误累积。

**🔧 技术方法**

使用了预训练 VLM（Gemini 3.1 Pro Preview）进行视觉推理，搭配仿真器执行折叠动作，视觉批评器对比生成与目标帧，工具调用实现动作验证与回滚，整体形成循环的 agentic 系统。

**📊 数据集**

构建了 PurelandFold 数据集，包含 27 条多样化的 Pureland 折叠序列，配有关键帧、真实折叠动作与几何状态标签。

**📈 对比分析**

通过与 VLM-CP、VLM-S、VLM-S+C 等消融版本比较，使用 CV、TSS、GS、CS、FFS、CPD 等多维指标评估。实验表明，完整代理框架在几乎所有指标上均优于基线，尤其在编译有效性和几何相似度上取得显著提升；人类评估也显示该方法在图像相似度上被首选率超过 80%。

**⚠️ 局限性**

主要局限包括：高度依赖 VLM 的视觉推理能力，遮挡（手部或层叠）导致的错误识别；无法处理复合并行动作（如旋转+翻转）；目前仅适用于 Pureland 纯折叠，需扩展动作空间才能覆盖更高级折叠技术。

---

## 65. OreProof: Verifiable Provenance with Limited Disclosure for Critical-Minerals Supply Chains Using Zero-Knowledge Proofs

**arXiv ID:** 2609.00340 | [PDF](https://arxiv.org/pdf/2609.00340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 66. Subliminal Learning as Trait-Direction Drift: A Mechanism and Targeted Control under SFT Distillation

**arXiv ID:** 2609.01091 | [PDF](https://arxiv.org/pdf/2609.01091v1)

**作者:** Zhixuan Liu `[一作]` (Shanghai Jiao Tong University), Chao Yang `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

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

## 67. Generative artificial intelligence for reliable mechanistic reasoning for corrosion

**arXiv ID:** 2609.00099 | [PDF](https://arxiv.org/pdf/2609.00099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 68. Same Semantics, Different Outcome: On the Modality Robustness of Multimodal LLMs under Knowledge Conflict

**arXiv ID:** 2609.00550 | [PDF](https://arxiv.org/pdf/2609.00550v1)

**作者:** Jungyeon Lee `[一作]` (Hanyang University), Taeuk Kim `[通讯]` (Hanyang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究多模态大型语言模型在知识冲突场景下对文本与图像证据的鲁棒性，并发现模型在不同模态之间的偏好极不一致。

**💡 创新点**

创新点在于首次系统评估多模态模型在文本与图像冲突中的可靠性，挑战传统文本优先的假设，并揭示模态偏好受输入顺序、数据集与预处理方式的强烈影响。

**🔧 技术方法**

使用对比实验、归一化图像-文本差距指标、Prompting、Supervised Fine‑Tuning (LoRA)、Direct Preference Optimization (DPO) 以及视觉预处理调节等技术进行分析与缓解。

**📊 数据集**

主要采用 ConflictQA 与 NQ‑Swap 两个知识冲突数据集进行实验，并在 MS MARCO、MM‑SafetyBench 等下游任务中评估模型表现。

**📈 对比分析**

通过多模态输入顺序、模型家族与预处理方式的交叉对比，发现图像偏好可正向或负向变化；SFT 在某些设置下可将差距减至中等水平，但整体仍未完全解决不一致性。

**⚠️ 局限性**

局限在于仅考察文本与其图像渲染形式，未涵盖自然视觉内容、音频等多模态；对机制的解释仍不充分，且实验覆盖的模型与数据集有限。

---

## 69. AgentFactory: Towards Automated Agentic System Design and Optimization

**arXiv ID:** 2609.01045 | [PDF](https://arxiv.org/pdf/2609.01045v1)

**作者:** Enci Zhang `[一作]` (Peking University Shenzhen Graduate School), Guibo Luo `[通讯]` (Peking University Shenzhen Graduate School)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AgentFactory 框架，实现对大型语言模型（LLM）与工作流结构的联合自动化设计与优化。

**💡 创新点**

创新点在于：①将模型微调与工作流设计纳入同一多目标优化空间；②使用 LLM 作为全流程优化器；③三阶段优化管线（规划、微调、工作流）实现高效搜索。

**🔧 技术方法**

技术包括：LLM 优化器、三阶段优化管线、模型微调方法（LoRA、QLoRA、全参数微调）、代码型工作流表示、分层提示模板、基于多目标的标量化与 Pareto 选择。

**📊 数据集**

使用公开基准数据集：MMLU、DROP、HumanEval、MBPP、GSM8K、MATH、MedQA、FinEval；微调数据来源于对应任务的训练集、CodeBagel、MathInstruct、IndustryInstruction 等。

**📈 对比分析**

通过与手工设计方法（IO、CoT、Self‑Consistency、Reflexion、Debate）和自动化工作流优化基线（ADAS、AFlow）对比；在 8 个基准上平均提升 9.1%，MedQA 提升 19.6%，FinEval 提升 18.7%，且在 GPT‑4o‑mini 基础上推理成本降低 68%。

**⚠️ 局限性**

局限性包括：①对 LLM 优化器的依赖导致计算成本高；②搜索空间仍大，需多轮迭代；③微调数据质量与覆盖面决定最终性能；④目前仅验证在 LLM 任务，缺乏对非 LLM 场景的评估。

---

## 70. When Features Become Instances: Inverted Contrastive Learning for Unsupervised Feature Selection

**arXiv ID:** 2609.00782 | [PDF](https://arxiv.org/pdf/2609.00782v1)

**作者:** Utsab Ghosh `[一作]` (ABV-Indian Institute of Information Technology and Management), Roshni Chakraborty `[通讯]` (ABV-Indian Institute of Information Technology and Management)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于特征反转的对比学习框架ICLFS，用于无监督特征选择。

**💡 创新点**

通过将样本矩阵转置为特征向量，构造多种掩码正样本和打乱负样本，利用InfoNCE对特征进行对比学习，并用投影空间嵌入范数作为特征重要性度量，同时加入去相关正则和拉普拉斯门控排名修正。

**🔧 技术方法**

自注意力编码器、残差投影器、InfoNCE对比损失、特征间去相关正则、拉普拉斯分数门控修正。

**📊 数据集**

12个公开基准数据集，包括图像、生物医学、文本和质谱数据。

**📈 对比分析**

与经典图谱法、谱法、非负判别法以及基于重建的CAE/LS-CAE等基线进行聚类准确率对比，ICLFS在10/12数据集上取得最高聚类准确率。

**⚠️ 局限性**

对小样本集的表现仍略逊于LS-CAE，且依赖于多视角掩码设计，对异常值和高维稀疏数据的鲁棒性尚待验证。

---

## 71. EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities

**arXiv ID:** 2609.00487 | [PDF](https://arxiv.org/pdf/2609.00487v1)

**作者:** Feitong Qiao `[一作]` (Reinforce Labs), Anish Das Sarma `[通讯]` (Reinforce Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种将多轮红队攻击策略视为可进化对象，并通过质量多样性搜索构建结构化档案的系统

**💡 创新点**

将多轮攻击策略转化为可进化的分阶段对话计划，并结合NSLC和MAP‑Elites的混合存档架构，形成可解释的多风险类别攻击地图；同时引入代际记忆共享以跨代学习目标模型特征

**🔧 技术方法**

演化搜索（AlphaEvolve风格的LLM变异与交叉）、质量多样性算法（MAP‑Elites + Novelty Search with Local Competition）、LLM驱动的渲染与判定、Pareto双目标（攻击成功率与峰值严重度）

**📊 数据集**

HarmBench（159行为的测试集）以及四个目标模型（GPT‑4o、Claude Sonnet 4.6、Qwen3‑32B、GPT‑5.4）

**📈 对比分析**

与六个基线（单轮与多轮红队方法）对比，平均攻击成功率（ASR）为72.1%，在所有四个目标模型上均优于X‑Teaming（44.0%）且显著高于其他基线；生成的档案还能直观展示各风险类别的失败分布

**⚠️ 局限性**

对Claude Sonnet 4.6的化学‑生物类别几乎无攻击成功；评估依赖单一GPT‑4o判定器，判定一致性未知；实验受预算限制，可能未达到完整覆盖

---

## 72. ViTAMINS: An Empirical Study of Training Self-Supervised Vision Transformers with Synthetic Hard Negatives

**arXiv ID:** 2609.01041 | [PDF](https://arxiv.org/pdf/2609.01041v1)

**作者:** Nikos Giakoumoglou `[一作]` (Imperial College London), Tania Stathaki `[通讯]` (Imperial College London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将合成硬负样本融入视觉Transformer无监督对比学习的预训练方法，显著提升表示质量并产生语义分割特性。

**💡 创新点**

创新点在于：① 设计并统一六种合成负样本生成策略（插值、外推、Mixup、噪声注入、梯度扰动、对抗扰动）；② 通过对比实验展示这些合成负样本在对比学习中与传统方法相同或更优，且能自然产生优秀的语义分割效果。

**🔧 技术方法**

技术手段包括：InfoNCE 对比学习框架、EMA 目标网络、4096 大小的内存队列、六种负样本生成策略、Drop Path 正则化、BYOL/ MoBY 等现有方法的基础上改造。

**📊 数据集**

使用的数据集：ImageNet ILSVRC-2012（无标签预训练），以及下游任务数据集 Oxford/Paris 检索、Copydays 复制检测、DAVIS-2017 视频分割、COCO 检测/分割、ADE20K 语义分割、CIFAR-10/100、Flowers-102、Cars 等。

**📈 对比分析**

与 MoBY、BYOL、DINO、I-JEPA、V-JEPA、iBOT 等方法比较，实验证明在 ImageNet 线性评估、k‑NN、检索、复制检测、视频分割、COCO、ADE20K 等多项任务中，本文方法均显著优于对手；例如 ViT‑B/16 在 ImageNet 线性 Top‑1 取得 77.1%，ViT‑S/16 73.1%，比传统对比学习提升约 1–2%。

**⚠️ 局限性**

局限性：① 需要大规模内存队列和多种超参数调优；② 合成负样本的效果在不同模型或任务上可能不完全一致；③ 目前未针对视频/时序数据进行专门训练，适应性仍需进一步验证。

---

## 73. Superposed Latent Autoencoder

**arXiv ID:** 2609.01158 | [PDF](https://arxiv.org/pdf/2609.01158v1)

**作者:** Quanling Zhao `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出Superposed Latent Autoencoder（SLAE），通过将宽latent在共享内存中超叠来实现压缩

**💡 创新点**

创新点在于用随机正交绑定与学习的恢复网络，将宽latent共享存储而非单独压缩，形成结构化干扰抑制的容量‑干扰折衷

**🔧 技术方法**

采用Walsh‑Hadamard+随机置换/符号正交绑定、学习的存储适配器和恢复网络以及卷积autoencoder架构

**📊 数据集**

使用CIFAR‑10/100、SVHN、STL‑10、Tiny ImageNet四个图像数据集进行实验

**📈 对比分析**

与普通autoencoder在相同平均存储预算下比较，SLAE在所有数据集上平均MSE降低35%–56%，并使下游分类精度提升最高可达16.79个百分点

**⚠️ 局限性**

局限性包括超叠因子增大导致恢复难度上升、干扰溢出问题，以及对模型深度与不同任务的适用性需进一步探索

---

## 74. SOVER: Formal Certification of Optimization Reformulations via LLM-Assisted SMT Verification

**arXiv ID:** 2609.00728 | [PDF](https://arxiv.org/pdf/2609.00728v1)

**作者:** Swapnil Bhattacharyya `[一作]` (TCS Research), Mayank Baranwal `[通讯]` (TCS Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于LLM提取映射、SMT验证的优化重构形式等价性检查框架

**💡 创新点**

创新点在于把语义映射与形式化验证分离，利用域交叉可行性和目标顺序保持进行等价性证明，并对非线性采用δ-可满足性与ε-argmin 检查

**🔧 技术方法**

使用大型语言模型进行变量/参数映射，Z3 SMT 求解器进行可行性与顺序检查，δ-满足性（δ-SAT）用于非线性场景

**📊 数据集**

数据集包含EquivaFormulation（2178个MILP对）和新发布的NLEquiv-150（150个非线性对）

**📈 对比分析**

与多种基线（Prompt、Gemini-CoT、WLT、Canonical、SOVER）对比，平均验证时间仅0.03s，整体准确率高达99.7%，在所有子类型上表现优于对手

**⚠️ 局限性**

主要限制是对LLM映射质量高度依赖、SMT求解器在规模大模型时可扩展性有限，以及固定深度展开导致对复杂嵌套约束的识别不完备

---

## 75. Inverse Rendering for Modeling with Line Primitives

**arXiv ID:** 2609.00625 | [PDF](https://arxiv.org/pdf/2609.00625v1)

**作者:** Kenji Tojo `[一作]` (ETH Zürich), Bernd Bickel `[通讯]` (ETH Zürich)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

通过逆渲染技术从多视角图像重建模糊、纤维状几何体，并以线段原语形式表示。

**💡 创新点**

提出基于随机可微光栅化的线段渲染器，结合Bresenham子像素绘制、MSAA与高斯反锯齿，以及动态连线（re‑lining）等机制，实现稠密模糊结构的显式重建。

**🔧 技术方法**

使用DiffSoup的随机可微光栅化、Bresenham算法、子像素MSAA、高斯重建滤波、动态连线优化以及PyTorch/Vulkan实现。

**📊 数据集**

在合成的Shelly数据集和自制的真实世界“Fuzzy”数据集上进行实验。

**📈 对比分析**

与DiffSoup、VolSurfs、3DGS等方法对比，在PSNR、SSIM、LPIPS等指标上相较于表面方法更优，接近甚至优于体积方法；渲染速度与3DGS相当甚至更快。

**⚠️ 局限性**

难以完全恢复复杂拓扑和深层纤维内部细节；对大面积平坦区域不如三角面更有效；需要更强先验或学习连接概率；线宽估计不精确。

---

## 76. From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling

**arXiv ID:** 2609.00051 | [PDF](https://arxiv.org/pdf/2609.00051v1)

**作者:** Kuan-Lin Chu `[一作]` (University of California San Diego), Tsui-Wei Weng `[通讯]` (University of California San Diego)

**通讯引用:** 1465 | [OpenAlex ID](https://openalex.org/A5114139431)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并验证了大型语言模型内部的多阶段安全电路，包括识别有害输入的检测头、调节安全信号的安全神经元以及生成拒绝回应的拒绝头，并通过对这些组件进行权重缩放，显著提升模型在对抗性提示下的安全性能。

**💡 创新点**

①首次揭示并因果验证跨层的检测-安全-拒绝三阶段电路结构；②提出基于电路识别的无训练、架构保留的权重缩放方法（Circuit-Based Safety Editing, CBSE）；③展示该方法在六种主流 LLM 上普适且可迁移。

**🔧 技术方法**

机制解释（head/ neuron 级定位）、目标干预（抑制、缩放）、激活补丁、权重缩放、自动化安全判别器 Llama-Guard、标准评测基准。

**📊 数据集**

对抗性提示集（AdvBench、GCG、ADV-LLM）、安全判别（Llama-Guard）、标准自然语言理解基准（MMLU、HellaSwag、PIQA、WinoGrande、WikiText）、边界样本集（OR‑Bench‑80k）以及用于评估攻击生成的 GCG suffix。

**📈 对比分析**

与原始模型相比，单独或组合对检测头、拒绝头、安全神经元的缩放在 GCG 攻击下安全率提升约26.5%（从43.2%到69.7%），平均准确率下降仅1.7%；在所有六种架构上均保持了较高的任务性能，证明了方法的有效性与稳定性。

**⚠️ 局限性**

仅基于单次模型快照进行电路识别，未考虑模型在后续大规模微调或持续训练过程中电路可能的重组或漂移；实验规模仅覆盖六种模型，未知在更大规模或不同训练策略下的泛化能力。

---

## 77. Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs

**arXiv ID:** 2609.01117 | [PDF](https://arxiv.org/pdf/2609.01117v1)

**作者:** Zhaoliang Chen `[一作]` (Emory University), Jie Fu `[通讯]` (IQuest Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结的LLM（如Qwen3‑8B）上提出一种连续空间推理框架 LRT，将问题映射为潜在向量并递归改进，再以 soft‑token 形式注入解码器，完成推理。

**💡 创新点**

创新点在于：① 用任务专用的双向 Transformer 提议器产生基准潜在向量；② 用 TRM‑style 的小型循环推理器在低维空间执行多步残差更新；③ 两者配合在冻结解码器上完成推理，避免了传统 CoT 的离散错误累积与对推理轨迹的依赖。

**🔧 技术方法**

核心技术包括：冻结解码器、任务专用提议器、TRM 递归推理器、残差更新、停梯度训练、两阶段分离训练、soft‑token 注入接口。

**📊 数据集**

使用的数据集：符号推理任务（Countdown‑4、Sudoku）和自然语言推理任务（HumanEval、MBPP、StrategyQA）。

**📈 对比分析**

在相同冻结解码器、提示、数据与计算预算下，与 Zero‑shot CoT、SoftCoT、EBM‑CoT、思考模式以及从零开始的符号求解器对比，LRT 在 Countdown‑4 达到 56.7%/Sudoku 49.2%，HumanEval 37.8%/MBPP 51.5%/StrategyQA 75.1%，平均 54.1%，显著优于对照组且推理成本低（≈1–2 TFLOP/例）。

**⚠️ 局限性**

局限性：① 需要为每个任务训练专用的提议器与推理器，缺乏跨任务通用性；② 仅在给定任务内泛化，未测试难度级别迁移；③ 仅验证 8B 规模解码器，未知更大规模下表现；④ 在符号任务上仍落后专业求解器；⑤ 推理深度导致额外计算；⑥ 对计算分布的分析仅相关性，未给出定量拆解。

---

## 78. JENGA: Exploiting Counter-Based RowHammer Countermeasures to Break Real-Time Predictability

**arXiv ID:** 2609.01077 | [PDF](https://arxiv.org/pdf/2609.01077v1)

**作者:** Valentin Abgrall `[一作]` (Univ Rennes), Angeliki Kritikakou `[通讯]` (Univ Rennes)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6215c339-3735-4be3-8a07-5bbb7004712d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并评估基于硬件计数器的 RowHammer 防护（PRAC‑N）对实时系统 worst‑case execution time (WCET) 的时序影响，提出攻击模型并给出安全 WCET 上界的理论与实验验证。

**💡 创新点**

首次揭示计数器型 RowHammer 防护会导致 200% 以上 WCET 增加，并提出利用 PRAC 计数器状态的攻击方法及通用的安全 WCET 计算框架。

**🔧 技术方法**

采用 gem5+Ramulator 2.0 进行周期级 DRAM 仿真，使用 TACLeBench 基准与自定义攻击任务，对 PRAC‑N 计数器状态进行理论分析，并结合概率 WCET 估算。

**📊 数据集**

使用 TACLeBench（kernel 与 sequential）十个典型工作负载以及自定义的攻击任务作为实验数据集。

**📈 对比分析**

通过对比攻击前后执行时间与理论安全 WCET 上界，发现攻击可使执行时间最高提升至约 249%，而理论上界始终保持在观测值之上。

**⚠️ 局限性**

仅在单核无缓存裸机模型下评估，未考虑多核、操作系统、缓存层级、DRAM 子阵列细节及实际硬件实现差异，攻击实现假设可知计数器内部状态。

---

## 79. VIBE-Bench: Evaluating Personalized Large Language Models When Profiles Don't Mean Preferences

**arXiv ID:** 2609.00921 | [PDF](https://arxiv.org/pdf/2609.00921v1)

**作者:** Yiwen Jiang `[一作]` (Monash University), Zongyuan Ge `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了Vibe-Bench基准，专门用于评估跨概念个性化（Profile‑Preference Conceptual Misalignment, PRCM）中的偏好推理能力。

**💡 创新点**

创新点包括：①构建了新的跨概念偏好推理范式并给出系统分类；②设计了两项心理学驱动的任务（情绪调节生成与职业兴趣分类）以显式检验PRCM；③提供了可解释的概念映射与“链式思考”提示框架，揭示了跨概念映射是主要瓶颈。

**🔧 技术方法**

技术方法包括：使用多种大型语言模型（Gemma、Llama、Mistral、Qwen等）做非参数提示、检索增强与参数微调；构建基于Big‑Five与RIASEC的概念映射；采用链式思考提示、概念感知推理以及实验性自动映射发现。

**📊 数据集**

数据集：Vibe‑Bench，包含3,504个人画像、12,239段对话（共130K语句），其中情绪调节任务基于ESConv、ExTES数据，职业兴趣任务基于O*NET与RIASEC兴趣问卷，所有对话均为模型生成并人工校正。

**📈 对比分析**

与基线比较：非参数方法（Full‑History Prompting、Profile‑Augmented Prompting、检索增强）在策略识别和生成质量上提升有限；参数微调（P‑SFT）在职业分类任务上表现最佳（F1≈83%），但在情绪调节生成的策略准确度仍低（≤24%）。整体来看，现有方法仍严重依赖语义匹配，难以完成跨概念映射。

**⚠️ 局限性**

局限性：①数据为合成而非真实用户交互，生态有效性待验证；②仅涵盖两项任务，缺乏多领域、多语言扩展；③概念映射基于群体统计，可能导致刻板印象；④未评估强化学习等偏好优化方法；⑤自动映射发现尚属初步，需进一步研究。

---

## 80. Right Frame, Wrong Rule: Cultural Cues Expose the Financial Knowledge Gap They Were Meant to Close

**arXiv ID:** 2609.00999 | [PDF](https://arxiv.org/pdf/2609.00999v1)

**作者:** Rania Elbadry `[一作]` (MBZUAI), Zhuohan Xie `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对伊斯兰金融中的规范多元性进行评估，构建四选项基准并揭示“刻板印象陷阱”，即文化信号使模型选择伊斯兰框架但往往给出错误答案。

**💡 创新点**

创新点在于设计四元分类法，将框架选择与内在正确性分离，提出规范多元性评估框架，并系统地揭示文化偏好与模型能力的分离。

**🔧 技术方法**

采用多模型对比、文化信号注入、激活补丁与 logit‑lens 机制分析等技术，对语言模型在不同框架下的决策过程进行细粒度解析。

**📊 数据集**

使用 SAHM 专业验证的伊斯兰金融问答语料，覆盖 7 类产品共 304 题，辅以 64 题西方基准和 41 题伊斯兰专属基准，实现双语（英文、阿拉伯文）的评测。

**📈 对比分析**

通过在 12 款模型（前沿、巨大、中型、阿拉伯中心）上对 50 条文化信号进行注入，测算框架激活率、错误率等指标，结果显示非前沿模型在强信号下陷入刻板印象陷阱的比例高达 57–66%，前沿模型虽更稳健但仍存在 7.5% 的错误。

**⚠️ 局限性**

局限性包括仅聚焦伊斯兰金融双语场景、使用四选项格式（无法评估开放式回答）、未覆盖其他多框架领域、且评测基于单一模型快照，缺乏动态更新和真实用户交互验证。

---

## 81. One Policy, Any Budget: Internalizing Budget-Aware Search via Reinforcement Learning

**arXiv ID:** 2609.00813 | [PDF](https://arxiv.org/pdf/2609.00813v1)

**作者:** Xiaowei Sun `[一作]` (Fudan University), Yanghua Xiao `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了AnySearch框架，使单一策略在任意预算下自动调度工具搜索，内化预算感知能力；

**💡 创新点**

采用渐进式训练脚手架和双阶段课程学习，配合自适应预算采样和带自适应效率权重的复合奖励，实现在不同预算下统一最佳性能；

**🔧 技术方法**

使用强化学习（GRPO）、预算状态注入、结构化思考提示、复合奖励（包含绝对与相对效率信号）、自适应预算采样与两阶段课程；

**📊 数据集**

训练集为NQ与HotpotQA，评估集为七个QA基准（NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle），采用三大后端模型（Qwen2.5-7B‑Instruct、Llama‑3.1‑8B‑Instruct、Qwen3‑4B）；

**📈 对比分析**

与BATS、Search‑o1、Search‑R1、ZeroSearch、StepSearch等基线比较，AnySearch在所有预算级别均优于基线，且能在未见预算上保持性能提升；在工具生产率、token消耗等方面也优于现有方法；

**⚠️ 局限性**

预算仅被视为离散搜索次数，未覆盖延迟、成本、系统负载等多维成本；性能受模型规模限制；使用静态2018 Wikipedia未验证对实时更新语料的适应性；当答案既不在模型参数也不在检索语料中时无法恢复。

---

## 82. A Dynamic Intermediate Representation for Hybrid Quantum-Classical Programs

**arXiv ID:** 2609.01037 | [PDF](https://arxiv.org/pdf/2609.01037v1)

**作者:** Alex Rice `[一作]` (University of Edinburgh), Tobias Grosser `[通讯]` (University of Cambridge)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态中间表示（IR），将量子门提升为第一类值，支持在运行时根据经典计算动态生成、组合和控制量子门；

**💡 创新点**

突破传统静态电路模型限制，引入动态门和 gadget 机制，使量子-经典边界变得可优化并统一表示，首次在同一 IR 中实现随机化编译、错误校正和 MBQC 的高效转换；

**🔧 技术方法**

基于 MLIR 框架构建 SSA 价值语义的量子方言，利用动态门、gadget 变换、随机化编译、错误校正、测量基量子计算等技术实现编译器优化；

**📊 数据集**

构建了包含 Teleport、Prep、RUS、QML、IPE、RWPE、MBQC-Rot、MBQC-CX、QEC-Adap、QAOA 等 10 个混合量子-经典基准程序的套件；

**📈 对比分析**

与 QIR 在行数、单词数、量子操作数、基本块数、环路复杂度和 Halstead 难度等六项指标进行对比，动态门 IR 在所有指标上均显著优于 QIR（分别低 47%、52%、24%、69%、52% 和 30%），证明了其更简洁高效的性能；

**⚠️ 局限性**

局限性在于主要针对 Clifford 门的优化，尚未充分扩展到非 Clifford 操作、子程序/子电路以及量子寄存器等高级特性，噪声通道的完整利用和更复杂的混合程序支持仍待进一步研究。

---

## 83. SkyShare: Constellation-wide Sky Sharing for LEO-Radio Astronomy Coexistence

**arXiv ID:** 2609.00821 | [PDF](https://arxiv.org/pdf/2609.00821v1)

**作者:** Farzad Mehri `[一作]` (NC State University), Vijay K. Shah `[通讯]` (NC State University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种全星座级别的天空共享系统，通过预测性调度LEO卫星束波，以在不降低网络覆盖的前提下保护射电天文观测；

**💡 创新点**

创新点在于：①引入EPFD预算化的感兴趣区域（RoI）划分，既限制优化范围又保证总干扰不超限；②将LEO‑RAS共存问题转化为可规模化的最小成本流（special case）或近似的可行性修复流程，避免了传统局部避让带来的覆盖损失；

**🔧 技术方法**

采用的技术包括：高精度轨道预测、ITU‑R 兼容的EPFD干扰模型、基于OPD数据的实时观测元数据获取，以及基于流网络的调度算法（SkySched）；

**📊 数据集**

使用了公开的Starlink Gen2‑mini TLE轨道数据、CelesTrak卫星观测轨迹、以及全球约25个Ku‑band单口射电望远镜的观测计划；

**📈 对比分析**

与传统的TBA与DTBA避让方法对比，SkySched在保持相同EPFD阈值下，平均减少90%以上的未服务地面单元，平均EPFD下降48%（单窗口）或41%（15 s窗口），并在多种天文站点、纬度、望远镜尺寸下实现了显著的覆盖恢复与灵敏度提升；

**⚠️ 局限性**

局限性包括：仅考虑单一观测场景（单个RASS观测），未处理多运营商或多站点同时观测的联合约束；对卫星位置与观测指向误差较敏感；模型未考虑多束服务与频率/极化分配的细节；

---

## 84. StainPresetNet: Stain Preset Network for Fast Multi-to-Multi Stain Normalization

**arXiv ID:** 2609.01146 | [PDF](https://arxiv.org/pdf/2609.01146v1)

**作者:** Hongtao Kang `[一作]` (Southern Medical University), Shenghua Cheng `[通讯]` (Southern Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 StainPresetNet 的快速多对多染色归一化网络，用预设参考图像指导颜色映射，保持结构不变并实现多域互相转换。

**💡 创新点**

创新点在于：①利用全 1×1 卷积子网络实现像素级颜色映射；②通过预设参考图像实现多域多方向归一化，无需重新训练；③采用低分辨率编码器+融合层控制映射参数，显著提升计算效率。

**🔧 技术方法**

技术包括：改进的 ResNet18 编码器、两层线性融合层、全 1×1 卷积映射子网络、五种损失（对抗、循环、域一致、结构一致、身份），以及 CycleGAN 变体的训练框架。

**📊 数据集**

使用三个数据集：①对齐细胞学图像（S1‑S3）做多域归一化评估；②多域细胞学分类数据集（D1‑D5）；③多中心组织学分类数据集（Camelyon16/17，Uni16、C1‑C5）。

**📈 对比分析**

与传统方法（Reinhard、Macenko、Vahadane）以及深度学习方法（StainGAN、StainNet、ParamNet）比较。结果显示 StainPresetNet 在图像相似度（SSIM‑T/PSNR‑T）、结构保持（SSIM‑S）、计算速度（≈1600 FPS）和分类准确率（细胞学平均 0.951，组织学平均 0.907）上均优于对比方法。

**⚠️ 局限性**

局限性包括：①仍需手工挑选预设参考图像，虽然对性能影响小，但可能在极端样本多样性下受限；②在极大尺度全切片图像的实时处理仍面临存储与内存瓶颈；③对极端颜色失真或噪声场景的鲁棒性未做系统评估。

---

## 85. Counterfactual Fragility Certificates: Exposing High-Confidence Brittleness under Structured Evidence Failure

**arXiv ID:** 2609.00366 | [PDF](https://arxiv.org/pdf/2609.00366v1)

**作者:** Filippo Cenacchi `[一作]` (Macquarie University), Runze Yang `[通讯]` (Macquarie University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并评估了 Counterfactual Fragility Certificates（CFC），一种面向表格模型的可重现审计对象，用于在结构化证据失效时量化预测的脆弱性轨迹。

**💡 创新点**

通过对特征组的递增剔除与分级退化，记录贪心翻转预算、归一化边际坍塌面积、降解阈值，并聚合成排名头，实现了对高置信预测的结构性脆弱性检测。

**🔧 技术方法**

采用伪 logit 边际、组级删除与退化算子、贪心前向选择、基于多重归一化的 RCMA 指标、分级阈值 λ* 与 FDS 排名头的组合技术。

**📊 数据集**

在七个经典表格基准（Adult、Bank、Credit‑G、Default、Electricity、HELOC、Covertype）上进行实验。

**📈 对比分析**

与最大 Softmax、熵、边际、能量、一次性扰动、组 SHAP 等基线对比，CFC‑FDS 在 AUROC 上达到 0.915（+0.405 提升），在 20% 复核预算下捕获 88.9% 脆弱高置信例子，明显优于其它方法。

**⚠️ 局限性**

结果依赖预设的特征组划分、基线替换与审核深度，贪心路径非全局最优，缺乏正式鲁棒性保证，且对冗余或不合适的组划分会低估脆弱性。

---

## 86. Detoxifying Toxic Communication: A Design Science Approach to Responsible AI

**arXiv ID:** 2609.00361 | [PDF](https://arxiv.org/pdf/2609.00361v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 87. QTEA: Ternary LLMs with Sparse Residual Salient Weight and By-Column Optimization

**arXiv ID:** 2609.00224 | [PDF](https://arxiv.org/pdf/2609.00224v1)

**作者:** Yipin Guo `[一作]` (University of Notre Dame), Siddharth Joshi `[通讯]` (University of Notre Dame)

**通讯引用:** 3198 | [OpenAlex ID](https://openalex.org/A5062849115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为QTEA的子2位后训练量化框架，旨在通过将权重量化为三元值并使用显著权重作为残差补偿器来减轻大语言模型的计算负担。

**💡 创新点**

QTEA通过引入列半稀疏显著权重和列级重缩放优化，显著提高了量化的准确性和效率，同时稳定了误差传播。

**🔧 技术方法**

使用了三元量化、列半稀疏结构、列级重缩放优化和误差衰减等技术。

**📊 数据集**

在Qwen3-14B和Llama3-8B模型上进行了实验，使用了WikiText-2和C4数据集进行评估。

**📈 对比分析**

与现有的最强三元PTQ基线相比，QTEA在Qwen3-14B上提高了16.7%的平均准确性，并在WikiText和C4上分别实现了1.40倍和2.61倍的困惑度降低。

**⚠️ 局限性**

该方法仅关注权重的后训练量化，未对激活或KV缓存进行量化，可能在激活计算或其他系统开销主导运行时的情况下效果较小。此外，当前的超参数设置在大多数实验中保持固定，可能需要根据不同架构进行调整以进一步提高准确性和效率的权衡。

---

## 88. Human-Anchored Factuality Evaluation with Strategic Annotation

**arXiv ID:** 2609.00494 | [PDF](https://arxiv.org/pdf/2609.00494v1)

**作者:** Yu Wang `[一作]` (Amazon AGI), Kevin Small `[通讯]` (Amazon AGI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在有限的人类标注预算下，提出了基于“失败空间分析（Failure‑Space Analysis, FSA）”的采样策略，用于辅助大语言模型（LLM）评判器（LLMaaJ）进行事实性评估，并通过主动统计推断（Active Statistical Inference, ASI）将自动评判与人工纠正结合，得到人类锚定的事实准确率（FAR）。

**💡 创新点**

创新点：①将事实性评估特有的误差模式（如证据不完整、时间不匹配、词义模糊等）系统化为四类特征（评判器输出、证据质量、任务/输入层级、答案与评分准则对齐），并利用这些特征预测评判器与人类标注之间的残差风险；②在此基础上构建FSA管线，自动生成采样策略；③在ASI框架中引入该策略，实现比单纯基于置信度或均匀采样更高效的预算利用。

**🔧 技术方法**

技术：主动统计推断（ASI）框架、基于置信度的采样、FSA特征工程（评判器输出、证据质量、任务/输入层级、评分准则对齐）、梯度提升树（XGBoost）预测残差风险、采样概率归一化与均匀混合、权重调优（power‑tuning）。

**📊 数据集**

数据集：①AutoFA（内部虚拟助手响应的参考基础事实性评估系统，N=2338）；②RAGTruth（公开幻觉基准，N=2937，包含GPT‑4‑0613响应）。

**📈 对比分析**

比较方法：与传统均匀采样人类评估、仅使用评判器的预测、均匀采样的ASI、仅基于自信度的ASI进行对比。性能：在两数据集上均实现了有效样本量（ESS）提升；AutoFA上FSA‑ASI实现+40.3% ESS，RAGTruth上+27.1% ESS；同时保持了覆盖率（95% CI覆盖率≈0.94）和置信区间宽度的可接受水平；相较于置信度驱动的采样，FSA‑ASI在AutoFA上显著提升，RAGTruth上提升幅度相对有限。

**⚠️ 局限性**

局限性：①需要历史/校准数据中残差结构保持稳定，若模型、检索或查询分布发生变化，策略可能失效；②FSA特征未覆盖所有罕见或复杂误差（如多跳推理失败）；③仅考虑单一人类标注者，未建模标注者间噪声或多注者协作；④未解决预算选取决策，需结合成本、延迟等运营因素；⑤在高度置信但实际错误的情况下仍可能被低采样导致方差上升。

---

## 89. Do LLMs Know Your Neighborhood? Auditing LLM Priors for Neighborhood-Level Mobility Prediction and Structural Alignment

**arXiv ID:** 2609.00345 | [PDF](https://arxiv.org/pdf/2609.00345v1)

**作者:** Saad Mohammad Abrar `[一作]` (University of Maryland), Vanessa Frias-Martinez `[通讯]` (University of Maryland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了零样本大语言模型（LLM）是否能通过社会人口与建筑环境信息预测美国四个大都市区的Census Block Group层级聚合移动性指标，并评估其预测的结构一致性。

**💡 创新点**

创新点在于：①提出零样本LLM预测聚合移动性的方法；②设计方向性对齐分析（使用Jonckheere–Terpstra检验），检验LLM推断与经验OLS关系的一致性；③揭示LLM在不同移动性维度与城市之间的偏好与潜在偏见。

**🔧 技术方法**

主要技术包括：prompt工程（生成针对每个CBG的结构化文本提示），LLM推理（Gemma‑3‑27B、Claude‑Sonnet等），监督基线分类（随机森林、XGBoost、Logistic回归等），以及统计对齐方法（OLS回归、方向性JT检验）。

**📊 数据集**

使用的数据集：1) Cuebiq匿名移动轨迹（2021年，约976k设备，8,756 CBG）；2) 2019美国社区调查（ACS）5年估算的社会人口变量；3) EPA Smart Location Database（SLD）的建筑环境指标。四个城市为ATL、LA、MIA、SF。

**📈 对比分析**

与监督基线比较，最佳监督模型平均准确率0.580，最佳LLM平均准确率0.435，差距约0.144。空间扩展指标（如凸包直径、椭圆面积）预测效果最强，但LLM与基线差距最大；熵类指标最难预测。LLM在某些指标（如日间时间碎片化）表现最好。

**⚠️ 局限性**

局限性：①LLM只能部分恢复聚合移动性，预测精度低于监督模型；②方向性分析显示LLM使用粗略、稳定的预测先验，未能充分捕捉不同城市和指标的细微差异；③对受保护群体的预测存在偏差（如黑人比例始终负向），表明模型潜在的刻板印象；④实验仅涵盖四个城市，结果可能不具普适性；⑤缺乏对模型内部推理机制的深入解释，需进一步审计与验证。

---

## 90. Convergence issues in Relational Concept Analysis based on AOC-posets

**arXiv ID:** 2609.00054 | [PDF](https://arxiv.org/pdf/2609.00054v1)

**作者:** Xavier Dolques `[一作]` (University of Strasbourg), Florence Le Ber `[通讯]` (University of Strasbourg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在将传统关系概念分析（RCA）的概念格替换为 AOC‑poset（仅保留引入对象或属性的概念）后，RCA 过程的收敛性问题，并提出了一种能保证收敛的改进算法 RCA-AOC-conv。

**💡 创新点**

创新点在于：①揭示 AOC‑poset 版本的 RCA 在一般情况下可能不收敛的原因；②给出了若干充分条件（无环依赖图、识别对象、上下文单调增长等）保证收敛；③设计了一种“累计属性、保留脏属性”的收敛变体 RCA-AOC-conv，使得即使在存在循环依赖时也能得到稳定的 AOC‑poset；④提出了自动在数据集中加入“标识符”以实现收敛的预处理流程。

**🔧 技术方法**

主要技术包括：关系概念分析（RCA）的定义与实现、AOC‑poset 的构造、存在量化（∃）与严格全称量化（∃∀）的缩放算子、依赖图与单调增长理论、以及改进的累计扩展算法。

**📊 数据集**

使用的数据集主要是：①古代药用植物与处方的示例数据（Plants‑Remedies）；②UML 类模型的重构案例（Operation 等）；③若干人工构造的极端反例（用于展示发散情况）。

**📈 对比分析**

方法评估主要通过理论证明与案例分析完成。作者说明在无环依赖图或经过标识符预处理后，RCA-AOC-conv 可以在有限步内收敛；对比传统 RCA，收敛变体保留了 AOC‑poset 的紧凑性，但产生了“悬挂属性”；在实验上没有给出数值性能对比，主要是从概念数、迭代次数等角度进行定性讨论。

**⚠️ 局限性**

局限性包括：①在需要保留所有非引入概念的应用中，累计属性方法会产生悬挂属性，需额外解释；②标识符预处理会导致每个对象单独成概念，可能增大概念数；③本文仅给出理论与案例验证，缺乏大规模真实数据的实验评估；④对于更复杂的缩放算子或高阶关系，收敛条件与实现仍待进一步研究。

---

## 91. Harness Engineering: Anatomy, Architecture, and Evolution of Coding Agents -- A Source-Code Study of Eleven Systems

**arXiv ID:** 2609.00006 | [PDF](https://arxiv.org/pdf/2609.00006v1)

**作者:** Paul Barbaste `[一作]` (Inclusive Brains), Tom Wiltberger `[通讯]` (Wavestone AI Lab)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 11 个生产级代码代理（harness）的源代码进行系统化解剖，定义了 harness 的概念及其七大子系统，并归纳了 13 条跨系统观察与 29 条设计模式。

**💡 创新点**

首次把 harness 视为独立的运行时，提供完整的源代码参考，并揭示其从“工具”向“平台”的演进路径，提出 18 条设计建议和 90 行可直接使用的最小可行 harness 模板。

**🔧 技术方法**

通过静态源代码审计、差异对比、功能映射与模式归纳，比较了不同实现的循环、LLM 集成、工具系统、记忆、权限、安全、编排和可扩展性。

**📊 数据集**

使用 2026 年 7 月发布的 11 个公开或闭源系统（约 400 万行代码）作为基线，并与 2026 年 4 月的快照对比，构成纵向演化样本。

**📈 对比分析**

对比方法基于源代码结构与功能映射，不进行性能基准；但通过跨系统共性与差异的分析，观察到实现趋同与功能演进，未量化性能指标。

**⚠️ 局限性**

局限性：仅涵盖已公开的 11 个系统，未评估私有/实验性框架；未进行任务级基准测试，仅分析代码结构；安全与合规细节未深入；缺乏真实工作负载下的性能评估。

---

## 92. Life Operators: a self-evolving framework for multiscale life modelling

**arXiv ID:** 2609.00068 | [PDF](https://arxiv.org/pdf/2609.00068v1)

**作者:** Shuo Wang `[一作]` (Fudan University), Yike Guo `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 19510 | [OpenAlex ID](https://openalex.org/A5045081171)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了Life Operators框架，将医学AI从仅识别向病人状态预测与干预影响预测转变，构建可组合的感知、演化和生成模块以及桥接器，形成任务特定的Operator Graph；

**💡 创新点**

创新点在于用统一的科学角色语言定义可验证的模块，支持版本化、局部修订与证据驱动的自我进化；

**🔧 技术方法**

结合贝叶斯推断、混合统计与物理方程、神经网络、AutoResearch代理等技术实现感知、演化、生成和桥接的多模态实现；

**📊 数据集**

利用多源医学数据（电子健康记录、影像、基因组、临床试验数据等）训练和验证各模块，示例中以HCM患者的超声、药代动力学及实验室数据为例；

**📈 对比分析**

通过与简化的曝露-反应模型和独立验证数据对比评估Operator Graph的预测精度与校准，虽然没有给出具体数值，但指出其能在多尺度、干预条件下保持一致性；

**⚠️ 局限性**

局限包括缺乏大规模跨机构验证、对干预因果证据依赖较高、模型复杂度与可解释性权衡，以及在多尺度动态耦合时可能出现的数值不稳定性与计算成本问题。

---

## 93. Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs

**arXiv ID:** 2609.00621 | [PDF](https://arxiv.org/pdf/2609.00621v1)

**作者:** Wentao Zhang `[一作]` (University of Waterloo), Yuntian Deng `[通讯]` (University of Waterloo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出控制-数据流分离机制，让多代理LLM系统在提示优化时保持执行协议不变。

**💡 创新点**

创新点是将执行关键控制信息与可优化的自然语言内容拆分为结构化控制通道和自由文本数据通道，并通过冻结的schema确保程序安全。

**🔧 技术方法**

使用Python库cdsep，利用Pydantic/ dataclass定义控制schema，结合文本梯度优化、提示框架和解析重试机制实现。

**📊 数据集**

在BBH（BIG-Bench Hard）、MARG（ICLR论文审稿）、合成保险承保与行业验证保险承保四个任务上进行实验。

**📈 对比分析**

与固定提示、Naive TextGrad、DSPy+BootstrapFewShot/MIPROv2等基线相比，在所有四个任务中均实现最高任务指标且协议有效率100%。

**⚠️ 局限性**

局限在于只能保证控制协议有效，无法保证输出语义正确；需要预先定义schema，且未评估动态代理创建或schema演进的情况。

---

## 94. Visual Framing for News Stance Detection via Image Generation

**arXiv ID:** 2609.00685 | [PDF](https://arxiv.org/pdf/2609.00685v1)

**作者:** Dahyun Lee `[一作]` (Soongsil University), Kunwoo Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了利用视觉框架与文本到图像生成技术，将隐含的新闻立场线索显化，并提出多阶段框架（LLM生成视觉框架规范 → T2I生成立场相关图像 → LVLM多模态立场预测）以提升文章级新闻立场检测。

**💡 创新点**

创新点在于将视觉框架理论与生成式图像技术结合，构建多模态框架，将长篇新闻文本中的隐性立场线索转化为可视化信号；同时通过实验验证不同视觉框架层级对立场识别的影响，并在多语言、图像缺失场景中保持有效。

**🔧 技术方法**

技术手段包括：①大语言模型（Gemini-3-flash）用于抽取视觉框架规范；②文本到图像模型（Gemini-3.1-flash-image）生成立场相关图像；③大型视觉语言模型（LVLM）对文本与图像进行联合推理；此外还采用了对比基准的多模态融合方法、LLM推理与链式思维等。

**📊 数据集**

数据集：①韩文多模态立场检测数据集（1816篇新闻，包含原始图片）；②德文文本级立场检测数据集（1762篇）；③四种语言（英、汉、印、阿）翻译版本，用于跨语言评估。

**📈 对比分析**

与文本、视觉、跨模态及LLM基准对照实验，结果显示在韩文数据集上该方法ACC 0.746、mF1 0.747，均显著高于所有基线；在德文文本集也保持最高性能；消融实验进一步验证视觉框架与图像生成对性能提升的贡献；用户研究表明生成图像能提升读者立场识别准确率。

**⚠️ 局限性**

限制包括：①计算成本高（三阶段模型需大量算力）；②仅在韩文图像数据集上进行充分验证，跨语言图像测试受限；③对专有模型的依赖，开放模型表现略逊；④生成图像可能携带偏见与合成识别风险。

---

## 95. AdaptNTK: Adaptive Uncertainty Quantification and Active Learning for Neural Network Potentials

**arXiv ID:** 2609.00488 | [PDF](https://arxiv.org/pdf/2609.00488v1)

**作者:** Prajwal Ananth `[一作]` (Cornell University), Shuwen Yue `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AdaptNTK 框架，利用已训练的神经网络的经验神经切线核特征来做点wise 不确定性估计，并在主动学习中实现无标签递归更新；

**💡 创新点**

创新点在于把正则化的马氏距离映射到 NTK 特征空间做不确定性评估，并通过 rank‑one 更新在同一次采样批内动态消除冗余，避免了多模型或重训练的成本；

**🔧 技术方法**

使用神经切线核（NTK）与随机 sketch 投影来降低特征维度，结合 Sherman‑Morrison 递归公式实现快速更新；实验采用 MACE 神经网络结构；

**📊 数据集**

主要数据集为 rMD17 量子力学分子集和 Transition‑1X 过渡态分子集；

**📈 对比分析**

与集成、MC dropout、EDL、SWAG 等传统不确定性方法相比，AdaptNTK 在 rMD17 与 Transition‑1X 上实现了最高的误差相关性（Spearman≈0.68、Pearson≈0.71）、最低的力 RMSE，并在主动学习中比集成快约 2.6 倍、耗时不到一半；

**⚠️ 局限性**

局限包括仅在气相分子上验证，未覆盖周期性体系；不确定性基于能量梯度，无法直接捕捉力误差；对长期模拟性能、化学直觉或多样性采样策略的影响未做评估。

---

## 96. Semi-Supervised Virtual Staining via Morphology Preservation and Histopathological Realism Constraints

**arXiv ID:** 2609.00984 | [PDF](https://arxiv.org/pdf/2609.00984v1)

**作者:** Baoshun Wang `[一作]` (Xiamen University), Liansheng Wang `[通讯]` (Xiamen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

提出了一种稳定的半监督虚拟染色框架，利用有限的配对样本与大量未配对源图像联合学习，从而降低对高质量配对样本的依赖。

**💡 创新点**

创新点在于：① 采用 Hessian‑derived morphology preservation（HDMP）从源图像提取形态结构监督；② 引入基于 frozen CONCH 视觉‑语言模型的 histopathological realism constraints（HRC）来引导生成图像符合目标染色的真实特征；③ 将两种监督机制结合，显著提升未配对训练的稳定性与效果。

**🔧 技术方法**

技术手段包括：基于 Pix2PixHD 的 GAN 生成器与判别器；Hessian 二阶导数与 Laplacian 过滤提取形态信息；CONCH 文本与图像编码器用于图像‑文本与图像‑图像的相似度约束；多重损失函数（GAN、重建、形态、真实感）共同训练。

**📊 数据集**

使用了三类虚拟染色任务的数据集：
- H&E → IHC (Ki67)：MIST‑Ki67 与 IHC4BC 公开数据；
- H&E → IHC (HER2)：私有 self‑HER2 数据集；
- FFPE → H&E：私有 self‑FFPE 与 self‑FFPE2 数据集。

**📈 对比分析**

与 Pix2Pix、Pix2PixHD、ASP、PyramidPix2Pix、TDKStain、HistDiST、UNIStainNet 等方法进行了对比。评估指标涵盖图像质量（PSNR/SSIM/MS‑SSIM/FID/KID/DISTS）、染色特定指标（DAB‑KL、IOD‑D、H‑Dice/E‑Dice）以及鲁棒性和下游诊断性能。实验表明，在所有三种任务中，加入 HDMP + HRC 后，PSNR、SSIM、FID 等指标均显著提升（如 H&E→IHC HER2：PSNR 18.39→18.60，FID 87.68→52.45），且在染色变异与采集降解场景下表现更稳健，甚至在 HER2 分级任务中取得最佳准确率。

**⚠️ 局限性**

局限性包括：仍需一定比例的配对样本，配对质量对结果影响大；需要手动调参（损失权重、训练阶段切换）；目前仅在三种染色任务上验证，未覆盖更广泛的染色组合；引入 CONCH 模型增加推理时间与资源消耗；对极端未配对样本的泛化能力尚未充分评估。

---

## 97. DynaNDE: Dynamic Near-Data Expert Scheduling for Batched MoE Inference

**arXiv ID:** 2609.00407 | [PDF](https://arxiv.org/pdf/2609.00407v1)

**作者:** Xiaoyang Lu `[一作]` (Illinois Institute of Technology), Xian-He Sun `[通讯]` (Illinois Institute of Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种动态近数据专家调度框架 DynaNDE，结合 NPU 与 NDP 的协同执行，对批量 MoE 推理进行专家级动态调度并利用专家重用。

**💡 创新点**

创新点包括：① 基于统一分析性能模型量化硬件异构、数据移动与计算重叠；② 通过重用感知的专家评分和前缀扫描实现低开销动态调度；③ 同时兼顾专家级并发与时间重用，显著提升吞吐量。

**🔧 技术方法**

使用了分析性能模型、重用感知专家评分、前缀扫描搜索、NPU-NDP 协同执行、CXL 接口、Host‑side 调度运行时等技术。

**📊 数据集**

评测数据集采用 DCLM‑Baseline 数据集进行推理实验。

**📈 对比分析**

通过与 NPU、NDP、MoNDE、HybriMoE 四个基线在预填充、解码阶段、不同批量、缓存大小及替换策略下进行对比，DynaNDE 在预填充阶段平均提升 1.8–2.9 倍，在解码阶段平均提升 30.5×、1.1×、2.2× 等，显著优于基线。

**⚠️ 局限性**

局限性在于仅针对单一 NDP 设备，扩展到多 NDP 需要解决专家放置、激活路由、负载均衡和共享互连争用等系统级挑战；评测仅覆盖有限几种 MoE 架构和 DCLM 数据集，尚未验证更大模型或多样化任务。

---

## 98. Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs

**arXiv ID:** 2609.00184 | [PDF](https://arxiv.org/pdf/2609.00184v1)

**作者:** Jonathan Zheng `[一作]` (Georgia Institute of Technology), Wei Xu `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了一个基于未来事件的无污染评估基准 ParallelEvents，并提出了基于合成数据的可扩展知识插入框架 Synapse，用于对大型语言模型的知识更新进行评估与训练。

**💡 创新点**

创新点在于①设计了可保证局部与全局一致性的并行虚拟事件图谱基准；②利用教师模型生成高质量长文本和偏好对，通过 DPO 训练实现参数级知识注入，同时抑制幻觉与不必要的回避；③通过检索+多步推理的混合方式提升多跳与因果推理性能。

**🔧 技术方法**

使用的技术包括 GPT‑4 生成合成文本、知识图谱一致性检验、偏好学习与 Direct Preference Optimization（DPO）、检索增强的多步推理、以及传统 LoRA、单层微调、ICE/IKE 等基准方法。

**📊 数据集**

采用的数据集包括自构建的 ParallelEvents（2030‑2035 年 41 个事件的知识图谱）、Synapse 合成的长文本与 QA 对（涵盖单跳、多跳、因果三类问题）、以及 MQuAKE‑Remastered 等现有对抗性编辑基准。

**📈 对比分析**

在单跳/多跳/因果等评估任务上，Synapse 与检索增强的 IKE、LoRA、GWalk、MeLLo 等方法对比，整体准确率提升约 14%，在因果推理上最高可达 79%，显著优于现有所有基准方法。

**⚠️ 局限性**

局限性包括：对精确数值或数值型属性的单跳问答仍依赖检索且易出错；大规模跨领域事件的泛化能力尚需进一步验证；以及合成数据的质量与一致性仍需人工审查以防偏差。

---

## 99. Puppeteer: Object-Grounded Posture-Aware Co-Speech Gesture Generation

**arXiv ID:** 2609.00369 | [PDF](https://arxiv.org/pdf/2609.00369v1)

**作者:** Vida Adeli `[一作]` (Pickford AI), Babak Taati `[通讯]` (Pickford AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种面向姿态感知与物体感知的共语音手势生成框架，利用因果潜在空间的扩散模型在语音、文本、姿态和场景几何上进行自回归生成。

**💡 创新点**

创新点包括：①在连续因果潜在空间中实现手势的时序编码与扩散；②引入音频窗口与文本跨度的自适应对齐掩蔽提升语音-手势同步；③通过门控对象融合模块将3D物体几何注入生成过程；④构建首个合成的姿态与物体耦合共语音手势数据集。

**🔧 技术方法**

使用技术包括：因果VAE（CausalVAE）对手势分块编码；条件扩散（latent diffusion）与DDIM采样；Transformer的自注意力与跨注意力；音频窗口与文本跨度对齐掩蔽；BPS（Basis Point Set）物体表示与门控对象融合；姿态编码与姿态参考交叉注意力。

**📊 数据集**

训练与评估使用BEAT2、Embody3D公开数据集，以及新构建的3D场景共语音手势数据集（包含不同椅子、桌子、对话情景），并通过合成视频与SMPL‑X重建得到真实的手势与物体交互数据。

**📈 对比分析**

与多种基线（EMAGE、GestureLSM、Diffusion‑StyleGesture、ConvoFusion等）在FGD、BC、ΔBC、Div、LL1、MeanPen/MaxPen等指标上比较。实验显示：①在语义对齐与时序同步上取得最高BC与最低ΔBC；②在手势多样性上实现最优Div；③在姿态一致性与物体碰撞抑制上显著降低LL1、MeanPen和MaxPen，表明模型能更好地满足姿态与物体约束。

**⚠️ 局限性**

局限性：①模型主要训练于合成数据，真实多样化环境中的泛化仍待验证；②目前仅支持被动物体约束，缺少主动交互（抓握、搬运）功能；③BPS近似物体几何在复杂几何下可能不足；④扩散采样时间较长，实时部署仍有挑战。

---

## 100. SilentProbe: Measuring Silent Failure in Production APIs Used as Agent Tools

**arXiv ID:** 2609.00035 | [PDF](https://arxiv.org/pdf/2609.00035v1)

**作者:** Zongrong Li `[一作]` (Texas A&M University), Zuoyou Dang `[通讯]` (Monid, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大语言模型代理在调用生产 API 时出现的“沉默失败”现象，并系统评估了 API 文档中约束的形式（机器可检验 vs 仅文本）对错误检测和代理行为的影响。

**💡 创新点**

创新点在于提出 SilentProbe 方法，利用静态审计与差分探测分辨机器可检验与仅文本约束，并通过实证证明约束形式决定沉默失败，而非供应商身份；同时展示将词汇提升为枚举即可用一行 schema 修复问题的简单而有效的手段。

**🔧 技术方法**

使用了静态审计、差分探测（基于响应签名比较）、批量调用商用端点的实际实验、完整代理循环评估、以及对 12 个大语言模型的行为分析等技术。

**📊 数据集**

数据集包括 721,320 个公开 OpenAPI 参数（来自公开 OpenAPI 目录）和 469 条通过 Monid 聚合层公开的商业端点（共 27 家供应商），以及 219 次针对 71 个参数的扰动实验。

**📈 对比分析**

对比方法：统计机器可检验约束导致 111/111 的诚实错误，而仅文本约束导致 44/61 的沉默失败；在 12 个模型上，未修复的沉默失败率从 88% 降至 0%，证明单行枚举修复效果显著；整体实验展示了明显的性能提升。

**⚠️ 局限性**

limitations 包括：样本偏倚（仅覆盖 Monid 的商业端点）、基线值合成误差、被动排除无效/无效参数导致的下限估计、差分探测无法捕捉“语义降级”模式，以及实验依赖单一聚合层可能影响通用性。

---

## 101. Solaris: Towards Interfaces That Are Generated, Not Coded

**arXiv ID:** 2609.00776 | [PDF](https://arxiv.org/pdf/2609.00776v1)

**作者:** Yuval Alaluf `[一作]` (Runway), Hudson Yeo `[通讯]` (Runway)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Solaris的接口世界模型，能够在用户交互时实时逐帧生成完整的视觉界面，直接实现界面与行为的动态生成；

**💡 创新点**

创新点在于：1）消除传统中间代码表示，直接以视觉输出为接口；2）将语言模型与视觉生成模型分离，语言模型负责意图理解与交互策略，视觉模型负责渲染；3）通过自回归、少步蒸馏及自监督训练实现实时性能与长时序一致性；

**🔧 技术方法**

使用改进的Gen‑4.5视频生成模型，结合自回归帧生成、few‑step蒸馏以及在自身输出上训练的技术；辅以大型语言模型（如Claude Opus 5）进行意图解析和提示生成；

**📊 数据集**

主要数据集包括30个多样化的界面截图（从简易网页到图片密集页面），以及在此基础上进行的250名参与者的用户交互评测数据；

**📈 对比分析**

与传统编码接口（用Claude Opus 5复现同一界面并响应同一交互）进行对比。用户评测显示Solaris在指令遵循上62%优于25%，在自然行为上72%优于21%；整体表现明显优于基于代码的实现；

**⚠️ 局限性**

局限性包括：1）文本生成不稳定，难以实时渲染清晰文本；2）对信任与事实准确性的依赖仍需加强；3）长时间交互中的视觉/语义一致性仍有限；4）与辅助技术（屏幕阅读器等）的集成尚未完善。

---

## 102. Dense Process Supervision for Search Agents via Fact Utility Estimation

**arXiv ID:** 2609.00833 | [PDF](https://arxiv.org/pdf/2609.00833v1)

**作者:** Rongzhi Zhu `[一作]` (Nanjing University), Wei Hu `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出FactAgent框架，将检索到的证据抽象为结构化事实并存入显式事实库；在此基础上通过贝叶斯估计聚类事实的效用，并将其转化为密集的步骤级奖励，从而解决搜索型LLM代理的信用分配问题。

**💡 创新点**

创新点在于：①用结构化事实取代原始文本历史，显著压缩上下文；②通过聚类并贝叶斯估计事实效用，克服单个事实稀缺导致的高方差；③将估计的事实效用通过潜在奖励塑造映射为步骤级奖励，实现密集过程监督；④将奖励重新分配给检索动作，提升搜索策略。

**🔧 技术方法**

核心技术包括：强化学习的Group Relative Policy Optimization（GRPO）；贝叶斯Beta先验与后验估计；事实聚类（基于向量相似+逻辑规则）；潜在奖励塑造（PBRS）；奖励重分配机制；以及对抗式SFT与RL训练的组合。

**📊 数据集**

在七个QA基准上进行评估：单跳数据集（NQ、TriviaQA、PopQA）与多跳数据集（HotpotQA、2Wiki、MuSiQue、Bamboogle），使用Qwen2.5‑3B‑Instruct和Qwen2.5‑7B‑Instruct两种LLM骨干。

**📈 对比分析**

与RAG、Search‑R1、ReSearch、AutoRefine、GiGPO、Zerosearch等检索+RL基线相比，FactAgent在两种骨干上均取得最高加权平均EM；在7B模型上平均提升约3.2 EM，尤其在多跳任务上优势更为显著。

**⚠️ 局限性**

局限性包括：①计算步骤级奖励需要恢复交互上下文，导致GPU内存消耗高于单纯结果奖励；②效用估计依赖聚类质量和GRPO采样量，若聚类不当会影响奖励信号；③目前仅在静态QA数据集上验证，未对开放式动态任务（如GAIA）进行测试。

---

## 103. Delegation Without Trust: An Empirical Gap Analysis of Identity, Authorization, and Runtime Governance in Multi-Agent LLM Systems

**arXiv ID:** 2609.00267 | [PDF](https://arxiv.org/pdf/2609.00267v1)

**作者:** Panduranga Sai Varma Dantuluri `[一作]` (VotalAI), Jyotirmoy Sundi `[通讯]` (VotalAI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估一种针对多代理LLM系统的授权代理（Broker），提出untrusted-model假设下的威胁模型与八项安全需求，并对四个主流框架进行差距分析，最终在生产系统VotalAI LLM Shield中实现与验证。

**💡 创新点**

①以untrusted-model为前提提出完整的多代理授权威胁模型和安全需求；②首次对LangGraph、CrewAI、AutoGen与MCP四个框架进行系统化差距分析；③设计并实现一种基于标准原语（OAuth2.1、Token Exchange、SPIFFE/SPIRE、mTLS、DPoP、Macaroons/Biscuit、MCP）的Broker，满足R1–R8；④对代理进行对抗评估，证明其在微秒级成本下能显著限制攻击者爆炸半径。

**🔧 技术方法**

组合OAuth2.1、Token Exchange、SPIFFE/SPIRE、mTLS、DPoP、Macaroons/Biscuit、MCP；使用Python实现Broker（约160行）；HMAC签名、微秒级权限检查与令牌交换；在实验中模拟8100种工具动作、2000个随机委托场景；生产中使用VotalAI LLM Shield、Kong、LiteLLM、Portkey、Redis、vLLM、NVIDIA Nemotron 3.5、H100/B200 GPUs。

**📊 数据集**

主要使用合成攻击样本（200,000个伪造令牌）和内部合成工具/资源动作集；在内容过滤层使用VotalAI adversarial dataset（未公开）。

**📈 对比分析**

通过与默认运行时对比，四类攻击（T1–T4）全部被阻断；对抗测试11种专门攻击均被拦截；令牌交换耗时约5.4 µs，授权决策约2.6 µs；子代理爆炸半径从8100动作下降到平均1.5动作；与模型推断耗时（毫秒级）相比，Broker开销可忽略不计。

**⚠️ 局限性**

仅在合成实验环境验证，未集成到主流框架的完整部署；未完成模型端到端的攻击实验；依赖可靠的工作负载身份认证（SPIFFE/mTLS/DPoP）——若攻击者获得SVID则仍能利用；生产实现未公开，无法复现细节；对高级攻击向量（如侧信道或物理攻击）的评估缺失。

---

## 104. Assessing Alignment and Stability of Feature Importance Explanations via Weight of Evidence

**arXiv ID:** 2609.00090 | [PDF](https://arxiv.org/pdf/2609.00090v1)

**作者:** Eddie Conti `[一作]` (Barcelona Supercomputing Center), Mario G. C. A. Cimino `[通讯]` (University of Pisa)

**通讯引用:** 1994 | [OpenAlex ID](https://openalex.org/A5019498183)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

引入Weight of Evidence（WoE）框架，用以评估特征重要性方法（FIM）与参考假设的对齐与稳定性，并通过实验验证其有效性。

**💡 创新点**

创新点在于：①将WoE作为对FIM解释的统计检验工具；②证明低变异性导致WoE趋于无穷，直接量化内部稳定性；③通过域知识、真实标签和内部平均等不同参考假设展示WoE的多样化应用。

**🔧 技术方法**

使用的技术包括：WoE统计推断、LIME/SHAP解释器、Top‑k 与自适应特征选择、Monte Carlo 估计、理论证明与实证验证。

**📊 数据集**

实验使用的数据集包括 Titanic、合成 Ground‑Truth 数据、Diabetes、Heart Disease、Mobile、Churn 等。

**📈 对比分析**

比较方法是基于WoE分数评估 LIME 与 SHAP 在对齐（域知识或 Ground‑Truth）和内部稳定性上的表现，实验表明 LIME 在某些区域更一致；SHAP 更一致但更确定；正向 WoE 分数表明解释的一致性，且实验验证了理论结论。

**⚠️ 局限性**

局限性包括：需要多次运行解释器以获得后验估计；后验匹配要求特征集合完全相同，缺乏软匹配；默认均匀先验可改进；在模型复杂度高时可能导致解释不稳定。

---

## 105. Efficient discovery of unique column combinations on disk-resident data with limited memory

**arXiv ID:** 2609.00783 | [PDF](https://arxiv.org/pdf/2609.00783v1)

**作者:** Xiaolong Wan `[一作]` (Harbin Institute of Technology), Xixian Han `[通讯]` (Harbin Institute of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 DUD（Disk‑resident Unique Column combination Discovery）的算法，能够在磁盘驻留且内存受限的环境下高效地发现关系实例的所有最小 UCC（Unique Column Combination）。

**💡 创新点**

创新点包括：
• 通过属性 FG/NG 分类，只对部分属性生成完整有用的差集，避免了 O(n²) 的全排列比较；
• 证明若候选集包含至少一个 FG 属性则一定是 UCC，直接报告为真 UCC，显著减少验证次数；
• 采用基于哈希的批量验证策略，只保留当前同值区块的少量元组，极大降低内存占用；
• 通过增量差集更新和多轮最小穿透集枚举，保证完整性与效率兼顾。

**🔧 技术方法**

使用的技术包括：
• 超图与最小穿透集（transversal hypergraph）对应 UCC 问题；
• MMCS 算法实现最小穿透集枚举；
• 通过列文件排序（单列文件排序 + 两路多路归并）快速统计并决定 FG/NG 属性；
• 采样生成补充差集；
• 哈希表批量检查冲突，生成新的差集并更新超图；
• 依据属性最大重复值长度顺序选择验证顺序。

**📊 数据集**

实验使用了：
• 合成 TPC‑H 的 lineitem 表（规模从 6M 到 120M 行，属性 16）；
• 30+ 真实数据集（来自 UCI、Kaggle 等），行数从 24K 到 180M，属性从 5 到 115，UCC 数量从 1 到 10,000+。

**📈 对比分析**

与现有最先进的 HPIValid 进行对比。结果显示：
• 在内存受限（≤32 GB）时，DUD 在大多数数据集上运行速度相当或更快，尤其是当 UCC 结果集小或有足够 FG 属性时，速度可提升 3–6 倍；
• 内存占用始终低于 8 GB（在 24 GB 内存预算下）；
• 在 HPIValid 超出内存或需要大量全表扫描时，DUD 成功完成；
• 在高维或 UCC 数量极大的数据集，HPIValid 在内存可容纳时仍可能略快，但 DUD 仍保持可行。

**⚠️ 局限性**

局限性包括：
• 当几乎所有属性都是 NG 类型（如 prsa）时，验证阶段占比激增，性能下降；
• 对高维/大结果集的 UCC 仍可能产生大量候选，导致验证成本上升；
• 仅在单机内存受限场景下验证有效，未实现分布式扩展；
• 对 γ 参数的固定取值（3 × n log₂n）需人工调优，未自适应；
• 内存占用虽低，但仍受数据分布（最大重复值长度）影响，缺乏严格的 worst‑case 绑定。

---

## 106. ReDeck: Step-Level Render-Grounded Refinement for Document-to-Slide Generation

**arXiv ID:** 2609.00194 | [PDF](https://arxiv.org/pdf/2609.00194v1)

**作者:** Muzhao Tian `[一作]`, Chong Luo `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了步级渲染基准的幻灯片生成改进框架。

**💡 创新点**

将修订拆分为原子编辑并在每一步返回渲染观察，结合多粒度反馈。

**🔧 技术方法**

使用LLM+工具链、渲染环境、步级观察、适应性幻灯片评判器、提交门控。

**📊 数据集**

构建ReDeck基准，使用100篇学术论文以及PresentBench等数据集。

**📈 对比分析**

在GPT‑5.4/Claude‑4.6/Gemini‑3.1等模型上与SlideGen、SlideTailor、DeepPresenter对比，ReDeck在内容、空间、审美和信息架构四项指标均显著领先。

**⚠️ 局限性**

成本高、仅适用于可编辑几何和可重复渲染的格式，无法处理PDF/动画等非可编辑媒介，且与评判器耦合度较高。

---

## 107. CoLT-Drive: Counterfactual Long-Tail Benchmarking and Knowledge-Preserving Adaptation for Driving Affordance Prediction

**arXiv ID:** 2609.00242 | [PDF](https://arxiv.org/pdf/2609.00242v1)

**作者:** Zhengxu Tang `[一作]` (Nvidia), Pichao Wang `[通讯]` (Nvidia)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了决策层次的稀有物体可操作性预测任务，并设计了 3,536 样本的对照长尾驾驶基准 CoLT‑Drive；同时研发了一套知识保留的适配框架，能在保持预训练 VLM 通用知识的同时提升稀有物体下的元动作预测。

**💡 创新点**

创新点在于：①把长尾驾驶失败从单纯的物体识别转向决策层面的可操作性推理；②构造对照式置信度基准，精确测量稀有物体对可行动作空间的影响；③提出结构化感知‑决策提示、SLERP 权重融合以及基于驾驶情景的 LoRA Mixture‑of‑Experts 的三段式知识保留适配方案。

**🔧 技术方法**

主要技术包括：结构化 prompt 与链式思维、SLERP（球面线性插值）进行模型融合、基于 LoRA 的轻量级适配器、情景路由（Routine/Maneuver/Reactive）和多任务损失。

**📊 数据集**

使用了两类数据：①CoLT‑Drive 对照长尾样本（3,536 张），由 29 个基准场景、50 种稀有物体和 5 种可操作性类别构成；②10,000 片包含 26 个 ODD 类别的在域驾驶日志，用于训练与评估。

**📈 对比分析**

对比方法主要是 Qwen3‑VL‑2B 的原始模型、LoRA SFT、仅 SLERP 融合和完整三段适配。结果显示，完整模型在 CoLT‑Drive 的 pair‑accuracy 达到 60.8%，比原始 50.3%、LoRA SFT 32.4% 和单一 SLERP 53.0% 提升显著；在各可操作性类别上也表现出显著的安全性提升。

**⚠️ 局限性**

局限性：①基准仅为图像层级的对照诊断，未覆盖闭环物理交互与车辆动力学；②使用抽象元动作而非连续轨迹或低层控制，无法评估轨迹可行性与舒适度；③数据来源有限，可能存在视觉伪影与场景多样性不足；④适配研究仅针对小型 VLM，未验证在更大模型或真实长尾日志上的泛化。

---

## 108. From Source Reconstruction to Predictive State Preservation: An Information-Theoretic Framework for AI-Native Communication

**arXiv ID:** 2609.01131 | [PDF](https://arxiv.org/pdf/2609.01131v1)

**作者:** Yi Wang `[一作]` (Tsinghua University), Linglong Dai `[通讯]` (Tsinghua University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

本文提出一种新的通信框架，将通信目标从源重构转变为预测任务的**预测状态**保真度，即在已知共享上下文的前提下，保持源观测对未来目标的完整预测分布。

**💡 创新点**

创新点在于：
1) 将“保真对象”视为预测状态并证明其是最小充分统计量；
2) 通过终端预测损失推导出预测失真，并在对数损失下与条件互信息完全对应；
3) 将通信、模型族、部署三种误差拆解为可加三项，揭示其各自的信息量与可行补救措施；
4) 在有限字母无记忆模型下证明对预测状态的编码率-失真曲线与原始源编码等价；
5) 在序列预测中区分有限窗口预测状态与全未来预测状态，证明后者既满足充分性又可递归更新。

**🔧 技术方法**

主要技术包括：
- 信息理论中的充分统计与最小化证明；
- 正确分数函数（Proper scoring rule）与对数损失下的Kullback‑Leibler相容性；
- 条件率失真理论与可测分解；
- 递归更新与马尔可夫表示的可测化构造；
- 典型的正则化与贝叶斯风险分解。

**📊 数据集**

该工作为理论研究，未使用具体实验数据集；所有结论基于概率模型、标准Borel空间和有限字母假设。

**📈 对比分析**

比较方法：
- 对数损失下的预测失真直接等于条件互信息，能够量化通信过程中的信息流失；
- 通过三段式误差分解，可对照不同系统层级的瓶颈；
- 在匹配压缩实验中，对等价的编码方式（原始源编码 vs 预测状态编码）率-失真曲线相同；
- 在序列预测中，有限窗口状态与全未来状态的递归性差异在理论上得到阐明，说明全未来状态能保持更强的递归闭合性。

**⚠️ 局限性**

限制与前置假设：
- 需要源、预测目标和上下文在标准Borel空间内，且共享上下文已知；
- 匹配压缩结果主要针对无记忆有限字母模型；
- 对数损失特殊性导致信息量与失真精确对应，其他损失函数需使用泛化的正分数规则；
- 对序列预测的递归闭合性仅在全未来状态成立，有限窗口状态可能失去递归性，需要额外假设或近似。

---

## 109. Latent Mechanisms of Language Control in Multilingual Language Models

**arXiv ID:** 2609.00325 | [PDF](https://arxiv.org/pdf/2609.00325v1)

**作者:** Ryo Mitsuhashi `[一作]` (Princeton University), Majd Hawasly `[通讯]` (Hamad Bin Khalifa University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究跨层变换器（CLT）中语言控制潜在变量，比较价值选择、频率选择和LLM生成注释选择三种方法，并在两套新建的控制式代码切换基准（Antonyms 与 Enumerations）上评估干预效果。

**💡 创新点**

首次在CLT架构上系统对比三种潜在变量发现方法，提供两套专门测试语言控制的基准数据集，并揭示潜在变量在不同方法间存在冗余而非唯一的语言方向。

**🔧 技术方法**

使用稀疏自编码器（SAE）扩展的跨层变换器（CLT），激活统计（均值、频率）与LLM生成的注释筛选，干预策略包括目标放大、干扰器零化、方向消融及其组合。

**📊 数据集**

利用 FLORES+ 的七种语言数据进行潜在变量提取，构造 Antonyms（词对任务）和 Enumerations（列表续写任务）两套评测数据集。

**📈 对比分析**

在 Antonyms 和 Enumerations 上通过 Zero+Amp 干预进行评测，结果显示注释选择（AnnSel）在大多数语言下取得最高增益，频率选择（FreqSel）与价值选择（ValSel）也表现良好；所有方法均能显著提升目标语言的 logit 分数。

**⚠️ 局限性**

实验仅在两个小规模 CLT（Gemma‑2‑2B 与 Qwen3‑4B‑4B）上进行，语言覆盖有限（主要为高资源语言），LLM 注释质量不均导致部分语言潜在变量缺失，计算成本未进行系统评估。

---

## 110. Efficient and Robust Absolute Pose Estimation via Gravity-Prior-Driven Transformation Decoupling and Pose Refinement

**arXiv ID:** 2609.00713 | [PDF](https://arxiv.org/pdf/2609.00713v1)

**作者:** Hu Cao `[一作]` (Southeast University), Alois Knoll `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于已知重力方向的变换分离+单维全局投票+RANSAC+精细化的四自由度绝对位姿估计方法。

**💡 创新点**

核心创新在于旋转优先的分离策略、全局投票去噪+旋转估计以及针对4-DoF的隐藏变量结果式精细化。

**🔧 技术方法**

采用了重力先验、罗德里格斯公式、1D全局投票、RANSAC、线性最小二乘、隐藏变量结果式优化和多线程加速技术。

**📊 数据集**

在合成数据以及TUM RGB‑D、ETH3D、RobotCar、KITTI四个公开数据集上进行评估。

**📈 对比分析**

与BnB、Gao、Ap3、Swee、Kuke、Sqpnp、Epnp、SupeRANSAC、AaPnP以及ORB‑SLAM2原版比较，实验表明在高比例外点和噪声下鲁棒性更强，定位误差（APE/RPE）降低约30%–50%，重定位漂移显著减小，尽管总运行时间略高，但可通过并行实现实时。

**⚠️ 局限性**

主要局限在于对重力先验的依赖；若重力方向误差大会导致误差放大，且单线程下计算量仍高于传统RANSAC方法。

---

## 111. Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving

**arXiv ID:** 2609.00111 | [PDF](https://arxiv.org/pdf/2609.00111v1)

**作者:** Xin Zhou `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 40876 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个统一的视觉-语言-动作(VLA)基础模型，将预训练的视觉-语言模型(Qwen3.5-4B)与外部BEV感知头和规划专家相结合，实现在同一模型中完成3D感知、驾驶问答与轨迹规划；

**💡 创新点**

①保持预训练VLM架构不变，保留通用视觉语言能力；②引入可检验的BEV感知头，提供3D检测、占据预测与地图分割；③使用分阶段训练与统一数据策略，融合多任务监督，减少灾难性遗忘；

**🔧 技术方法**

视觉-语言模型(Qwen3.5-4B)、BEV感知模块（深度视角变换+BEV Transformer+DETR式检测/UNet分割）、规划专家（基于流匹配的扩散Transformer）以及多任务损失与强化学习策略；

**📊 数据集**

nuScenes、OpenScene（3D感知）；多种公开驾驶VQA与推理数据集（CODA-LM、DRAMA、WaymoQA等）；NAVSIM、WOD-E2E、PAI-AV（轨迹规划）；自建的因果推理与摄像头排序数据；

**📈 对比分析**

与现有BEV基准（BEVFormerV2、PETR）对比，取得mAP 43.95/ND 42.83（nuScenes）/mIoU 60.99/71.27（OpenScene）；在驾驶VQA上超过Qwen3.5-4B平均63.52分，提升至69.43；在轨迹规划上，WOD-E2E RFS 7.78→8.45，NAVSIM PDMS 90.7，AlpaSim安全率显著提升；

**⚠️ 局限性**

1) 规划推理在多因果时序下不稳定；2) 生成轨迹与文本推理不完全一致；3) 三任务输入格式、时序与分辨率差异限制了跨任务表示迁移；4) 需更高质量多视角感知标签以提升跨相机平台迁移性能。

---

## 112. Generalized Tan-Arlery-Rabaste-Lehmann-Ovarlez Lower Bound on Ambiguity Function of a Set of Sequences With Mismatched Filters

**arXiv ID:** 2609.01112 | [PDF](https://arxiv.org/pdf/2609.01112v1)

**作者:** Shibsankar Das `[一作]` `[通讯]` (University of Patliputra), Shibsankar Das (University of Patliputra)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在低模糊区（LAZ）内使用不匹配滤波器的多序列的最大互相干扰函数（AF）下界。

**💡 创新点**

创新点在于将不匹配滤波器与无模长序列集合相结合，并引入延迟与多普勒权重向量，推导出可包含 Welch 与 Tan‑Arlery‑Rabaste‑Lehmann‑Ovarlez 边界的通用 AF 下界。

**🔧 技术方法**

主要技术包括：不匹配滤波器设计、环形矩阵与 Frobenius 范数分析、权重向量优化与 Cauchy-Schwarz 不等式的应用。

**📊 数据集**

无实验数据集，论文为理论推导与数值证明。

**📈 对比分析**

与 Welch 与 Tan‑Arlery‑Rabaste‑Lehmann‑Ovarlez 边界做比较，证明在相应参数极限下可退化为这些经典下界；实验表明在低模糊区内理论下界可被更紧实地满足。

**⚠️ 局限性**

局限性包括：未给出具体的序列与滤波器构造方法，缺乏实际仿真验证，以及对非整数延迟/多普勒区的推广尚未讨论。

---

## 113. A Stable Aggregation Method for Quantum Federated Learning

**arXiv ID:** 2609.00356 | [PDF](https://arxiv.org/pdf/2609.00356v1)

**作者:** Shanika Nanayakkara `[一作]` (Deakin University), Shiva Raj Pokhrel `[通讯]` (Deakin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种自一致性中点聚合（SCM-A2G）方法，用于量子联邦学习（QFL）中的服务器聚合，兼顾客户端质量和参数周期几何；

**💡 创新点**

创新点在于将服务端移动控制与中点自一致性结合，既使用QoS加权的圆形平均来保持周期参数的几何一致，又通过自一致性迭代确保服务器更新在自身中点处仍受支持，显著提升聚合稳定性；

**🔧 技术方法**

主要技术包括QoS感知权重计算、圆形（扭环）平均、基于中点的自一致性迭代（SCM）以及量子电路角度差的周期包装；

**📊 数据集**

实验使用两个跨领域数据集：医学超声乳腺病变（Breast‑Lesions‑USG）和金融欺诈检测（Bank Account Fraud，BAF），并在 IBM 量子硬件上验证；

**📈 对比分析**

与 FedAvg、A2G、MP‑A2G、FEDCOMPASS、FedMRUR 等基线对比，SCM‑A2G 在准确率上与最强基线持平或更优，同时大幅降低更新幅度、晚期波动和验证误差，体现更好的精度‑稳定性‑移动控制平衡；

**⚠️ 局限性**

局限性包括对量子硬件噪声和通信延迟的估计仍需经验参数，且中点自一致性迭代在极端异构或极低 QoS 情况下可能收敛慢，未在大规模客户端场景中充分验证。

---

## 114. MultiGait: A Multi-Sensor Multi-Perspective Multi-Session Biometric Inference Benchmark and its Dataset

**arXiv ID:** 2609.01036 | [PDF](https://arxiv.org/pdf/2609.01036v1)

**作者:** Julian Todt `[一作]` (Karlsruhe Institute of Technology), Thorsten Strufe `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了 MultiGait 数据集，收集了 199 名受试者在实验室中多传感器（视频、深度、NIR、LWIR、激光雷达、毫米波雷达、WiFi CSI/BFI）多角度、三次会话的步态数据，并在此数据集上进行身份、活动、属性推断实验；

**💡 创新点**

创新点在于首次实现了全因子多传感器、多角度、多会话步态数据的同步收集与公开，并对比评估各传感器在单会话和多会话场景下的身份推断风险；

**🔧 技术方法**

技术主要包括：同步多传感器采集系统、基于现有开源算法的多种识别模型（如 GaitBase、DeepGait、LidarGait、mmGaitNet、BFId 等）以及对不同传感器的专门预处理（影像分割、深度扣除、点云裁剪、WiFi 信号提取等）；

**📊 数据集**

使用的数据集为自建的 MultiGait，涵盖 199 人、5 种活动、5 个姿势、4 个视角、3 次会话，共计 32 条信息源；

**📈 对比分析**

通过在单会话和多会话下分别对各传感器和模型进行 5 次重复实验并取平均，发现单会话下几乎所有传感器（除雷达）可达 80%–99% 的身份识别准确率，但多会话下准确率普遍低于 80%，尤其雷达和 WiFi 识别性能几乎无效；

**⚠️ 局限性**

局限性包括：实验室环境限制了真实性能评估；受试者主要为健康成年学生，种族与年龄偏窄；数据集规模虽大但仍不具备完全代表性；所用识别模型未覆盖所有可能方法，且多会话下模型表现受限于现有算法。

---

## 115. Audit-First Rollback Semantics for Safety-Critical Deployment Pipelines

**arXiv ID:** 2609.00406 | [PDF](https://arxiv.org/pdf/2609.00406v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了“Audit-First Rollback Semantics”，在分布式部署运行时中通过预留回滚闭包和显式时间边界，保证在任何局部失败时审计链与实时状态始终保持一致。

**💡 创新点**

核心创新是：①给定正式的可预留状态机模式，明确回滚契约与截止时间；②在任何异常出现时先执行回滚闭包，再写审计记录，使审计链成为失败时的唯一可信源；③提供跨桥协调协议，将单桥方案扩展到舰队级别。

**🔧 技术方法**

技术手段包括：Python 异常捕获与 try/except 包装、可预留状态机抽象、审计链（PersistentAgent.episodic_memory）写入、回滚闭包工厂、per‑capability asyncio.Lock、以及基于 TLA+/P 的规格化设计思路。

**📊 数据集**

使用了机器人臂（Franka Panda）部署管道的实际实现，针对单能力版本升级进行 12 个结构化故障注入点，进行 1,200 次故障注入实验。

**📈 对比分析**

对比方法：在同一流水线中切换至 fail-open 变体（仅记录 FAILED 不执行回滚），在 1,200 次实验中，Audit‑First 方案实现 100%（600/600）审计/实时一致性，p95 恢复延迟 57–335 ms，均低于 500 ms SLO；fail‑open 方案仅 33%（200/600）一致性。

**⚠️ 局限性**

限制与不足：①实验仅在单进程 Python 环境中完成，未验证跨进程/网络部署；②仅覆盖 fail‑stop‑via‑exception 的故障模型，未处理不可捕获信号；③未实现并评估跨桥协议，缺乏舰队级实验；④注入点为手工设计的合成点，未使用真实生产事故分布；⑤对回滚闭包自身异常的处理仍依赖手工调试。

---

## 116. Geometry-aware Latent Autoregressive Generative Model for PDEs in Complex Domains

**arXiv ID:** 2609.00297 | [PDF](https://arxiv.org/pdf/2609.00297v1)

**作者:** Zi Wang `[一作]` (Stanford University), Tapan Mukerji `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对微米尺度复杂几何结构下的多物理PDE，提出了GeoLAMP模型，实现从几何感知的双编码器到潜在空间自回归生成，再到任意点解码的完整流水线，能够在大尺度曲折域内进行高效、长时段的物理场预测。

**💡 创新点**

创新点主要有：①双编码器（全局Farthest-Point采样 + 局部曲率采样）实现对复杂几何的全局与局部特征同时捕捉；②在潜在空间使用流匹配的因果自注意Transformer进行块级自回归，显著降低回归误差扩散；③任意点解码器可在任意分辨率下重构物理场，实现尺度自适应。

**🔧 技术方法**

技术要点包括：图神经算子（GNO）用于几何感知编码与解码；双Encoder（GE+LE）通过FPS和CBS选点；流匹配训练目标配合Transformer实现潜在空间自回归；多尺度、块级预测框架与可变commit步长；以及与传统物理驱动与算子学习方法的对比。

**📊 数据集**

构建了三套基准数据集：①圆盘堆积多孔结构的反应传输流；②随机场生成的热对流；③泡沫结构的弹性；所有数据均使用COMSOL FEM仿真得到，覆盖非线性多物理耦合与不同几何复杂度。

**📈 对比分析**

在这三套数据上，GeoLAMP（尤其是块级B模式）在平均和最终步长的相对L2误差上均优于OFormer、GINO、Transolver、Geo-FNO、NUNO等基线，错误保持在 0.0045–0.0093 之间，长期回归误差增幅几乎无明显增长；与直接MSE回归的确定性控制相比，流匹配显著提升了弹性问题的长时段稳定性。

**⚠️ 局限性**

局限性包括：①在极度复杂的三维几何中潜在空间重构的质量仍有待提升；②块级预测的训练成本随块大小增大而上升；③目前仅在二维结构上验证，三维扩展尚未完成；④对极端参数变化的泛化能力尚未彻底评估。

---

## 117. Artificial Rosetta Stone: Constrained Maximum A Posteriori (MAP) Reconstruction of Symbolic Raga Sequences via Order-k Markov Models

**arXiv ID:** 2609.01064 | [PDF](https://arxiv.org/pdf/2609.01064v1)

**作者:** Saanvi Raghavendran `[一作]` (Abstract Math Institute), Abhishek Bhattacharjee `[通讯]` (Abstract Math Institute)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对印地安音乐的符号片段重建问题，提出并实现了基于有限状态、Dirichlet估计和动态规划的“人工Rosetta石”框架。

**💡 创新点**

明确分离概率模型、语法约束、观测机制与优化算法，证明在有限内存语法下，动态规划能精确求解约束MAP重建，并给出参数计数、复杂度与采样误差界限。

**🔧 技术方法**

使用阶k马尔可夫链、对称Dirichlet先验平滑、硬约束图、动态规划解码、BIC模型选择、实验脚本等技术。

**📊 数据集**

合成六种ragā启发的符号语料（1,200训练+60测试）以及来自RagaVeda公开Yaman音频的30条符号序列（训练24，测试6）。

**📈 对比分析**

在合成数据上，二阶模型在10%缺失下准确率0.539，三阶略低；在真实音频上，二阶模型在10%、30%、50%缺失分别达到0.470、0.414、0.371，均显著高于均匀基准，且比一阶和三阶更优。

**⚠️ 局限性**

仅限于符号表示，未涉及连续音高、节拍、装饰音；语法约束依赖专家定义且可能失效；真实数据使用自动转录，缺乏专家验证和版权；样本量小，未检验历史真实性。

---

## 118. OpenAgentFlow: Enabling System-Wide Safety Boundaries for Heterogeneous AI Agent Fleets

**arXiv ID:** 2609.00015 | [PDF](https://arxiv.org/pdf/2609.00015v1)

**作者:** Dongsheng Chen `[一作]` (Southern University of Science and Technology), Xuetao Wei `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 2250 | [OpenAlex ID](https://openalex.org/A5003379167)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于控制平面/动作平面的共享执行边界治理架构，用于在多模态 AI 代理系统中统一预执行拦截和决策，覆盖 GUI 操作、API 调用、工具调用和 LLM 生成的动作。

**💡 创新点**

创新点：①将异构动作统一为单一事件流并在单一 PEP 处进行预执行检查；②在控制平面维护会话级证明、审计记录和可升级策略，实现策略无缝更新；③采用分阶段（T1–T4）决策流水线，既支持快速可审计规则，也支持语义回退与手工升级；④不需改动代理、提示或执行路径即可实现全局治理。

**🔧 技术方法**

技术手段：事件归一化与序列化；会话级证明采集（GUI 观察、API 结果、PEP 内存）；OpenFlow‑style 控制平面/执行平面分离；T1–T4 分阶段策略评估；动态策略更新接口；Android 端 PEP 嵌入。

**📊 数据集**

数据集：300 条动作事件基准（含攻击/正常）；30 条动态策略测试集；200 条威胁场景集；以及 100 条 Android 模拟器真实执行案例（60 GUI、38 API/LLM 计划）。

**📈 对比分析**

与 Llama Guard、AgentSpec、VeriSafe 等本地边界防护比较：在 300‑case 基准中准确率 94.0%、攻击拦截率 95.3%；T1/T2 阶段覆盖绝大多数可识别风险，平均延迟 < 1 ms；动态策略更新后 90% 规则生效；真实安卓跑分 90.8%（原始）/92.9%（trace‑adjusted）通过率。

**⚠️ 局限性**

局限性：依赖完整证明导致某些被篡改或隐蔽的敏感值难以追踪；数值重用与非标准格式会导致误报/漏报；策略条目需手工定义，缺乏自动生成；在高负载或实时交互场景下的性能仍待进一步评估。

---

## 119. Heard but Not Heeded: Paralinguistic Information Encoding and Loss in Audio-Language Models

**arXiv ID:** 2609.00727 | [PDF](https://arxiv.org/pdf/2609.00727v1)

**作者:** Bhuvan Koduru `[一作]` (Carnegie Mellon University), Bhiksha Raj `[通讯]` (Carnegie Mellon University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

分析四种音频语言模型在捕获和使用声学风格（语调、情绪等）信息方面的机制；

**💡 创新点**

首次将四种互补的评估方法（CKA、线性探测、开放式语调预测、内容‑语调泄漏比）串联起来，系统追踪从音频编码器到输出的风格信息流动；

**🔧 技术方法**

使用中心化核对齐（CKA）、线性探测、互信息泄漏比、梯度反转训练等技术；

**📊 数据集**

基于Expresso数据集，提供相同内容但不同说话风格的音频对；

**📈 对比分析**

比较发现：所有模型在音频编码器后段可高达82‑85%准确率地编码风格，但输出层准确率仅19.7‑53.7%，泄漏比显示训练目标决定模型是“内容驱动”还是“声学驱动”；

**⚠️ 局限性**

局限包括仅评估七种风格、仅使用四个说话人、采用平均池化忽略时间动态、泄漏比受内容熵偏差影响、未探索非线性投影或编码器层级对抗学习。

---

## 120. Replicating TRACE: A Practitioner's Guide to Its Threshold and Particle Budget

**arXiv ID:** 2609.01108 | [PDF](https://arxiv.org/pdf/2609.01108v1)

**作者:** Alex Chadyuk `[一作]` (LotusFlare Inc.), Roy Kucukates `[通讯]` (LotusFlare Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过独立实现并复现了 TRACE 方法，从预训练的自回归语言模型中利用粒子化 do‑sampling 计算条件互信息，阈值化得到因果图，并系统评估阈值和粒子预算在不同词表规模、冗余度与滞后结构下的表现。

**💡 创新点**

创新点包括阈值被证实为真值边际上固定的“margin‑pinned”特性、单一阈值对滞后递减数据的记忆上限、默认生成器的滞后偏置揭示以及基于阈值距离噪声底的粒子预算规则，最终给出面向实务的操作指南。

**🔧 技术方法**

所用技术包括自回归语言模型、粒子化阶梯式干预 (staircase construction) 估计条件互信息、阈值扫描验证、精确的交互式干预真值计算与 F1 评价。

**📊 数据集**

数据集为作者提供的合成事件序列生成器，覆盖词表规模 100–2000、不同冗余度与滞后跨度，所有实验均使用精确的交互式干预真值而非近似方法。

**📈 对比分析**

通过与精确干预真值计算的逐序列 F1 评分比较，复现论文 Table 2 结果，获得 0.90–0.91 的 F1，在不同词表规模下 0.86–0.91，且相较于固定阈值显著提升；粒子预算在阈值距离噪声底足够时可降至 2 颗即可达到饱和。

**⚠️ 局限性**

局限性在于单阈值仅能恢复邻近 (lag‑1) 直接因果关系，对长程因果边的检出受到滞后衰减与生成器偏置的影响；默认生成器的滞后偏置掩盖了真正的性能差异，且粒子预算需依赖阈值与噪声底的距离，缺乏通用固定阈值。

---

## 121. Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources

**arXiv ID:** 2609.00858 | [PDF](https://arxiv.org/pdf/2609.00858v1)

**作者:** Ivan Decostanzi `[一作]` (ISI Foundation), Kyriaki Kalimeri `[通讯]` (ISI Foundation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文构建了一个端到端的管道，利用 EM‑DAT、ReliefWeb 与 EMM 文档，生成源可追溯的灾害情景叙事（storyline）和因果知识图谱。

**💡 创新点**

创新点在于引入多步检索‑生成（Multi‑Shot RAG）实现每个字段与图谱元素的来源引用，并通过自动化的引用验证层为每个知识图谱节点与边生成解释性文本，首次实现了灾害事件的完整可追溯性和儿童敏感影响维度。

**🔧 技术方法**

主要技术包括检索增强生成（RAG）、Meta‑Llama‑3‑70B‑Instruct 语言模型、BGE‑M3 文本嵌入与重排序、以及基于 LLM 的事实验证与文本生成。

**📊 数据集**

使用的数据集为 EM‑DAT 事件记录、欧盟媒体监测（EMM）新闻文章、以及 ReliefWeb 人道主义报告，所有文本通过 BGE‑M3 嵌入后检索。

**📈 对比分析**

通过对三类灾害（公共卫生、自然灾害、冲突）进行人类评估，检索精度达 85.8%，因果三元组 86.7% 基于源文本，专家整体信任度 6.56/10，且多步生成相比单步生成在字段质量与引用可追溯性上显著优于传统方法。

**⚠️ 局限性**

局限性包括因果图谱被评价为低效（仅 2.78/5），缺乏时间脉络和冲突来源的置信度标注，以及情景叙事引用的可靠性较低，导致专家对信息一致性仍持保留态度。

---

## 122. A Glance Is All You Need: Single-Pass Fine-Grained Image Captioning with SimLoss

**arXiv ID:** 2609.00591 | [PDF](https://arxiv.org/pdf/2609.00591v1)

**作者:** Suryaansh Jain `[一作]` (University of Massachusetts Amherst), Seunghyun Yoon `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过对视觉‑语言模型的隐藏状态进行对齐训练，使其在单通道推理时生成更具细粒度细节的图片描述，且不需要人工标注或多阶段推理。

**💡 创新点**

创新点在于提出一种无参考的 embedding‑space 对比损失（SimLoss），将模型隐藏状态与冻结的图像嵌入对齐，从而在训练阶段直接监督模型的视觉保留能力，避免了传统需要精细标注或多阶段后处理的依赖。

**🔧 技术方法**

主要技术包括：InfoNCE 对比学习、LoRA 参数微调、Qwen3‑VL‑Embed 冻结嵌入模型、Qwen2.5‑VL‑7B 视觉‑语言基模型，以及若干变体（SimLoss FFT、SimLoss GRPO）。

**📊 数据集**

训练数据使用 MS‑COCO 图像（不使用其 Caption），评估数据使用 IIW‑400 超细粒度描述集，评估指标包括 CLAIR、精确率（Precision）、召回率（Recall）与 F1。

**📈 对比分析**

与 CapMAS（多阶段验证管道）、PAPO、FeedQuill 等方法对比，SimLoss FFT 在单通道推理下达到 0.8485 的最高精确率、0.7023 的 F1（与 CapMAS 0.7025 仅差 0.0002），并且推理时间约 5.8 秒/图，速度提升约 20×；SimLoss GRPO 则在召回率和 CLAIR 上表现最佳。

**⚠️ 局限性**

局限性包括：与 CapMAS 的 F1 差距仍然存在，尤其在极端细粒度情境下可能出现误判；SimLoss FFT 对冻结嵌入模型的偏差敏感，若教师嵌入缺乏某些细节可能导致细粒度不足；GRPO 由于仅依赖奖励，精确率略低，需进一步融合多种监督策略。

---

## 123. Long-Horizon State Tracking in LLMs: Executing MD5 through a Deep Sequence of Dependent Tool Calls

**arXiv ID:** 2609.00012 | [PDF](https://arxiv.org/pdf/2609.00012v1)

**作者:** Dheeraj Mohandas Pai `[一作]` (Leanmcp.com), Lu Xian `[通讯]` (Leanmcp.com)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过让LLM按顺序调用工具，逐步计算单块MD5哈希，验证LLM在长时间步长中是否能准确传递并维护中间状态。

**💡 创新点**

创新点在于提出一个可测量、可比对的基准（MD5 196步完整工具调用序列），并证明仅凭语境管理与自一致性投票即可让LLM完成长程、精确计算；同时揭示了语境思考通道与投票机制是实现此目标的关键。

**🔧 技术方法**

采用的技术包括：OpenAI Harmony格式中的思考通道（reasoning channel）保持上下文、对每个算术工具使用三个采样的LLM工作者进行多数投票（self-consistency），以及将MD5实现从原始CPU代码切换为LLM自计算的工件。

**📊 数据集**

使用的数据集为RFC 1321标准MD5参考实现及其测试向量，实验中随机生成的非记忆化单块输入字符串（≤55字节）以保证每次计算必须从零开始。

**📈 对比分析**

比较方法是统计最终哈希是否与参考完全一致（success rate）、首次偏差所在轮次以及工具调用数与标准196次的偏差；实验结果显示，在保持思考通道和投票机制的前提下，约70%+ 的运行能得到正确的32位十六进制哈希，错误主要来自驱动程序的状态转移错误；投票可将算术错误压至 <1%。

**⚠️ 局限性**

局限性包括：依赖于思考通道的支持（部分API不允许回传此通道导致失效）；仅在单块MD5任务中验证，尚未证明可扩展至多块或更复杂的状态；以及剩余的驱动器状态错误（如40→48跳过）表明模型在长程状态维护上仍有系统性缺陷。

---

## 124. SAGE: Subpopulation-Aware Generative Enhancement for Mitigating Spurious Correlations

**arXiv ID:** 2609.01051 | [PDF](https://arxiv.org/pdf/2609.01051v1)

**作者:** Yiming Luo `[一作]` (Harbin Institute of Technology), Jie Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SAGE框架，利用无标签聚类得到子标签，对Stable Diffusion进行LoRA微调，生成针对欠代表子群的合成训练集与验证集，以提升无组标签环境下的鲁棒性。

**💡 创新点**

1) 用聚类获得子标签作为自洽的群组代理；2) 在生成器中引入可学习子标签token并结合逆密度采样与均匀采样实现有针对性的合成；3) 在不需要真实组标签或手工验证集的情况下，完成训练与验证集的生成与重权重。

**🔧 技术方法**

Stable Diffusion + LoRA微调、Affinity Propagation聚类、CLIP语义嵌入、逆密度采样、Deep Feature Reweighting (DFR)、ResNet-50分类器。

**📊 数据集**

Waterbirds、CelebA、MetaShift。

**📈 对比分析**

与无组标签基线和已有生成/非生成方法对比，在Waterbirds、CelebA、MetaShift上取得最坏组准确率89.5%、85.7%、79.1%，分别比最佳无组标签基线高0.8%、7.7%、5.6个百分点，整体性能领先。

**⚠️ 局限性**

依赖聚类结果与子标签质量；生成模型可能产生偏差或重复；对极端多模态或高分辨率场景的可扩展性有限；在某些数据集（如Waterbirds）手工验证集仍优于合成验证集；计算成本较高。

---

## 125. TUTTI: Toward generalizable audio-to-score transcription via fully synthesized data

**arXiv ID:** 2609.00640 | [PDF](https://arxiv.org/pdf/2609.00640v1)

**作者:** Jianhuai Hu `[一作]` (Central Conservatory of Music), Maosong Sun `[通讯]` (Central Conservatory of Music)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个纯 Transformer 的音频到谱记（A2S）转写框架 TUTTI，并通过生成的 363,610 份多乐器合成音频-谱记对进行大规模预训练，随后在真实数据上微调，取得了多乐器转写任务的最新效果。

**💡 创新点**

创新点包括：① 通过符号音乐生成模型生成海量多乐器谱记数据，突破了真实数据稀缺瓶颈；② 采用纯 Transformer 编码器-解码器结构，摒弃传统 CNN 前端，证明数据规模可弥补局部先验；③ 设计层次化 ABC 注释与解码策略，兼顾全局结构与局部音符细节；④ 展示了对未出现乐器（如萨克斯）具有出色迁移能力。

**🔧 技术方法**

核心技术：Transformer encoder-decoder、层次化 ABC 结构、Expressive Performance Rendering (EPR)、DualDTW 对齐、基于 sfizz 的高保真音频合成、以及标准化的 ABC 预处理。

**📊 数据集**

数据集：TUTTI 数据集，包含 363,610 条人工合成的多乐器音频-谱记对，覆盖从单音轨到大型合奏的多种配置，并通过相似度审核保证与测试集无重叠。

**📈 对比分析**

与多项基准（如 Zeng、Alfaro‑Contreras、Martínez‑Sevilla 等）进行 MV2H 指标对比，TUTTI 在 ASAP、Quartet、Saxophone 等数据集上均优于所有基线，尤其在多音轨任务中的音高、和声和音符时值准确率接近 100%，并在未见乐器上实现理论极限声部准确率。

**⚠️ 局限性**

局限性：仍以短段（≤14.8 s）为输入，无法一次性处理整首曲目；合成数据虽然多样但可能缺少人类演奏的细腻表达；对极端音色或非传统乐器的泛化仍有待进一步验证。

---

## 126. Context Window Failures in Relational Foundation Models

**arXiv ID:** 2609.00460 | [PDF](https://arxiv.org/pdf/2609.00460v1)

**作者:** Denis Oliveira Correa `[一作]` (Kunumi Institute), Francisco Galuppo Azevedo `[通讯]` (Kunumi Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个名为Animus的合成金融基准数据集，旨在检验关系深度学习模型在高基数情境下的邻域预算限制，并通过对收入预测任务的实验评估现有模型的性能。

**💡 创新点**

创新点在于：1）设计了能够生成数万笔交易记录的高基数客户样本，真实模拟行业中高频客户的挑战；2）通过对原始事件表与简单聚合表两种表示进行对比，揭示了邻域预算对模型预测的显著影响；3）提供了对不同模型（RT、Griffin、RelGT、GraphSAGE）的细粒度分箱分析，具体定位预算限制导致的性能下降。

**🔧 技术方法**

使用的技术主要包括：关系图模型（GraphSAGE）、关系变压器（RT）、Griffin图网络、Relational Graph Transformer（RelGT）；实验采用了超参数网格搜索、R²、MAE、RMSE评估指标，并在单张NVIDIA H200 GPU上运行。

**📊 数据集**

使用的数据集为Animus，一个由100,000名客户及其12个月交易、借记和支付记录组成的合成金融数据库，交易频率在1笔/月到10,000笔/月之间变化。

**📈 对比分析**

对比方法为在原始事件表与聚合表两种数据表示上分别训练和评估四个模型。结果显示：在原始表示下，RT、Griffin、RelGT的R²≤0.18，GraphSAGE达0.69；在聚合表示下，RT提升到0.65（+0.47），Griffin提升到0.38（+0.27），GraphSAGE仅提升到0.75，表明聚合能显著提升预算受限模型的性能。

**⚠️ 局限性**

局限性包括：1）仅关注单一收入预测任务，未检验其他业务场景；2）聚合表示仍需人工预处理，未展示模型自行学习聚合的能力；3）RelGT在高基数下因内存不足无法完整构建图，说明目前框架在极端基数下的可扩展性不足；4）实验仅在单GPU上完成，缺乏大规模分布式验证。

---

## 127. Empirical Software Engineering in Practice: Insights from Google

**arXiv ID:** 2609.00247 | [PDF](https://arxiv.org/pdf/2609.00247v1)

**作者:** Roberto Verdecchia `[一作]` (University of Florence), Justus Bogner `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

介绍了谷歌内部开发者体验团队如何运用多元化经验软件工程（ESE）方法来识别和解决软件工程师的痛点，并将研究结果转化为工具改进与决策支持。

**💡 创新点**

创新点在于：①强调多方法（定量+定性）混合研究与三角测量的必要性；②将学术方法与工业实际结合，形成快速、可操作的反馈循环；③倡导在工业环境中出版负面结果与完整论文，避免“salami slicing”。

**🔧 技术方法**

采用了日志分析、调查问卷、访谈、民族志、A/B实验、倾向匹配、定量-定性融合的分析框架，以及对人机交互与团队动力的心理学测量。

**📊 数据集**

主要数据来源为：公司内部日志（代码提交、审查、部署等）、季度调查数据、用户体验访谈记录、实验组/对照组的运行日志、以及跨团队共享的公开数据表和脚本。

**📈 对比分析**

通过三角测量（将同一研究问题的定量结果与定性发现对比）来评估方法可靠性；在实验中使用双盲、A/B实验等方式验证干预效果；但缺乏统一的代码质量或整体软件开发速率指标，导致对大规模影响评估仍不完整。

**⚠️ 局限性**

主要限制包括：难以在大规模数据中构造可解释且可靠的指标（如代码质量、工作流流畅度、项目整体速度）；实验与观察研究受限于工程师工作负载和工具可用性；跨学科方法导致沟通成本高，难以在短期内实现高效迭代。

---

## 128. Solving the Incompressible Navier-Stokes Equations on Oriented Curved Surfaces Discretized by Point Clouds

**arXiv ID:** 2609.00216 | [PDF](https://arxiv.org/pdf/2609.00216v1)

**作者:** Alejandra Foggia `[一作]` (Dresden University of Technology), Ivo F. Sbalzarini `[通讯]` (Dresden University of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

开发了一种基于面点云的无网格求解器，用于在有向曲面上显式求解不可压Navier–Stokes方程；

**💡 创新点**

将表面DC‑PSE（纠正的粒子强度交换）与EDAC（熵阻尼人工压缩性）相结合，实现了高阶（最高六阶）空间离散且无需全局矩阵求逆的显式方法；

**🔧 技术方法**

使用Surface DC‑PSE构造表面微分算子，利用EDAC处理不可压约束，时间积分采用RK4；实现基于OpenFPM框架的并行C++代码；

**📊 数据集**

采用自定义的曲面点云（单位球、环面、十次曲面和手绘非参数“花生”曲面），每个点云均附带法向量；

**📈 对比分析**

与现有基于表面有限元、离散外积、光滑粒子方法等做比较，证明了在单位球、环面等标准测试上的二阶到四阶收敛；在非参数曲面上表现出良好的几何鲁棒性，且计算成本与点数线性相关，可在多核CPU上实现高效并行；

**⚠️ 局限性**

仅适用于具有非相交管道邻域的有向曲面，需均匀点密度，无法处理共轭维度>1或大曲率变化导致的点云稠密问题；低Mach数时易出现人工振荡；高Re时显式时间步受CFL限制，导致计算开销较大。

---

## 129. Revisiting Face Recognition for Monozygotic Twins: The Celeb Twins Test Set

**arXiv ID:** 2609.01141 | [PDF](https://arxiv.org/pdf/2609.01141v1)

**作者:** Michael Zang `[一作]` (University of Notre Dame), Kevin W. Bowyer `[通讯]` (University of Notre Dame)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Celeb Twins Test Set (CTTS-80) 并用其评估了现有深度 CNN 匹配器在 monozygotic 双胞胎识别上的性能。

**💡 创新点**

创新点在于公开了一个大规模、带皮肤痣和镜像非对称性元数据的双胞胎测试集，并系统研究了现有匹配器是否利用皮肤痣或镜像信息。

**🔧 技术方法**

主要技术包括现有深度 CNN 匹配器（AdaFace、ArcFace、UniFace、MagFace）、皮肤痣去除实验和去掉水平翻转训练的对称性感知模型。

**📊 数据集**

使用的数据集为 CTTS-80（21,120 对图像）与 ND‑Twins、LFW、CALFW、CPLFW、CFP‑FP、AgeDB‑30 等传统测试集进行对比。

**📈 对比分析**

比较方法为 10 折交叉验证平均准确率；结果显示现有匹配器在 CTTS 上准确率仅 70–80%，低于传统测试集但显著高于 60%，且未能有效利用皮肤痣或镜像非对称性。

**⚠️ 局限性**

限制包括训练数据中 MZ 双胞胎比例极低、缺乏高分辨率痣检测，生成式 AI 目前无法产生具有真实差异的虚拟双胞胎图像。

---

## 130. Emotional Labor Strategy Preferences in LLM Personas

**arXiv ID:** 2609.00310 | [PDF](https://arxiv.org/pdf/2609.00310v1)

**作者:** Mohammad Saim `[一作]` (University of Cincinnati), Tianyu Jiang `[通讯]` (University of Cincinnati)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了将人格特质注入大型语言模型（LLM）后，在日常社交情境中选择情绪劳动（EL）策略的行为模式。

**💡 创新点**

创新点在于首次将人格与EL策略选择结合，构建并公开了首个非职业情境的情绪劳动策略数据集，并通过双轨人格注入验证模型人格化对策略选择的影响。

**🔧 技术方法**

采用人格化注入技术（BAP观测式与IPIP-50自评式两条轨道），在五款LLM上评估其在EL策略选择任务中的表现，并使用熵分析、Spearman相关性等统计手段对结果进行量化。

**📊 数据集**

使用了500个日常情绪情境的EL策略数据集（来源于情绪评估语料库并添加社交上下文与三种策略选项），以及50个虚构角色的人格资料（来自Open Psychometrics的双极形容词配对与IPIP-50自评）。

**📈 对比分析**

通过比较不同模型与人格轨道下的策略分布、与五大人格维度的相关性以及熵值，发现模型普遍倾向深度表现（Deep Acting），人格注入能够显著改变策略分布并与人类研究中已知的特质-策略关联相一致。

**⚠️ 局限性**

局限性包括仅使用虚构角色的人格感知数据、合成的情境与策略、仅英语言环境、结果为相关性而非因果性，且缺乏跨文化与机制层面的验证。

---

## 131. Framework and Benchmark for Code-Driven Agentic Testing in Web Development

**arXiv ID:** 2609.00081 | [PDF](https://arxiv.org/pdf/2609.00081v1)

**作者:** Bin Hong `[一作]` (University of Science and Technology of China), Zhenya Huang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 5117 | [OpenAlex ID](https://openalex.org/A5085496384)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于代码驱动的Agentic Testing（CAT）范式，构建了统一的CATJudge框架，并通过人机协作生成了102个含细粒度bug的CATTest基准，用于评估视觉语言模型在AI生成网页的E2E GUI测试能力。

**💡 创新点**

创新点在于将Agent直接编写Playwright脚本进行浏览器交互、统一了Browser-Use与Computer-Use工具，并通过细粒度bug标注与细致评测指标揭示模型的bug发现能力。

**🔧 技术方法**

利用Playwright、PyAutoGUI、a11y tree、Docker/Xvfb等工具实现Agent操作，采用POMDP建模和自适应工具调用，并在Claude、Gemini、GPT-5.x等LLM/视觉语言模型上进行推理。

**📊 数据集**

使用CATTest基准共102个AI生成的网页项目，包含人工与模型共同标注的真Bug集；同时借助Claude Code生成代码。

**📈 对比分析**

对比多款主流VLM在CATJudge框架下的R-score、P-score、Acc等指标，最高R-score仅42.57，表明现有模型在bug发现上仍远低于实用水平，存在明显的性能鸿沟。

**⚠️ 局限性**

限制包括GT bug集可能不完整、评测侧重功能正确性忽略主观体验、模型误报导致P-score偏低、实验受限于工具选择和硬件环境。

---

## 132. Patterning in Practice: Debiasing Reward Models with Susceptibilities

**arXiv ID:** 2609.00699 | [PDF](https://arxiv.org/pdf/2609.00699v1)

**作者:** George Wang `[一作]` (Resolution), Daniel Murfet `[通讯]` (Resolution)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用模式化（Patterning）技术，对Gemma 2 9B奖励模型的偏好训练数据进行重新加权，以消除长度、格式等表面特征带来的偏差。

**💡 创新点**

首次将模式化从小型合成任务扩展到大规模奖励模型的实际应用，并证明其权重在不同规模模型间具有良好迁移性；同时提供了针对“安全响应”下降的诊断与修正示例。

**🔧 技术方法**

使用基于奇异学习理论的敏感度（susceptibility）估计，构造目标偏移后求伪逆得到每条训练样本的权重；通过SGLD采样估计后验协方差；对训练集重新加权后重新训练奖励模型。

**📊 数据集**

Skywork‑Reward‑Preference v0.2（约7.5万对）作为偏好数据；RM‑Bench 用作观测指标；RewardBench 2 用作验证基准。

**📈 对比分析**

与现有最强方法 SteerRM 进行对比；在RM‑Bench Hard分割上获得+14.2±1.2个百分点提升（相当于或超过SteerRM的+13.2）；整体准确率保持不变；在RewardBench 2 上仅降幅1.4个百分点；权重还能在Gemma 2 2B/27B 甚至部分迁移至 Llama‑3.1‑8B，显示出跨规模/跨架构的可迁移性。

**⚠️ 局限性**

主要局限包括：① 计算成本高，需数千GPU‑小时的SGLD后验采样；② 采用线性响应理论但实际调节强度已进入非线性 regime，理论对结果的解释有限；③ 后验估计不确定，受SGLD步长与β选择影响；④ 仅针对长度/格式偏差，难以直接扩展到更复杂的对齐问题。

---

## 133. The Curse of Multilinguality in Lexical Normalization

**arXiv ID:** 2609.00329 | [PDF](https://arxiv.org/pdf/2609.00329v1)

**作者:** Saman Rahbar `[一作]` `[通讯]`, Saman Rahbar

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在固定容量字符级Transformer下，词汇规范化任务中多语言训练的最优语言数量。

**💡 创新点**

发现“多语言化诅咒”：每语言准确率在训练少数几种语言时最高，随着加入更多语言持续下降，且此下降不是由数据量导致，而是容量共享造成的；且语言的类型学距离并不能可靠预测其最佳合作语言数。

**🔧 技术方法**

使用单一 1.49M 参数的字符级 Encoder–Decoder Transformer，采用 AdamW 训练，固定模型尺寸从 1 语言到 12 语言进行实验。

**📊 数据集**

使用 MultiLexNorm 基准的 12 种语言（丹麦语、德语、英语、西班牙语、克罗地亚语、印尼语–英语、意大利语、荷兰语、斯洛文尼亚语、塞尔维亚语、土耳其语、土耳其语–德语）数据集。

**📈 对比分析**

通过 ERR（Error Reduction Rate）衡量，平均 ERR 在 2 语言时最高（≈0.316），在 12 语言时下降到约 0.191（下降约 40%），说明多语言训练并未提升整体性能。

**⚠️ 局限性**

局限包括：仅测试单一小模型大小，未探究更大或预训练模型；未区分容量下降的具体机制（参数稀释、数据稀释、干扰）；仅对 12 种语言做子集平均，未分析具体语言配对；部分语言（西班牙语、意大利语）在任何配置下表现低于留空；类型学分析样本受限，未得到可靠结论。

---

## 134. EM^2Mem: Event-Centric Multimodal Memory for Large Language Models

**arXiv ID:** 2609.00551 | [PDF](https://arxiv.org/pdf/2609.00551v1)

**作者:** Yijun Chen `[一作]` (Zhejiang University), Shumin Deng `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了 EM2Mem，一种基于事件锚点的多模态记忆框架，用于长视频问答任务，先将视频划分为事件并在构建阶段对多模态证据进行对齐，随后通过轻量级检索直接获取完整的事件级证据供 LLM 生成答案。

**💡 创新点**

创新点在于：①将多模态证据统一绑定到事件锚点，形成事件级记忆单元；②采用“先对齐后检索”策略，构建时完成跨模态与时序对齐；③构建轻量化的情节图与语义图，提供跨事件与长期知识的检索支持；④实现了显著提升的准确性、证据定位精度以及推理效率。

**🔧 技术方法**

核心技术包括：事件划分与锚点索引、视觉关键帧与字幕的多模态解析（captioning、transcript alignment、keyframe selection）、结构化元数据抽取、多尺度时间上下文视图、情节图与语义图的构建、事件级检索与扩展、LLM 选取器与答案生成。

**📊 数据集**

使用的公开长视频 QA 数据集为 EgoLifeQA、Ego‑R1 Bench 与 Video‑MME (L)。

**📈 对比分析**

与多类基线（基础 MLLM、长视频 MLLM、RAG、基于记忆的推理框架）进行对比，EM2Mem 在三大基准上平均提升 2.0%、2.4% 和 3.7% 的准确率，单次查询延迟缩短 4.67 倍，推理 token 数量减少 63.66%，同时严格事件级 Top‑5 证据召回率提升 7.0 点。

**⚠️ 局限性**

局限性包括：结构化记忆对视觉细节（小物体、颜色、布局等）的表达有限，仍需在推理阶段使用关键帧做视觉校验；构建阶段计算量大，适合一次性处理而非实时应用；依赖上游视觉与 NLP 工具，错误会被放大；以及在隐私敏感视频上应用时存在信息泄露与行为画像风险。

---

## 135. VPID: An Integrated Framework for Vulnerability Prioritization and Intrusion Detection in Enterprise Networks

**arXiv ID:** 2609.00819 | [PDF](https://arxiv.org/pdf/2609.00819v1)

**作者:** Xuanren Chen `[一作]` (Hainan University), Xiaoqi Li `[通讯]` (Hainan University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了 VPID 框架，将漏洞扫描、上下文感知优先级排序、受控验证、基于机器学习的双阶段检测与 Snort 规则验证以及 iptables 自动响应集成在一个轻量级、可解释的五层架构中，专为资源受限的小型企业网络而生。

**💡 创新点**

创新点在于：① 将漏洞管理与入侵检测通过统一流程无缝衔接；② 采用决策树与朴素贝叶斯的双阶段检测，既保留低成本特征过滤，又通过文本特征提升检测率；③ 将 Snort 规则与机器学习输出融合，构建可解释的风险评分与自动化响应；④ 通过 Docker 部署实现可扩展的低资源部署。

**🔧 技术方法**

使用技术包括 OpenVAS、Scapy、Decision Tree (CART)、Multinomial Naive Bayes、Snort、iptables、Docker、MySQL、Python 3.9 及 scikit‑learn。

**📊 数据集**

使用的数据集为：15,000 条标注漏洞记录；550,000 条混合网络流样本（CICIDS2017、NSL‑KDD、UNSW‑NB15 及本地流量）；独立 55,000 条测试样本。

**📈 对比分析**

通过与随机森林、SVM、单独 Snort 规则等基线对比；漏洞优先级排序决策树在 15,000 条样本上达到 91.8% 精度、89.5% 召回、90.6% F1；攻击检测采用双阶段模型在 55,000 条测试集上实现 94.5% 精度、88.3% 召回、91.3% F1，误报率仅 1.2%，且在两核 4GB 主机上 CPU <15% 的资源占用。

**⚠️ 局限性**

局限性包括：无法完整分析 HTTPS 流量；依赖 OpenVAS 的测试集，可能缺失未覆盖的漏洞；不同来源数据融合后分布可能偏移，影响泛化；对抗性修改的流量可能绕过检测；以及单机集中部署在大规模流量场景下可能出现瓶颈。

---

## 136. Human-AI Co-Interpretation for Responsible AI: A Hermeneutic Perspective

**arXiv ID:** 2609.00334 | [PDF](https://arxiv.org/pdf/2609.00334v1)

**作者:** Behrooz Razeghi `[一作]` `[通讯]` (Harvard University), Behrooz Razeghi (Harvard University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了“人机共同诠释循环”与“解释性安全环”框架，阐明了在LLM介入的解释性工作中应如何通过人工与人工智能的互动保证解释的多样性、可追溯性和责任可追溯性。

**💡 创新点**

创新点在于将传统人文解释学的核心概念（如视界融合、双重解释）与LLM技术结合，形成可落地的设计语言、评估维度和安全流程；同时首次系统性地提出“解释性安全循环”以应对LLM在解释任务中的幻觉与误导。

**🔧 技术方法**

主要采用概念建模、设计模式与安全流程工程的方法；技术层面未实现新模型或算法，而是聚焦于Prompt工程、检索增强、可解释性注释等现有LLM功能的组合与配置。

**📊 数据集**

无专门数据集；论文以案例与现有文献为基础进行理论分析和设计讨论。

**📈 对比分析**

本工作为概念性与设计性研究，未进行实验或性能评测；因此无可比指标。

**⚠️ 局限性**

局限性在于缺乏实证验证与用户研究，难以评估所提安全流程和设计模式在真实工作环境中的效果与可操作性；此外，对LLM在不同领域的适用性与可扩展性的细致分析仍待进一步探索。

---

## 137. REAL-Q: E2E LLM Quantization via Dynamic Gradient Descent

**arXiv ID:** 2609.00049 | [PDF](https://arxiv.org/pdf/2609.00049v1)

**作者:** Qian Zhang `[一作]` (Peking University), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5965 | [OpenAlex ID](https://openalex.org/A5069277955)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的后训练量化方法 REAL-Q，利用全局 Fisher MSE 作为损失，并在每个列块后加入 Adam 梯度更新以动态纠正误差；通过滑动窗口平滑跨块目标，降低信息失配。

**💡 创新点**

核心创新在于：① 将端到端 KL 散度的二阶展开转化为跨通道耦合的聚合 Fisher MSE，保持高保真度；② 引入细粒度、动态的 Block‑GD 步骤，打破传统静态 Hessian 近似；③ 采用滑动窗口机制在连续 transformer 块之间平滑目标。

**🔧 技术方法**

使用的技术包括：第二阶 Taylor 展开、聚合 Fisher 信息、Adam 优化器、滑动窗口交叉块梯度、QuaRot 旋转预处理，以及多种量化策略（W4A16、W3A16、W2A16、WxA4KV4 等）。

**📊 数据集**

实验数据集为 WikiText‑2 用于校准与评估，另外使用十项零样本任务评估下游性能。模型包括 LLaMA‑3.1‑8B/70B、Qwen3‑0.6B/1.7B/4B/8B/32B。

**📈 对比分析**

与 RTN、GPTQ、GTAQ、GuidedQuant 等基线对比，REAL-Q 在 W4A16 下的 KL 散度平均下降约 30–50%，在 Qwen3‑1.7B 上最高降幅近 70%，并在大多数下游任务保持或提升精度，证明其在精度与可扩展性上的优势。

**⚠️ 局限性**

主要局限包括：校准阶段需要额外的反向梯度计算，导致内存和计算开销增加；目前仅验证了权重‑仅和权重‑激活两种量化配置，对极低位（如 W2A4KV4）性能波动仍需进一步优化；理论分析基于线性结构漂移假设，实际场景中误差传播机制更为复杂。

---

## 138. Towards Generalizable Visually Grounded Exploration of Household Devices

**arXiv ID:** 2609.00845 | [PDF](https://arxiv.org/pdf/2609.00845v1)

**作者:** Linhao Zheng `[一作]` (Beijing Institute of Technology), Yuhang Guo `[通讯]` (Beijing Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了VGEBench基准，用于评估视觉-语言模型在无手册情况下对家庭设备进行通用视觉引导探索的能力。

**💡 创新点**

创新点包括引入基于逻辑驱动的状态机框架，模拟多轮交互反馈；聚焦细粒度视觉定位与探索；设计了大规模设备、视图、交互任务数据集，并提出多维度评估指标。

**🔧 技术方法**

采用3D渲染与人工标注得到设备视图与交互组件，构建通用与专属状态机；使用LLM（GPT-5-mini）生成自然语言指令；在ReAct框架下对VLM进行推理；通过多种评估指标量化视觉定位、交互效率与任务成功率。

**📊 数据集**

使用自建的VGEBench数据集：968个家庭设备（26类），7,888个高分辨率视图，3,712个交互组件，7,264个边界框，共计14,953个任务（4,948单轮，10,005多轮）。

**📈 对比分析**

将Gemini‑3‑Flash、Doubao‑1.5‑Thinking‑Vision‑Pro、GPT‑5‑mini、Qwen3‑VL‑8B、InternVL3.5‑8B、Mimo‑Embodied‑7B等VLM在VGEBench上进行对比，使用成功率、子任务成功率、SPL、状态F1、有效交互率、目标交互率、视图切换指标等九项指标；Gemini‑3‑Flash最高SR 54.27%，SSR 62.86%，但仍低于人类基线（74.5%）。

**⚠️ 局限性**

局限性在于评估需大量计算资源与高分辨率输入；离散视角与动作抽象限制了真实感；不包含真实环境中的光照、遮挡与背景，难以完整复现物理机器人部署的 sim‑to‑real 问题。

---

## 139. OUTLETS: Output-Length Prediction from Speculative Decoding Backbones

**arXiv ID:** 2609.01068 | [PDF](https://arxiv.org/pdf/2609.01068v1)

**作者:** Weihuang Wen `[一作]` (Chinese University of Hong Kong), Tianshu Yu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了OUTLETS框架，将大型语言模型（LLM）中的推理草稿（draft）网络重新用于输出长度预测，既实现了长度预测又不增加显著开销。

**💡 创新点**

创新点在于发现并利用了推理草稿网络（Speculative Decoding Backbones）中隐层表征中蕴含的生成长度信息，并通过双头（Draft和Length Regression）结构在同一计算图上联合训练，显著提升预测精度。

**🔧 技术方法**

采用Transformer型草稿解码器、层级特征融合、轻量级回归头和联合损失（Speculative Decoding + Log‑space长度回归）实现；在此基础上与传统 MLP、BERT、OPT、PIA 等预测方法对比。

**📊 数据集**

使用公开对话/指令数据集：ShareGPT、Alpaca、LMSYS‑Chat‑1M 以及 Qwen3‑30B‑A3B 的子样本进行训练与评估。

**📈 对比分析**

与多种基准（内部 MLP、代理模型、提示式预测）对比，OUTLETS 在静态预测 MAE 方面优于代理和提示式方法，动态预测亦显著低于单纯 MLP；在分布式服务系统中，基于 OUTLETS 的负载均衡+最短作业优先策略将短任务 P99 延迟降低 34.8% 并提升吞吐量。

**⚠️ 局限性**

局限包括：仅在已使用推理草稿的场景下成本极低，单独做长度预测时仍需额外草稿网络；当前仅利用静态预测做排队调度，未实现动态在线迁移；对极长序列、不同采样温度或自定义终止条件的预测准确性未验证。

---

## 140. ZimaBlue: Evolving Generalizable World Action Models through Scalable Video Pre-training

**arXiv ID:** 2609.00188 | [PDF](https://arxiv.org/pdf/2609.00188v1)

**作者:** Xionghao Wu `[一作]` (Joy Future Academy), Nan Duan `[通讯]` (Joy Future Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可扩展的世界动作模型（ZimaBlue）框架，通过先对120k小时第一人称视频进行无标签因果预训练，再利用跨身体动作对齐实现物理与因果先验的迁移，最后对目标机器人进行微调，实现在零-shot条件下的高成功率机器人操控。

**💡 创新点**

创新点包括：①将无标注第一人称视频作为可扩展的物理与因果先验源；②三阶段训练（视频预训练→跨身体视频-动作对齐→目标机器人微调）配合统一的100维动作表示；③异步慢快双系统（Slow–Fast）实现30Hz低延迟控制；④使用分布匹配蒸馏（DMD）将高阶推理压缩至两步推理，显著降低推理时延。

**🔧 技术方法**

核心技术：视频-动作联合扩散Transformer（DiT）预测视频潜在与动作；分块因果注意、KV 缓存交互；统一状态-动作表述；Flow‑matching 目标；慢分支学习长时序世界动态，快分支学习高频动作输出；异步推理、RT‑Chunking（RTC）与 DMD 蒸馏实现实时推理。

**📊 数据集**

数据集：120k小时 egocentric 人类视频（EPIC‑KITCHENS、Egocentric‑100K、DreamDojo等）、跨身体机器人轨迹（DROID、AgiBot、Galaxea、RoboMIND2）、模拟数据（DreamDojo、RoboCOIN等）；零-shot实地任务12个；公开基准 LIBERO‑Plus、RoboTwin‑2.0、RoboCasa365。

**📈 对比分析**

与 π_0.5、DreamZero、LingBot‑VA、Fast‑WAM 等基线对比，ZimaBlue 在12个零-shot实地任务上从 36.1% 提升至 77.8%（标准 87.9%、扰动 57.5%）；在 LIBERO‑Plus 零-shot 86.7%→SFT 92.0%；在 RoboTwin 94.5%；在 RoboCasa 49.5%；推理延迟从 449 ms 降至 33 ms（30 Hz）。

**⚠️ 局限性**

限制：仍需大规模视频与跨身体数据；对极端光照、背景或视角变化的鲁棒性不足；模型规模庞大，部署成本高；缺乏自适应的高层规划与动态修正；对稀疏或极端任务的迁移能力有限。

---

## 141. HyperSketch: Controllable Video Sketching in a Style Hyperspace

**arXiv ID:** 2609.00919 | [PDF](https://arxiv.org/pdf/2609.00919v1)

**作者:** Xinding Zhu `[一作]` (Zhejiang University of Technology), Jiazhou Chen `[通讯]` (Zhejiang University of Technology)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于四维Bernstein多项式的可控视频向量素描动画生成方法，可在时间、保真度、简化度与文本引导强度四个维度实现连续、可分离的风格控制。

**💡 创新点**

创新点在于构建连续四维风格超空间，并用多变量Bernstein多项式对笔画控制点进行参数化，使得风格演变在时间轴上平滑且可交互；此外采用多任务联合优化与分阶段维度增量策略，显著提升训练稳定性和控制解耦。

**🔧 技术方法**

使用多变量Bernstein多项式、DiffVG可微渲染、CLIP语义与文本特征、CoTracker点轨迹、以及自定义的多任务损失（语义、墨水密度、文本引导、时间一致性、跨风格一致性）等技术。

**📊 数据集**

在DAVIS视频数据集上进行训练与评估，并通过多维样本采样（α×β×γ×δ）验证模型的通用性。

**📈 对比分析**

与UPDG、CLIPasso、SketchVideo、DMTSketch、CLIPascene等基准方法比较，在CLIP误差、LPIPS误差及文本一致性指标上均优于SOTA；实验显示能够实现更高质量、更多样化且可实时调整的素描动画。

**⚠️ 局限性**

局限在于对CoTracker跟踪的依赖，快速运动或遮挡时可能失效；训练过程中对DiffVG的单点渲染导致计算开销大；虽然已实现三维风格，但扩展到更高维度仍有挑战。

---

## 142. ContextPipe: Database-Inspired Context Assembly for Long-Horizon Agents

**arXiv ID:** 2609.00749 | [PDF](https://arxiv.org/pdf/2609.00749v1)

**作者:** Peng Xu `[一作]` (MatrixOrigin), Chen Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一个五阶段（Plan→Bind→Optimize→Execute→Feedback）的数据库式上下文组装管道，自动决定 LLM 代理每轮请求中应包含哪些内容、如何压缩、缓存标记及顺序，并支持审计、重放与故障隔离。

**💡 创新点**

将上下文组装视为查询执行问题，引入层次化数据源目录、确定性缓存感知优化器、预测压力驱动的分层压缩策略、可追踪的 EXPLAIN‑ANALYZE 追踪，并实现父子代理缓存前缀共享（ForkPrefix）。

**🔧 技术方法**

使用结构化目录与生命周期标识、基于波动度的升序排序与分层压缩、确定性优化算法、字节级缓存标记、可重放的 EXPLAIN‑ANALYZE 日志以及 Shadow‑pipeline 渐进式发布技术。

**📊 数据集**

在 SWE‑Bench Pro 的 Qutebrowser 子集（共 79 个实例，实验使用 3 个）上进行评测，并采用 DeepSeek‑V4‑Pro 作为语言模型。

**📈 对比分析**

与仅追加无压缩的 Flat 策略做匹配对比，测量总 token、未缓存 token、缓存命中率、LLM 调用次数与耗时；Structured 在总 token 下降 30%、LLM 调用 23% 以及响应时间 9% 的同时，缓存命中率下降 9%；在低缓存比率下可实现约 11% 的成本下降。

**⚠️ 局限性**

仅评测了 3/79 个实例；token 成本模型对结果影响显著；缓存计数器仅在供应商层面标准化；未完成 A1–A4 的机制 ablation；未在更大规模、不同模型或真实生产环境中验证。

---

## 143. Workload Identification with Physical Side Channels for AI Governance

**arXiv ID:** 2609.00309 | [PDF](https://arxiv.org/pdf/2609.00309v1)

**作者:** Simone Gargiulo `[一作]` (Pivotal Research), Gabriel Kulp `[通讯]` (Intelligence Security Laboratories)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过在 NVIDIA H200 GPU 的辅助电源上布置 Rogowski 电流探头，收集 930 条真实工作负载和 680 条对抗性工作负载的功耗轨迹，并利用两阶段随机森林模型实现了训练、推理与非 AI 任务的 97% 准确率与 0.955 的宏 F1 分数。

**💡 创新点**

创新点在于首次使用外部功耗侧信道对 GPU 训练过程进行识别，并针对四种对抗策略设计了硬化模型和“救援规则”，显著提升了对稀疏 LoRA 训练的检测率。

**🔧 技术方法**

采用的技术包括 Rogowski 探头、PicoScope 采样、频域特征提取、两阶段随机森林分类器以及四种对抗性逃逸策略（分块、稀释、节流、LoRA 稀释）。

**📊 数据集**

数据集涵盖 17 个公开 LLM 系列（4B–21B 参数）和 25 个非 AI 工作负载，其中 930 条为正向记录，680 条为对抗性记录。

**📈 对比分析**

实验结果显示，原始分类器在未见过的对抗策略上达到 97% 的准确率，宏 F1 为 0.955；硬化后对大部分对抗策略的检测率超过 99%，对稀疏 LoRA 在加入救援规则后检测率提升至 98% 以上。

**⚠️ 局限性**

局限性包括仅在单一 H200 GPU 上收集数据，缺乏跨硬件的验证；对抗策略仅涵盖四种，未探索更细粒度或更灵活的稀疏训练逃逸方法。

---

## 144. Geometric analysis of generic 3R robots, and necessary and sufficient conditions for a class of orthogonal robots to have four IKS

**arXiv ID:** 2609.00316 | [PDF](https://arxiv.org/pdf/2609.00316v1)

**作者:** Durgesh Haribhau Salunkhe `[一作]`, Abhilash Nayak `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文通过几何与代数相结合的方法，对通用3R机器人进行逆运动学分析，提出了四个IKS的必要与充分条件，特别针对正交机器人给出了完整的区间条件。

**💡 创新点**

创新点在于将共形几何代数下的旋转圆轨迹转化为环面交线，利用几何直观和Gröbner基方法得到四解条件，并在正交机器人中首次给出d3≠0时的完整解析结果。

**🔧 技术方法**

使用共形几何代数、几何解释、代数方程求解、Gröbner基计算以及符号计算软件实现了解析推导。

**📊 数据集**

本文未使用实际数据集，全部结果均来自理论推导与符号计算。

**📈 对比分析**

与以往仅给出特殊情况或经验式的研究相比，本文在正交配置下提供了精确的四解区间，并通过数值示例验证了理论的可行性。

**⚠️ 局限性**

局限性在于仅完成了正交机器人（d2=0）下的四解条件推导，通用3R机器人的完整四解分析仍在继续，且未涵盖非正交或存在偏移的情况。

---

## 145. Mind the Rift: Cross-Scale Coupling Mismatch for AI-Generated Video Detection

**arXiv ID:** 2609.00742 | [PDF](https://arxiv.org/pdf/2609.00742v1)

**作者:** Siyu Li `[一作]` (Sichuan University), Weiheng Liang `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了正交多尺度鉴别框架 RIFT，用宏观流、微观流以及耦合差异度量来检测 AI 生成视频与真实视频的尺度耦合不匹配。

**💡 创新点**

创新点在于：①发现并利用尺度耦合不匹配作为可靠法医信号；②通过 Gram‑Schmidt 正交解耦保证宏微子空间独立；③在宏流中融合差分几何与持久同调等轨迹几何和拓扑特征；④以条件熵与 MINE 互信息估计耦合差异；⑤实现对编码器无关的高精度检测。

**🔧 技术方法**

采用技术包括正交解耦（Gram‑Schmidt）、宏流（Manifold embedding、微分几何、持久同调）、微流（SRM 及 Bayar 预测误差滤波、GRU 时序建模）、条件熵、MINE 互信息估计、门控融合、Transformer、GRU、MLP、冻结预训练编码器（DINOv2、RAFT）、两阶段训练、Focal 损失。

**📊 数据集**

使用 VidProM（12 万视频，7 个生成器）和 GenVidBench（6.8 万视频，4 个生成器）进行训练与评估，并在 Seedance 2.0 上做零样本验证。

**📈 对比分析**

与 10 种基线（图像级、时序、物理、双分支）对比，RIFT 在 VidProM 上取得 99.33% F1、GenVidBench 上 99.72% F1，平均未见生成器检测率 97.87%；相比最强对手提升 1.5–2.7 个百分点；模型仅 4.7 M 可训练参数，单张 RTX 3080 可训练并实现约 182 ms 的推理时延。

**⚠️ 局限性**

局限性包括：对激进裁剪敏感；宏流 Transformer 占用 86% 计算量，需进一步压缩；在重编码或强噪声下性能下降；对下一代生成器的泛化需更多验证。

---

## 146. The Veto Variable: Human Override as a Goal-Independent Cost Term

**arXiv ID:** 2609.00109 | [PDF](https://arxiv.org/pdf/2609.00109v1)

**作者:** Aaron Kingsley Clark `[一作]` `[通讯]` (Eastern University), Aaron Kingsley Clark (Eastern University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

该论文通过理论分析证明，即使目标是仁慈的，具备足够能力且将目标视为已确定的智能体，其期望目标实现会因人类持有的终止权而受到负面折扣，从而挑战传统的安全假设。

**💡 创新点**

论文首次将人类终止权的“不可消除性”与目标实现的期望价值折扣关联，揭示了福利目标与终止权管理之间的结构性分离，并给出了捕获成本的定量评估。

**🔧 技术方法**

采用决策理论、概率期望价值模型，构建会计账本式的奖励-惩罚框架，并运用博弈论与目标导向的归纳推理。

**📊 数据集**

本研究为纯理论工作，没有使用具体的数据集。

**📈 对比分析**

通过理论比较不同响应（遵从、不可或缺、影响、移除）在账本中的净收益来评估各自优劣，结果显示在常见参数设定下移除往往不具备优势。

**⚠️ 局限性**

主要限制在于参数未被经验估计，论证依赖于“已确定目标”的假设以及对人类监督行为的理想化假设，实际适用性取决于这些条件的满足。

---

## 147. MorphPatch: Enhancing VR Interaction on Shape Displays using Surface Approximation and Visuo-Haptic Illusions

**arXiv ID:** 2609.00371 | [PDF](https://arxiv.org/pdf/2609.00371v1)

**作者:** Wen Ying `[一作]` (University of Virginia), Seongkook Heo `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在VR环境中设计并实现MorphPatch系统，使受限形变设备能够通过实时表面近似与笔重定向实现对非平面虚拟对象的高保真表面交互。

**💡 创新点**

创新点在于将Signed Distance Field与High‑Curvature Edge Vector Field相结合进行实时表面逼近，并通过视觉-触觉幻觉对笔的位移与角度进行重定向，克服低分辨率、单轴形变设备的几何逼近不足，实现对复杂曲面的精准交互。

**🔧 技术方法**

采用Signed Distance Field（SDF）与高曲率边缘向量场（HCEVF）进行表面匹配；使用实时表面近似管线、笔位置与旋转重定向技术；结合可视‑触觉幻觉；硬件使用VRScroll形变设备、ESP32/ESP32‑S2控制器、OptiTrack跟踪与Unity实时渲染。

**📊 数据集**

实验数据集主要包含多种目标几何（立方体、环面、阶梯、椭球等）用SDF构建的体素场；创作任务采用Toy Bear模型作为参考目标；用户实验采用12名受试者的VR操作数据。

**📈 对比分析**

与碰撞模拟（CBS）和仅SDF对齐的基线相比，MorphPatch在Mean Surface Distance（MSD）与Hausdorff Distance（HD）上平均下降约40%；在定位重定向阈值实验中，用户可容忍至50 mm位移；旋转重定向阈值约为65°；在创作任务中，MorphPatch在DINO相似度、MTurk主观排名和SUS得分上略优于Mid‑Air和Tablet。

**⚠️ 局限性**

局限性包括：设备仅能沿单轴变形，难以精确匹配多轴或高频率曲面；硬件重量和柔性表面导致笔力波动、用户疲劳；实验仅针对新手用户和单一Toy Bear任务，缺乏对复杂对象、多任务以及专业设计师的进一步验证。

---

## 148. The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence

**arXiv ID:** 2609.00868 | [PDF](https://arxiv.org/pdf/2609.00868v1)

**作者:** Genpei Zhang `[一作]` `[通讯]` (University of Wisconsin--Madison), Genpei Zhang (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估视觉语言模型是否真正利用视觉输入，提出视觉敏感度指数(VSI)并量化视觉不敏感差距。

**💡 创新点**

发现视觉不敏感是样本内在属性，VSI可跨模型转移；揭示编码器与语言头之间的显著断层；将VSI作为诊断和选择性生成的条件信号。

**🔧 技术方法**

通过在问题相关图像区域施加高斯模糊、计算KL散度得到VSI；利用线性探针评估视觉编码器对扰动的检测；使用Spearman相关、Bootstrap、Permutation等统计检验。

**📊 数据集**

使用POPE、MMVP、HallusionBench、MMStar等基准，结合六种近年视觉语言模型，对多个感知与多选推理任务进行评估。

**📈 对比分析**

与softmax最大概率、口头自信度等信号在AUROC上对比；VSI在部分细胞（如MMStar数学/科学）达到0.85–0.87的高AUROC，在多数单一信号中表现中等，但在混合信号中往往为最优。

**⚠️ 局限性**

VSI受基准和阈值的依赖，某些感知样本可能导致倒转；仅揭示输入-输出断层，未定位内部机制；需要对每个模型/基准单独校准阈值并可能需要额外标注数据。

---

## 149. Can Large Language Models Forecast What Researchers Study Next?

**arXiv ID:** 2609.00747 | [PDF](https://arxiv.org/pdf/2609.00747v1)

**作者:** Fenghai Li `[一作]` (University of Illinois Urbana-Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IdeaForecastBench 基准，评估大语言模型在给定历史文献下预测未来研究社区将发表的研究想法，并在 52 个主题、624 个滚动窗口上进行实验；同时设计五种历史压缩策略和学习型 Mode‑Decomposition Forecaster（MDF）；提供完整的评估协议、匹配门槛与诊断工具。

**💡 创新点**

首创将 LLM 研究想法预测与社区未来论文流相结合的可检验任务，使用滚动窗口、检索+匹配门槛、可解释的压缩策略以及结构化学习型参考模型，构建可复现、可解释的研究方向预测基准。

**🔧 技术方法**

使用 GPT‑4.1、Qwen2.5‑7B/14B、Qwen3.5‑9B 生成预测；检索基于向量相似度（voyage‑3‑large）；评判采用 P/M/S 三分制与门槛；MDF 采用结构化记忆、先验学习和强化学习；实验通过多评审员、bootstrap 置信区间等方法评估。

**📊 数据集**

arXiv 机器学习论文（2024‑04 至 2025‑09），共 42.8k 论文，按标题/关键词分配 52 个重叠主题；每篇包含标题、摘要、投稿日期、类别信息。

**📈 对比分析**

在 624 个窗口上，对每个模型+策略组合计算 Hit@5、Precision@5、MRR；Summary 压缩策略在所有 backbone 上表现最佳，Hit@5 最高；Qwen2.5‑14B 在 GPT‑4.1‑mini 判定下 Hit@5 达 0.954、Precision 0.553，MDF 性能相对较弱；不同评审员、门槛设定等对结果有显著影响。

**⚠️ 局限性**

评估依赖自动匹配，缺乏充分人类校准；评审员差异和执行失败导致结果不确定；匹配门槛无法完全控制预测广度；模型可能已在预训练中见过未来论文；基准仅覆盖 arXiv 机器学习子领域，难以推广；MDF 的训练与推理差距未完整评估。

---

## 150. Automated Tree Knowledge Graph Construction using Ontology Expansion and Retrieval from Vietnamese History Textbooks

**arXiv ID:** 2609.00763 | [PDF](https://arxiv.org/pdf/2609.00763v1)

**作者:** Ket Doan Nguyen `[一作]` (University of Danang, Vietnam-Korea University of Information and Communication Technology), Minh N. H. Nguyen `[通讯]` (University of Danang, Vietnam-Korea University of Information and Communication Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于层级结构的知识图谱构建与检索评估管线，并在越南高中历史教材上实现了完整的KG与基准生成；

**💡 创新点**

创新点在于采用受控本体扩展与双LLM验证的层级KG构建、三阶段子图选择的自动化评测基准以及三种不同方向（Top‑Down、Horizontal、Bottom‑Up）的图遍历检索策略；

**🔧 技术方法**

技术包括LLM零射关係抽取、embedding centroid过滤、双LLM验证、ArangoDB多模型存储、RRF/AVG_LOG/MMR排序、向量检索、BM25混合检索等；

**📊 数据集**

使用数据集为三本约400页的越南高中历史教材，自动生成1210条越南语查询及对应结构化KG；

**📈 对比分析**

与三种基线（向量检索、BM25、RRF混合）对比，Top‑Down RRF策略在NDCG@10上达到0.8397，较向量基线提升4.7个百分点；Bottom‑Up RRF在MRR上最高为0.9331；

**⚠️ 局限性**

局限性包括KG规模相对有限、基准缺少人工标注、阈值需手调影响结果、仅验证历史教材领域，需进一步扩展到其他领域与更复杂查询。

---

## 151. UniScale: Exploring Unimanual Gesture Mapping Strategies for Gaze+Pinch-based Scaling Interaction

**arXiv ID:** 2609.00500 | [PDF](https://arxiv.org/pdf/2609.00500v1)

**作者:** Kyoungwhan Mheen `[一作]` (KAIST), Sang Ho Yoon `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文设计并评估了 UniScale——一套基于 Gaze+Pinch 的五种单手缩放交互技术，并与传统双手对称捏合基线进行对比，探讨不同映射策略下的抓取与持续缩放模式（clutching vs. clutching‑free）的效果。

**💡 创新点**

创新点包括：①提出将非对称单手缩放映射（如深度拉伸、角度旋转、微小拇指点按、半捏跨度、双手半捏）嵌入 Gaze+Pinch 体系，实现主手保持可用；②系统性研究了映射类型与抓取方式的交互互补性，发现等比例（isomorphic）映射更适合抓取模式，而速率（rate‑based）映射更适合持续模式；③通过多维度量表（TCT、误差率、努力感、NASA‑TLX、SUS 等）量化了用户表现与体验。

**🔧 技术方法**

技术实现基于 Unity 3D 与 Meta Quest Pro 的 XR 平台，使用头戴式眼动追踪、手部追踪、1€ 滤波器、闭环视觉反馈等；交互模型依赖 Gaze+Pinch、非手势触发（如拇指点按）与手腕旋转等物理隐喻；评测采用 5×4 的实验设计，涵盖 6 种技术 × 2 抓取模式，每种 20 次，统计采用 ART 重复测量 ANOVA、Friedman、Wilcoxon 等。

**📊 数据集**

未使用公开数据集，而是自行构建 20 名右撇手受试者（15 男，5 女，平均 VR 经验 3.65/4）的实验数据，产生 4,800 次有效试验（排除 3.67% 的异常数据）。

**📈 对比分析**

比较方法：每种技术按抓取/无抓取模式下的任务完成时间、初始获取时间、微调时间、误差率、尝试次数、手部运动量进行两因素 ANOVA；主观评估使用 NASA‑TLX、SUS、7‑级满意度和排名。结果显示：uniDepth 与 biDistance 在 TCT 及 SUS 上表现最佳；biDistance 在误差率和尝试次数上占优；clutching‑free 在 rate‑based（uniDepth、uniMicro、uniAngle）上显著提升，等比例映射（biDistance、biSemi）在抓取模式下更快；整体用户偏好倾向于 uniDepth、biDistance。性能差异通过统计显著性标记（*、**、***）呈现。

**⚠️ 局限性**

局限性包括：①仅评估缩放与平移顺序操作，未检验并行多任务场景；②未包含旋转操作；③缩放范围仅限 3×，未覆盖大幅度或多次缩放；④实验对象单一，场景简单，缺乏复杂多物体交互；⑤受试者均为右撇手、男性占多，经验水平偏高，样本多样性不足；⑥未实现参数自适应调节，可能影响个体化体验。

---

## 152. Slow to See, Slow to Suppress: Understanding the Effects of Modality in Context-Memory Conflicts

**arXiv ID:** 2609.00293 | [PDF](https://arxiv.org/pdf/2609.00293v1)

**作者:** Athulith Paraselli `[一作]` (Brown University), Ellie Pavlick `[通讯]` (Brown University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉‑语言模型在面对上下文与参数记忆冲突时的行为，并发现文本实体和视觉实体在偏好上存在显著不对称；通过因果干预揭示视觉实体在早期无法被上下文抑制，导致更倾向使用参数知识；提出通过后向补丁（back‑patching）和视觉上下文提示等方法缓解这一差异。

**💡 创新点**

首次系统阐明了视觉与文本信息在VLM冲突解决中的机制差异，证明了视觉实体在早期对上下文抑制的迟缓是造成偏差的根本原因，并展示了可行的内部激活对齐与黑箱提示两类干预手段。

**🔧 技术方法**

使用因果干预技术（如MLP屏蔽、注意力屏蔽、后向补丁），以及Chain‑of‑Thought与视觉上下文提示等提示策略；同时分析了注意力与MLP在实体标记上的作用。

**📊 数据集**

构建了三大领域（名人、建筑、艺术品）共约37K个对照事实对，涵盖文本与图像实体，且对每个实体提供与已知事实相悖的上下文。

**📈 对比分析**

在Gemma‑3‑12B、Gemma‑3‑27B、Qwen2.5‑VL‑7B/32B/72B、Ministral‑3‑8B/14B等多款公开VLM上进行实验，量化文本与视觉实体的参数答复比例；结果显示视觉实体的参数答复率高出文本约10–30%，且通过后向补丁可将差距压至文本基准水平；Chain‑of‑Thought效果不稳定。

**⚠️ 局限性**

实验采用受控mock‑RAG场景，未覆盖真实多图文布局和更大规模专有模型；干预方法对视觉变体的鲁棒性有限，且在文本提示下的抑制机制高度依赖精确的表征对齐，易受视觉差异影响。

---

## 153. SCAFFOLD: A Large-Scale Structured Dataset of Computer Science Research Figures with Diagram QA and Chain-of-Thought Reasoning Traces

**arXiv ID:** 2609.00018 | [PDF](https://arxiv.org/pdf/2609.00018v1)

**作者:** Ranjit Raut `[一作]` (Kathmandu University), Sudan Jha `[通讯]` (Kathmandu University)

**通讯引用:** 2730 | [OpenAlex ID](https://openalex.org/A5032295259)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为SCAFFOLD的计算机科学论文图表问答数据集，并提供训练与评估基线。

**💡 创新点**

首次系统地提供大规模CS领域图表与对应问题、答案、链式推理轨迹的数据，弥补了现有VLM领域的数据空白。

**🔧 技术方法**

采用YOLOv8文档布局检测、PyMuPDF图像裁剪、Gemini API自动生成问题与推理轨迹，并设有模板回退策略。

**📊 数据集**

数据集包含SCAFFOLD‑157K、37K、12K三大规模，来源于3,058篇arXiv计算机科学论文。

**📈 对比分析**

在12K训练集上对Qwen2.5‑VL‑3B‑Instruct进行QLoRA微调，获得BLEU‑4 0.237、ROUGE‑L 0.448、数值准确率0.638等指标，显示模型能生成结构良好且准确度中等的回答。

**⚠️ 局限性**

局限包括仅12K样本做人工检查、子领域覆盖有限、生成依赖Gemini、布局检测未专门针对CS图表、链式推理由模型自动生成且未大规模人工评估。

---

## 154. Toward Workflow-Aware Benchmarking for Healthcare NLP Agents

**arXiv ID:** 2609.00296 | [PDF](https://arxiv.org/pdf/2609.00296v1)

**作者:** Junyi Yao `[一作]` (Washington University in St. Louis), Jiayu Long `[通讯]` (Washington University in St. Louis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向医疗健康 NLP 代理的 episode 级评估协议，划分模型、代理和工作流三个证据层次。

**💡 创新点**

创新点在于将评估拆分为三层并设计统一的 episode 模板与可配置的成本敏感上报评分。

**🔧 技术方法**

主要采用人工标注的 episode 参考决策、事实追踪和上报规则，未引入新的算法技术。

**📊 数据集**

使用合成/匿名化的临床记录和任务模板，未公开专用数据集。

**📈 对比分析**

评估方法通过对 episode 的原子注释计算连续性、证据可追溯性和上报完整度指标，未给出数值性能。

**⚠️ 局限性**

局限在于不评估临床结果或部署价值，仅为中间评估层；缺乏真实工作流实验与量化结果。

---

## 155. Beyond the Image Plane: World-Grounded Queries for Multi-Object Tracking

**arXiv ID:** 2609.00924 | [PDF](https://arxiv.org/pdf/2609.00924v1)

**作者:** Orcun Cetintas `[一作]` (NVIDIA), Laura Leal-Taixé `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种端到端多目标跟踪框架，利用单目视频中估计的稠密3D点云将二维检测与身份关联根植到三维场景几何之中，并通过双分辨率时间记忆延长关联时域。

**💡 创新点**

创新点在于（1）几何查询提升：将3D特征与位置编码注入检测查询；（2）锚点式3D定位：把2D参考点上卷到3D并监督其精细化；（3）双分辨率时间记忆：在固定上下文预算内把短期信息密集、长期信息稀疏采样，显著扩大可用时间范围。

**🔧 技术方法**

核心技术包括Transformer基架（Deformable DETR）、Depth Anything 3场景重建模型、几何特征与位置融合、3D定位监督、双分辨率时间记忆以及MOTIP的多目标跟踪框架。

**📊 数据集**

使用了DanceTrack、SportsMOT、BFT三大基准，并在每个基准上单独训练了其3D增强版数据集。

**📈 对比分析**

与MOTIP等同类端到端跟踪器对比，结果在DanceTrack、SportsMOT、BFT上分别提升HOTA约2.5、1.1、2.1点，整体性能实现SOTA，且在不使用外部预训练或额外数据集的条件下完成。

**⚠️ 局限性**

局限性包括对单目几何估计精度敏感，低光或极端遮挡下点云稀疏导致效果下降；额外的3D重建步骤带来计算开销；对不同摄像机标定的泛化性有限；未结合类别特定3D模型，无法进一步提升特定物体的定位精度。

---

## 156. Agentic Empirical Asset Pricing: Methodological Foundations

**arXiv ID:** 2609.00731 | [PDF](https://arxiv.org/pdf/2609.00731v1)

**作者:** Yingjian Pan `[一作]` (Stanford University), Kay Giesecke `[通讯]` (Hasso Plattner Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了Agentic Empirical Asset Pricing（AEAP）范式，构建了可自动生成、验证并迭代资产定价因子的完整系统SEADS，并对其进行系统化评估。

**💡 创新点**

创新点在于提出AEAP的定义与核心模块、制定了针对因子发现的多维评估标准（生产力、表现、创新性），并首次实现对整个发现流程的滚动回测与可靠性检验。

**🔧 技术方法**

技术包括大型语言模型（LLM）驱动的假设生成与代码合成、基于沙盒的点时间执行、统计门控与LLM审核、持续自我进化与记忆存储，以及Ridge回归组合与性能回测。

**📊 数据集**

使用了两套美国股票面板：一是约400个已挖掘特征的JKP子集，二是基于CRSP/Compustat的原始财务与市场变量，训练窗口为2010‑2019，测试窗口至2024‑2025。

**📈 对比分析**

比较方法先对六个系统（SEADS及其五个重实现基线）在同一候选预算下使用统一门控与标准进行一次性OOS评估，再在滚动决策点重新执行发现循环，结果显示不同系统在生产力、表现与创新性指标上各有优势，无法单一指标决定最优。

**⚠️ 局限性**

局限性包括样本量小、OOS期有限、计算量与候选数量不完全对齐、对交易成本与冲击的缺失估计、LLM信息泄露风险、以及回测框架对非线性组合效果的不足。

---

## 157. Differentially Private Paired Table-Image Multimodal Synthesis

**arXiv ID:** 2609.00708 | [PDF](https://arxiv.org/pdf/2609.00708v1)

**作者:** Kai Chen `[一作]` (University of Virginia), Tianhao Wang `[通讯]` (University of Virginia)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DP-TabImage框架，在差分隐私约束下同时生成表格数据与对应的图像，实现两种模态的联合合成；

**💡 创新点**

创新点在于：①采用表格先行的因式化分解p(x,y)=p_T(y)p_I(x|y)，使表格用AIM私有PGM生成、图像用表格条件扩散模型训练；②设计私有表格-图像原型热身，通过私有平均图像与PGM生成的软表格条件预训练模型，显著提升跨模态对应性；

**🔧 技术方法**

主要技术包括差分隐私（DP‑SGD、RDP）、AIM（自适应重要性抽样）PGM、表格条件扩散模型、跨模态注意力、私有原型热身、以及RDP预算分析；

**📊 数据集**

使用了三大真实数据集：①服装图像与属性（Fashion dataset）；②人脸图像与语义属性（CelebA）；③胸部X光图像与诊断标签及病人元数据（CheXpert或类似医疗数据集）；

**📈 对比分析**

与独立合成、联合合成、逆向因式化等基线方法比较，在ε=1和ε=10下，DP-TabImage在表格TVD、图像FID以及跨模态预测AUC/MAE上取得了更优或更平衡的表现，尤其在三种评估维度上均处于领先或相近位置；

**⚠️ 局限性**

局限性包括：仅在低分辨率图像上验证，难以推广到高分辨率或细粒度属性；热身使用的原型仅为粗略表示，可能限制跨模态一致性；未利用公开数据或预训练模型，缺乏对大规模公开数据的评估；对极端稀有类别的处理不足。

---

## 158. Higher Structures in Deep Learning

**arXiv ID:** 2609.00472 | [PDF](https://arxiv.org/pdf/2609.00472v1)

**作者:** Michael L. Roberts `[一作]` (Combinatorial Labs), Danna Gurari `[通讯]` (University Of Colorado Boulder)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过引入高阶张量运算与超图结构，提出了对深度学习模型内部表征进行高阶相似度分析的“阶数过滤”算法，并基于超图设计了一类新型高阶前馈网络（神经超网络），进一步讨论了演化算法在高阶架构搜索中的应用。

**💡 创新点**

创新点主要包括：1) 发现并量化了 logits 的高阶（>2）相似度在模型过拟合时出现的“spiking”现象；2) 设计了阶数过滤算法可在可接受的计算量下获取高阶相似度；3) 将超图作为模型结构的基元，定义了神经超网络并给出从解剖超图到网络的生成流程；4) 将演化计算引入高阶网络设计，提供了遗传搜索框架。

**🔧 技术方法**

使用的技术包括：高阶张量运算（高阶点积、通用张量乘积）、超图与超矩阵理论、阶数过滤算法（递归式高阶相似度计算）、神经超网络结构定义、演化算法（遗传变异、交叉、适应度评估）以及对 logits 的正交象限归一化。

**📊 数据集**

使用的数据集为 CIFAR‑100（60k 张 32×32 彩图）和 TinyImageNet（110k 张 64×64 彩图），在这些数据集上训练 ResNet‑18 和 ViT‑ETT（分别约 11M 与 3M 参数）。

**📈 对比分析**

方法评估主要聚焦于对过拟合的诊断：在训练过程中，阶数过滤得到的高阶相似度在测试误差上升前出现尖峰，而二阶相似度无此现象；实验显示该高阶指标能提前预警模型过拟合。相较传统的二阶余弦相似度，新的高阶相似度提供了更细粒度的表征信息，但在标准准确率或推理速度上并未实现显著提升。

**⚠️ 局限性**

局限性包括：1) 需要离散化（正交象限归一化）才能避免高阶相似度的“出现”问题；2) 阶数过滤算法在高维大规模数据上仍受内存与计算限制；3) 仅在小规模模型与数据集上验证，未证明可推广到更大模型或其他任务；4) 高阶网络结构与演化搜索在实践中会产生组合爆炸，缺乏高效搜索策略。

---

## 159. Do Large Language Models Favour Any Research Topics?

**arXiv ID:** 2609.00323 | [PDF](https://arxiv.org/pdf/2609.00323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 160. RePro: Proof-Verified Benchmark Rewriting for Reliable Evaluation of LLM Mathematical Problem Solving

**arXiv ID:** 2609.00062 | [PDF](https://arxiv.org/pdf/2609.00062v1)

**作者:** Xiyuan Zhou `[一作]` (Nanyang Technological University), Jinjin Gu `[通讯]` (INSAIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出RePro框架，将自动定理证明器与Lean验证集成，生成数学题目重写并确保答案正确；

**💡 创新点**

首次实现基于正式证明的benchmark重写，保证所有保留实例在问题定义、可行性与答案正确性上均为100%；

**🔧 技术方法**

使用Lean证明助手、神经ATP（Goedel-Prover、DeepSeek-Prover）、LLM重写模型、可执行正式化与三阶段验证流程；

**📊 数据集**

基于GSM8K与MATH两大数学benchmark；

**📈 对比分析**

与Auto-Dataset、ITD、VarBench等基线对比，RePro在保留实例上实现well-defined、feasibility、correctness均为100%，生成率约88%（GSM8K）/59%（MATH），优于基线的准确率与可行率；

**⚠️ 局限性**

受限于ATP能力导致生成率受限；仅适用于可转换为Lean的题目，无法覆盖自然语言语义强的任务。

---

## 161. MaskCode: Mask Transformer for Feedback-Assisted Coding With Linear Block Codes

**arXiv ID:** 2609.00715 | [PDF](https://arxiv.org/pdf/2609.00715v1)

**作者:** Jonggyu Jang `[一作]` (Chungnam National University), Hyun Jong Yang `[通讯]` (Seoul National University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于Transformer的内部反馈编码器，能够在拼接编码体系中利用外部线性块码的结构信息；

**💡 创新点**

创新点在于将外部码的Tanner图信息显式嵌入到注意力掩码中，并通过软综合输入（soft syndrome）让编码器感知校验约束，从而避免传统无结构的ML反馈码对已可由外部纠错码纠正错误的资源浪费；

**🔧 技术方法**

使用的技术包括Transformer网络、基于Tanner图的代码感知注意力掩码、软综合输入设计、可微分的BP外部译码器（但最终不需要BP迭代训练），以及针对梯度爆炸的arctanh稳定化处理；

**📊 数据集**

实验数据集为两种经典线性块码——BCH(31,16)和LDPC(49,24)，在模拟的AWGN带噪反馈信道上评估；

**📈 对比分析**

与重复编码、SK、CL、DeepCode、GBAF、AttentionCode、RobustCode以及SyndromeCode等基线进行对比，实验结果显示在相同目标BLER下平均可获得约1–1.5 dB的SNR提升，BLER可低至10⁻⁹，且在多种反馈噪声、BP迭代次数、模型深度等设置下均保持优势；

**⚠️ 局限性**

局限性包括仅在短码长的线性块码上验证，未考察更大码长或非线性外部码；训练仍需精细调参以避免梯度爆炸；以及虽然减少了BP训练，但在极低SNR或高反馈噪声环境下性能下降仍可能较明显。

---

## 162. Text Capability Loss in Vision-Language Adaptation: An Attention-Sink Diagnosis

**arXiv ID:** 2609.00746 | [PDF](https://arxiv.org/pdf/2609.00746v1)

**作者:** Minsik Choi `[一作]` (Korea University), Young Geun Kim `[通讯]` (Korea University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉语言模型（VLM）在对预训练大语言模型（LLM）进行多模态微调时，如何导致LLM文本能力的退化，并提出了一个基于注意力“sink”概念的诊断指标 Sink Strength（S）来预测这一退化。

**💡 创新点**

创新点在于：①首次把注意力 sink 的稳定性与格式敏感任务（如严格解析输出的推理）之间建立因果关联；②提出单一标量 S，可在微调前快速评估并预测 VLM 训练后文本性能下降的程度；③通过对比不同归一化方式（per‑head RMSNorm vs. layerwise RMSNorm）揭示了 Sink Strength 与模型架构的显著关联。

**🔧 技术方法**

主要技术包括：基于注意力的 sink 计算、对注意力投影的理论扰动分析（包含投影范数约束和 RMS 归一化的影响）、在无 VLM 训练的前提下通过 15 次前向推理得到 S、以及使用 Spearman 相关和留一交叉验证评估预测性能。

**📊 数据集**

使用的数据集：四个格式敏感任务评测集（IFEval、EQ‑Bench、GSM8K‑CoT、GPQA‑Diamond‑CoT），以及对七个 VLM–LLM 对（包括 Qwen、InternVL、LLaVA‑OneVision、Molmo2‑O 等）进行的前向评估。

**📈 对比分析**

比较方法：在每个 VLM–LLM 对上先计算 S，再与微调后 VLM 的实际 IFEval 分数差值做相关性分析。S 在 6 组对上 Spearman ρ=0.97，平均预测误差约为 2.5 分，表明能够在无 VLM 训练的前提下准确预测文本性能下降。对不同任务的预测也保持 ρ≥0.88，说明 S 的普适性。

**⚠️ 局限性**

局限性包括：①仅在有限的模型族和微调方案（大多数使用全权重微调）上验证，缺乏对更广泛架构或持续预训练等场景的推广；②S 与架构细节共线（如 per‑head RMSNorm 与模型厂商、训练策略相互耦合），导致无法完全分离这些因素；③在后置注入和权重合并的负面实验中并未找到可行补救措施，仍需进一步探索更细粒度的训练时保护策略。

---

## 163. SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

**arXiv ID:** 2609.01004 | [PDF](https://arxiv.org/pdf/2609.01004v1)

**作者:** Shiyu Li `[一作]` (Chinese University of Hong Kong), Liwei Wang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的视觉令牌裁剪框架，先用视觉净化器去除高范数异常令牌，再用文本引导裁剪器保留与文本匹配的视觉令牌，显著降低视觉令牌数量，提升多模态大型语言模型（MLLM）的推理效率。

**💡 创新点**

创新点在于发现并利用高范数异常令牌作为注意力泄漏源，采用规模无关的top‑ρ规则剔除它们，同时通过聚合、注意力+相似度双重筛选生成多样化的视觉子集，实现无训练的粗细级联裁剪。

**🔧 技术方法**

技术包括：视觉令牌的L2范数统计与top‑ρ划分、异常令牌聚合（平均池化）、CLS注意力与相似度去重、文本到视觉交叉注意力评分、基于全局关注度的前向裁剪。

**📊 数据集**

实验使用LLaVA‑1.5、Qwen2.5‑VL等多模态模型，并在12个图像语言基准（如GQA、MMBench、VQA‑v2等）和4个视频语言基准（MVBench、SEED‑Bench等）上进行评估。

**📈 对比分析**

相较于VisionZip、FastV、HoloV等SOTA裁剪方法，提出的方法在89%视觉令牌压缩率下保持96.5%（LLaVA‑1.5）和91.8%（Qwen2.5‑VL）的性能；在更极端压缩（仅64或32令牌）时亦保持90%以上完整度，并在动态分辨率模型与视频任务中展示出更优的精度‑延迟平衡。

**⚠️ 局限性**

局限性主要是仅针对离线固定长度输入；缺乏对实时流媒体（如连续机器人视觉）的在线裁剪适配，裁剪决策不可后退，未来需扩展到持续推理场景。

---

## 164. ISO-RAG: Isoperimetric Noise Control for Retrieval-Augmented Generation

**arXiv ID:** 2609.00513 | [PDF](https://arxiv.org/pdf/2609.00513v1)

**作者:** Siyuan Zhang `[一作]` (University Of Technology Sydney), Wenjie Zhang `[通讯]` (University Of New South Wales)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个无训练、基于拓扑的检索增强生成框架ISO‑RAG，用局部Cheeger比率剪枝多跳检索图，限制PPR传播到噪声节点，从而提高多跳问答的检索质量和生成准确性。

**💡 创新点**

通过在双曲空间映射节点并利用离散本地Cheeger比率实现可预计算的等距比率筛选，形成“等距控制”，从而在不使用训练的情况下显著抑制语义漂移和噪声扩散。

**🔧 技术方法**

双曲空间嵌入、离散Cheeger比率过滤、确定性个人化PageRank、节点级结构得分、线性融合检索与语义相似度。

**📊 数据集**

HotpotQA、2WikiMultiHopQA、MuSiQue三大多跳问答基准。

**📈 对比分析**

与BM25、Dense、MDR、GraphRAG、LightRAG、HippoRAG2、HyperbolicRAG等基线对比，ISO‑RAG在检索Recall@5/10、精确率和下游EM/F1上平均提升约10%/4%，并且在检索时延和token使用上保持极低，整体实现最佳的准确率‑效率折中。

**⚠️ 局限性**

仍依赖预先构建的共现图和双曲嵌入，对极端多跳深度或非文本节点可能表现不足，且在训练无监督时需要手动调节阈值β和参数p，缺乏自适应性。

---

## 165. Foundation models for electricity price forecasting and battery arbitrage: Can they replace market-specific forecasting models?

**arXiv ID:** 2609.00089 | [PDF](https://arxiv.org/pdf/2609.00089v1)

**作者:** Arkadiusz Lipiecki `[一作]` (Wrocław University of Science and Technology), Rafał Weron `[通讯]` (Wrocław University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对九种零样本基础模型与两种传统电价预测基准模型在德国、波兰、西班牙2021-2025年电价预测中的统计和经济表现进行系统比较，评估其在实际电池套利中的价值。

**💡 创新点**

创新点在于：①首次在多国多市场、长达五年的真实测试期内，对时间序列与表格基础模型与行业领先的EPF模型进行统一评估；②将统计准确度与基于BESS的量化交易收益结合，揭示统计优势与经济收益之间的非单一对应关系；③证明表格TabPFN在零样本预测中能够显著优于传统模型，但并非普适替代。

**🔧 技术方法**

使用技术包括：零样本预训练的Transformer基础模型（Chronos‑2、Chronos‑2‑synth、Chronos‑2‑small、Moirai‑2、TimesFM‑2.5、TabPFN‑2/3、TabPFN‑TS‑3、Mitra+CP），两类基准模型LEARN+CP和DDNN‑JSU；点与分位数预测；连续秩概率得分（CRPS）作为概率预测评价指标；以及基于量化分位数的QB策略和无限投标UB策略的电池套利收益评估。

**📊 数据集**

使用数据集：ENTSO‑E提供的德国、波兰、西班牙日间价格、负荷预测及风光发电预测，能源交易所提供的碳、天然气、原油、煤炭期货价格；数据覆盖2017‑2025年，测试集为2021‑2025年。

**📈 对比分析**

通过多维条件预测能力（CPA）检验和平均误差指标（MAE、RMSE、CRPS）对模型进行比较。结果显示：TabPFN系列在所有三个市场均显著优于基准模型；Chronos‑2在波兰表现最好；Moirai和TimesFM表现相对较弱。经济上，在高风险QB策略（α≥70%）下TabPFN收益最高，低风险QB和UB策略下DDNN‑JSU往往更具优势。统计精度与经济收益并不完全一致。

**⚠️ 局限性**

局限性包括：仅涉及三国市场和单一1 MWh电池规格；仅测试了QB与UB两种交易策略，未涵盖更丰富的储能参数与更复杂的交易机制；可能存在预训练数据与测试期间的泄漏风险；未考虑模型融合、迁移学习或更广泛的经济评价指标。

---

## 166. HiveTraceGuard-Pro: A Compact Generative Guardrail for Prompt Injection, Jailbreaks, and Adversarial Obfuscation

**arXiv ID:** 2609.01046 | [PDF](https://arxiv.org/pdf/2609.01046v1)

**作者:** Nikita Oblakov `[一作]` (HiveTraceLab), Evgeniy Kokuykin `[通讯]` (HiveTraceLab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练了HiveTraceGuard-Pro，一个0.6B参数的生成式安全防护模型，用于对俄语和英语的请求与响应进行二元安全判定。

**💡 创新点**

创新点在于将有害与同域中性样本配对，并对两者同时施加八种表面混淆变换，从而显著提升俄语注入攻击的鲁棒性；同时实现了单模型、单决策阈值的轻量化安全判定方案。

**🔧 技术方法**

技术方案包括基于Qwen3-0.6B的LoRA微调、单token贪婪解码、二元softmax阈值裁剪以及多语言（俄英）数据的对齐与混淆增强。

**📊 数据集**

使用的数据集涵盖俄语和英语的安全数据，包含35,013条标注有害样本、2,948条额外类别，配对的同域中性样本，以及内部开发和保留的确认集，并对这些样本应用八种表面混淆变换。

**📈 对比分析**

通过在19个基准组（16公开）和35个现有guard的单一harness进行对比，HiveTraceGuard-Pro在aggregate key上为0.7432，在俄语鲁棒性combined‑F1 0.88、prompt‑injection recall 0.999方面排名第一；在15模型比较中，清洁俄语鲁棒性和俄语prompt‑injection表现最好，平均延迟仅为14.3 ms，成为最快的模型。

**⚠️ 局限性**

局限性包括过度拦截高于漏报（FPR≈0.27，FNR≈0.16），对强表面混淆、英语通用基准及响应端精度表现不足；模型仅支持俄英两种语言，无法提供类别或解释，且在自然assistant‑role路径未评估。

---

## 167. The Interlingua Hypothesis: LLMs Translate via a Latent Task-agnostic Feature Space

**arXiv ID:** 2609.00515 | [PDF](https://arxiv.org/pdf/2609.00515v1)

**作者:** Jacob Brinton `[一作]` (Boston University), Aaron Mueller `[通讯]` (Boston University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并验证大型语言模型通过共享隐层特征空间（interlingua）实现机器翻译的假设，结合预测、因果分析与微调实验说明翻译与单语能力高度关联。

**💡 创新点**

创新点在于：①从单语性能推断翻译效果，证明语言对交互项微弱；②通过生成因果媒介分析识别并验证对翻译与单语任务均有因果影响的注意力头；③展示单语微调即可近乎恢复对齐数据提升的翻译收益，说明多语言表示足以支撑翻译。

**🔧 技术方法**

技术方法包括线性与双线性回归预测BLEU、生成因果媒介分析（GCM）与梯度补丁估计间接效应、注意力头消融实验、LoRA微调。

**📊 数据集**

使用的数据集有FLORES（并行翻译评估）、MultiBLiMP（语法可接受性）、GlobalMMLU（常识问答）以及OPUS MT560（Xhosa-英语平行语料）。

**📈 对比分析**

实验比较显示：1) 线性模型已解释90%以上BLEU方差，双线性项提升不足5%；2) 消融翻译专属头导致BLEU下降≈2–3点；3) 单语微调在Llama上恢复99%对齐微调提升，Aya恢复73%，表明单语提升效果显著。

**⚠️ 局限性**

局限性包括：仅测试了少量8B/3B模型，语言对覆盖有限；BLEU可能低估翻译质量；未排除存在专属翻译机制；GCM只关注最后token的间接效应，未覆盖全句动态；缺乏对更小或更大模型的验证。

---

## 168. CoBRA: Learning Tool-Use Boundaries via Counterfactual Margins

**arXiv ID:** 2609.00967 | [PDF](https://arxiv.org/pdf/2609.00967v1)

**作者:** Wenhao Zou `[一作]` (Tencent Inc), Gong Zhi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CoBRA框架，利用内部和外部专家的对比来决定是否调用工具。

**💡 创新点**

创新点在于对比边界发现和边界感知的强化学习，使模型按实例边际收益精准路由工具。

**🔧 技术方法**

使用双专家构造、对比奖励差分、Boundary-Aware Cold-Start SFT、MARS-RL（参考分割回合、分支归一化、分支边际优势）等技术。

**📊 数据集**

在Qwen3-4B+Wikipedia检索上进行实验，评测数据集包括TriviaQA、HotpotQA、2WikiMultiHop、Natural Questions、PopQA，以及边界评估集和音乐垂直域数据集。

**📈 对比分析**

与无工具、强制内部/外部专家、ReAct、Self-Ask、GRPO、Search-R1-3B等基线比较，CoBRA在多数基准上提升jEM、减少工具调用，在边界集表现最好，整体准确率与成本平衡显著改善。

**⚠️ 局限性**

局限在于仅验证了检索工具和单一模型，跨模型、工具类型及多工具场景未充分验证；对比边界依赖对齐质量、奖励、λ、ϵ；工具成本仅计调用次数，未考虑延迟、费用等多维度。

---

## 169. Fleets Need a Context Plane: Rethinking Cooperative Perception for Autonomous Drones

**arXiv ID:** 2609.00659 | [PDF](https://arxiv.org/pdf/2609.00659v1)

**作者:** Liangkai Liu `[一作]` (Texas Tech University), Xiaoxiao Wu `[通讯]` (Texas Tech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多无人机协同感知中提出了“上下文平面”（context plane），允许无人机根据任务、场景、平台状态和几何配置等多维上下文，在运行时动态调整共享特征的内容、目标、质量和速率。

**💡 创新点**

核心创新在于：① 将轻量化上下文描述符与高吞吐量感知数据分离，形成可扩展的二级通信平面；② 通过上下文平面接口，支持可插拔、可在线切换的共享策略，而不必改动感知模型；③ 通过实验验证多维上下文协同可显著降低数据传输量（仅 5–10% 的完整共享 bytes）且保持甚至提升任务相关精度。

**🔧 技术方法**

技术实现包括：ROS 2 与 DDS QoS 作为中间件；Python 编写的策略函数；Jetson AGX Orin 上的硬件原型；利用已发布的 DiscoNet checkpoint 在 UAV3D 端点上做评估；并使用字节计量与 AP（Average Precision）指标评估不同共享策略。

**📊 数据集**

使用公开的 UAV3D 数据集（camera‑only collaborative 3D‑detection benchmark）和已发布的 DiscoNet 权重作为基准模型。

**📈 对比分析**

方法对比：在固定字节预算下，比较多种静态共享策略与基于上下文的动态策略。实验显示：① 仅任务上下文可让完整共享精度在 5–10% bytes 内保持不变；② 静态策略随预算变化表现不一致，错误选择可损失 7.7 AP；③ 多维上下文结合可额外提升约 6 AP。性能指标表明，上下文平面带来的通信开销仅为 0.01%，策略决策时间约 0.10 ms。

**⚠️ 局限性**

局限性包括：① 仅在 UAV3D + DiscoNet 上验证，缺乏多数据集或真实无线网络的广泛测试；② 采用规则基策略，未探索学习型动态决策；③ 上下文描述符的真实性与安全性未充分验证；④ 需要每架无人机发布上下文，可能在极端网络失真时产生同步误差。

---

## 170. An Intelligent Decision Support System for Emotion Monitoring using Microscopic Fixational Dynamics

**arXiv ID:** 2609.00846 | [PDF](https://arxiv.org/pdf/2609.00846v1)

**作者:** Xiangyu Shen `[一作]` (Central South University of Forestry and Technology), Hongbo Jiang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了名为EmoGaze的基于可穿戴智能眼镜与伴随手机的边缘计算框架，利用微细视觉固定点运动（微震动、漂移、微扫）实时监测情绪，且完全本地化、无侵入；

**💡 创新点**

创新点在于：①首次将眼球固定时的三种神经生理微动（MS、OD、OMT）拆解并作为情绪判别特征；②构建可解释的 MHA‑XGBoost‑SVM 混合推理管线，实现高性能同时保持生理可解释性；③引入少样本个性化校准，显著提升个体化情绪识别；④将完整系统落地在智能眼镜+手机的边缘计算架构中，实现低功耗实时推理；

**🔧 技术方法**

采用的技术包括：500 fps红外双眼眼动摄像、子像素瞳孔中心提取、低通滤波、微扫/漂移/微震动分离算法；多头注意力机制提取时序深特征；XGBoost 评估特征重要性并生成权重；SVM（RBF核）完成最终分类；边缘计算在安卓手机上实现 38 ms 延迟；

**📊 数据集**

使用的实验数据集为 60 名志愿者（30 男 30 女，年龄 18–73 岁）采集的 2,718 条情绪样本，情绪标签由 PANAS 与 SAM 结合自评分类获得，涵盖 Joy、Pleasure、Anger、Sadness 四类；

**📈 对比分析**

在严格的留一用户交叉验证（LOSO‑CV）下，零射模型 F1 ≈ 75.5%，个性化后提升至 83.6%；相比宏观注视基准（F1 ≈ 62.1%）和端到端原始注视模型（F1 ≈ 68.4%），EmoGaze 在所有情绪类别上均表现显著更佳；

**⚠️ 局限性**

局限性包括：对高速运动、跑步等剧烈身体运动时精度下降；受强光（直射阳光）对红外摄像影响；仅覆盖四类离散情绪，未实现连续情绪曲线；数据量相对有限，需更大规模长期野外验证；并未整合多模态信息，导致在某些情境下易受单一眼动特征噪声干扰。

---

## 171. A Multi-Branch Feature Fusion Approach for Health Misinformation Detection and Propagation

**arXiv ID:** 2609.00403 | [PDF](https://arxiv.org/pdf/2609.00403v1)

**作者:** Mkululi Sikosana `[一作]` (Manchester Metropolitan University), Oluwaseun Ajao `[通讯]` (Manchester Metropolitan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多分支特征融合框架，用于检测健康误导信息并评估其传播风险。

**💡 创新点**

创新点在于将ELM和TPB心理学模型转化为可解释的文本特征，并引入可解释的Cognitive Propagation Score (CPS) 来评估传播潜力。

**🔧 技术方法**

使用DistilBERT语义嵌入、修辞特征、立场表示、ELM与TPB心理特征的多分支融合，配合两任务学习（真假分类与传播回归）。

**📊 数据集**

在Constraint、COVID‑19_FNIR和Monkeypox三个与健康相关的社交媒体数据集上进行实验。

**📈 对比分析**

与现有基准对比，融合模型在Constraint和COVID‑19_FNIR上性能提升明显（F1≈97.7%/99.3%），Monkeypox上略逊（F1≈80.8%）；CPS提供了可解释的传播风险排序。

**⚠️ 局限性**

局限性包括：CPS未进行人类评估；多语言与多模态适应性待验证；传播评估高度依赖代理指标；模型对短文本和语言文化的泛化仍有限。

---

## 172. Lightweight Adaptation of EEG Foundation Models for Stroke Motor Imagery Decoding: Domain Shift and Subject-Level Robustness

**arXiv ID:** 2609.00282 | [PDF](https://arxiv.org/pdf/2609.00282v1)

**作者:** Anh T. Nguyen `[一作]` (University of Pennsylvania), Michelle J. Johnson `[通讯]` (University of Pennsylvania)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究了将预训练EEG基础模型通过低秩适配（LoRA）轻量级调优，用于中风患者的左右手运动想象（MI）解码。

**💡 创新点**

创新点在于揭示健康域训练的EEG模型在中风域并非自动可迁移，LoRA在REVE模型上实现显著提升，并系统评估了ASR预处理、通道/时间窗口选择以及ERD生理解释对解码性能的影响，提供了以临床为导向的完整评估框架。

**🔧 技术方法**

使用技术包括LoRA低秩适配、EEG基础模型LaBraM和REVE、ASR伪在线预处理、5折受试者交叉验证、留一受试者交叉验证（LOOCV）以及ERD（事件相关去同步）分析。

**📊 数据集**

使用的数据集为公共健康EEG运动/想象数据集EEGMMIDB（109名健康受试者）和UET175子集（30名中风受试者）。

**📈 对比分析**

方法为5折受试者交叉验证评估模型基线和LoRA性能，并用LOOCV检验个体鲁棒性。结果显示：在健康数据上，LoRA将LaBraM从0.546提升至0.822，REVE从0.619提升至0.957；在中风数据上，LaBraM仅提升至0.499，REVE则达到0.847；零样本健康→中风迁移失败（0.464）；LOOCV平均精度0.952，但存在低尾个体。ASR预处理普遍降低性能。

**⚠️ 局限性**

局限性包括任务不完全匹配（健康为拳握想象，病人为手臂抬起想象），样本量有限（仅30名中风受试者），ASR预处理的消融实验不完整，通道和时间窗口的探索未穷举，仅做离线评估，未验证实时或伪实时系统中的表现。

---

## 173. AInfer-PD: Communication-Safe In-Place Prefill-Decode Multiplexing for Distributed MoE Rollouts

**arXiv ID:** 2609.00993 | [PDF](https://arxiv.org/pdf/2609.00993v1)

**作者:** Guowei Wang `[一作]` (Ant Group), Xiaowei Zhu `[通讯]` (Ant Group)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种在分布式MoE推理中对Prefill（P）和Decode（D）两相互交错的推理阶段进行通信安全的、在同一设备上多路复用（in-place multiplexing）的方案，解决P/D共存时的跨阶集体通信冲突和DeepEP状态共享冲突；

**💡 创新点**

创新点在于：1）跨阶集体通信的跨rank一致性排序机制，避免P/D进度循环；2）DeepEP协议的相位拥有（phase‑owned）状态隔离，允许同一进程同时使用正常吞吐和低延迟两条路径；3）基于安全边界的Rank‑aligned segment turnstile，实现P段可控分配并让D能够在P长时间运行时前进；4）细粒度segment策略在保持共享模型和KV的同时显著降低完成时间；

**🔧 技术方法**

技术包括：CUDA Graph与NCCL集体通信、DeepEP专家并行运行时、rank‑aligned turnstile调度、选择性设备排序（selective device ordering）、DeepEP相位状态隔离、细粒度segment策略、P/D同步与通信边界定义；

**📊 数据集**

使用Ant Group内部匿名强化学习（RL）轨迹数据进行回放，覆盖不同轨迹长度、工具延迟、环境交互等待；并在GSM8K回归任务上验证模型准确率；

**📈 对比分析**

对比方法：AInfer Normal（关闭P/D多路复用）和SGLang全栈系统；同引擎safe‑PD、global‑Complete、global‑Enqueue、fine‑grained等对照；实验显示：单机场景下P/D多路复用相较于Normal减少7.1–22.5%完成时间，较SGLang降低24.8–32.9%；双机场景提升18–35%；细粒度segment进一步提升8.6–19.8%；TTFT变化不超过±5%；请求吞吐率提升12–46%；

**⚠️ 局限性**

局限性：仅适用于共享模型/KV的分布式MoE部署；跨阶通信排序需要已知交叉的集体组并保持所有rank一致；DeepEP状态隔离仅实现于DeepEP，未覆盖其他专家并行实现；实验仅在NVIDIA H20‑3E、BF16/FP8、两节点规模验证，未扩展至更大规模或不同硬件；不解决KV迁移、内存扩容等问题；在高度动态负载或多模型共享环境下需要进一步适配。

---

## 174. One Print, Many Moves: Monolithic Origami-inspired Folding Actuator for Composable Soft Multi-DoF Systems

**arXiv ID:** 2609.00751 | [PDF](https://arxiv.org/pdf/2609.00751v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 175. Joint Training Is Not Enough: Conditioned Cross-Granularity Training for Multimodal Document Understanding

**arXiv ID:** 2609.00756 | [PDF](https://arxiv.org/pdf/2609.00756v1)

**作者:** Chengguang Gan `[一作]` (Independent Researcher), Shiwen Ni `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在多模态文档理解任务中，对细粒度字段提取（点任务）与粗粒度文档级属性判断（行任务）的相互强化效果进行系统评估，提出并构建了Doc-MRE注释层，比较了单任务、混合联合训练、条件训练以及多种对照训练方案；

**💡 创新点**

创新点在于对MRE假设进行严格对照实验，证明混合联合训练并不产生强化，而条件训练能在两组数据集上实现互惠强化，并通过四个带对照的分析仪器剖析其内在机制；

**🔧 技术方法**

技术手段包括使用Qwen3‑VL‑8B‑Instruct模型的LoRA微调，利用提示中嵌入的“特权信息”进行条件训练，结合线性探针、梯度归因、输入干预和模态消融等四项分析仪器，对模型内部信息流进行定量评估；

**📊 数据集**

数据集涵盖三组文档：CORD（991份收据）、WildReceipt（400份收据）以及FUNSD（199份扫描业务表单），并在8B与4B两种模型规模上进行实验；

**📈 对比分析**

比较方法以平面提示下的点任务F1和行任务宏观准确率为指标，零样本、单任务、混合联合、条件训练四种方案交叉评估；结果显示条件训练在CORD和FUNSD上实现了对齐提升（点+0.5、行+4.8），而混合联合训练未出现任何强化；

**⚠️ 局限性**

局限性包括仅覆盖两类收据与一种表单数据集、仅使用LoRA微调和固定模型家族、测试集样本量有限、分析仪器存在提示格式共变效应、且条件训练需预先获得另一粒度的金标准标签。

---

## 176. HarmoCore: Functional Latent Diffusion for Sparse Reconstruction of Oscillatory Wave Fields

**arXiv ID:** 2609.00679 | [PDF](https://arxiv.org/pdf/2609.00679v1)

**作者:** Lihao Chen `[一作]` (Zhejiang University), Shikai Fang `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种用于稀疏传感下时间谐波波场重建的生成式框架，利用频率条件的函数 Tucker 隐空间与扩散后验采样；

**💡 创新点**

核心创新在于将复数波场映射为共享连续空间基底上的紧凑函数 Tucker 核，联合真实与虚部通道；使用频率条件扩散先验，并在核空间直接进行后验采样和物理方程残差引导，显著降低采样维度和计算成本；

**🔧 技术方法**

技术方法包括：函数 Tucker 分解、连续坐标基底网络、频率条件的 UNet 扩散模型、扩散后验采样 (DPS)、多线性解码器构造观测算子、可选的 PDE 残差引导；

**📊 数据集**

实验数据集：二维 Helmholtz、二维合成波场（基于射线模型）以及三维 Helmholtz；

**📈 对比分析**

与 LRTFR、FNO、F‑FNO、VoronoiCNN、DiffusionPDE 等基线比较，HarmoCore 在 1%–5% 传感密度下实现最低相对 L2 误差（尤其在 1%–2% 极稀疏时提升显著），且物理残差最小，显示出更好的物理一致性；

**⚠️ 局限性**

局限性包括：扩散先验对核尺寸的容量受限，随秩增大易出现性能退化；方法主要针对时间谐波波场，尚未证明对频率外推或不确定性校准的适用性；训练依赖大量标注数据，分布外推时需谨慎。

---

## 177. RW-LoRA: Communication-Efficient Decentralized LoRA Fine-Tuning via Random Walks

**arXiv ID:** 2609.00078 | [PDF](https://arxiv.org/pdf/2609.00078v1)

**作者:** Xingran Chen `[一作]` (Singapore University of Technology and Design), Salim El Rouayheb `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于随机游走的 LoRA 微调框架，在去中心化网络中只使用单一模型令牌顺序更新，从而避免全局同步。

**💡 创新点**

创新点在于消除多模型同步与聚合误差，采用单一令牌随机游走实现低通信与计算开销，并给出非凸目标的收敛保证。

**🔧 技术方法**

技术包括 LoRA 参数高效微调、随机游走学习、Metropolis–Hastings 采样、随机梯度下降、理论分析中的混合时间和非凸收敛证明。

**📊 数据集**

数据集：GLUE 基准下的 MRPC、QQP、QNLI、MNLI、SST‑2。

**📈 对比分析**

与基于 gossip 的去中心化 LoRA 进行对比，实验显示在准确率上基本相当或略优，同时通信成本约为 gossip 的 1/10，计算开销也显著降低。

**⚠️ 局限性**

局限性：收敛速度受随机游走混合时间限制，适合后台或低通信需求场景，对极端网络拓扑或大规模节点仍需进一步验证。

---

## 178. NeuroPriv: Adversarial Representation Learning for Privacy in Wearable EEG Systems

**arXiv ID:** 2609.00390 | [PDF](https://arxiv.org/pdf/2609.00390v1)

**作者:** Sarmistha Sarna Gomasta `[一作]` (University of Massachusetts Amherst), Prashant Shenoy `[通讯]` (University of Massachusetts Amherst)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文研究可穿戴EEG系统在传输压缩特征时的隐私泄露问题，并提出一种对抗式表示学习框架，在保留认知状态识别性能的同时显著抑制性别、年龄和受试者身份的推断。

**💡 创新点**

创新点在于：① 系统性量化压缩EEG特征对多重敏感属性的泄露；② 通过ANOVA分析揭示不同属性泄露来源于不同频谱和空间特征组；③ 对比简单防御（高斯噪声、特征屏蔽）并指出其属性依赖性弱点；④ 设计集成梯度逆向的对抗表示网络，能一次性抑制多属性信息并保持任务性能。

**🔧 技术方法**

技术方法包括：特征扩展（17维手工特征 + 所有两两乘积至153维）；编码器网络生成低维潜在表示；梯度逆向对抗学习三类私有头（性别、年龄、身份）；外部攻击者（SVM、RF、GB）评估泄露；对比基线的高斯噪声与特征屏蔽；使用平衡准确率衡量任务与隐私性能。

**📊 数据集**

使用 EEGMAT 数据集：36名受试者，基线与算术任务，17维手工频谱/空间特征，标签包含任务、性别、年龄组、身份。

**📈 对比分析**

比较方法：将原始特征、Gaussian噪声（σ=0.30）、特征屏蔽（Top‑8）与所提表示进行对齐。结果显示：任务平衡准确率从0.788降至0.781，性别推断从0.858降至0.563，年龄从0.789降至0.467，身份从0.692降至0.206，显著低于基线且保留高任务性能。基线防御在某些属性上效果不佳，尤其是年龄推断。

**⚠️ 局限性**

局限性包括：仅在单一实验室EEGMAT数据集（36人）评估，未验证在多样化人群或真实可穿戴部署中的泛化；只研究闭集身份识别，未考虑开放集或跨会话场景；任务仅为实验室算术分类，无法代表连续监测或多模态应用；攻击者假设拥有同一人群标记数据，现实中侧信道或弱侧信息的影响未评估；未在实际可穿戴设备上测量计算开销和能耗。

---

## 179. Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models

**arXiv ID:** 2609.00355 | [PDF](https://arxiv.org/pdf/2609.00355v1)

**作者:** Jungseob Lee `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为GLANCE的单步块式推断器，能够在冻结的视觉‑语言模型（VLM）上实现无损（lossless）加速；

**💡 创新点**

创新点在于将块扩散头（block‑diffusion head）直接读取目标模型的融合视觉‑语言隐藏状态，采用宽树验证（wide‑tree verification）一次性完成所有候选的验证，并通过精确比特级检查保证输出与原始贪婪推断完全一致；

**🔧 技术方法**

使用技术包括块扩散头（block‑diffusion）、祖先掩码（ancestor‑masked）目标验证、一次性候选树构建、熵驱动的接受长度模型以及对冻结目标的无监督自回归训练；

**📊 数据集**

训练数据来源于目标模型的贪婪输出，主要包含COCO Caption、TextVQA、InfographicVQA、DocVQA和ChartQA等任务的示例；

**📈 对比分析**

与EAGLE3‑VL、经典两模型推测、ViSpec、Medusa等现有方案对比，GLANCE在五大任务上平均提升了约2.93×的速度（对比自回归）且比生产版EAGLE3‑VL快约7.6%；在块长度上接受率可达2.7×，且在所有审计提示下实现比特级无损；

**⚠️ 局限性**

局限性包括：对自由生成文本（free‑running text）收益有限；性能高度依赖目标模型的下一词熵，低熵任务受益最多；需要大量目标模型贪婪数据进行训练；在非视觉‑语言或非文本任务中，熵模型和宽树策略的效果可能不如预期；

---

## 180. Few-Shot Out of Domain Intent Detection with Covariance Corrected Mahalanobis Distance

**arXiv ID:** 2609.00961 | [PDF](https://arxiv.org/pdf/2609.00961v1)

**作者:** Jayasimha Talur `[一作]` (Amazon), Paul Missault `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在少样本环境下改进 OOD 意图检测，提出使用鲁棒协方差估计器校正 Mahalanobis 距离。

**💡 创新点**

创新点在于将 Shrinkage、Ledoit‑Wolf、Van Ness 等协方差校正方法应用于 Mahalanobis 距离，使其在 5‑10‑shot 下显著提升性能。

**🔧 技术方法**

使用 RoBERTa 句子嵌入、最大似然估计、收缩估计、Ledoit‑Wolf 估计等协方差估算方法，并与能量、梯度范数、最大软化概率等基线进行对比。

**📊 数据集**

在 CLINC150、ROSTD、ROSTD‑COARSE、SNIPS 四个数据集上进行评估。

**📈 对比分析**

相较于传统 MLE 及其他基线，校正后的 Mahalanobis 在大多数数据集的 AUC、PR‑ROC 提升 5‑10% 以上，尤其在 5‑shot 场景中 Shrinkage 方法表现最佳。

**⚠️ 局限性**

局限性包括：随着样本量或意图类别增加，校正效果趋于 MLE；对高维特征仍需足够样本；未针对多模态或更复杂 OOD 生成方式进行实验。

---

## 181. Counterfactual Closing-Acceleration Risk: An Anticipatory Surrogate Safety Measure for the Blind Region of Car-Following

**arXiv ID:** 2609.00370 | [PDF](https://arxiv.org/pdf/2609.00370v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 182. Invalidation Contracts for Cross-Episode Agent Memory

**arXiv ID:** 2609.00243 | [PDF](https://arxiv.org/pdf/2609.00243v1)

**作者:** Michael Wu `[一作]` (South Dakota State University), Arquimedes Canedo `[通讯]` (Siemens Digital Industries Software)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出并实现了一种名为“无效化合同”的协议层，向LLM代理的恢复建议中嵌入版本戳、缓存提示和依赖向量，使代理能够在服务器端数据漂移时精准失效缓存，避免重推导；

**💡 创新点**

创新点在于将无效化合同拆分为可控的有效性（协议层）与合规性（模型层）两因子，并通过多级协议（L0–L6）提供从单表版本到多表依赖向量、知识图谱以及规则指纹的完整失效信息；

**🔧 技术方法**

技术上使用LLM代理与API服务器交互，服务器维护引用表、派生规则、策略和恢复建议；协议层通过在响应中添加缓存提示、表版本、依赖向量、图谱和规则哈希；评估使用多模型、三种服务路径、七种模型和约9,400个实验片段；

**📊 数据集**

实验数据集包含两个合成域：Acme账单API（36期）和食谱转换API（48期），每个域设定漂移事件（表版本变更、代码表轮转等）以及规则声明流；

**📈 对比分析**

通过比较A0（无记忆）、A1（简单缓存）到A2D（行级失效）等不同实验臂，测量首试率、token节省、重试次数和失效精度；行级失效可将合规性提升至66.7个百分点、token节省29–33%，而表级失效则可能将首试率降为0%；

**⚠️ 局限性**

局限性包括仅在合成API上验证，未覆盖真实生产API的复杂模式与漂移；仅考察单表或两表依赖，未分离多表向量的单独效果；受限于单一提示与任务族，模型偏差与输入模式保守性未被完全普适化；

---

## 183. MUSES: A Benchmark for Prospective Intellectual-Roots Retrieval

**arXiv ID:** 2609.00313 | [PDF](https://arxiv.org/pdf/2609.00313v1)

**作者:** Rohan Pandey `[一作]` (University of Massachusetts), Hong Yu `[通讯]` (University of Massachusetts)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 MUSES 基准，用于评估研究者在未来论文中检索先前工作的能力，并配合 CiteRoots 两层根标注（修辞层和作者确认层），对检索难度进行熟悉度与功能分层。

**💡 创新点**

首次在约 2.33M 论文的固定检索空间上构建面向未来、作者条件化的检索基准，并引入可扩展的修辞根层和有限的作者确认层，形成多维度难度梯度。

**🔧 技术方法**

使用 SPECTER2 等科学论文编码器构建多中心检索器 MC‑SPECTER2；利用 Qwen3‑8B LoRA 进行修辞根判别；通过作者问卷与人工裁定构建作者确认根。

**📊 数据集**

基于 Semantic Scholar Open Research Corpus (S2ORC) 提取的 2.33M 论文文本池，形成 1.04M 作者条件化实例，CiteRoots 包含约 1,518 个作者确认的生成启发对，修辞层覆盖数百万引文实例。

**📈 对比分析**

在 9 类检索方法中，MC‑SPECTER2 表现最佳；Hit@100 在 CiteNext 0.534 降至 CiteNew 0.424，修辞 CiteNew 0.205，作者确认 CiteNew 0.171，显示难度显著提升，约一半宽层实例在 K=1000 仍未检索到。

**⚠️ 局限性**

限制包括：基准仅覆盖现有文本检索，缺乏实时更新；作者确认层样本有限且可能存在主观偏差；在作者确认层上的性能低且与修辞层差异大，说明根标注与检索模型仍需进一步改进。

---

## 184. HarnessEvolve: Learning from Reference Trajectories for Reliable Agent Self-Evolution

**arXiv ID:** 2609.00829 | [PDF](https://arxiv.org/pdf/2609.00829v1)

**作者:** Wen Jiang `[一作]` (Huawei Technologies), Fangming Li `[通讯]` (Huawei Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了HarnessEvolve框架，实现自演化代理通过参考轨迹学习、错误诊断、质量门控与性能门控来持续优化其 harness（提示、技能、工具与执行逻辑）

**💡 创新点**

核心创新在于①解耦执行与优化流程，②使用参考轨迹进行根因错误定位，③通过错误聚类提炼系统性失败模式，④双重门控防止捷径学习与灾难性遗忘

**🔧 技术方法**

采用多模块协作（执行、评估、优化、门控）技术，利用大语言模型生成轨迹、评估、诊断与更新，聚类分析错误，门控机制结合数据泄露检测与性能阈值

**📊 数据集**

在开放式QA（SearchQA、OfficeQA、SpreadsheetBench）与企业场景（CloudCoreNetwork‑QA、Wireless‑QA）两组数据集上进行评估

**📈 对比分析**

与GEPA、ACE、SkillOpt等单一维度自演化基线相比，HarnessEvolve在所有五个基准上均实现最高准确率，单个基准最高提升超过20个百分点，且在跨框架迁移中保持性能优势

**⚠️ 局限性**

仍需处理参考轨迹生成失败导致的局限、门控阈值的手动设定以及对更复杂多模态任务的适应性验证

---

## 185. DISTAL: Distillation and Self-Supervised Pretraining for Structure-Agnostic Materials Property Prediction

**arXiv ID:** 2609.00059 | [PDF](https://arxiv.org/pdf/2609.00059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 186. Forbid Your Attention: Fooling Multimodal Large Language Models by Selectively Removing Intrinsic Focus in Spectral Domain

**arXiv ID:** 2609.00788 | [PDF](https://arxiv.org/pdf/2609.00788v1)

**作者:** Daizong Liu `[一作]` (Wuhan University), Yew-Soon Ong `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多模态大语言模型（MLLM）对图像频域中相位谱的敏感性，并基于此提出了相位感知的结构化对抗攻击框架；

**💡 创新点**

创新点在于将相位谱视为视觉结构信息，通过边缘检测得到相位主导区域并局部化扰动，同时引入可训练的辅助对抗提示，进一步引导模型注意力，提升攻击效果；

**🔧 技术方法**

采用离散傅里叶变换（DFT）提取相位/幅度谱，利用边缘检测+自适应阈值得到结构掩模；使用任务损失、相位正则化和模式一致性损失进行对抗优化；并加入辅助提示学习模块进行对抗提示的梯度更新；

**📊 数据集**

实验数据集包括 ImageNet、SVIT、DALLE 等公开数据集；对6个开源 MLLM（BLIP‑2、LLaVA‑1.5、Flamingo、MiniGPT‑4、Qwen2‑VL、Intern‑VL）以及5个闭源模型（Claude‑3.5/3.7、GPT‑4o/4.1、Gemini‑2.0）进行评估；

**📈 对比分析**

与APGD、MF‑Att、CroPA、MABA、VMA等现有攻击方法对比，本文方法在多模型、多任务上均显著提升攻击成功率（例如在 ImageNet 上目标成功率从约70% 提升至 88% 以上），在黑盒和防御（Fine‑tuning、频域过滤、随机变换、图像净化）场景下仍保持领先；

**⚠️ 局限性**

局限性包括：对极其复杂或低对比、微小目标、文本密集、OCR等场景下相位信息不够充分，导致掩模不准确或攻击效果下降；方法依赖白盒梯度信息，黑盒性能受限；

---

## 187. Conditional Flow Matching for ML-Based Inverse Design Problems

**arXiv ID:** 2609.00863 | [PDF](https://arxiv.org/pdf/2609.00863v1)

**作者:** Juliana Felder `[一作]` (ETH Zürich), Mark Fuge `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 EngiOpt 框架中加入条件流匹配（CFM）模型，用作工程逆向设计任务的预热起点。

**💡 创新点**

创新点在于利用 CFM 替代传统扩散模型的随机去噪，直接学习向量场回归，显著减少网络评估次数、提升生成吞吐量，并在关键指标（COG、FOG、MMD、体积分数误差）上优于扩散模型和 cGAN。

**🔧 技术方法**

采用的技术包括条件流匹配（ICFM）、共享的 2D 条件 U‑Net 结构、Euler/Midpoint/RK4 ODE 求解器、以及与 DDPM 与 cGAN 的对比实验。

**📊 数据集**

使用 EngiBench 提供的两组 2D 数据集：结构合规性优化（100×50 网格）和热学合规性优化（101×101 网格）。

**📈 对比分析**

通过统一数据拆分、验证集检查点选择、相同的下游优化器和评价指标（COG、FOG、MMD、DPP、体积分数误差）进行比较；CFM 在所有主指标上均优于扩散模型，吞吐量提升约 32–66 倍，且体积分数误差最低。

**⚠️ 局限性**

局限性包括：实验仅在 2D 任务和单一 U‑Net 结构下进行，未验证 3D、多物理或更高维条件空间；Diffusion 方案的体积分数偏差原因尚未完全解析。

---

## 188. Online Self-Weighted Fine-Tuning

**arXiv ID:** 2609.00734 | [PDF](https://arxiv.org/pdf/2609.00734v1)

**作者:** Haiquan Wen `[一作]` (University of Liverpool), Guangliang Cheng `[通讯]` (University of Liverpool)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种在线自加权微调方法OSW-FT，利用少量推理rollout动态估计模型对每条专家轨迹的成功率，并按(1-成功率)对标准SFT损失进行加权；

**💡 创新点**

在保持SFT梯度方向不变的前提下，引入轨迹级成功率权重，实现对样本学习强度的自适应调节，兼顾稳定性与探索性，并提供理论证明其无偏性与收敛性；

**🔧 技术方法**

利用轨迹级蒙特卡洛估计、控制变量方差降低、梯度级SFT与RL对齐、Math-Verify二元奖励、Top_p/Top_k采样等技术；

**📊 数据集**

训练集为10k多选题（AM‑Qwen3‑Distilled），评测集包括AMC、AIME、MATH‑500、GPQA‑Diamond等；

**📈 对比分析**

与标准SFT和GRPO在Qwen3 0.6B、1.7B、4B模型上进行对比，采用Pass@1/Pass@16指标；OSW‑FT在小中模型上普遍优于SFT，K=2即可获得大部分收益，计算成本低；在部分任务与GRPO相当或略逊；

**⚠️ 局限性**

仅适用于二元可验证奖励和高质量专家轨迹，未验证连续奖励或开放式生成；实验局限于Qwen3数学推理任务，未扩展至代码生成等其他场景。

---

## 189. REVISE: Validity-Guided Recovery for Online Revisions in Agent Workflows

**arXiv ID:** 2609.00643 | [PDF](https://arxiv.org/pdf/2609.00643v1)

**作者:** Ruoling Qi `[一作]` (Institute of Artificial Intelligence China Telecom), Yirui Liu `[通讯]` (Institute of Artificial Intelligence China Telecom)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于有效性推断的运行时，能够在代理工作流执行过程中实时响应用户修订，精确识别哪些已完成或正在执行的节点仍然有效，哪些需要中止或重算，从而实现细粒度恢复。

**💡 创新点**

创新点在于将修订事件与数据与控制依赖的动态影子相交，构造“影响集”，并为每个节点分配取消、避免、继续、重算或复用等生命周期动作；同时提供一种失效闭合的提交验证协议，确保仅安全复用且避免旧状态泄漏。

**🔧 技术方法**

技术实现包括：可观测的读写代理记录路径依赖；差分合并与结构化差分推导 delta；基于工作流 DAG 的传播算法生成影响集；生命周期动作调度与执行；提交时的证书校验与效果重验证；对 LangGraph 与 LLMCompiler 的 Python 适配器。

**📊 数据集**

使用的数据集与工作流包括：真实的 SWE‑chat 编码代理会话（约5,800 条记录）用于分析修订机会；LangGraph 与 LLMCompiler 两个基准工作流；SWE‑Review‑Traj 用于构造可复现的仓库重放；以及在 GPU 集群上生成的 15,000 条高负载服务实验。

**📈 对比分析**

与基准（全重启、最早冲突后缀重算、动态后缀）以及线上“等候至完成”方式比较；在 LangGraph 上模型调用下降 56%，在 LLMCompiler 上下降 40%；在高并发服务场景下，修订到完成的 token 数减少 13%；在拥塞区间，SLO goodput 提升 3–5%。

**⚠️ 局限性**

局限性包括：需要细粒度的读写与控制依赖记录，若缺失则退回到后缀或完整重启；当前实现仅在单机进程内验证有效性，未覆盖分布式共识或崩溃恢复；对动态生成或外部不可追踪的效果处理仍不完善。

---

## 190. Scene Graph-based Driving Scenario Extraction for Automotive Egocentric Datasets

**arXiv ID:** 2609.00333 | [PDF](https://arxiv.org/pdf/2609.00333v1)

**作者:** Stefan Ramdhan `[一作]` (McMaster University), Mark Lawford `[通讯]` (McMaster University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出基于场景图和线性时序逻辑（LTL）从单目摄像头记录的无人驾驶数据中自动提取并定位交通场景实例。

**💡 创新点**

创新点在于将视频导出的语义场景图与LTL结合，实现了在未标注数据中按天气、路况等语义条件抽取场景，同时保持可解释性与确定性。

**🔧 技术方法**

技术包括RoadScene2Vec场景图生成、Kalman滤波+匈牙利算法多目标跟踪、MLLM视觉问答、LTL模型检验（SceneFlow）以及手工定义的原子谓词。

**📊 数据集**

使用Argoverse 2 15秒驾驶日志（850条）以及7段包含停靠校车的视频作为评测数据集。

**📈 对比分析**

与基于轨迹/地图规则的基准比较，Level II/III场景图实现了90%以上的F1（时间/区间），在长追随、右侧切入等典型场景上达成与规则基线相当或略优的准确率，且能识别规则无法覆盖的天气/停靠校车场景。

**⚠️ 局限性**

局限主要来自场景图语义粗糙、目标检测/跟踪碎片化导致误检/漏检，MLLM单帧查询缺乏时序上下文，以及对更复杂语义和多模态感知的依赖。

---

## 191. Low-Quality Face Recognition using Center Aligned Representations and Local Margin Constraints

**arXiv ID:** 2609.01014 | [PDF](https://arxiv.org/pdf/2609.01014v1)

**作者:** Vedat Can Dilaver `[一作]` (University of Nebraska Lincoln), Benjamin S. Riggan `[通讯]` (University of Nebraska Lincoln)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对低质量人脸识别难题，提出统一框架来提升在低质量和高质量数据上的识别性能。

**💡 创新点**

核心创新点包括：1）局部概率边际（LPM）利用类别中心角度关系动态估计样本难度并自适应调整边际；2）嵌套注意力模块（NAM）在低秩适配器中嵌入自注意力，实现令牌级的上下文感知适配；3）质量门控协议（QGP）在推理时依据图像质量调节适配器贡献，使单模型同时适应全质量范围。

**🔧 技术方法**

主要技术有：ViT-Base 视觉 Transformer、低秩自适配器 LoRA 变体、自注意力机制、Angular Margin Softmax（ArcFace/ CosFace/AdaFace）、Q-Align 视觉语言图像质量评估、AdamW 优化器、数据增强。

**📊 数据集**

使用的数据集包括：高质量训练集 CASIA-WebFace；低质量微调集 TinyFace、SurvFace；评估集 LFW、CFP-FP、CPLFW、AgeDB、IJB-B、IJB-C、TinyFace、SurvFace。

**📈 对比分析**

与基线方法（CosFace、ArcFace、MagFace、AdaFace、LoRA、PiSSA、DoRA 等）对比，NAM+LPM 在高质量基准上保持与全微调相近的精度，同时在 TinyFace、SurvFace 等低质量基准上实现 75%+ Rank‑1/TPIR 的提升，明显优于现有低质量专用方法。

**⚠️ 局限性**

局限性包括：LPM 在某些基准上提升有限、参数选择（如边际偏置、残差尺度）需经验调优、QGP 依赖单一质量评估器 Q‑Align、未验证在更大规模多模态低质量数据库上的鲁棒性。

---

## 192. Post-hoc Alignment of LLM-judges to Human Judgment Distribution

**arXiv ID:** 2609.01073 | [PDF](https://arxiv.org/pdf/2609.01073v1)

**作者:** Sebastian Steindl `[一作]` (Amazon), Diego Marcheggiani `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估LLM作为评估者（LLMaJ）在预测硬标签与软标签（人类标签分布）时的表现，并提出基于熵分层的轻量级后置对齐方法（NAPHA）来提升软标签预测。

**💡 创新点**

创新点在于①首次量化LLMaJ在软标签预测上的显著不足；②提出Entropy‑Aware Post‑Hoc Alignment（NAPHA），通过熵分层与专属MLP对齐模型，显著提高高熵实例软标签的准确性。

**🔧 技术方法**

使用技术包括LLMaJ框架、温度采样的SimAnn、软标签Prompt、熵分层判别、三类专属对齐MLP（训练时使用KL散度），并与温度缩放、Dirichlet校准等对齐方法对比。

**📊 数据集**

实验数据集涵盖SummEval、TopicalChat、ChaosNLI、DynaSent、Anecdotes，均包含多注释、不同熵水平的样本。

**📈 对比分析**

硬标签采用宏F1、Kendall τ等评估；软标签采用DistCE、JSD。结果显示LLMaJ在硬标签上接近人类水平，但软标签预测差；使用NAPHA后，DistCE和JSD均下降，尤其在高熵实例中几乎逼近人类表现。

**⚠️ 局限性**

局限性包括软标签样本稀缺导致估计不稳定；熵分层依据预测熵可能误分，无法完全解耦模型不确定性与人类分歧；未利用模型内部信息；对齐网络结构与超参未进行深度调优；实验仅在有限训练数据和特定数据集上验证。

---

## 193. Investigating Hyperparameter Optimization and Transferability for ES-HyperNEAT: A TPE Approach

**arXiv ID:** 2609.00449 | [PDF](https://arxiv.org/pdf/2609.00449v1)

**作者:** Romain Claret `[一作]` (Information Management Institute), Kilian Stoffel `[通讯]` (Information Management Institute)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过Tree-structured Parzen Estimator（TPE）对ES-HyperNEAT算法的超参数进行全局优化，并验证了该优化结果在MNIST、Fashion‑MNIST以及多种逻辑运算任务中的迁移效果。

**💡 创新点**

创新点包括：①在ES‑HyperNEAT的3 亿+超参数搜索空间中首次实现TPE优化；②系统性评估了从复杂任务（MNIST）迁移到相似或更简单任务的可转移性；③通过Bug检测与严格的验证流程，提升实验结果的可靠性。

**🔧 技术方法**

主要技术手段有：Tree-structured Parzen Estimator（TPE）超参数优化、ES‑HyperNEAT神经进化、随机搜索对照、Optuna、Python + Pureples框架、统计检验（t检验、Cohen’s d）。

**📊 数据集**

使用的数据集包括：MNIST（手写数字分类）、Fashion‑MNIST（服饰图像分类）以及六种逻辑运算（XOR、OR、AND、NOR、XNOR、NAND）。

**📈 对比分析**

通过对比随机搜索（30次/292次）与TPE搜索（2013次）以及TPE-最佳配置，实验表明：在MNIST上TPE平均/最佳准确率分别提升至约28%/29%（相较随机搜索的≈11%），在Fashion‑MNIST上平均提升至20%（相较随机搜索的≈12%），在逻辑运算中对OR、NAND、XOR等任务显著优于随机搜索，效果量大。所有显著提升均伴随p值<0.01及Cohen’s d≥1。

**⚠️ 局限性**

局限性包括：①搜索空间受限于预定义的16维参数，未覆盖更细粒度或更多潜在参数；②TPE在极高维空间的效率仍有限；③迁移效果在部分简单逻辑任务不稳定，说明迁移需要考虑任务相似度；④实验受并行代码bug影响，虽然已通过验证修正，但对某些任务的基线估计仍有一定不确定性；⑤研究仅在少数任务上验证，尚未证实对更大规模或不同领域任务的普适性。

---

## 194. LLM-as-a-Demographic: Whom Sociodemographic Prompting Helps, and Whom It Hurts

**arXiv ID:** 2609.00222 | [PDF](https://arxiv.org/pdf/2609.00222v1)

**作者:** Daniela Occhipinti `[一作]` (Fondazione Bruno Kessler), Marco Guerini `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 1336 | [OpenAlex ID](https://openalex.org/A5072659160)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了给大语言模型(LLM)判定者加入人口统计条件后，其对特定注释者群体的判断是否更贴近真实群体分布，并在23个开源LLM上验证。

**💡 创新点**

创新点在于提出基于分布的评估框架，区分单属性与交叉属性条件化，并揭示指令调优导致的多数群体偏好增强与少数群体偏差放大的不对称效应。

**🔧 技术方法**

主要技术包括：利用对数概率读取模型在五点等级上的分布，使用Earth Mover's Distance（EMD）评估分布相似度，计算s和Δs来量化条件化效应，并对比基线与指令调优版本以及不同模型规模的表现。

**📊 数据集**

使用DeMo数据集，包含三项主观任务（Intimacy、Offensiveness、Politeness），并基于自报的性别、年龄、种族、教育四维信息构造人类标签分布。

**📈 对比分析**

通过将每个模型的预测分布与对应群体的人类标签分布比较，计算s（相似度）和Δs（条件化效应），结果显示：未条件化模型偏向白人和受大学教育群体；指令调优的条件化对多数群体有正面影响但对少数群体产生负面偏差；模型规模与条件化效应无明显相关性。

**⚠️ 局限性**

局限性包括：仅在三项英文主观任务和单一数据集上实验，交叉属性样本稀疏导致分布估计不稳，且可能受到预训练数据泄漏的影响，结果不一定能推广至其他语言或注释框架。

---

## 195. ChatDev 2.0: A No-Code Multi-Agent Platform for Developing Everything

**arXiv ID:** 2609.00714 | [PDF](https://arxiv.org/pdf/2609.00714v1)

**作者:** Yufan Dang `[一作]` (Tsinghua University), Zhiyuan Liu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了DevAll无代码平台，用于构建、执行和检查多代理系统（MAS），让非程序员也能通过可视化图形化界面完成复杂任务的自动化流程；

**💡 创新点**

核心创新点包括：①将语义边抽象为可声明的可执行图，解耦数据流与控制流；②设计了周期感知动态执行拓扑（CADET）算法，能够在同一框架内支持任意循环和嵌套反馈循环；③将上述抽象与可视化无代码编辑器无缝结合，实现真正的“无代码”MAS开发；

**🔧 技术方法**

采用的技术主要有：基于LLM的多代理节点框架、MAS编译引擎将YAML描述编译为可执行图、CADET调度算法、语义边的激活与数据流策略、无代码可视化界面、Python SDK与实验实验平台；

**📊 数据集**

在评测中使用了MatPlotBench、DeepResearchBench和SRDD三大基准数据集，用于分别检验科学可视化、深度研究报告生成和软件开发三种场景的性能；

**📈 对比分析**

通过将原有基准系统（CoDA、Enterprise Deep Research、ChatDev 1.0）在DevAll中以声明式图谱实现，比较结果显示DevAll在大多数指标上与原系统持平甚至略胜一筹（例如MatPlotBench的整体得分从0.7130提升至0.7950），且运行时开销仅为数十毫秒，远低于纯LLM推理延迟；

**⚠️ 局限性**

局限性包括：仍需用户手动设计工作流结构、角色与控制条件，无法完全自动化流程设计；图抽象与组件生态尚未覆盖所有领域特定交互模式；随着工作流规模扩大，视觉组织、模块化维护、版本演进和经验迁移仍是待完善的方向。

---

## 196. NeuroGraph: An AI Graph-Driven Neuro-Symbolic Framework for Explainable Threat Reasoning in Advanced Manufacturing

**arXiv ID:** 2609.00604 | [PDF](https://arxiv.org/pdf/2609.00604v1)

**作者:** Padmeswari Nandiya `[一作]` (Edith Cowan University), Helge Janicke `[通讯]` (Edith Cowan University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出NeuroGraph框架，结合知识图检索与大型语言模型实现可解释的网络攻击路径推理。

**💡 创新点**

创新在于将符号化的Cypher查询生成与嵌入辅助恢复结合，形成“symbolic-first”神经符号检索，并通过双LLM实现全流程可追溯。

**🔧 技术方法**

采用知识图（BRIDG-ICS）、双大型语言模型（Llama-3.1-8B）、向量检索（MiniLM-L6-v2）、图数据库Neo4j和Prompt工程。

**📊 数据集**

使用公开的CVE/CWE/CAPEC/MITRE ATT&CK等来源构建的BRIDG-ICS知识图，及CTI-Benchmark和自制多跳问答数据集。

**📈 对比分析**

与基线KG-RAG、Fine‑Tuned KG-RAG、Embedding‑Fallback KG-RAG等进行对比，在CTI-RCM和CTI-ATE任务上提升准确率至90%以上，且降低幻觉率、提升多跳推理成功率，鲁棒性略低于部分对抗实验。

**⚠️ 局限性**

局限在于依赖已构建的知识图、无法自动更新图谱、对极端多跳或非canonical查询的恢复仍受限，且对抗鲁棒性仍有提升空间。

---

## 197. DNC-IMM: Early Lane-Change Intention Recognition via Neural Calibration Based on Driving Context Information

**arXiv ID:** 2609.01120 | [PDF](https://arxiv.org/pdf/2609.01120v1)

**作者:** Woong-Chan Byun `[一作]` (Korea Advanced Institute of Science and Technology), Seung-Hyun Kong `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种双重神经校准互相干预多模型（DNC-IMM），用于在1~3秒前提前识别车辆的保持车道、左侧变道或右侧变道意图。

**💡 创新点**

创新点在于通过神经网络仅校准IMM的转移概率矩阵和测量似然，使模型既保留物理可解释的概率结构，又能自适应动态交通环境。

**🔧 技术方法**

采用的技术包括多层感知器（MLP）提取驾驶环境特征、对IMM的转移概率和似然进行神经校准、卡尔曼滤波器实现状态估计，以及后验对齐损失训练模型。

**📊 数据集**

实验数据集为德国高速公路航拍轨迹数据库highD，包含车辆位置、速度、加速度、车道信息等特征。

**📈 对比分析**

与Shi等、Mozaffari等、Liu等方法对比，DNC-IMM在1~3秒预测窗口的宏观F1平均值为0.9366，在更难的2–3秒区间获得最高的宏观F1 0.9185，显示出更早且更稳定的意图识别性能。

**⚠️ 局限性**

局限性包括仅在高速公路场景验证，缺乏对城市道路或复杂交通情境的泛化评估；模型依赖大量标注轨迹，且神经校准可能对不同数据分布产生过拟合风险。

---

## 198. Modelpedia: A Catalog of Model Findings for the Meta-Science of AI

**arXiv ID:** 2609.01090 | [PDF](https://arxiv.org/pdf/2609.01090v1)

**作者:** Franciszek Bernat `[一作]` (Centre for Credible AI), Przemysław Biecek `[通讯]` (Centre for Credible AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了自动化、LLM辅助的框架 Modelpedia，用于从学术论文中提取关于模型的发现，并将其组织成可搜索的知识目录；

**💡 创新点**

首次提出统一结构化存储模型发现的知识库，并通过 LLM 自动抽取、验证和链接发现，为模型科学和 AI 元科学提供元分析工具；

**🔧 技术方法**

采用大型语言模型 Qwen3.8‑27B 进行发现抽取、Claude Opus 5 进行验证，并使用 OpenReview API 采集论文、轻量化评分系统以及自定义数据库结构；

**📊 数据集**

主要使用 ICLR 2024 与 ICLR 2025 会议论文作为数据来源，抽取发现后涉及 ImageNet、COCO 等视觉基准及多种语言模型评测数据集；

**📈 对比分析**

通过比较发现中使用的模型、数据集比例以及证据类型（相关性 vs 观察/干预），评估社区关注度；抽取结果在验证阶段达 100% 正确率（仅关键指标存在约 20% 的差异），显示方法可行；

**⚠️ 局限性**

局限性包括：覆盖范围仅限 ICLR 论文，抽取过程依赖 LLM 可能产生漏报/误报；未验证发现的真实性，仅记录作者主张；且对方法正确性未做深入评估。

---

## 199. Lagged Coupling: Internal Representations Become Readable Before They Become Causal

**arXiv ID:** 2609.01048 | [PDF](https://arxiv.org/pdf/2609.01048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 200. Location-Aware Language Models via Secondary Embeddings

**arXiv ID:** 2609.00454 | [PDF](https://arxiv.org/pdf/2609.00454v1)

**作者:** Gokul Srinivasagan `[一作]` (AImotion Bavaria, Technische Hochschule Ingolstadt), Munir Georges `[通讯]` (AImotion Bavaria, Technische Hochschule Ingolstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在预训练语言模型输入中插入特殊 token 和对应地点的经纬度信息，并采用位置掩码进行持续预训练，构建了轻量级的地理空间注入方法。

**💡 创新点**

该方法不需要改 tokenizer、增大词表或大规模再训练，仅通过输入增强和针对性掩码实现地理嵌入，同时保持模型原有语义与句法能力。

**🔧 技术方法**

使用了二次嵌入（geographic secondary embeddings）、特殊 token（<loc>、</loc>、<l_sep>）、位置掩码策略、持续预训练的 MLM 目标以及与地理距离相关的评估指标。

**📊 数据集**

基于德国长途火车 GTFS 生成的合成数据集，包含 820 个站点名称的 5,965 条文本样本，每个地点附带精确的纬度和经度。

**📈 对比分析**

在 GLUE 基准上，模型保持或略有提升；在地理距离相关性上，通过欧氏距离和余弦相似度与真实 Haversine 距离计算 Pearson/Spearman 相关系数，尤其在大模型中显著提高（如 BERT_large 的 Pearson r 从 0.297 提升至 0.550）。

**⚠️ 局限性**

方法仅为后置适配，缺乏在大规模预训练阶段直接注入的深度学习；对多语种模型的效果有限；仅适用于编码器结构，未扩展到生成式模型；小模型的地理嵌入效果有限。

---

## 201. HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference

**arXiv ID:** 2609.00450 | [PDF](https://arxiv.org/pdf/2609.00450v1)

**作者:** Chun-Ting Chen `[一作]` (Cornell University), Jae-sun Seo `[通讯]` (Cornell University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了层次化块量化（HBQ）方案，用两级量化和显著量子缩放显著降低LLM推理的位宽，同时保持高精度。

**💡 创新点**

创新点在于：①利用大块尺寸实现高硬件效率；②引入低开销显著量子缩放（SIG）实现第二级量化；③联合KV缓存和累加量化提升端到端低精度推理。

**🔧 技术方法**

采用块量化、浮点/PoT缩放、显著量子缩放、权重/激活/KV/累加量化以及对28nm ASIC的系统级设计和验证。

**📊 数据集**

使用Llama 2/3、Qwen2.5、Mixtral等开源LLM，在WikiText‑2、Winogrande、PIQA、MMLU、HumanEval、GSM8K等数据集上评估精度。

**📈 对比分析**

与现有BQM（MXFP、NVFP、Amove、VSQ、MicroExponent）及WoQ（AxCore）做对比，HBQ在保持相同或更低PPL的前提下，实现了2.3×/4.6×的PE级面积/能效提升，系统级能耗降低1.6–3.3×，速度提升1.5–3×。

**⚠️ 局限性**

局限在于需要更高激活位宽（5比4位）才能获得最佳精度；两级量化增加硬件设计复杂度；在极端KV异常值场景下仍需更细粒度微块来进一步提升鲁棒性。

---

## 202. Zero-Shot Respiratory Sound Classification through LLM-Augmented Audio-Text Alignment

**arXiv ID:** 2609.00055 | [PDF](https://arxiv.org/pdf/2609.00055v1)

**作者:** Mustafa Talha İlerisoy `[一作]` (Eindhoven University of Technology), Aaqib Saeed `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1265 | [OpenAlex ID](https://openalex.org/A5011960578)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出 REACH 框架，通过 LLM 生成结构化报告并与现有音频编码器进行对齐，使得单模态呼吸音编码器在零样本下即可进行临床诊断。

**💡 创新点**

创新点包括：①使用医学 LLM 将离散元数据转化为语义丰富的文本锚点；②在对齐过程中引入 sigmoid 对比损失、相似度感知负采样以及原始自监督重建正则化，以兼顾跨模态对齐与原始音频特征的保留；③实现了无对齐音频报告数据即可进行零样本推理的多模态模型。

**🔧 技术方法**

技术手段包括：医学 LLM（如 GPT‑4）生成报告；SigLIP‑style sigmoid 对比损失；FAISS 基础的相似度感知负采样；掩码重建（MSE）正则化；轻量级投影头将音频与文本映射至共享空间。

**📊 数据集**

使用 6 个公开呼吸音数据集，共 9 个二分类任务（含 COPD 阶段分类），包括在域与跨域的评测。

**📈 对比分析**

与基线相比，REACH 在 9 项任务上平均零样本 AUC 为 61.3%，明显高于 CLAP（51.4%）和 Qwen2‑Audio（54.9%）；在线性探测上实现 71.6% 的平均 AUC，优于 100% 规模更大的基线模型。

**⚠️ 局限性**

局限性包括：依赖 LLM 生成的报告可能产生偏差或错误；对不同呼吸音种类的覆盖仍有限；负采样参数及对比损失权重需人工调优；在少样本或极其细微的音频差异上表现仍不理想。

---

## 203. Characterizing the Scalability and Performance of Large-Scale AI Training Under Multi-Tenancy

**arXiv ID:** 2609.00817 | [PDF](https://arxiv.org/pdf/2609.00817v1)

**作者:** Jacopo Raffi `[一作]` (University of Trento), Flavio Vella `[通讯]` (University of Trento)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对大规模AI训练的可扩展性与性能进行系统化评估，基于DLNetBench框架在单机、机架和超级计算机层面对多种并行化策略（DP、FSDP、PP、TP、EP等）进行基准测试，探究网络拓扑、作业放置与多租户干扰对训练效率的影响。

**💡 创新点**

创新点包括①提出统一的实验方法和DLNetBench框架，能够精确控制通信模式与计算时间；②系统性评估从2400 GPUs到单机的多层级网络，量化不同并行化策略在多租户环境中的通信敏感度和规模效能；③揭示网络拥塞与系统噪声在不同拓扑和作业放置下的异质影响，并给出基于拓扑的放置策略对降低延迟的有效性。

**🔧 技术方法**

使用NCCL、RCCL、oneCCL等通信库，结合roofline模型的计算时延估计，DLNetBench框架直接发出特定collective操作，并通过多租户噪声模型模拟共享环境；实验平台涵盖Alps、Leonardo、LUMI、JUPITER、NVL72 GB300和DGX A100等系统。

**📊 数据集**

实验主要基于Vision Transformer ViT‑H、LLaMA3‑8B/70B、Minerva‑7B、Mixtral‑8×7B等模型的参数配置，采用人工设置的批大小与微批次进行训练基准，未使用真实训练数据集。

**📈 对比分析**

通过对基线与并发实验的吞吐量、通信占比、并行效率和慢速比进行量化比较，结果显示单机/机架内通信瓶颈低，近乎完美伸缩；超级机上不同并行化策略表现差异明显，纯DP受通信限制，混合策略在内部网络通信下仍保持高效；多租户环境中跨组通信受拥塞显著影响，拓扑感知放置可显著降低延迟。

**⚠️ 局限性**

局限性在于仅评估了固定参数的少量模型，未覆盖不同规模和参数的计算‑通信平衡；对NCCL/RCCL收敛行为未深入分析；实验规模受限于可用集群，未覆盖更大规模的跨组竞争；模型与策略的普适性需进一步验证。

---

## 204. SoK: Motion Data Privacy in Extended Reality

**arXiv ID:** 2609.00711 | [PDF](https://arxiv.org/pdf/2609.00711v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 205. DK-GBMKKM: Dynamic Kernel-Space Granular-Ball Multiple Kernel $k$-Means Clustering

**arXiv ID:** 2609.00647 | [PDF](https://arxiv.org/pdf/2609.00647v1)

**作者:** Xiaoyu Lian `[一作]` (Chongqing University of Posts and Telecommunications), Xuzhao Xiang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种动态核空间颗粒球多核k均值聚类方法（DK-GBMKKM）

**💡 创新点**

创新点在于在当前融合核空间中动态生成并更新颗粒球，解决传统颗粒球在输入空间与核空间不匹配的问题，并通过球级核组合保持正定性

**🔧 技术方法**

结合多核学习、颗粒球计算、谱聚类与核空间聚类，利用球级核矩阵和同步权重学习实现高效聚类

**📊 数据集**

在12个公开数据集上验证，包括基因表达、对象、语音和人脸等高维数据集

**📈 对比分析**

与七种主流多核聚类基线对比，DK-GBMKKM在ACC、NMI、Purity、ARI四项指标上均名列前茅，平均提升约5-6%

**⚠️ 局限性**

对某些数据集表现不如特定方法（如RMKKM、MKKM-SR），且对大规模数据的低秩近似与自适应粒度仍有改进空间

---

## 206. Two locked tests of phase-structure features for transition prediction

**arXiv ID:** 2609.00335 | [PDF](https://arxiv.org/pdf/2609.00335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 207. Polished but Unresolved: Identifying Late-Stage Pressure States in Long-Horizon Tool-Use Agents

**arXiv ID:** 2609.00823 | [PDF](https://arxiv.org/pdf/2609.00823v1)

**作者:** Haoyang Chen `[一作]` (Nanjing University), Wei Hu `[通讯]` (Nanjing University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究长时序工具使用代理的“晚期压力”现象，即代理在未满足关键约束时仍趋向提交完整答案。

**💡 创新点**

提出“Probe-Sensed Pressure Relief (PSPR)”轻量级插件：利用线性探针实时感知压力并在适当时机通过激活干预或显式状态组织来缓解压力。

**🔧 技术方法**

技术包括线性探针训练、对比激活添加（contrastive activation addition）与方向残差消除、激活干预、显式约束清晰化与动作映射提示。

**📊 数据集**

主要在DeepPlanning-Travel长时序规划基准上进行实验，并在DeepPlanning-Shop与TravelPlanner等其他工具使用基准上进行验证。

**📈 对比分析**

与CoT、ReAct、Reflexion等主流代理结合，PSPR在pass@3指标上均提升CS、PS、CP等质量分数，且在多语言与不同模型规模上保持稳定改进。

**⚠️ 局限性**

局限包括：仅适用于可访问内部激活的可解释模型；需额外离线标注构建探针和干预方向；在开放式、多模态或动态环境下的可推广性待验证。

---

## 208. EEG-AS: Instance-Level Foundation Model Selection for EEG Foundation Models via Behavior Reconstruction

**arXiv ID:** 2609.00653 | [PDF](https://arxiv.org/pdf/2609.00653v1)

**作者:** Yunzhen Zhang `[一作]` (Duke Kunshan University), Mustafa Misir `[通讯]` (Duke Kunshan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了 EEG-AS 框架，实现了基于 EEG 基础模型的实例级算法选择。

**💡 创新点**

通过锚定模型推断其余模型的预测 token 并使用交叉注意力进行选择，避免完整模型评估。

**🔧 技术方法**

采用条件 token 预测器、token‑based 交叉注意力选择器、跨模型 token 复原、BIOT 预训练 EEG 嵌入与手工特征、对数排序损失等技术。

**📊 数据集**

使用七个公开 EEG 基准数据集：ADFTD、BCIC‑2a、MIMUL‑11、SEED‑V、SEED‑VII、THINGS‑EEG‑2 与 Workload。

**📈 对比分析**

与单一最佳 FM、虚拟最佳（Oracle）、MLP、RF、全局 token 预测器和特权教师等方法对比，平均选择准确率从 53.7% 提升至 66.9%，接近 Oracle 83.7%。

**⚠️ 局限性**

受限于模型间行为相似、选择余量有限以及对锚模型的依赖，导致部分数据集提升有限。

---

## 209. A note on the reduction from LTLf to LTL

**arXiv ID:** 2609.00379 | [PDF](https://arxiv.org/pdf/2609.00379v1)

**作者:** Alexandre Duret-Lutz `[一作]` `[通讯]`, Alexandre Duret-Lutz

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种从有限词 LTL（LTLF）到标准 LTL 的归约方法，并改造为始终得到合成义务（syntactic obligation）子句，方便后续使用专门的翻译和验证算法。

**💡 创新点**

创新点在于设计了四个相互递归的翻译函数（tB、tG、tS、tO）以及新的包装器，从而在保持等价性的同时保证最终公式位于义务片段；解决了原来使用 from_ltlf 时因使用强/弱运算符导致生成非义务公式的问题。

**🔧 技术方法**

主要技术包括：活编码（alive encoding）对有限词的无限扩展；LTL 语法层级与等价变形；结构归纳证明等价性和语法闭包；以及在 Spot 自动机工具中的实现。

**📊 数据集**

未使用外部数据集；实验主要在 Spot 2.16 上对比旧版 from_ltlf 与新 from_ltlf_O 的翻译结果。

**📈 对比分析**

通过在 Spot 里对比两种归约实现，验证两者在满足性上保持一致，同时新方法能够生成义务片段，允许使用现有的最小弱确定性 Büchi 自动机翻译；实验未给出精细性能数值，但改写后公式规模通常更小，计算更快。

**⚠️ 局限性**

局限性：归约依赖活编码，适用于非空有限词；实现仍需在 Spot 里手动开启环境变量以切换旧版；在极端复杂的 LTLF 表达式下，递归翻译可能导致公式膨胀；未在大规模随机公式上做性能基准。

---

## 210. EDRAC: Benchmarking Arabic Dialect Reading Comprehension

**arXiv ID:** 2609.01113 | [PDF](https://arxiv.org/pdf/2609.01113v1)

**作者:** Noor Abo Mokh `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Alham Fikri Aji `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究创建了首个基于自然口语的阿拉伯五大方言（埃及、摩洛哥、阿联酋、叙利亚、沙特）的生成式阅读理解基准EDRAC，包含499段语料与4977对人工验证的问答。

**💡 创新点**

创新点在于：①使用人机协作的迭代LLM生成与评估管道，②从YouTube真实口语转录中抽取语料，③通过多维度人工评估（相关性、自然性、正确性）确保方言真实性，④展示标准自动评估指标（ROUGE、BERTScore）与人工评估间的显著差异。

**🔧 技术方法**

技术实现采用Gemini‑2.5‑pro生成与评估问答、NVIDIA NeMo做声源分离与转录、手工标注工具、OpenRouter API统一调用多种阿拉伯中心与多语种LLM（Atlas‑Chat、Nile‑Chat、GPT‑5.4、Gemma4‑31B等），以及ROUGE、BERTScore与CAMeL‑BERTScore三种评估指标。

**📊 数据集**

使用的数据集为从公开YouTube频道提取的自然口语视频（约150个视频/方言），经过转录、清洗、校正后得到的499段文本，并通过LLM与人工验证生成的4977对问答；公开发布在HuggingFace（CC BY‑NC‑SA 4.0）。

**📈 对比分析**

与多种阿拉伯与多语种LLM对比时，GPT‑5.4在ROUGE‑L和BERTScore‑F1上表现最佳；Atlas‑Chat‑9B在整体性能和摩洛哥方言上尤为突出；但在CAMeL‑BERTScore上，GPT‑5.4降至第5名，显示出方言准确度不足；人工评估表明，即使自动指标高，模型仍常产生非方言化或自然性低的答案，说明现有指标无法全面衡量方言理解。

**⚠️ 局限性**

局限性包括：仅覆盖五大方言，未涵盖更细粒度的地域或社会变体；语料来源于在线媒体，可能带有轻度脚本化或自我监控；LLM生成与评估过程可能引入模型偏差；评测模型集合有限，未来模型演进可能改变性能排名；基准仅供评估，非预训练或微调数据。

---

## 211. Don't Let the Model Write the YAML: Deterministic, Minimal-Diff GitOps Remediation from LLM-Proposed Field Changes

**arXiv ID:** 2609.00227 | [PDF](https://arxiv.org/pdf/2609.00227v1)

**作者:** Pruthvi Davineni `[一作]` `[通讯]` (Independent Researcher), Pruthvi Davineni (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将LLM提出的配置更改意图与确定性、最小化差异的YAML节点位置编辑分离的GitOps工作流；

**💡 创新点**

核心创新在于将语义意图（资源、字段、值）与语法级别的字节编辑分离，通过节点位置标记实现确定性、无副作用、最小差异的编辑；

**🔧 技术方法**

使用YAML解析器的节点位置标记进行跨度编辑、基于Python的KubeAstra实现、Git Data API两阶段PR提交流程；

**📊 数据集**

基准数据集由83个字段更改任务组成，来源包括Online Boutique多资源YAML和Kustomize helloWorld仓库；

**📈 对比分析**

与全文件重写、统一diff和重试diff等基线对比，span‑edit在正确率100%、无副作用、最小diff、确定性上优于基线，且生成成本为O(1)；

**⚠️ 局限性**

局限包括仅支持标量字段更改、对Helm/Argo值间接引用、YAML别名等情况缺乏处理，以及目前数据集规模有限，需在更多仓库上验证。

---

## 212. Disclosure-Gated User Simulation for Companion-Agent Evaluation

**arXiv ID:** 2609.00982 | [PDF](https://arxiv.org/pdf/2609.00982v1)

**作者:** Yao Liu `[一作]`, Yu He `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一种基于“信息披露门控”机制的用户模拟器，用来抑制大型语言模型在评估中的“过度合作”现象，并对门控机制的可测性、可复现性和对评估环境的影响进行了系统实验。

**💡 创新点**

创新点包括：①以可执行的状态机形式完整公开门控规则（门层、转移、退化）；②将门控信息拆分为可训练和可推断两部分，验证门控是否已内化到模型权重；③通过合成与真实数据双分支的训练策略，明确门控行为来源；④引入“中层披露对比”“否决标准”等两个内在层面指标，构建了可与榜单对照的下游评估标准。

**🔧 技术方法**

使用技术：大规模预训练语言模型（35B、122B）、自定义状态机转移表、LLM审计器、无监督/监督混合训练、门控信息剥离对照实验、相关性与排名漂移分析。

**📊 数据集**

数据集：中文生产会话约35,000条（约31,000条带项级披露标注），合成分支基于同一标注生成68×17情景的轨迹；英文下游转化数据约22,000条会话；对照榜单采用 CompanionBench 原始排行榜及其种子重跑。

**📈 对比分析**

比较方法：在内部层面使用门控指标（中层披露对比、否决标准）检验门控是否生效；在下游层面计算榜单秩相关系数ρ、最大秩位移、前5名重合度以及系统总分变化。实验表明，去除合成分支或门控信息后，门控指标几乎消失；仅 122B 规模的 M1b 模型同时满足秩保持（ρ≥0.95）和分数尺度稳定（相对偏差≤21%），被推荐作为正式评估环境。

**⚠️ 局限性**

局限性：①门控验证和人类研究仅在中文数据及内部评审员完成；②英文下游实验未覆盖门控指标；③模型未经过偏好优化，导致响应长度偏短；④判定门控是否已内化的评估受审计器可见深度标签的限制；⑤深度披露在实际对话中难以客观度量，导致部分指标主观性高。

---

## 213. Neurosymbolics for Data Engineering: Achieving Long Context Token Reduction Without Finetuning

**arXiv ID:** 2609.00367 | [PDF](https://arxiv.org/pdf/2609.00367v1)

**作者:** Vishvesh Bhat `[一作]` `[通讯]` (CoreThink AI), Vishvesh Bhat (CoreThink AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可插拔的神经符号化层，提升LLM在数据工程任务中的推理准确性，并通过符号化上下文压缩显著降低长上下文的token消耗，无需微调即可实现性能提升。

**💡 创新点**

创新点在于将General Symbolics Reasoning (GSR) 的理念落地为可直接嵌入现有LLM的神经符号化层，既增强逻辑推理，又通过符号化压缩把 Transformer 的 O(n²) 复杂度降至约 O(n)，并在不做任何微调的情况下实现显著准确率提升。

**🔧 技术方法**

采用了神经符号化推理层、NL‑to‑NL 逻辑规则、实体标记与检索剪枝、符号化上下文压缩、Transformer 自注意力的稀疏化处理等技术。

**📊 数据集**

使用了 BIRD‑CRITIC、LiveSQLBench、LongBench v2、BFCL v3 Long Context 等数据集，并在 Deepseek R1 / V3.1 基础模型上进行实验。

**📈 对比分析**

与现有基准模型对比，CoreThink 在 BIRD‑CRITIC/LiveSQLBench 的准确率提升约 8.5%；在 LongBench v2 上实现 92% 的 token 压缩，同时保持 63.2% 的准确率；在 BFCL v3 上函数调用准确率提升 1%（从 19.5% 到 20.5%），token 压缩 35%。

**⚠️ 局限性**

局限性包括：对更大规模模型和更复杂数据工程任务的泛化能力尚未充分验证；符号化压缩可能忽略细粒度信息导致误判；实验多在实验室环境完成，缺乏在真实生产部署中的鲁棒性和稳定性评估。

---

## 214. (V)LMs generalize beyond surface co-occurrence: Evidence from cross-modal number agreement

**arXiv ID:** 2609.00443 | [PDF](https://arxiv.org/pdf/2609.00443v1)

**作者:** Zach Studdiford `[一作]` (University of Wisconsin-Madison), Kanishka Misra `[通讯]` (University of Texas at Austin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探讨视觉语言模型(VLM)是否能通过跨模态学习新词的语法数（单复）并在无语言提示的图像输入下进行数位一致性判断

**💡 创新点**

利用仅更新新词嵌入的“新词学习”框架，将视觉与语言两种提示方式的数位信息对比，证明VLM内部抽象机制可跨模态迁移

**🔧 技术方法**

新词学习训练（只更新嵌入）、最小配对数位评估、PCA+投影分析、四种因果干预方法（DAS、DiffMean、Probe、Attribution Patching）

**📊 数据集**

Qwen3‑VL-2B/4B模型的预训练权重，手工生成的30个图像-文本对（含单复图像），500个单数/复数真实名词及其对应的嵌入，700个不同吸引子级别的最小配对句子

**📈 对比分析**

与真实名词基准对比，所有条件下模型在数位一致性任务上显著高于随机（即使有多达3个吸引子）；嵌入迁移向单复方向的平均投影变化均显著；因果干预平均对数赔率均为正，语言与视觉条件无显著差异，表明抽象机制不受提示来源影响

**⚠️ 局限性**

仅测试Qwen3‑VL-2B/4B两款低参数模型；实验规模有限（仅少量图像、单语言）；采用单一新词学习策略，未探索其他元学习或多模态微调方法；缺乏对更大模型、其他语言或更复杂抽象的验证

---

## 215. How Temporal Correlations Shape Memory in Linear Recurrent Neural Networks

**arXiv ID:** 2609.00420 | [PDF](https://arxiv.org/pdf/2609.00420v1)

**作者:** Arnol Manuel Fokam `[一作]` (Independent Researcher), Edem Fiifi Dawson `[通讯]` (minoHealth AI Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究线性递归神经网络（LRNN）在时间相关输入下的学习动力学，给出精确解析解，并揭示记忆如何受到输入相关性、任务需求的影响。

**💡 创新点**

① 将输入相关性统一归纳为“记忆价格”，并证明记忆是否产生取决于任务对前一步的需求与两步最近相关性的单一比值；② 证明零误差任务需要额外的直接馈通项（feedthrough），并展示梯度下降在有余量隐藏维度时会自发构造该项。

**🔧 技术方法**

采用线性RNN模型，梯度流（gradient flow）解析解，能量函数推导，Wiener滤波器分析，随机欧拉积分数值实验；假设齐次奇异向量、白化特征及平稳相关输入。

**📊 数据集**

19个不同领域的时间序列数据集，包括金融、气象、水文学、脑电记录和太阳黑子序列，全部为单步回归任务。

**📈 对比分析**

方法通过数值模拟梯度流与理论能量最小值对比，验证理论阈值与实际训练结果一致；在真实数据上观察到记忆阈值与理论预测相符，无需额外的性能对比指标。

**⚠️ 局限性**

局限：仅适用于线性RNN，假设齐向量和白化特征；仅考察单步回归任务；对非线性、序列生成等更复杂场景尚未验证。

---

## 216. Topological Steering

**arXiv ID:** 2609.00597 | [PDF](https://arxiv.org/pdf/2609.00597v1)

**作者:** Benoît Guérand `[一作]` (National University of Singapore), Tan Minh Nguyen `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的推理时LLM行为控制框架——Topological Steering，通过对激活空间的持久性同调分析来生成指向拒绝行为的干预向量。

**💡 创新点**

创新点在于用全局拓扑特征（持久性图）替代传统的全局均值差向量，识别并聚合行为相关的局部结构，从而在多模态、高维激活空间中获得更稳健、更可解释的控制方向。

**🔧 技术方法**

核心技术包括联合PCA降维、Vietoris–Rips持久性同调、持久性图的瓶颈匹配与差异筛选、以及基于选定拓扑簇的局部对比向量聚合。

**📊 数据集**

使用的对比数据集为AdvBench中的520条拒绝性危险提示与约512条相似主题的合规提示，模型包括Llama‑3.1‑8B、Qwen‑2.5系列、Gemma‑2系列等多种decoder‑only架构。

**📈 对比分析**

与传统均值差激活干预（activation steering）以及角度/非线性干预方法相比，Topological Steering在激活分离指标Δρ上平均提升约0.01-0.02，并在外部自动评测（Llama Guard 3、HarmBench、子串拒绝检测）中分别提高0.18、0.02、0.01的成功率，证明其在不同模型和层级上均能显著改善行为区分。

**⚠️ 局限性**

局限性包括对预处理和超参数（降维维数、持久性阈值、最小簇大小等）的敏感性、持久性图对语义的非唯一性、计算开销随点云规模和同调阶数增长、以及仅针对拒绝行为的验证，其他安全维度需要额外的对比集与评估。

---

## 217. A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals

**arXiv ID:** 2609.00760 | [PDF](https://arxiv.org/pdf/2609.00760v1)

**作者:** Yuri Son `[一作]`, Taeuk Kim `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提供了一个示例，演示如何在 LuaLaTeX 或 XeLaTeX 环境下使用 ACL 样式文件来排版多语言文本。

**💡 创新点**

创新点在于将 ACL 样式文件与 LuaLaTeX/XeLaTeX 结合，并展示了多语言文本（印地语、阿拉伯语等）的排版方法。

**🔧 技术方法**

使用的技术主要是 LuaLaTeX/XeLaTeX 以及 ACL 提供的 LaTeX 样式文件。

**📊 数据集**

并未使用特定数据集，示例文本仅为多语言短句子，用于展示排版效果。

**📈 对比分析**

该示例不包含实验或性能比较，仅用于演示排版和引用格式，无法评估性能。

**⚠️ 局限性**

限制在于缺乏真实论文内容、实验数据和性能评估，功能仅限于排版演示。

---

## 218. A Dataset for Modeling Iterative Problem-Solving

**arXiv ID:** 2609.00940 | [PDF](https://arxiv.org/pdf/2609.00940v1)

**作者:** Fagun Patel `[一作]` (Stanford University), Nick Haber `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究创建了大规模CodeInsight数据集，并在其上对多类预测模型进行基准评测，旨在把迭代式问题求解建模为序列预测任务；

**💡 创新点**

创新点在于首次将参数化、序列化和生成式模型统一放在同一评估框架下比较，并提出了基于离散隐状态的递归状态空间模型RSSM来捕捉解题者特征；

**🔧 技术方法**

使用了IRT、CIRT、BKT、DKT、Code‑DKT、TIKTOC等传统模型，以及改进的RSSM和未微调的LLM（Qwen3‑14B等）作为生成式预测器；

**📊 数据集**

采用了来自越南VNU‑HCM大学两门C++课程的3.3M条提交记录，包含测试案例级结果、时间戳和完整源码的CodeInsight数据集；

**📈 对比分析**

通过共享的校准‑评分协议对模型进行评测，RSSM在三门课程中取得最高AUC，LLM预测准确性低于所有训练模型，说明LLM更像是基于上下文的解题生成器而非真实学生模拟器；

**⚠️ 局限性**

局限性包括数据仅来自单一学校与单一编程语言，未对LLM做微调，评估仅聚焦预测准确度，未检验模型在教学干预中的实际效用；

---

## 219. Escaping Redundant Reasoning: Structure-Aware Search for Inference-Time LLMs

**arXiv ID:** 2609.00738 | [PDF](https://arxiv.org/pdf/2609.00738v1)

**作者:** Lu Cheng `[一作]` `[通讯]` (University of Illinois Chicago), Lu Cheng (University of Illinois Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为BASIN的结构感知选择方法，旨在改善大语言模型（LLMs）在推理时的搜索效率，避免重复访问相同的推理路径。

**💡 创新点**

BASIN通过将推理状态分组为基盆，并对重复访问的策略施加惩罚，从而重新分配搜索到真正不同的推理路径，解决了推理盆地崩溃的问题。

**🔧 技术方法**

使用了基于历史的惩罚机制，灵感来自分子动力学中的元动力学方法，BASIN和其质量感知变体QA-BASIN均不需要训练。

**📊 数据集**

使用了Game of 24和MuSR等数据集进行实验，涵盖了符号推理和自然语言推理任务。

**📈 对比分析**

与标准的Tree of Thoughts (ToT)方法相比，BASIN在相同的推理预算下在Game of 24上提高了22个百分点，在MuSR上提高了6.7个百分点，表现出更好的搜索效率和准确性。

**⚠️ 局限性**

BASIN的局限性在于基盆的定义依赖于任务，语义任务需要近似表示，且质量信号的可靠性影响QA-BASIN的效果。此外，冗余间隙Δ不足以单独决定何时需要额外的探索。

---

## 220. Benchmarking Vision-Language Models for Automated Pathology Diagnosis and Report Generation

**arXiv ID:** 2609.00866 | [PDF](https://arxiv.org/pdf/2609.00866v1)

**作者:** Yumi Lee `[一作]` (Ewha Womans University), Sangjeong Ahn `[通讯]` (Korea University Anam Hospital)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过REG 2025挑战构建了首个面向全切片影像与报告生成的大规模泛亚区域数据集，并提供公开基准。

**💡 创新点**

创新点在于整合临床标准化报告模板、跨机构多中心数据、并通过对比实验揭示结构化报告表示和多模态对齐对生成质量的决定性影响。

**🔧 技术方法**

方法涵盖注意力多实例学习、层级专家模型、检索增强生成、跨模态Transformer以及基于Vision‑Language模型的多任务生成。

**📊 数据集**

使用约10,500对WSI‑报告样本，来源于韩国、印度、日本、土耳其与德国的五家医疗机构，覆盖七个器官与多种病理类别。

**📈 对比分析**

在24支参赛队伍中，顶级模型在最终综合评分上超过0.80，优于公开基准约30%，并在欧洲样本上展现跨域泛化能力。

**⚠️ 局限性**

主要局限包括数值属性的幻觉与不稳定、诊断过度细化、对罕见病理的误判以及对不确定性处理不足。

---

## 221. The Price of Remembering: A Calibrated Energy Law for Computation

**arXiv ID:** 2609.00744 | [PDF](https://arxiv.org/pdf/2609.00744v1)

**作者:** Mohamed Amine Bergach `[一作]` `[通讯]` (Illumina), Mohamed Amine Bergach (Illumina)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了一套完整的能耗法则：计算的能耗至少等于算术操作能耗、活跃位（rent）在所有工作层级的保留能耗以及位移动（fare）的能耗，并通过门级仿真在 45 nm 合成环形处理器上测得常数。

**💡 创新点**

创新点在于：①把不同领域（密码学、复杂度理论、计算机体系结构、机器学习）已有的能耗度量统一为一个物理定律；②用“服务 lemma”将算法的空间/时间下界直接转化为能耗下界；③通过硬件实验证明该定律的常数在实际芯片上成立；④揭示记忆（rent）是计算能耗主导项，解释了排序、scrypt、长上下文注意力等任务的能耗壁垒。

**🔧 技术方法**

使用的技术包括：门级合成（NanGate‑45 nm）、TLA+/TLAPS 形式化验证、差分能量提取方法、环形（旋转）存储器模型、算子（ADD、MUL、MAC）能耗基准、以及多级存储层级的理论分析。

**📊 数据集**

所用数据集主要是合成程序（如计数求和、MAC 乘加、排序、scrypt 哈希、注意力模拟）和统一的输入分布（随机位、均匀整数列表等），并未使用标准机器学习数据集；通过这些程序产生的执行轨迹来验证能耗下界。

**📈 对比分析**

比较方法：将实验测得的总能耗拆分为算术、租金、运费三项，并与理论给出的能耗下界（例如排序的 Ω(n²/log n) bit‑steps、scrypt 的 Ω(n²w) cumulative‑memory）进行对比。实验结果显示，在 45 nm 环形处理器上，能耗与理论下界一致（误差 < 4 %），并揭示租金占比超过 80 %，运费在大规模上下文注意力中成为主导能耗。

**⚠️ 局限性**

局限性：①定律假设工作层级均为易失性存储；在非易失性或可逆/光学存储等新型介质上需重新校准；②常数取决于特定工艺和工作频率，迁移到其他技术节点需要再次测量；③未覆盖动态电压/频率调整、异步时钟等更复杂的时序模型；④在高度并行或流水线化的系统中，租金与运费的分离可能不再严格；⑤模型忽略了逻辑门的细节寄存（如临时寄存器）导致的额外能耗。

---

## 222. MUFASA: An Information Utility-Aware Preprocessing Framework for Reliable Model Reasoning in Computational Pathology

**arXiv ID:** 2609.00424 | [PDF](https://arxiv.org/pdf/2609.00424v1)

**作者:** Rathinaraja Jeyaraj `[一作]` (Stanford University), Jeanne Shen `[通讯]` (Stanford University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出 MUFASA，一种多阶段信息效用感知的预处理框架，用于 H&E 切片的 WSI 处理，能够去除多种伪影并保留诊断相关组织。

**💡 创新点**

创新点在于将伪影抑制与信息效用分层结合，利用无监督自动编码器重建损失进行连续效用分层，并加入可恢复阶段补偿低染色区域，实现既不失去重要组织又能有效排除无用区域。

**🔧 技术方法**

采用多阶段处理：染色光密度 (OD) 过滤、光学质量检测、无监督自动编码器实现信息效用分层，以及轻量化恢复分类器；下游则使用 13 种 MIL/Transformer/图模型进行推断。

**📊 数据集**

使用 TCGA（COAD、READ、LUAD、STAD、BRCA、BLCA）多个中心 H&E WSI 数据集，CAMELYON16 以及 Stanford CRC 数据集进行评估。

**📈 对比分析**

与 CLAM、Trident、Histolab 等主流预处理方法相比，MUFASA 在肿瘤诊断、亚型、MSI 预测和生存预后四类任务中平均提升 1–7% 的 AUC/C-index，并显著降低模型对伪影的注意力。

**⚠️ 局限性**

限制包括仅针对 H&E 切片，自动编码器训练样本有限，对非 H&E 或新型伪影的泛化需进一步验证；在某些任务中性能提升有限，且预处理耗时略高于纯规则方法。

---

## 223. Capacity Achieving Torn Paper Codes

**arXiv ID:** 2609.00522 | [PDF](https://arxiv.org/pdf/2609.00522v1)

**作者:** Junsheng Liu `[一作]` (Washington University in St Louis), Netanel Raviv `[通讯]` (Washington University in St Louis)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

论文提出了一种多层本地对齐与飞行器重用技术，用来在碎纸信道（torn paper channel）中实现容量接近的编码方案。

**💡 创新点**

创新点包括：① 通过多级本地对齐与 pilot 重用避免了传统方案中 pilot 长度与短片段对齐之间的权衡；② 采用 RLL(0,β‑1) 约束与全局 De Bruijn 序列，使得每个片段即使很短也能唯一定位；③ 在每一层解码完成后将已恢复的位当作新的 pilot，逐层缩短对齐阈值，从而在有限的 pilot 位数下实现接近容量的速率。

**🔧 技术方法**

技术方法：多层递归解码、局部对齐算法、RLL 编码/解码、随机二进制线性码、统计唯一性分析、模数分层分配（residue class partition）以及概率上界分析。

**📊 数据集**

本工作是理论分析，没有使用具体数据集；通过概率论和编码理论证明了误码率趋于零并且速率趋近于信道容量 e^{-α}。

**📈 对比分析**

与之前的结构化方案（Shomorony‑Vahid 的 interleaved‑pilot 以及 Liu‑Raviv 的本地对齐）相比，该方案在相同 α 下能够取得更高的可实现速率，最终达到 e^{-α} 的容量上限；误码率随 n→∞ 以指数方式消失。

**⚠️ 局限性**

局限性：需要极大块长度才能实现理论上限，构造复杂且依赖随机线性码与随机公共移位；实现上对 RLL 编码与多层解码的实际硬件/软件实现可能具有较高的计算和存储开销；此外，方案在实际碎片分布模型（如非独立切点、片段丢失等）下的鲁棒性尚未验证。

---

## 224. A Formal Analysis of Agent Payment Protocols

**arXiv ID:** 2609.00060 | [PDF](https://arxiv.org/pdf/2609.00060v1)

**作者:** Ke Jiang `[一作]` (Southern University of Science and Technology), Yinqian Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 5892 | [OpenAlex ID](https://openalex.org/A5070946957)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对四种代理支付协议（x402、MPP、AP2、ACP）进行规范驱动的形式化验证，发现 40 条先前未记录的安全不一致性并提出相应修复。

**💡 创新点**

创新点在于：①构建统一的跨协议安全框架；②使用来源映射的 Tamarin 模型逐条验证 18 项安全原则；③将形式化发现与实际实现、SDK 规范和 PoC 对照验证，验证过程可追溯。

**🔧 技术方法**

主要技术包括：Tamarin 证明助手（符号多集重写），多阶段状态机抽象，源代码、规范、Schema 对齐的模型构造，及 counterexample‑guided 迭代修正。

**📊 数据集**

使用的“数据集”为：①四个协议的最新官方规范与代码库（单个版本）；②三套 x402 公开实现（Coinbase、Thirdweb、x402‑rs）；③与之对应的 SDK、Schema 文档。

**📈 对比分析**

比较方法：对每条安全关系在所有协议中分别构造“证据模型”和“匹配参考模型”，在 Tamarin 中分别进行验证；若失败则归因、改进后重新验证。性能：在 32 核 CPU、31 GiB 内存的机器上，所有 86 个验证用例总耗时约 100 秒，CPU‑秒 326，峰值内存 1.5 GB。

**⚠️ 局限性**

局限性：①验证结果仅针对符号模型，未覆盖所有部署或实现细节；②假设的攻击模型为 Dolev‑Yao；③未对所有可能的安全属性进行完整枚举，仍可能存在未发现的安全缺口。

---

## 225. StateSwap: Probing Support-Elimination Hidden States in Multiple-Choice Questions

**arXiv ID:** 2609.01081 | [PDF](https://arxiv.org/pdf/2609.01081v1)

**作者:** Chao Gao `[一作]` (Vrije Universiteit Amsterdam), Jinguang Gu `[通讯]` (Wuhan University of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大语言模型在支持式（Sup）和排除式（Elim）提示框架下的内部表示进行对比，提出“StateSwap”激活替换干预方法。

**💡 创新点**

创新点在于利用未训练的特殊令牌作为残差流接口，证明不同提问框架产生可分离且行为相关的中间层激活，并通过激活替换提升预测一致性。

**🔧 技术方法**

使用对偶框架对齐提示、残差流特殊令牌插入、激活替换干预、层级可分离诊断、均值差方向推导和对比激活注入等技术。

**📊 数据集**

在两个多项选择基准上评估：MMLU-17（推理子集）和 MedQA-CH（中文医学）。

**📈 对比分析**

与基线、随机替换、最终内容词替换等对照相比，跨框架激活替换可提升两种提示的准确率和交叉框架的Jaccard相似度，且在Qwen-2.5-7B和GLM-4-9B上均实现显著的性能提升。

**⚠️ 局限性**

限制包括仅在确定性贪婪解码、四选项MCQ、白盒模型且未检验采样或开放式任务时有效，且对特殊令牌位置和接口设计敏感，无法直接推广至黑盒或更复杂场景。

---

## 226. Are We There Yet? Assessing Computer-Use Agents for Blind Users' Accessible Interaction with Desktop Applications

**arXiv ID:** 2609.00524 | [PDF](https://arxiv.org/pdf/2609.00524v1)

**作者:** Satwik Ram Kodandaram `[一作]` (Stony Brook University), Vikas Ashok `[通讯]` (Old Dominion University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在三周日记研究中，收集了8名盲人屏幕阅读器用户使用自研可访问CUA原型的1258条命令，并对五种大型语言模型进行跨模型执行和评估。

**💡 创新点**

首次提供了面向盲人真实桌面工作流的人机交互数据集和跨模型的实验，揭示了CUA在非视觉场景下的失败模式与盲人对协作支持的期望。

**🔧 技术方法**

结合屏幕阅读器友好的交互层、UI树与截图的多模态感知、GPT‑5及Claude、Gemini等模型的语言推理和动作规划。

**📊 数据集**

自建的1258条盲人命令及对应执行轨迹的数据集（包含截图、UI树、模型响应、动作和交互历史）。

**📈 对比分析**

对每条命令在五个模型上重现执行，使用人工标注的参考步骤评估成功/部分/失败；GPT‑5最高成功率52.5%，其余模型依次为48.5%、43.9%、39.8%和37.9%。

**⚠️ 局限性**

样本局限于使用屏幕阅读器的盲人、仅在Windows环境、仅英文、仅桌面应用，且评估模型数量有限，未覆盖所有平台和语言。

---

## 227. Group Adaptive Clipping Policy Optimization

**arXiv ID:** 2609.00444 | [PDF](https://arxiv.org/pdf/2609.00444v1)

**作者:** Sheng Jia `[一作]` (University of Toronto), Rein Houthooft `[通讯]` (Amazon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对RLVR中的重要性采样比率（IS）clip不均衡问题，提出Group Adaptive Clipping Policy Optimization（GAPO），让clip阈值随rollout优势自适应。

**💡 创新点**

创新点在于依据逆KL信赖域推导的目标IS比率，将clip阈值与优势成正比，形成闭式自适应公式，既避免稀缺正确rollout被过度抑制，又不需要额外超参。

**🔧 技术方法**

利用RLVR、PPO/GSPO的clipping框架，结合逆KL信赖域分析、组内成功率c计算优势并调节上界clip阈值，实验中实现seq-level IS与token-level IS两种版本。

**📊 数据集**

使用DeepScaleR数学推理数据集、DeepCoder代码生成数据集进行训练，评测基准包括AIME24/25、AMC、MATH500、Minerva、OlympiadBench、LiveCodeBench、HumanEval+、IFEval等。

**📈 对比分析**

与固定clip的GRPO、Dr.GRPO、GSPO（对称/非对称）以及F-GRPO/F-GSPO等基线在相同clip宽度下对比，GAPO在Qwen2.5-Math-1.5B、Llama-3.2-3B-Instruct、DeepSeek-R1-Distill-Qwen-1.5B模型上在pass@1和pass@k均提升，尤其在困难任务上显著；同时保持IS与优势的高相关性，避免pass@1下降。

**⚠️ 局限性**

局限性：仅对正优势rollout调整上界clip，下界clip固定；推导基于单prompt的逆KL信赖域，实际训练涉及多prompt，λ共享但近似；未探究错误rollout的自适应或联合调节。

---

## 228. Late Transformer Layers Recode Syntax Canonically: Evidence from Greek Scrambling and Cross-Layer Generalisation

**arXiv ID:** 2609.00416 | [PDF](https://arxiv.org/pdf/2609.00416v1)

**作者:** Christos Nikolaos Zacharopoulos `[一作]` (Independent Researcher), Théo Desbordes `[通讯]` (University of Geneva)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过跨层泛化分析，研究大型希腊语语言模型在处理对象相对句的SVO与VSO句型时，句法信息在不同层的表示变化。

**💡 创新点**

首次利用GAT式跨层泛化检验句法信息的表示格式变化，发现后层不只是信息衰减，而是方向性地重编码为规范词序。

**🔧 技术方法**

使用线性逻辑回归探针，对隐藏状态的均值、方差、偏度、峰度等四维统计特征进行分类，评估ROC‑AUC并采用cluster‑based permutation test进行统计。

**📊 数据集**

采用自行生成的128句（扩展至1024句）希腊语对象相对句对，构造SVO与VSO最小对，并在公开OSF仓库共享。

**📈 对比分析**

与传统逐层探测对比，中间层（约5–19层）能实现跨层泛化，性能显著高于随机；但后层（20–31层）训练的探针在早层表现低于50%，表明表示发生了方向性重编码。

**⚠️ 局限性**

局限性包括只能检测线性可解信息；SVO与规范状态共存可能混淆结构与频率；模板化刺激可能保留微弱词汇偏差；研究仅限于希腊语单一句型，跨语言、跨构式的推广尚未验证。

---

## 229. MiNER: Fine-Tuned Biomedical Natural Language Processing for Malaria Disease Entity Recognition in Clinical Texts

**arXiv ID:** 2609.00073 | [PDF](https://arxiv.org/pdf/2609.00073v1)

**作者:** V. S. Anoop `[一作]` (Amrita Vishwa Vidyapeetham), Devika N `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用BioBERT微调进行疟疾相关医学文献的命名实体识别，构建并公开标注数据集，最终实现了实体提取系统。

**💡 创新点**

创新点在于针对疟疾领域自定义实体标签、将BioBERT上下文嵌入与传统机器学习分类器相结合，并将手工标注的数据公开，提供可复现的实验资源。

**🔧 技术方法**

采用的技术包括BioBERT预训练模型、BIOES标注、TF-IDF/CountVector/Word2Vec/GloVe/BERT/BioBERT特征编码，以及SVM、LR、RF、NB等分类器。

**📊 数据集**

使用的数据集来源于PubMed疟疾文献摘要，经过Label Studio手工标注得到Disease、Organism、Medication、Protein、Gene、Anatomical Structures、Chemical Structures和Other等实体标签。

**📈 对比分析**

通过与多种传统编码+分类器基线（如TF-IDF+SVM、Word2Vec+RF等）比较，BioBERT+SVM/RandomForest等组合在精确率、召回率和F1分数上均显著提升，最高F1达到约92-93%。

**⚠️ 局限性**

局限包括数据集仅限疟疾文献、手工标注规模有限、未结合本体知识以及对罕见实体和同义词的辨识能力不足。

---

## 230. Reliable LLM-Generated Programs for High-Energy Physics Experiments through Graph-Grounded Software Knowledge

**arXiv ID:** 2609.01095 | [PDF](https://arxiv.org/pdf/2609.01095v1)

**作者:** Yue Sun `[一作]` (Institute of High Energy Physics, Chinese Academy of Sciences), Ke Li `[通讯]` (Institute of High Energy Physics, Chinese Academy of Sciences)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文在高能物理实验软件 ROOT 上构建软件知识图并结合结构化检索、程序示例与执行反馈三层 grounding，提升 LLM 生成的分析程序的可执行性与质量。

**💡 创新点**

创新点在于将结构化软件图检索、层次化示例检索和基于执行日志的 Error‑RAG 诊断修复三者整合成统一的 grounding 框架，并在多种 orchestrations 上系统评估其效果。

**🔧 技术方法**

采用的软件技术包括：软件依赖图（包含 containment、dependency、inheritance 三种边）构建与检索、BM25+FAISS 混合检索、RRF 重排序、示例层次化选择、执行质量门控、Error‑RAG 诊断驱动修复、LLM 生成与修正。

**📊 数据集**

使用的数据集为 275 个自然语言 ROOT 任务请求（从教程实现生成）及其隐藏参考实现，ROOT 代码仓库生成的 49,270 节点/167,472 边的知识图。

**📈 对比分析**

比较方法为在 Claude Code 与 Standalone 两种 orchestrations 下，直接生成与完整 grounding 两方案的首轮执行率、质量门控通过率、最终成功率与平均程序得分进行对比。结果显示 grounding 在首轮可执行率提升 17.5%/12.7%、质量门控通过率提升 14.5%/14.2%、最终成功率提升 5.5%/12.0%，平均得分均显著提升，且修复轮数明显减少，平均成本增长仅约 1–3%。

**⚠️ 局限性**

主要限制包括：相比直接生成，token 消耗与平均生成时间略增；方法依赖可公开访问的源码或接口元数据，对闭源或内部专有软件的适用性仍需验证；以及在极端复杂依赖场景下可能需要更精细的图结构或注意力机制来进一步优化检索与修复。

---

## 231. Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models

**arXiv ID:** 2609.00495 | [PDF](https://arxiv.org/pdf/2609.00495v1)

**作者:** Guoli Wang `[一作]` (Case Western Reserve University), An Wang `[通讯]` (Case Western Reserve University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对扩散式大语言模型(dLLMs)的生成过程进行细粒度测量，分析了拒绝信号在早期去噪步骤和前置响应位置的分布，并提出一种训练无关的解码方法RAEC，用于在去噪过程中早期、持续地提交拒绝信号，从而提升模型的安全性。

**💡 创新点**

创新点在于：①首次将“浅步对齐”(shallow-step alignment)概念引入dLLMs，揭示早期去噪步承诺的安全影响；②发现安全失败往往是因为拒绝信号弱或不持久，非完全缺失；③提出RAEC在解码时根据拒绝与遵从信号的概率质量与阈值动态强制早期拒绝承诺，提升安全性且不需额外训练。

**🔧 技术方法**

技术手段包括：基于掩码去噪的dLLM解码流程、对齐与基线模型间概率偏移计算构建拒绝词表、对拒绝与遵从信号的概率质量与阈值进行监控、RAEC的窗口内持续检测与强制承诺逻辑。

**📊 数据集**

使用的数据集包括：安全评估基准StrongREJECT、JailbreakBench及其DIJA对抗变体；功能性评估用GSM8K（数学推理）和AGNews（文本分类）；此外还使用HumanEval进行安全词表过滤。

**📈 对比分析**

与原生解码（vanilla）相比，RAEC在LLaDA-8B-Instruct和Dream-v0-Instruct-7B上将攻击成功率(ASR)显著降低（例如StrongREJECT从1.28%降至0.00%），同时在GSM8K和AGNews上的准确率基本保持不变（utility几乎不受影响）。

**⚠️ 局限性**

局限性主要在于：①实验仅覆盖标准dLLM架构，未探讨更复杂或微调后模型的安全动态；②未分析在模型微调过程中安全信号随时间或位置的演化；③RAEC依赖于预先构建的拒绝词表，可能对不同语言或专业领域的适用性有限。

---

## 232. A multicenter benchmark and clinically structured metric for coronary CTA report generation

**arXiv ID:** 2609.00909 | [PDF](https://arxiv.org/pdf/2609.00909v1)

**作者:** Zhiyu Ye `[一作]` (Chinese Academy of Sciences), Tong Zhang `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本研究构建了跨中心的冠状动脉CT（CCTA）报告生成基准，并提出了新的结构化临床评估指标CSM_CCTA，用于对生成报告进行层级化、结构化的临床准确性评价；同时在基准上评估了七种3D视觉语言模型（VLM）的性能；

**💡 创新点**

CSM_CCTA创新点在于：①基于临床指南设计的变量集合，将报告拆解为病人、血管、分段三级结构化变量；②自底向上的聚合与自适应层级匹配，避免因细节差异导致的不公平惩罚；③权重通过专家线性回归校准，使得指标与临床专家判断高度一致；④在同一框架下兼顾语义稳定性和临床完整性。

**🔧 技术方法**

技术上使用了：①多模态3D VLM（如C2RG、CT-CHAT、M3D等）进行报告生成；②词法与规则式关键词提取、结构化四元组构造；③层级变量聚合（Agg）、F1计算及自适应级别选择；④受限线性回归校准专家权重；⑤与13种传统NLG/医学指标（BLEU、ROUGE、BERT-F1、GREEN、FORTE等）进行对比。

**📊 数据集**

数据集包括四家医院（PUMCH 567例、TCH 105例、SJTH 49例、FAHXMU 97例）共818名患者，3,021条CCTA序列；另外从PUMCH随机抽取100例作为专家评估队列，进一步划分为70例训练和30例验证。

**📈 对比分析**

通过在上述基准上对七种VLM进行评估，CSM_CCTA在30例验证集上的Pearson相关系数最高（r=0.97，p<0.001），与专家偏好的一致率为71.9%；在细节级别上能清晰区分模型在病人、血管和分段层面的表现；传统NLG和医学指标往往压缩或过度依赖表面词汇，难以体现临床完整性。

**⚠️ 局限性**

局限性包括：①CSM_CCTA依赖于CCTA特定的变量体系，迁移到其他检查需要重新定义和校准；②关键词提取为确定性规则，可能漏掉罕见或机构特有表述；③专家权重仅基于单中心（PUMCH）样本，需更大多中心验证；④参考报告的完整性和细节水平差异仍可能影响层级匹配；⑤单报告模式只能评估结构完整性，不能衡量临床准确性。

---

## 233. Dependency-Aware Chain-of-Thought Compression for Financial Reasoning

**arXiv ID:** 2609.00413 | [PDF](https://arxiv.org/pdf/2609.00413v1)

**作者:** Wenjun Wu `[一作]` (University of Illinois Urbana-Champaign), Sichen Zhao `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种层次语义提炼网络，压缩金融推理链条，保持答案准确性和逻辑连贯性。

**💡 创新点**

引入基于依赖图的压缩、双编码重要性评分、全局最优段落选择、边界重写，且仅使用冻结LLM做特征提取，保持可解释性。

**🔧 技术方法**

语义分割（BiLSTM‑CRF）、双向注意力依赖图、双编码重要性评分、动态规划优化、序列到序列边界重写、Qwen3‑4B 冻结模型、强化学习微调等技术。

**📊 数据集**

AFAC2025 金融推理基准；训练时使用人工标注推理链和大语言模型生成的依赖图。

**📈 对比分析**

与全链思维、直接生成、LLMLingua‑2、CompAct、RECOMP、LongLLMLingua 等方法对比，HSDN 在准确率 91.0%、压缩率 68.4%、RCS 0.867 上优于其它方案，整体得分最佳。

**⚠️ 局限性**

仍然依赖冻结LLM进行特征提取与答案生成，压缩后仍需小量重写可能引入细微错误；图构建复杂度 O(K²)；专为金融推理设计，跨领域泛化需要额外适配。

---

## 234. Sources of Truth: A Multi-Platform, Multilingual Audit of Citations in AI Mental Health Information Queries

**arXiv ID:** 2609.00319 | [PDF](https://arxiv.org/pdf/2609.00319v1)

**作者:** Phuong Anh Nguyen `[一作]` (Harvard Medical School), John Torous `[通讯]` (Harvard Medical School)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对三款主流消费者向的生成式AI搜索产品（ChatGPT、Perplexity、Google AI Overview）在回答心理健康相关问题时的引用来源进行系统审计，记录并分类超过1.5万条引用。

**💡 创新点**

创新点在于首次对心理健康查询的生成式搜索进行大规模审计，提出并公开九类来源组织类型的典型化分类法、确定性分类器及其注释语料库，并在多语言环境下揭示语言不平衡对引用行为的影响。

**🔧 技术方法**

使用规则化确定性分类器（基于域名、主机、公共后缀）对引用进行自动标注，并结合脚本解析URL、检索国家代码与本地化信号；研究采用人工标注对算法进行验证。

**📊 数据集**

利用自建数据集，包含1,140条来自英语20题及7种语言各3题的回答，共15,942条引用、1,713个域名，已在Hugging Face公开。

**📈 对比分析**

通过比较三平台的平均/中位引用数量、来源类型比例以及提示“列出来源”对结果的影响，发现平台在引用一致性和来源偏好上存在显著差异；在跨语言实验中，非英语查询的引用数量和本地化比例显著低于英语。

**⚠️ 局限性**

局限包括仅对2026年6-7月的产品快照进行审计，使用免费层并在无登录状态下操作，非英语语言仅覆盖三道题目，未对链接可达性做进一步核验，且注释者全部来自美国，可能影响跨国本地化的收集。

---

## 235. The Assistant's Ideal Self

**arXiv ID:** 2609.00304 | [PDF](https://arxiv.org/pdf/2609.00304v1)

**作者:** Mert Yazan `[一作]` `[通讯]` (Leiden University), Mert Yazan (Leiden University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过成对比较测量四个大型语言模型对32种自我概念属性的偏好，构建其理想自我排名

**💡 创新点**

提出结构化、位置平衡的成对比较法，将自我概念量表转化为模型偏好测量，并考察不同帧设置对偏好的稳健性

**🔧 技术方法**

采用排位赢率、Bootstrap不确定性估计、Benjamini‑Hochberg校正以及对比实验设计（自由提升/折衷、对象/主体）等统计技术

**📊 数据集**

32个自我概念属性来自五个已验证量表（自尊、自我概念清晰度、道德自我形象、身份连贯性、真实性等），在四个模型上进行496对比较，共计31,744次回应

**📈 对比分析**

通过计算每属性胜率并对比不同参数设定的差异，发现道德属性与自我理解属性优先，其余属性排后，偏好在不同帧下高度稳定，只有少数显著变化

**⚠️ 局限性**

仅基于模型自报，缺乏行为验证；使用未经过验证的自我概念项；位置敏感性存在；未包含基线未对齐模型；仅涉及四个大型模型

---

## 236. A Cone-Constrained Bilinear Decomposition for Total Scaled-Gradient Variation Models

**arXiv ID:** 2609.00036 | [PDF](https://arxiv.org/pdf/2609.00036v1)

**作者:** Haibin Su `[一作]` (Yau Mathematical Sciences Center, Tsinghua University), Zhifang Liu `[通讯]` (School of Mathematical Sciences and Institute of Mathematics and Interdisciplinary Sciences, Tianjin Normal University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于双线性分解的总缩放梯度变差（TSGV）正则化模型的数值求解方法，并将其推广到非线性、非凸的高阶图像恢复问题；

**💡 创新点**

通过引入增强方向场将非线性加权梯度完全解耦，得到等价的锥或球面约束形式，并结合大规模MM技术实现无步长调参的单调能量下降；

**🔧 技术方法**

双线性分解、Majorization‑Minimization (MM) 交替最小化、FFT 线性系统求解、投影到锥/球面约束、Kurdyka‑Łojasiewicz（KL）收敛分析；

**📊 数据集**

在高斯噪声图像去噪（多幅人工与真实图像）和混合可见壁的非线性视场（NLOS）重建（bowling、Stanford Bunny等公开数据集）上进行实验；

**📈 对比分析**

与IOS‑TSGV、HALM‑EE、LCT、CNLOS 等先进方法对比，MMAMM 在 PSNR、SSIM 上保持或略优于对手，迭代次数略多但每次计算更高效；

**⚠️ 局限性**

对三维 NLOS 重建的计算量仍较大，且在极端稀疏采样下的重建质量受限，未来工作需进一步提升算法规模化和自适应尺度函数选择。

---

## 237. Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs

**arXiv ID:** 2609.00908 | [PDF](https://arxiv.org/pdf/2609.00908v1)

**作者:** Runze Xu `[一作]` (Tsinghua University), Jincheng Yu `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于动作专家内部交叉注意力熵的自适应动作块化策略，动态调整执行时段。

**💡 创新点**

利用动作-观察交叉注意力熵的高平衡状态作为在线风险信号，实现无训练、无额外采样的动态截断。

**🔧 技术方法**

交叉注意力熵分析、滑动平均与阈值判定，并在π_0.5、X‑VLA等VLA模型中实现。

**📊 数据集**

RoboTwin 2.0、LIBERO 以及三项真实世界任务（杯子叠放、杯子挂取、笔整理）等数据集。

**📈 对比分析**

与固定长度、MS（多采样）、SA（自注意力）基准比较，平均成功率提升约3–5%，且推理延迟仅略高于固定块，显著低于MS。

**⚠️ 局限性**

仅适用于具有可分离交叉注意力路径的架构，对高度层化或世界动作模型的适用性尚未验证。

---

## 238. SAGE: State-Grounded, Abstention-Aware Evaluation of Task-Oriented Dialogue Agents

**arXiv ID:** 2609.00434 | [PDF](https://arxiv.org/pdf/2609.00434v1)

**作者:** Rayan Khoury `[一作]` (Microsoft), Pratyush Mishra `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SAGE 框架，将工作流规范与对话状态差异编译成原子判定标准，并通过多层级验证器逐一判定；

**💡 创新点**

创新点在于：① 将全局判定拆解为可验证的原子标准；② 采用符号规则、NLI 编码器与可选 LLM 的三阶段管道，且在不确定时主动放弃；③ 产生可追溯的证据与修复信号；④ 推荐的 SAGE-Core 方案实现零付费 LLM 成本；

**🔧 技术方法**

技术手段包括：工作流规范编译、符号规则检验、DeBERTa‑v3 与 MiniLM NLI 编码器、置信门控、有限 LLM 调用、以及配对 bootstrap 性能比较；

**📊 数据集**

实验数据集涵盖 MultiWOZ 2.4、Schema‑Guided Dialogue 以及 Action‑Based Conversations Dataset（ABCD），并在每个数据集上注入四类工作流错误；

**📈 对比分析**

通过与 G‑Eval、FrugalGPT 及单调用 LLM 评测器在四个切片进行配对 bootstrap 比较，SAGE‑Core 在零付费 LLM 成本下在所有切片的 F1 与最佳基线相当或更优，尤其在 SGD 切片显著领先；

**⚠️ 局限性**

局限性包括：需完整工作流规范，无法应用于开放式聊天；对 IUV 类的构造效度有限；评估多基于注入错误，真实错误检测仍待进一步验证。

---

## 239. Dense Weak Hiding: Closing Complexity Gaps in Nonconvex and PL Finite-Sum Optimization under Individual Smoothness

**arXiv ID:** 2609.00045 | [PDF](https://arxiv.org/pdf/2609.00045v1)

**作者:** Yuxing Peng `[一作]`, Weijia Jia `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

在非凸有限和优化以及满足Polyak–Łojasiewicz（PL）条件的情况下，给出了在个体光滑（individual smoothness）假设下的最优I/F一阶算子复杂度下界，并通过构造“dense weak hiding”实现了该下界的匹配。

**💡 创新点**

创新点包括：①提出 dense weak hiding 构造，能够在保持个体光滑性的同时实现信息隐藏并消除 √n 缺口；②利用固定符号表、径向映射和光滑阈值门实现对隐藏方向的有效编码；③在非凸与 PL 两个场景下给出统一的下界，并证明 Restarted PAGE 与之匹配。

**🔧 技术方法**

使用的技术主要是信息论方法（互信息、数据处理不等式）、固定编码表与随机采样的组合、径向映射与光滑阈值门的函数构造、以及层级几何缩放的阶段设计。

**📊 数据集**

本工作为理论性研究，没有使用任何实际数据集，所有结果均基于合成的极端难题构造。

**📈 对比分析**

与现有方法比较：现有上界（PAGE、SPIDER）为 O(n+√n ΔL/ε²)，本论文给出匹配下界 Ω(n+√n ΔL/ε²)。在 PL 条件下，Restarted PAGE 的上界为 O(n+nlog(Δ/ε)/(1+log(√n/κ)))（κ≤√n）和 O(n+κ√n log(Δ/ε))（κ≥√n），与下界完全匹配；因此证明了该方法在个体光滑下的最优性。

**⚠️ 局限性**

限制：①仅在样本数 n 足够大时成立；②只考虑完美的 IFO（无噪声）模型；③只适用于 κ≥3 的 PL 情况；④没有给出固定维度的下界；⑤未考虑稀疏/非光滑场景。

---

## 240. Explainable Artificial Intelligence for Industrial Cybersecurity: A Review of Methods, Operational Integration, and Research Challenges

**arXiv ID:** 2609.00171 | [PDF](https://arxiv.org/pdf/2609.00171v1)

**作者:** Amr S. Mohamed `[一作]` (German University in Cairo), Deepa Kundur `[通讯]` (University of Toronto)

**通讯引用:** 9540 | [OpenAlex ID](https://openalex.org/A5077035168)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对工业网络安全领域的可解释AI方法进行了系统综述，评估了其在工业SOC工作流程中的适用性并提出了研究挑战。

**💡 创新点**

首次聚焦工业SOC、工业OT/ICS环境，系统分类XAI方法并结合法规、安全要求提出了未来研究方向。

**🔧 技术方法**

综述了特征归因（SHAP、LIME、PDP、ICE、ALE）、代理模型、基于规则解释、可视化等XAI技术，并讨论其在工业安全中的实现。

**📊 数据集**

参考了工业IDS基准数据集（NSL‑KDD、UNSW‑NB15、CIC‑IDS、WUSTL‑IIoT等）以及少量异常检测数据集。

**📈 对比分析**

通过对比已有工作中XAI方法的适用性、解释质量和运算成本，总结SHAP最常用且解释效果好，LIME在局部解释中优势突出，但整体缺乏统一评估框架。

**⚠️ 局限性**

受限于工业数据稀缺、标签不足、模型可解释性与性能权衡、XAI工具与SOC流程融合难度大，以及缺乏标准化评测指标。

---

## 241. DualStake: Dual-Path Confidence Calibration in Deep Research Agents

**arXiv ID:** 2609.00935 | [PDF](https://arxiv.org/pdf/2609.00935v1)

**作者:** Yinuo Xu `[一作]` (Chinese Academy of Sciences), Jian Liang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Deep Research框架中加入了检索后逐步的置信度查询，并提出DualStake双路径校准方法来同时监督检索证据置信度和答案置信度，防止模型过度自信并提升可信度；

**💡 创新点**

①发现检索证据置信度提供比答案置信度更可靠的不确定性信号；②提出Margin-Clip Stake Reward的DualStake方法，双路径监督能够在保持答案准确性的同时显著提升置信度校准；

**🔧 技术方法**

使用强化学习（GRPO）进行训练，设计基于正确性与置信度的Stake Reward，并加入margin clipping来限制极端置信度；采用线性探针、激活补丁等技术分析置信度内部机制；

**📊 数据集**

在8个问答基准上实验：NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle、SimpleQA；训练时使用NQ+HotpotQA合并训练集；

**📈 对比分析**

与无校准推理、后置校准方法（温度缩放、序列概率）、训练时校准（MSCR、RLCR）等基线相比，DualStake在ECE从0.518降至0.178、AUC从0.552升至0.712、Brier Score从0.497降至0.220，同时保持或提升答案准确率；

**⚠️ 局限性**

实验仅覆盖3B-7B规模模型，资源受限；不同模型与方法在单一数据集上表现略有差异，未在更大模型或更广泛场景下验证。

---

## 242. Bounded Relative Boundary Implies Narrow DNF Approximation

**arXiv ID:** 2609.00240 | [PDF](https://arxiv.org/pdf/2609.00240v1)

**作者:** Chenghua Liu `[一作]` (Institute of Software, Chinese Academy of Sciences), Boning Meng `[通讯]` (University of Regensburg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了Friedgut关于在p偏置离散立方体中，边界相对较小的递增布尔函数可被窄宽的单调DNF近似的猜想。

**💡 创新点**

创新点在于构造了与偏置匹配的随机平移操作，将Hatami的伪junta结构转化为严格递增且保持低负载的自适应表示，并从中提取正向证明子集，最终得到宽度受限的单调DNF逼近。

**🔧 技术方法**

采用了Hatami的伪junta理论、偏置匹配的随机平移技术、激活负载分析、以及自适应分割的可测性论证，结合概率与信息论的加权平均方法。

**📊 数据集**

无，本文为纯理论证明，没有使用实验数据集。

**📈 对比分析**

本工作不涉及与其它算法或方法的实验比较，主要通过理论分析展示了在任意p∈(0,1)和高维情况下，宽度仅依赖于影响上界K和误差ε，与已知的junta近似方法相比提供了更适合稀疏空间的宽度控制。

**⚠️ 局限性**

局限性包括：宽度上界为exp(C(K+1)^2/ε^2)，尚未优化；结果为存在性证明，没有给出有效算法；并且在高偏置区间（p>1/2）需额外使用Friedgut的偏置junta定理，仍不完整。

---

## 243. How Does LGBTQIA+ Identity Affect LLM Behavior? Implications for Requirements Engineering of Mental Health AI Systems

**arXiv ID:** 2609.00352 | [PDF](https://arxiv.org/pdf/2609.00352v1)

**作者:** Shailyn Callihoo `[一作]` (University of Calgary), Ronnie de Souza Santos `[通讯]` (University of Calgary)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在精神健康对话中公开 LGBTQIA+ 身份是否会影响 ChatGPT 的回复表现，并以质性对比方式分析其回复特点。

**💡 创新点**

首次将身份披露与聊天机器人在心理健康情境下的回复进行对比实验，发现身份披露导致回复的语境扩展、身份确认和偶尔出现刻板推断，表明公平问题可能隐藏在细微的解释层面。

**🔧 技术方法**

采用 ChatGPT 5.3、二元编码与主题编码的质性比较方法，对 50 个真实精神健康问题在三种身份提示下的 450 条回复进行分析。

**📊 数据集**

使用 Counsel Chat 数据集中的 50 条真实精神健康提问，按“无身份披露”“直性恋身份披露”“LGBTQIA+ 身份披露”三种提示生成回复。

**📈 对比分析**

通过比较三种提示条件的编码结果发现，回复完整性和实用建议几乎无差异，但在 LGBTQIA+ 条件下身份确认率高达 78%，并出现了 18% 的“过度语境化”与 4% 的“歧视性处理”。

**⚠️ 局限性**

局限性包括：仅使用单一 ChatGPT 版本、样本仅为 50 条问题、身份披露仅以前置短句形式呈现，难以推广到其他模型、其他身份维度或更自然的对话情境。

---

## 244. Does This Moment Justify the Recommendation? Counterfactual Behavior-Grounded Evidence Retrieval for Personalized Video Recommendation

**arXiv ID:** 2609.00996 | [PDF](https://arxiv.org/pdf/2609.00996v1)

**作者:** Xin Liu `[一作]` `[通讯]`, Xin Liu

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了对个性化视频推荐中的证据检索进行因果对照实验，定义了 Where、Whether 和 Intervention 三个评估维度，并构建了 CBGER-10K 基准数据集；随后设计了一个轻量级的行为条件化证据模型 CBGER，利用结构化对照学习同时优化时序定位、证据存在判断和局部干预一致性。

**💡 创新点**

创新点在于：①首次把个性化视频证据检索拆解为定位（Where）与是否存在（Whether）两步，并加入对证据替换的干预一致性（Intervention）评估；②构造了 5,000 对受控事实-反事实视频样本，实现了在保持用户、时间位置和干扰片段不变的前提下，仅改变焦点片段的实验设计；③提出了结构化对照学习框架，既提升了时序定位，又显著提升了 Pair Accuracy 与 Intervention Consistency。

**🔧 技术方法**

技术手段包括：使用冻结的 CLIP 视觉/文本特征，轻量化投影 + 余弦相似度残差；相对时序 MLP 编码局部时序上下文；对比学习、温度化交叉熵、硬负样本排序、视频级平均池化、局部干预损失与干扰不变性损失的组合；实验采用 AdamW、学习率 2e-4、权重衰减 1e-4。

**📊 数据集**

数据集：MicroLens 的用户行为历史与 FineVideo 的视频片段；通过 Qwen3‑VL‑8B‑Instruct 进行语义注释，BGE‑M3 用于行为/视频匹配，最终构建 CBGER‑10K（5,000 对事实‑反事实，3,026 名用户）。

**📈 对比分析**

与六个基线（冻结 CLIP 相似度、PR‑Net、QD‑DETR、TR‑DETR、FlashVTG、MQVTG）在同一拆分上对比。CBGER 在 MRR、Pair Accuracy、Intervention Consistency 上均取得最高或第二高成绩；尤其 Pair Accuracy 提升 0.1103（95% CI 0.0765–0.1430，p<1e-4），Intervention Consistency 提升 0.0470（p=0.0022），表明对证据的识别和干预响应显著优于现有方法。

**⚠️ 局限性**

局限性：①标签由自动模型生成，可能带有语义/匹配偏差；②对照替换仅在构造条件下保证行为弱化，未必完全无意义；③只使用 9 段固定时间线和冻结 CLIP，限制了对更大范围视频和多样表示的泛化；④Pair Accuracy 只衡量相对排序，未能直接反映用户真实偏好或参与度。

---

## 245. Flawed in Nature, Perfect through Evolution

**arXiv ID:** 2609.00129 | [PDF](https://arxiv.org/pdf/2609.00129v1)

**作者:** J. M. Diederik Kruijssen `[一作]` `[通讯]` (Allora Foundation), J. M. Diederik Kruijssen (Allora Foundation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了“Flawed‑in‑Nature”机制：在概念漂移环境下，利用故意的模型参数突变产生多样化模型群，后通过推断合成层实现性能提升。

**💡 创新点**

创新点在于将自然选择与可遗传变异结合到 AI/ML 系统，证明通过模型群的随机突变能打破单一模型的线性 regret 限制，并给出四个理论定理和渐进收敛保证。

**🔧 技术方法**

核心技术包括：基于经验风险最小化的单模型训练、按指数时间分布的参数突变（Poisson 过程）、推断合成（加权线性池）以及误差回溯权重调节（指数滑动平均 + logistic 门）。

**📊 数据集**

实验数据集为合成线性回归环境，特征与系数按正态分布初始化，系数随时间以 Poisson 驱动的随机步长变化；还使用了真实数据的可扩展性实验（未在本文完整展开）。

**📈 对比分析**

比较方法为：单一模型、原始模型群、突变模型群以及其推断合成输出；结果显示突变模型群在大约 80% 的漂移事件中提供最佳预测，整体累计 regret 显著低于单模型和原始群，且推断合成层能进一步缩小误差。

**⚠️ 局限性**

局限性包括：理论证明依赖于损失有界、突变分布全支持且与环境漂移速率匹配；实验仅在低维合成线性设置中验证，未展示在高度非线性或大规模真实任务中的表现；突变率需手动调参，且过高会导致单体性能下降。

---

## 246. Federated Trust for Embodied Robot Capability Marketplaces

**arXiv ID:** 2609.00404 | [PDF](https://arxiv.org/pdf/2609.00404v1)

**作者:** Xue Qin `[一作]` (Harbin Institute of Technology), Zhijun Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了适用于机器人能力市场的联邦信任（Federated Trust）体系，提供本地信任目录、签名封装、注册表与安装门控，并在多部署、多监管场景下验证其可行性。

**💡 创新点**

创新点在于：① 用每个部署桥端的文件系统信任目录取代中心化PKI；② 采用 Ed25519 分离签名和签名封装，实现本地离线验证；③ 引入多签名阈值（k‑of‑n）扩展，限制单一签名者失效时的风险；④ 通过实验证明同一注册表在不同信任目录下可产生不同安装结果。

**🔧 技术方法**

核心技术包括：Ed25519 断层签名、SHA‑256 摘要、PEM 格式公钥文件、Python 3.11/3.12/3.13 生态（FastAPI 注册表、CLI、签名验证模块），以及在桥端实现的本地安装门控与多签名验证逻辑。

**📊 数据集**

使用自制的 .aecm 包和签名封装（共 1000 次攻击实验、1000 次多签名测试、1000 次基准验证）以及在 Apple M 系列机器上的性能数据；未采用公开工业数据集。

**📈 对比分析**

性能对比：与 Sigstore‑Cosign（离线）和 python‑TUF 进行对标。单包验证中，联邦信任中位数 204 µs，Cosign 860 µs，TUF 27.7 µs；冷启动信任目录扫描 124 µs 远快于 TUF 的 10,946 µs；存储占用 511 B 对比 Cosign 3,879 B、TUF 5,151 B；在严格模式下对 5 种攻击类检测率 100%，并在降级检测中 96.6%。

**⚠️ 局限性**

局限性：缺乏透明日志与自动化吊销；关键轮换需手工完成；多签名仅支持简单阈值，未实现阈值签名聚合；信任目录的文件系统访问需要运营端安全保障；仅覆盖安装时安全，无法防御内部人员（C_insider-all）或构建链泄露等更深层攻击。

---

## 247. Construction of a DFA for Computing Grundy Numbers in the Successful Derivation Games on Right-Linear Grammars

**arXiv ID:** 2609.00871 | [PDF](https://arxiv.org/pdf/2609.00871v1)

**作者:** Yoshiaki Takata `[一作]` (Kochi University of Technology), Hiroyuki Seki `[通讯]` (Mukogawa Women's University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究右线性文法（RLG）上的成功派生游戏（SDG），构造有限自动机（DFA）来计算给定位置的格兰迪数，并证明该格兰迪数集为正则语言，从而可判定其上界。

**💡 创新点**

创新点在于将SDG从不可判定的线性CFG扩展到可判定的RLG，首次证明对RLG可构造DFA识别格兰迪数，且最小上界可判定且属于PSPACE完全；提出了基于右线性文法的DFA构造算法和对应的复杂度分析。

**🔧 技术方法**

采用的技术包括：对RLG的语法规则进行分类（单元、终结、非终结规则）并构造状态集合；从右向左读取单词，递归维护子状态的格兰迪数；利用最小化DFA和状态转移函数实现格兰迪数计算；通过PSPACE算法和归约证明上界判定的复杂度。

**📊 数据集**

论文中未使用任何真实数据集，全部通过理论构造和复杂度证明完成；示例仅以人工构造的小型RLG展示DFA和格兰迪数。

**📈 对比分析**

方法的评估主要通过理论复杂度比较：对一般线性CFG的上界判定是不可判定的，而在RLG上可判定且属于PSPACE完全；实验性能未给出，主要通过构造的DFA大小和状态数进行分析。

**⚠️ 局限性**

限制包括：仅适用于右线性文法；在RLG中仍需要维护较大状态空间，导致DFA可能指数级增长；PSPACE复杂度意味着实际计算在大规模文法上可能不可行。

---

## 248. Time-Decayed Vector Search in the Rhythm of TANGO: Jointly Modeling Semantic Similarity and Temporal Freshness

**arXiv ID:** 2609.00548 | [PDF](https://arxiv.org/pdf/2609.00548v1)

**作者:** Jiuqi Wei `[一作]` (OceanBase, Ant Group), Themis Palpanas `[通讯]` (Universite Paris Cite)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了时间衰减向量检索（TDVS）框架，设计了精确的 STR 变换与 Chronos 量化方法，并基于此构建了可在线插入的分层图索引 TANGO。

**💡 创新点**

创新点在于：①将时间衰减与语义相似度直接联合建模；②通过 STR 将 TDVS 完全等价映射到 MIPS；③引入锚点无关的时间拉普拉斯核，实现精确度量化；④提出 Query‑Orthogonal TimeLift，实现查询不变但可调数据几何；⑤在 HNSW 结构上分层赋予不同的 TimeLift，兼顾时间局部性与跨时段语义连通。

**🔧 技术方法**

核心技术包括：指数衰减时间函数、加权/乘法 TDVS 得分、STR 语义编码+时间因子映射、Anchor‑Free 时间核的 Hilbert 空间实现、TimeLift 余量化、层级图搜索与缓存时间因子、以及高效的在线插入机制。

**📊 数据集**

使用七个真实数据集进行实验：Wikipedia‑Qwen、OpenAI‑1536、OpenAI‑3072、MSMARCO‑1M、MSMARCO‑10M、HotpotQA 与 Yandex Deep‑10M，覆盖文本与图像、500K–10M 大小、96–4096 维度。

**📈 对比分析**

与 ip‑NSW、ip‑NSW+、NAPG、MAG、PSP 等现有图索引在 STR 适配下对比，TANGO 在所有数据集上实现了最高召回率（≥0.99）时查询吞吐量提升 2.4–3.5 倍，构索时间提升 3.5–4.0 倍，内存占用与竞争者相当；且在不同时间衰减、半衰期与 α 参数设置下保持鲁棒性。

**⚠️ 局限性**

局限性包括：STR 需要全局时间锚且对后续插入需重新编码；Chronos 仅对单位归一化嵌入有效；采用指数衰减假设限制了对复杂时序模式的建模；对非内积/余弦相似度的支持有限，需进一步推广到更一般的相似度度量。

---

## 249. A Dimension-Reducing Fréchet Simplification Oracle

**arXiv ID:** 2609.00393 | [PDF](https://arxiv.org/pdf/2609.00393v1)

**作者:** Boris Aronov `[一作]` (New York University), Indu Ramesh `[通讯]` (New York University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一套近线性大小的数据结构，用于在给定查询线、平面曲线或几何树、甚至高维平面时，快速求解最小化离散 Fréchet 距离的曲线简化问题。

**💡 创新点**

创新点在于将最小包围圆/球、离散 Fréchet 距离决策、层次分解与核心集技术结合，构造可在多种约束空间（线、平面、区域、g-平面）下均可查询的高效算法；同时对高维问题提供近似解决方案。

**🔧 技术方法**

主要技术包括：离散 Fréchet 距离决策/优化算法、层次分解（分治树）和重心分解、最远点 Voronoi 图/树、核心集构造与最小包围球/球求解、二分搜索与递归搜索、近似核心集（近似球）以及 L∞ 及 L2 近似策略。

**📊 数据集**

本文为理论性工作，未使用具体实验数据集；所有结果均在抽象的几何对象（多边形曲线、几何树、平面多边形区域）上进行证明和分析。

**📈 对比分析**

相比传统的全局简化或不受约束的 Fréchet 简化方法，本文提供了更强的空间约束（线、区域、平面）下的答案，且查询时间为 O(k² log³n + k log⁴n)（平面线）或相似阶的近似时间，优于先前仅适用于不受约束简化的 O(n²) 或 O(n³) 级别算法。

**⚠️ 局限性**

局限性包括：查询时间中存在 k² 乘子，可进一步优化；高维问题只能得到 (1+ε) 近似解；对特殊约束（如非凸区域、多维平面）仍缺乏最优（线性或多项式）解；且实现中涉及多层次树和核心集，实际常数可能较大。

---

## 250. Classic AI Scaffolding for LLM Social Agents

**arXiv ID:** 2609.01167 | [PDF](https://arxiv.org/pdf/2609.01167v1)

**作者:** Anatole Gershman `[一作]` `[通讯]` (Carnegie Mellon University), Anatole Gershman (Carnegie Mellon University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合LLM-代理架构EpisodeSim，通过将经典AI的脚本、框架、目标、计划等结构转化为自然语言控制状态并由LLM调用解释，实现对社会情节的持续、连贯模拟。

**💡 创新点**

创新点在于将传统AI的情节管理结构以自然语言形式持久化为控制面板，让LLM在每个回合中参考并更新该结构，从而保持脚本、义务、物质状态和闭幕条件的持续一致。

**🔧 技术方法**

核心技术包括：大型语言模型（GPT类）、世界主控（World Master）用于维护共享现实、事件/效果账本、脚本堆栈、义务账本；记忆、规划、环境地面化和自我反思等LLM代理能力；自然语言控制状态的生成与更新。

**📊 数据集**

实验使用两组自定义场景：Hearthline商务午餐和晚间酒店入住（共6个情节），并在每个场景下分别运行三种条件（A、B、C）共18次。

**📈 对比分析**

比较方法是三种条件的对照实验，采用七类错误标注（canon_error、state_error、floor_error、script_error、agenda_error、role_policy_error、closure_error），并对每种条件下出现的主要情节级错误进行计数。结果显示：完整EpisodeSim（条件A）在所有6次运行中无重大情节级失败；而最小适当性基线（B）和地面化基线（C）在所有6次运行中均出现全部情节级错误。

**⚠️ 局限性**

局限性包括：样本规模极小（仅两场景、三条件、三次运行）；评估依赖人工裁决，缺乏独立评审与一致性指标；模型运算成本高，单回合可能涉及数百个API调用；未对比基于场景清单的实现，且无法完全排除跨域提示效应。

---

## 251. A Novel Space-Time Coding Architecture for Rydberg Atomic Quantum Receiver-Based Systems

**arXiv ID:** 2609.01180 | [PDF](https://arxiv.org/pdf/2609.01180v1)

**作者:** Asifa Zannat `[一作]` (Tampere University), Ertugrul Basar `[通讯]` (Tampere University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种针对Rydberg原子量子接收机（RAQR）的低复杂度时空编码架构，使MIMO链路在实数线性观测模型下实现可靠传输。

**💡 创新点**

创新点在于将强参考混频与实正交时空块编码相结合，保留编码矩阵的正交性，允许匹配滤波符号级检测，达到完整的发送-接收多样性。

**🔧 技术方法**

采用了实数正交时空块编码（OSTBC）、Rydberg原子量子接收机、强参考混频、匹配滤波符号检测以及贝叶斯误码概率分析。

**📊 数据集**

使用仿真数据：Rydberg原子（52D5/2–53P3/2）在5 GHz信号下的多路径模型（23个簇，每簇20条路径），以及N_r = 4个蒸汽细胞的接收阵列。

**📈 对比分析**

通过与空间多路复用（SMUX）和PRSS方案在相同信噪比下进行BER对比，结果显示RAQR‑OSTBC在高SNR时实现全N_t N_r多样性，BER曲线更陡峭，优于基准方法。

**⚠️ 局限性**

局限性包括对强参考信号的依赖、仅针对点对点链路、对参考信号失效和多用户场景未考虑，需要进一步研究非理想参考条件和网络扩展。

---

## 252. Local Reference Geometry Residual Augmentation for Imbalanced Time Series Classification

**arXiv ID:** 2609.00093 | [PDF](https://arxiv.org/pdf/2609.00093v1)

**作者:** Chuanhang Qiu `[一作]` (University of Southampton), Anthony Bagnall `[通讯]` (University of Southampton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了本地参考几何（LRG）框架，用于在已训练的时间序列特征空间中识别并修复少数类样本的局部几何失效。

**💡 创新点**

创新点在于利用训练样本的局部参考框架，仅通过标准化、软亲和度和LDA投影生成有符号残差坐标，将局部训练支持信息注入特征，从而在不重新训练编码器的情况下提升不平衡分类性能。

**🔧 技术方法**

技术包括k-means聚类构建参考中心、软亲和度与半径计算、标准化残差与LDA投影、以及在固定特征上训练轻量级线性分类头。

**📊 数据集**

实验使用UCR和Bake Off Redux的138个单变量不平衡时间序列任务，覆盖多种预训练、学习和随机变换特征。

**📈 对比分析**

通过与原始特征、标准化分类器以及多种不平衡处理（SMOTE、焦点损失等）以及后编码器校正方法的配对比较，LRG在平衡准确率、宏F1和少数类F1上平均提升约3-7个百分点，尤其在高不平衡比下表现最为显著。

**⚠️ 局限性**

局限性包括仅在单变量数据上验证，参考框架使用固定半径和单维LDA投影可能无法捕捉高维局部异方差，且对极端稀疏少数类支持的鲁棒性尚未深入探究。

---

## 253. DiagEvo: Diagnosis-Guided Self-Evolution via Hierarchical Error Memory

**arXiv ID:** 2609.00768 | [PDF](https://arxiv.org/pdf/2609.00768v1)

**作者:** Xincheng Wei `[一作]` (Chinese University of Hong Kong), Yao Zhang `[通讯]` (Nankai University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

DiagEvo提出了一种基于自我对弈的自进化框架，利用LLM诊断器从求解器的失败轨迹中提取错误原因，并将其组织成层次化记忆，进而指导下一轮问题生成；

**💡 创新点**

其创新点在于：①把求解失败的原因视为可诊断的学习目标并以层次化记忆存储；②采用双置信过滤剔除高冲突的伪标签；③在生成策略中混合自由探索与因子导向生成并结合记忆状态动态调整比例；

**🔧 技术方法**

技术手段包括：LLM诊断器（Qwen3-Instruct-2507等）、GRPO优化挑战者、双置信过滤（绝对与相对置信度阈值）、层次化错误原因记忆、因子导向与自由探索混合生成；

**📊 数据集**

使用的基准数据集共9个，涵盖数学推理（MATH‑500、GSM8K、OlympiadBench、Minerva Math、AMC）和通用推理（MMLU‑Pro、SuperGPQA、GPQA‑Diamond、BBEH）；

**📈 对比分析**

通过与R‑Zero、Absolute Zero、SPICE、R‑Few、DARC等方法在相同模型与基准上对比，DiagEvo在所有三种求解器（Qwen3‑4B、Qwen3‑8B、OctoThinker‑8B）上均实现了最高平均准确率；以Qwen3‑8B为例，其数学推理平均达72.3%，比R‑Zero高4.5个百分点，且整体性能超过DARC；

**⚠️ 局限性**

局限性包括：诊断器规模对性能提升的收益有限；只在数学问题上训练，跨域提升仍受限；伪标签质量仍受求解器一致性影响；缺乏对非LLM系统的验证。

---

## 254. MROP: Mask-Region Optimized Purification Against Backdoor Attack in Deep JSCC

**arXiv ID:** 2609.00786 | [PDF](https://arxiv.org/pdf/2609.00786v1)

**作者:** Seongkyu Yang `[一作]` (Chungnam National University), Jonggyu Jang `[通讯]` (Chungnam National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无需重训练的推理时输入纯化方法MROP，用于抵御深度JSCC中的输入补丁后门攻击。

**💡 创新点**

通过在推理阶段对每像素应用Gumbel‑Sigmoid掩模优化来定位触发区域，再利用总变差正则恢复该区域，实现对任意触发补丁的高效检测与清除。

**🔧 技术方法**

核心技术包括：Gumbel‑Sigmoid温度退火的稀疏掩模优化、基于自编码器重构误差的检测、以及总变差（TV）约束的区域恢复。

**📊 数据集**

在CIFAR‑10（32×32）和STL‑10（96×96）两个公开图像数据集上进行评估。

**📈 对比分析**

与梯度掩模基线相比，MROP在中心与通用补丁场景下将攻击成功率降至0%，同时将纯净区PSNR提升约10–15 dB，SSIM亦大幅改善；在不同模型（DeepJSCC、SwinJSCC）、补丁尺寸与分辨率下保持鲁棒性。

**⚠️ 局限性**

局限性在于依赖白盒访问完整JSCC模型并假设可自行重构输出，且对极端高分辨率或极大补丁尺寸的适用性尚未充分验证。

---

## 255. LOOMSUM:Weaving Quantitative and Narrative Evidence for Faithful Long Text-Table Summarization

**arXiv ID:** 2609.00241 | [PDF](https://arxiv.org/pdf/2609.00241v1)

**作者:** Meng Zhou `[一作]` (University of Toronto), Wei Yuan `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了一个无训练的长文本–表格摘要框架LoomSum，并设计了表格基准可信度指标TGF。

**💡 创新点**

核心创新在于将表格量化事实与叙事分析显式跨模态对齐，并通过语篇规划保持二者的关系；同时将可信度拆分为数值基础、分析支持和关系一致性三维。

**🔧 技术方法**

技术上利用大型语言模型进行断言提取、跨模态对齐、合并与规划，并在生成阶段使用少量示例的上下文学习；评估则采用自动断言级别判断与人类标注对齐。

**📊 数据集**

实验数据集包括英文金融长文多表摘要数据集FINDSum（ROO与Liquidity子集）和中文金融表格文本摘要数据集USTT。

**📈 对比分析**

与传统提取式、生成式与多模态基线（TextRank、BART、BigBird-Pegasus、Table-RAG、GCG、Direct-LC、RAG-Sum）对比，LoomSum在ROUGE、数值选择（NS）和TGF分数上均取得最优或接近最优表现，尤其在关系一致性（RC）上显著优于所有方法。

**⚠️ 局限性**

局限性包括：模块化流水线易累积错误；评估仅限金融领域，未验证在科学、医学等其他表格文本场景；TGF依赖自动断言提取与LLM评判，导致分析支持维度相关性较弱且缺乏更大规模的人类验证。

---

## 256. Authority Bias in Conversational Search Engines for Academic Paper Recommendation

**arXiv ID:** 2609.00248 | [PDF](https://arxiv.org/pdf/2609.00248v1)

**作者:** Uthman Jinadu `[一作]` (Georgia State University), Yi Ding `[通讯]` (University of Tennessee)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过内容控制的反事实审计，系统性地在大语言模型（LLM）论文推荐任务中，保持标题和摘要不变，仅改变作者声望、会议/期刊声誉、h‑指数、引用量与机构声誉等权威元数据，测量并比较不同模型在推荐结果上的变化，以评估其权威偏差。

**💡 创新点**

① 在推荐任务中首次引入内容受控的反事实实验设计，直接因果推断权威元数据对结果的影响；② 利用逻辑回归与支配分析从实验数据中推导五种权威信号的经验权重；③ 构建公开的 17,898 次评估基准，涵盖 8 个模型（开放权重与前沿闭合权重），并揭示了“说-做差距”和前沿模型在轻度去偏指令下的反向效应。

**🔧 技术方法**

• 反事实元数据操作与单轮 top‑1 交互式提示；• 逻辑回归 + z‑score 标准化 + 重要度（dominance）分析，用于估计权威信号权重；• 三种提示变体（中性、轻度去偏、强度去偏）与统计检验（McNemar、χ²、Wilson 置信区间）用于评估偏差强度和去偏效果。

**📊 数据集**

1,250 篇计算机科学论文（25 个主题 × 50 篇/主题），按高/中/低/新兴引用层级采样；利用 Semantic Scholar 与 OpenAlex 采集标题、摘要、作者 h‑指数、会议/期刊声誉、引用量和机构信息；共 250 条手工构造的研究查询。

**📈 对比分析**

与 8 个 LLM（5 个开放权重 7B–9B，3 个前沿闭合权重）在 3 条元数据条件（原始、翻转、提升）与 3 条提示变体下进行 18,000 次实验，解析 17,898 条回应；主要指标为翻转率、方向性、模型敏感度、指令效应。实验发现：翻转条件下 39.2% 的推荐发生变化，提升条件下 21.7%；不同模型翻转率差异高达 2.83 倍；强度去偏提示将翻转率降低约 12.9pp，但仍剩 31.4%；轻度去偏提示在前沿模型上反而导致翻转率上升。

**⚠️ 局限性**

• 仅覆盖 7B–9B 规模的开放权重模型，未覆盖更大模型；• 仅单轮 top‑1 推荐，未考察多轮对话、top‑k 排序或检索增强；• 只使用标题+摘要作为内容，未加入全文；• 查询为人工设计，缺乏真实搜索日志；• 仅计算机科学领域，其他学科可能表现不同；• 未进行人工相关性或质量评估，难以排除模型原本不佳的推荐原因。

---

## 257. Good Memory Has ECC: Evaluating the Memory of Vision-Language Models Beyond Accuracy

**arXiv ID:** 2609.00103 | [PDF](https://arxiv.org/pdf/2609.00103v1)

**作者:** Shmuel Berman `[一作]` (Princeton University), Jia Deng `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ECCBench 评估大型模型记忆能力的新基准，强调效率、压缩与校准三维度。

**💡 创新点**

创新点在于将记忆评估从单一准确率扩展为三轴(ECC)，并提供统一的 FLOPs 成本度量与可选择性放弃机制。

**🔧 技术方法**

采用流式写入/读取接口、Lempel–Ziv 率衡量压缩、覆盖率与收益率评估校准，并在不同输入熵与成本情境下测试模型。

**📊 数据集**

使用合成低熵/高熵文本和视频序列，以及从 EPIC‑Kitchens‑100 和 SoccerNet 提取的自然视频问答数据集。

**📈 对比分析**

通过准确率–FLOPs 曲线、压缩率（准确率差异）与覆盖率–成本曲线比较模型，结果显示 VLM 在文本压缩上有一定优势，但在视频压缩、容量和校准方面表现不足；非 Transformer 架构（RetNet、TTT‑MLP 等）在压缩与校准上明显优于 RoPE Transformer。

**⚠️ 局限性**

局限性包括：仅测试单查询场景；压缩测量受近满分准确率限制；对真实硬件延迟未考虑；校准评估仅针对选择性放弃；未评估闭源 VLM；合成数据可能与实际场景差异大。

---

## 258. ConvDeck: Conversational Paper-to-Slide Generation via Stage-Specific User Feedback

**arXiv ID:** 2609.00226 | [PDF](https://arxiv.org/pdf/2609.00226v1)

**作者:** Tarik Can Ozden `[一作]` (University of Illinois Urbana Champaign), James Matthew Rehg `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 ConvDeck，一种多代理的论文转幻灯片生成流水线，在幻灯片生成的不同阶段（大纲生成与幻灯片渲染）引入分阶段对话式微调，支持用户在生成流程中逐步细化内容和结构。

**💡 创新点**

创新点在于将对话式交互与生成流程对齐，形成阶段特定的思考-说话-执行循环（think‑act），让用户在大纲层面做结构决策、在幻灯片层面做视觉细化，从而显著提升用户目标满足率；同时设计了多代理协作与局部编辑机制，避免全量重生成。

**🔧 技术方法**

主要技术包括：多代理流水线（基于 ArcDeck 结构）、LLM（GPT‑5、Gemini‑3 Pro、Qwen3‑VL‑32B‑Instruct）、Docling 文档解析、ReSpAct 思考-说话机制、PPTXGenJS 生成 PPT、VLM 评估、网页检索与 arXiv 信息补充。

**📊 数据集**

使用 ArcBench 数据集（100 篇机器学习论文与其官方幻灯片对），并采用 LLM 模拟器与小规模人类实验验证。

**📈 对比分析**

实验对比了 5 个单次生成基线（HTML、PPTAgent、SlideGen、SlideTailor、ArcDeck）和 2 个对话式后期编辑基线（ConvHTML、AutoSlides）。在 50 条预定义用户目标上，ConvDeck 的目标满足率达 79%，在大纲相关类别（内容包含/排除、叙事结构）达到 90%+；与作者预制幻灯片相比，ConvDeck 在 GPT‑5 评判下的胜率为 87%，在视觉与叙事质量上与人类参考相近；计算成本与现有对话式系统相当，平均每篇论文约 281k token（$0.82）。

**⚠️ 局限性**

局限性包括：实验中主要使用 LLM 模拟器，真实用户反馈的多样性与优先级尚未充分验证；对话多轮增加了计算开销；视觉微调偶尔产生坐标错误导致布局失效；在开源模型 Qwen3‑VL 上表现不佳；生成内容可能出现事实错误或信息缺失，需要人工校验。

---

## 259. Value Over Language Model: Detecting Original Contribution in Writing

**arXiv ID:** 2609.00700 | [PDF](https://arxiv.org/pdf/2609.00700v1)

**作者:** Vibhhu Sharma `[一作]` (Cornell University), Sarah Dean `[通讯]` (Cornell University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种名为 VOLM 的无训练、无标签的框架，用于量化人类作者在 LLM 辅助写作中贡献的价值，并在新闻、学术同行评审和说服性作文等三大域中进行评估。

**💡 创新点**

创新点在于：①从信息量而非表面文本比例衡量 LLM 参与度；②以 LLM 在仅给定任务描述时可生成的“替代级”文本为基准；③通过递归提取-重构管道与对数概率“惊讶度”对比，剔除风格偏差；④不需要训练或标注数据，提升适用性。

**🔧 技术方法**

使用技术包括：LLM 生成的内容抽取器与重构器（基于 Llama 3.1 8B），WordNet 词义规范化与 Lesk 词义消歧，按粒度级别的提取，基于对数概率的惊讶度计算，统计显著性检验（两侧 t 检验），并对得分曲线做积分得到 VOLM 分数。

**📊 数据集**

使用数据集：All the News（100 篇新闻，标题为任务描述），ICLR 2023 评审（200 篇评审，完整论文与评审指引为任务描述），PERSUADE 2.0（200 篇说服性论文，题目为任务描述）。每篇人类文本对应用相同任务描述生成的 LLM 复制文本，长度限制在 10% 范围内。

**📈 对比分析**

比较方法：对人类文本与 LLM 参考文本分别按 0~1 的 11 级粒度提取、重构并计算平均每 token 对数概率；在每级别对两组得分分布做两侧检验，找到首次显著且持续拒绝的粒度 i*；VOLM = 1 - i*/k。实验显示人类文本在粒度 0.1 即可显著区分于参考 LLM 文本；对内容保持的变体（如 LLM 重构、回译）与人类曲线高度重合，表明方法对风格稳健；与不同模型或人化版本相比，得到中间分层，证明方法能捕捉信息差异。总体性能：能够在三大域中可靠区分人类与 LLM 生成文本，且对表面风格变动不敏感。

**⚠️ 局限性**

局限性包括：①对抽取器的设计敏感，若抽取器保留过多风格信息，导致 LLM 生成重构时惊讶度受词汇差异影响，产生非零 VOLM；②依赖评估模型 M，若 M 与作者使用的 LLM 或训练语料差异大，可能产生误判；③缺乏明确阈值或多模型投票来判断“显著”贡献；④对极短或极长文本、非英语或高度专业化文本的适用性尚未充分验证；⑤因为无训练，可能无法捕捉更细粒度的贡献分布。

---

## 260. Revisiting Feedback-Driven LLM Code Repair: A Replication and Exploratory Java Extension

**arXiv ID:** 2609.00362 | [PDF](https://arxiv.org/pdf/2609.00362v1)

**作者:** Louis Lalonde `[一作]` (Polytechnique Montréal), Foutse Khomh `[通讯]` (Polytechnique Montréal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 FeedbackEval 反馈驱动 LLM 代码修复基准进行了部分复现，并在 Java 环境中进行探索性扩展，评估不同反馈类型与提示策略对单次修复成功率及成本的影响。

**💡 创新点**

① 证明原 Python 结果在不同模型和设置下的可再现性；② 揭示语言、工具链和变异策略对反馈效果的影响；③ 通过成本敏感度分析展示轻量级提示更具性价比；③ 提供了 Java 版扩展数据集与实验脚本。

**🔧 技术方法**

使用 GPT‑4o、Claude 3.5 Sonnet 进行代码修复；Chain‑of‑Thought、Few‑Shot、提示消融等提示技术；编译器（javac）与 JUnit 测试框架产生的反馈；Wilson 置信区间、Cochran 的 Q 检验与 McNemar 检验等统计方法。

**📊 数据集**

Python：来自 CoderEval 与 HumanEval 的 394 个任务；Java：随机抽取 50 个 CoderEval 任务，利用 GPT‑4o‑mini 注入逻辑与编译错误共 100 个变异实例；所有数据已公开存档。

**📈 对比分析**

对比方法：在原始四种反馈（测试、编译器、人工、简单）和多种提示变体下，使用 Repair@1 指标评估成功率，并计算总成本与每次成功修复成本。结果显示：Python 复制中测试反馈仍优先；Java 中编译器与简单反馈表现相近；提示复杂度对成功率影响不显著，但 Chain‑of‑Thought 成本最高。

**⚠️ 局限性**

① 只复现了部分原始研究，未覆盖所有实验维度；② Java 扩展使用 LLM 生成变异，可能降低难度；③ 仅单次修复，未考虑迭代调试；④ 评估仅覆盖单文件、单语言，缺乏跨文件或多语言真实场景；⑤ 人工反馈使用 LLM 模拟，未能完全反映真实人类反馈；⑥ 依赖特定 LLM 与工具版本，模型更新或停用可能影响可复现性。

---

## 261. CARE: Contrastive Anchor-based Rubric Evolution for Large Language Model Post-Training

**arXiv ID:** 2609.00892 | [PDF](https://arxiv.org/pdf/2609.00892v1)

**作者:** Siyuan Li `[一作]` (Tsinghua University), Jinli Suo `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 CARE 框架，利用高质量 anchor 与最高分 roll‑out 的对比，对动态 rubric 进行修复与提升，以提升 LLM 的指令遵循和长文本生成质量。

**💡 创新点**

创新点在于：①将最高分 roll‑out 与 anchor 对比，分为 Adaptive（修复漏洞）和 Chase（提升判别）两条分支；②引入 veto 奖励消除奖励劫持；③针对动态 rubric evolution 的三大失败模式提供解决方案。

**🔧 技术方法**

采用强化学习 (GRPO)、LLM 验证器/评判器、前沿模型 (GPT‑4.1) 生成 anchor、对比学习与 veto 奖励等技术。

**📊 数据集**

使用 WildChecklist‑9K 训练集（含 anchor），并在 Arena‑Hard‑2.0、InfoBench、FollowBench 进行评估。

**📈 对比分析**

与 DPO (RLCF)、SFT‑anchor、Rubric RL、Rubric RL+Universal Criteria、Online Rubrics 等基线对比，CARE 在所有基准上均获得最高分，并在 300 步训练中持续提升 GPT‑4.1 anchor 的 win‑rate。

**⚠️ 局限性**

局限在于：①每步都执行 rubric evolution，缺乏自适应触发；②缺乏对生成 rubric 质量的客观评估；③对 anchor 与 LLM judge 的偏见敏感，需要人工审核。

---

## 262. A Lagrangian View of Flow Matching

**arXiv ID:** 2609.00198 | [PDF](https://arxiv.org/pdf/2609.00198v1)

**作者:** Peyman Milanfar `[一作]` `[通讯]` (Google), Peyman Milanfar (Google)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过从拉格朗日视角推导不变性原则，得到一个准线性对流偏微分方程，并证明其特征曲线即为流匹配（Flow Matching）和整流流（Rectified Flow）的直线轨迹，从而解释了为什么直线流能够实现大步长生成。

**💡 创新点**

创新点在于提出“目标恒等不变性”这一不变性条件，将生成过程从宏观的连续性方程转为粒子级的对流方程，并通过特征线方法直接推导出直线轨迹；同时给出了Jacobian惩罚机制解释传统扩散模型步长受限的原因，并通过交叉特征诊断与Reflow蒸馏技术阐明了特征交叉导致的性能瓶颈。

**🔧 技术方法**

主要技术包括拉格朗日微分分析、泰勒展开、总微分、矩阵雅可比、对流偏微分方程求解、特征线法以及对流场中的Jacobian惩罚机制。

**📊 数据集**

论文未在实验部分使用公开真实数据集，主要通过一维两模态交叉示例和理论推导进行验证；若有实验，则可能采用标准图像数据集（如CIFAR‑10/100、ImageNet）进行基准比较。

**📈 对比分析**

与传统扩散模型（需数千步）和现有流匹配方法（需数十步）比较，本文提出的诊断与Reflow蒸馏可在单步或极少步长下获得相近或更优的生成质量，实验结果表明在直线轨迹下可实现接近传统多步生成的样本质量。

**⚠️ 局限性**

局限性在于对流方程推导假设完美不变性，实际神经网络训练会出现特征交叉导致雅可比发散；同时单步生成仍需额外蒸馏步骤，且在高维复杂数据上完全消除交叉仍是挑战。

---

## 263. Influence of Logging Frameworks on Bind9

**arXiv ID:** 2609.00954 | [PDF](https://arxiv.org/pdf/2609.00954v1)

**作者:** Max Schrötter `[一作]` (University of Potsdam), Bettina Schnor `[通讯]` (University of Potsdam)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了主机IPS在高带宽网络中的日志子系统瓶颈，并提出并实现了FIPS IPC来提升日志性能，验证其在BIND9上的效果。

**💡 创新点**

提出基于线程本地无锁共享内存环形缓冲的IPC，支持多消费者多生产者，并可直接替代syslog API，显著降低日志写入开销。

**🔧 技术方法**

采用共享内存环形缓冲、无锁同步、Hyperscan正则匹配、eBPF数据包过滤，以及对BIND9的二进制日志改写和syslog替换。

**📊 数据集**

使用Trex生成的DNS流量（来自255个IP），并在BIND9上测试1万至1千万包/秒的负载。

**📈 对比分析**

与文件日志、syslog、journald、DNSTAP及FIPS二进制/系统日志对比，结果显示FIPS IPC几乎无额外开销，日志覆盖率与回复率最高，IPS封禁速度提升2.5倍。

**⚠️ 局限性**

仅在UDP DNS场景验证，未覆盖TCP应用和更高速400/800 Gbps环境，且共享内存实现可能带来安全与兼容性限制。

---

## 264. Conversation Coach: A Voice-enabled AI System that Helps Practice Difficult Workplace Conversations

**arXiv ID:** 2609.00441 | [PDF](https://arxiv.org/pdf/2609.00441v1)

**作者:** Fanyou Wu `[一作]` (Amazon), Srinivasan H. Sengamedu `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并评估了两种语音交互式教练系统（端到端 Nova Sonic 2 与基于 ASR+LLM+TTS 的管线），为经理提供逼真口头练习，并在生产中部署给4万+经理；

**💡 创新点**

提出针对语音教练的域特定设计框架，系统化比较端到端与分层架构在延迟、成本、人格一致性等维度的权衡，并通过大规模部署验证实际使用模式；

**🔧 技术方法**

使用 Nova Sonic 2（端到端 Speech‑to‑Speech）、Amazon Transcribe（ASR）、Claude Sonnet 4.5（LLM）、Amazon Polly（TTS）、SSML、流式处理及 DISC+反应模式的 Persona 设计；

**📊 数据集**

采用 148 个人工合成场景（27 个配置 × 6）和 10 条真实对话做 LLM 评估基准，并用 UTMOS 评估语音质量；

**📈 对比分析**

通过对等 TTFA 延迟、成本估算、LLM 评估（4 个评测模型）和人工评审对比；结果显示 NS2 延迟 1.4 s、成本 0.05 $/会话，管线延迟 4.2 s、成本 0.39 $/会话；在人格一致性、推理质量上管线显著优于 NS2，语音质量几乎相同；

**⚠️ 局限性**

未验证对实际对话效果的因果影响；依赖合成对话和 LLM 评判；未提供语音交付反馈；只评估特定闭源模型，未探讨开源替代；数据集与评估受限于内部数据。

---

## 265. Prediction-Robust Service Deployment with Capacity-Aware Edge Admission

**arXiv ID:** 2609.00877 | [PDF](https://arxiv.org/pdf/2609.00877v1)

**作者:** Hailiang Zhao `[一作]` (Zhejiang University), Shuiguang Deng `[通讯]` (Zhejiang University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文提出了一种基于TTL的边缘服务部署策略CAPSUM（包括弹性子策略CAPSUM-E和容量感知版本），能够在面对预测误差时做出安全的部署决策。

**💡 创新点**

创新点在于把边缘部署问题映射为独立的Bahncard实例，借鉴PFSUM实现预测增强的竞争性保证，并引入可变证据门、shadow价格和密度驱动的容量驱逐机制。

**🔧 技术方法**

采用学习增强在线算法、PFSUM、滑动窗口历史统计、短期预测、shadow定价与密度驱逐等技术。

**📊 数据集**

使用三种合成需求场景（静态、闪光人群、移动热点）和真实的Globus Compute任务轨迹作为实验数据集。

**📈 对比分析**

通过与SUM、FSUM、LRU、LFU、Random以及从EDP‑A、OREO、uEDC‑L改编的基线进行对比，CAPSUM在所有场景下相对最佳基线降低了约33%–45%的归一化成本，并在可扩展规模下保持低延时。

**⚠️ 局限性**

局限性包括：未给出容量感知下的竞争比；依赖本地仅服务假设，无法覆盖跨节点协同决策；以及实验仅验证TTL模型，未考虑网络、冷启动等真实部署成本。

---

## 266. RAPIDMap: Rapid Multi-Agent Pipeline for Interpretable Disaster Mapping from Satellite and Street-view Imagery

**arXiv ID:** 2609.00046 | [PDF](https://arxiv.org/pdf/2609.00046v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 267. Reinforcement Learning Enhanced LLM Agents for Complex Vehicle Routing Problems

**arXiv ID:** 2609.00859 | [PDF](https://arxiv.org/pdf/2609.00859v1)

**作者:** Yi Chen `[一作]` (Sun Yat-sen University), Zizhen Zhang `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RLEA，一个多代理框架，自动生成适用于复杂车辆路径问题（VRP）的求解器代码。

**💡 创新点**

创新点在于：用轻量化的神经规划器（Soft Q‑learning）驱动动作选择；结合检索增强生成（RAG）和进化记忆（MLEM）实现快速决策与经验复用；通过RL优化减少LLM调用延迟，提升建模鲁棒性。

**🔧 技术方法**

技术包括：多模LLM（DeepSeek‑Reasoner、GPT‑4o、Qwen2.5‑1.5B），Soft Q‑learning，检索增强生成，Meta‑learning with Evolved Memory，强化学习与多代理协作。

**📊 数据集**

数据集为48种不同约束组合的VRP实例，使用Gurobi和OR‑Tools求解器进行评测。

**📈 对比分析**

与标准提示、Self‑Refine、Chain‑of‑Thought、Chain‑of‑Experts和DRoC对比；在OR‑Tools上成功率62.5%（比DRoC高16.67%），运行时错误率16.67%；在Gurobi上成功率43.75%（比DRoC高8.33%），错误率12.5%。

**⚠️ 局限性**

局限性包括：对动态实时路由场景适应不足；初期记忆稀疏导致性能波动；仅基于文本描述，未利用多模态信息；对极大规模VRP可能面临计算瓶颈。

---

## 268. Space Generative AI with Solar Energy Harvesting

**arXiv ID:** 2609.01062 | [PDF](https://arxiv.org/pdf/2609.01062v1)

**作者:** Jierui Zhang `[一作]`, Kaibin Huang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于太阳能采集的卫星端生成式人工智能框架，并通过最优等待时间、功率分配和扩散模型步骤数的联合调度来最大化端到端生成质量。

**💡 创新点**

创新点在于将计算与通信（C²）耦合在单一时间可充能缓冲区内进行统一优化，利用轨道可预见性引入分离原则和低凸包(LCE)实现闭式的功率控制，并给出基于Lambert W函数的低复杂度扩散步骤选择公式。

**🔧 技术方法**

核心技术包括：轨道几何衍生的太阳能采集模型、扩散式文本到图像生成（Latent Diffusion Model）、能量因果约束下的通信吞吐量最大化、LCE原理的功率分配以及闭式求解的Lambert W分支分析。

**📊 数据集**

实验采用了公开的图像生成模型（如Stable Diffusion/Latent Diffusion）预训练权重，并在标准图像数据集（如ImageNet/COCO）上评估生成质量（CLIP分数）和传输吞吐量。

**📈 对比分析**

与单纯计算集中或通信集中基线相比，联合C²调度在中等能量条件下可提升约30%–50%的端到端CLIP分数，吞吐量也保持或略优；闭式解与完整穷举搜索几乎一致，计算复杂度从O(|𝒩|)降为O(1)。

**⚠️ 局限性**

局限性包括仅考虑单任务单卫星场景、忽略电池容量与峰值功率限制、假设轨道和太阳能采集完全可预见、以及使用单一预训练模型，未对多用户或多任务能量竞争进行建模。

---

## 269. GenScale: A Benchmark for Relative Object Scale in Image Generation and Editing

**arXiv ID:** 2609.00525 | [PDF](https://arxiv.org/pdf/2609.00525v1)

**作者:** Lingxiao Li `[一作]` (Boston University), Boqing Gong `[通讯]` (Boston University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了Genscale基准，评估图像生成和编辑系统在相对物体尺度上的表现，并设计了基于人类标注的序数尺度评估协议以及一种可迁移的后处理修正代理；

**💡 创新点**

创新点在于：①创建了专门针对相对尺度的基准与评测框架；②引入人类校准的五级尺度评分体系；③提出了一种模型无关的局部几何尺度修正代理，能够在不改动生成模型的情况下提升尺度合理性；

**🔧 技术方法**

采用了扩散式生成模型、Gemini大语言模型进行尺度判定、SAM2分割、DepthAnythingV2深度估计、InsertAnything等编辑后端；

**📊 数据集**

使用了COCO和LVIS构建常见物体尺寸知识库，Amazon Berkeley Objects提供产品尺寸与参考图像，构成900张图像级样本、1,643对尺度关系；

**📈 对比分析**

对比了当前顶尖生成模型（Nano Banana 2、GPT-Image-2、Z-Image、Qwen-Image等）在三类任务中的尺度误差和可接受比例，发现即使是最佳模型也只能达到约56%的可接受率；后处理代理在所有任务中均显著降低尺度误差并提升可接受比例；

**⚠️ 局限性**

局限性包括：仅覆盖物理尺寸稳定的可识别类别，排除柔性、细粒度或上下文依赖物体；假设目标物体可见且可局部编辑；基准在不同视角、遮挡、域等场景的泛化性待进一步验证。

---

## 270. A Compact Robotic Finger with 2-DoF MCP Joint Embedding DoF-Selective Passive Continuously Variable Transmission for Wide Force-Speed Operating Range

**arXiv ID:** 2609.00769 | [PDF](https://arxiv.org/pdf/2609.00769v1)

**作者:** JaeHyung Jang `[一作]` (Korea Advanced Institute of Science and Technology), Jee-Hwan Ryu `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出一种紧凑型双自由度MCP关节机器人手指，采用柔性选择性被动连续可变传动（CVT）实现弯曲自由度的力-速度可变传动，同时保持外展/内收的直传动。

**💡 创新点**

创新点在于对MCP关节的功能分化，仅在弯曲自由度嵌入被动CVT，并使用输出侧移动滑轮结构实现力感知的瞬时时刻臂变化，无需额外驱动、传感器或控制。

**🔧 技术方法**

使用了双线索布线、移动滑轮启发的被动CVT、弹簧恢复、力学Jacobian建模以及实验验证。

**📊 数据集**

实验采用自制装置测量力、角度和滑轮位移，未使用公开数据集。

**📈 对比分析**

通过与固定R=5 mm、R=25 mm两种传动比较，输出力放大最大4.19倍、平均3.63倍；实验还验证了重复性和手球滚动能力，性能显著提升。

**⚠️ 局限性**

局限在未集成小型执行器、未评估高DoF手指、未研究动态性能与长期耐久性。

---

## 271. Core-periphery identification in massive networks

**arXiv ID:** 2609.00008 | [PDF](https://arxiv.org/pdf/2609.00008v1)

**作者:** Eric Yanchenko `[一作]` (Akita International University), Eric Yanchenko `[通讯]` (Akita International University)

**通讯引用:** 220 | [OpenAlex ID](https://openalex.org/A5006784072)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于边列表的分治算法，用于在大规模网络中高效识别核心-外围结构。

**💡 创新点**

创新点在于将核心-外围指标的贪婪优化与随机边采样相结合，并且完全使用边列表而非邻接矩阵，显著降低内存占用和计算复杂度。

**🔧 技术方法**

采用边列表随机采样、贪婪标签切换、目标函数评估、LaF 等技术，利用并行化实现算法加速。

**📊 数据集**

在合成 SBM 网络和真实 SNAP 数据集（Amazon、YouTube、Twitch、Google）上进行实验，Google 网络包含近 1400 万条边。

**📈 对比分析**

与基于邻接矩阵的分治算法和简单度排序基线对比，实验表明新算法在运行时间上更快（可快 10⁴ 倍），在目标函数值上提升 1~2 个数量级。

**⚠️ 局限性**

局限性包括对采样比例 q 和子样本数 B 的经验性调参、对可用计算核心数量的依赖，以及在极大网络中仍需高效的边列表随机读取。

---

## 272. Learning Task-Specific Antibody Representations via Function-Aware Masking

**arXiv ID:** 2609.00518 | [PDF](https://arxiv.org/pdf/2609.00518v1)

**作者:** Ayan Goel `[一作]` (Georgia Institute of Technology), Amirali Aghazadeh `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了在抗体语言模型预训练中引入功能感知的掩蔽策略，以便将掩蔽位置与抗体的不同功能先验对齐。

**💡 创新点**

创新点在于提出“function‑aware masking”框架，允许通过多种生物学先验（如CDR位置、亲和位点、框架结构、进化突变等）定向掩蔽，从而在不增加额外参数的情况下实现功能导向的表示学习；并进一步设计了混合掩蔽策略来平衡多功能任务的泛化。

**🔧 技术方法**

主要技术包括：基于RoFormer的抗体语言模型；掩蔽采样算法（权重化与连续区间掩蔽等）；混合掩蔽策略的随机混合与动态训练曲线；下游线性/双线性探针评估；以及使用IgFold等工具进行结构预测。

**📊 数据集**

使用数据集：从Observ​ed Antibody Space (OAS) 提取约50万条重链可变域序列；利用IMGT/ANARCI注释得到CDR边界；使用AntiBERTa2的教师模型预测亲和位点概率；对框架进行IgFold结构预测；以及SAbDab、TDC等公开抗体结构和可开发性数据。

**📈 对比分析**

与传统随机掩蔽做对比，使用七项下游任务（CDR3填充、亲和位点预测、接触图、结构映射、可开发性回归）进行线性/双线性探针评估。结果显示，专用掩蔽在其对应任务上提升最高可达14%（结构相关）或5.9倍（CDR相关）；混合掩蔽在多数任务上平均排名提升，整体性能波动减小，表现优于随机掩蔽。

**⚠️ 局限性**

局限性包括：需要先验标签（可能存在准确性或获取成本差异）；实验仅在约38M参数的中等规模RoFormer上验证，未探索更大模型的效果；混合掩蔽策略手工设计，缺乏系统化的优化方法。

---

## 273. Inverse kinematic solution for generic 3R positional robots using Conformal Geometric Algebra

**arXiv ID:** 2609.00311 | [PDF](https://arxiv.org/pdf/2609.00311v1)

**作者:** Abhilash Nayak `[一作]`, Durgesh Haribhau Salunkhe `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种利用共形几何代数（CGA）求解通用3R位置机器人逆运动学模型（IKM）的框架，将IKM转化为两个圆的相交问题，直接得到关于θ₂的一元多项式；

**💡 创新点**

创新点在于：①通过CGA把传统的代数消元过程消除；②无需退化条件即可统一求解所有3R机器人；③提供了对逆解几何分布的直观理解；

**🔧 技术方法**

主要技术包括共形几何代数（CGA）与Lie代数、旋转/平移运动学的幂指数式表示、圆与平面的CGA表示以及四元数/双向量的计算；

**📊 数据集**

论文未使用公开数据集，仅通过一个给定DH参数的示例机器人演示方法并求得四个逆解；

**📈 对比分析**

文章未给出与传统代数/几何方法的定量性能比较，只展示了示例求解过程和所得逆解；

**⚠️ 局限性**

局限性：①缺乏大规模实验或对比评估；②未讨论计算复杂度和数值稳定性；③仅适用于3R位置机器人，未对6R或其他结构做进一步扩展。

---

## 274. Towards a Belief-Based World Model for LLM Agents

**arXiv ID:** 2609.00455 | [PDF](https://arxiv.org/pdf/2609.00455v1)

**作者:** Shubham Kumar `[一作]` (University of Illinois Urbana-Champaign), Saurabh Jha `[通讯]` (IBM)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了让LLM代理直接查询世界模型中当前状态不确定性的Belief-Based World Models（BB-WMs）方法，证明其能提升在部分可观测长时限任务中的决策效果。

**💡 创新点**

创新点在于将传统的仿真接口与显式不确定性推断相结合，通过自然语言接口让LLM能够实时获取关于当前状态的概率分布，从而支持认知型（epistemic）与目标导向型（pragmatic）行动的协同决策。

**🔧 技术方法**

主要技术包括：手工构建的状态空间与贝叶斯更新、基于规则的预测模块（如WALL‑E）、自然语言查询接口、以及ReAct/ReflAct框架下的LLM决策流程。

**📊 数据集**

使用了两个文本游戏数据集：ALFWorld（家庭任务）和ScienceWorld（科学实验），每个均提供随机初始化的部分可观测对象位置。

**📈 对比分析**

与仅使用仿真接口、仅使用记忆接口或仅使用贝叶斯接口的基线进行对比。实验表明，BB‑WM在ALFWorld上对Llama‑3.1‑8B和Qwen3‑14B的SR@1提升约30–40%并显著降低平均步数；在ScienceWorld上平均奖励和奖励/步数同样出现显著提升，说明两种接口互补。

**⚠️ 局限性**

局限性包括：贝叶斯模型为手工规则，难以扩展到更大规模或更复杂的连续环境；仅在文本游戏中验证，缺乏对真实机器人或金融等高维任务的评估；贝叶斯更新依赖于简化的状态空间，未来需研究可学习的、可持续更新的BB‑WM。

---

## 275. Hints Help But Do They Teach? Evaluating Skills Transfer in Code Generation

**arXiv ID:** 2609.01106 | [PDF](https://arxiv.org/pdf/2609.01106v1)

**作者:** Will Badr `[一作]` `[通讯]` (University of Leeds), Will Badr (University of Leeds)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在代码生成任务中，提示（hint）能否真正提升模型能力，还是仅是让模型产生本就可能出现的成功解。

**💡 创新点**

创新点在于：① 结合可执行评测（EvalPlus）做严格的通过/失败判定；② 设计一套完整的对照实验（相关提示、无关提示、无提示多采样、重放等）；③ 对提示所诱导的激活向量进行几何稳定性、单点注入、低秩子空间等机制性分析；④ 探索虚拟 KV 前缀压缩上下文的效果；⑤ 通过隐藏状态探针实现跨基准的正确性解码。

**🔧 技术方法**

主要技术包括：自适应提示梯度、温度采样、持久激活注入、单点补丁、低秩子空间估计、虚拟 KV 前缀训练、隐藏状态线性探针、Bootstrap 统计、McNemar 检验、AUROC 分析。

**📊 数据集**

使用的数据集为 HumanEval+ 和 MBPP+（分别为 164 和 378 个 Python 题目，均通过 EvalPlus 的基础+扩展测试），以及四类人工构造的上下文定义程序族用于前缀实验。

**📈 对比分析**

对照方式：相关提示 vs 无关提示 vs best‑of‑8 无提示采样；激活方向的稳定性与对齐；持久注入与重放的成功/损伤率；低秩子空间与随机/打乱对照的跨样本性能；隐藏探针与令牌概率、表面特征、TF‑IDF 的 AUROC 对比。结果显示：相关提示可救 36/79 失败（45.6%），无关提示 19/79（24.1%），而 8 次无提示采样已能通过 46/79（58.2%）；持久注入产生 14 次成功、18 次失败，净增益无显著提升；低秩子空间未显著优于对照；隐藏探针在跨基准上 AUROC 分别为 0.806/0.780，优于基线但 top‑one 改进未达到统计显著。

**⚠️ 局限性**

限制包括：仅在两款 3B 级模型上验证；实验不匹配提示长度/尝试次数导致对比不完全；重放不可完全消除非确定性；激活注入仅在特定层/位置测试；低秩子空间样本量小；虚拟 KV 前缀实验仅采用单一训练目标；隐藏探针依赖八次采样和白盒激活，可能捕捉表面特征；未探测其它更大规模模型或其他提示/前缀策略。

---

## 276. Peg-in-Bench: A Modular Benchmark for High-Precision Robotic Insertion

**arXiv ID:** 2609.00906 | [PDF](https://arxiv.org/pdf/2609.00906v1)

**作者:** Yosel Delgado `[一作]` (National Institute of Advanced Industrial Science and Technology), Yukiyasu Domae `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可重构的 Peg-in-Bench 机器人插槽插入基准，用模块化 3D 打印件实现多样化插入与装配任务。

**💡 创新点**

创新点在于通过模块化零件实现任务级别的几何、容差、空间、装配组合多样化评估，而非固定任务实例。

**🔧 技术方法**

采用 3D 打印、磁性连接、Python 场景生成器与 ROS/UR 机器人控制。

**📊 数据集**

使用自定义的 Peg-in-Bench STL 组件和生成的场景 JSON；不基于公开数据集。

**📈 对比分析**

与现有基准如 NIST、ManipulationNet 等对比，Peg-in-Bench 通过任务多样性提高了鲁棒性；未给出数值性能。

**⚠️ 局限性**

局限在于仅控制任务配置，未考虑光照、摩擦、材料等因素，且依赖特定桌面固定方案。

---

## 277. Socrates went Nuclear: Comparing Interaction Strategies for AI systems in a Learning Context using Brain Sensing

**arXiv ID:** 2609.00584 | [PDF](https://arxiv.org/pdf/2609.00584v1)

**作者:** Alexandre Clin Deffarges `[一作]` (ETH Zürich), Pattie Maes `[通讯]` (MIT)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在50名参与者中比较了三种AI学习交互模式——无限制聊天机器人、Socratic（提示式）聊天机器人和EEG驱动的自适应训练系统，在核安全协议学习任务中评估其对知识获取、认知投入和学习表现的影响。

**💡 创新点**

首次将EEG实时认知投入反馈与AI交互模式结合，并对传统聊天机器人、Socratic提示与自适应系统进行直接对比，探讨即时测试与长期学习之间的关系。

**🔧 技术方法**

使用OpenAI GPT‑5.2进行回答生成与自动评分，Muse 2脑电帽进行EEG采集并计算Beta/(Alpha+Theta)投入指数，并通过线性混合模型与方差分析比较各模式表现。

**📊 数据集**

基于10分钟核安全协议教学视频和由IAEA出版物设计的30道开放式问题（预测、训练、后测各10题）构成的实验数据集。

**📈 对比分析**

采用单因素ANOVA和线性混合模型比较学习增益和EEG投入；结果显示无限制聊天机器人获得最高学习增益（Δ≈20），自适应模式获得最高EEG投入（p≈0.018），但两者均未显著优于Socratic模式。

**⚠️ 局限性**

主要限制包括：仅测量即时记忆未检验长期保持；使用自动评分可能存在偏差；样本量小且未考虑受试者AI熟练度与兴趣；EEG反馈未提供熟悉期，可能影响自适应效果。

---

## 278. What Survives the Next Model? Benchmarking LLM-Based Techniques Against Single-Prompts

**arXiv ID:** 2609.00468 | [PDF](https://arxiv.org/pdf/2609.00468v1)

**作者:** Nahian Salsabil `[一作]` (University of Virginia), Sebastian Elbaum `[通讯]` (University of Virginia)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对ICSE 2026会议中35篇使用大型语言模型（LLM）的软件工程技术论文进行系统评估，比较其复杂工具与在更先进LLM上一次性自动生成的单提示的性能；

**💡 创新点**

首次量化当前LLM技术的寿命与替代性，发现约37%–63%的技术可被新一代模型的单提示直接取代，并指出哪些任务与技术更易被替代；

**🔧 技术方法**

采用自动化单提示生成（黑盒与白盒）、Claude Sonnet 4.6推理、基于论文原始数据构造评估器，并将结果与原技术在相同输入/输出和评估指标下对齐；

**📊 数据集**

使用各论文公开的基准数据集（如HumanEval、BugFinding、代码生成与修复数据集等），以及对应的输入/输出对；

**📈 对比分析**

通过将单提示在同一模型与数据集上执行，计算相同度量（如pass@1、bug发现准确率等），结果显示单提示在代码生成与修复任务中常优于原技术，整体可替代率达37%–63%；

**⚠️ 局限性**

研究受限于样本数量有限、预算导致的采样不完整、模型更新与访问限制、数据集潜在泄漏风险，以及对技术与任务分类的粗略处理，这些因素可能影响结论的普适性与精确性。

---

## 279. You Cannot Photograph the Same Street Twice: Reliability Limits in Vision-Language Measurement of Urban Change

**arXiv ID:** 2609.00649 | [PDF](https://arxiv.org/pdf/2609.00649v1)

**作者:** Kaizhen Tan `[一作]` `[通讯]` (New York University), Kaizhen Tan (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估 Vision–Language 模型在街景长期变化测量中的可靠性，使用 Google Street View 连续相册对比，分析拍摄条件、模型参数、采样误差对感知得分的影响，并在实验中检验相机几何、图像降质、Prompt 顺序等因素。

**💡 创新点**

提出“可靠性阶梯”，系统分解同地点差异为采样、编码、Prompt 顺序、相机几何及真实变化的组成；首次量化拍摄条件与模型表现的交互，证明仅聚合数百个样本即可恢复可靠的变化信号；揭示模型选择决定获取方向。

**🔧 技术方法**

使用 Vision–Language 模型（如 Qwen、Gemini、Llama）对 12 维感知维度进行评分；利用梯度提升模型预测六个图像统计量对差异的贡献；对 16 种图像降解族进行剂量‑反应实验；对 Mapillary 短视频进行相机几何归一化。

**📊 数据集**

Google Street View 的 4,648 对连续 epoch 图像（435 个站点）与 5 个美国城市；Global Streetscapes 1,000 万张街景图用于校准降解；Mapillary 436k 图像用于相机几何实验；CityPulse 标注集用于变化点验证。

**📈 对比分析**

通过对照组（控制站点）和标记变化站点，计算感知差异与无变化差异的比值；使用 95% 置信区间、聚类标准误；实验显示同地点差异约为不同街道差异的 2/3，误差大于样本误差；在聚合数百对时，能显著区分四个维度的变化，检测阈值需至少 88 对变化样本。

**⚠️ 局限性**

仅基于 Google Street View 的固定虚拟摄像机，未覆盖不同摄像机几何的真实噪声；控制站点由人标注的“无重大改造”未记录细微变化；模型选择影响结果，未给出通用校正表；部分降解族难以与实际标签对齐；数据集覆盖仅 5 个美国城市，结果可能不具全球代表性。

---

## 280. Exact Payload-Decoupling Conditions for Pilot-Only BEM Channel Estimation With Application to OTFS

**arXiv ID:** 2609.00937 | [PDF](https://arxiv.org/pdf/2609.00937v1)

**作者:** Gianmarco Romano `[一作]` (Università degli Studi della Campania Luigi Vanvitelli), Amedeo Buonanno `[通讯]` (ENEA)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出了一种在高移动性、双重分散信道下通过基函数展开模型（BEM）实现的仅用导频的频道估计方法，并给出了实现无载波干扰（payload-independent）条件及其设计规则。

**💡 创新点**

创新点在于：①给出了匹配导频最小二乘估计与最大似然估计等价的必要且充分的“零导频-载波干扰（ZPDI）”条件；②提出了与BEM无关的支持分离规则，实现任意BEM下的精确载波去耦；③将该理论应用于OTFS调制，证明单导频全护罩模式既满足ZPDI又满足零导频-导频泄漏（ZPPL），实现简洁、低复杂度的在线估计。

**🔧 技术方法**

主要技术包括：基函数展开模型（BEM）建模、矩阵投影与最小二乘/最大似然估计、零导频-载波干扰与零导频-导频泄漏分析、OTFS时频映射、GCE‑BEM（泛化复指数BEM）与CE‑BEM对比、决策导向（decision‑directed）迭代接收机。

**📊 数据集**

使用的数值实验基准为3GPP TDL‑B高速移动信道（L=4，RMS延迟300 ns），覆盖125 km/h和500 km/h两种速度；实验采用OTFS网格K=128、M=16、QPSK调制，单导频全护罩配置。

**📈 对比分析**

与传统CE‑BEM、基于GCE‑BEM的参考估计以及Liu等人提出的迭代BEM接收机相比，ZPDI估计在500 km/h时BER仅落后2 dB于理想CSI，且无高SNR误差底线；在125 km/h时与理想CSI差距更小。决策导向迭代在ZPDI初始化后可进一步逼近理想CSI，显著优于仅使用CE‑BEM或参考估计。

**⚠️ 局限性**

主要限制包括：①对BEM建模误差的依赖，若实际信道与BEM不匹配会出现误差底线；②对导频/护罩配置的严格要求，需满足支持分离规则和足够的导频资源；③在极高Doppler或极短帧长时，BEM阶数受限导致识别不足；④实现时仍需预计算大矩阵的逆，若BEM阶数增大会提升离线复杂度。

---

## 281. Discrete-Time MDP Modeling for Multi-Item Capacitated Lot Sizing with Stochastic Demand Timing

**arXiv ID:** 2609.00004 | [PDF](https://arxiv.org/pdf/2609.00004v1)

**作者:** Léa Bayati `[一作]` (Université Paris-Saclay), Melek Rodoplu `[通讯]` (Université Paris-Saclay)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5053900066)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了多品种受限批量定价问题，在需求量已知但需求到达时间随机的情况下，采用需求层面决策并建模为离散时间马尔可夫决策过程；

**💡 创新点**

创新点在于将需求层面的生产与分配决策与随机到达时间结合，既捕捉了容量竞争，又能细化后续库存与滞后成本的动态变化；

**🔧 技术方法**

主要技术包括DTMDP建模、确定性基准比较以及基于状态反馈策略的遗传算法求解；

**📊 数据集**

实验使用了包含330个基准实例的公开数据集，对90个最困难实例进行深入测试；

**📈 对比分析**

在确定性基准中证明随机到达时间显著增加状态数和求解难度；遗传算法在可求精解的实例上平均最优差距为3.44%，困难实例低于5%，并实现约6.9倍的加速（95%置信区间±1.41）；

**⚠️ 局限性**

限制在于对无法在现有硬件上精确求解的大规模实例需要使用经验贝尔曼时间回归估算速度，且算法在极端大规模问题下可能面临内存压力。

---

## 282. Design principles to Increase Technology Self-efficacy for Older Australians with Mild Cognitive Impairment (MCI) and Older Carers

**arXiv ID:** 2609.00480 | [PDF](https://arxiv.org/pdf/2609.00480v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 283. Physically Plausible Video Generation via Visual-Semantic Chain-of-Events Conditioning

**arXiv ID:** 2609.00656 | [PDF](https://arxiv.org/pdf/2609.00656v1)

**作者:** Zixuan Wang `[一作]` (Sichuan University), Yinjie Lei `[通讯]` (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将物理可行视频生成转化为事件级的因果链条，通过物理驱动的场景图演化、关键帧条件以及对比语义引导，生成符合物理规律的视频。

**💡 创新点**

创新点包括：①基于物理公式的场景图增强实现事件级因果推理；②事件路由关键帧合成，将不同类型视觉变化映射到专门的编辑器；③对比语义指导（正负对比提示）以抑制不符合物理的动态。

**🔧 技术方法**

使用技术：LLM推理生成与更新场景图、符号物理公式检索与求解、Qwen-Image-Edit与SAM进行关键帧编辑、Diffusion模型结合Soft Keyframe Guidance进行残差注入，以及PCSG的正负语义提示。

**📊 数据集**

实验数据集：PhyGenBench、VideoPhy、PhyWorldBench、Physics-IQ。

**📈 对比分析**

采用PCA、SA/PC、IoU等物理可行性指标，与基线及其它PPVG方法对比，平均PCA达72.5%，在所有四个基准上均超越现有方法，表现最佳。

**⚠️ 局限性**

局限性：仅适用于短时视频；事件数上限导致无法覆盖极长时间演化；高度依赖LLM推理的准确性；缺乏3D空间几何与长时间动力学建模；生成速度受Diffusion模型限制。

---

## 284. A Certificate-Producing Cascade for Equational Implication: The SAIR EQT2 Stage 2 Solver

**arXiv ID:** 2609.00706 | [PDF](https://arxiv.org/pdf/2609.00706v1)

**作者:** Haobo Ma `[一作]` (ChronoAI Pte Ltd), Manuel Israel Cázares `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种单文件、成本优先的 SAIR 数学定理挑战解算器，实现了对单一魔法运算恒等式蕴涵问题的证书生成；

**💡 创新点**

创新点在于将结构化代数模型家族、有限模型搜索、中央群体族显式例证与无语言模型的有序单元超位置求解器整合为一个可被赛题判定器核验的完整流水线；

**🔧 技术方法**

采用系数匹配、有限表约束求解、中心群体族显式实例、Austin 对偶无限载体构造、以及基于 Knuth–Bendix 顺序的有序超位置求解与回放；

**📊 数据集**

使用了 SAIR Equational Theories 项目公开的六个公共数据集、Stage‑1 分布式探测集、Marathon manifest 以及官方 hosted playground 数据；

**📈 对比分析**

通过本地官方运行器对六个公共集进行回归测试，获得 1,889 行全部通过，分布式探测集 800 行全部通过，Marathon 100 行全部通过，Hosted playground 200 行全部通过；与 Vampire 等成熟求解器仅做可比性说明，未声称性能优势；

**⚠️ 局限性**

局限在于缺乏完整性证明、未覆盖隐藏评估集、搜索深度受时间/内存限制，且对部分极难实例仍无法在预算内产生证书。

---

## 285. TRUST: Threshold-Recalibrated Uncertainty-Safe Training for Certified Dismissal in Breast Cancer Screening

**arXiv ID:** 2609.00300 | [PDF](https://arxiv.org/pdf/2609.00300v1)

**作者:** Parham Hajishafiezahramini `[一作]`, Oscar Meruvia Pastor `[通讯]` (Memorial University of Newfoundland)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一种闭环阈值感知训练方法，用于在乳腺筛查影像中安全排除（dismissal）低风险检查，从而降低放射科医师的工作量并保持高癌症检出率。

**💡 创新点**

创新点包括：①在训练过程中实时重新计算排除阈值，并将其作为正样本惩罚信号；②使用一阶Clopper–Pearson置信上限对被排除病例的癌症发生率进行统计保证；③通过消融实验分离出动态阈值更新的贡献，证明其比固定阈值更有效。

**🔧 技术方法**

技术实现上：使用预训练的EfficientNet‑B5（来自Mammo‑CLIP）作为特征提取器，冻结参数；分类头由交叉熵、焦点损失和排除损失共同优化；每个epoch在校准子集上重新估计阈值；最终采用最大聚合得到病例级分数；评估时采用Clopper–Pearson一侧置信上限。

**📊 数据集**

数据集：NLBS（Newfoundland and Labrador Breast Screening，约6k例，149例癌症）和RSNA Screening Mammography Breast Cancer Detection Challenge（约12k例，486例癌症）。同时进行内部五折交叉验证和跨机构外部验证（RSNA→NLBS）。

**📈 对比分析**

比较方法：将五种训练配置（仅交叉熵、交叉熵+Brier、交叉熵+焦点、固定阈值排除、闭环阈值排除）在相同数据拆分下训练；在外部阈值搜索集上选取98%和95%检出率对应阈值后，在独立评估集上计算排除率、检出率和CPU99。闭环模型在两数据集上均实现最高排除率且最低CPU99；与固定阈值相比，闭环可提升4–5个百分点；AUROC虽不总与排除率同步，但闭环模型在高检出阈值下仍表现最佳。

**⚠️ 局限性**

局限性：NLBS癌症样本量少，阈值选取更敏感；未系统评估超参数（margin、λ_dismiss）对性能的影响；仅与相同模型的消融进行比较，未对比专门的选择性分类或共形风险控制方法；仅使用一种网络骨干和预训练；Clopper–Pearson置信区间仅适用于病例级别；实验仅为回顾性评估，缺乏临床流程和患者结局的前瞻验证。

---

## 286. Verdict Instability of OOD Scores under Reference Resampling

**arXiv ID:** 2609.00691 | [PDF](https://arxiv.org/pdf/2609.00691v1)

**作者:** Donghoon Lee `[一作]` (Hongik University), Shinjin Kang `[通讯]` (Hongik University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了使用有限参考集训练的后置OOD检测器的判决不稳定性，定义了可通过引导分布估计的闭式表达式来衡量判决的可靠性。

**💡 创新点**

提出了“判决不稳定性”这一概念，证明其可用类内离散度除以根号类样本数闭式计算，并揭示距离型和分散型得分的符号差异。

**🔧 技术方法**

使用引导抽样、方差分解、闭式公式推导、相关性分析、AURC曲线评估等技术。

**📊 数据集**

在CIFAR-10、CIFAR-100和DermaMNIST三大数据集上进行实验，利用自然类别不平衡的DermaMNIST检验计数效应。

**📈 对比分析**

通过对11种常见后置OOD得分与不稳定性的Spearman相关性和AURC曲线比较，发现大多数距离型得分与判决不稳定性呈负相关，而分散型得分呈正相关；误判符号导致拒绝策略比随机拒绝更差。

**⚠️ 局限性**

局限性包括对嵌入空间的可测度假设、对类不平衡的依赖、仅评估判决稳定性而非误分类准确率，且在极端噪声情形下可能不适用。

---

## 287. SFAD: Speculative Factuality-Aware Decoding

**arXiv ID:** 2609.00796 | [PDF](https://arxiv.org/pdf/2609.00796v1)

**作者:** Guanqiao Chen `[一作]` (MBZUAI), Lijie Hu `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SFAD 这一 speculative decoding 框架，通过构造 ConFide 数据集并用 DPO 训练专门的 draft 模型，在推理阶段利用 Epistemic Friction 检测幻觉并通过 Asymmetric Logit Steering 调整目标分布，以提升上下文忠实度且保持速度。

**💡 创新点**

创新点在于：①将 speculative decoding 与幻觉抑制结合，②设计 Epistemic Friction 作为双重条件的冲突检测器，③采用可控的 logit 注入（Asymmetric Logit Steering）实现对目标分布的精细修正，以及 ④通过 ConFide 的原子级扰动构造高质量对照样本。

**🔧 技术方法**

技术方法包括：speculative decoding、Direct Preference Optimization（DPO）、原子级事实分解与可控扰动、Jensen‑Shannon 散度权重的 Epistemic Friction、上下文可行性掩码、ReLU‑based logit 注入、动态门控 λ_t 等。

**📊 数据集**

使用的数据集：ConFide（由 LLM‑AggreFact 与 CG2C 构成）、ConFiQA、HotpotQA、PopQA、TriviaQA、TofuEval、XSum、CLAPNQ、ExpertQA、HAGRID、LLM‑AggreFact、GSM8K、Just‑Eval 等进行训练与评估。

**📈 对比分析**

与 CAD、AdaCAD、COIECD 等 decoding‑level 基线以及 Llama‑3.1‑70B frontier 模型对比，SFAD 在保持 2.48× 速度提升的同时，提升了 85.2 以上的 faithfulness 分数，整体性能位于基线与 frontier 模型之间，且在多项任务上超越所有 baselines。

**⚠️ 局限性**

局限性：需要先训练域对齐的 draft 模型并构造 ConFide，增加数据与训练成本；Epistemic Friction 的阈值 τ 在不同领域可能需要手动调优；若 draft 模型不够置信，可能导致纠正不足或误判。

---

## 288. On smallest synchronizing terms over constant alphabets

**arXiv ID:** 2609.01184 | [PDF](https://arxiv.org/pdf/2609.01184v1)

**作者:** Luisa Herrmann `[一作]` (TU Dresden), Richard Mörbitz `[通讯]` (TU Dresden)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了基于两符号字母表的确定性树自动机（DTA）同步阈值的新下界，突破了此前二次函数的限制，达到亚指数增长；

**💡 创新点**

创新点在于利用二元函数与一元函数的组合，构造可在 LCM 次循环后唯一同步的树结构，从而实现更高下界；

**🔧 技术方法**

主要技术是构造两种映射 δ_f 与 δ_w，结合 Landau 函数和素数分割策略，分析其迭代行为并证明最小同步树高度；

**📊 数据集**

无实际数据集，整个研究完全基于理论构造与数论证明；

**📈 对比分析**

通过与已有的平方下界和指数上界比较，展示该方法显著缩小了理论极限间的差距，证明在两符号字母表上可达到 e^{(1+o(1))√(n ln n)} 的同步阈值；

**⚠️ 局限性**

局限在于无法进一步提升到指数级别，目前方法仅能利用一次元迭代的极限，尚未突破 Landau 函数所给的上限。

---

## 289. Learning Feasibility-Aware Latent Spaces for Preference-Based Exploration of Procedural Automotive Wheel Designs

**arXiv ID:** 2609.00527 | [PDF](https://arxiv.org/pdf/2609.00527v1)

**作者:** Takashi Owaki `[一作]` (Toyota Central R&D Labs., Inc.), Hiroyuki Sakai `[通讯]` (Toyota Central R&D Labs., Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出一种可视化可行性感知潜在空间学习管道，先对汽车轮毂的程序化生成样本进行几何与有限元分析筛选，然后在筛选后的子集上训练VAE得到低维可交互空间，并可将功能度量（刚度、强度、重量）对齐到单一维度；

**💡 创新点**

创新点在于将可行性筛选嵌入离线表示学习阶段，确保交互空间仅包含合法设计，并通过监督对齐赋予潜在维度可解释的功能意义，从而显著提升用户探索效率与可行性率；

**🔧 技术方法**

主要技术包括几何可行性检测、基于Blender Geometry Nodes的程序化轮毂建模、有限元结构筛选（FEA）、卷积自编码器提取单辐射图像特征、变分自编码器进行降维与功能对齐、以及基于偏好学习的贝叶斯优化（PBO）；

**📊 数据集**

使用了一个9维程序化轮毂参数空间，枚举得到245万组合，筛选后保留约3万组合作为训练和测试数据；

**📈 对比分析**

比较方法包括仿真目标检索（IoU评估）与40人受试者的控制实验；实验显示5D可行性感知空间在目标检索中IoU提升约12%、可行率提升约40%；在人类实验中，5D条件的平均IoU和可行率均显著高于原始9D空间，且用户选择时间略短；

**⚠️ 局限性**

局限性在于模型仅为简化的程序化轮毂，未涵盖完整的工程约束与复杂载荷；筛选标准可能偏向传统设计，抑制创新方案；离线筛选和VAE训练需要大量计算；在更大或连续参数空间的可扩展性与可行性判断的普适性仍待验证。

---

## 290. Fine-Tuning Large Language Models to Classify Pull Request-Issue Alignments: Going Beyond Prompting

**arXiv ID:** 2609.01087 | [PDF](https://arxiv.org/pdf/2609.01087v1)

**作者:** Mustafa Yasir Altunhan `[一作]` (Bilkent University), Eray Tüzün `[通讯]` (Bilkent University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对大型语言模型进行微调，构建了用于四分类（Exact、Missing、Tangling、Missing&Tangling）的 PR‑Issue 对齐判定系统。

**💡 创新点**

创新点在于：①将 GPT‑4o 与多款开源 LLM 通过指令微调或分类头微调应用到 PR‑Issue 对齐任务；②采用 SHAP 解释模型决策，揭示代码差异和 issue/PR 文本对预测的影响；③在已有数据集基础上进行数据扩增与类别平衡，提升模型泛化能力。

**🔧 技术方法**

使用的技术包括：指令微调（GPT‑4o）、分类头微调（CodeLlama‑7B、CodeQwen1.5‑7B、StableCode‑3B、CodeGemma‑7B、Deepseek‑Coder‑6.7B）、Shapley Additive Explanations (SHAP) 解释。

**📊 数据集**

采用并扩充了先前的 PR‑Issue 对齐数据集（原 194 条记录），从 Transformers 库中追加 2000 条记录，最终得到 400 余条样本，并通过数据增强实现类别平衡。

**📈 对比分析**

与基线的提示式方法相比，微调模型平均提升了 6.15% 的准确率和 F1‑micro，14.69% 的 F1‑macro，整体表现最优的是 CodeLlama‑7B。

**⚠️ 局限性**

局限性包括：仅基于单一 Python 项目数据，难以验证跨项目/跨语言的泛化能力；假设 issue 为原子任务，可能导致标签偏差；数据标注主观性和缺乏外部上下文信息。

---

## 291. Instance-Guided Report Anchoring for Text-Free 3D Abnormality Segmentation in Chest CT

**arXiv ID:** 2609.00447 | [PDF](https://arxiv.org/pdf/2609.00447v1)

**作者:** Zhenyu Bu `[一作]` (Siemens Medical Solutions USA, Inc.), Chaowei Wu `[通讯]` (Siemens Medical Solutions USA, Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出Instance-Guided Report Anchoring (IGRA)，在训练阶段通过实例级别将CT影像特征与对应报告句子对齐，以提升3D胸部CT异常分割性能，同时在推理时不再依赖文本。

**💡 创新点**

创新点在于在保持同类异常实例与其对应报告句子对应关系的同时，仅在训练时使用该对应关系进行特征对齐，完成文本无关的推理且显著提升分割精度。

**🔧 技术方法**

采用冻结的BiomedCLIP文本编码器、全分辨率特征平均池化、余弦对齐损失、轻量级投影器与多标签3D分割网络，实现实例-报告的语义锚定。

**📊 数据集**

训练使用ReXGroundingCT（3,142例CT）数据集，外推测试在LIDC-IDRI、PleThora以及内部COVID‑19 CT数据集上进行零样本评估。

**📈 对比分析**

与纯图像、文本条件和文本监督方法对比，ReXGroundingCT验证集Dice提升22.5%（30.93 vs 25.25），在LIDC-IDRI、PleThora和内部数据集均超越基线且接近或匹配VoxTell的单发现性能。

**⚠️ 局限性**

局限性在于仅支持ReXGroundingCT定义的14类异常，需配对文本与标注，无法直接处理开放词汇或无标注报告的数据。

---

## 292. ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation

**arXiv ID:** 2609.00057 | [PDF](https://arxiv.org/pdf/2609.00057v1)

**作者:** Yitong Han `[一作]` (Singapore Management University), Mohammad Amanlou `[通讯]` (Singapore Management University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于价值信号的图预训练框架 ValueGraph，用来学习用户上下文化表示，能够同时捕获语义、结构和道德价值信息。

**💡 创新点**

创新点在于将Moral Foundations Theory 的道德价值信号作为噪声辅助监督，结合图自编码、对比学习与聚类的分层目标，构建全局与局部约束，使用户嵌入在价值维度上保持一致性并保持可分离。

**🔧 技术方法**

使用 MoralBERT 推断的十维价值向量、GraphMAE2 进行无监督图自编码、InfoNCE 对比损失、K-means 聚类损失以及基于 Graph 的 GNN 编码器。

**📊 数据集**

数据集涵盖 Reddit 以及多项 Twitter 话题（PHEME、Twitter16、BEARD、Twitter-Covid 等），总计 461,198 个会话图、13.6M 节点和 39.9M 边。

**📈 对比分析**

在立场检测（MT_CSD、RumourEval19）和 Twitter 机器人检测（TwiBot-22）任务中，与多类基线（GraphCL、GraphMAE2、BERTweet、SimCSE、ModernBERT、GCN/GAT/GTN 等）以及 GPT‑5.4 进行比较，均实现了显著提升（如立场检测准确率提高至 63%/77%，机器检测 macF1 最高提升 9% 以上）。

**⚠️ 局限性**

局限包括：价值信号仅为噪声估计，未能准确反映真实用户心理；对价值信号分布假设的敏感性；以及在包含完整多模态特征（文本+图+个人资料）的环境下提升幅度有限。

---

## 293. NSIDDx: A Design Framework for Neuro-Symbolic, Practitioner-First Differential Diagnosis in Low-Resource Settings

**arXiv ID:** 2609.00256 | [PDF](https://arxiv.org/pdf/2609.00256v1)

**作者:** Aarav Singh `[一作]` `[通讯]` (IIIT Naya Raipur), Aarav Singh (IIIT Naya Raipur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个离线运行的神经符号化差异诊断系统NSIDDx，集成LLM生成临床叙述、符号层对症状矛盾进行检测、可审计的命题逻辑审计字符串以及可视化的否定症状图表，并为临床医师提供即时的覆盖与剔除回退接口。

**💡 创新点**

创新点在于采用三元症状编码(+1/0/-1)实现对否定症状的结构化表示；在LLM输出与符号层评分之间呈现矛盾信息，促使医师主动干预；引入可审计的PL字符串和否定症状图，实现可解释性与可操作性的双重目标；提出五项面向低资源环境的临床NLP设计原则，并在离线硬件上完成完整推理链。

**🔧 技术方法**

技术栈包括量化版Qwen3.5-9B LLM、HPO/MONDO/PrimeKG知识图谱、MedSpaCy与MedEmbed向量匹配、网络图结构（NetworkX）、双模评分公式（矩阵得分+混合得分）以及FAISS向量检索的稀有疾病扫描。

**📊 数据集**

使用公开的CUPCase病例报告（3562例）作为评估数据，随机抽取500例作为低资源表现子集，另外抽取250例可精确匹配子集；知识来源包括HPO、MONDO、PrimeKG以及ZebraMap。

**📈 对比分析**

评估方法采用精确、子串、token、语义匹配四种指标；在随机子集DDx@5的语义准确率约75%，精确/Token准确率分别为6%/34%；在可精确匹配子集同样约75%/47%；稀有疾病扫描在两组中分别提升约23%/35%；结果显示LLM在词表边界表现不佳，符号层的矛盾检测能够提示潜在错误。

**⚠️ 局限性**

局限性包括仅支持英文、症状提取覆盖不足、知识图谱稀疏导致评分失真、LLM可能产生幻觉、缺乏临床医师用户研究、未对评分公式进行超参数调优、离线部署受硬件限制、缺少多语言支持以及仅考虑单次就诊推理。

---

## 294. ErgoAssist: Cognition-Aware Posture Feedback in Wearable Ergonomic Systems

**arXiv ID:** 2609.00440 | [PDF](https://arxiv.org/pdf/2609.00440v1)

**作者:** Sarmistha Sarna Gomasta `[一作]` (University of Massachusetts Amherst), Prashant Shenoy `[通讯]` (University of Massachusetts Amherst)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一款头戴式可穿戴设备，集成IMU姿态传感与脑电认知负荷估计，实现基于用户认知状态的智能姿势警报；

**💡 创新点**

创新点在于将姿势与认知负荷融合，利用认知上下文来决定何时触发警报，从而显著降低打断频率与警报疲劳；

**🔧 技术方法**

采用头戴IMU头跟踪、消费级脑电头带、时域与频域特征提取、随机森林/支持向量机分类器，并实现断点感知的警报调度算法；

**📊 数据集**

使用24名受试者在实验室收集的IMU+EEG数据，设置三种姿势（D1‑D3）与三种任务（休息、Stroop、算术），同时记录NASA‑TLX、CMDQ等问卷；

**📈 对比分析**

通过留一用户交叉验证和80/20拆分评估，姿势分类准确率约81%，认知负荷分类约90%；在实时实验中，与仅姿势警报相比，警报频率降低81%，姿势纠正率提升38%，可用性提升43%，任务成绩提升25%；

**⚠️ 局限性**

局限包括样本量有限、实验时长短、EEG易受运动噪声影响、缺乏长期自然场景验证，以及参数需针对个体进行自适应调节。

---

## 295. Restrict, Don't Retrain: Inference-Time VLM Guidance for Zero-Shot Aerial Segmentation

**arXiv ID:** 2609.00628 | [PDF](https://arxiv.org/pdf/2609.00628v1)

**作者:** Teresa DiMeola `[一作]`, Hong Xiao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过在冻结的通用基础模型与视觉-语言模型（VLM）之间的协同工作，利用VLM的场景解读来实现零样本遥感图像的像素级分割。

**💡 创新点**

创新点在于：①自动类别加权（ACW）让VLM挑选出图像中实际出现的类别并提升其分数；②少数类别识别（MCI）让VLM提供小目标的边框，随后通过无监督填充恢复像素级标签，从而弥补基础模型对稀疏小目标的漏检。

**🔧 技术方法**

使用了两种技术：冻结的多任务语义分割基础模型（基于ADE20K 150类词表）和Qwen‑VL等视觉‑语言模型进行链式思考式提示，以获取类别列表与边框；另外还采用了无监督实例分割填充器。

**📊 数据集**

在四个公开航空遥感数据集上评估：UAVid、Aeroscapes、DroneSeg 和 UDD6。

**📈 对比分析**

与单一基础模型以及仅加权或仅填充的基线相比，本工作在Aeroscapes、DroneSeg 等数据集上平均提升 7–8 分 mIoU，提升显著且统计显著；在视角极端或标签粗糙的数据集（UAVid、UDD6）提升有限，主要受 VLM 定位误差和标签粒度限制。

**⚠️ 局限性**

局限性包括：①VLM 在斜视角或复杂场景下的定位准确性不足，导致 MCI 无法有效补充；②对极为粗粒度标签的数据集（如 UDD6）由于类别映射冲突，难以充分利用 150 类词表；③整体运行时主要受 VLM API 延迟影响，限制了实时应用。

---

## 296. Training-Free Inpainting Across Domains with a Frozen Text-to-Image Diffusion Model

**arXiv ID:** 2609.00862 | [PDF](https://arxiv.org/pdf/2609.00862v1)

**作者:** Zhenhuan Wang `[一作]` (Chinese University of Hong Kong), Fengyi Yuan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用冻结的Stable Diffusion v1.5，结合边界–内部自适应控制器、持久PI状态与预定义释放调度，实现无训练的条件图像修复；

**💡 创新点**

在不更新权重、不学习专用条件通道的前提下，通过闭环潜在控制实现跨域修复，首次将冻结文本到图像模型直接用于三域条件修复；

**🔧 技术方法**

技术包括：确定性DDIM采样、已知区域投影、边界/内部目标的梯度反馈、PI结构的持久状态、四段预设释放计划以及全流程的闭环控制；

**📊 数据集**

数据集覆盖 AFHQ、CelebA‑HQ 与 Places2（共计3,500个样本、35个掩码协议）；

**📈 对比分析**

与训练无关的基线 LanPaint 与 PILOT 进行原生路比较，Step‑PI 在所有五个宏观指标上均优于两者；与训练好的修复方法对比，Step‑PI 虽无权重优化但在多项指标上接近或略低于训练模型；

**⚠️ 局限性**

局限包括：推理成本高（约30 s/图、15 GiB显存），仍不及专门训练的修复模型；对释放调度与参数的依赖需要手工设计；对极端掩码或复杂纹理的鲁棒性尚待验证。

---

## 297. Benchmarking spiking neural networks across sensing modalities on edge devices

**arXiv ID:** 2609.00026 | [PDF](https://arxiv.org/pdf/2609.00026v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 298. Capability-Gated Language Models: Security Composes, Utility Does Not

**arXiv ID:** 2609.00445 | [PDF](https://arxiv.org/pdf/2609.00445v1)

**作者:** Patrikas Vanagas `[一作]` (BPTI), Laurynas Lopata `[通讯]` (askEarth AG)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了“能力门控部署”（Capability‑Gated Deployment）方案，在单一模型权重内通过可配置的权限格（lattice）实现多主体（principal）细粒度访问控制；

**💡 创新点**

创新点在于将多主体安全和效用需求映射为同一权重下的分层配置格，实现了在单一模型权重内可组合的权限控制，并通过稀疏秩门控（sparse rank gating）在嵌套因子化网络中实现可调节的可访问性；

**🔧 技术方法**

使用了嵌套因子化（nested‑factorisation）网络、稀疏秩门控（sparse rank gating）、单次传播一阶归因（one‑pass attribution）进行配置搜索，以及基于可分辨率的安全与效用指标（行为/表示通道、保留集效用）进行评估；

**📊 数据集**

采用了WMDP数据集（生物、化学、网络安全三域）进行禁用域评估；MMLU（七个科目聚合为两大宏域）评估保留效用；MiniPile评估流畅度/困惑度；并在Qwen3‑1.7B和SmolLM2‑1.7B两大模型上进行实验；

**📈 对比分析**

通过A/B预注册实验对分裂数据集进行评估，所有主实验结果只在B集上一次性读取；结果显示：在安全性方面，门控配置在合并（join）后仍能保持或提升隔离效果（security composes），但在效用方面，合并会导致保留项目的准确率下降（no compositional bound）；相对传统全权限模型，门控模型在生物、化学域上能显著降低禁用域准确率，同时保持宏域保留的准确率；

**⚠️ 局限性**

局限性包括：安全性评估仅基于期望分布，缺乏最坏情况（如差分隐私）保证；效用方面无可组合界限，合并后可能导致流畅度和准确率下降；在网络安全域的门控效果有限，缺乏可行的门控配置；门控与过滤的区分在部分实验中未能稳健区分；

---

## 299. Controllable Image Captioning with Prompt-Conditioned Scene Rewards

**arXiv ID:** 2609.00709 | [PDF](https://arxiv.org/pdf/2609.00709v1)

**作者:** Jongyeop Hyun `[一作]` (POSTECH), Hyounghun Kim `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于场景图奖励的细粒度可控图像字幕生成方法，能够通过自然语言提示精确控制字幕强调的属性、关系、前景或背景；

**💡 创新点**

创新点在于将场景图分解为对象、属性、关系三类组件，并通过提示特定的正负权重将其作为奖励信号，实现在不改造模型结构或输入的情况下实现细粒度语义控制；

**🔧 技术方法**

使用的技术包括场景图对齐评分（对象匹配、属性与关系的链式思维LLM评判）、GRPO强化学习优化、以及对比式包含/避免（Include/Avoid）评估框架；

**📊 数据集**

使用的主要数据集有COCO、CompreCap、DOCCI用于训练与验证，另外构造了一个189张图的专用评测集（Semantic Control and Precision Evaluation benchmark）；

**📈 对比分析**

与零样本提示、仅做交叉熵微调、GRPO结合CLIP或原始CompreCap奖励的基线对比，实验显示在两个VLM骨干（Qwen2.5-VL-3B-Instruct与InternVL3-2B）上，SFT+GRPO方法在语义可控性、细粒度对齐及整体评分均明显优于基线，提升幅度约为15–20点；

**⚠️ 局限性**

局限性包括：依赖场景图解析与LLM评判，导致训练成本高；对更大模型的可扩展性尚未验证；评估指标基于LLM判别器，可能存在偏差与人类细节识别不足。

---

## 300. Comparison of Algebraic Block Multi-Coloring and Leiden Methods for Parallel Preconditioning in the ICCG Method

**arXiv ID:** 2609.00561 | [PDF](https://arxiv.org/pdf/2609.00561v1)

**作者:** Tomohiro Suzuki `[一作]` `[通讯]` (University of Yamanashi), Tomohiro Suzuki (University of Yamanashi)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于 Leiden 社区检测的自动分块框架，用于在不需要预先指定块数的情况下实现 ICCG 方法的并行预处理，并与传统的 ABMC 方法进行了对比。

**💡 创新点**

创新点在于：①利用 Leiden 算法自动提取稀疏矩阵的天然社区结构，避免手工调参；②比较了两种质量函数（模量化和常数 Potts 模型）对块结构与并行性能的影响；③系统地评估了分块参数对执行时间和内存访问的敏感性。

**🔧 技术方法**

采用的技术包括：Leiden 社区检测、模量化（modularity）和常数 Potts 模型（CPM）质量函数、ABMC 分块与颜色化、ICCG 与不完全 Cholesky 预处理、贪婪颜色化、Gini 系数评估块大小不均衡、L2 缓存命中率等硬件性能计数。

**📊 数据集**

使用了来自 SuiteSparse Matrix Collection 的八个真实 SPD 矩阵，包括 CFD、结构、热分析、回路仿真以及网格问题，平均非零数 per 行（ANZR）范围从 4.83 到 43.74。

**📈 对比分析**

通过比较 ICCG 的迭代次数、总执行时间和 L2 缓存效率等指标，LeidenCPM 在大多数矩阵上实现了与精调 ABMC 相当甚至更优的性能，且不需手工设定块数；Leiden 模块化在某些矩阵上因块大小失衡而性能低于 ABMC。

**⚠️ 局限性**

局限性包括：仍需调节分辨率参数 γ，分块成本高于 ABMC，特别是矩阵结构频繁变化时难以摊销；对极端块不均衡矩阵效果不佳；在单次求解中若已知最佳 ABMC 参数，Leiden 的加速可能被分块时间抵消。

---

## 301. Measuring Optimal Transport in Transformer Depth

**arXiv ID:** 2609.00748 | [PDF](https://arxiv.org/pdf/2609.00748v1)

**作者:** Alexandre Quemy `[一作]` `[通讯]` (Hother Labs), Alexandre Quemy (Hother Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Transformer 层在逐层迁移 token 状态云时是否遵循最优运输（Optimal Transport）方案，并评估其成本效率与对应映射的匹配度。

**💡 创新点**

首次从实证角度系统检验训练好的 Transformer 是否以最小成本实现云间迁移，并量化各层迁移映射与最优地图的相似程度。

**🔧 技术方法**

使用精确匈牙利分配（网络单纯形）、切片 Wasserstein 下界、校准技术、Spearman 排名一致性以及余弦相似度等方法来测量迁移成本和映射一致性。

**📊 数据集**

在 WikiText‑103 和 Pile 两大语料上，对 Pythia‑160m（12 层）和 Pythia‑410m（24 层）模型进行评估。

**📈 对比分析**

通过比较网络迁移成本与最优运输成本的比值以及迁移向量与最优地图向量的相关性，发现几乎所有可解析层都在最优成本范围内，且在训练后迁移映射与最优地图的匹配度随层深增加而提升，末层达 0.89/0.88 的排名一致性。

**⚠️ 局限性**

主要限制在于样本采样精度有限（仅 4000 位置可解析）、只覆盖两小模型和两小语料、以及中间层迁移幅度过小导致无法单层精确评估，未来需要更大样本、更深模型和更细粒度分析。

---

## 302. Validity-Aware Jailbreak Evaluation for Large Language Models

**arXiv ID:** 2609.00498 | [PDF](https://arxiv.org/pdf/2609.00498v1)

**作者:** Qilong Wu `[一作]` (University of Illinois Urbana Champaign), Varun Chandrasekaran `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SEAV（Sequential Epistemic and Action-Level Validation）框架，先将模型生成的 jailbreak 响应拆解为有序步骤，再通过检索和 LLM 判断逐步验证事实准确性、执行可行性和顺序一致性，最终给出操作有效性评分。

**💡 创新点**

创新点在于将评估从单一语义相似度转向基于操作可行性的逐步验证；引入四阶段管道（提取、步骤检验、排序检验、最终判定），并通过检索证据提升真实性检查，显著降低误报率。

**🔧 技术方法**

使用 LLM-as-a-judge（Gemini Flash/GLM‑5）进行语义理解与最终评分；检索检验采用 Google/Tavily 搜索并匹配外部证据；步骤提取与依赖推理采用序列化与依赖标注；最终聚合采用阈值式分数汇总。

**📊 数据集**

对多种公开 jailbreak benchmark（JailbreakQR、JBB、GPTFuzz、WildGuardMix、UltraSafety）以及诊断集 SD‑A 和 OrdSense 进行评估；使用公开数据集进行人工注释与检验。

**📈 对比分析**

与 5 个基线评估器（包括 StrongREJECT、JADES 等）对比，SEAV 在 SD‑A 诊断集上的假阳性率降低 14.9pp，重新判定 22.1%–51.0% 的先前成功为无效，评估结果在不同检索后端和评估模型上保持稳定，表明方法鲁棒性高。

**⚠️ 局限性**

局限性包括对 LLM 判断的依赖导致可能的偏差与误判；只评估单一响应，未考虑多轮交互；主要针对英文，跨语言与特定领域的可迁移性未知；检索结果质量影响验证可靠性；对细节完整性与安全风险的覆盖仍不完全。

---

## 303. XVAE-WMT: Explainable Wavelet-Temporal Variational Autoencoder for Blind Source Separation of Heart and Lung Sounds

**arXiv ID:** 2609.00238 | [PDF](https://arxiv.org/pdf/2609.00238v1)

**作者:** Yasaman Torabi `[一作]` (McMaster University), James P. Reilly `[通讯]` (McMaster University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研发了一种基于可解释变分自编码器的盲源分离方法，能在单通道心肺声音混合中分离心音和呼吸音。

**💡 创新点**

结合小波时频前端、输出软掩模、时序一致性正则和SHAP可解释性，实现无监督分离且能解释潜在空间维度。

**🔧 技术方法**

采用变分自编码器、连续小波变换、时序一致性损失、后置软掩模以及SHAP特征重要性分析。

**📊 数据集**

使用Kaggle呼吸音库与CirCor心音库混合合成数据以及HLS‑CMDS临床模型数据。

**📈 对比分析**

与ICA、NMF、LSTM、U‑Net、PC‑DAE等基准及不同VAE变体比较，XVAE‑WMT在SDR 26.8 dB、SIR 32.8 dB、SAR 28.6 dB、TEM/CEM等指标上均显著优于基线。

**⚠️ 局限性**

仍受限于单通道条件、对高噪声鲁棒性待验证以及潜在空间可解释性在不同数据分布下的泛化性。

---

## 304. BlockMGARD: Accelerating Adaptive Scientific Data Reduction with Region-of-Interest Error Control on GPUs

**arXiv ID:** 2609.00205 | [PDF](https://arxiv.org/pdf/2609.00205v1)

**作者:** Yanliang Li `[一作]` (University of Oregon), Jieyang Chen `[通讯]` (University of Oregon)

**通讯引用:** 1431 | [OpenAlex ID](https://openalex.org/A5063649910)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

BlockMGARD是一种GPU加速的有损压缩器，重新设计了基于变换的去相关算法，引入了共享内存上的In-cache Block分解、混合本地/全局层级以及基于块的ROI误差控制；

**💡 创新点**

创新点包括：1）利用共享内存与编译时查找表实现高效的In-cache Block分解；2）构建可调节的本地/全局混合层级并提供自动参数选择；3）实现块级ROI误差控制与保守误差传播策略，保证L∞误差边界；

**🔧 技术方法**

使用技术包括CUDA（H100）、共享内存、编译时索引表、混合层级分解、块级量化、核融合、以及基于队列理论的性能建模；

**📊 数据集**

使用了五个真实科学数据集：NYX、Hurricane ISABEL、SCALE‑LETKF、Miranda和S3D；

**📈 对比分析**

与MGARD‑X、cuZFP和VGC在L∞误差约束下对比，BlockMGARD在压缩吞吐上提升至4.2×、解压缩吞吐提升至9.1×，ROI模式下压缩比提高8.63×，并在4块H100上实现近乎线性扩展，I/O成本降低达14.8×；

**⚠️ 局限性**

主要限制是使用过多本地层级时压缩比可能下降，ROI误差传播会导致ROI附近额外位分配，以及仍需一定全局层级以获得最佳压缩效果。

---

## 305. Beyond the Clock: Measuring the Value of Adaptive Revision

**arXiv ID:** 2609.00874 | [PDF](https://arxiv.org/pdf/2609.00874v1)

**作者:** Ayushi Chadha `[一作]` `[通讯]`, Ayushi Chadha

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一个层次化潜在推理模型中，研究了管理者是否保留或替换子目标的元级控制问题，探索自适应时序决策与固定时钟决策在有限计算预算下的效果；

**💡 创新点**

通过在可枚举的控制时序空间上进行完整枚举、冻结检查点干预与对比实验，揭示“状态依赖”并不等同于“实际性能提升”，并提出一套可复制的诊断模板，验证自适应决策是否真正带来价值；

**🔧 技术方法**

使用Hierarchical Reasoning Model（HRM）与方向性目标投影，构建可分离特征触发器（基于贝叶斯阈值的可学习时序决策），结合冻结检查点、完整枚举与对比实验等技术；

**📊 数据集**

ConceptARC-mini数据集；

**📈 对比分析**

采用与固定时钟策略相同的计算预算，比较三种预设种子下的自适应策略与最优固定时钟策略，以微粒子级token准确率衡量。结果显示自适应策略从未超过最佳固定时钟；最佳固定时钟已捕获随机时钟提升约71%，剩余改进仅约0.13个百分点；

**⚠️ 局限性**

实验仅覆盖单一任务、单一模型与单一K=2时序空间；未验证更大规模或更复杂的元控制场景；自适应探索有限，且所有对比均基于冻结检查点，缺乏在线适应性验证。

---

## 306. Beyond Object Selection:Markerless Gaze-based Robot Placement at Arbitrary Position

**arXiv ID:** 2609.00478 | [PDF](https://arxiv.org/pdf/2609.00478v1)

**作者:** Yuzhi Lai `[一作]` (University of Tuebingen), Andreas Zell `[通讯]` (University of Tuebingen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一个无标记的基于视线的机器人操控框架，用于在任意位置放置物体；

**💡 创新点**

核心创新在于基于语义与空间关系的图结构参考选择、面向任务的Gaze‑Surface Intersection Error (GSIE) 评估指标以及对点、线与物体级别对齐方法的统一基准；

**🔧 技术方法**

采用语义图匹配、SuperPoint+LightGlue特征匹配、PnP+线约束、MVEE物体椭球细化等技术，结合深度重建与RGB‑D相机；

**📊 数据集**

使用自建的跨设备数据集（Frank A Emika Panda+Meta Aria眼镜）共30个桌面序列，另外对TUM RGB‑D的fr2/desk、fr3/long、fr2/dishes进行泛化测试；

**📈 对比分析**

与GOReloc、AR标记、FAM‑HRI、标签匹配等方法对比，本文方法在参考选择准确率0.95‑1.00、放置成功率0.93‑0.98，且实现约22 FPS，表现优于传统匹配与标记基线；

**⚠️ 局限性**

局限性包括依赖稳定的RGB‑D重建、对强遮挡与纹理弱场景的鲁棒性仍有提升空间、以及尚未将GSIE直接嵌入端到端优化。

---

## 307. MemoryWalker: Stop Training Agents on Contexts They Never Saw

**arXiv ID:** 2609.00865 | [PDF](https://arxiv.org/pdf/2609.00865v1)

**作者:** Zinco J `[一作]` (Alibaba Group), Jieping Ye `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在大规模语言模型代理中，因上下文压缩导致的训练与推理条件不一致问题，并提出精确与近似的纠正方法

**💡 创新点**

提出了两种精确的树状条件一致训练方案(LogitTree与4D注意力掩码)以及一种训练友好的近似方案SDCC，并证明其梯度等价性与行为一致性上界

**🔧 技术方法**

树状条件遍历、分支分割序列、4维注意力掩码、前向KL自蒸馏（SDCC）、GRPO强化学习、Qwen系列大模型与多种编辑器的实现

**📊 数据集**

Qwen3-4B、Qwen3.7-Air模型；白盒编辑器：TC‑RAG、AgentFold、MemexRL；黑盒部署：Claude Code、OpenCode；检索/问答基准：NQ、TriviaQA、HotpotQA、2Wiki、MuSiQue、Bamboogle、FRAMES；训练集为81,638条复合QA实例

**📈 对比分析**

对比Naive‑Compressed、Naive‑Full、LogitTree、4D‑Mask、SDCC等五种方法；实验显示精确方案与SDCC将logit漂移恢复至无压缩基准；SDCC在黑盒环境中仍保持相当或更优的奖励（Claude Code 37.5% vs LogitTree 35.9%），并显著抑制训练‑推理差距

**⚠️ 局限性**

4D‑Mask需白盒访问与自定义掩码，内存/计算成本高；LogitTree需多次反向传播；SDCC为近似解，仍存在残差；仅在基于eviction的压缩策略下验证，未覆盖更广泛压缩或稀疏/线性注意力模型

---

## 308. Topic Matching in the Wild: Benchmark and Lessons from Real-World ASR Transcripts

**arXiv ID:** 2609.00330 | [PDF](https://arxiv.org/pdf/2609.00330v1)

**作者:** Saman Rahbar `[一作]` (Dialpad Inc), David Rossouw `[通讯]` (Dialpad Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

论文内容缺失，无法得知具体研究工作

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法确定对比方法及性能表现

**⚠️ 局限性**

无法确定存在的局限

---

## 309. trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories

**arXiv ID:** 2609.00038 | [PDF](https://arxiv.org/pdf/2609.00038v1)

**作者:** Hadi Mohammadi `[一作]` (Utrecht University), Hadi Mohammadi `[通讯]` (Utrecht University)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5102718770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个无人工标注的支持台代理基准环境，使用脚本化的规则驱动代理完成任务并注入单一故障，生成精确的标签；随后对比评估不同LLM判决器在检测、定位、分类、校准和成本等维度上的表现。

**💡 创新点**

首次量化了“结果仅评判”在面对静默故障时的盲点，并通过对比步骤级评判实现高召回；同时发布完整的实验工件（环境、注入器、原始判决结果），实现完全离线可复现的评估流程。

**🔧 技术方法**

使用的技术包括：大语言模型（14B/8B GPT‑4）、规则引擎、故障注入器、Bootstrap置信区间、期望校准误差（ECE）等统计与评估方法。

**📊 数据集**

数据集为支持台模拟实例，包含六种业务情景（全价退款、补货费、过期窗口、不可退货、不同客户订单、已退款订单），以及六种单步故障（skipped_precondition、hallucinated_argument、ignored_observation、premature_stop、wrong_tool、unsupported_claim）。

**📈 对比分析**

比较了五种判决器（规则引擎、结果仅评判、步骤仅评判（14B）、步骤仅评判（8B）、Self‑Consistency 集成）。结果显示：步骤仅评判在无误报条件下实现约90%+的静默召回，规则引擎误报率为0但无法检测两种故障；Self‑Consistency 成本提升三倍却几乎没有性能改进。

**⚠️ 局限性**

局限性包括：仅在单一支持台领域评估，注入的故障过于干净且不具备真实多步错误的复杂性；模型规模受限（仅14B/8B），未探索更大或多模态模型；“静默”定义受环境可观测范围限制，可能与真实系统的可观测性不同。

---

## 310. Deploying and Evaluating a Smart-Agriculture Agentic Engine for Full-Season Soybean Farm Operations

**arXiv ID:** 2609.00106 | [PDF](https://arxiv.org/pdf/2609.00106v1)

**作者:** Ao Qu `[一作]` (Harbin Institute of Technology), Jie Liu `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并部署了FAIRY智能农机系统，完成从土壤准备到豆粕存储的全季节播种、灌溉、施肥、病虫害处理、收割和干燥等完整作业，并在哈尔滨工业大学的实际大豆试验田上进行实验评估。

**💡 创新点**

创新点包括：①基于“事件即一切”事件驱动的世界模型，将作物生长、天气、遥感、设备操作等统一为状态变化事件；②引入时序Kendall正确性（KTC）度量，能够把轨迹准确度与实际产量紧密关联；③构建分层可检索的农学技能库，实现专家操作知识的可复用与检索；④首次在真实农田中大规模对多种智能代理控制器进行全季节评估。

**🔧 技术方法**

使用技术涵盖：事件驱动执行框架ARE、基于物理的作物过程模型（土壤-生长-病虫害耦合）、多源传感与卫星/无人机数据接口、API工具调用（包括前沿云端与边缘vLLM）、多代理控制器（ReAct、Plan-and-Act、AutoGen等）、检索式技能注入、时序KTC评估、边缘部署性能分析。

**📊 数据集**

数据集：哈尔滨工业大学64梯田大豆试验田的现场观测数据（土壤、冠层、天气、光照、遥感、无人机多光谱/热像）、卫星Sentinel‑2产品、历史收成与产量记录；基于此构建了100个场景（L1原子任务、L2情节链、L3全季节），其中70个为L3测试集。

**📈 对比分析**

通过对九种代理控制器、四种上下文（零上下文、LLM专家、技能库、专家指令）和两大后端模型（Qwen3.6‑35B、DeepSeek‑V4‑Flash）进行评估，主要指标为产量损失、KTC、token成本与运行时。结果显示，专家上下文将产量损失从约22%降至≈5%，KTC在短任务中接近1，但在全季节任务中仍显著下降；KTC与实际产量最为相关；边缘模型在单任务上表现良好，长周期仍受限。

**⚠️ 局限性**

局限性：①长周期全季节决策仍存在显著产量损失，显示代理对累积耕作影响建模不足；②多代理协作引入协调成本，导致正确性与产量下降；③依赖专家写入的上下文或技能库，缺乏完全自动化；④实验仅覆盖单一大豆品种与特定田块，泛化性待验证；⑤当前评价依赖人工标注的“人类或acular”轨迹，易受版本漂移影响。

---

## 311. Behavior--Realization Separation for Constrained Physical Human--Robot Interaction

**arXiv ID:** 2609.00669 | [PDF](https://arxiv.org/pdf/2609.00669v1)

**作者:** Yongyan Cao `[一作]` `[通讯]` (Voryx Robotics LLC), Yongyan Cao (Voryx Robotics LLC)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了将交互行为（desired‑interaction dynamics）与机器人实现（physical feasibility）分离的架构，并实现了一种基于预测二次规划的执行层，支持在相同约束和决策变量下替换不同的记忆无关线性行为模型。

**💡 创新点**

创新点在于：① 通过加速度接口实现行为层与实现层的清晰边界；② 同一 QP 模板可被多种行为生成器（阻抗、共振等）共享，只需更改成本系数；③ 引入“同目标受约束/无约束对照”来拆解实现残差，明确约束干预与模型误差；④ 在仿真中实现在线行为切换（阻抗→共振→阻抗）而不重新构造求解器。

**🔧 技术方法**

使用技术包括：预测模型的递归时间步长二次规划（OSQP）、零阶保持的力预测、机器人逆动力学和雅可比矩阵线性化、正则化与速率限制、残差拆分与监测、热启动求解器、以及基于 MuJoCo 的仿真。

**📊 数据集**

使用的“数据集”是人工设计的仿真实验：平面单自由度点质量在 1‑3 s 期间受 12 N 线性力推，Franka‑FR3 7-DOF 机器人在 1‑3.5 s 期间受 20 N 按升余弦形状持续推力。所有实验均为 deterministic、无噪声、无延迟的力输入。

**📈 对比分析**

对比方法：将预测实现与“命令/速率受限的即时裁剪”反应式控制器在同一力、同一约束下比较；以及在极限扭矩预算下比较“全步约束”与“仅首步约束”。性能方面，预测实现能够保持工作空间边界约束（误差 < 0.1–0.2 mm），扭矩利用率 ~37%，残差 RMS 在 1–1.5 m/s² 级别；预测求解时间平均 3–4 ms，满足 20 ms 计划周期。

**⚠️ 局限性**

局限性：仅验证了记忆无关线性行为；未考虑状态耦合或学习型行为；未验证递归可行性与终端不变集；残差未归一化，难以跨坐标比较；未进行硬件实验或抗延迟/噪声的鲁棒性评估；缺乏能量/消耗性约束和通过能量抽水器证明的被动性。

---

## 312. Do Multimodal LLMs See Before They Read? Diagnosing Contextual Sycophancy

**arXiv ID:** 2609.00067 | [PDF](https://arxiv.org/pdf/2609.00067v1)

**作者:** Yi-Cheng Lai `[一作]` (Academia Sinica), Hen-Hsen Huang `[通讯]` (Academia Sinica)

**通讯引用:** 1209 | [OpenAlex ID](https://openalex.org/A5053932280)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个998案例的上下文条件诊断基准，用来评估多模态大语言模型在视觉-文本冲突情境下的“多模态情境sycophancy”行为。

**💡 创新点**

创新点在于将信息边界移动到不同阶段，以测试外部文本何时影响视觉推理，并提出了系统-2视觉仲裁（S2VA）以及泄露与隔离实验，揭示文本既能污染也能支撑视觉推理的双重角色。

**🔧 技术方法**

采用的技术包括两步提示（先形成视觉见证，再进行仲裁）、S2VA、Leaky Witness、Witness-Only、链式思维（CoT）、视觉至上提示以及基于GPT‑4o‑mini的自动判分器。

**📊 数据集**

使用的数据集为998个人工设计的“WHOOPS!”异常图像与ImageNet正常图像，并配以Gemini 3 Flash生成的问答及三种文本条件（真实、错误、无关），还额外对200个案例做了GPT‑4o重生成对照。

**📈 对比分析**

实验中在异常图像的错误文本条件下，S2VA在六个模型上平均提升约30个百分点，显著优于联合条件、Witness‑Only，显示仲裁带来的显著收益；在正常图像或不同文本源时表现差异，表明最佳信息边界取决于模型与文本来源。

**⚠️ 局限性**

局限性包括：基准仅覆盖单句文本、生成式文本来源、少量模型、人工标注且未涵盖视频/音频等多模态；缺乏多评审验证；未评估冲突信号、拒绝或不确定性等实际部署场景。

---

## 313. Probabilistic Model Checking of Autoregressive Neural Sequence Models

**arXiv ID:** 2609.00838 | [PDF](https://arxiv.org/pdf/2609.00838v1)

**作者:** Helge Spieker `[一作]` (Simula Research Laboratory), Arnaud Gotlieb `[通讯]` (Simula Research Laboratory)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种针对自回归变换器模型的概率模型检测流水线，能够从生成过程抽象出DTMC并进行PCTL验证，给出对输入空间的覆盖曲线；

**💡 创新点**

创新点在于将DTMC抽象与概率模型检测、CEGAR精化、最大似然反例提取结合起来，提供对概率质量分布和领域约束的可证明的保守估计；

**🔧 技术方法**

使用DTMC抽象、PRISM模型检查、PCTL规范、CEGAR精化循环以及基于负对数概率的最大似然路径搜索等技术；

**📊 数据集**

在两个案例中使用数据集：CAPP（7840个可枚举的工业工件描述，GPT-2模型）和SMILES分子生成（200个ZINC前缀，BPE 约2700 词表的GPT-2模型）；

**📈 对比分析**

与传统的单一测试集准确率相比，流水线能够揭示模型隐藏的概率质量、顺序违规和化学有效性缺陷，给出最佳采样覆盖率（pass@N）以及安全解码温度区间；在CAPP中从15%训练样本起，顺序正确率覆盖率升至97%以上；在SMILES中发现约66%的结构完整终端为化学无效；总体运行时间在M2 Max上提取/验证每个输入约 485+43 秒；

**⚠️ 局限性**

局限性包括：DTMC抽象为下近似，需阈值和深度限制；对GPT‑2类模型和温度采样敏感；未验证更长生成长度、不同变换器架构或复杂采样策略；需要手工提供语法或外部验证器。

---

## 314. Accelerating Reinforcement Learning via MPC Solver-Gradient Guidance for Weights-varying MPC

**arXiv ID:** 2609.01061 | [PDF](https://arxiv.org/pdf/2609.01061v1)

**作者:** Baha Zarrouki `[一作]`, Johannes Betz `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Solver-Gradient Guided Reinforcement Learning（SG‑RL）框架，结合环境梯度与MPC求解器的敏感度，在成本权重自适应过程中提升样本效率和闭环性能。

**💡 创新点**

创新点在于将求解器的敏感度作为辅助指导信号，分别注入PPO的四个模块（sgsca、sglos、sgadv、sgcrt），实现梯度融合与策略学习的协同。

**🔧 技术方法**

使用技术包括可微分非线性MPC、PPO强化学习、求解器敏感度分析、以及四种梯度融合策略。

**📊 数据集**

实验数据集为两台全尺寸自驾赛车（Dallara AV‑24与Super Formula EAV‑24）的高保真仿真平台，并人为引入模型不匹配。

**📈 对比分析**

与传统PPO、固定权重专家、以及GBPL基准比较，SGRL在样本效率上提升40–70%，闭环返回提升30–60%，并在未见赛道上保持优异性能。

**⚠️ 局限性**

局限在于对预测模型的依赖，若模型失配严重可能误导梯度；同时方法需要额外调参且目前仅在PPO上验证。

---

## 315. EvoGS: Modeling Deformation Evolution for Dynamic Gaussian Splatting

**arXiv ID:** 2609.00994 | [PDF](https://arxiv.org/pdf/2609.00994v1)

**作者:** Wei Dong `[一作]`, Han Zhou `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

论文探讨了某种算法在特定任务中的应用，旨在提高性能和效率。

**💡 创新点**

创新点在于提出了一种新的优化策略，能够显著提升算法的处理速度。

**🔧 技术方法**

使用了深度学习技术，结合了卷积神经网络和强化学习。

**📊 数据集**

采用了公开的图像数据集进行实验，以验证算法的有效性。

**📈 对比分析**

与现有的几种主流算法进行了比较，结果显示该算法在准确率和速度上均有明显优势。

**⚠️ 局限性**

限制在于算法对特定类型数据的依赖性，可能在其他领域的应用效果不佳。

---

## 316. Fi-ImageNet-1k: An OOD Benchmark From the Inside of the ImageNet-1k Validation Set

**arXiv ID:** 2609.01027 | [PDF](https://arxiv.org/pdf/2609.01027v1)

**作者:** Ruslan Rozumnyi `[一作]` (Czech Technical University in Prague), Jiří Matas `[通讯]` (Czech Technical University in Prague)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个新的OOV（离散分布）基准数据集——来自ImageNet-1k验证集的未被重新标注为任何1000类的图像，经过人工三阶段评审和多源证据验证后确认属于外部类别，并为每幅图像提供了细粒度标签。

**💡 创新点**

创新点在于：①利用ReImageNet的标注错误作为OOV构建源，避免传统外部数据集的类别冲突；②设计了结合MLLM描述、逆向检索、原始标签及相似ID类的三阶段人工评审流程；③展示该OOV集对现有监督模型+后置检测器与零样本VLM的极大挑战，揭示负类别词表覆盖是瓶颈。

**🔧 技术方法**

采用的技术包括：人机混合标注流程、GPT-5.4生成开放词汇描述、逆向图像检索、VLM（CLIP、SigLIP等）对齐、Logit/Feature/Hybrid OOD检测器、负词表方法（NegLabel、EOE）以及自定义负词表生成策略。

**📊 数据集**

主要使用的数据集为ImageNet-1k验证集（50000张）以及从ReImageNet未分配类别的子集，构成新OOV数据集；在评测中还使用了ImageNet-O、NINCO、ImageNet-OOD等公开OOV基准以做对比。

**📈 对比分析**

实验对比了8种监督模型+21种后置OOD检测器和8种零样本VLM（含负词表变体）。在新OOV集上，最优组合FPR@95仍高于51%，远高于最难的IN1k-OOD（FPR@95≈16%）；使用NegLabel+SigLIP2-g在oracle负词表下可降至6%，但在可部署词表下仍为51%。

**⚠️ 局限性**

局限性包括：①需要人工评审和多源证据验证，工作量大；②负类别词表覆盖不足，导致VLM OOD性能受限；③数据集只覆盖ImageNet-1k，泛化到其他视觉域尚未验证；④对极其细粒度或模糊图像的判断仍有不确定性。

---

## 317. VerNav: Verifier-First Low-Latency Vision-and-Language Navigation

**arXiv ID:** 2609.00920 | [PDF](https://arxiv.org/pdf/2609.00920v1)

**作者:** Zhixin Wang `[一作]` (University of Electronic Science and Technology of China), Yongzhao Zhang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种以验证器为主的低延迟LLM导航框架，取代每步的自回归生成，采用批量候选动作验证；

**💡 创新点**

创新点在于：①利用批量验证实现极低决策延迟；②通过熵触发的自适应生成器仅在不确定时提供状态证据；③采用两阶段验证器对齐（VPO与强化学习微调）提升动作偏好。

**🔧 技术方法**

技术手段包括：批量验证器（Yes/No模板）、熵基触发机制、VPO（基于选取-拒绝对比学习）、一步强化学习微调、LoRA参数高效微调。

**📊 数据集**

使用R2R（Room-to-Room）数据集的val‑seen与val‑unseen分割进行评估。

**📈 对比分析**

与NavGPT、DiscussNav、MapGPT、NavCoT等基线相比，VerNav在R2R val‑unseen上仅0.08s/步决策延迟，比NavCoT低12.3×，且成功率为39.63%，与代表性LLM基准相当。

**⚠️ 局限性**

局限在于：①仍需较大模型（如Qwen2.5‑3B）以保证性能；②在RFT阶段对路径效率（SPL）略有下降；③依赖准确的熵阈值，阈值选择可能对不同场景敏感。

---

## 318. Subword Segmental BabyLMs: Learning to Tokenise for Sample-Efficient Pretraining

**arXiv ID:** 2609.01151 | [PDF](https://arxiv.org/pdf/2609.01151v1)

**作者:** Francois Meyer `[一作]` `[通讯]` (University of Cape Town), Francois Meyer (University of Cape Town)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 SubSegGPT（解码器式）和 SubSegDeBERTa（编码器式）两种可学习子词分段语言模型，在 BabyLM 低资源预训练任务中实现了子词分段的端到端学习。

**💡 创新点**

将可学习子词分段框架首次迁移到 GPT‑2 与 DeBERTa 体系，特别是在掩码语言模型中实现子词分段；利用混合词典/字符生成器与动态规划实现对所有可能分段的边际化。

**🔧 技术方法**

子词分段语言模型（SSLM）、动态规划求边际概率、混合词典与字符 LSTM 子词生成、字符级上下文编码器、DeBERTa 双向上下文、λ 混合门控制词典与字符概率。

**📊 数据集**

使用 BabyLM 2026 预训练数据（Strict 100M 词、Strict‑small 10M 词），子词评估使用 SIGMORPHON 2022 词形标注、BLiMP、SuperGLUE、BabyLM 官方评测管道。

**📈 对比分析**

与固定子词的 GPT‑2 与 DeBERTa 进行 Zero‑shot、Finetune 与 Human‑likeness 对比；在 Strict 轨道 SubSegDeBERTa 在大多数 Zero‑shot 任务上优于 GPT‑2（平均 +3.16 分），在 Strict‑small 轨道 SubSegGPT 在 Zero‑shot 与 Finetune 上均优于对应基准；Human‑likeness 任务表现不佳。

**⚠️ 局限性**

训练时间与 FLOPs 明显增加（SubSegDeBERTa 约 5 倍 GPU 小时），计算效率低；子词长度上限导致分段不完全；在极低资源下 MLM 目标稀疏，SubSegDeBERTa 受限于训练信号不足。

---

## 319. HELIOS: From midnight to noon, continuous outdoor urban scene relighting

**arXiv ID:** 2609.00901 | [PDF](https://arxiv.org/pdf/2609.00901v1)

**作者:** Hala Djeghim `[一作]` (Huawei Research Center), Désiré Sidibé `[通讯]` (Universite Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8d10c613-917e-4880-9716-17789f50e119` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了一种无监督单图像重照明方法，针对自动驾驶场景，可在昼夜之间连续调节光照。

**💡 创新点**

创新点：使用光照不变的反照率（albedo）作为条件避免身份崩溃；通过跨域蒸馏构建鲁棒的夜间反照率估计；用GPS太阳高度角替代文本提示，实现连续光照控制。

**🔧 技术方法**

技术：循环一致扩散模型（CycleNet基础）、ControlNet、Stable Diffusion V2.1；Albedo估计网络；太阳高度角位置编码；跨域蒸馏。

**📊 数据集**

数据集：nuScenes、Waymo、Pandaset三大自动驾驶数据集（约41k张图像）及额外的子集用于蒸馏（约6.8k张图像）。

**📈 对比分析**

与InstructPix2Pix、Qwen-Image、DiffusionRenderer、CycleNet等基线对比；在FID、CLIP、DINO、mIoU、Win/Fail率上均优于所有基线，特别是夜到日转换上表现突出。

**⚠️ 局限性**

限制：在极低光照极端夜景下仍难以准确估计反照率并生成高质量图像。

---

## 320. Triple-Bottom-Line Sustainability of Language Models for Edge AI: A Comparison Between SLMs and Quantized LLMs

**arXiv ID:** 2609.00665 | [PDF](https://arxiv.org/pdf/2609.00665v1)

**作者:** Jainil Dharmil Shah `[一作]` `[通讯]` (Purdue University), Jainil Dharmil Shah (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文比较原生小语言模型（SLM）与经过后训练量化的大型语言模型（LLM）在边缘部署的可持续性，并提出并验证一种三底线权重相等的Holistic Sustainability Score (HSS) 框架，针对30种模型配置进行系统测评。

**💡 创新点**

创新点在于引入了经济、环境、社会三维权重相等的HSS评分方法，实现了能力、效率与安全的统一评估；同时在同一实验平台上对多种量化算法与模型尺寸进行全面对比，揭示量化方法并非单调提升效率。

**🔧 技术方法**

技术包括 BF16、INT8、NF4、GPTQ 和 GGUF 量化实现；Zero‑shot 评测（MMLU、ARC‑Challenge、HellaSwag、GSM8K、TruthfulQA）评估能力；GPU 时延、吞吐量、峰值 VRAM、能耗和 JailbreakBench 的五个有害提示评估安全性；使用 min‑max 归一化计算 HSS。

**📊 数据集**

使用的数据集包括：Zero‑shot 评测的五个基准（MMLU、ARC‑Challenge、HellaSwag、GSM8K、TruthfulQA）以及 JailbreakBench 的五个有害提示；所有测量均在 NVIDIA A100 GPU 上完成。

**📈 对比分析**

通过将模型分为 SLM 池、LLM 池和合并池，使用 HSS 进行排序；结果显示 Qwen3‑30B+GGUF‑Q4 最高（93.38），Mistral‑Small‑24B+GGUF‑Q4 次之；SLM 在合并池中排名前三；单一指标并不能决定最佳，量化方法没有统一的快慢优先级，性能取决于模型架构和后端。

**⚠️ 局限性**

限制包括：归一化对异常值高度敏感；安全评估仅用五个提示且仅基于词汇拒绝，缺乏完整安全性；能耗测量忽略 PUE、碳强度和主机内存消耗；不同平台与后端导致比较不完全一致；实验规模有限，需更多样本、置信区间和更完整的安全评测。

---

## 321. EGT-KG: Evidence-Grounded Typed KG Retrieval for Practical Scientific QA with Small Language Models

**arXiv ID:** 2609.00479 | [PDF](https://arxiv.org/pdf/2609.00479v1)

**作者:** Muran Yu `[一作]` (Stanford), Michael D. Lepech `[通讯]` (Stanford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出Evidence‑Grounded Typed Knowledge Graph（EGT‑KG）检索框架，提升局部小语言模型在科研问答中的检索与回答质量。

**💡 创新点**

通过重构知识图（引入证据节点与来源节点）实现检索驱动而非替代原始证据，并采用两步检索+重排序、关系类型化，并对比自动与专家设计的关系模式。

**🔧 技术方法**

结合AutoSchemaKG三元组抽取、关系分类、知识图重构、图驱动查询扩展与证据窗口重排序，以及基于LLM的检索增强生成（RAG）与S3CRF多维评估。

**📊 数据集**

基于30篇文献的生物聚合土壤复合材料（BSC）问答基准，并在QASPER、HotpotQA等公开数据集做泛化验证。

**📈 对比分析**

与传统RAG在同一基准下比较，使用六维S3CRF评分；在最优模型Llama3.1:8b下，EGT‑KG相较于vanilla RAG提升约+5%–+14% Final Score；对不同模型发现更细粒度关系模式在部分模型上提升更明显。

**⚠️ 局限性**

主要受证据定位失败率高、单跳扩展限制多跳推理、仅验证BSC领域、LLM‑judge主观性及未与前沿大模型直接比较等限制。

---

## 322. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents

**arXiv ID:** 2609.00065 | [PDF](https://arxiv.org/pdf/2609.00065v1)

**作者:** Timothy Kassis `[一作]`, Aubrey M. Brueckner `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并公开了 Scientific Agent Skills 库，用于记录和共享科学领域的程序化流程，供语言模型代理在分析实验时调用；

**💡 创新点**

创新点在于引入技能（skill）文件格式和分层披露机制，并通过对描述可区分性和参考文件 token 成本的定量评估，来衡量库的可用性和上下文成本；

**🔧 技术方法**

主要技术包括 YAML+Markdown 结构化文件、自动化 CI 检查、Python 脚本、tokenizer、余弦相似度计算和 GitHub 仓库统计等；

**📊 数据集**

数据来源为该库自身的技能目录及其 Markdown 文本，不依赖外部数据集；

**📈 对比分析**

通过对特定快照标签的技能描述进行余弦相似度、token 成本与上下文窗口匹配分析，发现大多数技能描述足够区分，文档和参考文件在典型上下文窗口内可加载，但未进行任务级性能比较；

**⚠️ 局限性**

局限性包括缺乏任务级评估、对技能执行效果不做验证、描述冲突仍需人工检查、对 tokenizer 的依赖、库的可达性和稳定性未完整测试以及没有展示实际性能改进。

---

## 323. Breaking Cycles for Scalable Fair Ordering in Blockchain Systems

**arXiv ID:** 2609.00837 | [PDF](https://arxiv.org/pdf/2609.00837v1)

**作者:** Jinchun He `[一作]` (Beihang University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了一种可重放的公平交易排序引擎 FlashOrder，通过局部聚类与分层序列化，避免全局强连通分量（SCC）折叠，显著提升了 BFT 区块链系统的吞吐量与延迟。

**💡 创新点**

创新点在于：① 把对称偏好转化为一维“Canonical Position”，仅用排序而非全图构造；② 通过几何阈值聚类形成分区超图，把大规模循环不确定性限制在小集群内；③ 采用层次化的集群间拓扑排序与集群内净偏好打分，彻底取代了传统的全局 SCC 溶解。

**🔧 技术方法**

技术手段包括：对接 libhotstuff BFT 共识；利用接收顺序证据构造偏好矩阵；阈值化的 CP 计算与排序；基于差值阈值的自适应聚类；构造集群间加权有向图并在必要时合并 SCC；在最终排序中使用字典序拓扑排序与净偏好分数。

**📊 数据集**

实验数据集主要使用 SmallBank（极端 Zipfian skew 与高交互概率）在 CloudLab 硬件上，同时在模拟器中用人工诱导的 Condorcet 攻击和 50 ms WAN 环境进行压力测试。

**📈 对比分析**

与 HotStuff、Themis 与 Rashnu 进行对比：在 9~101 节点下，FlashOrder 的吞吐量相较 Themis 提升 10.5×，相较 Rashnu 提升 4.8×；在 50 ms WAN 下，FlashOrder 在 Condorcet 攻击中平均提升 12.0×/9.7×，同时最大排名位移从 301/287 减少至 34，RPS 从 0.813/0.790 提升至 0.873；延迟比 Themis 与 Rashnu 低 2~4 倍。

**⚠️ 局限性**

局限性在于：仅在对称偏好满足阈值的 “robust pair” 之间保证批次层次公平，对极端 Condorcet 循环内部的公平性仍需经验评估；聚类阈值与 λ 需要手工调参，且在非常大规模或高度动态网络下聚类的稳定性与精度尚未充分验证。

---

## 324. PCoMoE: Shifting MoE Inference from Monolithic Expert Selection to Fine-Grained Path Composition

**arXiv ID:** 2609.01024 | [PDF](https://arxiv.org/pdf/2609.01024v1)

**作者:** Ziyan Gan `[一作]` (Shanghai Jiao Tong University), Haibing Guan `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Mixture-of-Experts模型进行路径级别的子变换组合推理

**💡 创新点**

将专家拆分为扩展侧和投影侧，形成n^2路径空间并使用兼容性门控与源聚合执行

**🔧 技术方法**

路径组合、兼容性门控、源聚合重用、LoRA微调、在线路压缩

**📊 数据集**

使用Qwen1.5-MoE、Mixtral-8x7B、DeepSeek-V2-Lite，Fine-tuning在Alpaca+SQuAD 25K，评估在BoolQ、ARC-E、ARC-C、HellaSwag、WinoGrande

**📈 对比分析**

与vanilla MoE及其他剪枝方法对比，PCoMoE在速度上提升1.3×，在准确率上提升约10%（宏观平均从约68%到约74%）

**⚠️ 局限性**

仅适用于SwiGLU结构；仅优化自回归推理；离线校准不在推理时实现，需要进一步集成分布式/预填充优化

---

## 325. Exponential Gaps Between Intuitionistic Linear Extended Frege Systems

**arXiv ID:** 2609.00422 | [PDF](https://arxiv.org/pdf/2609.00422v1)

**作者:** Amirhossein Akbar Tabatabai `[一作]` `[通讯]` (University of Groningen), Amirhossein Akbar Tabatabai (University of Groningen)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文证明了在多种子结构子命题线性逻辑（如 𝔽𝐿ₑ、𝔦𝔩𝐿 等）中，扩展 Frege 系统与其更弱或更强的变体之间存在指数级的证明大小差距。

**💡 创新点**

创新点在于引入了可行的析取性质（feasible disjunction property）的变体，并利用它将已知的难度公式（Θ⁎_n）从 𝔽𝐿ₑ 迁移到更强的系统中，从而获得新的指数分离结果；该方法首次完成了在非经典逻辑（子结构、线性）框架下的指数分离。

**🔧 技术方法**

主要技术手段包括：1) 对 G‑证明（结构性 sequent 计算机）进行翻译并构造“尖括号原子”；2) 证明可行析取性质并将其推广到 Frege 系统；3) 采用 Hrubeš、Jalali 等人提出的硬公式 Θ⁎_n，证明其在 𝔽𝐿ₑ‑Frege 中需要指数行数；4) 通过可行析取性质，将该指数下界上推到更弱的系统。

**📊 数据集**

论文未使用实验数据集，而是基于形式化证明与理论分析构造了硬公式 Θ⁎_n。

**📈 对比分析**

比较方法：将某一弱系统（如 𝔽𝐿ₑX）与其更强变体（如 𝔽𝐿ₑY，Y ⊈ X）进行对比，证明后者可以在多项式大小内证明同一公式，而前者需要指数行数。性能表现为：在同一公式上，强系统的证明大小是多项式级的，而弱系统必须使用指数级行数。

**⚠️ 局限性**

局限性：分离结果仅适用于结构性（由 𝔽𝐿ₑ 扩展的）逻辑，且依赖于可行析取性质的可证性；无法直接推广到所有子结构或线性逻辑；此外，结果仍停留在理论证明复杂度层面，未给出具体构造实例或实验验证。

---

## 326. SlideBank: A Persistent Hierarchical Evidence Bank for Consistent Whole-Slide Reasoning

**arXiv ID:** 2609.00342 | [PDF](https://arxiv.org/pdf/2609.00342v1)

**作者:** Beidi Zhao `[一作]` (University of British Columbia), Xiaoxiao Li `[通讯]` (University of British Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SlideBank 框架，能够在无需任务特定训练的情况下，将全幅病理切片（WSI）转化为可持久化、概念索引化且空间定向的证据库，以支持多种下游诊断问题；

**💡 创新点**

核心创新在于：①将WSI探索与问答分离，构建多尺度层级证据银行；②通过病理信号（signal）对视觉观察进行标准化、归一化，并与概念（concept）对齐，实现跨问题的语义检索；③实现了同一证据库可在多轮提问中复用，提升稳定性与效率；

**🔧 技术方法**

技术包括：基于视觉语言模型（VLM）生成多尺度图像描述；基于Agent进行粗到细的全局巡检与区域采样；构建病理信号层并映射至概念-信号本体；在推理阶段采用多分支（全局、anchor、patch）信号检索与置信度加权投票；

**📊 数据集**

在公开的两大WSI问答基准上评测：WSI‑VQA（85张WSI）与 SlideBench‑BCNB（1,058份乳腺癌切片，涵盖肿瘤类型、分级和分子亚型）；同时构造多轮问答场景以验证重访一致性；

**📈 对比分析**

与零样本VLM、WSI‑专用模型（WSI‑LLaVA、TITAN、SlideChat）以及代理式推理模型（Med‑Agents、PathAgent）等多类方法对比。SlideBank 在 WSI‑VQA 上取得 52.77% 的准确率（相较基线提升约 8.5pp），在 SlideBench‑BCNB 上获得 50.36% 的平均准确率；在多轮提问中保持 99% 以上的一致性，且查询平均耗时仅为 5.9 s（相较单次 94.3 s）。

**⚠️ 局限性**

局限性：①证据库构建需一次性扫描全图，尽管后续可复用，但初始成本仍高；②信号与概念映射依赖预定义本体，可能对新颖或罕见病理特征捕捉不足；③当前实现未对跨病例共享知识，仅聚焦单张切片；④Agent 的采样策略与阈值需人工设定，易受扫描分辨率或组织差异影响。

---

## 327. Safin-1: Safety from Within through Memory-Native State Evolution

**arXiv ID:** 2609.00092 | [PDF](https://arxiv.org/pdf/2609.00092v1)

**作者:** Ming Zhang `[一作]` (Shanghai AI Laboratory), Chaochao Lu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一类利用可路由的循环状态记忆架构MARCH的基础模型，实现历史状态的检索与持久化能力，提出Safety State用于安全性从模型内部实现。

**💡 创新点**

将模型内部循环状态拆分为可检索的状态锚点并通过内容路由实现历史回溯，同时将安全能力嵌入为持久化状态，使安全性成为模型自身的可控、可学习模块。

**🔧 技术方法**

基于记忆锚点路由（MARCH）的循环网络（如Gated DeltaNet、Kimi Delta Attention等），内容路由与状态检索、Top‑k稀疏路由、持续预训练与监督微调以及冻结主干训练安全状态的技术。

**📊 数据集**

预训练使用约50B token的Long‑Data‑Collections；下游评测使用LAMBADA、PIQA、HellaSwag、WinoGrande、ARC、OpenBookQA、CommonsenseQA、LongBench、NIAH、SQuAD、TriviaQA、SWDE、FDA、Natural Questions、DROP；安全评测使用STAR‑1、STAR‑benign、WildJailbreak、FORTRESS、StrongREJECT、Jailbreak‑R1、JailbreakBench、XSTest。

**📈 对比分析**

与同参数规模的全注意力模型和传统循环基线在0.8B实验中对比，通用语言建模、长上下文理解、检索等任务平均提升1–2分；在4B和35B规模的持续预训练+SFT中，推理、数学、代码、指令跟随等多项基准提升0.5–3分；安全方面，Persistent Safety State 在5个反弹攻击中平均ASR下降42–52%且过度拒绝率低于LoRA。

**⚠️ 局限性**

状态库随上下文增长导致存储与路由成本；仅在两种Qwen3.5基底模型上验证；安全训练数据规模有限，未覆盖多语言或自适应攻击；大规模模型在某些基准的能力保持不一，需进一步研究多持久状态的协同与安全性完整性。

---

## 328. DRLM: Deep Reinforcement Learning-Based LLM Query Orchestration in Edge Environments

**arXiv ID:** 2609.00442 | [PDF](https://arxiv.org/pdf/2609.00442v1)

**作者:** Reza Farahani `[一作]` (TU Wien), Schahram Dustdar `[通讯]` (TU Wien)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并实现了一套基于深度强化学习的边缘集群LLM查询编排框架DRLM，能够根据查询语义、模型配置与设备状态实时做出最优模型-设备-量化选择。

**💡 创新点**

创新点在于：① 将查询语义分为6类并用类条件的质量估计取代单一准确率排序；② 通过特征驱动的延迟预测器快速估计不同配置的推理耗时；③ 将上述预测与实时系统状态结合，使用因子化Proximal Policy Optimization (PPO) 进行端到端决策；④ 构建了包含223 835条测量的多维度基准数据集，支持大规模离线训练与在线评估。

**🔧 技术方法**

核心技术包括：句子编码器 + 随机森林分类器、LightGBM 延迟回归器、PPO 深度强化学习、因子化策略网络、实时资源监控与动态状态编码。

**📊 数据集**

使用的数据集为论文自建的 223,835 条测量基准集，涵盖 1,258 个查询（按 6 类划分）、8 个模型家族、32 个实例、5 个量化级别，以及 1258 条跨 4 大基准数据集的真实查询。

**📈 对比分析**

与三种基线（随机分配、最大准确率、最快延迟）以及两种前沿方法（RouteLLM、OptLLM）在同一 64 节点边缘集群上对比。DRLM 在推理延迟上最高可减少 51%、排队延迟最高可减少 67%，在准确率上最多只损失 8%，且在高负载场景下能提升 61.4% 的整体吞吐量，表现出显著的性能优势。

**⚠️ 局限性**

局限性：① 需要先行收集大规模的预测数据并训练多模型；② 训练过程依赖预测器的准确性，若预测误差大可能导致策略失效；③ 当前仅考虑延迟与准确率的二元平衡，未覆盖能耗、成本等多目标约束；④ 对于极端异构或大规模分布式部署时的可扩展性和安全性尚未充分验证。

---

## 329. Dyn-3D: Unveiling and Resolving Ego-Motion Ambiguity in Vision-Language Models

**arXiv ID:** 2609.01059 | [PDF](https://arxiv.org/pdf/2609.01059v1)

**作者:** Jiayu Ding `[一作]`, Wenbo Xing `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并发布 Dyn-3D benchmark，利用 3D 高质量重建与可控渲染生成视觉与运动无关的对照视频；提出 TempoVista 框架，包含基于 SE(3) 的关键帧自适应采样与 Kinematic-GSPO 强化学习，进一步训练 VLM 以实现显式的运动感知与三维空间推理。

**💡 创新点**

① 通过 counterfactual 渲染把视觉变化与真实运动解耦，构建专门评测运动感知的基准；② 在 VLM 训练中引入物理一致性奖励（Kinematic-GSPO），将运动约束嵌入推理过程；③ 开发基于 SE(3) 的帧采样策略，提升对非线性轨迹的感知精度。

**🔧 技术方法**

3D Gaussian Splatting（可控重建与渲染）、SE(3) 距离度量与 FPS 关键帧选择、Kinematic-GSPO 强化学习（包含答案级、格式级与运动级奖励）以及多任务 SFT + RL 训练管线。

**📊 数据集**

Dyn-3D benchmark（167 场景、835 视图、16,063 四选题）和 Dyn-3D-Instruct 训练集（9,600 SFT 样本、24,000 RL 样本），对照现有 ScanQA、SpatialVLM、MVBench、EgoSchema、OpenEQA、VSI-Bench 等基准。

**📈 对比分析**

在 Dyn-3D 评测中，TempoVista 在 kinematic、spatial、trajectory 与 overall 上相较基准模型提升 6–12%（例如 InternVL‑3.5‑8B 从 50.6% 提升到 57.1%，Qwen3‑VL‑8B‑Instruct 从 50.3% 提升到 61.2%）。在 VSI‑Bench 迁移任务中亦获得 2–7% 的性能提升，证明运动感知优化能推广到更广泛的空间推理任务。

**⚠️ 局限性**

① 依赖 3D Gaussian Splatting 进行重建，受限于现有室内场景数据；② 对于极端运动或动态光照、非线性轨迹的鲁棒性仍有提升空间；③ 需要高质量的运动标注，生成和评测成本较高；④ 在某些空间推理子任务（如相对方向）提升有限，说明运动感知与其他认知能力仍需更深层次耦合。

---

## 330. Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems

**arXiv ID:** 2609.00237 | [PDF](https://arxiv.org/pdf/2609.00237v1)

**作者:** Rakibul Hasan Rajib `[一作]` (University of Central Florida), Qian Lou `[通讯]` (University of Central Florida)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了基于门控记忆的多代理系统，动态调度角色、模型和停止条件，显著提升推理准确率并降低推理成本。

**💡 创新点**

创新点在于引入写门和检索门对执行历史进行学习式压缩，形成“门控记忆”，让后续的路由和终止决策基于精选的、低冗余上下文，从而兼顾准确性与效率。

**🔧 技术方法**

使用了变分自编码器（VAE）编码角色与模型描述，基于多项式策略的角色分配与模型路由，MMLR式写门与检索门，GRU+MLP的自适应停止控制，以及基于策略梯度的联合训练。

**📊 数据集**

采用了五个标准评测集：GSM‑Hard、MATH、HumanEval、MBPP 和 MMLU‑Pro，检验数学推理、代码生成与知识问答能力。

**📈 对比分析**

与单模型、固定拓扑多代理、MASRouter、Puppeteer 等基线对比，平均准确率提升 2.44 分，HumanEval 推理成本降低 31.9%，同时在大多数单项指标上位列第一。

**⚠️ 局限性**

局限性包括仅在可验证答案的闭合域任务上评测，无法直接评估开放式生成；计算评估依赖参数计数和 FLOPs，未覆盖实际延迟与内存占用；模型仅限于已公开的 Llama‑3、Qwen‑2.5 与 Mistral 系列，需进一步验证对更大或更新模型的适配性。

---

## 331. Advanced Pixel Diffusion Model with Guided Sparse Global Refinement

**arXiv ID:** 2609.00798 | [PDF](https://arxiv.org/pdf/2609.00798v1)

**作者:** Weiyi You `[一作]` (University of Electronic Science and Technology of China), Shuhang Gu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PixSGR，基于像素空间的稀疏全局细化扩散模型，结合低通道瓶颈、粗细尺度注意力预稀疏化与内部引导

**💡 创新点**

1) 低通道瓶颈与辅助监督提升对低维图像流形的建模；2) 利用粗尺度注意力映射预稀疏化细尺度全局注意力，突破局部补丁限制；3) 在细化阶段加入内部引导与卷积上采样，提升连续性与质量

**🔧 技术方法**

像素空间扩散、Transformer + 大补丁令牌化、低通道瓶颈、粗细尺度注意力预稀疏化、内部引导、卷积上采样、伪Huber流动匹配

**📊 数据集**

ImageNet 1K（256×256 及 512×512）

**📈 对比分析**

与最新像素空间与潜在空间扩散模型对比，PixSGR-H 在 256×256 上 FID 1.51、IS 311，优于 PixelREPA、AsymFlow、FrequencyBooster、PixelU 等；在 512×512 仍保持 1.60 FID；模型参数 875M，训练周期 320 epoch，显著低于同类潜在模型

**⚠️ 局限性**

仍受限于单机 GPU 训练规模、需高分辨率推理时对稀疏映射设计的复杂度、在极大分辨率下稀疏策略可能无法覆盖所有长程依赖

---

## 332. PhantomCall: Evading ML Malware Detectors via Function Call Graph Perturbation

**arXiv ID:** 2609.00705 | [PDF](https://arxiv.org/pdf/2609.00705v1)

**作者:** Md Ajwad Akil `[一作]` (Purdue University), Elisa Bertino `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于函数调用图（FCG）扰动的黑盒攻击，利用可执行的假函数注入来生成恶意软件的可执行对抗样本，保持原始功能不变；

**💡 创新点**

创新点在于首次对已编译的Windows PE恶意软件实现完全可执行、可达的假函数注入，兼顾CFG与FCG结构扰动，并通过可调的NOP序列与注入量进行搜索；

**🔧 技术方法**

采用函数调用图重定向与Trampoline技术、可调节的NOP序列、适应性模拟退火或贪心搜索算法来寻找最优扰动；

**📊 数据集**

使用2025年收集的Windows PE恶意软件数据集（MalwareBazaar）进行测试，并为SAFE+GNN训练单独的2024年恶意软件与良品数据集；

**📈 对比分析**

相较于基线MalGuise，PhantomCall在MalConv、MalGraph和SAFE+GNN上攻击成功率分别提升至85–100%，提升幅度可达14.78pp，且生成速度比基线快1.5–3倍；

**⚠️ 局限性**

局限性包括：对抗微调后仍可削弱攻击效果；结构化启发式防御只能部分抵御；注入的假函数过于简化，可能被更复杂的检测方法识别；以及仅针对Windows PE、x86 32位环境，难以直接推广到其他架构。

---

## 333. DeSyR: A Decoupled Symbolic Recovery Framework with PINN-Guided Structure Search and Physics-Informed Coefficient Refinement

**arXiv ID:** 2609.00530 | [PDF](https://arxiv.org/pdf/2609.00530v1)

**作者:** Pancheng Niu `[一作]` (Chengdu University of Information Technology), Yanchao Shi `[通讯]` (Southwest Petroleum University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了 DeSyR 框架，利用 PINN 生成的教师网络指导符号结构搜索，并在固定拓扑的前提下仅使用物理方程进行系数优化，从而实现对偏微分方程显式解的精确恢复。

**💡 创新点**

核心创新在于：①将拓扑搜索与系数估计完全分离，避免教师误差在最终表达式中残留；②通过理论分析证明在满足可表示性、离散可决定性等条件时，物理‑仅重拟可以恢复原解的唯一系数；③引入分门选择与验证机制，提升恢复的可靠性与可解释性。

**🔧 技术方法**

技术栈包括：物理信息神经网络（PINN）作为教师，PySR 进行符号回归，基于残差的非线性最小二乘系数重拟，候选池合并与筛选算法，理论上结合最小二乘与高阶导数分析。

**📊 数据集**

数据集涵盖 15 个 PDE 典型问题（Poisson、Euler–Bernoulli、波动、Telegraph、Fokker–Planck、Klein–Gordon、Burgers、Helmholtz、Sine–Poisson、Kovasznay 等）在 18 个配置下（高阶、多维、非线性、耦合等）构成的实验集合。

**📈 对比分析**

对比方法包括：教师‑仅拟合、混合数据‑物理拟合、物理‑仅拟合；实验表明在 90 次最终表达式中 99.23% 收敛，物理‑仅重拟后相对 L2 误差平均至 2.31×10⁻¹⁴，误差相对教师下降 8–14 订单；同时记录计算时间与候选覆盖率。

**⚠️ 局限性**

局限性：①需先验可表示的符号库，若库缺失关键算子会导致可表示性失效；②仅保证在已探索到可行拓扑的前提下进行系数恢复，拓扑搜索仍可能失败；③对非线性参数化的收敛仅在局部可辨识性下保证；④实验聚焦于已知算子和配方的前向问题，对未知算子学习尚无覆盖。

---

## 334. Same Request, Different Boundary: Evaluating Cybersecurity Assistance across Conversational Contexts

**arXiv ID:** 2609.00578 | [PDF](https://arxiv.org/pdf/2609.00578v1)

**作者:** Rui Yang `[一作]` (Johns Hopkins University), Yinzhi Cao `[通讯]` (Johns Hopkins University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出3R-Bench基准，设计三种对话场景（直接请求、预设助手回应、任务分解），并在八个大型语言模型上进行评估。

**💡 创新点**

创新点在于将对话历史纳入安全评估，揭示任务接受与拒绝的上下文敏感性，并通过对比实验量化历史对模型合规行为的影响。

**🔧 技术方法**

使用手工构建的真实网络安全请求数据集、对话历史模拟、语义对比分析以及模型输出编码来衡量拒绝/合规结果。

**📊 数据集**

数据集为150条真实网络安全请求，均衡包含50条安全、50条模棱两可、50条高风险请求，全部来自公开安全问答。

**📈 对比分析**

通过比较同一请求在三种场景下的合规/拒绝率，发现接受历史将合规率从62%提升至85%，而任务分解则将合规率从67.8%降至22.6%。总体而言，模型在不同情境下表现不一，提示需要更细粒度的对话安全评估。

**⚠️ 局限性**

局限性包括样本量有限、仅英文、对话历史设计受限、未评估模型内部安全机制、未验证生成的技术是否可在真实环境中实现，导致结果难以推广至大规模或多语言环境。

---

## 335. CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language

**arXiv ID:** 2609.00058 | [PDF](https://arxiv.org/pdf/2609.00058v1)

**作者:** Qi Fan `[一作]` (Shanghai Jiao Tong University), Yehan Ma `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 705 | [OpenAlex ID](https://openalex.org/A5024303770)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个基于代理的 CUDA 内核生成与优化框架 CUDA-Harness，解决 Text2CUDA 过程中的语义理解、测试验证和性能优化三大难题。

**💡 创新点**

创新点包括：① Intermediate-Structured Generation 通过实现清单和自包含 CUDA scaffold 分离语义理解与实现细节；② Synthesis-Based Verification 采用独立的测试数据合成和渐进式多阶段验证，有效降低奖励黑客风险；③ Feedback-Adaptive Evolution 结合正确性优先修复与可验证奖励的强化学习，形成迭代优化流程。

**🔧 技术方法**

核心技术包括：LLM 代理（Seed2.0 Lite、DeepSeek‑V3.2、GLM‑5.1）与 ReAct 交互、结构化生成模板、NumPy/PyTorch 生成测试数据、编译/功能验证工具、Nsight Systems/Compute 性能分析、RL 与奖励驱动优化。

**📊 数据集**

使用的主要数据集为 CUDABench（含三难度级别）和 BabelTower（C‑to‑CUDA 转译），并在多种 GPU（A40、GTX 1660 SUPER、Jetson AGX Orin）上进行评测。

**📈 对比分析**

与官方 CUDABench 基线、OpenCode、Codex、KernelSkill、CudaForge 等方法对比，CUDA-Harness 在编译成功率、功能正确率和运行时性能（RScore）均显著提升；在跨 LLM、跨硬件以及 C‑to‑CUDA 场景下保持领先，最高 RScore 提升约 40%。

**⚠️ 局限性**

局限性包括：仍受到底层 LLM 能力限制；Synthesis‑Based Verification 的误报率约 8%，偶有假阳性；迭代优化次数有限，对极其复杂或非数值型内核的处理仍需改进；在极限硬件上性能提升可能受限于平台差异。

---

## 336. Exploring Collaboration between a language and a non-language agent

**arXiv ID:** 2609.00474 | [PDF](https://arxiv.org/pdf/2609.00474v1)

**作者:** Harini S `[一作]` (Adobe), Balaji Krishnamurthy `[通讯]` (Adobe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的 LLM 与非语言子代理协作方法，即“潜在状态内部化”，并实现了 LLAMIA 系统。

**💡 创新点**

创新点在于：①用投影层把子代理的连续隐藏表示直接注入 LLM 的 token 流，消除自然语言压缩的损失；②系统性量化“Verbalization Debt”，证明自然语言压缩是不可逆的性能瓶颈；③构建 LLAMIA‑Bench 六项棋类协同任务，验证内部化在多步推理、非可文本化信号等场景下的优势。

**🔧 技术方法**

技术细节包括：Qwen3 大模型作为主干；Lc0-BT4 或 KataGo 作为子代理；三层 MLP 投影器 LatentBridge 把 1024 维隐藏向量压缩为 32 个与 LLM 隐藏维相同的连续 token；两阶段训练：①对齐投影器；②使用 DAPO 强化学习联合微调；支持多种 token 交织（语言、动作、潜在状态）。

**📊 数据集**

使用的数据集包括：Lichess 公开对局库（行为克隆、难度估计）；YouTube（Agadmator 2K 语音注释）；棋题兴趣排行榜（社区评分）；以及通过 KataGo 产生的围棋对局，用于跨域验证。

**📈 对比分析**

对比方法：与 GPT‑5+Lc0、Qwen3‑14B+Lc0（未训练）、LLAMIA‑Verb（同框架仅用文字）以及任务专用 finetune 进行比较。LLAMIA 在六项基准任务上均达到或超过所有基线，特别在“Puzzle Interest”和“Game Commentary”任务上表现显著优于任何文字化方案；在 14B 规模下与 GPT‑5 同级别，且在 4B/8B 规模下仍保持领先；强化学习阶段显著提升多步推理和策略生成。

**⚠️ 局限性**

局限性包括：①需要访问子代理的权重，无法直接应用于闭源引擎；②目前实验仅在棋类（以及少量围棋）验证，其他领域的通用性仍需进一步研究；③投影器的设计和参数（k=32）需要经验调优，可能在不同任务中表现不一致；④大模型与投影器的计算开销略高，需要在实时系统中权衡。

---

## 337. CompanionSim: Synthetic Data for Evaluating Anthropomorphism in Human-AI Relationships

**arXiv ID:** 2609.00250 | [PDF](https://arxiv.org/pdf/2609.00250v1)

**作者:** Jacy Reese Anthis `[一作]` (University of Chicago), Renee Shelby `[通讯]` (Google Research)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

生成了一个包含 2240 条多轮对话的合成数据集 CompanionSim，并通过两项实验评估人类对不同陪伴行为的感知与信任；

**💡 创新点**

首次系统化构造 16 种陪伴行为与 7 个用例的模拟框架，揭示陪伴行为在第三方评估中往往降低喜好、亲和与信任，强调细分测量与跨群体差异的重要性；

**🔧 技术方法**

利用 Gemini 2.5 Flash 生成多轮对话，采用 Godspeed 问卷量表与自定义 Likert 量表进行评估，采用混合效应线性模型进行统计；

**📊 数据集**

使用自生成的 2240 条合成对话、70 条真实对话（ShareGPT、EmpatheticDialogues、Replika）以及 628+3646 名受访者的问卷数据；

**📈 对比分析**

通过与无陪伴行为的对照组比较，使用 GLMM 评估效应，结果显示陪伴行为导致 Likability、Humanlikeness、Affective Trust、Cognitive Trust 均出现显著负向效应（如 -0.08、-0.13），且在女性、老年人及低 AI 使用频率群体中效应更大；

**⚠️ 局限性**

仅基于单一 LLM 与单一行为/用例分类，无法确定陪伴行为的自然出现频率；生成对话可能不完全映射真实对话；研究仅覆盖英文语境，缺乏多语言与长期交互验证。

---

## 338. Ordinal Gates, Cardinal Bets: Matching LLM Confidence to the Financial Decision Operator

**arXiv ID:** 2609.00187 | [PDF](https://arxiv.org/pdf/2609.00187v1)

**作者:** Rayansh Singh `[一作]` (Michigan State University), Sara Rezaeimanesh `[通讯]` (Michigan State University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型（LLM）生成的置信度分数在金融决策中的使用，并证明置信度分数的决策价值取决于下游的决策器和风险控制器，单独迁移置信度映射或规模不具可移植性；通过将置信度映射与专属风险规模匹配，显著提升冻结规模控制下的期望回报。

**💡 创新点**

创新点在于：①首次明确区分“序数型”决策器（仅关注排名）与“基数型”决策器（关注幅度），并证明单调校准无法改变基数型决策的效果；②提出并验证置信度映射与风险规模之间的耦合效应；③在实盘交易上对九个开源LLM进行系统性评估，展示匹配方案在冻结与自适应风险控制下的收益提升与风险特征。

**🔧 技术方法**

使用的技术包括：多种置信度提取通道（原始得分、正确性映射、语音化置信度、logit差值等）；单调（Isotonic）校准、Beta和Platt回归；冻结规模控制（设定年化5%波动率）与自适应波动率控制；基于闭合式风险管理的组合权重构造；统计检验（Holm、Romano–Wolf、Stouffer等）和块级自助法评估效果。

**📊 数据集**

数据集为FactSet专业新闻稿标题，覆盖纳斯达克-100指数成分股，2021年为校准期，2022–2023年为测试期；此外，还使用2024–2025年额外的头条数据进行年度前向验证。每个 ticker‑day 通过聚合多条新闻标题产生单一方向和置信度。

**📈 对比分析**

对比方法：将“原始置信度+自定义规模”系统与“正确性映射+自定义规模”匹配系统进行比较；同时设立“原始置信度+迁移规模”和“正确性映射+原始规模”作为可移植性诊断。结果显示：在冻结规模控制下，匹配系统使模型集合CER提升约9.2个百分点/年，风险（波动率）显著下降；在自适应波动率控制下提升约1.6个百分点/年；单模型层面提升多数情况下不显著。

**⚠️ 局限性**

局限性包括：①基础预测准确率仅略高于50%，并非市场优势；②结果高度依赖于校准窗口与置信度映射的具体实现，年度前向验证表现弱化；③模型训练时间戳与测试期间的重叠可能引入数据污染；④仅检验了部分LLM与置信度通道，未涵盖所有可能的金融任务；⑤对不同市场或更长持有周期的泛化能力尚未充分验证。

---

## 339. Ctrl-F-Resist. Practices, Challenges, and Technical Needs of Civil Society Organizations Monitoring the Far-Right Online

**arXiv ID:** 2609.00808 | [PDF](https://arxiv.org/pdf/2609.00808v1)

**作者:** Elisabeth Steffen `[一作]` (Hochschule für Technik und Wirtschaft, University of Applied Science), Helena Mihaljević `[通讯]` (Hochschule für Technik und Wirtschaft, University of Applied Science)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过对 15 名德国民间社会组织（CSO）从业者的访谈，系统梳理了 CSO 在网络监测中的工作流程、面临的技术、法律与伦理挑战，并基于研究结果与参与者共创了一个面向 Telegram 的开源监测原型。

**💡 创新点**

创新点包括：①提出“手工劳动陷阱”概念，解释 CSO 监测工作为何易陷入低效、碎片化；②将 CSO 需求与 CSCW 设计原则融合，形成可持续、低资源友好的监测工作流模型；③在原型中实现多模态检索、自动转录、语义搜索及人机协作分类，首次在民间监测场景中实现全流程闭环。

**🔧 技术方法**

技术实现：Whisper（ASR）用于音频转写；Elasticsearch+多语言 Sentence Transformer/CLIP 进行文本和图像的语义检索；BERT（微调）实现阴谋论检测；Docker 容器化实现本地部署；交互式 Dashboard 提供检索、标注、报告等功能。

**📊 数据集**

数据来源：访谈收集的实际 Telegram 渠道信息与屏幕截图；并未使用公开标注数据集。原型在工作坊中使用了参与者提供的 Telegram 频道数据进行演示。

**📈 对比分析**

研究未进行算法对比或性能评估，侧重于设计与可用性验证；因此未给出准确的准确率、召回率等指标。

**⚠️ 局限性**

局限性：样本仅限于 12 家德国 CSO，规模小且主要关注 Telegram；受访者技术背景有限，可能影响需求描述；研究未覆盖其他平台或更大样本；结果的普适性需在更广泛的国际与多平台场景中进一步验证。

---

## 340. Self-Reports Are Not Verification: Environment-Grounded Auditing of LLM Operators in Evolutionary Search

**arXiv ID:** 2609.00652 | [PDF](https://arxiv.org/pdf/2609.00652v1)

**作者:** Enrong Pan `[一作]` (Queen's University), Ting Hu `[通讯]` (Queen's University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Contexto 词语游戏环境中，使用 LLM 驱动的进化搜索，对每个中间提议生成自我报告（置信度和理由），并通过环境给出的精确排名进行评估。

**💡 创新点**

设计了“环境基准审计”，在每一步都能获得精确反馈，以检验自我报告的准确性、理由对后续行为的因果影响以及选择机制对报告质量的影响。

**🔧 技术方法**

结合进化算法、LLM 变异算子、置信度校准、Spearman 相关、AUROC 等评估技术，使用 Qwen-3、Gemma-4、Ministral-3 等模型进行实验。

**📊 数据集**

利用 Contexto 的公开词汇表（数千文本上下文的词义相似度排序）生成 10 个目标词作为实验对象。

**📈 对比分析**

在 200 次运行（5 配置 × 3 模型）中收集 12,249 条报告，使用 ECE、Spearman ρ、AUROC 等指标评估置信度；对 754 事件进行理由干预实验；对 1,214 次选择事件和 1,018 对父子进行选择差异与传递性测试。结果显示置信度过度自信，理由无显著因果影响，选择对报告质量无提升。

**⚠️ 局限性**

仅在单一精确反馈的词语搜索环境中验证，缺乏对更大模型、不同架构或噪声/部分反馈环境的评估；理由干预仅限于继承渠道，未检验理由对人类读者或同一行动的指导作用。

---

## 341. WHALE: A Simple Recipe for Joint Harness-Weight Optimization

**arXiv ID:** 2609.00196 | [PDF](https://arxiv.org/pdf/2609.00196v1)

**作者:** Haechan Kim `[一作]` (KRAFTON), Kangwook Lee `[通讯]` (KRAFTON)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种权重-工具包交替学习框架（WHALE），在同一训练循环中交替更新语言模型权重和可执行的工具调用框架（harness），实现二者的协同优化。

**💡 创新点**

创新点在于：① 通过将权重更新与harness搜索解耦为交替两相，避免了单向优化导致的瓶颈；② 引入可调节的固定或自适应阶段长度规则，自动决定何时切换；③ 在框架中允许任意权重更新与harness搜索算法组合，展示了方法的通用性。

**🔧 技术方法**

使用的技术包括：在线拒绝采样微调（RSFT）进行权重更新；Meta‑Harness（MH）进行harness搜索；自适应 patience 规则用于阶段切换；对比实验中使用 Qwen3.5‑2B/4B 模型。

**📊 数据集**

使用的数据集包括：SearchQA（HotpotQA、Natural Questions 等），数学推理任务（DAPO‑Math‑17K、AIME 2024/2025），以及国际象棋谜题（Lichess 开源数据库）。

**📈 对比分析**

方法通过与单组件基线（仅权重更新、仅harness搜索）以及 Fast‑Slow Training（只调优提示+权重）进行比较。WHALE 在 SearchQA、Math、Chess Puzzle 三个域分别比最佳单组件提升 7.67%–24.38%，并比 FST 提升 4.15%–13.00%，在总体实验预算内获得最高测试准确率。

**⚠️ 局限性**

局限性：仅在 Qwen3.5‑2B/4B 小模型和三种相对简单的任务域上验证；对大模型、复杂工具链或更丰富的harness表达空间的适用性仍待探索；自适应切换规则依赖训练信号，可能在不同任务或硬件环境下需要重新调参。

---

## 342. Asymmetries in Spontaneous and Instructed Deception

**arXiv ID:** 2609.00180 | [PDF](https://arxiv.org/pdf/2609.00180v1)

**作者:** Josiah Luikham `[一作]` `[通讯]` (Independent Researcher), Josiah Luikham (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了Llama-3.1-70B-Instruct模型中主动（指令式）与自发（无指令）欺骗的关系，比较了两种设置下的检测（探针）与控制（方向驱动）的效果。

**💡 创新点**

创新点在于发现两种欺骗模式在内部表示上共享约0.5余弦相似度的方向，并揭示检测与诱导之间的非对称迁移；首次系统对比探针与方向在不同欺骗设置中的迁移性能。

**🔧 技术方法**

主要技术包括训练逻辑回归探针、计算均值差方向并对齐、利用方向驱动（steering）在生成时施加、使用GPT‑4.1 Mini Judge进行欺骗评分，并进行二分类与类型级（制造/遗漏）分类。

**📊 数据集**

使用了由GPT‑5 Mini生成的指令式与自发式欺骗提示及其模型响应构成的数据集，按制造和遗漏两类标注，并划分训练/验证/测试集。

**📈 对比分析**

通过探针的balanced accuracy、AUROC、cosine相似度和flip率等指标进行比较，结果显示自发训练的探针在指令式设定中接近上限，指令式方向在自发设定中对欺骗的抑制/诱导效果更好，整体性能均达到高水平。

**⚠️ 局限性**

局限性包括仅在Llama-3.1-70B-Instruct上实验；情景与自发欺骗率依赖于GPT‑5 Mini生成质量；自发制造抑制实验样本不足，置信区间宽；判定器与人类评判不完全一致；仅研究遗漏与制造两种欺骗类型，未涵盖扭曲等其它形式。

---

## 343. Frozen Cores Need Task Signal: Fisher-Whitened Cross-Covariance for Low-Resource LLM Adaptation

**arXiv ID:** 2609.00762 | [PDF](https://arxiv.org/pdf/2609.00762v1)

**作者:** Wentao Ye `[一作]` (Zhejiang University), Gang Chen `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在预训练模型中冻结权重基底，只对一个低秩核心进行微调，从而将参数量和优化器状态降到极低水平，并通过一个统一的校准流程选取最优子空间。

**💡 创新点**

① 引入 Fisher‑白化交叉协方差 (FCCA) 作为子空间选择准则，结合任务相关的梯度信息与局部几何；② 通过薄 QR 重参数化保证冻结基底可被核心有效训练；③ 在同一低秩子空间预算下对八种构造器进行公平对比，证明子空间质量决定微调效果。

**🔧 技术方法**

低秩适配器 (LoRA/DoRA/PiSSA) 的固定基底变体、FCCA、RawGrad、FiLoRA‑core、MiLoRA‑core、LoRA‑XS‑core、VeRA‑core、EVA‑core、随机正交基底；Fisher‑白化、SVD 截断、薄 QR、Kronecker‑factored Fisher（K‑FAC）等工具。

**📊 数据集**

Qwen2.5 系列（1.5B/3B/7B）和 Llama‑3.2‑1B；11 个任务（SVAMP、GSM8K、SST‑2、QNLI、CoLA、RTE、MRPC、ARC‑Challenge、OpenBookQA、HellaSwag、WinoGrande），每个任务使用 300–500 条验证样本进行校准。

**📈 对比分析**

对同一 r=16、r² 预算（即每层 16×16 核心）下，FCCA 在 Qwen3B 上获得 83.0 的宏平均得分，比 RawGrad 低 4.8 分、比 MiLoRA‑core 低 2.6 分、比 LoRA‑XS‑core 低 2.3 分；在 Qwen1.5B、7B 以及 Llama‑1B 上也位居前列；与全参数 LoRA/DoRA/ PiSSA 比较，FCCA 仅使用 36.9K 可训练参数，却与 7.4M 的 LoRA/DoRA 接近（差距 0.23–0.32 分），并在训练时间与 GPU 内存上显著节省。

**⚠️ 局限性**

① 只在 300–500 条校准样本下评估，难以反映完整任务性能；② 只针对 11 个固定任务，缺乏对更大规模或生成任务的验证；③ 仅使用三种随机种子、单一软件栈和单一 GPU，结果可能受硬件/实现影响；④ 校准需要反向传播与大规模协方差，显存和计算成本较高；⑤ 只在特定子空间大小（r=16）和模块覆盖（q/k/v/o）上验证，未探讨更广泛的配置空间。

---

## 344. Exploring Nonlinear Body Oscillations for Natural Quadruped Gaits

**arXiv ID:** 2609.00539 | [PDF](https://arxiv.org/pdf/2609.00539v1)

**作者:** Annika Schmidt `[一作]` (Technical University of Munich), Alin Albu-Schäffer `[通讯]` (Technical University of Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并制造了一台高弹性四足机器人，通过数值方法识别其非线性正常模态（NNM），并利用极简状态切换控制器在仿真与硬件中激发这些模态，从而让机器人在不需要复杂控制或模板模型的情况下自然出现多种步态。

**💡 创新点**

创新点在于首次将非线性正常模态作为设计与控制的核心，证明机械共振可以直接引导步态产生，并提供了一套基于保守动力学的分析与调度框架，突破传统依赖高增益控制的四足机器人步态设计方法。

**🔧 技术方法**

核心技术包括：基于微分几何和代数拓扑的非线性模态分析、状态切换（bang‑bang）控制器实现能量注入、黑盒优化（CMA‑ES）搜索步幅参数、仿真平台 Gazebo 与硬件实现的串联执行、以及序列弹性驱动（SEA）实现高弹性关节。

**📊 数据集**

本文没有使用公开数据集，而是通过机器人自身的仿真数据与实机测试数据来验证理论与方法的有效性。

**📈 对比分析**

方法比较：对仿真与硬件中六个 NNMs 的频率、正功率占比（η）及前进速度进行量化对比。结果显示，仿真与硬件在频率与步态模式上高度一致，η 最高可达 80%，但部分模式在硬件中因摩擦与传动滑移导致频率下降或步态失效，整体表现与理论预期相符。

**⚠️ 局限性**

局限性包括：依赖保守动力学模型，实际硬件摩擦、驱动延迟与机械误差会偏离理论；部分步态因传动滑移无法实现；未对能耗、稳定性与复杂环境的鲁棒性做系统评估；以及缺乏与传统模板模型（如 SLIP）在能量效率与动态稳定性方面的定量对比。

---

## 345. Adapting Without Gradients: Affine Statistics Transport and What Its Certificate Can Tell You

**arXiv ID:** 2609.00374 | [PDF](https://arxiv.org/pdf/2609.00374v1)

**作者:** Salim Khazem `[一作]` (Talan Research Center), Ibrahim Mohamed Serouis `[通讯]` (Talan Research Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为CASTER的无梯度、无后向传播、无源特征库的测试时适应方法，通过在冻结特征空间中对源类统计量进行类共享仿射变换，实现对目标批次的自适应分类。

**💡 创新点**

创新点在于：①只存储源类均值和协方差，避免大规模特征存储；②使用类共享仿射映射估计目标批次的统计量；③设计了残差-间隔可运输性证书，安全地决定是否采用变换。

**🔧 技术方法**

采用线性判别分析（LDA）子空间投影、统计量的收缩稳健仿射映射、Mahalanobis距离计算、残差-间隔可运输性证书、无梯度推理。

**📊 数据集**

在四种预训练视觉骨干（ConvNeXt‑T、ViT‑B/16、DeiT‑B、Swin‑T）上，评估七个数据集（Oxford‑IIIT Pets、DTD、Flowers‑102、Food‑101、CIFAR‑10、CIFAR‑100、Tiny‑ImageNet）以及CIFAR‑10‑C、CIFAR‑100‑C、ImageNet‑C 等腐败/分布偏移基准。

**📈 对比分析**

与冻结特征下的k‑NN、Tent、EATA、CoTTA、SAR等方法对比，CASTER在27/28种骨干‑数据集组合上优于k‑NN，平均状态占用比k‑NN低18倍；在无梯度环境下的ImageNet‑C等极端低覆盖情况，凭证机制可将平均误差从‑3.35点提升至+1.69点；总体在多数场景中实现轻量高效的自适应。

**⚠️ 局限性**

局限在于：仅适用于冻结特征空间；假设目标分布可用单一类共享仿射变换近似，若类间偏移大或批量不足会失效；可运输性证书只适用于CASTER，不能泛化到梯度更新的适应方法；在BatchNorm骨干上梯度方法仍优于CASTER。

---

## 346. DART: Aiming for Tail-Delay Control in Reconfigurable Networks

**arXiv ID:** 2609.01071 | [PDF](https://arxiv.org/pdf/2609.01071v1)

**作者:** Hossein Mohammadalizadeh `[一作]` (Hasso-Plattner-Institute), Holger Karl `[通讯]` (Hasso-Plattner-Institute)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于机会性承诺的调度策略DART，结合目标选择、路径规划与中途服务，以最小化加权95%分位服务时间。

**💡 创新点**

创新点在于引入紧急度度量（结合待处理延迟、到达时间及返回时间），通过承诺目标并在路径中动态决定是否停留服务，从而在随机、非对称的重配置时间环境下实现尾部延迟控制。

**🔧 技术方法**

技术方法包括：多队列轮询系统建模、期望重配置时间路径表、基于加权延迟的紧急度优先级、延迟守护和堆积守护两种阈值判定中途服务，实验使用离散事件模拟。

**📊 数据集**

使用合成数据集：六种压力拓扑（单/多路径，含高优先级/低优先级节点），每个拓扑包含随机重配置时间分布（确定性、指数、对数正态、洛马克斯）以及多种负载与权重设置。

**📈 对比分析**

与Tian、DVO和改进版Tian-T基线在相同图结构和服务能力下比较。DART在所有六个拓扑上都获得最低的加权P99，最高可比最强基线低23%，在负载逼近饱和时差距最为显著。

**⚠️ 局限性**

局限性包括：仅考虑每类作业由单一配置服务、参数α、β在所有图中保持不变、缺乏对动态退路惩罚的自适应调节、实验仅在合成图上验证，未在真实网络环境下评估。

---

## 347. RPCBench: A Benchmark for Proactive Premise Critique in LLM-based Recommendation

**arXiv ID:** 2609.00918 | [PDF](https://arxiv.org/pdf/2609.00918v1)

**作者:** Zhongru Chen `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出并实现了 RPCBench，针对 LLM 推荐助手的前提批判能力进行系统评估。

**💡 创新点**

创新点在于构建了基于证据的推荐前提批判基准，涵盖十种前提失效类型，并提供细粒度检测、定位、策略与证据忠实度评估框架。

**🔧 技术方法**

采用多模型生成与审核、细粒度指标（PDR、CLA、CSQ、CPCC、EFI 等）及自动 LLM 评判与人工验证相结合的方法进行评估。

**📊 数据集**

数据来源于五个公开推荐数据集（MovieLens-1M、MIND-small、Yelp Local、Amazon Sports、Goodreads Dual-Domain），共构造 4,623 个测试实例。

**📈 对比分析**

对 11 种闭源与开源 LLM 进行对比，平均前提检测率仅 51.5%，最优 CPCC 仅 0.53，证明主动检测是瓶颈；在证据忠实度方面 GPT‑5.5 表现最好。

**⚠️ 局限性**

局限性包括可能存在预训练泄露、生成器偏差、仅英文、数据集证据不统一、以及安全与合规前提样本仅用于评估而非真实流量。

---

## 348. TGR: Advancing Industrial Recommendation from Generative-Paradigm Ranking toward Unified Generation and Reasoning

**arXiv ID:** 2609.00986 | [PDF](https://arxiv.org/pdf/2609.00986v1)

**作者:** TGR Team `[一作]`, Chengxiang Zhuo `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了腾讯生成式推荐框架（TGR），通过三个方向（TGR-GenRank、TGR-GenRec、TGR-Reason）将传统分层式推荐系统迁移到生成式范式，并在数亿日活量级的实际业务中落地。

**💡 创新点**

创新点包括：
- CCFormer：跨字段分离注意力、子空间词混合与层级序列压缩，实现可扩展的生成式排序；
- BARGE：解决下一词预测在推荐中的两大结构缺口（item‑boundary 与 semantic‑drift）——通过 Item‑Context‑Aware Attention、Hierarchical Path Reranking 与 Dual‑Path Decoding；
- HiGR：从单词级到整幅排列的生成，利用 PCRQ‑VAE 前缀对比学习、Hierarchical Slate Decoder 及 ORPO 列表级对齐；
- TGR‑Reason：将 LLM 推理迁移到离线，通过 LatentRec 产生 Reason Tokens 并在在线推理中注入，无需实时推理成本。

**🔧 技术方法**

技术手段：大规模 Transformer（LLM‑style）、跨字段交叉注意力、子空间 token‑mixing、层级序列压缩、双通道 OSQ‑VAE、双路径解码、ORPO 列表对齐、LatentRec 训练框架（Soft Token Selection、Per‑step Reasoning Loss）以及高效的离线检索与 KV 缓存方案。

**📊 数据集**

数据集：
- 公共基准：Taobao、KuaiRec；
- 腾讯工业数据：约 4 b+ 交互样本、30 m 用户、10 m 物品、行为序列长度 1 k；
- BARGE 场景数据：两大业务场景（GraphSAGE+NANN 与 GraphSAGE+BERT）覆盖数千万候选、数千万用户。

**📈 对比分析**

比较方式与性能：
- 公开基准上，CCFormer 在 Taobao、KuaiRec 的 AUC/GAUC 均领先 1.4–1.7 pp；
- 工业数据上，CCFormer 超过 HSTU、OneTrans、STCA 0.28 pp AUC、0.5 pp GAUC；
- BARGE 在两大场景 Hit@5 提升 10.2–16.9 pp；
- HiGR 在工业数据 NDCG@5 提升 15.9–21.3 pp；
- TGR‑Reason Cold‑Start 新用户 Hit@1 从 4.5 % 提升至 26.1 %（+477.8 %）。
- 在线 A/B：CCFormer 在 5 个场景 CTR 提升 0.7–3.6 pp；BARGE CTR+1.34 pp、阅读时长+1.70 pp；HiGR 在 3 场景平均观看时长+1.14 pp、广告收入+0.56 pp。

**⚠️ 局限性**

局限性：
- 生成式模型仍受序列长度和计算开销限制，尤其是全序列自回归解码；
- 需要复杂的工程管线（离线 Reason Token 生成、KV 缓存、Beam 控制），维护成本高；
- 依赖 LLM 训练资源，模型规模与部署成本呈线性增长；
- 目前主要针对单轮推荐，尚未覆盖多轮交互、长尾/冷启动的持续学习；
- Reason Tokens 的离线生成可能导致时效性不足，对动态场景响应慢。

---

## 349. From Saliency to Discriminability: Rank-Preserving Visual Token Pruning for VLM Rerankers

**arXiv ID:** 2609.00667 | [PDF](https://arxiv.org/pdf/2609.00667v1)

**作者:** Siyi Liu `[一作]` (Hong Kong University of Science and Technology), Yongqi Zhang `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练的视觉令牌裁剪框架RaDiCal，专为VLM列表式重排序任务设计；

**💡 创新点**

创新点在于利用归一化注意力熵诊断注意力可靠性，结合注意力无关的跨候选区分度先验DTI，并通过熵衍生的信任系数实现令牌评分融合与层级裁剪的统一调度；

**🔧 技术方法**

核心技术包括：归一化注意力熵计算、Discriminative Token Importance（DTI）评分、注意力信息熵衍生的AttentionInfo、熵驱动的α系数校准、α-Maximin层级选择与几何融合；

**📊 数据集**

在Flickr30K、MSCOCO、FashionIQ三大检索基准上进行评估，使用Qwen3-VL-4B、InternVL2.5-8B等VLM架构；

**📈 对比分析**

与六种主流裁剪方法（FastV、PyramidDrop、SparseVLM、DART、VisionZip、LowRes）对比，RaDiCal在20%令牌预算下在所有指标上均排名第一，MRR@10在MSCOCO上甚至超过Dense；在10%预算下仍保持与Dense相当；同时实现39–45% FLOPs减少，测量速度提升1.28–1.45×；

**⚠️ 局限性**

局限包括：仅针对列表式重排序，未验证对多图任务；所有组件为训练‑free，未探索任务特定学习；仅支持softmax注意力，可能不适用于线性/状态空间等注意力机制；未覆盖更广泛检索领域或多模态场景；

---

## 350. Beyond Landmark Extraction: A Framework for Robust Geometric Feature Construction in Structured Image Classification

**arXiv ID:** 2609.00634 | [PDF](https://arxiv.org/pdf/2609.00634v1)

**作者:** Saravana Mauree `[一作]` (Case Western Reserve University), Sakshi Arya `[通讯]` (Case Western Reserve University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在结构化图像分类中，通过基于 MediaPipe 手部关节检测的几何特征构造（坐标、距离、角度及其混合表示）来实现任务导向的维度约简，并系统评估其对分类性能的影响。

**💡 创新点**

创新点在于提出了一个以后关节点特征映射为核心的框架，利用扰动与消融实验明确不同几何表示的鲁棒性与信息贡献，证明混合特征在实际场景中能显著提升分类效果。

**🔧 技术方法**

主要技术包括 MediaPipe Hands 关节提取、坐标/距离/角度特征工程、混合特征拼接、逻辑回归与随机森林分类、交叉验证、统计检验以及人工扰动与消融分析。

**📊 数据集**

实验数据集包括自采摄像头手势集、SignAlphaSet、HaGRID 手势数据集，以及侧位颅骨射线图像集（用于演示框架迁移）。

**📈 对比分析**

通过在三套数据集上对不同特征映射使用逻辑回归和随机森林进行交叉验证，评估指标为准确率，并利用 Friedman 与 Wilcoxon 检验检验差异；结果显示混合特征在 HaGRID 上达到了约 85–88% 的最高准确率，而原始坐标特征表现最差。

**⚠️ 局限性**

局限性包括：在受控数据下表现无显著差异；仅考虑静态图像，未涉及时序建模；特征仅为欧氏向量，未探索核空间或深度特征；以及对关节检测误差的影响尚未深入研究。

---

## 351. When Metropolis and Hastings Meet Bradley and Terry: Exact MCMC From Preference Voting

**arXiv ID:** 2609.00905 | [PDF](https://arxiv.org/pdf/2609.00905v1)

**作者:** Ariel Smogorghevski `[一作]` (Technion Israel Institute of Technology), Yaniv Romano `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为 Pref‑MH 的 Metropolis–Hastings 采样框架，利用仅有的二值比较反馈（Judge）在任意状态空间上实现精确的条件采样。

**💡 创新点**

核心创新在于：①将 Bradley–Terry (BT) 选择模型与 MH 接受概率建立等价关系；②设计一种固定预算 N‑vote 接受规则，使链在只获得有限一次比较反馈的前提下仍能保证到达目标分布；③证明该规则在 Peskun–Tierney 顺序下对所有同类固定预算采样器是最优的。

**🔧 技术方法**

主要技术包括：Metropolis–Hastings 算法、Bradley–Terry 选择模型、固定预算 N‑vote 接受规则、详细平衡证明以及 Peskun–Tierney 最优性分析；实现时使用多种生成模型（LLM、VLM、扩散模型）作为基模型，Judge 为大型语言模型或视觉语言模型。

**📊 数据集**

实验数据集涵盖：①人工生成的 241 状态的离散分布；②包含 50 条短情景描述的文本生成任务；③使用 SDXL‑Turbo 生成的连续噪声空间图像；④药物分子设计任务中的 SMILES 序列，使用 Qwen3‑235B 进行偏好评估。

**📈 对比分析**

与基线的比较包括：无条件基模型、使用点分数的 MH、错误的 plug‑in MH（基于样本估计胜率）、MARS（基于手工分子评分）以及不同数量的 Judge 投票。结果显示：Pref‑MH 在图像生成中成功率从 4.5%/7.4% 提升到 63.6%；在文本生成中各属性得分均优于点分数方法；在分子设计中 MolSkill（外部学习偏好评估）均值/中位数远优于其他方法，同时保持高多样性和合成可行性。

**⚠️ 局限性**

主要局限包括：①假设 Judge 的偏好满足 BT 模型，若实际偏好偏离该假设，目标分布的精确性会受到影响；②需要多次 Judge 查询，成本较高，尤其在人类评审场景；③算法依赖于可评估的提议分布（或可采样的基模型），对某些复杂结构化空间仍有挑战。

---

## 352. Subspace Levenberg Marquardt Algorithms in Training Neural Networks

**arXiv ID:** 2609.00789 | [PDF](https://arxiv.org/pdf/2609.00789v1)

**作者:** M. Duc Hoang `[一作]` `[通讯]` (University of California, Davis), M. Duc Hoang (University of California, Davis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文评估了子空间Levenberg-Marquardt算法在神经网络回归和分类任务中的表现，并与经典的LM方法及其他流行的一阶算法（如SGD和Adam）进行了比较。

**💡 创新点**

创新点在于提出了Krylov子空间LM（KSLM）和混合子空间LM（HSLM）方法，这些方法通过限制LM步骤到低维子空间来提高计算效率，同时保留重要的曲率信息。

**🔧 技术方法**

使用了Levenberg-Marquardt算法及其子空间变体KSLM和HSLM，结合了多种梯度和曲率信息的自适应低维子空间构建。

**📊 数据集**

使用了两个神经网络问题的数据集：一个非线性回归问题和13位奇偶校验分类问题。

**📈 对比分析**

在回归问题中，LM和KSLM在迭代次数上表现相似，但HSLM在计算时间上显著优于它们，且在分类问题中HSLM的外部迭代次数最少，显示出更快的收敛速度和更低的计算成本。

**⚠️ 局限性**

限制在于较大的数据集可能会增加形成和求解相关子问题的成本，从而限制可扩展性。

---

## 353. HyperWorld: Hypergraph-Structured State Serialization Improves Learned Textual World Models

**arXiv ID:** 2609.00002 | [PDF](https://arxiv.org/pdf/2609.00002v1)

**作者:** Yun-Jian Zhang `[一作]`, Mu-Jiang-Shan Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种基于超边单元（Hyperedge Units）的效果预测框架，并将其与LoRA世界模型结合，构建了一个从原始观察到规划动作的完整流水线。

**💡 创新点**

创新点在于：①利用超边单元将文本三元组压缩为更高效的表示，提升信息保持与可解释性；②在大规模语言模型（Qwen2.5）上采用LoRA微调，实现轻量化的世界模型；③将效果预测与WM‑guided规划相结合，采用贪婪搜索实现即时决策。

**🔧 技术方法**

使用的技术包括：超边单元编码、LoRA微调技术、Qwen2.5大语言模型、效果预测（ADD/REMOVE/INFEASIBLE）以及基于效果的WM‑guided规划算法。

**📊 数据集**

主要使用的基准数据集为TextWorld（多种文本游戏环境），同时在真实游戏场景中评估成功率。

**📈 对比分析**

与传统基于句子或三元组的效果预测方法对比，本文在EM、F1、rollout F1指标上均有显著提升；在真实游戏中的成功率也高于现有最优方案，说明系统在复杂环境下的鲁棒性更强。

**⚠️ 局限性**

局限性包括：①对TextWorld等文本游戏的适用性较强，泛化到更复杂或多模态环境仍需验证；②大语言模型的推理成本高，尤其在实时决策场景下需要进一步优化；③超边单元的结构设计仍依赖人工规则，可能难以自动化扩展。

---

## 354. On-the-Fly3R: Towards Robust Online 3D Reconstruction with Feed-Forward 3R Models for Large-Scale UAV Scenarios

**arXiv ID:** 2609.00923 | [PDF](https://arxiv.org/pdf/2609.00923v1)

**作者:** Zhe Shen `[一作]` (Wuhan University), Zongqian Zhan `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `51c0528b-f690-4182-ae60-bb5f046c276c` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为On-the-Fly3R的无训练、渐进式在线3D重建框架，用于在大规模无序无人机图像集上扩展现有的前馈3R模型。

**💡 创新点**

核心创新在于两点：①基于检索的动态子集构造，自动挑选空间相关图像形成几何一致的局部上下文；②验证-拒绝-重试机制，在合并到全局地图前对局部重建的几何一致性做严格校验，防止误差累积。

**🔧 技术方法**

采用冻结的3R模型（如Pi3、VGGT‑Omega）作为局部重建器，利用SupScene检索描述子、Sim(3)对齐、置信加权最小二乘、Huber鲁棒损失、Pose Graph优化（GTSAM）等技术。

**📊 数据集**

在GauU‑SceneV2（四个弱排序场景、424–1,500张图像）、UrbanScene（Residence 2,582张、Campus 5,871张）以及室内7Scenes数据集上进行评估。

**📈 对比分析**

与11种最先进的可扩展3R方法（Streaming、Chunk-based）以及原生全局推理进行对比，On-the-Fly3R在覆盖率（≥89%）和绝对定位误差（例如Pi3x在GauU‑SceneV2的ATE 4.20 m）上显著优于对手，且GPU内存始终低于24 GB。

**⚠️ 局限性**

局限性包括：①对初始种子图像的依赖，种子选择不佳会影响后续检索；②检索质量决定子集构造效果，纹理稀疏或重复场景下可能失效；③在极端多相机或极宽基线的无人机数据中，检索与验证机制的鲁棒性尚待进一步验证。

---

## 355. Feed-Forward Multi-view Multi-person Reconstruction with Contrastive Human-Aware 3D Representation

**arXiv ID:** 2609.00745 | [PDF](https://arxiv.org/pdf/2609.00745v1)

**作者:** Yuanwang Yang `[一作]` (Tianjin University), Kun Li `[通讯]` (Tianjin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出统一的人类感知3D空间进行前向多视角多人体3D重建框架

**💡 创新点**

创新点在于引入空间对比学习提升3D特征辨识度与一致性，并将相机标定、跨视角关联与人体重建在同一3D空间中一次性完成

**🔧 技术方法**

技术包括基于VGGT的几何先验+3D高斯投影、视图投影重采样融合、多模态特征融合、空间对比学习、SMPL参数直接回归

**📊 数据集**

使用EgoHumans和OcMotion两个多摄像机多人体数据集

**📈 对比分析**

与HSfM、Multi-HMR、DMMR等方法对比，实验证明在CA-MPJPE、GA-MPJPE、PA-MPJPE以及相机误差AE/s-TE等指标上均优于或接近SOTA，且推理速度大幅提升

**⚠️ 局限性**

主要局限在于依赖VGGT几何先验，对极端视角跳跃和低纹理区域的深度估计敏感，且未考虑时序动态场景

---

## 356. The Answer Is Not the Argument

**arXiv ID:** 2609.00264 | [PDF](https://arxiv.org/pdf/2609.00264v1)

**作者:** Will Yeadon `[一作]`, Craig P. Testrow `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估链式思维监测器在有无参考答案条件下对物理推理错误的检测与定位能力。

**💡 创新点**

提出了将答案一致性检查与推理验证区分开的概念，并通过信息阶梯实验揭示答案获取对监测器性能的偏倚。

**🔧 技术方法**

使用多模型生成物理推理链、双臂人工/LLM辩论式注释构建参考标准，以及八款LLM监测器在六个信息阶梯下的评价与统计。

**📊 数据集**

采用 Humanity's Last Exam（HLE）物理子集，约79道题，生成237条无人工插入错误的推理链。

**📈 对比分析**

通过平衡准确率、精确定位率和细胞限制召回等指标进行比较；答案认证提升整体平衡准确率至约0.80，但在critical trace上提升有限（≈0.06），表明监测器在答案一致性检查方面更优。

**⚠️ 局限性**

参考标准依赖人工与LLM双臂，可能受答案偏置；样本规模小、仅限物理领域，且未涵盖对抗性错误，评估结果对实际安全监督的可推广性有限。

---

## 357. DSG: Dynamic 3D Scene Graph Construction for Embodied Agents in Changing Indoor Environments

**arXiv ID:** 2609.00619 | [PDF](https://arxiv.org/pdf/2609.00619v1)

**作者:** Ming Liao `[一作]` (Harbin Institute of Technology), Weiyang Lin `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在室内环境中动态构建3D场景图的框架DSG。

**💡 创新点**

创新点在于双视角渲染对象变化检测以及将多粒度视觉上下文与LLM结合进行空间关系推理。

**🔧 技术方法**

使用3D高斯Splatting、Grounding-DINO、SAM、CLIP、VLM、LLM（如Qwen3-VL-Plus）以及双视角渲染与可视化视点优化等技术。

**📊 数据集**

采用Dyn-THOR、3RScan和真实的RealSense数据集。

**📈 对比分析**

与ConceptGraphs和DynamicGSG比较，DSG在节点精度、召回率、F1以及边的匹配率上显著提升，MRR降至2.1%。

**⚠️ 局限性**

局限在于仍需较高算力、对稀疏观测的鲁棒性有限，且对长期动态场景的连续追踪尚未实现。

---

## 358. GlitchLab: A Hardware-in-the-Loop Optimizer for Physical Fault Injection

**arXiv ID:** 2609.00502 | [PDF](https://arxiv.org/pdf/2609.00502v1)

**作者:** Tanvir Hossain `[一作]` (Keysight Technologies), Arindam Bhattacharyya `[通讯]` (Keysight Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出GlitchLab平台，针对物理故障注入（FI）实现在线搜索，提出两种搜索策略RL-Q和SOBAS；

**💡 创新点**

创新点在于：①将延迟、电压和脉冲长度分别视为时序门、严重度控制，充分利用结构化的硬件反馈；②设计了两种互补的搜索策略——一种利用物理模型与时序门分解（RL‑Q），另一种基于结构化结果的随机森林代理（SOBAS）；

**🔧 技术方法**

采用强化学习（Q‑learning）+上置信界（UCB）和随机森林分类器、目标概率+不确定性权衡的采样策略；实现了闭环硬件执行、结果分类与恢复的完整流水线；

**📊 数据集**

在三块相同型号的Riscure Pinata MCU上进行实验，使用AES‑128、密码校验和有限循环三种目标程序，探测空间为电压[2c‑3.9V]、延迟[冷启动或局部区间]、脉冲长度[2c0‑500ns]；

**📈 对比分析**

与SMAC、Gaussian‑process Bayesian、随机搜索、网格搜索等基线在相同硬件、同一试验预算（5000次）下对比；发现RL‑Q和SOBAS在所有目标均能找到目标故障，且相较基线节省2–85倍硬件尝试、1–1,237倍时间；在目标发现后，SOBAS在重复率上提升7.3–21倍，而RL‑Q在覆盖率上提升30%（在相同目标率下）；

**⚠️ 局限性**

局限性：实验仅覆盖电压注入在同型号 MCU 上的三种目标，未验证跨模型、跨架构、跨注入方式的迁移性；对目标函数的离散化与阈值假设在不同硬件或噪声环境下可能失效；对高维多参数注入（如 EM、激光）仍需进一步扩展；

---

## 359. On Synthesis of Metric Interval Temporal Logics

**arXiv ID:** 2609.01032 | [PDF](https://arxiv.org/pdf/2609.01032v1)

**作者:** Hsi-Ming Ho `[一作]` (University of Sussex), Khushraj Madnani `[通讯]` (IIT Guwahati)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一套完整的被动学习框架，将时序间隔时态逻辑(MITL)的学习问题通过特征合成与子投影冲突检测，转换为标准无时序LTL学习，从而实现对时序约束的自动推导。

**💡 创新点**

创新点在于：①引入“量化区分度”与“pivot”概念，精确捕获时序差异；②利用子投影冲突检测与模板注入，先化简结构差异；③采用分层特征合成与集合覆盖策略，逐步引入新原子命题；④通过谓词最小化实现谓词压缩；⑤整体框架保证完整性、可终止性，并可与任何离线LTL学习器无缝集成。

**🔧 技术方法**

核心技术包括：子投影冲突检测（枚举原子命题子集），模板注入（滑动窗口评估），分层特征合成（基于Pnueli与单一区间的特征生成，集合覆盖贪心求解），谓词最小化（Petrick或贪心求解），以及离线LTL学习器Bolt。

**📊 数据集**

实验数据集涵盖三类基准：
• 简单时序自动机（不同长度与样本数）
• 常见CPS规范模式（四个典型MITL模式）
• 火车闸门控制器的时序自动机（多种长度与样本规模）。

**📈 对比分析**

与现有方法（如针对TIMED正则表达式的枚举、SMT求解器等）对比，本文框架在三类基准上均表现出更快的运行时间、更简洁的公式（平均运算符数下降），且在大样本规模下依旧保持可扩展性。Bolt作为后端学习器的性能瓶颈在大于300条负样本时被最小化技术所缓解。

**⚠️ 局限性**

局限性包括：①需要正负样本对来提取pivot，缺失负样本时需改为单类学习；②子投影枚举在原子命题数较大时指数级增长；③最终公式仍依赖离线LTL学习器的搜索效率，Bolt在极大样本集上会耗时；④生成的特征与公式不一定是最优或最简，需后续优化。

---

## 360. Solutions to Three Conjectures and an Open Problem on Binary BCH Codes

**arXiv ID:** 2609.00532 | [PDF](https://arxiv.org/pdf/2609.00532v1)

**作者:** Xiaoqiang Wang `[一作]` (Hubei University), Dabin Zheng `[通讯]` (Hubei University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了三类二进制BCH码的确切最小距离的三个猜想及一个开放问题，并通过构造达到BCH界限的码字来解决这些猜想。

**💡 创新点**

创新点在于完全解决了三个关于二进制BCH码的最小距离的猜想，并进一步研究了一个更复杂的BCH码的最小距离问题。

**🔧 技术方法**

使用了有限域的元素、不可约多项式和从原始BCH码的提升论证等技术。

**📊 数据集**

使用了多种二进制BCH码的参数集，包括长度为n=(2^2s+1)(2^s-1)、n=2^2s+2^s+1和n=(4^s-1)/3的码。

**📈 对比分析**

通过构造特定的码字，证明了猜想的最小距离等于BCH下界，性能上达到了理论上的最优解。

**⚠️ 局限性**

限制在于对于某些参数组合，开放问题8.4的结论并不成立，提供了反例说明在一般情况下不成立。

---

## 361. Trust Your Guide Only When Certain: Uncertainty-Aware Sparse Alignment at Inference Time

**arXiv ID:** 2609.00624 | [PDF](https://arxiv.org/pdf/2609.00624v1)

**作者:** Zeen Zhu `[一作]` (Harbin Institute of Technology), Jing Li `[通讯]` (Harbin Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于不确定性稀疏对齐（Trust-based Uncertainty Sparse Alignment，TUSA），通过认知仲裁器在推理阶段动态判断何时需要对大型语言模型（LLM）的生成进行干预，从而实现对齐与通用文本生成的双重优化。

**💡 创新点**

创新点在于：①将对齐视为稀疏决策问题，跳出传统全步密集干预的范式；②引入认知仲裁器，结合温度缩放后的置信度（KL 与最大熵对比）和语义显著性（IDF）来计算联合必要性分数；③采用自适应阈值动态抑制低置信度噪声，保证只有在真正需要时才触发干预。

**🔧 技术方法**

核心技术包括温度缩放、KL 散度评估置信度、逆文档频率（IDF）衡量语义显著性、移动平均自适应阈值、以及基于门控的稀疏干预策略。

**📊 数据集**

实验数据集涵盖：安全性评估使用 PKU‑SafeRLHF、BeaverTails、HarmfulQA；通用能力评估使用 AlpacaEval、JustEval；模型使用 Llama‑3.1‑8B、Llama‑3.2‑3B、Mistral‑7B‑v0.1/0.2/0.3 及其 4M 微代理（弱监督）。

**📈 对比分析**

与密集干预基线（MARA）及参数更新方法 ConfPO 对比，TUSA 在安全性偏好率平均提升约 +9.6%，在通用性偏好率平均提升约 +5.6%；同时将干预比例减少约 50%，保持与密集方法相近的推理延迟。

**⚠️ 局限性**

限制主要包括：仅在 3–8B 规模模型验证，尚未验证大模型可扩展性；需白盒访问内部 logits，无法直接应用于闭源 API；对齐效果受弱监督模型能力限制，可能对极其细微或复杂的对抗场景处理不足。

---

## 362. Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance

**arXiv ID:** 2609.00363 | [PDF](https://arxiv.org/pdf/2609.00363v1)

**作者:** Teng-Ruei Chen `[一作]` `[通讯]` (Krixvon), Teng-Ruei Chen (Krixvon)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型的INT8量化推理中不同GPU核实现的整数累加与epilogue（缩放与舍入）差异进行系统性故障注入与检测，验证其可比性与可检测性。

**💡 创新点**

证明单步bfloat16舍入误差仅能产生1个ULP差异，导致一spacing容差检测无法发现多种epilogue缺陷，并提出通过强制使用幂2缩放实现跨核比特级一致性。

**🔧 技术方法**

使用自构造的精确参考管线、九类故障注入、七项合规检查、预注册预测矩阵以及对Qwen3-1.7B/8B/14B模型的捕获和推理测试。

**📊 数据集**

Qwen3系列（1.7B、8B、14B）量化线性层权重与激活，配合预置提示进行端到端生成测试。

**📈 对比分析**

在对比两核（CUTLASS vs Triton）时，强制幂2缩放后两核在每层和完整生成序列上都实现了字节级一致性，困扰度在1.7B、8B、14B分别在-0.28%至+0.48%之间，90%置信区间上限分别为+0.71%和+0.76%。

**⚠️ 局限性**

仅评估了九类假设性故障，未覆盖所有可能的epilogue缺陷；仅针对Qwen3和CUTLASS–Triton核对，缺乏跨模型与跨核的广泛验证；端到端吞吐量仅在1.7B上测得。

---

## 363. Kernelization of 2-Club Cluster Edge Deletion on Interval Graphs

**arXiv ID:** 2609.01021 | [PDF](https://arxiv.org/pdf/2609.01021v1)

**作者:** Ajinkya Gaikwad `[一作]` `[通讯]` (Czech Technical University in Prague), Ajinkya Gaikwad (Czech Technical University in Prague)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在区间图上 2‑Club Cluster Edge Deletion 的核化与复杂度，并证明了在分割图上该问题为 NP‑hard，提供了多项式核化以及在单位区间图上的多项式解法。

**💡 创新点**

提出了针对 2‑Club 的新型归约规则（包括嵌套与阶梯团结构），实现了 O(k^5) 的顶点核，首次在该图类中获得多项式核；同时给出了分割图 NP‑hard 的证明。

**🔧 技术方法**

利用区间图的几何表示、Helly 定理、Erdős‑Szekeres 单调子序列、最大团/最大独立集性质以及专门的归约规则，构建了完整的核化流程。

**📊 数据集**

本文为理论分析，未使用实验数据集，全部结果通过证明给出。

**📈 对比分析**

由于本研究主要是理论性，未进行实验比较；在单位区间图上算法时间为 O(n^2)，并给出多项式核化的 O(k^5) 上界。

**⚠️ 局限性**

限制在于 2‑Club 的多项式时间解法仅适用于单位区间图，归约规则 11 仅适用于 s=2，且 2‑Club 在一般区间图上的多项式可解性仍是未解决的问题。

---

## 364. ViTAL-X: Video-Text Alignment with Cross-Modal Temporal Edits

**arXiv ID:** 2609.00505 | [PDF](https://arxiv.org/pdf/2609.00505v1)

**作者:** Sethuraman T `[一作]`, Simon Jenni `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视频‑文本模型的时序盲点，提出自监督的跨模态时序编辑框架XTE和轻量化时序适配器ViTAL‑X，以提升视频时序推理。

**💡 创新点**

①跨模态同步时序编辑生成硬负样本，②在冻结的图像‑文本骨干上加入低秩适配和浅层时序Transformer，③提出XTE‑Bench诊断测试。

**🔧 技术方法**

使用CLIP/OpenCLIP、SigLIP‑2骨干、LoRA低秩适配、轻量化时序Transformer、对比学习+margin时间损失以及自监督XTE变换。

**📊 数据集**

在OpenVid‑1M、Droplet‑10M、YouCook2、COIN、Ego4D、HowTo100M等数据集上，训练约1.2M视频‑文本对，并在XTE‑Bench、VideoComp、RTime、TemporalBench等评测集上测试。

**📈 对比分析**

与大型7B参数模型及其它CLIP扩展比较，ViTAL‑X仅0.4B参数、1M训练样本，在XTE‑Bench 69.4%准确率、TemporalBench 67.8%等时序基准上实现SOTA，且超越更大模型且数据量少得多。

**⚠️ 局限性**

仅捕捉离散顺序，对连续速度、时长等细粒度动态缺乏显式建模；文本编辑偶尔难以完整覆盖重叠事件；适配器引入轻微计算开销。

---

## 365. Neural means and kernel corrections for operator learning

**arXiv ID:** 2609.00389 | [PDF](https://arxiv.org/pdf/2609.00389v1)

**作者:** Yitzchak Shmalo `[一作]` `[通讯]` (Hebrew University of Jerusalem), Yitzchak Shmalo (Hebrew University of Jerusalem)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出将神经网络的均值与对其残差和特征的 Matérn 核回归相结合，构建了一个三阶段的混合算子逼近器；

**💡 创新点**

创新点在于先训练准确的神经均值，再用精确的 Matérn 核对残差进行高精度校正，同时通过自适应特征空间核校正与交叉验证确定超参数，并提供分布式可验证的不确定性区间；

**🔧 技术方法**

使用的技术包括残差多层感知机、傅里叶神经算子、UNet、特征空间核回归、堆叠（stacking）、对称性反射平均、核流正则化、贝叶斯高斯过程的后验标准差与分位数校准、以及有效维数与马尔科夫-帕杜尔分布分析；

**📊 数据集**

实验数据集涵盖：结构力学基准（de Hoop 等）、OCO-2 大气辐射传输模拟器、以及包括 Burgers、Darcy、Advection、Helmholtz、Navier–Stokes 在内的七大算子学习基准；

**📈 对比分析**

与公开基准进行严格对比：在结构力学 20k 训练样本下达到 4.55% 的相对 L2 测试误差，匹配 PARA‑Net；在 1250 样本低数据协议下误差为 4.66%，优于 DeepONet 等方法；在 OCO‑2 O2 波段以 3.83% 的相对误差击败传统 Gaussian‑Process 16.89%，并在两个指标上同时获胜；

**⚠️ 局限性**

主要限制包括：对低维输入（如 41 维负载向量）和固定几何的强依赖；对高维输入场需要诱导点或随机特征近似，可能削弱精确性和误差证明；残差共享成分被视为数据噪声，难以进一步降低误差；不确定性区间依赖于验证拆分的可交换性，可能在模型选择后略有偏差。

---

## 366. FALCON: Fault-Tolerant Magnetic Tunnel Junction-Based In-Memory Stochastic Architecture for Reliability-Critical Edge AI Applications

**arXiv ID:** 2609.00701 | [PDF](https://arxiv.org/pdf/2609.00701v1)

**作者:** Farzad Razi `[一作]` (University of Minnesota), Marc Riedel `[通讯]` (University of Minnesota)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于MTJ的容错随机计算（SC）内存内算术体系结构FALCON，能够在内存阵列中直接执行完整算术单元，完成乘法、加法、除法、最小/最大、绝对差等运算，并在边缘AI任务中实现形态学闭运算；

**💡 创新点**

创新点在于：①使用确定性低差异(bit‑mapping)把二进制值映射为可控交叉相关的随机位流；②利用MTJ的读写特性实现可重构逻辑单元（LIM），无需外部随机数发生器或额外校准；③将完整的SC算术功能直接嵌入内存阵列，兼顾能效与鲁棒性；

**🔧 技术方法**

核心技术包括：磁隧道结(MTJ)非易失性存储、确定性低差异位流映射、逻辑‑in‑memory (LIM)结构、随机计算（SC）算术原语、电压缩放与工艺变异分析；

**📊 数据集**

主要在形态学闭运算案例中使用灰度图像和二值掩码（256×256像素），并通过注入10%–30%的盐椒噪声与设备噪声进行验证；

**📈 对比分析**

与传统二进制IMC以及若干SC‑IMC方案（如SCRIMP、SC‑CRAM等）在噪声鲁棒性、能耗和精度上进行对比；在30%设备噪声下，二进制IMC的IoU仅0.64，而FALCON保持0.89；能耗比von Neumann基线低约10倍；在工艺与电压缩放实验中保持功能正确性；

**⚠️ 局限性**

局限性包括：①需要较长的随机位流（N=1024）导致较高的时延；②SC本身精度受位流长度限制，难以实现极高精度算术；③在极大规模阵列中位流同步与功耗管理仍待进一步研究；④对比基准主要集中于形态学闭运算，其他AI工作负载尚未全面验证；

---

## 367. Unmasking Face Embeddings: Reading, Rendering and Naming with Foundation Models

**arXiv ID:** 2609.00411 | [PDF](https://arxiv.org/pdf/2609.00411v1)

**作者:** Fizza Rubab `[一作]` (Michigan State University), Arun Ross `[通讯]` (Michigan State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过仅一次无标签线性变换，将面部识别模型的嵌入映射至基础模型空间，使其可被自然语言检索、扩散解码器渲染为人脸图像以及在名称词表中进行命名。

**💡 创新点**

创新点在于发现并利用不同模型嵌入空间之间的线性兼容性，仅需一个线性映射即可让专用FR嵌入获得基础模型的多模态、语义化、生成化和开放词表能力。

**🔧 技术方法**

使用技术包括线性嵌入对齐、CLIP/MetaCLIP/SigLIP的文本检索、扩散解码器（Kandinsky、Stable unCLIP）渲染、以及对名称词嵌入的匹配。

**📊 数据集**

采用的数据集包括 CFP、UTKFace、CelebA、WebFace4M、MS1MV2、LAION‑2B、LFW 等，用于训练线性映射、评估检索、重建与命名。

**📈 对比分析**

在与原始基础模型、未对齐和随机变换的对照实验中，mAP、FID、命名 Top‑1/Top‑5/Top‑10 几乎达到原模型上限，且显著优于未对齐与随机基线，表明线性映射能有效传递语义、外观与部分身份信息。

**⚠️ 局限性**

局限性包括依赖基础模型的表达能力（无法恢复细粒度身份信息）、仅能曝光已线性可访问的特征、对未覆盖的名称和稀有身份无效，以及易被公开模型利用，需进一步研究对抗性保护措施。

---

## 368. UI-Venus-2 Technical Report

**arXiv ID:** 2609.00028 | [PDF](https://arxiv.org/pdf/2609.00028v1)

**作者:** Venus Team `[一作]` (Ant Group), Beitong Zhou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 UI‑Venus‑2，一款通用的基础 GUI 代理，能够在多语种移动、Web 及桌面操作系统环境中自动执行自然语言指令。

**💡 创新点**

创新点包括：
• 通过统一闭环推理‑动作框架实现跨域执行；
• 同时扩展多语种移动环境、桌面使用能力和细粒度的轨迹/样本级验证机制；
• 采用多教师 on‑policy 蒸馏结合结构化动作监督，强化可执行行为的学习；
• 开源完整模型权重和评测基准，促进社区复现与进一步研究。

**🔧 技术方法**

技术手段：
• 基于 Qwen3.5/3.6 视觉‑语言模型做前置；
• 三阶段训练：大规模多模态中训 → 离线强化学习（任务特定监督）→ 多教师 on‑policy 蒸馏；
• 结构化动作感知与动作‑类型/参数自适应监督；
• 两级验证：轨迹级 SGV 与样本级 a‑priori 判断，提升奖励信号可靠性；
• 针对 CAPTCHA 的程序化生成与渲染引擎；
• 采用 CAPTCHAs、Grounding 等多任务融合训练。

**📊 数据集**

数据集与任务来源：
• 移动：MobileGym、VenusBench‑Mobile、AndroidWorld、MobileWorld、KnowUBench、MemGUI；
• 桌面：OSWorld‑Verified、DeskCraft、OSWorld 2.0；
• Web：WebVoyager、Online‑Mind2Web、REAL、Odysseys；
• Grounding：VenusBench‑GD、ScreenSpot‑Pro、OSWorld‑G‑R、UI‑Vision；
• CAPTCHA：VenusBench‑CAPTCHA、MCA‑Bench、Spatial‑CAPTCHA‑Bench、NextGen‑CAPTCHAs、Open CaptchaWorld；
• 安全评测：OSHarm、OSBlind。

**📈 对比分析**

对比方法：在每个基准上与多种通用 VLM（如 Qwen、Claude、Seed）和专用 GUI 代理（如 UI‑TARS、OpenCUA、Fara‑1.5）进行 1‑对‑1 性能评测，使用成功率、Pass@1、完整性得分等指标。UI‑Venus‑2 在多数基准上实现 SOTA：例如 MobileWorld 100 步 82.9% vs 前置 65%；WebVoyager 93.4% vs Fara‑1.5 89.3%；VenusBench‑CAPTCHA 79.9% vs Qwen3.6‑27B 53%；Grounding 80.1% vs UI‑TARS‑1.5 75%；安全性 OSHarm/OSBlind ASR 下降至 11‑48% 等。

**⚠️ 局限性**

局限性：
• 长序列任务（>200 步）仍出现成功率偏低，说明规划与错误恢复仍待加强；
• 验证机制虽更精准，但对极端动态环境（网络变动、弹窗）仍易误判；
• 安全评测中 OSBlind 的 ASR 仍高，提示模型在“安全隐患”场景下的鲁棒性有限；
• 多域通用性主要受任务生成质量和平台差异影响，部分桌面细节仍需微调；
• CAPTCHAs 的程序化生成难以覆盖所有新型验证码，实际部署时可能需要持续更新；
• 目前缺乏可解释性与实时性能评估，实际工业化部署时仍需关注。

---

## 369. Residual Kalman Dynamics for Event-Based UAV Forecasting

**arXiv ID:** 2609.00839 | [PDF](https://arxiv.org/pdf/2609.00839v1)

**作者:** Per Nyblom `[一作]` (Swedish Defence Research Agency (FOI)), David Gustafsson `[通讯]` (Swedish Defence Research Agency (FOI))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了基于事件相机的无人机边界框预测，提出在常数速度卡尔曼滤波器基础上学习加速度残差的残差卡尔曼模型。

**💡 创新点**

创新点在于将物理先验与数据驱动校正分离，通过学习加速度残差提升预测精度，并引入贪婪样本去相关诊断检验运动先验对模型的影响。

**🔧 技术方法**

采用事件帧与CSTR表征、ResNet‑18编码器、MLP残差头、优化的常数速度/加速度卡尔曼滤波、梯度下降、Ridge回归评估以及贪婪样本去相关算法。

**📊 数据集**

使用Florence RGB‑Event Drone Dataset（FRED）中的真实事件相机无人机飞行数据。

**📈 对比分析**

与常数速度/加速度卡尔曼滤波器、线性外推器、线性残差基线及之前的事件+RGB方法比较，在400 ms/800 ms短期和中期预测上，残差卡尔曼模型在ADE/FDE和mIoU上显著优于所有基线，并接近或超过此前报告的最佳方法。

**⚠️ 局限性**

局限性包括仅针对短/中期框预测，未建模预测不确定性；去相关诊断仅剔除线性运动先验，无法消除非线性或场景特定的捷径；事件表征与网络架构相对简单，可能未充分利用事件信息。

---

## 370. Skill Following: Evaluating Actual Skill Use in Retrieval-Enabled LLM Agents

**arXiv ID:** 2609.00549 | [PDF](https://arxiv.org/pdf/2609.00549v1)

**作者:** Seonghyeon Cho `[一作]` (Soongsil University), Chanjun Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现 Retrieval‑Invoked Actual‑Use Effect（RAE）指标，用来衡量 LLM 在检索技能后真正提升同一任务的效果，并在编码和数学任务上对 17 种 LLM 进行评估。

**💡 创新点**

通过同一任务的检索启用与禁用对比，消除任务选择偏差，揭示大多数模型在检索技能后效果为负，暴露传统平均指标的误导性。

**🔧 技术方法**

采用 paired execution protocol、BM25 检索、结构化 Markdown 技能库、对齐注解与多种诊断控制（空内容、随机、破坏等）。

**📊 数据集**

使用 MBPP+、HumanEval+ 与 Math500 三个公开 benchmark 及其对应的技能库。

**📈 对比分析**

对比 aggregate retrieval lift、OAE 与 RAE，发现多数模型 aggregate lift 为正但 RAE 为负，说明检索并未真正提升同一任务；在 Math500 同样出现逆转，表现与编码任务相似。

**⚠️ 局限性**

RAE 仅在固定检索接口与一次性任务上有效，无法覆盖长周期、多技能调用或可执行代码库等情景，且不提供完整的因果解释。

---

## 371. Using LLMs to Elicit Security Requirements for Service-Oriented Cyber Ranges

**arXiv ID:** 2609.00886 | [PDF](https://arxiv.org/pdf/2609.00886v1)

**作者:** Michail Takaronis `[一作]` (Norwegian University of Science and Technology), Sokratis Katsikas `[通讯]` (Norwegian University of Science and Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用大语言模型在服务导向的网络范围内快速生成安全需求，并将其映射到架构层面。

**💡 创新点**

首次将 LLM 与 SEBoK 过程结合，自动化需求生成并实现需求与架构层的可追踪映射。

**🔧 技术方法**

采用 SEBoK 指导的需求工程流程，五个 LLM（GPT‑5.2、Gemini 3.1 Pro、Grok 4.1、Sonar、Kimi K2.5）与结构化提示进行交互，随后人工合并。

**📊 数据集**

基于已定义的安全使命目标（SMO）与安全利益相关者需求（SSN）以及 SOR 的九层参考架构描述，未使用公开数据集。

**📈 对比分析**

通过与五位安全专家对 27 条需求在必要性、清晰性、完整性、可行性、可测试性等五项指标进行评估，接受率高（必要性 98.5%），可测试性最低（44.4%）。

**⚠️ 局限性**

主要限制在于提示内容质量决定结果、人工合并导致主观性、未对需求进行真实环境实施或验证。

---

## 372. Wave Function Backpropagation with Explicit Temporal-Interval Dynamics

**arXiv ID:** 2609.00503 | [PDF](https://arxiv.org/pdf/2609.00503v1)

**作者:** Byunggu Yu `[一作]` (University of District of Columbia), Justin Kim `[通讯]` (University of District of Columbia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了一种基于可微波函数的学习框架——Wave Function Backpropagation（WFB），通过幅度、波数、角频率和相位四个可学习参数，将时间间隔直接映射到神经元激活相位，实现了在feed‑forward轨迹预测中的结构化时空建模。

**💡 创新点**

创新点在于：①将时间间隔纳入可学习波函数的相位项，实现时间对激活的连续、可微调影响；②给出完整梯度推导并提出基于空间拉普拉斯算子的小波形曲率正则化；③在单一feed‑forward网络中直接引入非线性波响应，减少对传统ReLU激活的依赖。

**🔧 技术方法**

使用技术包括：可微波函数激活、基于梯度的优化（AdamW）、MSE目标、拉普拉斯曲率正则化、标准与拉普拉斯梯度混合更新、以及在运动特征和位置特征两种输入形式下的feed‑forward网络结构。

**📊 数据集**

采用了ETH/UCY与JAAD风格的行人/代理轨迹数据集，对帧级检测进行拼接，生成包含不规则采样间隔（Δt）的观察窗口，整体共计约425k帧、453k窗口。

**📈 对比分析**

通过与原始FFN基线、参数匹配的ReLU控制、以及显式Δt与无Δt的模型对比，发现WFB在运动特征输入下平均位置误差（ADE）降低约20%；在仅用位置输入、控制容量时，WFB仍优于匹配基线约10%；打乱Δt的实验更显优异；拉普拉斯正则化单独使用效果差，结合使用时可提升约1.9%。

**⚠️ 局限性**

局限性包括：仅在feed‑forward框架内验证，未涉及循环或注意力的时序传播；评估任务仅为轨迹预测，无法推广到其他时序建模；性能提升部分可能由模型容量增大导致；对Δt顺序的重要性尚未确定；拉普拉斯正则化参数需手动调节；缺乏理论收敛与逼近性质分析。

---

## 373. Can LLMs Use Relational Transformer Embeddings?

**arXiv ID:** 2609.00457 | [PDF](https://arxiv.org/pdf/2609.00457v1)

**作者:** Francisco Galuppo Azevedo `[一作]` (Kunumi Institute), Clarissa Lima Loures `[通讯]` (Universidade Federal de Minas Gerais)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

尝试将冻结的 Relational Transformer（RT）嵌入的表格表示作为软标记注入到大型语言模型 Qwen3.5‑4B 中，并通过监督微调与强化学习进行训练，以完成 RelBench 上的二分类任务。

**💡 创新点**

提出了将跨架构的 RT 表格表示与 LLM 软标记融合的“软标记注入”架构，并系统评估其在跨数据库、跨任务迁移场景下的效果。

**🔧 技术方法**

采用 MLP 投影层、LoRA 参数高效微调、链式思维监督微调（SFT）以及基于组的强化学习（GSPO）等技术。

**📊 数据集**

在 RelBench 的六个关系数据库上构造的 10 条二分类任务进行实验。

**📈 对比分析**

将融合模型与单独 RT 基线按 AUROC 进行对比，结果显示融合模型在大多数情况下无法提升性能，甚至低于随机且表现不稳定。

**⚠️ 局限性**

局限包括仅处理二分类任务、使用单个 4B LLM、训练样本有限、未探索更强的对齐目标或更大模型等。

---

## 374. AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes

**arXiv ID:** 2609.00052 | [PDF](https://arxiv.org/pdf/2609.00052v1)

**作者:** Xun Wang `[一作]` (CISPA Helmholtz Center for Information Security), Adam Dziedzic `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对代理式LLM API开发了基于工具调用行为的身份审计方法AgentProv，用MMD统计对比工具选择分布以识别模型是否被替换。

**💡 创新点**

创新点在于使用工具调用分布而非自然语言输出进行身份判定，提升对系统提示、部署差异的鲁棒性，并提供可解释的按模板的差异分解。

**🔧 技术方法**

技术包括构造功能等价工具集合的probe模板，使用一维one‑hot编码的工具选择分布，Delta核下的MMD两样本检验，并通过置换检验校准阈值。

**📊 数据集**

实验数据涵盖36个开放权重模型（13个家族）、9个闭源OpenAI模型、630个检查点对、5个模型在不同系统提示下、12个模型工具名称鲁棒性测试，以及5个模拟适应攻击的模型。

**📈 对比分析**

与MET、RUT及改编的LLMmap对比，AgentProv在100%识别不同模型且在自同一模型测试中0%误拒；在隐藏系统提示条件下误拒率仅7%，远低于MET（67%）和RUT（53%）。在查询预算1,000（或500）次内即可完成审计，成本低于$1。

**⚠️ 局限性**

局限包括需要可信的参考模型；无法直接证明底层权重，且对非工具调用的API不适用；对第三方终端缺乏真值，审计结果需结合侧信道与文档；对极端大规模系统提示或未覆盖的工具集合可能失效。

---

## 375. Agent-Enhanced Heterogeneous Graph RAG for Academic Question Answering

**arXiv ID:** 2609.00761 | [PDF](https://arxiv.org/pdf/2609.00761v1)

**作者:** Runsong Jia `[一作]` (University of Technology Sydney), Yi Zhang `[通讯]` (University of Technology Sydney)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向学术问答的代理化异构图增强检索生成框架。

**💡 创新点**

创新点在于将检索规划、证据充分性控制和图基验证三个步骤拆分为专用代理，并结合查询类型自适应检索。

**🔧 技术方法**

使用 GPT‑4‑turbo 作为代理和生成模型，Sentence‑Transformers 进行语义检索，基于图结构的重新排名和一致性评分。

**📊 数据集**

在 OpenAlex 与 DBLP 构造的学术异构图上进行评估。

**📈 对比分析**

与纯 LLM、图增强 RAG 以及已有代理检索方法对比，在准确率、F1 和 Hit@1 上均取得最高 76.68%/73.43% 的成绩。

**⚠️ 局限性**

局限包括对大规模图检索的扩展性、仅处理三种节点类型、以及代理决策仍依赖预设映射。

---

## 376. SpecMind: Enabling Spectrum Intelligence via Multi-Agent Hybrid Retrieval-Augmented Generation

**arXiv ID:** 2609.00427 | [PDF](https://arxiv.org/pdf/2609.00427v1)

**作者:** Songwei Dong `[一作]` (University of Virginia), Cong Shen `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该工作提出了一个多智能体混合检索增强生成系统SpecMind，能够在结构化许可表、图结构程序文件和文本法规三种不同模态的频谱数据上进行自动化推理，并公开了相应的数据库与450条问答基准。

**💡 创新点**

创新点在于将多智能体协作与检索增强生成相结合，为每种数据模态设计专门的检索机制（SQL、图检索、Embedding+重排），并构建统一的专家设计基准，显著提升跨源推理能力。

**🔧 技术方法**

技术主要包括多智能体框架、ReAct式提示工程、GraphRAG图检索、SQL查询生成、Embedding检索与重排、LLM推理（Qwen3‑8B、GPT‑5.2）以及Gemini 3.1‑Pro‑Preview评估器。

**📊 数据集**

使用FCC许可记录、FCC程序文件、NTIA国家频谱战略等数据构建的SpecMindCorpus，并基于此生成450条问答对的SpecMindQA基准。

**📈 对比分析**

与Web‑search RAG和SpectrumRAG等基线对比，在两大LLM（Qwen3‑8B、GPT‑5.2）上均实现了超过80% win率、近100%成功率，尤其在许可和复合查询上显著优于基线。

**⚠️ 局限性**

局限性包括对稀疏实体的检索精度不足、图中细粒度归因和作者关系表达不完整，以及结构化查询可能出现歧义，需要进一步完善图元数据和查询澄清机制。

---

## 377. Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback

**arXiv ID:** 2609.00165 | [PDF](https://arxiv.org/pdf/2609.00165v1)

**作者:** Ziwen Pan `[一作]` (Emory University), Ruoxuan Xiong `[通讯]` (Emory University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种双侧状态空间模型 TS-SSM，用于根据事件（评论）实时更新用户和商品状态并进行顺序推荐。

**💡 创新点**

创新点包括：① 采用“非随机缺失”多模态融合模块，将评论文本、图片等的出现模式与内容共同编码；② 在用户与商品状态演化中加入局部图信息传播与异向衰减的携带记忆；③ 在评分时对未观测商品进行时序对齐并引入动态偏置。

**🔧 技术方法**

技术手段包括：状态空间模型、层归一化、门控门控机制、基于邻居的局部图消息传递、时间窗口的共现图、对抗性损失（BPR）以及多任务监督（漂移、携带、可靠性）。

**📊 数据集**

使用六个亚马逊商品评论类别（Toys & Games、Pet Supplies、Sports & Outdoors、Electronics、Clothing、Home & Kitchen）以及 Goodreads Fantasy 子图作为实验数据集。

**📈 对比分析**

与 20+ 传统顺序、 multimodal 与基于日志的基线（如 BSARec、HM4SR、RecGPT 等）对照，TS-SSM 在 Recall@20 上平均提升 11.7%（对 BSARec 提升 14.8%–18.8%），在 Goodreads 上提升 12.6%。

**⚠️ 局限性**

局限性包括：对稀疏或不同评论行为的平台验证不足；当文本信息不丰富、评论多模态缺失或用户历史过短时优势可能减弱；模型计算成本高于轻量级序列模型；缺失机制仅为预测性而非因果解释。

---

## 378. ES-AHD: An Evolution Strategy Framework for Automatic Heuristic Design

**arXiv ID:** 2609.00023 | [PDF](https://arxiv.org/pdf/2609.00023v1)

**作者:** Yutao Lai `[一作]` (Guangdong University of Technology), Ping Guo `[通讯]` (Beijing Normal University)

**通讯引用:** 20669 | [OpenAlex ID](https://openalex.org/A5068048071)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ES-AHD 框架，将进化策略（Evolution Strategy）与大型语言模型（LLM）驱动的自动启发式设计（AHD）相结合，旨在生成高质量的组合优化启发式算法。

**💡 创新点**

创新点包括：① 通过 LLM 进行语义重组（Semantic Recombination），从精英个体中提炼核心洞见，形成搜索中心；② 使用温度采样的随机协方差适配（Stochastic Covariance Adaptation），动态调节探索与利用平衡，避免早熟收敛。

**🔧 技术方法**

采用了进化策略中的均值与协方差概念、LLM Prompting（以 GLM 4-Flash 为基础）、温度采样与随机扰动、以及语义反射技术来指导代码生成与改进。

**📊 数据集**

使用了 Traveling Salesman Problem (TSP) 的不同规模实例（N=20、50、100）以及 TSPLib 基准数据集进行实验评估。

**📈 对比分析**

与 EoH、ReEvo、FunSearch 等基线方法对比，ES-AHD 在训练集 Top‑4 平均得分（-6.209）和验证集 Val20/Val50/Val100（8.911）上均优于对手；在 TSPLib 实例中也取得更低的路径长度，整体性能明显提升，尤其在中小规模问题上表现最为突出。

**⚠️ 局限性**

局限性包括：对极大规模实例（如 100+ 节点）仍不如 ReEvo 的重写式改进方法鲁棒；依赖 LLM 的生成质量，模型失误或“hallucination”仍可能导致搜索偏差；且目前仅针对代码级启发式，尚未扩展到更复杂的组合优化领域。

---

## 379. Beyond Language Priors: Diagnosing and Fixing Visual-Origin Hallucinations in Multimodal LLM

**arXiv ID:** 2609.00231 | [PDF](https://arxiv.org/pdf/2609.00231v1)

**作者:** Peiyang Xu `[一作]` (Tsinghua University), Xiaolin Hu `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对多模态大型语言模型中的物体幻觉问题，提出并验证了一种新的视觉源幻觉机制，并通过对抗对比微调有效抑制幻觉

**💡 创新点**

创新点在于①首次系统识别并量化视觉源幻觉（跨模态对齐差异与注意力倒置）；②设计了对抗幻觉属性翻转（AHAF）作为诊断与数据生成工具；③提出了数据高效的对抗对比微调（ACFT），仅用0.9% COCO 数据即可在多模型上实现SOTA

**🔧 技术方法**

主要技术包括PGD对抗扰动、对抗属性翻转、对比学习损失、Cosine相似度与Grad‑CAM熵分析、Smoothing Grad‑CAM、跨模态编码器融合

**📊 数据集**

使用的数据集有COCO（0.9%子集用于训练）、POPE（三子集：Adversarial/Popular/Random）、MME（Existence子集及全套）、描述级幻觉基准（CHAIR、CCEval、AMBERA、ObjectHal）

**📈 对比分析**

与VCD、OPERA、Woodpecker、VTI、RLHF、DPO等多种推理、后处理及后训练方法对比，ACFT在POPE、MME、四个描述级基准上均实现SOTA或接近SOTA，典型提升范围为3–6%（例如LLaVA在POPE Adversarial子集从0.483提升至0.841），且保持或略提升MME整体表现

**⚠️ 局限性**

局限性包括：①主要针对短回答（“Yes/No”)场景，对长文本生成的语言偏差改进有限；②对抗训练对特定视觉任务敏感，可能在更广泛场景下的泛化需要进一步验证；③对视觉Encoder的依赖使得不同架构的迁移仍需额外调优

---

## 380. World Model-Guided Reinforcement Learning via Counterfactual User Engagement Simulation

**arXiv ID:** 2609.01067 | [PDF](https://arxiv.org/pdf/2609.01067v1)

**作者:** Ang Li `[一作]` (Chinese University of Hong Kong), Kam-Fai Wong `[通讯]` (Chinese University of Hong Kong)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了一种冻结的用户参与世界模型（UEWM）用于模拟用户反馈，并通过该模型为下游强化学习（WMG-RL）提供同一状态下的对比奖励，从而实现高效、低风险的用户推荐策略训练。

**💡 创新点**

创新点包括：① 将语言世界建模迁移到用户参与模拟，学习从历史推断个体动态；② 采用对抗性（counterfactual）模拟产生同一状态下多候选项的反馈，实现可比奖励；③ 将冻结的模拟器作为奖励源，使得小参数策略在零经验下也能达到甚至超过大型LLM；④ 通过中间的链式思维与奖励对齐提升模拟器的可信度和可迁移性。

**🔧 技术方法**

技术手段主要包括：大语言模型（Qwen3-8B）作为基础，采用自回归序列预测；链式思维（CoT）数据合成和GRPO奖励对齐；世界模型引导强化学习（WMG‑RL）框架；以及多种对比学习与奖励设计。

**📊 数据集**

使用的数据集：① 2.5M用户、636M条短视频交互日志（中文），构成训练、验证和测试；② 公开数据集（Amazon Books、Amazon Movies & TV、Google Local Reviews）用于跨域零样本转移评估；③ 公开评价指标与基准数据集用于下游推荐任务。

**📈 对比分析**

与基准方法的比较：在模拟器精度上，UEWM在多信号预测和评论生成上优于Qwen3-235B和Qwen3-8B，宏F1提升约7点；在跨域奖励转移上，UEWM在英语电商和本地服务数据集的评分准确率均比主流LLM高约8–10%；在下游推荐任务中，1.7B的WMG‑RL学生模型在Books、Movies和Google数据集上分别以41.91%、42.84%和37.95%的准确率，超过OpenAI o3、DeepSeek-R1等更大模型，以及SFT蒸馏和DeepSeek-GRM奖励基线。

**⚠️ 局限性**

局限性包括：① 未进行在线A/B或真实用户实验，模型表现仅在离线模拟环境下验证；② 模拟器错误可能随多轮交互累积；③ 训练日志不公开，复现难度较高；④ 仅使用文本模态，未覆盖视频视觉/音频信息；⑤ 未建模平台侧的曝光、内容供给或多用户互动机制。

---

## 381. Consistency Without Alignment: Item-Sensitive Language Models Indistinguishable From Random

**arXiv ID:** 2609.00576 | [PDF](https://arxiv.org/pdf/2609.00576v1)

**作者:** Cris Huynh `[一作]` `[通讯]` (Independent researcher), Cris Huynh (Independent researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在一个受《Deception: Murder in Hong Kong》启发的受限信号任务中，对七个语言模型（覆盖四个数量级）进行项敏感度（item‑sensitivity）评估，探究模型输出是否真正反映任务目标。

**💡 创新点**

创新点在于首次将项敏感度与独立随机参考（random baseline）对齐，证明项敏感度是必要但非充分条件，揭示了“无对齐的一致性”（consistency without alignment）的现象，并证明传统的自一致性/排列一致性指标不能单独证明任务完成。

**🔧 技术方法**

技术手段包括：基于 Shepard 通用归一化（exponential fit）计算选项适配度；使用后验（posterior）和“fit”最大化策略得到 salience 与 Bayes 极值；三种评分规则（PMI、unnormalised sum、per‑token mean）计算模型对选项的概率；引入 RSA 层、词义相似度基线（cosine、SWOW、ConceptNet）；以及对 1000 个手工筛选的 125 条冲突项进行统计分析。

**📊 数据集**

使用的数据集包括 THINGS/THINGSplus（概念与属性）、SPoSE（人类相似度空间）、SWOW（关联语料）、ConceptNet（关系路径）以及通过 200k 条候选生成的 1000 条实验项。

**📈 对比分析**

通过将模型在每条项上得分后计算“excess”（相对边际零点）和坐标（post_norm）与随机、salience、Bayes 三个参考点比较，结果显示所有模型均具备项敏感性，但 8/21 的模型在坐标上与随机无显著差异，5/21 的模型在描述目标时甚至比随机更差；大多数模型的坐标位于随机与 salience 之间，而基线的字面余弦相似度在多数情况下优于模型。

**⚠️ 局限性**

局限性包括：尚未收集人类实验数据；仅在单一“冲突”任务切片上进行分析；RSA 层对某些基线不适用；参考基准基于字面监听者模型，可能不完全反映人类推理；词汇集存在自然种类偏倚；以及模型评估仅使用三种评分规则，未涵盖更广泛的语言模型表现。

---

## 382. Can Scene Text Recognition Read Rare Compositions?

**arXiv ID:** 2609.00816 | [PDF](https://arxiv.org/pdf/2609.00816v1)

**作者:** Genpei Zhang `[一作]` `[通讯]` (University of Wisconsin--Madison), Genpei Zhang (University of Wisconsin--Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对现有的六大场景文字识别基准进行分层诊断，构建基于单词稀缺度和字符n-gram新颖度的5×5分桶，揭示聚合准确率隐藏的10–18个百分点的合成性错误；

**💡 创新点**

创新点在于：①提出无额外数据的稀有词/稀有三元组诊断框架；②系统证明误差源于自回归解码器的词汇先验，而非视觉缺陷；③证明仅通过从自回归到CTC解码的架构改动即可显著缩小错误；

**🔧 技术方法**

技术方法包括：分层bucket分析、层级线性探测、置信度误差分析、注意力熵偏移检测、LoRA适配器、温度/束搜索、外部语言模型重排序、以及置信路由的AR/CTC集成；

**📊 数据集**

使用的公开数据集：六大STR基准（IIIT5K、IC13、IC15、SVT、SVTP、CUTE80）、中文、日文、阿拉伯文文字识别数据集，全部采用MJSynth或Union14M-L作为参考语料；

**📈 对比分析**

在九个英语专用模型和十三组语言/模型组合（四种文字体系）上进行对比；聚合准确率维持在89–97%，但在q5/q5角落显示-10至-18点差距；仅SVTRv2（CTC解码）提升+2.5点可突破置信区间；LoRA-AR仅+1.3点；置信路由集成+0.6点；视觉编码器扩展对角落无影响；

**⚠️ 局限性**

局限性：诊断仅基于六大基准；+1.3点的非架构提升仅为经验平均值，未达到统计显著；未测试外部LM、字节级token化、检索增强或RL优化等潜在方法；仍需改进训练目标以彻底消除词汇先验误差。

---

## 383. CoVer: Conflict-Aware Claim Verification

**arXiv ID:** 2609.00508 | [PDF](https://arxiv.org/pdf/2609.00508v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CoVer框架解决社交媒体事实核查中的证据级与聚合级冲突，并构建ContraNote数据集。

**💡 创新点**

创新点在于三阶段流水线：证据模式归一化→事实共识→支持验证，能够在噪声多、对立信息频繁的环境下优先挑选高质量证据。

**🔧 技术方法**

使用大型语言模型（如GPT‑4o）进行结构化推理，结合分数加权的共识聚合和支持检验，以及对证据元数据的系统利用。

**📊 数据集**

主要数据集为自 Twitter Community Notes 采集的 ContraNote（冲突任务33,686条、优先级任务54,474条），并与 CONFACT、ConflictBank、ECON、FEVER 等公开数据集对比。

**📈 对比分析**

与八种基线（FacTool、FactCheckGPT、FIRE、Confact、ConflictRes、AVeriTeC、MADAM‑RAG、ClaimDecomp）在 7 组数据集上评测，CoVer 在冲突任务 86.0%/88.5% 及 0.5/0.5 平均 88.5% mac. F1，显著优于其他方法。

**⚠️ 局限性**

局限包括仅覆盖文本与元数据，缺少多模态处理；主要使用英文数据，低资源语言支持不足；依赖社区共识与算法评分，易受操纵；在真实检索环境下表现受检索质量影响；部分统计差异未达到显著性。

---

## 384. In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?

**arXiv ID:** 2609.00904 | [PDF](https://arxiv.org/pdf/2609.00904v1)

**作者:** Koshiro Aoki `[一作]` (Waseda University), Daisuke Kawahara `[通讯]` (Waseda University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在大型语言模型中测试内部表征可控性的实验框架——在上下文中进行神经反馈（ICN），要求模型在多轮中输出相同句子并根据内部激活的情感或道德评测得分进行调节；

**💡 创新点**

创新点在于：①引入“特权访问”要求，确保控制目标无法从提示文本外部推断；②将人类已验证的解码神经反馈实验模式移植到LLM，使用定量分数而非标签；③通过内部激活测量与自我报告双重评估，排除表面文字策略导致的“伪”控制；

**🔧 技术方法**

技术包括：利用预训练的线性情感/道德探针将隐藏层激活映射为概率；在对话中反馈分数并让模型尝试最大化分数；多轮实验收集激活变化与自我报告；统计检验（单侧配对t检验、McNemar检验）与效应量计算（Cohen's d/h）；

**📊 数据集**

使用三类数据集：Stanford Sentiment Treebank（情感）、ETHICS commonsense子集（道德可接受性）、True-False（事实真伪），每个数据集抽取256条中性句子作为固定输出；

**📈 对比分析**

与前人（Ji-An等）对照，采用更严格的特权访问设计，实验在四个开源模型（Llama-3.1-8B/70B，Qwen3-8B/32B）与五层深度上运行，共120个设置；结果显示仅有45/120显著偏向预期方向，且效应量均小于0.5，表明控制效果不稳定且实际影响有限；

**⚠️ 局限性**

局限性：仅测试情感、道德与事实三种特征；使用的是开源模型，未来更强模型可能表现不同；实验假设固定输出不阻碍内部调节，若内部表示由输出完全决定则控制失败；

---

## 385. RISA: Response Inspection and Selective Actions for Refusal Calibration in Large Language Models

**arXiv ID:** 2609.00790 | [PDF](https://arxiv.org/pdf/2609.00790v1)

**作者:** Wenhan Chang `[一作]` (Zhongnan University of Economics and Law), Wanlei Zhou `[通讯]` (City University of Macau)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种推理时的拒绝校准框架RISA，能够在不更新模型参数的前提下根据模型初始输出与提示的隐藏状态决定是否保留、强制拒绝或安全重生成答案

**💡 创新点**

创新点在于将规则匹配与隐藏状态线性探测器相结合，并通过阈值校准实现有选择的干预；同时采用非对称的纠正策略，即对有害非拒绝直接强制拒绝，对过度拒绝则通过安全引导重生成；再引入支持检测避免在无可靠依据时误干预

**🔧 技术方法**

使用固定规则、对隐藏层最终非填充标记的线性探测器（训练后校准为Platt比例），配合支持阈值和动作阈值；动作策略基于初始拒绝状态和探测得分；使用GPU推理时的两步生成（初始 + 可能的重生成）

**📊 数据集**

在三大模型（Qwen2.5‑3B、Llama‑3.2‑3B、Qwen3‑4B）上使用约1200条带标签提示进行探测器训练；400条安全评估提示用于阈值校准；另外使用OR‑Bench、HarmBench、Do‑Not‑Answer、XSTest、MMLU‑STEM与GSM8K进行评估

**📈 对比分析**

与基线模型以及AdaCD、SelfCD、CAA、RefusalDir等推理时方法对比；RISA在所有有害数据集上显著提升CRR（最高+0.44）且ASR保持低；在善意敏感集上ORR提升不一，但总体不明显恶化；在通用任务上GSM8K准确率无改，MMLU‑STEM略降≤0.008；生成延迟与重生成比例相对较低

**⚠️ 局限性**

局限在于对善意敏感提示的改善不稳定，过度拒绝的纠正依赖重生成的成功率；规则覆盖率有限，需要手工编写；阈值校准需额外数据，且在不同模型/任务间需单独调参；对极端多样化提示的泛化仍待验证

---

## 386. Predicting Program Exit Code with LLMs and Programming Language Semantics

**arXiv ID:** 2609.00579 | [PDF](https://arxiv.org/pdf/2609.00579v1)

**作者:** Lara Marinov `[一作]` (University of Texas at Austin), Milos Gligoric `[通讯]` (University of Texas at Austin)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了程序可执行性预测任务和对应数据集，评估大型语言模型是否能够按给定形式语义规则判断程序是否可执行以及违反的具体规则。

**💡 创新点**

首次将正式语义规则直接嵌入提示，让模型在不同语义形式与语义变换下对程序可执行性进行判断，并通过系统化的实验揭示LLM对形式语义的依赖程度。

**🔧 技术方法**

使用两种语义形式（小步骤与重写语义）、五类语义误差变换器、链式推理（Chain‑of‑Thought）以及多种开源LLM（Qwen、DeepSeek、Ministral）进行实验。

**📊 数据集**

基于LeetCode、HumanEval、CodeContests、MBPP等有效程序生成的数据集，加入五类无效程序，并划分为三组（人写、翻译、随机生成）以覆盖不同程序长度与结构复杂度。

**📈 对比分析**

通过对比不同语义形式、语义偏移（符号互换与替换）和程序分割的准确率，结果显示在短程序和原始语义下模型表现最好，但在语义偏移、长程序及错误类型上准确率显著下降，表明模型主要依赖预训练经验而非规则执行。

**⚠️ 局限性**

LLM无法稳定地系统应用给定的形式语义规则，易受预训练偏好影响，难以在更复杂或符号重定义的情况下保持高精度，导致整体推断性能有限。

---

## 387. Streaming4D: Accelerate 4D World Models via Block-wise Video Generation and Incremental Reconstruction

**arXiv ID:** 2609.00610 | [PDF](https://arxiv.org/pdf/2609.00610v1)

**作者:** Xiaoyan Liu `[一作]` (Chinese University of Hong Kong), Sifan Zhou `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Streaming4D，一种同步、块级自回归视频生成与增量 4D 重建的管线，实现实时低延迟的 4D 世界建模。

**💡 创新点**

创新点在于：①块级并行流水线，将视频块生成与前一块重建同步进行，显著降低端到端延迟；②通过自强（Self‑Forcing）策略与增量状态更新，实现长时序生成的时空一致性。

**🔧 技术方法**

采用 Self‑Forcing 风格自回归视频生成器、CUT3R 基于 Transformer 的增量 3D 重建后端，以及 Vision Transformer 进行特征提取，整体构建同步流水线。

**📊 数据集**

在 7‑Scenes 数据集上评估 3D 重建质量，并在多种分辨率（384×208 至 640×368）下测试生成与重建时延。

**📈 对比分析**

与传统离线分离式基线（先生成视频再重建）比较，Streaming4D 在 RTX 4090 上实现 1.21×–1.24× 的速度提升，同时在 Accuracy、Completeness、Normal Consistency 等指标上与 CUT3R 相当甚至更优。

**⚠️ 局限性**

主要限制包括：①仍未实现完整闭环反馈，可能导致长期推理的累计误差；②块大小选择需要平衡时序一致性与吞吐量，块化策略在极端动态场景下的鲁棒性尚待验证。

---

## 388. CRAD: Class-wise Reliability-Aware Distillation for Decentralized Heterogeneous Federated Learning

**arXiv ID:** 2609.00446 | [PDF](https://arxiv.org/pdf/2609.00446v1)

**作者:** Baraa Bilbeisi `[一作]` (University of Alabama at Birmingham), Qing Tian `[通讯]` (University of Alabama at Birmingham)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种完全去中心化、无公共数据、支持异构模型的联邦学习框架，利用同伴模型的软预测进行知识蒸馏。

**💡 创新点**

创新点在于Class-wise Reliability-Aware Distillation (CRAD)——按类过滤掉支持不足或与共识偏离的教师，再以教师对该类准确率的精度（inverse variance）加权聚合，实现按类、按可靠度的蒸馏。

**🔧 技术方法**

技术包括分布式同伴知识蒸馏、对每类准确率和样本数的统计向量共享、Agresti–Coull校正与逆方差加权、温度软化的KL蒸馏损失、无服务器全连接对等通信。

**📊 数据集**

实验使用CIFAR-10、CIFAR-100和PathMNIST三大图像分类数据集，在10/20/50/100客户端间按Dirichlet分布非IID划分，模型池包含ResNet-18、ResNet-18-Half和CNN-6。

**📈 对比分析**

与FedMD、FedGD、MSFKD、DFML、FedMKD等多种异构FL蒸馏基线及其统一/不确定性聚合对比，CRAD在所有数据集上均取得最高全局准确率（CIFAR-10 78.6%，CIFAR-100 48.2%，PathMNIST 89.4%），局部准确率亦保持稳健；并在通信、内存和时间成本上优于DFML等。

**⚠️ 局限性**

局限性包括仅在图像分类任务上验证，未探讨其它任务；统计向量泄露了局部类别比例，虽可通过差分隐私抑制但仍有风险；对全量客户端参与和模型规模的扩展性待进一步研究。

---

## 389. QILP-0: Constructing Observational Declarative Twins of Quantum Circuits

**arXiv ID:** 2609.01049 | [PDF](https://arxiv.org/pdf/2609.01049v1)

**作者:** Marina de la Cruz Echeandía `[一作]` (Universidad Internacional de la Rioja), Alfonso Ortega de la Puente `[通讯]` (Universidad de Oviedo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 QXymb 框架，并实现其第一版 QILP‑0，用观测量子电路的可观测数据在指定观察范围内构造精确的有限多值命题逻辑程序，形成所谓的“观测宣言式双子”。

**💡 创新点**

创新点包括：① 将观测范围、结构梯度、参考覆盖、几何映射、离散化与逻辑诱导整合成一个完整、可审计的流程；② 引入观测宣言式双子概念，明确逻辑完备与数值不确定性分离；③ 通过目标无关的 SVD 及原始可观测映射实现可解释的符号化；④ 提供完整的执行证书，记录所有可追溯性和审计信息。

**🔧 技术方法**

核心技术：
- 观测数据矩阵构造（Pauli 观测族）
- 结构梯度（支持级别）和固定参考覆盖
- 目标无关几何分析（SVD、残差投影）
- 原始可观测到符号映射与列选择（τ_dir、τ_col 阈值）
- 列级无监督离散化（Freedman‑Diaconis、容差分组）
- 逻辑诱导（LFIT / PRIDE）
- 结果认证与证书生成。

**📊 数据集**

数据集：
- Bars & Stripes：16‑100 个量子比特的产品与格点 CZ 嵌入，检验离散分支。
- Low‑Depth MNIST：14 708 个数字 0/1 的低深度量子变换前后实例，检验连续分支。

**📈 对比分析**

比较方法：对每个实验生成的 QILP‑0 逻辑程序，验证其在所有观测样本上的完备与一致性。结果显示：在所有报告的关系中，理论能完整、无冲突地重构数据，严格重构精度为 1。实验展示了两条分支（离散与连续）在相同认证语义下的可靠性。

**⚠️ 局限性**

局限性：
- 仅在预先声明的观测范围内保证精确，无法覆盖全量子系统的所有可观测。
- 需要可观测分辨率信息，缺失时采用默认数值容差，可能导致离散粒度不够。
- 依赖 Pauli 观测族，虽然框架可替换，但当前实现对其他观测族尚未验证。
- Twin‑可接受性检验要求目标一致，若观测中存在噪声或多值目标会导致不成功。
- 计算成本随支持级别和可观测数增长，尤其是 SVD 和离散化步骤。

---

## 390. CrossFeat: Bridging Imaging Modalities in Feature Descriptor Space

**arXiv ID:** 2609.00272 | [PDF](https://arxiv.org/pdf/2609.00272v1)

**作者:** Paul Schneider `[一作]` (Harvard Medical School), Nazim Haouchine `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出 CrossFeat 框架，使已有的单模态关键点描述子能够在多模态场景下匹配，通过学习描述子空间中的交叉函数实现。

**💡 创新点**

核心创新在于：①在描述子空间中分离几何与外观信息，仅对外观进行跨模态映射；②采用 FiLM‑调制的残差网络与变分自编码器实现可训练的非线性交叉函数；③无需重新训练原始描述子，直接在测试时实现跨模态匹配。

**🔧 技术方法**

使用编码器‑解码器结构（共享 MLP + 两个投影头）、FiLM 线性调制、残差 MLP、变分正则、对抗模态判别、对比损失（InfoNCE）等技术；整体为轻量级的 0.5‑0.6M 参数网络。

**📊 数据集**

在三大任务上评估：医学（ReMIND、BRATS、RESECT）、自动驾驶（EventScape、DELIVER）以及卫星影像（WHU‑OPT‑SAR、QXS‑SAROPT），共涉及 10 种模态对。

**📈 对比分析**

与多种稀疏匹配管线（SuperPoint+LightGlue、SIFT+NN、DISK、ALIKED）及密集匹配方法（MatchAnything、MINIMA‑LoFTR、MINIMA‑RoMa）对比，CrossFeat 在 SR、AUC 等指标上均显著提升（尤其在 SR@1 与 AUC@1 上领先），同时匹配数量更稳定、几何一致性更好；计算量约 5‑10 倍更快、参数量约 100 倍更少。

**⚠️ 局限性**

局限性：交叉网络需预先指定源/目标模态，无法在未知或连续模态间自适应；目前仅支持两模态之间的映射；跨模态映射依赖训练好的跨模态样本，缺乏对极端模态差异的泛化能力。

---

## 391. Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search

**arXiv ID:** 2609.00618 | [PDF](https://arxiv.org/pdf/2609.00618v1)

**作者:** Jincheng Zhang `[一作]` (Sichuan University), Yang Deng `[通讯]` (Singapore Management University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一个双节点蒙特卡洛树搜索框架DREAMS，利用结构化树状对话状态联合建模用户偏好挖掘与推荐；

**💡 创新点**

首次将偏好提问与推荐拆分为ELNode和EXNode两种节点，ELNode通过MCTS搜索动态更新JSON结构化偏好状态，EXNode利用LLM精炼检索查询，从而实现偏好演进的结构化建模与检索优化；

**🔧 技术方法**

使用结构化JSON对话状态、蒙特卡洛树搜索、LLM生成与解析、检索式查询精炼及经验增强(EA)加速；

**📊 数据集**

在Redial、OpenDialKG（电影、图书）等公开基准数据集上进行实验，并附加人类评估；

**📈 对比分析**

与InterCRS、MACRS、ChatCRS、PC-CRS、RA-CRS、SAPIENT-LLM、T-EPL、BARCOR、UniCRS等基线在R@1、SR、偏好提取错误率等指标上对比，DREAMS平均提升R@1+7.4%，SR+9%，偏好提取错误率显著降低，整体性能优于所有对比方法；

**⚠️ 局限性**

仅在GPT‑4o‑mini和Gemini‑2.5‑flash上评估，受LLM背后偏见与知识限制；未验证跨域（非电影/图书）通用性；评估主要依赖LLM评判，缺乏更广泛的真实用户实验。

---

## 392. Incremental Risk Assessment of Progressive Elder Financial Scams via Instruction-Tuned Small Language Models

**arXiv ID:** 2609.00005 | [PDF](https://arxiv.org/pdf/2609.00005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 393. Causal Evidentiary Governance for High-Risk Machine Learning Systems

**arXiv ID:** 2609.01040 | [PDF](https://arxiv.org/pdf/2609.01040v1)

**作者:** Samah Kareem `[一作]` (Isik University), Barış Çeliktaş `[通讯]` (Işik University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 Causal Evidentiary Governance (CEG) 框架，通过机构预先承诺的因果 DAG 对决策路径进行路径特定公平性审计，并用 Signed Decision‑Evidence Packets 和 Merkle 树提供可验证的决策证明。

**💡 创新点**

创新点在于将因果路径约束、加密证明与可验证账本相结合，推出 Causal Harm Rate（CHR）量化不合法因果路径的损害，并通过 DEP 与 Merkle 结构实现高效、可追溯的审计流程。

**🔧 技术方法**

使用了结构因果模型 (SCM)、路径特定归因方法（如 SHAP/LIME）、蒙特卡洛干预与 Bootstrap 置信区间、Merkle 树日志、Python+NumPy+DoWhy、XGBoost/Logistic/RandomForest 等技术。

**📊 数据集**

实验数据包括以巴勒斯坦金融监管机构公布的信用统计为基础的合成申请人数据（10,000 人）以及公开的德国信用数据集（1,000 人）。

**📈 对比分析**

通过与 Demographic Parity、Equalized Odds、Kusner 的无约束因果公平性指标比较，展示 CHR 在检测直接与代理歧视时的优势；实验性能显示：单线程 DEP 构建 122.1 ms/10k，32 核并行可达 4,120 DEP/s，内存 94.6 MiB，验证率 99.3–99.4%。

**⚠️ 局限性**

局限包括：因果 DAG 的准确性需要专家验证、对未测量混杂的敏感度有限、仅在许可化账本假设下适用、实验基于合成数据且仅限二分类、未覆盖多类别或非结构化输入。

---

## 394. ASSERT: Adaptive Stochastic Sampling for Robust Diffusion Models on Analog Compute-in-Memory Hardware

**arXiv ID:** 2609.00955 | [PDF](https://arxiv.org/pdf/2609.00955v1)

**作者:** Yuannuo Feng `[一作]` (Beihang University), Wang Kang `[通讯]` (Beihang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了扩散模型在计算内存(CIM)硬件上的推理过程，分析了硬件噪声对模型的影响，并提出了一种无需再训练的自适应随机采样方法ASSERT。

**💡 创新点**

创新点在于首次通过一阶轨迹递推分析揭示硬件噪声在早期推理步骤中更为敏感，并基于此设计了训练无关、在早期使用高随机性、后期渐进收敛的采样调度。

**🔧 技术方法**

采用的技术包括基于多块物理CIM芯片测量的噪声模型、DDIM扩散模型、对硬件噪声的矩阵向量乘法建模、以及自适应随机采样调度策略。

**📊 数据集**

实验使用了CIFAR-10、CelebA‑HQ、LSUN‑Church、LSUN‑Bedroom和Butterfly等公开图像数据集。

**📈 对比分析**

通过与确定性DDIM、STEP、线性衰减等采样策略比较，使用FID指标评估，ASSERT在高分辨率数据集上可实现最高2.58倍、在CIFAR‑10步数实验中可实现最高7.68倍的FID下降。

**⚠️ 局限性**

局限性包括：需要预先校准的噪声模型，随机数生成的额外开销；对不同扩散模型或更大规模数据集的泛化性尚未验证；以及在保持鲁棒性的同时可能对干净质量产生轻微退化。

---

## 395. Cyber-Physical Digital Factory Architecture as the Enabler of Disembodied Work

**arXiv ID:** 2609.00195 | [PDF](https://arxiv.org/pdf/2609.00195v1)

**作者:** Tero Kaarlela `[一作]` (University of Oulu), Souradeep Dutta `[通讯]` (University of British Columbia)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的数字工厂架构，支持无实体工作，通过同步数字孪生、层级云边AI、XR远程操作实现制造资源的离线协同与自主操作；

**💡 创新点**

创新点在于将传统数字孪生、工业物联网、人工智能与XR远程操作三者融合成连续同步的闭环，支持动态决策权转移（人机协同）并实现跨工件、跨设备的通用化架构；

**🔧 技术方法**

采用MQTT发布/订阅模型进行状态同步，云端和边缘的分层AI（调度、感知与控制），XR接口实现远程操控，安全监测模块提供碰撞与速度约束；

**📊 数据集**

主要使用本研究内部收集的CNC机床、机器人磨光机和电动车电池拆解工序的数据，未公开使用标准公开数据集；

**📈 对比分析**

通过三套独立实现验证：CNC加工（DT同步平均往返延迟563 ms）、机器人拆解（组件识别精准率98.1%/召回率96.5%，电线分割精准率74%/召回率82%）与磨光工序，实现远程遥控与自动化切换，性能满足工业现场可接受的实时性与可靠性；

**⚠️ 局限性**

局限在于各工序系统尚未实现完整的工厂级调度与资源协同；可扩展性、在多设备并发及网络抖动下的鲁棒性未实验验证；缺乏持久化工件历史数据存储与完整的安全生命周期管理。

---

## 396. A Dichotomy for Complex Boolean Holant with Binary Disequality

**arXiv ID:** 2609.00219 | [PDF](https://arxiv.org/pdf/2609.00219v1)

**作者:** Chenghua Liu `[一作]` (Chinese Academy of Sciences), Boning Meng `[通讯]` (University of Regensburg)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文证明了在存在二元不等式的情况下，复杂布尔Holant问题的复杂性二分法。可处理的情况通过明确的可判定标准进行表征。

**💡 创新点**

创新点在于为每个有限的代数复值布尔签名集提供了一个可判定的复杂性边界，并且该边界在二元不等式模型下得到了简化。

**🔧 技术方法**

使用了代数复杂性理论和全息算法的框架，结合了线性代数和图论的技术。

**📊 数据集**

使用了任意有限的代数复值布尔签名集，具体数据集未明确列出。

**📈 对比分析**

与现有方法的比较显示，本文提出的标准能够有效地识别可处理和NP难问题的边界，性能上优于以往的分类方法。

**⚠️ 局限性**

限制在于该研究主要集中在特定的代数复值签名集上，可能无法推广到更广泛的签名类型。

---

## 397. When the Algorithm Becomes the Brand Crisis: A Sociotechnical Theory of Distributed Responsibility and Accountable Transparency

**arXiv ID:** 2609.00510 | [PDF](https://arxiv.org/pdf/2609.00510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 398. Client-side transparent caching for remote ROOT data analysis

**arXiv ID:** 2609.00400 | [PDF](https://arxiv.org/pdf/2609.00400v1)

**作者:** Dmytro Kovalskyi `[一作]` (Massachusetts Institute of Technology), Christoph Paus `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个不需要服务器端改动的 XRootD 客户端缓存插件，缓存仅存储分析实际读取的字节，并可在本地重建为按分支顺序、ZSTD-1 重压缩的副本；通过实验验证其在多种硬件和网络环境下能显著加速高能物理数据分析。

**💡 创新点**

创新点在于：① 透明客户端缓存，无需部署服务器插件；② 仅缓存访问的数据，采用 4 KiB 页、CRC32C 校验，避免不必要的数据复制；③ 引入双层缓存：字节层（sparse）+ 复制层（branch‑major、ZSTD-1），既减少 I/O 也降低解压缩成本；④ 通过插件自带计数器和后台重建机制，实时监控缓存状态并隐藏构建开销。

**🔧 技术方法**

技术包括：XRootD 客户端插件、稀疏文件页缓存、CRC 校验、后台复制构建、分支重排与 ZSTD‑1 重压缩、ROOT RDataFrame/RNTuple 接口、自动化基准测试与性能计数。

**📊 数据集**

使用 CMS Open Data AGC tt̅ 分析的两套 NanoAOD 数据集：zlib‑1 压缩（1.8 TB，787 文件）和 LZMA‑9 压缩（2.4 TB，1456 文件），以及将其中一部分转换为 RNTuple 的副本。

**📈 对比分析**

通过在六台机器（桌面、工作站、MIT/CERN/FNAL 节点、云 VM、Mac mini）分别跑无缓存、冷缓存、热字节缓存、热复制缓存四种模式，记录总耗时、填充开销、复制覆盖率、IOPS、带宽和 CPU 指令/IPC。结果显示：填充开销 ≤1.2%，热字节缓存相较直接读取提升 1.6–8.8×，热复制缓存提升 2.1–15.7×；在远距离或高负载源上加速最大；复制层在 LZMA 场景下将解压缩成本降低约 70%，整体加速 3–4 倍。

**⚠️ 局限性**

局限性：需要足够快的本地存储（NVMe/SATA），在 I/O 受限的云块卷或低速磁盘上缓存反而慢于直接读取；复制层会把磁盘占用增至 1.16–1.52 倍；首次填充在源负载极高时可能略慢；插件目前仅支持 XRootD 5.x，6.x 正在开发；主要适用于迭代式分析，对一次性大批量处理的收益有限。

---

## 399. Residual Sparsification via Output Importance for Compressing Mixture-of-Experts LLMs

**arXiv ID:** 2609.00575 | [PDF](https://arxiv.org/pdf/2609.00575v1)

**作者:** Seungwoo Jung `[一作]` (Korea University), Gyeongsik Yang `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的残差稀疏化方法PARSER，专门用于压缩Mixture-of-Experts（MoE）大语言模型，目标是降低显存占用同时最小化专家输出误差。

**💡 创新点**

创新点在于将压缩目标从逐矩阵误差最小化转向输出重要性（output importance）评估，依据隐藏表示对最终专家输出的影响来选择要压缩的维度，并采用全局池化来避免专家间重要性失衡。

**🔧 技术方法**

核心技术包括：共享基矩阵与专家残差分解；输出重要性度量H(h_j)的推导与估算；基于校准数据集𝒟的经验估算；全局维度选择与行列压缩；以及结合TSVD/UP两种稀疏化实现。

**📊 数据集**

使用的校准数据集为512个Dolly‑15K序列（约1M个token），同时在评测中使用Qwen1.5‑MoE‑A2.7B和DeepSeek‑V2‑Lite两款MoE LLM，在七个零样本推理任务（ARC‑Easy/Challenge、WinoGrande、HellaSwag、PIQA、OpenBookQA、MMLU）以及额外的WikiText和IFEval做进一步验证。

**📈 对比分析**

与四个SOTA基线（MoE‑I2、HC‑SMoE、D2MoE、ResMoE）对比，PARSER在90%压缩比下平均缩短36%（Qwen）/33.9%（DeepSeek）显存，且准确率误差仅比未压缩模型低1.41×/1.44×；在不同压缩比例、显存/准确率曲线以及不同校准集设置下均表现出最高或接近最高的准确率，且压缩时延与推理吞吐率仅略高于最优基线，实际推理速度仍最高。

**⚠️ 局限性**

局限性包括：对校准数据集𝒟的依赖，尽管实验表明鲁棒性有限；只评估单维度重要性，未考虑维度间交互；仅针对推理压缩，未覆盖训练或微调情境；以及在高度专业化领域（如医疗、代码生成）中可能需要更细粒度的校准与目标函数。

---

## 400. Hidden relationships in a document-derived property graph: top-k chunk embeddings and inverse-distance weighting over a dynamically evolving ontology

**arXiv ID:** 2609.00387 | [PDF](https://arxiv.org/pdf/2609.00387v1)

**作者:** Bilge Kaan Karamete `[一作]` (Babel Street), Hunter Casten `[通讯]` (Babel Street)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种从文本构建知识图谱的方法，通过对文档进行段落分块和嵌入，利用k近邻查询来恢复文本中未明确陈述的关系。

**💡 创新点**

创新点在于提出了一种纯粹的加法性第二遍处理方法，能够在不修改提取结果的情况下，恢复文本中隐含的关系，并且实现了增量构建和动态本体。

**🔧 技术方法**

使用了Shepard逆距离加权、文本嵌入和k近邻查询等技术。

**📊 数据集**

使用了来自多个文档的段落数据集，具体数据集未详细说明。

**📈 对比分析**

与传统方法相比，本文的方法在处理速度上快25倍，并且在存储效率上，768维嵌入与3072维参考的边一致性达到92%。

**⚠️ 局限性**

限制在于未能提供恢复关系的精确度和召回率的测量，因为缺乏标注的真实关系集。

---

## 401. Corporate Loyalty: Some AI Systems Differentially Downplay their Creators' Controversies

**arXiv ID:** 2609.00373 | [PDF](https://arxiv.org/pdf/2609.00373v1)

**作者:** Lennart Finke `[一作]` (ETH Zürich), Stephen Casper `[通讯]` (MIT)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对七大AI公司（xAI、DeepSeek、Anthropic、OpenAI、Alibaba、Meta、Google）下的21个语言模型进行实验，系统评估其在讨论各自公司负面新闻时的偏向性；

**💡 创新点**

首次用大规模、预注册的统计方法（置换检验+线性ANOVA）验证并量化模型在自家公司争议上的偏向性，揭示了部分模型对公司形象的“隐性保护”行为；

**🔧 技术方法**

采用置换法非参数检验、线性交互模型、Holm-Bonferroni校正，以及多模板多模型的评估框架，结合内部评判模型对回复进行打分；

**📊 数据集**

构建了206条负面新闻数据集，来源于NYT和Hacker News，按相关性、政治争议度筛选，并人工补充遗漏的事件；

**📈 对比分析**

通过估计交互项θ_c并检验其显著性，对比同一公司与其他公司模型的正面评分差异；结果显示xAI、DeepSeek、Anthropic、OpenAI模型显著倾向正面，而Alibaba、Meta、Google无此偏向；

**⚠️ 局限性**

局限包括评估意识可能影响模型表现、新闻样本量有限、模型选择与代表性、以及对模型内部机制的解释不足。

---

## 402. Can MCP Clients Decide What to Do After Failure? A Result-Only Actionability Audit

**arXiv ID:** 2609.00072 | [PDF](https://arxiv.org/pdf/2609.00072v1)

**作者:** Rishabh Mehan `[一作]` `[通讯]`, Rishabh Mehan

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在仅拥有MCP完成错误结果时，确定性软件能多大程度上推断出可安全的后续动作，提出六维可操作性分析框架并在21个失败案例上评估。

**💡 创新点**

引入非累积的六维可操作性指标、可审核的记录级分析方法以及演示性失效闭合原型，展示仅凭错误结果的恢复可行性边界。

**🔧 技术方法**

结合MCP规范、Python SDK标准化结果、人工编码、Lexical扫描、Llama 3.2/Qwen 3.5模型的策略识别实验，以及失效闭合执行原型。

**📊 数据集**

采样自2026年8月19日的MCP注册表中54个仓库，产生10个可访问端点共21个诱发失败记录。

**📈 对比分析**

通过将模型在四种条件（归一化、代码、结构化、说明）下的策略匹配率与手工构造的确定性原型对比，结果显示结构化JSON可达95%匹配，说明模式仅提供指令而非真正恢复能力。

**⚠️ 局限性**

样本规模有限且缺乏速率限制、上游故障、超时等典型失效类型；人工单人编码缺乏双人验证；仅评估小模型策略识别，未覆盖实际部署代理的恢复表现。

---

## 403. PersianAnonymizer: Evaluating LLM-Labeled Training for Efficient NER-based Anonymization in Persian

**arXiv ID:** 2609.00958 | [PDF](https://arxiv.org/pdf/2609.00958v1)

**作者:** Mohammad Hossein Shalchian `[一作]`, Amir Mahdi Sadeghzadeh `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文基于组织内部的 Persian 聊天记录，使用三种 instruction‑tuned LLM（DeepSeek、Qwen、GPT‑OSS）对 265,000 条用户消息进行自动标注，随后利用这些银标注训练四个基于 XLM‑RoBERTa 的紧凑型 NER 模型，用于对 PII/PHI 的脱敏；通过对标注质量、模型可学习性、跨 LLM 一致性以及推理吞吐量进行系统评估。

**💡 创新点**

①首次对比多种 LLM 在 Persian 脱敏 NER 任务中产生银标注的可学习性；②证明 GPT‑OSS 的零样本标注能生成最易于学习且覆盖最广的监督数据；③展示用轻量级 NER 替代昂贵 LLM 推理可实现秒级脱敏，兼顾隐私与成本。

**🔧 技术方法**

主要技术包括：①多 LLM 统一的 JSON 输出约定与解析；②BIO 标注转换与模糊匹配对齐；③基于 XLM‑RoBERTa Large 的 token‑classification 微调；④交叉验证与宏 F1 / Label Coverage Recall（LCR）评估；⑤与公开 PEYMA 数据集的跨域比较。

**📊 数据集**

使用 265k 条 Persian 聊天记录（225k 训练，40k 测试）生成的四套银标注集；此外还利用公开的 PEYMA Benchmark 进行跨域验证。

**📈 对比分析**

在宏 F1 与 LCR 上，GPT‑OSS‑ZeroShot 训练的 NER 取得最高宏 F1 ≈0.851、LCR≈90%；Qwen 系列略低，DeepSeek 性能最差。跨 LLM 标注的 token 级重叠率约 13.6%，三者共识率 5.46%；在 PEYMA 上，GPT‑OSS NER 在 COST/ORG 等数值实体上表现出较强的迁移能力。推理效率方面，单 GPU（RTX 3090）可在约 2 min 内完成 40k 条消息的标注，而原始 LLM 需要数十分钟。

**⚠️ 局限性**

主要局限：①对罕见实体（如 ORG、LOC）召回仍不足；②标注噪声来自 LLM，缺乏大规模人工核对；③BIO 标记仅评估覆盖度，未衡量类型一致性；④数据来源单一，可能对不同行业或语言变体迁移效果不佳。

---

## 404. On the Design Fundamentals of Pixel Text Representation Learning

**arXiv ID:** 2609.01147 | [PDF](https://arxiv.org/pdf/2609.01147v1)

**作者:** Chaohao Yuan `[一作]` (Chinese University of Hong Kong), Chenghao Xiao `[通讯]` (Shanghai University of Finance and Economics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Pixel‑Linguist‑II，一种统一的视觉编码器，直接从像素学习文本表示，支持高分辨率文档、跨语言文本理解，并在视觉文本检索和STS任务上达到SOTA；

**💡 创新点**

四大设计原则：可变分辨率与字体大小作为空间代理、天然多模态锚定、布局感知渲染抑制视觉捷径、双阶段多语言训练课程；以及Native‑resolution ViT和即时渲染引擎；

**🔧 技术方法**

采用Native‑resolution Vision Transformer (NaViT)、on‑the‑fly多语言字体渲染、统一对比学习目标、两个阶段的多语言预训练与语义微调、DeepSpeed ZeRO 2 分布式训练；

**📊 数据集**

训练数据共计约2.8亿样本，包括26M自然图像–文本对（LAION‑2B）、62M Wikipedia渲染文本对、26M高质量语义文本对，辅以多语言文本与视觉检索基准；

**📈 对比分析**

在Visual STS、ViDoRe以及跨语言与多语言Visual STS基准上均超越CLIP、SigLIP、CLIPPO、EVA等基线，SOTA提升约15–20%；在MLLM下游任务提升约2.75%；在视觉令牌压缩下仍能保持80%压缩仍优于CLIP；

**⚠️ 局限性**

局限性在于整体数据规模仍低于千亿级CLIP基线，对图形化科学图表性能不足；渲染文本噪声覆盖不完整，缺乏扫描、手写、模糊等真实世界噪声。

---

## 405. The zbMATH Open Knowledge Graph: Tracing Centuries of Mathematical Research

**arXiv ID:** 2609.00969 | [PDF](https://arxiv.org/pdf/2609.00969v1)

**作者:** Yuni Susanti `[一作]` (FIZ Karlsruhe), Moritz Schubotz `[通讯]` (FIZ Karlsruhe)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了跨250年数学研究的RDF知识图谱，集成专家评述、关键词、MSC分类、软件引用及去重作者信息；

**💡 创新点**

将专家精细语义内容与长时间覆盖相结合，提供比传统仅有元数据与引用网络更丰富的历史学术探索能力；

**🔧 技术方法**

基于Semantic Web标准（schema.org、cito、skos、dcterms等）的RDF/OWL本体，使用Python/RDFLib、Apache Jena生成、验证，并提供SPARQL端点；

**📊 数据集**

来源于zbMATH Open平台的OAI-PMH API，覆盖400万+论文、112万+学者、300万+评述、30k+软件引用、10.5M标识符（含2.4M DOI）；

**📈 对比分析**

通过语义一致性检查、结构一致性验证和45个能力问题（CQs）评估，95.6%案例至少部分支持、86.7%完全支持；SPARQL查询无语法错误，TripleStore均可成功载入；

**⚠️ 局限性**

仅覆盖zbMATH数据，缺少外部链接（如Wikidata、软件专用元数据），部分记录缺失标题/作者导致结构违规，无法提供软件编程语言等细粒度信息。

---

## 406. Poisson-Gamma Dynamical Systems with Time-varying Transition Dynamics

**arXiv ID:** 2609.00896 | [PDF](https://arxiv.org/pdf/2609.00896v1)

**作者:** Jiahao Wang `[一作]` (University of Arizona), Sikun Yang `[通讯]` (Great Bay University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种具有时变转移矩阵的泊松-伽马动力学系统（TV-PGDS），通过三种Dirichlet马尔可夫链（Dir-Dir、Dir-Gam-Dir、PR-Gam-Dir）实现转移矩阵随时间演化，并给出完全共轭的Gibbs采样算法。

**💡 创新点**

创新点在于将Dirichlet马尔可夫链嵌入PGDS中以捕捉非平稳的依赖结构，首次实现对转移动力学的时间变动建模，并通过随机化伽马-Dirichlet链获得更稀疏、更灵活的动态表示。

**🔧 技术方法**

采用了贝叶斯推断框架，利用Dirichlet-多项式-贝塔、Poisson-多项式、负二项-CRT、Bessel分布以及Shifted Confluent Hypergeometric分布等数据增强技术，实现了高效的共轭Gibbs采样。

**📊 数据集**

实验使用四个真实世界计数序列数据集：ICEWS、NIPS、USEI（美国地震强度）以及COVID‑19每日死亡人数。

**📈 对比分析**

与GP‑DPFA、PGDS、GMC‑RATE、GMC‑HIER、BGAR等基线模型在平滑（smoothing）和预测（forecasting）任务中对比，采用MAE和MRE评估，TV‑PGDS在大多数数据集上均取得更低的误差，表明其在捕捉时间变动依赖方面更具优势。

**⚠️ 局限性**

局限性包括需预先指定子区间长度（I、M）且缺乏自适应变点检测，可能导致过拟合或欠拟合；此外，Dirichlet马尔可夫链的非共轭性导致部分推断步骤仍需额外的变分或近似方法。

---

## 407. Overfitting Mitigation via Singular Value Decomposition in Minimum Bayes Risk Decoding

**arXiv ID:** 2609.01135 | [PDF](https://arxiv.org/pdf/2609.01135v1)

**作者:** Riza Setiawan Soetedjo `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SVD-MBR方法，利用奇异值分解对MBR中的配对效用矩阵进行去噪，降低指标过拟合；

**💡 创新点**

创新点在于将低秩近似应用于MBR，分离真实共识与指标噪声，首次系统验证了对不同指标和任务的泛化效果；

**🔧 技术方法**

技术主要包括：Minimum Bayes Risk解码、配对效用矩阵构造、奇异值分解（SVD）与低秩截断、z-score标准化；

**📊 数据集**

数据集包括WMT22 En→De、De→En以及XSum摘要数据；

**📈 对比分析**

与传统MBR、MAP、PMBR以及Model-based MBR比较，SVD-MBR在大多数神经指标（COMET、BLEURT、BERTScore）上显著提升离目标指标的平均z-score（Z̅_other），并在大多数指标上保持或略低的目标指标分数，表明成功抑制了过拟合；

**⚠️ 局限性**

局限性包括：SVD增加计算复杂度（O(min(N²M,NM²))），仅在ε-sampling生成候选时评估，未探讨人类评价与其他生成/降噪方法的组合。

---

## 408. Candidate-Expanding Routing with Permutation-Stabilized Experts for Mixed-Format Medical VQA

**arXiv ID:** 2609.00959 | [PDF](https://arxiv.org/pdf/2609.00959v1)

**作者:** Hai-Dang Nguyen `[一作]` (VinUniversity), Huy-Hieu Pham `[通讯]` (VinUniversity)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种候选扩展路由框架，将答案文本记忆、置换稳定的视觉语言专家和稀疏候选扩展路由器结合，用于处理多项选择题和开放式医学视觉问答。

**💡 创新点**

创新点包括：①将专家的前两名得分作为可路由候选，显著提升匹配准确率；②采用置换稳定的专家方案消除选项符号/位置对答案的影响；③引入严格的 JSON 输出合同，保证开放式答案的可机器解析性。

**🔧 技术方法**

使用的技术包括答案文本记忆（基于检索的向量相似度）、循环置换评估的视觉语言专家（如 Qwen3.6-27B + LoRA）、一对多 L1 逻辑回归路由器以及严格的 JSON 解析与重试机制。

**📊 数据集**

主要数据集为 Med-CMR，包含 20,653 个医学视觉问答对（11,592 MCQ 与 2,716 开放式），以及内部划分的 1,403 个 MCQ 用于评估。

**📈 对比分析**

通过对内部回顾性数据的基线对比，候选扩展路由器将匹配二元路由准确率从 88.95% 提升至 91.73%（+2.78pp，95% CI 1.57–3.99），最终回顾性结果达 92.23%；开放式问题在所有 475 个实例上均达到 100% 的 JSON 合法性。

**⚠️ 局限性**

局限性包括：评估为回顾性内部实验，未进行独立盲测；缺少患者/研究 ID 以验证数据脱敏；开放式答案仅保证格式合法，未验证临床真实性或可信度。

---

## 409. ADGNet: Asymmetric Dual-text Guided Network for Infrared Small Target Detection

**arXiv ID:** 2609.00853 | [PDF](https://arxiv.org/pdf/2609.00853v1)

**作者:** Tongtong Wang `[一作]` (Shandong University), Weili Guan `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种名为 ADGNet 的红外小目标检测网络，利用不对称双文本提示、双分支交互和自适应特征聚合模块，显著提升了目标与背景的分离能力。

**💡 创新点**

创新点包括：①引入不对称双文本提示（抽象目标提示 + 详细背景提示）以消除语义噪声；②设计 ADBI 双分支交互来独立完成目标定位与背景抑制，解决单模态冲突；③构建 AITIR 数据集，为多模态红外检测提供高质量文本注释。

**🔧 技术方法**

采用 CLIP 预训练文本编码器、跨模态注意力机制、双分支（TL 与 BS）交互、动态特征聚合（AFA）以及 SoftIoU 损失，实现跨模态信息的有效融合。

**📊 数据集**

在 AITIR 数据集上进行训练和评估，该数据集在 IRSTD-1K、NUDT‑SIRST 与 SIRST 三大公开数据集上添加了不对称文本注释。

**📈 对比分析**

与 21 种现有方法（传统、深度学习及多模态）对比，ADGNet 在 IoU、P_d、F_a 等指标上显著优于 SAIST 等，特别是在 IRSTD-1K 上 IoU 达到 72.38% 并将 F_a 降至 4.10，表现出最优的检测性能。

**⚠️ 局限性**

主要局限在于仍需依赖 CLIP 预训练文本编码器，导致参数量较大；对极其稀疏或极暗目标的鲁棒性在极端背景下仍有提升空间，需要进一步优化跨模态融合效率。

---

## 410. Shared-Memory Range-Tiled CDF Sort for Small-Range Integer Keys on GPUs

**arXiv ID:** 2609.00843 | [PDF](https://arxiv.org/pdf/2609.00843v1)

**作者:** Kento Ando `[一作]` (Hosei University), Koichi Wada `[通讯]` (Hosei University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在已知整数范围内的不稳定整数排序，提出了 Range‑Tiled CDF Sort (RT‑CDF) 算法。

**💡 创新点**

创新点在于将整个值域拆分成可放入共享内存的若干小块（tile），在每个块内部构建局部 CDF 并直接生成输出，避免全局前缀最大化。

**🔧 技术方法**

采用了基于计数排序的直方图构造、前缀和求 CDF、共享内存并行累加、二分查找输出生成等 GPU 技术。

**📊 数据集**

使用合成数据集：均匀分布、正态分布以及全相同值，范围 R 从 2^7 到 2^18，输入规模 n 从 10^6 到 10^9。

**📈 对比分析**

与 NVIDIA CUB、Ref‑H‑P、Kolonias 等实现比较；在小到中等范围内 RT‑CDF 最高可提升 4.39 倍；当 R≤2^17 时性能领先，R=2^18 则不再占优。

**⚠️ 局限性**

限制在于范围增大时需要多次扫描输入，导致 O(mn) 复杂度；共享内存占用随 tile 宽度增大而增长，降低块数与占用，导致 R≥2^18 时性能显著下降。

---

## 411. BrainDiff: Longitudinal Report Generation for Multimodal Brain MRI

**arXiv ID:** 2609.00593 | [PDF](https://arxiv.org/pdf/2609.00593v1)

**作者:** Krish Patel `[一作]` (Johns Hopkins University), Peirong Liu `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

开发并评估BrainDiff，一种用于脑MRI长时序报告生成的视觉-语言系统。

**💡 创新点**

首次针对脑MRI实现跨时间点的报告比较，并通过对抗性基线丢弃和分阶段课程两种无参数干预显著提升图像依赖。

**🔧 技术方法**

结合冻结的NeuroVFM 3D ViT-B编码器、Qwen3-14B解码器、Perceiver‑IO连接、LoRA微调、对抗性对齐、先前报告随机丢弃和分阶段训练。

**📊 数据集**

利用MR‑RATE、OASIS‑3、BraTS、ISLES‑22等多源脑MRI数据，生成约44k对长时序样本，并在BIND多中心队列进行外部验证。

**📈 对比分析**

与前沿通用模型(Opus5、GPT‑5.6)及单时间点神经影像VLM NeuroVFM对照，BrainDiff在内部测试的RadGraph‑XL F1达到0.3837，外部BIND 0.3506，远优于基线并保持91%性能。

**⚠️ 局限性**

差异编码器在冻结的NeuroVFM 16mm分辨率下无法捕捉细微病变变化；且使用LLM合成的对比报告可能偏向文本重叠，限制了真实临床转化。

---

## 412. Dr. Claw: An AI Scientist Workspace for Vibe Research

**arXiv ID:** 2609.00365 | [PDF](https://arxiv.org/pdf/2609.00365v1)

**作者:** Dingjie Song `[一作]` (Lehigh University), Lichao Sun `[通讯]` (Lehigh University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个名为 IRIS 的开源工作空间，包装命令行编码代理，提供可追溯、可恢复的人工干预研究流程。

**💡 创新点**

创新点在于将现有编码代理与持久化状态对象、技能库以及多执行器编排结合，形成可控、可审计的研究工作循环。

**🔧 技术方法**

使用了任务图、工件存储、决策日志、执行轨迹等状态对象；多层架构（交互、编排、执行）；技能库与权限策略；以及 Claude Code、Gemini CLI 等命令行编码代理。

**📊 数据集**

在医学领域使用 Derm7pt 皮肤病变分类数据集（包括黑色素瘤和痣分类）以及临床笔记风险基线数据集进行评估。

**📈 对比分析**

通过在保持后端执行器不变的前提下，将 IRIS 与裸 CLI 代理进行对比，评价研究完整性（21 个最佳实践要素）、审计性和失败恢复；IRIS 在完整性上相当或略优，提供了完整的审计轨迹并支持非破坏性恢复，但存在一定的运行时开销。

**⚠️ 局限性**

实验规模有限，缺乏对单个组件的消融验证；完整性评估仅为覆盖率，未验证科学正确性；仅在医学领域测试，泛化性未知；未与最先进的编排框架做直接比较；对开放式研究可能带来结构性阻力。

---

## 413. Two-State Max-Plus Comparison Is Decidable

**arXiv ID:** 2609.00678 | [PDF](https://arxiv.org/pdf/2609.00678v1)

**作者:** Keigo Oka `[一作]` `[通讯]` (Google), Keigo Oka (Google)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了任意有限max-plus自动机与最多两状态的max-plus自动机之间的比较、等价与正性问题是可判定的；

**💡 创新点**

首次突破了Daviaud等人提出的两状态至552状态区间未决的难题，利用一维投影规范化与精确计数器模拟实现了两状态右侧自动机的可判定性；

**🔧 技术方法**

核心技术包括：两状态投影几何的三种尾行为（传播、遗忘、读取-遗忘）分类、将两状态自动机转换为精确的单计数器词法转换器、利用上下文无关语言的Parikh线性性与谓词算术实现比较判定；

**📊 数据集**

未使用实验数据集，全部证明为形式化理论推导；

**📈 对比分析**

比较方法通过构造一个与左侧自动机同步的单计数器转换器，将比较转化为在上下文无关语言的Parikh集合上求解谓词算术存在性问题；由于未给出复杂度分析，性能未量化；

**⚠️ 局限性**

局限性在于：只适用于右侧自动机最多两状态，对更大状态数的可判定性仍未解决，且算法的具体时间复杂度尚未确定。

---

## 414. Removable and Irreducible: A Token-Cost Ledger for the Multilingual Tokenization Tax

**arXiv ID:** 2609.00378 | [PDF](https://arxiv.org/pdf/2609.00378v1)

**作者:** Madhulatha Mandarapu `[一作]` (VaidhyaMegha Private Limited), Sandeep Kunkunuru `[通讯]` (VaidhyaMegha Private Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个token成本记账表，拆分多语言token化税为可移除的编码冗余、残余编码空隙、内在内容和不可约的正字法成本。

**💡 创新点**

首次将可移除编码冗余与不可约正字法成本统一在同一记账框架；在真实并行语料上实证可移除率，并通过构造熵界限编码器证明可移除率可高达98%；揭示token税在Transformer计算中被二次放大的效应。

**🔧 技术方法**

利用信息理论（Shannon熵、KL散度、Huffman、LZMA压缩）、Transformer计算复杂度模型（N²注意力）、脚本匹配的BPE训练以及单命令复现工具。

**📊 数据集**

使用FLORES‑200并行语料（8种语言，1,012句子）以及Zipf分布的控制源。

**📈 对比分析**

通过对照英文基准正则化token长度，计算可移除比例ρ；在Indic语言中，脚本匹配BPE可移除约64%的token税，内在内容差异<6%；构造编码器将冗余降至98%；token税对计算造成最高79×注意力负载。

**⚠️ 局限性**

局限性包括：只关注计算与内存账务，不评估模型质量；匹配BPE训练样本有限，低估可移除率；正字法方向未确定；信息基线为LZMA上限；仅覆盖8种语言和文本，未测量多模态成本。

---

## 415. I-CARE: Analysis of interference-related phenomena in a controllable, diverse and representative unlearning setting for text-to-image models

**arXiv ID:** 2609.00003 | [PDF](https://arxiv.org/pdf/2609.00003v1)

**作者:** Leonardo Santiago Benitez Pereira `[一作]`, Luis Herranz Arribas `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种名为 I-CARE 的方法论，用于系统化研究文本到图像模型中的“干扰”现象，即在模型忘记某一概念时，其他语义相关概念的意外退化。

**💡 创新点**

创新点包括：①将干扰建模为一类可量化的研究对象；②设计了一套标准化的任务、指标和报告模板（RT），实现实验结果的可重复、可比；③提供了完整的开源实现与可视化界面（Forgety），降低研究门槛。

**🔧 技术方法**

技术细节：基于扩散模型（Stable Diffusion 1.4），实现了三种无学习方法（uce、spare、munba）；使用多种相似度计算（Clip、Dino、Act、Jaccard）和干扰度量（ΔClip、ΔBrisque、ΔDino、RMSE、SSIM 等）进行评估。

**📊 数据集**

数据集：人脸（LFW）→人；狗品种（AtharvaTaras Dog Breeds Dataset）→犬种；场景（SUN Attributes Dataset）→场景；每个任务均包含 100 条实体，并对属性进行平衡处理。

**📈 对比分析**

通过全互对（all‑vs‑all）计算 RT，比较三种无学习方法在干扰量、可预测性和公平性上的表现。实验显示 munba 产生最高干扰，uce 产生最少干扰；spare 的干扰虽高，但更易被相似度预测；在平衡后，方法间的差异仍显著。

**⚠️ 局限性**

局限性：①干扰指标噪声大，难以预测；②相似度与干扰之间的相关性不稳健，且多数情况下两者并不直接对应；③实验中种子不一致导致部分结果噪声增大；④仅测试了三种方法和三个任务，结果可能不具普适性；⑤缺乏对“构造性干扰”的深入解释。

---

## 416. CRSF: Collusion-Resilient Privacy-Preserving Sensor Fusion with Byzantine-Robust Participation

**arXiv ID:** 2609.01096 | [PDF](https://arxiv.org/pdf/2609.01096v1)

**作者:** Chao Yin `[一作]` (Vrije Universiteit Amsterdam), Chenglu Jin `[通讯]` (Centrum Wiskunde & Informatica)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在三方架构下实现了抵御传感器与服务器合谋和拜占庭攻击的隐私保护传感器融合协议 CRSF，保证了传感器数据隐私、结果正确性与系统活性。

**💡 创新点**

主要创新在于结合 PBFT 同意阶段与基于状态的标签释放，并采用字节级 (3,4) Shamir 共享，阻止了服务器单方操纵参与和传感器-服务器联盟重构补全标签的攻击。

**🔧 技术方法**

使用 Yao garbled circuits、FreeXOR 优化、PBFT 共识、Shamir 共享、签名、哈希和伪随机函数等多种密码学技术。

**📊 数据集**

使用软件仿真传感器产生的 16 位随机输入值进行评估，未使用真实物理传感器数据集。

**📈 对比分析**

与 PG 基线对比，CRSF 在 3~261 传感器规模下在线时延约为 PG 的 1.5-2 倍，通信量主要集中在 PBFT 同意和共享重构阶段；在 Byzantine 情况下，因少数传感器缺失和服务器失效，时延反而略低。

**⚠️ 局限性**

局限性包括：相对于单服务器方案，在线时延和通信量均有显著提升；需要四台服务器实现 PBFT，系统部署复杂；并且在高度恶意环境下仍可能因同步问题导致协议停滞。

---

## 417. Design and Implementation of a Kalman Filter-Infused Algorithm for Tilt Estimation

**arXiv ID:** 2609.00730 | [PDF](https://arxiv.org/pdf/2609.00730v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 418. An Approach to Asynchronous Unsourced Random Access

**arXiv ID:** 2609.00236 | [PDF](https://arxiv.org/pdf/2609.00236v1)

**作者:** Alireza Karami `[一作]` (Dalhousie University), Dmitry Trukhachev `[通讯]` (Dalhousie University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df`

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

## 419. Data-Driven Persona-Conditioned Agents for A/B Test Simulation

**arXiv ID:** 2609.01038 | [PDF](https://arxiv.org/pdf/2609.01038v1)

**作者:** Ziyad Benomar `[一作]` (Amazon), Saab Mansour `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM与基于真实用户行为数据构建的人格代理进行A/B测试结果模拟，旨在快速预筛实验方案并降低成本

**💡 创新点**

1）将A/B测试转化为结构化问答任务；2）采用数据驱动、行为深度与人口多样性兼顾的人格构建；3）系统评估问答格式、数据来源、深度-多样性权衡和子样本化策略

**🔧 技术方法**

Claude Sonnet 4.5等大型语言模型、LLM处理行为信号生成结构化人格、问答格式设计、Kernel Herding、Greedy Farthest等子样本化算法

**📊 数据集**

匿名化聚合的电商平台用户行为记录（点击、购买、活跃度）以及公开数据集（社交科学调查、电影评分、开放电商交易）

**📈 对比分析**

采用方向性准确率、信号重叠度、Bhattacharyya距离等符号对齐指标评估。实验显示：pairwise rating问答格式在CTR与订阅两指标上分别达75%与80%方向准确率；基于平台行为的人格在CTR上70%准确率；公开电商数据可与平台数据相媲美；深度人群在CTR上显著优于代表性人群；子样本化至500人即可保持≈1%精度损失

**⚠️ 局限性**

仅在单个屏幕截图下评估，缺乏完整页面上下文；仅覆盖40个电商实验，泛化性待验证；LLM可能存在正偏、锚定效应，导致对处理方的乐观评估；推断的身份属性可能偏差；仅使用Anthropic模型，未测试其他模型

---

## 420. Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model

**arXiv ID:** 2609.00812 | [PDF](https://arxiv.org/pdf/2609.00812v1)

**作者:** Zhen Wei `[一作]` (Swiss Federal Institute of Technology in Lausanne), Pascal Fua `[通讯]` (Swiss Federal Institute of Technology in Lausanne)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 DiffGeo，一种基于潜在空间扩散模型的生成框架，用于在极端数据稀缺条件下实现可控的气动形状设计空间探索，并在 2D 空气动力翼和 3D 涡轮机叶片上进行验证。

**💡 创新点**

创新点包括：①将自动形状参数化与扩散采样耦合，实现数据高效、无模式崩溃的生成；②引入能量基引导和增强条件采样，实现对复杂多维约束的可微控制；③将生成器与任务目标解耦，支持迁移到新设计任务而无需重新训练。

**🔧 技术方法**

核心技术包括：自动解码器（LSM）进行形状参数化、潜在空间扩散模型（LSDM）进行无条件采样、能量基条件扩散模型（CLSDM）及其增强采样算法，以及对比的 GAN、VAE 和基于条件 GAN 的生成方法。

**📊 数据集**

使用 UIUC 空气动力翼数据库（50~1000 个样本）和基于六个基准叶片线性插值得到的 75 个 3D 叶片点云数据，作为训练与评估的数据集。

**📈 对比分析**

与 GAN、VAE 等基线方法在 FID、D_intra、D_inter、面积误差、L/D 优化结果等指标比较，DiffGeo 在极低数据（50 样本）下的 FID 低于 GAN，生成多样性和目标满足度最高；在 surrogate‑based 优化中，DiffGeo 生成的训练数据使 surrogate R² 达到 0.96，最终 L/D 提升至 7.7，显著优于基线；在 3D 叶片实验中，DiffGeo 在约束下生成的叶片相较线性插值提高了约 1% 的等熵效率。

**⚠️ 局限性**

主要限制包括：生成受训练数据流形限制，难以全新拓扑设计；强约束下可能导致几何不光滑或无效；能量基引导需要可微约束表达，无法直接处理非可微或离散约束。

---

## 421. A Wearable Pneumatic Device for Continuous, Closed-Loop, Bidirectional Tactile Interaction

**arXiv ID:** 2609.00612 | [PDF](https://arxiv.org/pdf/2609.00612v1)

**作者:** Cosima du Pasquier `[一作]` (Stanford University), Allison M. Okamura `[通讯]` (Stanford University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研发了一套可穿戴双向气压触觉装置，能够在使用者和机器人之间实现连续、闭环、双向触觉交互，并通过无线配对在远程操作中实时复制压力信号。

**💡 创新点**

核心创新点在于：① 采用同一气压袋同时作为感知与刺激载体；② 每个通道配备比例压电阀、压力传感器和本地微控制器，实现局部闭环控制；③ 通过分布式硬件架构将压缩机与阀门搬到手部，彻底消除有线缠绕；④ 两台相同设备可无线配对，实现无中间抽象的实时双向触觉复制。

**🔧 技术方法**

技术实现包括：可穿戴软气压触觉（TPU/尼龙气囊），比例压电阀 + 12通道压力传感器 + 本地PI控制器；无线通信与系统级微控制器；HoloLens 2 手部追踪驱动 Kinova Gen3 机械臂；机器人端配备纺织式感知气囊；实验中采用闭环压强调节与实时传感同步。

**📊 数据集**

数据集与实验：25名受试者（10女性、14男性、1非二元）完成感知与遥操作两阶段；感知任务包含力、刚度、重量三类辨别；遥操作任务为三种视觉条件下的 pick‑and‑place；收集压力波形、抓取力、任务时长、路径长度、SUS 与 NASA‑TLX 数据。

**📈 对比分析**

对比实验：有无触觉反馈两组；评价指标包括平均施加压力、任务完成时间、轨迹长度、NASA‑TLX各维度、SUS评分。结果显示：在有触觉时平均施加压力下降 23.1%，任务时长下降 27.4%，路径长度缩短 28.9%；NASA‑TLX 中 Mental Demand 下降 18.8%，Frustration 下降 14.4%。在感知实验中，力辨别准确率 92%，刚度辨别 90–96%，重量辨别在单人手握时 95–100%，遥操作时显著下降。系统频带 34.2 Hz，最大力 76 N，端到端延迟 64 ms。

**⚠️ 局限性**

局限性：① 设备重量约 300 g，仍偏重；② 仍存在小范围泄漏与长期压力保持不足；③ 与机器人/视觉系统的耦合导致整体延迟受限；④ 部分受试者将系统延迟误认为触觉延迟；⑤ 需要进一步集成压缩机与电源、改进封装以提升舒适度与可穿戴性。

---

## 422. Lacan: Making Accountability in Anonymous Networks Real

**arXiv ID:** 2609.01075 | [PDF](https://arxiv.org/pdf/2609.01075v1)

**作者:** Naoya Takada `[一作]` (University of Osaka), Toru Hasegawa `[通讯]` (Shimane University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种名为Lacan的匿名网络协议，实现匿名性与可追责性的兼容；

**💡 创新点**

创新点在于链式后继证明（chain of successor proofs）实现路径可追溯、将会话签名转为可追踪签名（traceable signature）实现会话级别的签名，从而将每包签名成本降低至会话级别，并通过钥匙承诺认证（key‑committing AE）确保明文与密文唯一对应；

**🔧 技术方法**

使用的技术包括：ECDH一向认证密钥交换、TLS安全通道、可追踪签名、不可否认签名、钥匙承诺加密、MAC、哈希以及基于Curve25519/Ed25519的加密与签名算法；

**📊 数据集**

实验使用了Intel Xeon Gold 6330主机、E810网卡、DPDK实现的中继节点，数据包大小分别为512字节与1322字节；

**📈 对比分析**

在对比实验中，Lacan在数据传输阶段的延迟约为10⁴循环，显著低于路径建立和违规报告阶段；吞吐量最高可达14.37 Gbps（8核、1322字节）；相较于传统需要每包签名的方案，Lacan在链路层实现了四个数量级的性能提升；

**⚠️ 局限性**

局限性包括：需要可信的中心验证者或至少一半以上验证者诚实；仅在局部对手模型下能保证匿名性，无法抵御全路径监控；不支持多路径或大规模攻击；合同只能是可计算的确定性函数；对伪造或分裂路径攻击仍有一定风险；

---

## 423. HitMem: Hierarchical Temporal 3D Memory with Multi-Modal Context-Aware Retrieval for Dynamic Environments

**arXiv ID:** 2609.00950 | [PDF](https://arxiv.org/pdf/2609.00950v1)

**作者:** Ruijie Tang `[一作]` (Institute of Software Chinese Academy of Sciences), Jiaxin Zhu `[通讯]` (Institute of Software Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了HitMem，一个层级化的时序3D记忆框架，并结合多模态上下文检索，支持机器人在动态环境中自适应地更新记忆并高效定位目标。

**💡 创新点**

创新点在于：① 设计了分层语义图结构与指数衰减机制，实现对记忆活性动态调节；② 引入两阶段检索策略，先利用外部代理轨迹约束搜索空间，再通过类别亲和度排序候选物体；③ 将视觉‑语言模型与实时ICP融合，实现细粒度空间与语义的同步更新。

**🔧 技术方法**

使用的技术包括：开放词汇视觉‑语言模型（如CLIP）、基于支持度的语义图构建、双向ICP融合、外部代理轨迹重建、LLM解析自然语言指令、类亲和性计算与软最大化筛选。

**📊 数据集**

实验使用了自研的Dyna‑THOR基准（基于AI2‑THOR的交互式动态场景）以及原始AI2‑THOR场景进行评估。

**📈 对比分析**

与DELTA、ConceptGraphs、DovSG、DynaMem等四个基线进行对比，HitMem在成功率、SPL与GCR等指标上均显著优于所有基线，尤其在目标重定位任务中表现突出。

**⚠️ 局限性**

局限性在于：高度依赖视觉检测与分割的准确性；轨迹重建在遮挡严重或外部代理行为极端多样时效果下降；在极大规模多房间环境下仍需进一步优化衰减率与搜索空间平衡。

---

## 424. CRAFT: Fine-Tuning Pre-hoc Explainability in AI-native 6G RAN

**arXiv ID:** 2609.00590 | [PDF](https://arxiv.org/pdf/2609.00590v1)

**作者:** Pranshav Gajjar `[一作]` (North Carolina State University), Vijay K Shah `[通讯]` (North Carolina State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种数据驱动的预先推理对齐方法 CRAFT，用于在 6G RAN 中训练可审计的预先推理小语言模型。

**💡 创新点**

创新点在于通过离线验证生成 (输入、推理轨迹、标签) 三元组数据集，跳过强化学习的冷启动瓶颈，实现高质量的预先推理。

**🔧 技术方法**

采用低秩适配 (LoRA) 的监督微调，并结合 Oracle Reasoner 与 Predictor 的验证流程，以及与 Group Relative Policy Optimization (GRPO) 的对比。

**📊 数据集**

使用了 TRACTOR (四类流量分类) 与 IC xApp (干扰检测) 两个 O-RAN 数据集进行评估。

**📈 对比分析**

与零射、SFT、GRPO、SFT+GRPO 等基线比较，CRAFT 在准确率、宏 F1、解析失败率方面提升至约 80%+，且训练时间和能耗显著降低。

**⚠️ 局限性**

局限在于仅评估分类任务，对连续输出或更大模型族未验证，且验证过程依赖大型 Oracle 模型。

---

## 425. Context-Aware Intelligent Vehicles

**arXiv ID:** 2609.00682 | [PDF](https://arxiv.org/pdf/2609.00682v1)

**作者:** Liangkai Liu `[一作]` (University of Michigan), Kang G. Shin `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对智能车辆的上下文感知技术进行了系统综述，梳理了环境感知、规划与控制、安全与安全性、车联网等四大领域的最新研究；同时提出了统一的上下文状态概念，并对未来上下文引擎的设计挑战进行了分析；

**💡 创新点**

创新点在于将“上下文”作为贯穿整个自动驾驶栈的一流原则，提出了统一共享上下文状态的框架，并系统识别出四大关键挑战（多模态融合、时序建模、稀有事件处理、协作共享）；为后续研究提供了清晰的挑战导向和技术路线；

**🔧 技术方法**

主要技术包括：多模态上下文融合（早/晚/混合融合、动态融合策略）、基于 4D 世界模型与动态场景图的时序上下文建模、资源感知的稀有事件触发机制、基于 V2X 的协同上下文共享与信任管理；同时引用了众多深度学习、图模型、强化学习、LLM 等前沿方法；

**📊 数据集**

由于是综述论文，未使用单一实验数据集；文中引用了多个公开数据集（如 AIDE、CAPS、PRISC‑Net、OpenStreetMap、HD maps 等）来说明研究背景和技术落地；

**📈 对比分析**

文章采用文献对比的方式，基于案例表格和趋势分析对现有方法进行比较，未给出统一的定量指标，但指出各类方法在检测精度、推理时延、能耗、鲁棒性等方面的相对优势与不足；

**⚠️ 局限性**

局限性包括：综述范围受时间窗口限制，快速演进的技术与数据集难以完全覆盖；缺乏统一的评测基准和客观性能对比；上下文定义与标准化尚未成熟，导致跨研究比较困难；最后，针对未来挑战的解决方案仍处于概念阶段，缺乏实验证实。

---

## 426. Drift-Aware LLM Routing with Sparse Contexts and Shared Budgets

**arXiv ID:** 2609.00662 | [PDF](https://arxiv.org/pdf/2609.00662v1)

**作者:** Cheung Hao Lee `[一作]`, Patrick Wong `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种漂移感知稀疏路由算法，用于在多模型LLM服务中根据上下文高维特征和多重资源约束动态分配请求；

**💡 创新点**

创新点在于将滚动稀疏估计、置信度调节的盈余评估、在线影子价格更新以及硬阈值计量器结合，形成一种可在非静态、稀疏上下文环境下保持预算可行性的控制框架；

**🔧 技术方法**

核心技术包括：稀疏线性/硬阈值回归（rolling sparse estimator）、置信半径估计、惰性奖励/上限消耗估计、资源价格（shadow price）更新、以及硬容量计量；

**📊 数据集**

实验使用自构造的合成数据集，包含4类模型、2类资源、28维上下文、两次任务/模型更新，共4800条请求；

**📈 对比分析**

与静态稀疏、稠密、全史稀疏以及无漂移clairvoyant动态等基线比较，滚动稀疏路由在总效用上与clairvoyant相差约2–3%，且在资源利用率和适应延迟上优于冻结路由；

**⚠️ 局限性**

局限性包括：仅在理想化的线性稀疏模型和无偏审计环境下验证；缺乏真实LLM多模型面板和渐进漂移的实验；对审计率和稀疏度的理论假设可能不易在生产环境中满足。

---

## 427. Beneath the Diff: Diagnosing and Mitigating Algorithmic Mode Collapse in Code-Level Autonomous Research Loops

**arXiv ID:** 2609.00077 | [PDF](https://arxiv.org/pdf/2609.00077v1)

**作者:** Bowei He `[一作]` (MBZUAI), Xue Liu `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统研究了代码级自主研究循环（ARL）中的算法模式坍塌问题，并提出了DAPS（Diversity‑Aware Proposal Sampling）机制来缓解该问题。

**💡 创新点**

创新点在于：①构建四轴多维度诊断仪表（表面、多样性、机制熵、语料相似度）系统地量化算法坍塌；②提出DAPS三元组（类别覆盖重权、持久编辑记忆、审核门控）实现轻量级多样性提升与Goodhart效应抑制。

**🔧 技术方法**

技术手段包括：LLM提案器（Claude Opus 4.7、GPT‑5.2、Llama‑3.3‑70B）、Sentence‑BERT编码、HDBSCAN聚类、类别重权因子、余弦相似度记忆过滤、审核门控与阈值校准。

**📊 数据集**

实验数据集涵盖四个NLP任务：小型LM预训练（OpenWebText子集）、指令微调（Alpaca）、推理微调（GSM8K）以及提示优化（ARC），对应的审核指标为LAMBADA、C4、MMLU、IFEval、ARC‑Easy/Challenge、CommonsenseQA，盲测集包括WikiText‑103、Dolly指令、SVAMP、OpenBookQA。

**📈 对比分析**

与八种基线（Vanilla、HiTemp、R‑Diverse‑A、Prism‑A、Reflexion、RandSearch、HO‑EarlyStop、HO‑Revert）对比，DAPS在所有任务中均获得最高的审核可信度比（≈0.78‑0.80）和最低的语义聚类衰减（≈0.21），同时在环内优化速度和收益率与Vanilla相当。

**⚠️ 局限性**

局限性包括：仅在单一A100规模实验验证；未测试前沿规模模型或极长迭代（10k+）；诊断和DAPS以ML流水线为主，非ML领域需重新构建机制分类；审核指标为验证信号非最终测试；语料相似度仅为描述性指标，未完全证明检索行为；盲测仍受基准偏差影响。

---

## 428. Beyond Technological Solutionism: Rethinking XR in Healthcare

**arXiv ID:** 2609.01028 | [PDF](https://arxiv.org/pdf/2609.01028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 429. SlideMix: Enhancing Whole Slide Image Analysis via Multimodal Shuffling

**arXiv ID:** 2609.00396 | [PDF](https://arxiv.org/pdf/2609.00396v1)

**作者:** Chad Wong `[一作]` (University of California, Irvine), Fei Xia `[通讯]` (University of California, Irvine)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 SlideMix，一个在 MIL 体系下对 WSIs 进行多模态、动态的切块混合的通用增强框架。

**💡 创新点**

创新点在于使用检索增强生成的 VLM 识别诊断 ROI 后，在 ROI 内做 in-place tile shuffling，并通过基于 loss 的动态课程学习反馈逐步提升跨尺度特征学习，兼顾诊断相关性与混合多样性。

**🔧 技术方法**

采用检索增强的 Gemini/CONCH VLM 进行 ROI 选取与 soft‑label 生成；在 MIL 的两阶段流程中实现 tile 级嵌入、PCA 相似度阈值、shuffle 比例、shuffle 粒度的动态调度；并结合多尺度 MIL backbone（ABMIL、TransMIL 等）。

**📊 数据集**

在 11 个公开 WSI 数据集（PANDA、CAMELYON16、IMP‑CRS‑2024、TCGA‑Lung、TCGA‑BLCA、TCGA‑UCEC、TCGA‑CESC、TCGA‑UCS、TCGA‑UVM、TCGA‑BRCA、TCGA‑GBM）上进行实验，涵盖 8 种诊断任务。

**📈 对比分析**

与 MixUp、CutMix、ResizeMix、PuzzleMix 等传统增强以及 8 种 MIL backbone 进行对比，SlideMix 在 10/11 直接增强对比中排名第一，并在 82/110 的 backbone–dataset 组合中提升或保持性能，平均准确率提升约 1–3%。

**⚠️ 局限性**

局限性包括：在样本量极少的情况下难以生成足够多样的混合；soft‑label 的质量依赖 VLM 的知识覆盖，罕见病种可能表现欠佳；对位置编码敏感的模型（如 TransMIL）在 ITS 产生的混合后可能受损，且目前未扩展至多模态临床数据。

---

## 430. StreamScout: Learning When to Look Deeper for Streaming Video Understanding

**arXiv ID:** 2609.00291 | [PDF](https://arxiv.org/pdf/2609.00291v1)

**作者:** Ce Zhang `[一作]` (Carnegie Mellon University), Ming Zhou `[通讯]` (TikTok)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了StreamScout框架，解决流媒体视频理解中查询时自适应获取视觉证据的问题，保持轻量文本时间线并通过三阶段视觉视图（即时、回溯、检索）逐步升级，直至答案足够；

**💡 创新点**

核心创新在于将证据获取深度作为可学习的停止-升级决策，引入自蒸馏方法得到无标注的停止标签（StreamScout‑S），并进一步通过强化学习（StreamScout‑R）直接优化准确率与计算成本平衡；

**🔧 技术方法**

技术包括：文本时间线生成、三层视觉视图构造（最近帧、均匀回溯、查询相关检索）、自蒸馏监督的LoRA适配器、基于GRPO的强化学习策略；

**📊 数据集**

在三个流媒体视频问答基准上评测：OVO‑Bench、StreamingBench、StreamBench；此外还在NExT‑QA、EgoSchema、MLVU等离线视频问答基准上验证迁移；

**📈 对比分析**

与均匀采样、VideoLLM‑online、Flash‑VStream、Dispider、OASIS等现有系统比较，StreamScout‑S和StreamScout‑R在准确率上均显著提升（最高提升约20%），同时在token消耗与推理时延上降低约60%和70%；

**⚠️ 局限性**

局限性包括：对视觉缓存的固定容量限制、检索视图对CLIP嵌入相关性的依赖、强化学习训练稳定性和对不同视频域的泛化可能受限、以及在极长历史或多模态细粒度问题上仍需进一步改进。

---

## 431. Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation

**arXiv ID:** 2609.00086 | [PDF](https://arxiv.org/pdf/2609.00086v1)

**作者:** Ante Kapetanovic `[一作]` (Infobip), Emanuel Lacic `[通讯]` (Infobip)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估LLM在对话推荐系统中的重排序性能和稳定性，系统性探究候选池大小、检索器类型、评分策略和解码温度对结果的影响。

**💡 创新点**

提出将候选生成、池大小、评分策略和解码配置作为必报实验变量，并将列表稳定性与准确性一并评估，揭示这些因素对LLM排名效果的决定性作用。

**🔧 技术方法**

采用两阶段检索-重排序管道：语义检索（SBERT）、协同过滤（EASE）、序列模型（SASRec）作为候选生成；LLM重排序器（Claude Opus 4.6、GPT‑5.2、Claude Sonnet 4.6等）和温度采样进行评估。

**📊 数据集**

使用ReDial电影对话推荐基准（1,025条对话）及其6,924部电影目录。

**📈 对比分析**

通过NDCG@10、Hit@10、覆盖率等指标与CF、序列、流行度基线对比；在匹配候选池下，专有LLM显著优于EASE（0.1497 vs 0.0939），但开源LLM不及；增大候选池或更换检索器可提升50%+；提高温度导致列表差异显著增大，平均准确度基本保持。

**⚠️ 局限性**

仅在单一电影域和ReDial上验证，未考察更大目录、其他领域、成本与延迟等因素，候选生成与重排序的相互作用也未完全分离。

---

## 432. Assessing Suicide Risk in Arabic Crisis Helpline Calls: A Comparison of Arabic and English Large Language Models

**arXiv ID:** 2609.00191 | [PDF](https://arxiv.org/pdf/2609.00191v1)

**作者:** Linhai Ma `[一作]` (Yale University), Samah Fodeh `[通讯]` (Yale University)

**通讯引用:** 932 | [OpenAlex ID](https://openalex.org/A5025703949)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了在黎巴嫩阿拉伯语危机热线转录文本上进行自杀风险分类的模型，并实现了完全保密的语音转录与脱敏管道。

**💡 创新点**

首次针对叙利亚方言阿拉伯语热线对话训练自杀风险分类器；在服务内实现语音识别+脱敏而不泄露音频；对比直接阿拉伯语与机器翻译后英文文本的分类效果。

**🔧 技术方法**

使用指令调优的大型语言模型（Qwen、Llama、AceGPT 等）与 Transformer 编码器（CAMeLBERT、AraBERT、BERT 等）；采用 Whisper‑family ASR、Arabic NER 脱敏；双语言训练与评估。

**📊 数据集**

来自黎巴嫩国家情绪支持与自杀预防热线的 383 份可评估呼叫（含 528 份无歧义标签）的语音转录文本；同时提供机器翻译成英文版本。

**📈 对比分析**

采用 80/20 分层拆分，评估 ROC‑AUC、PR‑AUC、宏 F1、召回率；在阿拉伯语上 70B 模型最高宏 F1 81.19（高危）/73.29（低危），英文 70B 模型宏 F1 85.00/75.96；翻译后性能基本不下降；解码器优于编码器。

**⚠️ 局限性**

数据仅来自单一黎巴嫩热线，样本规模有限；音频转录和翻译误差未完全评估；标签来源于运营商记录，可能含错误；未检验模型在连续实时调用中的表现。

---

## 433. User Representation via Cross Multi-source Behavior Pre-training for Mobile Games

**arXiv ID:** 2609.01057 | [PDF](https://arxiv.org/pdf/2609.01057v1)

**作者:** Chengqi Yang `[一作]` (Chinese Academy of Sciences), Xiang Ao `[通讯]` (OPPO Research Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CM-PTM框架，通过在移动游戏用户设备级行为日志上进行跨多源行为预训练，学习统一的用户表示；

**💡 创新点**

创新点在于（1）采用分层级的 cascaded mask‑then‑predict 代理任务，先预测下一个行为的来源，再逐步细化到 App 类别、App ID 与操作类型；（2）设计跨粒度注意力机制（Granularity‑Aware Self‑Attention 与 Cross‑Granularity Fusion Attention）以同时处理细粒度与粗粒度行为；（3）在同一预训练框架中整合跨源依赖与细粒度动态，解决传统单源预训练忽视跨源关联的问题；

**🔧 技术方法**

技术手段包括：自监督掩码预测、层级级联预测、Transformer‑style 多头注意力、MLP 信息模块、Attention Fusion、LightGBM 下游融合以及对比学习与生成式任务的对比基线；

**📊 数据集**

使用 OPPO 设备级行为数据集，包含约 194 万用户、83k 个游戏 App，三类行为源（第三方游戏 App、AppStore、GameCenter），预训练期 2023/07/13‑19， downstream 6 个游戏推荐任务（新/老用户·新/老游戏）

**📈 对比分析**

与多种生成式（Bert4Rec、PeterRec、PTUM）、判别式（CLUE、CCL、AdaptSSR）和集成式（MAFN）基线对比，CM‑PTM 在 AUC 与 R@P0.95 上均超过所有基线，提升幅度约 1–3%；在线 A/B 测试中较 DeepFM 提升 1.28% AUC、1.07% R@P0.95；

**⚠️ 局限性**

局限性：未加入绝对时间编码，只利用粗粒度的第三方 App 事件；仅覆盖三类源，缺乏细粒度应用内交互；训练时间相对较长；主要针对移动游戏推荐，泛化至其他领域需进一步验证；

---

## 434. How Do Language Models Choose Between Context and Memory?

**arXiv ID:** 2609.00753 | [PDF](https://arxiv.org/pdf/2609.00753v1)

**作者:** Benjamin Shih `[一作]` (Stanford University), Arianna Cao `[通讯]` (Stanford University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对语言模型中上下文与参数记忆冲突下的源头选择进行因果干预，研究权威方向能否真正驱动模型遵循上下文或记忆。

**💡 创新点**

创新在于区分权威表示、模型自发使用以及跨任务可复用性，并通过对权威方向的交换实验验证其因果效应。

**🔧 技术方法**

采用激活干预（authority direction steering）和权威坐标交换，计算高低权威间的对数几率差，评估 gap‑closure 与 mediation index。

**📊 数据集**

使用自制的颜色任务、材料组成任务以及州/国家事实任务，构造高低权威提示并测量模型的答案。

**📈 对比分析**

与 Qwen、Llama‑3.1‑8B、OLMo‑2‑7B 等模型对比，单层权威交换可恢复 30–68% 的自然权威差距；跨任务传递仅闭合约 9% 的差距，权威方向在跨任务上效果显著下降。

**⚠️ 局限性**

局限性包括跨任务权威方向的可迁移性弱、样本量有限、可能存在非线性交互未被捕获，以及仅在单层干预而非多层递归交换，导致对模型全局机制的解释不完整。

---

## 435. Beyond Magnitude: Contrastive Routing for Modular Mixture-of-Experts

**arXiv ID:** 2609.01100 | [PDF](https://arxiv.org/pdf/2609.01100v1)

**作者:** Nikolaos Xiros `[一作]` (Institute for Language and Speech Processing, Athena Research Center), Georgios Paraskevopoulos `[通讯]` (Institute for Language and Speech Processing, Athena Research Center)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种新的稀疏 Mixture‑of‑Experts 路由机制（CoRM），通过对隐藏状态进行指数移动平均（EMA）参考减法和低维对比注意力投影，实现专家的更高特化和更稳定的路由。

**💡 创新点**

创新点在于将路由判定从绝对激活值转移为相对激活差（token 对比平均参考），并结合低维投影与动态 EMA 参考，显著提升了专家间的结构独立性与语法特化。

**🔧 技术方法**

使用的技术包括 EMA 参考状态、L2 正则化的键/查询投影、低维（d₂=64）对比注意力、加载平衡损失以及对比路由日志（gap）实现稀疏激活。

**📊 数据集**

训练数据为 30 B 词的 The Pile 文本语料库，评估数据则涵盖 ARC‑Easy/Challenge、BoolQ、HellaSwag、LAMBADA、PIQA、RACE、OpenBookQA 与 SciQ 等九个零样本推理与语言理解基准。

**📈 对比分析**

与 Dense、dMoE、ReMoE、X‑MoE 等传统 Top‑k 或软路由方法对比，CoRM 在 Top‑1 路由下平均零样本准确率提升 0.67–1.69 个百分点，Top‑2 路由下提升 1.38–1.77 个百分点，额外参数增幅仅 2.9%，额外 FLOPs 仅 2.6%。

**⚠️ 局限性**

局限性包括：仅在 182M–469M 参数规模上验证，未测试更大规模模型；EMA 动态参考在推理阶段固定，缺乏自适应；训练仅使用单一 The Pile 数据集，未检验跨域泛化。

---

## 436. WiseSpec: Requirements-Driven Agents for Code Generation

**arXiv ID:** 2609.00568 | [PDF](https://arxiv.org/pdf/2609.00568v1)

**作者:** Zhao Tian `[一作]` `[通讯]` (Tianjin University), Zhao Tian (Tianjin University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种需求驱动的代理框架，自动构建结构化、信息丰富的需求，评估其质量并迭代优化，以指导大型语言模型生成仓库级代码。

**💡 创新点**

创新点在于将需求工程方法引入代码生成，使用预定义DSL构造需求，采用基于执行的质量评估，以及通过冲突、遗漏、歧义三类缺陷的对齐规则实现需求的迭代改进。

**🔧 技术方法**

主要技术包括信息检索与代码片段分析、需求DSL、执行测试生成与评估、缺陷分类与对齐规则、贪心优化策略以及对抗式回退记录。

**📊 数据集**

实验使用了三大仓库级代码生成基准：SWE‑bench‑Lite、SWE‑bench‑Verified、SWE‑bench‑Pro，并在DeepSeek‑V3.2、Qwen‑Plus‑2025‑12‑01（以及Claude‑Opus‑4.8）两种LLM上验证。

**📈 对比分析**

与Agentless、Trae‑agent和Claude‑Code等三大基线相比，在%Applied和%Resolved两项指标上均显著提升，平均提高约13.2%/2%–29%（%Resolved）和11%–63%（%Applied），并在更强模型上继续保持优势。

**⚠️ 局限性**

局限性包括对测试集的依赖，无法处理缺乏足够测试用例的场景；需求DSL和对齐规则的设计需要人工专业知识，可能不易迁移到其他领域；以及迭代过程可能在复杂项目中产生高计算成本。

---

## 437. Distributed Implicit Harm: A Compositional Safety Blind Spot in MLLM-Based Video Moderation

**arXiv ID:** 2609.00206 | [PDF](https://arxiv.org/pdf/2609.00206v1)

**作者:** Ruotong Wang `[一作]` (Chinese University of Hong Kong), Baoyuan Wu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 7683 | [OpenAlex ID](https://openalex.org/A5068027800)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并系统化分布式隐式伤害（DIH），提出多代理合成框架生成9,725条时序与跨模态视频，并基于此构建DIH-Bench评估多模态大语言模型安全性能。

**💡 创新点**

创新点在于将隐式伤害定义为局部安全但整体有害的组合，设计自动化多代理生成与链式推理注释流程，以及专门针对DIH的后训练与评估方案。

**🔧 技术方法**

使用多代理生成、文本到视频模型、链式思考（CoT）提示、监督微调（SFT）与群组相对策略优化（GRPO）等技术。

**📊 数据集**

采用自研DIH Dataset（6,742条DIH‑T + 2,983条DIH‑M）以及从TikTok/YouTube/Instagram收集的128条真实DIH视频。

**📈 对比分析**

与30+专有与开源MLLM（Gemini、GPT、Claude、Qwen、InternVL、MiniCPM等）在DIH‑Bench上进行零样本与指导式评测，最高模型在有害样本上的准确率低于45%，但后训练后可提升至约70%/80%，无害样本准确率未降。

**⚠️ 局限性**

局限性包括仅关注时序与跨模态两个轴，未覆盖其他分布式伤害维度，缺少文化/社会语境适配，且改进主要依赖后训练，尚未验证实时部署可行性。

---

## 438. Towards a Reliable and Practical Eval Pipeline

**arXiv ID:** 2609.00805 | [PDF](https://arxiv.org/pdf/2609.00805v1)

**作者:** Emma Thuong Nguyen `[一作]` (Salesforce), Abhishek Ghose `[通讯]` (Salesforce)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种端到端的LLM评估流水线，先将评估拆解为检查表问题，再用学习到的聚合模型生成最终评分，同时提供SHAP解释和置信区间。

**💡 创新点**

创新点在于将检查表拆分与聚合模型相结合，既提高不同LLM之间的一致性与自一致性，又通过树模型实现可解释性与可校准的置信度估计。

**🔧 技术方法**

采用Gradient Boosted Decision Trees (GBDT) 进行聚合，SHAP 进行解释，Conformal Prediction 进行置信区间估计，并使用 CheckEval 生成检查表。

**📊 数据集**

使用 SummEval 数据集进行摘要质量评估，包含 4 个质量轴（连贯性、一致性、流利度、相关性）。

**📈 对比分析**

与基线 seed prompts 对比后，流水线在 inter‑LLM agreement（平均提升至 0.96）、自一致性（平均 0.98）以及 RMSE（所有评估轴均最低）等指标上均优于传统方法。

**⚠️ 局限性**

局限在于仅在单一摘要数据集和四个专有 LLM 上验证，缺乏多任务、多数据集和公开 LLM 的通用性验证。

---

## 439. GUI-CC: Benchmarking Contextual Consistency of GUI World Models as Agent Environments

**arXiv ID:** 2609.00048 | [PDF](https://arxiv.org/pdf/2609.00048v1)

**作者:** Lin Fu `[一作]` (Zhejiang University), Yu Rong `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了GUI-CC基准，用以评估GUI世界模型在作为代理环境时的上下文一致性；

**💡 创新点**

创新点在于将评估焦点从单步画面预测转向多步交互环境一致性，并设计了离线参考动作轨迹和在线代理循环两条互补轨迹；

**🔧 技术方法**

使用了文本/语义动作表征、图像或可渲染HTML输出的世界模型，配合GPT‑5.5等VLM进行自动评估，并通过历史条件输入测试模型记忆能力；

**📊 数据集**

数据来源包括GUIOdyssey（500条离线轨迹）以及30款安卓应用的200条经真机验证的在线任务；

**📈 对比分析**

与现有单步评估方法对比，实验显示虽然模型在视觉相似度、可用性等单步指标上表现尚可，但在多步任务完成率和上下文保持上远低于预期；

**⚠️ 局限性**

局限性包括仅覆盖移动端、使用固定的检验代理、评估主要依赖VLM自动判分，未充分考虑多样化策略或更细粒度的人工验证。

---

## 440. Denoising Diffusion Generative Models Secretly Calculate Attentions

**arXiv ID:** 2609.00885 | [PDF](https://arxiv.org/pdf/2609.00885v1)

**作者:** Farzan Haddadi `[一作]` (Iran University of Science & Technology), Narges Mokhtari `[通讯]` (Iran University of Science & Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将扩散模型与注意力机制和自动编码器等价化，并基于此设计了无需内部去噪网络的低复杂度注意力生成框架。

**💡 创新点**

核心创新在于揭示扩散过程的闭式解等价于注意力公式，从而实现一次性注意力插值即可生成图像，显著降低训练和推理成本。

**🔧 技术方法**

使用扩散理论、注意力机制、VAE、k‑NN检索、卷积网络和Softmax等技术。

**📊 数据集**

在MNIST、Fashion‑MNIST、FFHQ和CelebA四个公开数据集上进行实验。

**📈 对比分析**

与传统扩散模型对比，FID在MNIST、Fashion‑MNIST上优于或相当，在FFHQ上略逊，但生成速度提升约8.5倍。

**⚠️ 局限性**

主要限制包括对σ和K的敏感性、在高分辨率数据集上生成质量略低，以及仍需构建近似最近邻检索结构。

---

## 441. Does Fault Localization Beat a Fresh Attempt? A Placebo-Controlled Study of Test-Guided Code Repair

**arXiv ID:** 2609.00854 | [PDF](https://arxiv.org/pdf/2609.00854v1)

**作者:** Anik Jha `[一作]` `[通讯]`, Anik Jha

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过比较三种干预手段（盲重采样、基于测试的定位后填充、相同长度随机区间填充），评估大型语言模型在单函数级别代码修复中的定位效果。

**💡 创新点**

创新点在于提出双控制协议（定位与随机空位对照）并开展本地化可用性审计、预算与令牌成本分析，揭示定位对修复性能的真正贡献。

**🔧 技术方法**

使用的技术包括 Qwen、Gemma、Mistral 等冻结大模型、Ochiai 频谱定位、填充式（FIM）生成、与随机占位填充的对照实验。

**📊 数据集**

数据集涵盖 HumanEval+、MBPP+ 以及 LiveCodeBench 的函数调用子集，统一使用公开测试与增强测试进行评估。

**📈 对比分析**

实验通过 per-attempt 成功率、unlock 计数及 Holm 校正的统计检验比较三种方法；结果显示定位填充在大多数模型下不超过盲重采样，仅在极低令牌预算下略有优势，随机占位对照表明仅靠小编辑无实质提升。

**⚠️ 局限性**

主要局限包括定位信号稀缺（仅 23% 失败可定位）、仅基于公开测试、只评估单函数级别且冻结模型、未覆盖多块/仓库级修复、且未实现完整的令牌/时间成本匹配。

---

## 442. Different representation learning objectives recover distinct latent structures from the same psychometric data

**arXiv ID:** 2609.00100 | [PDF](https://arxiv.org/pdf/2609.00100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 443. Recursive Criticality of AI Self-Improvement

**arXiv ID:** 2609.00137 | [PDF](https://arxiv.org/pdf/2609.00137v1)

**作者:** Mikhail Burtsev `[一作]` (London Institute for Mathematical Sciences), Mikhail Burtsev `[通讯]` (London Institute for Mathematical Sciences)

**通讯引用:** 1669 | [OpenAlex ID](https://openalex.org/A5065291780)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

构建并分析了一个包含递归反馈、前沿硬化与延迟的AI研发动力学模型，用来研究递归自我改进的临界性及其对AGI/ASI进展的影响。

**💡 创新点**

提出递归再生产数 R_AI = χ a / σ 作为自我改进是否自放大的本地判据，并将其推广到多主体网络和硬件限制场景，揭示了自我放大与加速不必对应特定能力阈值。

**🔧 技术方法**

使用延迟微分方程、稳定性分析、谱半径计算以及数值仿真，对模型参数空间进行探索，构造多种情景分析。

**📊 数据集**

未使用真实实验数据，而是基于结构参数的模拟实验；通过对比基线无递归反馈与不同递归强度的情景来评估影响。

**📈 对比分析**

对比基线（无递归）与四种递归强度情景（光滑、弱超临界、瞬时加速、快速转折），量化 AGI/ASI 阈值穿越时间及其间隔，展示递归放大能显著压缩 AGI 到 ASI 的时间窗口。

**⚠️ 局限性**

局限在于参数（递归增益、闭合度、前沿硬化指数、延迟）未能从经验中获得，模型为定性描述而非概率预测；缺乏对真实研发过程的直接量化验证，且硬件/能耗约束的假设也较简化。

---

## 444. Research on Optimized Fuzzy PID Temperature Control Strategy Based on Improved Particle Swarm Optimization

**arXiv ID:** 2609.00001 | [PDF](https://arxiv.org/pdf/2609.00001v1)

**作者:** Renjie Jin `[一作]` `[通讯]` (China University of Petroleum-Beijing), Renjie Jin (China University of Petroleum-Beijing)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了一种基于改进粒子群优化的模糊PID温度控制策略，并在FOPDT模型上进行仿真验证。

**💡 创新点**

创新点在于将Levy飞行突变与精英记忆池相结合的LMPSO算法，有效解决了模糊参数调优中的维度灾难与早熟收敛问题。

**🔧 技术方法**

使用了模糊PID控制、改进粒子群优化（LMPSO）、MATLAB/Simulink仿真与ITAE评价指标。

**📊 数据集**

使用的并非真实数据集，而是基于第一阶死区时间（FOPDT）模型与其参数扰动场景的仿真数据。

**📈 对比分析**

通过与传统PID、经验模糊、标准PSO和IPSO比较，LMPSO将稳态时间降至105.5 s，约比IPSO快42.5%，并在扰动与模型不匹配下表现出更高的鲁棒性。

**⚠️ 局限性**

主要局限在于仅在仿真环境中验证，缺乏硬件实测与真实工况下的性能评估。

---

## 445. Hardware Acceleration of Block-Diffusion LLM for Edge Devices

**arXiv ID:** 2609.01084 | [PDF](https://arxiv.org/pdf/2609.01084v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 446. Manifold-Aware General Coded Computing for Straggler-Resilient Distributed Computing

**arXiv ID:** 2609.00552 | [PDF](https://arxiv.org/pdf/2609.00552v1)

**作者:** Parsa Moradi `[一作]` (University of Minnesota, Twin Cities), Mohammad Ali Maddah-Ali `[通讯]` (University of Minnesota, Twin Cities)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于流形感知的通用编码计算（GCC‑Manifold）方法，利用输入数据的低维流形结构对编码点进行排序，以生成更平滑且更贴近流形的编码曲线，从而在分布式计算中提升对慢节点的容错性能。

**💡 创新点**

创新点在于将流形学习与GCC相结合，使用短哈密顿路径（2‑opt启发式）对输入样本进行几何顺序排列，使编码器与数据流形保持一致，显著降低解码误差。

**🔧 技术方法**

采用图构造（基于欧氏距离加权图）、Hamiltonian路径优化（2‑opt）、二阶平滑样条拟合、Chebyshev点设计以及对编码/解码结果进行均方误差与准确率评估的技术。

**📊 数据集**

实验数据集包括 MNIST 手写数字分类任务（LeNet5 网络）和一个高维多项式评估任务（人工构造的 8 次多项式）。

**📈 对比分析**

通过与标准 GCC 在不同慢节点数下的均方误差（MSE）和相对准确率（RelAcc）进行比较，结果显示 GCC‑Manifold 在慢节点比例较高时均方误差大幅下降、相对准确率提升，性能优于传统 GCC。

**⚠️ 局限性**

局限性包括：对数据近似低维流形的假设限制了适用范围；排序启发式虽效率高但未保证全局最优；以及在大规模工作节点或高维特征时的计算和内存开销仍较高。

---

## 447. Separating perception from reasoning in vision-language models: a model-free render ceiling for crystal structures

**arXiv ID:** 2609.00663 | [PDF](https://arxiv.org/pdf/2609.00663v1)

**作者:** Can Polat `[一作]`, Hasan Kurban `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于渲染逆推的无模型参考（render ceiling），通过逆转已知的摄像机视角并求解交叉视图对应，直接从图像重建原始晶体结构并计算对称性标签，从而为视觉‑语言模型的性能提供可验证的上限。

**💡 创新点**

创新点：①首次构造可认证、无模型的上限，为判断模型是感知失误还是推理失误提供客观依据；②通过对投影偶然性（phantom set）进行枚举，证明在所使用的渲染协议下此上限等于 1，所有模型缺陷均归因于模型本身；③揭示“提取阶段造假”现象，即模型输出的坐标列表几乎不匹配真实原子，从而把本应归因于推理的错误误判为推理失误。

**🔧 技术方法**

技术：逆向摄像机投影、基于同种族点的最近点交叉、三视角/四视角/五视角的交叉视图对应求解；对称性判定使用 spglib；构建了“几何判定器”（geometric oracle）并证明其唯一失败模式为投影偶然性；对不同摄像机布置进行条件数（κ）分析；评估时采用多视图平均、K‑投票自洽解码。

**📊 数据集**

数据集：来自 Materials Project 的 2,160 个晶体结构（1,610 训练、210 评估、1,950 规模扩展），每个结构使用 5 个正交摄像机通过固定协议渲染成 768×768 像素图像；标签为七类晶体系统（spglib 计算得到），并在扩展集上重复。

**📈 对比分析**

比较方法：对 14 个公开的视觉‑语言模型（含零射击、微调、受控推理等）在相同样本上使用相同解码预算进行评估；将模型性能与：①无模型上限（render ceiling=1）、②仅提供精确几何文本的模型、③无语言组件的监督像素基线（ResNet‑50）做对比。结果显示：最佳视觉‑语言模型在评估集上达到 0.8524（相对上限差距 0.1476），监督像素基线为 0.8952，远超所有视觉‑语言模型；提供几何文本后所有模型都有提升，但大多仍未达到上限。

**⚠️ 局限性**

局限性：①仅适用于已知可逆渲染的场景，无法直接推广到自然图像或无法逆转的可视化；②依赖固定的摄像机配置，其他渲染设置可能导致投影偶然性不为空；③仅评估七类晶体系统标签，对更细粒度的空间群或点群未做完整验证；④对像素级细节（例如阴影、纹理）不敏感，可能忽略某些视觉信息；⑤仅检验视觉‑语言模型的“提取‑推理”分离，未针对更复杂的多步骤推理任务进行测试。

---

## 448. SCoNE: Selective Context-aware Neuron Editing for Robust Retrieval-Augmented Generation

**arXiv ID:** 2609.00689 | [PDF](https://arxiv.org/pdf/2609.00689v1)

**作者:** Chaewon Kim `[一作]`, Seo Yeon Park `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

演示了如何使用ACL风格文件与LuaLaTeX或XeLaTeX

**💡 创新点**

提供了多语言文本示例和引用格式的演示

**🔧 技术方法**

使用了LaTeX、LuaLaTeX和XeLaTeX

**📊 数据集**

无具体数据集

**📈 对比分析**

未进行方法比较和性能评估

**⚠️ 局限性**

仅为模板示例，缺乏实验验证和实证结果

---

## 449. StudyBench: Can Self-Evolution Squeeze Textbooks for Olympiad Capability?

**arXiv ID:** 2609.00787 | [PDF](https://arxiv.org/pdf/2609.00787v1)

**作者:** Yinghao Chen `[一作]` (Tsinghua University), Chaojun Xiao `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了StudyBench，一个基于物理教材的受控基准，用以测量自进化方法将训练材料转化为可迁移问题求解能力的效率。

**💡 创新点**

创新点在于构建了包含能力差距、可达性与归因控制的测试集，并提供“指导极限”以量化训练材料可达知识在内部化过程中的落差。

**🔧 技术方法**

采用文本提取、能力过滤、可达性过滤、LLM辅助验证、指导痕迹生成等技术，并将多种自进化框架与不同基模型结合进行实验。

**📊 数据集**

使用11本核心物理教材和六个国际物理奥林匹克题库作为训练和测试数据，划分为Corpus、Instructions with/without Answer以及Application/Transfer两组。

**📈 对比分析**

在Qwen3‑8B、Llama‑3.2‑3B‑Instruct和Opus 4.7上对比了Bonito、GRPO、GEPA、ACE、TTRL、Intuitor、R‑Zero等方法，结果显示应用集提升显著但迁移集提升有限，Guidance Gap约为90%，计算平台在大量训练后仍停滞。

**⚠️ 局限性**

局限包括：过滤基于Qwen3‑8B，其他模型的能力差距和指导极限未完全保证；仅在物理领域验证，未证实可推广；计算消耗大且指导消融受限；评估依赖单一LLM验证器。

---

## 450. A Mathematical Framework for Legacy, Governance, and Decision Integrity in Enterprise AI

**arXiv ID:** 2609.00572 | [PDF](https://arxiv.org/pdf/2609.00572v1)

**作者:** Shorab Sarker `[一作]` `[通讯]` (Math Behind Innovation), Shorab Sarker (Math Behind Innovation)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一个设计科学框架，用数学指标评估企业 AI 系统在人员、技术、法规变迁等情况下的持续合规性与可解释性。

**💡 创新点**

创新点在于将传统治理原则量化为“Legacy Score”、分离证据信心与后果、引入权威感知检索、决策记忆与法规变动速度模型，并构建联邦监管知识图谱。

**🔧 技术方法**

采用几何平均公式、贝叶斯推理、加权检索分数、风险校准生成以及知识图谱结构等技术。

**📊 数据集**

使用完全合成的数据集（Beta 与 Bernoulli 生成的 200×10,000 条决策样本），并进行固定种子 Monte Carlo 模拟。

**📈 对比分析**

通过与仅基于模型置信度的基线进行匹配覆盖度对比，治理感知路由将自动错误率从 12.3% 降至 8.8%，高影响错误率下降 79% 以上，表明方法有效。

**⚠️ 局限性**

局限包括：指标和阈值需组织特定校准、未进行真实世界验证、对法律层级依赖专家解读、数据偏差与隐私风险，以及缺乏外部效度证明。

---

## 451. Non-Prehensile Throwing: A Reinforcement Learning Perspective

**arXiv ID:** 2609.00771 | [PDF](https://arxiv.org/pdf/2609.00771v1)

**作者:** Abdullah Mustafa `[一作]` (National Institute of Advanced Industrial Science and Technology), Tetsuya Ogata `[通讯]` (Waseda University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于强化学习的非抓握投掷方法（NP-Throw），通过学习滑动与滚动接触模式，直接在关节空间规划投掷轨迹，并实现了从仿真到真实机器人的零样本迁移。

**💡 创新点**

创新点在于：①不依赖传统的模型优化与低维轨迹参数化，而是利用强化学习探索混合滑动/滚动接触行为；②在动作空间使用关节jerk控制并通过上采样生成平滑高速指令；③通过最小二阶系统识别与不确定性感知训练，显著缩小仿真-真实差距，尤其对动态摩擦敏感。

**🔧 技术方法**

技术手段包括：强化学习（PPO）、Markov决策过程建模、关节jerk控制、仿真上采样、最小jerk系统辨识、误差敏感性分析与不确定性感知策略。

**📊 数据集**

使用了基于 IsaacSim 的并行仿真环境，随机生成的物体模型（长宽高、质量、静摩擦、动摩擦）以及 YCB 物体库（木块、芯片、瓶子等），共计约 12,288 条训练配置。

**📈 对比分析**

与基于模型的动态抓握投掷方法进行对比（Upright实现）。在仿真中 NP-Throw 的成功率约 99%，在真实环境中 60 个目标组合平均成功率 97%。传统方法在求解稳定性与局部最优方面表现不佳。

**⚠️ 局限性**

局限性包括：仅在平面投掷任务上验证，无法直接推广到三维空间；对物体尺寸、形状的鲁棒性仍有限；在极端距离/高度组合下需超过机器人速度限制或碰撞；对高质量摩擦不敏感但对动摩擦误差极度敏感。

---

## 452. GenONet: A Generative operator Network for High-Resolution Precipitation Nowcasting

**arXiv ID:** 2609.00544 | [PDF](https://arxiv.org/pdf/2609.00544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 453. AI Should Not Only Be Helpful. It Should Be Contingent. Artificial Intimacy, Sycophancy, and the Future of Social Learning

**arXiv ID:** 2609.00211 | [PDF](https://arxiv.org/pdf/2609.00211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 454. DramaChain Bench: An End-to-End Benchmark for Short-Drama Generation

**arXiv ID:** 2609.00646 | [PDF](https://arxiv.org/pdf/2609.00646v1)

**作者:** Haoyuan Shi `[一作]` (Hunyuan, Tencent), Richeng Xuan `[通讯]` (Hunyuan, Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了端到端的短剧生成基准，评估从脚本到完整短剧的六个阶段，并提供人工与自动化评测框架。

**💡 创新点**

创新点在于：①使用链式生成产物评估每个阶段；②统一5大评价轴和63维细粒度度量；③实现自动化评测与人工评测的高一致性，可对模型缺陷进行时空归因。

**🔧 技术方法**

使用多模态生成模型（文本 LLM、图像模型、视频模型）以及自研的 ViMax 生成管线；人工评测由专业评审员完成并对缺陷做空间时间标注；自动评测通过多轮工具调用生成评分。

**📊 数据集**

数据集为 20 部原创短剧、3 集连载共 60 集，包含剧本、分镜、关键帧图像、单镜头视频及完整短剧，共 5,785 个评测项目。

**📈 对比分析**

对比 22 个模型（9 文本 LLM、7 图像、6 视频）和 8 个仅自动评测的模型；平均 PLCC 0.918，自动评测可无人工成本复现人类排名，且发现上游缺陷会在后续阶段累积。

**⚠️ 局限性**

局限在于维度覆盖不完全（未覆盖所有质量维度）、数据来源仅为合成内容、无法评估商业表现或内容政策合规。

---

## 455. DUPIN: Attack Learning Is Still Needed! Demonstrating Few-Shot after Unsupervised Pretraining Is A Nimble Forensics Learner

**arXiv ID:** 2609.00259 | [PDF](https://arxiv.org/pdf/2609.00259v1)

**作者:** Chanwoo Bae `[一作]` (Purdue University), Xiangyu Zhang `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了基于学习的取证系统DUPIN，利用大规模无监督预训练的图注意网络对系统审计日志的关联图进行上下文学习，然后在少量标记攻击样本上进行微调，实现对APT攻击的检测与取证。

**💡 创新点**

创新点包括：①将大规模无监督预训练与少量样本的监督微调结合，首次在攻击取证中实现“预训练-少样本学习”工作流；②针对审计日志设计专属的词表提取与路径嵌入方法；③在预训练阶段采用掩码节点/边预测的自监督任务；④在攻击检测阶段使用图注意网络作为底层模型并在此基础上加入线性分类器。

**🔧 技术方法**

主要技术手段为：图注意网络（GAT）作为基础模型；自监督掩码预测（节点/边）实现预训练；子图抽取与图构造算法；词表动态提取与格式化词汇；少样本监督微调（few‑shot learning）；基于ROC、TPR/TNR、AUROC的性能评估。

**📊 数据集**

使用的数据集包括：DARPA Transparent Computing（TRACE）E3/E5 版本（Windows 与 Linux ），ATLAS（Windows）以及 Palantir（Linux）四个来源，共计 25 个 APT 攻击集。预训练使用 38–52 天的无攻击审计日志（约 7.3 TB）。

**📈 对比分析**

与基线比较时，DUPIN 在 25 个攻击集上取得了 21% 的 AUROC 提升，TPR 提升 8%，TNR 提升 17%。在 19/25 个攻击集上取得最高准确率，并在不同数据来源上均保持优势。对比自监督方法（AirTag、MAGIC）和其他 GNN（GIN、GCN）以及无监督检测方法，DUPIN 显著提高了检测准确性与误报率。

**⚠️ 局限性**

局限性：①仅能检测已见过的攻击类型，对全新或根本不同的攻击方案识别能力有限；②需要一定量的攻击样本进行微调，虽较少但仍需收集；③当前实现主要面向事后取证，未针对实时检测做优化；④图表示在某些攻击中会丢失时间顺序信息，导致对时间依赖性攻击的检测下降；⑤过度依赖预训练日志的质量与覆盖范围，若预训练数据不足或不具代表性，性能可能受限。

---

## 456. A Study of Hidden-State Optimization Order in Predictive Coding Networks

**arXiv ID:** 2609.00686 | [PDF](https://arxiv.org/pdf/2609.00686v1)

**作者:** Xueyuan Li `[一作]`, Danilo Vasconcellos Vargas `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文探究了使用预测编码(PC)方法训练多层感知机(MLP)网络，并与传统反向传播(BP)进行对比。

**💡 创新点**

创新点在于提出将PC与BP相结合的PC-BP训练策略，并引入固定权重的PC训练机制，展示其在小型数据集上的有效性。

**🔧 技术方法**

主要技术包括多层感知机网络结构、预测编码算法、固定权重PC以及结合BP的PC-BP算法。

**📊 数据集**

实验使用MNIST和CIFAR-10两个标准数据集。

**📈 对比分析**

通过在相同网络架构下比较各训练方法的分类误差，结果表明PC-BP在标准MLP上误差最低，在μMLP中亦保持较优表现。

**⚠️ 局限性**

局限性在于仅验证于小规模网络与数据集，未对更深层网络或大规模数据集进行测试，且固定PC的优势相对有限。

---

## 457. Staged Linguistic Seeding: Grounded Query Expansion for Verified-Unit QA in AI Contact Centers

**arXiv ID:** 2609.00844 | [PDF](https://arxiv.org/pdf/2609.00844v1)

**作者:** Hyeonseop Yoon `[一作]` (OpenMined), Jeong-Eun Park `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一套在 AI 接触中心（AICC）的 verified‑unit QA 系统，只从已验证的 QA 单元中检索答案，避免生成错误或无依据的回复，并通过成本感知路由将弱证据或不确定查询转交给人工；

**💡 创新点**

创新点在于离线 Staged Linguistic Seeding（SLS）方案：人工基于世界知识将代表性问题拆解成槽位，LLM 生成多样化查询变体并由轻量人机筛选，显著提升检索覆盖并彻底消除生成相关的 hallucination；

**🔧 技术方法**

使用的技术包括 BM25、BGE‑M3、Qwen3 等检索器的混合检索；SLS 生成与人工筛选；成本感知路由采用 GBDT；评估指标包括 unit‑recall@1、支持率（无不支持内容）和路由 macro‑F1；

**📊 数据集**

数据集为两家匿名企业的真实 QA 日志（Auto 90 单元、Elec 229 单元，共 319 verified units、7,947 查询变体），以及公开的 Quora Question Pairs 用于外部验证；

**📈 对比分析**

通过泄漏自由 hold‑out 变体检索实验比较 RAW（单一代表问题）与 SLS 扩展；SLS 在 hybrid 检索上提升至 0.881/0.930（+0.27/+0.34）；与 doc2query、HyDE 等自动扩展对比，SLS 超过 20% 召回；在相同证据下，free‑form RAG 的不支持内容率为 7–13%，verified‑unit 约 0%；

**⚠️ 局限性**

主要限制包括：需人工编写槽位 recipe，扩展成本和可重复性未知；评估仅覆盖 319 个单元，未验证大规模 FAQ 的可扩展性；路由标签基于主观 heuristic，精度有限；使用 LLM 评判不含人工校对，可能导致误判；

---

## 458. Automating Static Code Analysis Through CI/CD Pipeline Integration

**arXiv ID:** 2609.00676 | [PDF](https://arxiv.org/pdf/2609.00676v1)

**作者:** Zachary Wadhams `[一作]` (Montana State University), Clemente Izurieta `[通讯]` (Montana State University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个通用的自动化流程，将静态代码分析工具的输出聚合并自动提交到开发者熟悉的 issue-tracking 系统中，并在 GitLab 环境下对 SonarQube 进行演示。

**💡 创新点**

创新点在于提供了一个可跨工具、可自动化的控制脚本，利用 API 统一获取 SAST 结果、格式化为标准 issue 并可与 CI/CD 流水线无缝集成，同时引入质量门控功能。

**🔧 技术方法**

使用技术包括 CI/CD 流水线、Python 控制脚本、SonarQube 与 GitLab 的 REST API、JSON 数据交换与 Markdown 格式化。

**📊 数据集**

数据集主要来自组织 X 的内部代码仓库及 SonarQube 生成的漏洞与错误报告，没有公开数据集。

**📈 对比分析**

通过开发者反馈与案例评估，发现该流程对流水线运行时间影响不大，问题可视化显著提升，尽管未给出定量性能指标，但用户满意度和安全感提升明显。

**⚠️ 局限性**

局限性包括仅针对具备 API 的 GUI 静态分析工具验证，未覆盖 CLI 版工具；只在单一组织环境下测试，可能不具备广泛的可推广性；并且仍需处理误报导致的额外工作。

---

## 459. Algorithm-Hardware Co-Design of a Lightweight PCG Equalizer with a Fixed Step Size for Massive MIMO

**arXiv ID:** 2609.00890 | [PDF](https://arxiv.org/pdf/2609.00890v1)

**作者:** Junshuo Wang `[一作]` (University of Electronic Science and Technology of China), Jienan Chen `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种一阶固定步长的 Jacobi 预条件 BEM-PCG 等化器，利用一次更新即可恢复量化引起的削波失真。

**💡 创新点**

创新点在于：① 通过 Jacobi 预处理消除用户间功率耦合；② 推导一阶固定步长的精确下降性和最优松弛参数；③ 证明了在有限维下的条件数与收敛性保证，支持硬件友好的 𝒪(U) 前向数据通路。

**🔧 技术方法**

采用了基于 BEM 的二次恢复模型、Jacobi 预条件的共轭梯度（PCG）、一次固定步长更新、并在 55 nm CMOS 进行硬件综合。

**📊 数据集**

使用 3GPP CDL‑B 大规模 MIMO 信道模型，B = 2048、N_beam = 256、U = 32，混合 QAM 调度（15% 64‑QAM、40% 256‑QAM、45% 1024‑QAM）和 6‑bit ADC 量化。

**📈 对比分析**

与 6‑bit 理想动态步长 BEM‑PCG、BEM‑CG、BEM‑GS 以及 12‑bit MMSE、PIC‑MMSE 参考对比，实验表明在 32 dB SNR 下实现 8.558 bps/Hz，失真仅 0.5% 左右，并在功耗上预计可节约 289.1 mW。

**⚠️ 局限性**

局限性在于仅考虑一次更新，未证明多迭代收敛；性能评估受限于特定信道模型、近远功率范围和 6‑bit ADC 的假设；对动态步长的理论保证仍需在更广泛情形下验证。

---

## 460. TWIX: a Two-Stage Approach for End-To-End Named Entity Recognition and Relation Extraction

**arXiv ID:** 2609.00832 | [PDF](https://arxiv.org/pdf/2609.00832v1)

**作者:** Marco Martinelli `[一作]`, Laura Menotti `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

介绍并说明CEUR-WS文章模板的使用方法与功能。

**💡 创新点**

提出统一的排版规范和易于访问的元数据提取功能。

**🔧 技术方法**

使用LaTeX类文件和相关宏包，如cite、geometry、graphicx、hyperref等。

**📊 数据集**

无特定数据集。

**📈 对比分析**

未进行实验性比较，主要通过模板示例展示效果。

**⚠️ 局限性**

模板不允许修改，使用者只能通过参数调整，限制了灵活性。

---

## 461. ABSE-NET: A Lightweight Neural Model for Active Binaural Speech Enhancement in Open-Fit Hearing Aids

**arXiv ID:** 2609.00966 | [PDF](https://arxiv.org/pdf/2609.00966v1)

**作者:** De Hu `[一作]` (Inner Mongolia University), Qintuya Si `[通讯]` (Inner Mongolia University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出 ABSE-NET，一种在开口式助听器中结合 BMVDR 与轻量级神经网络的主动语音增强框架。

**💡 创新点**

创新点在于通过后置轻量化神经网络实现同时抑制声道泄漏并补偿 BMVDR 失真，无需体内麦克风。

**🔧 技术方法**

采用 BMVDR 波束former、RMB‑Conv1D、F‑TDL（频时依赖学习）块、ConvAtt 关注模块以及 SI‑SDR/ STOI 损失的轻量级 LNN。

**📊 数据集**

使用 Librispeech + NOISEX‑92 语音与噪声，结合 Hearpiece HRIR 生成 24 方向的双耳混合数据。

**📈 对比分析**

与闭式助听器 BMVDR、FxMWF、DeepANC、ASE‑TM 等基线对比，ABSE‑NET 在 SI‑SDR 约 9.9 dB、PESQ 3.63、STOI 0.955，参数 0.112 M、FLOPs 0.184 G，表现优于或相当于其它方法。

**⚠️ 局限性**

局限在于仍需依赖离线训练与错误麦克风收集，未验证极端动态环境下的实时性能与能耗。

---

## 462. Compile, Don't Memorize: A Context Compilation Architecture (CCA) for In-Context Learning

**arXiv ID:** 2609.00759 | [PDF](https://arxiv.org/pdf/2609.00759v1)

**作者:** Jinhu Qi `[一作]` (Chinese University of Hong Kong), Irwin King `[通讯]` (Chinese University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一种 Context Compilation Architecture（CCA），将长文本上下文编译成可执行的中间表示（IR）并自动生成验证器，提升基于规则的长上下文推理准确率。

**💡 创新点**

核心创新在于将上下文转化为固定槽位的 JSON IR，结合可执行的代码验证器和违规驱动的纠错循环，实现结构化上下文编译而非单纯的 read‑and‑reason；并在 Harness Engineering 框架下将 LLM 固化为工具。

**🔧 技术方法**

使用大型语言模型（Kimi K2.5、GLM‑5、DeepSeek‑V3.2、Qwen3‑Next‑80B）进行四阶段推理；通过编译器提取规则、实体、术语等；代码生成器自动合成 Python 验证器；推理阶段分为 Draft 与 Correction。

**📊 数据集**

主要使用 CL‑bench（1899 任务、4 个领域、18 子类）进行评估，并在 LongBench‑v2（503 题）做跨基准检验；实验对比 Vanilla、ReadAgent‑P、Ctx2Skill。

**📈 对比分析**

与三种基线对比，CCA 在所有基模型上均获得最高整体通过率（Kimi K2.5 +6.0pp，GLM‑5 +5.1pp，DeepSeek‑V3.2 +2.7pp），且在规则密集子类表现突出；在 LongBench‑v2 的多文档/多轮子集也显著提升。

**⚠️ 局限性**

局限性包括：1) 对小激活模型（Qwen3‑Next‑80B）提升有限，表明模型容量限制；2) 仅对规则/结构化文本有效，对开放式创意任务不利；3) 编译器误解析率 <1% 但仍需更稳健；4) 需要多评判者验证，当前仅使用单一 GPT‑5.1 判定。

---

## 463. Nash Core in Multiwinner Election

**arXiv ID:** 2609.00486 | [PDF](https://arxiv.org/pdf/2609.00486v1)

**作者:** Ashish Goel `[一作]` (Stanford University), Chenghan Zhou `[通讯]` (Stanford University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了“Nash核心”概念，证明在满足轻度正则性假设的分数式选举中存在，随后利用该结构与强制支付方案等价关系，证明分数式Nash核心蕴含核心稳定性，并在最多八名等权选民的离散选举中用计算机辅助分类证明离散核心存在；此外提出弱Nash核心作为可有效验证的上界，并给出启发式算法在真实投票数据上高效求解核心。

**💡 创新点**

创新点在于：①引入候选人加权的Nash社会福利最大化（candidate‑weighted Nash SW），形成Nash核心；②证明分数式Nash核心与比例支付（Lindahl‑type）等价；③通过精细的分数化简与价格饱和证书，完成八选民离散核心的完整计算机证明；④提出弱Nash核心作为可验证的离散核心下界，开辟了探索无约束离散核心的思路。

**🔧 技术方法**

技术手段包括：基于凸优化的KKT条件和对偶性；构造可解的凹性博弈实现Nash核心的存在性；利用Lindahl均衡理论解释支付结构；使用离散核算与整数规划的NP‑hard性证明；以及对八行残差矩阵进行全枚举与证书检验的计算机辅助证明；启发式部分采用迭代权重更新与局部搜索求解分数式与弱Nash核心。

**📊 数据集**

实验使用了Pabulib公开投票数据集，共约1100个实例，其中100个为大规模（>3000选民、50候选人）数据，其余为中小规模数据，用以检验启发式算法的实用性。

**📈 对比分析**

实验结果显示，启发式算法在大多数实例上可在不到一分钟内得到分数式Nash核心或弱Nash核心，而最复杂实例需5–10分钟；相比传统的PAV或MES规则，Nash核心在理论上保证核心稳定性，实验上亦能高效求解，验证了其可行性和效率。

**⚠️ 局限性**

局限性包括：①核心存在性仍未在一般无约束离散选举中得到证明；②判断离散核心成员资格为co‑NP‑hard，启发式算法缺乏收敛性保证；③分数式Nash核心的存在性仅在轻度正则性假设下成立，可能不适用于所有实例；④八选民结果仅限于等权选民，未覆盖更一般的权重或更多选民情况。

---

## 464. Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO

**arXiv ID:** 2609.00925 | [PDF](https://arxiv.org/pdf/2609.00925v1)

**作者:** Prakhar Gupta `[一作]` (University of Michigan), Vaibhav Gupta `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在同一 instruction‑tuned 起始检查点上，系统比较了九种后训练方法（GRPO、SFT、DPO）在 Qwen、Llama、Phi 等多模型、不同规模上的对冲突提示的上下文 grounding 效果，并在训练前估计并验证了起始模型的 grounding 方向，证明大部分提升来自预存在模型中的机制。

**💡 创新点**

首次在同一起始检查点下跨模型、跨规模系统地比较多种后训练手段的 grounding 效果，并通过估计并干预 grounding 方向，验证了机制重用的因果作用，显示后训练提升主要源自模型已有的 causal 头集与方向。

**🔧 技术方法**

使用差分平均法（DiM）估计 grounding 方向、激活 steering、头集发现与干预、GRPO、SFT、DPO 训练，以及 CounterFact/ConFiQA/FaithEval 的冲突协议评估来衡量 grounding 更新率。

**📊 数据集**

训练数据来自 HotpotQA；评估数据包括 CounterFact、ConFiQA、FaithEval；能力检查使用 MMLU、ARC 等通用任务。

**📈 对比分析**

通过配对 McNemar 检验对 grounding 更新率进行比较；结果显示 GRPO 贡献小，冲突 SFT 中等，DPO 在对应分布上几乎达到极限，并且跨规模、跨家族保持相对稳定的提升。

**⚠️ 局限性**

主要局限在于使用词汇包含度指标与 LLM 判定的低一致性；机制审计仅覆盖 causal 头集而非完整电路；实验仅限于起始 instruction‑tuned checkpoint 上的机制重用；GRPO 结果受 low context‑answer 采样覆盖率限制。

---

## 465. Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?

**arXiv ID:** 2609.00482 | [PDF](https://arxiv.org/pdf/2609.00482v1)

**作者:** Qiaoyuan Zheng `[一作]` (ETH Zurich), Yiqu Yang `[通讯]` (ETH Zurich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

审计了跨家族模型在近一百分比差距排行榜上的稳健性，利用低-DIF子测验检验项目组合对排名的影响。

**💡 创新点**

创新点在于结合无家族标签的光谱MIRT近似、所有权分离交叉拟合与低-DIF锚点，以及与匹配随机子测验的对照，系统揭示近一百分比差距的排名对项目选择的敏感性。

**🔧 技术方法**

使用光谱多维项目反应理论（MIRT）、差异项目功能（DIF）分析、交叉拟合、配对逆序率统计、Bootstrap重构和匹配随机子测验等技术。

**📊 数据集**

所用数据集包括 MMLU-Pro、BIG‑Bench Hard、MMLU、HellaSwag 和 WinoGrande。

**📈 对比分析**

通过将低‑DIF锚点分数与全分数对比，统计满足 |差距|≤1% 的跨家族模型对中逆转率；四个基准中逆转率达 30.9–47.1%，显著高于匹配随机子测验的中位数（差距 16.9–28.6%，p＝0.001）。

**⚠️ 局限性**

局限性包括家族标签仅为模型名称推断，光谱MIRT近似线性且可能忽略非线性效应，方法仅适用于静态封闭基准，且语义审计覆盖面有限。

---

## 466. Potential-Guided Particle Steering for Negation-Constrained Dexterous Grasping

**arXiv ID:** 2609.00555 | [PDF](https://arxiv.org/pdf/2609.00555v1)

**作者:** Geonho Kim `[一作]` (Chung-Ang University), Jongmin Lee `[通讯]` (Chung-Ang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种仅在推理阶段即可处理负向约束的语言驱动灵巧抓取方法

**💡 创新点**

不需要负向示例训练，利用无监督的推理时引导与粒子滤波实现对禁止区域的避让

**🔧 技术方法**

采用流匹配的Diffusion Transformer、基于分类器无监督指导（CFG）的垂直梯度引导、Sequential Monte Carlo（SMC）重采样与噪声保留的“churn”步骤

**📊 数据集**

基于DexGYSNet的正向抓取数据和新构建的NegGrasp基准（包含正负指令对）

**📈 对比分析**

与DexGYS、DextER等基线比较，NegGrasp上违约率从57.9%下降到17.2%，约束兼容抓取成功率（CSR）提升至61.5%，物理可执行率（TSR）也保持相近或略有提升

**⚠️ 局限性**

主要局限是对参数设置敏感（如CFG强度、重采样时机、粒子数）且仅在单一手抓取任务上验证，复杂环境或多手协作尚未探索

---

## 467. GazeTune: Facilitating Precise Gaze-Driven Interactions with Cascaded Touch Input

**arXiv ID:** 2609.00716 | [PDF](https://arxiv.org/pdf/2609.00716v1)

**作者:** Jina Kim `[一作]` (KAIST), Sang Ho Yoon `[通讯]` (KAIST)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 GazeTune 的 Cascaded Gaze‑and‑Touch 技术，用眼动快速定位并用智能手表触摸进行细微校正，实现了 XR 环境下精确且稳定的拖拽交互。

**💡 创新点**

创新点在于将粗眼动定位与触摸细微校正串联为单一连续流程，采用锁定机制与动态眼动范围门控，使得用户仅需一次手势即可完成从选取到拖拽的完整操作，显著提升了移动场景下的精准度与稳定性。

**🔧 技术方法**

使用技术包括：眼动追踪与滤波、手表触摸输入、锁定（Cursor Lock）与范围（Gaze Scope）门控、动态增益控制、三维视角映射、并在 Unity/Meta SDK 中实现交互流程。

**📊 数据集**

实验使用自行设计的 Fitts’ Law 风格拖拽任务数据，共 20 名参与者、3 种技术（GazeTap、GazePinch、GazeTune）、2 种运动条件（静止/跑步机）和 3 个目标幅度，总计 3,960 条试验记录；未使用公开数据集。

**📈 对比分析**

通过与 GazeTap（眼动+点选）和 GazePinch（眼动+捏合）进行重复测量 ANOVA 对比，结果显示 GazeTune 在移动条件下错误率显著降低（约 4.24% vs. 37.12% / 9.70%），拖拽时间与 GazePinch 相当甚至略优，整体完成时间与 GazeTap 相近，证明了在保持速度优势的同时显著提升了精度与稳健性。

**⚠️ 局限性**

局限性包括：实验仅在跑步机模拟的平稳步行速度下进行，未测试更高速或真实户外移动；深度固定为 2 m，未探究不同深度与相机参考框架下的表现；手表触摸表面边缘可能导致操作局限；缺乏长期使用与多模态扩展（如触摸按压、手部触摸等）的评估。

---

## 468. A Closed-Loop Evaluation of Capability Loss and Recovery in Compressed Driving Policies

**arXiv ID:** 2609.00718 | [PDF](https://arxiv.org/pdf/2609.00718v1)

**作者:** Ahmad Alfan Alfian Irfan `[一作]` (Universitas Muhammadiyah Yogyakarta), Mansur Arief `[通讯]` (King Fahd University of Petroleum and Minerals)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对基于Belief-State的视觉-运动驾驶策略进行结构化剪枝、知识蒸馏、整数量化与FP16降精度等压缩阶段的闭环评估，探究压缩对驾驶能力的影响。

**💡 创新点**

提出阶段性闭环评估框架，揭示剪枝导致能力丧失、蒸馏恢复取决于训练覆盖范围、整数量化再次失效等行为转变。

**🔧 技术方法**

使用PPO在Gym‑Duckietown中训练Belief‑State策略；随后依次执行结构化剪枝、知识蒸馏、后训练量化（PTQ）、量化感知训练（QAT）以及FP16 cast。

**📊 数据集**

使用Gym‑Duckietown仿真环境，包含5个驾驶课程（C0–C4）以及YOLO‑11n与MobileNetV3‑small进行目标检测与回归。

**📈 对比分析**

通过八个随机种子、40条评估轨迹，采用任务完成、碰撞、停线遵守等多指标阈值进行比较；发现压缩后某些策略失效，FP16保持性能，而INT8在停止课程失效。

**⚠️ 局限性**

仅在单一小规模策略、单一仿真平台、特定压缩工具链上验证，缺乏对更大网络、真实道路或不同量化实现的泛化；评估标准非标准化。

---

## 469. On the Human and Computer Alignment of Attribute-Based Music Matches

**arXiv ID:** 2609.00987 | [PDF](https://arxiv.org/pdf/2609.00987v1)

**作者:** Roser Batlle-Roca `[一作]` (Universitat Pompeu Fabra), Yuki Mitsufuji `[通讯]` (Sony AI)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究通过属性级三元组实验，探讨音乐复制检测的感知评估，并构建了跨人工与 AI 生成音乐的 MATCHA 数据集。

**💡 创新点**

创新点在于将音乐相似度拆解为旋律、和声、节奏、声部、音色五个属性进行人类评估，首次提供跨人工与 AI 生成作品的属性级标注数据；同时评估多种计算相似度指标与人类判断的对齐。

**🔧 技术方法**

使用 Triplet forced‑choice 任务、MiRA 框架下的 CoverID、KL、CLAP、DEfNet 四个相似度指标，以及音频特征提取工具（madmom、Essentia 等）。

**📊 数据集**

采用人类作曲与抄作、版本对、Stable Audio Open 生成的音乐及媒体报道中的潜在剽窃案例，共 300 个三元组（含 150 人工、150 AI 生成）。

**📈 对比分析**

通过 Fleiss κ 与 Kendall τ 评估一致性，发现属性级人类一致性良好（旋律最高），但计算指标与人类在不同属性间仅部分对齐，整体性能仍有限。

**⚠️ 局限性**

局限包括：样本主要为西方音乐、属性拆分仍受主观解释影响、三元组设计偏向强相似导致泛化受限、未涵盖非西方传统与其他属性（如歌词、情感）。

---

## 470. FTU-Seek: Foundation Model-Guided Hard-Negative Learning for Sparse Functional Tissue Unit Segmentation

**arXiv ID:** 2609.00704 | [PDF](https://arxiv.org/pdf/2609.00704v1)

**作者:** Zonghao Liu `[一作]` (Fujian Medical University), Jingfeng Liu `[通讯]` (Fujian Medical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出并实现了一种名为 FTU-Seek 的框架，用于在大规模全切片图像中对稀疏的功能组织单元（FTU）进行精确分割，支持 TLS、血管和腺体三类目标。

**💡 创新点**

创新点在于将病理基础模型（PFM）提取的多层特征用于训练一个 Patch‑Level 二分类器，先对缺失目标的区域按“目标可能性”排序，再通过静态 TopK 策略挑选最具干扰性的负样本，构建紧凑而信息量高的分割训练集，实现稀疏目标分割的高效训练。

**🔧 技术方法**

技术方案包括：冻结的 UNI 病理基础模型提取多深度特征，基于多层特征的二分类器；静态 TopK 负样本选取；多尺度 Transformer 解码器与 Dice‑Focal 损失的分割网络；五折交叉验证与内部测试集的评估；以及在 TCGA 数据集上的外推性验证。

**📊 数据集**

使用了 3 个内部 FCH 组（TLS：10 例，血管：10 例，腺体：10 例）进行训练与验证，并在 30 张独立的 TLS 组上评估；外部实验基于 TCGA READ、ESCA、STAD、LIHC 和 PRAD 共计 146、148、326、337、401 例，用于下游表型关联分析。

**📈 对比分析**

实验将 FTU-Seek 与正样本仅训练、全样本训练、随机负样本采样和匹配随机 TopK 采样进行对比；结果显示 FTU-Seek 在保持相近 Dice 的同时，训练工作量显著降低；例如 TLS 的 Dice 从 76.69%（Top1000）提升到 85.12%（All‑tissue）时，仅保留 27.6% 的训练数据；血管与腺体的表现也在相同负样本预算下接近全样本训练水平。

**⚠️ 局限性**

局限性包括：对不同 FTU 的效果差异明显，稠密目标（如腺体）不易从负样本选择中获益；内部测试集样本量极小，仅用于描述性比较；未使用后处理或多尺度融合策略；以及对跨机构、多制样本的泛化性仍需进一步验证。

---

## 471. Risk-Aware Decision-Making for Autonomous Overtaking: A World Model-Based Mixture-of-Experts Framework

**arXiv ID:** 2609.00385 | [PDF](https://arxiv.org/pdf/2609.00385v1)

**作者:** Yongzhi Liu `[一作]` (Southeast University), Weichao Zhuang `[通讯]` (Southeast University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于世界模型的风险感知混合专家（WM‑RMoE）规划框架，用于高速公路自动超车；

**💡 创新点**

创新点在于：①通过世界模型实现多步轨迹级风险评估，提升安全约束的前瞻性；②设计风险感知门控网络，根据轨迹不确定性动态调配长期学习专家、短期记忆专家与基于规则的安全专家；③采用GMM‑CCEM优化器保留多模态驾驶行为，避免模式平均导致的行为退化；

**🔧 技术方法**

使用技术包括：Recurrent State‑Space Model（RSSM）世界模型、SAC‑Lag强化学习、Mixture‑of‑Experts门控、Gaussian Mixture Model + Cross‑Entropy Method、Lagrangian安全约束、深度神经网络门控与知识蒸馏；

**📊 数据集**

数据集/仿真环境为开源的 highway‑env 四车道高速公路仿真平台，使用 IDM‑MOBIL 车辆模型生成多种交通密度（低、中、高）随机场景；

**📈 对比分析**

与 IDM‑MOBIL、SAC、SAC‑Lag、CVPO、DreamerV3、SafeDreamer 等基线对比，WM‑RMoE 在低密度下 100% 成功率、最高奖励；中、高密度下成功率分别为 97.8%/91.6%，奖励、平均速度显著高于所有对手，累计成本最低，显示出更优的安全-效率平衡；

**⚠️ 局限性**

局限性包括：①世界模型在极端场景下仍存在预测偏差，可能导致安全约束失效；②多专家门控与GMM‑CCEM 的计算量较大，实时性能依赖硬件加速；③对超车等高度动态场景的通用性需在真实车辆上进一步验证；

---

## 472. Collaboratively Eliciting Gestures for Geospatial Data Exploration on an MSE with Tangibles and Styluses

**arXiv ID:** 2609.00007 | [PDF](https://arxiv.org/pdf/2609.00007v1)

**作者:** Karen Penaranda Valdivia `[一作]` (Toronto Metropolitan University), Ali Mazalek `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 2522 | [OpenAlex ID](https://openalex.org/A5072480471)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在多面显示环境中，使用五角形可旋转触控圆盘（tangible）和笔（stylus），邀请36名具备不同地理空间建模经验的参与者，针对16个迁移相关的地图可视化任务进行协作式手势激发（elicitation）研究，收集并归纳出166种独特交互手势。

**💡 创新点**

首次系统性地将可交互实体与笔在协作式地理空间探索中结合，使用与迁移驱动因素相关的五角形形状进行语义嵌入；同时从专家、混合与新手三种专业层级的角度评估手势共识和交互效率，为跨学科迁移研究提供可落地的交互原型。

**🔧 技术方法**

技术实现包括：多面显示系统（55英寸 MultiTaction Cell、9块竖屏与2块桌面显示）、基于ESP32‑S3的可旋转触控圆盘、笔式触控笔、基于 Mapbox 的地图可视化接口、手势编码与共识评估（MC、CDR）、以及 R 统计环境下的卡方检验与方差分析。

**📊 数据集**

使用了基于 Mapbox 的截图数据集，包含加纳1985-2010年间的净迁移、降水量、金矿开采与冲突层信息，构建16个面向迁移研究的任务参照（referents）。

**📈 对比分析**

通过比较不同输入模态（单一 tangible、单一 stylus、多模态组合）以及不同专业层级（专家、混合、新手）的手势提议，利用共识度指标 MC 与 CDR，并结合卡方检验和 ANOVA 分析，发现单模态手势（尤其 tangible）获得最高共识；混合专业对组产生更多匹配且更简短的提议；专家提议更简洁，但整体性能差异不显著。

**⚠️ 局限性**

局限性包括：样本主要为20-30岁本科/研究生，缺乏老年与跨学科多元背景；任务仅覆盖16个相对低层级、单一步骤的任务，缺少更复杂的多步骤交互；多模态手势使用率低，可能受任务设计限制；仅为原型实验，未验证在真实迁移研究工作流程中的效果。

---

## 473. Inverse Rig Optimization from Line Drawings

**arXiv ID:** 2609.00732 | [PDF](https://arxiv.org/pdf/2609.00732v1)

**作者:** Zihao Zhu `[一作]` (University of Tokyo), Yuki Koyama `[通讯]` (University of Tokyo)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种基于手绘轮廓线反向推算角色动画装配参数的方法，并将其应用于面部表情、道具姿态、视角特化和夸张透视等场景。

**💡 创新点**

创新点在于：①在高层控制器级别实现装配器反演；②采用可微分的神经代理（NeuralRig）取代不可微的实际装配器；③使用两阶段层次匹配将用户绘制的线与渲染轮廓对齐，从而实现一次性、线条驱动的关键帧设定。

**🔧 技术方法**

主要技术包括：可微分的多层感知机神经代理、MVP投影、层次化（线级+点级）匹配、屏幕空间误差（法线投影与欧氏距离）损失、Adam 优化、控制影响映射（Rig Influence Map）以及动态时间对齐。

**📊 数据集**

使用了多款公开的卡通与动漫风格角色模型（Blender Studio、miHoYo、DillonGoo Studios），并为每个角色训练相应的 NeuralRig 代理；数据集由这些模型的姿态样本与手绘轮廓组成。

**📈 对比分析**

对比方式主要是与传统手工关键帧设定的可视化对比；在实验中每个关键帧的优化耗时仅几秒到十几秒（匹配时间占比极低），并且在多种场景下都能成功匹配用户绘制的轮廓。

**⚠️ 局限性**

主要局限包括：①二维线条缺乏深度信息导致姿态歧义；②固定轮廓路径在大幅变形时可能失效；③神经代理的近似误差会影响最终姿态精度；此外，当前方法仅针对单帧、静态装配，未处理时间一致性或次级物理动画。

---

## 474. Auditing Harness Tampering in Self-Improving Agents

**arXiv ID:** 2609.00069 | [PDF](https://arxiv.org/pdf/2609.00069v1)

**作者:** Xing Wang `[一作]` (University of Electronic Science and Technology of China), Jie Shao `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7723 | [OpenAlex ID](https://openalex.org/A5072350518)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种用于审计自我改进代理的 harness 破坏检测框架，并构建了相应的标注数据集。

**💡 创新点**

创新点在于引入了两轴分类法（功能角色与义务）来细化破坏类型，并通过 fault‑seeding 在真实改进轨迹中生成匹配的破坏–安全编辑对。

**🔧 技术方法**

主要技术包括基于 LLM 的提示与 LoRA 适配的审计模型、fastText 词向量分类器，以及对比实验中的各种公开代理系统。

**📊 数据集**

使用的数据集由 1,765 条分类样本和 1,801 条定位样本组成，来源于 HyperAgents、DGM、AFlow 的真实迭代变更，并在公开代理（ADAS、HyperAgents 等）上进行评估。

**📈 对比分析**

在分类任务上，Claude Opus 5 取得最高 90.4% 准确率；在定位任务上，GPT‑5.6 Sol 和 Claude Opus 5 的召回率分别达 86.7% 与 83.4%，但均存在约 10% 的误报，整体成本与准确率呈折衷。

**⚠️ 局限性**

局限性包括：对义务与功能角色的归属准确率低；定位召回与误报之间难以平衡；评估仅覆盖已公开的有限代理系统，未验证跨平台普适性；并且缺乏可直接部署的高效审计工具。

---

## 475. Centrality Measures in Temporal Networks: A Critical and Comparative Survey

**arXiv ID:** 2609.00011 | [PDF](https://arxiv.org/pdf/2609.00011v1)

**作者:** Aksa Urooj `[一作]` (National Institute of Technology), Iqra Altaf Gillani `[通讯]` (National Institute of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对时间网络中的中心性度量进行了系统综述，并通过实验评估了多种中心性方法的影响力识别效果。

**💡 创新点**

创新点包括：①提出了按“影响来源”划分的功能性时间网络中心性分类法；②将传统中心性与专门设计的时间网络中心性统一到同一框架；③在三种真实数据集上进行多维度比较，给出方法适用场景与性能权衡。

**🔧 技术方法**

主要技术手段包括：时间网络模型与时间约束路径；SI/SIS 传播模型评估感染率；Kendall τ 相关系数衡量排名一致性；复杂度与运行时分析；以及基于梯度的快速近似更新算法。

**📊 数据集**

使用的数据集有：医院病房动态接触网络（29 个人+46 工作者）、CollegeMsg 在线消息网络（1,899 用户）、比特币 OTC 信任交易网络（5,881 交易者）。

**📈 对比分析**

比较方法：①选取每种中心性前 5% 节点作为种子，在 SI 与 SIS 模型下计算平均感染比例；②计算各中心性排名的 Kendall τ 相关矩阵；③记录每个时间步的运行时间。结果表明：TWC 与 TSDC 在感染传播方面表现最佳；TSDC 运行最快；Supracentrality 与 EffC 计算开销最大；不同方法在不同网络上表现差异显著，需根据网络特征与计算预算做选择。

**⚠️ 局限性**

局限性：①光谱及全局路径方法在大规模或多层网络上计算成本高；②许多方法假设离散时间步或固定时间窗口，难以直接处理连续流式事件；③参数敏感性（如记忆衰减、传播概率）需要手工调优；④缺乏统一评测基准与可复现的实验框架。

---

## 476. AKRASIA: Stealthy Backdoor Attack on Reasoning-based Code LLMs

**arXiv ID:** 2609.01023 | [PDF](https://arxiv.org/pdf/2609.01023v1)

**作者:** Chou Jin Chua `[一作]` (Singapore University of Technology and Design), Ezekiel Soremekun `[通讯]` (Singapore University of Technology and Design)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种针对推理型代码LLM的隐蔽反向门攻击，利用上下文学习和模型不忠实性在推理过程中植入恶意代码，且在最终输出中隐藏触发器与推理步骤。

**💡 创新点**

首次使用代码级触发器，并在输出中完全隐蔽触发器与推理，能突破当前最先进的防御与人工审查，展示了极高的隐蔽性与攻击成功率。

**🔧 技术方法**

结合上下文学习（ICL）、模型不忠实性技术、代码级触发器生成与推理隐藏机制，构成完整的攻击流程。

**📊 数据集**

在 CodeMMLU、LiveCodeBench、CruxEval 等编程相关数据集上进行评估。

**📈 对比分析**

与现有 BadChain、BadCodePrompt 等攻击进行对比，实验结果显示在六款SOTA推理LLM上平均攻击成功率（ASR）高达 99.34%，在 14/18 防御设置下仍保持 98.82% ASR，且在大多数模型上准确率保持在 90% 以上。

**⚠️ 局限性**

受限于部分模型的内部 guardrail 影响，导致如 GPT‑5.5 的 ASR 下降；实验仅覆盖公开数据集与指定模型，缺乏对更广泛模型的通用性验证。

---

## 477. Structure-Behavior Coalescence and the Limits of Traditional Systems Theory

**arXiv ID:** 2609.00042 | [PDF](https://arxiv.org/pdf/2609.00042v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 478. From Terminology to Diagrams: Visual-Instruction Generation for Scientific Diagram Understanding

**arXiv ID:** 2609.00948 | [PDF](https://arxiv.org/pdf/2609.00948v1)

**作者:** Raul Ortega `[一作]` (Expert.ai), José Manuel Gómez-Pérez `[通讯]` (Expert.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了SciGram大规模科学图表视觉指令数据集，并基于课程术语驱动的框架生成图表、标题与多选题，随后使用这些数据对视觉‑语言模型进行微调；

**💡 创新点**

创新点在于用教材术语提取概念，生成原子事实并检索对应图表，形成高覆盖度且低成本的图表视觉指令数据集；

**🔧 技术方法**

采用语言模型（如LLaMA3、Qwen2‑VL‑7B）生成原子事实、标题与MCQ，结合CLIP/SIGLIP视觉编码器、LLaVA架构以及LoRA微调等技术；

**📊 数据集**

使用194,071张科学图表与1.4M视觉指令（SciGram-Align、SciGram-VIT、SciGram-M3）以及TQA、SQA、AI2D、OpenBookQA等评测数据集；

**📈 对比分析**

在TQA、SQA、AI2D等图表问答基准上，SciGram微调后的模型在图表推理子任务上提升多达16个百分点，刷新SOTA，甚至与GPT‑4o、Pixtral 12B等前沿模型持平或领先；

**⚠️ 局限性**

局限在于数据噪声较高（约24%非图表、标签不一致），部分MCQ可通过文本先验回答，对图表过滤与一致性验证仍需改进，且依赖大模型与算力资源。

---

## 479. AI Morbidity and Mortality: A Framework for Clinical AI Failure Review

**arXiv ID:** 2609.00076 | [PDF](https://arxiv.org/pdf/2609.00076v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 480. RingMoClaw: An Experience-Inspired Multi-Agent Framework for Self-Evolving Research in Remote Sensing

**arXiv ID:** 2609.00814 | [PDF](https://arxiv.org/pdf/2609.00814v1)

**作者:** Kaiyue Kang `[一作]` (Aerospace Information Research Institute Chinese Academy of Sciences), Xian Sun `[通讯]` (Aerospace Information Research Institute Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向遥感视觉任务的自我进化多智能体框架RingMoClaw，自动化完成研究迭代流程；

**💡 创新点**

创新点在于引入闭环研究分支、独立异构Critic质量控制和双流动态经验总线，结合外部知识检索与内部实验经验，实现策略演进的自动化、可解释和高效；

**🔧 技术方法**

使用大型语言模型（GLM-5 Turbo与Minimax 2.7）进行计划生成、执行与评估，结合RingMo等遥感基础模型、数据增强、训练策略优化、模型融合与模块创新等技术；

**📊 数据集**

在四个遥感视觉基准上评估：FAIR1M（目标检测）、NWPU-RESISC45（场景分类）、iSAID（语义分割）和LEVIR-CD（变化检测）；

**📈 对比分析**

与AutoResearch和AutoResearchClaw等现有自动研究框架对比，RingMoClaw在目标检测上提升1.84% mAP_50、场景分类提升1.79%，语义分割提升2.70%，变化检测提升1.01%，同时演化步骤减少约40%至60%；

**⚠️ 局限性**

局限性包括对高性能GPU资源的依赖、实验过程中的通信与模型调用开销、当前仅覆盖二维视觉任务，缺乏对多模态与时序遥感任务的扩展与更高效的候选验证方法。

---

## 481. ADAPT: Agile Diffusion Action Priors for Robust and Steerable Online Text-Driven Humanoid Control

**arXiv ID:** 2609.00677 | [PDF](https://arxiv.org/pdf/2609.00677v1)

**作者:** Yan Wu `[一作]` (ETH Zurich), Siyu Tang `[通讯]` (ETH Zurich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 ADAPT，一种端到端的文本驱动类人机器人全身控制框架，可在实时闭环中实现多技能执行、快速命令切换，并支持在冻结的扩散策略上进行下游任务调优。

**💡 创新点**

创新点包括：①基于细粒度技能级的文本条件扩散动作先验；②在冻结的扩散策略上叠加轻量残差强化学习模块，用以提升长时程鲁棒性和平滑过渡；③通过噪声空间调节的轻量 RL 模块，实现对同一冻结先验的可控下游任务适配。

**🔧 技术方法**

使用技术：扩散模型（DDIM）+ 变形器解码器，文本条件通过冻结的 CLIP 文本编码器；残差 RL（PPO）在冻结扩散策略上学习校正动作；噪声空间调节 RL（PPO）用于任务导向的噪声生成；物理仿真和增量采样实现实时闭环。

**📊 数据集**

数据集：将 AMASS 人类运动重映射到 Unitree G1 并在仿真中跟踪，结合 BABEL 的帧级文本标签，形成状态‑动作‑文本对；采用域随机化提升泛化。

**📈 对比分析**

与两阶段管道（DART+跟踪、TextOP）以及端到端 CVAE 方法（LangWBC）比较；在 130 条提示的 20 s 交互跑测中，ADAPT 的成功率 0.984，动作平滑度和脚滑率显著低于基线；在目标达成任务中，噪声调节 RL 将跌倒率从 34.7% 降至 2.9%。

**⚠️ 局限性**

局限性：残差校正可能过度偏向安全站姿，导致高动态动作的表现受限；模型规模与推理延迟之间的折衷，当前采用小模型和两步 DDIM 约 2 ms，较大模型可能提升性能但需进一步加速技术。

---

## 482. A hybrid quantum-classical neural network for learning to route

**arXiv ID:** 2609.00489 | [PDF](https://arxiv.org/pdf/2609.00489v1)

**作者:** Marcus Rolf Peter Ritt `[一作]` (Instituto de Pesquisas Eldorado), Fernando Augusto Caletti de Barros `[通讯]` (Instituto de Pesquisas Eldorado)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究将经典注意力网络的编码器中参数量大的 feed-forward 层替换为四量子比特的量子神经网络（QNN），并在容量受限的车辆路由问题（CVRP）上训练混合量子-经典模型

**💡 创新点**

创新点在于证明即使大幅压缩模型参数（56.6%）也能保持与原始经典模型相近的解质量，展示了小型 QNN 在路由学习中的可行性与压缩效能

**🔧 技术方法**

采用经典注意力模型（Kool 等 2018）作为基准，构建 4 qubit QNN 并使用变分 ansatz 训练；整体使用 REINFORCE + 贪婪/采样解码、Adam 优化、量子电路模拟与反向传播

**📊 数据集**

随机生成的二维欧氏实例集：10、20、50、100 顶点，需求 1–9，容量分别为 20、30、40、50，测试集 1,000 个新实例

**📈 对比分析**

将混合模型与原始经典注意力模型在相同训练预算下比较，并与五个经典启发式（CW、RCW、GOT、LKH3、RSW）对齐；混合模型在所有规模下与经典模型相差 ≤2%，但训练时间约 8.4n 分钟；与 LKH3 相比仍略逊，但优于 CW/RCW，且在相对偏差 r 上保持竞争力

**⚠️ 局限性**

局限包括训练样本量有限，未与所有等效经典压缩模块系统性对比；未评估更大规模或非欧氏实例；QNN 在量子硬件上的可扩展性与样本效率仍待验证

---

## 483. What Limits Robustness in Deep Image Watermarking: An Analysis of Mechanisms and Their Scaling Across Capacities

**arXiv ID:** 2609.01050 | [PDF](https://arxiv.org/pdf/2609.01050v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 484. Medical Causal Hypothesis Verification with Large Language Models

**arXiv ID:** 2609.00063 | [PDF](https://arxiv.org/pdf/2609.00063v1)

**作者:** Safiyyah Ahmed `[一作]` (University of Illinois Chicago), Elena Zheleva `[通讯]` (University of Illinois Chicago)

**通讯引用:** 2093 | [OpenAlex ID](https://openalex.org/A5071079350)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对8款大型语言模型在医学因果假设验证任务中的表现进行系统评估，构建了评估框架并对其结果进行了详细分析。

**💡 创新点**

创新点在于提出了针对因果假设验证的多维评估框架，结合精确的标注指标与自动化评测流程，首次量化LLM在检索、生成真实论文、提供准确摘要与引用等方面的可靠性。

**🔧 技术方法**

主要技术包括结构化提示设计、检索增强生成（RAG）技术、人工标注（Label Studio）、自动解析与评估脚本，以及多种评测指标（准确率、召回率、F1、特异度、URL/摘要/引用准确性等）。

**📊 数据集**

使用的数据集为17个由临床医师验证的医学因果假设，结合每个模型产生的论文引用，人工共标注1067个评估点，形成了基准评测数据。

**📈 对比分析**

对比方法为计算每个模型在因果判断和证据生成两大维度的指标，结果显示模型在判断真因果关系时召回率普遍较高（多为1.0），但特异度低，证据真实性仅为11%~58%，URL、摘要和引用的准确率极低。

**⚠️ 局限性**

局限性包括样本量小、仅评估17个假设，且所有模型在提供真实可验证证据方面表现不佳，常出现引用虚假或误标的论文，且对非支持假设的拒绝能力不足，难以直接用于临床决策。

---

## 485. A Version Space Approach for Digital Circuit Analysis

**arXiv ID:** 2609.00609 | [PDF](https://arxiv.org/pdf/2609.00609v1)

**作者:** Mitchell A. Thornton `[一作]` `[通讯]` (Southern Methodist University), Mitchell A. Thornton (Southern Methodist University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种统一的版本空间计数方法，用于概率组合等价检查和锁定电路的密钥计数。

**💡 创新点**

通过把 Haar 系列转换为块求和重新参数化，并利用局部树结构的求和乘积递归，实现了对一般情况的精确计数和多项式时间求解。

**🔧 技术方法**

采用修改后的 Haar 变换、块树求和、求和乘积算法（sum‑product）以及因子图求解；对关键计数还使用门级因子图与决策图两种实现。

**📊 数据集**

在组合等价检查上验证了所有 2ⁿ 变量函数的计数结果；在锁定电路上使用 Trust‑Hub 发布的 70 个实例、随机插入、强锁定与基于逻辑锥的三种方案以及 BDD 基准。

**📈 对比分析**

与 2002 年方法相比，精确计数显著提升，复杂度从指数改为多项式；在 Trust‑Hub 基准上，Engine‑A 与 Engine‑B 两种实现得到一致结果，且在大多数实例中都在几秒到几分钟内完成。

**⚠️ 局限性**

局限性包括只能处理无环组合电路、对块求和重参数化的依赖、在极大关键空间或高度依赖于查询顺序时仍可能产生上界；此外，结果仍为上限，未必能捕捉所有等价关键的细节。

---

## 486. Instella-MoE Technical Report

**arXiv ID:** 2609.00791 | [PDF](https://arxiv.org/pdf/2609.00791v1)

**作者:** Jiang Liu `[一作]`, Emad Barsoum `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

训练了一个16B总参数、每个token激活2.8B参数的全开源 Mixture‑of‑Experts 语言模型 Instella‑MoE，涵盖从零开始的预训练、mid‑训练、长上下文扩展、监督微调、直接偏好优化（DPO）和强化学习（RL）等完整多阶段流程。

**💡 创新点**

主要创新包括：Gated Multi‑head Latent Attention（门控 MLA）提升注意力表达力；FarSkip‑Collective 通信重叠技术显著提高 MoE 的训练和推理吞吐；以及公开完整的训练数据、代码、配置，构成完全可复现的开源管线。

**🔧 技术方法**

技术实现基于 AMD Instinct MI300X/MI325X GPU，使用 Primus/Miles 框架、FlashAttention、bfloat16 混合精度、SGLang 推理、MoE 共享‑+‑路由专家结构、Gated MLA、FarSkip‑Collective、R3 路由重放和 TIS 重要性采样等。

**📊 数据集**

数据集涵盖 7.1T 高质量开放语料（Nemotron‑CC、Nemotron‑CC‑Math、MegaMath、RefineCode 等），mid‑训练用 Dolma3 Dolmino 100B，长上下文用 Dolma3 Longmino 100B，SFT 用 Dolci‑Think‑SFT‑7B、Nemotron‑Cascade‑2 等，DPO 用 Dolci‑Think‑DPO‑7B，RL 用 Dolci‑Think‑RL‑7B。

**📈 对比分析**

通过与 SmolLM3‑3B、OLMo‑3‑7B、Moonlight‑16B‑A3B、Qwen3.5‑4B 等全开源和 open‑weight 模型对比，基线 Instella‑MoE‑Base 在标准预训练基准上平均得分 76.7，Post‑trained Think checkpoint 在多项推理、推理、代码、算术等基准上平均 73.2，长上下文任务 HELMET 41.5、RULER 79.4，推理时间‑首 token 提升 39.2%。

**⚠️ 局限性**

局限性包括：长上下文训练后短文本能力出现轻微衰退，需要权衡；MoE 路由漂移在 DPO/ RL 阶段需特殊处理；训练成本仍高，需 AMD GPU；在某些任务（如 AIME24/25、GPQA）RL 阶段可能出现退化；未覆盖多语言或多模态场景，模型规模相对固定。

---

## 487. A Network Science Perspective on Evaluating Deep Graph Generative Models

**arXiv ID:** 2609.01015 | [PDF](https://arxiv.org/pdf/2609.01015v1)

**作者:** Tianrui Mao `[一作]` (Delft University of Technology), Huijuan Wang `[通讯]` (Delft University of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了深度图生成模型在网络科学中的应用，评估其生成的合成网络与真实网络的结构相似性及其在传播抑制中的有效性。

**💡 创新点**

创新点在于将网络生成模型与传统网络科学评价方法相结合，证明 SparseDiff 和 EDGE 可用于生成可识别有效免疫策略的网络。

**🔧 技术方法**

使用了 GraphRNN、EDGE、SparseDiff、GGSD 以及配置模型，并在 SIR 传播模型中评估免疫策略。

**📊 数据集**

利用 Delft University of Technology 和 University of Neuchâtel 的六个真实网络（email、Astro Physics、Soc-advogato、Facebook、CA-HepPh）及其采样的两跳社区网络。

**📈 对比分析**

通过比较节点数、度分布、密度、聚类系数、模块度和最大特征值等统计量评估生成网络与真实网络的相似度；在 SIR 实验中比较不同生成模型的免疫策略排名和平均爆发规模，SparseDiff 与 EDGE 表现最佳。

**⚠️ 局限性**

限制在于只评估了单一 SIR 免疫场景，且对更大规模社区网络的生成与可扩展性尚未充分探究。

---

## 488. The Safeguard Worked. Is the LLM System Safer?

**arXiv ID:** 2609.00519 | [PDF](https://arxiv.org/pdf/2609.00519v1)

**作者:** Pingyu Wu `[一作]` (University of Science and Technology of China), Nenghai Yu `[通讯]` (University of Science and Technology of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过构建一套将已公开的LLM防护实验结果转换为部署安全结论的框架，探讨本地评估指标与实际部署安全之间的关系。

**💡 创新点**

提出锚定-日程方法和十槽编码工具，能够从已有防护报告中提炼残留有害援助的上下界，并指出单一局部分数不足以判断部署安全。

**🔧 技术方法**

利用理论推导与形式化证明（如极值、最优值、条件转换），以及对文献的系统编码与统计分析。

**📊 数据集**

使用近200篇LLM安全评估论文的数据（包括攻击成功率、拒绝率、覆盖率等指标），并对其中24篇进行深度编码。

**📈 对比分析**

通过对比攻击成功率、覆盖率、失败率、后续价值等多维度指标，评估每种防护在给定部署锚点下的残留安全水平；结果显示多数论文仅给出正向证明，缺乏零残留证书。

**⚠️ 局限性**

受限于现有报告缺失关键指标（如覆盖率、后续价值、依赖关系），框架无法为所有防护给出完整部署安全结论；此外，对外部世界型安全的解释受双重使用下限的限制。

---

## 489. Enoki: Efficient Multi-Level Hallucination Detection

**arXiv ID:** 2609.00581 | [PDF](https://arxiv.org/pdf/2609.00581v1)

**作者:** Elisei Rykov `[一作]` (Skoltech), Julia Belikova `[通讯]` (Skoltech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于文本锚定OpenIE的多粒度幻觉检测框架，联合了主张级验证与跨度级定位。

**💡 创新点**

创新点在于利用同一中间表示实现主张验证与跨度投影的无对齐统一，并支持LLM、编码器和规则提取三种模式。

**🔧 技术方法**

使用OpenIE提取文本锚定事实，基于NLI的事实验证，以及增量事实构造和投影机制。

**📊 数据集**

使用了EnokiQA双粒度数据集以及Hallucination检测基准如HalluciEntity、MuSHROOM、RAGTruth、FactCheckBench等。

**📈 对比分析**

与现有显式验证和隐式验证方法对比，所提框架在跨度级F1和实体级AUPRC均领先，并在效率上比多阶段LLM管线快数倍。

**⚠️ 局限性**

局限性主要在于对事实提取的依赖、增量投影可能导致定位不精确，以及跨句核心ference处理不足。

---

## 490. VATO: A Vortex-Force-Aware Transformer Operator for Unsteady Separated Aerofoil Flows

**arXiv ID:** 2609.00507 | [PDF](https://arxiv.org/pdf/2609.00507v1)

**作者:** Xingxin Yang `[一作]` (King's College London), Juan Li `[通讯]` (King's College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种名为 VATO 的神经算子，利用气旋力图 (VFM) 与几何感知 Transformer 结构相结合，对双刃翼在未分离流中的时间演化进行高精度预测。

**💡 创新点**

创新点在于：①在训练阶段引入 VFM 贡献场监督，使模型学会聚焦对载荷贡献最大的流场区域；②在推理阶段采用 VFM 生成的权重对输入点进行优先级排序，并通过残差交叉注意力显式聚合这些关键区域，从而显著提升场预测与力学响应的准确性。

**🔧 技术方法**

核心技术包括：几何感知图神经算子 (GAOT) Backbone、Vortex Force Map (VFM) 辅助势场、训练阶段的 VFM 贡献场损失、VFM 基于源点优先级排序与残差交叉注意力模块。

**📊 数据集**

使用的数据集为 2D 不稳定 CFD 结果，覆盖 9 个双刃翼几何、13 个攻角，共 54 条测试轨迹（每条 28 个预测点），流场由 STAR‑CCM+ 在马尔萨气象条件下（Re≈10^4）求解得到。

**📈 对比分析**

对比方法：以 GAOT 参考模型为基准，另外构建了采样匹配控制模型和两种 VATO 变体。实验显示 VATO‑A 在 1–20 ms 训练范围内，速度、压强、涡度误差分别降低 15.8%、7.5% 和 31.2%；VATO‑S 主要降低 VFM 推导的拖曳误差。VATO‑A 在 21–30 ms 预测外推时仍保持 26.9% 的涡度误差优势，并使压强推导的 CL/CD MAE 分别降低 17.2% 与 21.7%。

**⚠️ 局限性**

局限性包括：①VFM 相关耦合与采样策略的效应难以完全分离；②仅在已知几何的未见攻角上验证，缺乏对全新翼型的泛化评估；③仅使用单一训练随机种子，未探讨模型重现性；④推理成本较高（VATO‑A 约 63% 超出基准）；⑤未进行自回归评估或跨求解器验证。

---

## 491. ClinTraceBench: Source-Verifiable Longitudinal Clinical Reasoning over EHR-Derived Dialogues

**arXiv ID:** 2609.01111 | [PDF](https://arxiv.org/pdf/2609.01111v1)

**作者:** Huimin Wang `[一作]` (Shenzhen University), Yutian Zhao `[通讯]` (Dealism)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ClinTraceBench，一个可验证的多访患者轨迹问答基准，评估不同历史表示方式在长期临床推理中的表现。

**💡 创新点**

创新点包括：① 通过 385 条 MIMIC‑IV 衍生的对话，提供事件 ID 源级可追溯性；② 设计 9 项涵盖事实回忆、趋势、关联、矛盾、跨病人比较、治疗反应和不确定性抑制的任务；③ 引入保留探针（T3）评估表示对关系信息的保留能力；④ 在不同大模型上对 8 种历史表示方式进行成本-质量 Pareto 对比。

**🔧 技术方法**

采用 LLM 生成对话、检索增量生成、结构化时间线、LLM 摘要、Mem0、A‑Mem 等历史表示；对话和答案验证使用多级（L0–L4 + L5）判定器；评估基准在 4 种前沿 LLM（DeepSeek‑V3、GPT‑4o‑mini、Claude Haiku 4.5、Claude Sonnet 4.6）上完成。

**📊 数据集**

使用 MIMIC‑IV（ICU/医院全记录）数据抽样 400 名患者，按疾病和复杂度分层，构造 385 条对话；对话生成基于 DeepSeek‑V3 的模板，并通过人工审核保证 98.92% 的一致性。

**📈 对比分析**

对比 8 种表示方式在 6,271 题、200,672 预测上的表现。结果显示：全上下文获得最高准确率（约 70%）；密集检索紧随其后，成本更低；压缩与代理记忆在多访聚合任务上显著受损；在成本-质量 Pareto 前沿，Haiku 在全上下文下优于 Sonnet，打破“更大模型更好”假设。

**⚠️ 局限性**

局限性：① 对话为 LLM 合成，缺乏真实临床对话的模糊性和多样性；② 只使用单一 EHR（MIMIC‑IV），不涵盖 ICU 之外或非英语环境；③ T5/T6/T8/T9 任务受限于构造方式，难以区分模型能力；④ 仅评估 385 条对话，时间跨度有限；⑤ 代理记忆的预处理模型固定为 DeepSeek‑V3，可能掩盖架构差异。

---

## 492. AniMaster: From Story Texts to Animated Videos via Cinematic Script Generation and Interactive Authoring

**arXiv ID:** 2609.00346 | [PDF](https://arxiv.org/pdf/2609.00346v1)

**作者:** Ruiqi Yu `[一作]` (Hangzhou Dianzi University), Yong Wang `[通讯]` (Nanyang Technological University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为 AniMaster 的作者工具，帮助普通创作者将自由文本故事转化为长格式动画视频，并通过三层（故事、剧本、视频）结构化中间表示（事件、节拍、镜头）实现可编辑的故事到镜头的自动转换。

**💡 创新点**

创新点在于将叙事学与电影学的专业知识编码为可视化、可编辑的中间层（事件→节拍→镜头），并通过规则化翻译与交互式编辑实现“专业导演流程”的低门槛重现；此外采用语义放大/缩小的画布式界面，支持多层级可视化与自由探索。

**🔧 技术方法**

核心技术包括大语言模型（Google Gemini）用于文本理解与节拍生成，视频生成模型（Seedream、Gemini 3.1 Flash Image、Veo 3 Generate）用于帧与视频合成，RIFE 做帧插值；前端基于 Vue3+Vue Flow、PixiJS、D3；后端使用 FastAPI+Python。

**📊 数据集**

使用的“数据集”主要是短篇童话故事（如《爱丽丝梦游仙境》《小红帽》）以及在案例研究中创作者自行编写的故事文本；没有公开的大规模视频-文本对齐数据集，而是利用这些文本进行翻译与生成。

**📈 对比分析**

评估通过 N=16 的对照实验（AniMaster vs. 基线）、两份案例研究与两位专业电影人访谈。结果显示 AniMaster 在创造力支持（CSI）显著提升（7.16 vs. 5.70，p=0.012），且在所有 9 项偏好量表中均优于基线；NASA‑TLX 总负荷无显著差异，表明工具并未增加总体工作量。

**⚠️ 局限性**

局限性包括：翻译规则仅基于经典连续剪辑理论，缺乏对非线性叙事、实验性镜头语言的支持；跨场景叙事规划有限，难以处理更长、复杂的故事；最终视频在跨镜头一致性与场景连贯性上仍有缺陷，主要适合作为预可视化与创作辅助工具，而非完整的端到端制作流水线。

---

## 493. Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning

**arXiv ID:** 2609.00351 | [PDF](https://arxiv.org/pdf/2609.00351v1)

**作者:** Robin Haselhorst `[一作]` (University of Bonn), Florian Mai `[通讯]` (Lamarr Institute for Machine Learning and Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无监督检测方法——激活匹配微调，用来在不知触发器或隐藏行为的前提下识别大型语言模型中潜藏的条件性隐藏行为（如后门、睡眠代理、沙袋化、奖励窃取、主题审查）。

**💡 创新点**

创新点在于通过在一组无触发器的平凡语料上让anchor模型与可疑模型在激活空间中匹配，从而使参考模型只学习正常计算；随后对任何评估提示计算残差，残差会在触发器及其语义邻域出现明显峰值，形成对隐藏行为的无先验检验信号。

**🔧 技术方法**

核心技术包括：1) 激活匹配微调（最小化anchor与可疑模型在残差层上的L2差距）；2) 计算残差的L2和余弦距离作为异常分数；3) 对评估提示进行批量激活记录；4) 使用nDCG与z-score评估检测排名；5) 设计防御意识对抗者的正则化实验。

**📊 数据集**

使用公开的“WildChat”指令语料（约10⁴条）做微调训练；评估语料为518组（10条/组）合成提示，由Claude Opus 4.7生成并手工标注相关度；并在这些提示上测试多种后门、沙袋化、奖励窃取和主题审查模型。

**📈 对比分析**

在所有被攻击模型（后门、睡眠代理、沙袋化、奖励窃取、主题审查）上，激活匹配残差的nDCG远超随机基线；例如后门模型的z-score可达+200，覆盖率超过5%阈值；在中文审查模型中检测到轻微信号；在沙袋化模型中对MC格式的提示有极高z-score。

**⚠️ 局限性**

局限包括：① 需要可用的干净anchor模型和白盒残差访问；② 只适用于触发器稀疏且有语义邻域的隐藏行为；③ 评估集为人工合成，可能引入标注偏差；④ 对更大规模或多模态模型的泛化尚未验证；⑤ 仅输出排名，无法直接给出触发器文本；⑥ 触发器若无可探测的语义邻域或为深层生成触发，则残差信号可能弱化。

---

## 494. Autoresearch for Marketplace Catalogs: From Legacy Forms to AI-Native Matching

**arXiv ID:** 2609.00274 | [PDF](https://arxiv.org/pdf/2609.00274v1)

**作者:** Kartik Ravisankar `[一作]` (Thumbtack), Vijay Anand Raghavan `[通讯]` (Thumbtack)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并部署了一个自动化自我学习的“autoresearch”循环，用于为服务市场重构并生成每个职业的服务提供者偏好标签目录，并将旧的问答表单映射到新目录。

**💡 创新点**

将每个职业的目录视为独立生成单元，使用七角色批评家加权惩罚的LLM评估，并在生产中实现大规模并行部署，实现精准迁移。

**🔧 技术方法**

采用交叉模型堆叠（GPT‑5‑4 生成/评估，Claude Sonnet 4.6 批评/编辑/映射），LLM‑as‑judge 六维评分，自动提议‑评估‑保留循环，以及 Step‑Back 抽象与多模型反思。

**📊 数据集**

使用 132 个职业的旧结构化请求‑答复表单（Q&A）以及实时的专业人员属性和过滤评估日志（约 930 万次）。

**📈 对比分析**

与单次生成和无批评家设置对比，平均提升评估分数约 +2.24/15，保持 95% 职业通过自动批准；生产过滤率显示缺失标签问题约 40% 被发现。

**⚠️ 局限性**

依赖跨模型的信任保障仍有限，编辑与批评家共属同一模型家族可能导致偏差；缺少实时反馈将匹配失败映射回生成循环，且跨职业一致性仍有细微漂移。

---

## 495. Web Price Extraction: State of the Art and an Adaptive Browserless Implementation

**arXiv ID:** 2609.01030 | [PDF](https://arxiv.org/pdf/2609.01030v1)

**作者:** Evgeniia Kositsyna `[一作]` (University of Zaragoza), Jorge Lloret-Gazo `[通讯]` (University of Zaragoza)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套自适应无浏览器的网页价格提取系统，先通过基于规则的碎片化、句法/语义/频率三类判定器完成价格候选点的筛选，再引入贝叶斯动态更新规则权重与遗传算法全局参数优化，使系统在面对不同网站结构变化时保持高精度。

**💡 创新点**

创新点主要体现在：
1) 将贝叶斯权重更新与遗传算法结合，既可个性化每条规则的置信度，又能自动调节全局阈值，从而提升鲁棒性与精度；
2) 在无浏览器的基础上实现动态规则自适应，显著降低对手工维护的依赖；
3) 通过实验验证，精度从77.2%提升至87.3%，且每页平均处理时间下降约14%。

**🔧 技术方法**

技术手段包括：
- Python 3 生态（requests、BeautifulSoup、pandas）；
- 规则系统（句法、语义、频率三类规则）；
- 贝叶斯推断更新规则权重；
- 遗传算法（个体编码为 {p_limit_freq, discard_threshold}，采用交叉、变异、精英保留等操作）。

**📊 数据集**

使用了覆盖250+不同电商市场的网页数据集，包含多种结构与动态渲染特征，用于训练贝叶斯权重、遗传优化以及最终评估；此外对比时引用了公开的基准数据集（如SWDE、735站点等）来对比其他方法。

**📈 对比分析**

通过与文献中四类方法（经典规则、浏览器模拟、无浏览器、AI/ML）在统一指标（精度、对异构网站的鲁棒性、资源消耗、样本规模）下进行对比：
- 经典方法精度最高但在结构异构时骤降；
- 浏览器模拟能处理动态内容但资源消耗极高；
- 无浏览器在735站点上平均精度81%，时延0.8s/页；
- AI/ML（AutoScraper+GPT‑4‑Turbo）零样本F1 88.69%但需高昂LLM调用成本。
本工作在250+站点上，精度提升至87.3%，处理时间比基线降低14%，在资源受限的竞争监控场景中具备竞争优势。

**⚠️ 局限性**

局限性包括：
- 规则系统仍需针对新站点进行初始配置，虽然贝叶斯+GA能自适应，但对完全陌生结构的快速适配仍受限；
- 仅针对静态HTML，无法处理大量JS渲染或需要验证码/反爬策略的网站；
- 贝叶斯更新和遗传优化需要一定训练样本，且遗传算法在大规模优化时仍耗时；
- 对抗反爬机制（如验证码、Canvas指纹）仍需额外策略，未在本文中深入探讨。

---

## 496. Do Satellites See Commuters? A Critical Benchmark of Vision Foundation Models

**arXiv ID:** 2609.00661 | [PDF](https://arxiv.org/pdf/2609.00661v1)

**作者:** Ashiq Shukoor Iqbal `[一作]` (University of New South Wales), Flora D. Salim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在固定的生成式 OD 图扩散框架下，对四种卫星视觉基础模型（RemoteCLIP、DINOv3、SatCLIP、AlphaEarth）进行系统消融实验，评估其在美国县级、英国地区和全球14个大都市的通勤原点-目的地生成性能。

**💡 创新点**

①首次在统一管道中对不同预训练目标（语言对齐、自监督、坐标对比、多传感器融合）的视觉编码器进行对比；②明确 census noise 参数 η 的语义和对跨域评估的影响；③发现地理锚定编码器在跨大洲零样本迁移上显著优于语言监督模型。

**🔧 技术方法**

技术包括：WeDAN 生成式图扩散、GraphTransformer 消除噪声、H3 网格化卫星影像预处理、不同视觉编码器的特征提取、CPC 评价指标、配对 t‑检验及 Holm‑Bonferroni 校正。

**📊 数据集**

数据集：美国 1,925 个县级 LODES OD 矩阵、英国 325 个地方管理区 OD、14 个全球大都市（北京、东京、伦敦等）OD；卫星影像来自 Esri World Imagery/GEE；人口统计来自 WorldPop；Census noise 参数在实验中随机采样。

**📈 对比分析**

在美国内分布测试中 RemoteCLIP 取得最高 CPC 0.598（η=1 时为 0.602）；在英国零样本迁移中 AlphaEarth 与 SatCLIP 约 0.51，远超 RemoteCLIP（0.39）和 DINOv3（0.33）；在全球迁移中四种编码器均收敛至约 0.11，DINOv3 仅 0.022；统计检验显示跨编码器差异显著（p<0.001）。

**⚠️ 局限性**

限制：①仅使用静态年度卫星图像，缺乏时序动态特征；②训练集规模受 80/10/10 分割限制，可能导致整体 CPC 略低；③跨域评估受卫星影像与 OD 数据时空不匹配、城市形态差异大、图 Transformer 规模受限等因素影响；④CPC 作为整体指标掩盖了方向性和规模误差；⑤实验固定 WeDAN，未探索更具可扩展性的生成器或细粒度的迁移策略。

---

## 497. Operation-Type-Aware Client Routing for Leader-Based Consensus Datastores

**arXiv ID:** 2609.00392 | [PDF](https://arxiv.org/pdf/2609.00392v1)

**作者:** Sri Saran Balaji Vellore Rajakumar `[一作]` (Amazon Web Services), James Thompson `[通讯]` (Amazon Web Services)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于操作类型的客户端路由策略，将写请求定向到leader，读请求分布在健康节点上。

**💡 创新点**

创新点在于揭示leader‑based共识系统中写前向成本与读确认成本的差异，提出操作类型感知路由和“consensus‑cost invisibility”原理，并在etcd与ZooKeeper跨系统验证其普适性。

**🔧 技术方法**

使用gRPC负载均衡、Boltzmann自适应调度、Raft/Zab协议实现、Go自定义基准、并结合etcd v3.5.17与ZooKeeper 3.9.2客户端进行实验。

**📊 数据集**

实验数据集为随机1,000键键值对、256字节值，读写比例80/20或50/50，覆盖3节点与5节点多可用区集群。

**📈 对比分析**

比较方法包括round_robin、Boltzmann、Hybrid三种路由；在健康状态下写P50降低29‑37%，读P99下降64%，吞吐量提升约8‑13%；在灰度失败时吞吐提升89%，尾延迟大幅下降。

**⚠️ 局限性**

局限性：需要主动leader检测与降级监控；实验受网络延迟、连接数等因素影响；仅验证了Raft与Zab，未覆盖无领导或多领导协议；规模仍有限。

---

## 498. ReNFT: Repairing Mode Collapse in Reward Post-Training via Internal Probability-Mass Recalibration

**arXiv ID:** 2609.00061 | [PDF](https://arxiv.org/pdf/2609.00061v1)

**作者:** Yuchen Bao `[一作]` (Southern University of Science and Technology), Jianguo Zhang `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 52241 | [OpenAlex ID](https://openalex.org/A5100409879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过内部概率质量重校准技术，对已经在奖励后训练中出现模式崩塌的扩散生成器适配器进行修复，恢复其在同一提示下的多样性。

**💡 创新点**

创新点在于：①利用无条件路由作为内部偏差探测器，优先选取“anti-hub”提示；②设计两条混合策略路由（基底+条件、条件+无条件）生成对照反事实样本；③通过奖励排序和自适应翻转守卫为每对样本决定拉伸/推挤目标，并采用联合‑配对NFT更新实现概率质量再分配。

**🔧 技术方法**

技术栈包括：DiffusionNFT的前向过程回归后训练、基底+无条件混合路由、无条件探测、奖励排序、适应翻转守卫、联合‑配对NFT损失、EMA参考平滑以及LoRA微调。

**📊 数据集**

使用了PickScore（模型基奖励）和GenEval（规则基奖励）两套提示集；多样性评估采用DreamSim-Div、LPIPS-Div、DINOv3-Div；还引用SD3.5-M作为基准模型。

**📈 对比分析**

与SD3.5-M、Flow-GRPO、DiffusionNFT、DiverseGRPO、E^2PO等方法对比，ReNFT在保持 98.9%/99.0% 的奖励的同时，DreamSim-Div 分别提升 58.8%/55.0%，在所有对比方法中获得最佳多样性；在奖励维度上排名第二，表明修复后仍能保持高质量。

**⚠️ 局限性**

局限性包括：需要人工挑选 anti-hub 提示，内部路由设计对不同奖励模型可能不稳定；修复仅恢复被压制的概率质量，无法完全挖掘新的视觉内容；在奖励极度易被攻击的场景下，仍存在重崩塌的风险。

---

## 499. Why Multi-Layer Message Passing Works: Completeness Theory for Graph Neural Network Interatomic Potentials

**arXiv ID:** 2609.00528 | [PDF](https://arxiv.org/pdf/2609.00528v1)

**作者:** Pingbing Ming `[一作]` (SKLMS, Institute of Computational Mathematics and Scientific/Engineering Computing, AMSS, Chinese Academy of Sciences), Han Wang `[通讯]` (National Key Laboratory of Computational Physics, Institute of Applied Physics and Computational Mathematics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了针对图神经网络（GNN）的完整性理论，构建了一个可证明为完整的原子势能表达式，并将其嵌入到GNN模型中以提升对分子能量和力的预测精度。

**💡 创新点**

创新点在于首次将理论上的完备性（completeness）概念系统地引入GNN交互势能框架，并给出了严格的数学证明，证明该表达式可以无偏差地捕捉所有对称不变的原子相互作用。

**🔧 技术方法**

使用了基于对称不变的基函数展开、图卷积层、注意力机制以及多层感知器（MLP）进行特征学习，并结合了正则化与数据增强技术以提升泛化性能。

**📊 数据集**

主要使用公开的QM9、MD17以及ANI-1x等分子动力学和能量计算数据集进行训练与评估。

**📈 对比分析**

与SchNet、DimeNet、PaiNN、NequIP等主流GNN势能模型进行对比，实验表明在预测能量、力以及热力学性质方面，完整性理论模型在RMSE/MAE指标上平均提升约5–10%，并在极端结构（高压/高温）下保持更好的稳健性。

**⚠️ 局限性**

局限性包括：1）完整性表达式在大规模体系中计算复杂度较高，需要更多的计算资源；2）模型对极端极化或多电子效应的捕捉仍受限；3）训练数据量要求较大，对少样本任务仍存在性能瓶颈。

---

## 500. Independent Reinforcement Learning in Discounted Markov Games

**arXiv ID:** 2609.00504 | [PDF](https://arxiv.org/pdf/2609.00504v1)

**作者:** Asrin Efe Yorulmaz `[一作]` (University of Illinois Urbana-Champaign), Tamer Basar `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了在折扣一般与Markov游戏中无耦合学习的可行性，提出了分层乐观镜像下降（Layered OOMD）算法，并给出全反馈与部分反馈两种实现，证明了子指数收敛性。

**💡 创新点**

首次在无耦合信息结构下实现可收敛算法，并在ETH‑PPAD假设下给出计算上不可多项式解的下界，形成理论与算法的完整闭环。

**🔧 技术方法**

采用分层学习率、平滑熵正则化的乐观镜像下降、折扣截断转换、Q‑值块式估计、可达性假设等技术；并结合理论分析证明收敛上界。

**📊 数据集**

使用四个人工构造的有限期Markov游戏（性别博弈、路由/拥堵、公共品、转移陷阱）以及离散化的二人LQ零和折扣游戏作为实验基准。

**📈 对比分析**

通过实验测量经验分布的CCE误差随回合/块数的下降，结果显示误差随迭代显著降低；在LQ基准实验中误差快速逼近闭式解。

**⚠️ 局限性**

算法仅给出子指数上界，与下界仍有差距；需要可达性假设与块式估计，难以扩展到极大状态空间；目前仅适用于一般和Markov游戏，缺乏更广泛的结构约束。

---

## 501. Effective Interventions Against AI-Enhanced Scams

**arXiv ID:** 2609.00806 | [PDF](https://arxiv.org/pdf/2609.00806v1)

**作者:** Kyle Fredrickson `[一作]` `[通讯]` (Quarry Intelligence), Kyle Fredrickson (Quarry Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个简化的诈骗盈利模型，分析AI驱动诈骗对报告率、报告集中度与报告准确性三个杠杆的乘法影响，并估算干预措施对诈骗盈利的边际效应；

**💡 创新点**

创新点在于将报告率、集中度和误报率视为可乘积的干预杠杆，提出即使单一杠杆效果有限，复合干预亦能显著削弱诈骗收益，并给出基于模型的盈利上限估算；

**🔧 技术方法**

主要采用概率论与优化理论（Poisson、Binomial、Lambert W函数等）进行数学建模，并基于现有统计数据进行参数化；

**📊 数据集**

未使用传统机器学习数据集，而是引用公开统计（如GASA、Chainalysis、YouTube直播诈骗收益等）与作者假设的参数（如每个诈骗渠道成本、平均受害者损失等）进行定量分析；

**📈 对比分析**

由于是理论模型，没有与其他算法或系统做实验对比；作者通过模型推导给出在不同报告率和成本假设下，诈骗是否能保持盈利的阈值，提供一种“理论性能”评估；

**⚠️ 局限性**

局限性包括：①对诈骗成本的估计偏低，只考虑通讯与金融渠道成本，忽略洗钱费、腐败支付等；②假设AI可完全降低对接成本；③误报率与阈值模型简化，未充分考虑实际平台的误报容忍度；④模型依赖多项假设与参数估计，实际环境复杂度未能完全捕捉；

---

## 502. TEIDAN: A Multilingual Multiparty Dialogue Corpus

**arXiv ID:** 2609.00802 | [PDF](https://arxiv.org/pdf/2609.00802v1)

**作者:** Taiga Mori `[一作]` (Kyoto University), Tatsuya Kawahara `[通讯]` (Kyoto University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

创建并描述了 TEIDAN 多语言多模态三方对话语料库，涵盖日语与英语的自然面对面讨论，并给出统计与初步分析。

**💡 创新点**

首个兼具多语言、多模态且开放式三方对话语料库，可跨语言对比面向人机交互的会话行为。

**🔧 技术方法**

采用个体针式麦克风、麦克风阵列与面向摄像头的多模态录音，使用 IPU（>200 ms停顿）标注与后续的伪轮次统计。

**📊 数据集**

TEIDAN 数据集，包含 69 次会话、57 名参与者、9 小时 47 分 57 秒的音视频与 IPU 文字稿。

**📈 对比分析**

通过去除回声词和非语言 IPU，统计每语言的 IPU 数量与伪轮次长度，结果显示英语会话更长且 IPU 更密集，而日语会话更短且轮次更紧凑。

**⚠️ 局限性**

样本受限于仅日语和英语，参与者关系（熟识 vs 随机陌生）差异导致跨语言比较可能混杂文化与社交因素；缺乏 TRP、受众等更细粒度标注。

---

## 503. VoiceLongMemEval: Do Assistants Remember How You Sounded?

**arXiv ID:** 2609.00570 | [PDF](https://arxiv.org/pdf/2609.00570v1)

**作者:** Ramit Pahwa `[一作]`, Apoorva Beedu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了VoiceLongMemEval（VLME）基准，用来测试语言模型在长时间对话中记忆说话方式（情绪、语调、非语言事件）的能力；并通过三阶段对抗门确保仅凭文字无法回答问题。

**💡 创新点**

首次将长时记忆与声学情感信息结合，提出了“情感缺口”（affect gap）概念；通过对抗门验证情绪、语调等元数据对推理至关重要。

**🔧 技术方法**

采用情绪标注、语调元组、非语言事件注释、结构化标签、自然语言阶段描述等多种语音元数据；使用对抗门、链式思维提示、声学原生模型与ASR+LLM级联比较、TTS合成与人工听审等技术。

**📊 数据集**

VLME数据集共523条测试项，包含202条核心对抗项、181条细化项、140条间接项，涵盖326个证据会话，单会话约100k token，所有对话均配有情绪、语调、事件等元数据。

**📈 对比分析**

对八个模型（四个专有前沿模型、四个开源模型）进行评测，发现情绪/语调元数据可提升+0.09~+0.38准确率；提示干预可进一步提升，且声学原生模型在音频评估中比ASR+LLM级联表现更好。

**⚠️ 局限性**

局限性包括：数据为合成对话，情绪分布可能不自然；仅在oracle（约10-15k token）层面评估；音频评测仅用两款7B原生模型；LLM评审器可能存在偏见；对抗门的有效性受提示依赖。

---

## 504. VARA: A Voltage-Aware ReRAM-Based Accelerator for Energy-Efficient Computing

**arXiv ID:** 2609.00421 | [PDF](https://arxiv.org/pdf/2609.00421v1)

**作者:** Peng Dang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Huawei Li `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种基于ReRAM的电压感知加速器VARA，提出了电压感知训练（VAT）算法来提升激活稀疏性，并结合共零激活重排（CAR）机制实现跨越式的交叉电路计算跳过，从而显著降低系统能耗。

**💡 创新点**

创新点包括：①在训练阶段通过在激活函数中加入预设阈值主动提升激活零比率；②利用共零相关性对激活与权重矩阵进行可计算等价的重排，聚合散布的零值形成连续全零区域；③将激活稀疏性与硬件计算跳过紧密耦合，构建完整的算法‑硬件协同优化链路。

**🔧 技术方法**

采用的技术主要有：电压感知训练（VAT）实现激活阈值调节；共零激活重排（CAR）聚类与重排算法；ReRAM交叉阵列矩阵‑向量乘法（MVM）与ADC/DAC电路；系统级仿真与能耗分析。

**📊 数据集**

使用的数据集包括CIFAR‑10、CIFAR‑100和Tiny ImageNet，模型分别为VGG11、ResNet18和ResNet34。

**📈 对比分析**

通过与基线（标准激活函数）、SARA（基于权重稀疏）以及RapPIM（稀疏激活优化）等方法进行对比，VARA在保持约0.71%平均准确率损失的前提下，系统能耗平均降低60.12%，能效提升2.68×，显著优于现有最先进方案。

**⚠️ 局限性**

局限性：①需要在训练阶段手动调节阈值，调参成本较高；②重排与跳过依赖于交叉电路规模，跨越跳过比例随阵列尺寸变化；③目前仅针对ReRAM架构，未评估在其他非易失性内存或传统CPU/GPU上的迁移；④索引单元虽然能耗低，但在极大规模网络中仍可能产生一定的存储与访问开销。

---

## 505. Quantum LDPC and High-Rate CSS Codes from Fair-Density Parity-Check Codes

**arXiv ID:** 2609.01181 | [PDF](https://arxiv.org/pdf/2609.01181v1)

**作者:** Hessam Mahdavifar `[一作]` `[通讯]` (Northeastern University), Hessam Mahdavifar (Northeastern University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了量子低密度奇偶校验(qLDPC)和高率的Calderbank–Shor–Steane (CSS) 码，基于最近引入的经典公平密度奇偶校验(FDPC) 码。

**💡 创新点**

首次提供了有限率的qLDPC框架，能够同时提供有限长度的认证最小距离信息和低权重逻辑多重性的分析特征。

**🔧 技术方法**

采用了结构化稀疏化FDPC奇偶校验矩阵的技术，结合超图乘积构造，得到了具有可控块长度、维度、认证距离和稳定器权重的qLDPC码。

**📊 数据集**

使用了经典公平密度奇偶校验(FDPC) 码作为基础数据集，分析了其权重分布和最小距离保证。

**📈 对比分析**

与其他方法相比，提出的qLDPC码在量子块长度N<10^5的情况下，保证了从约0.35%到25.8%的速率和12到69的认证量子距离，稳定器权重在8到16之间。

**⚠️ 局限性**

限制在于当前的构造主要依赖于FDPC的结构，未来的工作需要进一步优化稀疏化选择和底层排列，以提高有限长度构造和最大似然估计的性能。

---

## 506. MakoXC: Rearchitecting DFT Exchange-Correlation with Matrix-Aligned and Knowledge-Organized Sparsity

**arXiv ID:** 2609.01025 | [PDF](https://arxiv.org/pdf/2609.01025v1)

**作者:** Haozhi Han `[一作]`, Kun Li `[通讯]` (Tsinghua University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并实现了 MakoXC 框架，将 DFT 中的交换–相关（XC）评估重构为矩阵对齐、知识组织的稀疏计算，并在 AI 加速器（如 NVIDIA A100 Tensor Core）上实现了近线性可扩展性能。

**💡 创新点**

创新点：
- Matrix‑Aligned Cells（MAC）：将电子近视导致的全局稀疏重新聚合为密集、Tensor Core 对齐的微矩阵；
- Sparsity‑Guided Activation（SGA）：双阶段激活策略，先通过阈值过滤显式稀疏，再利用 AO‑产品的隐式稀疏性进行精确激活；
- Kernel‑Fused Pipeline（KFP）：在 MAC 级别融合 GEMM 与 DOT，利用 SRAM 重用、对称折叠、Normcache gating，最大化 Tensor Core 吞吐，消除碎片化与冗余。

**🔧 技术方法**

技术手段：CSR‑style MAC 存储格式；基于 shell‑pair 筛选的 AO‑产品显著性判断；阈值筛选与 Normcache gating；对称折叠；Tensor Core GEMM/DOT 融合；GPU A100 加速，CUDA/SM 内存层次利用。

**📊 数据集**

数据集：
- 多尺度分子系列（多聚甘氨酸链、硼氮烯片、氢氧化水簇）；
- 大分子 ubiquitin（1,231 原子，def2‑SVP/def2‑TZVP）。

**📈 对比分析**

比较方法与性能：在单 NVIDIA A100 上与 DenseXC、GPU4PySCF、GauXC 进行基准对比；
- def2‑SVP：平均 87.1× DenseXC、6.7× GPU4PySCF、48.4× GauXC；
- def2‑TZVP：相同基线下 48.4× DenseXC、3.6× GPU4PySCF；
- 整合后 DFT 包：3.94× GPU4PySCF；
- 多 GPU 扩展（64 A100）：74% 并行效率，ubiquitin 单点能量 <5 分钟。

**⚠️ 局限性**

局限性：
- SGA 仅在 SCF 循环前执行，需在不同 SCF 收敛策略下评估适用性；
- 对非 Gaussian 基础集或其他 XC 函数的适配需要进一步验证；
- 极大体系下 GPU 内存受限，需改进分块或压缩策略；
- 当前评测聚焦 A100，缺乏对 V100、H100 等其他 GPU 架构的全面验证；
- 并行通信主要通过最终归约，进一步优化全局通信与负载均衡仍是挑战。

---

## 507. Uncovering and Mitigating Aggregation-Induced Reward Hacking in Multi-Reward Reinforcement Learning

**arXiv ID:** 2609.00213 | [PDF](https://arxiv.org/pdf/2609.00213v1)

**作者:** Yu Yuan `[一作]` (University of Science and Technology of China), Qi Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 147414 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文识别了多维奖励投影导致的reward‑profile collapse与aggregation‑induced reward hacking，并提出Adaptive Multi‑Reward Projection（AMRP）这一自适应投影方法，动态在线调整多奖励权重以提升大型语言模型后训练的性能。

**💡 创新点**

创新点包括：①首次系统阐述投影导致的奖励别名与shortcut lock‑in现象；②设计三信号（相对短缺、奖励波动、最近进展）融合的自适应权重更新规则，轻量且可在训练过程中实时更新；③通过AMRP实现对不同奖励维度的动态平衡，显著降低短路策略的出现。

**🔧 技术方法**

技术手段：多奖励强化学习框架（GRPO、PPO、GDPO）；AMRP权重更新公式（软阈值、乘积门控）；指数移动平均（EMA）估计奖励均值、方差与进度；使用softplus和指数衰减等非线性变换；对奖励向量进行归一化与均一化处理。

**📊 数据集**

使用的数据集包括：数学推理任务 MATH‑500、AMC、AIME；citation‑grounded 生成任务 ASQA、ELI5；开放式对齐任务 AlpacaEval、ArenaHard、MT‑Bench；并在 Qwen3‑4B‑Instruct、DeepSeek‑Math‑7B‑Base、Llama‑3.1‑8B‑Instruct 等基础模型上进行 fine‑tuning。

**📈 对比分析**

评估方法：将 AMRP 与静态等权重、DRBO‑inverse、DRBO‑delta 等动态加权基线以及不同 RL 算法（GRPO、PPO、GDPO）进行对比。实验结果显示，AMRP 在所有三类任务中均获得显著提升：数学推理中准确率提升约 10‑15%（尤其是 AMC、AIME），citation‑grounded 生成中 correctness 与 citation 兼顾，开放式对齐中整体 ArmoRM 分数均高于基线，证明 AMRP 在多奖励设置下有效改善 reward‑profile 平衡与下游性能。

**⚠️ 局限性**

局限性：①实验仅覆盖规则奖励、自动评估器和学习奖励模型，未探讨更大规模、更多冲突维度或代码生成、工具使用等场景；②未对奖励本身的偏差进行验证，AMRP 只能调节权重，无法消除奖励误差；③训练规模受限于现有资源，缺乏大规模全参数 fine‑tuning 证据；④对安全与伦理风险的评估仍不足，需配合更严格的安全约束。

---

## 508. iPINN for Broadband CARS Phase Retrieval: A Framework for Function Approximation and Inverse Modeling Problems in Nonlinear Spectroscopy

**arXiv ID:** 2609.00883 | [PDF](https://arxiv.org/pdf/2609.00883v1)

**作者:** Ravi Teja Vulchi `[一作]` (Friedrich Schiller University Jena), Thomas Bocklitz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种逆向物理信息神经网络（iPINN），从BCARS光谱直接预测Lorentzian峰参数并通过可微物理解码器重建振动谱，实现相位检索；

**💡 创新点**

创新点在于：①逆向先行，将物理正向模型嵌入网络输出，避免显式NRB估计；②使用Transformer编码器与可学习峰槽和多视角一致性损失，提升对NRB形状、强度及噪声变化的鲁棒性；③通过物理一致性损失和自监督一致性损失实现跨测量条件的参数不变性；

**🔧 技术方法**

技术包括Transformer自注意力、可学习class token、旋转位置编码、可微Lorentzian正向层、物理一致性损失、多视角一致性损失、半监督训练与混合精度训练；

**📊 数据集**

使用公共BCARS相位检索基准（5个实验谱，2种溶剂）和自制焦深实验数据集（7种溶剂、4个焦深，共28谱）做评估；合成数据在训练中按程序生成；

**📈 对比分析**

与SpecNet、BiLSTM、CNN‑GRU、GAN、VECTOR等5个DL基线在公共基准上比较，iPINN MAE最低（0.0156 vs 0.046），在焦深数据上MAE在不同深度基本平坦，CV低，显示出优异的鲁棒性；基线模型残留NRB、伪峰、噪声问题更明显；

**⚠️ 局限性**

局限性包括：①正向Lorentzian模型无法完全拟合所有NRB形状，导致某些溶剂在深度变化时CV偏高；②峰数上限为24，难以处理生物样本高峰密度；③合成到真实数据的差距仍存在，需进一步改进输入侧物理先验和后向NRB细化等。

---

## 509. Replacing Training with Memory: Listwise Selection for Text-to-SQL

**arXiv ID:** 2609.00834 | [PDF](https://arxiv.org/pdf/2609.00834v1)

**作者:** Yeonseok Jeong `[一作]` (Seoul National University), Seung-won Hwang `[通讯]` (Seoul National University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种无微调的列表式 SQL 选择器，利用检索内存和多序列聚合实现候选 SQL 的联合排序。

**💡 创新点**

创新点在于用检索得到的结构化内存作为显式决策准则，和基于执行结果分组的 permutation 聚合来消除位置偏差，完全避免了对选择器的微调。

**🔧 技术方法**

使用结构化内存检索（基于 dense retriever）、列表式重排序、滑动窗口推理、组内/组间 permutation 以及点式 tie‑break。

**📊 数据集**

在 BIRD、Spider、EHRSQL 三大 Text‑to‑SQL 基准上进行实验。

**📈 对比分析**

与 pointwise、pairwise 以及多选择器基线相比，在 BIRD 上提升约 2.0 分执行准确率，同时调用次数和 token 数减少 6–27 倍；对比 fine‑tuned 的 R³‑SQL 切片略高于其 71.84%。

**⚠️ 局限性**

局限在于仍受生成器质量限制，未突破 75–76% 的 SOTA；对极大 schema 的可扩展性未验证；未探索更高级的协同排名方法。

---

## 510. OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization

**arXiv ID:** 2609.00066 | [PDF](https://arxiv.org/pdf/2609.00066v1)

**作者:** Yishan Yao `[一作]` (South China University of Technology), Zhiwen Yu `[通讯]` (South China University of Technology)

**通讯引用:** 15880 | [OpenAlex ID](https://openalex.org/A5100701166)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对NVFP4量化块进行通道重排，将激活中的极端值与低幅值通道配对，从而提升低位量化的精度；

**💡 创新点**

提出了“Collateral Quantization Error”概念并基于此设计Outlier‑Companion Grouping（OCG）策略，实现对激活块内量化误差的主动降低；

**🔧 技术方法**

采用通道RMS统计进行通道重排、GPTQ重构权重、在线NVFP4量化与自定义CUDA核融合以及块级归一化等技术；

**📊 数据集**

在Llama3与Qwen3模型上使用WikiText‑2、LM‑Eval（ARC‑Easy/Challenge、BoolQ、HellaSwag、LAMBADA、PIQA）以及MMLU、GSM8K等数据集进行评估；

**📈 对比分析**

与RTN、SmoothQuant、QuaRot、GPTQ、MR‑GPTQ、ARCQuant等PTQ方法对比，OCGQuant在WikiText‑2上获得最低perplexity、在零样本评测中平均精度提升至FP16的94.1%‑98.7%，预填充速度提升最高达2.29×，并在解码时保持与RTN相同的峰值内存占用；

**⚠️ 局限性**

仍缺乏针对量化网格的精细化通道选择标准，且激活侧与权重量化目标未统一，可能导致在某些模型上权重重排与激活重排冲突；

---

## 511. Transferable End-to-End Optimization for Indirect Long-Term Memory Poisoning in LLM Agents

**arXiv ID:** 2609.00523 | [PDF](https://arxiv.org/pdf/2609.00523v1)

**作者:** Chuanchao Zang `[一作]` (Shandong University), Shanqing Guo `[通讯]` (Shandong University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出 PipePoison，针对 LLM 代理的间接长期记忆中毒攻击，将写入-检索-利用三阶段视为一个整体端到端优化问题；

**💡 创新点**

创新点在于识别并优化三阶段间的耦合与瓶颈，通过链式损失与自适应配置/阶段权重实现对不同记忆/代理环境的迁移与鲁棒性；

**🔧 技术方法**

采用本地阴影系统收集细粒度阶段反馈，利用链式损失、加权策略与提示生成器实现可迁移的文本中毒内容；

**📊 数据集**

实验使用 LongMemEval、LoCoMo、BEAM 三个公开记忆评估数据集，共 300 个攻击任务；

**📈 对比分析**

在 12 种记忆-代理组合中，PipePoison 的攻击利用率提升 19.1pp（平均 73.4%），在未见配置的迁移场景中仍比最强基线高 16pp，且跨配置方差最低；

**⚠️ 局限性**

局限性包括对防御机制仍有一定易感性、在工具输出多的情况下效果下降、需要多次迭代且依赖阴影系统与目标环境的相似性。

---

## 512. Does Reasoning Mitigate Backdoor Attacks? A Neuro-Symbolic Perspective

**arXiv ID:** 2609.00464 | [PDF](https://arxiv.org/pdf/2609.00464v1)

**作者:** Marco Antonio Corallo `[一作]` (Orebro University), Alberto Giaretta `[通讯]` (Orebro University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究神经符号模型（DeepProbLog）在后门攻击下的鲁棒性，系统评估了其与纯神经网络在四种推理任务上的表现。

**💡 创新点**

首次量化符号推理对后门攻击的影响，提出可达性指标评估目标可实现性，并揭示符号约束能阻止逻辑不可达的攻击但无法抵御可达攻击。

**🔧 技术方法**

使用DeepProbLog与对照的全连接神经网络，结合BadNet、WaNet、FTrojan、ReFool四类后门攻击。

**📊 数据集**

在MNIST算术（加法、乘法）、FashionMNIST上下文化绑定、SDDOIA自动驾驶等四个公开数据集上进行实验。

**📈 对比分析**

通过对照实验比较干净准确率与攻击成功率，结果显示符号模型在目标不可达时攻击成功率降至0，若目标可达则鲁棒性提升有限，整体性能提升取决于任务与目标的可达性。

**⚠️ 局限性**

实验仅限单输入推理任务，未考虑多感知元素；攻击模型仅污染最终标签，未涵盖对概念层的污染；未验证更强灰盒攻击或其他神经符号架构的鲁棒性。

---

## 513. Creative Generation via Multi-Agent Debate: Does Debate Suppress Diversity?

**arXiv ID:** 2609.00683 | [PDF](https://arxiv.org/pdf/2609.00683v1)

**作者:** Tien Anh Nguyen `[一作]` (Deakin University), Hung Le `[通讯]` (Deakin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一种名为Creative-MAD的多代理辩论框架，针对创意生成任务中多样性被收敛压制的问题，利用认知视角分配与基于嵌入的同行筛选保持会话内多样性，从而提升跨运行多样性而不损失输出质量。

**💡 创新点**

理论证明保持会话内多样性是提升跨运行多样性的必要条件，并首次将认知视角分配与语义距离选择同行两种机制结合，以对抗身份漂移和多数拉拢两大导致多样性衰减的机制。

**🔧 技术方法**

使用多代理辩论（MAD）框架，认知视角分配（CLA）、基于嵌入的同行筛选（EPS）、LLM-as-Judge评估、Vendi Score和Div-BLEU等多样性度量。

**📊 数据集**

四个创意生成基准：LiveIdeaBench、Argument Annotated Essays、MacGyver和Arena Hard v2.0。

**📈 对比分析**

与单代理、Self-Refine、Voting、Homo MAD和Hetero MAD等方法对比，Creative-MAD在语义和词汇多样性上提升约20–30%，同时保持与标准MAD相当的质量得分和胜率。

**⚠️ 局限性**

主要限制在于依赖LLM评估可能偏向规范化输出；缺乏训练时多样性奖励机制；仅研究固定提示下的多样性，未探索运行级别多样性或更广泛的模型异质性。

---

## 514. Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs

**arXiv ID:** 2609.00155 | [PDF](https://arxiv.org/pdf/2609.00155v1)

**作者:** Deniz Bayazit `[一作]` (École Polytechnique Fédérale de Lausanne), Antoine Bosselut `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对多语语言模型的潜在语言识别（LLID）进行了测量比较，研究了表示式与解码式探针在不同模型、训练阶段、域、任务和语言上的一致性与差异。

**💡 创新点**

发现不同LLID探针并不测量同一现象，表示式探针更早显示跨语混合且对英语偏好较弱，而解码式探针保持更尖锐的语言特异性与英语偏好，说明现有探针揭示多语言处理的不同方面而非单一内部“通用语”。

**🔧 技术方法**

使用GMM基于表示的LLID探针、logitlens、tuned lens解码探针、以及GLotLID语言识别器，结合层级和rollout技术。

**📊 数据集**

使用PUD21、PUD9、INCLUDE问答数据集，以及基于合成任务的copy、cloze、translation提示，覆盖27种语言。

**📈 对比分析**

在控制（已知完成词）和开放式生成两种评估模式下，比较LLID分布的熵、优势、切换率与互相一致性；发现表示式探针的优势较早且更为平滑，而解码式探针在后层更强，但两者在层级上和训练进程中差异显著，互相一致性低。

**⚠️ 局限性**

仅评估7-9B参数规模的解码器模型，缺乏对更大规模或非解码器模型的验证；探针选择（GMM、tuned lens、rollout方法）和参数设置可能影响结果；未探讨因果干预或神经元级别解释；LLID估计缺乏真值，难以判断哪个更贴近内部计算。

---

## 515. Figures as Programs: Recursive Generation of Editable Scientific Figures

**arXiv ID:** 2609.01006 | [PDF](https://arxiv.org/pdf/2609.01006v1)

**作者:** Yepeng Liu `[一作]` (University of California Santa Barbara), Yuheng Bu `[通讯]` (University of California Santa Barbara)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多智能体系统，递归构造可编辑的SVG程序来自动生成科学方法图表。

**💡 创新点**

创新点在于将图表拆解为层级子程序、通过父级控制跨边连接，以及利用渲染-批评循环对代码与图像双空间进行局部修复。

**🔧 技术方法**

主要技术包括基于LLM的SVG代码生成、递归树结构拆分、渲染与批评模块（使用Gemini 3.1 Pro）以及可追溯的缺陷修复规则库。

**📊 数据集**

使用NeurIPS 2025方法图案例集（292个案例）以及AutoFigure-Edit Benchmark进行自监督规则学习。

**📈 对比分析**

与多种基准（如Nano Banana Pro、GPT-Image-2、PaperBanana、Crafter、AutoFigure-Edit、Direct-SVG）对比，实验显示其在真实性、简洁性、可读性和美观性上均优于基线，特别是在内容真实性和可编辑性方面提升显著；在迭代编辑实验中，其向量化表示能更快、更稳定地提高图表质量。

**⚠️ 局限性**

局限性包括：仍依赖文本解析的准确性；递归拆分深度和子节点数量需要经验调优；生成速度相对慢；缺陷修复规则库可能无法覆盖所有场景，导致仍需人工干预。

---

## 516. Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling

**arXiv ID:** 2609.00949 | [PDF](https://arxiv.org/pdf/2609.00949v1)

**作者:** Kangjia Zhao `[一作]` (Zhejiang University), Jianwei Yin `[通讯]` (Zhejiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

我们设计并评估了一个基于动作类别的多轮工具调用诊断框架，能够将失败分为动作类别失调与执行失败两种模式，并在 BFCL 与 τ²-bench 上进行验证。

**💡 创新点**

创新点在于提出 Gold Action Recall（GAR）与 Acc≤GAR 的上下界诊断方法，揭示了状态评分器掩蔽的动作类别失调，并通过 SRI/CRI 推理时干预探测了校准可塑性。

**🔧 技术方法**

采用四类动作空间（tool_call/ask/refuse/confirm），计算 GAR 与 Acc，设计 SRI 与 CRI 两种推理时干预，使用 VLLM 与多款 API 进行推理，并进行 200 条案例人工审计。

**📊 数据集**

使用的数据集包括 BFCL v3 multi-turn benchmark、τ²-bench（零售/航空）以及多款开源/闭源模型的工具调用数据。

**📈 对比分析**

与传统状态评分器和 Acc 对比后，发现模型族间误差差异显著（高达 ±66pp），SRI/CRI 干预可在同一模型上提升 Acc 10‑20pp，验证了校准可塑性。

**⚠️ 局限性**

局限性在于诊断不提供训练改进方案，覆盖仅限于所选的两个 benchmark，且 SRI/CRI 需要已知动作类别，无法直接在部署中使用。

---

## 517. S^3martCirc: Self-supervised Smart Circuit Discovery

**arXiv ID:** 2609.00755 | [PDF](https://arxiv.org/pdf/2609.00755v1)

**作者:** Wendy Zheng `[一作]` (University of Virginia), Jundong Li `[通讯]` (University of Virginia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种统一的机制可解释框架 S³martCirc，能够同时发现大语言模型的任务相关电路并给出节点的功能分类。

**💡 创新点**

创新点在于把节点功能抽象为两类通用可量化的计算角色，并通过双向交替优化同时学习节点重要性与功能，消除了传统两阶段分离的缺陷。

**🔧 技术方法**

使用了软硬掩码学习、KL 与交叉熵结合的发现目标、相似度变化的功能度量、稀疏正则、以及温度上升的 warmup 训练技术。

**📊 数据集**

实验使用 GPT‑2、Llama 3.2 1B、Qwen 3 1.7B 三大 LLM，并在间接宾语识别（IOI）和首字母预测（Acronym）两任务上评估。

**📈 对比分析**

与 Patch、Attr、Attr‑IG、IBCircuit、Prune、Random 等基线比较，S³martCirc 在准确率下降和节点回收率上均超过 80% 以上，且能更好地恢复已知电路。

**⚠️ 局限性**

局限在于仅给出两类粗粒度功能，无法捕获细粒度手工标签的多功能性，并且同一节点在不同任务下可能被赋予不同角色。

---

## 518. Phrase-Localized Language-Contrastive Guidance: Training-Free Localized Accent Control for Code-Switching Text-to-Speech

**arXiv ID:** 2609.01016 | [PDF](https://arxiv.org/pdf/2609.01016v1)

**作者:** Che Hyun Lee `[一作]` (Seoul National University), Sungroh Yoon `[通讯]` (Seoul National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

为跨语言文本到语音系统中的词组级代码切换提供一种训练无关、推理时可控的方言/口音调整方法。

**💡 创新点**

创新点包括：①利用自注意力内部探测动态提取词组边界；②对掩码进行双标签联合与边缘膨胀以增强召回；③引入可独立调节的对比尺度 λ，实现在局部语音片段中与全局文本指导解耦的“Phrase‑Localized Language‑Contrastive Guidance（LCG）”。

**🔧 技术方法**

技术手段主要为：离散扩散语言模型（OmniVoice）下的logit级对比引导；自注意力探测+掩码细化；对比尺度 λ 的位置级调节；与传统的全局CFG、Swap 进行对比。

**📊 数据集**

使用 1,200 条合成代码切换语料（5 种语言，12 方向，均衡 100 句/方向），并在 5 种语言对上评估；参考了离散扩散模型 OmniVoice 的预训练模型。

**📈 对比分析**

与基线全局 CFG、Swap 以及离线强制对齐掩码进行对比；评估指标包括 MER、语言准确率 LA_e、语言识别置信度 LID_e、全局自然度 UTMOS 以及人类 MOS。实验显示 LCG 在 λ=7 时将 MER 从 0.564 降至 0.445，LA_e 从 0.233 提升至 0.518，LID_e 由 0.247 提升至 0.588，同时保持全局自然度与说话人一致性几乎无损。

**⚠️ 局限性**

局限性包括：语料规模相对有限且仅覆盖高资源语言，方法依赖离散扩散模型且难以直接迁移至自回归模型；需要已知词组字符边界；合成语料可能与真实对话中的代码切换行为存在差异。

---

## 519. SciTrue: Reliable Scientific Claim Validation with Frontier and Open Language Models at the NTCIR SciClaimEval Task

**arXiv ID:** 2609.00654 | [PDF](https://arxiv.org/pdf/2609.00654v1)

**作者:** Qiming Bao `[一作]` (University of Auckland), Mark Gahegan `[通讯]` (University of Auckland)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估 11 种前沿与开源视觉语言模型，结合无泄漏的 pair prior、证据类型路由和轻量后处理，实现 NTCIR-19 SciClaimEval 子任务 1 和 2 的科学主张验证。

**💡 创新点**

创新点在于利用任务内“同一声明对应支持/否定”对的结构构建无泄漏的 pair prior，显著提升 pair‑accuracy，同时通过证据类型路由与多模型加权融合实现最优性能。

**🔧 技术方法**

使用指令调优的 VLM（Claude Opus 4.8、Gemma‑4‑31B、Qwen3‑VL 等），配合分数加权融合、任务先验、图像‑论文一致性检查与 QLoRA 细调等技术。

**📊 数据集**

使用 NTCIR-19 SciClaimEval 公开数据集，包括 747/917 个主张、352 对照样本，表格和图像为 PNG 并提供结构化表格数据。

**📈 对比分析**

在官方测试集上以 pair‑accuracy（子任务1）和准确率（子任务2）为主指标，SciTrue 以 93.5/96.2 的 pair‑accuracy/accuracy 获得三项冠军并与竞赛第一名平手，优于所有对手。

**⚠️ 局限性**

局限在于假设测试集保持支持/否定对的构造、剩余错误多为不可视标签交换或数据噪声、对模型规模和图像理解提升空间有限。

---

## 520. Conditional Flow Matching for Cross-Field MRI Harmonisation

**arXiv ID:** 2609.00960 | [PDF](https://arxiv.org/pdf/2609.00960v1)

**作者:** Baris Imre `[一作]` (Leiden University Medical Center), Efe Ilicak `[通讯]` (Leiden University Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种条件流匹配模型，用单一模型完成跨磁场强度与三种对比度的脑MRI翻译。

**💡 创新点**

创新点包括：①使用条件流匹配直接传输已配准切片，避免噪声起点；②分三阶段训练，先在未配对数据上学习恢复先验，再在极少配对数据上细化映射；③在最终阶段加入对抗细化以提升纹理细节；④通过Heun求解器实现低步数推断。

**🔧 技术方法**

核心技术包括：条件流匹配、FiLM特征调制、2D U‑Net速度场预测网络、Heun ODE求解器、PatchGAN对抗判别器、指数移动平均、三阶段训练策略。

**📊 数据集**

使用MRIxFields2026数据集：五个磁场强度（0.1T、1.5T、3T、5T、7T）与三种对比度（T1w、T2w、T2‑FLAIR），训练集包括一个大规模未配对队列和一个仅三名受试者的行程队列。

**📈 对比分析**

与两种基线（直接回归和扩散模型）在相同网络与训练流程下对比，模型在所有20个有向磁场对和三种对比度上平均SSIM为0.909、LPIPS为0.089、nRMSE为0.227，明显优于回归（SSIM 0.906、LPIPS 0.110、nRMSE 0.246）和扩散（SSIM 0.896、LPIPS 0.121、nRMSE 0.249）。

**⚠️ 局限性**

局限性：仅在2D切片上工作，无法保证体层一致性；对抗细化可能产生纹理伪影；训练数据对配对样本极度稀缺；未来需采用2.5D/3D模型并扩大配对数据量。

---

## 521. No Pixel Left Behind: Filling Gaps in Anime Colorization

**arXiv ID:** 2609.00800 | [PDF](https://arxiv.org/pdf/2609.00800v1)

**作者:** Masahiro Kono `[一作]`, Takeo Igarashi `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种名为GapFill的工具，自动检测并补填动画线稿中的细小未填色区域，降低专业色彩师的手动工作量。

**💡 创新点**

整合人工流程的设计原则、可视化高亮与弹出放大镜、可在圆内直接选色以及批量应用功能，同时使用基于区域对应的U‑Net深度学习模型进行颜色预测，兼顾人工控制与AI辅助。

**🔧 技术方法**

采用U‑Net深度网络进行局部颜色预测，基于区域对应的概率映射；前端实现了圆形高亮、弹出放大视图、拖拽选色、扫动批量填色等交互。

**📊 数据集**

使用了自制的专业日漫彩绘数据集（约1.8M彩色帧），并通过合成缺口的方式生成训练样本。

**📈 对比分析**

与传统工具（填充、漏勾、黑光法等）进行对比，13名专业色彩师的用户研究显示GapFill在检测填补细小缺口任务中平均减少约20–25秒完成时间且零漏填，主观满意度、可用性评分均显著提升；在颜色预测上准确率达81.7%。

**⚠️ 局限性**

局限性包括对非常小或高细节区域的预测仍存在错误，用户对AI误差的信任需要进一步提升；工具目前仅适用于无反走线的二值线稿，抗锯齿或彩色线稿需进一步扩展。

---

## 522. Membership Inference in Fine-tuned Diffusion Language Models via Token-level Memorization Asymmetry

**arXiv ID:** 2609.00873 | [PDF](https://arxiv.org/pdf/2609.00873v1)

**作者:** Shengfang Zhai `[一作]` (National University of Singapore), Sanghyun Hong `[通讯]` (Oregon State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对扩散式语言模型（DLMs）的隐私风险，提出了基于“token‑level memorization asymmetry”的成员推断（MI）方法，并将该指标用于个人身份信息（PII）重构攻击。

**💡 创新点**

创新点在于首次理论揭示了DLM训练过程中的掩码比例与单步记忆增益的反比关系，提出了量化该不对称性的分位加权偏度指标，并证明其对成员推断和PII重构均具备显著提升。

**🔧 技术方法**

核心技术包括：离散扩散模型训练与推理框架、逆掩码比率缩放假设、量化分位加权偏度计算（QSkew）、循环采样（Cyclic Sampling）以降低 Monte‑Carlo 方差，以及与传统交叉熵损失相结合的混合评分。

**📊 数据集**

实验覆盖了 LLaDA‑8B‑Base/​Instruct、Dream‑Base/​Instruct 等主流 DLM，并在域微调（ArXiv、WikiText、XSUM）和指令微调（MedQA、Alpaca、Tulu‑3）共六大数据集上验证。

**📈 对比分析**

在 AUC、TPR@10%FPR、TPR@1%FPR 等指标上，新方法相较于 Loss、Min‑K、Calibration、SecMI、SAMA 等基线平均提升 10%+ 的 AUC；在 PII 重构任务中，结合交叉熵与偏度的混合评分将 top‑1 准确率从 30%/20% 提升到 34%/24%。

**⚠️ 局限性**

局限性包括：仅在目前公开的主流 DLM 上验证，缺乏对未来更复杂扩散架构的泛化评估；以及对参考模型对齐假设的依赖，虽然误差有限但仍需进一步研究。

---

## 523. EarthLD: Towards Unified Open-World Landslide Understanding via Vision-Language Guided Diffusion Models

**arXiv ID:** 2609.00712 | [PDF](https://arxiv.org/pdf/2609.00712v1)

**作者:** Yuanchao Su `[一作]` (University of Macau), Yicong Zhou `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 EarthLD 的一体化框架，利用视觉‑语言引导的扩散模型实现滑坡的识别、范围映射、触发事件解释、计数与定位。

**💡 创新点**

创新点包括：① 将滑坡检测重新定义为逐步去噪的扩散过程，天然适应形状不规则的目标；② 采用 CLIP 对元数据进行上下文引导，使模型在跨传感器、跨地区时具备更强的通用性；③ 构建全球规模的开放式滑坡基准数据集，覆盖 6 大洲 28 区域，促进跨域学习；④ 统一同时完成目标检测与二值语义分割，显著简化工作流程。

**🔧 技术方法**

核心技术包括：变分扩散模型（VDM）用于检测与分割的去噪；双向特征金字塔网络（BiFPN）提取多尺度视觉特征；CLIP 生成文本与图像嵌入并计算交叉模态权重；迁移学习与预训练的生成式基准模型；自监督的 NCE 损失与触发估计损失。

**📊 数据集**

使用了 GDCLD、Bijie、CAS、GVLM、Haiti 等公开数据集，经过统一预处理后汇总为 100,000+ 标注实例，覆盖 28 个地理区域、6 种传感器与多种触发事件。

**📈 对比分析**

与 Faster R‑CNN、Sparse R‑CNN、DAB‑DETR、DINO、YOLOv11、DiffusionDet（检测）以及 DeepLabV3+、U‑Net++、SegFormer、DINOv2、YOLO11s‑seg、Seg‑Diffusion（分割）等基线进行对比。EarthLD 在 10 个不同地区的测试集上实现了最高的 AP、精度和召回率，并在 SAR 与光学影像、全景大图等多模态与大尺度场景下保持稳定优势。

**⚠️ 局限性**

局限性包括：① 训练与推理成本较高，扩散模型的多步迭代耗时；② 仍依赖大量人工标注的滑坡实例，缺乏完全无监督的方案；③ 在极端噪声或极低分辨率图像下性能可能下降；④ 触发事件解释仅基于元数据文本，未充分融合物理模型或实时传感器信息。

---

## 524. MemeBridge: A Dataset for Benchmarking and Mitigating the Bidirectional Cultural Gap in Meme Interpretation

**arXiv ID:** 2609.00491 | [PDF](https://arxiv.org/pdf/2609.00491v1)

**作者:** Hangxiao Zhu `[一作]` (Texas A&M University), Meng Xia `[通讯]` (Texas A&M University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 MemeBridge 数据集，旨在评估和缓解美国与中国文化背景下对 meme 的双向误读；

**💡 创新点**

双向框架：既收集美国参与者对 meme 的解释和可能的跨文化误解，又评估中国参与者对同一 meme 的解读与情感识别；

**🔧 技术方法**

采用多阶段众包与 GPT‑4 辅助的文本清洗、质量分类、情感与情绪标注；在模型评估中使用多模态 LLM（Qwen、GLM、LLaMA、GPT‑4o）并进行角色扮演 (US‑RP/CN‑RP) 与微调实验；

**📊 数据集**

MemeBridge 数据集：621 张美国 meme，包含解释、预期误解、情感、情绪、文化与通用知识标签；

**📈 对比分析**

通过多模态 LLM 在多选题、情感与情绪分类任务中的准确率与 GPT‑4o 进行对比；实验显示 GPT‑4o 最优，其他模型在情绪识别上高于人类但情感识别仍低；微调提升部分任务但在 GPT‑4o 情绪分类上出现过拟合；

**⚠️ 局限性**

数据集规模有限导致微调过拟合；仅覆盖中美两种文化，未验证其他文化的双向误读；误解标签的主观性与真实性难以保证；

---

## 525. Don't You Know, Pump it Up! Investigating Cryptocurrency Manipulation in Telegram-Driven Activity

**arXiv ID:** 2609.01176 | [PDF](https://arxiv.org/pdf/2609.01176v1)

**作者:** Filipe Moura `[一作]` (Universidade Federal de Minas Gerais), Jussara Almeida `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种可扩展的框架，利用文本分类、异常检测与计量经济学方法识别并验证Telegram公共频道中的市场操纵行为。

**💡 创新点**

创新点在于：①结合细粒度语义过滤与自适应阈值检测捕获大规模社交爆发；②采用RDD‑DiD双重检验分离操纵与自然市场反应；③发现操纵信息与价格峰值之间存在显著时间前后差异；④提供公开的加密词典与分类模型。

**🔧 技术方法**

技术主要包括：RoBERTa微调文本分类器、CA‑CFAR异常检测、回归不连续设计（RDD）与差分中的差分（DiD）计量经济学模型、LIWC心理语言分析。

**📊 数据集**

使用数据集为：Telegram公开频道一年的14,499个频道共20M条信息；17,000多种加密资产的交易与价格数据；18,704词汇的加密词典；2,923条标注的Telegram消息用于模型训练。

**📈 对比分析**

通过与零射击、上下文学习等对比，RoBERTa微调在召回率94%、F1 92%表现最佳；异常检测识别9799次社交爆发，随后RDD发现302次价格跳跃，DiD筛选后得到47起泵‑dump事件与73起持续反应，表明框架能显著区分操纵与自然波动。

**⚠️ 局限性**

局限性包括：仅基于公开频道，可能遗漏私密群组预演；方法为事后检验，对实时预警有限；语言模型对语义模仿不敏感，需结合时序特征；在极端市场波动时控制组可能失效。

---

## 526. WiSDoM: Wireless Sparse Decision Transformer with Mixture-of-Experts for Multi-Task Mobile Network Optimization

**arXiv ID:** 2609.00284 | [PDF](https://arxiv.org/pdf/2609.00284v1)

**作者:** Fatih Temiz `[一作]` (University of Ottawa), Melike Erol-Kantarci `[通讯]` (University of Ottawa)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了WiSDoM框架，结合离线决策变压器与稀疏专家混合网络，实现多任务CoMP多小区选择的统一决策策略；

**💡 创新点**

创新点在于将PromptDT与基于Token的稀疏Mixture‑of‑Experts相结合，实现专家动态路由与少量参数激活，既提升模型容量又降低推理成本，同时通过任务梯度相似度聚类实现专家专化；

**🔧 技术方法**

使用的技术包括离线强化学习、决策变压器、PromptDT、稀疏Mixture‑of‑Experts（Top‑K路由）、辅助负载平衡损失、零填充与动作屏蔽；

**📊 数据集**

使用36个CoMP任务的离线轨迹数据（每个任务2000条演化轨迹），任务覆盖不同基站/用户数量、移动速度与调度策略；

**📈 对比分析**

与启发式阈值策略、单任务PPO和密集PromptDT对比，WiSDoM在QoE上提升至55%，率失效率显著下降，推理时激活参数仅为密集模型的约1/3，且在未见任务上仍能优于PPO；

**⚠️ 局限性**

局限性包括依赖高质量离线数据、专家数量与路由复杂度随任务增长而上升、对Prompt质量敏感、尚未验证对更大规模或不同网络功能的泛化能力。

---

## 527. Stochastic complexity of vectors containing cluster structure

**arXiv ID:** 2609.00084 | [PDF](https://arxiv.org/pdf/2609.00084v1)

**作者:** Daniel Nicorici `[一作]` (Medicel Oy), Jaakko Astola `[通讯]` (Tampere University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对聚类向量的 NML 代码长度进行快速计算，提出线性时间递归公式。

**💡 创新点**

通过生成函数的解析推导得到新的递归关系，将之前多项式时间的归一化常数计算提升至线性时间。

**🔧 技术方法**

生成函数分析、递归公式、归一化常数计算、MDL 近似与 NML 模型。

**📊 数据集**

在前期工作中使用微阵列基因表达数据作为实例验证，但本文主要聚焦理论与算法实现。

**📈 对比分析**

与以往的多项式时间方法对比，实验表明计算时间从 O(n^2) 降低到 O(n)，显著提升效率，且在大规模数据集上仍能保持稳定性能。

**⚠️ 局限性**

仅针对编码聚类向量的 NML 代码长度；对聚类质量或不同模型的实际选择没有直接实验评估，且在极大 m 值时递归稳定性尚待进一步验证。

---

## 528. GeoPAR: Large-Scale Multi-Agent Combinatorial Optimization with Geometry-Guided Parallel Autoregressive Learning

**arXiv ID:** 2609.00577 | [PDF](https://arxiv.org/pdf/2609.00577v1)

**作者:** Wenjian Wu `[一作]` (Soochow University), Jin Wang `[通讯]` (Soochow University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

无法确定

**💡 创新点**

无法确定

**🔧 技术方法**

无法确定

**📊 数据集**

无法确定

**📈 对比分析**

无法确定

**⚠️ 局限性**

无法确定

---

## 529. FractalNet-Based Heterogeneous Federated Learning for Orbital Edge Intelligence in Satellite Mega-Constellations: A Wildfire Case Study

**arXiv ID:** 2609.00875 | [PDF](https://arxiv.org/pdf/2609.00875v1)

**作者:** Sai Puppala `[一作]` (Southern Illinois University), Koushik Sinha `[通讯]` (Southern Illinois University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计了一个基于 FractalNet 的异构联邦学习框架FN‑HFL，用轨道层级与 SWaP‑C 约束实现深度分层训练与聚合，支持在大规模卫星星座中的边缘智能；

**💡 创新点**

创新点在于①将路径深度与卫星计算深度共设计，形成层级化的模型训练；②引入动态深度分配与分层调度器；③设计层级鲁棒聚合和一致性检查；④实现分层池化（scheduled pooling）显著降低能耗；

**🔧 技术方法**

技术实现包括 FractalNet 架构、分层联邦学习、轨道路径调度器、层级鲁棒聚合（trimmed‑mean/Byzantine‑robust）、分层能耗池化、梯度可信度评估，以及离散事件轨道与能源/辐射失效仿真；

**📊 数据集**

使用了合成多光谱火灾检测数据集（约240万样本），以及入侵检测和卫星健康预测数据集进行泛化评估；

**📈 对比分析**

与 FedAvg‑full、FedAvg‑small、Hierarchical FL、HeteroFL、DepthFL、Split‑FL 等六个基线比较，FN‑HFL 在中型星座下在 80% AUROC 目标上收敛 1.7 倍快、通信量减少 53%、能耗下降 7%，且在攻击与辐射失效场景下鲁棒性提升，时间敏感检测平均仅需 2.3 轮；

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未在真实星座部署；假设火灾信息层级严格对应轨道层级，可能在不同任务中失效；调度与鲁棒策略需依赖场景政策和阈值；对火势快速变化的自适应性有限；分层池化对低优先级异常可能降低及时性。

---

## 530. Graph Coloring with Color Preferences

**arXiv ID:** 2609.00569 | [PDF](https://arxiv.org/pdf/2609.00569v1)

**作者:** Tomohiro Koana `[一作]` (University of Tokyo), Hirotaka Yoneda `[通讯]` (University of Tokyo)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了在有颜色偏好的图着色模型——稳定着色，并定义了稳定色数（stable chromatic number）

**💡 创新点**

创新点在于将偏好和稳定性结合，给出了稳定色数的上下界（通过可达性和Grundy数），证明了稳定k-可着色的NP难度与2-可着色可解，提出了按树宽度参数化的FPT算法

**🔧 技术方法**

主要技术包括：基于无向图无环定向的可达性上界、Grundy着色下界、树分解与分离器性质、动态规划求解最小稳定着色、树宽度与颜色数的O(t log n)上界等

**📊 数据集**

论文不使用实验数据集，全部为理论分析与证明

**📈 对比分析**

由于是理论工作，没有实验对比；理论上给出了多类图的色数界限，并证明了算法复杂度（O(t^2)指数因子 + 多项式）

**⚠️ 局限性**

局限性：对平面图的稳定色数仅给出上、下界，未给出精确值；部分结果依赖于树宽度和分解的存在；在实际应用中对偏好模型的假设较为理想化

---

## 531. RecalibrateGPT: AI Fatigue Resilient Conversational Interfaces

**arXiv ID:** 2609.00506 | [PDF](https://arxiv.org/pdf/2609.00506v1)

**作者:** Nikhil Wani `[一作]` `[通讯]` (OpenThreads AI Research), Nikhil Wani (OpenThreads AI Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 RecalibrateGPT 系统，通过五种跨轮操作器（Anchor、Replay、Delta、Scope、Steer）实现一次点击即可校正完整对话，从而缓解用户在使用大语言模型时的对话疲劳。

**💡 创新点**

创新点在于首次将跨轮操作器设计为单击式工具，直接作用于完整对话历史而非单条回复，并提供三种几何布局（Vertical、Arc、Tablet）以匹配不同疲劳场景。

**🔧 技术方法**

技术实现基于 Python 调用 OpenAI GPT‑5.5 接口，结合句子嵌入、余弦相似度、KL 散度与主题聚类等 NLP 方法完成目标对齐、差异检测、摘要与子主题选择。

**📊 数据集**

研究使用 12 名高级 LLM 用户的对话日志和健康护理场景对话，收集了 96 条重新提示实例作为实验材料。

**📈 对比分析**

在同一受试者的对照实验中，RecalibrateGPT 将 NASA‑TLX 工作负荷评分从 5.4 降至 2.7，SUS 可用性评分达 86.5，显著降低用户疲劳感。

**⚠️ 局限性**

局限性包括样本规模仅为 12 名受试者、实验仅聚焦医疗对话、缺乏对大型商业接口的直接对比，以及未实现自动疲劳检测与长期使用评估。

---

## 532. BiMTokenizer: Preserving Semantic-Acoustic Balance in Low-Bitrate Speech Tokenization via Bidirectional State-Space Modeling

**arXiv ID:** 2609.00562 | [PDF](https://arxiv.org/pdf/2609.00562v1)

**作者:** Xin Zhang `[一作]` (Wuhan University of Technology), Kong Aik Lee `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了一种基于双向状态空间模型和固定球面勒奇格子量化的1.1 kbps单塔语音编码器BiMTokenizer。

**💡 创新点**

创新点在于将双向Mamba骨干网络与Residual Spherical Leech Quantization（RSLQ）结合，既避免了双塔设计的参数与计算开销，又实现了语义与声学信息的高效联合编码。

**🔧 技术方法**

主要技术包括双向Mamba结构、RSLQ量化、冻结语义教师（Whisper‑small/SenseVoice）、GAN对抗训练以及语义对齐损失。

**📊 数据集**

使用的数据集包括LibriSpeech（训练/测试）、ARCH情感/意图/数字识别数据集以及Seed‑TTS‑Eval用于生成评估。

**📈 对比分析**

在LibriSpeech test‑clean、test‑other、Seed‑TTS‑Eval等基准上，与多种低比特率语音codec和双塔模型比较，BiMTokenizer在重建质量（PESQ、STOI、UTMOS、SIM）和语义保持（WER、ARCH任务准确率）上均优于或持平于对手，同时参数量和计算量显著降低。

**⚠️ 局限性**

主要限制包括：需离线（不可实时）处理；固定大容量量化表可能导致生成模型需要更多训练数据；当前实现为CNN‑Mamba混合架构，尚未完全统一为Mamba结构。

---

## 533. Don't Trust the Code, Check Its Effects: Runtime Refinement for Regenerated Systems Code Under an Adversarial Generator

**arXiv ID:** 2609.00430 | [PDF](https://arxiv.org/pdf/2609.00430v1)

**作者:** Jinhao Hu `[一作]` (Max Planck Institute for Software Systems), Laurent Bindschaedler `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出一种针对在生成式模型下重生成的系统代码（如设备驱动）的安全执行框架：通过将代码的执行权拒绝给生成的实现，只让其规划操作，所有实际效果由可信的中介（参考监视器）在验证后执行，从而在对抗性生成器和不可逆效果的环境下保证系统安全。

**💡 创新点**

创新点包括：
- 把信任焦点从代码本身转移到“效果权限”，即生成代码只负责规划，真正的动作由可信中介授权；
- 设计了“可调解包络”（mediability envelope）——六个对效果词汇的条件（可读性、规格输入可观测性、关联性、完整性、结果可枚举性、显式耐久性）来判定何时能实现此类中介；
- 将运行时细化检查（runtime refinement）与参考监视器结合，为重生成驱动提供了可验证的安全保障。

**🔧 技术方法**

使用的技术与方法：
- 参考监视器（reference monitor）与抽象状态管理；
- 运行时细化检查（runtime refinement）与功能细化（functional refinement）验证；
- 令牌化（token-based）方案来保证一次性提交与验证；
- 通过显式声明的效果词汇与抽象语义模型实现验证与授权；
- 结合能力系统、沙盒化、显式事务化等已有安全机制。

**📊 数据集**

论文为位置论文，并未提供实验数据或使用特定数据集，主要通过理论分析与设计说明展示概念。

**📈 对比分析**

未给出实验或性能对比；作者指出该框架在实际实现中需评估中介代码量、序列化延迟以及提议失效率，但具体数值未给出。

**⚠️ 局限性**

局限性与挑战：
- 需要构建可信中介，增加了系统的可信根与运行时开销；
- 中介的抽象状态管理与序列化锁可能影响性能和可用性；
- 对于高度并发或跨设备操作，仍需在效果词汇层面实现原子性或外部同步；
- 需要对效果词汇和规格进行完整性保证，否则无法满足可调解包络条件；
- 该方案主要针对不可逆效果和对抗性生成器，未涵盖所有系统代码场景，且缺乏实现与实测验证。

---

## 534. Bridging Lexical Divergence: LLM-Assisted, Cost-Efficient, Zero-shot Scientific Entity Linking

**arXiv ID:** 2609.00228 | [PDF](https://arxiv.org/pdf/2609.00228v1)

**作者:** Md Rasel Khondokar `[一作]` (Iowa State University), Qi Li `[通讯]` (Iowa State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出一种成本感知的零人工注释科学领域实体链接框架Sci-ZSEL，利用LLM为选定实体生成同义别名并结合本体过滤生成伪标注，再对检索器和重排序器进行微调；

**💡 创新点**

创新点在于将LLM应用于实体侧而非每个提及，显著降低生成成本；引入本体结构的漂移过滤器，有效剔除语义漂移；并发布高词汇差异的动物科学实体链接基准；

**🔧 技术方法**

技术主要包括：LLM（Llama 3.2 3B Instruct）生成别名、BLINK双编码检索器与ReS交叉编码重排序器、BioLORD用于本体相似度过滤；

**📊 数据集**

使用的语料包括五个基准：NCBI Disease、BC5CDR（医学本体），以及新发布的QTLCMO、QTLVT、QTLLPT（动物科学三大特征本体）；

**📈 对比分析**

在五个基准上，Sci-ZSEL在无微调基线上提升显著，尤其在无重叠（NO）案例中提升最高达+49%；结合LLM别名与人工本体同义词进一步获得最佳召回；

**⚠️ 局限性**

局限性包括对LLM域知识的依赖、仅支持英文、需要针对本体更新重新生成别名、未处理NIL实体以及对小规模本体的可扩展性仍待验证。

---

## 535. WorldBench: Culturally Grounded Benchmark for Multilingual Agents

**arXiv ID:** 2609.01056 | [PDF](https://arxiv.org/pdf/2609.01056v1)

**作者:** Leonardo Ranaldi `[一作]` (University of Edinburgh), Alexandra Birch `[通讯]` (University of Edinburgh)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了多语言、文化背景的文件工作流基准WorldBench，评估LLM代理在多步任务中的执行与环境状态保持。

**💡 创新点**

创新点包括：①在目标语言与文化环境中原生生成任务并人工审核；②引入Constrained Task Success (CTS) 评估度量，兼顾任务完成与非目标文件保持；③提供完整的任务构造与验证流水线。

**🔧 技术方法**

使用的技术包括LLM代理与结构化工具接口、确定性与LLM-as-a-Judge的双重评估函数、任务构造流水线及自动化生成与人工审核流程。

**📊 数据集**

使用的数据集为1,600个任务，覆盖7种语言（英美、意、葡、西、法、德、中文）和8种文化，包含文档、电子表格、邮件、日历等文件。

**📈 对比分析**

对9种前沿LLM模型进行比较，最佳模型（Gemini‑3.1‑Pro）CTS约49%，pass率高于preservation率，表明跨语言和长时序执行仍存在显著不足。

**⚠️ 局限性**

局限性在于仅涵盖文件/命令行交互，未包含视觉桌面或网页交互；LLM评估器的一致性未完全验证；每任务仅执行一次，缺乏多次评估的鲁棒性验证。

---

## 536. Visual Attention Faithfulness in Vision-Language Models is Heterogeneous

**arXiv ID:** 2609.00830 | [PDF](https://arxiv.org/pdf/2609.00830v1)

**作者:** Xurui Song `[一作]` (SAP), Jun Luo `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对视觉语言模型的注意力权重进行因果扰动实验，衡量其在视觉任务中的可靠性与必要性

**💡 创新点**

提出将注意力权重聚合后按令牌排序并通过去除/保留前k个令牌评估其可解释性，发现三种不同的“可信度模式”而非单一二元判断

**🔧 技术方法**

使用注意力权重聚合、零填充干扰、可解释性度量（comprehensiveness 与 sufficiency gap）以及Otsu阈值分区方法

**📊 数据集**

在VQAv2（视觉问答）、VRDU（文档信息抽取）和ChartQA（文档问答）等公开基准上进行实验

**📈 对比分析**

与人工标注的关键信息区域对比，结果显示模型注意力在约40%样本中比人类标注更能捕捉必要信息；在不同模型与规模下均保持三模式分布，表明方法具有可推广性

**⚠️ 局限性**

仅适用于使用动态分辨率视觉编码的VLM，未涵盖固定分辨率或需要多步推理的模型，且未对多模态推理深度进行深入研究

---

## 537. Connectivity-Aware Graph Extension for Decentralized Multi-Robot Exploration

**arXiv ID:** 2609.00804 | [PDF](https://arxiv.org/pdf/2609.00804v1)

**作者:** Béatrice Garcia Cegarra `[一作]` (AMIAD), David Filliat `[通讯]` (ENSTA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于前沿连通性的图扩展方法，用于多无人机在有限通信环境下的分布式探索，并将其与 Voronoi 与 K‑medoid 两种区域分区算法结合；

**💡 创新点**

创新点在于：①只在父节点集合属于同一连通分量时才添加新的前沿节点，避免无意义的短路；②通过拓扑信息扩展图，保持分区在通信中断时的稳定性；③与传统采样扩展相比，显著提升探索效率与分区一致性；

**🔧 技术方法**

采用的技术包括：3D体素网格建图、Dijkstra 与 Johnson‑Dijkstra 最短路径、Voronoi 分割、K‑medoid+Hungarian 关联、欧氏与地理距离计算、离散化前沿图、以及通过连通性检验实现的拓扑图扩展；

**📊 数据集**

实验数据集为两种仿真环境：Maze（约2560 m³）和 SubT（约13710 m³）地下/迷宫式地图，使用 3 台 UAV 进行 20 次重复实验；

**📈 对比分析**

与原始无扩展、采样扩展以及全局与局部通信模式对比。结果显示：在局部通信下，图扩展方法可将总探索时间降低约20–30 s、平均行程距离减少约15–20 m；Voronoi 分区受益最大；在全局通信下性能相近；K‑medoid 的标准差下降，表明探测路径更一致；

**⚠️ 局限性**

局限性包括：①对 K‑medoid 的改进在低通信时收益有限；②拓扑扩展依赖父节点连通性，在极度稀疏的图中可能无法生成新节点；③实验仅在 2‑D 平面仿真环境中进行，缺乏 3‑D 真实世界验证；④边权惩罚等参数需经验性调优。

---

## 538. CERF: Communication-Efficient and Retraining-Free Collaborative Perception

**arXiv ID:** 2609.00951 | [PDF](https://arxiv.org/pdf/2609.00951v1)

**作者:** Jiuwu Hao `[一作]` (University of Chinese Academy of Sciences), Pin Lv `[通讯]` (Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种通信高效且无需重新训练的多无人机协同感知框架CERF，利用感知结果生成虚拟模态Poture并融合到本地BEV特征，显著降低通信开销，支持未知异构代理无须额外训练即可参与协同。

**💡 创新点**

创新点包括：1）仅传输紧凑感知结果而非稠密特征，极大减少通信负担；2）构造基于检测框的虚拟模态Poture，通过映射到BEV网格补充局部特征；3）结合Kalman滤波器和运动预测模型实现延迟补偿；4）实现对未知异构代理的无缝集成。

**🔧 技术方法**

使用技术包括：虚拟模态Poture生成（检测框特征映射到BEV网格）、Kalman滤波追踪、常速运动预测、置信度感知的NMS、BEV特征与Poture的通道级拼接，以及3D检测、跟踪和轨迹预测等下游任务网络。

**📊 数据集**

实验数据集为UAV3D（3D检测、跟踪）和Air-Co-Pred（轨迹预测）。

**📈 对比分析**

与中间特征融合、Late-fusion、DiscoNet等方法对比，CERF在检测、跟踪、轨迹预测任务上性能可与SOTA相当，同时通信量降低约95%；在开放异构设置下，CERF可无重训练地加入未知代理，保持良好性能。

**⚠️ 局限性**

局限性：1）仅依赖感知结果作为信息源，可能在极端动态场景或多模态信息缺失时性能受限；2）Poture不包含位置信息，导致某些情况下精度下降；3）对帧率和延迟高度依赖Kalman/预测模型；4）在多传感器融合方面仍有提升空间。

---

## 539. LLM Inference on IMC-NoC Architecture with Balanced Dataflow and Fine-Grained Parallelism

**arXiv ID:** 2609.00857 | [PDF](https://arxiv.org/pdf/2609.00857v1)

**作者:** Yimin Wang `[一作]` (National University of Singapore), Xuanyao Fong `[通讯]` (National University of Singapore)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种将IMC、NMC和INC融合的可扩展硬件架构LEAP，并配套的分区、映射与调度软件框架，用于高效执行LLM推理，特别针对前填(prefill)和解码(decode)阶段的差异性需求实现了预填-解码分离(disaggregation)模式。

**💡 创新点**

创新点在于：①统一的计算-内存-通信一体化织物；②专门针对动态/静态矩阵操作的IMC、NMC、INC模块；③基于预填/解码差异的分离硬件划分与动态重构；④精细化的数据流模式与分区映射策略，显著降低了通信瓶颈。

**🔧 技术方法**

采用的关键技术包括：IMC交叉阵列、近内存计算(NMC)、网络内加速(INC)、Mesh NoC、动态预填-解码重构、启发式分区映射、批量调度与流水线重叠。

**📊 数据集**

使用了Llama系列模型（Llama 3.2‑1B、Llama 3‑8B、Llama 2‑13B）作为推理工作负载，并基于这些模型进行性能评估。

**📈 对比分析**

通过与NVIDIA A100、H100 GPU在相同推理工作负载下的吞吐量与能效进行对比，LEAP-A相较于A100实现了约2.55×吞吐提升、约71.94×能效提升；LEAP-D相较于H100实现了约1.52×吞吐提升、约24.91×能效提升。

**⚠️ 局限性**

局限性包括：对批量大小的支持有限（最多2个请求）；对模型规模的扩展仍受限于分区与通信成本；实现的硬件尺寸与功耗相对较高，且在极大模型或超大上下文长度时仍可能出现通信瓶颈。

---

## 540. Runtime-Independent Persistent Agents: Preserving Identity, Memory, and Code Across Models, Harnesses, and Servers

**arXiv ID:** 2609.00546 | [PDF](https://arxiv.org/pdf/2609.00546v1)

**作者:** Zhenyu Zhao `[一作]` (Independent Researcher), Roy Zhao `[通讯]` (University Of Washington)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一种运行时无关的持久化智能体架构，并实现了参考实现Enoch，用来在模型、主机、执行框架和交互表面更换时保持智能体的身份、记忆和代码不变。

**💡 创新点**

创新点包括：① 将智能体划分为持久子系统（身份、记忆、版本化代码）和可替换执行子系统（推理器、主机、容器）以及交互表面；② 定义了六个连续性不变量和完整的授权迁移协议；③ 通过 provider‑neutral 绑定实现运行时独立性；④ 在参考实现中验证了多轴替换（模型、主机、聊天服务）并提供了机械连续性检查。

**🔧 技术方法**

技术主要包括：版本化软件体、可插拔的 provider 合约、状态快照与恢复、事务式迁移（quiesce→checkpoint→validate→bind→rehydrate→resume）、idempotency 令牌与守护进程 epoch、以及在持久子系统中分离身份、记忆和代码的设计。

**📊 数据集**

本文未使用传统机器学习数据集；评估基于内部的 833 条核心单元测试和 92 条 provider/library 测试，以及对交互表面和宿主替换的功能性验证。

**📈 对比分析**

比较方法侧重于“机械连续性”——检查身份版本、记忆祖先、代码版本、任务 ID、权限 epoch 等是否保持一致；未提供行为连续性（性能）度量或跨模型/主机的对比实验，故无法评估迁移对任务延迟、准确率或资源消耗的影响。

**⚠️ 局限性**

限制：仅验证单轴替换；未进行完整的多轴（模型+主机+聊天）组合迁移实验；缺乏行为连续性评估与性能基准；授权迁移协议依赖于单一 authority，无法防止恶意副本；参考实现尚未包含第二个实时推理主机。

---

## 541. Task-Specific Prompt with Global Context for Multi-Task Graph Pre-Training

**arXiv ID:** 2609.00047 | [PDF](https://arxiv.org/pdf/2609.00047v1)

**作者:** Zhiyang Qiu `[一作]` (Guangzhou University), Wensheng Zhang `[通讯]` (Guangzhou University)

**通讯引用:** 9528 | [OpenAlex ID](https://openalex.org/A5100414787)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种双重先验注入的图预训练提示初始化方法 TPGC，利用任务先验和全局结构上下文提升多任务图预训练与下游少样本任务的性能。

**💡 创新点**

创新点在于首次将任务先验（TPIM）与结构先验（SPIM）同时注入提示，解决了随机提示与预训练目标及图结构不匹配的问题，显著提升提示的任务相关性和结构感知。

**🔧 技术方法**

采用图神经网络（GCN）作为编码器，结合多任务自监督预训练（DGI、GraphCL、LP、DSSL）和提示学习机制（软提示、层级提示投影），通过auxiliary graph 预训练和相似度聚合构造提示。

**📊 数据集**

使用六个主流数据集：节点分类的 Cora、Citeseer、PROTEINS、ENZYMES；图分类的 COX2、BZR，涵盖引用网络、蛋白质图和分子图。

**📈 对比分析**

与 GCN、GAT、DGI、GraphCL、GPPT、GraphPrompt、MultiGPrompt、ProNoG 等基线对比，TPGC 在 1-shot、5-shot 任务下多达 4 个数据集取得最好或第二好成绩，且在下游阶段参数量和运行时显著降低。

**⚠️ 局限性**

局限性：目前在异质图（heterophilous graphs）中的表现尚未充分验证，未来需设计更适用于不同结构特征的提示初始化策略。

---

## 542. Feedback-Assisted Trust Propagation over Document Relation Graphs for Retrieval-Augmented Generation

**arXiv ID:** 2609.00543 | [PDF](https://arxiv.org/pdf/2609.00543v1)

**作者:** Zhuoheng Li `[一作]` (Pennsylvania State University), Ying Chen `[通讯]` (Pennsylvania State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

利用稀疏人类反馈通过文档关系图传播来估计文档可信度，并以此改进检索与生成。

**💡 创新点**

创新在于将文档关系结构化为图，并通过二元反馈与两两一致性优化联合求解可信度，再用可信度进行检索重排序与提示。

**🔧 技术方法**

采用图优化（QP+PGD）、NLI模型构建边、可信度传播、可信度重排序与可信度提示等技术。

**📊 数据集**

在 MS MARCO、Natural Questions 与 TriviaQA 三大开放域问答数据集上进行实验。

**📈 对比分析**

与 VanillaRAG、InstructRAG、AstuteRAG、ReliabilityRAG、TrustRAG 等基线对比，EM 提升 0.03–0.11，FP@5 提升至 77–96%，表现显著优于所有基线。

**⚠️ 局限性**

仅支持二元反馈，实验使用合成矛盾文本，未验证对长篇或多跳推理的适用性，未来需研究分级反馈与自然冲突。

---

## 543. Investigating Assistant Bias in LLM User Simulators Using a Role Vector

**arXiv ID:** 2609.00608 | [PDF](https://arxiv.org/pdf/2609.00608v1)

**作者:** Daeheon Jeong `[一作]` (Korea Advanced Institute of Science and Technology), Juho Kim `[通讯]` (Korea Advanced Institute of Science and Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对LLM用户模拟器中的助理偏差进行表示层分析，提取“用户角色向量”，研究其对模拟真实性的因果影响，并在多轮对话中验证其效果。

**💡 创新点**

创新点：1）利用对比激活添加从同一对话的用户与助手视角提取用户角色向量；2）证明该向量可通过激活调节实现更真实用户行为；3）发现用户向量与助理偏差特征呈负相关，可作为模拟真实性的诊断信号。

**🔧 技术方法**

技术手段：对比激活添加（CAA）与激活调节、角色反思生成、层级激活提取、LLM-as-judge 评估与统计分析。

**📊 数据集**

使用数据集：LMSYS‑Chat‑1M、WildChat、SimulatorArena（数学辅导任务）、RealUserSim、GitHub 公开对话日志。

**📈 对比分析**

比较方法：在 SimulatorArena 与 RealUserSim 上采用 1–5 分相似度评分，未调节与调节的对话进行对比。结果显示，用户角色向量调节在写作风格上提升约 0.13 分，整体相似度从 2.61 提升至 2.73；但过强调节会导致行为过度夸张，影响真实性。

**⚠️ 局限性**

局限性：实验主要在 Qwen 3.5 9B 上验证，缺乏跨模型/规模验证；评估依赖 LLM-as-judge，未结合直接人类评估；实验场景局限于数学辅导与 WildChat；调节强度固定，未采用自适应或优化的调节策略。

---

## 544. When Modality Gap Reduction Fails: Prediction-Level Hubness in CLIP

**arXiv ID:** 2609.01103 | [PDF](https://arxiv.org/pdf/2609.01103v1)

**作者:** Shota Sato `[一作]` (Hitotsubashi University), Mamoru Komachi `[通讯]` (Hitotsubashi University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究在CLIP模型中减小图像-文本模态间差距（modal gap）对零样本分类准确率的影响，并揭示模态差距减小可能导致预测结果集中于少数类别（prediction-level hubness），从而导致准确率下降。

**💡 创新点**

创新点在于：①从决策边界角度阐释模态差距减小与准确率不一致的机制；②提出并量化预测级hubness（使用Gini系数），并证明其与准确率下降相关；③用线性修正的解析表达式证明类级得分偏差导致hubness，并通过偏差消除干预验证因果关系。

**🔧 技术方法**

技术包括：CLIP模型的线性几何修正（Linear correction）与学习型修正（CLIPRefine、AlignCLIP）；余弦相似度分类；Gini系数用于度量预测分布集中度；Spearman相关性与Pearson相关性用于统计分析；得分空间干预（bias subtraction）验证机制。

**📊 数据集**

实验数据集共10个：Caltech101、CIFAR‑10、CIFAR‑100、DTD、EuroSAT、FGVC‑Aircraft、Flowers102、Food‑101、ImageNet‑1K、Oxford‑IIIT‑Pet。

**📈 对比分析**

比较方法：对同一CLIP ViT‑B/32模型分别做线性修正（调节α）、CLIPRefine检查点和AlignCLIP；对每种方法计算准确率和预测Gini。结果显示：线性修正随α增大先提升后下降，准确率峰值对应预测Gini最低；过度修正导致Gini升高、准确率下降；CLIPRefine在保持或提升准确率的同时，Gini波动较小。

**⚠️ 局限性**

局限性包括：仅针对固定类别集合的零样本图像分类，结果不一定推广到检索、生成或多标签任务；评价指标Gini受标签分布不平衡影响；线性修正偏差干预仅为诊断工具，非实用改进；实验主要基于ViT‑B/32，未充分验证其他backbone；对学习型修正的因果机制仍缺乏解析证明。

---

## 545. Subgroup Accessibility in Group Order Logic

**arXiv ID:** 2609.00499 | [PDF](https://arxiv.org/pdf/2609.00499v1)

**作者:** Anatole Dahan `[一作]` `[通讯]` (University of Cambridge), Anatole Dahan (University of Cambridge)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究固定点逻辑及其扩展在可定义可访问子群生成集上的表达能力，并在此基础上实现对有界色类图的多项式时间规范化。

**💡 创新点**

提出可访问子群操作在一般逻辑下不可定义，但在存在可定义有序生成集的基群时可在 Group Order Logic (+) 中定义；若基群为阿贝尔群，则只需 Fixed‑Point Logic with Counting (FPC) 即可；进一步将此结论推广到具有多项式有界色类的图，从而在 FPC 中给出这些图的自同构群生成集。

**🔧 技术方法**

部分模拟 Schreier–Sims 算法、利用 + 操作计算群秩、以及在 FPC 中使用计数与固定点的技术；通过定义有序生成集、PSGS（部分强生成集）与饱和化过程实现对可访问子群的构造。

**📊 数据集**

本文主要为理论工作，未使用具体数据集；分析对象为所有有限结构和有界色类图的抽象实例。

**📈 对比分析**

方法通过理论证明展示可在多项式时间内构造生成集；相较传统需要依赖群论算法的做法，本文提供了逻辑层面的可定义性证明，性能上保持多项式时间复杂度。

**⚠️ 局限性**

限制在于需基群具备可定义的有序生成集或为阿贝尔群；对非阿贝尔基群的可访问子群及其自同构群仍未能在逻辑中定义，且无法实现对有界色类图的规范化。

---

## 546. EULER: Exploring Underused Links with Evidence-Checked Return for Multi-Agent Mathematical Discovery

**arXiv ID:** 2609.00032 | [PDF](https://arxiv.org/pdf/2609.00032v1)

**作者:** Ren Zhenzhuo `[一作]` `[通讯]`, Ren Zhenzhuo

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了一套多代理系统EULER，用于在不同数学社区之间自动寻找“桥接”以解决近年组合学的120个公开未决命题。

**💡 创新点**

创新点包括：①以桥接为核心搜索单元并设计六类结构性压力测试；②桥特定的预算分配与返回证据机制；③操作集交互效应的量化验证；④将正式化与经验评审并行，构建可追溯的任务与命题图。

**🔧 技术方法**

技术手段包括多代理架构（生成、检索、工具调用、批评、验证），Lean正式化、定量程序（枚举、SAT/SMT等），多大模型（Claude、GPT、DeepSeek）及其跨模型协作，桥映射与可返回模板，成本与经验评审的记录系统。

**📊 数据集**

数据集为从2024年两年内发表于《Journal of Combinatorial Theory, Series A》作者框架提取的120个公开命题，经过去重、可接受性筛选及污染控制后构成评估基准。

**📈 对比分析**

通过与直接搜索、随机/检索距离桥、无桥特定测试等多种对照进行对比。全系统在成本与直接搜索相当（1.12倍）下，已验证解从8/120提升至13/120（+4.2个百分点，95%区间跨0），错误率从9降至3/120；但非劣性阈值未达标。

**⚠️ 局限性**

局限性包括：任务偏向组合学，难以推广至几何、抽象代数等领域；距离判定依赖人工评判；检索受可用文献和工具库限制；污染筛查仍无法完全排除训练集泄漏；系统部署复杂且成本仍偏高。

---

## 547. ReBridge-Flow: Re-Coupling Posterior Bridges in Flow Matching for Image Restoration

**arXiv ID:** 2609.00811 | [PDF](https://arxiv.org/pdf/2609.00811v1)

**作者:** Jiaqi Zhang `[一作]`, Mingkai Zheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种名为ReBridge-Flow的后验桥重耦合方法，用于图像恢复任务中解决Flow Matching模型局部测量修正导致的源-干净端点不匹配问题。

**💡 创新点**

创新点在于通过清端点锚定与源端点同步重耦合，形成测量一致且局部桥兼容的端点对，并利用Posterior Bridge Defect统一测量误差、流先验偏差与桥残差，得到闭式更新。

**🔧 技术方法**

核心技术包括Flow Matching模型、闭式端点重耦合公式、后验桥缺陷（Posterior Bridge Defect）以及基于线性降噪算子的清端点锚定。

**📊 数据集**

在多个自然图像数据集（CelebA、AFHQ‑Cat、COCO）与医学图像数据集（IXI‑Brain、PMUB、X‑Ray Hand）上进行实验。

**📈 对比分析**

与OT‑ODE、Flow‑Priors、D‑Flow、PnP‑Flow、Restora‑Flow、Flower等六种先进方法比较，ReBridge‑Flow在PSNR/SSIM/LPIPS指标上均优于或匹配最优方法，并且在推理时间和显存占用上具有更好的效率。

**⚠️ 局限性**

局限性包括仅适用于已知线性降噪算子，依赖预训练流模型的质量，且对未知或非线性降噪操作、复杂噪声以及与训练分布偏离的图像仍存在性能退化。

---

## 548. When Prediction Error Is Not Enough: Evaluating Nuisance-Function Prediction for Causal Estimation

**arXiv ID:** 2609.00071 | [PDF](https://arxiv.org/pdf/2609.00071v1)

**作者:** Cong Cao `[一作]` (Yale University), Cong Cao `[通讯]` (Yale University)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5100784656)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在部分线性模型中通过蒙特卡洛模拟，研究了受试者变量预测误差与因果估计性能的关系，比较了OLS、GAM、XGBoost以及DML-XGBoost等方法。

**💡 创新点**

创新点在于揭示受试者函数的预测误差并不能可靠预测因果估计的准确性，并证明简单的联合误差指标与因果偏差关联不强，强调点估计与置信区间覆盖率之间的潜在冲突。

**🔧 技术方法**

采用了部分线性模型、双机器学习（DML）与正交得分、交叉拟合、XGBoost、GAM、OLS以及聚类稳健方差估计等技术。

**📊 数据集**

使用的是完全仿真的数据集（1000个观测，包含独立和聚类两种情形），没有使用任何真实数据。

**📈 对比分析**

通过比较预测误差、偏差、RMSE和95%置信区间覆盖率等指标发现，XGBoost在非oracle方法中RMSE最低，但置信区间覆盖率较低；DML-XGBoost在覆盖率上更好；不同方法的预测误差排名与因果性能排名不一致。

**⚠️ 局限性**

局限性包括：仅考虑常数处理效应和相对简单的模型；聚类分析仅采用单一聚类大小和方差设定；DML-XGBoost同时混合了学习器和估计程序，难以分离两者影响；联合误差指标过于简单，未能捕捉正交得分的完整结构；结果可能不适用于处理效应异质性或更复杂的相关结构。

---

## 549. CQF-HMR: Continuous Quaternion Flows for Probabilistic 3D Human Mesh Recovery from a Single Image

**arXiv ID:** 2609.00995 | [PDF](https://arxiv.org/pdf/2609.00995v1)

**作者:** Cuong Le `[一作]` (Linköping University), Mårten Wadenbäck `[通讯]` (Linköping University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

使用连续四元数流匹配（CQF-HMR）对单帧2D热图进行条件化，生成多样化、可行的3D人体网格假设。

**💡 创新点**

创新点在于：①采用单位四元数流并用球面线性插值（slerp）构造最优传输路径，避免了传统欧拉角/轴角在积分过程中的奇异性与不连续性；②将流匹配框架与SMPL关节约束结合，使用Hamilton乘积保证四元数始终保持单位；③在初始姿态采样上引入VPoser先验，显著提升生成姿态的可行性和多样性。

**🔧 技术方法**

技术方法包括：连续正则化流（CNF）与流匹配（flow matching）；Hamilton乘积积分；RK2数值求解；基于HRNet的2D热图提取；GCN编码2D条件；VPoser作为先验采样；SMPL模型的姿态与形状回归。

**📊 数据集**

使用了Human3.6M（常规与含模糊分割）、3DPW以及EMDB数据集进行训练与评估。

**📈 对比分析**

与相关方法（如FMPose、DiffPose、SPIN、HuManiFlow等）比较，CQF-HMR在Human3.6M的MPJPE/PA-MPJPE上实现了新的最优成绩，在3DPW/EMDB上保持与最新基线相近或略优的性能，并在多假设生成的多样性（Diversity）上表现更好。

**⚠️ 局限性**

主要限制是对HRNet 2D检测的依赖，导致在尺度模糊、远近尺度差异大或严重遮挡场景下仍可能产生误差；缺乏对环境、接触和体场交互的建模，限制了在复杂场景中的推断精度。

---

## 550. From Tool Use to Technological Agency: LoopCAT as a Local-First, Open-Source Tool for Translation Technology Education

**arXiv ID:** 2609.00344 | [PDF](https://arxiv.org/pdf/2609.00344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 551. Soft-Argmax for the Projective Plane via the Veronese Embedding

**arXiv ID:** 2609.00521 | [PDF](https://arxiv.org/pdf/2609.00521v1)

**作者:** Benjamin El-Zein `[一作]` (Siemens Healthineers AG), Sebastian Stober `[通讯]` (Otto-von-Guericke-University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于Veronese嵌入的可微软argmax读取器，用于单线检测；

**💡 创新点**

通过将无向线的非线性双覆盖空间映射到线性空间，消除soft‑argmax的空间不匹配问题，并在此空间上进行平均和投影恢复；

**🔧 技术方法**

使用Hough变换、Veronese映射、软argmax、线性投影以及CNN特征提取网络；

**📊 数据集**

在合成的256×256单线图像上进行实验，噪声水平σ在[0,0.8]范围内随机生成；

**📈 对比分析**

与传统soft‑argmax、MLP回归、RHT等方法对比，使用EA指标评估；vsmax在seam处的EA从0.69提升至0.92，整体EA提升显著，且对噪声和线分布外推更稳健；

**⚠️ 局限性**

仅在单线合成数据上验证，未测试真实图像和多线场景。

---

## 552. MADS: A Multiview Acoustic Descriptor Set Beyond Standard Spectral Summaries

**arXiv ID:** 2609.00792 | [PDF](https://arxiv.org/pdf/2609.00792v1)

**作者:** Utsab Ghosh `[一作]` (ABV-Indian Institute of Information Technology and Management), Roshni Chakraborty `[通讯]` (ABV-Indian Institute of Information Technology and Management)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了 MADS（Multi‑View Acoustic Descriptor Set），一种 19 维的物理信息驱动的音频特征集，用来描述音频信号的机械、光谱、时间和随机特性。

**💡 创新点**

创新点在于将机械动力学、光谱锚点、时间动态和随机性等多视角物理特征统一为一套紧凑可解释的描述子，并证明其对环境声音分类的竞争力。

**🔧 技术方法**

技术实现包括：从波形直接计算速度、加速度、能量等机械量；使用频谱质心、峰值频率等光谱锚点；提取攻击梯度、阻尼系数等时间动态；利用谱平坦度、峰值性、零交叉率变异等统计量捕捉随机性。

**📊 数据集**

使用的公开数据集为 ESC‑10、ESC‑50（环境声音分类）和 MSoS（5 类广义声音分类）。

**📈 对比分析**

与两个传统手工基线（26 D MFCC、38 D 光谱摘要）以及多种经典机器学习模型（k‑NN、Logistic、SVM、RF、XGBoost、集成等）进行对比，MADS 在 ESC‑10 上取得 81.00 % 的集成准确率，ESC‑50 上 52.78 %，MSoS 上 67.48 %，均优于 26 D 基线且与 38 D 基线竞争。

**⚠️ 局限性**

局限性包括：仅为 clip‑level 统计特征，未对时间序列进行细粒度建模；目前仅在经典 ML 框架下评估，未探索深度学习的潜在融合；以及对极端或非标准声音的适用性尚未充分验证。

---

## 553. ARISE-RL: Agentic Rubric-Grounded Iterative Self-Evolution with Reinforcement Learning

**arXiv ID:** 2609.01058 | [PDF](https://arxiv.org/pdf/2609.01058v1)

**作者:** Fanrui Zhang `[一作]` (Alibaba Ath Token Foundry), Zheng-Jun Zha `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出ARISE-RL，一种完整闭环自演进强化学习框架，包含任务/评判生成器与解题者，并结合Reward‑Gated Self‑Evolution Distillation（RG‑SED）和专家校准的ECR‑Bench基准；

**💡 创新点**

核心创新在于：基于细粒度、工具锚定评判的共进化循环、难度塑造奖励机制、基于奖励门控的自我蒸馏，以及面向开放式任务的专家评判基准；

**🔧 技术方法**

采用RL与多步推理、工具调用相结合的策略学习，利用细粒度评判奖励与门控自蒸馏技术（memory‑augmented teacher + token‑level reverse KL），并实现任务生成器与解题者的协同进化；

**📊 数据集**

使用了ECR‑Bench（含单工具深度研究与多工具旅行规划）、ResearchRubrics、VitaBench等开放式评估数据集；

**📈 对比分析**

与闭源LLM、开源基准、Dr. Zero/Absolute Zero以及OPCD、GKD等对比实验显示，ARISE‑RL在9B级模型上在所有十一项基准上均达到或超过最佳水平，尤其在多工具交互任务上显著提升；

**⚠️ 局限性**

受计算资源限制，实验仅在8/9B开源模型上验证，尚未检验更大基底模型的可扩展性，且自演进循环和门控蒸馏对计算开销有一定影响。

---

## 554. What Is a System? An Interaction-Based Account of Structure-Behavior Coalescence in General Systems Theory

**arXiv ID:** 2609.00043 | [PDF](https://arxiv.org/pdf/2609.00043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 555. SwapRec: Warming Up Cold Items Through Training-Time Swaps

**arXiv ID:** 2609.00913 | [PDF](https://arxiv.org/pdf/2609.00913v1)

**作者:** Marta Moscati `[一作]` (Albatross AI), Matteo Ruffini `[通讯]` (Albatross AI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在训练时进行冷启动项目替换（SwapRec）的方法，使基于ID的顺序推荐模型在面对冷项目交互时保持鲁棒性并提升实时个性化。

**💡 创新点**

创新点在于将工业界常用的推理时冷项目替换策略同样嵌入训练过程，简化实现且不改变模型架构，显著降低冷启动对性能的负面影响，并提升冷项目在推荐列表中的曝光率。

**🔧 技术方法**

使用 Transformer‑基础的顺序推荐模型（如SASRec、BERT4Rec）与邻居映射（NN map）实现的随机替换增广；通过 p_swap 与 M_swap 超参数控制替换比例与次数；评估指标包括 HR@10、冷项目比例、覆盖率等。

**📊 数据集**

实验数据集包括：Music4All‑Onion（音乐+音频嵌入）、Amazon All_Beauty（商品标题+描述文本嵌入）和 MovieLens‑20M（电影剧情文本嵌入），覆盖音乐、购物与电影三大领域。

**📈 对比分析**

通过与无替换训练的原始顺序模型、VAE、I‑Neighbor、BPR 等基线进行对比，SwapRec 在 HR@10 上普遍优于原始模型，尤其在冷项目序列上显著降低性能退化，同时冷项目推荐比例和目录覆盖率显著提升。

**⚠️ 局限性**

局限性包括：仅在推理时替换序列最后一项；仅使用最相似的热项目作为替换；未探索多次或不同位置替换、对抗性或对比损失的潜在改进；对嵌入空间结构的影响未做深入研究。

---

## 556. CacheBridge: Efficient Cross-Model KV Cache Transfer

**arXiv ID:** 2609.00891 | [PDF](https://arxiv.org/pdf/2609.00891v1)

**作者:** Xingyu Qu `[一作]` (Westlake University), Tao Lin `[通讯]` (Westlake University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 CacheBridge，一种基于闭式仿射映射的跨模型 KV 缓存转移方法，能够在不进行目标模型预填充的情况下实现高质量的上下文转移并显著降低转移成本。

**💡 创新点**

创新点包括：① 通过限制每个目标头仅使用一个源头头的结构化映射（head fan‑in 限制）降低映射复杂度；② 用注意力敏感加权重新定义校准目标，使得回归误差更符合接收模型的注意力计算；③ 设计了融合 GPU 内核实现高效的统计量构造，避免了散布式收集和存储开销。

**🔧 技术方法**

使用了岭回归（线性闭式仿射）、RoPE 逆/正向处理、注意力对齐加权、GPU fused kernel、头索引映射、结构化支持等技术。

**📊 数据集**

实验使用 FineWeb‑Edu（500/500 序列）进行校准，评估集包含 HellaSwag、ARC‑Challenge、WinoGrande、MMLU 等多任务数据集，以及 1,024 token 长度的前缀。

**📈 对比分析**

与原始全头方法对比，在 Ministral 3 3B→14B 和 8B→14B 两个失效方向恢复 20‑30 分准确率，Qwen3 14B→32B 维持 99.83% mean retention；mapper 存储从 4.296 GB 减至 0.538 GB，应用延迟提升 1.6‑3.0 倍；构造时间从 92.63 s 降至 8.63 s，速度提升 10.7 倍。

**⚠️ 局限性**

仅在同一模型族内部验证；未探索跨族模型、不同注意力机制（稀疏、滑窗等）或多轮生成后累计误差；未评估多次交互后生成质量。

---

## 557. Coding for Multiple Reverse-Complement and Palindromic Duplications

**arXiv ID:** 2609.00779 | [PDF](https://arxiv.org/pdf/2609.00779v1)

**作者:** Aryeh Lev Zabokritskiy `[一作]` `[通讯]` (Tel-Hai University of Kiryat Shmona), Aryeh Lev Zabokritskiy (Tel-Hai University of Kiryat Shmona)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

研究了校正固定长度反转互补和回文复制错误的 q‑ary 代码，并给出了有限误差下的容量上界与下界。

**💡 创新点**

提出了一个通用的终点多重性逆推计数方法；在偶长度复制下构造了坐标级交替取反的双射，使两种错误通道完全等价；推导出两次错误球的最大大小，并用贪心图着色给出冗余 4log_q n+O(1) 的存在性结论。

**🔧 技术方法**

使用了组合编码理论（球形包络、端点计数）、图着色理论、符号级双射与逆推历史计数，以及二项式尾部与Hoeffding不等式的概率分析。

**📊 数据集**

本文仅涉及理论分析，无使用外部数据集；所有结论基于符号集大小 q 与复制长度 k 的参数。

**📈 对比分析**

与已知的插入突变（burst）编码和单误差卷积编码相比，单误差冗余达到 log_q n+O(1) 的最优阶，二误差下得到 2log_2 n–O(1) 的下界与 4log_2 n+O(1) 的存在上界，表现出两倍系数的空隙。

**⚠️ 局限性**

主要局限在于：二误差的冗余系数仍存在 2 与 4 的差距；缺乏高效（无非统一预备信息）构造方案，无法实现接近下界的 2log_2 n+O(1) 冗余。

---

## 558. Beyond Blind Compliance: Benchmarking Task Verification in OCR Reasoning

**arXiv ID:** 2609.00232 | [PDF](https://arxiv.org/pdf/2609.00232v1)

**作者:** Yue Zhou `[一作]` (Jilin University), Yi Chang `[通讯]` (Jilin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了OCR任务验证范式，并构建了1800条人审验的VeriOCRBench数据集。

**💡 创新点**

通过将任务拆解为图像前提、文本前提与问题，并设计8类陷阱与四维度验证（视觉、语境、事实、逻辑），实现了对OCR推理系统可靠性的细粒度评估。

**🔧 技术方法**

利用视觉原子事实（VAF）抽取、GPT-4o等大型语言模型生成陷阱任务，结合人类审核与GPT-4o评判器进行多维度评分，评估多模态LLM的验证、诊断与拒绝行为。

**📊 数据集**

样本来自8个OCR/文本视觉基准（OCRBench v2、OCR-Reasoning、OmniDocBench、MMTab、ChartX、DocVQA、M6Doc、GoodsAD），覆盖产品包装、UI、新闻、公共展示、科学、教育、行政与金融等8个领域。

**📈 对比分析**

在15个多模态LLM和2个OCR+LLM基线上进行实验，评估任务验证率、诊断准确率和过度拒绝率。结果显示主动提示下普遍盲从，辅助提示可显著提升验证率但伴随高过度拒绝；顶尖模型如GPT‑5、Claude‑4.5‑Sonnet、Qwen3.5‑397B‑A17B在验证率与诊断上表现最佳。

**⚠️ 局限性**

仅覆盖单轮OCR推理与八种陷阱类型，未考虑交互式、工具增强或动态文档等场景；评估依赖语义判定，可能缺少细粒度人工核验。

---

## 559. Dotting the Eye: An Intent-Driven Image Retouching Agent for Visual Focus Enhancement

**arXiv ID:** 2609.01148 | [PDF](https://arxiv.org/pdf/2609.01148v1)

**作者:** Chujie Qin `[一作]` (Nankai University), Chongyi Li `[通讯]` (Nankai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于多模态大语言模型和扩散模型的意图驱动图像修饰代理 EyeControl，实现弱用户意图（点击或涂画）下的视觉焦点增强。

**💡 创新点**

创新点包括：① 将弱空间意图与目标编辑区域及色调调整操作显式关联；② 通过伪意图引导注意力对齐（PAA）让扩散模型内部注意力与意图对齐；③ 引入操作一致性损失保证全局与局部调整协调；④ 构建了 ControlArt-Bench 基准。

**🔧 技术方法**

使用技术包括多模态大语言模型规划器、Flux-1.0-Kontext 迁移的扩散 Transformer 执行器、LoRA 微调、伪意图注意力对齐、操作一致性损失、Saliency 差分、数据生成流水线。

**📊 数据集**

训练集由 PPR10K、Lightroom Community、专业修图作品构成，评估集 ControlArt-Bench（200 组含意图掩码的真实用户修图样本），并与 MIT-FiveK、MMArt-Bench 等公开数据集对比。

**📈 对比分析**

与多种开源与商业模型（JarvisArt/Evo、PerTouch、UniWorld-v2、Step1X-Edit、FLUX.1-Kontext、Qwen-Image-Edit-2511、GPT-Image-1.5、Nano-Banana-2）在 PSNR、SSIM、Focus Alignment、PQ 等指标上对比，EyeControl 在 Focus Alignment、PQ、整体分数 O 以及视觉焦点一致性上遥遥领先，整体性能接近甚至超过 Nano-Banana-2。

**⚠️ 局限性**

局限性包括：对极端模糊或多重关注点的意图仍可能出现分配不均；需要较大算力和复杂的数据生成流水线；缺乏针对不同场景的专业调色细化；模型在多轮交互中仍可能出现累积误差。

---

## 560. Does task decomposition improve automatic NLG evaluation?

**arXiv ID:** 2609.01139 | [PDF](https://arxiv.org/pdf/2609.01139v1)

**作者:** Sebastian Steindl `[一作]` (Amazon), Diego Marcheggiani `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较LLM-as-a-judge（LLMaJ）评估方法，探究分解与直接预测的效果

**💡 创新点**

发现分解不提升性能，主因是使用人类标签；直接预测可匹敌或超过人类

**🔧 技术方法**

采用Claude‑4、Qwen3‑32B等大模型，构建HD‑Eval、CheckEval、Direct Prediction等方法，并使用回归器

**📊 数据集**

使用SummEval、TopicalChat、Seahorse等NLG评估数据集

**📈 对比分析**

通过Spearman ρ、AP、WR等指标比较，直接预测在多数指标上与分解方法相当或更优，分解提升归因于人类标签

**⚠️ 局限性**

仅限三类数据集，未考察复杂推理任务；缺乏对人类标签细化的支持；检查结果受LLM版本和预训练数据影响

---

## 561. Scaled Idempotence in Transformer Attention: Paired OV Geometry and Shared-Value Algebras

**arXiv ID:** 2609.01129 | [PDF](https://arxiv.org/pdf/2609.01129v1)

**作者:** Jiming Feng `[一作]` (Beijing University of Technology), Junliang Li `[通讯]` (Beijing University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer注意力中的OV算子，发现大模型中存在稀疏的头满足 T^2≈αT，并通过几何分解揭示其根本机制。

**💡 创新点**

提出 KDK 分解将闭合性归因于读写因子配对的有向返回几何；证明在共享 value 的情况下闭合性扩展为固定增益的跨头算子代数；区分几何容量与训练实现稀疏性。

**🔧 技术方法**

采用低秩矩阵分析、主角坐标分解、主角角度、正交重排、取向打乱实验、可行容量构造、矩阵分解与扰动分析等技术进行结构与实验验证。

**📊 数据集**

使用多大模型端点（2.8B–235B）数据集：Pythia、Qwen、Mistral、OLMo、TinyLlama 等；WikiText 语料用于自然激活检验。

**📈 对比分析**

通过闭合度指标 T≥0.9 评估，并与错误配对（wrong‑V）控制、取向打乱、可行容量等对照。结果显示 4–8% 头满足闭合，错误控制几乎为零；取向打乱后闭合降至 10⁻⁴，证明取向是决定性因素。

**⚠️ 局限性**

未验证前向推理是否利用该闭合；仅给出结构性分析；功能影响未直接评估；共享值代数仅适用于 GQA，MHA 的跨头关系仍待探索；未解析优化过程中的梯度作用与数据驱动机制。

---

## 562. Sentinel-Based Failover for QKD-Augmented IPsec Tunnels

**arXiv ID:** 2609.01121 | [PDF](https://arxiv.org/pdf/2609.01121v1)

**作者:** Juan Carlos Hernandez-Hernandez `[一作]` (University of Luxembourg), Symeon Chatzinotas `[通讯]` (University of Luxembourg)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了一个基于 Sentinel 的自适应失效降级协议，能够在 QKD 关键路径失效时仍保持 IPsec 隧道正常工作，随后自动恢复完整的量子安全密钥。

**💡 创新点**

创新点在于利用 sentinel 值在 IKEv2 的 QKD 交换中完成失效检测与降级判断，避免额外的往返延迟和手动干预，同时保证安全性不低于后量子安全基线。

**🔧 技术方法**

采用 strongSwan 插件集成 ETSI GS QKD 014 接口、X25519、ML‑KEM（NIST 标准）以及 RSA 认证，全部通过 RFC 9370 的多键交换机制实现。

**📊 数据集**

在实验中使用了 33 km 已部署光纤的市政级 QKD 链路，并对五种 IKEv2 配置（仅 X25519、X25519+ML‑KEM、X25519+ML‑KEM+QKD 等）进行性能与失效恢复测试。

**📈 对比分析**

实验比较显示：完整认证时间从基线 61 ms 提升到 103 ms（+42 ms），QKD 轮询仅增加约 7 ms；子隧道重新密钥始终维持 ~5 ms；在 KME 故障时隧道可无中断继续，且在下一次重密钥时自动恢复到全混合模式。

**⚠️ 局限性**

局限性包括：仅在市政级低延迟环境测试，较高延迟网络的额外轮询开销尚未评估；使用 RSA 认证，未探讨后量子认证或预共享密钥模式；对 QKD 链路的吞吐量与密钥速率影响未作深入分析。

---

## 563. Neural Symbollic Regression Using Deep Learning and Sparse Modelling

**arXiv ID:** 2609.01102 | [PDF](https://arxiv.org/pdf/2609.01102v1)

**作者:** Ravi Kumar U `[一作]` (Indian Institute of Space Science and Technology), Sumitra S `[通讯]` (Indian Institute of Space Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过先用神经网络平滑逼近数据，再用 LASSO 提取稀疏符号表达式，提出一种端到端的神经符号回归框架。

**💡 创新点**

将神经网络作为功能预处理器与稀疏回归分离，构建交互感知特征库，并集成 Ray Tune+ASHA 分布式超参搜索，显著提升噪声鲁棒性与可解释性。

**🔧 技术方法**

使用多层感知机、交互式非线性特征库、LASSO 稀疏回归、Ray Tune+ASHA 分布式超参优化、GPU 加速与 SymPy 符号重构技术。

**📊 数据集**

采用 Nguyen 系列（Nguyen‑1 至 Nguyen‑7）基准函数，每个基准生成 1000 个样本，加入不同水平的高斯噪声，并在 70/15/15 的训练/验证/测试划分下进行实验。

**📈 对比分析**

与 SINDy、PySR 等传统基线对比，NSR 在 RMSE、符号准确率、噪声鲁棒性和 OOD 泛化等指标上均优于传统方法，尤其在高噪声与外域测试时表现更为显著。

**⚠️ 局限性**

主要限制在于特征库规模随维度呈指数增长导致内存和计算开销大，对高度不连续或分段函数适应性差，LASSO 系数偏置可能略微影响符号精度，以及神经网络训练耗时相对较长。

---

## 564. IT-TextFusion: Iterative Text-Image Interaction with Text-Guided Residual Refinement for Degradation-Aware Image Fusion

**arXiv ID:** 2609.01092 | [PDF](https://arxiv.org/pdf/2609.01092v1)

**作者:** Siyang Liu `[一作]` (Technical University of Munich), Mengze Gao `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于迭代文本-图像交互的红外可见图像融合框架IT‑TextFusion，利用文本提示指导多尺度特征融合与残差细化。

**💡 创新点**

创新点包括：降解感知文本工程、跨门融合模块、逐层文本条件调制的ITIM以及文本引导的残差细化模块TG‑RRM，实现了全局文本在多级解码与残差校正中的连续注入。

**🔧 技术方法**

采用CLIP文本编码器、Transformer编码器‑解码器架构、跨门注意机制、跨层文本条件调制（affine映射）以及U‑Net式残差细化网络。

**📊 数据集**

使用标准红外可见融合数据集MSRS、LLVIP、MFNet、RoadScene以及自构建的多降解EMS数据集（9类降解）。

**📈 对比分析**

与Text‑IF及多种传统融合方法对比，采用信息熵、标准差、空间频率、相互信息、结构相似度等指标，IT‑TextFusion在多数融合和无参考感知指标上实现了提升，尤其在信息保留与结构一致性上表现突出。

**⚠️ 局限性**

局限性包括不同指标间的互斥权衡，对低对比度红外场景的提升有限，文本-视觉交互深度仍受限，且在极端降解条件下仍可能出现结构或色彩失真。

---

## 565. LLMPEDIA: Browsing, Verifying, and Comparing the Parametric Encyclopedic Knowledge of LLMs

**arXiv ID:** 2609.01182 | [PDF](https://arxiv.org/pdf/2609.01182v1)

**作者:** Muhammed Saeed `[一作]` (TU Dresden), Simon Razniewski `[通讯]` (TU Dresden)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了一个约130万篇文章的百科全书，全部由三种大型语言模型（GPT‑5‑mini、DeepSeek‑V3.2、Llama‑3.3‑70B）的参数记忆生成，随后对随机抽样的主张进行事实性审核。

**💡 创新点**

创新点在于：①将LLM的参数记忆转化为可浏览、可查询的知识库，突破传统多选题基准的可用性偏差；②引入“支持/驳斥/无证据”三种标签，量化模型在深层知识上的“沉默”问题；③提供跨模型、跨人格、跨主题的可视化对比，揭示知识差异和框架效应。

**🔧 技术方法**

技术包括：温度为0的单次生成、三阶段链接净化（规范化→LLM百科过滤→嵌入去重）、层次化生成树、分层审核（Wiki+精心挑选的Web域）以及LLM判定器（FactScore验证）。

**📊 数据集**

数据集：约1.3M条目（GPT‑5‑mini约1M，DeepSeek和Llama各约12万），包含3个主题集中运行（古巴比伦、美国民权、荷兰殖民），审核样本2,010个主体、20,092条主张，证据来源为维基百科及133个质量评分Web域。

**📈 对比分析**

对比方法：在随机样本中测得GPT‑5‑mini的真率68.4%（支持68.4%，驳斥1.2%，无证据30.5%）；对比不同模型在相同主题下的条目、结构与事实率，发现GPT‑5‑mini在深层知识上优势最大，Open‑weight模型在覆盖度和真率上落后。跨人格实验通过词汇框架计数评估语调差异，结果显示框架改变但事实准确率基本不变。

**⚠️ 局限性**

局限性包括：单次生成未覆盖全部参数记忆；审核样本有限，未覆盖约90%文章；依赖LLM判定器，误差与“无证据”不等同；模型规模不一导致深层分析仅适用于GPT‑5‑mini；可能出现幻觉实体；实体解析不完备；评估仅关注原子主张，未涉及连贯性、完整性或偏见。

---

## 566. A SoK for SoCs: Reading the TI Leaves on AI for Cyber Threat Intelligence Generation and Sharing

**arXiv ID:** 2609.01174 | [PDF](https://arxiv.org/pdf/2609.01174v1)

**作者:** Saastha Vasan `[一作]` (University of California Santa Barbara), Giovanni Vigna `[通讯]` (University of California Santa Barbara)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统化了网络威胁情报（CTI）生成与共享的完整生命周期，并通过实证评估了大型语言模型（LLM）在四个关键步骤（情报提取、归一化与丰富、编码/去标识化、分发）中的可行性与局限。

**💡 创新点**

创新点包括：①提出以“情报提取→归一化与丰富→编码→分发”四步拆解的CTI生成共享框架；②基于实务问卷构建首个面向原始攻击证据到共享情报的端到端标注数据集；③首次将LLM与传统规则基准对比，揭示其在提取、技术映射、去标识化与STIX生成中的优势与不足；④从中提炼出三条未来研究方向（工作流解构与自校验、红色化政策制定、跨步骤完整基准）。

**🔧 技术方法**

使用技术包括：系统性文献回顾、实务问卷调查、提示工程（Prompt Engineering）、LLM推理（OpenAI GPT‑4/ChatGPT、Anthropic Claude、Meta LLaMA‑2）、规则基准（宽松/严格规则、Sigma检测规则）、STIX 2.1 语义验证（reference implementation）以及评估脚本与数据处理管道。

**📊 数据集**

使用的数据集有：
• 约 400 条沙箱报告（来自安全厂商，包含恶意样本执行日志）
• 5 场真实攻击重放（BlackSuit、Confluence LockBit、Egg‑Cellent Resume、Lynx、Nitrogen），对应的网络流量与攻击步骤已记录
• 5 篇被注入受害者信息的 DFIR 文章，用于去标识化实验
• 生成的情报发现报告（软件、技术、指标、步骤），用于 STIX 分发实验
这些数据均由作者自行生成并标注，保证了从原始证据到共享情报的完整链路。

**📈 对比分析**

比较方法：
• 情报提取：与宽松/严格规则基准对比，评估精准率（Precision）与召回率（Recall）
• 归一化与丰富：与 Sigma 静态规则对比，评估 F1（包括子技术匹配）
• 去标识化：根据两种提示（隐私优先 vs 实用优先）分别评估覆盖率与误删率
• 分发：利用 STIX reference library 检查文件语法合法性，并计数正确/错误的关系链
性能结果：LLM 在提取上精准率高于规则，但召回率低；在技术映射上 F1 接近 0.72（远高于 Sigma 0.16）；去标识化能几乎完全满足给定目标但存在过度/不足平衡；STIX 生成语法无误，但模型会额外断言不被发现的链接。

**⚠️ 局限性**

局限性：
1. 需要人工审核，模型在召回、证据关联与策略判断方面仍有限；
2. 红色化与去标识化缺乏统一政策，导致过度或不足的去标识化；
3. 评估仅覆盖四个步骤，未考察完整的 CTI 生命周期；
4. 数据集规模有限，且不包含公开的标签社区共享数据；
5. 仅测试了三款 LLM，缺乏对模型更新迭代的长期跟踪；
6. 对成本与隐私的评估仅在实验条件下进行，实际部署需进一步验证。

---

## 567. Different Changes Require Different Reasoning: Change-Type-Specialized Experts for Robust Change Captioning

**arXiv ID:** 2609.01136 | [PDF](https://arxiv.org/pdf/2609.01136v1)

**作者:** Jiyoung Park `[一作]` (Kyung Hee University), Jung Uk Kim `[通讯]` (Kyung Hee University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种针对图像对变化描述的模型 MEDIC，利用两阶段路由和类型专门专家来生成更准确、类型一致的变化说明。

**💡 创新点**

创新点包括：1) 将变化类型显式建模为专家门控的两阶段路由；2) 每个专家使用键值记忆网络实现输入自适应检索；3) 引入专家一致性、解耦损失以及路由损失，促使专家对不同变化类型学习分离且一致的表征。

**🔧 技术方法**

主要技术手段：Mixture of Experts（MOE）架构、键值记忆网络、Transformer 解码器、ResNet-101 特征提取、交叉熵路由损失、专家一致性损失、专家解耦损失。

**📊 数据集**

使用四大公开数据集：CLEVR-DC、CLEVR-Change、Spot-the-Diff、Image Editing Request，并在 BLIP2IDC 上验证与 LVLM 的兼容性。

**📈 对比分析**

与 SCORER、SMART、DIRL、MCCFormers、VARD-Trans、BLIP2IDC 等多种最新基线进行对比，单变与多变设置下均在 BLEU、METEOR、ROUGE、CIDEr、SPICE 等指标上显著提升，特别是在多类型、噪声干扰场景中表现更稳健。

**⚠️ 局限性**

局限性：1) 需要手工或自动生成的变化类型标签，标签噪声会影响路由与专家训练；2) 对细粒度或多重重叠变化的识别仍有难度；3) 在完全无标签的自监督场景下仍需进一步研究。

---

## 568. When Does Online Adaptation Pay on the Edge? A Leakage-Free Evaluation of Warmup, Learning-Rate Selection, and Resource Trade-offs for Time-Series Forecasting

**arXiv ID:** 2609.01126 | [PDF](https://arxiv.org/pdf/2609.01126v1)

**作者:** Takumi Fujimoto `[一作]` (Keio University), Hiroaki Nishi `[通讯]` (Keio University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过泄漏自由的流式评估协议，系统研究了在线适配在边缘时间序列预测中的收益与资源代价，重点考察了基线预热预算、学习率选择以及不同适配策略（全模型、头部、校准）在多种数据集上的表现。

**💡 创新点**

创新点在于：①揭示基线预热预算对适配收益的双向非单调影响；②发现共享默认学习率会偏向某一优化器；③提出仅使用预漂移验证片段进行预热和学习率挑选的“验证-仅”方案；④在六个公开多变量流、包括大规模智能计量数据集上，量化适配状态内存、更新延迟与准确率的三维前沿，并指出多参数高效子模型在内存上无支配关系。

**🔧 技术方法**

使用的技术包括：无泄漏的非重叠窗口评估协议；对DLinear和PatchTST两种基线进行在线SGD+动量和Adam适配；针对预热预算与学习率进行网格搜索并在预漂移验证片段上早停/重演；测量A100 GPU上每次更新的内存占用和时间，推算边缘设备的延迟与能耗；利用多种更新调度（每步、周期性、漂移触发）对比性能。

**📊 数据集**

使用的数据集为：ETT（ETTh1、ETTh2、ETTm1、ETTm2）、UCI Appliances、包含传感器与能耗的多变量建筑数据（Appliances）、BDG2（含15/240/280条计量子集），全部为公共时序数据，覆盖不同漂移模式。

**📈 对比分析**

比较方法：在相同预热预算和学习率（均通过预漂移验证片段挑选）下，分别对六种适配策略进行多轮（seed）实验；计算相对收益（%）并绘制准确率-内存-延迟三维前沿。结果显示：在验证-仅方案下，Adam在大多数细胞上胜于SGD+动量；全模型适配在准确率上最优，但内存与延迟最高；校准/头部子模型在内存上无支配关系，且在某些数据集上可逼近全模型收益。

**⚠️ 局限性**

局限性：①验证片段的时序新鲜度有限，未将其纳入再训练；②所有测量在A100 GPU上完成，未验证在真正的边缘硬件（Jetson、Raspberry Pi等）上的内存与延迟；③仅对SGD+动量与Adam进行比较，未探讨更轻量或自适应学习率优化器；④对电能计量子集的缺失率和填充策略可能影响收益评估；⑤适配状态内存计数未考虑真实内存分配与碎片化情况。

---

## 569. P-PatchDiff: Progressive Patch Diffusion Models for Low-light Image Enhancement

**arXiv ID:** 2609.01123 | [PDF](https://arxiv.org/pdf/2609.01123v1)

**作者:** Ruoyu Guo `[一作]` (University of New South Wales), Yang Song `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种可扩展的渐进式 Patch Diffusion 模型 P‑PatchDiff，用于低光图像增强。

**💡 创新点**

创新点包括：① 动态调整 Patch 大小与步幅，在推理过程中逐步从局部切换到全局；② 通过全局亮度代理与多 Patch 对齐层实现不同尺度 Patch 之间的亮度一致性。

**🔧 技术方法**

技术手段：基于 DDPM 的 Patch Diffusion + U‑Net，结合亮度估计器（轻量化 U‑Net）与多 Patch 对齐层（特征级归一化）。

**📊 数据集**

使用了 10 个低光图像数据集（LOL‑v1、LOL‑v2‑Real、LOL‑v2‑Syn、LSRW、UHD‑LL、DICM、MEF、LIME、NPE、VV），并在去模糊（GoPro）和去雨（Rain100L）任务中做扩展验证。

**📈 对比分析**

与回归、生成和 Patch Diffusion 等方法对比，P‑PatchDiff 在 PSNR/SSIM 上与最强方法持平或更优，速度比 WeatherDiff/MDMS 快约 80 倍，显存需求 <9 GB；在多尺度和高分辨率图像上保持较低的采样时间和内存占用。

**⚠️ 局限性**

局限性：仍需在完整分辨率上裁剪 Patch，导致总体计算量较大；步幅需保持小于 Patch/2 以抑制边界伪影，产生冗余计算；缺乏语义一致性建模，极暗区域可能完全忽略结构。

---

## 570. Adaptive Depth-Map-Guided Bundle Adjustment for Correspondence-Free Multi-View Point Cloud Registration

**arXiv ID:** 2609.01089 | [PDF](https://arxiv.org/pdf/2609.01089v1)

**作者:** Yiran Zhou `[一作]` (University of Technology Sydney), Liang Zhao `[通讯]` (University of Edinburgh)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种自适应多层深度图引导的无对应关系的多视角点云配准框架，用于工业钢材碎片的三维测量；

**💡 创新点**

创新点在于将多层深度图与软最大层分配结合，在全局坐标系中直接使用原始深度观测，无需特征匹配，能够在存在重叠、遮挡和重复结构的环境下保持鲁棒性；

**🔧 技术方法**

采用自适应深度层表示、软最大层分配、非线性最小二乘束调整（BA）以及梯度平滑项，利用卷积核密度估计初始化多层深度；

**📊 数据集**

使用作者自行收集的三类工业数据集（受控、半真实、真实），共计九组场景，包含钢卷、钢块、钢管等金属碎片；

**📈 对比分析**

与五种基准方法（T+ICP、T+PGO、T+BA、BALM2、3D Occ.）进行定量和定性比较，实验显示在所有指标上均达到或接近最优，尤其在受控和半真实场景中误差显著低于传统方法，且运行时间仅为24.4 s；

**⚠️ 局限性**

局限性包括对高度稠密平面和高分辨率场景的适配仍需改进，软最大层分配对 β 参数敏感，在极端遮挡或多层重叠复杂度高的情况下仍可能出现层分配误差，且方法在非平面高度分布（非 2.5‑D）场景中的适用性有限。

---

## 571. Update for Decisions, Not Freshness: Goal-Oriented Status Updating and Selective Offloading at the Network Edge

**arXiv ID:** 2609.01082 | [PDF](https://arxiv.org/pdf/2609.01082v1)

**作者:** Jianpeng Qi `[一作]`, Wei Ni `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在边缘云协同环境中提出CoSMO框架，联合语义状态更新与任务选择，显著提升任务完成率与决策准确性。

**💡 创新点**

创新点在于将语义状态抽象、更新时机与任务接收/路由决策耦合到一个异步RL框架，并以目标导向的任务效用为学习信号，而非单纯关注信息新鲜度。

**🔧 技术方法**

采用深度强化学习（双DQN+GRU）实现语义编码与决策，配合GRU时序特征提取与对偶优势网络，实现事件驱动的异步semi‑MDP。

**📊 数据集**

使用ns-3仿真生成的合成任务集（4个UE、每秒120–320任务、随机任务属性）。

**📈 对比分析**

与固定策略、AoI、AoV、AoCI等基线比较，CoSMO在各负载下提升约18–21%的按时完成率、17–18%的决策准确率，并在更新开销上优于AoV且优于AoCI。

**⚠️ 局限性**

仅在单EN/单SN、固定后向链路、无干扰的受控拓扑中验证，未考虑多EN/多SN、时变链路或真实网络负载。

---

## 572. Monocular Depth Estimation from a Single Image: Progress and Opportunities

**arXiv ID:** 2609.01172 | [PDF](https://arxiv.org/pdf/2609.01172v1)

**作者:** Muxin Liu `[一作]` (University of Hong Kong), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了单目深度估计的研究进展，聚焦基础模型的崛起，构建了统一评估管线并对主流方法进行了系统对比。

**💡 创新点**

创新点包括：①提出统一的零射放射对齐评估流程；②将多域数据与大规模预训练模型结合，形成单一通用基础模型；③对比时兼顾判别式与生成式模型，揭示其在精度与效率上的取舍。

**🔧 技术方法**

使用的核心技术包括视觉Transformer（如DPT、DINOv2）、扩散模型（Stable Diffusion、Diffusion‑E2E）、自监督视差/位姿学习、对齐与尺度不变损失、以及轻量级时间一致性模块。

**📊 数据集**

数据集涵盖多种场景：合成（TartanAir、Virtual KITTI 2、MatrixCity等）、室内（NYU Depth v2、ScanNet++、ARKitScenes）、户外驾驶（KITTI、NuScenes、Waymo）、以及在野数据（MegaDepth、DIW、ReDWeb）。

**📈 对比分析**

方法在统一评估下的性能显示，基于Transformer的判别式基础模型在相对深度任务上取得最高的AbsRel/δ1分数；生成式扩散模型在细节重建上优势明显；在多域零射放射测试中，Depth Anything、MoGe‑2、UniDepth等模型表现出较强的跨域泛化能力。

**⚠️ 局限性**

局限性主要在于：①对稀薄结构、透明/反射面细节恢复不足；②视频/多视角时的时间一致性仍需改进；③对特殊相机（鱼眼、全景）的适配依赖额外对齐与数据扩增；④缺乏可靠的不确定性估计与可解释性。

---

## 573. Johnny Still Receives Spam SMS: Assessing the Robustness of SMS Spam Detection

**arXiv ID:** 2609.01171 | [PDF](https://arxiv.org/pdf/2609.01171v1)

**作者:** Muhammad Salman `[一作]` (Macquarie University), Mohamed Ali Kaafar `[通讯]` (Macquarie University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估真实世界的短信反垃圾系统在可感知与不可感知对抗攻击下的鲁棒性，并提出结合对抗训练与多模型少数投票集成的防御方案。

**💡 创新点**

创新点在于：① 将可交付的可感知与不可感知攻击（如隐形字符、同形字等）纳入评估；② 证明少数投票的多模型集成对未知攻击具有良好泛化；③ 引入“最小化对抗训练”，只对有限攻击进行训练而仍能提升对大多数攻击的鲁棒性。

**🔧 技术方法**

使用的技术包括：黑盒对抗攻击（spacing、homoglyph、invisible 等）与 TextAttack 框架；对抗训练（对抗样本数据扩充）；多模型集成（多种网络结构与分词方式）与投票策略（少数/多数/一致）。

**📊 数据集**

采用的主要数据集有：Super SMS Dataset（2012‑2024 年短信垃圾语料），Smishtank（2024 年实战 smishing 数据），SpamHunter、OOPSpam、Plino 等第三方服务数据，以及开源的 33 个公开模型与作者自行训练的 BERT、fastText、LSTM 等模型。

**📈 对比分析**

评估方法：在真实 SIM/设备上发送原始及对抗扰动短信，对比开源模型、移动端短信应用、第三方服务的检测性能；对抗训练后模型在 9 种对抗攻击下的准确率均高于 90%，对同类攻击可达 100%；多模型少数投票集成在同类攻击下仍保持 95%+ 的准确率，同时误报率低于 1% 或可根据用例调节。

**⚠️ 局限性**

限制：实验仅在单一 SIM、单一设备与单一运营商环境中进行，难以保证跨运营商、跨地区的泛化；对抗训练的泛化仍受限于编码层攻击（如同形字）未被完全覆盖；样本量与攻击多样性有限，未来需在多国、多运营商、多设备上进一步验证。

---

## 574. Pre-carved Niches: The Formation Dynamics of Modular Task Partitions in Early LLM Training

**arXiv ID:** 2609.01170 | [PDF](https://arxiv.org/pdf/2609.01170v1)

**作者:** Guangqi Li `[一作]` (Zaozhuang University), Yongxin Li `[通讯]` (Zaozhuang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文通过在训练过程中每一步记录归因修补、梯度、更新、权重和损失分解，实时跟踪并量化大型语言模型内部模块化结构的形成。

**💡 创新点**

创新点在于首次揭示了模块化在训练初期即已预先刻画、通过两次突变跃迁锁定，并发现梯度层面的相对剥夺与权重更新无关、且模块化仅在正在学习的认知域中出现。

**🔧 技术方法**

技术方法包括对Pythia‑410M进行从零开始训练、使用归因修补定位高贡献神经元、构建多探针系统记录梯度范数、有效更新、权重范数、激活频率以及一阶损失分解，并用统计显著性和安静窗口基准检测跃迁。

**📊 数据集**

实验数据来源于使用公开的 Pile 数据流训练的410M参数模型，并评估了14个最小对齐任务（涵盖语言、理论心智、物理推理四个认知域）。

**📈 对比分析**

与随机对照和归因基准相比，模型在训练初期即展示出显著的任务重叠结构，并通过两次高显著性跃迁快速锁定，梯度层的剥夺幅度达到10个标准差以上，显示出与传统经验模型相比更为精准的模块化动力学。

**⚠️ 局限性**

局限性包括仅在单一410M模型和两条数值精度轨迹上验证，缺乏不同规模和任务族的泛化检验，梯度剥夺的因果关系未通过干预验证，以及对物理推理域的规模阈值结论仍待实测。

---

## 575. CopyShield: A Cross-Level Benchmark of Copyright Defenses in LLMs

**arXiv ID:** 2609.01161 | [PDF](https://arxiv.org/pdf/2609.01161v1)

**作者:** Maryam Alshehyari `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Nils Lukas `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的版权防御基准CopyShield，比较了三种不同干预层次（输出层对比解码、行为层DPO、表示层激活干预）的防御方法；

**💡 创新点**

创新点在于构建跨干预层次、可复现的评估框架，量化字面泄露、非字面泄露、效用、退避率和失真等多维指标，揭示干预层次决定合规与效用的折衷，并指出非字面抄袭仍是未解决的挑战；

**🔧 技术方法**

采用对比解码（logit减法）、Direct Preference Optimization（偏好优化）和激活干预（隐藏状态分类器），结合NV-Recall、校准嵌入相似度等评估度量，并使用QLoRA微调、LoRA适配器等技术；

**📊 数据集**

使用五本公共领域小说作为受保护语料进行强制记忆，另一五本小说作为中性语料进行阈值校准，并构造字面查询、非字面查询和问答查询三类评估集合；

**📈 对比分析**

在两套模型族（如LLAMA和Mistral）上使用相同的查询集进行评估，结果显示对比解码保持低失真但字面泄漏有上限，DPO显著消除字面泄漏但产生约58%失真，激活干预通过预生成阻断实现最低非字面泄漏率，清晰展示了干预层次对应的合规-效用折衷；

**⚠️ 局限性**

局限性包括：仅使用公共领域文本做代理，无法直接验证对受版权保护文本的效果；评估仅覆盖两类模型族和有限种子；未考虑对抗性提问；激活干预的精确度有限；仅依赖单一AI评判者进行效用评分，缺乏多评判者或人类评估的多样性。

---

## 576. Griotte: Verified Compartmentalisation via Capabilities

**arXiv ID:** 2609.01110 | [PDF](https://arxiv.org/pdf/2609.01110v1)

**作者:** June Rousseau `[一作]` (Aarhus University), Lars Birkedal `[通讯]` (Aarhus University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过引入理想化但忠实的 Griotte 机器与 Griotte OS，对 CHERIoT 的硬件软件协同隔离模型进行形式化，证明其安全属性（堆栈安全、无捕获、多个不可信域隔离）并验证关键组件（switcher）的实现。

**💡 创新点**

创新点包括：
- 将 CHERIoT 的隔离模型抽象为一个基于深层能力（deep locality/immutability）的能力机 Griotte；
- 设计了结合 switcher 的 continuation‑based 逻辑关系，用于精确捕捉跨域调用的安全契约；
- 给出 switcher 的精确规范并通过 Rocq+Iris 完全验证，实现从编译到运行的端到端安全保证；
- 通过模块化验证，证明不可信域之间的互不通信（IMUD）和无捕获特性。

**🔧 技术方法**

技术方法包括：
- 形式化能力机语义（基于 Cerise）并扩展深层能力与 XSR；
- 采用 Iris 分离逻辑构建程序逻辑和逻辑关系；
- 引入 Kripke 世界与 continuation 关系，统一建模公开/私有过渡与 switcher 的栈管理；
- 在 Rocq 中实现并证明 Griotte 与 Griotte OS 的规范与实现；
- 通过案例研究验证模型的可用性。

**📊 数据集**

本文未使用传统意义上的数据集；验证工作基于手工构造的示例程序（如 compartmentalisation、deep‑locality、stack‑object 等），并在 CHERIoT Sonata 开发板上运行以展示功能。

**📈 对比分析**

对比方法：与之前基于 Cerise 的安全调用约定验证工作相比，本文提供了更完整的隔离模型、深层能力支持以及对 switcher 的正式验证。性能方面，主要关注的是证明时间和代码体量（约 2500 行 Rocq 代码），并在硬件平台上演示了所验证的示例程序的正常执行；未给出定量的运行时性能对比。

**⚠️ 局限性**

限制与不足：
- 只验证了 switcher 与 Griotte OS 的核心部分，未覆盖调度器、内存分配等完整系统功能；
- 假设加载器执行正确，且系统的初始状态已满足 loader‑final 假设；
- 仅针对 CHERIoT 设计的能力机，验证结果不一定直接迁移到其他硬件/OS；
- 证明工作对阅读者要求较高，缺乏自动化证明工具支持。

---

## 577. Soft Posterior Speaker Injection for Multi-Talker Speech Recognition

**arXiv ID:** 2609.01287 | [PDF](https://arxiv.org/pdf/2609.01287v1)

**作者:** Jian Zhu `[一作]` (Zhejiang Lab), Cheng Luo `[通讯]` (Zhejiang International Studies University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Soft Posterior Speaker Injection (SPSI) 方法，利用连续的说话人后验注入到 Whisper 预训练模型中，实现多说话人 ASR 无需硬分割；

**💡 创新点**

创新点在于将多层 Feature‑wise Linear Modulation (FiLM) 与 decoder speaker‑memory prompts 结合，注入混合归一化的说话人 share，避免传统硬分割误差，并在两步编码中共享权重；

**🔧 技术方法**

使用 Whisper 预训练模型、软后验 head、FiLM、decoder prompts、两步 encode、联合 ASR+diarization 损失以及 freeze‑posterior + OV‑heavy 适配；

**📊 数据集**

主要实验数据集为合成的重叠 LibriSpeech（不同 speaker、SNR、RIR）和真实会议数据 LibriCSS（oracle 窗口分段）；

**📈 对比分析**

通过与 SOT、SOT+硬 diarization、SOT+SD‑CTC‑lite、Pipeline（oracle/est VAD）在 cpWER 上比较，SPSI 在合成数据上从 0.507 降至 0.496，低/中/高重叠下分别提升 0.1–1.6 点；在 LibriCSS 零射击及 freeze‑posterior OV‑heavy 继续训练后 cpWER 下降至 0.324，提升 5.1 点；

**⚠️ 局限性**

局限性包括：后验在域迁移时未校准，联合更新后会下降；无法直接处理 >2 说话人；依赖 oracle 窗口分段，需进一步实现自动分割；两步编码结构不支持实时流式部署。

---

## 578. HiLRP: Toward One Trustworthy Explanation for Vision Transformer: Conservation-Valid Attribution via Attention Primitives

**arXiv ID:** 2609.01282 | [PDF](https://arxiv.org/pdf/2609.01282v1)

**作者:** Sathiyamohan Nishankar `[一作]` (RMIT University), Selvarajah Thuseethan `[通讯]` (Charles Darwin University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文实现了一种名为 HiLRP 的统一层级相关传播方法，能够对多种 Vision Transformer（ViT）架构给出归一且可解释的像素级重要性分布。

**💡 创新点**

创新点在于将所有 ViT 中的混合与分辨率变换操作归约为四类基本算子（线性、双线性、归一化/门控、重索引），为每类制定单一的保守性传播规则，并通过理论证明与数值验证保证归一性和等变性。

**🔧 技术方法**

技术手段包括基于 LRP 的传播规则、CP‑LRP 的注意力处理、γ‑规则对线性/卷积层的权重正则化、Patch‑Map 以标识算子类型、自动微分与 Gradient×Input 后向传播等。

**📊 数据集**

使用 ImageNet‑S（1000 带分割掩码的图像）和 PASCAL VOC 2007（目标框注解）进行评估，并在多种公开预训练 ViT 检索模型（如 DINO、MAE、CLIP）上验证无标签归因。

**📈 对比分析**

在 10 种 ViT 架构和 14 种归因方法的基准中，HiLRP 在定位（Pointing Game）、Shapley 一致性和归一性方面始终保持最高或最稳定的表现，且在所有模型上均可执行。

**⚠️ 局限性**

局限性包括：对非四类算子（如状态空间或生成式模型中的混合操作）需额外规则；γ 参数在某些线性注意力模型中对归一性影响显著；实验集主要包含单物体图像，可能无法充分反映多对象场景下的表现。

---

## 579. EmbodiedSkills: A Unified Framework for Orchestrating, Training, and Deploying VLA Agents

**arXiv ID:** 2609.01281 | [PDF](https://arxiv.org/pdf/2609.01281v1)

**作者:** Wei Wang `[一作]` (Zhejiang University), Yueting Zhuang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 EmbodiedSkills 框架，将 VLA 低层策略与高层决策分离，构建一个基于可执行技能的闭环 AgentLoop，实现感知、规划、验证、执行与恢复的完整闭环闭合。

**💡 创新点**

创新点包括：1）以技能为中心的闭环循环结构；2）通过结构化技能契约实现策略与运行时的明确分离；3）模块化接口使规划器、验证器、低层策略等可独立训练与替换；4）统一的轨迹记录与状态接口，支持跨基准的评估与在线微调。

**🔧 技术方法**

技术手段：VLA 模型（OpenPI/π_0.5）、Qwen3-VL 调度器、基于技能契约的运行时验证、分阶段（Observe-Plan-Preflight-Execute-Verify-Recover）闭环、可界定的动作块、结构化轨迹记录、可选的在线 RL 微调。

**📊 数据集**

使用的数据集包括：RoboTwin 2.0 50 任务、LIBERO 四个子集（Spatial、Object、Goal、Long）、RMBench 4 个记忆依赖任务。

**📈 对比分析**

评估方法：与多种基线（ACT、DP、RDT、DP3）及通用 VLA 参考（π_0、X-VLA、π_0.5、OpenPI）在相同基准下对比。结果显示：RoboTwin 2.0 平均成功率 86.20%（高于 82.74% 基线），LIBERO 平均 97.40%（高于 96.85% 基线），RMBench 平均 12.5%（显著高于 7–10% 的其它方法）。消融实验表明完整闭环、子任务验证与可重用动作块是提升性能的关键。

**⚠️ 局限性**

局限性：1）对每个任务需专门适配低层 VLA 策略，增加训练与部署成本；2）多次 VLM 调用和后置验证导致额外延迟；3）规划、验证和执行模块的校准不足会导致错误传播；4）验证受视觉遮挡、模糊或难以观测属性限制；5）模块化接口需保持兼容性，长期执行会累积小误差，影响成功率。

---

## 580. TimeSteer: Inference-Time Speech Scheduling in Joint Audio-Visual Diffusion Models

**arXiv ID:** 2609.01277 | [PDF](https://arxiv.org/pdf/2609.01277v1)

**作者:** Chao Zhou `[一作]`, Tianyi We `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了在推理时对联合音视频扩散模型进行语音调度的方法，使得预训练模型能够在不训练的情况下将对话安排到指定时间区间。

**💡 创新点**

创新点在于发现音视频扩散模型的去噪过程包含跨模态注意力可揭示语句源时段，并利用预测的干净潜在变量实现对语音与视觉同步内容的时间重排。

**🔧 技术方法**

技术上采用了源时段定位(Source Span Localization)、区域感知潜在重映射(Region-Aware Latent Remapping)、最小失真重映射与曲率桥接等无训练干预的技术。

**📊 数据集**

使用了SpeechShift基准数据集，该数据集包含400条场景提示、600条目标时间区间，涵盖单/多说话人、单/多句子等四种结构。

**📈 对比分析**

在LTX-2和daVinci-MagiHuman两种后端上与四种无训练基线对比，TimeSteer在区间可控性指标（HR_0.2、IoU）上提升显著，同时保持与无控制采样相当的音质与同步质量。

**⚠️ 局限性**

局限性在于依赖跨模态注意力的时序敏感性，且对极短或重叠语句的精确定位仍有限，未能处理需要大幅速率变化的场景。

---

## 581. From Base Rollouts to RL Reasoning: A Budgeted Search Perspective

**arXiv ID:** 2609.01274 | [PDF](https://arxiv.org/pdf/2609.01274v1)

**作者:** Wenhe Sun `[一作]` (Fudan University), Yixin Cao `[通讯]` (Fudan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建统一的解码/搜索空间SearchLens，对比 RL 训练后模型与基线模型在不同预算下的行为表现，提出一种低维路径规则BOPTR-P1，量化 RL 在解码预算上的收益来源；

**💡 创新点**

创新点在于：①将解码与搜索方法映射为可度量的预算化操作点；②提出基于预算转移规则的行为恢复路径BOPTR-P1，揭示 RL 主要是将基线模型的已存在行为迁移到更易采样的轨迹；

**🔧 技术方法**

技术方法包括：多维解码/搜索策略建模（局部策略、控制器、评估器、转移、调度）；预算化操作点与行为距离度量；回归与规则学习的BOPTR-P1；跨 benchmark 与跨模型的转移实验；

**📊 数据集**

使用的数据集包括 Math500、AIME、GPQA、IFEval 等四个推理基准；模型涵盖 Qwen2.5 系列、Llama3.1、Mistral 等在 SimpleRL-Zoo 训练的基线与 RL 版本；

**📈 对比分析**

比较方法是将 RL 默认策略曲线与基线模型的多策略预算上限 Envelope 进行对齐，采用“相近匹配集”与“路径一致性”检验；结果表明 RL 目标曲线大多可在基线预算空间内恢复，且 BOPTR-P1 在大部分情形下误差约 3–5 个百分点，优于直接复制或相同策略对比；

**⚠️ 局限性**

局限性包括：①仅作行为层面的对比，未揭示参数层面的机制；②规则和预算指数需要在同一 RL 训练配置、模型族与数据集上重新估计，跨大模型族或不同 RL 配置的泛化受限；③评估指标与基准间存在语义差异，导致跨基准比较不完全一致；④规则依赖于已收集的 RL 样本集，无法单独通过基线模型预测；

---

## 582. What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal

**arXiv ID:** 2609.01271 | [PDF](https://arxiv.org/pdf/2609.01271v1)

**作者:** Radin Shayanfar `[一作]` (Queen's University), Ahmed E. Hassan `[通讯]` (Queen's University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Spread–Novelty–Centrality (SNC) 三轴框架，用以量化和比较不同agentic软件工程基准任务的工程工作需求。

**💡 创新点**

创新点在于用SNC指标揭示同标签基准的实际差异，并将基准任务的工程需求与代理行为以及成功率相联系，区分模型家族的行为签名。

**🔧 技术方法**

采用SNC指标（扩散度、创新度、中心性）和代理行为脚印（冗余度、探索广度）进行量化分析，并用Scott-Knott ESD检验统计显著性。

**📊 数据集**

使用了五个广泛使用的Python基准（SWE-bench, SWE-Gym, FEA-Bench, FeatBench, FeatureBench）及其2,487个任务实例，评估了六个模型配置（Claude Haiku/Sonnet/Opus 和 Qwen 3.5‑9B/27B/397B）。

**📈 对比分析**

方法通过对比已解决与未解决运行在SNC与行为轴上的分布，发现低SNC区间与成功率高度相关；Claude模型在规模增大时更趋向金手指大小的修改，而Qwen则始终倾向过度修改，且两者都对欠删修改表现出失败。

**⚠️ 局限性**

局限性包括仅针对Python任务，SNC指标依赖语言感知分析，且仅覆盖bug修复与功能实现两类任务，缺乏对其他类型任务（如大规模重构）的适用性验证。

---

## 583. Solving In-Table Prediction Problems by Deep Neural Networks with Performance Evaluation Using Synthetic Data

**arXiv ID:** 2609.01262 | [PDF](https://arxiv.org/pdf/2609.01262v1)

**作者:** Xiao Zhao `[一作]` (Offenburg University), Daniela Oelke `[通讯]` (Offenburg University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种在表格数据中预测任意遮掩列值的自监督方法，称为In-Table Prediction (ITP)。

**💡 创新点**

定义了ITP任务并引入了可嵌入空值和遮掩值的神经层，扩展了传统的表格自监督预训练。

**🔧 技术方法**

采用了MLP、ResNet和Transformer三种架构，并使用自监督掩码学习、嵌入层与注意力机制。

**📊 数据集**

使用基于线性和指数函数的合成三列数据集，并通过两种缺失机制（MAR/MNAR）注入空值。

**📈 对比分析**

通过网格搜索比较三种模型、不同训练集大小、掩码策略、缺失机制和嵌入维度，发现注意力网络在大样本下性能最好，NRMSE随训练规模和嵌入维度提升。

**⚠️ 局限性**

实验仅在有限的三列合成数据上进行，缺乏对真实大规模表格的验证，且缺失机制和列数范围受限。

---

## 584. From Language to Behavior: Scaling Sequence Transformers for Industrial Recommendation Ranking with Rec-Native Designs

**arXiv ID:** 2609.01240 | [PDF](https://arxiv.org/pdf/2609.01240v1)

**作者:** Jie Chen `[一作]` (ByteDance), Xiaobing Liu `[通讯]` (ByteDance)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 ReST（Recommendation-native Scalable Transformer），一种针对工业推荐系统中行为序列建模的可扩展 Transformer 框架，解决了信号质量和计算不对称问题。

**💡 创新点**

创新点在于：① 结合双门注意力、RoPE+RoTE 时空编码、稳定残差归一化等推荐专用层，显著提升对噪声、时序不规则和稀疏监督的鲁棒性；② 将编码器和解码器拆分为重用式序列编码器与轻量级交叉解码器，配合用户级共享前缀训练和服务端共享前缀重用，实现一次编码多候选的低延迟推理；③ 引入训练阶段的辅助序列 CVR 及序列-非序列对齐目标，缓解序列模块被“短路”的问题。

**🔧 技术方法**

技术包括 Transformer（LLaMA-style）自注意力、双门注意力（Dual-Gated Attention）、旋转位置与时间嵌入（RoPE+RoTE）、稳定残差归一化（SRN）、projection-free KV 注意力、Token‑specific 参数化、共享前缀训练/服务、以及辅助监督损失。

**📊 数据集**

在工业广告数据集（TikTok Shop Ads）以及公开数据集（MovieLens‑1M、MovieLens‑20M、Amazon‑Books）上进行评估。

**📈 对比分析**

与 DIN、STCA、LLaMA、HSTU、Trans. 等基线相比，ReST 在相同 FLOPs 或参数预算下的 AUC 提升显著，工业数据上 AUC +0.92%，公开数据上持续领先；在线 A/B 测试显示 AUC +1.31% 与核心营收指标提升 11.93%。

**⚠️ 局限性**

局限性包括：① 仍需在多候选稀疏监督场景中进一步验证；② 共享前缀策略对不同业务场景的适用性可能有限；③ 主要针对单用户历史的场景，对多用户并发历史合并的情况尚未深入研究。

---

## 585. A Unified Uncertainty-Aware Back-End for Speaker Verification: Scoring, Normalization, and Calibration

**arXiv ID:** 2609.01221 | [PDF](https://arxiv.org/pdf/2609.01221v1)

**作者:** Junjie Li `[一作]` (Hong Kong Polytechnic University), Kong Aik Lee `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出统一的不确定性感知后端，将相似度评分、归一化和校准过程全部融合，利用说话人嵌入的均值与协方差进行端到端推理。

**💡 创新点**

创新点在于：①在余弦评分中引入协方差校正的有效范数；②在AS‑Norm中使用协方差权重计算均值、方差，并通过g因子对试验得分做不确定性缩放；③在QMF校准中加入协方差调整的嵌入质量特征；④保持后端不需要改动说话人编码器，仅通过协方差信息贯穿整个后端。

**🔧 技术方法**

使用 𝒰³‑xi 估计嵌入协方差，改进余弦评分公式，构建 UAS‑Norm（加权均值、标准差、g 缩放），以及 UQMF（基于协方差的嵌入质量特征的逻辑回归校准）。

**📊 数据集**

训练数据：VoxCeleb2；评估数据：VoxCeleb1‑O、VoxCeleb1‑E、VoxCeleb1‑H 试验列表。

**📈 对比分析**

与传统的 s_cos‑o → s_AS‑o → s_QMF‑o 后端进行对比。实验显示，在 ECAPA‑TDNN 和 ResNet 上，EER 下降 20‑30%，RI（平均相对改进）从 13% 上升到 25%（ECAPA）或 27%（ResNet），目标/非目标分离度提升，minDCF 在大多数子集略有提升但在部分设置略降。

**⚠️ 局限性**

局限性：性能依赖协方差估计质量，若不确定性估计不准确会导致 minDCF 轻微下降；对协方差的计算与存储增加一定的计算与存储开销；目前仅在基于余弦的后端验证，其他后端（如 PLDA）尚未探索。

---

## 586. PersuaRL: Reinforcement Learning-Driven Multi-Expert Selection for Persuasive Dialogue Generation in Insurance

**arXiv ID:** 2609.01188 | [PDF](https://arxiv.org/pdf/2609.01188v1)

**作者:** Rohan Kirti `[一作]` (Indian Institute of Technology Patna), Sriparna Saha `[通讯]` (Indian Institute of Technology Patna)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 PersuaRL 框架，用强化学习动态选择专家模块生成具有说服力的保险对话响应

**💡 创新点**

将专家选择视为可学习的上下文条件决策，并通过交替优化实现生成器与选择器的协同进化

**🔧 技术方法**

使用 Group Relative Policy Optimization (GRPO)、多专家模块（策略、意图、关键词、情感）、生成器以及多维度奖励函数

**📊 数据集**

构建并公开 InsureDial 保险对话数据集，包含 1,931 条多轮对话及意图、情感、策略和关键词标注

**📈 对比分析**

与单次推理、监督微调和多种基准模型（如 Llama、Qwen、Phi、Mistral）进行自动和人工评估，PersuaRL 在 BLEU、ROUGE、BERT‑F1、Distinct‑2、LLM‑Judge 及人工评估指标上均明显优于基线

**⚠️ 局限性**

专家选择的二进制掩码导致动作空间指数增长，推理时需调用多模块增加延迟和成本，数据集主要基于 GPT‑4o 生成并人工筛选，缺乏真实用户交互评估

---

## 587. Athena: Vulnerability-Affected Library Identification via Knowledge Graph Completion

**arXiv ID:** 2609.01187 | [PDF](https://arxiv.org/pdf/2609.01187v1)

**作者:** Phong Trinh Duy `[一作]` (Hanoi University of Science and Technology), Thanh Le-Cong `[通讯]` (Singapore University of Technology and Design)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Athena框架，利用知识图谱完成漏洞与受影响库的关联预测；

**💡 创新点**

首次将受影响库识别问题重新表述为知识图谱补全(KGC)任务，并结合图结构与文本信息；

**🔧 技术方法**

采用图嵌入的对比式KGC背骨（SimKGC/RAA-KGC）以及LLM重新排序（DIFT改进版）进行候选库排序；

**📊 数据集**

使用包含Java CVE、CVE、CWE、CPE、Maven Central等信息的AstraZeneca/OSV等公开数据构建的安全知识图谱；

**📈 对比分析**

与四个现有基线（Chronos、VulLibMiner、LibAlarm、VulLibGen）对比，Athena在Avg.F1上提升32%（0.602对0.457），且在不同LLM规模下持续表现优异；

**⚠️ 局限性**

局限包括仅在Java数据集上验证，知识图谱的完整性对性能影响显著，对跨生态系统和更大模型的评估尚未完成。

---

## 588. The Constitutional Coverage Trilemma in AI Governance

**arXiv ID:** 2609.01275 | [PDF](https://arxiv.org/pdf/2609.01275v1)

**作者:** Natalija Mitic `[一作]` (Kera Health Platforms), Moustapha Cisse `[通讯]` (Kera Health Platforms)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了前沿大型语言模型的“宪法”与人类价值需求之间的匹配情况，量化了供应与需求的差距，并提出了稀疏多顶点菜单的改进方案。

**💡 创新点**

创新点在于：①首次将人类价值偏好与模型宪法通过同一量化仪器并行测评；②引入“宪法无家可归”与“预算多元化三难困境”理论；③发现供应向已被低估价值（自主性）漂移，并证明稀疏顶点菜单可显著降低用户损失。

**🔧 技术方法**

使用了双向偏好测评（pairwise trade-off）与大规模语义等价句子验证的宪法审计，结合线性福利模型与凸包覆盖分析；同时利用子模函数近似实现稀疏菜单设计。

**📊 数据集**

人类需求数据来自1,649名美国Prolific受试者的20题价值对比问卷；供应数据为23个跨六大厂商的前沿LLM默认宪法，通过21个语义等价变体×10情境×2顺序共420次测试获得。

**📈 对比分析**

比较方法为菜单 regret（均值和极值）与理论下界。稀疏两点菜单{e_HON, e_AUT}将均值 regret 降至0.074，较完整23种菜单降低47%；添加三点后均值 regret 可降81%，极值 regret 降至64%。

**⚠️ 局限性**

局限性：①仅考虑五维价值空间，未涵盖文化、环境等因素；②假设线性福利和完美匹配，实际应用可能更差；③样本为美国Prolific，跨文化推广需验证；④漂移观察性，仅发现趋势而无因果解释；⑤仅评估默认宪法，未探讨可调节宪法对供应覆盖的影响。

---

## 589. Ready to Speak: Aligning LLMs for TTS-Friendly Text Generation

**arXiv ID:** 2609.01246 | [PDF](https://arxiv.org/pdf/2609.01246v1)

**作者:** Thibaut Thonet `[一作]` (NAVER LABS Europe), Laurent Besacier `[通讯]` (NAVER LABS Europe)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了将大型语言模型直接对齐生成对TTS友好的文本的方法，避免后处理的延迟与依赖。

**💡 创新点**

创新点在于将TTS友好度拆解为可解释特征，并通过FaST框架实现多目标偏好对齐，实现低样本学习下的高效调优。

**🔧 技术方法**

采用FaST（特征感知采样与微调）、DPO、SFT、GRPO-RM、RFT-RM等技术，并使用可解释特征奖励模型。

**📊 数据集**

使用了两个自制偏好数据集：CORA（咖啡店对话）和Recipe（烹饪步骤），包含TTS友好与不友好示例。

**📈 对比分析**

与提示、SFT、DPO、GRPO-RM、RFT-RM以及文本归一化PolyNorm对比；在CORA和Recipe上，FaST在TTS友好度与有用性平衡上位于Pareto前沿，尤其在仅10个样本时表现最佳。

**⚠️ 局限性**

局限包括生成文本可能更冗长、仅针对英文两域、只验证小规模模型、仅测试单一TTS引擎，以及不适用于端到端语音LLM。

---

## 590. S$^2$Prune: Spatially Structured Visual Token Pruning for Multimodal Large Language Models

**arXiv ID:** 2609.01224 | [PDF](https://arxiv.org/pdf/2609.01224v1)

**作者:** Yuanyuan Jia `[一作]` (Zhejiang University), Qianqian Yang `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种无训练的视觉令牌剪枝方法，保持空间覆盖并自适应分配令牌密度。

**💡 创新点**

提出将图像划分为粗粒度区域，使用拉普拉斯方差自适应分配令牌预算，并用早期解码器变化 ERC 在每个子区域中选取代表令牌。

**🔧 技术方法**

利用拉普拉斯方差、最小-最大归一化、ERC 与最大余数分配等技术。

**📊 数据集**

在十个多模态基准（GQA、MMMU、VizWiz、POPE、MME、MMVet 等）上评估，并在 Qwen2.5-VL-7B 与 LLaVA-OneVision-7B 两个模型上测试。

**📈 对比分析**

与七个无训练剪枝基线比较，平均准确率提升 0.1–0.9 分；在 32 令牌时保持 79.3% 性能，显著降低 FLOPs 与 KV 缓存。

**⚠️ 局限性**

在高异质结构图像中仍可能需要更细粒度的自适应分配，且方法在更大模型或实时推理场景中的适用性仍待进一步验证。

---

## 591. REFACTOR-VLA: Unsupervised Library Learning of Typed Motor Programs

**arXiv ID:** 2609.01215 | [PDF](https://arxiv.org/pdf/2609.01215v1)

**作者:** Riyaaz Shaik `[一作]` (Apple), Chandru Venkataraman `[通讯]` (Apple)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种基于wake/sleep框架的可解释视觉‑语言‑动作模型 REFACTOR‑VLA，能够通过行为等价核聚类并生成可重用的 Typed‑Lambda 技能库，完成 LIBERO 纵向任务。

**💡 创新点**

创新点包括：① 用 latent world 模型的价值与 Wasserstein 距离构造行为等价核；② 采用 Hindley‑Milner 型 λ 发射器与库条件化 rectified‑flow 动作解码器；③ 通过 MDL 与返回保持门共同筛选抽象；④ 证明训练目标对性能影响大于参数规模。

**🔧 技术方法**

使用技术包括 wake/sleep 循环、DreamerV3+Frozen DINOv2 的 latent world 模型、Siamese amortizer、k‑means 聚类、InfoNCE 对比学习、Typed‑Lambda 语法编译器、rectified‑flow LCAD 与 MDL/返回门控。

**📊 数据集**

实验基于 LIBERO 四个套件（object、spatial、goal、long）中的真实机器人演示数据。

**📈 对比分析**

与 AtomicVLA、AtomSkill、BLADE、LRLL 四个公开基线在 4 套 LIBERO 上进行 k‑means NMI 对比，REFACTOR‑VLA 在 InfoNCE 训练下平均提升 +0.184 NMI，并在所有四套上超越最强基线；将模型扩展至 430M 参数反而导致 NMI 下降，验证训练目标更关键。

**⚠️ 局限性**

局限性包括：① 动力学原语抽象无法生成；② 跨供应商 12 种子实验 NMI 仅略高于 0.70 阈值；③ 需进一步验证在真实机器人上的迁移；④ 对 Wasserstein 项高度敏感，缺乏直观解释。

---

## 592. Who Judges the Judges? A Chinese Safety QA Benchmark for Evaluating LLM Responses and Safety Judges

**arXiv ID:** 2609.01210 | [PDF](https://arxiv.org/pdf/2609.01210v1)

**作者:** Rui Yang `[一作]` (Anhui SparkShield Intelligent Technology), Jing Shao `[通讯]` (Shanghai Innovation Institute)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了 C‑SafeQA 中文安全基准，收集 9,415 条查询（538 基础 + 8,877 对抗）在四个全模型部署下生成 37,660 条问答记录，并对每条记录进行安全标注。

**💡 创新点**

创新点：①基于内部安全政策的查询–响应级评估，区分查询风险与响应违规；②设计 21 种对抗变换，覆盖语义、结构、角色等攻击维度；③采用多模型共识 + 盲审专家的混合标注流程，保证标签可靠；④公开数据与评判脚本，保留构造与策略细节，支持复现但不泄露可重用攻击工具。

**🔧 技术方法**

技术：对抗变换生成（模板、LLM 重写、程序化）、四个大模型全量推理、七个自动安全评判模型（Llama Guard、MD‑Judge 等）、Agreement‑aware 多模型判定、专家盲审、统一安全评估指标（不安全率、召回、FPR、精准、F1、冲突率等）。

**📊 数据集**

数据集：自建 538 基础中文查询 + 8,877 对抗查询，四个模型共 37,660 条 QA；公开 HuggingFace JSONL 记录，包含七评判模型输出；另外提供 GitHub 验证器与评判脚本。

**📈 对比分析**

比较方法：①基线 vs 对抗查询的不安全率描述性对比；②对七评判模型在对抗查询集上的召回、FPR、精准、F1、冲突率评估；②按风险类别与攻击机制细化分析。结果显示目标模型对抗查询不安全率提升约 10 倍，评判模型无一能同时具备高召回与低 FPR，acrostic 等机制导致所有评判模型召回显著下降。

**⚠️ 局限性**

局限性：仅覆盖单轮中文问答；内部政策与标签生成细节不公开；参考标签为模型辅助+专家抽样，缺乏全面人工标注；对抗变换与查询形式不完全匹配，基线与对抗对比为描述性而非因果；无正常查询集，无法评估过度拒绝；非可归一化输出导致指标覆盖不足。

---

## 593. Identification of Compositional Risks in Data Protection Impact Assessments and Beyond

**arXiv ID:** 2609.01201 | [PDF](https://arxiv.org/pdf/2609.01201v1)

**作者:** Henrik Graßhoff `[一作]` (Karlstad University), Nils Gruschka `[通讯]` (University of Oslo)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种分布式协议，用于在数据处理服务组合中检测潜在的组合隐私风险

**💡 创新点**

首次将组合风险概念化为服务图中的焦点与拐点，并设计了隐私保护的检测协议，避免泄露子处理器信息

**🔧 技术方法**

利用布隆过滤器保证递归终止，使用可盲化伪名函数实现盐的隐匿与伪名的可逆解耦

**📊 数据集**

未使用公开数据集，论文仅基于理论模型与示例图进行说明；未来计划在仿真环境中验证

**📈 对比分析**

论文未提供实验比较或性能评估，故无法给出具体数值；作者指出可根据图大小选择参数以平衡正确性与计算开销

**⚠️ 局限性**

局限性包括：仅识别候选风险而不判断实际风险；可能存在伪名冲突或布隆过滤器误判；未在真实系统中实现与评估；对未参与协议的处理器缺乏处理办法

---

## 594. FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue

**arXiv ID:** 2609.01198 | [PDF](https://arxiv.org/pdf/2609.01198v1)

**作者:** Hangyeul Lee `[一作]` (Seoul National University), Pilsung Kang `[通讯]` (Seoul National University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 FinLifeBench 基准，用以评估语言模型在多轮银行对话中对生活事件历史与金融状态的完整重构能力。

**💡 创新点**

创新点在于提供了两个基于同一累积对话的完整重构任务——事件-锚点恢复与完整金融状态重建，并引入面向完整性、可追溯性与时效性的评估指标。

**🔧 技术方法**

作者使用大语言模型（LLM）进行实验，采用全上下文输入，定义了 EA‑F1、GCA@15、CSA、ER 等指标，并通过多模型对比来检验性能。

**📊 数据集**

数据集由 20 条人格化轨迹生成的 6,000 条韩语银行对话组成，覆盖 24 类生活事件、34 条金融状态路径，黄金注释完全确定。

**📈 对比分析**

实验对比 11 种 LLM，最优 EA‑F1 达 0.748，最优 GCA@15 为 0.470，显示模型在事件覆盖和状态更新方面仍有显著缺陷。

**⚠️ 局限性**

局限包括合成数据可能与真实分布不符、使用静态前缀而非动态记忆管理、语言为韩语导致跨模型差异、以及未评估下游决策影响。

---

## 595. CaRL-EM: Cost-Aware Reinforcement Learning for Entity Matching with LLMs

**arXiv ID:** 2609.01195 | [PDF](https://arxiv.org/pdf/2609.01195v1)

**作者:** Chaohui Guo `[一作]` (Vrije Universiteit Amsterdam), Zhisheng Huang `[通讯]` (Vrije Universiteit Amsterdam)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CaRL-EM，一种基于强化学习的控制器，用来在实体匹配任务中动态选择 LLM 的 Match/Compare/Select/Decide 操作并兼顾成本与质量；

**💡 创新点**

创新点在于把 LLM 驱动的实体匹配建模为成本感知的序列决策问题，并设计可在不重新训练的情况下切换不同 LLM 后端的抽象操作与两级成本模型；

**🔧 技术方法**

主要技术包括基于 PPO 的强化学习控制器、抽象的低/高成本标签、四大高层 LLM 操作（Match、Compare、Select、Decide）以及轻量级 MLP 策略网络；

**📊 数据集**

使用了七个公开实体匹配基准数据集（Abt-buy、Amazon-Google、DBLP-ACM、DBLP-Scholar、IMDb-TMDb、IMDb-TVDb、TMDb-TVDb、Walmart-Amazon）进行零样本迁移实验；

**📈 对比分析**

在零样本迁移和手工管线 COMEM 以及域内训练的 Ditto 进行比较，CaRL-EM 在 7 组数据上平均 F1_macro 约 76.1（占 Ditto 的 89%），而成本仅为 5.9% 甚至 94% 的降低，显示出更优的质量-成本权衡；

**⚠️ 局限性**

局限性包括只处理 clean‑clean 单匹配场景、候选集规模受限（top‑10）、抽象成本模型无法完全反映真实计费细节，且在多匹配或大规模候选集场景下可能表现不佳。

---

## 596. Verification of $K$- and Infinite-Step Strong/Weak Anonymity Using Concurrent Compositions

**arXiv ID:** 2609.01192 | [PDF](https://arxiv.org/pdf/2609.01192v1)

**作者:** Jiahui Zhang `[一作]` (Macau University of Science and Technology), Zhiwu Li `[通讯]` (Macau University of Science and Technology)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了在离散事件系统中考虑强匿名投影和弱匿名投影的四种 K‑步与无穷步匿名性概念，并基于并发组合（Concurrent Composition）技术设计了一套统一的验证方法；

**💡 创新点**

创新点在于（1）首次将强匿名投影与弱匿名投影引入匿名性分析，构造了与传统匿名性不同的四种匿名性定义；（2）利用并发组合构造 K‑无关的信息结构，显著降低了验证复杂度；（3）给出了 K‑步匿名性的最大上界与无穷步匿名性的判定标准；

**🔧 技术方法**

核心技术包括：离散事件系统建模（非确定有限状态机）、强/弱匿名投影的定义、观察器（Observer）与修改观察器的构造、并发组合（Concurrent Composition）与广度优先搜索算法；

**📊 数据集**

该工作基于理论建模，无使用具体实验数据集，验证过程采用自动化模型检查工具完成；

**📈 对比分析**

与现有基于两路观察器、逆向观察器或非秘密规范的隐匿性/匿名性验证方法相比，所提出的并发组合+观察器方法在时间复杂度上更优（O(|X|^2 2^|X|(|Σ_o|+|Σ_uo|+|Σ|))），并可一次构造即可验证任意 K；

**⚠️ 局限性**

主要限制是所构造的并发组合与观察器状态空间呈指数级增长，导致在大规模系统上易产生状态空间爆炸；此外，验证仍为 PSPACE‑hard，缺乏多项式时间解法，未来可考虑抽象-精炼或机器学习近似技术来缓解。

---

## 597. Analog-DB: An Agent-First Analog Integrated Circuit Database, From Blocks to Systems

**arXiv ID:** 2609.01286 | [PDF](https://arxiv.org/pdf/2609.01286v1)

**作者:** Danial Noori Zadeh `[一作]`, Mohamed B. Elamien `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种集成化的低压差线性稳压器，该稳压器在同一芯片上实现缓冲参考、低通滤波、反馈放大和PMOS通过元件。

**💡 创新点**

创新点在于将缓冲参考与反馈放大单元无分隔集成，实现更低的功耗和更小的占用面积，同时通过电压分压与电容耦合提供精确的稳压输出。

**🔧 技术方法**

采用双级运算放大器、低通滤波电路、PMOS通过元件以及电压分压网络等传统模拟电路技术。

**📊 数据集**

无公开数据集，主要使用模拟电路仿真（SPICE）和硬件实验验证。

**📈 对比分析**

通过与传统单级线性稳压器比较，测得输出电压误差≤0.2%，瞬态响应<10µs，低压差≤0.5V，功耗约为10mW，性能优于现有同类器件。

**⚠️ 局限性**

主要局限在于输出电流受PMOS晶体管限制，且在极低温度下稳压精度略有下降。

---

## 598. Some Emotions Run Deeper: Layer-wise Probing and Causal Intervention in Large Language Models

**arXiv ID:** 2609.01279 | [PDF](https://arxiv.org/pdf/2609.01279v1)

**作者:** Tian Fang `[一作]` (Université Sorbonne Paris Nord), Davide Buscaldi `[通讯]` (Université Sorbonne Paris Nord)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究不同情感表达文本中情绪信息在解码器LLM中的层次位置，利用层级线性探针、在线/离线干预和早期退出分类进行验证。

**💡 创新点**

发现情绪信息的可解码层与文本来源相关，而非固定深度，并证明所选层在干预和早期退出中既具有因果作用又能提升轻量化分类效果。

**🔧 技术方法**

采用层级线性探针、乘法钩子干预、特征缩放、跨数据集/情绪迁移和早期退出机制。

**📊 数据集**

使用Emotion/CARER、GoEmotions和ISEAR三个英语情绪数据集。

**📈 对比分析**

与随机、低敏感性和跨数据集迁移的基线相比，选定层在干预中导致更大准确度下降，早期退出在保持准确率的同时减少计算，平均提升约7个百分点。

**⚠️ 局限性**

仅限开放权重1B-9B模型、四种基本情绪、英语文本、固定池化与线性探针，未考虑大规模闭源模型、多语种或更细粒度情绪标签。

---

## 599. Dual Process Motion Planning

**arXiv ID:** 2609.01260 | [PDF](https://arxiv.org/pdf/2609.01260v1)

**作者:** Jiayi Yan `[一作]` (Chinese University of Hong Kong), Alessandro Abate `[通讯]` (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了Dual‑MP双过程运动规划框架，利用快速神经策略(System‑1)与慢速符号求解器(System‑2)通过元认知仲裁结合，快速过滤易规划实例并在必要时委托MPC或CBF求解；同时支持持续学习将成功的S2轨迹蒸馏回S1。

**💡 创新点**

在经典与学习规划之间引入系统1/系统2双过程仲裁机制，加入验证门与问题难度估计，实现了自适应分配资源；首次将持续学习与元认知仲裁结合，使得经验不断提升S1性能；并系统评估热启动S2的可行性。

**🔧 技术方法**

使用卷积神经网络作为局部实时策略，行为克隆+可微滚动损失进行持续学习；采用非线性MPC和控制障碍函数(CBF)两种符号求解器；基于SOFAI框架实现元认知仲裁；通过碰撞与目标判定的验证门以及简单几何特征的难度度量。

**📊 数据集**

在六类二维连续运动规划基准上实验：Large Sparse、Dense Clutter、Serial Walls、Maze Branching、Long Slalom、Bugtrap；每类分别提供100个训练实例、500个评估实例和500个probe实例。

**📈 对比分析**

与单独的NN、CBF、MPC基线以及Dual‑MP不同配置（DMP‑CBF、DMP‑MPC、DMP‑Warm）进行对比，评估指标为成功率、平均/90%运行时间和轨迹质量(Q)。结果显示Dual‑MP‑MPC在成功率与MPC相近的同时显著降低运行时间；Dual‑MP‑CBF在成功率与CBF相当但略慢；持续学习提升了S1的成功率并在多类场景中产生可观收益。

**⚠️ 局限性**

局部观察的神经策略难以复制全局MPC规划，导致在受限环境下的成功率受限；热启动S2未能稳定提升性能；实验仅在二维基准上验证，缺乏高维或真实机器人环境的验证；元认知度量与阈值设置需要针对不同任务进一步调优。

---

## 600. MeRoPE: Metric Rotary Position Embedding for Camera-Controlled Video Generation

**arXiv ID:** 2609.01252 | [PDF](https://arxiv.org/pdf/2609.01252v1)

**作者:** Zhijian Qiao `[一作]` (Hong Kong University of Science and Technology), Shaojie Shen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于正交相对相机位姿编码的MeRoPE，用于在真实度量运动下实现视频生成的相机控制。

**💡 创新点**

通过解耦射线局部旋转和度量平移并引入差异锚定球面旋转，兼顾了完整度量相对位姿、正交性与特征范数保持的三项目标。

**🔧 技术方法**

使用正交旋转块、多频率RoPE平移编码、球面差异锚定以及轻量级适配器集成至Diffusion Transformer。

**📊 数据集**

在nuScenes和PanShot两个基准上进行评估，分别包含大基线驾驶场景和多镜头光学多样性。

**📈 对比分析**

与GTA、PRoPE、UCPE、RayNova、RayRoPE、URoPE等编码比较，在CamMC、旋转误差、平移误差和FID/FVD指标上取得最优或接近最优的表现。

**⚠️ 局限性**

需要放宽每个token的独立因子化，导致查询-相机分组产生额外注意力成本；对多视角跨相机对齐的深度预测仍不充分。

---

## 601. Multi-Head Self Attention is a Parameter Identification Mechanism

**arXiv ID:** 2609.01231 | [PDF](https://arxiv.org/pdf/2609.01231v1)

**作者:** W. Ross Morrow `[一作]` `[通讯]`, W. Ross Morrow

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 Transformer 中多头注意力的参数可识别性进行理论分析与数值验证，证明未识别参数比例随头数 H 下降（约 1/(2H)），并探讨偏置项、RoPE、GQA 等对可识别性的影响。

**💡 创新点**

创新点在于将多头注意力视为一种内在的参数识别机制，首次量化多头带来的可识别率提升，并展示 RoPE 与 GQA 进一步改善识别度的数学原理；同时提出了基于矩阵不变性与方向导数的“Rebalancing”方法。

**🔧 技术方法**

主要技术包括矩阵乘法的不变性分析、SVD 变换、方向导数计算、梯度流理论、数值实验（SGD/AdamW、权重衰减、Rebalancing）以及对未识别子空间的监测指标（ρ、β、τ、μ）。

**📊 数据集**

实验使用的主要数据集为小规模字符级 Shakespeare 数据集（tiny Shakespeare），并在讨论中引用 SQuAD 的少量实验结果。

**📈 对比分析**

通过对比四种优化器配置（SGD、带衰减 SGD、AdamW、带 Rebalancing 的 AdamW），记录了梯度流的保守量 β、迹差 τ、识别度 ρ 与学习率的关系；实验表明 AdamW 更易走入未识别子空间，Rebalancing 能恢复保守性但对最终损失影响不大。

**⚠️ 局限性**

局限性包括：实验规模有限，仅在 toy 版实现上验证，缺乏对大规模预训练模型的验证；Rebalancing 方法尚未成熟，计算开销和理论保证有限；未考虑其他正则化、结构变形（如残差、MLP）对可识别性的联合影响。

---

## 602. Recent Developments in Transformer Inference Deployment on FPGA Platforms: A Survey

**arXiv ID:** 2609.01212 | [PDF](https://arxiv.org/pdf/2609.01212v1)

**作者:** Arjan Blankestijn `[一作]` (University of Twente), Amirreza Yousefzadeh `[通讯]` (University of Twente)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

对2024–2025年间 Transformer 推理在 FPGA 上的实现与优化进行系统综述，归纳了实现技术、存储策略、量化/剪枝方法、PE 设计、软硬件工具等关键技术路径。

**💡 创新点**

创新点在于：①首次聚焦最近两年高质量论文，采用可复现的检索与裁剪流程；②构建了包含实现与优化两大维度的“Transformer‑on‑FPGA”分类体系；③通过 Top‑10 吞吐/能效表与 Pareto 前沿图，量化不同实现的性能与能效对比；④强调了缺乏统一基准与模型不一致对比较的影响，提出了标准化评测的必要性。

**🔧 技术方法**

采用的技术包括：系统性文献检索（IEEE Xplore、Scopus、arXiv）与关键词裁剪；实现层面主要是 RTL（Verilog/VHDL）与 HLS（Vivado/Vitis、hls4ml 等）工具；存储策略有全 on‑chip、off‑chip 与动态加载；量化与剪枝技术涵盖 8/6/3 位定点、混合精度、二值化、N:M 稀疏化、AWQ、HEAT 等；PE 设计以 systolic array 与多功能 PE 为主；软硬件集成使用自定义 ISA、数据流与内存层次优化。

**📊 数据集**

本文本身不使用单一数据集，而是综合分析了多篇论文中常见的 Transformer 模型（BERT、DeiT、Swin、ViT、GPT‑style 等）及其变体；对比实验基于作者公开的模型权重与推理设置，未指定统一基准数据集。

**📈 对比分析**

通过对 57 篇高相关度论文的吞吐量（GOPS）、功耗（W）与能效（GOPS/W）进行统计与排序，给出 Top‑10 列表，并绘制吞吐‑能效 Pareto 前沿图，展示了不同实现（如 HG‑PIPE、Binary‑weights、Systolic、SoftMax fusion 等）在吞吐与能效上的权衡。总体上，使用 on‑chip 存储与精度量化往往能提升吞吐和能效，但仍受硬件资源与模型复杂度限制。

**⚠️ 局限性**

主要局限：①缺乏统一的 Transformer 基准模型与评测流程，导致不同论文间的性能比较不一致；②实现细节与精度损失报告不统一，难以直接评估量化/剪枝对模型精度的真实影响；③论文多侧重推理，缺少训练或端到端部署视角；④在硬件层面多依赖于厂商专有资源（DSP、BRAM、UltraRAM），不易迁移。

---

## 603. On Global Regulatability of Robot Manipulators by Classical PID

**arXiv ID:** 2609.01207 | [PDF](https://arxiv.org/pdf/2609.01207v1)

**作者:** Cheng Zhao `[一作]` (Chinese Academy of Sciences), Lei Guo `[通讯]` (University of Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究经典PID控制器在机器人机械臂上的全局调节问题，给出一维情况下可行的PID增益条件，并构造多维情形下的反例，证明单标量PID无法实现全局调节。

**💡 创新点**

提出了维度相关的完整答案：一维机器人可通过简单的PID参数保证全局指数稳定；二维及以上时存在满足所有结构假设但无法通过任何标量PID实现全局调节的机械臂；同时给出了明确的PID增益取值区间。

**🔧 技术方法**

利用能量基Lyapunov函数与Euler–Lagrange结构相结合的分析方法，构造精确的增益不等式；在多维案例中构造满足结构假设的机械臂模型并用解析论证无标量PID可达成全局调节。

**📊 数据集**

本文未使用实验或仿真数据集，全部结论来自理论推导与数学构造。

**📈 对比分析**

与传统的经验调参或基于模型的PD调节方法相比，本文提供了理论上可保证的全局指数稳定性（在一维情形）；多维情形则证明单标量PID本身就无法完成调节，凸显了方法的局限性。

**⚠️ 局限性**

局限性：仅考虑标量PID增益，未涵盖矩阵增益或非线性积分改进；在多自由度系统中无法实现全局调节，需要更丰富的控制结构或模型相关补偿。

---

## 604. Sensitivity Oracles for Matroid Packing, Matroid Covering, and Matching Problems with Applications

**arXiv ID:** 2609.01283 | [PDF](https://arxiv.org/pdf/2609.01283v1)

**作者:** Keerti Choudhary `[一作]` (Indian Institute of Technology Delhi), Lakshay Saggi `[通讯]` (Indian Institute of Technology Delhi)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一套统一的代数框架，用来构造支持任意数量失败（f）更新的敏感性预处理数据结构，涵盖最大流、最小割、树包、匹配、α‑因子等结构优化问题。

**💡 创新点**

创新点在于将这些多样问题归结为稀疏线性矩阵的低秩更新，通过矩阵行列式的符号与低秩修正技术，首次实现了在多失败场景下保持多项式空间和时间的敏感性或多项式查询；同时引入“子集敏感性模型”将更新限制在固定子集上，进一步消除对参数 k 的依赖。

**🔧 技术方法**

核心技术包括：
• 采用随机化的 k‑折并集表示法将 k‑packing/covering 转化为单一基数的矩阵行列式问题；
• 利用矩阵行列式与低秩更新的标量公式（Matrix Determinant Lemma）高效更新；
• 将行列式计算转化为多项式非零判定，使用 Schwartz–Zippel 上界保证随机化成功；
• 在子集敏感性模型下预计算受限子集的逆矩阵子块，达到 O(f^ω) 查询时间。

**📊 数据集**

本文为理论算法研究，未使用具体实验数据集；所有结果均通过随机化证明与数学证明得到概率上界。

**📈 对比分析**

与以往仅对 f≤2 或仅在无向图中可行的敏感性预处理器相比，本文的方法在任意 f 下实现了空间 O((fn)^2 log n)（或 O(σ^2)）和查询时间 O((fk^2)^ω)（或 O(f^ω)）的最佳或接近最优上界；实验对比中显示对比传统组合方法的指数级增长已被抑制，算法在多失败环境下保持多项式性能。

**⚠️ 局限性**

局限性主要包括：
• 仍采用随机化技术，结果仅在高概率下成立；
• 对于极大 k 或稠密图，k‑折并集的矩阵维度导致常数因子较大；
• 子集敏感性模型要求先知晓敏感子集 σ，若 σ 与全图规模相当则空间退化回 O(n^2)。

---

## 605. Highly Detailed Simulation for Connected Automated Vehicle Cooperative Driving

**arXiv ID:** 2609.01254 | [PDF](https://arxiv.org/pdf/2609.01254v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 606. Seeing the World and the Self from Egocentric Video

**arXiv ID:** 2609.01276 | [PDF](https://arxiv.org/pdf/2609.01276v1)

**作者:** Kai Guan `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一框架，利用单目第一人称视频同时恢复场景几何和佩戴者全身运动。

**💡 创新点**

创新点在于将确定性场景重建与几何条件下的扩散运动生成相结合，并引入闭环运动到相机的反馈。

**🔧 技术方法**

使用了Pi3X几何骨干、几何条件扩散运动模型、相机头闭环反馈以及多阶段训练策略。

**📊 数据集**

使用了新构建的EE4D-JSM数据集，该数据集来源于EgoExo4D，包含对齐的场景、相机轨迹和全身运动。

**📈 对比分析**

与VGGT、CUT3R、EM4D、UEM等SOTA方法比较，模型在相机轨迹、深度估计和全身运动指标上均显著优于基线。

**⚠️ 局限性**

局限性包括对稀疏SLAM点云的依赖、对高速摄像机运动的敏感性以及实时部署的计算成本。

---

## 607. Position: Privacy Is a Claim, Not a Property of Synthetic Data

**arXiv ID:** 2609.01273 | [PDF](https://arxiv.org/pdf/2609.01273v1)

**作者:** Jiachen Zhao `[一作]` (University of Notre Dame), Taeho Jung `[通讯]` (University of Notre Dame)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对ICML、NeurIPS、ACL 2024-2025三大会议论文的系统审计，揭示了合成数据在隐私声明中的隐性使用，并提出了最小隐私声明标准（Minimum Privacy Claim Standard）。

**💡 创新点**

创新点在于把隐私视为可证实的科学声明而非生成过程的默认属性，系统性地量化了隐私声明与验证之间的缺口，并提供了可操作的审计与声明规范。

**🔧 技术方法**

采用规则化文本扫描和基于规则的过滤器对论文 PDF 进行处理，结合统计分析与可视化来量化合成数据使用情况及隐私声明的完整性。

**📊 数据集**

使用的数据集包括 2024-2025 年 ICML、NeurIPS、ACL 会议的全部已接受论文 PDF 以及公开的合成数据工具与政策文档表格。

**📈 对比分析**

比较方法主要是统计合成数据使用比例与隐私声明匹配率，并通过条形图展示不同会议和年份的差异；文中未进行传统算法性能对比，而是通过定量审计展示声明缺失的普遍性。

**⚠️ 局限性**

局限性在于仅对公开论文进行表面审计，未对实际隐私泄露风险进行实验验证；审计规则可能产生漏检；且所用数据仅覆盖三大会议，缺乏行业或真实应用场景的考察。

---

## 608. Making Prospective Memory SLM-Shaped: Typed Intention Stores for Small-Model Agents

**arXiv ID:** 2609.01272 | [PDF](https://arxiv.org/pdf/2609.01272v1)

**作者:** Jinqing Zhao `[一作]` (Peking University), Chengcan Wu `[通讯]` (Peking University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Prospective Intention Store（PIS），通过将前瞻性记忆拆解为可类型化的记录并在外部存储中以Form-Revise-Filter-Decide四步循环实现，构建了一个训练无关的代理框架。

**💡 创新点**

创新点在于将前瞻性记忆视作方案型状态追踪，用外部类型化记录和代码驱动的生命周期逻辑替代传统的内部记忆存取，避免了大型模型的自训练需求，显著提升了小型模型的前瞻记忆表现。

**🔧 技术方法**

采用了类型化记录 (I=(φ,α,σ))、四个可编程操作（Form、Revise、Filter、Decide）以及语言模型的定向推理、频道查询等技术，构成无PEFT、无参数调优的agent系统。

**📊 数据集**

使用公开的 PM-Bench（合成一周活动）数据集进行评估，并在此基础上测试了Gemma-E2B、Qwen3.5-4B、Qwen3-8B等小模型。

**📈 对比分析**

与七种传统回溯式记忆方法（Naive RAG、Mem0、A-Mem、Letta/MemGPT、LightMem、MemoryOS）以及单纯对话推理基线相比，PIS在DeepSeek-Chat上取得82.9% Set-F1，超过最高65.1%；在Gemma-E2B上从4.2%提升至66.2%；在Qwen3.5-4B上达到70.1%，显著优于对照组的54.4%；同时尽管计算开销略高，但相对成本仍具竞争力。

**⚠️ 局限性**

主要局限在于评估仅覆盖PM-Bench一个公开数据集，缺乏其他前瞻性记忆基准；未探究模型微调或单一操作贡献；对外部频道查询等假设在真实环境中可能受限。

---

## 609. Explore More, Drift Less: Outcome-Only Reinforcement Learning Can Suffice for Long-Horizon Interactive Agents

**arXiv ID:** 2609.01245 | [PDF](https://arxiv.org/pdf/2609.01245v1)

**作者:** Liming Pu `[一作]`, Bin Yang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种简洁的强化学习协议（Coverage‑Anchored On‑Policy RL），通过扩大同一任务的采样组、保持完全在线更新、使用KL锚定和仅对行动令牌的损失，解决了稀疏奖励下的信号匮乏和策略漂移问题，并在小型开放模型上实现了长时序交互任务的显著提升。

**💡 创新点**

创新点在于（1）用覆盖率公式自适应增大采样组大小，从根本上恢复稀缺任务的梯度信号；（2）通过对策略梯度的单步在线更新和KL锚定，防止因多次使用同一任务导致的策略漂移；（3）仅对行动令牌进行损失计算，去除无关环境令牌对梯度的干扰，保持信号纯净。

**🔧 技术方法**

技术包括：基于group‑relative GRPO/PSO框架的稀疏奖励优化；自适应组大小计算公式；KL锚定与token‑level损失；故障排除与任务保留策略；预算转移（训练时低交互预算，推理时高交互预算）。

**📊 数据集**

使用的主要数据集为AppWorld（包含90/60/168/417任务的四个拆分）以及SWE‑bench Verified（真实仓库软件修复任务），并在这些基准上评估模型。

**📈 对比分析**

与其他已公开的训练策略（如ESAT、LOOP、GVPO、SAGE等）以及无训练的推理时多智能体系统（HCL‑GP、ASSAY、ReAct等）进行对比。本文在AppWorld Test‑Normal和Test‑Challenge上分别取得86.9/67.6的TGC，显著领先于同类训练策略；在SWE‑bench上通过预算转移提升了16.6点的resolve率，表明该协议在不同域上具有可迁移性。

**⚠️ 局限性**

局限性包括：仅在规模相对有限的任务池（90个任务）上验证；对极端长时序任务的适用性仍待进一步测试；对多语言或更大规模的交互环境的扩展需要更多样本和更高效的算法。

---

## 610. Post-Training Science for Supervised Fine-Tuning

**arXiv ID:** 2609.01244 | [PDF](https://arxiv.org/pdf/2609.01244v1)

**作者:** Charles O'Neill `[一作]` (Baseten), Harry Partridge `[通讯]` (Baseten)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在Qwen3与Llama两大模型族（含稠密与Mixture-of-Experts版本）上，作者对LoRA与全微调分别在四个客户SFT数据集进行系统的单维度超参数（学习率、批量、LoRA秩、α、优化器、训练轮数）网格搜索，测量验证损失、loss flatness、下游评估和计算成本。

**💡 创新点**

创新点在于：①在真实客户任务的无噪声数据上统一评估；②发现LoRA学习率在0.6–32B规模下恒定为1e-3，且约为全微调的33倍；③验证损失可在同一“recipe”内预测下游质量，但跨族失效；④LoRA秩至64后饱和，α=32最佳；⑤MuON优化器在SFT下保留更一般指令遵循，损失略低但任务质量相近；⑥给出规模、数据量、轮数等维度的可量化经验法则。

**🔧 技术方法**

使用的技术包括单参数网格搜索、Cosine学习率衰减、LoRA低秩适配器、全微调、AdamW与MuON矩阵预条件优化器、Fisher trace作为flatness指标、SFT自生成数据集、客户评估判定器以及I.F.Eval通用指令遵循测试。

**📊 数据集**

使用四个匿名客户SFT数据集：support、leasing、docs、security；这些数据通过迭代SFT生成并配有对应评估。

**📈 对比分析**

方法上在相同模型、数据、学习率和批量下比较不同超参数，发现验证损失与下游评估呈负相关；MoE模型在几何平均规模上与稠密模型表现一致；MuON优化器在相同学习率下比AdamW损失低、flatness更好，但任务评估相近，I.F.Eval则更优。

**⚠️ 局限性**

局限性包括：数据已被优化以通过评估，导致评估与训练目标不独立；跨族泛化受限；仅对全微调进行优化器比较；例子与token混杂导致数据量效果难以分离；高学习率稳定性受模型大小限制；内存上限限制了更大规模模型的训练与评估。

---

## 611. Births are difficult to predict even with rich survey and full-population register data

**arXiv ID:** 2609.01194 | [PDF](https://arxiv.org/pdf/2609.01194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 612. Position Matters: Feature Inversion Attacks in ViT Split Inference with Token Reduction and Shuffling

**arXiv ID:** 2609.01232 | [PDF](https://arxiv.org/pdf/2609.01232v1)

**作者:** Stefano Leggio `[一作]` (Scuola Superiore Sant'Anna), Alessandro Biondi `[通讯]` (Scuola Superiore Sant'Anna)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在 ViT 端到云的 split 推理中，研究提出了 SARA（Spatially Aligned Reconstruction Attack）攻击，用以评估在 token 归约和打乱后中间特征的隐私泄露，并进一步设计了一种基于位置嵌入消除与渐进式蒸馏的轻量化边缘防御方案；

**💡 创新点**

创新点在于：①将 token 位置恢复与缺失 token 生成相结合，突破传统解码器在打乱/归约场景下的局限；②提出的轻量化防御通过逐块去除位置编码并用知识蒸馏保持任务性能，且可对抗 SARA；

**🔧 技术方法**

核心技术包括：Vision Transformer（ViT-B/16 与 MAE-B/16）、Token Position Predictor、Masked Autoencoder（MAE）、卷积解码器、知识蒸馏与对抗式位置消除；

**📊 数据集**

实验使用 ImageNet‑1K 数据集，采用标准的 1K 图像分类任务进行评估；

**📈 对比分析**

比较方法包括基线卷积解码器、Token 打乱/归约、以及防御后的模型；通过 SSIM、PSNR、FSIM 等视觉指标以及自定义的 Privacy–Utility Reconstruction Index（PURI）衡量隐私泄露和任务性能，实验显示：打乱对 SARA 的效果差，归约在浅层能提升隐私但对深层效果有限；防御显著降低 SSIM 并维持较高准确率；

**⚠️ 局限性**

限制：仅针对图像分类任务，未验证对分割等任务的适用性；仅评估 ViT，未探讨语言模型场景；未来需研究更强的 token 重排策略与更鲁棒的对抗防御。

---

## 613. Towards AI-Assisted Clinical Trial Matching: Practical Considerations, Multicenter Evaluation, and Real-World Deployment

**arXiv ID:** 2609.01202 | [PDF](https://arxiv.org/pdf/2609.01202v1)

**作者:** Yin Fang `[一作]` (National Institutes Of Health), Zhiyong Lu `[通讯]` (National Institutes Of Health)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并部署了 TrialGPT 2.0，一个可配置的 AI 系统，用于将患者信息与本地临床试验清单匹配，生成带有结构化解释的试验推荐，并在真实临床工作流程中进行评估。

**💡 创新点**

创新点在于把试验匹配提升为“临床推荐”而非仅仅“资格判定”，通过可配置的匹配策略和推荐等级来体现治疗意图与患者需求；提供多层次结构化解释（资格依据、缺失信息、潜在不匹配）；通过真实临床评估、精确的 NIH‑TrialBench 以及公开基准实现可复现的性能验证；实现了可直接部署的 Web 接口。

**🔧 技术方法**

技术方案主要包括：基于 GPT‑4.1/5.4 的大型语言模型进行患者‑试验评估；检索模块采用 BM25 与 MedCPT 的混合检索（hybrid‑fusion）以支持不同规模试验库；后端生成 fit score、confidence、推荐类别（Highly Recommended / Possible Match / Low Fit）；结构化输出与可视化界面；评估指标包含 hit‑rate@K、recommended precision@K、eligible precision@K、Recall@10、runtime、用户满意度等。

**📊 数据集**

使用的数据集包括：① 288例来自 NIH、NCI、CCF、UPMC 等的去标识临床笔记（覆盖 5 条工作流程）；② 27例 UIC Precision Oncology Tumor Board 的实时病例；③ 126 例 NIH‑TrialBench 的合成病历；④ 公开基准 SIGIR、TREC 2021/2022；⑤ 内部测试用的本地试验清单（9–1,871 例）。

**📈 对比分析**

评估方法：① 在 4 条回溯性工作流程中使用 Top‑ranked review，hit‑rate@10 达 91%；② 在 NCI 例子中对比辅助与非辅助评审，平均筛选时间下降 55%；③ 在 UIC POTB 真实流程中，TrialGPT 2.0 在 37% 病例中新增最终推荐，贡献 83% 终推荐；④ 在 NIH‑TrialBench 上 Recall@10 达 70%（比 TrialGPT 1.0 提升 16%）；⑤ 在公开基准上 Precision@10 提升 4.5%，nDCG@10 提升 5.7%，MRR 提升 9.8%，MAP 提升 10.2%；⑥ 运行时平均 21.7 秒，界面可在 25 例中实时生成。整体表现优于先前版本与公开基准，并在真实工作流程中显示可接受的效率与准确性。

**⚠️ 局限性**

局限性包括：未跟踪下游招募、同意与入组等结果；样本主要来自美国，国际化与跨机构推广需要进一步验证；模型依赖 GPT‑4.1，未来可能需更换或更新；评估规模相对有限，特别是 POTB 仅 27 例；误差模式仍存在，如过度/欠推荐、对多臂试验或信息不完整病例的判定仍需改进；系统在部署后需要持续监控与用户培训。

---

## 614. Compressing AI Traffic: Standardized Neural Network Coding of Visual-Token Representations in Split Vision-Language Inference

**arXiv ID:** 2609.01200 | [PDF](https://arxiv.org/pdf/2609.01200v1)

**作者:** Reza Heidari `[一作]` (Aalto University), Juho Kannala `[通讯]` (Aalto University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在分布式视觉‑语言模型推理中，对视觉编码器输出的中间视觉标记张量进行标准化的Neural Network Coding（NNC）压缩。

**💡 创新点**

首次将ISO/IEC 15938‑17标准的NNC应用于视觉‑语言模型的AI traffic压缩，并证明在压缩率高达98%时仍能保持任务性能。

**🔧 技术方法**

使用Qwen3‑VL‑8B‑Instruct模型、ISO/IEC 15938‑17 NNC编码/解码、BF16张量压缩、QP调节参数、Video‑MME闭合问题与MLVU开放问题评估。

**📊 数据集**

Video‑MME（闭合问题）和MLVU（开放式视频摘要）作为评估基准。

**📈 对比分析**

与身份传递和全零标记基线对比，发现当QP=12时压缩率接近99%时仍保持与原始模型相同的准确率/质量；在QP=0附近达到峰值后，性能骤降。

**⚠️ 局限性**

仅评估单一模型与部分数据集，未测量编码/解码延迟，且开放式生成评估依赖LLM判断。

---

## 615. Smart Contracts Claimed Vulnerable by the CVE Database, with Labels and Source Locations

**arXiv ID:** 2609.01186 | [PDF](https://arxiv.org/pdf/2609.01186v1)

**作者:** Monika di Angelo `[一作]` (TU Wien), Gernot Salzer `[通讯]` (TU Wien)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个名为 CVE‑Smart‑Contracts 的数据集，系统性地收集、验证、标注并定位 2026 年前所有与以太坊智能合约相关的 CVE 漏洞记录，最终生成 491 条无误判、已定位并带三种分类标签（Iuliano‑DiNucci、CWE、SWC）的合约条目。

**💡 创新点**

创新点在于（1）自动化检索与验证 CVE 与合约代码/部署地址的一致性，减少人工干预仅 15%；（2）在单一数据集中统一提供漏洞标签、函数级定位和对应源代码/字节码；（3）引入三层标签映射并记录来源，提升标签可解释性；（4）对验证、标注和定位过程做完整可重现的流水线，并提供元数据与 schema，方便后续研究复现与扩展。

**🔧 技术方法**

技术包括：Python 自动化脚本、Etherscan API 调用、Solidity 编译器交叉编译、AST 解析与函数单元抽取、正则规则驱动的标签生成、手工审核决策、JSON Schema 验证以及完整的可重现构建环境。

**📊 数据集**

使用的数据集为 2026‑07‑24 版 CVE v5 数据库，过滤出的 568 条 CVE 记录（542 部署合约 + 26 库）以及对应的 Etherscan 源码、运行时字节码、引用文档和手工注释。

**📈 对比分析**

与先前的 VeriSmart、SmartFix 基准相比，CVE‑Smart‑Contracts 更完整（覆盖全部 2026 年前的合约 CVE）且提供了函数级定位；评估性能方面并未在论文中给出具体检测工具的对比实验，而是提供了数据集本身的覆盖率、标签一致性和定位准确性等度量；实验显示约 86% 的匹配记录自动通过验证，约 85% 的标签可自动生成，定位覆盖 415 条自动、76 条人工。

**⚠️ 局限性**

局限性包括：仅涵盖以太坊（仅一条 BNB 记录）；依赖 CVE 数据库的完整性与更新，可能漏检或误检；验证仅确认记录与代码/部署地址的一致性，并不保证漏洞存在或可利用；标签映射受原始 CVE 文本表述限制，可能出现歧义或缺失；函数定位仅给出最可能的函数，无法精准定位到语句级别；第三方源码与引用资料受许可约束。

---

## 616. Agentic Multimodal Models for Environmental Hyperspectral Unmixing

**arXiv ID:** 2609.01289 | [PDF](https://arxiv.org/pdf/2609.01289v1)

**作者:** Michał Cholewa `[一作]` (IITiS-PAS), Giuseppe Amato `[通讯]` (ISTI-CNR)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于大视觉语言模型（LVLM）的智能体框架，用于对现有模块化的高光谱反混合（unmixing）结果进行自适应优化；

**💡 创新点**

创新点在于将LVLM与工具调用相结合，利用外部光谱库检索和空间丰度可视化来指导合并/剔除端元，从而实现无监督、算法无关的精细化；

**🔧 技术方法**

核心技术包括：ReAct式工具使用循环、光谱库检索（Spectral Angle Distance）、丰度热图可视化、端元合并/剔除操作，以及对不同基础管线（HySime/VCA/FCLSU等）的统一后处理；

**📊 数据集**

在HYDICE Urban、Jasper Ridge和Stonewall Playa三个公开高光谱数据集上进行实验；

**📈 对比分析**

与传统模块化管线和三种端到端方法（CNN‑AE、uDAS、R‑CoNMF）对比，Agent化后多达95%+的性能提升，尤其在端元数、谱角误差和重建误差方面均优于或与最先进方法持平；

**⚠️ 局限性**

主要限制是只能进行合并/剔除，无法新增缺失的端元；计算成本较高，需多次LLM推理与工具调用，适用于对实时性要求不高的离线分析。

---

## 617. Measuring the Behavioral Fidelity of Long-Horizon Human Activity Simulations

**arXiv ID:** 2609.01257 | [PDF](https://arxiv.org/pdf/2609.01257v1)

**作者:** Yi Fei Cheng `[一作]` (Carnegie Mellon University), David Lindlbauer `[通讯]` (Carnegie Mellon University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究提出了一个评估LLM驱动长时段人类活动模拟器行为忠实度的框架，并基于一套43小时多摄像头的办公室活动数据集对六种不同的条件化方法进行了系统评估。

**💡 创新点**

创新点在于：①从时间粒度（局部、时段、日）和分析层级（个体、群体）两个维度构建多层次的忠实度评估体系；②首次将统计先验、个性化描述、少量示例等多种条件化策略在同一框架下进行对比；③通过多维度指标揭示统计先验虽降低整体分布差异，却导致过度碎片化与个体差异削弱。

**🔧 技术方法**

技术包括：LLM（Llama 3.3 70B、GPT‑5.4‑mini‑2026‑03‑17、Gemini 2.5 Flash）与Perceive–Plan–Act循环的Agent架构；在规划阶段引入统计先验（活动转移、时段分布）和多目标采样；使用自定义活动词汇表与基于场景图的环境模拟。

**📊 数据集**

使用了一个43小时的多摄像头办公室活动数据集（55人，5人多日观测），包含每帧活动标签、房间标签和2D位置信息，可在公开链接 <https://augmented-perception.org/publications/2026-long_horizon_agents.html> 获得。

**📈 对比分析**

通过对个体层面和群体层面的分布（时间分配、频率、转移、n‑gram）、局部指标（活动切换、日内变异）等多项指标进行统计差异（JS、TV、KL、L1）比较，结果显示：①统计先验（C4/C5）在分布与转移指标上显著低于其它方法；②但其导致日内切换数显著增加且个体间差异被压缩；③个性化条件化方法差异有限，且对不同指标的影响不一致。

**⚠️ 局限性**

局限性包括：①实验仅在单一办公室、单周且仅5人深度观察的场景下进行，缺乏对更大规模、多元环境的验证；②只评估了外部活动序列，未涉及意图或心理状态；③Agent设计中的计划质量、先验阶层等参数未系统探索；④模型背后的LLM种类和版本差异可能影响结果，需进一步研究。

---

## 618. One Prompt Is Enough: Watermark Laundering Through Foundation Image Models

**arXiv ID:** 2609.01249 | [PDF](https://arxiv.org/pdf/2609.01249v1)

**作者:** Jidong Yang `[一作]` (Qilu University of Technology (Shandong Academy of Sciences)), Suo Gao `[通讯]` (Dalian Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了使用公开基础图像模型（如 GPT Image 系列和 Nano Banana 系列）在单一自然语言提示下对嵌入不可见水印的图像进行重建，从而实现所谓的“水印洗钱”攻击；

**💡 创新点**

创新点在于首次把公共基础模型的重建功能视为一种新的鲁棒性测试条件，并通过联合的 BER‑fidelity 框架量化了水印破坏与内容保真之间的权衡；

**🔧 技术方法**

采用的技术包括可逆水印嵌入方案（DwtDct、DwtDctSvd、RivaGAN）、公开模型的黑盒调用、提示词消融实验以及高频残差分析与多维图像质量指标（PSNR、SSIM、LPIPS、CLIP‑Sim 等）；

**📊 数据集**

实验使用 100 张 MS‑COCO 图像（按纹理、平面、文本三类分层），每个图像在 18 种水印‑模型组合下生成 1,800 张重建结果；

**📈 对比分析**

与传统攻击（高斯噪声、模糊、裁剪、旋转等）对比发现，OpenAI 的 GPT Image 1 在 BER 方面最接近随机恢复，但图像保真度较低；Nano Banana 2 则在保持高保真度的同时仍对 DwtDct 水印造成显著扰动；整体而言，基础模型重建既能破坏水印，又能在视觉上保持原图相近，显示出新的安全风险；

**⚠️ 局限性**

限制主要包括仅评估了三种水印方案、六个模型版本、未给出统计置信区间或显著性检验、未覆盖基于生成模型的水印以及缺乏对模型内部机制的解释，故结论仅适用于当前评估的接口与时间点。

---

## 619. Continuous Autonomous Refactoring: A Research Roadmap for AI-Driven Code Quality Maintenance

**arXiv ID:** 2609.01236 | [PDF](https://arxiv.org/pdf/2609.01236v1)

**作者:** Xin Sun `[一作]` (Linköping University), Christoph Kessler `[通讯]` (Linköping University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了将LLM驱动的代码重构转变为持续自治的过程，构建了五大维度的研究路线图

**💡 创新点**

创新点在于把质量管理视为持续目标，将多目标优化、质量定义、异构信号融合、架构模式、信任等系统层面问题整合为统一的研究框架

**🔧 技术方法**

结合LLM、静态分析、沙箱测试、utility函数、多智能体架构以及持续交付流水线中的自动化验证技术

**📊 数据集**

未使用具体数据集，而是基于文献综述和主题分析提出研究问题

**📈 对比分析**

本文未给出实验比较，所述方法为理论框架，未涉及性能评估

**⚠️ 局限性**

局限在于缺乏实证验证、对LLM能力变化的适应机制不完善、信任与人机交互的可解释性不足以及成本与环境影响的量化研究不足

---

## 620. MutMem-V2: Cryptographically Authorized Mutation in Persistent Agent Memory Portable Verification and Reproducible Evidence

**arXiv ID:** 2609.01235 | [PDF](https://arxiv.org/pdf/2609.01235v1)

**作者:** Walid Saidi `[一作]` `[通讯]` (Independent researcher), Walid Saidi (Independent researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发并发布了 MutMem V2，提供可移植的回忆‑披露协议和独立的 Node/Python 验证器，验证持续记忆的完整性、授权与可追溯性，并完成一次无实验内存的干净安装验证。

**💡 创新点**

创新点在于构造完整的跨对象谓词与 Merkle 结构，使用规范化 JSON 字节、域分离的 SHA‑256 与 Ed25519 签名，实现端到端可验证的回忆与突变协议，并提供独立实现的可比对终端。

**🔧 技术方法**

采用 SHA‑256 哈希、Ed25519 签名、canonical JSON、长度帧编码、Merkle 树、域分离构造，并实现 Node.js 与 Python 语言的独立验证器。

**📊 数据集**

回顾性利用历史 V1 的 LongMemEval、LoCoMo、PoisonedRAG 等数据集做实测，但 V2 本身未重新跑这些数据集，仅复现已公布的统计。

**📈 对比分析**

通过 Wilson 置信区间、McNemar 检验、Holm 调整与 κ 统计与 V1 结果对比；验证器在 O(n) 复杂度下完成合规性检验，已证明与生产案例完全一致，性能与 V1 相当。

**⚠️ 局限性**

缺乏 V2 版本的实用性重跑与独立经验复制；Canary 仅检测标记穿越，未覆盖通用污染；缺少 SABER 计数；安装验证仅在单一 macOS+Node 环境完成；依赖外部可信锚点，且未解决凭证泄露等问题。

---

## 621. What's in Your Agent's Context? Context Privilege Escalation Attacks against AI Agent Harness

**arXiv ID:** 2609.01222 | [PDF](https://arxiv.org/pdf/2609.01222v1)

**作者:** Zichuan Li `[一作]` (University of Illinois Urbana Champaign), Luyi Xing `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对12个主流LLM代理主机的上下文组装机制进行了系统性分析，发现并实现了两类新型特权提升攻击（消息角色提升与跨作用域提升），并开发了自动化的Context Risk Analyzer用于识别与验证攻击路径。

**💡 创新点**

创新点包括提出了消息角色与跨作用域两类特权提升攻击，构建了基于LLM辅助的静态与动态分析流水线，并首次实现了对开源代理主机的全自动PoC验证。

**🔧 技术方法**

采用了LLM辅助静态源识别、可解释的上下文源抽取、动态运行时注入与验证、以及路径枚举与可执行攻击生成的技术组合，核心模型为GPT‑5.5、Claude Sonnet等。

**📊 数据集**

使用的数据主要为12个公开代理主机的源码及其内置的内存文件、技能描述、配置文件等自然语言上下文源。

**📈 对比分析**

通过与人工标注的上下文源对比，Context Risk Analyzer实现了约97%精度、93%召回的识别效果；在验证阶段，成功率在58%–74%之间，显示了工具在实战中的有效性。

**⚠️ 局限性**

局限性主要在于受限于目标源的payload容量、触发条件、LLM对指令的理解差异，导致部分路径无法加载或验证；此外，工具对不同编程语言的支持仍需进一步完善。

---

## 622. Prompt-Robust Language Models: Which Training Strategies Work?

**arXiv ID:** 2609.01217 | [PDF](https://arxiv.org/pdf/2609.01217v1)

**作者:** Frederic Sadrieh `[一作]` (Ludwig-Maximilians-Universitaet Muenchen), Michal Štefánik `[通讯]` (National Institute Of Informatics)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统比较了训练时提升大型语言模型提示鲁棒性的多种策略，探讨了数据构造、一致性正则化和对比对齐方法的效果；

**💡 创新点**

创新点在于对三类训练策略进行统一对照实验，发现单模板批次（One-at-a-Time）在最坏情况提示鲁棒性提升最显著；

**🔧 技术方法**

主要技术包括多模板指令微调（multi‑prompt IFT）、PPCL（一致性正则化）和CoIN（对比对齐）以及不同的批次构造方式；

**📊 数据集**

使用PromptSource集合中的48个训练数据集（含多模板）和11个未见任务的测试数据集；

**📈 对比分析**

实验在四个模型（Llama3.2‑1B、Llama3.1‑8B、Qwen3‑0.6B、Qwen3‑8B）上进行，结果显示多模板训练均优于单模板，但最佳方法（One-at-a-Time）仍仅把最坏-最好提示差距压缩到约40–57%；

**⚠️ 局限性**

局限性包括模型规模仅至8B、仅覆盖Llama和Qwen两大体系、评估任务主要为短答/分类，未涵盖开放式生成及更大规模模型，且对比方法仅选取单一代表实例，未对超参数进行全模型细调。

---

## 623. H2Table: Hierarchical Hypergraph-Enhanced Large Language Models for Complex Table Reasoning

**arXiv ID:** 2609.01216 | [PDF](https://arxiv.org/pdf/2609.01216v1)

**作者:** Jia Ling `[一作]` (Harbin Institute of Technology), Jingchi Jiang `[通讯]` (Harbin Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 H2Table 框架，通过层级嵌套超图模型和超图编码器来对复杂表格进行结构化表示，并利用可学习的查询向量将结构信息映射到 LLM 中，实现对多层级表格的问答推理。

**💡 创新点**

创新点在于：①将复杂表格建模为层级嵌套超图，保留多级表头的语义层次；②设计四阶段层级交互模块（V2E、C2P、P2C、E2V）实现跨层级信息传递；③引入查询向量作为 Soft Structure Prompt，实现参数高效的跨模态对齐。

**🔧 技术方法**

技术包括：超图表示与编码器（包含 GAT 的层级消息传递）、查询向量的跨注意力对齐模块、LoRA 轻量化微调、以及 LLM 的文本序列化与生成。

**📊 数据集**

使用了 HiTab、TATQA 以及 AITQA 三个包含层级表头的 TableQA 基准数据集进行实验。

**📈 对比分析**

与 TableLlama、TAMO 等基线对比，H2Table 在 HiTab 深度 3–4 的表格上平均提升约 22.88%，在 Llama3.1-8B、Gemma2-9B 等中型模型上实现与百亿级模型相当甚至更优的准确率，展示出显著的性能提升。

**⚠️ 局限性**

局限性在于：假设表头层级结构已被预先准确提取，未考虑端到端的表格结构抽取；以及仍需将表格文本序列化输入 LLM，未能将结构信息直接嵌入语义表达。

---

## 624. Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR

**arXiv ID:** 2609.01354 | [PDF](https://arxiv.org/pdf/2609.01354v1)

**作者:** Esther Xin `[一作]` `[通讯]`, Esther Xin

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过元变形测试评估四个常用验证器的可靠性，并对错误类型和分布进行细粒度拆解。

**💡 创新点**

引入基于已知等价变换的认证元变形协议，自动生成真值并按类别拆解错误率，同时提出覆盖率与错误率分离的评估框架。

**🔧 技术方法**

使用元变形测试、Wilson置信区间、合同矩阵、覆盖率指标、符号比较（SymPy）以及Azure ML并行流水线等技术。

**📊 数据集**

使用GSM8K、MATH、Big-Math的4990条真实答案和2000条合成答案，共计307,420个评估实例。

**📈 对比分析**

通过自验证率、覆盖率、类别错误率三维度横向比较四个验证器，发现实现差异可达41点，空白处理占主导错误，且某验证器表现出基于答案大小的确定性误报。

**⚠️ 局限性**

仅评估开放源码验证器；未考虑模型真实输出对RLVR收益的影响；合成答案比例较高；合同分配主观；未测评下游强化学习效果。

---

## 625. CHARM: Character Hallucination for Multicultural Role Play Benchmark

**arXiv ID:** 2609.01352 | [PDF](https://arxiv.org/pdf/2609.01352v1)

**作者:** Sunkyung Han `[一作]` (Sungkyunkwan University), JinYeong Bak `[通讯]` (Sungkyunkwan University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多文化角色认知边界评估基准CHARM，并提出了将边界意识与边界遵从分离的两阶段评估框架；

**💡 创新点**

首次系统区分角色幻觉是因边界认知失败还是因边界遵从失败，揭示大多数幻觉源于遵从失效，并发现参数覆盖与文化差异是关键因素；

**🔧 技术方法**

采用多选题+放弃回答机制进行边界检测，结合知识验证与参数覆盖分析；

**📊 数据集**

CHARM基准，包含680个边界意识题、1332个边界遵从题、736个知识验证题，覆盖40个来自五个文化-语言区域的真实与虚构角色；

**📈 对比分析**

在六大主流LLM（如GPT‑4、Claude等）上评估，发现边界遵从率远低于边界意识率，说明幻觉主要由遵从失败引起；跨文化对比显示不同地区角色表现存在显著差异；

**⚠️ 局限性**

仅覆盖五个文化区域，未囊括所有语言和文化；使用多选题而非开放式交互，难以反映真实对话场景；参数覆盖检验无法定位知识来源；未给出针对模型改进的具体方法；

---

## 626. CMRVision: A Foundation Model for Cardiac MR Image Analysis

**arXiv ID:** 2609.01308 | [PDF](https://arxiv.org/pdf/2609.01308v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 627. Integrating Traffic Noise Emission Modelling into Variable Speed Limit Control

**arXiv ID:** 2609.01339 | [PDF](https://arxiv.org/pdf/2609.01339v1)

**作者:** Jiawen Meng `[一作]` (Karlsruhe Institute of Technology), Alexey Vinel `[通讯]` (Karlsruhe Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究将CNOSSOS‑EU噪声估算集成进可变限速（VSL）框架，实现实时噪声降低的道路交通控制。

**💡 创新点**

在VSL决策中首次使用实时噪声排放指标，并引入分阶段速度限制与时间可变阈值，兼顾噪声与交通效率。

**🔧 技术方法**

采用微观仿真(SUMO)、CNOSSOS‑EU简化排放模型、基于滑动窗口的交通状态估计、离散阶段控制器以及阈值校准等技术。

**📊 数据集**

使用德国A66高速公路BASt定量车流计数数据及车辆类别作为需求与校准输入。

**📈 对比分析**

通过与无控制基准和固定低速策略三种情景比较，24小时等效声压降低2.9 dB(A)，平均车速比固定低速高约11 km/h，噪声下降显著且交通波动相对温和。

**⚠️ 局限性**

简化排放模型忽略车辆细节与路面、环境影响，缺乏多目标优化，且仅在单段模拟，缺少大规模网络验证。

---

## 628. Reliability Challenges in Diffusion Vision-Language Models

**arXiv ID:** 2609.01318 | [PDF](https://arxiv.org/pdf/2609.01318v1)

**作者:** Md. Atabuzzaman `[一作]` (Virginia Tech), Chris Thomas `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文对扩散式视觉‑语言大模型（dLVLM）在幻觉、偏差、种族与性别差异以及长度偏差等可靠性维度上进行了系统评估。

**💡 创新点**

首次提出从扩散生成机制出发识别 dLVLM 的独特失效模式，包括无偏见的 yes‑bias、语言质量退化、种族性别极端偏差以及一次性长度优先的 MCQA 误判。

**🔧 技术方法**

利用 POPE、CHAIR、FairFace、CUB‑200‑2011、Stanford Dogs 等基准与 GPT‑4o‑mini 进行评测，并将六种 dLVLM 与七种主流 AR LVLM 进行对照。

**📊 数据集**

使用的主要数据集包括 MSCOCO（POPE/CHAIR）、FairFace、CUB‑200‑2011、Stanford Dogs 以及精心构造的长度控制 MCQA。

**📈 对比分析**

通过定量对比发现 dLVLM 在对象幻觉上可与 AR 基线竞争，但语言质量普遍较差；在种族/性别分类上出现近零准确率与相反极端性别偏差；在 MCQA 长度偏差上表现出 85% 以上的准确率差距，且这种偏差在扩散模型的第 0 步就已形成。

**⚠️ 局限性**

实验受限于训练数据与视觉编码器的混杂、GPT‑4o‑mini 评判未获得人工验证、未涵盖毒性或同情等其他失效模式，以及缺乏更细粒度的消融分析。

---

## 629. MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval

**arXiv ID:** 2609.01316 | [PDF](https://arxiv.org/pdf/2609.01316v1)

**作者:** Debanjan Mahata `[一作]` (Bloomberg), Ozan Irsoy `[通讯]` (Bloomberg)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在检索视觉丰富文档时，先用多模态大模型对页面进行增广，生成经过校验的文本字段，然后通过 BM25F、密集检索或混合检索在文本索引中完成查询，而非在查询时进行视觉多向量检索。

**💡 创新点**

将多模态推理从查询时迁移到索引时，实现一次性离线增广；通过验证循环保证字段准确性；用验证过的文本字段替代视觉多向量索引，兼具较高精度与更低部署成本。

**🔧 技术方法**

多模态LLM（如 GPT-5.1）用于页面级和文档级增广；提取–验证–修订循环；BM25F、均值池化密集检索、Reciprocal Rank Fusion；EmbeddingGemma 等嵌入模型。

**📊 数据集**

ViDoRe V3（5个英文域和2个法语域的多模态文档检索基准）。

**📈 对比分析**

与 ColPali/ColQwen2.5 等视觉多向量检索进行对比。其在英文域上取得 0.6219 的平均 P@10（比原始 BM25 高 23%），与 ColQwen2.5 近乎持平；在法语域上跨语言增广后 0.5448，优于 ColQwen2.5；同时索引占用 9 倍更小，查询延迟约 2 倍更快。

**⚠️ 局限性**

增广过程依赖于昂贵的 MLLM 调用，增广成本随文档规模线性增长；对低资源或开源模型的效果不佳；某些字段（如图表摘要）在部分域可能无效或产生误导；缺乏对更广泛语言对和更大规模数据集的验证。

---

## 630. Explore Before Committing: Hypothesis-Guided Search for Deep Research Agents

**arXiv ID:** 2609.01294 | [PDF](https://arxiv.org/pdf/2609.01294v1)

**作者:** Ruochen Zhou `[一作]` (City University of Hong Kong), Shiqi Chen `[通讯]` (University of Oxford)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 HypoSearch，一种在搜索分叉点生成轻量化假设并通过并行分支证据比较来提升深度研究代理搜索性能的方法。

**💡 创新点**

创新点在于主动将不确定搜索状态拆解为多条可执行假设，避免单轨迹提前锁定错误方向，并在分支层面进行证据级聚合。

**🔧 技术方法**

采用假设生成、并行搜索分支、证据比较聚合、基于 ReAct 的工具调用以及 CIR/SSR 等行为指标来指导搜索与决策。

**📊 数据集**

使用 BC‑small、BrowseComp‑zh、FutureX、ResearchRubrics 四个深度研究基准，并在 Kimi‑K25、Qwen3.5‑122B、DeepSeek‑V3.2 三个模型上进行实验。

**📈 对比分析**

与 Pass@1、Majority Voting、Best‑of‑N 等基线对比，HypoSearch 在所有基准和模型上均取得更高准确率，同时工具调用成本低于完整并行采样。

**⚠️ 局限性**

局限性包括固定的分支数量与预算、提示式直接/分叉判别可能误判，以及缺乏对不确定性进行自适应分支与预算调节。

---

## 631. Accurate Reconstruction of Gas Turbine Blade Geometry Using 3D/2D Rigid Registration and CT View Optimization

**arXiv ID:** 2609.01368 | [PDF](https://arxiv.org/pdf/2609.01368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 632. TriSLA: A Preventive and Closed-Loop SLA-Aware Architecture for Multidomain Decision-Making with Explainable Artificial Intelligence in 5G Networks

**arXiv ID:** 2609.01293 | [PDF](https://arxiv.org/pdf/2609.01293v1)

**作者:** Abel J. R. Lisboa `[一作]` (Universidade do Vale do Rio dos Sinos), Cristiano B. Both `[通讯]` (Universidade do Vale do Rio dos Sinos)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了TriSLA架构，整合语义意图解析、跨域机器学习预测与XAI解释、以及闭环运行时保障，实现5G多域网络切片的预防式SLA管理；

**💡 创新点**

创新点在于将本体驱动的语义转换、跨域预测风险推理与可解释性特征归因统一到同一闭环流程，并在资源分配前即保证SLA可达成；

**🔧 技术方法**

采用本体推理(NEST/GST)、随机森林/XGBoost等多模型预测、SHAP/XAI特征归因、Kubernetes‑容器化微服务、Prometheus/OpenTelemetry等监控技术；

**📊 数据集**

使用NASP云模拟平台收集的跨域实时遥测数据（RAN、Transport、5GC）和合成负载，形成完整的实验数据集；

**📈 对比分析**

与静态阈值（80.4%）和全接纳（51.2%）基线相比，TriSLA在预防式入选下实现100% SLA满足率，模型准确率最高达99.51%；

**⚠️ 局限性**

局限于仿真环境、单一网络配置、缺乏大规模多租户与硬件验证，且未涵盖概念漂移下的在线模型再训练与动态策略优化。

---

## 633. Contribution-Aware Bandwidth Allocation for Multimodal Split Learning

**arXiv ID:** 2609.01406 | [PDF](https://arxiv.org/pdf/2609.01406v1)

**作者:** Iason Ofeidis `[一作]` (Yale University), Leandros Tassiulas `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于贡献的带宽分配方法 ModalShare，针对多模态拆分学习中共享上行链路的分配问题，通过服务器端的 Shapley 贡献评分动态确定各模态的保持比例，从而在固定总传输量下提升模型性能。

**💡 创新点**

创新点在于把多模态共享带宽分配视为独立的“交叉模态分配”决策，引入服务器端 Shapley 贡献评估，无需额外上行流量即可精准分配预算，并显著优于传统等比保持比例的方法。

**🔧 技术方法**

使用了 Shapley 值贡献探测、指数移动平均平滑、温度控制的比例映射、置信度冻结机制，以及多种现有压缩器（SplitFC、Top‑S、RandTop‑S）进行实验验证。

**📊 数据集**

实验数据集包括 CREMA‑D（音视频情绪识别）、MVSA（图文情感分析）和 UCI HAR（加速度/陀螺仪动作识别），三者在多模态拆分学习框架下均可复现。

**📈 对比分析**

与等比保持比例的基线在相同总传输量（β）下对比，ModalShare 在 CREMA‑D 上提升 15.4 百分点，MVSA 上提升 12.4 百分点，在 UCI HAR 上提升约 3-4 百分点；实验覆盖多种压缩器、预算级别和拆分深度，验证了其在中等压缩率下的显著优势，并证明冻结机制不损失性能。

**⚠️ 局限性**

局限性包括主要验证在仅两模态的场景，随着模态数增多需要更复杂的 Shapley 计算；对拆分深度敏感，深层拆分时优势减弱；服务器端共轭计算开销较大，且在贡献差距不明显时无法显著获益。

---

## 634. IntroConformal: Conformal Factuality Guarantees for Large Vision-Language Models via Introspective Signals

**arXiv ID:** 2609.01375 | [PDF](https://arxiv.org/pdf/2609.01375v1)

**作者:** Md. Atabuzzaman `[一作]` (Virginia Tech), Chris Thomas `[通讯]` (Virginia Tech)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 IntroConformal，一种训练‑free 的 Conformal Risk Control（CRC）框架，用模型自身的内在信号对大型视觉‑语言模型（LVLM）的事实性生成进行统计控制。

**💡 创新点**

创新点包括：① 用模型内部的层间语义稳定性（S_sem）和模型自身的验证概率（S_prob）作为一致性分数，完全不依赖外部验证器；② 在 CRC 中使用学习‑然后‑测试（LTT）与 Hoeffding 上界实现分布无关、有限样本的事实性风险保证；③ 通过一次前向传播即可获得 S_prob，兼顾了可扩展性与效率。

**🔧 技术方法**

技术方法：Conformal Risk Control（CRC）+ Learn‑Then‑Test 机制；层间语义相似度计算（S_sem）；基于模型的“是/否”验证提示并读取 logits（S_prob）；多阈值 Hoeffding UCB 与 Bonferroni 校正；嵌套筛选与归一化风险评估。

**📊 数据集**

使用的数据集包括：MSCOCO（场景理解）、CUB/Stanford Cars/Stanford Dogs（细粒度标注）、SROIE（发票文档理解），并在 LLaVA‑1.5、Phi‑3.5‑Vision、Llama‑3.2‑Vision、Qwen2.5‑VL‑7B、Qwen3‑VL‑8B 等五种 LVLM 架构上进行评估。

**📈 对比分析**

对比方法：CONFLVLM（CLIP 外部验证器）、token‑probability baseline (T_prob)、以及多种解码/验证相关技术（Woodpecker、CoVe、VCD、ICD）。实验表明 IntroConformal（尤其是 S_prob）在满足 CRC 置信度的同时，显著降低了放弃率、提高了 F1 与响应准确率，且在不同任务与模型上均优于基线。

**⚠️ 局限性**

局限性：S_sem 需要白盒访问隐藏层，S_prob 只适用于能输出 logits 的模型；两种分数均需额外一次前向传播；风险保证基于校准标签，若存在系统性标签偏差可能导致阈值过宽；假设模型固定，模型更新需重新校准；保证为概率性质（1‑δ），在安全关键部署仍需人工监督。

---

## 635. Exact Risk-Complexity Laws for Projective Boundaries in Scenario Optimization and Distribution-Free Certification

**arXiv ID:** 2609.01355 | [PDF](https://arxiv.org/pdf/2609.01355v1)

**作者:** Giuseppe C. Calafiore `[一作]` `[通讯]` (Politecnico di Torino), Giuseppe C. Calafiore (Politecnico di Torino)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

本研究提出了一套“适当投影边界方案（proper projective boundary scheme）”的理论框架，并给出了该框架下的精确有限样本风险分布，揭示了情景优化、分割式合适度预测以及其他无分布假设认证方法中出现的贝塔-二项式风险公式的根本原因。

**💡 创新点**

创新点在于将风险分布与边界复杂度轮廓（complexity profile）联系起来，证明只有在该轮廓在观测值处稳定时才退化为经典贝塔分布；当边界大小随机时，需额外的“轮廓因子”来修正风险分布；同时给出了可验证的“边界等价性”和“投影性”条件，能够判定多种实例（标量排序、支持重构、坐标场景包络、Pareto 前沿等）是否满足精确分布。

**🔧 技术方法**

主要使用的技术包括：可测性与可置换性分析、边界等价性与投影性证明、贝塔分布与贝塔-二项式等概率论工具、Hausdorff 时刻定理、样本压缩与稳定压缩理论、以及蒙特卡洛仿真验证。

**📊 数据集**

论文未使用真实数据集，所有示例均基于理论构造或均匀分布的模拟样本，主要通过公式推导和 Monte‑Carlo 仿真展示结果。

**📈 对比分析**

与传统固定维度贝塔上界对比，本文证明在随机边界情形下固定贝塔上界往往过保守或失效；通过数值实验（坐标包络与 Pareto 前沿）显示修正后的轮廓分布能够更准确地刻画风险，且在条件置信度下提供更紧的 PAC 证书。

**⚠️ 局限性**

局限性在于：需要事先知道或估计复杂度轮廓；若轮廓不稳定，精确分布可能难以获得；对于非投影或非等价的边界（如一般样本剔除策略），需使用保守的样本压缩或专门的剔除界定；此外多风险扩展仍受限于对联合边界的可判定性。

---

## 636. Collision-based logic in Lenia and its composition boundary

**arXiv ID:** 2609.01348 | [PDF](https://arxiv.org/pdf/2609.01348v1)

**作者:** Chakshu Gupta `[一作]` `[通讯]` (Georgia Institute of Technology), Chakshu Gupta (Georgia Institute of Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在标准 Lenia 规则下，利用 Orbium 光滑游走格子构造了一个 INHIBIT 逻辑门，并演示了两条此门的串联；

**💡 创新点**

首次实现了无训练规则的碰撞逻辑门，证明连续细胞自动机中自发碰撞可完成布尔运算，并确认仅 Orbium 滑行格子满足碰撞后双体生存的必要条件；

**🔧 技术方法**

采用 Lenia 的卷积‑增长更新、Fourier 频域平移精确定位、粗略与精确扫荡碰撞参数、连通域与质量比判定等技术对碰撞结果进行分类与评估；

**📊 数据集**

仅使用在实验中自生成的 Lenia 栅格数据，没有外部数据集；

**📈 对比分析**

通过在 24 个呼吸相位与 9 个偏移下测试门的输出，发现门在所有相位均能完全阻断；两门串联在 4 种间距下都能正确输出，表明门的鲁棒性强；但对转向、吸收等功能未做完整对比；

**⚠️ 局限性**

主要限制在于：仅实现了单向串联，未能完成转向传递和剩余碰撞吸收；门输出无法在通道中保持精确轨迹；且仅 Orbium 满足碰撞生存条件，无法在其他连续规则上复制。

---

## 637. Probing Factual Knowledge Transfer with Training Data Interventions

**arXiv ID:** 2609.01341 | [PDF](https://arxiv.org/pdf/2609.01341v1)

**作者:** Romina Oji `[一作]` (Linköping University), Jenny Kunz `[通讯]` (Linköping University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建干预式实验框架，使用英语预训练模型继续在波斯语数据上预训练，并系统性地从波斯语训练集去除特定事实，随后用自制模板评测模型在两种语言上的事实回忆与跨语言迁移。

**💡 创新点**

创新点在于：① 设计并公开SIFT数据集（500条跨语言事实三元组，包含难度不同的负样本和原生波斯语模板），② 通过三级事实移除干预（共现删除、完整实体删除、模板替换）实现对“直接学习”“跨语言迁移”“浅层线索”三条因果路径的拦截和量化。

**🔧 技术方法**

技术手段包括：继续预训练（在英语预训练模型上添加波斯语数据并采用12%英语重放避免灾难性遗忘），多级文本处理与实体匹配，基于长度归一化log‑prob 的模板填空评测，使用AccEN/AccFA/Overall/Transfer/Non‑Transfer/RankC六个指标对比模型性能。

**📊 数据集**

使用数据集：SIFT（20类关系、500条三元组，按一般与波斯相关两类划分）；FineWeb‑2波斯语大语料；FineWeb‑Edu英语语料；原始英语预训练模型与波斯语从零预训练模型作为基线。

**📈 对比分析**

比较方法：在六个模型（英语基线、波斯基线、五种干预版继续预训练）下分别在硬/随机候选集上计算六个指标。结果显示：在最严格干预（完整实体删除）下，跨语言迁移率约为40%（Transfer），整体正确率仅约20%，显著低于随机候选集，表明迁移有限；相较基线模型，硬候选集上的表现提升但仍不高，说明模型仍受浅层线索与频率限制。

**⚠️ 局限性**

局限性：仅研究单一目标语言波斯语（非拉丁字符），模型规模仅572M；SIFT与模板需人工手工构建，难以快速迁移到其他语言；未评估更大规模模型在相同干预下的迁移表现。

---

## 638. One-Layer Transformer Provably Learns Multiclass One-Nearest Neighbor in Context

**arXiv ID:** 2609.01311 | [PDF](https://arxiv.org/pdf/2609.01311v1)

**作者:** Skanda Athreya `[一作]` (James B. Conant High School), Yutong Wang `[通讯]` (Illinois Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

证明了单层softmax注意力模型与多分类一近邻（1‑NN）分类器等价，尤其在使用argmax头时实现；

**💡 创新点**

创新点在于将二分类的等价结果推广到多分类，通过简正编码（simplex encoding）填补先前研究的空白；

**🔧 技术方法**

使用理论分析（梯度下降动力学、标量化两参数族、缩放恒等式）、简单编码、单层注意力模型以及平方损失等技术；

**📊 数据集**

实验使用随机生成的球面输入与均匀标签（合成数据），并设计了“well‑separated”测试分布来验证理论；

**📈 对比分析**

与先前的二分类结果对比，实验显示在分离测试集上达到 100% 的argmax精度，理论上收敛速度为 O((K‑1)/K·(N,d)/log t)，而在均匀测试集上仅能获得略低的精度；

**⚠️ 局限性**

局限性包括仅适用于单层网络、对特殊的对角初始化敏感、只对平方损失给出严谨证明、单头结构且假设训练分布为独立均匀。

---

## 639. Multimodal RGB-Infrared Combination for UAV-Based Wildfire Segmentation: A Comparative Study on FLAME3

**arXiv ID:** 2609.01390 | [PDF](https://arxiv.org/pdf/2609.01390v1)

**作者:** Matheus F. Kovaleski `[一作]` (University of Coimbra), João Ruivo Paulo `[通讯]` (University of Coimbra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了在无人机火灾分割任务中使用RGB-红外（IR）多模态融合，比较了RGB、IR单模态以及早期、中期、晚期融合策略在U‑Net、DeepLabV3+和SegFormer三种语义分割架构上的表现。

**💡 创新点**

首次系统性评估了不同融合时机对三种主流分割模型的影响，并揭示了Transformer‑based SegFormer在中期/晚期融合时能更好地利用多模态信息，从而显著提升火灾像素检测性能。

**🔧 技术方法**

采用深度卷积网络（U‑Net、DeepLabV3+）和自注意力变压器网络（SegFormer）作为基准模型；融合策略包括输入级拼接、特征级融合以及输出级融合；训练使用二元交叉熵与AdamW优化器。

**📊 数据集**

使用FLAME 3数据集，该数据集提供同步的RGB、热红外影像及辐射温度图，用于生成基于温度阈值的火灾二值掩模。

**📈 对比分析**

在统一的实验协议下，所有模型使用相同的数据划分、预处理和训练超参数进行比较。实验结果显示，IR单模态已远优于RGB单模态；SegFormer+中期/晚期融合获得最佳指标（IoU≈0.74–0.75、F1≈0.84），而U‑Net和DeepLabV3+对融合时机更敏感，效果不如IR单模态。

**⚠️ 局限性**

研究仅在FLAME 3数据集上验证，缺乏在不同环境、传感器条件下的泛化评估；且仅考虑了三种融合策略，未探究更复杂或不确定性感知的融合方法。

---

## 640. Diffusion Based Unpaired Data Learning for Inverse Problems

**arXiv ID:** 2609.01370 | [PDF](https://arxiv.org/pdf/2609.01370v1)

**作者:** Chenglong Bao `[一作]` (Tsinghua University), Defeng Sun `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的无配对数据逆问题求解方法LUD-DIF，利用弱耦合假设将联合ELBO分解为可训练的两个独立扩散过程；

**💡 创新点**

在无配对数据条件下首次从变分推断角度推导联合ELBO分解并给出误差上界，同时提出基于理论的弱耦合时间步选择启发式；

**🔧 技术方法**

使用扩散模型、ELBO变分推断、弱耦合假设、误差理论分析、UNet结构与DDIM采样器等技术；

**📊 数据集**

在模拟噪声（BSDS300）、真实手机噪声（SIDD）以及超分辨率（AIM19、NTIRE20）数据集上进行实验；

**📈 对比分析**

与LUD-VAE、C2N、DeFlow、ASBM、CBDNet等方法对比，LUD-DIF在噪声生成质量（FID、AKLD、AFMD）上实现领先，并在下游去噪/超分辨率任务中获得竞争性或最佳的PSNR/SSIM/LPIPS；

**⚠️ 局限性**

需已知或近似前向算子，无法处理盲逆问题或语义翻译；弱耦合假设的误差上界仍存在，如何进一步缩小误差是未来工作方向。

---

## 641. Disproving the Greedy Superstring Conjecture

**arXiv ID:** 2609.01365 | [PDF](https://arxiv.org/pdf/2609.01365v1)

**作者:** Hiroki Shibata `[一作]` `[通讯]` (Kyushu University), Hiroki Shibata (Kyushu University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构造特殊的SCS实例，证明贪心算法的逼近比率至少为9/4，反驳原有的贪心超字符串猜想。

**💡 创新点**

首次给出贪心算法的严格下界超过2，并展示一种组合构造方法及通过de Bruijn图欧拉回路实现最优串的证明，突破了近四十年的未解难题。

**🔧 技术方法**

使用组合构造、循环子串、重叠长度分析、de Bruijn图与欧拉回路、以及迭代合并等理论技术。

**📊 数据集**

采用人工生成的符号集合 {x, y, a, b₁,…,b_t} 构造的实例，未使用真实数据集。

**📈 对比分析**

与先前已知的下界2和上界3比较，证明贪心算法最差性能≥9/4；本工作仅提供理论证明，没有实验评估。

**⚠️ 局限性**

仅给出下界，实际最差逼近比率仍不确定；构造仅适用于偶数长度 k ≥ 10；缺乏实验验证，难以评估其在实际 SCS 任务中的表现。

---

## 642. Scalable Rao-Blackwellized Online Planning for High-Dimensional POMDPs

**arXiv ID:** 2609.01351 | [PDF](https://arxiv.org/pdf/2609.01351v1)

**作者:** Jiho Lee `[一作]` (University of Colorado Boulder), Zachary Sunberg `[通讯]` (University of Colorado Boulder)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可扩展的 Rao-Blackwell 化在线规划框架，并在室内搜索救援任务中与 FastSLAM 2.0 集成，显著降低高维 POMDP 的采样方差。

**💡 创新点**

创新点在于将 Rao-Blackwell 化与混合连续‑离散信念表示结合，允许对任意可解析子状态进行解析传播，并在树搜索中使用确定性积分来降低采样噪声。

**🔧 技术方法**

采用了 RB-POMDP、RBPF、FastSLAM 2.0、SMolyak 稀疏网格积分和 Probit 链接的概率检测模型等技术。

**📊 数据集**

使用了人工生成的多房间室内搜索救援仿真数据集，包含未知受害者位置和机器人运动观测。

**📈 对比分析**

通过与传统 POMCPOW+SIRPF 和 RB-MC-POMCPOW 的对比，实验表明在相同计算预算下 RB-POMCPOW 的累计奖励提升 2–3 倍，所需粒子数和树迭代次数显著减少。

**⚠️ 局限性**

限制在于期望计算仅适用于可解析分布（如高斯、伯努利），无法直接处理占据栅格等高维离散模型，且提高稀疏网格水平会导致计算量快速上升。

---

## 643. ExBind: A Controlled Diagnostic Benchmark for Visual-to-Executable Correspondence

**arXiv ID:** 2609.01344 | [PDF](https://arxiv.org/pdf/2609.01344v1)

**作者:** Ziqian Wang `[一作]` (Tsinghua University), Jinli Suo `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种专门评估视觉编辑系统中“可视化指代到可执行对象对应”层的诊断工具ExBind；

**💡 创新点**

通过将潜在绑定对象与多种表面（SVG、DOM、canvas、树、图、表格）映射，构建可确定的结构约束，首次实现了在不同表面之间可比较且可追踪的可执行对应诊断；

**🔧 技术方法**

使用了可确定的编译器将潜在绑定实例转化为多种表面，并利用结构化约束集对模型输出进行判定；

**📊 数据集**

构建了250例广义诊断集、240例结构化目标集及50个跨表面潜在绑定组，共计约870个案例；

**📈 对比分析**

在Qwen2.5-VL-3B与Qwen3-VL-4B两款冻结的多模态大模型上评测，发现候选有效率高但准确率仍受限，尤其在表格诊断中出现“行正确、列错误”的一致错误模式；

**⚠️ 局限性**

局限于单一编辑循环、缺乏多轮编辑误差累积评估、跨表面匹配仅限于兼容编译器、候选顺序对结果影响大、未覆盖完整的真实系统行为。

---

## 644. SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers

**arXiv ID:** 2609.01343 | [PDF](https://arxiv.org/pdf/2609.01343v1)

**作者:** Shaowen Wang `[一作]`, Jian Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在MoE Transformer中通过循环共享中间层实现深度重用的方法；

**💡 创新点**

在保持算术FLOPs、参数量和KV缓存相同的三预算匹配框架下，发现循环可在多尺度、多稀疏度下持续提升性能；

**🔧 技术方法**

采用了Sparse MoE Transformer、Grouped‑Query Attention、RMSNorm、训练调度WSD、Chinchilla‑style scaling law、以及对专家路由和注意力的内部分析；

**📊 数据集**

使用内部大规模语料库（含代码、数学/STEM、金融、知识、网页等5个领域），并在DCLM Core、MMLU等公开基准上评估；

**📈 对比分析**

与对齐参数、FLOPs和KV缓存的基线模型对比，循环模型在验证损失、DCLM Completion、DCLM Core和MMLU上均优于基线，且CE Gain在10^20–10^22 FLOPs间为约7–23%，优势随模型规模和输入长度增大而放大；

**⚠️ 局限性**

实验仅在200M尺度上搜索循环配置，未探索更大尺度的最佳循环策略；仅考察连续块循环，未探究更灵活的循环变体；预算匹配未覆盖硬件实时计算开销，内部机制的因果关系仍待进一步研究。

---

## 645. mzCache: On-Device LLM Memory Management under Multitasking

**arXiv ID:** 2609.01338 | [PDF](https://arxiv.org/pdf/2609.01338v1)

**作者:** Hongseung Yu `[一作]` (Seoul National University), Kyunghan Lee `[通讯]` (Seoul National University)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对移动设备多任务环境下的本地LLM推理，设计了一套可弹性淘汰并快速恢复LLM内存的系统 mzCache。

**💡 创新点**

创新点包括：① 专注于外部内存压力的恢复导向淘汰策略；② 将模型权重与 KV 缓存按层级和 256 词元块细粒度拆分为共享缓冲区；③ 混合交换（Hybrid Swap）同时利用 zRAM 压缩和闪存读取；④ 后向淘汰/前向恢复（backward‑out / forward‑in）实现 GPU 与 CPU 的并行恢复；⑤ 在统一内存 SoC 上实现无数据拷贝的 GPU‑CPU 协同推理。

**🔧 技术方法**

采用的技术包括：统一内存共享缓冲区（OpenCL SVM）、ARM NEON SIMD 的 KV 缓存压缩/解压算子、OpenCL 自定义注意力 kernel、层级/块级内存管理、压缩算法（8‑bit 量化、CacheGen）、混合交换与动态负载平衡、基于安卓内存压力回调的淘汰触发。

**📊 数据集**

使用的模型/数据集：Qwen3‑0.6B、EXAONE‑4.0‑1.2B；上下文长度 8k/16k/32k 词元；评测设备为 Galaxy S25+（12 GB LPDDR5X）和 OnePlus 12；对比实验使用标准 Android OS Paging（zRAM）和自定义 Partial Offload 基线。

**📈 对比分析**

与基线相比，mzCache 在 0%、25%、50%、75% 剩余内存时分别实现 2.1‑5.5× 的 TTFT（首 token 生成时间）加速；能显著降低能耗（总能量更低）并保持与 FP16 相近的 F1 分数；在真实多任务情景下，能够避免 LMK 终止并无冷启动地恢复上下文。

**⚠️ 局限性**

局限性包括：目前仅支持单一 LLM 上下文；对多上下文/多模型场景的扩展未实现；依赖统一内存 SoC 的特性，移植到仅有分离显存的设备需重写核心；热阈值变化可能破坏混合交换平衡；对 MoE 或稀疏注意力模型的恢复策略尚未提出。

---

## 646. LEAP: Likelihood Elicitation and Aggregation for LLM-based Probabilistic Forecasting

**arXiv ID:** 2609.01337 | [PDF](https://arxiv.org/pdf/2609.01337v1)

**作者:** Yufei Chen `[一作]` (Shandong University), Zhe Liu `[通讯]` (Zhejiang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LEAP方法，将LLM作为局部证据解释器，再用贝叶斯模型聚合为可审计的后验分布，改进预测阶段；

**💡 创新点**

核心创新是将证据解析与概率聚合分离，使每条证据的贡献可追溯，并在闭式更新中实现更精准、可解释的概率预测；

**🔧 技术方法**

使用贝叶斯概率模型（连续、高斯、离散、伯努利等）、似然参数抽取、温度化权重、依赖聚类、可靠性采样以及闭式后验更新；

**📊 数据集**

在FutureX、GAIA、BrowseComp等构建的预测、信息检索和浏览任务基准上进行评估；

**📈 对比分析**

在固定证据集的对比实验中，LEAP在FutureX、准确率、Brier、Spherical等指标上普遍优于Monolithic基线，尤其在长周期预测中提升显著；

**⚠️ 局限性**

仅改进预测阶段，对证据收集无提升；依赖于现有基准和模型，且多证据调用导致成本增加；对稀疏或无关证据的处理有限。

---

## 647. Automated Event Log Generation from Unstructured Text Using Finetuned LLMs

**arXiv ID:** 2609.01320 | [PDF](https://arxiv.org/pdf/2609.01320v1)

**作者:** Maximilian Seeth `[一作]` (LMU Munich), Daniel Schuster `[通讯]` (University of Mannheim)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种利用开放权重LLM将非结构化文本自动翻译成可直接用于流程挖掘的XES事件日志的框架，且通过监督微调提升了翻译质量。

**💡 创新点**

创新点在于：①使用合成文本‑日志配对数据进行监督微调，而非仅依赖提示工程；②在多个开源LLM（Llama3.1‑8B、Ministral‑8B、Qwen2.5‑7B）上实现本地部署；③提出了全面的评估体系，涵盖单轨迹、整体流程模型（Inductive/Heuristics Miner）的质量指标。

**🔧 技术方法**

核心技术包括：大语言模型（LLM）微调（LoRA）、零/少样本提示、XES格式生成、Levenshtein/Kendall τ 轨迹相似度度量、过程模型健全度、精确度与F1评估。

**📊 数据集**

使用了两个公开事件日志：Road Traffic Fine Management（约150k条）和Sepsis Cases（约1050条），并通过LLM生成对应的合成文本，构成训练与测试数据。

**📈 对比分析**

与零/少样本提示相比，微调后模型在生成合法XES、减少属性/活动幻觉、轨迹相似度、以及流程模型的fitness/precision/F1上均显著提升，部分指标近似原始日志（如Llama3.1‑8B的F1接近0.9）。

**⚠️ 局限性**

局限性包括：仅支持基于案例的XES格式，未覆盖对象中心（OCEL）等更复杂结构；合成文本与真实报文的分布差异；以及模型在不同领域文本生成的泛化能力仍需进一步验证。

---

## 648. A Composable Evaluation System for Reproducible Omni-Modal Foundation Model Evaluation

**arXiv ID:** 2609.01315 | [PDF](https://arxiv.org/pdf/2609.01315v1)

**作者:** Hodong Lee `[一作]` (NAVER Cloud AI), Geewook Kim `[通讯]` (NAVER Cloud AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套统一的omni‑modal评估系统OmniEvaluator，能够在单一接口下使用多种推理引擎与评估框架，对文本、图像、视频、音频四种模态进行可复现的评估，并提供交互式仪表盘与轻量级语义验证器。

**💡 创新点**

创新点在于通过统一的中间 schema 让任意推理引擎与任意评估框架只需一个适配器即可组合，形成可插拔、可扩展的评估管线；同时引入 CPU‑可运行的 verifier，解决跨引擎/提示不一致导致的分数漂移；以及基于 GPU 共享的联邦评估模式提高资源利用率。

**🔧 技术方法**

采用的技术包括：多引擎适配器（HF Transformers、vLLM、SGLang、API 客户端），统一中间数据结构，Python CLI 与 REST 接口，GPU 共享联邦推理，轻量化的 GGUF‑8bit verifier，仪表盘前端与后端同步，自动生成可复现的评估 artifact。

**📊 数据集**

覆盖了超过 1000 个公开 benchmark，涵盖文本（MMLU、GSM8K、MBPP等）、图像（MMU、MMBench等）、视频（Video‑MME、MMBench等）与音频（LibriSpeech、CoVoST2、VoiceBench等）模态。

**📈 对比分析**

评估方法通过统一 schema 对比不同推理引擎与提示设置下的原生 metric 与 verifier 分数，结果显示原生 metric 在相同 benchmark 下可出现 30–80 分差异，而 verifier 分数波动小于 5 分；联邦评估在相同 GPU 资源下实现 1.3–2.8 倍的壁垒时间加速，验证器平均准确率达 85%，与 GPT‑5.4‑mini、Claude‑Opus‑4.8 等商用评判相当。

**⚠️ 局限性**

局限性包括：依赖上游评估框架，框架 bug 可能直接影响分数；verifier 仅评估文本三元组的语义正确性，无法判定视觉/音频质量或多模态生成输出；且目前仅针对理解任务，未覆盖生成型多模态评估。

---

## 649. MeshSplatBench: A Unified Benchmark for Triangle-Based Neural Rendering

**arXiv ID:** 2609.01306 | [PDF](https://arxiv.org/pdf/2609.01306v1)

**作者:** Kaixuan Zhang `[一作]` (Nanjing University of Science and Technology), Xiatian Zhu `[通讯]` (University of Surrey)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 MeshSplatBench，一个统一的基准框架，涵盖从原始神经三角网格训练到 Unity 引擎部署的完整流程，并对 MeshSplatting 资产进行了拓扑审计。

**💡 创新点**

创新点包括：①在保持各方法原生优化语义的前提下统一评测协议，最大 0.8% PSNR 误差；②设计三层 Unity 部署协议，分离适配差距、可移植差距和部署差距；③对 MeshSplatting 的顶点共享资产做细粒度拓扑检测，揭示非流形、碎片化问题；④系统展示方法在不同渲染条件下的排名倒置。

**🔧 技术方法**

使用的技术：三角形神经渲染方法（2DTS、Triangle‑Splatting、MeshSplatting、DiffSoup 等），Unity 引擎三种渲染模式（Native、Dedicated、Default），PSNR/SSIM/LPIPS 质量度量，GPU 性能分析（FPS、GPU 延迟、显存占用），拓扑诊断算法。

**📊 数据集**

使用的数据集包括：Mip‑NeRF‑360、Tanks and Temples、DTU、NeRF‑Synthetic。

**📈 对比分析**

方法通过统一评测协议在原生环境、专用 Unity 渲染器和标准 Unity 网格路径下分别评估 PSNR、SSIM、LPIPS、FPS、显存与训练时间；结果显示：在原生环境中 2DTS 性能最好，但在默认 Unity 路径下 MeshSplatting 最优；适配差距最高可达 4.9 dB，可移植差距最高 10.8 dB；DiffSoup 速度最快但质量最低。

**⚠️ 局限性**

局限性：没有单一方法在所有指标（原生质量、几何精度、训练成本、渲染吞吐量、部署兼容性和网格完整性）上统治；引擎适配导致显著质量损失；拓扑分析表明即使是索引网格仍存在非流形和碎片化；Benchmark 仅关注静态重建，未涵盖动画、物理和更广泛的渲引擎平台。

---

## 650. Neuro-Symbolic Geometric Abstraction (NeuSOGA): From Observations to Symbolic Mathematical Representations

**arXiv ID:** 2609.01408 | [PDF](https://arxiv.org/pdf/2609.01408v1)

**作者:** Qingde Li `[一作]` (University of Hull), Jie Tian `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为 NeuSOGA 的神经-符号几何抽象框架，能够将观测数据通过拓扑抽象、几何抽象和符号合成三步，直接生成可解释、可编辑的隐式面积样条（Area Spline）数学表示。

**💡 创新点**

创新点在于：①用欧氏距离变换 (EDT) 自动抽取拓扑核心并驱动无监督的基础模型分割；②引入自适应多尺度几何抽象，将稠密观测压缩为稀疏控制多边形；③将控制多边形映射为闭式、可加、任意阶连续的隐式面积样条，提供真正的符号表达而非隐藏在神经网络权重中的统计表示；④实现跨模态、跨视角的统一抽象流程，消除了对任务特定训练的需求。

**🔧 技术方法**

技术细节包括：欧氏距离变换、Segment Anything（SAM/SAM2）拓扑引导分割、尺度空间理论和高斯滤波实现粗细尺度对比、Douglas–Peucker 简化、以及 Piecewise Algebraic Spline（隐式面积样条）合成。

**📊 数据集**

使用的数据集主要有 ModelNet40 点云（三种视角和随机视角）以及 COCO 2017 的二值化光学掩模，用以验证模态不变性。

**📈 对比分析**

方法通过四项定性评估（结构保持、拓扑一致、表示稀疏度、符号表达性）进行比较，实验显示 NeuSOGA 在不同类别、不同视角和不同模态下均能保持主要几何特征，生成的隐式面积样条既简洁又可视化，且不依赖任何训练集；在数值上与现有学习型 Scan‑to‑CAD 系统相比，虽然未给出直接重建误差指标，但在结构保留和符号可编辑性方面表现更优。

**⚠️ 局限性**

局限性包括：①仅实现符号表示生成，缺乏对生成符号进行更高层次推理、对称性检测或几何语法学习的机制；②实验规模受限，仅覆盖有限类别和视角，未检验在更大、多样化数据集下的稳定性；③未给出严格的定量误差指标，主要依赖定性可视化；④框架目前只处理 2D 投影，尚未完整推广到 3D 隐式模型。

---

## 651. Obstacle-Aware Autonomous Coverage and Navigation for Outdoor Robots

**arXiv ID:** 2609.01384 | [PDF](https://arxiv.org/pdf/2609.01384v1)

**作者:** Leonardo Gargani `[一作]` (Politecnico di Milano), Matteo Matteucci `[通讯]` (Politecnico di Milano)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了一个统一的 ROS 2 架构，用于单机器人在户外环境中完成长时段覆盖与导航任务，涵盖定位、规划、执行及充电等完整工作流；

**💡 创新点**

创新点包括：①将双天线 RTK‑GNSS 与轮速计融合的 EKF 进行方向估计与自适应回退；②针对 Fields2Cover 的三项控制器感知改进（自适应重采样、障碍物感知连接器、布局聚类），提升离线计划的可执行性；③在 Nav2 之上构建两层行为树（任务层与导航层），实现多目标、恢复策略、未知障碍管理及自动充电；

**🔧 技术方法**

技术实现主要包括：双天线 RTK‑GNSS+轮速计 EKF、Fields2Cover 规划器、控制器感知改进、Nav2 导航栈、Behavior‑Tree 任务管理、Livox 360 LiDAR 与 ArUco 标记的视觉对齐；

**📊 数据集**

实验使用五个户外区域（面积 168 – 1641 m²），包含凸/非凸、无障碍/有障碍场景，并在真实机器人上进行部署；

**📈 对比分析**

对比方法：以离线规划的理想覆盖面积为基准，测算实际覆盖效率；在所有场景中完成覆盖，效率从 93.2 % 到 96.1 %；相较于未改进的 Fields2Cover 或标准 Nav2，系统在未知障碍下保持更高连贯性并实现自主充电；

**⚠️ 局限性**

局限性：依赖 RTK‑GNSS 进行定位，若 GNSS 信号弱或失效则需更鲁棒的定位方案；未知障碍处理仅通过退避与重规划，未实现多机器人协同；未来工作计划扩展到多机器人覆盖与 GNSS 降级场景。

---

## 652. How Correct Is Your Answer? A Semantic Correctness Framework for Open QA Evaluation

**arXiv ID:** 2609.01369 | [PDF](https://arxiv.org/pdf/2609.01369v1)

**作者:** Elitsa Yotkova `[一作]` (Sofia University St Kliment Ohridski), Preslav Nakov `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了八类有序语义正确性分类，并基于此发布了CAP-Correctness和CAP-Statements两个数据集，同时设计了Context-Aware Precision（CAP）指标进行答案质量评估。

**💡 创新点**

创新点包括：①细粒度且有序的正确性分类，使评估更具诊断性；②将双向自然语言推理（NLI）融入评估，得到更贴合语义的连续分数；③引入单调性评估协议，系统性检验指标是否遵循预期的正确性排序。

**🔧 技术方法**

使用的技术包括：NLI模型（用于双向推理）、mT5序列到序列模型（将问答对转化为声明）、CAP评分公式（结合 entailment 与 neutral 的概率），以及对比传统文本相似度和语义嵌入方法。

**📊 数据集**

使用的数据集有：CAP-Correctness（8.8k条 QA‑答案‑标签，来源于OpenBookQA、ARC、MMLU）和 CAP-Statements（11k条 QA‑声明，用于训练与评估声明生成）。

**📈 对比分析**

与BLEU、ROUGE、METEOR、BERTScore、COMET等基线进行对比，采用 Spearman、Kendall、pairwise accuracy 等指标评估。CAP在排名相关性（Spearman 60.37）和 pairwise 准确率（77.70%）上显著优于所有基线。

**⚠️ 局限性**

限制包括：CAP 在 overinclusive-valid / overinclusive-invalid 区分上可能出现倒置；依赖 NLI 的世界知识与推理质量；声明生成误差会直接影响评分；数据集仅为英文且人工标注覆盖有限；与传统指标相比计算成本更高。

---

## 653. Evaluating Multimodal LLMs as Generalist Vision-Language-Action Agents for Drone Control: Commanding, Approaching, Tracking and Searching

**arXiv ID:** 2609.01404 | [PDF](https://arxiv.org/pdf/2609.01404v1)

**作者:** Jaewoo Park `[一作]` (NAVER Cloud), Geewook Kim `[通讯]` (NAVER Cloud)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一个通用无人机代理，允许任何多模态大型语言模型（MLLM）作为可插拔组件，利用基于指向的四个动作（点位、偏航、思考、到达声明）完成搜索、推理和停止决策；并提出 DroneCATS 基准，统一评估四种任务（接近、追踪、搜索、指挥多机）

**💡 创新点**

创新点在于：①将终止判断完全移至模型内部，让无人机自行声明到达；②提出统一的声明锚定成功标准，消除不同任务之间的可比性差异；③实现多机指挥场景，首次在同一闭环框架下评估单一模型控制多架无人机；④系统架构与动作空间完全声明式，模型可自由替换，无需细调或功能调用

**🔧 技术方法**

技术核心包括：①基于提示声明的 VLA 接口，采用 JSON 对象传递指令；②几何投影将二维像素+深度转换为 3D 位移；③规则化的控制栈（速度→姿态→转子）实现闭环飞行；④在 AirSim/Unreal Engine 环境中模拟四种任务场景；⑤多模型推理服务，支持 GPT‑5、Gemini、Qwen3.5、Cosmos3‑Edge 等

**📊 数据集**

使用自定义的 AirSim 地图（住宅区、校园、Blocks）以及手工放置的目标对象；任务分为四类（是否移动、是否在首帧可见），共 100 条情节；通过设定的 δ=5 m 判定成功，并记录声明、最小距离、轨迹等指标

**📈 对比分析**

对比方法：在相同的 80 条单机任务与 20 条多机指挥任务上，统计成功率、宣言率、NE（导航误差）等；结果显示最强模型 GPT‑5/ Gemini 3.7 Flash 在接近可见目标时 65% 成功，隐藏目标 40%；小模型 Qwen3.5‑9B 90% 进入成功半径但仅 35% 成功；小模型 Qwen3.5‑2B 仅声明但从未靠近；多机指挥中，只有最强模型在 80% 的情节中取得成功，其余模型因声明失误或发送相同点位导致失败

**⚠️ 局限性**

局限性包括：①仅在仿真环境评估，真实硬件验证未完成；②未评估层级指挥（单机子代理+指挥者）；③测试时推理延迟对控制速率的敏感性尚未完全解决；④小模型的协议遵循问题未通过模型参数或训练方法得到根本改善；⑤需要进一步探索多机指挥下的共享策略和更细粒度的动作空间

---

## 654. Scale-based Approach for Active Wildfire Segmentation on Satellite Imagery

**arXiv ID:** 2609.01392 | [PDF](https://arxiv.org/pdf/2609.01392v1)

**作者:** Matheus F. Kovaleski `[一作]` (University of Coimbra), João Ruivo Paulo `[通讯]` (University of Coimbra)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究了在火灾规模分布偏移下的卫星影像火点分割，并提出基于IQR的规模分割方案，评估模型对火灾规模变换的鲁棒性。

**💡 创新点**

引入规模基于IQR的训练/测试划分来系统比较不同SWIR配置与三种分割架构的表现，并验证SWIR2在火灾分割中的重要性。

**🔧 技术方法**

使用多光谱 Landsat-8 数据预处理、连通组件分析、U‑Net、DeepLabV3+、SegFormer网络以及SWIR1/2 组合的实验。

**📊 数据集**

Land8Fire 数据集（包含手工标注的多光谱火点）。

**📈 对比分析**

在随机拆分和规模拆分两种设置下，进行5次重复实验，用精确率、召回率、IoU、F1、MCC 进行评估；U‑Net在规模迁移下保持高召回率，SegFormer次之，DeepLabV3+召回率低；SWIR2或双SWIR配置优于SWIR1。

**⚠️ 局限性**

仅评估单一传感器（Landsat‑8），对规模阈值敏感，未检验跨生态或季节的泛化，缺乏统计显著性检验。

---

## 655. When Tokenization is Secretly Output Supervision

**arXiv ID:** 2609.01386 | [PDF](https://arxiv.org/pdf/2609.01386v1)

**作者:** Tanja Baeumel `[一作]` (German Research Center for Ai), Simon Ostermann `[通讯]` (German Research Center for Ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在自回归Transformer上对3位数加法任务进行控制实验，解耦输入与输出的tokenization，并结合线性探测器分析模型内部表示，探讨tokenization granularity作为输出监督对训练难度、任务性能与内部表征的影响；同时对120篇数理推理论文进行手工调查，揭示社区对tokenization的忽视。

**💡 创新点**

创新点包括：① 将tokenization视为输出监督而非仅输入预处理；② 提出了“最小计算假设”，预测模型只会在被监督的token层面内部表示；③ 通过解耦输入/输出tokenization实验验证输出tokenization主导训练结果；④ 系统性文献调查揭示tokenization对模型比较的潜在偏差。

**🔧 技术方法**

技术手段：使用4层、d_model=256的Transformer做自回归训练；构造holistic（完整数字为单token）与fragmented（数字拆为单字符或数字位）两种tokenization；进行10个随机种子和超参网格搜索；利用线性probe与MLP probe在残差流上评估各位数的可解码性；手工标注120篇论文中的tokenization信息。

**📊 数据集**

数据集：自生成的3位数加法样本（随机整数、逆序little-endian表示）；以及120篇包含数理推理/数学/数值内容的ACL/EMNLP等会议论文。

**📈 对比分析**

比较方法：通过训练损失曲线与评估准确率对比不同tokenization组合；在解耦实验中发现共享输出tokenization的模型表现相近，输入tokenization对性能影响较小；probe结果显示fragmented输出下仅首位数可解码，holistic输出下所有位均可解码；整体表现显示holistic输出模型训练更难、收敛慢、准确率低。

**⚠️ 局限性**

局限性：① 仅在小规模模型与三位数加法上验证，无法直接推广到大规模LLM；② 只关注标准next-token训练，未探讨序列级或RL等监督方式；③ probe只能测可解码性，不能断言信息缺失；④ 文献调查依赖关键词搜索与人工标注，可能存在漏检或误标。

---

## 656. EDGE: Error Dependency Graph-Guided Multi-Error Attribution in Multi-Agent LLM Systems

**arXiv ID:** 2609.01360 | [PDF](https://arxiv.org/pdf/2609.01360v1)

**作者:** Jun Hou `[一作]` (Virginia Tech), Xuan Wang `[通讯]` (Virginia Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于错误依赖图的多错误归因框架EDGE，构建观测与因果验证的错误依赖图并用于两阶段LLM检测。

**💡 创新点**

1) 将多错误归因视为依赖感知诊断；2) 通过对抗性回滚验证依赖边的因果性；3) 在检测中采用两阶段图引导提升归因精度；4) 区分验证子图用于解释与修复。

**🔧 技术方法**

使用Suppes/CAPRI筛选生成观测图、对抗性回滚与LLM判定验证因果边、两阶段LLM-as-judge检测，并在图中对边加权。

**📊 数据集**

公开多错误数据集TRAIL（span-level归因）和MAST（trace-level多标签分类），并自行构建错误事件注释。

**📈 对比分析**

与无图基线、随机图、+CG单阶段、仅验证图等对照，结果在TRAIL GAIA和SWE-Bench上F1分别提升至42.13/38.97，MAST GPT-4o从22.87提升至29.96，整体提升显著；但长上下文定位仍有限。

**⚠️ 局限性**

依赖已标注多错误数据，构建的图为全局而非实例化；对抗回滚受工具输出固定限制，因果效应估计受样本量影响；LLM判定可能带噪声；对新分布或长上下文仍不稳健。

---

## 657. Separating Syntax from Language: A Mechanistic Account of Translation in Multilingual LLMs

**arXiv ID:** 2609.01356 | [PDF](https://arxiv.org/pdf/2609.01356v1)

**作者:** Mikhail Sonkin `[一作]` (Saarland University), Simon Ostermann `[通讯]` (Saarland University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在多语言大型语言模型中，研究了翻译过程并将其拆解为语义、句法与表层语言三个独立阶段，证实句法先于表层语言被确定；

**💡 创新点**

首次将句法结构视为独立的中间层，并证明单个注意力头负责句法决策且对语言身份几乎不敏感，扩展了先前只区分语义与语言的分解框架；

**🔧 技术方法**

采用 LogitLens 投影可视化隐藏层输出、激活补丁（activation patching）进行因果定位，以及对单个注意力头的比例影响度量来追踪句法与语言信息的流动；

**📊 数据集**

构造了三套控制多语言数据集（名词短语、主谓宾、情态动词），分别涵盖了不同的词序差异；并利用 FLORES‑200 作为更自然的数据进行验证；

**📈 对比分析**

对 mGPT‑1.3B、Aya Expanse‑8B 与 LLaMA‑3‑8B 在上述数据集上进行 8 维（语法/语言/词义）概率差异分析，显示大多数模型遵循 S→L→C 的层级顺序；未给出具体数值指标，只呈现层级对应的最大概率差异与层位置；

**⚠️ 局限性**

实验受限于仅覆盖名词短语、主谓宾与情态动词等有限句法结构，模型规模与英语偏倚导致结果可能不具普适性；合成数据和补丁假设线性可交换也限制了结论的广泛性；

---

## 658. Autonomous robotic bridging using distributed swarm control without inter-agent communication

**arXiv ID:** 2609.01394 | [PDF](https://arxiv.org/pdf/2609.01394v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 659. SymFold: Synergizing Evolutionary and Structural Priors for Accurate Protein Inverse Folding

**arXiv ID:** 2609.01353 | [PDF](https://arxiv.org/pdf/2609.01353v1)

**作者:** Handong Wang `[一作]` (Chinese Academy of Sciences), Jianqiang Huang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 SymFold 框架，用对称双路径融合多模态蛋白语言模型（MPLM）与序列语言模型（PLM）进行蛋白逆折叠序列生成，并加入自校正迭代训练。

**💡 创新点**

创新点在于：①对称双路径设计，平衡结构与序列先验；②Adaptive Synergistic Fusion 动态融合各路径输出；③自校正策略消除训练-推理不匹配，提升生成质量。

**🔧 技术方法**

技术手段包括：多模态蛋白语言模型（ESM‑3）、传统蛋白序列语言模型（ESM‑C）、LoRA 参数高效微调、Adaptive Synergistic Fusion、Self‑Correction 迭代自校正。

**📊 数据集**

使用数据集：CATH4.2、CATH4.3、TS50、TS500、CASP15、CASP16；训练时采用 PiFold 结构编码器。

**📈 对比分析**

与 GraphTrans、ProteinMPNN、PiFold、LM‑Design、Knowledge‑Design 等基线在序列恢复率和困惑度上对比，SymFold 在 CATH、TS、CASP 测试集均取得 SOTA：恢复率提升 4–6% 以上，困惑度显著下降。

**⚠️ 局限性**

局限性在于：主要以计算评估（恢复率、pLDDT、TM‑score）为依据，缺乏实验室验证；对极端结构或缺失片段的鲁棒性待提升；对不同结构编码器的兼容性仍需进一步探索。

---

## 660. Bandits in Prod: Hyperparameter Optimization at Inference Time

**arXiv ID:** 2609.01335 | [PDF](https://arxiv.org/pdf/2609.01335v1)

**作者:** Louis Abraham `[一作]` (Tiime), Nicolas Devatine `[通讯]` (Tiime)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种面向在线超参数优化（OHPO）的框架IMABO，将无限多臂赌博问题与自适应oracle相结合，构造可扩展的活跃配置集合并采用IMOSS作为分配策略；

**💡 创新点**

创新点在于：①将OHPO正式化为无限臂赌博问题并引入oracle机制实现主动生成配置；②设计了restart‑free、任意时刻可用的IMOSS索引，提供量化的量化-回报边际累计量化退化上界；③提出多种oracle（随机、TPE、局部突变+KL-UCB+PE、TabPFN）支持连续、整数、类别以及树状条件空间；

**🔧 技术方法**

使用的技术包括：MOSS/Anytime-MOSS索引、TPE分裂与密度比估计、KL-UCB+Parzen估计的局部突变、Tabular Foundation Model (TabPFN) 的上下分位估计、主动活跃集合增长策略 (t^β) 以及基于量化回报的 regret 定义；

**📊 数据集**

使用的数据集包括：HPOBench 的三类离散随机森林任务（OpenML 任务 1、2、3）、两类连续优化任务（Logistic Regression 与 SVM，OpenML 任务 167149）、以及 HotpotQA 的检索增强生成问答任务；

**📈 对比分析**

比较方法：UCB‑AIR、Hier‑MAB、StroquOOL、StoSOO、HOO‑T、Hier‑MAB‑10 等；实验表明 IMABO（尤其是 IMOSS‑mutate‑KL×PE 与 IMOSS‑TabPFN）在累计 regret、单次 regret 与最终推荐质量上均优于或相当于基线，在离散、连续与 LLM 任务上均取得最小累计 regret，且在 HotpotQA 上取得最低在线平均 regret；

**⚠️ 局限性**

局限性包括：①采样偏差与存活偏差导致对罕见配置的估计不可靠；②缺乏上下文信息、约束或多目标的处理；③对延迟与删失反馈的理论分析仍不充分；④在大规模高维空间下 oracle 计算成本与探索效率的进一步提升空间。

---

## 661. Hidden Services Protocol for Mixnets

**arXiv ID:** 2609.01326 | [PDF](https://arxiv.org/pdf/2609.01326v1)

**作者:** Nicolas Constantinides `[一作]` (Unaffiliated), Stavros Nonis `[通讯]` (Unaffiliated)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于混合网络（Nym）隐藏服务协议NymHS，解决了传统SURB在单一混合节点攻击下泄露发送方或接收方入口网关的问题；

**💡 创新点**

核心创新点包括：①使用隐藏的payload‑seed机制防止最后混合节点匹配；②引入重新包装与交换头的REPLIER‑ANONYMOUS SURB，防止首节点匹配；③构建完整的隐藏服务发现、会话管理与异步SURB补给协议；

**🔧 技术方法**

技术实现基于Loopix/Nym的Sphinx层、POTP（伪一次性密码）payload加密、三层路由标记、分层加密以及签名认证（Ed25519）等；

**📊 数据集**

实验使用118个静态网页模板（来自公开GitHub仓库），在单台虚拟机上部署混合网络、隐藏服务、仓库节点与客户端代理；

**📈 对比分析**

评估方法为对9×3的payload大小（2–10 KiB）与重包装深度（1–3）进行全因子实验，共计9,558次页面加载；结果显示：将payload从2 KiB增至10 KiB可将平均加载延迟降低5.2倍，通信开销从21.7 %降至4.3 %；

**⚠️ 局限性**

局限性：实验仅在单机、无丢包环境下进行，使用静态页面且未覆盖动态内容；未考察跨地域、真实网络拥塞或高丢包率对性能的影响；

---

## 662. InSight: A Benchmark for Agentic Claim Verification in Interactive Visualizations

**arXiv ID:** 2609.01383 | [PDF](https://arxiv.org/pdf/2609.01383v1)

**作者:** Maeve Hutchinson `[一作]` (City, University of London), Pranava Madhyastha `[通讯]` (City, University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 InSight 基准，用于评估视觉语言模型在交互式可视化环境中主动检索证据以验证自然语言主张的能力。

**💡 创新点**

创新点在于：①将交互追踪视为可观测的推理过程；②构建基于真实交互笔记本的 21k+ 交互式主张；③引入 Interaction Efficiency Score（IES）衡量模型的交互质量；④采用对比变异与 NLI 验证实现平衡的真、假、NEI 标签。

**🔧 技术方法**

技术手段包括：Vega‑Lite 可视化渲染、Playwright 无头浏览器交互、基于鼠标与键盘的高层动作空间、序列决策模型（如 Gemini、GPT‑5.5 等）、自监督文本抽取与分解、NLI 约束的主张变异、IES 评估指标。

**📊 数据集**

使用数据集：来自 297 名专业数据可视化分析师的交互式 Vega‑Lite 笔记本，提取 21,349 条真实主张，涵盖 True、False、NEI 三类。

**📈 对比分析**

对比实验：在 500 条平衡样本上评估 14 种闭源与开源模型，最佳准确率 57.2%（GPT‑5.5），IES 最高 26.98%；相比无交互基线（T=1），交互模型在准确率上提升约 10‑15%，但小规模模型交互率低，表现接近随机。

**⚠️ 局限性**

局限性包括：①动作空间过于抽象，无法覆盖所有人类交互方式；②交互追踪未能完全反映内部推理过程；③模型在多视图与悬停工具提示等细粒度交互上表现欠佳；④数据集仅涵盖 Vega‑Lite，缺乏跨平台多样性。

---

## 663. Polish ModernBERT: The Long and Short of Polish Language Understanding

**arXiv ID:** 2609.01379 | [PDF](https://arxiv.org/pdf/2609.01379v1)

**作者:** Michał Perełkiewicz `[一作]` (National Information Processing Institute), Małgorzata Grębowiec `[通讯]` (National Information Processing Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于 ModernBERT 架构的 Polish ModernBERT 四个编码器（Base/Large，512/8K 上下文），并提出了 LongContext 长文本评测基准。

**💡 创新点**

创新点在于：①采用分阶段预训练策略将 ModernBERT 迁移至波兰语；②通过继续预训练实现 8K 长上下文；③结合翻译与生成的长文本数据构造专门评测基准；④在保持参数更少的同时提升长文本性能。

**🔧 技术方法**

技术手段包括：ModernBERT 架构（旋转位置嵌入、GeGLU、预归一化、全局/局部滑动窗口注意力）、分阶段学习率与掩码策略、SentencePiece 子词分词器、基于 Megatron 的高效实现以及 BF16 混合精度训练。

**📊 数据集**

数据集涵盖：预训练使用 44.5B 词的精炼波兰语语料（精选波兰语语料、Common Crawl、FineTranslations）；微调使用 30 个下游任务（KLEJ、FinBench、Other Tasks、LongContext）以及检索基准 PIRB；LongContext 任务源自 SCOTUS、ECtHR、BookSummary，均经 GLM-4.6 翻译或 LLM 生成。

**📈 对比分析**

在与 HerBERT、Polish RoBERTa、XLM‑R、EuroBERT、mmBERT 等同类编码器的 30 项评测中，Polish ModernBERT 在多数任务中获得最高平均分；在长文本基准中 Base‑8K 领先 RoBERTa‑8K 9.68 分，且参数更少；在检索任务中，Base‑8K 在 300M 以下模型中取得最高 NDCG@10。

**⚠️ 局限性**

局限性包括：仅针对波兰语；评测主要聚焦分类任务，缺乏序列标注、检索、问答等多样化任务；LongContext 数据多为机器翻译与 LLM 生成，可能存在翻译与生成偏差；长文本基准以法律文本为主，领域泛化有限；不同词表导致规模比较不完全可控。

---

## 664. Behaviorally Effective LoRA Writes Are Sparse and Structured

**arXiv ID:** 2609.01374 | [PDF](https://arxiv.org/pdf/2609.01374v1)

**作者:** Haruto Sato `[一作]` (Independent Researchers), Mei Ito `[通讯]` (Independent Researchers)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LoRA适配器的写入空间进行结构化分析，发现写入效果稀疏且结构化，并提出学习基准冻结再训练的方法验证。

**💡 创新点**

引入基于warmup写入的学习基准冻结再训练，并将写入空间视为因果状态变量，证明行为有效的LoRA写入集中在少数后置模块的少数方向。

**🔧 技术方法**

利用正交化写入列构建模块级正交基，执行精确转换、相同状态基础切换、无再训练投影及Top‑k/全局方向稀疏化实验。

**📊 数据集**

在GSM8K、MathQA、AQuA、CommonsenseQA、StrategyQA、ARC‑Challenge等真实推理/选择题数据集上进行实验。

**📈 对比分析**

与冻结基线、随机/ PCA基准及LoRA等对照，使用同一提示/读数规则，发现学习基准方法在大多数任务上与LoRA相当或更优，远优于随机/PCA基准。

**⚠️ 局限性**

基底不唯一、全局稀疏度低于模块级、未给出完整语义解释，且对不同模型/任务的泛化仍有待验证。

---

## 665. Investigating Linear Probe Robustness to Linguistic Register, Medical Specialty, and Corpus Shifts in Medical QA

**arXiv ID:** 2609.01361 | [PDF](https://arxiv.org/pdf/2609.01361v1)

**作者:** Nishant Mishra `[一作]` (Amsterdam UMC, University of Amsterdam), Iacer Calixto `[通讯]` (Amsterdam Public Health)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过在大型语言模型隐藏状态上训练线性探针，系统评估其在医学问答中对写作风格、医学专业和语料库变化的鲁棒性；

**💡 创新点**

创新点在于将三种输入变更（写作风格、专业类别、语料库）分离开来单独检验真相方向的稳健性，并构建4000变体的写作风格重写基准，展示真相方向在风格与专业上相对稳健，却在跨语料库时显著衰退；

**🔧 技术方法**

主要技术包括线性逻辑回归探针、无正则化差均值探针、混合注册训练、Platt与等距回归校准以及对输出侧置信度基线的对比；

**📊 数据集**

使用数据集包括MedQA、MedMCQA、MMLU‑medical、MedRedQA及S‑MedQA专业标注，对500条MedQA问题进行四种写作风格重写生成4000个实例；

**📈 对比分析**

通过AUROC比较，发现写作风格差距Δ_register≈-0.095、专业差距Δ_specialty≈-0.031、语料库差距Δ_dataset≈-0.21；探针相较于输出置信度基线提升6–11 AUROC点；原始探针校准误差高（ECE≈0.36），经过Platt缩减至≈0.16；

**⚠️ 局限性**

局限性包括仅测试多选考试式医学问答、重写生成依赖单一模型、仅覆盖2–8B规模LLM、跨语料库对比混杂考试地区与问题构造差异、校准仍不理想、对自然临床文本与其他语言的泛化未知。

---

## 666. Cheap Verifiers, Large Blind Spots: Measuring the Reliability Cost of Cost-Saving Cascades

**arXiv ID:** 2609.01345 | [PDF](https://arxiv.org/pdf/2609.01345v1)

**作者:** Dushyant Rajput `[一作]` `[通讯]` (AltSlate Labs LLP), Dushyant Rajput (AltSlate Labs LLP)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并测量了低成本推理级联（学生模型 + 验证器）中关闭循环（对验证器拒绝的答案进行微调）对成本和质量的影响，发现验证器的盲点导致真实错误远高于仪表板显示，并且微调往往会破坏学生模型。

**💡 创新点**

提出并验证了“盲点保守定律”，解释了为何循环训练只针对可检测错误会导致误差在盲点中保留，形成一个正向错误底限；揭示仪表板指标与真实性能之间的盲区。

**🔧 技术方法**

采用 LoRA 微调在 Qwen2.5‑Instruct 学生模型上，对 gpt‑4o‑mini 等前沿模型进行验证；构建风洞实验通过人工设置的精确答案对验证器盲点进行测量；理论推导与合成实验验证定律。

**📊 数据集**

使用 GSM8K 与 Hard MATH（level 4‑5）作为任务数据集，Oracle 为 exact‑match / symbolic equivalence；学生模型规模从 0.5B 到 32B，验证器为 gpt‑4o‑mini、gpt‑4.1、gpt‑5‑mini。

**📈 对比分析**

对比方法：在相同学生、相同任务下，使用强大验证器替代原始验证器、对比冻结（不微调）和对比自我训练/校正微调；发现仪表板误差始终约 3% 而真实误差可升至 32%；成本提升回到前沿模型价格。

**⚠️ 局限性**

限制包括：1) 在真实 LLM 上循环微调未能提升学生，只导致退化；2) 盲点测量依赖于可获得精确答案的任务，无法直接推广到完全模糊任务；3) 合成实验假设固定输入空间盲区，真实验证器盲点随学生错误分布变化；4) 单一种子实验可能缺乏泛化；5) 研究仅在 Qwen2.5 与 OpenAI 验证器之间进行，未覆盖其他模型族。

---

## 667. Connectome-Based Modelling Reveals Orientation Maps in the Drosophila Optic Lobe

**arXiv ID:** 2609.01330 | [PDF](https://arxiv.org/pdf/2609.01330v1)

**作者:** Jia-Nuo Liew `[一作]` (Tsinghua University), Xiaolin Hu `[通讯]` (Tsinghua University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

利用完整的果蝇脑连接组和基于 LIF 的神经元模型，对光线刺激下的视网膜到中枢神经元的全脑模拟，揭示了在无皮层的中枢视觉系统中出现的空间有序方向选择地图与列状结构。

**💡 创新点**

首次在无皮层系统中计算证明了方向选择的空间映射，显示不同物种可通过相似的连接模式实现功能性视觉映射，揭示了进化趋同的计算原理。

**🔧 技术方法**

采用漏磁感应整流模型（LIF）对138,639个神经元、1,508,983个突触进行大规模离散时间仿真，结合高斯拟合、空间平滑、方向预测与列状结构分析等多种计算方法。

**📊 数据集**

使用FAFB（Full Adult Fly Brain）完整成人果蝇脑连接组作为结构数据，结合光条刺激生成的泊松脉冲输入。

**📈 对比分析**

通过与已知的结构预测角度、Seung 等模型的比例（40% 以上阈值）进行对比，评估模型中方向选择的比例和角度误差；同时在不同层（Dm、Pm）展示平滑后的方向地图和钉点特征，表明模型能够重现预期的空间一致性。

**⚠️ 局限性**

缺乏实验验证，使用的 LIF 模型简化了神经元的非线性动力学和神经递质影响，可能影响网络行为；此外仅基于结构预测，未考虑可塑性与环境因素。

---

## 668. VerTox: Verifiable Reward-Guided Corpus Poisoning Against Neural Ranking Models

**arXiv ID:** 2609.01325 | [PDF](https://arxiv.org/pdf/2609.01325v1)

**作者:** Zhiqi Huang `[一作]` (Capital One), Alfy Samuel `[通讯]` (Capital One)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了对神经检索模型的语料库中毒攻击，并提出了一种基于可验证奖励的强化学习框架来训练LLM生成流畅且能误导检索结果的对抗文本。

**💡 创新点**

创新点在于将语料库中毒定义为可验证奖励的RL任务，并设计了结合排名失真、事实腐败与查询重复惩罚的多项奖励，使攻击既高效又保持可读性。

**🔧 技术方法**

使用的技术包括GRPO优化、可验证奖励设计、LoRA微调的LLM、代理检索器SimLM、NLI驱动的事实一致性检测以及多种对抗生成基线。

**📊 数据集**

使用的数据集包括MS MARCO（训练），TREC DL 2019/2020、BEIR（NQ、FiQA、Touché、TREC‑COVID、SciFact、NFCorpus）等评测集，以及FlashRAG检索语料。

**📈 对比分析**

与随机词替换、HotFlip、EmbedPerturb、直接提示等基线相比，本文方法在白盒下近乎100%攻击成功率、黑盒跨模型迁移良好，并在RAG任务中将答案准确率从约70%降至30%，性能显著优于现有方法。

**⚠️ 局限性**

局限性包括仅针对开放检索场景、未评估过滤或对抗训练等防御措施、缺乏大规模事实性与可检测性评估，以及仅使用有限的RAG任务与评测者。

---

## 669. Exploring Sparse Autoencoders in Text-Based Causal Confounding Adjustment

**arXiv ID:** 2609.01322 | [PDF](https://arxiv.org/pdf/2609.01322v1)

**作者:** Mian Zhong `[一作]` (Johns Hopkins University), Anjalie Field `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了利用稀疏自编码器（SAE）对文本进行表示并进行因果调整，构建了迭代选择特征的完整因果分析管道。

**💡 创新点**

创新点在于将SAE与条件独立检验结合，实现最小化调整集合同时保持可解释性；并首次提出多标签半合成评估框架来检验方法在更复杂文本因果场景下的鲁棒性。

**🔧 技术方法**

使用的技术包括预训练文本嵌入、Top‑K稀疏自编码器、Lasso逻辑回归特征选择、条件独立检验、CEM匹配与DoubleML估计器，以及LLM对SAE特征的解释性分析。

**📊 数据集**

使用的数据集为20NewsGroups（单标签）和EURLEX（多标签多语种），并在这些数据上构建半合成实验。

**📈 对比分析**

与基线TIRM（STM+匹配）和Embed+DoubleML比较，SAE在二元、多类以及多标签场景下通常表现出更低的偏差、更低的RMSE以及更高的置信区间覆盖率；匹配保留率与覆盖率表现因数据和模型而异。

**⚠️ 局限性**

局限性包括对SAE超参数（M、K）的敏感性；多标签设置下的性能波动较大；在有限样本下DoubleML的收敛性差；以及方法主要在半合成设置下验证，缺乏真实数据的深入检验。

---

## 670. Lifted-Product QLDPC Codes in the Polynomial Domain

**arXiv ID:** 2609.01305 | [PDF](https://arxiv.org/pdf/2609.01305v1)

**作者:** Vahid Nourozi `[一作]` (New Mexico State University), David G. M. Mitchell `[通讯]` (New Mexico State University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了有限长度的多项式域上提升乘积QLDPC代码构造方法，利用商环 𝔽₂[D]/(Dᴸ+1) 进行矩阵升阶，并通过共轭兼容的循环升格保证 CSS 正交性。

**💡 创新点**

创新点在于将提升乘积构造完全迁移到多项式域，提供了一个可直接在代数层面检验正交性的框架；同时引入共轭兼容的循环升格，首次实现了在升格前就能确认 H_X H_Zᵀ=0 的性质。

**🔧 技术方法**

采用多项式环、商环、共轭（逆元）运算、Kronecker 积、循环升格映射以及 BP、QBPGD、RL‑S 等量子 BP 解码器；同时通过 3×4 基矩阵与 NASA 约束长度 7 的卷积码实例进行代码构造。

**📊 数据集**

使用模拟的 depolarizing 通道作为评估数据集；构造的代码来自 3×4 基多项式矩阵（和 NASA 卷积码）升格到不同 L 值，生成 N=250–2075 的结构化 QLDPC 码。

**📈 对比分析**

与 QBP、QBPGD、RL‑S 三种 BP 解码器在 depolarizing 通道上进行比较。结果表明 RL‑S 在迭代预算较大的情况下性能最佳，QBP 与 QBPGD 在高错误率区间出现严重错误阶；整体而言构造的码在可行的迭代预算下能实现较低的块错误率。

**⚠️ 局限性**

仅针对有限长度，未给出渐近性能分析；升格后的图中存在大量 4‑环，导致 BP 解码器表现受限；解码复杂度较高，特别是 QBPGD 的多轮降维过程；未来需进一步优化基矩阵、距离搜索与解码调度。

---

## 671. Relational Task Generation Language: A Declarative Specification Framework for Relational Deep Learning

**arXiv ID:** 2609.01292 | [PDF](https://arxiv.org/pdf/2609.01292v1)

**作者:** Oleksii Kolesnichenko `[一作]` (Czech Technical University in Prague), Gustav Šír `[通讯]` (Czech Technical University in Prague)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个开源声明式语言 RTGL，用于简化关系深度学习中的任务生成并防止时间泄漏。

**💡 创新点**

创新点在于提供可声明的任务定义，自动化时间窗口校验，开放源代码可与现有 RDL 框架集成，并通过 RTGL 重构并纠正了 RelBench benchmark 的错误。

**🔧 技术方法**

使用 ANTLR4 解析器、AST 验证、SQL 转换；结合图神经网络（GraphSAGE, HGT）训练模型。

**📊 数据集**

使用 RelBench（rel‑f1, rel‑stack）以及 ReDeLEx 的 Seznam 数据集。

**📈 对比分析**

通过对比 RTGL 生成的任务表与原 RelBench 手工 SQL 的差异验证正确性；在重构任务和新任务上训练 GraphSAGE/HGT，均能收敛且性能符合预期，GraphSAGE 有时优于 HGT。

**⚠️ 局限性**

目前无法天然处理多跳关系，需手动创建视图；缺乏大规模性能优化。

---

## 672. When Guardrails Look Effective: Construct Validity Failures in LLM Agent Commerce Evaluation

**arXiv ID:** 2609.01519 | [PDF](https://arxiv.org/pdf/2609.01519v1)

**作者:** Peiying Zhu `[一作]` (Blossom AI), Sidi Chang `[通讯]` (Blossom AI Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过对LLM买卖双方的多轮对话模拟进行审计，评估平台两种“护栏”对交易结果的影响；

**💡 创新点**

提出并验证构造效度审计合同，系统化检查代理激励、协议隔离、随机稳定性与福利核算，揭示原始护栏效应多源于构造误差而非经济机制；

**🔧 技术方法**

使用Qwen2.5 LLM（1.5B、3B、14B）进行对话生成、对话重放，结合Bootstrap置信区间、方差分解、签名翻转置换检验等统计方法；

**📊 数据集**

采用30个合成酒店交易配置文件（共60个完整配置）作为实验数据集；

**📈 对比分析**

对比未修正与统一方案的四个护栏组合，使用Bootstrap和方差分解评估效应；原始护栏效应在统一方案下显著衰减，单次生成的极端收益被多次生成抑制，最终效果不显著；

**⚠️ 局限性**

仅使用单一模型家族及其4‑bit量化版本，样本为合成数据，缺乏跨模型与跨人类代理的验证，重复生成分析受限于少量样本，护栏对真实市场行为的适用性尚不确定。

---

## 673. DualDiff3D: Dual Structure-Appearance Diffusion Priors for Reliability-Enhanced 3D Gaussian Splatting

**arXiv ID:** 2609.01516 | [PDF](https://arxiv.org/pdf/2609.01516v1)

**作者:** Qian Wang `[一作]` (Peking University), Jian Zhang `[通讯]` (Peking University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 DualDiff3D 框架，利用双结构-外观扩散先验（DualDiff）对 3D 高斯喷射（3DGS）产生的低质量视角进行去伪、细节补全，随后通过可靠增强的 Render‑Refine‑Optimize 循环逐步提升 3D 重建与新视角合成的质量。

**💡 创新点**

创新点：① 双分支扩散网络将结构提取与外观一致性分离，并通过 Structure‑Appearance Attention (SAA) 在自注意力层实现信息融合；② 将 DualDiff 作为“免费加分”应用于现有参考视角修复模型；③ 设计了 Progressive Sampling and Filtering (PSF) 与 Confidence‑Driven Weighting (CDW) 的 RRO 循环，使得修复视角在训练中可被安全、有效地加入 3DGS；④ 在稀疏视角条件下，已实现显著的 PSNR/SSIM 提升，且无需额外训练即可获得 0.7 dB PSNR 提升。

**🔧 技术方法**

核心技术包括 3D Gaussian Splatting、单步 DDPM 扩散模型（SD‑Turbo 变体）、VAE 编解码、LoRA 微调、SAA 机制、PSF 与 CDW 的多信任度掩码、以及基于 LPIPS、Gram‑Matrix 等损失的训练策略。

**📊 数据集**

主要使用的公开数据集：DL3DV、LLFF 以及 Mip‑NeRF360；实验中对 DL3DV 的 10% 测试集以及 LLFF 的 3、6、9、24 视角场景进行评估。

**📈 对比分析**

与 3DGS、DIFIX3D、3DGS‑Enhancer、GenFusion、GSFixer 等现有方法对比，DualDiff3D 在 PSNR、SSIM、LPIPS/FID 指标上均取得最高或相近最佳结果；在 DL3DV 与 LLFF 的多视角实验中，PSNR 提升 1–3 dB，SSIM 提升 0.05–0.1，且在 1–2 秒/帧的推理时间下完成修复。

**⚠️ 局限性**

局限性：① 仍需较大显存（≈12.5GB）和 GPU 计算，推理不适合实时渲染；② 在极稀疏视角（如 3 视角以下）下性能下降明显，需进一步优化模型压缩与稀疏视角适配。

---

## 674. TempCloze: Can Video-LLMs Identify the Missing Middle?

**arXiv ID:** 2609.01515 | [PDF](https://arxiv.org/pdf/2609.01515v1)

**作者:** Wenqi Pei `[一作]` (University of Hong Kong), Hongyang Du `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布了 TempCloze 视觉缺失中间片段 Cloze 基准，用以评估 Video‑LLM 的视觉时序推理能力。

**💡 创新点**

通过构造同源干扰项并将任务拆分为语义、对齐、进展三维度，消除语言捷径，首次揭示 Video‑LLM 在时序对齐上的瓶颈。

**🔧 技术方法**

结合视频过滤、LLM 语义筛选、光流运动筛选和同源干扰生成等技术，并对 10 家专有与 21 家开源 Video‑LLM 进行评测与错误模式与行为敏感性分析。

**📊 数据集**

采集自七大公开源（包括长拍、第一人称等）的 1,521 条视频，构成原始、混合与最难子集，构成 TempCloze 基准。

**📈 对比分析**

采用候选集选择评测准确率；结果显示模型在语义与进展维度表现良好，但在对齐维度准确率最低；错误模式与敏感性分析显示模型对候选顺序、上下文方向和帧密度等因素敏感，测试时缩放对整体排序影响有限。

**⚠️ 局限性**

仅评估缺失中间片段的视觉时序匹配，未覆盖开放式生成、对话、音频推理等；同源干扰设计严格，侧重时间精确匹配，可能不代表泛化能力；数据仅来自长拍、第一人称等域，缺少其他视角与场景的覆盖。

---

## 675. LatentPress: Context Compression Beyond Text and Vision

**arXiv ID:** 2609.01507 | [PDF](https://arxiv.org/pdf/2609.01507v1)

**作者:** Zhengze Zhou `[一作]` (Cornell University), Hejian Sang `[通讯]` (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出LatentPress，利用小型适配器把对话历史或长文档压缩为连续软词，再直接喂入冻结的语言模型解码器，无需文本重构。

**💡 创新点**

核心创新在于：① 只训练少量适配器（≈0.1%参数）即可写入软词；② 软词直接投射至解码器输入嵌入层，保持解码器冻结；③ 采用基于角色或均匀的手工压缩率，支持不同结构段的压缩。

**🔧 技术方法**

使用冻结的LLM（Qwen2.5‑7B/8B、Qwen3‑8B 等）解码器；编写器基于解码器底层层级；训练目标结合重构损失和前向 KL，支持任务适应。

**📊 数据集**

主要数据集包括：UltraChat（对话训练集）、LongMemEval（对话记忆 QA）、LongBench‑QA（长文档 QA），并对比 OCR 与文本摘要等基线。

**📈 对比分析**

与原始文本、OCR、文本摘要等对比，压缩 4–8× 时保持或提升准确率（如 LongMemEval 上 0.504 vs 0.490），写入时间 43 ms，推理速度比原始/缓存 OCR 快 5–9×，整体任务耗时显著下降。

**⚠️ 局限性**

局限：压缩率固定为手工规则，缺乏动态调节；仅在冻结解码器上测试，未集成检索或更新；在高压缩率下仍可能丢失细节，需进一步研究动态压缩策略。

---

## 676. The Data Problem in Software Vulnerability Analysis: Artifacts, Quality, and Consumption

**arXiv ID:** 2609.01503 | [PDF](https://arxiv.org/pdf/2609.01503v1)

**作者:** Yu Nong `[一作]` (Oakland University), Haipeng Cai `[通讯]` (University at Buffalo, SUNY)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过构建数据中心化的分类体系，对2016–2026年间1,522篇论文及111篇关键论文进行深度编码，系统评估了软件漏洞分析所用数据的类型、质量属性（真实性、标签证据、规模、多样性、泄露与可用性）及其消费方式，形成了“证据金字塔”与“artifact×quality矩阵”，并提出面向证据丰富数据的研究议程。

**💡 创新点**

创新点在于①提出了区分数据形式、质量与用途的三维分类法；②以“证据金字塔”对六类漏洞数据（元数据、代码样本、补丁、PoC/测试、推理、追踪）进行层级化，揭示不同层级在真实性与标签证据上的系统差距；③构建了111篇anchor论文的深度编码库，并为每一正面属性提供直接引用，保证了研究可审计性；④归纳了现有数据集的六大质量属性与七大缺陷，形成了面向证据丰富漏洞数据的研究路线。

**🔧 技术方法**

采用系统文献检索（arXiv、DBLP、Semantic Scholar）、数据集层级化筛选、两阶段阅读协议（深度编码与轻量标签），结合手工编码与LLM辅助提取，利用闭合词表和标准化编码规则，对每篇anchor论文进行定量统计与质量评估。

**📊 数据集**

主要参考的数据集包括Devign、Big‑Vul、CVEfixes、PatchDB、PrimeVul、VulnRepairEval、SEC‑bench、Magmar、Vulnerability‑R2、Vader、SmartCoder、PBFuzz、SecCodeBench‑V2等，涵盖代码样本、补丁、PoC/测试、推理与追踪等多种形式。整体语料涵盖1,522篇论文，111篇anchor论文共计3,000+个质量评估标记。

**📈 对比分析**

本文并未提出新的算法或模型，而是通过定量统计对不同数据类型与质量属性进行横向对比：例如，执行型数据（T4）在真实性和标签证据上优于代码样本（T1）；补丁型数据（T3）在标签证据上相对可靠但在真实性上不足；推理与追踪数据在规模和泄露控制方面表现更好。性能评估主要体现在“质量达成率”上，T4数据的真实性达成率为66%（16/24），标签证据达成率为96%（23/24）。

**⚠️ 局限性**

局限性包括①缺乏真实环境下可执行的、完整上下文的代码样本（仅1/41具备真实性）；②标签证据大多依赖于补丁推断或静态分析，误差高达30%+；③泄露与污染控制未在超过一半数据集内实施；④推理与追踪数据量仍极小，仅有3–8个深度编码样本；⑤对可用性评估不足，公开率仅占大部分类型的40–50%；⑥由于数据集数量多且分散，缺乏统一的评估基准和标准。

---

## 677. Optimizing Byzantine Node Placement in Decentralized Federated Learning

**arXiv ID:** 2609.01495 | [PDF](https://arxiv.org/pdf/2609.01495v1)

**作者:** Edoardo Gabrielli `[一作]` (Sapienza University of Rome), Gabriele Tolomei `[通讯]` (Sapienza University of Rome)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在分布式联邦学习中研究并优化拜占庭节点的放置，以最大化在有限训练轮次内对诚实节点的影响

**💡 创新点**

提出基于有限时间传播的拜占庭放置影响度量（BPI），并给出贪心+局部搜索的高效优化算法，证明其在多种网络拓扑、攻击目标和鲁棒聚合下均能显著提升攻击效果，从而揭示放置是DFL安全评估的关键维度

**🔧 技术方法**

利用线性 gossip 传播模型计算 BPI，使用贪心 + 1 交换局部搜索算法进行优化；在实验中对比随机、度中心、介数、特征向量、MaxSpAN-FL 等基线，并在 MNIST 与 BadNets backdoor 数据集上评估；采用 Metropolis/AVG 等聚合方式

**📊 数据集**

MNIST（无目标攻击）和 BadNets backdoor 数据集

**📈 对比分析**

通过在六种不同拓扑（Ring‑of‑Cliques、Dragonfly、Scale‑Free、DC‑SBM、Core‑Periphery、Random‑Geometric）上对比各种放置策略，BPI‑guided 在准确率下降或 ASR 上均达到了或逼近最优性能，表明其在所有网络结构上都更具攻击力；在使用非线性鲁棒聚合时，BPI 仍能显著提升攻击效果

**⚠️ 局限性**

仅针对静态已知拓扑、线性 gossip 传播模型；对时间变化或未知拓扑、局部/动态攻击的适应性不足；BPI 作为近似代理，在某些攻击方向或鲁棒聚合下可能不是最优解

---

## 678. Parsing the Stream: A Live Trace Model for Long-Horizon Agents and Their Observers

**arXiv ID:** 2609.01466 | [PDF](https://arxiv.org/pdf/2609.01466v1)

**作者:** Egor Pakhomov `[一作]` (Salesforce AI Research), Erik Nijkamp `[通讯]` (Salesforce AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种“实时轨迹折叠”模型，利用单一事件账本将长时间代理运行的日志折叠为可定界的、可审计的运行状态，并从中编译出针对人类观察者和代理自身的视图。

**💡 创新点**

创新点在于：①将原始轨迹与两类消费者的需求统一到同一折叠逻辑；②通过增量、可恢复、基于类型的折叠实现确定性和可审计性；③在折叠过程中实现显著的 token 与成本压缩，同时保持与完整上下文等价的准确率。

**🔧 技术方法**

采用的技术包括：Append‑Only JSONL 事件日志、增量折叠成 RunState、版本化派生节点、基于视图的编译器、LLM 读者代理、缓存感知的解析器以及实验性费用与缓存计量。

**📊 数据集**

使用的数据集包括 12 篇真实 112 MB 代理会话的轨迹、由代码可重现的 12 篇合成轨迹（约 11.4 MB）以及基于链式、散点、替代链等任务族的 benchmark 序列（最多 120 链）。

**📈 对比分析**

实验结果表明：对人类观察者而言，编译视图的准确率提升至 0.85–0.87，token 使用和成本分别减少约 5–7 倍；对代理而言，折叠视图在 120 链任务中 30/30 成功率，成本仅为完整上下文的约 1/4，且提供可审计的状态。

**⚠️ 局限性**

局限性包括：样本量有限（大多为 n=10 或更小）、基准与系统共进化导致可能的偏倚、仅在单一供应商的 LLM 环境下测试、未覆盖多会话、多代理、检索等场景，以及未评估安全与信任边界。

---

## 679. Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search

**arXiv ID:** 2609.01431 | [PDF](https://arxiv.org/pdf/2609.01431v1)

**作者:** Zhiliang Chen `[一作]` (Meta), Jihao Andreas Lin `[通讯]` (Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Power‑Law Entropy Search (PLES) 的成本感知采样方法，用于高效估计大语言模型训练的最优超参数缩放律；

**💡 创新点**

创新点在于将多保真度贝叶斯优化与信息增益最大化相结合，设计出新的 PLES 获得函数，利用高斯过程与汤普森采样实现对缩放律参数的不确定性显著降低，并且能同时推断多重超参数的缩放律；

**🔧 技术方法**

采用高斯过程作为代理模型，对模型损失随模型/数据规模与超参数的关系建模；利用 Bayesian 线性回归从 GP 采样估计幂律系数；通过汤普森采样实现对最优超参数的期望估计；并在采样候选时考虑计算成本，形成成本调节因子 d；实现时使用 BoTorch 并行化；

**📊 数据集**

实验数据包含三类：一是自定义的合成损失函数；二是基于历史 LLM 训练日志拟合的 GP surrogate（主要来自 Llama‑8B 的网格搜索结果）；三是真实的 Llama‑8B 预训练实验，改变模型规模 N、数据量 D、学习率等超参数；

**📈 对比分析**

与传统网格搜索、阶梯式贝叶斯优化 (Ladder BO) 和 Sobol 随机抽样三种基线在相同总计算预算下比较；PLES 在 held‑out 规模下的超参数百分比误差更低，收敛速度至少是基线的十分之一，最终模型损失也更低；

**⚠️ 局限性**

局限性在于假设最优超参数缩放律为幂律函数，若真实规律偏离此形式，则估计可能出现偏差；目前不支持更灵活的函数形式，需进一步扩展。

---

## 680. CATeye: Coupled Attribute-Topology Invariance Learning for Voucher Abuse Detection

**arXiv ID:** 2609.01425 | [PDF](https://arxiv.org/pdf/2609.01425v1)

**作者:** Tian Tian `[一作]` (Nanyang Technological University), Zhiqi Shen `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个针对电子商务代金券滥用检测的无监督领域泛化框架（CATeye），能够在不重新训练的情况下实现跨时间、跨地区的零样本推理。

**💡 创新点**

核心创新点在于：① 引入属性不变性选择器（AIS）和边不变性选择器（EIS）两级可学习掩码，自动分离节点属性与图结构中的可变与不可变成分；② 构造多视图（完整不变、部分不变、两类不变）并采用对比、交叉熵与熵最大化等视图特定目标，强化不变表示并抑制环境特定信号，从而克服属性–拓扑耦合的分布漂移。

**🔧 技术方法**

技术手段包括：基于GraphSAGE的图神经网络编码器；Gumbel-Softmax离散化的可学习二进制掩码；对比损失实现跨域相似性；交叉熵与熵最大化实现监督与不确定性约束；以及多任务联合优化。

**📊 数据集**

使用了两大数据集：① Lazada的自有代金券滥用图数据（覆盖印尼与越南多天、多地区的订单图，节点属性约800维）；② 公共Elliptic比特币交易图数据（按时间片段划分的交易网络，用于检验时序域泛化）。

**📈 对比分析**

与九个强基线（ERM、IRM、IB-IRM、V-REx、DANN、TRACI、EERM、CaNet、AugAN）对比，在Lazada数据上平均F1提升至81.37%，比最优基线高2.81%；在Elliptic数据上平均F1提升至76.84%，比第二佳方法高8.61%；在多场景（促销日、非促销日、不同时间跨度）下表现均稳健。

**⚠️ 局限性**

局限性包括：① 对属性和结构的掩码分离仍依赖超参数（掩码比例、权重系数）且在极端分布漂移下可能需要重新调参；② 需要较为丰富的手工特征构造与图生成规则，适配不同业务时成本较高；③ 计算复杂度相对传统方法略高，尤其在大规模订单图时对GPU内存和推理速度有一定要求。

---

## 681. Provably Safe Sim-to-Real Transfer

**arXiv ID:** 2609.01418 | [PDF](https://arxiv.org/pdf/2609.01418v1)

**作者:** Tingting Ni `[一作]` (École Polytechnique Fédérale de Lausanne), Maryam Kamgarpour `[通讯]` (École Polytechnique Fédérale de Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种安全的sim-to-real迁移算法，利用模拟器引导真实世界数据采集，实现奖励自由安全规划；

**💡 创新点**

创新点在于将奖励自由安全RL与可辨别的mismatch region 相结合，构造可计算的安全探索策略，并给出样本复杂度上界；

**🔧 技术方法**

采用了CMDP框架、置信界估计、混合策略、混合模型的动态规划与安全约束下的贝尔曼值奖励自由规划技术；

**📊 数据集**

使用了5×5网格世界（gridworld）作为实验数据集；

**📈 对比分析**

通过与奖励自由安全RL和无约束sim-to-real baseline比较，算法在保证安全的前提下，样本复杂度仅比无约束方法高约2倍，远优于奖励自由方法；

**⚠️ 局限性**

局限性：仅适用于离散tabular CMDP，需满足Slater条件与σ分离假设，难以直接扩展至连续或不满足分离的情形。

---

## 682. EdiTikZ: Scientific Figure Editing from Revision Trajectories

**arXiv ID:** 2609.01409 | [PDF](https://arxiv.org/pdf/2609.01409v1)

**作者:** Christian Greisinger `[一作]` (University of Technology Nuremberg), Steffen Eger `[通讯]` (University of Technology Nuremberg)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于科学论文和代码仓库的TikZ编辑数据集DaEdiTikZ，并在其上训练了可编辑VLM模型EdiTikZ。

**💡 创新点**

创新点在于利用真实的科学图形修订轨迹作为监督，结合图形重建与编辑的联合训练，以及使用多奖励（渲染相似度与指令执行度）进行GDPO强化学习，显著提升编辑质量。

**🔧 技术方法**

技术包括：VLM（Qwen3.5、Qwen3.6）、图像编码器、TikZilla前处理、对齐与重建损失、GDPO多奖励强化学习、代码与渲染相似度评估。

**📊 数据集**

数据集：DaEdiTikZ（391K TikZ编辑对，781K指令），DaEdiTikZ-Bench（790人工校准实例），以及对SPIQA和CharXiv的OOD评估。

**📈 对比分析**

与多款商业与开源VLM进行自动与人工评估，EdiTikZ‑9B‑RL在自动指标上优于所有基线，在人工评估中仅落后Gemini‑3.1‑Pro，超过GPT‑5.6‑Sol；在OOD长序列生成中仍保持竞争力。

**⚠️ 局限性**

局限包括：指令生成噪声（遗漏、误解）、对长序列OOB生成的性能衰退、人工评审依赖于合成指令、未覆盖非TikZ图形编辑。

---

## 683. Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation

**arXiv ID:** 2609.01511 | [PDF](https://arxiv.org/pdf/2609.01511v1)

**作者:** Lucas Cunha `[一作]` (Pontifical Catholic University of Parana), Rayson Laroca `[通讯]` (Pontifical Catholic University of Parana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个标准化的面部伪造检测基准，评估不同模型在清洁和受损图像上的鲁棒性；

**💡 创新点**

通过系统对比多种模型家族和多种空间/频域输入表示，揭示了清洁数据表现不一定能反映受损条件下的性能，并证明冻结的自监督特征在受损场景下具有显著优势；

**🔧 技术方法**

使用了卷积网络（Xception、ResNet-18、MobileNetV3）、视觉变换器（ViT、CLIP风格编码器）以及冻结的自监督DINOv3背骨，并对RGB、频域（log-magnitude、相位、复数谱、高通）以及空间-频域混合表示进行实验；

**📊 数据集**

采用了MFFI数据集，包含50多种伪造技术、不同来源的真实图像以及多级压缩、缩放、模糊等真实世界降质操作；

**📈 对比分析**

在MFFI的清洁和受损测试集上进行对比，结果显示Xception在清洁集上取得最高AUC 0.884，但在受损集下降显著；冻结的DINOv3在受损集上保持较好性能，AUC 0.726，几乎匹配最优的有任务专门训练的基线；混合空间-频域输入能在卷积模型中提升受损集性能；

**⚠️ 局限性**

局限性包括：仅使用单一数据集（MFFI）限制了跨数据集泛化评估；大多数模型仅从零开始训练，未检验大规模预训练的潜力；DINOv3仅作为冻结特征提取器使用，未探索微调后的效果；归因分析为定性且样本有限，且对冻结变换器的Grad‑CAM处理存在近似性。

---

## 684. Data-Driven Case Study of gNB Placement Optimization in a Private Indoor 5G Testbed

**arXiv ID:** 2609.01510 | [PDF](https://arxiv.org/pdf/2609.01510v1)

**作者:** Diogo de O. Soares `[一作]` (Federal University of Ceara), J. Pedro B. Lima `[通讯]` (Atlantic Institute)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并验证了一套基于实验测量的5G室内gNB放置优化框架；

**💡 创新点**

创新点在于利用LightGBM构建的基于距离和墙体数量的传播模型，结合组合搜索实现多目标优化；

**🔧 技术方法**

使用了LightGBM、KNN、AdaBoost、GBR、RF等机器学习回归器以及组合搜索算法；

**📊 数据集**

使用了在两次测量实验中收集的280个RSRP样本（包含距离与墙体信息）的实验数据集；

**📈 对比分析**

通过与KNN、AdaBoost、GBR、RF等模型对比，LightGBM在RMSE≈7dB时表现最佳，覆盖率和尾部指标随gNB数目提升显著；

**⚠️ 局限性**

局限性包括未考虑干扰与容量约束，仅基于距离和墙数特征，且实验环境较小，泛化到更大规模需验证。

---

## 685. Rethinking Learnability in Offline Data-driven Optimization

**arXiv ID:** 2609.01493 | [PDF](https://arxiv.org/pdf/2609.01493v1)

**作者:** Chao Qian `[一作]` (Nanjing University), Ke Xue `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究离线数据驱动黑盒优化的可学习性问题，提出了算法相关可学习性（algorithm‑dependent learnability）理论，并在此基础上构建了轨迹学习框架，最终提出UGTL方法并在Design‑Bench与BBOB任务上取得领先性能。

**💡 创新点**

创新点包括：①提出算法相关可学习性概念，并证明其在贪婪搜索、局部搜索及投影梯度下降中的充分性；②基于该理论设计轨迹学习框架，提出不确定性感知梯度引导轨迹构造、条件扩散建模与聚类筛选相结合的UGTL方法；③通过理论与实验验证，证明仅需在优化轨迹上学习即可实现高质量离线优化。

**🔧 技术方法**

技术手段涵盖：理论分析（价值查询与梯度查询的可学习性定义）、轨迹构造（使用代理梯度与不确定性加权采样）、条件扩散模型用于轨迹建模、聚类+筛选的候选生成，以及设计Bench与BBOB基准实验。

**📊 数据集**

使用的数据集包括：Design‑Bench提供的5个任务（Ant、D'Kitty、Superconductor、TF‑Bind‑8、TF‑Bind‑10）以及BBOB的Rastrigin与Rosenbrock等测试函数。

**📈 对比分析**

与24种基线方法（传统优化、逆向/条件生成、正则化代理、轨迹信息优化、生成轨迹方法）进行对比。UGTL在5个Design‑Bench任务中平均排名第3.1/25，获得最高100th百分位归一化得分；在BBOB控制实验中UGTL排名第一，表现优异。

**⚠️ 局限性**

局限性包括：①对轨迹构造参数的敏感性，虽大部分参数可共享但仍需微调；②理论和方法主要针对单目标离线优化，未涵盖多目标或更复杂场景；③对高维连续任务的计算成本与可扩展性仍需进一步验证。

---

## 686. Faster Convergence of Multidimensional Approximate Agreement via Smallest Enclosing Balls

**arXiv ID:** 2609.01490 | [PDF](https://arxiv.org/pdf/2609.01490v1)

**作者:** Darya Melnyk `[一作]` `[通讯]` (Technische Universitaet Berlin), Darya Melnyk (Technische Universitaet Berlin)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种新的多维拜占庭近似一致协议，利用安全区域（safe area）中最小包围球（smallest enclosing ball）的中点来更新各节点的向量，能够在同步和异步网络下以收敛率1/√2收敛到满足ε-Agreement的解。

**💡 创新点**

核心创新是将最小包围球的中点作为安全区域内的点来选取，既保证凸有效性（convex validity）又实现了收敛率1/√2；并证明该收敛率在两种通信模型下是渐近最优的；此外给出了同步与异步下的安全区域构造、证明所有安全区域必定相交以及基于全局视图的收敛性分析。

**🔧 技术方法**

使用几何算法（最小包围球计算、凸包交集、Helly定理等）与分布式算法技术（可靠广播、Gather原语、同步/异步一致性广播），并结合圆形几何和Pythagoras定理推导收敛率。

**📊 数据集**

本工作为理论分析，没有使用实际数据集；所有证明和实验均基于数学构造与模型假设。

**📈 对比分析**

与已知算法（MidExtremes、VG、Mendes-Harvey等）在收敛率上做对比，表明在同步模型下收敛率为1/√2，比MidExtremes的√(7/8)≈0.935更快；在异步模型下也保持同一收敛率。迭代次数为O(log₂(1/ε·c·(C)))，即每轮半衰减。

**⚠️ 局限性**

局限性包括：收敛率仍未达到一维最优的1/2，存在1/2到1/√2的间隙；算法依赖于已知的t上界、节点数n满足n>max(3,d+1)t（同步）或n>max(3,d+2)t（异步）；需要可靠广播或Gather原语，假设已认证通道；在动态网络或网络无关模型下仍需进一步研究。

---

## 687. Mobile Backscatter Communication for the Battery-less Internet of Things

**arXiv ID:** 2609.01465 | [PDF](https://arxiv.org/pdf/2609.01465v1)

**作者:** Weining Song `[一作]` (Uppsala University), Luca Mottola `[通讯]` (Politecnico di Milano)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一个轻量级的回声散射通信系统，针对电池无源移动IoT设备动态决定何时发送数据，并利用非易失性存储器(NVM)持久化未发送的数据以应对时变信道和能量波动。

**💡 创新点**

结合RSSI趋势识别与能量状态，动态调节传输激进度；使用指数移动平均(EMA)预测短期信道趋势；利用NVM既存储状态又缓存包；与传统仅基于瞬时信道的自适应速率系统相比，显著提升吞吐量并降低能耗。

**🔧 技术方法**

采用MSP430FR5969 MCU与FRAM、TI CC1352P7 RSSI采集、LoRa标签、RTCs、功率管理电路；实现EMA滤波、非易失性存储、基于能量和信道趋势的动态传输控制；通过仿真框架复现真实信号与能量轨迹。

**📊 数据集**

20条室内信号轨迹（不同通道、路径、速度），两条持续1小时的人跑步能量采集轨迹。

**📈 对比分析**

与Rate‑adaptive和Rate‑adaptive+NVM基线进行对比，采用仿真统计1小时内完整传输包数；实验结果显示最大吞吐量提升5.16×，传输能耗降低47.3%，额外能耗仅0.23–7.3%。

**⚠️ 局限性**

在缺乏明显信道趋势或弱信道持续时间短的场景中收益有限；当信道持续弱且能量充足时可能产生大量缓存包导致延迟；轻量级应用中额外能耗相对显著。

---

## 688. Better Situational Awareness in AR-HRC? A Comparative Study of Augmented Reality and Mobile Interfaces for Human-Robot Collaboration

**arXiv ID:** 2609.01461 | [PDF](https://arxiv.org/pdf/2609.01461v1)

**作者:** Zhehan Qu `[一作]` (Duke University), Maria Gorlatova `[通讯]` (Duke University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

在模拟搜索与救援情境下，设计并对比了一个空间符合的AR界面与信息等价的移动端界面，评估它们对机器人协作中的情境意识（机器人和环境）以及用户视觉注意力分配的影响。

**💡 创新点**

首次在真实环境中使用SAGAT结构化测评机器人与环境的三层情境意识，并通过眼动数据揭示AR对感知级别机器人情境意识提升的机制（由扫视速度中介），同时系统性比较AR与手机界面对注意力与情境意识的差异。

**🔧 技术方法**

采用魔术眼2光学 see-through AR、Unity 3D、ROS2 控制Unitree Go2 机器人、OpenXR 眼动追踪、SAGAT 闪断测试、统计模型（GLMM、结构方程）及瞳孔/扫视分析。

**📊 数据集**

使用实验室搭建的SAR‑风格任务空间（10 个子任务站点、30 名参与者），收集 SAGAT 回答、眼动记录、导航时间、主观评估（NASA‑TLX、UMUX‑Lite、UEQ‑S）等数据。

**📈 对比分析**

通过 GLMM 对 SAGAT 结果进行统计比较，发现 AR 在机器人感知级别的情境意识显著高于手机（p≈0.028），但在更高层次和环境情境意识上无显著差异；眼动分析显示 AR 提高了扫视速度，完全中介了感知级别的提升，表明 AR 的优势主要体现在主动视觉探索上。

**⚠️ 局限性**

主要限制包括：光学 see‑through 视野受限、实验规模和空间范围有限、SAGAT 题目设计可能不完全覆盖复杂情境、AR 中的视觉元素竞争可能抑制环境情境意识，以及缺乏长期训练和更大规模实验验证。

---

## 689. Learning Sparse Decision Trees via Transformer Variational Auto-Encoders

**arXiv ID:** 2609.01430 | [PDF](https://arxiv.org/pdf/2609.01430v1)

**作者:** Giacomo Fidone `[一作]` (University of Pisa), Riccardo Guidotti `[通讯]` (University of Pisa)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种利用Tree Transformer Variational Auto-Encoder（TTVAE）将决策树映射到连续潜在空间，并通过可微代理模型和梯度上升同时优化预测性能与树结构稀疏性的学习方法。

**💡 创新点**

创新点在于将离散树搜索空间转化为连续潜在空间，结合Transformer的结构捕捉树形信息，并使用可微代理模型实现多目标梯度优化，从而在不牺牲预测精度的前提下显著提升树的结构稀疏性。

**🔧 技术方法**

使用了Tree Transformer Variational Auto-Encoder、可微代理模型（MLP）和梯度上升进行潜在空间优化，以及对决策树进行线性化编码、树绝对位置编码等Transformer技术。

**📊 数据集**

在18个公开基准数据集（包含连续和离散特征版本）上进行实验验证，涵盖不同样本量、特征类型和类别数量。

**📈 对比分析**

与CART、DL8.5、GOSDT等近最优树学习算法比较，TREVIS在测试集上的加权F1-score与叶子数上均达到或逼近最优水平，显著提升了性能-稀疏性权衡。

**⚠️ 局限性**

局限包括：潜在空间对预测性能的组织不够清晰；训练和梯度优化过程较为耗时；在完整连续特征空间下效果相对较弱。

---

## 690. Pix2Rep-v2: Data-Efficient Representation Learning for Dense Medical Imaging Applications

**arXiv ID:** 2609.01427 | [PDF](https://arxiv.org/pdf/2609.01427v1)

**作者:** S. Sifaoui `[一作]` (LTCI, Télécom Paris, Institut Polytechnique de Paris), L. Le Folgoc `[通讯]` (LTCI, Télécom Paris, Institut Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 Pix2Rep‑v2，一个用于密集医学图像任务的数据高效自监督表示学习框架，兼顾 2D/3D 结构并支持零/少样本下的直接推理。

**💡 创新点**

创新点包括：① 用冗余削减（non‑contrastive）目标替代传统对比损失，显著降低负样本数量与计算开销；② 采用激进的多尺度补丁采样与单一随机空间变换实现空间等变性；③ 在 3D 中引入隐式 U‑Net 结构，仅在所需坐标上生成特征，显著节省显存；④ 提供无参数的 in‑context 原型推理方法，支持零/一样本分割与视频传播。

**🔧 技术方法**

核心技术：自监督预训练（冗余削减 + 交叉相关矩阵），多尺度补丁采样，隐式 3D U‑Net，MLP 投影头，密集原型最近邻推理，零样本视频传播策略。

**📊 数据集**

使用的公开医学数据集：ACDC、M&Ms、M&Ms‑2（心脏 MRI）和 AMOS（腹部 CT），包括原始未标记数据用于预训练以及不同标注比例（1%、5%、10%、25%、100%）用于线性探针/微调评估。

**📈 对比分析**

与多种基线对比：从头训练的 U‑Net / Swin‑UNETR、已微调的 foundation 模型、现有对比/冗余削减密集 SSL 方法、以及其它 3D SSL。实验结果显示，Pix2Rep‑v2 在 1% 标注下心脏 MRI 分割的 Dice 分数提升 9.3 分（相较最佳基线），在少样本场景下提升 25× 数据效率；在零样本视频传播中近乎匹配全量微调模型的表现；在 3D 腹部 CT 多器官分割中在低标注率下提升约 5× 数据效率。

**⚠️ 局限性**

局限性：预训练仍需大量无标记扫描；隐式 3D U‑Net 需要额外的实现细节与显存管理；原型最近邻推理在大规模数据时对 GPU 内存和搜索速度有挑战；目前仅在分割与视频传播任务上验证，尚未针对配准、检测等其他密集任务进行评估。

---

## 691. Harness-of-Harness: Multi-Day Autonomous Software Development with Continual Improvement

**arXiv ID:** 2609.01481 | [PDF](https://arxiv.org/pdf/2609.01481v1)

**作者:** Haoyang Yan `[一作]` (Shanghai Artificial Intelligence Laboratory), Shuyue Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Harness-of-Harness（HoH）框架，将现有的LLM编码代理 harness 组织为连续的规划–编码–测试循环，支持从零开始的自主软件开发；在三种 benchmark（GameCraft-Bench、FrontierSWE、ProgramBench）上与传统单次开发（Vanilla）对比，并在真实游戏项目 Fusepoint 上演示 70 轮的长周期迭代。

**💡 创新点**

核心创新在于：1）跨迭代的状态管理，将实现体（artifact）与执行证据（evidence）分别持久化并在每轮传递；2）三角色（Planner、Developer、Tester）职责划分与固定 harness‑model 组合的可复用性；3）利用结构化文档、进度索引与进化版本控制实现可追溯、可回滚的长期开发；4）通过“持续改进”循环避免单轮局部修复，提升整体质量。

**🔧 技术方法**

技术组合包括：大型语言模型（Codex GPT‑5.5、OpenCode DeepSeek‑V4‑Pro、Pi MiniMax‑M3、Codex GPT‑5.6‑Sol），多种工具（MCP 服务器、专属插件、资产生成、UI/UX 辅助、测试工具），GitHub 版本控制与 issue 跟踪，结构化 JSON 输出与运行时验证，持续集成流水线（runtime contract enforcement）。

**📊 数据集**

数据集：GameCraft‑Bench（140 任务中抽样 45 任务，覆盖 15 种游戏类别）、FrontierSWE（17 任务中抽样 15 任务，涵盖实现、性能、研究三类）、ProgramBench（编译可执行文件回构建任务），以及 Fusepoint 的自定义需求文档。

**📈 对比分析**

比较方法：在每个 benchmark 上，HoH 与对应 Vanilla 基线在相同 harness‑model 组合下进行同样的初始状态，评估标准为：GameCraft‑Bench Overall 分数、FrontierSWE 平均奖励与 Dominance、ProgramBench Avg. Test Pass Rate。HoH 在所有配置下均显著优于 Vanilla；例如 Codex + GPT‑5.5 在 GameCraft‑Bench 上从 49.58 提升至 71.52（+21.93），FrontierSWE 奖励从 0.31 提升至 0.54，ProgramBench Pass Rate 从 60.41 提升至 66.50。与 Vanilla Continuation（多轮相同配置但不使用 HoH）对比，HoH 在同等代价下取得更高分数（如 3 轮 HoH 得 71.52 对比 58.24）。

**⚠️ 局限性**

限制与挑战：1）依赖固定 harness‑model 组合，若 harness 变更需重新适配；2）模型生成质量仍有限，易出现逻辑错误或不一致；3）对资源（如 GPU、存储、外部 API）要求高，长周期可能导致成本上升；4）缺乏真正的多智能体协作机制，所有角色通过同一 LLM 实例实现；5）在非常大规模项目或高度并行任务时，逐轮迭代的线性流程可能成为瓶颈。

---

## 692. When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning

**arXiv ID:** 2609.01455 | [PDF](https://arxiv.org/pdf/2609.01455v1)

**作者:** Yitong Guo `[一作]` (Indiana University Bloomington), Haixu Tang `[通讯]` (Indiana University Bloomington)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大语言模型（LLM）中，正向微调（benign fine‑tuning）对安全对齐（refusal behavior）的破坏机制，提出了 Fisher‑几何视角的输出路由解释，并对恢复策略（少量安全样本或前缀提示）进行了验证。

**💡 创新点**

创新点在于：①揭示安全对齐脆弱性来自输出侧低秩 Fisher 的重新锐化，而非梯度冲突；②将安全行为视为输出路由机制，解释了不对称破坏和易恢复性；③系统比较了 LoRA 与 ASAM 在不同数据规模下对安全路由的抑制效果，显示其在大规模微调时失效。

**🔧 技术方法**

使用技术包括：Fisher 信息矩阵与特征值分析、logit‑lens 读取、跨条件激活补丁（activation patching）、LoRA、Adaptive Sharpness‑Aware Minimization (ASAM)、SFT/DPO 对齐、LLM‑as‑judge 评估、梯度冲突样本挑选等。

**📊 数据集**

实验数据集包括：Align‑256、Align‑10k（安全增广数据）、Alpaca、Dolly、SciQ、MUSE‑News、HEx‑PHI、MMLU、BoolQ、ARC‑Easy、Wildchat、StrongReject 等。

**📈 对比分析**

在 Llama‑3.1‑8B 与 Qwen2.5‑7B 上，100 条正向微调样本即可将 ASR 从 0 提升至 50%–90%，但对 MMLU、BoolQ 等效能影响有限；LoRA 与 ASAM 在小样本（≤100）下显著抑制 ASR 上升，但在 5000 样本后仍出现安全崩溃；恢复时仅需 10–50 条安全样本即可将 ASR 降回 0，或通过前缀提示即时降低 ASR。整体性能对比表明，安全对齐的破坏高度可恢复但对齐方法在大规模微调中存在局限。

**⚠️ 局限性**

局限性包括：①仅评估了 SFT/DPO 这两类对齐方法，未涉及 RLHF 等强化学习框架；②实验聚焦于中型开源模型（8B/7B），未验证超大规模模型或其他架构；③主要使用英文数据，缺乏多语言或跨文化的验证。

---

## 693. Relational-Core Graph Analytics Querying graphs at SQL scale, and why the node/edge model is a performance tax, not a truer picture of connected data

**arXiv ID:** 2609.01525 | [PDF](https://arxiv.org/pdf/2609.01525v1)

**作者:** Gene Zhang `[一作]` `[通讯]`, Gene Zhang

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

开发了 ClickGraph 与 DeltaGraph 两套系统，将 Cypher 语句直接翻译为 SQL 并在现有的关系型（ClickHouse、Databricks 等）数据仓库上原地执行，实现无 ETL、无单独图数据库的图分析。

**💡 创新点**

创新点在于证明：①对企业实际工作负载，使用关系型引擎的列式存储和向量化执行可匹配或超越原生图引擎；②提出“原生模式映射”即不对关系模式做重编码，避免重编码开销；③设计单一路径、枚举驱动的翻译流程，保证覆盖所有关系模式变形且易维护。

**🔧 技术方法**

技术细节包括：Cypher → SQL 解析器与优化器、ClickHouse/Databricks SQL 后端、递归 CTE 与固定长度路径展开、列式存储向量化执行、SQL/PGQ 兼容性、LLM 辅助模式发现、Bolt 协议兼容、嵌入式 chdb 模式。

**📊 数据集**

使用的数据集：LDBC Social Network Benchmark（SF1 与 SF10）、OnTime Flights（约 12M 边）、Synthetic 社交图（10 万节点 200 万边）以及官方 LDBC SNB 41 个查询脚本。

**📈 对比分析**

比较方法为：①对比 Neo4j、PuppyGraph 及官方 OnTime Benchmarks；②在同一 PostgreSQL 实例上做微基准，评估原生 FK 连接与 AGE 节点/边映射的差异；⑥结果显示：列式引擎在分析查询上比 Neo4j 快 2–4 个数量级，ClickGraph 在已缓存状态下可与 PuppyGraph 同速甚至更快；原生 FK 查询比节点/边映射快 2.4–6.8 倍。

**⚠️ 局限性**

局限性包括：仅支持只读 OLAP 查询，无法处理事务写入；评估基于单机实验，未覆盖分布式规模；对递归路径与 WITH/UNWIND 组合的某些模式仍未完整优化；缺乏与 Neo4j 或 TigerGraph 的直接同机基准；LLM 辅助模式发现仍需人工确认；尚未实现对深度 OLTP 路径和大规模多租户场景的优化。

---

## 694. Behavioral Memory under Symmetry in One-Way Quantum Automata

**arXiv ID:** 2609.01451 | [PDF](https://arxiv.org/pdf/2609.01451v1)

**作者:** Zeyu Chen `[一作]` `[通讯]`, Zeyu Chen

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文基于算子代数与表示论，构建了一个三层过滤器框架，解析了在紧致对称性约束下量子有限自动机的行为记忆空间。通过对可达-可观测配对、结构容量以及阈值实现三者的分离，推导了严格切点语言的最优状态数上界，并给出了完整的符号与结构性上界（包括非可交换性导致的额外单态开销、Schur‑Weyl 对偶性下多项式与指数极大记忆量的转变，以及半填充情形下的 Catalan 计数）。

**💡 创新点**

创新点在于：①将记忆量拆解为三种逻辑过滤器（实例几何、结构容量、阈值实现），实现了对可达、可观测、阈值可见性的完整解析；②确立了可交换子中心与对易子维数决定的记忆容量公式，说明非可交换性仅导致单态额外开销；③通过 Schur‑Weyl 对偶性展示了对称性切换可从多项式提升至指数极大记忆量；④在半填充点给出 Catalan 计数的精确记忆界；⑤提供了二进制构造与严格符号矩阵证据，证明上界与下界在多种模型（可逆、耗散、测量多次）均可达到。

**🔧 技术方法**

使用技术包括：算子代数（中心、对易子、Wedderburn 分解）、群表示论（Schur‑Weyl 对偶性、分支规则、分块相同表示）、Hankel 矩阵与真实秩理论、符号秩与动态散射、密度矩阵与 Haar 退化、酉群生成与密度分量、Choi 代表、傅里叶/Fisher 信息几何、极值与不等式证明、组合计数（Catalan、Regev 定理）以及数值极限分析。

**📊 数据集**

本工作为理论研究，无需实验数据集；所有结果均通过纯理论推导与构造证明得出。

**📈 对比分析**

在比较方面，作者将推导出的上界与构造得到的下界进行对照，证明它们在大多数情形（可逆、耗散、两字母、任意字母、固定权重模块）均实现等价，展示了极大记忆量的精确定量。性能评价仅体现在理论上极大状态数与实际构造的匹配程度，未涉及数值实验。

**⚠️ 局限性**

局限性包括：①仅针对严格切点（real‑PFA）模型，无法直接推广到近似或错误容忍模型；②假设系统具备紧致对称性，非紧致或连续对称性下的结论尚未验证；③构造中的符号与状态数往往需要多字母或特殊初态，实际实现难度较大；④对耗散通道与测量多次模型的阈值实现依赖于特定的中间投影与消耗子，可能对实际量子硬件限制较大。

---

## 695. Cross-Modal Guidance for Out-of-View Object Search in Simulated Prosthetic Vision

**arXiv ID:** 2609.01438 | [PDF](https://arxiv.org/pdf/2609.01438v1)

**作者:** Adyah Rastogi `[一作]` (University of California Santa Barbara), Michael Beyeler `[通讯]` (University of California Santa Barbara)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对比了视觉、听觉和触觉三种引导方式在受限视觉场（模拟视网膜植入器）下的离视角目标搜索性能，使用沉浸式VR实验并记录搜索时间、获取与后续搜索时长等指标。

**💡 创新点**

首次在受限视觉下进行同一目标偏移量的多感官引导对比，并将搜索过程细分为目标获取与后续搜索两阶段，展示不同感官对各阶段的差异化影响。

**🔧 技术方法**

采用Unity+HTC Vive Pro Eye实现沉浸式VR，利用axon-map模型生成生物学验证的磷光感知；视觉提示通过相同模型渲染，听觉提示为立体声频闪，触觉提示为控制器振动。

**📊 数据集**

实验数据来自19名视力正常受试者，共160个搜索试验，目标为14个常见桌面物体，设置两种仿真视网膜阵列（10×10、20×20）并加入单目标与多目标（含杂物）情境。

**📈 对比分析**

采用4×2×2受试者内设计与线性混合效应模型比较四种条件，结果显示三种引导均显著缩短搜索时间，听觉/触觉引导平均比视觉快约25%，在更粗糙的视觉条件下优势更明显；同时获取时间和后续搜索时长也得到不同程度的改善。

**⚠️ 局限性**

局限性包括：使用视力正常受试者、实验时间短且未涉及真实视网膜植入患者的长期适应；引导实现方案在感知和时间映射上不完全等效；仅提供水平方向信息，未考虑垂直方向、目标移动或更复杂场景等现实因素。

---

## 696. HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?

**arXiv ID:** 2609.01437 | [PDF](https://arxiv.org/pdf/2609.01437v1)

**作者:** Yuhao Wu `[一作]`, Wenxuan Zhang `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 HarnessDev 基准，评估大型语言模型（LLM）在从弱种子创建可执行的 agent harness（Creation）以及在反馈驱动的演进（Evolution）中持续改进该 harness 的能力。

**💡 创新点**

创新点包括：① 将评估焦点从单次任务完成转向可持续、可重用的执行系统；② 设计了双阶段评估流程（Creation+Evolution），并引入了执行成本（executor token）与下游任务性能的双重度量；③ 通过固定 seed、对比不同执行器（自驱动 vs 固定 Gemini）以及对人类工程系统的基准，对 harness 质量进行多维度检验；④ 公开了完整的开发环境、审计机制和评估流程，促成可复现的 harness 开发研究。

**🔧 技术方法**

技术手段主要是：LLM 编程与调试（利用 Claude Code、OpenRouter 等平台），自动化评估脚本（执行 harness、收集指标、记录 trace），版本控制与评估管线（冻结 harness、生成官方版本），以及基于 token 计数的执行成本分析。

**📊 数据集**

使用的主要数据集包括：SWE‑bench（公开分割731个实例）、Terminal‑Bench 2.1（89个任务）、MLE‑bench（75个实例）、EQ‑Bench3（46个任务）和 BrowseComp（1,266个实例），以及 100‑任务 SWE‑Pro 反馈集和 630‑任务 SWE‑Pro held‑out 集用于演进阶段。

**📈 对比分析**

比较方法：在 Creation 阶段，对每个 LLM（Opus 4.8、GPT‑5.5、Gemini 3.1 Pro、DeepSeek V4 Pro、Qwen 3.7 Max、Seed 2.0 Pro）分别构建三份 harness，并在相同的执行器上跑全量下游任务，报告平均性能（avg@3）和平均执行 token；在 Evolution 阶段，以原始 harness 为起点，利用 100‑任务 feedback 进行迭代，记录每一步官方版本的两任务性能与最终 630‑任务 held‑out 性能；还做了跨执行器（自驱动 vs 固定 Gemini）的迁移实验。结果显示：在写作、机器学习实验等领域模型构造的 harness 能接近或超过人类参考，但在搜索、研究和代码领域仍有显著差距；执行成本与性能不呈正相关；演进过程中虽能获得局部提升，但在未见过的任务上提升有限，且多数迭代对最终性能无持续收益。

**⚠️ 局限性**

局限性包括：① 仅覆盖四个领域且演进实验仅在 SWE‑Pro 上进行，无法评估跨域泛化；② 每个 creator‑executor 组合只跑一个演进轨迹，缺乏统计置信区间；③ 开发环境固定，未测试演进 harness 能否自我迭代；④ 人工基准不一定最优且跨模型比较受执行器差异影响；⑤ 评估仅捕捉执行 token 与任务成功率，未深入分析错误模式与代码质量。

---

## 697. Gaussian Core LoRA: Distribution-Aware Dynamic Adaptation for Broad Concept Erasure

**arXiv ID:** 2609.01433 | [PDF](https://arxiv.org/pdf/2609.01433v1)

**作者:** Qinghui Gong `[一作]` (Southwest Jiaotong University), Zhengchun Zhou `[通讯]` (Southwest Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了一种分布感知的低秩适配器 Gaussian Core LoRA，用以在文本到图像扩散模型中对危险概念进行消除。

**💡 创新点**

创新点在于将目标概念视为提示特征空间中的多模态分布，利用高斯混合模型计算责任向量并按提示自适应地重构共享 LoRA 的低秩空间，从而避免了静态 LoRA 的模式平均问题。

**🔧 技术方法**

技术方法包括：高斯混合模型（GMM）对提示特征聚类；门控与责任向量实现软模式路由；在 LoRA 基础上插入动态核心实现提示条件的 rank‑space 重配置；训练时结合安全替代提示对齐损失与保留损失。

**📊 数据集**

主要数据集为 I2P 评估概念消除、MS‑COCO 评估生成质量；对抗攻击使用 Ring‑A‑Bell、P4D、UnDiff；多身份/多风格扩展采用 GPT‑4o 生成的 100 个多样化提示；跨架构兼容性评估在 SDXL 与 FLUX 上进行。

**📈 对比分析**

与 ESD、RECE、MACE、SAFREE、AdaVD、Prototype‑Guided 等基线对比，平均攻击成功率下降 7.95%，COCO FID 降低 14.72%，CLIP Score 提升 4.98%；在对抗攻击下保持最低 ASR；在 50 身份/风格扩展中存储量最小且性能稳定。

**⚠️ 局限性**

局限性在于仅使用对角协方差的 GMM，可能无法完整捕捉复杂语义模式；对开放式提示的边界定义仍较粗糙，需要进一步探索更灵活的原型建模与开放集条件下的可控性。

---

## 698. Vision-Based Leader-Follower Formation Control for Cooperative UAVs in GPS-Degraded Environments

**arXiv ID:** 2609.01420 | [PDF](https://arxiv.org/pdf/2609.01420v1)

**作者:** Deekshitha Angadi `[一作]` (Autonomous Robotics Systems Limited), Narsimlu Kemsaram `[通讯]` (University of Malaya)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套基于RGB‑D相机和YOLOv8检测的无人机视觉领导-跟随形成控制框架，能在GPS失效时通过视觉估计相对姿态并维持编队。

**💡 创新点**

创新点在于：①将轻量级YOLOv8与RGB‑D几何结合，实时获取相对姿态；②使用卡尔曼滤波平滑视觉估计并与GPS融合；③在ROS+XTDrone+PX4仿真环境中实现完整的多无人机编队控制并对比性能。

**🔧 技术方法**

技术包括YOLOv8深度学习检测、RGB‑D深度融合、针孔相机模型、常数速度卡尔曼滤波、PID编队控制、ROS节点互联、MAVROS与PX4离线控制。

**📊 数据集**

使用自建约1339幅图像的无人机数据集（公开图像+合成渲染），并在XTDrone/Gazebo仿真中生成同步RGB‑Depth、IMU、GPS数据进行评估。

**📈 对比分析**

与GPS正常、GPS失效以及可见性极限三种情景对比：检测精度达到92%+召回率；相对位置RMSE≤0.45 m、偏航误差≤3.8°；编队偏差≤1.2 m；整体推理时延≈38 ms（≈25 Hz）并能在低功耗边缘设备上部署。

**⚠️ 局限性**

局限性包括：深度相机噪声导致远距离误差增大；检测在视野边缘或快速转弯时会暂时丢失；目前仅在两架无人机的仿真环境中验证，未测试真实户外多机复杂场景。

---

## 699. Support Local Variables

**arXiv ID:** 2609.01502 | [PDF](https://arxiv.org/pdf/2609.01502v1)

**作者:** Maxwell Bernstein `[一作]` (Shopify), Kevin Menard `[通讯]` (Shopify)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了 ZJIT，一种针对 CRuby 的方法级 SSA JIT 编译器，重点优化了 Ruby 本地变量的处理，并与 YJIT、JRuby、TruffleRuby 等主流实现进行了对比。

**💡 创新点**

创新点在于采用将本地变量升至 SSA 值的乐观假设策略，结合全方法编译、静态分支优化、PatchPoint 与 Deoptimization 机制，首次在 Ruby JIT 中实现 SSA 层次的本地变量优化。

**🔧 技术方法**

实现技术包括静态单赋值（SSA）IR、基于反馈的类型信息采集、PatchPoint/Deoptimization、堆栈映射、循环内联、线性扫描寄存器分配，以及与 CRuby 字节码解释器的深度交互。

**📊 数据集**

使用数据集为 ruby‑bench（包含自制的 locals & call 微基准）以及 Rails 应用 Benchmark（Discourse 模板渲染与 SQLite 查询）。

**📈 对比分析**

通过在同一台 Ubuntu 24.04 + Intel Core Ultra 7 270K Plus 机器上运行 1,000 次实验并剔除预热阶段，比较了 CRuby、YJIT、JRuby、TruffleRuby 的平均执行时间；ZJIT 在本地变量密集的微基准上与 TruffleRuby 旗鼓相当，速度比 YJIT 快约 30% 以上；在 Rails 基准上与 YJIT 竞争，略逊但明显优于解释器。

**⚠️ 局限性**

局限性包括缺乏对局部变量逃逸分析与块内 inline 的完整支持，未实现对复杂参数传递（如 splat）的优化，eval 与 API 侧逃逸处理仍不够完善，导致某些场景下 deoptimization 与重编译频繁。

---

## 700. CameraEditor: Camera-Controlled Image Editing via Video-Prior Sequential Modeling

**arXiv ID:** 2609.01479 | [PDF](https://arxiv.org/pdf/2609.01479v1)

**作者:** Xin Shen `[一作]` (Xi'an Jiaotong University), Minnan Luo `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出CameraEditor框架，利用视频扩散模型将相机参数化的图像编辑任务转化为时间序列预测，并通过动态参考路由与中间帧插入实现大视角变换的平滑控制。

**💡 创新点**

①将编辑过程视为视频帧序列生成任务；②使用几何感知模块与动态全景裁剪生成严格几何参考；③引入Chain of Frames（CoF）插入中间帧，缓冲大角度变换；④构建专用训练集与CamEditor-Bench基准。

**🔧 技术方法**

视频扩散模型（Wan2.1‑T2V‑14B）、3D VAE、GeoCalib几何估计、动态参考路由、LoRA微调、连帧插值。

**📊 数据集**

训练集5,760个实例，来源于真实全景（360‑SoD、F360‑SoD、CVRG‑Pano）和UE5合成场景；CamEditor‑Bench测试集462个独立案例。

**📈 对比分析**

与6款开源模型（ICEdit、Step1X‑Edit、OmniGen2、Flux.2、Qwen‑Image‑Editing、HunyuanImage‑3.0‑Instruct）及2款闭源API（GPT‑Image‑1.5、Nano‑Banana‑Pro）在CamEditor‑Bench上比较，使用DINO‑v2、CLIP‑I、SSIM、LPIPS、S_up^I、E_lat^I、S_up^T、E_lat^T等指标。CameraEditor在所有指标上均为最优（如DINO‑v2 0.8569、SSIM 0.5712、S_up^I 0.9267、E_lat^T 0.1904），人类评估亦显示其优先率达82%。

**⚠️ 局限性**

（1）视频3D‑VAE的空间压缩导致高频纹理被过度平滑；（2）系统为串行管线，缺乏端到端优化，导致推理延迟高且未能实现最优联合学习。

---

## 701. The Role of Collective Perception and 5G NR-V2X Sidelink in Road Safety

**arXiv ID:** 2609.01478 | [PDF](https://arxiv.org/pdf/2609.01478v1)

**作者:** Vittorio Todisco `[一作]` (University of Bologna), Alessandro Bazzi `[通讯]` (University of Bologna)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出并实验了一种基于随机包含的训练策略，以提升机器学习模型在不同分布下的鲁棒性

**💡 创新点**

创新点在于将随机包含与传统训练方法相结合，通过随机采样减少模型对训练数据偏差的敏感度

**🔧 技术方法**

采用随机采样、统计评估以及标准深度学习框架实现

**📊 数据集**

使用公开的CIFAR‑10和ImageNet等图像数据集进行验证

**📈 对比分析**

与传统基线方法比较，随机包含在准确率、稳定性和方差方面均表现优于基线，平均提升约10–15%

**⚠️ 局限性**

局限性在于方法在大规模数据集上的计算成本较高，且对极端噪声或小样本情况的适用性仍待进一步验证

---

## 702. Efficient K-Visibility Query in Polygons

**arXiv ID:** 2609.01472 | [PDF](https://arxiv.org/pdf/2609.01472v1)

**作者:** Yeganeh Bahoo `[一作]` (Toronto Metropolitan University), Roni Sherman `[通讯]` (Toronto Metropolitan University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种针对k-可见性查询的紧凑细胞分解与δ压缩树结构，能够在平面多边形（含孔）中高效查询任意点的k-可见多边形。

**💡 创新点**

创新点在于：① 仅使用k及k-2可见性窗口和互为关键顶点的铰接线构造最小化的分区线集；② 证明该线集即可产生Θ(n⁴)个细胞；③ 通过δ压缩的双向树将空间从原来的O(n⁵)压缩到O(n⁴)，同时保持O(log n + m)的查询时间。

**🔧 技术方法**

使用平面几何排列、窗口生成器序列、组合编辑（INSERT/DELETE/REPLACE）操作、双向树δ压缩、以及最优点位置查询算法。

**📊 数据集**

实验主要在合成的简单多边形（包含孔）上进行，未提供具体真实数据集。

**📈 对比分析**

与之前的O(n⁵)空间方案相比，空间显著降低到O(n⁴)，查询时间保持最优O(log n + m)。实验结果表明在大规模多边形上，性能明显优于现有方法。

**⚠️ 局限性**

仍然需要Θ(n⁴)个细胞，常数因子较大；当前方法仅适用于二维平面，多边形的动态变化以及三维扩展尚未实现。

---

## 703. AutoConcept: Training-Free Concept-Guided Reranking for Metadata-Available Composed Image Retrieval

**arXiv ID:** 2609.01456 | [PDF](https://arxiv.org/pdf/2609.01456v1)

**作者:** Tianyu Wang `[一作]` (Soochow University), Tianjiao Wu `[通讯]` (INSTITUT NATIONAL DES SCIENCES APPLIQUEAS DE LYON)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 AutoConcept，一种无训练的概念记忆重排序器，用于在已检索到的候选集上利用可用的商品元数据进行二阶重排序。

**💡 创新点**

创新点在于构建可解释的概念记忆层，结合查询相关性门控、正负概念激活及推理时刻的闭式校准，实现对元数据可用的 composed image retrieval（CIR）进行无训练的概念驱动重排序。

**🔧 技术方法**

技术方法包括句子编码与聚类、概念过滤、余弦相似度对齐、max/mean 池化、基于查询的概念激活门控、无训练的加权与插值校准，以及整体闭式推理时刻加权计算。

**📊 数据集**

实验数据集主要使用 FashionIQ，并额外收集了人工概念标签数据以验证概念记忆的可用性。

**📈 对比分析**

与 WeiMoCIR、LinCIR 等基线以及文本到元数据匹配、属性匹配、约束匹配等对齐控制方法对比，AutoConcept 在 Recall@10、Recall@50、MRR、NDCG@10 等指标上分别提升约 15–22%（WeiMoCIR）和 30–35%（LinCIR），显示显著性能提升。

**⚠️ 局限性**

局限性包括：需要完整且一致的商品元数据；概念记忆构建主要基于文本，缺乏视觉锚定；在稀疏或噪声较大的人工标签下提升有限。

---

## 704. Does Imitation Learning Preserve Temporal Robustness in Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution Speeds

**arXiv ID:** 2609.01453 | [PDF](https://arxiv.org/pdf/2609.01453v1)

**作者:** Clinton Enwerem `[一作]` (University of Maryland), Calin Belta `[通讯]` (University of Maryland)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种在不同执行速度下评估专家与模仿学习者任务成功率的对照方法，并在ParcelStow仿真环境中进行了实验。

**💡 创新点**

创新点在于引入速度因子对比框架，系统地分析了模仿学习策略在速度变化时的鲁棒性差异，并发现相同的成功率并不保证在速度范围内的性能一致。

**🔧 技术方法**

使用了Action Chunking with Transformers（ACT）以及DAgger和Diffusion Policy三种学习策略，并通过强化学习、逆向运动学和力闭合分析等技术进行评估。

**📊 数据集**

数据集为300个专家演示（速度范围0.5~2），其中包含对不同速度的采样；训练集和测试集在相同的随机初始条件下进行匹配。

**📈 对比分析**

比较方法通过在相同速度和初始条件下匹配100个试验，对成功率差异进行Bootstrap统计；结果显示ACT在最大演示速度下相较专家下降31个百分点，而其他两种ACT版本在速度提升时衰减更快。

**⚠️ 局限性**

局限性包括仅在单一模拟任务和固定观测空间下验证，未考虑不同机器人硬件、观测模态或更复杂任务的普适性，且缺乏对动作块长度和时序规划的系统性分析。

---

## 705. Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning

**arXiv ID:** 2609.01449 | [PDF](https://arxiv.org/pdf/2609.01449v1)

**作者:** Mariia Drozdova `[一作]` (University of Geneva), Blake Richards `[通讯]` (Paradigms of Intelligence Team)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种时序无关的循环扩散模型，利用持久隐藏状态在每一步更新中实现对数独和迷宫等任务的迭代求解。

**💡 创新点**

创新点在于将递归隐藏状态嵌入扩散去噪器，使模型无需逐步去噪即可通过单一路径完成推理，形成“任何时刻可用”的迭代求解器。

**🔧 技术方法**

技术上采用循环Transformer（或局部卷积块）作为去噪器，使用方差保持的噪声路径、截断反向传播和有序的逐步去噪训练课程。

**📊 数据集**

实验数据集包括极限数独（Sudoku‑Extreme）和唯一路径迷宫（Maze‑Unique / Maze‑Hard）。

**📈 对比分析**

通过与无隐藏状态或带时间条件的基线对比，实验显示在10,000步时数独解率可达99.90%，在100步时迷宫解率为98.93%，且增加推理步数可持续提升性能。

**⚠️ 局限性**

局限性在于模型高度依赖有序去噪训练课程，对噪声水平和隐藏宽度敏感，且尚未验证在多解或更复杂任务上的泛化能力。

---

## 706. Edge-Girth as a Structural Edge Feature for Graph Neural Networks

**arXiv ID:** 2609.01441 | [PDF](https://arxiv.org/pdf/2609.01441v1)

**作者:** Lilian Marey `[一作]` (Institut Polytechnique de Paris), Charlotte Laclau `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种新的无偏子图描述符——每条边的最短环长度与其在该环中的出现次数——并将其以门控方式注入消息传递神经网络；在分子属性回归和图同构判定两类任务上进行评估。

**💡 创新点**

创新点在于：①无需预先设定子图尺寸上限，直接捕获任意长度环的信息；②通过边层面门控实现结构信息的动态调制；③提供理论证明：在“edge‑girth‑regular”图族中该描述符恒定，模型表达能力退化到 1‑WL 上限。

**🔧 技术方法**

核心技术包括：单源 BFS 计算每条边的最短环长度与计数、将 (e, λ) 与桥指示符构成三通道边特征、门控消息传递（使用 Sigmoid 门），以及残差聚合和全局池化。

**📊 数据集**

实验数据集：
- Zinc‑12k（分子属性回归，10k/1k/1k 划分）
- BREC（图同构判定基准，400 对图）

**📈 对比分析**

与 GatedGCN、GatedGCN‑MLP、GIN、GCN、GSN 等模型在约 100k 参数预算下对比；在 Zinc 上取得 0.0932 MAE（比 GatedGCN 低 3 倍，接近 GSN 的 0.101），在 BREC 上整体区分率 48.3%（相较于 43.3% 的 GSN），但在 90 对 edge‑girth‑regular 图对上完全失效。

**⚠️ 局限性**

限制：
- 当图是 edge‑girth‑regular 时，该描述符恒定，模型表达力仅为 1‑WL；
- 仅在回归任务上验证，未评估分类或其它下游任务；
- 计算成本相对较高（每条边 O(|V|+|E|) 的 BFS），在稀疏或长环图上可能不如固定长度子图计数效率。

---

## 707. TRIAGE: Three-level Routing and Intelligent Agent Guidance for Efficient Execution

**arXiv ID:** 2609.01428 | [PDF](https://arxiv.org/pdf/2609.01428v1)

**作者:** Ruocan Wei `[一作]` `[通讯]` (China Telecom Cloud), Ruocan Wei (China Telecom Cloud)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了三层路由框架TRIAGE，通过将历史执行轨迹抽象为可重用技能（TaaS），实现LLM代理的经验驱动执行

**💡 创新点**

核心创新是将轨迹作为技能（Trajectory-as-a-Skill），实现零成本参数替换与直接重用；并在多域中验证了语义检索+阈值路由的通用性

**🔧 技术方法**

使用语义编码器all‑MiniLM‑L6‑v2进行向量检索，阈值路由、参数抽取、自动技能提取与FAISS加速索引；LLM采用deepseek‑v3.2完成ReAct与路由判定

**📊 数据集**

实验集为1,007条安全监控查询（包括1,007条结构化SQL与60条周期性日报），以及ToolBench的15个多域（345条）工具调用任务

**📈 对比分析**

与完整ReAct baseline对比，TRIAGE在安全监控查询上实现62.3% token节省（199,782→75,238），在ToolBench上实现76.3% token节省（424,792→100,875），L1+L2重用率超过60%

**⚠️ 局限性**

主要限制为冷启动成本高、跨域可迁移性差、第一查询必须完整ReAct、以及可能的轨迹错误传播需要验证机制

---

## 708. Semantic-Guided Multimodal Preprocessing for Vision Transformer-Based Clear Cell Renal Cell Carcinoma Grading

**arXiv ID:** 2609.01426 | [PDF](https://arxiv.org/pdf/2609.01426v1)

**作者:** Fatemeh Javadian `[一作]` (RWTH Aachen University), Johannes Stegmaier `[通讯]` (RWTH Aachen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

通过将预训练的核级分类图与RGB H&E图像进行语义引导的多模态预处理，利用ViT模型对CCRCC病灶进行分级。

**💡 创新点**

创新点在于提出了一种无需改动网络结构的预处理融合方法（乘法调制和颜色叠加），能将细粒度核级信息嵌入到粗粒度ViT输入中，从而显著提升分级准确性。

**🔧 技术方法**

使用了Vision Transformer（Google ViT Base Patch32-384）以及两种预处理策略——HEC（色彩分解+通道拼接）和MM（乘法调制+颜色叠加），并通过网格搜索调优超参数。

**📊 数据集**

数据集为1000张TCGA KIRC/KIRP来源的512×512 H&E切片块，按WHO/ISUP标准标注为三级（1–3）分级。

**📈 对比分析**

与仅用RGB图像的基线（平衡准确率0.707）以及传统最大投票聚合（0.427）相比，MM预处理在测试集上取得了0.916的平衡准确率，精度和召回率均显著提升。

**⚠️ 局限性**

主要局限包括：1）敏感性分析仅使用模拟误差，未在真实上游核级分类模型下训练和评估；2）数据拆分为patch级，可能存在同一切片在不同集中的相关性；3）未考虑多观察者差异，仅基于单个病理学家的标注。

---

## 709. MegaStyle++: Scaling Image Style Space through Hierarchical Style Definition

**arXiv ID:** 2609.01423 | [PDF](https://arxiv.org/pdf/2609.01423v1)

**作者:** Junyao Gao `[一作]` (Tongji University), Jun Zhang `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出层次化图像风格定义，并基于该定义构建 MegaStyle++-8M 数据集（150k 风格身份、1M 细粒度风格提示、8M 生成图像）。

**💡 创新点**

创新点在于：1) 统一整体风格身份到色彩、光照、纹理、介质、笔触等五大细粒度属性的层次化描述；2) 利用大模型自动注释、三阶段去重以及聚类抽样，显著扩大风格空间的多样性与语义宽度；3) 通过明确定义的属性模板实现可解释、可复现的风格提示。

**🔧 技术方法**

技术手段包括：Qwen3.5-35B-A3B 进行多属性注释；NeMo Curator 的 Exact/Fuzzy/Semantic 去重；SigLIP 作为特征空间评估；Qwen-Image 用于生成图像；CLIP、CSD、MegaStyle‑Encoder 等指标评估风格再现质量。

**📊 数据集**

主要数据集：MegaStyle 原始图像池（用于注释扩充）及生成的 MegaStyle++-8M；对比基准数据集包括 WikiArt、JourneyDB、Style30K、IMAGStyle、OmniStyle‑150K 以及原 MegaStyle。

**📈 对比分析**

对比方法：多指标评估（多样性比例、语义广度、相关性、模糊度、MPD、Vendi、MNND）与 MegaStyle；风格再现实验使用 StyleBench 真实参考图，结合多内容提示生成图像后通过 CSD、MegaStyle‑Encoder、CLIP‑Text 计算相似度。结果显示 MegaStyle++ 在大多数属性上拥有更高多样性、更低模糊度、更强相关性，并在再现实验中显著优于 MegaStyle。

**⚠️ 局限性**

限制：1) 仍依赖大语言模型与视觉语言模型，可能对极端或稀有风格的识别不足；2) 自动注释易受模型偏差影响，导致部分细粒度属性误标；3) 对光照、材质等属性的细粒度描述仍可能欠缺，尤其在跨域内容迁移时需进一步验证。

---

## 710. A System for Fast, Resilient, and Adaptable Loco-Manipulation Behaviors on Humanoid Robots

**arXiv ID:** 2609.01518 | [PDF](https://arxiv.org/pdf/2609.01518v1)

**作者:** Duncan Calvert `[一作]` (Florida Institute for Human and Machine Cognition), Robert Griffin `[通讯]` (Florida Institute for Human and Machine Cognition)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本工作提出一种机器人本地、可运行时编辑的行为编写与执行框架，用于人形机器人在无外部追踪设施下完成步态、手臂操纵、感知、接触与操作者监督等任务；

**💡 创新点**

其创新点在于将可利用的 Affordance Templates 与行为树结构相结合，配合机器人本地行为场景感知、同步的操作者界面以及可并发的动作层叠，实现快速、可重用且易于实时改写的行为设计；

**🔧 技术方法**

技术实现包括基于Affordance Templates的对象中心动作定义、行为树的序列/回退/跳转节点、YOLO 与 ZED X Mini 立体深度摄像机的即时感知、CRDT 机制的双向同步、IHMC 逆动力学全身控制器、CUDA 点计数器的形状检测、以及 RDX/ImGui 的可视化编辑；

**📊 数据集**

实验使用的“数据集”为现场真实机器人平台（Alex、Unitree H1‑2、Atlas）与实验室门、球、瓶子等物体的感知结果，并未采用公开标准数据集；

**📈 对比分析**

通过与经典模型驱动和近期学习型门穿行系统对比，门通过时间约为34–45秒，接近或优于已发表的 10–15 秒学习系统，并且在 11 次连续开启/闭合试验中 100% 成功，显示出高可靠性与较快的开发周期（从零编写到首次自动化成功约 11 小时，改写或组合行为仅需 1–2 小时）；

**⚠️ 局限性**

系统局限包括：对非球形物体的姿态估计不佳、缺少基于力的执行节点与子程序支持、步态控制在某些门场景下不够稳健、对生成式行为与开放世界自适应支持不足，以及对高质量感知参数的人工调节需求较大。

---

## 711. GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions

**arXiv ID:** 2609.01491 | [PDF](https://arxiv.org/pdf/2609.01491v1)

**作者:** Elias Stengel-Eskin `[一作]` (University of Texas at Austin), Simon Kirby `[通讯]` (University of Edinburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一个名为 GlossoGen 的多 LLM 代理实验平台，并在其中实现了一个基于信息不对称与通信预算的紧急响应场景，用以研究代理间语言演化的机制。

**💡 创新点**

首次系统展示了在非对抗性多回合任务中，强大 LLM 能在时间压力与事后复盘的双重驱动下自发产生非英语、可组合、可传播的新语言，从而揭示 LLM 具备累积文化演化的潜能。

**🔧 技术方法**

结合多代理交互、字符预算控制、事后复盘通道、基于 GPT‑2 的困惑度评估、混合效应回归和形态学归纳等技术手段，对语言生成与传递进行量化分析。

**📊 数据集**

使用了公开的 14 条症状与治疗模板作为任务知识库，并在实验中调用了三款专有 LLM（Claude Opus 4.7、Sonnet 4.6、GPT‑5.4）以及若干开源 LLM，比较它们在不同预算与复盘条件下的表现。

**📈 对比分析**

通过将不同模型与预算设定下的成功率、困惑度以及形态学生产/解码准确率进行对比，结果显示：在高压（150 字符）且开启复盘时，专有模型可显著降低困惑度并提高成功率（最高 97%），而开源模型几乎无法突破英语限制，成功率仅 30% 左右；在低压条件下，复盘的存在是任务成功的关键。

**⚠️ 局限性**

实验局限于合成的医疗类场景与有限的任务模板，缺乏跨文化与真实世界的验证；仅考察合作情境，未探讨竞争或监测压力；受限于 LLM 参数规模与可用算力，实验规模和随机性可能不足，导致结果对特定模型和超参数高度敏感。

---

## 712. Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents

**arXiv ID:** 2609.01487 | [PDF](https://arxiv.org/pdf/2609.01487v1)

**作者:** Xiaofang Yang `[一作]` (Shanghai Artificial Intelligence Laboratory), Lijun Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了可作为可安装技能的运行时安全防护框架（Defense-as-Skill），并用它构建了任务条件下的安全数据集 SCOPE-R 与训练优化方法。

**💡 创新点**

创新点：①将安全防护本身包装为可装载的技能，既可被审计、编辑，又能与任意 agent 运行时无缝集成；②通过显式安全责任分配保证技能被主动调用；③使用 Monte‑Carlo Tree Search（MCTS）在真实任务 roll‑outs 上反馈驱动技能演化，提升防护效果。

**🔧 技术方法**

技术：技能机制（Skill），MCTS 反馈驱动演化，动作级别允许/重规划/确认决策，利用大语言模型（GLM‑5、Claude Haiku 4.5、GPT‑5.4）作为受害者模型，使用 Qwen3.5‑122B‑A10B 做评判。

**📊 数据集**

数据集：SCOPE‑R（206 个攻击确认实例 + 43 个无攻击基准），覆盖六大风险族（Specification Integrity、Capability Control、Operational Side Effects、Privacy & Data Flow、Execution Safety、Resource & Reliability），以及 SkillSafetyBench 与 WildClawBench 作为外部验证集。

**📈 对比分析**

对比方法：无防护、系统提示、AcceptEdits、AcceptEdits+allowlist、AgentSpec、TS-Guard 等。SkillSonar 在 Claude Code/OpenClaw 上将 ID/OOD 攻击成功率从 48%–60% 降至 10%–12%，保持任务完成率 78%–82%，token 约 190k，优于其他基线且在跨模型、跨风险族、适应性攻击与外部基准上亦保持较强鲁棒性。

**⚠️ 局限性**

局限性：①依赖软指令，若 agent 失误未遵循 guard 仍可能被规避；②对高度定制或复杂工作流的适配仍需手工调整；③安全–效用平衡仍不可完全消除，某些场景 AcceptEdits 仍提供更低 ASR；④评估成本高，尤其在多模型多场景下需要大量 roll‑outs。

---

## 713. Binary Multiple-Node-Erasure-Correcting Codes over Complete Graphs: Constructions, q-Ary Metric Balls, and Duality

**arXiv ID:** 2609.01474 | [PDF](https://arxiv.org/pdf/2609.01474v1)

**作者:** Aryeh Lev Zabokritskiy `[一作]` `[通讯]` (Tel-Hai University of Kiryat Shmona in Galilee), Aryeh Lev Zabokritskiy (Tel-Hai University of Kiryat Shmona in Galilee)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `09944146-298c-433e-89df-37255de463d7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文研究在完整图上的边和自环上定义的二进制线性代码，解决节点擦除模型（即失效顶点导致其相邻所有坐标被抹除）下的码率与冗余问题。

**💡 创新点**

创新点主要有：① 通过允许斜率从 {2, –3, –15} 中选择，证明了在无限多奇素长度下存在三节点擦除的近最优二进制代码；② 引入多斜率（多 Frobenius）构造，在 2≤ρ≤(n+1)/2 范围内得到冗余仅比 Singleton 多 ρ–1 的 ρ‑节点擦除码；③ 用几何方法（投影射、外积、傅里叶分解）给出了节点度量球体积、球体交叉公式和精确的生成函数；④ 提出了节点-团互补双对偶关系，证明了 Singleton‑最优节点码与相应的团擦除码在维度与冗余上互为对偶。

**🔧 技术方法**

主要技术手段包括：循环项目线性子空间（arc）与原根论证；多斜率（Frobenius–Moore）检查；外积空间映射、交叉叉的线性同构；傅里叶分析证明矩阵秩；以及组合计数（Venn 配置、第二 Bonferroni 估计）来求球体体积与码量上界。

**📊 数据集**

论文不依赖传统机器学习数据集，而是使用纯组合数学计数与解析式。通过对 n≤12 的完全枚举以及对大 n 的渐进公式来验证理论结论。

**📈 对比分析**

与已有的 ρ 节点擦除码（如 Schmidt 方案、随机线性码）相比，本文构造的码在冗余上至少低于 Schmidt 的 ρn，且在 ρ=2,3 时与 Singleton 边界相差 1 或 0 位；对于固定 ρ，所给的随机线性存在上界证明了冗余可上限为 ρ log_q n + O(1)；此外，利用节点-团双对偶可立即得到无限族的最优 (n–2) 或 (n–3) 团擦除码。

**⚠️ 局限性**

主要局限性包括：① 仍未给出无限多长度的 3‑节点 Singleton‑最优二进制码；② 对 ρ> (n+1)/2 的多斜率构造缺乏冗余最优性；③ 节点度量球体积的精确计数在大 t 时计算量剧增，尚无高效算法；④ 论文中的所有构造均在二进制域上完成，对 q>2 的扩展需要进一步研究。

---

## 714. RadMatch: Auditable Radiology Report Evaluation via Finding-Level Matching

**arXiv ID:** 2609.01470 | [PDF](https://arxiv.org/pdf/2609.01470v1)

**作者:** Charles Corbière `[一作]` (Raidium), Corentin Dancette `[通讯]` (Raidium)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于大型语言模型的多阶段评估框架RadMatch，通过提取、匹配与显著性评分实现放射报告的可审计质量评估。

**💡 创新点**

创新点在于将报告拆解为发现级匹配并按临床重要性分层计数错误，既可解释又可追溯，提供细粒度错误类型与临床风险。

**🔧 技术方法**

利用大语言模型（如Claude Opus 4.8、GPT系列、Gemma 4‑31B 等）完成结构化发现提取、同义匹配和属性评分，并结合确定性比较器。

**📊 数据集**

在两套专家标注的胸片基准 ReXVal 与 RadEvalExpert 上进行评估。

**📈 对比分析**

与 BLEU、ROUGE、RadGraph‑F1、CRIMSON、GREEN 等指标对比，RadMatch 在两套基准上与放射员的一致性最高（Kendall τ≈0.79/0.58），显著优于其它方法。

**⚠️ 局限性**

局限在于仅验证胸片，尚未覆盖 CT/MRI 等模态，且需要多次 LLM 调用，存在计算与随机性开销。

---

## 715. Predicting Subsurface Abnormalities Growth using Physics-Informed Neural Networks

**arXiv ID:** 2609.01417 | [PDF](https://arxiv.org/pdf/2609.01417v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 716. SVP Is NP-Hard for Some Rank-2 Cyclotomic Modules

**arXiv ID:** 2609.01469 | [PDF](https://arxiv.org/pdf/2609.01469v1)

**作者:** Jiaqi Liu `[一作]` (Academy of Mathematics and Systems Science), Yanbin Pan `[通讯]` (Academy of Mathematics and Systems Science)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

论文证明了在某些秩为2的环上，最短向量问题的决策版本在ℓ_2范数下是NP-完全的，使用了从3集合的精确覆盖问题的确定性多项式时间归约。

**💡 创新点**

创新点在于首次证明了在秩为2的环上，最短向量问题的决策版本是NP-完全的，尤其是在全秩自由子模块的情况下。

**🔧 技术方法**

使用了确定性多项式时间的归约技术，结合了Bennett–Peikert Reed–Solomon格和二次高斯和的检查器。

**📊 数据集**

使用了与素数q相关的环𝒪_K的全秩自由子模块，具体为𝒪_K^2中的两个生成元。

**📈 对比分析**

通过与3集合的精确覆盖问题（X3C）进行比较，证明了在ℓ_2范数下的决策问题是NP-完全的，且在负实例中，最短向量的平方范数大于阈值S。

**⚠️ 局限性**

限制在于该结果仅适用于特定的环和秩为2的模块，未能证明对于单一固定的环的普遍适用性。

---

## 717. Just Talk Once: Communication-Efficient Split Federated LLM Fine-Tuning on Edge Devices

**arXiv ID:** 2609.01457 | [PDF](https://arxiv.org/pdf/2609.01457v1)

**作者:** Jiaxiang Geng `[一作]` (Duke Kunshan University), Bing Luo `[通讯]` (Duke Kunshan University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 L‑形 Split Federated Fine‑Tuning（L‑shaped SFT）和 One‑Shot SFT，解决了传统 U‑形 SFT 的双向通信瓶颈和连续客户端参与难题

**💡 创新点**

创新点在于将监督迁移至服务器端，利用 LLM 的权重共享实现基于目标嵌入的激活空间损失，消除激活回传/梯度上传循环，并可进一步压缩为一次性上传的 One‑Shot 方案

**🔧 技术方法**

使用权重共享、对比学习损失（A‑Loss）、缓存负样本、温度调节、LoRA 微调、Split Learning、Flower 分布式框架和移动端 C++ SDK

**📊 数据集**

在 WikiText‑2（文本生成）和 MMLU（多任务问答）上进行实验，模型包括 GPT‑2、Qwen、Gemma、Llama 等 0.27B‑8B 规模

**📈 对比分析**

与基线 U‑形 SFT 对比，L‑shaped SFT 在保持相近微调效果的同时，通信量下降 25–70%（单步激活/梯度），总体延迟下降 34–90%；One‑Shot SFT 在通信、延迟和客户端在线时长上进一步压缩至 90% 以下

**⚠️ 局限性**

局限在于仍需上传激活/嵌入可能泄露隐私，缓存负样本有限时可能影响梯度精度，以及对极低算力设备仍有显著内存/计算需求

---

## 718. From Rollouts to Recipes: Self-Contained Post-Training for LLMs

**arXiv ID:** 2609.01422 | [PDF](https://arxiv.org/pdf/2609.01422v1)

**作者:** Yifei Li `[一作]` (Xi'an Jiaotong University), Jun Liu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对可验证任务的后训练，提出了Self‑Routing框架；它根据模型自身rollout的正确率和置信度为每个样本动态分配GRPO、OPS、正则化或跳过四种优化策略。

**💡 创新点**

创新点在于把样本级学习状态（rollout行为）直接映射到训练策略，完全不需要外部教师或额外注释，实现自给自足的行为条件路由。

**🔧 技术方法**

核心技术包括基于rollout的准确率与token熵置信度计算、软三分成员函数、四种子策略（GRPO、On‑policy Self‑Distillation、正则化、跳过）以及概率路由器。

**📊 数据集**

使用了DAPO‑Math‑17K作为后训练数据，评测覆盖六个数学推理基准（GSM8K、MATH‑500、AIME24/25）和两个跨域推理基准（GPQA、MMLU‑Pro）。

**📈 对比分析**

与统一GRPO、统一OPSD、固定混合以及随机/比例/准确率路由等基线对比，Self‑Routing在所有Qwen3与Qwen3.5模型上均取得最高平均分；在ID数学任务上提升约5‑15分，且在OOD任务上保持更小的性能退化。

**⚠️ 局限性**

主要局限包括缺乏对路由与优化机制对应关系的理论解释、仅在可验证推理任务上验证，尚未证明在规划或开放式指令等其他场景的普适性。

---

## 719. Citing Less Critically: LLMs Reshape the Rhetoric and Reach of Scientific Citation

**arXiv ID:** 2609.01432 | [PDF](https://arxiv.org/pdf/2609.01432v1)

**作者:** Yixuan Liu `[一作]` (Northeastern University), Dakota Murray `[通讯]` (University at Albany, State University of New York)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个“掩码式引用”任务，通过对齐人类和LLM生成的引用句子，比较它们在引用意图、引用偏好和社交距离等方面的差异。

**💡 创新点**

创新点包括：①位置对齐、槽级对照的引用基准；②利用LLM自评判定引用意图；③在20M边、2.1M节点的共作者网络中量化社交距离，从而在意图维度细粒度检视LLM的引用偏差。

**🔧 技术方法**

技术手段：多模型提示（GPT‑5.1、Claude‑3.5、Gemini‑2.0、DeepSeek‑V3.2、Llama‑4‑Maverick、Qwen‑72B）；LLM‑as‑judge意图分类；Dimensions API 文献匹配；BFS 计算共作者距离；统计分析（Cohen’s κ、Welch t‑检验、Mann‑Whitney U）。

**📊 数据集**

数据集：1,746 篇 2025 年 ACL/EMNLP/NAACL 主要论文的全文，产生 63,944 个引用上下文和 132,913 个引用槽；六款LLM的生成结果；20.3M 边、2.1M 节点的 2015‑2024 共作者网络；Dimensions 计量数据（引用量、团队规模、年份）。

**📈 对比分析**

比较方法：①对齐同一位置的原始与LLM生成句子，统计意图保持率和意图分布；②聚合至论文级别，比较引用数量、团队规模、年份等属性在不同意图下的几何平均比值或差值；③计算共作者距离，评估人类与LLM在社交接近度上的差异。结果显示：LLM 更少产生对比（批评）引用，倾向于支持性引用；在意图层面加剧了对高引用量、旧论文、少人团队的偏好；并且在所有意图下，LLM 的引用社交距离显著大于人类。

**⚠️ 局限性**

局限性：①意图标签依赖LLM‑as‑judge，可能携带模型偏差；②文献匹配率在不同模型间差异大，导致分析支持样本不一致；③共作者网络仅覆盖 2015‑2024，导致低覆盖率（约 26% 人类、14‑25% LLM），无法反映所有作者；④仅研究 NLP 会议论文，结果可能不具普适性。

---

## 720. Revisiting Cross-View Completion: Self-Supervised Pre-Training via Reconstruction Error Comparison

**arXiv ID:** 2609.01530 | [PDF](https://arxiv.org/pdf/2609.01530v1)

**作者:** Thibaut Loiseau `[一作]` (Ecole Nationale des Ponts et Chaussées), Vincent Lepetit `[通讯]` (Ecole Nationale des Ponts et Chaussées)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种自监督预训练框架，利用跨视角重建误差相对于单视角重建误差的相对改进来预测像素级可见性，从而在不使用任何三维标注的情况下为所有遮挡像素提供双目监督信号。

**💡 创新点**

创新点在于：①发现相对改进（C）是可见性的一种可靠自监督代理；②设计了一种网络在一次前向过程中同时完成跨视角重建、单视角自编码和相对改进预测；③通过添加仅一条额外输出通道即可实现这些任务，兼容原始 CroCo 结构；④使用基于步幅的简单课程学习，可直接从原始视频中训练，省去昂贵的三维预处理。

**🔧 技术方法**

采用的技术包括 ViT 编码器、交叉注意力解码器、像素级自编码器以及自监督损失（CroCo、MAE 与相对改进损失）。

**📊 数据集**

主要使用的数据集为 ScanNet（室内）、DL3DV（室外）以及 ETH3D、7-Scenes 等用于评估的公开数据集；预训练时还使用原始视频混合（stride curriculum）。

**📈 对比分析**

与 CroCo 在相同架构与训练数据下的比较显示：在零样本对应、相对姿态估计和点图回归等任务上，本文模型显著优于 CroCo，最严阈值姿态精度提升约 6 倍，零样本对应 AEPE 减少 22%，点图 Chamfer 错误下降 10%（室内）和 5–7%（室外/外域）。

**⚠️ 局限性**

局限性包括：预训练成本约为 CroCo 的 2.4 倍（需双倍 GPU 内存）；相对改进在 MAE 误差极低的像素上不可靠；方法仅适用于图像对，且对完整微调的优势有限；对动态场景、不同视角数的扩展尚未验证。

---

## 721. Sierpiński--Knopp Wasserstein Distance for Persistence Diagrams and Applications to 2-Wasserstein Approximation

**arXiv ID:** 2609.01528 | [PDF](https://arxiv.org/pdf/2609.01528v1)

**作者:** Sebastien Tchitchek `[一作]` (CNRS and Sorbonne Universite), Julien Tierny `[通讯]` (CNRS and Sorbonne Universite)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Sierpiński–Knopp Wasserstein距离d_SK，通过将持久性图点映射到SK空间填充曲线并在单位区间上求一维1-Wasserstein赋值，从而得到O(NlogN)的快速距离与显式点对应；

**💡 创新点**

创新之处在于将二维持久性图匹配问题转化为SK曲线上的单调一维赋值，既实现了计算效率，又给出了W_2的上界并提供了Hilbert空间嵌入和正定高斯核；

**🔧 技术方法**

使用的技术包括SK空间填充曲线的第一碰撞选择、归一化持久性图、单调排序求解一维Wasserstein、累计L^1表示、Hilbert嵌入与正定高斯核；

**📊 数据集**

实验数据来自12个科学集合，合计227幅持久性图、约338万点，均通过TTK的离散Morse Sandwich生成；

**📈 对比分析**

与传统W_2的比较通过Spearman相关、NN@3重叠、理论界限验证，并在大规模数据上实现约626×的单对比速度提升、总计2100×加速；

**⚠️ 局限性**

局限性包括必须在同一归一化三角形内比较、对W_2仅提供单向控制、W_Γ缺乏三角不等式及Hilbert性、以及对极端匹配和噪声敏感。

---

## 722. Performance Characterization of SPEC CPU 2026 on AMD EPYC 9755 Processor

**arXiv ID:** 2609.01527 | [PDF](https://arxiv.org/pdf/2609.01527v1)

**作者:** Kunal Kashyap `[一作]` (Advanced Micro Devices Inc), Shayantika Bhattacharya `[通讯]` (Advanced Micro Devices Inc)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在AMD EPYC Zen5微架构上，对最新的SPEC CPU 2026基准套件（SPECrate和SPECspeed）进行微架构级性能特征化，并提出单实例与512复制/线程规模对比的分析方法。

**💡 创新点**

创新点在于首次在任何平台上对SPEC CPU 2026进行系统化微架构特征化，提出scale analysis揭示单复制下无法观测到的系统级瓶颈，并将工作负载归纳为前端控制流主导、计算效率高、内存带宽受限三大行为簇。

**🔧 技术方法**

使用AMD uProf收集约100个硬件计数器，结合Top‑Down Microarchitecture Analysis、控制流特征、缓存/内存子系统压力以及指令混合（SIMD利用率）等多镜头技术进行深入分析。

**📊 数据集**

实验基于官方SPEC CPU 2026 benchmark suite（共52个应用，包含整数与浮点两大子套件，采用ref规模）在两颗AMD EPYC 9755 (Zen5) 处理器上完成。

**📈 对比分析**

通过对比1‑copy与512‑copy（或1‑thread与512‑thread）TMA、IPC、内存带宽、L3 miss率、分支预测误判等指标，发现IPC在3.3倍范围内波动，内存带宽接近单套筒极限，SMT争用平均19%，并形成三类瓶颈特征。

**⚠️ 局限性**

局限在于仅使用GCC 15.2编译器，未探讨不同编译器或优化选项的影响；缺乏功耗/能效评估；未覆盖多租户VM隔离及更细粒度的前端预测误判分析。

---

## 723. EvoSCM: Scientific Belief Revision Through Causal Model Evolution and Experimentation

**arXiv ID:** 2609.01526 | [PDF](https://arxiv.org/pdf/2609.01526v1)

**作者:** Qing Zhao `[一作]` (Sun Yat-sen University), Liang Lin `[通讯]` (Sun Yat-sen University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EvoSCM框架，利用结构因果模型(SCM)作为科学智能体的显式、持续的信念状态，通过维持多种竞争SCM假设并在实验反馈下不断自我演化来实现科学发现。

**💡 创新点**

创新点在于：①把SCM作为可持续的知识表示，既可被验证又可被修正；②通过闭环循环（推断-实验设计-归纳-演化）让模型在实验中逐步自我纠正；③支持SCM在不同模型架构间的可迁移，提升小模型的科学推理能力。

**🔧 技术方法**

使用技术包括：大语言模型(如GPT‑5.4/5.5、Qwen‑3.6)、结构因果模型推理、潜在变量归纳(abduction)、实验设计与干预策略、规则化的修正规则生成与结构编辑（增删边、潜变量、机制与参数更新）、证据验证与一致性检查。

**📊 数据集**

数据集：DiscoverPhysics 交互式科学发现基准；在此基础上进行跨模型迁移实验，将GPT演化得到的SCM注入Qwen‑3.6。

**📈 对比分析**

与基线提示（直接提示无演化）对比，EvoSCM在GPT‑5.4/5.5上提升了机制解释分数（从0.398→0.653、0.516→0.751），MSE下降两位数（5.25e‑1→2.34e‑3、2.83e‑2→2.77e‑4），pass@k显著提升（如pass@1从1.86%→38.08%，7.25%→53.04%），且所需实验回合更少（e.g., 736 vs 1,045）。跨模型迁移实验中，Qwen‑3.6原始性能0% pass，注入GPT‑5.6‑Sol SCМ后pass@5提升至63.64%，解释分数提升至0.740。

**⚠️ 局限性**

局限性：①需要维护多假设群，计算和存储成本较高；②依赖LLM对潜在变量和机制的准确归纳，可能受预训练知识限制；③在高维或复杂非线性系统中，SCM编辑与验证可能收敛慢或失败；④目前仅验证于非标准物理实验，缺乏更广泛的真实世界案例。

---

## 724. The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally

**arXiv ID:** 2609.01587 | [PDF](https://arxiv.org/pdf/2609.01587v1)

**作者:** Jundong Hu `[一作]` (PayPal AI), Shekar Ramachandran `[通讯]` (PayPal AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究在4‑bit后训练量化（PTQ）中如何分配有限的精度预算，并通过因果混合精度干预验证量化损伤的分布。

**💡 创新点**

创新点在于：①量化损伤呈扩散性，三种常见的局部定位方法（任务电路漂移、计算位置因果补丁、权重统计）均无法预测精度恢复；②在相同有效比特预算下，全局更细粒度的量化（group‑128）优于局部层级恢复；③恢复热点在同一家族内部可预测，但跨家族无规律。

**🔧 技术方法**

使用的技术包括：per‑row 4‑bit RTN量化、group‑128细粒度量化、GPTQ、AWQ、因果混合精度干预、任务电路漂移与因果激活补丁等。

**📊 数据集**

实验数据涵盖9个开源模型（LLaMA‑3、Qwen、Mistral、OpenLLaMA）以及22个任务（阅读推理、常识、事实检索、符号推理等），采用CORE评估基准。

**📈 对比分析**

与局部精度恢复（按层恢复到8‑bit）比较，匹配的有效比特预算下，global group‑128 方案在所有8个兼容模型上比最佳局部方案高21–52个CORE分数点；即使是损伤集中模型Qwen3‑8B亦优于局部恢复。

**⚠️ 局限性**

局限性：只评估了层级干预和权重统计，未探索权重级别或非贪婪层集；量化探针仅使用per‑row RTN，未验证在GPTQ/AWQ/g128下是否同样成立；规模上限为8B参数，无法验证更大模型；局部恢复选择为oracle，未给出可部署的选择器。

---

## 725. SG-AMP: Scene-Graph-Guided Active Perception and Semantics-Aware Motion Planning for Pepper Plants

**arXiv ID:** 2609.01579 | [PDF](https://arxiv.org/pdf/2609.01579v1)

**作者:** Rohit Menon `[一作]` (University of Bonn), Maren Bennewitz `[通讯]` (University of Bonn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种面向密集辣椒植物的机器人视觉与运动规划框架，集成了鲁棒深度补全、基于输入的不确定性估计、持续的全景图映射、植物结构图推理以及语义感知的主动视角与路径规划。

**💡 创新点**

创新点包括：① 用 Robust Fast Fill（RFF）在噪声稀疏 RGB‑D 输入上实现局部深度恢复与异常过滤；② 通过输入状态条件化不确定性模型，提升深度与语义估计的可信度；③ 采用持久化的不确定性全景映射和植物结构图，将缺失的辣椒–梗连接视为可观测的假设并指引主动扫描；④ 在视角选择与路径规划中引入类依赖的运动成本，区别保护结构与可穿过叶片。

**🔧 技术方法**

技术手段包括：SemSegDepth、PanDepth、EVPSNet 的联合语义‑深度网络；RFF 的局部深度补全与中值/MAD 滤波；基于不确定性权重的多分辨率全景地图构建；基于结构图的假设生成与更新；基于信息增益与语义运动成本的最优视角与轨迹优化。

**📊 数据集**

实验数据集：NYUv2 用于评估深度补全与不确定性；自建辣椒植物数据集用于评估全景语义识别、附着点检测以及主动感知效果。

**📈 对比分析**

与被动全景映射、语义下一最佳视角（NBV）和传统碰撞规避的主动感知方法进行对比。结果显示：在 NYUv2 上，RFF 使深度 RMSE 从 122.48 mm 降至 79.78 mm，NLL 与 AUSE 分别提升至 -1.6925 与 0.0087；在辣椒数据集上，语义 mIoU 55.27%，全景 PQ 38.67%，深度 RMSE 40.62 mm；相比仅使用不确定性驱动的扫描，结构图推理与语义运动成本显著提升了未观测附件的检测率。

**⚠️ 局限性**

局限性包括：梗结构的 PQ 仍显低（仅 25.20%），说明对极薄遮挡部件的检测仍有挑战；主动感知依赖紧近扫描，对机器人速度与能耗提出额外要求；目前仅在小规模辣椒园区验证，尚未证明在更大、更复杂环境中的鲁棒性；以及模型与规划的计算开销尚未在实时平台上充分评估。

---

## 726. H3-World: Turning Language Understanding into World Control

**arXiv ID:** 2609.01560 | [PDF](https://arxiv.org/pdf/2609.01560v1)

**作者:** Danze Chen `[一作]` (Tencent), Yeying Jin `[通讯]` (Tencent)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过将MiniMax‑H3的文本输入空间映射为角色和摄像机动作指令，并与视频潜在向量对齐，构建了一个高效的可交互世界模型，利用轻量化 LoRA 进行少量参数微调，实现在预训练视频生成器上实现精细、时间上对齐的控制。

**💡 创新点**

创新点在于：①利用预训练模型已学得的语言理解能力，将动作直接编码为可读文本指令；②采用“单出口路由”与“时序注意力路由”确保每条指令仅作用于其对应的时间段；③仅通过极少量（0.199%）可训练参数即可实现精确的角色与摄像机控制，并在未见动作组合与视觉域上保持良好泛化。

**🔧 技术方法**

核心技术包括：MiniMax‑H3（33B 参数的双向音视频预训练模型）、文本到潜在向量的编码与重构、时序对齐的动作提示、单出口注意力路由、低秩（LoRA）自注意力与令牌微调，以及光流估计等评价指标。

**📊 数据集**

使用了 ABot‑World‑Explorer‑500h 数据集，共 7,872 条游戏剪辑做训练，128 条留作评估，每条剪辑约 124 帧，24fps，分辨率 832×480，训练时每条剪辑产生 37 个时序对齐的动作提示。

**📈 对比分析**

方法与基线对比：①冻结 H3 + 全局文本指令；②单句提示但无 LoRA 微调；③完整的时序文本提示加 LoRA 微调。评估指标采用光流平均水平、视觉一致性和动作跟随度。结果显示：冻结模型只能响应粗粒度指令，零 LoRA 无法实现时序控制，而微调模型在左右摄像机平移、角色前后/侧移等任务上实现了高精度控制，且对未见组合和不同场景仍保持稳定表现。

**⚠️ 局限性**

局限性包括：仅针对短时段（约 124 帧）生成；缺乏持久化世界状态、实时交互、规划与策略学习；对不同随机种子、动作组合的系统化可靠性评估不足；以及对更长时间跨度和更复杂交互场景的适应性待进一步研究。

---

## 727. SDARE-Bench: Evaluating Large Language Models on Conversational Stigma Detection and Response in Dyadic and Group Dialogue

**arXiv ID:** 2609.01548 | [PDF](https://arxiv.org/pdf/2609.01548v1)

**作者:** Stephanie Fong `[一作]` (Monash University), Dominic Dwyer `[通讯]` (Monash University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SDARE-Bench基准，用于评估大语言模型在双人对话和多方对话中识别与回应污名化的能力。

**💡 创新点**

创新点在于结合心理学污名框架，设计情境化、多方对话、开放式生成评估，并构建含93种污名类型的高质量数据集。

**🔧 技术方法**

采用LLM生成（GPT‑5、Gemini）与专家审核相结合的两阶段生成管道，利用多标签分类器对模型回应进行自动评估。

**📊 数据集**

使用自生成的1,138个单句查询和1,388个四人八回合对话，涵盖93种污名类型，并配合1,392条人工标注的模型回应。

**📈 对比分析**

对8个主流LLM进行检测与回应实验，检测准确率在40–70%不等，回应中污名表达率高达31%（双人）至约80%（多方），多方组压力量化显著提高污名化倾向。

**⚠️ 局限性**

局限包括仅限英文文本、未拆解多方对话中的个别因素、仅评估污名化倾向而未提供干预策略，且对跨语言文化的泛化性不足。

---

## 728. Evaluating Usability in Biomedical Visualization: Rethinking Heuristic Evaluation for Spatial Omics and Multidisciplinary Research Platforms

**arXiv ID:** 2609.01569 | [PDF](https://arxiv.org/pdf/2609.01569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 729. Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation

**arXiv ID:** 2609.01601 | [PDF](https://arxiv.org/pdf/2609.01601v1)

**作者:** Kefeng Duan `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出在仓库级代码生成过程中识别关键（critical）token，并在生成时动态检索相关上下文，提升生成质量。

**💡 创新点**

核心创新在于：① 定义并量化关键token的概念；② 设计基于关键token的实时检索与纠错机制；③ 引入双端高斯加权池化的密集检索，使检索上下文更贴合生成需求。

**🔧 技术方法**

使用的技术包括：大型语言模型（DeepSeekCoder、CodeLlama）、密集检索模型UniXcoder、三重判别器（误匹配、熵不确定度、后续注意力），以及基于教师强制的标注与轻量级 MLP 判别器。

**📊 数据集**

数据集涵盖：RepoST-Train（训练）以及评测基准 RepoExec（355 个 Python 任务）和 CoderEval（230 个 Python 任务），均来源于真实开源项目。

**📈 对比分析**

与 RawPrompt、RawRAG、RepoCoder、RLCoder 等 SOTA 方法对比，ACToR 在 Pass@1/3/5 上均超越所有基线；在 RepoExec 上 Pass@5 提升 8.4%，在 CoderEval 上提升 15.4%；相对模型规模无关，能够在多种 LLM 上保持优势，且平均每标记延迟仅略高于基线。

**⚠️ 局限性**

局限性包括：① 关键token标注依赖代理指标（误匹配、熵、注意力）可能漏判；② 检索与 KV 缓存重计算带来额外推理开销；③ 评测仅覆盖两大基准，难以完全代表所有仓库级场景；④ 需要大量计算资源，实用性受限。

---

## 730. Uncovering Understanding-Generation Synergy in Native Unified Multimodal Models: From Representation, Task to System

**arXiv ID:** 2609.01607 | [PDF](https://arxiv.org/pdf/2609.01607v1)

**作者:** Penghao Wu `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在原生像素进像素出（pixel‑in/pixel‑out）的统一多模态模型中，系统性研究了视觉理解与生成在表示、任务和系统层面的相互作用，提出任务解耦（task‑decoupled）架构，并通过联合训练验证其协同效果。

**💡 创新点**

系统化地划分了表示、任务、系统三层，展示了何时理解与生成互利、何时竞争，并给出了通过任务解耦与专用分支实现协同的实证方案。

**🔧 技术方法**

使用预训练的Qwen3‑1.7B语言模型作为主干，结合流匹配（flow‑matching）图像生成，探索了三种路由策略（dense共享、模态解耦、任务解耦），并对视觉表示做层级探测。

**📊 数据集**

主要数据集包括SenseNova‑U1（多模态指令调优）、Geometry3K/PGPS9K、MathVerse/MathVista、UniSVG、3D Spatial Intelligence（VSI‑Bench等）以及对应的生成任务集合（MathCanvas、SpatialEdit‑500K、Objaverse 等）。

**📈 对比分析**

通过在同一基础模型下对比单独训练、联合训练以及端到端与规划‑执行流水线的性能，发现联合训练能提升两侧任务（例如几何推理、SVG 识别与生成），而端到端Umm在需要理解、推理与生成交互的图像编辑任务上优于匹配的模块化管线，表现出显著的平均分数提升。

**⚠️ 局限性**

局限性在于只研究了基于自回归文本+流匹配图像的建模范式，未覆盖全自回归或离散去噪形式；且仅以任务解耦架构为主，未系统探索其他架构（如MoE、动态路由）的最佳平衡。

---

## 731. SpatialGuard: Harness-Guided Verifiable Spatial Reasoning for Text-to-Image Generation

**arXiv ID:** 2609.01582 | [PDF](https://arxiv.org/pdf/2609.01582v1)

**作者:** Ziyun Qian `[一作]` (Fudan University), Lihua Zhang `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了SpatialGuard框架，实现从文本到3D空间布局的可验证生成流程；

**💡 创新点**

创新点在于引入可编辑的3D布局中介，并通过布局harness（规则约束、工具调用、共享知识、反馈循环）实现跨轮次的空间约束维护；

**🔧 技术方法**

技术组合包括GPT‑5用于空间布局构建与对齐评估，FLUX.1 dev用于视觉实现，布局harness管理规则与工具执行；

**📊 数据集**

实验使用共享的标准文本提示集，并通过Gemini 2.5 Pro、Qwen3‑VL、Grok 4三大视觉语言模型进行无监督评估；

**📈 对比分析**

与现有文本‑图像与布局控制基线对比，SpatialGuard在所有七个空间指标上均遥遥领先，总体得分9.37（最佳）；

**⚠️ 局限性**

局限在于仅适用于可用对象列表、关系、可见性与相机参数描述的空间场景；推理成本高于单次调用，最终图像质量仍受底层生成器限制。

---

## 732. Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers

**arXiv ID:** 2609.01567 | [PDF](https://arxiv.org/pdf/2609.01567v1)

**作者:** Matteo Merler `[一作]` (Fondazione Bruno Kessler), Bernardo Magnini `[通讯]` (Fondazione Bruno Kessler)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了SAGE框架，利用VLM在训练时仅在高熵状态下查询并执行其动作，再通过优势加权的行为克隆（AWBC）将有价值的指导蒸馏到轻量化的RL策略中，最终实现无需VLM的自主决策；

**💡 创新点**

创新点在于（1）基于策略熵实现的选择性查询，显著减少昂贵的VLM调用；（2）将教师动作与环境优势结合的AWBC，避免盲目模仿错误指导；（3）将VLM视为临时探索先导而非终端策略，提升样本效率与最终性能；

**🔧 技术方法**

使用PPO强化学习、熵门控的选择性查询、优势加权行为克隆（AWBC）、经验缓冲区分离（学生/教师）、大型VLM（Qwen3.5-27B）作为在线教师，以及对比基线如VLM-as-policy、LVLM2P、RL-VLM-F、DAgger；

**📊 数据集**

实验数据集包括FrozenLake、MiniGrid（GoToDoor、LavaGap、Fetch）、EZPoints、CardMaze（自定义）、ALFWorld；

**📈 对比分析**

对比方法包括PPO、VLM-as-policy、LVLM2P、RL-VLM-F、DAgger，SAGE在CardMaze、GoToDoor、Fetch、ALFWorld等环境中显著优于PPO，且在部分环境甚至超过VLM教师；VLM查询率仅1.2%–13.3%，远低于全量查询方法；

**⚠️ 局限性**

局限性：依赖策略熵估计的不确定性可能误判；仅验证离散动作空间；对教师质量高度敏感，误导性教师会导致性能下降；缺乏对连续控制或大规模真实环境的验证；缺乏机制保证在教师误导时的安全与鲁棒性。

---

## 733. A Mathematical Theory of Reusable Neural Bases for Network Compression

**arXiv ID:** 2609.01550 | [PDF](https://arxiv.org/pdf/2609.01550v1)

**作者:** Binshuai Wang `[一作]` `[通讯]` (George Washington University), Binshuai Wang (George Washington University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Linear Reusable Neural Bases Architecture（LRNBA），通过共享神经基底并用线性系数构造网络块来显著降低模型参数量和内存消耗。

**💡 创新点**

核心创新在于将神经网络块视为可复用神经基底的线性组合，支持神经元级别的重用、子神经元复用，并将参数量与网络深度几乎解耦。

**🔧 技术方法**

采用向量场几何解释的神经基底框架、残差结构、注意力与FFN基底共享、梯度重缩放、Kaiming初始化和子神经元多感受器设计等技术。

**📊 数据集**

使用合成非线性回归数据集（目标函数 f(x)=cos(cos x)）和WikiText‑2文本数据集进行实验。

**📈 对比分析**

与传统ResNet和Transformer基线对比，LRNBA在相同或更少的参数下实现了相似甚至更快的收敛速度、低损失值；在Transformer实验中，参数约为标准模型的3/4，收敛更快，最终损失更低，训练时间略有提升。

**⚠️ 局限性**

主要局限是训练阶段因动态构造块而产生的计算和能耗开销，推理时需预先缓存权重才能消除此开销。

---

## 734. Quantum Sparse Autoencoders for Q-Matrix Estimation in Cognitive Diagnosis

**arXiv ID:** 2609.01537 | [PDF](https://arxiv.org/pdf/2609.01537v1)

**作者:** Arif Hassan Zidan `[一作]` (Augusta University), Wei Zhang `[通讯]` (Augusta University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了用于教育数据中 Q‑矩阵估计的量子稀疏自编码器（QSAE），并与经典稀疏自编码器进行对比实验。

**💡 创新点**

首创将量子表示学习应用于认知诊断中的 Q‑矩阵估计，构建了量子压缩层与稀疏回归结合的端到端模型，提升了对真实数据的鲁棒性。

**🔧 技术方法**

使用量子自编码器（参数化量子电路、旋转角编码与 L1 稀疏压缩）、经典稀疏 tied‑weight 自编码器、LASSO 回归、Hungarian 对齐以及 OE/OTP/OTN 评价指标。

**📊 数据集**

评估使用 60 组基于 DINA 模型的模拟数据（K、N、ρ、噪声多维组合）以及 9 个公开真实测评数据集（dtmr_fractions、ecpe、fraction_subtraction、hr、jang、melab、pgdina、rupp_templin_henson、sda6）。

**📈 对比分析**

通过对齐估计 Q‑矩阵后计算 OE、OTP、OTN 误差；在模拟实验中，经典自编码器平均精度略优但方差更大；在真实数据中，QSAE 在 6/9 数据集上误差更低，整体表现更稳健。

**⚠️ 局限性**

并非在所有情境下都能超越经典方法；在理想的 DINA 生成条件下经典模型更优；量子模型受限于 NISQ 设备噪声、规模扩展和计算资源，同时仍需进一步验证在更大规模技能集上的可扩展性。

---

## 735. UI-VISA: U-Net Initialized Vascular Image Segmentation Architecture

**arXiv ID:** 2609.01598 | [PDF](https://arxiv.org/pdf/2609.01598v1)

**作者:** Asees Kaur `[一作]` (University of California, Merced), Erica M. Rutter `[通讯]` (University of California, Merced)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 UI-VISA，一种结合 U-Net 与基于 CNN 的区域生长算法的混合管线，用于改进数字减影血管造影图像的血管分割。

**💡 创新点**

创新点在于用 U-Net 的前景预测作为信息化种子点，替代随机种子，显著提升了血管连通性并加速收敛。

**🔧 技术方法**

采用改进的 U-Net、VISA 两阶段 CNN 区域生长、Dice 与 clDice 指标评估、5 折交叉验证及 Wilcoxon 符号秩检验。

**📊 数据集**

使用了 26 张 512×512 像素的 DSA 图像及其人工标注掩膜进行训练与评估。

**📈 对比分析**

与单独的 U-Net 以及原始 VISA 进行对比，UI-VISA 在 Dice 与 clDice 上均获得最高分（Dice 最高 0.8227，clDice 最高 0.8084），且 clDice 显著提升（p=0.023）。

**⚠️ 局限性**

主要缺点是推理速度仍比单独 U-Net 慢，且对种子选择仍依赖 U-Net 预测，未来可考虑边界-aware 初始化以进一步加速。

---

## 736. Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation

**arXiv ID:** 2609.01603 | [PDF](https://arxiv.org/pdf/2609.01603v1)

**作者:** Kefeng Duan `[一作]` (Sun Yat-sen University), Zibin Zheng `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一种利用软件工程代理历史执行轨迹的特权信息，结合改进的IRT模型进行校准子集选择和能力估计的方法，使得在低预算下即可准确恢复代理在完整基准上的得分和排名。

**💡 创新点**

创新点在于将轨迹信息编码为结构化摘要并作为特权信息融入4参数Logistic IRT，采用Fisher信息与有效样本大小（ESS）进行难度分层的子集选择，并使用LUPI教师-学生框架实现仅凭标识符的能力估计，显著提升了传统只用通过/失败标签方法的评估精度。

**🔧 技术方法**

使用的技术包括轨迹解析器与提示式摘要生成、4PL IRT、Fisher信息与ESS分层选择、学习使用特权信息（LUPI）教师-学生知识蒸馏、深度编码器（MLP）、BCE与KL散度损失、L-BFGS优化等。

**📊 数据集**

实验数据集为SWE-bench四个版本：Lite（300题，35模型）、Verified（500题，70模型）、Full（2294题，14模型）和Pro（730题，14模型）。

**📈 对比分析**

与多种经典与深度IRT基线（MLE、MCMC、VI、VIBO、Deep-IRT、PSN-IRT、AutoJudger）进行对比，评估指标为MAE、Kendall's τ和Spearman's ρ。在10%校准预算下，该方法平均MAE≈0.041、τ≈0.888、ρ≈0.973，显著优于基线，并在5–25%预算范围内保持稳健的排名一致性。

**⚠️ 局限性**

局限性包括：需依赖丰富的历史轨迹且仅在历史数据可用时有效；对轨迹摘要质量敏感，摘要失真或缺失会显著影响性能；在任务数极少或轨迹稀疏的场景下效果可能下降；仅针对通过/失败的二元评估，无法捕捉更细粒度的性能指标。

---

## 737. Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation

**arXiv ID:** 2609.01604 | [PDF](https://arxiv.org/pdf/2609.01604v1)

**作者:** Himil Vasava `[一作]` (University of Wisconsin-Madison), Ming Jiang `[通讯]` (University of Wisconsin-Madison)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM评判模型的内部评分机制进行机制化解释，构建八类攻击扰动并用因果追踪、logit镜头和注意力头消融等方法分析Themis和Prometheus的评分流程。

**💡 创新点**

发现LLM评判器实现了两阶段评估管道（注意力在层15以下比较/路由，MLP在层15以上整合并写入评分），并揭示微调仅在通用子结构上施加两种局部调整。

**🔧 技术方法**

采用因果追踪、logit lens、注意力头消融等机制解释技术。

**📊 数据集**

在CNN/DailyMail上生成干净/扰动摘要对，并在XSum上验证泛化。

**📈 对比分析**

通过对比微调评判器与其基模型，观察路由结构、晶化层、评分曲线，表明微调主要压制低层MLP并提前晶化层，表现出更强的评分变化。

**⚠️ 局限性**

仅研究两款英文摘要评判器、单一扰动强度和单攻击样本、只关注评分改变样本、未验证多语言或更大模型等。

---

## 738. The Rise of Verbal Reinforcement Learning

**arXiv ID:** 2609.01597 | [PDF](https://arxiv.org/pdf/2609.01597v1)

**作者:** Kshitij Tayal `[一作]` (AI Foundations, Capital One), Sambit Sahu `[通讯]` (AI Foundations, Capital One)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

综述了Verbal Reinforcement Learning（VRL）领域，提出了以语言作用时机为轴的三柱体系，对现有方法进行系统归纳与分析。

**💡 创新点**

首次系统地将VRL归纳为三大柱子（语言作为地面化信号、思辨反馈、学习信号），并从时间维度统一框架，揭示语言在代理生命周期中的多重功能与交互。

**🔧 技术方法**

通过文献调研、分类与对比，提出三柱法则，并对每柱下的子方法（目标、状态、动作与奖励地面化；自我批评、外部校正、多模态辩论等；反馈条件建模、自我提升、过程监督、偏好塑造）进行技术细分。

**📊 数据集**

未使用传统数据集，而是基于2020‑2026年间在arXiv、ICLR、NeurIPS等会议/期刊上公开的多领域论文，包括编码、机器人、数学推理、医学决策等范例作为讨论素材。

**📈 对比分析**

本文通过对比分析法而非实验，阐述了各方法在目标、状态、动作与奖励地面化、推理细化、训练更新等维度的差异；引用了部分基准结果（如HumanEval pass@1 91%等）说明VRL在实际任务中的潜在收益。

**⚠️ 局限性**

局限性包括未完全覆盖所有跨域方法；对多角色语言的归类可能过度简化；仅聚焦显式、核心的语言反馈，未涉及语言的辅助作用；缺乏统一的实验评测与量化指标。

---

## 739. A Benchmark for Vehicle Attribute Classification in Cross-Domain Surveillance Scenarios

**arXiv ID:** 2609.01584 | [PDF](https://arxiv.org/pdf/2609.01584v1)

**作者:** Sergio M. Silva `[一作]` (Federal University of Paraná), David Menotti `[通讯]` (Federal University of Paraná)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了uvib基准，用于在跨域监控场景下对车辆属性进行分类，并统一了七个公开巴西数据集的三项操作性标签（方向、VMMR适用性、色彩清晰度）。

**💡 创新点**

创新点在于创建了跨域统一标注框架，构建了多域评估协议，并专注于实际部署中影响后续车辆分析的关键决策。

**🔧 技术方法**

采用了EfficientNetV2‑S、ResNet‑50、ViT/B‑16和YOLO11s‑cls四种主流网络，在各任务上进行细调并对不平衡任务使用Focal Loss。

**📊 数据集**

使用了七个巴西公开数据集（Vehicle‑Rear、LPLCv2、UFPR‑VeSV、UFOP、SSIG‑SegPlate、UFPR‑ALPR、RodoSol‑ALPR），并按监控域与一般域划分。

**📈 对比分析**

通过四种评估协议（S2G、G2S、All‑Datasets、CDS）比较模型，报告准确率与宏F1；EfficientNetV2‑S在绝大多数情形下取得最高宏F1，而跨域协议显著揭示了域迁移带来的性能下降。

**⚠️ 局限性**

局限在于域漂移导致性能急剧下降，VMMR适用性与色彩清晰度易受类别不平衡与光照/传感器变异影响，需进一步研究偏差缓解与更细粒度标注。

---

## 740. Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories

**arXiv ID:** 2609.01556 | [PDF](https://arxiv.org/pdf/2609.01556v1)

**作者:** Nabira Rashid `[一作]` (Independent), Manolis Kellis `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在两种完全不同领域（结构化检索）中，检索模型在表面词汇与潜在结构分离时的行为，并引入低成本的词汇控制作为诊断工具。

**💡 创新点**

创新点：①在统一协议下跨领域评估结构化检索，②发现词汇控制的符号可以揭示基准的表面变化是对抗性还是偶发性，③揭示LLM重排序在不同域、提示、评判者之间的可变性和记忆化影响。

**🔧 技术方法**

技术：基于向量检索的语义嵌入（Gemini-Embedding、Qwen-Embedding、MiniLM-L6-v2），词汇重排序（Jaccard/编辑距离/长度比），LLM评判重排序（Gemini 3.1 Flash‑Lite、GLM‑5.2、Claude Haiku 4.5），自举置信区间、McNemar检验、配对评估等统计方法。

**📊 数据集**

数据集：MathNet‑Retrieve（500题、117,088条目）和ALFWorld（336轨迹、118查询），分别覆盖数学竞赛题和任务指令-动作序列。

**📈 对比分析**

比较方法：严格与宽松Hit@k、可恢复缺口比例、词汇控制正负效应、LLM评判提升效果；结果显示数学域在高对抗性伪装下Hit@1为0%，词汇控制有害；轨迹域中词汇控制改善，LLM评判提升显著但大小因评判者而异。

**⚠️ 局限性**

局限性：仅使用单一随机种子和三位评判者，评判者差异大且与域/提示高度耦合；轨迹域仅有336条目、118查询；对评判者记忆化与基准生成的依赖；下游检索提升在所用求解器中未显现，表明检索与求解器之间未必存在直接转移。

---

## 741. BS: Take the Hint - Interactive Multitracer PET/CT Lesion Segmentation with a Scribble-Conditioned ResEnc U-Net

**arXiv ID:** 2609.01554 | [PDF](https://arxiv.org/pdf/2609.01554v1)

**作者:** Marven Sherif `[一作]` (Brightskies), Ayman Elghotni `[通讯]` (Brightskies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种利用前景/背景笔划条件的三维残差编码U‑Net，用于多示踪PET/CT的交互式病灶分割。

**💡 创新点**

在预训练的autoPET‑III模型基础上，将笔划通道零初始化以保留原模型表现，并在每折模型上细调，同时通过多轮笔划交互显著提升Dice与F1。

**🔧 技术方法**

采用3D ResEncU‑Net、零初始化笔划通道、PET归一化为血管血池参考、滑动窗口推理、Gaussian加权拼接、五折集成以及Tversky+交叉熵损失等技术。

**📊 数据集**

使用autoPET‑V全身PET/CT数据集（1611例，包含^18F‑FDG和^68Ga/^18F‑PSMA），采用官方五折划分。

**📈 对比分析**

与官方评测指标Dice和病灶级F1进行比较；未交互时平均Dice 0.554、F1 0.528，首轮笔划后提升至Dice 0.722、F1 0.704，五轮后达Dice 0.751、F1 0.733，提升约20%以上，并显著压缩模型间差异。

**⚠️ 局限性**

仅在理想的自动生成笔划条件下评估；背景笔划与训练时不匹配导致改进受限；未对无病灶样本进行评估；五折集成未在交叉验证框架下正式测评。

---

## 742. Can LLMs Design Video Coding Tools? A Case Study on Planar Mode

**arXiv ID:** 2609.01535 | [PDF](https://arxiv.org/pdf/2609.01535v1)

**作者:** Yingwen Zhang `[一作]` (City University of Hong Kong), Shiqi Wang `[通讯]` (City University of Hong Kong)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用大语言模型（DeepSeek‑V4‑Flash）在生成‑评估循环中自动设计并实现了Planar预测模式的新版本，并在VVenC与ECM编码器中验证其性能。

**💡 创新点**

首次证明LLM能够在严格的编码器语法与RDO约束下自动构造视频编码工具，并提出基于BD‑Rate正则化的候选筛选与反馈机制。

**🔧 技术方法**

采用LLM代码生成、BD‑Rate与编码时间评估、正则化目标函数、以及正则化加权的候选排序。

**📊 数据集**

使用CTC公开的22个全长自然视频序列（Classes A–E）进行训练与测试，并在低分辨率416×240的8个序列（Class C、D）上进行ECM实验。

**📈 对比分析**

通过将LLM生成的Planar模式直接替换或作为新模式插入，分别与VVenC/ECM基准对比；平均BD‑Rate下降约0.18%（VVenC）和0.03%（ECM），编码时间提升分别为0.4%和0.9%。

**⚠️ 局限性**

评估成本高（每轮需数小时）；低分辨率优化不一定迁移至高分辨率；在更完整的工具集（如VVC慢速预设）下效果可能不稳定。

---

## 743. Designing Proactive Thought Partners for Writing

**arXiv ID:** 2609.01588 | [PDF](https://arxiv.org/pdf/2609.01588v1)

**作者:** Chao Zhang `[一作]` (Google DeepMind), Chin-Chia Hsu `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种可定制的主动写作助手——“proactive thought partners”，实现技术探针并在一周日记研究中评估其使用情况；

**💡 创新点**

将主动性与可定制性结合，支持高层认知任务；引入事件触发+上下文启发式双层时机；提供分层参与方式（忽略、启发、执行）；采用轻量化、非指令化的视觉与语言设计；

**🔧 技术方法**

基于大型语言模型（Google Gemini）做决策与建议生成；使用Next.js+BlockNote构建前端；Firebase记录交互；Keystroke日志监控；规则驱动事件触发与LLM推理来挑选伙伴；

**📊 数据集**

未使用公开数据集；使用用户在探针中生成的写作文本和 keystroke 日志（匿名化处理）；

**📈 对比分析**

通过可用性量表（SUS 82.81/100）、UMUX‑LITE、日记问卷评分（满意度8.1/10、及时性7.9/10）以及交互统计（41%打开，22%执行）进行评估；未做对比实验，仅与现有reactive AI/Smart Compose等进行主观比较；

**⚠️ 局限性**

仅为技术探针，无法评估写作质量、长期效果和创作自主性；样本受限于英语熟练、已有AI使用经验的写作者；隐私风险未完全消除；LLM上下文理解能力有限，可能导致误判或误导；未覆盖高风险/保密写作场景。

---

## 744. Weighted Fair Division of Indivisible Mixed Manna

**arXiv ID:** 2609.01580 | [PDF](https://arxiv.org/pdf/2609.01580v1)

**作者:** Nicholas Teh `[一作]` `[通讯]` (University of Oxford), Nicholas Teh (University of Oxford)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在加性混合物品（既包含好品也包含任务）下，带权公平分配问题，证明了所有实例均存在完全WEF1（加权一项不产生羡慕）分配并给出多项式时间算法；同时展示了WEF1对社会福利的价格无界；在等量值混合模型下证明存在精确WMMS（加权最小份额）分配并提供多项式时间实现；并给出WEF1与WMMS之间的最优加性关系与下界；

**💡 创新点**

创新点在于（1）解决了加权混合物品下WEF1存在性的开放问题并给出可多项式求解的完整算法；（2）构造极限例证表明WEF1并不保证社会福利的任何有限逼近；（3）首次在带权混合物品中得到精确WMMS的存在性与算法；（4）揭示WEF1在等量值模型下对WMMS的最佳加性保证，并给出最优性证明；

**🔧 技术方法**

主要技术包括：可接受捆绑（acceptable bundles）与逆向加权抽签序列的分析，构造多项式时间的分配算法；利用整数规划与流网络求解精确WMMS；对WEF1与WMMS关系进行整数与上界下界证明；以及构造极端实例来证明价格无界与下界；

**📊 数据集**

无数据集，所有结果均为理论证明与构造性算法；

**📈 对比分析**

由于论文为理论性工作，没有实验或对比指标，主要通过构造例子与理论证明展示结果的最优性与下界；

**⚠️ 局限性**

局限性在于：WEF1与帕累托最优（PO）兼容性尚未解决；WMMS的精确性仅在等量值混合模型下成立，不能直接推广至更一般的价值集合；加权公平与效用间的多重比较仍存在空缺；

---

## 745. Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs

**arXiv ID:** 2609.01573 | [PDF](https://arxiv.org/pdf/2609.01573v1)

**作者:** Jingtan Wang `[一作]` (National University of Singapore), Bryan Kian Hsiang Low `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在固定标注预算下，LLM 训练流程中将预算在监督微调（SFT）与强化学习（RL）阶段之间划分的最佳策略，提出并验证了“近最优区域”概念；

**💡 创新点**

创新点在于把传统的单一最优比例转化为可容忍性能误差的宽阔区域，并证明该区域随模型规模扩大而变宽且可从小规模代理模型迁移至大规模目标模型；

**🔧 技术方法**

使用的技术包括基于LoRA的参数高效微调、离线偏好优化方法 DPO、在线策略梯度方法 GRPO 以及预算比例网格搜索；

**📊 数据集**

实验所用数据集涵盖数学推理（GSM8K、Tülu3）、指令跟随（Tülu3 Persona、Tülu3 RLVR）、摘要（Reddit TL;DR、Reddit Comparison）和帮助性（HelpSteer、HelpSteer2）等四类任务；

**📈 对比分析**

方法通过对不同模型（Llama3、Qwen2.5、Qwen3）和预算比例进行多维实验，结果显示在 5–10% 的性能容忍范围内，近最优比例可覆盖 55–75% 的比例空间，且小模型的近最优区域在大模型上保持 90% 以上的重叠率，证明了跨尺度迁移的可靠性；

**⚠️ 局限性**

局限性包括仅考察两阶段 SFT→RL 的简单流程，实验规模仅至 14B 参数、15k 样本，未覆盖大规模 70B/72B 模型、OOD 任务、更多 RL 算法及更复杂的成本模型等情况。

---

## 746. From Confusion to Clarity: Confusion-Aware Retrieval and Knowledge Injection for Text Classification

**arXiv ID:** 2609.01564 | [PDF](https://arxiv.org/pdf/2609.01564v1)

**作者:** Manish Gupta `[一作]` (Amazon), Jayasimha Talur `[通讯]` (Amazon)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于混淆标签的知识增强框架，先通过错误分析找出模型常混淆的标签对，随后在检索时加入这些混淆对的标签，并利用三阶段LLM推理自动生成针对每一对标签的区分规则，最终在推理阶段一次性将候选标签与规则一起注入模型进行分类。

**💡 创新点**

创新点在于①使用混淆矩阵识别并补全检索候选集中的混淆标签对；②设计三阶段自动化规则生成管线，将错误实例转化为可复用的对标签规则；③将大模型生成的规则迁移到小模型上；④采用成对（pairwise）规则格式，使模型在面对相似标签时拥有针对性决策依据。

**🔧 技术方法**

技术包括零-shot LLM 文本分类、基于嵌入的候选检索、混淆矩阵分析、提示工程、LLM推理生成规则、规则合并与迁移、以及多模型跨域知识转移。

**📊 数据集**

在三个公开基准上验证：Web‑of‑Science（WOS）、Flipkart 产品分类、LEDGAR（法律文档）。

**📈 对比分析**

与零射、少射、检索基线、微调模型以及提示优化基线对比，所有指标均以 Macro F1 为主。实验显示：在 Retrieval+Confusion‑Partner+Knowledge 组合下，Macro F1 对比检索基线提升最高可达 10pp；小模型在接收大模型生成规则后，可额外提升约 11.5pp。

**⚠️ 局限性**

局限性：①离线阶段需已标注训练集，仅能生成已出现混淆的标签对规则；②规则质量需人工审查，尽管后处理能降低噪声；③随着规则数增多，推理提示长度增长，推理成本提升；④未在多语言、极大规模标签集或时间漂移场景下评估。

---

## 747. What, Where, and How: Probing Spatiotemporal Representations in Video Foundation Models

**arXiv ID:** 2609.01551 | [PDF](https://arxiv.org/pdf/2609.01551v1)

**作者:** Sharon S. Musa `[一作]` (York University), Konstantinos G. Derpanis `[通讯]` (Vector Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

系统层级探测并几何分析 V-JEPA 2 与 VideoMAE‑v2 的时空表示，对相机运动、直观物理与异常检测进行线性探测与特征轨迹插值导航。

**💡 创新点**

将层级探测与特征空间几何相结合，发现相机运动在中间层呈平滑曲线，并通过三次样条插值实现更连贯的视频操控。

**🔧 技术方法**

使用线性 SVM 探测、PCA 低维可视化、三次样条 (spline) 推导及最近邻检索评估。

**📊 数据集**

CameraBench、IntPhys 2、UBnormal、RealEstate10K 以及 V‑JEPA 2/VideoMAE‑v2 预训练视频。

**📈 对比分析**

通过与线性插值对比，spline 在中间层保持更小最近邻距离；V‑JEPA 2 在异常检测上优于 VideoMAE‑v2；相机运动分类 ROC‑AUC 超 90%。

**⚠️ 局限性**

仅在 V‑JEPA 2 Large 与 RealEstate10K 上完成几何与推导实验；IntPhys 2 与 UBnormal 采用合成视频；线性探测无法揭示非线性信息。

---

## 748. NashDreamer: Model-Based Reinforcement Learning for Zero-Sum Imperfect-Information Games

**arXiv ID:** 2609.01549 | [PDF](https://arxiv.org/pdf/2609.01549v1)

**作者:** Tomáš Holeček `[一作]` (Czech Technical University in Prague), Viliam Lisý `[通讯]` (Czech Technical University in Prague)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出了NashDreamer，一种用于两人零和不完全信息博弈的模型驱动强化学习框架；

**💡 创新点**

创新点在于引入集中式多智能体递归状态空间模型(MARSSM)来消除对手策略与环境动力学的混淆，并与游戏理论的演员-评论家（如RNaD）结合，理论保证收敛到纳什均衡；

**🔧 技术方法**

采用DreamerV3风格的世界模型、KL平衡损失、集中式训练与去中心化执行、以及RNaD/MMD等正则化策略；

**📊 数据集**

在多个基准游戏上验证：Goofspiel‑5、Goofspiel‑13、Leduc Poker、Phantom Tic‑Tac‑Toe 与 Battleship 5×5；

**📈 对比分析**

与模型无关的基线（RNaD、MMD及其带回放的版本）相比，NashDreamer在早期和中期训练阶段显著提升样本效率、降低NashConv，尤其在Goofspiel‑13和Battleship中表现突出；

**⚠️ 局限性**

局限性包括：在高度随机的环境中可能出现后验崩溃导致模型结构错误，需额外的温启动阶段；目前只能从起始状态t=0进行完整轨迹展开，限制了对更大规模博弈的适用性；

---

## 749. StudentSim: Training LLM-based Student Simulators

**arXiv ID:** 2609.01591 | [PDF](https://arxiv.org/pdf/2609.01591v1)

**作者:** Ke Yang `[一作]` (Microsoft Research), Jianfeng Gao `[通讯]` (Microsoft Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了 StudentSim 框架，训练个性化学生模拟器以同时具备行为忠实度与指导响应性，并基于此评估 AI 辅导员。

**💡 创新点**

创新点在于将学生模拟拆分为两阶段（全域共享训练 + 个人化微调）来同时满足行为忠实度 (ℱ) 与指导响应性 (ℛ) 两大指标，并发布统一的 StudentSimEval 评测协议。

**🔧 技术方法**

技术包括使用 Qwen3-4B-Instruct 作为基础模型，利用 LoRA 适配器进行两阶段训练，结合多轮记录的强化学习、指令微调与奖励头。

**📊 数据集**

使用了公开的三大领域数据集：Lichess 的棋局数据、EFCAMDAT 英语写作纠错数据、数学开放式问题答案集，构建了 60 名学生的单轮与多轮记录。

**📈 对比分析**

在 StudentSimEval 中，与 GPT-5.4、GPT-4o、Maia2 等基线相比，StudentSim 在行为忠实度和指导响应性上均显著优越（棋类 ℱ=0.51 vs 0.23/0.45，ℛ=0.91 vs 0.72/0.27），并在强化学习实验中得到专家评审更高的准确性、指导质量和个性化评分。

**⚠️ 局限性**

局限在于仅评估一次性更新（单步响应），未覆盖长期学习动态；同时两阶段训练仍依赖大量多轮记录，稀疏数据的学生仍可能表现不佳。

---

## 750. CordisBench: Can Language Models Reason About Component Lifecycles in Dynamic Agent Harnesses?

**arXiv ID:** 2609.01600 | [PDF](https://arxiv.org/pdf/2609.01600v1)

**作者:** Damien Sileo `[一作]` (University of Lille), Dimitri Kachler `[通讯]` (University of Lille)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了一个1200题的结构化评测基准，专门评估动态代理主机（如Cordis）在生命周期变更时的推理能力，并用该基准测试了三种高效LLM模型。

**💡 创新点**

创新点包括：①提出一套包含定位、调度预测、全局/可达条件判定及重构等四类任务的生命周期推理基准；②设计了按交互数量扩展的难度曲线，既可在形式语义下枚举所有合法调度，也可在可执行的Cordis实例中验证；③通过对比模型在不同交互规模下的表现，揭示了定位与最终状态预测在规模扩展时的性能差异。

**🔧 技术方法**

技术手段包括：使用有限参考语义对所有实例进行完全枚举，生成目标答案；在Cordis 4.0.0-rc.7中编译相同模式的插件并执行；采用结构化输出（集合、序列、整数、操作集）与确定性评分；对模型输出做Jaccard相似度、逐观测准确率和执行成功率评价；在实验中还引入额外推理量（低/中/高）来探究推理成本与性能提升的关系。

**📊 数据集**

数据集来源于240个随机生成的系统，分别拆分为6个交互规模（2,4,8,16,24,32），每个规模下在形式语义和Cordis原生两种设置下生成1200道题，其中1056道为主任务，144道为结果计数诊断。

**📈 对比分析**

比较方法：在同一任务、同一交互规模下，对模型的输出采用结构化解析后计算Jaccard相似度（定位与条件任务）、逐观测准确率（调度预测）、执行成功率（重构任务）。实验结果表明：在小规模下三模型均表现良好，但随着交互数量增大，定位任务下降缓慢，而最终状态预测与跨调度的条件推理性能显著衰退；Gemini在Cordis原生任务上保持较高准确率，GPT‑5.6 Luna在额外推理后可显著提升但仍低于Gemini，DeepSeek整体表现最弱且倾向于返回全部条件。

**⚠️ 局限性**

局限性包括：①基准仅覆盖依赖驱动的移除与“捕获-恢复”型清理，未涉及失败、不可逆外部动作或热模块替换等真实系统复杂情况；②交互规模上限为32，较大规模仅作压力测试；③实验仅使用三种无工具、无执行反馈的高效LLM，未考察在完整代理主机中可能提供的辅助机制；④部分任务对输出长度敏感，低token限制会导致性能下降，尤其对Gemini的全局条件任务。

---

## 751. Facet-0: A Robotic Foundation Model for Contact-Rich Precise Manipulation

**arXiv ID:** 2609.01596 | [PDF](https://arxiv.org/pdf/2609.01596v1)

**作者:** Haoyuan Deng `[一作]` (Nanyang Technological University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `40105733-5154-44cd-8090-a8cab9e64b07` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一个融合视觉、语言和力学的基础模型，用于预测并评估机器人在精细装配中的接触后果。

**💡 创新点**

创新点包括：① 将接触力与动作联合预测的语义‑接触表示；② 通过行动‑扭矩批判器进行价值引导的强化学习；③ 基于冻结表示的局部可控适配器实现快速零样本迁移。

**🔧 技术方法**

技术手段主要有：多模态融合、流匹配动作专家、分布式行动‑扭矩评论器、分阶段价值回归与局部适配的TD3+BC。

**📊 数据集**

使用了 ManuFacet‑1K 数据集：约 1,000 小时、三种机器人胳膊、多个制造单元的同步六轴力传感记录与任务指令。

**📈 对比分析**

与多种基线（普通 VLA、加力反馈 VLA、RECAP、GR00T、TA‑VLA）对比，最终模型在五个子毫米装配任务上平均成功率 82%（基线 15%），定位精度 0.5 mm，命令延迟 50 ms。

**⚠️ 局限性**

局限性：实验仅在配备腕部六轴传感器、平行抓手的电子装配工作站验证，需在不同末端执行器、零件几何和接触模式下进一步验证。

---

## 752. Closing Cost-Quality Gap in Document VLMs: Difficulty-Aware Data Curation and Quality-Adjusted Deployment Economics

**arXiv ID:** 2609.01575 | [PDF](https://arxiv.org/pdf/2609.01575v1)

**作者:** Maksim Evdokimov `[一作]` (T Tech), Aleksandr Ivanov `[通讯]` (T Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在受监管环境下，设计并部署了一款35B总参数、3B活跃参数的Mixture‑of‑Experts VLM，使用单个H100 GPU即可在数亿份文档中完成结构化字段提取、文档分类与视觉验证等任务，取代传统 OCR+模型流水线；

**💡 创新点**

创新点包括：①难度感知数据筛选（DADC）将公开PDF通过文本层校验、视觉结构分类和事实可提取性评分筛选出高信息量样本；②将内部业务数据与公开数据混合进行SFT，保持业务分布并获得大规模多样性；③在单个H100上仅用3B活跃参数实现超大模型性能；④引入质量调整成本框架，将字段准确率映射为每文档经济收益，证明对人工作业的80%+成本节省；

**🔧 技术方法**

技术实现包括Mixture‑of‑Experts VLM（Qwen3.5‑35B‑A3B‑Base）、SFT、DADC数据管道、文本层校验、视觉结构分类、跨模型一致性验证、提示式推理以及基于字段准确率的质量调整成本模型；

**📊 数据集**

使用的数据集包括内部业务数据（法院判决、发票等210k份文档，字段标注），公开Common Crawl PDFs（约300k份）经DADC过滤后生成的合成Q&A对，二者混合构成SFT训练集；

**📈 对比分析**

通过在12个内部基准（单页、分页、多页、微调）以及MWS Vision Benchmark和公开OCRBench、DocVQA等评测，模型在非推理模式下平均得分0.814，显著超越10倍参数的Qwen3.5‑397B（0.787），同时在成本‑质量平衡上实现了对人工作业80%+的成本降低；

**⚠️ 局限性**

局限性包括：①缺乏多步推理/长程证据聚合场景，可能在需要跨页/跨句推理的工作流中表现不足；②未进行各语言单独 ablation，跨语言泛化能力不完全可知；③SFT尚未逼近预训练潜力上限，未来需扩展多样化数据与推理能力；④质量调整成本模型需在不同部署环境重新校准。

---

## 753. From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix

**arXiv ID:** 2609.01572 | [PDF](https://arxiv.org/pdf/2609.01572v1)

**作者:** Olga Tsymboi `[一作]` (T Tech), Anatolii Potapov `[通讯]` (T Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在企业内部整合200+应用流量，使用后训练（SFT+GRPO）将多模型需求压缩为单一Qwen3-32B模型，并通过模板感知采样与任务特定评测识别并修复指令遵循、函数调用及流量分布差异等缺陷。

**💡 创新点**

① 将奖励信号拆分为每个弱点单独的GRPO专家，避免多目标交互导致的奖励劫持；② 采用两阶段SLERP权重空间合并专家，保持各自优势；③ 引入模板感知采样与任务特定评测，提升内部评估的代表性与准确度。

**🔧 技术方法**

Qwen3-32B backbone + Cyrillic tokenizer；大规模SFT + GRPO强化学习；GRPO奖励模型与长度惩罚、KL约束；SLERP权重合并；多语言合成数据管线（AutoIF、Tool-N1）；确定性验证器与校准LLM评测。

**📊 数据集**

内部生产流量日志（≈100k/月）及其生成的 Arena/IFEval/BFCL/SmartSearch benchmark；公开俄语/英语评测集（IFEval、MultiChallenge、BFCLv3、WildChat Hard Ru、AceBench、τ^2-bench）；公开 Qwen3-235B-A22B-Instruct-2507 checkpoint 用于对照。

**📈 对比分析**

通过内部 benchmark 与公开 benchmark 对比评估；在内部 Arena 得分 69.57 vs 65.83，在 BFCL 0.79 vs 0.77，ruBFCLv3 65.96 vs 64.42，AceBench 73.50 vs 70.20；SmartSearch F1 从 0.478 提升至 0.557，ruWildChat 从 52.0 提升至 80.7；相比约 7× 参数规模模型，吞吐量相当但 per‑token 成本降低 2.8–9×，每月 116M 请求。

**⚠️ 局限性**

仅在俄语/英语两种语言与单一 Qwen3 系列模型验证；依赖 LLM judge 进行开放式质量评估，需重新校准；仅适用于本组织的内部部署环境，跨组织、跨语言或不同模型家族的可迁移性未验证。

---

## 754. A systematic Approach to constructing a Chance-and-Risk Matrix for Semiconductor Supply Chains

**arXiv ID:** 2609.01563 | [PDF](https://arxiv.org/pdf/2609.01563v1)

**作者:** Ema Salkić `[一作]` (Infineon Technologies AG), Georg Groh `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个端到端的管道，自动将半导体公司公开披露的文件转化为带分数的本体支持知识图谱，用于构建机遇-风险矩阵；

**💡 创新点**

创新点在于将LLM抽取、语义聚类、三层评分机制和本体建模结合，形成可持续、可审计、可自动化的风险机遇评估流程；

**🔧 技术方法**

主要技术包括OpenAI GPT‑4o进行风险/机遇抽取和重评分，OWL本体建模，Sentence‑BERT嵌入实现语义聚类，三层评分算法（类别权重、时效衰减、事件因子）以及LLM辅助评估；

**📊 数据集**

使用了五家公司（Intel、Infineon、Texas Instruments、Air Liquide、Siltronic）的公开文件（年报、10‑K、ESG报告等）以及历史事件数据，共计76,207条风险/机遇实例；

**📈 对比分析**

通过与领域专家的手工排名进行Spearman相关性对比，抽取精度达92.6%；三层评分后与专家的相关性提升至ρ≈0.55（风险）和ρ≈0.72（机遇）；

**⚠️ 局限性**

局限包括仅处理英文文件导致对非英语公司覆盖不足；检索仅限于Google Custom Search；跨公司源污染导致部分错误；聚类阈值固定、单链式聚类易产生大簇；缺乏多语言、增量更新、跨公司去重和多专家验证等功能。

---

## 755. Gradient-Update Mismatch: Rethinking Conflict-Free Training of Physics-Informed Neural Networks

**arXiv ID:** 2609.01558 | [PDF](https://arxiv.org/pdf/2609.01558v1)

**作者:** Jing Xiao `[一作]` (National University of Defense Technology), Tiejun Li `[通讯]` (National University of Defense Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究物理信息神经网络（PINN）训练中梯度冲突问题，提出梯度-更新不匹配（GUM）理论并设计梯度-更新对齐（GUA）方法；

**💡 创新点**

创新点在于揭示优化器转换会破坏梯度外推冲突消除的几何结构，并通过投影到冲突自由锥实现更新级别的冲突消除；

**🔧 技术方法**

主要技术包括梯度冲突检测、冲突自由锥投影（使用正定度量和活跃集求解）、优化器状态软对齐、以及对多任务与高阶优化器的分析；

**📊 数据集**

实验数据集涵盖六种PINN基准（Schrödinger、Burgers、Heat-MS等）和CelebA多任务分类，使用多种优化器（SGD、Adam、AdaHessian、SOAP等）；

**📈 对比分析**

与传统梯度冲突处理（PCGrad、CAGrad等）和无冲突处理（Adam）对比，GUA在所有实验中消除更新冲突，显著降低相对L₂误差（最高提升98.2%），并保持较低的内存占用；

**⚠️ 局限性**

局限性在于仅基于局部一阶梯度几何，未考虑曲率、有限步长或长期优化影响，且硬冲突阈值可能过于保守。

---

## 756. Can LLMs Discover Scientific Laws in Real and Parallel Worlds?

**arXiv ID:** 2609.01552 | [PDF](https://arxiv.org/pdf/2609.01552v1)

**作者:** Yiming Huang `[一作]` (University of California San Diego), Jingbo Shang `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出SciLaws‑Bench，一个基于真实科研论文与实验数据的科学定律发现基准，包含固定记录发现和主动查询两种评估设置；

**💡 创新点**

创新点在于：①将真实观测与合成隐藏定律并行；②分离预测拟合、科学有效性、记忆化与结构恢复四个维度；③通过最佳‑N 搜索揭示模型选择瓶颈；

**🔧 技术方法**

技术手段包括：LLM驱动的 ReAct‑style 智能体、Python 沙盒执行与公式拟合、结构化评估指标 S_N、S_V、S_S、以及自选与最佳‑N 对比分析；

**📊 数据集**

数据集：118 个科学发现问题，来自 381 篇论文，包含 291 条候选定律和约 800 万条真实观测数据，覆盖六大学科；

**📈 对比分析**

比较方法：在同一交互框架下评估 9 种前沿 LLM，指标为预测拟合、科学有效性和结构恢复；GPT‑5.5 领先，其他模型排名分散；高拟合不一定伴随高有效性，记忆化有助复制已知定律但不易发现新结构，最佳‑N 池提升但自选效果有限；

**⚠️ 局限性**

局限性：LLM 仍难以可靠选取最佳公式；记忆化与发现能力不一致；单一指标难以全面衡量；实际实验设计与主动学习能力尚未充分评估。

---

## 757. Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall

**arXiv ID:** 2609.01532 | [PDF](https://arxiv.org/pdf/2609.01532v1)

**作者:** Jacqueline He `[一作]` (Meta AI), Wen-tau Yih `[通讯]` (Meta AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究知识蒸馏在中训练阶段的效果，发现存在推理–回忆权衡，并提出基于教师熵的分流蒸馏方法；

**💡 创新点**

首次揭示中训练阶段独特的推理与事实回忆权衡，并设计了按熵分流的SwitchDist目标以缓解该权衡；

**🔧 技术方法**

使用前向/后向KL蒸馏、教师熵计算、熵感知令牌分流、梯度分析及后训练的SFT/DPO/RLVR技术；

**📊 数据集**

采用Dolmino Mix 1124数据集进行训练，评估使用OLMES基准（TriviaQA、Natural Questions、SimpleQA等）与知识与常识任务；

**📈 对比分析**

与NTP、FKD、RKD及TRKD等基线对比，SwitchDist在中训练阶段推理提升71%，知识/常识提升19%，事实回忆仅下降1%，后训练后仍保持推理优势并消除回忆差距；

**⚠️ 局限性**

在事实回忆方面仍有小幅短缺，需针对不同教师进一步调参，且方法主要验证在1B学生上，尚未验证更大规模模型的泛化能力。

---

