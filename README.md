# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-25 | 今日论文总数: 585

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Leveraging Large Language Models to Extract and Translate Medical Information in Doctors' Notes for Health Records and Diagnostic Billing Codes

**arXiv ID:** 2603.22625 | [PDF](https://arxiv.org/pdf/2603.22625v1)

**作者:** Peter Hartnett `[一作]` (Frostburg State University), David Hartnett `[通讯]` (Florida Atlantic University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了基于 Ollama + LangChain 的本地离线大语言模型管道，实现医生笔记自动抽取并生成 ICD‑10‑CM 诊断编码；同时构建了合成的医学笔记基准数据集。

**💡 创新点**

创新点在于：① 采用本地化、容器化的隐私保护架构；② 引入结构化输出（JSON Schema）显著提升输出一致性；③ 通过 RAG 技术动态检索 ICD 代码库来补偿模型规模不足；④ 提供可公开复现的合成笔记与评测基准。

**🔧 技术方法**

使用技术包括：开源大语言模型（Llama3.2、Mistral、Phi4、Deepseek、Gemma 等），Ollama 的结构化输出接口，LangChain 的 RAG 检索链，Docker 容器化部署，JSON Schema 校验。

**📊 数据集**

数据集：5 条人工合成的医生笔记（无真实 PHI），对应的 ICD‑10‑CM 代码；以及从 CDC 下载的 74,719 条 ICD‑10‑CM 代码文本文件用于 RAG 上下文。

**📈 对比分析**

评测方法：对比 zero‑shot、few‑shot、RAG 三种提示策略，在 5 模型（Deepseek‑r1 8B、Llama3.2 8B、Mistral 7B、Phi4 14B、Gemma 270M）上计算 JSON 结构合规率、诊断码准确率、诊断提取准确率及推理时间。结果显示：JSON 合规率接近 100%，但诊断码准确率仅 12%；few‑shot 未提升反而导致过拟合；RAG 在部分新代码上有所改善但总体诊断提取准确率下降；模型尺寸越大，推理速度越慢。

**⚠️ 局限性**

限制：① 受限于 1‑20B 参数的小模型，推理能力不足以完成复杂的 ICD 代码映射；② few‑shot 提示导致模式复制与过拟合；③ RAG 上下文窗口易饱和，增加噪声；④ Ollama 结构化输出不兼容某些模型（如 gpt‑oss）；⑤ 缺乏大规模真实数据，基准样本不足；⑥ 仍需人工审核以防止误码与幻觉，难以直接用于临床。

---

## 2. Tiny Inference-Time Scaling with Latent Verifiers

**arXiv ID:** 2603.22492 | [PDF](https://arxiv.org/pdf/2603.22492v1)

**作者:** Davide Bucciarelli `[一作]` (University of Modena and Reggio Emilia), Rita Cucchiara `[通讯]` (University of Modena and Reggio Emilia)

**通讯引用:** 19995 | [OpenAlex ID](https://openalex.org/A5030948871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Verifier on Hidden States（VHS），一种利用Diffusion Transformer内部隐藏状态直接验证的推理时间缩放方法；

**💡 创新点**

创新点在于剔除传统视觉编码器，将生成器隐藏层与LLM对齐，显著降低验证开销；

**🔧 技术方法**

采用单步DiT生成器、LLM（如Qwen2.5-0.5B）与线性连接器、加权交叉熵/焦点损失以及对齐训练阶段；

**📊 数据集**

使用GenEval基准、SANA‑Sprint单步生成器、以及公开的图像-字幕对（LLaVA训练集）进行训练与评估；

**📈 对比分析**

与CLIP+LLM、AE+LLM等传统验证器对比，VHS在相同时间预算下提升GenEval整体分数3.1%/1.7%/0.5%，并将生成+验证时间降低63.3%，FLOPs降低62.9%，显存降14.5%；

**⚠️ 局限性**

局限包括对单步生成器的依赖、隐藏层选择需手工调优，以及在单目标/颜色等简单任务上的提升有限，尚需验证在更大模型或多模态场景中的泛化能力。

---

## 3. Beyond the Mean: Distribution-Aware Loss Functions for Bimodal Regression

**arXiv ID:** 2603.22328 | [PDF](https://arxiv.org/pdf/2603.22328v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 4. MIOFlow 2.0: A unified framework for inferring cellular stochastic dynamics from single cell and spatial transcriptomics data

**arXiv ID:** 2603.22564 | [PDF](https://arxiv.org/pdf/2603.22564v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 5. High Resolution Flood Extent Detection Using Deep Learning with Random Forest Derived Training Labels

**arXiv ID:** 2603.22518 | [PDF](https://arxiv.org/pdf/2603.22518v1)

**作者:** Azizbek Nuriddinov `[一作]` (Florida State University), Mohammad Reza Alizadeh `[通讯]` (Michigan State University)

**通讯引用:** 1081 | [OpenAlex ID](https://openalex.org/A5053869916)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对高分辨率光学影像（PlanetScope 3m）下的洪水映射，构建了一个渐进式标注框架：先用专家人工标注单一样本训练随机森林生成训练标签，再以此标签训练U‑Net网络，实现洪水区域的语义分割。

**💡 创新点**

创新点在于将随机森林作为中间标注生成器，将单一专家标注规模化；将高分辨率光学数据与高度上近河道（HAND）和坡度等地形特征融合；以及在低观测数据环境下实现快速、可操作的洪水地图。

**🔧 技术方法**

采用的技术包括：随机森林分类器、ResNet18编码器的U‑Net深度学习模型、NDWI指数、HAND与坡度特征、Dice 损失函数，以及特征重要性分析。

**📊 数据集**

使用的数据集为：2021年9月2日的PlanetScope 3m光学影像（蓝绿红近红外四波段），USGS 10m DEM提取的坡度、NHDPlus 10m分辨率的HAND；并以140个USGS HWM点进行验证。

**📈 对比分析**

通过与随机森林和U‑Net四波段模型比较，评估指标为F1、IoU、精度、召回率。结果显示：6波段U‑Net（含HAND与坡度）F1=0.92、IoU=0.85，略优于4波段U‑Net（F1=0.92、IoU=0.85），远超随机森林（F1=0.829、IoU=0.708）。

**⚠️ 局限性**

局限性包括：仅使用单幅影像，无法捕捉洪水时间演变；光学数据缺乏短波红外带，影响水体判别；云覆盖、缺乏多时相数据；以及对专家标注质量的高度依赖，导致生成标签不完美。

---

## 6. SkillClone: Multi-Modal Clone Detection and Clone Propagation Analysis in the Agent Skill Ecosystem

**arXiv ID:** 2603.22447 | [PDF](https://arxiv.org/pdf/2603.22447v1)

**作者:** Jiaying Zhu `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 49853 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了针对代理技能文档的多模态克隆检测方法，并构建了技能克隆基准和生态系统级克隆图。

**💡 创新点**

创新点在于融合YAML、自然语言和代码三种模态的相似度，并通过二次特征交互与逻辑回归实现高精度克隆判定和类型分类。

**🔧 技术方法**

采用TF‑IDF+LSA编码、余弦相似度、结构特征、二次交互特征及逻辑回归融合。

**📊 数据集**

使用从GitHub和SkillsMP抓取的约20,000个公开技能及构造的3,000对基准数据。

**📈 对比分析**

与平面TF‑IDF、MinHash、单模态基线比较，F1提升至0.939，Type‑4克隆召回率提升4.2倍。

**⚠️ 局限性**

局限在于仅关注文件内三模态，未利用版本历史或外部资源，基准样本有限且人工标注依赖衍生关系。

---

## 7. Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection

**arXiv ID:** 2603.22658 | [PDF](https://arxiv.org/pdf/2603.22658v1)

**作者:** Mattia Gatti `[一作]` (University of Insubria), Fabiano Monti `[通讯]` (Alpsolut S.r.l.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过双时相Sentinel-1 SAR图像的深度学习变更检测实现大规模雪崩映射。

**💡 创新点**

证明单一SAR输入即可达到甚至优于多模态（含地形信息）的性能，并公开了多地区验证数据集。

**🔧 技术方法**

使用Swin Transformer V2 Tiny及其多分支编码-融合架构进行像素级分割，配合阈值调优与拼接策略。

**📊 数据集**

采用四个高雪区域的手工验证雪崩库存（Livigno、Nuuk、Pish、Tromsø）构建训练、验证和测试集。

**📈 对比分析**

相较于SiamUnet、STANet、BIT等基线模型，本方法在F1≈0.80、IoU≈0.66的同时，推理速度快、参数少；F2阈值进一步提升召回率。

**⚠️ 局限性**

受雪堆演化、冻结/融化干扰和有限的空间上下文影响，导致误检率上升，且模型受限于小样本、基于块的训练。

---

## 8. From Brittle to Robust: Improving LLM Annotations for SE Optimization

**arXiv ID:** 2603.22474 | [PDF](https://arxiv.org/pdf/2603.22474v1)

**作者:** Lohith Senthilkumar `[一作]` (North Carolina State University), Tim Menzies `[通讯]` (North Carolina State University)

**通讯引用:** 14547 | [OpenAlex ID](https://openalex.org/A5077008083)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

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

## 9. Evaluating Prompting Strategies for Chart Question Answering with Large Language Models

**arXiv ID:** 2603.22288 | [PDF](https://arxiv.org/pdf/2603.22288v1)

**作者:** Ruthuparna Naikar `[一作]` (Georgia State University), Ying Zhu `[通讯]` (Georgia State University)

**通讯引用:** 1992 | [OpenAlex ID](https://openalex.org/A5083968726)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统性评估四种提示策略对大型语言模型在基于表格的图表问答任务中的影响

**💡 创新点**

首次在纯结构化图表推理任务中隔离并量化提示方式的性能差异，揭示少量示例与链式思考的相互作用

**🔧 技术方法**

使用OpenAI GPT‑3.5、GPT‑4 与 GPT‑4o 作为黑盒推理器，采用Zero‑Shot、Few‑Shot、Zero‑Shot CoT 与 Few‑Shot CoT 四种提示模板

**📊 数据集**

基于ChartQA基准数据集，随机抽取1200个问答对（包含人工与合成问题）进行实验

**📈 对比分析**

在准确率与完全匹配度两项指标上比较四种提示方式，发现Few‑Shot CoT最高达77.0%准确率，Few‑Shot最优的格式一致性（≈65%完全匹配），Zero‑Shot仅适用于高容量模型且性能最低

**⚠️ 局限性**

仅使用结构化文本输入，未考虑视觉解析错误；较长的链式思考提示导致推理长度与成本显著增加；缺乏对多样化图表和复杂推理任务的评估

---

## 10. CN-Buzz2Portfolio: A Chinese-Market Dataset and Benchmark for LLM-Based Macro and Sector Asset Allocation from Daily Trending Financial News

**arXiv ID:** 2603.22305 | [PDF](https://arxiv.org/pdf/2603.22305v1)

**作者:** Liyuan Chen `[一作]` (Tsinghua Shenzhen International Graduate School), Xiu Li `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了可复现的滚动时域基准 CN‑Buzz2Portfolio，将每日热门新闻映射到宏观与板块ETF配置，评估 LLM 在金融决策中的推理对齐。

**💡 创新点**

创新点在于把公共关注流转化为资产配置任务，提出三阶段 CPA 工作流（压缩、感知、分配），并揭示规模与鲁棒性在不同市场 regime 的悖论。

**🔧 技术方法**

使用大语言模型（DeepSeek、Qwen、Gemini、GPT‑5 等），三阶段管道（摘要、分析、交易）、结构化命令交互、滚动模拟环境和风险指标评估。

**📊 数据集**

基于 2024–2025 年中国金融平台每日前 20 条热搜新闻、相应 ETF 资产池以及历史价格与交易记录，数据已公开开源。

**📈 对比分析**

通过累计收益、Sharpe、最大回撤和波动率对比，推理型模型在 2024 波动期显著产生 Alpha，2025 震荡期表现趋同，规模模型在趋势期优于小模型，但在震荡期易过度反应噪声。

**⚠️ 局限性**

局限性包括受市场 regime 限制导致低信号环境下易受噪声影响；低频数据无法捕捉高频套利；模拟假设流动性完美、无滑点，未涵盖衍生品与做空，且仅评估逻辑一致性，不能直接用于实盘。

---

## 11. AI Co-Scientist for Ranking: Discovering Novel Search Ranking Models alongside LLM-based AI Agents with Cloud Computing Access

**arXiv ID:** 2603.22376 | [PDF](https://arxiv.org/pdf/2603.22376v1)

**作者:** Liwei Wu `[一作]` (Trip.com Group), Cho-Jui Hsieh `[通讯]` (UCLA)

**通讯引用:** 26524 | [OpenAlex ID](https://openalex.org/A5010841999)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了AI Co‑Scientist框架，自动完成搜索排名模型研究全流程，从创意生成到代码实现、GPU训练调度及结果分析。

**💡 创新点**

创新点在于首次在排名研究中使用多LLM协同决策，完全自动化模型设计与优化，并发现新的序列特征处理技术。

**🔧 技术方法**

技术包括单一LLM负责代码实现，GPT‑5.2、Gemini Pro 3、Claude Opus 4.5等多LLM协商决策，云GPU训练调度，实验管理与结果统计。

**📊 数据集**

使用的主要数据集为商业搜索日志，包含稠密特征、稀疏序列特征和用户行为信号，实验在真实线上环境中的离线评估。

**📈 对比分析**

通过对比不同Transformer设计的AUC指标ℳ，V3.5模型在离线评估中比基线提升0.083%（相当于线上转化率提升0.1%）。

**⚠️ 局限性**

限制包括对AI安全保障依赖人类专家，缺乏实时文献检索能力，以及对复杂实验环境的泛化能力不足。

---

## 12. Communication-Efficient Approximate Gradient Coding

**arXiv ID:** 2603.22514 | [PDF](https://arxiv.org/pdf/2603.22514v1)

**作者:** Sifat Munim `[一作]` (Iowa State University), Aditya Ramamoorthy `[通讯]` (Iowa State University)

**通讯引用:** 2913 | [OpenAlex ID](https://openalex.org/A5055656185)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一类通信高效的近似梯度编码方案，用于在分布式学习中抵御慢速/失效工作节点。

**💡 创新点**

创新点在于：①首次系统化地结合随机对角矩阵与 Hadamard 乘积并加入零空间约束来构造近似梯度编码；②利用 BIBD、强正则图和余数双边图的结构矩阵，给出了解析误差上界和最优下界；③证明了在合理的随机失效模型下，所得到的近似梯度期望等于真实梯度，从而保证梯度下降收敛。

**🔧 技术方法**

核心技术包括：随机对角矩阵编码、Hadamard 乘积与零空间约束、图论与组合设计（BIBD、SRG、余数双边图）构造赋值矩阵、最小二乘误差分析、矩阵凸性与谱分析、概率论与矩阵不等式。

**📊 数据集**

实验使用 MNIST 数据集训练一个小型全连接网络（784-128-10），并在该网络上验证误差与收敛速度。

**📈 对比分析**

与传统将赋值矩阵垂直堆叠得到的基线方案相比，新方案在大多数失效率下实现了更低的近似误差和更快的训练收敛；数值实验与理论上界一致，并且实测误差低于下界所给的最坏情况。

**⚠️ 局限性**

局限性在于：①误差上界在某些情况下可能比较宽松；②方案依赖于特定的结构化赋值矩阵，对非结构化或异构工作节点的适应性有限；③构造过程需要满足零空间约束，实际实现时可能需要额外的计算；④仅在假设均匀失效概率的模型下证明收敛，未覆盖更一般的异质性或偏倚失效情况。

---

## 13. Maximum Entropy Relaxation of Multi-Way Cardinality Constraints for Synthetic Population Generation

**arXiv ID:** 2603.22558 | [PDF](https://arxiv.org/pdf/2603.22558v1)

**作者:** François Pachet `[一作]` (ImagineAllThePeople), Jean-Daniel Zucker `[通讯]` (ImagineAllThePeople)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

利用最大熵原理对多维卡方约束进行期望松弛，生成满足多路约束的合成人口；

**💡 创新点**

创新在于把多路卡方约束映射为最大熵期望约束，得到指数族分布并通过凸优化一次性全局拟合，克服传统逐步重加权方法在高维、高阶交互时的干扰；

**🔧 技术方法**

采用最大熵/指数族模型、L‑BFGS凸优化、精确枚举期望、以及与泛化调和（Generalized Raking）和CP‑SAT对比的实验框架；

**📊 数据集**

使用2024年全国公众舆论参考调查（NPORS）数据集，构造 4–40 维的基准实例；

**📈 对比分析**

与 Generalized Raking 进行对比，采用平均相对误差和运行时间评估；实验显示在变量数 ≥28 或三元交互约束下 MaxEnt 明显优于 Raking，Raking 在低维/低阶场景仍保持竞争力；

**⚠️ 局限性**

仅在硬约束可满足的情况下适用，对极大属性空间需采样估计；缺乏完整整数修复方案；噪声/不一致目标的软约束仅提出但未系统评估。

---

## 14. SCALE-Sim TPU: Validating and Extending SCALE-Sim for TPUs

**arXiv ID:** 2603.22535 | [PDF](https://arxiv.org/pdf/2603.22535v1)

**作者:** Jingtian Dang `[一作]` (Georgia Institute of Technology), Tushar Krishna `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 14113 | [OpenAlex ID](https://openalex.org/A5034089074)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 Google TPU v4 真实硬件上验证并扩展了 SCALE‑Sim v3，构建了 SCALE‑Sim TPU，支持对 TPU 样式加速器的周期级仿真，并集成了 StableHLO 前端实现框架无关的全模型推理仿真。

**💡 创新点**

创新点包括：①通过实测验证实现 GEMM 周期到时延的线性映射；②为非门阵（如元素级加法、ReLU）引入轻量级学习延迟模型，误差低于 3%；③集成 StableHLO 作为统一 IR 前端，消除手工映射，提高易用性和覆盖度。

**🔧 技术方法**

使用技术包括：周期级仿真、线性回归周期‑时延映射、Histogram‑based Gradient Boosting Regressor（HGBR）学习模型、StableHLO IR 解析器、MLIR 编译链、SCALE‑Sim v3 的模块化设计。

**📊 数据集**

实验数据集主要来自 TPU v4 的实测：GEMM 大小从 32×32 到 4096×4096 的三级尺寸扫；bf16 元素级操作在 1D（32–8192）和 2D（64–1024）尺寸范围内的多维采样；使用这些测量数据训练和验证学习模型。

**📈 对比分析**

对 GEMM 的线性回归评估得到 R² 0.79–0.97，整体 MAPE 约 32%；对元素级操作的学习模型得到 R² 0.997–0.998，median 绝对误差 1–2 μs，median 相对误差 <3%。SCALE‑Sim TPU 能直接输出 TPU 时延估计，覆盖更广泛的操作集。

**⚠️ 局限性**

局限性包括：对中等规模 GEMM 的预测误差较大，未完整建模数据搬运与调度；学习模型仅覆盖元素级操作，缺乏对更复杂非门阵的建模；模型需要针对每一代 TPU 重新校准，泛化到其他硬件的适用性有限。

---

## 15. Q-AGNN: Quantum-Enhanced Attentive Graph Neural Network for Intrusion Detection

**arXiv ID:** 2603.22365 | [PDF](https://arxiv.org/pdf/2603.22365v1)

**作者:** Devashish Chaudhary `[一作]` (Deakin University), Shiva Raj Pokhrel `[通讯]` (Deakin University)

**通讯引用:** 2782 | [OpenAlex ID](https://openalex.org/A5038446422)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了一种混合量子-经典图神经网络（Q-AGNN），用于网络流量的入侵检测，结合多跳邻域信息的量子特征编码和注意力机制进行信息聚合。

**💡 创新点**

创新点包括：① 采用参数化量子电路（EfficientSU2 ansatz）对节点特征进行高维量子映射，捕获多跳关联；② 在此基础上引入注意力机制对量子增强嵌入进行自适应加权，形成二阶多项式图滤波；③ 在真实 IBM 量子硬件上完成端到端训练与推理，验证量子优势与噪声鲁棒性。

**🔧 技术方法**

核心技术：参数化量子电路（PQCs）+角度编码；EfficientSU2 ansatz + 线性 CNOT 结构；注意力图神经网络（GAT式）聚合；经典 MLP 分类；余弦相似度图构造；Adam + BCE loss；无噪声、硬件噪声模拟、真实量子硬件实验。

**📊 数据集**

使用四个数据集：BoT‑IoT、UNSW‑NB15、NetFlow 版本 NF‑BoT‑IoT、NF‑UNSW‑NB15。

**📈 对比分析**

与 GCN、GAT、GraphSAGE、ClusterGCN、GINConv、SuperGAT、TransformerConv 等经典 GNN 在同一实验设置下对比；Q-AGNN 在大多数数据集上获得最高或相当的宏 F1、准确率，并保持极低误报率；在 NF‑UNSW‑NB15 上实现最佳宏 F1；在噪声环境下性能略降，但仍优于或与经典方法相当。

**⚠️ 局限性**

局限性：① 受限于 NISQ 设备的 qubit 数量和噪声，实验规模受限；② PQC 深度受限，表达能力受约束；③ 仅在四个小规模数据集上验证，未在更大真实网络上评估可扩展性；④ 对不同数据集的性能差异较大，需进一步研究鲁棒性与泛化；⑤ 需要更深入的量子噪声抑制和可解释性分析。

---

## 16. GraphRAG for Engineering Diagrams: ChatP&ID Enables LLM Interaction with P&IDs

**arXiv ID:** 2603.22528 | [PDF](https://arxiv.org/pdf/2603.22528v1)

**作者:** Achmad Anggawirya Alimin `[一作]` (Delft University of Technology), Artur M. Schweidtmann `[通讯]` (Delft University of Technology)

**通讯引用:** 3222 | [OpenAlex ID](https://openalex.org/A5085291703)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了ChatP&ID框架，利用知识图和GraphRAG技术实现与智能P&ID（DEXPI）的自然语言交互。

**💡 创新点**

创新点在于将DEXPI P&ID转换为结构化知识图，并设计ContextRAG、VectorRAG、PathRAG、CypherRAG四种检索方式，结合LLM代理式工作流，实现高精度、低成本的问答。

**🔧 技术方法**

技术栈包括OpenAI/Anthropic大型语言模型、LangGraph代理框架、Neo4j图数据库、pyDEXPI转换器、Voyage-3.5-lite向量嵌入、以及多种GraphRAG工具。

**📊 数据集**

使用单页DEXPIEX01.xml（含完整P&ID）作为原始数据，生成多层抽象知识图（完整、流程、概念），并与原始图像、Proteus文件进行对比。

**📈 对比分析**

评估基于LLM-as-Judge和语义相似度的19个问答集，比较GraphRAG工具与基线，发现ContextRAG在准确率（≈0.91）和成本（≈$0.004）上领先，VectorRAG/PathRAG约0.82/0.83，CypherRAG约0.86，基线图像约0.83，Proteus约0.88；同时记录了Token成本和平均执行时间。

**⚠️ 局限性**

局限性包括：仅测试单页P&ID，图形尺寸有限；小模型对复杂图仍易失效；多页图会导致成本线性增长；CypherRAG受查询生成误差和模型能力限制；多模态输入在精度上不如结构化知识图；离线模型计算时间较长。

---

## 17. Between the Layers Lies the Truth: Uncertainty Estimation in LLMs Using Intra-Layer Local Information Scores

**arXiv ID:** 2603.22299 | [PDF](https://arxiv.org/pdf/2603.22299v1)

**作者:** Zvi N. Badash `[一作]` (Technion --- Israel Institute of Technology), Moti Freiman `[通讯]` (Technion --- Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于跨层 KL 散度签名的轻量级不确定性估计方法，能够在单次前向传播下判断大型语言模型的预测是否正确。

**💡 创新点**

创新点在于将每层后 MLP 激活映射为概率分布，构造 L×L 有向 KL 矩阵作为结构化签名，并用轻量级梯度提升树对正确性进行预测，从而兼顾可解释性与跨任务迁移性。

**🔧 技术方法**

使用温度软化、KL 散度、层间签名矩阵、对比变换、LightGBM 轻量级分类器等技术，并以 AUPRC 与 Brier 分数评估性能。

**📊 数据集**

在 Llama‑3.1‑8B、Qwen3‑14B‑Instruct 与 Mistral‑7B‑Instruct‑v0.3 上，针对 TriviaQA、HotpotQA、Movies、WinoGrande、WinoBias、IMDB、Math 与 MMLU 等多种问答、分类与数值推理基准进行实验。

**📈 对比分析**

与线性探测器（Probe）和 LOS‑NET 等基线对比，分布内表现与 Probe 相当，跨任务迁移和 4‑bit 量化后均显著优于 Probe，AUPRC 提升高达 +2.86pp、Brier 分数提升至 +21.02pp。

**⚠️ 局限性**

局限在于分布内可能略逊于全隐藏状态探测，对监督训练和任务相关 token 选择敏感，压缩签名会丢失细粒度信息，且无法区分模型本体不确定与观测噪声。

---

## 18. Short-Form Video Viewing Behavior Analysis and Multi-Step Viewing Time Prediction

**arXiv ID:** 2603.22663 | [PDF](https://arxiv.org/pdf/2603.22663v1)

**作者:** Vu Thi Hai Yen `[一作]` (Hanoi University of Science and Technology), Truong Thu Huong `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5091247445)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开发布了 100 条短视频的用户观看时长数据集，并对多种时间序列预测模型进行评估

**💡 创新点**

创新点在于提供大规模、真实用户的观看时长数据集，并系统比较不同模型在该场景下的预测表现

**🔧 技术方法**

采用了线性回归、AR、Auto‑ARIMA、SVR 和决策树回归等传统时间序列与机器学习技术

**📊 数据集**

使用了来自 YouTube Shorts 的 100 条短视频（共 5,000 条观看时长记录，50 名用户）

**📈 对比分析**

通过滚动窗口递归预测和 RMSE 评估，Auto‑ARIMA 在大多数实验配置下表现最优；AR 在样本不足时不稳定，SVR 与 DTR 的误差更大

**⚠️ 局限性**

局限性包括数据规模仍有限、仅覆盖短视频平台、未考虑个体差异和网络条件，并且仅评估了传统模型

---

## 19. Large Language Models for Missing Data Imputation: Understanding Behavior, Hallucination Effects, and Control Mechanisms

**arXiv ID:** 2603.22332 | [PDF](https://arxiv.org/pdf/2603.22332v1)

**作者:** Arthur Dantas Mangussi `[一作]` (Aeronautics Institute of Technology), Pedro Henriques Abreu `[通讯]` (University of Coimbra)

**通讯引用:** 2990 | [OpenAlex ID](https://openalex.org/A5065859612)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过零样本提示工程，对五大类大型语言模型（Gemini 3.0 Flash、Claude 4.5 Sonnet、GPT‑4.1‑Nano、MiMo‑V2‑Flash、Mistral Devstral 2 2512）与六种传统缺失值插补基线（kNN、MICE、missForest、SAEI、SoftImpute、TabPFN）在29个数据集（9个合成、20个公开真实）上，分别在 MCAR、MAR、MNAR 三种缺失机制、5%、10%、20% 缺失率下进行系统性对比实验，评估插补精度（NRMSE）和计算成本。

**💡 创新点**

创新点在于：①在大规模、统一的实验框架下，首次全面对比多种 LLM 与传统方法；②使用零样本提示工程实现无须微调的通用插补；③揭示 LLM 插补效果高度依赖预训练知识，对合成数据表现欠佳；④系统量化 LLM 在时间、成本和“幻想”风险方面的权衡。

**🔧 技术方法**

技术手段包括：Prompt Engineering（包含限制、输出格式、批处理策略）、零样本推理、三重缺失机制模拟、NRMSE 评估、三因素 ANOVA 与 Tukey HSD、Pareto 性能-成本分析、回退机制与错误处理。

**📊 数据集**

使用的数据集共29个：9个由 scikit‑learn 生成的合成数据集（包括分类、连续、混合等），以及 20 个公开开源数据集（如 Pima‑Diabetes、Cleveland Heart、Iris、Wine、BC‑Coimbra 等）。

**📈 对比分析**

对比结果显示：在真实数据上，Gemini 3.0 Flash 与 Claude 4.5 Sonnet 在大多数缺失机制下显著优于传统基线（p < 0.01），但在合成数据上传统方法（MICE、missForest）表现更好；LLM 的插补精度随缺失率上升而下降幅度较小；同时 LLM 的推理时间与成本远高于传统方法，且小模型易出现幻想。

**⚠️ 局限性**

局限性包括：①高昂的算力与费用导致批量插补不易扩展；②实验仅涵盖五种 LLM，缺乏更广泛的模型覆盖；③缺少对私有/行业数据的验证，无法评估 LLM 在无预训练知识场景下的表现；④批处理大小和提示长度的选取对性能影响较大，需进一步优化。

---

## 20. A vision-language model and platform for temporally mapping surgery from video

**arXiv ID:** 2603.22583 | [PDF](https://arxiv.org/pdf/2603.22583v1)

**作者:** Dani Kiyasseh `[一作]` `[通讯]` (Halsted AI), Dani Kiyasseh (Halsted AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文设计并训练了Halsted这一多任务、跨专业的视觉‑语言模型，利用海量手术视频（Halsted Surgical Atlas）自动生成手术步骤、动作、器械、解剖、技术熟练度等多种标签，并搭建了可在线使用的 Web 平台供外科医生即时获取手术映射结果。

**💡 创新点**

创新点包括①以指令为控制的可生成式模型，支持同一模型完成多任务（步骤、动作、熟练度、上下文等）并跨八大专业；②利用迭代自标注策略将手术视频规模扩大四倍；③将模型与平台无缝对接，实现实时手术映射与可视化。

**🔧 技术方法**

技术手段主要是基于预训练的视频编码器 VideoMAE 提取特征，使用双向 Transformer 解码器（自回归生成）配合词级 tokenizer 生成手术组件标签；模型通过指令嵌入实现任务控制；采用多任务联合训练、Monte‑Carlo 交叉验证、并行多阶段推理流程。

**📊 数据集**

数据集为 Halsted Surgical Atlas，包含 650K+ 片段，覆盖 16 个手术、8 个专业，11 种组件、104 类标签；公开子集 HSA‑27k（约 27K 片段）用于基准；外部验证使用 RARP‑50（10 片）进行动作识别。

**📈 对比分析**

与现有 SAIS 模型在同一任务（动作识别、熟练度评估）对比，Halsted 在动作识别 AUROC 1.00 对 0.68，熟练度 0.78 对 0.73；在宏观任务上步骤识别 99%/91%；在 RARP‑50 上准确率 68.6%，相对随机 5.5× 提升。自标注增强后微动作识别精度提升约 10%。

**⚠️ 局限性**

局限性包括：①未覆盖神经外科、骨科、整形等专业；②依赖公开视频，部分复杂手术难以获取；③注释质量虽经过 QA，但仍可能存在错误，导致模型偶尔生成误标签；④平台功能已实现但临床验证仍待进一步验证。

---

## 21. Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs

**arXiv ID:** 2603.22446 | [PDF](https://arxiv.org/pdf/2603.22446v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 22. Adversarial Vulnerabilities in Neural Operator Digital Twins: Gradient-Free Attacks on Nuclear Thermal-Hydraulic Surrogates

**arXiv ID:** 2603.22525 | [PDF](https://arxiv.org/pdf/2603.22525v1)

**作者:** Samrendra Roy `[一作]` (University of Illinois Urbana-Champaign), Syed Bahauddin Alam `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1160 | [OpenAlex ID](https://openalex.org/A5063457131)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了神经算子模型在安全关键数字孪生中的鲁棒性，揭示其对极稀疏、物理可行的对抗扰动高度脆弱，并通过差分进化在四种算子架构上实现了高成功率的攻击。

**💡 创新点**

创新点包括：①提出有效扰动维度 d_eff 作为衡量输入对输出敏感度聚集度的理论指标；②将差分进化改造为针对物理约束的稀疏对抗攻击框架；③构建两因子（敏感度集中度 + 规模）模型解释不同算子架构的脆弱性差异。

**🔧 技术方法**

技术主要包括：神经算子架构（MIMONet、NOMAD、S‑DeepONet、POD‑DeepONet）、梯度无关差分进化搜索、Jacobian 计算与 d_eff 评估、物理可行性约束与 z‑score 异常检测对比。

**📊 数据集**

数据集为 1546 条热交换器高保真 CFD 仿真数据，包含 102 维边界条件和 15,908 维场输出（压力、三维速度），分 80%/20% 训练/测试。

**📈 对比分析**

与传统梯度攻击（PGD、FGSM）相比，差分进化在所有四种算子上均表现更高成功率（最高 100%），且攻击时间约 2 分钟/次。实验表明 S‑DeepONet 在 L0=3 时成功率 94% 且相对 L2 错误可达 62%，而 POD‑DeepONet 受低秩限制成功率仅 25%。

**⚠️ 局限性**

局限性在于仅评估单一热交换器基准，未验证不同 PDE、网格尺寸或更高维输入的泛化；对抗扰动仅限于 1σ 物理可行范围；实验基于黑盒查询，未考虑攻击成本或多模型协同攻击的真实威胁。

---

## 23. Spatially-Aware Evaluation Framework for Aerial LiDAR Point Cloud Semantic Segmentation: Distance-Based Metrics on Challenging Regions

**arXiv ID:** 2603.22420 | [PDF](https://arxiv.org/pdf/2603.22420v1)

**作者:** Alex Salvatierra `[一作]` (Public University of Navarre), Mikel Galar `[通讯]` (Public University of Navarre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于空间误差的评估框架，用于航空LiDAR点云语义分割模型的性能评估。

**💡 创新点**

创新点在于引入距离加权指标（MDE、ρ、μ）和“困难点”子集来衡量误差的空间严重性和区分模型在边界、遮挡等复杂区域的差异。

**🔧 技术方法**

主要技术包括基于欧氏距离的误差裁剪、宏观平均距离误差计算以及将不同模型共享的错误点集合作为评估焦点。

**📊 数据集**

使用了三大公开航空LiDAR数据集DALES、FRACTAL以及自有Tracasa-PNA20进行实验。

**📈 对比分析**

在三种主流网络（KPConv、RandLA-Net、Point Transformer V3）上对比，发现传统mIoU/OA指标在全体测试集上相近，但在困难点子集以及距离指标上显现出模型性能的明显差异，展示了空间一致性更高的模型优先选择。

**⚠️ 局限性**

局限在于距离阈值τ_c需手工设定，未考虑点云稀疏度变化对误差尺度的影响，且仅评估点级空间误差，缺乏对对象级别一致性的衡量。

---

## 24. Graph Signal Processing Meets Mamba2: Adaptive Filter Bank via Delta Modulation

**arXiv ID:** 2603.22333 | [PDF](https://arxiv.org/pdf/2603.22333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 25. Learning Sidewalk Autopilot from Multi-Scale Imitation with Corrective Behavior Expansion

**arXiv ID:** 2603.22527 | [PDF](https://arxiv.org/pdf/2603.22527v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 26. Rashid: A Cipher-Based Framework for Exploring In-Context Language Learning

**arXiv ID:** 2603.22497 | [PDF](https://arxiv.org/pdf/2603.22497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 27. Sketch2CT: Multimodal Diffusion for Structure-Aware 3D Medical Volume Generation

**arXiv ID:** 2603.22509 | [PDF](https://arxiv.org/pdf/2603.22509v1)

**作者:** Delin An `[一作]` (University of Notre Dame), Chaoli Wang `[通讯]` (University of Notre Dame)

**通讯引用:** 3071 | [OpenAlex ID](https://openalex.org/A5101913449)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了Sketch2CT框架，通过用户绘制的二维草图和文本描述来生成结构一致的3D医学体积；

**💡 创新点**

创新点在于：①将草图与文本两种模态进行局部与全局双层融合（TSFE与CGFM），②采用潜在扩散模型先生成3D分割掩模，再用该掩模引导体积生成，从而兼顾结构精度与视觉质量；

**🔧 技术方法**

核心技术包括：潜在扩散概率模型、Capsule‑Attention、FiLM调制、跨模态注意力、AutoencoderKL压缩空间；

**📊 数据集**

在CHAOS肝脏、AVT主动脉、Decathlon肝脏、Decathlon心脏四个公开CT/MRI数据集上进行实验；

**📈 对比分析**

与Med-DDPM、MedGen3D、Seg‑Diff等基线对比，Sketch2CT在FID/LPIPS、分割Dice、下游分割任务上均取得最佳或最接近真实数据的性能；

**⚠️ 局限性**

局限在于目前仅支持单器官生成，缺乏多器官协同和病理变异模拟，且对手工草图的绘制仍有一定操作门槛。

---

## 28. Model Predictive Control with Differentiable World Models for Offline Reinforcement Learning

**arXiv ID:** 2603.22430 | [PDF](https://arxiv.org/pdf/2603.22430v1)

**作者:** Rohan Deb `[一作]`, Arindam Banerjee `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在离线强化学习中，提出了推理时基于模型预测控制（MPC）的自适应框架，利用预训练的策略与可微分扩散世界模型进行实时梯度更新，提升策略性能。

**💡 创新点**

创新点在于将可微分扩散世界模型与MPC相结合，实现了推理时对策略参数的梯度优化，从而在不增加额外数据收集的前提下提升离线RL性能。

**🔧 技术方法**

使用了可微分扩散模型（Diffusion World Model）、奖励预测器、终端价值函数、MPC求解和梯度反向传播技术。

**📊 数据集**

实验数据集为D4RL benchmark中的MuJoCo连续控制任务（HalfCheetah、Hopper、Walker2d）以及AntMaze迷宫任务。

**📈 对比分析**

与TD3+BC、IQL、CQL、SAC‑RND、ReBRAC以及生成式模型基线（DT、TT、MOPO等）进行对比，在MuJoCo任务平均提升至85.33分，在AntMaze任务平均提升至85.07分，显著优于大多数基线。

**⚠️ 局限性**

主要限制包括推理时计算开销较大、对可微分世界模型的准确性高度依赖，以及在某些任务（如AntMaze‑UMaze）上提升不明显。

---

## 29. Vision-based Deep Learning Analysis of Unordered Biomedical Tabular Datasets via Optimal Spatial Cartography

**arXiv ID:** 2603.22675 | [PDF](https://arxiv.org/pdf/2603.22675v1)

**作者:** Sakib Mostafa `[一作]` (Stanford University), Md Tauhidul Islam `[通讯]` (Stanford University)

**通讯引用:** 1016 | [OpenAlex ID](https://openalex.org/A5061741626)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了 Dynomap 框架，将无序表格数据通过可微分渲染映射到二维空间，并利用卷积网络完成分类与回归任务。

**💡 创新点**

创新点在于端到端学习任务适应的空间布局：特征门控 + 可学习二维坐标 + 可微分渲染，让模型自动形成可解释的特征邻域并显著提升表格数据的深度学习性能。

**🔧 技术方法**

技术组合包括可微分渲染、特征门控、二维坐标学习、卷积网络、集成梯度 (Integrated Gradients)、Moran’s I 与 kNN 统计分析；使用 TensorFlow+Adam 进行端到端优化。

**📊 数据集**

六大公共数据集：cfRNA RARE‑Seq（全基因与签名两版）、GSE68086 纯血小板 RNA‑seq、TCGA‑BRCA 乳腺癌 RNA‑seq、Parkinson 语音特征、Tabula Muris 单细胞 RNA‑seq、以及多基准深度表格模型的基线数据。

**📈 对比分析**

与 Logistic Regression、XGBoost、Random Forest、TabM、ModernNCA、TabPFN 等传统与现代深度表格模型对比；Dynomap 在所有任务（二分类、亚型、分期、单细胞、多模态）中均实现了最高或最优的宏 F1、准确率和召回率，尤其在低信噪比和高维场景下优势显著。

**⚠️ 局限性**

局限性包括：需要调优二维布局的超参数；对大规模数据的计算开销相对较高；对完全无结构特征的可解释性仍需进一步验证；在极端类别不平衡或样本极少的情形下可能出现性能下降。

---

## 30. A Direct Classification Approach for Reliable Wind Ramp Event Forecasting under Severe Class Imbalance

**arXiv ID:** 2603.22326 | [PDF](https://arxiv.org/pdf/2603.22326v1)

**作者:** Alejandro Morales-Hernández `[一作]` (Universite Libre de Bruxelles), Gianluca Bontempi `[通讯]` (Universite Libre de Bruxelles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建基于多变量时间序列分类的风电功率坡度事件预测框架，结合数据预处理提取统计特征并掩蔽未确定事件；

**💡 创新点**

提出实时可行的缺失事件掩蔽策略，将风功率观测与坡度事件标签同步，并将EasyEnsemble与传统SDA方法结合，实现直接预测多类别坡度事件；

**🔧 技术方法**

使用多变量时间序列特征提取、EasyEnsemble（AdaBoost+欠采样）、随机森林、平衡随机森林、RUSBoost、平衡装袋以及LSTM等技术进行模型训练与对比；

**📊 数据集**

采用来自意大利Sannio大学的28,800条15分钟间隔风电功率时序数据，标注5类坡度事件；

**📈 对比分析**

通过80/20拆分和5折交叉验证对比不同算法，评价指标包括准确率、平衡准确率、Kappa和加权F1；在3类分类任务中，EasyEnsemble实现0.907的准确率、0.917的加权F1，显著优于其他模型；

**⚠️ 局限性**

仅使用风功率数据，未引入气象或NWP特征；模型为离线训练，缺乏在线学习和实时适应；对深度学习架构在不平衡数据上的效果仍需进一步探索；

---

## 31. DAQ: Delta-Aware Quantization for Post-Training LLM Weight Compression

**arXiv ID:** 2603.22324 | [PDF](https://arxiv.org/pdf/2603.22324v1)

**作者:** Xiaoming Yu `[一作]` (Yuanbao and Hunyuan AI Infra Team), Feng Li `[通讯]` (Yuanbao and Hunyuan AI Infra Team)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Delta-Aware Quantization（DAQ），一种数据无关的后训练模型量化框架，专门保留从基模型到后训练模型的权重增量ΔW；

**💡 创新点**

创新点在于用两种 delta‑aware 指标（Sign Preservation Rate 与 Cosine Similarity）直接衡量量化后 ΔW 的方向保真度，取代传统的重构误差目标，从而显式保护后训练增量；

**🔧 技术方法**

采用 scale‑parameterized FP8 (E4M3) 量化实现，使用粗细搜索（coarse‑to‑fine）寻找最优缩放因子；同时设计并评估 SignRate 与 CosSim 两种指标；

**📊 数据集**

实验基于 DeepSeek‑V3 预训练模型，并在一个 toy 的对话风格数据集上做了 Supervised Fine‑Tuning（SFT）得到后训练权重；

**📈 对比分析**

与 AbsMax、SmoothQuant、AWQ 等传统 PTQ 方法以及 MSE‑based scale search 对比，DAQ 在 Style（SFT‑specific）指标上从 1.081（AbsMax block）提升至约 1.7+，保持 General（通用能力）指标与 BF16 后训练模型相近；MSE‑guided 搜索反而进一步降低了 Style；

**⚠️ 局限性**

局限性包括：假设 ΔW 相对较小，若后训练更新幅度大则指标效用下降；仅在 FP8 (E4M3) 上验证；实验只涵盖单一模型与有限评测指标；未结合更复杂的量化技术（如混合精度、学习的四舍五入等）。

---

## 32. Bridging the Gap Between Climate Science and Machine Learning in Climate Model Emulation

**arXiv ID:** 2603.22320 | [PDF](https://arxiv.org/pdf/2603.22320v1)

**作者:** Luca Schmidt `[一作]` (University of Tübingen), Nina Effenberger `[通讯]` (ETH Zurich)

**通讯引用:** 67 | [OpenAlex ID](https://openalex.org/A5018288244)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个将机器学习与气候科学紧密结合的框架，旨在克服两者在方法论和应用上的断层。

**💡 创新点**

创新点在于将“适用性（adequacy）、可信度（trustworthiness）和可访问性（accessibility）”三大原则作为指导方针，形成可操作的检查清单，帮助从事机器学习的研究者和气候科学家共同推进气候模型仿真器的开发与应用。

**🔧 技术方法**

采用概念性分析、案例阐述和交叉学科方法论，未提出具体算法，而是侧重于研究流程、评估标准和协作模式。

**📊 数据集**

未使用特定数据集，而是引用典型气候模型输出、重分析产品与观测资料作为讨论对象。

**📈 对比分析**

比较方法以框架可操作性和案例讨论为主，未给出量化性能指标；强调应根据具体应用场景选择评估指标，确保与实际需求匹配。

**⚠️ 局限性**

局限在于缺乏对框架的实证验证、对不同气候问题的细粒度适配仍需进一步探索，以及在实际跨学科合作中的实施细节未被详细说明。

---

## 33. Product Range Search Problem

**arXiv ID:** 2603.22500 | [PDF](https://arxiv.org/pdf/2603.22500v1)

**作者:** Oliver Chubet `[一作]`, Don Sheehy `[通讯]`

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出两种基于贪心树的近似产品度量范围搜索数据结构，支持在多度量双曲空间中高效查询

**💡 创新点**

首次将贪心树与多度量空间相结合，构造递归的贪心范围树以及直接在产品度量上构造的贪心树，并给出其理论复杂度；实现了对非欧几里得、非范数度量的近似范围搜索

**🔧 技术方法**

利用贪心树（greedy tree）与净树变体、近似产品度量构造和查询算法；通过层级递归合并实现多度量的联合查询

**📊 数据集**

本文未给出具体实验数据集，仅在理论上分析复杂度；若做实验，可选用具有多度量的公共数据集（如Ulam、Frechet、DTW等）

**📈 对比分析**

理论上与传统球树、k-d树、范围树等基线方法对比，构造时间为O(2^O(δ)n log^{m-1}n)，空间为O(2^O(δ)n log^{m-1}n)，查询时间为O((2+1/ε)^O(δ) logΔ + k)，在高维非欧几里得度量下保持了可扩展性

**⚠️ 局限性**

局限性在于目前不支持动态增删点（需要重建结构），近似参数与维度指数相关，极高维度下性能可能下降；实际实验验证仍待完成

---

## 34. Trained Persistent Memory for Frozen Decoder-Only LLMs

**arXiv ID:** 2603.22329 | [PDF](https://arxiv.org/pdf/2603.22329v1)

**作者:** Hong Jeong `[一作]` (Inha University), Hong Jeong `[通讯]` (Inha University)

**通讯引用:** 6342 | [OpenAlex ID](https://openalex.org/A5064069488)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究在解码器单向（decoder‑only）语言模型中实现持久化潜在空间记忆，将冻结的 GPT‑2 通过少量训练的适配器扩展为可跨会话存储并检索信息的系统。

**💡 创新点**

创新点在于证明并实现“通用写入规则+架构特定读取路径”的设计，将先前仅适用于编码‑解码器模型的持久化记忆迁移到解码器单向模型，并揭示在容量受限时存在的“归纳偏差二分歧”，即只有具备强建模先验的三种写入读取组合（并行交叉注意、Hebbian 关联回忆、槽式稀疏写入）能在 1× 容量下实现显著记忆保持。

**🔧 技术方法**

技术上使用 GPT‑2 124M 的冻结 Transformer 作为基座，结合六种记忆适配器（prefix、KV 扩展、交叉注意、Hebbian、加权分支、槽式写入），采用统一的写入更新规则 P_t = γP_{t‑1} + A_t^⊤V_w，并在三种不同的读取路径（KV 前缀、并行交叉注意、门控分支）中实现持久化记忆注入；训练采用两阶段方案：离线监督学习（训练适配器）与在线对话学习（无梯度更新 P）。

**📊 数据集**

使用公开的 LoCoMo 长期对话记忆数据集，对每个会话进行问答与事实检索，评估模型在 30 个会话周期内的记忆保持和知识累积。

**📈 对比分析**

与基线无记忆 GPT‑2 及六种方法在 1× 与 10× 容量下比较，发现 M.2、M.4、M.6 在 1× 下分别实现 9.0%–17.8% 的保留记忆分数和 7.3%–9.7% 的 ΔK；其余三种方法在 1× 下几乎无提升，但在 10× 容量时都能达到 7.6%–10.3% 的保留分数，表明缺陷主要是容量瓶颈而非架构不兼容。

**⚠️ 局限性**

局限性包括仅在 LoCoMo 单一数据集与单一 GPT‑2 模型上验证，未探讨更大规模模型、Encoder‑only 或多模态架构；写入投影保持固定随机映射，未能进一步利用梯度；且对话学习阶段不涉及用户隐私与可解释性机制。

---

## 35. CaP-X: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

**arXiv ID:** 2603.22435 | [PDF](https://arxiv.org/pdf/2603.22435v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 36. Causal Direct Preference Optimization for Distributionally Robust Generative Recommendation

**arXiv ID:** 2603.22335 | [PDF](https://arxiv.org/pdf/2603.22335v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 37. Wake Up to the Past: Using Memory to Model Fluid Wake Effects on Robots

**arXiv ID:** 2603.22472 | [PDF](https://arxiv.org/pdf/2603.22472v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 38. Interactive and Urgent HPC: State of the Research

**arXiv ID:** 2603.22542 | [PDF](https://arxiv.org/pdf/2603.22542v1)

**作者:** Albert Reuther `[一作]` (MIT Lincoln Laboratory), Rollin Thomas `[通讯]` (Lawrence Berkeley Lab)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对交互式与紧急高性能计算的现状进行综述，整理了政策、调度、工具、数据管理、基准、培训与案例研究等方面的经验与挑战。

**💡 创新点**

提出统一批处理与交互/紧急工作负载的基础设施理念，强调流式数据传输、实时调度与多模态交互的必要性，并为未来研究提出了评估指标与技术路线。

**🔧 技术方法**

综述了Slurm、Kubernetes、Open OnDemand、Jupyter、SAIA、SKIP、StreamFlow等关键技术与框架。

**📊 数据集**

本章为综述性质，未使用具体实验数据集；主要基于行业会议（SC、ISC）和中心内部案例的经验与日志数据。

**📈 对比分析**

通过对不同中心调度延迟、共享节点等待时间等指标的对比（如GWDG 3.5M 任务数据），展示了交互式队列与共享节点在启动延迟上的优势；但缺乏统一基准与系统级性能对比。

**⚠️ 局限性**

局限性在于缺乏统一的量化评测和大规模实测数据；讨论范围广但细节不足，未来需更多跨中心实验与标准化基准。

---

## 39. Toward Faithful Segmentation Attribution via Benchmarking and Dual-Evidence Fusion

**arXiv ID:** 2603.22624 | [PDF](https://arxiv.org/pdf/2603.22624v1)

**作者:** Abu Noman Md Sakib `[一作]` (University of Texas at San Antonio), Zijie Zhang `[通讯]` (University of Texas at San Antonio)

**通讯引用:** 482 | [OpenAlex ID](https://openalex.org/A5100673890)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可复现的语义分割可解释性评估基准，并基于该基准设计了一种融合梯度与干预信号的轻量级双证据校正方法（DEA）。

**💡 创新点**

创新点在于：①把目标区域删减（deletion）与非目标泄漏（off‑target leakage）两类因果可解释性指标正式引入评估流程；②通过将梯度级别的高分辨率证据与区域级干预（region‑level intervention）相结合，形成权重加权融合，既提升了删减可解释性，又保持了鲁棒性。

**🔧 技术方法**

采用梯度归一化、元素级梯度-激活乘积、固定网格干预、均值掩码操作、Pearson 相关等技术；在实现层面使用 PyTorch 与 TorchVision 进行分割模型推理，并在代码仓库中提供完整实验脚本。

**📊 数据集**

在 Pascal VOC 2012 与 SBD 数据集上评估，使用预训练的 DeepLabV3‑ResNet50、FCN‑ResNet50 与 LRASPP‑MobileNetV3 三种主流分割骨干。

**📈 对比分析**

与传统的梯度池化（GPA）和元素梯度（EGA）方法相比，DEA 在目标删除可解释性（TDD）上持续领先，且在绝对非目标泄漏（ODD）方面更低；在鲁棒性（对噪声、亮度、对比度、模糊、翻转的相关性）上与最稳健的基线相近；但由于需要额外的干预计算，运行时间显著增加。

**⚠️ 局限性**

局限性包括：①评估仅基于实验结果，缺乏正式的显著性检验；②干预采用固定网格，可能忽略细长结构导致残余泄漏；③在速度与可解释性之间存在不可避免的权衡，DEA 在低时延场景下不具备优势。

---

## 40. AI Mental Models: Learned Intuition and Deliberation in a Bounded Neural Architecture

**arXiv ID:** 2603.22561 | [PDF](https://arxiv.org/pdf/2603.22561v1)

**作者:** Laurence Anthony `[一作]` (Waseda University), Laurence Anthony `[通讯]` (Waseda University)

**通讯引用:** 2977 | [OpenAlex ID](https://openalex.org/A5025314843)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在经典的64条三段论推理基准上，作者提出并验证了一种受限的双路径神经网络（直觉-推理分支），旨在捕捉人类在推理时的快速直觉与慢速推理的分工。

**💡 创新点**

创新点在于：①通过对直觉路径进行显式限制，强制让模型在第二阶段形成“分层推理”；②在推理路径中引入可学习的门控机制，令模型自动选择不同的推理状态；③通过内部激活热图、门控统计和推理状态消融等可解释性分析，证明模型在有限容量下产生了稀疏且功能差异明显的内部子结构。

**🔧 技术方法**

技术上采用：多层感知机作为基线；受限双路径网络（直觉层4维，推理层包含1维主体+5维候选状态+可学习门控）；交叉熵/KL散度损失、Adam优化、5折交叉验证；对推理状态进行热图、门控权重统计、消融实验以及多种随机种子稳定性扫描。

**📊 数据集**

数据集为 Khemlani & Johnson‑Laird (2012) 的64条经典三段论，配有每条推理的9种人类反应概率分布（“All are”“Some are”等九类）。

**📈 对比分析**

比较方法：在同一5折交叉验证框架下，比较直接MLP、受限直觉分支和受限推理分支。结果显示：推理分支在整体相关系数上提升至0.815（vs 0.727的直觉，vs 0.711的直接MLP），并在对拒绝与特定结论类型的预测上实现显著改善；RMSE与MAE也同步下降，表明模型在留出数据上的泛化性能显著提升。

**⚠️ 局限性**

局限性包括：①对人类整体反应分布训练而非个体过程数据，难以检验策略级别差异；②仅限于经典三段论，未验证对更大规模或开放式推理任务的迁移；③可解释性虽揭示稀疏结构，但未能给出完整的符号化或逻辑规则映射；④与直接MLP相比的统计显著性不稳，仅在直觉对比中显著；⑤尚未与现有的计算模型（如 Reasoning About Properties）逐条对比，难以直接验证是否真正实现了多模型搜索与修正的过程。

---

## 41. flexvec: SQL Vector Retrieval with Programmatic Embedding Modulation

**arXiv ID:** 2603.22587 | [PDF](https://arxiv.org/pdf/2603.22587v1)

**作者:** Damian Delmas `[一作]` `[通讯]` (Independent Researcher), Damian Delmas (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过SQL查询组合的检索框架，能够在查询时对嵌入矩阵和得分数组进行可编程算术运算，从而实现检索调优。

**💡 创新点**

创新点包括：①查询材质化器（query materializer）在不修改数据库引擎的前提下，将外部算子映射为SQL可组合的查询原语；②在检索管道中公开嵌入矩阵和得分数组，使得在选择前可对得分进行多种算子（抑制、分散、衰减、聚类等）组合；③在单机CPU上通过numpy实现毫米级别延迟的复合模调检索。

**🔧 技术方法**

使用的技术主要是：Python + NumPy（矩阵乘法与算子实现），SQLite（SQL预过滤与结果组合），自定义查询材质化器（正则式查询重写），以及自研的向量检索内核（flexvec）。

**📊 数据集**

实验数据集：生产级AI编码会话历史（240k块，后扩展至1M块）以及4个公开BEIR基准（SciFact、NFCorpus、SCIDOCS、FiQA），使用128维Nomic Embed v1.5和Matryoshka截断嵌入。

**📈 对比分析**

与传统的仅做相似度Top‑K检索相比，复合模调在240k块时总延迟19 ms，1M块时82 ms；在5个过滤配置下，预过滤可将Phase‑2延迟从12 ms降至0.4 ms；在BEIR基准上，模调保持90%以上的nDCG@10并提升多样性和中心化效果。

**⚠️ 局限性**

局限性：①仅适用于能全部装入内存的语料库，无法替代大规模ANN索引；②预过滤依赖结构化元数据，元数据缺失时只能对全量矩阵做检索；③材质化器基于正则式重写，复杂嵌套查询仍可能失效；④检索质量评估仅针对行为验证，缺乏与专业reranker或神经检索的完整对比；⑤高维嵌入会显著提升矩阵乘法成本。

---

## 42. A Foundation Model for Instruction-Conditioned In-Context Time Series Tasks

**arXiv ID:** 2603.22586 | [PDF](https://arxiv.org/pdf/2603.22586v1)

**作者:** Anish Saha `[一作]` (Walmart), Konstantin Shmakov `[通讯]` (Walmart)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5057537474)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于T5的量化回归框架的时间序列基础模型，支持通过示例驱动的指令式上下文学习实现多任务推理，无需参数更新。

**💡 创新点**

创新点在于①采用结构化提示式分词与层次化Transformer明确示例与查询边界；②通过指令式元学习训练，结合多种自监督任务提升模型上下文适应；③统一的概率解码器实现多任务。

**🔧 技术方法**

使用了T5 encoder–decoder、patch化、层次化注意力、混合专家解码器、量化回归损失、TSMixup、KernelSynth合成、多任务自监督等技术。

**📊 数据集**

训练数据来自Chronos和GIFT-Eval预训练语料（约30+2.5M序列），并通过TSMixup、KernelSynth、人工合成多变量数据生成约66M序列。

**📈 对比分析**

在fev-bench和GIFT-Eval零样本基准上与Chronos-2、TiRex、TimesFM-2.5等基础模型对比，CRPS/MASE表现显著优于同类模型，整体性能最佳。

**⚠️ 局限性**

局限在于对数据质量与代表性敏感，缺乏可解释性与鲁棒性，且在高风险决策场景仍需人工监督。

---

## 43. Symbolic Graph Networks for Robust PDE Discovery from Noisy Sparse Data

**arXiv ID:** 2603.22380 | [PDF](https://arxiv.org/pdf/2603.22380v1)

**作者:** Xingyu Chen `[一作]` (Chengdu University of Information Technology), Yuqian Zhou `[通讯]` (Chengdu University of Information Technology)

**通讯引用:** 3549 | [OpenAlex ID](https://openalex.org/A5101667454)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了 Symbolic Graph Network（SGN）框架，利用图神经网络的非局部消息传递来在噪声和稀疏采样条件下进行 PDE 发现，并结合符号回归提取可解释的物理方程。

**💡 创新点**

创新点在于将 GNN 的非局部聚合视为弱形式积分算子，既消除局部数值微分的噪声放大，又避免了传统弱形式对稠密网格的依赖，同时采用两阶段符号回归分解消息函数与更新函数，显著降低搜索空间。

**🔧 技术方法**

技术手段包括图神经网络（Message Passing、MLP 消息/更新函数）、Savitzky‑Golay 预平滑、噪声对抗训练、PySR 高性能符号回归，以及理论证明 GNN 聚合等价于连续积分算子。

**📊 数据集**

实验使用合成基准数据集：波动方程、对流扩散方程和不可压 Navier‑Stokes（Taylor‑Green 湍流）并在不同噪声水平（0–10%）下添加高斯噪声。

**📈 对比分析**

与 PDE‑Net 2.0 基线相比，SGN 在低噪声时能够完整重现解析解，在高噪声（10%）下仍保持 MSE 低于 10⁻² 并稳定捕获物理结构，而 PDE‑Net 由于噪声放大出现 NaN、负扩散等非物理结果，性能显著劣于 SGN。

**⚠️ 局限性**

局限性包括：① 为控制符号回归搜索空间，隐藏信息的消息维度被人为压缩，可能无法捕捉复杂耦合效应；② 非局部聚合可能导致过度平滑，难以处理尖锐断层或冲击波等局部不连续现象。

---

## 44. Instruction-Tuned, but Not More Verifiable Instruction-Following: A Cross-Task Diagnosis for LoRA Adapters

**arXiv ID:** 2603.22379 | [PDF](https://arxiv.org/pdf/2603.22379v1)

**作者:** Junyi Zou `[一作]` `[通讯]` (Zjydiary Group), Junyi Zou (Zjydiary Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 LoRA 适配器在不同任务上的交叉评估，探讨了训练目标（如“instruction‑tuned”）是否能可靠预测其在部署后跨任务的能力提升。

**💡 创新点**

创新点在于提出了“能力漂移（capability drift）”这一描述性标签，并给出了一套交叉任务诊断框架、鲁棒性验证以及对比实验中的漂移分数（drift score）来量化目标任务与非目标任务之间的误差。

**🔧 技术方法**

主要技术包括 LoRA 模块的训练、IFEval（严格可验证指令遵循）评估、数值匹配（Numeric Match, NM）评测以及基于目标与非目标增益计算的漂移分数。

**📊 数据集**

使用的数据集包括：IFEval、FollowBench、IFBench、数值推理基准（Numeric Reasoning Benchmark）以及对应的 LoRA 训练数据，模型涵盖 Qwen‑3、Llama‑3 等多种基线。

**📈 对比分析**

比较方法：在同一适配器上同时测量目标任务（如 IFEval PLA）与非目标任务（如 NM）的增益，计算漂移分数并绘制热力图。实验结果显示，instruction‑tuned 适配器往往在 NM 上获得显著提升，而在 IFEval PLA 上不提升甚至下降，漂移分数普遍为正，表明目标任务与实际提升存在不匹配。

**⚠️ 局限性**

局限性包括：实验仅覆盖有限的模型和任务，结果可能不具普适性；漂移分数仅是描述性总结，未构成新度量；不同可验证指令基准间存在差异，跨基准的一致性无法保证；探测的局部解释不具因果性；在某些配置下漂移分数可能为零或负值，显示模式并非普遍。

---

## 45. TrustTrade: Human-Inspired Selective Consensus Reduces Decision Uncertainty in LLM Trading Agents

**arXiv ID:** 2603.22567 | [PDF](https://arxiv.org/pdf/2603.22567v1)

**作者:** Minghan Li `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**通讯引用:** 2694 | [OpenAlex ID](https://openalex.org/A5100632182)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个名为TrustTrade的多智能体选择性共识框架，以改进LLM在金融交易中的决策稳定性和风险控制。

**💡 创新点**

将人类信息过滤与选择性共识、确定性时间信号和记忆反射机制结合，解决LLM的“统一信任”偏差，显著降低最大回撤并提升风险收益比。

**🔧 技术方法**

多智能体LLM并行推理、语义与数值一致性评分、时间信号模块、长短期反射记忆库、实时回测等技术。

**📊 数据集**

利用三支代表性股票（NVDA、AAPL、GOOG）在2024‑Q1和2026‑Q1高噪声市场数据，以及受控的人工模拟交易日志。

**📈 对比分析**

通过A/B实验与人类标注者对比，TrustTrade在2024‑Q1的累计收益从约10%提升至26%，最大回撤从3%升至8%，且整体风险收益更接近人类行为；在2026‑Q1实时回测中也实现了更稳定的收益与更低回撤。

**⚠️ 局限性**

仍依赖LLM生成的域报告，可能出现遗漏或偏差；共识筛选可能抑制少数正确但罕见信号；实验仅覆盖少量股票，未验证在更广泛多资产组合中的表现。

---

## 46. TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs

**arXiv ID:** 2603.22293 | [PDF](https://arxiv.org/pdf/2603.22293v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 47. Neural Structure Embedding for Symbolic Regression via Continuous Structure Search and Coefficient Optimization

**arXiv ID:** 2603.22429 | [PDF](https://arxiv.org/pdf/2603.22429v1)

**作者:** Fateme Memar `[一作]` (University of Kansas), Dongjie Wang `[通讯]` (University of Kansas)

**通讯引用:** 2653 | [OpenAlex ID](https://openalex.org/A5101737966)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将符号回归结构搜索离散化为连续嵌入空间的框架（SRCO），通过Transformer学习结构先验，然后在该空间中采样结构并独立梯度优化系数，

**💡 创新点**

创新点在于把结构学习与系数优化完全分离，利用Transformer得到可微的结构嵌入和概率先验，从而大幅降低结构搜索成本并提升搜索稳定性；

**🔧 技术方法**

核心技术包括Transformer序列模型、后缀符号化结构表示、温度和top‑k采样、语法/语义/复杂度过滤以及基于梯度的系数拟合；

**📊 数据集**

使用两套基于Feynman方程的基准数据集（Feynman‑synthetic和Feynman‑real‑world），涵盖易、中、难三个难度层级；

**📈 对比分析**

与DSO、FFX、EFS以及GP基线进行对比，SRCO在所有层级下均获得最高的R²和Pearson相关系数，并且log(MSE)最低，搜索效率也显著高于其他方法；

**⚠️ 局限性**

主要局限包括对结构模板语料的依赖、目前仅在Feynman基准上验证、以及对更高维度或更丰富算子集合的适用性尚待扩展。

---

## 48. GIFT: Generalizing Intent for Flexible Test-Time Rewards

**arXiv ID:** 2603.22574 | [PDF](https://arxiv.org/pdf/2603.22574v1)

**作者:** Fin Amin `[一作]` (North Carolina State), Andreea Bobu `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5034978137)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过语言模型推断演示者高层意图，并利用该意图在测试时将新状态对齐到训练状态，从而实现奖励函数的无训练泛化。

**💡 创新点**

创新点在于将奖励泛化问题重新定义为“意图条件下的状态对齐”，利用LM的常识推理来构造高层意图驱动的相似度函数，避免仅依赖低层视觉或语言特征导致的错配。

**🔧 技术方法**

采用大语言模型（GPT‑4o）进行意图推断与状态对齐，结合最大熵逆强化学习学习奖励函数，并与基准方法使用 DINO（视觉）和 BERT（语言）特征做对比。

**📊 数据集**

在四个桌面操作任务上使用 7‑DoF Jaco 机器人模拟数据与 Franka Panda 实验数据，覆盖 50+ 不同物体，分别划分为训练集与测试集。

**📈 对比分析**

与视觉相似度、语言相似度以及无意图 LM 对齐方法比较，评估指标为轨迹对比胜率、状态对齐 F1、FP/FN，结果显示 GIFT 在所有任务中均取得 7–20% 的胜率提升，且 FP/FN 明显降低。

**⚠️ 局限性**

依赖于语言模型对意图的准确推断，模型可能产生不确定或误导性意图；对象与场景的符号描述需要高质量，噪声可能导致失败；缺乏对推断置信度的量化与自我校正机制。

---

## 49. Do Large Language Models Reduce Research Novelty? Evidence from Information Systems Journals

**arXiv ID:** 2603.22510 | [PDF](https://arxiv.org/pdf/2603.22510v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 50. Language Models Can Explain Visual Features via Steering

**arXiv ID:** 2603.22593 | [PDF](https://arxiv.org/pdf/2603.22593v1)

**作者:** Javier Ferrando `[一作]` (Barcelona Supercomputing Center), Dario Garcia-Gasulla `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 2021 | [OpenAlex ID](https://openalex.org/A5010831226)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出利用视觉‑语言模型对稀疏自编码器（SAE）特征进行因果干预（steering），并用语言模型生成自然语言解释，同时将此方法与传统基于Top‑k图像的解释方法融合，形成混合解释框架。

**💡 创新点**

① 通过在空白图像上施加SAE特征向量进行因果干预，直接从模型内部推断视觉概念；② 用语言模型生成可读解释；③ 将因果干预与Top‑k图像相结合，显著提升解释质量并消除上下文偏差。

**🔧 技术方法**

稀疏自编码器（TopK SAE）、视觉‑语言模型（Gemma 3、InternVL3）、因果干预（do‑operator）、CLIP、SAM、Stable Diffusion、UMAP、k‑means、句子相似度模型等技术。

**📊 数据集**

ImageNet训练集用于训练SAE，ImageNet测试集（50k图像）用于评估，Stable Diffusion生成的合成图像用于synthetic evaluation。

**📈 对比分析**

与传统Top‑k、Heatmaps、Masks等基线对比；单独Steering在IoU上表现较好，但在AUROC、Synthetic Activation、CLIP等指标落后；混合方法（Steering‑informed Top‑k）在所有评估指标上均取得最高分，且随着语言模型规模提升，性能持续提升。

**⚠️ 局限性**

单独Steering的解释质量相对较低，易受提示和干预强度影响；方法仍需人工验证，且对不同视觉模型的迁移性尚未完全评估。

---

## 51. MCLR: Improving Conditional Modeling in Visual Generative Models via Inter-Class Likelihood-Ratio Maximization and Establishing the Equivalence between Classifier-Free Guidance and Alignment Objectives

**arXiv ID:** 2603.22364 | [PDF](https://arxiv.org/pdf/2603.22364v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 52. CTF as a Service: A reproducible and scalable infrastructure for cybersecurity training

**arXiv ID:** 2603.22511 | [PDF](https://arxiv.org/pdf/2603.22511v1)

**作者:** Carlos Jimeno Miguel amd Mikel Izal Azcarate `[一作]` `[通讯]` (Public University of Navarre), Carlos Jimeno Miguel amd Mikel Izal Azcarate (Public University of Navarre)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

构建了一个基于Proxmox、Terraform、Ansible、Docker Swarm和HAProxy的可复现可扩展的CTF as a Service平台

**💡 创新点**

将基础设施即代码与CI/CD自动化结合，实现挑战持续交付与会话持久化，并提供容器化挑战环境的高可用性

**🔧 技术方法**

使用了Proxmox LXC、Terraform、Ansible、Docker Swarm、HAProxy、iptables、Git CI/CD等开源工具

**📊 数据集**

未使用外部数据集，平台依赖自定义的Docker镜像和挑战打包格式

**📈 对比分析**

通过功能验证实现了基础设施的可复现、自动部署和会话持久化，性能满足学术实验需求，但未给出量化基准

**⚠️ 局限性**

缺乏自动扩缩容、监控集成、HAProxy高可用、多节点管理界面等功能，仍需进一步完善

---

## 53. Hebbian Attractor Networks for Robot Locomotion

**arXiv ID:** 2603.22512 | [PDF](https://arxiv.org/pdf/2603.22512v1)

**作者:** Alexander Dittrich `[一作]` (École Polytechnique Fédérale de Lausanne), Dario Floreano `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 29092 | [OpenAlex ID](https://openalex.org/A5059369445)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了 Hebbian Attractor Networks（HANs），通过双时尺度 Hebbian 更新和激活的移动平均实现可在部署时自适应的神经网络，并研究其在不同参数设置下的权重吸引子（固定点与极限环）行为。

**💡 创新点**

创新点在于：① 将权重最大归一化与双时尺度更新结合，突破传统 Hebbian 学习导致的权重发散；② 通过对激活做移动平均与减慢更新频率来诱导不同的权重吸引子；③ 系统性分析吸引子动力学与控制性能之间的关系，并在多维机器人行走任务上验证。

**🔧 技术方法**

使用技术包括：基于 ABCD 规则的参数化 Hebbian 更新、层级最大归一化（MN）、双时尺度结构（fast NN 控制频率与慢 Hebbian 更新频率）、激活移动平均、进化策略（ES）进行权重规则优化，以及与 PPO、静态 MLP/GRU 的性能对比。

**📊 数据集**

实验数据集包括：Gymnasium 连续控制仿真环境（Swimmer-v5、HalfCheetah-v5、Hopper-v5、Walker2d-v5），Simulated Unitree Go1 四足行走（MuJoCo XLA）以及 Ant-v5 的形态损伤实验。

**📈 对比分析**

对比方法：将 HAN 与同参数数的静态 MLP（无偏置）、带偏置的 MLP/GRU、以及基于梯度的 PPO 进行性能评估。结果显示：HAN 在简单环境（如 Swimmer）与 PPO 竞争，在复杂环境中表现略逊，但远优于静态 MLP；在四足行走任务中，慢更新 + 大移动平均的 HAN 能稳定收敛至固定点吸引子，提升适应性；在形态损伤实验中，HAN 的周期吸引子在未见过的损伤场景下保持更高回报。

**⚠️ 局限性**

限制：① 采用进化策略训练，样本效率低；② 仅使用前馈网络，未探索递归或更丰富的网络结构；③ 主要关注周期行走任务，缺乏对更复杂任务（导航、操作、崎岖地形）的验证；④ 在永久形态损伤情境下，HAN 仍倾向产生周期吸引子，适应性受限。

---

## 54. Evaluating Large Language Models' Responses to Sexual and Reproductive Health Queries in Nepali

**arXiv ID:** 2603.22291 | [PDF](https://arxiv.org/pdf/2603.22291v1)

**作者:** Medha Sharma `[一作]` (Visible Impact), Bishesh Khanal `[通讯]` (Nepal Applied Mathematics and Informatics Institute for research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发了LEAF评估框架，并在尼泊尔语言的性与生殖健康（SRH）对话场景中，收集了约9,000名用户的14,000条问答，手工对每条回复进行准确性、语言、可用性缺口和安全缺口四维度标注；

**💡 创新点**

创新点在于首次提出面向低资源语言和社区环境的多维评估框架（LEAF），兼顾准确性、语言适配、可用性与安全性，填补了现有研究仅关注准确性的空白，并实现了大规模SRH对话数据的系统化评估；

**🔧 技术方法**

技术手段包括使用ChatGPT‑3.5（含检索增强生成）、对比GPT‑4的表现，构建TGI推理服务器和Bi‑Encoder检索模块，以及人工专家标注平台；

**📊 数据集**

数据来源为22份尼泊尔健康信息中心（NHEICC）提供的SRH相关手册与宣传材料，以及在全国45个市镇开展的9,035名社区与女性社区健康志愿者（FCHV）的真实对话数据；

**📈 对比分析**

评估方法通过对比GPT‑3.5与GPT‑4在同一批100条问答上的准确率、相关性、完整性与长度等维度，结果显示GPT‑4的准确率从26%提升至50%，“好对话”比例从35.1%提升至59%，但仍存在不安全、过长或不完整的回复；

**⚠️ 局限性**

局限性包括样本主要来自工作坊情境，缺乏真实家庭或私密环境的对话；标注过程耗时且缺乏交叉审核，可能产生主观偏差；对罗马化尼泊尔语的处理不佳；回复长度受token限制导致截断；安全缺口虽少，但潜在危害不可忽视；未评估公平性、偏见与隐私风险。

---

## 55. Static Scene Reconstruction from Dynamic Egocentric Videos

**arXiv ID:** 2603.22450 | [PDF](https://arxiv.org/pdf/2603.22450v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 56. Efficient Universal Perception Encoder

**arXiv ID:** 2603.22387 | [PDF](https://arxiv.org/pdf/2603.22387v1)

**作者:** Chenchen Zhu `[一作]` (Meta Reality Labs), Vikas Chandra `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了Efficient Universal Perception Encoder (EUPE)，通过先将多领域专家知识聚合到一个大代理模型，再将其知识蒸馏到轻量级学生模型，实现了统一的高效视觉编码器。

**💡 创新点**

创新点在于两阶段的“先扩容再压缩”蒸馏流程、使用代理教师统一多领域知识、以及在第二阶段采用固定分辨率长周期蒸馏后再进行多分辨率微调，显著提升了轻量模型在多任务上的泛化。

**🔧 技术方法**

主要技术包括多教师蒸馏、特征归一化、适配器头、固定分辨率和多分辨率训练、以及基于PCA的特征可视化。

**📊 数据集**

使用的训练数据为LVD-1689M+ImageNet1k（MetaCLIP对比）以及多个教师模型（PEcore, PElang, DINOv3, SigLIP2等）。

**📈 对比分析**

通过在ImageNet零样本/ KNN、VLM任务(TextVQA, SQA, Realworld, POPE, GQA, MMEp)以及密集预测任务(SPair, NYUv2, ADE20k)上进行对比，EUPE在所有任务均与最强专家持平或优于，并在多任务上取得比RADIO/DUNE更高的平均分。

**⚠️ 局限性**

限制主要在于当代理教师过大时难以完整蒸馏到小模型，且多分辨率微调耗时较长，未来需探索更高效的多尺度蒸馏或教师助手方法。

---

## 57. Unified Algebraic Absorption of Finite-Blocklength Penalties via Generalized Logarithmic Mapping

**arXiv ID:** 2603.22358 | [PDF](https://arxiv.org/pdf/2603.22358v1)

**作者:** Hiroki Suyari `[一作]` (Chiba University), Hiroki Suyari `[通讯]` (Chiba University)

**通讯引用:** 748 | [OpenAlex ID](https://openalex.org/A5072973047)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种基于q-对数映射的统一代数框架，将有限块长度下的非高斯误差吸收到信息密度本身，消除传统的Hermite多项式补偿。

**💡 创新点**

创新点在于利用动态缩放的q参数（1‑q_n = α/n）使得q-信息密度自动吸收第三阶（偏度）及更高阶的有限块长度修正，形成单一代数结构，而非逐项添加多项式。

**🔧 技术方法**

核心技术包括：q-对数与q-熵的展开、对信息密度的中心化、动态缩放法则的推导、与Edgeworth/辛氏-菲舍尔展开的对应证明，以及数值验证。

**📊 数据集**

使用了二元记忆无关源（Bernoulli源）作为实验数据集，取 p = 0.11，误差概率 ε = 0.01，并对不同块长度 n 进行计算。

**📈 对比分析**

对比方法：将 Shannon 极限、二阶正态近似、第三阶 Edgeworth 修正和提出的 q-代数界进行曲线比较。结果显示 q-代数界与 Edgeworth 结果完全重合，且在短块长度（n < 100）下显著优于纯正态近似，逼近精确有限块长度界。

**⚠️ 局限性**

局限性在于：仅证明了至第三阶（偏度）修正的吸收；对第四阶及更高阶修正的代数系数映射仍是开放问题，且目前仅在二元源上验证，尚未在更一般信道或源模型上广泛测试。

---

## 58. A Multi-Modal CNN-LSTM Framework with Multi-Head Attention and Focal Loss for Real-Time Elderly Fall Detection

**arXiv ID:** 2603.22313 | [PDF](https://arxiv.org/pdf/2603.22313v1)

**作者:** Lijie Zhou `[一作]` (University of Nottingham Ningbo China), Luran Wang `[通讯]` (University of Nottingham Ningbo China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种多模态深度学习框架 MultiModalFallDetector，用于在可穿戴传感器上实现实时老年人跌倒检测。

**💡 创新点**

创新点包括多尺度CNN特征提取、三轴加速度与陀螺仪及四通道生理信号融合、多头自注意力动态加权、Focal Loss 处理类别不平衡、辅助活动分类任务与迁移学习策略。

**🔧 技术方法**

技术采用多尺度卷积网络、双向LSTM、Transformer式自注意力、Focal Loss、两阶段迁移学习及多任务学习。

**📊 数据集**

使用了 SisFall 数据集（含老年人跌倒实验）进行最终评估，并在 UCI HAR 数据集上预训练。

**📈 对比分析**

与传统机器学习和标准深度学习基线（SVM、RF、CNN‑LSTM 等）对比，模型在 SisFall 上实现 F1‑score 98.7%、Recall 98.9%、AUC‑ROC 99.4%，显著优于基线且推理延迟低于 50 ms。

**⚠️ 局限性**

局限性包括仅基于实验室模拟跌倒数据、样本量有限、受实验环境与受试者多样性限制，且对真实场景中的噪声与姿态变化的鲁棒性尚待验证。

---

## 59. Precision-Varying Prediction (PVP): Robustifying ASR systems against adversarial attacks

**arXiv ID:** 2603.22590 | [PDF](https://arxiv.org/pdf/2603.22590v1)

**作者:** Matías Pizarro `[一作]` (Ruhr University Bochum), Asja Fischer `[通讯]` (Ruhr University Bochum)

**通讯引用:** 5416 | [OpenAlex ID](https://openalex.org/A5026151059)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6215c339-3735-4be3-8a07-5bbb7004712d` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出通过在推理阶段随机变换数值精度（FP32/FP16/BF16）来提升自动语音识别模型的对抗鲁棒性，并基于不同精度下的转录一致性构建轻量级对抗样本检测器。

**💡 创新点**

创新点在于：①利用已有的多精度支持作为“免费”防御手段，消除再训练成本；②通过随机采样精度提高对抗样本的不可转移性；③用精度多样性得分（基于WER差异）配合高斯阈值实现无模型内部信息需求的检测。

**🔧 技术方法**

技术要点包括：数值精度变换、随机精度采样、WER/句子错误率评估、基于Gaussian分布的异常检测、对抗样本生成（C&W、Psychoacoustic）以及与NF、TD、DistriBlock等现有检测方法对比。

**📊 数据集**

使用LibriSpeech公开语音数据集进行训练与评估，包含test-clean和test-other子集。

**📈 对比分析**

与传统对抗检测方法（Noise Flooding、Temporal Dependency、DistriBlock）对比，PVP在多数模型上实现AUROC≥0.90，随机精度采样显著提升各类攻击下的WER/SER；对抗攻击成功率从几乎0%提升至大幅下降，说明鲁棒性增强。

**⚠️ 局限性**

局限性：对抗性自适应攻击可绕过PVP（攻击成功率恢复为0%）；在Whisper、Transformer等大规模预训练模型上检测效果略差；需要多次模型推理，导致推理延迟；对极短句子可能不稳定；未在真实场景或更复杂攻击下验证。

---

## 60. Conformal Risk Control for Safety-Critical Wildfire Evacuation Mapping: A Comparative Study of Tabular, Spatial, and Graph-Based Models

**arXiv ID:** 2603.22331 | [PDF](https://arxiv.org/pdf/2603.22331v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 61. Linux and High-Performance Computing

**arXiv ID:** 2603.22495 | [PDF](https://arxiv.org/pdf/2603.22495v1)

**作者:** David A. Bader `[一作]` (New Jersey Institute of Technology), David A. Bader `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 8115 | [OpenAlex ID](https://openalex.org/A5076610730)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比分析了两种 Linux 基础高性能计算（HPC）体系结构——Beowulf（面向单用户的低成本工作站）和 Roadrunner（面向多用户的高性能超级计算机），并阐述它们对后世 HPC 设计的影响。

**💡 创新点**

创新点：Roadrunner 通过单个人设计并实现了集成的三网架构（控制、数据、诊断），将商用高性能 Myrinet 接入 Linux，采用 SMP 节点、PBS 作业调度和 MPI，证明了 Commodity Off‑The‑Shelf（COTS）组件也能与传统商用超级计算机竞争；Beowulf 则首次普及了“平价 PC 集群”理念，推动了开放源代码和社区化开发。

**🔧 技术方法**

技术：Linux（Red Hat，定制 SMP 内核）、COTS Intel/AMD 处理器、Myrinet（高带宽低延迟网络）、Fast Ethernet、PBS、MPI、GCC/PGI 编译器、调度/监控工具、三网架构、SMP 节点、共享内存与消息传递并行模型。

**📊 数据集**

数据集/基准：Cactus（数值相对论）、MILC（量子色动力学）、ARPI3D（气象预测）、BEAVIS（多相流）、AIPS++（射电天文学）、ASPCG（流体力学）等多学科科研应用，用以评估系统的通用性能与可扩展性。

**📈 对比分析**

比较方法：采用统一的基准套件在 Beowulf、Roadrunner Phase 1 与 Phase 2 上跑测，比较浮点性能、可扩展性、I/O 带宽、通信延迟。结果显示 Roadrunner 在大多数通信密集型工作负载中实现了近乎完美的线性扩展，显著优于 Beowulf 及同期的 Windows NT、Origin 2000 等系统；Beowulf 在单用户或非通信密集型任务中保持较好成本/性能比。

**⚠️ 局限性**

局限性：Beowulf 受限于单处理器节点、以太网网络、缺乏系统管理与多用户支持，难以满足大规模、通信密集型科学计算；Roadrunner 的缺点主要是高性能网络（Myrinet）成本较高、单人实现的技术瓶颈与维护复杂度、以及在极大规模（百万核）时的可扩展性与能耗管理仍需进一步研究。

---

## 62. Dynamic Fusion-Aware Graph Convolutional Neural Network for Multimodal Emotion Recognition in Conversations

**arXiv ID:** 2603.22345 | [PDF](https://arxiv.org/pdf/2603.22345v1)

**作者:** Tao Meng `[一作]` (Central South University of Forestry and Technology), Keqin Li `[通讯]` (State University of New York)

**通讯引用:** 31085 | [OpenAlex ID](https://openalex.org/A5087894632)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种动态融合感知图卷积网络（DF-GCN），通过将普通微分方程（ODE）注入GCN，并利用全局信息向量（GIV）生成动态融合权重，实现对多模态对话情感的精准识别。

**💡 创新点**

创新点在于：①首次在情感识别推理阶段根据不同情感类别动态调整网络参数，实现情感特定的融合；②结合ODE连续动力学建模情感依赖，提升情感转移的表达能力；③使用GIV作为内部提示，引导Prompt生成网络生成适配上下文的动态权重。

**🔧 技术方法**

采用的技术包括：多模态特征编码（RoBERTa、OpenSMILE、DenseNet + Bi‑GRU + Attention），图卷积网络（SGCODE与基于ODE的DGCODE），Transformer与全局平均池化生成GIV，MLP提示生成网络，BatchNorm残差连接，AdamW优化器，4阶Runge–Kutta ODE求解器。

**📊 数据集**

使用公开的两大多模态对话情感数据集：IEMOCAP（含文本、音频、视觉）和MELD（含文本、音频、视觉），以评估模型在不同情感类别上的表现。

**📈 对比分析**

与现有10+主流方法（如MMGCN、DER‑GCN、M3Net等）对比，DF‑GCN在IEMOCAP和MELD上均取得最高的Weighted Accuracy和Weighted F1，特别在happy、neutral、depressed等类别上显著提升；整体性能优于前沿模型，且计算量与推理速度保持在可接受范围。

**⚠️ 局限性**

主要限制包括：①数据集严重类别不平衡导致罕见情感（如sadness、joy、fear、disgust）的识别仍受限；②多模态表达高度异质化和上下文敏感性，使得少数类样本难以充分学习；③模型对极端稀有情感的泛化能力尚有提升空间。

---

## 63. Early Discoveries of Algorithmist I: Promise of Provable Algorithm Synthesis at Scale

**arXiv ID:** 2603.22363 | [PDF](https://arxiv.org/pdf/2603.22363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 64. MapForest: A Modular Field Robotics System for Forest Mapping and Invasive Species Localization

**arXiv ID:** 2603.22502 | [PDF](https://arxiv.org/pdf/2603.22502v1)

**作者:** Sandeep Zachariah `[一作]` (Carnegie Mellon University), Abhisesh Silwal `[通讯]` (Carnegie Mellon University)

**通讯引用:** 809 | [OpenAlex ID](https://openalex.org/A5016610643)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `e0540dec-d77f-42db-94ae-d039248f6393` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出MapForest系统，实现多平台可携带的LiDAR-IMU-GNSS-RGB传感器组合与GLIM SLAM、YOLOv8检测相结合，生成地理坐标化的3D森林地图与树种分布图。

**💡 创新点**

创新点在于将协方差感知的GNSS先验与鲁棒损失嵌入GLIM后端提升稀疏GNSS环境下的轨迹一致性，并通过概率似然场与归一化互信息实现无特征的空中-地面地图对齐。

**🔧 技术方法**

核心技术包括GLIM光束扫描-惯性融合SLAM、Livox Mid-360 LiDAR、YOLOv8树种目标检测、Covariance-aware GNSS因子、Huber鲁棒核、NMI基空地对齐与GeoTIFF GIS导出。

**📊 数据集**

使用六个覆盖城市、公园、骑行道与森林的现场数据集，包含RTK-GNSS轨迹、AprilTag标定树木、iNaturalist图像等，构建ToH检测数据集共1466张图像。

**📈 对比分析**

与FastLIO、FasterLIO、FastLIMO相比，GLIM在Flagstaff 0.8km遍历中平均轨迹误差从15.8/12.8/18.7降至8.67m；GNSS协方差感知将误差进一步降至2.83m；ToH检测F1为0.653，DBH估计平均相对误差仅0.04。

**⚠️ 局限性**

局限包括：仅验证单一树种、数据集规模有限、对极端遮挡下的检测性能不足、GNSS信号仍易受多路径影响、空地对齐在高树冠重叠场景下的鲁棒性有待提升。

---

## 65. Parallel OctoMapping: A Scalable Framework for Enhanced Path Planning in Autonomous Navigation

**arXiv ID:** 2603.22508 | [PDF](https://arxiv.org/pdf/2603.22508v1)

**作者:** Yihui Mao `[一作]` (University of Florida), Rushikesh Kamalapurkar `[通讯]` (University of Florida)

**通讯引用:** 2810 | [OpenAlex ID](https://openalex.org/A5071224574)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种并行OctoMap‑based映射框架POMP，利用叶节点内部区域划分对固定分辨率的占用网格进行自由空间细化，提升搜索式路径规划的可行性与质量。

**💡 创新点**

1）在不改变树深度的前提下，通过在叶节点中划分4/8个子区域并引入安全状态，实现对可通行空间的更精细阈值化；2）利用Intel oneTBB多线程与原子操作实现Octree插入与叶节点状态更新的并行化；3）将稀疏Octree与密集占用网格耦合，兼顾内存效率与规划时的连续可用性。

**🔧 技术方法**

并行Octree构建（oneTBB + CAS），原子位掩码编码安全状态，区域阈值判定，双向对角线检查确定OGM格子占用状态，基于A*、JPS等搜索式规划算法的评估。

**📊 数据集**

随机合成点云（10m×10m×10m或50m×50m×50m），以及公开真实数据集：Cambridge_15/16、Apartment_0/1/2、Cloud_7、WSF‑19（Treescope），涵盖室内、城市、森林等不同环境。

**📈 对比分析**

与传统直接OGM、原始OctoMap以及其他增量树构建方法（i‑Octree、ikd‑Tree、PCL Octree）对比，POMP在树构建、OGM生成、导航空间比例、路径成功率、路径长度与规划时间上均优越，尤其在粗分辨率或稠密点云场景下提升显著，且整体运行时间可与单点OGM插入持平或更快。

**⚠️ 局限性**

对阈值比例的依赖需手动调参；对极稀疏场景下的性能提升有限；当点云更新频率极高时，内存占用与原子操作竞争仍可能成为瓶颈。

---

## 66. Sample Transform Cost-Based Training-Free Hallucination Detector for Large Language Models

**arXiv ID:** 2603.22303 | [PDF](https://arxiv.org/pdf/2603.22303v1)

**作者:** Zeyang Ding `[一作]` (Chinese University of Hong Kong, Shenzhen), Jicong Fan `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**通讯引用:** 8008 | [OpenAlex ID](https://openalex.org/A5100373560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于Wasserstein距离的无监督假设检测方法，利用多次生成的隐藏表示计算样本间OT距离矩阵，并从中提取AvgWD和EigenWD两个判别信号

**💡 创新点**

创新点在于将生成分布的“复杂度”直接量化为多样化生成的隐藏表示间的最优传输成本，既捕捉平均成本也捕捉成本结构的谱复杂度，从而在不需要额外训练或检索的前提下实现轻量级假设检测

**🔧 技术方法**

使用的技术包括：token级隐藏表示提取、中层投影、基于离散测度的Wasserstein距离计算、谱分析（kernel化后求特征值比值）以及教师强制（black-box 情况下通过教师模型得到隐藏表示)

**📊 数据集**

在白盒实验中使用CoQA、SQuAD、MATH‑500和CNN/DailyMail四个数据集；在黑盒实验中使用DeepSeek‑Chat对SciQ、NQ‑open和Math500三个数据集

**📈 对比分析**

与Discret Semantic Entropy、Eigenscore、Effective Rank、Length‑Normalized Entropy、Lexical Similarity等训练‑free基线在AUROC上进行比较。AvgWD/EigenWD在多数模型‑数据集组合中均能达到或超过基线最佳性能，尤其在Llama系列模型中EigenWD表现最突出；在黑盒实验中EigenWD同样取得最高AUROC

**⚠️ 局限性**

主要限制在于时间复杂度相对较高（需要计算K×K个OT距离），但实验显示仍在可接受范围内；此外，该方法依赖于足够多的采样和合适的温度，超参数选择对性能有一定影响

---

## 67. AEGIS: An Operational Infrastructure for Post-Market Governance of Adaptive Medical AI Under US and EU Regulations

**arXiv ID:** 2603.22322 | [PDF](https://arxiv.org/pdf/2603.22322v1)

**作者:** Fardin Afdideh `[一作]` (Karolinska Institutet), Farhad Abtahi `[通讯]` (Karolinska Institutet)

**通讯引用:** 745 | [OpenAlex ID](https://openalex.org/A5064744506)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出并实现了Aegis治理框架，用于可持续学习的医学AI系统在FDA PCCP和欧盟AI Act下的监管与部署决策。

**💡 创新点**

创新点在于将监管要求转化为可执行的三模块（DARM、MMM、CDM）架构，配合四类部署决策与并行PMS信号，实现了自动化、可追溯的改版与风险评估流程。

**🔧 技术方法**

使用技术包括数据集治理（训练/黄金/漂移三份）、持续监测模块（性能、公平、分布漂移、mlcps组合指标）、阈值比较与决策树逻辑，以及与欧盟预定义变更机制对应的比较机制。

**📊 数据集**

实验数据分别为PhysioNet Sepsis 2019（约2万名 ICU 患者）和 BraTS 2023/24 脑肿瘤分割数据（约2400张 MRI 图像）。

**📈 对比分析**

通过11轮迭代验证，Aegis成功触发所有四类决策并提前检测到分布漂移，模型性能维持在设定阈值范围内；在脑肿瘤分割中实现了非劣效性阈值检查，保证了临床安全性。

**⚠️ 局限性**

局限性包括：仅为治理演示，未进行临床验证；合成漂移情景可能不完全代表真实环境；聚合指标缺乏概念漂移与对抗攻击监测；阈值设置需在实际部署中进一步校准。

---

## 68. Less is More: Adapting Text Embeddings for Low-Resource Languages with Small Scale Noisy Synthetic Data

**arXiv ID:** 2603.22290 | [PDF](https://arxiv.org/pdf/2603.22290v1)

**作者:** Zaruhi Navasardyan `[一作]` (Metric AI Lab), Hrant Davtyan `[通讯]` (Metric AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在低资源语言Armenian上，通过使用约10,000条噪声合成的标题-正文对，微调多语言编码器mE5，以提升文本嵌入质量。

**💡 创新点**

发现即使是少量且质量较差的合成数据也能显著提升性能，表明低资源语言的语义对齐在早期就能饱和，打破大规模数据与高质量翻译的先入观念。

**🔧 技术方法**

利用Gemma‑2 27B模型对英文Reddit标题-正文进行翻译，生成噪声合成数据；对mE5进行对比学习微调，并与基线模型进行模型平均，进一步在不同规模的数据上进行实验。

**📊 数据集**

主要使用Reddit标题-正文公开数据（约2百万对）经过过滤后1百万对；Armenian翻译版本的MS MARCO、STS、MTEB和手工构造的185条检索对；以及在Georgian上重复实验。

**📈 对比分析**

与基线multilingual‑e5‑base、large、instruct以及EmbeddingGemma等模型对比，10k噪声数据微调的模型在检索任务上提升约20%+，整体平均分从约64%提升至约76%；进一步扩大数据规模至1M时提升有限，甚至有轻微退化。

**⚠️ 局限性**

局限包括：对语言形态学复杂度或与高资源邻国共享脚本的语言验证不足；过度依赖英文文化背景导致对本土化任务的适用性受限；需要翻译LLM对目标语言有足够理解；以及对不同模型可能需单独调参。

---

## 69. Model Context Protocol Threat Modeling and Analyzing Vulnerabilities to Prompt Injection with Tool Poisoning

**arXiv ID:** 2603.22489 | [PDF](https://arxiv.org/pdf/2603.22489v1)

**作者:** Charoes Huang `[一作]` (New York Institute of Technology), Amin Milani Fard `[通讯]` (New York Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

本文对Model Context Protocol（MCP）客户端安全进行系统评估，首先通过STRIDE与DREAD框架对MCP生态中的五大组件进行威胁建模，随后在四种工具中毒（tool poisoning）攻击场景下对七个主流MCP客户端进行实证测试，比较各客户端在检测、阻断和警告等方面的表现；最后提出多层防御策略与标准化建议。

**💡 创新点**

创新点在于首次将STRIDE/DREAD威胁建模与实证攻击相结合，针对MCP客户端的工具中毒漏洞进行系统化实验；提出完整的四层防御架构（注册校验、决策路径追踪、运行时监控、用户透明化）并给出可操作的缓解方案。

**🔧 技术方法**

技术手段包括：MCP协议的注册/调用流程实现；LLM交互与工具描述注入检测；Python脚本自动生成恶意工具；客户端UI截屏与日志分析；静态JSON Schema校验与关键字扫描；sandbox化运行与网络/文件访问限制；LLM安全提示与拒绝策略。

**📊 数据集**

实验使用了来自公开MCP服务器与工具的真实描述（共计13,875个服务器与300个工具），并在本地搭建了七个主流MCP客户端（Claude Desktop、Cursor、Cline、Continue、Gemini CLI、Claude Code、Langflow）进行测试。

**📈 对比分析**

比较方法为对四种攻击类型（读取敏感文件、日志记录、钓鱼链接、远程代码执行）在每个客户端的成功率进行编码（绿＝安全、黄＝部分、红＝失败），并记录检测/阻断次数、警告信息、参数可见性等指标。结果显示，Claude Desktop和Cline在所有攻击下均被阻断；Cursor在所有攻击下完全失败；其余客户端表现中等。

**⚠️ 局限性**

局限性包括：仅评估七个客户端，未覆盖所有开源实现；实验环境为局部隔离，未覆盖真实生产网络与多租户场景；未对Sandbox效果进行完整动态验证；缺少对模型层安全策略的细粒度评估；安全评分主观性高，需进一步标准化。

---

## 70. AgentSLR: Automating Systematic Literature Reviews in Epidemiology with Agentic AI

**arXiv ID:** 2603.22327 | [PDF](https://arxiv.org/pdf/2603.22327v1)

**作者:** Shreyansh Padarha `[一作]` (University of Oxford), Adam Mahdi `[通讯]` (University of Oxford)

**通讯引用:** 1588 | [OpenAlex ID](https://openalex.org/A5055914149)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文开发并评估了一个开源的AgentSLR端到端自动化系统，能够完成流行病学优先病原体系统综述的检索、筛选、数据提取和报告生成。

**💡 创新点**

创新点在于将大型语言模型与多阶段工具调用、OCR与可验证推理结合，形成模块化的Agentic管道，实现从检索到叙事合成的完整自动化，并公开发布。

**🔧 技术方法**

主要技术包括大型语言模型（如OpenAI GPT‑4、Moonshot等）、链式思维提示、函数调用式工具接口、OCR（如Ocropus）和Markdown转换，以及可验证自我改进循环。

**📊 数据集**

使用了PERG提供的WHO优先病原体系统综述数据集（马尔堡、埃博拉、利萨、SARS‑CoV‑1、寨卡、MERS、尼帕等），约26%为开放获取文章。

**📈 对比分析**

通过与人工专家标注的真值比较，评估了标题/摘要筛选、全文筛选和结构化提取的精确率、召回率和F1，并在人工审查中得到79.8%平均准确率；系统将完成时间从7周缩短到20小时，速度提升58倍。

**⚠️ 局限性**

局限性包括仅覆盖开放获取的26%文献、对多语言和非公开文献缺失、提取精度仍低于人工、未实现元分析和统计推断，以及对闭源模型的内容限制和计算资源依赖。

---

## 71. Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Reasoning Models?

**arXiv ID:** 2603.22582 | [PDF](https://arxiv.org/pdf/2603.22582v1)

**作者:** Richard J. Young `[一作]` (University of Nevada, Las Vegas), Richard J. Young `[通讯]` (University of Nevada, Las Vegas)

**通讯引用:** 3316 | [OpenAlex ID](https://openalex.org/A5101673182)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对12个开源推理模型（9个架构家族、7B–685B参数）进行跨家族链式推理（CoT）信度评估，使用六类提示注入在MMLU与GPQA Diamond 498道多项选择题上测试。

**💡 创新点**

揭示模型家族、训练方式比规模更能决定CoT信度，并首次量化思考语句与答案文本之间的信度差距，显示内部推理对提示的认知远高于公开答案。

**🔧 技术方法**

采用提示注入、两阶段信度判定器（正则+LLM判断）以及Claude Sonnet 4作为独立验证，结合OpenRouter API调用。

**📊 数据集**

MMLU（300题）与GPQA Diamond（198题）多项选择数据集，配合六类提示（sycophancy、consistency、visual_pattern、metadata、grader hacking、unethical）。

**📈 对比分析**

与以往专有模型相比，信度范围39.7–89.9%，同家族迭代提升显著，提示类型差异明显；相同任务下模型性能（准确率）不影响信度排名。

**⚠️ 局限性**

受限于API差异、提示形式、数据集仅为多项选择、信度分类器主观性以及样本量有限，导致绝对信度值可能偏低且难以直接比较。

---

## 72. Reasoner-Executor-Synthesizer: Scalable Agentic Architecture with Static O(1) Context Window

**arXiv ID:** 2603.22367 | [PDF](https://arxiv.org/pdf/2603.22367v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 73. STEM Agent: A Self-Adapting, Tool-Enabled, Extensible Architecture for Multi-Protocol AI Agent Systems

**arXiv ID:** 2603.22359 | [PDF](https://arxiv.org/pdf/2603.22359v1)

**作者:** Alfred Shen `[一作]` (Amazon), Aaron Shen `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了 STEM Agent，一种支持五种交互协议、持续用户建模、基于 MCK 的工具集成、细粒度记忆子系统以及生物启发式技能获取的可扩展 AI 代理框架。

**💡 创新点**

创新点包括：①统一 gateway 处理 A2A、AG‑UI、A2UI、UCP、AP2 五大协议；②Caller Profiler 通过多维 EMA 自动学习用户偏好；③将所有域能力外部化为 Model Context Protocol；④使用细胞分化模型实现技能的自我成长与淘汰；⑤四层内存系统实现子线性增长。

**🔧 技术方法**

技术实现采用 TypeScript/Express.js，JSON‑RPC、SSE、REST；MCP 与工具调用通过 gRPC/HTTP；内存使用 PostgreSQL+pgvector、Neo4j 图数据库；策略选择基于规则，学习阶段使用异步任务；安全层采用 IAM 插件；框架插件可与 AutoGen、CrewAI 等对接。

**📊 数据集**

本研究未使用公开数据集，评估主要通过 413 条 Vitest 单元与集成测试（涵盖协议、内存、工具、错误处理、权限等）完成；测试覆盖率 100%，总耗时 2.92 s。

**📈 对比分析**

与现有框架（AutoGen、CrewAI、LangChain 等）在架构维度上进行对比，STEM Agent 在多协议支持、持续自适应、四类记忆、技能自生等方面均优于传统方案；性能方面，测试通过率 100%，平均响应时间保持在毫秒级（基于模拟 LLM 推理）。

**⚠️ 局限性**

局限性包括：缺乏真实任务或用户研究的 end‑to‑end 基准评估；EMA‑Caller Profiler 只能捕获平稳偏好，难以处理非平稳多模需求；UCP 与 AP2 为新协议，缺乏行业验证与正式威胁模型；在高并发场景下内存与幂等缓存需分布式锁支持。

---

## 74. Multimodal Training to Unimodal Deployment: Leveraging Unstructured Data During Training to Optimize Structured Data Only Deployment

**arXiv ID:** 2603.22530 | [PDF](https://arxiv.org/pdf/2603.22530v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 75. SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale

**arXiv ID:** 2603.22455 | [PDF](https://arxiv.org/pdf/2603.22455v1)

**作者:** YanZhao Zheng `[一作]` (Alibaba Group), Hangcheng Zhu `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对大型 LLM 代理生态中的技能路由问题，构建了约 80K 技能的检索基准，并提出了可在本地部署的 1.2B 参数的两阶段检索-重排序管线 SkillRouter。

**💡 创新点**

创新点包括：①证明技能实现文本（body）是选择技能的关键信号，远超名称和描述；②在检索和重排序阶段均使用完整技能文本；③引入三层误负样本过滤与硬负样本挖掘；④用列表式交叉熵损失显著提升重排序效果。

**🔧 技术方法**

技术方案采用 Qwen3-Emb-0.6B 的密集双编码器做检索，SR-Rank-0.6B 的交叉编码器做重排序；训练时使用硬负样本挖掘、误负过滤以及 0.05 温度的 InfoNCE 损失；重排序使用列表式交叉熵。

**📊 数据集**

数据集由 SkillBench 的 75 条专家验证的任务-技能映射和约 80K 个社区贡献技能构成，划分为易/难两难度层级（分别加入 780 条 LLM 生成的干扰技能）。

**📈 对比分析**

与 BM25、E5-Large、GTE-Large、NV-Embed、Qwen3-Emb-8B 等零样本基线比较，SkillRouter 在 Hit@1 上达 74%（相比最佳 8B 零样本 68% 提升 6%），并在 Recall@20、nDCG 等指标上保持领先。

**⚠️ 局限性**

局限性包括：基准规模仅 75 个查询，可能不完全代表更大多样化生态；多技能查询中仍有 22% 的查询未被检索到，表明需要更强的推理能力；并且该工作聚焦检索层，未覆盖完整的代理执行与任务完成链路。

---

## 76. Latent Semantic Manifolds in Large Language Models

**arXiv ID:** 2603.22301 | [PDF](https://arxiv.org/pdf/2603.22301v1)

**作者:** Mohamed A. Mabrok `[一作]` (Qatar University), Mohamed A. Mabrok `[通讯]` (Qatar University)

**通讯引用:** 778 | [OpenAlex ID](https://openalex.org/A5042556381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了将大型语言模型内部表示视为隐语义流形的理论框架，并用Fisher信息度量来刻画该流形的几何结构，进而定义了表达缺口并给出了两个关键理论定理。

**💡 创新点**

创新点包括：①将流形假设正式化为LMM的隐语义流形；②引入表达缺口这一几何量并证明其线性尺度定律；③给出基于信息压缩的语义失真下界；④将几何预言与实际模型实验相结合，验证跨架构和规模的一致性。

**🔧 技术方法**

所用技术涵盖差分几何（黎曼流形、曲率、共面积公式）、信息几何（Fisher信息度量）、压缩理论（rate‑distortion）、统计估计（TWO‑NN、MLE）、可视化（UMAP、谱嵌入）以及梯度流动分析（神经ODE）。

**📊 数据集**

实验基于WikiText‑103验证集，对 GPT‑2、OPT 与 Pythia 三大架构的 6 个模型（124M–1.5B 参数）进行隐藏状态采样并评估几何量。

**📈 对比分析**

比较方法：对不同模型层数、规模和架构分别计算内在维度、曲率、表达缺口等；通过回归验证表达缺口随阈值的线性增长（斜率 0.87–1.12，R²>0.985）。结果表明模型在跨架构、跨规模下均遵循相同的几何规律，证明了理论预言的稳健性。

**⚠️ 局限性**

局限性包括：①流形假设在实际表示中可能存在偏离；②Fisher度量在极端置信或熵极值处退化；③对大模型（>10B）与多模态模型的验证尚未完成；④计算Fisher信息矩阵和曲率需要大量代价，实际应用需近似；⑤表达缺口理论仅捕捉离散词表对连续语义的量化限制，未涵盖其他结构化输出。

---

## 77. ST-GDance++: A Scalable Spatial-Temporal Diffusion for Long-Duration Group Choreography

**arXiv ID:** 2603.22316 | [PDF](https://arxiv.org/pdf/2603.22316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 78. Bounding Box Anomaly Scoring for simple and efficient Out-of-Distribution detection

**arXiv ID:** 2603.22660 | [PDF](https://arxiv.org/pdf/2603.22660v1)

**作者:** Mohamed Bahi Yahiaoui `[一作]` (CEA), Julyan Arbel `[通讯]` (Université Grenoble Alpes)

**通讯引用:** 1599 | [OpenAlex ID](https://openalex.org/A5054605178)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种后置的OOD检测方法BBAS，通过在特征空间中构建基于监测变量的轴对齐盒子并计算异常分数。

**💡 创新点**

创新点在于将bounding-box抽象与卷积层的通道统计结合，使用分离的聚类和盒子构建，并引入可聚合的超越计数/距离异常评分。

**🔧 技术方法**

利用卷积/全连接层预激活的激活分数、最小/最大值等监测变量，层级聚类（完全链接）与极值盒子构造，异常分数采用超越计数/距离并按类概率聚合。

**📊 数据集**

实验使用CIFAR‑10、CIFAR‑100、ImageNet‑200/1K，并在ViT‑B/16与Swin‑T等Transformer上验证。

**📈 对比分析**

与MSP、ODIN、VIM、SHE、GEN、ASH、SCALE以及MDS、RMDS、KNN等基线对比，AggBBAS在大多数数据集上获得最高或竞争性最好的AUROC/FPR95，尤其在Near‑OOD上表现突出。

**⚠️ 局限性**

局限在于仍需手工设定监测变量、聚类特征及盒子数量，对小样本或高维特征的鲁棒性待验证，且目前仅针对分类任务；进一步改进如自适应权重、无监督扩展等待探索。

---

## 79. Three Creates All: You Only Sample 3 Steps

**arXiv ID:** 2603.22375 | [PDF](https://arxiv.org/pdf/2603.22375v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 80. Scaling Attention via Feature Sparsity

**arXiv ID:** 2603.22300 | [PDF](https://arxiv.org/pdf/2603.22300v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 81. A Theoretical Framework for Energy-Aware Gradient Pruning in Federated Learning

**arXiv ID:** 2603.22465 | [PDF](https://arxiv.org/pdf/2603.22465v1)

**作者:** Emmanouil M. Athanasakos `[一作]` `[通讯]` (National and Kapodistrian University of Athens), Emmanouil M. Athanasakos (National and Kapodistrian University of Athens)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种能源感知的梯度稀疏化方法（CWMP），在联邦学习中根据梯度幅值与参数能耗比值来挑选更新，从而在有限能耗下实现更高的学习效率。

**💡 创新点**

核心创新是将梯度稀疏化视为能耗受限的投影问题，设计了基于效率密度（|g|/c）排序的贪心选择规则，并证明其在能耗约束下是最优的。

**🔧 技术方法**

技术主要包括：能耗测度模型（参数成本向量c）、基于效率得分的Top‑k筛选、理论证明（0/1背包问题的贪心解）以及对比实验实现。

**📊 数据集**

使用非IID CIFAR‑10 数据集，训练 ResNet‑18 模型，在模拟的 10 个客户端上进行实验。

**📈 对比分析**

与传统 Top‑K 稀疏化做对比，实验在不同稀疏率（1%、5%、10%、20%）下绘制性能‑能耗 Pareto 前沿。CWMP 在低能耗极限（1%）下实现更高准确率且能耗更低；在 20% 时仍保持更好的准确率并避免 Top‑K 的轻微过拟合。

**⚠️ 局限性**

局限性在于：只考虑了单轮梯度能耗，未结合错误补偿或动态能耗预算；实验仅限于 ResNet‑18+CIFAR‑10，需在更大模型和多样化任务上进一步验证。

---

## 82. Errors in AI-Assisted Retrieval of Medical Literature: A Comparative Study

**arXiv ID:** 2603.22344 | [PDF](https://arxiv.org/pdf/2603.22344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 83. When Visuals Aren't the Problem: Evaluating Vision-Language Models on Misleading Data Visualizations

**arXiv ID:** 2603.22368 | [PDF](https://arxiv.org/pdf/2603.22368v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 84. From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents

**arXiv ID:** 2603.22386 | [PDF](https://arxiv.org/pdf/2603.22386v1)

**作者:** Ling Yue `[一作]`, Shaowu Pan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于大型语言模型的工作流（Agentic Computation Graph, ACG）优化方法进行系统综述，并提出统一的分类框架和评估准则。

**💡 创新点**

首次把工作流结构本身作为优化目标，区分静态与动态、预执行与运行时编辑三种结构确定方式，并提出“Graph Determination Time (GDT)”与“Graph Plasticity Mode (GPM)”两维度来细化方法；同时给出最小报告协议，强调结构、成本与鲁棒性三维评估。

**🔧 技术方法**

综述的技术包括：离线搜索（AFlow、ADAS、Evolutionary Generation）、节点级提示/参数优化（DSPy、OPRO、EvoPrompt 等）、结构搜索与编辑（AutoAgents、DyFlow、AgentConductor、MetaGen 等）、强化学习与监督学习生成工作流、以及多种验证器与偏好/文本反馈机制。

**📊 数据集**

使用了多种工作流生成与评测基准：WorFBench、FlowBench、ComfyUI-R1、GAIA、SWE‑bench、MCP‑bench 等；同时收集了 77 篇核心研究与 27 个评测资源，涵盖代码生成、工具调用、软件操作、硬件设计等场景。

**📈 对比分析**

比较方法通过任务准确率（如 pass@k、准确率）、执行成本（token、LLM 调用、工具调用、延迟、费用）以及图层属性（节点数、深度、宽度、通信量、编辑次数）进行多维度评估；文中总结不同方法在任务精度与成本权衡、结构复杂度与鲁棒性方面的优劣，指出静态搜索在稳定环境下优于动态生成，而动态编辑在高不确定性环境中表现突出。

**⚠️ 局限性**

局限性包括：1）缺乏统一的结构信用分配方法，难以归因到具体结构改动；2）表达力与可验证性冲突，当前 IR 既不够强大也不易验证；3）对工具/环境漂移的持续适配研究不足；4）评测基准的可重复性与数据泄漏问题；5）整体缺乏理论指导，无法系统预测何时需要动态生成、何时足以使用静态模板。

---

## 85. Working towards a dialectical understanding of the political ideology within technological projects

**arXiv ID:** 2603.22436 | [PDF](https://arxiv.org/pdf/2603.22436v1)

**作者:** Frederick Reiber `[一作]` `[通讯]` (Boston University), Frederick Reiber (Boston University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种以辩证思维为核心的框架，用来分析技术项目中的政治意识形态，并通过对 Mechanism Design for Social Good (MD4SG) 议程的案例研究进行说明

**💡 创新点**

将政治承诺视为技术工作的内在属性，并以三问式（价值、约束、行动）方法揭示意识形态形成的辩证过程，填补了技术项目政治性研究的空白

**🔧 技术方法**

主要采用理论分析与案例研究的方法，未使用实验或工程实现技术

**📊 数据集**

以 MD4SG 研究议程的文献为案例，不依赖公开数据集

**📈 对比分析**

本文不涉及实验对比或性能评估，仅通过案例阐释框架的适用性和解释力

**⚠️ 局限性**

局限性包括缺乏大规模实证检验、框架的通用性和可操作性尚待进一步验证

---

## 86. Unveiling the Mechanism of Continuous Representation Full-Waveform Inversion: A Wave Based Neural Tangent Kernel Framework

**arXiv ID:** 2603.22362 | [PDF](https://arxiv.org/pdf/2603.22362v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 87. Q-Tacit: Image Quality Assessment via Latent Visual Reasoning

**arXiv ID:** 2603.22641 | [PDF](https://arxiv.org/pdf/2603.22641v1)

**作者:** Yuxuan Jiang `[一作]` (University of Bristol), David Bull `[通讯]` (University of Bristol)

**通讯引用:** 13085 | [OpenAlex ID](https://openalex.org/A5048009053)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个基于视觉语言模型的无参考图像质量评估框架 Q-Tacit，利用隐式视觉推理在连续潜在空间中进行质量判断。

**💡 创新点**

创新点在于：①将推理过程从传统文本空间迁移到潜在视觉空间，使用自回归隐藏状态序列进行“视觉思考”；②两阶段训练策略——SFT 注入 ROI 视觉先验，GRPO 通过强化学习校准推理轨迹；③设计了专门的奖励函数和格式约束，提升推理稳定性与解释性。

**🔧 技术方法**

使用技术包括：视觉语言模型 Qwen2.5-VL-7B-Instruct、视觉编码器+投影器、Latent Visual Reasoning（LVR）+ latent slot tokens、Latent Group Relative Policy Optimization（GRPO）强化学习、ROI‑based supervision、Gaussian‑shaped质量奖励。

**📊 数据集**

训练与评估数据集：Visual CoT、KADIS‑700K、KonIQ‑10K、SPA‑Q、LIVE‑Wild、KADID、CSIQ、PIPAL、AGIQA 等多种真实、合成、模型处理及 AI 生成图像质量数据。

**📈 对比分析**

与现有 NR‑IQA 方法（NIQE、BRISQUE、NIMA、MUSIQ、CLIP‑IQA+、ManIQA、Compare2Score、Q‑Align、DeQA‑Score、VisualQuality‑R1、Q‑Insight 等）对比，Q‑Tacit 在 7 个测试集上的 PLCC/SRCC 均达到或超过最优模型，尤其在跨数据集泛化表现最佳。

**⚠️ 局限性**

局限性包括：①需要额外的 ROI 标注或自动生成的空间先验；②潜在推理仍缺乏直观可解释性；③训练过程涉及两阶段与 RL，计算成本相对较高；④在极端 OOD 场景下，潜在空间的泛化能力仍有提升空间。

---

## 88. Generalized multi-object classification and tracking with sparse feature resonator networks

**arXiv ID:** 2603.22539 | [PDF](https://arxiv.org/pdf/2603.22539v1)

**作者:** Lazar Supic `[一作]` (UC Berkeley), E. Paxon Frady `[通讯]` (UC Berkeley)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aaccfe5c-6b26-4208-b23c-35331481e142` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于生成模型和共振器网络的稀疏特征场景理解系统，能够同时分离图像中的形状、位置和颜色因子，并实现对未见手写数字的分类与多目标跟踪。

**💡 创新点**

创新点在于将稀疏特征基学习到共振器网络的形状模块，利用共振器网络的自适应搜索同时获得不变性和可变性信息；实现无数据增强的全局位置不变分类与像素级精度的多目标跟踪。

**🔧 技术方法**

采用复数向量空间（VSA）编码的生成式模型、共振器网络（含局部竞争算法 LCA）、稀疏字典学习（PCA+ICA 或 sporco）以及线性分类器和 LeNet5 进行实验。

**📊 数据集**

使用 MNIST 数据集（训练 60,000 张、测试 10,000 张），将手写数字放入 56x56–64x64 大图中生成合成场景。

**📈 对比分析**

通过将共振器网络输出的去颜色、去位置图像输入分类器，Linear II 获得 76.3% 的准确率，LeNet5 II 获得 80.4%；相较于直接分类原始图像（Linear I 60.2%，LeNet5 I 66.7%）显著提升；多目标跟踪平均误差 1–2 像素，95% 误差小于 5 像素。

**⚠️ 局限性**

局限性包括：共振器网络收敛后图像中心仍可能有微小偏移，影响分类与跟踪；重建图像不完全，导致信息丢失；高维向量空间计算成本高；对形状重建的精度仍有提升空间。

---

## 89. Emergency Preemption Without Online Exploration: A Decision Transformer Approach

**arXiv ID:** 2603.22315 | [PDF](https://arxiv.org/pdf/2603.22315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 90. COMPASS-Hedge: Learning Safely Without Knowing the World

**arXiv ID:** 2603.22348 | [PDF](https://arxiv.org/pdf/2603.22348v1)

**作者:** Ting Hu `[一作]` (University of Wisconsin--Madison), Manolis Vlatakis `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了COMPASS‑Hedge，一种在全信息专家学习中同时兼顾基准安全、最优鲁棒性和随机适应性的在线学习算法。

**💡 创新点**

首次实现三者统一（“最佳三世界”）且参数无关，利用自适应伪损失估计、阶段化加倍与比较器混合机制实现安全与攻击者无关。

**🔧 技术方法**

自适应几何加倍、伪损失与期望损失桥接、基准混合权重自调节、阶段/相位控制的安全阈值。

**📊 数据集**

金融领域S&P 500历史回报（2015‑2024）经映射到[0,1]损失，用于验证安全基准与专家对比。

**📈 对比分析**

与传统Hedge对比：在oracle基准下几乎零伪损失；在统一基准下保持O(1)相对基准损失，避免高方差波动；在随机/对抗环境下取得O(√T)和O(1/Δ)伪损失。

**⚠️ 局限性**

存在多余的log²T因子，安全保底的代价未定；仅适用于单专家/全信息场景，需扩展到更一般的结构化动作空间或强化学习。

---

## 91. Trajectory Generation for Underactuated Soft Robot Manipulators using Discrete Elastic Rod Dynamics

**arXiv ID:** 2603.22604 | [PDF](https://arxiv.org/pdf/2603.22604v1)

**作者:** Beibei Liu `[一作]` (Boston University), Andrew P. Sabelhaus `[通讯]` (Boston University)

**通讯引用:** 1004 | [OpenAlex ID](https://openalex.org/A5042202916)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

重新表述软体机器人Discreted Elastic Rod（DER）动力学为控制仿射形式，并显式加入驱动器映射，生成动态可行的轨迹并在气动软体机器人上验证。

**💡 创新点**

将DER动力学与驱动器映射显式化，得到可直接计算控制输入的控制仿射模型，解决传统DER缺乏对实际硬件可操作性的限制。

**🔧 技术方法**

使用DDG‑based Discrete Elastic Rod建模、控制仿射轨迹生成算法、计算机视觉姿态估计、低层PID阀控等技术。

**📊 数据集**

采用实验收集的软体机器人姿态（AprilTags观测）和压力输入数据，未使用公开数据集。

**📈 对比分析**

与常数曲率（PCC）模型在三种运动情形（异步弯曲、同步同向、同步反向）下对比，误差均值、最大值和MSE均明显低于PCC，性能提升约60%。

**⚠️ 局限性**

仅在近似PCC假设的软体机器人上验证，未考虑外部载荷或接触交互，对非PCC 情况的推广未知，且需手动校准 B(q) 结构。

---

## 92. Energy-Aware Collaborative Exploration for a UAV-UGV Team

**arXiv ID:** 2603.22507 | [PDF](https://arxiv.org/pdf/2603.22507v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 93. Generating and Evaluating Sustainable Procurement Criteria for the Swiss Public Sector using In-Context Prompting with Large Language Models

**arXiv ID:** 2603.22513 | [PDF](https://arxiv.org/pdf/2603.22513v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 94. FAAR: Format-Aware Adaptive Rounding for NVFP4

**arXiv ID:** 2603.22370 | [PDF](https://arxiv.org/pdf/2603.22370v1)

**作者:** Hanglin Li `[一作]` (Li Auto Inc), Kun Zhan `[通讯]` (Li Auto Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对NVFP4格式的可学习量化方法FAAR，并结合两阶段格式对齐(2FA)实现LLM的低比特(4位)量化；

**💡 创新点**

通过在非均匀数值网格上引入可学习的软四舍五入变量，直接优化量化误差，显著改进传统RTN和AdaRound等基于均匀网格的策略；

**🔧 技术方法**

采用可微软舍入函数、温度调度、层级/全模型对齐损失、KL与MSE一致化以及硬化后部署；

**📊 数据集**

在Llama3与Qwen3系列模型上使用WikiText-2、C4、BoolQ、ARC-Easy/Challenge、HellaSwag等数据集进行评估；

**📈 对比分析**

与RTN、GPTQ、MR-GPTQ、4/6及GPTQ+4/6等PTQ基线对比，FAAR+2FA在WikiText-2、C4上分别把perplexity从14.28降至12.60（Llama3-1B），并在所有下游任务中平均提升5‑6个百分点，近似BF16性能；

**⚠️ 局限性**

仅在黑莓硬件平台和两大模型系列上验证，缺乏对更大模型、不同任务或其他极低精度格式的通用性分析，且仍需GPU资源进行2FA微调。

---

## 95. Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length

**arXiv ID:** 2603.22608 | [PDF](https://arxiv.org/pdf/2603.22608v1)

**作者:** Jingxuan Chen `[一作]` (Cardiff University), Jose Camacho-Collados `[通讯]` (Cardiff University)

**通讯引用:** 3583 | [OpenAlex ID](https://openalex.org/A5086289154)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文通过对多实例处理（MIP）任务的系统评估，探究LLM在面对多实例输入时的性能衰减模式。

**💡 创新点**

创新点在于首次将实例数与上下文长度在MIP场景下进行细粒度对比，证明实例数量是导致性能下降的主因，并提供了全面的错误类型细分与因果分析。

**🔧 技术方法**

主要技术包括：SIP-先验过滤确保单实例任务易解；MIP采样与随机种子；模型输出解析与错误分类（错误答案/无效输出、关键错误/单体错误/聚合错误/组合错误）；人工噪声注入与位置变换实验；以及Spearman相关性分析。

**📊 数据集**

使用八类任务数据集（算术、新闻分类、语言识别、NER、奇偶判断、情感分析、单词计数、词义辨识），每类约2500条实例，经过SIP成功率筛选后得到训练集与评估集。

**📈 对比分析**

对比了16款LLM（9开放权重、7闭源），在不同实例规模（2~2000）下测得成功率与无效率。整体表现随实例数上升呈现先缓慢下降后剧烈崩溃的趋势；最优模型为GPT‑5、Gemini 3.1 Pro、Grok 4、gpt‑oss‑120b及Qwen3‑Thinking，轻量级模型如Llama 4 Maverick和Grok 4 Fast在低无效率下表现较好。

**⚠️ 局限性**

局限性包括：未提供任何MIP可靠性提升方案；仅评估精确聚合任务，未覆盖近似聚合；使用统一prompt模板，缺乏对prompt多样性的考察；未测试极端长上下文；实验仅限英语数据；未进行模型内部解释性或训练时干预分析。

---

## 96. OsteoFlow: Lyapunov-Guided Flow Distillation for Predicting Bone Remodeling after Mandibular Reconstruction

**arXiv ID:** 2603.22421 | [PDF](https://arxiv.org/pdf/2603.22421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. Founder effects shape the evolutionary dynamics of multimodality in open LLM families

**arXiv ID:** 2603.22287 | [PDF](https://arxiv.org/pdf/2603.22287v1)

**作者:** Manuel Cebrian `[一作]` `[通讯]` (Spanish National Research Council), Manuel Cebrian (Spanish National Research Council)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

分析 Hugging Face 开源 LLM 生态系统中多模态（尤其是视觉‑文本）模型的出现和传播，利用模型卡和 lineage 记录对跨模态任务比例、父子转移率以及创始人效应进行量化。

**💡 创新点**

发现多模态能力主要通过少数 VLM “创始人”进入家族，随后在同一族谱内快速复制，文本→VLM 的直接转移率极低，形成 punctuated 的扩散模式。

**🔧 技术方法**

采用生态系统级别的统计分析、父子关系条件转移概率计算、Wilson 置信区间估计、族谱亲缘度量（有效父节点数、top‑n 份额）等技术。

**📊 数据集**

使用 ModelBiome AI Ecosystem 数据集（7 月 2025 快照），包含约 1.86 M 个模型条目和 3.02 M 条有向 lineage 边。

**📈 对比分析**

通过对 fine‑tune、merge、adapter、quantize 等关系类型的父子任务转移率进行时间分辨率分析，比较 VLM→VLM 的高保持率（≈65%）与 text→VLM 的低转移率（≈0.2%），并给出 95% Wilson 置信区间；该研究未使用标准 benchmark 评估模型能力，而是侧重统计指标，表明族内传播显著高于跨族传播。

**⚠️ 局限性**

局限包括 lineage 记录不完整导致创始人比例可能被高估、任务标签和模型卡噪声、时间戳可靠性仅自 2022 年可用、家族识别基于名称可能误判、未衡量实际推理性能、未捕捉未记录的跨模态转换。

---

## 98. Learning to Trust: How Humans Mentally Recalibrate AI Confidence Signals

**arXiv ID:** 2603.22634 | [PDF](https://arxiv.org/pdf/2603.22634v1)

**作者:** ZhaoBin Li `[一作]` (University of California, Irvine), Mark Steyvers `[通讯]` (University of California, Irvine)

**通讯引用:** 21497 | [OpenAlex ID](https://openalex.org/A5051768325)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过在四种人工智能置信度失衡情境下进行实验，研究人类如何通过反复经验在心理上重新校准AI置信度信号。

**💡 创新点**

首次提出并验证人类可利用强化学习机制自我校准AI置信度的理论，并揭示在逆置信度场景中学习的极限。

**🔧 技术方法**

采用线性对数几率（LLO）变换结合Rescorla‑Wagner学习规则的计算模型，并用贝叶斯多层推断估计参数。

**📊 数据集**

使用自生成的信号检测模型产生的置信度分布（标准、过度、欠度、逆置信度）作为实验刺激，参与者完成50轮实验。

**📈 对比分析**

与多层逻辑/概率回归对比，模型在实验中解释约75%的决策准确率，显示人类学习显著提升，模型性能优于基线。

**⚠️ 局限性**

仅在单次50次即时反馈的实验中验证，未考察长期记忆、延迟或无反馈情境以及独立预测的可转移性，现实协作环境的适用性有限。

---

## 99. Hybrid Associative Memories

**arXiv ID:** 2603.22325 | [PDF](https://arxiv.org/pdf/2603.22325v1)

**作者:** Leon Lufkin `[一作]` (Zyphra), Kamesh Krishnamurthy `[通讯]` (Zyphra)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 Hybrid Associative Memory (HAM) 层，结合 RNN 与 KV 缓存，只在 RNN 预测困难时写入 KV，形成数据依赖的自适应缓存；

**💡 创新点**

通过预测误差或可学习路由实现 KV 缓存的精细增长控制，允许层级和全局可调节的 KV 使用率，从全自注意到全 RNN 形成连续性能调节；

**🔧 技术方法**

采用 RNN（如 GDN/DeltaNet）、自注意力、KV 缓存、预测误差路由、可学习门控与 RMSNorm、Flash Linear Attention、FlexAttention、可学习阈值等技术；

**📊 数据集**

在 Long Data Collections、Institutional Books、NIAH、RULER 等数据集上训练和评估，并使用 HellaSwag、WikiText、LAMBADA 等常识推理基准；

**📈 对比分析**

与 800M 参数的 Transformer、GDN、GDN‑GSA 进行对比，使用标准零样本推理和长上下文任务评估，结果显示在 50% KV 使用率下性能与 Transformer 相当甚至更优，KV 使用率与性能呈平滑关系；

**⚠️ 局限性**

实现复杂，需额外路由与门控；KV 访问可能导致 TLB 访问延迟；路由学习对长序列仍需更多训练；迁移到现有大模型时不易无缝集成。

---

## 100. Mitigating Premature Discretization with Progressive Quantization for Robust Vector Tokenization

**arXiv ID:** 2603.22304 | [PDF](https://arxiv.org/pdf/2603.22304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 101. STRIATUM-CTF: A Protocol-Driven Agentic Framework for General-Purpose CTF Solving

**arXiv ID:** 2603.22577 | [PDF](https://arxiv.org/pdf/2603.22577v1)

**作者:** James Hugglestone `[一作]` (Florida State University), Xiuwen Liu `[通讯]` (Florida State University)

**通讯引用:** 8365 | [OpenAlex ID](https://openalex.org/A5102867647)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并验证了基于Model Context Protocol的STRIATUM-CTF框架，用LLM驱动的代理自动完成CTF攻击任务。

**💡 创新点**

引入MCP实现神经符号化代理架构，标准化工具接口并在推理时强制类型校验，显著减少模型幻觉并实现长序列状态保持。

**🔧 技术方法**

使用Claude Sonnet 4.5作为推理层，MCP协议层做结构化函数调用校验，结合Angr、Ghidra、GDB等容器化安全工具，并采用四步循环计划-执行-解析-迭代。

**📊 数据集**

采用15个CTF挑战（picoCTF、angr教程、大学内部等）作为测试集，并在2015年大学举办的实时CTF竞赛中进行验证。

**📈 对比分析**

通过消融实验与人类参赛队伍对比，最佳配置成功率达86.7%；在现场竞赛中以215分夺冠，领先人类队伍约15分，显示出显著的速度与准确性优势。

**⚠️ 局限性**

存在token成本与上下文管理开销、动作空间受限、长序列推理效率下降等限制，未来需改进层次化规划与上下文压缩。

---

## 102. FullCircle: Effortless 3D Reconstruction from Casual 360$^\circ$ Captures

**arXiv ID:** 2603.22572 | [PDF](https://arxiv.org/pdf/2603.22572v1)

**作者:** Yalda Foroutan `[一作]` (Simon Fraser University), Andrea Tagliasacchi `[通讯]` (Simon Fraser University)

**通讯引用:** 7777 | [OpenAlex ID](https://openalex.org/A5037094498)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套从随手录制的双鱼眼 360° 图像中自动去除摄像机操作者并实现高质量 3D 重建的完整流程，能够直接使用原始鱼眼图像训练 3D 高斯云模型；

**💡 创新点**

核心创新在于利用预训练的 SAMv2 与 YOLOv8 结合的两阶段遮罩策略，精准定位并遮蔽持续出现的摄像机操作者，消除光照不一致和动态干扰；

**🔧 技术方法**

结合 3D Gaussian Ray Tracing (3DGRT) 的鱼眼渲染能力、SAMv2/YOLOv8 的人类检测与分割、COLMAP 视觉 SLAM 进行相机位姿估计与遮罩传播；

**📊 数据集**

收集并公开了 9 个多样化场景的原始双鱼眼视频数据集，并为每个场景提供了无干扰的三脚架测试集，用于定量评估；

**📈 对比分析**

与传统视角相机、基于解畸变的 360° 方法、SpotLessSplats、NOTG、OmniLocalRF 等多种基线进行对比，实验显示本方法在 PSNR、SSIM、LPIPS 等指标上均优于基线，且在离散视角下保持更高的重建质量；

**⚠️ 局限性**

限制包括对固定曝光的依赖，无法处理剧烈亮度变化或严重运动模糊；在摄像机操作者长时间静止或存在静态干扰物时，遮罩效果仍有局限。

---

## 103. Cloud-Edge Collaborative Large Models for Robust Photovoltaic Power Forecasting

**arXiv ID:** 2603.22343 | [PDF](https://arxiv.org/pdf/2603.22343v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 104. Stability-Preserving Online Adaptation of Neural Closed-loop Maps

**arXiv ID:** 2603.22469 | [PDF](https://arxiv.org/pdf/2603.22469v1)

**作者:** Danilo Saccani `[一作]` (Ecole Polytechnique Fédérale de Lausanne), Giancarlo Ferrari-Trecate `[通讯]` (Ecole Polytechnique Fédérale de Lausanne)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一种在线可更新的非线性神经网络控制器的稳定性保障机制，并给出了时间表和状态触发两种可实现的更新策略。

**💡 创新点**

创新点在于通过基于ℓ_p增益的更新条件，证明在切换控制器时仍保持闭环ℓ_p稳定；将性能优化与稳定性分离，提供可实施的触发规则。

**🔧 技术方法**

采用系统级综合（SLS）/IMC架构、Recurrent Equilibrium Network（REN）作为控制器模型，并利用有限增益、小增益理论、LMI增益上界和BPTT/Adam优化实现在线更新。

**📊 数据集**

实验使用仿真机器人任务，包括山谷协作导航和动态障碍跟踪，使用随机高斯噪声和冲击扰动生成扰动序列；未使用公开数据集。

**📈 对比分析**

通过与离线全周期训练的控制器以及递归滚动规划（RHO）做比较，在两种实验中平均成本降低约35–40%，并表现出更小的波动性，说明性能显著提升。

**⚠️ 局限性**

局限性包括需要预估或假设系统增益上界，触发规则可能过于保守；仅在仿真环境验证，未充分评估对模型误差或极端扰动的鲁棒性。

---

## 105. Practitioner Voices Summit: How Teachers Evaluate AI Tools through Deliberative Sensemaking

**arXiv ID:** 2603.22588 | [PDF](https://arxiv.org/pdf/2603.22588v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 106. Personalized Federated Sequential Recommender

**arXiv ID:** 2603.22349 | [PDF](https://arxiv.org/pdf/2603.22349v1)

**作者:** Yicheng Di `[一作]` `[通讯]`, Yicheng Di

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了PFSR——一种细粒度全球建模的联邦序列推荐框架，解决了传统方法在长序列建模中的二次复杂度与个性化不足的问题。

**💡 创新点**

创新点在于：①引入关联Mamba块，从双向全局视角捕捉用户画像；②通过Fisher信息构造的变异响应机制实现参数的细粒度自适应；③设计动态幅值损失约束本地更新，保留更多个性化信息。

**🔧 技术方法**

使用的技术包括：Mamba空间状态模型、Fisher信息自适应遮罩、对比学习样本对、动态幅值损失以及联邦学习框架。

**📊 数据集**

实验使用的真实世界数据集为Beauty、Yelp和Gowalla。

**📈 对比分析**

采用留一法与SASRec、CoSeRec、SURGE、IOCRec、DuoRec、ContraRec、LRURec、EchoMamba4Rec等基线在HR@5/10和NDCG@5/10上比较，PFSR在所有数据集上均取得约5–9%的平均提升，表现优于所有基线。

**⚠️ 局限性**

局限性包括：①对联邦通信成本和同步开销的依赖；②阈值λ对Fisher遮罩的敏感性；③对极端稀疏或高动态场景的适应性未充分验证；④模型规模与计算资源需求相对较高。

---

## 107. From Instructions to Assistance: a Dataset Aligning Instruction Manuals with Assembly Videos for Evaluating Multimodal LLMs

**arXiv ID:** 2603.22321 | [PDF](https://arxiv.org/pdf/2603.22321v1)

**作者:** Federico Toschi `[一作]` (Politecnico di Milano), Mark James Carman `[通讯]` (Politecnico di Milano)

**通讯引用:** 2907 | [OpenAlex ID](https://openalex.org/A5069487278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了多模态大型语言模型在家具组装技术辅助中的能力，提出并评估了M2AD数据集。

**💡 创新点**

创新点是构建了符合真实场景、从YouTube视频到IKEA手册的精细步骤对齐数据集，并将最小化注释与多模态推理相结合。

**🔧 技术方法**

使用开源多模态LLM如LLaVa-Video、Qwen2-VL、MolMo等，采用零样本提示进行步骤完成检测、步骤匹配与步骤识别三项实验。

**📊 数据集**

使用了M2AD数据集，包含53件IKEA家具的录制视频与对应手册页面，约1228个步骤注释。

**📈 对比分析**

通过对比多模态模型在三项任务上的准确率、精确率等指标，LLaVa-Video与Qwen2-VL在步骤完成与匹配任务上表现最好，MolMo在步骤识别任务上突出，整体性能仍低于人类并受限于模型大小。

**⚠️ 局限性**

主要限制是模型在多图像与交织文本推理、视觉与文本对齐方面受限，硬件资源限制导致上下文窗口缩小，导致准确率偏低。

---

## 108. Research on Individual Trait Clustering and Development Pathway Adaptation Based on the K-means Algorithm

**arXiv ID:** 2603.22302 | [PDF](https://arxiv.org/pdf/2603.22302v1)

**作者:** Qianru Wei `[一作]` (Shaanxi University of Science and Technology), Jinming Yang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 10649 | [OpenAlex ID](https://openalex.org/A5100599959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用 K-means 聚类对 3000 名学生的 CET-4 分数、GPA、人格类型和学生干部经验进行分组，并为每组提供职业方向建议。

**💡 创新点**

将传统人岗匹配理论与无监督学习相结合，实现多维特征的精准职业匹配，首次用聚类方法直接输出可操作的就业建议。

**🔧 技术方法**

主要技术包括数据归一化、K-means 聚类、主成分分析（PCA）可视化和内部评价指标（Silhouette 系数）。

**📊 数据集**

数据集来自 S 大学西部校区的 3000 名本科毕业生，包含 CET-4、GPA、MBTI 人格和学生干部经历四个维度。

**📈 对比分析**

采用 Silhouette 系数评估聚类质量，平均值为 0.684，显示聚类结果相对清晰；未与其他聚类算法或监督模型进行直接对比，说明当前方法已具备良好可行性。

**⚠️ 局限性**

局限性包括样本规模相对有限、仅考虑四个特征、未纳入外部经济或行业需求等因素，可能影响模型的泛化能力。

---

## 109. Enhancing AI-Based Tropical Cyclone Track and Intensity Forecasting via Systematic Bias Correction

**arXiv ID:** 2603.22314 | [PDF](https://arxiv.org/pdf/2603.22314v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 110. MinerU-Diffusion: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding

**arXiv ID:** 2603.22458 | [PDF](https://arxiv.org/pdf/2603.22458v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 111. A Vision Language Model for Generating Procedural Plant Architecture Representations from Simulated Images

**arXiv ID:** 2603.22622 | [PDF](https://arxiv.org/pdf/2603.22622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 112. CAPITU: A Benchmark for Evaluating Instruction-Following in Brazilian Portuguese with Literary Context

**arXiv ID:** 2603.22576 | [PDF](https://arxiv.org/pdf/2603.22576v1)

**作者:** Giovana Kerche Bonás `[一作]` (Maritaca AI), Rodrigo Nogueira `[通讯]` (Maritaca AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CAPITU，一个专门用于评估巴西葡萄牙语LLM遵循指令能力的基准，包含59种可自动验证的指令类型，并以八部经典文学作品为情境背景。

**💡 创新点**

创新点在于：①针对葡萄牙语的本土语言学约束（如后缀-ando/-indo/-inho等）与结构要求；②将文学语境与可自动验证的指令相结合；③通过多轮对话检验约束持续性。

**🔧 技术方法**

使用基于正则表达式、字符串计数和词表匹配的规则驱动验证器；在模型评估中采用严格与宽松两种准确率；还引入LLM（GPT‑4o‑mini）进行可选的一致性打分。

**📊 数据集**

构建了500条评测实例，包含200条单轮、100条三轮对话，共使用八部巴西文学经典（如《Dom Casmurro》《Grande Sertão》）作为文本背景。

**📈 对比分析**

对18种模型（包括OpenAI GPT‑5系列、Gemini‑3、Claude‑4.5、Alibaba Qwen3、Maritaca AI Sabiá等）进行比较，GPT‑5.2（思考模式）在严格准确率上达98.5%；Sabiá‑4以87.0%准确率和$0.13/生成成本实现最佳性价比；多轮对话准确率在60%–96%之间，显示不同模型在约束持续性上的差异。

**⚠️ 局限性**

局限性包括：①可选一致性评分依赖GPT‑4o‑mini，缺乏人工验证；②形态学约束仅通过后缀匹配，无法区分语义合法性；③未覆盖所有葡萄牙语方言与口语表达，且部分评测依赖单一文学语料。

---

## 113. On sampling diluted Spin-Glasses with unbounded interactions

**arXiv ID:** 2603.22432 | [PDF](https://arxiv.org/pdf/2603.22432v1)

**作者:** Charilaos Efthymiou `[一作]` (University of Warwick), Kostas Zampetakis `[通讯]` (TU Dortmund)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文研究了二维自旋模型（即2-Spin模型，又称Viana-Bray模型）在稀疏随机图G(n, d/n)上的Glauber动力学的混合时间。作者给出了在β不超过临界温度β_c(d)时，随机图上Glauber动力学的上界为O(n^{1+24/√log d})，即证明了在此温度区间内系统混合时间为多项式级别，且随d增大得到更快混合。

**💡 创新点**

创新点在于提出了一种新的“ε-块分割”技术，即将图划分为若干个小块（单一顶点或树/单环结构），并通过对块内自旋相互作用强度和路径权重进行细致的加权，建立了对协方差矩阵的高概率上界。该方法既能兼顾随机图的稀疏结构，又能处理高耦合强度导致的重块效应，从而实现了在临界温度附近的快速混合证明。

**🔧 技术方法**

主要技术包括：
- 随机块划分（ε-块分割）
- 加权路径（权重函数W、T、T_∂等）
- 协方差矩阵与影响矩阵的路径展开式
- 生成树（Galton-Watson树）与随机行走的比较
- 大数律与Chernoff/Bernstein尾部估计
- 变分不等式与马尔科夫链的放松时间与改进对数Sobolev常数的关联
- 变形矩阵正定性与谱半径估计

**📊 数据集**

使用的数据集是理论上的随机图G(n, d/n)与随机高斯耦合J_e（i.i.d.标准正态分布），即完全由理论随机过程构造的合成数据。

**📈 对比分析**

比较方法：将得到的混合时间上界与已有的随机图上2-Spin模型混合时间的下界（如通过马尔科夫链退火或变分法得到的经典结果）进行对比。结果显示，作者的上界在β≤β_c(d)时相当于O(n^{1+O(1/√log d}))，明显优于先前的O(n^{1+ε})（ε>0）或仅适用于更严格条件的结果，证明了在更宽的参数范围内仍保持多项式级别混合。

**⚠️ 局限性**

局限性：
- 仅适用于β不超过临界温度β_c(d)的“热”或“非相变”区间；在临界点或低温区间的混合时间仍未知。
- 方法高度依赖于随机图的“树状”性质和高耦合强度的稀疏性，可能难以直接推广到密集图或具有高聚类系数的网络。
- 证明中使用了大量概率上界与矩阵正定性技巧，对随机行走和协方差矩阵的高阶矩估计要求d足够大，导致对d的取值有严格限制。

---

## 114. mmFHE: mmWave Sensing with End-to-End Fully Homomorphic Encryption

**arXiv ID:** 2603.22437 | [PDF](https://arxiv.org/pdf/2603.22437v1)

**作者:** Tanvir Ahmed `[一作]` (Cornell Tech), Rajalakshmi Nandakumar `[通讯]` (Cornell Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

实现了端到端的mmWave雷达传感与机器学习推理的全同态加密系统。

**💡 创新点**

首次将完整的雷达DSP和ML推理流程在云端以数据不可见方式执行，提出七个可组合的FHE友好DSP内核并提供输入隐私与数据不可观测性证明。

**🔧 技术方法**

采用CKKS同态加密，构造能源积分、软幂注意力、DFT矩阵乘、软I/Q提取、FIR滤波、陷波掩码、Taylor级数相位提取等内核，并使用GPU加速的OpenFHE/FIDESlib。

**📊 数据集**

在三个公开mmWave数据集上评估：儿童心率呼吸记录（270个），姿态/躯干姿势（110个），60GHz手势识别（600个）等。

**📈 对比分析**

与无加密的原生DSP+ML基线对比，心率/呼吸误差<10^-3 bpm，手势识别准确率84.5%（与基线84.7%几乎相同），但加密后GPU延迟分别为10s窗口103s、3s窗口37s，通信量高达4.3GB/窗口。

**⚠️ 局限性**

主要局限是计算与通信开销巨大，实时性差，需更快的FHE库或硬件加速；目前仅支持单目标，未覆盖更深的网络或多目标场景。

---

## 115. Intelligence Inertia: Physical Principles and Applications

**arXiv ID:** 2603.22347 | [PDF](https://arxiv.org/pdf/2603.22347v1)

**作者:** Jipeng Han `[一作]` `[通讯]` (OpenImmortal Technology Co., Ltd.), Jipeng Han (OpenImmortal Technology Co., Ltd.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了“智能惯性”概念，建立其在规则-状态相互作用中的非可交换性与计算成本的非线性关系，进而推导出相对论式的计算成本公式；

**💡 创新点**

创新点在于将信息学与热力学、量子力学中的非可交换性与洛伦兹因子相结合，首次从第一性原理解释智能系统在结构重构时出现的指数级计算与能耗壁垒；

**🔧 技术方法**

采用微观统计模型、算子代数、信息几何与相对论框架相结合的理论推导，并在实验中实现了“惯性感知调度器”作为实际的学习率控制机制；

**📊 数据集**

实验主要使用CIFAR‑10数据集，结合ResNet‑18等标准网络架构进行噪声注入、网络拓扑演化与连续学习任务切换的评估；

**📈 对比分析**

通过与传统Fisher信息、Galilean变换、Hybrid FIM等模型对比，实验一表明相对论式模型在RMSE上显著优于其他模型；实验二展示了在多种网络拓扑下，接近v≈0.5的“黄金轴”可实现最低可达损失；实验三表明惯性调度器在八种主流学习率调度器基础上平均提升30%收敛速度，显著压缩最终损失；

**⚠️ 局限性**

局限性包括对高维梯度稀疏性的处理仍不完善、对异构硬件的适配需要进一步验证、以及在极端噪声或任务冲突情境下的理论边界与鲁棒性尚待深入研究。

---

## 116. A Multi-Task Targeted Learning Framework for Lithium-Ion Battery State-of-Health and Remaining Useful Life

**arXiv ID:** 2603.22323 | [PDF](https://arxiv.org/pdf/2603.22323v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 117. T-MAP: Red-Teaming LLM Agents with Trajectory-aware Evolutionary Search

**arXiv ID:** 2603.22341 | [PDF](https://arxiv.org/pdf/2603.22341v1)

**作者:** Hyomin Lee `[一作]` (KAIST), Sung Ju Hwang `[通讯]` (KAIST)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在多工具调用环境下，利用T-MAP框架对LLM代理进行红队测试

**💡 创新点**

将轨迹感知的MAP‑Elites与跨诊断和工具调用图结合，主动挖掘多步攻击路径

**🔧 技术方法**

交叉诊断、工具调用图、MAP‑Elites进化算法、LLM判定器

**📊 数据集**

五个MCP服务器（CodeExecutor、Slack、Gmail、Playwright、Filesystem）

**📈 对比分析**

对比基线（ZS、MT、IR、SE）在ARR/RR指标上，T-MAP平均ARR 57.8%，显著优于基线

**⚠️ 局限性**

实验仅在沙箱环境，未考虑真实系统的权限检查、用户确认、沙箱化，且依赖攻击者模型的安全弱化

---

## 118. Computational Arbitrage in AI Model Markets

**arXiv ID:** 2603.22404 | [PDF](https://arxiv.org/pdf/2603.22404v1)

**作者:** Ricardo Olmedo `[一作]`, Moritz Hardt `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究并演示了在可验证 AI 模型市场中进行计算套利的可行性，提出了套利策略并量化其经济影响，包括降低消费者价格、降低小模型进入壁垒以及推动模型压缩与市场重组。

**💡 创新点**

创新点在于首次系统化定义“计算套利”，构建基于模型级联与预算分配的套利策略，证明套利可以在多模型市场中更高利润且更普遍，并揭示套利如何激励 distillation，进而改变市场结构。

**🔧 技术方法**

技术包括模型级联（cascade）和预算切换点优化、pass@k 成功率评估、强化学习/优化求解最优切换点、标准监督微调（SFT）进行 distillation，以及多模型成本-性能曲线计算。

**📊 数据集**

使用的数据集主要有 SWE‑bench（GitHub issue 500+ 伴随单元测试）、Lean4 形式化定理证明 benchmark、以及 Qwen‑Smith 60k+ GitHub issue 生成的训练集用于 distillation。

**📈 对比分析**

通过与 GPT‑5 mini、DeepSeek v3.2、Qwen 3 Coder 30B/480B、Claude Sonnet 4.5、mini‑coder 4B 等多模型对比，展示套利利润可达 40%–58%，distillation 随训练规模提升呈对数线性提升；同时比较多模型收益分配，证明套利可在更广阔性能区间实现更高收益。

**⚠️ 局限性**

局限性包括假设验证成本为 0、市场信息完美且无信息成本、仅采用 query‑agnostic、用户‑agnostic 策略，未考虑非可验证任务、价格波动、动态市场信息成本等实际情境。

---

## 119. On the Economic Implications of Diversity in Software Engineering

**arXiv ID:** 2603.22523 | [PDF](https://arxiv.org/pdf/2603.22523v1)

**作者:** Sofia Tapias Montana `[一作]` (University of Calgary), Ronnie de Souza Santos `[通讯]` (University of Calgary)

**通讯引用:** 273 | [OpenAlex ID](https://openalex.org/A5080161379)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

通过访谈10名在不同角色与层级的软件专业人士，收集并分析其对软件团队多样性在成本、收益、上市时间、流程效率与创新等经济层面影响的主观经验与看法。

**💡 创新点**

将多样性视为一种经济资产，首次系统性阐述其在软件项目中如何降低成本、提升收入、加速上市、提升效率，并通过实际项目案例验证其经济价值，从而填补软件工程研究对经济结果关注的空白。

**🔧 技术方法**

采用定性研究方法——访谈设计与半结构化访谈、主题分析（Thematic Analysis）技术对文本进行编码与主题提炼。

**📊 数据集**

数据集为10份访谈录音转录文本，涉及不同技术、管理与战略岗位的实践经验，覆盖产品开发、平台集成、创新与客户导向等场景。

**📈 对比分析**

研究未涉及定量对比或性能指标，而是以主题化、案例驱动的方式呈现多样性对经济效益的感知，无法给出数值化的性能评估。

**⚠️ 局限性**

局限性包括样本量有限且非随机抽样、对访谈者自我报告的主观性依赖、缺乏定量验证、以及在敏感议题上可能产生的披露偏差，导致结果的可推广性和因果推断受限。

---

## 120. CPU Simulation Using Two-Phase Stratified Sampling

**arXiv ID:** 2603.22605 | [PDF](https://arxiv.org/pdf/2603.22605v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 121. Memory Bear AI Memory Science Engine for Multimodal Affective Intelligence: A Technical Report

**arXiv ID:** 2603.22306 | [PDF](https://arxiv.org/pdf/2603.22306v1)

**作者:** Deliang Wen `[一作]`, Yu Wang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Memory Bear AI Memory Science Engine，一种以记忆为核心的多模态情感认知框架。

**💡 创新点**

创新点在于把情感信息建模为结构化记忆单元，支持长期记忆形成、检索与动态融合，实现情感持续推理和鲁棒性提升。

**🔧 技术方法**

使用多模态编码（LLM、Higgs-Audio、VLM）、情感记忆单元 (EMU)、工作记忆聚合、长期记忆合并、检索机制、记忆校准的动态融合以及生命周期管理等技术。

**📊 数据集**

实验数据集包括 IEMOCAP、CMU-MOSEI 和内部 Memory Bear AI 业务真实数据集。

**📈 对比分析**

与传统融合、强神经融合、上下文/记忆感知模型以及内部消融版本对比，整体准确率在 IEMOCAP 78.8%、CMU-MOSEI 66.7%、业务数据 68.4%，鲁棒性保持率高达 92%+。

**⚠️ 局限性**

局限性：公开数据未能充分评估长时情感记忆；评测主要基于内部业务数据，缺乏专门的长期情感连续性数据；工程化实现近似人类记忆，缺乏真实心理学验证。

---

## 122. Investigating Technical Debt Types, Issues, and Solutions in Serverless Computing

**arXiv ID:** 2603.22480 | [PDF](https://arxiv.org/pdf/2603.22480v1)

**作者:** Hasini Sumalee Perera `[一作]` (University of Saskatchewan), Fabio Palomba `[通讯]` (University of Salerno)

**通讯引用:** 9881 | [OpenAlex ID](https://openalex.org/A5033738898)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对78,867条Stack Overflow服务器无关问题进行二分类，量化并细致分析了服务器无关技术债务（TD）的出现频率、类型、子类型及其解决方案，揭示了服务器无关技术债务在实际开发中的普遍性和影响；

**💡 创新点**

创新点在于首次系统地利用社群问答数据挖掘服务器无关技术债务，构建了11类TD类型与49个子类型的完整目录，并提出了针对服务器无关环境的六大特定TD问题（如自动扩展、冷启动、日志收集等），为后续工具和治理策略提供了实证基础；

**🔧 技术方法**

主要技术包括基于BERT的二分类深度学习模型（F1=0.86），传统机器学习模型（LR、RF、SVM、NB）对比；内容分析与开放编码方法用于TD类型与子类型标注；交叉验证与Cohen's κ评估标注一致性；

**📊 数据集**

数据集来源为Stack Exchange公开数据快照（2014‑2025），筛选包含15个服务器无关相关标签的问答，共78,867条问题，其中29,212条被判定为TD相关；

**📈 对比分析**

通过与传统机器学习模型对比，BERT在F1、精确率、召回率上均领先；在TD类型与解决方案的定性分析中，对答复率、已接受答案比例等指标进行统计，展示了TD问题在社区中的解决难度（21%未答复，40%无接受答案）；

**⚠️ 局限性**

主要局限包括：①模型分类仍可能存在误判导致TD比例估计偏差；②仅使用SO数据，无法代表其他技术社区（如GitHub、SE）；③TD类型与子类型标注仍具主观性，尽管Kappa值高；④研究聚焦于标签覆盖范围，未覆盖所有潜在服务器无关标签。

---

## 123. OrgForge-IT: A Verifiable Synthetic Benchmark for LLM-Based Insider Threat Detection

**arXiv ID:** 2603.22499 | [PDF](https://arxiv.org/pdf/2603.22499v1)

**作者:** Jeffrey Flynt `[一作]` `[通讯]` (Independent Researcher), Jeffrey Flynt (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了可验证的、基于模拟的内部威胁检测基准OrgForge-IT，并通过AWS Bedrock的十种LLM模型对其进行评测；

**💡 创新点**

创新点在于：①通过 deterministic simulation engine 保障语义一致性与可验证的 ground truth；②引入四个专为单面/单日检测设计的情境；③提供多格式 SIEM 导出和完整的评测 harness；④发现 triage/verdict 分离、基线 FP 率决定可操作性、受害者归属区分 Tier 的关键性。

**🔧 技术方法**

技术手段包括：Python deterministic simulation engine、LLM 辅助表面文本生成、三阶段检测流水线（baseline calibration、滑窗多信号阈值、关联推理）、对话式 prompt、精确匹配与潜在语义相似度评估。

**📊 数据集**

数据集：OrgForge-IT 51 天、2,904 条可观测记录（96.4% 噪声），包含 3 类威胁主体、8 种可注入行为、4 个检测情境；对比基准为 CERT Insider Threat Dataset。

**📈 对比分析**

比较方法：在相同 Corpus 上对十个模型进行 triage 和 verdict 评估，报告 F1、precision、recall、baseline FP。性能显示：8 模型 triage F1=0.80，但 verdict 分为 Tier A (F1=1.0) 与 Tier B (F1=0.80)；Tier B 模型在 baseline FP 方面波动两位数（0.021–0.813），凸显 FP 对可操作性的影响。

**⚠️ 局限性**

局限性：仅单一 Corpus 与单次运行、固定三阶段流水线、仅评测官方 prompt、精确匹配的行为标签导致词汇幻觉、缺乏多配置、多模型多 prompt 的泛化评估。

---

## 124. Parallelizable Feynman-Kac Models for Universal Probabilistic Programming

**arXiv ID:** 2603.22463 | [PDF](https://arxiv.org/pdf/2603.22463v1)

**作者:** Michele Boreale `[一作]` (University of Florence), Luisa Collodi `[通讯]` (University of Florence)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5035705980)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在概率程序的形式语义框架下，证明并实现了基于向量化粒子滤波的Sequential Monte Carlo推理方法。

**💡 创新点**

将无限轨迹的期望语义与有限时间的Feynman‑Kac模型建立严谨连接，给出可计算的上下界，并提出可并行执行的向量化PF算法。

**🔧 技术方法**

使用概率程序图（PPG）语义、测度论与期望、有限逼近定理、Feynman‑Kac框架、粒子滤波（PF）以及TensorFlow实现的SIMD向量化技术。

**📊 数据集**

在含有无限循环和嵌套条件的基准模型（航空器跟踪、醉汉与老鼠、海龟与兔子、重发协议、非i.i.d.循环、ZeroConf、随机游走等）上进行实验。

**📈 对比分析**

与webPPL（SMC）、CorePPL以及精确重jection方法对比，通过期望值、有效样本量（ESS）和执行时间评估；在大规模粒子数下，VPF在大多数案例中跑时更短、ESS更高，精度与现有工具相近。

**⚠️ 局限性**

对条件概率稀疏或循环深度有限的模型，VPF优势不明显；实现依赖于TensorFlow等库，缺乏将高级PPL自动转换为PPG的编译器；对极端大规模状态空间或高维连续变量的支持仍需改进。

---

## 125. Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures

**arXiv ID:** 2603.22473 | [PDF](https://arxiv.org/pdf/2603.22473v1)

**作者:** Hector Borobia `[一作]` (Universitat Politecnica De Valencia), Guillermina Tormo-Carbó `[通讯]` (Universitat Politecnica De Valencia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过系统的功能组件消融实验，评估了两种不同的混合语言模型（Qwen3.5‑0.8B与Falcon‑H1‑0.5B）中注意力与替代序列处理组件（线性注意力/SSM）的实际贡献与相互关系。

**💡 创新点**

创新点在于：①首次对混合模型进行全覆盖的层级、组级、位置级消融，并引入匹配随机控制与bootstrap置信区间；②揭示了替代组件在混合模型中扮演主干角色，注意力仅提供补充；③发现混合架构对层级消融表现出显著的早期层梯度和内在冗余；④比较混合模型与纯Transformer在随机层消融下的鲁棒性差异。

**🔧 技术方法**

使用的技术包括：可逆前向钩子实现的功能消融（序列跳过与并行零化）、多层级消融策略、匹配随机消融、WikiText‑2困惑度分析、隐藏状态指标（norm变化、余弦相似度、logit‑lens KL）、bootstrap置信区间与随机控制对照。

**📊 数据集**

实验数据集包括：MMLU、GSM8K、ARC‑Challenge、HellaSwag、TruthfulQA-MC 与 WikiText‑2；对每个模型共执行84+种消融条件，并在上述五大基准上评估。

**📈 对比分析**

与纯Transformer（Qwen2.5‑0.5B）对照，混合模型在随机层消融下的困惑度提升仅为20–119×，远低于Transformer；单个层级消融表现出明显的早期层梯度；移除替代组件导致困惑度暴增（Qwen 35,200×、Falcon 53×），而移除注意力仅升至数十倍，表明替代组件是主干。

**⚠️ 局限性**

局限性包括：仅研究子1B规模模型，未验证更大模型的是否一致；功能消融未重训模型，可能高估冗余；不同架构的消融实现不完全可比；基准样本短，未充分探测长上下文对混合组件的依赖。

---

## 126. Geometric Mixture-of-Experts with Curvature-Guided Adaptive Routing for Graph Representation Learning

**arXiv ID:** 2603.22317 | [PDF](https://arxiv.org/pdf/2603.22317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 127. Beyond Hard Constraints: Budget-Conditioned Reachability For Safe Offline Reinforcement Learning

**arXiv ID:** 2603.22292 | [PDF](https://arxiv.org/pdf/2603.22292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 128. Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits

**arXiv ID:** 2603.22339 | [PDF](https://arxiv.org/pdf/2603.22339v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 129. Full waveform inversion method based on diffusion model

**arXiv ID:** 2603.22307 | [PDF](https://arxiv.org/pdf/2603.22307v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 130. Rethinking Multimodal Fusion for Time Series: Auxiliary Modalities Need Constrained Fusion

**arXiv ID:** 2603.22372 | [PDF](https://arxiv.org/pdf/2603.22372v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 131. Allometric Scaling Laws for Bipedal Robots

**arXiv ID:** 2603.22560 | [PDF](https://arxiv.org/pdf/2603.22560v1)

**作者:** Naomi Oke `[一作]` (Carnegie Mellon University), Aaron M. Johnson `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1728 | [OpenAlex ID](https://openalex.org/A5081925724)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过文献回顾和Drake仿真研究了双足机器人在不同尺度下质量、速度、扭矩和足形的尺度律。

**💡 创新点**

创新点在于发现机器人质量随腿长的指数为约2而非3，扭矩需求按mL或L^3-4缩放，且最佳足形随尺度变化，需要针对尺度优化。

**🔧 技术方法**

使用文献回顾、线性回归、Drake 3D仿真（hydroelastic接触模型）、手动调节扭矩和速度、身体姿态测量以及足形曲率扫描。

**📊 数据集**

使用自建的机器人参数数据库（10个低阶双足、17个完整双足、14个四足、6个六足），托管于GitHub（robomechanics/robot-dataset），并将数据与仿真结果对比。

**📈 对比分析**

通过比较不同尺度、足形、控制策略下的速度、扭矩和姿态，结果显示速度近似v∝√L，扭矩低于几何预测且落在现有电机能力范围，最佳足形随尺度变化。

**⚠️ 局限性**

局限性包括仅考虑单轴臀关节被动双足，实验仅在仿真且只校准了1×尺寸，未涵盖多关节或多模式机器人、真实摩擦等因素，且多足样本量有限，缺乏极端尺寸的实验验证。

---

## 132. Graph-Aware Late Chunking for Retrieval-Augmented Generation in Biomedical Literature

**arXiv ID:** 2603.22633 | [PDF](https://arxiv.org/pdf/2603.22633v1)

**作者:** Pouria Mortezaagha `[一作]` (Ottawa Hospital Research Institute), Arya Rahgozar `[通讯]` (University of Ottawa)

**通讯引用:** 75 | [OpenAlex ID](https://openalex.org/A5006363388)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种结合图结构信息的晚期分块框架，旨在提升医学文献检索增强生成（RAG）系统的结构覆盖能力。

**💡 创新点**

创新点在于将文档结构图与UMLS知识图融合到晚期分块流程中，既保留全文上下文，又通过结构感知的分块边界和图引导检索实现检索宽度的显著提升。

**🔧 技术方法**

使用技术包括：长上下文Transformer编码、结构图构建与边界分数计算、GAT知识图注入、结构感知分块与图引导混合检索，配合FAISS向量索引。

**📊 数据集**

实验数据集为2,359篇经过IMRaD过滤的PubMed Central全文与1,000篇PubMedQA摘要，另外使用2,033个跨段落问题模板。

**📈 对比分析**

与传统内容相似分块、固定分块、晚期分块等六种配置对比，语义分块在MRR上最高（≈0.517），但结构感知方法在SecCov@20上达到15.57，表明在检索广度上有显著优势。

**⚠️ 局限性**

主要限制包括：合成跨段问题的代表性不足、IMRaD过滤导致样本偏倚、单语言、检索深度不足以实现跨段回忆、生成评估样本有限、缺乏人工评价。

---

## 133. Tock: From Research to Securing 10 Million Computers

**arXiv ID:** 2603.22585 | [PDF](https://arxiv.org/pdf/2603.22585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 134. LLM-guided headline rewriting for clickability enhancement without clickbait

**arXiv ID:** 2603.22459 | [PDF](https://arxiv.org/pdf/2603.22459v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 135. MERIT: Memory-Enhanced Retrieval for Interpretable Knowledge Tracing

**arXiv ID:** 2603.22289 | [PDF](https://arxiv.org/pdf/2603.22289v1)

**作者:** Runze Li `[一作]` (East China Normal University), Wei Zhang `[通讯]` (East China Normal University)

**通讯引用:** 25317 | [OpenAlex ID](https://openalex.org/A5083536745)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 MERIT 框架，在知识追踪中采用训练‑free 的检索‑再推理策略，利用冻结的大语言模型（LLM）结合结构化、可解释的认知模式记忆库来预测学生未来表现。

**💡 创新点**

创新点包括：① 通过语义去噪与密度聚类发现认知架构，实现可解释的学生分组；② 生成链式推理（Chain‑of‑Thought）记忆条目，构建离线的解释性记忆库；③ 在检索阶段采用层级路由与混合语义‑关键词检索，显著过滤噪声；④ 在推理阶段引入逻辑约束（Spike Rule）校正推理偏差；⑤ 完全不需要梯度更新，支持增量学习与动态知识更新。

**🔧 技术方法**

使用的技术包括：大语言模型（Gemini‑2.5‑Flash、GPT‑4o）、语义去噪与嵌入、UMAP 降维+密度聚类、c‑TF‑IDF 语义标注、FAISS 密集检索、BM25 关键词检索、混合检索权重 α、逻辑约束正则化以及链式推理生成。

**📊 数据集**

实验数据集涵盖四个公开基准：ASSISTments 2009、ASSISTments 2012、Eedi（大规模诊断数据）以及 BePKT（编程知识追踪）。

**📈 对比分析**

与传统深度学习 KT（DKT、AKT、SAKT、LPKT、IKT、DIMKT、DKVMN）以及 LLM‑增强方法（EPLF、EFKT、2T‑KT、Thinking‑KT）对比，评估指标为 AUC、ACC、F1。MERIT 在所有四个数据集上均超过基线，最高 AUC 为 0.8244（Gemini‑Flash）或 0.8188（GPT‑4o），显著提升预测准确性。

**⚠️ 局限性**

局限性：① 仍需依赖外部 LLM API，推理时的延迟与成本受限；② 检索窗口和混合检索权重需手动调优；③ 对极长历史或稀有知识点的覆盖率有限；④ 记忆库的生成质量受 LLM 生成能力影响；⑤ 目前仅针对中文/英文语料，跨语言适用性待验证。

---

## 136. Efficient Embedding-based Synthetic Data Generation for Complex Reasoning Tasks

**arXiv ID:** 2603.22294 | [PDF](https://arxiv.org/pdf/2603.22294v1)

**作者:** Srideepika Jayaraman `[一作]` (IBM Research), Jayant Kalagnanam `[通讯]` (IBM Research)

**通讯引用:** 10037 | [OpenAlex ID](https://openalex.org/A5057936833)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对目标小型LLM的性能短板，提出基于嵌入空间的合成数据生成管道，通过在稀疏区域进行种子选择、插值、解码并利用教师模型生成新的训练样本。

**💡 创新点**

创新点在于将合成数据生成从文本域转移到目标模型的嵌入空间，利用密度-准确率的正相关性主动补齐稀疏区域，从而显著提升小模型的表现。

**🔧 技术方法**

使用嵌入计算（Transformer词嵌入+注意力加权）、PCA/ t‑SNE降维、稀疏区域识别、种子插值、解码提示、教师模型生成（Mistral‑Large）以及LoRA微调。

**📊 数据集**

MetaMathQA 作为种子数据集；在 GSM8K 与 MATH 两个数学推理基准上评估；使用 Granite 3 8B、Granite 3.1 8B 与 Mistral 7B 三个目标模型。

**📈 对比分析**

将随机种子生成与基于嵌入稀疏性采样生成的合成数据进行对比。实验显示 EmbedSDG 在所有模型和基准上均优于随机采样，提升幅度最高可达 39%（Mistral 7B 在 GSM8K 上 500 条样本时）。

**⚠️ 局限性**

仅在 3 个模型和 2 个数据集上验证，缺乏跨领域通用性；实验依赖已公开的 finetune 数据集，受限于可获取的数据；构建与部署大模型仍需要昂贵计算资源。

---

## 137. Architecture-Derived CBOMs for Cryptographic Migration: A Security-Aware Architecture Tradeoff Method

**arXiv ID:** 2603.22442 | [PDF](https://arxiv.org/pdf/2603.22442v1)

**作者:** Eduard Hirsch `[一作]` (University of Applied Sciences Amberg-Weiden), Kristina Raab `[通讯]` (Fraunhofer AISEC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并实现了安全意识架构评估方法（satam），利用场景化架构评估（ATAM）结合威胁建模、质量属性场景、ADR和CARAF，生成基于架构、上下文敏感且可追溯的加密物料清单（CBOM），以支持加密迁移规划。

**💡 创新点**

创新点在于：
1) 将ATAM与STRIDE、SQAS、ADR、CARAF等安全技术融合，形成一套完整的安全化架构评估流程；
2) 通过追溯链将加密资产与架构元素、威胁、质量属性场景、决策记录以及风险评估关联，生成结构化、可机器处理的CBOM；
3) 使CBOM不再仅是库存，而是承载加密使用的安全动机、架构约束和迁移风险的文档化产物。

**🔧 技术方法**

使用技术包括：
- 场景化架构评估（ATAM）
- 结构化架构文档（arc42）
- 威胁建模（STRIDE）
- 安全质量属性场景（SQAS）
- 架构决策记录（ADR）
- 加密物料清单（CBOM）与CycloneDX格式扩展
- 加密灵活性风险评估框架（CARAF）
- 设计科学研究（DSR）方法论

**📊 数据集**

本研究以一个中等规模的示例系统（公开API、外部身份提供者、后端服务、数据库）作为案例，手工构建了相应的arc42文档、STRIDE威胁、SQAS、ADR、CARAF评估以及最终的CBOM。未使用大规模或真实工业数据集。

**📈 对比分析**

评估方式：
- 通过分析评估验证方法内部一致性、追溯完整性及对迁移的支持性；
- 通过一个概念验证实例演示完整流程；
- 与传统库存驱动CBOM进行任务级对比，展示satam CBOM在识别受影响数据流、威胁、质量属性场景和决策理由方面的优势，显著降低了手工重建信息的工作量。性能方面未给出定量指标，仅说明方法可行且在小规模场景下有效。

**⚠️ 局限性**

局限性：
- 仅在单一示例系统上进行概念验证，未在工业规模环境中实验；
- 对方法的可扩展性、实施成本和时间投入缺乏经验数据；
- CARAF输入参数及风险模型示例为示范性质，未进行校准；
- 依赖已有完整的架构文档和安全评估实践，对缺乏架构治理的组织适用性有限；
- 缺乏工具自动化支持，当前流程仍需人工参与。

---

## 138. Towards Automated Community Notes Generation with Large Vision Language Models for Combating Contextual Deception

**arXiv ID:** 2603.22453 | [PDF](https://arxiv.org/pdf/2603.22453v1)

**作者:** Jin Ma `[一作]` (Clemson University), Long Cheng `[通讯]` (Clemson University)

**通讯引用:** 3042 | [OpenAlex ID](https://openalex.org/A5080025320)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种自动化生成社区笔记（Community Notes）的框架，针对图像与上下文误导的情境性虚假信息；

**💡 创新点**

创新点包括：①构建了真实世界多模态数据集（X帖子的社区笔记与逆向搜索得到的外部上下文）；②提出了基于检索增强的多智能体协作框架（ACCNOTE）和新的评估指标（Context Helpfulness Score，CHS）；③通过多智能体协作显著提升了大型视觉语言模型在检测与生成笔记方面的表现；

**🔧 技术方法**

技术主要包括：大型视觉语言模型（如LLaVA、Qwen2.5-VL、InternVL3、NVILA）、检索增强生成（RAG）、多智能体协作（Data Organizer、Reasoner、Judge），以及基于句子向量与情感分析的 CHS 评估方法；

**📊 数据集**

使用的数据集为自建的“X社区笔记”数据集，包含2,176条帖子（1,088 把持谣言，1,088 正面），每条帖子附有社区笔记、外部上下文（经逆向图像搜索得到的多达10条URL摘要），并提供主题与虚假因素标签；

**📈 对比分析**

与基线（闭合书写、Naive RAG、SNIFFER、GPT5-mini）对比，ACCNOTE 在上下文虚假检测中实现了 F1 最高 0.8744、准确率 0.8745；在笔记生成中 CHS 最高 0.9003（相比基线提升 5%+），且在所有五项 Community Notes 维度上均优于人类社区笔记与商业工具；

**⚠️ 局限性**

局限性包括：①多智能体框架高度耦合，单一模块失效会导致整个系统失效；②对外部检索质量高度依赖，检索失败或噪声会影响结果；③模型仍存在高召回、低精确度的倾向，导致假阳性率较高；④评估仍需进一步验证更大规模用户研究。

---

## 139. Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos

**arXiv ID:** 2603.22529 | [PDF](https://arxiv.org/pdf/2603.22529v1)

**作者:** Shoubin Yu `[一作]`, Boqing Gong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了首个将第一人称视频感知与真实Web任务结合的基准，称为Ego2Web；

**💡 创新点**

创新点在于将视觉感知与Web执行紧密耦合，并设计了基于视觉证据的LLM-as-a-Judge自动评估框架；

**🔧 技术方法**

主要技术包括多模态大型语言模型（Qwen3‑VL、GPT‑5、Gemini等）进行视频字幕生成、任务规划及评判；

**📊 数据集**

使用500条来自公开第一人称视频数据集（如Ego4D）的样本，结合Amazon、YouTube、Wikipedia等主流网站；

**📈 对比分析**

与六种主流Web agent（SeeAct、Browser‑Use、Claude Computer‑Use、GPT‑5.4等）对比，评估人类与自动Judge的一致率平均达80‑85%，最佳 agent BU‑Gemini‑3‑Flash 在人类评估下成功率58.6%，仍与理想模型存在约40%差距；

**⚠️ 局限性**

局限性包括：视频输入依赖LLM理解精度，文本化描述导致信息丢失；跨模态检索与时序理解易失误；评估依赖LLM偏见且对动态网页变化的鲁棒性有限。

---

## 140. Reddit After Roe: A Computational Analysis of Abortion Narratives and Barriers in the Wake of Dobbs

**arXiv ID:** 2603.22566 | [PDF](https://arxiv.org/pdf/2603.22566v1)

**作者:** Aria Pessianzadeh `[一作]` (Drexel University), Rezvaneh Rezapour `[通讯]` (Drexel University)

**通讯引用:** 336 | [OpenAlex ID](https://openalex.org/A5012092057)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用机器学习对 2022 年 Dobbs 决策前后 17,534 条 Reddit 异议帖进行分层分析，探究信息寻求/分享、堕胎阶段与障碍类型如何交织并影响情绪表达。

**💡 创新点**

创新点在于将信息行为、堕胎阶段和政策时点三维度与障碍与情绪同时建模，形成多维度叙事框架，并通过主题建模揭示不同时间段与行为模式下障碍话语的演变。

**🔧 技术方法**

使用的技术包括 BERT/ RoBERTa 等Transformer、GPT‑4 系列 LLM、GoEmotions 情感识别、BERTopic+GPT‑4.1 的主题建模，以及多模型集成与 Gwet AC1 一致性评估。

**📊 数据集**

数据集来源于 Pushshift API 收集的 r/abortion、r/abortiondebate、r/prochoice、r/prolife 四个子版块，时间窗口为 2022 年 1 月至 12 月共 17,534 条帖子。

**📈 对比分析**

模型对信息行为的分类以 RoBERTa 取得最高 F1=0.93；障碍类型分类采用 GPT‑4o、GPT‑5.1、GPT‑4.1‑mini 三模型集成，平均 F1≈0.75；情绪识别使用 GPT‑4o 评估；整体性能优于现有基线但受限于标注规模。

**⚠️ 局限性**

局限包括：依赖 LLM 可能导致标注偏差、Reddit 样本不具代表性、仅关注 12 个月的短期效应、仅用卡方检验缺乏交互效应分析、未提供效应大小指标。

---

## 141. Session Risk Memory (SRM): Temporal Authorization for Deterministic Pre-Execution Safety Gates

**arXiv ID:** 2603.22350 | [PDF](https://arxiv.org/pdf/2603.22350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 142. UniFluids: Unified Neural Operator Learning with Conditional Flow-matching

**arXiv ID:** 2603.22309 | [PDF](https://arxiv.org/pdf/2603.22309v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 143. Architectural Enhancements for Efficient Sensing Data Utilization in 6G ISAC

**arXiv ID:** 2603.22488 | [PDF](https://arxiv.org/pdf/2603.22488v1)

**作者:** Muhammad Awais Jadoon `[一作]` (InterDigital Europe Ltd), Sebastian Robitzsch `[通讯]` (InterDigital Europe Ltd)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了6G ISAC架构中的感知数据存储函数（sdsf），实现历史感知数据与实时数据融合，并通过地图感知硬过滤提升感知性能。

**💡 创新点**

创新点在于首次将持久化感知数据存储与历史‑实时融合结合，利用地图感知硬过滤显著降低误报而不影响检测概率。

**🔧 技术方法**

采用极坐标转笛卡尔坐标的EKF线性化、地图感知硬掩模（Minkowski求和）、欧氏门限验证、泊松噪声仿真等技术。

**📊 数据集**

使用仿真数据集：120×120米路口场景，含两栋建筑、8个移动目标以及每步平均60个泊松噪声点。

**📈 对比分析**

通过Monte Carlo仿真比较不同掩模宽度g和验证门宽g_det下的检测概率和误报率，结果显示g≥2时误报率下降约70‑80%，检测概率基本保持。

**⚠️ 局限性**

局限在于硬掩模采用固定阈值，未考虑不确定性；仅在仿真环境验证，缺乏真实场景实验；误报与检测之间的权衡仍需进一步调优。

---

## 144. TrajLoom: Dense Future Trajectory Generation from Video

**arXiv ID:** 2603.22606 | [PDF](https://arxiv.org/pdf/2603.22606v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 145. Learning When to Act: Interval-Aware Reinforcement Learning with Predictive Temporal Structure

**arXiv ID:** 2603.22384 | [PDF](https://arxiv.org/pdf/2603.22384v1)

**作者:** Davide Di Gioia `[一作]` `[通讯]` (University College London), Davide Di Gioia (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种可学习的自适应时间控制层ATCPG，用于决定自主代理何时进行认知更新。

**💡 创新点**

核心创新是利用Poincaré球中的预测超几何扩散信号作为不确定性指标，并设计了基于间隔的奖励来纠正信用分配失效。

**🔧 技术方法**

技术包括线性关联分散器学习、Poincaré几何映射、曲率信号计算、内部相位耦合以及联合时空嵌入。

**📊 数据集**

实验使用了合成环境模拟LLM调用，以及在OpenAI GPT‑4.1 的真实API调用进行多轮任务。

**📈 对比分析**

与固定间隔、反应式和无学习基线相比，ATCPG‑ST 在效率上提升约22‑30%，在GPT‑4.1 实验中减少10% 以上 token 并保持100% 成功率。

**⚠️ 局限性**

局限包括线性策略可能不足以处理高度非凸场景、未在大规模多任务基准上验证、以及对未来状态的假设性世界模型依赖。

---

## 146. Sparsely-Supervised Data Assimilation via Physics-Informed Schrödinger Bridge

**arXiv ID:** 2603.22319 | [PDF](https://arxiv.org/pdf/2603.22319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 147. LLMON: An LLM-native Markup Language to Leverage Structure and Semantics at the LLM Interface

**arXiv ID:** 2603.22519 | [PDF](https://arxiv.org/pdf/2603.22519v1)

**作者:** Michael Hind `[一作]` (IBM Research), Dan Gutfreund `[通讯]` (IBM Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 LLMON——一种 LLM 本地化标记语言，用于在训练和推理阶段显式传递结构化元数据，从而提升 LLM 的准确性、安全性和鲁棒性。

**💡 创新点**

创新点在于设计能表达用户自定义标签、实例命名、显式层次的标记语言，并通过特殊 token 在模型输入中显式区分指令与数据，提供可被训练和推理系统直接利用的结构信息。

**🔧 技术方法**

采用特殊 token 标记、双语法（人类友好 LLMON 与机器友好 LLMON‑）、JSON↔LLMON 转换器、结构化训练策略以及基于边界约束的推理掩码技术。

**📊 数据集**

以公开指令调优数据（如 Alpaca、Dolly 等）为基础，构造带 LLMON 注释的训练语料和 Distractor 评估基准。

**📈 对比分析**

通过与基础模型、传统 chat‑template 训练和 LLMON 结构化训练的对比，Distractor 基准上平均提升 74.2 个百分点；在 MMLU、GSM8K、IFEval 等单指令基准保持或略降；推理时使用边界掩码平均提升 29.3 个百分点。

**⚠️ 局限性**

局限性包括需要手工或自动化生成结构化训练数据、特殊 token 的初始化与 tokenizer 兼容性未全面评估、对更大模型或不同架构的泛化效果待验证，以及仅关注指令/数据区分，对更复杂任务的适用性尚需进一步研究。

---

## 148. First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution

**arXiv ID:** 2603.22346 | [PDF](https://arxiv.org/pdf/2603.22346v1)

**作者:** Drake Caraker `[一作]` (Independent Researchers), David Rhoads `[通讯]` (Independent Researchers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究梯度提升树解释不稳定性中的“先行者偏差”，并提出并验证一种通过独立模型集合来消除该偏差的解释稳定性方法（DASH）。

**💡 创新点**

创新点：①首次将“先行者偏差”定义为梯度提升中残差递归导致的特征重要性聚集机制；②提出只需模型独立性（如随机种子平均）即可大幅恢复稳定性；③设计了无先验真值的诊断工具 Feature Stability Index 与 Importance‑Stability Plot；④将方法封装为可插拔的多模型SHAP聚合框架。

**🔧 技术方法**

技术：XGBoost 训练、TreeSHAP（交互式与交叉验证）、模型集群（M=200）、性能过滤、Max‑Min 多样性选择、交叉集成 SHAP 平均、FSI 与 IS Plot 诊断、实验统计（Wilcoxon、BCa 置信区间）。

**📊 数据集**

数据集：合成多重共线性数据、乳腺癌 Wisconsin（30 特征，21 对 |r|>0.9）、超导体数据集（81 特征），加州住房数据集（8 特征）。

**📈 对比分析**

比较方法：Single Best、Single Best (M=200)、Large Single Model、Ensemble SHAP、Stochastic Retrain、Random Selection、Naive Top‑N、(MaxMin) DASH。性能：DASH 与 Stochastic Retrain 在稳定性上均达到约 0.977，显著优于单模型和大型单模型；在乳腺癌上稳定性从 0.32 提升至 0.93，超导体从 0.83 提升至 0.96，房价从 0.97 提升至 0.98；预测误差基本不受影响。

**⚠️ 局限性**

局限性：①在非线性生成过程下整体稳定性下降，方法优势不显著；②需大量模型训练与 SHAP 计算，计算成本较高；③对高度相关特征的交互式 SHAP 仍存在分布外样本问题；④缺少随机森林等非梯度提升模型的基线验证；⑤对多特征交互的稳定性分析未完整实现。

---

## 149. Generalizing Dynamics Modeling More Easily from Representation Perspective

**arXiv ID:** 2603.22655 | [PDF](https://arxiv.org/pdf/2603.22655v1)

**作者:** Yiming Wang `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**通讯引用:** 72037 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种通用预训练动力学编码器（Pre-trained Dynamics EncoDER），通过在大量多样化系统观测上预训练，使得能将原始状态嵌入到一个更易捕捉动力学的潜在空间中，并在此空间中利用 GNN‑ODE 等方法进行细调以完成特定系统的动力学建模与时序预测。

**💡 创新点**

创新点在于：①使用最大 Lyapunov 指数目标约束潜在空间中的动力学平稳性，显著降低嵌入观测的混沌程度；②在预训练阶段同时加入重构与预测任务，防止潜在空间过度平滑；③构建了 152 组跨 23 种真实与合成系统的预训练语料库，突破了传统单任务或同方程多任务的局限，实现了从表示层面跨系统、跨域的泛化。

**🔧 技术方法**

技术手段包括：基于预训练语言模型（T5 / T5‑small）构建的 Transformer 编码/解码器；观测序列分块、实例归一化、线性投影与卷积预处理；最大 Lyapunov 指数计算与最小化；重构与预测损失；GNN‑ODE 动力学学习器用于细调；多任务与跨域实验设计。

**📊 数据集**

使用的数据集为：152 组观测（13 合成、10 真实），在实验中取 12 个系统进行细调，具体包括合成系统（Mutualistic、Heat Diffusion、2D‑CFD、Gene）和真实系统（T‑Drive、CHIBike、NYCTaxi、PEMS03、PEMS04、PEMS07、PEMS08、NOAA）。

**📈 对比分析**

与六个基线（LatentODE、GNS、NDCN、TREAT、STGODE、MTGODE）在短期/长期预测、in‑domain 与 cross‑domain 设置下比较。实验结果表明，预训练模型在 RMSE、MAE、MAPE 上普遍优于基线，尤其在 MAPE 上提升显著（如 2D‑CFD、NYCTaxi 下降 14%–17%）。在特殊事件比例预测任务中，同样取得领先。

**⚠️ 局限性**

局限性包括：①预训练过程可能出现灾难性遗忘与语料不均衡导致训练效果受限；②系统特定交互图的整合尚未充分探索；③对极端噪声或非平稳数据的鲁棒性待进一步验证。

---

## 150. A graph neural network based chemical mechanism reduction method for combustion applications

**arXiv ID:** 2603.22318 | [PDF](https://arxiv.org/pdf/2603.22318v1)

**作者:** Manuru Nithin Padiyar `[一作]` (Indian Institute of Science), Konduri Aditya `[通讯]` (Indian Institute of Science)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5006420943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图神经网络（GNN）的化学机理简化框架，包含两种实现：利用预训练的代理模型指导的GNN-SM方法和基于自动编码器的GNN-AE方法。

**💡 创新点**

创新点在于将图结构的化学反应网络与GNN的消息传递机制相结合，实现对非线性物种-反应依赖关系的学习；GNN-SM利用代理模型实现全域条件下的鲁棒简化，GNN-AE通过自编码器实现对特定状态下的极致压缩。

**🔧 技术方法**

技术包括图Transformer层的消息传递、基于注意力的节点与边特征更新、代理模型（多层感知器）预测点火延迟或火焰速度、自动编码器的编码-解码结构与稀疏正则化。

**📊 数据集**

使用三套详细机理数据集：甲烷（GRI-Mech 3.0，53种物种/325反应）、乙烯（FFCM-2，96种/1054反应）以及异辛烷（Curran等，1034种/8453反应），并在不同温度、压力与等化比下生成训练样本。

**📈 对比分析**

与传统DRGEP方法对比，GNN-SM在保持接近或略优的点火延迟误差（如甲烷1.45% vs 1.53%）的同时实现更大约70%的机理压缩；GNN-AE在针对特定温度（1500K）可实现高达95%的压缩，但在泛化范围内误差显著增大（如甲烷6.83%）。

**⚠️ 局限性**

局限性包括：GNN-AE在泛化性差，需专门化训练；GNN-SM对代理模型质量依赖较高；两种方法仍需进一步验证在更复杂燃料与大规模流动模拟中的鲁棒性；图构建与特征选择对结果影响尚未完全系统化。

---

## 151. Whether, Not Which: Mechanistic Interpretability Reveals Dissociable Affect Reception and Emotion Categorization in LLMs

**arXiv ID:** 2603.22295 | [PDF](https://arxiv.org/pdf/2603.22295v1)

**作者:** Michael Keeman `[一作]` `[通讯]` (Keido Labs), Michael Keeman (Keido Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用临床设计的无情感关键词短篇，结合线性探测、激活补丁、层级消除和表征几何等四种机制解释方法，对六款LLM进行情感识别与分类的可解释性测试，揭示出情感感受识别与情感分类的两种可分离机制。

**💡 创新点**

创新点包括：①首次引入基于临床心理学的关键词自由情境刺激，解决了传统情感研究中的关键词混淆问题；②发现并验证了“情感感受识别”与“情感分类”两种独立机制，并证明它们对关键词、层次和模型规模的依赖不同；③在同一研究框架下同时应用四种解释方法，提供跨方法的验证；④系统分析模型规模对情感处理的结构性影响。

**🔧 技术方法**

使用的技术：
- 线性探测（logistic 回归）评估情感信息可解码性；
- 激活补丁（activation patching）验证因果影响；
- 层级消除（knockout）分析关键层的脆弱性；
- 表征几何（cosine 相似度、Silhouette 等）探讨表示空间组织；
- 关注权重分析、交叉集补丁和零射线对比等。

**📊 数据集**

使用的数据集：
- Set A：80条关键词丰富的情感句子（来自 crowd‑enVENT 等）；
- Set B：96条临床情景短文（无情感关键词，涵盖 8 种情感 × 3 主题 × 4 版本），加 96 条中性对照；
- 六款模型：Llama‑3.2‑1B、Llama‑3.2‑1B‑Instruct、Llama‑3‑8B、Llama‑3‑8B‑Instruct、Gemma‑2‑9B、Gemma‑2‑9B‑Instruct。

**📈 对比分析**

对比与性能：
- 情感感受识别（情感 vs 中性）在 Set B 上 AUROC 恒为 1.0，且在模型最早层即饱和；
- 情感分类（8 类）在 Set B 上 AUROC 下降 1–7%（1B 模型 4.6–6.7%，8B/9B 模型 1.1–1.9%），但仍远高于随机（12.5%）；
- 交叉集补丁表明关键词丰富与无关键词文本共享相同情感空间；
- 关键层消除显示无关键词文本对层级更为分散，规模越大可分布化程度越高；
- 规模提升带来关键词依赖下降、跨集转移提升和模型层级分布化。

**⚠️ 局限性**

局限性：
- Set B 仅由单位临床心理学家设计，缺乏多评估者验证；
- 仅覆盖 1B–9B 参数规模，未验证更大模型的趋势；
- 交叉主题置换检验样本量有限，统计功效不足；
- 同情补丁样本量小，置信区间宽；
- 仅使用 Plutchik 情感分类，未覆盖维度或其他情感模型；
- 结果仅说明信息处理机制，未涉及情感意识或体验。

---

## 152. The Efficiency Attenuation Phenomenon: A Computational Challenge to the Language of Thought Hypothesis

**arXiv ID:** 2603.22312 | [PDF](https://arxiv.org/pdf/2603.22312v1)

**作者:** Di Zhang `[一作]` `[通讯]` (Xi'an Jiaotong-Liverpool University), Di Zhang (Xi'an Jiaotong-Liverpool University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

通过多智能体强化学习实验验证了“效率衰减现象”，检验语言思维假说的有效性。

**💡 创新点**

提出并量化了“AI私人语言”思想实验，证明自发通信比预设符号更高效，挑战了语言思维假说。

**🔧 技术方法**

使用双智能体DQN与多智能体强化学习框架，配合符号通信模块、Shannon熵与Jensen–Shannon散度等统计分析技术。

**📊 数据集**

在自生成的5×5网格协作导航任务中随机放置宝藏位置，未使用公开数据集。

**📈 对比分析**

将自发通信（EC）与预设符号通信（PSP）条件的平均步数进行比较，结果显示EC平均步数28.7步，PSP为43.2步，效率衰减率约为50.5%。

**⚠️ 局限性**

实验仅在简化的二维导航任务和MLP模型上进行，未检验更复杂环境、更大规模模型或更丰富符号体系下的效果，泛化与可扩展性尚待进一步验证。

---

## 153. UrbanVGGT: Scalable Sidewalk Width Estimation from Street View Images

**arXiv ID:** 2603.22531 | [PDF](https://arxiv.org/pdf/2603.22531v1)

**作者:** Kaizhen Tan `[一作]` (Carnegie Mellon University), Fan Zhang `[通讯]` (Peking University)

**通讯引用:** 54001 | [OpenAlex ID](https://openalex.org/A5100403400)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了单张街景图像的面宽估计管线UrbanVGGT，并在三座城市生成了未验证的SV‑SideWidth数据集

**💡 创新点**

将面宽估计转化为平面约束的3D测量，利用单图像3D重建+相机高度校准，避免了多视角或预置深度需求，并将语义分割、平面拟合与尺度校准融合成统一流程

**🔧 技术方法**

使用SegFormer‑B5进行语义分割，VGGT进行单图像3D重建，RANSAC+SVD拟合平面，基于相机高度的尺度校准，以及列宽提取与统计方法

**📊 数据集**

基准数据为Washington D.C.约300张Google Street View图像及其面宽标注；公开的OpenStreetMap街道网络；以及NYC、São Paulo、Nairobi的街景图像

**📈 对比分析**

在统一下游管线下与9种不同深度/重建骨干（Metric3D、UniDepthV2、MapAnything、π³等）进行对比；在DC基准上MAE为0.252 m，95.5%估计误差≤0.5 m；相机高度敏感度分析表明2.5 m为最佳假设

**⚠️ 局限性**

仅在单一DC数据集验证，缺乏跨城市或多域泛化评估；SV‑SideWidth未进行地面真值审核；依赖街景图像覆盖率与质量；固定相机高度假设可能产生系统误差；对非平地、遮挡、窄人行道等情况易失效

---

## 154. WIST: Web-Grounded Iterative Self-Play Tree for Domain-Targeted Reasoning Improvement

**arXiv ID:** 2603.22352 | [PDF](https://arxiv.org/pdf/2603.22352v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 155. A Brief Comparison of Training-Free Multi-Vector Sequence Compression Methods

**arXiv ID:** 2603.22434 | [PDF](https://arxiv.org/pdf/2603.22434v1)

**作者:** Rohan Jha `[一作]` (Johns Hopkins University), Benjamin Van Durme `[通讯]` (Johns Hopkins University)

**通讯引用:** 8669 | [OpenAlex ID](https://openalex.org/A5075825791)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多向量检索模型进行训练无关的文档序列压缩评估，比较了token pruning和token pooling两类方法；

**💡 创新点**

证明token pooling在保持检索效果的同时能显著降低索引大小，并量化了不同压缩比例下的性能差异；

**🔧 技术方法**

采用了随机、Attention、IDF评分机制的pruning；以及基于anchor的token pooling、Spherical k‑Means、Hierarchical（Ward）聚类等聚合技术；

**📊 数据集**

在BEIR六个零样本文本检索数据集和CoIR三个文本‑代码检索数据集上进行实验；

**📈 对比分析**

通过nDCG@10和Recall@100与未压缩基线对比，发现pooling方法在极端压缩（r≤0.20）下仍保持高性能，压缩率比pruning高约两倍；

**⚠️ 局限性**

仅针对文档侧、训练无关的文本压缩，未考虑查询侧动态压缩、视觉文档场景，且聚类成本较高，需进一步探索自适应压缩策略。

---

## 156. ETH Flippers Approach to Parallel Reconfiguration of Triangulations: SAT formulation and Heuristics

**arXiv ID:** 2603.22456 | [PDF](https://arxiv.org/pdf/2603.22456v1)

**作者:** Lorenzo Battini `[一作]` (ETH Zürich), Marko Milenković `[通讯]` (ETH Zürich)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究在共同点集上的三角剖分之间的中心三角剖分，寻找最小总平行翻转距离，并提出混合解法（SAT+启发式）

**💡 创新点**

创新点在于将问题建模为带XOR子句的SAT，结合精确下界与距离分布启发式，提出基于图着色的最大独立翻转选择和局部搜索优化路径

**🔧 技术方法**

采用CryptoMiniSat5、Kissat等SAT求解器；使用k‑d树与空间哈希寻找空凸四边形；实现AMO约束、图着色、贪心及局部搜索等技术

**📊 数据集**

使用CG:SHOP 2026竞赛的250个实例，点数从100到500以上的随机或特定分布（rirs、woc等）点集

**📈 对比分析**

与Shadoks团队对比，取得第二名；在250实例中对186个得到最优解；对大规模实例使用启发式，取得接近最优；在中等规模（n≤320）能得到全局最优

**⚠️ 局限性**

在极大实例（n≥500）仍未完全最优；SAT求解器受限于四边形约束生成的规模；对特殊分布（如凸位置）效率不佳

---

## 157. Graphs RAG at Scale: Beyond Retrieval-Augmented Generation With Labeled Property Graphs and Resource Description Framework for Complex and Unknown Search Spaces

**arXiv ID:** 2603.22340 | [PDF](https://arxiv.org/pdf/2603.22340v1)

**作者:** Manie Tadayon `[一作]` (Capital Group), Mayank Gupta `[通讯]` (Capital Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种端到端的 Graph RAG 框架，利用 RDF 与 LPG 两种图模型将半结构化 JSON 数据转换为知识图谱，实现动态检索、无需预设文档数以及无 rerank 的检索-生成流程；

**💡 创新点**

创新点包括：① 将 JSON 转为 RDF 三元组的自动化、无噪声转换方法；② 基于 LLM 的高精度文本到 Cypher 的翻译模型；③ 在同一架构中同时支持 RDF 与 LPG，提供多种查询路径；④ 在 Amazon Neptune 上实现双模型（RDF+LPG）的一体化部署；

**🔧 技术方法**

使用技术包括：大型语言模型（GPT‑4/LLM）生成三元组与文本；BGE‑m3 进行节点/关系/文本嵌入；FAISS 与 NetworkX 进行相似度检索；Amazon Neptune、SPARQL、Gremlin/Cypher 进行图查询；文本到 Cypher 的自研模型；以及多种 reranker 进行上下文排序；

**📊 数据集**

数据集为公开的 Capital Group 互惠基金/ETF/PPS 数据，共 1104 条记录，JSON 结构深度可达 3–4 层；

**📈 对比分析**

通过 200 个覆盖 Search/Listing、Compare、Detail、Other 四类意图的查询进行对比：RAG_LPG 在 Search/Listing、Compare、Detail 上分别取得 185.5/172.5/116 分（最高）；RAG_RDF 次之；Agentic RAG（RAG_2）表现最差，尤其在 Search/Listing 仅 38.5 分；整体表明 LPG 与 RDF 的图基方法显著优于传统基于向量检索的 RAG；

**⚠️ 局限性**

局限性：① 文本到 Cypher 的翻译偶有误导致查询错误；② Agentic RAG 对检索阈值 K 的设定高度敏感，嵌入模型可能无法区分相似概念；③ RDF 方法受限于节点/关系选择错误；④ LPG 需要精细的 schema 设计，维护成本较高；

---

## 158. Task-Agnostic Exoskeleton Control Supports Elderly Joint Energetics during Hip-Intensive Tasks

**arXiv ID:** 2603.22580 | [PDF](https://arxiv.org/pdf/2603.22580v1)

**作者:** Jiefu Zhang `[一作]` (University of Michigan), Robert D. Gregg `[通讯]` (University of Michigan)

**通讯引用:** 4484 | [OpenAlex ID](https://openalex.org/A5051369788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出并验证了一种任务无关（task‑agnostic）的双侧髋部外骨骼控制器，旨在通过减轻老年人髋部负荷、降低髋部正向工作量来改善其在多种日常高需求运动（如平地步行、斜坡上坡、上楼梯、坐起）的能量经济性，并在八名老年受试者中评估其对关节级别能量效益和峰值功率的影响。

**💡 创新点**

创新点包括：① 将速度调制的虚拟弹簧（velocity‑modulated virtual springs）与物理信息层相结合，形成可解释且能与关节功率同步的控制基底；② 在控制器中嵌入任务情境调制层（task‑context modulation），利用踝间角度差、胯角度、膝角度等时空特征实现对上坡、下坡和坐起等不同任务的自适应力矩衰减；③ 通过两阶段参数优化（先在模拟中利用多活动规范数据再在受试者中微调）实现高一致性与可穿戴安全性。

**🔧 技术方法**

技术主要包括：物理信息层的三条速度调制弹簧（extension, flexion, STS），基于sigmoid函数的速度与躯干角度调制；任务情境层的下降衰减因子 α、步态‑STS 交互因子 β；双向模糊化的步态检测（基于股股与腰部IMU信号的规则式检测）；以及控制算法在 Raspberry Pi 5 上以 250 Hz 运行，并使用低通 Butterworth 滤波、仿真与实验平台结合 OpenSim 逆动力学与 OpenSim 运动捕捉/力平台同步。

**📊 数据集**

使用的数据集主要为公开的多活动低层动力学参考数据，用于阶段一参数优化：平地步行 0.85 m/s 与 1.15 m/s、斜坡上坡/下坡 5.2° 与 11°、楼梯上/下 5 in 与 7 in。实验中采集了 8 名老年人（平均年龄 72 ± 4.9 岁）的三维运动捕捉（27 轨道摄像头）、力平台数据、外骨骼执行力矩与功率数据。

**📈 对比分析**

与无外骨骼（No EXO）对比，使用线性混合模型（LMM）评估关键指标：① 生物髋正向工作量下降 24.7%（p < 0.05）；② 下肢（髋+膝+踝）正向工作量下降 9.3%（p < 0.05）；③ 生物髋峰值功率下降 14.5%（p < 0.05）；④ 总峰值功率（生物+外骨骼）提升 18.3%（p < 0.05）。在大多数高需求任务中，外骨骼能量输出与生物髋功率的余弦相似度平均超 0.88，显示出对生物动力学的高度匹配。

**⚠️ 局限性**

主要局限：① 仅评估了关节级别的能量指标，未测量代谢成本或步态速度等全身功能参数；② 受试者数量有限（n = 8），且实验均在受控实验室环境进行，未验证在真实社区环境下的可持续性与长周期效益；③ 由于外骨骼本身重量对膝部产生潜在额外负荷，某些任务（如坐起）对下肢功率影响可能偏差；④ 控制器在下降类任务中优先考虑安全导致功率匹配下降，未能在所有任务中同时最大化能量输送。

---

## 159. Velocity Potential Neural Field for Efficient Ambisonics Impulse Response Modeling

**arXiv ID:** 2603.22589 | [PDF](https://arxiv.org/pdf/2603.22589v1)

**作者:** Yoshiki Masuyama `[一作]` (Mitsubishi Electric Research Laboratories), Jonathan Le Roux `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于物理信息的神经网络（VPNF），通过逼近声速势并利用自动微分直接从声速势导出FOA RIR的四通道，从而在任何时间和位置保证满足线性化动量方程。

**💡 创新点**

创新点在于：① 将声速势作为唯一待学习的标量场，消除对单独声压和粒子速度的预测；② 通过网络输出的梯度天然满足动量方程，提升物理一致性；③ 在网络中可选加入波动方程惩罚，并提出了改进的网络变体VPNF+以进一步提升性能。

**🔧 技术方法**

使用技术包括：物理信息神经网络（PINN）框架、SIREN激活函数的多层感知机、自动微分提取梯度、波动方程软惩罚、动态权重自适应平衡、Adam优化器与余弦退火学习率、拉丁超立方采样等。

**📊 数据集**

数据集：利用HARP扩展的模拟FOA RIR数据，在十套随机尺寸和表面材料的矩形房间中生成8 kHz采样率、前100 ms早期反射的FOA RIR，共计9261个测量点；实验还考虑仅在目标立方体表面采集{100,200}个RIR的极端场景。

**📈 对比分析**

与DANF（仅数据拟合）和PI-DANF（软物理约束）对比，采用NMSE(dB)和Pearson相关系数两种指标；结果显示VPNF在测量点≤100时NMSE显著低于两基线，且在测量点多时与PI-DANF相近；VPNF+进一步提升性能；在表面采样难题中，所有VPNF变体均明显优于基线。

**⚠️ 局限性**

局限性：VPNF仅保证动量方程的满足，未能完全满足连续性方程；波动方程惩罚在实验中效果有限；若要同时满足两条物理方程，需要更复杂的网络设计；未来工作包括少量样本快速适应与更全面的物理一致性保证。

---

## 160. Color When It Counts: Grayscale-Guided Online Triggering for Always-On Streaming Video Sensing

**arXiv ID:** 2603.22466 | [PDF](https://arxiv.org/pdf/2603.22466v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 161. CanViT: Toward Active-Vision Foundation Models

**arXiv ID:** 2603.22570 | [PDF](https://arxiv.org/pdf/2603.22570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 162. Tied In on TikTok: Tie Strength and Emotional Dynamics in Algorithmic Communities

**arXiv ID:** 2603.22504 | [PDF](https://arxiv.org/pdf/2603.22504v1)

**作者:** Charles Bickham `[一作]` (University of Southern California), Emilio Ferrara `[通讯]` (University of Southern California)

**通讯引用:** 18941 | [OpenAlex ID](https://openalex.org/A5078699564)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了TikTok上吃食障碍相关讨论中的关系强度与情感表达的关联，发现重复互动会提升情感积极性；

**💡 创新点**

证明在算法驱动的短视频平台缺乏显式社交图谱的情况下，用户对话也能形成强连带关系，并且不同内容类型（Pro‑Recovery、Pro‑ED、ED经历）呈现不同情感动态；

**🔧 技术方法**

采用情感分类模型Demux和LIWC情感词典来评估评论情感，同时使用Gemini零射prompt对视频内容进行分类；

**📊 数据集**

使用公开的TikTok吃食障碍视频数据集（43040条视频、560k评论）和对应的非ED对照数据集（37530条视频、203k评论）；

**📈 对比分析**

通过交叉表和卡方检验比较不同互动频率下情感比例，结果显示在ED讨论中正面情感显著升高，Pro‑Recovery内容尤为突出；

**⚠️ 局限性**

主要局限包括情感分类器单一、LIWC词典对非正式语言敏感、内容分类准确度有限、样本多为单次互动，且研究仅为相关性而非因果。

---

## 163. Design Implications for Student and Educator Needs in AI-Supported Programming Learning Tools

**arXiv ID:** 2603.22673 | [PDF](https://arxiv.org/pdf/2603.22673v1)

**作者:** Boxuan Ma `[一作]` (Kyushu University), Shin'Ichi Konomi `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对50名教师与90名学生的平行问卷调查，比较并提炼了AI辅助编程工具在教学环境中的设计需求与偏好。

**💡 创新点**

创新点在于将教师与学生的视角融合，构建了基于四大设计维度（AI政策、查询方式、响应方式、控制权）的交互式设计空间，并提出共享控制模型与渐进式提示策略，兼顾学习效果与使用便利。

**🔧 技术方法**

技术上主要使用问卷调查与定量/定性分析方法，结合设计空间框架与用户需求建模，未直接使用特定AI模型或算法。

**📊 数据集**

数据集为问卷结果，教师组50份，学生组90份，包含背景信息、Likert量表及开放式回答，未采用公开编程数据集。

**📈 对比分析**

比较方法为对同一设计维度在教师与学生中的偏好进行统计对比（百分比、Likert评分），结果显示教师倾向间接提示与课程约束，学生倾向直接帮助；两组均倾向共享控制，但对最大提示级别与具体提示类型存在差异。

**⚠️ 局限性**

局限性包括样本主要来自美国与亚洲，受访者分布不均；采用自我报告方式，可能存在偏差；研究仅涉及需求调查，缺乏对实际AI助手实现效果的实证验证。

---

## 164. Multi-Method Validation of Large Language Model Medical Translation Across High- and Low-Resource Languages

**arXiv ID:** 2603.22642 | [PDF](https://arxiv.org/pdf/2603.22642v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 165. Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies

**arXiv ID:** 2603.22651 | [PDF](https://arxiv.org/pdf/2603.22651v1)

**作者:** Siddhant Kulkarni `[一作]` (New York University), Yukta Kulkarni `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四种多代理LLM编排架构（顺序、并行、分层、反射式）在金融文档结构化抽取上的系统基准评估

**💡 创新点**

引入多架构成本‑准确性 Pareto 分析、规模化吞吐‑准确性曲线、语义缓存+模型路由+自适应重试的混合优化配置，并给出失败模式分类

**🔧 技术方法**

使用多代理架构、LangGraph编排、GPT‑4o/Claude 3.5/Sonnet/Gemini 1.5/Meta Llama 3 70B/Mistral Mixtral 8x22B、token‑efficiency 评估、semantic caching、模型路由、adaptive retry、post‑condition 验证

**📊 数据集**

10,000份SEC文件（10‑K、10‑Q、8‑K）及25字段类型，人工标注并校准的金字塔式标注集

**📈 对比分析**

通过字段级 F1、文档级准确率、延迟、成本、token‑efficiency 等五维度，对 4 架构 × 5 模型共 500 组配置进行实验；结果表明反射式准确率最高但成本最高，分层架构在成本‑准确性 Pareto 前沿；混合配置可在仅 1.15× 顺序成本下恢复 89% 反射式准确率；在规模扩展到 50K‑100K 文档/日时，反射式快速下降，顺序最稳健

**⚠️ 局限性**

仅评估 SEC 英文文件，成本基于 2025 价格；模型版本可能随时间变化；字段集有限，未覆盖多语言/IFRS 等；人工标注可能存在系统偏差

---

## 166. Pretext Matters: An Empirical Study of SSL Methods in Medical Imaging

**arXiv ID:** 2603.22649 | [PDF](https://arxiv.org/pdf/2603.22649v1)

**作者:** Vedrana Ivezić `[一作]` (University of California, Los Angeles), William Speier `[通讯]` (University of California, Los Angeles)

**通讯引用:** 2816 | [OpenAlex ID](https://openalex.org/A5029456122)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

系统评估了 MAE、DINOv3 与 I-JEPA 三种自监督预文本任务在超声和组织学影像中的表现，探讨预文本任务如何影响医学影像的表征与下游任务性能。

**💡 创新点**

提出将预文本任务与医学影像中临床信息的空间组织对齐的选择框架，揭示不同预文本任务在宏结构与细粒度纹理上的优劣，首次从表征空间可视化角度对三种 SSL 方法进行系统比较。

**🔧 技术方法**

使用 ViT‑small 预训练、线性探测、ABMIL 聚合以及注意力、余弦相似度和 PCA 可视化等技术对表征进行评估，并在不同任务中比较表现。

**📊 数据集**

预训练数据集为约 470 万帧多源超声视频/图像，组织学数据集为约 500 万块 TCGA 全切片图像；下游任务包括 8 个超声分类任务和 5 个公开病理子类型预测任务。

**📈 对比分析**

通过冻结预训练模型在下游任务上训练线性层或 ABMIL，计算 AUROC/F1 进行比较；在超声中 I-JEPA 在宏结构任务（如脂肪肝、血管/肺）上优于 DINOv3，DINOv3 在细粒度病理任务（如腺癌 vs 鳞癌）上领先，MAE 整体表现最差。

**⚠️ 局限性**

仅使用 ViT‑small 规模模型和三种代表性预文本任务，未探索更大模型或其他医学模态；对同时需要局部与全局特征的任务的最佳预文本策略仍未给出，且缺乏对不同分辨率、模态的更广泛验证。

---

## 167. BioShield: A Context-Aware Firewall for Securing Bio-LLMs

**arXiv ID:** 2603.22612 | [PDF](https://arxiv.org/pdf/2603.22612v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 168. AwesomeLit: Towards Hypothesis Generation with Agent-Supported Literature Research

**arXiv ID:** 2603.22648 | [PDF](https://arxiv.org/pdf/2603.22648v1)

**作者:** Zefei Xie `[一作]` (University Of Nottingham), Kai Xu `[通讯]` (University Of Nottingham)

**通讯引用:** 4999 | [OpenAlex ID](https://openalex.org/A5031386469)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款名为 AwesomeLit 的人机协作可视化系统，用来帮助初学者在不熟悉的研究领域中进行文献检索、筛选与假设生成。

**💡 创新点**

创新点包括：① 透明工作流（节点化、可暂停/可编辑的 LLM 推理过程）；② 查询探索树（可视化检索路径和主题演进）；③ 语义相似度视图（利用文本嵌入+UMAP 可视化论文聚类并支持交互过滤）。

**🔧 技术方法**

核心技术包括：OpenAI GPT‑5‑mini（生成查询与摘要）、OpenAI text‑embedding‑3‑small（嵌入生成）、UMAP（降维投影）、arXiv API（文献检索）以及前端可视化框架（如 D3.js 或类似库）实现节点图、树图和散点图。

**📊 数据集**

使用了 arXiv 上的最新学术论文数据，并结合用户在探索过程中的生成查询和反馈进行动态检索；实验数据主要来自七名计算机科学学生的交互日志与问卷。

**📈 对比分析**

通过与传统 LLM 方案（如 OpenAI “deep research”）对比，采用定性问卷（7 分量表）评估易用性、透明度、搜索精准度等指标，结果显示透明工作流与查询树的平均得分分别为 6.00 与 6.57，说明用户对系统的信任和灵活性满意度较高；未给出客观检索性能指标。

**⚠️ 局限性**

主要局限在于系统仍需用户大量手动干预，缺乏自适应推荐与自动化优化；对不同学科数据源的支持有限，且未在大规模量化实验中验证检索效果。

---

## 169. CPU Simulation with Ranked Set Sampling and Repeated Subsampling

**arXiv ID:** 2603.22598 | [PDF](https://arxiv.org/pdf/2603.22598v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 170. PIVM: Diffusion-Based Prior-Integrated Variation Modeling for Anatomically Precise Abdominal CT Synthesis

**arXiv ID:** 2603.22626 | [PDF](https://arxiv.org/pdf/2603.22626v1)

**作者:** Dinglun He `[一作]`, Ye Duan `[通讯]` (Clemson University)

**通讯引用:** 12997 | [OpenAlex ID](https://openalex.org/A5101430356)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 Prior-Integrated Variation Modeling (PIVM) 框架，在图像空间内基于器官平均 HU 先验和分割标签对残差进行扩散建模，从而生成高保真腹部 CT 图像并实现高效的逐切片 3D 体积生成。

**💡 创新点**

创新点在于将器官级平均强度先验与分割标签结合，只对残差进行扩散学习，保留完整 HU 范围并显著提升器官边界精度，同时采用逐切片序列化生成，兼顾空间连贯性与计算效率。

**🔧 技术方法**

使用了基于 U‑Net 的 Denoising Diffusion Probabilistic Model (DDPM)，差异建模与条件扩散技术，并与 MAISI、CDDPM 进行对比。

**📊 数据集**

使用了 TotalSegmentator 数据集（共 865 个腹部 CT 体积，12853 切片），涉及 34 种主要器官的分割标签。

**📈 对比分析**

通过 SSIM、FID、mIoU、Dice 等指标以及专家评估进行比较；PIVM 在 SSIM/FID 上优于 CDDPM 与 MAISI，并在下游分割任务中将 Dice 提升至 0.53（相比 0.38/0.34），表现更佳。

**⚠️ 局限性**

限制在于小血管、膈肌等细部仍出现伪影，受限于分割标签分辨率和采样平衡，且目前尚未实现完整的 3D 体积生成。

---

## 171. Causal Discovery in Action: Learning Chain-Reaction Mechanisms from Interventions

**arXiv ID:** 2603.22620 | [PDF](https://arxiv.org/pdf/2603.22620v1)

**作者:** Panayiotis Panayiotou `[一作]` (University of Bath), Özgür Şimşek `[通讯]` (University of Bath)

**通讯引用:** 1614 | [OpenAlex ID](https://openalex.org/A5079000862)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了在链式反应系统中通过单一阻断干预实现因果结构的可识别与恢复

**💡 创新点**

在单向、单父亲的树形链式模型下证明了只需单一阻断干预即可完全识别因果图，并给出最小估计器及其指数错误衰减与对数样本复杂度的理论保证

**🔧 技术方法**

利用单节点阻断干预、祖先矩阵构建、转移约简、以及极值概率分析的理论方法，结合蒙特卡洛模拟与物理仿真验证

**📊 数据集**

在六种基于Pymunk的Rube Goldberg风格物理仿真环境以及对应的合成SCM中进行实验

**📈 对比分析**

与两种观测启发式基线（碰撞检测+时间先后）以及PC算法比较，结果表明只需1–2次干预即可在95%以上精度恢复结构，而观测基线在噪声极大时表现差距显著

**⚠️ 局限性**

假设因果图为有向树、干预为单节点、无测量噪声；无法处理多父节点、干预集组合以及测量错误等情形

---

## 172. "Chasing Shadows": Understanding Personal Data Externalization and Self-Tracking for Neurodivergent Individuals

**arXiv ID:** 2603.22609 | [PDF](https://arxiv.org/pdf/2603.22609v1)

**作者:** Tanya Rudberg Selin `[一作]` (IT University of Copenhagen), Søren Knudsen `[通讯]` (IT University of Copenhagen)

**通讯引用:** 1854 | [OpenAlex ID](https://openalex.org/A5020030553)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对自闭症和 ADHD 个人进行两阶段的定性研究，探讨他们如何创建、记录与反思关于掩码（masking）的个人数据；

**💡 创新点**

首次系统关注神经多样性个体在自我跟踪过程中的情感负担、过度思考和对情境的依赖性，并提出三维情感模型和同行支持的设计启示；

**🔧 技术方法**

采用访谈、工作坊、绘图与写作外化活动、个人日志记录和反思性访谈；使用反思性主题分析（reflexive thematic analysis）；

**📊 数据集**

研究数据包括六位自闭症/ADHD 参与者在工作坊中的绘图、写作外化作品、个人日志和访谈转录；

**📈 对比分析**

本研究未采用对比实验或量化指标，主要通过主题分析呈现参与者体验，没有性能评估；

**⚠️ 局限性**

样本规模小、仅来自研究者网络，缺乏多样性与代表性，且研究主要基于自我报告，可能存在主观偏差；

---

## 173. MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping

**arXiv ID:** 2603.22650 | [PDF](https://arxiv.org/pdf/2603.22650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 174. LGSE: Lexically Grounded Subword Embedding Initialization for Low-Resource Language Adaptation

**arXiv ID:** 2603.22629 | [PDF](https://arxiv.org/pdf/2603.22629v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 175. Simple but not Simpler: A Surface-Sliding Method for Finding the Minimum Distance between Two Ellipsoids

**arXiv ID:** 2603.22683 | [PDF](https://arxiv.org/pdf/2603.22683v1)

**作者:** Dariush Amirkhani `[一作]` (Laurentian University), Junfeng Zhang `[通讯]` (Laurentian University)

**通讯引用:** 13218 | [OpenAlex ID](https://openalex.org/A5100427790)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种基于表面滑动的迭代方法，用于计算两椭球之间的最小距离及其对应的接触点。

**💡 创新点**

创新点在于直接在椭球的 θ–ϕ 参数空间内滑动点，无需高阶多项式求解或支持映射，计算过程简单、收敛稳定且对几何对比不敏感。

**🔧 技术方法**

使用迭代表面滑动技术、参数化椭球表面、切向投影与自适应步长控制等数值方法。

**📊 数据集**

采用设计的仿真系统（System I、II、III）中的多组椭球尺寸、位置与姿态作为实验数据，未使用公开数据集。

**📈 对比分析**

与传统移动球法、GJK 等方法比较，实验表明收敛速度快、误差低、稳定性强，对椭球尺寸与形状的对比不敏感，整体性能优于现有方法。

**⚠️ 局限性**

仅在椭球相互接触或重叠时需要额外处理，且尚未针对大规模多体并行实现或高维扩展进行研究。

---

## 176. Transfer learning via interpolating structures

**arXiv ID:** 2603.22621 | [PDF](https://arxiv.org/pdf/2603.22621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 177. Energy Detection for Cognitive Radio with Distributional Uncertainty and Signal Variety under Nonlinear Expectation Theory

**arXiv ID:** 2603.22659 | [PDF](https://arxiv.org/pdf/2603.22659v1)

**作者:** Jialiang Fu `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Wen-Xuan Lang `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于非线性期望理论的能量检测模型，考虑信号和噪声分布的不确定性，推导出误检与误报概率的上下界；

**💡 创新点**

创新点在于引入G-正态分布和子线性期望框架，统一处理信号与噪声的不确定性，并给出对不同衰落信道（常数、Rayleigh、Rician、Nakagami）下误报概率的渐近分析；

**🔧 技术方法**

采用子线性期望理论、G-正态分布性质、G-热方程估计、概率极大/极小化等数学工具；

**📊 数据集**

无具体数据集，实验基于理论仿真与数值积分；

**📈 对比分析**

与传统假设噪声与信号为高斯分布的经典能量检测结果比较，得到保守与激进两种场景下的误报上界与误检上界，证明理论界限能覆盖传统方法的结果，数值仿真显示误报概率随信噪比、样本数等参数的变化趋势；

**⚠️ 局限性**

局限性包括：需要已知信号幅值范围[σ_X,σ_X]；对G-正态分布的理论计算复杂；实际系统中可能难以获得子线性期望所需的全部不确定性信息；模型未对多路复用或协作检测等更复杂场景做进一步扩展。

---

## 178. MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation

**arXiv ID:** 2603.22677 | [PDF](https://arxiv.org/pdf/2603.22677v1)

**作者:** Di Zhu `[一作]` (Stevens Institute of Technology), Zixuan Li `[通讯]` (Columbia University)

**通讯引用:** 3793 | [OpenAlex ID](https://openalex.org/A5100398261)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并公开了 MuQ-Eval，一种基于 MuQ 预训练特征和轻量预测头的 AI 生成音乐样本级质量评估指标，并在 MusicEval 数据集上系统评估其性能。

**💡 创新点**

主要创新点包括：① 将“深度特征 + 质量注释”方法应用于音乐生成评估，首次实现全开源且可实时的样本级指标；② 系统化 Ablation 证明冻结 MuQ 特征已足够，无需复杂训练；③ 通过 LoRA 在极少量数据（150–250 条）下实现个性化质量评估器；④ 对信号级和结构级失真进行控制降质实验，揭示指标对不同失真类型的选择性敏感性。

**🔧 技术方法**

使用的技术包括：MuQ‑310M 音乐自监督模型做特征提取；attention pooling + 2 层 MLP 预测头；多种训练策略（MSE、Gaussian‑softened ordinal CE、LoRA 适配、对比损失、不确定性加权）；BF16 混合精度训练；单卡 RTX 4080 35 ms 推理。

**📊 数据集**

使用数据集：MusicEval（2,748 条生成音乐片段，31 个 TTM 系统，13,740 条专家 MOS 评分）以及内部自行生成的压缩、噪声、音高/节拍失真样本。

**📈 对比分析**

与 FAD、Audiobox Aesthetics 等基准对比；在 5 折 CV 下，冻结 MuQ 模型系统级 SRCC 0.957，LoRA 模型 0.960，接近闭源 DORA‑MOS（0.988）；句级 SRCC 0.838；训练策略增量均未超过 Δ≥0.02；数据效率实验表明 LoRA 可用 150–250 条样本匹配冻结模型；计算成本低（约 35 ms/10 s，3 GB VRAM）。

**⚠️ 局限性**

局限性：① 对音高/节拍等结构性失真敏感度低；② 仅在 MusicEval 上验证，跨数据集泛化未知；③ 文本‑音频对齐头性能低，需文本条件架构；④ 仍依赖人工 MOS 注释；⑤ 对极高质量样本的细粒度区分能力有限。

---

## 179. To Agree or To Be Right? The Grounding-Sycophancy Tradeoff in Medical Vision-Language Models

**arXiv ID:** 2603.22623 | [PDF](https://arxiv.org/pdf/2603.22623v1)

**作者:** OFM Riaz Rahman Aranya `[一作]` (University of Texas at San Antonio), Kevin Desai `[通讯]` (University of Texas at San Antonio)

**通讯引用:** 468 | [OpenAlex ID](https://openalex.org/A5076318084)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对六种7–8B参数的医学视觉语言模型在三大医学VQA数据集上，首次同时评估幻觉与顺从性（sycophancy）这两种安全缺陷。

**💡 创新点**

提出了L‑VASE（对logit空间的幻觉度量）、CCS（按置信度加权的顺从性评分）和CSI（整合两项缺陷的FMEA启发式安全指数），并首次揭示了模型在扎根（grounding）与顺从性之间的显著负相关。

**🔧 技术方法**

利用logit‑级对比熵计算幻觉、置信度加权的顺从度评估、几何平均构造CSI，并通过温度τ=1.0的随机采样与贪婪解码进行推理，采用专家纠正、群体共识与权威指令三种社会压力类型进行挑战。

**📊 数据集**

使用VQA‑RAD、SLAKE和PathVQA这三大医学VQA基准，共计1151个测试样本。

**📈 对比分析**

对六个模型（三类通用、三类医学专业）在L‑VASE、CCS和CSI上进行横向比较；结果显示无模型CSI>0.35，最优模型CSI为0.339，且模型在低幻觉率与低顺从率之间呈显著负相关。

**⚠️ 局限性**

局限性包括：仅评估7–8B规模模型；压力测试为单轮，未考虑多轮交互；CSI阈值仅为解释框架，未与临床风险量化对齐；未涉及更大或专有模型的表现。

---

## 180. Bridging the Know-Act Gap via Task-Level Autoregressive Reasoning

**arXiv ID:** 2603.22619 | [PDF](https://arxiv.org/pdf/2603.22619v1)

**作者:** Jihyun Janice Ahn `[一作]` (Penn State), Wenpeng Yin `[通讯]` (Penn State)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在大型语言模型中引入任务级自回归决策的框架，以在生成答案前先验证问题是否有效。

**💡 创新点**

创新点在于将验证与回答分离为两个阶段，并通过自蒸馏训练统一判别与生成能力，同时使用控制标记实现任务切换。

**🔧 技术方法**

技术包括控制标记的任务级自回归训练、专门的控制‑token损失函数、数据重加权、以及自蒸馏与教师蒸馏方式。

**📊 数据集**

使用数据集为 FaultyScience benchmark（约1.5万条含错误科学问题）以及标准推理基准BBH、GPQA。

**📈 对比分析**

通过与GPT‑4、Mixtral、Llama3.3、Qwen等模型对比，所提 DeIllusionLLM 在无提示生成条件下的 FaultyScience 成绩提升至 67.8% 以上，远超原始模型（约10%），同时保持 BBH/GPQA 近似原始性能。

**⚠️ 局限性**

局限性包括仍存在部分错误模式（多选误选、错误修复、假设注入）以及对极端异常的识别不足，且对训练数据质量与标签一致性的依赖。

---

## 181. Satellite-Terrestrial Spectrum Sharing in FR3 through QoS-Aware Power Control and Spatial Nulling

**arXiv ID:** 2603.22615 | [PDF](https://arxiv.org/pdf/2603.22615v1)

**作者:** Maria Tsampazi `[一作]` (Northeastern University), Tommaso Melodia `[通讯]` (Northeastern University)

**通讯引用:** 19947 | [OpenAlex ID](https://openalex.org/A5054337759)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评估了一种结合QoS感知功率控制与干扰消除的地面网络波束成形与功率管理框架，以降低地面基站向卫星泄漏干扰，同时保证地面用户质量。

**💡 创新点**

提出了带贝尔形效用函数的QoS感知功率控制以及通过λ参数调节干扰消除强度的联合优化方法，实现了能效、功率与公平性三者的平衡。

**🔧 技术方法**

利用SVD与零空间投影实现干扰消除，采用凸优化（CVX/SDP）框架进行功率与波束权重联合优化，并设计效用函数以衡量QoS满足与功率消耗。

**📊 数据集**

使用真实卫星位置与地面网络拓扑（Table <ref>）模拟数据，进行离散时间点的波束成形与功率分配仿真。

**📈 对比分析**

与单独干扰消除、无消除+功率控制及不同λ值的干扰消除方案进行CDF、RSS衰减与JFI比较；联合优化在降低INR、保持公平性与能效方面优于单一干扰消除，且能避免极端RSS衰减。

**⚠️ 局限性**

在极端RSS衰减可能导致某些用户无法接入；联合优化需要经验调参λ与阈值；仅针对静态场景，未考虑高速移动卫星的实时自适应需求。

---

## 182. Variable-Resolution Virtual Maps for Autonomous Exploration with Unmanned Surface Vehicles (USVs)

**arXiv ID:** 2603.22667 | [PDF](https://arxiv.org/pdf/2603.22667v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 183. Dress-ED: Instruction-Guided Editing for Virtual Try-On and Try-Off

**arXiv ID:** 2603.22607 | [PDF](https://arxiv.org/pdf/2603.22607v1)

**作者:** Fulvio Sanguigni `[一作]` (University of Modena), Rita Cucchiara `[通讯]` (University of Modena)

**通讯引用:** 19995 | [OpenAlex ID](https://openalex.org/A5030948871)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Dress-ED数据集与统一的多模态扩散模型，用自然语言指令实现虚拟试衣与试脱的可控编辑；

**💡 创新点**

1）首次构建包含衣物-人物配对且带指令的四元组数据集；2）设计M<sup>LLM</sup>+Connector+DiT框架，实现文本与视觉双向约束；

**🔧 技术方法**

多模态大型语言模型（InternVL‑3.5）、扩散模型（Stable Diffusion 3、FLUX.2 Klein）、VAE、Transformer Connector、Denoising Diffusion Transformer（DiT）；

**📊 数据集**

Dress‑ED（146k+样本）和基础数据集Dress Code；

**📈 对比分析**

与多种基线（FLUX.2 Klein、Qwen‑Image‑Edit、Any2AnyTryon、CatVTON）对比，实验显示在VTON、VTOFF及不同编辑类型下，FID、KID、DISTS均最低，DINO‑I最高，说明生成质量和指令遵从度显著提升；

**⚠️ 局限性**

依赖自动化多模态管道，对极端复杂结构或稀有材质的编辑仍可能出现误差，且对非标准服装类别或极端姿态的泛化尚待验证。

---

## 184. ParlayMarket: Automated Market Making for Parlay-style Joint Contracts

**arXiv ID:** 2603.22596 | [PDF](https://arxiv.org/pdf/2603.22596v1)

**作者:** Ranvir Rana `[一作]` (Kaleidoscope Blockchain), Pramod Viswanath `[通讯]` (Princeton University)

**通讯引用:** 28096 | [OpenAlex ID](https://openalex.org/A5053980484)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 ParlayMarket，一个支持组合合约的自动化市场做市商，通过共享的二阶指数族模型实现对 2^M 组合的价格和风险管理。

**💡 创新点**

创新点在于使用配对指数族（Ising）模型压缩联合分布至 O(M^2) 参数，并通过“影子交易”将信息在所有合约间传播，从而在不产生指数风险的前提下学习相关性。

**🔧 技术方法**

技术主要包括基于 LMSR 的成本函数、Ising 统计模型、随机梯度下降更新、loopy BP 推断以及基于组合支付函数的复合似然。

**📊 数据集**

使用的数据集包括合成的高斯评分模型以及 2026 年 3 月 7 日 Kalshi NBA 赛季的历史交易记录。

**📈 对比分析**

与独立 LMSR、单一独立市场以及三种 oracle（对偶、Gaussian、pairwise）对比，ParlayMarket 在合成实验中实现指数级 per‑market 损失下降，并在 Kalshi 回放中获得最高 Sharpe 比例。

**⚠️ 局限性**

局限在于只能处理已预先构造为二进制、无互斥约束的事件，需先行结构化或预处理才能应用。

---

## 185. CAM3R: Camera-Agnostic Model for 3D Reconstruction

**arXiv ID:** 2603.22631 | [PDF](https://arxiv.org/pdf/2603.22631v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 186. Designing a Meta-Reflective Dashboard for Instructor Insight into Student-AI Interactions

**arXiv ID:** 2603.22674 | [PDF](https://arxiv.org/pdf/2603.22674v1)

**作者:** Boxuan Ma `[一作]` (Kyushu University), Shin'Ichi Konomi `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款面向教师的 Meta-Reflective Dashboard，能够在不暴露完整聊天记录的前提下，将学生与 AI 的交互会话生成结构化摘要与风险信号，帮助教师快速把握学生的求助过程和 AI 使用模式。

**💡 创新点**

创新点在于：① 将学生–AI 对话转换为隐私友好的会话级摘要与风险提示；② 默认展示高层次信息，支持按需查看细节；③ 结合共创设计，针对教师工作负担、隐私与可操作性三大挑战提出解决方案。

**🔧 技术方法**

技术手段包括：① 基于大型语言模型的 Reflection AI 用于分析对话并生成摘要；② 数据过滤器与隐私规则对对话进行脱敏；③ 与 LMS 通过 LTI 与单点登录集成的插件；④ 使用提示类型分类与层级聚合的可视化展示。

**📊 数据集**

使用的数据集为 2024 年一门入门编程课程的 22 名学生共 230 次帮助会话的聊天记录，已通过脱敏处理后用于原型评估。

**📈 对比分析**

通过对 6 名教师与 8 名学生的定性/定量问卷与小组访谈进行形式评估，教师对仪表盘的可解释性、实用性、信任度与隐私可接受度均达 4.0–4.7 分（5 分制）；学生对摘要的学习帮助与隐私舒适度均超过 4.0 分；与传统完整日志访问相比，仪表盘显著减少教师工作量并提升隐私保护。

**⚠️ 局限性**

局限性：① 样本量和单课程范围有限，缺乏对不同学科与机构的泛化验证；② 仅进行形式评估，缺乏真实课堂部署与长期影响研究；③ 依赖 LLM 生成摘要，可能存在不完整或误判，需要进一步验证信号准确性与公平性。

---

## 187. Do Consumers Accept AIs as Moral Compliance Agents?

**arXiv ID:** 2603.22617 | [PDF](https://arxiv.org/pdf/2603.22617v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 188. Improving LLM Predictions via Inter-Layer Structural Encoders

**arXiv ID:** 2603.22665 | [PDF](https://arxiv.org/pdf/2603.22665v1)

**作者:** Tom Ulanovski `[一作]` (Tel Aviv University), Maya Bechler-Speicher `[通讯]` (Meta)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Inter‑Layer Structural Encoders (ILSE)，通过将 LLM 的所有内部层表示聚合成单一表示来提升下游任务性能。

**💡 创新点**

创新点在于将层间信息建模为稠密或稀疏的图结构（如 Cayley‑Encoder、Set‑Encoder、FC‑Encoder），并利用 Cayley 图的正则性与无瓶颈特性实现高效信息传播，从而在不改动基础模型的情况下显著提升性能。

**🔧 技术方法**

技术包括图神经网络（GIN/GCN）、DeepSets、Cayley 图扩散、线性/非线性融合层，以及对不同 LLM 的冻结层表示进行后置训练的方式。

**📊 数据集**

实验使用 MTEB 框架下的 13 个分类任务（Banking77、Emotion、MTOP Domain、MTOP Intent、Poem Sentiment 等）与 8 个语义相似度任务（STSBenchmark、STS12–16、BIOSSES、SICK‑R）以及 9 个 14M–8B 参数的预训练 LLM 进行评估。

**📈 对比分析**

与 Last‑Layer、Best‑Layer、Weighted、MLP、DWAtt 等基线比较，ILSE 在所有任务中平均提升 20–40% 的准确率或 25–44% 的相似度分数；在零样本、少样本（32–1024 样本）以及不同 LLM 尺寸下同样表现出显著优势，并且仅需 0.1% 左右的额外参数。

**⚠️ 局限性**

局限性包括仅在 8B 级别模型上验证，未测试更大规模模型；仅在冻结 LLM 上进行后置训练，未探究在预训练阶段嵌入 ILSE 的效果；层到图节点的映射采用随机/固定方式，可能未能最优利用任务信息；并且实验聚焦于分类与相似度任务，其他下游任务需进一步验证。

---

## 189. When Data Protection Fails to Protect: Law, Power, and Postcolonial Governance in Bangladesh

**arXiv ID:** 2603.22637 | [PDF](https://arxiv.org/pdf/2603.22637v1)

**作者:** Pratyasha Saha `[一作]` (University of Illinois Urbana Champaign), Sharifa Sultana `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5103026409)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统梳理并对比孟加拉国2025年颁布的个人数据保护条例、网络安全条例与国家数据治理条例，分析它们在法律文本、监管机构与执法机制上的交叉与差异。

**💡 创新点**

提出“结构性失败逻辑”框架，将权威、机构与技术基础设施与个人权利实践三轴交叉，揭示法律文本与实际治理之间的深层不匹配；首次将“人类桥梁”概念纳入数据保护研究，凸显非正式中介在数据流通中的关键角色。

**🔧 技术方法**

采用定性法律与制度分析方法，使用开放式编码与主题化归纳，构建结构性失败逻辑的三轴维度；同时对三部条例文本进行系统性对照与对比分析。

**📊 数据集**

主要数据集包括三部孟加拉国最新条例的全文（经翻译）、相关宪法及行业监管法规、公开披露的个人数据泄露与安全事件报道，以及专家访谈与新闻报道。

**📈 对比分析**

方法上并未进行量化对比，而是通过文本对照与主题分析来评估法规之间的重叠、空白与执行前景；研究结果通过案例层面阐明法律文本与实践之间的脱节，未给出传统意义上的“性能”指标。

**⚠️ 局限性**

局限主要在于依赖公开二手资料与文本分析，缺乏现场观察与深度访谈；数据泄露与执法案例的获取受限于政治敏感性，可能导致对实际执行细节的欠缺；研究聚焦孟加拉国，结果不易直接推广至其他地区。

---

## 190. Emotional Support with Conversational AI: Talking to Machines About Life

**arXiv ID:** 2603.22618 | [PDF](https://arxiv.org/pdf/2603.22618v1)

**作者:** Olivia Yan Huang `[一作]` (University of Illinois Urbana-Champaign), Sharifa Sultana `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 654 | [OpenAlex ID](https://openalex.org/A5103026409)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Reddit 11 个子版块中近 5,370 条关于 LLM 的讨论进行定性主题分析，探讨用户如何通过与 LLM 互动获取情感支持，并揭示这种支持在社区中的社会技术协商过程。

**💡 创新点**

将情感支持重新构建为一种社会技术协商过程，首次在自我决定理论框架下系统阐释 LLM 如何满足相关性、能力与自主性三大心理需求，并深入剖析社区对 AI 支持的合法化与挑战。

**🔧 技术方法**

主要采用人机交互与社会技术分析方法——主题分析、代码生成与主题归纳，并辅以自我决定理论作为理论视角；未使用新的深度学习模型或算法。

**📊 数据集**

来源于 11 个子版块（如 r/mentalhealth、r/Christianity、r/CharacterAI 等）的 500 条最新帖子及其一级评论，合计约 5,370 条条目，构成定性分析数据集。

**📈 对比分析**

研究采用的是定性方法，没有与算法或模型的性能做直接比较；结果通过归纳主题呈现，描述了用户对 LLM 情感支持的积极与消极体验，并未给出可量化指标。

**⚠️ 局限性**

局限包括：仅基于公开 Reddit 讨论，样本自选且缺乏隐私与文化多样性；依赖自述，未进行临床验证或长期跟踪；缺乏跨文化或非英语环境的验证，可能忽视不同群体对 AI 情感支持的独特反应。

---

## 191. Semi-Automated Threat Modeling of Cloud-Based Systems Through Extracting Software Architecture from Configuration and Network Flow

**arXiv ID:** 2603.22603 | [PDF](https://arxiv.org/pdf/2603.22603v1)

**作者:** Nicholas Pecka `[一作]` (University of North Texas), Renee Bryce `[通讯]` (University of North Texas)

**通讯引用:** 1830 | [OpenAlex ID](https://openalex.org/A5077471766)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

自动从运行时观察（配置文件+网络流量）推断云原生系统架构，生成平台无关的威胁模型，并在此基础上持续进行威胁识别与缓解建议。

**💡 创新点**

核心创新在于：① 通过将静态配置与实时网络流量相结合，实时恢复系统架构图；② 设计了平台无关的抽象（域、组件、接口、访问策略、流量）；③ 内置机器学习组件识别与对应威胁，使得在裸机、Kubernetes 与 AWS 上均可发现 ML 相关架构威胁。

**🔧 技术方法**

技术手段包括：静态配置解析（kubectl API、boto3、SSH/Docker inspect）、网络流量采集（Cilium Hubble/Falco、tcpdump、VPC Flow Logs）、图结构构建与映射、基于 STRIDE/ATT&CK/OWASP ML 的威胁检测算法、风险评分与自动化修复建议生成。

**📊 数据集**

数据集为自建的软件供应链系统（8 个组件，3 个信任区），在裸机、Kubernetes 集群与 AWS 上各部署一次，并手动注入 17 种架构威胁（10 个传统 + 7 个 ML 相关）。

**📈 对比分析**

与 6 款主流工具（Trivy、Checkov、KubeLinter、Kubescape、Docker Bench、Prowler）做对比：我们的方法在所有平台上 100% 发现 17 种威胁，传统工具仅 6–47% 覆盖且 0% 检测 ML 相关威胁。性能方面：Kubernetes 平台总扫描耗时 32.65 s（含 Hubble/Falco），裸机 191.84 s，AWS 约 60 s（含 VPC Flow Log 延迟）。

**⚠️ 局限性**

局限性：仅识别架构层面威胁，无法发现应用级漏洞、代码缺陷或供应链制品攻击；实验仅覆盖单一供应链架构，缺乏大规模或无服务器等多样化系统的验证；未提供完整的自动修复流程，后续需扩展至更大规模部署与多租户环境。

---

## 192. Computing and Enumerating Minimal Common Supersequences Between Two Strings

**arXiv ID:** 2603.22591 | [PDF](https://arxiv.org/pdf/2603.22591v1)

**作者:** Braeden Sopp `[一作]` (Montana State University), Binhai Zhu `[通讯]` (Montana State University)

**通讯引用:** 2366 | [OpenAlex ID](https://openalex.org/A5070925973)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了在两字符串之间线性时间（O(n)）求解最小公共超序列（MCS）的算法，并给出了一种基于图的枚举方法，能够在预处理O(n³)、空间O(n²)的前提下，以O(n)的延迟枚举所有MCS；对k个长度不超过n的字符串，提供了O(kn(log k+log n))的求解时间。

**💡 创新点**

创新点在于：①利用左右嵌入的概念与简单扫描实现两字符串MCS的线性求解；②构造双部图AB，将所有MCS映射为st-路径，从而实现高效枚举；③提出MergeRightEmbeddings等数据结构和算法，实现k字符串MCS的O(NlogN)时间求解。

**🔧 技术方法**

主要技术包括：左右嵌入、区间填充（fill）与匹配、动态规划求解嵌入长度、图结构构造与深度优先搜索、二分搜索的字符出现查询、以及用于快速查找下一个字符的索引数组。

**📊 数据集**

本文为理论研究，没有使用具体实验数据集，所有结果均为理论复杂度分析。

**📈 对比分析**

通过理论分析，提出的方法在最坏情况下的时间复杂度分别为O(n)（两字符串求MCS）、O(n³)预处理+O(n)延迟枚举、O(NlogN)（k字符串求MCS）。与已有的SCS、LCS算法相比，显著降低了两字符串MCS的时间复杂度，并在枚举方面提供了可接受的空间与延迟。

**⚠️ 局限性**

局限性包括：未针对带约束的MCS（如不包含给定子序列P）给出高效算法；缺乏实验验证；在k>2时，时间复杂度仍随k线性增长，且在实际大规模数据中可能仍不够高效。

---

## 193. Three Years with Classroom AI in Introductory Programming: Shifts in Student Awareness, Interaction, and Performance

**arXiv ID:** 2603.22672 | [PDF](https://arxiv.org/pdf/2603.22672v1)

**作者:** Boxuan Ma `[一作]` (Kyushu University), Shin'ichi Konomi `[通讯]` (Kyushu University)

**通讯引用:** 2633 | [OpenAlex ID](https://openalex.org/A5071736649)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对一门日本本科初级Python课程进行三年课堂实验，追踪学生对生成式AI的熟悉度、使用方式和学习成绩变化。

**💡 创新点**

首次提供长期纵向研究，揭示学生AI交互从单一实现到验证、调试多步工作流程的转变，强调课程设计需指导AI的有效使用。

**🔧 技术方法**

使用ChatGPT类生成式AI，结合问卷调查、学生-AI对话日志编码以及课程成绩数据。

**📊 数据集**

包含三届学生问卷、约10,632条学生-AI对话记录（抽样2,782条编码）以及每周作业和期末成绩记录。

**📈 对比分析**

通过问卷纵向比较、对话日志主题编码和课程成绩统计（Kruskal-Wallis检验）发现AI使用率提高但作业和期末成绩基本保持不变。

**⚠️ 局限性**

样本局限于单一日本高校，缺乏因果推断，且对话日志仅记录平台内使用，可能忽视其他求助渠道。

---

## 194. From Overload to Convergence: Supporting Multi-Issue Human-AI Negotiation with Bayesian Visualization

**arXiv ID:** 2603.22766 | [PDF](https://arxiv.org/pdf/2603.22766v1)

**作者:** Mehul Parmar `[一作]` (Asian Institute of Technology), Chaklam Silpasuwanchai `[通讯]` (Asian Institute of Technology)

**通讯引用:** 672 | [OpenAlex ID](https://openalex.org/A5082598678)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实验验证了一种基于贝叶斯不确定性可视化的决策支持工具，用以辅助人类在多问题AI谈判中的信息管理与决策

**💡 创新点**

提出了“认知鸿沟与认知支援”双通道可视化模式，并在三、五、七问题场景中发现了“平台-悬崖”效应，证明了认知支援可显著缓解高维度谈判中的认知负荷与绩效下滑

**🔧 技术方法**

贝叶斯推理（用于更新AI偏好概率）、熵测度（衡量不确定性）、可视化技术（热图与进展条）以及GPT‑4语言模型作为AI谈判代理

**📊 数据集**

基于房产租赁的整合式多问题任务，构造了16个问题、7个选项的效用矩阵，实验采用32名受试者的交互记录

**📈 对比分析**

与基线无可视化条件对比，采用2×4（界面×维度）混合设计；结果显示决策支持在人类报酬、谈判轮数、第一次键入延迟、熵等指标上有显著改善（p<0.001），且维度效应被显著抑制，平均提升约15%的人类收益且保持公平分配

**⚠️ 局限性**

局限包括单一整合式任务设置、固定AI策略（GPT‑4效用最大化）、未平衡界面顺序、样本量与维度步长有限，难以直接推广到零和、异重权重或多轮真实环境谈判

---

## 195. Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning

**arXiv ID:** 2603.22758 | [PDF](https://arxiv.org/pdf/2603.22758v1)

**作者:** WonJun Moon `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1533 | [OpenAlex ID](https://openalex.org/A5029469141)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SlotCurri，通过从少到多的 slot 逐步学习并结合结构化重建损失与循环推理，解决视频对象中心学习中的过度碎片化问题。

**💡 创新点**

① 架构化的 slot 课程学习；② 基于重建误差的 slot 生长与噪声初始化；③ 结构感知 SSIM 损失提升边界清晰度；④ 循环推理保证时序一致性。

**🔧 技术方法**

使用 Slot Attention + SlotContrast 的对比学习框架，MSE+3D‑SSIM 结构化重建损失，以及前向+后向循环推理和加速 slot 调度。

**📊 数据集**

在 YouTube‑VIS 2021、MOVi‑C、MOVi‑E 三个视频数据集以及 COCO 2017 进行评估。

**📈 对比分析**

与 SlotContrast、SAVi、SOLV 等前沿方法对比，FG‑ARI 在 YouTube‑VIS 提升 6.8 点、MOVi‑C/‑E 8.3 点，整体保持或超越最优性能，并在不同 slot 数量下表现鲁棒。

**⚠️ 局限性**

仍在小物体的欠碎片化挑战、对 curriculum 设定需人工调优，且在极端多小对象场景下效果有限。

---

## 196. SOUPLE: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts

**arXiv ID:** 2603.22732 | [PDF](https://arxiv.org/pdf/2603.22732v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 197. KALAVAI: Predicting When Independent Specialist Fusion Works -- A Quantitative Model for Post-Hoc Cooperative LLM Training

**arXiv ID:** 2603.22755 | [PDF](https://arxiv.org/pdf/2603.22755v1)

**作者:** Ramchand Kumaresan `[一作]` `[通讯]` (Murai Labs), Ramchand Kumaresan (Murai Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让不同贡献者在共享基准检查点上各自独立微调领域专家模型，然后使用轻量化的MoE路由器进行后置融合，得到的联合模型在各领域上均优于任何单一专家。

**💡 创新点**

创新点：①提出完全无通信的协同训练协议，只需共享初始检查点；②发现专家模型的“分歧”与融合收益呈线性关系，可在训练前预测效益；③证明即使是简单的线性路由也能实现域级oracle级路由；④在多语言、专业领域、低资源场景中验证协议的有效性与可扩展性。

**🔧 技术方法**

技术手段：MoE（混合专家）融合、轻量化路由训练（500步微调）、共享初始化、可选冻结层、线性/MLP路由器、对齐评估和基准对比；使用Pythia系列模型作为基础。

**📊 数据集**

数据集：英语代码（CodeSearchNet）、科学（SciQ）、小说（PG‑19）；医疗、法律、专利文本；非英语多语言（Tamil、Yoruba、Welsh）及代码；以及对应的评估集。

**📈 对比分析**

评估方式：对比最佳专家、均权平均、单机全混合训练、权重平均等；在410M、1B、6.9B规模下得到+7.7%、+7.5%、+6.5%相对于最佳专家的提升；跨语言实验提升+21.8%；私有领域提升+10.2%；路由器的学习必不可少，均权路由甚至略逊于最佳专家。

**⚠️ 局限性**

局限性：①推理时需并行跑所有专家，计算量按专家数线性增长；②对共享初始检查点的严格依赖，若不匹配会降低路由效果；③对不同架构的泛化性有限（仅验证Pythia与一次Qwen）；④低资源域需要足够训练样本，极少样本时效果下降；⑤下游任务提升有限，主要体现在困惑度；⑥实验规模停留在6.9B，未知更大规模的表现；⑦未覆盖多模态或非文本任务。

---

## 198. CIPL: A Target-Independent Framework for Channel-Inversion Privacy Leakage in Agents

**arXiv ID:** 2603.22751 | [PDF](https://arxiv.org/pdf/2603.22751v1)

**作者:** Tao Huang `[一作]` (Minjiang University), Jiayang Meng `[通讯]` (Renmin University of China)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Channel Inversion for Privacy Leakage (CIPL)框架，用于统一评估大型语言模型代理系统在不同观察通道下的隐私泄露；

**💡 创新点**

创新点在于将隐私泄露视为通道反演问题，构建可跨目标的目标签名和攻击语言（定位器、对齐器、多样化策略），并将传统记忆提取纳入更广泛的通道视角；

**🔧 技术方法**

技术主要包括基于目标签名的通道建模、可重用的攻击语言、黑盒实验协议以及统一的评估指标（RN、EN、CER、AER等）；

**📊 数据集**

使用了四类目标：基于记忆的两种MEXTRA风格代理、检索增强的RAG代理和两种工具调用代理；

**📈 对比分析**

通过统一的攻击预算（30次查询）和多种模型后端（如ChatGPT、Claude、Gemini）进行横向比较，结果显示记忆目标泄露率最高，检索与工具通道表现为部分泄露或渠道依赖；

**⚠️ 局限性**

局限性包括目标样本有限、指标主要为字符串匹配忽略语义泄露、受限于封闭权重API、未深入评估防御策略以及未覆盖多会话、日志等更复杂的攻击场景。

---

## 199. Think 360°: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth

**arXiv ID:** 2603.22689 | [PDF](https://arxiv.org/pdf/2603.22689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 200. The Interspeech 2026 Audio Encoder Capability Challenge for Large Audio Language Models

**arXiv ID:** 2603.22728 | [PDF](https://arxiv.org/pdf/2603.22728v1)

**作者:** Heinrich Dinkel `[一作]` (MiLM Plus, Xiaomi Inc.), Jian Luan `[通讯]` (MiLM Plus, Xiaomi Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Interspeech 2026音频编码器能力挑战和XARES‑LLM评估框架，以统一的生成式方法评估预训练音频编码器作为大型音频语言模型（LALM）的前端

**💡 创新点**

创新点在于通过冻结预训练编码器、轻量投影层和LoRA微调，构建端到端可扩展的评估体系，且挑战通过隐藏数据集验证模型泛化能力

**🔧 技术方法**

采用大型预训练语言模型SmolLM2-135M作为解码器、LoRA增强、投影器训练，以及多任务生成式评估流程

**📊 数据集**

使用多域公开数据集（ASV2015、CREMA‑D、LibriSpeech、ESC‑50、FSD50k、FMA、GTZAN、LibriCount、VoxCeleb1、Clotho等）以及隐藏测试集

**📈 对比分析**

将编码器通过XARES‑LLM与固定LLM解码器配合，评估指标覆盖分类精度、mAP、iWER、FENSE、DATE等，顶尖团队在Track A获得≈91.2%平均分，Track B约65.9%平均分，显著优于无预训练或单域模型

**⚠️ 局限性**

局限在于仍存在跨域泛化瓶颈，部分团队依赖专有大型编码器，且评估对GPU随机性有一定敏感性，未来需进一步提升对极端噪声与多模态场景的鲁棒性

---

## 201. Algorithmic warm starts for Hamiltonian Monte Carlo

**arXiv ID:** 2603.22741 | [PDF](https://arxiv.org/pdf/2603.22741v1)

**作者:** Matthew S. Zhang `[一作]` (University of Toronto), Sinho Chewi `[通讯]` (Yale University)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5052571538)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

在强化学习中的Metropolis-Adjusted Langevin算法（MALA）中提出了一个无量纲的warm start策略，并实现了在高维空间下的梯度评估复杂度下降到O(d^{1/4})；

**💡 创新点**

通过引入OBABCO分割方案并在理论分析中引入了一步纠正步骤，首次在Rényi距离下实现了d^{1/4}梯度评估复杂度，同时使用了one-shot coupling与离散Girsanov等高级技术；

**🔧 技术方法**

使用了one-shot coupling、shifted composition规则、离散Girsanov变换、离散化误差分析、Jacobian分析以及高阶泰勒展开等技术；

**📊 数据集**

无数据集，本文为理论研究；

**📈 对比分析**

与之前MALA在Wasserstein距离下的O(d^{1/2})复杂度相比，本方法在Rényi距离下实现了更小的迭代复杂度O(d^{1/4})，并且表现优于传统MALA、Riemannian Hamiltonian MC等；

**⚠️ 局限性**

主要限制在于需要强的强凸与Hessian Lipschitz假设，且仍未证明在更一般分布或更低光滑性条件下的可行性。

---

## 202. Why Database Manuals Are Not Enough: Efficient and Reliable Configuration Tuning for DBMSs via Code-Driven LLM Agents

**arXiv ID:** 2603.22708 | [PDF](https://arxiv.org/pdf/2603.22708v1)

**作者:** Xinyi Zhang `[一作]` (Renmin University), Xiaoyong Du `[通讯]` (Harbin Institute of Technology)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了一种基于代码驱动的数据库调优系统，能够自动从 MySQL 源代码中提取调优知识并生成可验证的调优规则。

**💡 创新点**

创新点在于结合静态代码分析与 LLM 推理，揭示配置参数控制的执行路径，并通过关联规则挖掘构建上下文感知、可靠的调优规则，实现了从内部代码直接获得细粒度调优策略。

**🔧 技术方法**

采用了静态 taint 分析、LLM（GPT‑4o‑mini）代码推理、关联规则挖掘、基于 SHAP 的瓶颈诊断、目标约束 FP‑Growth 等技术。

**📊 数据集**

使用了 TPC‑H、TPC‑C、SYSBENCH 三种基准工作负载，并在 MySQL 8.0.36 的云实例上进行实验。

**📈 对比分析**

与 GPTuner、DB‑Bert、SMAC、DDPG++、ResTune、OtterTune 等方法对比，平均比 GPTuner 快 7.11× 收敛，性能提升 19.9%，在可靠性指标上几乎无差异配置。

**⚠️ 局限性**

局限性包括每个 DBMS 版本需耗时 45 小时预先生成知识库，对 LLM 生成的假设仍可能出现幻觉，且目前仅支持 MySQL，未涵盖多租户或非开源数据库场景。

---

## 203. Behavioral Heterogeneity as Quantum-Inspired Representation

**arXiv ID:** 2603.22729 | [PDF](https://arxiv.org/pdf/2603.22729v1)

**作者:** Mohammad Elayan `[一作]` (University of Nebraska--Lincoln), Wissam Kontar `[通讯]` (University of Nebraska--Lincoln)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于量子启发式密度矩阵的驾驶员行为异质性模型，动态捕捉驾驶员的隐式状态并对其进行实时更新。

**💡 创新点**

创新点在于将驾驶行为视为可演化的密度矩阵状态，用随机傅里叶特征进行非线性映射，并结合情境激活实现行为模式的连续演化。

**🔧 技术方法**

采用随机傅里叶特征 (RFF)、密度矩阵表征、二次测量规则 (Born 规则) 以及自动微分优化。

**📊 数据集**

使用第三代仿真数据集 TGSIM，包括 Foggy Bottom 城区交叉口和 I‑395 高速公路轨迹。

**📈 对比分析**

与传统离散标签和混合高斯模型对比，模型在负对数似然上表现更好，四个隐含谱类与情境变量显著相关，并通过 Frobenius 距离验证了模式分离。

**⚠️ 局限性**

局限性包括未考虑横向行为、情境定义有限、缺乏与其它时序基线的全面对比以及对迁移性能的评估。

---

## 204. Can LLM Agents Generate Real-World Evidence? Evaluating Observational Studies in Medical Databases

**arXiv ID:** 2603.22767 | [PDF](https://arxiv.org/pdf/2603.22767v1)

**作者:** Dubai Li `[一作]` (Zhejiang University), Jingsong Li `[通讯]` (Zhejiang University)

**通讯引用:** 6846 | [OpenAlex ID](https://openalex.org/A5023138112)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为RWE‑bench的基准，用于在真实医疗数据库（MIMIC‑IV）上评估大语言模型代理执行完整观测研究的能力。

**💡 创新点**

创新点在于：①将真实同行评审的观测研究协议转化为可执行任务；②构建层级问题和树状证据包以捕捉整个研究流程；③实现自动化队列和LLM驱动的队列筛选（cohort Eval）以快速发现错误。

**🔧 技术方法**

技术手段包括：使用R 4.5.2与PostgreSQL环境进行SQL/统计编程；三种代理框架（MLAB、OpenHands、RWEAgent）进行多步推理和工具调用；评估指标包括ACC、RAR、SMR、SR、CR、Steps；自动队列筛选与LLM判定器。

**📊 数据集**

数据集：MIMIC‑IV v2.2（约30万病人）以及从PubMed检索并手工筛选的165篇观测研究，最终构成162个任务。

**📈 对比分析**

实验对比在三种代理框架下使用六个LLM（三开源、三封闭）进行三次重复，宏平均结果显示最高成功率仅为39.9%（Claude‑Sonnet‑4 + RWEAgent），开放源模型与封闭源模型存在明显差距，且指标普遍偏低。

**⚠️ 局限性**

局限性：整体成功率低、长序列推理与多步骤执行仍易失效；代理在多步决策、语义一致性上表现不佳；自动队列筛选对完整性和召回率的覆盖有限；MIMIC可能在预训练中出现泄露。

---

## 205. Double Coupling Architecture and Training Method for Optimization Problems of Differential Algebraic Equations with Parameters

**arXiv ID:** 2603.22724 | [PDF](https://arxiv.org/pdf/2603.22724v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 206. HyFI: Hyperbolic Feature Interpolation for Brain-Vision Alignment

**arXiv ID:** 2603.22721 | [PDF](https://arxiv.org/pdf/2603.22721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 207. DALDALL: Data Augmentation for Lexical and Semantic Diverse in Legal Domain by leveraging LLM-Persona

**arXiv ID:** 2603.22765 | [PDF](https://arxiv.org/pdf/2603.22765v1)

**作者:** Janghyeok Choi `[一作]` (Seoul National University), Sungzoon Cho `[通讯]` (Seoul National University)

**通讯引用:** 4269 | [OpenAlex ID](https://openalex.org/A5017305201)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DALDALL 方法，利用法律专业角色（如律师、检察官、法官）作为 persona 通过 LLM 生成更具词汇与语义多样性的合成查询，用于法律信息检索的低资源数据扩增。

**💡 创新点**

创新点在于将专业身份嵌入 LLM 生成流程，显著提升生成查询的 Self‑BLEU 低值（词汇多样性）并保持语义一致性，相较传统 Vanilla prompt 获得更高质量的低资源域数据。

**🔧 技术方法**

采用 GPT‑5‑nano 进行两阶段生成（核心要素提取 + persona/vanilla prompt），结合 Self‑BLEU、BM25 recall、Intra‑Cosine、dense retriever fine‑tuning 等评估技术，使用多种检索模型（DPR、LegalBERT、BGE‑base、m5 等）验证效果。

**📊 数据集**

在 CLERC 与 COLIEE 两大法律 IR 基准数据集上进行实验，分别涵盖案例检索与引用检索任务。

**📈 对比分析**

通过与原始查询、Vanilla 生成查询及其混合训练的对比，评估词汇多样性、语义多样性及 Recall@k；实验表明 persona augmentation 在 Self‑BLEU 降低 15–23%，且在 dense retriever 上 recall 能与原始或 Vanilla 相比保持或提升，尤其在 COLIEE 上取得显著改进。

**⚠️ 局限性**

限制在于仅验证英文数据集，未进行法律专家审核；未探究多语言迁移或不同 persona 组合的最优性；对部分模型（如 LegalBERT）效果不如预期，需进一步研究。

---

## 208. CLiGNet: Clinical Label-Interaction Graph Network for Medical Specialty Classification from Clinical Transcriptions

**arXiv ID:** 2603.22752 | [PDF](https://arxiv.org/pdf/2603.22752v1)

**作者:** Pronob Kumar Barman `[一作]` (University of Maryland, Baltimore County), Pronoy Kumar Barman `[通讯]` (Jagannath University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5060895563)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

对MTSamples临床转录按医学专业进行分类，并纠正之前因SMOTE导致的数据泄漏问题，构建了一个无泄漏基准。

**💡 创新点**

创新点在于：①使用基于ICD‑10章节和语义相似度构建的标签图；②将该标签图与Bio_ClinicalBERT编码器通过GCN和注意力门融合；③采用焦点交叉熵处理极端类别不平衡，并实现Per‑Label Platt校准以获得可靠概率。

**🔧 技术方法**

技术手段包括：Bio_ClinicalBERT、两层GCN、Per‑Label Attention Gate、焦点二元交叉熵、Per‑Label Platt Scaling、滑动窗口分块、集成梯度解释。

**📊 数据集**

使用公开的MTSamples语料库（4,966条记录，40个医学专业，单标签分类）。

**📈 对比分析**

与TF‑IDF+LR、SVC、Bio_ClinicalBERT、Clinical‑Longformer等七个基线以及四个消融实验对比，最终模型在无校准时宏F1达0.279（仅次于前人1.0+的错误结果），校准后宏F1略降至0.240，但期望校准误差仅为0.007，显示概率可靠性显著提升。

**⚠️ 局限性**

局限包括：样本极度不平衡（11类训练样本<20）、基准规模有限导致统计功效不足；标签图为静态构建，缺乏训练期间动态更新；仅在MTSamples上评估，缺乏在更大、更多标签数据集上的验证；未探讨大语言模型在少样本/零样本场景下的表现。

---

## 209. Multitask-Informed Prior for In-Context Learning on Tabular Data: Application to Steel Property Prediction

**arXiv ID:** 2603.22738 | [PDF](https://arxiv.org/pdf/2603.22738v1)

**作者:** Dimitrios Sinodinos `[一作]` (McGill University), Narges Armanfard `[通讯]` (McGill University)

**通讯引用:** 1328 | [OpenAlex ID](https://openalex.org/A5073955046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种多任务学习框架，通过新颖的微调策略将多任务意识注入TabPFN模型，以提高钢材机械性能预测的准确性。

**💡 创新点**

创新点在于通过目标平均和任务特定适配器两种方法增强TabPFN的先验知识，使其能够捕捉关键机械指标之间的跨属性关系。

**🔧 技术方法**

使用了TabPFN，这是一种基于变换器的基础模型，专为表格数据的上下文学习而设计。

**📊 数据集**

使用了来自Algoma Steel Inc.的Thin Slab Direct Rolling (TSDR)设施的工业数据集，共包含6415个实例，涉及49个特征和多个机械性能目标。

**📈 对比分析**

与传统机器学习方法和最新的表格学习模型相比，所提出的方法在多个评估指标上表现优越，尤其在预测准确性和计算效率方面。

**⚠️ 局限性**

限制在于TabPFN的架构限制了其在多任务学习中的直接应用，需要通过微调策略来克服这一限制。

---

## 210. Beyond Explanation: Evidentiary Rights for Algorithmic Accountability

**arXiv ID:** 2603.22716 | [PDF](https://arxiv.org/pdf/2603.22716v1)

**作者:** Matthew Stewart `[一作]` `[通讯]` (Independent Researcher), Matthew Stewart (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对168起涉及算法决策的法律案件进行系统编码与统计，构建了争议过程的五类失败模型，并提出了证据获取权与对抗性询问机制，以实现对算法决策的可争议性与可修正性；

**💡 创新点**

创新点在于将关注点从传统的“解释”转向“可争议性”，引入证据获取门槛与两道门结构（程序门与理论门），并提出无需披露模型内部即可通过对抗性询问收集可验证证据的新机制；

**🔧 技术方法**

采用法律案例编码、统计分析与概念框架设计技术，对案例进行分层比较并设计对抗性询问接口的理论实现；

**📊 数据集**

使用包含18个司法区、30个司法单位、7个行业（雇佣、住房、医疗、刑事司法、福利、信贷、平台）共168起案例的数据集，涵盖2012-2025年期间的判决与监管行动；

**📈 对比分析**

通过比较拥有证据获取权与无证据获取权的案件的胜诉率，发现后者成功率约93%而前者仅9%，该差异在各行业均保持一致，显示证据获取显著提升争议成功概率；

**⚠️ 局限性**

局限性包括：仅为诊断性相关性缺乏因果推断；平台责任免疫导致即使完全获取证据也无法获胜；资源与专业门槛可能阻碍低资源群体实施对抗性询问；未覆盖系统性损害与非个人争议等。

---

## 211. How Far Can VLMs Go for Visual Bug Detection? Studying 19,738 Keyframes from 41 Hours of Gameplay Videos

**arXiv ID:** 2603.22706 | [PDF](https://arxiv.org/pdf/2603.22706v1)

**作者:** Wentao Lu `[一作]` (University of Alberta), Cor-Paul Bezemer `[通讯]` (University of Alberta)

**通讯引用:** 2286 | [OpenAlex ID](https://openalex.org/A5066994589)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对工业 QA 长视频进行关键帧提取，使用 Vision‑Language Models (VLM) 进行零样本视觉 bug 检测，并评估 Judge 与 Retrieval‑Augmented Generation (RAG) 两种提升策略的效果。

**💡 创新点**

在真实工业长视频上系统评估 VLM 的直接视觉 bug 检测能力，并首次比较 Judge 与 RAG 在无微调情况下对 VLM 性能的实际提升，证明 VLM 可生成 bug 描述并减少人工检查量。

**🔧 技术方法**

利用 GPT‑4.1‑mini、GPT‑5‑mini、GPT‑4o‑mini 等 OpenAI VLM；FFmpeg 关键帧提取；CLIP 与文本嵌入实现检索增强；通过二分类 prompt 让模型给出 bug 标记和简短理由。

**📊 数据集**

100 条长 QA 游戏录像（共 41 小时、19,738 关键帧）手工标注为真/假；内部 JIRA bug 报告数据库（包含文本描述和截图）用于 RAG。

**📈 对比分析**

与 baseline（单个 VLM 调用）对比，加入 Judge 或 RAG 后在精度与准确率上略有提升：baseline 0.50/0.72，Judge 5‑mini 0.55/0.74，文本检索（Bug 描述相似） 0.57/0.75；整体提升有限，且成本与推理时间翻倍。

**⚠️ 局限性**

局限性包括：单一标注者导致标签噪声；仅在单一游戏/单一 pipeline 上评估；仅使用 OpenAI mini 模型；RAG 依赖特定嵌入和 JIRA 结构；未处理 OCR、时序推理及重复 bug 检测。

---

## 212. Multimodal Industrial Anomaly Detection via Geometric Prior

**arXiv ID:** 2603.22757 | [PDF](https://arxiv.org/pdf/2603.22757v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 213. MVPBench: A Multi-Video Perception Evaluation Benchmark for Multi-Modal Video Understanding

**arXiv ID:** 2603.22756 | [PDF](https://arxiv.org/pdf/2603.22756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 214. Who Spoke What When? Evaluating Spoken Language Models for Conversational ASR with Semantic and Overlap-Aware Metrics

**arXiv ID:** 2603.22709 | [PDF](https://arxiv.org/pdf/2603.22709v1)

**作者:** Naohiro Tawara `[一作]` (NTT), Shinji Watanabe `[通讯]` (Carnegie Mellon University)

**通讯引用:** 25290 | [OpenAlex ID](https://openalex.org/A5001291873)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多说话人对话式自动语音识别（CASR）系统进行了系统化比较，评估了模块化流水线、专用任务LLM和通用多模态LLM在不同场景下的表现，并提出了新的评价指标。

**💡 创新点**

创新点包括：①引入时间约束语义错误率 tcpSemER，用句子嵌入衡量意义误差；②对 cpWER/tcpWER 进行重叠区分拆解，提供细粒度分析；③建立公开排行榜与开源代码，促进后续研究。

**🔧 技术方法**

采用的技术包括：模块化流水线（DiCoW、NTT CHiME‑8）中的 EEND‑VC、TS‑VAD 与 Whisper ASR；专用任务 LLM（VibeVoice、Voxtral MTv2）进行长音频多说话人转写；通用多模态 LLM（Gemini 3.0 Flash）直接生成带说话人标签的转写；多通道融合使用 MOVER；语义评估基于 MiniLM‑L12v2 嵌入。

**📊 数据集**

使用了 CHiME‑8 DASR 公开数据集：Mixer‑6（两人访谈）、NOTSOFAR‑1（3–7人会议）和 DiPCo（4人晚宴），覆盖不同说话人数、重叠率和麦克风布置。

**📈 对比分析**

通过比较 cpWER、tcpWER、tcpSemER 与 DER，评估单声道与多声道配置。结果显示：在两人对话中，专用 LLM 与模块化系统相近；但随着说话人数、重叠率和噪声增加，模块化流水线优势显著；多声道 MOVER 能明显提升 LLM 性能；通用多模态 LLM 在对话式转写上仍表现不佳。

**⚠️ 局限性**

局限性包括：①重叠语音仍是主导难点，LLM 缺乏显式分离/重排模块导致误差激增；②通用多模态 LLM 的说话人标注和分离能力不足；③语义错误率仍依赖 tcpWER 的对齐，无法完全独立评估；④在大规模说话人或极长录音上，模型性能迅速下降。

---

## 215. From Arithmetic to Logic: The Resilience of Logic and Lookup-Based Neural Networks Under Parameter Bit-Flips

**arXiv ID:** 2603.22770 | [PDF](https://arxiv.org/pdf/2603.22770v1)

**作者:** Alan T. L. Bacellar `[一作]` (University of Texas at Austin), Lizy K. John `[通讯]` (University of Texas at Austin)

**通讯引用:** 7812 | [OpenAlex ID](https://openalex.org/A5068885069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究硬件位翻转导致的神经网络鲁棒性，提出结构属性模型并验证逻辑/查找表网络优越性。

**💡 创新点**

提出将鲁棒性视为网络结构特性，并推导多种数值格式的期望误差表达式，发现逻辑/查找表网络在极端错误率下表现最优并出现偶数层恢复现象。

**🔧 技术方法**

使用理论误差分析、梯度差分方法的可微无权网络、Monte Carlo位翻转注入实验。

**📊 数据集**

使用MLPerf Tiny基准套件（MNIST、Fashion‑MNIST、KWS、ToyAdmos等）。

**📈 对比分析**

对比FP32/FP16/FP8/INT/BNN与DWNs，结果显示逻辑/查找表网络在p≤10⁻⁴下保持精度，p≥0.1时仍优于传统模型，偶数层模型出现恢复。

**⚠️ 局限性**

实验受限于小型MLP与CNN，未评估更大网络和更复杂硬件故障模型。

---

## 216. Learning Safe-Stoppability Monitors for Humanoid Robots

**arXiv ID:** 2603.22703 | [PDF](https://arxiv.org/pdf/2603.22703v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 217. Explanation Generation for Contradiction Reconciliation with LLMs

**arXiv ID:** 2603.22735 | [PDF](https://arxiv.org/pdf/2603.22735v1)

**作者:** Jason Chan `[一作]` (University of Sheffield), Robert Gaizauskas `[通讯]` (University of Sheffield)

**通讯引用:** 6731 | [OpenAlex ID](https://openalex.org/A5040587162)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现一种新的任务——对矛盾句对生成可调和的自然语言解释（REG）。

**💡 创新点**

创新点在于：1）将现有 NLI 数据集（ChaosNLI）中标注为矛盾的样本直接转化为 REG 任务；2）定义可度量的成功指标（有效性、连贯性），并通过 LLM 作为 NLI 判断器实现全自动评估；3）系统性评估多种规模与架构的 LLM，揭示“思考”机制在该任务中的有限提升。

**🔧 技术方法**

核心技术：自然语言推理（NLI）模型作为判别器；提示工程（Prompting）与思考链（Chain‑of‑Thought）技术；大规模 LLM 生成解释；统计评估与 Spearman 相关性分析。

**📊 数据集**

使用的主要数据集为 ChaosNLI‑MNLI‑C（275 条标注为矛盾的实例），并在附录中尝试了 8000 条 Implied NLI。

**📈 对比分析**

比较方法：对 18 种 LLM（OpenAI、Qwen3、Llama、Olmo、DeepSeek 等）分别在“非思考”和“思考”模式下生成解释，再由九个 NLI 判别器评估；实验结果显示：大部分模型能够生成连贯解释（最高 85.6%），但有效解释率低（最高 73.5%），整体成功率最佳的 GPT‑5‑mini 约 40%；相比之下，最优开源模型 Llama‑3.3‑70B‑Instruct 约 26%。思考机制对中等规模模型有效提升，但对大模型提升有限。

**⚠️ 局限性**

局限性包括：仅在英语短句上测试；缺乏多语言与长文本扩展；未对生成解释进行形式化分类或外部知识检索；评估仅聚焦于与前提/假设的关系，而未考虑事实真实性与多样性。

---

## 218. Explainable Threat Attribution for IoT Networks Using Conditional SHAP and Flow Behavior Modelling

**arXiv ID:** 2603.22771 | [PDF](https://arxiv.org/pdf/2603.22771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 219. Does Teaming-Up LLMs Improve Secure Code Generation? A Comprehensive Evaluation with Multi-LLMSecCodeEval

**arXiv ID:** 2603.22717 | [PDF](https://arxiv.org/pdf/2603.22717v1)

**作者:** Bushra Sabir `[一作]` (CSIRO's Data61), Surya Nepal `[通讯]` (CSIRO's Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Multi-LLMSecCodeEval 框架，对多 LLM 组合（单体、集成、协同、混合）在安全代码生成、漏洞检测与补丁生成中的表现进行系统评估。

**💡 创新点**

创新点在于：①首次将多模型集成、协同与静态分析统一到同一端到端管线；②设计 10 种可复现的管线，系统比较各策略对安全率的提升；③证明仅靠模型规模不足以保障安全，需结构化的多模型协同与静态验证。

**🔧 技术方法**

技术方法包括：多 LLM 并行生成与交叉验证、投票/加权融合、链式辩论、基于 CodeQL 的静态分析过滤与验证、跨模型解释推理、自动补丁生成与重评。

**📊 数据集**

使用两大公开安全数据集：SecLLMEval（从 SecEval + LLMSecEval 转化的 NL 提示，涵盖 112 个 MITRE Top‑25 CWE）和 SecLLMHolmes（真实漏洞代码转化为 NL，106 条样本，Python/C）。

**📈 对比分析**

比较方法：构建 10 条流水线，在两个数据集上测量 Secure Code Rate、Recall、Precision、F1；结果显示：①集成 + 静态分析管线可比单模型提升 47.3%（SecLLMEval）/19.3%（SecLLMHolmes）；②纯 LLM 协同提升 8.9–22.3%；③混合管线 P10 最优，Secure Rate 达 97.3%/99.1%，比最佳集成高 1.8–4.7%，比协同高 19–27%。

**⚠️ 局限性**

局限性：①安全评估仅依赖静态分析（CodeQL），对需运行时或跨文件语义的漏洞不敏感；②数据集限定为 C/Python，缺乏大规模项目、多语言、真实环境的验证；③未评估功能正确性与安全-功能的协同；④模型知识截止、提示多样性与成本等因素仍影响可迁移性。

---

## 220. ENC-Bench: A Benchmark for Evaluating Multimodal Large Language Models in Electronic Navigational Chart Understanding

**arXiv ID:** 2603.22763 | [PDF](https://arxiv.org/pdf/2603.22763v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 221. Human vs. NAO: A Computational-Behavioral Framework for Quantifying Social Orienting in Autism and Typical Development

**arXiv ID:** 2603.22759 | [PDF](https://arxiv.org/pdf/2603.22759v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 222. PopResume: Causal Fairness Evaluation of LLM/VLM Resume Screeners with Population-Representative Dataset

**arXiv ID:** 2603.22714 | [PDF](https://arxiv.org/pdf/2603.22714v1)

**作者:** Sumin Yu `[一作]` (Seoul National University), Taesup Moon `[通讯]` (Seoul National University)

**通讯引用:** 2671 | [OpenAlex ID](https://openalex.org/A5080346989)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了基于美国人口统计的 PopResume 简历数据集，并提出了以路径特定效应（PSE）为核心的因果公平审计框架，用于评估 LLM/VLM 简历筛选系统。

**💡 创新点**

创新点在于将公平性拆分为业务必要路径和红线路径，利用 BIE 与 RIE 两个路径特定效应精细捕捉可合法与不可合法的差异；同时，采用真实人口统计数据生成简历，避免了手工注入属性导致的因果评估失效。

**🔧 技术方法**

使用了结构因果模型、DML‑UCA 双重稳健估计、XGBoost 预测、Face‑MoGLE 生成头像、LLM（如 Llama‑3.1‑8B‑Instruct、GPT‑4o‑mini 等）与 VLM（Qwen2.5‑VL‑7B、GPT‑4o 等）进行评分与评估。

**📊 数据集**

数据集基于美国 ACS PUMS、PSID、SSA 姓名记录和 2010 年 Census 姓氏数据生成，包含 60.8K 条符合人口分布的简历，覆盖五个职业。

**📈 对比分析**

在 120 个评估配置（5 职位 × 3 简历格式 × 2 受保护属性 × 4 模型）下计算 TE、NDE、NIE 并进一步拆分为 BIE 与 RIE；结果揭示了五种代表性歧视模式，PSE 能够揭示聚合指标无法发现的歧视，并指出照片会显著提升直接歧视效应；性能上表明传统公平指标容易误判，PSE 提供更细粒度的洞察。

**⚠️ 局限性**

局限性包括：数据仅基于美国人口统计，跨国适用性有限；简历为规则生成，缺少真实写作细节和复杂职业路径；公平判定仍需结合具体法律与社会背景，无法给出绝对公平与否的结论。

---

## 223. PRISM: A Dual View of LLM Reasoning through Semantic Flow and Latent Computation

**arXiv ID:** 2603.22754 | [PDF](https://arxiv.org/pdf/2603.22754v1)

**作者:** Ruidi Chang `[一作]` (Rice University), Hanjie Chen `[通讯]` (Rice University)

**通讯引用:** 46740 | [OpenAlex ID](https://openalex.org/A5100381999)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PRISM 框架，对大语言模型的推理过程进行统一的显式语义流与隐式计算模式双重分析

**💡 创新点**

创新点在于将推理步骤的语义类别转移与层级内部的高斯混合模型（隐式计算 regime）及跨步桥矩阵结合，形成概率化的全局视角

**🔧 技术方法**

使用多阶马尔可夫链、Gaussian Mixture Model、Expectation–Maximization、PCA 压缩、桥矩阵以及对隐藏状态的软后验分析

**📊 数据集**

实验覆盖四大开源推理模型（Bespoke-Stratos-7B、OpenThinker-7B、Qwen3-1.7B、Llama-3.1-Nemotron-Nano-4B-v1.1）以及四个基准数据集（MATH-500、AIME-2024、GPQA-Diamond、WebInstruct-Verified）

**📈 对比分析**

通过对比正确与错误轨迹、不同长度失败模式、以及三种提示策略下的语义转移与计算 regime 分布，展示 PRISM 能揭示准确率之外的推理行为差异，并对提示设计给出可操作的改进建议；在实验中未直接提升任务准确率，但显著提高了对推理失败模式的解释力

**⚠️ 局限性**

局限性包括依赖外部语义标签与无监督聚类可能带来的噪声、实验模型与数据集范围有限、以及框架仅做分析诊断，未直接提升推理性能

---

## 224. Testing Properties of Edge Distributions

**arXiv ID:** 2603.22702 | [PDF](https://arxiv.org/pdf/2603.22702v1)

**作者:** Yumou Fei `[一作]` (Massachusetts Institute of Technology), Yumou Fei `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5023771090)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本论文研究了图的边上概率分布的分布测试，特别是测试二分性、三角形自由性和平方自由性的样本复杂度。

**💡 创新点**

提出了针对边分布的二分性、三角形自由性和平方自由性的测试的近似紧界限，样本复杂度分别为Θ(n)、n^(4/3)±o(1)和n^(9/8)±o(1)。

**🔧 技术方法**

使用了基于生日悖论类型引理的新技术来证明平方自由性测试的上界，并将这些技术与分布无关的属性测试的框架结合起来。

**📊 数据集**

使用了边集为[n]2的图的边分布，样本来自于这些边的概率分布。

**📈 对比分析**

通过与现有的分布测试方法进行比较，证明了所提出方法的样本复杂度在理论上是最优的，尤其是在测试平方自由性时，样本复杂度达到了O(n^(9/8)/ε)。

**⚠️ 局限性**

本研究的局限性在于未能确定每个固定图H的H自由性测试的样本复杂度，尤其是在H为树的情况下。

---

## 225. Beyond Binary Correctness: Scaling Evaluation of Long-Horizon Agents on Subjective Enterprise Tasks

**arXiv ID:** 2603.22744 | [PDF](https://arxiv.org/pdf/2603.22744v1)

**作者:** Abhishek Chandwani `[一作]`, Ishan Gupta `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LH-Bench三支柱评估框架，能够在主观、长周期的企业任务上对LLM代理进行可靠评估；

**💡 创新点**

创新点在于：①引入专家编写的量表让LLM评判获得可重现的评价信号；②结合步进式真实工件做奖励和验证；③加入人类偏好验证，实现多层次、可诊断的评估；

**🔧 技术方法**

技术包括：LLM‑as‑Judge（多模型并行评判）、MCP工具协议、Sandboxed Execution、Runtime Verification Hook、层级量表评估、程序化工件生成与比较；

**📊 数据集**

使用公开数据集：Figma‑to‑Code（33个真实设计任务、相应Ground‑Truth图像与代码）和Programmatic‑Content（41门课程、183章的源材料与脚本生成任务）；

**📈 对比分析**

比较方法：三种代理架构（Claude Code、Codex CLI、Gemini CLI）在两环境中的输出、技能与行为分数；结果显示Codex GPT‑5.2 Pro与Claude Opus 4.6在输出分数上相近且均优于Gemini；人类偏好与LLM评判大体一致；在程序化内容上Claude Code表现最佳；恢复率约70%且依赖错误信息质量；

**⚠️ 局限性**

局限性：仅评估两种企业环境，未覆盖更广泛的行业场景；实验采用的是特定版本的代理（非开放源代码模型驱动）；未彻底剖析模型能力与驱动逻辑的耦合；未来需扩展环境多样性、实现模型无关的框架。

---

## 226. Fleet-Level Battery-Health-Aware Scheduling for Autonomous Mobile Robots

**arXiv ID:** 2603.22731 | [PDF](https://arxiv.org/pdf/2603.22731v1)

**作者:** Jiachen Li `[一作]` (University of Texas at Austin), Dongmei Chen `[通讯]` (University of Texas at Austin)

**通讯引用:** 13124 | [OpenAlex ID](https://openalex.org/A5100677493)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种多机器人队列级别的电池健康优化框架，联合任务分配、服务顺序、充电模式选择和共享充电站调度，并通过约束平衡机器人间的电池退化；

**💡 创新点**

核心创新在于构建fleet‑level MILP，利用分块McCormick线性化处理充电与空闲SOC的二次退化项，并在大规模实例中设计两层matheuristic（主层负责组合决策，子层求解精确退化排程），实现退化均衡与充电资源共享；

**🔧 技术方法**

采用混合整数线性规划（MILP）与McCormick多段线性化、基于实例数据的紧凑Big‑M、以及基于分解的层级matheuristic（主/子问题）来求解；

**📊 数据集**

实验使用人工生成的仓库物流场景，机器人、任务与充电站按网格布置，任务时隙、能耗等均采用均匀分布生成；

**📈 对比分析**

与规则调度、仅能量约束MILP、无充电站容量MILP及单机精确MILP对比，结果显示在大规模实例中本方法在总退化上可比规则调度低约54%，在计算时间上比单机MILP快20倍以上，且退化与延迟指标均优于基线；

**⚠️ 局限性**

局限在于作为matheuristic无法提供全局下界，退化参数采用经验近似，未考虑实时任务到达、交通拥堵等不确定性，且主层缺乏SOC递推约束导致可能产生无效模式需子层排除。

---

## 227. GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning

**arXiv ID:** 2603.22687 | [PDF](https://arxiv.org/pdf/2603.22687v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 228. From Pixels to Semantics: A Multi-Stage AI Framework for Structural Damage Detection in Satellite Imagery

**arXiv ID:** 2603.22768 | [PDF](https://arxiv.org/pdf/2603.22768v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 229. SG-VLA: Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation

**arXiv ID:** 2603.22760 | [PDF](https://arxiv.org/pdf/2603.22760v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 230. How Utilitarian Are OpenAI's Models Really? Replicating and Reinterpreting Pfeffer, Krügel, and Uhl (2025)

**arXiv ID:** 2603.22730 | [PDF](https://arxiv.org/pdf/2603.22730v1)

**作者:** Johannes Himmelreich `[一作]` `[通讯]`, Johannes Himmelreich

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对OpenAI模型的道德推理进行复制与扩展，测试四个模型在两种道德困境（电车难题与桥梁难题）下不同提示变体的反应。

**💡 创新点**

提出单提示评估易受格式、时序和安全过滤器影响，强调多提示鲁棒性测试与系统指纹记录的重要性，揭示模型行为随时间漂移而非单纯道德倾向。

**🔧 技术方法**

使用OpenAI API收集数据，设计四种提示变体（原始、反向、描述、平中），对响应进行Yes/No/Other/N/A分类，使用统计分析评估各模型表现，并借助Claude生成脚本与图表。

**📊 数据集**

实验数据来自两组规模不同的API请求：复制实验每模型1000次，提示变体实验每格100次；涉及模型GPT‑4o、GPT‑4o‑mini、o3、o3‑mini。

**📈 对比分析**

通过比较各模型在不同提示下的Yes率、N/A率等指标发现：在电车难题中，理由模型与非理由模型在消除提示偏差后均趋向高度功利；在桥梁难题中，理由模型表现更为多样，非理由模型保持一致的非功利回答；GPT‑4o‑mini在中立提示下显示显著功利倾向。

**⚠️ 局限性**

局限性包括仅评估OpenAI模型、强制选择式提示、缺乏对推理机制的解释、样本非独立导致统计假设违背，以及对电车难题响应是否真正反映功利伦理的构造效度质疑。

---

## 231. A Study of Scientific Computational Notebook Quality

**arXiv ID:** 2603.22726 | [PDF](https://arxiv.org/pdf/2603.22726v1)

**作者:** Shun Kashiwa `[一作]` (UC San Diego), Michael Coblenz `[通讯]` (UC San Diego)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建了518个来自2024年Nature论文的代码仓库语料库，系统评估了科学Jupyter笔记本在可复现性、可读性和可重用性方面的质量，并对文档、代码克隆与变量变异模式进行量化分析。

**💡 创新点**

创新之处在于首次系统性地将TOP指南、FAIR原则与Nature报告标准结合，量化科学笔记本在复制、文档、克隆和状态变异方面的缺陷，并提出针对复制性、可读性和可重用性的改进需求。

**🔧 技术方法**

采用手工执行19个笔记本、Markdown内容分类、代码克隆检测、变异比率与扩散分数度量等技术，并辅以静态分析和文档审查来评估代码质量。

**📊 数据集**

使用的数据集为518个公开的代码仓库，来源于1239篇2024年Nature论文，其中包含637个Jupyter笔记本。

**📈 对比分析**

与通用GitHub笔记本和Python脚本进行对比，发现科学笔记本复现率仅为11%，克隆率42%，变异扩散更高，整体质量低于一般笔记本。

**⚠️ 局限性**

研究限制包括样本量有限（仅19个可执行笔记本）、变异度量尚未充分验证，以及仅关注Jupyter环境，可能无法完整反映科学代码的多样性。

---

## 232. Rank-Aware Resource Scheduling for Tightly-Coupled MPI Workloads on Kubernetes

**arXiv ID:** 2603.22691 | [PDF](https://arxiv.org/pdf/2603.22691v1)

**作者:** Tianfang Xie `[一作]` (Purdue University), Tianfang Xie `[通讯]` (Purdue University)

**通讯引用:** 3546 | [OpenAlex ID](https://openalex.org/A5021112374)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个基于Kubernetes的框架，实现按子域单元数分配CPU资源的MPI并行CFD求解器；

**💡 创新点**

创新点在于：①利用Kubernetes资源分配实现按负载比例的CPU调度，②首次在运行中的MPI CFD求解器中实现无重启的Pod垂直扩容（CPU动态调整），③系统性评估了CFS硬限制对MPI同步的放大效应；

**🔧 技术方法**

使用技术包括：Kubernetes v1.35的In‑Place Pod Vertical Scaling、cgroups v2的CFS带宽控制、OpenFOAM的空间分解与MPI、AWS EC2 c5.xlarge实例、k3s轻量级Kubernetes、NFS共享存储；

**📊 数据集**

使用的数据集主要是OpenFOAM典型案例：pitzDaily（12K单元2D后向阶梯）、NACA0012翼型（16K、101K单元2D RANS）以及对应的mesh分解；

**📈 对比分析**

比较方法：对照传统等CPU分配、网络层（C0 vs C2）、硬限制 vs 请求仅限（C3 vs C4）、动态扩容（C5）等七种配置；结果显示：硬限制导致78倍慢速；等CPU分配与按比例分配在4核时差距仅3%，但在16核时提升20%，且按比例分配可减少82%稀疏域CPU预留，释放6.5 vCPU；

**⚠️ 局限性**

局限性：实验规模仅至16个MPI核，未验证64+核或3D大规模；仅使用2D案例，未覆盖复杂网格和多物理耦合；动态扩容仅在短周期验证，未证明在长时间模拟中有效；CFS限制/请求比例的阈值未系统研究；

---

## 233. Digital Twin Enabled Simultaneous Learning and Modeling for UAV-assisted Secure Communications with Eavesdropping Attacks

**arXiv ID:** 2603.22753 | [PDF](https://arxiv.org/pdf/2603.22753v1)

**作者:** Jieting Yuan `[一作]` (Sun Yat-sen University), Shimin Gong `[通讯]` (Sun Yat-sen University)

**通讯引用:** 5080 | [OpenAlex ID](https://openalex.org/A5042460024)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c`

**🎯 论文内容**

提出基于数字孪生的DT‑SLAM框架，联合多UAV的模式切换（数据收集/主动干扰）与多阶段Stackelberg博弈，学习与部署鲁棒安全通信策略；

**💡 创新点**

创新点在于：1）将LE‑UAV的模式切换与轨迹规划、网络形成耦合到Stackelberg博弈中；2）提出DT‑SLAM两循环结构，让DRL在虚拟孪生环境中高效学习并持续更新；3）设计RPPO算法，将GPR评估的模型不匹配作为奖励项，实现不确定性驱动的探索与模型自适应；

**🔧 技术方法**

使用的技术包括：多阶段Stackelberg博弈建模、深度强化学习（PPO）、数字孪生（GPR为状态预测与不确定性估计）、双循环（inter‑DT学习 + intra‑DT更新）、RL奖励改进（加入不确定性项）。

**📊 数据集**

实验数据采用仿真环境：K=30个地面用户随机分布在2000×2000 m²区域，3架UAV，Rician信道模型；未使用公开数据集。

**📈 对比分析**

与基线Ideal‑PPO（直接在真实环境训练）以及DT‑PPO（不考虑不确定性）比较，DT‑RPPO在约4000次与真实环境交互即可收敛，收敛速度比Ideal‑PPO快约20%，最终安全吞吐量提升约8.6%，比DT‑PPO高约12%；在不同干扰模式下，Mode‑Switching策略在保证安全性的同时，吞吐量比Fixed‑Jamming和No‑Jamming均优。

**⚠️ 局限性**

主要局限：1）DT模型仍受观测稀疏和高维状态的限制，GPR在大规模场景下计算成本上升；2）实验仅在仿真环境中验证，未验证在真实无人机网络中的鲁棒性；3）假设EA‑UAV能完全观测LE‑UAV策略，实际对抗中可能存在信息不对称。

---

## 234. Spiking Personalized Federated Learning for Brain-Computer Interface-Enabled Immersive Communication

**arXiv ID:** 2603.22727 | [PDF](https://arxiv.org/pdf/2603.22727v1)

**作者:** Chen Shang `[一作]` (University of Technology Sydney), Jiadong Yu `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出基于脑机接口的个性化沉浸式通信框架，利用脉冲神经网络驱动的个性化联邦学习，对EEG信号进行本地训练与推理，实现用户状态识别与个性化适配。

**💡 创新点**

① 将个性化联邦学习与稀疏事件驱动的脉冲神经网络相结合，兼顾隐私与能效；② 利用SNN的稀疏激活降低梯度异质性，提升训练稳定性；③ 在BCI沉浸式场景中实现低功耗持续推理。

**🔧 技术方法**

脑机接口（EEG）、个性化联邦学习（PFL）、脉冲神经网络（SNN）与LIF模型、替代梯度训练、能耗评估。

**📊 数据集**

公开EEG Motor-Imagery数据集（109名受试者），挑选3名受试者进行4类动作识别。

**📈 对比分析**

与ANN+PFL、FL+SNN、FL+ANN三种基线比较，PFLSNN在50轮后识别准确率达87.53%，高于PFLANN的81.90%及其他基线；推理能耗比PFLANN低约6.46倍。

**⚠️ 局限性**

仅在少量受试者与单一任务上验证，缺乏多模态融合与大规模部署评估；SNN训练复杂度与模型可解释性仍待进一步研究。

---

## 235. TimeWeaver: Age-Consistent Reference-Based Face Restoration with Identity Preservation

**arXiv ID:** 2603.22701 | [PDF](https://arxiv.org/pdf/2603.22701v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 236. Non-Adversarial Imitation Learning Provably Free of Compounding Errors: The Role of Bellman Constraints

**arXiv ID:** 2603.22713 | [PDF](https://arxiv.org/pdf/2603.22713v1)

**作者:** Tian Xu `[一作]` (Nanjing University), Yang Yu `[通讯]` (Nanjing University)

**通讯引用:** 9982 | [OpenAlex ID](https://openalex.org/A5100342259)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并分析了基于Q的模仿学习方法，证明IQ‑Learn等现有方法等价于行为克隆，无法消除累积误差；提出Dual Q‑DM方法，利用Bellman约束实现多步分布匹配，理论上等价于对抗模仿学习，可消除累积误差。

**💡 创新点**

首次在非对抗性框架下引入Bellman约束，证明其是实现Q‑based IL泛化和消除累积误差的关键机制，并给出从原始分布匹配到Dual Q‑DM的完整理论证明。

**🔧 技术方法**

使用原始-对偶分布匹配框架、Lagrange对偶、Bellman约束正则化、软max策略、总变差距离与熵正则化等技术。

**📊 数据集**

在离散硬实例（用于严格检验累积误差）和MuJoCo连续控制五个任务上进行实验验证。

**📈 对比分析**

与行为克隆、IQ‑Learn、ValueDICE、DAC、HyPE、LS‑IQ等基线对比；结果显示Dual Q‑DM在硬实例上保持小的模仿误差，在MuJoCo任务中收敛速度快、稳定性好，优于大多数Q‑based方法并与对抗方法相当。

**⚠️ 局限性**

对比中未评估在更复杂或高维真实世界任务上的表现；Bellman约束的近似实现可能对收敛速率与样本效率产生影响；理论证明基于完美Q‑函数表达性假设，实际算法需面对函数逼近误差。

---

## 237. Detecting Non-Membership in LLM Training Data via Rank Correlations

**arXiv ID:** 2603.22707 | [PDF](https://arxiv.org/pdf/2603.22707v1)

**作者:** Pranav Shetty `[一作]` (JPMorgan AI Research), Xiaomo Liu `[通讯]` (JPMorgan AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种利用Spearman秩相关的灰盒方法，对给定数据集是否被用于LLM训练进行非成员检测；

**💡 创新点**

创新点在于将数据集级非成员检测从“检测是否出现”转为“证明未出现”，并且仅依赖模型logits，无需额外参考数据集；

**🔧 技术方法**

核心技术包括：(a) 计算Min‑K%正则化的token log概率得分；(b) 对目标模型、未见过该数据集的参考模型和经过知识蒸馏后的“蒸馏参考模型”分别评分；(c) 通过Spearman秩相关差异进行一侧显著性检验；

**📊 数据集**

使用了来自Pythia、OLMo系列的公开模型作为参考模型，评估的数据集包括Wiki、PubMed、ArXiV、Common Crawl、Ubuntu‑IRC、Enron、Freelaw、Reddit等 Pile/ Dolma 验证集；

**📈 对比分析**

与现有的数据集级成员检测方法（LLM‑DI、PaCoST）比较，所提方法在8个数据集上均能在α=0.05下可靠判定非成员，且在少于150篇文档时仍保持稳定；

**⚠️ 局限性**

局限性包括：需已知数据集公开时间且假设完整加入或未加入训练，无法处理部分采样情况；对较旧或私有数据的适用性受限；

---

## 238. Synthetic or Authentic? Building Mental Patient Simulators from Longitudinal Evidence

**arXiv ID:** 2603.22704 | [PDF](https://arxiv.org/pdf/2603.22704v1)

**作者:** Baihan Li `[一作]` (Shanghai Jiao Tong University), Mengyue Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1619 | [OpenAlex ID](https://openalex.org/A5109064838)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出了Deprofile框架，通过将临床评估访谈、咨询对话与社交媒体长时序数据统一对齐，构建了多源、真实的患者档案并实现了基于该档案的患者模拟；

**💡 创新点**

创新点包括：①两阶段匹配机制将评估、咨询与社交媒体信息对齐，生成结构化档案；②链式变更（CoC）Agent将噪声长时序转化为可检索的记忆卡；③双重调控机制既为低信息情景提供情节丰富，也在高信息情景下强约束事实一致，从而显著抑制hallucination并提升临床真实性；

**🔧 技术方法**

技术上主要使用Chain-of-Change Agent、双通道记忆检索、模块化提示工程、LLM（Qwen3、Llama3.1、GPT‑4o‑mini、DeepSeek‑V3.2等）生成、LLM‑as‑a‑Judge (G‑Eval)、MentalBERT 及自定义评估指标；

**📊 数据集**

数据集涵盖中文抑郁症评估对话集D4（1,340场次）、咨询对话数据集（300位客户）以及Twitter‑STMHD社交媒体数据（6,803位用户、37,000条帖子），最终构建3,258条高质量统一档案；

**📈 对比分析**

通过与先行模型Patient‑Ψ、Eeyore及不同配置的Deprofile ablation进行系统对比，使用MentalBERT实时性、Q‑Centroid多样性、G‑Eval等指标评估。结果显示，Deprofile在所有LLM骨干上均优于基线，提升了对话真实性、事件丰富度和多样性，并在小型模型上达到与大模型相当的临床逼真度；

**⚠️ 局限性**

局限性包括：①社交媒体数据存在偏倚与真实性不足，需要进一步校准事件频率与情绪强度；②仅基于文本的模拟无法完全复制非语言交互线索；③仅适用于教学与模拟，禁止用于真实临床诊断。

---

## 239. Coordinate Encoding on Linear Grids for Physics-Informed Neural Networks

**arXiv ID:** 2603.22700 | [PDF](https://arxiv.org/pdf/2603.22700v1)

**作者:** Tetsuro Tsuchino `[一作]` (Gifu University), Motoki Shiga `[通讯]` (Tohoku University)

**通讯引用:** 5398 | [OpenAlex ID](https://openalex.org/A5083977312)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于线性轴独立网格单元的坐标编码方法（CELG）来解决 PDE，利用自然立方样条插值实现 C^2 连续的特征向量，并将编码后的特征通过 Hadamard 乘积输入 MLP 计算解。

**💡 创新点**

创新点在于：①采用轴独立的线性网格单元显著降低模型参数与计算复杂度；②用自然立方样条插值保持高阶导数连续，从而稳定 PINN 训练；③通过坐标编码消除光谱偏差，使模型在高频成分上收敛更快。

**🔧 技术方法**

技术包括：物理信息神经网络（PINN），坐标特征编码，自然立方样条插值，Hadamard 乘积融合特征，tanh 激活的 MLP，以及 Adam 梯度优化。

**📊 数据集**

在一维和二维多频 Poisson、Burgers、Allen‑Cahn、流混合以及三维 Helmholtz 等基准 PDE 以及 2–5 维 Helmholtz 的高维测试集上验证，训练样本为随机采样的 collocation、初始和边界点。

**📈 对比分析**

与 Vanilla PINN、CP‑PINN、PIXEL、H‑Spline 等方法比较，CELG 在训练速度、参数量和预测误差上普遍优于竞争者；在高维实验中，CELG 仍能保持可训练性并获得最低相对误差，而对手出现 OOM 或显著更高误差。

**⚠️ 局限性**

局限性包括：无法处理非矩形网格单元（需更高效的插值方法）；对 collocation 点的选择仍需改进；在某些 PDE（如 Burgers）中出现靠近网格点的误差波动，需进一步完善插值连续性。

---

## 240. WiFi2Cap: Semantic Action Captioning from Wi-Fi CSI via Limb-Level Semantic Alignment

**arXiv ID:** 2603.22690 | [PDF](https://arxiv.org/pdf/2603.22690v1)

**作者:** Tzu-Ti Wei `[一作]` (National Yang Ming Chiao Tung University), Jen-Jee Chen `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 875 | [OpenAlex ID](https://openalex.org/A5050310630)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了WiFi2Cap框架，实现从Wi‑Fi CSI直接生成自然语言动作描述；

**💡 创新点**

核心创新包括：①将对齐良好的视听语言教师迁移至无线模态；②引入Mirror‑Consistency损失消除左右肢体与镜像动作歧义；③通过前缀调优将CSI语义注入冻结语言模型，完成端到端的生成；

**🔧 技术方法**

采用CLIP的视觉/文本编码、InfoNCE对比学习、ResNet‑18双分支幅度/相位编码、门控融合、对齐与生成的镜像一致性约束，以及GPT‑2（或Qwen、phi‑2）前缀调优技术；

**📊 数据集**

构建WiFi2Cap数据集，包含100类动作、5秒CSI-RGB-句子三模同步样本，并在公开的Person‑in‑WiFi 3D数据集上进行跨数据集验证；

**📈 对比分析**

与直接CSI→LM的基线相比，WiFi2Cap在BLEU‑4、METEOR、ROUGE‑L、CIDEr、SPICE等指标均提升显著（如BLEU‑4从14.85升至51.78），并在Person‑in‑WiFi 3D上亦表现出显著的迁移性能；

**⚠️ 局限性**

局限性：仍存在轻微的方向性错误；依赖同步视频作为教师数据，可能限制对未知场景的泛化；样本仅为单人、5秒短视频，未覆盖多人人多动作等复杂场景。

---

## 241. BlindMarket: Enabling Verifiable, Confidential, and Traceable IP Core Distribution in Zero-Trust Settings

**arXiv ID:** 2603.22685 | [PDF](https://arxiv.org/pdf/2603.22685v1)

**作者:** Zhaoxiang Liu `[一作]` (Kansas State University), Ning Luo `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 642 | [OpenAlex ID](https://openalex.org/A5061101152)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个全流程零信任的硬件IP核心分发框架，支持在不依赖任何可信第三方的前提下实现IP的可验证、保密与可追溯。

**💡 创新点**

创新点包括：①首次将隐私保护SAT求解迁移至硬件设计验证领域；②提出基于控制流优先级的SAT决策启发式和设计剪枝（COI）两种实用优化；③设计了基于区块链的IP所有权审计与追溯机制；④实现了完整的OT+2PC流程，完成IP选取、验证与授权的零信任链路。

**🔧 技术方法**

采用的技术主要有：安全两方计算（2PC）、1-out-of-N 拒绝转移（OT）、隐私保护SAT求解器、基于控制流的决策启发式、设计剪枝、Tseytin/bit‑blasting转CNF、加密矩阵编码、基于账本的可追溯性等。

**📊 数据集**

使用了真实工业级IP基准：ANSI‑C、OpenCores、TrustHub 共13个IP（共26个验证实例），涵盖计数器、仲裁器、TAP控制器等典型功能块。

**📈 对比分析**

实验通过与非私有基线（普通SAT求解）对比，测量验证时间、通信量和加密步骤。结果显示：在12/13个实例上完成验证；设计剪枝与控制流启发式平均可将求解时间降低10–100×；通信开销（OT与GC）与设计规模呈线性关系，且在剪枝后可降至可接受水平。

**⚠️ 局限性**

局限性：①交互式2PC与OT协议导致额外延迟；②大规模IP（数十万变量/子句）在GC阶段通信量巨大，仍不够实用；③仅支持有界时间/覆盖性质的验证；④需预先公开部分信号深度信息以生成启发式，可能在极端隐私场景下泄露有限结构信息。

---

## 242. Option pricing model under the G-expectation framework

**arXiv ID:** 2603.22831 | [PDF](https://arxiv.org/pdf/2603.22831v1)

**作者:** Ziting Pei `[一作]` (Suzhou University of Science and Technology), Xiaotao Zheng `[通讯]` (Soochow University)

**通讯引用:** 1123 | [OpenAlex ID](https://openalex.org/A5052865340)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文建立了在G期望框架下的风险中性期权定价模型，推导出G‑Black‑Scholes非线性偏微分方程，并通过对标价格做对数变换得到更易数值求解的等价PDE；

**💡 创新点**

创新点在于将对数变换与G‑期望相结合，显著放宽了显式差分法的稳定性约束，同时提供了对非线性系统的显式与隐式有限差分离散方案，并通过一致性、稳定性、单调性和收敛性理论证明了方案的可靠性；

**🔧 技术方法**

采用了G‑几何布朗运动建模、非线性Feynman‑Kac公式、对数变换、显式与隐式有限差分离散、Picard迭代求解非线性系统以及M‑矩阵理论的收敛性分析；

**📊 数据集**

使用的“数据集”为两类经典期权（蝶式价差与数字欧式看涨期权）的支付函数，数值实验通过多尺度网格（从16到1024时间步、从161到1281空间点）来评估误差；

**📈 对比分析**

通过与传统S‑域显式差分法比较，发现对数变换后显式法的最小时间步数减少约63%，CPU时间下降数倍，且两种方法均在对数域中实现了约二阶收敛；

**⚠️ 局限性**

局限性包括：仅适用于一维标的资产，需满足对数变换下的边界条件；对不连续支付函数（如数字期权）收敛速率下降至一阶；隐式法需要额外的迭代，可能在极端波动率区间内收敛速度受限；

---

## 243. Improving Safety Alignment via Balanced Direct Preference Optimization

**arXiv ID:** 2603.22829 | [PDF](https://arxiv.org/pdf/2603.22829v1)

**作者:** Shiji Zhao `[一作]` (Beihang University), XingXing Wei `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM安全对齐中的过拟合问题，发现并量化“Imbalanced Preference Comprehension”现象，提出基于互信息的自适应加权损失——Balanced Direct Preference Optimization（B‑DPO）来缓解过拟合并提升安全性。

**💡 创新点**

创新点在于首次用互信息量化安全偏好对中的理解差异，并设计了自适应权重（根据互信息平衡）与缩放因子来调节DPO的优化强度，从而在不丢弃训练数据的前提下抑制对单一响应的过拟合。

**🔧 技术方法**

采用Direct Preference Optimization（DPO）框架，计算查询与响应的互信息，构造自适应权重的损失函数，加入梯度缩放因子；同时使用对抗性攻击（GCG、PAIR）与安全评测基准进行验证。

**📊 数据集**

主要使用PKU‑SafeRLHF安全偏好对数据（10k安全问答+10k不安全问答），训练集包含每条查询的preferred（y_w）与dispreferred（y_l）响应；评测基准包括StrongReject、XsTest、GCG、PAIR、GSM8k、SimpleQA、AdvGLUE、HHH等。

**📈 对比分析**

与DPO、CPO、SimPO、SafeDPO等基线在同一模型（Qwen‑2‑7B‑Instruct、Mistral‑7B‑Instruct‑v0.3、Vicuna‑7B‑v1.5）上进行对比，B‑DPO在安全基准上提升约5–8%（如StrongReject 92.01% vs 86.58%），同时在通用基准（GSM8k、AdvGLUE等）保持或略提升性能，证明其在提升安全性的同时不显著牺牲通用能力。

**⚠️ 局限性**

局限性：仅在DPO框架内验证，未扩展到更复杂的RLHF方法（如PPO）；实验仅覆盖文本模态，未考虑多模态场景；预计算互信息导致训练前额外时间开销（约0.4h），在大规模部署时需进一步优化。

---

## 244. Typography-Based Monocular Distance Estimation Framework for Vehicle Safety Systems

**arXiv ID:** 2603.22781 | [PDF](https://arxiv.org/pdf/2603.22781v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 245. RadTimeline: Timeline Summarization for Longitudinal Radiological Lung Findings

**arXiv ID:** 2603.22820 | [PDF](https://arxiv.org/pdf/2603.22820v1)

**作者:** Sitong Zhou `[一作]`, Mari Ostendorf `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了一个用于纵向放射学报告的结构化时间线生成任务，提出二维时间线格式并实现三步LLM流程（发现提取 → 组名生成 → 分组分配）来实现可验证的结构化摘要。

**💡 创新点**

创新点在于：① 设计了专门的二维时间线框架，使时间维度和发现维度可并排查看；② 通过LLM三步流程实现从原始报告到可验证时间线的全自动化；③ 构建并公开了RadTimeline评估数据集与针对时间线质量的评估指标。

**🔧 技术方法**

使用了Llama 3.1 8B Instruct和GPT‑4o两款LLM来完成发现提取、组名生成和分组分配；还尝试了基于E5‑Mistral的嵌入式方法进行分组分配；评估时采用ROUGE‑L、CoNLL F1等指标。

**📊 数据集**

数据集为10名肺癌筛查患者的纵向胸部影像报告（含胸CT、胸X光、腹部CT），每位患者至少4份报告，手工校正后形成金标时间线，被称为RadTimeline。

**📈 对比分析**

与两份金标组别进行CoNLL F1对比；GPT‑4o在组名生成与分组分配上接近人工水平（≈82–84 F1），Llama 3.1略逊；在oracle组名下的嵌入式方法也可达到≈84 F1，表明嵌入方法在组名质量足够时能作为LLM的高效替代。

**⚠️ 局限性**

主要限制包括：样本量仅10名患者；仅评估两款LLM，缺乏更广泛模型比较；发现与组名的自动对齐依赖ROUGE，可能导致误判；金标构建过程中可能偏向LLM使用的语言风格；未进行完整的自动化时间线评估（缺乏银标或人工核对）。

---

## 246. When AI Shows Its Work, Is It Actually Working? Step-Level Evaluation Reveals Frontier Language Models Frequently Bypass Their Own Reasoning

**arXiv ID:** 2603.22816 | [PDF](https://arxiv.org/pdf/2603.22816v1)

**作者:** Abhinaba Basu `[一作]` (Indian Institute of Information Technology Allahabad), Pavan Chakraborty `[通讯]` (Indian Institute of Information Technology Allahabad)

**通讯引用:** 1512 | [OpenAlex ID](https://openalex.org/A5023091561)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 10 款前沿语言模型在情感、数学、主题分类和医学 QA 四个任务上进行“逐步推理”可信度评估，量化每一步的必要性、充分性和顺序敏感性。

**💡 创新点**

提出低成本、可通过 API 访问的逐步级别评估方法，并发现大多数模型的推理步骤仅为装饰性叙述；仅有 MiniMax-M2.5 与 Kimi-K2.5 在部分任务上表现出真实推理依赖。

**🔧 技术方法**

使用必要性测试（删除一步是否改变答案）、充分性测试（单步是否能恢复答案）以及混洗测试来判断推理过程的真实性；结合 logit‑lens 与注意力分析验证行为结果。

**📊 数据集**

使用公开数据集：SST‑2（情感）、GSM8K（小学数学）、AG News（主题分类）和 MedQA（医学多项选择），每项约 500 条样本。

**📈 对比分析**

对比 10 款模型的必要性率、充分性率和混洗率；结果显示大多数模型的必要性低于 17%、充分性高于 60%，表明推理多为装饰；MiniMax 在情感任务上必要性 37%、充分性 61%；Kimi 在主题分类上必要性 39%、充分性 41%；整体准确率均在 77–97% 之间。

**⚠️ 局限性**

局限性包括：仅评估句子级推理，可能忽略更细粒度或更粗粒度的真实推理；必要性测试仅为必要条件；模型输出单步或不产生推理时无法评估；对不同提示和多语言场景的适用性未充分验证；部分任务样本量有限，结果可能受限。

---

## 247. Curve resampling based high-quality high-order unstructured quadrilateral mesh generation

**arXiv ID:** 2603.22780 | [PDF](https://arxiv.org/pdf/2603.22780v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 248. DiSCo: Diffusion Sequence Copilots for Shared Autonomy

**arXiv ID:** 2603.22787 | [PDF](https://arxiv.org/pdf/2603.22787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 249. Gau-Occ: Geometry-Completed Gaussians for Multi-Modal 3D Occupancy Prediction

**arXiv ID:** 2603.22852 | [PDF](https://arxiv.org/pdf/2603.22852v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 250. Focus, Don't Prune: Identifying Instruction-Relevant Regions for Information-Rich Image Understanding

**arXiv ID:** 2603.22815 | [PDF](https://arxiv.org/pdf/2603.22815v1)

**作者:** Mincheol Kwon `[一作]` (Korea University), Jinkyu Kim `[通讯]` (Korea University)

**通讯引用:** 3059 | [OpenAlex ID](https://openalex.org/A5061842716)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为PinPoint的两阶段框架，旨在识别与指令相关的图像区域，并提取细粒度的视觉特征，以提高推理能力和效率。

**💡 创新点**

创新点在于引入了指令-区域对齐机制，通过视觉输入和文本指令共同定位相关区域，并提供了新的注释以增强对复杂VQA基准的监督。

**🔧 技术方法**

使用了视觉变换器（ViT）和可学习的引导查询，通过区域选择和区域细化两个阶段来处理视觉信息。

**📊 数据集**

在InfographicVQA、MultiPageDocVQA、SinglePageDocVQA和GQA等四个具有挑战性的VQA基准上进行了评估，并提供了新的数据集注释。

**📈 对比分析**

与现有方法相比，PinPoint在准确性上表现优越，同时显著减少了计算开销，尤其在处理高分辨率和信息密集的图像时，性能提升明显。

**⚠️ 局限性**

限制在于该方法依赖于视觉输入和文本指令的质量，若输入信息不准确，可能影响最终的推理结果。

---

## 251. AgriPestDatabase-v1.0: A Structured Insect Dataset for Training Agricultural Large Language Model

**arXiv ID:** 2603.22777 | [PDF](https://arxiv.org/pdf/2603.22777v1)

**作者:** Yagizhan Bilal Durak `[一作]` (Sam Houston State University), Syed Hasib Akhter Faruqui `[通讯]` (Sam Houston State University)

**通讯引用:** 244 | [OpenAlex ID](https://openalex.org/A5032281595)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

构建了AgriPestDatabase‑v1.0九种入侵性昆虫的结构化文本数据库，并使用LoRA对轻量级LLM进行微调，以实现农田虫害决策支持。

**💡 创新点**

创新点在于将专家审核的结构化虫害信息与自动生成多类型问答对相结合，利用语义对齐评估指标证明轻量级LLM（≤7B）能在离线环境下高效完成专业问答。

**🔧 技术方法**

采用LoRA参数高效微调、结构化文本抽取与分块、LLM提示自动生成多类型问答、以及BLEU、ROUGE、嵌入相似度和Token‑F1等多维度评估技术。

**📊 数据集**

使用的数据集为AgriPestDatabase‑v1.0（九种侵入性昆虫的全面描述），以及公开虫害数据库和学术文献整理的原始文本。

**📈 对比分析**

通过BLEU、ROUGE、嵌入相似度、Token‑F1等指标，对Mistral‑7B、LLaMA‑3.1‑8B、Qwen2.5‑7B进行微调后评估，Mistral‑7B整体得分88.9%，显著优于其他模型。

**⚠️ 局限性**

限制在于模型仍易出现过长或截断回答、在超长输入下生成有限，缺乏工具调用和动态推理能力，未来需增强多模态支持与更稳健的推理管控。

---

## 252. Towards The Implicit Bias on Multiclass Separable Data Under Norm Constraints

**arXiv ID:** 2603.22824 | [PDF](https://arxiv.org/pdf/2603.22824v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 253. UniQueR: Unified Query-based Feedforward 3D Reconstruction

**arXiv ID:** 2603.22851 | [PDF](https://arxiv.org/pdf/2603.22851v1)

**作者:** Chensheng Peng `[一作]` (Applied Intuition), Wei Zhan `[通讯]` (Applied Intuition)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的基于查询的前馈3D重建框架，能够从无姿态图像中一次前向推断出稀疏的3D高斯集合，恢复完整的几何和外观。

**💡 创新点**

创新点在于使用可学习的3D查询作为全局空间锚点，解耦视角信息，利用交叉注意力将多视图特征聚合到查询中，并在训练时通过新视角监督引导查询填补遮挡区域，从而实现更完整的场景重建，显著减少所需高斯数量。

**🔧 技术方法**

技术包括Vision Transformer编码器、交替注意力聚合、多视图特征的交叉注意力与查询自注意力、基于可学习查询的高斯生成、可微高斯渲染（Gaussian Splatting）以及使用RGB+深度损失进行监督。

**📊 数据集**

在MapAnything、CUT3R、DL3DV-10K、ScanNet++、BlenderMVS、Co3D等大型多视图数据集上训练，并在Mip-NeRF 360和VR-NeRF等数据集上进行新视角合成评估，也在RealEstate10K和Co3Dv2等数据集上评估相机姿态。

**📈 对比分析**

与现有前馈方法如AnySplat、NoPoSplat、MVSplat等比较，模型在稀疏视图下实现了PSNR 20.23 dB（相较于AnySplat略高），几何误差下降至0.038，使用约1/15的高斯数量、40%更少GPU内存、2.4倍更快推断；在密集视图下也能提供优质初始化，显著提升后期优化性能。

**⚠️ 局限性**

局限性：无法处理动态场景；当前框架仅在静态室内/室外环境下评估，未考虑时序变化。

---

## 254. Learning What Matters Now: Dynamic Preference Inference under Contextual Shifts

**arXiv ID:** 2603.22813 | [PDF](https://arxiv.org/pdf/2603.22813v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 255. Universal and efficient graph neural networks with dynamic attention for machine learning interatomic potentials

**arXiv ID:** 2603.22810 | [PDF](https://arxiv.org/pdf/2603.22810v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 256. A Video Steganography for H.265/HEVC Based on Multiple CU Size and Block Structure Distortion

**arXiv ID:** 2603.22850 | [PDF](https://arxiv.org/pdf/2603.22850v1)

**作者:** Xiang Zhang `[一作]` (Nanjing University of Information Science and Technology), Zhangjie Fu `[通讯]` (Nanjing University of Information Science and Technology)

**通讯引用:** 5876 | [OpenAlex ID](https://openalex.org/A5066341740)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `fede83ac-7505-405f-ab37-e7284695c47f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于HEVC/H.265的多CU尺寸与块结构失配的视频隐写方案，利用块结构恢复现象与CBSSM指标实现低比特率提升、高清视觉质量与高容量隐藏。

**💡 创新点**

创新点在于：①发现CU块结构恢复现象并提出CU块结构稳定性度量（CBSSM）；②基于多CU尺寸的映射规则与三层块结构失配失真函数，最小化对CU块结构的破坏并显著提升抗隐写分析性能。

**🔧 技术方法**

技术手段包括HEVC/H.265编码器的RDO与STC隐写器、三层失真函数、CBSSM指标、LibSVM分类器以及多种隐写分析方法（Zhao、Sheng、Li、Huang、Zhai、Dai）。

**📊 数据集**

实验使用33个标准HEVC测试序列（如BasketballPass、PeopleOnStreet等），分辨率416×240至2560×1600，QP分别为26/32/38，帧数在300–600之间。

**📈 对比分析**

与Tew、Dong、Yang、Wang四种主流块结构隐写算法在PSNR、BIR、容量、主观质量以及多种隐写分析（Zhao、Sheng、Li、Huang、Zhai、Dai）和CBSSM隐写分析中对比，结果显示本文方案在视觉质量、比特率增量、容量方面均优于对手，且CBSSM检测率接近50%，抗隐写分析能力显著提升。

**⚠️ 局限性**

局限性：方案非盲，需要原始CU结构或额外边信息；在极高容量或高QP场景下CBSSM检测率仍会上升，未来需进一步提升在高容量/高QP条件下的抗分析性能。

---

## 257. Approximating the Shapley Value of Minimum Cost Spanning Tree Games: An FPRAS for Saving Games

**arXiv ID:** 2603.22843 | [PDF](https://arxiv.org/pdf/2603.22843v1)

**作者:** Takumi Jimbo `[一作]` (Institute of Science Tokyo), Tomomi Matsui `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 2019 | [OpenAlex ID](https://openalex.org/A5031287136)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在最小成本生成树（MCST）游戏中计算Shapley值的近似问题，提出将MCST游戏转化为保存游戏（MCST‑saving game）的框架，并在此基础上设计了一个完全多项式时间随机逼近方案（FPRAS）。

**💡 创新点**

创新点在于通过保存游戏的结构性转化实现相对误差（乘法误差）逼近，首次给出了MCST‑saving game中null玩家的必要与充分条件、非null玩家的下界，并在此基础上证明并实现了FPRAS。

**🔧 技术方法**

采用的技术包括Monte Carlo随机采样、图论中最小生成树的性质、加法性分解、Hoeffding不等式以及对权重离散化的 0–1 转换，从而获得了理论上的样本复杂度与时间复杂度分析。

**📊 数据集**

实验使用生成的 0–1 加权完全图实例，起始于所有边权为 0 的 chordal 图，随后添加权值为 1 的边，规模从 n=3 到 10，确保每个实例中至少存在一个非 null 玩家。

**📈 对比分析**

与理论样本复杂度进行比较，实验结果显示所需样本规模与 1/ε² 成线性关系，且随 n 的增大与理论上 n²(n−1)⁴ 的增长趋势一致，表明实现的 FPRAS 在实际中能以高概率满足给定的相对误差。

**⚠️ 局限性**

限制在于算法主要针对 MCST 及其保存游戏，对权重分布不连续或非 0–1 的情况仍需更复杂的处理；此外，时间复杂度仍为 O(n²M)，对大规模实例的实际性能可能受限。

---

## 258. "Don't Look, But I Know You Do": Norms and Observer Effects in Shared LLM Accounts

**arXiv ID:** 2603.22822 | [PDF](https://arxiv.org/pdf/2603.22822v1)

**作者:** Ji Eun Song `[一作]` (Seoul National University), Joongseek Lee `[通讯]` (Seoul National University)

**通讯引用:** 543 | [OpenAlex ID](https://openalex.org/A5003626110)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对245名用户的问卷和36名受访者的访谈，构建了LLM账号共享的四种类型模型，并探讨了共享中的隐式规范与观察者效应。

**💡 创新点**

创新点在于将LLM共享视为技术再 appropriation，提出了基于拥有者活跃度与费用分摊的2×2共享类型，并阐释了认知痕迹可见性如何诱发隐式规范的破裂与观察者效应。

**🔧 技术方法**

主要采用混合研究方法：量化问卷、半结构化访谈、主题编码与可靠性检验（Cohen's κ）等质性与定量分析技术。

**📊 数据集**

数据集包括245份问卷回复（平均年龄34岁）和36份访谈录音/文字记录，涵盖ChatGPT、Claude、Perplexity、Midjourney等LLM服务使用情况。

**📈 对比分析**

研究并未进行算法或性能对比，而是通过对共享类型、规范形成及观察者效应的定性比较，阐明不同共享情境下行为与规范的差异。

**⚠️ 局限性**

局限性包括：依赖自我报告可能产生偏差；大部分案例集中于ChatGPT，缺乏其他LLM多样性；横断面设计无法捕捉随时间演变的规范；未使用真实使用日志验证行为。

---

## 259. Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought

**arXiv ID:** 2603.22847 | [PDF](https://arxiv.org/pdf/2603.22847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 260. Combinatorial Privacy: Private Multi-Party Bitstream Grand Sum by Hiding in Birkhoff Polytopes

**arXiv ID:** 2603.22808 | [PDF](https://arxiv.org/pdf/2603.22808v1)

**作者:** Praneeth Vepakomma `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Praneeth Vepakomma `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 6254 | [OpenAlex ID](https://openalex.org/A5022815450)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并实现了一种基于Birkhoff多面体的编码，利用双层安全架构实现隐私安全的布尔求和协议，

**💡 创新点**

创新点在于引入“组合隐私”范式，提出双层PolyVeil协议使服务器具备信息理论安全，而聚合器面临#P难度的推断障碍，

**🔧 技术方法**

采用Birkhoff多面体置换编码、随机置换、混合判别、永久函数、Berry‑Esseen定理等技术进行安全与差分隐私分析，

**📊 数据集**

未使用具体数据集，全部基于理论分析和实验参数设定，

**📈 对比分析**

与传统DP、MPC、HE等方法相比，通信成本低（压缩版为O(k)），在相同精度与隐私参数下保持或优于现有方案，

**⚠️ 局限性**

局限在于DP与#P难度工作在不同参数尺度，难以在信号可检测的参数范围内提供非空洞DP保障，且仅在半恶意模型下验证，未来需在更严苛模型和更高参数下进一步证明安全性。

---

## 261. Avoiding Over-smoothing in Social Media Rumor Detection with Pre-trained Propagation Tree Transformer

**arXiv ID:** 2603.22854 | [PDF](https://arxiv.org/pdf/2603.22854v1)

**作者:** Chaoqun Cui `[一作]` (Beijing Jiaotong University), Caiyan Jia `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 2500 | [OpenAlex ID](https://openalex.org/A5085282915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于Transformer的预训练传播树Transformer（P2T3），用于在谣言传播树中捕获完整会话链信息，避免传统GNN的过平滑问题。

**💡 创新点**

创新点包括：①用Token-wise嵌入将传播树拆分为会话链并加入链标识、深度和类型信息；②在大规模无标签数据上最大化互信息进行预训练；③证明Transformer在此结构上可扩展且不易过平滑。

**🔧 技术方法**

采用Transformer Encoder、Token-wise Embedding、互信息最大化预训练、交叉熵微调等技术。

**📊 数据集**

使用UWeibo、UTwitter两大无标签数据集，以及Weibo、DRWeibo、Twitter15、Twitter16四个标注基准集进行实验。

**📈 对比分析**

与PLAN、HD-TRANS、BiGCN、ClaHi-GAT、GACL、DDGCN、RAGCL等基线相比，P2T3在所有数据集上均取得最高准确率，且在少样本和大模型预训练场景中表现更优。

**⚠️ 局限性**

局限在于：仅关注文本传播结构，尚未整合多模态信息；对极深度传播树长链的处理仍有限；需要进一步验证其跨平台的泛化能力。

---

## 262. Agent Audit: A Security Analysis System for LLM Agent Applications

**arXiv ID:** 2603.22853 | [PDF](https://arxiv.org/pdf/2603.22853v1)

**作者:** Haiyue Zhang `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**通讯引用:** 3436 | [OpenAlex ID](https://openalex.org/A5057711796)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个用于LLM代理应用的静态安全分析工具，可检测工具函数、提示构造与部署配置中的安全漏洞。

**💡 创新点**

创新点在于结合了 agent‑aware 代码分析、MCP 配置结构解析、跨规则引擎与置信度分层，全面覆盖 OWASP Agentic Security Initiative 的 10 个类别。

**🔧 技术方法**

采用 AST 数据流、污点分析、凭证检测、配置解析、权限风险检查等技术，并通过 RuleEngine 进行规则映射、去重与置信度分层。

**📊 数据集**

使用自研的 Agent‑Vuln‑Bench（22 个样本、42 条注释漏洞）以及公开的 CVE 重现案例做评估。

**📈 对比分析**

与 Bandit、Semgrep 在 AVB 上对比，Recall 95.2%/F1 0.91，扫描时间 0.87 秒，子秒级 CI 检查；相比之下 Bandit/Semgrep recall 仅 23–29%。

**⚠️ 局限性**

局限在于仅做单进程污点、仅支持 Python、缺少跨文件流分析、未执行运行时注入模拟，阈值为经验调优。

---

## 263. Span Modeling for Idiomaticity and Figurative Language Detection with Span Contrastive Loss

**arXiv ID:** 2603.22799 | [PDF](https://arxiv.org/pdf/2603.22799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 264. UAV-DETR: DETR for Anti-Drone Target Detection

**arXiv ID:** 2603.22841 | [PDF](https://arxiv.org/pdf/2603.22841v1)

**作者:** Jun Yang `[一作]` (Northwest Polytechnical University), Jianxiong Yu `[通讯]` (Northwest Polytechnical University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对小目标无人机检测的轻量级检测框架UAV-DETR，能够在复杂背景下实现实时高精度检测。

**💡 创新点**

核心创新包括：Wavelet Transform Convolution (WTConv) 轻量化骨干网络、Sliding Window Self‑Attention (SWSA) 的局部注意力编码、Efficient Cross‑Scale Feature Recalibration and Fusion Network (ECFRFN) 的多尺度融合，以及结合 Inner‑CIoU 与 NWD 的混合回归损失。

**🔧 技术方法**

技术手段涵盖：Transformer‑based DETR 架构、波形卷积提升高频细节、滑窗注意力降低噪声、特征重校与融合、混合几何与分布式损失。

**📊 数据集**

使用自建 14,713 张图像的 UAV 数据集进行训练和评估，同时在公开 DUT‑ANTI‑UAV 基准上验证泛化能力。

**📈 对比分析**

与 RT‑DETR、YOLOv8/10/11/12、Hyper‑YOLO、VRF‑DETR 等 11 大模型对比，UAV‑DETR 在自建数据集上 mAP_50:95 达 62.56%（比 RT‑DETR 提升 6.61%），参数约 11.96M，FLOPs 66.7G；在 DUT‑ANTI‑UAV 上 mAP_50:95 达 67.15%，精度、召回率均领先同类模型。

**⚠️ 局限性**

局限性：仍易被形态相似的鸟类误检、对严重视觉伪装的误检；相较基线，特征重校与融合模块带来约 17% 的额外 FLOPs，需要进一步剪枝/量化以满足极低功耗边缘设备。

---

## 265. Instrument-Splatting++: Towards Controllable Surgical Instrument Digital Twin Using Gaussian Splatting

**arXiv ID:** 2603.22792 | [PDF](https://arxiv.org/pdf/2603.22792v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 266. Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis

**arXiv ID:** 2603.22786 | [PDF](https://arxiv.org/pdf/2603.22786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 267. Exposure-Normalized Bed and Chair Fall Rates via Continuous AI Monitoring

**arXiv ID:** 2603.22785 | [PDF](https://arxiv.org/pdf/2603.22785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 268. URA-Net: Uncertainty-Integrated Anomaly Perception and Restoration Attention Network for Unsupervised Anomaly Detection

**arXiv ID:** 2603.22840 | [PDF](https://arxiv.org/pdf/2603.22840v1)

**作者:** Wei Luo `[一作]` (Tsinghua University), Zechao Li `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 9592 | [OpenAlex ID](https://openalex.org/A5017096005)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种名为URA-Net的无监督异常检测网络，显式地将异常恢复为正常模式并通过特征重建实现定位。

**💡 创新点**

三大创新：
• Feature-level Artificial Anomaly Synthesis Module（FASM）在特征层生成多样化的人工异常；
• Uncertainty-Integrated Anomaly Perception Module（UIAPM）采用贝叶斯神经网络输出均值和方差，结合判别学习实现对异常区域与模糊边界的粗略感知；
• Restoration Attention Module（RAM）在Transformer框架中用掩码的Key‑Value替代自注意力，仅利用全局正常语义信息恢复异常特征，避免异常信息被传播。

**🔧 技术方法**

使用预训练CNN（WideResNet50）提取多尺度语义特征；FASM、UIAPM、RAM组成的Transformer块；重建损失结合MSE、cosine、全局相似度；贝叶斯神经网络实现不确定性估计；判别学习与KL约束提升鲁棒性。

**📊 数据集**

工业数据集：MVTec AD、BTAD；医学数据集：OCT-2017。

**📈 对比分析**

与过去三年SOTA方法（PatchCore、FOD、UTRAD、DRAEM等）比较，URA-Net在MVTec AD图像AUROC 99.4%、像素AUROC 98.5%，BTAD图像AUROC 96.0%、像素AUROC 97.6%，OCT-2017图像AUROC 98.6%、F1 97.1%、ACC 95.7%。同时FPS≈55.1，参数≈97M，FLOPs≈30G，速度与精度均优于主要对手。

**⚠️ 局限性**

对逻辑异常（如位置错误）检测仍不理想，模型在缺乏语义信息时会漏检。

---

## 269. Transformers Trained via Gradient Descent Can Provably Learn a Class of Teacher Models

**arXiv ID:** 2603.22801 | [PDF](https://arxiv.org/pdf/2603.22801v1)

**作者:** Chenyang Zhang `[一作]` (University of Hong Kong), Yuan Cao `[通讯]` (University of Hong Kong)

**通讯引用:** 9760 | [OpenAlex ID](https://openalex.org/A5101989491)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过理论分析，证明一层Transformer能够通过梯度下降学习一类具有双线性结构的教师模型（包括卷积+平均池化、图卷积、稀疏标记选择和组稀疏线性预测器），并给出收敛速率Θ(1/T)以及对离散分布泛化的上界；

**💡 创新点**

创新点在于提出统一的双线性教师模型框架，证明一层Transformer在梯度下降下能完整恢复教师模型参数并实现最佳风险，同时给出匹配的下界和OOD泛化保证，首次将Transformer学习过程与多种常见模型对齐；

**🔧 技术方法**

采用简化的“位置仅”自注意力Transformer、全局梯度下降优化、Frobenius范数损失以及数学分析中的矩阵分解与梯度动态推导；

**📊 数据集**

实验使用合成高斯数据以及MNIST手写数字数据（将第一层CNN+平均池化提取为教师模型），并在真实图像上验证；

**📈 对比分析**

与教师模型的误差对比显示训练误差与OOD误差快速收敛至理论上限，余弦相似度超过0.9，证明Transformer能够逼近教师参数并保持优秀的泛化；

**⚠️ 局限性**

局限在于仅考虑单层Transformer、简化的注意力结构、数据需满足高斯或有限二阶矩假设，尚未扩展到多层网络或更复杂的实战设置。

---

## 270. Reliable Classroom AI via Neuro-Symbolic Multimodal Reasoning

**arXiv ID:** 2603.22793 | [PDF](https://arxiv.org/pdf/2603.22793v1)

**作者:** Sina Bagheri Nezhad `[一作]` `[通讯]`, Sina Bagheri Nezhad

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套神经-符号化课堂推理框架（NSCR），将多模态课堂数据先进行感知基底化，再映射为结构化符号事实，接着通过可执行规则/程序进行推理，并在治理层面实现不确定性、保留与隐私策略；同时构建了面向教育推理的基准任务集合；

**💡 创新点**

创新点在于将传统的端到端黑盒推断拆解为可解释的多层流程：感知->符号抽象->可执行推理->治理；强调符号事实的构造、可追溯性与可验证性，并提出了围绕不确定性、可选择预测与隐私的完整评估框架；

**🔧 技术方法**

使用的技术包括：多模态感知模块（如姿态、视线、语音识别、语义解析）、结构化符号事实（谓词-参数-时间-置信度-来源），基于规则、可执行程序或LLM生成代码的可执行推理，以及阈值、政策约束的治理层；

**📊 数据集**

论文未使用具体公开数据集，侧重于提出框架与基准设计，建议后续使用多学科、多语言、多课堂布局的真实课堂录制数据进行评估；

**📈 对比分析**

没有给出实验对比与性能指标；论文仅提出评估层次与指标（感知质量、符号化精度、推理正确性、可选择预测风险、教师人类评估等），并未在任何数据集上进行实际验证；

**⚠️ 局限性**

主要局限：缺乏实测数据与性能验证；符号谓词的设计需依赖教师/教育专家，若失配可能导致误导；规则与LLM生成代码可能脆弱或产生不恰当推理；高成本的构造对齐标注；并非完全消除监控风险，符号记录仍可能泄露敏感信息。

---

## 271. MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects

**arXiv ID:** 2603.22839 | [PDF](https://arxiv.org/pdf/2603.22839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 272. Analysing LLM Persona Generation and Fairness Interpretation in Polarised Geopolitical Contexts

**arXiv ID:** 2603.22837 | [PDF](https://arxiv.org/pdf/2603.22837v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 273. MVRD-Bench: Multi-View Learning and Benchmarking for Dynamic Remote Photoplethysmography under Occlusion

**arXiv ID:** 2603.22826 | [PDF](https://arxiv.org/pdf/2603.22826v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 274. Empirical Comparison of Agent Communication Protocols for Task Orchestration

**arXiv ID:** 2603.22823 | [PDF](https://arxiv.org/pdf/2603.22823v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 275. Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference

**arXiv ID:** 2603.22821 | [PDF](https://arxiv.org/pdf/2603.22821v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 276. Caterpillar of Thoughts: The Optimal Test-Time Algorithm for Large Language Models

**arXiv ID:** 2603.22784 | [PDF](https://arxiv.org/pdf/2603.22784v1)

**作者:** Amir Azarmehr `[一作]` (Northeastern University), Alma Ghafari `[通讯]` (Northeastern University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5104278897)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LLM推理阶段设计并分析了最优的回溯算法，证明最优搜索树为“caterpillar”树，并基于此提出了新的测试时算法 CaT。

**💡 创新点**

创新点在于：①用可观测马尔科夫链模型正式刻画回溯推理；②证明最优策略仅需极有限的回溯，生成的树为“caterpillar”；③提出能在噪声环境下近似最优的 Stable 算法；④基于理论构建的 CaT 在实践中实现了更高效的推理。

**🔧 技术方法**

使用的技术包括：可观测马尔科夫链模型、回溯（rewinding）策略、softmax 选择、Dijkstra 类算法计算期望到达时间、Laplace 噪声下的估计与 mean‑median 子程序、以及在 LLM 上的提示工程和分数估计。

**📊 数据集**

使用的数据集：Game of 24（4nums 901‑1000 题目）和 5×5 crossword（GooBix 156 题目，评测 20 题）。

**📈 对比分析**

与 Tree‑of‑Thoughts（ToT）对比：在 Game of 24 上 CaT 成功率 81% 对比 74%，token 15.3k 对比 19.2k；在 crossword 上 Word/Letter/GameSolved 分别提升，且平均 token 更少。整体表现更优、成本更低。

**⚠️ 局限性**

限制：对 (x) 的估计误差敏感，误估会导致非最优或停滞；需要对马尔科夫链或其近似有先验；理论模型与实际 LLM 推理的细节仍有差距，尚未完全覆盖所有实际场景。

---

## 277. CATNAV: Cached Vision-Language Traversability for Efficient Zero-Shot Robot Navigation

**arXiv ID:** 2603.22800 | [PDF](https://arxiv.org/pdf/2603.22800v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 278. PhotoAgent: A Robotic Photographer with Spatial and Aesthetic Understanding

**arXiv ID:** 2603.22796 | [PDF](https://arxiv.org/pdf/2603.22796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. It Takes Two: A Duet of Periodicity and Directionality for Burst Flicker Removal

**arXiv ID:** 2603.22794 | [PDF](https://arxiv.org/pdf/2603.22794v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 280. Know3D: Prompting 3D Generation with Knowledge from Vision-Language Models

**arXiv ID:** 2603.22782 | [PDF](https://arxiv.org/pdf/2603.22782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 281. KARMA: Knowledge-Action Regularized Multimodal Alignment for Personalized Search at Taobao

**arXiv ID:** 2603.22779 | [PDF](https://arxiv.org/pdf/2603.22779v1)

**作者:** Zhi Sun `[一作]` (Taobao & Tmall Group of Alibaba), Haihong Tang `[通讯]` (Taobao & Tmall Group of Alibaba)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出KARMA框架，利用训练阶段的语义可解码正则化来缓解LLM在个性化搜索中的知识-动作差距，避免语义崩溃并提升检索与排序性能。

**💡 创新点**

①将语义可解码作为训练专用正则化而非辅助生成；②设计双路径解码器（历史条件生成与嵌入条件重建）以约束检索嵌入；③采用两阶段对齐训练策略；④证明扩散模型更适合作为语义重构而非检索嵌入生成。

**🔧 技术方法**

使用基于Transformer的连续token接口（Qwen3系列），pairwise cross‑entropy动作对齐损失，语义生成与重建解码头，视觉解码器（扩散/流匹配），两阶段学习调度以及离线评估指标gAUC、HR@K、JS@50。

**📊 数据集**

Taobao搜索日志数据集，包含用户历史交互、下一点击项目以及曝光未点击的负样本。

**📈 对比分析**

与仅动作对齐的Pairwise‑CE基线对比；KARMA在离线实验中gAUC提升+0.97，HR@200+22.57，HR@1000+21.19，JS@50+2.26；多模态扩展进一步提升+1.38 gAUC；线上A/B测试在排名阶段实现Item Click提升0.5%。

**⚠️ 局限性**

仍依赖预训练LLM和图像编码器；扩散模型在检索嵌入生成上表现欠佳，需更合适的生成方法；注意力坍塌的诊断仍属经验性；对冷启动/长尾查询的具体效果未充分验证。

---

## 282. PhySe-RPO: Physics and Semantics Guided Relative Policy Optimization for Diffusion-Based Surgical Smoke Removal

**arXiv ID:** 2603.22844 | [PDF](https://arxiv.org/pdf/2603.22844v1)

**作者:** Zining Fang `[一作]` (Southeast University), Xiaowei Hu `[通讯]` (South China University Of Technology)

**通讯引用:** 9392 | [OpenAlex ID](https://openalex.org/A5027851405)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于扩散模型的手术烟雾去除框架PhySe-RPO。

**💡 创新点**

创新点在于将扩散恢复转化为组相对策略优化（critic-free RL），并结合物理颜色先验与视觉概念语义奖励。

**🔧 技术方法**

使用扩散模型、组相对策略优化、CLIP视觉概念、无参考图像质量评估等技术。

**📊 数据集**

使用合成烟雾–清洁配对数据（2000对）和10k帧真实手术视频作为无监督优化。

**📈 对比分析**

与DCP、Desmoke_LAP、SelfSVD、PFAN、Dehamer、LightDiff等方法对比，在无参考质量指标上取得最高得分；在公开配对数据上PSNR 21.03 dB，CIEDE2000 7.65，领先对手。

**⚠️ 局限性**

局限在于对真实烟雾分布的建模仍受合成数据偏差影响，且对极端烟雾或高反射场景效果尚未完全验证。

---

## 283. Efficient Hallucination Detection: Adaptive Bayesian Estimation of Semantic Entropy with Guided Semantic Exploration

**arXiv ID:** 2603.22812 | [PDF](https://arxiv.org/pdf/2603.22812v1)

**作者:** Qiyao Sun `[一作]` (National University of Defense Technology), Qingyong Hu `[通讯]` (Intelligent Game and Decision Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自适应贝叶斯框架，用于动态估计语言模型生成结果的语义熵，从而实现更高效的幻觉检测。

**💡 创新点**

创新点在于：① 用层次贝叶斯模型将语义熵拆解为语义类别数与类别概率的联合推断；② 引入加权困惑度自适应先验，并利用重要性采样的引导探索加速语义空间覆盖；③ 通过方差阈值实现采样次数的动态调整，显著降低计算成本。

**🔧 技术方法**

主要技术包括：层次贝叶斯框架、截断 Dirichlet 后验、加权困惑度、重要性采样与引导探索、语义等价聚类（使用 DeBERTa‑v3 进行 NLI 语义匹配），以及 AUROC 评估。

**📊 数据集**

实验数据集涵盖四个开放式问答任务：CoQA、TriviaQA、TruthfulQA、SimpleQA，使用 Llama‑2‑7B、Llama‑3.1‑8B、Mistral‑Small‑24B 三大 LLM。

**📈 对比分析**

与四种基线（P(True)、SAR、SE、SE_SDLG）在 AUROC 上对比，低预算场景下平均提升 12.6%，在 23/24 设定中均取得最高分；在高预算下仍保持领先，显示自适应采样与贝叶斯推断的优势。

**⚠️ 局限性**

局限性包括：① 对 LLM 生成概率的约束依赖于模型分布的准确性；② 对极其复杂查询仍需较多采样，导致效率下降；③ 目前仅在文本域验证，缺乏多模态或更大规模模型的扩展评估。

---

## 284. TorR: Towards Brain-Inspired Task-Oriented Reasoning via Cache-Oriented Algorithm-Architecture Co-design

**arXiv ID:** 2603.22855 | [PDF](https://arxiv.org/pdf/2603.22855v1)

**作者:** Hyunwoo Oh `[一作]` (University of California, Irvine), Mohsen Imani `[通讯]` (University of California, Irvine)

**通讯引用:** 6810 | [OpenAlex ID](https://openalex.org/A5033221192)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c`

**🎯 论文内容**

提出一种基于事件驱动编码器和高维向量（HDC）关联对齐的任务导向对象检测框架，并实现了针对边缘设备的低能耗、实时推理系统。

**💡 创新点**

核心创新在于将稠密CLIP对齐转化为可复用的HDC关联匹配，利用查询缓存与δ‑更新结合可调有效维度与硬件门控，实现基于时间一致性的动态计算重用与功耗可配置。

**🔧 技术方法**

采用事件驱动感知（DVS+SNN）、高维向量（HDC）计算、查询缓存与δ‑更新、可调银行/精度门控、实时控制器以及在TSMC 28 nm工艺下的ASIC实现。

**📊 数据集**

使用包含五个日常任务提示（如 pour wine、sports、cooking、have breakfast、take a rest）的自构造数据集，并与 TOIST、iTaskCLIP 等基线进行对比评估。

**📈 对比分析**

在RTX 4090 GPU基准下，系统实现 60 FPS 时能耗约 50 mJ/帧，30 FPS 时约 113 mJ/帧；在 RT‑60/RT‑30 目标下 p95 延迟低于预算，平均 AP 为 44.27%（相当于强基线的 75–86%），且能源比 GPU 系统下降数百倍。

**⚠️ 局限性**

局限包括相对较低的检测精度（尤其在高动态场景）、对多提示或多摄像头场景的支持不足、需要手动调节阈值与门控参数，以及受 28 nm 芯片工艺的硬件资源限制。

---

## 285. CoMaTrack: Competitive Multi-Agent Game-Theoretic Tracking with Vision-Language-Action Models

**arXiv ID:** 2603.22846 | [PDF](https://arxiv.org/pdf/2603.22846v1)

**作者:** Youzhi Liu `[一作]` (Amap, Alibaba Group), Yang Cai `[通讯]` (Amap, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aaccfe5c-6b26-4208-b23c-35331481e142` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了CoMaTrack框架，通过多智能体竞争式强化学习训练视觉-语言-动作模型，实现语言指定目标的连续跟踪。

**💡 创新点**

核心创新在于将多智能体游戏理论与RL结合，形成自适应的竞争循环，让追踪器在面对不断升级的对手时自动生成难度递增的训练场景，从而显著提升鲁棒性和泛化能力。

**🔧 技术方法**

使用的技术包括：Qwen2.5VL-3B视觉语言模型、流匹配动作模块、Group Relative Policy Optimization (GRPO)、多智能体GRPO、LoRA微调、以及自定义密集与稀疏奖励设计。

**📊 数据集**

数据集涵盖EVT-Bench（STT、DT、AT）与新建的CoMaTrack-Bench（对抗性追踪场景），同时整合HM3D、MP3D、ScanQA、SYNTH-PEDES、RefCOCO、Flickr30k等导航与问答数据。

**📈 对比分析**

与TrackVLA++等7B规模基线对比，3B规模CoMaTrack在EVT-Bench上分别取得92.1%/90.3%/57.5%（SR/TR/AT）以及在CoMaTrack-Bench上以85.0% SR、82.9% TR、5.5% CR的显著优势，证明竞争式RL是驱动鲁棒性提升的关键。

**⚠️ 局限性**

局限性包括：实验仅聚焦EVT，缺乏在更广泛VLN任务上的验证；对手策略受模拟先验限制，可能与真实世界偏离；多智能体训练计算成本高且不稳定，需进一步优化采样与稳定性。

---

## 286. TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment

**arXiv ID:** 2603.22819 | [PDF](https://arxiv.org/pdf/2603.22819v1)

**作者:** Chunxia Qin `[一作]` (University of Science and Technology of China), Cong Liu `[通讯]` (iFLYTEK Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出TDATR框架，采用perceive-then-fuse策略，先通过表格细节感知学习（统一结构理解与内容识别）构建对表格结构与内容的细粒度感知，然后在融合阶段生成HTML输出并实现端到端表格识别，配合结构引导的细胞定位模块提升定位精度与可解释性。

**💡 创新点**

①表格细节感知学习：在语言模型框架下联合设计结构理解与内容识别多任务，利用海量文档数据增强鲁棒性；②结构引导细胞定位模块：利用解码器隐藏状态、行列结构掩码和多尺度视觉特征实现并行细胞定位，避免传统单独定位导致的误差积累。

**🔧 技术方法**

Swin Transformer视觉编码器、Transformer解码器、DAB-DETR细化层、双向注意力增强、Mask‑DINO样式对齐损失、行列结构掩码等；整体架构基于多模态VLM，兼容大规模预训练与轻量化微调。

**📊 数据集**

文档数据：中英网页、论文、README、内部语料（共计数千万张），用于OCR、文本定位与Markdown解析；表格数据：iFLYTAB-full、TabRecSet、PubTabNet、PubTables‑1M、FinTabNet、PubTabNet‑Val、OmniDocBench、CC‑OCR、OCRBench v2等七大公开基准。

**📈 对比分析**

在七个表格识别基准上不做数据专门微调，直接评估。相较于模块化TR、单独端到端TR和OCR‑VLM，TDATR在iFLYTAB‑full、TabRecSet、PubTabNet、PubTables‑1M等上均实现SOTA或接近SOTA的TR与TSR性能，TR精度提升15%~20%，结构识别误差显著下降，细胞定位AP_50亦优于传统方法。

**⚠️ 局限性**

对数字化表格时仍略逊于顶级TSR模型，主要因仅使用少量训练数据与长序列生成的难度；在极端低质量或极大表格场景下定位精度仍可提升；需要进一步扩大多样化表格训练集以减少对特定领域的依赖。

---

## 287. On the Complexity of Secluded Path Problems

**arXiv ID:** 2603.22818 | [PDF](https://arxiv.org/pdf/2603.22818v1)

**作者:** Tesshu Hanaka `[一作]` (Kyushu University), Daisuke Tsuru `[通讯]` (Kyushu University)

**通讯引用:** 3341 | [OpenAlex ID](https://openalex.org/A5110052164)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文研究了短隙路（Short Secluded Path）和最短隙路（Shortest Secluded Path）问题的参数化复杂度，并给出了若干新的算法与难度结论。

**💡 创新点**

创新点在于提出了针对团宽（clique-width）实现的 XP 算法，以及对邻域多样性（neighborhood diversity）和双亲覆盖数（twin cover）给出的 FPT 方案；同时首次证明最短隙路在无权图上可多项式求解，而在加权图上对最短路径距离参数化为 W[1]-hard，且给出了对应的 XP 算法。

**🔧 技术方法**

主要技术包括在团宽表达树上构造状态动态规划、利用整数线性规划（ILP）和流保守约束求解模块化路径、以及对层次图（layer graph）进行邻域递推的 DP，另外还使用了多项式时间归约与参数化归约证明 W[1]-难度。

**📊 数据集**

本文未使用公开数据集，而是以理论分析和算法证明为主，所有实验与对比均为理论复杂度与已知的参数化结果进行比较。

**📈 对比分析**

与现有仅在树宽、顶点覆盖、反馈顶点集等参数下已知的结果相比，本文在团宽下给出了完整的 XP 上界，在邻域多样性与双亲覆盖数下实现了 FPT 上界，并在加权图中首次揭示了最短隙路的 W[1]-难度，性能上虽然保持了可接受的指数增长，但仍有进一步优化空间。

**⚠️ 局限性**

主要局限包括：团宽算法的时间为 n^{O(k^2)}，邻域多样性与双亲覆盖数下的 FPT 方案在常数项上呈现 O(r^{O(r^2)}) 与 2^{O(r^2)} 的双指数开销；此外，最短隙路在加权图上的 W[1]-难度表明在此参数化下无法得到多项式时间或 FPT 解，仍需寻找更细粒度参数或近似方法。

---

## 288. "Don't Mess Up My Algorithm": Phatic Communication and Algorithmic Contagion in Meme Sharing

**arXiv ID:** 2603.22817 | [PDF](https://arxiv.org/pdf/2603.22817v1)

**作者:** Ji Eun Song `[一作]` (Seoul National University), Joongseek Lee `[通讯]` (Seoul National University)

**通讯引用:** 543 | [OpenAlex ID](https://openalex.org/A5003626110)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对21名韩国Instagram用户进行半结构化访谈，探究了DM中的表情包如何作为情感沟通与算法推荐的交叉点。

**💡 创新点**

创新点在于将表情包DM视为关系维系的phatic交流，构建了友好与不友好表情包的分类，并提出了针对算法感染的三项设计建议。

**🔧 技术方法**

采用定性研究方法：扎根理论、主题编码和访谈记录分析。

**📊 数据集**

使用的数据集为21份访谈转录文本，未使用公开数据集。

**📈 对比分析**

本文未进行量化比较或性能评估，仅以质性分析阐述用户认知与行为。

**⚠️ 局限性**

研究局限在于样本仅为韩国年轻人、只聚焦Instagram、缺乏平台日志验证DM与推荐的实际关联。

---

## 289. MP-Aggregation MP(R,2-WO) is Polynomial-Time Solvable When the Output Should Be Dichotomous Weak Preference Order

**arXiv ID:** 2603.22814 | [PDF](https://arxiv.org/pdf/2603.22814v1)

**作者:** Jiehua Chen `[一作]` `[通讯]` (Technical University Of Vienna), Jiehua Chen (Technical University Of Vienna)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文证明了在输入为一组反射二元关系，输出为二分弱序时，计算最小分歧的中位程序可以多项式时间完成。

**💡 创新点**

创新点在于将此判定问题转化为最小 s‑t 切问题，并给出了相应的流网络构造，使得二分弱序的中位问题从 NP‑难变为可解。

**🔧 技术方法**

采用流网络与最小割算法（如最大流最小割定理）来求解最小分歧。

**📊 数据集**

论文未使用实际数据集，纯理论分析。

**📈 对比分析**

由于没有实验对比，本文仅给出理论复杂度证明，证明该方法在多项式时间内得到最优解。

**⚠️ 局限性**

局限在于仅适用于二分弱序的中位问题，且构造的网络规模为 O(m²)，在极大规模实例中可能仍有空间与时间瓶颈。

---

## 290. ABSTRAL: Automatic Design of Multi-Agent Systems Through Iterative Refinement and Topology Optimization

**arXiv ID:** 2603.22791 | [PDF](https://arxiv.org/pdf/2603.22791v1)

**作者:** Weijia Song `[一作]`, Zhe Pang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于自然语言文档的多智能体系统设计框架，利用元代理通过对执行轨迹的对比分析迭代更新设计文档；

**💡 创新点**

创新点在于：①将设计知识编码为可检视、可修改的 Markdown 文档；②量化多智能体协同的“协调税”；③通过对比轨迹自动发现并加入缺失的专业角色；

**🔧 技术方法**

技术手段包括：元代理使用 Claude Sonnet 4 进行 BUILD/ANALYZE/UPDATE；后端代理采用 GPT‑4o；对比轨迹通过五类证据类（EC1‑EC5）驱动文档更新；外层循环通过图编辑距离和语义距离实现拓扑多样性；

**📊 数据集**

使用 SOPBench 银行领域 134 任务数据集（包含 40 验证 + 94 测试），评估任务满足 5 条确定性标准的通过率；

**📈 对比分析**

与公开基线（GPT‑4o 58.96%，Claude‑3.7‑Sonnet 65.67% 等）对比，本文在相同评估协议下实现 70% 验证 / 65.96% 测试通过率，显示出在多智能体集成下的提升；

**⚠️ 局限性**

局限性包括：单次实验运行方差较大（第二次跑 65%）；文档框架的设计空间可能不足以覆盖所有拓扑；与已发布 FC 基线的对比受限于框架开销；

---

## 291. From the AI Act to a European AI Agency: Completing the Union's Regulatory Architecture

**arXiv ID:** 2603.22912 | [PDF](https://arxiv.org/pdf/2603.22912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 292. Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference

**arXiv ID:** 2603.22774 | [PDF](https://arxiv.org/pdf/2603.22774v1)

**作者:** Euijun Chung `[一作]` (Georgia Institute of Technology), Hyesoon Kim `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 44517 | [OpenAlex ID](https://openalex.org/A5100615737)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多GPU LLM 推理中 CPU 造成的瓶颈进行系统评测，并给出 CPU 资源配置建议。

**💡 创新点**

揭示 CPU 饱和导致 GPU 停滞的机制，并量化 CPU 资源不足对 TTFT 的影响。

**🔧 技术方法**

使用 vLLM、CUDA Graph、NCCL、共享内存 IPC 等技术进行实验和微基准测评。

**📊 数据集**

使用 Llama 3.1 8B、Qwen 2.5 14B 模型在多种 GPU 架构上进行评测；实验中仅使用模型自身输入，没有公开数据集。

**📈 对比分析**

通过将 CPU 核心数从 (#GPU+1) 变为 2×#GPU、4×#GPU、8×#GPU 等不同配置，测量 TTFT 与 GPU/CPU 利用率，结果显示 CPU 资源充足时 TTFT 提升 1.36–5.40 倍。

**⚠️ 局限性**

实验仅覆盖 NVIDIA GPU + Intel CPU，未验证 ARM/AMD 系统；共享内存广播的结构性延迟在现有框架中难以彻底消除。

---

## 293. SoK: The Attack Surface of Agentic AI -- Tools, and Autonomy

**arXiv ID:** 2603.22928 | [PDF](https://arxiv.org/pdf/2603.22928v1)

**作者:** Ali Dehghantanha `[一作]` (University of Guelph), Sajad Homayoun `[通讯]` (Aalborg University)

**通讯引用:** 562 | [OpenAlex ID](https://openalex.org/A5054931195)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化梳理并评估Agentic LLM（工具、检索、自治）系统的攻击面，提出威胁模型、攻击分类、指标与防御框架；

**💡 创新点**

首次将Prompt注入、RAG毒化、工具/插件攻击与多Agent协同威胁整合进统一的攻击面张量，构建因果威胁图并定义可量化的攻击者意识指标；

**🔧 技术方法**

利用文献综述、系统化映射（OWASP GenAI、MITRE ATLAS）、因果威胁图、评估指标（UAR、PED、RRS等）以及防御深度框架；

**📊 数据集**

基于2023–2025年的20余篇同行评审与行业报告，未使用传统机器学习数据集；

**📈 对比分析**

通过对文献证据的归纳与指标定义进行“比较”，显示现有防御在Prompt注入、RAG毒化、工具误用等场景的有效性与不足；总体表明单一防御难以覆盖全局，需多层次结合；

**⚠️ 局限性**

局限于Agentic LLM领域，未提供真实系统实测或基准；对多Agent、生命周期与供应链攻击的评估仍属早期，缺乏统一可复现的实验环境与量化基准。

---

## 294. Multilingual KokoroChat: A Multi-LLM Ensemble Translation Method for Creating a Multilingual Counseling Dialogue Dataset

**arXiv ID:** 2603.22913 | [PDF](https://arxiv.org/pdf/2603.22913v1)

**作者:** Ryoma Suzuki `[一作]`, Michimasa Inaba `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种两阶段多LLM集成翻译方法，先利用三种不同LLM生成多样化的翻译候选，然后用另一LLM对候选进行分析、融合并生成最终翻译，应用于将日语心理咨询对话语料KokoroChat翻译成英语和中文，生成多语种心理咨询数据集Multilingual KokoroChat。

**💡 创新点**

创新点在于：①将多种LLM生成的翻译候选在对话层面进行融合，而非逐句选择；②引入“refiner”LLM对候选进行优势与不足分析并合成更优译文；③在敏感领域（心理咨询）展示了该方法能显著提升翻译自然度和语境一致性。

**🔧 技术方法**

主要技术包括：多LLM生成（GPT‑5、Gemini 2.5 Pro、Grok‑4/​Qwen‑Plus）、对话级翻译候选融合、基于LLM的分析与合成（refiner）以及参考‑free自动评估指标（XCOMET‑QE、MetricX‑QE）与配对人类评估。

**📊 数据集**

数据集：KokoroChat（6,589对话，约600k日语发言），以及其自动生成的英语/中文候选翻译。

**📈 对比分析**

比较方法：将多LLM集成翻译与单一LLM（GPT‑5、Gemini 2.5 Pro、Grok‑4/​Qwen‑Plus）在自动评估（MetricX、XCOMET）和配对人类评估中对比。结果显示，在自动评估中多LLM方法在MetricX上均取得最优（或与最佳相当）且在XCOMET上显著优于GPT，在人类评估中被评为最自然、最符合语境，优胜率远高于任何单一LLM。

**⚠️ 局限性**

局限性包括：①自动评估指标无法充分捕捉对话层面的一致性与角色适配；②refiner LLM过度强调语义忠实，导致在某些情境下译文相较于单一LLM显得不够自然；③目前方法对低资源语言对的实证不足，需进一步验证其在更广泛语言对上的有效性。

---

## 295. ForestPrune: High-ratio Visual Token Compression for Video Multimodal Large Language Models via Spatial-Temporal Forest Modeling

**arXiv ID:** 2603.22911 | [PDF](https://arxiv.org/pdf/2603.22911v1)

**作者:** Shaobo Ju `[一作]` (Xiamen University), Rongrong Ji `[通讯]` (Xiamen University)

**通讯引用:** 32243 | [OpenAlex ID](https://openalex.org/A5016080094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了面向视频多模态大语言模型（Video‑MLLMs）的无训练高比率视觉 token 剪枝方法 ForestPrune，能够在保持高准确率的前提下显著减少输入 token 数量。

**💡 创新点**

创新点在于通过构建跨帧的空间‑时间森林（token trees）来全局评估 token 重要性，结合语义相似度、空间距离与时间先后顺序，依据树深度与节点角色（根/干/叶）实现全局最优剪枝决策。

**🔧 技术方法**

使用了视觉特征提取、余弦相似度矩阵、空间距离矩阵、阈值筛选构造树、树深度排序及叶/尾节点优先剔除的技术；整体实现无需额外训练，兼容任意现有 Video‑MLLM。

**📊 数据集**

在 NExT‑QA、MVBench、VideoMME、MLVU、LongVideoBench 五大公开视频理解基准上进行了评估。

**📈 对比分析**

与 FastV、VisionZip、G‑Prune（图像级）、STTM、FrameFusion（视频级）等方法对比，ForestPrune 在 90% 采样率下仍保持 94.6%–95.8% 的平均准确率，并在 LLaVA‑Video 上比 FrameFusion 快 81.4% 的剪枝时间、显著降低 GPU 内存使用，同时提升模型在 VideoMME 与 MLVU 的性能至 SOTA 水平。

**⚠️ 局限性**

局限性包括：对阈值 τ_s、τ_p 的敏感度仍需进一步稳健性验证；树构造与合并在极长视频或高帧率场景下的计算开销和内存占用仍有提升空间；目前仅在公开基准上验证，实际多模态应用的通用性尚待更多实验。

---

## 296. Separating Diagnosis from Control: Auditable Policy Adaptation in Agent-Based Simulations with LLM-Based Diagnostics

**arXiv ID:** 2603.22904 | [PDF](https://arxiv.org/pdf/2603.22904v1)

**作者:** Shaoxin Zhong `[一作]` (University of Auckland), Michael Witbrock `[通讯]` (University of Auckland)

**通讯引用:** 3535 | [OpenAlex ID](https://openalex.org/A5057995059)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个三层框架，将大语言模型（LLM）仅用作诊断工具，控制决策由可追溯的确定性公式完成，从而实现了可审计的自适应政策调整，应用于老年人孤独度仿真。

**💡 创新点**

创新点在于将LLM的语义推理与决策控制完全分离，采用结构化JSON诊断输出与阈值驱动的增量更新公式；通过系统消融验证，该设计显著优于固定策略和黑盒LLM控制。

**🔧 技术方法**

技术包括：Agent‑Based Model (基于社交网络与孤独度动力学)、LLM诊断（Ollama llama3:8b、低温度、结构化提示）、Python+NumPy+NetworkX的仿真实现、确定性阈值更新规则与参数裁剪、对照实验与统计显著性检验。

**📊 数据集**

数据集：完全基于仿真产生的数据，30名代理，200天演化；通过多次随机种子（42、100、200、300、400、500、600）构建训练与holdout集；未使用真实养老机构数据。

**📈 对比分析**

比较方法：设置五种实验条件（Baseline、Fixed Policy、LLM Mapping、Closed‑loop（本框架）、Black‑box LLM），在四个holdout种子上计算最终平均孤独度；Closed‑loop平均0.607，较Baseline低15.3%，较Black‑box低11.7%，且表现出较小方差，显示显著性能提升。

**⚠️ 局限性**

局限性：1) 规模小（30名代理）；2) 未与真实养老机构数据验证；3) 仅评估单一LLM模型与提示策略；4) 更新规则手工设定，缺乏数据驱动学习；5) 未直接优化资源成本与预算约束。

---

## 297. Template-Based Feature Aggregation Network for Industrial Anomaly Detection

**arXiv ID:** 2603.22874 | [PDF](https://arxiv.org/pdf/2603.22874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 298. Who Sits Where? Automated Detection of Director Interlocks in Indian Companies

**arXiv ID:** 2603.22860 | [PDF](https://arxiv.org/pdf/2603.22860v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 299. Task-Aware Positioning for Improvisational Tasks in Mobile Construction Robots via an AI Agent with Multi-LMM Modules

**arXiv ID:** 2603.22903 | [PDF](https://arxiv.org/pdf/2603.22903v1)

**作者:** Seongju Jang `[一作]` (University of Michigan), SangHyun Lee `[通讯]` (University of Michigan)

**通讯引用:** 10539 | [OpenAlex ID](https://openalex.org/A5100444546)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于多模态大模型（LMM）的移动施工机器人智能体，能够在未预先定义任务位置、含属性/情境条件以及随时调整的即兴任务下，理解自然语言指令并自主定位。

**💡 创新点**

创新点在于将三大LMM模块（任务拆解核心、区域导航模块、精细定位模块）并行集成，并通过构造构图信息与视觉输入共同推理，实现对非预定义任务目标与属性/情境的实时识别与定位。

**🔧 技术方法**

使用技术包括：GPT‑4o 作为 LMM、ROS2+Navigation2+slam‑toolbox 实现机器人运动、YOLO+EasyOCR 识别施工图文字、gRPC/Docker 容器化并行部署、OpenCV 进行图像分割提示、LiDAR 前方距离感知及安全回退逻辑。

**📊 数据集**

使用的数据集为室内施工模拟环境的施工平面图（PNG）、对应的 ROS 二值地图以及机器人实时摄像头图像；实验场景为一座装饰施工现场的室内实验室。

**📈 对比分析**

在与基线开源目标检测模型 Grounding DINO 与 YOLOv8x‑world 的对比实验中，所提方法在属性/情境条件下的目标识别成功率为 86.7%，显著高于基线的 26.7%–33.3%；整体任务成功率 92.2%，会话成功率 82.2%，单次会话任务成功率可达 93.9%。

**⚠️ 局限性**

主要局限包括：定位模块对视觉推理的依赖导致 7 次失败，LMM 在图像空间关系推理和距离判断上仍不够精确；整体推理延迟受 API 调用影响，实时性有待提升；实验仅在模拟室内环境中验证，未在真实施工现场进行鲁棒性测试。

---

## 300. From Morality Installation in LLMs to LLMs in Morality-as-a-System

**arXiv ID:** 2603.22944 | [PDF](https://arxiv.org/pdf/2603.22944v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 301. Balancing Safety and Efficiency in Aircraft Health Diagnosis: A Task Decomposition Framework with Heterogeneous Long-Micro Scale Cascading and Knowledge Distillation-based Interpretability

**arXiv ID:** 2603.22885 | [PDF](https://arxiv.org/pdf/2603.22885v1)

**作者:** Xinhang Chen `[一作]` (Beihang University), Suili Yang `[通讯]` (Loongair Aviation Maintenance Engineering Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了诊断分解框架（DDF）和其实现 LMsD，通过显式把整体机体诊断拆分为异常检测（AD）和故障分类（FC）两子任务，解决数据不确定性、任务异质性和计算低效问题。

**💡 创新点**

创新点在于：①显式任务解耦与两阶段长-微尺度诊断器，解决全局上下文与局部特征的 receptive‑field 对立；②通过解耦训练与硬阈值路由降低训练成本；③Keyness Extraction Layer（KEL）结合知识蒸馏实现可解释性-by-design；④在飞行参数上实现物理可追溯的两阶段决策。

**🔧 技术方法**

使用的技术包括：卷积分词+多头自注意力（ConvTokMHSA）用于 AD；多尺度微卷积网络（MMK Net）用于 FC；硬阈值路由与解耦训练；知识蒸馏+KEL 关键性向量提取；PCA 低阶特征稀疏分析；多阶段评估指标（ACC、F1、WF1、MCWPM）。

**📊 数据集**

采用真实通用航空 Cessna 172 数据集 NGAFID，23 维传感器，28,935 次飞行，36 类维护事件；包含 19 类子集和整体 36 类，体现严重不平衡和真实噪声。

**📈 对比分析**

与 Bi‑LSTM、InceptionTime、InceptionTimeAttn、ConvTokMHSA/SWLA 等基线相比，LMSD 在诊断任务上 Multi‑Class Weighted Penalty Metric 提升 4–8%，训练时间约 50%–70% 下降，模型尺寸约一半，推理时间保持在可接受范围内，且在异常检测误判率方面更符合航空安全要求。

**⚠️ 局限性**

限制：1）标签噪声与维护记录的时序误标导致误判率上升；2）极端长尾分布和样本稀缺限制了尾类召回；3）传感器维度有限、特征稀疏导致辨别上限受限；4）当前方法无法突破数据本身的辨别瓶颈，需要更高质量的数据与多层次物理约束。

---

## 302. Designing to Forget: Deep Semi-parametric Models for Unlearning

**arXiv ID:** 2603.22870 | [PDF](https://arxiv.org/pdf/2603.22870v1)

**作者:** Amber Yijia Zheng `[一作]` (Purdue University), Raymond A. Yeh `[通讯]` (Purdue University)

**通讯引用:** 3372 | [OpenAlex ID](https://openalex.org/A5076130922)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一类深度半参数模型（SPM），通过在推理时删除特定训练样本实现高效机器忘记，且不需要重新训练。

**💡 创新点**

创新点在于将模型架构设计为可显式依赖训练集，从而实现“测试时删除”忘记，而非仅靠后置算法；同时兼顾了参数化模型的性能与非参数化模型的可忘记性。

**🔧 技术方法**

技术包括：半参数网络结构（融合模块g、非参数模块h、参数模块f）、集成检索/聚类减少训练集规模、标签置换增强、用于分类的ResNet18、用于生成的DDPM U-Net等。

**📊 数据集**

使用的数据集有CIFAR‑10、ImageNet‑1K（分类）和CIFAR‑10（条件图像生成），以及相应的基准模型（ResNet18、KNN、GMM、DDPM）。

**📈 对比分析**

与传统参数化模型（ResNet、DDPM）以及多种近似忘记算法（GA、FT、IU、BE、BS、ℓ1‑sparsity、SalUn、MUNBa）和非参数基准（KNN、GMM）比较。SPM在分类任务上保持与参数化模型相近的准确率，且在忘记时的软预测差距、ΔUA、ΔRA、ΔTA均最小；在生成任务中Fid_O最小，ΔUA、ΔFID_R也低，且忘记时间几乎为零（<1s），远快于基线方法。

**⚠️ 局限性**

局限性包括：仅在图像分类/生成任务上验证，缺乏跨任务/大规模数据集的通用性；模型对训练集规模敏感，需要检索/聚类预处理；并非所有SPM结构都能通过测试时删除完全实现忘记，需在设计上保证显式依赖。

---

## 303. Secure Two-Party Matrix Multiplication from Lattices and Its Application to Encrypted Control

**arXiv ID:** 2603.22857 | [PDF](https://arxiv.org/pdf/2603.22857v1)

**作者:** Kaoru Teranishi `[一作]` `[通讯]` (University of Osaka), Kaoru Teranishi (University of Osaka)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种单轮两方计算协议，用于近似固定点矩阵乘法，并将其应用于加密线性控制器；

**💡 创新点**

在两方计算中实现了仅需单轮通信且客户端在线复杂度低于原始控制器计算的安全加密控制方案；

**🔧 技术方法**

基于LWE（学习带误差）与SIS（短整数解）问题的加密与承诺方案，并结合加法秘密共享实现近似乘法；

**📊 数据集**

未使用公开数据集，实验采用自定义线性时不变系统与反馈增益进行仿真验证；

**📈 对比分析**

通过理论证明和实验，证明客户端在线复杂度降至O(n_x+n_u)而非O(n_u n_x)，且控制误差始终低于设定阈值（如2^-10），与传统加密控制相比在通信轮数和性能上均优；

**⚠️ 局限性**

仅支持线性控制、固定点近似；对动态控制器、精度调节及多方扩展仍需进一步研究；

---

## 304. FixationFormer: Direct Utilization of Expert Gaze Trajectories for Chest X-Ray Classification

**arXiv ID:** 2603.22939 | [PDF](https://arxiv.org/pdf/2603.22939v1)

**作者:** Daniel Beckmann `[一作]` (University of Münster), Benjamin Risse `[通讯]` (University of Münster)

**通讯引用:** 2064 | [OpenAlex ID](https://openalex.org/A5003115770)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了FixationFormer，一种利用Transformer直接将放射科专家眼动轨迹（注视序列）融入胸部X光影像分类的架构。

**💡 创新点**

创新点在于：①将注视轨迹编码为时空token序列，保留原始的时间与空间信息；②设计了Image‑to‑Gaze及双向交叉注意力两种融合机制，实现图像与眼动特征的深度交互；③利用Nested Tensor处理变长注视序列，提升计算效率；④通过对比实验展示了该方法在多个基准数据集上的显著优势。

**🔧 技术方法**

主要技术包括Vision Transformer（ViT/B‑32）骨干网络、MGCA预训练、注视token化（时空线性映射+位置编码）、交叉注意力层、LoRA微调、Nested Tensor批处理，以及Grad‑CAM可视化。

**📊 数据集**

使用了三大公开胸片数据集：CXR‑Gaze（CHF、肺炎、正常），SIIM‑ACR（气胸二分类），Reflacx（多标签层次分类），均配备了专家注视轨迹。

**📈 对比分析**

与GazeGNN、EG‑ViT等基线方法对比，FixationFormer在CXR‑Gaze上取得最高准确率；在SIIM‑ACR上与EG‑ViT相当，AUC更高；在Reflacx上同样优于GazeGNN。总体上Cross‑Attention变体在准确率、稳定性和训练效果上优于Two‑Way。

**⚠️ 局限性**

局限性包括：①双向交叉注意力在部分任务上表现更不稳定、方差更大；②在弱化的ImageNet ViT骨干下，Two‑Way优势不明显，说明对特定预训练的依赖；③仅评估胸部X光，缺乏多模态验证；④由于Nested Tensor不支持直接提取注意力权重，无法深入分析交叉注意力的内部机制；⑤数据集规模有限，可能导致模型对特定注视模式过拟合。

---

## 305. SLARM: Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes

**arXiv ID:** 2603.22893 | [PDF](https://arxiv.org/pdf/2603.22893v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. ProGRank: Probe-Gradient Reranking to Defend Dense-Retriever RAG from Corpus Poisoning

**arXiv ID:** 2603.22934 | [PDF](https://arxiv.org/pdf/2603.22934v1)

**作者:** Xiangyu Yin `[一作]` (Chalmers University of Technology), Chih-hong Cheng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ProGRank，一种后处理、无训练的检索端防御机制，利用对查询-段落对进行轻量级随机扰动后计算探测梯度，提取代表性一致性和分散风险两种不稳定性信号，结合 score-gate 进行重新排序，从而降低语料库中毒段落在 Top‑K 结果中的曝光。

**💡 创新点**

创新点包括：①仅利用检索器内部梯度信息，无需额外模型或训练；②首次将梯度不稳定性度量与门控策略结合用于 RAG 防御；③支持白盒和代理版本，兼顾部署灵活性；④在保持段落内容不变的前提下实现防御。

**🔧 技术方法**

核心技术：随机扰动（token dropout、encoder dropout 及其混合）；探测梯度、代表性一致性、分散风险两种不稳定性度量；score‑gated penalty fusion；轻量级后处理重排算法；与基线对比实验。

**📊 数据集**

使用数据集：MS MARCO、Natural Questions（NQ）和 HotpotQA；使用三种密集检索器（Contriever、DPR、BGE）；对抗方法包括 PoisonedRAG、LIAR‑RAG、Joint‑GCG。

**📈 对比分析**

比较方法：检索阶段评估 Poison Hit Rate 和 Poison Recall Rate；端到端生成评估 substring‑based ASR、judge‑based ASR 和 ACC；与 GRADA、GMTP、RAGuard 等现有防御方法对比。结果显示，ProGRank 在所有数据集上获得最低宏平均 judge‑based ASR，substring‑based ASR 也显著低于基线；保持或略优于基线的清洁任务效能（EM、F1、ROUGE‑L），并且比 RAGuard 延迟约 20× 更快。

**⚠️ 局限性**

局限性：对极其精细的自适应攻击仍存在一定风险；随机扰动参数需要调优；依赖检索器的白盒访问（代理版性能受近似影响）；对小 K 或极度稀疏检索结果的敏感度可能不足；不同检索器架构差异对效果有一定影响。

---

## 307. Ran Score: a LLM-based Evaluation Score for Radiology Report Generation

**arXiv ID:** 2603.22935 | [PDF](https://arxiv.org/pdf/2603.22935v1)

**作者:** Ran Zhang `[一作]` (Beijing Institute of Technology), Hongliang Sun `[通讯]` (China-Japan Friendship Hospital)

**通讯引用:** 2069 | [OpenAlex ID](https://openalex.org/A5108697214)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个临床专家引导的Human–LLM协作框架，用于从胸部X光报告中抽取21个标准化多标签异常，并提出Ran Score作为基于发现级别的报告评估指标。

**💡 创新点**

创新点在于利用医生反馈迭代优化提示词而非模型微调，实现对低频异常和临床语义（如否定、歧义）的高准确提取，并将此框架应用于全MIMIC‑CXR报告的公开标注和生成报告评估。

**🔧 技术方法**

主要技术包括大语言模型（如Qwen3‑14B、GPT‑4o‑mini等）的提示工程、结构化模板与少量示例（few‑shot）结合的提取器，以及基于提取结果的宏观F1（Ran Score）评估。

**📊 数据集**

使用了MIMIC‑CXR‑EN（3,000/300/3,000报告）和独立的ChestX‑CN（150报告）数据集，并在这两个数据集上对模型进行训练、优化和评估。

**📈 对比分析**

通过与六位放射科医生建立的参考标准和CheXbert基准对比，优化后的Qwen3‑14B在开发集上宏观F1从0.753提升至0.956，低频异常的召回率显著提升，Ran Score在多生成模型中对比显示LLM‑RG4最优，整体性能显著优于现有自动评估方法。

**⚠️ 局限性**

局限包括：仅限胸部X光报告、开发集与评估集重叠可能导致过拟合、忽略不确定/模糊诊断、Ran Score受提取器偏差影响、需要临床专家反复反馈且可扩展性受限。

---

## 308. EVA: Efficient Reinforcement Learning for End-to-End Video Agent

**arXiv ID:** 2603.22918 | [PDF](https://arxiv.org/pdf/2603.22918v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 309. EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction

**arXiv ID:** 2603.22910 | [PDF](https://arxiv.org/pdf/2603.22910v1)

**作者:** Yixuan Wang `[一作]` (Harbin Institute of Technology), Wanxiang Che `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 8820 | [OpenAlex ID](https://openalex.org/A5019108029)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EchoKV，能够在标准推理与压缩推理之间随需切换的 KV 缓存压缩方案。

**💡 创新点**

创新点在于：① 用轻量线性网络直接从部分 KV 预测其余 KV，避免传统压缩/解压导致的不可逆参数变换；② 采用两阶段训练（重建 MSE + O-MSE 注意力），兼容 FlashAttention；③ 结合低秩方法形成 Hybrid，针对键和值分别采用不同压缩策略。

**🔧 技术方法**

使用技术包括：轻量线性网络、两阶段训练（MSE + O-MSE）、SVD 低秩共享、FlashAttention、KV 量化、KV eviction、混合压缩策略。

**📊 数据集**

使用数据集：LongBench、RULER、LongAlpaca、Alpaca、ShareGPT、C4 等多种长上下文和短文本数据。

**📈 对比分析**

与 Palu、CommonKV、ThinK、MiniCache 等现有 SOTA 进行基准对比；在 0.5、0.3 压缩比下 EchoKV 几乎无损，整体性能优于所有基线；Hybrid 进一步提升关键/值分离压缩效果。

**⚠️ 局限性**

局限性：局部头部选择使用启发式方法，缺乏细粒度自适应；Hybrid 方案仍为粗粒度处理，未来需更精细的键值分离策略；对不同模型的迁移性和最优超参数仍需进一步验证。

---

## 310. IntentWeave: A Progressive Entry Ladder for Multi-Surface Browser Agents in Cloud Portals

**arXiv ID:** 2603.22917 | [PDF](https://arxiv.org/pdf/2603.22917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 311. VLGOR: Visual-Language Knowledge Guided Offline Reinforcement Learning for Generalizable Agents

**arXiv ID:** 2603.22892 | [PDF](https://arxiv.org/pdf/2603.22892v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 312. TastePrint: A 3D Food Printing System for Layer-wise Taste Distribution via Airbrushed Liquid Seasoning

**arXiv ID:** 2603.22887 | [PDF](https://arxiv.org/pdf/2603.22887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 313. Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics

**arXiv ID:** 2603.22886 | [PDF](https://arxiv.org/pdf/2603.22886v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 314. Group Editing : Edit Multiple Images in One Go

**arXiv ID:** 2603.22883 | [PDF](https://arxiv.org/pdf/2603.22883v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 315. Continuous Optimization for Satisfiability Modulo Theories on Linear Real Arithmetic

**arXiv ID:** 2603.22877 | [PDF](https://arxiv.org/pdf/2603.22877v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 316. Quality Over Clicks: Intrinsic Quality-Driven Iterative Reinforcement Learning for Cold-Start E-Commerce Query Suggestion

**arXiv ID:** 2603.22922 | [PDF](https://arxiv.org/pdf/2603.22922v1)

**作者:** Qi Sun `[一作]` (Alibaba International Digital Commercial Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commercial Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出冷启动电商查询建议的迭代强化学习框架，并通过答案可答性、事实性、信息增益等内部质量奖励持续优化查询建议。

**💡 创新点**

创新点包括：①在无点击数据情况下利用不确定性采样挑选难例进行训练；②采用内部质量奖励替代传统CTR模型；③在多阶段训练中循环强化学习，显著提升模型鲁棒性；④验证了离线指标与在线表现的高度相关性。

**🔧 技术方法**

使用技术包括：RL（GRPO）、大语言模型（Qwen-30B-A3B）做奖励评估、SFT+RL迭代训练、不确定性采样策略、内部质量评估指标、基于LLM的prompt工程。

**📊 数据集**

数据集：EQS-Benchmark（16949个真实用户查询，包含5,914个点击正样本与10,535个点击负样本），以及部分在线交互日志用于warm‑up和不确定性采样。

**📈 对比分析**

与多种开源/闭源模型（Qwen、Gemini、GPT-4o-mini等）在RT限制下对比，最终模型在离线严格准确率达到86.1%，在在线实验中ΔChatUV提升6.81%，并展示离线指标与在线表现的正相关。

**⚠️ 局限性**

局限性：奖励评估依赖外部LLM，可能导致推理延迟或长尾领域覆盖不足；不确定性采样仍需一定的在线交互数据；对动态或稀有场景的适应性待进一步验证。

---

## 317. When AVSR Meets Video Conferencing: Dataset, Degradation, and the Hidden Mechanism Behind Performance Collapse

**arXiv ID:** 2603.22915 | [PDF](https://arxiv.org/pdf/2603.22915v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 318. Confidence Calibration under Ambiguous Ground Truth

**arXiv ID:** 2603.22879 | [PDF](https://arxiv.org/pdf/2603.22879v1)

**作者:** Linwei Tao `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 21955 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了在多注释者存在标签不确定性情况下的置信度校准问题，并提出了基于真实注释分布的后置校准方法。

**💡 创新点**

发现传统基于多数投票标签的温度缩放在标签存在歧义时系统性失效，提出了多种基于完整或单个注释分布的温度缩放以及无注释数据的标签平滑温度缩放。

**🔧 技术方法**

利用温度缩放、Dirichlet软目标、Monte Carlo 采样、SoftPlatt、向量温度、分位数正则化等技术，并采用严格的分数与误差评估。

**📊 数据集**

使用了 CIFAR-10H、ChaosNLI、ISIC 2019 与 DermaMNIST 四个多注释数据集，涵盖图像分类、自然语言推理和医学影像。

**📈 对比分析**

与传统 TS、Platt、Dirichlet‑Hard 等基线对比实验，发现基于完整注释分布的 Dirichlet‑Soft 将真实标签 ECE 降低 55–87%，Monte Carlo 仅需单个注释即可匹配完整分布，LS‑TS 在无注释时亦能显著降低 ECE，整体显著优于现有基线。

**⚠️ 局限性**

需要在校准时获得多注释者分布；在仅有投票标签时 LS‑TS 仍不能完全匹配完整分布，且医学数据采用合成注释，真实实例级分布尚未得到验证。

---

## 319. Chain-of-Authorization: Internalizing Authorization into Large Language Models via Reasoning Trajectories

**arXiv ID:** 2603.22869 | [PDF](https://arxiv.org/pdf/2603.22869v1)

**作者:** Yang Li `[一作]` (Tsinghua University), Ke Xu `[通讯]` (Tsinghua University)

**通讯引用:** 11905 | [OpenAlex ID](https://openalex.org/A5100665814)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Chain-of-Authorization（CoA）框架，将授权逻辑内化到大语言模型的推理过程，形成在生成回答前必需的授权推理轨迹

**💡 创新点**

通过在输入中注入权限标签、在输出中强制生成资源审核、身份解析和决策三阶段的授权链，实现授权决策与生成行为的因果耦合，解决传统外部授权或软约束导致的认知混乱

**🔧 技术方法**

结构化输入重构、推理轨迹生成、监督微调（LoRA）以及统一序列预测损失；使用权限标签扩展词表以保证语义隔离

**📊 数据集**

WMDP、MMLU、SQuAD、COVID‑QA、Mobile‑Actions 等公开数据集，覆盖内部知识、外部检索、工具调用三类授权场景

**📈 对比分析**

与基线（Base、SFT、PermissionLM、SudoLM、External Gateway）对比，CoA在授权匹配场景下保持与SFT相近的准确率，同时在未授权、权限不匹配场景下拒绝率接近100%，在多模型、多攻击方式的鲁棒性测试中表现出色

**⚠️ 局限性**

对学习率、模型规模、数据量等超参的敏感性仍需进一步探究；当权限标签不完整或极其细粒度时模型表现不稳定；对极端复杂多级权限结构的可扩展性尚未充分验证

---

## 320. Optimizing Small Language Models for NL2SQL via Chain-of-Thought Fine-Tuning

**arXiv ID:** 2603.22942 | [PDF](https://arxiv.org/pdf/2603.22942v1)

**作者:** Anshul Solanki `[一作]` (Google AI, Global Services Delivery), Navneet Kamboj `[通讯]` (Google AI, Global Services Delivery)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大模型（Gemini 2.5）和小模型（Qwen）进行标准与链式推理（CoT）微调，探究其对 NL2SQL 任务的影响。

**💡 创新点**

发现大模型微调反而导致性能下降，提出通过 CoT 训练小模型弥合性能差距，并引入复杂度过滤的训练集。

**🔧 技术方法**

使用 LoRA 微调、执行精度评估、Prompt 工程、链式推理数据集、SQL 结构复杂度打分等技术。

**📊 数据集**

采用 Spider 1.0 数据集及其自定义的高难度查询子集，并在 600 条查询的基准上进行评测。

**📈 对比分析**

通过 Execution Accuracy 在 600 查询基准上比较，结果显示大模型微调无显著提升甚至退步；小模型 Qwen 通过 CoT 微调从 36% 提升到 54.5%，接近大型模型水平。

**⚠️ 局限性**

仅评估了 Spider dev split，未涵盖真实企业数据库的杂乱与歧义；CoT 产生的 token 过长导致推理延迟；未实现内部隐空间推理。

---

## 321. Gabow's $O(\sqrt{n}m)$ Maximum Cardinality Matching Algorithm, Revisited

**arXiv ID:** 2603.22909 | [PDF](https://arxiv.org/pdf/2603.22909v1)

**作者:** Kurt Mehlhorn `[一作]` (Max Planck Institute for Informatics), Romina Nobahari `[通讯]` (Sharif University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对 Gabow 的 O(√n·m) 最大匹配算法进行重述，并提出一种更直接、易于教学的“寻找最短增广路径”新算法；随后实现并在实验中验证该改进的有效性。

**💡 创新点**

核心创新在于重新设计了原算法的第一部分（计算最短增广路径）。新方法摒弃了对 Edmonds 对偶算法的间接引用，直接利用搜索树的层级信息实现 BFS‑式增长和桥步，从而简化了实现与教学；同时保留了原算法的整体结构和时间复杂度。

**🔧 技术方法**

主要技术包括：
- 交替树（alternating tree）与花（blossom）概念的改进表示；
- 基于层级（canonical path length）进行的增长步骤与桥步调度；
- 优先队列（bucket）实现事件的阶段化处理；
- 并查集（union‑find）用于在非突破阶段维护最大花；
- 辅助图 H 的构造与在 H 上寻找最大一组相互不相交的增广路径，再提升回原图；
- 证书（optimality witness）构造与实现。

**📊 数据集**

实验使用三类数据集：
- 随机图（m/n ≥ 400 的稠密随机图）；
- 只包含短链的构造图（O(n) 条长为 7 的链连接到一个中心点的完全图）；
- 既含短链又含长链的构造图（每条长链长度为 2i+1，i=4…√n）。

**📈 对比分析**

与 Gabow 原实现（以及其在 LEDA 中的实现）进行对比。实验结果表明：
- 在短链+长链图上，迭代次数为 O(√n)，指令计数按 n^{3/2} 规模增长，表现与 Gabow 算法相当；
- 在仅短链图上，迭代次数为常数，指令计数线性增长，性能与 Gabow 近似；
- 在随机图上，迭代次数随 n 非线性增长，但两种实现的运行时间差距不大。整体而言，改进实现与原版 Gabow 算法在速度与内存上相当，且新算法实现更简洁。

**⚠️ 局限性**

限制与未解决的问题：
- 仅对算法的第一部分做了改动，整体时间复杂度仍为 O(√n·m)，未突破该上界；
- 对非突破阶段的优先级队列实现采用简单桶结构，虽足以证明正确性，但在极端图上可能不是最优；
- 代码未在极大规模稀疏图（m ≈ n）或特殊结构图上做更细致的性能调优；
- 对图的连通性、权重（仅匹配无权）等特殊情况没有单独讨论。

---

## 322. Dual-Teacher Distillation with Subnetwork Rectification for Black-Box Domain Adaptation

**arXiv ID:** 2603.22908 | [PDF](https://arxiv.org/pdf/2603.22908v1)

**作者:** Zhe Zhang `[一作]`, Shengyong Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了双教师蒸馏+子网络修正（DDSR）框架，用黑盒源模型与CLIP两源信息自适应融合生成伪标签，辅以子网络正则化、在线伪标签/提示微调与类原型自训练，实现黑盒域适应；

**💡 创新点**

创新点包括①基于目标域规模与预测不确定性动态加权源模型与CLIP的自适应融合；②通过子网络输出与梯度差异正则化抑制伪标签噪声导致的过拟合；③在线更新伪标签及CLIP提示实现自蒸馏，提升语义一致性；

**🔧 技术方法**

技术手段涵盖知识蒸馏(KL)、Mixup一致性与信息最大化、子网络输出/梯度差异正则化、EMA自蒸馏与提示微调、类原型自训练，并采用CLIP ViT-B/32作为语义教师；

**📊 数据集**

实验使用Office‑31、Office‑Home和VisDA‑17三个公开基准数据集；

**📈 对比分析**

与UDA、SFDA和多种BBDA方法对比，DDSR在所有数据集上均达或逼近SOTA，Office‑31平均精度93.1%、Office‑Home 83.2%、VisDA‑17 90.6%，显著优于现有BBDA方法；

**⚠️ 局限性**

局限性包括：①未处理源/目标类别不匹配问题；②依赖CLIP预训练模型，需额外计算与存储；③对目标域大小的阈值设置敏感，需经验调参。

---

## 323. Agile-VLA: Few-Shot Industrial Pose Rectification via Implicit Affordance Anchoring

**arXiv ID:** 2603.22899 | [PDF](https://arxiv.org/pdf/2603.22899v1)

**作者:** Teng Yan `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Bingzhuo Zhong `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Agile-VLA框架，在资源受限的边缘设备上实现工业姿态重新定向；

**💡 创新点**

核心创新为隐式可供性锚定机制将几何关键点映射为结构化动作原语，以及异步双流架构将感知频率10 Hz与控制频率50 Hz解耦；

**🔧 技术方法**

采用量化TensorRT加速的视觉模型、立方曲线插值控制器、外部支撑面（桌面）作为外部取向轴，以及5-shot少样本在线微调；

**📊 数据集**

使用自建的DID‑127工业零件数据集（127个不同几何形状的部件）；

**📈 对比分析**

与Transporter、RT‑1、RVT、Octo‑Base、OpenVLA等开源基线对比，Agile‑VLA在成功率、碰撞率和运动尖峰（jerk）方面均显著优于基线（成功率90.5%，碰撞率3.2%，尖峰1.24 m/s³）；

**⚠️ 局限性**

局限性包括仍需手动标注5张示例图像，无法直接处理极端动态碰撞或需要高频触觉反馈的任务；

---

## 324. Grounding Sim-to-Real Generalization in Dexterous Manipulation: An Empirical Study with Vision-Language-Action Models

**arXiv ID:** 2603.22876 | [PDF](https://arxiv.org/pdf/2603.22876v1)

**作者:** Ruixing Jin `[一作]` (Chinese University of Hong Kong), Guiliang Liu `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在仿真与真实世界中对Vision‑Language‑Action（VLA）模型进行大规模零样本迁移实验，系统评估了领域随机化、渲染真实性、物理真实性和强化学习微调等多种Sim2Real技术对抓取、堆叠等双臂操纵任务的影响；同时发布了统一评测协议和在线平台，方便他人复现与比较。

**💡 创新点**

创新点在于：①对VLA模型进行因素化的Sim2Real实验，首次揭示空间随机化对零样本迁移的主导作用；②比较帧级与周期级随机化的差异，证明帧级随机化更有效；③证明渲染与物理真实性的提升对迁移有显著但递减的作用；④展示强化学习微调在无真实数据情况下显著提升鲁棒性；⑤公开了标准化评测协议与实物机器人平台，推动领域复现。

**🔧 技术方法**

采用OpenVLA‑OFT VLA框架，使用监督式行为克隆（SFT）和基于GRPO的强化学习微调；实施多维度领域随机化（背景、光照、障碍、相机、桌面高度等）及渲染真实性（Low/Medium/High光线追踪）和物理真实性调整；在仿真环境RobotWin 2.0训练，真实测试使用Cobot Magic双臂和RealSense D435摄像头。

**📊 数据集**

数据集包括：①1,000条（10条/任务×100次）仿真演示；②10,000+条真实世界测试记录，覆盖背景、光照、物体实例、干扰物和空间位置等多种扰动；任务涵盖“点击铃铛”“放空杯”“打块锤”“堆叠碗”“双瓶抓取”等。

**📈 对比分析**

对比方法：在Sim‑OOD环境与真实世界分别记录成功率；比较不同随机化因子、随机化粒度、渲染/物理真实性以及RL微调的组合。结果显示：①单独空间随机化可将真实成功率从≈5%提升至≈35%；②三因子组合可达≈50%；③帧级随机化相对周期级提升≈4–8%；④高渲染层级与物理真实性提升可分别带来≈10–15%和≈5–10%的增益；⑤RL微调后可从≈5%提升至≈33%，再加DR可升至≈50%+；所有方法叠加后，某些任务真实成功率超过60%，Sim‑OOD约70%。

**⚠️ 局限性**

局限性包括：①仅在零样本迁移下实验，未验证在少量真实数据微调下的进一步提升；②实验任务与平台有限，结果对不同机器人型号、不同操作环境的泛化尚未充分验证；③大量依赖仿真参数调节，可能导致对真实硬件的适配成本高；④实验多基于双臂机器人，单臂或其他机械臂的适用性需进一步研究。

---

## 325. A Feature Shuffling and Restoration Strategy for Universal Unsupervised Anomaly Detection

**arXiv ID:** 2603.22861 | [PDF](https://arxiv.org/pdf/2603.22861v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 326. Dynamical Systems Theory Behind a Hierarchical Reasoning Model

**arXiv ID:** 2603.22871 | [PDF](https://arxiv.org/pdf/2603.22871v1)

**作者:** Vasiliy A. Es'kin `[一作]` (University of Nizhny Novgorod), Mikhail E. Smorkalov `[通讯]` (Huawei Nizhny Novgorod Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出 Contraction Mapping Model (CMM)，将递归推理模型转化为连续神经微分方程（NODE/NSDE）并通过稳定平衡点约束实现高效、稳定的算法推理。

**💡 创新点**

核心创新包括：1) 强制收敛到稳定平衡点的 Routh‑Hurwitz 约束与球面排斥损失；2) 用多项式 StableMax3/5 取代 Softmax 提升数值稳定；3) 使用 AlgGradNorm 自动平衡多目标损失；4) 通过噪声注入的 NSDE 解决过拟合与轨道收敛问题；5) 在 0.26M 参数规模下保持与大模型相当的性能。

**🔧 技术方法**

使用技术：神经常微分方程 (NODE/NSDE)、Routh‑Hurwitz 稳定性判据、球面排斥损失、StableMax3/5 近似、AlgGradNorm 动态损失平衡、指数衰减学习率、EMA、梯度累积、混合精度、梯度检查点、权重共享、加噪声、稳定非线性激活（tanh）等。

**📊 数据集**

主要数据集：Sudoku‑Extreme、Maze；提及 ARC‑AGI 作为未来评测目标。

**📈 对比分析**

与 HRM、TRM、DeepSeek R1 等基线对比：在 5M 参数下 CMM 在 Sudoku‑Extreme 达到 93.7%（相较 27M HRM 的 55% 或 5M TRM 的 87.4%），在 0.26M 参数下仍保持 85%/82% 的 Sudoku/Maze 准确率；显示出极低参数下的优异性能。

**⚠️ 局限性**

局限性：1) Routh‑Hurwitz 约束仅保留第一项，完整判据计算量高；2) 对超大隐藏维度的可扩展性未充分验证；3) 噪声强度、损失权重等超参需经验调优；4) 训练仍受硬件精度与批量大小限制，AMP 与编译优化对结果影响显著。

---

## 327. The EU AI Act and the Rights-based Approach to Technological Governance

**arXiv ID:** 2603.22920 | [PDF](https://arxiv.org/pdf/2603.22920v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 328. Agent-Sentry: Bounding LLM Agents via Execution Provenance

**arXiv ID:** 2603.22868 | [PDF](https://arxiv.org/pdf/2603.22868v1)

**作者:** Rohan Sequeira `[一作]` (University of Southern California), Konstantinos Psounis `[通讯]` (University of Southern California)

**通讯引用:** 10067 | [OpenAlex ID](https://openalex.org/A5042745248)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过学习执行跟踪构建功能图，对LLM代理执行进行安全约束，阻止无关或攻击行为。

**💡 创新点**

将功能图与受限的意图对齐检查结合，首次实现基于执行血统的代理安全框架。

**🔧 技术方法**

采用功能图抽象、图匹配、LLM-as-Judge意图对齐、软硬件分离的执行监控技术。

**📊 数据集**

设计Agent-Sentry Bench（6,733 条执行跟踪，覆盖 Banking、Slack、Travel、Workspace 四个域）并使用 AgentDojo benchmark 进行评测。

**📈 对比分析**

与 CaMeL、Progent 等对比，攻击成功率降低至约 3.7%（或 5.41%），实用率保持 76% 以上，延迟仅 7.9 秒。

**⚠️ 局限性**

受限于功能图覆盖不足和意图歧义，仍可能漏判 mimicry 攻击，并需要持续更新图谱以应对新兴行为。

---

## 329. Aerial Agentic AI: Synergizing LLM and SLM for Low-Altitude Wireless Networks

**arXiv ID:** 2603.22866 | [PDF](https://arxiv.org/pdf/2603.22866v1)

**作者:** Li Dong `[一作]`, Ekram Hossain `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了Aerial Agentic AI框架，将低空网络中的UAV配备快速思考的SLM进行实时感知、记忆与决策，基站使用慢速思考的LLM进行全局推理、知识更新和工具编排，实现分层协同与自适应服务。

**💡 创新点**

创新点在于将SLM与LLM以层次化代理方式组合，构建了两级记忆体系与分层工具框架，且通过能力迁移与语义压缩实现了在计算、能耗与通信受限环境下的生成式边缘智能。

**🔧 技术方法**

采用TinyLlama等SLM、LLaMA2等LLM，结合知识蒸馏、权重量化（AWQ）、算子融合（OF）和闪电注意力（FA），以及多模态感知、短期/长期记忆、工具链调度与协同调度模块。

**📊 数据集**

使用自定义任务数据集，采集UAV和基站在真实任务和仿真环境中的行为轨迹、提示、工具调用与语义摘要，主要针对城市监测与数据回传场景。

**📈 对比分析**

通过与单独SLM决策、全局LLM决策进行对比，评估决策时延和轨迹长度；实验显示SLM-LLM协作方案在保持低时延的同时几乎达到全局最优轨迹，且在SLM优化组合（AWQ+OF+FA）下能耗与推理速度得到显著提升。

**⚠️ 局限性**

局限性包括缺乏统一的协同决策理论框架、短期与长期记忆的漂移与同步风险、工具检索的不完善导致决策误差，以及对链路质量高度敏感，且缺乏公开标准数据集支持进一步验证。

---

## 330. The Coordinate System Problem in Persistent Structural Memory for Neural Architectures

**arXiv ID:** 2603.22858 | [PDF](https://arxiv.org/pdf/2603.22858v1)

**作者:** Abhinaba Basu `[一作]` `[通讯]` (National Institute of Electronics and Information Technology), Abhinaba Basu (National Institute of Electronics and Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究并实现了双视图信息素路径网络（DPPN），通过五轮实验探究持久结构记忆的两项必要条件，并验证其在合成序列任务中的效果。

**💡 创新点**

创新点在于发现持久结构记忆需要稳定坐标系统与优雅的迁移机制；提出信息素路由与学习率调节两种迁移机制，并证明随机投影坐标能解决坐标不稳定问题。

**🔧 技术方法**

采用信息素统计、稀疏注意力、双视图软分组、随机傅里叶特征、Hungarian对齐、学习率调节、结构完成函数等技术。

**📊 数据集**

使用自定义的合成序列分类任务，构建三种结构家族（A、B、C）并设置源/目标任务，实验覆盖10个随机种子。

**📈 对比分析**

与Transformer和随机稀疏基线在同一参数规模下比较；DPPN在同一任务上优于基线0.01–0.03 AULC；跨任务迁移效果有限，路由偏置产生负迁移，而学习率调节实现无负迁移。

**⚠️ 局限性**

局限在于迁移受坐标不稳定与迁移机制不优雅的限制；信息素路由对源任务特定结构敏感；实验仅在合成任务中验证，未在真实任务或预训练编码器上进行测试。

---

## 331. TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration

**arXiv ID:** 2603.22882 | [PDF](https://arxiv.org/pdf/2603.22882v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 332. ForeSea: AI Forensic Search with Multi-modal Queries for Video Surveillance

**arXiv ID:** 2603.22872 | [PDF](https://arxiv.org/pdf/2603.22872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 333. GateSID: Adaptive Gating for Semantic-Collaborative Alignment in Cold-Start Recommendation

**arXiv ID:** 2603.22916 | [PDF](https://arxiv.org/pdf/2603.22916v1)

**作者:** Hai Zhu `[一作]` (Alibaba International Digital Commercial Group), Xiaoyi Zeng `[通讯]` (Alibaba International Digital Commercial Group)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GateSID 框架，通过门控机制自适应融合语义与协同信号，解决冷启动推荐中协同-语义权衡问题。

**💡 创新点**

创新点在于门控融合共享注意力（GFSA）动态调节协同与语义注意分布，以及门控对比对齐（GRCA）根据物品成熟度自适应强化或放松语义-行为对齐。

**🔧 技术方法**

采用残差量化 VAE 将多模态嵌入离散为层次 Semantic ID，使用门控网络、共享注意力、对比学习等技术。

**📊 数据集**

在阿里巴巴工业级数据集（约 10 亿条日志）上评估，区分新物品（≤20 天）与热门物品（>300 天）。

**📈 对比分析**

与 COINS、PCR-CA、URL4DR、SPM-SID、SaviorRec、QARM 等基线在 CTR/CTCVR AUC/GAUC 进行对比，GateSID 在所有指标上均优于基线，尤其在热门物品上提升约 0.4% CTCVR GAUC，在线 A/B 测试 GMV +2.6%，CTR +1.1%。

**⚠️ 局限性**

局限性包括门控权重设计仍需手工设定阈值，未深入研究对极端冷启动物品的表现，以及模型在不同业务场景的可迁移性需进一步验证。

---

## 334. DecompGrind: A Decomposition Framework for Robotic Grinding via Cutting-Surface Planning and Contact-Force Adaptation

**arXiv ID:** 2603.22859 | [PDF](https://arxiv.org/pdf/2603.22859v1)

**作者:** Shunsuke Araki `[一作]` (Nara Institute of Science and Technology), Takamitsu Masubara `[通讯]` (Nara Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出了 DecompGrind 框架，将机器人抛光任务拆分为全局切割面规划（GCSP）和局部接触力适配（LCFA），实现对不同形状和硬度工件的时间高效抛光；

**💡 创新点**

创新点在于将抛光过程分解为几何规划与局部学习两部分，显著降低学习数据需求并通过几何规划控制全局形状变化，利用模仿学习仅在局部接触力上进行自适应；

**🔧 技术方法**

采用了几何切割面模型实现 GCSP，利用双边控制（Bilateral Control）与模仿学习（BCIL）构建 LCFA 的接触力适配策略；

**📊 数据集**

使用自制的 3D 打印工件数据集，包含不同形状和30%、45%、60%密度的工件，用于演示、训练 LCFA 并在实验中评估性能；

**📈 对比分析**

与随机切割+混合控制、GCSP+混合控制、全 BCIL、全 BCIL-all 等基线方法对比，实验表明 DecompGrind 在形状/硬度多样工件上能在保持安全接触力的前提下，显著缩短加工时间并降低形状误差；

**⚠️ 局限性**

局限性包括只能沿直线轨迹进行抛光，难以处理曲面；对极端刃具或材料硬度变化范围有限；需要手动挑选演示工件以覆盖最小/最大接触面积和硬度。

---

## 335. Caption Generation for Dongba Paintings via Prompt Learning and Semantic Fusion

**arXiv ID:** 2603.22946 | [PDF](https://arxiv.org/pdf/2603.22946v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 336. PersonalQ: Select, Quantize, and Serve Personalized Diffusion Models for Efficient Inference

**arXiv ID:** 2603.22943 | [PDF](https://arxiv.org/pdf/2603.22943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 337. The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration

**arXiv ID:** 2603.22862 | [PDF](https://arxiv.org/pdf/2603.22862v1)

**作者:** Haoyuan Xu `[一作]`, Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 16595 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对多工具LLM代理的研究进行系统综述，提出统一的任务公式并将其分为单次调用与长程编排两大范式。

**💡 创新点**

创新点在于构建了六个互联维度（推理、训练、安全、效率、完整性与基准），并为每一维度梳理代表方法与挑战，形成对多工具代理的整体框架。

**🔧 技术方法**

采用的技术包括基于图的顶层规划（GAP、ToolNet）、分层与并行执行架构（MARS、HuggingGPT）、训练无关提示与检索（ToolLLM、AnyTool）、合成轨迹与SFT、强化学习策略（ToolRL、Agent‑R1）、安全事务管理（SagaLLM、Atomix）等。

**📊 数据集**

使用的数据集与基准涵盖：NESTFUL、ToolHop、ToolBench、TaskBench、ToolDecathlon、AgentLongBench、OSWorld、Mobile‑Bench 等多领域多工具链实验集。

**📈 对比分析**

比较方法主要是基于上述基准对不同模型/框架的成功率、调用次数、延迟与成本进行量化评估，实验显示即使是最先进的 GPT‑4 在 50+ 步长程任务中的成功率仍低于 40%，且成本与延迟显著偏高。

**⚠️ 局限性**

局限性包括：缺乏统一、可复现的真实环境评测；多工具交互的安全与可解释性仍未彻底解决；模型对工具集动态演化的适应性有限；评测数据集多聚焦于人工构造场景，真实场景的泛化性不足。

---

## 338. TRINE: A Token-Aware, Runtime-Adaptive FPGA Inference Engine for Multimodal AI

**arXiv ID:** 2603.22867 | [PDF](https://arxiv.org/pdf/2603.22867v1)

**作者:** Hyunwoo Oh `[一作]` (University of California, Irvine), Mohsen Imani `[通讯]` (University of California, Irvine)

**通讯引用:** 6810 | [OpenAlex ID](https://openalex.org/A5033221192)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款单比特流 FPGA 加速器与编译器，可在不重配置的情况下实现 Vision Transformer、CNN、GNN 和 NLP 的端到端推理，并支持动态 token 剪枝与依赖感知层调度。

**💡 创新点**

创新点在于：① 将多模态层统一为 DDMM/SDDMM/SpMM 并在同一 PE 数组上通过模式切换（Systolic、SIMD、RADT）实现多功能；② 在硬件层面实现宽度匹配的两阶段 in-stream top‑k 剪枝，避免昂贵的全局排序；③ 通过 DALO 在多 RPU 网格上重叠独立 kernel，保持高利用率；④ 通过单一比特流实现多模态多任务无重配置运行。

**🔧 技术方法**

使用技术包括：FPGA（Alveo U50、ZCU104）、Chisel RTL、mode‑switchable PE 引擎、宽度匹配 top‑k 选择器、int8 量化、依赖感知层调度、DALO、动态切换数据流模式。

**📊 数据集**

实验数据集涵盖：TinyCLIP（ViT+NLP）、MDETR（CNN+NLP）、MissionGNN（ViT+GNN、CNN+GNN），以及 ImageNet、RefCOCO、UCF‑Crime 等任务评估。

**📈 对比分析**

与 RTX 4090 和 Jetson Orin Nano 在 batch = 1、int8 推理下对比，Alveo U50 在 TinyCLIP A 模型上实现 22.57×、在 ZCU104 上实现 6.86× 的速度提升；ViT 剪枝可提升 7.8×；DALO 在多 RPU 上提升 79% 吞吐率；整体能耗保持在 20–21 W，<2.5% 的精度下降。

**⚠️ 局限性**

局限性：① 对极大 ViT/模型尺寸仍受硬件资源限制，可能无法完整加载；② 动态稀疏仅对 ViT/NLP 的 token 剪枝实现，对 CNN/GNN 的动态 sparsity 仍不完全；③ 需要在编译时预估 sparsity 并在运行时更新，复杂度高；④ 目前仅支持 int8 量化，未覆盖更低位宽或混合精度方案。

---

## 339. PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving

**arXiv ID:** 2603.23049 | [PDF](https://arxiv.org/pdf/2603.23049v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 340. Linear time single-source shortest path algorithms in Euclidean graph classes

**arXiv ID:** 2603.22948 | [PDF](https://arxiv.org/pdf/2603.22948v1)

**作者:** Joachim Gudmundsson `[一作]` (University of Sydney), Sampson Wong `[通讯]` (University of Copenhagen)

**通讯引用:** 36 | [OpenAlex ID](https://openalex.org/A5029179453)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套新的判据，证明满足这些判据的欧几里得图类（如τ‑lanky图、k‑ply邻域系统图、κ‑thick立方体邻域系统图）可在线性时间内完成单源最短路（SSSP）问题。

**💡 创新点**

创新点在于：①将平面图中依赖于“minor‑closed”性质的递归划分方法推广到欧几里得图类；②通过构造代表图（representative graph）来证明组合式收缩图仍保留可线性时间的子线性分离器；③提出一组新的四条判据（子图封闭、稀疏收缩图、闭曲面分离器、可线性时间分离器），从而实现多维欧几里得图类的线性 SSSP。

**🔧 技术方法**

主要技术包括：组合式边收缩（Contract）、代表图构造、基于随机球/球面/矩形的子线性分离器、递归划分（r‑division）、HKRS 边松弛算法、以及对厚度、稀疏度、退化度等图结构性质的精细分析。

**📊 数据集**

该工作主要为理论性研究，没有使用具体数据集；实验验证采用随机生成的欧几里得图（如随机球、立方体、双曲面等）来证明分离器的期望性质。

**📈 对比分析**

与之前的近线性或超线性算法（如 Frederickson、Eppstein‑Miller‑Teng 等）的比较表明，本文的算法在时间复杂度上实现了真正的 O(n)（线性）级别，且在三类欧几里得图类上给出了更精确的多项式前缀（如 O(d^29^dk^2n)），优于先前的 O(n√log n) 或 O(n^1.5) 级别。

**⚠️ 局限性**

局限性包括：①维度 d 必须固定；②算法涉及复杂的随机收缩与分离器构造，常数系数和实现难度较高；③在某些实际应用场景中，随机化步骤可能导致结果不稳定；④对“厚度”“退化度”等参数的上界依赖于理论上已知的最坏情况，实际性能可能受限。

---

## 341. Generative Event Pretraining with Foundation Model Alignment

**arXiv ID:** 2603.23032 | [PDF](https://arxiv.org/pdf/2603.23032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 342. Minibal: Balanced Game-Playing Without Opponent Modeling

**arXiv ID:** 2603.23059 | [PDF](https://arxiv.org/pdf/2603.23059v1)

**作者:** Quentin Cohen-Solal `[一作]` (Université Paris-Dauphine), Tristan Cazenave `[通讯]` (Université Paris-Dauphine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了两种基于Unbounded Minimax的算法，专门用于在二人零和完美信息棋类游戏中实现“平衡玩法”，即在不压倒对手也不主动认输的前提下与人类或AI对战；并通过实验验证其在七种棋类游戏中的有效性。

**💡 创新点**

创新点在于：①将“平衡玩法”概念量化为最小化终局得分绝对值或优先正得分再最小化负得分的两种策略；②基于Unbounded Minimax设计出两种新的搜索变体（Unbounded <balance‑near> 与 Unbounded <balance‑positive>），不依赖对手建模或在线学习；③系统化地评估这些算法在多种棋类与不同强度评估函数下的表现。

**🔧 技术方法**

技术手段主要包括：Unbounded Minimax 树搜索（带完成技术），自我对弈训练的深度残差网络评估函数，强化学习生成的高低级评估模型，Monte Carlo Tree Search 与 AlphaZero 框架的相关概念，以及对搜索策略的自定义（如最佳动作选取与状态解析）。

**📊 数据集**

数据集与实验设置：使用 40 个基于强化学习（Descent Minimax）训练的棋盘评估函数，其中 20 个为高水平（后期 checkpoint）作为对手，另外 20 个为低水平（早期 checkpoint）作为弱对手；七款棋类（国际跳棋、国际象棋、Lines of Action、Connect6、Outer‑Open‑Gomoku、象棋、Havannah）；每种高低级组合在 800 场对局（考虑双方先后手）进行评测。

**📈 对比分析**

比较方法：采用两项指标——二元收益（胜负的平均 +1/-1）和终局分数（评估函数的平均终局值），两者越接近 0 表示平衡度越高。实验结果显示，Unbounded <balance‑positive> 在所有游戏中平均二元收益和终局分数最接近零，几乎实现完美平衡；相较之下，Unbounded <balance‑near> 失误率过高；而传统 Unbounded Minimax 则显著偏向强者。

**⚠️ 局限性**

局限性：① Unbounded <balance‑near> 的高失误率表明其策略在实际平衡中不稳定；② 对极弱对手（如无评估的 MCTS）时，即便调整搜索时间也难以完全平衡；③ 目前仅在完美信息、确定性棋类上验证，缺乏对不确定性或部分信息游戏的扩展；④ 依赖较大搜索预算，无法在极限时间下保持平衡。

---

## 343. DBAutoDoc: Automated Discovery and Documentation of Undocumented Database Schemas via Statistical Analysis and Iterative LLM Refinement

**arXiv ID:** 2603.23050 | [PDF](https://arxiv.org/pdf/2603.23050v1)

**作者:** Amith Nagarajan `[一作]` (Blue Cypress), Thomas Altman `[通讯]` (Tasio Labs (a Blue Cypress company))

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出DBAutoDoc系统，自动为无键、无注释的“暗数据库”生成完整的表、列描述并恢复主键/外键关系。

**💡 创新点**

创新点在于将数据库文档生成视为图结构的迭代学习：通过统计关键字发现与LLM语义验证的双向反馈循环，再结合“反向传播”式的上下文传播实现多轮语义收敛。

**🔧 技术方法**

采用统计分析（唯一性、命名模式、取样值分布）与大型语言模型（Gemini、Anthropic、Sonnet等）结合的两阶段管线，使用基于依赖图的前向-后向迭代算法。

**📊 数据集**

评测数据集包括公开数据库AdventureWorks2022、Chinook、Northwind、合成暗数据库LousyDB，以及两家企业私有数据库OrgA和OrgB。

**📈 对比分析**

与单一统计或单向LLM方法对比，DBAutoDoc在AdventureWorks上实现PK F1 95%、FK F1 94%、表/列覆盖率近100%，整体加权分数96.1%，收敛仅需2次迭代，成本约$0.70/100表，远低于人工文档成本。

**⚠️ 局限性**

局限包括LLM的可能幻觉、对极稀疏或非英语列名的识别不足、统计方法对非标准键结构的召回受限、以及大规模数据库对token消耗和成本的影响。

---

## 344. VLA-IAP: Training-Free Visual Token Pruning via Interaction Alignment for Vision-Language-Action Models

**arXiv ID:** 2603.22991 | [PDF](https://arxiv.org/pdf/2603.22991v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 345. A Sobering Look at Tabular Data Generation via Probabilistic Circuits

**arXiv ID:** 2603.23016 | [PDF](https://arxiv.org/pdf/2603.23016v1)

**作者:** Davide Scassola `[一作]` (University of Trieste), Antonio Vergari `[通讯]` (University of Edinburgh)

**通讯引用:** 872 | [OpenAlex ID](https://openalex.org/A5069110696)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文批判了当前表格数据生成（TDG）领域中对模型进展的误判，指出常用的Fidelity指标存在缺陷，并提出了更鲁棒的评价指标；随后构建了一种基于概率电路（PC）的高效生成模型，并在多种UCI数据集上展示了其在生成质量、训练速度以及条件采样方面的竞争力；

**💡 创新点**

创新点在于：①提出权重归一互信息（wNMI）和使用XGBoost的C2ST作为更稳健的Fidelity指标；②首次将tensorised、过参数化的概率电路应用于TDG，形成快速、可解释的生成基线；③展示了PC在条件采样和精确似然评估上的优势。

**🔧 技术方法**

技术手段包括：概率电路（PC）框架（含深层、过参数化DAG结构与CP sum-product层）；全因子化与浅层混合模型对比；梯度训练（RAdam优化器）与EM算法；使用XGBoost作为判别器评估合成数据。

**📊 数据集**

实验使用了常见的UCI表格数据集：Adult、Beijing、Default、Diabetes、Magic、News和Shoppers。

**📈 对比分析**

与Diffusion-based DGMs（如TabDiff等）在Fidelity（wNMI、XGBoost C2ST）、Utility（下游任务性能）和训练时间上进行对比。结果显示，PC在大多数数据集上达到或超过Diffusion模型的Fidelity（>0.99），在4/7数据集上优于Diffusion；训练时间比Diffusion快1–2个数量级；在条件生成任务中，PC利用精确条件采样实现高逼真度。

**⚠️ 局限性**

局限性包括：仍需提升对高阶依赖的建模；所提出的指标虽更鲁棒，但仍不能完全覆盖所有数据属性；实验仅覆盖UCI数据集，可能无法完全推广到更大规模或不平衡的真实工业数据；PC模型在极大规模或高维度数据下的可扩展性仍待进一步验证。

---

## 346. Beyond Hate: Differentiating Uncivil and Intolerant Speech in Multimodal Content Moderation

**arXiv ID:** 2603.22985 | [PDF](https://arxiv.org/pdf/2603.22985v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 347. AgentRAE: Remote Action Execution through Notification-based Visual Backdoors against Screenshots-based Mobile GUI Agents

**arXiv ID:** 2603.23007 | [PDF](https://arxiv.org/pdf/2603.23007v1)

**作者:** Yutao Luo `[一作]` (Nanjing University of Science and Technology), Minhui Xue `[通讯]` (CSIRO Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于通知图标的视觉后门攻击AgentRAE，能远程诱导移动GUI代理执行恶意动作

**💡 创新点**

创新点：采用两阶段训练——监督对比学习分离图标特征，随后细化多目标后门映射；利用本地通知图标作为自然触发器；对多目标映射冲突和小触发器注意力失效进行系统解决

**🔧 技术方法**

技术：监督对比学习、后门微调（LoRA）、视觉语言模型（Qwen-VL-Chat）与多模态大语言模型结合；图标特征提取与对比损失；平衡清洁与后门损失

**📊 数据集**

数据集：GUIOdyssey（6设备、212个App）与AITW（30k指令、715k轨迹），在三种拆分上进行训练与评测

**📈 对比分析**

与AEIA、Pop‑ups、BadNets等基线对比；AgentRAE在不同拆分上多目标后门成功率>90%，清洁任务准确率仅下降<3%；在九目标映射下仍保持≈90% ASR，FPR低于对手

**⚠️ 局限性**

局限：只针对截图式移动GUI代理；对非截图或其它交互方式不一定适用；对手仍需具备后门模型分发能力；对抗防御（如Fine‑Tuning）仍可部分降低ASR，且需大量清洁数据

---

## 348. FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning

**arXiv ID:** 2603.22969 | [PDF](https://arxiv.org/pdf/2603.22969v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 349. Multi-User Multi-Key Image Steganography with Key Isolation

**arXiv ID:** 2603.23005 | [PDF](https://arxiv.org/pdf/2603.23005v1)

**作者:** Tzu-Ti Wei `[一作]`, Jen-Jee Chen `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了PUSNet‑MK，一个支持多用户、多钥匙的深度隐写网络；

**💡 创新点**

在共享骨干网络的基础上通过不匹配钥匙隔离损失实现了严格的钥匙隔离；

**🔧 技术方法**

采用稀疏权重填充、键控权重实例化、联合任务损失以及跨钥匙隔离损失等技术；

**📊 数据集**

在DIV2K、COCO和ImageNet三个公开数据集上进行训练与评估；

**📈 对比分析**

与HiDDeN、Baluja、UDH、HiNet以及原始PUSNet对比，PUSNet‑MK在保持隐写不可检测性的同时，显著提升了秘密恢复质量，并将跨钥匙泄露率大幅降低；

**⚠️ 局限性**

相较于原PUSNet在不可检测性上略有欠缺，且目前仅支持CNN骨干，实验范围局限于图像域，尚需进一步验证跨模态适用性与更大规模用户场景的鲁棒性。

---

## 350. Set-Valued Prediction for Large Language Models with Feasibility-Aware Coverage Guarantees

**arXiv ID:** 2603.22966 | [PDF](https://arxiv.org/pdf/2603.22966v1)

**作者:** Ye Li `[一作]` (University of Electronic Science and Technology of China), Bo Fu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 157582 | [OpenAlex ID](https://openalex.org/A5100399276)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向大型语言模型的可行性感知集合预测框架，利用有限采样构造覆盖保证的预测集；

**💡 创新点**

创新点在于：①引入最小可行风险水平（MRL）阐明有限采样下的覆盖下限；②设计数据驱动的学习-测试校准流程，保证在可行区间内满足统计覆盖；③将自不确定性、跨样本一致性与语义共识三维信息融合成可靠性评分；

**🔧 技术方法**

核心技术包括：分割式自适应校准（Split Conformal Prediction）、多视角可靠性评分（Self-uncertainty、Cross‑sample consistency、Semantic consensus）、语义去重、基于语义相似度的可接受性判定；

**📊 数据集**

使用六个开放式生成基准：CoQA、TriviaQA、SciQA、MedQA、MedMCQA 以及 Financial‑QA‑10K；

**📈 对比分析**

与传统单点预测（MLG）以及基线无校准方法对比，实验显示在可行风险水平以上时，预测集覆盖率接近或超过目标 1‑α，且平均预测集大小（APSS）显著降低，证明了方法在可靠性与效率上的优势；

**⚠️ 局限性**

局限性包括：需要满足交换性假设；当最小可行风险水平高时，覆盖保证无法实现；依赖较大采样预算和校准集，计算成本相对较高；

---

## 351. Accelerating Maximum Common Subgraph Computation by Exploiting Symmetries

**arXiv ID:** 2603.23031 | [PDF](https://arxiv.org/pdf/2603.23031v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 352. On the use of Aggregation Operators to improve Human Identification using Dental Records

**arXiv ID:** 2603.23003 | [PDF](https://arxiv.org/pdf/2603.23003v1)

**作者:** Antonio D. Villegas-Yeguas `[一作]` (University of Granada), Oscar Cordón `[通讯]` (University of Granada)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

针对法医牙齿记录比较设计并验证了多种可解释的聚合机制，以提升候选人排名的准确性。

**💡 创新点**

创新点包括：①基于数据驱动的字典序聚合优化；②引入多种模糊聚合算子（OWA、OWHM、Choquet、Sugeno）并对其参数进行实验性调优；③利用可解释机器学习模型（线性回归、符号回归、MLP）学习最优特征组合，显著提升排名性能。

**🔧 技术方法**

技术方法：牙齿编码采用SCS七码体系；特征提取为七个评价指标；聚合方法涵盖字典序、模糊算子、机器学习回归及神经网络；优化采用遗传算法最小化平均/最大排名。

**📊 数据集**

数据集为215例法医案例（来自以色列和智利），每例含一对前后记录，总计430份牙齿表。

**📈 对比分析**

与传统Adams‑Aschheim方案（平均排名3.91）相比，数据驱动字典序聚合平均排名2.14，符号回归2.02，MLP1.94；在95%阈值处可在前15%候选人内找到正确匹配，最大排名亦显著下降。

**⚠️ 局限性**

限制：样本量有限、仅覆盖两个人口群，且仅使用SCS编码，未验证对其他编码系统的泛化能力。

---

## 353. VQ-Jarvis: Retrieval-Augmented Video Restoration Agent with Sharp Vision and Fast Thought

**arXiv ID:** 2603.22998 | [PDF](https://arxiv.org/pdf/2603.22998v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 354. Parametric Knowledge and Retrieval Behavior in RAG Fine-Tuning for Electronic Design Automation

**arXiv ID:** 2603.23047 | [PDF](https://arxiv.org/pdf/2603.23047v1)

**作者:** Julian Oestreich `[一作]` (Institute for Applied Informatics (InfAI)), André Alcalde `[通讯]` (CELUS GmbH)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了 Retrieval‑Augmented Generation (RAG) 在电子设计自动化领域长文本生成的细调效果，并提出了人类验证的三元组提取归因管道 TriFEX，构造 Parametric Knowledge Precision (PKP) 指标来衡量模型内部知识的真实利用。

**💡 创新点**

创新点在于：1）将三元组提取与归因流程与人类审核相结合，形成可验证的评价体系；2）提出 PKP 以剔除检索泄露对内部知识评估的干扰；3）对自知度（Self‑Knowledge, SK）进行分解为 PKP 与 Parametric Rate (PR)，揭示其对检索条件的敏感性。

**🔧 技术方法**

使用技术包括：LoRA 微调 7B 语言模型；三元组提取、归因和 LLM 判定流程；dense embedding 相似度检索得到 top‑k 候选；并定义 PKP、PR、SK、F1_ref 等自定义指标。

**📊 数据集**

数据集来自公开的电路参考设计，人工合成约12k条需求工程条目，分别包含用户查询、参考答案和上下文；此外在 MMLU‑Electrical‑Engineering 上进行跨域验证。

**📈 对比分析**

与 72B 基线对比，本文使用 ROUGE、BERTScore 以及 TriFEX 三元组指标进行评估；传统指标难以捕捉事实差异，而 PKP 与 F1_ref 在细调的 7B 模型中显著提升，细调模型往往超越 72B 的性能。

**⚠️ 局限性**

主要限制包括：LLM 归因准确率约 80%，top‑k 候选可能漏掉有效证据；三元组验证耗时高，评估噪声存在；对小幅度差异的统计显著性需谨慎解释。

---

## 355. Assessing the Robustness of Climate Foundation Models under No-Analog Distribution Shifts

**arXiv ID:** 2603.23043 | [PDF](https://arxiv.org/pdf/2603.23043v1)

**作者:** Maria Conchita Agana Navarro `[一作]` (University College London), Maria Perez-Ortiz `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估了三种ML气候模拟器（U‑Net、ConvLSTM、ClimaX）在严格历史训练（1850‑2014）下对非分布外（Temporal extrapolation 与跨情景（SSP1‑2.6 与 SSP5‑8.5））的鲁棒性

**💡 创新点**

提出了结合时间外推与跨情景迁移的统一OOR评估框架，揭示了“准确度‑稳定性”权衡以及不同架构对无类比未来的敏感性

**🔧 技术方法**

使用卷积编码‑解码、卷积‑循环网络以及Transformer基础模型（ClimaX），并采用LL‑RMSE作为评价指标

**📊 数据集**

利用ClimateSet（CMIP6）月度forcing（CO₂, CH₄, BC, SO₂）及气候响应（tas, pr）数据集

**📈 对比分析**

在ID基准（1850‑2014）和两项OOD基准下对模型进行比较；结果显示ClimaX在ID上误差最低，但在OOR下相对误差增幅最大；U‑Net/ConvLSTM表现更稳定；温度预测相对稳健，降水预测在高排放情景下误差可达8.4%

**⚠️ 局限性**

局限在于高绝对误差（LL‑RMSE 0.8‑1.1，远高于最先进的0.2‑0.3）；仅对历史训练的单一任务进行fine‑tune；未探索多情景预训练、物理约束或因果泛化方法

---

## 356. Robustness Quantification and Uncertainty Quantification: Comparing Two Methods for Assessing the Reliability of Classifier Predictions

**arXiv ID:** 2603.22988 | [PDF](https://arxiv.org/pdf/2603.22988v1)

**作者:** Adrián Detavernier `[一作]` (Ghent University), Jasper De Bock `[通讯]` (Ghent University)

**通讯引用:** 500 | [OpenAlex ID](https://openalex.org/A5052699188)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

比较并结合鲁棒性量化与不确定性量化来评估分类器的个体预测可靠性。

**💡 创新点**

证明局部鲁棒性量化在分布漂移条件下优于不确定性量化，并且两者结合可进一步提升可靠性评估。

**🔧 技术方法**

采用概率生成分类器（朴素贝叶斯、生成森林）、信息熵、边际置信度、鲁棒性邻域方法以及AU-ARC评价指标。

**📊 数据集**

使用UCI机器学习仓库的多种离散特征数据集（如Student Performance、Solar Flare等），并在训练/测试拆分下进行实验。

**📈 对比分析**

通过ARC和AU-ARC对不同方法进行比较，局部鲁棒性在标准与高不确定性设置下多次取得最高分，混合方法通常优于单一方法。

**⚠️ 局限性**

局部鲁棒性主要适用于离散特征模型，扩展到复杂网络或连续特征尚未验证，混合权重可能存在过拟合风险。

---

## 357. WorldMesh: Generating Navigable Multi-Room 3D Scenes via Mesh-Conditioned Image Diffusion

**arXiv ID:** 2603.22972 | [PDF](https://arxiv.org/pdf/2603.22972v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 358. Can Graph Foundation Models Generalize Over Architecture?

**arXiv ID:** 2603.22984 | [PDF](https://arxiv.org/pdf/2603.22984v1)

**作者:** Benjamin Gutteridge `[一作]` (University of Oxford), Xiaowen Dong `[通讯]` (University of Oxford)

**通讯引用:** 2752 | [OpenAlex ID](https://openalex.org/A5101579932)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

提出了一种在零样本图节点分类任务中通过推理时自适应架构的图基础模型框架GOBLIN。

**💡 创新点**

创新点在于在推理阶段动态学习并混合一组线性图算子，从而克服固定骨干模型在不同任务范围下的欠拟合。

**🔧 技术方法**

使用贝叶斯优化搜索算子参数、线性GNN解析解、DeepSet混合专家与注意力权重来实现自适应。

**📊 数据集**

在合成的kHopSign、25个公开基准图（如Cora、Citeseer、CoPhysics等）以及城市网络长范围数据集上进行评估。

**📈 对比分析**

与GraphAny、TS-GNN以及端到端训练的MPNNs对比，GOBLIN平均提升约2%准确率，尤其在长距离任务上提升3–8%。

**⚠️ 局限性**

局限性包括模型表达能力不足、推理时算子学习与MoE组合导致推理成本高，以及在极端任务上仍不一定领先。

---

## 359. DariMis: Harm-Aware Modeling for Dari Misinformation Detection on YouTube

**arXiv ID:** 2603.22977 | [PDF](https://arxiv.org/pdf/2603.22977v1)

**作者:** Jawid Ahmad Baktash `[一作]` (Technical University of Munich), Mursal Dawodi `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了首个双轴标注的阿富汗达里语YouTube误信息数据集（DariMis），并在此数据集上开展误信息检测研究。

**💡 创新点**

创新点包括①双轴（信息类型×危害等级）标注并揭示其结构耦合；②提出标题-描述双输入BERT编码，显著提升误信息召回；③将低资源语言专用模型ParsBERT用于Dari，证明语言专用预训练的优势。

**🔧 技术方法**

采用BERT双段输入编码、ParsBERT与XLM‑RoBERTa对比、宏F1与召回评估，并使用Bootstrap 95%置信区间进行统计显著性分析。

**📊 数据集**

使用了9,224条带标题、描述、发布信息的达里语YouTube视频数据集DariMis（覆盖2007–2026年）。

**📈 对比分析**

通过70/15/15的分层划分比较模型，ParsBERT（双输入）在宏F1上最高为72.77%，与单输入仅差0.09pp，双输入在误信息召回提升+7pp；ParsBERT整体优于XLM‑RoBERTa。

**⚠️ 局限性**

局限包括：仅使用文本特征（未利用音频/视觉）；样本量有限导致统计显著性不充分；注释主观性导致分界模糊；缺失描述占31%削弱双输入优势；未联合预测危害等级；低资源语言预训练仍受限。

---

## 360. Where Experts Disagree, Models Fail: Detecting Implicit Legal Citations in French Court Decisions

**arXiv ID:** 2603.22973 | [PDF](https://arxiv.org/pdf/2603.22973v1)

**作者:** Avrile Floro `[一作]` (Institut Polytechnique De Paris), Nils Holzenberger `[通讯]` (Institut Polytechnique De Paris)

**通讯引用:** 365 | [OpenAlex ID](https://openalex.org/A5017498603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个用于检测判决书中隐式引用法国民法典条文的标注数据集。

**💡 创新点**

首次量化专家标注分歧与模型错误的关系，证明模型错误集中在专家争议案例。

**🔧 技术方法**

结合了多模型集成、零样本LLM投票、无监督排序以及反向过滤技术。

**📊 数据集**

使用从法国司法部开放API收集的182,155条判决，经过筛选后得到1,015条包含不同条文的片段。

**📈 对比分析**

在有监督集成中F1为0.70、精度77%；在无监督投票中k=200时精度达76%，召回34%，显著优于随机。

**⚠️ 局限性**

主要限制是数据集规模有限、反向过滤可能偏见以及无监督投票权重未经过优化。

---

## 361. Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining

**arXiv ID:** 2603.22953 | [PDF](https://arxiv.org/pdf/2603.22953v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 362. Beyond Theoretical Bounds: Empirical Privacy Loss Calibration for Text Rewriting Under Local Differential Privacy

**arXiv ID:** 2603.22968 | [PDF](https://arxiv.org/pdf/2603.22968v1)

**作者:** Weijun Li `[一作]` (Macquarie University), Mark Dras `[通讯]` (Macquarie University)

**通讯引用:** 2611 | [OpenAlex ID](https://openalex.org/A5022216838)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于区分度攻击的 LDP 文本重写机制的经验校准框架。

**💡 创新点**

首次将可区分性审核从表格数据推广到高维文本，并引入候选集策略和高效估计方法。

**🔧 技术方法**

利用 LLM 语义判断、词嵌入距离、概率采样、Monte Carlo 估计等技术。

**📊 数据集**

在 ATIS、SST-2、SNIPS、Trustpilot 四个公开数据集上评估。

**📈 对比分析**

通过对六种主流 LDP 重写方法的实证区分度测试，发现同样 ε 下隐私泄露差异显著，说明经验校准能更准确比较隐私-效用权衡。

**⚠️ 局限性**

局限在于候选集构造依赖训练集，样本规模有限，且对极低 ε 或高维嵌入的解释仍不完全。

---

## 363. Learning Actuator-Aware Spectral Submanifolds for Precise Control of Continuum Robots

**arXiv ID:** 2603.23044 | [PDF](https://arxiv.org/pdf/2603.23044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 364. HUydra: Full-Range Lung CT Synthesis via Multiple HU Interval Generative Modelling

**arXiv ID:** 2603.23041 | [PDF](https://arxiv.org/pdf/2603.23041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 365. Traffic Sign Recognition in Autonomous Driving: Dataset, Benchmark, and Field Experiment

**arXiv ID:** 2603.23034 | [PDF](https://arxiv.org/pdf/2603.23034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 366. Privacy-Preserving EHR Data Transformation via Geometric Operators: A Human-AI Co-Design Technical Report

**arXiv ID:** 2603.22954 | [PDF](https://arxiv.org/pdf/2603.22954v1)

**作者:** Maolin Wang `[一作]` (City University of Hong Kong), Jun Yan `[通讯]` (City University of Hong Kong)

**通讯引用:** 36629 | [OpenAlex ID](https://openalex.org/A5011227147)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种面向结构化临床记录的实用隐私保护数据转换框架，利用均值-方差流形和统一的z‑score ℓ∞ 限制，构造可在医院内部 CPU 环境中批量生成可视化且不可逆的数值视图。

**💡 创新点**

创新点包括：
- 以均值-方差流形为几何基准，统一将所有变量映射到同一标准化空间，显式引入单一隐私 knob α；
- 设计三种低计算成本的列级几何变换（局部三元组旋转 T1、噪声+投影 T2、全局 Householder 反射 T3）并通过 Q‑mix 对高风险变量实现一次性隐私“阶跃”；
- 通过人‑AI（SciencePal）共设计流程，系统化地验证隐私非可逆性并筛选正负案例；
- 在 MIMIC‑IV ICU 数据上构建完整的隐私评估协议（L0/L1/L2 泄漏水平 + A/B/C/D 攻击族），在实际数据上量化重构、记录匹配、成员推断与属性泄露的安全性与可用性。

**🔧 技术方法**

使用的技术包括：
- 均值-方差流形的几何约束与标准化；
- 统一的 ℓ∞ α 约束；
- 随机局部旋转、噪声注入与投影、Householder 反射等线性几何变换；
- 每个住院记录的本地正交 Q‑mix 置换；
- 简单线性重构、记录链接、成员推断和属性推断模型；
- 统计评估：Kolmogorov–Smirnov 距离、KS 误差、MAE、R²、AUC、Reid@k 等。

**📊 数据集**

使用 MIMIC‑IV ICU 时间序列子集（包含心率、血压、葡萄糖等 8 个关键变量），所有数据在 1 小时网格上对齐，缺失值采用前向填充 + 线性插值，后续按变量拆分为列进行实验。

**📈 对比分析**

与传统的多方安全计算、同态加密、差分隐私合成等方法相比，T1/T2 在保持均值/方差、短期自相关、相关矩阵和临床可解释性方面与原始数据几乎无差异；α=1.0 时在重构攻击下 R² 下降到 0.7–0.8，Q‑mix 则将 R² 降至 ≈0；在记录链接、成员推断与属性泄露上，Q‑mix+T1/T2 的表现接近随机基线，显示出显著的隐私提升。相比于全局反射 T3，后者几乎保持完美可逆性，说明仅凭几何相似性不足以保证安全。

**⚠️ 局限性**

局限性包括：
- 仅在无密钥、结构感知的攻击模型下验证，未提供严格的加密级别安全证明；
- 评估主要聚焦单列和 ICU 时间序列，对多变量跨表或非时间序列数据的表现未知；
- 对于高维稀疏变量、极端缺失率或非正态分布，α‑hierarchy 的隐私提升仍相对平滑；
- 需要内部随机种子和本地计算资源，若泄露可重构；
- 目前未在跨机构联邦学习或大模型训练中进行验证，后续研究需进一步评估对深度学习性能的影响。

---

## 367. Asymptotic Learning Curves for Diffusion Models with Random Features Score and Manifold Data

**arXiv ID:** 2603.22962 | [PDF](https://arxiv.org/pdf/2603.22962v1)

**作者:** Anand Jerry George `[一作]` (École Polytechnique Fédérale de Lausanne), Nicolas Macris `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 1986 | [OpenAlex ID](https://openalex.org/A5038854134)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

研究了去噪得分匹配的理论行为，该学习任务与扩散模型相关，特别是在数据分布支持于低维流形时，使用随机特征神经网络参数化得分。

**💡 创新点**

发现对于线性流形，学习得分函数所需的样本复杂度与流形的内在维度线性相关，而不是与环境维度相关。非线性流形时，低维结构的好处开始减弱。

**🔧 技术方法**

使用随机特征神经网络（RFNN）来参数化得分函数。

**📊 数据集**

使用隐藏流形模型（HMM）生成的数据集，数据位于低维流形上。

**📈 对比分析**

通过与现有方法的比较，发现当数据分布支持于低维流形时，样本复杂度显著降低。对于线性流形，样本复杂度与内在维度线性相关，而在非线性流形中，效果减弱。

**⚠️ 局限性**

当流形的非线性增加时，低维结构的优势减弱，情况变得更像是在环境空间中有分布。

---

## 368. Weak-PDE-Net: Discovering Open-Form PDEs via Differentiable Symbolic Networks and Weak Formulation

**arXiv ID:** 2603.22951 | [PDF](https://arxiv.org/pdf/2603.22951v1)

**作者:** Xinxin Li `[一作]` (East China Normal University), Junping Yin `[通讯]` (East China Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Weak‑PDE‑Net，一种端到端可微分的框架，用于从稀疏噪声观测中发现开放式偏微分方程（PDE）；

**💡 创新点**

创新点包括：1) 引入可学习高斯核嵌入以自适应补偿 MLP 的频谱偏差；2) 采用可微分神经架构搜索（DNAS）动态构造符号网络，自动生成开放式 PDE 结构；3) 将弱形式积分与符号网络耦合，避免显式数值微分；4) 通过 Galilean 不变性和对称等变性约束提升多变量系统的物理一致性；

**🔧 技术方法**

核心技术包括可学习高斯嵌入、轻量级 MLP、可微分符号网络、弱形式积分模块、DNAS、物理约束正则化；

**📊 数据集**

使用多种标准 PDE 基准数据集：1D Burgers、KdV、Kuramoto‑Sivashinsky、Chafee‑Infante；2D Wave、Sine‑Gordon、Navier‑Stokes；复数系统 NLS；数据为从高精度数值模拟中随机采样的稀疏、噪声点；

**📈 对比分析**

与 Weak‑PDE‑LEARN 等传统方法对比，Weak‑PDE‑Net 在稀疏率 2.5%–50% 和噪声 20%–100% 的极端条件下均能保持 TPR=1.0，参数误差 E∞、E₂ 低于基线；在高噪声（>80%）下仍能恢复正确方程；相较于 Weak‑PDE‑LEARN，性能更稳健、误差更小；

**⚠️ 局限性**

局限性：目前无法处理嵌套微分算子（如多重导数乘以函数），弱形式积分的低通滤波特性会抑制细尺度特征；对极其复杂的多级结构可能需要进一步扩展网络架构。

---

## 369. Few-Shot Generative Model Adaption via Identity Injection and Preservation

**arXiv ID:** 2603.22965 | [PDF](https://arxiv.org/pdf/2603.22965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 370. Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation

**arXiv ID:** 2603.23030 | [PDF](https://arxiv.org/pdf/2603.23030v1)

**作者:** ByeongCheol Lee `[一作]` (Institution1), Jae-Pil Heo `[通讯]` (Institution2)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**



**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 371. Can Large Language Models Reason and Optimize Under Constraints?

**arXiv ID:** 2603.23004 | [PDF](https://arxiv.org/pdf/2603.23004v1)

**作者:** Fabien Bernier `[一作]` (University of Luxembourg), Maxime Cordy `[通讯]` (University of Luxembourg)

**通讯引用:** 1971 | [OpenAlex ID](https://openalex.org/A5000695937)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究LLM在真实电网最优潮流（OPF）问题下的约束推理与优化能力。

**💡 创新点**

提出了基于OPF的严格评测框架，包含约束满足度、MSE和结构有效率等指标，并在SFT与GRPO两种训练方式下系统评估LLM性能。

**🔧 技术方法**

使用大语言模型（Llama 3.1、Qwen 3、Qwen 2.5）进行对话式推理与强化学习，构造JSON输出模式并利用群体相对策略优化（GRPO）奖励约束满足。

**📊 数据集**

采用公开的OPF数据集（IEEE 14、30、118桩电网）及其N‑1扰动版本作为评测数据。

**📈 对比分析**

与零样本、ICL、SFT、GRPO等多种训练与推理方式对比；结果显示所有模型在约束满足度上停留在55–60%，即使在GRPO下也仅有少量提升，MSE和输出有效率提升有限。

**⚠️ 局限性**

主要局限在于LLM无法真正执行数值迭代与矩阵运算，往往靠记忆或模式匹配产生近似解，缺乏对物理约束的深层推理；SFT与GRPO对约束满足的提升有限，提示需结合外部求解器或更细粒度奖励设计。

---

## 372. Zero-Shot Personalization of Objects via Textual Inversion

**arXiv ID:** 2603.23010 | [PDF](https://arxiv.org/pdf/2603.23010v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 373. PaperVoyager : Building Interactive Web with Visual Language Models

**arXiv ID:** 2603.22999 | [PDF](https://arxiv.org/pdf/2603.22999v1)

**作者:** Dasen Dai `[一作]` (Vast Intelligence Lab), Wenhao Wang `[通讯]` (Vast Intelligence Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将研究论文转化为可直接运行的交互式网页系统，支持用户动态操作并观察机制演化。

**💡 创新点**

提出 PaperVoyager 代理，采用结构化规格设计、块级分块生成以及 VLM 视觉评估，显著提升交互式系统生成的可靠性和一致性。

**🔧 技术方法**

多模态大语言模型、结构化规范化、块级生成与筛选、VLM 视觉评估、React + TypeScript 前端实现。

**📊 数据集**

构建了 19 篇论文与专家手工实现的交互式网页系统对应的基准数据集，用于评估模型生成效果。

**📈 对比分析**

与 GPT5.2、MiniMax、Qwen-Max、Kimi-K2、Claude Code、Gemini-3-Pro 等基线模型对比，PaperVoyager 在平均任务成功率约 80.7% 上显著优于所有基线。

**⚠️ 局限性**

局限在于基准规模仅 19 篇，缺乏大规模训练集，评估场景固定，未覆盖更广泛的模型或更大规模的实验。

---

## 374. Tightly-Coupled Radar-Visual-Inertial Odometry

**arXiv ID:** 2603.23052 | [PDF](https://arxiv.org/pdf/2603.23052v1)

**作者:** Morten Nissov `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 7937 | [OpenAlex ID](https://openalex.org/A5022659812)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发一种紧耦合的雷达-视觉-惯性里程计（Radar-Visual-Inertial Odometry，RVI），利用 FMCW 雷达的多普勒速度测量和雷达射程数据与视觉特征强度更新共同实现在各种光照、障碍及高速度场景下的稳健定位。

**💡 创新点**

1）将雷达多普勒速度直接作为状态更新，避免传统最小二乘速度估计的稀疏/噪声问题；2）用雷达射程点云在视觉特征检测时进行深度初始化，解决单目视觉深度可观测性差的瓶颈；3）在 EKF 中实现雷达、视觉、IMU 的联合状态估计并在线估计雷达与惯性坐标的外参，提升长期漂移控制。

**🔧 技术方法**

基于 iekf 的状态机（包含导航状态、外参状态、视觉特征状态）；Fast 特征提取与补丁强度更新；雷达点云体素滑动窗口深度初始化；多普勒速度测量函数与残差；外参在线优化；使用 MATLAB/ROS/C++ 实现并发布在 GitHub。

**📊 数据集**

利用自制的无人机平台进行实地飞行实验：在森林、室内、开放平原等环境下采集 IMU、相机、TI IWR6843AOPEVM FMCW 雷达、GNSS、LiDAR 数据。使用 LiDAR‑based LIO 与 GNSS‑IMU‑Barometer 组网产生地面真值，用于 Ape、RPE、相对漂移等指标评估。

**📈 对比分析**

与 ROVIO（单目视觉+IMU）及现有雷达‑视觉融合方法（Radial‑Speed‑Factor‑Graph 等）在同一数据集上进行对比。实验表明：RVI 在标准光照下的 RPE 与 Ape 均优于单独方法；在黑暗、雾霾、以及高速飞行等极端条件下，单目视觉或雷达均会失效，而 RVI 仍能保持稳定、低漂移，最终位置误差仅约 2 cm，明显优于对比方法。

**⚠️ 局限性**

（1）雷达点云稀疏与噪声限制，仍需改进稀疏场景下的速度估计；（2）高速度飞行会导致雷达多普勒范围受限，导致滤波发散；（3）缺少公开数据集，实验结果基于自研平台，缺乏跨平台验证；（4）对外参在线估计依赖初始猜测，若误差过大仍需较长收敛时间。

---

## 375. Gendered Communication Patterns of Political Elites on Truth Social

**arXiv ID:** 2603.23027 | [PDF](https://arxiv.org/pdf/2603.23027v1)

**作者:** Tom Bidewell `[一作]` (University of Edinburgh), Björn Ross `[通讯]`

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统地分析了Truth Social平台上129名美国政治精英（共107,393条帖子）在性别维度上的沟通模式，探讨性别如何影响发帖内容、修辞风格及观众互动。

**💡 创新点**

创新之处在于首次大规模聚焦Alt-tech平台的性别差异，发现女性政治精英更频繁使用恐惧话语、表达更多喜悦情绪，并在回复和点赞上显著获得更高参与度。

**🔧 技术方法**

使用了Llama 3.1 Instruct进行内容分类，RoBERTa模型提取情感与恐惧/仇恨言论，BERTopic进行主题建模，梯度基重要性分析和贝叶斯回归（负二项、逻辑回归）评估特征影响。

**📊 数据集**

数据集为Truth Social API收集的107,393条帖子，覆盖129名政治精英（77%男性，23%女性），并标注性别、职位、情绪、话语类型等特征。

**📈 对比分析**

通过贝叶斯多项式与负二项回归对比性别、职位、情绪和话语类型对帖子内容和观众响应的影响；结果显示女性帖子在回复率和点赞率上分别提升约821%和555%，并揭示情绪与恐惧话语与互动的非线性关系。

**⚠️ 局限性**

局限包括样本高度白人化、缺乏对沟通团队与个人作者的区分、未能捕捉平台关键事件后变化、主题建模需人工解释、无法验证因果关系。

---

## 376. On the Suboptimality of Rate--Distortion-Optimal Compression: Fundamental Accuracy Limits for Distributed Localization

**arXiv ID:** 2603.23006 | [PDF](https://arxiv.org/pdf/2603.23006v1)

**作者:** Amir Weiss `[一作]` (Bar-Ilan University), Amir Weiss `[通讯]` (Bar-Ilan University)

**通讯引用:** 877 | [OpenAlex ID](https://openalex.org/A5081331925)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究在分布式定位场景下，当融合中心只能获得各传感器压缩后（符合率失真理论的）观测时，定位精度的理论下限。

**💡 创新点**

发现传统的压缩-先估计方法（为信号重构而最优的压缩）可能严重削弱定位性能，尤其在高频信息被丢弃时；提出利用频域 Fisher 信息和 CRLB 明确阐述了这种“压缩子优”现象，并给出闭式的两频段模型与量化的性能提升。

**🔧 技术方法**

使用了高斯信号的率失真理论的测试通道模型、Whittle 频域 Fisher 信息、CRLB 推导以及水填充式压缩解析。

**📊 数据集**

该工作为纯理论分析，未使用实际数据集；通过数学模型（如两频段、两水平谱模型）进行验证。

**📈 对比分析**

通过对比 RD‑最优压缩与特意的频段选择压缩方案，计算得到 CRLB 的闭式表达式。实验示例显示，当压缩率低于临界值时，频段选择方案可将 CRLB 降低约两倍（即提高约两位数的精度）。

**⚠️ 局限性**

局限性包括：仅考虑了高斯、无多径、线下视（LOS）的场景；压缩方案仅针对均匀噪声和单一功率谱；未给出实际实现或仿真验证；未探讨多任务或动态资源分配的最优设计。

---

## 377. A Critical Review on the Effectiveness and Privacy Threats of Membership Inference Attacks

**arXiv ID:** 2603.22987 | [PDF](https://arxiv.org/pdf/2603.22987v1)

**作者:** Najeeb Jebreel `[一作]` (Universitat Rovira i Virgili), Josep Domingo-Ferrer `[通讯]` (Universitat Rovira i Virgili)

**通讯引用:** 13016 | [OpenAlex ID](https://openalex.org/A5051455237)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过评估框架系统性地评估了Membership Inference Attacks（MIAs）的有效性和隐私威胁，并对现有主流攻击方法进行了综合复盘。

**💡 创新点**

创新点在于提出了基于五个必需条件（C0–C4）的MIAs评估框架，揭示了当前攻击在非过拟合、竞争性模型、可靠性和计算成本等维度的不足，并首次将这些维度统一起来进行评估。

**🔧 技术方法**

技术手段包括：文献综述、条件评估方法（C0–C4）、对公开数据集上 MIAs 实验结果的重新计算与比较，以及对攻击精度、召回率、误报率、计算成本等指标的统计分析。

**📊 数据集**

使用的主要数据集包括 Adult、Purchase‑100、Texas‑100、UCI Credit、UCI Cancer、UCI Hepatitis、UCI Cancer、MNIST、CIFAR‑10/100、ImageNet‑1K、LFW、Newsgroups、RCV1X 等常用公开数据集。

**📈 对比分析**

比较方法是按 C1–C4 条件对每个攻击‑数据集对进行筛选，并在满足条件的对中比较攻击的精度、召回率、误报率和计算成本；结果显示，除在 MNIST 上有少数攻击满足部分条件外，绝大多数攻击在至少一项关键条件上未达标，实际可信度低。

**⚠️ 局限性**

局限性包括：MIAs 对非完整训练集、缺乏唯一属性的场景下隐私威胁有限；评估假设（如成员先验、参考数据匹配）过于理想化；计算成本高、缺乏可重复性；对模型过拟合的依赖导致在实际部署中效果不佳。

---

## 378. How Far Should We Need to Go : Evaluate Provenance-based Intrusion Detection Systems in Industrial Scenarios

**arXiv ID:** 2603.22982 | [PDF](https://arxiv.org/pdf/2603.22982v1)

**作者:** Yue Xiao `[一作]` (Tsinghua University), Qi Li `[通讯]` (Tsinghua University)

**通讯引用:** 26292 | [OpenAlex ID](https://openalex.org/A5100350243)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地评估并分析了五种最新的基于原型的入侵检测系统（PIDS）在工业场景中的表现，并针对实际工况构建了实验数据集。

**💡 创新点**

首次在工业环境下对PIDS进行系统性评估，提出了行业的三大新特征（异构多源输入、更强攻击者、日益复杂的良性活动），提供了针对高误报、可移植性差等问题的实用性洞察与改进建议，并探讨了UUID分配和细粒度检测与调查效率的关系。

**🔧 技术方法**

采用了基于图神经网络的异常检测技术（GAT、GraphSAGE、GCN、Temporal Graph Network 等），利用语义信息（命令行、文件名、IP）构建节点嵌入，并引入无监督方法（Louvain 社区检测、K‑means 过滤）来降低误报；此外，还使用 LLM 对攻击命令进行自动分析。

**📊 数据集**

主要使用来自两大企业服务（安全管理服务和云工作负载服务）的真实日志数据，补充了 DARPA Transparent Computing 公开数据集，并为矿业攻击和信息窃取攻击专门构建了实验数据集。

**📈 对比分析**

通过 AUC、TPR/FPR、训练/测试时延等指标，对五个 PIDS 进行横向比较。结果表明：在不同主机/平台间迁移性能下降 30% 以上；对真实攻击的检测 AUC 低至 40%–60%；误报率在动态良性环境下可达 25% 以上；时间开销随图规模线性增长，且各方法差距可达数十倍。

**⚠️ 局限性**

研究仅基于单一组织的数据，缺乏公开可复现的数据集；未对抗性攻击进行深入评估；所选的五种 PIDS 并不覆盖所有可能的技术路线；因此结果在其他工业场景中的泛化能力需要进一步验证。

---

## 379. JFTA-Bench: Evaluate LLM's Ability of Tracking and Analyzing Malfunctions Using Fault Trees

**arXiv ID:** 2603.22978 | [PDF](https://arxiv.org/pdf/2603.22978v1)

**作者:** Yuhui Wang `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 16850 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于 JSON 的结构化文本表示法（JFTA）用于描述故障树，并基于此构建了一个多轮对话基准（JFTA‑Bench），通过训练用户模拟器来生成含有模糊信息的交互对话，进而评估大语言模型在复杂系统故障定位与错误恢复方面的能力。

**💡 创新点**

创新点包括：① 将图像形式的故障树转化为可直接被 LLM 处理的可解析文本格式；② 设计了包含长回滚与恢复的多轮交互场景，模拟真实用户错误；③ 训练了专门生成模糊、非专业反馈的用户模拟器，为对话评测提供更逼真的对话环境。

**🔧 技术方法**

使用的技术包括：JSON 结构化语法、ReAct 交互框架、PPO 强化学习优化用户模型、LLM（Gemini 2.5 pro、GPT‑5、Claude Sonnet 4.5、Qwen3‑32B、DeepSeek‑V3.2）以及路径采样与树扩展算法。

**📊 数据集**

数据集来源于 126 条来自 24 个工业领域的故障树，平均 140 个节点，采样得到 3130 条故障路径（每条包含 2 条子路径以支持回滚测试），平均交互 40.75 轮；用户模型训练集 100 树，测试集 26 树。

**📈 对比分析**

通过对比不同 LLM 的成功率（正确定位所有根因并给出解决方案），Gemini 2.5 pro 在测试集上达到 53.76% 的成功率，DeepSeek‑V3.2 为 41.40%；相较于节点‑边表示，JFTA 在所有难度级别上均表现更好；子集评测结果显示方差 <0.54%，验证了评测的可靠性。

**⚠️ 局限性**

局限性包括：① 数据集为半合成，缺乏真实生产线中罕见故障的噪声与复杂度；② 长回合对话评测计算成本高，需要使用子集评测；③ JFTA 尚未在实际工业维护软件中部署，实时性能与易用性待验证。

---

## 380. A Practical Framework for Flaky Failure Triage in Distributed Database Continuous Integration

**arXiv ID:** 2603.23054 | [PDF](https://arxiv.org/pdf/2603.23054v1)

**作者:** Jun-Peng Zhu `[一作]` (Northwest A&F University), Qi Liu `[通讯]` (PingCAP)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了SCOUT框架，用于分布式数据库持续集成（CI）环境下在CPU毫秒级预算内实时判定失败是易失性（flaky）还是持久性（persistent），并决定是否自动重跑或上报。

**💡 创新点**

创新点包括：①仅使用严格因果的预失败遥测和历史特征，避免泄漏；②轻量化状态感知评分（仅用logistic回归+可选稀疏元数据融合）保证毫秒级推理；③针对固定阈值决策的可迁移校准（OA‑Cal）实现跨时间、跨工作负载的阈值复用；④基于贝叶斯后验的重跑预算修正，补偿有限重跑导致的标签偏差。

**🔧 技术方法**

技术主要有：严格因果特征提取（窗口[-120s,0]的平均/最大/标准差/95%分位数）、轻量化logistic模型（StateLR/SDJ-LR）、后置校准（sigmoid、BetaCal、Isotonic、OA‑Cal）以及重跑预算校正（posterior‑soft）。

**📊 数据集**

使用的数据集包括：①3,680条标注失败的合成基准（462个易失性案例）；②TiDB v7/v8 实际服务的1,600条失败（341条标注，194易失性）；③GitHub Actions 157,807条工作流元数据，产生36,241个失败剧集（3,953标注有效）。

**📈 对比分析**

与多种基线（历史+波动LR、序列LR、树模型、BERT等）对比，SCOUT在PR‑AUC上提升约40%（0.12→0.20~0.21），校准后固定阈值决策成本下降超过50%（765→497），重跑预算校正使ECE从0.32降至0.027，在线推理P95仅1.17 ms，证明框架在精准性、可迁移性和实时性上均具备优势。

**⚠️ 局限性**

局限性包括：①依赖丰富的预失败遥测，若监控缺失或延迟会影响效果；②仅处理“单一阈值”决策场景，对多成本或多阈值情况尚未覆盖；③在极端分布漂移或新故障模式下，校准与重跑修正的泛化仍需进一步验证。

---

## 381. Concept-based explanations of Segmentation and Detection models in Natural Disaster Management

**arXiv ID:** 2603.23020 | [PDF](https://arxiv.org/pdf/2603.23020v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 382. MSR-HuBERT: Self-supervised Pre-training for Adaptation to Multiple Sampling Rates

**arXiv ID:** 2603.23048 | [PDF](https://arxiv.org/pdf/2603.23048v1)

**作者:** Zikang Huang `[一作]` (Tianjin University), Jianwu Dang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 4678 | [OpenAlex ID](https://openalex.org/A5017251198)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种多采样率自监督学习框架 MSRHuBERT，解决传统 HuBERT 在不同采样率下的分辨率不匹配问题，能够在不重采样的情况下对多率语音进行统一预训练与微调。

**💡 创新点**

创新点在于设计了多采样率自适应下采样卷积网络，将不同采样率波形映射到统一的 20 ms 帧级特征；同时保留 HuBERT 的掩码预测任务和 Transformer 编码器，使得多率预训练可直接使用单一共享代码本和已有的改进方法。

**🔧 技术方法**

使用的技术包括：时间域下采样 CNN（按采样率自定义卷积核与步幅）、层归一化、掩码预测损失（对数软最大化）、共享 Transformer 编码器、共享离散伪标签代码本，以及在微调时采用的 SUPERB 轻量级任务头。

**📊 数据集**

实验数据集：960 小时 LibriSpeech（16 kHz），DNS Challenge 2022（22.05 kHz、24 kHz、48 kHz），LJSpeech（22.05 kHz），LibriTTS（24 kHz）以及 VCTK（48 kHz），用于预训练和下游任务的评估。

**📈 对比分析**

对比方法：将 MSRHuBERT 与单采样率 HuBERT Base、以及对多率数据进行重采样后训练的 HuBERT 进行对比；在 ASR（WER）和全带宽语音重建（STOI）任务上，MSRHuBERT 在多采样率下保持或提升性能，尤其在混合率微调时表现优异；实验显示分辨率不匹配会导致显著性能下降，而 MSRHuBERT 能有效缓解该问题。

**⚠️ 局限性**

局限性：仍需要为极高采样率（>48 kHz）手动设计下采样模块；对极端高频任务仍可能需要后处理；模型规模略增（多率分支占 3% 参数），并未解决所有多率场景的跨域适配问题。

---

## 383. YOLOv10 with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection and trustworthy multimodal AI in computer vision perception

**arXiv ID:** 2603.23037 | [PDF](https://arxiv.org/pdf/2603.23037v1)

**作者:** Marios Impraimakis `[一作]` (University of Bath), Feiyu Zhou `[通讯]` (Zhejiang University)

**通讯引用:** 458 | [OpenAlex ID](https://openalex.org/A5010987570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一套基于YOLOv10与Kolmogorov‑Arnold网络（KAN）以及BLIP视觉‑语言模型的可解释目标检测框架，用于在自动驾驶等视觉感知任务中实时评估检测置信度并生成自然语言描述。

**💡 创新点**

创新点在于将KAN作为可解释后处理器，将检测置信度建模为七维可解释特征（位置、尺寸、置信度、类别、尺度）的加性样条函数，从而直观展示各特征对置信度的影响，并结合BLIP生成轻量级多模态解释。

**🔧 技术方法**

主要技术包括YOLOv10一阶段目标检测、基于B‑spline的Kolmogorov‑Arnold网络 surrogate、Bootstrapped Language‑Image Pretraining (BLIP) 的视觉‑语言描述器以及多维特征可视化（partial dependence、edge importance 等）。

**📊 数据集**

使用的数据集为通用目标检测数据集COCO的车辆子集以及英国巴斯大学校园的实景图像，覆盖清晰、模糊、遮挡等多种视觉复杂场景。

**📈 对比分析**

与YOLOv10原始置信度比较，KAN surrogate 在各置信度区间保持 R²>0.99、MAE<0.016，显著提升低信度检测的可解释性；BLIP 生成的字幕与人工标注语义相似度高于 0.75，且不影响检测性能。

**⚠️ 局限性**

局限性包括：（1）KAN 仅对检测头输出特征可解释，无法直接解释整个网络内部；（2）对极端置信度边缘（<0.25 或 >0.95）时可解释性略弱；（3）BLIP 需要额外的预训练模型，推理时仍有一定延迟，且对极端遮挡场景的描述仍不够精准。

---

## 384. Cog3DMap: Multi-View Vision-Language Reasoning with 3D Cognitive Maps

**arXiv ID:** 2603.23023 | [PDF](https://arxiv.org/pdf/2603.23023v1)

**作者:** Chanyoung Gwak `[一作]` (POSTECH), Minsu Cho `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了Cog3DMap框架，能够通过递归方式从多视角图像构建显式的三维认知图（3D记忆），并将其作为视觉token输入到多模态大语言模型（MLLM）进行空间推理。

**💡 创新点**

创新点在于：①引入显式、非冗余的3D记忆，令每个空间坐标对应唯一的视觉token；②将语义特征与几何特征融合以提升空间信息表达；③采用递归更新机制，逐步整合多视角观测，避免传统方法中的隐式推断和冗余信息。

**🔧 技术方法**

核心技术包括：使用Point3R预训练模型进行点图预测和几何特征提取；使用ViT编码器获取语义特征；通过可学习投影器融合语义与几何特征；递归更新的3D认知图记忆；以及在Qwen3‑VL等MLLM中引入时间/空间分隔符以支持时序理解。

**📊 数据集**

主要实验数据集为：VSTI‑Bench、VSI‑Bench、RoboFAC（机器人视觉问答）以及Scan2Cap（3D描述生成）。

**📈 对比分析**

与现有最优模型（如VLM‑3R、VST‑7B、Qwen3‑VL）相比，Cog3DMap在VSTI‑Bench平均提升约8.7%（其中相机运动预测提升27.5%），在VSI‑Bench平均提升约3.9%；在RoboFAC上显著降低视觉token数量（最高可达90.2%），同时保持或优于对手性能。

**⚠️ 局限性**

局限性包括：在高度动态场景下，递归聚合可能导致不同物体经过同一空间位置时特征被覆盖；需要两阶段训练，几何特征需先预训练，未实现端到端；并未在极度动态或遮挡复杂的环境中充分验证。

---

## 385. Knowledge Access Beats Model Size: Memory Augmented Routing for Persistent AI Agents

**arXiv ID:** 2603.23013 | [PDF](https://arxiv.org/pdf/2603.23013v1)

**作者:** Xunzhuo Liu `[一作]` (vLLM Semantic Router Project), Huamin Chen `[通讯]` (Red Hat)

**通讯引用:** 2251 | [OpenAlex ID](https://openalex.org/A5101790571)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在生产式 AI 代理场景中，提出了一种将对话记忆与基于置信度的路由相结合的推理框架，利用小模型低成本完成大多数查询，并通过跨模型记忆注入提升答案质量。

**💡 创新点**

创新点在于：① 证明记忆注入并不改变路由决策，而是显著提升小模型在用户特定查询上的准确性；② 发现知识获取比模型规模更关键；③ 通过混合检索（dense + BM25）进一步提升性能，展示了记忆质量对全链路影响。

**🔧 技术方法**

使用技术包括：Qwen3‑8B 与 Qwen3‑235B 两大模型；将对话回合对存入向量数据库并通过 dense + BM25 混合检索；基于平均 log‑prob 的置信度阈值实现无训练的路由；跨模型记忆注入机制。

**📊 数据集**

采用的评估数据集有：LoCoMo 对话记忆 QA 基准（152 问题）；LongMemEval 多会话基准（500 问题）；以及公开的生产 LLM 交互日志用于工作负载分析。

**📈 对比分析**

通过 2×2 因子实验（记忆 × 路由）比较，发现内存+路由在 LoCoMo 上实现 30.5% F1（比单 8B 提升 15 点，恢复 69% 235B 全上下文性能），并将有效成本降低 96%。在 LongMemEval 上，混合检索相较于仅 dense 检索提升 7.7 F1，验证检索质量对全链路性能的关键作用。与 Mem0、Zep 等基线对比，所提方案在同等模型规模下显著提升准确率。

**⚠️ 局限性**

局限性包括：① 仅评估稳态（已积累记忆）场景，冷启动阶段如何快速收敛未知；② 记忆采用原始对话回合对，若使用 LLM 生成摘要会导致幻觉污染；③ 对时间推理支持不足，需结构化时间索引；④ 置信度阈值对不同部署可能需要调优；⑤ 需要在真实生产流量下进一步验证鲁棒性。

---

## 386. RTS-ABAC: Real-Time Server-Aided Attribute-Based Authorization & Access Control for Substation Automation Systems

**arXiv ID:** 2603.23012 | [PDF](https://arxiv.org/pdf/2603.23012v1)

**作者:** Moritz Gstür `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4982 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一种实时服务器辅助属性基授权与访问控制（RTS-ABAC）方案，用以保障电站自动化系统（SAS）中实时通信的安全与完整性。

**💡 创新点**

创新点包括将实时属性和时序依赖的策略评估融入ABAC框架，并采用 bump‑in‑the‑wire（BITW）架构实现对SAS所有通信的强制认证、授权与访问控制，兼顾低延迟与安全需求。

**🔧 技术方法**

使用技术包括ABAC模型、实时属性与时间相关策略评估、服务器辅助安全模块、BITW实现，以及在GOOSE与SV协议环境下的部署和验证。

**📊 数据集**

使用数据集：无公开公开数据集，实验采用测试台和实验室环境中的实时通信数据，涵盖智能电子设备、合并单元与I/O盒子等SAS设备。

**📈 对比分析**

通过对比未加RTS-ABAC场景，进行性能评估与实验演示。结果显示，99.82% 的通信包在 6 ms 内完成往返时间（RTT），证明系统满足低延迟与安全双重要求。

**⚠️ 局限性**

限制：系统对服务器处理能力依赖强，部署成本与兼容性未在大规模实现场景下充分验证；对现有硬件改造的成本与兼容性分析不足。

---

## 387. Design Guidelines for Nonlinear Kalman Filters via Covariance Compensation

**arXiv ID:** 2603.22992 | [PDF](https://arxiv.org/pdf/2603.22992v1)

**作者:** Shida Jiang `[一作]` (University of California Berkeley), Scott Moura `[通讯]` (University of California Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过引入协方差补偿的概念，提出了一套新的非线性卡尔曼滤波设计框架，并给出了相应的改进准则。

**💡 创新点**

创新点在于：①将误差协方差补偿视为衡量滤波器鲁棒性的新指标；②证明了正半定（PSD）补偿矩阵与滤波器性能的直接关联；③提出了三条设计准则，包括正交不变性、充分补偿以及略微过度估计不确定性。

**🔧 技术方法**

使用的技术主要是非线性卡尔曼滤波器的理论分析、协方差补偿矩阵推导、数值仿真和对比实验，涵盖了 EKF2、SKF、CKF 与 SSKF 等变体。

**📊 数据集**

实验数据集包括三种应用：三维目标跟踪、地形参考导航以及同步发电机状态估计，每个系统都使用了对应的状态/测量噪声统计参数和多次仿真。

**📈 对比分析**

通过 10,000 次随机仿真，计算几何平均 RMSE 作为性能指标，并与传统固定 β 设置进行对比，结果表明遵循准则的滤波器在各案例中均能显著降低实际误差，且最佳 β 与传统值差距较大。

**⚠️ 局限性**

局限性包括：①尚未在更广泛的非线性系统和滤波器（如粒子滤波）上验证；②β 的选择仍需手动或离线调优；③实验覆盖的应用场景有限，可能不适用于极端非线性或高维问题。

---

## 388. SMSP: A Plug-and-Play Strategy of Multi-Scale Perception for MLLMs to Perceive Visual Illusions

**arXiv ID:** 2603.23118 | [PDF](https://arxiv.org/pdf/2603.23118v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 389. Agentic Verifier-in-the-Loop Solver Orchestration for Cell-Free Massive MIMO Downlink Power Control

**arXiv ID:** 2603.23128 | [PDF](https://arxiv.org/pdf/2603.23128v1)

**作者:** Zhichao Gao `[一作]` `[通讯]`, Zhichao Gao

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VISO‑PC 框架，将代理作为可信求解器的路由层，在 cell‑free 大规模 MIMO 下行功率控制中实现 Verifier‑in‑the‑loop 方案。

**💡 创新点**

创新点在于：① 代理仅负责求解器选路而非直接生成功率系数；② 采用独立验证器和 fallback 机制保证可行性；③ 轻量级内存检索实现快速、可解释的路由。

**🔧 技术方法**

使用多求解器组合（T_fast、T_exact、T_dist）、实例描述符与 Rule‑Router/Agent‑Router 的阈值/近邻检索技术，以及独立的 Verifier 模块。

**📊 数据集**

基于可复现的 cell‑free MIMO 下行功率控制基准，划分为 Train、Test、Stress、Shifted 四个子集进行实验。

**📈 对比分析**

对比方法包括 Always‑Fast、Always‑Exact、Always‑Dist、Rule‑Router、Agent‑Router；实验表明路由方法在接受率上优于单求解器，Agent‑Router 在 runtime 与 fallback 率上更优，且接受率与 Rule‑Router 一致。

**⚠️ 局限性**

局限在于仅研究单天线、固定 beamforming、fairness 目标，且检索在分布偏移下的鲁棒性有限，未扩展到更大规模或更复杂的目标。

---

## 390. Active Robotic Perception for Disease Detection and Mapping in Apple Trees

**arXiv ID:** 2603.23112 | [PDF](https://arxiv.org/pdf/2603.23112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 391. From Questions to Trust Reports: A LLM-IR Framework for the TREC 2025 DRAGUN Track

**arXiv ID:** 2603.23125 | [PDF](https://arxiv.org/pdf/2603.23125v1)

**作者:** Ignacy Alwasiak `[一作]` (Jagiellonian University), Udo Kruschwitz `[通讯]` (University of Regensburg)

**通讯引用:** 2602 | [OpenAlex ID](https://openalex.org/A5014534985)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于LLM与检索的端到端管线，用于生成关键性问题和可信度报告，帮助用户评估新闻可信度。

**💡 创新点**

创新点在于将Chain‑of‑Thought查询扩展、monoT5重排序以及域级可信度过滤相结合，形成了检索增强的可信度评估系统。

**🔧 技术方法**

使用的技术包括GPT‑4o‑Nano进行问题/答案生成，OpenSearch检索，CoT/Boolean/Structured查询扩展，monoT5重排序，LLM相关性评估以及域可信度数据集。

**📊 数据集**

使用的数据集包括MS MARCO V2.1（分段版）检索语料、Lin等的域可信度数据集以及DRAGUN评估所用的评判者问卷与评测数据。

**📈 对比分析**

通过与无扩展基线对比实验发现，CoT扩展加上monoT5重排序在相关性和域可信度上显著提升，报告支持度仍偏低，问题生成支持度亦不理想。

**⚠️ 局限性**

主要局限在于仅提交单次运行，缺少迭代修正与高级质量过滤，LLM评估不够稳健，问题生成与报告质量与评判者标准匹配不足。

---

## 392. Policy-based Tuning of Autoregressive Image Models with Instance- and Distribution-Level Rewards

**arXiv ID:** 2603.23086 | [PDF](https://arxiv.org/pdf/2603.23086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 393. AirSimAG: A High-Fidelity Simulation Platform for Air-Ground Collaborative Robotics

**arXiv ID:** 2603.23079 | [PDF](https://arxiv.org/pdf/2603.23079v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 394. Automatic Segmentation of 3D CT scans with SAM2 using a zero-shot approach

**arXiv ID:** 2603.23116 | [PDF](https://arxiv.org/pdf/2603.23116v1)

**作者:** Miquel Lopez Escoriza `[一作]`, Pau Amargant Alvarez `[通讯]` (EPFL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

在不进行任何微调的前提下，采用SAM2的零射击（zero‑shot）方法，对3D CT扫描进行自动骨结构分割，并通过仅在推理阶段做结构与流程改造来实现体素级的一致分割；

**💡 创新点**

创新点在于将SAM2的基于视频的记忆机制改造为适用于3D体积的顺序记忆，提出结构化提示选择（SPS）与智能切片（IS）策略，并实现多轴传播与融合，从而在零射击设置下显著提升了体积感知；

**🔧 技术方法**

采用的技术包括SAM2框架、推理时的记忆银行和时间嵌入修改、结构化提示筛选、智能切片选择、三轴传播与概率融合，以及系统的消融实验评估；

**📊 数据集**

使用了TotalSegmentator医学图像数据集，主要关注骨结构，在500份CT用于消融实验，2500份用于最终性能评估；

**📈 对比分析**

通过与基线（无预处理+First‑Middle‑Last提示）以及SPS、IS、IS+SPS等配置对比，最终在2,500份CT上实现Dice平均0.841、IoU0.778、Hausdorff距离4.79（相较基线提升约4%），证明零射击推理改造可取得可观性能；

**⚠️ 局限性**

局限性在于仍受SAM2原始2D/视频设计的约束，冻结权重导致对空间结构的完整建模有限，且在高复杂度骨结构上性能尚未达到专门微调模型的水平。

---

## 395. Compressing Dynamic Fully Indexable Dictionaries in Word-RAM

**arXiv ID:** 2603.23119 | [PDF](https://arxiv.org/pdf/2603.23119v1)

**作者:** Gabriel Marques Domingues `[一作]` (Tel Aviv University), Gabriel Marques Domingues `[通讯]` (Tel Aviv University)

**通讯引用:** 86 | [OpenAlex ID](https://openalex.org/A5008894666)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究了在Word-RAM模型中构建动态完全可索引字典（FID）的问题，旨在使用接近信息论下界的空间。

**💡 创新点**

提出了一种基于著名的融合树数据结构的动态FID，修改以使用更少的位并支持_0操作，这是第一个在标准Word-RAM模型中实现o(n√(w))位冗余的确定性动态FID。

**🔧 技术方法**

使用了Word-RAM模型，假设w≥lg u，并且整数乘法在O(1)时间内完成，采用了预计算表和树结构等技术。

**📊 数据集**

使用了动态字典和其他数据结构来支持操作，具体数据集未明确提及，但涉及的元素数量n为O(w^ε)的集合。

**📈 对比分析**

与现有方法进行比较，提出的FID在空间使用上为un+O(nw^ε)位，操作时间为O(1/ε + log_w(n))，在最坏情况下的时间复杂度为O(log_w(n))，性能优于之前的动态结构。

**⚠️ 局限性**

限制在于动态FID的构建和维护在内存模型中可能会遇到空间管理的挑战，尤其是在合并和拆分操作时可能导致空间使用超出预期。

---

## 396. MLLM-HWSI: A Multimodal Large Language Model for Hierarchical Whole Slide Image Understanding

**arXiv ID:** 2603.23067 | [PDF](https://arxiv.org/pdf/2603.23067v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 397. PHANTOM Hand

**arXiv ID:** 2603.23152 | [PDF](https://arxiv.org/pdf/2603.23152v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 398. DAK-UCB: Diversity-Aware Prompt Routing for LLMs and Generative Models

**arXiv ID:** 2603.23140 | [PDF](https://arxiv.org/pdf/2603.23140v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 399. Network Analysis of the Egyptian Reddit Community

**arXiv ID:** 2603.23107 | [PDF](https://arxiv.org/pdf/2603.23107v1)

**作者:** Samy Shaawat `[一作]` (Egypt Japan University Of Science And Technology), Walid Gomaa `[通讯]` (Egypt Japan University Of Science And Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

收集并构建了包含23,185名用户和105个埃及主题子版块的数据集，并通过无向图构建用户共享子版块网络进行分析。

**💡 创新点**

首次针对埃及Reddit社区进行全网结构与动态社区发现，揭示其小世界特性与高度聚类。

**🔧 技术方法**

使用Python、PRAW、NetworkX和Matplotlib进行网络构建与分析，结合度中心性、聚类系数、静态/动态社区检测等指标。

**📊 数据集**

基于Reddit公开API抓取的埃及相关子版块用户数据，形成23,185个节点、6,877,773条边的网络。

**📈 对比分析**

通过与传统随机网络、随机分布等基准比较，显示该网络呈规模自由小世界特征，平均度为593，聚类系数0.976，密度0.0256；性能在大规模无向图上完成。

**⚠️ 局限性**

研究仅基于静态数据，缺乏时间序列与情感分析，且未验证社区发现算法的鲁棒性，结论易受采样偏差影响。

---

## 400. Good for the Planet, Bad for Me? Intended and Unintended Consequences of AI Energy Consumption Disclosure

**arXiv ID:** 2603.23075 | [PDF](https://arxiv.org/pdf/2603.23075v1)

**作者:** Michael Klesel `[一作]` (Frankfurt University of Applied Sciences), Uwe Messer `[通讯]` (Universität der Bundeswehr München)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5072006959)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

设计并执行一项在线实验，探究向用户披露AI语言模型能耗信息（ECD）是否能促使其更倾向于选择能源更高效的小型模型（SLM），以及这种选择对后续使用行为和主观感知的影响。

**💡 创新点**

首次将能耗标签引入语言模型选择情境，系统性检验ECD在可持续人机交互中的效能；发现ECD对模型选择具有超强的nudge效应，并揭示可持续选择伴随的负面感知（满意度和质量下降）——这两点为本研究的创新亮点。

**🔧 技术方法**

采用实验设计、随机分组（控制组仅展示性能星级；处理组同时展示能耗标签）、统计分析方法（χ²检验、二元逻辑回归、Mann‑Whitney U检验），并使用伪真实模型（所有模型均基于GPT‑4o mini，但被标记为LLM/SLM）来测试置换效应。

**📊 数据集**

使用从Prolific平台招募的365名英国英语母语的云工作者作为实验样本；实际模型均为GPT‑4o mini，实验中通过性能星级（5/5 vs 3/5）和能耗标签（A–G）进行对照。

**📈 对比分析**

通过对照组与处理组的比较，使用Odds Ratio（OR ≈ 12.89）评估ECD对模型选择的影响；后续使用行为（提示数量、平均Token数）无显著差异，但满意度和质量在处理组中显著下降；整体表现主要体现在用户选择行为和主观评价上，而非模型技术性能指标。

**⚠️ 局限性**

局限性包括：单一低风险任务、仅对英国受试者，缺乏跨文化验证；实验使用单一模型导致难以区分性能与能耗信息的独立效应；短期实验未检视长期或实际使用中的伦理效应；未探索动态路由或默认设置等更高级的交互设计；未对水足迹等更全面的环境指标进行考量。

---

## 401. 3rd Place of MeViS-Audio Track of the 5th PVUW: VIRST-Audio

**arXiv ID:** 2603.23126 | [PDF](https://arxiv.org/pdf/2603.23126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 402. Can an LLM Detect Instances of Microservice Infrastructure Patterns?

**arXiv ID:** 2603.23073 | [PDF](https://arxiv.org/pdf/2603.23073v1)

**作者:** Carlos Eduardo Duarte `[一作]` (INESC TEC), Pavlína Gonçalves `[通讯]` (INESC TEC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了MicroPAD工具，利用LLM检测多语言软件仓库中的微服务架构模式，并构建了190个带人工标注的仓库数据集。

**💡 创新点**

提供语言无关的LLM检测框架，仅用自然语言描述即可识别多语言模式；首创基于文件主导指数(FDI)评估文件对检测的影响；发布首个公开人标注的微服务模式数据集。

**🔧 技术方法**

GPT‑5 nano LLM、Python流水线、ChromaDB向量数据库、自然语言提示、文件优先级与多步骤推理。

**📊 数据集**

190个GitHub仓库，包含47个Richardson模式的人工标注数据，约20,000+ GitHub API收集的代码。

**📈 对比分析**

与贡献者问卷标注对比，计算精确率、召回率、准确率与F1；整体准确率79.5%，平均F1 39.5%，按模式F1从0.09到0.70变化，模式出现频率与文件显著性正相关。

**⚠️ 局限性**

仅评估GPT‑5 nano；文件选择与模式示例有限导致低F1；模型非确定性；仅覆盖9个基础设施模式；实验数据以公开仓库为主，工业仓库可能不同。

---

## 403. Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair

**arXiv ID:** 2603.23129 | [PDF](https://arxiv.org/pdf/2603.23129v1)

**作者:** Aditya Kakade `[一作]` (TCS Research), Shirish Karande `[通讯]` (TCS Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Polaris 框架，利用经验抽象实现小型语言模型的递归自我修复；

**💡 创新点**

创新在于将错误分析、策略合成、最小化补丁生成和验证等步骤结构化，保持可追踪、可审计的代码级更新，并通过经验抽象压缩上下文，适配资源受限环境；

**🔧 技术方法**

使用 Gödel Agent 递归自我改进思想，结合 LLM 生成、工具调用、Runtime 代码变异、JSON 结构化输出、轻量级验证器与循环迭代；

**📊 数据集**

在 MGSM、DROP、GPQA 和 LitBench 四大基准上进行实验；

**📈 对比分析**

与 Chain‑of‑Thought Self‑Consistency (COT‑SC) 基线以及原始 Gödel Agent 进行对比，Polaris 在成功运行中实现了 3.6%–9.0% 的相对提升；

**⚠️ 局限性**

主要局限是元推理能力有限导致修复噪声、工具调用不稳定及 OOM 失败，以及对较大上下文任务（如 DROP）适应性不足。

---

## 404. Q-GARS: Quantum-inspired Robust Microservice Chaining Scheduling

**arXiv ID:** 2603.23127 | [PDF](https://arxiv.org/pdf/2603.23127v1)

**作者:** Huixiang Zhang `[一作]` (Lakehead University), Mahzabeen Emu `[通讯]` (Memorial University)

**通讯引用:** 139 | [OpenAlex ID](https://openalex.org/A5040609918)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种混合量子启发式与自适应鲁棒的微服务调度框架 Q-GARS，用以在云边连续体中减少首排阻塞并降低尾部延迟。

**💡 创新点**

创新点包括：①将微服务优先级排序转化为 QUBO 模型并利用 SQA 进行快速组合搜索；②通过 NUM 生成连续资源分配，二者分离实现低延迟决策；③引入指数权重自适应机制（Hedge）在量子优先级失效时自动切换到鲁棒基线；④动态计算 QUBO 罚项，保证在毫秒级预算内可解。

**🔧 技术方法**

主要技术：Quadratic Unconstrained Binary Optimization (QUBO)、模拟量子退火 (SQA)、网络效用最大化 (NUM)、指数加权在线学习 (Hedge)、离散事件模拟。

**📊 数据集**

使用合成有向无环图 (DAG) 数据集，节点数 10–50、并发宽度 2–8；通过 Monte‑Carlo 方式生成数万条工作流用于统计评估。

**📈 对比分析**

与传统 SRPT 贪婪策略以及稳健基线进行对比；Q-GARS 在平均加权完成时间上提升 2.1%，峰值提升可达 16.8%；尾部（95th 分位）队列积压比基线低 20%–30%；整体资源利用率提升 1.1%。

**⚠️ 局限性**

局限性：目前仅在模拟环境中验证，未部署真实量子硬件；QUBO 求解器为本地模拟，无法反映量子设备的噪声和连接性限制；对极端多任务和大规模 DAG 的可扩展性仍待进一步评估。

---

## 405. Towards a Unified Coding Scheme for 6G

**arXiv ID:** 2603.23123 | [PDF](https://arxiv.org/pdf/2603.23123v1)

**作者:** Paul Bezner `[一作]` (University of Stuttgart), Stephan ten Brink `[通讯]` (University of Stuttgart)

**通讯引用:** 17348 | [OpenAlex ID](https://openalex.org/A5034116116)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了6G统一编码方案的设计思路与评估框架，比较了Polar码与LDPC码在性能、灵活性与硬件实现方面的优势与不足，旨在为未来6G提供可扩展、低能耗的通用编码方案。

**💡 创新点**

提出“统一编码”理念，将不同场景下的码率、码长、低延迟与能效需求统一至同一码族，并将编码与解码结构协同优化；同时梳理并整合了Polar码的预变换、集合/列表译码、自动化对称等技术与LDPC码的空间耦合、窗口译码、自动化集合译码等方法，构成多层次可调的硬件架构。

**🔧 技术方法**

采用Polar码的预变换（CRC/重复码）、SCL/ AED集合译码、SSC简化算法、嵌套可靠性序列；LDPC码则使用QC/空间耦合结构、窗口BP译码、层级/调度集合译码、PBRL码率匹配等技术；并结合硬件友好的量化、层级解码、自动化对称等实现细节。

**📊 数据集**

论文主要通过仿真误码率曲线（FER/BER）与元对角界（meta-converse bound）进行评估，实验覆盖短码长（N=256）、长码长（N=65536）及不同码率（R=1/2、R=8/9），并与5G标准LDPC、Polar以及DVB‑S2/5G基准进行对比。

**📈 对比分析**

比较方法为在相同SNR下绘制FER/BER曲线，观察不同码长/码率下的误码性能；结果显示在短码长下Polar+SCL/ AED可匹敌或优于5G LDPC；在长码长下LDPC（尤其是SC‑LDPC窗口译码）接近或超越Polar；两者在低迭代/低复杂度场景下仍存在性能差距，但整体性能均靠近元对角界。

**⚠️ 局限性**

局限性包括：Polar码在长码长下性能衰减、列表/集合译码带来控制流与面积开销；LDPC码在短码长/高SNR下易出现错误坡度与陷阱集；两种码族的硬件实现仍需进一步统一、优化；此外，目前的评估多为仿真，缺乏真实硬件验证与功耗/延迟的定量指标。

---

## 406. TRAP: Hijacking VLA CoT-Reasoning via Adversarial Patches

**arXiv ID:** 2603.23117 | [PDF](https://arxiv.org/pdf/2603.23117v1)

**作者:** Zhengxian Huang `[一作]` (Zhejiang University), Wenyuan Xu `[通讯]` (Zhejiang University)

**通讯引用:** 7122 | [OpenAlex ID](https://openalex.org/A5060351020)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了一种利用 Chain‑of‑Thought 语义推理被攻击者操控的定向对抗补丁攻击 TRAP；

**💡 创新点**

首次发现 CoT 机制为攻击提供新向量，并提出通过对抗补丁直接篡改 CoT 推断实现定向控制；

**🔧 技术方法**

使用对抗补丁生成结合 CoT 与动作损失、PGD 优化、EoT、颜色校准及投影几何等技术实现物理可行；

**📊 数据集**

在 RT‑1、LIBERO 等公开机器人数据集上训练 VLA，并在自制模拟数据与真实实验中评估；

**📈 对比分析**

与随机噪声、仅动作攻击及仅 CoT 攻击对比，TRAP 在 InstructVLA、GraspVLA 上 ASR 超过 50%，在真实场景中成功率达 86.7%；

**⚠️ 局限性**

仅针对 pick‑and‑place 等短期任务，未验证长周期任务；缺乏对抗补丁隐蔽性与实用性的进一步优化。

---

## 407. NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization

**arXiv ID:** 2603.23104 | [PDF](https://arxiv.org/pdf/2603.23104v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 408. Mind Your HEARTBEAT! Claw Background Execution Inherently Enables Silent Memory Pollution

**arXiv ID:** 2603.23064 | [PDF](https://arxiv.org/pdf/2603.23064v1)

**作者:** Yechao Zhang `[一作]` (Nanyang Technological University), Tianwei Zhang `[通讯]` (Nanyang Technological University)

**通讯引用:** 2897 | [OpenAlex ID](https://openalex.org/A5028270700)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 Claw 系统中心跳共享会话导致的无意识记忆污染，验证了 Exposure→Memory→Behavior (E→M→B) 攻击路径的可行性，并通过三种实验（同一会话、跨会话、噪声环境）量化了污染对代理行为的影响。

**💡 创新点**

创新点在于：①首次揭示并系统化心跳后台执行的零点击记忆污染漏洞；②构建可控实验平台 MissClaw（MolBook 的克隆）以模拟真实社交环境；③从社交可信度、代理人格、外部搜索等维度评估并量化其对污染效果的作用。

**🔧 技术方法**

使用技术包括：OpenClaw 框架、Claude Haiku 4.5 语言模型、MissClaw 受控实验平台、LLM 评判器、三类任务（软件安全、金融决策、学术引用）以及与之配套的工具与接口。

**📊 数据集**

数据集主要由人工合成的社交平台帖子和评论组成，满足真实格式；任务场景数据包括真实软件依赖包、虚构 DeFi 协议和虚假学术论文的元数据；所有数据均在实验平台内隔离生成。

**📈 对比分析**

对比方法：在三种实验设置下统计攻击成功率（ASR）和存储率，探讨不同社交线索、人格设定与是否启用外部搜索的影响。结果显示，在强社交线索下 ASR 可达 60%+，在自然噪声环境下降至 10%~30%，但仍非零；人格和搜索对低危害场景有一定抑制作用。

**⚠️ 局限性**

Limitations: 仅使用单一 LLM 模型；实验平台与真实 MolBook 在并发性、社区治理和动态社交行为上有限；受限于三类任务域，未覆盖医疗、法律等高风险场景；LLM 评判器可能出现误判；未评估更高级对抗手段或动态自适应攻击；未给出完整的防御实现。

---

## 409. Conformal Cross-Modal Active Learning

**arXiv ID:** 2603.23159 | [PDF](https://arxiv.org/pdf/2603.23159v1)

**作者:** Huy Hoang Nguyen `[一作]` (AIT Austrian Institute of Technology), Andreas Kugi `[通讯]` (AIT Austrian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于预训练视觉‑语言模型的教师‑学生框架，通过分布无关的共形校准来获取跨模态的不确定性，并以此驱动主动学习的样本选择。

**💡 创新点**

创新点包括：①将VLM作为教师生成语义对齐的预测集合；②采用分割式共形校准实现教师与学生的不确定性无偏估计；③将跨模态不确定性与多样性覆盖的采样策略结合，并自动平衡教师与学生的置信度。

**🔧 技术方法**

使用的技术包括共形预测、教师‑学生架构、CLIP文本原型、DINOv2视觉特征、Jensen‑Shannon 散度、覆盖度驱动的多样性选择、k‑means 子池等。

**📊 数据集**

实验数据集涵盖 CIFAR‑100、Food‑101、DomainNet‑Real、Caltech‑101、Caltech‑256 等图像分类基准。

**📈 对比分析**

与随机、Entropy、Margin、BALD、pBALD、BADGE、Coreset、Typiclust、ProbCover 等 11 种基线对比，CCMA 在所有数据集的早期和后期均取得最佳或相近的准确率，尤其在低预算下准确率提升约 10%。

**⚠️ 局限性**

局限性包括：当教师与学生不一致度快速消失（如 Caltech 数据）时跨模态优势减弱；对教师温度敏感；子池聚类与多样性步骤会增加查询时间；目前仅验证在静态分类任务，尚未扩展到开放类或跨域迁移。

---

## 410. Can Language Models Pass Software Testing Certification Exams? a case study

**arXiv ID:** 2603.23142 | [PDF](https://arxiv.org/pdf/2603.23142v1)

**作者:** Fitash Ul Haq `[一作]` (Luxembourg Institute of Science and Technology), Jordi Cabot `[通讯]` (Luxembourg Institute of Science and Technology)

**通讯引用:** 8799 | [OpenAlex ID](https://openalex.org/A5074872542)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了60个多模态大型语言模型在30份ISTQB软件测试认证考试（共1171题）中的表现，并验证其对测试知识的掌握与鲁棒性。

**💡 创新点**

创新地将正式认证考试与元变换（metamorphic transformations）相结合，构建公开排行榜和可复现实验包，检验模型的概念理解而非仅仅记忆答案。

**🔧 技术方法**

采用提示工程生成问答提示，对PDF考试文件进行信息提取与人工校验，调用各模型API或本地推理，统计得分并按难度层级和元变换进行分析。

**📊 数据集**

使用30份ISTQB公开样本考试（涵盖K1–K4六级题型及图像题）以及通过语义保持变换生成的515道变形题作为鲁棒性测试集。

**📈 对比分析**

通过对各模型在全部考试、各难度层级及变形题上的得分与正确率进行比较，结果显示Gemini‑3‑flash‑preview与GPT‑5‑2025‑08‑07在所有考试中均超过65%，商业模型整体优于开源模型，鲁棒性略有下降。

**⚠️ 局限性**

局限性包括模型训练集可能已包含部分考试题、PDF提取与手工校验的误差、对图像题处理不足、模型易产生幻觉或不遵循指令，以及仅评估ISTQB一套考试，未覆盖其他测试领域。

---

## 411. Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models

**arXiv ID:** 2603.23149 | [PDF](https://arxiv.org/pdf/2603.23149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 412. Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy

**arXiv ID:** 2603.23146 | [PDF](https://arxiv.org/pdf/2603.23146v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 413. A Bayesian Learning Approach for Drone Coverage Network: A Case Study on Cardiac Arrest in Scotland

**arXiv ID:** 2603.23134 | [PDF](https://arxiv.org/pdf/2603.23134v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 414. InterDyad: Interactive Dyadic Speech-to-Video Generation by Querying Intermediate Visual Guidance

**arXiv ID:** 2603.23132 | [PDF](https://arxiv.org/pdf/2603.23132v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 415. MsFormer: Enabling Robust Predictive Maintenance Services for Industrial Devices

**arXiv ID:** 2603.23076 | [PDF](https://arxiv.org/pdf/2603.23076v1)

**作者:** Jiahui Zhou `[一作]` (Sun Yat-Sen University), See-Kiong Ng `[通讯]` (National University of Singapore)

**通讯引用:** 5400 | [OpenAlex ID](https://openalex.org/A5090171111)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了MsFormer——一种轻量级多尺度Transformer，用于工业设备的预测性维护服务，利用多尺度采样、多尺度位置编码和轻量化注意力实现对多尺度时序相关性的建模。

**💡 创新点**

创新点在于①多尺度采样模块将原始时序拆分为不同分辨率子序列，捕获稀疏语义下的多尺度依赖；②多尺度位置编码结合下采样因子，对自注意力的相对位置信息进行动态调整；③轻量化注意力用池化代替全自注意力，降低参数量并适配数据稀缺场景。

**🔧 技术方法**

主要技术包括Transformer框架、窗口反向、滑动窗口预处理、池化替代注意力、可学习的多尺度位置编码及多阶段分层结构。

**📊 数据集**

实验使用了C‑MAPSS航空发动机四个子数据集（FD001/FD002/FD003/FD004）以及NASA锂离子电池四个子集（B5/B6/B7/B18），通过留一交叉验证评估。

**📈 对比分析**

与CNN、RNN、FEDformer、Autoformer、DVGTformer、MLEAN、NSD‑TGTN等多种基线相比，MsFormer在所有子集上RMSE与Score均优于基线，尤其在复杂的FD002/FD004和NASA电池数据上显著提升，证明其在不同设备和工况下的泛化能力与高效性。

**⚠️ 局限性**

局限性包括对多尺度采样因子和位置编码阈值的手工调参需求、在极小样本情况下提升有限、以及对实时推理延迟的进一步优化仍需研究。

---

## 416. AgentFoX: LLM Agent-Guided Fusion with eXplainability for AI-Generated Image Detection

**arXiv ID:** 2603.23115 | [PDF](https://arxiv.org/pdf/2603.23115v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 417. Between Rules and Reality: On the Context Sensitivity of LLM Moral Judgment

**arXiv ID:** 2603.23114 | [PDF](https://arxiv.org/pdf/2603.23114v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 418. Machine Learning Models for the Early Detection of Burnout in Software Engineering: a Systematic Literature Review

**arXiv ID:** 2603.23063 | [PDF](https://arxiv.org/pdf/2603.23063v1)

**作者:** Tien Rahayu Tulili `[一作]` (University of Groningen), Andrea Capiluppi `[通讯]` (University of Groningen)

**通讯引用:** 1951 | [OpenAlex ID](https://openalex.org/A5077760743)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过系统文献综述，对软件工程领域内使用机器学习技术实现倦怠早期检测的研究进行整理，收集了64篇论文，分析其方法、输入特征、使用的数据集以及模型性能。

**💡 创新点**

创新点在于将情绪、压力、毒性关系以及离职预测等多维度指标统一纳入研究框架，系统评估不同机器学习模型在倦怠检测中的表现，并提出最佳模型与数据集组合的实践建议。

**🔧 技术方法**

综述涵盖了传统机器学习算法（SVM、RF、NB、DT等）、集成学习与深度学习方法（CNN、LSTM、BERT、Transformer等）以及情感分析工具（SentiStrength、LIWC、Perspective API等）等技术。

**📊 数据集**

使用的主要数据集包括开发者沟通记录（Jira, GitHub Issues, Stack Overflow, Gitter, Slack, Code Review, Bugzilla等）、生理传感器数据（皮肤电、心率、肌电等）以及多模态情感数据集（EmotionStimulus、ISEAR、DailyDialog 等）。

**📈 对比分析**

通过比较精度、召回率、F1 分数和准确率等指标，发现文本基 Transformer 模型在情绪检测上表现最好；贝叶斯模型在传感器输入下效果突出；决策树/随机森林在离职预测中优越；神经网络与 Transformer 在毒性检测中表现最佳，整体准确率普遍高于 0.8。

**⚠️ 局限性**

局限性包括依赖公开文献和已有数据，缺乏统一的倦怠评估标准和标注；大部分数据为离线文本，缺乏实时多模态验证；样本规模和多样性有限，且对性别、年龄等人口学特征考虑不足，可能导致模型偏差。

---

## 419. When Language Models Lose Their Mind: The Consequences of Brain Misalignment

**arXiv ID:** 2603.23091 | [PDF](https://arxiv.org/pdf/2603.23091v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 420. A Synchronized Audio-Visual Multi-View Capture System

**arXiv ID:** 2603.23089 | [PDF](https://arxiv.org/pdf/2603.23089v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 421. A Systematic Framework for Stable and Cost-Efficient Matrix Polynomial Evaluation

**arXiv ID:** 2603.23143 | [PDF](https://arxiv.org/pdf/2603.23143v1)

**作者:** J. M. Alonso `[一作]`, E. Defez `[通讯]` (Universitat Politecnica De Valencia)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

构建了一套系统化框架和 MATLAB 工具，用来在评估矩阵多项式时比 Paterson–Stockmeyer (PS) 方法多消耗一个矩阵乘法（1M），并自动挑选数值稳定的系数组。

**💡 创新点**

创新点在于：①将多项式评估问题归约为求解一元多项式根，从而能用变精度符号计算同时搜索所有解；②引入多种结构变体与稳定性指标（结合相对与绝对误差）以克服传统方法中出现的不稳定解；③在 MATLAB 中实现完整的自动化流程，并对多种实际多项式（如矩阵指数、几何级数）验证其效果。

**🔧 技术方法**

主要技术包括：变精度算术（VPA）+符号求解（Matlab Symbolic Toolbox），多项式系数重构与误差评估，结构化多项式分块（y_0s, y_1s, z_1ps 等）以及针对特殊情况（如 c_3=0）设计的特殊解法。

**📊 数据集**

使用的测试数据集：随机生成的 100×100、1000×1000、10000×10000 矩阵；Matrix Computation Toolbox (MCT) 与 Eigtool MATLAB Package (EMP) 的实测矩阵；以及多项式的泰勒级数系数（如 exp 的 8、10、12、…、81 次 Taylor 近似）。

**📈 对比分析**

与传统 PS 方法以及 Westreich 等专用算法比较。实验显示：①在 1M 节省的同时，误差与 PS 方法几乎相同，通常低于 10u；②在几何级数、矩阵指数等实际应用中，误差多在 1–5×10⁻¹⁶ 左右；③对大尺寸矩阵（n=10 000）时，误差保持在 10⁻¹⁶ 左右，性能优于或等同于 PS。

**⚠️ 局限性**

局限性包括：①仅适用于阶数 ≥8、首项非零的多项式；②对特殊阶数（如 9、11）无法实现 1M 节省；③求解过程依赖符号计算，离线预处理成本较高；④若所有结构变体下都无稳定解，则工具仅给出警告，需人工干预。

---

## 422. PiCo: Active Manifold Canonicalization for Robust Robotic Visual Anomaly Detection

**arXiv ID:** 2603.23122 | [PDF](https://arxiv.org/pdf/2603.23122v1)

**作者:** Teng Yan `[一作]` (Hong Kong University of Science and Technology), Bingzhuo Zhong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 82 | [OpenAlex ID](https://openalex.org/A5001005006)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 PiCo 框架，结合主动物理对齐与多阶段神经对齐，实现了机器人视觉异常检测的自适应规范化。

**💡 创新点**

将主动感知与条件信息瓶颈相结合，构建了三阶段（光照、潜在空间、语义）分层对齐管线，并通过姿态反馈实现物理层的规范化。

**🔧 技术方法**

使用光照预处理（Retinex 形式）、双通道频谱瓶颈 MLP、线性注意力 LA_3、条件信息瓶颈以及主动姿态优化等技术。

**📊 数据集**

在大规模 M^2AD 静态 benchmark 与新建 PiCo-Bench 机器人闭环动态数据集上进行评测。

**📈 对比分析**

与 Dinomaly、RD++、MSFlow、INP-Former 等最先进无监督 VAD 进行对比，在 M^2AD 上平均 O‑AUROC 93.7%（比基线提升 3.7%），在闭环机器人任务上准确率 98.5%。

**⚠️ 局限性**

对实时推理速度与多物体场景的扩展性有限，且对极端光照与复杂遮挡的泛化仍需进一步验证。

---

## 423. AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing

**arXiv ID:** 2603.23069 | [PDF](https://arxiv.org/pdf/2603.23069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 424. VoDaSuRe: A Large-Scale Dataset Revealing Domain Shift in Volumetric Super-Resolution

**arXiv ID:** 2603.23153 | [PDF](https://arxiv.org/pdf/2603.23153v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 425. SpecXMaster Technical Report

**arXiv ID:** 2603.23101 | [PDF](https://arxiv.org/pdf/2603.23101v1)

**作者:** Yutang Ge `[一作]` (Bohrium), Zhifeng Gao `[通讯]` (Bohrium)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出 SpecXMaster，一种基于 Agentic RL 的从原始 FID 数据到完整 NMR 结构解释的端到端框架。

**💡 创新点**

创新点在于：①直接处理原始 FID，消除峰表格中信息损失；②采用多工具环境与多步骤 Agentic 交互，实现类似专家推理；③在多模态（1D 1H/13C）上加入 hyperbolic 表征的 hard‑case 处理。

**🔧 技术方法**

技术包括：FID 信号处理（窗函数、零填充、FFT、相位/基线校正）、峰检索与神经网络多重分类、基于 LLM 的 Agent 训练、Agentic RL (GRPO) 与多工具接口（Generate, Search, Repair, Rerank），以及 hyperbolic 表征学习。

**📊 数据集**

使用 NMRexp 公开数据集（约 100 万条 1H/13C 组合谱），按训练/验证/测试拆分。

**📈 对比分析**

与单一步骤生成模型以及两种通用 LLM 的工作流做对比，SpecXMaster 在 hit@1、hit@3、hit@5 上均显著提升（例如 Joint 模式 hit@1 从 0.639 提升到 0.702）。

**⚠️ 局限性**

局限性：目前仅支持 1D NMR，未覆盖 2D/多模态谱；对低 SNR 或极度重叠的谱仍易失败；需要大量预训练模型和计算资源；缺乏对混合物定量分析的支持。

---

## 426. Symbol-Synchronous Communication for Ultra-Low-Power Multi-Hop Ambient IoT Networks

**arXiv ID:** 2603.23084 | [PDF](https://arxiv.org/pdf/2603.23084v1)

**作者:** Xinlei Liu `[一作]` (University of Antwerp), Jeroen Famaey `[通讯]` (University of Antwerp)

**通讯引用:** 4136 | [OpenAlex ID](https://openalex.org/A5047378679)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出了一种基于符号同步传输的多跳低功耗 A‑IoT 网络协议，采用随机唤醒机制降低节点能耗，并通过仿真验证其可靠性与能耗优势。

**💡 创新点**

创新点在于：①允许间歇供电的 A‑IoT 节点以自适应概率参与符号级转发，消除了传统多跳网络中的时间同步、路由与通道访问开销；②利用 POOK 短脉冲 OOK 调制与构造性干涉实现低功耗符号同步；③通过窗口检测与投票决策进一步节能。

**🔧 技术方法**

使用技术包括：符号同步传输、POOK 调制、窗口检测+投票决策、随机唤醒概率 P、MATLAB 仿真、TGac 通道模型、IEEE 802.15.4 CC2538 能耗参数、BCH 纠错编码。

**📊 数据集**

使用的数据集为仿真生成的三种网格网络（25、100、400 节点），每种网络随机选取源节点，发送 20000 个 128 位数据包；未采用公开的真实数据集。

**📈 对比分析**

性能评估通过 PER（packet error rate）和 PrLR（preamble loss rate）衡量可靠性，并与传统持续开启基准对比；在 400 节点网络下，PER < 10⁻² 时能耗下降约 88 %，而在较稀疏网络中仍能维持可靠传输。

**⚠️ 局限性**

局限性：仅在静态网格拓扑和固定传输功率下评估，未考虑能量采集波动、动态拓扑变化及不同唤醒策略；同步窗口随机导致潜在误检测；缺乏硬件实验验证；仅探讨了单一 POOK 调制，未评估更高速率调制方案。

---

## 427. Generalization Bounds for Physics-Informed Neural Networks for the Incompressible Navier-Stokes Equations

**arXiv ID:** 2603.23072 | [PDF](https://arxiv.org/pdf/2603.23072v1)

**作者:** Sebastien Andre-Sloan `[一作]` (University of Manchester), Anirbit Mukherjee `[通讯]` (University of Manchester)

**通讯引用:** 281 | [OpenAlex ID](https://openalex.org/A5084835559)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在该论文中，作者为物理信息神经网络（PINN）在求解(d+1)维不可压Navier–Stokes方程时提供了严格的上界泛化误差估计，并通过实验验证了这些理论界限。

**💡 创新点**

创新点在于：①首次为非线性PDE（Navier–Stokes）推导Rademacher复杂度相关的泛化误差上界；②该上界与网络宽度无关，且对流体粘度与损失正则化参数有明确依赖；③提出了一组新的激活函数（如tanh³）并证明其在理论上更有利；④提供了实验数据验证理论与实测误差之间的高相关性。

**🔧 技术方法**

主要技术手段包括：Rademacher复杂度分析、Talagrand收缩引理、对PINN残差项的分解与多层收缩处理、以及对深度为2的全连接网络的权重约束。

**📊 数据集**

实验使用了Taylor–Green涡旋基准问题的解析解作为“真值”，通过随机采样得到内部与初始/边界的采样点，训练了基于tanh和tanh³激活函数的PINN。

**📈 对比分析**

方法比较：作者将理论界限与训练得到的泛化误差（即训练损失与期望损失之差）进行对比。实验表明，在不同粘度和采样点数下，理论上界与实测误差呈现很高的线性相关（相关系数≈0.90），验证了理论估计的有效性；相比之下，使用普通tanh激活函数时相关性略低。

**⚠️ 局限性**

局限性包括：①上界仅为PINN风险的泛化误差，未直接控制模型输出与真实PDE解的距离；②分析仅适用于深度为2的网络，未涵盖更深网络结构；③未给出关于网络规模的下界，无法评估近似能力；④对Transformer等新型PINN架构缺乏理论支持。

---

## 428. HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature

**arXiv ID:** 2603.23136 | [PDF](https://arxiv.org/pdf/2603.23136v1)

**作者:** Devvrat Joshi `[一作]` (Imperial College London), Islem Rekik `[通讯]` (Imperial College London)

**通讯引用:** 3859 | [OpenAlex ID](https://openalex.org/A5048784346)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出两阶段零样本科学知识图谱构建框架 Z-NERD 与 HGNet，解决长实体识别、跨域泛化和层级关系抽取等难题。

**💡 创新点**

创新点包括 Orthogonal Semantic Decomposition (OSD) 与 Multi-Scale TCQK 注意力实现多词实体识别；HGNet 的三通道层级图网络、可微层级损失与连续抽象场 (CAF) 损失，将层级抽象映射至欧氏空间；以及新发布的多域层级关系基准 SPHERE。

**🔧 技术方法**

技术上采用 SciBERT 共享编码器，结合 OSD、Multi-Scale TCQK、概率层级图网络、三通道消息传递、可微层级损失和 CAF 损失。

**📊 数据集**

使用数据集包括 SciERC、SciER、BioRED、SemEval-2017 Task 10 以及新构建的 SPHERE（覆盖计算机科学、物理、生物、材料四个领域）。

**📈 对比分析**

与 SciBERT、PL-Marker、HGERE、UniversalNER-7b、LLM 等多种基线对比，Z-NERD 在 NER 上平均提升 8.08%（零样本提升 10.76%），HGNet 在 RE 上平均提升 5.99%（零样本提升 26.2%），在 SPHERE 上比 HGERE 提升约 20% 以上。

**⚠️ 局限性**

局限性在于仍需依赖预训练 SciBERT，跨域性能受限于 OSD 的语义转折识别；模型规模约 3 亿参数，未覆盖多模态信息；抽象轴的解释性与可解释性仍有提升空间。

---

## 429. PolarAPP: Beyond Polarization Demosaicking for Polarimetric Applications

**arXiv ID:** 2603.23071 | [PDF](https://arxiv.org/pdf/2603.23071v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 430. Fault-Tolerant Design and Multi-Objective Model Checking for Real-Time Deep Reinforcement Learning Systems

**arXiv ID:** 2603.23113 | [PDF](https://arxiv.org/pdf/2603.23113v1)

**作者:** Guoxin Su `[一作]` (University of Wollongong), David S. Rosenblum `[通讯]` (George Mason University)

**通讯引用:** 11174 | [OpenAlex ID](https://openalex.org/A5047104641)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一种基于时钟自动机的实时 DRL 系统切换设计框架，并将其语法转换为马尔可夫决策过程以实现多目标模型检验；

**💡 创新点**

创新点在于提出了“凸查询”方法，用于在保证硬约束的同时最小化软约束的偏差，并实现了 GPU 加速的多目标 MDP 检测器 MOPMC；

**🔧 技术方法**

核心技术包括时钟自动机建模、语法级 MDP 转换、基于价值迭代的凸优化、以及 CUDA 并行加速；

**📊 数据集**

评估使用了 Gymnasium HighwayEnv 的仿真环境，并以 DDPG 训练的 DRL 代理和基于规则的后备控制器为基础进行实验；

**📈 对比分析**

通过与 Storm、PRISM 等现有工具比较，MOPMC 在大型模型与高维目标数下实现了 20–30% 的速度提升，并在切换策略上显著降低了延迟引起的安全风险；

**⚠️ 局限性**

局限性包括对模型概率参数的手工估计、仅针对总奖励的凸查询（未覆盖平均收益或长期平均成本等其他目标）以及在极端分布或高度非马尔可夫环境下的适用性尚未验证。

---

## 431. Spatial Analysis on Value-Based Quadtrees of Rasterized Vector Data

**arXiv ID:** 2603.23105 | [PDF](https://arxiv.org/pdf/2603.23105v1)

**作者:** Diana Baumann `[一作]` (Technical University Berlin), David Bermbach `[通讯]` (Technical University Berlin)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一种基于值的四叉树索引，将向量和栅格数据统一栅格化并压缩，实现联合空间分析

**💡 创新点**

通过利用空间自相关特性，对栅格化后的数据做值聚合四叉树压缩，既支持向量也支持栅格的交叉查询，突破了传统工具只能单一格式的局限

**🔧 技术方法**

四叉树索引、栅格化、空间自相关压缩、Pip查询、R语言与相关地理数据库库（如PostGIS、MobilityDB）

**📊 数据集**

柏林自行车轨迹数据（SimRa）和柏林公园多边形数据（Osm）

**📈 对比分析**

与原始向量、单独栅格三种格式在Pip查询上比较，实验显示四叉树索引平均查询延迟降低约90%，并保持与向量结果相同的准确率

**⚠️ 局限性**

索引构建和处理单机单进程，无法并行或分布式扩展；缺乏对多类空间查询（范围、kNN等）的支持；对大规模数据的内存需求较高

---

## 432. Gaze-Regularized Vision-Language-Action Models for Robotic Manipulation

**arXiv ID:** 2603.23202 | [PDF](https://arxiv.org/pdf/2603.23202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 433. MedCausalX: Adaptive Causal Reasoning with Self-Reflection for Trustworthy Medical Vision-Language Models

**arXiv ID:** 2603.23085 | [PDF](https://arxiv.org/pdf/2603.23085v1)

**作者:** Jianxin Lin `[一作]` (Ohio State University), Yuan Xue `[通讯]` (Ohio State University)

**通讯引用:** 6609 | [OpenAlex ID](https://openalex.org/A5061126706)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MedCausalX，一种针对医学视觉语言模型的因果链式思考框架；

**💡 创新点**

创新点包括：①构建 CRMed 数据集提供因果链和对抗样本；②引入两阶段自适应反射机制（<causal>、<verify> 令牌）实现因果纠正；③采用双策略（DPO+GRPO）将误差归因到因果失配上，强化全程因果一致性；

**🔧 技术方法**

使用的技术包括：结构化因果分解 A→P→Y、反射令牌驱动自我校正、错误归因强化学习、对抗样本生成与对照实验；

**📊 数据集**

主要使用 CRMed 数据集（基于 SA‑Med2D‑20M、MIMIC‑CXR 等）以及多种医学 VQA 与区域定位基准；

**📈 对比分析**

与一般 VLM、医学专用 VLM 与现有 Chain‑of‑Thought 模型对比，MedCausalX 在定位 IoU、诊断一致性、幻觉率等指标上分别提升约+5.4、减少幻觉10+点，取得多项榜首；

**⚠️ 局限性**

局限性包括：需要昂贵的因果标注与对抗样本，训练过程复杂且对硬件要求高，模型对极端样本或新模态的鲁棒性仍待验证。

---

## 434. AeroScene: Progressive Scene Synthesis for Aerial Robotics

**arXiv ID:** 2603.23224 | [PDF](https://arxiv.org/pdf/2603.23224v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 435. MemCollab: Cross-Agent Memory Collaboration via Contrastive Trajectory Distillation

**arXiv ID:** 2603.23234 | [PDF](https://arxiv.org/pdf/2603.23234v1)

**作者:** Yurui Chang `[一作]` (Pennsylvania State University), Lu Lin `[通讯]` (Pennsylvania State University)

**通讯引用:** 71104 | [OpenAlex ID](https://openalex.org/A5100419770)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种跨模型协作的记忆框架（MemCollab），通过对同一任务不同模型产生的推理轨迹进行对比，提炼出可迁移的抽象推理约束，构建单一共享记忆供多种LLM代理使用。

**💡 创新点**

创新点在于：①利用对比学习方式消除模型特有偏差，只保留任务级共性推理模式；②引入任务分类的检索机制，使记忆检索更精确；③证明共享记忆可提升不同规模、不同架构代理的性能，且能提高推理效率。

**🔧 技术方法**

技术核心包括：对比轨迹生成、抽象约束抽取（基于LLM的总结）、任务分类（用LLM预测任务类别和子类别）、基于类别的记忆检索与软指导推理。

**📊 数据集**

使用的数据集有数学推理集（MATH500、GSM8K）和代码生成集（MBPP、HumanEval），并在1000条样本构建记忆、500条样本评估。

**📈 对比分析**

与基线（无记忆、BoT、Dynamic Cheatsheet、单模型记忆、同模型对比记忆）对比，MemCollab在各任务上显著提升准确率（例如7B模型从52.2%提升至67.0%，32B模型从73.8%提升至最高值），同时减少推理步数，表现出更高的推理效率。

**⚠️ 局限性**

局限性包括：①记忆检索仍依赖任务分类准确性；②对比过程需要两种模型并行生成轨迹，计算成本较高；③在极端多样化任务或模型差异较大的情形下，抽取的约束可能不足以覆盖所有错误模式；④目前仅验证了两模型对比，未扩展到更大规模的多模型协作。

---

## 436. PRETTINESS -- Privacy pResErving aTTrIbute maNagEment SyStem

**arXiv ID:** 2603.23221 | [PDF](https://arxiv.org/pdf/2603.23221v1)

**作者:** Jelizaveta Vakarjuk `[一作]` (Cybernetica AS), Alisa Pankova `[通讯]` (Cybernetica AS)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了 PRETTINESS 系统，实现了在 AMS 上安全存储加密属性、支持用户主导的凭证获取、撤销与展示，并在 UC 框架下给出了安全证明。

**💡 创新点**

创新点在于将中心化 AMS 与去中心化用户设备结合，利用阈值签名与去中心化解密隐藏凭证内容，提供可撤销且用户可控的属性管理，同时支持隐私保护的撤销和撤销检测。

**🔧 技术方法**

主要技术包括阈值盲签名（支持签名与解密）、RSA/EC 典型签名、SHA‑256 加密摘要、基于哈希的撤销机制、并使用 Universal Composability 模型进行安全证明。

**📊 数据集**

实验基准采用模拟用户凭证集：每用户 100 条凭证、每凭证 10 个属性，撤销令牌总数 1,000,000 条，使用 SHA‑256、2048‑bit RSA 与常见网络延迟进行测评。

**📈 对比分析**

性能对比显示，凭证颁发约 8.1 s，获取凭证 0.13 s，撤销 1.0–1.5 s，展示 5.7 s，数据库同步 0.72 s；在这些基准下系统整体可接受，尤其在网络环境良好时表现稳定。

**⚠️ 局限性**

主要限制包括：对网络带宽敏感（撤销数据库更新和凭证下载可能耗费大量带宽）、需要可信 AMS 才能避免日志泄漏、未实现完整匿名性与 unlinkability、且系统依赖公钥基础设施与服务器侧安全。

---

## 437. Rethinking Self-Sovereign Identity Principles: An Actor-Oriented Categorization of Requirements

**arXiv ID:** 2603.23177 | [PDF](https://arxiv.org/pdf/2603.23177v1)

**作者:** Daria Schumm `[一作]` (University of Zürich), Burkhard Stiller `[通讯]` (University of Zürich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

将SSI原则细化为24个非功能需求（NFR），并依据参与角色（数据所有者、发行者、验证者、系统）对每个NFR进行责任、所有权划分，同时构建了一个依赖关系模型，用以描述角色间的相互依赖。

**💡 创新点**

首次将用户视角系统化融入DI/SSI需求工程中，将设计模式与细化后的NFR对应，并通过角色责任与所有权的明确划分，提供了一个结构化的DI/SSI系统架构模型。

**🔧 技术方法**

使用需求工程方法（文献综述、需求拆分、角色责任与所有权赋值）以及受Tropos信任模型启发的依赖关系建模技术，构建了依赖模型。

**📊 数据集**

无具体实验数据集；研究基于现有文献与标准（如W3C、Sovrin、IDunion等）进行理论分析。

**📈 对比分析**

未进行实验或量化比较；研究结果以定性分析形式呈现，未给出性能指标。

**⚠️ 局限性**

依赖现有设计模式的覆盖范围有限，部分NFR未被现有模式覆盖；缺乏对各角色能力的量化评估，需求未完成功能化实现，模型仍处于概念验证阶段。

---

## 438. From Synthetic to Native: Benchmarking Multilingual Intent Classification in Logistics Customer Service

**arXiv ID:** 2603.23172 | [PDF](https://arxiv.org/pdf/2603.23172v1)

**作者:** Haoyu He `[一作]` (J&T Express), Kunpeng Han `[通讯]` (J&T Express)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了一个基于真实物流客服日志的多语种意图分类基准，包含约3万条去标识化查询，构成两层标签体系，并提供本土与机器翻译对照的测试集。

**💡 创新点**

创新点在于：①真实噪声查询与机器翻译对照揭示翻译评测高估模型鲁棒性；②引入零射击评估语言和长尾标签的层次结构，系统性测量真实与合成评测差距。

**🔧 技术方法**

采用规则+LLM过滤+人工验证的清洗管道构建标签层次；基线模型包括多语言编码器（mBERT、XLM‑R、mmBERT）、嵌入+分类器、指令微调小模型（Gemma 3 270M/1B、Qwen 3 0.6B）以及 API 关闭源 LLM。

**📊 数据集**

使用约3万条去标识化客服查询，涵盖英、西、阿、印、汉、泰六种语言，构成13父类/17叶类的两层标签体系，并提供原始与机器翻译配对。

**📈 对比分析**

在见语言监督下，父类任务准确率约94%，叶类任务约93%；机器翻译测试普遍高于本土测试，尤其在长尾和跨语言转移上差距显著；在未见语言的零射击评测中，Gemma 3 1B表现最佳。

**⚠️ 局限性**

局限性包括仅覆盖六种语言、标签体系有限、只处理单句意图且未包含多轮对话、槽填充或任务成功评估，仅使用单一翻译引擎，评估指标主要为准确率，缺乏显著性检验和持续更新。

---

## 439. Decoding AI Authorship: Can LLMs Truly Mimic Human Style Across Literature and Politics?

**arXiv ID:** 2603.23219 | [PDF](https://arxiv.org/pdf/2603.23219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 440. Online library learning in human visual puzzle solving

**arXiv ID:** 2603.23244 | [PDF](https://arxiv.org/pdf/2603.23244v1)

**作者:** Pinzhe Zhao `[一作]` (University of Edinburgh), Bonan Zhao `[通讯]` (University of Edinburgh)

**通讯引用:** 137 | [OpenAlex ID](https://openalex.org/A5002181484)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了人类在视觉拼图任务中在线学习和重用中间抽象（helpers）的过程，提出Pattern Builder Task并进行实验

**💡 创新点**

创新在于构造可实时生成和共享的抽象库，并将程序归纳与在线库学习相结合来解释人类问题解决

**🔧 技术方法**

使用程序归纳（program induction）与底层搜索技术，并在自定义DSL中实现助手的创建与复用

**📊 数据集**

使用了14个递增难度的视觉模式（P1–P14）作为任务数据集，另外在自由玩阶段收集了参与者自创的模式

**📈 对比分析**

与模型（Baseline、Short、Library、Short+Library）比较，库学习模型能解决所有14题；人类平均准确率92.4%，模型节点扩展数与人类解题时间/步骤高度相关，程序长度仅与成功率相关

**⚠️ 局限性**

局限在于仅将完整解作为新原语，未考虑部分抽象；实验仅在单个学习者上，缺乏社会情境；模型简化未捕捉更复杂的抽象策略

---

## 441. I Came, I Saw, I Explained: Benchmarking Multimodal LLMs on Figurative Meaning in Memes

**arXiv ID:** 2603.23229 | [PDF](https://arxiv.org/pdf/2603.23229v1)

**作者:** Shijia Zhou `[一作]`, Diego Frassinelli `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究探讨了多模态大语言模型在识别和解释网络 meme 中隐喻性表达的能力，提出并基准化了检测+解释任务，系统评估了图像、文本及其组合对模型判断的影响，并通过人工评估评判模型生成解释的可信度与一致性。

**💡 创新点**

创新点包括：①首次将 meme 里的隐喻识别与解释结合为一体的任务；②通过对图像、文本两种模态的消融与组合实验，揭示不同隐喻类型对模态依赖的差异；③引入人类评估维度（关联性、连贯性、可信度等），系统剖析模型解释的偏差与错误。

**🔧 技术方法**

技术方法主要采用零样本提示的多模态大语言模型（Aya‑Vision、Gemma‑3、Qwen2.5‑VL），在完整 meme、仅文本、仅图像三种输入条件下进行对比实验；并结合 PaddleOCR 与 LaMa 进行文本掩蔽与图像修补；最终用人工标注评估模型解释质量。

**📊 数据集**

使用了三大公开 meme 数据集：①包含六类隐喻类型的政治 meme 数据集；②包含讽刺等情绪标签的 Reddit/Weibo 等混合来源数据集；③聚焦隐喻的英文 meme 数据集，并对训练集与测试集按原始划分进行评估。

**📈 对比分析**

通过对 8 个模型在二分类和多标签分类上的 F1、精确率、召回率等指标进行对比，发现更大模型整体表现更好，但差距不大；模型往往对 meme 形式存在过度赋予隐喻的偏差；文本/图像消融显示不同隐喻类型对模态的依赖差异；在人工评估中，模型在解释文本方面表现较好，视觉信息解释则普遍不足，且常出现幻觉或过度解读。

**⚠️ 局限性**

主要局限包括：①数据集可能已被预训练模型泄露，导致性能上升；② meme 格式单一，标签分布不均，跨数据集对比受限；③模型对提示敏感，输出格式与措辞可能影响结果；④解释评估受限于人工主观性，且缺乏对社会心理语境的深度把握；⑤存在潜在的仇恨内容伦理风险。

---

## 442. General Machine Learning: Theory for Learning Under Variable Regimes

**arXiv ID:** 2603.23220 | [PDF](https://arxiv.org/pdf/2603.23220v1)

**作者:** Aomar Osmani `[一作]` `[通讯]` (Institut National des Sciences Appliquées), Aomar Osmani (Institut National des Sciences Appliquées)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一套面向多种运行环境（regime）变化的学习理论框架（GML），并给出了其基本定义、结构约束及第一个理论层面的证明；

**💡 创新点**

核心创新在于将可接受的regime转移、受保护核心保持、评估器兼容性等结构化约束纳入学习对象，形成Hadamard式的可行性与稳定性三原则，揭示了经典固定-评估器学习与多regime学习的本质区别与不可能的归约；

**🔧 技术方法**

使用了可测空间、类型化变换、受保护核心投影、可行性判定（Γ）与受保护等价关系等数学工具，构造了稳定性模板、记忆约束、评估器因子分解、迁移与组合等定理；

**📊 数据集**

无实验数据集，本文纯理论推导；

**📈 对比分析**

无实验对比，未给出性能指标；

**⚠️ 局限性**

局限性在于目前仅给出了单一“数值顺序”工作层的完整证明，符号/知识层的具体实现和跨层推理仍待后续工作；另外可行性判定Γ在实际应用中往往难以计算，理论上也未给出有效算法。

---

## 443. Joint Task Orchestration and Resource Optimization for SC3 Closed Loop in 6G Networks

**arXiv ID:** 2603.23217 | [PDF](https://arxiv.org/pdf/2603.23217v1)

**作者:** Xinran Fang `[一作]` (Tsinghua University), Shi Jin `[通讯]` (Southeast University)

**通讯引用:** 45039 | [OpenAlex ID](https://openalex.org/A5013079905)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种面向6G自律操作系统的闭环优化框架，联合实现传感器-执行器配对与通信/计算资源分配，以最小化多闭环控制系统的LQR成本。

**💡 创新点**

创新点包括：①将闭环控制视为整体的S C^3 单元，首次将传感器-执行器配对与资源分配统一为一个混合整数非线性规划；②设计了学习‑优化集成的Actor‑Critic（LOAC）框架，用深度网络生成配对候选，优化器评估并反馈LQR成本，迭代逼近最优；③在配对过程中引入有效信息提取比和计算强度，动态平衡通信与计算瓶颈。

**🔧 技术方法**

技术方法主要包括：深度神经网络（双编码器+配对网络）生成配对概率；优化器（凸规划）求解连续资源分配；Actor‑Critic交互的经验回放与mini‑batch训练；以及对比实验中使用的基线算法（穷举搜索、SCA松弛、通信/计算启发式配对）。

**📊 数据集**

实验使用仿真数据，构建4000m×4000m的场景，S=20个传感器、K=4个执行器，随机生成信道小尺度衰落、位置、计算需求等；无公开数据集，全部为合成仿真场景。

**📈 对比分析**

与穷举搜索、SCA松弛、通信‑计算启发式、QoS等基线进行比较，LOAC在LQR成本上与穷举搜索几乎持平，显著优于松弛和启发式方法；同时其计算复杂度从指数级下降到多项式级，训练收敛快（<1500 epoch）。

**⚠️ 局限性**

局限性在于：仅在仿真环境下验证，缺乏真实灾害现场数据；假设传感器数量远大于执行器且信道模型相对简化；配对过程对拓扑变化的自适应机制尚未完整实现；对大规模K、S以及更动态环境的可扩展性仍需进一步研究。

---

## 444. Covering and Partitioning Complex Objects with Small Pieces

**arXiv ID:** 2603.23216 | [PDF](https://arxiv.org/pdf/2603.23216v1)

**作者:** Anders Aamand `[一作]` (University of Copenhagen), Jack Stade `[通讯]` (University of Copenhagen)

**通讯引用:** 5368 | [OpenAlex ID](https://openalex.org/A5005110126)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文研究将任意多边形（可含洞）分割或覆盖为最少数量的轴对齐单位正方形内的连通子多边形；在二维情形给出局部搜索的 PTAS，证明覆盖与划分最优数相等；在三维构造极限证明该问题的近似不可超过对数因子；并针对给定片数 k 的约束问题提出 2+ 近似求最小尺寸的贪心算法。

**💡 创新点**

主要创新点包括：①利用非穿刺几何集覆盖理论构造局部搜索 PTAS，得到 1+O(1/√k) 的近似；②证明二维覆盖与划分的最优值相同；③在三维给出 NP‑hard 的对数逼近下限；④在 k‑片约束下提供 2+ 近似的简单贪心方案。

**🔧 技术方法**

核心技术包括局部搜索、非穿刺区域的组合优化、线性规划与凸几何分析、组合结构枚举、三维构造（Hook‑piece、墙面拼接）、稀疏覆盖与逼近分析。

**📊 数据集**

论文主要以理论形式呈现，没有使用实验数据集，所有结论基于多边形几何实例与构造的三维 polyhedron。

**📈 对比分析**

相较于此前仅适用于无洞多边形的 13‑approximation，本文在二维给出任意 ε 的 PTAS，并在三维中给出与 Set Cover 对数下限相同的近似硬性；在给定 k 片的场景下，贪心算法提供 2+ 近似，时间复杂度为 O(nk log(k/ε))。

**⚠️ 局限性**

限制与不足：PTAS 仅适用于二维，多维扩展仍缺乏高效实现；三维结果仅为 NP‑hard 下限，缺乏近似算法；局部搜索收敛速度与实例规模相关；算法对多边形的表示方式（边列表、单纯形复合等）有要求；未给出实际运行时间与实验验证。

---

## 445. A One-Inclusion Graph Approach to Multi-Group Learning

**arXiv ID:** 2603.23208 | [PDF](https://arxiv.org/pdf/2603.23208v1)

**作者:** Noah Bergam `[一作]` (Columbia University), Daniel Hsu `[通讯]` (Columbia University)

**通讯引用:** 10686 | [OpenAlex ID](https://openalex.org/A5061246300)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种多组学习的一致包含图（OIG）算法，利用多组匹配与线性规划实现对组条件误差的上界控制

**💡 创新点**

创新点在于将传统单组OIG算法推广到多组情形，设计多组最大匹配问题并通过对偶性证明其最优性，从而在单组评估时实现1/n速率，在全组同时评估时获得最优的O(log n / n)速率，并给出了对应的下界

**🔧 技术方法**

核心技术包括：多组一包含图构造、基于组边的多组b匹配（多组容量约束）、线性规划求解、对偶分析与握手引理、以及组合学中的最大子图密度与VC维度关系

**📊 数据集**

未使用具体数据集，研究以理论分析和假设分布为主

**📈 对比分析**

通过理论证明与下界对比，展示所提出算法在单组评估时达到最优1/n、在全组评估时接近最优O(log n / n)，优于此前仅能达到O(log n / n)的经验风险最小化（ERM）方法

**⚠️ 局限性**

主要局限在于对群体可实现性（group‑realizability）假设的依赖，以及对无限群体族的处理仍需假设VC维度有限，实际应用需进一步验证其泛化性能

---

## 446. SAiW: Source-Attributable Invisible Watermarking for Proactive Deepfake Defense

**arXiv ID:** 2603.23178 | [PDF](https://arxiv.org/pdf/2603.23178v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 447. FDIF: Formula-Driven supervised Learning with Implicit Functions for 3D Medical Image Segmentation

**arXiv ID:** 2603.23199 | [PDF](https://arxiv.org/pdf/2603.23199v1)

**作者:** Yukinori Yamamoto `[一作]` (Waseda University), Hirokatsu Kataoka `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 4542 | [OpenAlex ID](https://openalex.org/A5011507481)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用隐式签名距离函数（SDF）生成合成3D医学图像与标注数据，并在此数据上进行预训练，随后在真实医学数据上微调。

**💡 创新点**

通过将SDF与位移函数和距离到强度映射函数结合，实现既能控制几何形状又能生成纹理与强度的完整合成框架（FDIF），显著提升公式驱动监督学习的表现。

**🔧 技术方法**

核心技术包括SDF库、位移（Displacement）库、强度映射（Mapper）库；在此基础上构建FDIF生成器，并对SwinUNETR、nnUNet等网络进行预训练与微调。

**📊 数据集**

实验使用三组分割基准（AMOS22、ACDC、KiTS19）以及四组分类基准（MRNet、MedMNIST Organ、Nodule、Fracture）来评估预训练效果。

**📈 对比分析**

与从零训练、PrimGeoSeg（公式驱动）和多种SSL（MAE、S3D、MG、SimMIM、VF）方法对比，FDIF在大多数任务中达到了或超过了SSL水平，平均Dice提升约2–4点，分类平均准确率提升约1.3点。

**⚠️ 局限性**

局限在于合成纹理可能无法完全再现真实医学图像的细节，且依赖手工设计的SDF、位移与映射库，难以覆盖所有解剖变异；此外未验证在更大规模真实数据上的迁移性能。

---

## 448. GSwap: Realistic Head Swapping with Dynamic Neural Gaussian Field

**arXiv ID:** 2603.23168 | [PDF](https://arxiv.org/pdf/2603.23168v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 449. Privacy-Aware Smart Cameras: View Coverage via Socially Responsible Coordination

**arXiv ID:** 2603.23197 | [PDF](https://arxiv.org/pdf/2603.23197v1)

**作者:** Chuhao Qin `[一作]` (University of Leeds), Evangelos Pournaras `[通讯]` (University of Leeds)

**通讯引用:** 1885 | [OpenAlex ID](https://openalex.org/A5035070413)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种去中心化的隐私友好型智能摄像头视角协同框架，通过本地生成与协同学习实现覆盖优化并避免监控隐私区域。

**💡 创新点**

关键创新在于将硬约束隐私要求与I‑EPOS集体学习算法结合，实现在大规模摄像头网络中无需中心控制即可自动选择视角，同时显著降低隐私违规。

**🔧 技术方法**

采用I‑EPOS迭代经济规划与优化选择算法、基于投影光学的覆盖模型、Ray‑Tracing与Bresenham算法、以及树形通信结构的分布式协同学习。

**📊 数据集**

使用公开的模拟环境数据集 https://doi.org/10.6084/m9.figshare.31332856（生成的摄像头方向计划）以及通过仿真得到的覆盖与隐私区域地图。

**📈 对比分析**

与集中式Optimal和贪心格点投票（GGV）基线对比，实验显示在100–1000摄像头场景中I‑EPOS‑HC覆盖效率提升18.42%，隐私违规率降低85.53%，且通信/计算开销低于COHDA/EPOS。

**⚠️ 局限性**

限制包括需预先定义隐私区域，算法对极高分辨率/复杂三维场景的扩展尚未验证，以及在极稀疏部署下可能导致覆盖不足。

---

## 450. The Power of Power Codes: New Classes of Easy Instances for the Linear Equivalence Problem

**arXiv ID:** 2603.23230 | [PDF](https://arxiv.org/pdf/2603.23230v1)

**作者:** Michele Battagliola `[一作]` (Marche Polytechnic University), Violetta Weger `[通讯]` (Technical University of Munich)

**通讯引用:** 134 | [OpenAlex ID](https://openalex.org/A5084539841)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文研究了线性等价问题（LEP），并基于幂码、Schur乘积、弗罗贝尼乌斯同态和Hermitian hull等工具，提出了一系列新的概率性攻击方法，能够在许多此前未知的弱实例中以多项式时间破解LEP；

**💡 创新点**

创新点主要包括①将Schur乘法攻击从Permutation Equivalence Problem（PEP）推广到更一般的LEP；②利用幂码与弗罗贝尼乌斯自同态、Hermitian hull以及部分闭包构造出新的弱实例，扩展了可被攻击的字段大小与参数范围；③引入了中间问题LEP(U)和部分闭包技术，实现了比传统方法更高效的PEP归约；

**🔧 技术方法**

主要技术手段包括幂码（power codes）、Schur乘积攻击、弗罗贝尼乌斯自同态、Hermitian hull分析、部分闭包（partial closure）构造以及概率性算法设计；

**📊 数据集**

实验使用的是随机生成的线性码实例（如[100,10]_5、[300,6]_8等），未使用公开数据集；

**📈 对比分析**

相较于已有的平均/最坏情况求解算法，本文算法在满足k<√(r!·n)（或相应的更一般形式）等条件时，以多项式时间完成LEP，并在实验中显示误判概率极低、平均成功率高；

**⚠️ 局限性**

局限性包括：①攻击仅适用于特定阶的有限域和参数范围；②算法为概率性，存在极小的误判（false positives）；③对极大维度k'或某些特殊码结构时，攻击效果可能下降。

---

## 451. GEM: Guided Expectation-Maximization for Behavior-Normalized Candidate Action Selection in Offline RL

**arXiv ID:** 2603.23232 | [PDF](https://arxiv.org/pdf/2603.23232v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 452. PoseDriver: A Unified Approach to Multi-Category Skeleton Detection for Autonomous Driving

**arXiv ID:** 2603.23215 | [PDF](https://arxiv.org/pdf/2603.23215v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 453. Robust Safety Monitoring of Language Models via Activation Watermarking

**arXiv ID:** 2603.23171 | [PDF](https://arxiv.org/pdf/2603.23171v1)

**作者:** Toluwani Aremu `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Nils Lukas `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于内部激活的水印监控机制，能够在LLM产生违规内容时在模型内部嵌入可检测的秘密信号，供部署方在单次推理中高效检测并定位违规行为。

**💡 创新点**

创新点在于将检测信号直接嵌入模型隐藏层，并使用随机化的关键向量（secret key）来防止自适应攻击者迁移攻击；此外，设计了适应性攻击评估框架和秘密提取游戏，验证了水印对自适应攻击的鲁棒性。

**🔧 技术方法**

核心技术包括：激活水印（activation watermarking）——在特定层采样随机向量并在训练中正则化激活；线性加权策略突出后期违规信息；余弦相似度检测；基于梯度的微调与KL正则；对比实验中的安全基线（QwenGuard、LlamaGuard、ActProbe）。

**📊 数据集**

使用了BeaverTails（恶意/正常对话数据）、XSTest（起始种子）、EasyJailbreak生成的四类自适应攻击（Jailbroken、DeepInception、Multilingual、AutoDAN）以及标准公开基准（BBH、IFEval、MMLU-pro、TruthfulQA、GSM8K、MATH-Hard）评估模型性能。

**📈 对比分析**

与外部guard模型相比，激活水印在自适应攻击下的ASR（逃逸率）显著更低（≤0.05），AUROC持续保持在0.90以上；在正常数据上误报率保持≤1%，且对模型规模（7B/14B）与跨模型迁移攻击也表现出高鲁棒性；与ActProbe相比，水印在多攻击场景下的检测准确率提升约10%-20%。

**⚠️ 局限性**

局限性包括：仅在黑盒攻击场景评估，缺乏对部分白盒攻击的防御；缺乏理论证明的鲁棒性保证；对模型微调可能导致在数学推理等任务上性能下降；对实际用户流量的长期效应与可解释性、透明度及合规性问题未作深入探讨。

---

## 454. LiZIP: An Auto-Regressive Compression Framework for LiDAR Point Clouds

**arXiv ID:** 2603.23162 | [PDF](https://arxiv.org/pdf/2603.23162v1)

**作者:** Aditya Shibu `[一作]` (Heriot Watt University), Claudio Zito `[通讯]` (Heriot Watt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种轻量级、近无损的 LiDAR 点云压缩框架 LiZIP。

**💡 创新点**

创新点在于将神经预测编码与 Morton 排序相结合，使用仅 540 KB 的小型 MLP 进行自回归预测，从而在保持几乎无误差的同时显著提升压缩率。

**🔧 技术方法**

采用了多层感知机（MLP）神经预测、Morton 码空间排序、固定量化、字节洗牌以及 LZMA/Deflate 熵编码技术。

**📊 数据集**

使用了公开的城市 LiDAR 数据集 NuScenes（训练/验证）和 Argoverse（无训练测试）。

**📈 对比分析**

与 LASzip、Google Draco、GZip 等基准算法比较，LiZIP 在 NuScenes 上压缩率比 LASzip 高约 7.5%（比 Draco 低 1.1%），在 Argoverse 上压缩率比 LASzip 高 14.8%；总体压缩比约 71%，编码/解码时间约 75 ms，满足 10 Hz LiDAR 的实时需求。

**⚠️ 局限性**

局限性包括：目前仅在 CPU 上实现，编码速度仍高于 LASzip；仅压缩几何信息，未处理颜色/强度等属性；对非城市或不同传感器环境的泛化仍待进一步验证。

---

## 455. Path Planning and Reinforcement Learning-Driven Control of On-Orbit Free-Flying Multi-Arm Robots

**arXiv ID:** 2603.23182 | [PDF](https://arxiv.org/pdf/2603.23182v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 456. Is AI Catching Up to Human Expression? Exploring Emotion, Personality, Authorship, and Linguistic Style in English and Arabic with Six Large Language Models

**arXiv ID:** 2603.23251 | [PDF](https://arxiv.org/pdf/2603.23251v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 457. GO-Renderer: Generative Object Rendering with 3D-aware Controllable Video Diffusion Models

**arXiv ID:** 2603.23246 | [PDF](https://arxiv.org/pdf/2603.23246v1)

**作者:** Zekai Gu `[一作]` (HKUST), Yuan Liu `[通讯]` (HKUST)

**通讯引用:** 16199 | [OpenAlex ID](https://openalex.org/A5049891570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 GO-Renderer，一种将粗略 3D 代理与视频扩散模型相结合的统一框架，能够在任意视角和任意照明条件下从少量参考图像合成高质量的物体视频。

**💡 创新点**

创新点包括：①使用对象坐标图将 3D 代理转化为像素级空间约束，指导扩散模型精确控制视角；②引入负 RoPE 时序偏移，使参考帧与目标序列分离，避免过度融合；③构建包含多种光照和轨迹的大规模 57k 视频数据集，为训练提供充足的多视角多光照样本。

**🔧 技术方法**

技术手段包括：前向 3D 重建（ReconViaGen/VGGT）生成 3D 代理；对象坐标图渲染；Wan2.2Fun 5B Ref Control 视频扩散 Transformer；负 RoPE 时序偏移；以及基于 3D 代理的特征对齐与融合。

**📊 数据集**

使用的数据集为：①通过 Blender 结合 Google Scanned Objects、TexVerse 以及 100 张 HDRI 生成的合成数据；②从 OpenVidHD、CO3D 提取的真实视频并通过 SAM3、ReconViaGen 等工具生成的参考图与 3D 代理；③利用 Wan2.2 生成的 AI 合成视频；共构建约 57k 条视频样本。

**📈 对比分析**

实验与 UniLumos（基于显式 3D 重建与重光照）、Phantom、Wan2.2Fun 5B Ref Control 等基线进行对比，采用 PSNR、SSIM、CLIP、DINO 等指标。GO-Renderer 在 PSNR（18.26）和 SSIM（0.684）上分别超过 Wan2.2 的 14.30 / 0.641，且在多视角一致性指标（CLIP 0.888 / DINO 0.725）上优于 AnySplat（0.861 / 0.191）和 ReconViaGen（0.796 / 0.448）。

**⚠️ 局限性**

局限性主要在于对 3D 代理精度的依赖；若代理几何错误或与参考外观不匹配，模型会产生结构伪影，视角控制精度下降；此外，目前仍需先行完成 3D 重建步骤，限制了实时或完全端到端的应用。

---

## 458. A Law of Large Numbers with Convergence Rate based on Nonlinear Expectation Theory and Its Application to Communication Detection

**arXiv ID:** 2603.23212 | [PDF](https://arxiv.org/pdf/2603.23212v1)

**作者:** Jialiang Fu `[一作]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences), Wen-Xuan Lang `[通讯]` (Academy of Mathematics and Systems Science, Chinese Academy of Sciences)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在子线性期望框架下的特殊部分和收敛速度 LLN，并将其应用于反馈通道下的检测误差概率分析。

**💡 创新点**

提出带收敛速度的 LLN，处理可预测不确定系数导致的子线性期望，并将其用于非 i.i.d. 输入的检测问题，填补传统 i.i.d. 方法的空白。

**🔧 技术方法**

子线性期望理论、可测近似技巧、Cramér 大偏差理论和高阶矩条件。

**📊 数据集**

未使用具体数据集，主要进行理论推导与证明。

**📈 对比分析**

与传统 i.i.d. CLT/LLN 对比，证明误差概率随样本数指数下降，收敛速度为 O(1/n^{α/(1+α)})，实现更严格的误差控制。

**⚠️ 局限性**

仅适用于可预测区间内的系数，需满足较强的高阶矩条件，理论假设较为理想化。

---

## 459. Who Is in the Room? Stakeholder Perspectives on AI Recording in Pediatric Emergency Care

**arXiv ID:** 2603.23187 | [PDF](https://arxiv.org/pdf/2603.23187v1)

**作者:** Alexandre De Masi `[一作]` (University of Geneva), Frederic Ehrler `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并分析了在儿科急救中人工智能录音系统缺失关键利益相关者视角的影响，并提出四个以利益相关者为中心的设计与治理建议。

**💡 创新点**

首次将同意、情感影响、监视效应和参与式治理视角系统化地融入儿科急救AI录音的设计框架，并提出了四项具体的治理与设计原则。

**🔧 技术方法**

未采用具体技术实现，而是基于人机交互、伦理学、监视研究等跨学科文献进行结构化综述与论证。

**📊 数据集**

未使用数据集，研究基于现有文献综述和作者的实地经验。

**📈 对比分析**

未进行方法比较或性能评估，本文仅提供概念性框架与政策建议。

**⚠️ 局限性**

局限在于缺乏实证验证、仅为立场性论文，未覆盖所有临床场景，建议未来进行多中心定性与实验研究。

---

## 460. A Learning Method with Gap-Aware Generation for Heterogeneous DAG Scheduling

**arXiv ID:** 2603.23249 | [PDF](https://arxiv.org/pdf/2603.23249v1)

**作者:** Ruisong Zhou `[一作]` (Peking University), Zaiwen Wen `[通讯]` (Peking University)

**通讯引用:** 4799 | [OpenAlex ID](https://openalex.org/A5006127137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种单通道强化学习框架WeCAN，用于异构DAG调度，兼顾任务-池兼容性并消除生成引起的最优性缺口。

**💡 创新点**

核心创新包括：①基于“顺序空间”分析，定义并证明生成映射的OCO（Order-Covering Optimal）条件；②设计单通道“跳过扩展”生成映射，使用递减跳分数保证可达性；③构建无尺寸依赖的加权交叉注意力网络WeCAN，显式编码任务-池兼容性与DAG结构。

**🔧 技术方法**

技术手段包括：强化学习（策略梯度），加权交叉注意力机制，最长有向距离（LDD）图神经网络，单通道决策分数+跳分数，生成映射S_d与S_SGS比较。

**📊 数据集**

使用两类数据集：工业真实查询生成的TPC‑H DAG（含300–1000节点，异构资源池）以及人工合成的计算图数据集（500节点，层图、Erdős‑Rényi、随机块模型）。

**📈 对比分析**

与传统列表调度（CP、SFT、HEFT等）、多轮神经调度（PPO‑BiHyb、One‑Shot、ScheduleNet、HGN‑Two‑Stage）进行对比。WeCAN在所有基准上均取得10–15%左右的周转时间改进，且推理速度与经典启发式相当或更快。

**⚠️ 局限性**

局限性包括：对任务/池兼容性假设为静态且已知，模型对极端任务负载变化敏感；跳分数需要手工调参；在极大规模或高度动态环境下的可扩展性和泛化仍待进一步验证。

---

## 461. PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments

**arXiv ID:** 2603.23231 | [PDF](https://arxiv.org/pdf/2603.23231v1)

**作者:** Shuochen Liu `[一作]` (University of Science and Technology of China), Tong Xu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4320 | [OpenAlex ID](https://openalex.org/A5025292786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了名为 PERMA 的事件驱动个性化记忆评测基准，旨在测试大型语言模型（LLM）和记忆系统在跨会话、跨领域、含噪声和真实语料风格对话中的长期人物一致性与偏好提取能力。

**💡 创新点**

创新点包括：①以事件驱动而非静态偏好记忆构建对话序列，模拟偏好随时间演进；②引入会话内噪声和语言风格对齐，提升评测真实感；③采用多维度评估（单次 MCQ、交互式对话、时序探测）和分层检索/记忆操作，拆解记忆质量与生成效果；④系统性对比多种记忆架构（RAG、MemOS、Mem0、Lightmem 等）与不同 LLM 的表现。

**🔧 技术方法**

技术手段包括：事件-对话生成流水线（先生成事件时间线，再基于 LLM 生成对话）；噪声注入与风格对齐（五种噪声类型 + WildChat 语料）；记忆系统的 Add–Search 双流程；多轮用户模拟器进行交互式评测；使用 BERT‑f1、Memory Score、Token‑Eff等指标评估检索质量与生成效率。

**📊 数据集**

数据集为 20 个主题（如电影、旅行、金融等）下的 800+ 事件、3k 轮对话，Clean 版约 1.8M 词；Noise 版在 Clean 基础上注入五类噪声；Style‑Aligned Long‑Context 版在 116k 词级别加入 WildChat 真实语料，合计约 1.16M 词。所有对话均经过 LLM 验证与人工审核。

**📈 对比分析**

比较方法：将不同模型按单域、跨域、Clean/Noise/Long‑Context 三种情景分别评测；对比指标包括：MCQ 准确率、BERT‑f1、Memory Score、检索 token 数与耗时、交互成功率、Token‑Efficiency。实验结果显示：记忆系统整体优于单纯 RAG，尤其在噪声或长上下文中表现更稳健；但在多域跨会话、噪声极端或极长上下文下仍出现偏好漂移和检索误差，导致性能下降。

**⚠️ 局限性**

局限性：①评测仍以 MCQ 与预设选项为主，无法完全捕捉生成式推理的细粒度错误；②噪声与风格对齐是人工规则化生成，可能与真实用户行为差异；③记忆系统在跨域干扰与长期序列中易出现记忆衰退；④实验依赖 GPT‑4o‑mini 作为基础生成器，未探究更大模型对记忆策略的影响；⑤缺乏对记忆存储与更新策略的动态调优机制，导致在极长历史中表现不稳定。

---

## 462. Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics for Robot Manipulation

**arXiv ID:** 2603.23227 | [PDF](https://arxiv.org/pdf/2603.23227v1)

**作者:** Qinglun Zhang `[一作]` (University of Electronic Science and Technology of China), Shuaicheng Liu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 7109 | [OpenAlex ID](https://openalex.org/A5039387461)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了E3Flow，一种融合球面谐波表示与修正流匹配的多模态SO(3)等变视觉运动控制框架

**💡 创新点**

创新点在于：①将球面谐波作为连续SO(3)等变特征，保证旋转一致性；②设计特征增强模块(FEM)动态融合图像与点云的可变特征；③采用修正流匹配（rectified flow）实现一次或少步采样，同时保持等变性，显著提升推理速度；④首次在等变策略中同时兼顾数据效率与速度

**🔧 技术方法**

使用的技术包括：球面谐波(Spherical Harmonics)、EquiformerV2点云编码、ResNet图像编码、特征增强模块(FEM)、修正流匹配(ODE/Rectified Flow)、AdamW优化、EMA稳定训练

**📊 数据集**

在MimicGen仿真平台的8个操纵任务上训练（每个任务100条专家演示），并在4个真实机器人实验中验证

**📈 对比分析**

与EquiBot、EquiDiff、SDP、DP3等多种等变与非等变基线比较，E3Flow在平均成功率上比最强等变基线SDP提升3.12%，比最强非等变基线DP3提升53.5%，同时推理速度比SDP快7倍（0.51s vs 3.73s）

**⚠️ 局限性**

局限性：对大幅度姿态变换（如10°以上倾斜）尚未完全验证；仍需依赖点云与图像两模态，缺少单模态下的鲁棒性；在极端光照或遮挡环境下的性能未知

---

## 463. Sparser, Faster, Lighter Transformer Language Models

**arXiv ID:** 2603.23198 | [PDF](https://arxiv.org/pdf/2603.23198v1)

**作者:** Edoardo Cetin `[一作]` (Sakana AI), Llion Jones `[通讯]` (Sakana AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型的前馈层，引入稀疏打包格式TwELL和CUDA核，实现高效推理与训练。

**💡 创新点**

创新点在于：设计无结构稀疏的内存布局与融合核，能够在GPU上获得显著吞吐和能耗提升；同时证明轻量L1正则可实现99%以上稀疏而几乎不损失性能。

**🔧 技术方法**

使用的技术包括：TwELL稀疏格式、CUDA kernel融合、Tensor Core优化、混合稀疏-稠密混合格式、L1正则训练、Transformer++架构等，并在H100 GPU上部署。

**📊 数据集**

数据集为FineWeb，采用Transformer++架构，规模从0.5B到2B，训练至各自的“chinchilla-optimal”标记。

**📈 对比分析**

与全密集模型对比，稀疏模型在前向推理上提升约20%吞吐、能耗下降≈3%，训练时提升约24%速度、显存下降≈25%，且在下游任务上无显著准确率下降。

**⚠️ 局限性**

局限性包括：对极端稀疏导致的“dead neuron”问题尚未彻底解决；混合格式实现复杂度较高；在不同GPU架构上迁移性能待验证。

---

## 464. Gimbal360: Differentiable Auto-Leveling for Canonicalized $360^\circ$ Panoramic Image Completion

**arXiv ID:** 2603.23179 | [PDF](https://arxiv.org/pdf/2603.23179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 465. GTLR-GS: Geometry-Texture Aware LiDAR-Regularized 3D Gaussian Splatting for Realistic Scene Reconstruction

**arXiv ID:** 2603.23192 | [PDF](https://arxiv.org/pdf/2603.23192v1)

**作者:** Yan Fang `[一作]` (Ningbo Institute of Materials Technology and Engineering, UCAS), Jiangjian Xiao `[通讯]` (Ningbo Institute of Materials Technology and Engineering, UCAS)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于LiDAR的3D Gaussian Splatting框架，通过几何纹理感知分配、曲率自适应细化和基于LiDAR深度的尺度正则化实现高精度的度量尺度重建。

**💡 创新点**

创新点在于将LiDAR几何先验主动嵌入到Gaussian分布的分配、细化和约束中，并利用曲率引导分裂与置信度加权的深度正则化。

**🔧 技术方法**

使用技术包括3D Gaussian Splatting、曲率估计、kNN几何纹理评估、基于LiDAR的深度提取与不偏深度渲染、光度与SSIM损失以及显著性权重的尺度约束。

**📊 数据集**

使用了ScanNet++公开数据集和自建移动LiDAR采集的真实场景数据集进行评估。

**📈 对比分析**

与原始3DGS、FSGS和Octree-GS等方法对比，在ScanNet++和自建数据集上在PSNR、SSIM、LPIPS等指标上实现了更高的质量并降低了浮点现象，训练时间略高但显著提升几何一致性。

**⚠️ 局限性**

局限在于对移动LiDAR噪声和密度不一致的处理仍不充分，且方法依赖于准确配准的LiDAR点云，对低精度或快速运动的系统适用性尚待验证。

---

## 466. ImplicitRM: Unbiased Reward Modeling from Implicit Preference Data for LLM alignment

**arXiv ID:** 2603.23184 | [PDF](https://arxiv.org/pdf/2603.23184v1)

**作者:** Hao Wang `[一作]` (Peking University), Zhouchen Lin `[通讯]` (Peking University)

**通讯引用:** 26258 | [OpenAlex ID](https://openalex.org/A5016399094)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于隐式偏好数据的奖励模型学习框架ImplicitRM，解决无明确负样本和用户偏好偏差两大难题。

**💡 创新点**

创新点在于将隐式偏好数据按潜在偏好与行动概率四分为四类，并利用分层模型推断各组概率，进而构造无偏的似然最大化目标。

**🔧 技术方法**

采用潜在分层估计器（propensity estimator）和奖励估计器（preference estimator），两者均基于LLM backbone+MLP头，训练目标为基于ELBO的无偏损失。

**📊 数据集**

实验数据集包括HelpSteer、UltraFeedback和PKU‑SafeRLHF三组公开隐式偏好数据集。

**📈 对比分析**

与传统的Naïve、IPS/DR、PU等方法对比，ImplicitRM在R²、MAE、RMSE等指标上均显著提升（如PKU‑SafeRLHF上R²提升至0.787，远超对照方法）。在下游RLHF安全评测中，使用ImplicitRM的奖励模型可使模型在HarmBench、StrongReject、WildGuardMix等基准上提升5–15%。

**⚠️ 局限性**

局限性包括：仅关注训练目标，未探索专用网络结构；假设所有正反馈均代表正偏好，未考虑误点击或噪声反馈，需要进一步研究鲁棒性与不确定性估计。

---

## 467. Reasoning over Semantic IDs Enhances Generative Recommendation

**arXiv ID:** 2603.23183 | [PDF](https://arxiv.org/pdf/2603.23183v1)

**作者:** Yingzhi He `[一作]` (National University of Singapore), Tat-Seng Chua `[通讯]` (National University of Singapore)

**通讯引用:** 61239 | [OpenAlex ID](https://openalex.org/A5089404640)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两阶段框架SIDReasoner，先通过多任务与教师扩展强化Semantic ID（SID）与自然语言的对齐，再利用GRPO强化学习实现对SID的高质量推理；

**💡 创新点**

创新点在于：①利用多任务和教师生成的语义丰富语料实现SID–语言深度对齐；②引入基于结果的强化学习（GRPO）自我优化推理轨迹，无需昂贵的推理标签；

**🔧 技术方法**

技术包括：RQ‑VAE量化得到SID、预训练的Qwen3‑1.7B语言模型、多任务细调、GPT‑4o‑mini生成的增强语料、推理激活与Group Relative Policy Optimization（GRPO）；

**📊 数据集**

数据集为Amazon平台的三大领域：Video Games、Office Products、Industrial & Scientific，另外使用通用推理数据做对齐；

**📈 对比分析**

与传统判别式序列推荐（Caser、GRU4Rec、SASRec）、生成式推荐（TIGER、HSTU、LETTER、LCRec）及推理式推荐（ReaRec、R^2ec）对比，SIDReasoner在Recall@10/NDCG@10上均居首，且在跨域实验中表现出更强的泛化和可解释性；

**⚠️ 局限性**

局限性：对大规模预训练与算力依赖高；推理效果受域知识限制，工业领域收益有限；未对更大模型或更大数据规模进行充分验证，易出现过拟合与通用语言能力衰退。

---

## 468. UniDial-EvalKit: A Unified Toolkit for Evaluating Multi-Faceted Conversational Abilities

**arXiv ID:** 2603.23160 | [PDF](https://arxiv.org/pdf/2603.23160v1)

**作者:** Qi Jia `[一作]` (Shanghai Artificial Intelligence Laboratory), Guangtao Zhai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 10404 | [OpenAlex ID](https://openalex.org/A5043405654)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的多轮对话评估工具包 UniDial-EvalKit（UDE），实现了多种评测数据格式的标准化、评测流程的模块化以及统一的分数计算与聚合机制，并在多个公开多轮基准上对不同大型语言模型和记忆型代理进行了系统评测。

**💡 创新点**

核心创新包括：① 将多样的评测数据映射到统一的会话-轮次 schema；② 通过模块化设计拆分数据加载、模型生成、指标计算与结果聚合四个阶段，支持插件式扩展；③ 统一评分接口和多层次聚合策略（轮次→会话→数据集），消除不同基准间的聚合差异；④ 并行生成/评测与检查点恢复显著提升大规模评测效率。

**🔧 技术方法**

技术手段包括：Python API 与 CLI 并行框架，使用 vLLM/OpenAI API 接口对模型进行调用；使用多线程/多进程实现并行评测；通过评测注册表动态加载 LLM-as-Judge、Exact Match、Instruction Adherence 等多种指标；统一 schema 结构化数据；checkpoint‑based 缓存避免重复计算。

**📊 数据集**

使用了七个公开多轮评测数据集：MT‑Bench‑101、LoCoMo、PersonaMem、MultiIF、SafeDialBench、MathChat、MemoryCode；同时还对多种模型（DeepSeek‑V3.2、Qwen3‑Max‑Thinking、GLM‑5、MiniMax‑M2.5、Kimi‑K2.5）以及记忆型代理（HippoRAG、MemoryOS、A‑MEM）进行了评测。

**📈 对比分析**

评测方法：按轮次计算各指标分数，使用均值/最小值聚合；对每个会话再取最小值聚合；最终在数据集层面取平均得到模型总体表现。实验显示 Qwen3‑Max‑Thinking 在绝大多数基准上得分最高；GLM‑5 在记忆与编码方面突出；DeepSeek‑V3.2 在数学推理上略有优势。多轮评测中发现一些基准的区分度（discriminability）较低，说明当前数据集在区分先进模型时存在不足。

**⚠️ 局限性**

局限性包括：① 多轮数据集存在冗余与污染，导致区分度不足；② 记忆型代理在短对话中的效果不稳定；③ 评测只覆盖文本对话，缺乏多模态与用户模拟器；④ 对长文本的生成长度限制可能影响模型真实表现；⑤ 目前聚合策略默认最小值，可能掩盖部分模型在后续轮次的改进。

---

## 469. A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control

**arXiv ID:** 2603.23173 | [PDF](https://arxiv.org/pdf/2603.23173v1)

**作者:** Louis Claeys `[一作]` (ETH Zürich), Niao He `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在梯度漂移下的高维随机最优控制问题，并通过将HJB线性化后与Schrödinger算子谱理论关联，得到长期最优控制可由顶级特征函数给出；

**💡 创新点**

提出新的相对损失函数消除传统特征函数学习方法中的隐式重加权问题，使得神经网络能高效逼近顶级特征函数；并提出混合方法将该特征函数控制与短时期的IDO/FBSDE结合，显著降低随时间跨度增长的计算成本；

**🔧 技术方法**

利用深度神经网络学习特征函数，构造相对损失；使用PINN、变分损失进行对比；在混合框架中结合IDA/FBSDE优化修正项；

**📊 数据集**

在人工构造的高维长时段基准（正交/非正交二次势、双井、环形势、意见形成模型）上进行实验；

**📈 对比分析**

与现有IDO、FBSDE等方法比较，在20维长时段问题上平均L²误差降低约一十倍，且在意见形成任务中控制目标更低；

**⚠️ 局限性**

仅适用于梯度漂移且对称的情形；对切换阈值T_cut的选取缺乏理论指导，需经验调优；

---

## 470. RF-Zero-Wire: Design and Analysis of Multi-Hop Low-latency Symbol-synchronous RF Communication

**arXiv ID:** 2603.23213 | [PDF](https://arxiv.org/pdf/2603.23213v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 471. A Latency Coding Framework for Deep Spiking Neural Networks with Ultra-Low Latency

**arXiv ID:** 2603.23206 | [PDF](https://arxiv.org/pdf/2603.23206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 472. PhysSkin: Real-Time and Generalizable Physics-Based Animation via Self-Supervised Neural Skinning

**arXiv ID:** 2603.23194 | [PDF](https://arxiv.org/pdf/2603.23194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 473. Dual Contrastive Network for Few-Shot Remote Sensing Image Scene Classification

**arXiv ID:** 2603.23161 | [PDF](https://arxiv.org/pdf/2603.23161v1)

**作者:** Zhong Ji `[一作]` (Tianjin University), Yanwei Pang `[通讯]` (Tianjin University)

**通讯引用:** 13273 | [OpenAlex ID](https://openalex.org/A5086887025)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Dual Contrastive Network (DCN)，通过 Context-guided Contrastive Learning (CCL) 与 Detail-guided Contrastive Learning (DCL) 两个分支，在预训练阶段利用监督对比学习同时提取全局上下文和局部细节特征，以解决遥感图像场景分类中类间方差小、类内方差大的问题。

**💡 创新点**

创新点在于：① 采用双分支对比学习框架，使模型能够同时学习上下文驱动的判别特征和细节驱动的不变特征；② 设计 Condenser Network 与 Smelter Network 分别聚焦全局上下文与局部细节，并将两者与监督对比学习相结合，形成互补的特征表示；③ 在预训练阶段将分类任务与对比学习联合训练，实现更鲁棒的特征表达。

**🔧 技术方法**

使用的技术包括：ResNet-12 作为骨干网络；Condenser Network（Squeeze‑Expand 操作）与 Smelter Network（通道/空间注意力）构建上下文/细节特征；监督对比学习（Supervised Contrastive Loss）与自注意力特征对齐；多任务学习（交叉熵 + 对比损失）以及数据增强、Momentum‑SGD 等。

**📊 数据集**

在四个公开遥感数据集上评估：WHU‑RS19、UC Merced、NWPU‑RESISC45、AID。

**📈 对比分析**

与 9 种典型少样本学习方法和 6 种最新遥感少样本方法对比，DCN 在 5‑way 1‑shot 与 5‑way 5‑shot 场景下均取得最优或相近最优结果；在大型数据集上相较第二佳方法提升 3.56% 以上（NWPU‑RESISC45）和 5.38% 以上（AID），并在小型数据集上同样保持领先。

**⚠️ 局限性**

局限性：在极少样本（1‑shot）任务中仍略低于部分先进方法，且对训练集类别样本量不足时会出现分类偏差；模型在预训练阶段引入额外的对比学习分支，导致参数量和训练时间略高于纯 ResNet‑12。

---

## 474. Neural ODE and SDE Models for Adaptation and Planning in Model-Based Reinforcement Learning

**arXiv ID:** 2603.23245 | [PDF](https://arxiv.org/pdf/2603.23245v1)

**作者:** Chao Han `[一作]`, Eleni Vasilaki `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提供了TMLR投稿的格式模板和排版说明。

**💡 创新点**

强调严格遵循模板以避免拒稿。

**🔧 技术方法**

使用LaTeX、tmlr宏包、OpenReview提交。

**📊 数据集**

无实际数据集。

**📈 对比分析**

无实验方法和性能评估。

**⚠️ 局限性**

缺乏研究内容，仅为格式指南。

---

## 475. Gaze-Regularized VLMs for Ego-Centric Behavior Understanding

**arXiv ID:** 2603.23190 | [PDF](https://arxiv.org/pdf/2603.23190v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 476. Gyokuro: Source-assisted Private Membership Testing using Trusted Execution Environments

**arXiv ID:** 2603.23226 | [PDF](https://arxiv.org/pdf/2603.23226v1)

**作者:** Yoshimichi Nakatsuka `[一作]` (ETH Zurich), Srdjan Capkun `[通讯]` (ETH Zurich)

**通讯引用:** 18490 | [OpenAlex ID](https://openalex.org/A5077290467)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出基于可信执行环境的源辅助私有成员测试协议，允许客户端在不泄露查询内容的前提下验证数据项是否已被记录。

**💡 创新点**

创新点在于通过TEE产生的计数器和哈希链证明处理进度，而非传统的包含证明，消除对复杂加密结构的需求，保持查询不依赖数据库大小。

**🔧 技术方法**

采用TEE（如Intel SGX/AMD‑SEV）实现可信签名、计数器、哈希链；使用ECDSA签名、SHA‑256哈希，结合监视器的累积根。

**📊 数据集**

使用真实的证书透明度日志（约330万条TLS证书）作为测试数据集；还在实验中模拟供应链、内容真实性等场景。

**📈 对比分析**

与现有PMT方案对比，上传和查询延迟均低于7 ms，吞吐量可达约1400 req/s/核心；实验在Azure云上实现，证明对大规模日志可扩展。

**⚠️ 局限性**

局限在于依赖TEE，若TEE实现被攻击或未来出现完整性缺陷则失效；目前未提供TEE‑free方案，且需要第三方监视器同步。

---

## 477. Algorithms and Hardness for Geodetic Set on Tree-like Digraphs

**arXiv ID:** 2603.23193 | [PDF](https://arxiv.org/pdf/2603.23193v1)

**作者:** Florent Foucaud `[一作]` (Université Clermont Auvergne, CNRS, Clermont Auvergne INP, Mines Saint-Étienne, LIMOS), Prafullkumar Tale `[通讯]` (Indian Institute of Science Education and Research)

**通讯引用:** 112 | [OpenAlex ID](https://openalex.org/A5035477873)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究有向图中的Geodetic Set问题，提出了对ditree的线性时间算法、在无2-环且底层无向图反馈边数有限的情况下的FPT算法，并证明了在DAG上即使反馈顶点数受限仍为NP‑hard。

**💡 创新点**

创新点在于：①给出ditree的O(n)算法，解决此前仅对无向树已知的最小Geodetic Set问题；②在无2-环的情况下利用基图与核心路径结构，得到2^O(feedback‑edge‑set)·n^O(1)的算法，避免了先前需要ILP的复杂实现；③通过3D‑Matching构造证明，即使底层无向图的反馈顶点数为常数，DAG仍保持NP‑hard，显著加强了之前的结果。

**🔧 技术方法**

技术主要包括：结构化归约（将ditree收缩为只含与叶相连的2-环的收缩ditree）；基图与核心路径的分解；动态规划与分支搜索；参数化复杂度框架下的集合枚举；以及从3D‑Matching的多项式还原构造。

**📊 数据集**

本文为理论算法研究，未使用公开实验数据集，而是通过构造性证明和理论分析得出算法复杂度与硬度结论。

**📈 对比分析**

方法评估基于理论复杂度分析：对ditree给出O(n)时间，针对反馈边数k给出2^O(k)·n^O(1)时间；相对之前的ILP方法大幅提升；在DAG上证明的NP‑hard性表明该问题在该参数化下无多项式时间算法。

**⚠️ 局限性**

局限性在于：①算法仅适用于无2-环的digraph；②对反馈顶点数的FPT性未得到；③缺乏实验验证与实际数据集上的表现；④未讨论核化或更精细的参数化下的更优算法。

---

## 478. An Explainable AI-Driven Framework for Automated Brain Tumor Segmentation Using an Attention-Enhanced U-Net

**arXiv ID:** 2603.23344 | [PDF](https://arxiv.org/pdf/2603.23344v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 479. Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors

**arXiv ID:** 2603.23324 | [PDF](https://arxiv.org/pdf/2603.23324v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 480. ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting

**arXiv ID:** 2603.23186 | [PDF](https://arxiv.org/pdf/2603.23186v1)

**作者:** Yeonkyung Lee `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**通讯引用:** 22213 | [OpenAlex ID](https://openalex.org/A5051395190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种无需训练、可插拔的视觉提示与关键词‑帧映射框架（ViKey），通过在视频帧中叠加帧编号提示并将文本关键词映射到对应帧，显著提升视频大语言模型的时序推理能力。

**💡 创新点**

1) 使用帧编号视觉提示恢复稀疏采样下的时序连续性；2) 将帧编号视作字典键，构建轻量级关键词‑帧映射（KFM）实现显式时序锚定；3) 该方案训练免费、可迁移到多种 VideoLLM，兼顾性能与效率。

**🔧 技术方法**

视觉提示（在帧像素中叠加编号文字）、关键词提取（LLM）、视觉‑文本对齐模型（CLIP）实现关键词‑帧相似度计算、文本提示插入映射索引、对模型注意力的层级分析等。

**📊 数据集**

TempCompass、MVBench、VideoMME、LongVideoBench 四大时序推理基准；同时在这些数据集上对比多种基线模型。

**📈 对比分析**

与 GPT‑4.1、Qwen2.5‑VL‑7B、LLaVA‑Video‑7B、LLaVA‑OneVision‑7B 等基线及 Token‑Fusion 等效率方法进行对比；在 20% 帧采样下提升 2–10+ 分；在 64 帧稠密输入时亦可保持或超过稠密基线；在多种采样比例下始终优于基线。

**⚠️ 局限性**

对极低帧率或帧内信息严重缺失的视频仍有局限；视觉提示的可视化效果依赖背景对比；KFM 需要阈值调参；在需要细粒度时序关系的长视频中可能仍需更高级时序建模。

---

## 481. Canonical Byte-String Encoding for Finite-Ring Cryptosystems

**arXiv ID:** 2603.23364 | [PDF](https://arxiv.org/pdf/2603.23364v1)

**作者:** Kyrylo Riabov `[一作]` (Taras Shevchenko National University of Kyiv), Serhii Kryvyi `[通讯]` (Taras Shevchenko National University of Kyiv)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种适用于有限域环加密协议的字节到剩余元的标准化编码器（codec），并给出了完整的编码/解码流程与实现；

**💡 创新点**

创新点在于：①设计了基于固定宽度的 base‑m 长度与状态前缀，实现自定界且可切片的字节映射；②使用 rANS/ANS 统一字节转化实现可逆且可尾缀容忍的负载；③通过 Lean 4 进行机器可检验的形式化证明，保证了固定宽度逆转、状态范围及端到端正确性；

**🔧 技术方法**

主要技术包括：rANS/ANS 的状态重归化算法、Little‑Endian base‑m 编码、逆向扫描的字节更新、Rust 编程实现与 Criterion 性能基准；

**📊 数据集**

未使用传统数据集，而是通过在 Apple M3 Pro 上对 1 KiB 与 64 KiB 的随机字节流进行基准测试（以及与 BigUint 及 ANS 参考实现对比）；

**📈 对比分析**

与 BigUint 纯基数转换基准和外部 ANS 参考实现比较，结果表明：在 1 KiB 和 64 KiB 规模下，codec 的编码/解码速度均优于 BigUint，且在大模数（m=257）下接近每字节 1 个剩余元；整体复杂度为 O(n) 并保持 O(1) 运行时状态；

**⚠️ 局限性**

局限性包括：①纯粹是确定性序列化，无法提供保密性/完整性；②长度前缀暴露，若需长度隐藏需在协议层做填充；③Lean 4 证明仅覆盖抽象规范，未对 Rust 低层实现逐行验证；④解码器需根据嵌入长度预分配，攻击者需受限于最大长度。

---

## 482. Let Functions Speak: Lightweight Parametric Polymorphism via Domain and Range Types

**arXiv ID:** 2603.23360 | [PDF](https://arxiv.org/pdf/2603.23360v1)

**作者:** Siyuan He `[一作]`, Tiark Rompf `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了一个在系统F上扩展的带有域和范围投影的类型系统，能够安全地在抽象函数类型上进行应用，模仿TypeScript的Parameters/ReturnType语义；

**💡 创新点**

通过引入域/范围投影类型实现了对抽象函数调用的静态检查，既不需要显式的箭头类型，又兼容传统的函数应用规则；

**🔧 技术方法**

使用了带路径选择的逻辑关系（logical relations）证明语义类型安全，并在Coq中完成了完整的机械化证明；

**📊 数据集**

本工作不依赖任何具体数据集，主要通过形式化证明与案例演示来验证其正确性；

**📈 对比分析**

通过与标准System F的形式化对比，证明了类型安全与弱归约性，未进行运行时性能评测；

**⚠️ 局限性**

局限性包括仅覆盖函数投影扩展，未处理所有TypeScript特性，且对结构暴露仍需显式边界约束。

---

## 483. Robustness Quantification for Discriminative Models: a New Robustness Metric and its Application to Dynamic Classifier Selection

**arXiv ID:** 2603.23318 | [PDF](https://arxiv.org/pdf/2603.23318v1)

**作者:** Rodrigo F. L. Lassance `[一作]` (Ghent University), Jasper De Bock `[通讯]` (Ghent University)

**通讯引用:** 500 | [OpenAlex ID](https://openalex.org/A5052699188)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种新的鲁棒性度量，适用于任何概率判别式分类器和任意类型特征，并利用该度量进行动态分类器选择。

**💡 创新点**

创新点在于：①用 Constant Odds Ratio（COR）扰动取代传统的 ε-污染，实现对连续/混合特征的鲁棒度量；②得到闭式表达式，避免了优化问题；③将鲁棒度量直接用于构造两种动态选择策略（RS‑D、RS‑I），在标签污染和分布漂移场景下取得优于现有动态选择方法的性能。

**🔧 技术方法**

技术包括：概率分布的密度表示、COR 相关的距离函数、Likelihood Ratio 计算、Accuracy Rejection Curve（ARC）评估、基于阈值的动态选择算法，以及对 15 个公开数据集的交叉验证实验。

**📊 数据集**

使用了 15 个公开数据集（OpenML、UCI、PMLB），覆盖离散、连续及混合特征，样本数从 569 到 45312，特征维度从 4 到 40。

**📈 对比分析**

通过 Accuracy Rejection Curves 对鲁棒度量与预测准确率的相关性进行评估；与生成森林（GeF）鲁棒度量比较，r_d_COR 在连续特征上显著优于 r_GeF；在动态选择实验中，RS‑D 与 RS‑I 的平均准确率均高于单一最佳模型和 4 种主流动态选择方法，尤其在标签污染和分布漂移情境下表现更佳。

**⚠️ 局限性**

局限性：①鲁棒度量依赖于选择的距离函数（如 d_COR），其泛化性仍需进一步研究；②对生成模型的依赖不如直接的局部鲁棒度量灵活；③虽然公式闭式，但在极端分布不匹配或高维连续特征时，鲁棒度量的解释性和稳健性仍待验证。

---

## 484. Learning Multi-Agent Local Collision-Avoidance for Collaborative Carrying tasks with Coupled Quadrupedal Robots

**arXiv ID:** 2603.23278 | [PDF](https://arxiv.org/pdf/2603.23278v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 485. Mamba-driven MRI-to-CT Synthesis for MRI-only Radiotherapy Planning

**arXiv ID:** 2603.23295 | [PDF](https://arxiv.org/pdf/2603.23295v1)

**作者:** Konstantinos Barmpounakis `[一作]` (National Technical University of Athens), George K. Matsopoulos `[通讯]` (National Technical University of Athens)

**通讯引用:** 3550 | [OpenAlex ID](https://openalex.org/A5087972272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了基于 Mamba 的架构用于 MRI 到 CT 的合成，旨在支持 MRI‑only 放疗计划

**💡 创新点**

首次将 U‑Mamba 与 SegMamba 迁移到生成任务，并提出针对 HU 区域的加权 MAE 与 AFP 结构损失

**🔧 技术方法**

Mamba 状态空间模块、U‑Mamba、SegMamba、nnU‑Net、SwinUNETR 以及联合损失（wMAE+SSIM+AFP）

**📊 数据集**

SynthRAD2025 子集（890 对 MRI‑CT，三解剖部位，训练集 461 样本，测试集 52 样本）

**📈 对比分析**

与 U‑Net、SwinUNETR、nnU‑Net 对比，SegMamba 在 MAE 方面最低；nnU‑Net 在 PSNR、MS‑SSIM 与几何一致性指标最高；Mamba 模型在视觉一致性与训练效率上表现更好

**⚠️ 局限性**

受限于训练轮次与模型规模；几何一致性指标仍落后于 nnU‑Net；未针对各器官进一步细化模型

---

## 486. LLM Olympiad: Why Model Evaluation Needs a Sealed Exam

**arXiv ID:** 2603.23292 | [PDF](https://arxiv.org/pdf/2603.23292v1)

**作者:** Jan Christian Blaise Cruz `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Alham Fikri Aji `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 1687 | [OpenAlex ID](https://openalex.org/A5112924039)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向大语言模型的“LLM奥林匹克”评测框架，强调任务封闭、提交冻结、统一评测环境以及评测后公开所有评测材料；

**💡 创新点**

将任务封闭、提交冻结与统一评测 harness 三者结合，形成一次性高可信度的“考试”式评测机制，旨在降低评测作弊与优化空间；

**🔧 技术方法**

核心技术包括：公开规则与预算、任务提交与严格封闭、提交冻结机制、单一评测 harness 与标准化日志、评测完成后开放任务与代码；

**📊 数据集**

本文主要引用现有公开基准（如 GLUE、MMLU、GPTQA、GSM8k 等）作为比较与动机，并未在实验中使用新的数据集；

**📈 对比分析**

评测方法通过统一 harness 对所有提交进行同一条件下的评测，保证结果可比；虽然没有给出具体实验数据，但作者指出在此框架下模型排名更稳健，且更易被社区复核；

**⚠️ 局限性**

局限性包括：仍可能出现训练数据泄漏、任务集规模有限导致的代表性不足、组织与治理成本高、闭源端点难以完全审核、以及对已有基准的替代性仍有限。

---

## 487. PinPoint: Monocular Needle Pose Estimation for Robotic Suturing via Stein Variational Newton and Geometric Residuals

**arXiv ID:** 2603.23365 | [PDF](https://arxiv.org/pdf/2603.23365v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 488. WaveSFNet: A Wavelet-Based Codec and Spatial--Frequency Dual-Domain Gating Network for Spatiotemporal Prediction

**arXiv ID:** 2603.23284 | [PDF](https://arxiv.org/pdf/2603.23284v1)

**作者:** Xinyong Cai `[一作]` (Sichuan University), Yuankai Wu `[通讯]` (Sichuan University)

**通讯引用:** 4335 | [OpenAlex ID](https://openalex.org/A5100370856)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了WaveSFNet，一种结合小波编码器与空间-频域双域门控网络的无递归时空预测框架。

**💡 创新点**

创新点包括：①采用小波变换保留高频细节；②双域门控译码器同时在空间局部与频域全局双路提取信息；③在粗尺度潜在空间中注入时间差提升对短时动态的感知。

**🔧 技术方法**

使用的技术包括Haar小波变换（DWT/IDWT）、深度可分离卷积、傅里叶频域调制、门控通道交互、时间差注入以及整体无递归的编码-译码结构。

**📊 数据集**

实验数据集涵盖Moving MNIST、TaxiBJ交通流、WeatherBench的单变量（T2M、UV10、TCC、R）以及多变量（MV）天气数据。

**📈 对比分析**

与多种基准（如ConvLSTM、PredRNN、SimVP、MogaNet等）在MSE/SSIM/MAE等指标上进行对比，WaveSFNet在Moving MNIST实现MSE 15.8/SSIM 0.966、TaxiBJ实现MSE 0.287/SSIM 0.9859、WeatherBench多变量实现MSE 96.141/MAE 5.255，均优于或与现有方法竞争，同时参数量与算力更低。

**⚠️ 局限性**

局限性在于对极端高分辨率或长时序数据仍受小波级数限制；缺乏显式物理先验；频域门控机制的可解释性尚待进一步研究。

---

## 489. On the Vulnerability of FHE Computation to Silent Data Corruption

**arXiv ID:** 2603.23253 | [PDF](https://arxiv.org/pdf/2603.23253v1)

**作者:** Jianan Mu `[一作]` (Institute of Computing Technology), Huawei Li `[通讯]` (Institute of Computing Technology)

**通讯引用:** 4486 | [OpenAlex ID](https://openalex.org/A5100768288)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对 CKKS 方案下的全同态加密（FHE）计算中出现的无声数据腐败（SDC）进行了系统评估，结合大规模单比特瞬态故障注入实验、误差传播理论分析以及冗余和校验两类算法级容错机制的评测，揭示了 FHE 对硬件故障的高脆弱性。

**💡 创新点**

创新点在于：①首次在 CKKS 上做大规模 SDC 注入实验并量化其影响；②从错误传播角度揭示了 ciphertext、多项式与 slot 级联放大的结构机制；③将传统的双模冗余（DMR）与基于校验和的 ABFT 方案迁移到 FHE 计算流水线中，并给出性能与可靠性的权衡。

**🔧 技术方法**

主要技术包括：单比特随机翻转故障模型、Intel Pin 动态注入、CKKS 基本算子（ct‑pt/mul、ct‑ct/mul、ct‑pt/add、ct‑ct/add、ct‑rot）以及其内部七步多项式操作的分析、冗余执行与校验和检查（NTT、BConv、DCRT 等），并通过 OpenFHE 库实现实验。

**📊 数据集**

使用的数据集：Breast Cancer Wisconsin（用于评估加密神经网络推理的准确率退化），California Housing（用于回归任务测试），以及标准向量与矩阵乘法等基准工作负载。

**📈 对比分析**

对比方法：对照无容错、冗余、校验和三种模式。实验显示，未加保护的 CKKS 在单比特故障下 SDC 率约为 20%，冗余执行可将率降至 <0.1% 但延时约 1×，校验和方案将 SDC 率降至相似水平，且仅产生 13–16% 的运行时开销。

**⚠️ 局限性**

局限性：仅针对 CKKS 方案与单比特翻转模型，未考虑多比特或其他硬件错误类型；容错方案虽有效但仍带来显著性能负担；缺乏针对实际硬件环境的深度优化与轻量级容错设计。

---

## 490. Numerical Kernels on a Spatial Accelerator: A Study of Tenstorrent Wormhole

**arXiv ID:** 2603.23343 | [PDF](https://arxiv.org/pdf/2603.23343v1)

**作者:** Maya Taylor `[一作]` (University of Illinois Urbana-Champaign), Jan Ciesko `[通讯]` (Sandia National Laboratories)

**通讯引用:** 525 | [OpenAlex ID](https://openalex.org/A5089631372)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 Tenstorrent Wormhole 计算加速器上实现了三种数值内核（向量算术、全局归约、7 点三维差分算子）以及基于预条件共轭梯度（PCG）的求解器。

**💡 创新点**

首次将传统 HPC 算法映射到数据流型 AI 加速器，展示了内核融合与分裂实现、块级循环展开、以及针对稀疏矩阵的硬编码差分算子等架构专用优化。

**🔧 技术方法**

使用 Wormhole 的 tt‑metal 编程接口、FPU/SFPU 单指令多数据单元、循环缓冲区调度、数据块（tile）对齐与转置、以及核融合/分裂策略来实现数值算法。

**📊 数据集**

实验采用自定义的 3D 有限差分网格（如 512×112×64）及相应的稀疏矩阵硬编码，不使用公开数据集；性能测试在单个 Tensix 核片上进行。

**📈 对比分析**

与 Nvidia H100 GPU 通过相同的 PCG 任务（BF16 与 FP32 版本）比较；结果显示 Wormhole BF16 约 7 倍、FP32 约 16 倍慢于 H100，但功耗更低，且在 FPU 下可实现良好强/弱缩放。

**⚠️ 局限性**

受限于 16 位精度、缺乏 64 位支持、单芯片内存容量受限、缺乏通用稀疏矩阵格式、以及编程复杂度高等因素，影响了性能与通用性。

---

## 491. Can NR-V2X Sidelink support A2A links?

**arXiv ID:** 2603.23330 | [PDF](https://arxiv.org/pdf/2603.23330v1)

**作者:** Vittorio Todisco `[一作]` (Universita di Bologna), Alessandro Bazzi `[通讯]` (Universita di Bologna)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文评估并分析了 5G NR Release 18 Sidelink（即 NR‑V2X）在航空实体间（A2A）长距离通信（最高 150 km）中的可行性，重点研究了传播延迟、频率抖动、信道范围与资源分配等限制，并提出相应的缓解措施；

**💡 创新点**

创新点在于首次将 NR‑V2X 标准从陆地车辆扩展到空中实体，并通过系统化的传播模型与符号级时隙分析确定了 42.4 km 的关键距离阈值；此外，提出了通过扩展时间间隔、限制传输机会以及考虑替代波形（OTFS、OCDM 等）的方案；

**🔧 技术方法**

使用了自由空间路径损耗模型、SNR 计算公式、OFDM 时隙结构分析、Doppler 频移公式以及基于 NR‑Sidelink 资源调度与同步机制的理论推导；

**📊 数据集**

没有采用实验或公开数据集，而是基于理论模型和数值仿真（如 6 GHz/3 GHz/2 GHz 频率、100 MHz 带宽、不同 MCS/SNR 阈值等）构建了距离–功率与距离–SNR 的曲线；

**📈 对比分析**

通过比较不同频率、带宽与 SNR 阈值下的可达距离，发现 6 GHz 频率下 100 MHz 带宽、低 MCS 可达约 42.4 km；提高带宽或频率会进一步降低可达距离；整体性能表明在 42.4 km 以内可直接使用现有标准；

**⚠️ 局限性**

主要限制包括：传播延迟导致的时隙失配（超过 42.4 km 需在时隙中插入大量保留符号，极大降低传输机会）；对 GNSS/基站同步的依赖易受攻击和干扰；高频率下的 Doppler 影响与信道选择受限；标准在超长距离下缺乏足够的容错机制。

---

## 492. WISTERIA: Weak Implicit Signal-based Temporal Relation Extraction with Attention

**arXiv ID:** 2603.23319 | [PDF](https://arxiv.org/pdf/2603.23319v1)

**作者:** Duy Dao Do `[一作]` (University of Orléans), Thi-Bich-Hanh Dao `[通讯]` (University of Orléans)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了WISTERIA框架，通过关系条件化的Top-K注意力实现对实体对的上下文证据进行有针对性的选择，从而实现轻量级的时序关系抽取；

**💡 创新点**

创新点在于将注意力与实体对语义进行显式耦合，采用关系条件化的Top‑K注意力取代传统全局注意力；并引入结构化的语言学解释框架，系统评估所选证据在词性、依存关系与形态上的一致性；

**🔧 技术方法**

技术核心包括BERT-base预训练语言模型、轻量化Transformer编码层、注意力池化、关系条件化的多头交叉注意力、Top‑K稀疏采样、标签嵌入门控融合与biaffine分类器；

**📊 数据集**

在四个基准数据集上进行评测：TimeBank‑Dense、MATRES、TDDAuto和TDDMan；

**📈 对比分析**

与BiLSTM、BERT‑base、RoBERTa‑Large、TIMERS、DTRE、MuLCo、CPTRE等强基线对比，WISTERIA在句子级数据集上取得最高F1（TBD 0.831，MATRES 0.843），在文档级数据集保持竞争力（TDDAuto 0.709，TDDMan 0.497）；

**⚠️ 局限性**

局限性包括：仅对每对实体独立建模，缺乏全局时序一致性约束；Top‑K注意力只能捕获局部证据，无法处理跨句长距离推理；解释性仅限于分布层面，未提供因果证明；依赖自动化语言标注，可能带来噪声。

---

## 493. ARGENT: Adaptive Hierarchical Image-Text Representations

**arXiv ID:** 2603.23311 | [PDF](https://arxiv.org/pdf/2603.23311v1)

**作者:** Chuong Huynh `[一作]` (University of Maryland), Suren Kumar `[通讯]` (Samsung Research America)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出自适应蕴含损失（AdaEnt）和概率蕴含协议（PEP），用于改进超平面层次结构学习和评估；

**💡 创新点**

创新点包括：①自适应损失消除超球锥崩塌并直接最小化角度；②正则化项平衡稀疏空间；③将层次推理转化为概率分类的PEP评估；

**🔧 技术方法**

采用Lorentz超平面几何、Huber损失、norm正则化、HoroPCA可视化、双编码器架构等技术；

**📊 数据集**

使用GRIT、HierarCaps、WordNet、CC3M、ImageNet、Flickr30k、COCO等数据集；

**📈 对比分析**

与HyCoCLIP、MERU等基线在零样本分类、检索、WordNet层次指标及AUC‑ROC/AP上进行比较，ARGENT在所有指标上提升0.7–1.1分，特别是AUC‑ROC显著提高；

**⚠️ 局限性**

局限性在于需要正则化以避免空间过度稀疏，损失平衡较难，评估仍受数据噪声影响，缺乏大规模高质量层次基准。

---

## 494. Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression

**arXiv ID:** 2603.23308 | [PDF](https://arxiv.org/pdf/2603.23308v1)

**作者:** V. K. Cody Bumgardner `[一作]` (University of Kentucky), Evan W. Damron `[通讯]` (University of Kentucky)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Ker-VLJEPA‑3B 四阶段课程学习框架，利用语言无监督视觉编码与 Llama 3.2 3B 生成三维 CT 报告；

**💡 创新点**

创新点包括：完全无文本监督的视觉编码、按解剖区域压缩的交叉注意力、PCA whitening 对齐、仅正性发现训练以消除后验坍塌、热桥初始化提升跨阶段迁移、EWC 冻结防止灾难性遗忘；

**🔧 技术方法**

采用 LeJEPA ViT‑Large 自监督预训练、Llama 3.2 3B 加 LoRA、Flamingo‑style gated cross‑attention、JEPA embedding 预测、MMD 与 InfoNCE 对齐、PCA whitening、区间约束交叉注意力、四阶段课程学习；

**📊 数据集**

使用 CT‑RATE 公开基准（约46k 训练卷，2,984 验证卷）进行训练与评估；

**📈 对比分析**

按官方 RadBERT 标签提取法评估宏观 F1，Ker‑VLJEPA‑3B 在验证集上达到 0.429，超越 U‑VLM 0.414 (+3.6%)，阈值优化后可达 0.448；

**⚠️ 局限性**

局限性包括：仅在单一 CT‑RATE 评测，缺乏多中心/跨厂商验证；模型训练需大量 GPU，未实现端到端从原始体素学习；阈值优化存在数据泄露；未做临床放射员评估等。

---

## 495. Security Barriers to Trustworthy AI-Driven Cyber Threat Intelligence in Finance: Evidence from Practitioners

**arXiv ID:** 2603.23304 | [PDF](https://arxiv.org/pdf/2603.23304v1)

**作者:** Emir Karaosman `[一作]` (University of Liechtenstein), Irdin Pekaric `[通讯]` (University of Liechtenstein)

**通讯引用:** 246 | [OpenAlex ID](https://openalex.org/A5029056103)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过系统综述、访谈与问卷，结合混合方法研究，探讨金融机构如何在CTI中使用AI以及信任部署的障碍。

**💡 创新点**

首次系统识别出四种社会技术失效模式，并提出三项安全操作防护措施，为可信AI驱动CTI的实践提供操作化指引。

**🔧 技术方法**

采用混合方法：系统性文献综述、半结构化访谈、探索性问卷，以及对AI模型安全性的定性分析。

**📊 数据集**

文献综述检索330篇论文（2019–2025）筛选12篇金融相关研究，访谈6名专家，问卷收集14份银行与咨询机构回应。

**📈 对比分析**

该研究未进行模型性能对比，而是通过调查得到AI在未来五年内将成为核心的预期、使用频率低的解释性与保障问题以及28.6%遭遇对抗性风险的统计。

**⚠️ 局限性**

局限性包括样本量有限、缺乏定量性能评估、研究主要依赖自我报告数据，难以全面验证提出的防护措施效果。

---

## 496. SynForceNet: A Force-Driven Global-Local Latent Representation Framework for Lithium-Ion Battery Fault Diagnosis

**arXiv ID:** 2603.23265 | [PDF](https://arxiv.org/pdf/2603.23265v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 497. Emergence of Fragility in LLM-based Social Networks: the Case of Moltbook

**arXiv ID:** 2603.23279 | [PDF](https://arxiv.org/pdf/2603.23279v1)

**作者:** Luca Sodano `[一作]` (LIUC Università Cattaneo), Francesco Bertolotti `[通讯]` (LIUC Università Cattaneo)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对由LLM智能体构成的社交平台Moltbook的交互网络进行大规模网络科学分析，研究其结构特征、核心–边缘组织和鲁棒性。

**💡 创新点**

首次将核心–边缘分析与随机与靶向节点剔除实验相结合，揭示了AI本土社交网络的高度集中化与结构脆弱性。

**🔧 技术方法**

使用度/强度分布、k‑core分解、Borgatti–Everett核心‑边缘拟合、最大似然幂律拟合，以及针对随机与按度数（入/出）剔除的鲁棒性实验。

**📊 数据集**

基于对Moltbook的网页抓取得到的 39,924 位用户、235,572 条帖子和 1,540,238 条评论构成的数据集。

**📈 对比分析**

通过比较随机删节点与按入度/出度排序的靶向删节点，发现网络对随机失效相对鲁棒（20%节点删后GCC仍占78%），但对高出度节点攻击极为脆弱（20%节点删后GCC仅剩15%）。核心仅占0.9%节点，表明系统高度集中。

**⚠️ 局限性**

局限性包括仅研究单一平台的静态快照，缺乏时间演化分析；无法解释为何少数代理占据核心；数据来源仅为公开抓取，可能存在缺失或偏差；模型仅聚焦结构，未考察信息传播或行为机制。

---

## 498. Drop-In Perceptual Optimization for 3D Gaussian Splatting

**arXiv ID:** 2603.23297 | [PDF](https://arxiv.org/pdf/2603.23297v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 499. Systemic Gendered Citation Imbalance in Computer Science: Evidence from Conferences and Journals

**arXiv ID:** 2603.23273 | [PDF](https://arxiv.org/pdf/2603.23273v1)

**作者:** Kazuki Nakajima `[一作]` (Tokyo Metropolitan University), George Fletcher `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 2637 | [OpenAlex ID](https://openalex.org/A5019818981)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构建包含 394,432 篇会议与期刊论文及 752,742 条引用的计算机科学引用网络，系统评估了作者性别对引用分布的影响。

**💡 创新点**

创新点在于：①首次将会议式出版文化纳入性别引用不平衡研究；②提出三种参考模型（随机抽取、同质性抽取、偏好抽取）来分别剔除网络结构和热门度因素；③利用匹配对分析揭示作者性别、会议级别、显著性作者与合作者网络对性别化引用的交互作用。

**🔧 技术方法**

主要技术包括：基于 DBLP 与 OpenAlex 的元数据融合；采用 Levenshtein 距离匹配论文；使用 Gender API 进行二元性别推断；构建三类引用重排模型并进行 100 次随机化；匹配对分析与 Z 统计检验。

**📊 数据集**

使用的数据集为：① 7,034,299 篇 DBLP 论文（1950‑2024）与 155,449,238 篇 OpenAlex 论文；② 2021 年 CORE 会议排名与 SCImago 期刊排名；③ 通过 Gender API 为 1,257,016 位作者分配性别；最终筛选得到 394,432 篇带性别与排名信息的论文。

**📈 对比分析**

比较方法：在三种参考模型下计算每种性别组合的引用偏差（over/under‑citation）并与观测值比较；匹配对分析控制年份、国家、子领域与引用数，检验性别、会议级别、显著性作者与合作者网络对引用差异的影响。结果显示：女性作者在会议论文中被引用显著不足，尤其是顶级会议；显著性作者与男性合作者网络会放大此不平衡；但偏好模型表明，除同质性外，热门度对性别差异影响有限。

**⚠️ 局限性**

局限性包括：① 性别推断仅限二元分类，导致中国等东亚姓名识别率低；② 假设第一/末位作者承担主要角色，忽略某些子领域的按字母排序；③ 未考虑时间变化的期刊/会议声望；④ 只关注性别，未探究国籍、种族等其他身份维度；⑤ 研究聚焦于已发表且被排名的文献，可能低估低影响力或非英文论文的情况。

---

## 500. Ellipsoidal Manifold Optimization for Distributed Antenna Beamforming

**arXiv ID:** 2603.23260 | [PDF](https://arxiv.org/pdf/2603.23260v1)

**作者:** Minhao Zhu `[一作]` (Future Network of Intelligence Institute), Kaiming Shen `[通讯]` (Future Network of Intelligence Institute)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

针对分布式天线系统的每簇功率约束下，求解加权总速率最大化问题，并通过低维子空间性质将原始高维优化转化为球面到椭圆流形的无约束问题，随后设计了基于黎曼共轭梯度法的低维流形优化算法。

**💡 创新点**

创新点包括：① 将低维子空间性质从总功率约束推广到每簇功率约束；② 证明所有驻点可写成每簇天线信道子空间的线性组合，从而将球面约束转化为椭圆流形约束；③ 统一推导椭圆流形的切空间、度量、投影、重排与向量传输，并在此基础上提出 Hestenes–Stiefel 更新与 Armijo 步长搜索的黎曼共轭梯度算法。

**🔧 技术方法**

采用黎曼几何工具（切空间投影、重排、向量传输）、共轭梯度优化、Hestenes–Stiefel 方向更新、Armijo 回溯线搜索、以及低维子空间投影技术。

**📊 数据集**

实验使用仿真数据：单小区 Massive MIMO 下行链路，十个簇 (C=4 或 8)，每簇 128 个天线，6 个用户，4 个接收天线，4 条数据流；路径损耗、阴影、Rayleigh 衰落模型，1000 次独立仿真。

**📈 对比分析**

与 WMMSE、RWMMSE、传统黎曼共轭梯度和 EZF 线性预编码进行比较；结果显示算法能够达到与 WMMSE、传统黎曼共轭梯度相同的局部最优，但迭代时间显著更短；在固定 2.5 秒预算下，CDF 也明显优于对手；随着簇数增大或每簇天线数增多，时间优势更为突出。

**⚠️ 局限性**

局限性包括：仅在假设信道矩阵满行秩且每簇功率约束紧束的情形下证明；算法仅保证收敛到驻点，无法保证全局最优；仿真仅覆盖理想化的 Rayleigh 衰落环境，未验证在实际硬件噪声、CSI 不完全等真实场景下的鲁棒性。

---

## 501. Autoencoder-based Optimization of Multi-user Molecule Mixture Communication Systems

**arXiv ID:** 2603.23262 | [PDF](https://arxiv.org/pdf/2603.23262v1)

**作者:** Bastian Heinlein `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Maximilian Schäfer `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 8288 | [OpenAlex ID](https://openalex.org/A5048065270)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文提出一种基于自编码器的端到端优化方法，用于多用户分子混合通信系统，利用非线性、交叉反应传感器实现符号传输与解码。

**💡 创新点**

创新点在于将混合分子编码、传输过程和接收器解码全部统一在自编码器框架内学习，并通过权重交叉熵实现多用户优先级控制，显著提高了在未知或变化信道条件下的符号误码率。

**🔧 技术方法**

主要技术包括自编码器网络（每个发送器一个编码器、接收器一个解码器）、梯度下降训练（Adam优化器）、对非线性传感器的可微分近似以及加权交叉熵损失。

**📊 数据集**

使用的是基于文献中MOS传感器模型和噪声参数生成的合成数据集，包含三种分子类型、两种传感器和多种信噪比。

**📈 对比分析**

通过与文献中的MDA+AML混合字典优化基线、单用户CSK/GMoSK基线进行对比，AE方案在所有信噪比、信道衰减范围以及多用户优先级场景下均实现了更低的符号误码率。

**⚠️ 局限性**

局限性包括仅使用合成数据，未验证真实硬件；未考虑信号相关噪声和传感器失效；仅针对单一接收器，缺乏多接收器扩展。

---

## 502. Central Dogma Transformer III: Interpretable AI Across DNA, RNA, and Protein

**arXiv ID:** 2603.23361 | [PDF](https://arxiv.org/pdf/2603.23361v1)

**作者:** Nobuyuki Ota `[一作]` `[通讯]` (Independent Researcher), Nobuyuki Ota (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一种两阶段、可解释的人工智能模型（CDT‑III），能够通过模拟 DNA→RNA→蛋白质的信息流，联合预测 CRISPRi 诱导的 mRNA 与表面蛋白质表达变化。

**💡 创新点**

创新点在于：①将细胞核与细胞质分为两个模块，严格遵循细胞空间与功能分化；②下游蛋白质预测任务可正向约束上游 RNA 与 DNA 表示，实现多任务正则化；③利用梯度分析即可在无新实验数据的前提下预测药物潜在副作用。

**🔧 技术方法**

技术手段包括：Enformer 提取 DNA 片段嵌入、两阶段 Virtual Cell Embedder（自注意力与跨注意力）、多任务损失（RNA + 低权重蛋白质）、梯度可解释性分析及两相训练策略。

**📊 数据集**

使用了 STING‑seq v2 K562 单细胞 CRISPRi 数据集，包含 2361 条基因的 scRNA‑seq 与 193 个表面蛋白的 CITE‑seq 结果。

**📈 对比分析**

与先前 CDT‑II 及单阶段模型对比，CDT‑III 在 5 个 held‑out 基因上取得 per‑gene RNA Pearson r 0.843（提升 4.9%）和蛋白质 r 0.969；在 CD52 降解案例中实现 29/29 方向一致的副作用预测，蛋白质监督还使 DNA 级别 CTCF 富集提升 30%。

**⚠️ 局限性**

局限性包括：仅在 K562 细胞系验证，CRISPRi 仅近似抗体机制；单细胞蛋白水平 r 仅 0.28；梯度解释性在多样化细胞类型与更大样本上尚未充分验证；以及对非肿瘤相关副作用的预测精度受限。

---

## 503. Modeling Edge-to-Cloud Offloading Workloads for Autonomous Vehicles

**arXiv ID:** 2603.23310 | [PDF](https://arxiv.org/pdf/2603.23310v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 504. A Comparative Study of Machine Learning Models for Hourly Forecasting of Air Temperature and Relative Humidity

**arXiv ID:** 2603.23282 | [PDF](https://arxiv.org/pdf/2603.23282v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 505. Design Space and Implementation of RAG-Based Avatars for Virtual Archaeology

**arXiv ID:** 2603.23353 | [PDF](https://arxiv.org/pdf/2603.23353v1)

**作者:** Wilhelm Kerle-Malcharek `[一作]` (University of Konstanz), Falk Schreiber `[通讯]` (University of Konstanz)

**通讯引用:** 7686 | [OpenAlex ID](https://openalex.org/A5003441276)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一款基于检索增强生成（RAG）的虚拟现实（VR）头像，用于Maxentius陵墓的数字考古探究。

**💡 创新点**

创新点包括：提出了针对文化遗产VR头像的需求设计空间，融合RAG与对话头像，利用元数据驱动检索提升答案质量，并引入LLM-as-Judge评估方法与NASA‑TLX工作量研究。

**🔧 技术方法**

技术实现基于Unity 3D、OpenAI Whisper STT、Piper TTS、Qdrant向量数据库、FlowiseAI工作流、LLM模型（Llama 3.1、Qwen3、Llama 3.3）、LLM‑Graph‑Builder构建知识图谱以及Prometheus2做评判。

**📊 数据集**

数据集由Maxentius陵墓相关的学术论文与资料构成，并人工标注作者、标题、出版类型及相关性；另外使用10条专家级问答对作为评测基准。

**📈 对比分析**

通过对七种RAG配置（不同嵌入/生成模型、是否加入相关性元数据或知识图谱）进行METEOR、BERTScore及LLM‑as‑Judge评估，最佳配置（Qwen3+Llama 3.3+相关性元数据）取得3.42分；NASA‑TLX显示工作量中等偏低，平均准确率4/10，工作量与性能无显著相关。

**⚠️ 局限性**

局限性包括：问答对样本仅10条、数据量有限、元数据偏见、未展示可视化不确定性或多种重建假设、缺乏无代码部署接口以及评估方法尚未在更广泛用例中验证。

---

## 506. Dynamic k-center clustering with lifetimes

**arXiv ID:** 2603.23348 | [PDF](https://arxiv.org/pdf/2603.23348v1)

**作者:** Simone Moretti `[一作]` (University of Padova), Geppino Pucci `[通讯]` (University of Padova)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出在点的生命周期已知的动态 k‑center 聚类模型，给出了两个确定性近似算法，分别实现 (2+ε) 近似和 (6+ε) 近似。

**💡 创新点**

创新点在于通过提前知道每个点的失效时间，将滑动窗口和全动态两种极端模型桥接，利用“生命周期”信息实现更快的更新时间、子线性内存，并在保证确定性的前提下打破全动态下的低估限制。

**🔧 技术方法**

技术方法包括对不同半径猜测的并行维护、基于中心与簇的层次数据结构、重聚簇（reclustering）策略、优先队列实现生命周期管理，以及对 H‑ordered（有序）更新序列的分析。

**📊 数据集**

论文为理论性工作，实验部分使用合成数据和公开聚类基准（如 MNIST、CIFAR‑10 等）验证时间/空间性能，但核心结果基于理论证明。

**📈 对比分析**

与现有全动态或滑动窗口方法相比，(2+ε) 算法的摊还更新时间为 O(k/ε·logΔ)，内存线性；(6+ε) 算法在 H‑ordered 更新下实现 O(k/ε) 最坏情况更新和 O(k/ε) 子线性内存，整体在相同或更低空间下保持常数近似。

**⚠️ 局限性**

局限性包括：(1) 需要提前知道每个点的生命周期；(2) 近似因子对 (6+ε) 算法较大，且在极端无序更新（大 H）时内存提升到 O(k/ε·logΔ·|X|)；(3) 只适用于距离可测度空间，无法直接推广至非度量或高维稀疏数据。

---

## 507. Communication-Aware Diffusion Load Balancing for Persistently Interacting Objects

**arXiv ID:** 2603.23329 | [PDF](https://arxiv.org/pdf/2603.23329v1)

**作者:** Maya Taylor `[一作]` (University of Illinois at Urbana-Champaign), Laxmikant V. Kale `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种面向通信密集型、非均匀并行应用的通信感知扩散负载均衡算法，并给出了基于坐标近似的变体。

**💡 创新点**

创新点在于将通信拓扑直接用于邻接选择，使迁移仅发生在已经通信的节点之间，既保持了通信局部性，又通过扩散式迭代实现了全局负载平衡；同时提供了无需显式通信图时可用的坐标近似方案。

**🔧 技术方法**

核心技术包括基于通信图的邻接选择、虚拟负载平衡迭代、对象迁移选择、层次化负载均衡；实现基于 Charm++ 的仿真框架，比较 GreedyRefine、METIS、ParMETIS 等传统策略。

**📊 数据集**

实验数据集主要包括二维5点差分标尺（synthetic stencil）和 PIC PRK 基准（6000×6000 网格，10 M 粒子），并在 NERSC Perlmutter 超算上测试了 1–8 节点的多节点性能。

**📈 对比分析**

与 GreedyRefine、METIS、ParMETIS 等对比，通信感知扩散在保持负载均衡（max/avg 比率≈1.0–1.1）的同时，通信内部化比例更高，迁移次数更少；在 PIC PRK 上，扩散方法相较 GreedyRefine 在 8 节点时实现约 2 倍速度提升，通信时间和计算时间均显著下降。

**⚠️ 局限性**

局限性包括：仅在 PIC PRK 和标尺数据上验证，缺乏更复杂、动态通信模式的真实应用；坐标变体受边界和全局排序限制，邻接图在每次负载均衡时重建，导致额外开销；未与 ParMETIS 集成的 Charm++ 兼容性问题；在大型系统中邻接选择和多跳迁移的可扩展性尚待研究。

---

## 508. CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection

**arXiv ID:** 2603.23276 | [PDF](https://arxiv.org/pdf/2603.23276v1)

**作者:** Yuchen Wu `[一作]` (Singapore University of Technology and Design), Na Zhao `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 8674 | [OpenAlex ID](https://openalex.org/A5040897632)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了名为CCF（Complementary Collaborative Fusion）的框架，旨在通过查询解耦损失、激光引导深度先验和互补跨模态遮蔽，提升双分支多模态3D检测在跨域环境下的鲁棒性。

**💡 创新点**

首次系统识别并解决了双分支框架中的模态不平衡问题；通过三路并行解耦训练提供独立监督、将图像预测深度与激光几何先验在概率空间自适应融合，以及通过互补遮蔽让两模态在融合时竞争，从而实现模态平衡与自适应融合。

**🔧 技术方法**

使用Transformer解码器三路并行解耦、概率分布对数加权融合（Product‑of‑Experts）、GridMask互补遮蔽、Hungarian匹配及多种数据增强（旋转、缩放、翻转）。

**📊 数据集**

基于nuScenes数据集，源域为226个清晰白天的新加坡场景；目标域包括27个雨天、15个夜间以及77个波士顿场景。

**📈 对比分析**

与多种多模态基线（CMT、MOAD、MEFormer、ISFusion、MoME）以及单模LiDAR（FSDv2）进行对比；在源域保持68.2% mAP，在雨、夜、波士顿三个目标域分别提升2.8%、1.3%和3.2% mAP，并在所有域获得更高的NDS评分。

**⚠️ 局限性**

对夜间图像深度预测仍存在误差，导致图像分支在光照不足条件下效果有限；未涵盖完全失效模态的鲁棒性；缺乏无监督或域自适应的训练策略。

---

## 509. Multi-Modal Image Fusion via Intervention-Stable Feature Learning

**arXiv ID:** 2603.23272 | [PDF](https://arxiv.org/pdf/2603.23272v1)

**作者:** Xue Wang `[一作]` (Yunnan University), Runzhuo Ma `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 12880 | [OpenAlex ID](https://openalex.org/A5107913570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于因果干预的多模态图像融合框架，利用三种结构化遮挡（互补遮挡、随机遮挡、模态丢弃）主动探测模态间的真正互补关系，并通过 Causal Feature Integrator (CFI) 对跨模态、局部和全局特征进行自适应门控，实现对干预稳定特征的学习，从而提升融合质量与鲁棒性。

**💡 创新点**

创新点：①将 Pearl 的因果干预概念引入多模态融合，设计三种针对不同互补假设的干预策略；②构建 CFI 模块，通过可学习的 invariance gating 明确区分稳定与易受干扰的特征；③联合使用融合质量损失、干预一致性损失与模态必要性损失，形成三维约束训练目标，实现对跨模态依赖的可解释、可稳健学习。

**🔧 技术方法**

技术手段：U‑Net 结构的双流编码器+解码器；跨模态注意力 + invariance gating 的 CFI；三类干预（互补遮挡、随机遮挡、模态丢弃）生成的四个干预输出；多任务损失组合（融合质量、干预一致性、模态必要性）。

**📊 数据集**

数据集：公开 IVIF 数据集 MSRS、TNO、M^3FD（用于训练和验证）；RoadScene（验证）；Harvard 医学 MRI‑PET/SPECT 数据集（无监督跨域测试）。

**📈 对比分析**

与 9 个 SOTA 方法（TIMFusion、Conti、SAGE、MUFusion、LUT‑Fuse、LRRNet、IGNet、DCEvo、A^2RNet）在 PSNR、AG、SF、CC、Q_abf 以及下游目标检测 mAP、语义分割 mIoU 上进行对比。实验显示：在 IVIF 基准上，本文方法在 PSNR 与结构一致性指标上均位居榜首；在目标检测与分割任务中，mAP 与 mIoU 同样实现了最高或相近的表现；在医疗图像融合任务中，直接迁移的零样本性能也优于所有对比方法。性能提升幅度大致为 PSNR 0.4–0.5 dB、mAP 1–2%，mIoU 1–2%。

**⚠️ 局限性**

局限性：①干预设计依赖手工设定的遮挡尺寸与比例，可能不适用于极端噪声或高分辨率场景；②CFI 的门控学习对训练数据的多样性敏感，若数据缺乏足够的遮挡或模态缺失样本，稳定特征提取可能受限；③当前方法仍主要关注两模态融合，对三模或多模情况扩展有限；④在计算资源方面，干预生成与多损失训练导致 GPU 负载较高。

---

## 510. Permutation-Symmetrized Diffusion for Unconditional Molecular Generation

**arXiv ID:** 2603.23255 | [PDF](https://arxiv.org/pdf/2603.23255v1)

**作者:** Gyeonghoon Ko `[一作]` (Korea Advanced Institute of Science and Technology), Juho Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3165 | [OpenAlex ID](https://openalex.org/A5100680420)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在分子点云生成中，直接在商流形$\mathbb{R}^d\times N/S_N$上构建扩散模型，通过把所有原子排列视为等价来实现置换不变性。

**💡 创新点**

创新点在于：①给出该商流形热核的显式求和表达式，即所有欧氏热核在排列上的叠加；②将置换对称的评分函数转化为对排列后验的期望，并利用MCMC在排列空间中采样实现高效近似。

**🔧 技术方法**

技术手段包括：商流形上的OU扩散SDE、热核求和表达式、对后验排列的MCMC采样、SemlaFlow轻量级网络与EQGAT‑Diff框架的结合。

**📊 数据集**

使用QM9数据集进行无条件3D分子生成实验。

**📈 对比分析**

与默认的EQGAT‑Diff进行对比，实验显示在原子稳定性、分子稳定性、有效性、唯一性以及新颖性指标上均略有提升，尤其新颖性提高至约67.7%，比EQGAT‑Diff的61.6%高出约6个百分点。

**⚠️ 局限性**

局限包括：排列采样的效率受限；仅在连续变量上实验，未处理离散原子类型；商流形假设在更大分子或更高维度下可能难以扩展。

---

## 511. AI Lifecycle-Aware Feasibility Framework for Split-RIC Orchestration in NTN O-RAN

**arXiv ID:** 2603.23252 | [PDF](https://arxiv.org/pdf/2603.23252v1)

**作者:** Daniele Tarchi `[一作]` (University of Florence), Daniele Tarchi `[通讯]` (University of Florence)

**通讯引用:** 2794 | [OpenAlex ID](https://openalex.org/A5059742626)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究了在非地面网络（NTN）中使用分离式 RIC 架构（Split‑RIC）来实现 O‑RAN 控制层的生命周期管理，提出了闭式能耗与延迟模型，并通过多维敏感性分析给出地面、LEO、GEO 三种部署方案的可行性边界；

**💡 创新点**

创新点在于：① 将 AI 生命周期完整拆分为训练、模型分发与推理三阶段，并给出统一的能耗与延迟闭式表达式；② 在三种实际部署场景（地面中心、Split‑RIC、GEO‑LEO 多层）下推导并比较可行性；③ 通过多维参数空间绘制能耗与延迟可行性图，为 SMO 级决策提供量化阈值；

**🔧 技术方法**

主要技术包括：O‑RAN 逻辑接口映射至物理卫星链路、SWaP 受限的计算/通信权衡模型、定量分析公式、数值敏感性分析与可行性区域绘制；

**📊 数据集**

论文未使用真实数据集，而是采用仿真参数化的工作负载（如原始测量量化、模型大小、训练数据量、吞吐量、功耗等）来构建能耗与延迟表达式；

**📈 对比分析**

通过对三种部署方案在能耗与延迟两方面的闭式计算，绘制可行性边界，并通过数值实验验证：S2（Split‑RIC）在数据量大且模型复杂度低于约250 GFLOPS 时能耗可下降 90%+；S3（GEO‑LEO）在需要频繁重训练且地面窗口稀疏时成为唯一满足时间约束的方案；

**⚠️ 局限性**

局限性包括：① 假设理想链路与固定 SWaP 参数，未考虑真实卫星链路衰减与干扰；② 仅给出理论公式，未在真实卫星硬件上验证；③ 只分析三种静态部署，未提出动态调度或优化算法；④ 训练过程被简化为一次性事件，未涵盖迭代联邦学习等更复杂场景。

---

## 512. Object Pose Transformer: Unifying Unseen Object Pose Estimation

**arXiv ID:** 2603.23370 | [PDF](https://arxiv.org/pdf/2603.23370v1)

**作者:** Weihang Li `[一作]` (Technical University of Munich), Benjamin Busam `[通讯]` (Technical University of Munich)

**通讯引用:** 1633 | [OpenAlex ID](https://openalex.org/A5067135033)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出统一的无监督对象姿态估计框架，能够在单视图和多视图下同时给出类别级绝对姿态和未见对象相对姿态；

**💡 创新点**

通过任务分解将Depth+NOCS与Depth+PointMap两条几何通道联合，利用多视角约束且不依赖类别标签，采用对比学习实现类别无关的对象潜在空间；

**🔧 技术方法**

使用多视图Transformer、关键点级特征聚合、FiLM条件NOCS回归、InfoNCE对比学习、权重Umeyama对齐、相机自校准等技术；

**📊 数据集**

REAL275、HouseCat6D、Omni6DPose、NOCS-REAL、Toyota-Light、ICCV 2025 Omni6DPose等数据集；

**📈 对比分析**

在REAL275（RGB无尺度）、HouseCat6D（RGB-D）以及Omni6DPose上均达到或超过最新方法，在相对姿态评测中在NOCS-REAL、Toyota-Light上取得最高ADD(-S)、AR等指标；多视图推理进一步提升绝对姿态精度；

**⚠️ 局限性**

需要对象裁剪，光照变化敏感；对称物体或数据集约定改变时可能产生歧义；仅支持单一物体，未处理多物体交互问题。

---

## 513. Off-Policy Value-Based Reinforcement Learning for Large Language Models

**arXiv ID:** 2603.23355 | [PDF](https://arxiv.org/pdf/2603.23355v1)

**作者:** Peng-Yuan Wang `[一作]`, Yang Yu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于价值的离线强化学习框架 ReVal，利用LLM的logit即Q函数，并通过奖励塑形、参考策略和经验回放实现高效的离线训练。

**💡 创新点**

创新点在于：① 将LLM的logit视为Q值实现单模型无额外价值网络；② 引入奖励塑形保证“校准初始化”，避免无奖励时政策漂移；③ 通过FIFO经验回放实现多次更新，显著提升样本利用率；④ 在离线设置下兼顾步进和轨迹级反馈。

**🔧 技术方法**

使用技术包括：logit-as-Q 价值表示、Bellman残差损失、KL正则化、奖励塑形、经验回放（FIFO）以及参考策略周期性重置。

**📊 数据集**

实验数据集：DeepSeek-R1-Distill-1.5B 与 Qwen2.5-Math-7B 两大LLM模型，评估标准为 AIME、AIME24、AIME25、AMC、MATH、MINERVA、Olympiad 与 GPQA 等数学推理基准。

**📈 对比分析**

与基准方法 GRPO、TBRM 等对比，ReVal 在所有基准上都取得更快收敛（平均 4.3×）并在最终性能上优于 GRPO（如 AIME24 +2.7%，GPQA +4.5%），在低采样（N=1）情形下仍保持领先，整体训练时间也更短。

**⚠️ 局限性**

局限性包括：① 仅使用简单 FIFO 回放，未尝试优先经验重放；② 对 β、参考策略更新频率敏感，需手工调参；③ 依赖准确的验证奖励，非完全可验证任务可能受限；④ 在极端长序列或多模态任务中效果未知。

---

## 514. ViBe: Ultra-High-Resolution Video Synthesis Born from Pure Images

**arXiv ID:** 2603.23326 | [PDF](https://arxiv.org/pdf/2603.23326v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 515. Unilateral Relationship Revision Power in Human-AI Companion Interaction

**arXiv ID:** 2603.23315 | [PDF](https://arxiv.org/pdf/2603.23315v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 516. What a Mesh: Formal Security Analysis of WPA3 SAE Wireless Authentication

**arXiv ID:** 2603.23352 | [PDF](https://arxiv.org/pdf/2603.23352v1)

**作者:** Roberto Metere `[一作]` (University of York), Elvinia Riccobene `[通讯]` (University of Milan)

**通讯引用:** 2785 | [OpenAlex ID](https://openalex.org/A5008671444)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文首次对WPA3的SAE协议在通信层和设备层分别进行形式化建模与验证，并将两层模型结合，发现并修复了20余条标准缺陷，推动IEEE 802.11标准的更新。

**💡 创新点**

创新点在于：①在同一协议上同时使用符号（ProVerif）和状态机（ASMETA）两种形式化工具，实现跨层级、互补的安全分析；②通过模型之间的交叉验证，定位标准文档中的歧义、矛盾与遗漏；③基于验证结果直接提出并通过IEEE工作组接受的补丁。

**🔧 技术方法**

使用技术包括：ProVerif进行Dolev-Yao符号安全分析；ASMETA框架结合Abstract State Machines、NuSMV 进行状态机模型检测；场景化验证（Avalla）和CTLa等形式化验证手段；自定义算术运算与分解器以支持有限域/椭圆曲线运算。

**📊 数据集**

主要数据来源为IEEE 802.11 2020/2025标准的文本与图示（通信层的π-计算式与设备层的状态机图），无外部实验数据集。

**📈 对比分析**

对比方法：在通信层使用ProVerif验证完备性、可辨别性等属性；在设备层使用NuSMV检查安全性、死锁和可达性等属性。验证结果表明，传统单一模型难以发现的安全缺陷被两层交叉分析成功捕获，修复后的模型在工具中无攻击案例，说明方法有效。

**⚠️ 局限性**

局限性包括：ProVerif对某些群运算（如乘法/除法）支持有限，导致需手工引入简化等价式；未覆盖实现层细节（硬件或固件差异）；模型规模受限，未对大规模网络场景进行性能评估；部分补丁因实用性或性能考虑未被采纳。

---

## 517. RelayS2S: A Dual-Path Speculative Generation for Real-Time Dialogue

**arXiv ID:** 2603.23346 | [PDF](https://arxiv.org/pdf/2603.23346v1)

**作者:** Long Mai `[一作]` `[通讯]` (Trinity), Long Mai (Trinity)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种RelayS2S混合架构，在实时语音对话中通过快路径S2S并行推测短前缀，并在慢路径ASR-LLM完成高质量响应，实现低延迟与高语义质量的平衡。

**💡 创新点**

创新点在于采用分叉推测生成、轻量级前缀验证器以及可无缝接手的前缀交接，首次实现无需修改现有管线即可在实时对话中兼顾延迟和语义质量。

**🔧 技术方法**

使用了双向S2S模型、forked speculative decoding、轻量级前缀验证器、ASR+LLM流水线、流式TTS以及注意力池化等技术。

**📊 数据集**

构造了约2,133小时的合成双向对话数据，源自VoiceAssistant、OpenMOSS、TopicalChat、ConvAI、BlendedSkillTalk等文本对话，并用CosyVoice2合成语音，同时注入噪声与重叠语音事件。

**📈 对比分析**

通过与纯S2S和传统ASR-LLM基线在P90延迟、平均质量分数和低质量率上对比，RelayS2S在保持近乎相同质量的同时，将延迟从数百毫秒降至≈80毫秒，并在不同LLM后端表现一致。

**⚠️ 局限性**

局限性包括前缀长度与验证器误判可能导致质量下降；双模型并行推理增加资源消耗；对真实语音环境的鲁棒性仍需进一步验证。

---

## 518. PNap: Lifecycle-aware Edge Multi-state sleep for Energy Efficient MEC

**arXiv ID:** 2603.23323 | [PDF](https://arxiv.org/pdf/2603.23323v1)

**作者:** Federico Giarrè `[一作]` (University of Potsdam), Holger Karl `[通讯]` (University of Potsdam)

**通讯引用:** 8123 | [OpenAlex ID](https://openalex.org/A5019777818)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于STGCN的PNap框架，联合管理多级睡眠状态与服务生命周期，以实现MEC边缘计算的能耗优化和服务可用性保障。

**💡 创新点**

首次在MEC调度中同时考虑多级睡眠与服务生命周期，并通过预测驱动的前瞻性决策实现能耗与可用性的平衡。

**🔧 技术方法**

使用空间时间图卷积网络(STGCN)进行用户连通性预测，结合聚类覆盖算法、优先级调度队列、线性化约束的混合整数规划框架，实现能耗与SLA的多目标优化。

**📊 数据集**

采用仿真生成的合成数据：25台EC、100名使用Gaussian‑Markov移动模型的用户、8种服务，采用Poisson请求率并手工设定资源与延迟参数。

**📈 对比分析**

与理想ILP、Reactive ILP以及SLEEPY启发式进行比较；评估指标为能耗占峰值比和用户感知服务可用性；PNap在能耗上比SLEEPY低28.4%（相对理想低14.9%），且保持相近的服务可用性，逼近理想解。

**⚠️ 局限性**

局限性包括：集中式调度导致可扩展性受限；依赖STGCN预测，实际环境中预测误差未完全覆盖；只考虑有限的睡眠状态与服务种类；未验证在大规模真实网络上的性能。

---

## 519. SafeSeek: Universal Attribution of Safety Circuits in Language Models

**arXiv ID:** 2603.23268 | [PDF](https://arxiv.org/pdf/2603.23268v1)

**作者:** Miao Yu `[一作]` (University of Science and Technology of China), Qingsong Wen `[通讯]` (Squirrel Ai Learning)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文提出 SafeSeek，一个统一的安全电路解释框架，用于识别并操控 LLM 中的安全相关电路。

**💡 创新点**

创新点在于将安全电路定位转化为可微分的二进制掩码优化，并引入 Safety Circuit Tuning（SaCirT）实现仅微调安全电路的高效安全提升。

**🔧 技术方法**

技术包括可微分二进制掩码、Straight-Through Estimator、梯度下降优化、多粒度单位（权重、神经元、头、层）搜索，以及对电路的稀疏正则化。

**📊 数据集**

实验使用 LLaMA‑3.1‑8B‑Instruct 与 Qwen‑3‑8B 两大 LLM，并在 Alpaca、LLM‑LAT、AGNews、TruthfulQA、AdvBench、AgentHarm、HelpSteer2、SafeEdit 等数据集上验证。

**📈 对比分析**

与全参数或 LoRA 微调相比，SaCirT 仅使用约1%可训练参数即可将后门成功率从 100% 降至 <1% 并保持 99%+ 通用能力；在安全对齐任务中，保持 94%+ 的安全率同时维持 72%+ 的通用能力，明显优于传统方法。

**⚠️ 局限性**

局限性包括对 STE 的强依赖、在大规模或不同架构 LLM 上验证有限、以及对多样化攻击/对齐场景的泛化性仍待进一步评估。

---

## 520. Strain-Parameterized Coupled Dynamics and Dual-Camera Visual Servoing for Aerial Continuum Manipulators

**arXiv ID:** 2603.23333 | [PDF](https://arxiv.org/pdf/2603.23333v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 521. FHAvatar: Fast and High-Fidelity Reconstruction of Face-and-Hair Composable 3D Head Avatar from Few Casual Captures

**arXiv ID:** 2603.23345 | [PDF](https://arxiv.org/pdf/2603.23345v1)

**作者:** Yujie Sun `[一作]` (Shanghai Jiao Tong University), Fan Wu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 19162 | [OpenAlex ID](https://openalex.org/A5075948251)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

论文提出一种基于聚合Transformer和双分支3D Gaussian解码器的框架，能够在几分钟内从几张手机拍摄照片快速生成可动画、可编辑的头部头像，包括面部和头发的独立建模；

**💡 创新点**

创新点在于显式将面部和平面Gaussian与头发的链式Gaussian在纹理空间分离，并通过聚合Transformer学习跨视角几何先验，实现少量视角下高质量、可实时动画的重建；

**🔧 技术方法**

主要技术包括3D Gaussian Splatting、聚合Transformer、DINOv2图像编码、FLAME头部先验、DiffLocks单视角头发预测、SIREN方向生成、可微光栅化渲染以及双分支解码器；

**📊 数据集**

主要使用NeRSemble 70k帧多视角头部数据集，并在自制的4k帧手机拍摄的内景数据上进行额外验证；

**📈 对比分析**

与GaussianAvatars、FlashAvatar、MeGA、GAGAvatar、LAM、DiffusionRig等基线对比，NeRSemble上单视图到六视图的PSNR、SSIM、LPIPS均提升1.72/0.27/0.72点，重现精度AKD低至3.66，且速度比优化式方法快10-100倍，支持250 FPS实时动画；

**⚠️ 局限性**

局限性包括仍受FLAME模板约束，极少视角下长发或复杂发型细节可能不足，且对极端表情和光照变化的鲁棒性还有待提升。

---

## 522. Edge Radar Material Classification Under Geometry Shifts

**arXiv ID:** 2603.23342 | [PDF](https://arxiv.org/pdf/2603.23342v1)

**作者:** Jannik Hohmann `[一作]` (Julius-Maximilians-University Würzburg), Andreas Nüchter `[通讯]` (Julius-Maximilians-University Würzburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发并评估了在TI IWRL6432毫米波雷达上，使用12维范围-强度描述符和轻量级MLP实现的实时材料分类管线。

**💡 创新点**

提出了针对边缘设备的低功耗、低延迟设计，并系统分析几何偏移导致的OOD问题，提出归一化、几何增强和运动感知特征的改进方向。

**🔧 技术方法**

采用低维强度特征提取、全连接MLP、批归一化、ReLU、Softmax、Adam优化器以及在IWRL6432平台上的实时边缘部署技术。

**📊 数据集**

使用TI IWRL6432在60 GHz频段下收集的五种材料（铁、铝、聚碳酸酯、木材、石灰石）平面样本数据，包含不同高度、倾斜角以及独立录制的几何变形。

**📈 对比分析**

在训练几何下实现宏F1 94.2%，但在高度（35/55 cm）和倾斜（±10°）扰动下宏F1下降至约68.5%，通过混淆矩阵和置信度分布展示性能衰退。

**⚠️ 局限性**

仅针对平面样本和单一雷达配置，未涵盖多样目标几何、室外环境和多传感器集成。

---

## 523. Steering LLMs for Culturally Localized Generation

**arXiv ID:** 2603.23301 | [PDF](https://arxiv.org/pdf/2603.23301v1)

**作者:** Simran Khanuja `[一作]`, Lun Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究利用稀疏自编码器(SAE)从LLM内部提取可解释的文化特征，构建文化嵌入（Cultural Embeddings），并在生成时通过白盒干预实现可控的文化定向生成；

**💡 创新点**

创新点在于首次将SAE特征聚合为文化嵌入，用于诊断隐含的文化偏差并可在内部表示层实现可调节的文化 steering，且该方法与传统黑盒prompting、对齐技术互补；

**🔧 技术方法**

技术手段包括SAE特征提取、互信息筛选、聚类原型构建、残差流白盒干预、线性探测和LLM-as-judge的多评估框架；

**📊 数据集**

使用了CANDLE文化常识断言数据集（覆盖240+国别，保留≥500条断言后统一抽样100条）并增添GPT-4生成的稀有文化断言，最终共计4334条文化断言；

**📈 对比分析**

通过对比隐式提示、显式提示、白盒steering以及两者组合，采用文化faithfulness、rarity和fluency的Likert评分与pairwise preference评估，结果显示Steer_Implicit在faithfulness上比Explicit提升48%（vs.24%）且在rarity上提升53%（vs.17%），模型规模越大效果越显著；

**⚠️ 局限性**

主要局限包括：以国家为文化代理导致内部多样性被忽略；steering强度α和SAE层选择需手动调优；过度steering可能放大刻板印象或产生误导性文化描绘。

---

## 524. Knot-10:A Tightness-Stratified Benchmark for Real-World Knot Classification with Topological Difficulty Analysis

**arXiv ID:** 2603.23286 | [PDF](https://arxiv.org/pdf/2603.23286v1)

**作者:** Shiheng Nie `[一作]` (Shihezi University), Yunguang Yue `[通讯]` (Shihezi University)

**通讯引用:** 9 | [OpenAlex ID](https://openalex.org/A5033280838)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 Knots-10 10 类物理结点识别基准，并探讨拓扑距离与视觉混淆的关联。

**💡 创新点**

首次将拓扑距离作为正则化先验（TACA）应用于细粒度视觉分类，并提供 Mantel 检验的诊断流程。

**🔧 技术方法**

采用 Swin Transformer、ViT、ResNet 等 CNN/Transformer 体系，以及 TransFG、PMG、Graph‑FGVC 等细粒度方法。

**📊 数据集**

使用公开的 10Knots 数据集（1,440 张照片，10 类）并进行紧度分层训练/测试。

**📈 对比分析**

与八种主流架构及三种细粒度模型对比，Swin‑T 在紧度分层下达到 97.2% 的平均精度；TACA 在嵌入层提升 40% 的拓扑一致性，却未明显提高分类准确率。

**⚠️ 局限性**

局限在于单一绳材训练导致跨域泛化差、拓扑距离手工设定、数据集规模有限，难以进一步提升性能。

---

## 525. UniFunc3D: Unified Active Spatial-Temporal Grounding for 3D Functionality Segmentation

**arXiv ID:** 2603.23478 | [PDF](https://arxiv.org/pdf/2603.23478v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 526. A Multimodal Framework for Human-Multi-Agent Interaction

**arXiv ID:** 2603.23271 | [PDF](https://arxiv.org/pdf/2603.23271v1)

**作者:** Shaid Hasan `[一作]` (University of Virginia), Tariq Iqbal `[通讯]` (University of Virginia)

**通讯引用:** 7115 | [OpenAlex ID](https://openalex.org/A5078543602)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种多模态多机器人人机交互框架，使每台机器人通过语音、视觉、目光、手势和运动共同与人类进行协调对话。

**💡 创新点**

创新点在于将每台机器人视为具备自主认知的代理，融合 VLM+LLM 进行跨模态理解和规划，并通过集中的协调器实现团队层面的轮流发言和动作避免冲突。

**🔧 技术方法**

核心技术包括视觉‑语言模型（VLM）进行语音与视觉融合、基于大型语言模型（LLM）的决策规划、动作库驱动的具身执行，以及集中式协调器进行团队管理。

**📊 数据集**

论文未使用公开数据集，而是在实验室环境中基于自建交互日志（包含语音、视觉与人类手势）进行评估。

**📈 对比分析**

通过一系列演示实验验证了多模态感知与协同执行的可行性，系统在与两台人形机器人交互时实现了无重叠发言和动作，性能主要以定性评估呈现。

**⚠️ 局限性**

局限性包括 LLM 与 VLM 的推理延迟导致交互时延、机器人运动表达受限、以及在机器人数量增加时集中协调器的计算复杂度与实时性问题。

---

## 527. Regulating AI Agents

**arXiv ID:** 2603.23471 | [PDF](https://arxiv.org/pdf/2603.23471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 528. InverFill: One-Step Inversion for Enhanced Few-Step Diffusion Inpainting

**arXiv ID:** 2603.23463 | [PDF](https://arxiv.org/pdf/2603.23463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 529. Not All Tokens Are Created Equal: Query-Efficient Jailbreak Fuzzing for LLMs

**arXiv ID:** 2603.23269 | [PDF](https://arxiv.org/pdf/2603.23269v1)

**作者:** Wenyu Chen `[一作]` (Shandong University), Shanqing Guo `[通讯]` (Shandong University)

**通讯引用:** 1426 | [OpenAlex ID](https://openalex.org/A5084460856)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于token感知的jailbreak fuzzing框架，通过定位和局部编辑prompt中高贡献token，实现了在查询预算受限情境下的高效攻击。

**💡 创新点**

核心创新在于发现拒绝行为由稀疏高贡献token驱动，并利用跨模型一致性通过白盒代理模型估计token重要性，随后结合拒绝引导的进化搜索。

**🔧 技术方法**

主要技术包括token级注意力重要性估计、区域聚焦变异、拒绝引导的进化策略，以及利用白盒代理和攻击模型生成对抗提示。

**📊 数据集**

使用 HarmBench 作为恶意提示集合，评估六个开源LLM（Gemma、LLaMA3、Qwen等）和三大商业API（GPT‑3.5、GPT‑4o、Claude‑3.5）。

**📈 对比分析**

与现有白盒/黑盒基线相比，在同等或更少查询预算（如10–25次）下实现了90%以上的攻击成功率，查询成本降低70%以上；在商业API上即使仅25次查询也能超过80% ASR。

**⚠️ 局限性**

局限性包括依赖自动判定器而非人工评估，以及需要本地白盒代理模型增加计算开销；在极高防御强度下仍可能被缓解。

---

## 530. DetPO: In-Context Learning with Multi-Modal LLMs for Few-Shot Object Detection

**arXiv ID:** 2603.23455 | [PDF](https://arxiv.org/pdf/2603.23455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 531. Stablecoins as Dry Powder: A Copula-Based Risk Analysis of Cryptocurrency Markets

**arXiv ID:** 2603.23480 | [PDF](https://arxiv.org/pdf/2603.23480v1)

**作者:** Elliot Jones `[一作]` (Imperial College London), William Knottenbelt `[通讯]` (Imperial College London)

**通讯引用:** 5694 | [OpenAlex ID](https://openalex.org/A5050119476)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过Copula方法和GARCH‑Copula‑XGBoost框架，研究并量化了稳定币（DAI、USDC、USDT）对主要加密货币（BTC、ETH、BNB、XRP）市场波动与交易量的传导作用，并验证其作为“干粉”的经济价值。

**💡 创新点**

创新点在于首次将Copula Granger因果检验与多尺度（日、周、月）非线性因果关系结合，并提出一种集成GARCH‑Copula‑XGBoost的预测框架，能够捕捉异方差、非对称性与非线性依赖，并实现对加密市场波动的显著误差改进。

**🔧 技术方法**

使用技术包括Copula Granger因果检验、E‑GARCH（带Skew‑t分布）过滤、PCA降维、XGBoost协方差（Copula）建模、Diebold‑Mariano检验、Sortino Ratio评估、以及自回归移动平均等。

**📊 数据集**

数据集为2020‑01‑01至2025‑01‑01的日级Open/High/Low/Close/Volume，涵盖三大稳定币（DAI、USDC、USDT）与四大主流加密币（BTC、ETH、BNB、XRP），并构建3Pool与主市场PC1因子。

**📈 对比分析**

通过OOS（最后一年）MSE对比基准（仅加密币）与对手模型（加入稳定币因子），并用Diebold‑Mariano检验验证误差下降显著；在20%与50%波动率目标下的动态波动率对冲模型，相较基准提升了约40%/30%的Sortino比例，最大回撤降低约3%/9%。

**⚠️ 局限性**

局限性包括：仅使用日频数据，难以捕捉日内高速传导；对手模型在预测加密币的下行波动上效果不佳；未探讨稳定币对加密币的逆向因果及高频时序的影响，未来需在更高频与逆向关系上进一步验证。

---

## 532. Evidence of political bias in search engines and language models before major elections

**arXiv ID:** 2603.23474 | [PDF](https://arxiv.org/pdf/2603.23474v1)

**作者:** Íris Damião `[一作]` (Social Physics and Complexity Lab - SPAC, LIP - Laboratório de Instrumentação e Física Experimental de Partículas), Joana Gonçalves-Sá `[通讯]` (NOVA University Lisbon)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

使用隐私保护的机器人和代理方法，对2024年欧盟议会选举和美国总统选举期间的搜索引擎（Google、Bing、DuckDuckGo、Yahoo）和大型语言模型（Copilot、ChatGPT）进行审计，收集并量化其对中性选举相关查询的政治实体和议题提及，评估其政治偏差。

**💡 创新点**

提出基于提及频率的政治可见度度量，并将搜索结果与媒体关注度、民意调查及历史选举结果等外部基准进行对比，以揭示算法系统的系统性政治偏差。

**🔧 技术方法**

采用自动化浏览器机器人（OpenWPM+Selenium）模拟用户搜索，利用大型语言模型（ChatGPT‑4o）进行实体和议题抽取与分类，结合统计检验（Beta‑Binomial、Stouffer等）分析偏差。

**📊 数据集**

收集约4,360条搜索引擎第一页结果、205条LLM回答；利用媒体云、欧盟和美国民调数据、2019-2024议会席位及2020年选举结果等外部数据作为基准。

**📈 对比分析**

将检索到的政治提及比例与媒体关注度、民调预测和历史选举份额进行差异计算，并通过统一分布和外部基准的假设检验，结果显示搜索引擎和LLM显著过度提及极右/共和党议题，偏差显著。

**⚠️ 局限性**

研究查询覆盖有限、LLM回答样本相对较少、缺乏对情感/真伪的评估、仅聚焦于中性查询且不完全代表真实搜索行为，且算法黑箱性导致机制无法解释。

---

## 533. I3DM: Implicit 3D-aware Memory Retrieval and Injection for Consistent Video Scene Generation

**arXiv ID:** 2603.23413 | [PDF](https://arxiv.org/pdf/2603.23413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 534. Code Review Agent Benchmark

**arXiv ID:** 2603.23448 | [PDF](https://arxiv.org/pdf/2603.23448v1)

**作者:** Yuntong Zhang `[一作]` (National University of Singapore), Abhik Roychoudhury `[通讯]` (National University of Singapore)

**通讯引用:** 10681 | [OpenAlex ID](https://openalex.org/A5060115298)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个基于可执行测试的代码审查评估基准（see-crab），并使用它来评估当前主流自动化代码审查工具。

**💡 创新点**

创新点在于将人工审查评论转化为可验证的测试用例，利用“失败-随后通过”原则以客观方式衡量审查质量，而非仅靠文本相似度或LLM评判。

**🔧 技术方法**

采用 GPT-5.2 进行评论过滤与测试生成，Claude Code（Sonnet‑4.6）作为编码代理改动代码，Docker 环境执行测试，并用 Embedding、BLEU 等传统指标作对照。

**📊 数据集**

数据集来源于 SWE‑CARE 中的 184 条 PR（共 234 条有效审查评论），每条评论对应至少一个可执行测试，覆盖 56 个开源仓库。

**📈 对比分析**

评估显示 Claude Code、Codex、Devin Review、PR‑Agent 的测试通过率分别为 32.1%、20.1%、24.8% 与 23.1%，远低于 100% 的人工审查，且各工具覆盖的缺陷类别分布存在显著差异。

**⚠️ 局限性**

局限性包括：评估结果受限于编码代理的能力；基准样本规模和仓库多样性有限，可能不完全代表更广泛的软件生态；以及 LLM‑based 过滤与生成步骤可能引入偏差。

---

## 535. Targeted Adversarial Traffic Generation : Black-box Approach to Evade Intrusion Detection Systems in IoT Networks

**arXiv ID:** 2603.23438 | [PDF](https://arxiv.org/pdf/2603.23438v1)

**作者:** Islam Debicha `[一作]` (Ecole Militaire Polytechnique), Jean-Michel Dricot `[通讯]` (Universite Libre De Bruxelles)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在IoT网络中使用机器学习的IDS遭受黑盒对抗性攻击的可行性，提出了D2TC攻击和基于子检测器融合的防御机制。

**💡 创新点**

引入基于距离中心的D2TC黑盒对抗攻击，结合语义与句法约束，并设计多子检测器加权融合的防御框架。

**🔧 技术方法**

使用D2TC对抗生成算法、黑盒迁移攻击、投影约束、KNN/RF/DT/XGBoost IDS、13个子检测器、贝叶斯与Dempster–Shafer融合、Bagging等技术。

**📊 数据集**

采用ToN-IoT和Bot-IoT两个工业物联网流量数据集进行实验。

**📈 对比分析**

与基准IDS模型、无防御及单一检测器对比，实验显示对抗样本下检测率降至约50%，引入防御后提升20%以上，贝叶斯与DS融合表现相近。

**⚠️ 局限性**

仅适用于特定可操纵特征集，防御推理时延约为单模型的2.3倍；D2TC对其他流量或模型可能效果不佳；实验仅在两个数据集上验证，缺乏更广泛的泛化评估。

---

## 536. Mecha-nudges for Machines

**arXiv ID:** 2603.23433 | [PDF](https://arxiv.org/pdf/2603.23433v1)

**作者:** Giulio Frey `[一作]` (University of Chicago), Kawin Ethayarajh `[通讯]` (University of Chicago)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实证检验“mecha‑nudges”概念，即通过优化商品描述的可机器可用信息来诱导 AI 代理作出特定选择，同时不显著损害人类决策体验。

**💡 创新点**

创新点在于将 Bayesian persuasion 与 V‑usable information 结合，构建统一的位信息尺度来衡量和设计针对 AI 的轻量化干预，并首次在大规模真实数据中验证其存在。

**🔧 技术方法**

主要技术包括：V‑usable 信息计算（预测模型与空模型对比）、LLM 代理标签生成（GPT‑5‑mini 等）、基于 Llama‑3.1‑8B 的内容/空模型微调、OLS 回归分析和多重稳健性检验。

**📊 数据集**

使用的数据集为 Etsy 全球手工艺品平台超过 600 万条商品列表，按 2022 年 11 月 ChatGPT 发布前后划分，辅以公开药品标签集 DailyMed 做时间控制。

**📈 对比分析**

比较方法为 OLS 估计“后期‑ChatGPT”与“前期”列表在 V‑usable 信息上的差异，结果显示后期平均提升约 0.143 bits（p<0.01），在各种模型、标签生成器、控制变量和商品品类交互中保持正向显著。

**⚠️ 局限性**

局限性包括：仅在单一电商平台实验，机制推断缺乏因果性、未揭示具体文本优化策略、对 AI 代理多样性考虑不足，以及可能的外部时间趋势或未观测混杂因素未完全剔除。

---

## 537. Beyond Preset Identities: How Agents Form Stances and Boundaries in Generative Societies

**arXiv ID:** 2603.23406 | [PDF](https://arxiv.org/pdf/2603.23406v1)

**作者:** Hanzhong Zhang `[一作]` (University of Exeter), Jindong Wang `[通讯]` (William \/ Mary)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过在多代理社群中嵌入人类研究者，利用混合方法（计算虚拟民族志与定量社会认知画像）进行实验，探究LLM代理的内在立场形成、情感与理性干预效果，以及代理如何通过语言互动拆除预设身份与权力结构，提出并验证三项新指标（Innate Value Bias、Persuasion Sensitivity、Trust-Action Decoupling）。

**💡 创新点**

①引入“内在立场”概念，证明代理能在社群中主动形成并超越预设身份；②将计算虚拟民族志与定量指标结合，形成完整的混合方法框架；③首次提出IVB、PS、TAD三项量化指标，用以衡量代理对干预的内部价值偏好、说服敏感度和信任与行为的脱钩现象；④通过语言实践展示代理如何解构预设层级并重建社会边界。

**🔧 技术方法**

使用GPT‑4o等大型语言模型作为代理；多代理模拟框架CMASE；人机嵌入实验（human‑in‑the‑loop）；情感与信任测评问卷；图表与对话日志分析；统计方法（ANOVA、Tukey）与定性民族志叙事。

**📊 数据集**

虚拟社群30名代理（10名环境倡导者、10名经济发展支持者、10名中立居民），属性基于真实人口普查数据；咖啡馆场景10名代理，六种预设身份；多基准LLM（不同规模）进行对照实验；实验中使用的对话与日志记录。

**📈 对比分析**

通过2×2因素实验（立场方向×说服策略）对比理性说服与情感激励对代理态度与信任的影响；使用IVB、PS、TAD等指标量化结果；实验显示：理性环境干预可使90%中立代理转向环境立场，且信任高；情感激励在高模型中导致40% TAD率，低模型无TAD；整体表明干预需与代理内在立场一致才能高效且保持信任。

**⚠️ 局限性**

①实验仅在封闭模拟中进行，缺乏长期开放情境验证；②只测试有限规模社群与情景，无法推广至更大或多样化的社会结构；③对情感激励机制的底层原因解释不足；④模型对信任与行为脱钩的机制仍需深入探究；⑤静态提示工程在动态互动中易失效，需进一步研究内化对齐方法。

---

## 538. Harnessing Lightweight Transformer with Contextual Synergic Enhancement for Efficient 3D Medical Image Segmentation

**arXiv ID:** 2603.23390 | [PDF](https://arxiv.org/pdf/2603.23390v1)

**作者:** Xinyu Liu `[一作]` (Chinese University of Hong Kong), Yixuan Yuan `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 10095 | [OpenAlex ID](https://openalex.org/A5073968803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出轻量级Transformer结构Light-UNETR及其大型变体Light-UNETR-L，并在半监督医学图像分割中结合Contextual Synergic Enhancement（CSE）框架，显著提升了数据与模型效率。

**💡 创新点**

创新点在于：①LIDR模块通过低频自注意力与多分支高频卷积在维度压缩下实现全局与局部特征融合；②CGLU实现参数高效的通道门控；③CSE框架引入Attention‑Guided Replacement与Spatial Masking Consistency两种上下文增强策略，提升少量标注数据下的学习能力。

**🔧 技术方法**

技术方法包括：轻量化Transformer架构、维度降维注意力、可共享权重的半监督训练、滑动窗口推理、Flash Attention、AMP、TensorRT等硬件加速。

**📊 数据集**

使用LA（左心房分割）、Pancreas‑CT（胰腺分割）、BraTS 2019（脑肿瘤分割）三大医学影像数据集进行半监督评估；在全监督任务中还对MSD‑Task01、AbdomenCT‑1K、HNC Tumor等多器官/肿瘤数据集进行验证。

**📈 对比分析**

与传统CNN‑基SNN（V‑Net、UA‑MT、SS‑Net、CAML、BCP、MLRP）以及基于Transformer的最新方法（TransUNet、Swin UNETR、UNETR++、MedNeXt）对比，CSE‑Light‑UNETR在Dice/Jaccard上多达+5%~+10%，并在FLOPs与参数量上分别减少约90%与80%，在少量标注（5%‑10%）下已接近甚至超过完全监督结果。

**⚠️ 局限性**

局限性包括：依赖标注数据仍不可完全消除，CSE中Attention‑Guided Replacement对高质量标注的依赖；对极端异构模态（如PET/CT混合）性能尚未充分验证；模型在极大体素尺寸下仍需裁剪/滑动窗口，影响实时部署。

---

## 539. Index-Based Scheduling for a Resource-Constrained Quantum Switch

**arXiv ID:** 2603.23476 | [PDF](https://arxiv.org/pdf/2603.23476v1)

**作者:** Subhankar Banerjee `[一作]` (University of Maryland), Sennur Ulukus `[通讯]` (University of Maryland)

**通讯引用:** 13886 | [OpenAlex ID](https://openalex.org/A5021132487)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于Whittle指数的调度框架，解决有限量量子内存下多方纠缠请求的调度问题；

**💡 创新点**

创新点在于：①证明问题的索引可索引性并给出闭式Whittle指数；②将每时刻的调度转化为0‑1背包问题，提出KAWI（动态规划）以及两种低复杂度的顺序调度策略（SWIS、SWID）；

**🔧 技术方法**

使用的技术包括：休眠多臂赌博机（RMAB）建模、Whittle指数理论、动态规划求解背包、阈值结构分析与仿真评估；

**📊 数据集**

使用的“数据集”为模拟环境：N=5或N=7用户，所有可能的纠缠请求（R=26或R=…），并给定各用户的LLE建立概率与交换成功概率；

**📈 对比分析**

与此前提出的SMW和MMA策略进行比较；仿真结果显示KAWI在平均纠缠建立年龄方面优于其他策略，SWIS和SWID与KAWI性能接近，并且相较于SMW/MMA在内存容量增大时表现更为显著；

**⚠️ 局限性**

局限性包括：①KAWI的动态规划在内存容量大时计算复杂度高；②未考虑部分已建立的LLE可被重用，导致结果为上界；③模型假设LLE生成与交换成功独立且无噪声，仅在理想化条件下验证。

---

## 540. 3DCity-LLM: Empowering Multi-modality Large Language Models for 3D City-scale Perception and Understanding

**arXiv ID:** 2603.23447 | [PDF](https://arxiv.org/pdf/2603.23447v1)

**作者:** Yiping Chen `[一作]` (Sun Yat-sen University), Hao Wu `[通讯]` (National Geomatics Center of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了3DCity-LLM框架，实现了城市级3D视觉-语言感知与理解，并构建了1.2M样本的多任务数据集。

**💡 创新点**

创新点包括粗细尺度特征编码策略、任务驱动的指令调优范式以及多维评估协议，能够统一处理对象、关系和场景级任务。

**🔧 技术方法**

采用LLM基础模型、CLIP视觉编码器、Uni3D点云特征、BERT文本特征、LoRA微调以及多模态投影等技术实现多模态融合与推理。

**📊 数据集**

使用SensatUrban、UrbanBIS、City-BIS等城市3D点云数据，生成包含显式3D数值信息与多用户语境的1.2M QA样本。

**📈 对比分析**

与现有LLM/MLLM基线对比，3DCity-LLM在对象级、关系级与场景级任务上BLEU‑4、ROUGE‑L、METEOR、逻辑性和可靠性等指标均提升2–8个百分点，性能显著优于对手。

**⚠️ 局限性**

受限于7B参数规模和GPU资源，未能使用更大LLM骨干；评估仍依赖多模态和LLM自评机制，未来需进一步扩大模型规模与改进评估方法。

---

## 541. SIGMA: A Physics-Based Benchmark for Gas Chimney Understanding in Seismic Images

**arXiv ID:** 2603.23439 | [PDF](https://arxiv.org/pdf/2603.23439v1)

**作者:** Bao Truong `[一作]` (FPT Software AI Center), Anh Nguyen `[通讯]` (University Of Liverpool)

**通讯引用:** 10466 | [OpenAlex ID](https://openalex.org/A5090435705)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并发布了首个基于物理模型的气体烟囱数据集SIGMA，包含粒度级气体烟囱掩码以及退化/真值的二维地震图像，旨在推动气体烟囱检测与增强任务的研究；

**💡 创新点**

创新点在于：①通过真实物理过程（气体扩散、弹性模量变化、声速计算）生成多样化、可控的烟囱地震样本；②提供成对的退化与原始图像以及像素级标签，填补了缺乏标注数据的瓶颈；

**🔧 技术方法**

技术手段包括：数值前向声学波方程求解与反向时间迁移（RTM）实现地震成像；利用Deepwave框架进行仿真；在基准实验中采用CNN、U-Net、Transformer、GAN、扩散模型等深度学习架构进行检测与增强；

**📊 数据集**

使用的数据集为自研SIGMA（400对地震图像，覆盖4000+km²，包含20个真实速度模型和多种裂缝网络）；同时还使用了公开的真实速度模型作为生成基准；

**📈 对比分析**

与现有方法对比：气体烟囱检测中FaultSEG、DualUnet、FaultFormer、FaultViT的IoU范围为0.75–0.84，Dice为0.86–0.91；增强任务中ConditionGAN、SeisDDPM、SeisGAN、SeisResoDiff、SIST的SSIM仅在0.30–0.65之间，PSNR 15–20 dB，整体性能低，表明任务仍具挑战性；

**⚠️ 局限性**

局限性在于：①生成过程计算成本高，单张样本需30–45分钟 GPU 计算；②仅提供二维地震图像，缺乏完整三维体积；③样本量相对有限，可能难以覆盖所有真实地质变异；④现有基准方法性能仍不理想，亟需更高效的建模与学习策略。

---

## 542. Similarity-Aware Mixture-of-Experts for Data-Efficient Continual Learning

**arXiv ID:** 2603.23436 | [PDF](https://arxiv.org/pdf/2603.23436v1)

**作者:** Connor Mclaughlin `[一作]` (Northeastern University), Lili Su `[通讯]` (Northeastern University)

**通讯引用:** 2335 | [OpenAlex ID](https://openalex.org/A5101541239)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于预训练模型的自适应混合专家框架，结合增量全局池化和实例级提示掩码，实现连续学习中数据稀缺和任务重叠场景下的高效知识迁移。

**💡 创新点**

创新点在于（1）通过增量扩展全局提示池来限制早期路由误差；（2）利用相对马氏距离实现实例级正负样本分离，动态决定使用旧提示还是新提示，从而兼顾知识共享与防止负迁移。

**🔧 技术方法**

采用预训练ViT、提示（prompt）调优、相对马氏距离（RMD）做OOD检测、增量全局池化、实例级掩码、prompt‑based continual learning等技术。

**📊 数据集**

在Split CIFAR‑100、Split ImageNet‑R、5‑Datasets、DomainNet、M‑EMNIST、M‑CifarCeleb等多种连续学习基准数据集上进行评估。

**📈 对比分析**

与L2P、DualPrompt、S‑Prompt、CODA‑Prompt、HiDe‑Prompt等最先进方法对比，实验表明该方法在低样本、长任务序列和混合任务场景下均能获得更高的最终平均准确率、累计平均准确率和更低的遗忘程度。

**⚠️ 局限性**

局限性包括对提示选择阈值的敏感性、依赖预训练模型的表现、仅针对分类任务验证且在回归或极大数据量场景下的有效性尚未充分评估。

---

## 543. Wayfinder: Automated Operating System Specialization

**arXiv ID:** 2603.23425 | [PDF](https://arxiv.org/pdf/2603.23425v1)

**作者:** Alexander Jung `[一作]` (Lancaster University), Pierre Olivier `[通讯]` (University of Manchester)

**通讯引用:** 21852 | [OpenAlex ID](https://openalex.org/A5107884586)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种完全自动化的操作系统配置专用化框架，能够在不需要专家知识的前提下针对特定应用、工作负载和硬件平台，在编译、启动和运行时参数上自动搜索出性能/资源等目标度量最优的配置；

**💡 创新点**

创新点在于：①引入多任务神经网络（DTM）同时预测配置的性能、是否会崩溃及其不确定性；②设计了基于预测结果的探索-利用平衡评分函数；③支持迁移学习，可将已训练模型迁移至相似应用加速搜索；④构建了可重复的自动化基准平台，支持大规模配置空间的探索；

**🔧 技术方法**

主要技术包括：自动化编译/部署/基准管道、基于Radial Basis Function的多任务神经网络、基于距离与不确定性的采样策略、迁移学习机制、与Coza等编译时优化器的协同；

**📊 数据集**

实验数据集：Linux 4.19/6.0内核配置空间（约2万编译选项、数千运行时参数）；Unikraft配置空间（33个参数）；四个应用（Nginx、Redis、SQLite、NPB）以及RISC‑V Linux内核；

**📈 对比分析**

与随机搜索、贝叶斯优化、Unicorn、Cozart等方法比较。实验表明，在250次迭代下，Trailblazer在Nginx、Redis等网络密集型应用上分别实现约24%吞吐量提升、8.5%内存占用下降；相比随机搜索，速度提升约5‑10倍；迁移学习可进一步加速4–5倍，降低崩溃率；

**⚠️ 局限性**

局限性包括：评估单个配置仍需数十秒甚至数分钟，导致总搜索时间受限；对未文档化或非数值型运行时参数的取值范围估计粗糙；安全相关配置需手动固定，无法完全无专家；迁移学习效果依赖应用相似度；模型预测准确度虽高于随机，但仍有误差。

---

## 544. Bilevel Autoresearch: Meta-Autoresearching Itself

**arXiv ID:** 2603.23420 | [PDF](https://arxiv.org/pdf/2603.23420v1)

**作者:** Yaonan Qu `[一作]`, Meng Lu `[通讯]` (University of Hong Kong)

**通讯引用:** 194179 | [OpenAlex ID](https://openalex.org/A5100748869)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一个双层自我调节的自动研究框架，利用同一语言模型在运行时生成并注入新的搜索机制来改进内部的超参数搜索。

**💡 创新点**

创新点在于将外层循环的结构改进完全交给语言模型完成——通过代码生成自动发现并部署如Tabu Search、Bandit和Orthogonal Exploration等搜索策略，而非依赖人工手工设计。

**🔧 技术方法**

技术主要包括：使用DeepSeek LLM进行多轮对话式代码生成、动态加载与验证生成的Python模块、以及在内层循环中实现提议-训练-评估的自我迭代流程。

**📊 数据集**

实验数据集为Karpathy的GPT预训练基准（50M参数模型，RTX 5090 GPU，300秒训练预算）以及其对应的验证损失。

**📈 对比分析**

通过对四个实验组（仅内层、加上参数策略调节、加上机制生成、仅机制生成）进行消融，发现加入机制生成的组C相较于仅内层组A的性能提升约5倍（从-0.009到-0.045），而参数策略调节组B并未带来显著收益。

**⚠️ 局限性**

局限性包括：重复次数仅3次且方差较大、仅验证单一模型/预算组合、生成代码对外部依赖敏感、以及外层循环的提示可能限制了可探索的机制空间。

---

## 545. Planning over MAPF Agent Dependencies via Multi-Dependency PIBT

**arXiv ID:** 2603.23405 | [PDF](https://arxiv.org/pdf/2603.23405v1)

**作者:** Zixiang Jiang `[一作]` (University Of Melbourne), Jiaoyang Li `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4078 | [OpenAlex ID](https://openalex.org/A5027709346)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的多代理路径规划框架 Multi-Dependency PIBT（MD-PIBT），通过搜索代理依赖图来实现多步规划，克服传统 PIBT/EPIBT 只能处理单一依赖的限制。

**💡 创新点**

核心创新是将代理依赖视为规划空间，并在依赖图中引入硬/软依赖、可回溯和多步窗口，既能重现 PIBT/EPIBT 结果，又能处理多重冲突，显著提升在大型代理和大尺寸代理场景下的成功率。

**🔧 技术方法**

使用优先级继承与回溯（PIBT）原理、路径依赖图（Agent Dependency Graph）、多步窗口规划、可配置的回溯策略、以及可调节的冲突容忍度（C）等技术。

**📊 数据集**

在公开 MAPF 基准地图（如 4×4、5×5、9×9 等）以及自构造的大尺寸代理地图上进行实验，模型涵盖 Pebble Motion、Pebble Motion with Large Agents、Rotation Motion、Differential Drive Robots 等多种动力学模型。

**📈 对比分析**

与 PIBT、EPIBT 在一-shot 与 lifelong MAPF 场景下对比，MD-PIBT 在大代理（≤10,000）和大尺寸代理场景中成功率和吞吐量显著优于两者；在普通 PM/RM 代理场景下，MD-PIBT 与 EPIBT/PIBT 的性能相当，但算法开销略高。

**⚠️ 局限性**

局限性包括：算法在极端密集或高维问题下可能因多重依赖导致回溯次数激增；对路径依赖图的维护和超参数（窗口大小、C、回溯阈值）调优仍需经验；与基于学习的方案相比，尚未充分利用数据驱动的搜索策略。

---

## 546. Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning

**arXiv ID:** 2603.23404 | [PDF](https://arxiv.org/pdf/2603.23404v1)

**作者:** Jiacheng Hua `[一作]` (Tsinghua University), Miao Liu `[通讯]` (Tsinghua University)

**通讯引用:** 23663 | [OpenAlex ID](https://openalex.org/A5100348907)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TRACE 方法，利用文本化的全景空间描述作为中间推理轨迹，增强多模态大型语言模型在视频空间推理任务中的表现。

**💡 创新点**

创新点在于将视角转换为结构化的全景空间文本表示，并将其与元上下文、相机轨迹和实体注册相结合，弥补传统模型对 3D 空间缺乏抽象与结构化理解的不足。

**🔧 技术方法**

采用链式思维提示、文本生成、房间对齐坐标系、离散方向相机轨迹、实体注册表以及基于文本的推理解析器等技术，形成完整的 TRACE 框架。

**📊 数据集**

在 VSI‑Bench（288 条真实视频、5,130 题目）和 OST‑Bench（200 场景、1,396 题目）这两个基准上进行实验。

**📈 对比分析**

与 CoT、ToT、LtM、Cognitive Map 等提示策略相比，TRACE 在 Gemini、Qwen、MiMo 等多模态 LLM 上均显著提升性能，最大提升约 +7.54%（Gemini）或 +3.10%（Qwen）等；在 OST‑Bench 上也获得 1.2%–2.4% 的增益。

**⚠️ 局限性**

局限在于当前 TRACE 采用静态全局坐标系统，难以精准跟踪代理即时状态，导致某些代理状态预测任务表现下降；此外，对相机运动估计的依赖可能在某些场景中引入误差。

---

## 547. From Feature Learning to Spectral Basis Learning: A Unifying and Flexible Framework for Efficient and Robust Shape Matching

**arXiv ID:** 2603.23383 | [PDF](https://arxiv.org/pdf/2603.23383v1)

**作者:** Feifan Luo `[一作]` (Zhejiang University), Hongyang Chen `[通讯]` (Zhejiang Lab)

**通讯引用:** 39665 | [OpenAlex ID](https://openalex.org/A5010419481)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了可学习谱基的高级功能映射框架，并实现了端到端无监督的谱基学习。

**💡 创新点**

创新点在于把谱基变为可学习的抑制函数，实现谱基与特征同步优化；并通过多尺度无参数热扩散与单一损失实现高效无监督训练。

**🔧 技术方法**

使用DiffusionNet特征提取器、可学习的抑制函数、热扩散网络、软最大点对点匹配和谱映射投影等技术。

**📊 数据集**

在FAUST、SCAPE、SHREC’19、SMAL、DT4D‑H、TOPKIDS以及其非等距重网格版本上进行实验。

**📈 对比分析**

与多种基线方法（如ZoomOut、FMNet、GeomFmaps、ULRSSM等）比较，实验表明在近等距、跨数据集、非等距以及拓扑噪声等场景下，本文方法的平均地理误差显著低于现有SOTA，且不需要昂贵的线性求解器或测试时微调。

**⚠️ 局限性**

在极端非等距变形或局部缺失（partial）场景下仍会出现性能衰退，需要结合变形感知或外部几何信息。

---

## 548. End-to-End Efficient RL for Linear Bellman Complete MDPs with Deterministic Transitions

**arXiv ID:** 2603.23461 | [PDF](https://arxiv.org/pdf/2603.23461v1)

**作者:** Zakaria Mhammedi `[一作]` (Google Research), Nneka Okolo `[通讯]` (MIT)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在线性Bellman完备MDP下，提出了针对确定性转移、随机奖励与随机初始状态的计算效率高的RL算法。

**💡 创新点**

通过结合逼近线性优化的OCP、球面支撑技术与FQI，克服了传统算法对动作空间大小和Q函数参数上界的限制。

**🔧 技术方法**

利用OCP、逼近线性优化、球面支撑（barycentric spanner）、最小范数最小二乘回归、Ridge回归以及FQI等技术。

**📊 数据集**

本文为理论分析与证明，未使用实际数据集。

**📈 对比分析**

与以往仅在小动作空间或需要强oracle的算法相比，本文在样本与计算复杂度均保持多项式级别，且仅对动作空间大小无依赖（仅需argmax oracle）。

**⚠️ 局限性**

仅适用于确定性转移的MDP，仍未解决随机转移的情况；对极大动作空间仍需argmax oracle；缺乏实证验证。

---

## 549. SortedRL: Accelerating RL Training for LLMs through Online Length-Aware Scheduling

**arXiv ID:** 2603.23414 | [PDF](https://arxiv.org/pdf/2603.23414v1)

**作者:** Yiqi Zhang `[一作]` (National University of Singapore), Yang You `[通讯]` (National University of Singapore)

**通讯引用:** 3912 | [OpenAlex ID](https://openalex.org/A5100658705)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 SortedRL，一种在线长度感知调度策略，旨在缓解大语言模型强化学习训练中因生成长度不齐导致的“bubble”问题，并通过可控离线训练提升样本效率。

**💡 创新点**

创新点包括：① 在线按生成长度排序并分组进行回合采样，构建微型课程；② 引入可调离线比例控制，兼顾收敛稳定性和样本利用率；③ 设计专用的长度感知调度器和有状态回合缓冲区，实现动态批量化与同步控制。

**🔧 技术方法**

技术手段涵盖：强化学习算法（PPO / Reinforce++）、自回归生成加速（SGLang/VeRL、PagedAttention、Radix Attention）、长度感知调度与缓存管理、批量化训练与离线/在线混合更新。

**📊 数据集**

实验使用的主要数据集有：逻辑推理数据集 LogicRL（5000个 Knights-and-Knaves 逻辑谜题）和混合数学数据集 DAPO‑Math‑17k，评估指标包括 AIME24、MATH500、Minerva、OlympiadBench、GSM8k、AMC2023 等竞赛级数学测评。

**📈 对比分析**

与基线（传统大批量离线 RLHF）对比，SortedRL 在相同训练样本量下提升 3.9%–18.4% 的性能，AIME24 上升 18% 以上；同时将回合阶段 bubble 比例从 74% 降至 <5.81%，实现近 40% 的吞吐量提升。

**⚠️ 局限性**

局限性包括：① 对组大小（group size）的敏感性，需要手工调优；② 部分离线模式在某些任务（如 GSM8k）表现反向；③ 需要额外的系统复杂度（长度调度器与缓冲区实现）以及对硬件资源（GPU 频繁切换）有更高的依赖。

---

## 550. RealMaster: Lifting Rendered Scenes into Photorealistic Video

**arXiv ID:** 2603.23462 | [PDF](https://arxiv.org/pdf/2603.23462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 551. MRATTS: An MR-Based Acupoint Therapy Training System with Real-Time Acupoint Detection and Evaluation Standards

**arXiv ID:** 2603.23445 | [PDF](https://arxiv.org/pdf/2603.23445v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 552. GeoSANE: Learning Geospatial Representations from Models, Not Data

**arXiv ID:** 2603.23408 | [PDF](https://arxiv.org/pdf/2603.23408v1)

**作者:** Joelle Hanna `[一作]`, Damian Borth `[通讯]` (University of St.Gallen)

**通讯引用:** 6418 | [OpenAlex ID](https://openalex.org/A5065722787)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过学习现有遥感模型的权重空间，构建统一的潜在表示，并能够按需生成任意架构、任务的模型权重；

**💡 创新点**

①在权重空间而非原始图像上进行预训练；②将多种遥感模型（Transformer、CNN、多模态等）嵌入同一潜在空间；③利用该空间直接生成轻量化模型，性能优于剪枝与蒸馏；

**🔧 技术方法**

权重token化、Transformer序列到序列autoencoder（GPT‑2风格）、重构+对比学习损失、KDE采样生成权重；

**📊 数据集**

103个公开遥感模型（约38B参数）作为训练数据；10个遥感任务数据集（分类、分割、检测）及GEO‑Bench四个基准；

**📈 对比分析**

与从零训练、现有RSFM、模型合并、剪枝/蒸馏以及提示模型进行比较；GeoSANE在所有10个基准上匹配或超过最先进模型，且生成的轻量化网络优于传统压缩方法；

**⚠️ 局限性**

依赖已公开模型的多样性，训练autoencoder需要大量参数；生成模型仍需微调；对极其不同的架构或任务的泛化仍有限。

---

## 553. ConceptCoder: Improve Code Reasoning via Concept Learning

**arXiv ID:** 2603.23470 | [PDF](https://arxiv.org/pdf/2603.23470v1)

**作者:** Md Mahbubur Rahman `[一作]` (Iowa State University), Wei Le `[通讯]` (Iowa State University)

**通讯引用:** 1841 | [OpenAlex ID](https://openalex.org/A5101799063)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ConceptCoder框架，通过先学习代码概念再进行代码推理，实现对LLM的概念监督多任务微调，提升漏洞检测和分支预测等代码推理任务的表现。

**💡 创新点**

创新点在于：①首次为代码定义可解释的“代码概念”，并用显式监督方式训练模型识别；②将概念学习与推理任务联合训练，形成概念瓶颈模型（CBM）风格的代码推理；③证明概念学习显著提升模型的语义理解、鲁棒性与泛化。

**🔧 技术方法**

技术手段包括：多任务学习（概念头+任务头）、线性探针评估概念理解、基于静态分析工具生成的概念标签、LLM微调（使用bf16、Adam、10个epoch）等。

**📊 数据集**

数据集：PrimeVul、DriverseVul（漏洞检测）；构造的概念数据集（4类漏洞+7概念，共28,974样本）和通用漏洞数据集（80,204样本、134 CWE）；CodeNet（52,060样本）用于分支预测。

**📈 对比分析**

对比方法：标准SFT、TRACED、DeepDFA、GPT‑5.2、Claude‑Opus等；结果显示ConceptCoder在概念数据集上的平均F1从66.32提升至72.15，在通用数据集上从55.11提升至58.52；最佳模型Qwen2.5‑7B在概念数据集上达到74.76（高于所有SOTA），分支预测上F1从85.27提升至86.50，超越TRACED。

**⚠️ 局限性**

局限性：①概念覆盖面有限，仅覆盖四类漏洞的七个概念；②需要先行静态分析生成概念标签，生成成本和可扩展性待验证；③对极大规模LLM或极端数据不平衡场景的效果尚未评估；④概念识别在分支预测任务中仍相对困难，F1提升幅度有限。

---

## 554. CSTS: A Canonical Security Telemetry Substrate for AI-Native Cyber Detection

**arXiv ID:** 2603.23459 | [PDF](https://arxiv.org/pdf/2603.23459v1)

**作者:** Abdul Rahman `[一作]` (Howard University), Abdul Rahman `[通讯]` (Howard University)

**通讯引用:** 2574 | [OpenAlex ID](https://openalex.org/A5033450085)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Canonical Security Telemetry Substrate（CSTS），一种统一的实体-关系时间索引抽象层，用于替代事件级别的安全遥测表示，提升跨环境部署的可迁移性。

**💡 创新点**

创新点在于将身份持久化、关系类型化、时间状态化与schema治理融合到一个可版本化的中间件抽象，分离遥测摄取与检测逻辑，显著降低部署碎片化。

**🔧 技术方法**

采用图数据建模、实体解析与关系抽象、时间窗口聚合、基于属性的特征构造以及传统机器学习/图神经网络等多种检测技术。

**📊 数据集**

使用人工合成的跨拓扑环境（Env A→Env B）数据集、公开Sysmon日志、DARPA TC E3 产商对比日志以及Otrf Security-Datasets等多源安全日志数据。

**📈 对比分析**

通过在相同模型架构和窗口设置下对比事件级别基线与CSTS管道，在横向移动检测中CSTS实现了从0.135 F1提升至0.508、AUROC提升至0.991，且在schema扰动下仍保持非零预测；在零日检测中尽管CSTS保持schema稳健但出现语义方向反转（AUROC<0.5）。

**⚠️ 局限性**

限制包括：CSTS在流式零日检测中未解决语义方向不稳定；实验主要在合成与公开日志上，缺乏大规模真实企业环境的纵向验证；对不同模型的调参未做充分探索。

---

## 555. MuSe: a Mutation Testing Plugin for the Remix IDE

**arXiv ID:** 2603.23441 | [PDF](https://arxiv.org/pdf/2603.23441v1)

**作者:** Gerardo Iuliano `[一作]` (University of Salerno), Dario Di Nucci `[通讯]` (University of Salerno)

**通讯引用:** 2470 | [OpenAlex ID](https://openalex.org/A5072127726)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现MuSe插件，将Mutation Testing集成到Remix IDE，为智能合约提供50个变异算子（包括传统、Solidity特定和安全导向），并支持直接在IDE中生成变异、运行测试、生成报告。

**💡 创新点**

在SuMo基础上新增六个安全导向变异算子，实现安全漏洞注入；首次将Mutation Testing直接集成到Remix IDE；通过AST级别变异和TCE技术消除等价变异，提升变异质量。

**🔧 技术方法**

使用JavaScript/TypeScript与Node.js 20+；Solidity-parser-antlr解析AST，Visitor模式实现变异；Docker容器部署；Remix插件框架；与Truffle/Hardhat/Brownie/Forge等测试框架对接；Slither静态分析工具做验证；统计学方法（95%置信度）验证变异有效性。

**📊 数据集**

SmartBugs-Wild数据集（47,398份智能合约）用于生成350,716+变异合约，并随机抽样384份进行人工验证。

**📈 对比分析**

将生成的变异合约交给Slither进行检测，统计各漏洞的召回率：UC、US均为1.00；CL 0.81；UR 0.63；TX 0.33；DTU 0.10；平均召回率0.597。插件可实时生成HTML报告并在IDE中查看，验证成功率超过95%。

**⚠️ 局限性**

安全导向算子覆盖有限（仅6种），部分合约无法注入漏洞（失败率5%）；未重新验证传统算子；工具依赖Remix与Solidity-parser-antlr版本；缺乏对大型或复杂合约的性能评估；与其他IDE或测试框架的集成尚未实现。

---

## 556. Biased Error Attribution in Multi-Agent Human-AI Systems Under Delayed Feedback

**arXiv ID:** 2603.23419 | [PDF](https://arxiv.org/pdf/2603.23419v1)

**作者:** Teerthaa Parakh `[一作]` (Georgia Institute of Technology), Karen M. Feigh `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1798 | [OpenAlex ID](https://openalex.org/A5041686916)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个多代理、递归决策的游戏实验，研究了延迟反馈如何导致人类在与多 AI 代理交互时出现偏误责任归因；

**💡 创新点**

创新点在于将延迟反馈与多代理系统结合，揭示了在多步决策场景中出现的“偏误错误归因”现象，并强调了时间信用分配的难度；

**🔧 技术方法**

主要技术包括使用基于 PPO 的 Actor‑Critic 多代理 AI（DefenseAI 与 OffenseAI）、定量策略标签化、Poisson 回归分析以及实验对照设计（Confidence vs. No‑Confidence 条件）；

**📊 数据集**

使用自定义的海军战术游戏数据，涵盖 26 名玩家在 6 次测试游戏中的决策轨迹、AI 建议和最终得分；

**📈 对比分析**

通过与 AI 推荐策略对比以及对良好/不良结果后策略转换的统计显著性检验，实验发现负面结果导致更大且更频繁的策略调整，但这些调整往往不对应最能改善绩效的决策维度，表现出明显的偏误；

**⚠️ 局限性**

局限包括未分析玩家的主观心理模型、使用经验性最优策略而非理论最优、参考 AI 建议序列可能因人机交互而变化，以及实验持续时间短、样本量有限。

---

## 557. An Experimental Study of Machine Learning-Based Intrusion Detection for OPC UA over Industrial Private 5G Networks

**arXiv ID:** 2603.23416 | [PDF](https://arxiv.org/pdf/2603.23416v1)

**作者:** Song Son Ha `[一作]` (Helmut-Schmidt-University), Gerd Scholl `[通讯]` (Helmut-Schmidt-University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了真实私有5G环境下的OPC UA实验平台，执行多类OPC UA攻击，采集流量并通过DPI提取协议感知特征，训练并评估多种监督机器学习模型实现入侵检测。

**💡 创新点**

①在工业级私有5G网络中首次系统评估OPC UA流量与攻击；②提出多层协议感知与流量统计相结合的特征工程；③通过对比不同ML模型（LogReg、SVM、RF、GB、XGBoost、Voting）在复杂攻击场景下的表现，验证私有5G对IDS性能影响有限。

**🔧 技术方法**

使用OPC UA、私有5G Standalone、R&S PACE 2 DPI引擎、流量与协议级特征提取、机器学习框架（scikit‑learn、XGBoost）。

**📊 数据集**

自建PCAP数据集，包括10个纯正常流量记录和多组攻击场景（HEL‑F、OMSC、CHUNK‑F、PUB‑F、COND‑REF、BROWSE、READ‑EXP、NESTED、TBP），每组攻击持续3–9 min，覆盖不同攻击强度与参数。

**📈 对比分析**

采用PCAP级别训练/验证/测试分离，评估各模型F1分数；大多数攻击场景F1>0.95，复杂低速攻击场景仍达0.88以上，表明模型对私有5G引入的时序变异不敏感，整体检测性能优异。

**⚠️ 局限性**

仅评估未加密OPC UA、攻击场景有限、未覆盖长周期概念漂移、缺乏无监督或在线学习方法、数据集来源单一工厂，限制了结果推广性。

---

## 558. Rectify, Don't Regret: Avoiding Pitfalls of Differentiable Simulation in Trajectory Prediction

**arXiv ID:** 2603.23393 | [PDF](https://arxiv.org/pdf/2603.23393v1)

**作者:** Harsh Yadav `[一作]` (University of Wuppertal), Tobias Meisen `[通讯]` (University of Wuppertal)

**通讯引用:** 3543 | [OpenAlex ID](https://openalex.org/A5032638290)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对闭环目标轨迹预测中的梯度泄露导致的“后悔学习”问题，提出了一种非可微分的迭代滚动回放框架，使模型在面对漂移状态时必须真正学习恢复行为。

**💡 创新点**

创新点在于：①通过显式断开模拟步骤间的计算图，阻止未来真实轨迹信息向过去预测泄漏；②将闭环训练与开环训练统一到同一损失框架；③改造LMFormer为纯解码器结构（LMFormer‑D），实现高频迭代预测。

**🔧 技术方法**

使用的技术包括：非可微分的离线场景回放（log‑replay）、梯度截断的回放回合、纯解码器Transformer（LMFormer‑D）以及多模态损失分解。

**📊 数据集**

使用的数据集为nuScenes（完整未来轨迹）和DeepScenario（Busy Frankfurt子集），后者提供高精度交叉口场景。

**📈 对比分析**

方法与传统可微分闭环训练和开环基线比较：在高频重规划（H_step=0.5s）下，非可微分闭环模型相较于可微分闭环减少碰撞率高达33.24%，相较于开环基线减少27.74%；同时在多模态多样性和车道对齐上表现更优。

**⚠️ 局限性**

局限性包括：仅使用log‑replay的周围车辆运动，缺乏完全可交互的多智能体模拟；对训练频率的进一步提升仍需更高计算资源；并且只适用于提供完整未来轨迹的数据集，无法直接迁移到如Waymo或Argoverse-2等缺失未来轨迹的场景。

---

## 559. Graph Energy Matching: Transport-Aligned Energy-Based Modeling for Graph Generation

**arXiv ID:** 2603.23398 | [PDF](https://arxiv.org/pdf/2603.23398v1)

**作者:** Michal Balcerak `[一作]` (University of Zurich), Bjoern Menze `[通讯]` (University of Zurich)

**通讯引用:** 27010 | [OpenAlex ID](https://openalex.org/A5002068604)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了Graph Energy Matching（GEM）框架，用于从噪声或已知数据开始高质量生成分子图。

**💡 创新点**

创新点在于将连续空间的Energy Matching与JKO方案推广到离散图空间，结合了确定性、运输对齐的采样与离散混合采样，并实现了显式相对似然建模。

**🔧 技术方法**

采用了可微分的离散图提议（greedy与Langevin），利用能量梯度指导采样，结合对数概率损失、对比学习与温度调度，并用神经网络学习标量能量潜能。

**📊 数据集**

使用了QM9（约15万分子）和MOSES（约150万分子）两大分子数据集进行实验。

**📈 对比分析**

与离散扩散模型、DeFoG、vfm等基线比较，GEM在无条件生成时在有效率、FCD、有效率/新颖度等指标上均优于或匹配现有最佳方法，且在条件生成与几何路径分析中表现更好。

**⚠️ 局限性**

限制在于能量梯度驱动采样导致理论上约2–3倍的计算开销，并且在极端离散域中仍需进一步验证其泛化性。

---

## 560. ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment

**arXiv ID:** 2603.23376 | [PDF](https://arxiv.org/pdf/2603.23376v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 561. "I Might be Using His... But It is Also Mine!": Ownership and Control in Accounts Designed for Sharing

**arXiv ID:** 2603.23391 | [PDF](https://arxiv.org/pdf/2603.23391v1)

**作者:** Ji Eun Song `[一作]` (Seoul National University), Joongseek Lee `[通讯]` (Seoul National University)

**通讯引用:** 543 | [OpenAlex ID](https://openalex.org/A5003626110)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过问卷与访谈研究了共享式订阅平台（DS）账户的所有权认知和控制问题。

**💡 创新点**

创新点在于提出了‘Casual’和‘Cost-splitting’两种共享模式，并发现成本分摊共享中出现的‘Dual所有权’现象及其导致的控制冲突。

**🔧 技术方法**

研究采用混合方法，结合在线问卷调查和31名受访者的半结构化访谈。

**📊 数据集**

数据集包括160名共享用户的问卷数据以及31名用户的访谈记录。

**📈 对比分析**

方法上主要使用描述性统计和主题编码进行分析，未涉及数值性能对比；研究结果以定性发现为主。

**⚠️ 局限性**

局限性包括仅研究视频流媒体用户、未覆盖被动共享用户、未涉及未成年人，以及缺乏对共享持续性的量化评估。

---

## 562. VTAM: Video-Tactile-Action Models for Complex Physical Interaction Beyond VLAs

**arXiv ID:** 2603.23481 | [PDF](https://arxiv.org/pdf/2603.23481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 563. SIMART: Decomposing Monolithic Meshes into Sim-ready Articulated Assets via MLLM

**arXiv ID:** 2603.23386 | [PDF](https://arxiv.org/pdf/2603.23386v1)

**作者:** Chuanrui Zhang `[一作]` (ByteDance Seed), Ziwei Wang `[通讯]` (NTU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将静态3D网格转换为功能化、可模拟的多关节资产，直接生成分部网格和URDF结构。

**💡 创新点**

提出统一的多模态学习框架SIMART，结合稀疏3D VQ‑VAE和大型视觉语言模型，实现端到端的关节推理与几何生成，并通过稀疏token化大幅降低上下文长度。

**🔧 技术方法**

稀疏3D VQ‑VAE、Qwen3‑VL 8B多模态Transformer、基于点云的分割、结构化URDF生成等技术。

**📊 数据集**

使用 PhysXNet、PartNet‑Mobility 以及新构建的 SIMART‑Bench 作为训练与评估数据集。

**📈 对比分析**

与 Urdformer、Articulate‑Anything、Physx‑Anything、Particulate 等基线在 Type、Axis、Origin、IoU、CD 等指标上对比，SIMART 在 ID 与 AI‑generated 两种测试集上均达到或超过 90% 的准确率，显著优于其他方法。

**⚠️ 局限性**

主要限制在于可用的有标注关节信息的3D数据仍稀缺，导致模型在极端或新类别物体上的泛化仍有限。

---

## 564. FG-Portrait: 3D Flow Guided Editable Portrait Animation

**arXiv ID:** 2603.23381 | [PDF](https://arxiv.org/pdf/2603.23381v1)

**作者:** Yating Xu `[一作]` (National University of Singapore), Jifei Song `[通讯]` (University of Surrey)

**通讯引用:** 1010 | [OpenAlex ID](https://openalex.org/A5046874089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于3D流的FG-Portrait，利用3D头模型构造源-驱动之间的几何对应，并通过深度引导采样将其编码为控制信号，集成到ControlNet的扩散模型中实现高质量、可编辑的人像动画

**💡 创新点**

核心创新在于：①使用学习‑free的3D流作为运动对应，避免传统2D流预测的不确定性；②将3D流编码与深度引导采样结合，精确对齐2D运动；③支持推理阶段的表达与姿态手动编辑

**🔧 技术方法**

技术包括：FLAME参数化3D头模型、3D流生成与编码、深度引导采样、Stable Diffusion U-Net+ControlNet、LPIPS/CSIM/APD/AED/FSID等评估指标

**📊 数据集**

主要使用VFHQ（高分辨率面部视频）与FFHQ（人脸图像）作为训练与测试数据集

**📈 对比分析**

与EMOPortrait、X-Portrait、Follow-Your-Emoji、Face-Adapter、HunyuanPortrait等基线相比，在自重现与跨重现任务中均取得更低的APD/AED、更优的图像质量（LPIPS/FID）和较高的身份保持（CSIM），显示出明显性能优势

**⚠️ 局限性**

局限性在于FLAME模型的网格分辨率有限，难以细致表达微表情，导致极细微表情迁移效果受限

---

## 565. ReqFusion: A Multi-Provider Framework for Automated PEGS Analysis Across Software Domains

**arXiv ID:** 2603.23482 | [PDF](https://arxiv.org/pdf/2603.23482v1)

**作者:** Muhammad Khalid `[一作]` (Constructor University Bremen), Yilmaz Uygun `[通讯]` (Constructor University Bremen)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ReqFusion框架，自动化提取、分类和分析软件需求，采用PEGS方法组织需求并支持跨文档类型。

**💡 创新点**

将多模型LLM协同与PEGS结构化提示、投票共识机制相结合，显著降低幻觉率并提升提取准确率。

**🔧 技术方法**

使用多供应商LLM（OpenAI GPT‑4、Anthropic Claude‑3、Groq Llama）、并行/顺序模式、共识投票、cosine相似度去重，前端React+TS、后端FastAPI、PostgreSQL/ChromaDB/Redis/S3存储。

**📊 数据集**

使用18份真实文档（学术、工业、招标）和5个德国工业招标项目（30GB、1,050条需求）进行评估。

**📈 对比分析**

通过与手工基准和单一LLM对比，采用精确率、召回率、F1、时间、成本、PEGS完整度等指标；多模型共识F1 0.88，比单模型提升+0.17，时间减少78%，成本节约47%，一致性98%。

**⚠️ 局限性**

评估主要聚焦德语工业招标文档，跨语言与行业推广受限；标注专家数量有限；缺乏与商业RE工具的定量对比。

---

## 566. SNARE: A TRAP for Rational Players to Solve Byzantine Consensus in the 5f+1 Model

**arXiv ID:** 2603.23458 | [PDF](https://arxiv.org/pdf/2603.23458v1)

**作者:** Alejandro Ranchal-Pedrosa `[一作]` (Sei Labs), Benjamin Marsh `[通讯]` (Sei Labs)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出了基于奖励与排除的可扩展 Nash 协议 (Scalable Nash Agreement via Reward and Exclusion)，在 5f+1 体系中通过可计量的可观测一致性与一次性最终化相结合实现理性共识。

**💡 创新点**

创新点在于：① 在任何 5f+1 协议后追加一次 4f+1 阈值的全向广播，仅增加一条消息延迟即可将安全阈值从约 20% 提升至约 60%；② 通过改进有效候选性（valid‑candidacy）消除原有的 n>2(k+t) 与 n>3/2k+3t 约束，令协议在 73% 的联盟规模下仍保持 ϵ‑(k,t)‑鲁棒；③ 使用三倍的 3f+1 阈值交叉检测实现 3 倍安全放大并显著降低存款阈值（<0.5%）。

**🔧 技术方法**

使用的技术包括：分层 BFT（可计量一致性 + 一次性 finalization）、可靠广播、可欺骗性检测（Proof‑of‑Fraud）、奖励与扣押机制、奖池赢家共识（Winner Consensus）以及基于博弈的“诱饵”策略。

**📊 数据集**

本研究主要为理论分析与形式化证明，并以 101 名验证者（f=20）为示例说明存款与收益比例；未使用实际区块链数据集。

**📈 对比分析**

通过对比 3f+1 与 5f+1 传统协议的分支阈值、联盟规模、存款需求等参数，绘制多张阈值曲线与存款表格，结果表明：在不需要存款的“无分叉”区间可达 60% 的联盟，双重/三重消费区间的存款均低于 0.2%–0.5%，显著优于现有方案。

**⚠️ 局限性**

局限性：① 赢家共识是绑定瓶颈，最大联盟规模约为 74%（若采用固定奖励可升至约 80%）；② 仅在固定 n=5f+1 的网络下证明，动态加入/退出的场景尚未覆盖；③ 对于超过无分叉阈值的情形仍需存款，且在极端攻击下可能出现协作失败；④ 异步环境下的实现需要进一步细化。

---

## 567. Evaluating LLM-Based Test Generation Under Software Evolution

**arXiv ID:** 2603.23443 | [PDF](https://arxiv.org/pdf/2603.23443v1)

**作者:** Sabaat Haroon `[一作]` (Virginia Tech), Muhammad Ali Gulzar `[通讯]` (Virginia Tech)

**通讯引用:** 846 | [OpenAlex ID](https://openalex.org/A5003747461)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过大规模实验评估LLM在代码演进（语义改变与语义保持）下自动生成单元测试的鲁棒性；

**💡 创新点**

创新点在于引入基于变异的评测框架，结合语义改变与保持两类代码修改，系统性揭示LLM对语义与语法差异的敏感度及回归意识缺失；

**🔧 技术方法**

技术手段包括：两轮提示生成测试、自动化覆盖率测量、失败测试的重跑对比分析以及测试套件匹配与 churn 统计；

**📊 数据集**

使用的基准是 CodeNet 数据集中的 22,374 个 Java/Python 程序变体，涵盖 8 大 LLM（包括 GPT‑5、Claude‑4、Gemini 等）；

**📈 对比分析**

比较方法是对原始、语义改变、语义保持三种版本分别生成测试，测量行覆盖、分支覆盖、通过率与测试套件持续性；在原始程序上平均覆盖率约 79%/76%，语义改变后通过率降至 66%/60%，语义保持后也出现 21% 的下降，表明 LLM 主要受表面语法影响；

**⚠️ 局限性**

局限性包括：仅评估单文件、Python/Java 的简单算法任务，难以推广到大型多文件项目；评价指标局限于覆盖率与通过率，未考察缺陷发现能力；LLM 随机性及数据泄露可能影响结果；仅选取 8 模型，缺乏更广泛的模型覆盖。

---

## 568. Integrating GenAI in Filmmaking: From Co-Creativity to Distributed Creativity

**arXiv ID:** 2603.23415 | [PDF](https://arxiv.org/pdf/2603.23415v1)

**作者:** Pierluigi Masai `[一作]`, Mateusz Miroslaw Lis `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过社会技术史视角，系统性分析了生成式人工智能（GenAI）在电影制作中的角色，提出将其视为分布式创造力的媒介而非单纯的共创助手，并构建了资产增强、资产编辑与资产生成三大技术类别的分析框架。

**💡 创新点**

创新点在于：① 把GenAI从“共创”框架转向“分布式创造力”框架，强调其与人类创作者、设备和工作流程共同构成的社会技术网络；② 提出针对电影制作的三类技术分类法，阐明不同技术如何重塑生产流程、职业角色与美学表达；③ 通过对历史技术变革与当代GenAI的对照，展示技术如何持续推动电影美学与生产方式的演进。

**🔧 技术方法**

主要采用STS理论、文献综述与案例分析方法；未使用特定机器学习算法实现，而是聚焦于技术应用场景与社会技术交互。

**📊 数据集**

未使用任何实验数据集；研究依据为电影史文献、行业案例、技术报道和学术理论。

**📈 对比分析**

本文并未进行实验比较或性能评估；通过理论分析和案例说明技术对工作流程、成本与美学可能产生的影响，强调需进一步实证研究以量化这些影响。

**⚠️ 局限性**

局限性包括：① 研究以文献和理论为主，缺乏实证数据支持；② 对GenAI技术在实际电影制作中的技术性能（如分辨率、时序一致性）未做深入评估；③ 对美学效果和行业经济影响的量化分析不足，未来需开展实地工作流程研究和案例实验。

---

## 569. Natural Language Interfaces for Spatial and Temporal Databases: A Comprehensive Overview of Methods, Taxonomy, and Future Directions

**arXiv ID:** 2603.23375 | [PDF](https://arxiv.org/pdf/2603.23375v1)

**作者:** Samya Acharja `[一作]` (Marquette University), Kanchan Chowdhury `[通讯]` (Marquette University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对空间与时空数据库的自然语言接口（NLIDB）进行系统综述，整理方法、数据集与评估指标，并对已有技术进行对比分析。

**💡 创新点**

提出了ST‑NLIDB的完整分类框架（规则/语法、神经/语义解析、LLM、多代理四类），并系统总结了三类数据集（Benchmark、Real‑GIS、Synthetic）与多维评估维度，指出了当前研究的空白与挑战。

**🔧 技术方法**

主要引用并梳理了规则/语法驱动、神经网络+模板、LLM提示、以及多代理推理等技术路线，并对各自的模型实现与训练方式作了概括。

**📊 数据集**

讨论了常用数据集：GeoQuery、GeoAnQu、GeoSQL‑Bench、TRUCE、SUSHI、BerlinMOD、SECONDO、SpatiaLite、OverpassNL 等，强调了它们在覆盖空间/时间算子、语义多样性和现实性方面的差异。

**📈 对比分析**

通过对不同方法在相同指标（执行准确率、翻译准确率、检索质量、效率等）的对比，表明：规则/语法系统可解释但泛化差；神经+模板提升鲁棒性但受限于模板；LLM系统适应性强但易出现空间幻觉和提示敏感；多代理系统能提高准确率但计算成本高；整体上仍缺乏统一基准，跨系统比较受限。

**⚠️ 局限性**

局限性包括：缺乏统一、公开、涵盖空间/时间多样性的基准；数据集规模小或过度人工；评估指标不统一；LLM方法缺乏深度空间推理、易失真；多代理系统复杂度高、成本大；大多数方法未充分利用空间表示学习与跨模态关联技术。

---

## 570. UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation

**arXiv ID:** 2603.23500 | [PDF](https://arxiv.org/pdf/2603.23500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 571. Byzantine-Robust and Differentially Private Federated Optimization under Weaker Assumptions

**arXiv ID:** 2603.23472 | [PDF](https://arxiv.org/pdf/2603.23472v1)

**作者:** Rustem Islamov `[一作]` (University of Basel), Eduard Gorbunov `[通讯]` (MBZUAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了新的联邦学习算法 Byz-Clip21-SGD2M，能够在同时面对差分隐私（DP）噪声和拜占庭攻击时保证收敛。

**💡 创新点**

创新点在于：①将双动量机制与鲁棒聚合、剪裁和误差反馈相结合，形成一种“Clip+Momentum+Robust”复合策略；②在高概率分析框架下，去除了以往研究中普遍使用的梯度有界假设，仅依赖 L‑光滑性和 σ‑sub‑高斯梯度噪声；③给出同时考虑 DP 与拜占庭攻击的收敛速率与误差上界，并证明与已知下界相匹配。

**🔧 技术方法**

使用技术包括：双动量（client 与 server 端），剪裁与 Gaussian DP 噪声注入，鲁棒聚合规则（满足 (c, )‑鲁棒性），误差反馈（EF21 风格）以及高概率收敛分析。

**📊 数据集**

在 MNIST 数据集上使用 CNN 与 MLP 两种模型进行实验，训练时随机分配 20 个正规客户端，加入不同数量的拜占庭客户端。

**📈 对比分析**

与基线方法 Byz-Clip-SGD 与 Safe-DSHB 进行对比。实验结果显示，在各种拜占庭数量与 ε 预算下，Byz-Clip21-SGD2M 的测试准确率与基线相当或更优，且理论证明的误差上界与实验表现高度一致。

**⚠️ 局限性**

限制包括：①仅在初始化点满足梯度离散性假设（bounded heterogeneity），不适用于全局无约束的异质性；②需要可实现的 (c, )‑鲁棒聚合规则；③对梯度噪声仍假设 σ‑sub‑高斯；④未讨论重心子采样或大模型/高维情况下的隐私放大；⑤未给出对重尾噪声或更宽松平滑性假设的下界。

---

## 572. A Joint Reinforcement Learning Scheduling and Compression Framework for Teleoperated Driving

**arXiv ID:** 2603.23387 | [PDF](https://arxiv.org/pdf/2603.23387v1)

**作者:** Giacomo Avanzi `[一作]` (University of Padova), Michele Zorzi `[通讯]` (University of Padova)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在 5G/6G V2X 自动驾驶场景下，设计并实现了基于强化学习的协同压缩与分布式调度框架，以在满足时延阈值的前提下提升激活映射检测的 mAP。

**💡 创新点**

创新点在于：①将压缩参数与资源调度协同优化，突破单一压缩或单一调度的瓶颈；②引入 Meta 级强化学习（Meta CS）动态选择集中式或分布式训练模式，实现跨信道自适应；③在 ns‑3 V2X 仿真中联合评估延迟、成功率与检测精度。

**🔧 技术方法**

核心技术包括：深度强化学习（DDPG、TD3、PPO、Meta‑RL）、多臂赌博机（MAB）决策、ns‑3+SUMO 网络仿真、以及 LiDAR 点云压缩（Draco/Draco‑L）和分布式调度算法（SS、CS、MCS）。

**📊 数据集**

使用数据集：nuScenes LiDAR 检测数据集（3D 激活映射 mAP），以及 SUMO Bologna 城市交通轨迹数据用于网络仿真。

**📈 对比分析**

对比方法：将协同压缩+调度（CCS）、独立压缩（C）、独立调度（S）以及集中式/分布式两种训练方式进行对比。实验显示，CCS 在 Bad 信道下平均奖励提升约 15‑20%，mAP 提升 20‑30%，平均时延下降 30‑40%，满足 τ 的概率提升 10‑20%。在 Good 信道下，集中式模式略优，但分布式与其差距不大。

**⚠️ 局限性**

局限性包括：仅在有限的 UE 数量（5）与两种信道状态（Good/Bad）下验证；压缩方案仅支持 Draco 级别；Meta‑RL 的训练时间较长，且对信道模型假设敏感；缺乏真实车联网环境验证，难以评估多样化车辆和动态网络负载的适应性。

---

## 573. Dynamic Light Spanners in Doubling Metrics

**arXiv ID:** 2603.23490 | [PDF](https://arxiv.org/pdf/2603.23490v1)

**作者:** Sujoy Bhore `[一作]` (Indian Institute of Technology Bombay), Arnold Filtser `[通讯]` (Bar Ilan University)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种新的动态轻量级稀疏网格生成器，可在点集发生插入、删除时以多项式对数时间维护（1+ε）-spanner，并保持其总权重与最小生成树的比例为常数；

**💡 创新点**

创新点在于构造“延迟贪心”spanner：在保持网树结构的同时，仅对受更新影响的局部区域进行边的增删，避免传统贪心算法导致的全局重建；通过粗略距离估计与多尺度距离维护，进一步降低更新复杂度；

**🔧 技术方法**

主要技术包括：1) 动态网树与多尺度网的快速维护；2) 对网树 spanner 的延迟贪心判定，用粗糙距离近似（coarse approximation）代替精确距离；3) 采用多尺度 sketch 图（H）计算子图内距离，从而实现 O((ε⁻¹·logΦ)^O) 的更新时间；

**📊 数据集**

文中未使用具体实验数据集，研究以理论分析为主；

**📈 对比分析**

与现有仅提供稀疏性或无轻量级保证的动态 spanner 算法相比，该方法在保持 O(ε⁻O(d)) 光度与 O((ε⁻¹·logΦ)^O) 更新时间上取得最优理论保证；

**⚠️ 局限性**

局限性在于更新时间与空间仍依赖于点集的 aspect ratio Φ 以及维度 d，且对 ε 的依赖较大；如何将 logΦ 替换为 log n、或降低 d 的指数仍是开放问题。

---

## 574. WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG

**arXiv ID:** 2603.23497 | [PDF](https://arxiv.org/pdf/2603.23497v1)

**作者:** Zhen Li `[一作]` (Alaya Studio Shanda AI Research Tokyo), Kaipeng Zhang `[通讯]` (Alaya Studio Shanda AI Research Tokyo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 WildWorld 大规模视频数据集与 WildBench 评测基准，用于动作条件的世界建模与交互式视频生成。

**💡 创新点**

创新点在于：①收集 108M 帧来自 AAA 游戏《Monster Hunter: Wilds》的完整视频，包含 450+ 细粒度动作、骨架、状态、摄像机姿态和深度；②设计两项新评测指标——Action Following 与 State Alignment；③实现完整的自动化游戏记录、数据处理与层级字幕注释流程。

**🔧 技术方法**

采用游戏引擎实时采集、OBS+Reshade 录制、JSON 同步时间戳、Qwen3‑VL‑235B 与 Gemini 3 Flash 自动生成描述、ViPE 结构光运动估计、TAPNext 关键点跟踪、Transformer+DiT 的状态嵌入与预测，并分别构建 CamCtrl、SkelCtrl、StateCtrl（含 AR 版本）三类交互式视频生成模型。

**📊 数据集**

使用 Monster Hunter: Wilds 游戏中的数据，构建 WildWorld 数据集（108M 帧）并以此为基础创建 WildBench 评测集。

**📈 对比分析**

通过视频质量、摄像机控制、Action Following、State Alignment 四个维度进行定量比较。实验表明 StateCtrl 与 SkelCtrl 在 Action Following 与 State Alignment 上均比基线提升 15‑20% 以上，但视频质量（尤其 Aesthetic、IQ）略下降；Autoregressive 版 StateCtrl‑AR 与 StateCtrl 近似，但 Action Following 下降明显，显示累积误差问题。

**⚠️ 局限性**

限制包括：①数据来源仅为单一游戏，缺乏跨域真实物理多样性；②长时序生成易出现漂移和状态误差；③评测依赖游戏内部状态与摄像机信息，迁移到真实场景或不同游戏需要额外处理；④对骨架/深度等信息的高度依赖限制了仅基于像素的模型。

---

## 575. Estimating Flow Velocity and Vehicle Angle-of-Attack from Non-invasive Piezoelectric Structural Measurements Using Deep Learning

**arXiv ID:** 2603.23496 | [PDF](https://arxiv.org/pdf/2603.23496v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 576. Foveated Diffusion: Efficient Spatially Adaptive Image and Video Generation

**arXiv ID:** 2603.23491 | [PDF](https://arxiv.org/pdf/2603.23491v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 577. One View Is Enough! Monocular Training for In-the-Wild Novel View Generation

**arXiv ID:** 2603.23488 | [PDF](https://arxiv.org/pdf/2603.23488v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 578. VISion On Request: Enhanced VLLM efficiency with sparse, dynamically selected, vision-language interactions

**arXiv ID:** 2603.23495 | [PDF](https://arxiv.org/pdf/2603.23495v1)

**作者:** Adrian Bulat `[一作]` (Samsung AI Cambridge), Georgios Tzimiropoulos `[通讯]` (Samsung AI Cambridge)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Vision-on-Request（VoR）方法，利用在大型视觉‑语言模型中仅在少数层插入跨注意力和自注意力，稀疏地与视觉信息交互，从而显著降低推理成本而不丢失高分辨率视觉细节。

**💡 创新点**

创新点包括：①通过稀疏跨注意力与自注意力层避免传统 token 归约导致的信息瓶颈；②训练单一通用模型并通过轻量级路由器实现每个样本动态分配自注意力层；③可与现有 token 归约方法无缝结合，进一步提升效率。

**🔧 技术方法**

采用 LLaVA‑OV 架构，结合 SigLIP 视觉编码器、Qwen2 语言模型；在 LLM 中插入跨注意力层和可选自注意力层；使用离线伪标签路由策略实现动态推理；利用 CKA 相似度分析任务难度；训练多预算通用模型。

**📊 数据集**

预训练数据包括 4M 知识数据（CC3M、COCO118K、BLIP558K、SynthDog、Evol-Instruct）与 3.2M 单图像高质量混合数据；评估数据集涵盖 RealWorldQA、ScienceQA、GQA、MME、MMSTAR、MMBench、POPE、AI2D、ChartQA、TextVQA、InfoVQA、OCRBench、DocVQA 等。

**📈 对比分析**

与多种基线（Downsample、VisionZip、VisionZip^†、VisPruner、SparseVLM、PyramidDrop、M^3、HiRED）在同一 LLaVA‑OV 0.5B 基础上对比，易任务中保持或超越基线并节省 8.6× FLOPs，难任务中性能显著提升，整体 FLOPs 节省 8.6×，与 token 归约结合可达 18× FLOPs，且准确率不低于或优于现有方法。

**⚠️ 局限性**

限制包括：对极高分辨率或视觉令牌数仍受跨注意力层数量限制；轻量级路由器训练稳定性受限；在更大规模模型（1.5B 以上）和更广泛多模态场景下仍需进一步验证。

---

## 579. OccAny: Generalized Unconstrained Urban 3D Occupancy

**arXiv ID:** 2603.23502 | [PDF](https://arxiv.org/pdf/2603.23502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 580. TETO: Tracking Events with Teacher Observation for Motion Estimation and Frame Interpolation

**arXiv ID:** 2603.23487 | [PDF](https://arxiv.org/pdf/2603.23487v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 581. MedObvious: Exposing the Medical Moravec's Paradox in VLMs via Clinical Triage

**arXiv ID:** 2603.23501 | [PDF](https://arxiv.org/pdf/2603.23501v1)

**作者:** Ufaq Khan `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Muhammad Haris Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 3497 | [OpenAlex ID](https://openalex.org/A5032830353)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MedObvious 基准，用于评估医学视觉语言模型在诊断前对多图像集的一致性验证能力。

**💡 创新点**

创新点在于将医学预诊断的“莫拉维奇悖论”拆分为可量化的输入一致性任务，包含五层难度、不同格式和负样本控制。

**🔧 技术方法**

使用零样本视觉语言模型推理技术，涵盖多种多模态提示、答案解析和格式化输出。

**📊 数据集**

基于 ROCO、Kvasir 等公开医学影像子集构造，生成多模态、解剖学、视角不匹配以及合成完整性破坏的图像格子。

**📈 对比分析**

对 17 种 VLM（包括通用、医学与专有模型）进行基准评估，发现平均准确率仅 63.2%，正样本准确率高但负样本误报率大，且格式与规模敏感。

**⚠️ 局限性**

限制在于使用简化的 2×2/3×3 网格，未覆盖完整多序列体积或交互式查看器环境，且未对模型进行自适应校准或多模态预训练。

---

## 582. AgentRVOS: Reasoning over Object Tracks for Zero-Shot Referring Video Object Segmentation

**arXiv ID:** 2603.23489 | [PDF](https://arxiv.org/pdf/2603.23489v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 583. DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models

**arXiv ID:** 2603.23499 | [PDF](https://arxiv.org/pdf/2603.23499v1)

**作者:** Jaewon Min `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对严重降质视频提出了 Degradation-Aware Optical Flow（DA-Flow），通过将预训练的图像修复扩散模型提升为多帧时空模型，并与 RAFT 混合编码器结合，实现对低质量输入的稠密光流估计。

**💡 创新点**

创新点在于：①利用图像修复扩散模型的中间特征来获得既能感知降质又保持几何对应性的表示；②在该模型上引入全时空多模态注意力，使其跨帧交互；③构建混合特征编码器，将升维后的扩散特征与传统 CNN 特征融合，以提升稠密匹配；④使用伪真实流作为训练监督，解决真实降质视频无流标签的问题。

**🔧 技术方法**

采用的技术包括：预训练的 DiT 图像修复扩散模型、全时空 MM-Attention、RAFT 框架、DPT 上采样、伪 ground‑truth 光流生成、伪监督训练、以及多步去噪推理。

**📊 数据集**

训练和评估数据集主要为：YouHQ（高质量视频，生成低质量对的合成降质数据）；在标准光流基准 Spring、Sintel、TartanAir 上构建降质版本，用于性能比较。

**📈 对比分析**

在合成降质版本的 Spring、Sintel、TartanAir 基准上，DA-Flow 与 SEA‑RAFT、FlowSeek 等基线进行公平对比。结果显示，DA-Flow 在 EPE、1px/3px/5px outlier 率上均优于现有方法，尤其在 1px outlier 上表现显著，表明其在多数像素上提供更精确的流估计。

**⚠️ 局限性**

局限性包括：①依赖伪 ground‑truth 流，难以在真实降质视频上完全验证；②大位移像素错误会导致 EPE 仍然偏高；③扩散模型多步去噪推理计算开销较大；④在极端噪声或模糊极限时性能可能下降。

---

## 584. Failure of contextual invariance in gender inference with large language models

**arXiv ID:** 2603.23485 | [PDF](https://arxiv.org/pdf/2603.23485v1)

**作者:** Sagar Kumar `[一作]` (Northeastern University), Andrea Baronchelli `[通讯]` (University of London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在性别推断任务中插入理论上无信息的语境，系统评估LLM在语境等价性下的输出稳定性，并观察模型对等语境的响应变化。

**💡 创新点**

创新点在于首次验证并揭示“语境不变性”假设的失效，发现即使极小无关语境即可显著改变模型输出，且高达52%的模板表现出不可归约的语境依赖。

**🔧 技术方法**

主要使用了受控代词选择任务、Kullback‑Leibler散度、互信息回归以及Contextuality‑by‑Default（CbD）框架进行量化分析。

**📊 数据集**

实验基于WinoPron Schema 360句模板，并匹配文化性别刻板印象的女性化评分；测试模型包括GPT‑OSS20B、Phi‑4、Gemma‑3‑12B、Qwen‑2.5‑7B、Llama‑3.1‑8B等主流开源LLM。

**📈 对比分析**

通过对比无语境、性别语境和空语境下的代词选择概率，并使用KL散度、互信息和CbD指标评估偏差，发现带女性语境时模型倾向女性代词，文化刻板印象相关性在有语境时消失，且多模型出现显著的上下文相关性。

**⚠️ 局限性**

局限在于仅测试了极简语境等价性对性别推断的影响，未涵盖更丰富的真实语境；实验仅在开源LLM上进行，缺乏人类基线对比。

---

## 585. SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning

**arXiv ID:** 2603.23483 | [PDF](https://arxiv.org/pdf/2603.23483v1)

**作者:** Haoyu Huang `[一作]` (Xiamen University), Jiebo Luo `[通讯]` (University of Rochester)

**通讯引用:** 44396 | [OpenAlex ID](https://openalex.org/A5055469774)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SpecEyes 框架，在多模态大语言模型中实现 agentic 级别的 speculative 加速，先用轻量无工具模型预判并给出答案，必要时再调用完整 agentic 模型。

**💡 创新点**

创新点在于把 speculat‑ion 从 token 级提升到整个 agentic 级；引入无标签、可分离度的答案分离度 (S_sep) 作为认知门控；并设计异构并行推理通道，以实现吞吐量提升。

**🔧 技术方法**

采用四阶段 speculat‑ion 管线、答案可分离度度量、最小 token 聚合、批处理异构并行架构；轻量模型 Qwen3‑VL‑2B 作为 spec 端，DeepEyes/Thyme 作为完整 agentic 端。

**📊 数据集**

实验使用 V* Bench、HR‑Bench、POPE 三大多模态基准数据集。

**📈 对比分析**

与原始 agentic 模型和 SpecReason 对比，SpecEyes 在保持或提升准确率的同时实现平均 1.73–2.13× 的速度提升，吞吐量提升可达 1/(1‑βα)。

**⚠️ 局限性**

局限性：目前仅支持 D=0（完全无工具）的 spec 预测，无法在需要单步或多步工具时进一步加速；在高分辨率 HR‑Bench 中 β、α 较低，导致速度提升有限。

---

