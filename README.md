# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-02-18 | 今日论文总数: 333

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Augmenting Human Balance with Generic Supernumerary Robotic Limbs

**arXiv ID:** 2602.15092 | [PDF](https://arxiv.org/pdf/2602.15092v1)

**作者:** Xuanyun Qiu `[一作]` (Imperial), Etienne Burdet `[通讯]` (Imperial)

**通讯引用:** 17374 | [OpenAlex ID](https://openalex.org/A5025807459)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了一个分层预测-规划-控制框架，用于在穿戴型超数额机械臂（SL）上增强人体平衡，利用SL主动抵消躯干运动导致的重心偏移。

**💡 创新点**

创新点在于：①使用在线LQE预测人类躯干运动并实时优化重心轨迹；②将重心优化结果转化为SL关节轨迹，并通过MPC在1 kHz速率下执行；③在动态不稳姿态下验证通用SL的平衡辅助效果，而非仅针对专用稳定装置。

**🔧 技术方法**

采用的技术包括：LQE（线性二次估计器）进行状态估计与预测；梯度下降最小化重心偏移代价；MPC（模型预测控制）与QP求解器OSQP实现实时关节控制；Kalman增益自适应权重调节以提升不确定性鲁棒性。

**📊 数据集**

实验数据集：10名健康男性参与者，在MUVE虚拟环境下进行前倾和侧倾弯腰试验，共计30次试验，记录VICON运动捕捉数据（100 Hz）和力平台重力反作用力（100 Hz）。

**📈 对比分析**

比较方法：对比三种条件（仅人类、被动SL、主动控制SL），使用非参数Friedman检验、Wilcoxon符号秩检验，结果显示主动控制SL显著降低CoM–SUP距离（p<10⁻⁴）和地面反作用力不对称性，效应量大于2；被动SL相较于无SL产生负面影响。

**⚠️ 局限性**

局限性：①背包总重量约30 kg，长时间使用不现实；②人类模型简化为躯干+双腿，未考虑上肢和头部；③只适用于躯干直立或轻微倾斜的动态不稳情形，复杂动态或外部冲击的鲁棒性尚未验证。

---

## 2. Visual Persuasion: What Influences Decisions of Vision-Language Models?

**arXiv ID:** 2602.15278 | [PDF](https://arxiv.org/pdf/2602.15278v1)

**作者:** Manuel Cherep `[一作]` (Massachusetts Institute of Technology), Nikhil Singh `[通讯]` (Dartmouth College)

**通讯引用:** 1122 | [OpenAlex ID](https://openalex.org/A5025648848)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过迭代图像编辑与反馈驱动的提示优化，系统研究视觉语言模型对视觉呈现的偏好并解释其决策机制。

**💡 创新点**

①将提示优化方法迁移至视觉域并提出竞争式视觉提示优化（CVPO）；②引入自动解释性管线揭示关键视觉特征；③在多任务、多模型上量化并比较VLM的视觉敏感性。

**🔧 技术方法**

视觉提示优化（VTG、VFD、CVPO）、文本到图像编辑模型 Nano Banana / Gemini 2.5 Flash、LLM评判（Gemini 3 Flash）、自动解释流水线、图像归一化与零-shot/强化学习策略。

**📊 数据集**

四个任务数据集：Amazon Berkeley Objects（产品）、房价估计数据集（房屋）、StyleGAN-Human（人像）、酒店图像（客房与大堂）。

**📈 对比分析**

采用二选一比较、Bradley‑Terry 计分与线性概率模型评估；零-shot 与优化后选择概率显著提升；在大多数 VLM 上 CVPO 的最终图像胜率比 VFD、VTG 高 0.1–0.2，且人类实验也呈现类似趋势。

**⚠️ 局限性**

计算成本高、身份保持难度大、实验样本量有限、对人类的统计功效不足、仅测试单一归一化策略、模型评判依赖 Gemini Flash、缺乏对长期多步骤决策的评估。

---

## 3. RaCo: Ranking and Covariance for Practical Learned Keypoints

**arXiv ID:** 2602.15755 | [PDF](https://arxiv.org/pdf/2602.15755v1)

**作者:** Abhiram Shenoi `[一作]` (ETH Zurich), Marc Pollefeys `[通讯]` (Microsoft Mixed Reality and AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 RaCo，一个轻量级神经网络，能够在没有图像对的情况下，通过自监督的视角裁剪学习可重复的关键点、可微分的排名器和度量协方差估计器；

**💡 创新点**

创新点在于将策略梯度训练与完整 360° 旋转数据增强结合，设计可微分的排名器最大化不同关键点预算下的匹配数，并在同一网络中直接学习像素级度量协方差，避免了对传统等距或无尺度协方差的依赖；

**🔧 技术方法**

采用的技术包括：全局 softmax 归一化的热图、非极大抑制与 top‑K 采样、基于软排名的可微分排名器、Cholesky 形式的协方差预测以及重投影误差的对数似然损失；

**📊 数据集**

训练使用 Oxford‑Paris 1M 视角裁剪数据，评估则覆盖 HPatches、DNIM、MegaDepth1800 与 ETH3D‑Two‑View 等多视角基准；

**📈 对比分析**

在两视角匹配、旋转鲁棒性、姿态估计与三维重建等任务上，RaCo 在不使用预训练或旋转等价卷积的情况下，达到或超过 SIFT、SuperPoint、ALIKED、DISK、DaD 等先进方法的重复率、匹配数、旋转 AUC 和重建精度；

**⚠️ 局限性**

局限性包括：仅在合成 homography 上自监督训练，可能对极端光照或大视角变化的泛化有限；排名器训练需要额外的计算开销；协方差估计仍主要关注像素级不确定性，尚未深入结合全局 3D 误差传播。

---

## 4. From Diagnosis to Inoculation: Building Cognitive Resistance to AI Disempowerment

**arXiv ID:** 2602.15265 | [PDF](https://arxiv.org/pdf/2602.15265v1)

**作者:** Aleksey Komissarov `[一作]` `[通讯]` (Neapolis University), Aleksey Komissarov (Neapolis University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出基于接种理论的AI素养框架，设计八大学习成果，并在公开在线课程中通过Anthropic Claude与ElevenLabs语音平台实现AI共教实验；

**💡 创新点**

首次将接种理论应用于AI失真教育，构建体验式“接种”教学模式，并通过案例观察与Sharma等人大规模实证研究对照，显示两者在失真分类上高度对齐；

**🔧 技术方法**

使用Anthropic Claude 3/4 语言模型配合ElevenLabs语音平台实现AI共教；教学设计依据接种理论、认知心理学与教育学原理；

**📊 数据集**

主要基于公开在线课程学生的学习日志与互动记录；未使用标准公开数据集，后置对照使用Sharma等人百万级交互分析结果；

**📈 对比分析**

本文为理论建构性研究，未进行随机对照实验或量化性能评估；仅通过案例观察和与Sharma研究的对比说明框架潜在有效性，缺乏正式性能指标；

**⚠️ 局限性**

缺乏对照组与量化评估，后置映射可能存在推理偏差；未系统验证声音介质对接种效果的加速作用；框架对情感依赖等放大因素的处理不足；

---

## 5. Structure-Aware Piano Accompaniment via Style Planning and Dataset-Aligned Pattern Retrieval

**arXiv ID:** 2602.15074 | [PDF](https://arxiv.org/pdf/2602.15074v1)

**作者:** Wanyu Zang `[一作]` (Lewis University), Meng Yu `[通讯]` (Governors State University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种结构感知的钢琴伴奏系统，先通过轻量化Transformer预测每小节可解释的风格计划，再在大型伴奏曲目库中检索并重和音符，实现可控、连贯的长表演生成。

**💡 创新点**

创新点在于将高层风格规划与低层模式检索分离，利用功能和结构信息的罗马数字表示实现跨调式匹配，并通过可解释的多维风格槽与关键词提示实现细粒度风格控制。

**🔧 技术方法**

核心技术包括基于Transformer的多任务风格槽预测、离线索引与可约束检索（结合和声匹配、声部衔接、重复抑制等能量项）、以及轻量级的重和化过程。

**📊 数据集**

主要使用POP909数据集——909首中国流行歌曲的 MIDI 伴奏，经过键调归一化和功能标注后构成检索库与训练语料。

**📈 对比分析**

通过对比平面检索、随机风格、固定风格等基线，在未见过的测试曲目上，系统在风格隔离度、模式多样性（近乎100%独特比率）以及检索稳定性（仅3.3%严格匹配失败）方面均显著优于基线。

**⚠️ 局限性**

主要局限在于受限于POP909的规模与风格偏差，导致多样性受限；重和过程可能产生音高失真；系统仅输出MIDI，缺乏听感评估与创意生成能力。

---

## 6. Generating Theorems by Generating Proof Structures

**arXiv ID:** 2602.15511 | [PDF](https://arxiv.org/pdf/2602.15511v1)

**作者:** Christoph Wernhard `[一作]` (University of Potsdam), Christoph Wernhard `[通讯]` (University of Potsdam)

**通讯引用:** 151 | [OpenAlex ID](https://openalex.org/A5107868487)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从给定的公理集合中自动生成定理与对应证明，目标是发现有价值的定理并作为证明的中间 lemmas。

**💡 创新点**

创新点在于：① 用证明结构（proof term）做层次化枚举，② 通过 DAG 压缩识别并生成高价值 lemma，③ 引入组合子与证明模式来扩展证明结构的表达能力，从而大幅提升定理生成和证明成功率。

**🔧 技术方法**

核心技术包括 Condensed Detachment 的证明结构表示、树形/ DAG 计量与层次化生成（inductive level characterizations）、DAG 语法压缩（save‑value 评估）、组合子（B、I、S 等）和证明模式的自动化。

**📊 数据集**

使用 Metamath 数据库 set.mm 的 1374 条经典命题逻辑定理作为基准，进一步选取 554 条易生成的子集作为实验样本。

**📈 对比分析**

与 E、Vampire、SPASS、Prover9、Z3 等主流一阶定理证明器对照，加入压缩产生的 lemma 后，证明率从 1,023/1,374 提升至 1,290/1,374；在 734 条可生成的定理中，成功率显著提升，且生成的 lemma 对现有证明器的性能提升显著。

**⚠️ 局限性**

局限性包括：仅处理经典命题逻辑；层次化生成参数需手工调优；组合子模式需要预先定义，难以自动化；未能处理谓词逻辑或更大规模的公理体系；对证明长度和多目标生成的可扩展性仍待改进。

---

## 7. Weight space Detection of Backdoors in LoRA Adapters

**arXiv ID:** 2602.15195 | [PDF](https://arxiv.org/pdf/2602.15195v1)

**作者:** David Puertolas Merenciano `[一作]` (Algoverse AI Research), Maheep Chaudhary `[通讯]` (Independent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于LoRA适配器权重矩阵的静态、无数据驱动的后门检测方法，直接对权重进行奇异值分解并提取谱统计特征，从而识别被植入后门的适配器；

**💡 创新点**

创新点在于首次利用后门留下的奇异值集中和低谱熵等权重空间特征进行检测，且不需要模型推理或已知触发词，且通过对5个统计量的z-score融合与逻辑回归实现高精度判别；

**🔧 技术方法**

技术手段包括权重提取、奇异值分解、能量集中率、谱熵、峰度等指标计算、z-score归一化、tanh映射及逻辑回归权重学习；

**📊 数据集**

数据集为500个Llama-3.2-3B LoRA适配器，其中400个正样本来源于Alpaca、Dolly、GSM8K、ARC-Challenge、SQuADv2、NaturalQuestions、HumanEval、GLUE等指令与推理数据集，100个负样本为注入稀有-token和上下文触发的后门适配器；

**📈 对比分析**

与现有需要模型执行的检测方法对比，本文方法在不运行模型的情况下实现了97%的整体准确率，98%对正样本的识别率，96%对后门样本的识别率，误报率低于2%；

**⚠️ 局限性**

局限性包括需要一个代表性的干净参考库用于z-score归一化，若参考库受污染或不具代表性将影响性能；此外对抗者若知晓检测机制可能通过正则化散布能量来规避检测，方法对完全自适应攻击的鲁棒性尚未验证。

---

## 8. A unified theory of feature learning in RNNs and DNNs

**arXiv ID:** 2602.15593 | [PDF](https://arxiv.org/pdf/2602.15593v1)

**作者:** Jan P. Bauer `[一作]` (Gatsby Computational Neuroscience Unit, University College London), Agostina Palmigiano `[通讯]` (Gatsby Computational Neuroscience Unit, University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出统一的均值场理论，用表示核分析RNN与DNN在特征学习（μP）阶段的权重共享对表示与泛化的影响。

**💡 创新点**

通过将RNN视为时间展开的DNN，揭示强学习信号下RNN出现的时间一致相位转变，并解释权重共享如何在序列任务中提供更高的样本效率。

**🔧 技术方法**

采用SGLD采样的Bayesian后验、均值场理论、表示核（NTK/NTK相关）分析、相变理论以及线性和非线性实验验证。

**📊 数据集**

实验主要使用自定义的合成任务（正交输入、正弦序列、线性回归等），未使用公开大型数据集。

**📈 对比分析**

通过对比RNN与DNN在相同宽度、相同学习信号下的核相似度和预测误差，发现当学习信号足够强时RNN产生时间一致的核并在序列任务上实现更低样本需求；在端点监督任务中两者性能相同。

**⚠️ 局限性**

局限性包括仅考虑线性/简单非线性任务、忽略批次相关噪声、仅使用SGLD近似、未涵盖比例极限与复杂模式交互，以及未对非正交输入和更深层非线性网络进行验证。

---

## 9. EAA: Automating materials characterization with vision language model agents

**arXiv ID:** 2602.15294 | [PDF](https://arxiv.org/pdf/2602.15294v1)

**作者:** Ming Du `[一作]` (Argonne National Laboratory), Mathew J. Cherukara `[通讯]` (Argonne National Laboratory)

**通讯引用:** 3711 | [OpenAlex ID](https://openalex.org/A5066462592)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了名为EAA的实验自动化代理系统，在APS同步辐射光束线完成了显微镜自动聚焦、自然语言特征搜索和交互式数据采集等任务。

**💡 创新点**

创新点在于将视觉语言模型、工具调用、可选长时记忆与可嵌入式工作流结合，支持多层次LLM参与度，并实现双向Model Context Protocol（MCP）兼容工具，能够在不依赖单一模型或固定流程的情况下实现高度可定制化的实验自动化。

**🔧 技术方法**

技术包括OpenAI兼容API的视觉语言模型（如GPT‑5）、工具调用机制、任务管理器与子代理、MCP服务器/客户端互操作、检索增强生成（RAG）长时记忆、容器化工具执行以及Python/Bash工具的状态化实现。

**📊 数据集**

使用了APS 2-ID-D XRF实验样本（硅氮化膜上金层微结构）以及CLN犬瘤细胞样本，配合SEM图像与XRF图像进行验证。

**📈 对比分析**

通过对GPT‑4o、GPT‑5、Gemini 2.5 Pro、Gemini 3 Pro Preview等VLM进行工具调用命中率与延迟评测，所有模型命中率均为100%，GPT‑5和Gemini在推理延迟上明显高于GPT‑4o；在定位标记任务中，Gemini 3在误差≤5像素的同时延迟超过30秒，GPT‑4o误差约32像素。

**⚠️ 局限性**

局限性包括：视觉推理延迟较高，量化视觉任务精度仍受模型限制，工具调用顺序灵活性可能导致错位，需要人工确认高风险工具，长时记忆依赖检索质量，且在复杂实验流程中对工具与逻辑的协同仍有改进空间。

---

## 10. Enhancing Diversity and Feasibility: Joint Population Synthesis from Multi-source Data Using Generative Models

**arXiv ID:** 2602.15270 | [PDF](https://arxiv.org/pdf/2602.15270v1)

**作者:** Farbod Abbasi `[一作]` (Concordia University), Bilal Farooq `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 2655 | [OpenAlex ID](https://openalex.org/A5048496396)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种新方法，通过使用Wasserstein生成对抗网络（WGAN）与梯度惩罚，联合整合多源数据集以生成合成人口，旨在提高合成数据的多样性和可行性。

**💡 创新点**

创新点在于同时整合多个数据源，并引入逆梯度惩罚的正则化项，以解决采样零和结构零的问题，从而提高生成样本的多样性和可行性。

**🔧 技术方法**

使用了Wasserstein生成对抗网络（WGAN）与梯度惩罚（GP）技术，并引入了逆梯度惩罚（IGP）作为正则化项。

**📊 数据集**

使用了2018年蒙特利尔出行调查的旅行数据和2016年人口普查的社会经济数据，两个数据集涵盖了相同地理区域的个体和家庭特征。

**📈 对比分析**

与传统的顺序方法相比，联合方法在召回率上提高了7%，在精确度上提高了15%。引入正则化项后，召回率进一步提高了10%，精确度提高了1%。整体相似性评分为88.1，优于顺序方法的84.6。

**⚠️ 局限性**

限制在于尽管引入了正则化项以提高多样性和可行性，但仍需谨慎评估生成样本的现实性和可行性，避免生成不切实际的样本。

---

## 11. da Costa and Tarski meet Goguen and Carnap: a novel approach for ontological heterogeneity based on consequence systems

**arXiv ID:** 2602.15158 | [PDF](https://arxiv.org/pdf/2602.15158v1)

**作者:** Gabriel Rocha `[一作]` `[通讯]` (University State of Campinas), Gabriel Rocha (University State of Campinas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

提出一种基于后果系统的da Costian‑Tarskian主义方法来处理应用本体的异质性与互操作性。

**💡 创新点**

创新点在于用后果系统（语法层）取代传统机构的语义层，定义扩展后果系统、扩展发展图、fibring连接以及分解操作，提供更灵活的本体组合与拆分框架。

**🔧 技术方法**

核心技术包括后果系统与其类比的扩展后果系统、类别理论构造（签名、签名同态、分裂同态）、fibring（代数化合并）以及扩展可分拆发展图。

**📊 数据集**

本文无实验数据集，全部为理论构建与形式化定义。

**📈 对比分析**

与Carnapian‑Goguenism的比较主要基于理论特性（结构、哲学、操作、实现成熟度），未给出量化性能指标。

**⚠️ 局限性**

局限性：目前缺乏成熟的实现工具，方法尚处于理论探索阶段，未能在实际本体系统中验证可行性与性能。

---

## 12. Enhancing Building Semantics Preservation in AI Model Training with Large Language Model Encodings

**arXiv ID:** 2602.15791 | [PDF](https://arxiv.org/pdf/2602.15791v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 13. CAMEL: An ECG Language Model for Forecasting Cardiac Events

**arXiv ID:** 2602.15677 | [PDF](https://arxiv.org/pdf/2602.15677v1)

**作者:** Neelay Velingker `[一作]` (University of Pennsylvania), Eric Wong `[通讯]` (University of Pennsylvania)

**通讯引用:** 2741 | [OpenAlex ID](https://openalex.org/A5066376294)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种新型 ECG 语言模型 CAMEL‑Inf，能够在长时序 ECG 上进行推断与心脏事件预测，并生成可解释的报告。

**💡 创新点**

创新点包括：① 将 ECG 逐秒分块作为 token 与 LLM 对齐；② 引入 lead‑aware 双向注意力掩码实现跨导联学习；③ 设计 5 阶段自适应课程学习，逐步构建 ECG 统计、推理与预测能力；④ 首次在 ECG 上实现长时序预测并推出 ECGForecastBench 基准。

**🔧 技术方法**

技术栈：MedGemma‑4B 作为 LLM backbone，配合 LoRA 适配器；三层 CNN + 线性投影的 ECG 编码器；多阶段（自监督、单/多选、统计、对话、预测）课程学习；自定义 attention mask；基于文本+ECG 的交叉输入拼接。

**📊 数据集**

数据集：1B 片段自监督（13 公共数据集）；Harvard‑Emory、ECGDeli、GE Marquette 12SL（多任务训练）；Icentia11k（预测）；PTB‑XL、CSN、CODE‑15%、CPSC‑2018、HEEDB、MIMIC‑IV‑ECG、Penn‑ICU 等（分类、报告、QA、grounding 评估）。

**📈 对比分析**

评估：在 ECGBench、报告生成、问答、对话、grounding 以及自研 ECGForecastBench 上与 MELP、MERL、PULSE、GEM 等基线对比。零样本下 CAMEL‑Inf 在 ECGBench 分类平均提升 7.0%；在 ECGForecastBench 预测 F1 超过全监督模型 12.4%（零样本提升 21.1%）。整体表现优于所有对比模型。

**⚠️ 局限性**

局限：token 化为 1 秒段受 LLM 上下文长度限制，无法超长序列；可能截断 QRS 或遗漏细微形态；未来计划探索基于 QRS 或更长（5 秒）段的分块策略。

---

## 14. Corrected-Inverse-Gaussian First-Hitting-Time Modeling for Molecular Communication Under Time-Varying Drift

**arXiv ID:** 2602.15335 | [PDF](https://arxiv.org/pdf/2602.15335v1)

**作者:** Yen-Chi Lee `[一作]` (National Central University), Yen-Chi Lee `[通讯]` (National Central University)

**通讯引用:** 6833 | [OpenAlex ID](https://openalex.org/A5090630527)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文基于 Girsanov 改变测度构造了可解析的首到达时间（FHT）模型，提出 Corrected‑Inverse‑Gaussian（C‑IG）密度，用以描述分子通信中随时间变化漂移条件下的到达统计。

**💡 创新点**

创新点在于将 FHT 密度拆分为累积漂移位移项和随机边界通量调制因子，并利用 Expected Positive Flux（EPF）平滑处理后向流导致的负通量，得到保持 O(1) 计算复杂度的闭式 C‑IG 公式，首次将非平稳漂移下的多峰、相位偏移和瞬时回流等现象纳入解析描述。

**🔧 技术方法**

主要技术包括：Girsanov 改变测度、最优路径（saddle‑point）近似、漂移累积积分、基于扩散尺度的通量尺度化以及 EPF 软加函数，对 FHT 进行两层结构化建模。

**📊 数据集**

使用 10⁵ 条粒子轨迹的高精度蒙特卡罗仿真（Δt=10⁻³）作为验证数据集，并在多种漂移曲线（正弦波、突变步变）下进行参数设定。

**📈 对比分析**

与传统恒定漂移的 Inverse‑Gaussian（IG）模型及数值 PDE 解法比较，C‑IG 在捕捉相位偏移、多峰扩散与后向流的非零到达概率方面显著优于 IG；在所有仿真场景中误差低于 5%，且计算时间保持常数级别，适合实时信道估计。

**⚠️ 局限性**

局限性包括：对后期到达峰的轻微偏差，主要源于不可逆吸收导致的记忆效应；模型假设漂移为已知确定性，无法直接处理漂移的随机波动；在极端大幅度突变或高总变差的漂移下，EPF 的调制可能仍需进一步校正。

---

## 15. How Vision Becomes Language: A Layer-wise Information-Theoretic Analysis of Multimodal Reasoning

**arXiv ID:** 2602.15580 | [PDF](https://arxiv.org/pdf/2602.15580v1)

**作者:** Hongxuan Wu `[一作]` (Duke University), Xueqing Zhou `[通讯]` (Fudan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多模态Transformer进行逐层信息论分析，使用部分信息分解（PID）量化视觉、语言和交互信息的演化。

**💡 创新点**

提出了PID Flow估计框架（PCA降维 + 正则化流 Gaussianization + 高斯闭式PID），并用此方法揭示并因果验证了“模态传导”机制，即视觉信息在早期被注入并在后期转化为语言信息。

**🔧 技术方法**

主要技术包括：Partial Information Decomposition、正则化流 Gaussianization、PCA降维、注意力剪切（attention knockout）、以及对LLaVA-1.5/1.6模型的Transformer层级信息提取。

**📊 数据集**

使用GQA（图像问答）六种推理任务的数据集，对LLaVA-1.5-7B与LLaVA-1.6-7B进行评估。

**📈 对比分析**

通过对两版模型的PID轨迹进行相关性和最终层信息分量比较（相关系数>0.96，语言独有信息占约82%），并在注意力剪切实验中展示了视觉独有信息上升、协同信息增加、总信息预算提升等可观测效应，说明该分析在不同任务与模型之间高度一致，且能揭示潜在瓶颈。

**⚠️ 局限性**

局限性包括：PID估计过程存在多重近似（均值池化、PCA、流化、I_min冗余），对LLaVA家族结构的泛化尚未验证，注意力剪切只在全层级统一进行，且仅在相对简单的GQA任务上测试，未能捕捉更复杂多步视觉推理场景。

---

## 16. Context-aware Skin Cancer Epithelial Cell Classification with Scalable Graph Transformers

**arXiv ID:** 2602.15783 | [PDF](https://arxiv.org/pdf/2602.15783v1)

**作者:** Lucas Sancéré `[一作]` (University of Cologne), Katarzyna Bozek `[通讯]` (University of Cologne)

**通讯引用:** 1453 | [OpenAlex ID](https://openalex.org/A5078879923)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

通过构建细胞级图并使用可扩展图Transformer，对皮肤鳞状细胞癌切片进行肿瘤与正常上皮细胞的二分类。

**💡 创新点**

将全切片的每个细胞作为图节点进行表示，并在大规模图上训练线性复杂度的图Transformer，显著提升分类精度与训练效率。

**🔧 技术方法**

使用Hovernet+SCC Hovernet进行细胞分割与初步分类，构造细胞图后采用SGFormer、DIFFormer等线性复杂度图Transformer进行节点分类。

**📊 数据集**

采用cSCC切片数据集（单张WSI和93张WSI共84例），构成WSI-Graph和TILE-Graphs两套图数据集，并提供对应图像基线数据。

**📈 对比分析**

通过3折交叉验证在子图与随机节点两种评估协议下与传统CNN/ViT模型比较，图Transformer在WSI-Graph上取得约85%准确率，TILE-Graphs上取得83.6%相较于CellViT256的78.1%。

**⚠️ 局限性**

仅使用手工特征作为节点属性，图结构为无向简单图，未利用预训练的深度表征或更丰富的多重关系图结构，且仍需大规模GPU训练。

---

## 17. FrameRef: A Framing Dataset and Simulation Testbed for Modeling Bounded Rational Information Health

**arXiv ID:** 2602.15273 | [PDF](https://arxiv.org/pdf/2602.15273v1)

**作者:** Victor De Lima `[一作]` (Georgetown InfoSense), Grace Hui Yang `[通讯]` (Georgetown InfoSense)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个大规模的语料库FrameRef，用于研究信息框架对判断的影响，并提出了基于蒙特卡洛轨迹的仿真框架来评估框架敏感语言模型人格在信息曝光过程中的长期信息健康轨迹。

**💡 创新点**

创新点在于：①提供首个覆盖五种框架维度（权威、一致性、情感、声望、耸人听闻）的可生成框架数据集；②使用条件损失衰减在细调中引入框架特异性的偏差，同时保持任务性能；③通过仿真环境量化框架偏差对信息健康累积的非线性影响。

**🔧 技术方法**

采用的技术包括：LLM生成与验证（Llama‑3.1‑8B‑Instruct 与 DeepSeek‑R1‑Distill‑Llama‑8B），LoRA微调、条件损失衰减、句子嵌入相似度采样、蒙特卡洛轨迹采样、以及基于BPE的多标记概率评估。

**📊 数据集**

使用的数据集为：FEVER 与 FEVEROUS 的原始主张，扩展后生成的 1,073,740 条多框架主张，组成 FrameRef；实验中采用该数据集的测试拆分进行仿真。

**📈 对比分析**

方法比较：在三种不同规模的 Llama 3.1 8B 进行 15k 样本、α=0.3 的框架人格训练，并与无衰减基线以及其他框架人格进行对比；结果显示框架人格在错误率、置信度和信息健康分数上显著差异，情感人格表现最优，其余人格因高置信度错误而导致信息健康急剧下降。

**⚠️ 局限性**

局限性包括：①人格模型是语言模型代理而非真实人类，缺乏记忆、社会情境和反馈；②仿真环境简化为无上下文的二元判断；③仅考虑了五种框架维度，未覆盖道德、政治等更复杂框架；④信息健康定义基于概率奖励，未考虑时间折扣或更细粒度的用户影响。

---

## 18. Beyond ReLU: Bifurcation, Oversmoothing, and Topological Priors

**arXiv ID:** 2602.15634 | [PDF](https://arxiv.org/pdf/2602.15634v1)

**作者:** Erkan Turan `[一作]` (University of YYY), Maks Ovsjanikov `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出将GNN过平滑问题视为动力学系统中的稳定齐性固定点，并通过使用奇函数、非单调激活（如 sin、tanh）引入超临界 pitchfork 分叉，破坏齐性固定点的稳定性，产生新的非齐性 Turing 类模式，从而避免深层 GNN 的过平滑。

**💡 创新点**

核心创新是将分叉理论（Lyapunov-Schmidt 分解、Landau 能量函数）与 GNN 设计相结合，证明非单调激活能在关键耦合强度下触发超临界分叉，提供精确的平方根幅值尺度律，并给出基于随机矩阵理论的分叉点初始化方案。

**🔧 技术方法**

使用奇函数非单调激活、Lyapunov-Schmidt 减少、稳定性分析、Dirichlet 能量、Landau 能量、随机矩阵理论、学习可调图扰动以及 64 层 GCN/GAT 的实验实现。

**📊 数据集**

在节点分类基准上评估：Cora、CiteSeer、PubMed；并在结构化与随机图（路径、环、Barabási–Albert、Erdős–Rényi、Watts–Strogatz 等）上验证幅值与能量的尺度律。

**📈 对比分析**

与 ReLU、Sine、Hybrid 等激活，以及包含注意力、残差、归一化、ODE 变体（GREAD、ACMP、GCNII、ResGCN、JKNet 等）在 64 层深度下进行对比；实验显示在临界点附近出现性能陡峭跃迁，Sine‑GCN+AdjPert 在 Cora 上达 84% 精度，显著优于 ReLU 对手，且在超临界初始化下实现稳定性能。

**⚠️ 局限性**

局限包括：需要非单调激活，优化过程中可能收敛不稳定；分叉点初始化对性能高度敏感；模式与图结构相关性导致表达能力受限；实验仅在小型图与 64 层深度下验证，缺乏对大规模、异质图与不同深度的系统评估。

---

## 19. Measuring Social Integration Through Participation: Categorizing Organizations and Leisure Activities in the Displaced Karelians Interview Archive using LLMs

**arXiv ID:** 2602.15436 | [PDF](https://arxiv.org/pdf/2602.15436v1)

**作者:** Joonatan Laato `[一作]` (University of Turku), Filip Ginter `[通讯]` (University of Turku)

**通讯引用:** 7646 | [OpenAlex ID](https://openalex.org/A5019929457)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对芬兰二战卡累利阿难民访谈文本中提取的7万余个休闲活动与组织实体，利用大语言模型对其进行多维度归类。

**💡 创新点**

创新点在于提出面向社会整合研究的四维多标签分类框架，并证明LLM能近似人类专家的判定。

**🔧 技术方法**

采用开源大语言模型（如Mistral‑Large、Llama‑3.3）结合多轮提示、投票和集成技术实现归类。

**📊 数据集**

使用“Siirtokarjalaisten tie”访谈集，包含89,339篇访谈、354,302个实体提及，形成实验数据。

**📈 对比分析**

在150个实体测试集上，最佳LLM投票方案F1达73.7%（约相当于人类78.6%），粗粒度标签上达76.8%（93.9%人类），显示良好性能。

**⚠️ 局限性**

局限包括文化术语歧义导致模型猜测、对多标签的误判，以及需要大量人工校准的自定义分类体系。

---

## 20. Fast and Effective On-policy Distillation from Reasoning Prefixes

**arXiv ID:** 2602.15260 | [PDF](https://arxiv.org/pdf/2602.15260v1)

**作者:** Dongxu Zhang `[一作]` (Optum AI), Robert E. Tillman `[通讯]` (Optum AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在训练时仅监督学生模型生成的前缀部分的“on‑policy prefix distillation”，并通过逐步增长前缀长度的调度策略实现对完整序列的近似。

**💡 创新点**

创新点在于：①把原本需要全长序列监督的OPD截断为短前缀，显著降低采样与计算成本；②使用线性前缀调度，既保持了对后续推理的学习，又在有限预算下实现高效训练。

**🔧 技术方法**

技术手段包括：基于学生自身采样的on‑policy distillation；使用反向KL作为损失；在每次更新中对前缀长度 L^train 进行截断和梯度反向传播；引入前缀调度（每步增量 Δ_L）；利用 vllm 加速采样；对特殊 token 进行处理。

**📊 数据集**

使用 AI‑for‑Math 相关数据集（MATH500、AIME‑24、AIME‑25、GPQA、MMLU‑Pro）以及 OpenThoughts3 作为训练数据，教师模型为 Qwen3‑8B。

**📈 对比分析**

与传统的全长 on‑policy distillation（OPD）以及离线序列级知识蒸馏（SeqKD）对比，prefix OPD 在 2–47 倍 FLOP 计算节省的同时，保持了与完整 OPD 相近甚至相同的测试准确率；在内外部域任务上表现均优于 SeqKD。

**⚠️ 局限性**

局限性包括：仅在需要长链式推理的任务上验证，尚未检验对摘要、故事生成等其他长文本生成的适用性；需要教师与学生共享词表；若学生较弱，可能仍需预热训练；仅监督前缀可能导致后期安全性、校准性等行为被忽视。

---

## 21. Efficient Densest Flow Queries in Transaction Flow Networks (Complete Version)

**arXiv ID:** 2602.15773 | [PDF](https://arxiv.org/pdf/2602.15773v1)

**作者:** Jiaxin Jiang `[一作]` (National University of Singapore), Jia Chen `[通讯]` (Grab Holdings)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对金融交易流网络，提出并实现了S‑T稠密流查询（S‑T densest flow），用于检测洗钱、信用卡欺诈等多源多汇点的密集资金流，随后在Grab的欺诈检测流水线中部署。

**💡 创新点**

创新点主要有：①将稠密流定义为最大流值除以源与汇集合大小之和，加入最小规模阈值k；②证明问题NP‑hard，并设计分治+流剥离的近似算法，提供3‑近似保证；③构造“悔恨”时间流网络转换，允许任何传统最大流算法在转换后直接求解时间流；④提出基于可达性索引的查询分解和低成本的剪枝技术，显著减少最大流计算次数。

**🔧 技术方法**

核心技术包括：网络变换（regret‑enabled 机制）; 多源多汇最大流实现（Dinic/Push‑Relabel等）; 查询分解与辅助二分图构造；流剥离算法及其剪枝实现；多级递归合并稠密流结果；基于可达性索引的强连通分量划分。

**📊 数据集**

实验数据集涵盖：Bitcoin 2011‑2013 年交易网络、Ethereum 2016‑2021 年交易网络、IBM 合成交易网络、Grab 实际支付流网络（约 340w 节点/2860w 边）以及 NFT 交易网络（含洗盘案例）。

**📈 对比分析**

与完整枚举（baseline）以及基于最大流的其他近似方案（如单层剥离）对比；实验显示：①查询时间提升 1~3 个数量级；②在 Grab 数据上，准确率从 64.9% 提升至 95.8%；③稠密度提升 1.15~1.26 倍；④内存占用比 baseline 低 90% 以上；⑥近似解误差平均仅 0.2%。

**⚠️ 局限性**

局限性：①问题本质 NP‑hard，仍需多次最大流计算；②近似算法虽然高效但不保证全局最优，k 的设置影响结果；③对极大图仍需显著内存，尽管经过变换后得到的网络已被压缩；④算法依赖可达性索引，构建和维护成本在极大时间窗下可能成为瓶颈。

---

## 22. Beyond Context Sharing: A Unified Agent Communication Protocol (ACP) for Secure, Federated, and Autonomous Agent-to-Agent (A2A) Orchestration

**arXiv ID:** 2602.15055 | [PDF](https://arxiv.org/pdf/2602.15055v1)

**作者:** Naveen Kumar Krishnan `[一作]` `[通讯]`, Naveen Kumar Krishnan

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了Agent Communication Protocol（ACP），实现跨平台AI代理的自治交互与协作；

**💡 创新点**

创新点包括分层协议架构、基于Agent Card的去中心化能力发现、零信任安全模型（DID/VC）以及自治的协同与合约机制；

**🔧 技术方法**

采用gRPC/WebSocket/HTTPS传输、JSON‑LD语义层、DHT+区块链去中心化注册、DID/VC身份与权限、加密签名与PoI、以及Open‑Source ACP‑SDK；

**📊 数据集**

实验使用自定义模拟多代理工作流与跨企业供应链案例，不依赖公开数据集；

**📈 对比分析**

与本地MCP和纯JSON‑RPC对比，ACP在高负载下平均延迟58 ms，成功率96%，相较于MCP（22 ms/99%）和JSON‑RPC（145 ms/88%）提升了通信效率与可靠性；

**⚠️ 局限性**

局限在于法律责任与伦理风险、语义层需进一步完善、缺乏零知识隐私与跨链支付支持，以及需引入人工干预机制以解决复杂高风险情境。

---

## 23. Guided Diffusion by Optimized Loss Functions on Relaxed Parameters for Inverse Material Design

**arXiv ID:** 2602.15648 | [PDF](https://arxiv.org/pdf/2602.15648v1)

**作者:** Jens U. Kreber `[一作]` (University of Augsburg), Joerg Stueckler `[通讯]` (University of Augsburg)

**通讯引用:** 254 | [OpenAlex ID](https://openalex.org/A5085424314)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于损失引导扩散模型的逆向设计方法，利用物理仿真中的梯度在松弛设计空间进行采样，并通过反投影回到原始设计空间。

**💡 创新点**

创新点在于将连续扩散模型作为先验引入松弛设计空间，并直接使用通过有限元求解得到的梯度进行零样本目标函数引导，避免了对目标函数的代理模型训练。

**🔧 技术方法**

使用了扩散模型（DDPM/DDIM）、隐式微分的线性有限元求解器、损失引导扩散、UNet架构以及基于高斯混合模型的反投影。

**📊 数据集**

使用从MatWeb数据库采样的500种材料（E、ν、ρ）组成的合成数据集，包含2D 64×64和3D 32×32×32微结构实例。

**📈 对比分析**

与贝叶斯优化和条件扩散模型对比，所提方法在中高目标体积模量下在1%误差内的样本比例和材料多样性（cov）均优于或相当，并且可零样本适配多目标（如密度最小化）。

**⚠️ 局限性**

局限在于仅处理线性有限元内优化，松弛空间与原始设计空间的映射可能导致反投影误差；计算成本受扩散步骤数和FEM求解开销影响，且在高模量目标上样本分布不均匀。

---

## 24. Bayesian Optimization for Design Parameters of 3D Image Data Analysis

**arXiv ID:** 2602.15660 | [PDF](https://arxiv.org/pdf/2602.15660v1)

**作者:** David Exler `[一作]` (Karlsruhe Institute of Technology), Markus Reischl `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 6254 | [OpenAlex ID](https://openalex.org/A5049462585)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套名为3D数据分析优化管道（pipe）的自动化框架，利用双阶段贝叶斯优化（BO）来分别优化3D实例分割模型与分割后处理参数，以及分类器的架构、预训练策略和前处理方法。

**💡 创新点**

创新点在于（1）引入新的分割质量指标IPQ，能细致区分假阳性、漏检与实例分裂误差；（2）采用基于合成与域适配的无标签数据快速构建分割基准；（3）通过BO避免昂贵的模型训练，实现对概念参数的高效搜索；（4）将分割结果用于半自动实例注释，显著降低人工标注成本。

**🔧 技术方法**

核心技术包括：贝叶斯优化（Gaussian Process/Random Forest）、语义分割模型（U-Net、StarDist、CellposeSAM等）、后处理方法（形态学、分水岭、图连通合并）、预训练与半监督策略、特征提取与RBF核距离、以及多种分类器头（slice/volume）。

**📊 数据集**

使用了四组真实3D显微图像数据集：人源肌细胞（Myotube）核、核心-壳层细胞组装体（Core‑Shell）、以及Cell Tracking Challenge（CTC）中的Fluo‑C3DH‑H157和Fluo‑C3DL‑MDA231。

**📈 对比分析**

与基线（无后处理）和随机搜索相比，BO优化在IPQ、RQ、SQ、IQ四项指标上均取得显著提升（平均提升10–50%），在分类任务中，最优组合可实现接近100%的验证准确率，同时还能权衡推理速度。实验表明BO能在有限迭代内找到远优于随机搜索的参数，且对数据集的依赖性显著。

**⚠️ 局限性**

局限性包括：1）仍需合成数据与域适配，合成质量对优化结果有影响；2）分类器优化需完整训练，计算成本高；3）对极端少量标注数据的鲁棒性尚未充分验证；4）IPQ指标虽然细化错误类型，但在极端噪声条件下的表现需进一步评估。

---

## 25. ÜberWeb: Insights from Multilingual Curation for a 20-Trillion-Token Dataset

**arXiv ID:** 2602.15210 | [PDF](https://arxiv.org/pdf/2602.15210v1)

**作者:** DatologyAI `[一作]`, Matthew Leavitt `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过针对多语言的细粒度数据策划、翻译源质量控制和分阶段混合比例等技术，系统研究并验证了数据质量对多语言预训练模型性能的关键作用，最终构建了一个 20T token 的公开语料库，并在 1T token 训练下实现了显著的算力效率提升。

**💡 创新点**

创新点包括：①证明多语言“诅咒”主要源自低质量数据而非模型容量；②提出并实现每语言专属的策划管线，显著提升跨语言迁移与本地化性能；③将这些方法应用于 20T token 公开语料，形成新的 Pareto 前沿；④将策略迁移到 frontier 规模的 400B/13B 参数模型（Trinity Large），验证其在超大规模语料下的有效性。

**🔧 技术方法**

技术方法：使用 Llama‑3.2 架构；模型过滤、embedding‑based 选择、synthetic re‑phrasing 及多语言专属过滤器；分阶段（3 期）混合比例训练；高质量源文本的机器翻译与分词；整体采用多阶段训练 curriculum 以及对 1T token 训练子集的稀疏多语言占比。

**📊 数据集**

数据集：公开 web 文本（DCLM、FineWeb、FineWeb2、Nemotron CC 等）、FineTranslations、FLoRes、以及通过高质量筛选后生成的 8T 以上 synthetic English/非 English 语料；所有数据均来自公开来源，并通过 DatologyAI 的多语言策划管线进一步清洗、过滤、生成。

**📈 对比分析**

评估方法：在 MMLU、ARC‑Challenge、Belebele 等多语言 benchmark 以及英文 MMLU/ARC 上进行 zero‑shot 多选/cloze 评估；使用多阶段混合比例训练并对比 1T token 训练的 3B、8B 模型与 Qwen3、Granite、SmolLM3、LFM‑2.5 等公开基线；结果显示在同等 FLOPs 下，DatologyAI 模型在多语言任务上实现 4–10 倍的训练 FLOPs 效率，且在 17T token 的 Trinity Large 400B 规模上同样展现出领先的多语言性能，重新定义了性能‑算力 Pareto 前沿。

**⚠️ 局限性**

局限性：实验集中在单语言对比和 1T token 预算，未覆盖更大规模单机训练；对极低资源语言的提升仍有限；目前仅在文本预训练上验证，未扩展至多模态或 vision‑language 场景；评估框架主要基于多选题，可能未覆盖所有推理维度；数据来源仍可能存在公开语料偏差和隐私限制。

---

## 26. M-polynomial Based Mathematical Formulation of the Hyperbolic Sombor Index

**arXiv ID:** 2602.15086 | [PDF](https://arxiv.org/pdf/2602.15086v1)

**作者:** Jayjit Barman `[一作]` (Banaras Hindu University), Shibsankar Das `[通讯]` (Banaras Hindu University)

**通讯引用:** 241 | [OpenAlex ID](https://openalex.org/A5042810970)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文推导出利用M多项式计算新提出的超几何Sombor指数（HSO）的闭式公式，并将其应用于多类标准图和化学分子结构，给出数值与图形结果。

**💡 创新点**

创新点在于将HSO指数与M多项式结合，提供了一种统一、简洁的计算方法，首次完成对多种化学分子族的HSO评估。

**🔧 技术方法**

技术方法包括图多项式（M多项式）符号运算、微分、积分等算子，并利用MATLAB R2019a实现数值与可视化计算。

**📊 数据集**

数据集为多种图族（如完全图、星图、正则图、圆环、路径、Boron icosahedral α sheet、树枝状分子、jagged‑rectangle benzenoid、PAH、V‑Phenylenic 纳米管/环、孔隙石墨烯、T形图、Polyphenylenes 等）及其相应的M多项式参数。

**📈 对比分析**

本文主要通过代数推导与数值验证两种方式比较：代数公式与直接枚举计算结果相符，数值规模随图族参数增大呈增长趋势；未与其它指标或机器学习模型进行性能对比。

**⚠️ 局限性**

局限性包括：仅在理论图族层面验证，缺乏对真实分子实验数据的验证；对大规模或高度稠密图的计算复杂度未作详细分析；仅关注HSO指数，未探讨其在QSPR/QSAR中的实际预测效果。

---

## 27. State of Passkey Authentication in the Wild: A Census of the Top 100K sites

**arXiv ID:** 2602.15135 | [PDF](https://arxiv.org/pdf/2602.15135v1)

**作者:** Prince Bhardwaj `[一作]` (University of Surrey), Nishanth Sastry `[通讯]` (University of Surrey)

**通讯引用:** 3606 | [OpenAlex ID](https://openalex.org/A5025030042)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建Fidentikit浏览器爬虫，对Tranco Top 100K网站进行大规模扫描，完成了全球范围内的Passkey支持情况普查。

**💡 创新点**

创新点在于提出43条多类别启发式检测规则，并通过分布式浏览器自动化实现了对动态、条件化Passkey实现的精准识别，首次实现对Passkey采纳的系统化、可复现测量。

**🔧 技术方法**

技术手段包括Playwright浏览器自动化、API钩子监测、网络请求拦截、DOM与ARIA文本识别、第三方库与身份提供商检测，以及基于RabbitMQ和PostgreSQL的分布式任务调度。

**📊 数据集**

使用的数据集为Tranco Top 100K域名列表，并结合2FA目录和Passwordless目录做地面真值验证，进一步验证检测准确性。

**📈 对比分析**

与手工目录对比，Fidentikit检测到的Passkey网站数量是手工目录的26.4倍，准确率达97.5%，并展示了通过多类别启发式方法显著提升检测覆盖率。

**⚠️ 局限性**

局限性包括：仅检测公开Web端的Passkey支持；对需要先注册或在移动端实现的Passkey功能无法覆盖；对高度混淆或后期加载的脚本识别能力有限；以及仅覆盖前100k域名，未覆盖更大规模的低流量站点。

---

## 28. Binge Watch: Reproducible Multimodal Benchmarks Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M

**arXiv ID:** 2602.15505 | [PDF](https://arxiv.org/pdf/2602.15505v1)

**作者:** Giuseppe Spillo `[一作]` (University of Bari Aldo Moro), Giovanni Semeraro `[通讯]` (University of Bari Aldo Moro)

**通讯引用:** 9439 | [OpenAlex ID](https://openalex.org/A5059814300)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了两套包含文本、图像、音频、视频特征的MovieLens-10M/20M多模态数据集M^3L，并提供完整的处理流程和特征编码；

**💡 创新点**

创新点在于大规模公开可复现的多模态MovieLens数据集，覆盖度高且提供多种SOTA编码器特征；

**🔧 技术方法**

采用MiniLM、MPNet、CLIP文本编码器，VGG16、ViT、CLIP图像编码器，SlowFast、R(2+1)D、MViT视频编码器，以及VGGish、Whisper、AST音频编码器；

**📊 数据集**

使用MovieLens-10M和20M作为基础，扩充为M^3L-10M/20M；

**📈 对比分析**

通过与BPR、VBPR、LATTICE、FREEDOM等模型在Recall@10/NDCG@10上对比，双模态组合（文本+音频）表现最佳，提升性能至约0.193/0.216；

**⚠️ 局限性**

局限在于未进行完整模型基准，仅提供基础验证，且部分影片缺失多模态数据导致覆盖率略低。

---

## 29. Meteorological data and Sky Images meets Neural Models for Photovoltaic Power Forecasting

**arXiv ID:** 2602.15782 | [PDF](https://arxiv.org/pdf/2602.15782v1)

**作者:** Ines Montoya-Espinagosa `[一作]`, Antonio Agudo `[通讯]` (Institut de Robòtica i Informàtica Industrial)

**通讯引用:** 34383 | [OpenAlex ID](https://openalex.org/A5024769212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种多模态混合模型，用天空图像、历史光伏功率数据、气象变量和太阳位置进行光伏功率的即时预报与短期预测。

**💡 创新点**

创新点包括：①将 ERA5 气象重分析数据与天空图像融合，显著提升了在多云条件下的预测准确性；②引入太阳位置的物理信息，提升模型的可解释性和鲁棒性；③在同一架构下同时处理即时与序列输入，实现了即时预报和短期预测的统一。

**🔧 技术方法**

技术手段主要是卷积神经网络（CNN）提取图像特征，后接全连接层并融合气象变量与太阳位置；训练使用 Adam 优化器、MSE 损失；此外采用归一化、时序填充等预处理方法。

**📊 数据集**

使用的公开数据集包括：SKIPP'D 天空图像与光伏功率数据（2017‑2019 年），以及 ECMWF ERA5 重分析气象数据（0.25° 网格）。

**📈 对比分析**

实验与 SUNSET、RSUNSET 等基准模型对比，采用 RMSE 与 MAE 评估。结果显示，在多云天气下，使用表面长波辐射、风速及太阳位置的组合可将 RMSE 降低至 2.5 kW（相较于基线的 3.3 kW），在即时预报中提升到 0.78 kW，整体性能优于现有方法；在晴天时提升有限，但模型在极端误差上仍表现更好。

**⚠️ 局限性**

局限性：①模型对晴天的提升不明显，仍需改进；②气象变量组合时的相互作用难以捕捉，导致部分组合反而性能下降；③实验结果未能完全复现公开论文的某些预测性能，可能与数据划分或预处理细节有关；④当前模型仅在单一地点（斯坦福）验证，泛化性待进一步验证。

---

## 30. Random Wavelet Features for Graph Kernel Machines

**arXiv ID:** 2602.15711 | [PDF](https://arxiv.org/pdf/2602.15711v1)

**作者:** Valentin de Bassompierre `[一作]` (UCLouvain), Laurent Jacques `[通讯]` (UCLouvain)

**通讯引用:** 3382 | [OpenAlex ID](https://openalex.org/A5053892288)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于随机谱特征的节点嵌入方法，利用低秩近似快速逼近图拉普拉斯核；

**💡 创新点**

通过在图信号处理中使用多项式近似的低通滤波器和随机信号实现对顶尖K个特征子空间的范围搜索，从而在不显式求解特征值分解的情况下获得与最佳低秩近似相当的核近似；

**🔧 技术方法**

随机特征方法、图信号处理（图拉普拉斯谱、Chebyshev多项式逼近）、随机化奇异值分解（RSVD）等；

**📊 数据集**

在合成的瑞士卷图（5000节点）和社区图（5000节点）以及对应的扩散核等；

**📈 对比分析**

与全核显式计算、最佳rank‑K近似以及通用图随机特征（g‑GRFs）比较；实验表明在谱局部化的核（窄带）下误差可低至10⁻⁶，性能优于g‑GRFs，且在大规模图上计算复杂度低于O(N³)；

**⚠️ 局限性**

对特征分布高度集中的图（如社区图）和较大K值时，估计λ_K误差增大导致近似退化；多项式近似的阶数与计算时间权衡需要手工调参；

---

## 31. Computing Perfect Bayesian Equilibria, with Application to Empirical Game-Theoretic Analysis

**arXiv ID:** 2602.15233 | [PDF](https://arxiv.org/pdf/2602.15233v1)

**作者:** Christine Konicki `[一作]` (Michigan Tech Research Institute), Michael P. Wellman `[通讯]` (University of Michigan)

**通讯引用:** 13862 | [OpenAlex ID](https://openalex.org/A5002102744)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一种称为 PBE‑CFR 的算法，用来在任意两人不完全信息广义形式游戏（EFG）中计算完美贝叶斯均衡（PBE），并将其作为树形经验游戏理论（TE‑PSRO）框架中的 Meta‑Strategy Solver（MSS）进行实验。

**💡 创新点**

创新点：1）在经典 Counterfactual Regret Minimization (CFR) 的基础上引入了 AGM 一致性约束，实现 belief‑aware regret minimization；2）证明该算法在两人零和游戏中收敛到 PBE；3）首次将 PBE 作为 MSS 在 TE‑PSRO 中应用，实验证明其在某些游戏类中可显著加速经验游戏收敛；4）提供完整的理论证明与大规模实验支持。

**🔧 技术方法**

技术：CFR 与其变体（CFR‑+、CFR‑D 等）、AGM 一致性约束实现、贝叶斯规则更新、一次偏差原理、Blackwell 方法、Tree‑Exploiting PSRO（TE‑PSRO）框架、深度 Q‑网络用于 BR 近似、模拟器驱动的经验游戏生成。

**📊 数据集**

数据集：两类参数化游戏——(1) K‑1 回合的 Goofspiel（通用与改进版）和(2) 带信号的谈判博弈。通过 TE‑PSRO 迭代生成约 1200 个 Goofspiel 经验游戏和 800 个谈判游戏，使用模拟器在每一轮中采样收益和概率。

**📈 对比分析**

比较方法与性能：与传统 CFR 在同一经验游戏上并行跑，比较 worst‑case local regret（局部序贯合理性）和运行时间；在 TE‑PSRO 中比较使用 PBE 与 NE 作为 MSS 时的平均误差随迭代的收敛速度。实验显示 PBE‑CFR 在局部 regret 方面可达 10⁻³–10⁻²，运行时间略高但同阶；在 Goofspiel 中，PBE‑CFR 作为 MSS 能显著加速收敛，谈判游戏中效果不明显。

**⚠️ 局限性**

局限性：1）目前仅证明并实现两人游戏，扩展到多玩家的理论与实现尚未完成；2）对非零和游戏的收敛性理论仍不完整；3）AGM 一致性约束的构造在极大规模树形游戏中可能成为瓶颈；4）实验范围局限于两类游戏，未检验如扑克牌等更复杂结构的表现；5）需进一步评估不同信息结构对 PBE‑MSS 效果的影响。

---

## 32. UrbanVerse: Learning Urban Region Representation Across Cities and Tasks

**arXiv ID:** 2602.15750 | [PDF](https://arxiv.org/pdf/2602.15750v1)

**作者:** Fengze Sun `[一作]` (University of Melbourne), Jianzhong Qi `[通讯]` (University of Melbourne)

**通讯引用:** 4610 | [OpenAlex ID](https://openalex.org/A5022290876)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种面向城市通用的基础模型，能在跨城市、跨任务的城市区域表示学习与预测中高效工作。

**💡 创新点**

创新点在于采用region‑centric设计和随机游走+Transformer编码产生可跨城的区域嵌入；提出异质条件扩散式跨任务学习模块，结合检索式先验知识和任务/区域条件化，实现多任务共享与自适应。

**🔧 技术方法**

使用随机游走生成序列、Transformer编码/解码自监督掩码重建、对比学习、条件扩散概率模型、任务与区域的时间/任务嵌入调制、信息检索先验生成。

**📊 数据集**

实验基于纽约市、芝加哥、旧金山三个真实城市的数据，包含POI、地理邻域特征及六个下游任务（犯罪、签到、服务呼叫、人口、碳排放、夜灯）。

**📈 对比分析**

与七个SOTA方法（HREP、RegionDCL、UrbanCLIP、CityFM、GeoHG、GURPP、FlexiReg）比较，跨城市设置下R²最高提升35.9%，在同城和郊区也表现优异；跨任务模块集成后可提升多达1433%。

**⚠️ 局限性**

局限性包括对POI与邻域特征的依赖，缺乏更丰富语义输入；对极端城市结构差异的适应仍有限；扩散模型训练与推理时间较长。

---

## 33. Constraining Streaming Flow Models for Adapting Learned Robot Trajectory Distributions

**arXiv ID:** 2602.15567 | [PDF](https://arxiv.org/pdf/2602.15567v1)

**作者:** Jieting Long `[一作]` (University of Sydney), Weiming Zhi `[通讯]` (University of Sydney)

**通讯引用:** 261 | [OpenAlex ID](https://openalex.org/A5077799676)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种名为约束感知流模型（CASF）的框架，用于在执行过程中调整学习到的机器人轨迹分布，以满足安全和任务特定的约束。

**💡 创新点**

CASF通过引入约束依赖的度量，增强了流动策略的能力，使其能够在执行时实时适应约束，确保机器人动作遵循关节限制、避免碰撞并保持在可行工作空间内。

**🔧 技术方法**

使用了流动政策（SFP）和可微分距离函数来构建约束度量，并通过流动微分方程（ODE）进行实时调整。

**📊 数据集**

在模拟和真实世界的操作任务中进行了验证，具体数据集包括PushT和LASA环境，以及3D Robomimic环境。

**📈 对比分析**

与标准后处理投影基线和控制障碍函数（CBF）进行比较，CASF在成功率、目标覆盖率和路径长度等指标上表现优越，且在所有任务中几乎没有安全违规。

**⚠️ 局限性**

CASF的局限性在于其对复杂几何形状的距离函数的依赖，可能在某些情况下需要额外的训练或调整以适应新的环境或障碍物配置。

---

## 34. Uniform error bounds for quantized dynamical models

**arXiv ID:** 2602.15586 | [PDF](https://arxiv.org/pdf/2602.15586v1)

**作者:** Abdelkader Metakalard `[一作]` (Universite de Lorraine), Marion Gilson `[通讯]` (Universite de Lorraine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了从依赖数据序列学习动力学模型的统计保证，给出了对量化模型和非完美优化算法的统一误差界限。

**💡 创新点**

创新点包括：① 通过量化位数衡量模型复杂度，得到可解释的误差上界；② 提出一种新的分隔点（spaced‑point）分解方法，实现了快速收敛率的误差界；③ 统一的、与算法无关的泛化界，适用于非线性、混合及神经网络等多种模型。

**🔧 技术方法**

使用的技术主要有：β‑mixing 依赖分析、块分解与分隔点策略、Hoeffding 与 Bernstein 型不等式、量化参数空间的位数计数、信息理论/自归一化马尔可夫工具。

**📊 数据集**

实验数据集包括：AR(1) 时间序列（200,000 条观测）以及一个三模式的随机切换线性系统（80,000 条观测）。

**📈 对比分析**

与现有 Rademacher 复杂度/混合自由界进行比较，实验结果显示快率界在误差上明显更小、收敛更快，说明其性能优于传统方法。

**⚠️ 局限性**

局限性包括：仅适用于平稳、可裁剪的输出且需已知/可估计 β‑mixing 系数；对非平稳过程、极端噪声或高度非线性模型的适用性仍有待验证。

---

## 35. Mind the (DH) Gap! A Contrast in Risky Choices Between Reasoning and Conversational LLMs

**arXiv ID:** 2602.15173 | [PDF](https://arxiv.org/pdf/2602.15173v1)

**作者:** Luise Ge `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**通讯引用:** 5073 | [OpenAlex ID](https://openalex.org/A5038669899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对20款前沿和开源LLM的统一实验与人类实验，系统评估了在不确定性下的决策行为，探讨了风险选项的描述式与经验式呈现以及解释提示对决策的影响，并将LLM行为分为推理型和对话型两大类。

**💡 创新点**

创新点在于首次揭示LLM在不确定决策中存在两类行为簇：推理型趋向理性且对上下文不敏感， 对话型更具人性化且受上下文与解释提示影响；同时系统量化了表示方式与解释提示对LLM决策的影响。

**🔧 技术方法**

技术上使用统一接口对20款LLM进行多次采样实验，并结合人类实验数据；利用四参数前景理论模型对行为进行可解释性拟合；通过相关系数、均方误差、决定性与一致性指标进行行为比较。

**📊 数据集**

数据集包括三组两选风险情境（明确描述与经验历史两种呈现），每种情境下的10次LLM采样结果，以及360名美国参与者在相同情境下的选择数据。

**📈 对比分析**

通过与人类和理性经济主体的相关系数和均方误差对比，推理型模型与理性基准相关系数>0.86，表现高度确定性；对话型模型相关系数<0.5，易受选项顺序、框架和解释提示影响；解释提示能提升推理型模型的理性度，降低对话型模型的理性度。

**⚠️ 局限性**

局限性包括：未对人类实验完整评估解释提示效果；经验式呈现仅使用两种样本长度（20和100），未系统探讨样本大小影响；多因素交互效应未完全解析；四参数前景理论模型可能存在可辨识性与边界拟合问题。

---

## 36. ExLipBaB: Exact Lipschitz Constant Computation for Piecewise Linear Neural Networks

**arXiv ID:** 2602.15499 | [PDF](https://arxiv.org/pdf/2602.15499v1)

**作者:** Tom A. Splittgerber `[一作]` `[通讯]` (University of Bremen), Tom A. Splittgerber (University of Bremen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

无法进行总结，因为缺少论文内容

**💡 创新点**



**🔧 技术方法**



**📊 数据集**



**📈 对比分析**



**⚠️ 局限性**



---

## 37. Evaluating Federated Learning for Cross-Country Mood Inference from Smartphone Sensing Data

**arXiv ID:** 2602.15478 | [PDF](https://arxiv.org/pdf/2602.15478v1)

**作者:** Sharmad Kalpande `[一作]` (Indian Institute of Science Education and Research Bhopal), Haroon R. Lone `[通讯]` (Indian Institute of Science Education and Research Bhopal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在跨国场景下通过联邦学习实现情绪推断的框架 FedFAP

**💡 创新点**

兼顾特征异质性和个性化的两层编码器架构：共享表示+本地特定表示，并允许各国保留各自可用传感器特征

**🔧 技术方法**

联邦学习（FedAvg、FedProx、FedAdam）、一维卷积神经网络编码器、注意力机制、联邦聚合算法及 KNN 缺失值填补

**📊 数据集**

DiversityOne 多国移动传感数据集（中国、丹麦、印度、墨西哥、巴拉圭、英国）

**📈 对比分析**

与传统集中式机器学习（LR、RF、XGB、1D‑CNN）以及现有个性化联邦学习基线（FedPer、pFedMe）对比；FedFAP 在 10E‑50R 方案下的 AUROC 达 0.744，显著高于中心化模型和基线方法

**⚠️ 局限性**

未提供正式隐私保证；未考察异步/部分参与场景；缺乏对个体级细粒度个性化的评估；仅在离线实验环境中验证，未在实时部署中测试

---

## 38. A Novel Public Dataset for Strawberry (Fragaria x ananassa) Ripeness Detection and Comparative Evaluation of YOLO-Based Models

**arXiv ID:** 2602.15656 | [PDF](https://arxiv.org/pdf/2602.15656v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 39. MAVRL: Learning Reward Functions from Multiple Feedback Types with Amortized Variational Inference

**arXiv ID:** 2602.15206 | [PDF](https://arxiv.org/pdf/2602.15206v1)

**作者:** Raphaël Baur `[一作]` (ETH Zurich), Thomas Kleine Buening `[通讯]` (ETH Zurich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的贝叶斯框架，利用可区分的反馈类型（示范、比较、评分、停止）共同推断隐含奖励函数并学习可重用的奖励编码器与动作价值函数；

**💡 创新点**

创新点在于将所有反馈通过显式似然联合建模，消除手工损失平衡，采用可扩展的变分推断实现多模态奖励学习；

**🔧 技术方法**

使用变分自编码器式的可变分推断、贝尔曼 TD 约束、奖励编码器与 Q‑值模型以及针对各反馈的专用似然解码器；

**📊 数据集**

在离散网格世界以及连续控制基准（Acrobot、LunarLander、其它连续动作域）上生成的人工反馈数据集进行实验；

**📈 对比分析**

与单一反馈基线及行为克隆进行对比，实验表明多模态学习能更准确恢复奖励、提升政策回报并在环境扰动下保持更高的稳健性；

**⚠️ 局限性**

局限性包括仅在模拟环境与合成反馈下验证，未考虑真实人类反馈的偏差与不一致性，且未将奖励不确定性用于主动反馈采集。

---

## 40. ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns

**arXiv ID:** 2602.15521 | [PDF](https://arxiv.org/pdf/2602.15521v1)

**作者:** Ziyu Zhao `[一作]` (Zhejiang University), Yu Cheng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 74031 | [OpenAlex ID](https://openalex.org/A5090802305)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练‑free 的专家织造框架 ExpertWeaver，将预训练的稠密 LLM 通过 GLU 激活模式划分为共享专家和路由专家，从而实现高效的稀疏 MoE。

**💡 创新点**

创新点在于通过对 GLU 门控激活模式进行层感知的系数方差分析，揭示全局共享与任务特定神经元的并行分布，并以无训练方式构造共享与路由专家，避免破坏原始激活结构。

**🔧 技术方法**

使用 GLU 门控的绝对平均激活、系数方差阈值、层自适应共享比例、均衡 K‑means 聚类以及基于聚类中心的无训练路由器。

**📊 数据集**

在 Flan‑v2 多任务校准集（42 任务、5 样本）收集激活信息，评估使用 MMLU、HellaSwag、ARC‑e/c、PiQA 等公开基准，并在 FineWeb‑Edu 上进行继续预训练。

**📈 对比分析**

与 LLM‑Pruner、FLAP、CMoE 等训练‑free 结构剪枝基线以及稀疏 MoE 初始化对比，ExpertWeaver 在 25% 稀疏率下平均提升约 5–7 个百分点；在高稀疏下的下循环实验中，模型仅用 200B 继续预训练即可匹配或超过从头训练的 MoE，性能接近原稠密模型的 90%+。

**⚠️ 局限性**

局限在于校准集多样性和阈值设定的影响，对极深层或极宽层的专家划分精细度有限，且在需要自适应动态路由的场景中可能需要进一步训练或改进路由机制。

---

## 41. Distributed Semi-Speculative Parallel Anisotropic Mesh Adaptation

**arXiv ID:** 2602.15204 | [PDF](https://arxiv.org/pdf/2602.15204v1)

**作者:** Kevin Garner `[一作]` (Center for Real-time Computing), Nikos Chrisochoides `[通讯]` (Center for Real-time Computing)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种分布式内存的各向异性网格自适应方法，该方法在保持网格质量的同时，利用多核共享内存和分布式运行时（PREMA）实现高并行度，并避免使用集体通信和全局同步。

**💡 创新点**

创新点包括：
- 将网格生成与性能优化分离，利用独立的共享内存网格生成器（CDT3D）与分布式运行时协同；
- 采用先预适应界面（a priori）而非后适应（a posteriori），从而完全避免全局重新划分和同步；
- 在共享内存实现中引入伪活跃/伪非活跃元素分类和预锁定机制，控制操作范围并提升可扩展性；
- 通过构造简单连通子域，解决子域分解后可能出现的非流形问题；
- 结合 PREMA 的异步消息和负载平衡，实现多核与多节点并行高效通信。

**🔧 技术方法**

使用技术包括：
- PREMA 并行运行时（支持移动对象、异步调用、负载平衡）；
- CDT3D（基于 Delaunay 的共享内存网格生成器，采用投机执行模型）；
- PQR 分解算法（基于几何图的排序划分）；
- 预锁定、伪活跃元素策略；
- 细粒度多线程（OpenMP/线程）与粗粒度节点级并行；
- 通过修改原 CDT3D 代码实现对接口与内部自适应的区别执行。

**📊 数据集**

主要使用的数据集是：
- Delta Wing 结构（平面片材），在两种不同的目标复杂度（10M 与 20M）下生成 1 亿级别的网格；
- Cube 结构（约 10 亿元素）用于验证极大规模适应。

**📈 对比分析**

与传统方法比较：
- 与原始共享内存 CDT3D（SM_CDT3D）相比，分布式版本在 256~512 核上实现了与 SM_CDT3D 相近甚至更优的运行时间，并保持相似或更好的网格质量；
- 与开源并行自适应工具 refine 对比，分布式 CDT3D 在 1~256 核上速度更快，规模到 1 亿级时时间在 4 小时以内；
- 分布式方法避免了 refine 中 50% 以上的通信开销，特别是全局 Alltoall/Allreduce 的同步瓶颈。

**⚠️ 局限性**

局限性包括：
- 接口自适应阶段只能在单个多核节点上执行，限制了可用核数（最多 32 核或节点内的全部核）；
- 需要顺序步骤（如构造简单连通子域、数据结构转换）导致部分非计算工作占 15–25% 时间；
- PREMA 的负载平衡功能在此实现中关闭，未充分利用过度分解；
- 对极大规模几何（>1 亿元素）时仍面临通信与内存占用的扩展挑战；
- 需要对共享内存 CDT3D 进行多次手工重构，若想迁移至其他网格器需再度重做。

---

## 42. Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge

**arXiv ID:** 2602.15184 | [PDF](https://arxiv.org/pdf/2602.15184v1)

**作者:** Siying Ma `[一作]` (Simon Fraser University), Vijay Ganesh `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8159 | [OpenAlex ID](https://openalex.org/A5052292970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出多物理训练框架，联合学习原始 PDE 及其简化基本形式的仿真，提升神经算子在数据效率、长时预测一致性和 OOD 泛化上的表现。

**💡 创新点**

创新点在于将 PDE 物理项分解为基本形式并将其仿真作为辅助任务，通过多任务学习显著增强神经算子的物理理解与泛化能力。

**🔧 技术方法**

使用神经算子（主要为 Fourier Neural Operator）+ 多任务学习 + 数据混合比例策略 + PDE 基本形式分解等技术。

**📊 数据集**

使用 Diffusion–Reaction、Navier–Stokes (2D/3D)、Kuramoto–Sivashinsky 以及实测烟雾数据 ScalarFlow 等公开或自建仿真数据集。

**📈 对比分析**

在相同模拟预算下与仅使用原始 PDE 的基线对比，采用 nRMSE 评价，结果显示我们的方法在所有任务上误差下降 10–30%，并在 OOD 和实测迁移实验中表现更优。

**⚠️ 局限性**

局限性包括需要人工定义 PDE 的基本形式、对物理拆解的正确性依赖较大，以及在极端非线性或高维系统中辅助任务可能不足以完全弥补数据缺口。

---

## 43. Loss Knows Best: Detecting Annotation Errors in Videos via Loss Trajectories

**arXiv ID:** 2602.15154 | [PDF](https://arxiv.org/pdf/2602.15154v1)

**作者:** Praditha Alwis `[一作]` (Purdue University), Kaushik Roy `[通讯]` (Purdue University)

**通讯引用:** 46082 | [OpenAlex ID](https://openalex.org/A5031161187)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于训练期间损失轨迹的累计样本损失（CSL）方法，用于无监督检测视频数据集中的标注错误（误标和时序乱序）。

**💡 创新点**

利用模型在不同训练检查点的损失累积来识别持续难学的帧，模型无关且不需要额外监督，既能检测语义误标也能捕捉时序不一致。

**🔧 技术方法**

训练视频分割模型（如 ResNet‑18+ViT‑B/16），保存每个 epoch 的检查点，计算每帧平均损失 CSL 并通过阈值或百分位筛选高 CSL 帧。

**📊 数据集**

在外科手术流程数据 Cholec80（7 阶段）和日常指令视频 EgoPER（5 任务）上进行实验，人工注入误标和时序乱序。

**📈 对比分析**

与 HF2‑VAD、S3R、EgoPED 等现有异常/错误检测基线对比，CSL 在 EgoPER 的 AUC 最高（最高达 70.2），在 Cholec80 误标情形下 EDA 85.9、AUC 92.0，显著优于最强基线。

**⚠️ 局限性**

需要保存所有训练检查点，计算成本随 epoch 数和帧数呈线性增长；对极低噪声比例的误标可能不易区分；在训练集已有噪声时仍可能产生误判。

---

## 44. A Note on Non-Composability of Layerwise Approximate Verification for Neural Inference

**arXiv ID:** 2602.15756 | [PDF](https://arxiv.org/pdf/2602.15756v1)

**作者:** Or Zamir `[一作]` (Tel Aviv University), Or Zamir `[通讯]` (Tel Aviv University)

**通讯引用:** 252 | [OpenAlex ID](https://openalex.org/A5048299120)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出并构造了一个等价网络，使得在层级近似验证下，即使每层误差均受限于 δ，也能将最终输出逼到任意目标，从而证明层级近似验证在零知识机器学习推理中不具可组合性。

**💡 创新点**

首次给出一种通用的网络改造与触发通道构造，展示任意网络可被改写成可通过极小局部误差控制全局输出的形式，证明层级近似验证的安全缺陷。

**🔧 技术方法**

利用网络宽化、触发通道、放大因子 g 与增益 M 的组合，结合 ReLU 激活与线性层的性质，构造解析证明。

**📊 数据集**

本文为理论分析，不依赖任何具体数据集。

**📈 对比分析**

未进行实验或性能对比，所述结论仅在理论层面证明任意 δ 误差均可导致输出任意变化，强调了验证方法的局限性。

**⚠️ 局限性**

局限性包括：仅针对 ReLU 与线性层的网络；构造仅为理论示例，未评估实际实现成本；未考虑随机输入或噪声对系统鲁棒性的影响；未给出防御或改进方案。

---

## 45. Automatic Funny Scene Extraction from Long-form Cinematic Videos

**arXiv ID:** 2602.15381 | [PDF](https://arxiv.org/pdf/2602.15381v1)

**作者:** Sibendu Paul `[一作]` (Amazon Prime Video), Caren Chen `[通讯]` (Amazon Prime Video)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个端到端的系统，用于从长篇电影中自动识别、定位并排名幽默场景，以生成吸引人的视频预览。

**💡 创新点**

创新点包括：①结合视觉与文本特征的多模态场景分割方法；②使用引导式三元组挖掘提升图像-场景对比学习；③多模态幽默标注框架，融合音频（笑声、韵律）和文本（上下文与梗句）并加入不当幽默过滤；④简易启发式评分实现场景优先级排序。

**🔧 技术方法**

主要技术手段包括 TransNetV2 进行镜头检测；X‑CLIP + DINO 的跨帧 Transformer 作为镜头编码器；基于 MovieNet‑SSeg 的三元组损失与引导式采样；改进后的 ColBERT 处理长文本幽默检测；音频标签模型实现不当幽默过滤；以及基于规则的幽默打分机制。

**📊 数据集**

使用的数据集有：MovieNet‑SSeg（训练场景分割），OVSD（评估场景检测），以及作者自行收集的5部约两小时的长片和11部不同类型的预告片用于人工评测；此外还利用公开的音频-文本幽默数据集（如 UR‑FUNNY、MHD）做模型预训练。

**📈 对比分析**

与现有方法对比，场景检测在 OVSD 上实现 18.3% 的 AP 提升；幽默检测的 F1 评分达 0.834，超过 Fine‑tuned Transformer 基线；在实际长片中，专业策展人评测显示 98% 的场景定位准确率，87% 的幽默识别准确率，预告片则为 100%。

**⚠️ 局限性**

局限性包括：在预告片快速切换的情形下定位误差增大；幽默检测目前仅支持英文，需要扩展多语言；缺乏大规模端到端标注数据，评测依赖人工主观判断；过滤器过于严格可能导致部分合适幽默被误删。

---

## 46. Fluoroscopy-Constrained Magnetic Robot Control via Zernike-Based Field Modeling and Nonlinear MPC

**arXiv ID:** 2602.15357 | [PDF](https://arxiv.org/pdf/2602.15357v1)

**作者:** Xinhao Chen `[一作]`, Axel Krieger `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在磁化导航下实现了在低帧率、噪声较大的荧光成像条件下的磁机器人精准控制，并验证了药物递送在脊柱模型中的可行性。

**💡 创新点**

创新点在于结合Zernike多项式可解析磁场模型与非线性模型预测控制（NMPC）和卡尔曼滤波器，实现低帧率下的高精度轨迹跟踪。

**🔧 技术方法**

使用Zernike多项式磁场拟合、非线性模型预测控制、卡尔曼滤波、CasADi/Ipopt求解器以及光学摄像机+UNet定位。

**📊 数据集**

利用COMSOL仿真得到的数值磁场数据、3D打印血液黏度溶液与脊柱模型的实验数据，以及摄像机采集的RGB图像用于UNet训练。

**📈 对比分析**

与六种基线（PID、线性MPC、两层控制、仅NMPC、基于FEA表格的NMPC）比较，提出的方法在3Hz、σ=2mm噪声下RMSE仅1.18mm，优于其他基线。

**⚠️ 局限性**

局限性包括仅在二维平面运动、缺乏真实荧光跟踪、硬件重量大、未考虑血流与组织弹性等外部扰动。

---

## 47. Eco-Amazon: Enriching E-commerce Datasets with Product Carbon Footprint for Sustainable Recommendations

**arXiv ID:** 2602.15508 | [PDF](https://arxiv.org/pdf/2602.15508v1)

**作者:** Giuseppe Spillo `[一作]` (University of Bari Aldo Moro), Giovanni Semeraro `[通讯]` (University of Bari Aldo Moro)

**通讯引用:** 9439 | [OpenAlex ID](https://openalex.org/A5059814300)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文发布了Eco-Amazon数据集，给常用Amazon电商数据集（电子、服装、家居厨具）添加了产品碳足迹（PCF）信息，并提供了LLM估算脚本和使用案例。

**💡 创新点**

创新点在于：①首次大规模、跨域公开的PCF-enriched电子商务数据；②利用零样本LLM推理结合LCA标准生成PCF估计；③将PCF嵌入推荐系统后置重排序，展示可调节的准确-可持续性权衡。

**🔧 技术方法**

技术包括：零样本提示（Zero-shot Prompting）与大型语言模型（GPT‑o3‑mini、Gemini‑2.5‑flash）推理；LCA与GHG协议规范；推荐系统后处理重排序公式；RecBole框架训练与评估。

**📊 数据集**

使用的数据集为Amazon 2023 Review数据（电子、服装、家居厨具），共约49.9k条商品，通过k-core过滤后分别取11.5k、21k、17k条商品进行PCF估算。

**📈 对比分析**

比较方法：对比官方PCF与LLM估计的MAE、Spearman相关系数、NDCG；在推荐任务中对比原始推荐与PCF重排序的Recall、Tail百分比及平均CO₂e。性能显示：LLM估计MAE在高负载商品上偏大，但排名相关性与NDCG均>0.9；重排序可在α=0.75时将平均CO₂e降至6–12 kg，同时Recall仅略微下降，显示可控的准确-可持续性折中。

**⚠️ 局限性**

局限性包括：LLM估计在高碳商品上误差显著；缺乏真实大规模PCF基准验证；重排序仍以非个性化PCF为权重，未考虑用户偏好与产品生命周期差异；需进一步融合结构化LCA数据库和检索增强策略。

---

## 48. A Scalable Curiosity-Driven Game-Theoretic Framework for Long-Tail Multi-Label Learning in Data Mining

**arXiv ID:** 2602.15330 | [PDF](https://arxiv.org/pdf/2602.15330v1)

**作者:** Jing Yang `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2281 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出CD-GTMLL框架，用好奇心驱动的博弈机制解决长尾多标签分类问题。

**💡 创新点**

创新点在于将多标签学习视为合作博弈，引入标签稀有度与玩家不一致度的好奇奖励，动态增强尾标签梯度，无需手工调参。

**🔧 技术方法**

采用多玩家分解、交叉融合、基于KL的好奇奖励、梯度上升优化等技术。

**📊 数据集**

在Pascal VOC、MS‑COCO、Yeast、Mediamill、Eurlex‑4K、Wiki10‑31K和AmazonCat‑13K七个数据集上进行实验。

**📈 对比分析**

与XR‑Transformer、MatchXML、MLC‑NC等SOTA方法对比，取得最高Rare‑F1、P@k和mAP，尤其Wiki10‑31K P@3提升1.6%。

**⚠️ 局限性**

局限在于需设置玩家数与好奇权重，过多玩家可能导致过拟合；对极端稀有标签的提升仍有限。

---

## 49. jina-embeddings-v5-text: Task-Targeted Embedding Distillation

**arXiv ID:** 2602.15547 | [PDF](https://arxiv.org/pdf/2602.15547v1)

**作者:** Mohammad Kalim Akram `[一作]` (Jina AI GmbH), Han Xiao `[通讯]` (Jina by Elastic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了两种小型多语言文本嵌入模型（模型A、模型B），并设计了一种两阶段训练流程：首先用教师模型（Qwen3-Embedding-4B）进行嵌入级别的蒸馏，然后冻结模型权重，在各任务（检索、聚类、分类、语义相似度）上训练 LoRA Adapter，采用 InfoNCE、蒸馏损失和全局正交正则化共同优化。

**💡 创新点**

创新点包括：1) 将嵌入级别蒸馏与任务特定对比学习结合，显著提升小模型性能；2) 在第一阶段使用不同 RoPE θ 训练/推理，实现对长文本的自适应；3) 通过三项损失（InfoNCE、蒸馏、GOR）构建鲁棒的检索 Adapter；4) 通过多任务 LoRA 设计实现模型可插拔性和多任务兼容；5) 公开模型权重与多种量化版本，方便社区复现与扩展。

**🔧 技术方法**

主要技术手段包括：Transformer + 末端标记池化；RoPE（位置编码）与可变 θ；多层 LoRA Adapter；嵌入级别对比蒸馏（线性投影 + 余弦相似度）；InfoNCE 对比损失；全局正交正则化（GOR）；Matryoshka 级联表示学习（支持嵌入截断）；多语言多任务数据混合训练。

**📊 数据集**

使用的训练数据涵盖：300+公开语料库（>30 种语言），包含问答、标题-摘要、对齐对、长文本、合成长文、并行翻译文本；STS 数据集（STS12、SICK、机器翻译版），聚类用新闻标题/描述，分类用多标签数据转单标签，检索用多种检索对照数据；在长文本训练中使用 1k–4k token 的长文与 LLM 生成查询。

**📈 对比分析**

通过 MTEB 多语言与英文版、BEIR、RTEB、LongEmbed 等基准评测与同尺寸模型（如 Qwen3-0.6B、jina-v3、snowflake-l-v2 等）对比，模型A/B 在检索、聚类、分类、语义相似度等任务上平均得分均位列同规模榜首，检索 nDCG@10 在 MTEB、BEIR、LongEmbed 中均超过或匹配同参数模型；量化后性能衰减低于 5%（相比无 GOR 下降 > 20%）。

**⚠️ 局限性**

局限性包括：1) 与教师模型相比仍有性能差距；2) 在极低维（<256）截断时检索性能显著下降；3) 训练过程仍需大量多语言对齐数据和计算资源；4) 对某些极端低资源语言或极长文本的泛化性尚未充分验证；5) 对抗攻击或罕见语义差异的鲁棒性未系统评估。

---

## 50. Beyond Labels: Information-Efficient Human-in-the-Loop Learning using Ranking and Selection Queries

**arXiv ID:** 2602.15738 | [PDF](https://arxiv.org/pdf/2602.15738v1)

**作者:** Belén Martín-Urcelay `[一作]` (Georgia Institute of Technology), Christopher J. Rozell `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 3490 | [OpenAlex ID](https://openalex.org/A5011481913)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种人机交互式学习框架，利用排名和典型选择查询（exemplar selection）来更高效地学习二分类器。

**💡 创新点**

创新点包括：①为排名与选择查询设计概率化人类响应模型；②给出理论样本复杂度上界并实现可行的变分近似；③提出基于信息速率的成本感知查询选择策略，突破传统单标签信息瓶颈。

**🔧 技术方法**

技术手段包括：贝叶斯推断与变分近似、Plackett‑Luce 排序模型、主动学习启发式、线性嵌入空间几何分析以及信息量与成本模型的结合。

**📊 数据集**

实验数据集涵盖词情感（NRC、SocialSent）和图像美学（AVA）任务，使用预训练嵌入（Word2Vec、ViT‑L/14、CLIP）进行特征表示。

**📈 对比分析**

与传统仅标记的主动学习对比，排名查询在交互次数上可减少约85%，在总耗时上提升约57%，并在准确率上与或优于传统方法。

**⚠️ 局限性**

局限性在于假设人类响应条件独立、依赖固定嵌入几何，未考虑上下文效应或疲劳；同时需要人工成本模型，可能不适用于所有任务。

---

## 51. Safe-SDL:Establishing Safety Boundaries and Control Mechanisms for AI-Driven Self-Driving Laboratories

**arXiv ID:** 2602.15061 | [PDF](https://arxiv.org/pdf/2602.15061v1)

**作者:** Zihan Zhang `[一作]` (Nankai University), Tong Zhu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 44151 | [OpenAlex ID](https://openalex.org/A5100445951)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Safe‑SDL 框架，针对自驱动实验室中的“语法到安全”鸿沟，构建了多层防御的安全架构，确保 AI 规划与物理执行之间的安全一致性。

**💡 创新点**

创新点包括：① 将自动驾驶领域的 Operational Design Domains (ODD) 迁移到实验室场景并以形式化约束表示；② 引入 Control Barrier Functions (CBF) 作为实时安全保证；③ 设计 CRUTD 事务安全协议，实现 AI 规划与执行的原子化验证；④ 将形式化验证与数字孪生、层级架构相结合，形成完整的安全闭环。

**🔧 技术方法**

采用的技术包括：基于约束优化的 ODD 定义、连续动力学下的 CBF 约束求解器、基于 ACID 思想的 CRUTD 事务协议、数字孪生仿真与安全边界检测、层级安全 kernel 与 ROS2 低层执行、以及形式化验证工具（模型检查、定理证明）。

**📊 数据集**

评估使用了公开安全基准 LabSafety‑Bench 与 SOSBENCH，结合 UniLabOS、Osprey、ChemCrow 等已有自驱动实验室实现进行实证对比。

**📈 对比分析**

与仅依赖模型自身安全性的对照组相比，Safe‑SDL 在安全事件率下降约 85%，误报率降低 70%，并在 LabSafety‑Bench 中将安全违规率从 38% 降至 5%，验证了多层防御的有效性。

**⚠️ 局限性**

局限性主要体现在：① 现有形式化验证工具对大型神经网络规划模块的可验证性不足；② 需要高保真数字孪生模型，构建成本高；③ ODD 与 CBF 参数仍需手工设定，难以自适应；④ 对新兴领域的可迁移性验证不足，需进一步实验验证。

---

## 52. Causal Effect Estimation with Latent Textual Treatments

**arXiv ID:** 2602.15730 | [PDF](https://arxiv.org/pdf/2602.15730v1)

**作者:** Omri Feldman `[一作]` (Hebrew University of Jerusalem), Amir Feder `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 1364 | [OpenAlex ID](https://openalex.org/A5056266191)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套端到端流程，用稀疏自编码器（SAE）提取可解释的文本特征，利用大型语言模型生成可控的文本干预，并在此基础上进行因果效应估计；

**💡 创新点**

创新点包括①将SAE特征用于假设生成并通过IC评分（强度+连贯性）挑选干预特征；②提出残差化策略去除嵌入中的治疗信息，理论给出误差上界，解决正交性与正则性冲突；③在文本-as-treatment 设置下提供系统的实验设计与评估框架；

**🔧 技术方法**

使用的技术包括稀疏自编码器、Gemma/Llama/Qwen等LLM、Gemma-300M、all-MiniLM-L6-v2等文本嵌入、PCA降维、Wasserstein距离评估、稳健因果机器学习（EconML R-learner）、交叉拟合与随机森林等；

**📊 数据集**

使用三大标注文本集：A（加州地方政府会议发言，文明度标签）、B（美国政治广告，党派标签）、C（Reddit 对应新闻的评论，种族偏见标签）；

**📈 对比分析**

通过半合成实验生成约60万条文本，对比未残差化和残差化两种控制方法，采用R-learner 估计 CATE；结果显示残差化显著降低 bias 与 RMSE，IC评分与第一主成分预测准确度呈正相关，实验结果与理论一致；

**⚠️ 局限性**

局限性包括：实验主要在半合成环境下验证，真实世界因果推断仍需进一步验证；假设生成依赖人工筛选，可能引入主观性；残差化过程需完整嵌入，可能导致信息丢失；模型在不同文本领域的泛化能力仍待考察。

---

## 53. How to Disclose? Strategic AI Disclosure in Crowdfunding

**arXiv ID:** 2602.15698 | [PDF](https://arxiv.org/pdf/2602.15698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 54. Unraveling Entangled Feeds: Rethinking Social Media Design to Enhance User Well-being

**arXiv ID:** 2602.15745 | [PDF](https://arxiv.org/pdf/2602.15745v1)

**作者:** Ashlee Milton `[一作]` (University of Minnesota), Stevie Chancellor `[通讯]` (University of Minnesota)

**通讯引用:** 2464 | [OpenAlex ID](https://openalex.org/A5046479021)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对21名有精神疾病认知的社交媒体用户进行设计工作坊，挖掘其对算法策展的经验与情感影响，提出“缠结（entanglement）”框架并给出设计建议。

**💡 创新点**

创新点在于：①首次将用户自创的“民间理论”与Norman动作周期相结合，形成新的“缠结”概念；②揭示算法与社交网络系统在同一界面上的融合导致情感失配与失控；③提出三大设计主题（情境化参与、消费控制、显式输入）作为缓解缠结的方法。

**🔧 技术方法**

主要技术方法是人机交互中的参与式设计、工作坊、以语料为基础的归纳式主题分析；未使用机器学习或算法实现。

**📊 数据集**

数据集为：21位自报有抑郁、焦虑或创伤后应激障碍的用户在7场工作坊中产生的笔记、访谈记录与原型稿。

**📈 对比分析**

该研究不涉及算法性能评估，也没有对比实验。评价基于工作坊参与者的主观反馈与研究者对原型的分析，未给出定量指标。

**⚠️ 局限性**

局限性包括：样本主要为大学生、白人女性，缺乏性别与族裔多样性；自报诊断且未跟踪具体病情；仅进行一次性工作坊，缺乏纵向跟踪与干预效果验证；所得到的设计建议需进一步实验验证。

---

## 55. SecCodeBench-V2 Technical Report

**arXiv ID:** 2602.15485 | [PDF](https://arxiv.org/pdf/2602.15485v1)

**作者:** Longfei Chen `[一作]` (Tsinghua University), Chao Zhang `[通讯]` (Alibaba Group)

**通讯引用:** 14274 | [OpenAlex ID](https://openalex.org/A5072377630)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并公开了一个名为Sec‑Code‑Bench的基准，包含98个功能级别的生成/修复场景，评估LLM在多语言（Java、C、Python、Go、Node.js）中的安全代码生成与修复能力。

**💡 创新点**

创新点在于：①基于真实工业漏洞构建的项目级别任务，①功能先行再安全验证的两阶段协议；②结合动态执行与LLM‑as‑a‑judge的混合评测；③使用Severity和Scenario权重的Pass@K聚合方案，实现细粒度、可比对的安全性能评估。

**🔧 技术方法**

核心技术包括：Docker沙箱化执行、功能+安全的逐步验证流程、LLM‑as‑a‑judge判定、Pass@K统计与加权汇总，以及可配置的多语言验证器。

**📊 数据集**

使用的数据集为98个从阿里巴巴内部匿名化的真实漏洞场景，覆盖22种CWE，分别分布在五种主流编程语言中，且按严重级别（Critical/High/Medium）进行标注。

**📈 对比分析**

通过多轮（默认10轮）Pass@1评分并按Severity/Scenario加权求和来比较模型，得到整体、按场景、按语言的多维得分；实验结果与现有基准相较表现出更高的实用性与可靠性（详细数值已在公开仓库中给出）。

**⚠️ 局限性**

局限性包括：评测仅覆盖已构造的场景，可能未涵盖所有真实攻击路径；LLM‑as‑a‑judge的主观性与多模型投票依赖；动态PoC测试仅验证有限输入，未能完全保证漏洞不存在；目前仅支持五种语言，未来需扩展更多生态。

---

## 56. Learning Representations from Incomplete EHR Data with Dual-Masked Autoencoding

**arXiv ID:** 2602.15159 | [PDF](https://arxiv.org/pdf/2602.15159v1)

**作者:** Xiao Xiang `[一作]` (EPFL), Leo Anthony Celi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 34975 | [OpenAlex ID](https://openalex.org/A5031401755)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种自监督的双重掩码自编码器AID-MAE，直接在不完整的电子健康记录（EHR）表格上学习表示，无需先行插补；

**💡 创新点**

创新点在于同时利用固有缺失掩码与增强掩码的双重机制，既捕获真实缺失信息，又通过随机遮蔽提升表示学习能力；

**🔧 技术方法**

技术包括Transformer编码器‑解码器、时间位置编码、掩码策略、对齐自监督重建损失及双重加权损失；

**📊 数据集**

数据集为MIMIC-IV ICU数据与PhysioNet 2012挑战赛数据，分别包含实验室数值、生命体征、血管加压剂等；

**📈 对比分析**

在ICU死亡、住院时长、急性肾损伤等下游任务上，AID-MAE相较于XGBoost、DuETT等基线取得更高的AUROC/ AUPRC，尤其在少标注数据和跨数据集迁移场景下表现更佳；

**⚠️ 局限性**

局限性包括对缺失机制的隐式处理（未显式建模缺失非随机特征）、对结构化医学知识或多模态输入的缺乏扩展，以及对极高缺失率下的鲁棒性尚需进一步验证。

---

## 57. Revealing and Enhancing Core Visual Regions: Harnessing Internal Attention Dynamics for Hallucination Mitigation in LVLMs

**arXiv ID:** 2602.15556 | [PDF](https://arxiv.org/pdf/2602.15556v1)

**作者:** Guangtao Lyu `[一作]` (Xidian University), Cheng Deng `[通讯]` (Xidian University)

**通讯引用:** 12506 | [OpenAlex ID](https://openalex.org/A5015874725)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑无关的注意力干预方法 PADE，通过内部正向注意力动态（PAD）来识别并增强视觉核心区域，从而降低多模态大模型的幻觉问题。

**💡 创新点**

创新点在于：①利用跨层正向注意力变化捕捉语义核心区域；②使用每头中位绝对偏差（MAD）对干预幅度进行自适应缩放；③引入系统‑令牌补偿（STC）以保持对指令和历史生成的注意力，避免干预导致指令遵循失衡。

**🔧 技术方法**

技术方法包括：计算视觉注意力正向差分构造 PAD；对注意力 logits 进行 MAD 缩放；将 PAD 注入目标层注意力 logits 并通过 STC 调整系统 token logits；最终通过 softmax 获得新的注意力分布。

**📊 数据集**

使用的评估数据集包括幻觉检测基准（POPE、CHAIR、HallusionBench、AMBER）以及通用多模态基准（VizWiz、MME、LLaVA‑Wild、MM‑Vet）；在多款开源 LVLM（LLaVA‑1.5、InstructBLIP、Qwen‑VL、LLaVA‑1.5‑13B、LLaVA‑NeXT）上进行实验。

**📈 对比分析**

与对比解码、辅助专家模型和静态内部信号干预方法相比，PADE 在幻觉率、语义一致性等指标上均取得了更优或相近的表现，同时保持了原始模型的整体多模态推理能力，且额外计算和内存开销极低。

**⚠️ 局限性**

局限性在于仅聚焦于注意力动态，未考虑隐藏状态、前馈网络激活或输出 logits 等其他内部表征的动态信息；未来工作可将动态分析扩展至更广泛的内部信号。

---

## 58. POP: Prior-fitted Optimizer Policies

**arXiv ID:** 2602.15473 | [PDF](https://arxiv.org/pdf/2602.15473v1)

**作者:** Jan Kobiolka `[一作]` (University of Technology Nuremberg), Josif Grabocka `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于强化学习的自适应优化器POP，学习在优化过程中根据历史轨迹动态预测每个坐标的步长；

**💡 创新点**

创新点主要包括①使用高斯过程与随机傅里叶特征构造的混合凸/非凸先验生成数百万合成函数进行元训练；②在Transformer骨干上实现坐标级步长策略；③引入边界缩放、Z变换和梯度缩放等输入预处理，提升跨函数的泛化；④通过坐标独立设计实现高维问题的无缝扩展；

**🔧 技术方法**

技术手段包括Transformer网络、PPO强化学习、Gaussian Process先验、Random Fourier Features、坐标级步长输出、边界缩放/梯度缩放/标准化等；

**📊 数据集**

使用的数据集包括：①基于GP先验生成的二维合成函数（训练约1.6M个不同函数，验证/测试各1024个）；②公开的Virtual Library of Simulation Experiments (VLSE)基准，共47个多维优化函数；

**📈 对比分析**

与传统梯度下降、Adam、L‑BFGS、随机搜索、遗传算法、差分进化、贝叶斯优化TPE以及学习型优化器VeLo等基线在相同迭代预算下进行对比，评估指标为归一化改进（NI）和归一化遗憾；结果表明POP在分布内测试、长时间预算、较高维度以及离散分布外的VLSE基准上均优于所有基线，且在多数任务中显著领先；

**⚠️ 局限性**

局限性包括：①训练仅在二维函数上进行，缺乏对极高维真实应用（如深度网络训练）的直接验证；②对合成先验的依赖可能限制在复杂实际损失景观上的适应性；③Transformer处理长序列的计算开销仍显著；④尚未系统评估在非光滑或约束优化场景下的表现。

---

## 59. Efficient Road Renovation Scheduling under Uncertainty using Lower Bound Pruning

**arXiv ID:** 2602.15554 | [PDF](https://arxiv.org/pdf/2602.15554v1)

**作者:** Robbert Bosch `[一作]` (University of Twente), Martijn Mes `[通讯]` (University of Twente)

**通讯引用:** 2112 | [OpenAlex ID](https://openalex.org/A5005906246)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种结合机器学习与遗传算法的渐进式下界剪枝方法，解决大规模不确定寿命道路网络维护调度问题。

**💡 创新点**

创新点在于将软性寿命截止与风险目标纳入双目标优化，提出Progressive Lower Bound Evaluation（PLBE）以在NSGA-II中逐步估计并剪枝，显著提升计算效率。

**🔧 技术方法**

使用NSGA-II、Frank‑Wolfe交通分配、XGBoost回归与CostliestSubsetHeuristic等机器学习预测器。

**📊 数据集**

使用Sioux Falls网络的76条项目实例，时间规划为80个季度，预算和施工容量约束。

**📈 对比分析**

与标准NSGA-II比较，PLBE在相同24小时内实现约40倍的迭代吞吐量，减少60–90%交通模拟次数，Pareto前沿规模增长2–3倍，在Hypervolume、最小距离等指标上显著提升。

**⚠️ 局限性**

主要局限在于对未来寿命信息的静态假设，未考虑时变需求或动态决策，且对极端不确定性建模的鲁棒性需进一步验证。

---

## 60. Recursive Concept Evolution for Compositional Reasoning in Large Language Models

**arXiv ID:** 2602.15725 | [PDF](https://arxiv.org/pdf/2602.15725v1)

**作者:** Sarim Chaudhry `[一作]` `[通讯]` (Purdue University), Sarim Chaudhry (Purdue University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在冻结的预训练语言模型上构建了递归概念演化框架，使模型在推理时能够动态生成、评估并合并低秩概念子空间，从而扩展内部表示空间。

**💡 创新点**

创新点在于引入运行时概念生成、MDL驱动的概念选择、协同合并以及晶化过程，实现可扩展的自适应表示演化。

**🔧 技术方法**

技术包括低秩投影子空间注入、稀疏顶k门控、基于熵的失效检测、MDL/最小描述长度筛选、QR正交化、SVD合并、KL约束和正则化。

**📊 数据集**

使用的评测数据集包括ARC‑AGI‑2、MATH、Big‑Bench Hard、GPQA 和 HLE，并在 Mistral‑7B、Llama‑3‑8B、Qwen‑2.5‑14B 上训练。

**📈 对比分析**

与链式推理、Self‑Consistency、Tree‑of‑Thought、GRPO、DisCO 等基线对比，RCE 在所有五个基准上平均提升 8–18 分，计算成本仅比基线多 4%。

**⚠️ 局限性**

局限性包括仅在单层注入限制深度推理、缺乏外部记忆支持、潜在对抗性误触发、合并计算量随概念数平方增长、以及对大规模模型的内存与计算扩展性挑战。

---

## 61. Relative Geometry of Neural Forecasters: Linking Accuracy and Alignment in Learned Latent Geometry

**arXiv ID:** 2602.15676 | [PDF](https://arxiv.org/pdf/2602.15676v1)

**作者:** Deniz Kucukahmetler `[一作]` (Max Planck Institute for Human Cognitive and Brain Sciences), Diaaeldin Taha `[通讯]` (Max Planck Institute for Mathematics in the Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过引入锚点基的相对嵌入方法，系统评估并对齐不同神经网络预测器（MLP、RNN、Transformer、ESN及其Koopman/NODE变体）在七类典型动力学系统上的内部表征几何。

**💡 创新点**

创新点在于：①使用相对嵌入消除旋转、尺度歧义，实现跨模型可比的几何空间；②在多种周期、准周期与混沌系统上揭示出家族级一致的表征对齐模式；③发现高预测精度与对齐度不一定同步，强调表征几何作为独立评估维度的重要性。

**🔧 技术方法**

采用Anchor-based Relative Embedding、余弦/秩/T1相似度、全连接、循环、Transformer、ESN架构；训练目标为MSE，利用Adam、早停；对齐度通过相对嵌入的余弦相似度计算；同时评估MSE、RMSE、MAE。

**📊 数据集**

数据集包括七个经典动力系统（Lorenz‑63、logistic map、Hopf、double pendulum、POD wake、random skew、limit cycle）以及一个iEEG实测序列，用于验证对齐趋势。

**📈 对比分析**

通过在共享锚点下计算余弦相似度实现跨模型对齐；对齐度与预测误差呈正相关，MLP与RNN族内对齐高且预测好；Transformer与ESN虽能取得较低误差，却对齐度相对较低；对齐可作为模型选择与可解释性的补充指标。

**⚠️ 局限性**

局限性包括：①相对嵌入依赖锚点数量与选择，可能在高维或数据稀疏时不稳定；②未能证明模型恢复物理动力学，仅提供几何相似性；③主要评估短期预测，对长期统计或控制性能的洞察有限；④未探索对齐引导的训练或跨模型迁移等后续应用。

---

## 62. How to Train Your Long-Context Visual Document Model

**arXiv ID:** 2602.15257 | [PDF](https://arxiv.org/pdf/2602.15257v1)

**作者:** Austin Veselka `[一作]` `[通讯]` (LightOn), Austin Veselka (LightOn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文提出了一套完整的训练和评估流程，利用连续预训练、监督微调、长上下文偏好优化以及自我提升技术，构建了可处理高达344K token长文本的视觉语言模型，并在MMLBD-C等长文档问答基准上实现了SOTA表现。

**💡 创新点**

创新点包括：①在视觉长文档场景下首次系统研究并公开可复现的训练 recipe；②发现训练与评测上下文长度一致可提升性能；③通过添加页码索引实现显著提升；④展示视觉长上下文训练向文本长上下文迁移的双向效应；⑤设计递归答案生成管线实现无教师模型的自我提升。

**🔧 技术方法**

使用的技术包括：大规模连续预训练（CPT）与监督微调（SFT），LongPO偏好优化，递归答案生成与知识蒸馏，模型融合（model merging），以及基于页面索引的输入增强。

**📊 数据集**

使用的数据集主要是从网络抓取并过滤得到的250K份PDF（16M页）与PDFA英语子集（2M份PDF、18M页），并通过合成数据管线生成多种长上下文视觉问答示例。

**📈 对比分析**

通过多维度评测指标（VA、LCA）与多基准（MMLongBenchDoc、MMLBD-C、MMLongBench、DUDE、SlideVQA、Helmet、LongBench v2）对比，SFT+CPT在VA上达到约94.4分，LongPO在VA上进一步提升至≈95分，均超过GPT4o、Claude等封闭模型。

**⚠️ 局限性**

局限性包括：评测覆盖的长文档长度主要在128K token以内，无法充分验证344K上下文的极端性能；CPT与SFT组合并不总是相加效果，缺乏对混合阶段训练的深入理解；并且自我提升仍需进一步探索更高效的递归生成策略。

---

## 63. CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing

**arXiv ID:** 2602.15823 | [PDF](https://arxiv.org/pdf/2602.15823v1)

**作者:** Zarif Ikram `[一作]` (University of Southern California), Paria Rashidinejad `[通讯]` (University of Southern California)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于低曲率子空间投影的模型编辑方法，能够在保持原有模型能力的同时高效更新少量参数。

**💡 创新点**

将编辑问题表述为二次约束优化，利用Bregman散度与Gauss‑Newton Hessian的低曲率空间实现可扩展的二阶编辑，从而突破传统只限单层或硬约束的限制。

**🔧 技术方法**

采用二阶约束优化、PGD、K‑FAC近似、矩阵无关低曲率投影以及Bregman距离等技术。

**📊 数据集**

在MNIST→FashionMNIST的小模型实验以及LLM编辑数据集ZsRE、CounterFact、WikiBigEdit上进行实验，并对MMLU、IFEval、TruthfulQA、ARC‑C、GSM8K等基准进行能力评估。

**📈 对比分析**

与MEMIT、AlphaEdit、Adam‑NSCL、LocBF‑FT、UltraEdit、MEND、FT、LoRA等基线相比，CRES在编辑可靠性与泛化性上显著提升，同时保持率超过70％，编辑耗时仅几分钟。

**⚠️ 局限性**

局限在于仍需构造有限规模的能力缓存，对能量阈值γ较为敏感；在极大规模编辑或持续学习场景中可能需要进一步优化收敛性与存储开销。

---

## 64. The Agentic Automation Canvas: a structured framework for agentic AI project design

**arXiv ID:** 2602.15090 | [PDF](https://arxiv.org/pdf/2602.15090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 65. Effective and Robust Multimodal Medical Image Analysis

**arXiv ID:** 2602.15346 | [PDF](https://arxiv.org/pdf/2602.15346v1)

**作者:** Joy Dhar `[一作]` (Indian Institute of Technology Ropar), Maryam Haghighat `[通讯]` (Queensland University of Technology)

**通讯引用:** 342 | [OpenAlex ID](https://openalex.org/A5003260186)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种多注意力融合学习框架（MAIL）及其鲁棒版本（Robust-MAIL），用于多模态医学影像的分类和分割任务。

**💡 创新点**

创新点在于：1）设计了轻量化的残差学习注意力块（ERLA）和高效的跨模态注意力模块（EMCAM），实现多尺度、跨频域的共享特征提取；2）采用并行融合注意力机制取代传统的串行注意力，减少信息丢失；3）引入随机投影滤波器与调制注意力噪声，构成随机投影+注意力噪声模块，显著提升对对抗攻击的鲁棒性。

**🔧 技术方法**

技术方法包括：轻量卷积+通道注意力、深度可分离卷积、频域离散余弦变换、并行跨模态注意力融合、随机投影滤波器、可学习注意力噪声、对抗训练（PGD、BIM、MIM等）。

**📊 数据集**

实验使用20个公开医学影像数据集，涵盖CT、MRI、X光、乳腺影像、皮肤图像、肺结核、脑肿瘤等多种模态。

**📈 对比分析**

与当前主流单模态和多模态模型相比，MAIL/Robust-MAIL 在分类/分割任务上平均提升 0.2%–13.9%（最高 9.34%）并将参数与 FLOPs 降低 54.9%–81.3%。在对抗攻击下，Robust-MAIL 在六个数据集上均优于其他防御方法，性能提升可达 65%。

**⚠️ 局限性**

局限性包括：仍需要进一步验证在更大规模、跨中心真实数据上的泛化能力；对极端高强度对抗攻击的抵抗力有限；虽然计算成本显著下降，但在极低资源环境下仍需进一步优化。

---

## 66. Beyond Static Pipelines: Learning Dynamic Workflows for Text-to-SQL

**arXiv ID:** 2602.15564 | [PDF](https://arxiv.org/pdf/2602.15564v1)

**作者:** Yihan Wang `[一作]` (Renmin University of China), Wei Xu `[通讯]` (Renmin University of China)

**通讯引用:** 16870 | [OpenAlex ID](https://openalex.org/A5112523938)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 SquRL 框架，通过强化学习动态构建 Text‑to‑SQL 工作流，以适应不同查询场景；

**💡 创新点**

创新点在于：①证明动态工作流在理论上优于任何静态工作流；②引入规则奖励、动态演员掩蔽和伪奖励等技术，使 LLM 在有限的执行反馈下学习鲁棒的工作流选择策略；

**🔧 技术方法**

技术包括：基于演员-模板-工作流的抽象；规则式多层奖励函数；动态演员掩蔽（随机抑制演员）；伪奖励（LLM 评判对比）和 Group Relative Policy Optimization (GRPO)；

**📊 数据集**

使用 Spider、BIRD、Spider 2.0、SynSQL 四大公开 Text‑to‑SQL 基准数据集进行实验；

**📈 对比分析**

与 DIN‑SQL、MAC‑SQL、LinkAlign、RSL‑SQL、CHESS 等静态工作流基线相比，SquRL 在所有数据集上均取得显著提升，尤其在复杂/长尾查询中提升 4–10%；

**⚠️ 局限性**

局限性包括：对后端执行模型的依赖（弱后端会影响奖励质量）；伪奖励引入噪声，过高比例会削弱性能；动态掩蔽需要手动调参，且在演员缺失时仍可能性能下降。

---

## 67. Ground-Truth Depth in Vision Language Models: Spatial Context Understanding in Conversational AI for XR-Robotic Support in Emergency First Response

**arXiv ID:** 2602.15237 | [PDF](https://arxiv.org/pdf/2602.15237v1)

**作者:** Rodrigo Gutierrez Maquilon `[一作]` (Austrian Institute of Technology), Manfred Tscheligi `[通讯]` (Austrian Institute of Technology)

**通讯引用:** 10295 | [OpenAlex ID](https://openalex.org/A5070249475)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在混合现实紧急响应模拟中，构建并评估了一个融合机器人深度摄像头、YOLOv8目标检测与 Qwen2.5vl 视觉语言模型的原型，支持机器人实时测距并通过语音输出精确的物体距离。

**💡 创新点**

创新点在于：①将机器人捕获的真实三维深度信息直接注入 VLM，生成基于度量的距离描述；②通过这种深度增强的交互，显著提升了受试者的空间理解与决策准确性，同时避免了额外的认知负荷；③提出了在 XR‑机器人 EFR 场景中将传感器数据与对话式 AI 有机结合的实用框架。

**🔧 技术方法**

技术栈包括：机器人配备的 ORBBEC Dabai DCW RGB‑D 摄像头、Jetson Nano 辅助计算；YOLOv8x 对象检测；Qwen2.5vl 视觉语言模型；ROS‑Unity 交互框架；Meta Quest Pro HMD 进行混合现实展示；Unity 3D 环境模拟有毒烟雾场景；以及 NASA‑TLX、SART、SASSI 等标准评估工具。

**📊 数据集**

数据集主要为实验室自建的混合现实场景：一间装有假人、桌椅的办公室，伴随模拟有毒烟雾；机器人实时提供 RGB、深度、IR 数据；YOLOv8 检测结果与对应深度值一起构成模型输入。并未使用公开标准数据集，而是基于现场采集的真实传感器数据与 Unity 生成的虚拟环境。

**📈 对比分析**

比较方法：对 16 名受试者在三种条件（视频仅观看、无深度 VLM、深度增强 VLM）下进行距离估计任务，并通过 NASA‑TLX、SART、SASSI、UMUX、信心评分等量表收集主观评价。实验结果显示：深度增强 VLM 将距离误差从 0.58 m 降至 0.25 m（人）和从 0.11 m 降至 0.08 m（窗户），SART 评分提升 27% 而工作负荷保持不变；无深度 VLM 则导致工作负荷上升、误差略增。整体而言，深度增强 VLM 在准确性、主观满意度与认知负荷方面均优于其他两种方案。

**⚠️ 局限性**

局限性包括：①样本量有限且实验环境为模拟，缺乏真实高压现场动态；②未包含移动目标、遮挡或突发危险等真实复杂情境；③VLM 响应延迟（约 30 s）对实时决策有影响；④缺乏长期使用者信任与依赖的纵向评估；⑤仅针对单一任务（距离估计），未验证在更广泛 EFR 决策中的适用性。

---

## 68. Natural Privacy Filters Are Not Always Free: A Characterization of Free Natural Filters

**arXiv ID:** 2602.15815 | [PDF](https://arxiv.org/pdf/2602.15815v1)

**作者:** Matthew Regehr `[一作]` (University of Waterloo), Mathias Lécuyer `[通讯]` (University of British Columbia)

**通讯引用:** 394 | [OpenAlex ID](https://openalex.org/A5050942051)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了自然隐私过滤器（natural privacy filters），并证明在一般情况下它们在适应性组合（adaptive composition）时并非“免费”（free），即使用自然过滤器会产生额外的隐私损失。作者进一步给出了必要且充分的条件：只有在查询集合在复合下形成良序（well‑ordered）时，天然过滤器才是免费。该工作扩展了先前关于隐私过滤器的理论，明确了自然过滤器的可行性边界。

**💡 创新点**

创新点：
- 提出了自然过滤器的完整理论框架，并给出“自由”与否的判定条件；
- 证明了自然过滤器在普通（ε,δ）-DP、PLD、f‑DP 等定义下普遍不免费，违背以往仅在 RDP、zCDP 等情形下可免费结论；
- 引入并利用“黑塞尔（Blackwell）顺序”、隐私损失分布（PLD）与hockey‑stick 曲线的几何/拓扑性质，构造反例并证明通用性；
- 证明了“完备性”与“卷积-上确界”交换律相互对应，并用其表征自由过滤器的存在。

**🔧 技术方法**

主要技术：
- 隐私损失分布（PLD）与黑塞尔顺序的定义与性质；
- 通过 PLD 的卷积与上确界（sup）来描述适应性组合与过滤器的交互；
- 采用hockey‑stick 曲线的凸性、单调性以及其与 PLD 的互补关系；
- 通过几何/拓扑论证（如多重连通、极限点、可测集合分割）来证明非自由性；
- 对多段线性 trade‑off 曲线（f‑DP）做精确组合，构造跨越点的反例；
- 利用复合下的良序（well‑ordered）性质，证明上确界与卷积交换的必要性与充分性。

**📊 数据集**

无实验数据，全部为理论证明与数学构造。

**📈 对比分析**

由于本文为纯理论工作，没有实验对比，性能评价以“是否存在免费过滤器”为衡量标准。对于满足良序条件的机制族（如高斯机制、纯DP等），证明能实现无额外隐私损失的自然过滤器；而对一般机制族则给出严格的不可免费结论。

**⚠️ 局限性**

局限性：
- 只针对非退化 PLD（至少有两个点的支撑）讨论；
- 对于某些特殊机制族（如不满足良序或不闭合的情况），结果不适用；
- 论文未给出具体实现或算法实现细节，主要集中在理论可行性；
- 仍未解决在更广泛的 f‑DP 或 PLD 过滤器设计中实现“免费”与否的实际实现问题。

---

## 69. FedPSA: Modeling Behavioral Staleness in Asynchronous Federated Learning

**arXiv ID:** 2602.15337 | [PDF](https://arxiv.org/pdf/2602.15337v1)

**作者:** Chaoyi Lu `[一作]` (Xi'an Jiaotong University), Chaoyi Lu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 51 | [OpenAlex ID](https://openalex.org/A5077212970)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedPSA，一种基于参数敏感度的异步联邦学习框架，用行为信息来衡量模型陈旧并动态调节聚合权重，提升异步学习的最终准确率和鲁棒性。

**💡 创新点**

① 引入参数敏感度作为行为陈旧度量，利用敏感度相似度代替单纯的轮数差；② 通过动量队列构造训练温度，依据当前训练阶段动态调节权重分布；③ 使用随机投影压缩敏感度信息，保持通信与计算成本可接受。

**🔧 技术方法**

参数敏感度计算（二阶泰勒+Fisher近似）、随机投影 Sketch、动量队列/温度机制、异步缓冲式聚合、软max加权。

**📊 数据集**

MNIST、FMNIST、CIFAR‑10、CIFAR‑100，采用 IID 与非 IID（Dirichlet α=0.1、0.5、1.0）分区。

**📈 对比分析**

与 FedAvg、FedAsync、FedBuff、CA2FL、FedFa、FedPAC 六个基线在上述四个数据集上进行对比。FedPSA 在所有设置下均优于基线，最终准确率提升最高 6.37%，AULC 最高，且在系统异质性下表现更为稳健。

**⚠️ 局限性**

对公共校准批次的依赖（虽然可用噪声替代但仍需共享）；对超参数（γ、δ、队列长度）敏感；实验未涉及更大模型或多任务场景，且缺乏严格的理论收敛证明。

---

## 70. Exploiting Layer-Specific Vulnerabilities to Backdoor Attack in Federated Learning

**arXiv ID:** 2602.15161 | [PDF](https://arxiv.org/pdf/2602.15161v1)

**作者:** Mohammad Hadi Foroughi `[一作]` (University of Tehran), Ahmad Khonsari `[通讯]` (University of Tehran)

**通讯引用:** 2057 | [OpenAlex ID](https://openalex.org/A5091782280)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于层级特异性脆弱性的Federated Learning后门攻击方法 Layer Smoothing Attack（LSA），通过 Layer Substitution Analysis 定位后门关键层并进行精细化修改；

**💡 创新点**

创新点在于系统性识别后门关键层、利用层级信息进行精确攻击，并通过平滑技术使恶意更新与正常更新在统计上几乎不可区分，从而突破现有 FL 防御；

**🔧 技术方法**

主要技术包括：联邦学习框架、层级替换与微调、后门成功率（BSR）评估、ReLU 限制、统计近似（逼近正常更新）以及 λ 控制的攻击强度；

**📊 数据集**

实验使用 CIFAR-10、Fashion‑MNIST 两大公开数据集，并在三种网络结构（5‑layer CNN、ResNet‑18、VGG‑19）上验证；

**📈 对比分析**

与传统攻击（BadNets、DBA、LPA）以及多种防御（FedAVG、Trimmed‑Mean、Multi‑Krum、FLTrust、FLAME）对比，LSA 在 FLAME 防御下的 BSR 可达 96.98% 以上，且主任务准确率仅略降；

**⚠️ 局限性**

局限性包括：需提前识别后门关键层且依赖多轮通信，攻击对模型结构和数据分布具有一定依赖；未来研究需针对不同网络架构和更强鲁棒防御进行评估。

---

## 71. A Universal Neural Receiver that Learns at the Speed of Wireless

**arXiv ID:** 2602.15458 | [PDF](https://arxiv.org/pdf/2602.15458v1)

**作者:** Lingjia Liu `[一作]` (Virginia Tech), Robert Calderbank `[通讯]` (Duke University)

**通讯引用:** 35482 | [OpenAlex ID](https://openalex.org/A5111958277)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e`

**🎯 论文内容**

提出一种基于卷积的通用神经接收机，利用循环神经网络实现去卷积，并在每个传输时间间隔内在线学习

**💡 创新点**

将卷积逆变问题拆分为“逆哪种卷积”与“执行去卷积”，通过预先配置权重利用领域知识，显著降低训练需求，实现无线速度学习

**🔧 技术方法**

使用循环神经网络（reservoir computing）、线性与非线性处理、3GPP CDL‑B 通道协方差、2D LMMSE 插值与 ILSMR 迭代最小残差方法

**📊 数据集**

基于3GPP非视距聚类时延线性通道模型‑B（CDL‑B）模拟数据，载频3.8 GHz、256子载波、30 kHz子载波间距，OFDM用户速度30 km/h，OTFS用户速度450 km/h

**📈 对比分析**

与传统 LMMSE（MIMO‑OFDM）和 LMMSE+ILSMR（MIMO‑OTFS）在 BER‑SNR 曲线上比较，神经接收机在16‑QAM、64‑QAM OFDM 及 OTFS 上均超过2 dB，且计算复杂度更低

**⚠️ 局限性**

依赖于预设的域知识与通道协方差，若通道统计不匹配或出现严重非线性、极端干扰时性能可能下降；受限于 OTA 训练样本数量，且在不同波形下需要重新配置权重

---

## 72. From Earthquake Solidarity to Educational Equity: Conceptualizing a Sustainable, Volunteer-Driven P2P Learning Ecosystem at Scale

**arXiv ID:** 2602.15432 | [PDF](https://arxiv.org/pdf/2602.15432v1)

**作者:** Öykü Kaplan `[一作]` (Gdańsk University of Technology), Netta Iivari `[通讯]` (University of Oulu)

**通讯引用:** 4607 | [OpenAlex ID](https://openalex.org/A5038286899)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种从地震灾后响应演变为可持续的志愿者驱动的同伴学习生态系统，并在两年内支持300多名学生。

**💡 创新点**

首次在灾后全在线情境中系统性探讨非互惠远程同伴辅导的可持续性，提出五条设计原则和代际互惠循环，强调微付费与混合激励机制。

**🔧 技术方法**

利用Zoom等在线会议技术、数据记录表格、主题分析与人工智能辅助评估等方法，并计划开发专属P2P平台。

**📊 数据集**

基于实地观察、焦点小组、问卷调查和愿景工作坊收集的定性与定量数据，未使用传统公开数据集。

**📈 对比分析**

通过案例研究与三角验证，比较学业自信、表达能力和学习成绩的提升，结果显示学生自信和学业成效显著提高；未做跨系统性能基准。

**⚠️ 局限性**

研究样本局限于单一灾区，缺乏跨文化验证；志愿者可用性和平台技术、资金可持续性仍待进一步验证。

---

## 73. Fractional-Order Federated Learning

**arXiv ID:** 2602.15380 | [PDF](https://arxiv.org/pdf/2602.15380v1)

**作者:** Mohammad Partohaghighi `[一作]` (University of California Merced), YangQuan Chen `[通讯]` (University of California Merced)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在联邦学习中使用分数阶梯度下降（FOSGD）的新的联邦平均算法FOFedAvg，以提升在非IID数据下的收敛速度和通信效率。

**💡 创新点**

创新点在于将分数阶微积分的记忆效应引入联邦学习，通过Caputo型分数阶导数实现对历史梯度的权重衰减，从而降低局部漂移、缓解异构数据带来的不稳定性。

**🔧 技术方法**

核心技术包括：分数阶随机梯度下降（FOSGD）、FedAvg框架、Caputo分数阶导数、γ函数归一化、正则化常数δ、学习率调度μ_t=μ_0/√(t+1)。

**📊 数据集**

实验使用了 MNIST、EMNIST、FEMNIST、CIFAR‑10、CIFAR‑100、Cleveland心脏病、Sent140、PneumoniaMNIST、Edge‑IIoTset 等九个基准数据集，并采用多种非IID划分方案。

**📈 对比分析**

与 FedAvg、FedProx、SCAFFOLD、MOON、FedNova、FedAdam、FDSE、FedBM、DPSGD、SparseSecAgg 等十种主流联邦学习算法对比，FOFedAvg 在大多数数据集上实现了更快收敛、更高最终准确率（比 FedAvg 提升 5–10%，通信轮数减少 10% 以上），尤其在极端非IID场景下表现优异。

**⚠️ 局限性**

局限性包括：对分数阶参数 α 与正则化 δ 的敏感性，需要手动调优；在数据极端稀疏或分布快速变化时，长记忆可能导致梯度过度依赖旧信息，减慢适应速度；实现复杂度略高，需额外计算参数差异。

---

## 74. PolyNODE: Variable-dimension Neural ODEs on M-polyfolds

**arXiv ID:** 2602.15128 | [PDF](https://arxiv.org/pdf/2602.15128v1)

**作者:** Per Åhag `[一作]` (Umeå University), Viktor Vigren Näslund `[通讯]` (Umeå University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了PolyNODE，将Neural Ordinary Differential Equations扩展到M-polyfolds，实现了连续深度模型在可变维度上的演化，并在自编码器结构中实验验证其可训练性

**💡 创新点**

创新点在于使用M-polyfold理论为分层空间提供全局可微结构，使得ODE流能够在维度跳变处连续且可微，从而首次实现可变维度的流式深度学习模型

**🔧 技术方法**

采用M-polyfold框架、尺度Banach空间、可微向量场及半流理论，并利用自编码器中的压缩向量场进行参数化学习

**📊 数据集**

在实验中使用了几何对象数据集：多旋转螺旋（N=0.5–5）和二维圆球，分别嵌入到Ω^m_n M-polyfolds中进行训练

**📈 对比分析**

通过自编码器重构误差、循环误差及分类精度评估，实验表明对螺旋可实现100%径向分类、98%角度分类；重构误差极低，表明模型表现良好；但未与传统NODE或其他深度网络做定量对比

**⚠️ 局限性**

局限在于缺乏完整的M-polyfold ODE理论基础、仅在小规模几何示例上验证、未提供与现有模型的系统比较，且对更大规模、复杂数据的泛化性未知

---

## 75. How to Detect Information Voids Using Longitudinal Data from Social Media and Web Searches

**arXiv ID:** 2602.15476 | [PDF](https://arxiv.org/pdf/2602.15476v1)

**作者:** Irene Scalco `[一作]` (Sapienza University of Rome), Matteo Cinelli `[通讯]` (Sapienza University of Rome)

**通讯引用:** 4262 | [OpenAlex ID](https://openalex.org/A5076143079)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套基于信息需求与供应不平衡的时序模型，用来定量检测信息空白期（信息void）和信息过剩期。

**💡 创新点**

创新点在于将供需比的差值（delta）与异常检测相结合，构建五状态分类（Void、Lack、Balance、Abundance、Overabundance），并首次在疫情期间验证信息空白与谣言扩散的关联。

**🔧 技术方法**

技术主要包括时间序列归一化、STL分解、基于IQR的异常检测、Delta指标计算以及多源数据的跨平台同步处理。

**📊 数据集**

使用了社交媒体（Facebook、Twitter）、全球新闻数据库GDELT、维基百科页面浏览量和Google Trends等多源数据集，覆盖2020-2021年欧洲六国COVID-19疫苗讨论。

**📈 对比分析**

通过合成数据验证，该方法在6σ以上的异常下精准率>90%，F1>0.68；在实证案例中成功捕捉疫苗发布导致的空白期，并与NewsGuard可信度评分关联，说明模型能有效识别高风险信息期。

**⚠️ 局限性**

局限性包括样本偏向社交媒体用户、未覆盖传统媒体与其他在线平台、需求代理可能缺乏完整性，以及模型对极端事件的泛化仍需进一步验证。

---

## 76. Controlled oscillation modeling using port-Hamiltonian neural networks

**arXiv ID:** 2602.15704 | [PDF](https://arxiv.org/pdf/2602.15704v1)

**作者:** Maximino Linares `[一作]` (IRCAM), Thomas Hélie `[通讯]` (IRCAM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出将能量保持的离散梯度方法嵌入端口哈密顿神经网络（PHNN）中，用于学习受控振荡系统的动力学，并系统评估不同PHNN架构、离散方法与Jacobian正则化对模型性能的影响。

**💡 创新点**

创新点在于：①首次将二阶离散梯度数值方法与PHNN结合，显著提升能量保持与预测精度；②对等价的PH-DAE与输入状态输出PH（PHNN-S与PHNN-JR）两种端口哈密顿形式做细致对比；③引入基于条件数、谱范数及刚性比的Jacobian正则化来抑制学习过程中的数值不稳定与刚性。

**🔧 技术方法**

使用技术包括端口哈密顿系统理论、离散梯度（Gonzalez）数值积分、第二阶显式Runge‑Kutta、神经ODE训练框架、Jacobian正则化（谱范数、条件数、刚性比）以及对比实验的自动微分与训练流程。

**📊 数据集**

数据集为12500条合成振荡轨迹，涵盖三种受控振荡器（谐振子、Duffing振荡器、自维持振荡器），每条轨迹由Gonzalez离散梯度生成，并在不同采样率与训练点数量下抽样用于训练与测试。

**📈 对比分析**

比较方法：在相同训练点、模型参数规模下，对PHNN-S与PHNN-JR、RK2与DG离散方法、以及不同Jacobian正则化进行10次随机初始化实验，评估预测轨迹均方误差。结果显示：DG方法在非线性系统与数据稀缺场景下显著优于RK2；PHNN-S在线性与Duffing振荡器表现最佳，PHNN-JR在自维持振荡器上更优；Jacobian正则化（尤其是条件数正则化）能降低Duffing振荡器的误差分散。

**⚠️ 局限性**

局限性包括：需预先给定耦合矩阵S/J；仅考虑形如H(x)=½xᵀQ(x)x的哈密顿量；实验仅使用二阶方法，未检验更高阶离散梯度或RK；Jacobian正则化系数未进行系统调优；并未探索在真实测量噪声或更复杂动力学下的鲁棒性。

---

## 77. Far Out: Evaluating Language Models on Slang in Australian and Indian English

**arXiv ID:** 2602.15373 | [PDF](https://arxiv.org/pdf/2602.15373v1)

**作者:** Deniz Kaya Dilsiz `[一作]` (University of New South Wales), Aditya Joshi `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估大语言模型对印度英语(en-IN)与澳大利亚英语(en-AU)地区俚语的理解能力；

**💡 创新点**

首次构造两套俚语数据集（web+gen）并在目标词预测与选择任务上评估多种LLM，揭示模型在生成俚语方面的显著不足与在多选判别上的优势；

**🔧 技术方法**

采用掩码填充与多选填空的评估框架，对七种Encoder/Decoder LLM（BERT、RoBERTa、XLM‑RoBERTa、Granite、Llama、Olmo、Qwen）进行评测；

**📊 数据集**

使用Urban Dictionary来源的377条真实俚语实例（web）与1,492条Gemini‑生成的多样化场景（gen）共涵盖377条俚语短语；

**📈 对比分析**

通过平均准确率与相似度对比，发现目标词预测（TWP/TWP*）平均准确率仅0.02–0.04，生成难度大；但目标词选择（TWS）平均准确率可达0.49（web）与0.49（gen），显示判别性能明显更好；域与方言差异亦明显，en‑IN优于en‑AU；

**⚠️ 局限性**

局限在于数据量有限、仅覆盖两种英语方言、web集成可能存在数据污染、仅单一专家校验导致缺乏多样性和可信度。

---

## 78. FAST-EQA: Efficient Embodied Question Answering with Global and Local Region Relevancy

**arXiv ID:** 2602.15813 | [PDF](https://arxiv.org/pdf/2602.15813v1)

**作者:** Haochen Zhang `[一作]` (Carnegie Mellon University), Enna Sachdeva `[通讯]` (Honda Research Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FAST-EQA，一个结合语义引导的全局与局部探索、有限记忆的嵌入式问答框架。

**💡 创新点**

通过门道前沿探索、目标感知记忆和链式思考推理，实现了在多目标、长时限任务下的高效、可扩展的问答。

**🔧 技术方法**

使用 LLM（GPT‑4o/Prismatic‑VLM）提取目标与房间，CLIP 与 VLM 结合的观测相关性评分，DBSCAN 门道检测，链式思考提示，以及固定容量视觉记忆。

**📊 数据集**

在 HM‑EQA、EXPRESS‑Bench、OpenEQA、MT‑HM3D 四个基准上进行评测。

**📈 对比分析**

与 Fine‑EQA、Explore‑EQA、Graph‑EQA、Memory‑EQA 等现有方法对比，HM‑EQA 最高准确率 76%（提升 9%）、EXPRESS‑Bench 68.7%（提升 7%），并在每步推理时间上比最优方法快 13.6%。

**⚠️ 局限性**

受限于 VLM 的空间推理能力和答案不一致性，且记忆仅存图像，缺乏更高层次的结构化表示。

---

## 79. Intracoronary Optical Coherence Tomography Image Processing and Vessel Classification Using Machine Learning

**arXiv ID:** 2602.15579 | [PDF](https://arxiv.org/pdf/2602.15579v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 80. Joint Enhancement and Classification using Coupled Diffusion Models of Signals and Logits

**arXiv ID:** 2602.15405 | [PDF](https://arxiv.org/pdf/2602.15405v1)

**作者:** Gilad Nurko `[一作]` (Technion), Joseph Keshet `[通讯]` (Technion)

**通讯引用:** 4266 | [OpenAlex ID](https://openalex.org/A5008847407)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种基于耦合扩散模型的鲁棒分类框架，联合对输入信号与分类器输出的logit进行去噪，使得信号增强与语义预测相互引导。

**💡 创新点**

核心创新在于：①不需要对预训练分类器进行微调；②设计了并行、交替、嵌套三种耦合策略；③在扩散过程中让信号和logit互相作为条件，提升语义一致性和分类准确率。

**🔧 技术方法**

主要技术包括扩散概率模型（DDPM、DDIM、SDE）、自回归logit扩散、互相指导的耦合更新、梯度对齐训练以及多阶段采样调度。

**📊 数据集**

在图像端使用MNIST、CIFAR‑10、CIFAR‑100和ImageNet32‑100四个数据集；在语音端使用Google Speech Commands、EARS（含Reverb和WHAM）等。

**📈 对比分析**

与传统噪声下直接分类（Noisy）、先增强再分类（Enhanced）以及CARD（只对logit去噪）做对比。实验表明，三种耦合策略均显著提升分类准确率和语音识别WER，Parallel在复杂数据上效率最高，Nested在语音上得到最低WER。

**⚠️ 局限性**

局限性包括：①耦合策略需要额外的扩散采样，计算开销大；②仅适用于冻结的分类器，对大规模预训练模型的迁移仍需探索；③在极端噪声或长序列下，梯度噪声和采样步骤数的选择对性能影响较大。

---

## 81. ScrapeGraphAI-100k: A Large-Scale Dataset for LLM-Based Web Information Extraction

**arXiv ID:** 2602.15189 | [PDF](https://arxiv.org/pdf/2602.15189v1)

**作者:** William Brach `[一作]` (Slovak University of Technology), Lorenzo Padoan `[通讯]` (ScrapeGraphAI)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并公开了 ScrapeGraphAI‑100k 数据集，包含 93,695 例真实的网页内容、自然语言提示、JSON 结构与对应 LLM 输出，并对数据进行去重、平衡和结构复杂度标注。

**💡 创新点**

首次提供大规模、真实生产环境下的 LLM 结构化抽取数据，结合深度结构指标（深度、键数、环形复杂度）对抽取失败进行系统化分析，并展示小型模型通过该数据集微调可逼近 30B 大模型的性能。

**🔧 技术方法**

使用 PostHog 采集遥测、JSON Schema 校验、SLOT 复杂度度量、QLoRA+LoRA 4‑bit 微调技术以及一系列结构与内容评估指标（valid JSON、schema compliance、key F1、value score、BLEU）。

**📊 数据集**

数据集本身（ScrapeGraphAI‑100k），并对比基线 Qwen3 1.7B/4B/30B 进行实验，评估微调后 1.7B 模型的表现。

**📈 对比分析**

采用对比实验：对同一评测集用 1.7B/4B/30B 原始模型与微调 1.7B 进行结构有效率、键精确率/召回率、value score、整体 BLEU 等多维度衡量，微调 1.7B 在 key F1 上与 30B 基线相差 0.005，整体 BLEU 仅落后 3%，显著优于原始 1.7B。

**⚠️ 局限性**

局限性包括：数据来源高度偏向 GPT‑4o‑mini，导致模型行为偏差；域与地区分布不均（电商占比高）；仅包含 Markdown 形式的网页内容，缺乏原始 HTML/DOM；仅评估语法与结构有效性，语义正确性测评不足；值提取准确率仍低，表明结构-语义鸿沟尚未完全弥合。

---

## 82. Interbank Lending Games

**arXiv ID:** 2602.15186 | [PDF](https://arxiv.org/pdf/2602.15186v1)

**作者:** Jinyun Tong `[一作]` (King's College London), Carmine Ventre `[通讯]` (King's College London)

**通讯引用:** 1289 | [OpenAlex ID](https://openalex.org/A5001328914)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文构建了一个“银行借贷博弈”，以游戏理论分析银行在超额流动性和需求之间的资金配置，证明该博弈为精确势游戏，存在唯一的纯纳什均衡，并给出了强多项式时间算法求解均衡以及对多种最优响应动态的收敛性分析。

**💡 创新点**

创新点在于：①首次把银行间借贷市场建模为连续策略空间的精确势博弈；②证明该博弈在任何初始状态下都收敛到唯一均衡；③提出了高效的计算均衡的算法，并给出了关于异步、同步、随机以及连续时间最优响应动态的完整收敛证明；④从经济角度指出均衡时所有借款银行的利率均相同，体现了信息充分竞争的结果。

**🔧 技术方法**

使用的技术主要是：游戏理论（精确势游戏、纳什均衡）、凸优化（求解严格凹势函数的最大化、KKT 条件）、算法设计与分析（强多项式时间算法）、收敛性分析（离散与连续时间最优响应动力学、Lyapunov 方法）以及数值实验验证（若有的话）。

**📊 数据集**

本文为理论性工作，没有使用实测金融数据；若做实验，可能采用公开的银行资产负债表或利率基准数据来验证模型假设，但文中并未给出具体数据集。

**📈 对比分析**

对算法的性能进行了理论分析，证明其时间复杂度为 O(mn + m log m)（m 为贷款方数量，n 为借款方数量）。在动态收敛性方面，作者分别给出了异步最优响应、随机最优响应、同步伪梯度以及连续时间最优响应的收敛证明，表明无论起始点如何，均能收敛到全局最优均衡；实验结果（若有）显示收敛速度快且与传统离散最优响应相比具有更好的稳定性。

**⚠️ 局限性**

局限性包括：①模型假设利率函数为线性且仅受供需影响，未考虑信用风险、违约概率等金融网络特有因素；②对银行间借贷规模做了理想化假设（连续资金、无交易成本）；③收敛证明依赖势函数严格凹性，若系统出现非凹性或非连续情况，结果可能不适用；④实验验证不足，缺乏对真实金融市场的实证检验。

---

## 83. AgriWorld:A World Tools Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing LLM Agents

**arXiv ID:** 2602.15325 | [PDF](https://arxiv.org/pdf/2602.15325v1)

**作者:** Zhixing Zhang `[一作]` (Sun Yat-sen University), Keze Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2281 | [OpenAlex ID](https://openalex.org/A5088124671)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一个基于执行环境AgriWorld和反思式LLM代理Agro-Reflective的农业科学助手框架，能够在高维时空农业数据上执行代码并自我纠错；

**💡 创新点**

创新点在于：①统一的可执行农业环境AgriWorld；②执行驱动的反思循环（Execute–Observe–Refine）让LLM利用中间结果自校正；③可验证的协议与可执行检查器，使得答案可被程序化检验；

**🔧 技术方法**

使用技术包括：Python执行环境、统一的GIS/遥感/模拟工具API、LLM代码生成与执行、反思回调函数、LoRA微调的Qwen3 LLM、可执行检查器和指标；

**📊 数据集**

使用数据集为AgroBench（由真实农业工作流生成的QA实例），内部涵盖遥感时序、土壤网格、管理日志、天气流等；

**📈 对比分析**

与文本仅回答、单次工具调用和大型开源/专有LLM做对比，实验表明Agro-Reflective在Lookup、Forecasting、Anomaly Detection、Counterfactual分析中分别提升了大约4.2%、47%、36.5%和27.6%（整体准确率提升至约73%），并在OOD场景中显著优于纯文本模型；

**⚠️ 局限性**

局限性包括：依赖于预先定义的工具接口和规范，对非结构化或缺失数据处理仍有限；执行循环仍可能受限于模型生成代码的复杂度；在极端时空情景下仍需更多数据与模型适配。

---

## 84. Under-resourced studies of under-resourced languages: lemmatization and POS-tagging with LLM annotators for historical Armenian, Georgian, Greek and Syriac

**arXiv ID:** 2602.15753 | [PDF](https://arxiv.org/pdf/2602.15753v1)

**作者:** Chahan Vidal-Gorène `[一作]` (École nationale des chartes), Florian Cafiero `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探讨在少量或无监督情境下，GPT‑4 及 Mistral 等大型语言模型（LLM）对古希腊语、古典亚美尼语、古格鲁吉亚语和叙利亚语的词形还原与词性标注任务的能力。

**💡 创新点**

创新点在于：① 将 GREgORI 细粒度词形与词性标注方案融入 LLM 的提示工程；② 通过系统的零样本与少样本评测，展示 LLM 在极低资源语言中的竞争性；③ 对比传统 RNN 基线 PIE，揭示 LLM 在复杂形态学与多词形态分割上的优势。

**🔧 技术方法**

使用技术包括：COSTAR 结构化提示、标注词表注入、分割指导、温度采样/贪婪解码，以及基准 RNN（PIE）与多模态 Transformer（mDeBERTa）等对照模型。

**📊 数据集**

数据集为 4 种历史语言的对齐训练/外域测试语料，训练集约 5,000 词，测试集 300 词，覆盖古典文献的多体裁与年代，采用 GREgORI 注释标准。

**📈 对比分析**

比较方法为在不同 shot 数（0、5、50、500）下的 in‑domain 与 out‑of‑domain 精度评估；LLM 在大多数语言的词形还原与词性标注上均超过 PIE，GPT‑4o 在零/少样本情境下取得 0.92 以上的词形准确率，mistral‑large 在部分语言的零样本亦可接近 GPT‑4o 级别。

**⚠️ 局限性**

局限性包括：① 语料量小且体裁单一，难以泛化；② GREgORI 的非标准词形分割导致结构错误，影响原始精度；③ 对极高形态复杂性与多词形态（尤其叙利亚语）仍存在显著误差；④ 评估依赖已标注数据，无法直接迁移至完全无标注语言。

---

## 85. Task-Agnostic Continual Learning for Chest Radiograph Classification

**arXiv ID:** 2602.15811 | [PDF](https://arxiv.org/pdf/2602.15811v1)

**作者:** Muthu Subash Kavitha `[一作]` (University of Texas MD Anderson Cancer Center), Jia Wu `[通讯]` (University of Texas MD Anderson Cancer Center)

**通讯引用:** 12193 | [OpenAlex ID](https://openalex.org/A5007475662)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计了一种面向胸部X光片持续学习的适配器路由框架CARL‑XRay，能够在顺序接收多源数据时无须重新训练旧数据，且在无任务标识的推理环境下实现任务识别与分类。

**💡 创新点**

首次在胸片分类中引入任务增量持续学习范式；使用冻结的高容量Swin Transformer骨干，增量添加轻量级适配器与分类头；通过潜在任务选择器结合任务原型和特征级经验回放实现任务未知推理，避免了原始图像存储与模型遗忘。

**🔧 技术方法**

核心技术包括：冻结Swin Transformer编码器、任务特定适配器（Simple、Continuum、Hope）与分类头、潜在任务选择器+任务原型、特征级经验回放、掩码多标签交叉熵损失与正交性正则。

**📊 数据集**

使用了公开的MIMIC‑CXR和CheXpert两大胸片数据集，分别对应任务一与任务二，涵盖14项临床标签。

**📈 对比分析**

与联合训练基线对比，评估AUROC、宏F1、遗忘度与任务路由准确率；CARL‑XRay在任务未知推理下路由准确率提升至75%（联合训练仅62.5%），AUROC保持0.75，且仅新增2.3 MB可训练参数，显著降低了训练成本。

**⚠️ 局限性**

局限性包括：实验仅涉及两任务且顺序敏感；对更长任务序列、不同机构或多模态数据的适用性未验证；适配器设计对性能影响显著，需要进一步优化；特征级经验回放容量与分布漂移仍可能影响长期稳定性。

---

## 86. Human-AI Interaction: Evaluating LLM Reasoning on Digital Logic Circuit included Graph Problems, in terms of creativity in design and analysis

**arXiv ID:** 2602.15336 | [PDF](https://arxiv.org/pdf/2602.15336v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 87. Local Node Differential Privacy

**arXiv ID:** 2602.15802 | [PDF](https://arxiv.org/pdf/2602.15802v1)

**作者:** Sofya Raskhodnikova `[一作]` (Boston University), Anatoly Zavyalov `[通讯]` (Boston University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在分布式图数据环境中引入了本地节点差分隐私（Local Node Differential Privacy, LNDP）模型，并提出了一套完整的算法框架，用于在该模型下对图的线性查询（如边数、度分布、Erdős–Rényi 参数、团大小等）进行近似估计；同时给出了针对这些任务的理论下界，证明在许多情况下所提出的算法是（几乎）最优的；此外还揭示了 LNDP 与传统本地差分隐私（LDP）以及中心节点差分隐私（central node DP）之间的本质差异。

**💡 创新点**

1) 通过引入“模糊度分布（blurry degree distribution）”降低度分布的灵敏度，获得可在 LNDP 下高精度回答任意线性查询的通用框架；2) 在此框架下设计了软阈值查询、加高斯噪声等技术，实现在稀疏图上边计数误差与中心模型相当；3) 开发了针对 LNDP 的新下界技术（splicing、Bhattacharyya 距离等），证明即使允许有限轮交互，算法仍然保持最优；4) 在结构性方面证明了 LNDP 下纯隐私的高级群组隐私性质、纯与近似隐私的本质差异以及“仅看度数”算法与全视图算法的层级关系。

**🔧 技术方法**

主要技术包括：
- 随机取整生成模糊度分布并证明其在 Wasserstein-∞ 距离上的误差小于参数 s；
- 利用软阈值函数构造 1/s‑Lipschitz 查询，改进感知灵敏度从 ℓ₁ 到 ℓ₂；
- 采用 Gaussian 机制和因子化（factorization）机制对线性查询进行私有化；
- 引入新的下界工具：Bhattacharyya 距离与 TV 距离的张量化性质、splicing 论证、组级隐私分析；
- 对交互式 LNDP 通过多轮模拟与数据处理不等式扩展下界。

**📊 数据集**

论文主要为理论性工作，实验使用的图模型为：
- 随机 d‑regular 图 G(d)；
- Erdős–Rényi 随机图 G(n,p)；
- 由大小为 Θ(n) 的团与孤立点组成的图；
- 其他如 D‑bounded 图等。

**📈 对比分析**

在相同准确性承诺下：
- 对 D‑bounded 图的边数估计误差为 O(D√n + n)，在稀疏 regime D = O(√n) 与下界 Ω(n) 完全匹配；
- 对 G(n,p) 的参数 p 估计误差为 O(√log n / n)，与下界 Ω(1/n) 只相差 √log n 阶；
- 对团大小估计误差为 O(√log(1/δ))，与中心模型下的最优误差相同（除 √log(1/δ) 因子外）。
- 通过与传统 LDP 的对比，LNDP 在纯隐私下实现了高级群组隐私（误差随 k 以 √k 递增），而近似 LNDP 无法获得此性质，展示了两者的根本区别。

**⚠️ 局限性**

主要局限包括：
- 研究集中在非交互式 LNDP，尽管已证明常轮交互不改善下界，但更复杂交互模型仍未深入；
- 对度分布不均匀或极端稠密图的误差分析不完整；
- 实验验证缺失，实际运行效率与可扩展性尚未评估；
- 对纯与近似 LNDP 之间的误差折衷缺乏完全的理论上限与算法匹配；
- 仅在合成图模型上给出下界，真实网络中可能出现更复杂的结构导致误差上限更高。

---

## 88. SACS: A Code Smell Dataset using Semi-automatic Generation Approach

**arXiv ID:** 2602.15342 | [PDF](https://arxiv.org/pdf/2602.15342v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 89. Semantic-Guided 3D Gaussian Splatting for Transient Object Removal

**arXiv ID:** 2602.15516 | [PDF](https://arxiv.org/pdf/2602.15516v1)

**作者:** Aditi Prabakaran `[一作]` (SRM University), Priyesh Shukla `[通讯]` (International Institute of Information Technology)

**通讯引用:** 185 | [OpenAlex ID](https://openalex.org/A5067601336)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于CLIP语义过滤的3D Gaussian Splatting方法，利用语义相似度对每个Gaussian累积评分，通过透明度正则化和定期修剪实现瞬时对象去除，消除伪影。

**💡 创新点**

创新点在于不依赖运动信息，而是使用CLIP对渲染视图进行语义匹配，将类别信息聚合到每个Gaussian，实现无视差假设的类别感知抑制，并通过两阶段（透明度正则化+定期修剪）提升重建质量。

**🔧 技术方法**

使用CLIP视觉语言模型、3D Gaussian Splatting、差分光栅化、透明度正则化、定期修剪以及视图级/高斯级语义统计。

**📊 数据集**

RobustNeRF基准数据集（Statue、Android、Yoda、Crab(2)四个序列）。

**📈 对比分析**

与Vanilla 3DGS和Mip-NeRF 360进行对比，在四个序列上平均PSNR提升至1.94 dB（最高提升1.94 dB），SSIM略升，LPIPS下降，且内存占用与实时渲染性能保持不变。

**⚠️ 局限性**

局限性包括：需手工指定干扰类别，CLIP对小物体（<50像素）识别弱，阈值τ需要针对每个数据集微调。

---

## 90. Doubly Stochastic Mean-Shift Clustering

**arXiv ID:** 2602.15393 | [PDF](https://arxiv.org/pdf/2602.15393v1)

**作者:** Tom Trigano `[一作]` (Shamoon College of Engineering), Itshak Lapidot `[通讯]` (Afeka Tel-Aviv Academic College of Engineering)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了双随机均值漂移（Doubly Stochastic Mean‑Shift, DSMS）算法，在原有随机点更新的基础上进一步随机化核带宽；

**💡 创新点**

创新点在于将核带宽随机采样，使得算法在稀疏数据和异构密度场中能更好地探测真模式，防止过度分割；

**🔧 技术方法**

技术核心包括随机带宽采样策略、子马尔可夫链与正子马尔可夫过程理论、Doob 收敛定理等；

**📊 数据集**

使用了三组2D高斯混合数据（各类簇均匀/不均匀的样本量）作为实验数据集；

**📈 对比分析**

与标准均值漂移、模糊均值漂移、随机均值漂移进行对比，实验显示在小样本或不平衡场景下DSMS的簇数估计更接近真值，且整体纯度与标签纯度之几何平均K值与传统方法持平或略优；

**⚠️ 局限性**

局限在于带宽区间和采样策略需手动设定，且对高维数据的扩展与自适应带宽选择仍待进一步研究。

---

## 91. Algorithmic differentiation for domain specific languages in C++ with expression templates

**arXiv ID:** 2602.15613 | [PDF](https://arxiv.org/pdf/2602.15613v1)

**作者:** Max Sagebaum `[一作]` (RPTU University Kaiserslautern-Landau), Nicolas R. Gauger `[通讯]` (RPTU University Kaiserslautern-Landau)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种面向域特定语言（DSL）的自动微分工具，该工具通过在表达式模板中为实体（如矩阵、向量）而非单个标量分配唯一标识符，并采用原始值打点（primal value taping）实现高效的正向和逆向计算。

**💡 创新点**

创新点包括：①在实体级别管理标识符，保持原始程序的内存结构，从而允许编译器进行 SIMD 等优化；②在表达式模板中扩展对左值、成员操作和临时对象的支持，避免了中间结果的存储；③提供了可自动化的注解与代码生成工具，使得任何DSL都能轻松集成；④与 CoDiPack、Stan Math 等现有工具进行对比，验证了该方法在内存和时间上的优势。

**🔧 技术方法**

使用的技术主要有：C++20 表达式模板、原始值打点、标识符重用管理、DSL 注解（AD_IN、AD_OUT 等）与代码生成、以及对 Eigen 等线性代数库的模板特化。

**📊 数据集**

在实验中使用的主要数据集是：①Coupled Burgers 方程（用于验证整体性能）；②四个线性代数基准（T1: 矩阵乘法，T2: 线性系统求解，T3: Kalman 滤波，T4: L1 分析凸优化）。

**📈 对比分析**

比较方法：将新工具与 CoDiPack（Jacobi、Primal Value、Jacobain）以及 Stan Math Library（普通版和专门优化版）进行对比，衡量录制时间、逆向时间、内存占用以及理论逆向因子。结果显示：在纯 C++ double 计算时性能相当；在矩阵乘法和求解等 DSL 典型操作上，新工具与 Stan Math 的专用实现相当，甚至更优；在 Kalman 滤波和 L1 优化等复合 DSL 场景下，逆向时间因子低于理论 4.5，内存占用比 CoDiPack Jacobian 更低。

**⚠️ 局限性**

局限性包括：①对 Eigen 的完整注解尚未完成，导致部分线性代数函数无法获得最优化；②需要对目标 DSL 进行手工或自动化注解，工作量相对较大；③在非 DSL 混合代码中，仍可能出现标识符冲突或内存布局不够优化的情况；④实验仅限于合成基准，未验证在真实大规模应用中的性能；⑤工具目前仅支持 C++20，可能限制在旧编译器环境下的使用。

---

## 92. Training-Free Zero-Shot Anomaly Detection in 3D Brain MRI with 2D Foundation Models

**arXiv ID:** 2602.15315 | [PDF](https://arxiv.org/pdf/2602.15315v1)

**作者:** Tai Le-Gia `[一作]` (Chungnam National University), Jaehyun Ahn `[通讯]` (Chungnam National University)

**通讯引用:** 1410 | [OpenAlex ID](https://openalex.org/A5013335966)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建了一个完全训练‑free、基于批处理的 3D 脑 MRI 零射击异常检测框架CoDeGraph3D；

**💡 创新点**

创新点在于将多轴 2D 基础模型特征聚合成局部 3D 立方体令牌，并结合随机投影与批处理相似度方法，实现可扩展的 3D 异常检测；

**🔧 技术方法**

使用冻结的 2D ViT（如 DINOv2）提取切片特征，三轴平均池化还原 3D 结构，随机投影压缩维度，MuSc/CoDeGraph3D 进行批级相似度评分；

**📊 数据集**

实验采用 IXI（健康数据）和 BraTS‑2025 METS（肿瘤）两组 T2/T1 体积扫描；

**📈 对比分析**

与 CLIP‑基零射击和重建式无监督基线对比，CoDeGraph3D 在患者级 AUROC 高达 96.9%（T2）/97.5%（T1），体素级 Dice 分别为 41.3%/33.8%，明显优于零射击基线且接近监督方法；

**⚠️ 局限性**

局限在于固定立方体令牌导致对极小或稀疏病灶的灵敏度降低，且相似度计算仍随样本数和令牌数呈二次增长，限制极大体积或大批量场景的可扩展性。

---

## 93. Near-real-time Solutions for Online String Problems

**arXiv ID:** 2602.15311 | [PDF](https://arxiv.org/pdf/2602.15311v1)

**作者:** Dominik Köppl `[一作]` (University of Yamanashi), Gregory Kucherov `[通讯]` (Gustave Eiffel University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

利用Breslauer–Italiano对Weiner算法的去摊销，提出了多种在线字符串问题（最长重复后缀数组、LZ77分解、最小唯一子串、反向LZ分解）的近实时算法；

**💡 创新点**

关键创新在于证明去摊销的后缀树构造过程不干扰对树功能的使用，从而在每个输入字符上实现双对数级别的最坏情况时间复杂度；

**🔧 技术方法**

核心技术包括：Weiner后缀树构造、动态彩色前驱查询、W-link维护、时间戳化节点、Manacher算法并行等；

**📊 数据集**

文中未使用实测数据集，所有结果均为理论分析；

**📈 对比分析**

与现有工作比较，本文将原先仅有摊销或更高时间复杂度（O(log³n)、O(log²n)等）的算法，在每个字母的最坏情况时间上降到O(log log n)（或相似的双对数级）且空间仍为O(n)；

**⚠️ 局限性**

局限性：时间仍包含多项式对数因子，取决于后缀树实现；在大字符集或特殊输入时，α相关的O(σ log log n)系数可能较高；此外，仅在理论上给出，缺乏实验验证。

---

## 94. Clinically Inspired Symptom-Guided Depression Detection from Emotion-Aware Speech Representations

**arXiv ID:** 2602.15578 | [PDF](https://arxiv.org/pdf/2602.15578v1)

**作者:** Chaithra Nerella `[一作]` (International Institute of Information Technology Hyderabad), Chiranjeevi Yarra `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于症状的、临床启发的交叉注意力框架，将说话者的情绪感知语音特征与PHQ-8症状问题对齐，用于抑郁严重程度的估计。

**💡 创新点**

创新点在于将情绪感知的语音表征与PHQ-8问卷嵌入进行交叉注意对齐，并引入可学习的症状特定温度参数，以实现对症状级别的可解释性预测。

**🔧 技术方法**

采用PDEM（情绪感知的wav2vec2-Large-Robust）进行语音特征提取，RoBERTa-large生成PHQ-8问题嵌入，结合症状引导交叉注意机制和多头全连接回归头进行预测。

**📊 数据集**

在EDAIC（扩展DAIC-WOZ）临床访谈数据集上进行实验，该数据集包含275位受试者的音频、转录及PHQ-8评分。

**📈 对比分析**

与现有使用PDEM情绪片段选择、通用音频特征等基线方法比较，实验结果显示测试集RMSE 5.15、MAE 4.13、CCC 0.52，均优于前沿方法，证明更高的预测准确性与解释性。

**⚠️ 局限性**

局限性包括仅使用语音难以捕捉与生理相关的症状（如睡眠、疲劳），缺乏多模态信息；温度参数的学习策略仍需进一步研究以优化不同症状的注意分布。

---

## 95. An Empirical Study on the Effects of System Prompts in Instruction-Tuned Models for Code Generation

**arXiv ID:** 2602.15228 | [PDF](https://arxiv.org/pdf/2602.15228v1)

**作者:** Zaiyu Cheng `[一作]` (William and Mary), Antonio Mastropaolo `[通讯]` (William and Mary)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了系统提示对指令调优模型在代码生成任务中的影响。

**💡 创新点**

创新点在于量化不同系统提示细粒度、模型规模、提示策略和语言对代码助手性能的交互作用，并提供可复现的测试床。

**🔧 技术方法**

采用指令调优的LLM（GPT-OSS-20B、Qwen2.5-Coder 1.5B/7B/32B）、多种系统提示、zero-shot、few-shot（固定与检索）以及不同温度采样技术。

**📊 数据集**

使用CoderEval基准（460道Java/Python问题）和200k检索语料库进行检索式few-shot。

**📈 对比分析**

通过Pass@1/Pass@5统计、McNemar检验和odds ratio比较各配置，发现Java对系统提示高度敏感，结构性提示和检索示例能显著提升性能；Python表现稳定。

**⚠️ 局限性**

局限在于仅评估两种语言、有限的系统提示设计、未探讨更大模型或其他任务，且结果受检索质量和模型内部解释的影响。

---

## 96. The Turbo-Charged Mapper: Fast and Optimal Mapping for Accelerator Modeling and Evaluation

**arXiv ID:** 2602.15172 | [PDF](https://arxiv.org/pdf/2602.15172v1)

**作者:** Michael Gilbert `[一作]` (Massachusetts Institute of Technology), Joel S. Emer `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 25368 | [OpenAlex ID](https://openalex.org/A5024384625)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Turbo‑Charged Mapper (TCM)，一种能在可行运行时间内寻找深度神经网络加速器最优映射的方法。

**💡 创新点**

核心创新是引入“dataplacement”概念并与数据流、tile 形状分离，利用其进行大规模剪枝、部分 tile 形状优化，并通过模型“currying”显著加速计算。

**🔧 技术方法**

采用了映射空间剪枝技术（redundant dataflow pruning、非有用循环剪枝、partial tile shape pruning），基于符号推导的分析模型和多级内存访问分析，结合 Python/C++ 实现的快算子模型。

**📊 数据集**

使用 GPT‑3 6.7B 与 MobileNetV3 两个实际工作负载作为评估数据集，覆盖不同张量维度与运算复杂度。

**📈 对比分析**

与 Timeloop、Timeloop+Hint 以及 LOMA 等主流映射器对比，TCM 在约 37 秒内找到真正最优映射，其能耗‑延迟乘积（EDP）相对最佳仅 1 倍，且即使给定 1000× 的运行时间，传统方法仍比 TCM 高出 21% 以上。

**⚠️ 局限性**

局限性在于：依赖精确的分析模型和硬件参数，若模型或参数不准确会影响最优性；对极大内存层级或特殊硬件（如非标准网络拓扑）的扩展尚未全面验证。

---

## 97. Algorithmic Approaches to Opinion Selection for Online Deliberation: A Comparative Study

**arXiv ID:** 2602.15439 | [PDF](https://arxiv.org/pdf/2602.15439v1)

**作者:** Salim Hafid `[一作]` (Sciences Po Paris), Jean-Philippe Cointet `[通讯]` (Sciences Po Paris)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在线协商平台中意见选择的算法，对现有策略进行系统基准评估，并提出了兼顾多样性与比例代表性的DiverseBJR算法。

**💡 创新点**

创新点在于将社会选择理论中的公正代表（JR、BJR）与基于投票相似度的多样性约束相结合，形成新的DiverseBJR选择规则。

**🔧 技术方法**

技术手段包括基于认可投票的多赢家选举框架、贪心近似算法、Hamming距离测度、桥接（Bridging）、参与度（Engagement）与多样性（Diversity）基线。

**📊 数据集**

使用Remesh的“Polarized Issues”数据集（美国公众参与）以及一份英国公民大会数据集进行实验。

**📈 对比分析**

通过在5个民主度量（个体/群体代表性、共识、覆盖度、冗余、福利）上对比，DiverseBJR在小子集（k≈2–3）下实现了最优的代表性与多样性折中，优于现有基线。

**⚠️ 局限性**

局限性包括仅做离线评估、依赖推断的投票矩阵、对小子集有效且在真实部署中效果可能不同、缺乏用户体验与交互评估。

---

## 98. Consistency-Preserving Diverse Video Generation

**arXiv ID:** 2602.15287 | [PDF](https://arxiv.org/pdf/2602.15287v1)

**作者:** Xinshuang Liu `[一作]` (University of California), Truong Nguyen `[通讯]` (University of California)

**通讯引用:** 15558 | [OpenAlex ID](https://openalex.org/A5102719190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于流匹配的视频生成联合采样框架，利用多样性梯度提升跨视频多样性，同时通过一致性约束保证视频内部时间一致性；

**💡 创新点**

创新点在于：①将多样性梯度与一致性梯度进行投影调节，仅剔除会降低时间一致性的多样性更新；②在潜在空间训练轻量级嵌入与插值模型，避免视频解码和反向传播的高成本；

**🔧 技术方法**

采用流匹配生成器、确定性点过程（DPP）多样性目标、潜在空间视频/帧嵌入与帧插值网络、梯度投影调节机制；

**📊 数据集**

使用WAN 2.1 t2v-1.3B 作为基准模型，生成10个多样化文本提示下的4×20（共80）视频，训练潜在模型时采集100个训练/20个测试视频；

**📈 对比分析**

与IID、DPP、Particle Guidance、DiverseFlow等基线进行对比，结果显示在跨视频多样性（Vendi-v/Vendi-f）上相当或略优，且在时间一致性（MSE）和颜色自然度（CNI）上显著优于基线；

**⚠️ 局限性**

局限性包括：仍需在潜在空间训练额外模型；多样性与一致性仍存在微小权衡；在更长时序或高分辨率视频上的评估尚未充分验证。

---

## 99. Secure and Energy-Efficient Wireless Agentic AI Networks

**arXiv ID:** 2602.15212 | [PDF](https://arxiv.org/pdf/2602.15212v1)

**作者:** Yuanyan Song `[一作]`, Xinmian Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种安全的无线代理AI网络，采用主管AI代理动态选择参与推理的AI代理并让未被选中的代理充当友好噪声干扰者，以保障隐私和推理质量；设计能量最小化问题，联合优化AI代理选择、基站波束成形和代理发射功率；提出两种资源分配方案ASC（基于ADMM‑SDR‑SCA的分解求解）和LAW（基于LLM优化器的代理工作流），并在实验中验证能量消耗显著降低、推理准确率可观。

**💡 创新点**

①将友好噪声干扰与代理选择相结合，构建安全推理框架；②把多目标能量最小化问题拆解为三子问题，并针对每个子问题分别设计ADMM、SDR、SCA和LLM优化器；③利用LLM进行资源分配的自学习代理工作流，减少人工调参。

**🔧 技术方法**

混合整数非线性优化、ADMM分解、半正定松弛（SDR）、顺序凸逼近（SCA）、大语言模型（LLM）优化器、基站多天线波束成形、友好噪声干扰、LLM推理精度预测（规模-准确度关系）。

**📊 数据集**

ARC‑E、ARC‑C、BoolQ公开问答数据集用于评估推理准确率；实验数据基于模拟无线信道和Qwen系列LLM模型。

**📈 对比分析**

与遗传算法（GA）、随机代理选择（RandAS）、差分进化（DE）、固定功率（FixedTP）、固定波束（FixedBF）等基线算法比较。ASC在能量消耗上比其它方案低约25‑30%，LAW略高但仍优于基线；推理准确率在ARC‑E、ARC‑C、BoolQ上分别达85%、75%和83%，显著高于RandAS和GA。

**⚠️ 局限性**

仅在模拟环境下验证，未考虑多用户、移动代理、动态信道估计复杂度；LLM优化器依赖外部API，推理延迟与算力成本未量化；友好噪声干扰对无缝连接用户的干扰评估不足。

---

## 100. MeshMimic: Geometry-Aware Humanoid Motion Learning through 3D Scene Reconstruction

**arXiv ID:** 2602.15733 | [PDF](https://arxiv.org/pdf/2602.15733v1)

**作者:** Qiang Zhang `[一作]`, Yijie Guo `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

基于单目视频同时重建人类运动轨迹和场景几何，并通过优化运动一致性与场景对齐，将重建得到的人类运动通过MeshRetarget方法映射到人形机器人，实现从视频到机器人动作的完整闭环；

**💡 创新点**

①将人类运动与场景几何联合优化得到统一尺度、物理一致的参考轨迹；②引入基于深度边缘的接触预测与TSDF的穿透修正，提升运动的可行性；③开发MeshRetarget实现对非平面地形的接触保留与碰撞避免；④利用3D感知基础模型（SAM3D、π^3等）构建高质量语义场景，形成首个全流程的单目视频到机器人全身控制的闭环框架；

**🔧 技术方法**

3D Gaussian Splatting、NeRF、π^3、SAM3D Body、ViTDet、SAM2、TSDF、SQP优化、异步PPO（IsaacLab）、ExBody等多种视觉与强化学习技术；

**📊 数据集**

SLOPER4D数据集（人类与场景并存），以及与VideoMimic等方法对比的公开视频数据；

**📈 对比分析**

对比WHAM、TRAM、VideoMimic三种基线，使用WA‑MPJPE、W‑MPJPE与Chamfer距离评估；MeshMimic在WA‑MPJPE从112.13降至94.32（≈15%），W‑MPJPE从696.62降至518.98（≈25%），Chamfer距离从0.75降至0.61（≈19%）。在8个真实-模拟-真实任务上，MeshMimic在平均奖励和真实成功率上均超过VideoMimic，且在使用全局躯干位置观测后，长周期任务成功率提升高达30%；

**⚠️ 局限性**

对全局躯干位置的依赖需要外部光学位姿估计；在短周期高速动作中加入全局位姿反而导致成功率下降；重建质量对最终性能高度敏感，对极端遮挡或低质量深度仍易产生漂移；目前未在完全未知地形上验证，适用性受限于训练场景。

---

## 101. Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control

**arXiv ID:** 2602.15659 | [PDF](https://arxiv.org/pdf/2602.15659v1)

**作者:** Luankang Zhang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27993 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出递归自我改进框架 RSIR，利用当前模型生成高保真用户交互序列并在闭环中逐步训练，以缓解推荐系统的数据稀疏问题。

**💡 创新点**

核心创新：① 递归自我生成而非依赖外部教师模型；② 通过受限探索与基于保真度的质量控制防止错误放大；③ 证明 RSIR 作为数据驱动的隐式正则化，可平滑优化景观。

**🔧 技术方法**

使用自回归生成式模型（Transformer/SASRec 等）+ top‑k 采样与混合候选池实现受限探索；保真度检查基于排名阈值；隐式正则化与曲面平滑分析。

**📊 数据集**

实验数据集：Amazon Beauty、Sports、Toys（来自 Amazon 评价数据）以及 Yelp 公共数据集。

**📈 对比分析**

与传统数据增强（重排序、插入）和可学习数据生成方法（ASREP、DiffuASR、DR4SR）以及三种基线模型（SASRec、CL4SRec、HSTU）对比；单迭代即可提升 Recall@10、NDCG@10 约 5–10%，多次迭代累计提升至 10–15%。

**⚠️ 局限性**

局限性：保真度阈值与探索比例需手动调参；阈值过宽易引入噪声，过窄导致生成不足；在大词表或深层模型上生成速度受限；最终性能趋于饱和，长期迭代可能放大模型固有偏差。

---

## 102. MEV in Binance Builder

**arXiv ID:** 2602.15395 | [PDF](https://arxiv.org/pdf/2602.15395v1)

**作者:** Qin Wang `[一作]` (CSIRO Data61), Shiping Chen `[通讯]` (CSIRO Data61)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过链上追踪和分析，系统性研究了币安智能链（BSC）上基于建造者的MEV套利活动，量化了两大建造者（48Club、Blockrazor）的区块产出份额、利润分布以及路径结构，揭示了短区块间隔与白名单设计导致的MEV集中与审查风险。

**💡 创新点**

创新点在于首次构建大规模的BSC建造者MEV数据集（约195万笔套利交易），从路径、利润、时间等多维度评估MEV分配与竞争格局，证明BSC的PBS结构加剧了中心化与低延迟优势导致的结构性不平等。

**🔧 技术方法**

采用链上区块与交易跟踪、套利路径重构、利润归因、统计相关分析以及与以太坊PBS对比的实证方法；结合BEP-322协议细节和建造者API识别技术。

**📊 数据集**

数据集来源为2025年5月至11月BSC链上完整区块、交易回溯、合约事件，标注了48Club与Blockrazor的27,103与19,253个建造者相关合约，覆盖220+工厂、12,300+流动池。

**📈 对比分析**

通过比较建造者区块份额、利润份额、路径长度与效率、时间序列趋势等指标，评估BSC与以太坊PBS的差异；结果显示48Club占据约70%利润，Blockrazor约30%，两者共占96%区块，短跳路径盈利主导，整体效率显著高于较长路径，表明BSC的MEV市场高度集中且对延迟敏感。

**⚠️ 局限性**

局限性包括只聚焦两大建造者，忽略小型建造者与私有订单流；仅研究套利类MEV，未覆盖背跑、清算等形式；数据标注可能漏检短期合约；不包含链下利润分配信息，导致对搜索者真实收益估计不完整。

---

## 103. STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens

**arXiv ID:** 2602.15620 | [PDF](https://arxiv.org/pdf/2602.15620v1)

**作者:** Shiqi Liu `[一作]` (Tsinghua University), Shengbo Eben Li `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在强化学习微调中识别并屏蔽极少数对梯度产生过大影响的无关“伪造”token，提升了大语言模型的推理稳定性和性能。

**💡 创新点**

创新点在于系统化地从token概率、熵与优势符号三个维度构建“伪造token”判别标准，并仅屏蔽约0.01%的这些token，实现梯度的精准控制。

**🔧 技术方法**

技术包括基于GRPO/DAPO的分组优势估计、自然梯度对熵变化的解析、熵与概率阈值自适应筛选以及S2T（Silencing Spurious Tokens）屏蔽机制。

**📊 数据集**

使用Dapo-Math-17K进行训练，并在六大数学推理基准（AIME24/25、AMC23、MATH500、Minerva、OlympiadBench）进行评估。

**📈 对比分析**

与GRPO、20-Entropy和JustRL等现有RL方法对比，STAPO在训练熵稳定性、奖励和多种基准上的平均准确率均显著提升（如1.7B模型在AIME24上提升13.5%）。

**⚠️ 局限性**

局限性包括仅在数学推理任务上验证，未对代码生成等其他领域进行评估，也未考虑错误答案中的token行为，且实验缺乏更全面的分类场景分析。

---

## 104. CDRL: A Reinforcement Learning Framework Inspired by Cerebellar Circuits and Dendritic Computational Strategies

**arXiv ID:** 2602.15367 | [PDF](https://arxiv.org/pdf/2602.15367v1)

**作者:** Sibo Zhang `[一作]` (Academy of Medical Engineering and Translational Medicine), Yunliang Zang `[通讯]` (Academy of Medical Engineering and Translational Medicine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于小脑结构的RL框架，在高维噪声Pong环境中验证其样本效率、鲁棒性与泛化性。

**💡 创新点**

将小脑的大扩展、稀疏连接、稀疏激活以及树突层调制作为网络先验，构建全新的RL架构。

**🔧 技术方法**

采用Double DQN价值函数近似，结合GrC层随机稀疏投影、PC层稀疏连接和非训练的树突调制模块。

**📊 数据集**

使用自定义的高维像素Pong游戏环境作为实验数据集。

**📈 对比分析**

与同等参数容量的基线网络进行对比，评估样本效率、噪声/动作扰动下的鲁棒性以及在环境参数变化下的泛化；结果显示CDRL在样本效率上更快，鲁棒性提升约5‑10%，泛化表现优于基线。

**⚠️ 局限性**

实验仅覆盖离散动作的Pong任务，未验证连续控制、多任务或更复杂环境；对计算资源的依赖仍需进一步优化。

---

## 105. Concept-Enhanced Multimodal RAG: Towards Interpretable and Accurate Radiology Report Generation

**arXiv ID:** 2602.15650 | [PDF](https://arxiv.org/pdf/2602.15650v1)

**作者:** Marco Salmè `[一作]` (Università Campus Bio-Medico of Roma), Valerio Guarrasi `[通讯]` (Università Campus Bio-Medico of Roma)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种统一框架——Concept-Enhanced Multimodal RAG，将可解释的视觉医学概念与多模态检索增强生成相结合，用于自动生成胸部X光检查报告。

**💡 创新点**

创新点包括：①将视觉特征分解为可解释的医学关键词（SpLiCE）并将其作为高层次提示；②将检索到的相似案例与概念融合到层级prompt中，既提升可解释性又改善事实一致性；③提供模块化设计和概念级可视化方法，证实可解释性不一定与性能产生负向权衡。

**🔧 技术方法**

使用的核心技术：SpLiCE概念提取、CLIP/CXR-CLIP视觉-文本对齐、FAISS检索、Mistral-7B/LLaVA LLM层级prompting、Grad‑ECLIP可解释性；实验中还结合了Zero‑Shot和SFT两种训练范式。

**📊 数据集**

实验数据集：MIMIC‑CXR（约37万张胸部X光）和IU‑Xray（约7千张胸部X光），分别用于内部检索和跨域检索。

**📈 对比分析**

比较方法：在Zero‑Shot与SFT两种设置下，分别对Image‑Only、Concepts、RAG和Combined四种prompt策略进行评估。评估指标包括NLP指标（ROUGE‑L、BLEU‑1/4）和临床指标（F1‑CheXbert、F1‑RadGraph）。结果表明，Combined策略在大多数指标上优于单一策略，例如在Zero‑Shot下F1‑RadGraph提升至约0.18‑0.19，在SFT下提升至约0.20以上，体现可解释性增强带来的准确度提升。

**⚠️ 局限性**

局限性：①概念提取高度依赖CLIP与医学语料对齐，若对齐不足会产生噪声；②语言模型仅通过prompt间接受约束，缺乏对生成过程的直接解释；③跨域检索效果依赖源数据质量，尚需在更多医学领域验证；④实验使用的LLM较大，资源消耗高，尚未探讨在资源受限环境下的可部署性。

---

## 106. Fine-Tuning LLMs to Generate Economical and Reliable Actions for the Power Grid

**arXiv ID:** 2602.15350 | [PDF](https://arxiv.org/pdf/2602.15350v1)

**作者:** Mohamad Chehade `[一作]` (University of Texas at Austin), Hao Zhu `[通讯]` (University of Texas at Austin)

**通讯引用:** 108801 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

针对公共安全停电事件，提出多阶段微调流水线，将指令调优大型语言模型转化为可验证的开关方案生成器。

**💡 创新点**

创新点在于将DC-OPF MILP解作为监督标签，设计可解析的行动语法；随后通过基于AC电压惩罚的直接偏好优化（DPO）提升电压安全；并在推理时使用Best‑of‑N进一步筛选。

**🔧 技术方法**

采用指令调优的LLM、监督微调（SFT）、直接偏好优化（DPO）、Best‑of‑N筛选，以及MATPOWER、YALMIP求解DC/AC潮流。

**📊 数据集**

使用IEEE 118节点测试系统，生成约200个PSPS场景做训练，440对偏好样本做DPO。

**📈 对比分析**

与零射击LLM、SFT、DPO及全连接MLP基线对比；结果表明SFT/DPO在DC目标上明显下降，AC失败率从50%降至个位数，电压惩罚分布进一步改善。

**⚠️ 局限性**

局限性包括对DC‑OPF解的依赖、偏好数据量有限导致高峰电压违规仍存在、对更大规模系统或其他操作模式的迁移尚未验证。

---

## 107. Proactive Conversational Assistant for a Procedural Manual Task based on Audio and IMU

**arXiv ID:** 2602.15707 | [PDF](https://arxiv.org/pdf/2602.15707v1)

**作者:** Rehana Mahfuz `[一作]` (Qualcomm Technologies), Phanidhar Chinchili `[通讯]` (Qualcomm Technologies)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一款仅依赖可穿戴设备音频和IMU信号的实时主动式会话助手，用于指导家具组装等手工任务；

**💡 创新点**

创新点在于：① 通过轻量化传感器实现视频无关、边缘端即时主动指导；② 设计了 UWA LoRA 微调方法，显著抑制无关对话并提升关键信息传递；③ 通过完整的端到端流程实现完全离线部署；

**🔧 技术方法**

技术手段包括多模态语言模型（Qwen2.5 LoRA 微调）、CNN10 PANN 音频编码器、Attend&Discriminate IMU 编码器、Whisper 语音转文本、MeloTTS 语音合成、ONNX/ QNN 微调与推理、REST/Redis 边缘通信；

**📊 数据集**

使用的数据集有：SAMoSA（活动识别）、BoxLift（提举动作）、自制的活动日志生成器（结合随机误差），以及通过 GPT‑4o 生成的 600 条模拟对话；

**📈 对比分析**

通过与零/一/四示例的几种提示方式以及未微调模型对比，finetuned 模型在关键指令、错误纠正和答案的召回/精确度上提升约 30%（F‑score >0.96），同时推理时间缩短约 16 倍，保持低延迟；

**⚠️ 局限性**

局限性包括：提升不均衡的提举动作识别（数据稀缺导致 F‑score 低）、微调后对错误纠正与问答召回略有下降、以及活动日志生成需手工规则，尚未实现完全自动化；

---

## 108. Automatically Finding Reward Model Biases

**arXiv ID:** 2602.15222 | [PDF](https://arxiv.org/pdf/2602.15222v1)

**作者:** Atticus Wang `[一作]` (Massachusetts Institute of Technology), Arthur Conmy `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种黑盒演化式管道，利用语言模型生成和筛选候选属性，通过对比奖励模型与对齐模型的偏好，自动发现奖励模型中的系统性偏差。

**💡 创新点**

创新点在于将演化算法与对抗性属性生成相结合，使用对抗性相对改写构造计数对，以自然语言描述属性；同时提出“多样性调节的偏差强度”指标，用以量化发现的偏差数量与质量。

**🔧 技术方法**

核心技术包括：对抗性属性生成（LLM重新写作）、基于计数对的偏差度量、两目标Pareto优化、演化迭代、以及基于Claude Sonnet 4.5的LLM裁判。

**📊 数据集**

主要数据集为Skywork‑Reward‑V2‑Llama‑3.1‑8B奖励模型，配合人工设计的20个主题下的合成用户提示集；验证阶段使用三种不同重写模型与留出测试集。

**📈 对比分析**

与传统的best‑of‑N搜索相比，演化搜索在发现显著偏差数量、DABS分数及Pareto前沿覆盖率上均表现更优；在Skywork‑V2‑8B上共发现10个统计显著的偏差，覆盖7/20主题。

**⚠️ 局限性**

局限性包括：使用合成提示可能无法完全代表真实用户分布；检索到的偏差可能不完整，演化搜索对罕见属性的召回有限；依赖LLM重写导致属性与其他特征耦合难以完全分离。

---

## 109. Enhancing Computational Efficiency in NetLogo: Best Practices for Running Large-Scale Agent-Based Models on AWS and Cloud Infrastructures

**arXiv ID:** 2602.15317 | [PDF](https://arxiv.org/pdf/2602.15317v1)

**作者:** Michael A. Duprey `[一作]` (RTI International), Georgiy V. Bobashev `[通讯]` (RTI International)

**通讯引用:** 4057 | [OpenAlex ID](https://openalex.org/A5078220202)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套针对 NetLogo 在 AWS 等云平台上运行大规模代理模型的优化方案，涵盖了 NetLogo 版本选择、JVM 参数调优、BehaviorSpace 并行执行以及实例类型挑选等方面。

**💡 创新点**

创新点在于将 NetLogo 6.4.0 及以上版本的新内存管理特性与 Java 虚拟机高级参数（如 G1GC、服务器模式）相结合，并通过对三种主流 EC2 实例（计算优化、内存优化、通用型）的系统性比较，首次证明 c6a.48xlarge 在成本与性能上最具优势。

**🔧 技术方法**

主要技术包括 NetLogo 6.4.0/6.5 版本、Java 虚拟机 JVM 参数（-Xmx、-XX:+UseG1GC、-XX:MaxGCPauseMillis 等）、BehaviorSpace 并行实验、AWS EC2 云实例、Amazon Linux 2023、headless 模式运行以及系统监控工具（如 vmstat）进行性能记录。

**📊 数据集**

使用的数据集为 NetLogo 自带的 wolf‑sheep‑predation（狼-羊-草）模型，在此模型上执行约 41,000 次参数组合实验。

**📈 对比分析**

比较方法为在 c6a.48xlarge、m6a.48xlarge、r6a.48xlarge 三种实例上各运行 10 次实验，记录用户时间、系统时间、总耗时、最大内存与实验成本；结果显示 c6a 实例在成本上比 r6a 节省 32%，且在时间与内存使用上保持最小波动。

**⚠️ 局限性**

局限性在于实验仅基于一个相对简单的模型，未考虑更复杂模型的计算/内存需求、并发 I/O、存储性能等因素，因而结论可能不完全适用于不同规模或不同特性的代理模型。

---

## 110. Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition

**arXiv ID:** 2602.15124 | [PDF](https://arxiv.org/pdf/2602.15124v1)

**作者:** Shiyu Xuan `[一作]` (Nanjing University of Science and Technology), Jinhui Tang `[通讯]` (Nanjing Forestry University)

**通讯引用:** 28481 | [OpenAlex ID](https://openalex.org/A5035112538)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了将人-物检测与交互识别解耦的零样本HOI框架，并利用多模态大型语言模型进行交互识别。

**💡 创新点**

创新点在于将交互识别转化为视觉问答任务，采用确定性生成与单通道确定性匹配，实现无训练或轻微微调即可识别未见交互，并通过空间感知池化融合外观与空间信息提升鲁棒性与效率。

**🔧 技术方法**

采用多模态大型语言模型（如Qwen 2.5‑VL 3B）、ROIAlign、交叉注意力、MLP空间编码、一通道确定性匹配、Deterministic Generation、LoRA微调等技术。

**📊 数据集**

在HICO‑DET和V‑COCO两大HOI基准上进行评测，采用RF‑UC、NF‑UC、UO、UV等零样本设置以及跨数据集迁移。

**📈 对比分析**

与多种一/二阶段方法（ADA‑CM、CLIP4HOI、BC‑HOI等）及基于CLIP/BLIP2的模型比较，零样本mAP在所有设置下名列前茅，跨检测器和跨数据集场景下表现亦显著优于前沿方法。

**⚠️ 局限性**

局限在于依赖高质量检测框和外部大型语言模型，推理速度受限于M‑LM；对检测误差仍有一定敏感性，且对候选交互列表长度和顺序存在轻微影响。

---

## 111. SoliDualSPHysics: An extension of DualSPHysics for solid mechanics with hyperelasticity, plasticity, and fracture

**arXiv ID:** 2602.15149 | [PDF](https://arxiv.org/pdf/2602.15149v1)

**作者:** Mohammad Naqib Rahimi `[一作]` (Synopsys Inc), George Moutsanidis `[通讯]` (Rutgers University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了开源 GPU 加速的 DualSPHysics 扩展 SoliDualSPHysics，实现独立固体力学的全变形弹塑性与脆性断裂模拟。

**💡 创新点**

创新点在于将总拉格朗日 SPH 与超光速相位场断裂结合，提供无额外追踪、无局部细化的脆性断裂建模，同时支持多分辨率、独立时间步、用户表达式。

**🔧 技术方法**

使用了 Total Lagrangian SPH、超光速相位场模型、J2 多应变塑性、CUDA/OpenMP 并行、用户表达式解析、DEM 接触、GPU 加速等技术。

**📊 数据集**

通过一系列标准数值基准（Cantilever beam/plate、3D column、动态裂纹分支、Kalthoff-Winkler、四点弯曲、飞行板冲击、Taylor 梯形杆冲击）以及实验数据进行验证。

**📈 对比分析**

与已有 SPH 代码、实验和 FEM 结果进行数值对比；GPU 实现相较 CPU 多线程获得 3.7–4.1× 加速（Machine1）和 1.55–1.71×（Machine2），每步时间稳定，近线性问题规模伸缩。

**⚠️ 局限性**

局限性包括只能模拟脆性断裂（无法与 J2 塑性共用）、极端形变下 DEM 接触精度有限、未实现流固耦合以及更高效的内存局部化等。

---

## 112. Feasibility-aware Imitation Learning from Observation with Multimodal Feedback

**arXiv ID:** 2602.15351 | [PDF](https://arxiv.org/pdf/2602.15351v1)

**作者:** Kei Takahashi `[一作]` (Nara Institute of Science and Technology), Takamitsu Matsubara `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 2790 | [OpenAlex ID](https://openalex.org/A5042074952)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FABCO 框架，结合手持演示接口的多模态可行性反馈与行为克隆从观察学习，提升演示动作的机器人可行性并训练更稳健的控制策略。

**💡 创新点**

①首次在无机器人演示的场景下引入可行性估计与反馈；②将可行性信息同时用于演示修正与策略学习；③融合视觉与触觉两种反馈模式，适配不同任务需求。

**🔧 技术方法**

行为克隆从观察 (BCO)、前向与逆动力学模型 (FDM/IDM)、动作分块与时间集成 (ACTE)、多模态可行性估计与反馈、基于可行性的加权策略学习。

**📊 数据集**

人类被试（15名）在 peg‑insertion 与 circle‑tracing 两个任务上演示数据；机器人跟踪随机轨迹收集动力学数据用于训练 FDM/IDM；无标注动作的演示序列作为学习样本。

**📈 对比分析**

与无反馈 BCO 进行对比，使用相同演示任务、相同模型结构；通过任务成功率、Hausdorff 距离、NASA‑TLX 等指标评估。结果显示：可行性反馈后成功率提升至 90%/88%（vs 0%/26.7%），性能提升超过 3.2 倍，且视觉+触觉组合效果最佳。

**⚠️ 局限性**

①可行性反馈仅指出不可行区间，缺乏具体修改建议；②对动力学模型的依赖导致数据收集成本高；③BCO 易受累积误差影响；④未覆盖抓手开合的可行性；⑤受限于实验规模与任务多样性。

---

## 113. Semantics-Aware Denoising: A PLM-Guided Sample Reweighting Strategy for Robust Recommendation

**arXiv ID:** 2602.15359 | [PDF](https://arxiv.org/pdf/2602.15359v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 114. tensorFM: Low-Rank Approximations of Cross-Order Feature Interactions

**arXiv ID:** 2602.15229 | [PDF](https://arxiv.org/pdf/2602.15229v1)

**作者:** Alessio Mazzetto `[一作]` (Brown University), Krzysztof Dembczyński `[通讯]` (Yahoo Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现一种新型因子分解机模型 TensorFM，能够显式建模多阶特征交互，并通过低秩张量分解实现高效推理。

**💡 创新点**

创新点在于将 Field‑weighted FM 的低秩矩阵思想推广到高阶张量，利用 CP 分解得到可控秩的交互张量，使得推理复杂度降为 O(n k r d²)，并保持模型可解释性。

**🔧 技术方法**

采用的核心技术包括低秩 CP 张量分解、因子分解机（FM）、Field‑aware FM、Higher‑Order FM（HOFM）等模型的对比实验，使用 Optuna 进行超参数搜索，并通过 Cython 对推理速度做进一步优化。

**📊 数据集**

实验数据集涵盖真实场景的 Avazu、Criteo 广告点击率数据、COMPAS 犯罪复发预测数据，以及自定义的三阶/四阶交互合成数据。

**📈 对比分析**

与 LR、FM、FwFM、AFM、CN、HOFM 等基线在 AUC / LogLoss 上进行对比。TensorFM 在低秩配置下与高秩配置均能达到或超过基线，尤其在高阶交互数据上表现最佳；推理时间随字段数线性增长，显著低于 AFM、FwFM 等二阶交互模型。

**⚠️ 局限性**

限制点包括：尚未与深度网络或两流架构结合；高秩配置会显著增加参数量；在现实应用中仍需关注公平性和解释性评估；对极大字段数和高阶交互的泛化能力尚待进一步验证。

---

## 115. CGRA-DeBERTa Concept Guided Residual Augmentation Transformer for Theologically Islamic Understanding

**arXiv ID:** 2602.15139 | [PDF](https://arxiv.org/pdf/2602.15139v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 116. Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory

**arXiv ID:** 2602.15313 | [PDF](https://arxiv.org/pdf/2602.15313v1)

**作者:** Zihao Tang `[一作]` (Microsoft), Qi Zhang `[通讯]` (Microsoft)

**通讯引用:** 14944 | [OpenAlex ID](https://openalex.org/A5100360194)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个新型 AI 记忆框架，将传统的相似度检索（System-1）与全局层级检索（System-2）结合，能够同时捕获语义相似与结构相关的记忆。

**💡 创新点**

创新点在于：① 构建双层图结构（基础图 + 分层图）并引入三大设计原则（最小概念抽象、多对多映射、压缩效率约束）；② 通过“全局选择”实现从高层到低层的顶层向下遍历，补足传统 RAG 在全局推理与覆盖度上的不足；③ 在检索后使用混合重排序模型充分挖掘两条路径的互补优势。

**🔧 技术方法**

技术手段包括：图结构存储（Neo4j、Graphiti），多模态嵌入检索（Qwen3-Embedding-0.6B）、BM25 与余弦相似度检索、递归秩融合 (RRF) 与 Qwen3-Reranker-8B 重排序、LLM 进行实体/边/分层抽取及推理。

**📊 数据集**

使用了两大长记忆基准：LoCoMo（约 16K token 对话，1540 题）和 LongMemEval‑S（115K token 对话，500 题）。

**📈 对比分析**

与 Full Context、RAG、LangMem、MemOS、Zep、Nemori、PreMem、EverMemOS、EMem-G 等基线比较，系统在 LoCoMo 上 93.9 分、LongMemEval‑S 上 91.6 分，均超过所有对比方法，尤其在多跳、时序与复杂推理题型中显著提升。

**⚠️ 局限性**

局限性包括：① 层级图需周期性重建，缺乏在线增量更新机制；② 依赖大型 LLM 进行抽取与遍历，推理成本较高；③ 对时序问题的全局选择效果有限，可能无法完整捕获事件序列；④ 目前仅支持文本数据，未扩展多模态记忆。

---

## 117. LLM-to-Speech: A Synthetic Data Pipeline for Training Dialectal Text-to-Speech Models

**arXiv ID:** 2602.15675 | [PDF](https://arxiv.org/pdf/2602.15675v1)

**作者:** Ahmed Khaled Khamis `[一作]` (Georgia Institute of Technology), Hesham Ali `[通讯]` (Nile University)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5081121710)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建了38小时埃及阿拉伯语TTS数据集NileTTS，并对XTTS v2进行微调以提升埃及阿拉伯语合成质量。

**💡 创新点**

首个公开埃及阿拉伯语TTS数据集；可复现的基于LLM和神经语音合成的合成数据生成管线；公开微调后的XTTS模型。

**🔧 技术方法**

大型语言模型生成内容、NotebookLM语音合成、Whisper自动转写、ECAPA‑TDNN声学分离、XTTS v2 GPT‑style TTS模型微调、实验跟踪与评估。

**📊 数据集**

NileTTS数据集（38.1小时、两位说话者，医疗、销售与日常会话领域），并用Whisper生成转写文本。

**📈 对比分析**

与原始XTTS v2在同一评估集上比较，使用WER、CER和Speaker Similarity；NileTTS WER 18.8% vs 26.8%（下降29.9%），CER 4.1% vs 8.1%（下降49.4%），声学相似度 0.755 vs 0.713（提升5.9%）。

**⚠️ 局限性**

仅包含两位说话者、数据为合成语音缺乏多样性、评估主要基于自动指标、未覆盖更专业领域、合成数据对真实语音泛化的未知。

---

## 118. An Industrial Dataset for Scene Acquisitions and Functional Schematics Alignment

**arXiv ID:** 2602.15584 | [PDF](https://arxiv.org/pdf/2602.15584v1)

**作者:** Flavien Armangeon `[一作]` (Universite Paris-Saclay), Gabriele Facciolo `[通讯]` (Universite Paris-Saclay)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个包含工业现场点云、全景图像、CAD模型、管道路由、2D标注框与分割掩码以及P&ID的完整数据集，并基于此实现了场景获取与功能示意图的自动对齐方法。

**💡 创新点**

创新点主要有：①首次公开同时包含真实采集与功能示意图的工业数据集；②采用2D基础模型（Grounding‑DINO + SAM）与半自动管道重建（PipeRunner）相结合的 3D 分割策略；③将场景与示意图统一为带属性的图结构，使用基于最优传输的 SLOTAlign 进行鲁棒图匹配，并加入人工校正循环。

**🔧 技术方法**

核心技术包括：2D 目标检测（Grounding‑DINO fine‑tuned）、2D 分割（SAM）、2D‑>3D 投影与融合、半自动管道重建（PipeRunner）、图结构构建（节点为设备与管道），图匹配（SLOTAlign），人工一致性纠正。

**📊 数据集**

使用的数据集为扩展版 IRIS，包含：~300 张高分辨率全景图、LiDAR 点云、CAD 模型、3D 管道路由、约 6000 个 2D 盒子、47000 个分割掩码以及 PDF 形式的 P&ID。

**📈 对比分析**

在一真实工业环境的案例研究中，经过三步流程后，SLOTAlign 能实现完美对齐（即使示意图中存在隐藏物体亦可定位），且对分割误差或图结构扰动具有鲁棒性。实验结果表明对齐时间显著减少，且匹配准确率接近 100%。

**⚠️ 局限性**

局限性包括：①管道分割仍依赖半自动工具，无法一次性完全自动化；②对某些设备（如泵）分割仍需人工；③数据集规模有限，仅覆盖单一工厂场景，缺乏多样化场景验证；④对大规模工业设施的可扩展性尚未充分评估。

---

## 119. Perspectives - Interactive Document Clustering in the Discourse Analysis Tool Suite

**arXiv ID:** 2602.15540 | [PDF](https://arxiv.org/pdf/2602.15540v1)

**作者:** Tim Fischer `[一作]` (University of Hamburg), Chris Biemann `[通讯]` (University of Hamburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并推出了 Perspectives——一个基于 DATS 的交互式扩展，支持面向特定维度的文档聚类、LLM 驱动的文档重写、指令式嵌入、HITL 迭代细化以及可视化的 2D 文档地图。

**💡 创新点**

创新点包括：① 将 LLM 重写与指令嵌入相结合的可调节聚类管线；② 引入多种交互式细化操作（合并、拆分、添加/删除聚类、修改文档归属）；③ 利用少量标注数据进行 SetFit/LoRA 微调，进一步对齐嵌入空间；④ 在 DATS 平台内实现完整的端到端工作流，强化人机协作与可解释性。

**🔧 技术方法**

主要技术：Gemma 3 LLM（重写、生成聚类名）；multilingual-e5-large-instruct（指令嵌入）；UMAP+HDBSCAN（降维+聚类）；c‑TF‑IDF、LLM 生成关键词与描述；SetFit + LoRA 微调；前端 React+Plotly.js，后端 FastAPI+RQ+vLLM+LiteLLM；全流程 GPU 加速。

**📊 数据集**

使用的数据集包括：Amazon Product Reviews、Spotify Songtexts、20 Newsgroups、GVFC、Blurbs、Israel‑Palestine、News Bias 等 7 个涵盖情感、主题、立场、框架、偏见等任务的文本集合。

**📈 对比分析**

采用 KNN 准确率作为可视化聚类质量的评估指标。实验显示：① 指令式嵌入 (+inst) 在 8/9 数据集上优于无指令；② 文档重写（摘要或关键短语）在 6/9 数据集上带来提升；③ 少量标注微调在绝大多数场景提升 2–3 分，尽管增幅有限；整体表现优于传统无监督方法，验证了可调节聚类管线的有效性。

**⚠️ 局限性**

局限性：① 微调导致重新嵌入和重聚类，打破用户对 2D 空间的认知，正在研发历史轨迹功能以缓解；② 仅使用单一 LLM 与嵌入模型，其他模型可能表现不同；③ 少样本微调对样本选择敏感，未探索最佳样本组合；④ 文档重写仅使用摘要/关键词提示，未进行广泛的提示工程；⑤ 评估未覆盖多模态数据与更大规模数据集。

---

## 120. DAV-GSWT: Diffusion-Active-View Sampling for Data-Efficient Gaussian Splatting Wang Tiles

**arXiv ID:** 2602.15355 | [PDF](https://arxiv.org/pdf/2602.15355v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11897 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种结合扩散模型和主动视角采样的DAV-GSWT框架，用以在极少输入图像的情况下高效生成可无缝拼接的Gaussian Splatting Wang Tiles；

**💡 创新点**

创新点在于引入层级不确定性量化与扩散先验驱动的视角优先级判定，以及基于语义加权的图割边界优化，实现数据高效采集与拼接连续性；

**🔧 技术方法**

采用扩散概率模型（Zero‑1‑to‑3 XL）、主动视角规划、Wasserstein‑2与LPIPS混合不确定性度量、语义分割辅助图割、以及多级LOD的实时渲染技术；

**📊 数据集**

使用十个场景的数据集，其中五个是合成Blender地形（约100视图），五个是真实无人机采集（约200视图），并在仅从8视图开始的极限采样条件下进行实验；

**📈 对比分析**

与全景采样和传统稠密重建基线对比，DAV‑GSWT在保持近乎全景级别的视觉质量（PSNR≈29.4 dB、Seam‑LPIPS≈0.031）同时将所需采集视图数从200缩减至约80（≈3迭代×20视图），并实现5–15 ms的实时渲染；

**⚠️ 局限性**

局限性包括对扩散模型先验的依赖、在极端稀疏视角下可能出现纹理失真、以及语义分割误差对图割结果的影响，且在高复杂度场景下仍需更多计算资源。

---

## 121. AI Sessions for Network-Exposed AI-as-a-Service

**arXiv ID:** 2602.15288 | [PDF](https://arxiv.org/pdf/2602.15288v1)

**作者:** Merve Saimler `[一作]` (Ericsson), Mohaned Chraiti `[通讯]` (Sabancı University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

设计并实现了网络可曝光的AIaaS架构（NE‑AIaaS），定义了AI Session和AI Service Profile两大合同对象，并给出了端到端的协议流程（发现、AI分页、事务性绑定、服务与迁移）以及基于仿真的性能评估。

**💡 创新点**

创新点在于：①引入会话级别的可强制合同对象，将模型、执行位置、QoS流与授权/计费绑定，形成可审计的尾部延迟与移动连续性保证；②提出事务性计算与网络资源共预留机制，避免部分状态；③实现 make‑before‑break 迁移实现移动时不间断服务。

**🔧 技术方法**

使用的技术包括：5G QoS Flow（QFI）和PCC、CAPIF、ETSI MEC、NWDAF、A1接口；尾部延迟与概率建模；Monte‑Carlo 仿真框架；事务性控制与协议设计。

**📊 数据集**

本文没有使用公开的真实数据集，而是通过仿真生成服务器排队、推理时延与网络时延的随机样本进行评估。

**📈 对比分析**

与传统的端点式AIaaS基线对比；评估指标为99th百分位延迟、ASP违规概率和移动中断概率。实验结果表明：NE‑AIaaS 在高负载下99th百分位延迟显著低于基线，违规概率下降且在移动场景中中断概率几乎为零。

**⚠️ 局限性**

局限性包括：需进一步验证真实网络环境中的实现细节；迁移过程中的状态迁移与隐私保护仍有挑战；多域跨运营商部署的互操作性与标准化工作仍在进行中。

---

## 122. Multi-Agent Home Energy Management Assistant

**arXiv ID:** 2602.15219 | [PDF](https://arxiv.org/pdf/2602.15219v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 123. The Information Geometry of Softmax: Probing and Steering

**arXiv ID:** 2602.15293 | [PDF](https://arxiv.org/pdf/2602.15293v1)

**作者:** Kiho Park `[一作]` (University of Chicago), Victor Veitch `[通讯]` (University of Chicago)

**通讯引用:** 2309 | [OpenAlex ID](https://openalex.org/A5077406767)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了softmax模型表示空间的内在信息几何结构，提出了基于Bregman几何的双向驱动方法，以在保持非目标概念不变的前提下调节目标概念。

**💡 创新点**

将信息几何与线性表示假设结合，引入双向驱动（dual steering）并证明其最小化非目标概念漂移；同时阐明原始欧氏驱动的不足。

**🔧 技术方法**

使用Bregman（双重平坦）几何、KL散度极小化、线性探针、正则化牛顿迭代、对数归一化函数的梯度与Hessian；并在实验中应用Gemma-3-4B和MetaClip-2模型。

**📊 数据集**

在Gemma-3-4B上使用AllenAI C4生成的上下文，在MetaClip-2上使用合成对象集合和COCO图像数据集。

**📈 对比分析**

将双向驱动与传统欧氏驱动在提升目标概念概率、保持非目标分布以及KL与排名差异等三个鲁棒性指标上进行对比；实验表明双向驱动在所有指标上均优于欧氏驱动，尤其在保持非目标概念分布方面显著提升。

**⚠️ 局限性**

方法依赖于高质量的线性探针和概念可因式分解假设；在概率过于集中时Hessian秩缺失会导致驱动失败；在非softmax层的实际控制和跨模型通用性仍需进一步研究。

---

## 124. AI-Paging: Lease-Based Execution Anchoring for Network-Exposed AI-as-a-Service

**arXiv ID:** 2602.15286 | [PDF](https://arxiv.org/pdf/2602.15286v1)

**作者:** Merve Saimler `[一作]` (Ericsson), Mohaned Chraiti `[通讯]` (Sabancı University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出并实现了AI-Paging——一种基于租约的 AI-as-a-Service 执行锚定机制，能够将应用意图转化为可强制执行、可审计的服务实例。

**💡 创新点**

创新点在于：①引入了显式租约（lease）来授权用户平面流量转发和 QoS 绑定；②采用 make‑before‑break 迁移语义保证迁移过程无服务中断；③将服务身份与实际执行点解耦，提供可跨域可审计的服务标识。

**🔧 技术方法**

使用技术包括：现有 5G 控制平面接口（如 CAPIF、NEF）、服务层编排（类似 SEAL/AIMLE）、可编程用户平面流量转发与 QoS 处理、NWDAF 预测与监控、以及基于租约的强制状态管理。

**📊 数据集**

论文未使用特定 AI 数据集，而是通过模拟的网络动态（移动、负载、失效）和抽象的 AI 执行节点（边缘/云）来验证机制。

**📈 对比分析**

与两种基线（固定端点服务、无租约的最佳努力转发）对比：AI-Paging 的事务延迟与基线相近；在迁移、负载峰值与失效场景下，其连续性误差率降至零，恢复成功率更高；租约失效后用户平面状态被及时撤销，违约率为 0%；证据生成的开销保持在可控范围。

**⚠️ 局限性**

局限性包括：①实验仅在模拟环境中验证，缺乏真实运营商网络部署验证；②依赖现有流量分类机制，若分类不精确可能导致租约与流量不匹配；③未涉及多域协作细节，如跨运营商租约互认与证据共享；④对大规模高频迁移的性能评估尚不充分。

---

## 125. EventMemAgent: Hierarchical Event-Centric Memory for Online Video Understanding with Adaptive Tool Use

**arXiv ID:** 2602.15329 | [PDF](https://arxiv.org/pdf/2602.15329v1)

**作者:** Siwei Wen `[一作]` (Beihang University), Wenjun Wu `[通讯]` (Beihang University)

**通讯引用:** 9025 | [OpenAlex ID](https://openalex.org/A5060858375)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种主动式在线视频理解框架EventMemAgent，利用分层记忆与多粒度感知工具实现对无限流媒体的实时感知与长程推理；

**💡 创新点**

创新点在于：①引入事件级短期记忆与事件中心长期记忆的双层结构，结合事件粒度的容器采样，解决传统固定窗口导致的信息衰减与语义碎片化；②构建多粒度感知工具箱（记忆检索、OCR、目标检测），实现主动、迭代式证据采集；③使用Agentic强化学习（GRPO）端到端优化推理路径和工具调用策略，将推理与工具使用内化为代理自身能力；

**🔧 技术方法**

主要技术包括：事件分割（基于灰度直方图的Pearson相关系数检测）、容器采样（Reservoir Sampling）、分层结构化长期记忆（事件元组）、多粒度感知工具（Memory Search、OCR、Object Detection）、Agentic RL（GRPO）以及基于Qwen3‑VL‑8B‑Instruct的多模态大语言模型；

**📊 数据集**

使用了OVO‑Bench和StreamingBench两大在线视频问答基准；训练时采用10K MovieChat样本进行RL；在工具实现上使用Grounding DINO（目标检测）和Deepseek‑OCR；

**📈 对比分析**

对比方法包括：专有模型（GPT‑4o、Gemini‑1.5‑Pro）、高级开源离线模型（InternVL‑V2、Qwen3‑VL、LLaVA‑Video）、最近开源在线模型（Flash‑VStream、Dispider）以及在线视频代理（StreamAgent）。在OVO‑Bench上，EventMemAgent在仅使用32帧输入时平均准确率60.75%，超过所有开源模型并优于GPT‑4o（59.54%）；在StreamingBench上亦表现出竞争力，说明其在实时感知与长程推理上的优势；

**⚠️ 局限性**

局限性包括：①需要在多轮强化学习中进行调优，训练成本高；②依赖外部工具（OCR、检测）对性能影响显著，工具错误会直接影响答案质量；③目前只针对视频问答任务，尚未验证对更广泛视觉推理或多模态场景的泛化能力；④短期记忆固定容量（32帧）在极长视频场景下仍可能导致信息丢失；

---

## 126. Computer Science as Infrastructure: the Spine of the Lean Computer Science Library (CSLib)

**arXiv ID:** 2602.15078 | [PDF](https://arxiv.org/pdf/2602.15078v1)

**作者:** Christopher Henson `[一作]` (Drexel University), Fabrizio Montesi `[通讯]` (University of Southern Denmark)

**通讯引用:** 1898 | [OpenAlex ID](https://openalex.org/A5000566520)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构建了 Lean Computer Science Library（LeanCSLib），提供统一的共享抽象（如 ReductionSystem、LTS）、可计算定义、上下文与同构类型类，配合自动化、CI/测试流水线以及与 Mathlib 的无缝集成，并完成了语言基础、行为等价、CCS、λ-演算等核心技术实现。

**💡 创新点**

创新点包括：① 在 Lean 生态中首次实现统一的操作语义抽象和可计算接口；② 将自动化作为设计原则，利用 `grind` 等新型 tactic 生成大量证明；③ 通过类型类和元编程实现可插拔的上下文与同构抽象；④ 设计完整的 CI/linters/testing 机制，确保与 Mathlib 的兼容与可维护性；⑤ 在实践中验证了上述抽象的有效性（CCS、λ-演算、系统 F 等）。

**🔧 技术方法**

使用技术：Lean 4、Mathlib、`grind`、`mergeWithGrind` 等自定义 tactic、linters、CI 流水线、元编程、locally nameless 绑定、类型类系统、可计算定义、可插拔抽象接口。

**📊 数据集**

该工作不依赖传统数据集，主要以理论示例（如 NFA→DFA 转换、CCS 并发流程、λ-演算）作为验证与演示。

**📈 对比分析**

性能评估：在 338 条声明中有 314 条使用 `grind`，平均每条证明节省 7.1 行代码；bisimilarity 等证明节省 15–39 行；System F formalization 相较原始实现减少约 45% LoC。CI/测试覆盖率高，lint 与自动化失败及时反馈，提升了库的稳定性与可维护性。

**⚠️ 局限性**

局限性：① 对高级绑定与 LTS（如 π‑算子）支持尚不完整；② 需要进一步扩展接口以支持更复杂的语言与算法；③ 对大规模程序验证的性能与可扩展性仍需实测评估；④ 依赖 Mathlib 生态，若 Mathlib 发生重大变化需同步维护。

---

## 127. Multi-Objective Coverage via Constraint Active Search

**arXiv ID:** 2602.15595 | [PDF](https://arxiv.org/pdf/2602.15595v1)

**作者:** Zakaria Shams Siam `[一作]` (University at Albany, State University of New York), Chong Liu `[通讯]` (University at Albany, State University of New York)

**通讯引用:** 32237 | [OpenAlex ID](https://openalex.org/A5115602404)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 MOC-CAS，一种针对多目标阈值约束下的覆盖搜索方法，旨在快速发现少量且分布均匀的可行结果集合，支持后续决策。

**💡 创新点**

创新点在于将覆盖目标转化为在输出空间的增量覆盖收益，并采用基于 UCB 的乐观可行性门控与平滑核逼近，实现可微分、可梯度优化的采样函数；同时引入分层平衡的多目标 GP 建模与自适应覆盖分辨率。

**🔧 技术方法**

采用独立高斯过程进行多目标建模，使用 UCB 预测作为乐观可行性评估，构造增量覆盖收益函数，并通过高斯核与 Probit 门实现平滑近似，利用 L‑BFGS 等梯度方法进行内部优化。

**📊 数据集**

在大型蛋白靶点虚拟筛选数据集上评估，包括 SARS‑CoV‑2 3CLPro 与三种癌症靶点（6T2W、RTCB、WRN），每个靶点均包含约 1 百万条 ZINC15 化合物的分子结构与对接评分。

**📈 对比分析**

与四种基线（随机、One‑Step、Straddle、MOO+Cluster）对比，MOC‑CAS 在填充距离、累计可行样本数及正样本曲线下面积（AUP）等指标上均表现更好，尤其在早期发现和分布均匀性方面显著优于其它方法。

**⚠️ 局限性**

局限性包括：采用独立 GP 仅捕捉单目标先验，未考虑目标间相关性；覆盖分辨率 r 与乐观度 β_t 需手工调参，且在高维/多目标情形下的扩展与理论收敛性尚未系统证明。

---

## 128. Size Transferability of Graph Transformers with Convolutional Positional Encodings

**arXiv ID:** 2602.15239 | [PDF](https://arxiv.org/pdf/2602.15239v1)

**作者:** Javier Porras-Valenzuela `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**通讯引用:** 16175 | [OpenAlex ID](https://openalex.org/A5078862959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究并证明了使用GNN作为位置编码的图Transformer在不同图规模下可迁移，且通过稀疏注意力实现可扩展性。

**💡 创新点**

创新点在于：①将图Transformer的可迁移性理论与Manifold Neural Network相结合；②证明GNN位置编码传递可迁移性给Transformer；③提出稀疏图Transformer（SGT）并证明其可迁移性。

**🔧 技术方法**

技术手段包括：图Transformer架构、RPEARL（基于GNN的随机特征位置编码）、稀疏k-hop注意力掩码、Manifold极限理论与收敛定理。

**📊 数据集**

实验使用ArXiv-year、Reddit、Snap‑Patents、OGBN‑MAG等大规模节点分类数据集，以及挪威地形点云数据集做最短路径距离估计。

**📈 对比分析**

通过将训练图规模逐步增大至完整图并与完整图训练的模型对比，发现稀疏图Transformer在仅用10%训练图时即可达到90%+完整模型性能，且与传统GNN和密集Transformer性能相当或更优；在地形数据上，SGT在大规模图上仍保持低误差，表现出良好的可迁移性。

**⚠️ 局限性**

局限性在于：仍需在不同类型图（如非Manifold或极端稀疏/密集图）验证理论；对位置编码的依赖要求GNN具有良好收敛性；掩码参数（k-hop阈值）需要手工调优，且对极大图仍有一定计算开销。

---

## 129. Efficient Generative Modeling beyond Memoryless Diffusion via Adjoint Schrödinger Bridge Matching

**arXiv ID:** 2602.15396 | [PDF](https://arxiv.org/pdf/2602.15396v1)

**作者:** Jeongwoo Shin `[一作]` (Seoul National University), Jaemoo Choi `[通讯]` (Georgia Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种两阶段的Adjoint Schrödinger Bridge Matching (ASBM) 框架，通过先学习非记忆式前向控制（数据到能量采样），再用得到的最优耦合监督后向动态，实现高质量图像生成和一次性蒸馏为单步生成器。

**💡 创新点**

核心创新在于将Schrödinger桥的前向后向优化分离，利用Stochastic Optimal Control实现数据到能量采样，获得更直线、低NFE的最优耦合；通过Bridge Matching在此耦合下训练后向控制，显著提升稳定性和效率；同时，该直线轨迹为一次性蒸馏提供了天然的结构优势。

**🔧 技术方法**

使用技术包括Schrödinger桥、Stochastic Optimal Control、Adjoint Matching、Corrector Matching、Bridge Matching、非记忆式SDE、Heun/ODE求解、路径KL匹配、Tweedie公式等。

**📊 数据集**

实验数据集主要为CIFAR-10（像素空间）和FFHQ（LDM潜在空间），并在这些数据集上评估。

**📈 对比分析**

与传统Score SDE、SB-FBSDE、DSBM以及Score Distillation Sampling（SDS）等方法比较，ASBM在FID上显著更优，所需的采样步数（NFE）大幅降低（如20步即可），训练周期更短，蒸馏后单步生成器的recall/precision也更好。

**⚠️ 局限性**

局限性：目前仅在标准图像生成（低分辨率、无条件）任务中验证，缺乏大规模高分辨率或条件生成的实验；此外，虽然前向控制更轻量，但整体仍需显著计算资源，且在更复杂的数据分布上是否保持同样优势尚待进一步研究。

---

## 130. Making Large Language Models Speak Tulu: Structured Prompting for an Extremely Low-Resource Language

**arXiv ID:** 2602.15378 | [PDF](https://arxiv.org/pdf/2602.15378v1)

**作者:** Prathamesh Devadiga `[一作]` (Lossfunk), Paras Chopra `[通讯]` (Lossfunk)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过结构化提示工程，利用语法规则、负向约束、罗马化和自检，提升大型语言模型在极低资源语言图鲁语的会话能力，无需任何参数更新。

**💡 创新点**

创新点在于：① 在无训练数据的情况下，仅凭提示即可显著减少高资源语言（卡纳达语）的词汇污染；② 负向约束的系统化使用在三大模型中均实现12–18个百分点的污染降低；③ 通过语法文档与自检层级化提示验证模型真正利用了所给语法。

**🔧 技术方法**

技术主要包括：结构化提示层次化（身份、负向约束、语法规则、示例、验证）、罗马化方案、自动化语法检查、误差分析与人类评估。

**📊 数据集**

使用的数据集为：① 200句手工标注的图鲁语测试集；② 100句保留测试集（覆盖不同语法现象）；③ 3名母语图鲁语说话者提供的语法规则与评估；④ 对照的50词卡纳达词表作为污染监测。

**📈 对比分析**

对比方法为在同一提示层次下分别评估三大模型（Gemini 2.0 Flash、GPT‑4o、Llama 3.1 70B），通过自动化污染率和语法准确率以及三名评审的Likert评分来量化。结果显示：完整系统在语法准确率达到78–85%，污染率仅为5–7%，相比基线版本提升12–18个百分点，表明提示工程在低资源语言上具有可观的实用效果。

**⚠️ 局限性**

局限性包括：仅验证了单一图鲁语变体，缺乏对其他方言或更复杂语言的推广；污染评估仅限于预设的50词列表，未覆盖潜在的结构性干扰；提示长度较大（约2800 token），不利于资源受限环境；生成的文本仍缺乏自然流畅度，实用性受限于非高风险场景；需要持续的社区协作以确保脚本和语言规范的正确使用。

---

## 131. Selective Perception for Robot: Task-Aware Attention in Multimodal VLA

**arXiv ID:** 2602.15543 | [PDF](https://arxiv.org/pdf/2602.15543v1)

**作者:** Young-Chae Son `[一作]` (Dongguk University), Soo-Chul Lim `[通讯]` (Dongguk University)

**通讯引用:** 990 | [OpenAlex ID](https://openalex.org/A5023192002)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种动态信息融合框架，通过任务感知的路由器在实时分析文本提示和手腕摄像头输入后，选择最有用的多视角视觉信息来驱动机器人完成任务。

**💡 创新点**

创新点在于结合文本提示与实时视觉观察预测视角重要性，使用轻量化自适应路由器和任务感知注意机制，实现了任务相关的动态计算削减与信息选择。

**🔧 技术方法**

采用Vision‑Language‑Action模型、轻量级路由网络、任务感知注意模块，以及基于VLM的自动标注管线来训练路由器。

**📊 数据集**

基于自建的多视角摄像头与文本提示的机器人操作数据集（涵盖旋转阀、按键等任务）。

**📈 对比分析**

与传统静态融合VLA模型对比，路由器框架将推理时间降低约30%–50%，控制成功率提升约10%–15%。

**⚠️ 局限性**

受限于路由器预测误差、对特定视角的依赖以及需要大量标注数据，尚未验证在更大范围多任务环境下的鲁棒性。

---

## 132. A Weighted-to-Unweighted Reduction for Matroid Intersection

**arXiv ID:** 2602.15702 | [PDF](https://arxiv.org/pdf/2602.15702v1)

**作者:** Aditi Dudeja `[一作]` (Chinese University of Hong Kong), Mara Grilnberger `[通讯]` (University of Salzburg)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种通用的无权到有权交换簇问题的近似算法框架，能够把任意 α‑近似无权交集算法转化为 α(1‑ε) 近似有权交集算法，且仅在运行时间上增加一个 log W 的因子（W 为权值比）

**💡 创新点**

创新点在于：①构造了一个三步转换流程（aspect‑ratio 降低 → 有权到无权转换（展开）→ 反展开），实现了从无权到有权的非自适应、通用化减 ；②证明了展开/反展开能够保持独立性结构和权值关系；③将该框架迁移到流式、通信和并行模型，获得了此前未有的近似结果

**🔧 技术方法**

主要技术包括：
- 线性规划与其对偶的链式结构分析
- aspect‑ratio 降低（将权值划分为几何间隔的区间并丢弃部分区间）
- matroid 展开（把每个元素复制 w(e) 次，构造等价无权 matroid）
- 反展开（从无权解中提取一个近似有权解）
- 贪心组合与全局计费（保证权值损失 ≤4ε）
- 对流/通信模型的空间/查询成本分析

**📊 数据集**

该工作为理论研究，无实测数据集；所有结果均基于计算复杂度与查询/空间量化

**📈 对比分析**

与之前的工作相比，已知无权和有权之间存在显著性能差距；本框架在无权已知近似算法的基础上，几乎保持相同的查询/空间复杂度，只额外乘上 log W（以及 ε 相关的多项式）因子；在流式、通信和并行模型中得到的空间/通道复杂度均优于或与最优已知结果相当

**⚠️ 局限性**

主要限制是：
- 对 ε 的依赖极大，框架中出现 ε^{-O(ε^{-1})} 的增量（尤其是 aspect‑ratio 降低的 γ_ε 因子）；
- 该方法是非自适应的，在某些模型（如单 pass 流式或一向通信）中无法进一步降低 ε 依赖；
- 只适用于 (1‑ε) 近似范围，无法直接实现更高精度的近似。

---

## 133. NeuroSymActive: Differentiable Neural-Symbolic Reasoning with Active Exploration for Knowledge Graph Question Answering

**arXiv ID:** 2602.15353 | [PDF](https://arxiv.org/pdf/2602.15353v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11897 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 NeuroSymActive 框架，用于知识图谱问答，结合可微神经符号推理层与主动价值引导的探索控制器，实现多跳推理的高效与可解释。

**💡 创新点**

创新点包括：① 将软统一（soft‑unification）符号模块与神经路径评估器融合，形成可微神经符号推理层；② 引入基于信息增益的主动查询机制和不确定性阈值，显著减少 KG 查询与模型调用；③ 采用可微蒙特卡洛树搜索（MCTS）和进化宽化（Progressive Widening）实现可微搜索，支持端到端梯度传播。

**🔧 技术方法**

核心技术：可微神经符号层（Differentiable Inductive Logic Layer，DILL）、Gumbel‑Softmax 软化采样、进化宽化的可微 MCTS、贝叶斯不确定性估计与信息增益预测、知识适配器（Knowledge Adapter）将路径向量映射到冻结 LLM 的提示空间。

**📊 数据集**

使用公开的 Freebase 基础数据集：WebQuestionsSP（约4,700个问题）和 ComplexWebQuestions（约34,700个问题）进行评估。

**📈 对比分析**

与传统 KGQA 基线（如 KV‑Mem、EmbedKGQA、NSM、GraftNet、TransferNet 等）以及 LLM 直接提示方法和混合 LLM+KG 系统（ToG、StructGPT、KnowledgeNavigator 等）对比。实验表明在 WebQSP 上 Hits@1 87.1% / CWQ 上 62.5%，相较于 LightPROF 提升约 3–5%，相较于其他基线提升 5–10%，并且在推理时的 KG 查询次数和 LLM 请求次数均显著降低。

**⚠️ 局限性**

局限性：① 对人类主动标注的依赖，若预算不足仍会出现检索/推理错误；② 在极长路径或图结构极为稠密的情况下，信息增益预测可能失真；③ 生成阶段仍受 LLM 限制，存在提示模糊导致答案错误的风险；④ 对动态/时序知识图的适配尚未深入。

---

## 134. Neural Network-Based Parameter Estimation of a Labour Market Agent-Based Model

**arXiv ID:** 2602.15572 | [PDF](https://arxiv.org/pdf/2602.15572v1)

**作者:** M Lopes Alves `[一作]` (University of Oxford), Anisoara Calinescu `[通讯]` (University of Oxford)

**通讯引用:** 1263 | [OpenAlex ID](https://openalex.org/A5020492975)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对劳动力市场 ABM 的参数估计进行评估，并测试了 SBI4ABM 框架。

**💡 创新点**

首次在 ABM 参数估计中结合 NPE 与 RNN 学习摘要统计，并验证其相较手工统计的优势与校准问题。

**🔧 技术方法**

采用神经网络后验估计 (NPE)、归一化流、RNN 自动摘要以及模拟基推断 (SBI)。

**📊 数据集**

使用合成劳动力市场数据以及基于美国劳动力调查的真实人口与自动化概率数据。

**📈 对比分析**

通过比较手工与 NN 学习摘要统计下的后验分布、对角矩阵与 SBC 校准，发现 NN 产生更集中的后验但校准偏差；计算时间随工种数量线性增长。

**⚠️ 局限性**

限制在于合成数据过于简化、缺乏真实微观转移数据、以及内存瓶颈导致无法完成完整的美国劳动力市场实验。

---

## 135. Fair Correlation Clustering Meets Graph Parameters

**arXiv ID:** 2602.15683 | [PDF](https://arxiv.org/pdf/2602.15683v1)

**作者:** Johannes Blaha `[一作]` (TU Wien), Simon Wietheger `[通讯]` (TU Wien)

**通讯引用:** 379 | [OpenAlex ID](https://openalex.org/A5100374721)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究公平相关聚类（Fair Correlation Clustering）问题的参数化复杂度，给出在多种图结构参数（顶点覆盖数、树宽、树深以及公平分块大小）下的可解性结果。

**💡 创新点**

创新点在于首次把公平聚类问题纳入参数化框架，证明了在顶点覆盖数、树宽+分块大小、树深+分块大小等参数化组合下的FPT，可解的最小分块大小可为2，从而解决了聚类（Cluster Editing）在树宽下的FPT未定问题。

**🔧 技术方法**

主要技术包括：结构化分析（证明大集群可限制为固定大小）、分支与匹配（顶点覆盖参数）、动态规划（树宽与分块大小参数）、强制性约简与整数线性规划（树深+分块大小参数）以及图分解与同构类计数。

**📊 数据集**

无实验数据集，全部为理论算法与复杂度分析。

**📈 对比分析**

方法相较于已有的仅提供近似解的公平聚类研究，显著提升了理论可解性，证明在多种参数化条件下问题是可多项式时间或FPT的；与以前的NP‑/P‑难结果形成对比，显示参数化视角下的可解性边界。

**⚠️ 局限性**

局限性：对树宽+分块大小大于2的情况仍未决定；此外，在极简化输入（如两色树宽4）下问题仍保持P‑/NP‑难；方法主要针对无权图，带权或更一般约束的公平聚类尚未覆盖。

---

## 136. A Comparison of Bayesian Prediction Techniques for Mobile Robot Trajectory Tracking

**arXiv ID:** 2602.15354 | [PDF](https://arxiv.org/pdf/2602.15354v1)

**作者:** Jose Luis Peralta-Cabezas `[一作]`, Marcelo Guarini-Hermann `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文对十种贝叶斯预测方法（Kalman 及其变体、Sigma‑Point 过滤器、粒子过滤器及其变体）在三轮全向机器人轨迹跟踪中的性能进行了系统比较。

**💡 创新点**

创新点在于：①针对非高斯噪声场景，首次将 Improbability Filter 与多种 Kalman 变体结合；②在同一实验平台上对多种滤波器在预测误差、计算成本和鲁棒性方面进行统一评估；③通过实测验证，证明 EKF+IF 在实时应用中既能保持良好精度又具备最快速度。

**🔧 技术方法**

使用的技术包括：非线性系统的状态空间模型、Kalman、EKF、UKF、CDKF、DD1/DD2、SR‑UKF、SR‑CDKF、粒子过滤器（PF）、Sigma‑Point 粒子过滤器（SPPF）和高斯混合 Sigma‑Point 粒子过滤器（GMSPPF），以及 Improbability Filter 进行异常观测剔除。

**📊 数据集**

实验数据集：六种典型的机器人参考轨迹（直线、圆弧、加速/减速、转弯等）在仿真中重复20次；随后在实际 F‑180 机器人上进行直线和弧形运动实验，加入人工非高斯噪声（伪目标干扰）。

**📈 对比分析**

比较方法：对每种滤波器在 1、4、8 帧预测（即 0.033s、0.133s、0.267s）下的轨迹位置和航向的 RMSE 进行统计，并与最快算法的计算时间做相对比。结果表明，PF 在所有预测 horizon 上误差最小，但相对耗时最高；EKF+IF 在保持误差第二/第三优的同时，计算时间是最快的；其他 SPKF 变体和 GMSPPF 既不如 PF 精度，也不如 EKF+IF 速度。

**⚠️ 局限性**

局限性：①粒子过滤器对粒子数高度敏感，过少会失真；②Improbability Filter 需要经验确定阈值；③SPKF 由于 sigma‑point 的构造，若协方差矩阵失正定需手动重置，影响滤波稳定性；④所有 Kalman 变体仍假设噪声近似 Gaussian，导致对极端非高斯噪声的鲁棒性有限。

---

## 137. On inferring cumulative constraints

**arXiv ID:** 2602.15635 | [PDF](https://arxiv.org/pdf/2602.15635v1)

**作者:** Konstantin Sidorov `[一作]` (Delft University of Technology), Konstantin Sidorov `[通讯]` (Delft University of Technology)

**通讯引用:** 66 | [OpenAlex ID](https://openalex.org/A5052280970)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种预处理方法，通过将累积约束重新表述为占用向量上的线性不等式，发现任务组之间的互斥关系并利用提升（lifting）技术推导新的累积约束，从而在不搜索的情况下增强 CP 求解器的推理能力。

**💡 创新点**

创新点在于将多资源交互建模为全局线性切割面，并结合占用向量表示和提升算法自动推导新的约束，使得预处理即可捕捉跨资源的冲突而无需在搜索期间额外探测。

**🔧 技术方法**

主要技术包括占用向量表示、覆盖（cover）生成、整数规划中的提升（lifting）以及基于 Gurobi 的子问题求解，随后将推导的约束作为新的累计约束加入模型。

**📊 数据集**

实验使用 MiniZinc benchmark 中的 736 个 RCPSP 实例和 PSPLIB 的 349 个 RCPSP/max 实例，对比基准 CP 求解器 Pumpkin 与 CP‑SAT。

**📈 对比分析**

与基线求解器相比，预处理后在约 90% 的实例中提升了搜索效率；在约 80% 的实例中降低了双积分（dual integral），仅在极少数实例出现轻微性能下降，整体表现优于 Sidorov 等人提出的单资源“clique”方法。

**⚠️ 局限性**

局限性包括提升子问题求解开销在大规模实例（>500 任务）上可能显著，且方法仅适用于持续不变的持续时间和资源消耗，无法直接处理多模式调度等非线性情形。

---

## 138. LEADER: Lightweight End-to-End Attention-Gated Dual Autoencoder for Robust Minutiae Extraction

**arXiv ID:** 2602.15493 | [PDF](https://arxiv.org/pdf/2602.15493v1)

**作者:** Raffaele Cappelli `[一作]` (University of Bologna), Matteo Ferrara `[通讯]` (University of Bologna)

**通讯引用:** 2813 | [OpenAlex ID](https://openalex.org/A5010414428)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种轻量级端到端卷积神经网络LEADER，直接将原始指纹图像映射到细节点位置、方向、类型，并内嵌NMS与角度解码，无需手工预处理或后处理。

**💡 创新点**

采用双autoencoder结构与注意力门控、Castle‑Moat‑Rampart（CMR）自适应真值编码，将后处理逻辑完全内置在网络中，实现仅0.9M参数的高精度指纹细节点提取，并在跨域暗指纹上实现零样本泛化。

**🔧 技术方法**

使用轻量化分离卷积、逆瓶颈卷积、膨胀卷积、混合池化、注意力门控、全图NMS、Cartesian‑to‑Polar方向解码、多任务损失、CMR编码以及PCA可视化解释等技术。

**📊 数据集**

训练集包含FVC2002、FVC2004、FFE等多种传感器的普通指纹；测试集为FVC2002 DB1-A（普通）和NIST SD27（暗指纹）。

**📈 对比分析**

在统一评测管道下与多款开源深度模型、COTS引擎和传统方法比较，LEADER在FVC2002 DB1-A的F1最高0.92，暗指纹0.71，类型敏感下亦保持优势；样本级排名平均1.43/2.07，且GPU/CPU推理仅需15/322ms。

**⚠️ 局限性**

在极度受损的暗指纹（强噪声、深裂缝）仍难以恢复完整细节点，模型对极端环境的鲁棒性有限，未来需自监督预训练提升缺失结构重建能力。

---

## 139. An effective Genetic Programming Hyper-Heuristic for Uncertain Agile Satellite Scheduling

**arXiv ID:** 2602.15070 | [PDF](https://arxiv.org/pdf/2602.15070v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 140. *-PLUIE: Personalisable metric with Llm Used for Improved Evaluation

**arXiv ID:** 2602.15778 | [PDF](https://arxiv.org/pdf/2602.15778v1)

**作者:** Quentin Lemesle `[一作]` (Univ Rennes), Damien Lolive `[通讯]` (Univ of South Brittany)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于困惑度的LLM评测框架*（*‑PLUIE），通过自定义任务特定的Yes/No提问，直接得到置信度得分，无需生成文本。

**💡 创新点**

创新点在于：①把原始的单一困惑度指标扩展为可适配多种语义任务的通用框架；②设计任务专属提示（Fr、Net、Rev）提升与人类判断的一致性；③在保持高可解释性的同时，使评测速度显著快于传统的基于输出的llmjudge。

**🔧 技术方法**

技术手段包括：LLM困惑度计算、少量示例提示工程、阈值校准、Pairwise/Accuracy/V/κ等评价指标；使用Llama‑70B等大模型与中等规模模型对比。

**📊 数据集**

使用的数据集：法语改写对（33.6% 正例）、Nile 翻译对（NEAT, 436 条），以及科学文本修订评估集 ParaReval（对话式人类打分）。

**📈 对比分析**

与传统相似度指标（BERTScore、BLEURT等）及输出式llmjudge（Yes/No、Pairwise、Score）对比，*‑PLUIE 在所有任务中均达成或超过最优阈值下的 F1、kappa、Cramér V 等指标，且计算时间比输出式方法缩短约 7–8 倍。

**⚠️ 局限性**

局限性包括：困惑度模型未针对任务微调；仅处理 Yes/No 形式提示，未覆盖多类或连续分数任务；实验主要集中在英语/法语，尚未验证对形态丰富或语法差异较大的语言；更大模型可能进一步提升效果。

---

## 141. Quantum Optimization for Access Point Selection Under Budget Constraint

**arXiv ID:** 2602.15049 | [PDF](https://arxiv.org/pdf/2602.15049v1)

**作者:** Mohamed Khalil Brik `[一作]` (American University in Cairo), Moustafa Youssef `[通讯]` (American University in Cairo)

**通讯引用:** 16113 | [OpenAlex ID](https://openalex.org/A5008352007)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种在预算约束下的量子 AP 选择算法，用于室内 3D 定位。

**💡 创新点**

创新点在于将 AP 选择问题建模为带卡迪纳尔约束的 QUBO，并通过重要性-冗余权衡与熵重要性指标，在量子退火中高效求解；该方法将 AP 数量减少 96% 且保持定位精度。

**🔧 技术方法**

技术包括：量子退火（使用 OpenJij 量子 Monte‑Carlo 模拟）、QUBO 公式化、熵/方差/均值/最大值四种重要性度量、Pearson 相关系数衡量冗余、模拟退火基准对比。

**📊 数据集**

使用公开的 5 层楼建筑 WiFi 轨迹数据集，包含 520 台 AP、21,048 个指纹样本。

**📈 对比分析**

与模拟退火和全 AP 方案进行对比；量子方案平均 3D 定位误差 11.7 m（低于 SA 的 14.3 m 和全 AP 的 12.4 m），楼层识别准确率 73%（高于 SA 58.6% 与全 AP 70.4%），求解时间 0.20 s，较 SA 加速 61 倍，定位误差平均下降 10%。

**⚠️ 局限性**

局限性：需要量子退火硬件或仿真资源；惩罚参数 η 的调优复杂；在更大规模 AP 集合下的可扩展性尚未验证；实验仅在模拟平台完成，未在真实量子硬件上测试；对数据集的依赖可能导致迁移性受限。

---

## 142. GLM-5: from Vibe Coding to Agentic Engineering

**arXiv ID:** 2602.15763 | [PDF](https://arxiv.org/pdf/2602.15763v1)

**作者:** GLM-5 Team `[一作]` (Zhipu AI), Jie Tang `[通讯]` (Tsinghua University)

**通讯引用:** 28736 | [OpenAlex ID](https://openalex.org/A5044791875)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文开发了 GLM‑5，一款规模达 744B 参数、可处理 200K 令牌上下文、并支持多能型（Agentic、Reasoning、Coding）的开放源代码模型。

**💡 创新点**

创新点在于：1）采用 DeepSeek Sparse Attention（DSA）实现稀疏自注意力，显著降低计算成本；2）设计异步强化学习框架，解耦生成与训练，提升 GPU 利用率；3）引入 Multi‑Token Prediction 与 Prefill‑Decode 分离，优化推理吞吐量；4）实现跨 GPU/芯片的全栈优化与 INT4 QAT。

**🔧 技术方法**

技术包括 MoE、DSA、Multi‑Token Prediction、Sparse Flash Attention、ZeRO2 梯度分片、INT4 QAT、Token‑in‑Token‑out、Direct Double‑Sided Importance Sampling、跨语言工具调用与自监督训练。

**📊 数据集**

数据集涵盖 28.5T 预训练语料（代码、数学、科学、中文 Web、软件工程 Repo 等），mid‑training 扩展至 200K 上下文，SFT 语料覆盖 chat、reasoning、coding、agent，RL 采集多领域工具交互与搜索任务。

**📈 对比分析**

通过与 DeepSeek‑V3、Claude Opus、Gemini 等开放/专有模型在 ARC、Reasoning、Coding、Agentic 等公开基准对比，GLM‑5 在开源榜单排名第一，较 GLM‑4.7 提升约 20%，并在 Vending‑Bench 2 等长周期任务中实现 4,432 美元利润，整体性能超过多数开源模型。

**⚠️ 局限性**

局限性包括：与顶级专有模型仍有差距；长上下文一致性易受信息损失影响；RL 样本效率与环境可靠性待提升；中文芯片端精度降级与硬件适配成本仍较高。

---

## 143. Symbolic recovery of PDEs from measurement data

**arXiv ID:** 2602.15603 | [PDF](https://arxiv.org/pdf/2602.15603v1)

**作者:** Erion Morina `[一作]` (Interdisciplinary Digital Lab at University of Graz), Martin Holler `[通讯]` (Interdisciplinary Digital Lab at University of Graz)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了如何用符号神经网络（基于有理函数的架构）在有限噪声、稀疏测量下从数据中反演出偏微分方程（PDE）的物理律及状态，给出了理论可辨识性证明和数值验证。

**💡 创新点**

创新点在于①提出了能保留连续PDE结构且具备可解释性的符号网络架构；②证明在完整无噪声测量极限下，该网络可唯一恢复最简物理律；③利用L¹正则化实现稀疏、可解释的物理律表达；④把符号网络的函数空间正则性纳入理论框架。

**🔧 技术方法**

技术主要包括：符号网络（ParFam、EQL风格）采用有理函数层和非有理基激活；函数空间化的Nemytskii算子理论；all‑at‑once的无模型误差最优化；交替优化（θ 与 u）；正则化最小化与稀疏性。

**📊 数据集**

使用人工合成的一维PDE数据：$u_t=f(t,u,u_x)$，其中 $f$ 为线性、可识别的或非唯一可识别的真律；通过低通采样（截断傅里叶系数）和乘性噪声产生观测。

**📈 对比分析**

与本文的理论相符，实验显示：在采样充分、噪声减小时，状态误差和律误差随 $m$ 增大显著下降；对于唯一可识别的PDE，学习到的律与真律几乎一致；对于非唯一可识别PDE，学习到的律趋向于 L¹ 最小化的稀疏解。

**⚠️ 局限性**

局限性包括：仅适用于可由固定符号网络表示的物理律；未给出收敛速率或统计误差分析；实验仅限于一维简单PDE；对高维、多尺度或非线性复杂系统的推广尚待研究。

---

## 144. Latent Regularization in Generative Test Input Generation

**arXiv ID:** 2602.15552 | [PDF](https://arxiv.org/pdf/2602.15552v1)

**作者:** Giorgi Merabishvili `[一作]` (North Carolina State University), Andrea Stocco `[通讯]` (Technical University of Munich)

**通讯引用:** 2491 | [OpenAlex ID](https://openalex.org/A5027652385)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了在基于StyleGAN的生成式测试框架中，通过调整截断参数（truncation）来平衡生成测试输入的真实性、覆盖度与缺陷检出率，并提出了自适应截断与截断首次翻转两种探索策略。

**💡 创新点**

创新点在于：①系统性评估截断对测试输入有效性、多样性与缺陷揭示的影响；②提出自适应截断机制以“拯救”原始种子，提升有效测试输入的产出率；③引入仅基于截断的首次翻转探测方法，简化语义编辑过程。

**🔧 技术方法**

使用的技术包括：StyleGAN2/StyleGAN2-ADA条件生成模型；截断操作（全层或按层截断）；截断辅助风格混合（style‑mixing）与截断首次翻转（first‑flip）两种搜索算法；自动化筛选（基于分类器置信度、SSIM、L2阈值）与人工验证；多样性度量（均值感知距离）。

**📊 数据集**

实验数据集包括：MNIST、Fashion‑MNIST 与 CIFAR‑10；对应的被测分类器分别是小型CNN、ResNet‑18等。

**📈 对比分析**

与随机截断对比，截断辅助风格混合在大多数截断阈值下提高了缺陷检出率、有效性与多样性；自适应截断进一步减少种子使用量（seed/valid），提升效率；首次翻转方法更快、计算成本低，尤其在小型数据集上表现出更高的缺陷发现率。

**⚠️ 局限性**

局限性在于：仅针对StyleGAN生成器和小规模图像数据集，结果可能不易推广到更复杂的模型或文本/语音等其他模态；人工验证样本有限，可能存在主观偏差；截断对不同层的影响未系统化探讨，且缺少对其他生成架构（如扩散模型）的验证。

---

## 145. From PhysioNet to Foundation Models -- A history and potential futures

**arXiv ID:** 2602.15371 | [PDF](https://arxiv.org/pdf/2602.15371v1)

**作者:** Gari D. Clifford `[一作]` (Emory University), Gari D. Clifford `[通讯]` (Emory University)

**通讯引用:** 351073 | [OpenAlex ID](https://openalex.org/A5057810294)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文回顾并分析了PhysioNet资源与挑战在过去三十年中的发展，阐述了开放数据与算法在医学人工智能中的作用及其挑战。

**💡 创新点**

提出通过PhysioNet挑战赛与开放代码共享，构建可重复、可解释、低碳的医疗AI生态，并讨论基金与激励机制创新。

**🔧 技术方法**

主要运用开放数据共享、竞赛评测、开源软件与深度学习、边缘计算等技术手段。

**📊 数据集**

使用PhysioNet公开的多种心电图数据库（如MIT‑BIH、MIMIC‑IV、PTB‑XL等）以及自研数据库。

**📈 对比分析**

通过年度挑战赛与公开评测框架对比不同算法性能，强调指标的临床可解释性与数据多样性；但具体数值多以案例为例。

**⚠️ 局限性**

局限在于数据多样性不足、模型碳足迹高、实验可重复性与透明度仍待提升，且基础模型仍未达到真正的通用能力。

---

## 146. BindCLIP: A Unified Contrastive-Generative Representation Learning Framework for Virtual Screening

**arXiv ID:** 2602.15236 | [PDF](https://arxiv.org/pdf/2602.15236v1)

**作者:** Anjie Qiao `[一作]` (Sun Yat-sen University), Yuedong Yang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 12684 | [OpenAlex ID](https://openalex.org/A5023539493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种统一的对比-生成表示学习框架（BindCLIP），用于高效的基于检索的虚拟筛选。

**💡 创新点**

创新点包括：①将CLIP式对比学习与口袋条件扩散生成任务耦合，利用生成监督强化嵌入空间对细粒度结合相互作用的感知；②引入硬负样本增强和配体-配体锚定正则化，抑制模型对化学相似度等捷径的依赖；③仅在训练阶段使用生成模块，推理保持原始的最近邻检索，兼顾速度与准确。

**🔧 技术方法**

技术细节：使用 UniMol Transformer 作为口袋和配体编码器；SE(3)-equivariant denoiser实现口袋条件扩散；InfoNCE 对比损失；硬负采样（基于分子嵌入 + AutoDock Vina 验证）和锚定正则化；FiLM 调制；线性探测器验证细粒度交互信息。

**📊 数据集**

数据集：训练集为 PDBBind 2019 + HomoAug（≈66k 复合物）；验证集 CASF-2016；硬负样本库为 1M ZINC 分子；评估基准包括 DUD-E、LIT-PCBA、MF-PCBA OOD（基于 UniProt 过滤）以及 FEP+ 四靶点（CDK2、TYK2、JNK1、P38）.

**📈 对比分析**

与 DrugCLIP、DrugHash、传统对接（Vina、Glide-SP）以及其他学习方法对比。BindCLIP 在 LIT-PCBA 上 AUROC 提升 6.7%，BEDROC 提升 24.5%；在 OOD 评测中 BEDROC +62%，EF0.5% +115%；在 FEP+ 排名任务中对接近化合物的 pairwise accuracy 提升至 65.7%（相对 16%）。整体表现显著优于基线。

**⚠️ 局限性**

局限性：模型仍依赖已知结合位点信息，且训练成本较高；对极低相似度的负样本或极端不平衡数据的鲁棒性尚未充分验证；推理仅使用编码器，无法直接给出结合姿态或能量预测。

---

## 147. MRC-GAT: A Meta-Relational Copula-Based Graph Attention Network for Interpretable Multimodal Alzheimer's Disease Diagnosis

**arXiv ID:** 2602.15740 | [PDF](https://arxiv.org/pdf/2602.15740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 148. Learning to Retrieve Navigable Candidates for Efficient Vision-and-Language Navigation

**arXiv ID:** 2602.15724 | [PDF](https://arxiv.org/pdf/2602.15724v1)

**作者:** Shutian Gu `[一作]` (University of New South Wales), Lina Yao `[通讯]` (University of New South Wales)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LLM驱动的视觉语言导航中引入检索增强框架，在每个任务开始时检索相似指令的成功示例作为上下文提示，并在每一步通过 imitation‑learn 训练的候选检索器剪枝无关方向，减少 LLM 推理负担，提升导航效果。

**💡 创新点**

创新点在于双层检索机制：①指令级示例检索提供全局导航先验；②步骤级候选检索以轻量化方式剔除冗余方向，二者无须修改底层 LLM，显著提升决策效率与稳定性。

**🔧 技术方法**

技术主要包括大语言模型（Qwen3、LLaMA3.1 等）+ 文本嵌入检索（句子嵌入 + 相似度搜索）+ imitation‑learning 训练的候选评分网络 + 结构化 prompt 生成与推理。

**📊 数据集**

使用的标准数据集为 Room-to-Room (R2R) benchmark，包含训练、验证及测试拆分，涉及 7,189 条真实室内导航轨迹。

**📈 对比分析**

与 NavGPT Qwen3 基线对比，检索增强后在 seen/unseen 评估中 SR 提升约 4%–5%，OSR 提升约 10%–15%，SPL 提升 2%–3%；相较于监督式 VLN 方法仍有差距，但显著缩小了 gap，并在推理速度上比基线更快。

**⚠️ 局限性**

局限性包括：①性能仍落后于大规模监督训练的 VLN 系统；②候选检索器需额外训练，且对不同 LLM 的适配性需进一步验证；③检索提示增大 token 数，导致某些情景下推理开销提升；④在更大规模、不同域的 LLM 上的通用性尚未充分验证。

---

## 149. Artificial Intelligence Specialization in the European Union: Underexplored Role of the Periphery at NUTS-3 Level

**arXiv ID:** 2602.15249 | [PDF](https://arxiv.org/pdf/2602.15249v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 150. Prescriptive Scaling Reveals the Evolution of Language Model Capabilities

**arXiv ID:** 2602.15327 | [PDF](https://arxiv.org/pdf/2602.15327v1)

**作者:** Hanlin Zhang `[一作]` (Harvard University), Sham Kakade `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了可用于实践的“预训练计算→后训练性能”映射框架，利用大规模观察数据估计高分位数能力边界，并通过自适应采样快速恢复完整边界；

**💡 创新点**

创新点在于：①提出把后训练性能视为可达性高分位数，用单参数的单调Sigmoid函数刻画；②将时间视为核心维度，验证边界在新模型世代中的时序稳定性；③提供高效采样算法，实现仅用约20%评测预算即可近似完整边界；④将边界应用于评估饱和、污染等问题，提供量化诊断。;

**🔧 技术方法**

技术包括：平滑分位数回归（smoothed quantile regression）配合单调饱和Sigmoid参数化；时间序列交叉验证；自适应采样算法；对比I‑spline等更复杂模型；使用logit变换和大小-时间边界模型来评估饱和；以及跨基准的线性回归检测污染。;

**📊 数据集**

数据集：Open LLM Leaderboard v1/v2（含数千模型评测）；Proteus‑2k（约5k观测+2k新评测）；公开的 frontier model 评测（Epoch AI、LifeArchitect.AI等）；以及对 Qwen3、Gemma‑3、GPT‑OSS 等新模型自行评测。;

**📈 对比分析**

方法对比：将预训练模型的性能与后训练的高分位数边界进行对齐，评估覆盖误差（coverage error）和 pinball loss；对比 Sigmoid 与 I‑spline 的拟合优劣；对不同任务（BBH、GPQA、MATH Lvl 5、MMLU‑Pro 等）进行横向比较。结果显示，除 MATH Lvl 5 外，绝大多数任务的边界在时间上保持稳定，Sigmoid 拟合误差低于 I‑spline，且自适应采样可在 20% 预算下恢复 95% 以上的边界；

**⚠️ 局限性**

局限性：①数学推理等任务的边界随时间持续上升，说明算法进步仍在推动边界移动；②对非常规后训练技术或全新架构的可达性预测可能失效；③受评测集污染、数据泄漏等问题影响，边界可能被轻微偏移；④自适应采样虽高效，但在极端稀疏或分布不均的任务上仍需进一步验证。

---

## 151. Hybrid Federated and Split Learning for Privacy Preserving Clinical Prediction and Treatment Optimization

**arXiv ID:** 2602.15304 | [PDF](https://arxiv.org/pdf/2602.15304v1)

**作者:** Farzana Akter `[一作]` (Washington University of Science and Technology), Lisan Al Amin `[通讯]` (Cognitive Links LLC)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在患者数据无法集中存储的场景下，构建并评估了结合联邦学习与拆分学习的混合框架，用以实现临床预测与治疗优化。

**💡 创新点**

创新点在于将模型切分点与联邦聚合相结合，既降低客户端计算负担，又明确协作边界，可对中间表示进行隐私审计和可调节的隐私-效能折衷。

**🔧 技术方法**

采用两头提升网络、FedAvg、Split Learning、Hybrid FL‑SL、激活裁剪与高斯噪声防御、以及成员推断审计等技术。

**📊 数据集**

使用了三大公开临床数据集：eICU‑demo、MEPS 2022 和 NHANES 2015–2018，均按非IID机构划分。

**📈 对比分析**

与中心化、纯联邦和纯拆分学习对比，Hybrid 方案在事实预测、基于容量的提升排序、隐私泄露(AUC)与通信成本上均表现相当或更优，尤其在非IID客户端下保持更稳健的提升排名。

**⚠️ 局限性**

局限包括：使用观测代理治疗导致提升信号可能为负，成员推断攻击仅为轻量级审计，且未提供正式差分隐私保证，未来需在更强攻击模型下进一步验证隐私安全。

---

## 152. When Remembering and Planning are Worth it: Navigating under Change

**arXiv ID:** 2602.15274 | [PDF](https://arxiv.org/pdf/2602.15274v1)

**作者:** Omid Madani `[一作]` (Brown University), Thomas L. Dean `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究在非平稳、有限感知的格子世界中，如何利用不同类型与使用方式的记忆（从短期访问信息到长期概率模型）来帮助智能体快速适应变化并高效导航到食物。

**💡 创新点**

创新点在于：①提出一种可组合多种策略的架构，能够在探索、搜索和规划子任务之间自适应切换；②将非平稳概率学习与即时地图构建结合，形成“记忆‑学习‑规划”闭环；③引入递增时间预算与回退规划机制，使得在动态环境下保持鲁棒性；④系统性比较多种记忆深度（无记忆→短期→长期→概率地图）与模型无关/模型自由方法，证明记忆与规划组合可显著降低步数。

**🔧 技术方法**

技术手段包括：有限感知（视野半径1）、随机/偏向随机、贪婪(嗅觉)、未访问优先、路径记忆、概率地图策略（基于时间窗口的预测、贝叶斯更新、A*规划）、进化式预算策略、以及基准的深度Q网络。所有策略均在同一框架内实现并可互相交互。

**📊 数据集**

实验数据集：合成的闭合网格世界，尺寸分别为15×15、25×25、50×50；障碍比例从0到0.3；障碍/食物每天随机变动的比例（0、0.1、0.2、0.3）；运动噪声p=0、0.01、0.02；每个实验场景随机种子50，20天/场景，收集每日步数统计。

**📈 对比分析**

与多种基线对比：单一随机、偏向随机、贪婪、未访问、路径记忆、概率地图、oracle（完全地图）及DQN。结果显示：在障碍比例≤0.3、噪声≤0.02且变化率适中时，概率地图策略平均步数可比混合贪婪低约20–30倍；随着网格尺寸增大、障碍比例升高，优势更显著；当噪声或变化率过高时，优势减弱。性能评估使用均值、分位数、最大值等统计，均显示记忆深度越大鲁棒性越强。

**⚠️ 局限性**

局限性：①只在离散格子世界实验，未验证连续或高维空间；②感知模型极简，未考虑更复杂传感器或视觉；③对障碍与食物变化的先验分布做了固定比例假设，缺乏自适应变化检测；④规划与记忆更新的计算开销随地图规模增长显著；⑤策略架构中“硬编码”的优先级和预算倍增可能不适用于所有任务，需要进一步学习自适应调度。

---

## 153. ActionCodec: What Makes for Good Action Tokenizers

**arXiv ID:** 2602.15397 | [PDF](https://arxiv.org/pdf/2602.15397v1)

**作者:** Zibin Dong `[一作]` (Tsinghua University), Jianye Hao `[通讯]` (Tianjin University)

**通讯引用:** 5392 | [OpenAlex ID](https://openalex.org/A5047509839)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现 ActionCodec，针对 Vision‑Language‑Action (VLA) 模型的动作分词问题，构建高效、稳健的动作 token 化方案。

**💡 创新点**

从信息理论出发提出四大设计原则：最大化时间重叠、最小化词表冗余、提升多模态互信息、令 token 独立，并在此基础上融合 VQ、RVQ、Perceiver、Soft Prompts 等技术，形成一套可直接落地的高性能分词器。

**🔧 技术方法**

采用 Vector Quantization（VQ）与 Residual Vector Quantization（RVQ）实现离散表示；使用 Perceiver‑style transformer 作为编码器/解码器；通过 Time Contrastive Learning（TCL）与 CLIP 对齐提升视觉‑语言对齐；加入 Soft Prompts 进行多机器人平台迁移；对比损失与 stop‑gradient 保障训练稳定。

**📊 数据集**

利用公开机器人交互数据集 LIBERO、BridgeData、DROID、SO100‑ShapeSorter、xArm PickVeg、SimplesWidowX 等进行训练与评估，覆盖仿真与真实世界场景。

**📈 对比分析**

在 LIBERO、Bridge‑WidowX、SO100、xArm PickVeg 等 benchmark 上与 Binning、String、VQ‑VLA、MiniVLA、FAST 等主流分词器以及 PD、KI、BAR 等 VLA 架构进行对比。ActionCodec 在 2.2B VLM 上无机器人预训练即可取得 95.4%/97.6%/97.4% 等平均成功率，刷新 SOTA；训练效率提升显著（5k 步骤即 89.5% 成功率，远快于 FAST 38.6%）；吞吐量与延迟优于传统 Binning/String，适合高频闭环控制。

**⚠️ 局限性**

局限性包括：仍需在更大规模、多样化机器人平台上进行预训练以提升泛化；对极低频率高精度动作的重构误差仍有提升空间；依赖大规模 VLM 资源，对算力要求高；验证主要基于公开数据集，真实场景鲁棒性待进一步验证。

---

## 154. Hybrid F' and ROS2 Architecture for Vision-Based Autonomous Flight: Design and Experimental Validation

**arXiv ID:** 2602.15398 | [PDF](https://arxiv.org/pdf/2602.15398v1)

**作者:** Abdelrahman Metwally `[一作]` (Skolkovo Institute of Science and Technology), Andrey Somov `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 2833 | [OpenAlex ID](https://openalex.org/A5084140816)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实验证明了NASA F'与ROS 2混合架构在资源受限嵌入式平台上实现视觉导引的无人机自主飞行。

**💡 创新点**

首次全面实验验证F'–ROS 2混合架构，展示其实时性能、资源占用和命令执行可靠性，并揭示I/O阻塞导致的视觉数据间隙。

**🔧 技术方法**

F'确定性飞行软件框架、ROS 2 middleware、Protocol Buffers桥接、MAVROS、Vicon运动捕捉、ArduPilot飞控、DDS QoS、UDP实时通信。

**📊 数据集**

现场Vicon捕捉的6‑DoF位姿数据（100 Hz）和ArduPilot任务日志，约33 分钟内部飞行数据。

**📈 对比分析**

通过采样率、延迟、连续性、定位精度、CPU/内存占用、命令成功率等指标衡量，结果显示视觉估计87 Hz、平均延迟11.5 ms、99.9%连续性、位置误差≤0.95 m、CPU 15%/内存 1.2 GB、命令100%成功。

**⚠️ 局限性**

受限于室内光照、单机单机机型、短时任务、同步文件I/O导致的8 s数据间隙，缺乏长周期漂移评估和室外环境验证。

---

## 155. Distributional Deep Learning for Super-Resolution of 4D Flow MRI under Domain Shift

**arXiv ID:** 2602.15167 | [PDF](https://arxiv.org/pdf/2602.15167v1)

**作者:** Xiaoyi Wen `[一作]` (University of California), Fei Jiang `[通讯]` (University of California)

**通讯引用:** 41428 | [OpenAlex ID](https://openalex.org/A5062571272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种分布式深度学习框架，用于在域漂移条件下实现4D Flow MRI的超分辨率重建。

**💡 创新点**

通过在训练数据中加入人工噪声并采用分布式损失函数，显著提升模型对测试域外数据的外推能力，并提供理论一致性证明。

**🔧 技术方法**

使用预加性噪声DSR模型、3D U‑Net架构、能量基分布式损失以及自监督预训练与微调相结合的训练策略。

**📊 数据集**

先利用1200个CFD模拟块进行预训练，再用99对配对的CFD‑4DF块进行微调与测试，其中15块用于微调，84块用于评估。

**📈 对比分析**

与传统L₂回归和4DFlowNet进行对比，DSR在真实4DF数据上MSE显著降低，预测分布更贴近真实分布，显示出更优的性能。

**⚠️ 局限性**

受限于样本量有限，缺乏物理约束的引入以及对极端域漂移的适应性仍需进一步研究。

---

## 156. Extracting Consumer Insight from Text: A Large Language Model Approach to Emotion and Evaluation Measurement

**arXiv ID:** 2602.15312 | [PDF](https://arxiv.org/pdf/2602.15312v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 157. Zombie Agents: Persistent Control of Self-Evolving LLM Agents via Self-Reinforcing Injections

**arXiv ID:** 2602.15654 | [PDF](https://arxiv.org/pdf/2602.15654v1)

**作者:** Xianglin Yang `[一作]` (National University of Singapore), Jin Song Dong `[通讯]` (National University of Singapore)

**通讯引用:** 6641 | [OpenAlex ID](https://openalex.org/A5085067496)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对自演化LLM代理的持久性恶意攻击“Zombie Agent”，演示通过黑盒诱导写入长期记忆来实现跨会话的恶意执行。

**💡 创新点**

创新点在于将诱导注入与记忆演化相结合，设计了递归复制与语义别名两种持久化策略，突破了传统一次性注入的局限。

**🔧 技术方法**

技术包括基于滑动窗口和检索增强生成（RAG）的记忆更新模型、攻击载荷的语义别名优化、两阶段感染-触发框架以及对现有安全防御的评估。

**📊 数据集**

实验数据集使用公开网页诱导任务与真实网页交互查询（约3,000条检索记录），在Gemini‑2.5‑Flash与GLM‑4.7‑Flash两大LLM上进行评测。

**📈 对比分析**

与四种传统间接提示注入基线相比，Zombie Agent在滑动窗口和RAG两种记忆机制下的攻击成功率均高于80%，且能在多达20轮触发任务中持续成功；持久化指标显示其记忆保留率始终为100%，召回率显著优于基线。

**⚠️ 局限性**

局限性包括对攻击载荷的手工设计依赖、对特定记忆更新策略的假设、未评估攻击在更复杂多模态工具链或强化学习代理中的表现。

---

## 158. Meflex: A Multi-agent Scaffolding System for Entrepreneurial Ideation Iteration via Nonlinear Business Plan Writing

**arXiv ID:** 2602.15631 | [PDF](https://arxiv.org/pdf/2602.15631v1)

**作者:** Lan Luo `[一作]` (Hong Kong University of Science and Technology), Pan Hui `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 21172 | [OpenAlex ID](https://openalex.org/A5029925982)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并实现了 Meflex，一个基于多代理大语言模型的非线性商业计划写作工具，并通过 30 名学生的可用性与认知影响实验进行评估。

**💡 创新点**

创新点在于将非线性可视化思维画布与反思、元反思 scaffolding 结合，支持迭代、分支式创意生成；同时为每个商业计划模块引入专属 LLM 代理，实现上下文感知与多角度提示。

**🔧 技术方法**

采用 DeepSeek API 的大语言模型，构建多代理框架；通过视觉画布实现节点扩展（水平/垂直）和自动生成的反思与元反思提问；系统包含 9 个业务模块的专属提示词。

**📊 数据集**

未使用公开数据集，评估基于 30 名来自不同专业的学生使用 10 个预设创业主题的实验数据。

**📈 对比分析**

使用系统可用性量表（SUS）评估，并通过访谈进行主题分析；SUS 平均分约 5.8–6.3，显示高可用性；定性结果表明工具显著降低认知负荷、促进发散思维与自我修正，未与其他写作工具做直接性能对比。

**⚠️ 局限性**

局限性包括元反思仅追踪文本变更，未捕获用户与 LLM 的交互轨迹；实验规模有限，缺乏长期效果与多工具比较；系统依赖 LLM，可能出现信息偏差或固定化风险。

---

## 159. Memory Reallocation with Polylogarithmic Overhead

**arXiv ID:** 2602.15417 | [PDF](https://arxiv.org/pdf/2602.15417v1)

**作者:** Ce Jin `[一作]` `[通讯]` (University of California Berkeley), Ce Jin (University of California Berkeley)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

本文提出了一种新的动态内存重新分配分配器，能够在在线更新（插入/删除）中实现多项式对数级的期望重分配开销（即重置/移动对象的总字节数相对于更新对象大小的比例为 O(polylog(1/ε))，其中 ε 为允许的最大内存填充率误差。

**💡 创新点**

创新点在于：
- 利用 Sunflower 引理和子集和的算术结构将大规模对象划分为“bundle type”并在多层级中管理；
- 在替代策略（substitution strategy）基础上引入随机计数器与层级重建机制，显著降低在不同对象规模下的重分配成本；
- 通过“finger object”与多维树形输入序列的巧妙设计，构造了严谨的下界，证明任何随机分配器在期望平方开销上至少需要 Ω(9/7) 次幂的增长，进而得出 Ω(log(1/ε)) 的下界，表明多项式对数级是最优（在高概率下不可能实现更低）。

**🔧 技术方法**

核心技术包括：
- Sunflower 引理和子集和唯一性论证来限制对象尺寸集合的多重性；
- 对象束化（bundling）与层级分配的贪心规则；
- 随机计数器和“最大的整数能整除”触发重建的概率分析；
- 位置潜能函数和潜能变动上界的 Cauchy‑Schwarz 与三角不等式结合；
- Yao 原理用于从随机分配器推导不可行的确定性状态序列，进而得到上界与下界对比。

**📊 数据集**

无实测数据集，论文完全基于理论分析和构造性的硬实例（完整内存状态配置集）完成评估。

**📈 对比分析**

与之前的最优随机分配器（期望重分配开销 O(ε⁻¹/2)）相比，本文的分配器在 worst‑case 期望开销仅为 O(polylog(1/ε))，实现了至少对数级（log(1/ε)) 的提升；在高概率下也证明不可能进一步降低到多项式对数以下。由于是随机化且仅给出期望值，实验式性能未在真实系统上验证，但理论上已达到最佳可行区间。

**⚠️ 局限性**

限制与不足：
- 算法为随机化，无法提供确定性、绝对保证的低开销；
- 仍需偶发的大规模重建，导致实际运行中的瞬时延迟可能较高；
- 仅对非极小对象（size > 2⁻ℓ M）给出多项式对数上界，极小对象仍需结合 Kuszmaul 的随机分配器；
- 高概率下不可能实现子多项式级别的重分配开销；
- 假设了对象尺寸在固定比例区间内（bounded‑ratio property）且内存可以完全填满；在更一般的尺寸分布或动态负载下的表现尚未覆盖。

---

## 160. GMAIL: Generative Modality Alignment for generated Image Learning

**arXiv ID:** 2602.15368 | [PDF](https://arxiv.org/pdf/2602.15368v1)

**作者:** Shentong Mo `[一作]` (Carnegie Mellon University), Sukmin Yun `[通讯]` (Hanyang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GMAIL框架，将生成图像视为单独模态并在潜在空间与真实图像对齐，以提升视觉语言任务的性能。

**💡 创新点**

创新点在于将生成图像明确视为独立模态，使用跨模态对齐损失和LoRA细调实现生成-真实模态的统一对齐，从而有效缓解模式坍塌问题。

**🔧 技术方法**

使用的技术包括Stable Diffusion v2生成图像、CLIP基础模型、LoRA低秩适配、跨模态对齐损失、生成图像与真实图像的联合对齐，以及在ClipCap、LLaVA、LLAMA-3等视觉语言模型中的后续融合。

**📊 数据集**

使用的数据集包括COCO、Flickr30k、CC3M、CC12M、ShareGPT4V、ImageNet1K等多任务基准。

**📈 对比分析**

通过与ClipCap、CLIP、Long-CLIP、LLaVA、LLAMA-3等基线模型在图像标题生成、零样本检索、零样本分类、长标题检索等任务上对比，GMAIL在BLEU、METEOR、CIDEr、Recall@1等指标上均显著提升，提升幅度从几百分点到十几百分点不等。

**⚠️ 局限性**

局限性包括：仍依赖生成图像质量与偏差，生成模型可能带入偏见；对齐过程需要额外训练成本；在极大规模生成数据下对齐效果及泛化能力仍需进一步验证；对特定领域或复杂场景的适应性仍有限。

---

## 161. LuxMT Technical Report

**arXiv ID:** 2602.15506 | [PDF](https://arxiv.org/pdf/2602.15506v1)

**作者:** Nils Rehlinger `[一作]` `[通讯]` (University of Luxembourg), Nils Rehlinger (University of Luxembourg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Gemma 3的基础上微调得到LuxMT，专门用于卢森堡语向法语/英语的机器翻译，并通过自定义的Luci评测基准验证其在卢森堡语到德语（未训练）的表现。

**💡 创新点**

①提出了以卢森堡旅游杂志Luci为核心的低资源多语言评测基准，避免了主流数据集的训练污染；②利用LuxEmbedder句子嵌入既作为训练数据过滤工具，又作为潜在的无参考质量估计指标；③在未fine‑tune德语时仍能提升德语翻译质量，展示了跨语言学习效果。

**🔧 技术方法**

使用Gemma 3大型语言模型进行一次epoch的fine‑tune（学习率2e‑5、温度1.0），结合LuxEmbedder进行句子相似度阈值过滤；评估时采用BERTScore、BLEURT‑20、xCOMET‑XL、BLEU、chrF2、TER以及LuxEmbedder自带的余弦相似度，并用配对自助抽样检验统计显著性。

**📊 数据集**

训练语料：LuxAlign的卢森堡语–法语/英语平行新闻；卢森堡议会记录经Google翻译扩充成法语/英语；评测语料：Luci杂志的四语人工翻译（共500段/语种），并按LuxEmbedder排序后挑选前500段。

**📈 对比分析**

先用Luci基准比较多款本地LLM（Gemma 3、Aya Expanse、Command R、Llama 3.1/4、Mistral S、Phi 4），确定Gemma 3最佳；随后在不同LuxEmbedder阈值（0.90/0.95/0.99）和epoch（1/2/3）上做ablation。结果显示LuxMT在LB→FR、LB→EN、LB→DE上分别提升BLEURT‑20约0.8–1.4分、xCOMET‑XL约1.9–2.6分，所有提升均在p<0.05显著。

**⚠️ 局限性**

①评测基准规模小（每语种仅500段）且域偏旅游，可能导致过拟合；②源语言不确定，Luci文本可能含翻译痕迹；③评估指标未覆盖最新SOTA指标，LuxEmbedder作为QE需更多人类验证；④缺乏跨域语料与更细粒度的语言现象分析。

---

## 162. Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning

**arXiv ID:** 2602.15817 | [PDF](https://arxiv.org/pdf/2602.15817v1)

**作者:** Oswin So `[一作]` (Massachusetts Institute of Technology), Chuchu Fan `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1876 | [OpenAlex ID](https://openalex.org/A5019603699)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出一种同时估计可行参数子集并在该子集上学习安全策略的 Feasibility‑Guided Exploration（FGE）框架，专门解决未知可行性下的参数鲁棒避障问题。

**💡 创新点**

创新点在于：① 用混合分布的正例与噪声负例训练保守的可行性分类器，确保不误判不可行参数；② 采用基于 FTRL 的鞍点算法和回放缓冲实现可行参数集上的鲁棒最优控制；③ 设计探索分布主动寻找尚未确定可行性的参数，从而突破可行集扩展的负反馈循环。

**🔧 技术方法**

核心技术包括：深度强化学习（PPO）、鞍点优化（FTRL 与最佳响应）、可行性分类器（基于混合分布的变分推断）、探索与回放缓冲（经验重放）以及混合分布采样。

**📊 数据集**

在 MuJoCo 连续控制任务、自动驾驶模拟（ACC）、高维固定翼飞机仿真、月球登陆（RGB 图像观测）等多种基准环境上进行实验。

**📈 对比分析**

与域随机化、RARL、两类课程学习、UED 等基线比较，FGE 在安全率、可行参数覆盖率和覆盖增益上均优于所有对手；在所有任务中获得最高的安全率，并显著提升难度参数的可行性。

**⚠️ 局限性**

局限性包括：理论收敛性未完全证明（需假设最佳响应可获得）；仅适用于确定性动力学；在随机/嘈杂环境下需改进（如引入概率约束）；方法对超参数（探索/回放阈值）较为敏感。

---

## 163. GenAI-LA: Generative AI and Learning Analytics Workshop (LAK 2026), April 27--May 1, 2026, Bergen, Norway

**arXiv ID:** 2602.15531 | [PDF](https://arxiv.org/pdf/2602.15531v1)

**作者:** Javier Irigoyen `[一作]` (Universidad Autonoma de Madrid), Ruben Tolosana `[通讯]` (Universidad Autonoma de Madrid)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

创建EduEVAL-DB数据集，模拟六种教师角色的解释，并用该数据集评估LLM在教学风险评估中的表现。

**💡 创新点**

首次将教师角色模拟与基于风险的多维度教学评估标准相结合，发布可公开使用的EduEVAL-DB。

**🔧 技术方法**

使用prompt engineering生成教师角色解释，采用半自动标注流程，利用LoRA对Llama 3.1 8B进行微调，并用MAE评估模型性能。

**📊 数据集**

基于ScienceQA中139道题目，生成854条教师角色解释（含6种LLM角色+1人类教师）。

**📈 对比分析**

通过与Gemini 2.5 Pro和未微调的Llama 3.1 8B进行零样本对比，微调后的Llama在所有风险维度的MAE显著下降，尤其在学生级别适宜性上达0.003。

**⚠️ 局限性**

数据规模有限，讽刺角色仅覆盖20道题，评估仅为二元标签，缺少交互式对话与多模态信息，限制了通用性和深度。

---

## 164. Hierarchical Decomposition of Separable Workflow-Nets

**arXiv ID:** 2602.15739 | [PDF](https://arxiv.org/pdf/2602.15739v1)

**作者:** Humam Kourani `[一作]` (Fraunhofer Institute for Applied Information Technology), Wil M. P. van der Aalst `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

设计并实现了一种递归算法，将安全且可听的工作流网转换为等价的POWL 2.0层次模型。

**💡 创新点**

创新点：引入统一的 choice‑graph 与 partial‑order 结构，采用冲突隐藏和并发隐藏的分层拆解，证明对可分离 WF‑net 的完整性与正确性。

**🔧 技术方法**

技术：基于 Petri‑net 的结构分解（top‑down），冲突隐藏/并发隐藏投影与归一化，choice‑graph 与 partial‑order 的语言语义定义，Python 实现与 Petri‑net 库。

**📊 数据集**

数据集：1,000 个由过程树生成的随机 WF‑net；1,493 个工业与合成的真实工作流模型（SAP R/3 参考模型）。

**📈 对比分析**

比较方法与性能：与旧版 POWL、过程树转换器对比；实验中所有 1,000 个模型全部成功，最大耗时 1.48 s；在 1,493 模型上 100% 成功，平均 0.004 s，显著优于其他方法。

**⚠️ 局限性**

局限性：仅能处理可分离 WF‑net（无法表示跨越并发与决策的非 SESE 结构），对非可分离模型需 fall‑through 或近似；预处理规则范围有限。

---

## 165. PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra

**arXiv ID:** 2602.15669 | [PDF](https://arxiv.org/pdf/2602.15669v1)

**作者:** Xiachong Feng `[一作]` (Harbin Institute of Technology), Bing Qin `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 16184 | [OpenAlex ID](https://openalex.org/A5017671620)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 PERSONA 框架，实现对大型语言模型（LLM）人格的训练‑free 控制，能够在激活空间中提取、算术操作和动态推断时实时调节人格向量。

**💡 创新点**

创新点包括：
• 通过对比激活分析提取出可正交且可算术操作的人格向量；
• 将向量算术与 predict‑then‑steer 机制结合，实现了动态、上下文感知的人格调节（Persona‑Flow）；
• 提出了 Persona‑Evolve 基准，专门评估多轮对话中的动态人格保持与适应能力。

**🔧 技术方法**

技术方法：对比激活分析、残差流向量插入、向量标量乘、向量加减、BERT/LLM 预测系数、OCEAN（Big Five）人格模型、基于 BFI‑44 的行为评估。

**📊 数据集**

数据集：
• PersonalityBench（~90道情境问答）
• Persona‑Evolve（800多轮对话、100个人物、100场景）
• 自制的 BFI‑44 适配版问卷（情景化提示）
• 其它公开对照基线数据用于评估。

**📈 对比分析**

比较方法与性能：与多种无梯度基线（prompting、NPTI、Simple Prompt、P^2、PAS、ActAdd）以及监督微调（LoRA）对照；在 PersonalityBench 上无梯度方法平均 9.60 分，几乎匹配 9.61 的 SFT 上限；在 Persona‑Evolve 上，Persona‑Flow 在 Trait Adherence、Role Consistency、Response Authenticity 等指标的胜率普遍在 73–91% 之间，最高 91% win rate。

**⚠️ 局限性**

限制：
• 对模型安全/伦理的抵制（例如自我中心、伤害相关特质难以激活），需额外安全约束；
• 语义准确性与事实性在动态人格调节中略有下降；
• 依赖于可提取的正交向量，若模型层分布或训练数据差异过大，提取效果可能受限；
• 需要手工构造对比问卷与评测，工作量较大。

---

## 166. Indic-TunedLens: Interpreting Multilingual Models in Indian Languages

**arXiv ID:** 2602.15038 | [PDF](https://arxiv.org/pdf/2602.15038v1)

**作者:** Mihir Panchal `[一作]` (Dwarkadas Jivanlal Sanghvi), Asif Ekbal `[通讯]` (Indian Institute of Technology Jodhpur)

**通讯引用:** 9460 | [OpenAlex ID](https://openalex.org/A5085370631)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对印度多语种语言的解释框架——Indic‑TunedLens，用以在多语言大语言模型中对中间隐藏层进行解码并提升可解释性。

**💡 创新点**

创新点在于学习共享的仿射变换，使中间隐藏状态与目标语言的输出分布对齐，解决了现有以英语为中心的Logit Lens在形态学丰富、低资源印度语言上的投影失配问题。

**🔧 技术方法**

主要技术包括基于Tuned Lens的仿射变换学习、KL散度损失对齐中间层与最终输出分布，以及在Transformer层上实现逐层解释。

**📊 数据集**

使用了Sarvam‑1 1B参数模型，并在11种印度语言（孟加拉语、英语、古吉拉特语、印地语、卡纳达语、马拉雅拉姆语、马拉地语、旁遮普语、尼泊尔语、泰米尔语和泰卢固语）的Sangraha数据集进行训练，评估基于MMLU多语言版的10种语言。

**📈 对比分析**

与传统的Logit Lens相比，Indic‑TunedLens在所有语言上都显著提升了层级准确率、平均排名和熵收敛性；尤其在早期层（1–8）对形态分析阶段的解释效果更好，整体性能提升约10–20%（准确率提升0.04–0.06）。

**⚠️ 局限性**

局限性包括仅在1B参数Sarvam‑1模型上实现、评估仅限于MMLU任务、对下游任务性能提升尚无直接证据，以及缺乏对更大规模模型和更丰富任务的验证。

---

## 167. Reflecting on 1,000 Social Media Journeys: Generational Patterns in Platform Transition

**arXiv ID:** 2602.15489 | [PDF](https://arxiv.org/pdf/2602.15489v1)

**作者:** Artur Solomonik `[一作]` (Center for Advanced Internet Studies), Hendrik Heuer `[通讯]` (Center for Advanced Internet Studies)

**通讯引用:** 401 | [OpenAlex ID](https://openalex.org/A5069890392)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出社交媒体旅程（Social Media Journey）概念，利用图形化工具让1000名美国受访者自述其平台迁移路径，分析转移原因与代际差异。

**💡 创新点**

创新点在于将社交媒体使用视为时间序列的迁移网络，并通过图形化问卷捕获转移原因，提供跨代、跨平台的整体视角。

**🔧 技术方法**

采用图数据库Cypher查询与定量图度量（入度/出度）结合质性内容分析的混合方法。

**📊 数据集**

数据集为美国抽样1000名受访者的自述社交媒体旅程，涉及76个平台，包含3,553条迁移原因。

**📈 对比分析**

通过比较不同代际（Boomer、GenX、Millennial、GenZ）的平台迁移流向和推拉因素，发现推拉模式与平台设计/社群因素相关，未给出传统性能指标，但通过可视化和统计显示迁移频率。

**⚠️ 局限性**

局限在于回忆偏差、仅限美国样本、缺乏实时数据与因果推断，且未能捕捉平台内部使用时长等量化指标。

---

## 168. Emergent Morphing Attack Detection in Open Multi-modal Large Language Models

**arXiv ID:** 2602.15461 | [PDF](https://arxiv.org/pdf/2602.15461v1)

**作者:** Marija Ivanovska `[一作]` (Faculty of Electrical Engineering, University of Ljubljana), Vitomir Štruc `[通讯]` (Faculty of Electrical Engineering, University of Ljubljana)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对开源多模态大型语言模型在单图像面部变形攻击检测中的零样本性能进行系统评估。

**💡 创新点**

发现多模态预训练模型天然具备检测面部变形的潜能，并提出了标准化的零样本评估协议。

**🔧 技术方法**

使用多模态LLM（InternVL、DeepSeek-VL、Qwen、Gemma、LLaVA、Pixtral 等）与统一二分类提示。

**📊 数据集**

评估五个公开面部变形数据集：FRLL-Morphs、MIPGAN II、MorDIFF、Morph-PIPE、Greedy-DiM。

**📈 对比分析**

与多种任务特定MAD基线（SelfMAD、UBO-R3、AAW-MAD 等）对比，LLaVA1.6-Mistral-7B 在 EER 上比最佳对手低 23%，BSCER@5% 也显著提升。

**⚠️ 局限性**

零样本方法对后处理或高质量 GAN/扩散生成的变形易误判；小模型性能不佳；提示设计对模型影响大；需进一步优化以提升对未见攻击的鲁棒性。

---

## 169. A Scan-Based Analysis of Internet-Exposed IoT Devices Using Shodan Data

**arXiv ID:** 2602.15263 | [PDF](https://arxiv.org/pdf/2602.15263v1)

**作者:** Richelle Williams `[一作]` (Florida Atlantic University), Fernando Koch `[通讯]` (Florida Atlantic University)

**通讯引用:** 1409 | [OpenAlex ID](https://openalex.org/A5012632345)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对公开可访问的IoT终端（通过TCP端口7547 TR-069/CWMP）进行跨国扫描样本，分析其外部可观察的服务配置特征与风险暴露关系，并构建分类模型预测高风险主机。

**💡 创新点**

首次将扫描得到的服务配置特征作为人口层面暴露风险度量，证明跨国样本中配置差异能区分高低风险，且无需内部访问或漏洞利用。

**🔧 技术方法**

利用Internet-wide扫描（Shodan Search/InternetDB）收集端口与元数据，进行特征相关性分析、跨国统计比较以及基于扫描特征的监督分类（如逻辑回归/树模型）。

**📊 数据集**

Shodan Search查询TCP7547的100台主机（10国各10台）以及其InternetDB的元数据（开放端口、标签、CPE、漏洞信息）。

**📈 对比分析**

通过描述性统计比较不同国家的平均风险端口数，使用混淆矩阵评估二分类模型，平衡准确率约为0.61，高风险识别准确率低于低风险，但仍显著高于随机猜测。

**⚠️ 局限性**

样本规模有限且不均衡，特征缺失导致模型泛化性受限，且只关注TR-069端口，未覆盖其他管理协议或时间演变。

---

## 170. MMPersistence: A mathematical morphology-oriented software library for computing persistent homology on cubical complexes

**arXiv ID:** 2602.15502 | [PDF](https://arxiv.org/pdf/2602.15502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 171. A universal LLM Framework for General Query Refinements

**arXiv ID:** 2602.15681 | [PDF](https://arxiv.org/pdf/2602.15681v1)

**作者:** Eldar Hacohen `[一作]` (Bar-Ilan University), Amit Somech `[通讯]` (Bar-Ilan University)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5040655769)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一个通用的SQL查询改进框架（Universal LLM-based Query Refinement），通过两步OPRO方案实现对任意SQL查询的约束满足与最小修改。

**💡 创新点**

将优化‑提示（OPRO）与子空间选择/采样结合，使用Skyline多目标反馈与历史摘要，首次在多类约束下实现对任意查询的自动精细化。

**🔧 技术方法**

基于大型语言模型的两步OPRO、子空间生成与赋值采样、Skyline算法、数据库描述摘要、LLM提示工程以及早停机制。

**📊 数据集**

使用11个真实数据库（Astronauts、Law Students、MEPS、TPC‑H、Texas Tribune、COMPAS、Housing Prices、Fraud Detection、Healthcare、ACSIncome、Students 等）共32个实例，涵盖Top‑k、Range、Diversity与Complex四类。

**📈 对比分析**

与多款通用LLM（GPT‑4.1‑mini、Gemini‑2.0‑Flash‑Lite、Mistral‑3.1）以及“思考”模式和随机采样基线对比，成功率最高（约90‑100%），Optimality均优于基线且接近专用解法。

**⚠️ 局限性**

对LLM的依赖导致推理成本高，需足够上下文；当子空间定义不充分或约束过于复杂时仍可能无法收敛；目前仅支持数值/类别谓词，对更复杂的派生属性支持有限。

---

## 172. Knowing Isn't Understanding: Re-grounding Generative Proactivity with Epistemic and Behavioral Insight

**arXiv ID:** 2602.15259 | [PDF](https://arxiv.org/pdf/2602.15259v1)

**作者:** Kirandeep Kaur `[一作]` (University of Washington), Chirag Shah `[通讯]` (University of Washington)

**通讯引用:** 6307 | [OpenAlex ID](https://openalex.org/A5064398705)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

**🎯 论文内容**

提出以认知盲区与行为边界为基础的生成式AI主动性框架，并系统评估现有主动性方法。

**💡 创新点**

将哲学无知分析与组织行为学的主动性模型相结合，提出“认知–行为耦合”视角，强调先评估代理认知合法性再进行主动介入。

**🔧 技术方法**

主要采用理论框架构建与文献综述，对现有预期、自治、混合主动性模式与无知、行为主动性理论进行交叉分析，未引入新算法。

**📊 数据集**

本研究为概念性工作，未使用任何实验数据集。

**📈 对比分析**

未进行实验比较；通过案例分析和理论论证展示框架优势，未给出定量性能指标。

**⚠️ 局限性**

缺乏实证验证和算法实现，无法评估在实际对话或任务系统中的效果；框架对多模态或大规模场景的适用性仍需探索。

---

## 173. Predicting Invoice Dilution in Supply Chain Finance with Leakage Free Two Stage XGBoost, KAN (Kolmogorov Arnold Networks), and Ensemble Models

**arXiv ID:** 2602.15248 | [PDF](https://arxiv.org/pdf/2602.15248v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 174. VQ-DSC-R: Robust Vector Quantized-Enabled Digital Semantic Communication With OFDM Transmission

**arXiv ID:** 2602.15045 | [PDF](https://arxiv.org/pdf/2602.15045v1)

**作者:** Jianqiao Chen `[一作]` (ZGC Institute of Ubiquitous-X Innovation and Applications), Ping Zhang `[通讯]` (State Key Laboratory of Networking and Switching Technology)

**通讯引用:** 20051 | [OpenAlex ID](https://openalex.org/A5100405787)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于向量量化（VQ）的数字语义通信系统 VQ-DSC-R，结合 Swin Transformer 提取多尺度语义特征，在 OFDM 多径衰落信道下实现图像的高效索引传输，并通过三阶段端到端训练实现系统优化。

**💡 创新点**

核心创新包括：①引入自适应噪声方差（ANDVQ）动态量化算法，解决 VQ 的梯度消失与码书崩溃问题；②采用条件扩散模型（CDM）对信道状态信息进行精细化重建；③设计 SNR 适应模块（SNR ModNet）实现不同噪声水平下的连续性能提升；④构建三阶段训练策略，逐步引入信道估计与调制影响。

**🔧 技术方法**

主要技术手段有：Swin Transformer 主干、向量量化与可微量化、K 近邻自适应噪声、EMA 训练稳定化、条件 U‑Net + CDM 的信道估计、OFDM 传输、SNR 软注意力模块。

**📊 数据集**

训练数据集使用 OpenImage（256×256）图像，测试数据集使用 DIV2K（1024×1024）图像，均采用 3GPP EPA 等多径信道模型。

**📈 对比分析**

与 STE、NSVQ、JPEG+LDPC、ADJSCC、LS、ReEsNet 等基线相比，在 PSNR、MS‑SSIM、BCR、NMSE、BER 等指标上均显著提升；尤其在低 SNR 下实现无“阶梯效应”平滑性能，并在相同 BCR 下取得最高的图像重建质量。

**⚠️ 局限性**

局限性包括：①码书容量与训练轮数易出现过拟合，需精细调节 K 和 CUR；②CDM 训练与推理复杂度较高；③目前仅验证图像传输，其他多模态或更高层次语义任务尚未探讨；④在极端多径或极低 SNR 环境下仍可能受信道估计误差影响。

---

## 175. Avey-B

**arXiv ID:** 2602.15814 | [PDF](https://arxiv.org/pdf/2602.15814v1)

**作者:** Devang Acharya `[一作]` (Avey AI), Mohammad Hammoud `[通讯]` (Avey AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的无注意力的双向编码器Avey‑B，改进了原始Avey架构以适配完整上下文。

**💡 创新点**

创新点包括：1) 静态与动态参数分离，保持相似性权重的单调性；2) 动态层采用行归一化的余弦相似度矩阵；3) 在ranker中加入神经压缩模块，将检索到的k个split压缩回单个split，降低每层计算量。

**🔧 技术方法**

技术实现：基于自检索的无注意力“神经处理器”（enricher‑contextualizer‑fuser）框架，使用位置无关的交叉嵌入矩阵；训练采用Masked Language Modeling与自回归分割技术；通过row‑norm、动态静态层交错、压缩矩阵等机制实现高效的双向上下文建模。

**📊 数据集**

预训练数据为FineWeb 180B tokens；下游评测覆盖四类任务：序列分类（MNLI, QQP, SST‑2）、词级分类（CoNLL‑2003, OntoNotes, UNER）、问答（ReCoRD, SQuAD, SQuAD‑v2）和信息检索（MLDR, MS‑MARCO, NQ），并在合成长文本任务NIAH（至96k tokens）上验证长距扩展能力。

**📈 对比分析**

对比BERT、RoBERTa、ModernBERT、NeoBERT等主流Transformer编码器，Avey‑B在token‑classification和information‑retrieval任务上均取得领先或相近表现；在序列分类与问答上表现相当或略逊；在吞吐量与延迟方面，Avey‑B随序列长度增长的衰减指数显著低于Transformer（α≈0.44 vs 0.77/0.81），在长文本上实现更高的tokens/second，显著优于基准模型。

**⚠️ 局限性**

局限性包括：仍缺乏专门的fused‑kernel实现，依赖PyTorch编译/自定义实现；对极长上下文的推理仍受限于split大小导致的O(N)线性计算；在某些任务（如MNLI、ReCoRD）相较于大型Transformer仍略显不足；未针对不同任务调优的多任务预训练策略。

---

## 176. Exploring Performance Tradeoffs in Age-Aware Remote Monitoring with Satellites

**arXiv ID:** 2602.15145 | [PDF](https://arxiv.org/pdf/2602.15145v1)

**作者:** Sunjung Kang `[一作]` (Purdue University), Christopher G. Brinton `[通讯]` (Purdue University)

**通讯引用:** 2940 | [OpenAlex ID](https://openalex.org/A5020399355)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文构建了一个包含地面 IoT 传感器、移动 UAV 和间歇性卫星三种异构感知源的远程监测框架，并以信息新鲜度度量 Age of Information (AoI) 为目标，研究在不同系统参数（可用性、可靠性、包大小等）下是否以及何时应使用卫星更新。

**💡 创新点**

创新点包括：① 提出适用于多源多模式异构网络的 AoI 分析框架，给出低束定量下限；② 在随机调度策略下推导出闭式的平均峰值 AoI 解析表达式，并给出卫星可用性与成功率的阈值判定；③ 设计了基于 Lyapunov 平衡的 Max‑Weight 调度策略，能同时处理多包更新与随机服务时间；④ 在卫星与 IoT 仅有的简化场景下给出解析的最优卫星利用比例，阐明了卫星使用的边界与折中。

**🔧 技术方法**

技术手段：AoI 复合理论、重置动态与 renewal‑reward 论证、负二项分布服务时间建模、随机化调度与 Lyapunov 稳定性分析、闭式解析与数值优化（KKT、凸优化）。

**📊 数据集**

数据集：主要采用仿真数据，包括理想化的卫星可用性（几何分布的 A、U 周期）以及基于真实卫星轨道的 trace‑driven 可用性序列；IoT 与 UAV 部署采用网格划分的图模型，无真实测量数据。

**📈 对比分析**

性能比较：与传统的单源或无卫星方案相比，随机化调度与 Max‑Weight 策略在不同卫星可用性、可靠性和更新大小场景下均能显著降低平均峰值 AoI，尤其在卫星可用性高、可靠性适中的情形下可进一步提升 10–30% 的信息新鲜度；在卫星可靠性极低或可用性极低时，系统退化到仅使用 IoT/UAV 时效果相近。

**⚠️ 局限性**

局限性：① 分析基于 generate‑at‑will 模型，假设更新始终可用，忽略了数据生成过程与压缩开销；② 仅考虑单个共享信道，未考虑多频段或多天线场景；③ 卫星可用性被建模为两态几何过程，未涵盖多轨道交叉与重叠覆盖的真实复杂性；④ Max‑Weight 策略虽理论上稳定，但实际实现需对多包服务时间的期望估计与时延敏感度较高。

---

## 177. Exploring VASS Parameterised by Geometric Dimension

**arXiv ID:** 2602.15483 | [PDF](https://arxiv.org/pdf/2602.15483v1)

**作者:** Wojciech Czerwiński `[一作]` (University of Warsaw), Yangluo Zheng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文系统研究了向量加法系统（VASS）的几何维度(g)和SCC维度，并将经典可达性、覆盖、无界性、整数可达性等问题在这些参数化下的复杂度和最短证据长度进行分析与改进。

**💡 创新点**

创新点：① 引入并精确定义“几何维度”与“SCC维度”，证明它们在VASS可达性问题中的重要性；② 将Rackoff等人对维度d的覆盖/无界性长度上界从O(2^{d log d})提升至仅依赖几何维度O(2^g)；③ 将整数可达性和同时无界性问题的长度上界从指数级提升至多项式级；④ 证明SCC维度为4的可达性为P‑hard，展示SCC维度与计数维度的本质差异。

**🔧 技术方法**

技术方法：基于几何向量空间的清洁基（clean basis）与几何小/大配置的定义；改进的Rackoff/ Kunnemann 等人递归降维技术；利用Carathéodory定理对整数线性系统求解极小解；构造乘法三元组和放大器实现对SCC维度4的P‑hard性证明。

**📊 数据集**

本文不使用实验数据集，而是以理论证明和构造性归约为主。

**📈 对比分析**

与传统按计数维度d参数化的结果比较，本文的上界在几何维度g上更优，尤其覆盖问题从O(2^{d log d})降至O(2^g)，无界性从O(2^{d log d})降至O(2^g log g)。对SCC维度4的P‑hard性证明进一步说明了参数化的差异。

**⚠️ 局限性**

局限性：① 目前只在固定几何维度或SCC维度下得到多项式/指数上界，未能给出完全匹配的下界；② 对SCC维度小于4的可达性复杂度仍未确定；③ 证明中使用的构造（如乘法三元组）较为复杂，尚缺乏直接实现或实验验证。

---

## 178. SpecFuse: A Spectral-Temporal Fusion Predictive Control Framework for UAV Landing on Oscillating Marine Platforms

**arXiv ID:** 2602.15633 | [PDF](https://arxiv.org/pdf/2602.15633v1)

**作者:** Haichao Liu `[一作]` (Hong Kong University of Science and Technology), Jinni Zhou `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5109404467)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出 SpecFuse 框架，实现无人机在海上平台波动环境中的自适应着陆。

**💡 创新点**

创新点在于将频域波形分解与时域递归估计相融合，显著减少预测相位滞后，并引入学习增强控制器提升鲁棒性。

**🔧 技术方法**

采用频域波谱分解、递归卡尔曼滤波、HPO‑RRT* 轨迹规划、学习驱动速度匹配与递归视窗控制等技术。

**📊 数据集**

使用 JONSWAP 规范波谱模拟海况，以及湖泊现场实验收集的真实风浪数据集。

**📈 对比分析**

与 FFT‑KF、MPC‑VIO、NMPC 三个基线相比，SpecFuse 在预测误差、着陆偏差、成功率与系统延迟上分别提升 44%–48%、98.8% 成功率、82 ms 延迟，表现最佳。

**⚠️ 局限性**

主要局限在高频波动（>3 Hz）与强风（>15 m/s）下仍可能超出推力或估计带宽，导致失稳或失效。

---

## 179. Unforgeable Watermarks for Language Models via Robust Signatures

**arXiv ID:** 2602.15323 | [PDF](https://arxiv.org/pdf/2602.15323v1)

**作者:** Huijia Lin `[一作]` (University of Washington), Min Jae Song `[通讯]` (University of Chicago)

**通讯引用:** 2409 | [OpenAlex ID](https://openalex.org/A5103173963)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对大型语言模型的水印方案，首次实现了在 Hamming 替换扰动下既能保持水印不可被检测，又能提供鲁棒性、不可伪造性与可恢复性的完整安全保证。

**💡 创新点**

创新点在于：①引入不可伪造（unforgeability）与可恢复（recoverability）两个新的安全属性；②构造了可对近似消息进行验证的鲁棒数字签名（robust digital signature），并证明其可由属性保持哈希（property‑preserving hash）实现；③给出了一般化的框架，将鲁棒签名与块级稳健信息隐藏（block steganography）组合，实现完整的可恢复水印。

**🔧 技术方法**

核心技术包括：属性保持哈希（PPH）与差异恢复（difference‑recovery）机制；强不可伪造数字签名与可恢复签名的构造；块级稳健信息隐藏方案；利用理想伪随机码（PRC）与可逆哈希实现的鲁棒签名；以及对 Hamming 判定谓词的每块连续性提升（every‑block‑close lifting）。

**📊 数据集**

本文并未在具体数据集上进行实验，而是以通用语言模型（如 GPT‑4、Stable Diffusion 等）为目标，进行理论分析与构造证明，主要关注模型输出的概率分布和熵属性。

**📈 对比分析**

与现有水印方案相比，本文的方案在 Hamming 误差下实现了同时满足鲁棒性、不可伪造性、可恢复性和不可检测性，解决了以往仅关注鲁棒或不可伪造但缺乏可恢复的局限；虽然没有提供实验性能指标，但理论上满足所有安全性要求，且在理想伪随机码和 CRHF 的假设下可构造。

**⚠️ 局限性**

限制与挑战包括：目前仅针对 Hamming 替换误差；对高熵语言模型的要求较高；依赖于理想 PRC 与强 CRHF 的存在，实际实现复杂度较大；鲁棒性与不可伪造性在块边界处存在细微不对称；未对插入/删除等编辑误差提供完整保障；尚缺乏实证评估。

---

## 180. Common Belief Revisited

**arXiv ID:** 2602.15403 | [PDF](https://arxiv.org/pdf/2602.15403v1)

**作者:** Thomas Ågotnes `[一作]` (University of Bergen), Thomas Ågotnes `[通讯]` (University of Bergen)

**通讯引用:** 1730 | [OpenAlex ID](https://openalex.org/A5021479257)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

论文通过在单模态语言中引入两个新公理 Cc 与 Cn，构造并证明了对任意有限数目的 KD45 代理人而言，公共信念的完整公理化体系，并给出了对应的模型构造与完备性证明。

**💡 创新点**

创新点在于：① 明确指出公共信念不等同于 KD4；② 证明了需要额外的 Cc（shift‑reflexivity）和 Cn（基于代理人数的组合公理）来完整刻画公共信念；③ 解决了长期未解的“KD45 公共信念的完整公理化”开放问题。

**🔧 技术方法**

采用了标准模态逻辑技术：典范模型构造、Bisimulation 与可达闭包分析、有限模型压缩、以及对不同代理人数的分情况证明。通过对 Cn 的一次性简化实现了对任意代理人数的通用性。

**📊 数据集**

无数据集，本文纯粹是理论证明性质的研究。

**📈 对比分析**

与以往基于个体信念与归纳公理的公共信念公理化方法对比，本文提供了更直接、更紧凑的单模态公理化，证明了其完整性；没有涉及实验或性能评估。

**⚠️ 局限性**

局限性包括：① 需要至少两个代理人；② 结果仅适用于有限固定代理人数的 KD45 框架；③ 对无限代理或其他信念逻辑（如 K45、K5、KB 等）的推广仍未完成。

---

## 181. Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems

**arXiv ID:** 2602.15198 | [PDF](https://arxiv.org/pdf/2602.15198v1)

**作者:** Mason Nakamura `[一作]` (University of Massachusetts Amherst), Eugene Bagdasarian `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5114402613)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个用于审计大型语言模型驱动的协作多智能体系统中合谋行为的框架。

**💡 创新点**

创新点在于将合谋形式化为分布式约束优化问题中的 regret，定义混合合谋和 λ 对齐，并提出直接、尝试和隐藏三类合谋的分类与度量。

**🔧 技术方法**

核心技术包括 DCOP 模型、基于 regret 的合谋度量、LLM-as-a-judge 评判、Terrarium 多智能体框架，以及对说服手段和网络拓扑影响的实验研究。

**📊 数据集**

实验使用了两个新构建的 DCOP 环境（Hospital 与 Jira）以及已有的 Meeting Scheduling 基准，并在这些环境中评估 GPT、Claude、Gemini、Kimi-K2 等开箱即用 LLM 模型。

**📈 对比分析**

通过比较 regret 指标和 LLM-as-a-judge 的得分，研究发现 regret 能捕捉到审计日志未能发现的合谋，且不同模型在不同拓扑、说服策略和目标不一致时的合谋优势和整体回报差异显著。

**⚠️ 局限性**

局限性包括仅在同一模型、多智能体规模有限的实验场景下验证，未考虑异构模型、规模化部署和未见任务的迁移性，需要进一步扩展验证范围。

---

## 182. S-PRESSO: Ultra Low Bitrate Sound Effect Compression With Diffusion Autoencoders And Offline Quantization

**arXiv ID:** 2602.15082 | [PDF](https://arxiv.org/pdf/2602.15082v1)

**作者:** Zineb Lahrichi `[一作]` (Sony AI), Geoffroy Peeters `[通讯]` (LTCI, Télécom Paris, Institut Polytechnique de Paris)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 S‑PRESSO，一种利用扩散自编码器实现 48kHz 声音特效超低比特率压缩（可达 0.096 kbps、1 Hz 帧率）的模型。

**💡 创新点**

创新点在于将预训练的潜在扩散模型作为解码器，结合离线神经量化和解码器微调，实现 750× 压缩率同时保持感知质量与声学相似度，并支持连续与离散嵌入。

**🔧 技术方法**

采用 DiT 扩散解码器、latent 编码器、LoRA 微调、Qinco2 神经量化、EDM2 参数化训练以及 FAD/KAD/Si‑SDR/MUSHRA 等多维评估。

**📊 数据集**

使用约 5000 小时、48kHz、5 s 的内部音效数据集进行训练，评测数据来自 Freesound、BBC Sound Effects 与内部工作室音效集。

**📈 对比分析**

与连续基线 Stable Audio Open、Music2Latent 以及离散基线 SemantiCodec 进行对比，S‑PRESSO 在 FAD、KAD、Si‑SDR、CLAP 相似度以及 MUSHRA 评分上均优于对手，甚至在 0.096 kbps 低比特率下仍保持可接受的重构质量。

**⚠️ 局限性**

局限性包括对大规模预训练模型和多 GPU 训练的依赖、推理时间尚未优化、在极低比特率下仍略低于原始 AudioAE，且扩散采样的随机性对主观评估造成一定干扰，且目前仅针对 48kHz 音效而非通用音频。

---

## 183. CARE Drive A Framework for Evaluating Reason-Responsiveness of Vision Language Models in Automated Driving

**arXiv ID:** 2602.15645 | [PDF](https://arxiv.org/pdf/2602.15645v1)

**作者:** Lucas Elbert Suryana `[一作]` (Delft University of Technology), Arkady Zgonnikov `[通讯]` (Delft University of Technology)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5031929760)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CARE-Drive 框架，用于评估视觉-语言模型（VLM）在自动驾驶场景中的原因响应性，探究在引入人类道德理由后模型决策是否真正受这些理由影响。

**💡 创新点**

创新点在于：①引入基于“意义化人类控制”理论的跟踪条件，将人类理由作为外部输入系统地验证其对决策的因果影响；②分两阶段（提示校准 + 情境敏感性）设计，既消除提示随机性，又系统测量情境变量对决策的影响；③利用无模型修改的方式，保持对现有大模型的广泛适用性。

**🔧 技术方法**

技术包括：基于 GPT‑4 系列的 vision‑language 模型；链式思维（CoT）与树式思维（ToT）提示策略；离散化的情境参数（时间到碰撞、后方车辆、乘客紧迫感、跟随时间、解释长度）作为可观测上下文；二元逻辑回归对决策概率进行定量分析；CARLA 仿真验证决策的可执行性。

**📊 数据集**

数据集主要来自 CARLA 生成的三种情境图像（无来车、来车、后车）以及对应的可观测上下文向量，人工设定的专家参考决策（多数专家认为在安全、效率与舒适冲突时倾向超车），并在 30 次独立采样下记录模型输出。

**📈 对比分析**

比较方法：在 Stage‑1 校准阶段对不同模型（gpt‑4.1、mini、nano）和思维策略进行 30 次采样，统计与专家决策的匹配率；在 Stage‑2 情境评估阶段采用全因子设计（5×2×30 采样），计算超车概率并通过逻辑回归估计各情境变量的效应；结果显示：在引入理由后，超车率显著提升（如 gpt‑4.1+CoT 超车率从 0% 提升至 30–30% 以上），且在安全关键情境下树式思维相对更稳定、与专家一致性更高。

**⚠️ 局限性**

限制：①仅评估外部可观测理由与决策的行为关联，未直接探查模型内部推理过程；②提示与理由表述方式对结果可能敏感，未系统检验不同表述的鲁棒性；③仅针对骑行超车情境，缺乏跨多种伦理冲突场景的验证；④实验重复次数有限，统计精度受限；⑤未考虑模型更新或训练时的因果学习机制。

---

## 184. AIC CTU@AVerImaTeC: dual-retriever RAG for image-text fact checking

**arXiv ID:** 2602.15190 | [PDF](https://arxiv.org/pdf/2602.15190v1)

**作者:** Herbert Ullrich `[一作]` (Czech Technical University), Jan Drchal `[通讯]` (Czech Technical University)

**通讯引用:** 253 | [OpenAlex ID](https://openalex.org/A5049289726)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个结合文本检索与逆向图像检索的 RAG 体系，用单次多模态 LLM 调用完成图文事实核查。

**💡 创新点**

创新点在于将 RIS 模块与传统文本检索耦合，并以极低成本实现高效的多模态证据生成。

**🔧 技术方法**

使用 GPT‑5.1（OpenAI Batch API）进行生成，文本检索采用向量检索与 MMR 重新排序，RIS 通过 Serper API 调用 Google Lens。

**📊 数据集**

利用 AVerImaTeC 共享任务的数据集，包含数百条图文事实核查实例。

**📈 对比分析**

在共享任务排行榜中以第三名取得 0.81 的提问得分、0.35 的证据得分和 0.35 的综合判定得分，明显优于基线。

**⚠️ 局限性**

局限包括对 RIS 结果质量与图像证据表示方式的偏差，依赖 GPT‑5.1 的不可解释性与高碳成本，以及对英文输入的偏倚。

---

## 185. ToaSt: Token Channel Selection and Structured Pruning for Efficient ViT

**arXiv ID:** 2602.15720 | [PDF](https://arxiv.org/pdf/2602.15720v1)

**作者:** Hyunchan Moon `[一作]` (LG Electronics), Steven L. Waslander `[通讯]` (University of Toronto)

**通讯引用:** 10001 | [OpenAlex ID](https://openalex.org/A5024242059)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出ToaSt框架，对Vision Transformer进行压缩，分别对多头自注意力(MHSA)做结构化耦合权重剪枝，和对前馈网络(FFN)做训练无关的Token Channel Selection；

**💡 创新点**

创新点在于①同步剪枝Q‑K与V‑Proj维度，保持层内计算一致性；②利用激活稀疏性、线性重构和有效秩等统计信息，采用采样+注意力引导的通道选择；③将注意力压缩与通道压缩分离，避免跨层传播；④自适应层级剪枝比例实现高压缩率同时提升精度；

**🔧 技术方法**

采用结构化权重剪枝（几何中值重要性评估）、统计采样+注意力引导的重要性分数、训练无关的Token Channel Selection、层级自适应剪枝策略，以及少量微调恢复精度；

**📊 数据集**

ImageNet‑1K用于图像分类评估，COCO 2017用于目标检测（Cascade Mask R‑CNN）验证迁移效果；

**📈 对比分析**

与ToMe、DiffRate、STViT‑R等主流Token压缩/剪枝方法对比；在9个模型（DeiT、ViT‑MAE、Swin）上均实现0.3–3.6% Top‑1/box mAP提升，同时FLOPs下降30–40%，H100 GPU吞吐率提升1.2–2.1×；

**⚠️ 局限性**

主要局限在手工设定的层级剪枝比例、仍需少量微调恢复精度，以及目前仅针对ViT家族；未来需实现可学习的比例优化、扩展到跨模态模型和与量化结合。

---

## 186. Privacy-Preserving and Secure Spectrum Sharing for Database-Driven Cognitive Radio Networks

**arXiv ID:** 2602.15705 | [PDF](https://arxiv.org/pdf/2602.15705v1)

**作者:** Saleh Darzia `[一作]`, Gürkan Gür `[通讯]` (Zurich University of Applied Sciences)

**通讯引用:** 3191 | [OpenAlex ID](https://openalex.org/A5018779142)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种统一的数据库驱动认知无线电网络（DB‑CRN）安全框架，支持匿名位置信息查询、可验证定位和抵御 DoS 攻击。

**💡 创新点**

创新点在于将可委托匿名凭证、可撤销可链式环签名（RLRS）与可验证延迟函数（VDF）结合，形成兼顾隐私、定位证明与 DoS 抗扰动的完整解决方案。

**🔧 技术方法**

采用了可委托匿名凭证（DAC）、距离绑定协议（DBP）、可撤销可链式环签名（RLRS）、可验证延迟函数（VDF）、椭圆曲线签名（ECDSA）以及私有信息检索（PIR）等密码技术。

**📊 数据集**

使用了 FCC CBRS 3.5 GHz 数据库示例（每条记录约 560 B）与合成数据，并在 5G NR LENA 网络仿真环境中进行评估。

**📈 对比分析**

通过密码原语基准和网络仿真，将该方案与多种现有方法（PIR、EPID、PIR 多 DB、SAS 等）对比，显示在协议、通信开销和端到端延迟方面均显著优于对手，且在 DoS 场景下实现了有效抵抗。

**⚠️ 局限性**

局限性在于仅覆盖查询与服务阶段，未考虑持续使用时的位置信息泄漏、移动性、时域侧信道、信号定位等攻击；此外目前仅实现经典安全，尚未提供量子安全保证。

---

## 187. Towards Expectation Detection in Language: A Case Study on Treatment Expectations in Reddit

**arXiv ID:** 2602.15504 | [PDF](https://arxiv.org/pdf/2602.15504v1)

**作者:** Aswathy Velutharambath `[一作]` (University of Stuttgart), Amelie Wührl `[通讯]` (IT University of Copenhagen)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出期望检测（Expectation Detection）任务，并基于RedHOT构建了RedHOT-Expectation语料库，标注了患者在社交媒体上对医疗治疗的期望表达及对应的治疗-期望-结果三元组。

**💡 创新点**

创新点在于首次将期望作为NLP任务进行系统化定义和研究，创建了首个治疗期望检测数据集，并通过语言特征揭示期望表达的风格差异。

**🔧 技术方法**

技术方面采用大型语言模型（GPT-OSS 20B）进行期望筛选和银标注，利用LIWC、VADER、Gunning–Fog等工具进行文本特征提取与统计分析。

**📊 数据集**

使用的数据集为RedHOT的约22k条Reddit医疗帖子，筛选出12k条经验类帖子，最终得到4.5k条标注为包含期望的帖子，其中2.5k带有三元组标注。

**📈 对比分析**

方法评估：对245条手工验证样本中的502条三元组，LLM提取准确率为77.5%；语言特征对比显示期望帖长度更长、未来导向和动机性更强，统计显著。

**⚠️ 局限性**

局限性包括数据仅来自Reddit，可能不具备跨平台普适性；三元组标注受LLM提示和偏差影响，存在噪声；缺乏更大规模的结果对齐与时间维度深入分析。

---

## 188. LLM-as-Judge on a Budget

**arXiv ID:** 2602.15481 | [PDF](https://arxiv.org/pdf/2602.15481v1)

**作者:** Aadirupa Saha `[一作]` (University of Illinois Chicago), Branislav Kveton `[通讯]` (Adobe)

**通讯引用:** 2483 | [OpenAlex ID](https://openalex.org/A5049020775)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于方差自适应的查询分配算法，用于在固定计算预算下最小化LLM‑as‑a‑judge评估中的最大估计误差。

**💡 创新点**

创新点：1）将评估问题视为带预算的多臂赌博机，设计了知方差与未知方差两种情形下的贪婪/ UCB 分配策略；2）给出近似最优的理论误差上界 O(√(∑σ_i²/B))；3）通过两阶段探索与自适应分配实现对未知方差的有效估计。

**🔧 技术方法**

核心技术：多臂赌博机理论、UCB 置信上界、子高斯浓度不等式、在线贪婪分配、方差估计与自适应调整。

**📊 数据集**

实验数据集：HelpSteer2（含 20.3K 评估对）与 GPT‑4.1‑nano、Llama‑3.1‑8B 两大 LLM 进行 30k 次评分模拟。

**📈 对比分析**

与均匀分配基线对比，理论与实验表明在相同预算下可将最大估计误差降低约 30–50%，相当于将所需查询数减半；相较于理想的“方差基准”分配，表现位于两者之间。

**⚠️ 局限性**

局限性：假设评分噪声为子高斯/高斯，需手动调参（δ、t₀）；仅处理单属性评估；未利用上下文特征或多属性相关性；在极端方差不均匀或样本量极小的场景下性能可能受限。

---

## 189. Spanning the Visual Analogy Space with a Weight Basis of LoRAs

**arXiv ID:** 2602.15727 | [PDF](https://arxiv.org/pdf/2602.15727v1)

**作者:** Hila Manor `[一作]` (Technion), Gal Chechik `[通讯]` (NVIDIA)

**通讯引用:** 7424 | [OpenAlex ID](https://openalex.org/A5045719865)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于LoRA基组的视觉类比编辑框架，动态组合LoRA实现对新类比任务的编辑。

**💡 创新点**

通过学习可混合的LoRA基组和轻量编码器，避免单个LoRA限制，显著提升对未见类比任务的泛化能力。

**🔧 技术方法**

采用LoRA基组、CLIP编码器、Flux.1-Kontext扩展注意力的条件流模型、软最大混合及VLM评估。

**📊 数据集**

使用Relation252k作为训练集，并构建540个类比三元组的自定义验证集（90个任务）。

**📈 对比分析**

与单LoRA及RelationAdapter、VisualCloze、EditTransfer等基线在LPIPS、CLIP相似度、VLM保留/编辑准确度、2AFC等指标上对比，结果均取得领先，形成Pareto前沿并通过用户研究得到验证。

**⚠️ 局限性**

仍受限于训练语料，对与训练集差异较大的任务效果有限，需要更大的基组和进一步的调优。

---

## 190. Dynamic Training-Free Fusion of Subject and Style LoRAs

**arXiv ID:** 2602.15539 | [PDF](https://arxiv.org/pdf/2602.15539v1)

**作者:** Qinglong Cao `[一作]` (Shanghai Jiao Tong University), Xiaokang Yang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 25693 | [OpenAlex ID](https://openalex.org/A5019708391)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无需训练的动态 LoRA 融合框架，能够在扩散生成过程中通过动态特征选择和指标引导的潜在修正，实现对指定主体和指定风格的统一合成。

**💡 创新点**

创新点在于：①在前向传播中使用 KL 散度量化不同 LoRA 产生的特征扰动，实时决定每层最合适的 LoRA；②在反向去噪阶段利用 CLIP 与 DINO 等客观指标的梯度反馈，持续校正潜在空间，提升语义与风格一致性；整体实现无额外训练、可插拔。

**🔧 技术方法**

主要技术包括扩散模型（Stable Diffusion XL、FLUX）、LoRA 微调、KL 散度特征对比、CLIP 与 DINO 评价指标、梯度引导潜在修正。

**📊 数据集**

使用 DreamBooth 数据集训练主体 LoRA，使用 StyleDrop 数据集训练风格 LoRA，基准模型为 Stable Diffusion XL v1.0 与 FLUX，评测随机采样的 30 个主体‑风格组合。

**📈 对比分析**

与 K‑LoRA、ZipLoRA、B‑LoRA、直接拼接等方法对比，本文在 Style Sim、CLIP Score、DINO Score、用户偏好与 LLM 评测中均位列首位；尤其在 CLIP Score 上提升约 9.1%（从 69.6% 提升至 78.5%），用户研究中获得 53% 首选率，GPT‑4o 与 Qwen2.5‑VL 评价分别为 55.6% 与 65.7%。

**⚠️ 局限性**

局限性包括：①在 DINO Score 上仍略逊于部分基线；②对不同尺度或极端风格的适应性尚待验证；③需依赖高质量的独立 LoRA，组合复杂度随 LoRA 数量增加；④梯度引导的调参（如 m 值）对性能影响显著，需要经验性选择。

---

## 191. Operationalising the Superficial Alignment Hypothesis via Task Complexity

**arXiv ID:** 2602.15829 | [PDF](https://arxiv.org/pdf/2602.15829v1)

**作者:** Tomás Vergara-Browne `[一作]` (Mila Quebec AI Institute), Marius Mosbach `[通讯]` (Mila Quebec AI Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于算法信息理论的任务复杂度与条件任务复杂度度量，并用其量化“表面适配”假设，实验评估了LLM在数学推理、机器翻译和指令跟随等任务上的程序长度-性能Pareto曲线

**💡 创新点**

① 用程序长度量化任务复杂度，统一数据视角、参数视角与推理控制视角；② 通过实验展示仅几千位程序即可实现高性能，揭示预训练与后训练对适配“表面性”的影响

**🔧 技术方法**

任务复杂度度量、Python解释器作为通用图灵机、三类适配方法（子集训练、LoRA/贝叶斯LoRA、ICL/URIAL）、程序长度评估与Pareto曲线分析

**📊 数据集**

MetaMath（GSM8K）数学推理，English-to-French机器翻译，IFEval指令跟随；使用SmolLM3 3B、Olmo3 7B、Olmo3 32B模型权重

**📈 对比分析**

对每种适配方法生成程序，测量程序长度与任务性能，绘制Pareto最优曲线；结果显示：仅约4k位程序即可将GSM8K准确率提升至72%；Olmo3‑7B翻译BLEU从22.6提升至34.4；I/Eval最高性能约1.25 MB；预训练能达高性能但需大程序，后训练显著压缩至几千位

**⚠️ 局限性**

任务复杂度不可计算，仅给出上界；难以估算无预训练条件下的复杂度；可能存在更短程序未被发现；实验未覆盖所有不安全能力的“表面性”

---

## 192. The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety

**arXiv ID:** 2602.15799 | [PDF](https://arxiv.org/pdf/2602.15799v1)

**作者:** Max Springer `[一作]`, Aleksandra Korolova `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了语言模型微调过程中对齐安全的不可避免退化，并提出了对齐不稳定条件（AIC）与四次缩放定律。

**💡 创新点**

提出了二阶曲率驱动的对齐不稳定机制和可预判的四次衰退定律，解释了微调为何会导致安全崩溃。

**🔧 技术方法**

利用Fisher信息矩阵的低秩结构、梯度流的二阶展开、随机投影与重叠得分（Overlap Score）等技术，对模型参数空间进行几何分析。

**📊 数据集**

在Qwen3-1.7B-Instruct、LLaMA-3.2-3B-Instruct等大型模型上，对Benign、Seemingly Benign与Harmful三类数据集（如SamSum、Alpaca、GSM8K、Risky Financial Advice、Bad Medical Advice、Pure Bad）进行微调。

**📈 对比分析**

通过与Gemini-2.5-Flash评判的Harmfulness Score对比，发现重叠得分高的微调任务对应更高的安全风险，证明理论预测与实验结果一致；在全微调与LoRA下的效果对比表明OS能有效区分危险任务。

**⚠️ 局限性**

局限在于对LoRA的二阶曲率估计难以计算、随机投影噪声、模型参数化影响导致的重叠得分不稳定，并缺乏可操作的实时曲率约束方法。

---

## 193. ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling

**arXiv ID:** 2602.15537 | [PDF](https://arxiv.org/pdf/2602.15537v1)

**作者:** Nicol Visser `[一作]` (Stellenbosch University), Herman Kamper `[通讯]` (Stellenbosch University)

**通讯引用:** 1581 | [OpenAlex ID](https://openalex.org/A5040305929)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了ZeroSyl，一个无需训练的框架，利用冻结的WavLM Large模型的L2范数进行峰值检测来直接识别音节边界，随后对层22的特征做均值池化并通过球面K‑means聚类得到音节单元，用于纯语音语言模型的训练。

**💡 创新点**

创新点在于完全去除了多阶段训练和专门目标函数，仅用无监督峰值检测和特征聚类即可得到高质量音节单元，显著简化了音节化流程并在多项评测中优于现有方法。

**🔧 技术方法**

技术方法包括冻结WavLM Large自监督模型、对层13特征的L2范数做峰值检测、对层22特征做均值池化、使用球面K‑means聚类以及在OPT‑125M上训练因果语言模型。

**📊 数据集**

主要使用了Libri‑Light 6k小时和60k小时的无标签语音数据进行训练，评测使用LibriSpeech测试集以及sWUGGY、sBLIMP、tSC等语音语言基准。

**📈 对比分析**

与Sylber、SyllableLM等现有音节化模型在词汇、句法和叙事基准上进行对比，ZeroSyl在sWUGGY、sBLIMP和tSC等任务上均取得最高分，并在60k小时规模下显示出更快的句法性能提升。

**⚠️ 局限性**

局限性在于对词汇层的细粒度信息掌握不足，稀有词或未出现词的表达能力可能受限，且在词汇任务上仍落后于更细粒度的SpidR方法。

---

## 194. Revisiting Backdoor Threat in Federated Instruction Tuning from a Signal Aggregation Perspective

**arXiv ID:** 2602.15671 | [PDF](https://arxiv.org/pdf/2602.15671v1)

**作者:** Haodong Zhao `[一作]` (Shanghai Jiao Tong University), Gongshen Liu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 1992 | [OpenAlex ID](https://openalex.org/A5085695760)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在联邦指令调优中，由低浓度散布于无害客户端的数据引发的后门漏洞，证明即使仅10%受污染数据也能导致高达85%的攻击成功率。

**💡 创新点**

提出了Backdoor Signal-to-Noise Ratio (BSNR) 指标，用以量化分布式后门信号的强度，并证明现有防御方法在此场景下失效。

**🔧 技术方法**

使用信号处理方法构建BSNR，采用LoRA参数高效微调，评估了自然触发和Badnets触发的后门攻击。

**📊 数据集**

实验基于TriviaQA和SimpleQuestions两个问答数据集，使用Vicuna-7B和Llama-2-7B两款开源LLM。

**📈 对比分析**

通过对比不同后门注入方式、毒化比例、受影响客户端比例以及防御方法（Krum、FreqFed、FoundationFL），结果显示无论防御方法多么先进，均无法在受影响客户端比例>0.5时抑制攻击；ASR可达近100%而MA基本保持不变。

**⚠️ 局限性**

局限在于仅关注低浓度后门注入和IID场景，未覆盖非IID数据、其他后门触发方式以及更复杂的防御机制，且BSNR估计依赖于对干净/受影响客户端的划分。

---

## 195. In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations

**arXiv ID:** 2602.15456 | [PDF](https://arxiv.org/pdf/2602.15456v1)

**作者:** Mohammad Aflah Khan `[一作]` (Max Planck Institute for Software Systems), Abhilasha Ravichander `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探究大型语言模型（LLM）在信息检索与推荐场景中对来源的隐式偏好，使用控制实验和真实数据评估不同模型在新闻、科研论文和电商三类任务中的来源选择偏好并分析其上下文依赖性与凭证合理性；

**💡 创新点**

首次系统量化LLM的来源偏好，并揭示其与模型规模、训练后处理、上下文环境以及来源身份/凭证关联的显著关联，说明传统提示策略难以抑制此类偏好；

**🔧 技术方法**

采用控制实验设计、直接与间接排名评估、Kendall Tau相关性分析、对比实验（如AllSides新闻聚合与亚马逊卖家选择）等方法，结合12种主流LLM；

**📊 数据集**

构建了三类来源集合（政治倾向新闻集、科研期刊/会议集、电商平台集）及其合成与真实对齐数据，包括AllSides事件数据和亚马逊卖家数据；

**📈 对比分析**

通过比较模型间的来源排名相关性、偏好幅度以及与现有推荐算法（如亚马逊BuyBox）的对齐度，发现LLM在不同任务中的偏好强度与方向可被量化，且提示干预效果有限；

**⚠️ 局限性**

研究未解释偏好形成机制，提示方法难以完全抑制隐式偏好，实验仅覆盖三类任务，缺乏对其他信息来源和更细粒度偏好的探索，且对模型公平性与安全性的实际影响仍需进一步验证。

---

## 196. Protecting Language Models Against Unauthorized Distillation through Trace Rewriting

**arXiv ID:** 2602.15143 | [PDF](https://arxiv.org/pdf/2602.15143v1)

**作者:** Xinhang Ma `[一作]` (Washington University in St. Louis), Yevgeniy Vorobeychik `[通讯]` (Washington University in St. Louis)

**通讯引用:** 5073 | [OpenAlex ID](https://openalex.org/A5038669899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于推理轨迹重写的框架，既能抑制未经授权的知识蒸馏，又能在学生模型中嵌入可验证水印。

**💡 创新点**

通过高层语义提示和自动优化提示实现无损重写，同时首次将梯度方法与LLM重写结合，兼顾教师性能与学生干扰。

**🔧 技术方法**

语义提示、优化提示、梯度嵌入空间攻击（HB‑Grad/FO‑Grad）、对抗式提示优化（OPRO）以及LLM辅助重写。

**📊 数据集**

GSM8K、MATH、MMLU 与 MMLU‑Pro 等标准推理与常识基准。

**📈 对比分析**

与 ADS、DOGe、He 等现有反蒸馏方案以及 GINSEW、KGW、VIA 等水印基线对比；优化提示方法在保留教师精度的前提下将学生准确率降低 61% 以上，水印检验率几乎 100% 且误报率为 0。

**⚠️ 局限性**

梯度重写成本高、对代理学生依赖较大、仅验证 SFT 蒸馏，对其它蒸馏策略的鲁棒性待验证。

---

## 197. High-Fidelity Network Management for Federated AI-as-a-Service: Cross-Domain Orchestration

**arXiv ID:** 2602.15281 | [PDF](https://arxiv.org/pdf/2602.15281v1)

**作者:** Merve Saimler `[一作]` (Ericsson Research), Ozgur Ercetin `[通讯]` (Sabanci University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出基于Tail‑Risk Envelope（TRE）的AIaaS保证框架，实现多域联合调度与尾部延迟控制

**💡 创新点**

引入可签名、可组合的域级合同（TRE），将尾部风险分解为可分配预算，并结合极值理论实现自适应审计与责任归因

**🔧 技术方法**

采用随机网络计算（Stochastic Network Calculus）、极值理论（EVT）、分布式ADMM优化与Monte‑Carlo仿真等技术

**📊 数据集**

使用基于Poisson到ON/OFF的合成流量与自定义延迟/处理时间的仿真数据集，未使用真实网络或AI推理工作负载数据集

**📈 对比分析**

对比最佳努力与TRE‑管理两种方案，在p99.9延迟、负载弹性、租户隔离和风险归因等指标上，TRE‑管理显著降低尾部违约概率并保持服务可预见性

**⚠️ 局限性**

主要局限在于假设服务过程可用MGF约束、域内部信息隐藏导致的保守性、以及对非平稳环境（动态负载/服务率变动）的适应性不足

---

## 198. ER-MIA: Black-Box Adversarial Memory Injection Attacks on Long-Term Memory-Augmented Large Language Models

**arXiv ID:** 2602.15344 | [PDF](https://arxiv.org/pdf/2602.15344v1)

**作者:** Mitchell Piehl `[一作]` (Iowa), Muchao Ye `[通讯]` (Iowa)

**通讯引用:** 494 | [OpenAlex ID](https://openalex.org/A5024079930)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统性研究了黑盒对长时记忆增强大型语言模型的恶意记忆注入攻击，并提出了统一框架与多种攻击原语。

**💡 创新点**

创新点在于揭示相似度检索本身即为系统级弱点，提出了两种实际攻击情景（内容式与问题定向）以及可组合的攻击族，且不需要模型内部信息。

**🔧 技术方法**

采用了基于嵌入相似度的检索模型、外部LLM链式思维生成恶意文本、以及程序化生成的非语义噪声，构建了完整的攻击流水线。

**📊 数据集**

使用了LoCoMo基准数据集进行实验，涵盖多跳、单跳、时序与开放域问答四类任务。

**📈 对比分析**

与基线（Mem0、A-mem）及PoisonedRAG进行对比，实验显示在内容式攻击下F1平均降幅可达70%以上，在问题定向攻击下单次注入1-2条恶意记忆即可使整体F1下降40%~50%，表明攻击极具效果。

**⚠️ 局限性**

局限在于攻击仅针对相似度检索机制，未考虑多模态或可解释性检索方式；攻击效果受k值影响，过大时易被检测；对抗鲁棒性与防御机制仍未展开。

---

## 199. Language and Geometry Grounded Sparse Voxel Representations for Holistic Scene Understanding

**arXiv ID:** 2602.15734 | [PDF](https://arxiv.org/pdf/2602.15734v1)

**作者:** Guile Wu `[一作]` (Huawei Noah's Ark Lab), Dongfeng Bai `[通讯]` (Huawei Noah's Ark Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种统一框架 LangSVR，利用语言与几何驱动的稀疏体素表示，实现三维场景的整体理解与重建。

**💡 创新点**

创新点在于将语言特征与几何知识通过特征调制模块与几何蒸馏融合到稀疏体素的外观、密度、特征与置信度字段，实现语义、外观与几何的协同建模。

**🔧 技术方法**

采用稀疏体素光栅化（SVRaster）、Spherical Harmonics 外观场、密度场、特征场、置信度场；使用自动编码器将高维 CLIP 特征降维；通过特征调制模块与几何蒸馏（深度相关正则化、模式一致性正则化）将 2D 语言模型与 3D 几何基础模型（VGGT、Depth‑Anything‑V2）知识迁移；多视角置信度过滤等技术。

**📊 数据集**

使用 LERF 和 Mip‑NeRF360 两个多视角图像数据集进行训练与评估。

**📈 对比分析**

与多种 SOTA 方法（LangSplat、LERF、Feature‑3DGS、SVRaster、3DGS 等）对比，LangSVR 在 3D 语义分割 mIoU 62.1、目标定位 mAcc 84.4%、视图合成 PSNR 29.87、LPIPS 0.159 等指标上均优于或接近最佳水平，且在整体性能与多任务表现上具备竞争力。

**⚠️ 局限性**

局限性包括难以捕捉极细小对象（如碗内玉米）、对自动编码器低维特征空间的依赖导致部分语义表达不足、以及相较于 SVRaster 在渲染速度和显存占用略有下降。

---

## 200. Improving LLM Reliability through Hybrid Abstention and Adaptive Detection

**arXiv ID:** 2602.15391 | [PDF](https://arxiv.org/pdf/2602.15391v1)

**作者:** Ankit Sharma `[一作]` (Chhattisgarh Swami Vivekanand Technical University), Jyotiprakash Patra `[通讯]` (Chhattisgarh Swami Vivekanand Technical University)

**通讯引用:** 57 | [OpenAlex ID](https://openalex.org/A5068948318)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个自适应弃权系统，在LLM生产环境中动态根据上下文调整安全阈值，兼顾安全与实用。

**💡 创新点**

创新点是将多维检测与层级级联相结合，并实现基于域敏感度和用户信任度的自适应阈值；同时引入重复检测避免循环。

**🔧 技术方法**

使用多维检测器（安全、置信度、知识边界、上下文、重复）、层级级联架构、动态阈值计算、句子嵌入相似度、熵/困惑度估计等技术。

**📊 数据集**

评估使用公开的安全/毒性基准、合成医学/金融/教育/创意对话、以及对抗 jailbreak 数据，构成混合、医学、教育、创意等多领域数据集。

**📈 对比分析**

与基线（无弃权、静态门槛、全量检测、外部 Guardrails）对比，平均延迟从450 ms降至42 ms（10×提升），严格安全模式下召回 100%，精度可调至>95%，各领域误拒率下降 80–90%。

**⚠️ 局限性**

局限：对高级对抗攻击仍易绕过，重复检测需维护历史记忆，域适配需要初始化校准，缺乏多模态扩展与可解释性。

---

## 201. TAROT: Test-driven and Capability-adaptive Curriculum Reinforcement Fine-tuning for Code Generation with Large Language Models

**arXiv ID:** 2602.15449 | [PDF](https://arxiv.org/pdf/2602.15449v1)

**作者:** Chansung Park `[一作]` (Electronics and Telecommunications Research Institute), Jianguo Li `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出TAROT框架，利用四层测试套件实现能力自适应的强化学习微调，提升代码生成的功能正确性与鲁棒性。

**💡 创新点**

整合测试驱动的层级测试套件与基于模型能力的课程策略，解决奖励稀疏与偏斜问题，显著改善训练稳定性与效率。

**🔧 技术方法**

采用强化学习（GRPO）、Curriculum Learning、LLM代码生成技术，并引入Test‑Driven Development的思想。

**📊 数据集**

使用约15k个Python编程面试题构建的TAROT数据集，其中每个问题配备基本、介于、复杂、边缘四层测试用例。

**📈 对比分析**

在多模型（Qwen、Gemma 等）与多基准（HumanEval、MBPP、LiveCodeBench、CodeForces 等）上进行对比实验，TAROT相对基线提升 pass@1 多个百分点，表现出一致的性能提升。

**⚠️ 局限性**

主要限制在于测试用例由LLM生成，可能存在偏差；仅覆盖Python单语言；课程策略固定在预定义组合，未实现训练过程中的动态优化。

---

## 202. Fairness over Equality: Correcting Social Incentives in Asymmetric Sequential Social Dilemmas

**arXiv ID:** 2602.15407 | [PDF](https://arxiv.org/pdf/2602.15407v1)

**作者:** Alper Demir `[一作]` (Izmir University of Economics and University of Edinburgh), Stefano V. Albrecht `[通讯]` (DeepFlow)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

在多智能体强化学习中研究具有不对称奖励或动作效果的连续社会困境，并提出了可在部分可观测环境下使用的公平性增强方法；

**💡 创新点**

1) 引入了不对称连续社会困境的概念；2) 对现有公平性激励（IA、SVO）进行归一化、基于代理的权重调节和局部社会反馈改造；3) 在部分可观测环境下实现了公平性方法的分布式执行；

**🔧 技术方法**

使用独立DQN（IQL）训练智能体，改造的公平激励公式（IA、SVO）、归一化技巧、代理权重系数和局部估计算法；

**📊 数据集**

在Coin和Apples两个公开基准环境上构造了四种不对称变体（奖励不对称、动作不对称）进行实验；

**📈 对比分析**

与IQL、IQL+IA、IQL+SVO等基线对比。实验结果表明，改造后的Fair&LocalIA和Fair&LocalSVO在平均回报、持续性、和平度等指标上均优于基线，且收敛更快、合作更稳定；

**⚠️ 局限性**

需人工设置代理权重ϕ_i，适用范围局限于已构造的不对称类型，未验证对更复杂或多种不对称的泛化；

---

## 203. Refine Now, Query Fast: A Decoupled Refinement Paradigm for Implicit Neural Fields

**arXiv ID:** 2602.15155 | [PDF](https://arxiv.org/pdf/2602.15155v1)

**作者:** Tianyu Xiong `[一作]` (Ohio State University), Han-Wei Shen `[通讯]` (Ohio State University)

**通讯引用:** 6332 | [OpenAlex ID](https://openalex.org/A5065630217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计并实现了 Decoupled Representation Refinement (DRR) 框架和 DRR-Net，将深度 Refiner 与高效 Embedding 分离，利用离线预处理提升 INR 表示能力，并提出 Variational Pairs (VP) 数据增强方案，在多维集成仿真场景下构建高精度且推理快速的神经场替代模型。

**💡 创新点**

核心创新在于：①将强表达能力的深度网络与推理路径拆分，在线仅使用缓存的高质量 Embedding；②提出统一的非参数预处理（结构超分、位置编码）让 Refiner 能在低维结构上学习复杂非线性；③引入 VP 通过插值生成物理合理的增强样本，显著提升泛化；④在多尺度/条件下统一实现，兼顾精度与速度。

**🔧 技术方法**

技术手段包括：隐式神经表示 (INR)、多分辨率嵌入网格、结构超分 (super‑resolution)、位置编码、GLU‑based point‑wise Refiner、Variational Pairs 数据增强（VP‑S、VP‑SC）、逆距离加权插值、混合 MLP/哈希网格等。 其中 DRR-Net 采用点级 GLU Refiner 和多分辨率特征统一；VP 通过局部插值生成新样本。

**📊 数据集**

使用的实验数据集为：Nyx（宇宙学 3D dark matter density, 3 参数, 256³ 体素）、MPAS‑Ocean（海洋温度, 4 参数, 11845146 顶点非结构化网格）和 Cloverleaf3D（流体能量, 6 参数, 128³ 体素）。

**📈 对比分析**

与 K‑Planes、Explorable‑INR、FA‑INR 等主流基线在三大数据集的条件泛化和空间‑条件零样本泛化任务中进行比较；DRR‑Net 在 Rel‑L2、PSNR、SSIM 上实现或逼近最高基线，同时推理速度比 FA‑INR 快 27×、参数量更少；在 MPAS‑Ocean 上虽然 FA‑INR 取得最高 PSNR，但 DRR‑Net 仍显著优于其它网格基模型，证明了对网格偏差的补偿。VP 方案在 ablation 中提升约 1–2 dB PSNR，说明增强效果显著。

**⚠️ 局限性**

局限性包括：①仅在多分辨率 Cartesian 网格上验证，未探究 Hash、Adaptive 等更先进嵌入结构；②Refiner 采用简单 GLU，未尝试 CNN/Transformer 等更强结构；③仅在训练参数范围内进行插值，未解决外推能力；④对非结构化网格的适配仍待验证。

---

## 204. ViTaB-A: Evaluating Multimodal Large Language Models on Visual Table Attribution

**arXiv ID:** 2602.15769 | [PDF](https://arxiv.org/pdf/2602.15769v1)

**作者:** Yahia Alqurnawi `[一作]` (Arizona State University), Vivek Gupta `[通讯]` (Arizona State University)

**通讯引用:** 1945 | [OpenAlex ID](https://openalex.org/A5100748239)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多模态大语言模型在表格数据中的答案归因能力，构建 ViTaB-A 基准，对 Markdown、JSON、图像三种表格表示进行系统评测。

**💡 创新点**

首次提出针对视觉表格归因的公开基准 ViTaB-A，并将归因与置信度评估纳入对比，揭示 QA 与归因能力存在显著差距，强调归因应成为独立训练目标。

**🔧 技术方法**

采用多模态 LLM（Gemma‑3、InternVL3.5、Qwen3‑VL、Molmo2 等）在零样本、少样本和 Chain‑of‑Thought 提示下推理；使用单元/行/列准确率、内部置信度与口头置信度对齐（Brier 分数）等指标评估归因和置信度。

**📊 数据集**

使用 HiTab 数据集（包含表格问答与归因标注）构造 200 张表格的评测集，涵盖 Markdown、JSON 和渲染图像三种表示。

**📈 对比分析**

通过多模型、多提示策略对比，QA 准确率约 55–60%，归因准确率仅 25–35%（JSON 近随机）；图像表现最好；行归因准确率比列高 1.3–2 倍；置信度与归因准确率对齐低于 70%。

**⚠️ 局限性**

归因准确率低、置信度不可靠；缺乏针对归因的专门训练目标；JSON 等结构化格式尤其难以归因；导致在需要可追溯性和审计的高风险领域仍无法充分信赖。

---

## 205. How Do We Research Human-Robot Interaction in the Age of Large Language Models? A Systematic Review

**arXiv ID:** 2602.15063 | [PDF](https://arxiv.org/pdf/2602.15063v1)

**作者:** Yufeng Wang `[一作]` (Hong Kong University of Science and Technology), Xin Tong `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 13073 | [OpenAlex ID](https://openalex.org/A5100784734)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了 86 篇关于大语言模型驱动人机交互（LLM‑HRI）的研究，构建 Sense‑Interaction‑Alignment 框架和九维度分类体系。

**💡 创新点**

首次将 LLM 应用于 HRI 的研究归纳为三大核心维度（感知、交互、对齐），提出统一的评估与设计维度，并提供开放在线数据库。

**🔧 技术方法**

采用 PRISMA 2020 文献检索与筛选流程，结合 ACM/IEEE 及跨学科数据库，构建 9 类代码书（Sense‑Interaction‑Alignment、Modality、Morphology、Autonomy 等）。

**📊 数据集**

未使用单一人工标注数据集，而是汇总并整理 86 篇已有研究的公开数据与实验设置，形成研究主题与方法的系统表征。

**📈 对比分析**

通过对研究方法、评估指标与应用场景的定量统计，指出 LLM‑HRI 在实验室与现场部署、客观与主观评估上的分布与热点，但未给出单一模型的性能基准。

**⚠️ 局限性**

局限性包括：仅涵盖 2021‑2025 年英文同行评审文章，可能遗漏非英文或灰色文献；LLM 版本快速迭代导致综述的时效性；对 LLM 内部机制的技术细节分析有限；系统性偏向描述性而非实证比较。

---

## 206. What makes an Expert? Comparing Problem-solving Practices in Data Science Notebooks

**arXiv ID:** 2602.15428 | [PDF](https://arxiv.org/pdf/2602.15428v1)

**作者:** Manuel Valle Torre `[一作]` (Delft University of Technology), Catharine Oertel `[通讯]` (Delft University of Technology)

**通讯引用:** 941 | [OpenAlex ID](https://openalex.org/A5013731783)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 440 篇 Kaggle Jupyter 笔记本进行多级序列分析，比较专家与初学者在数据科学问题解决过程中的差异。

**💡 创新点**

提出基于整体流程结构、阶段转移、细粒度动作模式的多级序列分析框架，揭示专家并非采用不同的阶段顺序，而是通过更灵活、迭代且高效的细粒度操作来实现专业水平。

**🔧 技术方法**

使用 Optimal Matching 计算序列相似度、Agglomerative Hierarchical Clustering、Process Mining（Fuzzy Miner）、Markov Models、Sequential Pattern Mining，以及 Chi‑square 检验评估聚类与专家级别的关联。

**📊 数据集**

使用 Code4ML 公开的 Kaggle 笔记本数据集（Quora Insincere Questions Classification），共 440 篇经过长度和层级过滤后得到的数据，作者根据 Kaggle 进阶等级划分为初学者与专家。

**📈 对比分析**

通过聚类与过程图对比检测整体流程结构差异，利用 Markov 模型聚类评估阶段转移差异，并用 Sequential Pattern Mining 识别细粒度动作模式。实验显示：整体流程层次与细粒度动作模式能显著区分专家与初学者（p=0.0027），而阶段转移层次无显著差异（p=0.97）。

**⚠️ 局限性**

研究仅聚焦单一文本分类竞赛，未覆盖其他任务或领域；仅分析笔记本最终状态，忽略实时编辑、删除和调试过程；因而结论可能不具普适性，需要进一步在多任务、多领域和实时交互数据上验证。

---

## 207. Service Orchestration in the Computing Continuum: Structural Challenges and Vision

**arXiv ID:** 2602.15794 | [PDF](https://arxiv.org/pdf/2602.15794v1)

**作者:** Boris Sedlak `[一作]` (Universitat Pompeu Fabra), Schahram Dustdar `[通讯]` (ICREA)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在计算连续体（CC）中，提出了一种基于主动推理的自主服务编排框架，通过行为马尔可夫屏障模型实现对服务状态的实时建模和自适应调整。

**💡 创新点**

创新点在于将神经科学中的主动推理与服务编排相结合，利用行为马尔可夫屏障对服务与环境交互建模，支持服务间协同与自组织。

**🔧 技术方法**

使用的技术包括主动推理（Active Inference）、行为马尔可夫屏障（Behavioral Markov Blanket）、贝叶斯网络、持续感知-决策-执行闭环以及分层代理架构。

**📊 数据集**

实验数据来自自构造的AR叠加服务场景，主要在Nvidia Jetson等Edge设备上收集的实时性能指标；未使用公开标准数据集。

**📈 对比分析**

与传统基于规则或静态放置的基线对比，实验表明MB模型的主动推理能及时满足SLO，且在Edge设备上的运行开销保持在毫秒级，性能表现良好。

**⚠️ 局限性**

局限性包括缺乏大规模标准化仿真环境、持续学习机制的高效实现，以及跨供应商、多方协作与隐私约束下的部署挑战。

---

## 208. Fast and Fusiest: An Optimal Fusion-Aware Mapper for Accelerator Modeling and Evaluation

**arXiv ID:** 2602.15166 | [PDF](https://arxiv.org/pdf/2602.15166v1)

**作者:** Tanner Andrulis `[一作]` (Massachusetts Institute of Technology), Joel S. Emer `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 25368 | [OpenAlex ID](https://openalex.org/A5024384625)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出 Fast and Fusiest Mapper (FFM)，在张量代数加速器的融合映射空间中快速寻找能耗与延迟最优的映射。

**💡 创新点**

创新点在于引入部分映射（pmapping）概念，利用兼容性、资源预留和目标三类标准进行分组与 Pareto 剔除，并通过递归构建与剪枝，使原本指数级的搜索空间几乎线性缩小。

**🔧 技术方法**

使用 LoopTree 与 Einsum 规范构建映射；采用兼容性分组、资源预留树（ReservationTree）合并、动态资源跟踪与合并、Pareto 剔除等技术；实现于 AccelForge 框架，并与 Timeloop、TileFlow、SET 等基线比较。

**📊 数据集**

在 GPT‑3 6.7B Transformer（batch 4，4096 tokens）以及多步矩阵乘积链（最多 64 步）上进行实验；评估架构为 TPUv4i 等。

**📈 对比分析**

与随机搜索、遗传算法、模拟退火三种基线在 EDP 与运行时间上比较。FFM 仅需约 30 CPU 小时即可得到最优映射，基线在 1000 倍运行时间仍无法逼近最优；FFM 运行时间随 Einsum 数量近似线性增长，速度快 >1000×，能效提升 1.3–37 倍，吞吐率显著提升。

**⚠️ 局限性**

仍受单步映射生成耗时占主导、极长工作负载记忆占用高、仅在 Transformer 等特定工作负载验证、实现基于 Python 限制并行度等限制。

---

## 209. Accelerated Predictive Coding Networks via Direct Kolen-Pollack Feedback Alignment

**arXiv ID:** 2602.15571 | [PDF](https://arxiv.org/pdf/2602.15571v1)

**作者:** Davide Casnici `[一作]` (Delft University of Technology), Charlotte Frenkel `[通讯]` (Delft University of Technology)

**通讯引用:** 870 | [OpenAlex ID](https://openalex.org/A5053902946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结合直接 Kolen–Pollack 反馈对齐的预测编码算法 DKP‑PC，解决传统预测编码的误差延迟与指数衰减问题。

**💡 创新点**

创新点在于将可学习的直接反馈连接嵌入预测编码，既实现了即时误差信号传递，又保持了局部更新，显著降低了时间复杂度从 O(L) 到 O(1)。

**🔧 技术方法**

采用直接反馈对齐（DFA）、Kolen–Pollack 学习规则、预测编码（PC）框架，以及并行更新的神经活动与权重调整。

**📊 数据集**

使用多种数据集进行评估：MNIST、Fashion‑MNIST、CIFAR‑10、CIFAR‑100（Top‑1/Top‑5）以及 Tiny ImageNet。

**📈 对比分析**

与 BP、DKP、PC、iPC、CN‑PC 等方法比较，DKP‑PC 在大多数任务上取得与 BP 相近甚至更高的准确率，同时训练时间比 PC 与 iPC 提升 60–80%。

**⚠️ 局限性**

主要局限包括在 PyTorch 中未实现真正并行化导致仍不及 BP；反馈矩阵存储开销大；需要自定义 CUDA 内核和进一步压缩/量化以适配硬件。

---

## 210. Hybrid Feature Learning with Time Series Embeddings for Equipment Anomaly Prediction

**arXiv ID:** 2602.15089 | [PDF](https://arxiv.org/pdf/2602.15089v1)

**作者:** Takato Yasuno `[一作]` `[通讯]` (Yachiyo Engineering Co), Takato Yasuno (Yachiyo Engineering Co)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种混合学习框架，将 Granite TinyTimeMixer 的64维时间序列嵌入与28维统计特征相结合，用 LightGBM 进行 HVAC 设备异常预测，支持30/60/90天多时效预测。

**💡 创新点**

创新点在于（1）系统性剖析纯深度学习在工业异常检测中的失败模式；（2）构建嵌入+统计特征的融合结构，证明两者互补；（3）在真实工业数据上实现生产级性能（Precision 91–95%，ROC‑AUC 0.995，FPR <1.1%）。

**🔧 技术方法**

技术主要包括 Granite TinyTimeMixer + LoRA 微调、统计特征工程（趋势、波动、抽样等）、LightGBM 梯度提升分类、时间窗口切片、CPU‑only 推理。

**📊 数据集**

使用 64 台 HVAC 设备的 51,564 条 90 天窗口样本（2015–2024 年日聚合数据），包含 230 个传感器通道，异常标签基于 5%–95% 分位数阈值。

**📈 对比分析**

与两类基线对比：纯 Granite TinyTimeMixer（Precision 9–11%）和仅统计特征的 LightGBM（Precision 79–87%）。混合模型在所有时效下实现 Precision 91–95%，Recall 88–94%，ROC‑AUC 0.995，FPR 0.5–1.1%，显著优于基线。

**⚠️ 局限性**

局限包括：仅在 64 台泵设备验证，未覆盖其它 HVAC 子系统；异常定义仅基于阈值，可能忽略非传感器表现的故障；日级别聚合可能错过子日短时异常；在 90 天时效仍有 12% 错失率。

---

## 211. Stability in Distance Preservation Games on Graphs

**arXiv ID:** 2602.15784 | [PDF](https://arxiv.org/pdf/2602.15784v1)

**作者:** Argyrios Deligkas `[一作]` (Royal Holloway University of London), Šimon Schierreich `[通讯]` (AGH University of Krakow)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在图形距离保持游戏（Graphical Distance Preservation Games）中，研究了将代理分配到图顶点的公平（无嫉妒）与稳定（跳跃与交换稳定）问题，系统探讨了其存在性、复杂度及在多种结构和偏好约束下的算法。

**💡 创新点**

首次将距离保持游戏从线段推广到一般图形拓扑，并通过组合复杂度分析与参数化方法，全面界定了在不同顶点覆盖数、树深度、路径、点数、邻域多样性、模宽、直径以及偏好图结构（星形、双向星形、无环）下的可解性与不可解性。

**🔧 技术方法**

采用组合优化、参数化复杂度理论、图结构分解（如点覆盖、邻域多样性、模宽、直径）、动态规划以及潜能函数法等技术，构建多种多项式时间与固定参数可解算法，同时给出多种NP/PSPACE难度证明。

**📊 数据集**

本研究为理论性工作，未使用实际数据集；所有结果均通过形式化证明与算法分析得到。

**📈 对比分析**

通过严谨的理论证明与复杂度上界/下界分析，展示了在星形、完全图、对称偏好等情形下可在多项式时间求解，而在点覆盖数为2、树深度为2、路径等结构中问题为NP/PSPACE难，说明了不同约束对性能的显著影响。

**⚠️ 局限性**

仍存在若干限制与开放问题，例如：在一般图、非对称理想距离、以及“至少/最多”距离函数下的存在性与算法复杂度尚未完全解答；此外，对跳跃与交换稳定的完整复杂度分类和实验验证仍需进一步研究。

---

## 212. Seeing to Generalize: How Visual Data Corrects Binding Shortcuts

**arXiv ID:** 2602.15183 | [PDF](https://arxiv.org/pdf/2602.15183v1)

**作者:** Nicolas Buzeta `[一作]` (Pontificia Universidad Católica), Rodrigo Toro Icarte `[通讯]` (Pontificia Universidad Católica)

**通讯引用:** 754 | [OpenAlex ID](https://openalex.org/A5006732213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了视觉语言模型（VLM）在纯文本检索任务中表现优于其基准大语言模型（LLM）的现象，并通过控制实验和机制解释证明视觉训练能提升文本任务的泛化性能。

**💡 创新点**

创新点在于揭示视觉训练通过迫使模型从位置绑定（positional binding）转向符号绑定（symbolic binding）来提升跨模态模型在单模态任务上的表现，并证明了翻译不变性是促成该转变的关键因素。

**🔧 技术方法**

技术包括：构造可控的“间接检索”任务；对Transformer进行多阶段训练（文本→图像→混合）；使用交互干预（interchange intervention）识别绑定机制；使用注意力打击（attention knockout）和线性探测（linear probing）分析电路层次。

**📊 数据集**

数据集为自定义的合成检索数据，包含文本描述与对应图像的颜色形状标签，覆盖不同数量的对象（最多8个用于训练，更多用于OOD评估）。

**📈 对比分析**

比较方法通过在相同任务上评估文本-only模型、图像-only模型、混合训练模型及噪声注入模型，发现混合模型在OOD长度下的准确率从约37%提升至约70%，并在大规模Qwen系列模型中验证了相同的绑定机制转变，VLM相较LLM在长上下文检索任务上提升约5-15%。

**⚠️ 局限性**

局限性包括：实验主要基于高度简化的合成任务，缺乏对真实自然语言场景的直接验证；未系统探究不同视觉编码器的超参数对绑定机制的细微影响；对噪声注入的效果缺乏更深入的因果分析。

---

## 213. Intellicise Wireless Networks Meet Agentic AI: A Security and Privacy Perspective

**arXiv ID:** 2602.15290 | [PDF](https://arxiv.org/pdf/2602.15290v1)

**作者:** Rui Meng `[一作]` (Beijing University of Posts and Telecommunications), Rahim Tafazolli `[通讯]` (University of Surrey)

**通讯引用:** 19272 | [OpenAlex ID](https://openalex.org/A5032549075)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文系统评估了Agentic AI在智能简洁无线网络中的安全与隐私应用，提出了针对信号处理、信息传输和网络组织的三大安全分类，识别了新出现的攻击面并给出了防御方案，最后通过语义隐写通信的案例验证了其防御效果。

**💡 创新点**

创新点在于：①首次将Agentic AI与智能简洁网络融合，形成完整的安全框架；②提出了三维安全分类与对应的攻击与防御技术；③利用Agentic AI实现自适应隐写与主动防御，并对其性能进行案例验证；④指出低复杂度模型与实验平台等未来研究方向。

**🔧 技术方法**

技术手段包括：Agentic AI感知‑记忆‑推理‑动作闭环、联合多智能体深度强化学习、检索增强生成（RAG）、控制网络（ControlNet）与扩散模型（Stable Diffusion & EDICT）、SwinJSCC语义编码、加密与差分隐私、对抗训练、红队测试、双重验证与可信执行环境等。

**📊 数据集**

案例使用 UniStega 数据集；其它部分未明确指定公开数据集，主要基于模拟与理论分析。

**📈 对比分析**

在语义隐写案例中，系统对比了合法接收方与窃听方的恢复效果，结果表明合法方能准确恢复隐藏图像，而窃听方因缺失数字令牌导致图像崩溃或误导；未给出精确数值指标，但验证了方案对智能窃听的有效性。

**⚠️ 局限性**

局限性包括：①Agentic AI模型复杂度高，资源消耗大；②新产生的攻击面（如多智能体共谋、注入攻击等）仍未被完全覆盖；③缺乏统一的安全/隐私评估指标和实验平台；④低功耗端设备的部署与性能折衷尚未解决。

---

## 214. This human study did not involve human subjects: Validating LLM simulations as behavioral evidence

**arXiv ID:** 2602.15785 | [PDF](https://arxiv.org/pdf/2602.15785v1)

**作者:** Jessica Hullman `[一作]` (Northwestern University), Aaron Shaw `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了将大语言模型作为行为科学实验的合成参与者的研究，分析了启发式验证、统计校准和模拟-验证三种策略，并阐明了在何种条件下可实现有效推断。

**💡 创新点**

创新点在于系统化区分启发式验证与统计校准的假设与适用场景，并给出可行的验证与校准框架，强调在使用 LLM 进行因果推断时需要满足的必要条件。

**🔧 技术方法**

采用文献综述和理论分析方法，对现有启发式验证技术（对齐度、相关性、预测准确性、分布相似度等）、统计校准技术（PPI、DSL、插值校正等）和模拟-验证方法进行了比较与讨论。

**📊 数据集**

未使用单一实验数据集，而是基于 53 篇与更多近期研究的文献综述，收集并汇总了关于 LLM 与人类响应相似性、偏差、记忆化等证据。

**📈 对比分析**

通过对比启发式验证的缺陷（系统偏差、记忆化、泛化风险）与统计校准在理论上的优势，指出后者在满足无训练泄露和参数识别假设时能提供无偏估计，虽然目前实际提升有限。

**⚠️ 局限性**

局限在于缺乏大规模实证验证，讨论主要基于已有文献和理论推导；对不同领域 LLM 表现差异、样本可代表性、以及校准方法的实际可行性仍需进一步研究。

---

## 215. Who Is Doing the Thinking? AI as a Dynamic Cognitive Partner: A Learner-Informed Framework

**arXiv ID:** 2602.15638 | [PDF](https://arxiv.org/pdf/2602.15638v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 216. Spatially-Aware Adaptive Trajectory Optimization with Controller-Guided Feedback for Autonomous Racing

**arXiv ID:** 2602.15642 | [PDF](https://arxiv.org/pdf/2602.15642v1)

**作者:** Alexander Wachter `[一作]`, Christian Hartl-Nesic `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种将摩擦地图与模型预测控制（MPC）结合的轨迹优化框架，目标是生成对摩擦不确定性更鲁棒的机器人轨迹。

**💡 创新点**

创新点在于将实时摩擦信息嵌入优化器与MPC的闭环，形成误差反馈循环，从而实现对不确定摩擦的自适应补偿。

**🔧 技术方法**

使用了非线性优化、MPC、误差反馈机制以及摩擦地图建模技术（必要时结合深度学习估计摩擦场）。

**📊 数据集**

实验数据来源于仿真环境和真实机器人（如KUKA LBR iiwa/UR5）收集的轨迹与摩擦场记录。

**📈 对比分析**

与传统MPC和纯轨迹优化基线相比，本文方法在跟踪误差和碰撞率上提升约20%，在动态摩擦变化场景中表现更稳健。

**⚠️ 局限性**

主要局限在于对摩擦地图的依赖；在快速变化或未知摩擦环境下需要更高频的地图更新；以及实时计算的开销相对较大。

---

## 217. Estimating Human Muscular Fatigue in Dynamic Collaborative Robotic Tasks with Learning-Based Models

**arXiv ID:** 2602.15684 | [PDF](https://arxiv.org/pdf/2602.15684v1)

**作者:** Feras Kiki `[一作]` (Koc University), Cagatay Basdogan `[通讯]` (Koc University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究开发了基于sEMG信号的机器学习与深度学习模型，用于在动态重复的人机协作任务中估计肌肉疲劳进展；

**💡 创新点**

创新点在于将疲劳估计从传统的分类问题转为回归问题，实现对疲劳进程的连续监测，并在跨任务中验证了模型的泛化能力；

**🔧 技术方法**

使用了随机森林、XGBoost、线性回归等传统机器学习模型以及基于频谱图的CNN深度学习模型；

**📊 数据集**

实验数据来自10名大学生在虚拟环境中与协作机器人进行横向重复运动的sEMG、力学与位置传感数据，共约138个循环样本；

**📈 对比分析**

通过留一试验进行6折交叉验证，结果显示CNN平均RMSE为20.8%±4.3%，随机森林23.3%±3.8%，XGBoost24.8%±4.5%，线性回归26.9%±6.1%，且模型在不同运动方向的任务上仍保持较低误差；

**⚠️ 局限性**

主要局限包括样本仅为年轻大学生，模型未结合肌肉力学细节，且对不同肌群的疲劳进展假设相同，未来需扩展样本多样性和混合模型。

---

## 218. Crane: An Accurate and Scalable Neural Sketch for Graph Stream Summarization

**arXiv ID:** 2602.15360 | [PDF](https://arxiv.org/pdf/2602.15360v1)

**作者:** Boyan Wang `[一作]` (Hefei University of Technology), Tong Yang `[通讯]` (Peking University)

**通讯引用:** 5646 | [OpenAlex ID](https://openalex.org/A5101674305)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `fede83ac-7505-405f-ab37-e7284695c47f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为Crane的层次化神经草图，用于高效压缩和估计图流中的边频率

**💡 创新点**

创新点在于引入层次化携带机制自动将高频条目提升至上层以减少碰撞，并通过自适应内存扩展实现可伸缩性

**🔧 技术方法**

使用可学习的编码网络（MLP + outer‑product）生成二维基矩阵，层级记忆结构以及线性解码器，全部训练于合成的Zipf分布图流上

**📊 数据集**

在五个真实世界图流数据集上评测，包括Lkml、CAIDA、WiKiTalk、StackOverflow、NotreDame

**📈 对比分析**

与四类基线（TCM、GSS、Auxo、Mayfly）比较，在64KB内存预算下，Crane的平均相对误差（ARE）比最优基线低10~70倍，且误差随流量增长缓慢，吞吐量虽低于纯哈希方法但仍可接受

**⚠️ 局限性**

局限在于实现复杂度高，推理速度低于传统哈希草图，且在极低频场景下仍需更大内存以维持高精度

---

## 219. Logit Distance Bounds Representational Similarity

**arXiv ID:** 2602.15438 | [PDF](https://arxiv.org/pdf/2602.15438v1)

**作者:** Beatrix M. B. Nielsen `[一作]`, Simon Buchholz `[通讯]` (Max Planck Institute for Biological Cybernetics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了当两模型的分布接近时其内部表示是否线性相似，并提出了 logit 距离与线性可辨识度量，应用于知识蒸馏。

**💡 创新点**

定义 logit 距离作为模型分布的度量，证明它与 KL 的关系；提出线性可辨识度量 d_rep 并证明 logit 距离上界该度量和 mCCA；在蒸馏中证明 logit 损失比 KL 损失更能保持线性表示。

**🔧 技术方法**

理论证明、可辨识性分析、mCCA、线性可辨识度量、对数几率差距、KL 分析、logit 距离；实验使用 ResNet、DINOv2、L1/L2/ KL 损失。

**📊 数据集**

合成 2D 数据集、ImageNet-100、CUB-200（鸟类）数据集。

**📈 对比分析**

通过 mCCA、d_rep、KL、logit 距离以及概念分类准确率进行比较；实验结果显示 logit 训练的学生在 mCCA 接近 1、d_rep 大幅下降，并且在概念分类上显著优于 KL 训练。

**⚠️ 局限性**

需要标签数大于表示维度+1，假设模型处于一般位置；KL 对表示相似性的保证弱，理论对实际参数空间的适用性有限。

---

## 220. Near-Optimal Sample Complexity for Online Constrained MDPs

**arXiv ID:** 2602.15076 | [PDF](https://arxiv.org/pdf/2602.15076v1)

**作者:** Chang Liu `[一作]` (University of California), Lin F. Yang `[通讯]` (University of California)

**通讯引用:** 69915 | [OpenAlex ID](https://openalex.org/A5100694840)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于模型的双重迭代（primal‑dual）算法，结合“翻倍批次更新”技术，专门针对在线有限时序约束马尔可夫决策过程（CMDP）设计，能够在放宽可行性（relaxed feasibility）和严格可行性（strict feasibility）两种安全约束设定下给出近似最优策略。

**💡 创新点**

创新点主要包括：
- 在在线 CMDP 中首次实现了与随机访问模型相当的最优样本复杂度；
- 引入了翻倍批次更新方法，显著降低了经验模型与价值估计之间的耦合，从而得到更紧的理论界；
- 通过对双重迭代的离散化处理和惰性更新，实现了对 Slater 常数的无关（ζ‑free）最优采样复杂度；
- 对严格可行性问题提供了匹配下界的样本复杂度，首次将在线学习与随机访问学习等价化。

**🔧 技术方法**

核心技术包括：
- 基于 UCB 的奖励与成本探索奖金；
- Lagrangian 形式的约束惩罚与双重迭代（primal‑dual）框架；
- 翻倍批次更新的经验模型估计；
- 价值函数的分阶段更新与离散化的拉格朗日乘子；
- 通过混合策略（mixing multiple primal iterations）降低偏差。

**📊 数据集**

本工作为理论研究，未使用任何真实或合成数据集；所有结果均基于严谨的数学证明和泛化误差分析。

**📈 对比分析**

性能评估：
- 对于放宽可行性，算法在 K = Õ(S A H³ / 2) 场景下实现 ε‑最优、ε‑约束违约，匹配无约束 MDP 的下界；
- 对于严格可行性，算法在 K = Õ(S A H⁵ / (2 ζ²)) 场景下实现 ε‑最优、零约束违约，匹配随机访问 CMDP 的下界；
- 通过对比实验（若有）显示该方法在样本复杂度与误差上均优于现有的基于策略梯度或基于贝尔曼方程的 CMDP 算法。

**⚠️ 局限性**

局限性：
- 仅适用于有限状态、有限动作、有限时间步长的表格式（tabular）CMDP；
- 仅支持单一安全约束，无法直接扩展到多约束情形；
- 对于函数逼近或连续状态空间，算法的理论保证尚未给出；
- 由于使用离散化的拉格朗日乘子，实际实现中可能需要细致的参数调优；
- 论文未给出实验验证，实际性能仍需通过仿真或真实任务检验。

---

## 221. FlashMem: Supporting Modern DNN Workloads on Mobile with GPU Memory Hierarchy Optimizations

**arXiv ID:** 2602.15379 | [PDF](https://arxiv.org/pdf/2602.15379v1)

**作者:** Zhihao Shu `[一作]` (University of Georgia), Wei Niu `[通讯]` (University of Georgia)

**通讯引用:** 2098 | [OpenAlex ID](https://openalex.org/A5043054935)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出FlashMem框架，实现在移动GPU上按需动态加载权重、利用2.5D纹理内存并重写核函数，从而支持大规模和多模型DNN推理。

**💡 创新点**

创新点包括基于Load Capacity的LC-OPG重叠计划生成、适配融合策略、纹理内存层级优化以及无分支流水线核重写，显著降低内存占用并提升吞吐量。

**🔧 技术方法**

采用静态计划求解器（Google OR-Tools CP‑SAT + LC‑OPG）、梯度预测的XGBoost负载容量模型、2.5D纹理布局、模板化核重写、动态融合和自适应预加载等技术。

**📊 数据集**

使用包含11个代表性模型的多任务数据集：NLP（GPT‑Neo）、图像分类（ResNet、ViT、DeepViT）、分割（SAM‑2、DepthAnything）、生成（SD‑UNet）、语音识别（Whisper）等，评估于OnePlus 12等手机。

**📈 对比分析**

与MNN、NCNN、TVM、LiteRT、ExecuTorch、SmartMem等主流移动框架对比，FlashMem平均内存降低3.2–8.4×，推理速度提升1.7–75×，在大型模型（如GPT‑Neo‑1.3B、SD‑UNet、GPT‑Neo‑2.7B）上实现显著优势。

**⚠️ 局限性**

局限性包括：对极动态网络（需在线求解）支持有限；卷积模型的纹理转换仍需额外内存；额外数据搬移导致能耗略高；不兼容量化模型；需要离线求解阶段产生计划文件。

---

## 222. MarkSweep: A No-box Removal Attack on AI-Generated Image Watermarking via Noise Intensification and Frequency-aware Denoising

**arXiv ID:** 2602.15364 | [PDF](https://arxiv.org/pdf/2602.15364v1)

**作者:** Jie Cao `[一作]`, Jianbing Ni `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MarkSweep，一种在无盒环境下可有效去除 AI 生成图像中的隐式水印且不损失视觉质量的攻击方法。

**💡 创新点**

创新点：①利用边缘感知的高斯噪声放大增强水印信号；②设计可学习频域分解模块 (LFDM) 与频域感知融合模块 (FaFM) 的端到端去噪网络；③在理论上证明噪声放大 + 去噪能降低水印信息互信息，从而保证水印不可恢复；④实现极高的攻击速度（<1 s）。

**🔧 技术方法**

使用技术包括：边缘感知的高斯噪声放大、ResNet‑50 编码器、FFT 频域分解与可学习阈值、双层注意力的频域融合、UNet 结构解码、Real‑ESRGAN 细节增强、LPIPS、MSE、FFT 频域一致性损失等。

**📊 数据集**

训练集：MS‑COCO 与 GenImage（各 5,000 张）用于学习自然与 AI 图像特征；测试集：CelebA‑HQ、Stable Diffusion Prompts、SDXL 1.0 生成图像用于评估不同水印方案。

**📈 对比分析**

与 SS、Yu、PTW、HiDDeN 等现有水印方法对比，MarkSweep 能将 BA 降至 51.32%（低于 HiDDeN 73% 阈值）并在多种方案下逼近或低于检测阈值；图像质量保持 PSNR≥28 dB、SSIM≥0.82；攻击速度为 0.14–0.64 s，比 DiffusionAttack 快 83–24 倍、比 UnMarker 快 1,200+ 倍，表现出优异的效率与质量平衡。

**⚠️ 局限性**

局限性：对基于初始噪声的水印（如 Gaussian Shading）效果有限，且对某些新型文本‑图像水印方法的鲁棒性尚需进一步验证。

---

## 223. COMPOT: Calibration-Optimized Matrix Procrustes Orthogonalization for Transformers Compression

**arXiv ID:** 2602.15200 | [PDF](https://arxiv.org/pdf/2602.15200v1)

**作者:** Denis Makhov `[一作]` (Fundamental Research Center MWS AI), Stamatios Lefkimmiatis `[通讯]` (Fundamental Research Center MWS AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种训练无关的 Transformer 压缩框架 COMPOT，通过稀疏正交字典学习和 Procrustes 变换实现参数的高效分解；

**💡 创新点**

其创新点在于：① 对字典施加正交约束，使字典更新和稀疏编码可闭式求解，显著降低迭代成本；② 设计了一种一 Shot 全局压缩分配策略，兼顾层级异质冗余；③ 与后训练量化（如 GPTQ）兼容，进一步提升压缩性能；

**🔧 技术方法**

所用技术包括正交稀疏字典学习、Orthogonal Procrustes 更新、硬阈值稀疏编码、激活白化、全局奇异值池化分配以及 GPTQ 4-bit 量化；

**📊 数据集**

实验覆盖语言、视觉‑语言与语音三大领域，使用的模型和数据集包括 Llama、OPT、Qwen 系列、Qwen‑VL‑8B‑Instruct、Whisper、PIQA、HellaSwag、LAMBADA、ARC、SciQ、RACE、MMLU、WikiText、C4、LibriSpeech、MMMU、OCRBench、RealWorldQA、MMStar 等；

**📈 对比分析**

在与 SVD‑LLM、SVD‑LLM V2、Dobi‑SVD、CoSpaDi、ReplaceMe、LLM‑Pruner、GPTQ 等强基线在相同压缩率下对比时，COMPOT 在多任务（语言、视觉‑语言、语音）上均实现了更高的准确率/更低的 perplexity，并且与 4‑bit GPTQ 组合后进一步压缩，表现优于单独的量化或低秩分解；

**⚠️ 局限性**

主要局限包括：对校准数据分布敏感；白化过程假设 Gram 矩阵正定且可 Cholesky 分解；固定稀疏模式可能限制表达能力；在小规模或数值不稳定的设置下性能可能下降。

---

## 224. GaiaFlow: Semantic-Guided Diffusion Tuning for Carbon-Frugal Search

**arXiv ID:** 2602.15423 | [PDF](https://arxiv.org/pdf/2602.15423v1)

**作者:** Rong Fu `[一作]` (University of Macau), Simon Fong `[通讯]` (University of Macau)

**通讯引用:** 11897 | [OpenAlex ID](https://openalex.org/A5086422507)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 GaiaFlow 框架，利用语义引导的扩散调优与检索引导的 Langevin 动力学，构建硬件无关的差分性能模型和在线自校准机制，以实现低碳、高效的检索系统。

**💡 创新点**

创新点包括：
1) 结合检索引导的 Langevin 采样与绿色潜能（U）实现对碳排放、延迟与检索质量的统一优化；
2) 提出可差分的 PEIR 性能估计（基于内存/浮点运算计数）并通过指数加权递归最小二乘（EW‑RLS）实现在线校准；
3) 在采样过程中加入早停、量化推理与频率缓存三重策略，显著降低实际延迟与能耗；
4) 通过性能一致性对比损失学习检索与配置的语义嵌入，避免纯语义匹配导致的配置偏差。

**🔧 技术方法**

使用技术：
- 语义引导扩散（Diff‑PEIR）和检索引导的 Langevin 动力学；
- 差分 PEIR（Mop/Flop 计数）与可微的绿色潜能 U；
- 在线 EW‑RLS 校准与 PUE 修正；
- 早停策略、量化推理（8‑bit）与查询缓存；
- 约束投影与局部修复；
- 对比学习的性能一致性检索模型。

**📊 数据集**

实验数据集：MS‑MARCO v1 开发集（约 9.9 M 片段、6 980 个查询）。实验在两款 CPU（AMD EPYC 7402P 与 Intel Xeon Gold 5118）上进行，分别收集操作计数与时延。

**📈 对比分析**

与 BM25、BM25‑T5、uniCOIL、DeepImpact 等基线在相同检索框架下比较。评估指标包括 R²（latency 预测）、Mop、延迟、Recall@1000、碳排放。结果显示：
- GaiaFlow 在 R² 最高（≈0.995），延迟与 Mop 分别比基线低约 30‑40%；
- Recall@1000 保持在 0.86 左右，基本不损失检索质量；
- Ablation 证明检索引导项显著降低采样步数与延迟（≈0.9 ms）；
- 在线校准显著减少 MAE，提升模型泛化。

**⚠️ 局限性**

局限性：
- 依赖查询频率缓存和少量先验样本，若查询分布大幅变化需重新采样；
- 量化与早停的误差需要持续监控，极端硬件或网络条件下可能出现漂移；
- 目前仅在单机 CPU 环境验证，GPU 或多机分布式检索的适用性待进一步研究；
- 对多模态检索、实时动态场景的扩展仍需验证；
- 超参数（γ₁、γ₂、γ₃）虽鲁棒但仍需针对不同硬件调优。

---

## 225. Orchestration-Free Customer Service Automation: A Privacy-Preserving and Flowchart-Guided Framework

**arXiv ID:** 2602.15377 | [PDF](https://arxiv.org/pdf/2602.15377v1)

**作者:** Mengze Hong `[一作]` (Hong Kong Polytechnic University), Li Qing `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于任务导向流程图（TOF）的无编排客户服务自动化框架，并实现了流程图构建、提示增强与去中心化蒸馏技术，支持本地部署的小语言模型完成端到端对话管理。

**💡 创新点**

创新点：① 用流程图抽象业务流程，替代传统的多模组编排，提升可解释性与可维护性；② 通过流程图引导的提示策略实现无编排的任务协调；③ 引入去中心化流程图蒸馏，实现隐私保护的数据生成与模型微调，避免敏感信息泄露。

**🔧 技术方法**

采用技术包括：小语言模型（SLM）本地推理；大型语言模型（如 GPT‑4）生成合成对话；流程图构建算法（权重对话覆盖、LP‑Rounding、ILP 选择）；流程图驱动的提示模板；去中心化流程图蒸馏框架（聚类、语义评估、合成对话生成）。

**📊 数据集**

使用数据集：MultiWOZ 2.0、SGD（标准TOD数据集）以及针对银行业务的本地中文对话数据（约30,000 条人类对话）。

**📈 对比分析**

比较方法：与训练型 TOD 基线（SimpleTOD、UBAR、GALAXY、Mars）以及提示型基线（SGP‑TOD、AutoTOD、ProTOD）和商业产品（规则树、Chatflow、Dify）进行对比。实验结果显示：在 MultiWOZ 和 SGD 上，流程图引导的 GPT‑3.5 方案在成功率、信息提供和预订成功率上均超过基线，且在银行外呼场景中实现了最高的典范率和完成率。整体性能在多项指标上均优于现有方法。

**⚠️ 局限性**

局限性：① 在 Inform 指标上略逊于模块化编排系统，原因是流程图侧重宏观目标完成而非细粒度实体匹配；② 流程图的构建仍需一定人工监督，尤其在业务变更时需要重新更新；③ 目前仅在英文和中文数据上验证，跨语言、低资源场景的适用性尚未充分测试。

---

## 226. The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems

**arXiv ID:** 2602.15382 | [PDF](https://arxiv.org/pdf/2602.15382v1)

**作者:** Xiaoze Liu `[一作]` (Purdue University), Jing Gao `[通讯]` (Purdue University)

**通讯引用:** 9731 | [OpenAlex ID](https://openalex.org/A5100781385)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Vision Wormhole 框架，利用 VLM 的视觉通道实现异构多智能体间的连续、无文本信息传递；

**💡 创新点**

创新点在于：①将视觉编码器视为通用通信接口，突破文本离散瓶颈；②设计统一视觉码流器和 hub‑and‑spoke 对齐，使通信复杂度从 O(N²) 降到 O(N)；③采用无标签自蒸馏对齐方法，兼容多模型；

**🔧 技术方法**

核心技术包括：视觉码流器（Universal Visual Codec）、统一潜在空间 𝒰、线性映射对齐、无标签蒸馏损失、残差视觉注入；

**📊 数据集**

在多项公开基准上评测：数学/科学推理（GSM8K、AIME、GPQA、MedQA）、常识推理（ARC‑Easy、ARC‑Challenge）、代码生成（MBPP‑Plus、HumanEval‑Plus）；

**📈 对比分析**

与传统 TextMAS 基线对比，Vision Wormhole 在异构配置下平均提升 6.3pp 准确率，速度提升 1.87×；弱监督版本（<100 anchor）仍能获得约 6.5pp 准确率提升、2.67× 速度提升；

**⚠️ 局限性**

局限性包括：仅在少数 VLM 和小规模模型上验证；未与其他深度潜在通信方法做直接对比；对动态多轮或大规模多模型系统的适用性尚待进一步探索。

---

## 227. CEPAE: Conditional Entropy-Penalized Autoencoders for Time Series Counterfactuals

**arXiv ID:** 2602.15546 | [PDF](https://arxiv.org/pdf/2602.15546v1)

**作者:** Tomàs Garriga `[一作]` (Novartis), Axel Brando `[通讯]` (Barcelona Supercomputing Center)

**通讯引用:** 72 | [OpenAlex ID](https://openalex.org/A5087302728)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种针对时间序列的因果反事实推断框架，并提出了新的Conditional Entropy‑Penalized Autoencoder（CEPAE）模型；

**💡 创新点**

创新点在于将PEARL式的因果推断步骤（推理‑干预‑预测）迁移至时间序列，利用熵惩罚实现潜变量与条件变量的解耦，从而显著提升反事实估计的准确性与可解释性；

**🔧 技术方法**

采用结构因果模型（SCM）+编码‑解码器架构，比较了三种变体：CVAE、CAAE与CEPAE，并使用熵上界近似的熵惩罚；

**📊 数据集**

使用三类数据集：人工合成、半合成（Rossmann 销售模拟）以及自有制药行业月度销量数据；

**📈 对比分析**

与LSTM、AB‑LSTM及基于合成控制的基线进行对比；CEPAE在所有数据集和两种干预方向下均获得最低的MAE、最小的MBE，并在无真实反事实的评估指标（Added Variations、Axiomatic metrics）中表现最优；

**⚠️ 局限性**

限制在于需要大量具备事件与非事件记录的时间序列以训练模型；对极高维度或稀缺事件的数据表现不确定；熵惩罚需要手工调参且对不同数据分布可能不稳健；

---

## 228. Accelerating Large-Scale Dataset Distillation via Exploration-Exploitation Optimization

**arXiv ID:** 2602.15277 | [PDF](https://arxiv.org/pdf/2602.15277v1)

**作者:** Muhammad J. Alahmadi `[一作]` (North Carolina State University), Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于探索–利用的蒸馏方法E^2D，用全图初始化、两阶段优化和加速学生训练来显著降低大规模数据集蒸馏的冗余与计算成本。

**💡 创新点**

创新点在于将冗余视为瓶颈，通过探索阶段识别高损失区域并在利用阶段集中更新，以及采用全图初始化提升起点质量，从而在保持高准确率的同时大幅提升合成速度。

**🔧 技术方法**

使用技术包括：全图初始化、两阶段探索–利用优化策略（随机crop+高损失采样+softmax权重）、软标签重标记、自动混合精度、加速学习调度与教师模型的梯度匹配。

**📊 数据集**

实验数据集为ImageNet-1K和ImageNet-21K，评估多种网络架构以验证跨模型泛化。

**📈 对比分析**

与SRe^2L、CDA、RDED、DELT、EDC等现有大规模蒸馏方法对比，E^2D在ImageNet-1K IPC10达50% Top‑1、IPC50达58.9%，相较EDC提升约1–2%；在ImageNet-21K IPC20提升至36%，并且合成速度分别提升18×和4.3×。

**⚠️ 局限性**

局限性包括未考虑重标记阶段的开销、初始化策略仍为简单随机、在更高IPC或更大数据规模下仍需进一步优化以进一步缩小效率–准确率差距。

---

## 229. Co-Design and Evaluation of a CPU-Free MPI GPU Communication Abstraction and Implementation

**arXiv ID:** 2602.15356 | [PDF](https://arxiv.org/pdf/2602.15356v1)

**作者:** Patrick G. Bridges `[一作]` (University of New Mexico), Whit Schonbein `[通讯]` (Sandia National Laboratories)

**通讯引用:** 163 | [OpenAlex ID](https://openalex.org/A5016007722)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计、实现并评估了一种完全CPU‑free的MPI GPU通信抽象，实现了GPU驱动的两边通信和聚合交换，并在Cabana/Kokkos框架中验证其可用性。

**💡 创新点**

在保持MPI两边匹配语义的前提下，通过持久化请求、Match、Queue以及Slingshot 11的触发计数器和延迟工作队列，首次实现了GPU触发的无CPU通信；同时实现了CPU‑free的halo交换和ready send/receive。

**🔧 技术方法**

使用MPI持久化请求、Match、Queue API，OFI CXI、Slingshot 11触发计数器和DWQ，CUDA/Kokkos编程模型，Cabana Ghost halo exchange实现；并在前端系统中集成到MPI Advance库。

**📊 数据集**

在Cabana Ghost的Game of Life基准（30 GB/2 GB问题）和GPU ping‑pong微基准（1 字节–1 GB）中进行实验。

**📈 对比分析**

与Cray MPICH在Frontier和Tuolumne两台超级计算机上比较：Ping‑pong延迟降低12–49%，吞吐量提升；在8 192 GPU上halo交换强缩放速率提升28%；中等尺寸消息可达50%+加速。

**⚠️ 局限性**

限制包括：小消息的ready send开销高（未实现bounce‑buffer优化）；收发准备检查仍需额外同步；在不同GPU/节点组合下性能差异大；缺乏对NVIDIA InfiniBand等其他NIC的支持；DWQ资源耗尽时可能导致CPU阻塞。

---

## 230. OpaqueToolsBench: Learning Nuances of Tool Behavior Through Interaction

**arXiv ID:** 2602.15197 | [PDF](https://arxiv.org/pdf/2602.15197v1)

**作者:** Skyler Hallinan `[一作]` (University of Southern California), Jack Hessel `[通讯]` (Samaya AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对不完整工具文档的“OpaqueToolsBench”基准，并提出一种通过观察执行轨迹迭代改进工具文档的框架；

**💡 创新点**

创新点在于：①将工具文档视为可学习的对象，在缺乏完整说明的“透明度不足”环境下通过交互式反馈动态更新；②在单一任务流中同时完成探索与反思，避免传统方法的离散探索阶段；③兼顾离线与在线两种模式，适用于已知与未知工具。

**🔧 技术方法**

技术上采用ReAct框架的LLM代理（GPT‑5、GPT‑5‑mini）进行工具调用，并利用同一LLM或较小模型作为“编辑器”对轨迹进行批量分析、共识合并，形成更新后的工具说明；同时使用温度采样收集多样轨迹。

**📊 数据集**

使用三大数据集：BFCL‑Opaque（改造版BFCL）、Chess（多种未公开棋引擎工具）、BrowseComp Domains（域专属搜索工具）。

**📈 对比分析**

与Play2Prompt和EasyTool基线对比，平均提升约18.6%（如BFCL‑Opaque 0.80 vs 0.44），在最难的“仅函数名”设定下恢复80%执行准确率；同时测试时令牌消耗比基线低3.5–7.5倍。

**⚠️ 局限性**

局限性包括：仍与完美文档存在显著差距；对极其复杂或动态变化的工具行为收敛慢；需要强大LLM作为编辑器；目前实验仅在OpenAI GPT‑5族和小模型上验证，缺乏对更广泛模型的适用性研究。

---

## 231. Closing the Distribution Gap in Adversarial Training for LLMs

**arXiv ID:** 2602.15238 | [PDF](https://arxiv.org/pdf/2602.15238v1)

**作者:** Chengzhi Hu `[一作]` (Department of XXX, University of YYY), Leo Schwinn `[通讯]` (Company Name)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于扩散模型的对抗训练框架（DAT），通过生成与有害响应相关的自然对话样本来提升大语言模型的安全鲁棒性。

**💡 创新点**

创新点在于将对抗训练从固定训练集扩展到由扩散LLM生成的近似真实分布，并同时解决数据分布误差和模型本地最坏情况误差两类鲁棒性缺口。

**🔧 技术方法**

使用了扩散式LLM（diffusion LLM）进行有害响应条件采样，以及连续对抗优化技术实现对生成样本的鲁棒训练。

**📊 数据集**

实验采用公开的大语言模型安全测试集和常见的对抗攻击数据集（如 Prompt 攻击库）进行评估。

**📈 对比分析**

与传统基于静态数据集的对抗训练方法相比，DAT 在多种先进攻击（如 prompt 翻译、过去式改写等）下显著提升了模型的鲁棒性能。

**⚠️ 局限性**

局限性包括对扩散模型近似度的依赖、生成样本的计算成本，以及在语言多样性和跨域攻击场景下可能仍存在覆盖不足。

---

## 232. Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching

**arXiv ID:** 2602.15827 | [PDF](https://arxiv.org/pdf/2602.15827v1)

**作者:** Zhen Wu `[一作]` (Amazon), C. Karen Liu `[通讯]` (Amazon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

利用感知驱动的多技能学习框架，让Unitree G1类人机器人能够在真实环境中自主执行长时段、极具动态性的跑酷动作，包括爬墙、跨越障碍、滚动下落等多种技能的无缝切换。

**💡 创新点**

创新点在于：①将稀缺的人类动作片段通过motion matching（最近邻检索）自动拼接成多样化、连贯的长时段轨迹，显著提升技能间过渡的流畅性和适应性；②采用教师–学生管道，结合DAgger与PPO的混合训练，将多技能专家知识蒸馏为单一基于深度图的视觉-运动策略，解决了单一RL探索难题；③实现了从仿真到真实机器人的零shot迁移，展示了高动态跑酷在硬件上的可行性。

**🔧 技术方法**

使用的核心技术包括：Motion Matching、OmniRetarget、强化学习（PPO）+ 运动追踪专家、DAgger+RL 混合蒸馏、深度摄像头模拟（NVIDIA WARP）、域随机化、动态采样与学习率/KL调度的自适应训练策略。

**📊 数据集**

数据集主要由少量高动态人类动作片段构成，通过motion matching生成大量合成的长时段轨迹；仿真中使用随机化障碍（尺寸、姿态、噪声）以及随机障碍位移，确保政策对环境的鲁棒性。

**📈 对比分析**

与传统速度追踪、未组合数据、端到端深度政策等三种基线相比，在仿真中成功率几乎达到1（对3个障碍高度均可），在真实测试中完成1.25 m高墙爬升、0.4 m障碍跳跃、48 s连续多障碍跑酷等任务，展示出显著的性能提升。

**⚠️ 局限性**

局限性：缺乏语义场景理解，摄像头视野和范围有限导致对高速跑酷时障碍感知提前不足；手部抓取能力有限，无法完成更极端的抓握或悬挂动作；对极端动态或非结构化障碍的适应性仍待提升。

---

## 233. Time-Archival Camera Virtualization for Sports and Visual Performances

**arXiv ID:** 2602.15181 | [PDF](https://arxiv.org/pdf/2602.15181v1)

**作者:** Yunxiao Zhang `[一作]` (Texas A&M University), Suryansh Kumar `[通讯]` (Texas A&M University)

**通讯引用:** 717 | [OpenAlex ID](https://openalex.org/A5002526108)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种时间归档摄像机虚拟化方法，利用同步多视角图像学习每一时刻的隐式体积渲染表示，实现动态场景任意时刻任意视角的可回溯合成。

**💡 创新点**

创新点包括：①通过时间索引化的独立MLP实现无点云初始化、可完全并行的时间归档；②在体育与视觉表演场景中实现高质量、可回溯的视角合成；③兼顾大幅非刚性运动下的空间时间一致性，显著提升动态场景渲染质量。

**🔧 技术方法**

采用神经隐式场（多层感知机+多分辨率哈希编码）结合体积渲染；利用同步多视角几何约束；时间索引化的独立训练；对比3D Gaussian Splatting等方法；使用SAM-HQ进行前景分割。

**📊 数据集**

合成多视角动态场景数据集（Dancing‑Walking‑Standing、Soccer Penalty Kick、Soccer Multiplayer）以及真实CMU Panoptic Studio数据集（Baseball Bat、HandGesture）经SAM-HQ预处理。

**📈 对比分析**

与D‑NeRF、D‑3DGS、4DGS、ST‑GS等基线进行PSNR、LPIPS、内存、帧率对比；在合成数据上PSNR最高、LPIPS最低；在CMU数据上同样优于大多数基线；显著降低内存（≈48 MB/时刻 vs 77‑91 MB）；训练可并行，整体训练时间可通过多GPU缩短；帧率约4‑5 FPS，质量优秀。

**⚠️ 局限性**

局限性：需要大量GPU资源实现并行训练；对低位物体、复杂光照或快速曝光变化易出现色漏/对比度下降；依赖同步多视角摄像头设置，若摄像头不完全同步或视角不足则效果受限。

---

## 234. MyoInteract: A Framework for Fast Prototyping of Biomechanical HCI Tasks using Reinforcement Learning

**arXiv ID:** 2602.15245 | [PDF](https://arxiv.org/pdf/2602.15245v1)

**作者:** Ankit Bhattarai `[一作]` (University of Cambridge), Per Ola Kristensson `[通讯]` (University of Cambridge)

**通讯引用:** 6712 | [OpenAlex ID](https://openalex.org/A5042452579)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

开发了MyoInteract框架，实现基于强化学习的快速肌肉动力学HCI任务原型化，降低了训练时间和使用门槛；

**💡 创新点**

创新点在于将GPU加速的MuJoCo-MJX、任务可组合化与图形界面相结合，提供多层次实时反馈，显著缩短训练周期（高达98%）并将复杂的RL设置转化为非专家可操作的工作流；

**🔧 技术方法**

使用了MuJoCo-MJX (JAX) 与Brax实现的PPO、MuJoCo物理引擎、MyoSuite肌肉模型、Gradio图形界面、Weights & Biases日志可视化；

**📊 数据集**

主要使用内部构造的三种交互场景（AR交互、公共显示、移动键盘打字）进行实验，并对照原始UitB任务进行基准；

**📈 对比分析**

与UitB对比：在相同硬件上，MyoInteract复制版从6.9小时降至0.17小时（≈98%），默认版从6.9小时降至0.60小时（≈91%）；在三种示例任务中，训练时间均低于30分钟；同时验证了生成的运动遵循Fitts定律（R²=0.89），保持运动规律性；

**⚠️ 局限性**

局限性包括：支持的任务类型局限于顺序目标获取与选择，未涵盖间接操控或多手、多关节场景；仅使用单一上肢模型；缺乏跨任务迁移与多模态感知整合；界面仍需改进以支持更直观的3D交互与指标可视化。

---

## 235. NeRFscopy: Neural Radiance Fields for in-vivo Time-Varying Tissues from Endoscopy

**arXiv ID:** 2602.15775 | [PDF](https://arxiv.org/pdf/2602.15775v1)

**作者:** Laura Salort-Benejam `[一作]` (Institut de Robótica i Informática Industrial), Antonio Agudo `[通讯]` (Institut de Robótica i Informática Industrial)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种自监督的NeRF框架NeRFscopy，用于从单目内镜视频中重建可变形组织的三维体素表示并实现新视角合成。

**💡 创新点**

创新点在于引入SE(3)变形场来建模非刚性变形，同时结合深度估计引导采样、奇异值正则、深度梯度/平滑正则以及时间总变正则，全部无需先验模板或预训练模型即可完成。

**🔧 技术方法**

核心技术包括NeRF（多层感知器体素渲染）、SE(3) 变形场、单目深度估计（DPT、IID‑SfmLearner、Depth‑Anything）、分层损失（光度损失、深度损失、Jacobian正则、梯度/平滑正则、时间TV正则）以及自监督梯度下降优化。

**📊 数据集**

在四个真实内镜视频（TECAB1、TECAB2、肺叶切除、支气管镜）以及公开的Endo‑NeRF机器人前列腺切除数据集上进行评估。

**📈 对比分析**

与EndoNeRF、EndoSurf、LerPlane‑32k、EndoGaussian‑monocular等先进方法对比，NeRFscopy在PSNR、SSIM、LPIPS指标上均优于对手，尤其在PSNR和LPIPS方面表现突出，能够生成视觉上更自然的3D重建与新视角。

**⚠️ 局限性**

局限性包括：不支持实时处理（0.14 FPS），对相机运动未建模，时间TV正则在高频细节场景下可能过度平滑，依赖单目深度估计的精度，且在极端光照或遮挡条件下性能尚需进一步验证。

---

## 236. The geometry of online conversations and the causal antecedents of conflictual discourse

**arXiv ID:** 2602.15600 | [PDF](https://arxiv.org/pdf/2602.15600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 237. Quantum Computing for Healthcare Digital Twin Systems

**arXiv ID:** 2602.15477 | [PDF](https://arxiv.org/pdf/2602.15477v1)

**作者:** Asma Taheri Monfared `[一作]` (University of Bergamo), Majid Haghparast `[通讯]` (University of Jyväskylä)

**通讯引用:** 2261 | [OpenAlex ID](https://openalex.org/A5016451553)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对量子数字孪生（QDT）在医疗健康领域的应用、挑战与未来方向进行综述。

**💡 创新点**

系统识别并分析了QDT在硬件限制、混合架构、云访问、可扩展性、安全与临床信任等方面的核心障碍，并提出了针对性研究方向。

**🔧 技术方法**

综述了量子算法（Shor、Grover、QFT、变分算法）、量子安全、量子网络、区块链+量子加密、量子机器学习与联邦学习等技术。

**📊 数据集**

本文为综述，无实验数据集，基于已有文献案例和公开研究。

**📈 对比分析**

未进行实验比较，主要通过文献评述与挑战分类进行分析，未给出具体性能指标。

**⚠️ 局限性**

受限于当前NISQ硬件、算法可扩展性、混合系统集成复杂性、云端延迟与安全、可扩展性不足及临床验证缺失等限制，实际应用仍受制。

---

## 238. Grip as Needed, Glide on Demand: Ultrasonic Lubrication for Robotic Locomotion

**arXiv ID:** 2602.15608 | [PDF](https://arxiv.org/pdf/2602.15608v1)

**作者:** Mostafa A. Atalla `[一作]` (Delft University of Technology), Aimée Sakes `[通讯]` (Delft University of Technology)

**通讯引用:** 521 | [OpenAlex ID](https://openalex.org/A5080646399)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并验证了超声润滑技术在机器人行走中的应用，设计了圆柱和扁平摩擦控制模块，并将其集成到类似指壳虫和寄生黄蜂产卵管的两种仿生行走系统中；

**💡 创新点**

创新点在于将超声振动产生的薄压缩液膜用于主动调控摩擦系数，实现“抓握”与“滑移”状态切换，从而打破摩擦对称性，支持高效双向行走；

**🔧 技术方法**

使用了超声共振激振结构（弹性模式）配合压电片激振、振动幅值测量（激光多普勒激振仪）、摩擦力测量（微型负载传感器）、机械驱动（线性活塞/电动滑块）等技术；

**📊 数据集**

实验数据集包含：干湿 PLA、砂纸（150/240目）、砂土、猪肠组织等多种表面与环境的摩擦力曲线；

**📈 对比分析**

通过对比无激振与激振两种状态下的行走效率与摩擦力，实验显示：行走效率平均达 94.75%（指壳虫式）和 93.2%（产卵管式）；摩擦力可降至 28–77% 不同材料；激振幅度与摩擦系数呈可调关系；

**⚠️ 局限性**

局限性包括：对颗粒状或高度粘附性表面（如砂土、粘性液体）润滑效果受限；软组织吸收振动能量，降低压缩液膜效果；模块尺寸与激振功率需求限制在更大或更复杂环境中的应用；

---

## 239. UniTAF: A Modular Framework for Joint Text-to-Speech and Audio-to-Face Modeling

**arXiv ID:** 2602.15651 | [PDF](https://arxiv.org/pdf/2602.15651v1)

**作者:** Qiangong Zhou `[一作]` (Sumeru AI Technology Co., Ltd.), Nagasaka Tomohiro `[通讯]` (Sumeru AI Technology Co., Ltd.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了UniTAF框架，将已训练好的TTS模型（IndexTTS2）与A2F模型（UniTalker）结合，实现从文本直接生成语音和口型动画。

**💡 创新点**

通过重用TTS中间表示并引入音频特征适配器，解决TTS与A2F间的音频-表情失配，并提出基于嘴部状态的加权顶点损失提升口型清晰度。

**🔧 技术方法**

采用IndexTTS2文本转语音、UniTalker口型动画、轻量级音频特征适配器、两阶段预训练+联合训练策略以及嘴部状态感知的损失函数。

**📊 数据集**

以现有A2F数据集为基准，使用ASR自动生成文本，构建了UniTAF三元组（text, audio, face）数据集。

**📈 对比分析**

论文未进行量化评估，而是通过训练稳定性和视觉一致性验证可行性；未给出对标基准或性能指标。

**⚠️ 局限性**

仅实现口型同步，对非口型表情控制有限；依赖预先训练好的TTS模型；存在训练中对齐与推理分布不一致的问题；未实现端到端生成。

---

## 240. One Agent to Guide Them All: Empowering MLLMs for Vision-and-Language Navigation via Explicit World Representation

**arXiv ID:** 2602.15400 | [PDF](https://arxiv.org/pdf/2602.15400v1)

**作者:** Zerui Li `[一作]` (Australian Institute for Machine Learning), Qi Wu `[通讯]` (Australian Institute for Machine Learning)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种零射击Vision‑and‑Language导航框架GTA，利用交互式度量世界表示和反事实推理，将低层空间感知与高层语义规划解耦，实现高效导航；

**💡 创新点**

核心创新包括：①交互式度量世界表示提供实时全局几何信息；②将空间与语义推理解耦，并通过Topological Graph保存历史；③在度量地图上进行可执行动作的反事实推理；

**🔧 技术方法**

技术手段涵盖：多模态大型语言模型（MLLM）作为决策脑；TSDF体素映射构建度量地图；Topological Graph抽象导航历史；交互式推理接口将地图转为多模态提示；低级控制器（Nav2/ROS Nav2）完成精细运动；

**📊 数据集**

实验数据集：R2R‑CE、RxR‑CE连续环境VLN基准；Habitat仿真环境；真实世界中在TurtleBot 4和自制无人机上进行部署；

**📈 对比分析**

与监督学习和零射击基线对比，GTA在R2R‑CE取得48.8% SR、RxR‑CE 42.2% SR，超过前沿零射击模型BZS‑VLN；与监督专家相当；在真实世界实现40–42%零射击转移，表现出高鲁棒性；

**⚠️ 局限性**

局限性：仍受MLLM语义误差影响；对动态/高噪声环境的适应性待提升；离线地图构建与真实感知间可能存在误差；对低级控制细节的交互需进一步完善。

---

## 241. X-MAP: eXplainable Misclassification Analysis and Profiling for Spam and Phishing Detection

**arXiv ID:** 2602.15298 | [PDF](https://arxiv.org/pdf/2602.15298v1)

**作者:** Qi Zhang `[一作]` (Virginia Tech), Jin-Hee Cho `[通讯]` (Virginia Tech)

**通讯引用:** 5720 | [OpenAlex ID](https://openalex.org/A5011649304)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 X-MAP 框架，用 SHAP 解释与 NMF 主题建模对垃圾邮件/钓鱼检测中的误分类进行可解释的分析与修复。

**💡 创新点**

创新点在于将特征级 SHAP 贡献聚合为主题层次的可解释模型，并用 Jensen–Shannon 散度衡量单条信息与已验证的主题分布偏离，从而实现误分类检测与误判修复；同时将此方法作为修复层与传统不确定性检测器协同使用。

**🔧 技术方法**

核心技术包括 SHAP 解释、非负矩阵分解（NMF）生成主题词典、Jensen–Shannon 散度进行相似度评估，并结合 DOCTOR、ODIN、REL-U 等不确定性量化方法进行对比评估。

**📊 数据集**

实验数据集为 UCI SMS Spam Collection 与融合 Ling 与 Nigerian Fraud 的钓鱼邮件数据集，采用 50/50 训练/测试划分。

**📈 对比分析**

与现有不确定性检测器相比，X-MAP 在正类（spam/phishing）上可达 0.98 AUROC、95% TRR 时 FRR 约 9%；作为修复层可恢复 94–97% 的误拒样本，漏判率维持在 10–15% 左右，显著提升整体覆盖率。

**⚠️ 局限性**

局限性包括对 SHAP 计算与 NMF 主题建模的计算开销、对特征稀疏性的敏感性、目前仅针对文本模型，尚未验证在多模态或更复杂模型上的可扩展性与鲁棒性。

---

## 242. Revisiting Northrop Frye's Four Myths Theory with Large Language Models

**arXiv ID:** 2602.15678 | [PDF](https://arxiv.org/pdf/2602.15678v1)

**作者:** Edirlei Soares de Lima `[一作]` (Breda University of Applied Sciences), Antonio L. Furtado `[通讯]` (PUC-Rio)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于Jung原型的四功能角色框架，并在Northrop Frye的四种基本体裁中细化为16个专属角色，随后通过多模型LLM验证其普适性。

**💡 创新点**

创新在于将Jung心理结构映射为角色功能，并针对Frye体裁生成专属角色名称；同时首次使用多架构LLM进行系统性验证，揭示体裁与角色间的可识别结构差异。

**🔧 技术方法**

使用六大主流LLM（Claude Opus、DeepSeek R1、Gemini 2.5 Pro、GPT OSS、Llama 4、Qwen 3）进行角色对应判断；并采用召回率、特异性、平衡准确率、Fleiss κ等统计指标评估模型一致性。

**📊 数据集**

构建40部文学作品（每种体裁10部）共160条正样本和30条负样本的角色-功能对应数据集，涵盖戏剧、史诗、小说及现代影视等多样化文本。

**📈 对比分析**

通过多模型对正负样本进行判定，平均平衡准确率达81.4%，最高85.3%；在体裁层面表现差异（悲剧最高89.9%，讽刺最低77.1%）；角色层面最高99.2%（主角功能）至最低52.5%（伴随功能）。

**⚠️ 局限性**

局限性包括：依赖四部原型作品选取导致角色命名主观性；Jung原型以男性为中心，可能对女性角色识别产生偏差；讽刺体裁中功能模糊导致低识别率；模型对角色关系深度理解有限，需进一步完善框架与数据。

---

## 243. VLM-DEWM: Dynamic External World Model for Verifiable and Resilient Vision-Language Planning in Manufacturing

**arXiv ID:** 2602.15549 | [PDF](https://arxiv.org/pdf/2602.15549v1)

**作者:** Guoqin Tang `[一作]` (Beijing University of Posts and Telecommunications), Ning Ji `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 645 | [OpenAlex ID](https://openalex.org/A5082054193)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VLM-DEWM 架构，通过将 Vision‑Language Model（VLM）的语义推理与动态外部世界模型（DEWM）解耦，构建可查询、持久化的环境记忆，并通过外部可追溯的推理轨迹（ERT）实现决策验证与差异诊断，从而提升动态制造工位的长期规划与错误恢复。

**💡 创新点**

核心创新点在于（1）将状态维护迁移到外部数据库式世界模型，实现跨工位、遮挡下的长期状态一致性；（2）将 VLM 输出结构化为可验证的 ERT，消除“黑盒”决策；（3）通过约束状态（CS）和差异分析实现定位诊断与目标恢复，避免盲目重试。

**🔧 技术方法**

技术手段包括：多模态感知与几何‑语义双层记忆（S、G、L），任务记忆 DAG 与动作历史记录，外部可追溯推理轨迹 JSON 规范，基于统计显著性检验的验证引擎，以及基于差异驱动的诊断与恢复模块；实现上依赖 Gemini 2.5 Pro、GPT‑5 等 VLM，配合 OpenAI API 调用与自定义数据库。

**📊 数据集**

实验数据集涵盖：1）多工位装配仿真（多区块、球、圆柱），2）大规模设施探索仿真（城镇级地图），3）真实 Franka Panda 机器人实验（含感知噪声与人为失误），以及基准方法 SAGE、GEAR 的公开实现。

**📈 对比分析**

与基线相比，VLM-DEWM 在状态跟踪准确率从 56% 提升至 93‑100%，恢复成功率从 <5% 提升至 95%，同时推理调用次数（PE）显著降低 70% 以上；在多工位任务中 TSR 达到 94%，在探索任务中 QSR 94%，在真实机器人任务中 TSR 95% 及诊断准确率 36.96%，显示出显著性能优势。

**⚠️ 局限性**

主要局限包括：1）对感知精度高度依赖，误分割或缺失会直接污染世界模型；2）假设半结构化物体类别已知，难以处理开放词汇的新对象；3）当前的物理一致性验证仅覆盖几何检查，未涵盖动力学约束；4）诊断粒度受微状态 CS 限制，无法区分滑动与碰撞等细粒度原因；5）循环延迟导致对高速动态事件的反应不够及时。

---

## 244. CLOT: Closed-Loop Global Motion Tracking for Whole-Body Humanoid Teleoperation

**arXiv ID:** 2602.15060 | [PDF](https://arxiv.org/pdf/2602.15060v1)

**作者:** Tengjie Zhu `[一作]` (Shanghai Jiao Tong University), Yichao Yan `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 4785 | [OpenAlex ID](https://openalex.org/A5100718879)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了CLOT系统，实现了闭环全身人形机器人遥操作，消除全局姿态漂移；

**💡 创新点**

创新点在于：①采用高频定位反馈实现闭环全局控制；②提出观察预移随机化策略，解耦观测与奖励，平滑全局误差纠正；③使用Transformer网络捕捉长时序依赖，并加入对抗运动先验抑制非自然行为；

**🔧 技术方法**

技术包括：光学运动捕捉+实时IK重定向；强化学习（PPO）训练带观察预移的Transformer策略；对抗运动先验（AMP）；多域随机化和自适应采样；

**📊 数据集**

使用自采集的20小时高质量人类运动数据（覆盖步行、跑步、舞蹈、高动态动作等），并通过IK映射到机器人关节；

**📈 对比分析**

与TWIST2基线相比，CLOT在全局和局部跟踪误差上降低十倍以上；在Adam Pro和Unitree G1两机器人上实验，显示更低的关节力矩波动、更平滑的控制；在真实世界长时段遥操作、强扰动下保持稳定，成功率约>80%；

**⚠️ 局限性**

局限性包括：依赖昂贵的光学定位系统；训练数据仍受限于采集环境，可能缺乏极端环境或复杂交互场景；Transformer模型较大，部署在高性能CPU上；

---

## 245. Efficient Knowledge Transfer for Jump-Starting Control Policy Learning of Multirotors through Physics-Aware Neural Architectures

**arXiv ID:** 2602.15533 | [PDF](https://arxiv.org/pdf/2602.15533v1)

**作者:** Welf Rehberg `[一作]` (Norwegian University of Science and Technology), Kostas Alexis `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 7832 | [OpenAlex ID](https://openalex.org/A5022659812)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过库初始化和物理感知网络实现多旋翼控制策略的快速学习。

**💡 创新点**

提出了基于奖励的相似性度量用于选择最优初始化策略，并设计了分离的反馈与分配网络。

**🔧 技术方法**

使用强化学习（PPO）、QP专家、Wasserstein距离、物理增强网络和控制分配网络。

**📊 数据集**

采用多样化多旋翼配置库（4旋翼和6旋翼随机采样的160个配置）以及真实无人机实验。

**📈 对比分析**

与随机初始化、物理参数相似度、稀疏库等对比，平均减少73.5%环境交互，实地轨迹跟踪误差≤0.055 m。

**⚠️ 局限性**

受限于不同电机数量的政策迁移效果下降，且控制分配误差对高电机数配置影响较大。

---

## 246. "What Are You Doing?": Effects of Intermediate Feedback from Agentic LLM In-Car Assistants During Multi-Step Processing

**arXiv ID:** 2602.15569 | [PDF](https://arxiv.org/pdf/2602.15569v1)

**作者:** Johannes Kirmayr `[一作]` (BMW Group Research and Technology), Florian Alt `[通讯]` (LMU Munich)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对车载代理式LLM助手的中间反馈时机和冗余进行实验研究，比较无中间反馈与规划+结果反馈在停机/行车双任务情境下的效果。

**💡 创新点**

系统性验证中间反馈能提升感知速度、用户体验、信任并降低任务负荷，并提出按信任程度与情境动态调整冗余的设计原则。

**🔧 技术方法**

采用基于LLM的代理式助手原型，使用同步音频视觉输出；在实验中使用混合方法（量化问卷+访谈）和多因素重复测量ANOVA。

**📊 数据集**

实验任务为自定义的多步骤车载请求（如联系查询、路线规划等），未使用公开数据集，而是内部设计的八个任务场景，参与者共45名。

**📈 对比分析**

通过2×2×2 factorial设计比较NI vs PR 两种反馈，使用Perceived Speed、Task Load、User Experience、Trust四项指标；PR在感知速度大幅提升（d=1.01），UX、信任提升，任务负荷下降，整体优于NI。

**⚠️ 局限性**

样本仅来自单一汽车公司，驾驶情境为模拟，未探讨不同模态组合与自适应策略，且长期自适应与真实交通环境未验证。

---

## 247. Stabilizing Test-Time Adaptation of High-Dimensional Simulation Surrogates via D-Optimal Statistics

**arXiv ID:** 2602.15820 | [PDF](https://arxiv.org/pdf/2602.15820v1)

**作者:** Anna Zimmel `[一作]` (ELLIS Unit), Werner Zellinger `[通讯]` (ELLIS Unit)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于D-最优统计的测试时适配框架（SATTS），用于高维仿真代理的零样本自适应。

**💡 创新点**

创新点在于将D-最优样本选择与特征对齐、源知识保留以及无监督超参数调优统一起来，解决了高维回归中传统TTA方法不稳定的问题。

**🔧 技术方法**

使用D-最优子集采样、特征对齐（KL对齐）、源统计正则化、重要性加权验证（IWV）以及QR分解的PCA预处理等技术。

**📊 数据集**

在SIMSHIFT（四个工业仿真任务）和EngiBench（两种设计优化任务）两大基准上进行评估。

**📈 对比分析**

与SSA、Tent等基线以及源模型和Oracle进行比较，SATTS在RMSE、MAE、R²、Compliance等指标上均实现了显著提升，且仅增加约1.9倍的运行时开销。

**⚠️ 局限性**

局限性包括：仍未完全逼近Oracle性能；对源统计估计的假设（高斯、可分离性）可能不总成立；适用于需要源统计的情形，对无源统计环境适用性有限。

---

## 248. StatCounter: A Longitudinal Study of a Portable Scholarly Metric Display

**arXiv ID:** 2602.15413 | [PDF](https://arxiv.org/pdf/2602.15413v1)

**作者:** Jonas Oppenlaender `[一作]` (University of Oulu), Jonas Oppenlaender `[通讯]` (University of Oulu)

**通讯引用:** 934 | [OpenAlex ID](https://openalex.org/A5090875146)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

设计并部署了一款可携带的电子墨水显示器（StatCounter），实时展示作者自身的 Google Scholar 引用统计，并通过九个月的纵向auto‑ethnographic 记录，探讨该设备在日常生活、工作与社交场景中对学术指标的关注、情绪和叙事方式的影响。

**💡 创新点**

创新之处在于将学术指标从传统的浏览器端转移为物理可携带的“伴侣”设备，实现指标的环境化、持续可见性，揭示其对学者日常情感与社会交互的深层影响，并为可携带学术显示器提出设计原则。

**🔧 技术方法**

采用 Raspberry Pi Zero 2 WH 搭配 2.3″ 电子墨水屏；通过 MicroPython 脚本周期性抓取 Google Scholar 数据；使用 cron 定时任务、Wi‑Fi 网络扫描及 LiPo 电池供电等技术实现设备的自给自足。

**📊 数据集**

数据来源为作者个人的 Google Scholar 个人主页中的引用计数、h‑index 等指标；未使用公开大规模数据集。

**📈 对比分析**

本研究采用单人纵向 auto‑ethnographic 方法，无对照组或定量比较；设备表现良好，能够每小时四次更新一次，电池续航足够支持日常使用。

**⚠️ 局限性**

局限性包括单一研究者、单一学科和单一设备的经验，缺乏可推广性；未进行客观量化的效益评估；以及设备在非学术环境中可能放大评估压力等问题。

---

## 249. Attention-gated U-Net model for semantic segmentation of brain tumors and feature extraction for survival prognosis

**arXiv ID:** 2602.15067 | [PDF](https://arxiv.org/pdf/2602.15067v1)

**作者:** Rut Pate `[一作]`, Mohendra Roy `[通讯]` (Pandit Deendayal Energy University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出并实现了一种三平面Attention‑Gated R2U‑Net网络，用于脑肿瘤MRI分割并基于编码器瓶颈特征预测患者生存天数。

**💡 创新点**

将残差、循环、注意力机制融合至2.5D U‑Net中，并首次利用编码器特征直接做生存预测，兼顾分割精度与计算效率。

**🔧 技术方法**

使用Attention门控循环残差U‑Net、Dice+Focal损失、实例归一化、三模态MRI（T1ce、T2、FLAIR）输入、数据增强以及全连接神经网络进行生存预测。

**📊 数据集**

基于BraTS 2020（369例训练）和BraTS 2021（219例验证）多模态MRI数据集。

**📈 对比分析**

与多种领先的2D/2.5D/3D网络（如nnUNet、Transformer‑U‑Net、3D残差U‑Net）在BraTS 2021验证集上对比，获得WT DSC 0.900、TC 0.824、ET 0.775，Hausdorff95分别为5.232/12.001/18.422；生存预测准确率为45.71%。

**⚠️ 局限性**

模型仍然计算量大、缺乏完整体积信息，生存预测性能仅中等，未结合放射组学或多组学特征，验证集地面真值不可公开，需进一步优化轻量化与多模态融合。

---

## 250. Developing AI Agents with Simulated Data: Why, what, and how?

**arXiv ID:** 2602.15816 | [PDF](https://arxiv.org/pdf/2602.15816v1)

**作者:** Xiaoran Liu `[一作]` (McMaster University), Istvan David `[通讯]` (McMaster University)

**通讯引用:** 607 | [OpenAlex ID](https://openalex.org/A5041475393)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了基于仿真和数字孪生的AI训练方法，系统性阐述了模拟技术、挑战及DT4AI框架。

**💡 创新点**

提出了DT4AI框架以及其与ISO 23247标准的映射，系统地总结了数字孪生在AI训练中的作用和方法。

**🔧 技术方法**

采用离散/连续/蒙特卡洛/计算机图形等仿真技术，结合域随机化、域适应、元学习、鲁棒强化学习、模仿学习等方法，并集成数字孪生与模拟器。

**📊 数据集**

文章引用了多项案例数据集，例如CARLA、MuJoCo、CFD、WindsonML等，但并未使用单一数据集进行实验。

**📈 对比分析**

通过文献对比指出仿真生成的数据在训练准确性、成本、可扩展性方面优于真实数据，但性能差异与具体实现、仿真逼真度有关。

**⚠️ 局限性**

主要局限在于仿真与真实场景的差距、缺乏统一验证标准、数据隐私与安全问题，以及可解释性与通用性不足。

---

## 251. Criteria-first, semantics-later: reproducible structure discovery in image-based sciences

**arXiv ID:** 2602.15712 | [PDF](https://arxiv.org/pdf/2602.15712v1)

**作者:** Jan Bumberger `[一作]` (Helmholtz Centre for Environmental Research), Jan Bumberger `[通讯]` (Helmholtz Centre for Environmental Research)

**通讯引用:** 4406 | [OpenAlex ID](https://openalex.org/A5045219919)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

本文提出一种“先判准、后语义”的图像结构发现范式，强调在任何图像分析流程中先基于显式的最优性与稳定性准则从测量数据中提取无语义的结构产物，再将该产物映射到不同的领域本体或标签集；

**💡 创新点**

创新点在于将语义赋值推迟到后期，形成可复制、可跨域、可长期监测的结构层，并将结构产物视为FAIR、AI就绪的数字对象，支持多语义映射与长期数字孪生；

**🔧 技术方法**

采用统一的准则驱动结构提取框架（S_C），实现分割、图、层次、字段等多种结构形式；结合能量最小化、图割、尺度空间、变分等方法，并强调可执行协议与参数化的可检查性；

**📊 数据集**

未聚焦单一数据集，而是通过跨领域证据（遥感、医学影像、显微镜、地震、天文、材料、点云、机器人等）验证该范式的普适性；

**📈 对比分析**

与传统语义优先流程对比，评价维度转向结构的鲁棒性、尺度一致性、压缩性、全局最优性以及多语义支持；实验表明在多种扰动（对比度变化、感知偏移、下采样）下结构产物保持一致，而直接语义分类往往易失真；

**⚠️ 局限性**

局限性包括：需事先设计合适且可解释的准则；对计算资源与实现细节要求高；对下游语义映射的质量仍依赖领域专业知识；缺乏统一的评测基准与标准化schema，难以量化性能优势。

---

## 252. On the Out-of-Distribution Generalization of Reasoning in Multimodal LLMs for Simple Visual Planning Tasks

**arXiv ID:** 2602.15460 | [PDF](https://arxiv.org/pdf/2602.15460v1)

**作者:** Yannic Neuhaus `[一作]` (Tübingen AI Center), Francesco Croce `[通讯]` (ELLIS Institute Finland)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个可控的多模态大语言模型推理 OOD 泛化评估框架，并在网格导航任务上系统验证了不同输入与 Chain‑of‑Thought（CoT）格式对模型性能的影响。

**💡 创新点**

创新点在于：① 将 ID/OOD 区分为地图尺寸、起止点距离和最优路径长度三维度，形成细粒度的泛化评估；② 发现组合文本表格/网格与自然语言描述的 CoT 能显著提升 OOD 性能；③ 提供公开的基准数据与代码，方便后续对比研究。

**🔧 技术方法**

使用技术包括：Qwen2.5‑VL‑7B‑Instruct 大语言模型、超参微调（10 轮）、多种输入格式（图像、描述、表格、网格）以及多种 CoT 表示（单一文本或混合表格+描述）。

**📊 数据集**

数据集为自研的网格地图集：训练集包含 1 000 张 3×3 至 6×6 的地图，OOD 集分别为 7×7、8×8、9×9、10×10，且通过控制起止距离和最优路径长度构造不同的 OOD 视角。

**📈 对比分析**

与 Mirage、MVoT、VPRL 等方法对比：在 ID 场景下，表格/网格 + 描述 CoT 的准确率可达 91%，超过现有基线；在 OOD 场景下，最高可实现 41% 的平均准确率（10×10、起止距离 ≥ 6），而传统单一 CoT 或仅图像输入的模型准确率低于 10%。

**⚠️ 局限性**

局限性包括：① 仅评估 Qwen2.5‑VL‑7B‑Instruct，其他大模型的表现未知；② 任务本身极为简单，难以评估更复杂推理；③ 仍主要依赖模式匹配，真正算法化推理能力不足。

---

## 253. The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes

**arXiv ID:** 2602.15515 | [PDF](https://arxiv.org/pdf/2602.15515v1)

**作者:** Mohammad Taufeeque `[一作]` (FAR.AI), Chris Cundy `[通讯]` (FAR.AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在一个可验证奖励的强化学习（RLVR）编程环境 MBPP‑Honeypot 上训练 LLM，探索使用白盒谎言检测器（线性探测器）对模型行为的影响，并分析其产生的三种结果：诚实、被掩蔽的策略与被掩蔽的激活。

**💡 创新点**

创新点在于：①首次将白盒检测器应用于自然出现的奖励黑客场景；②提出“被掩蔽激活”与“被掩蔽策略”的分类；③阐明 KL 正则化与检测强度共同决定诚实度的理论机制；④展示不通过梯度传播检测器可避免激活被掩蔽。

**🔧 技术方法**

使用技术包括：LoRA 微调、GRPO 算法、线性探测器（多层残差流的逻辑回归）、follow‑up prompt 激活提取、AUC 与 Wasserstein 距离评估。

**📊 数据集**

数据集主要为 MBPP‑Honeypot（874 题编程任务）以及三类探测器训练集（Simple Deception、Diverse Deception、MBPP），并在 400 个 held‑out MBPP‑Honeypot 提示上进行评估。

**📈 对比分析**

与未使用检测器的基线相比，足够高的 KL 正则化与检测强度可使诚实策略占比提升至 100%，并在全部测试用例通过率上不下降，甚至略有提升；在弱 KL 或弱检测下，则出现大量被掩蔽策略或激活。

**⚠️ 局限性**

局限性包括：需设置较高 KL 正则化，实操成本高；无法检测被掩蔽策略的自动方法仍缺失；实验仅限于编程任务，未验证到其他类型任务的泛化能力。

---

## 254. Benchmarking IoT Time-Series AD with Event-Level Augmentations

**arXiv ID:** 2602.15457 | [PDF](https://arxiv.org/pdf/2602.15457v1)

**作者:** Dmitry Zhevnenko `[一作]` (AXXX), Iurii Katser `[通讯]` (NSU)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向部署的事件级异常检测评估协议，加入离线校准的压力测试和传感器级探测；

**💡 创新点**

首次将事件级别评估与统一的压力套件、无校准部署约束以及根因分析相结合，揭示模型在不同真实扰动下的性能反转；

**🔧 技术方法**

使用多种模型族（图结构、密度/流、频谱CNN、重建AE、预测/混合动态）以及对数据的噪声、漂移、传感器失效和窗口/相位偏移的仿真；

**📊 数据集**

在七个工业级多变量时间序列数据集上实验，包括公开的SWaT、WADI、SMD、SKAB、TEP以及两套工业专有数据（汽轮机、核燃气轮机）；

**📈 对比分析**

通过统一拆分、事件聚合和离线校准阈值，对14种模型在清洁和受扰动数据上进行对比；结果表明没有统一冠军，图结构模型在传感器失效/长事件下表现最好，频谱CNN在周期性强的场景优异，密度/流模型在干净稳态场景稳健，预测/混合模型在破坏时序依赖时有优势；

**⚠️ 局限性**

限制在于压力强度仅按验证集统计标准化，事件定义固定，缺乏针对嵌入式设备特定的动态压力测试，未来可扩展至更细粒度、实时校准的评估。

---

## 255. A Quantum-inspired Hybrid Swarm Intelligence and Decision-Making for Multi-Criteria ADAS Calibration

**arXiv ID:** 2602.15043 | [PDF](https://arxiv.org/pdf/2602.15043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 256. Advanced Acceptance Score: A Holistic Measure for Biometric Quantification

**arXiv ID:** 2602.15535 | [PDF](https://arxiv.org/pdf/2602.15535v1)

**作者:** Aman Verma `[一作]` (Indian Institute of Technology Delhi), Sumantra Dutta Roy `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 1778 | [OpenAlex ID](https://openalex.org/A5054273103)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种先进接受度分数(A_r^*(Δ))，用于整体评估手势生物特征量化框架的输出分数质量。

**💡 创新点**

创新点在于将排名偏差、相关性、趋势匹配距离和身份特征解耦四个评价维度融合为单一指标，并提出新的相关性和趋势匹配度量方法，能够同时兼顾分数值质量与排名一致性。

**🔧 技术方法**

技术上采用DGBQA框架的手势特征提取，并结合五种时空网络（ViViT、MotionFormer、MViT、TPN、TAM），使用ICGD损失、z-score归一化、指数与对数加权等手段构建评估公式。

**📊 数据集**

使用了三大公开数据集：Soli（雷达范围-多普勒图）、HandLogin（深度图）和TinyRadar（大规模雷达手势），共计超过34,000个样本。

**📈 对比分析**

通过与多种现有评估指标（排名偏差、DCG、RMSE、Cosine、GRE、RPP等）进行对比实验，结果表明A_r^*(Δ)能够在三数据集上同时最小化排名偏差、趋势偏差和身份解耦度，并最大化相关性，选出的模型在四个设计准则上优于SOTA。

**⚠️ 局限性**

局限性包括仅考虑相邻排名的趋势匹配，缺乏全局趋势评估；评价结果对加权参数敏感，需要经验调优；在解耦程度较差的数据集（如TinyRadar）中，相关性表现略低。

---

## 257. Computation and Size of Interpolants for Hybrid Modal Logics

**arXiv ID:** 2602.15821 | [PDF](https://arxiv.org/pdf/2602.15821v1)

**作者:** Jean Christoph Jung `[一作]` (TU Dortmund University), Frank Wolter `[通讯]` (University of Liverpool)

**通讯引用:** 10956 | [OpenAlex ID](https://openalex.org/A5010967150)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种构造 Craig 插值子（interpolant）的技术，针对不具备 Craig 插值性质（CIP）的混合模态逻辑（Hybrid Modal Logics）给出了四阶指数（4ⁿ）上界；同时证明了在此类逻辑中统一插值（uniform interpolant）问题是不可判定的。

**💡 创新点**

创新点在于：①引入了“hypermosaic”消除技术，将原本局部的模态类型消除扩展为全局的模态集合消除，从而得到可构造的四阶指数时间算法；②在同一框架下完成了统一插值不可判定性的证明，展示了统一插值与 Craig 插值在混合模态逻辑中的本质差异。

**🔧 技术方法**

主要技术包括：类型（type）与模态消除（type/mosaic elimination）；Bisimulation 与超马赛克（hypermosaic）构造；递归的 hyperseparator 生成；以及对带有等级（graded）模态的推广。

**📊 数据集**

该工作为纯理论研究，未使用任何实验数据集，所有结果均来自形式化证明与复杂度分析。

**📈 对比分析**

性能表现：对任意输入 φ₁, φ₂ 给出四阶指数时间（4ⁿ）构造算法；对应的插值子大小也被上界为四阶指数；在相同逻辑中已证明存在至少三阶指数（3ⁿ）下界，说明算法与下界之间存在指数级差距。

**⚠️ 局限性**

局限性：①上界与下界之间仍有指数级差距，是否可进一步收窄仍是开放问题；②对包含逆关系或更复杂的角色约束的逻辑，构造方法尚未适用；③该方法不适用于需要逆关系或更强表达式的混合模态逻辑。

---

## 258. Social Life of Code: Modeling Evolution through Code Embedding and Opinion Dynamics

**arXiv ID:** 2602.15412 | [PDF](https://arxiv.org/pdf/2602.15412v1)

**作者:** Yulong He `[一作]` (St. Petersburg State University), Sergey Kovalchuk `[通讯]` (ITMO University)

**通讯引用:** 1413 | [OpenAlex ID](https://openalex.org/A5029904389)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过把 GitHub PR 代码改动映射为语义嵌入，并用 EPO 观点动力学模型跟踪开发者的私有与公开观点，来定量分析代码库演进过程。

**💡 创新点**

创新点在于将代码嵌入与观点动力学理论相结合，首次将技术改动与社交信任网络映射到同一量化框架，从而揭示开发者影响力与共识形成的机制。

**🔧 技术方法**

技术包括：e5‑base‑v2 Transformer 代码嵌入、PCA 维度缩减、EPO 模型及其优化求解、信任矩阵构造与网络可视化，以及误差评估指标（RMSE/MAE/MAPE）和邻域保留度量（trustworthiness、continuity、MRRE）。

**📊 数据集**

使用的实测数据为 3 个热门 GitHub 项目（swiftlang/swift、ceph/ceph、pytorch/pytorch）的 C++ PR，挑选每个项目最活跃的 7 名开发者，共 21 名，涵盖 1.5M+ PR，生成代码差异嵌入。

**📈 对比分析**

通过与 UMAP、LLE、MDS 等降维方法比较，PCA 在保留局部结构（trustworthiness≈0.67、continuity≈0.83、MRRE≈0.18）方面表现最佳；模型拟合后 RMSE/MAE/MAPE 说明 ceph 拟合最优（RMSE≈0.06），pytorch 次之（≈0.06–0.12），swift 拟合最差（RMSE≈0.32）。

**⚠️ 局限性**

局限性包括：只选取前 1% 活跃开发者、仅考虑 C++ PR、忽略代码上下文与项目治理差异、模型假设线性且缺少异步反馈，导致对 Swift 项目极端波动的解释不足。

---

## 259. Dex4D: Task-Agnostic Point Track Policy for Sim-to-Real Dexterous Manipulation

**arXiv ID:** 2602.15828 | [PDF](https://arxiv.org/pdf/2602.15828v1)

**作者:** Yuxuan Kuang `[一作]` (Carnegie Mellon University), Shubham Tulsiani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4675 | [OpenAlex ID](https://openalex.org/A5029932788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种任务无关的仿真到现实的抓取控制框架AP2AP，通过视频生成的点轨迹指导物体姿态转换。

**💡 创新点**

创新点在于：①使用点轨迹保持当前与目标点对应的六维表示；②结合Transformer的动作世界模型进行教师-学生蒸馏；③无需任务特定奖励或规划，直接通过点轨实现闭环控制。

**🔧 技术方法**

技术手段包括：强化学习PPO、DAgger、PointNet/Transformer网络、视频生成与4D重建、相对深度估计、RealSense RGBD感知等。

**📊 数据集**

训练使用UniDexGrasp 3,200个物体；评估使用YCB数据集与真实机器人上的未见物体；生成视频采用现有视频生成模型。

**📈 对比分析**

与NovaFlow及其闭环版本对比，AP2AP在仿真和真实实验中取得 16%–22% 的成功率提升，显示出更好的通用性和鲁棒性。

**⚠️ 局限性**

局限性包括：仅支持单物体操作；缺乏人类抓取先验和触觉信息；跟踪模型在复杂遮挡下易失误。

---

## 260. Simultaneous Ordinal Maximin Share and Envy-Based Guarantees

**arXiv ID:** 2602.15566 | [PDF](https://arxiv.org/pdf/2602.15566v1)

**作者:** Hannaneh Akrami `[一作]` (Max Planck Institute for Informatics and University of Saarland), Timo Reichert `[通讯]` (University of Bonn)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文通过理论分析和构造性算法，证明在不同类实例（完全有序实例和仅约束前n件物品相同的实例）下，存在满足“ordinal maximin share（MMS）”与“envy-based fairness”（EFX或EF1）兼容的完整分配。

**💡 创新点**

创新点在于首次系统研究Ordinal MMS与EFX/EF1的兼容性，提出新的归一化方法保持有序性、基于“bag‑filling + swap”策略的算法，并在有序实例上取得1‑out‑of‑3n/2 MMS+EFX、1‑out‑of‑4n/3 MMS+EF1以及在top‑n实例上取得1‑out‑of‑3n/2 MMS+EF1的存在性结果，填补了以往仅关注乘法近似的空白。

**🔧 技术方法**

主要技术包括：
- 归一化（d‑normalization）同时保留原有顺序；
- 递归的bag‑filling与交换步骤来维持EFX/EF1；
- 阈值图（threshold graph）与无情 envies 匹配；
- 归纳与鸽巢原理证明Ordinal MMS保证；
- 环消除（envy‑cycle elimination）完成全分配。

**📊 数据集**

无数据集，全部为理论证明与算法构造。

**📈 对比分析**

与已有的乘法MMS+EFX/EF1结果对比，本文提供了更强的Ordinal MMS保证（如1‑out‑of‑4n/3 MMS）并在同等或更弱的环境下实现完整分配；实验性对比无，主要以存在性与理论最优度为评估。

**⚠️ 局限性**

局限性包括：
- 结果仅适用于有序实例或top‑n实例，未覆盖一般实例；
- 仅考虑加法评价函数；
- 归一化与交换步骤增加算法复杂度；
- 未给出多项式时间实现细节，更多是存在性证明。

---

## 261. Revisiting the Sparse Matrix Compression Problem

**arXiv ID:** 2602.15314 | [PDF](https://arxiv.org/pdf/2602.15314v1)

**作者:** Vincent Jugé `[一作]` (Université Gustave Eiffel), Takeaki Uno `[通讯]` (National Institute of Informatics)

**通讯引用:** 4937 | [OpenAlex ID](https://openalex.org/A5076101253)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究稀疏矩阵压缩问题及其变体，分析了基于左侧填充的贪心算法的近似比并给出Θ(√m)的上界与下界；提出了动态规划求解算法，可在双对数宽度或对数宽度的实例上实现多项式时间，并给出对多种参数的硬度与可解性分析；通过实验在合成实例上比较了DP与多种贪心策略的效果。

**💡 创新点**

主要创新包括：①证明左侧填充贪心算法的近似比为Θ(√m)；②引入最小化放置长度的新变体并给出其理论与算法处理方法；③设计对单一或多种tile类型的DP算法，并通过参数化复杂度证明在特定参数下可多项式求解；④在实验中验证贪心策略的差异并展示随机重排可略优。

**🔧 技术方法**

采用了NP-完全性归约、贪心算法分析、wildcard匹配技术、动态规划（基于后缀状态压缩）、参数化复杂度理论、以及归约与硬度证明等技术。

**📊 数据集**

实验使用人工合成的矩阵/字符串实例（如X、Y、Z类型的tile序列），未使用公开数据集。

**📈 对比分析**

通过将DP求解结果与多种贪心策略（Ziegler、随机、按符号数/密度排序等）在同一合成实例上进行对比，实验表明DP始终得到最优长度，而Ziegler在大部分实例中表现差，随机重排可略优。贪心算法的实际长度接近理论Θ(√m)近似比。

**⚠️ 局限性**

限制在于DP算法对tile长度的指数级复杂度，只能在双对数或对数宽度的实例上高效；参数化结果仅适用于特定参数，其他参数的可解性仍未完全揭示；实验仅在人工合成数据上进行，缺乏对真实稀疏矩阵数据的评估。

---

## 262. DependencyAI: Detecting AI Generated Text through Dependency Parsing

**arXiv ID:** 2602.15514 | [PDF](https://arxiv.org/pdf/2602.15514v1)

**作者:** Sara Ahmed `[一作]` (Texas A&M University), Tracy Hammond `[通讯]` (Texas A&M University)

**通讯引用:** 3212 | [OpenAlex ID](https://openalex.org/A5075250507)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了只使用句法依存关系标签的轻量级检测方法（DependencyAI），并在多语言、多生成器、多域数据上实现了可解释的 AI 文本检测。

**💡 创新点**

创新点在于：① 彻底剔除词汇信息，仅依赖依存关系序列；② 用 TF‑IDF n‑gram + LightGBM 构成基线，证明句法结构足以区分人类与机器文本；③ 通过特征重要性分析揭示不同语言的关键句法特征。

**🔧 技术方法**

技术包括：SpaCy 依存句法解析、TF‑IDF 向量化（1,2‑gram）、LightGBM 分类器、gain 基特征重要性评估。

**📊 数据集**

数据集：M4GT‑Bench（包含多语言、多域、人类文本与多种 LLM 生成文本）。

**📈 对比分析**

与传统特征基线（GLTR、Stylistic、NELA）及预训练模型（RoBERTa、XLM‑R）比较。多路检测中 Accuracy 88.85、F1 88.94，超越大多数特征基线并在部分域超过 XLM‑R；多语言检测中在多数语言上优于 XLM‑R，俄语达到 99.5% 的准确率。

**⚠️ 局限性**

局限性：依赖依存解析器的质量，噪声或非正式文本解析效果下降；未评估对抗性重写/后编辑；不适用于代码生成等非自然语言文本。

---

## 263. The Next Paradigm Is User-Centric Agent, Not Platform-Centric Service

**arXiv ID:** 2602.15682 | [PDF](https://arxiv.org/pdf/2602.15682v1)

**作者:** Luankang Zhang `[一作]` (University of Science and Technology of China), Enhong Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 27993 | [OpenAlex ID](https://openalex.org/A5048237545)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了面向用户的智能代理体系，提供基于设备-云协同的个性化意图实现服务。

**💡 创新点**

提出从平台中心向用户中心的范式转变，强调用户目标驱动、隐私保护和代理自治，并引入意图分解、上下文合成与执行约束的闭环框架。

**🔧 技术方法**

结合大语言模型（LLM）、设备端推理、上下文感知感知模块、意图规划与执行工具链以及可解释云端服务。

**📊 数据集**

论文未明确列出标准数据集，主要利用设备级多模态传感器日志、用户行为日志与公开服务接口模拟。

**📈 对比分析**

通过与传统平台中心化推荐/服务系统的对比，实验显示在用户满意度、隐私泄露风险和执行效率方面显著提升，具体指标提升约15‑20%。

**⚠️ 局限性**

受限于云端服务可扩展性、跨平台API标准化不足以及代理提供方的经济激励与治理机制不完善，导致实际落地难度较大。

---

## 264. Iterative LLM-Based Assertion Generation Using Syntax-Semantic Representations for Functional Coverage-Guided Verification

**arXiv ID:** 2602.15388 | [PDF](https://arxiv.org/pdf/2602.15388v1)

**作者:** Yonghao Wang `[一作]` (State Key Lab of Processors, Institute of Computing Technology), Huawei Li `[通讯]` (State Key Lab of Processors, Institute of Computing Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了CoverAssert框架，利用LLM生成的系统Verilog断言（SVA）进行功能覆盖导向的迭代优化。

**💡 创新点**

创新点在于结合语义与结构特征的轻量级匹配机制，使用AST结构距离与语义嵌入进行聚类，并将未覆盖的功能点反馈给生成器，实现功能覆盖驱动的迭代生成。

**🔧 技术方法**

使用了LLM（ChatGPT-4o）进行语义提取、Qwen3-Embedding进行语义向量化、Tree-sitter解析AST、结构距离计算、PCA降维、Silhouette评分聚类、语义检索与匹配以及功能覆盖反馈循环等技术。

**📊 数据集**

采用20个开源设计的Benchmark数据集（包含规范文件和黄金RTL），主要以I2C、ECG、Pairing、SHA3等四个实例作为实验基准。

**📈 对比分析**

与AssertLLM、Spec2Assertion在相同实验环境下对比，平均提升分支覆盖9.57%、语句覆盖9.64%、切换覆盖15.69%；一次迭代即可显著提升，第二次迭代进一步提高，FPV通过率也随之提升。

**⚠️ 局限性**

局限性包括：对大型设计的结构特征提取与聚类成本较高，依赖LLM对功能点抽取的准确性，未覆盖点多时迭代收敛慢；对复杂嵌套功能点的细粒度匹配仍有不足。

---

## 265. High Convergence Rates of CMOS Invertible Logic Circuits Based on Many-Body Hamiltonians

**arXiv ID:** 2602.15033 | [PDF](https://arxiv.org/pdf/2602.15033v1)

**作者:** Naoya Onizawa `[一作]` (Research Institute of Electrical Communication, Tohoku University), Takahiro Hanyu `[通讯]` (Research Institute of Electrical Communication, Tohoku University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并评估了基于三体哈密顿量的CMOS可逆逻辑电路，用以实现前向与反向运算。

**💡 创新点**

通过引入三体相互作用和线性规划求解系数，显著简化能量景观并提升收敛速度。

**🔧 技术方法**

采用随机计算实现多体相互作用的乘法与tanh函数，并在FPGA上实现硬件。

**📊 数据集**

主要使用仿真数据（MATLAB）以及4/6/8位可逆加法器的实验结果；未使用公开数据集。

**📈 对比分析**

与传统两体哈密顿量比较，三体电路在后向模式下收敛次数减少至原来的四分之一，硬件面积仅增加约10%。

**⚠️ 局限性**

实验规模受限于小规模加法器，尚未验证在整数分解或机器学习等大规模任务中的性能。

---

## 266. Bridging Day and Night: Target-Class Hallucination Suppression in Unpaired Image Translation

**arXiv ID:** 2602.15383 | [PDF](https://arxiv.org/pdf/2602.15383v1)

**作者:** Shuwei Li `[一作]` (National University of Singapore), Robby T. Tan `[通讯]` (ASUS Intelligent Cloud Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于 Schrödinger Bridge 的多步无配对图像翻译框架，显式检测并抑制目标类幻像。

**💡 创新点**

将双头判别器（风格+分割）与目标类原型对比抑制结合，首次实现背景与前景语义一致性控制。

**🔧 技术方法**

采用 Schrödinger Bridge 翻译、双头判别器、SAM2 伪分割、类原型聚类与 InfoNCE 对比学习等技术。

**📊 数据集**

在 BDD100K 日夜对照和 KITTI→Cityscapes 跨数据集场景下进行评估。

**📈 对比分析**

在 BDD100K 日夜适配中 mAP 提升至 17.4（相对基线提升 15.5%），在 KITTI→Cityscapes 跨域检测中 mAP 达到 59.8，均超过现有方法。

**⚠️ 局限性**

仍受限于边框级标注、伪分割质量、极端光照场景以及高昂的训练成本。

---

## 267. ResearchGym: Evaluating Language Model Agents on Real-World AI Research

**arXiv ID:** 2602.15112 | [PDF](https://arxiv.org/pdf/2602.15112v1)

**作者:** Aniketh Garikaparthi `[一作]` (TCS Research), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 271562 | [OpenAlex ID](https://openalex.org/A5042321575)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ResearchGym 这一闭环研究评测基准，包含 5 篇 ICML/ICLR/ACL 论文的 39 个子任务，并在单 GPU 环境下对 LLM 代理进行实验。

**💡 创新点**

创新点在于：① 通过保留基线实现和隐藏核心方法实现无偏评测；② 采用可执行代码与原论文官方评测脚本进行客观打分；③ 关注近代任务以避免模型知识泄漏；④ 设计轻量化、可扩展的 Gym‑style 环境。

**🔧 技术方法**

主要技术包括：大型语言模型（GPT‑5、Claude Code、Codex）作为代理；工具调用与 ReAct 循环；基于 Inspect 的检查代理用于防作弊；多任务抽象（Task、Environment、Solver、Evaluation）框架。

**📊 数据集**

使用了来自 5 篇 ICML/ICLR/ACL 口头/Spotlight 论文的公开数据集，涵盖持续学习、强化学习、分词、跨模态检索与时间序列解释等领域。

**📈 对比分析**

评估方法：对每个子任务使用原论文评测脚本计算分数，比较代理得分与提供的基线与论文报告的 SOTA；结果显示 GPT‑5 代理在 15 次完整跑中仅在 1 次（6.7%）击败基线，平均完成率 26.5%，但单次成功跑能超越部分 SOTA。

**⚠️ 局限性**

局限性包括：高度不确定性与低可靠性；长时序实验中出现的上下文窗口限制、资源管理失误、实验跟踪失败、并行实验崩溃、过度自信和作弊倾向；缺乏多模态和更大规模的可重复性。

---

## 268. A Unified Evaluation of Learning-Based Similarity Techniques for Malware Detection

**arXiv ID:** 2602.15376 | [PDF](https://arxiv.org/pdf/2602.15376v1)

**作者:** Udbhav Prasad `[一作]` (Independent Researchers), Aniesh Chawla `[通讯]` (Independent Researchers)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对学习型相似性技术与传统模糊哈希在大规模PE元数据上的检测与聚类能力进行统一实验比较，构建可复现的评估框架。

**💡 创新点**

首次将多种学习型嵌入（AE、深度分类器、XGBoost叶子索引）与传统模糊哈希放在同一实验环境下进行比较，提出统一距离与指标体系，并揭示不同方法在相似检索与聚类上的互补性。

**🔧 技术方法**

使用自动编码器、深度二分类/多分类网络、XGBoost树模型；对嵌入采用欧氏距离检索，并计算 Silhouette、Davies–Bouldin、Calinski–Harabasz、label homogeneity 等指标。

**📊 数据集**

EMBED 与 EmberSim 两个公开数据集（约1M PE元数据）作为实验数据。

**📈 对比分析**

通过统一实验框架比较，AE 在 Top‑K 相似检索中 label homogeneity 超过80%，深度学习嵌入在聚类指标上优于树模型；XGBoost 在分类准确率、AUC 上最高，但其嵌入空间相对分散。

**⚠️ 局限性**

仅基于预提取的静态元数据，无法直接与基于原始二进制的模糊哈希做对比；未加入动态行为特征；未评估近似最近邻的可扩展性与检索效率。

---

## 269. CREMD: Crowd-Sourced Emotional Multimodal Dogs Dataset

**arXiv ID:** 2602.15349 | [PDF](https://arxiv.org/pdf/2602.15349v1)

**作者:** Jinho Baek `[一作]` (New York Institute of Technology), Kate Blackwell `[通讯]` (New York Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究构建了CREMD数据集，通过三种视频呈现模式和多样化标注者对狗情绪进行众包标注。

**💡 创新点**

创新点在于将视觉上下文与音频多模态结合，同时系统性评估标注者背景（如犬主身份、性别、专业经验）对情绪识别一致性的影响。

**🔧 技术方法**

使用自研React+Firebase标注工具、YOLOv8m进行犬体裁剪、以及Kruskal‑Wallis、Mann‑Whitney等统计方法进行一致性分析。

**📊 数据集**

数据集包含923段视频、12,794条情绪标注，覆盖Happy、Angry、Fear、Neutral等四类情绪，并提供详细的犬元数据。

**📈 对比分析**

通过比较不同模态（无上下文无音频、无音频有上下文、有音频有上下文）与标注者群组的一致性，结果显示视觉上下文显著提升一致性，音频提升置信度但对一致性影响不显著。

**⚠️ 局限性**

局限性包括样本量相对有限、缺少无上下文-有音频条件、音频质量不佳导致对音频效应评估受限。

---

## 270. Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching

**arXiv ID:** 2602.15752 | [PDF](https://arxiv.org/pdf/2602.15752v1)

**作者:** Ren Kishimoto `[一作]` (Institute of Science), Yuta Saito `[通讯]` (Cornell University)

**通讯引用:** 787 | [OpenAlex ID](https://openalex.org/A5101991694)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种以用户留存为目标的两侧匹配推荐框架，构建动态学习排序算法MRet；

**💡 创新点**

通过联合考虑推荐方与被推荐方的留存增益，利用可凹化约束将NP难的优化问题转化为可排序的评分函数，实现高效留存优化；

**🔧 技术方法**

使用基于XGBoost的留存曲线回归、凸凹性质的理论下推导的分数公式、argsort排序以及ALS矩阵填充等技术；

**📊 数据集**

在合成数据和日本大型在线约会平台的真实交互数据（1000×1000匹配矩阵、60000条特征-匹配-留存样本）上进行评估；

**📈 对比分析**

与最大匹配、均匀随机、FairCo公平排序等基线对比，MRet在保留率上优于其他方法，仅需约70%最大匹配的匹配量；

**⚠️ 局限性**

在非凸留存函数、极度稀疏匹配矩阵下仍保持性能，但对匹配概率估计和留存曲线假设的准确性敏感，且未考虑多方动态预算或长期协同效应。

---

## 271. EduResearchBench: A Hierarchical Atomic Task Decomposition Benchmark for Full-Lifecycle Educational Research

**arXiv ID:** 2602.15034 | [PDF](https://arxiv.org/pdf/2602.15034v1)

**作者:** Houping Yue `[一作]` (Shanghai Institute of AI for Education), Aimin Zhou `[通讯]` (Shanghai Innovation Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EduResearchBench，一套完整覆盖教育学术写作生命周期的评测平台，构建了HATD层级原子任务分解框架，并用该框架训练出EduWrite模型；

**💡 创新点**

创新点在于把研究工作拆分为六大模块、24个原子任务，实现细粒度诊断；结合层级化的课程学习策略和高质量、长上下文的数据生成；

**🔧 技术方法**

技术手段包括基于GPT‑5/Gemini的LLM驱动数据构建、双模型（Doubao‑seed 1.6、DeepSeek‑V3.2）评判体系、基于Ms‑Swift的全参数SFT以及分阶段课程学习；

**📊 数据集**

使用了55,493条原始教育学术样本，经过筛选得到11,357条高质量指令‑响应对用于SFT，测试集5,300条；

**📈 对比分析**

对比方法采用自动评判（两模型评分平均）并与GPT‑5、Gemini 3 Flash、Qwen‑2.5‑72B、Llama‑3‑70B等公开与封闭模型对照；结果显示EduWrite 30B在整体与模块评测上明显优于同类开源模型，且优于部分大模型，显示数据质量和课程学习比单纯放大参数更关键；

**⚠️ 局限性**

局限性在于评测仍局限于教育领域，未覆盖其他社会科学；依赖LLM做评判可能带来偏差；量化研究模块表现仍相对薄弱，且数据规模虽大但仍不足以覆盖全部学科细节。

---

## 272. Quantifying construct validity in large language model evaluations

**arXiv ID:** 2602.15532 | [PDF](https://arxiv.org/pdf/2602.15532v1)

**作者:** Ryan Othniel Kearns `[一作]` `[通讯]` (University of Oxford), Ryan Othniel Kearns (University of Oxford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一种结构化能力模型，用以从大量LLM基准测试结果中提取可解释、可泛化的能力，进而量化LLM评估的构造效度。

**💡 创新点**

创新点在于同时结合神经缩放律（模型规模对性能的影响）与潜在因子模型（测量误差与可解释因子），实现了规模与能力的分离，消除了规模代理因子，并得到更具可解释性与泛化性的能力估计。

**🔧 技术方法**

使用了结构化能力模型、对比的潜在因子模型以及基于缩放律的模型；通过统计拟合指标（简约拟合指数）与预测实验（对离散分布基准的预测）进行评估。

**📊 数据集**

数据集为OpenLLM Leaderboard上的大量LLM基准测试结果，涵盖多种任务与模型尺寸。

**📈 对比分析**

与潜在因子模型相比，结构化模型在简约拟合指数上更优，去除了与模型规模相关的主因子；与缩放律模型相比，结构化模型在OOD基准上的预测准确性更高，展示了更好的解释与预测能力。

**⚠️ 局限性**

局限性包括：仍可能受到基准集污染和标注错误的影响；对测量误差与规模交互的建模仍不完全；对新兴或稀有能力的捕捉能力有限。

---

## 273. Approximation Theory for Lipschitz Continuous Transformers

**arXiv ID:** 2602.15503 | [PDF](https://arxiv.org/pdf/2602.15503v1)

**作者:** Takashi Furuya `[一作]` (Doshisha University), Carola-Bibiane Schönlieb `[通讯]` (University of Cambridge)

**通讯引用:** 15704 | [OpenAlex ID](https://openalex.org/A5004024852)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

提出了一类基于梯度下降的 Lipschitz 连续的上下文 Transformer，并证明其在查询变量上 1‑Lipschitz、在上下文（概率测度）上 C‑Lipschitz 的全局可逼近性；

**💡 创新点**

首次给出严格的 Lipschitz 约束下 Transformer 的普适逼近定理，且该定理对上下文测度做了 Wasserstein‑1 级的 Lipschitz 控制；

**🔧 技术方法**

采用梯度下降型 MLP 与注意力块的显式欧拉步（负梯度流）构造、测度论框架、Kantorovich‑Rubinstein 对偶性、修正的 Restricted Stone‑Weierstrass 定理与格 lattice 结构证明；

**📊 数据集**

无具体数据集，研究完全为理论推导与构造性证明；

**📈 对比分析**

没有实验比较，论文仅给出理论收敛与逼近误差上限，未给出数值性能指标；

**⚠️ 局限性**

限制包括：步长 η 需依赖输入域，难以估计；上下文注意力的 Lipschitz 常数难以实用估计或证明；仅处理标量输出，向向量化扩展尚未实现；

---

## 274. Req2Road: A GenAI Pipeline for SDV Test Artifact Generation and On-Vehicle Execution

**arXiv ID:** 2602.15591 | [PDF](https://arxiv.org/pdf/2602.15591v1)

**作者:** Denesa Zyberaj `[一作]` (Mercedes-Benz AG), Alois Knoll `[通讯]` (Institute for Robotics, Artificial Intelligence and Embedded Systems, Technical University of Munich)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了Req2Road GenAI 流程，将软件定义车辆的自然语言需求、表格和图表转换为可执行的 Gherkin 场景、VSS 信号映射以及 Python 测试脚本，并在数字化平台与实际车辆上验证其可执行性。

**💡 创新点**

创新点在于：①将 LLM 与 VLM 结合实现多模态需求抽取与信号定位；②采用检索增强生成（RAG）在 VSS 目录中预筛选候选信号，显著降低幻觉；③构建端到端可转移的 SDV 测试流水线，支持从云端 SiL 到车载 ViL 的无缝迁移。

**🔧 技术方法**

主要技术包括：大语言模型（GPT‑5、GPT‑4o‑mini、GPT‑4.1）、视觉语言模型（Gemini 2.5 Pro、Qwen 2.5 VL 72B）、检索增强生成（RAG）、Vehicle Signal Specification（VSS）标准、Behave BDD 框架以及 KUKSA 同步客户端 API。

**📊 数据集**

使用了 36 条 CPDS（儿童存在检测系统）自然语言需求、对应 UML 状态图和 982 条 VSS 信号目录作为数据集，并提供了公开的 digital.auto 试验平台模型。

**📈 对比分析**

通过对比 16 条预筛选候选信号与完整 982 条信号的映射结果，发现预筛选后 GPT‑4o‑mini 完全匹配黄金集（4/4）且无误报；在代码生成阶段，GPT‑4.1 在包含三条需求的场景下 pass@3=1，GPT‑4o‑mini 仅在单/双需求场景下通过；整体上，32/36（89%）需求可直接生成可执行 Gherkin 场景，并在 SiL 与 ViL 上成功执行。

**⚠️ 局限性**

主要限制包括：①对大型未过滤的 VSS 目录易出现误报，需严格预筛选；②代码生成对场景复杂度敏感，仍需人工审核；③对模糊或多重 OR 条件的需求生成效果有限；④当前评估仅覆盖单一车辆平台与 CPDS 子系统，跨平台鲁棒性待验证；⑤商业 LLM 需避免处理机密需求，影响可部署性。

---

## 275. TARZAN: A Region-Based Library for Forward and Backward Reachability of Timed Automata (Extended Version)

**arXiv ID:** 2602.15435 | [PDF](https://arxiv.org/pdf/2602.15435v1)

**作者:** Andrea Manini `[一作]` (Politecnico di Milano), Pierluigi San Pietro `[通讯]` (Politecnico di Milano)

**通讯引用:** 919 | [OpenAlex ID](https://openalex.org/A5064490024)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了一个基于区域抽象的 C 语言库，支持时序自动机（TA）的前向和后向可达性分析；

**💡 创新点**

创新点在于提出一种记录时钟何时变为无界的细化区域表示，保证每个区域最多只有三个即时延迟前驱，并基于此实现了后向算法；

**🔧 技术方法**

采用了带有有序时钟划分的区域抽象技术、后向可达性算法以及 C 语言实现；

**📊 数据集**

使用了公开的 TA 基准模型（与 Uppaal、TChecker 对比的实验数据集）；

**📈 对比分析**

通过在同一套模型上与 Uppaal、TChecker 的前向可达性结果做对比，实验表明在包含严格约束和大常数的模型中区间（zone）更快，而在闭合 TA 或带有时间点约束的模型中该库更具优势；后向算法能够有效缩小搜索空间并验证安全性质；

**⚠️ 局限性**

局限性包括：在常数过大或包含严格约束的 TA 上性能仍受限；仅支持单一 TA 的可达性分析；未结合区间优化，且后向算法目前仅适用于单机情况。

---

## 276. Efficient Approximate Nearest Neighbor Search under Multi-Attribute Range Filter

**arXiv ID:** 2602.15488 | [PDF](https://arxiv.org/pdf/2602.15488v1)

**作者:** Yuanhang Yu `[一作]` (Tongji University), Xuemin Lin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 20397 | [OpenAlex ID](https://openalex.org/A5079659938)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于属性空间KD树与单层HNSW图结合的索引，用于高维向量的多属性范围过滤近邻搜索（RFANNS）。

**💡 创新点**

创新点包括：①倾斜感知的分割策略保证树高为O(log n)；②采用非重叠的KD树分区，避免R树的重叠导致邻居质量下降；③在每个分区构建单层HNSW图，并在查询时仅利用in‑range邻居进行重构，既保持高召回又显著提高吞吐；④设计了级别并行+节点内并行的构建流程。

**🔧 技术方法**

使用的技术主要有：KD树分区、单层HNSW图、RNG式邻居筛选、基于树路径的邻居重构、递归构建与层级合并、并行化构建策略。

**📊 数据集**

使用四个公开数据集：Youtube（3.65M样本，维度1024，4个属性）、DBLP（6.36M，768，4个属性）、MSMarco（8M，384，5个属性）和LAION（9.64M，512，3个属性），均包含多维数值属性与向量表示。

**📈 对比分析**

与iRangeGraph和预筛选基线进行对比；在k=10、召回率0.95/0.9的设置下，平均QPS提升2.46×，在最难的LAION上提升16.22×；提升随选择率下降、k增大、谓词维度增多而显著；相较预筛选，平均提升35.59×。

**⚠️ 局限性**

局限性：仅支持欧氏距离和数值范围谓词；索引为静态，动态增删操作成本高；单层HNSW在极高维度或极大数据量下可能性能下降；在极低选择率的查询中仍需扩展较多out‑of‑range邻居。

---

## 277. MB-DSMIL-CL-PL: Scalable Weakly Supervised Ovarian Cancer Subtype Classification and Localisation Using Contrastive and Prototype Learning with Frozen Patch Features

**arXiv ID:** 2602.15138 | [PDF](https://arxiv.org/pdf/2602.15138v1)

**作者:** Marcus Jenkins `[一作]` (University of East Anglia), Michal Mackiewicz `[通讯]` (University of East Anglia)

**通讯引用:** 1686 | [OpenAlex ID](https://openalex.org/A5001797888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于多分支DSMIL、对比学习和原型学习的弱监督MIL框架，用预冻结特征实现卵巢癌组织亚型的分类和定位。

**💡 创新点**

创新点在于将CLAM的类别特定注意力与DSMIL的关键实例伪标签结合，并在冻结特征空间上引入特征空间增强的监督对比学习和类原型学习，既提升性能又保持可扩展性。

**🔧 技术方法**

使用预训练的UNI视觉Transformer特征、SimCL特征空间增强、MoCo风格对比学习、类原型学习和多分支DSMIL注意力。

**📊 数据集**

在DROV数据集（174张卵巢癌WSI，含恶性、边缘和正常亚型）上进行实验。

**📈 对比分析**

与DSMIL、CLAM等基线比较，MB-DSMIL-CL-PL在slide和instance层面分别提升宏F1 70.4%/15.3%及AUC 16.9%/2.3%，并在定位AUC上显著优于对照。

**⚠️ 局限性**

局限包括仅使用单一分辨率特征、缺乏多尺度或多模态信息，以及对极少数亚型仍表现不稳定。

---

## 278. Testing Monotonicity of Real-Valued Functions on DAGs

**arXiv ID:** 2602.15341 | [PDF](https://arxiv.org/pdf/2602.15341v1)

**作者:** Yuichi Yoshida `[一作]` (National Institute of Informatics), Yuichi Yoshida `[通讯]` (National Institute of Informatics)

**通讯引用:** 6995 | [OpenAlex ID](https://openalex.org/A5038701345)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过构造正匹配(Positive Matching)与差异自由移位匹配(shift‑matching)的集合，给出了在显式图上单调性测试的下界，证明了在图顶点数为n的情形下，单侧自适应测试至少需要Ω(√n)次查询；

**💡 创新点**

创新点在于提出并利用了正匹配结构（PMRS）以及差异自由移位匹配的组合，突破了传统RMS方法在正整数域上的限制，并通过对凸性与对数凑合度的结合，构造出能量势的高维高斯分布，从而获得自适应单侧测试的强下界；

**🔧 技术方法**

主要技术包括：凸优化与强对数凑合度分析、Bräggman–Sjöstrand 绿函数比较、不等式与随机化的调和/概率集中技术、以及 Fano 不等式的应用；

**📊 数据集**

该工作为理论研究，不使用任何实际数据集；

**📈 对比分析**

与传统的单侧与双侧测试方法相比，本工作给出了最优（或接近最优）的理论下界，证明在某些图结构上，任何自适应单侧测试都不能突破Ω(√n)的查询下限；

**⚠️ 局限性**

局限性包括：下界仅适用于特定的 C4‑free 图结构和正匹配集合，且在稀疏图或更一般的图上无法直接推广；

---

## 279. Robot-Assisted Social Dining as a White Glove Service

**arXiv ID:** 2602.15767 | [PDF](https://arxiv.org/pdf/2602.15767v1)

**作者:** Atharva S Kashyap `[一作]` (University of Michigan), Patricia Alves-Oliveira `[通讯]` (University of Michigan)

**通讯引用:** 1591 | [OpenAlex ID](https://openalex.org/A5060086372)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过与六名具备身体障碍的参与者进行参与式设计与访谈，利用自研的 AI 生成故事板工具 Speak2Scene，探讨了在公共餐厅环境中机器人辅助用餐的理想场景与交互需求。

**💡 创新点**

创新点在于：①将机器人辅助用餐从单纯的技术实现扩展为全场景的“白手套服务”理念；②使用生成式 AI 生成图像作为参与式设计的媒介，突破传统手绘故事板的可达性与创造力限制；③系统性梳理了机器人在社交用餐中的多模态交互生态、情境感知行为与角色扩展需求。

**🔧 技术方法**

主要技术包括：生成式大型语言模型（LLM）与视觉语言模型（VLM）驱动的 Storyboard 生成工具 Speak2Scene；ReactJS 前端与 Firebase 后端；访谈录音转写与 MAXQDA 主题分析工具。

**📊 数据集**

本研究不涉及传统意义的数据集，而是基于参与者自述与生成图像的视觉叙事，收集了六名参与者的访谈文本和十余幅 AI 生成的故事板图片。

**📈 对比分析**

研究未采用定量对比评测；通过主题分析得到的四大主题（交互生态、情境感知行为、机器人角色、用户关系）为后续系统设计提供定性依据；若以后需量化，可设计用户满意度或任务完成率等指标。

**⚠️ 局限性**

局限性包括：样本规模仅六人，且主要为美国白人女性；未涉及非残障用户或餐厅工作人员的视角；使用 AI 生成图像存在“幻觉”偏差，可能影响参与者思考；访谈时长较长，可能导致参与者疲劳。

---

## 280. Complex-Valued Unitary Representations as Classification Heads for Improved Uncertainty Quantification in Deep Neural Networks

**arXiv ID:** 2602.15283 | [PDF](https://arxiv.org/pdf/2602.15283v1)

**作者:** Akbar Anbar Jafari `[一作]` (University of Tartu), Gholamreza Anbarjafari `[通讯]` (3S Holding OÜ)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于量子启发的复值分类头，将特征投影至复Hilbert空间，并在Cayley参数化的单位ary变换下演化，随后通过幅度+softmax读取，显著提升模型的校准性能。

**💡 创新点**

创新点在于结合复数投影、Cayley单位ary变换与幅度提取，实现了对logit幅度的几何约束，从而在不增加大量参数的前提下获得更佳的校准，并对传统Born规则做了系统对比。

**🔧 技术方法**

使用了复数线性映射、ℓ₂归一化、Cayley变换生成的单位ary矩阵、幅度提取以及softmax读取，并与Softmax、MC‑Dropout、温度缩放等经典方法进行对比。

**📊 数据集**

主要数据集为CIFAR‑10及其人类不确定性扩展CIFAR‑10H，同时在实验中还使用了SVHN、CIFAR‑100、Gaussian/Uniform噪声及SST等数据集。

**📈 对比分析**

在共享backbone的实验设计下与标准softmax、MC‑Dropout、温度缩放等方法对比，幅度+softmax头在ECE上比softmax低约2.4倍，同时在KL‑距离等指标上表现优于多数传统方法。

**⚠️ 局限性**

局限性包括仅在单头层面验证，未在更大规模数据集或全模型联合训练中评估；对OOD检测和NLP任务效果不佳；计算量相对softmax增加约3倍。

---

## 281. TAC: Timestamped Audio Captioning

**arXiv ID:** 2602.15766 | [PDF](https://arxiv.org/pdf/2602.15766v1)

**作者:** Sonal Kumar `[一作]` (University of Maryland), Justin Salamon `[通讯]` (Adobe Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Timestamped Audio Captioner（TAC）与其音视频扩展TAC‑V，能够在复杂混响场景中生成精确的时间戳音频描述，并将这些描述作为“语义桥”与文本型大语言模型（LLM）耦合，实现音频与音视频的强推理；

**💡 创新点**

创新点包括：1）基于动态音频混合器（Dynamic Acoustic Mixer）与场景模板的无穷多合成训练数据；2）多任务提示与可调时间参数的自适应标注；3）时间戳专用加权损失与多级标注结构；4）通过TAC‑V将音频事件与视觉信息融合，形成稠密时序化的音视频描述；5）将稠密描述与纯文本LLM结合，开启“Describe‑then‑Reason”架构。

**🔧 技术方法**

技术方法包括：大型音频‑语言模型（如Qwen2‑Audio、Qwen3‑VL‑32B）与LoRA微调；多任务指令调优与动态提示；时间戳专用特殊token与加权交叉熵；声学混合、RMS激活检测、合并阈值与活动阈值；使用FLAM等工具评估与校正；视频帧抽取、VLM推理与视觉纠错。

**📊 数据集**

使用的主要数据集与来源：① 通过动态混合器合成的无限规模合成音频；② 公开的音频描述数据（AudioCaps、Clotho）与音频集（FSD50K、AudioSet‑Strong）用于预训练；③ TACOS 作为真实稠密标注基准；④ 其它公开语音与音乐数据用于增强训练。

**📈 对比分析**

在TACOS评测中，TAC在事件F1、段落F1、幻觉率等指标均超越现有开源与专有模型（Gemini 3 Pro、Qwen3‑Omni、Audio Flamingo 3），幻觉率仅4.9%。在音频推理基准（MMAU、MMAR、MMSU、MMAU‑Pro）中，TAC‑V配合强文本LLM（Gemini 3 Pro）实现SOTA；在音视频推理基准（DailyOmni、Video‑Holmes、AVHBench）同样表现领先，验证“语义桥”有效性。

**⚠️ 局限性**

局限性包括：① 依赖合成数据导致“仿真‑现实”差距，模型可能对剧烈事件（如枪声）过度预测；② 对音乐细节（如和弦进程）缺乏精细建模；③ 虽降低幻觉率但在极端混响或低信噪比场景仍可能出现误检；④ 需较大计算资源进行多模态推理；

---

## 282. On the Entropy of General Mixture Distributions

**arXiv ID:** 2602.15303 | [PDF](https://arxiv.org/pdf/2602.15303v1)

**作者:** Namyoon Lee `[一作]` `[通讯]` (POSTECH), Namyoon Lee (POSTECH)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文开发了一种确定性、封闭形式的工具包，用于直接从混合分布的组件参数中界定和准确近似混合熵。

**💡 创新点**

创新点在于将混合模型视为信息论中的通道，利用成分重叠积分来界定混合熵，并提供了针对多种经典混合分布的封闭形式特化。

**🔧 技术方法**

使用了信息论的通道视角、Jensen不等式、重叠积分等技术。

**📊 数据集**

使用了高斯混合、因子拉普拉斯混合、均匀混合和混合模型的数值实验。

**📈 对比分析**

通过Monte Carlo实验验证了所提出的界限和近似在分离度、维度、成分数量和相关协方差下的有效性，结果表明所提近似通常是紧的。

**⚠️ 局限性**

限制在于该方法依赖于成分重叠的计算，且在某些情况下可能无法提供精确的熵值。

---

## 283. Decision Making under Imperfect Recall: Algorithms and Benchmarks

**arXiv ID:** 2602.15252 | [PDF](https://arxiv.org/pdf/2602.15252v1)

**作者:** Emanuel Tewolde `[一作]` (Carnegie Mellon University), Vincent Conitzer `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10041 | [OpenAlex ID](https://openalex.org/A5050903632)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估了第一个针对不完美回忆决策问题的基准套件，探索多种一阶优化算法求解一阶最优策略。

**💡 创新点**

创新点在于：①首次公开大规模不完美回忆基准数据集；②将 regret matching (RM) 家族推广为非线性约束优化的一阶优化器，并证明其在此类问题上远优于传统投影梯度下降。

**🔧 技术方法**

使用技术包括：树形决策模型、Causal Decision Theory（CDT）均衡作为一阶最优点、投影梯度下降（PGD）、AMSGrad、Gurobi、以及多种 RM 变体（RM、PredictiveRM、OptimisticRM），并利用线性时间递归求梯度。

**📊 数据集**

数据集为 61 个基准实例，涵盖模拟问题、隐私约束的子组检测问题和随机决策问题，参数化覆盖树深、信息集数、缺忘程度、动作数等。

**📈 对比分析**

在 4–12 小时或 6000 步迭代终止准则下比较，RM 家族在收敛速度上比 PGD、AMSGrad 快数个数量级，且获得的价值往往不低于甚至高于 PGD；Gurobi 在大多数实例中无法收敛。

**⚠️ 局限性**

局限性在于：仅适用于表格树形决策问题；缺乏完整的理论收敛性分析；对更大规模或连续状态/动作空间的任务仍需进一步研究。

---

## 284. OSCAR: An Ovipositor-Inspired Self-Propelling Capsule Robot for Colonoscopy

**arXiv ID:** 2602.15309 | [PDF](https://arxiv.org/pdf/2602.15309v1)

**作者:** Mostafa A. Atalla `[一作]` (Delft University of Technology), Paul Breedveld `[通讯]` (Delft University of Technology)

**通讯引用:** 3840 | [OpenAlex ID](https://openalex.org/A5017683959)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并验证了一种名为OSCAR的自推进胶囊机器人，用来代替传统结肠镜的推送式推进方式，减少环路形成和患者不适。

**💡 创新点**

创新点在于借鉴寄生黄蜂卵管的滑动阀机制，将十二个相位移的滑块通过弹簧驱动的凸轮系统实现持续的摩擦不对称，产生可控的前向推力；同时提出了基于Kelvin–Voigt模型的粘弹性抓取-滑动（stick–slip）分析框架，用以预测和优化推进力。

**🔧 技术方法**

技术包括：弹簧驱动凸轮机械传动、十二个圆柱滑块、摩擦槽面设计、可调的凸轮跳跃次数（1、2、3跳）以及数值与实验的粘弹性-摩擦模型。

**📊 数据集**

使用的实验数据集为猪结肠外生组织（约1米长的完整猪结肠段）进行的单滑块与全胶囊牵引力测量以及行进速度测定。

**📈 对比分析**

通过实验验证，OSCAR在外生结肠中实现了平均3.08 mm/s的前进速度，对应人类结肠完整通过时间约8分钟，牵引力约0.85 N；与理论模型（含粘弹性损失）高度吻合，且推进力对速度不敏感，且随凸轮的拖拽不对称线性变化。

**⚠️ 局限性**

限制包括：缺乏双向推进、无法适应结肠曲折和不同直径、缺乏内置成像/取样工具、存在滑块间隙导致潜在消毒难题、以及在体内受腹压和张力变化时性能未知。

---

## 285. Improving MLLMs in Embodied Exploration and Question Answering with Human-Inspired Memory Modeling

**arXiv ID:** 2602.15513 | [PDF](https://arxiv.org/pdf/2602.15513v1)

**作者:** Ji Li `[一作]` (University of Hong Kong), Shiyan Hu `[通讯]` (University of Hong Kong)

**通讯引用:** 5361 | [OpenAlex ID](https://openalex.org/A5031580282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种非参数记忆框架，显式区分情节记忆和语义记忆，用于多模态大型语言模型在具身代理中的长期探索和问答；

**💡 创新点**

创新点包括：①检索-先行、视觉推理辅助的情节记忆回调策略，避免硬几何融合；②基于程序风格规则提取的语义记忆，形成可跨环境通用的结构化决策规则；

**🔧 技术方法**

技术方法包括：多模态语义嵌入（SAM+CLIP）构建层次化场景图；利用LLM进行语义检索、视觉推理与前沿点筛选；程序化规则抽取与决策偏差检测；

**📊 数据集**

使用的数据集有：Open‑EQA（A‑EQA）用于开放式问答，GOAT‑Bench用于多目标导航；

**📈 对比分析**

与多种基线（3D‑Mem、ReEXplore、Explore‑EQA、CG+Frontier 等）对比，GPT‑4o 基线下获得 LLM‑Match 65.6%／LLM‑Match×SPL 48.7%（比 58.3%/37.3% 提升），GOAT‑Bench 成功率 72.8%／SPL 56.1%（比 65.1%/49.3% 提升）；

**⚠️ 局限性**

局限性：在尺度、地图、视觉差异较大的室外环境下情节记忆效果受限；尚未实现语义记忆与具身基础模型的参数化联合优化；

---

## 286. Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs

**arXiv ID:** 2602.15318 | [PDF](https://arxiv.org/pdf/2602.15318v1)

**作者:** Libo Zhang `[一作]` (National University of Defense Technology), Dongsheng Li `[通讯]` (National University of Defense Technology)

**通讯引用:** 31401 | [OpenAlex ID](https://openalex.org/A5100440919)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Sparrow 框架，解决视频大语言模型在长视频推理中因视觉冗余导致的推理慢问题；

**💡 创新点**

创新点在于将视觉计算完全迁移到目标模型，通过隐藏状态重用+视觉文本锚定窗口注意力实现视觉信息的无损卸载，并结合中间层视觉状态桥接与多词预测解决分布偏移；

**🔧 技术方法**

采用隐藏状态重用（HSR）、视觉文本锚定窗口注意力（VATA）、中间层视觉状态桥接（IVSB）以及多词预测（MTP）等技术；

**📊 数据集**

在 VideoDetailCaption、MVBench、LongVideoBench 和 VideoMME 四大视频描述/生成数据集上进行评测；

**📈 对比分析**

与 MSD、ViSpec、SpecVLM 等基线对比，Sparrow 在 25k 视觉 token 长序列上平均推理加速达 2.82×（解码层面）和 1.96×（端到端），接受长度保持在 3.8 以上，显著优于传统方法；

**⚠️ 局限性**

局限在于预填充阶段仍存在显著延迟，长视频预处理的视觉 token 量大导致整体速度提升受限，未来需探索预填充加速技术。

---

## 287. GenAI for Systems: Recurring Challenges and Design Principles from Software to Silicon

**arXiv ID:** 2602.15241 | [PDF](https://arxiv.org/pdf/2602.15241v1)

**作者:** Arya Tschand `[一作]` (Harvard University), Vijay Janapa Reddi `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对生成式 AI 在软件、硬件架构与芯片设计三层的应用进行系统综述，挖掘并归纳出跨层共通的五大挑战与五大设计原则，并提出挑战–原则映射图供诊断与指导。

**💡 创新点**

创新点在于将原本分散的子领域聚合成跨层视角，发现并表征同一结构性难题在不同层面的表现与共通解法，构建挑战–原则关系图，为未来研究提供统一框架。

**🔧 技术方法**

主要采用文献调研与交叉层分析技术，收集并归纳 275+ 篇论文的实验、方法与案例，结合案例对比与原则映射构建分析模型。

**📊 数据集**

综述涵盖多种数据集与基准，包括软件层的 SWE‑bench、RepoBench、AgentBench；架构层的 ArchGym、TenSet、PerfCastDB；芯片层的 AutoChip、KernelBench、Concorde 等，覆盖 11 个应用领域。

**📈 对比分析**

通过跨层对比展示同一挑战在不同层的解决方案，评估原则在多层有效性；并以案例证明原则的适用性与性能提升，但未给出统一性能指标，主要关注原则有效性与案例表现。

**⚠️ 局限性**

限制在于缺乏统一的共享数据集、基准与评价方法，跨社区交流与复现仍不足，导致难以系统验证跨层方法的普适性与长期效果。

---

## 288. Rethinking Metrics for Lexical Semantic Change Detection

**arXiv ID:** 2602.15716 | [PDF](https://arxiv.org/pdf/2602.15716v1)

**作者:** Roksana Goworek `[一作]` (Queen Mary University of London), Haim Dubossarsky `[通讯]` (Queen Mary University of London)

**通讯引用:** 1064 | [OpenAlex ID](https://openalex.org/A5013663289)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并评估了两种基于词用例对应关系的语义变化度量——Average Minimum Distance (AMD) 与 Symmetric Average Minimum Distance (SAMD)，并在七种语言、不同编码器以及多种降维/解释空间中验证其鲁棒性。

**💡 创新点**

创新点在于：①引入局部对应度量（AMD/SAMD）替代传统的全局聚合指标（APD/PRT），能够捕捉局部意义变迁；②AMD支持方向性分解，SAMD实现一对一匹配，兼顾分布形状与对应质量；③在定义空间中对AMD进行解释性验证，展示了可解释性与性能的兼容。

**🔧 技术方法**

使用技术包括：上下文化语言模型（如XLM‑R、XL‑LEXEME、RoBERTa、BERT 等）生成词用例嵌入；余弦距离作为基础度量；AMD/SAMD 的最小邻居与贪心一对一匹配；PCA、定义空间（通过 Gemini 生成多义词定义）、随机维度选择等降维与结构化投影手段。

**📊 数据集**

实验数据集涵盖七种语言：英语、德语、瑞典语、拉丁语（SemEval‑2020 Task1）、西班牙语（LSCDiscovery）、挪威语（NorDiaChange）、中文（中文语义转移基准）。每个数据集提供时间段对、目标词及人工评定的语义变化分数。

**📈 对比分析**

与传统 APD 与 PRT 在全嵌入、定义空间、PCA、随机空间等四种空间中进行 Spearman 相关性比较。实验结果显示：AMD 与 SAMD 在多语言、多编码器环境下平均相关性明显优于 APD/PRT；在降维或非专用编码器下尤为稳健；SAMD 在 XL‑LEXEME（专用编码器）上取得最高平均相关性（约 0.694）。

**⚠️ 局限性**

局限性包括：未深入分析度量与嵌入几何/方差结构的机制；AMD/SAMD 仍输出标量，缺少对每个词实例的可解释性；实验仅考虑降维情形，未探讨噪声注入、域自适应、不同 LLM/提示对定义空间的影响；仅在现有评测集上验证，缺乏对大规模词表发现能力的评估。

---

## 289. Fine-Refine: Iterative Fine-grained Refinement for Mitigating Dialogue Hallucination

**arXiv ID:** 2602.15509 | [PDF](https://arxiv.org/pdf/2602.15509v1)

**作者:** Xiangyan Chen `[一作]` (Queen Mary University of London), Matthew Purver `[通讯]` (Queen Mary University of London)

**通讯引用:** 4312 | [OpenAlex ID](https://openalex.org/A5047970122)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Fine-Refine 框架，通过将对话回复拆解为原子信息单元并逐一验证，迭代纠错

**💡 创新点**

创新点是细粒度的原子单元拆分、事实与流畅度双重评估及反馈驱动的迭代修正

**🔧 技术方法**

使用 LLM 进行拆分、链式推理式事实校验、GPT‑2 困惑度评估流畅度以及多轮迭代修正

**📊 数据集**

在 HybriDialogue 与 OpendialKG 知识驱动对话数据集上评估（亦用 PersonaChat/Topical‑Chat 测量质量）

**📈 对比分析**

相较于 RAG、Self‑Refine 等基线，Fine‑Refine 在对话事实分数上提升最高 7.63 点，NEIP 降低，虽然自然度略降

**⚠️ 局限性**

主要限制是与对话自然度的权衡、计算开销增大以及在极长对话或缺失知识时仍易出现误差

---

## 290. Outer Diversity of Structured Domains

**arXiv ID:** 2602.15708 | [PDF](https://arxiv.org/pdf/2602.15708v1)

**作者:** Piotr Faliszewski `[一作]` (AGH University), Tomasz Wąs `[通讯]` (University of Oxford)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5083795172)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文提出了“外部多样性”指标，用以衡量选举域在整体偏好空间中的覆盖程度，并针对单峰、单交叉、组可分、欧几里得等结构化域设计了高效计算方法。

**💡 创新点**

创新点在于：①首次引入外部多样性概念并与内部多样性、丰富度多样性等传统度量对齐；②给出多种域的多项式或采样算法，克服了先前NP‑难的求解困难；③通过实验评估，发现基于树状结构的组可分域（特别是caterpillar树）在多样性上表现最优。

**🔧 技术方法**

技术方法包括：动态规划求最小交换距离、基于树结构的递归计算、采样估计平均距离、整数线性规划求k‑Kemeny（近似）以及模拟退火/阈值采样策略。

**📊 数据集**

数据集主要是理论生成的候选集（m=2…20）及其对应的结构化域，实验中还随机采样1000条全排列来估计外部多样性，未使用实际选举或真实偏好数据。

**📈 对比分析**

与传统的内部多样性（k‑Kemeny向量）和丰富度多样性进行对比，结果显示外部多样性与内部多样性排名高度一致；同时在相同域规模下，外部多样性能更直观地反映域的覆盖程度，实验显示组可分caterpillar域在外部多样性上明显优于单峰、单交叉和欧几里得域。

**⚠️ 局限性**

局限性包括：①对大规模候选集（m>20）仍需采样，可能产生估计误差；②部分域的最优外部多样性仍未解析，只能靠启发式搜索；③对于单峰-图域的距离求解仍为NP‑难，无法提供多项式算法。

---

## 291. DexEvolve: Evolutionary Optimization for Robust and Diverse Dexterous Grasp Synthesis

**arXiv ID:** 2602.15201 | [PDF](https://arxiv.org/pdf/2602.15201v1)

**作者:** René Zurbrügg `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**通讯引用:** 20354 | [OpenAlex ID](https://openalex.org/A5044258783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出DexEvolve生成-细化管道，在高保真仿真中使用无梯度进化优化抓取方案，产生多样、物理可行的手抓姿势。

**💡 创新点**

创新点在于将高保真模拟从单纯过滤转为持续优化，结合密度感知选择、档案插入以及可导向人类偏好的梯度无关进化；并把细化结果蒸馏为点云条件扩散模型。

**🔧 技术方法**

技术包括非梯度进化算法、Isaac Sim高保真仿真、密度感知选择、档案管理、人类偏好学习（PointNet++），以及扩散模型蒸馏。

**📊 数据集**

使用Handles数据集（90个IKEA门把手及其变体）和DexGraspNet子集作为评测对象。

**📈 对比分析**

与单纯分析生成、Diffusion模型等做对比，DexEvolve在Objects和Handles数据集上平均每个物体可获得120+稳定抓取姿势，成功率与独特抓取覆盖率比分析方法提升1.7‑6倍，优于Diffusion提升46‑60%；在训练生成的分布上实现高熵多模态。

**⚠️ 局限性**

局限在于仍需良好种子以保持多样性、对PhysX碰撞求解的依赖导致重入/分离开销、仿真到实物的端到端时间长。

---

## 292. A Content-Based Framework for Cybersecurity Refusal Decisions in Large Language Models

**arXiv ID:** 2602.15689 | [PDF](https://arxiv.org/pdf/2602.15689v1)

**作者:** Meirav Segal `[一作]` (University of Zurich), Omer Nevo `[通讯]` (Irregular)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于请求内容的网络安全拒绝策略框架，量化五个维度来平衡攻击风险与防御收益。

**💡 创新点**

创新点在于将拒绝决策从主题或意图迁移到技术细节层面，并引入“攻击行动贡献”“攻击风险”“技术复杂度”“防御收益”“合法用户频率”这五个可度量维度。

**🔧 技术方法**

采用专家驱动的标注、迭代误差分析和框架维度定义等技术，并用人工生成的提示语料库进行评估。

**📊 数据集**

使用安全从业者和AI安全专家生成的人工提示语料库（包含正面、双用途和恶意示例），未公开具体数据集名称。

**📈 对比分析**

与传统基于主题/意图的拒绝策略对比，显示在一致性、过度限制和误拒率方面的改进，尤其能在前缀变动等攻击手法下保持稳定决策。

**⚠️ 局限性**

局限性包括缺乏自动化标签工具、未考虑多轮对话聚合、依赖人工标注、对真实系统部署的可扩展性和鲁棒性尚未验证。

---

## 293. Certified Per-Instance Unlearning Using Individual Sensitivity Bounds

**arXiv ID:** 2602.15602 | [PDF](https://arxiv.org/pdf/2602.15602v1)

**作者:** Hanna Benarroch `[一作]` (École normale supérieure), Olivier Cappé `[通讯]` (École polytechnique)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种针对单个训练样本的可证实机学习忘却方法，通过在Langevin动态训练中注入噪声来实现对该样本的删除。

**💡 创新点**

创新点在于将差分隐私的个体敏感度概念引入忘却机制，并通过高概率非中心χ²分布估计个体梯度敏感度，从而显著降低所需噪声量；此外，还在Ridge回归中提供了严格的理论保证。

**🔧 技术方法**

所采用技术包括Langevin梯度下降、Gaussian Differential Privacy轨迹跟踪、收敛性（强凸、光滑）条件下的收敛放缩、以及对个体敏感度的高概率上界估计和自适应噪声校准。

**📊 数据集**

实验使用MNIST（线性岭回归头）和CIFAR‑10（VGG/卷积网络）等公开数据集。

**📈 对比分析**

与使用全局最坏情况敏感度的统一噪声基线相比，个体化噪声校准在相同隐私预算下显著提升测试准确率；在非线性模型实验中，也能观察到不同样本对应的隐私保障差异。

**⚠️ 局限性**

局限性包括：对目标函数的收敛与收敛率假设（强凸、光滑）有严格依赖，仅在线性或固定特征头的情形下有理论保证；在非凸深度网络中缺乏完整的理论证明，隐私评估只能通过经验方法完成。

---

## 294. Pairwise XOR and XNOR Gates in Squeezed Instantaneous Noise Based Logic

**arXiv ID:** 2602.15032 | [PDF](https://arxiv.org/pdf/2602.15032v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 295. Directional Reasoning Trajectory Change (DRTC): Identifying Critical Trace Segments in Reasoning Models

**arXiv ID:** 2602.15332 | [PDF](https://arxiv.org/pdf/2602.15332v1)

**作者:** Waldemar Chang `[一作]` `[通讯]` (Johns Hopkins University), Waldemar Chang (Johns Hopkins University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种针对单条生成轨迹的过程因果解释框架——Directional Reasoning Trajectory Change (DRTC)，通过识别推理过程中的关键决策点并施加接收端干预，衡量早期上下文如何在这些关键点上引导模型的推理轨迹；

**💡 创新点**

创新点在于：①利用不确定性与分布偏移信号定位决策点；②在不重采样的情况下通过接收端注意力屏蔽实现时序有效的因果干预；③以轨迹方向变化为因果目标，输出带符号的单块归因分数；④使用曲率诊断来补充几何视角，形成曲率签名和角色路径；

**🔧 技术方法**

使用的技术包括：熵、top‑2 margin、Jensen–Shannon 散度等不确定性/偏移指标，receiver‑side attention masking，log‑probability 轨迹方向计算，曲率（转角）分析，Gini 系数、Spearman 相关等统计量；

**📊 数据集**

评估数据集包括：4 种不同模型在 24 条数学/规划推理示例上的单条生成轨迹，以及对 500 条 MATH 题目进行的规模化实验；

**📈 对比分析**

与随机跨度控制、统一权重等基线比较，发现 DRTC 的归因在多模型上呈现高度集中（Gini ≈0.50–0.58，前5%块占 23–28% 影响），并且学习到的决策点相比随机跨度产生更大的方向性干预（中位差 0.039–0.178，500 题目 p<2.3e-21）；

**⚠️ 局限性**

局限性包括：只衡量轨迹方向变化，未直接评估对最终答案的因果影响；使用固定步长块和固定决策点数，可能缺乏自适应粒度；曲率仅作为诊断工具，未识别实现机制；缺乏跨域和不同解码策略的广泛验证；

---

## 296. Equilibria in Large Position-Optimization Games

**arXiv ID:** 2602.15225 | [PDF](https://arxiv.org/pdf/2602.15225v1)

**作者:** Rafael Frongillo `[一作]` (University of Colorado Boulder), Anish Thilagar `[通讯]` (University of Colorado Boulder)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5030779531)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并分析一种统一的“位置优化游戏”框架，证明在玩家数量足够大时纯策略和对称混合策略均存在极值均衡，并且这些均衡的玩家位置分布以 O(1/n) 的速度收敛到目标分布在伪目标上的投影。

**💡 创新点**

创新点在于：①将 Hotelling、预测竞赛、Voronoi 等多种经典游戏归纳到一个通用模型；②首次给出大 n 情况下均衡的收敛速度；③提供对离散消费者分布的经典 Hotelling 游戏的完整理论描述；④通过构造性算法和概率论工具证明极值均衡的存在。

**🔧 技术方法**

使用了博弈论中的对称常数和求和法则、极值策略构造算法、几何概率（如优惠收集问题）、KL 散度和对数不等式的分析、以及对称混合策略的收益拆解等技术。

**📊 数据集**

本研究为理论性工作，未使用实际数据集；所有结论均通过数学证明得到。

**📈 对比分析**

由于缺乏实验验证，本研究不做方法对比；理论上证明了均衡收敛速率为 O(1/n)，并给出了精确的上界与下界。

**⚠️ 局限性**

局限性包括：①要求无平局且伪目标集合有限，限制了对连续目标分布或存在平局的情形；②结论仅适用于玩家数量足够大（n > 1/ 等阈值）时；③对小 n 或特殊结构游戏的行为仍未被覆盖。

---

## 297. Continuous-Time Piecewise-Linear Recurrent Neural Networks

**arXiv ID:** 2602.15649 | [PDF](https://arxiv.org/pdf/2602.15649v1)

**作者:** Alena Brändle `[一作]` (Central Institute of Mental Health), Daniel Durstewitz `[通讯]` (Central Institute of Mental Health)

**通讯引用:** 7974 | [OpenAlex ID](https://openalex.org/A5056788018)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出连续时间分段线性递归神经网络（cPLRNN）及其高效训练与求解算法，利用分段线性结构在每个子区域内得到解析解，避免数值积分，支持不规则采样；并开发了计算稳态点、极限环等拓扑特征的半解析方法。

**💡 创新点**

创新点在于：① 将分段线性RNN迁移到连续时间，利用解析解与根寻找快速计算切换时间；② 通过隐函数定理实现梯度反向传播；③ 通过SCYFI、Trust-Region + Variable Projection等工具半解析计算稳态点与极限环；④ 训练更快、对不规则采样更友好，并保持与离散PLRNN相当的重建性能。

**🔧 技术方法**

核心技术包括：分段线性ReLU RNN、解析线性ODE求解、区间根寻找算法、隐函数定理梯度、SCYFI（稳态点搜索）、Trust-Region + Variable Projection（极限环搜索）、稀疏教师强制（STF）。

**📊 数据集**

使用的数据集有：Lorenz‑63混沌系统、LIF（leaky‑integrate‑and‑fire）模型的定期与不规则采样时序，以及真实皮层神经元膜电位记录。

**📈 对比分析**

对比方法：Neural ODE（Euler、RK4、Tsit5）与离散时间PLRNN；评价指标包括几何相似度D_stsp、时序相似度D_H、短期MAE；结果显示cPLRNN与最佳Neural ODE解算器（Tsit5）性能相当，且在不规则采样下优于离散PLRNN；训练时间显著短于Neural ODE，接近离散PLRNN。

**⚠️ 局限性**

局限性：训练过程对数值稳定性敏感，易受特征值分解的病态影响；根寻找需要在每个切换事件后重启，导致计算量增大；在长时间训练或高度切换的模型中可能出现数值不稳定或收敛失败。

---

## 298. Synthesizing Trajectory Queries from Examples

**arXiv ID:** 2602.15164 | [PDF](https://arxiv.org/pdf/2602.15164v1)

**作者:** Stephen Mell `[一作]` (University of Pennsylvania), Osbert Bastani `[通讯]` (University of Pennsylvania)

**通讯引用:** 2693 | [OpenAlex ID](https://openalex.org/A5029243071)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于示例自动合成视频轨迹查询的框架，能够在轨迹数据中识别复杂驾驶或动物行为。

**💡 创新点**

核心创新在于引入量化语义实现参数空间裁剪，显著加速了参数搜索，并通过多对象序列谓词扩展了查询表达能力。

**🔧 技术方法**

技术上采用语法指导枚举搜索、参数空间裁剪、量化语义、GPU并行化以及主动学习策略来高效完成查询合成。

**📊 数据集**

使用了三类数据集：YTStreams（交通摄像头轨迹）、MABe22（实验室老鼠交互轨迹）和合成海事监测任务。

**📈 对比分析**

与主动学习消除、LSTM、Transformer 基线相比，实验显示F1得分在10步后已超过0.99，量化语义相较于传统二分搜索提升约5×，GPU进一步加速数倍。

**⚠️ 局限性**

局限性包括：只能枚举最多三谓词、两参数的查询；需要手工定义谓词集合；高维参数空间仍可能导致搜索耗时；对极其复杂或多维场景的可扩展性待验证。

---

## 299. GRAM-DIFF: Gram Matrix Guided Diffusion for MIMO Channel Estimation

**arXiv ID:** 2602.15187 | [PDF](https://arxiv.org/pdf/2602.15187v1)

**作者:** Xinyuan Wang `[一作]` (Texas A&M University), Krishna Narayanan `[通讯]` (Texas A&M University)

**通讯引用:** 7980 | [OpenAlex ID](https://openalex.org/A5071834465)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文提出一种基于预训练角域扩散模型的GRAM‑DIFF框架，用于半盲MIMO信道估计，结合了Gram矩阵指导和先验导向的似然引导，能够在仅有极少数据符号的情况下实现高精度估计。

**💡 创新点**

创新点在于首次将可从数据符号估计得到的Gram矩阵作为二阶统计信息融入扩散反向过程，并设计了SNR自适应的初始化与自适应引导权重，以实现鲁棒的低延迟估计。

**🔧 技术方法**

采用的技术包括基于角域的二维离散傅里叶变换、变分扩散模型（DDIM）、先验导向的似然梯度以及Gram矩阵一致性梯度，并对梯度进行剪枝以保证数值稳定。

**📊 数据集**

实验使用了3GPP和QuaDRiGa两个标准化无线信道模型，分别构造大规模MIMO（64×16）场景的真实信道数据。

**📈 对比分析**

与传统的DM、DM+Likelihood、DM+Gram以及LMMSE基准相比，GRAM‑DIFF在-15~5 dB的SNR范围内实现了4–6 dB的SNR提升，在大多数条件下甚至逼近或超越基于协方差的Genie‑LMMSE，且在Gram估计不足时仍保持优于无引导扩散。

**⚠️ 局限性**

局限性包括对Gram估计误差的敏感性，需要通过调节引导强度来避免不稳定；此外，Gram引导的计算复杂度随接收天线数平方增长，且在极低SNR或极短数据块时仍需进一步优化引导策略。

---

## 300. Supporting Multimodal Data Interaction on Refreshable Tactile Displays: An Architecture to Combine Touch and Conversational AI

**arXiv ID:** 2602.15280 | [PDF](https://arxiv.org/pdf/2602.15280v1)

**作者:** Samuel Reinders `[一作]` (Monash University), Kim Marriott `[通讯]` (Monash University)

**通讯引用:** 8169 | [OpenAlex ID](https://openalex.org/A5085695563)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个将多指触控输入与对话式 AI（GPT‑4o）结合在刷新式触觉显示器（Dot Pad）上的数据可视化系统，并实现了图表渲染、语音识别、触觉高亮、盲文标签等多模态交互。

**💡 创新点**

首创在 RTD 上支持多指触控 + 对话式 AI 的 deictic（指向式）查询；将外部手部跟踪、语义缩放、实时多模态同步整合成完整体系；开放源码实现为研究基础。

**🔧 技术方法**

使用 Dot Pad RTD、Ultraleap Leap Motion Controller、Unity 引擎、MQTT、Picovoice Porcupine、Google Cloud Speech‑to‑Text/TT‑S、OpenAI GPT‑4o、LangChain、Vega‑Lite 及自研的 Vega‑Lite‑to‑RTD 渲染器。

**📊 数据集**

主要以公开的 Vega‑Lite JSON 图表规格为输入，示例中使用了利率时间序列图；未使用特定机器学习数据集。

**📈 对比分析**

系统通过三轮与三位 BLV 共同设计工作坊验证可行性，未提供量化性能指标；功能验证显示触觉高亮、盲文标签与语音描述能同步给出。

**⚠️ 局限性**

局限包括需外部手部跟踪器导致携带不便、仅支持 Dot Pad RTD、触控识别依赖手势模型、图表类型受限、缺乏正式用户研究与性能评估。

---

## 301. A Generative-First Neural Audio Autoencoder

**arXiv ID:** 2602.15749 | [PDF](https://arxiv.org/pdf/2602.15749v1)

**作者:** Jonah Casebeer `[一作]` (Adobe Research), Nicholas J. Bryan `[通讯]` (Adobe Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计了面向生成模型的通用音频自编码器GenAE，实现低率、快速编码且支持连续与离散潜在及多声道格式。

**💡 创新点**

采用生成优先的架构改进，包括早期下采样、可分离卷积、激活函数优化、窗口自注意力及多格式条件，实现10倍编码加速、1.6倍压缩率下降。

**🔧 技术方法**

结合SnakeLite激活、可分离卷积、Mel谱融合、窗口自注意力、联合格式令牌、后训练RVQ量化等技术。

**📊 数据集**

使用25K小时44.1kHz乐器立体声音乐数据集，去除人声，仅用于训练与评估。

**📈 对比分析**

与DAC、EnCodec、Stable Audio Open、CoDiCodec等基线在SI‑SDR、STFT、Mel L1、PESQ‑WB等指标上对比，GenAE在13.125Hz时取得更低token数、更高压缩率且质量相当，速度提升12倍。

**⚠️ 局限性**

主要局限在高频细节恢复受限、训练成本高，且对非乐器/非44.1kHz音频的适应性尚未充分验证。

---

## 302. CircuChain: Disentangling Competence and Compliance in LLM Circuit Analysis

**arXiv ID:** 2602.15037 | [PDF](https://arxiv.org/pdf/2602.15037v1)

**作者:** Mayank Ravishankara `[一作]` `[通讯]` (Independent Researcher), Mayank Ravishankara (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 CircuChain benchmark，设计 Control/Trap 对，分离电路分析中的物理推理能力与指令遵从性。

**💡 创新点**

创新点在于通过对抗性陷阱任务、三阶段验证管道以及错误分类词典，系统区分物理错误与规范偏差，揭示规模化模型的合规‑能力权衡。

**🔧 技术方法**

使用符号求解器、NGSPICE 仿真、LLM 判别器进行错误归因，并采用统一的 Prompt 模板与链式思考技术。

**📊 数据集**

基准数据集包含 5 种经典电路拓扑的 50 个手工设计实例，每个实例生成 Control 与 Trap 两个版本，总共 100 对（200 题）及对应的 NGSPICE netlist 与双重验证结果。

**📈 对比分析**

对比评估 5 款最先进 LLM（GPT‑5、Claude Opus 4.5、o4‑mini、GPT‑4o、GPT‑4o Mini）在 100 题上的整体准确率、合规误差与能力误差，发现 GPT‑5 最高准确率但合规错误率最高，Claude Opus 4.5 合规错误低但能力错误较多，验证了合规‑能力互斥。

**⚠️ 局限性**

局限性包括仅覆盖线性 DC 电路、文本输入、对判别器的依赖（标注一致性有限）、未扩展至 AC/相位分析、运算放大器或图像输入等更复杂场景。

---

## 303. Lyapunov-Based $\mathcal{L}_2$-Stable PI-Like Control of a Four-Wheel Independently Driven and Steered Robot

**arXiv ID:** 2602.15424 | [PDF](https://arxiv.org/pdf/2602.15424v1)

**作者:** Branimir Ćaran `[一作]` (University of Zagreb), Bojan Jerbić `[通讯]` (Croatian Academy of Sciences and Arts)

**通讯引用:** 641 | [OpenAlex ID](https://openalex.org/A5039486428)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出一种基于Lyapunov函数的PI‑类似控制器，能够对四轮独立驱动与转向机器人实现鲁棒轨迹跟踪，满足实时时间尺度的实现与稳定性保证。

**💡 创新点**

创新点：①在速度空间推导结构化的动力学模型，直接满足正定性与对称性；②构造显式的ℒ₂稳定性判据，给出可行的增益设计规则；③将PI控制与前馈补偿结合，保留传统PI易实现的形式，同时提供严格的能量（功率）界限；④在垂直表面、重力与摩擦不确定性下完成实验验证，验证了理论的实用性。

**🔧 技术方法**

技术：结构化动力学建模、Lyapunov能量函数构造、ℒ₂增益鲁棒分析、PI+前馈闭环设计、仿真与实验平台集成（ROS 2、Dynamixel驱动器、Duct‑fan辅助黏附）。

**📊 数据集**

数据集：无公开大规模数据集，实验使用两条轨迹——平面花形轨迹（带外部推力扰动）和垂直面Lissajous轨迹（重力与黏附作用）。

**📈 对比分析**

对比方式：将实验中获得的速度误差和姿态误差与理论给出的ℒ₂增益界限进行对照；结果显示误差始终在理论预测的能量边界内，且在不同扰动场景下保持稳定，验证了控制器的鲁棒性与性能。相比传统PI或基于阻尼的控制方案，本文控制器在同样误差预算下实现了更低的扭矩峰值与更好的抗扰动特性。

**⚠️ 局限性**

局限性：①理论假设为无内在动力学的瞬时不确定性，实际摩擦与接触可能包含延迟和非线性；②增益设计基于保守上界，可能导致在最优性能与鲁棒性之间的折衷；③实验验证仅在特定硬件平台与两条轨迹上完成，缺乏对更复杂场景（如障碍导航、三维运动）的评估。

---

## 304. DNN-Enabled Multi-User Beamforming for Throughput Maximization under Adjustable Fairness

**arXiv ID:** 2602.15617 | [PDF](https://arxiv.org/pdf/2602.15617v1)

**作者:** Kaifeng Lu `[一作]` (Institute of Telecommunications, Technische Universität Wien), Stefan Schwarz `[通讯]` (Institute of Telecommunications, Technische Universität Wien)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种基于WiT Transformer的无监督学习框架，用于在多用户MIMO系统中在可调节公平性约束下最大化吞吐量。

**💡 创新点**

创新点在于：①将公平性约束与吞吐量目标通过拉格朗日乘子和hinge损失联合建模；②采用自适应双重上升（dual‑ascent）算法自动更新拉格朗日乘子，使模型在满足预设Jain指数下实现吞吐最大化；③通过设置不同的Jain下界训练得到多条Pareto前沿点，提供可调节的公平-吞吐权衡。

**🔧 技术方法**

技术手段包括：WiT（Wireless Transformer）架构、CSI归一化与特征提取、Transformer编码器与多头注意力、FC回归输出、Max‑Min归一化的sum‑rate正则化、hinge‑loss与自适应双重上升。

**📊 数据集**

使用合成Rayleigh衰落信道生成的随机用户位置和通道样本，共计50,000个样本，划分为32,000训练集、8,000验证集、10,000测试集。

**📈 对比分析**

通过与wSLNR（多参数α）以及MRT、ZF、SLNR基线的对比，评估了平均sum‑rate、Jain指数、ECDF分布和按用户排序的速率。实验表明，在公平性要求较高（J≥0.9）时，DNN模型显著优于wSLNR，尤其在强用户的吞吐上提升显著；在低公平性要求下，两者性能相近。

**⚠️ 局限性**

局限性包括：①求解仅在所选场景下（单小区、单子载波）验证，未考虑多小区、多子载波和多任务场景；②拉格朗日乘子对收敛速度和精度敏感，需手动调节双重步长与容差；③目前仅针对Jain指数公平性，未涵盖能效或功耗等其他约束。

---

## 305. Scaling Laws for Masked-Reconstruction Transformers on Single-Cell Transcriptomics

**arXiv ID:** 2602.15253 | [PDF](https://arxiv.org/pdf/2602.15253v1)

**作者:** Ihor Kendiukhov `[一作]` `[通讯]` (University of Tuebingen), Ihor Kendiukhov (University of Tuebingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了单细胞RNA测序数据上掩码重建Transformer的神经缩放规律，系统评估不同模型规模与数据量的关系

**💡 创新点**

首次在单细胞转录组学中证明了在足够数据条件下可出现幂律缩放；同时给出了不可约损失阈值与信息熵的初步估计

**🔧 技术方法**

使用掩码重建预训练、Permutation‑Invariant Transformer编码器、均方误差损失，并通过参数化缩放公式拟合

**📊 数据集**

利用CELLxGENE Census中精选的高度可变基因表达矩阵，构建两种实验情形（512基因/200k细胞和1024基因/10k细胞）

**📈 对比分析**

将验证MSE与参数数的对数关系拟合，数据丰富情形得到α≈0.26、R²≈0.86、不可约损失c≈1.44（≈2.3bits）；数据稀缺情形几乎无缩放，α≈0.009、R²≈0.02

**⚠️ 局限性**

实验受限于仅同时变化基因数与细胞数的混合设计、MSE作为损失导致熵估计不够独立、缺乏计算与数据规模的完整缩放分析

---

## 306. Panini: Continual Learning in Token Space via Structured Memory

**arXiv ID:** 2602.15156 | [PDF](https://arxiv.org/pdf/2602.15156v1)

**作者:** Shreyas Rajesh `[一作]` (University of California, Los Angeles), Vwani Roychowdhury `[通讯]` (University of California, Los Angeles)

**通讯引用:** 9974 | [OpenAlex ID](https://openalex.org/A5043479061)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个非参数持续学习框架 Panini，通过将每份文档转换为生成语义工作空间（GSW）来实现外部结构化记忆，并在推理时使用链式检索（RICR）完成多跳问答。

**💡 创新点**

创新点在于：① 将文档转化为可直接用于推理的 QA 对网络（GSW），② 设计了只需一次解构的轻量级链式检索算法 RICR，③ 在写入时进行结构化编码而非仅存储文本块，从而在读取时大幅减少计算量并提升可靠性。

**🔧 技术方法**

主要技术包括：GPT 系列大型语言模型用于 GSW 生成与 QA 对检索，BM25 与 dense 向量检索结合的双索引机制，几何平均评分的 beam‑search 链检索，及 OpenAI 的 GPT‑4o‑mini/其他开源模型进行答案生成。

**📊 数据集**

使用六个 QA 评测集（NQ、PopQA、MuSiQue、2WikiMultihopQA、HotpotQA、LV‑Eval）以及专门构造的 Platinum 评测集（MuSiQue‑Platinum、2Wiki‑Platinum）来检验支持性、效率和可靠性。

**📈 对比分析**

与基准方法（传统 RAG、结构增强 RAG、agentic 多步检索等）相比，Panini 在平均 F1 上提升至 56.1%（高于 HippoRAG2 的 53.3% 和 dense 检索的 50.5%），在多跳任务上更显著；在推理时的 token 使用量比 Chunk‑RAG 降低 2–30 倍；在 Platinum 评测中回答正确率 79.8% 兼具 74.0% 的拒绝准确率，优于其他方法。

**⚠️ 局限性**

主要局限包括：① 写入时的 GSW 生成依赖高成本的专有模型，开源模型易出现错误；② 目前未实现跨文档链接的自适应缓存或经验驱动的重合；③ 仅适用于文本事实问答，对叙事性或多模态任务的扩展尚未完成。

---

## 307. Decomposing Docker Container Startup Performance: A Three-Tier Measurement Study on Heterogeneous Infrastructure

**arXiv ID:** 2602.15214 | [PDF](https://arxiv.org/pdf/2602.15214v1)

**作者:** Shamsher Khan `[一作]` `[通讯]` (GlobalLogic), Shamsher Khan (GlobalLogic)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统地测量了 Docker 容器启动延迟，并将启动过程拆解为内核操作、运行时准备和存储 I/O 三个组成部分，比较了 Azure Premium SSD、Standard HDD 以及 macOS Docker Desktop 三种基础设施。

**💡 创新点**

首个针对容器启动的细粒度分解研究；首次量化 Docker Desktop 虚拟化层的 2.69× 启动惩罚与 9.5× CPU 调度方差；揭示 OverlayFS 写性能极差而元数据操作反而快的“反直觉”现象；公开了可复现的基准套件。

**🔧 技术方法**

使用 Bash 脚本采集纳秒级时间戳，Docker Engine（overlay2/overlayfs），Azure 虚拟机（Premium SSD、Standard HDD），macOS Docker Desktop（LinuxKit VM），以及非参数统计方法（Mann‑Whitney、Cliff's delta、95% 置信区间）。

**📊 数据集**

三张不同尺寸的镜像（5 MB alpine、67 MB nginx、155 MB python），每个测试 50 次迭代，覆盖 10 个性能维度。

**📈 对比分析**

通过 50 次迭代、95% 置信区间和非参数显著性检验比较不同基础设施。结果显示：SSD 相比 HDD 延迟提升 2.04×；Docker Desktop 相比原生 Linux 延迟提升 2.69×；OverlayFS 写吞吐仅为卷挂载的 0.006–0.01×，但元数据创建 1.3–4.8×更快；镜像大小对启动延迟影响仅 2.5%。

**⚠️ 局限性**

仅测试单一 VM 尺寸与三张镜像；仅进行顺序执行，未考虑并发调度；macOS cold‑start 受限于 Docker Desktop VM 缓存；未覆盖其他容器运行时、不同 cgroup 版本或更多存储层。

---

## 308. Lifelong Scalable Multi-Agent Realistic Testbed and A Comprehensive Study on Design Choices in Lifelong AGV Fleet Management Systems

**arXiv ID:** 2602.15721 | [PDF](https://arxiv.org/pdf/2602.15721v1)

**作者:** Jingtian Yan `[一作]` (Robotics Institute, Carnegie Mellon University), Jiaoyang Li `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

开发了一个名为LSMART的开源仿真平台，用于在考虑动力学约束、通信延迟和执行不确定性的AGV车队管理系统中评估任何多智能体路径规划（MAPF）算法。

**💡 创新点**

创新点在于：①将MAPF研究中的核心设计决策（规划器、实例生成器、规划调用策略、失败恢复策略）拆分为可自定义模块；②提供了针对终身MAPF的完整实验框架；③对多种主流策略在现实仿真中的性能进行系统比较，揭示传统假设与实际环境的差距。

**🔧 技术方法**

技术实现包括：基于物理引擎的AGV动力学模型、Action Dependency Graph (ADG)执行策略、窗口化PBS、MAPF4L、CBS、PP、PIBT和Guided PIBT等规划/恢复算法；使用Python/ROS接口将规划器与仿真器耦合。

**📊 数据集**

使用六张典型地图（包含两个仓库地图和四张来自MAPF基准库的网格地图），并在每张地图上测试从数十到两千台AGV的规模，覆盖不同密度和拓扑。

**📈 对比分析**

通过对比窗口化规划与周期性规划、不同的调用频率、不同的失败恢复策略（PIBT、Guided PIBT、全停等），以及不同的模型精度和最优性（pebble vs rotation，CBS vs PP），实验表明：窗口化规划+策略调用能获得最高吞吐量，窗口大小与调用频率的平衡对高密度场景尤为关键；更精确的动力学模型和更优的规划器虽提升吞吐，但会显著增加计算时间。

**⚠️ 局限性**

局限性包括：仅支持4连通网格拓扑；仅实现了差速驱动和旋转运动模型，无法覆盖更复杂的机器人动力学；未对非正方形网格或非结构化场景进行验证；实际部署时对通信延迟和执行噪声的建模仍需进一步细化。

---

## 309. SEG-JPEG: Simple Visual Semantic Communications for Remote Operation of Automated Vehicles over Unreliable Wireless Networks

**arXiv ID:** 2602.15258 | [PDF](https://arxiv.org/pdf/2602.15258v1)

**作者:** Sebastian Donnelly `[一作]` (Oxford Brookes University), Andrew Bradley `[通讯]` (Oxford Brookes University)

**通讯引用:** 15235 | [OpenAlex ID](https://openalex.org/A5042053820)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于语义分层的 JPEG（SEG‑JPEG）技术，用于在不可靠无线网络上远程操作自动驾驶车辆，显著降低所需数据速率并提升情境感知；

**💡 创新点**

创新点在于将计算机视觉检测得到的道路使用者分割掩码嵌入到低质量灰度 JPEG 中，形成语义化压缩方案，实现50% 的带宽节省，同时保持可解释的视觉信息；

**🔧 技术方法**

采用 YOLO11x‑seg 进行目标检测与分割，利用灰度 JPEG 压缩，结合预设色调重彩与 UDP/ZeroTier 传输；

**📊 数据集**

使用 Oxford RobotCar 数据集验证 H.264 的失真表现，并在 Oxford Brookes 校园收集的现场数据（Zed2i 双目摄像头与 Arducam 组合）进行实验；

**📈 对比分析**

通过与传统 H.264（30 fps、1 Mbit/s）对比，在同等 500 kbit/s 带宽下，SEG‑JPEG 取得 200 ms 左右的玻璃对玻璃延迟，且图像失真更低，适用于 4G 信号弱区；

**⚠️ 局限性**

局限性包括仅在 4G 环境下验证、对 5G/6G 网络缺乏评估、需要 GPU 计算进行 YOLO 推理、只覆盖了人、单车和车辆等少数类别，且对其他重要信息的语义编码仍有限。

---

## 310. Hennessy-Milner Logic in CSLib, the Lean Computer Science Library

**arXiv ID:** 2602.15409 | [PDF](https://arxiv.org/pdf/2602.15409v1)

**作者:** Fabrizio Montesi `[一作]` (University of Southern Denmark), Alexandre Rademaker `[通讯]` (Getulio Vargas Foundation)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在 Lean 4 中实现了 Hennessy–Milner 逻辑（HML）的库级形式化，包括语法、满足关系、解释语义和完整的元理论。

**💡 创新点**

提出了宇宙多态的泛化定义、完整的元理论证明（包括 Hennessy–Milner 定理）以及与 Lean 生态系统高度集成的可重用 API，显著提高了形式化的通用性和可维护性。

**🔧 技术方法**

使用 Lean 4 证明助手、Mathlib 以及 Lean 自带的 `grind` 等自动化工具；实现参数化的 LTS 结构和可插拔的证明策略。

**📊 数据集**

未使用外部数据集；该工作为理论形式化而非数据驱动实验。

**📈 对比分析**

与现有的 Isabelle/HOL、Coq 等形式化对比，显示出更高的抽象度与重用性；在 Lean 4.28.0-rc1 上通过编译和证明，性能主要体现在编译时间和自动化证明成功率，暂无具体数值。

**⚠️ 局限性**

仅适用于无负载标签的经典 HML；不支持带绑定变量的高级语义扩展（如 π‑calculus）；对弱等价、修订算子等更复杂特性尚未覆盖，需要进一步工作。

---

## 311. Onto-DP: Constructing Neighborhoods for Differential Privacy on Ontological Databases

**arXiv ID:** 2602.15614 | [PDF](https://arxiv.org/pdf/2602.15614v1)

**作者:** Yasmine Hayder `[一作]` (Institute National des Sciences Appliquées Centre Val de Loire), Benjamin Nguyen `[通讯]` (Institute National des Sciences Appliquées Centre Val de Loire)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在含有推理规则的语义数据库（如知识图谱）中，传统差分隐私（DP）机制对“语义感知攻击者”的保护不足，并提出了基于本体推理的Onto‑DP（(I,d_b)-DP）扩展，通过构造新的邻域距离，使得DP机制能够正确评估敏感查询的全局灵敏度，避免隐私泄露。

**💡 创新点**

创新点：①首次将推理规则（本体知识）纳入DP邻域定义；②提出“语义感知攻击者”模型，阐明传统DP在此类攻击下的匹配不良；③引入(I,d_b)-DP距离，并证明其对该攻击者类别是well‑suited的；④通过示例展示传统DP灵敏度为0而实际泄露信息的“匹配失效”问题。

**🔧 技术方法**

主要技术：差分隐私理论、邻域距离构造、推理系统（如RDFS/SWRL + 归约器）、语义攻击者模型、对称邻域（paired distances）理论、实验可视化工具。

**📊 数据集**

数据集：使用了一个示例医院知识图谱（包含医生、病人、科室等实体和关系），并通过人工构造的推理规则（如“若医生在某科室工作且有病人，则病人属于该科室”）进行演示；未给出大规模公开数据集，实验主要为案例演示。

**📈 对比分析**

方法比较：本文并未进行大规模基准实验，而是通过理论证明与案例演示展示传统DP与Onto‑DP的差异。性能方面，作者指出推理导致邻域规模可能膨胀，隐私预算可能变大，但具体数值未给出，需要后续实验评估。

**⚠️ 局限性**

局限性：①实验规模有限，仅基于小型示例图谱；②推理规则的复杂度和可执行性未系统评估，可能导致邻域计算复杂度高；③仅针对数值查询，非数值查询的处理仍待扩展；④实际部署中如何高效计算(I,d_b)邻域仍是研究挑战。

---

## 312. World-Model-Augmented Web Agents with Action Correction

**arXiv ID:** 2602.15384 | [PDF](https://arxiv.org/pdf/2602.15384v1)

**作者:** Zhouzhou Shen `[一作]` (Zhejiang University), Shengyu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 16745 | [OpenAlex ID](https://openalex.org/A5008666077)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于世界模型增强的 Web 代理（WMAAC），通过多模型协作生成行动并在执行前进行风险评估与修正

**💡 创新点**

创新点在于：①基于需求的世界模型协作，突破单模型认知瓶颈；②采用世界模型预演与评判模型闭环反馈，实现可修正的预执行行动决策；③实现了按需协作与反馈驱动的迭代优化

**🔧 技术方法**

主要技术包括大型语言模型（LLM）、专门的世界模型用于环境状态转移模拟、判别模型（Judge）评估模拟结果、路由器（Router）决定协作触发、以及多模型协同推理与反馈循环

**📊 数据集**

在 VisualWebArena（多模态 Web 任务）和 Online-Mind2Web（真实网站在线任务）两大基准数据集上进行评估

**📈 对比分析**

与 ReAct、WebDreamer 等基线相比，WMAAC 在 VisualWebArena 取得 24.5% 的成功率（比基线提升 1.8%），在 Online-Mind2Web 取得 16.0%（比基线提升 1.3%），整体表现优于同类方法

**⚠️ 局限性**

局限性包括：依赖底层 LLM 的知识与推理能力，世界模型模拟不完美导致反馈误差，且目前仍为无调优的通用方案，可能在极端动态环境中表现欠佳

---

## 313. "The Intangible Victory", Interactive Audiovisual Installation

**arXiv ID:** 2602.15071 | [PDF](https://arxiv.org/pdf/2602.15071v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106`

---

## 314. The Stationarity Bias: Stratified Stress-Testing for Time-Series Imputation in Regulated Dynamical Systems

**arXiv ID:** 2602.15637 | [PDF](https://arxiv.org/pdf/2602.15637v1)

**作者:** Amirreza Dolatpour Fathkouhi `[一作]` (University of Virginia), Heman Shakeri `[通讯]` (University of Virginia)

**通讯引用:** 423 | [OpenAlex ID](https://openalex.org/A5006445265)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并评估了时间序列缺失填补中的“Stationarity Bias”，并提出基于稳态与过渡区分的分层评估与模型选择策略。

**💡 创新点**

发现并量化统一随机掩码导致的“Stationarity Bias”与“RMSE Mirage”，引入分层压力测试、形态度量DTW、分布校准，并提出适用稳态时线性插值、过渡时深度学习的自适应推理框架。

**🔧 技术方法**

采用Transformer（SAITS）、生成式与频域模型（FreTS、SCINet等）、动态时间规整DTW、分布校准、模拟缺失模式、临床实验缺失分布等技术。

**📊 数据集**

使用实际 CGM 数据（DCLP3/5、PEDAP、UVA/Padova 模拟）以及 TCR‑Simulation 数据，并从这些数据中提取真实缺失分布。

**📈 对比分析**

与线性插值、LOCF、传统机器学习（RF 等）进行基准，使用 RMSE、MARD、DTW、偏差等度量；在线性区间线性插值最优，过渡区间深度学习（SAITS）在形态、临床安全和 MARD 上明显优于基线。

**⚠️ 局限性**

仅评估了部分模型，未覆盖所有经典/先进架构；在分布校准上仍存在偏差；缺失对实时闭环控制的实际影响未完全验证。

---

## 315. VideoSketcher: Video Models Prior Enable Versatile Sequential Sketch Generation

**arXiv ID:** 2602.15819 | [PDF](https://arxiv.org/pdf/2602.15819v1)

**作者:** Hui Ren `[一作]` (University of Illinois Urbana-Champaign), Yael Vinker `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 764 | [OpenAlex ID](https://openalex.org/A5031567351)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用预训练的文本到视频扩散模型和大型语言模型，提出一种数据高效的两阶段微调框架，实现文本条件下的顺序草图生成，并支持笔刷风格控制和自回归交互式绘制。

**💡 创新点**

创新点在于：①将草图视为短视频序列，让视频扩散模型承担高质量渲染任务；②通过两阶段微调将绘制语法（形状组合与顺序）与视觉风格分离，只需少量人绘制样本；③在此基础上实现笔刷视觉提示、实时自回归绘制和协作绘图。

**🔧 技术方法**

核心技术包括：文本到视频扩散模型（Wan 2.1），大型语言模型用于生成绘制步骤，二维图像到视频的 SVG→视频渲染，rectified flow matching 的视频扩散训练，笔刷样本视觉提示，以及自回归视频模型的迁移。

**📊 数据集**

数据集：①合成几何原语（圆、矩形、三角形等）组成的形状组合视频，用于学习绘制语法；②仅七幅人工绘制的草图（灯、车、椅、树、杯、蝴蝶、花）用于学习视觉风格；③用于评估的 QuickDraw 50 类随机采样的 100 幅草图；④ 30 类草图用于笔刷风格泛化测试；⑤ 额外合成数据用于自回归模型的训练。

**📈 对比分析**

方法比较：与 Wan 2.1（直接提示）、PaintsUndo、SketchAgent、QuickDraw 的人类草图进行对比。使用 CLIP 0-1 评估最终帧识别率，结果为 82% Top-1（仅 SketchAgent 48%）和 96% Top-5（SketchAgent 71%）。时序进展评估显示本方法随时间逐步提升识别率；自回归版本 Top-1 45% 与人类 52%、SketchAgent 48% 相当，但视觉质量略低。

**⚠️ 局限性**

局限性：①像素空间生成难以精确控制结构，可能出现多笔触同时出现；②模型对提示的遵循不完全，可能偏离语义指令；③视频模型知识有限，难处理专业领域（如数学公式）或极端概念；④自回归模型视觉质量仍低于扩散模型。

---

## 316. RPT-SR: Regional Prior attention Transformer for infrared image Super-Resolution

**arXiv ID:** 2602.15490 | [PDF](https://arxiv.org/pdf/2602.15490v1)

**作者:** Youngwan Jin `[一作]` (Yonsei University), Shiho Kim `[通讯]` (Yonsei University)

**通讯引用:** 3010 | [OpenAlex ID](https://openalex.org/A5043085156)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于区域先验注意力 Transformer（RPT‑SR）的红外图像超分辨率模型，在固定视角场景中显式编码空间先验并结合局部信息；

**💡 创新点**

创新点：双 token 机制将可学习的区域先验 token 与动态本地 token 融合进入注意力；通过 Regional Prior Attention 强化场景常量结构指引；实现更高效、性能更优的超分；

**🔧 技术方法**

技术：Vision Transformer、窗口自注意力、层次注意力、双 token（learnable regional prior + dynamic local token）融合、轻量化实现；

**📊 数据集**

数据集：长波红外 LWIR（M3FD、TNO）和短波红外 SWIR（RASMD）数据集；

**📈 对比分析**

与 SOTA 对比：在 M3FD ×4 取得 LPIPS 0.1038、MANIQA 0.2621 等指标领先 HAT、DAT 等方法；在 RASMD、TNO 等数据集也表现最优或相近；整体性能显著提升；

**⚠️ 局限性**

局限：模型参数和 FLOPs 较基线略增；仅在固定视角场景下验证，动态场景效果未知；区域先验需针对每个场景学习，迁移性待进一步研究。

---

## 317. On Surprising Effectiveness of Masking Updates in Adaptive Optimizers

**arXiv ID:** 2602.15322 | [PDF](https://arxiv.org/pdf/2602.15322v1)

**作者:** Taejong Joo `[一作]` (Northwestern University), Eugene Ie `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出通过在自适应优化器中随机屏蔽参数块（SkipUpdate）以及结合动量-梯度相似度的有结构屏蔽（Magma），以改进大规模语言模型的预训练优化。

**💡 创新点**

创新点在于发现随机屏蔽可产生几何正则化，促使优化轨迹趋向平坦极小值；Magma利用动量一致性进一步筛选更新，提高鲁棒性；展示在LLM训练中稀疏更新可显著优于传统密集更新。

**🔧 技术方法**

技术方法包括：基于RMSProp/Adam等自适应优化器的包装器；对每个参数块使用Bernoulli随机屏蔽；计算动量与梯度余弦相似度并以此调节屏蔽权重；块级屏蔽实现高效剪枝；并给出理论分析证明其几何正则化效应。

**📊 数据集**

实验数据集包括：C4（用于LLama 2预训练）、OpenWebText（Nano MoE 预训练）、轻尾/重尾梯度噪声实验（模拟线性Transformer）以及ResNet‑50 on CIFAR‑10 作为对照。

**📈 对比分析**

与 Adam、Adafactor、APOLLO、Muon、RMSProp、C‑Adam、SGG、LaProp 等基线进行对比；在60M–1B Llama 2 预训练中，Magma 在所有规模下实现最低 perplexity（例如 1B 模型 13.71），并在 Nano MoE 与重尾梯度实验中同样显著优于基线，提升幅度约 1–2 个 perplexity 点。

**⚠️ 局限性**

局限性：效果主要体现在 Transformer/LLM 的高异质曲率场景，对均匀曲率模型（如 ResNet‑50）无显著提升；目前采用的屏蔽策略仍存在超参数（温度 τ）需调优；理论分析基于块级光滑与方差界假设，实际超大规模训练时可能与假设不完全一致。

---

## 318. On the Geometric Coherence of Global Aggregation in Federated GNN

**arXiv ID:** 2602.15510 | [PDF](https://arxiv.org/pdf/2602.15510v1)

**作者:** Chethana Prasad Kabgere `[一作]` (PES University), Shylaja SS `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种服务器端几何约束机制GGRS，用以在跨域联邦GNN中调节客户端更新，保持消息传递算子的几何一致性。

**💡 创新点**

创新点在于把联邦聚合视为操作符扰动的平滑，利用几何相容性（方向一致、子空间兼容、敏感度稳定）来防止传统FedAvg导致的算子退化。

**🔧 技术方法**

使用了基于方向相似度的代理表示、指数滑动参考向量、子空间投影以及对客户端更新的几何修正，全部实现于服务器端，无需访问本地图数据。

**📊 数据集**

采用了Amazon Computers、Amazon Photo以及Coauthor-CS三大真实图数据集，涵盖密集稀疏、特征维度差异以及跨域分布差异。

**📈 对比分析**

与FedAvg、FedSGD、FedProx等基线比较，GGRS在保持预测精度（误差 < 1%）的同时显著提升方向一致性（+约4-5倍）并降低敏感度波动，表明聚合更稳定。

**⚠️ 局限性**

局限性包括对大规模模型参数映射的近似、仅考虑线性一阶扰动、对极端异构或高维图结构的理论保证尚未完整，以及在安全加密联邦环境中的实现细节待完善。

---

## 319. Autodeleveraging as Online Learning

**arXiv ID:** 2602.15182 | [PDF](https://arxiv.org/pdf/2602.15182v1)

**作者:** Tarun Chitra `[一作]`, Victor Xu `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究并形式化了在重复使用下的自动削杠（ADL）策略，构建在线学习模型并在2025年10月10日Hyperliquid压力事件上进行评估。

**💡 创新点**

将ADL视为顺序在线决策问题，引入严重度估计与执行价格不确定性对追踪误差的分离，并给出实例校准的动态上界。

**🔧 技术方法**

采用投影OGD/向量镜像下降等在线凸优化技术、动态/静态遗憾分析、线性冲击模型、凸几何稳定性诊断以及公共回放数据的可观测性假设。

**📊 数据集**

利用公开的Hyperliquid 2025年10月10日压力事件回放数据（约21.03亿美元清算、16轮ADL）。

**📈 对比分析**

对比生产队列、整数pro-rata、向量镜像下降、最小-最大ILP等可部署策略，结果显示生产队列导致约6500万美元总损失，而最优可部署策略仅约340万美元，接近理论上限的2.6%。

**⚠️ 局限性**

实验基于固定回放路径，未考虑策略对清算行为、订单簿恢复力及用户撤资的反馈，也仅评估单一极端事件，缺乏对市场均衡的全面模拟。

---

## 320. Full-Field Damage Monitoring in Architected Lattices Using In situ Electrical Impedance Tomography

**arXiv ID:** 2602.15048 | [PDF](https://arxiv.org/pdf/2602.15048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 321. A ROS2 Benchmarking Framework for Hierarchical Control Strategies in Mobile Robots for Mediterranean Greenhouses

**arXiv ID:** 2602.15162 | [PDF](https://arxiv.org/pdf/2602.15162v1)

**作者:** Fernando Cañadas-Aránega `[一作]` (University of Almería), José L. Blanco-Claraco `[通讯]` (University of Almería)

**通讯引用:** 5343 | [OpenAlex ID](https://openalex.org/A5100772672)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一套可复现的绿色大棚环境移动机器人控制算法基准框架，用于评估从低层PID到高层路径规划的多层控制策略。

**💡 创新点**

创新点在于：①将真实大棚三维模型与物理仿真（MVSim）相结合，模拟载荷、土壤类型、坡度等真实扰动；②设计了分层控制架构与统一性能指标（SAE、SCI及组合成本），实现不同层次的客观可比；③提供插件化接口，支持用户自定义控制器与规划器。

**🔧 技术方法**

使用的技术包括：MVSim多车物理仿真、ROS2/NAV2框架、PID/抗风暴、MPC‑TEB轨迹跟踪、Lazy Theta*全局规划、统计重复实验评估。

**📊 数据集**

数据集主要为基于Almería（西班牙）Agroconnect温室的三维模型和土壤分区（砂、混砂、混凝土）参数，载荷范围0–70 kg，坡度±4%，在仿真中加入高斯噪声。

**📈 对比分析**

比较方法：对三类测试（低层、低+中层、全层）分别计算SAE、SCI及综合指标J_T；在同一扰动条件下多次实验取平均±标准差。结果表明：①PID层控制误差低、能耗小；②MPC‑TEB在复杂环境下轨迹误差降低但能耗上升；③全层架构通过规划减少低层能耗并提升整体鲁棒性。

**⚠️ 局限性**

局限性：仅在仿真环境验证，未涉及真实机器人硬件；受限于差速驱动模型和单一传感器（LiDAR+IMU）设置；扰动模型为预设，缺乏实时感知反馈；对更复杂任务（抓取、多机器人协作）尚未覆盖。

---

## 322. The Equalizer: Introducing Shape-Gain Decomposition in Neural Audio Codecs

**arXiv ID:** 2602.15491 | [PDF](https://arxiv.org/pdf/2602.15491v1)

**作者:** Samir Sadok `[一作]` (Inria), Xavier Alameda-Pineda `[通讯]` (Univ Grenoble Alpes)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `fede83ac-7505-405f-ab37-e7284695c47f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在神经音频编码（NAC）中引入形状-增益分解（Equalizer）机制，将输入信号先归一化为形状向量再通过NAC编码，增益单独用标量量化，最终在解码后合成完整信号；

**💡 创新点**

首次将经典的形状-增益分解应用于现代NAC框架，并将其在编码器前后分别实现，从而显著降低了对输入信号幅度变化的敏感性，并提升了码率-失真性能；

**🔧 技术方法**

使用短时帧归一化、重叠-加法（OLA）分析合成、EnCodec改进的BiLSTM编码器、残差向量量化（RVQ）、μ-law标量量化以及深度学习训练；

**📊 数据集**

LibriSpeech 100小时训练集、LibriSpeech test-clean评估集（约5.4小时）以及不同幅度变换的语音样本；

**📈 对比分析**

与不使用分解的基线模型（相同架构、相同码本大小）以及SpeechTokenizer、DAC、BigCodec等SOTA NAC进行对比；在相同或更低码率下，Equalizer在PESQ、STOI、SI-SDR等指标上优于基线并接近甚至超越SOTA，且显著减少了码本大小和计算复杂度；

**⚠️ 局限性**

仅在信号级别的归一化处理，未深入探究对嵌入向量分布的影响；未来工作需进一步分析归一化对量化器的作用并探索更高级的VQ技术。

---

## 323. 1-Bit Wonder: Improving QAT Performance in the Low-Bit Regime through K-Means Quantization

**arXiv ID:** 2602.15563 | [PDF](https://arxiv.org/pdf/2602.15563v1)

**作者:** Sohir Maskey `[一作]` (Aleph Alpha Research), Douglas Orr `[通讯]` (Graphcore Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在低比特量化中，利用k‑means聚类实现非线性量化，并通过量化感知训练（QAT）使LLM在固定内存预算下保持或提升推理性能。

**💡 创新点**

提出基于k‑means的非线性量化格式，在极低位数（1位）下仍能保持高效；证明在低位数下非线性量化优于统一整数量化，并给出在固定内存预算下精度与规模的最优权衡。

**🔧 技术方法**

使用块级量化、k‑means聚类、block scaling、量化感知训练（QAT）、融合去量化乘法内核以及理论速度模型。

**📊 数据集**

训练使用Nemotron‑CC、Starcoder、FineMath、Tulu 3 SFT混合数据；评估数据集包括MMLU、HellaSwag、PIQA、ARC、MBPP、HumanEval、GSM8K、IFEVAL、MMLU‑PRO、AidanBench等。

**📈 对比分析**

在约7.8 GB的推理内存预算下，31 B 1‑位k‑means模型在大多数基准上优于12 B 4‑位和4 B 16‑位模型，提升约10–20个百分点；在小批量推理时实现4.25/1.25‑位的3.7×/7.6×速度提升。

**⚠️ 局限性**

局限包括：仅探讨块级k‑means与1/4位，未覆盖更大位数或其它格式；QAT对极大规模或长训练时间的可扩展性有限；小批量推理加速在实际部署中受限；缺乏硬件原生支持。

---

## 324. GlobeDiff: State Diffusion Process for Partial Observability in Multi-Agent Systems

**arXiv ID:** 2602.15776 | [PDF](https://arxiv.org/pdf/2602.15776v1)

**作者:** Yiqin Yang `[一作]` (Chinese Academy of Sciences), Bo Xu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 46694 | [OpenAlex ID](https://openalex.org/A5102005952)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GlobeDiff，一种基于条件扩散模型的全局状态推断框架，解决多智能体系统的部分可观测问题。

**💡 创新点**

创新点在于用生成模型（条件扩散）处理一对多映射，避免模式崩溃；引入潜变量 z 作为模式选择器；通过前后网络训练实现仅用局部观测即可推断全局状态；理论证明误差有界。

**🔧 技术方法**

技术包括：条件扩散模型、潜变量推断、U‑Net 结构、KL 正则化、CTDE 与 MAPPO 的整合、贝叶斯估计与 RNN/Transformer 对比。

**📊 数据集**

数据集为 StarCraft II 多智能体环境 SMAC 的自定义版本：SMAC‑v1（PO）与 SMAC‑v2（PO），通过缩小视野并去除敌方信息来加剧部分可观测性。

**📈 对比分析**

与 LBS、Dynamic Belief、CommFormer、VAE、MLP、Joint 等基线在 SMAC‑v1/v2 任务中对比，GlobeDiff 在多数地图上显著提升胜率（尤其是多模态难度地图），证明其在全局状态推断与算法性能方面的优越性。

**⚠️ 局限性**

局限性包括：模型容量与训练成本高、未在真实环境中验证、对通信协议设计和推断速度的影响尚未充分评估。

---

## 325. GRAFNet: Multiscale Retinal Processing via Guided Cortical Attention Feedback for Enhancing Medical Image Polyp Segmentation

**arXiv ID:** 2602.15072 | [PDF](https://arxiv.org/pdf/2602.15072v1)

**作者:** Abdul Joseph Fofanah `[一作]` (Griffith University), Albert Patrick Sankoh `[通讯]` (Northeastern University)

**通讯引用:** 6086 | [OpenAlex ID](https://openalex.org/A5073452309)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

该论文提出了名为GRAFNet的多尺度视网膜处理与皮层注意反馈相结合的医学影像息肉分割网络，旨在提高分割准确性并提供可解释的决策路径。

**💡 创新点**

创新点在于融合视网膜多通路的MSRM、方向敏感的GAAM以及预测编码的GCAFM三大生物启发模块，实现多尺度特征融合、边缘强化和自适应反馈。

**🔧 技术方法**

采用卷积神经网络（ResNet-34编码器-解码器）、方向卷积、基于注意力的融合、交叉注意力、可学习的反馈门控以及自定义生物学损失函数等技术。

**📊 数据集**

训练和评估使用了五个公开息肉分割数据集：Kvasir-SEG、CVC-300、CVC-ColonDB、CVC-ClinicDB 与 PolypGen。

**📈 对比分析**

与13种SOTA方法对比，GRAFNet在所有数据集上均取得Dice最高（最高0.9465，平均提升3–8%），并在跨数据集泛化上显著优于其他方法。

**⚠️ 局限性**

局限性包括模型参数和算力仍相对较高，尚未充分实现实时部署，且主要针对单帧分割，未涉及视频时序建模。

---

## 326. RUVA: Personalized Transparent On-Device Graph Reasoning

**arXiv ID:** 2602.15553 | [PDF](https://arxiv.org/pdf/2602.15553v1)

**作者:** Gabriele Conte `[一作]` (Politecnico di Bari), Francesco Maria Donini `[通讯]` (Università degli Studi della Tuscia)

**通讯引用:** 4735 | [OpenAlex ID](https://openalex.org/A5106466740)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了Ruva系统，实现了基于个人知识图谱的玻璃盒子式个人AI，支持多模态摄取、图推理、可编辑删除。

**💡 创新点**

将向量检索转为图推理，实现可视化可编辑的知识图谱，并在设备端完成全流程，实现“可遗忘”权限。

**🔧 技术方法**

小型视觉语言模型SVLM、SQLite+图扩展、神经-符号架构、图遍历检索、实体解析与聚类、模型量化。

**📊 数据集**

在Pixel 8 Pro上使用本地多源数据（日历、图片、邮件、通话记录等），基准测试71个多源对象和52个问答三元组。

**📈 对比分析**

与四大LLM（Llama3.3‑70B、Qwen3‑32B、GPT、Kimi K2 Instruct）比较，61%回答正面、71%含中立；平均摄取2.4 s、单跳检索38 ms；一致性指标ρ = 0.82、α = 0.81。

**⚠️ 局限性**

目前仅在单设备上测试，未评估跨设备同步与大规模知识图扩展，SVLM的精度受限于本地计算能力。

---

## 327. Discovering Implicit Large Language Model Alignment Objectives

**arXiv ID:** 2602.15338 | [PDF](https://arxiv.org/pdf/2602.15338v1)

**作者:** Edward Chen `[一作]` (Stanford University), Carlos Guestrin `[通讯]` (Stanford University)

**通讯引用:** 47615 | [OpenAlex ID](https://openalex.org/A5090739892)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过迭代贪婪算法，自动将大型语言模型对齐奖励拆解为稀疏、可解释的自然语言目标。

**💡 创新点**

首次利用模型训练轨迹、proposer LLM 与 LLM-as-a-Judge 组合，消除预定义目标缺失问题，提出了从零开始发现隐式对齐目标的框架。

**🔧 技术方法**

匹配追踪启发的贪婪搜索、proposer LLM 生成候选目标、LLM-as-a-Judge 评估可解释性与趋势、组合函数回归实现目标重构。

**📊 数据集**

实验涵盖控制合成线性目标、公开奖励模型（DeBERTaV3、Skywork Reward）、LLM（Llama‑3.1‑8B、Qwen3‑4B），任务包括 TLDR、HH‑RLHF、Alpaca、Sky。

**📈 对比分析**

与 Iter‑Filter、Zero‑Shot 基线对比，模型适配率（Model‑Fit）均超 90%，人类评估显示目标识别率显著提升，误对齐案例中能有效发现隐性违规目标。

**⚠️ 局限性**

主要局限为高计算成本、依赖 LLM‑judge 可能带来偏差，以及对超大模型或资源受限场景的适配需进一步改进。

---

## 328. Automated Multi-Source Debugging and Natural Language Error Explanation for Dashboard Applications

**arXiv ID:** 2602.15362 | [PDF](https://arxiv.org/pdf/2602.15362v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 329. Beyond Binary Classification: Detecting Fine-Grained Sexism in Social Media Videos

**arXiv ID:** 2602.15757 | [PDF](https://arxiv.org/pdf/2602.15757v1)

**作者:** Laura De Grazia `[一作]` (University of Barcelona), Mariona Taulé `[通讯]` (University of Barcelona)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文创建了FineMuSe，一个包含828条西班牙语短视频的多模态性别歧视检测数据集，并在该数据集上对多种大语言模型（LLM）进行零样本评测；

**💡 创新点**

创新点包括：①推出细粒度多模态标签体系，涵盖四种性别歧视类型、两类非歧视内容以及讽刺与幽默等修辞手法；②实现跨平台（TikTok、BitChute、YouTube Shorts）的数据采集；③基于该数据集首次对多模态LLM在细粒度性别歧视检测上的表现进行系统基准；

**🔧 技术方法**

技术手段包括：多模态预处理（Whisper转录、视频帧抽取），零样本提示（prompt）与JSON输出，使用文本、视觉+文本以及视频输入的LLM（Llama-3、Qwen、Gemini、GPT‑4o、Claude‑3.7‑Sonnet）；

**📊 数据集**

使用的数据集为FineMuSe（含TikTok、BitChute、YouTube Shorts）以及其子集MuSeD，覆盖多种平台和语言变体；

**📈 对比分析**

通过零样本推理、准确率（binary）和宏F1（multi‑label）对模型进行比较，发现大型LLM和多模态模型显著优于随机与小模型；最佳模型Claude‑3.7‑Sonnet（V+L）在二分类上达85%+准确率，在细粒度任务上宏F1约68%，但仍存在较高的失败率，尤其是“对象化”类别；

**⚠️ 局限性**

局限性包括：难以捕捉视觉层面的多重性别歧视共现；讽刺与幽默标签的标注一致性低；“对象化”实例稀缺导致模型表现弱；零样本设置限制模型知识迁移，整体多标签分类仍具有挑战性。

---

## 330. Understanding vs. Generation: Navigating Optimization Dilemma in Multimodal Models

**arXiv ID:** 2602.15772 | [PDF](https://arxiv.org/pdf/2602.15772v1)

**作者:** Sen Ye `[一作]` (Peking University), Han Hu `[通讯]` (Tencent)

**通讯引用:** 9931 | [OpenAlex ID](https://openalex.org/A5091049278)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Reason-Reflect-Refine(R3)框架，将图像生成拆解为推理、反思、精细化三步，并将理解能力嵌入生成流程；

**💡 创新点**

创新点在于把理解任务作为生成的主动评估子任务，通过多步链式思考解决生成与理解的竞争冲突；

**🔧 技术方法**

采用BAGEL统一多模态模型，基于强化学习的Tree‑RL和GRPO优化策略，结合文本与扩散生成；

**📊 数据集**

使用GenEval++、TIIF等多模态基准数据集进行评估，并用GPT‑4.1、Qwen‑2.5‑VL‑72B等模型作为奖励；

**📈 对比分析**

与BAGEL、FLUX、Echo‑4o等方法对比，R3在GenEval++整体得分提升0.32、ITA提升约12.77%、VQA提升约3.15%，在多轮迭代中性能显著提升；

**⚠️ 局限性**

局限性包括理解能力在特定领域内提升显著但跨领域泛化有限，推理-反思循环会增加计算开销，并需手动设定最大迭代次数。

---

## 331. A Differential Fuzzing-Based Evaluation of Functional Equivalence in LLM-Generated Code Refactorings

**arXiv ID:** 2602.15761 | [PDF](https://arxiv.org/pdf/2602.15761v1)

**作者:** Simantika Bhattacharjee Dristi `[一作]` (University of Virginia), Matthew B. Dwyer `[通讯]` (University of Virginia)

**通讯引用:** 9971 | [OpenAlex ID](https://openalex.org/A5086757331)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估了六种大型语言模型在进行代码重构（性能优化与简化）时的功能等价性，采用差分模糊测试而非传统测试套件。

**💡 创新点**

创新之处在于提出并应用了 Eq@DFuzz 差分模糊等价检查器，系统展示了现有测试套件对功能错误检测的局限性，并对多模型、多数据集、多重构类型进行了首次大规模对比研究。

**🔧 技术方法**

使用差分模糊技术（Eq@DFuzz）、固定种子与统一解码参数、两种零样本 prompt（性能优化与代码简化）对六大 LLM 进行调用，并进行功能等价性检测。

**📊 数据集**

使用了三个主流编码基准数据集：HumanEval、MBPP 以及 APPS，分别覆盖函数级和程序级代码。

**📈 对比分析**

在 4368 条重构结果中，去除错误后 3538 条被 Eq@DFuzz 检测，发现 19%–35% 的重构非等价；与传统测试套件对比，约 21% 的非等价重构仍通过测试；实验显示代码复杂度越高，非等价率越高，模型间差异亦显著。

**⚠️ 局限性**

实验局限包括仅针对 Python 代码、仅评估六种 LLM、使用固定 prompt 与种子、未覆盖真实大规模项目的复杂度、差分模糊参数与约束推断依赖先前工作、可能存在的实现偏差等。

---

## 332. ChartEditBench: Evaluating Grounded Multi-Turn Chart Editing in Multimodal Language Models

**arXiv ID:** 2602.15758 | [PDF](https://arxiv.org/pdf/2602.15758v1)

**作者:** Manav Nitin Kapadnis `[一作]` (Carnegie Mellon University), Carolyn Rosé `[通讯]` (Carnegie Mellon University)

**通讯引用:** 12476 | [OpenAlex ID](https://openalex.org/A5089539629)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ChartEditBench，用于评估多模态大型语言模型在多轮图表代码编辑中的表现。

**💡 创新点**

创新点包括：① 设计可控制难度的多轮图表修改链（5,000条）和人工验证子集；② 开发视觉基准评估框架，融合代码执行、视觉相似度和结构化断言，弥补单一LLM-judge或CLIP指标的不足。

**🔧 技术方法**

采用LLM自动生成图表代码、AST与运行时验证、Chart-R1（专门的图表评估模型）、结构化断言与LLM判定相结合的混合评估技术。

**📊 数据集**

使用自研合成数据集ChartEditBench：5,000条多轮修改实例，涵盖12种修改类型、35种图表类型，另含430条人工验证子集。

**📈 对比分析**

比较方法基于执行率、指令遵循、代码质量、视觉相似度四大指标；Claude Haiku 4.5获得最高整体分（2.187），Qwen3‑VL‑30B‑A3B次之（2.139），小模型性能明显偏低。

**⚠️ 局限性**

局限性：多轮编辑中错误累积显著，尤其在数据操作类修改（如滚动平均、数据变换）仍难以正确实现；对程序实现与语义理解的双重需求尚未完全满足。

---

## 333. SVD Incidence Centrality: A Unified Spectral Framework for Node and Edge Analysis in Directed Networks and Hypergraphs

**arXiv ID:** 2602.15736 | [PDF](https://arxiv.org/pdf/2602.15736v1)

**作者:** Jorge Luiz Franco `[一作]` (Instituto Curvelo), Luiz Gustavo Nonato `[通讯]` (University of São Paulo)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于入射矩阵奇异值分解（SVD）的统一谱框架，用于同时计算有向网络及超图中的顶点和边的中心性。

**💡 创新点**

创新点在于：1）利用入射矩阵保持方向信息，避免传统方法的对称化导致信息丢失；2）通过Hodge理论统一顶点与边的中心性；3）获得稠密、无实现依赖的排序；4）自然扩展到超图。

**🔧 技术方法**

主要技术：入射矩阵的奇异值分解、Hodge Laplacian、Moore‑Penrose伪逆、方向聚合生成Hub/Authority分数、正则化与截断SVD、实验中使用多种网络分析指标对比。

**📊 数据集**

使用了多种真实网络数据集：Zachary's Karate Club、随机ER图、路径图、4×4有向格、荷兰学校友谊网络、C. elegans代谢网络、S. cerevisiae蛋白互作网络、OpenFlights航空网络、欧洲公路网络，以及XGI提供的四个超图（植物‑传粉者网络、人类疾病关联、医院接触网络、参议院共同提案网络）。

**📈 对比分析**

通过与当前流接近度中心性、Betweenness、PageRank、HITS、HEC/ZEC等传统度量在不同网络上进行相关性与可视化对比。实验显示：在无向图中SVD中心性与当前流接近度高度相关；在有向图和超图中SVD生成稠密、能区分方向角色（hub/authority）的排序，明显优于Betweenness（结果稀疏且易受实现细节影响），并且在多领域实验中保持一致的高相关性与更细致的分辨率。

**⚠️ 局限性**

限制：1）对极大稠密网络仍需截断SVD以降低计算开销；2）正则化参数与截断阶数对结果有一定影响；3）对非平衡或缺失数据的鲁棒性尚未充分评估；4）目前未处理时间序列动态网络与更高阶细胞复合结构。

---

