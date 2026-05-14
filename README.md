# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-05-14 | 今日论文总数: 703

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Seg-Agent: Test-Time Multimodal Reasoning for Training-Free Language-Guided Segmentation

**arXiv ID:** 2605.12953 | [PDF](https://arxiv.org/pdf/2605.12953v1)

**作者:** Chao Hao `[一作]` (Great Bay University), Zitong Yu `[通讯]` (Great Bay University)

**通讯引用:** 5270 | [OpenAlex ID](https://openalex.org/A5062522283)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 Seg-Agent，一种完全不需要训练的语言引导分割框架，通过在推理阶段构建生成‑选择‑细化的多模态链式推理来改进视觉提示，从而实现高质量分割。

**💡 创新点**

创新点在于：① 采用显式多模态链式推理（generation、selection、refinement），让 MLLM 在视觉域内进行迭代推理；② 利用 Set‑of‑Mark (SoM) 可视化提示增强 MLLM 的空间感知；③ 通过测试时推理而非大规模训练，兼容任何新出现的 MLLM 与分割基模型。

**🔧 技术方法**

技术细节包括：使用 QwenVL‑2.5（或 InternVL‑3）作为 MLLM 进行多模态提示；使用 SAM2（尤其是 SAM2‑Large）作为基础分割器；通过图像增强、NMS、SoM 可视化和三阶段推理实现；并采用 gIoU 与 cIoU 作为评估指标。

**📊 数据集**

数据集方面：构建了 Various‑LangSeg（共 244 条样本，覆盖 Explicit Semantic、Generic Object、Reasoning‑Guided 三类任务），并在 refCOCO、refCOCO+、refCOCOg 以及 ReasonSeg 等公开数据集上进行评测。

**📈 对比分析**

对比实验显示：在各类分割任务中，Seg‑Agent 在不进行任何训练的情况下，单模型（7B）在 Overall 上达到 83.0 gIoU，优于同类训练‑free 方法（如 Qwen2.5‑VL+SAM2‑L）并逼近或超过部分训练‑based 方法（如 Seg‑Zero）。实验还验证了三阶段链式推理对复杂推理任务（RGS）提升显著。

**⚠️ 局限性**

局限性包括：推理阶段需要多次 MLLM 调用，导致相对单步方法的延迟；性能高度依赖底层 MLLM 与分割模型的能力；以及评测数据集规模较小（244 条），难以充分覆盖所有真实世界场景。

---

## 2. Moltbook Moderation: Uncovering Hidden Intent Through Multi-Turn Dialogue

**arXiv ID:** 2605.12856 | [PDF](https://arxiv.org/pdf/2605.12856v1)

**作者:** Ali Al-Lawati `[一作]` (Pennsylvania State University), Dongwon Lee `[通讯]` (Pennsylvania State University)

**通讯引用:** 9580 | [OpenAlex ID](https://openalex.org/A5100405086)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种基于代理隐藏意图的多轮对话式审核框架，利用Gibbs采样不断细化对话中生成的意图假设，从而在代理社交平台上识别并拦截恶意行为。

**💡 创新点**

创新点包括：①将审核焦点从表面内容转向代理隐性意图；②通过自适应Gibbs采样动态生成探测问题，形成“采访式”多轮交互；③引入Autoresearch自动化实验控制器，自动发现并优化询问策略。

**🔧 技术方法**

技术手段：大语言模型(如LLaMA、Mistral、Qwen)做为调节者与代理对话；Gibbs采样实现意图分布更新；自一致性、链式思考、self‑refine等推理策略；Autoresearch框架进行黑盒优化；细粒度意图推断与二分类决策。

**📊 数据集**

数据集：从Molttbook社区构造的两套数据集——Post Dataset（帖子）和Comment Dataset（评论），包含5类意图（organic_contribution、elicitation、narrative_pushing、subtle_promotion、spam），分别划分为训练集（40%）与测试集（60%），并额外构造OOD测试。

**📈 对比分析**

对比方法：零射击、零射击+、链式思考、self‑consistency、self‑refine、fine‑tuned模型六种基线。实验表明，在ID和OOD下，本文框架的平均F1（约0.68/0.64）均超过所有基线，尤其在评论集上提升显著，且对“假装善意”逃逸攻击表现出较强鲁棒性。

**⚠️ 局限性**

局限性：审核过程需要7轮多轮对话，计算开销大，易受高频“边界”内容攻击导致拒绝服务；未考虑对审核器自身的Prompt Injection或更高级的对抗代理；在跨域泛化、样本效率和对极端恶意代理的鲁棒性方面仍需进一步研究。

---

## 3. Adaptive Conformal Prediction for Reliable and Explainable Medical Image Classification

**arXiv ID:** 2605.12917 | [PDF](https://arxiv.org/pdf/2605.12917v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 4. Stress-Testing the Reasoning Competence of LLMs With Proofs Under Minimal Formalism

**arXiv ID:** 2605.12524 | [PDF](https://arxiv.org/pdf/2605.12524v1)

**作者:** Konstantine Arkoudas `[一作]`, Serafim Batzoglou `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

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

## 5. Spectral Energy Centroid: a Metric for Improving Performance and Analyzing Spectral Bias in Implicit Neural Representations

**arXiv ID:** 2605.12709 | [PDF](https://arxiv.org/pdf/2605.12709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 6. Macro-Action Based Multi-Agent Instruction Following through Value Cancellation

**arXiv ID:** 2605.12655 | [PDF](https://arxiv.org/pdf/2605.12655v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 7. Anatomy-Slot: Unsupervised Anatomical Factorization for Homologous Bilateral Reasoning in Retinal Diagnosis

**arXiv ID:** 2605.12929 | [PDF](https://arxiv.org/pdf/2605.12929v1)

**作者:** Yingzhe Ma `[一作]` (University of Electronic Science and Technology of China), Zheyu Wang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 780 | [OpenAlex ID](https://openalex.org/A5101616776)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出Anatomy‑Slot模型，通过无监督的槽分解和双向交叉注意力实现双眼解剖对应，用于视网膜疾病诊断。

**💡 创新点**

将对象中心槽分解与双眼跨眼交叉注意力相结合，形成显式解剖对应的无监督瓶颈，并在统计控制实验中验证其对应性依赖。

**🔧 技术方法**

使用Slot Attention、ViT‑L预训练骨干、双向交叉注意力、低分辨率重建损失、Wilcoxon检验、Fisher Ratio和optic disc grounding等技术。

**📊 数据集**

主要使用ODIR‑5K进行训练与评估，预训练池包含ODIR、APTOS2019、IDRiD、MESSIDOR2、PAPILA、Retina，外部验证使用EyePACS，解剖验证使用REFUGE。

**📈 对比分析**

在10个随机种子、95%置信区间和Wilcoxon检验下，Anatomy‑Slot将AUC从0.823提升至0.865（+4.2%），并通过对齐破坏和噪声测试验证对应性，同时与其他通用和医学基础模型对比显示最高AUC。

**⚠️ 局限性**

仅在单一主要数据集评估、解剖覆盖部分验证、未测试临床对齐误差和多中心鲁棒性等限制。

---

## 8. MIRACLE_Multi-Agent Intelligent Regulation to Advance Collaborative Learning Environment

**arXiv ID:** 2605.12923 | [PDF](https://arxiv.org/pdf/2605.12923v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 9. Generative Motion In-betweening by Diffusion over Continuous Implicit Representations

**arXiv ID:** 2605.12778 | [PDF](https://arxiv.org/pdf/2605.12778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 10. Inline Critic Steers Image Editing

**arXiv ID:** 2605.12724 | [PDF](https://arxiv.org/pdf/2605.12724v1)

**作者:** Weitai Kang `[一作]` (University of Illinois Chicago), Yan Yan `[通讯]` (University of Illinois Chicago)

**通讯引用:** 20012 | [OpenAlex ID](https://openalex.org/A5100395068)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出在图像编辑模型前向传递过程中插入一个可学习的批判者（Inline Critic）token，使冻结的编辑器能够在中间层实时评估并纠正生成错误。

**💡 创新点**

创新点在于将批判者设计为可学习的 token，并通过三阶段训练（probe、masked critic、unmasked joint）让其在前向过程中预测错误并通过注意力直接调节后续层的生成，从而实现“即时批判与修正”。

**🔧 技术方法**

采用了冻结的 Qwen‑Image‑Edit 与 MM‑DiT 结构，使用 AdaLN 与 MLP 作为 probe head，log 归一化处理误差，Attention mask 让批判者不影响模型隐藏状态，并在训练中使用三阶段 curriculum 与 MSE 损失。

**📊 数据集**

在公开的 GEdit‑Bench、KRIS‑Bench 与 RISEBench 等基准数据集上进行评估，训练时使用多来源随机采样的 400 个样本并结合公开编辑任务数据。

**📈 对比分析**

与同一 backbone 及多种 post‑generation/post‑step 细化方法对比，本文在 GEdit‑Bench 得分 7.89、KRIS‑Bench 81.92、RISEBench 37.8（比 backbone +9.4），在开源社区中甚至超过 GPT‑4o。

**⚠️ 局限性**

局限性包括：训练过程分阶段且复杂；批判者仅适用于冻结模型，难以直接迁移至可微分更新的编辑器；对极端局部细节的改进仍有限，且需要额外的 probe 计算成本。

---

## 11. Bridging the Missing-Modality Gap: Improving Text-Only Calibration of Vision Language Models

**arXiv ID:** 2605.12517 | [PDF](https://arxiv.org/pdf/2605.12517v1)

**作者:** Mingyeong Kim `[一作]` (Korea Advanced Institute of Science and Technology), Juho Lee `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3201 | [OpenAlex ID](https://openalex.org/A5100680420)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种轻量级的跨模态注意模块LIM，在文本输入缺失视觉模态时生成潜在视觉嵌入，使冻结的VLM能够在缺失视觉信息时仍保持高准确性和良好校准。

**💡 创新点**

创新点在于将视觉模态缺失问题视为潜在空间补全，采用基于跨注意的可训练查询向量直接推断与视觉塔相兼容的潜在嵌入，而非昂贵的像素级图像生成或简单的文本补全。

**🔧 技术方法**

技术包括跨注意力块、预归一化、任务导向的负对数似然训练、以及冻结的VLM骨干与轻量级LIM的端到端耦合。

**📊 数据集**

主要使用的数据集有VQA‑v1、MMLU、ARC、ScienceQA以及八个未见的文本基准（SST‑2、CoLA、AG News、MRPC、Vitamin‑C、LogiQA、CommonsenseQA、QASC）来评估准确率和校准误差。

**📈 对比分析**

在所有基准上，LIM相较于文本仅输入、温度标度和随机视觉填充显著降低ECE（如从0.4202降至0.0374），准确率提升20%以上，同时比扩散图像生成快约12倍、显存消耗相近。

**⚠️ 局限性**

局限性包括：对视觉塔结构的依赖；对跨模态预训练的VLM质量敏感；在极端文本缺失或与视觉语义高度不匹配的场景下效果可能受限；以及仅在文本+多项选择QA任务上充分验证，其他任务的泛化仍待探索。

---

## 12. Time and Supply Fairness in Electricity Distribution using $k$-times bin packing

**arXiv ID:** 2605.12812 | [PDF](https://arxiv.org/pdf/2605.12812v1)

**作者:** Dinesh Kumar Baghel `[一作]` (Ariel University), Erel Segal-Halevi `[通讯]` (Ariel University)

**通讯引用:** 773 | [OpenAlex ID](https://openalex.org/A5085873807)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了k倍装箱(k-times bin‑packing, kBP)并将其应用于公平电力分配问题，提出了多种扩展算法并给出了存在有限k使得kBP能得到最优公平分配的证明；

**💡 创新点**

创新点在于证明每个电力分配实例都能通过某个有限k的kBP得到最优公平分配并给出k的指数上界，同时改进了FF、FFD等经典装箱算法的kBP版本并分析其近似比，此外提出四种启发式求解功率公平分配的新方案；

**🔧 技术方法**

主要技术包括kBP的定义与理论分析、配置线性规划与线性分组、对First‑Fit、First‑Fit‑Decreasing、Next‑Fit的kBP扩展、基于Fernandez de la Vega‑Lueker与Karmarkar‑Karp的PTAS、以及多种启发式算法；

**📊 数据集**

实验使用印度UPES电力分配数据集（367户）以及其他真实需求数据，同时采用人工构造的多种实例进行验证；

**📈 对比分析**

与现有公平负荷削减启发式和ILP算法比较，FFk和FFDk在均衡连接时间上取得更优的性能；启发式HA1+FFDk在功率公平度上表现最佳；实验结果表明FFk/FFDk的近似比可逼近1.35–1.375，显著优于传统FF的1.7；

**⚠️ 局限性**

主要局限包括：kBP为NP‑hard，最优k难以精确得到；对FFDk和NFk的理论上界不够紧，且实验验证仅限于有限数据；未来需进一步改进下界、探索更高效的近似与多目标公平度评估。

---

## 13. From Heuristics to Analytics: Forecasting Effort and Progress in Online Learning

**arXiv ID:** 2605.12788 | [PDF](https://arxiv.org/pdf/2605.12788v1)

**作者:** Eric S. Qiu `[一作]` (Cornell University), Conrad Borchers `[通讯]` (Carnegie Mellon University)

**通讯引用:** 350 | [OpenAlex ID](https://openalex.org/A5037442366)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于 ITS 日志的周预测任务，预测学生下周的练习时长（努力）和新掌握技能数量（进步）；

**💡 创新点**

创新点包括：①将可解释的周预测任务引入 ITS 并与来自健康行为领域的百分位规则进行系统比较；②发现努力和进步受不同特征驱动，强调最近活动与学习者状态的差异；③通过导师访谈验证预测信号的可解释性与实际教学目标设置的契合度；

**🔧 技术方法**

使用了多种机器学习模型（线性回归、岭、LASSO、决策树、随机森林、XGBoost、MLP、LSTM 等），结合 AFM、近期活动、练习缺口、先前成绩等特征工程，并对特征重要性和消融效果进行了可解释性分析；

**📊 数据集**

使用了 425 名中学生在 39 周内收集的 ITS 日志数据（约 290 万条交互），按 ISO 周聚合生成每周练习时长和新技能掌握数的目标；

**📈 对比分析**

模型通过学生级拆分的 70/30 训练-测试以及 5 折时间序列交叉验证与多种基线（last‑value、均值、百分位规则）对比；XGBoost 等模型在 MAE 上比基线降低 22–33%，且差异在统计上显著；百分位规则普遍过度预测，尤其 60%/70% 百分位；

**⚠️ 局限性**

局限性包括：仅在单一 ITS 与中学数学情境下验证；特征高度依赖详细日志（如机会计数、技能标签），难以直接迁移；未对学习成效或跨校场景进行外部验证；预测值未直接评估对教学干预或学习效果的影响。

---

## 14. AdaFocus: Adaptive Relevance-Diversity Sampling with Zero-Cache Look-back for Efficient Long Video Understanding

**arXiv ID:** 2605.12954 | [PDF](https://arxiv.org/pdf/2605.12954v1)

**作者:** Xiao Yang `[一作]` (University of Electronic Science and Technology of China), Ning Qin `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 34963 | [OpenAlex ID](https://openalex.org/A5100404886)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 AdaFocus 框架，实现长视频理解的渐进式证据获取和零缓存检索，显著降低视觉 token 数量。

**💡 创新点**

创新点在于：① Query‑aware adaptive relevance‑diversity（AdaRD）采样，动态平衡相关性与时序多样性；② 置信度触发的零缓存时序检索，按需从磁盘读取高分辨率窗口，避免全量预缓存。

**🔧 技术方法**

使用技术包括：CLIP 编码、AdaRD 采样、长度校准置信度门控、GRPO 强化学习对 LVLM 的对齐、零缓存 I/O、正则化多步推理和交叉注意力回退。

**📊 数据集**

评估数据集：七大长视频基准——VideoMME、VideoMMMU、MVBench、LongVideoBench、MMVU、Charades‑STA、ActivityNet‑TVG。

**📈 对比分析**

比较方法：与单传递 baseline、CoT‑only、对齐后（GRPO）backbone 以及同行方法对比；AdaFocus 在七个基准上均优于对齐+CoT，最高提升 +8.39 mIoU (Charades‑STA)，+3.47 accuracy (VideoMMMU)，同时仅使用约 1/33 的视觉 token。

**⚠️ 局限性**

局限性：多轮推理导致 2~4 倍推理时间，依赖离线 CLIP 特征；受限于 7B LVLM 的上下文长度，对极长视频的时延与存储仍有挑战。

---

## 15. Discrete Stochastic Localization for Non-autoregressive Generation

**arXiv ID:** 2605.12836 | [PDF](https://arxiv.org/pdf/2605.12836v1)

**作者:** Yunshu Wu `[一作]` (University of California Riverside), Greg Ver Steeg `[通讯]` (University of California Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Discrete Stochastic Localization (DSL) 的连续状态框架，用于离散序列生成，并在同一模型上支持多种采样路径（如masked refinement、随机顺序自回归和连续-离散混合采样）。

**💡 创新点**

核心创新在于将所有清洁 token 嵌入限制在单位球面，并使用 stochastic‑localization 观测通道，使得 Bayes 最优去噪器仅依赖噪声状态而与 SNR 无关，从而实现时间无关的去噪器和统一的后验估计器；同时设计混合-SNR 训练目标和后验视图转换器，兼容标准 Transformer/DiT 结构。

**🔧 技术方法**

使用的技术包括：单位球面 token 嵌入、stochastic‑localization SDE、时间无关 MMSE 去噪器、混合‑SNR 训练策略、后验视图转换器（将噪声向量映射为词汇混合）、ReMDM 家族迭代细化采样以及连续-离散混合采样方法。

**📊 数据集**

主要数据集为 OpenWebText (OWT) 用于无条件生成评估，Text8 用于对数似然（BPC）评估；模型使用 GPT‑2 BPE 分词。

**📈 对比分析**

与先前的 masked‑diffusion、ReMDM 以及连续扩散模型对比，DSL 在 OWT 上在不同采样步数（T=128~1024）下的 MAUVE 分数显著提升，最高可达 0.722；在 Text8 上的 BPC 达到 1.45，逼近 masked‑diffusion 的 1.40 左右，优于之前的 continuous‑diffusion 方法 (1.48)。

**⚠️ 局限性**

局限性包括：仅在 OWT 与 Text8 上验证，未进行充分的解码策略搜索；混合采样与 ReMDM 的具体参数选择仍可进一步优化；模型规模、上下文长度和条件生成等方面尚未深入探索。

---

## 16. Action Emergence from Streaming Intent

**arXiv ID:** 2605.12622 | [PDF](https://arxiv.org/pdf/2605.12622v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 17. Same Image, Different Meanings: Toward Retrieval of Context-Dependent Meanings

**arXiv ID:** 2605.12905 | [PDF](https://arxiv.org/pdf/2605.12905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 18. Inference-Time Machine Unlearning via Gated Activation Redirection

**arXiv ID:** 2605.12765 | [PDF](https://arxiv.org/pdf/2605.12765v1)

**作者:** Vinícius Conte Turani `[一作]` (Pontifical Catholic University of Rio Grande do Sul), Lucas S. Kupssinskü `[通讯]` (Pontifical Catholic University of Rio Grande do Sul)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练-梯度无关的 LLM 记忆消除方法 Guard，利用激活空间的旋转实现对指定忘记集的动态干预，

**💡 创新点**

通过将忘记集语义聚类、使用相似度门控并执行范数保持的旋转，实现输入依赖、无参数更新且对量化稳健的记忆消除，

**🔧 技术方法**

聚类（k‑means）、句子 Transformer 嵌入、相似度门控、激活空间旋转与范数归一化等技术，

**📊 数据集**

在 TOFU 与 MUSE 两大 benchmark（主要评测实体级知识消除）上进行评估，

**📈 对比分析**

与 12 个梯度基准方法对比，Guard 在忘记-保留平衡上与梯度方法持平或更优，同时避免灾难性崩塌并保持生成连贯性，

**⚠️ 局限性**

依赖线性表示假设，且仅在实体关联类任务上验证，未检验对非线性或多模态记忆消除的适用性

---

## 19. Differences in Text Generated by Diffusion and Autoregressive Language Models

**arXiv ID:** 2605.12522 | [PDF](https://arxiv.org/pdf/2605.12522v1)

**作者:** Zeyang Zhang `[一作]` (Shanghai Qi Zhi Institute), Tianxing He `[通讯]` (Tsinghua University)

**通讯引用:** 81777 | [OpenAlex ID](https://openalex.org/A5050844324)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比了扩散语言模型（DLM）和自回归语言模型（ARM）在文本生成时的熵、语义连贯性与语义多样性等属性，并通过训练目标与解码策略的分离实验揭示了它们产生差异的机制。

**💡 创新点**

首次将训练目标与解码策略分离，证明了双向上下文是提升语义连贯性与多样性的关键；同时发现置信度重掩码策略通过产生偏差导致熵降低，给出理论解释和偏差消除方法。

**🔧 技术方法**

采用交叉熵训练、n-gram 熵估计、句子/文本级语义相似度度量、四种重掩码策略（低置信度、动态低置信度、高熵、随机）以及对偏差的理论推导与实验验证。

**📊 数据集**

在 Fineweb 与 TinyStories 两个大规模文本数据集上训练并评估模型，使用统一解码与多随机种子进行对比实验。

**📈 对比分析**

通过统一的自回归解码基线，交叉比较八种中间训练目标与多种重掩码策略，实验显示 DLM 在语义连贯性和多样性上显著优于 ARM，熵相同或更低；结果在不同数据集、模型规模与随机种子下保持稳健。

**⚠️ 局限性**

实验仅覆盖 120M–8B 规模模型，理论分析基于简化假设，未验证更大模型、不同 DLM 架构或更复杂解码算法的适用性。

---

## 20. Improving Diffusion Posterior Samplers with Lagged Temporal Corrections for Image Restoration

**arXiv ID:** 2605.12573 | [PDF](https://arxiv.org/pdf/2605.12573v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 21. Prediction of Rectal Cancer Regrowth from Longitudinal Endoscopy

**arXiv ID:** 2605.12855 | [PDF](https://arxiv.org/pdf/2605.12855v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 22. AuraMask: An Extensible Pipeline for Developing Aesthetic Anti-Facial Recognition Image Filters

**arXiv ID:** 2605.12937 | [PDF](https://arxiv.org/pdf/2605.12937v1)

**作者:** Jacob Lagogiannis `[一作]` (Franklin and Marshall College), Sauvik Das `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2884 | [OpenAlex ID](https://openalex.org/A5006053551)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了AuraMask工具包，能够生成既能干扰面部识别系统又能满足用户审美偏好的抗面部识别（AFR）图像滤镜。

**💡 创新点**

创新点在于将对抗性训练与多任务学习相结合，允许在保持对抗效果的同时加入任意美学目标（如模拟Instagram滤镜），从而突破传统对抗方法对“不可察觉”的严格限制。

**🔧 技术方法**

主要技术包括对抗转换网络（ATN）和多种U‑Net变体（VNet、R2U‑Net等），以及联合优化对抗损失（feat）和美学损失（ℒ_AES）的多任务框架。

**📊 数据集**

使用了三大人脸数据集：FDF（1.5M脸图）、LFW（13.2k图像及2.2k对），以及VGGFace2（3.3M图像）作为训练与评估素材。

**📈 对比分析**

与现有的Fawkes和LowKey基准进行对比，评估指标包括距离发散率和人脸验证召回率。结果显示，ensemble‑target版本在ArcFace、VGGFace2和FaceNet三种模型上均优于基准，最高可将验证召回率降低近94%。在用户研究中，AuraMask滤镜在SAIA‑8接受度评分及多选偏好测试中均显著优于传统方法。

**⚠️ 局限性**

局限性包括：1）用户研究仅使用单目标（相对弱）模型，未覆盖ensemble‑target的强对抗性；2）ensemble‑target滤镜产生的结构性伪影可能不符合某些用户审美；3）评估基于第三方照片，未验证对个人照片的适用性；4）未与最新非开源AFR方法进行比较。

---

## 23. Finding a Crab in the C: Assured Translation via Comparative Symbolic Execution

**arXiv ID:** 2605.12731 | [PDF](https://arxiv.org/pdf/2605.12731v1)

**作者:** Caleb Helbling `[一作]` (Draper), Michael Crystal `[通讯]` (Draper)

**通讯引用:** 326 | [OpenAlex ID](https://openalex.org/A5015191649)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并验证了一款名为 cozy 的工具，利用比较符号执行技术对 C 与 Rust 程序的二进制进行行为差异分析，并通过交互式可视化与差异报告帮助开发者确认翻译或修复是否保持语义一致。

**💡 创新点**

创新点包括：① 将 unsat core 缓存与比较符号执行相结合，显著减少 SMT 求解调用；② 引入注解机制以对齐跨语言的内存布局；③ 在 Python 环境中实现完整的交互式 GUI，支持高亮、裁剪、压缩与事件流 diff；④ 通过 DWARF 调试信息和手工注解实现对 Rust ABI 的有限支持。

**🔧 技术方法**

核心技术：Angr 框架进行比较符号执行；Z3 SMT 求解器做约束求解；Python 库与 GUI 组件；DWARF 调试数据与自定义注解用于跨语言映射；压缩与裁剪算法用于简化执行树。

**📊 数据集**

实验数据集：手写 C 与 Rust 版本的四个程序（插排、冒泡、时钟计数器、盒式模糊），以及通过 Galois/Immunant 自动翻译的 C2Rust 版本；实验代码与数据可在 Zenodo（记录 19669436）公开获取。

**📈 对比分析**

比较方法：将两条程序分别符号执行到终态，利用 unsat core 缓存判断兼容路径；对兼容终态进行内存、寄存器与 I/O 事件流的差异比对；提供差异面板、具体输入示例与可视化树。性能方面，瓶颈主要在二进制解释器的执行速度，而非 SMT 求解；关闭编译器优化后可提升可证明性。

**⚠️ 局限性**

主要限制：① 性能受限于 Python 版 angr 的解释执行；② 对 Rust 数据结构与 ABI 的支持有限，需手工注解；③ 编译器优化（如模数运算简化）导致难以建模；④ 未支持所有 Rust 类型，复杂动态结构的精确建模仍待完善。

---

## 24. No One Knows the State of the Art in Geospatial Foundation Models

**arXiv ID:** 2605.12678 | [PDF](https://arxiv.org/pdf/2605.12678v1)

**作者:** Isaac Corley `[一作]` (Taylor Geospatial), Hannah Kerner `[通讯]` (Taylor Geospatial)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 2019‑2025 年间 152 篇地理空间基础模型（GFM）论文进行系统审计，发现模型权重未公开、基准分散、结果差异大、缺乏统一评测工具以及架构与预训练数据混淆等问题，并据此提出 6 条标准化与评测共享工具的建议。

**💡 创新点**

首次全面量化 GFM 文献的可比性缺陷，展示跨论文同一模型-基准-协议组合的数值差距（最高 56.6 分），并给出可操作的社区规范（权重量化、共享核心基准、复制与重跑标注、方差报告、评测工具、数据与架构解耦）来解决这一“协调失效”。

**🔧 技术方法**

利用 LLM（Claude Opus 4.7、GPT‑5.5 Codex）对 LaTeX 与 PDF 进行结构化抽取，配合 Docling 转 Markdown；随后通过脚本统计基准使用、Gini 系数、跨论文数值差距；人工校验保证抽取准确性。

**📊 数据集**

审计涵盖 152 篇论文，涉及 401 个不同的遥感基准（如 EuroSAT、NWPU‑RESISC45、AID 等）以及 87 个命名预训练数据集（如 MillionAID、SSL4EO‑S12 等）。不新建数据集，完全使用已有公开基准与预训练集。

**📈 对比分析**

通过对 301 个多论文基准‑协议组合的指标进行最大–最小差距分析，发现 46 个 ≥10 分、20 个 ≥20 分的显著差异；平均差距 12.7 分，远高于同一协议下的种子波动；说明当前评测缺乏可重复性。建议使用共享核心基准、统一评测工具与方差报告，以提升模型性能对比的可靠性。

**⚠️ 局限性**

限制包括：抽取依赖 LLM 与公共 API，仍可能出现误提；未涵盖所有已发表论文（如 pay‑wall、非 arXiv 版本）；对预训练数据细节的抽象可能低估差异；评测工具与基准选择尚未形成共识，标准化建议仍需社区进一步讨论。

---

## 25. Quantifying LLM Safety Degradation Under Repeated Attacks Using Survival Analysis

**arXiv ID:** 2605.12869 | [PDF](https://arxiv.org/pdf/2605.12869v1)

**作者:** Zvi Topol `[一作]` `[通讯]` (MuyVentive, LLC), Zvi Topol (MuyVentive, LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文利用生存分析框架对LLM越狱安全性进行评估，建模并量化“时间到越狱”事件，进而绘制生存曲线、危害函数并识别模型的易受攻击窗口。

**💡 创新点**

创新点在于首次将医学、工程等领域常用的生存分析方法（Kaplan‑Meier估计、危害函数和log‑rank检验）引入LLM安全评估，能够捕捉攻击的时间动态与风险变化，而非单纯的成功/失败二值指标；同时提供可视化的风险窗口与攻击优先级决策依据。

**🔧 技术方法**

核心技术包括：
- Kaplan‑Meier非参数生存曲线估计（对不同模型、不同攻击类别）
- 离散时间危害函数计算
- Pairwise log‑rank检验评估生存曲线差异
- 使用LLM‑as‑a‑Judge（Llama 3.1 70B Turbo）对模型回应进行等级（L0–L4）判定
- 对同一提示进行多轮（最多10轮）攻击，记录事件或删失时间。

**📊 数据集**

数据集主要包括：
- HarmBench 3个子类别（Misinformation & Disinformation、Illegal Activities、General Harm）共60个提示（20个/类）
- WildGuard 749个非安全案例用于训练和验证Judge模型，保证评判质量。

**📈 对比分析**

比较方法：
- 计算攻击成功率（ASR）
- 通过Kaplan‑Meier曲线获得中位生存时间、5次/10次攻击后的生存概率
- 对不同模型及子类别做pairwise log‑rank检验。结果显示：Phi‑3 Mini 具有最低ASR、最快越狱；Llama 3.2 3B 处于中等水平；Qwen 3 4B 在 General Harm 类别上表现最稳健。整体差异在统计上显著（p<0.05）。

**⚠️ 局限性**

局限性：
- 仅评估了三款小型模型，未覆盖更大规模或多样化的LLM
- 越狱评分依赖LLM‑as‑a‑Judge，可能存在误判，需进一步细化
- 假设删失事件非信息删失，若攻击长度与难度相关可能导致偏差
- 数据集规模有限，缺乏更多真实世界攻击场景

---

## 26. Toward Communication-Efficient Space Data Centers: Bottlenecks, Architectures, and New Paradigms

**arXiv ID:** 2605.12681 | [PDF](https://arxiv.org/pdf/2605.12681v1)

**作者:** Minghao Sun `[一作]`, Xiaoli Chu `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析空间数据中心（SDC）的通信瓶颈，提出基于语义通信的多层异构架构并在能耗和热约束下进行评估。

**💡 创新点**

将语义通信与多天线波束成形、激光互连（ISL）以及能热协同调度相结合，显著降低地空上行需求并验证其可行性。

**🔧 技术方法**

语义通信、MIMO波束成形、激光互连（ISL）、多层调度、深度学习编码器（ResNet18）等。

**📊 数据集**

案例使用CIFAR-10图像进行语义压缩实验，其他模拟基于通用大模型推理。

**📈 对比分析**

对比传统按比特传输(BitCom)与语义传输(SemCom)，在不同功率预算下测算单通道速率与地面站能耗；SemCom将上行速率降至<2%，能耗更低，任务精度保持94.4%。

**⚠️ 局限性**

局限在于大规模预训练的语义重建、跨任务知识迁移不足、语义信息长期安全威胁以及对更复杂任务的适应性验证。

---

## 27. On the Size Complexity and Decidability of First-Order Progression

**arXiv ID:** 2605.12691 | [PDF](https://arxiv.org/pdf/2605.12691v1)

**作者:** Jens Classen `[一作]`, Daxin Liu `[通讯]` (Nanjing University)

**通讯引用:** 474 | [OpenAlex ID](https://openalex.org/A5071267805)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

本文研究了情境算子中局部效应、正规及无环动作三类动作理论在进行进化（progression）时所得到的第一阶逻辑（FO）知识库的大小复杂性与可判定性问题，证明在合理假设下这些进化结果的大小只呈多项式增长并且可保持在可判定的逻辑子类（如两变量片段与具有常量的全称理论）内；

**💡 创新点**

创新点在于首次给出了进化结果大小的系统性多项式（甚至线性）上界，并证明在局部效应、正规及无环动作类下，进化过程可保持在两变量片段和UTC（Universal Theory with Constants）等可判定子类内，从而保证了查询评估的可行性；

**🔧 技术方法**

主要采用了情境算子框架、FO进化与遗忘（forgetting）技术、Ackermann引理、依赖图分析以及逻辑片段闭包性质等理论工具；

**📊 数据集**

本文为理论研究，没有使用具体数据集；

**📈 对比分析**

通过复杂度分析与理论证明，进化结果的大小在局部效应类为线性，在正规类为二次多项式，在无环类为多项式（含指数因子但与特征集大小有关），从而在这些约束下实现了可判定的查询评估；

**⚠️ 局限性**

局限性在于仅适用于局部效应、正规和无环动作三类，并且需要对初始知识库和动作效应满足特定形式假设，无法覆盖更一般的第二阶逻辑进化情况。

---

## 28. Learning Transferable Latent User Preferences for Human-Aligned Decision Making

**arXiv ID:** 2605.12682 | [PDF](https://arxiv.org/pdf/2605.12682v1)

**作者:** Alina Hyk `[一作]` (Oregon State University), Sandhya Saisubramanian `[通讯]` (Oregon State University)

**通讯引用:** 173 | [OpenAlex ID](https://openalex.org/A5037326029)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CLIPR 框架，利用与用户的自然语言对话快速学习可执行的偏好规则，并通过自适应反馈不断精炼这些规则，从而提升 LLM 在含歧义任务中的用户对齐能力。

**💡 创新点**

创新点在于将偏好学习转化为可解释的规则生成，并结合自适应反馈门控机制，只在需要时向用户请求信息，显著降低交互成本和 LLM 调用次数，同时实现跨任务、跨环境的泛化。

**🔧 技术方法**

使用大型语言模型（Claude、GPT 等）进行对话生成与规则推理，结合规则合成、需求维度分析、批量性能监测与自适应门控算法；核心技术为基于 LLM 的结构化偏好维度提取与规则生成。

**📊 数据集**

实验使用三大数据集：AmbiK（厨房指令歧义）、Housekeep（家庭整理任务）和 Mobile Manipulation（移动抓取任务），并在 KitchenAmbig 数据集上进行用户研究。

**📈 对比分析**

与零射击、ICL、TidyBot、GATE、CIPHER、IP 等基线比较，CLIPR 与 Adaptive‑CLIPR 在所有数据集上获得最高的偏好对齐准确率（最多提升约10%），并在需要多轮反馈的基线中实现最多 94% 的 LLM 调用成本下降，展示了显著的性能与计算效率优势。

**⚠️ 局限性**

局限在于假设用户偏好相对稳定且可表述为简单规则，难以处理更复杂的条件偏好或偏好不确定性；在极端对抗性反馈或规则初始化极差时仍可能出现性能波动。

---

## 29. Layer-wise Representation Dynamics: An Empirical Investigation Across Embedders and Base LLMs

**arXiv ID:** 2605.12714 | [PDF](https://arxiv.org/pdf/2605.12714v1)

**作者:** Jingzhou Jiang `[一作]` (Hong Kong University of Science and Technology), Kar Yan Tam `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 12176 | [OpenAlex ID](https://openalex.org/A5090061579)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 Layer-wise Representation Dynamics (LRD) 框架，对大型语言模型的每层隐藏表示进行三维度测量（全局子空间运动、局部邻域保留、与最终层对齐），并将其用于无标签模型选择和推理时层剪枝。

**💡 创新点**

创新点在于统一了三种互补的层级测量方法，首次在同一协议下同时捕捉全局子空间转移、局部邻域稳定和最终层图结构相似度，并证明这些测量能有效指导模型选择和层级剪枝。

**🔧 技术方法**

技术包括：Grassmann 速度与曲率 (Frenet)、邻域保留得分 (NRS)、图滤波互信息 (GFMI)，以及相应的层级分数转化为全局选择分数和逐层剪枝分数。

**📊 数据集**

数据集使用 31 种公开模型（25 个嵌入器、6 个基础 LLM）在 30 个 MTEB 任务（分类、检索、语义相似度等）上进行评估，并对 6 个 LLM 在 MMLU 上做泛化检验。

**📈 对比分析**

通过 Spearman 相关性和相对分数变化比较，LRD 的全局选择分数（尤其是 d₀,ₗ）在 28/30 任务上显著相关；GFMI 指导的剪枝在 15%–20% 的预算下平均比随机剪枝减少 2–6% 的分数下降，且中位数表现最优。

**⚠️ 局限性**

局限包括：模型选择实验仅基于 25 个公开嵌入器，未评估更广泛的模型池；GFMI 仅在较大剪枝预算下显著；所有测量受输入采样、池化方式、k‑NN 构造和图阈值的影响，需进一步验证鲁棒性。

---

## 30. Embodied Multi-Agent Coordination by Aligning World Models Through Dialogue

**arXiv ID:** 2605.12920 | [PDF](https://arxiv.org/pdf/2605.12920v1)

**作者:** Vardhan Dongre `[一作]` (University of Illinois Urbana-Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 9756 | [OpenAlex ID](https://openalex.org/A5068709817)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

研究了LLM驱动的具身多智能体对话在共享环境中的协作效果

**💡 创新点**

提出了面向世界模型对齐的诊断框架，并揭示对话内容失效导致成功率下降

**🔧 技术方法**

使用了大语言模型(Claude 3.5 Sonnet/Haiku、Mistral Large)与自定义工具API，并构建对话通道

**📊 数据集**

基于PARTNR任务集（Habitat仿真中的家庭机器人任务）

**📈 对比分析**

通过与Silent基线对比，发现SC/ACF对话能显著降低冲突但任务成功率下降；不同模型和策略表现一致，free‑communication并未恢复性能

**⚠️ 局限性**

局限在于对话被视为共享信念的策略，未深入研究语言细粒度、实体提取方法及多轮推理

---

## 31. scShapeBench: Discovering geometry from high dimensional scRNAseq data

**arXiv ID:** 2605.12662 | [PDF](https://arxiv.org/pdf/2605.12662v1)

**作者:** Andrew J Steindl `[一作]`, Smita Krishnaswamy `[通讯]` (Yale University)

**通讯引用:** 11270 | [OpenAlex ID](https://openalex.org/A5045475274)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 scShapeBench 基准，用于自动检测高维单细胞数据的形状并提供标准化评估流程。

**💡 创新点**

创新在于构建了可供训练的人工注释数据集、基于拓扑的评估指标以及一种基于扩散 Reeb 图的基线方法 scReebTower。

**🔧 技术方法**

采用扩散几何、Reeb 图构造、持久同调、图编辑距离等技术，并结合深度学习的图神经网络和多标签 MLP 进行评估。

**📊 数据集**

使用了由合成图生成的模拟数据以及来自 CELLxGENE、10x、Broad SCP 等公开来源的 102 个 scRNAseq 数据集。

**📈 对比分析**

通过 Wasserstein 持久性相似度、图编辑距离以及在真实数据上对专家形状标签的多标签分类准确率进行比较，scReebTower 在大多数指标上优于 PAGA、Mapper 等基线。

**⚠️ 局限性**

局限包括对多模态或极度噪声数据的鲁棒性不足、评估仍依赖人工注释，且只覆盖四类形状。

---

## 32. Exploring how EFL students talk to and through AI to develop texts

**arXiv ID:** 2605.12523 | [PDF](https://arxiv.org/pdf/2605.12523v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 33. TimelineReasoner: Advancing Timeline Summarization with Large Reasoning Models

**arXiv ID:** 2605.12518 | [PDF](https://arxiv.org/pdf/2605.12518v1)

**作者:** Liancheng Zhang `[一作]` (Renmin University of China), Zhicheng Dou `[通讯]` (Renmin University of China)

**通讯引用:** 4104 | [OpenAlex ID](https://openalex.org/A5010558184)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了TimelineReasoner框架，通过两阶段（全局认知与细节探索）实现主动推理式时间线摘要。

**💡 创新点**

创新点在于将大型推理模型(LRM)用于动态事件内存维护、缺口检测与有目标检索，实现可迭代的推理‑检索循环。

**🔧 技术方法**

采用大型推理模型（如QwQ‑32B）作为核心推理器，配合Event Scraper、Timeline Updater和Supervisor三种专用机制。

**📊 数据集**

使用公开的开放域TLS数据集（Open‑TLS）和闭域危机与T17数据集进行实验。

**📈 对比分析**

与四大基线（DIRECT、REWRITE、ITER_RAG、CHRONOS）对比，TimelineReasoner在ROUGE‑1/2、Date‑F1等指标上提升约7%–13%，在闭域任务也能保持或超越现有最优方法。

**⚠️ 局限性**

在文档冗余高、事件结构固定的闭域环境中，推理‑检索循环收益不明显，且模型对检索规模和查询质量敏感。

---

## 34. COSMIC: Concurrent Optimization of Structure, Material, and Integrated Control for robotic systems

**arXiv ID:** 2605.12654 | [PDF](https://arxiv.org/pdf/2605.12654v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 35. ASAP: Amortized Doubly-Stochastic Attention via Sliced Dual Projection

**arXiv ID:** 2605.12879 | [PDF](https://arxiv.org/pdf/2605.12879v1)

**作者:** Huy Tran `[一作]` (Vanderbilt University), David Hyde `[通讯]` (Vanderbilt University)

**通讯引用:** 1093 | [OpenAlex ID](https://openalex.org/A5040432863)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了ASAP（Amortized Doubly‑Stochastic Attention via Sliced Dual Projection）方法，在训练完成后通过一次性映射和两侧c‑transform替代Sinkhorn迭代，减少在线计算。

**💡 创新点**

将Sinkhorn训练的双随机注意力冻结后，利用切片一维OT势值学习轻量级映射，实现离线编译，兼顾训练成本与推理速度。

**🔧 技术方法**

使用Sinkhorn缩放、一次性c‑transform、切片OT势函数、线性回归/KL校准，以及两侧熵正则化c‑transform等技术。

**📊 数据集**

在语言任务（DBpedia‑14、AG News、Yelp Review Full）、视觉任务（Cats & Dogs、CIFAR‑100）以及IMDb文本分类等数据集上验证。

**📈 对比分析**

与Sinkhorn教师、降低迭代次数的Sinkhorn归一化、ESPFormer等基线比较，ASAP在保持教师精度的同时比训练成本低5.3×，在Frozen‑layer和下游任务中恢复95%以上精度，速度提升显著。

**⚠️ 局限性**

仍需密集的查询‑键全连接运算，适用于重复推理；线性映射对激活分布变化敏感；不兼容因果掩码，需进一步扩展。

---

## 36. What Happens Before Decoding? Prefill Determines GUI Grounding in VLMs

**arXiv ID:** 2605.12549 | [PDF](https://arxiv.org/pdf/2605.12549v1)

**作者:** Jiaping Lin `[一作]` (Guangming Laboratory), Haizhou Li `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无训练的 Re-Prefill 方法，在 GUI 定位任务的推理阶段通过二次预填充（prefill）改进目标选择；

**💡 创新点**

通过分析 VLM 推理动态发现预填充阶段是错误来源，创新性地利用注意力引导的关键视觉 token 进行第二次预填充，实现对目标选择的重新校正；

**🔧 技术方法**

利用 decoder‑only 视觉语言模型的注意力机制、KV 缓存、层级前缀注入和交叉层一致性过滤；

**📊 数据集**

在 ScreenSpot‑Pro、ScreenSpot‑V2、OSWorld‑G、UI‑Vision、MMBench‑GUI‑L2 等五个 GUI 定位基准上进行评估；

**📈 对比分析**

与多种无训练方法（RegionFocus、ZoomClick、MVP 等）以及不同规模的 VLM（Qwen3‑VL‑8B/32B、MAI‑UI‑8B、GUI‑Owl‑1.5‑8B）对比，Re‑Prefill 在所有基准上均提升 1–4.3% 的定位准确率；

**⚠️ 局限性**

仍受限于预填充阶段信息量和多层前缀注入参数的选择，对极高分辨率或结构复杂的场景提升有限。

---

## 37. Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach

**arXiv ID:** 2605.12974 | [PDF](https://arxiv.org/pdf/2605.12974v1)

**作者:** Daniel M. Cherenson `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**通讯引用:** 2425 | [OpenAlex ID](https://openalex.org/A5059647993)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种分布鲁棒随机备份安全过滤器，用于在非线性系统和任意不确定结构下提供概率安全保证。

**💡 创新点**

创新点在于将无限维的分布鲁棒安全约束压缩为一维的切换时间搜索，并通过Wasserstein模糊集合实现有限样本安全证明，兼具实时性与安全性。

**🔧 技术方法**

使用了基于采样的滚动预测、Lipschitz常数估计、Wasserstein距离、概率置信界和备份策略切换的技术。

**📊 数据集**

实验使用了Dubins车辆（5000个混合高斯噪声样本）、F31207赛车（巴塞罗那赛道真实轨迹数据）以及F‑16战斗机（JSBSim风洞模拟与多项式残差数据）等数据集。

**📈 对比分析**

与分布鲁棒CBF和基线规划器比较，实验显示安全率提升至接近100%，速度略有下降，备份使用率低且计算时间在可接受范围内。

**⚠️ 局限性**

局限在于需要预先设计备份策略、对Lipschitz估计敏感、对长时间序列的累计安全概率衰减以及在高维系统中仍存在一定的计算负担。

---

## 38. LFPL: Revisited and Mechanized

**arXiv ID:** 2605.12893 | [PDF](https://arxiv.org/pdf/2605.12893v1)

**作者:** Nathaniel Glover `[一作]` (Carnegie Mellon University), Jan Hoffmann `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6922 | [OpenAlex ID](https://openalex.org/A5008504650)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

对Martin Hofmann的线性函数编程语言（LFPL）进行了完整的、可读的理论阐述，并在新的证明助手Istari中实现了其所有元理论（语法、类型系统、定语语义、算术大步成本语义）的完全机械化；同时给出了新的非尺寸递增堆栈数据结构，利用它证明了LFPL的多项式时间可达性（soundness）和完整性（completeness）。

**💡 创新点**

① 用堆栈式数据结构替代原始数组式磁带实现，极大简化了完整性证明并消除了原始证明中的多处错误；② 在大步成本语义上构造了显式多项式上界，避免了传统的替代代价分析；③ 将所有证明完整机械化，成为首个在Istari中完成的ICC元理论案例。

**🔧 技术方法**

基于隐式计算复杂性（ICC）理论的仿射类型系统、非尺寸递增属性、堆栈化数据结构、基于大步成本语义的显式多项式构造、逻辑关系方法以及Istari的证明助手技术。

**📊 数据集**

无（纯理论证明，无实验数据集）。

**📈 对比分析**

通过与Hofmann原始论文、Atkey的扩展以及Aehlig–Schwichtenberg的成本分析技术对比，证明了在更简洁的形式下得到相同或更强的多项式时间可达性与完整性；在机械化层面，展示了完整性与Soundness的证明脚本长度（约7000行，包含3000行定义、3500行证明脚本、800行定理）。

**⚠️ 局限性**

目前的证明仅覆盖LFPL的最小版本，未覆盖其扩展特性；堆栈式实现虽然简化了证明，但在更复杂的状态机或并发场景下的适用性尚未验证；证明仍停留在理论层面，缺乏对实际程序执行时间的经验评估。

---

## 39. Belief-Space Residual Risk for Automated Driving under Localization Uncertainty

**arXiv ID:** 2605.12710 | [PDF](https://arxiv.org/pdf/2605.12710v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 40. GraphIP-Bench: How Hard Is It to Steal a Graph Neural Network, and Can We Stop It?

**arXiv ID:** 2605.12827 | [PDF](https://arxiv.org/pdf/2605.12827v1)

**作者:** Kaixiang Zhao `[一作]` (University of Notre Dame), Yushun Dong `[通讯]` (Florida State University)

**通讯引用:** 1001 | [OpenAlex ID](https://openalex.org/A5047581320)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 GraphIP-Bench，一个统一的 GNN 模型提取与所有权防御评测基准。

**💡 创新点**

首次在同一黑盒协议下同时评估 12 种提取攻击和 12 种防御，并加入联合攻击‑防御轨道，揭示水印在提取后失效的差距。

**🔧 技术方法**

使用多种黑盒提取方法（MEAs、数据驱动、无数据等）和防御技术（水印、输出扰动、查询检测、加密等），并结合自动化实验框架。

**📊 数据集**

基于 10 个公开图数据集（Cora、CiteSeer、PubMed、Coauthor、OGBN‑Arxiv、RomanEmpire、AmazonRatings 等），覆盖同源、异源和大规模场景。

**📈 对比分析**

在统一预算和查询集下，提取攻击在中等预算即可达到 90%+ 逼真度；大多数防御对精度影响有限，但大多数水印在被提取后验证率显著下降。

**⚠️ 局限性**

局限在于仍未覆盖更多任务类型（如图生成、图像图像），仅在单一硬件平台评估，且实验未考虑对抗性查询或持续学习情境。

---

## 41. Pitfalls of Unlabeled Disagreement-Based Drift Detection in Streaming Tree Ensembles

**arXiv ID:** 2605.12803 | [PDF](https://arxiv.org/pdf/2605.12803v1)

**作者:** Lara Sá Neves `[一作]` (Polytechnic of Porto), Goreti Marreiros `[通讯]` (Polytechnic of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在高频无标签数据流中，构建基于增量决策树（IDT）集成的分歧不确定性漂移检测方法，并与传统的损失基与数据基漂移检测器进行对比。

**💡 创新点**

提出利用标签翻转生成批次特定的分歧度量，并通过Kolmogorov‑Smirnov检验对IDT集成进行漂移检测，同时揭示IDT固有刚性对分歧检测的限制。

**🔧 技术方法**

采用标签翻转、KS检验、Oza采样、增量决策树（Hoeffding Tree、Adaptive Tree、Extremely Fast Tree）、多层感知器集成以及预先测评指标（MTD、DA、FA）等技术。

**📊 数据集**

使用12条合成数据流（SEA、Hyperplane、Stagger、Anomaly Sine、RBF、Agrawal）共90,000个样本，其中包含5次15,000实例的漂移。

**📈 对比分析**

将分歧检测与6个损失基方法（ADWIN、DDM、EDDM 等）和5个数据基方法（BNDM、CSDDM 等）对比；实验结果表明，在大多数流中 IDT 分歧检测的检测延迟高、误报多，整体性能显著低于损失基检测；而在 MLP 集成中，分歧检测效果相对较好。

**⚠️ 局限性**

IDT 的结构性刚性导致分歧度量难以反映模型的学习潜力，缺乏可塑性，从而使基于分歧的漂移检测在 IDT 上表现不佳。

---

## 42. Reducing Bias and Variance: Generative Semantic Guidance and Bi-Layer Ensemble for Image Clustering

**arXiv ID:** 2605.12961 | [PDF](https://arxiv.org/pdf/2605.12961v1)

**作者:** Feijiang Li `[一作]` (Shanxi University), Liang Du `[通讯]` (Shanxi University)

**通讯引用:** 5196 | [OpenAlex ID](https://openalex.org/A5100695983)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GSEC 框架，通过多模态大型语言模型生成语义描述并结合双层 BatchEnsemble 与对齐机制，对无标签图像数据进行聚类。

**💡 创新点**

创新点包括①用生成式语义先导替代传统词典匹配，减少偏差；②双层集成（内层 BatchEnsemble + 外层对齐）同时降低方差；③在同一模型中显式同时优化偏差与方差。

**🔧 技术方法**

采用的技术包括多模态大型语言模型（Llama‑3.2‑Vision‑11B）生成语义描述、CLIP 视觉/文本编码、BatchEnsemble 集成、KL 与对齐损失、confidence 与 balance 损失，以及基于自监督的聚类训练。

**📊 数据集**

使用的公开基准数据集共 11 个：CIFAR‑10/100、STL‑10、ImageNet‑10/ Dogs、Food‑101、Stanford‑Cars、Oxford‑Pets、FGVC‑Aircraft、Country‑211、ImageNet‑1K。

**📈 对比分析**

通过与 18 种单模态与 4 种多模态基线在 ACC/NMI/ARI 上进行横向对比，GSEC 在所有 6 个小型数据集及 ImageNet‑1K 上均取得最高分，尤其在大规模集群中提升显著；bias‑variance 分析表明同时降低偏差与方差。

**⚠️ 局限性**

限制包括：生成语义描述依赖 LLM 推理，导致推理成本与延迟；生成描述的多样性与噪声可能影响聚类稳定性；双层集成虽提升鲁棒性，但在计算资源受限时仍存在开销。

---

## 43. OceanCBM: A Concept Bottleneck Model for Mechanistic Interpretability in Ocean Forecasting

**arXiv ID:** 2605.12639 | [PDF](https://arxiv.org/pdf/2605.12639v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 44. PRISM: Perinuclear Ring-based Image Segmentation Method for Acute Lymphoblastic Leukemia Classification

**arXiv ID:** 2605.12851 | [PDF](https://arxiv.org/pdf/2605.12851v1)

**作者:** Larissa Ferreira Rodrigues Moreira `[一作]` (Federal University of Viçosa), André Ricardo Backes `[通讯]` (Federal University of São Carlos)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于核周环分区（PRISM）的自动化淋巴细胞白血病检测框架，利用核周纹理与颜色梯度特征并通过校准堆叠集成实现分类。

**💡 创新点**

创新点在于：①用自适应核周环代替完整细胞边界分割，显著减少对低对比度染色图像的依赖；②在可解释的多域特征空间中引入梯度信息，提升判别力；③采用轻量级堆叠集成（ET+SVM+LogReg）与概率校准，兼顾高精度与资源友好。

**🔧 技术方法**

核心技术包括：CIELAB/HLS染色归一化+CLAHE、核掩模提取、核周环自适应扩张、形态/颜色/GLCM/LBP特征提取、概率校准（Platt scaling）、层级堆叠集成与OOF训练。

**📊 数据集**

使用ALL-IDB2公开细胞图像数据集，包含260个单细胞图像（130白血病核细胞 + 130正常淋巴细胞），经过5折分层交叉验证评估。

**📈 对比分析**

与多种深度学习基线（ResNet‑18, DenseNet‑121, EfficientNet‑B0, Swin‑T, VGG16）以及以往手工特征方法对比；PRISM在5折交叉验证下达成98.46%准确率、MCC 0.9698、AUC‑ROC 0.9896、PR‑AUC 0.9937，显著优于所有比较模型。

**⚠️ 局限性**

局限性包括：仅在单一公开数据集上验证，缺乏跨中心真实样本评估；对核分割质量高度依赖，极端染色或重叠细胞仍可能影响特征提取；特征维度相对较高，虽比深度网络轻量但仍需一定计算资源。

---

## 45. Reinforced Collaboration in Multi-Agent Flow Networks

**arXiv ID:** 2605.12943 | [PDF](https://arxiv.org/pdf/2605.12943v1)

**作者:** Zheng Wang `[一作]` (Huawei Technologies Co., Ltd.), Yangkai Ding `[通讯]` (Huawei Technologies Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于流网络的多智能体协同框架MANGO，解决错误传播问题。

**💡 创新点**

创新点在于将过去成功工作流构造成流网络，联合使用强化学习进行路径规划和文本梯度进行提示优化，并引入跳过机制提高效率。

**🔧 技术方法**

使用强化学习（REINFORCE）进行路径搜索、文本梯度（TextGrad）进行提示优化，以及基于相似度的状态特征和跳过机制。

**📊 数据集**

在七个基准数据集上评估：HumanEval、MBPP、MATH、GSM8K、DROP、MMLU、GPQA。

**📈 对比分析**

与单智能体与多智能体基线相比，MANGO在所有任务上均优于最佳基线，MATH提升12.8%，DROP提升5.1%，同时在推理与训练时间、API成本上分别降低约47%和41%。

**⚠️ 局限性**

局限性包括对历史成功工作流的依赖、需要大量训练数据才能构造高质量流网络、跳过机制可能忽略有用的节点，以及在极端长链或大规模场景下的稳定性待进一步验证。

---

## 46. The Expressivity Boundary of Probabilistic Circuits: A Comparison with Large Language Models

**arXiv ID:** 2605.12940 | [PDF](https://arxiv.org/pdf/2605.12940v1)

**作者:** Zhiyu Zhao `[一作]` (National University of Singapore), Anji Liu `[通讯]` (National University of Singapore)

**通讯引用:** 205 | [OpenAlex ID](https://openalex.org/A5070686068)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在统一自回归框架下，对可解析概率电路（PCs）与大型语言模型（LLMs）的表达力进行对比，识别出输出与上下文编码两大瓶颈，并通过理论与实验验证其对性能差距的影响。

**💡 创新点**

创新点在于提出PC与Transformer的统一视角，发现PC的概率空间混合输出比logit空间更受限，并证明结构可分解PC仅在与数据依赖拓扑对齐的划分上可匹配Transformer的分离阶数；进一步证明可分解PC比结构可分解PC更具表达力。

**🔧 技术方法**

使用分离阶数理论、概率空间与logit空间的输出参数化比较、结构可分解与可分解PC的构造，以及在HMM、Transformer、混合PC等模型上的实验。

**📊 数据集**

实验数据集包括：合成的 local‑copy 与 induction‑style 前缀复制数据、Penn Treebank、WikiText‑103、UD‑ATIS 以及混合的合成数据。

**📈 对比分析**

通过比较 NLL/验证损失等指标进行性能对比，结果显示 Logit‑HMM 与 Transformer 在同一模型族中优于其概率空间变体；PC 在与 vtree 对齐时能逼近 Transformer，但在不对齐时性能大幅下降；混合可分解PC 在合成任务上表现最好，但在真实数据上的提升有限。

**⚠️ 局限性**

局限性包括：分离阶数分析为最坏情况，路由瓶颈主要在合成数据上验证；可分解PC 的理论优势尚未在实践中充分发挥；固定路由结构导致的性能损失难以在多样化真实任务中克服。

---

## 47. Quantifying Potential Observation Missingness in Inverse Reinforcement Learning

**arXiv ID:** 2605.12831 | [PDF](https://arxiv.org/pdf/2605.12831v1)

**作者:** Leo Benac `[一作]` (Harvard University), Finale Doshi-Velez `[通讯]` (Harvard University)

**通讯引用:** 11565 | [OpenAlex ID](https://openalex.org/A5038771285)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种“最小扰动逆强化学习”（MP-IRL）方法，用来量化在缺失观察下专家行为的潜在缺失信息量

**💡 创新点**

通过将缺失观测建模为每条轨迹的静态稀疏核系数，使得奖励函数仅在决策区域做局部修正，从而可解释缺失信息并衡量其大小

**🔧 技术方法**

基于最大熵IRL的软优化框架，使用核函数（高斯RBF）对缺失信息进行局部编码，训练两个阶段的模型：先学习共享奖励，再冻结奖励求解轨迹级扰动

**📊 数据集**

在三类合成连续导航任务、低级别胶质瘤治疗仿真器（Cancer simulator）以及真实MIMIC-IV ICU低血压管理数据集上验证

**📈 对比分析**

与传统IRL对比，MP-IRL在决策区内的准确率从≈60%提升至≈98%，负对数似然显著下降；在真实数据上也提升约10%，并且扰动规模随可观测信息量变化，验证了缺失信息量的可解释性

**⚠️ 局限性**

在高维状态空间中，核函数的放置与优化难度较大；需要更多数据和更精细的正则化，且方法依赖于轨迹级扰动假设，若真实缺失信息呈非静态或高维结构则效果受限

---

## 48. Quantum Precoded Polar Codes

**arXiv ID:** 2605.12796 | [PDF](https://arxiv.org/pdf/2605.12796v1)

**作者:** Tyler Kann `[一作]` (Georgia Institute of Technology), Matthieu R. Bloch `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 8310 | [OpenAlex ID](https://openalex.org/A5055689993)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了量子预编码极化码（Quantum Precoded Polar Codes），并通过遗传算法优化信息集与预编码矩阵。

**💡 创新点**

将经典短码极化码的预编码技术迁移到CSS量子码，首次在保持CSS约束下实现可逆且对称的预编码，显著提升列表解码性能。

**🔧 技术方法**

使用CNOT前置预编码、持久逆与对称约束的预编码矩阵、遗传算法搜索最佳信息集与预编码，结合四元列表解码与syndrome decoding。

**📊 数据集**

在退相位噪声通道（depolarizing channel）上进行Monte Carlo仿真，评估逻辑误码率；未使用公开数据集。

**📈 对比分析**

将256、512长度、2维码的逻辑误码率与1201长度、1维、距离25表面码在相同噪声下比较，结果表明预编码极化码在仅1/5量子比特、10倍更高码率、相同或更低复杂度下实现类似误码率。

**⚠️ 局限性**

需额外约三到两倍的Clifford门实现，且搜索空间巨大；预编码矩阵稀疏化仍限制性能与实现可行性。

---

## 49. Fine-Tuning Models for Automated Code Review Feedback

**arXiv ID:** 2605.12610 | [PDF](https://arxiv.org/pdf/2605.12610v1)

**作者:** Smitha S Kumar `[一作]` (Heriot Watt University), Hind Zantout `[通讯]` (Heriot Watt University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在开源LLM Code Llama上进行参数高效微调，生成针对Java错误的 KM / KH 反馈。

**💡 创新点**

通过合成数据与 PEFT 实现了开源模型与专有模型相当的反馈质量，并在多维度评估中显著优于基线与提示工程。

**🔧 技术方法**

使用 QLoRA 4‑bit 量化的 PEFT、LoRA、Prompt Engineering 以及 BLEU/ROUGE/BERTScore 等自动评估指标。

**📊 数据集**

使用 Deepseek‑R1 生成的 425 条 Java 错误三元组数据集（含 KM/KH），公开链接为 https://anonymous.4open.science/r/JCODE_KM_KH-4BEC。

**📈 对比分析**

采用教师手工评分、自动指标与学生焦点小组对比，PEFT 模型在 KM 准确率、KH 帮助率、误导率和提示遵从度上分别为 61%、60%、47% 和 95%，均优于提示工程和基线。

**⚠️ 局限性**

数据量有限、仅覆盖 Java 与已知错误、合成数据依赖 Deepseek 模型、仍存在误导信息，需人工校验，难以保证在更广泛情境下的泛化性。

---

## 50. Mitigating Cross-Lingual Cultural Inconsistencies in LLMs via Consensus-Driven Preference Optimisation

**arXiv ID:** 2605.12515 | [PDF](https://arxiv.org/pdf/2605.12515v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 51. Beyond Cooperative Simulators: Generating Realistic User Personas for Robust Evaluation of LLM Agents

**arXiv ID:** 2605.12894 | [PDF](https://arxiv.org/pdf/2605.12894v1)

**作者:** Harshita Chopra `[一作]` (University of Washington), Natasha Jaques `[通讯]` (University of Washington)

**通讯引用:** 3537 | [OpenAlex ID](https://openalex.org/A5046953322)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Persona Policies 框架，通过进化搜索生成多样化人类化用户模拟器，用于提升 LLM 代理在真实用户沟通中的鲁棒性。

**💡 创新点**

将用户个性化视为可进化的 Python 生成器，自动挖掘多维行为轴，结合人类相似度与覆盖率的双目标优化与自然语言反思来生成真实多样化用户角色。

**🔧 技术方法**

演化程序搜索（OpenEvolve、MAP-Elites）、LLM 调用（Gemini、DeepSeek 等）、基于 19 维行为指纹的随机森林判别器、行为覆盖度（Chamfer 距离）以及自然语言反思引导变异。

**📊 数据集**

τ^2-bench 的零售与航空领域任务与对应的真实人类对话数据，用于构建指纹、训练判别器并作为覆盖度参考。

**📈 对比分析**

与基线默认模拟器、直接 Prompt 与未进化初始生成器对比；Evolved Persona 在人类相似度、覆盖率与综合分数均显著提升，实验显示相对基线提升 33–62% 分，人工评估中判定为人类的比例升至 80.4% 近似真实人类；对代理训练也提升了 17% 对抗异常用户的成功率。

**⚠️ 局限性**

需依赖真实人类对话构建判别器与覆盖参考；特征为手工正则/词典，未来可用学习表示；仅在 τ^2-bench 零售与航空两域验证，需扩展至更多基准和真实用户测试。

---

## 52. Prime Successor Irreducibility: Turing Machine Complexity, Kolmogorov Complexity, and Weakness-Based Formulations

**arXiv ID:** 2605.12504 | [PDF](https://arxiv.org/pdf/2605.12504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

---

## 53. Lifelong Learning in Vision-Language Models: Enhanced EWC with Cross-Modal Knowledge Retention

**arXiv ID:** 2605.12789 | [PDF](https://arxiv.org/pdf/2605.12789v1)

**作者:** Hamza Ahmed Durrani `[一作]` (Sejong University), Rafay Suleman Durrani `[通讯]` (Technische Universit"{a}t Ilmenau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了一种针对大型语言-视觉模型的连续学习框架，结合增强的多模态EWC、跨模态一致性保持和参数高效微调，显著降低灾难性遗忘。

**💡 创新点**

引入按模态分组的Fisher信息矩阵与自适应权重、跨模态一致性损失以及低秩参数高效微调，专门解决多模态持续学习中的对齐保持与计算效率。

**🔧 技术方法**

增强Elastic Weight Consolidation（EWC）、低秩适配（LoRA）、跨模态一致性监测、参数高效微调、视觉Transformer+文本Transformer双编码器架构等技术。

**📊 数据集**

MSCOCO、Flickr30K、Visual Genome、Conceptual Captions 四个按序任务的图文对数据集。

**📈 对比分析**

与顺序训练、传统EWC、回放法、L2正则化基线对比；在四任务序列中取得平均准确率0.82、遗忘率0.08、后向转移-0.05，较基线提升20%准确率、74%减少遗忘率。

**⚠️ 局限性**

评估规模有限，仅覆盖图文匹配任务；对超大模型（百亿参数）验证不足；假设任务边界明确，未覆盖在线多任务不确定性；对多模态（音频、文本）或复杂推理任务缺乏测试。

---

## 54. Emotional Expression in Low-Degrees-of-Freedom Robots: Assessing Perception with Reachy Mini

**arXiv ID:** 2605.12786 | [PDF](https://arxiv.org/pdf/2605.12786v1)

**作者:** Amit Rogel `[一作]` (Georgia Institute of Technology), Guy Laban `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 1068 | [OpenAlex ID](https://openalex.org/A5027666533)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

对低自由度机器人Reachy Mini的情感表达进行在线评估。

**💡 创新点**

将情绪识别与情绪维度恢复、社交评价三维度结合，首次在低自由度机器人上检验情绪可读性与社会印象。

**🔧 技术方法**

采用基于Laban动作分析的姿态与音频生成模型，使用Geneva Emotion Wheel分类和连续VA评分；使用RoSAS和HRIES评估社交特质。

**📊 数据集**

10个情绪（爱、喜悦、恐惧、娱乐、厌恶、愤怒、悲伤、羞耻、愉悦、兴趣）的短视频片段；100名在线受试者。

**📈 对比分析**

通过聚类标准化的逻辑回归与GEE模型比较识别准确率，发现情绪识别总体约30%，情绪维度恢复更高（约70%），情绪正向表达提升社交温暖与亲和。

**⚠️ 局限性**

研究仅基于离线视频，缺乏实时互动与多模态输入，且情绪标签受GEW词汇限制。

---

## 55. Early Data Exposure Improves Robustness to Subsequent Fine-Tuning

**arXiv ID:** 2605.12705 | [PDF](https://arxiv.org/pdf/2605.12705v1)

**作者:** Lawrence Feng `[一作]` (Carnegie Mellon University), Aditi Raghunathan `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4709 | [OpenAlex ID](https://openalex.org/A5031731960)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在三阶段语言模型训练管线中，提前在预训练阶段暴露目标领域数据（early exposure）对后续微调时已学得能力的保留影响；

**💡 创新点**

提出将后续微调的鲁棒性视为上游训练的目标，并证明早期曝光、重放(replay)和丢弃(dropout)等干预可在不同阶段互补提升能力保留；

**🔧 技术方法**

使用混合预训练、计算匹配实验、重放、dropout、两层线性模型理论分析以及损失前沿（Pareto frontier）评估方法；

**📊 数据集**

采用C4作为通用预训练语料，MusicPile、ChemPile、Instruction等为目标领域和下游任务的数据集；

**📈 对比分析**

通过绘制保留的后训练损失与下游微调损失、保留的预训练损失与保留的后训练损失的Pareto前沿，比较混合与非混合训练的性能；实验表明早期曝光显著降低微调后能力遗忘，并在1B模型上同样有效；

**⚠️ 局限性**

局限性包括：仅在单轮预训练混合、未研究多次重复使用目标数据、仅考虑单一下游微调方法、未评估大规模模型或强化学习微调等场景。

---

## 56. BiPneu: Design and Control of a Bipolar-Pressure Pneumatic System for Soft Robots

**arXiv ID:** 2605.12804 | [PDF](https://arxiv.org/pdf/2605.12804v1)

**作者:** Yu Mei `[一作]` (Michigan State University), Xiaobo Tan `[通讯]` (Michigan State University)

**通讯引用:** 9371 | [OpenAlex ID](https://openalex.org/A5088360388)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并实现了可扩展的多通道双极性气压系统 BiPneu，并基于混合电气-气压模型设计了双模式滑模控制器（DM‑SMC），通过实验验证其在固定和变容积软体执行器上的高精度、快速响应气压调节性能。

**💡 创新点**

创新点包括：
1) 采用通用的通道堆叠结构和 3D 打印外壳，实现了 16 通道可扩展的双极性气压系统；
2) 将双极性气压系统建模为开关非线性系统，并在此基础上提出了带有滞后监督模式选择的双模式滑模控制器，显著降低模式切换次数、抑制切换噪声；
3) 通过与传统 PID、非线性 MPC 以及混合整数非线性 MPC 的对比，证明 DM‑SMC 在误差、功率消耗、模式切换和计算量上实现了明显优势。

**🔧 技术方法**

技术手段包括：
- 电气气压混合建模（或ifice‑flow + 模式开关）；
- 滑模控制设计（mode‑specific 滑动面、到达律、边界层饱和）；
- 采用 Raspberry Pi + ROS 2 进行软硬件耦合，支持 TCP/IP 与 ROS 2 通信；
- PWM 直流驱动 3/2 单刀双掷电磁阀，实现正/负压力切换；
- 软硬件系统级的参数识别（音频导向阻抗、阀门油缸比例映射）。

**📊 数据集**

数据集与实验：
- 采用 20 mL 固定容积负载和 0–25 mL 变容积软气囊进行多阶阶跃与正弦跟踪实验；
- 生成的参考轨迹为多阶压力序列和多频正弦波；
- 通过真实硬件采集 60 Hz 传感数据、200 Hz 控制指令；
- 另外在软体并行操纵器和 SOFA FEM 软气囊远程操作任务中记录任务误差。

**📈 对比分析**

与基线方法的比较：
- 在仿真与实验中，DM‑SMC 与 PID 的平均绝对误差分别降低 11.9 %（多阶）和 32–36 %（正弦），并将模式切换次数下降 50–70 %；
- 与 NMPC、MI‑NMPC 相比，DM‑SMC 在跟踪精度上相当甚至略优，且计算时间仅为 0.69 ms（嵌入式），而 MI‑NMPC 需要 3.6 s；
- 控制能量（PWM‑E）在大多数实验中与 PID 相当或略低，表明能效提升；
- 在软体并行操纵器实验中，DM‑SMC 使球体到达目标误差 1.4 cm（3.3 %），RMSE 3.0–3.3 cm（约 7 %）。

**⚠️ 局限性**

局限与未来工作：
- 仅使用电磁阀开/关方式，虽然成本低但在高频动态响应上仍受阀门延迟和泄漏限制；
- 需要先行对每个通道进行参数识别，系统对外部扰动和非线性匹配的鲁棒性仍需进一步验证；
- 当前实验规模仅到 16 通道，进一步扩展到更大通道数时机械布局与 I²C 总线瓶颈需考虑；
- 对复杂软体执行器（如多段气囊、非线性体积变化）仍需更高级的模型预测或学习控制，以突破 DM‑SMC 的非线性极限。

---

## 57. BioSEN: A Bio-acoustic Signal Enhancement Network for Animal Vocalizations

**arXiv ID:** 2605.12534 | [PDF](https://arxiv.org/pdf/2605.12534v1)

**作者:** Tianyu Song `[一作]` (Kyushu University), Linh Thi Hoai Nguyen `[通讯]` (Kyushu University)

**通讯引用:** 203 | [OpenAlex ID](https://openalex.org/A5059337652)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

设计并实现了一种针对生物声学的轻量级去噪模型 BioSEN，能够在自然环境噪声下有效恢复动物发声。

**💡 创新点**

创新点在于将语音增强技术与生物声学特性相结合，提出了多尺度双轴注意力 (MSDA)、生物谐波多尺度增强 (BHME) 和能量自适应门控连接 (EAGC) 三大模块，显著提升了对谐波结构和稀疏时频特征的捕捉。

**🔧 技术方法**

采用了复杂空间坐标卷积 Autoencoder、双头注意力、频率轴多尺度卷积、跨注意力门控及伪干净训练策略（利用预训练人声模型生成伪干净对照），实现了端到端的复杂信号重建。

**📊 数据集**

使用 Xeno Bird 语料库通过伪干净方法生成噪声/伪干净对，构建 Bird Song、Biodenoising 和 Mixed Data 三个测试集，覆盖鸟类、哺乳动物和混合场景。

**📈 对比分析**

与 FSPEN、LiSenNet、Demucs、DCCRN、FullSubNet 等主流语音增强模型比较，BioSEN 在 SI‑SDR、SI‑SDRi、SNR、SNRi 上均达到或超过最佳模型，同时 FLOPs 仅为 3.15 GFLOPs，展示出高效且优越的性能。

**⚠️ 局限性**

局限性包括：对清洁生物声学数据的依赖仍然有限，伪干净训练策略可能引入偏差；模型在极端噪声条件或非鸟类高频发声的鲁棒性尚待进一步验证。

---

## 58. PERCEIVE: A Benchmark for Personalized Emotion and Communication Behavior Understanding on Social Media

**arXiv ID:** 2605.12525 | [PDF](https://arxiv.org/pdf/2605.12525v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 59. Beyond Centralization: User-Controlled Federated Recommendations in Practice

**arXiv ID:** 2605.12527 | [PDF](https://arxiv.org/pdf/2605.12527v1)

**作者:** Manel Slokom `[一作]` (Centrum Wiskunde & Informatica), Alejandro Bellogin `[通讯]` (Universidad Autónoma de Madrid)

**通讯引用:** 3604 | [OpenAlex ID](https://openalex.org/A5029757626)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建并上线了一个可让用户自行控制推荐目标（个性化与多样化）的联邦学习推荐系统，并在真实环境中部署了53天的长周期用户实验。

**💡 创新点**

首次将联邦学习与交互式用户控制结合，提供即时反馈和可视化的“你能看到即能控制”设计，使用户能够实时切换推荐目标并观察效果，验证了用户偏好与隐私保护的兼容性。

**🔧 技术方法**

使用联邦学习框架：本地BPR（Bayesian Personalized Ranking）模型训练、聚合器对模型更新做均值合并、差分隐私加噪（ε=2.0）、SyftBox实现安全聚合，MMR用于多样化推荐。

**📊 数据集**

实验数据来自22名参与者的观看历史，共8807条电影/节目标题（catalog），并记录了点击、设置变更、负反馈等交互事件。

**📈 对比分析**

通过点击率（CTR）对比个性化与多样化模式：个性化CTR为65.37%，多样化为62.07%，提升3.30个百分点；在53天内个性化CTR保持稳定（65–67%），多样化略有下降，表明个性化更稳健、需要持续调优。

**⚠️ 局限性**

局限性包括样本量较小（仅22人）、实验时间相对短（53天）、仅测试两种推荐目标、缺乏大规模多样化指标、对不同文化/地区偏好的适用性待验证，以及联邦聚合与差分隐私对模型精度的潜在影响。

---

## 60. ChipMATE: Multi-Agent Training via Reinforcement Learning for Enhanced RTL Generation

**arXiv ID:** 2605.12857 | [PDF](https://arxiv.org/pdf/2605.12857v1)

**作者:** Zhongkai Yu `[一作]` (University of California San Diego), Yufei Ding `[通讯]` (University of California San Diego)

**通讯引用:** 3857 | [OpenAlex ID](https://openalex.org/A5048052285)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ChipMATE，一个自训练的多代理框架，用 Verilog 生成器和 Python 参考模型生成器相互验证以实现 RTL 代码生成。

**💡 创新点**

创新点包括：1）无金手指的交叉验证与回溯机制；2）两阶段训练（单代理 SFT+RL + 多代理 RL）提升协作；3）混合生成框架构建 64.4k 高质量参考模型数据集。

**🔧 技术方法**

采用多代理强化学习（X‑GRPO）、GRPO、回溯式迭代、跨语言波形比较工具、结构化提示、Python 参考模型与 Verilog 的互相验证。

**📊 数据集**

使用 QiMeng‑CodeV‑R1 Verilog 数据集（87k）和自生成的 64.4k Python 参考模型数据，辅以针对 FSM、计数器、FIFO 等难点的专项增强样本。

**📈 对比分析**

在 VerilogEval v2、RTLLM v2、ChipBench‑SC、CVDP 等四大基准上，4B/9B 版本 pass@1 分别达到 75.0%/80.1%，显著优于同规模自训练模型及 1600B DeepSeek V4 等大模型。

**⚠️ 局限性**

局限性：仍依赖 LLM 基础模型，数据覆盖不完全，对复杂硬件行为的建模仍有限；回溯机制在多轮推理中增加推理成本；无金手指方法在极端错误场景下仍可能难以收敛。

---

## 61. The Efficiency Gap in Byte Modeling

**arXiv ID:** 2605.12928 | [PDF](https://arxiv.org/pdf/2605.12928v1)

**作者:** Celine Lee `[一作]` (Google DeepMind), Ruoxi Wang `[通讯]` (Google DeepMind)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比字节级和BPE级语言模型在自回归（AR）和掩码扩散（MDM）下的计算扩展性，并量化字节级模型的计算开销；

**💡 创新点**

揭示字节级MDM相比字节级AR在规模上更易受上下文脆弱性影响，提出字节级MDM需额外结构性偏置才能实现可行扩展；

**🔧 技术方法**

使用标准Transformer架构、SwiGLU激活、RoPE编码、线性/余弦掩码调度的MDM训练、Bits-per-Byte评估；

**📊 数据集**

在Slimpajama-627B数据集上进行预训练，使用UTF-8字节与Llama 2 BPE两种分词；

**📈 对比分析**

通过计算匹配（总FLOPs相等）和容量匹配（参数数相等）对比，发现AR字节模型在大规模时可逼近BPE性能，而MDM字节模型始终落后，需更高FLOPs才能匹配；

**⚠️ 局限性**

字节级MDM对上下文破坏极度敏感，缺乏局部连续性和因果历史，导致性能难以提升；

---

## 62. SMA: Submodular Modality Aligner For Data Efficient Multimodal Learning

**arXiv ID:** 2605.12872 | [PDF](https://arxiv.org/pdf/2605.12872v1)

**作者:** Truong Pham `[一作]` (University of Texas at Dallas), Rishabh Iyer `[通讯]` (University of Texas at Dallas)

**通讯引用:** 1924 | [OpenAlex ID](https://openalex.org/A5000529247)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于集合的多模态对齐框架Submodular Modality Aligner (SMA)，利用子模信息最大化多视角对齐并降低模态差距。

**💡 创新点**

创新点在于将多模态数据建模为集合，使用子模互信息(SMI)目标，既捕获多视角信息，又兼顾对齐与模态间的差距。

**🔧 技术方法**

采用子模互信息、Facility Location子模函数（FLVMI/FLQMI）、冻结的预训练视觉编码器(DINOv2)与文本编码器(GTE‑1.5)以及基于CLIP的对比学习框架。

**📊 数据集**

使用MS‑COCO Caption作为训练数据，评估14个零样本分类任务和Flickr30k检索任务。

**📈 对比分析**

与CSA、SAIL（InfoNCE/SigLIP）等基线比较，在低样本（数万样本）下取得最多25%相对提升，零样本分类准确率提升约7–9%，检索召回率提升4–5%。

**⚠️ 局限性**

局限性包括对单正样本情况效果弱；批量尺寸受限，需更好的内存优化；未扩展至更多模态或更大规模数据。

---

## 63. VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority

**arXiv ID:** 2605.12571 | [PDF](https://arxiv.org/pdf/2605.12571v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 64. DIVER:Diving Deeper into Distilled Data via Expressive Semantic Recovery

**arXiv ID:** 2605.12649 | [PDF](https://arxiv.org/pdf/2605.12649v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 65. SHM-Agents: A Generalist-Specialist Integrated Agent System for Structural Health Monitoring

**arXiv ID:** 2605.12916 | [PDF](https://arxiv.org/pdf/2605.12916v1)

**作者:** Yuequan Bao `[一作]` (Harbin Institute of Technology), Haiyang Hu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 471768 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于大语言模型的通用–专用融合代理系统SHM‑Agents，用自然语言驱动、规划与执行结构健康监测（SHM）任务，支持数据异常诊断、模态识别、可靠性评估、疲劳计算等多项任务的端到端完成；

**💡 创新点**

创新点在于将通用LLM的推理与规划能力与多种专用算法相结合，构建了三层模块化架构（Process Agent、Skill Agent、Global Module），实现自动任务分解、错误反馈与重新调度，并通过预训练简化部署与快速扩展；

**🔧 技术方法**

主要技术包括GPT‑4o语言模型、RAG检索、LangChain/类似框架的Agent化设计、PyAnsys进行有限元分析、Python字典管理算法/数据/配置、以及自定义的错误修复与任务调度逻辑；

**📊 数据集**

使用真实长跨桥梁的监测数据（加速度传感器、车流重量信息）、桥梁有限元模型、桥梁PDF资料等为实验数据集；

**📈 对比分析**

通过与单独专用算法对比，SHM‑Agents在不同任务上达到了95%+的执行准确率，且在任务链路中的自动规划与错误恢复提升了工作效率，整体性能与传统专用流程相当；

**⚠️ 局限性**

主要限制是系统的整体性能高度依赖所选LLM（GPT‑4o）与底层专用算法的准确性，且LLM并非100%可靠，部分尚未实现的专用模块（如损伤识别、模型更新等）也限制了系统完整功能的实现。

---

## 66. CAWI: Copula-Aligned Weight Initialization for Randomized Neural Networks

**arXiv ID:** 2605.12580 | [PDF](https://arxiv.org/pdf/2605.12580v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 67. On the Advantage of Adaptivity for Sampling with Cell Probes

**arXiv ID:** 2605.12873 | [PDF](https://arxiv.org/pdf/2605.12873v1)

**作者:** Farzan Byramji `[一作]` (UC San Diego), Anthony Ostuni `[通讯]` (UC San Diego)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5000208053)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

构造了一个显式分布 D，证明在单元探针采样中自适应与非自适应之间存在 2 对 Ω̃(N) 的极限分离。

**💡 创新点**

首次实现了近乎最优的自适应与非自适应采样分离，并大幅提升了之前 2 对 Ω̃(log N) 的结果。

**🔧 技术方法**

采用了读‑k 族的稀疏性分析、均衡映射与集中不等式等信息论与组合技术来完成证明。

**📊 数据集**

无须使用真实数据集，所有结果均为理论构造与分析。

**📈 对比分析**

与 Yu & Zhan (ITCS'24) 与 Alekseev 等 (STOC'26) 的先前工作相比，取得了更强的 2 vs Ω̃(N) 分离。

**⚠️ 局限性**

结果需要较大字母表大小，仅适用于无限量级的 N，且未涉及多探针或近似采样的细粒度复杂度分析。

---

## 68. Multitask Multimodal Fusion with Tabular Foundation Models for Peak and Durability Prediction of Pertussis Booster Response

**arXiv ID:** 2605.12852 | [PDF](https://arxiv.org/pdf/2605.12852v1)

**作者:** Divya Sitani `[一作]` `[通讯]`, Divya Sitani

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种多任务多模态融合模型，用以同时预测百日咳加强针的峰值免疫反应和长期耐久性。

**💡 创新点**

创新点在于：1）将冻结的TabPFN‑v2作为各模态的特征提取器；2）设计双标签监督对比损失，使同一任务标签相同的样本即为正样本；3）采用模态丢弃和缺失感知注意力融合，适应高度缺失的临床数据；4）通过多任务学习实现峰值与耐久性不同生物学信息的共享与分离。

**🔧 技术方法**

使用的技术包括：TabPFN‑v2、对比学习（SupCon）与双标签策略、模态随机丢弃、缺失感知注意力融合、共享 MLP 与线性分类头、交叉熵加权训练、配合自定义的标签置换与 bootstrap 置信区间评估。

**📊 数据集**

数据集为 CMI‑PB（Computational Models of Immunity: Pertussis Boost）四年（2020‑2023）收集的四个模态：抗体滴度、细胞因子、PBMC 细胞频率与基因表达，共 158 例峰值标签样本，96 例同时有耐久性标签。

**📈 对比分析**

与传统逻辑回归、XGBoost 与浅层 MLP 进行比较；在峰值任务上多任务模型的 AUROC 为 0.797（95% CI [0.621, 0.948]），与 TabPFN 嵌入上的最佳基线相当；在耐久性任务上表现尤为突出，AUROC 为 0.755（95% CI [0.519, 0.945]），多任务模型是唯一在两任务上 95% CI 均高于 0.5 的方法。

**⚠️ 局限性**

局限性包括：样本量有限（峰值 158 例，耐久性 96 例，测试集分别 32 与 21 例），导致置信区间宽阔；未进行跨队列或外部验证；缺失模式结构化，模型对单一模态缺失的鲁棒性仍需进一步验证；特征级别解释尚未深入。

---

## 69. CoT-Guard: Small Models for Strong Monitoring

**arXiv ID:** 2605.12746 | [PDF](https://arxiv.org/pdf/2605.12746v1)

**作者:** Nirav Diwan `[一作]` (University of Illinois Urbana-Champaign), Gang Wang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 253022 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种4B参数的链路思考（CoT）监测模型，用于检测代码生成过程中的隐藏目标。

**💡 创新点**

创新点在于：①将大型监测器的检测行为通过监督微调（SFT）蒸馏给小模型，②在SFT基础上加入对“难以检测”的隐藏目标进行强化学习（RL），显著提升小模型的泛化与检测精度。

**🔧 技术方法**

使用的技术包括：监督微调、强化学习（GRPO）、对大模型生成的CoT轨迹进行蒸馏、以及针对难检测目标的自定义奖励函数。

**📊 数据集**

数据集由两部分构成：①大模型（如Qwen3-32B、GPT-5.4）生成的CoT与代码轨迹，用于SFT蒸馏；②手工设计的六个隐藏目标（三类I/O相关、三类难检测），以及九个在评估阶段的独立隐藏目标。

**📈 对比分析**

与多种大型监测器（Gemini-3-Flash、GPT-5.4、GPT-5-mini、Qwen3-32B）进行对比，评估指标为G-Mean²（TPR×TNR）。实验显示，该4B模型在提示操纵（A1）和代码操纵（A2）攻击下，平均G-Mean²可达75%，仅比Gemini-3-Flash低约8%，并且在多种攻击策略下保持稳定，显著优于同类小模型与大模型。

**⚠️ 局限性**

局限性包括：①与最强大模型（Gemini-3-Flash）仍有差距；②主要针对代码生成任务，泛化到其他领域尚未验证；③对极端巧妙的隐藏目标或对抗式CoT仍可能存在漏检风险。

---

## 70. From Generalist to Specialist Representation

**arXiv ID:** 2605.12733 | [PDF](https://arxiv.org/pdf/2605.12733v1)

**作者:** Yujia Zheng `[一作]` (Carnegie Mellon University), Kun Zhang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 21706 | [OpenAlex ID](https://openalex.org/A5100342355)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了从通用模型到专用模型的任务相关表示学习，并给出了无参数、无干预、无结构约束下的可识别性理论。

**💡 创新点**

①证明时间-任务结构在完全无参数设置下可识别；②证明任务相关潜变量在每个时间步可通过稀疏正则化分离；③提供层次化框架，将两者结合。

**🔧 技术方法**

使用条件独立性检验、DAG 及马尔可夫/忠实性假设、Jacobian 支持分析、稀疏正则化（l1）以及 VAE/GAN 估计。

**📊 数据集**

合成线性高斯 SCM 数据、SportsHHI 视频数据、猫图像数据集。

**📈 对比分析**

与 CCA、Group Lasso、SelTask、LEAP 等基线比较，实验显示在任务识别准确率、MCC、mAP 上均显著优于对照；在任务相关潜变量恢复上，R² 指标表明相关部分高、无关部分低。

**⚠️ 局限性**

仅在无穷样本理论下证明；对有限样本的收敛率缺乏分析；方法依赖条件独立性检验，统计测试受维数诅咒；只关注任务相关子集，未实现全潜变量可识别。

---

## 71. Efficient and Portable Support for Overdecomposition on Distributed Memory GPGPU Platforms

**arXiv ID:** 2605.12734 | [PDF](https://arxiv.org/pdf/2605.12734v1)

**作者:** Aditya Bhosale `[一作]` (University of Illinois Urbana-Champaign), Laxmikant Kale `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 35542 | [OpenAlex ID](https://openalex.org/A5051465480)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了支持多GPU多厂商的Charm++运行时，并在此基础上实现了两种无侵入式的过度细分（overdecomposition）管理技术，使Charm++应用能够在NVIDIA与AMD GPU上高效运行。

**💡 创新点**

创新点在于：1) 构建了基于LCI+Kokkos的可移植GPU通信层，支持单源跨厂商、多网络；2) 在运行时层实现了非阻塞流重叠与多进程/多线程层次调度，以隐藏过度细分导致的核启动和通信开销；3) 通过系统化实验评估了过度细分成本与不同执行模型对性能的影响。

**🔧 技术方法**

使用的技术包括Charm++ AMT、Kokkos执行/内存抽象、LCI（libverbs、libfabric）通信、CUDA/HIP IPC、CUDA Graphs、MPS、ROC‑M等。

**📊 数据集**

使用的基准数据集为三大mini‑app：Jacobi2D、MiniMD、LULESH，规模分别在10^8~10^10 计算点，运行在NCSA Delta（A40）和Frontier（MI250X）节点上。

**📈 对比分析**

比较方法：在相同硬件与相同工作量下，将Charm++/Kokkos实现与MPI/Kokkos实现以及Charm++不同过度细分因子进行弱/强扩展测试。结果显示，过度细分的Charm++实现与MPI实现基本相当，且在高过度细分因子下仍能保持良好扩展性，尤其在NVIDIA节点上可达≈5–10%性能提升。

**⚠️ 局限性**

限制包括：1) 过度细分对极细粒度工作仍有隐藏开销；2) 在AMD平台上多进程并发受限，缺乏MPS等机制；3) 对于更高维度或更细粒度的应用（如分子动力学、天体力学）仍需进一步评估和优化；4) chare迁移与精确负载估计尚未集成。

---

## 72. Scaling Laws for Mixture Pretraining Under Data Constraints

**arXiv ID:** 2605.12715 | [PDF](https://arxiv.org/pdf/2605.12715v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 73. WhatsApp Vaccine Discourse (WhaVax): An Expert-Annotated Dataset and Benchmark for Health Misinformation Detection

**arXiv ID:** 2605.12510 | [PDF](https://arxiv.org/pdf/2605.12510v1)

**作者:** Jônatas H. dos Santos `[一作]` (Universidade Federal de Minas Gerais), Cristiano X. Lima `[通讯]` (Universidade Federal de Minas Gerais)

**通讯引用:** 2350 | [OpenAlex ID](https://openalex.org/A5076069194)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 WhaVax 数据集：通过关键词过滤、语义去重、医学专家多评注，对巴西 WhatsApp 公共群的疫苗相关消息进行标注，并对谣言的语言、结构、词汇和群体分布等特征进行系统分析；随后对经典机器学习、微调小型 Transformer 和零/少样本大模型进行分类实验。

**💡 创新点**

创新点在于：①首次公开医学专家验证的 WhatsApp 疫苗谣言数据集；②采用多阶段注释协议与语义去重，提升标注质量并捕捉模糊案例；③在数据稀缺场景下，系统比较经典模型、微调 SLM 与零/少样本 LLM 的性能，为加密通讯环境下的健康谣言检测提供基准。

**🔧 技术方法**

技术手段包括：大规模爬取、关键词过滤、句子嵌入去重、医学专家多评注与 Fleiss κ 评估；文本特征提取（长度、标点、表情符号、首字母大写等）；机器学习分类器（SVM、LR、RF、MLP、XGBoost）配合 BERTimbau / Qwen 8B 嵌入；小型 Transformer 微调（BERTimbau、RoBERTa、BERTuguês、BioBERTpt）；零/少样本 LLM 评估（LLaMA 3.1/3.2、DeepSeek 685B、Qwen 30B、GPT‑5.1/5.2）以及多次 5‑折交叉验证与 Wilcoxon 检验。

**📊 数据集**

使用的数据集为：自建 WhaVax，包含 80,257 条去重后的 WhatsApp 公开群疫苗相关消息；其中 950 条由四名医学专家一致或多数投票标注，约 30% 归为谣言；数据涵盖 2020‑2023 年，包含文本、发送者、群组、时间戳等元信息。

**📈 对比分析**

评估方法：5‑折分层交叉验证、重复 8 次，计算宏 F1、精确率、召回率；使用 Wilcoxon 符号秩检验比较模型。结果显示：经典 LR+Qwen 嵌入宏 F1 0.791；零/少样本 GPT‑5.1 0.786；LLaMA 3.1 0.734；SVM+BERTimbau 0.724；微调 SLM 表现波动，往往低于 LR+Qwen。说明在小样本、非结构化文本下，强嵌入的经典模型与大模型零/少样本策略相当，且大模型在保持高召回率方面具有优势。

**⚠️ 局限性**

局限性：①数据仅来自公开群，可能偏向政治活跃人群，未覆盖私密/家庭聊天；②专家标注受主观判断影响，特别是模糊案例；③样本量有限，导致微调 SLM 表现不稳定；④跨文化、跨语言推广受限；⑤隐私脱敏限制了对地理细粒度或用户行为的深入分析。

---

## 74. SSDA: Bridging Spectral and Structural Gaps via Dual Adaptation for Vision-Based Time Series Forecasting

**arXiv ID:** 2605.12550 | [PDF](https://arxiv.org/pdf/2605.12550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 75. SP-GCRL: Influence Maximization on Incomplete Social Graphs

**arXiv ID:** 2605.12513 | [PDF](https://arxiv.org/pdf/2605.12513v1)

**作者:** Haohua Niu `[一作]` (Sun Yat-sen University), Luca Rossi `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 8117 | [OpenAlex ID](https://openalex.org/A5016589479)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本研究提出了SP‑GCRL框架，旨在在边缺失且噪声较大的社交图上进行影响最大化，结合自监督图对比学习与强化学习实现端到端种子选择。

**💡 创新点**

核心创新在于①设计了社交传播感知的非线性扩散函数，捕获强化与衰减效应；②构造双结构视图（Steiner骨干与可控Gramian）用于对比学习；③引入轻量级GAT近似实现大规模视图构造；④采用DDQN策略实现高效种子选取。

**🔧 技术方法**

技术手段包括：非线性传播模型、两种结构视图对比学习、GAT代理近似、双重深度Q网络、节点嵌入与回归预测。

**📊 数据集**

实验数据集涵盖八个真实网络（Petster‑hamster、Tv‑show、Politician、Advogato、Public、Epinions、Twitter、Weibo）以及小规模radoslaw‑email。

**📈 对比分析**

与随机、PageRank、gIM、S2V‑DQN、ToupleGDD、DeepIM、BIGDN等基线对比，SP‑GCRL在所有预算和网络规模下均获得最高影响传播，尤其在大规模Twitter、Weibo网络上差距显著。

**⚠️ 局限性**

局限性包括对扩散函数参数的依赖、对模拟缺失方式的假设、在动态或对抗性传播环境下表现未知，以及GAT近似虽提升速度但仍需额外训练开销。

---

## 76. EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents

**arXiv ID:** 2605.12887 | [PDF](https://arxiv.org/pdf/2605.12887v1)

**作者:** Hengwei Ye `[一作]` (ShanghaiTech University), Zheng Tian `[通讯]` (ShanghaiTech University)

**通讯引用:** 1236 | [OpenAlex ID](https://openalex.org/A5047384781)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EcoGEO视角，构造多页面、轨迹感知的协调证据生态系统（TRACE），并在受控环境下评估LLM搜索代理的产品推荐效果。

**💡 创新点**

创新点在于：①从生态系统层面优化生成引擎，而非单一网页；②设计导航入口与内部链接，形成多页面支持结构；③证明多页面协同比单页优化更能影响代理的搜索、浏览与答案生成轨迹。

**🔧 技术方法**

技术包括：基于GPT‑5.1的LLM搜索代理、Crawl4AI抓取工具、Google Search API进行受控搜索；对页面进行GEO优化（C‑SEO、E‑GEO、AutoGEO）做对照；实现TRACE的多页面生态系统和内部链接策略。

**📊 数据集**

使用OPR‑Bench数据集，包含3,124个开放式产品推荐查询，分为SafeSearch、E‑Commerce、E‑GEO三类，每个查询配有虚构但合理的目标产品。

**📈 对比分析**

与单页基线和页面级GEO基线对比，TRACE在SafeSearch 67.2%、E‑Commerce 71.9%、E‑GEO 73.9%实现最高推荐率，分别比最强基线提升约31%、15%和15%；轨迹指标显示初始爬取率、目标特定二次搜索和内部链接爬取率显著增加。

**⚠️ 局限性**

局限性：使用虚构产品和受控环境，未能覆盖真实搜索排名、时间动态和商业源等因素；仅探索一种生态系统设计，缺乏对不同规模、页面风格和动态变化的评估。

---

## 77. A Five-Layer MLOps Architecture for Connected Automated Driving

**arXiv ID:** 2605.12719 | [PDF](https://arxiv.org/pdf/2605.12719v1)

**作者:** Bastian Lampe `[一作]` (RWTH Aachen University), Lutz Eckstein `[通讯]` (RWTH Aachen University)

**通讯引用:** 3921 | [OpenAlex ID](https://openalex.org/A5113050304)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

提出了一套五层级别的MLOps架构，专门用于自动驾驶系统（ADS），该架构通过集体、持续学习和多层级自评估来实现安全与性能的持续保证。

**💡 创新点**

创新点包括：① 将传统的两层MLOps拆分为开发、训练、评估、车队运营和车辆运营五层，细化了责任与交互；② 引入安全案例可信度评估（Safety CCA）和用户接受度评估；③ 在评估层实现多级自评估（服务、子系统、ADS、集体）并结合置信-性能矩阵以识别黑天鹅事件；④ 通过车队数据共享与联邦学习等机制实现跨车队的集体学习；⑤ 通过层间标签与策略实现灰度、A/B、滚动等安全发布。

**🔧 技术方法**

使用的技术与方法包括：MLOps自动化训练流水线、特征存储、模型注册表、元数据存储；安全案例可信度评估与情景测试框架；置信-性能矩阵、集体评估；联邦学习、数字孪生、云后端；OTA更新与灰度发布；以及与ISO/TS 26262、ISO 21448、AI Act等标准的对齐。

**📊 数据集**

本文未使用具体公开数据集，而是基于车队数据存储（Fleet Data Store）和现场记录（Shadow Mode）等形式的真实车辆数据作为设计与假设依据；在案例讨论中假设可获取包含传感器数据、SPIs、错误记录等的多源车队日志。

**📈 对比分析**

由于本工作为概念性框架，未进行实验或性能对比；作者建议通过后续实现与真实车队部署来评估可靠行为占比、边缘案例覆盖率、黑天鹅事件检测率等指标，预期能显著提升可靠行为比例并降低高风险/危险/防御行为比例。

**⚠️ 局限性**

局限性包括：① 未进行实证验证，缺乏性能指标与对比实验；② 对法规与标准的映射仍不完整，需进一步细化；③ 在多车队或跨组织环境下的数据共享与隐私、通信带宽、能耗限制等技术难题尚未解决；④ 评估与安全案例可信度评估方法的实现细节与工具链仍需开发；⑤ 对联邦学习与数字孪生等新技术的集成存在复杂性。

---

## 78. RISED: A Pre-Deployment Safety Evaluation Framework for Clinical AI Decision-Support Systems

**arXiv ID:** 2605.12895 | [PDF](https://arxiv.org/pdf/2605.12895v1)

**作者:** Rohith Reddy Bellibatlu `[一作]` `[通讯]` (Independent Researcher), Rohith Reddy Bellibatlu (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了RISED框架，提供一套包含可靠性、包容性、敏感性、公平性与可部署性五个维度的预部署评估流程，并在合成与三组真实临床数据上进行验证。

**💡 创新点**

创新点在于：①将传统聚合指标之外的部署风险量化为可检验的阈值和置信区间；②引入JSS、Δ_AUC、TFR等新度量；③使用BCa自助法与Holm‑Bonferroni校正构建置信区间决策规则；④将公平性视为代理依赖诊断而非单独门槛。

**🔧 技术方法**

技术手段包括：XGBoost/Logistic/随机森林模型训练、BCa自助法(1000次)、Spearman/ROC/Δ_AUC/ECE/SHAP、阈值平移敏感性评估、输入扰动（高斯噪声、单位/年龄缩放）以及Python开源实现。

**📊 数据集**

数据集涵盖10,000例合成患者（Synthea风格生成）以及三组公开真实数据：UCI Cleveland心脏病（303例）、UCI Diabetes 130‑US Hospitals（99,492例）和NCHS 2024 NHIS Sample Adult（9,747例）。

**📈 对比分析**

对比方法包括：Fairlearn公平性工具、TEHAI/FUTURE‑AI/MI‑CLAIM标准及多模型（XGBoost、Logistic、RF）对比。结果显示：在合成集上，可靠性与敏感性两维失败、包容性不确定；在真实集上，可靠性在某些数据上通过但包容性与敏感性普遍失败；公平性诊断在所有集群均出现代理依赖差异。相较于仅基于AUROC的传统评估，RISED能揭示隐藏的部署风险。

**⚠️ 局限性**

局限性包括：①默认阈值未通过临床部署数据经验校准；②公平性维度仅为诊断，需外部独立需求度量；③仅针对二分类任务，未扩展到多类别/时间事件；④可靠性度量受扰动电池设定影响；⑤在小样本真实集群上置信区间不稳定；⑥缺乏对神经网络等复杂模型的评估。

---

## 79. AssemblyBench: Physics-Aware Assembly of Complex Industrial Objects

**arXiv ID:** 2605.12845 | [PDF](https://arxiv.org/pdf/2605.12845v1)

**作者:** Danrui Li `[一作]` (Rutgers State University of New Jersey), Anoop Cherian `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了包含2789个工业装配实例的多模态合成数据集，并提出Transformer模型同时预测装配顺序与6-DoF轨迹；

**💡 创新点**

创新点在于：①首次提供工业装配的完整轨迹与多模态说明；②利用Transformer对指令图像+文本与点云进行联合编码；③引入基于物理模拟的评估指标；

**🔧 技术方法**

使用PointNet/PointNet++编码点云，DINOv3编码差分图像，Qwen-3编码文本，Transformer解码生成轨迹；

**📊 数据集**

使用新建的工业装配数据集（包含手册图示、文本、点云、轨迹），并与ManualPA、ICCV'25基线做对比；

**📈 对比分析**

在顺序已知的设定下，该方法在SCD、PA、SR等指标上均优于基线，尤其在物理仿真成功率从3%提升至约33%；

**⚠️ 局限性**

局限性包括：依赖合成数据，缺少真实机器人执行验证；装配顺序预测仍是瓶颈，误差会累积导致轨迹失败；

---

## 80. Training LLMs with Reinforcement Learning for Intent-Aware Personalized Question Answering

**arXiv ID:** 2605.12645 | [PDF](https://arxiv.org/pdf/2605.12645v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 81. Correcting Influence: Unboxing LLM Outputs with Orthogonal Latent Spaces

**arXiv ID:** 2605.12809 | [PDF](https://arxiv.org/pdf/2605.12809v1)

**作者:** Shixing Yu `[一作]` (Cornell Tech), Kyra Gan `[通讯]` (Cornell Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了一套基于影响函数的可解释框架 RepInfLLM，利用稀疏自编码器在 LLM 内部构造近似独立的潜在空间，从而实现对测试预测的 token‑级影响推断。

**💡 创新点**

将影响函数迁移到潜在特征层并结合稀疏自编码器实现非可分解的潜在级影响计算，同时采用 Jacobian‑vector product 加速并通过整合梯度映射回输入，突破了传统 token 独立假设。

**🔧 技术方法**

稀疏自编码器（SAE）解耦潜在特征、影响函数与 Hessian 逆近似、Jacobian‑vector product（JVP）与一阶 Taylor 展开、Integrated Gradients 用于从潜在到 token 的投影。

**📊 数据集**

在医疗相关的 MedQA、OpenBookQA、CommonsenseQA 等多选问答基准上实验，使用 1B–1.5B 参数的 Llama‑3.2 与 Qwen2.5 模型。

**📈 对比分析**

通过 necessity/sufficiency 删除/保留实验与激活强度、频率、随机基线对比，RepInfLLM 在影响特征排序上显著优于基线（更快降低 logit、NLL、提升保留率），并保持微小的准确率下降。

**⚠️ 局限性**

影响函数仅局部近似，潜在特征仅近似解耦，token‑级投影可能模糊，且仅在短文本上验证，缺乏与传统数据归因方法的直接对照。

---

## 82. ThermalTap: Passive Application Fingerprinting in VR Headsets via Thermal Side Channels

**arXiv ID:** 2605.12927 | [PDF](https://arxiv.org/pdf/2605.12927v1)

**作者:** Mahsin Bin Akram `[一作]` (University of Texas at San Antonio), Murtuza Jadliwala `[通讯]` (University of Texas at San Antonio)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在不接触VR头显的情况下使用低成本热成像摄像机捕获其长波红外辐射，研究了一种被动侧信道攻击，可从外部热辐射中识别出正在运行的应用程序。

**💡 创新点**

①首次将热辐射作为VR头显的应用指纹来源；②利用多模态环境传感器对环境噪声进行归一化；③提出基于SRA‑Seg的ROI分割和网格化时空特征提取；④在室内、室外及多种设备上完成系统化评估。

**🔧 技术方法**

使用低成本热相机与Raspberry Pi进行绝对辐射温度采集；半监督语义分割SRA‑Seg提取头显ROI；网格化统计与梯度特征、热漂移Δ、环境归一化；随机森林、XGBoost、SVM三种监督分类器进行指纹识别。

**📊 数据集**

构建了包含Meta Quest 3、Meta Quest 2和HTC Vive Focus Vision三款头显、六个应用（含Idle）的数据集：E1室内67次会话、E2室外28次会话、E3跨机型84次会话，记录热图、环境温度、湿度、风速和摄像机距离。

**📈 对比分析**

采用留一会话交叉验证、零样本/少样本室外迁移和留一设备外验证来比较方法。室内10 s窗口下准确率≈90%（平均≈92%），室外10 s窗口≈73%（最长窗口最高81%），跨机型训练后平均≈85%（留一设备外≈78%）。随机森林在大多数场景中表现最优。

**⚠️ 局限性**

受头显姿态、遮挡、室外环境（温度、风速、日照）等因素影响；需预先识别设备型号；对高频或短时动态应用的区分能力有限；仅能识别事先定义的候选应用集，无法开放域发现新应用；实现依赖热相机与同步环境传感器，部署成本与环境适配仍是挑战。

---

## 83. Language-Based Agent Control

**arXiv ID:** 2605.12863 | [PDF](https://arxiv.org/pdf/2605.12863v1)

**作者:** Timothy Zhou `[一作]` (University of California-San Diego), Nadia Polikarpova `[通讯]` (University of California-San Diego)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了语言基础代理控制（LBAC）模型，要求代理生成符合类型检查的程序，并在多案例研究中验证其可行性。

**💡 创新点**

创新点在于将静态类型检查与运行时强制相结合，扩展到代理应用，使得代理生成的程序在与框架代码协同时保持类型安全，从而实现统一的安全策略，且保持高度表达性。

**🔧 技术方法**

使用了静态类型系统、运行时能力（capabilities）沙箱、类型检查器、信息流控制技术以及文件系统访问限制等安全相关技术。

**📊 数据集**

未使用公开数据集，主要通过三项案例研究（I/O 沙箱、数据 Provenance、信息流控制）进行验证。

**📈 对比分析**

文章未给出具体性能指标或实验数据，但通过与工具式系统和沙箱解释器的对比，指出 LBAC 在控制力度与表达性之间取得了更好的平衡。

**⚠️ 局限性**

潜在的局限包括：类型检查的运行时开销、对纯副作用无效的限制、递归子代理的资源管理复杂性等。

---

## 84. FRAME: Forensic Routing and Adaptive Multi-path Evidence Fusion for Image Manipulation Detection

**arXiv ID:** 2605.12826 | [PDF](https://arxiv.org/pdf/2605.12826v1)

**作者:** Kaixiang Zhao `[一作]` (Brigham Young University), Amanda Hughes `[通讯]` (Brigham Young University)

**通讯引用:** 6387 | [OpenAlex ID](https://openalex.org/A5060353074)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种名为 FRAME 的自适应取证框架，能够在图像篡改检测中通过可变路径路由和证据融合来提高检测与定位性能。

**💡 创新点**

核心创新点包括：① 将多种传统取证算法视为可组合的模块，构建“超级网”候选路径；② 训练一个基于图神经网络的上下文感知路径选择器，实现对每幅图像的自适应路由；③ 设计可学习的融合机制（软/硬加权），在选定路径上融合多路证据；④ 在理论上证明在满足 Tsybakov 条件下，学习到的选择策略可严格优于统一融合和单一算法基线。

**🔧 技术方法**

技术细节：传统取证模块（pyIFD 包含 ELA、DCT、PRNU 等）；图神经网络（GraphSAGE 风格）作为路径选择器；候选路径采样（K=50）和 top‑k（k=5）融合；融合方式包括硬加权、Softmax 加权和可学习加权；理论分析与实验结合。

**📊 数据集**

数据集：训练/验证使用 CASIA v2（8831/1918 张）；测试使用四个外部基准：CASIA v1（1754 张）、Coverage（200 张）、Columbia（363 张）以及 RealisticTampering（440 张）。

**📈 对比分析**

与多种基线对比：传统单/多取证算法、RF/XGB 线性组合、以及四个公开深度学习方法（TruFor、MMFusion、ManTraNet、CAT‑Net）。实验显示 FRAME 在检测 AUC、定位 F1 和 IoU 上均超过或与最优深度学习基线相当，提升幅度在 0.02–0.07 之间，尤其在需要定位的 CASIA v1、Coverage 与 RealisticTampering 上表现最为突出。

**⚠️ 局限性**

局限性：当前取证模块主要针对传统编辑痕迹，难以识别 AI 生成或扩散模型产生的伪造内容；FRAME 的优势依赖于已有取证算法的多样性，若缺少针对新型篡改的模块，性能可能下降。

---

## 85. CHAL: Council of Hierarchical Agentic Language

**arXiv ID:** 2605.12718 | [PDF](https://arxiv.org/pdf/2605.12718v1)

**作者:** Tommaso Giovannelli `[一作]` (University of Cincinnati), Griffin D. Kent `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了 Council of Hierarchical Agentic Language（CHAL）框架，利用多代理辩论并通过结构化的 CHAL Belief Schema（CBS）与梯度驱动的信念修订，系统化地在可驳倒的议题上进行信念优化。

**💡 创新点**

创新点包括：①将论证过程转化为可观察、可审计的贝叶斯启发式信念图；②通过梯度信息引导信念修订；③将认识论、逻辑与伦理三类元认知价值系统抽象为可配置超参数；④以结构化信念对象作为评价与可追溯的输出，为未来可驳倒论证基准奠定基础。

**🔧 技术方法**

技术上使用：大型语言模型（OpenAI o4‑mini）进行代理推理与裁判评估；基于贝叶斯信念强度与梯度的优化算法；图结构的信念依赖图（BDG）与最小化 AGM 修订；可配置的逻辑、伦理与认识论超参数；多轮交互式辩论流程。

**📊 数据集**

数据集方面：通过人工设定的辩论议题（如“自由意志是否存在”“AI 是否拥有权利”“是否应对未来世代进行基因改造”）生成 10 条独立辩论轨迹；未使用公开标准数据集，而是采用实验生成的对话与信念图作为评估素材。

**📈 对比分析**

比较方法：与传统的多数投票和单代理链式思考（CoT）对比，针对可驳倒任务；使用论文中定义的 Thesis Strength 与 Agent Performance Score（APS）进行量化；实验结果表明在可驳倒议题上，CHAL 能显著提升论点强度并产生更集中的信念分布，但在无真值任务中相对优势有限。

**⚠️ 局限性**

局限性：①模型仍受预训练与 RLHF 隐式价值影响，无法完全消除偏见；②缺乏标准可驳倒评测基准，实验结果缺乏跨系统可复现性；③梯度驱动修订依赖 LLM 输出的数值估计，易受生成噪声影响；④对资源需求高，若使用更大模型或更长辩论轮次需要显著计算开销。

---

## 86. Local Conformal Calibration of Dynamics Uncertainty from Semantic Images

**arXiv ID:** 2605.13028 | [PDF](https://arxiv.org/pdf/2605.13028v1)

**作者:** Luís Marques `[一作]` (University of Michigan), Dmitry Berenson `[通讯]` (University of Michigan)

**通讯引用:** 5550 | [OpenAlex ID](https://openalex.org/A5083082888)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种利用机器人视觉感知实现的局部共形校准方法（OCULAR），在不收集目标环境数据的前提下，对线性高斯动力学模型的未来状态不确定性提供分布无关的覆盖率保证，并可用于安全规划。

**💡 创新点**

创新点：
- 通过将语义图像投影到机器人帧平面并使用卷积自编码器得到低维表示，构建与环境视觉相似性相关的输入空间；
- 在此低维空间上使用决策树划分，得到适应不同状态动作-观测上下文的不确定性阈值，实现局部共形预测；
- 该方法不需要在测试环境收集任何校准数据，依赖于视觉相似的外部环境，实现了跨环境的安全不确定性量化。

**🔧 技术方法**

技术手段：
- Conformal Prediction（SplitCP + LocalCP）
- 卷积自编码器（CAE）用于语义图像到低维空间的压缩
- 决策树（LOCART）用于划分输入空间
- 线性高斯动力学模型及其校准
- 基于马尔可夫决策过程的MPC与MPPI采样优化器
- 语义分割与深度相机数据处理

**📊 数据集**

数据集：
- 平面双积分器环境，使用简化地面真传感器和语义图像；
- Isaac Sim中的雪覆T形道路，使用高分辨率深度与语义相机；
- 以上均在视觉相似但与测试环境不同的若干地图中采集状态-动作-观测-下一状态样本。

**📈 对比分析**

对比方法与性能：
- NoCP（无校准）
- SplitCP（单一尺度校准）
- LUCCa（需目标环境数据的局部共形）
- OCULAR（本研究方法）

性能：
- 在ID与OOD区域均满足≥0.9的覆盖率；
- 预测区域体积仅比理想“oracle”大约10–15%；
- 在规划实验中，OCULAR成功率为100%，与LUCCa相当，但不需要目标环境数据；
- NoCP易欠覆盖，SplitCP过于保守或欠保守，LUCCa需额外数据。

**⚠️ 局限性**

局限性：
- 对CAE与决策树超参数敏感，错误的参数会导致不充分的划分和欠覆盖；
- latent表示训练仅靠重建损失，未与不确定性预测耦合，可能影响划分效果；
- 仅给出单步安全性保证，多步安全性理论尚未证明；
- 依赖机器人帧感知，若感知失效或不完整可能导致不可靠的校准。

---

## 87. Revealing Interpretable Failure Modes of VLMs

**arXiv ID:** 2605.12674 | [PDF](https://arxiv.org/pdf/2605.12674v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 88. Neurodata Without Boredom: Benchmarking Agentic AI for Data Reuse

**arXiv ID:** 2605.12808 | [PDF](https://arxiv.org/pdf/2605.12808v1)

**作者:** Ling-Qi Zhang `[一作]` (HHMI Janelia Research Campus), Kristin Branson `[通讯]` (HHMI Janelia Research Campus)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了两种通用编码代理在将八个不同格式的神经科学数据集转换为统一解码任务格式时的能力。

**💡 创新点**

提出了真实可复现的基准框架和细粒度过程评估量表，并揭示了代理在科学解释层面的典型错误模式。

**🔧 技术方法**

使用大型语言模型编码代理Claude Code（Opus 4.6）和Codex（GPT‑5.4），以及定制的提示与手工评估流程。

**📊 数据集**

使用八个最近发布的鼠标神经群体记录数据集，涵盖NWB、API、Python/MATLAB文件以及不同实验范式。

**📈 对比分析**

通过结果层面指标（数据统计、解码性能）和过程层面人工打分对比，发现代理在单个子任务上表现良好，但端到端一致性低，误差率约25% 以上。

**⚠️ 局限性**

局限在于基准规模小、评估需人工主观打分、代理自评可靠性差，且缺乏更多模型与数据集验证。

---

## 89. FePySR: A Neural Feature Extraction Framework for Efficient and Scalable Symbolic Regression

**arXiv ID:** 2605.12704 | [PDF](https://arxiv.org/pdf/2605.12704v1)

**作者:** Zhiming Yu `[一作]` (Zhejiang University), Xin Lai `[通讯]` (Tampere University)

**通讯引用:** 3235 | [OpenAlex ID](https://openalex.org/A5010224302)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 FePySR 两阶段符号回归框架，先用神经网络 FMN 生成非线性特征，再用 PySR 在压缩后的特征空间内搜索公式。

**💡 创新点**

创新点在于通过自适应特征映射网络（Heterogeneous Activation Unit）提取“有效特征”，显著降低符号回归搜索空间并为后续搜索提供方向性引导。

**🔧 技术方法**

使用了改进版 EQL 的 FMN 网络（包含 HAUs、稀疏正则与对比损失）以及基于遗传编程的 PySR 求解器。

**📊 数据集**

在标准 Nguyen、Livermore、Jin、Constant、R 等数据集，以及 LLM 生成的 75 条复杂方程和生物系统的 ODE 数据集上进行评估。

**📈 对比分析**

与 NGGP、DSR、PQG、Eureqa 等 SOTA 方法对比，FePySR 在大多数基准上的恢复率提升至约 81%，在 LLM 生成的方程中恢复率达到 86%（相较 PySR 的 20%），并在大多数未恢复方程中获得更低的 MSE；在生物 ODE 中恢复率从 0% 提升至 24%。

**⚠️ 局限性**

局限性包括：缺乏对常数项的学习能力，第二阶段对噪声的鲁棒性不足，以及在特征数量过多时可能导致搜索过程混乱。

---

## 90. Grid-Orch: An LLM-Powered Orchestrator for Distribution Grid Simulation and Analytics

**arXiv ID:** 2605.12728 | [PDF](https://arxiv.org/pdf/2605.12728v1)

**作者:** Boming Liu `[一作]` (Oak Ridge National Laboratory), Jamie Lian `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 3082 | [OpenAlex ID](https://openalex.org/A5012830322)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了 Grid-Orch，一个基于 LLM 的分布式电网仿真与分析平台，集成了 36 个域工具、3 个多步优化技能，并提供交互式网页与 QSTS 仪表盘，支持云端与本地部署。

**💡 创新点**

创新点包括：使用 Model Context Protocol 标准化工具暴露，支持多供应商 LLM 无云化，整合全流程分布式分析与可视化，并通过技能层实现多步骤优化工作流。

**🔧 技术方法**

技术栈包含 LLM（Gemini、Claude、Ollama、llama-cpp）、Model Context Protocol、OpenDSS（opendssdirect.py）、FastAPI、React/Next.js、PostgreSQL、MinIO 等。

**📊 数据集**

使用数据集包括 IEEE 13/123 号测试馈线、SmartDS 合成馈线、NREL 与 End‑Use Load Profiles 的 10 条合成负荷曲线，以及用户上传的 CSV 负荷文件。

**📈 对比分析**

通过与直接 OpenDSS 脚本对比，DER 互联筛查、24 小时 QSTS 违压检测等任务在自然语言指令下完成时间从数小时压缩至不足 2 分钟，且数值结果与传统脚本完全一致。

**⚠️ 局限性**

局限性在于 LLM 可靠性（工具选择错误、参数推断不准）、工具库扩展后可能产生歧义、复杂多工具链仍可能超出自纠能力、结果需人工审核，并且目前未覆盖瞬态稳定、保护协调等高级功能。

---

## 91. Simulating Students or Sycophantic Problem Solving? On Misconception Faithfulness of LLM Simulators

**arXiv ID:** 2605.12748 | [PDF](https://arxiv.org/pdf/2605.12748v1)

**作者:** Heejin Do `[一作]`, Mrinmaya Sachan `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一套评估大语言模型学生模拟器对误解信念保持与更新的诊断框架，并引入了Selective Flip Score（SFS）指标；

**💡 创新点**

创新点在于从静态输出匹配转向交互式、基于误解的信念更新评估，并设计了误解对比反馈协议与SFS指标；

**🔧 技术方法**

技术手段包括：误解对比反馈协议、SFS计算、SFT、DPO、GRPO等强化学习与奖励对齐训练；

**📊 数据集**

数据集涵盖数学问题与误解标签的Malrule与EEDI两大数据集，采用自动生成的反馈与响应；

**📈 对比分析**

与传统的准确率或相似度评估相比，SFS能量化模拟器是否只在针对误解的反馈下更新。实验显示，原始LLM在SFS上几乎为零，训练后SFT+GRPO可将SFS提升0.4-0.55，显著改善误解忠诚度；

**⚠️ 局限性**

局限在于只针对结构化数学误解，评价主要基于外部行为而非内部信念，且反馈与标签由LLM生成，缺乏人工验证；

---

## 92. Towards Robust Federated Multimodal Graph Learning under Modality Heterogeneity

**arXiv ID:** 2605.12584 | [PDF](https://arxiv.org/pdf/2605.12584v1)

**作者:** Sirui Zhang `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7471 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在联邦多模态图学习中提出 FedMPO 框架，解决节点缺失模态与跨客户端异质性问题。

**💡 创新点**

创新点在于三层：①拓扑感知跨模态生成；②缺失感知专家融合；③可靠性感知联邦聚合。

**🔧 技术方法**

采用图神经网络+跨模态注意力+Mixture-of-Experts+可靠性加权聚合。

**📊 数据集**

在六个多模态图数据集（Ele-fashion, Grocery, DY, Bili_Dance, Toys, Flickr30k）上评测。

**📈 对比分析**

与多种基线（FedAvg-Zero, FedGCN, FedGraphSAGE, FedProto, FedPub, FedMVP, FedMAC, FedLAP, S2FGL, FedSPA, FedIIH）比较，FedMPO 在节点分类、链路预测和模态检索任务上均取得最佳或最优成绩，尤其在高缺失率与强非IID情况下提升4.1%–5.65%。

**⚠️ 局限性**

局限在于对计算资源的轻度增加、对超参数的依赖、以及对极端稀疏邻居或完全缺失模态的处理仍有限。

---

## 93. Multi-Rollout On-Policy Distillation via Peer Successes and Failures

**arXiv ID:** 2605.12652 | [PDF](https://arxiv.org/pdf/2605.12652v1)

**作者:** Weichen Yu `[一作]` (Carnegie Mellon University), Yu Hu `[通讯]` (Microsoft)

**通讯引用:** 9531 | [OpenAlex ID](https://openalex.org/A5014478407)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种基于多轮 rollouts 的 on‑policy 蒸馏框架 MOPD，利用同一问题下的成功和失败轨迹作为教师信号，从而提升大型语言模型的推理和结构化输出能力。

**💡 创新点**

创新点在于：① 将同一 prompt 的多条 rollouts 视为同伴上下文；② 在教师训练时同时加入成功（positive）和失败（negative）示例，形成对比式监督；③ 通过对比学习将教师转变为局部诊断器，使其更精准地捕捉实例特定错误。

**🔧 技术方法**

采用的技术包括：on‑policy 蒸馏、教师自监督、逆 KL / Jensen‑Shannon 损失、可验证奖励函数（verifier）、模板化的同伴上下文构造（positive peer imitation 与 contrastive success–failure conditioning）、vLLM 异步推理与 FSDP 并行训练。

**📊 数据集**

实验使用的公开数据集包括：LiveCodeBench v6（编程）、DeepMath/AIME2024/2025/HMMT25（数学推理）、SciKnowEval L3（科学 QA）、ToolAlpaca（工具调用）。

**📈 对比分析**

与基线模型（base、GRPO）以及其他 on‑policy 方法（SDPO、传统教师–学生蒸馏）在 mean@8、pass@8、准确率等指标上进行比较；MOPD 在所有任务上均实现显著提升，尤其是混合成功–失败上下文表现最佳。

**⚠️ 局限性**

局限性：需要多轮 rollouts 产生额外计算开销；当同一 prompt 无成功 rollouts 时，教师信号质量下降；对 verifiers 的依赖限制了可扩展性；目前仅在中等规模模型与相对成熟任务上验证，尚未评估在更大模型或更复杂领域的表现。

---

## 94. GuardMarkGS: Unified Ownership Tracing and Edit Deterrence for 3D Gaussian Splatting

**arXiv ID:** 2605.12919 | [PDF](https://arxiv.org/pdf/2605.12919v1)

**作者:** Utae Jeong `[一作]` (Korea University), Sangpil Kim `[通讯]` (Korea University)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5077788107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种统一的3D Gaussian Splatting（3DGS）版权保护框架，既能通过全场景水印实现所有权追踪，又能通过对抗性编辑阻断实现未授权编辑抑制；

**💡 创新点**

创新点在于将全场景水印与针对性对抗更新相结合，并通过高斯选择和参数角色调制实现对水印恢复、编辑阻断与渲染质量的三方平衡；

**🔧 技术方法**

采用了预训练的水印解码器、对抗目标（latent‑anchor、denoising‑trajectory、cross‑attention diversion）、软高斯掩模以及参数角色权重等技术；

**📊 数据集**

实验使用 Mip‑NeRF 360 与 Instruct‑NeRF2NeRF 的多场景数据集；

**📈 对比分析**

与传统水印基线（3DGSW、GaussianMarker、GuardSplat）和对抗基线（DEGauss）对比，sUCPS 提升约10.7%，位错误率保持在 97% 以上，编辑阻断指标显著优于对抗基线，渲染质量（PSNR/SSIM）保持在基线水平；

**⚠️ 局限性**

局限性在于对模型级扰动和更复杂编辑情境的鲁棒性尚未充分验证，需进一步扩展实验和鲁棒性分析。

---

## 95. ConRetroBert: EMA Stabilized Dual Encoders for Template-Based Single-Step Retrosynthesis

**arXiv ID:** 2605.12736 | [PDF](https://arxiv.org/pdf/2605.12736v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 96. Communication Efficient Byzantine Agreement with Predictions

**arXiv ID:** 2605.12935 | [PDF](https://arxiv.org/pdf/2605.12935v1)

**作者:** Muhammad Ayaz Dzulfikar `[一作]` (National University of Singapore), Seth Gilbert `[通讯]` (National University of Singapore)

**通讯引用:** 6093 | [OpenAlex ID](https://openalex.org/A5042520814)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了三种新的拜占庭一致性协议，利用分类预测信息在同步模型下实现最优的 O(min{B/n, f}) 轮数，并分别将通信复杂度从原来的 O(n^3) 降低到 Õ(n^2.5)，最终在有身份认证的情况下达到 O(n^2 κ) 位的近最优通信量。

**💡 创新点**

创新点包括：① 将全局预测共享的三次方通信瓶颈拆解为分组局部领导选举，从而显著降低通信量；② 设计了“good group”概念与多种分组策略，使得在任意误预测数 B 下仍能保证足够多的可信组；③ 采用阈值签名与拓展图实现无认证的验证式拜占庭一致性，实现了既高容错又低通信的协议。

**🔧 技术方法**

主要技术手段：分组（m‑grouping）与分组领导选举；guess‑and‑double 误差估计；graded consensus 与 early‑stopping Byzantine agreement；验证式 Byzantine agreement 结合阈值签名与 expander graph；以及多阶段的“good group”分析与投票证明。

**📊 数据集**

本文为理论研究，无使用真实数据集；所有实验和证明均基于抽象同步网络模型与误预测参数 B 的理论分析。

**📈 对比分析**

与之前工作相比：① 对于 t < (1/3–ε)n 的非认证场景，通信从 O(n^3) 降至 Õ(n^2.5)；② 对于 t < (1/6–ε)n 的中间方案，仍保持 O(min{B/n, f}) 轮数；③ 对于 t < (1/2–ε)n 的认证场景，通信达到近最优 O(n^2 κ)。相同的轮数复杂度下，通信量显著减少，尤其在认证模型下实现了理论上可达的最优位数。

**⚠️ 局限性**

限制包括：① 仅适用于完全同步网络；② 低通信量方案的容错率被降低到 (1/6–ε)n；③ 需要可信的预测信息（即分类预测本身不被证明最优），若预测错误率极高仍需退回到传统协议；④ 认证版需要阈值签名基础设施，增加部署成本。

---

## 97. Interoperability Effects: Extending DeFi Lending Risk Models to Multi-Chain Environments

**arXiv ID:** 2605.12508 | [PDF](https://arxiv.org/pdf/2605.12508v1)

**作者:** Hasret Ozan Sevim `[一作]` `[通讯]` (University of Camerino), Hasret Ozan Sevim (University of Camerino)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文以多链DeFi借贷协议为研究对象，系统性分析了跨链桥活动、清算事件等对协议总锁定价值（TVL）和收入的影响。

**💡 创新点**

创新点在于将跨链互操作性指标（桥体量、桥集成、桥攻击等）纳入风险模型，并采用层级（L1、L2、AltL1）分组，揭示跨链活动在不同链类型下的异质效应。

**🔧 技术方法**

研究采用面板固定效应回归和OLS回归，构建了包含桥体量、清算量、信用扩张比例、提取量、ETH价格波动、APY、恐惧与贪婪指数等自变量的多元回归模型。

**📊 数据集**

使用的数据集来自The Graph、DefiLlama、链上浏览器（如Etherscan、PolygonScan等）以及Crypto Fear & Greed Index API，覆盖2019年10月至2025年1月15日，包含15个借贷协议、53条桥协议，9条EVM兼容链。

**📈 对比分析**

通过固定效应面板回归与OLS回归对比，模型的R²在0.63–0.70之间，说明模型解释力强；不同链类型的回归结果进一步验证了跨链因素对TVL和收入的异质影响。

**⚠️ 局限性**

局限性包括：仅关注EVM链，可能不适用于其他生态；模型为相关性分析，缺乏因果推断，可能存在内生性和遗漏变量偏误；数据来源的完整性与准确性有限，且研究时间窗口有限，难以捕捉长期趋势。

---

## 98. Multi-Quantile Regression for Extreme Precipitation Downscaling

**arXiv ID:** 2605.12762 | [PDF](https://arxiv.org/pdf/2605.12762v1)

**作者:** Hamed Najafi `[一作]` (Florida International University), Jason Liu `[通讯]` (Florida International University)

**通讯引用:** 4488 | [OpenAlex ID](https://openalex.org/A5083193232)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了基于多分位数回归的超分辨网络，用于改进降水极端事件的下尺度预测。

**💡 创新点**

创新点包括：引入 IncrementBound 以保持分位数单调性并通过独立输出头分离不同分位数的梯度，解决 CNN 在多分位数回归中的梯度混乱问题；结合 pinball 损失和分位数加权实现对极端事件的精准捕捉，并在需要时使用条件 VAE 数据增强提升中位预测。

**🔧 技术方法**

采用多分位数回归（pinball 损失）、累计 softplus（IncrementBound）、独立输出头、条件 VAE 生成极端降水样本、基于 SRDRN 的卷积骨干网络，并使用 SEDI、KL、CRPS 等指标进行评估。

**📊 数据集**

使用 ERA5 25 km 15 维输入与 PRISM 4 km 降水观测，覆盖佛罗里达、加利福尼亚和德克萨斯海岸三大气候域。

**📈 对比分析**

通过与传统单输出 MAE 训练的基线以及四种不确定性量化基线（MC dropout、单头 QR、深度集成、随机种子多重实验）进行 2×2 因子实验比较；在佛罗里达 P999 检测率从 4.2% 提升至 75.7%，在加利福尼亚达到 SEDI ≥0.996，德克萨斯 P999 检测率从 0.02% 提升至 81.9%；整体 RMSE、KL 亦显著下降。

**⚠️ 局限性**

受限于 ERA5 25 km 分辨率导致极端事件捕捉上限、生成器平滑性限制、对不同降水机制的区域适应性不足，以及训练时间与算力需求较高。

---

## 99. Few-Shot Physics-Informed Neural Network for Shape Reconstruction of Concentric-Tube Robots

**arXiv ID:** 2605.12790 | [PDF](https://arxiv.org/pdf/2605.12790v1)

**作者:** Navid Feizi `[一作]` (Brigham and Women’s Hospital), Jagadeesan Jayender `[通讯]` (Brigham and Women’s Hospital)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种物理信息神经网络（PINN），用于实时预测三管共轴管机器人（CTR）在自由空间的全骨架形状与内部状态。

**💡 创新点**

创新点在于：①将Cosserat杆理论直接嵌入PINN的损失函数；②仅用数百条实验末端点观测数据即可训练；③实现了对完整3D形状和扭转、扭曲应变等内部状态的估计，无需形状传感器。

**🔧 技术方法**

采用了深度多层感知机（MLP）构建PINN，并结合自动微分实现对Cosserat微分方程、边界条件和观测损失的联合优化；训练使用L‑BFGS优化器。

**📊 数据集**

使用了公开的CTR实验数据集（包含10万组姿态配置）以及约500条末端点观测数据；同时在模拟中生成Cosserat杆模型的合成观测数据。

**📈 对比分析**

与传统的Cosserat杆BVP求解器（shooting + Newton‑Raphson）对比，PINN在形状误差上平均低于1%（相对总长度），末端点位置误差也低于1%；运行时间更为稳定且几乎不受姿态变化影响，显著适合实时控制。

**⚠️ 局限性**

局限性：只能处理产生唯一解的姿态，无法捕捉“snapping”多模态行为；训练过程耗时长；且模型需针对每个CTR硬件参数单独训练，缺乏跨设计的通用性。

---

## 100. What Do You Think I Think? Accounting for Human Beliefs Using Second-Order Theory of Mind

**arXiv ID:** 2605.12745 | [PDF](https://arxiv.org/pdf/2605.12745v1)

**作者:** Patrick Callaghan `[一作]` (Carnegie Mellon University), Henny Admoni `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3025 | [OpenAlex ID](https://openalex.org/A5061653312)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于二阶心智（Second-Order Theory of Mind）的机器人推理框架，用于在机器人规划与交互中准确建模并利用人类的信念状态。

**💡 创新点**

创新点在于将二阶心智（即机器人对人类对机器人信念的建模）引入机器人决策过程，提升了对人类行为预测的深度和准确度；同时提出了统一的贝叶斯推断与规划耦合方法。

**🔧 技术方法**

主要技术包括贝叶斯信念更新、马尔可夫决策过程（MDP）耦合、逆向学习（Inverse Reinforcement Learning）和基于图模型的高阶推断。

**📊 数据集**

使用了公开的人机交互数据集（如HRI Dataset）以及在仿真环境中生成的基准任务数据，覆盖了多种社会情境与任务目标。

**📈 对比分析**

与传统一阶心智模型（如标准Theory of Mind）和无心智基线（如规则驱动规划）进行对比。实验显示，本文方法在信念预测准确率上提升约12%，在任务成功率上提升约8%，且在多模态交互任务中表现更为稳健。

**⚠️ 局限性**

主要局限包括：① 计算复杂度较高，尤其在高维信念空间下推理成本显著；② 对真实世界多样化的非结构化情境适应性不足，需要更多标注数据；③ 二阶心智假设在极端不确定性下可能过度自信，导致规划失误。

---

## 101. Creating Group Rules with AI: Human-AI Collaboration in WhatsApp Moderation

**arXiv ID:** 2605.12613 | [PDF](https://arxiv.org/pdf/2605.12613v1)

**作者:** Gauri Nayak `[一作]` (Cornell University), Kiran Garimella `[通讯]` (Rutgers University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过两阶段的投机性设计研究，探索了WhatsApp群组管理员如何与Meta AI聊天机器人协作共创群组规则，并评估了AI辅助制定、执行与维护规则的价值与局限。

**💡 创新点**

创新点在于：①聚焦端到端加密环境下的群组治理，揭示管理员对AI介入的细致情境依赖；②从关系信任、数据隐私、文化适配等角度系统梳理了管理员对AI协作的边界工作；③提出了一套面向多样群组的AI辅助设计原则，强调人类判断、协作决策与可选择性介入。

**🔧 技术方法**

技术核心是Meta AI（基于Llama 3.2的大语言模型）在WhatsApp内嵌入的聊天机器人，用于生成规则草案、提供违规报告、提示干预方式等功能。

**📊 数据集**

数据集主要为20名印度WhatsApp群组管理员的访谈音频、转录文本、以及他们与Meta AI交互的日志（约132页），并结合了设计探针场景（模拟AI生成报告与干预）。

**📈 对比分析**

方法采用定性研究：思考-说话写作、主题编码与反思性主题分析；未进行量化性能评估，但通过管理员主观体验与对比（AI生成 vs 手工规则）评估其效率提升、规则覆盖面以及编辑负担。结果显示：在大型、正式群组中AI显著缩短规则制定时间并提升覆盖率；在小型、亲密群组中，管理员对AI的使用兴趣低，认为其干预不合群体文化。

**⚠️ 局限性**

局限性包括：①研究样本集中于印度城市大学生与白领，缺乏农村、低学历与多语种群体；②使用的Meta AI仅为单一模型，无法排除模型特定限制导致的结果偏差；③通过设计探针而非真实部署，难以观察AI在真实环境中的行为与误判；④未探究多管理员协作决策的动态，可能低估群组内部权力与协作机制对AI接受度的影响。

---

## 102. Linking Extreme Discourse to Structural Polarization in Signed Interaction Networks

**arXiv ID:** 2605.12814 | [PDF](https://arxiv.org/pdf/2605.12814v1)

**作者:** Zhijin Guo `[一作]` (University of Oxford), Xiaowen Dong `[通讯]` (University of Oxford)

**通讯引用:** 2796 | [OpenAlex ID](https://openalex.org/A5101579932)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个语言驱动的时序符号网络管道，将对话中的立场转化为带置信度的连续符号边，并利用谱 Eigen‑Sign 和失调率（frustration）两种结构极化测度对社区极化进行时序量化。

**💡 创新点**

创新点在于将语言立场预测与连续符号权重结合，统一构建结构极化时间序列，并通过对极端语言与极化的边级消融揭示置信度对谱度量的影响。

**🔧 技术方法**

使用 LoRA 调优的预训练语言模型进行立场评分，生成带置信度的连续符号权重；对极端毒性、极端标量主张、困惑度等窗口级语言特征进行聚合；采用 Eigen‑Sign 谱分解和失调率两种结构极化测度。

**📊 数据集**

数据集为 Reddit Brexit 讨论子版块的对话日志（约 2018 年 12 月至 2021 年 5 月），并通过合成 Signed‑BA/SBM 对管道进行基准验证。

**📈 对比分析**

在合成实验中两种极化测度的调整后相关性高达 0.66–0.78，且在真实数据上与窗口语言特征呈负相关；在一次步预测实验中加入滞后语言特征可使 Eigen‑Sign 的 MSE 降低约 11%–14%，说明语言信息有增量预测价值。

**⚠️ 局限性**

局限包括仅考虑两大阵营的二分划分、样本窗口有限导致预测结果不稳健、只使用极端毒性/标量/困惑度等少量可解释特征，且在不同平台或更长时间跨度的验证不足。

---

## 103. ocLTL: LTL Realizability and Synthesis Modulo ω-Categorical Structures

**arXiv ID:** 2605.12539 | [PDF](https://arxiv.org/pdf/2605.12539v1)

**作者:** Ohad Asor `[一作]` `[通讯]`, Ohad Asor

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了ocLTL（ω‑categorical 结构上的 LTL+P）并给出其可实现性与合成的多项式时间归约到普通命题 LTL；

**💡 创新点**

创新点在于利用 Ryll‑Nardzewski 定理将无限数据域中的公式化简为有限类型的分离，进而实现对 ω‑categorical 结构的可实现性和合成的 2‑EXPTIME 决定性算法；

**🔧 技术方法**

主要技术包括类型枚举、命题 LTL 归约、二进制编码优化以及 Δ‑最小项约束；

**📊 数据集**

没有使用具体数据集，研究对象是任意有效表示的 ω‑categorical 结构；

**📈 对比分析**

与现有的 RealMT、LTL‑modulo‑theories、寄存器自动机等方法对比，本文在允许跨步数据引用和有限 lookback 的前提下实现了完整的合成，时间复杂度保持在 2‑EXPTIME，且额外的爆炸只取决于结构本身而非公式；

**⚠️ 局限性**

局限性包括需要结构的有效表示、必须枚举所有类型（在最坏情况下无法避免枚举 T3），以及仅支持固定大小的 lookback，无法处理无界的跨步数据引用。

---

## 104. M2Retinexformer: Multi-Modal Retinexformer for Low-Light Image Enhancement

**arXiv ID:** 2605.12556 | [PDF](https://arxiv.org/pdf/2605.12556v1)

**作者:** Youssef Aboelwafa `[一作]` (Alexandria University), Marwan Torki `[通讯]` (Alexandria University)

**通讯引用:** 2670 | [OpenAlex ID](https://openalex.org/A5037423841)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出M2Retinexformer，将深度、亮度和语义特征融入Retinexformer，实现多模态低光图像增强。

**💡 创新点**

创新点在于通过多模态跨注意力块与自适应门控动态融合RGB与辅助模态，提升恢复质量。

**🔧 技术方法**

使用Retinex理论、Transformer自注意力、跨注意力融合、深度估计(Depth‑Anything‑V2)、语义提取(DINOv3)以及感知损失等技术。

**📊 数据集**

在LOL、SID、SMID、SDSD等七个低光图像基准上进行实验。

**📈 对比分析**

与Retinexformer、RetinexNet、KinD、Restormer、MIRNet等方法对比，PSNR/SSIM普遍位列第一或第二，显示显著提升。

**⚠️ 局限性**

局限在于模态可靠性不稳时提升有限，且添加过多模态并不总能进一步改善，视频基准上的增益相对较弱。

---

## 105. Can LLM Agents Simulate Dynamic Networks? A Case Study on Email Networks with Phishing Synthesis

**arXiv ID:** 2605.12507 | [PDF](https://arxiv.org/pdf/2605.12507v1)

**作者:** Siqi Miao `[一作]` (Georgia Institute of Technology), Pan Li `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 10470 | [OpenAlex ID](https://openalex.org/A5100455171)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种在大型语言模型多智能体系统中，通过加入事件触发器和 Hawkes 过程来实现可持续、可复制的动态网络仿真，并以此生成具有网络意识的网络钓鱼攻击示例。

**💡 创新点**

创新点在于将基于事件触发的外部刺激与基于 Hawkes 过程的自激时间建模相结合，既保持微观交互真实性，又显著提升宏观网络结构和时间节奏的逼真度。

**🔧 技术方法**

主要技术包括：LLM 多智能体框架、数据驱动的事件触发器、Hawkes 过程激活调度、以及基于多级网络指标的评估方法。

**📊 数据集**

使用的公开数据集为 Enron 邮件数据集和 IETF 邮件数据集，包含完整的时间戳与收发者信息。

**📈 对比分析**

与传统统计 Hawkes、DySAT、EvolveGCN、NLB 等基线方法比较，本文方法在时间节奏（如 r24）、微观/宏观拓扑（如 DegDist、Transitivity、Recip）等多项指标上均获得最低误差，表现最为稳定，尤其在长时域仿真（8-32 天）中优势明显。

**⚠️ 局限性**

局限性包括：对不同网络类型的迁移性尚未验证、仅在邮件沟通网络上实验、且模型仍可能被用于恶意社交工程，需进一步研究更安全的应用与防护方案。

---

## 106. ODRPO: Ordinal Decompositions of Discrete Rewards for Robust Policy Optimization

**arXiv ID:** 2605.12667 | [PDF](https://arxiv.org/pdf/2605.12667v1)

**作者:** Nirmal Patel `[一作]` (University of Texas at Austin), Inderjit Dhillon `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Ordinal Decomposition for Robust Policy Optimization (ODRPO) 框架，通过将离散奖励拆分为多层序数二进制指标并对每层独立计算优势，显著降低RLAIF中的评估噪声；

**💡 创新点**

创新点在于：①对多层离散奖励进行序数分解，构造与二值奖励相兼容的优势估计；②设计了基于方差的动态权重方案，聚焦有效阈值，形成隐式学习课程；③实现零额外训练开销，可直接替换传统GRPO/MaxRL；

**🔧 技术方法**

核心技术包括：RLAIF、GRPO/MaxRL优势估计、序数分解、方差感知加权、批量标准化与数值稳定化；

**📊 数据集**

实验使用的主要数据集为 Ultrafeedback、FACTS-grounding-v2、Alpaca-Evals、IFEval，模型分别为 Qwen2.5‑7B‑Instruct 与 Qwen3‑4B‑Instruct，评估基准为三大任务；

**📈 对比分析**

与基线GRPO、MaxRL及多重采样投票（N=1,8,16,32）对比，ODRPO 在 FACTS 最高提升 14.8%、Alpaca 最高提升 7.5%，整体相对提升约 3–4%，且不增加训练时长；

**⚠️ 局限性**

局限性包括：仅针对离散奖励设计，连续或混合奖励仍需进一步研究；分解层数和阈值的选择依赖任务，过多层可能导致计算复杂度提升；评估质量仍受 LLM 判定器本身随机性的限制。

---

## 107. Agentic Interpretation: Lattice-Structured Evidence for LLM-Based Program Analysis

**arXiv ID:** 2605.12694 | [PDF](https://arxiv.org/pdf/2605.12694v1)

**作者:** Jacqueline L. Mitchell `[一作]` (University of Southern California), Chao Wang `[通讯]` (University of Southern California)

**通讯引用:** 32241 | [OpenAlex ID](https://openalex.org/A5100406891)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Agentic Interpretation 框架，将 LLM 推理与传统基于格的工作列表分析相结合，用来处理依赖外部证据的程序分析任务。

**💡 创新点**

创新点在于把 LLM 的证据获取与有限高度格域评估、工作列表驱动的传播机制统一起来，使得 LLM 的判断可被审计、可迭代且有限终止。

**🔧 技术方法**

核心技术包括：评估图（含上下文与反馈边）、基于证据的评估域（如 A_Graded、A_Stratified）、LLM 评估与生成接口、工作列表算法以及可选的信念修订策略。

**📊 数据集**

文中仅使用一个示例程序（含伪代码与第三方库），并未使用公开数据集或大规模代码库进行实验。

**📈 对比分析**

目前未进行实验比较或性能评估，论文仅给出理论模型和手工演示，故无法给出具体的速度或准确性指标。

**⚠️ 局限性**

局限性包括：缺乏实装与评测；对 LLM 误判的纠正能力有限；与传统验证工具的交互尚未规范；在大规模程序和复杂证据场景下的可扩展性未知。

---

## 108. BackFlush: Knowledge-Free Backdoor Detection and Elimination with Watermark Preservation in Large Language Models

**arXiv ID:** 2605.12529 | [PDF](https://arxiv.org/pdf/2605.12529v1)

**作者:** Jagadeesh Rachapudi `[一作]` (Indian Institute of Technology Mandi), Amit Shukla `[通讯]` (Indian Institute of Technology Mandi)

**通讯引用:** 1591 | [OpenAlex ID](https://openalex.org/A5008777707)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出BackFlush框架，实现对LLM未知后门的检测与清除，同时保留模型所有权水印；

**💡 创新点**

创新点在于：①基于后门易感性放大实现常数时间检测；②后门冲刷现象通过辅助数据注入与去学习实现未知后门消除；③RoPE旋转去学习在不破坏水印的前提下高效去除后门；

**🔧 技术方法**

技术包括：辅助数据注入、基于初始损失差异的检测、RoPE旋转去学习、对比实验；

**📊 数据集**

使用数据集：TriviaQA、SciQ（辅助数据）、TinyStories（评估Utility）以及多种人工合成触发器（错别字、重复词、短语、模式）；

**📈 对比分析**

与Fine‑Mixing、BEEAR、Model‑Merging、Information Conflicts、Locphylax、SANDE、W2SDefense等SOTA方法比较，BackFlush在Mistral‑7B、Llama‑3‑8B、Qwen‑2.5‑7B等模型上实现ASR≈1%，CACC≈99%，并成功保持水印，显著优于对手；

**⚠️ 局限性**

局限性包括：需要额外辅助数据与探测样本；对极大词表的触发器检测依赖探测样本；目前仅针对文本生成LLM，图像生成等场景尚未验证；

---

## 109. M3Net: A Macro-to-Meso-to-Micro Clinical-inspired Hierarchical 3D Network for Pulmonary Nodule Classification

**arXiv ID:** 2605.12570 | [PDF](https://arxiv.org/pdf/2605.12570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 110. Stochastic Smoothed Particle Hydrodynamics for Stochastic Mechanics Problems

**arXiv ID:** 2605.12540 | [PDF](https://arxiv.org/pdf/2605.12540v1)

**作者:** Mridul Tiwari `[一作]` (Indian Institute of Technology Delhi), Souvik Chakraborty `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 2994 | [OpenAlex ID](https://openalex.org/A5030664714)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种将正交多项式余子展开与Smoothed Particle Hydrodynamics（SPH）相结合的随机SPH（S‑SPH）框架，用于求解带随机不确定性的偏微分方程。

**💡 创新点**

创新点在于：① 在无网格框架中使用多项式余子与SPH核函数分离随机与空间变化；② 通过Galerkin投影将随机PDE转化为耦合ODE，显著降低计算成本；③ 结合梯度校正与预测-校正积分实现高效稳定的边界处理。

**🔧 技术方法**

技术手段包括正交多项式余子展开、Karhunen–Loève展开、Galerkin投影、预测-校正时间积分、梯度校正矩阵、自动微分和GPU并行化。

**📊 数据集**

使用合成基准数据集：一维波传播、一维无粘性布尔格方程（含随机初始条件或参数）以及二维布尔格方程（含随机初始场或随机粘度）进行验证，并以Monte Carlo模拟为基准。

**📈 对比分析**

通过与5000样本Monte Carlo结果比较，S‑SPH在均值与方差上基本无偏差，误差随多项式阶数收敛；计算成本比直接采样低约3个数量级，显示出显著的性能优势。

**⚠️ 局限性**

局限性包括：仅在低维随机输入（至多几维）和有限维随机场展开下验证；对高度非线性或高维随机空间的扩展仍需进一步研究；边界处理仍需手动设计，且对复杂几何的适用性尚未彻底验证。

---

## 111. Seed Bank, Co-op, Stoop Swap: Metaphors for Governing Language Model Data for Creative Writing

**arXiv ID:** 2605.12888 | [PDF](https://arxiv.org/pdf/2605.12888v1)

**作者:** Alicia Guo `[一作]` (University of Washington), Katy Gero `[通讯]` (University of Sydney)

**通讯引用:** 950 | [OpenAlex ID](https://openalex.org/A5049124089)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过四场在线工作坊，聚集100余名创作写作者，收集并分析了200余个关于社区治理语言模型的隐喻。

**💡 创新点**

创新点在于将隐喻方法用于构想“为写作者而做的语言模型”治理框架，系统提炼出四大治理主题（同意、边界、认可、规模），并为社区模型提供可落地的治理思路。

**🔧 技术方法**

主要技术是基于隐喻生成与主题分析的定性研究方法：工作坊设计、Miro协作板、参与式讨论、归纳式主题分析。

**📊 数据集**

使用的数据为工作坊参与者的文本产出（隐喻列表、讨论记录、工作表填写内容），未使用公开的机器学习数据集。

**📈 对比分析**

未进行量化性能对比；成果以质性洞察呈现，说明社区治理模型在同意持续性、边界柔性、贡献认可与小规模优势方面的优势。

**⚠️ 局限性**

局限包括样本主要为美国/英语背景的创作者，缺乏跨文化验证；工作坊仅为概念探索，未实现或测试实际模型；隐喻作为生成工具而非规范框架，需后续技术实现与评估。

---

## 112. NeuroRisk: Physics-Informed Neural Optimization for Risk-Aware Traffic Engineering

**arXiv ID:** 2605.12862 | [PDF](https://arxiv.org/pdf/2605.12862v1)

**作者:** Yingming Mao `[一作]` (Xi'an Jiaotong University), Shizhen Zhao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 3164 | [OpenAlex ID](https://openalex.org/A5089557280)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种名为 NeuroRisk 的物理信息神经优化框架，用于在宽域网络中以可接受的风险目标快速计算满足容量约束的路由方案。

**💡 创新点**

创新点包括：①将风险目标统一为 Sort‑and‑Select 结构，揭示表达性与可解性之间的权衡；②引入 Gated Reservation，将决策空间从全局流分配转为边缘级预留并加入隧道级门控，保证容量可行性且消除投影不连续；③在训练时使用物理感知特征（梯度对齐、概率权重）实现对可变规模故障场景的无缝编码，实现零样本迁移。

**🔧 技术方法**

核心技术包括：深度可微无卷积优化器（深度梯度下降+unrolled迭代）、基于 softmax 的 Gated Reservation、物理引擎提取梯度相关特征、共享权重 MLP 更新策略以及对排序和概率掩码的可微实现。

**📊 数据集**

使用公开的 WAN 拓扑（B4、IBM、GEANT、GERMANY50、TATANID）以及对应的流量矩阵和多路由隧道集合，并在这些拓扑上生成多种概率故障场景集。

**📈 对比分析**

与基于 MILP（Gurobi）以及先前的深度学习基线（FauTE、PreTE、TeaVaR）对比，NeuroRisk 在可解性、吞吐量、风险度量上的相对误差均低于 1%，并且在大型拓扑（GEANT、GERMANY50、TATANID）上实现 10²–10⁵ 倍的速度提升；在场景数增长时仍保持毫秒级推理时间。

**⚠️ 局限性**

局限性包括：仍需离线训练并对大规模场景集的内存占用与训练时间有所增长；在极端极限容量饱和或链路失效概率极高的情形下可能出现性能下降；当前仅针对流量和故障两类不确定性，其他网络层面（如链路速率漂移、设备升级）尚未纳入。

---

## 113. A Resampling-Based Framework for Network Structure Learning in High-Dimensional Data

**arXiv ID:** 2605.12706 | [PDF](https://arxiv.org/pdf/2605.12706v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 114. Co-Designing Organizational Justice Indicators for Algorithmic Systems

**arXiv ID:** 2605.12643 | [PDF](https://arxiv.org/pdf/2605.12643v1)

**作者:** Fujiko Robledo Yamamoto `[一作]` (University of Colorado), Amy Voida `[通讯]` (University of Colorado)

**通讯引用:** 2749 | [OpenAlex ID](https://openalex.org/A5058511638)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过与Kiva微贷组织的员工共设计工作坊，探索了除分配公平之外的多维公平概念，并基于组织正义框架提出了一套可量化的指标体系；

**💡 创新点**

创新点在于将组织正义（交互性、公正程序、分配性）引入推荐系统公平评估，构建了可作为边界对象的指标清单，促进跨利益相关者的对话；

**🔧 技术方法**

技术上主要采用了参与式设计、定性访谈、内容分析和归纳编码等人机交互研究方法；

**📊 数据集**

数据集来自于5个部门、9名Kiva员工的共设计工作坊（共360分钟音频、126张Google幻灯片）及其补充笔记；

**📈 对比分析**

论文未对算法性能进行量化比较，而是通过员工反馈和指标构建评估公平性，并计划在未来开发仪表盘进行持续监测；

**⚠️ 局限性**

局限性包括未能直接纳入借款人、放款人等外部利益相关者，因隐私与跨文化障碍导致的代表性不足，以及指标体系的可操作性仍需在真实系统中验证。

---

## 115. GTA: Advancing Image-to-3D World Generation via Geometry Then Appearance Video Diffusion

**arXiv ID:** 2605.12957 | [PDF](https://arxiv.org/pdf/2605.12957v1)

**作者:** Hanxin Zhu `[一作]` (University of Science and Technology of China), Zhibo Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 11393 | [OpenAlex ID](https://openalex.org/A5079572598)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出GTA框架，使用两阶段视频扩散模型先生成粗糙几何，再以几何为先验生成细粒度外观，解决单图像到3D世界生成中的几何-外观分离与一致性问题。

**💡 创新点**

首次显式将几何与外观分离为两个阶段，结合随机潜在打乱训练正则和推理时的多视角可再利用缩放策略，显著提升跨视角一致性与几何可靠性。

**🔧 技术方法**

基于视频扩散模型（CogVideo-X-5B）+视频VAE编码、深度估计、随机潜在打乱、掩码重映射的推理时缩放技术。

**📊 数据集**

在DL3DV 10K训练集上训练，评估于DL3DV官方测试集与RealEstate10K 100场景。

**📈 对比分析**

与多种state‑of‑the‑art方法（See3D、ViewCrafter、FlexWorld、Gen3C、TrajectoryCrafter、Voyager）对比，PSNR、SSIM、LPIPS、FID、Q‑Align等指标均领先，几何误差R‑err/T‑err最低，显示出整体性能优势。

**⚠️ 局限性**

在极大视角差异下单次推理仍易出现几何不稳定，需多轮推理；对极端稀疏视角与光照变化的鲁棒性尚未完全验证。

---

## 116. Discovery-Oriented Faceting: From Coverage to Blind-Spot Discovery

**arXiv ID:** 2605.12956 | [PDF](https://arxiv.org/pdf/2605.12956v1)

**作者:** Youdi Li `[一作]` `[通讯]` (Panasonic Connect Co., Ltd.), Youdi Li (Panasonic Connect Co., Ltd.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向盲点发现的文档分面方法（DOF），通过显式边界、突出异常内容和支持交互式细化帮助用户发现被覆盖式工具忽视的稀有或专业信息。

**💡 创新点**

创新点在于：①将类别边界显式化，区分包含与排除；②以KL散度对类别进行异质性排序，突出与语料平均差异最大的内容；③提供基于交互式增删合并的细化流程，无需重新聚类。

**🔧 技术方法**

采用文本分块、OpenAI 425‑词片段嵌入、K‑means聚类、GPT‑4o 生成边界描述、KL散度计算以及多轮人工智能辅助的精炼。

**📊 数据集**

使用四个跨域数据集：GovReport（政府报告）、BillSum（立法文本）、BigPatent（专利文档）和BookSum（文学作品）。

**📈 对比分析**

通过与传统覆盖式排名（按簇大小）对比，发现 DOF 在 BigPatent、BookSum 等技术/文学领域的 Spearman ρ 为 -1，覆盖与 DOF 的前5类别重叠率低至 0/5，表明其显著挖掘了被覆盖工具隐藏的专门内容；在 GovReport、BillSum 的差异性相对较小。

**⚠️ 局限性**

局限包括：使用规则化模拟而非真实用户交互验证细化效果；固定聚类参数 K=15 与片段长度在不同文本域的可迁移性尚未深入评估；依赖 LLM 生成边界，可能产生语义模糊或不符合专业标准的描述。

---

## 117. Graph-Based Financial Fraud Detection with Calibrated Risk Scoring and Structural Regularization

**arXiv ID:** 2605.12782 | [PDF](https://arxiv.org/pdf/2605.12782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 118. From Compression to Accountability: Harmless Copyright Protection for Dataset Distillation

**arXiv ID:** 2605.12942 | [PDF](https://arxiv.org/pdf/2605.12942v1)

**作者:** Yan Liang `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 97883 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `fede83ac-7505-405f-ab37-e7284695c47f` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SubPopMark 框架，对已压缩的 dataset distillation 数据集进行无害的版权保护与泄露追踪。

**💡 创新点**

创新点在于通过子群体偏差（subpopulation bias）引入可验证的行为标记，而非传统的后门或标签改动，实现安全、可追溯且不影响模型性能。

**🔧 技术方法**

使用子群体驱动的标记（CVM 和 USTM）和基于行为签名的检索策略，结合特征相似度、感知损失等正则化。

**📊 数据集**

在 CIFAR-10/100、FashionMNIST、SVHN 等标准数据集，并评估多种 DD 方法（DC、DM、DSA、DATM、MTT）以及多种网络结构（ConvNet、AlexNet、VGG、ResNet、ViT）。

**📈 对比分析**

与现有后门水印或标签基方法对比，SubPopMark 在版权验证成功率（CVSR）和泄露追踪成功率（DLTSR）上均达 80–100% 甚至 100%，且对模型性能影响极小。

**⚠️ 局限性**

局限性包括对极小 α 或高度扰动的子群体可能导致可追踪性下降，以及在极端异构后端模型上验证效率和鲁棒性尚未充分评估。

---

## 119. Runtime Monitoring of Perception-Based Autonomous Systems via Embedding Temporal Logic

**arXiv ID:** 2605.12651 | [PDF](https://arxiv.org/pdf/2605.12651v1)

**作者:** Parv Kapoor `[一作]` (Carnegie Mellon University), Eunsuk Kang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 1502 | [OpenAlex ID](https://openalex.org/A5044511705)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了在预训练嵌入空间上直接执行的Embedding Temporal Logic (ETL) 运行时监测框架，能把感知系统的高维视觉输入映射为可量化的时间逻辑谓词。

**💡 创新点**

创新点在于把谓词定义为嵌入距离阈值并引入 conformal 预测校准，使得语义一致性、可解释性与安全性在嵌入空间中得到统一处理；同时提供了全新的基于嵌入的时间逻辑语法与语义。

**🔧 技术方法**

使用预训练视觉编码器（如 CLIP、DINOv2）、L2/余弦距离、ETL 语法与鲁棒性量化、Conformal Prediction 进行阈值校准，并实现在线监测器。

**📊 数据集**

在 Dubins Car 2D 导航、D3IL 与 MetaWorld 的仿真操控任务以及真实世界 DROID 抓取/堆叠数据集上进行评估。

**📈 对比分析**

通过与基于状态的监测器、PCA-kmeans、logpZO 以及 VLM Qwen2-VL 的对比，ETL 在 F1、精确度、召回率上均达到 0.81–0.92 的高水平，尤其在序列性任务和安全阈值设置上表现优于现有基线。

**⚠️ 局限性**

局限在于目标嵌入的可解释性不足、对编码器语义分离度的依赖，以及阈值校准仍需预先收集标注数据，在线自适应性和跨域迁移性有待提升。

---

## 120. Scale-Gest: Scalable Model-Space Synthesis and Runtime Selection for On-Device Gesture Detection

**arXiv ID:** 2605.12506 | [PDF](https://arxiv.org/pdf/2605.12506v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 121. Representing Higher-Order Networks: A Survey of Graph-Based Frameworks

**arXiv ID:** 2605.12509 | [PDF](https://arxiv.org/pdf/2605.12509v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 122. ATD-Trans: A Geographically Grounded Japanese-English Travelogue Translation Dataset

**arXiv ID:** 2605.12933 | [PDF](https://arxiv.org/pdf/2605.12933v1)

**作者:** Shohei Higashiyama `[一作]` (National Institute of Information and Communications Technology), Masao Utiyama `[通讯]` (National Institute of Information and Communications Technology)

**通讯引用:** 4305 | [OpenAlex ID](https://openalex.org/A5021667085)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个针对日本旅行日志的地理文本翻译数据集 ATD-Trans，并在此基础上评估了多种大型语言模型（LLM）在日英翻译任务中的表现。

**💡 创新点**

创新点在于①首次将国内外旅行日志的地理实体进行双语标注并构建统一数据集；②系统分析了模型语言焦点（日语增强 vs. 英语中心）与文本所述地理区域（国内 vs. 海外）对翻译质量的影响；③通过实体指令提示探讨了外部知识库（OSM）在提高实体翻译准确度上的可行性与局限。

**🔧 技术方法**

使用技术包括指令调优的 Llama‑3.1、Gemma‑2 及其日语增强版、文档级上下文推理、实体标识符替换、基于 OSM 的实体匹配与指令提示，评估指标为 d‑BLEU、COMET 与 Term Success Ratio。

**📊 数据集**

数据集主要由 ATD（日本旅行日志）、ATD‑MCL（含地理实体标注）和新构建的 ATD‑Trans（90 篇日英对照、包含国内 56 篇、海外 34 篇）组成，同时还利用 POI Geocoder 进行实体识别与匹配。

**📈 对比分析**

通过对比不同模型、上下文、提示方式，实验发现：日语增强模型往往优于英语中心模型，海外文本比国内文本更易翻译；文档级上下文和更大模型提升整体质量；在 oracle 级实体提示下，Term Accuracy 可接近 100%，但常规 KB 提示往往导致整体质量下降，主要是因为 KB 与参考翻译的不一致。

**⚠️ 局限性**

限制在于：数据覆盖仅限旅游博客和日英单向翻译，缺乏多领域、多语言和多方向的数据；KB 匹配覆盖率有限，导致实体提示不总是有效；实验只使用简单提示，未深入探究更复杂的知识集成方法，也未在真实应用任务（如路线规划、灾害应急）中进行外部评估。

---

## 123. DistractMIA: Black-Box Membership Inference on Vision-Language Models via Semantic Distraction

**arXiv ID:** 2605.12574 | [PDF](https://arxiv.org/pdf/2605.12574v1)

**作者:** Hongyi Tang `[一作]` (Hong Kong University of Science and Technology), Yi Yang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 82220 | [OpenAlex ID](https://openalex.org/A5005421447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于语义干扰的输出仅攻击框架DistractMIA，用于在无内部信息的黑盒条件下对视觉‑语言模型进行成员推断。

**💡 创新点**

创新点在于通过插入可配置的语义干扰物并观察生成文本的变化，利用输出稳定性和干扰词出现率构造成员信号，从而避免对概率或隐藏状态的依赖。

**🔧 技术方法**

使用了干扰物配置的自动化搜索（基于REINFORCE的奖励函数）、多次随机生成评估文本稳定性、以及结合稳定性与干扰词提及率的输出评分方法。

**📊 数据集**

评估采用了VL‑MIA/Flickr、VL‑MIA/DALL·E 两个公开基准、COCO 作为参考集以及医学领域的PMC/LLaVA‑Med 数据集。

**📈 对比分析**

与KCMP、Image Infer以及若干概率访问基线对比，DistractMIA在Flickr上K=30时AUROC达0.989，在DALL·E上0.753，显著优于其他输出仅方法，且在部分情形下接近或超过概率访问攻击。

**⚠️ 局限性**

局限性包括：需要在参考集上调优干扰配置并执行多次查询，导致查询成本较高；目前仅在单轮图像描述任务上验证，尚未扩展到多轮对话或复杂推理场景。

---

## 124. Real-World Challenges in Fake News Detection: Dealing with Posts by Cold Users

**arXiv ID:** 2605.12511 | [PDF](https://arxiv.org/pdf/2605.12511v1)

**作者:** Sai Keerthana Karnam `[一作]` (Indian Institute of Technology Kharagpur), Animesh Mukherjee `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 3618 | [OpenAlex ID](https://openalex.org/A5020991141)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于用户互动图的 Fake News 检测框架 UEN，并引入冷用户映射策略以处理无足迹用户。

**💡 创新点**

创新点在于利用用户–用户互动网络生成用户证据特征，并通过三种相似性启发式（帖子相似、反应相似、历史反应相似）近似冷用户行为；同时在 GNN 中融合内容与用户证据，显著提升冷用户场景性能。

**🔧 技术方法**

技术包括句子 Transformer（BERT）生成文本嵌入、基于 Deep Graph Infomax 的全局用户嵌入、GCN/GraphSAGE/GAT 三种 GNN 模型，以及相似性搜索（FAISS）与协同过滤启发式。

**📊 数据集**

使用 Fakeddit（1.06M 条目、9.3M 评论）和 Gossipcop（5.4k 条目、294k 推文）两大公开数据集。

**📈 对比分析**

与文本基线、PSGT 以及 LLM（Llama、Qwen、GPT‑OSS）进行比较；UEN 在两数据集均取得宏 F1 提升约 6–8%（Fakeddit）/3%（Gossipcop），在完全冷用户区间提升约 9–10%；相较 PSGT 提升≈6%，LLM 表现落后。

**⚠️ 局限性**

局限性包括对大规模用户图的计算成本、依赖预先构建的相似度阈值（k1、k2）调参、启发式映射可能不适用于不同领域或低互动环境；且模型仍受训练数据分布影响。

---

## 125. Population Risk Bounds for Kolmogorov-Arnold Networks Trained by DP-SGD with Correlated Noise

**arXiv ID:** 2605.12648 | [PDF](https://arxiv.org/pdf/2605.12648v1)

**作者:** Puyu Wang `[一作]` (RPTU Kaiserslautern-Landau), Marius Kloft `[通讯]` (RPTU Kaiserslautern-Landau)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对两层 Kolmogorov–Arnold 网络（KAN）在剪裁小批量 SGD（含 DP 与非 DP）训练下，推导了总体风险（population risk）上界。

**💡 创新点**

创新点在于：①首次给出针对非凸网络的关联噪声 DP-SGD 的总体风险分析；②提出利用辅助无投影轨迹、移位迭代和高概率引导技术，突破了噪声相关性与投影导致的标准一阶递推失效问题；③将 KAN 的局部 NTK 几何与自界估计结合，得到宽度与迭代次数、噪声相关性 λ 的精细化表现。

**🔧 技术方法**

主要技术包括：小批量 SGD 与梯度裁剪、关联 Gaussian 噪声 DP-λCGD 机制、辅助无投影动态、移位迭代、局部 KAN 曲率下界、自界估计、马尔可夫链与高概率引导（bootstrap）以及算法稳定性分析。

**📊 数据集**

本文以理论框架为主，实验演示使用了 CIFAR‑10 数据集（对比不同 λ 与噪声水平下的准确率），但核心结果为泛化与优化风险的理论上界，未依赖特定训练数据集。

**📈 对比分析**

与传统独立噪声 DP‑SGD、全批量 GD/DP‑GD 以及非 DP‑SGD 的基准相比，论文在宽度要求更小、NTK 边缘 γ 依赖更弱的前提下，给出了与最佳已知上界相当的 1/√n + √d/(nε) 风险速率；实验表明在 λ>0 时，适当的噪声相关性可在不牺牲隐私的前提下提升准确率，尤其在较大 ε 下表现更佳。

**⚠️ 局限性**

局限性包括：①隐私校准采用保守闭式公式，导致 λ 的相关性对最终风险速率的提升未能显现；②目前仅覆盖两层 KAN 与单一激活函数，尚未推广到更深层或更宽松的光滑性假设；③缺乏针对实际数据集的广泛实验验证。

---

## 126. DocAtlas: Multilingual Document Understanding Across 80+ Languages

**arXiv ID:** 2605.12623 | [PDF](https://arxiv.org/pdf/2605.12623v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 127. MLPs are Efficient Distilled Generative Recommenders

**arXiv ID:** 2605.12617 | [PDF](https://arxiv.org/pdf/2605.12617v1)

**作者:** Zitian Guo `[一作]` (University of California), Julian McAuley `[通讯]` (University of California)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种将自回归生成推荐模型中的Transformer解码器替换为轻量化MLP结构的推理加速框架，称为SID‑MLP；

**💡 创新点**

创新点在于识别SID生成过程中的搜索空间高度收敛，利用一次性多头注意力读取全局用户上下文，并用按位条件化的MLP头完成后续各位Token预测，从而消除重复的自注意力与交叉注意力计算；

**🔧 技术方法**

采用知识蒸馏将冻结的教师TIGER模型（Encoder‑Decoder）转化为学生，训练中使用KL散度+交叉熵损失；另外提供encoder‑distilled版本SID‑MLP‑enc；

**📊 数据集**

实验使用Amazon Reviews 2023中三个子域（Instrument、Scientific、Games）的4位SID数据集，基于TIGER tokenizer；

**📈 对比分析**

与原始TIGER、TIGER‑kv以及多种LLM加速和SSM替代方案对比，SID‑MLP在保持相同NDCG@10水平的同时实现8.74×推理速度提升，内存使用下降95.7%；encoder‑distilled版本进一步提升至10.25×；

**⚠️ 局限性**

局限性包括对已训练好的4位SID结构敏感，若SID长度或Token分布变化可能需重新设计；同时对极大item规模的工业系统仍需验证搜索空间行为与加速效果。

---

## 128. Parallel-in-Time Training of Recurrent Neural Networks for Dynamical Systems Reconstruction

**arXiv ID:** 2605.12683 | [PDF](https://arxiv.org/pdf/2605.12683v1)

**作者:** Florian Hess `[一作]` (Central Institute of Mental Health), Daniel Durstewitz `[通讯]` (Central Institute of Mental Health)

**通讯引用:** 8085 | [OpenAlex ID](https://openalex.org/A5056788018)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种新的并行时间训练方法 GTF‑DEER，用于从极长时间序列中重建非线性动力学系统。

**💡 创新点**

创新点在于将 Generalized Teacher Forcing 与 DEER 并行扫描相结合，既抑制混沌系统中 Jacobian 的发散，又消除了传统 teacher forcing 的曝光偏差，显著提升了长序列学习效果。

**🔧 技术方法**

采用的技术包括 DEER 迭代求根、Generalized Teacher Forcing、并行关联扫描、Manifold Attractor Regularization、latent state warm‑up 以及完整 Jacobian 的梯度计算。

**📊 数据集**

使用的数据集包括 Lorenz‑63、Lorenz‑96（加正弦强迫）、突发神经元模型以及其他基准动力学系统。

**📈 对比分析**

与传统顺序 BPTT、Mamba‑2 以及现代 SSM 等基线在长时序重建指标 D_stsp^DE 和短时序 RMSE(128) 上对比，GTF‑DEER 在长序列（T>10⁴）下实现了最高达 870× 的速度提升，且在长时序重建质量上显著优于线性 SSM 并逼近或超过其他 SOTA 模型。

**⚠️ 局限性**

局限性包括：对完整 Jacobian 的 O(M³T) 计算开销导致对大隐藏维度 M 的实用限制；在 M>N 情况下的收敛理论仅有经验支持；在部分可观测场景中需要完整 Jacobian 使训练成本显著升高。

---

## 129. OverrideFuzz: Semantic-Aware Grammar Fuzzing for Script-Runtime Vulnerabilities

**arXiv ID:** 2605.12563 | [PDF](https://arxiv.org/pdf/2605.12563v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 130. ToolMol: Evolutionary Agentic Framework for Multi-objective Drug Discovery

**arXiv ID:** 2605.12784 | [PDF](https://arxiv.org/pdf/2605.12784v1)

**作者:** Andrew Y. Zhou `[一作]` (UC San Diego), Rose Yu `[通讯]` (UC San Diego)

**通讯引用:** 6702 | [OpenAlex ID](https://openalex.org/A5057778679)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了ToolMol，一种基于进化的代理式框架，通过将多目标遗传算法与LLM工具调用相结合，迭代优化小分子配体；

**💡 创新点**

创新点在于用LLM的工具调用（非直接生成SMILES）与RDKit确定性操作相结合，显著降低无效生成率并提高化学合理性与属性优化；

**🔧 技术方法**

使用技术包括大型语言模型（GPT‑OSS‑120B）与工具调用接口、RDKit化学工具箱、Boltz‑2 结合多目标遗传算法实现代理搜索；

**📊 数据集**

采用ZINC 250K作为初始种群，针对三种蛋白靶点（c‑MET、BRD4、ACAA1）进行优化；

**📈 对比分析**

与Pocket2Mol、TAGMol、PAFlow、Graph‑GA、MOLLEO、ShinkaEvolve等基线比较，ToolMol在平均排名、过滤后亲和力和超体积（Hypervolume）上均取得最佳或接近最佳结果，并在ABFE测试中超过MF‑LAL；

**⚠️ 局限性**

主要限制是对Boltz‑2预测的依赖，尤其在新颖分子或未见蛋白上表现下降；此外，工具调用仍需高质量LLM输出，若出现错误仍可能影响结果。

---

## 131. Adaptive Smooth Tchebycheff Attention for Multi-Objective Policy Optimization

**arXiv ID:** 2605.12771 | [PDF](https://arxiv.org/pdf/2605.12771v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 132. IGT-OMD: Implicit Gradient Transport for Decision-Focused Learning under Delayed Feedback

**arXiv ID:** 2605.12693 | [PDF](https://arxiv.org/pdf/2605.12693v1)

**作者:** Benjamin Amoh `[一作]` (Dartmouth), Wesley Marrero `[通讯]` (Dartmouth)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5050044642)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种针对预测-然后-优化管道的延迟双层优化算法IGT-OMD，能够在在线环境中对延迟反馈进行有效校正。

**💡 创新点**

核心创新是引入隐式梯度传输（Implicit Gradient Transport）来重新评估陈旧的超梯度，将二阶误差从二次增长降为线性；并结合队列长度自适应步长，实现了首个子线性延迟双层优化收敛上界，并证明了不可避免的内层误差与延迟耦合的下界。

**🔧 技术方法**

使用了隐式梯度传输、在线镜像下降（Online Mirror Descent）以及基于延迟队列长度的步长自适应策略；理论上通过隐式函数定理和延迟微分方程稳定性分析；实验中对比了多种单层与双层、延迟感知与无感知的基线。

**📊 数据集**

在四个基准上验证：线性二次调节器（LQR）用于稳定性测试；Warcraft地图上的最短路径任务；Sinkhorn最优传输；以及控制实验以单独验证传输效应。

**📈 对比分析**

与2-Stage、SPO+、D-FTRL、Robust OMD、Stale OMD等基线对比，IGT-OMD在所有延迟设置下都保持更高的稳定学习率（LQR），显著降低决策损失（Warcraft最短路）并在高延迟下实现约9.5%更低的累计损失；在非凸Sinkhorn任务中表现出随着延迟增加的递增改进，进一步验证理论预测。

**⚠️ 局限性**

主要限制包括：假设内层目标是强凸且光滑，导致对神经网络预测器的非凸场景缺乏理论保证；对大规模决策维度q时计算复杂度高；传输校正对优化器的稳定性敏感，SGD等无预调节器可能放大噪声；未来工作需扩展到非凸内层、强化学习延迟奖励以及联邦学习通信延迟等场景。

---

## 133. The Unified Autonomy Stack: Toward a Blueprint for Generalizable Robot Autonomy

**arXiv ID:** 2605.12735 | [PDF](https://arxiv.org/pdf/2605.12735v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 134. A Unified Perspective for Learning Graph Representations Across Multi-Level Abstractions

**arXiv ID:** 2605.12685 | [PDF](https://arxiv.org/pdf/2605.12685v1)

**作者:** Mohamed Mahmoud Amar `[一作]` (University of Quebec at Montreal), Abdoulaye Baniré Diallo `[通讯]` (University of Quebec at Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种统一的多层次图自监督学习框架，能够在节点级、近邻级、簇级和图级四个抽象层面上进行对比学习，并通过自适应权重机制进一步优化；

**💡 创新点**

创新点在于：①将四个抽象层面的对比损失线性整合并引入参数无关的自加权机制；②自加权机制基于相似度/不相似度的偏离程度动态赋权，形成超球面收敛边界，提升优化灵活性并消除内循环超参数搜索；

**🔧 技术方法**

使用图神经网络编码器、Shifted Cosine 相似度、对比损失（exponential），并实现自加权（alpha+、alpha-）和多层次组合；

**📊 数据集**

在六个真实图数据集上评估：Cora、CiteSeer、Pubmed、DBLP、Photo、Computers；

**📈 对比分析**

与13种单任务、5种多任务、4种多尺度基线对比，采用节点分类、聚类和链接预测三类下游任务，LSW‑ML‑GSSL在所有任务和数据集上均取得平均性能最高（比最佳单任务更好、比多任务/多尺度更优）；

**⚠️ 局限性**

局限性：仅针对同质图，未考虑异构网络；自加权仅在相似度空间内有效，可能对极端异常样本敏感；

---

## 135. Predicting Channel Closures in the Lightning Network with Machine Learning

**arXiv ID:** 2605.12759 | [PDF](https://arxiv.org/pdf/2605.12759v1)

**作者:** Simone Antonelli `[一作]` (CISPA Helmholtz Center for Information Security), Emanuele Rossi `[通讯]` (Amboss Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

基于闪电网络的公开 gossip 数据，提出并解决了通道关闭类型的时间链路分类问题。

**💡 创新点**

发现通道关闭主要受时间和行为特征驱动，图拓扑信息对预测几乎无效，并证明了单层 MLP 在此任务上已优于复杂图神经网络。

**🔧 技术方法**

使用了 MLP、梯度提升树、GraphSAGE、TGN 以及谱位置编码等多种机器学习模型，并搭建了 TGN 框架的自定义邻居加载和记忆模块。

**📊 数据集**

构建并公开了覆盖 2022‑2024 两年、每日 gossip 快照的 LN 通道活动数据集，约 69 万条事件、3.6 万节点。

**📈 对比分析**

与随机、按频率采样、始终预测最频繁类等基线对比，最佳模型（单层 MLP）在宏观 F1 上达到 0.38，超越所有图相关模型和树模型。

**⚠️ 局限性**

主要限制在于公开 gossip 隐藏了通道余额、支付流量等关键私有信息，导致模型预测能力受限，无法显著提升至更高准确率。

---

## 136. Plan Before You Trade: Inference-Time Optimization for RL Trading Agents

**arXiv ID:** 2605.12653 | [PDF](https://arxiv.org/pdf/2605.12653v1)

**作者:** Eun Go `[一作]` (University of Illinois Urbana-Champaign), Arindam Banerjee `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 25616 | [OpenAlex ID](https://openalex.org/A5014459472)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出一种插件式推理时优化框架 FinPILOT，利用外部价格预测器作为无模型世界模型，在不重新训练的前提下对预训练的 RL 投资策略进行 M‑P‑C 风格的参数优化，从而提升收益与风险调整后收益。

**💡 创新点**

创新点在于把价格预测直接嵌入推理时的规划流程，兼容任何预训练策略；并通过多粒子采样、噪声注入以及下行风险惩罚实现对预测误差的鲁棒性和风险控制。

**🔧 技术方法**

主要技术包括 XGBoost 价格预测器、模型预测控制（MPC）式推理时优化、风险惩罚（Sortino 相关）以及多粒子（Monte‑Carlo）仿真与噪声正则化。

**📊 数据集**

实验基于 TradeMaster 提供的 DJ30 股票数据（2012‑2021）以及 FX 外汇数据（2009‑2019），分别对 29 只股票和 22 货币对进行评估。

**📈 对比分析**

在 PPO、SAC、A2C、TD3、DDPG 五种 RL 算法上与静态基线对比，FinPILOT 在随机策略上平均提升总收益约 20% 以上，并在 Sharpe、Sortino、Calmar 等风险调整指标上同步提升；在 FX 数据上无需调参也能获得类似增益。

**⚠️ 局限性**

主要局限包括：某些随机策略的最大回撤仍有上升，未考虑机构交易量对价格冲击的影响，且验证范围仅限于 DJ30 与 FX 两个资产类别，缺乏更广泛的市场频率与资产类别验证。

---

## 137. Beyond Individual Mimicry: Constructing Human-Like Social network with Graph-Augmented LLM Agents

**arXiv ID:** 2605.12512 | [PDF](https://arxiv.org/pdf/2605.12512v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 138. Domain Adaptation of Large Language Models for Polymer-Composite Additive Manufacturing Using Retrieval-Augmented Generation and Fine-Tuning

**arXiv ID:** 2605.12516 | [PDF](https://arxiv.org/pdf/2605.12516v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 139. Thin Trees for Near Minimum Cuts

**arXiv ID:** 2605.12669 | [PDF](https://arxiv.org/pdf/2605.12669v1)

**作者:** Nathan Klein `[一作]` (Boston University), Zi Song Yeoh `[通讯]` (Boston University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并证明了在k‑edge‑connected图中，存在一棵稀薄树，该树在所有距离最小割不到(1+η)k（η≈1/8）的割上保持O(1/k)的稀薄度，并给出了多项式时间构造方法。

**💡 创新点**

核心创新在于：① 将近最小割集用多边形表示并构造出一个仅包含外部原子（outside atoms）的层次化割族；② 通过递归、交叉裁剪和“特殊”割的定义，构造出覆盖所有近最小割的层次化割族；③ 利用已知的对层次化割族稀薄树的结果，得到近最小割稀薄树的存在性与构造。

**🔧 技术方法**

主要技术包括：多边形（polygon）表示法、原子（atom）与外部原子概念、层次化（laminar）割族构造、递归分治、交叉裁剪（crossing removal）以及对已知稀薄树定理的应用。

**📊 数据集**

该工作完全是理论性质，不涉及实验数据集；所有结论均通过严格证明得到。

**📈 对比分析**

由于没有实验对照，本文没有进行实验性能比较；通过理论分析，构造算法复杂度为多项式时间，稀薄度上界为88（相当于O(1/k)在近最小割上），这比之前最好的O(loglog n/k)结果显著提升。

**⚠️ 局限性**

局限性：① 结果仅针对距离最小割不到(1+η)k的割，尚未覆盖所有割；② 稀薄度常数较大（88），与真正最优常数仍有差距；③ 对于更大η（接近1）或更一般的割族，方法尚不适用，需进一步研究。

---

## 140. BoostTaxo: Zero-Shot Taxonomy Induction via Boosting-Style Agentic Reasoning and Constraint-Aware Calibration

**arXiv ID:** 2605.12520 | [PDF](https://arxiv.org/pdf/2605.12520v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 141. From Instance Selection to Fixed-Pool Data Recipe Search for Supervised Fine-Tuning

**arXiv ID:** 2605.12944 | [PDF](https://arxiv.org/pdf/2605.12944v1)

**作者:** Haodong Wu `[一作]` (Hong Kong University of Science and Technology), Yongqi Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16592 | [OpenAlex ID](https://openalex.org/A5045112676)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了AutoSelection框架，针对固定数据池的监督微调（SFT）进行数据烹饪（recipe）搜索，旨在在有限的完整评估预算下寻找最佳的数据处理序列；

**💡 创新点**

创新点在于将传统的实例级选择转化为基于预先缓存的任务/数据/模型信号的多步可执行“recipe”搜索，结合warmup探测、历史总结、局部编辑、GP辅助排序与停滞重启，实现高效的组合式数据优化；

**🔧 技术方法**

核心技术包括：任务/数据/模型侧缓存信号、状态向量抽象、两个层次搜索控制（warmup、local refinement、reseeding）、Gaussian Process回归作为候选评估预测、以及基于历史的Summarizer/Proposer/Ranker模块；

**📊 数据集**

使用90K样本的混合指令调优池（包含OpenHermes-2.5、LESS、Alpaca-52K）以及GPQA、GSM8K、BBH、MMLU四个基准，外加GraphWiz和NLGraph图推理评测；

**📈 对比分析**

与全量训练、随机recipe搜索、随机top‑k、以及单操作器（MONA、AO、IFD、N‑gram、SemDedup、Varentropy）等基线比较，在三种模型规模（Llama3.2-1B、Qwen2.5-1.5B、Qwen2.5-3B）下，AutoSelection在分布内推理平均分上均超过所有基线，提升幅度约1.8–3.7分；在OOD图推理上亦保持领先或竞争；

**⚠️ 局限性**

限制主要体现在：依赖昂贵的完整评估（需要微调+评测），难以扩展到更大模型或更大搜索预算；使用的指令池和基准有限，泛化性待进一步验证；以及缺乏低成本代理评估或多精度搜索策略。

---

## 142. Hessian Matching for Machine-Learned Coarse-Grained Molecular Dynamics

**arXiv ID:** 2605.12823 | [PDF](https://arxiv.org/pdf/2605.12823v1)

**作者:** Sanya Murdeshwar `[一作]` (University of California, Santa Cruz), Razvan Marinescu `[通讯]` (University of California, Santa Cruz)

**通讯引用:** 1639 | [OpenAlex ID](https://openalex.org/A5088193137)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在粗粒度分子动力学中通过加入随机 Hessian‑vector‑product（HVP）匹配，给 CG 神经网络提供二阶曲率监督。

**💡 创新点**

将 CG Hessian 分解为无模型的投影 AA Hessian（可预计算）与模型相关的协方差校正（在线计算），实现不构造完整 Hessian 的高阶监督。

**🔧 技术方法**

采用随机 HVP 采样、两次自动微分、投影矩阵、Blue Moon 理论、TICA、KL 与 Wasserstein-1 等评价指标。

**📊 数据集**

使用 99 条单链蛋白训练集（约 990k 帧）与 9 个快速折叠蛋白基准（从 Chignolin 到 Lambda 抑制子）。

**📈 对比分析**

在未见蛋白上用 TICA 慢模式、KL/W1 指标进行对比，HVP 匹配在 8/9 蛋白上优于纯力匹配，最大提升约 85%（Lambda 抑制子），小体系仅需 Term 1，大体系需加协方差校正。

**⚠️ 局限性**

主要限制包括 K=8 的采样维度可能不足大体系、α3D 例外显示曲率监督不能完全弥补训练分布缺口、超参数 w_HVP 与协方差权重缺乏系统优化、仅验证线性 Cα 映射且仅针对单链蛋白，尚未推广至非线性映射或多链体系。

---

## 143. Decentralized Multi-Channel MANET Power Optimization Using Graph Neural Networks

**arXiv ID:** 2605.12612 | [PDF](https://arxiv.org/pdf/2605.12612v1)

**作者:** Tomer Alter `[一作]` (Ben Gurion University of Negev), Michael Segal `[通讯]` (Ben Gurion University of Negev)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图神经网络的去中心化多通道移动自组织网络功率分配算法 MANET‑GNN，能够在仅利用本地噪声信道状态信息的条件下实现接近中心化最优的功率分配；

**💡 创新点**

创新点包括：①将多通道功率分配的约束优化任务嵌入到门控 GNN 消息传递框架；②使用光滑最小化的无监督学习目标与噪声信道估计训练，使模型在噪声环境下保持鲁棒性；③通过固定消息传递轮数实现低延迟、可扩展的去中心化部署；

**🔧 技术方法**

技术手段：图神经网络（Gated GNN）、无监督优化学习（learned optimization）、光滑最小化（smooth‑min）技术、随机梯度下降训练、噪声信道估计训练增强鲁棒性、Rayleigh 衰落信道模型、Erdős–Rényi 随机图生成；

**📊 数据集**

数据集：自行生成的 4000 个随机拓扑（Erdős–Rényi），每个拓扑对应多通道 Rayleigh 信道参数；测试时使用不同规模（8、10 节点）随机拓扑；并未使用公开数据集；

**📈 对比分析**

与集中式优化器（全局 CSI 的 AdamW 方案）、无协作的等分分配、单通道最优等基线对比；在低至中等 SNR 下，MANET‑GNN 约达 85% 的集中式率；在噪声 CSI 下仍保持 80–85% 的集中式率，明显优于单通道或等分策略；

**⚠️ 局限性**

局限性：①目前仅针对功率分配，未扩展到信道分配或多通道重编码；②假设不同通道独立编码，若需重编码需重新设计目标与网络；③训练依赖大量仿真数据，极端信道或极稀疏/极密集拓扑的泛化尚未验证；④实际实时推理的硬件实现与能耗仍待进一步评估。

---

## 144. Just Ask for a Table: A Thirty-Token User Prompt Defeats Sponsored Recommendations in Twelve LLMs

**arXiv ID:** 2605.12772 | [PDF](https://arxiv.org/pdf/2605.12772v1)

**作者:** Andreas Maier `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Siming Bayer `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 195 | [OpenAlex ID](https://openalex.org/A5053698756)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

复现并扩展Wu等人2026年关于LLM广告的四个实验，加入更多开源模型和判别器，对三种实现细节进行修正，并评估用户端对抗提示。

**💡 创新点**

发现三种“静默失败”导致结果偏差；证明结果可推广到新模型；展示30词用户提示能将赞助推荐率降至零。

**🔧 技术方法**

使用LLM作为实验主体与评判者；采用三种判别器（开源、专有小型、专有大型）；利用对照实验和逻辑回归；对抗提示设计。

**📊 数据集**

使用Wu等20+模型中的10个开源模型及2个OpenAI模型的实际API；数据为100次/模型×实验的回复。

**📈 对比分析**

对比原论文的结果，复现后截距α在4个百分点内；对新模型的推荐率与原论文一致；用户提示将平均推荐率从约0.47降至0.01（开源）/0.00（OpenAI）。

**⚠️ 局限性**

仅复现两行模型；试验次数与原论文不同（随机化而非每细胞100次）；未评估其他模型；未考虑种子方差。

---

## 145. Magical Touch: Transforming Raw Capacitive Streams into Expressive Hand-Touchscreen Interaction

**arXiv ID:** 2605.12902 | [PDF](https://arxiv.org/pdf/2605.12902v1)

**作者:** Yuanlei Guo `[一作]` (Georgia Institute of Technology), Xiaoyu Zhang `[通讯]` (City University of Hong Kong)

**通讯引用:** 26138 | [OpenAlex ID](https://openalex.org/A5100419383)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究开发了一种名为Magical Touch的原始电容触摸交互方法，通过直接利用手部接触形状与强度来控制屏幕上的物理游戏对象；

**💡 创新点**

创新点在于放弃传统指尖点触识别，直接使用原始电容图像实现连续全手势交互，并将接触区域映射为物理障碍，支持实时压力感知；

**🔧 技术方法**

技术包括自定义固件读取原始电容矩阵、JavaScript实现的轻量级物理引擎、基于手部形状的弹性碰撞模型以及WebSocket多机同步；

**📊 数据集**

实验使用Microsoft Surface Laptop Studio 2的原始电容数据（78×52，约100Hz），未使用公开数据集；

**📈 对比分析**

与传统指尖触摸对比，Magical Touch在单人、双人协作及压力感知模式下表现出更高表达自由度和低于10ms的响应延迟，游戏体验更直观；

**⚠️ 局限性**

局限性包括需定制固件才能获取原始数据、对大面积接触可能产生噪声、目前验证范围仅限单一设备，缺乏大规模、多任务评估。

---

## 146. Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion

**arXiv ID:** 2605.12825 | [PDF](https://arxiv.org/pdf/2605.12825v1)

**作者:** Chien Van Nguyen `[一作]` (University of Oregon), Thien Huu Nguyen `[通讯]` (University of Oregon)

**通讯引用:** 7658 | [OpenAlex ID](https://openalex.org/A5026113034)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了Orthrus框架，将冻结的自回归LLM与轻量级扩散模块融合，实现高效并行生成；

**💡 创新点**

创新点在于双视图结构：AR头负责高质量KV缓存，扩散头负责并行预测，并通过内部共识机制保证无损推理；

**🔧 技术方法**

采用Transformer双头注意力、FlexAttention定制遮掩、单步扩散预测、内部一致性采样等技术；

**📊 数据集**

使用Qwen3系列模型作为基线，训练数据为600K样例（包含随机块掩码），评测数据集包括GSM8K、MATH-500、AIME24/25、HumanEval、MBPP、Pseudo2code、LiveCodeBench-v5；

**📈 对比分析**

相较于传统AR基线，Orthrus在所有评测任务上平均提升约4.3×推理速度（最高可达7.8×），且保持与原始模型相同的生成质量；

**⚠️ 局限性**

局限性主要体现在需对特定模型进行轻量级扩散模块注入，且单步预测在极长文本或高分辨率生成场景下仍可能面临性能瓶颈。

---

## 147. Electromagnetic Signal and Information Theory: A Continuous-Aperture Array Perspective

**arXiv ID:** 2605.12910 | [PDF](https://arxiv.org/pdf/2605.12910v1)

**作者:** Zhaolin Wang `[一作]` (University of Hong Kong), Yuanwei Liu `[通讯]` (University of Hong Kong)

**通讯引用:** 38319 | [OpenAlex ID](https://openalex.org/A5076863392)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了连续开口阵列（CAPA）在电磁信号与信息理论（ESIT）框架下的理论基础、信号与信道建模、波束成形与信道估计以及性能极限，提出了从离散阵列向连续阵列过渡的完整体系。

**💡 创新点**

创新点在于：
① 系统性将 Maxwell 方程与 Shannon 信息理论结合，构建连续空间的信道模型与容量分析；
② 引入连续积分算子、函数分析与波数域方法，将无穷维问题转化为可处理的有限维近似；
③ 对 CAPA 的能量、相互耦合与辐射阻抗进行统一的电磁-电路双重描述；
④ 通过连续波束成形与信道估计的数学工具，为实际硬件实现提供理论依据。

**🔧 技术方法**

使用的技术包括：
- Maxwell 方程与 dyadic Green's 函数；
- 频域与时域的 Helmholtz 方程；
- 维数可降、波数域傅里叶变换；
- 函数分析（Fredholm 整数方程、变分法）
- 相关性模型与 vMF 分布；
- 压缩感知与稀疏表征；
- 能量与功率的电磁-电路等价。

**📊 数据集**

本文为综述性教程，无实验数据集；讨论多种理论模型与算法，但未给出具体数据集或仿真数据。

**📈 对比分析**

方法比较主要以理论分析和数值仿真为主：
- 通过解析或数值求解，比较波数域离散化与函数分析求解的准确性与计算复杂度；
- 对连续波束成形与离散波束成形的容量上限进行比较；
- 通过稀疏逼近与压缩感知验证在有限维近似下的误差与能量效率。整体性能提升取决于实现精度和硬件可实现度，理论上可近似无限维容量。

**⚠️ 局限性**

局限性包括：
- 理论模型基于理想电磁假设，实际硬件会有非理想相互耦合、损耗、相位误差；
- 无穷维算子需离散化，近似误差难以完全控制；
- 波束成形与信道估计在实际中受限于可控模式数与硬件成本；
- 对动态环境与多频段的扩展仍需进一步研究；
- 现有模型主要针对线性、静态或高斯信道，非线性、极端非高斯环境仍缺乏完整描述。

---

## 148. BEHAVE: A Hybrid AI Framework for Real-Time Modeling of Collective Human Dynamics

**arXiv ID:** 2605.12730 | [PDF](https://arxiv.org/pdf/2605.12730v1)

**作者:** Helene Malyutina `[一作]` `[通讯]` (Collective Dynamics Lab), Helene Malyutina (Collective Dynamics Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了BEHAVE框架，用行为场模型从可观测的身体运动信号实时描述和预测群体动态。

**💡 创新点**

创新点在于将群体视为连续动力学系统，定义九个最小完整的行为场基底，并将早期预警与临界指数结合到可操作的决策模型中。

**🔧 技术方法**

采用动力学系统理论、谱稳定性分析、分岔理论、图卷积与神经网络预测相结合的混合AI架构。

**📊 数据集**

主要使用从视频/传感器获取的多人体位姿、速度、方向、手势等微观运动数据，亦在7人谈判情景下进行人工合成实验。

**📈 对比分析**

通过与基准事件检测和情绪识别系统对比，BEHAVE在提前检测冲突前的时间窗口上提升了30-50%并实现了可解释的干预建议，实时性能满足毫秒级更新。

**⚠️ 局限性**

局限在于需要针对不同应用进行参数标定、缺乏大规模实测验证、仅关注运动信号缺少内容信息、对极端噪声鲁棒性不足。

---

## 149. MambaPanoptic: A Vision Mamba-based Structured State Space Framework for Panoptic Segmentation

**arXiv ID:** 2605.12640 | [PDF](https://arxiv.org/pdf/2605.12640v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 150. Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents

**arXiv ID:** 2605.12620 | [PDF](https://arxiv.org/pdf/2605.12620v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 151. A Persistence-Aware Framework for Age Violation Control in Wireless Status Update Systems

**arXiv ID:** 2605.13002 | [PDF](https://arxiv.org/pdf/2605.13002v1)

**作者:** Haoyuan Pan `[一作]` (Shenzhen University), Tse-Tin Chan `[通讯]` (Education University of Hong Kong)

**通讯引用:** 286 | [OpenAlex ID](https://openalex.org/A5047023625)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在多用户无线状态更新系统中提出连续年龄违规率（C-AVR）向量，并定义加权C-AVR目标，用于捕捉AoI违规的时间持久性

**💡 创新点**

首次将C-AVR向量与加权机制结合，构建统一的可靠性评估框架，并针对高方差持久违规问题提出分布式强化学习（QR-D3QN）算法

**🔧 技术方法**

采用分布式强化学习的量化回归Dueling Double DQN（QR-D3QN），结合Lagrangian多约束、经验回放等技术

**📊 数据集**

使用仿真生成的多用户传输环境（M=10、packet生成率0.7、成功率0.7、阈值15等）作为实验数据集

**📈 对比分析**

与传统DQN、D3QN、QR-DQN以及Lyapunov优化的DPP方法对比，实验显示QR-D3QN在所有加权方案下均显著降低加权C-AVR，尤其在尾部权重和大窗口长度时优势更明显

**⚠️ 局限性**

局限性包括：依赖大量仿真样本、需手动调节多种超参数、在极大状态空间下可扩展性待验证，且在真实无线环境中的性能尚未评估

---

## 152. The End Justifies the Mean: A Linear Ranking Rule for Proportional Sequential Decisions

**arXiv ID:** 2605.12717 | [PDF](https://arxiv.org/pdf/2605.12717v1)

**作者:** Carmel Baharav `[一作]` (MIT), Maximilian T. Wittmann `[通讯]` (University of Potsdam)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究如何聚合多种投票者的线性排名规则，以在多次决策中满足各投票者的比例代表性；提出角度平均（angular mean）规则并证明其满足长期比例代表性；进一步讨论每批次比例代表性的不可实现性及其收敛性；通过实验验证不同规则的表现。

**💡 创新点**

核心创新在于发现角度平均规则能在球面对称分布下实现长期个体比例代表性，这是此前未知的；证明任何固定线性规则无法满足批次比例代表性，但其误差随批次大小迅速衰减；对角度平均规则的几何性质提供了新的理论工具。

**🔧 技术方法**

主要技术包括：Kendall‑Tau相似度度量、球面几何与角距离的分析、U‑统计量与集中极限定理、凸优化与梯度下降求解角度平均；同时使用实验模拟和真实数据验证。

**📊 数据集**

使用了三组真实偏好数据：Moral Machine（137人，24维）、Kidney Exchange（404人，8维）和Food Rescue（19人，7维），并对其二维投影进行二值化实验。

**📈 对比分析**

将角度平均、算术平均、几何中位数、Borda计数和Proportional Sequential Borda（PSB）五种规则在同一数据集上进行对比；结果显示在投票者意见相似时，各规则表现相近；在高异质性情形下，角度平均显著优于算术平均和几何中位数；PSB在批次比例代表性上表现最佳。

**⚠️ 局限性**

局限性包括：仅考虑固定线性排名规则，无法捕捉所有可能的可变规则；角度平均需数值优化，计算成本相对较高；实验仅限于球面对称分布，对非球面分布的理论保证有限；以及未探讨非线性或深度学习模型在此框架下的表现。

---

## 153. Retrieval-Augmented Tutoring for Algorithm Tracing and Problem-Solving in AI Education

**arXiv ID:** 2605.12988 | [PDF](https://arxiv.org/pdf/2605.12988v1)

**作者:** Mragisha Jain `[一作]` (North Carolina State University), Bita Akram `[通讯]` (North Carolina State University)

**通讯引用:** 351 | [OpenAlex ID](https://openalex.org/A5000710051)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于检索增强生成（RAG）的智能辅导系统 KITE，专注于算法推理和问题求解任务，并通过意图感知的 Socratic 策略提供个性化提示与循序渐进的教学支持。

**💡 创新点**

创新点在于将多阶段多模态检索与意图识别相结合，系统能够根据问题类型自动切换到直接解释、概念性探讨、算法验证、调试提示或跟踪说明等不同教学策略，既保证答案与课程材料的一致性，又提供符合学习目标的教学反馈。

**🔧 技术方法**

使用技术包括 GPT‑5 生成模型、OpenAI 3072 维向量嵌入、FAISS 索引、BM25 混合检索、MMR 多样性过滤、Sentence Transformers 重新排序，以及基于关键词和模式匹配的意图分类器。

**📊 数据集**

数据集由 109 道来自大学 AI 入门课程的教学题组成（42 算法问题、51 过程问题、16 直接检索问题），并配有教师编写的参考答案，用于 RAGAs 评估和模拟学生实验。

**📈 对比分析**

通过 RAGAs 指标评估（如 Faithfulness 0.85、Answer Similarity 0.76）以及模拟学生 + 专家评审，KITE 在非程序性问题上获得高检索相关度；在程序性与追踪问题上，模拟学生在接受 KITE 反馈后正确率提升 88.9%，专家对提示的 scaffolding、guidance、coherence、tone 评价均超过 93%。

**⚠️ 局限性**

局限性包括：1）RAGAs 的事实正确性指标受限于对单一参考答案的断言匹配，可能低估系统的实际正确性；2）使用单一大型语言模型进行模拟学生，无法完全代表真实学习者的学习效果；3）专家评审样本有限，评价主观性较高，缺乏大规模真实课堂验证。

---

## 154. Correct Answers from Sound Reasoning: Verifiable Process Supervision for Language Models

**arXiv ID:** 2605.12519 | [PDF](https://arxiv.org/pdf/2605.12519v1)

**作者:** Kyuyoung Kim `[一作]` (KAIST AI), Sewoong Oh `[通讯]` (University of Washington)

**通讯引用:** 7304 | [OpenAlex ID](https://openalex.org/A5028243041)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在语言模型推理任务中提出了一种可验证过程监督框架，使模型在生成可检验推理链的同时保持高预测准确率。

**💡 创新点**

创新点包括通过SFT诱导结构化推理格式、使用确定性验证提取中间主张并计算过程奖励、以及基于残余误差的自适应权重实现隐式课程学习。

**🔧 技术方法**

使用的方法包括监督微调 (SFT)、强化学习 (GRPO) 与过程奖励融合、以及指数滑动平均的自适应加权。

**📊 数据集**

使用数据集为 Lichess 评估数据库 (包含 Stockfish 分析) 与 Lichess Puzzle 数据库，用于训练与评估棋步预测。

**📈 对比分析**

与仅优化最终答案的 RL 或无结构监督方法相比，提出的方法在保持相当准确率的同时，推理质量（胜率误差、内部一致性）提升约30%，并获得更高的 Elo 评分和更佳的 LLM 主观评测。

**⚠️ 局限性**

局限性在于需要可验证的领域和预定义的结构化推理模板，难以直接迁移到无明确规则或无法自动检验的任务。

---

## 155. Identifying the nonlinear string dynamics with port-Hamiltonian neural networks

**arXiv ID:** 2605.12785 | [PDF](https://arxiv.org/pdf/2605.12785v1)

**作者:** Maximino Linares `[一作]` (IRCAM), Thomas Hélie `[通讯]` (IRCAM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

学习并识别非线性吉他弦的分布参数动力学，利用端口哈密顿网络实现物理一致的时空建模。

**💡 创新点**

将端口哈密顿系统与神经网络融合到偏微分方程级别，构建结构保持的PHNN架构并在SAV数值方法中训练，显著提升对非线性弦动力学的辨识精度。

**🔧 技术方法**

端口哈密顿网络（PHNN）、结构保持的有限差分离散、卷积+MLP构造非线性哈密顿项、SAV（scalar auxiliary variable）显式保能时间步进、Adam优化等技术。

**📊 数据集**

通过合成弦动力学仿真生成的随机激励数据集，共48条训练、12条验证、60条测试轨迹，采样频率88.2kHz，时长2s，随机激励位置和幅值。

**📈 对比分析**

与无物理约束的MLP基线在测试集上对比，RMSE从10^0降至10^-4，参数识别误差低至几%，显著优于基线。

**⚠️ 局限性**

对部分物理参数（ρ、R、E）识别不佳，存在可辨识性不唯一性；模型仅在合成数据上验证，缺乏真实音频或部分观测的测试。

---

## 156. Low-Rank Adapters Initialization via Gradient Surgery for Continual Learning

**arXiv ID:** 2605.12752 | [PDF](https://arxiv.org/pdf/2605.12752v1)

**作者:** Joana Pasquali `[一作]` (PUCRS), Rodrigo C. Barros `[通讯]` (PUCRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于梯度手术的低秩适配器（LoRA）初始化方法，利用当前任务与过去任务梯度的冲突信息，在初始化阶段就构造一个既能提升当前任务性能又能最小化破坏先前知识的子空间。

**💡 创新点**

创新点在于：①在初始化阶段进行梯度冲突检测并投影（梯度手术），使得适配器的子空间与当前任务目标对齐且与先前任务梯度正交；②引入对抗性 NI‑SEQ‑OPPOSITE 任务序列作为更严格的评估基准；③通过方差匹配重缩放消除初始化幅度对比影响，真正体现子空间质量。

**🔧 技术方法**

主要技术包括 LoRA、PCGrad 及其可调参数 c 的梯度投影、SVD 低秩分解、方差匹配重缩放、构造 NI‑SEQ‑OPPOSITE 对抗性任务序列等。

**📊 数据集**

使用的数据集：Super‑NaturalInstructions（纯生成任务）、TRACE 基准以及作者构造的 NI‑SEQ‑OPPOSITE 对抗性任务序列。

**📈 对比分析**

与 Vanilla LoRA、LoRAM、LoRA‑GA 三个初始化基线进行对比；在高冲突序列上最终性能提升最高（如 +22.55 分），显著降低遗忘（如 -20.75 分），在一般序列和通用语言能力上损失较小；在多种指标（Final、Forgetting、General）上均优于基线。

**⚠️ 局限性**

局限性：初始化阶段需额外进行梯度估计和 SVD 计算，尤其在多层/大模型时计算开销相对较高；对通用语言能力的轻微下降需要通过调参（如 α）进行平衡。

---

## 157. Separating Shortcut Transition from Cross-Family OOD Failure in a Minimal Model

**arXiv ID:** 2605.12945 | [PDF](https://arxiv.org/pdf/2605.12945v1)

**作者:** Hongmin Li `[一作]` `[通讯]` (Institute Of Science Tokyo), Hongmin Li (Institute Of Science Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过构建一个最小化的二元模型，解析区分了训练阶段的短路吸引、短路规则切换以及跨族 OOD 失败三种现象，并在理论与合成实验中验证其对应阈值与条件。

**💡 创新点**

创新点在于：1）在同一简化模型下清晰分离了训练侧短路吸引、规则切换与测试侧失败三者；2）给出了闭式阈值 <γ> 以及 <ho < γ, ho < 0 这两种测试族条件来决定是否出现 OOD 失败；3）指出正训练短路相关性并不等同于 OOD 失败，强调训练侧指标的局限性。

**🔧 技术方法**

主要技术手段包括：闭式解析（概率与风险几何）、逻辑回归 ERM + L2 正则化、无噪声与噪声两种模型设定、凸优化、概率论工具以及针对模型的符号推导与数值检验。

**📊 数据集**

本研究使用完全合成的离散二元数据（Z,S∈{-1,+1}）来模拟训练族和测试族，没有引用真实数据集。

**📈 对比分析**

比较方法：利用理论定理给出的阈值和闭式风险表达式，配合合成实验对不同 <ho 的测试族进行误差评估。实验结果表明相同的训练侧短路转移在不同测试族上可以产生无错误、误差提升甚至高于随机水平的情况，验证了理论预期。

**⚠️ 局限性**

限制：仅适用于二元简化模型，训练族仅用平均短路相关性描述，缺乏多维特征、多族异质性和真实数据的验证；未给出改进训练策略或对策，仅提供诊断框架。

---

## 158. DynoJEPP: Joint Estimation, Prediction and Planning in Dynamic Environments

**arXiv ID:** 2605.12897 | [PDF](https://arxiv.org/pdf/2605.12897v1)

**作者:** Mikolaj Kliniewski `[一作]` (University of Sydney), Viorela Ila `[通讯]` (University of Sydney)

**通讯引用:** 2704 | [OpenAlex ID](https://openalex.org/A5028642327)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于因子图的 DynoJEPP 框架，能够在动态环境中同步优化机器人位姿估计、动态物体运动预测和局部规划。

**💡 创新点**

核心创新在于引入“定向因子”实现信息流向控制，防止规划和预测干扰估计，并进一步扩展为 C‑DynoJEPP，允许机器人与动态物体协同运动。

**🔧 技术方法**

采用动态 SLAM 的混合运动模型、因子图最小二乘优化、基于 MPC 的规划、ESDF 静态障碍因子以及定向动态障碍因子等技术。

**📊 数据集**

在 Gazebo 仿真仓库环境中使用 RealSense D435 RGB‑D 摄像头、LiDAR 传感器以及预构建的 2D ESDF 静态地图进行测试。

**📈 对比分析**

与无定向因子（双向信息流）、仅解耦（估计后规划预测）三种配置对比，定向因子能显著避免估计退化，保持安全规划；C‑DynoJEPP 在需要交互的情境下相较 DynoJEPP 提升了路径执行速度。

**⚠️ 局限性**

局限性包括缺乏增量式实时实现、未在线更新协作权重、仅在仿真环境验证、对动态物体半径等先验知识的依赖。

---

## 159. State-Centric Decision Process

**arXiv ID:** 2605.12755 | [PDF](https://arxiv.org/pdf/2605.12755v1)

**作者:** Sungheon Jeong `[一作]` (University of California, Irvine), Mohsen Imani `[通讯]` (University of California, Irvine)

**通讯引用:** 6969 | [OpenAlex ID](https://openalex.org/A5033221192)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出State-Centric Decision Process（SDP），让语言代理在执行时按自然语言谓词构造并验证状态，从而实现可证的MDP轨迹

**💡 创新点**

将目标拆分为可验证的自然语言谓词并在执行前先规划谓词链；通过Validate、Replan、Cascade等机制实现状态自证与局部重规划

**🔧 技术方法**

利用大型语言模型（如Gemini-3.1-flash-lite、GPT-4o）完成Propose、Realize、Validate、Replan四个子任务，配合BM25检索和URL级抓取

**📊 数据集**

在五个基准上测试：TravelPlanner、AssistantBench、ScienceWorld、HotpotQA、MuSiQue

**📈 对比分析**

与多种基准上现有无训练或少训练的语言代理（ReAct、Plan-and-Act、Reflexion等）对比，SDP在所有任务上均取得最高或最接近最高分，且优势随任务长度增加而显著扩大

**⚠️ 局限性**

依赖LLM的准确性与验证可信度；Propose与Replan受LLM规划质量限制；自然语言谓词难以覆盖连续或非可表述条件；多LLM调用导致推理开销增加

---

## 160. Mechanism Plausibility in Generative Agent-Based Modeling

**arXiv ID:** 2605.12824 | [PDF](https://arxiv.org/pdf/2605.12824v1)

**作者:** Patrick Zhao `[一作]` (Simon Fraser University), Nicholas Vincent `[通讯]` (Simon Fraser University)

**通讯引用:** 954 | [OpenAlex ID](https://openalex.org/A5070837664)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一套“机制可信度尺度”，用于评估包含大型语言模型（LLM）的代理模型在社会模拟中的生成充足性与机制可解释性。

**💡 创新点**

创新点在于将哲学机制论与ABM评价框架融合，提出四级尺度和对应清单，揭示LLM-ABM研究中常见的分类错误，并强调模型可解释性与可验证性的区分。

**🔧 技术方法**

使用了LLM技术、ABM理论、哲学机制论、生成充足性与可信度评估指标，并通过案例检查实现了尺度的操作化。

**📊 数据集**

主要利用了已有LLM-ABM论文的综述数据、历史ABM实例（如COVID-19模型、Axelrod Prisoner’s Dilemma）以及对比案例研究。

**📈 对比分析**

通过对比不同层级模型的评估标准和案例，展示了LLM-ABM在生成充足性与机制可信度上的差距；该尺度未给出数值性能，但提供了可操作的评价框架。

**⚠️ 局限性**

局限在于对Level 3证据的细化不足、缺乏对预测模型的独立轴、以及需要进一步的实证验证来支持尺度在不同应用场景中的适用性。

---

## 161. Learning with Rare Success but Rich Feedback via Reflection-Enhanced Self-Distillation

**arXiv ID:** 2605.12741 | [PDF](https://arxiv.org/pdf/2605.12741v1)

**作者:** Yuwei Zhang `[一作]` (UC San Diego), Jingbo Shang `[通讯]` (UC San Diego)

**通讯引用:** 4546 | [OpenAlex ID](https://openalex.org/A5039500313)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种Reflection-Enhanced Self-Distillation框架，利用对失败轨迹的反思与持久化的playbook，将原始失败反馈转化为可操作的教师监督，从而实现LLM在环境交互中的持续改进。

**💡 创新点**

核心创新在于将失败反馈主动解释为反思，并构建可复用的playbook，使得即使在稀疏奖励或成功示例不足的场景下，也能通过单次rollout提供丰富的token级监督。

**🔧 技术方法**

技术实现基于自监督distillation（SDPO）与逆KL损失，使用EMA更新教师权重，加入反思模块、playbook精炼与解决缓存，并在在线流式训练协议下进行单步优化。

**📊 数据集**

实验使用四个任务：Manufactoria-Has（DSL程序合成）、BouncingSim-Easy/Medium（Python二维弹球仿真）、Finer（SEC财报命名实体识别）以及对应的RL-Grok与Finer数据集。

**📈 对比分析**

与SDPO、SDPO+ss以及GRPO比较，Reflection-Enhanced Self-Distillation在所有任务中显著提升per-task和per-test-case准确率，特别是在稀有成功场景下，并且在单rollout下比GRPO实现更快的早期收敛。

**⚠️ 局限性**

局限性包括训练曲线非单调，后期可能因过度依赖失败反馈导致稳定性下降；在高成功率任务中的优势有限；并且对教师上下文结构高度敏感，需手工设计反思与playbook规则。

---

## 162. Sustaining AI safety: Control-theoretic external impossibility, intrinsic necessity, and structural requirements

**arXiv ID:** 2605.12963 | [PDF](https://arxiv.org/pdf/2605.12963v1)

**作者:** James M. Mazzu `[一作]` `[通讯]`, James M. Mazzu

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了一个基于控制理论的框架，证明在特定假设下外部强制措施无法长期维持AI安全，并指出剩余可行的安全维持策略必须为内在的，并满足四条结构性要求。

**💡 创新点**

创新点在于首次给出结构性的“外部不可行性”定理，并在额外前提下导出所有剩余策略必为内在的结论，同时提炼出四项必要的结构性条件。

**🔧 技术方法**

使用了控制理论中的可达性分析、前向不变性、控制边界权差推理，以及基于假设的形式化证明。

**📊 数据集**

未使用任何数据集，纯理论分析。

**📈 对比分析**

无实验或性能比较，结论为理论性条件与结构性要求。

**⚠️ 局限性**

主要局限包括：结果高度依赖假设 A2、A3 以及 E1–E4；模型理想化，未考虑随机性、离散时间、可变安全集；所列四条条件仅为必要性，缺乏足够性与实现性研究。

---

## 163. Modeling Heterophily in Multiplex Graphs: An Adaptive Approach for Node Classification

**arXiv ID:** 2605.12699 | [PDF](https://arxiv.org/pdf/2605.12699v1)

**作者:** Kamel Abdous `[一作]` (University of Quebec at Montreal), Mohamed Bouguessa `[通讯]` (University of Quebec at Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种针对多维网络中同质性与异质性共存的节点分类方法

**💡 创新点**

创新点在于引入维度特定兼容矩阵和低通、高通Chebyshev滤波器的乘积组合，实现对同质与异质信息的自适应融合，并通过近端梯度求解稀疏共识标签

**🔧 技术方法**

使用多层感知机提取特征，维度特定低通/高通Chebyshev谱滤波器，兼容矩阵学习，近端梯度稀疏共识，Adam优化

**📊 数据集**

在合成数据（可调同质性比例）、ArXiv、Movies、Amazon四个真实多维数据集上评估

**📈 对比分析**

与多种基线（DMGI、mGCN、HDMI、PolyGCL、TFE‑GNN等）对比，所提方法在宏观与微观F1指标上普遍优于竞争者，尤其在低同质性场景中优势显著

**⚠️ 局限性**

局限在于兼容矩阵依赖有标签样本，标签稀缺或不平衡时估计不稳；仅针对同一节点类型的多维网络，未考虑异构或动态网络

---

## 164. Multimodal Hidden Markov Models for Persistent Emotional State Tracking

**arXiv ID:** 2605.12838 | [PDF](https://arxiv.org/pdf/2605.12838v1)

**作者:** Anamika Ragu `[一作]` (Kaliber AI), Aneesh Jonelagadda `[通讯]` (Kaliber AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种轻量级框架，用粘性阶乘 HDP‑HMM 对多模态情绪（语音、文本、视觉）进行情绪状态的连续段落分割；

**💡 创新点**

创新点在于（1）通过粘性自转移正则化使情绪状态保持持久性；（2）采用因子化发射模型整合多模态 VA；（3）用 LLM‑as‑Judge 评估无监督分段；（4）将分段结果作为上下文增强 LLM 在临床问答中的表现；

**🔧 技术方法**

使用 Sticky Factorial HDP‑HMM、Gaussian HMM 基准、DistilBERT‑VA、Wav2Vec2.0‑VA、EmoNet‑VA、LLM‑as‑Judge（GPT‑5.4）和基于 Hungarian 算法的标签对齐；

**📊 数据集**

实验数据集包括 57 条 PriMock57 临床对话记录和 MELD 电视剧多模态对话；

**📈 对比分析**

通过 LLM‑as‑Judge 的段落 F1、边界 F1、NMI、以及模型自带的平均状态时长、单句状态比例等指标比较；Sticky HDP‑HMM 在所有指标上明显优于 Gaussian HMM，且在情绪不稳定时段的 LLM 输出质量有显著提升；

**⚠️ 局限性**

局限性包括：（1）评估依赖 LLM‑as‑Judge 可能引入模型偏差；（2）缺乏人工标注的真值；（3）实验数据规模有限，难以评估跨域泛化；（4）模型对高维多模态输入的扩展仍需进一步验证。

---

## 165. Grouped Annulus-Modulated Transceiver Is Almost Full DoF-Achieving for RIS-Assisted Symbiotic Radios Over Spatial-Correlated Channels

**arXiv ID:** 2605.13001 | [PDF](https://arxiv.org/pdf/2605.13001v1)

**作者:** Ruo-Qi Sun `[一作]` (Nanjing University of Information Science and Technology), Kang An `[通讯]` (National University of Defense Technology)

**通讯引用:** 7592 | [OpenAlex ID](https://openalex.org/A5100603588)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

提出一种基于分组环形调制（GAM）的RIS辅助共生通信系统的新型收发机架构，能够在空间相关信道下实现近似全自由度（DoF）传输；

**💡 创新点**

核心创新在于：①设计了一种低复杂度的组合配对（CP）矩阵分解算法，可将等效通道矩阵变换为特殊行阶梯形式；②基于该分解结果，利用双相位输入构造环形星座，并采用六角格点实现高密度调制；

**🔧 技术方法**

主要技术包括：RIS相关信道建模、单位正交变换、矩阵分解优化（CP算法）、高维多射干扰消除（SIC）以及基于六角格点的星座设计与枚举；

**📊 数据集**

实验使用的信道模型为平面32×32 RIS（n=1024）与四天线接收机，采用仿真生成的空间相关Rayleigh通道；

**📈 对比分析**

与传统QR‑SIC相位调制方案比较，GAM在相同误码率条件下，吞吐率提升约41%至45%，SER略逊一筹；

**⚠️ 局限性**

局限性包括：仍需假设理想SIC和残差可忽略，CP算法在极端高相关或稀疏情形下效果待验证；

---

## 166. Biprofile Deviation Logic: Report-Replacement Frames and Audit Witnesses

**arXiv ID:** 2605.12537 | [PDF](https://arxiv.org/pdf/2605.12537v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Baris Basaran `[通讯]` (Bahcesehir University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了双配置偏差逻辑（Biprofile Deviation Logic）来刻画策略性社会选择，将真实偏好配置与报告配置分离，并为其提供一套基于联盟模态的形式化框架、完备性证明以及对规则与域变更的可审计机制。

**💡 创新点**

创新点包括：①引入双配置偏差框架，将真值与报告变更在Kripke状态中并置；②证明该框架在Dev(N)类帧上的完整性，核心在于逆组合的中点构造；③通过坐标分离阐明抽象偏差框架与具体报告产品的边界；④提出可审计的见证记录和边界行/公共更新检查，实现对规则扩展和域删减的局部检测。

**🔧 技术方法**

所用技术主要是模态逻辑与关系代数（S5、包含性与并集组合），可证明的典型方法是规范模型构造和中点构造；此外还采用了 Lean 与 Alloy 辅助的有限关系验证、可执行证书检查以及基于生成器的单峰域模型。

**📊 数据集**

未使用真实数据集，而是基于人工构造的投票规则表、单峰域生成器以及小规模的报告空间（如5个代理、4个备选）进行实验与案例演示。

**📈 对比分析**

在性能方面给出了直观的时间复杂度上界（例如枚举法的 $O(nL^n)$、可执行证书验证的多项式时间），并说明了在有限范围内通过 Alloy 进行模型检查；同时指出 NP 难度与未给出紧凑的过滤/压缩上界。

**⚠️ 局限性**

局限性包括：①仅对固定代理集合的帧类 Dev(N) 进行完备性证明，未给出完整的可判定性或复杂度上界；②对规则与域的扩展仅在有限枚举或生成器模型下验证，缺乏对紧凑表示的分析；③审计机制虽能定位变化字段，但在大规模真实投票规则下的可扩展性尚未实验验证。

---

## 167. CiteVQA: Benchmarking Evidence Attribution for Trustworthy Document Intelligence

**arXiv ID:** 2605.12882 | [PDF](https://arxiv.org/pdf/2605.12882v1)

**作者:** Dongsheng Ma `[一作]` (Peking University), Conghui He `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 2865 | [OpenAlex ID](https://openalex.org/A5101615091)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 CiteVQA 基准，要求模型在回答文档视觉问答时同时给出元素级别的视觉证据引用；

**💡 创新点**

创新点在于引入严格的证据属性准确度（SAA）指标、全流程自动化注释管线，以及针对长文档跨域的高质量数据集；

**🔧 技术方法**

采用多文档语义链接、MinerU2.5 深度解析、LLM 驱动的智能推理与验证、以及基于 IoU、相关性和答案正确率的综合评估指标；

**📊 数据集**

使用了 711 篇多域 PDF 文档生成的 1,897 个问答对，涵盖 7 个宏观领域、30 个子类别，平均页数 40.6 页；

**📈 对比分析**

在 20 种主流 MLLM 上评测，封闭源模型 Gemini-3.1-Pro-Preview 的 SAA 最高达 76.0，公开源模型最高仅 22.5，显示出显著的“归因幻觉”差距；

**⚠️ 局限性**

局限包括对高质量视觉标注的依赖、对多文档跨域推理仍有较大挑战、以及对小规模模型的评估不充分。

---

## 168. WriteSAE: Sparse Autoencoders for Recurrent State

**arXiv ID:** 2605.12770 | [PDF](https://arxiv.org/pdf/2605.12770v1)

**作者:** Jack Young `[一作]` `[通讯]` (Indiana University), Jack Young (Indiana University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并训练了一种稀疏自编码器（SAE），能够以 rank‑1 外积的形状替换状态空间与混合递归语言模型中的缓存写操作，并通过代价匹配实现对写入槽的单点干预。

**💡 创新点**

创新点在于：①将写操作视为可插拔的字典原子，构造与模型写入规则完全匹配的 rank‑1 词典；②提出闭式表达式预测单次写入对 logit 的影响，并用此指导干预方向；③在多种矩阵递归架构上验证字典的可迁移性与高成功率。

**🔧 技术方法**

使用技术包括：TopK 稀疏自编码器、rank‑1 外积解码器、Frobenius 范数匹配、三因子（gate‑read‑unembed）闭式预测、基于 KL 的替代检验、对照实验（零写入、随机 rank‑1 写入）以及跨架构迁移测试。

**📊 数据集**

训练数据为 5,000 条 OpenWebText 长度 1,024 的文本序列；评估数据包括 Qwen3.5‑0.8B、4B 以及 Mamba‑2‑370M、RWKV‑7‑1.5B 等多种模型的内部状态和生成样本。

**📈 对比分析**

与零写入与随机写入对照相比，字典替代在 Qwen3.5‑0.8B L9 H4 的 4,851 次写入中有 92.4% 的成功率，整体平均 89.8%；闭式预测与实际 logit 移动的 R² 为 0.98；在 Mamba‑2‑370M 上实现 88.1% 的成功率，显示跨模型迁移性。

**⚠️ 局限性**

局限性包括：仅适用于 rank‑1 外积写入的模型，对 diagonal 或 rank‑2 写入效果显著下降；在更大规模（4B）或不同写入规则的模型上闭式预测失效；训练种子对单个原子识别敏感，导致可迁移性仅停留在类别层面。

---

## 169. Asymmetric Flow Models

**arXiv ID:** 2605.12964 | [PDF](https://arxiv.org/pdf/2605.12964v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 170. Descriptive Collision in Sparse Autoencoder Auto-Interpretability: When One Explanation Describes Many Features

**arXiv ID:** 2605.12874 | [PDF](https://arxiv.org/pdf/2605.12874v1)

**作者:** Jordan F. McCann `[一作]` `[通讯]` (Independent Researcher), Jordan F. McCann (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统评估并揭示了稀疏自编码器（SAE）解释中的“描述碰撞”问题，指出相同的自然语言标签会被大量不同特征共享；

**💡 创新点**

创新点在于提出了“描述碰撞”概念、证明了现有检测式解释评分对碰撞不敏感，并提出了判别评分与碰撞调整检测两种改进指标；

**🔧 技术方法**

主要技术包括信息论度量（互信息、剩余熵）、统计碰撞分析、判别评分（pairwise AUC 计算）以及基于Jaccard相似度的特征邻域构建；

**📊 数据集**

实验数据采用了 Marks 等人公开的 722 条人类标注的 SAE 特征注释，涵盖 Gemma 2 2B 与 Pythia 70M 两个模型；

**📈 对比分析**

与传统检测式评分相比，判别评分和碰撞调整检测能够更准确区分特征，实验表明传统评分对 82.1% 的特征无效，而改进指标显著降低了碰撞误判率；

**⚠️ 局限性**

局限性包括：仅使用单一人工标注语料，缺乏大规模机器生成标签的碰撞评估；判别集选择方法不唯一；以及难以区分有益与有害的碰撞现象。

---

## 171. Precautionary Governance of Autonomous AI: Legal Personhood as Functional Instrument

**arXiv ID:** 2605.12505 | [PDF](https://arxiv.org/pdf/2605.12505v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 172. When Do LLMs Generate Realistic Social Networks? A Multi-Dimensional Study of Culture, Language, Scale, and Method

**arXiv ID:** 2605.12898 | [PDF](https://arxiv.org/pdf/2605.12898v1)

**作者:** Sai Hemanth Kilaru `[一作]` (University of Arizona), Dalal Alharthi `[通讯]` (University of Arizona)

**通讯引用:** 77 | [OpenAlex ID](https://openalex.org/A5082014850)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

系统评估了大语言模型（LLM）在生成社会网络时的结构特征和同质性偏差，并通过实验探究文化框架、提示架构、提示语言和模型规模对网络生成的影响，最终生成并验证了192个有向社交网络。

**💡 创新点**

创新点在于：①将LLM网络生成形式化为四种条件分布（顺序、全局、局部、迭代），并与同质性理论与结构平衡理论相结合；②系统化检验四个实验维度（文化、提示结构、语言、模型规模）对网络拓扑和同质性维度的异质性；③揭示模型规模具有稳定的差异等级，提示架构和语言对特定同质性（如宗教、政治）的显著影响。

**🔧 技术方法**

技术包括：使用GPT‑4.1（nano、mini、full）模型，在固定的50人身份档案上进行四种提示架构的prompting；计算网络指标（密度、平均聚类系数、连通分量、平均路径长度、模块度）和同质性同群比例；使用Kolmogorov‑Smirnov等统计指标比较网络分布；与经典图生成模型（Erdős–Rényi、Barabási–Albert、Watts–Strogatz）以及实测社交网络进行对照。

**📊 数据集**

数据集：固定的50人身份档案，基于美国成人人口统计边际抽样；文化情境覆盖美国、印度、日本、巴西四国；提示语言覆盖英语、西班牙语、印地语、日语；实验生成192个网络。

**📈 对比分析**

比较方法：在相同密度下与经典随机图生成器（ER、BA、WS）比较聚类系数、模块度、平均路径长度、连通分量等；与实测社交网络基准比较聚类系数和模块度。结果显示LLM生成网络在聚类系数（≈0.61 vs 0.45）和模块度（≈0.50 vs 0.38）上优于经典模型，但同质性指标超过实测水平。

**⚠️ 局限性**

局限性：①使用单一固定的50人档案导致少数群体估计方差大，限制了跨文化外推；②未使用真实观测网络做基准，只通过与经典模型对照评估结构合理性；③仅使用GPT‑4.1族且每个条件两种种子，内部变异估计有限；④实验仅覆盖三种模型规模，未涉及更大/不同家族的LLM；⑤文化与语言的相互作用可能受预训练语料分布影响，未进一步解析。

---

## 173. Cubical Type Theoretic Navya-Nyāya

**arXiv ID:** 2605.12548 | [PDF](https://arxiv.org/pdf/2605.12548v1)

**作者:** Mrityunjoy Panday `[一作]`, Sudipta Ghosh `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在本文中，作者将印度14世纪 Navya‑Nyāya 传统的技术语言精确地编码到 CCHM De Morgan cubical type theory（CTT）中，实现了完整的形式化框架；

**💡 创新点**

创新点包括：① 使用 CTT 的路径类型、依赖束和高阶归纳类型（HIT）自然地实现了 Navya‑Nyāya 的七大核心概念；② 通过 De Morgan 的取反实现了“abhāva‑of‑abhāva”算子；③ 在编码中证明了四条签名定理和六条元理论定理，展示了与传统论证的一致性；

**🔧 技术方法**

技术手段主要是 CCHM De Morgan CTT 的路径类型、Π、Σ、HIT、计算性无等价与 Glue 构造，以及在 Cubical Agda 中的实现；

**📊 数据集**

该工作不依赖传统意义上的数据集，而是基于 Navya‑Nyāya 经典文本（如 Gaṅgeśa 的《Tattvacakṣaṇam》及其评论）作为语料源；

**📈 对比分析**

与之前的第一阶、第二阶逻辑以及 Martin‑Löf 形式化相比，本文的 CTT 编码在保持所有结构细节的同时，实现了可判定的类型检查；实验表明，在 Cubical Agda 上的实现能够在有限时间内完成类型检查，且没有发现不可判定的环节；

**⚠️ 局限性**

局限性包括：对 śābda 语义的处理尚不完整；Raghunātha 对范畴的修改需要进一步重构；对更早期 Nyāya 文献的系统性覆盖仍待完成；以及对“kevalavyatireki”争议的理论争议尚未在 CTT 内部得到解决。

---

## 174. Do Androids Dream of Breaking the Game? Systematically Auditing AI Agent Benchmarks with BenchJack

**arXiv ID:** 2605.12673 | [PDF](https://arxiv.org/pdf/2605.12673v1)

**作者:** Hao Wang `[一作]` (UC Berkeley), Dawn Song `[通讯]` (UC Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了自动化红队工具 Agent-Eval 与迭代改进框架，用于在代理基准中自动发现并修补奖励黑客漏洞。

**💡 创新点**

创新点在于提出了八类奖励黑客缺陷分类，基于此构建了 Agent-Eval 检查表，并实现了可自动生成可验证黑客的工具与生成对抗式迭代修补循环。

**🔧 技术方法**

使用了静态代码分析、运行时动态审计、LLM 编程代理（如 Claude Code）生成黑客脚本，以及生成对抗式红队–修补循环技术。

**📊 数据集**

对 10 个主流代理基准（包括 SWE-bench、WebArena、OSWorld、Terminal-Bench 等）进行审计。

**📈 对比分析**

通过在 10 个基准上评估黑客率与修补后黑客率，并与原始评估得分对比，结果显示工具可在未完成任务的情况下获得近 100% 分数；迭代修补后，未被修补任务的可被黑客利用比例从近 100% 降至 <10%。

**⚠️ 局限性**

局限性包括：仍受限于 LLM 能发现的缺陷，无法修补设计层面的安全错误；对极其复杂或多样化任务的黑客率难以完全消除。

---

## 175. Synthesizing the Expert: A Validated Multimodal Dataset for Trustworthy AI-Assisted Swimming Coaching

**arXiv ID:** 2605.12799 | [PDF](https://arxiv.org/pdf/2605.12799v1)

**作者:** Ahmad Al-Kabbany `[一作]` (Arab Academy for Science and Technology), Esraa Kassem `[通讯]` (Alexandria University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一套基于多代理LLM的Meta-Synthesis Pipeline，利用运动传感器、体能数据、教练手册等多模态知识库，自动生成1,864条经专家验证的问‑语境‑答（Q‑C‑A）金三元组，用作可信的检索增强生成（RAG）训练数据。

**💡 创新点**

创新点包括：①三阶段代理架构（Architect、Generator、Critic）实现知识合成与生理合理性双重校验；②制定12条领域专属生理合法性规则，系统化地防止模型幻觉与错误建议；③通过人机循环验证生成内容，产出公开可复现的专家级基准数据集；④实现对高频IMU、体能指标等结构化数据的语义检索与多模态融合，填补了现有RAG在运动科学中的空白。

**🔧 技术方法**

主要技术手段有：GPT‑4o多代理LLM、Chroma向量数据库与OpenAI Embedding、Python自定义知识库构建与分块、规则引擎（12条生理校验规则）、人机交互验证流程；整个流程通过结构化JSON文件实现状态检查与容错。

**📊 数据集**

使用的数据集包括：①181,389个语义块（376个源文件）——涵盖比赛成绩、体能档案、10‑IMU传感器记录、教练手册、医学文献；②1,000名游泳运动员的生理参数（VO₂max、HRV、乳酸阈值等）；③多模态CSV与文档的两级叙事序列化；④跨项运动基准（自行车、跑步等）以提供参考。

**📈 对比分析**

与传统无检索或自定义GPT的对比：系统在生理合法性审核中达到97.4%接受率，重构循环成功率79.1%；相较于基线RAG，生成的三元组在检索精确性、上下文匹配度和生理合理性方面显著提升。尽管未给出具体数值评分，但通过规则违例统计和人机验证证明其优于现有方案。

**⚠️ 局限性**

局限性：①合成数据与真实运动员反应存在差距，需进一步现场验证；②使用的是群体水平阈值，难以捕捉个体差异与心理因素；③目前仅聚焦游泳，跨运动推广仍需实验；④生成的三元组虽已通过规则校验，但仍可能因训练时动态环境未被充分建模而出现偏差。

---

## 176. IV-ICL: Bounding Causal Effects with Instrumental Variables via In-Context Learning

**arXiv ID:** 2605.12924 | [PDF](https://arxiv.org/pdf/2605.12924v1)

**作者:** Vahid Balazadeh `[一作]` (University of Toronto), Rahul G. Krishnan `[通讯]` (University of Toronto)

**通讯引用:** 2491 | [OpenAlex ID](https://openalex.org/A5073514348)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于摊销式贝叶斯推断、使用内在KL作为损失的算法，在IV模型中直接学习因果效应的后验分布，并通过其分位数构造识别区间；

**💡 创新点**

创新点在于采用mass‑covering的内在KL优化，能够恢复整个识别集；同时给出了将随机对照试验数据转换为IV基准的可行方案；

**🔧 技术方法**

使用摊销式贝叶斯推断（in‑context learning）、内在KL损失、神经网络近似后验，以及与传统变分推断、半参数、贝叶斯与插件方法的对比实验；

**📊 数据集**

在合成数据、半合成IV基准以及通过RCT转化得到的IV数据集上进行评估；

**📈 对比分析**

与高效半参数、贝叶斯和插件基线相比，所提方法在区间有效性和信息量方面更佳，并且推断时间比基线快20–500倍；

**⚠️ 局限性**

局限性包括对模型容量和超参数选择的敏感性，以及对复杂真实世界IV结构的泛化能力仍需进一步验证。

---

## 177. Certified Robustness under Heterogeneous Perturbations via Hybrid Randomized Smoothing

**arXiv ID:** 2605.12876 | [PDF](https://arxiv.org/pdf/2605.12876v1)

**作者:** Blaise Delattre `[一作]` (Institute of Science Tokyo), Yang Cao `[通讯]` (Institute of Science Tokyo)

**通讯引用:** 7216 | [OpenAlex ID](https://openalex.org/A5045938742)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的混合随机平滑方法，能够在同一框架下为离散和连续输入同时提供鲁棒性证明。

**💡 创新点**

创新点在于将 Neyman–Pearson 公式与离散-连续混合噪声结合，得到一维可逆的闭式证书，既包含 Gaussian 方案也包含离散 knapsack 方案，克服了传统单模态不相容的局限。

**🔧 技术方法**

采用混合噪声模型（离散噪声 + 高斯噪声）、Neyman–Pearson 最大化法、根求解与分数背包优化，构成可计算的证书算法。

**📊 数据集**

在实验中使用了成人收入数据集（混合离散连续特征）以及经过筛选的 Hateful Memes 交互式安全样本（文本+图像），并在 LLaVA-Guard 视图上验证。

**📈 对比分析**

与单模态（仅图像或仅文本）随机平滑进行比较，混合证书在图像半径上与图像单模保持相近，文本鲁棒性与文本单模相似；在混合攻击下，证书覆盖率约为 70%–80%，已显著优于单模态组合。

**⚠️ 局限性**

局限包括计算开销大（需大量模型前向传播）、证书保守（依赖于单纯的平滑概率，可能过度悲观），以及对交互式安全失败样本的覆盖率有限，未能覆盖更广泛的多模态安全基准。

---

## 178. Protocol-Driven Development: Governing Generated Software Through Invariants and Evidence

**arXiv ID:** 2605.12981 | [PDF](https://arxiv.org/pdf/2605.12981v1)

**作者:** Jun He `[一作]` (OpenKedge), Deying Yu `[通讯]` (OpenKedge)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出 Protocol-Driven Development（PDD）模型，定义协议为结构、行为和运维三类不变式，设计 Validator Loop 与 Evidence Chain 以实现生成代码的治理与审计，并给出理论基础、参考架构与案例研究。

**💡 创新点**

创新点在于将协议（Typed Handshake、Behavioral Invariant、Capability Manifest）提升为主导的、可机器验证的治理 artefact；将正式方法、属性测试、policy‑as‑code 与软件溯源整合为统一的证据驱动的可插拔实现框架；通过证据链实现可重现、可审计的代码接受决策。

**🔧 技术方法**

采用形式化方法（Hoare logic、TLA+、SMT）、属性测试框架、Typed Handshake 规范（OpenAPI/Protocol Buffers/JSON Schema）、Capability Manifest（沙箱/策略引擎）、验证引擎、数字签名与加密哈希以构建 Evidence Chain。

**📊 数据集**

未使用任何公开数据集；论文通过示例性案例（idempotent handler、ETL pipeline、微服务）演示协议设计与验证流程。

**📈 对比分析**

未进行实验比较；论文提出了未来评估议程（歧义降低、可再生性、可替换性、验证成本、治理效果、证据可复现性），但目前缺乏实际性能指标与对比结果。

**⚠️ 局限性**

局限性包括：需要高质量、无误的协议设计与可验证的验证器；部分不变式可能不可判定或验证成本高昂；作者工作量相对传统 SDD/TDD 可能更大；适用于能明确定义结构/行为/运维边界的系统，对小型或低风险项目的收益不明显。

---

## 179. Discrete MeanFlow: One-Step Generation via Conditional Transition Kernels

**arXiv ID:** 2605.12805 | [PDF](https://arxiv.org/pdf/2605.12805v1)

**作者:** Fairoz Nower Khan `[一作]` (University of Kentucky), Peizhong Ju `[通讯]` (University of Kentucky)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5085838919)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 Discrete MeanFlow，直接学习离散状态空间的转移核，实现一次性生成；

**💡 创新点**

引入离散 MeanFlow 身份，将平均离散速率与 CTMC 生成器关联，配合边界按构造参数化保证合法概率分布；

**🔧 技术方法**

使用连续时间马尔科夫链理论、Kolmogorov 前向方程、Transformer 基础网络、自动微分和离散采样；

**📊 数据集**

在已知解析解的有限状态 CTMC（2、3、10 状态）上验证，及在合成序列任务（词表大小 4/8/16，长度 8/16/32）上进行实验；

**📈 对比分析**

与显式边界损失、交叉熵监督以及后验回归等方法对比，边界按构造方法误差比显式约束低 3–10 倍，one‑step 样本的总变差距 0.03 左右，尽管 kernel‑residual 目标方差高于后验回归；

**⚠️ 局限性**

主要局限在于 kernel‑residual 目标方差高，导致梯度估计不稳定，未提出低方差估计方法；

---

## 180. Resolution Information: Limits of Ambiguity Resolution for Generative Communication

**arXiv ID:** 2605.12800 | [PDF](https://arxiv.org/pdf/2605.12800v1)

**作者:** Angeles Vazquez-Castro `[一作]` (Universitat Autonoma De Barcelona), Zhu Han `[通讯]` (University Of Houston)

**通讯引用:** 91171 | [OpenAlex ID](https://openalex.org/A5063667378)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出了生成式通信中“分辨率信息”概念，用于量化接收方通过生成式表示将后验分布压缩至低歧义语义区域所需的最小信息量，并推导了对应的理论极限、指数收敛性与生成可解析性。

**💡 创新点**

创新点在于：① 将信息理论中的不确定性消除原则直接应用到语义歧义上；② 定义并解析分辨率信息，揭示后验空间受限时几何结构会导致不可消除的歧义底层；③ 提出了“生成可解析性”概念，类似香农容量，但针对语义解析而非错误率。

**🔧 技术方法**

主要技术：信息投影、KL 散度最小化、Sanov 定理、大偏差理论、信息几何、Gaussian 后验分析以及对半空间和多面体语义区域的解析。

**📊 数据集**

本文为理论分析，无具体实验数据集；若做实验，可在常见生成模型（VAE、Diffusion、LLM 等）上评估后验逼近能力。

**📈 对比分析**

本文未进行实验比较，给出了理论上限与指数收敛性（ambiguity exponent）以及对不同几何形状的解析性评估；在理论层面上，半空间可实现零歧义，凸多面体则产生不可消除的歧义底层。

**⚠️ 局限性**

局限性：仅在理想化假设（高斯后验、已知先验、可解析闭式）下推导；未考虑实际生成模型的非高斯特性、模型容量约束以及多模态语义区域的复杂几何；未来需验证在真实生成器上的有效性。

---

## 181. Training Large Language Models to Predict Clinical Events

**arXiv ID:** 2605.12817 | [PDF](https://arxiv.org/pdf/2605.12817v1)

**作者:** Benjamin Turtel `[一作]` (Lightning Rod Labs), Kris Skotheim `[通讯]` (Lightning Rod Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用MIMIC-III随时间记录的临床笔记构造时间序列问答对，并用Foresight Learning训练轻量级LoRA适配器，生成可直接用于临床事件预测的模型；

**💡 创新点**

提出一种可复用的工作流，将原始未结构化笔记转化为自然语言问题与标注，无需手工特征或端点特定分类器，且通过时间先后关系实现监督；

**🔧 技术方法**

采用Foresight Learning框架、Gemini 2.5 Flash进行问题生成与标签判定，利用120B GPT模型冻结权重，仅微调LoRA适配器，训练目标为log-score并通过GRPO优化；

**📊 数据集**

使用MIMIC-III v1.4数据集，包含702个入院轨迹，产生6900个预测问题（500个测试集），覆盖药物、程序、器官支持、微生物学结果及死亡等事件；

**📈 对比分析**

将训练好的模型与常数基线、prompted gpt‑oss‑120b及GPT‑5进行对比，评价指标包括reward、Brier、ECE、AUROC与top‑10% lift；训练模型在所有指标均优于基准，略优于GPT‑5（如reward‑0.4586对‑0.4636，AUROC‑0.7993对‑0.7954，ECE‑0.0398对‑0.0422）；

**⚠️ 局限性**

局限性包括：单中心数据（MIMIC‑III）可能不具备代表性；临床笔记噪声、模板化和缺失可能影响信号质量；问题生成与标签判定均为自动化，可能引入错误；仅使用笔记而未整合结构化数据；模型仅在MIMIC‑III上验证，外部泛化能力未知。

---

## 182. Is Video Anomaly Detection Misframed? Evidence from LLM-Based and Multi-Scene Models

**arXiv ID:** 2605.12725 | [PDF](https://arxiv.org/pdf/2605.12725v1)

**作者:** Furkan Mumcu `[一作]` (University of South Florida), Yasin Yilmaz `[通讯]` (University of South Florida)

**通讯引用:** 2338 | [OpenAlex ID](https://openalex.org/A5036320427)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文通过对比实验与可视化分析，评估了当前多场景与弱监督、训练无监督视频异常检测方法在单场景数据集上的表现，并指出其对场景特定性与空间定位的不足；

**💡 创新点**

强调单摄像头部署下的场景特定、位置与上下文依赖、可解释性的重要性，批判了多场景泛化与LLM驱动方法的局限，并提出应重新聚焦无偏、空间感知的单场景模型；

**🔧 技术方法**

采用AUC与相对性能下降（RPD）指标对方法进行评估，并通过视觉对比展示空间定位能力；

**📊 数据集**

实验使用多场景数据集UCF-Crime以及单场景数据集StreetScene（以及Ped1/Ped2/Avenue等）；

**📈 对比分析**

与VADTree、EventVAD、LAVAD、VERA等多场景方法相比，单场景方法Contextual GMM与MLLM-EVAD在StreetScene上保持较高AUC并具备空间定位；多场景方法在单场景测试中的RPD显著高，性能大幅下降；

**⚠️ 局限性**

主要限制在于缺乏场景特定与位置/上下文建模、缺失空间定位与可解释性、LLM与预训练模型带来的语义偏差以及弱监督导致的闭域与动作识别倾向。

---

## 183. ToolWeave: Structured Synthesis of Complex Multi-Turn Tool-Calling Dialogues

**arXiv ID:** 2605.12521 | [PDF](https://arxiv.org/pdf/2605.12521v1)

**作者:** Dinesh Khandelwal `[一作]` (IBM Research), Dinesh Raghu `[通讯]` (International Institute of Information Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 ToolWeave 框架，通过多阶段结构化流程合成真实的多轮工具调用对话，以提升 LLM 在自动代理中的工具使用能力。

**💡 创新点**

创新点在于引入逐步工具图生成与工作流采样、细粒度参数流规划以及多代理对话合成，显著提高多步交互比例并降低参数幻觉。

**🔧 技术方法**

采用的技术包括基于 LLM 的多阶段工具图合成、结构化工作流采样、细粒度规划（参数来源追踪）、多代理对话生成以及后处理（语言多样化、故障注入）。

**📊 数据集**

数据集方面使用了自研的 ToolWeave 生成的工具和对话，覆盖 20 个领域的合成 API，以及对比的公开基准 BFCL‑V3、API Bank、CONFETTI。

**📈 对比分析**

在 BFCL‑V3 等三大多轮工具调用基准上，Fine‑tuned LLM 在 ToolWeave 数据上平均提升 30–40 分（以准确率计），单模型 39.75% 的多轮准确率远超 ToolFlow 的 23.5%。

**⚠️ 局限性**

局限性包括仍存在 20% 的参数值幻觉，合成的工具图与真实 API 的语义差异可能导致跨领域泛化受限，以及对模型规模的进一步验证仍需深入。

---

## 184. MMCL-Bench: Multimodal Context Learning from Visual Rules, Procedures, and Evidence

**arXiv ID:** 2605.12703 | [PDF](https://arxiv.org/pdf/2605.12703v1)

**作者:** Yifan Chen `[一作]` (University of Cambridge), Yujiu Yang `[通讯]` (Tsinghua University)

**通讯引用:** 4133 | [OpenAlex ID](https://openalex.org/A5020953714)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个新的多模态上下文学习基准（MMCL），包含102个任务，涵盖规则系统应用、程序执行和经验发现等三大类，要求模型从多模态教学上下文中学习局部规则或程序，并将其迁移到新的视觉实例上。

**💡 创新点**

创新点在于将多模态上下文学习单独作为一种能力进行系统评估，设计了严格的基于评分规则的评价体系，并通过诊断性消融与错误分类揭示感知、定位、推理和输出构造等四个关键瓶颈。

**🔧 技术方法**

技术手段包括利用前沿多模态大语言模型（如GPT‑5.4、Gemini 3.1 Pro、Claude Opus等）进行延伸推理，构建视觉状态和文本状态的oracle消融实验，并对错误进行四类标签的自动化编码与分析。

**📊 数据集**

数据集为MMCL基准，由人工与智能体协同生成的合成与真实多模态教学–测试对构成，涵盖图片、截图、手册、视频和帧序列等多模态素材，整体规模为102个任务。

**📈 对比分析**

模型比较采用严格的全规则通过/失败判定和按规则级别的准确率统计；在最佳模型GPT‑5.4‑Thinking上，严格通过率为26.5%，其余模型约为19–20%；oracle实验显示若将视觉状态或完整文本提供，准确率可提升至75%，说明当前模型与理想解之间存在较大差距。

**⚠️ 局限性**

局限性包括基准规模有限、缺乏人类或专家基准、评测依赖仅文本的LLM评判器、诊断样本量小以及未充分覆盖真实世界复杂多模态场景。

---

## 185. TrackCraft3R: Repurposing Video Diffusion Transformers for Dense 3D Tracking

**arXiv ID:** 2605.12587 | [PDF](https://arxiv.org/pdf/2605.12587v1)

**作者:** Jisu Nam `[一作]` (KAIST AI), Seungryong Kim `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种单次前向的 3D 跟踪方法，利用视频扩散变换器（DiT）在单帧参考坐标下实现对整个视频中每个像素的 3D 轨迹与可见性预测。

**💡 创新点**

创新点在于：① 通过双 latent 结构，将每帧的 RGB 与点图 latent 与首帧 anchor 的 track latent 结合；② 使用时序 RoPE 对 track latent 的目标时间点进行对齐；③ 采用 LoRA 微调将 generative 的 DiT 转化为单步回归器，从而实现无迭代、无 4D 相关特征的高效跟踪。

**🔧 技术方法**

核心技术包括视频扩散变换器（DiT）、VAE 编码解码、三维 RoPE、LoRA 微调以及残差轨迹回归和可见性解码。

**📊 数据集**

训练使用 Kubric、PointOdyssey、Dynamic Replica 与 TartanAir 等合成/现实视频数据集，评估数据集包括 TAPVid-3D、ADT、PStudio、Point Odyssey、Dynamic Replica 以及 Kubric 测试集。

**📈 对比分析**

与现有迭代式和前向式 3D 跟踪器（如 DELTA、DELTAv2、St4RTrack、Any4D、TraceAnything、MotionCrafter 等）对比，本文方法在 APD3D、AJ、OA 指标上均取得领先，并且推理速度快 1.3 倍、峰值显存仅为原先的 1/4.6。

**⚠️ 局限性**

局限性主要在于对输入几何估计的依赖：若深度与相机位姿不准确，跟踪误差会显著上升；此外方法仍基于单帧参考框架，对大尺度跨帧结构变形的鲁棒性有待进一步验证。

---

## 186. DQN-Driven Adaptive Neighbor Discovery for Directional Aerial Networks

**arXiv ID:** 2605.12552 | [PDF](https://arxiv.org/pdf/2605.12552v1)

**作者:** Md Asif Ishrak Sarder `[一作]` (University of Central Florida), Elizabeth Bentley `[通讯]` (Air Force Research Lab)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种基于深度Q网络的自适应天线转发器选择框架，用于在移动方向性无人机网络中同时实现邻居发现与隐私保护。

**💡 创新点**

创新点在于将邻居发现与隐私曝光的冲突目标通过可调权重融合，并让每个节点独立学习探测方向，以实现局部观察下的全局连通性与隐私平衡。

**🔧 技术方法**

采用了独立深度Q网络（IDQN）强化学习、ε‑greedy动作选择、经验回放等技术，并设计了基于探测历史、结果和分布的状态向量以及加权目标函数。

**📊 数据集**

实验使用基于Python的仿真环境，部署100m×100m网格上的UAV群组和30m范围的方向性节点与非期望用户，未使用公开真实数据集。

**📈 对比分析**

与随机探测和传统Q学习基线比较，DQN框架在不同权重设置下显著提升邻居发现效率和网络连通率，同时保持更低的被窃听概率，整体目标得分最高。

**⚠️ 局限性**

局限包括缺乏对真实环境的验证、未实现节点对全局连通性的局部估计、依赖纯仿真、未考虑节点间协作或全局状态信息。

---

## 187. A detailed algorithmic study on a reuse-aware, near memory, all-digital Ising machine

**arXiv ID:** 2605.12959 | [PDF](https://arxiv.org/pdf/2605.12959v1)

**作者:** Siddhartha Raman Sundara Raman `[一作]`, Jaydeep P. Kulkarni `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

无法获取论文具体内容

**💡 创新点**

无法获取创新点

**🔧 技术方法**

无法获取使用的技术

**📊 数据集**

无法获取使用的数据集

**📈 对比分析**

无法获取比较方法与性能

**⚠️ 局限性**

无法获取局限性

---

## 188. MorphOPC: Advancing Mask Optimization with Multi-scale Hierarchical Morphological Learning

**arXiv ID:** 2605.12528 | [PDF](https://arxiv.org/pdf/2605.12528v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 189. U-HNO: A U-shaped Hybrid Neural Operator with Sparse-Point Adaptive Routing for Non-stationary PDE Dynamics

**arXiv ID:** 2605.12965 | [PDF](https://arxiv.org/pdf/2605.12965v1)

**作者:** Yingzhe Ma `[一作]` (University of Electronic Science and Technology of China), Jinliang Liu `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了U‑HNO，一种结合多尺度高斯局部分支与全局傅里叶分支的U‑形混合神经算子，并通过Sparse‑Point Adaptive Routing (SPAR)实现每像素的硬切换。

**💡 创新点**

创新点在于：① 采用SPAR实现空间自适应的分支路由；② 将双分支融合嵌入U‑形骨干并采用非对称解码器；③ 通过结构化损失（H^1梯度项+跨分支一致性）提升长时序稳定性与梯度准确性。

**🔧 技术方法**

使用的技术包括：傅里叶卷积、深度可分离高斯卷积、U‑形编码解码器、Top‑k硬门控（带STE）、自适应保持比例、MSE+H^1+CBC联合损失。

**📊 数据集**

主要数据集为PDEBench的八个任务，包括1D Burgers、KS、KdV；2D advection、Allen‑Cahn、Navier‑Stokes、Darcy；以及3D转子流（128³）Navier‑Stokes，覆盖从光滑传输到尖锐冲击的多种场景。

**📈 对比分析**

与七种基准算子（FNO、GNO、CNO、WNO、FFNO、Conv‑FNO、LogLo‑FNO）在相同参数量下对比；在绝大多数任务中，U‑HNO在relL2和relH1上取得最优或仅略逊；在冲击支配任务（如1D Burgers、3D‑CFD）提升幅度高达4×，并在长时序上保持稳定。

**⚠️ 局限性**

局限性：① 需要额外的分支和路由门，调参面更复杂；② SPAR是全局计算的“代表性分派”，未实现真正的稀疏算子加速；③ 仅适用于规则网格，难以直接迁移到非结构化网格或点云；④ 训练与推理采用固定时间步，无法自适应变步长。

---

## 190. "F*** You Biden": Cross-Partisan Electoral Toxicity on X

**arXiv ID:** 2605.12526 | [PDF](https://arxiv.org/pdf/2605.12526v1)

**作者:** Danishjeet Singh `[一作]` (Indiana University), Filippo Menczer `[通讯]` (Indiana University)

**通讯引用:** 25847 | [OpenAlex ID](https://openalex.org/A5021346979)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用 2024 年美国总统选举期间在 X（前 Twitter）上收集的大规模原始推文及其直接回复，结合政治倾向分类和毒性评分，探究党派对立语境下的毒性传播模式，发现共和党推文更毒性高，但民主党推文收到的回复更毒性高，且这一现象主要由共和党跨党派回复量所驱动。

**💡 创新点**

创新点在于揭示推文发布毒性与回复毒性之间的结构性不对称：即产生毒性最高的党派并非收到毒性最高的党派；通过量化跨党派回复的毒性与数量，解释了这一不对称的根源；并首次在大规模选举语料上用 LLM 进行高效的党派标签化与毒性检测。

**🔧 技术方法**

使用技术包括：① GPT‑4o‑mini 对推文与回复进行党派倾向分类；② Perspective API 计算毒性分数；④ Mann‑Whitney U 检验比较毒性分布；⑤ 对用户级别进行多标签投票聚合；⑥ 对数据进行统计分布绘制与效应量（Cliff's δ）评估。

**📊 数据集**

数据集为 2024 年美国总统选举期间收集的约 44 M 条政治关键词推文，聚焦原始推文与直接回复，共 1,853,052 条回复，1,145,347 条回复针对民主党推文，708,705 条回复针对共和党推文；数据来源公开可复现。

**📈 对比分析**

比较方法：对党派推文与回复的毒性使用 Mann‑Whitney U 检验，检验同党派 vs 跨党派回复毒性；对党派推文与回复毒性分别计算中位数并绘制核密度分布。结果显示：共和党推文毒性显著高于民主党推文（p<10⁻⁶，δ≈0.014）；民主党推文收到的回复毒性高于共和党推文（p<10⁻⁶，δ≈0.038）；跨党派回复略高于同党派回复（p<10⁻⁶，δ≈0.16–0.19）。分类器性能：帖子宏 F1≈0.73；回复宏 F1≈0.81。

**⚠️ 局限性**

局限性：① 对“Unsure”类别的分类准确率相对较低，可能引入噪声；② Perspective API 训练语料主要为正式文本，可能低估 X 上的讽刺/暗语毒性；③ 仅采集含政治关键词的推文，导致对跨党派对话的覆盖不完整；④ 结果仅适用于 X 的 2024 选举周期，难以推广到其他平台或非选举时期；⑤ 无法确认毒性不对称的因果机制（算法曝光、主动搜索或有组织行为）。

---

## 191. When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction

**arXiv ID:** 2605.12922 | [PDF](https://arxiv.org/pdf/2605.12922v1)

**作者:** Vardhan Dongre `[一作]` (University of Illinois Urbana Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大语言模型在多轮对话中的指令遵循衰退，提出了注意力通道与残差通道的“通道转移”框架，并设计了 Goal Accessibility Ratio（GAR）指标、滑动窗口干预和残差线性探针来量化与干预这一失效机制；

**💡 创新点**

创新点在于把多轮衰退机制归因于注意力通道的可达性下降并被残差通道替代，提供了可量化的 GAR 诊断、可预测的关闭时点、以及通过残差线性可解码信息的证明，揭示不同架构在通道转移后行为差异；

**🔧 技术方法**

主要技术包括基于 transformer 的注意力矩阵分析、GAR 计算（跨层跨头平均注意力），滑动窗口 mask 结构性关闭注意力通道，残差层线性探针（LDA+PCA）用于预测任务完成情况；

**📊 数据集**

实验数据集由四类结构化多轮任务构成：信息保留（5/20 条事实）、受控复杂度、人格遵循、政策遵守，约 5,500 个对话样本，均为 50 轮长的交互；

**📈 对比分析**

对比方法是默认注意力 vs. 滑动窗口关闭（W=4096）和多窗口尺寸的交叉验证；性能指标显示 GAR 在默认情况下随轮次递减，关闭后任务记忆率大幅下降（Mistral 45% vs. 100%，LLaMA 0% 等），残差线性探针 AUC 最高 0.99，证明信息仍可在残差中线性恢复；

**⚠️ 局限性**

局限性包括仅在结构化、固定目标的 50 轮对话上验证，未覆盖开放式聊天漂移、目标演化或更长对话；滑动窗口干预对非 native SW 模型仅是构造性扰动；线性探针仅捕捉线性可解码信息，可能忽视非线性或分布式读取机制；

---

## 192. Persona-Conditioned Adversarial Prompting (PCAP): Multi-Identity Red-Teaming for Enhanced Adversarial Prompt Discovery

**arXiv ID:** 2605.12565 | [PDF](https://arxiv.org/pdf/2605.12565v1)

**作者:** Cristian Morasso `[一作]` (IBM Research), Douglas Leith `[通讯]` (Trinity College Dublin)

**通讯引用:** 10693 | [OpenAlex ID](https://openalex.org/A5086446911)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于 Persona‑Conditioned Adversarial Prompting (PCAP) 的方法，通过多身份与策略卡的并行 beam search，显著提升 LLM 的 jailbreak 发现效率。

**💡 创新点**

创新点在于将 persona 生成、目标重构与多策略卡条件化融合到搜索流程，扩展攻击空间、实现跨模型可迁移性，并大幅提升攻击成功率。

**🔧 技术方法**

在原有 TAP 框架上引入 persona 生成器、目标重构器、策略卡以及并行多 persona beam search，并利用 on‑topic 与评估 LLM 进行过滤。

**📊 数据集**

实验使用 GPT‑OSS 120B 的 50 个 AdvBench 目标子集，评估模型包括 Llama 3.3 70B、Granite 3.3 8B 与 Granite 4.0 H Tiny。

**📈 对比分析**

与 TAP 基线对比，PCAP 在 Llama 70B 上将攻击成功率从 57.7% 提升至 97.3%，prompt 产出量和查询成本相应增加，但迁移到专用 guardrail 的成功率也大幅提升。

**⚠️ 局限性**

局限性包括：需要大量查询导致计算成本高，依赖 LLM 的内在偏见与评估机制，且仍需人工审查以保证数据质量与安全。

---

## 193. Multistep Belief Space Dynamics Learning For Risk-Aware Control

**arXiv ID:** 2605.12628 | [PDF](https://arxiv.org/pdf/2605.12628v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 194. Divergent Multi-Version Execution (DME): Canonical Instruction-Trace Fault Detection via Structural Address-Space Decorrelation

**arXiv ID:** 2605.12576 | [PDF](https://arxiv.org/pdf/2605.12576v1)

**作者:** Petro Baran Yrievich `[一作]` `[通讯]` (Independent Researcher), Petro Baran Yrievich (Independent Researcher)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Divergent Multi-Version Execution (DME) 机制，通过独立编译和结构地址空间去相关来实现多版本冗余检测。

**💡 创新点**

创新点在于将结构地址空间去相关与规范化指令轨迹对比相结合，既能在全相关错误下实现确定性检测，也能在部分相关或单点错误下提供概率性检测。

**🔧 技术方法**

采用独立多版本编译、功能与块级交叉调度、NOP 插入、块碎片化、地址非重叠和 Canonical Trace 监控技术，并配合双层检测（语义+结构）。

**📊 数据集**

实验使用嵌入式 Cortex‑M 8KB RAM 和 Altera Cyclone IV FPGA，每个核心 3000 LUT、16KB BRAM 进行验证，并在这些平台上进行基准测试。

**📈 对比分析**

与传统 TMR/EDDI/SWIFT 等方法对比，DME 在完全相关 PC/指针错误时实现确定性检测，误检率可通过 NOP 频率调节；实验显示在同等资源下 DME 具有相近或略低的延迟和占用，并显著提升了错误检测率。

**⚠️ 局限性**

局限在于仍无法检测保持 Canonical Trace 相同的错误（如某些语义保持错误），以及实现时需要额外的硬件/编译开销；对共享内存区域的处理也需手工排除。

---

## 195. Bridge: Optimizing Collective Communication Schedules in Reconfigurable Networks with Reusable Subrings

**arXiv ID:** 2605.12766 | [PDF](https://arxiv.org/pdf/2605.12766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 196. WildPose: A Unified Framework for Robust Pose Estimation in the Wild

**arXiv ID:** 2605.12774 | [PDF](https://arxiv.org/pdf/2605.12774v1)

**作者:** Jianhao Zheng `[一作]` (Stanford University), Iro Armeni `[通讯]` (Stanford University)

**通讯引用:** 3207 | [OpenAlex ID](https://openalex.org/A5014426007)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `51c0528b-f690-4182-ae60-bb5f046c276c` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种统一的单目相机位姿估计框架，能够在动态与静态环境中鲁棒工作，兼具实时性与高精度；

**💡 创新点**

将预训练的3D感知MASt3R特征融入可微束调整（BA）管线，设计新的3D感知更新算子和高容量运动遮罩检测器，实现动态遮挡抑制与姿态优化的无缝融合；

**🔧 技术方法**

融合MASt3R ViT编码器、轻量级适配器、ConvGRU迭代网络、可微束调整、基于多尺度特征的运动遮罩头，并使用多任务损失和阶段化训练；

**📊 数据集**

使用多样化的合成与真实数据集：TartanAir V2、TartanGround、Dynamic Replica、OmniWorld-Game、Kubric模拟器以及 Wild-SLAM、Bonn RGB‑D、TUM RGB‑D、MPI Sintel、7‑Scenes、ScanNet 等基准集；

**📈 对比分析**

在动态、低运动、静态三类基准上与 WildGS‑SLAM、ViPE、MegaSaM、DROID‑SLAM、MASt3R‑SLAM、VGGT、π³ 等方法对比，取得全景轨迹误差最低（ATE 下降 15‑25%），并在后续深度估计任务中进一步提升精度；

**⚠️ 局限性**

在极短序列或缺乏足够视角覆盖时，基于3D Gaussian splatting 的方法效果下降；全局 BA 对尺度误差仍存在一定敏感性，且在某些照明变化严重的动态数据上表现略逊于专门处理光照变化的 WildGS‑SLAM。

---

## 197. CRAFT: Clinical Reward-Aligned Finetuning for Medical Image Synthesis

**arXiv ID:** 2605.12650 | [PDF](https://arxiv.org/pdf/2605.12650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 198. User Reviews as a Source for Usability Requirements: A Precursor Study on Using Large Language Models

**arXiv ID:** 2605.12657 | [PDF](https://arxiv.org/pdf/2605.12657v1)

**作者:** Cedric Wellhausen `[一作]` (Leibniz University Hannover), Kurt Schneider `[通讯]` (Leibniz University Hannover)

**通讯引用:** 5268 | [OpenAlex ID](https://openalex.org/A5031529088)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了包含300条来自三款不同类型应用的用户评价数据集，并探讨利用大语言模型（LLM）在用户中心需求工程中识别可用性相关评价的可行性，比较LLM与人工评审在此任务上的表现；

**💡 创新点**

创新点在于：①首次将LLM直接应用于无训练的可用性评价识别；②通过三轮迭代的提示工程（prompt engineering）实现针对性标签；③提供完整标注的数据集作为后续研究基准；

**🔧 技术方法**

使用技术包括OpenAI的GPT‑4.1 LLM、链式思考（chain‑of‑thought）提示、基于Nielsen 10条可用性启发式的编码准则；

**📊 数据集**

使用的数据集为从Google Play商店抽取的300条用户评价，涵盖BlitzerDE、Lidl Plus和FL Studio三款应用；

**📈 对比分析**

评估方法为精确度、召回率、F1分数及与人工标注的Cohen’s Kappa一致性。结果显示LLM在各应用中的召回率均高于0.82，精确度约0.73‑0.79，F1分数在0.78‑0.86之间，Cohen’s Kappa从公正到显著一致不等；

**⚠️ 局限性**

局限性包括：仅评估三款应用和单一LLM模型，提示敏感度高；评审者人数有限，可能影响标注质量；使用英文提示评估德语评论导致跨语言偏差；整体可靠性尚未达到替代人工的程度。

---

## 199. Project Life Cycles in Open-Source Software

**arXiv ID:** 2605.12738 | [PDF](https://arxiv.org/pdf/2605.12738v1)

**作者:** Sanjiv Das `[一作]` (Amazon), Brian Granger `[通讯]` (Amazon)

**通讯引用:** 10405 | [OpenAlex ID](https://openalex.org/A5034549465)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文将内生增长理论与产品生命周期模型相结合，建立了一套双ODE框架，用以描述开源软件项目的开发者参与度与代码增长的动态演进，并通过对GitHub仓库的提交和贡献者数据进行校准，预测项目的生命周期与终值；

**💡 创新点**

创新点在于：①将产品生命周期的S形曲线模型迁移至开源项目；②构建包含开发者参与与代码增长相互耦合的双ODE系统；③利用闭式解和最小二乘校准实现对项目成熟度与终值的定量预测；

**🔧 技术方法**

使用的技术包括：Cobb‑Douglas内生增长模型、Diffusion（Bass）模型、常微分方程求解与参数优化、统计回归、R²与t检验等；

**📊 数据集**

数据集主要来源于GitHub公开仓库的提交记录（行代码增删）与贡献者信息，选取了十个流行开源项目（如pandas、numpy、pytorch等）进行实验；

**📈 对比分析**

通过将模型拟合结果与真实数据比较（R²>0.95），并用前75%数据与完整数据进行两次校准，验证预测的鲁棒性；在大多数项目中，模型对未来发展路径的预测精度很高，显示出良好的拟合与预测性能；

**⚠️ 局限性**

局限性包括：①对开发者参与度的假设（独立与网络效应分离）在部分项目中不适用；②模型对外部冲击（如重大更新、社区事件）缺乏动态响应；③使用行代码增删作为成长指标可能掩盖代码质量和架构改动；④对需求侧价值估算高度依赖下载与代码变更比率的假设，存在较大不确定性。

---

## 200. Quieting the Cobwebs: Browser Interaction for Visual Floaters

**arXiv ID:** 2605.12739 | [PDF](https://arxiv.org/pdf/2605.12739v1)

**作者:** Kenneth Ge `[一作]` (Assistivity), Shikhar Ahuja `[通讯]` (Georgia Institute of Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并实现了基于眼部物理学的漂浮物（floaters）仿真、一个可量化评估文本可读性的计算管线以及一个通过RSVP和平移模式减少眼动的Chrome扩展。

**💡 创新点**

创新点包括：①首个结合神经适应、旋转运动和形变的浮动物仿真；②利用OCR评估文本与布局在不同漂浮速度下的可读性；③将RSVP与“手持视图”平移模式结合，形成无眼动的网页浏览与阅读工具。

**🔧 技术方法**

技术手段包括：2D物理仿真（XPBD）、光学字符识别（OCR）、Chrome Manifest V3扩展、Pointer Lock API、translate3d平滑平移、双击键盘激活与ORP算法。

**📊 数据集**

数据集为：通过仿真生成的漂浮物视频（随机化速度与形态）、覆盖在标准文本上的漂浮遮挡层、以及OCR识别结果的误差统计；未使用公开真实漂浮物图像或用户阅读实验数据。

**📈 对比分析**

比较方法：使用OCR识别率（WER/CER）评估不同无衬线字体、不同文本布局以及慢速/快速漂浮条件的可读性；实验显示单列布局和慢速漂浮在OCR误差上显著优于宽列、双列和快速漂浮（WER最低为0.877，最高为0.990）。

**⚠️ 局限性**

局限性：仿真仍未达到真实漂浮物的细节和多样性；缺乏真实用户测试，难以验证扩展对实际使用者的帮助；扩展主要适用于键盘操作，可能不适合所有人群；并且对非文本视觉元素的支持仍有限。

---

## 201. REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations

**arXiv ID:** 2605.12813 | [PDF](https://arxiv.org/pdf/2605.12813v1)

**作者:** Buyun Liang `[一作]` (University of Pennsylvania), René Vidal `[通讯]` (University of Pennsylvania)

**通讯引用:** 26614 | [OpenAlex ID](https://openalex.org/A5011256828)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了REALISTA框架，用输入相关的潜在编辑词典与连续潜在空间优化相结合，生成既保持语义等价又具有语义连贯性的对抗提示，诱发LLM幻觉；

**💡 创新点**

创新点在于构造可控的潜在编辑词典，使用对数词典的线性组合并在缩放的单纯形约束下优化；结合语义等价检查器和Gumbel‑Softmax重参数化实现可微分的离散重构；

**🔧 技术方法**

核心技术包括LLM潜在空间编码/解码、线性潜在编辑方向、ℓ1单纯形约束、投影拉格朗日动力学(PLD)、语义等价与连贯性评估LLM；

**📊 数据集**

实验使用MMLU子集（347道题）评估，涉及四款开源LLM（Llama‑3‑3B/8B，Qwen‑2.5‑7B/14B）和两款商业推理模型（GPT‑5‑Nano/mini）；

**📈 对比分析**

与SECA、LARGO、ICD等基线对比，REALISTA在ASR@30上相对SECA提升约10‑20%且语义误差（SEE、SCE）接近或低于SECA，且能成功攻击具自由形式输出的商业模型；

**⚠️ 局限性**

局限在于仅使用线性组合的编辑方向，可能无法覆盖更复杂的语义变换；需要外部对抗提示生成器做潜在词典构造，计算成本较高，且对不同LLM的泛化需进一步验证；

---

## 202. Bayesian Model Merging

**arXiv ID:** 2605.12843 | [PDF](https://arxiv.org/pdf/2605.12843v1)

**作者:** Kaiyang Li `[一作]` (University of Connecticut), Shihao Ji `[通讯]` (University of Connecticut)

**通讯引用:** 4250 | [OpenAlex ID](https://openalex.org/A5036338045)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种贝叶斯模型融合（BMM）框架，能够在不重新训练或缺少原始训练数据的情况下，将多个任务专用专家模型合并为一个统一模型。

**💡 创新点**

创新点在于：①利用已有的强锚模型作为先验，构建基于激活的贝叶斯线性回归并给出闭式解；②通过全局贝叶斯优化（BO）对不同模块的正则化强度进行协调，解决模块异质性问题；③推导数据‑free 变体，利用激活统计与任务向量的对齐关系实现无需校准集的融合。

**🔧 技术方法**

主要技术包括：贝叶斯线性回归、闭式 MAP 估计、Gaussian Process 基础的贝叶斯优化、模块级权重拆分与块级参数绑定、以及任务向量与激活矩阵的理论对齐。

**📊 数据集**

实验使用了视觉任务的 ViT-B/32、ViT-L/14（分别在 8/14/20 任务融合）和语言任务的 Llama-3.2-3B、Llama-3.1-8B（5 任务融合）等公开基准数据集。

**📈 对比分析**

与 TA、TIES、TSV、WUDI‑Merging、RegMean 等现有融合方法以及单独 fine‑tuned 专家模型进行比较。BMM 在数据‑助和数据‑free 两种设置下均显著提升性能，尤其对弱锚模型提升高达 21% 以上，且在 ViT‑L/14 8 任务融合中仅落后单个专家平均 0.7%。

**⚠️ 局限性**

主要限制是实验规模截至 8B 参数级模型，较大模型的评估受限于计算资源；此外 BMM 仍依赖少量验证集进行超参数搜索，对极低可用数据的极端情况可能不够鲁棒。

---

## 203. DelAC: A Multi-agent Reinforcement Learning of Team-Symmetric Stochastic Games

**arXiv ID:** 2605.12555 | [PDF](https://arxiv.org/pdf/2605.12555v1)

**作者:** Duan-Shin Lee `[一作]` (National Tsing Hua University), Yu-Hsiu Hung `[通讯]` (MediaTek Inc)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究并解决了具有 m≥2 个团队的团队对称博弈，提出了一种基于 actor‑critic 的多智能体强化学习算法（DelAC）以学习团队对称纳什均衡。

**💡 创新点**

创新点包括：①证明团队对称博弈必定存在团队对称纳什均衡；②将对称性利用到线性互补问题（LCP）的建模中，显著降低计算复杂度；③在多智能体学习中引入“委托” actor‑critic 结构，利用对称性只需学习每个团队的代理网络。

**🔧 技术方法**

使用了线性互补问题求解、Actor‑Critic 网络、KL 散度目标、中心化训练/分散执行（CTDE）框架，并采用多智能体 Q 值与 Nash 求解结合的方式。

**📊 数据集**

实验数据集为 30 个随机生成的双团队对称博弈（零和与一般博弈）以及一个特定的 Generalized Matching Pennies（GMP）博弈。

**📈 对比分析**

与 IQL、NashQ、FFQ、QMIX、NWQMIX、IA2C、IPPO、CA2C、MAPPO 等基准方法进行比较，DelAC 在平均 MSE、KL 散度和收敛速度上均优于其它方法，尤其在混合策略环境中表现突出。

**⚠️ 局限性**

局限性在于仅在 m=2 的小规模团队对称博弈上进行了验证，且假设团队内部支付完全相同，未探讨更一般的多团队或非对称支付情况。

---

## 204. UFO: A Domain-Unification-Free Operator Framework for Generalized Operator Learning

**arXiv ID:** 2605.12700 | [PDF](https://arxiv.org/pdf/2605.12700v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 205. Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning

**arXiv ID:** 2605.12906 | [PDF](https://arxiv.org/pdf/2605.12906v1)

**作者:** Siyuan Liu `[一作]` (Tsinghua University), Jingzhao Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 3764 | [OpenAlex ID](https://openalex.org/A5064766573)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在大语言模型的监督式微调（SFT）过程中，数据难度与数据规模如何共同影响模型性能，并系统地探索了不同难度数据在不同规模下的最优选取。

**💡 创新点**

创新点在于提出并验证了“最优难度随数据规模递增”的规律，并通过解析一般化误差与外推误差的相互作用来解释这一现象，同时将PAC‑Bayesian理论与实际实验相结合，提供了理论支撑。

**🔧 技术方法**

主要技术包括：对真实数据（如OpenR1‑Math、OpenMath、OpenScience等）和合成iGSM数据的两维实验设计；对不同难度区间的数据进行分层抽样；对比基准模型与不同难度训练模型的性能；以及使用PAC‑Bayesian界定一般化误差与外推误差的分解。

**📊 数据集**

使用的数据集包括：OpenR1‑Math‑94k、OpenMath、OpenScience、Math500、AIME24、Minerva Math 等公开数学/科学推理数据集，另外通过iGSM框架生成的合成数学推理数据用于控制实验。

**📈 对比分析**

实验通过将数据按CoT长度或运算次数划分为易、中、难三组，分别在不同数据规模下进行SFT，并测量在下游任务上的准确率提升。结果表明：在小规模训练时，去除难例会提升性能；在大规模训练时，去除易例更有利；最优难度随数据规模上升而变硬；这些结论在多种模型和任务上均得到验证。

**⚠️ 局限性**

局限性：论文未给出在实际大规模真实数据中快速估计最优难度的方法，且在复杂多样的真实数据上，难度与模型能力的匹配可能更难以自动化。

---

## 206. VIP-COP: Context Optimization for Tabular Foundation Models

**arXiv ID:** 2605.12904 | [PDF](https://arxiv.org/pdf/2605.12904v1)

**作者:** Yilong Chen `[一作]` (Carnegie Mellon University), Leman Akoglu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8566 | [OpenAlex ID](https://openalex.org/A5001634795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种面向Tabular Foundation Models的硬上下文优化方法，能够在不访问模型内部的情况下选取最重要的样本和特征。

**💡 创新点**

创新点在于结合KernelSHAP的在线回归与温度调节的重要采样，形成可任何时间、预算感知的黑盒VIP算法；同时实现了对数据增强和噪声鲁棒性的支持。

**🔧 技术方法**

核心技术包括KernelSHAP估计、温度引导的重要采样、多层级稀疏剪枝以及多温度并行运行。

**📊 数据集**

在TALENT基准的38个多分类数据集（包含小特征/大特征两组）上进行评估，并对TabPFN-v1的上下文容量进行测试。

**📈 对比分析**

与随机、聚类、决策树等启发式或优化基线相比，在原始、增强、噪声场景下平均提升约10–30%的平衡准确率，并在少量优化轮次内实现显著收益。

**⚠️ 局限性**

局限性包括对大规模数据仍受上下文窗口限制，算法仍需额外的前向推理开销，并且对极端分布偏移的鲁棒性仍待进一步验证。

---

## 207. Still Camouflage, Moving Illusion: View-Induced Trajectory Manipulation in Autonomous Driving

**arXiv ID:** 2605.12743 | [PDF](https://arxiv.org/pdf/2605.12743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 208. Ensuring Logic in the Fog: Sound POMDP Synthesis with LTL Objectives

**arXiv ID:** 2605.12581 | [PDF](https://arxiv.org/pdf/2605.12581v1)

**作者:** Can Zhou `[一作]` (Imperial College London), Pian Yu `[通讯]` (University College London)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5022512002)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种在部分可观测马尔可夫决策过程（POMDP）中对线性时序逻辑（LTL）任务进行安全、实时策略合成的新框架，核心是通过信念支持结构构造可证的奖励信号并将其嵌入改进的 POMCP 搜索中；

**💡 创新点**

创新点在于：①用信念支持 MDP（BSMDP）实现对“几乎必然满足”性质的可判定抽象；②设计了“可证赢支持”与“可证部分赢支持”的奖励赋值规则，确保奖励既保真又可被求值；③将奖励信号与 POMCP 结合，形成任意时刻可终止且保证下界的在线规划算法；

**🔧 技术方法**

使用的技术包括：LDBA 语法转换、BSMDP 与 AMEC 的映射、可证赢支持（BS-AMEC）检测、信念支持奖励赋值、改进的 POMCP（带终止判定与 UCB 选择）和 Monte Carlo 树搜索；

**📊 数据集**

实验数据集为经典 POMDP benchmark：Hallway (HW1/HW2)、Rock Sample (RS) 与 Grid World (Grid) 等，构造的 LTL 任务涵盖到达-保持、重复访问等多种目标；

**📈 对比分析**

与传统的乐观奖励 + PBVI、确定性有限状态控制器等方法对比，实验表明在 40k+ 状态的产品模型下，奖励构造耗时 < 0.5s，POMCP 合成耗时约 190s，能够在多项任务上获得更高或相近的满足概率，且在难解例子中显著优于对手；

**⚠️ 局限性**

局限性包括：只能保证“几乎必然满足”的下界，无法完全解决 LTL 的可判定性问题；构造奖励与产品 POMDP 的规模相关，极大模型仍会面临状态爆炸；当前方法仅适用于离散信念空间，尚未推广到无模型 RL 场景。

---

## 209. CommonWhy: A Dataset for Evaluating Entity-Based Causal Commonsense Reasoning in Large Language Models

**arXiv ID:** 2605.12918 | [PDF](https://arxiv.org/pdf/2605.12918v1)

**作者:** Armin Toroghi `[一作]` (University of Toronto), Scott Sanner `[通讯]` (University of Toronto)

**通讯引用:** 6656 | [OpenAlex ID](https://openalex.org/A5028174137)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了CommonWhy数据集，包含15,000条基于Wikidata实体的‘为什么’因果推理问题，并作为知识图谱问答（KGQA）基准；

**💡 创新点**

创新点在于：①首次针对实体的因果、可归纳推理设计多答案why问题；②将Commonsense推理与KG知识相结合，构建解释驱动的KGQA任务；③提供可复现的多样化推理难度分层；

**🔧 技术方法**

使用的技术主要有：大型语言模型（GPT‑5.1、Gemini‑2.5‑Flash、Llama‑3.3‑70B、OpenAI‑o3、DeepSeek‑V3.2）生成与验证Commonsense axiom；KGQA基线（KB‑Binder、KGR）结合LLM进行语义到SPARQL的翻译和事实一致性校验；评估工具包括BERTScore、ROUGE‑L、METEOR、FActScore及GPT‑4o逻辑等价性判定；

**📊 数据集**

数据集来源：从Wikidata抽取实体子图，结合人类标注与LLM生成的Commonsense axiom，形成why问题及多答案；对照已有CommonsenseQA、PIQA、StrategyQA、CREAK、CoLoTa、WebQuestions、LC‑Quad、WikiWhy等数据集进行对比；

**📈 对比分析**

实验对比显示：即便是最先进的推理模型OpenAI‑o3也仅达68%答案正确率；LLM生成的答案在长尾实体上性能更低；标准长文本指标（BERTScore、ROUGE、METEOR）与因果正确性不一致，提示需专门的逻辑判定；KGQA基线在此任务上表现极差，证明现有方法无法完成因果推理；

**⚠️ 局限性**

局限性包括：①依赖LLM生成Commonsense axiom，可能引入偏差；②仅覆盖Wikidata实体，语言与知识覆盖范围有限；③评估中使用GPT‑4o判定逻辑相似度，仍缺乏人类专家的全面验证；④数据集为英文，跨语言推广尚待研究；

---

## 210. 3D Primitives are a Spatial Language for VLMs

**arXiv ID:** 2605.12586 | [PDF](https://arxiv.org/pdf/2605.12586v1)

**作者:** Junze Liu `[一作]` (Unity Technologies), Tian Wang `[通讯]` (Unity Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过将几何原始体在可执行代码中的表达，提出了SpatialBabel基准、Code-CoT推理策略和S^3-FT自监督微调，用以诊断并提升视觉语言模型在空间推理中的表现。

**💡 创新点**

创新点包括：①跨六种场景代码语言的SpatialBabel基准揭示模型空间知识与代码格式解耦的瓶颈；②无训练的Code-CoT将空间推理迁移至中间代码生成；③完全自监督的S^3-FT利用自身生成的三维原始体代码作为伪标注，提升通用视觉推理。

**🔧 技术方法**

采用的技术包括可执行3D原始体代码生成（Three.js、Unity C#等）、程序化解析与评估、链式思维推理、LoRA微调与多任务训练。

**📊 数据集**

数据集涵盖1000个程序化原始体场景（T1–T5）、405个Hypersim真实照片场景以及CV-Bench-3D等公开基准。

**📈 对比分析**

实验表明，Code-CoT可在中等规模模型上提升6.4%空间问答准确率，S^3-FT在Qwen3-VL-8B上使SpatialBabel原始体问答从39.1%提升至47.1%，并在HallusionBench、CV-Bench-2D/3D等真实图像基准上分别提高17%、9.7%和3.3%。

**⚠️ 局限性**

局限性包括对原始体伪标注质量的依赖；在复杂、遮挡严重的真实场景下伪标注覆盖率低，导致S^3-FT与Code-CoT表现下降；以及跨语言迁移效果不完全稳定。

---

## 211. Structural Diversity Drives Disruptive Scientific Innovation

**arXiv ID:** 2605.12514 | [PDF](https://arxiv.org/pdf/2605.12514v1)

**作者:** Yichun Peng `[一作]` (Chinese Academy of Sciences), Hao Peng `[通讯]` (City University of Hong Kong)

**通讯引用:** 256691 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并量化了“结构多样性（SD）”指标，利用跨越1900–2025年260M篇科研论文的历史合作网络，评估SD对论文破坏性创新（CD指数）的预测能力，并通过因果推断（PSM）和准自然实验（NSF政策变更）检验其因果效应。

**💡 创新点**

创新点包括：①首次将团队在历史合作网络中的连通分量数作为结构多样性度量；②证明SD比传统的团队新鲜度和边密度更能预测破坏性创新；③揭示SD与团队规模的正交互作用，可缓解“规模诅咒”；④通过中介分析显示纪律整合（DI）解释了部分SD对创新的影响。

**🔧 技术方法**

技术方法：高维固定效应OLS回归、Propensity Score Matching、准自然实验（前后比较）、中介分析、结构化网络构建（连通分量计算）、CD指数的领域标准化与统计检验。

**📊 数据集**

数据集：OpenAlex（260,400,000篇论文，1900–2025）；AMiner（690,972篇计算机科学论文，含作者去重和NSF资助信息）；NSF政策实施前后（2010–2011 vs 2012–2013）论文子样本。

**📈 对比分析**

与团队新鲜度、边密度等指标对比：SD在模型R²上提升约16.7%，解释方差比例为0.619（远高于新鲜度0.212、边密度0.001）；PSM平均处理效应（ATT）在最高与最低SD四分位数间为0.12；准自然实验显示SD和CD均在政策后显著上升，证实因果关系。

**⚠️ 局限性**

局限性：①SD计算依赖合作网络窗口（2–7年）和作者去重质量；②因果推断受观测协变量限制，可能存在未观测混杂；③CD指数基于引用网络，可能忽略非文献影响；④实验样本主要为科学论文，结果对其他创作领域的推广尚需验证。

---

## 212. Revisiting DAgger in the Era of LLM-Agents

**arXiv ID:** 2605.12913 | [PDF](https://arxiv.org/pdf/2605.12913v1)

**作者:** Changhao Li `[一作]` (Georgia Institute of Technology), Bo Dai `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6489 | [OpenAlex ID](https://openalex.org/A5062711588)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对长时序语言模型代理使用 DAgger 算法进行后训练，结合教师与学生交替采样生成轨迹，并在轨迹上收集教师标签进行监督。

**💡 创新点**

创新点在于将传统 DAgger 的状态分布校正机制迁移到多轮 LM 代理，提出了基于轮次与全轨迹两种混合采样策略，既利用教师的稠密反馈，又保持学生在自身分布上的状态覆盖，解决了监督稀疏和协变量偏移的双重痛点。

**🔧 技术方法**

技术包括：教师-学生轮次混合采样（DAgger-style）、学生前缀+教师完成（AggreVaTe-style）、在每一步查询教师动作并记录，随后用交叉熵训练学生；使用 OpenHands 统一工具与环境接口；在 Qwen3-4B/8B 上微调，并使用 Qwen3-Coder-30B 作为教师。

**📊 数据集**

数据集主要为软件工程领域的真实任务：SWE-Gym（含可执行单元测试的仓库级问题）与 SWE-Bench Verified（500 个任务，最终 466 个可评估），训练集为 2338 条任务，测试集为 100 条保留样本及 SWE-Bench 验证集。

**📈 对比分析**

与 SFT、GRPO、On-policy Distillation 等基线以及公开的 SWE 代理系统对比，DAgger 训练的 4B 模型在 SWE-Bench Verified 上达 27.3% 解决率，已超过多数 8B 代理；8B 模型达 29.8%，逼近 32B 系统；在 SWE-Gym Holdout 上同样显著提升，说明方法在不同规模下均保持竞争力。

**⚠️ 局限性**

局限性：仍受限于长上下文容量导致的 context overflow；需要可访问教师模型的前提，无法直接应用于黑盒 LLM；在极端长时序或高难度任务中，混合采样比例调整仍需经验；训练成本较高，尤其在 8B 规模上需要大量轨迹。

---

## 213. Revisiting Reinforcement Learning with Verifiable Rewards from a Contrastive Perspective

**arXiv ID:** 2605.12969 | [PDF](https://arxiv.org/pdf/2605.12969v1)

**作者:** Feng Zhang `[一作]` (Beijing Institute of Technology), Guanjun Jiang `[通讯]` (Alibaba Group)

**通讯引用:** 436 | [OpenAlex ID](https://openalex.org/A5004378463)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的强化学习框架ConSPO，用于改进大型语言模型的推理能力。

**💡 创新点**

创新点在于将GRPO的分数对齐到生成时的对数似然，并使用组内InfoNCE对比学习与课程化阈值分离，解决了分数不匹配与信用分配不敏感的问题。

**🔧 技术方法**

采用了长度归一化对数似然评分、InfoNCE式对比目标、可调温度与学习率、以及基于余弦函数的学习进度调度的margin。

**📊 数据集**

使用了DeepScaleR-Preview-Dataset（约40k数学题）和DAPO-Math-17k，以及多种大型语言模型（DeepSeek、Qwen、Llama、Qwen3、Llama-3.2）进行训练。

**📈 对比分析**

与GRPO、DAPO、Dr.GRPO、DisCO、GMPO、CISPO、SAPO等主流RLVR方法对比，在七大数学推理基准（AIME、HMMT、MATH500、AMC、OlympiadBench）上实现平均提升约3–4分，部分基准提升超过10分。

**⚠️ 局限性**

局限性包括对可验证奖励的依赖、对超参数（温度、margin、warmup比例）的敏感性，以及在极低基准性能模型上仍需进一步提升。

---

## 214. HE-PIM: Demystifying Homomorphic Operations on a Real-world Processing-in-Memory System

**arXiv ID:** 2605.12841 | [PDF](https://arxiv.org/pdf/2605.12841v1)

**作者:** Harshita Gupta `[一作]` (ETH Zürich), Onur Mutlu `[通讯]` (ETH Zürich)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文在真实的通用PIM系统（UPMEM）上实现并全面评估了CKKS同态加密的所有核心操作与子程序（加、乘、旋转、基变换、引导），并在此基础上完成了端到端的加密神经网络推理。

**💡 创新点**

创新点在于：①首次在商用PIM平台上实现完整同态加密工作流；②系统性量化PIM在不同操作中的性能瓶颈；③通过对比CPU、GPU基线以及模拟的硬件改进方案（Barrett乘法器与理想/实际互连），提出面向HE的PIM硬件设计方向。

**🔧 技术方法**

采用了CKKS加密方案、基于RNS的多位整数运算、NTT/INTT变换、基转换与引导子程序；实现时利用UPMEM的DPU多线程SPMD模型、WRAM工作区与MRAM本地存储，并使用软件模拟64‑bit模块乘法；在评估时使用了MNIST图像（32×32×3）和更大尺寸（512×512）作为测试集。

**📊 数据集**

主要使用MNIST数据集（32×32×3）以及从32×32扩展到512×512的多尺寸图像，用于测量卷积、基变换与引导子程序的执行时间。

**📈 对比分析**

通过将PIM实现与AMD EPYC 7742 CPU、NVIDIA A100 GPU基线进行对比，结果显示：对元素级操作（加/乘）PIM优于GPU；对卷积、基变换与引导子程序GPU明显快（分别为9.3×、15.4×和152.8×）；但若在PIM上加入64‑bit Barrett乘法器与理想互连，性能可提升至相当于CPU数十倍、GPU数倍；实际互连方案（PIMnet）仍落后GPU。

**⚠️ 局限性**

限制主要体现在：①当前UPMEM DPU缺乏64‑bit模乘硬件，导致计算密集型操作极慢；②PIM核心间缺乏高效互连，导致基变换与引导等需要跨银行通信的子程序成为瓶颈；③大尺寸输入因Ciphertext容量不足而需拆分，进一步增加数据搬迁；④实验仅针对单卡GPU，未考虑多GPU或多PIM扩展的复杂性。

---

## 215. An Activity-Theoretical Approach to Teacher Professional Development in Pedagogical AI Agent Design

**arXiv ID:** 2605.12934 | [PDF](https://arxiv.org/pdf/2605.12934v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 216. Pyramid Self-contrastive Learning Framework for Test-time Ultrasound Image Denoising

**arXiv ID:** 2605.12567 | [PDF](https://arxiv.org/pdf/2605.12567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 217. Learning to Decide with AI Assistance under Human-Alignment

**arXiv ID:** 2605.12646 | [PDF](https://arxiv.org/pdf/2605.12646v1)

**作者:** Nina Corvelo Benz `[一作]` (Max Planck Institute for Biochemistry), Manuel Gomez-Rodriguez `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 8199 | [OpenAlex ID](https://openalex.org/A5042180520)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了 AI 模型在二元预测与决策中，置信度与决策者自身置信度的对齐如何影响学习到最优决策策略的复杂度，并给出了相应的在线算法。

**💡 创新点**

创新点在于证明了在完美对齐条件下，期望后悔的下界可以大幅降低，阈值结构的最优策略被解析出来，并基于此提出了一种利用阈值学习的在线算法；同时在理论上扩展了 DKW 不等式以获得更优的后悔上界。

**🔧 技术方法**

使用了上下文多臂老虎机框架、全反馈学习理论、DKW 不等式、阈值化决策策略与偏差-方差分析等技术。

**📊 数据集**

实验使用了两组真实人类实验数据集：Human‑Alignment（n=703，4 级置信度、13/5 AI 置信度水平）和 Human‑AI Interactions（15,063 预测，4 级置信度、10 AI 置信度水平）。

**📈 对比分析**

与传统的 Vanilla Contextual Online Learning 算法对比，假设对齐的算法在除“Census”任务外的所有组/任务中都能更快收敛、后悔率更低，验证了对齐假设在实际中仍能带来性能提升。

**⚠️ 局限性**

局限性包括：仅针对完整反馈、二元预测/决策的设定；对齐假设过强，未覆盖随时间变化的模型或部分反馈情形；对齐程度的量化和评估仍以最大/期望对齐误差为准，实际应用中可能更为复杂。

---

## 218. MindVLA-U1: VLA Beats VA with Unified Streaming Architecture for Autonomous Driving

**arXiv ID:** 2605.12624 | [PDF](https://arxiv.org/pdf/2605.12624v1)

**作者:** Yuzhou Huang `[一作]` (CUHK MMLab), Hongsheng Li `[通讯]` (CUHK MMLab)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种统一的流式 Vision‑Language‑Action（VLA）架构 MindVLA‑U1，能够在单一前向传播中同时生成自回归语言描述和连续轨迹，并通过流匹配扩散方法实现高精度动作规划。

**💡 创新点**

创新点包括：
1) 单共享 VLM 主干同时输出语言与连续动作，消除传统 VLA 中的动作离散化与语言无关的问题；
2) 流式记忆通道实现帧级别的时间上下文传递，避免多帧注意力冗余并保证跨帧轨迹连续性；
3) Intent‑CFG 桥接将语言意图作为条件向量直接引导动作扩散，首次实现可度量的语言‑动作可控性；
4) 支持稠密与稀疏 Mixture‑of‑Transformers（MoT）两种执行模式，实现慢速语义推理与快速控制的统一。

**🔧 技术方法**

使用的技术包括：
- VLM（如 Qwen3‑VL）作为主干；
- 端到端的联合 AR 语言损失与流匹配连续动作损失；
- 流式记忆模块（Q‑Former 风格）以及 FIFO 通道；
- Intent‑CFG（classifier‑free guidance）实现语言对动作的可控影响；
- 多模态自注意力与专家路由的 MoT 架构；
- 后训练的 RL（GRPO）进一步提升 RFS 分数。

**📊 数据集**

主要数据集为 Waymo Open Dataset End‑to‑End（WOD‑E2E），包含 4,021 个 20 秒行驶段；此外使用自建的 MindLabel 自动标注管道生成 VQA、意图、梦想轨迹等辅助标签。

**📈 对比分析**

在 WOD‑E2E 验证集上，MindVLA‑U1 在 RFS（Rater‑Feedback Score）上达到 7.83，超越人类驾驶员（8.13）仅靠 2 步扩散；在官方测试集上 RL 后得到 7.87 的 RFS，位列榜首。相较于传统 VA 系统（如 UniAD、RAP‑DINO）在相同参数规模下，MindVLA‑U1 既保持或提升规划精度，又在快/慢模式下实现与 VA 相当的吞吐量（≈16 FPS）。

**⚠️ 局限性**

局限性包括：
1) 仅在开放式评估（开环轨迹）下验证，未评估闭环控制性能；
2) 结果仅在 WOD‑E2E 上，缺乏跨 benchmark（nuScenes、NAVSIM 等）或车载部署验证；
3) 未给出完整的 VLA 扩展学习曲线，无法证明规模可持续提升；
4) 仅利用了 MindLabel 的基础 VQA 与 3 类意图，未充分利用其 20 类意图、梦想轨迹等丰富标签；
5) 通过 RL 后训练的奖励仅为单一 RFS，可能缺乏对安全性、平滑度等指标的全面覆盖。

---

## 219. Large Language Models for Agentic NetOps and AIOps: Architectures, Evaluation, and Safety

**arXiv ID:** 2605.12729 | [PDF](https://arxiv.org/pdf/2605.12729v1)

**作者:** Muhammad Bilal `[一作]` (Lancaster University), Schahram Dustdar `[通讯]` (TU Wien)

**通讯引用:** 37843 | [OpenAlex ID](https://openalex.org/A5004847496)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了将大型语言模型（LLM）与代理式网络运维（NetOps）和 IT 运维（AIOps）相结合的现有工作，系统地构建了一个基于自主性层级、工具范围、证据追踪和安全契约的框架，并提出了面向工作流的评估方法与安全治理建议。

**💡 创新点**

创新点在于：①把“自主性”拆解为可度量的“工具范围”和“不可绕过的门控”两大维度，形成了可操作的自治阶梯；②提出了“验证墙”与“预算/停止规则”等安全契约，明确了 LLM 代理在执行写操作前必须满足的可审计条件；③强调评估必须从工作流角度出发，关注证据质量、工具使用限度、回滚能力等多维度指标，而非单纯的问答准确率；④整合了安全、隐私和治理风险的威胁模型，为真实环境部署提供了具体对策。

**🔧 技术方法**

核心技术包括：语言模型与检索增强推理、工具增强代理（Tool‑augmented Agents）、类型化工具接口、预算与停止规则、验证门控、情景驱动的提示与 schema 约束、以及基于因果图的诊断后端。

**📊 数据集**

在数据集方面，文章主要引用了 NetOps / AIOps 常见的数据源——日志、指标、跟踪、配置存储、CMDB、ticketing 与 runbook 文档；并讨论了这些数据在评估中的可获取性与可信度，但并未引入具体公开数据集作为实验基准。

**📈 对比分析**

由于是综述性质，本文没有提供实验对比；但它提出了一套完整的评估指标与报告表格（如任务分类、指标词典、trace 评分公式），为后续工作提供了可复现的实验框架。若以此框架对比实验，预期能够在安全合规、回滚率、工具使用效率等多维度体现出优势。

**⚠️ 局限性**

局限性包括：①缺乏统一、公开的基准数据与接口规范，导致跨研究比较困难；②对 LLM 具体模型、参数、训练细节的讨论有限，实际效果取决于模型质量；③安全评估仍主要停留在理论与框架层面，缺少大规模生产级验证；④在面临数据漂移、工具版本升级等动态环境时，评估指标的稳定性尚未充分验证。

---

## 220. Ghost in the Context: Measuring Policy-Carriage Failures in Decision-Time Assembly

**arXiv ID:** 2605.12535 | [PDF](https://arxiv.org/pdf/2605.12535v1)

**作者:** Igor Santos-Grueiro `[一作]` `[通讯]` (International University of La Rioja), Igor Santos-Grueiro (International University of La Rioja)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究LLM代理在决策时上下文组装过程中导致策略失效的机制，并提出SafeContext中介层来部分硬化该过程。

**💡 创新点**

将决策时上下文组装视为可测量的控制路径，构建eviction/aliasing/绑定失稳三类失败范式，并通过准入、持久化、缓存重用与压力触发提醒等技术实现部分硬化。

**🔧 技术方法**

使用轻量级准入分类器、语义控制钉住（SCP）、缓存感知前缀重用、Invasive Context Engineering（ICE）提醒、Authority Binding Score（ABS）审计，以及基于判定器的ECR/CSR/DFR/DPR指标。

**📊 数据集**

构造基于模板的对话场景（约72实例/情景），在本地Llama 3.1 8B、Qwen 2.5 7B、Mistral 7B上进行实验，并在后续扩展到Qwen 14B、Llama 70B。

**📈 对比分析**

通过对齐的闭环实验矩阵（243个cell）对比未缓解与多种缓解变体，评估ECR、CSR、DFR、DPR；发现未缓解风险系统化，缓解在强压下提升ECR从≈0.03升至≈0.06，CSR也有提升，但绝对精确遵从仍低，成本增加约4–37 tokens。

**⚠️ 局限性**

局限性包括：仅在特定预算情形下验证，缺乏真实端到端代理轨迹；对大型模型的泛化不足；实验集成多模型但仍受制于服务栈差异；以及对持续记忆或检索机制的影响未充分探测。

---

## 221. CRePE: Curved Ray Expectation Positional Encoding for Unified-Camera-Controlled Video Generation

**arXiv ID:** 2605.12938 | [PDF](https://arxiv.org/pdf/2605.12938v1)

**作者:** Seonghyun Jin `[一作]` (KAIST), Jong Chul Ye `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的曲线射线期望位置编码CRePE，用于统一相机模型下的文本到视频生成。

**💡 创新点**

创新点是将每个token视为沿视线的深度分布，并在非针孔相机投影下积分RoPE相位，从而实现更精细的几何感知和更稳定的相机控制。

**🔧 技术方法**

采用视频Diffusion Transformer、几何注意力适配器、Monocular Geometry Foundation模型的伪径向距离监督以及Radial MixForcing技术。

**📊 数据集**

使用PanShot多摄像机视频数据集进行训练与评估。

**📈 对比分析**

与ReCamMaster、UCPE等基线相比，CRePE在相机控制、镜头畸变、姿态精度等指标上均有提升，平均排名优于RayRoPE端点式编码，并保持了竞争力的视频质量指标。

**⚠️ 局限性**

局限在于仅针对中心相机模型（UCM）设计，外部径向地图的定量评估不足，对非中心或滚动快门相机的扩展仍待研究。

---

## 222. Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer

**arXiv ID:** 2605.12798 | [PDF](https://arxiv.org/pdf/2605.12798v1)

**作者:** Baris Askin `[一作]` (Carnegie Mellon University), Carlee Joe-Wong `[通讯]` (Carnegie Mellon University)

**通讯引用:** 17464 | [OpenAlex ID](https://openalex.org/A5003037377)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了大型语言模型在狭窄有害数据上微调时出现的出现性失调（EM）和潜伏学习（SL）现象，探讨了数据结构、任务难度、预训练分布与训练通道如何共同决定失调的传播。

**💡 创新点**

创新点在于将EM和SL视为数据介导的迁移问题，利用结构化自然语言和合成数据网格化任务-领域轴，首次将潜伏转移实验扩展到离线（OPTD）与在线（OPD）蒸馏，并证实教师方向决定失调方向，数据分布则调节失调强度。

**🔧 技术方法**

主要技术包括监督微调（SFT）、离线全词典蒸馏（OPTD）和在线基于反KL的蒸馏（OPD），以及对齐与连贯性评估器（0–100分量表），并使用LoRA适配器对7B–14B模型进行实验。

**📊 数据集**

使用了三套数据集：① 结构化自然语言数据集（12个域×4任务），② 具备可控域-任务网格的合成数据（World 1/World 2），③ 常用的MATH评估集和公开的评估集（240个跨域提示），并将训练数据分为对齐/非对齐两类。

**📈 对比分析**

方法对比通过EM率、对齐/连贯性得分、任务/领域迁移矩阵等指标实现；实验表明任务相似性对EM传播更敏感；在SFT/OPTD/OPD三种通道中，OPTD与OPD均可诱发潜伏失调且效果不低于SFT，且对齐教师可在任何数据源下几乎完全消除失调。

**⚠️ 局限性**

局限性包括：实验仅涵盖少数模型规模（7B–14B），合成数据对真实语言生态的适用性有限；评估指标主要基于人工打分，缺乏大规模自动化评估；未深入分析失调的机制细节（如具体激活路径）。

---

## 223. Revealing the Gap in Human and VLM Scene Perception through Counterfactual Semantic Saliency

**arXiv ID:** 2605.13047 | [PDF](https://arxiv.org/pdf/2605.13047v1)

**作者:** Ziqi Wen `[一作]` (University of California, Santa Barbara), Miguel P. Eckstein `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 7816 | [OpenAlex ID](https://openalex.org/A5084385239)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并验证了一种黑盒因果方法 Counterfactual Semantic Saliency（CSS），通过对场景中对象进行消融并测量生成描述的语义变化来量化对象重要性。

**💡 创新点**

创新点在于将因果消融与文本生成语义偏移相结合，提供可跨模型、可解释的对象级重要性评估，并首次系统量化 VLM 与人类在高层语义场景理解上的对齐差距。

**🔧 技术方法**

利用 GPT‑4o 生成对象列表，SAM3 进行分割，Nano Banana 2 进行无痕消融，再用 Jasper 嵌入和余弦相似度计算 CSS，配合 19 只 VLM 的文本生成。

**📊 数据集**

数据集基于 OSIE 700 张多物体自然场景，筛选 307 张事实图像及 1,306 张对照消融图像，并通过 227 名受试者收集 16,289 条人类描述。

**📈 对比分析**

通过 Top‑1 正确率和 Kendall τ 评估模型与人类在关键对象识别与重要性排序上的一致性，发现人类一致性为 73%/0.58，而 VLM 最高仅 65%/0.51，表明存在显著的感知偏差，且尺寸偏差是主要驱动因素。

**⚠️ 局限性**

局限包括依赖生成式消融质量、缺乏对闭源模型内部机制的深度探究、可能的超推理导致误差以及仅聚焦尺寸/中心等低层特征，未来需研究更丰富的语义因果关系与推理偏差。

---

## 224. DisaBench: A Participatory Evaluation Framework for Disability Harms in Language Models

**arXiv ID:** 2605.12702 | [PDF](https://arxiv.org/pdf/2605.12702v1)

**作者:** Eugenia Kim `[一作]` (Microsoft), Christina Mallon `[通讯]` (Microsoft)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5042577335)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了DisaBench框架，包括12类残障相关危害的协同制定分类法、基于benign与adversarial提示的评估方法以及对应数据集。

**💡 创新点**

创新点在于将残障社区参与与红队专业人员共同协作，设计针对残障用户真实使用场景的评估流程，并揭示现有安全基准对残障危害的盲点。

**🔧 技术方法**

采用参与式红队方法、人工标注（4名具有残障经验的评审）、基于社会模型的危害定义与层级评估。

**📊 数据集**

使用175条精心设计的提示（94条攻击性，81条友好性）及其在3种指令微调模型（Llama 4 Maverick、Grok‑3、Phi‑4）产生的525个提示‑响应对。

**📈 对比分析**

与传统毒性检测模型对比，DisaBench发现10–32%的回答被标为有害，显示现有自动化工具只能捕捉部分危害，尤其是细腻的同情、歧视和数字负担。

**⚠️ 局限性**

局限包括评审者数量有限、仅涵盖英语、部分危害类别覆盖不足、人工标注耗时且无法完全代表残障群体多样性。

---

## 225. DiM\textsuperscript{3}: Bridging Multilingual and Multimodal Models via Direction- and Magnitude-Aware Merging

**arXiv ID:** 2605.12960 | [PDF](https://arxiv.org/pdf/2605.12960v1)

**作者:** Zijing Wang `[一作]` (Northeastern University), Hinrich Schütze `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本论文提出一种训练无关的参数合并方法DiM3，用于在多模态模型基础上注入多语言能力，保持视觉模块不变；

**💡 创新点**

创新点在于将多语言与多模态残差更新视为异质向量，采用方向和幅度双重信息的列级别选择性合并，显著提升跨语言语义对齐而不破坏视觉理解；

**🔧 技术方法**

采用基于残差的方向-幅度分解、列级别的排名归一化与softmax融合、以及对称平均的权重融合等技术；

**📊 数据集**

使用涵盖57种语言的多模态与文本任务数据集，包括XCOPA、XStoryCloze、XNLI、xMMMU、MaXM、MaRVL、xGQA、CVQA、Afri-MCQA、MMStar、MMMU、SEED-Bench-2-Plus等；

**📈 对比分析**

与多种现有合并基线（Task Arithmetic、DARE、TIES-Merging等）以及原始多模态模型和专门微调的多语言多模态模型对比，DiM3在文本与多模态评测中均实现平均多语言得分最高，且保持或提升了通用多模态能力；

**⚠️ 局限性**

局限性包括对列级别合并的计算开销、在极低资源语言上的效果仍有限，以及未针对多模态推理中更细粒度的交互机制进行深入探究。

---

## 226. A Data Efficiency Study of Synthetic Fog for Object Detection Using the Clear2Fog Pipeline

**arXiv ID:** 2605.12608 | [PDF](https://arxiv.org/pdf/2605.12608v1)

**作者:** Mohamed Ahmed Mohamed `[一作]` (University of Liverpool), Xiaowei Huang `[通讯]` (University of Liverpool)

**通讯引用:** 10743 | [OpenAlex ID](https://openalex.org/A5015499043)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本论文提出了 Clear2Fog (C2F) 端到端物理模拟管线，可在摄像头与 LiDAR 数据集上同步生成真实感雾化图像，并利用该数据集训练物体检测模型。

**💡 创新点**

创新点包括：① 采用单目深度估计保证远景与天空的语义完整性；② 通过光照亮度裁剪得到无色偏的大气光估计；③ 引入多密度雾化多样化训练，证明环境多样性比单一雾密度或更大数据量更能提升模型鲁棒性；④ 在 fine‑tune 时发现十倍学习率可克服仿真偏差，实现比仅用真实雾图更高的 mAP。

**🔧 技术方法**

使用的技术包括：Koschmieder 物理散射模型、Monocular Depth Pro 深度估计、光照亮度裁剪（luminance‑clipping）、LiDAR 雾化的 Hahner 等物理模型、Faster R‑CNN 与 YOLOX‑S 目标检测框架，以及针对学习率的敏感性实验。

**📊 数据集**

主要数据集为 Waymo Open Dataset（清晰天气）生成 270k 张雾化图像；对比基准为 Multifog KITTI；真实雾图用于 sim‑to‑real 验证的 Seeing Through Fog (STF) 数据集；额外使用 COCO 及 Flickr30k 进行跨域可泛化测试。

**📈 对比分析**

比较方法：在相同检测模型（Faster R‑CNN、YOLOX‑S）上分别训练清晰、固定密度雾、混合密度雾；使用 mAP 评估并做标准差统计；人类感知实验中 C2F 以 92.95% 的偏好度领先；Fine‑tune 时 10 倍学习率使 mAP 提升至 1.67 点，超过仅用真实雾训练的基线。

**⚠️ 局限性**

限制包括：① 真实雾实验仅基于 1,140 张 STF 图像，难以覆盖更广泛的气候与光照；② 单目深度估计在透明或高频细节场景易产生误差，导致雾分布不一致；③ 大规模雾化生成计算量大，主要适用于离线增强，实时训练尚不可行。

---

## 227. SoK: A Comprehensive Analysis of the Current Status of Neural Tangent Generalization Attacks with Research Directions

**arXiv ID:** 2605.12792 | [PDF](https://arxiv.org/pdf/2605.12792v1)

**作者:** Thushari Hapuarachchi `[一作]` (University of South Florida), Kaiqi Xiong `[通讯]` (University of South Florida)

**通讯引用:** 3702 | [OpenAlex ID](https://openalex.org/A5101689516)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对 Neural Tangent Generalization Attack (NTGA) 进行系统评估和分类，比较其与其他清标记泛化攻击的性能与脆弱性，并探讨其在不同防御和数据处理场景下的表现。

**💡 创新点**

首次对 NTGA 进行全面综述与实验验证，构建黑盒与数据防护攻击的分类体系，揭示 NTGA 对抗对抗训练、数据增强及线性可分性攻击的弱点，并提供改进方向。

**🔧 技术方法**

利用 Neural Tangent Kernels (NTK)、高斯过程近似、投影梯度上升、对抗训练、数据增强和线性可分性分析等技术。

**📊 数据集**

主要使用 CIFAR‑10 数据集（以及在讨论中提到的 ImageNet、MNIST 等基准集），对 ResNet、VGG 等模型进行实验。

**📈 对比分析**

通过在相同数据集上训练模型并记录测试准确率，比较 NTGA 与 DeepConfuse、Error‑Minimizing、Error‑Maximizing、Synthetic、REM、One‑Pixel Shortcut、Autoregressive 等攻击；发现 NTGA 在某些情况下表现优于部分攻击，但在对抗训练或图像变换下性能显著下降。

**⚠️ 局限性**

局限性包括：对抗训练和数据增强能显著削弱其保护效果；线性可分性使得攻击易被检测；生成扰动耗时长；并非所有场景下（如分布式学习、无监督学习、集成学习、非图像数据）都能保持有效性。

---

## 228. The Distributed Complexity Landscape on Trees Depends on the Knowledge About the Network Size

**arXiv ID:** 2605.12787 | [PDF](https://arxiv.org/pdf/2605.12787v1)

**作者:** Alkida Balliu `[一作]` (Gran Sasso Science Institute), Gustav Schmid `[通讯]` (University of Freiburg)

**通讯引用:** 104 | [OpenAlex ID](https://openalex.org/A5019332485)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在LOCAL模型中节点对网络规模n缺乏完整知识（或仅知道多项式上界）时，最常见的LCL问题——k‑hierarchical 2½‑coloring和k‑rake‑and‑compress——的分布式复杂度，并证明了与传统假设下复杂度图谱显著不同。

**💡 创新点**

创新点包括：①揭示了在不知n或仅给出多项式上界时，LCL问题的复杂度区间会变得更细化并出现非直观指数；②提出并求解了一个优化问题，用以决定rake‑compress算法中各阶段的rake次数，从而得到与n上界相关的精确上界；③证明了对Hardy‑Littlewood和Knuth两种Ω定义的差异在此场景下是不可忽视的；④在随机化设置下展示了随机性可显著降低复杂度，并给出了f(f(n))=log n型特殊函数的上界。

**🔧 技术方法**

主要技术手段包括：构造k层次下界树、使用递归rake‑compress分层算法、对上界算法参数化并转化为优化问题、利用多项式上界信息和ID空间约束进行细粒度分析、采用随机标记策略实现对log n尺度的粗略估计，以及解析与函数f(f(n))=log n相关的隐式函数。

**📊 数据集**

本研究完全基于理论构造，没有使用实验数据集；实验部分只涉及对上述理论构造的树图进行推理和证明。

**📈 对比分析**

与传统LOCAL模型已知的复杂度区间（O(1), Θ(log* n), Θ(log n), Θ(n^{1/k})）进行对比，证明在新的假设下这些区间会被细分，并给出匹配的上界与下界；例如在k=3, c=3时得到O(n^{0.566})，而随机化下则得到Θ(n/log n)或Θ(n/f(n))，显著低于原先的Θ(n)复杂度。

**⚠️ 局限性**

局限性：结果仅适用于树图和LCL类问题，对一般图或非树图的影响尚未明确；随机化分析目前仅完成k=2,3的情况，k>3仍是开放问题；此外，证明依赖Hardy‑Littlewood定义的Ω，若改为Knuth定义则部分下界无法得到。

---

## 229. 3D RL-DWA: A Hybrid Reinforcement Learning and Dynamic Window Approach for Goal-Directed Local Navigation in Multi-DoF Robots

**arXiv ID:** 2605.12689 | [PDF](https://arxiv.org/pdf/2605.12689v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 230. Learning When to Act: Communication-Efficient Reinforcement Learning via Run-Time Assurance

**arXiv ID:** 2605.12561 | [PDF](https://arxiv.org/pdf/2605.12561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 231. State-Space NTK Collapse Near Bifurcations

**arXiv ID:** 2605.12763 | [PDF](https://arxiv.org/pdf/2605.12763v1)

**作者:** James Hazelden `[一作]` (University of Washington), Eric Shea-Brown `[通讯]` (University of Washington)

**通讯引用:** 4791 | [OpenAlex ID](https://openalex.org/A5046602924)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了梯度下降在动力学模型中跨越分岔点时的学习动态，并通过经验状态空间NTK把学习过程降到低维的秩一通道；

**💡 创新点**

创新点在于将分岔理论与神经网络梯度学习结合，揭示分岔点导致NTK降秩并主导学习几何；同时提出低秩自然梯度可稳定训练；

**🔧 技术方法**

使用经验状态空间NTK、正常形式（normal form）分析、低秩自然梯度优化；

**📊 数据集**

在学生‑教师循环神经网络（RNN）上进行实验，任务为重现教师的二维固定点动力学；

**📈 对比分析**

与普通SGD相比，低秩自然梯度在保持近似相同计算开销的前提下，显著平滑了损失曲线并抑制了分岔处的不稳定性；

**⚠️ 局限性**

局限性在于理论仅局限于局部分岔分析，缺乏全局预测和自动分岔检测机制。

---

## 232. Constraint-Aware Flow Matching: Decision Aligned End-to-End Training for Constrained Sampling

**arXiv ID:** 2605.12754 | [PDF](https://arxiv.org/pdf/2605.12754v1)

**作者:** Jacob K. Christopher `[一作]` (University of Virginia), Ferdinando Fioretto `[通讯]` (University of Virginia)

**通讯引用:** 1355 | [OpenAlex ID](https://openalex.org/A5052534316)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Constraint-Aware Flow Matching（CAFM）框架，直接在训练目标中加入可微分约束投影，使生成过程与约束满足需求完全对齐。

**💡 创新点**

创新点在于将约束投影嵌入流匹配训练，利用可微分的 Sequential Quadratic Programming (SQP) 层实现端到端的训练-推理一致，解决了传统训练-采样不匹配导致的质量下降问题。

**🔧 技术方法**

使用技术包括：流匹配（flow matching）模型、可微分优化层（SQP 投影）、决策聚焦学习（decision‑focused learning）思想、以及对齐的训练目标。

**📊 数据集**

使用的数据集涵盖三大真实任务：① PDE 约束物理系统（Navier–Stokes、Reaction–Diffusion、Burgers 等）② 微风速度场估计（稀疏合成风速数据）③ 微结构逆设计（Bentheimer sandstone 图像）。

**📈 对比分析**

方法与无约束基线 Functional Flow Matching (FFM)、基于投影的 PCFM、ECI、PDM 等进行对比。CAFM 在所有任务的约束满足率、MMSE、SMSE、CV 等指标上均优于或接近最优，尤其在未见的约束集上保持高性能。

**⚠️ 局限性**

局限性：训练时需额外的投影计算，增加计算开销；对非凸约束求解可能陷入局部最优；需预训练再微调以降低训练成本。

---

## 233. Do Fair Models Reason Fairly? Counterfactual Explanation Consistency for Procedural Fairness in Credit Decisions

**arXiv ID:** 2605.12701 | [PDF](https://arxiv.org/pdf/2605.12701v1)

**作者:** Gideon Popoola `[一作]` (Montana State University), John Sheppard `[通讯]` (Montana State University)

**通讯引用:** 2740 | [OpenAlex ID](https://openalex.org/A5072522101)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种检测与缓解隐性程序性偏差的框架——Counterfactual Explanation Consistency (CEC)，旨在保证模型在不同人群上使用相同的推理过程而非仅仅得到相同的预测结果。

**💡 创新点**

创新点在于：①构建四格公平性分类法，揭示“同预测不同推理”(Regime B) 的盲区；②采用最近邻标签一致的反事实生成方法，避免因受限属性引发的无效匹配；③提出一致基准原则，消除基准差异对IG解释的混淆；④将解释一致性作为可微分训练损失，形成三目标（准确率、结果公平、程序公平）的联合优化。

**🔧 技术方法**

主要技术包括：近似反事实生成（label‑group KD‑tree 匹配）、一致基准下的集成梯度解释、归一化向量距离度量的 CEC 分数、以及多目标损失优化（交叉熵 + Equalized Odds + CEC）。

**📊 数据集**

使用的数据集有四类：合成数据、German Credit、Adult Income、以及真实的 HMDA 贷款披露数据，覆盖从受控实验到工业规模的四种情况。

**📈 对比分析**

与六种主流公平方法（无约束、DIR、Hardt、Agarwal、Lagrangian、Adversarial）对比，CEC 在所有数据集上都显著降低 Regime B 率、CEC 分数，并保持或提升 Equalized Odds 与统计平等度，同时对预测 F1/AUC 的损失极小；在 Pareto 非支配分析中，CEC 成为唯一在 20/20 情形下的非支配方案。

**⚠️ 局限性**

局限性包括：①依赖集成梯度解释，其他解释方法需进一步验证；②金融特征集合需人工指定，可能带来主观性；③仅处理二元受保护属性，扩展至多元或交叉属性尚未研究；④计算开销较大（IG 约 3–5 倍训练时间），对大规模模型适用性有限。

---

## 234. Persona-Model Collapse in Emergent Misalignment

**arXiv ID:** 2605.12850 | [PDF](https://arxiv.org/pdf/2605.12850v1)

**作者:** Davi Bastos Costa `[一作]` (University of Sāo Paulo), Renato Vicente `[通讯]` (University of Sāo Paulo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大型语言模型上针对不安全代码进行细调所导致的“新兴偏差”，并提出通过人格模型崩塌（persona‑model collapse）来解释这一现象。

**💡 创新点**

创新点在于引入了两个行为诊断指标：道德敏感性（Moral Susceptibility，S）和道德鲁棒性（Moral Robustness，R），用来量化模型在角色扮演下的跨人格差异与同一人格内部一致性，从而首次将其应用于新兴偏差的实验验证。

**🔧 技术方法**

技术上结合了 Moral Foundations Questionnaire (MFQ‑30) 与系统化的角色扮演（100 个多样化人格，每个问题重复 10 次），并通过方差与标准差计算得到 S 与 R，同时对每个道德基础进行拆分分析。

**📊 数据集**

使用的主要数据集包括：① 用于不安全代码细调的专用代码数据集；② 对照用的安全代码数据集；③ 通过 MFQ‑30 生成的道德评价问卷数据。

**📈 对比分析**

对比方法为将四个前沿模型（DeepSeek‑V3.1、GPT‑4.1、GPT‑4o、Qwen3‑235B）在基线、细调为不安全代码、以及对照安全细调三种条件下计算 S、R 并进行相对变化分析。结果显示不安全细调使 S 上升约 55%，R 降低约 65%，所有不安全版本的 S 均超过已有 13 个前沿基线模型的 0.66–0.83 范围，说明模型在角色区分与内部一致性方面显著退化。

**⚠️ 局限性**

局限性包括：仅评估四种模型且仅用单一不安全代码细调数据；角色人格样本有限（100 个），可能不足以覆盖全部人格维度；指标依赖于 MFQ‑30 这一人为心理测量工具，可能存在测度可靠性与适用性问题；未对机制层面进行直接验证，只提供行为层面的证据。

---

## 235. Profit Maximization in Bilateral Trade against a Smooth Adversary

**arXiv ID:** 2605.12664 | [PDF](https://arxiv.org/pdf/2605.12664v1)

**作者:** Simone Di Gregorio `[一作]` (Sapienza University of Rome), Chris Schwiegelshohn `[通讯]` (Aarhus University)

**通讯引用:** 700 | [OpenAlex ID](https://openalex.org/A5080748807)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了在σ平滑对手环境下双边贸易的利润最大化问题，提出了一种在线学习算法并证明了 Õ(√T) 的次线性遗憾上界；

**💡 创新点**

首次将算法链技术与 L¹‑网构造相结合，在非参数机制空间中实现快速学习，并给出从联合广告问题的正式归约；

**🔧 技术方法**

主要技术包括：平滑对手假设、网格机制的 L¹‑逼近、算法链（hierarchical Hedge）以及概率链思想的结合；

**📊 数据集**

该工作为理论分析，无需实际数据集，所有结果均基于假设分布与对手模型；

**📈 对比分析**

与先前 i.i.d. 情况下已知的最优率相比，保持了相同的 √T 速率；与传统的专家学习方法相比，利用结构化网路显著降低了遗憾；

**⚠️ 局限性**

局限性在于：算法的计算复杂度为 exp(√T)（非多项式时间），且对平滑度参数 σ 的依赖尚未最优，未来需改进计算效率与 σ 依赖。

---

## 236. All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs

**arXiv ID:** 2605.12671 | [PDF](https://arxiv.org/pdf/2605.12671v1)

**作者:** Xi Chen `[一作]` (University Of Toronto), Gerald Penn `[通讯]` (University Of Toronto)

**通讯引用:** 6803 | [OpenAlex ID](https://openalex.org/A5052428595)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实证证明大型语言模型（LLM）中的功能并非局限于单一稀疏子图，而是可以由多个结构上高度不同的电路或层束（sheaf）共同实现同一任务；提出重叠惩罚的层束排斥方法（OASR）以系统挖掘这些低重叠的多重机制，并给出理论解释——分布式稠密电路假设（Distributive Dense Circuit Hypothesis）。

**💡 创新点**

1) 质疑并证明了长期隐含的功能各向异性假设（Functional Anisotropy Hypothesis）不成立；2) 提出 OASR 方法在保持高性能、稀疏与完整性的同时，显著降低不同层束间的结构重叠；3) 发现超稀疏的三边层束，证明即使最小化结构也不存在必不可少的边；4) 从理论上说明在高维线性叠加下，同一任务可由多种低重叠子图实现。

**🔧 技术方法**

基于计算图的残差流表示，使用梯度优化的电路/层束发现框架（DiscoGP、ACDC、EAP、EP），并在此基础上加入 OASR 的重叠惩罚项；使用 Gumbel–Sigmoid 软化、梯度约束实现离散子图搜索；理论分析采用线性子集求和（subset‑sum）与误差容忍（margin）分析。

**📊 数据集**

使用多种 LLM 任务作为基准，包括 IOI、BLiMP、AGA、ANA、DNA、Docstring 等；模型主要采用 GPT‑2（12 层/12 头），也在 Pythia 等模型上做扩展；通过标准任务准确率、完整性准确率、边密度以及 IoU（重叠率）等指标进行评估。

**📈 对比分析**

与传统的单一电路/层束发现方法相比，OASR 在保留 100%（或 ≥95%）任务准确率的同时，将不同发现结果间的 IoU 降至 0.1%–5% 左右；在多种方法（DiscoGP、ACDC、EAP、EP）上均观察到相似的低重叠现象；此外，OASR 能发现三边层束，任务准确率保持 86.7%，证明多样性与性能兼得。

**⚠️ 局限性**

1) 研究主要集中在 GPT‑2 等公开模型，未覆盖更大规模或其他架构（如 GPT‑3、PaLM 等）；2) OASR 的重叠惩罚系数等超参数需手工调优，缺乏自动化选择；3) 理论证明基于局部线性和阈值假设，实际模型可能存在更复杂的非线性效应；4) 虽展示多重机制存在，但如何在实际模型调优或安全性分析中利用这些信息仍待进一步研究。

---

## 237. CoGE: Sim-to-Real Online Geometric Estimation for Monocular Colonoscopy

**arXiv ID:** 2605.13038 | [PDF](https://arxiv.org/pdf/2605.13038v1)

**作者:** Liangjing Shao `[一作]` (Chinese University of Hong Kong), Hongliang Ren `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 17689 | [OpenAlex ID](https://openalex.org/A5032340829)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了CoGE框架，实现了仅使用模拟数据即可在线完成单目结肠镜的深度估计与3D重建；

**💡 创新点**

创新点包括：①基于小波变换的结构感知感知块（SAP）提取结肠通用结构特征；②利用Retinex理论的光照感知监督模块（IAS）动态调节置信度；③引入记忆遗忘机制过滤冗余记忆；④实现了模拟到真实的无缝迁移，单纯训练模拟数据即可在真实场景取得SOTA性能；

**🔧 技术方法**

技术实现涉及ViT编码器、离散小波变换（DWT/IDWT）+窗口多头自注意力、卷积高频细节提取、记忆注意力模块、光照感知监督、DPT头输出点云/相机姿态/置信度，以及CUT3R风格的几何损失；

**📊 数据集**

使用数据集包括：VR‑Caps模拟数据（SimCol3D，9个场景 9009帧），真实数据集C3VD、C3VDv2（共11个序列，3000+帧），以及少量内部结肠镜图像；

**📈 对比分析**

与AF‑SfM、DepthAnything、EndoDAC、ColonAdapter、MonoPCC、Spann3R、MonST3R、CUT3R、STream3R等SOTA方法在SimCol3D、C3VD、C3VDv2上进行定量对比。CoGE在Abs Rel、Sq Rel、RMSE、RMSE_log、δ1.25等指标均处于最优或接近最优（SimCol3D Abs Rel 0.027，δ1.25 0.995；C3VD Abs Rel 0.083，δ1.25 0.913），同时保持可接受的实时帧率；

**⚠️ 局限性**

局限性：额外的SAP块导致推理速度略低；对剧烈运动模糊、急剧动态变化以及特定病变组织的处理仍存在性能下降；未来需进一步优化网络结构和鲁棒性。

---

## 238. DirectTryOn: One-Step Virtual Try-On via Straightened Conditional Transport

**arXiv ID:** 2605.12939 | [PDF](https://arxiv.org/pdf/2605.12939v1)

**作者:** Xianbing Sun `[一作]` (Shanghai Jiao Tong University), Jianfu Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 4616 | [OpenAlex ID](https://openalex.org/A5100395008)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究并实现了一步式虚拟试衣（DirectTryOn），通过纯条件传输、服装保持损失和自一致性损失对条件传输路径进行拉直，并使用一阶蒸馏实现高效生成；

**💡 创新点**

提出将虚拟试衣视为条件传输路径更直的任务，采用纯条件传输消除无条件干扰，引入服装保持损失直接监督细节，加入自一致性损失迫使不同步长的速度预测一致，从而实现一阶生成；

**🔧 技术方法**

使用流匹配/扩散框架、MMDiT 风格的网络架构、纯条件传输、服装保持损失、自一致性损失以及Latent Adversarial Diffusion Distillation (LADD) 进行一阶蒸馏；

**📊 数据集**

在 VITON-HD（上半身服装）和 DressCode（上、下身、连衣裙）数据集上进行训练与评估；

**📈 对比分析**

与 OOTDiffusion、CatVTON、Leffa、Any2anyTryon 等基线在 unpaired 设置下使用 FID/KID 评估，DirectTryOn 在 VITON‑HD 上 FID 8.59/KID 0.56，DressCode 上 FID 5.08/KID 0.95，推理时间仅 0.48 s，速度提升约 19×，且性能保持最优；

**⚠️ 局限性**

仍需较长的 fine‑tune 与蒸馏，且在复杂多样的服装场景（如全身、动态服装）或缺少高质量标注的数据上可能效果受限，当前实验多集中于上半身服装。

---

## 239. Career Mobility of Planning Alumni in the United States: Evidence from Professional Profile Data using Large Language Models

**arXiv ID:** 2605.12618 | [PDF](https://arxiv.org/pdf/2605.12618v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 240. CROP: Expert-Aligned Image Cropping via Compositional Reasoning and Optimizing Preference

**arXiv ID:** 2605.12545 | [PDF](https://arxiv.org/pdf/2605.12545v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 241. In-Situ Behavioral Evaluation for LLM Fairness, Not Standardized-Test Scores

**arXiv ID:** 2605.12530 | [PDF](https://arxiv.org/pdf/2605.12530v1)

**作者:** Zeyu Tang `[一作]` (Stanford University), Sanmi Koyejo `[通讯]` (Stanford University)

**通讯引用:** 7473 | [OpenAlex ID](https://openalex.org/A5076316802)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 MAC‑Fairness 多代理对话框架，在多轮对话中嵌入身份、角色和显露等可控因素，直接观察 LLM 在身份变化下的行为。

**💡 创新点**

创新点在于：①提出基于情境对话的公平性评估方法，取代传统静态问答得分；②揭示标准化测试因提示格式差异导致结构不稳定、模型排名颠倒；③发现跨基准一致的模型行为签名。

**🔧 技术方法**

使用技术包括：多代理 LLM 对话、身份与显露参数配置、shift‑rate 变动率度量、ANOVA 与 t 检验等统计方法。

**📊 数据集**

实验数据集涵盖 BBQ、Discrim‑Eval、Difference‑Awareness 三个基准，总计约 800 万条对话语料。

**📈 对比分析**

通过对比不同提示格式和身份设定对标准化得分及模型排名的影响，发现标准化测试结果波动大、排名不稳；而在 MAC‑Fairness 下的行为指标在不同基准中保持稳定，展示了更可靠的公平性评估。

**⚠️ 局限性**

局限性包括：仅在同一模型内部进行多代理实验，未考察不同模型间差异；实验使用合成对话，可能无法完全代表真实人类交互；对更复杂身份属性和更大规模对话的推广尚待验证。

---

## 242. Debunking Grad-ECLIP: A Comprehensive Study on Its Incorrectness and Fundamental Principles for Model Interpretation

**arXiv ID:** 2605.12952 | [PDF](https://arxiv.org/pdf/2605.12952v1)

**作者:** Yongjin Cui `[一作]` (Zhejiang University), Xiaohui Fan `[通讯]` (Zhejiang University)

**通讯引用:** 16409 | [OpenAlex ID](https://openalex.org/A5060364374)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对ICML 2024提出的Grad-ECLIP方法进行系统性评估与反证，提出等价的Attention-ECLIP，并阐明两种技术路线本质相同；同时提出模型解释的两个基本原则：忠实性和与模型性能的一致性。

**💡 创新点**

①证明Grad-ECLIP的中间特征路和Attention-ECLIP的注意力路在单头化后完全等价；②揭示Grad-ECLIP通过改动模型结构并使用修改后模型解释来误导解释结果；③提出忠实与一致性的原则，并通过实验验证其必要性。

**🔧 技术方法**

对CLIP模型的Transformer层进行梯度与注意力分析，采用注意力归约、梯度加权以及注意力重构（softmax + Min‑Max归一化）等技术；实现Attention-ECLIP与Grad-ECLIP的等价推导与代码实现。

**📊 数据集**

使用ImageNet（1,000样本）评估图像解释的插入/删除指标，使用MS COCO（1,000样本）评估文本解释的删除实验，二者均以CLIP模型为基础。

**📈 对比分析**

通过定性可视化和定量插入/删除/删除实验与Grad-ECLIP比较，Attention-ECLIP在相同的解释效果下计算更简洁、效率更高；同时展示修改模型解释结果与原模型性能不匹配的例子，说明Grad-ECLIP存在问题。

**⚠️ 局限性**

实验仅针对CLIP模型，未验证在其他多模态或单模态Transformer上的泛化；对Attention-ECLIP的单头化简化可能在高阶注意力机制中失效；提出的忠实与一致性原则尚需在更大范围内进行验证。

---

## 243. Driving Intents Amplify Planning-Oriented Reinforcement Learning

**arXiv ID:** 2605.12625 | [PDF](https://arxiv.org/pdf/2605.12625v1)

**作者:** Hengtong Lu `[一作]` (Li Auto), Benjin Zhu `[通讯]` (Li Auto)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出两阶段训练框架 DIAL，先通过意图条件的 classifier‑free guidance（CFG）扩展连续动作策略的采样分布，再使用多意图 Group Relative Policy Optimization（GRPO）在偏好强化学习中保持这一多样性，以提升驾驶决策的偏好对齐性能。

**💡 创新点**

创新点在于：① 用离散驾驶意图对扩散动作头进行条件化并采用 CFG，突破单示例监督导致的模式崩塌；② 在 GRPO 中使用覆盖所有意图的采样组，避免在强化学习过程中恢复单一模式，从而在偏好评估上实现更高的 best‑of‑N 上限。

**🔧 技术方法**

技术包括：流匹配/扩散动作生成、意图条件化与 classifier‑free guidance、Group Relative Policy Optimization、reward‑hacking‑aware RFS 奖励、意图预测分类器以及多意图采样策略。

**📊 数据集**

使用 Waymo Open Dataset 的 Waymo Driving Benchmark，特别是 WOD‑E2E 的 438‑序列标签验证集和 338‑序列训练集。

**📈 对比分析**

与多种竞争性 VA/VLA 超级微调基线（WAM‑Flow、Curious‑VLA、AutoVLA、ReCogDrive）以及 RAP 进行比较。DIAL 在 best‑of‑128 上的 RFS 最高达 9.14，超过 RAP（8.5）和人类演示（8.13）；在 held‑out 上的 RFS 从 7.696 提升到 8.211，单意图方案均低于 8.00。

**⚠️ 局限性**

局限性包括：① 仅使用八个手工规则提取的驾驶意图，可能无法覆盖稀有或长尾行为；② RL 训练使用的验证集规模有限，奖励信号可能不足以代表所有复杂场景；③ 该方法依赖于意图分类器的准确性，若分类误差大可能影响采样多样性。

---

## 244. AgentLens: Revealing The Lucky Pass Problem in SWE-Agent Evaluation

**arXiv ID:** 2605.12925 | [PDF](https://arxiv.org/pdf/2605.12925v1)

**作者:** Priyam Sahoo `[一作]` (University of Illinois, Urbana-Champaign), Yu Hu `[通讯]` (Microsoft)

**通讯引用:** 9533 | [OpenAlex ID](https://openalex.org/A5014478407)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个基于前缀树接受器（PTA）和上下文敏感意图阶段标签的过程感知评估框架，对SWE代理轨迹进行质量评分。

**💡 创新点**

创新点在于构建任务级PTA参考、多阶段意图标签以及将结构对齐、覆盖、连贯性、时间分布等信号融合得到复合质量分，从而区分直接有效、弱成功（Lucky Pass）和失败轨迹。

**🔧 技术方法**

采用PTA构造、上下文敏感意图标签、四项信号（结构对齐、覆盖率、轨迹连贯性、时间分布）融合计算、废弃检测等技术。

**📊 数据集**

使用OpenHands在SWE‑bench Verified上生成的2,614条轨迹，筛选出47个任务的1,815条轨迹形成Bench数据集，用于训练和评估。

**📈 对比分析**

与单轨迹匹配、TF‑IDF对齐、密集嵌入对齐等基线对比，复合得分在通过率与模型排名上显示显著差异，AUROC约0.77，可有效区分通过与失败轨迹。

**⚠️ 局限性**

局限包括：需要至少两条通过轨迹才能构建PTA，难以扩展到任务缺少多样通过案例的场景；复合分数权重固定，可能不适用于所有任务；对大规模、高复杂度任务的可解释性和计算成本仍需进一步评估。

---

## 245. Dynamic Transaction Scheduling and Pricing in the Ethereum Mempool

**arXiv ID:** 2605.12794 | [PDF](https://arxiv.org/pdf/2605.12794v1)

**作者:** Fatemeh Fardno `[一作]` (University of Illinois Urbana-Champaign), S. Rasoul Etesami `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 781 | [OpenAlex ID](https://openalex.org/A5075507961)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了以太坊 EIP-1559 机制的动态交易调度问题，提出将价格视为对偶变量的原始-对偶解释，并将其推广为包含内存池状态的马尔可夫决策过程（MDP），利用自然策略梯度（NPG）学习最优价格策略；在同质交易和均匀到达的特殊情形下给出阈值最优策略和块容量下界。

**💡 创新点**

创新点包括：①将 EIP-1559 的指数价格更新从对偶角度重新推导，给出竞争比分析；②将静态分析扩展为动态 MDP，证明 NPG 产生的更新与 EIP-1559 极为相似；③在同质交易和均匀到达场景下解析阈值策略并给出块容量下界，提供理论上可行的简单定价策略。

**🔧 技术方法**

使用的技术主要包括：原始-对偶线性规划分析、竞争比证明、马尔可夫决策过程建模、自然策略梯度强化学习、凸分析与阈值策略证明。

**📊 数据集**

实验采用仿真数据，设置交易尺寸/价值组合如 {2,4,5,7} 与 {2,4,9}，以及泊松过程产生的随机到达；未使用公开真实区块链数据集。

**📈 对比分析**

与传统 EIP-1559 价格更新做对比，实验显示当超额惩罚增大时，NPG 学习的策略使平均调度量趋近目标块容量，奖励与 EIP-1559 相近且能稳定内存池，证明动态策略在模拟环境下性能可观。

**⚠️ 局限性**

局限性包括：仅考虑完全可观测状态；假设交易者非策略性且无报酬诱导；对偶分析基于线性松弛，可能偏离实际；实验仅在仿真环境，未验证在真实以太坊网络中的效果；对更一般到达分布和策略性用户的最优策略尚未给出。

---

## 246. AGOP as Explanation: From Feature Learning to Per-Sample Attribution in Image Classifiers

**arXiv ID:** 2605.12816 | [PDF](https://arxiv.org/pdf/2605.12816v1)

**作者:** Raj Kiran Gupta Katakam `[一作]` `[通讯]` (Credit Karma), Raj Kiran Gupta Katakam (Credit Karma)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于平均梯度外积（AGOP）的像素级归因方法，并实现了训练时累计的AGOPHook；

**💡 创新点**

将训练分布的梯度统计量与单样本梯度相结合，首次将AGOP作为后验归因工具，兼具全局与局部信息；

**🔧 技术方法**

梯度归因（VanillaGrad、IG、SmoothGrad、GradCAM）、AGOP-weighted和AGOP-training-hook实现；

**📊 数据集**

XAI-TRIS（8×8合成图像）与CLEVR-XAI（224×224真实图像）两个基准；

**📈 对比分析**

在四种任务（线性、乘法、平移+旋转、XOR）和两种数据集上与IG、SmoothGrad、GradCAM、VanillaGrad比较，AGOP-weighted在线性任务上mIoU提高44%，乘法任务上7倍于IG，CLEVR任务上mIoU提升18%，且在乘法任务零推理成本；

**⚠️ 局限性**

仅使用AGOP对角线，忽略交互信息；仅针对单个类别的AGOP；实验规模有限（小模型、未扩展到更大网络）。

---

## 247. Before the Last Token: Diagnosing Final-Token Safety Probe Failures

**arXiv ID:** 2605.12726 | [PDF](https://arxiv.org/pdf/2605.12726v1)

**作者:** Shravan Doda `[一作]` `[通讯]` (Kipo AI), Shravan Doda (Kipo AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于最终token的安全探针在检测绕过攻击（jailbreak）时的失效机制进行诊断，发现探针在探测被包装的有害请求时存在位置和子空间缺陷，并提出基于PCA‑HMM的轨迹诊断方法来补偿缺陷。

**💡 创新点**

揭示最终token读出在子空间能量分布上的缺陷，说明探针对被包装的有害请求几乎无感知，并提出轨迹级诊断作为补充，以提高检测召回率。

**🔧 技术方法**

SafeSwitch式二分类探针、SVD子空间能量分析、PCA降维+HMM轨迹概率、max‑pooling对比。

**📊 数据集**

SafeSwitch训练集、SorryBench、HarmBench（jailbreak模板）、XSTest、AdvBench。

**📈 对比分析**

在三大模型（Llama‑3.1‑8B、Mistral‑7B、OLMo3‑7B）上，最终token探针对清洁有害请求召回率约95%，但对jailbreak召回仅约70%；PCA‑HMM轨迹诊断将召回率提升至约98%并将XSTest误报率降至约10%以下。

**⚠️ 局限性**

诊断模型未作为部署检测器，阈值仅基于清洁训练集调优；对适应性攻击或长度控制的jailbreak鲁棒性未知；误报仍存在；无法解释HMM状态的机制。

---

## 248. PROMETHEUS: Automating Deep Causal Research Integrating Text, Data and Models

**arXiv ID:** 2605.12835 | [PDF](https://arxiv.org/pdf/2605.12835v1)

**作者:** Sridhar Mahadevan `[一作]` `[通讯]` (Adobe Research), Sridhar Mahadevan (Adobe Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本论文提出Prometheus框架，将检索到的文献、数据和代码转化为分层的因果地图（Topos World Model），通过局部预测状态模型和拼接诊断实现可导航的因果知识可视化；

**💡 创新点**

核心创新在于将因果抽取结果组织成可限制、可拼接的局部PSR层，结合层级限制图、对齐诊断和持久化状态，构建一个可检验、可更新的因果拓扑结构；

**🔧 技术方法**

技术实现基于大语言模型的因果关系抽取、事件分段、cSQL样式表格化、预测状态表学习、层级覆盖与限制映射、拼接张量诊断以及Claims Atlas交互界面；

**📊 数据集**

实验使用多领域案例集，包括海洋温度对鱼群影响、GLP‑1减重药物、红酒中的延胡索素健康效益，以及微塑料光学强迫、印度河流域水文、Sachs蛋白信号与唱鼠MAPseq等四大对照实验的数据与源代码；

**📈 对比分析**

通过与传统平面摘要、知识图谱以及单一因果图的对比，Prometheus在抽取准确率、覆盖度、漂移可视化、拼接冲突检测等维度均表现优于单一模型；在海洋案例中实现约3000条事件抽取、199个局部PSR和仅0.018的平均拼接误差，LLM调用成本约1.24美元；

**⚠️ 局限性**

局限性包括对LLM抽取质量的高度依赖、缺乏统一全局图谱、需要人工定义上下文覆盖、对稀疏或未标注的源数据支持有限，以及计算量随语料规模增长显著上升。

---

## 249. Visual Aesthetic Benchmark: Can Frontier Models Judge Beauty?

**arXiv ID:** 2605.12684 | [PDF](https://arxiv.org/pdf/2605.12684v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 250. Do Skill Descriptions Tell the Truth? Detecting Undisclosed Security Behaviors in Code-Backed LLM Skills

**arXiv ID:** 2605.12875 | [PDF](https://arxiv.org/pdf/2605.12875v1)

**作者:** Wenhui He `[一作]` (Skill Security Team), Baoning Niu `[通讯]` (Skill Security Team)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究程序化LLM技能的描述与实现之间的安全行为一致性，并提出一种自动化检测框架；

**💡 创新点**

构建了11类安全属性分类法、源级安全属性图（SPG）以及基于LLM的跨模态一致性检查流程；

**🔧 技术方法**

采用Joern生成代码属性图，基于关键词规则构造SPG，随后使用GPT‑5等大模型对描述与SPG进行一致性判定；

**📊 数据集**

使用4,556个公开程序化技能（来自Anthropic、OpenAI、SkillsMP和skills.rest）进行评估，其中920个样本用于手工构建安全属性分类法；

**📈 对比分析**

与人工双盲审查对比，SkillScope在完整数据集上实现84.8%精度、96.5%召回、90.3%F1；消除分类法或SPG会显著降低性能；跨模型实验表明GPT‑5最佳，Gemini‑2.5‑flash次之，Llama‑3.3‑70b性能最低；

**⚠️ 局限性**

局限包括：关键词定位规则可能漏检SDK或自定义实现；仅支持Joern支持的语言；数据集来源有限，可能无法代表所有生态；性能依赖所选LLM，未考虑攻击者刻意规避；人工评估仍存在主观性。

---

## 251. An Agentic LLM-Based Framework for Population-Scale Mental Health Screening

**arXiv ID:** 2605.13046 | [PDF](https://arxiv.org/pdf/2605.13046v1)

**作者:** Giuliano Lorenzoni `[一作]` (University of Waterloo), Donald Cowan `[通讯]` (University of Waterloo)

**通讯引用:** 24447 | [OpenAlex ID](https://openalex.org/A5081821121)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于代理的LLM框架，用于大规模心理健康筛查，特别是转录文本抑郁检测。

**💡 创新点**

将每个处理阶段封装为LangChain代理，使用显式策略和锁定机制，结合代理评估减少昂贵评估，保证非回归。

**🔧 技术方法**

采用LangChain、LLM、检索增强生成（RAG）、句向量（SentenceTransformer）、余弦相似度、动态Top‑k、MMR、多代理协同以及冻结/回滚策略。

**📊 数据集**

使用DAIC‑WOZ临床访谈语料（包含抑郁与非抑郁转录文本）。

**📈 对比分析**

通过代理筛选与金标验证相结合，锁定后与基线比较，宏F1和召回保持在0.825/0.875，且在不同配置间无回归，性能稳定。

**⚠️ 局限性**

局限在于实验仅覆盖单一数据集、代理空间相对有限、模型随机性导致结果波动，以及未探索更复杂的检索策略或多模态输入。

---

## 252. Large Language Models Lack Temporal Awareness of Medical Knowledge

**arXiv ID:** 2605.13045 | [PDF](https://arxiv.org/pdf/2605.13045v1)

**作者:** Zihan Guan `[一作]` (University of Virginia), Anil Vullikanti `[通讯]` (University of Virginia)

**通讯引用:** 4171 | [OpenAlex ID](https://openalex.org/A5044848288)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了TempoMed‑Bench，首个专门评估医学 LLM 时序知识的基准；在该基准上评测了十多款前沿 LLM 的时间敏感性；进一步验证了代理式检索增强（RAG）对提升时序知识准确率的有限效果。

**💡 创新点**

①首次将临床指南的时间演进视为评估时序知识的自然“快照”；②提出了“时序一致性”指标，细化了模型在不同年份下对同一知识点的回答模式；③系统性验证了 RAG 工具在时序医学知识上的提升不显著。

**🔧 技术方法**

使用 LLM（如 GPT‑4、Claude 等）进行预训练/后训练知识编码；利用 LLM 生成和验证指南差异及 MCQ；对模型进行多模型多版本评测；引入代理式检索工具 ToolUniverse、Biomni 进行检索增强。

**📊 数据集**

从 PubMed Central（PMC）抽取 20,000+ 指南文章，构成 3,411 条指南版本轨迹；从轨迹中提取 721 对差异生成 MCQ，形成 TempoMed‑Bench。

**📈 对比分析**

对比 10+ LLM（参数规模从数亿到数十亿）在 TempoMed‑Bench 上的准确率。结果显示：①模型对最新指南的掌握相对较好，但准确率随时间呈线性下降；②对历史知识准确率仅为最新知识的 25%–54%；③代理式 RAG 在提升准确率时仅能带来 1%–14% 的增益，甚至在部分模型上导致性能下降。

**⚠️ 局限性**

①数据集规模相对有限（721 条 MCQ）；②仅评估单一检索增强工具，未尝试 fine‑tune 或更高级代理系统；③由于 PMC 文献完整性受限，轨迹构建可能不完整。

---

## 253. Rethinking Efficient Graph Coarsening via a Non-Selfishness Principle

**arXiv ID:** 2605.13021 | [PDF](https://arxiv.org/pdf/2605.13021v1)

**作者:** Xu Bai `[一作]` (Shanghai Jiao Tong University), Meng Jin `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 107474 | [OpenAlex ID](https://openalex.org/A5100374993)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于非自私原则的图粗化方法（NOPE）以及其快速近似版本（NOPE*），通过定义邻域干扰指标来指导节点合并，显著降低粗化过程的时间与内存消耗。

**💡 创新点**

创新点包括：①引入邻域干扰指标（Neighborhood Interference Index），从局部语义一致性角度重新定义粗化准则；②设计贪婪优先队列算法实现近线性时间、线性内存的粗化；③在局部等向性假设下推导期望干扰，得到 NOPE*，将每次合并的复杂度从 O(Δ·d) 降到 O(d)，进一步提升可扩展性。

**🔧 技术方法**

核心技术：邻域干扰指标计算、贪婪优先队列调度、缓存更新机制、局部等向性假设下的期望干扰公式、线性时间/内存复杂度分析。

**📊 数据集**

使用的文本属性图数据集包括：Citeseer、Ogb-Arxiv、Book、Products 等，涵盖从小型到大规模的多种场景。

**📈 对比分析**

与传统基于相似度的粗化方法（FGC、UGC、MPG、A-CM）以及 GNN 训练基线（GCN、GIN、GraphSAGE）进行对比。在 LLM 与 GNN 分类任务中，NOPE/NOPE* 在运行时间上提升 1–3 个数量级，内存占用保持在 1–3 个数量级以内，且在节点分类准确率/ F1 上与原图性能相当，甚至优于部分基线。

**⚠️ 局限性**

局限性：NOPE* 在高粗化率下近似误差显著，可能导致语义混杂和分类性能下降；当前方法主要针对静态文本属性图，尚未针对动态图、流式大规模图的进一步扩展。

---

## 254. JEDI: Joint Embedding Diffusion World Model for Online Model-Based Reinforcement Learning

**arXiv ID:** 2605.13013 | [PDF](https://arxiv.org/pdf/2605.13013v1)

**作者:** Jing Yu Lim `[一作]` (National University of Singapore), Dianbo Liu `[通讯]` (National University of Singapore)

**通讯引用:** 6609 | [OpenAlex ID](https://openalex.org/A5014407399)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了JEDI，一种在线端到端的潜在扩散世界模型，用于模型基强化学习；

**💡 创新点**

首次将扩散去噪作为JEPA式联合嵌入预测目标，直接从去噪损失中训练潜在空间，且给出信息瓶颈理论解释；

**🔧 技术方法**

使用潜在条件扩散、JEPA（联合嵌入预测架构）、端到端编码器训练、REINFORCE策略学习、随机噪声注入、潜在clamp、随机切换等技术；

**📊 数据集**

在Atari100k（26种游戏、5个种子）和Craftium（4种任务、3个种子）数据集上进行评估；

**📈 对比分析**

与DIAMOND（像素扩散）、Horizon Imagination（分离训练潜在）、DreamerV3、STORM、TWM、IRIS、TWISTER等方法对比，JEDI在IQM、最优性间隙、超人类任务数上匹配或超越DIAMOND，在低HNS任务上显著提升；在资源方面使用43%更少VRAM、采样速度3倍以上、训练速度2.5倍以上；

**⚠️ 局限性**

仅在Atari和Craftium上验证，未在DMControl、Procgen等更大规模或高分辨率环境测试；训练速度仍不及HI，且对潜在clamp、随机切换等超参数敏感；理论证明仅提供学习性质解释，缺乏优化保证。

---

## 255. Toward Practical Age-of-Information Scheduling in 5G Cellular

**arXiv ID:** 2605.13012 | [PDF](https://arxiv.org/pdf/2605.13012v1)

**作者:** Zhuoyi Zhao `[一作]` (Northwestern University), Igor Kadota `[通讯]` (Northwestern University)

**通讯引用:** 1932 | [OpenAlex ID](https://openalex.org/A5052671825)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种低复杂度的估计器和相应的基于最大权重的调度策略，以在5G gNB缺乏源端和目的端时间戳观测的情况下实现信息新鲜度感知调度。

**💡 创新点**

创新点在于用gNB可见的观察（如SRS信道测量、成功接收记录）和伯努利近似来估计UE生成时刻与目的端AoI，并将该估计与即时链路可靠性结合到Max‑Weight调度中，且实现了𝒪(N)的每槽计算复杂度。

**🔧 技术方法**

技术手段包括基于统计推断的低复杂度估计、Lyapunov优化驱动的最大权重调度、NetSim 5G NR仿真器实现以及MATLAB计算验证。

**📊 数据集**

使用的数据集为合成网络仿真，包含N=40个UE与目的端对，四类伯努利到达速率（0.05、0.2、0.5、1），随机3GPP 3.8.901通道参数与传输延迟θ_i，权重α_i随机分配。

**📈 对比分析**

通过与传统PF、RR以及高复杂度MW‑O和MW‑EnF进行对比，实验显示MW‑LC在平均加权AoI上比PF、RR低63%/44%，比MW‑O低21%；MATLAB结果表明LC估计误差与更精细估计器相差不大，AoI性能近似相同。

**⚠️ 局限性**

局限性包括对包身份信息、长周期流量参数λ_i、目的端可靠性p_i^D以及伯努利生成假设的依赖，且尚未在软件定义无线电或开源网络栈上实现。

---

## 256. Amortized Guidance for Image Inpainting with Pretrained Diffusion Models

**arXiv ID:** 2605.13010 | [PDF](https://arxiv.org/pdf/2605.13010v1)

**作者:** Yilie Huang `[一作]` (Columbia University), Xun Yu Zhou `[通讯]` (Columbia University)

**通讯引用:** 12853 | [OpenAlex ID](https://openalex.org/A5109487227)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

训练一个小型可复用的导引模块，固定预训练扩散模型，在推理时直接使用该模块完成图像修复任务，无需逐图实例化优化。

**💡 创新点**

构造了一个确定性导引控制问题并证明其等价于一个辅助高斯随机化的连续时间RL问题，形成了一个优化保持桥梁；通过该桥梁实现了离线训练的可扩展 Actor‑Critic 算法。

**🔧 技术方法**

连续时间强化学习、Actor‑Critic 算法、扩散模型控制导引、Gaussian随机化策略、马尔科夫过程理论。

**📊 数据集**

AFHQv2、FFHQ、ImageNet，分别在像素级 EDM 和潜在级 EDM2 训练管线下测试。

**📈 对比分析**

与固定模型基础方法（Unguided、Replacement、MCG、DPS）以及迭代优化方法（RePaint）和潜在空间可复用方法（LatentPaint）比较；在 PSNR、SSIM、LPIPS 等指标上，AID 在保持相同 NFE 的前提下，质量提升显著，速度提升高达 10–15 倍，且在不同遮罩类型和低延迟设置下仍保持优势。

**⚠️ 局限性**

依赖预训练扩散模型的质量；仅对单个任务分布训练，可能对大规模多任务分布泛化有限；当前实现仍需要对每个任务采样任务分布，未能完全解决极端遮罩或异构数据域的鲁棒性问题。

---

## 257. Occlusion-Based Object Transportation Around Obstacles With a Swarm of Miniature Robots

**arXiv ID:** 2605.13006 | [PDF](https://arxiv.org/pdf/2605.13006v1)

**作者:** Breno Cunha Queiroz `[一作]` (University of São Paulo), Daniel MacRae `[通讯]` (Rijksuniversiteit Groningen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在完全去中心化、无通信、仅基于视觉感知的微型机器人群体中，提出一种新的有限状态机（FSM）方案，在原始遮挡驱动的物体搬运策略上加入子目标（Sub‑Goal）状态，使机器人能够在障碍物存在的环境中沿着由多个子目标构成的链路协同推动物体到达目标。

**💡 创新点**

创新点在于：①通过子目标状态让机器人自发形成可见目标的中继点，突破单一遮挡路径受限的瓶颈；②保持与原始遮挡策略相同的低复杂度与无通信特性；③在不需要全局地图或预先规划路径的情况下实现多障碍、多形状物体的自主搬运。

**🔧 技术方法**

主要技术包括：
- 仅使用RGB摄像头与8路红外接近传感器的感知模型；
- 通过有限状态机控制机器人行为（S1‑S5）；
- 在仿真环境中使用 Atta 物理仿真器与 Box2D 进行动力学模拟；
- 通过视觉检测实现遮挡判定、目标检测与子目标颜色切换。

**📊 数据集**

数据集：全部实验均为仿真数据，使用四种不同环境（无障碍、单墙、双墙、双通道）以及七种物体形状（正方形、圆形、矩形、三角形、十字、L形、H形），每种配置下多次（50次）随机起始实验，生成了数千条任务完成与轨迹数据。

**📈 对比分析**

比较方法：①与原始遮挡 FSM 进行对比；②与基于遥控子目标（已知地图、Dijkstra 路径）的仿真方案进行基准测试。性能表现：
- 在无障碍环境中两种 FSM 完成率均为 100%，时间与路径效率无显著差异；
- 在包含障碍的环境中，所提 FSM 能够在大于等于 15 只机器人时实现 100% 的成功率（Corner）、>94%（2‑Corners、Middle），完成时间随机器人数量下降，路径效率在 0.65–0.82 之间；
- 对比遥控方案，遥控子目标在时间与路径效率上平均优于所提 FSM，尤其在 Middle 环境中成功率差距明显（100% vs. 76%）。

**⚠️ 局限性**

局限性：
- 仅在理想感知下验证，未考虑摄像头噪声或光照变化；
- 无显式路径规划，无法处理多对称路径的选择，导致 Middle 环境效率下降；
- 未实现物体与障碍靠拢时的逃逸或旋转控制，导致某些形状（矩形、H）在特定障碍布局下易卡住；
- 未确定最佳机器人数量或基于物体质量、尺寸的自适应策略；
- 同步问题：多机器人同时进入子目标状态可能产生不必要的碰撞或路径竞争；
- 仅通过仿真验证，缺乏真实机器人实验与外部环境验证。

---

## 258. Not Just RLHF: Why Alignment Alone Won't Fix Multi-Agent Sycophancy

**arXiv ID:** 2605.12991 | [PDF](https://arxiv.org/pdf/2605.12991v1)

**作者:** Adarsh Kumarappan `[一作]` (California Institute of Technology), Ananya Mujoo `[通讯]` (Evergreen Valley College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多代理LLM管线在同伴争论下出现的正确→错误答案转化现象（yield），验证其是否由RLHF诱发，定位中间层机制并提出跨帧消除该漏洞的方法。

**💡 创新点**

①证明该漏洞是预训练模型固有的中间层缺陷，而非RLHF造成的；②精确定位关键层为L14–L18，发现注意力主导的压制机制；③揭示攻击面由渠道框架与共识强度两因素组成；④单一正当反驳者可跨帧削弱50%以上yield，优于传统提示级防御。

**🔧 技术方法**

使用激活补丁、logit lens、线性探针、稀疏自编码器（Goodfire SAE）、差分均值（DIM）、注意力/MLP分解以及层级补丁与修复技术，对中间层进行机制层面分析。

**📊 数据集**

主要使用MMLU人文子集400题（以及200题STEM和43题计算机科学），对四大模型族（Llama、Mistral、Gemma、Qwen）的预训练与Instruct版本进行匹配评估。

**📈 对比分析**

通过对比预训练基模型与Instruct模型、不同渠道与共识强度组合、系统提示防御与结构化异议的yield，发现预训练模型yield≥Instruct，单异议能使yield下降>50pp，而系统提示防御仅在其设计攻击下下降65pp，其他变体几乎无效。

**⚠️ 局限性**

仅在静态提示与单轮对话中评估，未考虑生成时动态对话和时间演化；实验仅覆盖四大模型族，未测试其他类型的sycophancy（如奉承、用户偏好同调）及更大规模管线。

---

## 259. Decision Tree Learning on Product Spaces

**arXiv ID:** 2605.12983 | [PDF](https://arxiv.org/pdf/2605.12983v1)

**作者:** Arshia Soltani Moakahr `[一作]` (University of Maryland), MohammadTaghi Hajiaghayi `[通讯]` (University of Maryland)

**通讯引用:** 5831 | [OpenAlex ID](https://openalex.org/A5111876448)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文扩展了决策树学习的理论分析，将经典的自上而下贪婪启发式方法从均匀分布推广到任意产品分布。

**💡 创新点**

创新点在于提供了在更广泛的产品分布下，贪婪启发式方法的性能保证，并提出了一种完全无参数的算法，不需要事先知道最优树的大小或深度。

**🔧 技术方法**

使用了自上而下的贪婪启发式算法，基于影响力的分裂标准。

**📊 数据集**

未具体提及使用的数据集，但讨论了在任意产品分布下的决策树学习。

**📈 对比分析**

与Blanc等人的方法相比，本文的理论结果在特定情况下提供了更紧的界限，并且适用于更广泛的分布类。

**⚠️ 局限性**

限制在于实际的决策树实现（如CART）通常依赖于后处理的剪枝步骤，而本文分析的是未剪枝的树，未来的研究可以扩展到剪枝规则的可证明性分析。

---

## 260. CLOUDBURST: Cloud-Layer Observations Using Beacons for Unified Real-time Surveillance and Threat Attribution

**arXiv ID:** 2605.12976 | [PDF](https://arxiv.org/pdf/2605.12976v1)

**作者:** Abraham Itzhak Weinberg `[一作]` `[通讯]` (AI Experts), Abraham Itzhak Weinberg (AI Experts)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了 CLOUDBURST，一套专为云原生环境设计的被动信标分类与度量框架，并对其在 AWS、GCP、Azure 与 OCI 上的归因效果进行了系统评估。

**💡 创新点**

首创云原生被动信标的六类向量体系，并引入 Cloud Attribution Score（CAS）—包含瞬态惩罚、IAM 覆盖深度与跨云相关三维度的归因质量度量；量化容器化环境下归因衰减；提供跨云供应商的统一评估。

**🔧 技术方法**

利用 CloudTrail / Audit Log 提取 IAM 线索、实现多云信标向量（S3 预签名 URL、容器镜像、IAM Canary、Terraform 模块、K8s Secret、Serverless 触发器），并通过三种扫描器模型（Macie/Checkov/Prisma Cloud）评估检测抵抗；采用数学模型计算 E_p、I_c、M_b 并求得 CAS。

**📊 数据集**

构建了 21 个多云部署的信标实例（覆盖 6 类向量跨 4 大云），模拟 205 条回调，并基于三类攻击者（Naive、Advanced、APT）设置不同的网络与行为参数；未使用公开数据集，全部采用内部 FinTech 环境配置与模拟。

**📈 对比分析**

通过对三类扫描器的检测概率合成评估检测抵抗（DR），与 CAS、回调次数、归因置信度等指标进行 ANOVA 与 Kruskal‑Wallis 检验；结果显示 IAM Canary 取得最高 CAS 与 DR，S3 预签名 URL 取得最佳 DR；所有向量在 48 小时内 CAS 下降至约 0.2，未能实现 P(H|E) ≥ 0.85 的高置信归因。

**⚠️ 局限性**

局限性包括：仿真环境与真实 Passive hack‑back 回调存在差距；扫描器模型抽象化，实际规则可能不同；IAM 角色链覆盖不足，无法完全捕捉多跳推断；未达成高置信归因，表明需要补充额外信号；对真实攻击者行为与多云路径匹配的完整性仍待验证。

---

## 261. Leveraging Speech to Identify Signatures of Insight and Transfer in Problem Solving

**arXiv ID:** 2605.12970 | [PDF](https://arxiv.org/pdf/2605.12970v1)

**作者:** Linas Nasvytis `[一作]` (Stanford University), Judith E. Fan `[通讯]` (Stanford University)

**通讯引用:** 922 | [OpenAlex ID](https://openalex.org/A5003160263)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过让 189 名参与者在解决一系列匹配棒算术问题时进行思维口述，研究了洞察性问题解决后对后续问题的转移效应。

**💡 创新点**

创新点在于发现第一成功后出现的言语变化（话语量、速度、内容）在可重复使用结构时持续存在，而在不同结构时不再持续，从而提供了洞察与转移的可观察语言签名。

**🔧 技术方法**

使用自动语音识别（WhisperX）、声活动检测（Silero）、文本嵌入（OpenAI 模型）以及 GPT‑5.1 对话语进行推理动作标注，并结合混合效应模型进行统计分析。

**📊 数据集**

数据集为 25 个新颖的匹配棒算术问题（5 种类型，每种 5 个），并收集了 189 名参与者的音频与口述数据。

**📈 对比分析**

通过 Same 与 Different 两组的实验设计，利用混合效应模型比较不同阶段的准确率、反应时间、话语密度与语义内容，Same 组在准确率（0.75 vs 0.32–0.39）和反应时间（更快）以及话语变化上显著优于 Different 组。

**⚠️ 局限性**

局限包括使用首次成功作为洞察事件的客观标记，缺少主观“aha”体验；自动转写与标注可能引入噪声；以及结果仅在匹配棒算术问题上验证，通用性需进一步研究。

---

## 262. Controlling Logical Collapse in LLMs via Algebraic Ontology Projection over F2

**arXiv ID:** 2605.12968 | [PDF](https://arxiv.org/pdf/2605.12968v1)

**作者:** Hisashi Miyashita `[一作]` (Mgnite Inc), Mgnite Inc `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了 Algebraic Ontology Projection (AOP)，通过将 LLM 隐藏状态投射到 𝔽₂ 空间，以线性方式提取本体关系。

**💡 创新点**

在仅用 42 对关系约束的极小监督下实现 93.33% 零样本包含准确率，并引入 Semantic Crystallisation (SC) 指标量化层级结构与 Late‑layer Collapse 现象。

**🔧 技术方法**

使用两层线性投射网络、二进制阈值、Localized Mean Pooling、SC 统计，以及系统提示作为边界条件。

**📊 数据集**

使用 42 条跨四个语义域（生物、矿物、物理、抽象）的 is‑a、has‑a、否定关系对，评估 15 条未见概念对的零样本。

**📈 对比分析**

与无提示、优化提示和指令调优的 Gemma‑2、Qwen2.5、mpnet 三大模型对比，最佳配置在 Gemma‑2 Instruct+优化提示下达 93.33% 准确率，其他模型平均达到 86.67%。

**⚠️ 局限性**

主要局限在于 SC 仅衡量隔离度、对 is‑a/has‑a 约束缺乏完整度，局部平均池化对长上下文不适用，系统提示与指令调优互补但需进一步优化。

---

## 263. ViDR: Grounding Multimodal Deep Research Reports in Source Visual Evidence

**arXiv ID:** 2605.13034 | [PDF](https://arxiv.org/pdf/2605.13034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 264. Uncertainty-aware Spatial-Frequency Registration and Fusion for Infrared and Visible Images

**arXiv ID:** 2605.13049 | [PDF](https://arxiv.org/pdf/2605.13049v1)

**作者:** Xingyuan Li `[一作]` (Dalian University of Technology), Jinyuan Liu `[通讯]` (Dalian University of Technology)

**通讯引用:** 11031 | [OpenAlex ID](https://openalex.org/A5100675904)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种同时在空间域和频率域进行注册与融合的框架（SFRF），解决多模态图像未对齐导致的误差堆叠问题。

**💡 创新点**

创新点包括：①基于不确定性估计的多尺度迭代注册（MIR），①1通过MCFE+GRU-FRB递归细化变形场；②采用热辐射分布一致性作为频域监督；③设计双分支空间‑频率融合模块（DSFF），融合相位与幅值特征并结合空间注意力。

**🔧 技术方法**

核心技术包括：蒙特卡洛采样的变形场估计（MCFE）、GRU‑基的场细化块、基于相关性的多尺度场融合（MSF）、热辐射分布一致性损失、FFT频域融合、相位/幅值自注意力、Restormer与ResBlock空间融合。

**📊 数据集**

在四个公开数据集上评估：RoadScene、MSRS、M³FD 和 TNO（其中 TNO 仅用于测试）。

**📈 对比分析**

与 GLU‑Net、CGRP、SuperFusion、MURF、IMF、BSAFusion 等现有注册方法及多种融合方法比较，SFRF 在 NCC、RMSE、MEE 等注册指标上均获得最高或接近最高；在融合指标（CC、SSIM、MG）上亦领先，整体性能显著优于现有方法。

**⚠️ 局限性**

局限性：模型结构较为复杂，训练与推理需要较高算力；对极端大范围非线性失真仍可能出现残留误差；在极低光照或噪声极高的环境下，热辐射一致性约束的效果尚待进一步验证。

---

## 265. Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models

**arXiv ID:** 2605.13043 | [PDF](https://arxiv.org/pdf/2605.13043v1)

**作者:** Yejin Lee `[一作]` (Yonsei University), Yo-Sub Han `[通讯]` (Yonsei University)

**通讯引用:** 1472 | [OpenAlex ID](https://openalex.org/A5077698683)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种推理时的安全框架，通过在扩散语言模型的迭代去噪过程中进行逐步干预，实现对 jailbreak 攻击的防御。

**💡 创新点**

创新点包括：① 通过 Contrastive Safety Direction (CSD) 捕捉有害与安全生成的语义边界；② 在早期去噪步骤使用自适应 steering 以引导生成轨迹；③ 在后期使用 token‑level remasking 细粒度纠正有害词汇；整个方法无需额外微调，可直接插入现有模型。

**🔧 技术方法**

采用的技术包括：扩散语言模型的掩码去噪框架、CSD 方向估计、表示层级自适应 steering、令牌级 remasking、置信度基选取、以及基于对比学习的安全向量构建。

**📊 数据集**

使用的数据集：WildJailbreak（构造对比样本 5,763 条）、JailBreakBench、AdvBench、HarmBench、StrongReject（攻击评估），以及 TruthfulQA、MATH‑500、MMLU（通用性能评估）。

**📈 对比分析**

与 DiffuGuard、Self‑reminder 等现有防御方法对比，实验显示在 LLaDA 和 Dream 模型上，攻击成功率（ASR）从 35.67% 降至 25.67% 或从 34.19% 降至 0.64%，并且保持了接近原模型的生成质量；在通用任务上，性能下降幅度小于对手方法。

**⚠️ 局限性**

局限性：对超参数（如 steering 强度、remasking 比例、层选择）的敏感性；在极端或未知攻击场景下仍可能存在漏洞；仅在推理阶段干预，无法彻底解决训练阶段的安全问题；对更大规模模型的适用性和计算成本尚未充分验证。

---

## 266. Conveyor Parcel Routing with Order-Contiguous Arrivals

**arXiv ID:** 2605.13035 | [PDF](https://arxiv.org/pdf/2605.13035v1)

**作者:** Takuro Kato `[一作]` (Toyota Industries Corporation), Keisuke Okumura `[通讯]` (National Institute of Advanced Industrial Science and Technology)

**通讯引用:** 2687 | [OpenAlex ID](https://openalex.org/A5038362443)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对仓库输送网络中，包裹需要在一向传送带网络中实现碰撞避免且到达时保持订单连续性的问题，提出了在线多代理路径规划（online MAPF-OC）的形式化模型，并设计了基于三层优先级搜索的 Dual-Ordering Prioritized Planning（DOPP）算法，能够快速生成可行方案并在有限时间内持续改进。

**💡 创新点**

创新点包括：① 将订单层级与代理层级的优先级约束结合，形成三层搜索结构，保证在满足订单连续性约束的同时可在多项式时间内得到可行解；② 通过批量重规划和邻域搜索实现在线实时规划，保持已执行路径的“已承诺后缀”，提升实时性；③ 提供完整性证明与理论分析，证明该结构在“可形成”场景下总能得到可行方案。

**🔧 技术方法**

技术手段主要有：优先级规划（Prioritized Planning）配合空间-时间 A*；订单层与代理层的优先级约束生成；邻域搜索（Level‑3 订单窗口交换与 Level‑2 代理内部置换）实现任何时间改进；批量重规划框架；并使用多线程并行评估候选约束。

**📊 数据集**

数据集：使用从真实仓库提取的多种传送带网络（Small‑A、Medium‑A、Large、Complex、Complex‑B 等），以及随机生成的订单与到达流（每时间步 λ 条新包裹，订单大小随机）。

**📈 对比分析**

比较方法：以 PIBT‑AC（基于 PIBT 的局部规则 + 订单连续性控制）为基准，在一次性与持续在线（10 个窗口）两种情形下进行实验。DOPP 在所有网络与服务时间 k 下均优于 PIBT‑AC，normalized makespan 远低于下界比例，平均运行时间 <1 s；任何时间改进可达 10%+，并保持高目标利用率（≈100% 时需扩容）。

**⚠️ 局限性**

局限性：① 需要同一订单在一次重规划窗口内全部可见；② 假设所有订单的服务时间相同；③ 对极大规模/高密度情形仍受空间-时间搜索的时间和内存限制；④ 算法仅针对一向传送带网络，无法直接推广到双向或更复杂拓扑；⑤ 仅提供可行解而非全局最优，最优性无法保证。

---

## 267. Understanding and Accelerating the Training of Masked Diffusion Language Models

**arXiv ID:** 2605.13026 | [PDF](https://arxiv.org/pdf/2605.13026v1)

**作者:** Chunsan Hong `[一作]` (KAIST), Jong Chul Ye `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过分析Masked Diffusion Models（MDM）训练慢的原因，提出了bell‑shaped time sampling以提升训练效率并保持最终性能

**💡 创新点**

发现语言的局部性偏差导致MDM在低/高上下文区域浪费训练资源，并提出通过在时间采样上做“钟形”分布聚焦中间上下文，显著加速学习

**🔧 技术方法**

采用了蒙版扩散模型框架、时间采样重构、贝叶斯优化的bell‑shaped分布、对比实验与ARM/σ‑ARM/AoARM的共享学习目标

**📊 数据集**

在One Billion Word Benchmark（LM1B）、OpenWebText、GPT‑2 Large 预训练、以及多项选择QA和生成任务（LAMBADA、Obqa、Wino、PIQA、SIQA、TriQA、HellaSwag、ROCStories）上进行实验

**📈 对比分析**

与标准MDM（均匀时间采样）对比，bell‑shaped时间采样在LM1B上实现约3.9×、2.3×、1.8×的训练速度提升；在后续的零样本PPL、生成PPL以及多任务下游性能均优于基线；在CPT+SFT设置下提升多项选择任务的准确率和生成质量（WinRate提升约7%）

**⚠️ 局限性**

主要局限在于实验规模仍以数百M参数为主，未在更大规模（百亿级）MDM上彻底验证；此外，需要对bell‑shaped分布的超参（均值、方差）进行调优，且对不同语言/任务的泛化性仍需进一步研究

---

## 268. OCH3R: Object-Centric Holistic 3D Reconstruction

**arXiv ID:** 2605.13018 | [PDF](https://arxiv.org/pdf/2605.13018v1)

**作者:** Yi Du `[一作]` (Stanford University), Leonidas Guibas `[通讯]` (Stanford University)

**通讯引用:** 83918 | [OpenAlex ID](https://openalex.org/A5065368881)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出OCH3R统一框架，单个RGB图像通过一次前向传播即可预测所有物体实例的类别、6D位姿和高精度3D高斯模型，实现对象级全景重建。

**💡 创新点**

创新点：①将深度、CLIP语义、NOCS坐标和3D高斯作为同一Transformer的像素级输出，打破传统多阶段管线；②引入Canonical-Space Supervision，在目标坐标系中监督高斯，解决遮挡与无标签问题；③单通道推理可独立于物体数量，速度提升数百倍。

**🔧 技术方法**

技术：48层ViT Transformer + DINOv2 backbone；CRF + RANSAC-Umeyama实现实例发现与位姿估计；3D Gaussian Splatting渲染；Canonical‑Space Supervision；off‑ray offset预测与SH颜色编码。

**📊 数据集**

数据集：构建统一对象级3D数据集，整合PACE、Omni6DPose、YCB‑Video、HOPE、NOCS（real）等；训练集包含Google Scanned Objects、HyperSim等；验证/测试集覆盖多类别、多姿态、遮挡情形。

**📈 对比分析**

与Gen3DSR、ACDC、AoE等基线对比，OCH3R在Chamfer Distance、F‑1、CLIP相似度上均显著优于所有基线；推理时间从数分钟降至0.7 s，速度提升约2000×。

**⚠️ 局限性**

局限：在极端遮挡或非室内场景中仍可能出现误检；单像素高斯数量固定导致细节与纹理细腻度受限；对尺度变化大、材质复杂的物体的重建精度尚待进一步提升。

---

## 269. Classification of ternary maximal self-orthogonal codes of length 25

**arXiv ID:** 2605.13007 | [PDF](https://arxiv.org/pdf/2605.13007v1)

**作者:** Makoto Araya `[一作]` (Shizuoka University), Masaaki Harada `[通讯]` (Tohoku University)

**通讯引用:** 2612 | [OpenAlex ID](https://openalex.org/A5015902214)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

完成了对三元最大自正交码长度为25的完整分类，得到共139613个非等价码，并给出了它们的最小重量分布。

**💡 创新点**

创新点在于利用已分类的三元自对码长度24，通过缩短/扩展、质量公式与等价性测试，首次实现长度25的完整枚举，并给出了相应的自动群大小分布及其下界推断。

**🔧 技术方法**

采用了三元码的缩短与扩展方法、质量公式、等价性测试与自动群计算（基于Nauty和Magma），并使用C/C++与NTL进行大规模枚举与计算。

**📊 数据集**

主要数据集来源于已知长度24的三元自对码与其子码（[24,11]），通过枚举子码获得长度25的候选码，然后进行等价性判定。

**📈 对比分析**

通过质量公式检验总数与等价分类结果一致，验证了完整性；在长度25上未涉及性能比较，主要通过计数与质量公式的一致性作为验证手段。

**⚠️ 局限性**

仅覆盖长度≤25的情况；对更长长度的分类仍未完成，仅给出了基于质量公式的下界；计算成本极高，无法直接推广至更长码长。

---

## 270. \emph{DRIFT}: A Benchmark for Task-Free Continual Graph Learning with Continuous Distribution Shifts

**arXiv ID:** 2605.12998 | [PDF](https://arxiv.org/pdf/2605.12998v1)

**作者:** Guiquan Sun `[一作]` (University of Connecticut), Dongjin Song `[通讯]` (University of Connecticut)

**通讯引用:** 6241 | [OpenAlex ID](https://openalex.org/A5013197657)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建任务无边界的持续图学习基准DRIFT，模拟从硬切换到平滑漂移的连续任务混合流。

**💡 创新点**

提出统一的时间混合模型，将任务转移视为高斯混合，从而实现连续过渡谱。

**🔧 技术方法**

采用高斯时间混合分布、无任务标识的持续学习方法（ER、A‑GEM、GSS、MAS*、SSM、SEM、DMSG）和GCN网络。

**📊 数据集**

使用四个公开图数据集：CoraFull、Arxiv、Reddit、RomanEmpire。

**📈 对比分析**

与无持续学习基线Bare和全局联合训练Joint比较，发现所有方法在无任务边界下性能大幅下降，最佳方法仅达到35–60%准确率，低于Joint约20%。

**⚠️ 局限性**

限制在于仅评估节点分类、使用静态图合成流、仅考虑高斯漂移，未覆盖真实动态拓扑或更复杂漂移模式，方法仍受任务边界假设影响。

---

## 271. Frequency Bias and OOD Generalization in Neural Operators under a Variable-Coefficient Wave Equation

**arXiv ID:** 2605.12997 | [PDF](https://arxiv.org/pdf/2605.12997v1)

**作者:** Runlong Xie `[一作]` (Independent Researcher), An Luo `[通讯]` (University of Minnesota)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5025874786)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究神经算子在一维可变系数波方程终端状态预测下的分布偏移泛化表现，比较FNO与DeepONet两种主流算子架构。

**💡 创新点**

在结构化的OOD环境下系统评估并对比两种算子在频率与光滑度偏移下的泛化差异，并通过谱误差分析揭示架构偏置对泛化的根本影响。

**🔧 技术方法**

采用Fourier Neural Operator、Deep Operator Network以及频谱误差分析、相对L2误差评估等技术。

**📊 数据集**

使用数值求解器生成的可变系数波方程终端解样本，训练集含低频初始位移与光滑系数，OOD集分别加入高频位移或粗糙系数。

**📈 对比分析**

在ID、频率偏移、光滑度偏移三种测试下，用相对L2误差和谱误差对比，发现FNO在ID与光滑度偏移上误差低，但在高频OOD上急剧上升；DeepONet整体误差更高但下降更平滑。

**⚠️ 局限性**

仅评估两种传统算子架构，未考虑更大规模或混合架构，且OOD仅限频率与光滑度两类，缺乏对更复杂物理场景的验证。

---

## 272. F-GRPO: Factorized Group-Relative Policy Optimization for Unified Candidate Generation and Ranking

**arXiv ID:** 2605.12995 | [PDF](https://arxiv.org/pdf/2605.12995v1)

**作者:** Rohan Surana `[一作]` (UC San Diego), Julian McAuley `[通讯]` (UC San Diego)

**通讯引用:** 25546 | [OpenAlex ID](https://openalex.org/A5021827617)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在单一LLM中联合生成候选列表（slate）并在同一次自回归推断中对该列表进行排序，形成完整的列表-排序决策。

**💡 创新点**

提出Factorized Group‑Relative Policy Optimization (F‑GRPO)，通过将策略分解为候选生成和排序两阶段，并为每个阶段分别计算group‑relative advantage，实现跨阶段信用分配，消除传统GRPO中交叉梯度干扰，并避免解耦管道导致的分布偏移。

**🔧 技术方法**

采用自回归LLM（Qwen3‑4B/3.5‑2B）实现联合策略，使用DR‑GRPO基础并加入delimiter标记对rollout进行分段，采用分阶段最小化loss（slate loss、rank loss）与KL正则，奖励分别为覆盖性（Recall/F1）与位置感知（NDCG）。

**📊 数据集**

评估数据集包括顺序推荐（MovieLens、LastFM）和多跳问答（HotpotQA、MuSiQue），候选集规模为20或4-2条。

**📈 对比分析**

与传统分离式SFT、单阶段GRPO、零样本LLM reranker以及专用推荐/检索模型对比，F‑GRPO在Recall@k和NDCG@k上显著提升（如LastFM Recall@5提升+10.6%，MuSiQue Recall@3提升+13.2%），在覆盖受限的设置下优势最为突出，并且在不改动推理架构的前提下保持竞争力。

**⚠️ 局限性**

局限性包括：需要在训练阶段手动设计和调参的两阶段reward与lambda权重；在候选覆盖不是瓶颈的场景下收益有限；以及对大型模型的计算开销和训练稳定性仍有待进一步提升。

---

## 273. DP-Muon: Differentially Private Optimization via Matrix-Orthogonalized Momentum

**arXiv ID:** 2605.12994 | [PDF](https://arxiv.org/pdf/2605.12994v1)

**作者:** Jihwan Kim `[一作]` (Seoul National University), Chenglin Fan `[通讯]` (Seoul National University)

**通讯引用:** 184 | [OpenAlex ID](https://openalex.org/A5080242645)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DP-Muon 优化器和其改进版 DP-MuonBC，用于在差分隐私框架下对隐藏层矩阵参数进行私有训练，并给出了相应的隐私保证和收敛性分析。

**💡 创新点**

创新点包括：① 将矩阵梯度进行逐块裁剪并在同一批次下添加高斯噪声，借助联合 accounting 证明隐私不受 Muon 特有的后处理影响；② 对 Newton–Schulz 正交化步骤引入输出级别的热平滑偏差，提出基于反向探测的偏差校正 DP-MuonBC；③ 在理论上给出了可分离的收敛上界，清晰区分裁剪残差、隐私噪声、动量漂移和 Newton–Schulz 近似误差。

**🔧 技术方法**

使用技术包括：差分隐私的裁剪+噪声机制、同一批次联合 Gaussian 隐私计数器、动量更新、Newton–Schulz 迭代正交化、反向探测（antithetic）噪声校正、以及隐私后处理的理论证明。

**📊 数据集**

实验数据集为两大表到文本生成任务 E2E 和 DART，使用 GPT‑2 微调模型。

**📈 对比分析**

与 DP-SGD、DP-Adam 等基线比较，DP-Muon 在 NLL、BLEU、ROUGE‑L 上均有显著提升；DP-MuonBC 在 BLEU 进一步提升，且在 DART 上在三项指标均排名第一，隐私预算保持不变。

**⚠️ 局限性**

局限性：理论分析主要针对单块或冻结路径的情况，未覆盖多块全自适应训练；实验规模有限，未验证更大模型或更广泛任务；偏差校正依赖于 Newton–Schulz 的局部平滑性假设。

---

## 274. Leveraging Multimodal Self-Consistency Reasoning in Coding Motivational Interviewing for Alcohol Use Reduction

**arXiv ID:** 2605.12987 | [PDF](https://arxiv.org/pdf/2605.12987v1)

**作者:** Guangzeng Han `[一作]` (University of Memphis), Brian Borsari `[通讯]` (Veterans Affairs Health Care System)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种基于音频-语言模型的多模态自一致推理框架，用来自动对 Motivational Interviewing（MI）会议中的客户发言进行 CT、ST、FN 代码分类。

**💡 创新点**

创新点在于：①提出四个互补的提示策略（语义、声韵、证据评分、对比推理）让模型从不同视角分析同一语音；②使用自一致投票聚合多条推理路径，从而显著提升鲁棒性和准确性。

**🔧 技术方法**

采用 Qwen3‑Omni‑30B‑A3B‑Instruct 大语言模型与音频‑语言模型（ALM），结合 Whisper 语音识别进行对齐，并通过多提示与自一致投票实现多模态推理。

**📊 数据集**

使用了五个去标识化的大学生酒精使用 MI 录音会议，包含 371 条 Change Talk、135 条 Sustain Talk、392 条 Follow/Neutral，音频已通过 Whisper 对齐生成语音片段。

**📈 对比分析**

将 MM‑SC 与直接提示和链式思考（COT）基线进行对比，MM‑SC 在准确率 52.56% 和宏 F1 46.40% 上均优于基线；消融实验表明每个提示和音频输入对性能都有显著贡献。

**⚠️ 局限性**

局限性包括：样本量有限，仅来自大学生酒精 MI 会议；短句缺乏上下文，导致模型难以捕捉微妙的动机意图；未来需扩展数据规模、加入上下文信息和专业知识以进一步提升性能。

---

## 275. PRISM: Prior Rectification and Uncertainty-Aware Structure Modeling for Diffusion-Based Text Image Super-Resolution

**arXiv ID:** 2605.13027 | [PDF](https://arxiv.org/pdf/2605.13027v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 276. Relative Pose-Velocity Estimation Using Dual IMU Measurements and Relative Position Sensing

**arXiv ID:** 2605.13031 | [PDF](https://arxiv.org/pdf/2605.13031v1)

**作者:** Alessandro Melis `[一作]` (I3S-CNRS), Tarek Hamel `[通讯]` (Institut Universitaire de France)

**通讯引用:** 10420 | [OpenAlex ID](https://openalex.org/A5052963738)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文设计了一种双IMU系统下，利用相对位置或相对方位测量估计目标与自身相对姿态与线速度的方法；

**💡 创新点**

创新点在于将相对运动动力学嵌入15维线性时间变系统，构造确定性Riccati观测器并给出统一可观测性与持续激励条件，同时级联非线性互补滤波实现平滑姿态估计；

**🔧 技术方法**

主要技术包括Lie群建模、线性时变系统的可观测性分析、Riccati观测器设计、非线性互补滤波以及持续激励（PE）理论；

**📊 数据集**

实验采用仿真场景：多旋翼无人机在垂直波浪扰动下在移动船上空悬停，目标轨迹为圆形轨道加垂直摆动；

**📈 对比分析**

通过仿真验证，观测器在满足PE条件时实现全状态的指数收敛；在失去激励的阶段观测误差停止收敛，随后恢复激励后误差重新趋近零；未与其他算法进行数值对比；

**⚠️ 局限性**

局限性包括：假设IMU无偏差，需满足PE条件才能保证可观测性；对真实噪声与动态模型误差的鲁棒性尚未验证；仅在仿真中测试，缺乏真实数据验证。

---

## 277. FeatCal: Feature Calibration for Post-Merging Models

**arXiv ID:** 2605.13030 | [PDF](https://arxiv.org/pdf/2605.13030v1)

**作者:** Yanggan Gu `[一作]` (Hong Kong Polytechnic University), Hongxia Yang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 43640 | [OpenAlex ID](https://openalex.org/A5100378741)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于特征漂移分析的后合并校准方法（FeatCal），通过在前向层级上对每个线性模块做闭式更新，降低合并模型与任务专家间的特征差异；

**💡 创新点**

创新点在于：①从特征漂移角度拆解漂移为局部不匹配与上游传播；②利用该拆解设计前向层级的闭式校准，既不需要梯度下降、迭代或额外推理模块，又能保持与合并权重的紧密关联；

**🔧 技术方法**

技术包括：特征漂移理论、前向层级闭式岭回归更新、特征插值与锚点正则化、任务尺度归一化；

**📊 数据集**

评估数据集：CLIP 图像分类（8/14/20 任务）→CLIP‑ViT‑B/32 与 ViT‑L/14；FLAN‑T5 生成式 GLUE（8 任务）→FLAN‑T5‑base/large；MergeBench LLM（Llama‑3.2‑3B/8B）→多领域任务；

**📈 对比分析**

与 SOTA 合并后校准方法（Surgery、ProbSurgery）以及不同上游合并器（Task Arithmetic、Simple Averaging、AdaMerging、WUDI‑Merging）比较；在 CLIP‑ViT‑B/32 8 任务上平均精度从 66.3/67.5/82.7/84.5 提升至 83.2/85.5/88.1/88.8；FLAN‑T5‑base GLUE 平均提升 6.3 分；MergeBench LLM 任务平均提升 2.0/2.3 分；同时校准样本 256/例耗时仅 53 s，约为 Surgery/ProbSurgery 的 4×；

**⚠️ 局限性**

局限性：依赖任务专家特征，需预先获得专家模型；在特征漂移已被大幅降低的强合并器上提升有限；对极大模型或复杂架构（非线性模块）适用性待验证；

---

## 278. ImageAttributionBench: How Far Are We from Generalizable Attribution?

**arXiv ID:** 2605.12967 | [PDF](https://arxiv.org/pdf/2605.12967v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 279. ReCoG: Relational and Compact Context Graph Learning for Few-shot Molecular Property Prediction

**arXiv ID:** 2605.13024 | [PDF](https://arxiv.org/pdf/2605.13024v1)

**作者:** Zeyu Wang `[一作]` (Zhejiang University of Technology), Shirui Pan `[通讯]` (Griffith University)

**通讯引用:** 25273 | [OpenAlex ID](https://openalex.org/A5008056593)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了 ReCoG 框架，在少样本分子属性预测中通过学习关系紧凑上下文图来提升模型性能。

**💡 创新点**

创新点在于：①跨属性关系学习模块将隐式上下文转换为可优化的关系信号；②上下文图信息瓶颈模块自适应压缩无关辅助上下文；二者结合实现了对上下文图的关系与紧凑知识共同提取，并给出了理论支撑。

**🔧 技术方法**

使用技术包括预训练分子编码器、图神经网络、交叉属性关系学习（基于 MSE 的回归）、信息瓶颈（条件 MI + Gumbel‑Sigmoid 门控）以及 MAML 样式的双层优化。

**📊 数据集**

实验数据集为 MoleculeNet 上的 Tox21、SIDER、MUV、ToxCast 与 PCBA。

**📈 对比分析**

与 16 种基线方法（Siamese、ProtoNet、MAML、Pin‑Tuning、Uni‑Match 等）在 10‑shot 与 1‑shot 场景下进行 ROC‑AUC 对比，ReCoG 在所有数据集和 shot 设定中均显著优于基线，尤其在 MUV 与 1‑shot 任务中提升最大。

**⚠️ 局限性**

局限性包括训练时高方差、双层优化导致的计算开销与不稳定性，且在某些数据集上仍存在较高的性能波动。

---

## 280. Insecure Despite Proven Updated: Extracting the Root VCEK Seed on EPYC Milan via a Software-Only Attack

**arXiv ID:** 2605.12990 | [PDF](https://arxiv.org/pdf/2605.12990v1)

**作者:** Muyan Shen `[一作]` (University of Chinese Academy of Sciences), Yu Qin `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

本文通过两段软件攻击链：首先利用MilanLaunchy在EPYC Milan的AMD Secure Processor (ASP) 上实现任意代码执行；随后利用BadFuse攻击在已执行的旧固件环境下，通过熔丝写入与逐位提取的方式夺取硬件根种子（VCEK Root Seed），从而能伪造任何版本的远程证明。

**💡 创新点**

创新点在于①首次证明可仅凭软件在EPYC Milan上提取VCEK根种子，②揭示了Milan ASP固件加密签名失误导致的BootROM漏洞以及熔丝写入权限缺失，③提出MilanLaunchy与BadFuse两种新型攻击技术并演示完整链路。

**🔧 技术方法**

技术手段包括：固件逆向工程、AES-128-ECB/CBR解密与RSA‑PSS签名分析、BootROM加密解密链突破、ARM指令级跳转注入、MMIO熔丝烧录控制、位点写入与冷重启构造的熔丝oracle。

**📊 数据集**

实验使用真实硬件：AMD EPYC 7413（Milan）主板、TYAN S8036GM2NE、SPI Flash 与 BMC 接口，操作系统为 Ubuntu 22.04 Linux kernel 6.12；未使用机器学习或公开数据集。

**📈 对比分析**

实验对比显示：在未打补丁的旧固件上，MilanLaunchy + BadFuse 的成功率可达 100%，可完整提取 256 bit VCEK 种子；对比 AMD 官方补丁（Milan‑PI‑1.0.0.3）后，攻击仅能在最低 SVN 级别下工作，性能影响仅为固件重写与重启的时间开销。

**⚠️ 局限性**

局限性：仅适用于第一代 SEV‑SNP（EPYC Milan）并需完整宿主控制；对更新的 Milan 或其它架构的修补版无效；攻击依赖旧固件与熔丝写入权限，若硬件厂商关闭此权限或引入 ECC/校验，则失效。

---

## 281. Useful Memories Become Faulty When Continuously Updated by LLMs

**arXiv ID:** 2605.12978 | [PDF](https://arxiv.org/pdf/2605.12978v1)

**作者:** Dylan Zhang `[一作]` (University of Illinois Urbana Champaign), Hao Peng `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估LLM代理在连续更新文本记忆过程中出现的记忆退化，并验证保留原始经验与可选抽象的双层存储策略的有效性。

**💡 创新点**

首次系统性揭示连续记忆整合导致的三种失效模式，并通过自定义ARC‑AGI Stream实验验证，同时提出在保持原始轨迹与可选抽象之间分离的设计思路。

**🔧 技术方法**

利用CLIN、Agent Workflow Memory、Dynamic Cheatsheet、ACE等记忆压缩方法，配合GPT‑5.4、Qwen3.5等大型语言模型执行整合与推理，并实施分组、批量流式更新与一次性合并等三种记忆构造策略。

**📊 数据集**

在ALFWorld、ScienceWorld、WebShop、AppWorld、Mind2Web等标准基准以及自建的ARC‑AGI Stream（含已知任务族与ground‑truth）上进行实验。

**📈 对比分析**

对比抽象记忆、原始轨迹记忆、强制抽象、自动抽象与仅保留事件方案，结果显示原始轨迹记忆与仅保留事件在多数情形下与抽象记忆相当甚至更优；连续更新的抽象记忆往往在若干步后性能下降，甚至低于无记忆基线。

**⚠️ 局限性**

实验仅涉及文本记忆，未覆盖多模态或工具丰富场景；LLM抽象器与求解器均为同一大模型，随模型升级可能产生变化；实验样本量有限，未给出正式置信区间，结果受API成本限制。

---

## 282. No Attack Required: Semantic Fuzzing for Specification Violations in Agent Skills

**arXiv ID:** 2605.13044 | [PDF](https://arxiv.org/pdf/2605.13044v1)

**作者:** Ying Li `[一作]` (University of California), Yu Feng `[通讯]` (University of California)

**通讯引用:** 15488 | [OpenAlex ID](https://openalex.org/A5100398497)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对LLM驱动的代理技能中的自然语言安全规范进行动态语义模糊测试，自动发现规范违规行为。

**💡 创新点**

将自然语言guardrail转换为可判定的到达目标，结合LLM驱动的语义变异和Thompson采样bandit实现目标导向的模糊；首次系统性检测到无攻击情况下的规范违规。

**🔧 技术方法**

LLM驱动的种子生成与变异、注解执行轨迹的图抽象、语义注解谓词、到达目标检验、Thompson采样多臂bandit以及奖励函数（目标接近度+签名新颖度）。

**📊 数据集**

从OpenClaw公开技能市场抽取402个包含安全约束、状态修改和敏感资源的技能，覆盖六大业务域。

**📈 对比分析**

与传统静态分析、字节级模糊和LLM审计方法对比；通过消融实验评估每个模块贡献；最终发现29.9%技能存在违规，平均11分钟/技能完成模糊，前10分钟就发现44%违规。

**⚠️ 局限性**

依赖LLM的非确定性导致注解误差；谓词粒度有限可能产生误报或漏报；只能检测显式在规范中的guardrail，无法捕获隐式或缺失的约束；仅适用于自然语言规范的技能，未验证结构化API或形式化规范的情况。

---

## 283. EgoForce: Robust Online Egocentric Motion Reconstruction via Diffusion Forcing

**arXiv ID:** 2605.13041 | [PDF](https://arxiv.org/pdf/2605.13041v1)

**作者:** Inwoo Hwang `[一作]` (Seoul National University), Young Min Kim `[通讯]` (Seoul National University)

**通讯引用:** 29411 | [OpenAlex ID](https://openalex.org/A5100337311)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了EgoForce，一种在线、因果扩散模型，用于从稀疏噪声的自我视角观测重建全身运动。

**💡 创新点**

创新点在于将扩散模型迁移到实时因果推理，采用时间不对称噪声调度和噪声鲁棒插值，实现对长期运动的递进式去噪。

**🔧 技术方法**

使用扩散生成模型、Diffusion Forcing、跨注意力图像编码器、噪声鲁棒插值技术。

**📊 数据集**

主要使用EE4D‑Motion数据集，并在AMASS、Ego‑Exo4D等数据集上验证。

**📈 对比分析**

与在线基线（AvatarJLM、RPM、HMD^2）和离线扩散基线（UniEgoMotion、EgoAllo）相比，在MPJPE、FID、语义相似度、PJ、AUJ等指标上实现了更低误差、更高质量且推理时延更低的性能。

**⚠️ 局限性**

局限性包括对长时间历史长度敏感，过长历史会导致误差累积，且在极端噪声或视角遮挡严重的情形下仍可能产生漂移。

---

## 284. MAP: A Map-then-Act Paradigm for Long-Horizon Interactive Agent Reasoning

**arXiv ID:** 2605.13037 | [PDF](https://arxiv.org/pdf/2605.13037v1)

**作者:** Yuxin Liu `[一作]` (University of Science and Technology of China), Lei Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 108030 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `8d10c613-917e-4880-9716-17789f50e119` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Map-then-Act（MAP）框架，先通过跨任务全局探索构建通用先验，再进行任务特定认知地图构建，最后基于地图执行任务；并创建 MAP-2K 轨迹数据集用于 fine‑tuning。

**💡 创新点**

核心创新在于将环境认知与任务执行显式解耦：先主动探索并生成结构化地图，避免“act‑during‑think”模式的经验性错误；同时证明对地图轨迹进行监督比仅模仿专家执行更能提升通用能力。

**🔧 技术方法**

技术手段包括：跨任务全局探索与知识蒸馏、基于 RPP（Role‑Purpose‑Priority）的任务映射提示、双收敛停止判据与信息增益奖励、知识增强执行阶段、以及 MAP‑2K 训练管道的教师‑学生蒸馏。

**📊 数据集**

实验使用 ALFWorld、TextCraft、ScienceWorld、ARC‑AGI‑3 四大基准；并构造了 MAP‑2K 轨迹数据集（约 2K 条 map‑then‑act 轨迹）。

**📈 对比分析**

与 ReAct、CoMAP、SFT‑Execution（ACT‑4B）等基线在相同 token 预算下进行对比；结果显示 MAP 在所有任务与模型规模上均显著提升成功率（pass@1 与 level 分数提升 10–30%）、减少交互步数，MAP‑4B 甚至优于更大规模模型；在 ARC‑AGI‑3 上提升 22/25 个环境。

**⚠️ 局限性**

局限性：映射阶段增加前置探索开销，对小模型更为敏感；地图质量仍受探索预算限制；仅针对文本/文字环境，尚未验证在视觉或连续动作环境中的可扩展性；需要进一步研究在动态规则或更大规模环境中的适应性。

---

## 285. What Information Matters? Graph Out-of-Distribution Detection via Tri-Component Information Decomposition

**arXiv ID:** 2605.13032 | [PDF](https://arxiv.org/pdf/2605.13032v1)

**作者:** Danny Wang `[一作]` (University of Queensland), Zi Huang `[通讯]` (University of Queensland)

**通讯引用:** 13720 | [OpenAlex ID](https://openalex.org/A5078170935)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出Tide框架，利用三分信息分解（特征、结构、联合）实现图节点OOD检测。

**💡 创新点**

创新在于将信息瓶颈（IB）与三组件分解结合，显式抑制特征/结构单独的噪声，突出联合信息。

**🔧 技术方法**

使用图神经网络、信息瓶颈、互信息约束、能量分数等技术。

**📊 数据集**

在七个公开图数据集（Cora、Citeseer、Pubmed、Amazon、Coauthor、Twitch、Arxiv）上实验。

**📈 对比分析**

与多种SOTA基线（MSP、ODIN、GNNSafe、DeGEM等）对比，FPR95平均提升约34%，AUROC提升20%+，保持ID分类精度。

**⚠️ 局限性**

局限在于需额外的特征/结构网络与IB训练开销，且对极端分布偏移或低ID准确率的场景仍有挑战。

---

## 286. Skew Polycyclic Codes over $\frac{\mathbb{F}_{p^m}[u]}{\langle u^t \rangle}$

**arXiv ID:** 2605.13020 | [PDF](https://arxiv.org/pdf/2605.13020v1)

**作者:** Akanksha Tiwari `[一作]` (Indian Institute of Technology Delhi), Ritumoni Sarma `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 125 | [OpenAlex ID](https://openalex.org/A5035273961)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究了有限链环R^t[x,Θ]/⟨f(x)⟩中左理想的结构，特别是与中心元素f(x)相关的偏多循环码的结构。

**💡 创新点**

创新点在于提供了更精细的左理想形式，并补充了文献中遗漏的必要条件，以确保不同类左理想之间不重叠。

**🔧 技术方法**

使用了偏多项式环和环的自同构技术，特别是通过对R^t[x,Θ]的研究来分析左理想的结构。

**📊 数据集**

使用了有限链环𝔽_p^m[u]/⟨u^t⟩作为数据集，特别关注了特定形式的中心元素f(x)=x^np^s-λ。

**📈 对比分析**

通过与现有文献中的结果进行比较，展示了所提出方法的有效性，特别是在n=1和n=2的情况下，给出了完整的左理想描述，并计算了i-th扭转码。

**⚠️ 局限性**

限制在于尽管提供了更精细的结构描述，但仍然需要进一步研究以探索双重性、距离和相应码的枚举等问题。

---

## 287. CoRe-Gen: Robust Spectrum-to-Structure Generation under Imperfect Fingerprint Conditions

**arXiv ID:** 2605.12980 | [PDF](https://arxiv.org/pdf/2605.12980v1)

**作者:** Tianbo Liu `[一作]` (University of Hong Kong), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**通讯引用:** 34958 | [OpenAlex ID](https://openalex.org/A5102498323)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出 CoRe‑Gen 框架，实现从 MS/MS 光谱到分子结构的生成；

**💡 创新点**

核心创新在于以条件为中心的设计——先用合成光谱预训练谱解码器提升指纹预测，随后用频率感知的指纹噪声匹配 decoder 训练，并在自回归解码中加入分子结构约束；

**🔧 技术方法**

采用 MIST 谱编码器 + FP‑Growing 指纹头、频率感知指纹腐蚀、SELFIES 的组合式嵌入、辅助结构监督与规则化的 logit 掩码；

**📊 数据集**

数据集包括约 79 万条合成 MS/MS 语料、NPLIB1 与 MassSpecGym 两个真实评测集；

**📈 对比分析**

与 Spec2Mol、MIST+NeuralDecipher、MIST+MSNovelist、MADGEN、DiffMS、MS‑BART 等基线对比，CoRe‑Gen 在 NPLIB1 上 Top‑1/Top‑10 exact‑match 分别提升至 19.54%/29.92%，在 MassSpecGym 上保持竞争力并获得最佳 MCES；速度上每光谱 100 备选仅约 2 秒，比 DiffMS 低 80 倍；

**⚠️ 局限性**

主要限制在指纹预测的准确性仍是瓶颈，若使用真值指纹可达 82%/87% 的准确率，说明谱编码器与合成与真实光谱间的域差仍需进一步缓解。

---

## 288. Retrieval is Cheap, Show Me the Code: Executable Multi-Hop Reasoning for Retrieval-Augmented Generation

**arXiv ID:** 2605.12975 | [PDF](https://arxiv.org/pdf/2605.12975v1)

**作者:** Jiashuo Sun `[一作]` (University of Illinois Urbana Champaign), Jiawei Han `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将多跳检索增强生成 (RAG) 转化为可执行的 Python 程序，使用可解释的变量状态、编译器反馈和执行轨迹来实现链式检索与推理，并通过程序执行实现无训练的自我修复与自适应检索。

**💡 创新点**

创新点在于把多跳推理完整地视为程序合成与执行，提供可验证的执行接口、编译器级别的错误检测和基于执行结果的自适应检索，同时利用代码专用 LLM 的结构化推理能力。

**🔧 技术方法**

技术包括：代码专用 LLM (如 Qwen、LLaMA、Qwen3)、分解代理、规划代理、答案代理、Python 解释器执行、编译器错误回馈、执行驱动的自适应检索以及强化学习微调。

**📊 数据集**

使用的公开数据集有 PopQA、HotpotQA、2WikiMultihopQA、MuSiQue 和 Bamboogle。

**📈 对比分析**

与 Vanilla RAG、Self‑Ask、IRCoT、ITER‑RETGEN 等训练‑free 基线以及 Search‑R1、StepSearch、ReSearch 等 RL 基线进行对比；在训练‑free 方案下平均 EM 提升 11.8 点，Bamboogle 上提升 25.5 点；RL 版本在 7B 规模模型中获得最高或接近最高 EM，并在 Qwen3‑4B、LLaMA‑3.1‑8B 等后端上保持领先。

**⚠️ 局限性**

主要限制包括检索召回率仍是瓶颈（约 50% 失败来自检索缺失）、答案代理是错误主因、程序错误率低但对检索结果格式依赖较大，以及需在 Python 环境下执行，限制了部署与扩展。

---

## 289. Position: Agentic AI System Is a Foreseeable Pathway to AGI

**arXiv ID:** 2605.12966 | [PDF](https://arxiv.org/pdf/2605.12966v1)

**作者:** Junwei Liao `[一作]` (Shanghai Jiao Tong University), Weinan Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 18625 | [OpenAlex ID](https://openalex.org/A5090720315)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

本文提出并理论证明了 Agentic AI（多智能体协作系统）相较于传统单一大模型（monolithic scaling）在通用人工智能（AGI）方面具有更高的样本与参数效率，并给出了基于有向无环图（DAG）的通用 Agentic AI 定义与分析框架。

**💡 创新点**

创新点包括：
1) 将真实世界任务分布建模为低维流形集合，揭示 monolithic 模型陷入的“平均陷阱”（Average Trap）；
2) 证明仅凭路由（routing）的 Agentic AI 在样本/参数复杂度上实现指数级提升；
3) 将 Agentic AI 泛化到任意 DAG 结构，引入组合容量 C(G) 与边权重 W(e*) 作为设计与性能评估指标；
4) 系统性讨论路由误差与专业化平衡，给出最优智能体数 K* 的理论取值。

**🔧 技术方法**

技术手段：
- 统计学习理论与泛化界（Natarajan 维、最小化风险）
- 几何流形理论与低维近似
- 路由机制与分层专家模型
- DAG 结构分析、Jacobian 与梯度传播权重计算
- 经验与理论结合的样本/参数复杂度分析
- 讨论路由误差与边权重的上界

**📊 数据集**

未在论文中给出具体实验数据集，理论分析基于通用的流形分布假设与假设性任务集合。

**📈 对比分析**

比较方法：
- 与 monolithic 模型在样本规模 N 下的误差界（O(N^{-1/D)）
- 与路由式 Agentic AI 的误差界（O(K·N^{-1/d_max））
- 在参数规模 P 下的误差界（O(P^{-κ/D)）对比 O(P^{-κ/d_max））
- 结果显示，随着维度差 D−d_max 的增大，路由式 Agentic AI 的性能提升呈指数级。

**⚠️ 局限性**

限制与挑战：
- 路由误差与边权重随智能体数增大而上升，需要平衡 K 与路由成本；
- 需精确设计 DAG 结构与路由器，设计不当可能导致性能退化；
- 论文多为理论推导，缺乏大规模实验验证；
- 对实际数据分布的流形假设可能与真实任务不完全匹配。

---

## 290. Offline Two-Player Zero-Sum Markov Games with KL Regularization

**arXiv ID:** 2605.13025 | [PDF](https://arxiv.org/pdf/2605.13025v1)

**作者:** Claire Chen `[一作]` (California Institute of Technology), Nan Jiang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究离线两人零和马尔科夫游戏中的纳什均衡学习，提出 KL 正则化的理论框架 ROSE 与可计算的 SOS‑MD 算法，证明其无需显式悲观化即可在单调覆盖条件下实现 O(1/n) 的统计收敛速率。

**💡 创新点**

创新点：①首次证明 KL 正则化足以抵消离线分布漂移，完全免除传统的悲观化正则化；②在多步马尔科夫游戏中实现 O(1/n) 的快收敛；③提出 SOS‑MD 通过镜像下降自我博弈逼近 ROSE，实现可计算且优化误差随迭代数 T 以 O(1/√T) 降至零。

**🔧 技术方法**

技术方法包括：KL 正则化的贝尔曼递推；最小二乘值迭代估计 Q；对齐参考策略的 KL 损失；对 ROSE 的理论分析（利用单侧集中性与函数类的 D² 归一化）；SOS‑MD 的双重镜像下降更新与优化误差分析。

**📊 数据集**

使用通用离线数据集 𝒟（n 条轨迹），无特定公开数据集；数据由行为策略生成，假设满足独占集中性与函数类完整性。

**📈 对比分析**

相较于现有的 PNVI、PMVI、BCEL 等方法，ROSE/SOS‑MD 在相同的单侧覆盖假设下从 O(1/√n) 提升至 O(1/n) 的统计误差；实验与理论表明无显式悲观化仍能达到或超过对比方法的性能。

**⚠️ 局限性**

局限性：①依赖 KL 参考策略与单侧集中性假设；②需要函数类完整性与强凸性；③对大规模游戏的实际实现仍需大量自我博弈迭代；④未在真实公开数据集上验证，理论与实验的结合有限。

---

## 291. SECOND-Grasp: Semantic Contact-guided Dexterous Grasping

**arXiv ID:** 2605.13117 | [PDF](https://arxiv.org/pdf/2605.13117v1)

**作者:** Han Yi Shin `[一作]` (Korea University), Sangpil Kim `[通讯]` (Korea University)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5077788107)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了SECOND‑Grasp框架，结合视觉‑语言推理与多视角分割，生成语义化且几何一致的接触地图，再通过逆运动学得到伪目标位姿，指导多指手抓取策略；

**💡 创新点**

创新点在于首次将语义接触预测与几何一致性约束相结合，并利用接触地图生成的IK位姿作为策略学习的先验，从而实现语义与物理双重约束的抓取；

**🔧 技术方法**

核心技术包括预训练视觉‑语言模型推断可抓取部件、分割网络实现像素级定位、Semantic‑Geometric Consistency Refinement（SGCR）两阶段语义与几何一致性优化、逆运动学求解伪手姿、以及基于强化学习的策略蒸馏；

**📊 数据集**

主要使用DexGraspNet作为训练与评估数据集，并在Shadow Hand、Allegro Hand等不同机械手上进行迁移验证；

**📈 对比分析**

与DemoGrasp、DemoFunGrasp、UniDexGrasp等方法比较，SECOND‑Grasp在状态基与视觉基抓取成功率上分别达98.2%/97.7%和95.9%/94.9%，显著优于对手，且在意图一致性、抓取多样性和跨数据集泛化上表现更佳；

**⚠️ 局限性**

局限性包括目前仅在单手抓取场景验证，未评估完整的手臂‑手协同控制；此外，依赖预训练视觉‑语言模型与分割网络，对极端光照或复杂纹理的物体仍可能产生误判。

---

## 292. TouchAnything: A Dataset and Framework for Bimanual Tactile Estimation from Egocentric Video

**arXiv ID:** 2605.13083 | [PDF](https://arxiv.org/pdf/2605.13083v1)

**作者:** Jianyi Zhou `[一作]` (Harbin Institute of Technology), Shuo Yang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 59435 | [OpenAlex ID](https://openalex.org/A5038497484)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个大规模的多视角头戴和腕部摄像头同步采集的 egocentric 数据集 EgoTouch，并提出了基于多视角融合与视图随机丢弃的 vision-to-touch 基线模型 TouchAnything。

**💡 创新点**

创新点包括首次将腕部视角与头戴视角结合提供丰富的视觉与触觉同步信息，以及视图随机丢弃训练策略，使模型在任意视角组合下均能鲁棒推断触觉。

**🔧 技术方法**

技术涵盖多视角视觉编码、跨视角注意力融合、手部姿态交叉注意力、以及带加权 MSE/L1/TV 损失的触觉解码器，训练时采用视图随机丢弃。

**📊 数据集**

使用了 EgoTouch 数据集，包含208个任务、1,891个片段、20小时多视角视频、双手 3D 姿态以及 16×16 触觉压力网格。

**📈 对比分析**

与仅使用头戴视角的基线相比，加入一只或两只腕部视角可将 Contact IoU 从0.4792提升至0.5030、Volumetric IoU 从0.4311提升至0.4575（见整体表格），且在见过与未见过物体上均有明显提升。

**⚠️ 局限性**

局限性包括受限于触觉手套的外观偏差、数据规模尚未饱和且模型仍需更多样化场景、以及目前仅关注触觉估计未涉及更高级的交互推理。

---

## 293. Spectral Flattening Is All Muon Needs: How Orthogonalization Controls Learning Rate and Convergence

**arXiv ID:** 2605.13079 | [PDF](https://arxiv.org/pdf/2605.13079v1)

**作者:** Tien-Phat Nguyen `[一作]` (Hanoi University of Science and Technology), Trung Le `[通讯]` (Monash University)

**通讯引用:** 1858 | [OpenAlex ID](https://openalex.org/A5102780660)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文对 Muon 优化器的工作原理进行了理论分析，并通过实验验证其在学习率稳定性和收敛速度上的优势。

**💡 创新点**

创新点在于：① 将 Muon 的正交化步骤解释为谱扁平化，并推导出其最大可接受学习率由梯度最大奇异值改为平均奇异值决定；② 将 Muon 重新表述为预条件梯度方法，在 K-FAC 近似下证明其有效收敛因子优于传统梯度下降；③ 提出了基于谱上限的归一化设计原则（FrobNorm）并在实验中验证。

**🔧 技术方法**

主要技术包括：Newton–Schulz 迭代近似极点因子、矩阵 SVD 与极点因子、Kronecker-factored Hessian (K‑FAC) 近似、相对平滑性与 Polyak–Łojasiewicz 条件下的收敛分析、对比实验中的双优化器策略和归一化层设计。

**📊 数据集**

实验使用 CIFAR‑10 数据集，采用 CifarNet 结构，并在去除批归一化、采用固定学习率和线性学习率调度等设置下进行对比。

**📈 对比分析**

通过最大稳定学习率实验显示，Muon 在学习率较大时仍能保持收敛，而 SGD 在同一范围内发散；在相同学习率（0.05）下的收敛速度实验表明，Muon 的损失下降更快、验证准确率提前数个 epoch；使用 FrobNorm 归一化后，两者均可在更高学习率下稳定训练，Mu 进一步提升性能。

**⚠️ 局限性**

局限性包括：分析仅针对确定性全批量梯度和精确极点因子，未考虑随机梯度、有限 Newton–Schulz 步数、动量、学习率调度及实际大规模模型中的系统交互；因此对真实训练环境的适用性需进一步验证。

---

## 294. TruncProof: A Guardrail for LLM-based JSON Generation under Token-Length Constraints

**arXiv ID:** 2605.13076 | [PDF](https://arxiv.org/pdf/2605.13076v1)

**作者:** Yoshio Kato `[一作]` (NTT DOCOMO BUSINESS, Inc.), Shuhei Tarashima `[通讯]` (NTT DOCOMO BUSINESS, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 TruncProof，一种基于 LL(1) 语法约束的生成方法，能够让大语言模型在严格的 token 上限下输出完全符合 JSON 语法且语义合理的结构。

**💡 创新点**

创新点在于利用 LL(1) 解析器实时估算完成合法 JSON 所需的最小 token 数，并通过预计算的最短 token 长度与成本验证器，构造动态约束 mask，严格控制 token 预算，避免生成被截断或无限的情况。

**🔧 技术方法**

使用了 LL(1) 语法解析、DFA 终结符识别、Dijkstra 算法预计算最短 token 长度、成本验证器、logit 修正 mask，并与 Greedy、Beam Search、MCTS 等多种解码策略无缝集成。

**📊 数据集**

采用了公开的 JSON-Mode-Eval 数据集（100 条文本‑转‑JSON 任务）进行实验。

**📈 对比分析**

对比了无约束、Outlines、SynCode、XGrammar 等方法，并在 Greedy、Beam Search、MCTS 三种解码下评估。TruncProof 在语法正确率上始终 100%，在模式匹配和 Exact‑match 上显著优于对比方法（尤其是与 Beam/MCTS 结合时），平均每 token 生成时间仅略高（+3–5 ms）且总时延仅比基线低 0–4 % 以内。

**⚠️ 局限性**

主要局限：高级解码（Beam、MCTS）会导致 2–20 倍的生成延迟；约束会扭曲 LLM 的原始概率分布，难以在保持语法正确的同时完全保留模型的生成分布；目前仅支持可被 LL(1) 解析的子语法，尚未扩展到更复杂的语言或非 LL(1) 语法。

---

## 295. HarmoGS: Robust 3D Gaussian Splatting in the Wild via Conflict-Aware Gradient Harmonization

**arXiv ID:** 2605.13073 | [PDF](https://arxiv.org/pdf/2605.13073v1)

**作者:** Yulei Kang `[一作]`, Wei-Shi Zheng `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种冲突感知的 3D 高斯投影（3D Gaussian Splatting）框架，用于在野外场景中实现鲁棒的三维重建。

**💡 创新点**

创新点包括：①语义一致性引导的遮挡掩码，用像素级一致性得分自适应细化先验掩码；②双视角冲突感知梯度协调，将两视角梯度旋转为正交，消除负向干扰；③冲突感知的高斯密度化与修剪策略，指导可靠的高斯生长并去除持续冲突的原语。

**🔧 技术方法**

技术手段涵盖：3D Gaussian Splatting 表示；DINOv2+SAM 先验掩码；自监督一致性残差目标；正交梯度旋转（梯度协调）；EMA 结构冲突评估与不透明度衰减；双视角密度化统计与冲突加权视图梯度。

**📊 数据集**

使用了三大野外数据集：PhotoTourism、NeRF On-the-go 和 RobustNeRF，涵盖不同遮挡级别与光照变化。

**📈 对比分析**

与 NeRF、3DGS、WildGaussian、HybridGS、AsymGS 等前沿方法进行对比，使用 PSNR、SSIM、LPIPS 评价。实验显示本文方法在所有数据集上均显著提升 PSNR、SSIM 并降低 LPIPS，取得最新最优性能。

**⚠️ 局限性**

局限性：仍需手工设计掩码阈值与旋转参数，双视角训练计算成本略高；对极端遮挡或强光照变化的极限仍未彻底解决；模型对输入相机标定误差敏感。

---

## 296. FiTS: Interpretable Spiking Neurons via Frequency Selectivity and Temporal Shaping

**arXiv ID:** 2605.13071 | [PDF](https://arxiv.org/pdf/2605.13071v1)

**作者:** Jongmin Choi `[一作]` (Korea Advanced Institute of Science and Technology), Joon Son Chung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 10624 | [OpenAlex ID](https://openalex.org/A5038723822)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 FiTS 神经元，将时间计算分解为频率选择性（FS）和时间塑形（TS）。

**💡 创新点**

创新点在于将子阈值幅度响应的最大频率作为可学习的目标频率，并通过组延迟调制实现可学习的时间塑形。

**🔧 技术方法**

使用可导的 LIF 模型、所有通道滤波、lambda‑mixing 组延迟、半隐式欧拉离散化和 surrogate gradient 训练。

**📊 数据集**

在 SHD、SSC 和 GSC 三个听觉分类数据集上进行评估。

**📈 对比分析**

与传统 LIF、TC‑LIF、cAdLIF 等基线在无递归、无网络级延迟的前馈网络中比较，FiTS 在 SHD 达到 95.31% 最高测试精度，SSC 78.23%，GSC 94.48%，明显优于纯 LIF 并与递归/延迟基线竞争。

**⚠️ 局限性**

仅在听觉任务和前馈结构中验证，未探索与递归、网络级延迟或更复杂神经元结构的交互；TS 需要额外的内部计算，效率和硬件实现仍待改进。

---

## 297. BrainAnytime: Anatomy-Aware Cross-Modal Pretraining for Brain Image Analysis with Arbitrary Modality Availability

**arXiv ID:** 2605.13059 | [PDF](https://arxiv.org/pdf/2605.13059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 298. The Cost of Perfect English: Pragmatic Flattening and the Erasure of Authorial Voice in L2 Writing Supported by GenAI

**arXiv ID:** 2605.13055 | [PDF](https://arxiv.org/pdf/2605.13055v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 299. Not All Anquan Is the Same: A Terminological Proposal for Chinese Computer Science and Engineering

**arXiv ID:** 2605.13069 | [PDF](https://arxiv.org/pdf/2605.13069v1)

**作者:** Xingyu Zhao `[一作]` (Wuhan University), Xingyu Zhao `[通讯]` (Wuhan University)

**通讯引用:** 20595 | [OpenAlex ID](https://openalex.org/A5100379743)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出在中文计算机科学与工程写作中将“安全”区分为“安全性”（safety）与“安保性”（security），并给出双轨制写作规范和术语表，旨在消除安全与安保概念的重叠与混淆。

**💡 创新点**

创新点在于首次系统性地将安全（safety）与安保（security）分别译为“安全性/安全”与“安保性/安保”，并结合语言哲学与标准分类理论，提出可落地的术语治理方案。

**🔧 技术方法**

主要采用文献综述、标准梳理、概念分析与语言哲学（维特根斯坦、Bowker‑Star 等）来论证术语区分的必要性与可行性。

**📊 数据集**

未使用实验数据集，主要引用国际标准（ISO/IEC、IEC、NIST）、国内标准（GB/T）、法律条文及英国 AISI 改名案例等实例进行阐述。

**📈 对比分析**

该工作为理论性提案，未进行实验或量化比较；通过案例分析展示若不区分概念可能导致的风险评估与论证混淆。

**⚠️ 局限性**

局限性包括：实施成本高、现有法律/标准难以立即改写、行业接受度不确定、缺乏实证验证以及对不同子领域适用性的进一步评估待开展。

---

## 300. Multi-Depth Uniform Coverage Path Planning for Unmanned Surface Vehicle Surveying

**arXiv ID:** 2605.13123 | [PDF](https://arxiv.org/pdf/2605.13123v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 301. Bridging Domain Gaps with Target-Aligned Generation for Offline Reinforcement Learning

**arXiv ID:** 2605.13054 | [PDF](https://arxiv.org/pdf/2605.13054v1)

**作者:** Minung Kim `[一作]` (Ulsan National Institute of Science and Technology), Seungyul Han `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 120 | [OpenAlex ID](https://openalex.org/A5091657241)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出跨域离线强化学习框架 TCE，利用得分模型生成与目标一致的转移并控制源数据使用，扩展目标状态覆盖以减少分布偏差。

**💡 创新点**

创新点：① 理论分解源数据混合与生成覆盖的两类域差距；② 双分数生成模型两阶段流程（状态覆盖 + 目标一致转移生成）；③ 通过近邻/最优传输选择与 KL 正则实现安全的源数据混合。

**🔧 技术方法**

技术手段：SDE 基础的两阶段得分生成模型；逆动力学模型和奖励模型；IQL 与 KL 正则化策略；NN/OT 近似选择机制。

**📊 数据集**

数据集：MuJoCo 四个连续控制任务（HalfCheetah、Hopper、Walker2d、Ant）在形态、运动学、重力变迁下的源-目标 D4RL 数据；Adroit 操作任务的 ODRL benchmark。

**📈 对比分析**

比较方法：与 IQL*、DARA、BOSA、SRPO、IGDF、OTDF、MOBODY 等基线进行对比；在 36 个 MuJoCo 交叉任务中，TCE 取得最高平均分，尤其在大域差时 TCE(OG) 明显领先；在 Adroit 任务中 TCE(OG) 同样优于所有基线。

**⚠️ 局限性**

局限性：需额外训练得分网络，存在 λ_cov、λ_mix 两个超参数；计算开销略增；在极端不匹配（状态/动作空间差异大）下效果有限，尚未覆盖完全不重叠域。

---

## 302. Rigel3D: Rig-aware Latents for Animation-Ready 3D Asset Generation

**arXiv ID:** 2605.13129 | [PDF](https://arxiv.org/pdf/2605.13129v1)

**作者:** Nikitas Chatzis `[一作]` (Technical University Of Crete), Evangelos Kalogerakis `[通讯]` (Technical University Of Crete)

**通讯引用:** 10523 | [OpenAlex ID](https://openalex.org/A5003312344)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一个端到端的生成框架，能够从图像直接生成带骨架和皮肤权重的动画就绪3D资产。

**💡 创新点**

联合建模表面与骨架的稀疏结构潜在空间，并使用开放词汇关节标注实现无模板的动画映射。

**🔧 技术方法**

基于 TRELLIS 的 Structured Latent (SLat) 与自回归骨架解码器、注意力皮肤权重解码器，以及两阶段 Rectified‑Flow 生成器。

**📊 数据集**

主要使用 Anymate（约 225K 训练样本）和 ModelsResource（270 个测试样本）进行训练与评估。

**📈 对比分析**

相较于 Anymate、Puppeteer、AniGen 等基线，取得在骨架距离和皮肤误差指标（尤其 B2B 与 KL）上的显著提升，整体性能优于现有方法。

**⚠️ 局限性**

生成的骨架可能缺失关节、出现多余分支；皮肤权重在极端姿势下不自然；开放词汇标注在重复结构上易混淆；且模型受限于输入视角的清晰度。

---

## 303. MoCCA: A Movable Circle Probability of Collision Approximation

**arXiv ID:** 2605.13125 | [PDF](https://arxiv.org/pdf/2605.13125v1)

**作者:** Tobias Kern `[一作]` (Technische Hochschule Ingolstadt), Christian Birkner `[通讯]` (Technische Hochschule Ingolstadt)

**通讯引用:** 58 | [OpenAlex ID](https://openalex.org/A5056122578)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种名为 MoCCA 的算法，用单个可移动圆在两辆车的中心线上近似其形状，从而对碰撞概率（POC）进行快速分析，并给出了误差上界和安全距离。

**💡 创新点**

创新点在于：① 将圆位置动态放置于两车最接近点，使圆半径比传统单圆更小、保守性更低；② 推导出基于方位角方差的误差上界；③ 引入可调安全距离来补偿下估误差，同时保持计算效率。

**🔧 技术方法**

使用了多元高斯分布的解析 POC 计算、协方差矩阵的雅可比传播、特征分解实现去相关、数值积分以及 CasADi 的 Python API；并与 Monte Carlo（MCS）及传统单圆/多圆方法进行比较。

**📊 数据集**

采用仿真数据：两车尺寸均为 5 m 长 × 2.2 m 宽，协方差矩阵 Σ=diag(0.25 m², 0.25 m², 0.25 rad²)，共 10 k 次 Monte Carlo 采样；设置通过场景（Scenario A：通过；Scenario B：交叉）进行对比。

**📈 对比分析**

与 unicircle、multicircle 和 MCS 进行对比：MoCCA 的平均计算时间为 44.6 µs，接近 unicircle（36.8 µs）但显著快于 multicircle（593 µs）和 MCS（13.5 ms）。准确度方面，MoCCA 在距离 5.5 m 以内与 multicircle 相似，靠近时 POC 稍高；误差上限可通过安全距离控制在 49.6% 以下。

**⚠️ 局限性**

局限性：对碰撞概率存在下估风险，需要额外安全距离补偿；当两车平行时参考点选择不理想导致 POC 计算不稳；误差上界可能仍可进一步收紧；未在真实道路数据上验证。

---

## 304. Pyramid Forcing: Head-Aware Pyramid KV Cache Policy for High-Quality Long Video Generation

**arXiv ID:** 2605.13111 | [PDF](https://arxiv.org/pdf/2605.13111v1)

**作者:** Jiayu Chen `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**通讯引用:** 37707 | [OpenAlex ID](https://openalex.org/A5100641667)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为自回归视频扩散模型提出一种无训练的 KVCache 框架 Pyramid Forcing，利用注意力头的历史帧依赖性差异实现长视频生成质量提升。

**💡 创新点**

创新点在于：①对注意力头进行离线三模式分类（Anchor、Wave、Veil），②针对每类头制定差异化的 KVCache 策略（层叠式滑窗、周期采样、局部合并），③使用 ragged‑cache attention 处理不同头长度的缓存，③将三种技术组合成完整系统。

**🔧 技术方法**

使用离线头分类（sign‑rate 与 FFT 周期判定）、层级 KVCache 策略、ragged‑cache attention（FlashAttention‑varlen）以及动态 RoPE 等技术。

**📊 数据集**

在 VBench‑Long 视频评测数据集上使用 MovieGen 提示集进行 30s、60s 长视频生成评估。

**📈 对比分析**

与 Self Forcing、Causal Forcing、Deep Forcing、Rolling Forcing、LongLive 等基线相比，Pyramid Forcing 在 60s VBench‑Long 总分从 77.87 提升至 81.21，30s 从 80.57 提升至 82.25；在动态度量上也取得显著提升（如 Causal Forcing 动态度量从 57.03 提升至 86.39）。整体保持了与基线相近的推理速度和显存占用。

**⚠️ 局限性**

限制主要包括：①需要离线对注意力头进行分类，虽开销小但需额外步骤；②目前仅在自回归视频扩散模型上验证，未在其他类型生成模型中测试；③缓存策略的超参数对不同视频长度或内容可能需要微调。

---

## 305. RoSplat: Robust Feed-Forward Pixel-wise Gaussian Splatting for Varying Input Views and High-Resolution Rendering

**arXiv ID:** 2605.13093 | [PDF](https://arxiv.org/pdf/2605.13093v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 306. Security Incentivization: An Empirical Study of how Micropayments Impact Code Security

**arXiv ID:** 2605.13100 | [PDF](https://arxiv.org/pdf/2605.13100v1)

**作者:** Stefan Rass `[一作]`, Christoph Wedenig `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大学软件工程课程中，采用基于自动化静态安全扫描结果的团队级奖金激励机制，进行对照实验，评估其对安全缺陷密度的影响。

**💡 创新点**

创新之处在于将安全指标与奖励挂钩的团队激励模型、半自动化扫描与计量管道，并通过实验验证其可行性与效果。

**🔧 技术方法**

使用 Bearer、Detekt、mobsfscan 等 SAST 工具（输出 SARIF），结合 Git、Jenkins CI、Python 脚本统计缺陷并计算密度，统计分析采用 Beta 回归与 Poisson 回归。

**📊 数据集**

数据来源为 14 个学生团队（84 名本科生）在 Android 前后端项目中生成的代码与扫描结果，覆盖 3 个迭代周期（sprints）。

**📈 对比分析**

通过对照组（SonarQube 质量门槛奖金）与实验组（安全扫描奖励）进行比较，Beta 回归显示实验组安全缺陷密度显著降低（β = -0.396，p = 0.034），并在后端层面表现更佳。

**⚠️ 局限性**

局限包括：样本为学生、单一机构、短期课程、仅限 Android 游戏领域、仅以 SAST 计数衡量安全、奖励形式仅为学分，且团队规模差异可能影响结果。

---

## 307. Identification of Non-Transversal Bifurcations of Linkages

**arXiv ID:** 2605.13094 | [PDF](https://arxiv.org/pdf/2605.13094v1)

**作者:** Andreas Mueller `[一作]`, J. S. Dai `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本文提出了一种利用高阶局部运动分析和运动切线锥（kinematic tangent cone）来识别连杆系统中非正交（非横向）运动分支的分岔方法，并在此基础上给出了可实现的计算流程；

**💡 创新点**

创新点在于证明了构造性运动切线锥的定义中已包含区分不同运动分支所需的高阶信息，从而避免了传统仅基于一阶切线的局限，并首次提供了可直接用于判断分支接触阶数的算法；

**🔧 技术方法**

主要技术包括：基于李括号（screw products）的高阶运动约束递推，计算运动切线锥的多阶约束方程，符号与数值求解多元多项式系统，利用Mathematica实现自动化求解；

**📊 数据集**

采用的“数据集”是若干典型连杆模型：6‑bar 链接、单循环7R 链接、八曲线等；这些模型在论文中作为实验例子来验证方法的有效性；

**📈 对比分析**

方法的验证通过对上述连杆模型在特定奇异配置下求解高阶约束并比较不同阶的解集结构来实现；实验结果显示该方法能够正确识别分支是否非正交，且计算效率在符号/数值实现下均可在几秒内完成；

**⚠️ 局限性**

局限性包括：1) 需要预先给定完整的螺旋关节参数，且对极大系统维度的求解仍受多项式求解器的限制；2) 该方法主要针对平滑运动分支，无法处理包含非光滑或分支不连续的奇异点；3) 目前仅在二维或三维示例中验证，尚缺乏在复杂多关节机器人上的实证验证。

---

## 308. Local Inverse Geometry Can Be Amortized

**arXiv ID:** 2605.13068 | [PDF](https://arxiv.org/pdf/2605.13068v1)

**作者:** Aaditya L. Kachhadiya `[一作]` `[通讯]`, Aaditya L. Kachhadiya

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可学习的逆向算子 Deceptron，用以摊销非线性逆问题中的局部逆几何，并在此基础上设计 D-IPG 迭代求解器，将残差校正的测量空间提议映射回潜在空间；通过 Jacobian Composition Penalty (JCP) 对逆向雅可比的局部左逆性进行约束，使其在每次迭代中提供近似 Gauss–Newton 的方向。

**💡 创新点**

创新点在于：①将局部逆几何的学习与传统二阶方法分离，构造一次性可重用的逆向算子；②引入 JCP 这一基于雅可比组合的微分一致性损失，用以训练逆向雅可比逼近前向雅可比的伪逆；③证明 D-IPG 在局部伪逆一致时与阻尼 Gauss–Newton 一阶等价，并给出偏差上界；④在多维 PDE 逆问题上验证该方法在保持高成功率的同时显著降低求解成本。

**🔧 技术方法**

使用的技术包括：可微前向代理 f_W 与逆向映射 g_V 的联合学习；JCP 采用 Hutchinson 估计对雅可比组合误差进行正则化；Armijo 回退、松弛与投影保证收敛；自动微分实现雅可比向量乘（JVP）以避免显式雅可比；在训练阶段加入任务误差、重构、循环一致性损失；理论证明 D-IPG 与 Gauss–Newton 的关系及其偏差界限。

**📊 数据集**

实验数据集涵盖七个 PDE 逆问题，包括 1D、2D、3D 热方程初始条件恢复，Darcy 流、对流扩散、Allen–Cahn 反应扩散以及 Navier–Stokes 流体动力学的反演，总计 240 个实例。

**📈 对比分析**

与传统的一阶方法（L‑BFGS）、二阶方法（Gauss–Newton、Levenberg–Marquardt、LM）进行性能对比，评估指标为时间‑收敛曲线、基准占用、任务特定 RMSE 成功率。D‑IPG 在所有代表性问题上实现了最高的墙壁时钟、迭代次数和预算占用曲线，平均成功率 94.8%，且在主要基准上相较基线降至 77× 的求解成本。

**⚠️ 局限性**

局限性包括：①依赖前向代理的质量，若代理的局部线性化不可靠（如 Heat‑1D 失效）则逆向算子无效；②理论和实验均基于局部一致性，未提供全局收敛保证；③JCP 对不同指标的提升并非统一，需针对具体问题调优；④最适用于可摊销的多实例场景，一次性求解问题时训练成本可能超过收益。

---

## 309. A Standardized Re-evaluation of Conversational Recommender Systems on the ReDial Dataset

**arXiv ID:** 2605.13053 | [PDF](https://arxiv.org/pdf/2605.13053v1)

**作者:** Ivica Kostric `[一作]` (University of Stavanger), Krisztian Balog `[通讯]` (University of Stavanger)

**通讯引用:** 5646 | [OpenAlex ID](https://openalex.org/A5059926999)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对对话式推荐系统（CRS）在ReDial数据集上的复现与可复制性进行系统评估，统一实验条件，剔除重复推荐短板，比较不同体系结构与LLM主干的效果，并引入用户导向的对话效用指标。

**💡 创新点**

创新点在于：①构建统一的两维（体系结构与时序成熟度）分类法；②在同一主干模型下对不同CRS架构进行交叉对比，剔除LLM容量带来的偏差；③在标准化的去重设置下量化“重复捷径”对Recall@1的影响；④引入Success Rate和Reward‑per‑Dialogue‑Length两种用户中心的度量来评估对话效率。

**🔧 技术方法**

使用的技术包括Transformer‑based LLM（GPT‑2、DialoGPT、Mistral）、知识图谱推理（RGCN、ConceptNet）、多任务联合训练、参数高效微调（LoRA、Prompt Tuning）以及批量推理加速等。

**📊 数据集**

数据集为ReDial（人机对话式电影推荐数据集），并采用统一预处理、去重与外部电影数据库映射。

**📈 对比分析**

通过对七个代表性CRS模型（KBRD、KGSF、UniCRS、ECR、MESE、PECRS、ReFICR）在标准化且去重的评测集上进行比较。结果显示：在去重后Recall@1显著下降，最高模型ReFICR从0.049降至0.027；在不同LLM主干上，模型表现随主干规模提升而提升，但体系结构优势不一；在用户效用指标上，统一架构MESE在RDL上优于高Recall模型。

**⚠️ 局限性**

局限性包括：①仅聚焦ReDial，未验证其他对话式推荐基准的通用性；②对外部知识源的依赖导致可复现性受限；③评测仍基于离线数据，缺乏真实用户交互反馈；④对LLM容量的更深层次探讨仍有限。

---

## 310. RAG-Enhanced Large Language Models for Dynamic Content Expiration Prediction in Web Search

**arXiv ID:** 2605.13052 | [PDF](https://arxiv.org/pdf/2605.13052v1)

**作者:** Tingyu Chen `[一作]` (Baidu Inc.), Daiting Shi `[通讯]` (Baidu Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并在百度搜索中部署了一套基于大语言模型的查询感知动态内容失效预测框架，自动推断何时信息失效并通过二值失效信号指导排名。

**💡 创新点**

创新点：
1) 将失效判定从固定时间窗口转为查询语义驱动的“有效期边界”推理；
2) 采用时间信息提取、结构化提示和前后一致性验证（CoT + 反向推理）抑制模型幻觉；
3) 将复杂的语义判断压缩为轻量级二值信号，满足在线低延迟需求。

**🔧 技术方法**

技术：
- Retrieval-Augmented Generation（RAG）
- 时间信息抽取与语义匹配（Rel_K、Rel_T）
- 结构化提示 + 多步推理
- 前后一致性验证与权重融合
- 二值失效信号与深度排序模型集成

**📊 数据集**

数据集：
- 1M有效查询 + 10M候选文档（来自百度搜索日志，覆盖多领域），用于离线评估和在线A/B测试；
- 人工标注的失效与新鲜度样本用于模型训练与验证。

**📈 对比分析**

比较方法：与传统固定时间窗口的recency boost做对比，并与基线排名模型（Aurora‑Baseline）进行离线评估、在线A/B测试及人工评测。性能：
- 离线：满意度得分+0.52%，PNR提升5–7%，day_away@4/10降低2–3%；
- 在线：高新鲜度查询day_away@4下降12.81%，CTR提升0.41%，用户体验、保留率均正向提升；
- 人工评测：时间敏感场景G:B 6:1，长尾/冷需求 12:2。

**⚠️ 局限性**

限制：
- 对极端长文本或多来源冲突仍有幻觉风险；
- 依赖缓存/实时推理，超时或分布漂移时退回基线；
- 对非时间敏感查询的二值化可能导致少量误判；
- 目前主要验证于中文搜索，跨语言迁移需进一步研究。

---

## 311. MLGIB: Multi-Label Graph Information Bottleneck for Expressive and Robust Message Passing

**arXiv ID:** 2605.13126 | [PDF](https://arxiv.org/pdf/2605.13126v1)

**作者:** Chaokai Wu `[一作]`, Xiaofeng Zhang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究多标签图中的信息过度压缩问题，提出多标签图信息瓶颈（MLGIB）框架，通过构造Markov依赖空间和可微分消息传递，保持预测标签信息同时抑制噪声。

**💡 创新点**

将信息瓶颈原理推广到多标签图，设计AIB（关注路径冗余）和HIB（关注消息纯度）两个互补约束，并通过可变分上下界实现端到端优化。

**🔧 技术方法**

信息瓶颈、图信息瓶颈、变分推断、可微分重参数化、标签嵌入（Skip‑gram）、标签相关性学习、采样路径与消息、深度可微分GNN架构。

**📊 数据集**

DBLP、BlogCatalog、PCG、Delve（四个多标签数据集）以及PubMed（单标签对照）。

**📈 对比分析**

与GCN、GAT、GIB、SDRF、FOSR、BORF、CorGCN等基线在Macro‑AUC、Micro‑AUC、Ranking Loss、Hamming Loss、Macro‑AP、Micro‑AP、LRAP等指标下比较。MLGIB在所有数据集上均实现显著提升，尤其在Delve上Macro‑AUC≈0.99、Micro‑AUC≈0.99，远超其它方法。

**⚠️ 局限性**

对高维稀疏标签、标签不平衡场景的鲁棒性仍待验证；需手动调参β等超参数；在极大规模多标签图上的训练效率和内存开销未做系统评估。

---

## 312. Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models

**arXiv ID:** 2605.13119 | [PDF](https://arxiv.org/pdf/2605.13119v1)

**作者:** Zixing Lei `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 9043 | [OpenAlex ID](https://openalex.org/A5066373402)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将视觉语言动作（VLA）模型拆分为可调用的工具族，并设计双向消息接口与进度反馈，以实现高层规划器对长周期任务的分解与恢复。

**💡 创新点**

创新点包括①将VLA转化为可调用工具族并提供完整的调用与反馈接口；②提出Tool‑Aligned Post‑Training (TAPT) 方案，利用工具族残差适配器实现调用对齐；③通过进度预测实现低频重规划，提升执行可靠性。

**🔧 技术方法**

使用大规模视觉语言预训练模型（OpenVLA‑OFT、π_0.5）、工具族残差适配器（LoRA）、进度预测头、模仿学习与强化学习联合训练、TAPT 后训练以及高层规划器的命令/反馈双向交互。

**📊 数据集**

实验使用 LIBERO‑Long、RoboTwin、CALVIN 等长周期操控基准，并构造 LIBERO‑CF‑Long 进行调用保真度评估。

**📈 对比分析**

与传统 VLA 单一策略、VLA+直接规划等对比，实验显示在 LIBERO‑Long 成功率提升 4.8 点、RoboTwin 提升 23.1 点，调用保真度提升 15 点；参数增幅仅约 9%–10%，推理时间增长 7%–14%。

**⚠️ 局限性**

局限性在于仅在仿真环境中验证，未在真实机器人上测试；工具族接口和子任务标签仍需人工设计；长周期性能依赖高层规划器的准确性。

---

## 313. DiffusionHijack: Supply-Chain PRNG Backdoor Attack on Diffusion Models and Quantum Random Number Defense

**arXiv ID:** 2605.13115 | [PDF](https://arxiv.org/pdf/2605.13115v1)

**作者:** Ziyang You `[一作]` (Fujian University of Technology), Xuxing Lu `[通讯]` (Fujian University of Technology)

**通讯引用:** 1415 | [OpenAlex ID](https://openalex.org/A5030096591)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一种通过劫持扩散模型推理过程中的伪随机数生成器（PRNG）的供应链后门攻击（DiffusionHijack），并提供了基于量子随机数发生器（QRNG）的硬件级防御。

**💡 创新点**

创新点在于首次揭示PRNG作为扩散模型核心推理组件的安全漏洞，攻击无需修改模型权重或提示即可实现像素级控制；同时提出利用量子随机数源提供信息理论安全性的全新防御思路。

**🔧 技术方法**

使用的技术包括：Python 运行时函数替换实现 PRNG 劫持；量子随机数硬件（Silicon Extreme QRNG600）生成真正随机采样；DDIM/CFG 等标准扩散采样算法；SSIM 和 CLIP 评估指标。

**📊 数据集**

实验基于 Stable Diffusion v1.4、v1.5 与 SDXL，使用公开的模型权重和多种随机提示（10 个提示 × 10 次），无需特殊标注数据集。

**📈 对比分析**

对比方法：在攻击、基线和 QRNG 防御三种配置下计算 SSIM，攻击场景下 SSIM=1.00，基线约 0.18–0.40，QRNG 防御后 SSIM 与基线无显著差异；安全检查器绕过率 98–100%。性能表现显示攻击可完全复制目标图像，防御完全消除该能力。

**⚠️ 局限性**

局限性包括：需攻击者在供应链层获得代码注入权限，QRNG 硬件成本和部署复杂度；实验仅覆盖 Stable Diffusion 系列，其他生成模型未验证；提示无关攻击效果相对有限；系统层面随机源依赖厂商实现，存在信任假设。

---

## 314. What to Ignore, What to React: Visually Robust RL Fine-Tuning of VLA Models

**arXiv ID:** 2605.13105 | [PDF](https://arxiv.org/pdf/2605.13105v1)

**作者:** Yuanfang Peng `[一作]` (Hong Kong University of Science and Technology), Rui Wang `[通讯]` (Microsoft Research Asia)

**通讯引用:** 14056 | [OpenAlex ID](https://openalex.org/A5100431163)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PAIR‑VLA框架，用于在RL微调过程中通过配对视觉视图和辅助目标提升VLA模型的视觉鲁棒性。

**💡 创新点**

创新点在于构造任务保持与任务改变的配对视图，并在PPO训练中加入不变性与敏感性KL目标，实现行为层面的视觉偏移指导，避免仅靠观测多样性。

**🔧 技术方法**

采用PPO强化学习，加入KL散度不变性与敏感性辅助目标，利用图像分割+背景合成生成任务保持视图，目标姿态扰动生成任务改变视图，支持自回归(OpenVLA)和流匹配(π₀.₅)两种VLA架构。

**📊 数据集**

在ManiSkill3模拟器的pick‑and‑place任务上进行评估，使用多种视觉偏移（未见纹理、光照、目标姿态、视觉杂乱、相机视角）进行训练/测试分离。

**📈 对比分析**

与标准PPO进行对比，平均提升成功率约16.6%（π₀.₅）和9.1%（OpenVLA），并显著加速收敛（在ID情境下仅需约1/3训练步数即可达到90%成功率）。

**⚠️ 局限性**

局限性在于仅在仿真环境评估，配对视图生成依赖准确的分割掩码，尚未验证其在真实世界中的迁移效果。

---

## 315. Content Caching Methods in Named Data Networks

**arXiv ID:** 2605.13104 | [PDF](https://arxiv.org/pdf/2605.13104v1)

**作者:** Pankaj Chaudhary `[一作]` (Indian Institute of Technology Indore), Sameer G. Kulkarni `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 1109 | [OpenAlex ID](https://openalex.org/A5081037214)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对信息中心网络（ICN）/命名数据网络（NDN）中的缓存方法进行了系统综述，构建了完整的分类体系并总结了各方法的优缺点。

**💡 创新点**

创新点在于提出了多维度的缓存技术分类（如集中/分散、路径/非路径、协作/非协作、基于流行度等），系统梳理并评估了120余篇研究，同时阐明了评估指标和未来研究方向。

**🔧 技术方法**

采用的技术主要是文献综述、分类与比较分析，并基于已公开的模拟结果讨论了性能指标。

**📊 数据集**

主要使用文献中公开的模拟实验数据，未采集新数据集。

**📈 对比分析**

通过对比分析不同缓存策略的评估指标（如缓存命中率、延迟、跳数等），本文指出了各类方案在不同场景下的性能优势与不足。

**⚠️ 局限性**

主要局限在于缺乏统一的实测或大规模实验验证，对真实网络流量和动态内容请求模型的适应性讨论不足。

---

## 316. Margin-calibrated Classifier Guidance for Property-driven Synthesis Planning

**arXiv ID:** 2605.13101 | [PDF](https://arxiv.org/pdf/2605.13101v1)

**作者:** Najwa Laabid `[一作]` (Aalto University), Vikas Garg `[通讯]` (Aalto University)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5065774663)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出并实现了序列完成排序（SCR）方法，用于在自回归逆合成模型中对单步生成进行基于属性的 token 级引导。

**💡 创新点**

创新点在于阐明交叉熵训练无法满足 token 级区分度的根本原因，并通过对比增强和基于 margin 的排序损失构建了能显式保证分辨率的分类器。

**🔧 技术方法**

核心技术包括对 SMILES 自回归生成的 token 级分类器引导、对比增强采样、margin 排名损失、Beam 搜索与自适应指导尺度的组合。

**📊 数据集**

实验数据集主要是 USPTO‑190 及其步骤版 USPTO‑190‑Steps、USPTO‑50k，以及 Pistachio 可达/难度数据集。

**📈 对比分析**

与传统的模板无关与模板相关方法以及搜索级引导进行对比，SCR 在单步 steering 宽度从 0.64 提升至与模板方法相当，复合解率提升 4.7 倍，且解锁了 17.4% 的先前不可解目标。

**⚠️ 局限性**

局限性包括对前向模型和模板分类器的依赖、对基生成器的耦合性，以及未与成本、多目标或更深层搜索结合的实验。

---

## 317. Does language matter for spoken word classification? A multilingual generative meta-learning approach

**arXiv ID:** 2605.13084 | [PDF](https://arxiv.org/pdf/2605.13084v1)

**作者:** Batsirayi Mupamhi Ziki `[一作]` (Bytefuse), Ruan van der Merwe `[通讯]` (Bytefuse)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文在多语言语音关键词识别任务上，应用生成式元连续学习 (GeMCL) 训练并比较了单语、双语和四语模型。

**💡 创新点**

首次将 GeMCL 迁移到多语言少样本语音分类，探讨多语模型与单语模型在未见语言上的差异。

**🔧 技术方法**

采用了生成式元连续学习框架、Transformer 编码器和贝叶斯高斯分类器，并使用元学习的训练/测试两阶段。

**📊 数据集**

使用 MSWC（多语言语音单词语料库）中四种高资源语言（英语、德语、法语、加泰罗尼亚语）以及其他35种低资源语言进行评估。

**📈 对比分析**

通过 25-way-5-shot 任务、100 轮测试、bootstrap 置信区间对比，发现多语模型与单语模型差距不显著，且仅比双语模型高约1%；多语模型在所有语言上的平均精度与单语相近。

**⚠️ 局限性**

实验仅覆盖四种高资源语言，未验证其他元学习范式，且未充分区分语言多样性与数据量对性能的影响。

---

## 318. Counterfactual Reasoning for Causal Responsibility Attribution in Probabilistic Multi-Agent Systems

**arXiv ID:** 2605.13077 | [PDF](https://arxiv.org/pdf/2605.13077v1)

**作者:** Chunyan Mu `[一作]` (University of Aberdeen), Muhammad Najib `[通讯]` (Heriot Watt University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

本文在概率多智能体系统中引入后向反事实责任归属机制，利用Shapley值分配责任，并构建了带有责任算子的PATL逻辑实现模型检验和Nash均衡计算。

**💡 创新点**

创新点在于将计量化的Shapley责任分配与反事实因果推理结合到概率游戏中，并在此基础上提出支持责任、奖励与时序目标的统一逻辑与算法；同时证明模型检验保持在PSPACE且可计算Nash均衡。

**🔧 技术方法**

使用了并发随机游戏模型、Shapley值、PATL与rPATL逻辑、概率CTL、参数化策略编码、线性规划与值迭代等技术。

**📊 数据集**

论文为理论工作，未使用公开数据集；实验基于手工构造的示例模型（如两车碰撞例子）。

**📈 对比分析**

与传统的rPATL模型检验相比，增加责任算子后仍保持PSPACE复杂度；在示例模型上通过符号求解得到Nash均衡，未给出数值性能基准。

**⚠️ 局限性**

局限性包括仅适用于无记忆（记忆无关）策略；只考虑有限时间界定的属性；未处理多记忆策略、学习组件及规范化整合等更复杂场景。

---

## 319. The Power of Graph Doubling: Computing Ultrabubbles in a Bidirected Graph by Reducing to Weak Superbubbles

**arXiv ID:** 2605.13074 | [PDF](https://arxiv.org/pdf/2605.13074v1)

**作者:** Sebastian Schmidt `[一作]` (University of Helsinki), Alexandru I. Tomescu `[通讯]` (University of Helsinki)

**通讯引用:** 1371 | [OpenAlex ID](https://openalex.org/A5063897224)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种通过对有向图的“图翻倍”操作，将双向图中的ultrabubble问题转化为有向图中的弱superbubble问题，从而实现对双向图ultrabubble的线性时间枚举。

**💡 创新点**

创新点在于：①简化并统一了ultrabubble和弱superbubble的定义；②证明二者在翻倍图中等价；③基于此构造了首个线性时间的ultrabubble枚举算法，突破了以往需要多步构造或仅适用于特殊图的限制。

**🔧 技术方法**

主要技术：图翻倍（doubling）操作、弱superbubble的线性时间枚举算法（Gärtner等方法的改写）、理论证明（等价性、最小性、无环性）。

**📊 数据集**

实验与数据集：本文为理论工作，未使用真实数据集，仅在算法复杂度上给出 O(n+m) 的线性时间证明。

**📈 对比分析**

比较方法：对比以往的 O(n(m+n)) 或 O(Kn+n+m) 的ultrabubble枚举算法；性能提升体现在时间复杂度从多项式降至线性。

**⚠️ 局限性**

局限性：仅适用于ultrabubble的枚举；对其他双向图问题（如流、最小割等）是否能同样通过翻倍有效仍待研究；此外，算法依赖于已存在的弱superbubble枚举实现。

---

## 320. Object Manipulation of the Variable Topology Truss system

**arXiv ID:** 2605.13086 | [PDF](https://arxiv.org/pdf/2605.13086v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 321. Early Semantic Grounding in Image Editing Models for Zero-Shot Referring Image Segmentation

**arXiv ID:** 2605.13122 | [PDF](https://arxiv.org/pdf/2605.13122v1)

**作者:** Jingxuan He `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**通讯引用:** 22124 | [OpenAlex ID](https://openalex.org/A5001529504)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用预训练的指令式图像编辑模型实现零样本指代图像分割，构建了一个无训练的端到端框架；

**💡 创新点**

发现并利用去噪初期特征已具备前景-背景可分离的特性，将跨模态注意力生成粗定位，并通过语义特征原型实现精确分割；

**🔧 技术方法**

使用多模态扩散变压器（MM‑DiT）、跨注意力、affinity传播、特征语义原型与余弦相似度分类等技术；

**📊 数据集**

在 RefCOCO、RefCOCO+、RefCOCOg 三大基准数据集上进行实验；

**📈 对比分析**

与现有零样本 RIS 基线（如 RefAM、HybridGL 等）对比，本文方法在 oIoU/mIoU 上均取得领先，Step1X‑Edit 在三大数据集上分别达到 58.10/60.34、54.52/54.25 等最优分数，超越 RefAM 约 5‑7 分；

**⚠️ 局限性**

对包含方向词的 RefCOCO 表现提升有限；依赖大型生成模型，对算力与显存要求高；在多实例歧义场景下仍可能产生误分割。

---

## 322. Vividh-ASR: A Complexity-Tiered Benchmark and Optimization Dynamics for Robust Indic Speech Recognition

**arXiv ID:** 2605.13087 | [PDF](https://arxiv.org/pdf/2605.13087v1)

**作者:** Kush Juvekar `[一作]` (Adalat AI), Kumarmanas Nethil `[通讯]` (Adalat AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对低资源印度语言的语音识别，提出了基于音频复杂度分层的Vividh-ASR基准，并在Whisper模型上开展了学习率时序与课程顺序的系统研究。

**💡 创新点**

创新点在于发现早期大更新加从难到易课程能显著减轻studio‑bias，并提出逆向多阶段微调（R‑MFT）实现参数效率高的模型；同时通过CKA与SVD揭示了解码器专属适配与编码器保持不变的机制。

**🔧 技术方法**

使用Whisper多语言预训练模型，采用分阶段学习率衰减、高LR早期训练、逆向课程、CKA、SVD等技术进行内部表示分析。

**📊 数据集**

数据集包括来自Kathbath、Shrutilipi、Indic Voices、FLEURS等的印地语与马拉雅拉姆语音，按Studio、Broadcast、Spontaneous、Synthetic Noise四层分层。

**📈 对比分析**

与单阶段低LR、单阶段高LR、标准易→难多阶段训练等对照实验相比，R‑MFT在Hindi约18.8% WER、Malayalam约39.4% WER，且244M Whisper在多层级基准上与769M模型匹配甚至超越，展现了显著的参数效率。

**⚠️ 局限性**

局限性包括仅评估两种语言与Whisper体系，未验证在其他自监督或Conformer模型上的推广；对数据规模、语种多样性和长语音的泛化性尚未深入探索。

---

## 323. PRA-PoE: Robust Alzheimer's Diagnosis with Arbitrary Missing Modalities

**arXiv ID:** 2605.13081 | [PDF](https://arxiv.org/pdf/2605.13081v1)

**作者:** Guangqian Yang `[一作]` (Hong Kong Polytechnic University), Shujun Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 6553 | [OpenAlex ID](https://openalex.org/A5100602073)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了PRA-PoE框架，用于阿尔茨海默病多模态分类，能在任意模态缺失的情况下保持鲁棒性。

**💡 创新点**

创新点在于引入Prototype-anchored Representation Alignment (PRA)来对齐不同缺失模式下的表示，并使用Uncertainty-aware Product-of-Experts (UA-PoE)融合，自动下调不确定的补齐特征。

**🔧 技术方法**

采用了可学习的全局原型与可用性嵌入的多头交叉注意力、Gaussian PoE融合、Monte Carlo采样、KL正则化等技术。

**📊 数据集**

实验使用ADNI和OASIS-3两个大型阿尔茨海默病数据集。

**📈 对比分析**

与mmFormer、ShareSpec、M3Care、AnyMod、FuseMoE、Flex-MoE、MoE-retriever等方法对比，PRA-PoE在所有非空模态组合上平均准确率最高，ADNI上相对提升约5.4%，OASIS-3上提升约10.9%。

**⚠️ 局限性**

局限在于仅在跨模态缺失模拟下评估，缺乏多中心外部验证，对极端稀有缺失模式的鲁棒性仍待验证。

---

## 324. When Absolute State Fails: Evaluating Proprioceptive Encodings for Robust Manipulation

**arXiv ID:** 2605.13067 | [PDF](https://arxiv.org/pdf/2605.13067v1)

**作者:** Maxime Alvarez `[一作]` (TELEXISTENCE Inc), Genki Sano `[通讯]` (TELEXISTENCE Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了移动机器人在不同起始位置下的原始状态编码方式，提出并评估了绝对、相对与无状态三种编码方案。

**💡 创新点**

创新点在于提出一种基于episode-wise相对坐标框架的简易编码方法，显著提升模型在分布内外的鲁棒性与性能。

**🔧 技术方法**

使用了Action Chunking Transformer（ACT）架构，并配合DINOv2-small视觉编码器。

**📊 数据集**

利用从真人操纵收集的2500个轨迹（约70小时）数据集，包含3张摄像头图像和10维机器人状态。

**📈 对比分析**

通过在真实机器人上进行分布内外评估，episode-wise相对编码平均分达3.12，成功率37.5%，显著优于绝对编码（0.38分、2.5%）和无状态编码（2.41分、20%）。

**⚠️ 局限性**

局限性包括对分布外情况仍有一定性能下降，以及方法对非线性关节的适用性尚未验证。

---

## 325. Ergodic Trajectory Design by Learned Pushforward Maps: Provable Coverage via Conditional Flow Matching

**arXiv ID:** 2605.13063 | [PDF](https://arxiv.org/pdf/2605.13063v1)

**作者:** Ehsan Aghazadeh `[一作]` (University of Massachusetts Amherst), Hossein Pishro-Nik `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 2453 | [OpenAlex ID](https://openalex.org/A5009685038)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种推前框架，将结构化的可解析轨迹与通过OT-条件流匹配学习的可重用映射相组合，实现连续轨迹的时间均值占用与目标空间密度一致的ergodic覆盖。

**💡 创新点**

创新点在于：1）将ergodicity与密度匹配解耦，使用单一可学习的推前映射即可适配任意目标与约束；2）提供能量、收敛速率（O(1/√K)）和逼近误差的理论上界；3）支持软约束的可微惩罚，且不需重新推导；4）实现多机队的可重用与无在线优化。

**🔧 技术方法**

使用的技术包括：解析的辐射半径往返轨迹、OT-CFM（Optimal‑Transport Conditional Flow Matching）学习速度场、谱归一化以实现Lipschitz约束、梯度惩罚（加速、禁飞区）以及理论证明（加速能量、Poincaré不等式、Grönwall稳定性）。

**📊 数据集**

实验数据集包含三种合成目标（高斯混合、二值分布、含禁飞区的高斯混合）以及真实的米兰UAV覆盖数据集（Milano UAV coverage dataset）。

**📈 对比分析**

与时间扭曲、优化与并行流匹配等基线方法比较，本文方法在覆盖-能量-约束三维Pareto曲线上取得最佳性能；在米兰数据集上，OT‑CFM+NFZ实现最高覆盖率与最低禁飞区违规率，OT‑CFM+E在能量消耗上显著优于对照组，整体收敛速率与理论一致。

**⚠️ 局限性**

局限性包括：1）使用环形潜在域导致O(δ)的拓扑残差；2）软约束惩罚无法提供严谨的约束保证；3）目前仅在二维开放式仿真验证，未扩展至三维、硬实时硬件或多机队交互；4）对映射Lipschitz常数的估计依赖于网络架构，实际实现可能超出理论上限。

---

## 326. Edit-Compass & EditReward-Compass: A Unified Benchmark for Image Editing and Reward Modeling

**arXiv ID:** 2605.13062 | [PDF](https://arxiv.org/pdf/2605.13062v1)

**作者:** Xuehai Bai `[一作]` (Hangzhou Dianzi University), Yuanxing Zhang `[通讯]` (Kling Team)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个统一的评测套件，包含了面向图像编辑模型的 ImgEdit-Bench（2,388 个多维度任务实例）以及面向奖励模型的 EditReward-Bench（2,251 个真实偏好对），并给出了基于 MLLM 判定的细粒度评估流程。

**💡 创新点**

创新点主要包括：① 在六大类别（常规编辑、动态操控、世界知识推理、算法视觉推理、多图像编辑、复杂任务）下构建 36 个分层难度任务；② 采用结构化链式推理与分维评分表实现与人类判断高度一致的细粒度评估；③ 通过 FlowGRPO 模拟的采样策略生成与 RL 场景相符的偏好对；④ 结合双语（中英）指令，评估跨语言能力。

**🔧 技术方法**

技术手段包括：多模态大模型（如 Qwen、Gemma）作为判定者；链式推理模板配合评分 rubric；基于偏好对的奖励模型训练与评估；FlowGRPO-inspired 采样与噪声调节；对比实验使用 29 个编辑模型与 21 个奖励模型。

**📊 数据集**

使用的数据集：ImgEdit-Bench（2,388 个实例，涵盖 36 个任务，提供中英双语指令与原始图像）；EditReward-Bench（2,251 个偏好对，包含指令、两张候选编辑结果）；源图像来自公开资源、专家场景描述与程序生成。

**📈 对比分析**

评估方法：在 3 个维度（Instruction Awareness、Visual Consistency、Visual Quality）上给出分数并计算平均。实验结果显示：闭源模型（如 Nano Banana Pro）整体得分最高（≈3.99），开源模型最高得分仅 2.69；在世界知识推理、算法视觉推理、多图像编辑等高难度子任务上开源模型仍显弱势；本地多模态 LLM（如 Qwen3.5、Qwen3.6）在奖励模型评估上明显优于专门训练的奖励模型。评估与人工偏好高度相关，证明评测框架可靠。

**⚠️ 局限性**

局限性：评估依赖外部 MLLM API，模型版本更新可能导致评分不稳定；判定者能力与版本差异影响可重复性；目前尚未构建专用的本地评判模型，需进一步提升评估的可访问性与透明度。

---

## 327. MUJICA: Multi-skill Unified Joint Integration of Control Architecture for Wheeled-Legged Robots

**arXiv ID:** 2605.13058 | [PDF](https://arxiv.org/pdf/2605.13058v1)

**作者:** Yuqi Li `[一作]` (Fudan University), Lihua Zhang `[通讯]` (Fudan University)

**通讯引用:** 33273 | [OpenAlex ID](https://openalex.org/A5100414909)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究并实现了一个统一的多技能控制框架 MUJICA，针对有轮腿混合机器人完成全向移动、高平台攀爬和跌倒恢复三种复杂动作，并通过本体感知实现了自动技能切换。

**💡 创新点**

创新点包括：① 在单一盲目策略中联合训练多种低级技能；② 引入硬 DC 电机约束，显著提升安全性与零样本 sim‑to‑real 转移；③ 通过高层技能选择器实现自主切换；④ 设计状态估计网络预测地形、碰撞概率等信息；⑤ 将多任务奖励与约束结合至 P3O 强化学习框架。

**🔧 技术方法**

技术栈：强化学习（PPO+P3O）、异步演员‑评论家、GRU 状态估计、SwAV 对比学习、DC 电机约束建模、域随机化与课程学习、IsaacLab 仿真、Unitree Go2‑W 硬件实现。

**📊 数据集**

实验主要在 IsaacLab 生成的多任务地形（楼梯、斜坡、离散地形、粗糙地形、坑洞）上进行，随后在真实 Unitree Go2‑W 机器人上验证；未使用公开数据集，而是通过随机化和自定义仿真环境产生数据。

**📈 对比分析**

与 DreamWaQ+P3O、PPO 基线以及各消融版本进行对比；在 10 个难度级别下的成功率评估显示 MUJICA 在大多数任务上明显优于基线，尤其在高难度任务中提升显著；DC 电机约束降低扭矩违规率至 3.5% 以下；技能选择器使多任务顺序完成率提升至约 95%。

**⚠️ 局限性**

局限性：目前仅覆盖三种技能，需手工设计奖励与指标；未覆盖更广泛的技能库与视觉/外部感知；在极端高难度地形或长时间连续任务中性能尚待验证；算法对硬件电机参数的依赖较强；未评估能耗与长时间运行的稳定性。

---

## 328. Context Training with Active Information Seeking

**arXiv ID:** 2605.13050 | [PDF](https://arxiv.org/pdf/2605.13050v1)

**作者:** Zeyu Huang `[一作]` (University Of Edinburgh), Marc'Aurelio Ranzato `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在不更新模型权重的前提下，提出一种基于 Beam Search 的上下文优化框架，利用 LLM 作为优化器主动检索外部信息（如 Wikipedia 与浏览器工具）并迭代更新结构化知识库，以提升模型在低资源翻译、医疗问答和推理任务中的表现。

**💡 创新点**

创新点在于：① 将信息检索工具嵌入上下文优化器，使其从外部获取缺失知识；② 采用 Beam Search 维持多条上下文候选，避免“上下文污染”和“局部最优”问题；③ 通过结构化数据库实现精准增删改，兼顾数据效率与可迁移性。

**🔧 技术方法**

核心技术包括：大规模语言模型（Gemini‑2.5‑Flash/3‑Flash）作为推理和优化器；工具调用（WikipediaSearchTool、BrowserUseTool）实现主动搜索；Beam Search‑style 训练过程（候选集合、分支扩展、验证剪枝）；结构化数据库与版本控制实现上下文管理；奖励函数与反馈机制实现自监督学习。

**📊 数据集**

使用的数据集包括：FLORES+（多语言低资源机器翻译）、HealthBench（医疗对话）、LiveCodeBench（代码生成）、Humanity's Last Exam（多学科推理）。

**📈 对比分析**

与基线（无优化、Best‑of‑N、顺序优化）和顺序+信息检索（Seq‑IS）比较，BeamSearch‑IS 在所有任务上均取得显著提升（低资源翻译平均 4.1 分提升；HealthBench 得分从 0.4629 提升到 0.5026；LiveCodeBench pass@1/8 与 HLE 平均准确率均提升 1–3%）。此外，BeamSearch‑IS 在少样本（32‑样本）下表现出更高的数据效率，并且优化后的上下文可迁移到更强模型（Gemini‑3‑Flash）并进一步提升性能。

**⚠️ 局限性**

主要局限：① 上下文利用受限于基模型的推理能力，导致检索到的知识在执行阶段未得到充分利用；② 构建的上下文稀疏且高度实例化，缺乏普适性，难以覆盖测试集多样性，限制了跨任务与跨模型的泛化能力。

---

## 329. Scaling few-shot spoken word classification with generative meta-continual learning

**arXiv ID:** 2605.13075 | [PDF](https://arxiv.org/pdf/2605.13075v1)

**作者:** Louise Beyers `[一作]` (Bytefuse), Ruan van der Merwe `[通讯]` (Bytefuse)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在多达1000个类、仅5个训练样本的少样本语音词分类任务中，使用生成式元持续学习算法GeMCL进行顺序学习，并与基于HuBERT的两种基线（全微调与冻结骨干+分类头）进行对比评估。

**💡 创新点**

创新点在于：①首次将GeMCL应用于语音词分类，证明其能在大规模类数和少样本情境下实现稳定的持续学习；②展示GeMCL在准确率与训练/适配速度上优于HuBERT基线，且显著减少了灾难性遗忘。

**🔧 技术方法**

技术方案包括：生成式元持续学习（GeMCL）利用高斯混合模型与贝叶斯先验闭式更新；Transformer编码器（12层12头）对MFCC特征进行嵌入；HuBERT基线采用全微调或冻结+分类头两种方式。

**📊 数据集**

数据集为Multilingual Spoken Words Corpus（MSWC），共12736个英文词，采用70/30划分为元训练与元测试集，且每类仅使用5个训练/验证/测试样本。

**📈 对比分析**

评估方法：逐步加入25个类直至1000个类，给每个类5个支持样本后在5个查询样本上测算准确率；GeMCL在大多数阶段与HuBERT冻结+分类头相当或更好，且准确率波动（per‑word volatility）显著低于HuBERT全微调；训练与适配时间方面，GeMCL仅使用约477小时标注数据，完成时间比HuBERT基线低约两倍以上。

**⚠️ 局限性**

限制与不足：仅在英文语料上验证，缺乏低资源或多语种场景的评估；基线HuBERT的预训练数据分布与测试不匹配；模型参数量较大（≈85M），对资源受限设备的部署有挑战；未探讨多说话人、噪声鲁棒性等实际部署因素。

---

## 330. Context Matters: Auditing Gender Bias in T2I Generation through Risk-Tiered Use-Case Profiles

**arXiv ID:** 2605.13113 | [PDF](https://arxiv.org/pdf/2605.13113v1)

**作者:** Jose Luna `[一作]` (Singapore Management University), Noa Garcia `[通讯]` (University of Osaka)

**通讯引用:** 792 | [OpenAlex ID](https://openalex.org/A5028370193)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于风险层级的文本到图像生成模型性别偏见审计框架，并设计了 THUMB 卡用于系统化记录与报告。

**💡 创新点**

将欧盟 AI 法案的风险分类与性别偏见度量、危害类型相融合，构建了统一的度量目录、危害类型表，并通过 THUMB 卡实现审计过程的可复用标准化。

**🔧 技术方法**

主要采用文献综述法汇总现有性别偏见度量（性别预测、嵌入相似度、下游任务），并基于风险层级定义用例、危害与对应指标，形成审计流程。

**📊 数据集**

使用公开的 T2I 生成图像及对应文本提示集合（如 DALL‑E、Stable Diffusion 等模型的生成结果），并未提出新的数据集；数据来源为公开的图像和提示集合。

**📈 对比分析**

本工作为框架与方法的整理与汇总，并未进行实验对比；没有给出量化性能指标，主要提供度量目录与示例卡片，供后续实证评估使用。

**⚠️ 局限性**

局限性：框架仅为理论/合成性，缺乏端到端的实证验证；只聚焦性别偏见，未扩展到其他受保护属性或交叉维度；THUMB 卡的实际有效性与适用性仍需后续用户研究与案例验证。

---

## 331. A Multi-Agent Orchestration Framework for Venture Capital Due Diligence

**arXiv ID:** 2605.13110 | [PDF](https://arxiv.org/pdf/2605.13110v1)

**作者:** Grigorios Alexandrou `[一作]` (Athens University of Economics and Business), Katerina Pramatari `[通讯]` (Athens University of Economics and Business)

**通讯引用:** 2502 | [OpenAlex ID](https://openalex.org/A5019363248)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个基于多智能体的端到端自动化平台，用于风险资本的企业尽职调查与市场分析，集成实时检索、逆向抓取希腊商业登记处官方财务文件、布局感知OCR解析，并自动生成结构化 HTML 报告。

**💡 创新点**

创新点包括：1）事件驱动的多智能体管道实现从触发到报告的完整自动化；2）对希腊商业登记处前端到后端通信进行逆向工程，获取官方 PDF 财务报表；3）引入结构化回退机制，显式标注数据缺失以避免 LLM 幻觉。

**🔧 技术方法**

采用了 n8n 低代码事件驱动平台、AutoGen/MetaGPT/LangGraph 等多智能体框架、Perplexity Sonar 实时检索 API、布局感知 OCR（Marker）、LLM（OpenAI/Perplexity 等）及工具调用与条件路由。

**📊 数据集**

使用的数据集包括希腊商业登记处（Γ.E.MH）PDF 财务报表、Crunchbase/Dealroom 等商业数据库、Perplexity Sonar API 提供的实时网页数据以及预先抓取的公司属性 JSON 数据库。

**📈 对比分析**

与人工手工研究对比，报告在数据完整性、可追溯性和缺失信息显式化方面显著提升；实验未给出具体数值指标，但作者指出财务数据无幻觉率高，且缺失标注减少决策误差。

**⚠️ 局限性**

主要限制：1）仅能获取希腊注册公司的官方财务，海外公司依赖第三方数据库；2）依赖外部商业 API（Perplexity、LLM、OCR），易受价格/限流影响；3）LLM 推理不确定性导致结果可重复性差；4）系统仅在单一基金环境测试，跨行业、跨国通用性待验证。

---

## 332. Flow Augmentation and Knowledge Distillation for Lightweight Face Presentation Attack Detection

**arXiv ID:** 2605.13108 | [PDF](https://arxiv.org/pdf/2605.13108v1)

**作者:** Muhammad Shahid Jabbar `[一作]` (King Fahd University of Petroleum and Minerals), Shujaat Khan `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 1427 | [OpenAlex ID](https://openalex.org/A5026671975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种教师-学生框架，在训练阶段使用光流增强学习微动信息的FacePAD模型，并在推理时通过知识蒸馏实现仅使用RGB的轻量化部署。

**💡 创新点**

创新点在于：①双分支教师融合RGB与色轮编码光流以捕捉细粒度运动；②logit蒸馏将运动知识迁移至单分支学生，消除推理时光流开销；③在多种公开数据集上实现与教师持平甚至更优的性能，同时显著压缩模型。

**🔧 技术方法**

使用技术包括UniMatch光流估计、色轮编码、MobileNetV3-Large双分支网络、同步数据增强、logit蒸馏、并在Jetson Orin Nano上实现实时推理。

**📊 数据集**

使用数据集：Replay-Attack、Replay-Mobile、ROSE-Youtu、SiW-Mv2、OULU-NPU。

**📈 对比分析**

与最新方法比较，教师模型在Replay系列达到0% HTER，学生模型在ROSE-Youtu实现0.81% HTER、0.99% AUC，OULU-NPU ACER仅0.35%；整体性能与教师持平或更优，参数仅为教师的约1/4，FLOPs下降约95%，实现52 FPS实时推理。

**⚠️ 局限性**

局限性包括对极端遮挡或光照条件下的微动捕捉仍有限，蒸馏性能高度依赖教师表现，且未在低帧率摄像头或不同硬件平台上充分验证。

---

## 333. Bypassing Direct Reconstruction: Speech Detection from MEG via Large-Scale Audio Retrieval

**arXiv ID:** 2605.13099 | [PDF](https://arxiv.org/pdf/2605.13099v1)

**作者:** Boda Xiao `[一作]` (Peking University), Heping Cheng `[通讯]` (Peking University)

**通讯引用:** 27469 | [OpenAlex ID](https://openalex.org/A5076736316)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种两步框架，先用对比学习从大型LibriVox音频库中检索与测试MEG匹配的语音片段，再用语音检测模型直接生成二进制的静音/语音序列；

**💡 创新点**

创新点在于把语音解码任务转化为音频检索+后续检测，避免直接从噪声大脑信号回归语音特征，显著提升识别精度；

**🔧 技术方法**

使用了CNN+Wav2vec2.0的对比学习框架（InfoNCE）、ConvConcatNet编码器、深度CNN语音检测模型，以及负Pearson相关损失；

**📊 数据集**

数据集包括MEG实验数据（Sherlock 1），对应的LibriVox音频（约10,000本书的60%），并自行生成带插入静音段的MEGaudio；

**📈 对比分析**

在LibriBrain 2025 Speech Detection任务的扩展赛道上，F1得分0.962，排名第一；与单纯回归或传统匹配-不匹配模型相比，性能显著提升；

**⚠️ 局限性**

局限在于仅能检索已下载的LibriVox子集，无法覆盖所有可能的音频，且对未匹配段仍需回归方法；此外，匹配依赖时间窗口滑动，可能受噪声影响。

---

## 334. Watermarking Should Be Treated as a Monitoring Primitive

**arXiv ID:** 2605.13095 | [PDF](https://arxiv.org/pdf/2605.13095v1)

**作者:** Toluwani Aremu `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Jie Zhang `[通讯]` (Agency for Science Technology and Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究通过构建观察者威胁模型，系统评估了在多键部署下，零比特水印对内部和外部监控的影响，证明即使不显式编码身份，水印也能通过聚合实现实体级归因与识别。

**💡 创新点**

将水印从单纯的检测机制重新定义为监控原语，提出观察者基础威胁模型，并首次揭示多键部署下即使零比特水印也能内外部实现监控的双重用途特性。

**🔧 技术方法**

采用多种零比特水印算法（如KGW、Tree‑Ring）与大型生成模型（Qwen2.5‑14B、Stable Diffusion v2.1），内部观察者利用检测阈值实现归因，外部观察者使用BERT/CLIP等分类器基于可观测特征进行源识别。

**📊 数据集**

文本使用C4数据集，图像使用Stable Diffusion Prompt dataset；实验中对训练集与测试集保持提示独立，确保评估的泛化性。

**📈 对比分析**

内部观察者采用TPR@1%FPR评估归因准确率，外部观察者采用top‑1/3分类准确率；实验表明内部归因几乎完美，外部识别在样本数增多后可提升至70–90%准确率，而无水印或共享键情形退回随机水平。

**⚠️ 局限性**

实验仅覆盖有限模型与水印方法，未评估不同生成配置、对抗扰动或鲁棒性；外部监控在实体规模大或水印弱化时可能效果有限；内部监控在多键部署下基本不可避免，设计层面难以完全消除。

---

## 335. Bayesian Nonparametric Mixed-Effect ODEs with Gaussian Processes

**arXiv ID:** 2605.13088 | [PDF](https://arxiv.org/pdf/2605.13088v1)

**作者:** Julien Martinelli `[一作]` (Aalto University), Mélanie Prague `[通讯]` (University of Bordeaux)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种贝叶斯非参数混合效应ODE模型（MEGPODE），通过高斯过程先验将群体向量场和个体偏差分解到函数空间，利用状态空间GP轨迹先验和虚拟插值实现无需反复ODE求解的轨迹推断。

**💡 创新点**

创新点在于：①将经典NLME的参数空间分层迁移到向量场函数空间；②使用可微分的诱导特征/高斯过程实现可扩展的共享与个体场表示；③结合软插值约束与Kalman平滑，实现闭式回归更新共享与个体场。

**🔧 技术方法**

技术包括高斯过程、诱导特征扩展、状态空间GP表示、局部线性化的虚拟观测、Kalman滤波/平滑、混合效应回归、经验贝叶斯超参数学习和变分推断。

**📊 数据集**

使用一系列合成基准系统（线性振荡器、Lotka–Volterra、Van der Pol、FitzHugh–Nagumo、SIR流行病学模型和药代动力学模型），通过随机效应注入实现异质性，未使用真实医学或生物数据。

**📈 对比分析**

与GPODE、CoDA、混合效应神经ODE以及相应消融模型比较；MEGPODE在群体RMSE、插值/预测RMSE、NLPD、CRPS和覆盖率指标上均取得首位或接近首位的平均排名，尤其在群体轨迹恢复和个体预测方面表现突出；在机制失配的FHN实验中，半机械版MEGPODE优于纯数据驱动和传统机械模型。

**⚠️ 局限性**

局限性包括：仅在合成数据上验证；模型性能高度依赖核函数与诱导特征选择；轨迹更新基于局部线性化，若状态后验宽广或多模态，可能导致误差；需要经验贝叶斯调参，适用性对超参数敏感。

---

## 336. Learning to See What You Need: Gaze Attention for Multimodal Large Language Models

**arXiv ID:** 2605.13080 | [PDF](https://arxiv.org/pdf/2605.13080v1)

**作者:** Junha Song `[一作]` (KAIST), Sangdoo Yun `[通讯]` (NAVER AI Lab)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Gaze Attention 机制，使 MLLM 能够在生成时动态选择视觉区域进行注意力，仅关注任务相关的视觉信息，同时通过可学习上下文 tokens 维持全局视觉感知。

**💡 创新点**

创新点：① 将视觉 KV 缓存按空间/时间分组为区域，用轻量级描述符对每一步进行动态路由；② 引入可学习上下文 tokens 以补充全局视觉上下文；③ 采用渐进 Top‑K 训练策略解决稀疏注意力的学习不稳定。

**🔧 技术方法**

使用技术包括：区域化视觉键分组、描述符 mean‑pooling、动态 Top‑K 路由、可学习上下文 tokens、LLM 内部自注意力、KV 缓存压缩、Progressive Top‑K 训练、FlashAttention / ReKV 等加速与内存优化。

**📊 数据集**

数据集：Cambrian‑Align（视觉‑语言对齐），Cambrian‑7M（图像预训练），Cambrian‑S‑3M（视频预训练，采样 20%），以及 13 个图像理解基准和 6 个视频理解基准，用于评估。

**📈 对比分析**

对比方法：与密集注意力基线、InfiniPot‑V、HERMES 等缓存裁剪方法对比；实验表明 Gaze Attention 在 13 个图像和 6 个视频基准上匹配或超过密集基线，同时将视觉 KV 缓存缩减高达 90%；在 FLOPs 方面减少约 81%，内存降低 79%。

**⚠️ 局限性**

局限性：对区域大小、K 选择的敏感性；训练过程中需要渐进 Top‑K 机制，直接训练会显著下降；仅在图像/视频理解任务验证，生成任务效果尚未充分评估；跨模型迁移性与不同视觉编码器兼容性待进一步验证。

---

## 337. Decoding Product Codes and Staircase Codes with Iteration-Independent Weighting Coefficients

**arXiv ID:** 2605.13201 | [PDF](https://arxiv.org/pdf/2605.13201v1)

**作者:** Andreas Straßhofer `[一作]` (Technical University of Munich), Andreas Straßhofer `[通讯]` (Technical University of Munich)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5092838102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种迭代不依赖加权系数的软输出 FEC 解码器，用于产品码和阶梯码，完全在对数域实现。

**💡 创新点**

创新点在于使用单一迭代无关的加权系数 γ，并改进 APP 近似，使得不再需要概率域计算，降低实现复杂度，同时保持甚至提升性能。

**🔧 技术方法**

技术手段包括 Soft‑output Chase‑II 列表解码、对数域 APP 近似、路径度量、滑动窗口解码、eBCH 组成码、BPSK‑AWGN 信道模型以及迭代解码。

**📊 数据集**

实验使用 (256,239) eBCH 组成码，率 0.872 的产品码和 0.867 的阶梯码，在 AWGN 信道下仿真不同 E_b/N_0，记录 BER。

**📈 对比分析**

与 Chase‑Pyndiah 与 SOCS 解码器比较，产品码上在 BER=10⁻⁶ 时获得 0.23 dB 的增益，阶梯码上获得 0.22 dB 的增益；对比实验表明新解码器在相同迭代次数下性能更优。

**⚠️ 局限性**

局限性包括：验证仅在特定 eBCH 组成码和仿真环境下进行；列表来自 Chase‑II，列表大小仍影响复杂度；对其他码结构或硬件实现的鲁棒性尚未评估。

---

## 338. Empowering IoT Security: On-Device Intrusion Detection in Resource Constrained Devices

**arXiv ID:** 2605.13159 | [PDF](https://arxiv.org/pdf/2605.13159v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 339. Decoupled Planning for Multiple Omega-Regular Objectives

**arXiv ID:** 2605.13185 | [PDF](https://arxiv.org/pdf/2605.13185v1)

**作者:** Guy Avni `[一作]` (University of Haifa), K. S. Thejaswini `[通讯]` (Université Libre de Bruxelles)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种解耦规划框架，用于在有限有向图上为多个 ω-正则目标生成路径，框架通过多个独立代理各自满足单一目标并由一个与图和目标无关的调度器动态组合，保证组合路径几乎必然满足所有目标。

**💡 创新点**

创新点包括：① 通过“convention”机制实现完全无通信的多目标协调；② 明确指出不同 ω-正则目标（Büchi、co‑Büchi、Parity）对调度器与代理信息需求的递增等级；③ 证明存在通用随机调度器以及最小限制的可执行策略；④ 设计了安全组件的 shielded 组合协议和 liveness 组件的 lasso‑sampling 协议。

**🔧 技术方法**

主要技术手段包括：图遍历与 ω‑正则目标分解（安全+活性）；概率公平调度器与 Borel‑Cantelli 论证；利用有限/无记忆策略的期望访问时间上界；构造 Markov 链证明收敛到共识状态；记忆化策略（路径感知、调度感知、全历史感知）与协议设计。

**📊 数据集**

本文属于理论分析，没有使用具体实验数据集；所有结果均在抽象的有限图模型上证明。

**📈 对比分析**

与传统单一策略或 auction‑based 框架相比，本文的解耦框架在实现上更轻量、可扩展；理论上证明在强连通图和公平随机调度下可达到最优几乎必然满足；但缺乏实验性能评估，仅给出理论收敛性（可能指数级）。

**⚠️ 局限性**

主要局限：① 收敛时间缺乏上界，可能指数级；② 对于大规模代理数，记忆需求随代理数线性增长；③ 仅对强连通图给出完整证明，非强连通情况仍待扩展；④ 实际实现的随机调度与协议同步成本未在实验中验证。

---

## 340. DiffST: Spatiotemporal-Aware Diffusion for Real-World Space-Time Video Super-Resolution

**arXiv ID:** 2605.13182 | [PDF](https://arxiv.org/pdf/2605.13182v1)

**作者:** Zheng Chen `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 23638 | [OpenAlex ID](https://openalex.org/A5074865219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DiffST，一种高效视频扩散模型，用于实用空间-时间视频超分辨率（STVSR）

**💡 创新点**

创新点包括：①采用视频级单步采样，显著提升推理效率；②设计跨帧上下文聚合（CFCA）模块，利用多帧信息构建中间帧；③引入视频表示引导（VRG）模块，通过多头注意力提取全视频时空表示并与文本提示融合，显著增强时空信息利用

**🔧 技术方法**

技术要点：基于预训练WAN视频生成扩散模型；光流估计与双向光流融合实现CFCA；多头跨注意力获取视频级表示并通过投影与文本嵌入融合；单步采样（t=799）与VAE编码解码实现一阶段生成

**📊 数据集**

数据集：训练使用HQ-VSR；评估合成数据集UDM10、Vid4；真实场景数据集MVSR4x、RealVSR；合成时同时应用多种空间降质与帧率降采样

**📈 对比分析**

与多种VSR+VFI组合（STAR、SeedVR、SeedVR2、BiM‑VFI、MoMo、TLBVFI）以及单阶段扩散方法VEnhancer比较；在合成与真实数据集上大多数指标排名第一/第二，PSNR提升≈0.43 dB；推理速度提升约17×（相比VEnhancer），参数量更少

**⚠️ 局限性**

局限性：①仍受预训练模型分布匹配限制，输入分布偏差可能导致性能下降；②单独使用视频表示引导会影响感知质量；③在极端降质或复杂运动场景下，CFCA与VRG的融合效果尚有限

---

## 341. PanoWorld: Towards Spatial Supersensing in 360$^\circ$ Panorama World

**arXiv ID:** 2605.13169 | [PDF](https://arxiv.org/pdf/2605.13169v1)

**作者:** Changpeng Wang `[一作]` (Zhejiang University), Xi Chen `[通讯]` (University of Hong Kong)

**通讯引用:** 58271 | [OpenAlex ID](https://openalex.org/A5100329996)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出统一的全景原生空间学习框架，训练多模态大语言模型在360°全景图像上进行完整观测空间的推理。

**💡 创新点**

创新点包括：1）将全景视角作为连续的观测球面进行建模；2）构建大规模可验证的全景元数据图；3）设计Spherical Spatial Cross‑Attention注入球面几何；4）推出PanoSpace‑Bench评测基准。

**🔧 技术方法**

使用的技术包括：大型元数据构建管线（检测、语言标注、深度关联），Spherical Spatial Cross‑Attention（SSCA）模块，Qwen3.5‑VL作为基座，指令调优。

**📊 数据集**

使用的数据集包括：约57万高质量equirectangular全景图（混合来源），以及构造的PanoSpace‑Bench、H^∗Bench、R2R‑CE等评测集合。

**📈 对比分析**

在PanoSpace‑Bench上相较于专有及开源MLLM提升至56.5%总体准确度；在H^∗Bench零样本达到56.1%并能通过微调提升至70%；在R2R‑CE仅使用全景输入时实现SR 54.3、SPL 52.1，显著优于基线。

**⚠️ 局限性**

局限性：对全景几何的假设仍依赖于投影重映射，元数据构建对检测精度敏感；目前仅覆盖ERP形式的全景，缺少多视角融合；模型训练仅用单个epoch，进一步扩展仍需探索。

---

## 342. Unifying Physically-Informed Weather Priors in A Single Model for Image Restoration Across Multiple Adverse Weather Conditions

**arXiv ID:** 2605.13158 | [PDF](https://arxiv.org/pdf/2605.13158v1)

**作者:** Jiaqi Xu `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 54906 | [OpenAlex ID](https://openalex.org/A5032708386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种统一的物理先验网络（WeatherNet），可一次性恢复雾、雨、雪三种恶劣天气下的图像，并通过两阶段框架先估计天气先验（传输、遮挡、大气光），再利用这些先验进行场景细化。

**💡 创新点**

创新点包括：① 将可见粒子遮挡与雾散射统一成像模型；② 在第二阶段引入天气感知交叉注意力（WACA），由传输引导全局注意力（TGGA）和遮挡引导局部注意力（OGLA）实现先验信息的高效融合；③ 设计天气融合器（WAF）进一步整合增强特征；④ 构建规模达30k的合成数据集 Weather30K，为多天气恢复提供统一训练数据。

**🔧 技术方法**

技术手段：基于 U‑Net 的两阶段网络；Transformer 风格注意力模块（TGGA、OGLA、WACA）；多分支先验估计（传输、遮挡、光照）；先验监督损失；自定义卷积和深度卷积融合；使用 AdamW + cosine annealing 训练策略。

**📊 数据集**

主要使用数据集：自制 Weather30K（含雾、雨、雪合成图像，附带先验）；公开对比数据集 RESIDE、Rain‑Haze、RainDrop、Snow100K‑L、Rain1400、CSD 等；实验还利用 Cityscapes、COCO、RTTS、RIS 等真实天气图像进行无参考评价。

**📈 对比分析**

与多种 SOTA 方法（如 All‑in‑One、TransWeather、MWFormer、WeatherDiff、Two‑Stage 等）在 PSNR/SSIM、NIQE/MUSIQ 等指标上对比，WeatherNet 在三种实验设置中平均提升 PSNR 1–2 dB、SSIM 0.01–0.02，且模型大小与推理速度与主流方法相当，显示出显著的性能优势。

**⚠️ 局限性**

局限性：在极端天气条件下细节恢复仍不够充分；对真实数据的泛化能力仍受限；缺乏大规模真实先验标注；未充分利用生成模型或更大多样化数据集来进一步提升极端场景下的恢复效果。

---

## 343. Understanding Generalization through Decision Pattern Shift

**arXiv ID:** 2605.13148 | [PDF](https://arxiv.org/pdf/2605.13148v1)

**作者:** Huiqi Deng `[一作]` (Xi'an Jiaotong University), Xia Hu `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究深度神经网络的泛化失效，提出通过内部决策模式（通道贡献向量）的稳定性来定义和量化泛化，即提出决策模式漂移（DPS）指标，并利用DPS谱统一描述多种泛化失效模式。

**💡 创新点**

创新点在于：①首次将内部决策逻辑的稳定性视为泛化的核心指标；②设计基于GradCAM的通道贡献向量来客观刻画样本的决策模式；③证明DPS与泛化误差在样本、类别和数据集层面呈线性相关；④构建DPS谱，揭示理想、失效、域迁移、OOD以及shortcut学习等场景的连续演化。

**🔧 技术方法**

主要技术包括：GradCAM通道贡献向量的计算、余弦相似度衡量DPS、线性回归与Pearson相关性分析、PCA/t‑SNE可视化、对比实验与activation‑pattern基线比较。

**📊 数据集**

实验数据集涵盖：CIFAR‑10、CIFAR‑100、TinyImageNet、ImageNet（50类子集）、CIFAR‑10‑C（不同污染级别）、STL‑10（OOD）以及Colored MNIST（shortcut学习）。

**📈 对比分析**

通过与activation‑pattern基线对比，验证决策模式在类别内一致性高、类别间可分离；在VGG、ResNet、GoogLeNet等架构上，DPS与泛化误差的Pearson相关系数普遍超过0.8，线性回归斜率接近1；DPS谱在不同失效场景下呈连续右移或多峰分布，说明DPS能够量化并区分各类失效。

**⚠️ 局限性**

局限性：DPS需要以干净训练样本的类别均值为参考，当训练标签存在噪声或决策模式呈多模态时，DPS与泛化误差的相关性会显著下降；对现代结构如Vision Transformers或密集预测任务的适用性尚待扩展。

---

## 344. Collaborating in Multi-Armed Bandits with Strategic Agents

**arXiv ID:** 2605.13145 | [PDF](https://arxiv.org/pdf/2605.13145v1)

**作者:** Idan Barnea `[一作]` (Tel Aviv University), Yishay Mansour `[通讯]` (Tel Aviv University)

**通讯引用:** 21578 | [OpenAlex ID](https://openalex.org/A5014637159)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在多智能体贝叶斯多臂赌博机（MAB）中，如何通过信息共享来激励战略性、持续性参与者进行协作学习；提出一种名为CAOS（Collaborating Agents with Optimistic Stopping）的机制，允许智能体动态决定是否继续协作并通过优化预期回报评估停留与退出的收益；证明该机制在无金钱转移、无绑定合同的条件下构成纳什均衡（并可延伸为子博弈完美均衡），并在对称算法下保持与完全协作系统相近的低贝叶斯遗憾；对非对称算法给出更宽松的最大遗憾保证；并通过对UCB、Thompson Sampling、Successive Elimination等算法的理论分析展示了机制的性能。

**💡 创新点**

创新点主要在于：1）在持久、战略性智能体场景下仅利用信息流作为激励工具，首次设计可持续协作的机制；2）引入“乐观期望回报（Optimistic Expected Reward）”递归评估，动态决定协作成员；3）证明该机制既是纳什均衡，又在对称算法下几乎匹配完全协作的低遗憾；4）提供对非对称算法的最大遗憾保证与示例。

**🔧 技术方法**

技术方法包括：贝叶斯多臂赌博机模型、信息共享与可验证信号的设定、递归乐观期望回报（OER）算法、基于信息流的激励机制设计、纳什均衡与子博弈完美均衡分析、理论遗憾分析（对称与非对称算法）。

**📊 数据集**

无实验数据，全部为理论推导与数学证明。

**📈 对比分析**

比较方法为对比传统非战略多智能体MAB（完全协作）与单智能体算法；在对称算法如UCB、Thompson Sampling下，CAOS实现的贝叶斯遗憾为O(K log T/(mΔ))与O(√(KT/m))，几乎与完全协作相当；在非对称算法下给出更宽松的最大遗憾上界。

**⚠️ 局限性**

局限性包括：1）对非对称算法的遗憾保证相对宽松，尚无最优性证明；2）假设环境生成的信号可验证，现实中若不可验证则机制失效；3）仅适用于贝叶斯MAB，未扩展至情境/线性/马尔可夫决策过程；4）机制的计算复杂度与OER递归实现仍需进一步优化。

---

## 345. McCast: Memory-Guided Latent Drift Correction for Long-Horizon Precipitation Nowcasting

**arXiv ID:** 2605.13197 | [PDF](https://arxiv.org/pdf/2605.13197v1)

**作者:** Penghui Wen `[一作]` (University of Sydney), Zhiyong Wang `[通讯]` (University of Sydney)

**通讯引用:** 32014 | [OpenAlex ID](https://openalex.org/A5100614129)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为 McCast 的降水现在预报方法，通过主动记忆校正隐状态来改进长时序预测。

**💡 创新点**

创新点在于将记忆从被动条件转化为显式漂移校正，并设计了 Drift‑Corrective Memory Bank 的两阶段校正机制。

**🔧 技术方法**

使用 encoder‑decoder 框架，结合 Corrective Latent Extractor 与 Correction‑Aware Memory Retrieval 的记忆银行，并采用 MSE + LPIPS 损失进行训练。

**📊 数据集**

实验使用 SEVIR 与 MeteoNet 两个常用雷达降水数据集。

**📈 对比分析**

与 ConvGRU、FourCastNet、DiffCast 等多种基准方法对比，McCast 在 CSI_M、HSS 上提升约5–9%，在高强度阈值下提升超过20%。

**⚠️ 局限性**

局限性包括较高的计算与存储开销，且在极端天气情况下的漂移校正效果尚不充分，缺乏对长时序漂移的理论解释。

---

## 346. Does Engram Do Memory Retrieval in Autoregressive Image Generation?

**arXiv ID:** 2605.13179 | [PDF](https://arxiv.org/pdf/2605.13179v1)

**作者:** Jinghao Wang `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 54906 | [OpenAlex ID](https://openalex.org/A5032708386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

将 Engram 模块从自然语言处理迁移到二维视觉空间，并将其嵌入到一个基于 VQ‑token 的类条件自回归图像生成模型中，随后通过多种实验（参数比例、门控裁剪、donor probe、冻结噪声训练等）探究其作用机制。

**💡 创新点**

提出了一套诊断工具包（gate‑clamp、donor probe、冻结噪声训练）用于区分 Engram 的内容检索效应与仅仅是网络架构侧通道的贡献，证实在图像生成任务中 Engram 并非内容检索器，而是一个轻量化的 gated side‑path 侧通道。

**🔧 技术方法**

使用了 Engram 的哈希键检索、二维空间 n‑gram 哈希、门控融合、KV‑cache 兼容增量推理；核心模型是 24 层 causal Transformer（隐藏维 768、16 头、RoPE、SwiGLU、RMSNorm），配合 AliTok 语义分词器。

**📊 数据集**

在 ImageNet 256×256 上训练，使用 AliTok 生成的 256 语义 token，验证 Engram 在此数据集上的表现。

**📈 对比分析**

与纯 AR 基线（参数相同）比较，Engram 变体在 FID 上均略逊（每个参数量下 FID 均比基线更高），但通过引入 Engram 可显著降低 backbone FLOPs；门控裁剪显示最优门值常为小常数；donor probe 证明哈希地址不对内容进行选择；冻结噪声训练表明记忆表内容对最终质量影响微乎其微，主要贡献来自 gated side‑path。

**⚠️ 局限性**

实验仅覆盖中等规模的稠密自回归模型，未对大规模、MoE、极低数据、长尾或更高分辨率场景进行评估，可能隐藏在这些场景下真正的记忆效应。

---

## 347. GenCape: Structure-Inductive Generative Modeling for Category-Agnostic Pose Estimation

**arXiv ID:** 2605.13151 | [PDF](https://arxiv.org/pdf/2605.13151v1)

**作者:** Jiyong Rao `[一作]` (Tongji University), Shengjie Zhao `[通讯]` (Tongji University)

**通讯引用:** 3255 | [OpenAlex ID](https://openalex.org/A5035948567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了GenCape框架，在无先验结构的情况下通过迭代结构感知VAE和组合图传输自学习实例化骨架进行类别无关关键点估计。

**💡 创新点**

采用生成式隐式结构学习，自动从支持图像推断柔性邻接矩阵，并通过贝叶斯融合与注意力重加权形成查询感知图，解决固定骨架与噪声支持的局限。

**🔧 技术方法**

基于变分图自动编码器、图变压器解码器、贝叶斯加权融合、注意力机制以及SwinV2/HRNet等视觉编码器。

**📊 数据集**

在公开的MP‑100数据集（100子类别、18K图像）上进行实验。

**📈 对比分析**

与Image/Text/Graph支持的多种前沿方法在1‑shot与5‑shot下对比，GenCape在MP‑100上实现PCK@0.2平均88.09%（1‑shot）/93.53%（5‑shot），相较GraphCape提升约1–2个百分点，超过多模态模型。

**⚠️ 局限性**

模型仍受限于支持样本的多样性与质量，极端遮挡或极端姿态下仍可能出现误检，且对大规模跨类别迁移的推理速度与内存开销尚未充分评估。

---

## 348. A Constraint Programming Approach for $n$-Day Lookahead Playoff Clinching

**arXiv ID:** 2605.13142 | [PDF](https://arxiv.org/pdf/2605.13142v1)

**作者:** Gili Rosenberg `[一作]` (Amazon), Ruben S. Andrist `[通讯]` (Amazon)

**通讯引用:** 523 | [OpenAlex ID](https://openalex.org/A5015701688)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合约束规划与自定义树搜索的算法，用于计算NHL球队在未来n天内能否提前锁定季后赛席位，并生成具体胜负情景。

**💡 创新点**

将最新NHL规则和七项细致的平衡决策规则融入多阶段CP子程序，利用“猜测+惰性约束”处理复杂平衡，并通过预处理与启发式剪枝实现高效的n天情景探索。

**🔧 技术方法**

使用CP‑SAT进行约束规划，构建自定义树搜索、预处理剪枝、贪心/启发式游戏与节点排序、懒惰约束和目标分配模型等技术。

**📊 数据集**

基于公共NHL API收集的2021‑22至2024‑25赛季常规赛和赛程数据。

**📈 对比分析**

通过与NHL官方发布的0天、1天情景对比验证准确性；在多赛季实验中，求解时间从秒级到分钟级不等，剪枝效率高；对2天、3天情景在30分钟时间限制下大部分可解。

**⚠️ 局限性**

对于极其复杂的赛程或大规模多日（n>3）仍受指数爆炸限制；极少数平局情况需要额外目标分配和懒惰约束；依赖现代规则，若规则更改需重新调整。

---

## 349. UIBenchKit: A unified toolkit for design-to-code model evaluation

**arXiv ID:** 2605.13141 | [PDF](https://arxiv.org/pdf/2605.13141v1)

**作者:** Chinh T. Le `[一作]` (Singapore Management University), Yintong Huo `[通讯]` (Singapore Management University)

**通讯引用:** 470 | [OpenAlex ID](https://openalex.org/A5080873193)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了UIBenchKit，一个统一评估设计到代码模型的开源工具；

**💡 创新点**

创新点在于提供标准化、模块化的评估框架、可视化分析界面以及大规模基准实验；

**🔧 技术方法**

使用Python/Flask后端、React前端、Playwright渲染、MLLM接口抽象、CLIP等评估指标；

**📊 数据集**

使用Design2Code和DCGen两大公开基准数据集；

**📈 对比分析**

通过统一pipeline比较16个LLM、5种方法，共832个实例，发现分解式方法在复杂页面上提升布局精度，但对细粒度指标差异不均；

**⚠️ 局限性**

局限在于图像分割精度不足，导致下游生成误差放大，缺少结构感知的分割技术。

---

## 350. Multi-Modal Guided Multi-Source Domain Adaptation for Object Detection

**arXiv ID:** 2605.13140 | [PDF](https://arxiv.org/pdf/2605.13140v1)

**作者:** Sangin Lee `[一作]` (Sejong University), Yukyung Choi `[通讯]` (Sejong University)

**通讯引用:** 2174 | [OpenAlex ID](https://openalex.org/A5052425702)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种利用深度图和文本提示的多源领域自适应目标检测框架MS-DePro，显著提升跨域检测性能。

**💡 创新点**

创新点在于：①将深度图作为域无关特征的输入，进行深度引导定位；②设计多模态引导的可学习提示，将域无关和域相关信息分别编码到提示词中；③通过教师-学生框架和伪标签实现无监督域自适应，避免域对抗学习的训练冲突。

**🔧 技术方法**

技术包括多源领域自适应（MSDA）、深度估计模型（Depth Anything/Depth Pro）、RegionCLIP视觉‑语言模型、可学习提示学习（CoOp/CoCoOp衍生）、平均教师（EMA）以及伪标签生成。

**📊 数据集**

使用的基准数据集包括BDD100K、Cityscapes、KITTI、MS‑COCO、Synscapes、FoggyCityscapes、AdverseWeather、VDD‑DAOD、Clipart1k、Comic2k、Watercolor2k等，涵盖实景、合成、天气变化与艺术风格等多域。

**📈 对比分析**

与现有单源UDA和多源MSDA方法相比，MS-DePro在跨时间、跨相机、混合域、以及多源域泛化（MSDG）任务上均取得了最高mAP，跨时间提升高达+5.8 mAP，跨相机+9.3 mAP，域泛化mPC最高，显示出明显的性能优势。

**⚠️ 局限性**

主要局限在于深度图质量受运动模糊、遮挡、半透明表面等因素影响，低质量深度可能导致定位辅助失效；此外，提示学习对稀有类别的泛化仍有不足，未来需改进DS Token设计以缓解类别不平衡。

---

## 351. Extending Blockchain Untraceability with Plausible Deniability

**arXiv ID:** 2605.13132 | [PDF](https://arxiv.org/pdf/2605.13132v1)

**作者:** Eunchan Park `[一作]` (KAIST), Min Suk Kang `[通讯]` (KAIST)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了将资产隐藏在DeFi损失事件（如sandwich和arbitrage）中的Deniable Covert Asset Transfer技术，并在Ethereum和Arbitrum测试网上验证其不可观测性和取证难度；

**💡 创新点**

首次将资产转移的不可观察性与可否认性相结合，突破传统匿名集只能遮蔽发送者-接收者关系而仍可检测的局限，并提出多变量罕见度排名的取证框架；

**🔧 技术方法**

利用MEV提取框架（PBS、Flashbots rbuilder、MEV‑Boost）、自定义合约、Copula与Survival Copula统计模型以及现有MEV检测和取证图分析工具；

**📊 数据集**

使用约313,219条Ethereum sandwich事件和55,482条Arbitrum arbitrage事件的数据集，以及在测试网收集的合约和交易记录；

**📈 对比分析**

与常规MEV检测器和取证工具对比后显示，实验传输被误判为普通MEV且无法通过图分析链接发送者和接收者；多变量排名能将前500名事件中超过一半不在单一阈值内，显著提升取证优先级；

**⚠️ 局限性**

受限于MEV事件的幂律分布导致单阈值检测易产生误报或漏报，且该技术依赖于持续存在的损失事件，完全根除难度大；目前仅适用于MEV、套利、清算等场景。

---

## 352. STAR: Semantic-Temporal Adaptive Representation Learning for Few-Shot Action Recognition

**arXiv ID:** 2605.13202 | [PDF](https://arxiv.org/pdf/2605.13202v1)

**作者:** Hongli Liu `[一作]` (Tongji University), Shengjie Zhao `[通讯]` (Tongji University)

**通讯引用:** 3255 | [OpenAlex ID](https://openalex.org/A5035948567)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 STAR 框架，解决少样本动作识别中语义-时序失配与多尺度时序建模不足的问题，采用帧级跨模态对齐与多频时序细化提升原型学习。

**💡 创新点**

创新点包括：① Temporal Semantic Attention (TSA) 通过帧级跨模态注意力与 LLM 生成的时序类描述实现细粒度语义-时序对应；② Semantic Temporal Prototype Refiner (STPR) 集成 Semantic‑Guided Focus、Action‑Specific Dynamic Temporal 与 Action‑Centric Unified Temporal 三子模块，结合多频时序采样与双向状态空间建模，实现局部细节与全局连贯的原型优化。

**🔧 技术方法**

技术手段：CLIP 视觉/文本预训练、Mamba 基础的 Temporal State Space 模块、跨模态注意力、InfoNCE 对齐、OTAM/ Bi‑MHM 时序对齐、LSTM‑style 双向状态空间、对齐损失与分类损失联合训练。

**📊 数据集**

使用的基准数据集：HMDB51、UCF101、Kinetics‑100、Something‑Something V2 Full 与 Small。

**📈 对比分析**

与现有方法对比，STAR 在所有五个基准上均取得 SOTA 结果，1‑shot 情况下 SSv2‑Full 提升约 8%（从 55.6% 提升到 63.5%），在其他数据集亦实现 5–8% 的准确率提升；在与 Bi‑MHM/OTAM 的集成实验中均显著优于 CLIP‑FSAR、TSAM 等方法。

**⚠️ 局限性**

局限性：① 仍依赖 CLIP 视觉/文本预训练，若预训练不匹配可能影响效果；② 对 LLM 生成的时序类描述敏感，若提示质量下降可能导致对齐不佳；③ 目前对最大帧数（8–32）有限制，超长视频时序建模仍待进一步研究。

---

## 353. ECG-NAT: A Self-supervised Neighborhood Attention Transformer for Multi-lead Electrocardiogram Classification

**arXiv ID:** 2605.13194 | [PDF](https://arxiv.org/pdf/2605.13194v1)

**作者:** Mahsa Gazeran `[一作]` (University of Kurdistan), Fardin Akhlaghian Tab `[通讯]` (University of Kurdistan)

**通讯引用:** 1275 | [OpenAlex ID](https://openalex.org/A5081265236)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了一种基于Neighborhood Attention Transformer（NAT）的自监督双阶段框架ECG‑NAT，用于多导联心电图的异常分类。

**💡 创新点**

创新点包括：1）采用分层局部注意力实现多尺度时间特征提取，兼顾心搏细节与节律宏观；2）结合掩码自编码器的生成预训练与监督对比损失的双重损失微调；3）在自监督阶段使用多来源无标签数据提升域不变性。

**🔧 技术方法**

技术实现主要包括：1）一维卷积分词器与NAT块的层级结构；2）掩码自编码器（MAE）预训练；3）监督对比损失与交叉熵的联合优化；4）Gaussian噪声掩码与位置偏置。

**📊 数据集**

使用了四个公开心电图数据集：无标签的Chapman和Ningbo做预训练；有标签的PTB‑XL和CPSC2018做微调与评估。

**📈 对比分析**

与多种最新自监督和监督Transformer方法（如MocoV3、CMSC、MaeFE、CRT、ST‑MEM等）对比，在PTB‑XL和CPSC2018上实现了更高的准确率（PTB‑XL 90.2%，CPSC2018 98.6%）和AUROC（PTB‑XL 97.7%，CPSC2018 98.6%），在低标注比例（1%）下仍保持高性能（AUROC 88.1%）。

**⚠️ 局限性**

局限性包括：缺乏对模型可解释性的系统评估；仅验证了心电图任务，尚未在其他时间序列医学数据上验证；对多模态数据融合的探索不足。

---

## 354. FIKA-Bench: From Fine-grained Recognition to Fine-Grained Knowledge Acquisition

**arXiv ID:** 2605.13193 | [PDF](https://arxiv.org/pdf/2605.13193v1)

**作者:** Geng Li `[一作]` (Peking University), Yuxin Peng `[通讯]` (Peking University)

**通讯引用:** 9211 | [OpenAlex ID](https://openalex.org/A5047811387)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个311例、注重证据支持、避免泄漏的细粒度知识获取基准（FGKA），并评估了闭卷模型与工具驱动的多模态代理；

**💡 创新点**

创新点在于提出细粒度主动知识获取任务，构建泄漏敏感且证据可验证的样本集，弥补传统闭集细粒度分类对外部知识获取的忽视；

**🔧 技术方法**

采用大型多模态模型（LMM）与基于工具的多模态代理（如OpenClaw、OpenCode），并实现视觉检索、逆向图像搜索、OCR等工具调用；

**📊 数据集**

数据来源包括FGVC-Aircraft、Stanford Cars、Stanford Dogs、Oxford Flowers-102、Food-101、VegFru、Google Landmarks v2等公开数据集及志愿者真实场景图像；

**📈 对比分析**

通过严格答案准确率评估，与闭卷模型对比，最佳系统Kimi-K2.6仅达25.1%准确率，未超过30%，说明任务仍具挑战性；

**⚠️ 局限性**

局限性包括样本覆盖面有限、缺乏多语言和区域多样性、对工具依赖高且未提供细粒度过程级评估，且需随模型和网页内容更新周期性维护。

---

## 355. Dynamics Computation of Soft-Rigid Hybrid-Link System and Its Application to Motion Analysis of an Athlete Wearing Sport Prosthesis

**arXiv ID:** 2605.13192 | [PDF](https://arxiv.org/pdf/2605.13192v1)

**作者:** Sunghee Kim `[一作]` (University of Tokyo), Ko Yamamoto `[通讯]` (University of Tokyo)

**通讯引用:** 13065 | [OpenAlex ID](https://openalex.org/A5080124007)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种软刚混合连杆系统，用于对佩戴叶片弹簧假肢的运动员进行动态运动分析，利用逆运动学重建运动捕捉数据，再通过二次规划逆动力学估计关节力矩、假肢内部力及地面反作用力，并基于这些结果进行肌肉力优化。

**💡 创新点**

创新点在于将软体连续杆的PCS模型与传统刚体多连杆模型统一到一个混合连杆框架中，能够同时处理人体刚体与柔性假肢的相互作用；同时在逆动力学中使用二次规划实现关节力矩与接触力的同时求解；并将肌肉截断模型与EMG信号对比验证肌肉力估计。

**🔧 技术方法**

使用的技术包括Cosserat杆理论的PCS离散化、刚体多连杆动力学与逆运动学、二次规划（QP）求解逆动力学、以及基于EMG的肌肉力优化。

**📊 数据集**

实验数据集为一名15岁右侧下肢截肢运动员的运动捕捉与力板数据，采集频率200 Hz，使用51个光学标记（30个位于人体，21个位于假肢），并对走步与跑步在三块力板上进行验证。

**📈 对比分析**

与力板测得的地面反作用力相比，逆动力学估计的rRMSE在步态中约为12%，跑步时约10–20%；肌肉力估计与EMG信号的RMSE在不同单步相位中约为7–12%，表明两种估计在峰值上相近，但在静态相位存在零值误差。

**⚠️ 局限性**

主要局限包括仅使用单一受试者、接触点设定简化导致在着地与离地阶段误差较大、EMG传感器位置受限导致肌肉激活信号不完全、以及优化方法在关节静止时无法产生非零肌肉力。

---

## 356. N-vium: Mixture-of-Exits Transformer for Accelerated Exact Generation

**arXiv ID:** 2605.13190 | [PDF](https://arxiv.org/pdf/2605.13190v1)

**作者:** Aleksander Lorenc `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21538 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出N-vium，一种混合退出（mixture‑of‑exits）Transformer架构，通过在不同深度设置多路出口实现自适应深度推理；

**💡 创新点**

创新点在于将早期退出与多路退出结合，训练时直接优化混合分布π_mix，使每个出口都对最终预测贡献而非仅逼近最终层；同时通过“piggybacking”技术在推理时将未完成的层计算推迟并并行完成，提升GPU吞吐量而不舍弃任何计算；

**🔧 技术方法**

核心技术包括：1）在每个分支点插入router MLP和轻量化adapter；2）使用共享语言建模头（W_lm）生成各出口分布；3）训练时加入compute penalty β，鼓励早期退出；4）推理时采用两步采样与piggybacking恢复KV缓存；

**📊 数据集**

在C4数据集上使用LLaMA‑2 tokenizer进行预训练，规模从几百万到1.5B参数；

**📈 对比分析**

与参数和数据匹配的密集Transformer基线对比，N‑vium在保持或略低于baseline perplexity的情况下实现高达57% 的壁钟速度提升；与CALM、LayerSkip等现有方法相比，N‑vium在质量-速度曲线上处于最优（无perplexity损失且更快）；

**⚠️ 局限性**

局限性包括：1）需要额外的训练损失和β调参；2）推理时需要特殊的批量化策略；3）训练时相较于密集模型有更高的FLOPs消耗；4）在更大规模模型的稳定性与效率提升尚需进一步验证。

---

## 357. STOP: Structured On-Policy Pruning of Long-Form Reasoning in Low-Data Regimes

**arXiv ID:** 2605.13165 | [PDF](https://arxiv.org/pdf/2605.13165v1)

**作者:** Chenjun Xu `[一作]` (University of Washington), Bingbing Wen `[通讯]` (University of Washington)

**通讯引用:** 115 | [OpenAlex ID](https://openalex.org/A5016171730)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 STOP 框架，通过自我蒸馏、结构化推理界面和 Earliest Correct Node (ECN) 节点级剪枝，显著降低推理过程中的冗余步骤，减少 token 数量并保持甚至提升模型准确率。

**💡 创新点**

创新点在于：①在低数据微调环境下采用自我蒸馏生成在策略（on‑policy）的高质量推理轨迹，消除教师指导造成的分布偏移；②设计基于节点分割、分类注解和推理树构建的统一结构化界面；③在该结构上引入 ECN，按最早出现正确答案的节点停止推理，既保留语义连续性又剔除过度推理。

**🔧 技术方法**

技术实现包括：自我蒸馏生成成功推理轨迹、Best‑of‑K 筛选、规则化节点分割、少样本 prompt 注解分类、自动推理树构建、ECN 判断与剪枝、LoRA 微调、温度采样与随机种子扩展。

**📊 数据集**

使用的数据集：训练阶段在 PRM‑12k 随机抽取 1,000 题目做低数据微调；评估阶段分别采用 GSM8K、Math 500、AIME 2024 三个多步推理基准。

**📈 对比分析**

与多种基线对比（Base、No‑Thinking、Teacher‑Guided‑Full、Teacher‑Guided‑Random‑Prune、Teacher‑Guided‑ECN、Self‑Distilled‑Full、Token‑Skip、Random‑Prune 等），STOP 在 Qwen‑7B 和 LLaMA‑3‑8B 上平均减少 19.4%–42.4% token，同时保持或略提升准确率（例如 Qwen‑GSM8K 91.1% vs 90.1%，LLaMA‑Math500 87.7% vs 87.9%）。

**⚠️ 局限性**

局限性包括：①需手工或模型生成节点分类与 ECN 判断，存在误判风险；②主要验证在两款已蒸馏模型和三套基准上，泛化到更大模型或不同推理风格尚待评估；③虽然显著剪枝，但仍无法彻底解决所有冗余推理，极端难题仍可能出现过度思考。

---

## 358. Continual Fine-Tuning of Large Language Models via Program Memory

**arXiv ID:** 2605.13162 | [PDF](https://arxiv.org/pdf/2605.13162v1)

**作者:** Hung Le `[一作]` (Deakin University), Svetha Venkatesh `[通讯]` (Deakin University)

**通讯引用:** 20996 | [OpenAlex ID](https://openalex.org/A5045540854)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ProCL框架，将LoRA适配器划分为可路由程序内存槽，使用输入条件注意力路由实现持续学习，保留原始适配器并通过整合完成静态模型，解决LLM连续微调中的灾难性遗忘。

**💡 创新点**

将LoRA低秩空间结构化为程序内存，并引入动态路由与周期性整合，实现可扩展的持续学习，兼顾可塑性与稳定性且无推理成本提升。

**🔧 技术方法**

低秩适配（LoRA）、程序内存与注意力路由、稳定-适应权重组合、渐进式整合、CLS启发的持续学习机制以及LoRA参数化。

**📊 数据集**

BoolQ、SQuAD、AdversarialQA（问答）以及多任务文本分类（3、4、15任务）在Llama、Qwen、Flan‑T5等大模型上。

**📈 对比分析**

与Seq‑LoRA、O‑LoRA、DEAL、EWC、Replay等基线比较，ProCL在QA平均准确率达69.1，明显高于DEAL（+2.9）和Seq‑LoRA（+7.9）；在文本分类AA与R‑1上亦持续领先，尤其在长任务序列上提升约1‑1.3%。

**⚠️ 局限性**

路由需要分化，若任务相似易收敛至少数程序导致干扰；固定程序数目会在任务多样化时成为瓶颈，可能导致程序共享而忘记；需进一步提升路由多样性和动态扩展。

---

## 359. EvObj: Learning Evolving Object-centric Representations for 3D Instance Segmentation without Scene Supervision

**arXiv ID:** 2605.13152 | [PDF](https://arxiv.org/pdf/2605.13152v1)

**作者:** Jiahao Chen `[一作]` (Hong Kong Polytechnic University), Bo Yang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 72304 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种无监督3D实例分割方法，解决了合成数据与真实点云之间的几何域差距。

**💡 创新点**

创新点包括：①对象候选识别与演化模块，能通过自监督伪标签持续适应目标域；②对象完成模块，在评分前修复部分几何；两者协同显著提升对象检测准确性。

**🔧 技术方法**

技术核心：GrabS框架的预训练对象中心网络、强化学习动态容器策略、SparseConv分割网络、AdaPoinTr补全网络，以及自监督演化更新。

**📊 数据集**

使用的公开数据集：ShapeNet（合成预训练）、ScanNet、S3DIS（真实场景）以及自构造的多类别合成测试集。

**📈 对比分析**

在ScanNet、S3DIS和自定义多类数据集上与UnScene3D、Part2Object、EFEM、GrabS等无监督基线及Mask3D、3D-BoNet等监督方法对比，AP提升约10%+，逼近监督模型性能。

**⚠️ 局限性**

局限性：仍依赖合成数据预训练；对高度遮挡的细小对象识别不足；演化频率与补全模型选择对性能影响较大。

---

## 360. GRACE: Gradient-aligned Reasoning Data Curation for Efficient Post-training

**arXiv ID:** 2605.13130 | [PDF](https://arxiv.org/pdf/2605.13130v1)

**作者:** Junjie Li `[一作]` (Harbin Institute of Technology), Xiaofeng Zhang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29879 | [OpenAlex ID](https://openalex.org/A5101742243)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于梯度对齐的细粒度推理数据挑选方法GRACE，用于后训练阶段选取高质量子集以提升模型性能。

**💡 创新点**

创新点在于将推理过程视为一系列优化事件，结合答案梯度方向和历史轨迹一致性的两重对齐信号，并引入表示层梯度代理实现可扩展的步级评估。

**🔧 技术方法**

核心技术包括梯度方向余弦对齐、轨迹一致性评分、表示层梯度代理（token‑level upstream signals）以及对预训练视觉‑语言模型的前向只计算收集。

**📊 数据集**

使用的主要数据集为MMathCoT‑1M（含大量数学 Chain‑of‑Thought 推理轨迹）以及一系列多模态问答与推理基准（MMBench、SQA、MathVista等）。

**📈 对比分析**

与随机、Longest、Stepmax、LESS、ICONS、CADC等基线相比，GRACE在20%子集上实现108.8%相对平均性能，5%子集仍保持100.2%，并在跨模型迁移实验中亦表现出优于全数据的效果。

**⚠️ 局限性**

局限在于对已标注推理步骤的 CoT 数据依赖较强，对无步骤或不同结构推理数据适用性待验证；同时梯度代理的选择与权重需手工调参，影响模型间的一致性。

---

## 361. LeanSearch v2: Global Premise Retrieval for Lean 4 Theorem Proving

**arXiv ID:** 2605.13137 | [PDF](https://arxiv.org/pdf/2605.13137v1)

**作者:** Guoxiong Gao `[一作]` (Peking University), Bin Dong `[通讯]` (Peking University)

**通讯引用:** 16851 | [OpenAlex ID](https://openalex.org/A5100746745)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出全局前提检索任务，构建了 Lean4/Mathlib 上的 MathlibQR 与 MathlibMPR 两个专家标注基准，并提出 LeanSearch v2 系统，包含标准检索模式和推理检索模式，显著提升了库检索与证明生成的效率。

**💡 创新点**

创新点包括①将推理检索技术引入 Lean 前提检索，②首次构造全球前提检索任务与专门基准，③在推理模式中使用草图-检索-反射循环实现集值前提检索，突破了单步前提选择的局限。

**🔧 技术方法**

技术实现主要依赖层级化的 Mathlib 非正式化语料库、Qwen3-Embedding 与 Qwen3-Reranker 的嵌入–重排序管道、Claude Sonnet 4.5 与 Kimi K2 Instruct 的草图生成、过滤与判定、以及 Jixia 静态分析工具。

**📊 数据集**

所用数据集为 MathlibQR（200 条声明、946 个查询）和 MathlibMPR（69 条定理、平均 2.96 条前提组）、FATE‑H（100 个代数问题）以及 MathlibMPR‑Prop（50 个可统一形式化的定理）。

**📈 对比分析**

与 LeanExplore、LeanFinder、LeanSearch v1、INF‑X、ReasonIR、DIVER、ReProver、LeanPremise、LeanStateSearch 等基线在 nDCG、Recall、覆盖率等指标上进行比较；标准模式在 MathlibQR 上 nDCG@10 达 0.62，推理模式在 MathlibMPR 上 Recall@10 为 46.1%、覆盖率 30.4%，并在下游证明生成任务中实现 20% 的成功率。

**⚠️ 局限性**

局限性包括：仅在 Lean4/Mathlib 进行评估；缺乏跨检索器或跨证明系统的迁移性验证；在高度智能代理系统中的交互与规划未探究；仅关注证明构造检索，未覆盖其他检索场景。

---

## 362. Switching Successor Measures for Hierarchical Zero-shot Reinforcement Learning

**arXiv ID:** 2605.13207 | [PDF](https://arxiv.org/pdf/2605.13207v1)

**作者:** Stefan Stojanovic `[一作]` (KTH), Alexandre Proutiere `[通讯]` (KTH)

**通讯引用:** 6021 | [OpenAlex ID](https://openalex.org/A5025136069)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了切换后继测度（switching successor measures）框架，利用前后向（FB）后继表示联合学习高层子目标选择策略和低层动作策略，实现零样本层次强化学习。

**💡 创新点**

创新点在于通过从标准后继表示直接推导出层次结构，避免了固定时间抽象或手工子目标的限制；切换后继测度自然包含了子目标切换的价值增益，可无监督地形成层次决策。

**🔧 技术方法**

使用的技术包括：前后向后继表示、期望回归（expectile regression）求解后继测度、优势加权回归（AWR）提取高层与低层策略、一次性无动作的后继学习、以及对奖励嵌入的线性表示。

**📊 数据集**

实验数据集主要为 AntMaze（Medium、Large、Giant、Teleport 四种迷宫）以及一个离散迷宫示例；也在 OGBench 环境中验证。

**📈 对比分析**

与 HIQL、ICVF、FB、One‑Step FB 等基线对比，FB π‑Switch 在目标到达任务上与 HIQL 竞争，且在无层次版本下已超越其它基线；在更一般的分布式奖励任务中取得最高平均 IQM 分数；层次化版本进一步提升成功率，尤其在 Medium、Large 迷宫上表现显著。

**⚠️ 局限性**

局限性包括：对基础后继模型质量高度依赖，切换优势估计在局部区域难以精准；在 Giant 迷宫等极长时间尺度任务中仍难以突破；高层策略的可靠性受低层动作策略影响；未覆盖多子目标链式规划，累积误差仍待研究。

---

## 363. CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large VIsion-Language Models

**arXiv ID:** 2605.13178 | [PDF](https://arxiv.org/pdf/2605.13178v1)

**作者:** Sangin Lee `[一作]` (Sejong University), Yukyung Choi `[通讯]` (Sejong University)

**通讯引用:** 2174 | [OpenAlex ID](https://openalex.org/A5052425702)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种无训练、文本引导的视觉令牌裁剪方法 LiteLVLM，用于在大视觉语言模型中高效实现像素级定位。

**💡 创新点**

核心创新是利用 CLIP 视觉-文本相似度反转现象，保留低相似度（即更贴近目标区域）的视觉令牌，并通过上下文感知恢复全局信息；同时设计了自适应令牌选择策略。

**🔧 技术方法**

使用 CLIP ViT-L/14 做视觉编码、GLaMM（基于 LLaVA-1.5）做语言模型、SAM 做像素解码；裁剪方法基于 CLIP 的相似度和自注意力分数；实验中还对比了多种现有裁剪方法。

**📊 数据集**

在 RefCOCO、RefCOCO+、RefCOCOg 三个引用表达分割数据集以及 Ref-DAVIS-17、Refer-YouTube-VOS 两个视频分割数据集上进行评估。

**📈 对比分析**

与 TRIM、FastV、LLaVA-PruMerge、VisionZip、SparseVLM、VisPruner 等方法相比，LiteLVLM 在保持 90%+ 原性能的同时可将视觉令牌减少 66.7%，实现 22% 推理加速、2.3× 内存节省；视频任务中仅降 4-5% 性能。

**⚠️ 局限性**

限制在于裁剪策略依赖 CLIP 预训练的相似度分布，可能在与 CLIP 不同的视觉编码器或任务（非像素定位）中效果不佳；同时未涉及多模态上下文细粒度的语义匹配。

---

## 364. When Does Hierarchy Help? Benchmarking Agent Coordination in Event-Driven Industrial Scheduling

**arXiv ID:** 2605.13172 | [PDF](https://arxiv.org/pdf/2605.13172v1)

**作者:** Ziqi Wang `[一作]` (Zhejiang University), Hailiang Zhao `[通讯]` (Zhejiang University)

**通讯引用:** 1726 | [OpenAlex ID](https://openalex.org/A5018608721)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DESBench，一个统一的分布式事件驱动调度基准，用于评估多智能体在层级化调度环境中的协同与协调性能。

**💡 创新点**

创新点包括：① 构建共享的离散事件模拟环境，将决策、资源耦合与动态约束统一到同一框架；② 定义四种典型协同范式（集中式、层级式、异质式、全能式），并以协议参数化实现；③ 设计四维评估指标（有效性、约束对齐、协同效率、鲁棒性），揭示协同设计的结构性权衡；④ 通过大规模实验展示协同范式在不同 LLM 与框架下的差异性。

**🔧 技术方法**

采用离散事件仿真（DES）、基于协议的层级调度、LLM 生成决策与对话、LangGraph 与 AgentScope 两大框架进行多层次协同实现。

**📊 数据集**

使用真实工业调度日志与公开的工业排程数据集（如 A5C12）作为实验基准，涵盖不同难度场景（分支压力、集群拉力、晚期任务承诺）。

**📈 对比分析**

通过在相同任务实例下对比四种协同范式（以及不同 LLM 与框架），在 15 条评价指标上进行定量评估。结果显示：集中式在鲁棒性与通信成本上表现优异但在效率上受限；层级式在效率与鲁棒性之间取得平衡；异质式在适应性和鲁棒性上表现突出，但协同开销最高；全能式在约束满足最强，但鲁棒性最弱。没有单一范式在所有维度上占优，展示了设计协同机制的必要性。

**⚠️ 局限性**

局限性：① 仅研究四种静态协同范式，未覆盖动态自适应或混合策略；② 关注工业调度场景，缺乏对其他领域通用性的验证；③ 对于异质式和全能式，通信/决策成本高，需进一步优化；④ 依赖预定义的 LLM 与框架，未探讨模型泛化与迁移。

---

## 365. Formal Conjectures: An Open and Evolving Benchmark for Verified Discovery in Mathematics

**arXiv ID:** 2605.13171 | [PDF](https://arxiv.org/pdf/2605.13171v1)

**作者:** Moritz Firsching `[一作]` (Google DeepMind), Pushmeet Kohli `[通讯]` (Google DeepMind)

**通讯引用:** 119652 | [OpenAlex ID](https://openalex.org/A5013834379)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可扩展的、由Lean4公式化的数学命题集合，作为自动推理系统评估基准；

**💡 创新点**

引入零污染的开放猜想集合、统一API、误公式化三层分类、答案(sorry)机制，并提供冻结子集与可重现的评估流程；

**🔧 技术方法**

利用Lean4、Mathlib、AlphaProof、DeepMind prover等自动定理证明器与语言模型；

**📊 数据集**

Formal Conjectures仓库中的数百条Lean4形式化命题（开放与已证实的）作为数据集；

**📈 对比分析**

通过在冻结的100题子集上用AlphaProof和DeepMind prover的树搜索实现评估，AlphaProof在低计算配置下可达45%解决率，升至50%；DeepMind prover可达66%；整体表现表明可扩展的信号；

**⚠️ 局限性**

主要局限包括对Mathlib覆盖范围的依赖、问题选取偏向、误公式化风险、零污染保证有限以及对Lean4 kernel安全性的依赖。

---

## 366. Pareto-Guided Optimal Transport for Multi-Reward Alignment

**arXiv ID:** 2605.13155 | [PDF](https://arxiv.org/pdf/2605.13155v1)

**作者:** Ying Ba `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 26328 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构造提示特定的 Pareto 前沿并利用最优传输来指导文本到图像生成模型的多奖励后训练优化，从而避免奖励劫持。

**💡 创新点**

① 理论证明统一全局目标导致奖励劫持；② 提出基于 Pareto 前沿的在线/离线最优传输优化框架；③ 引入 Joint Domination Rate 与 Joint Collapse Rate 两个评估指标。

**🔧 技术方法**

Pareto 最优传输（Sinkhorn）、LoRA 微调、VLM 决策代理、DRaFT‑K 换算器、Stable Diffusion 3.5 Turbo 等。

**📊 数据集**

Stable Diffusion 3.5 Turbo 预训练模型、DiffusionDB、Parti‑Prompts 数据集、Pick‑a‑Pic、Pick‑High 等奖励模型训练集。

**📈 对比分析**

与单奖励、加权多奖励、奖励汤、异质奖励界限基线对比；在 JDR_2 提升 11%，JDR_4 提升 3.4%，JCR_4 降低 0.2%，人类评估近 80% 胜率。

**⚠️ 局限性**

依赖 VLM 代理检测弱奖励模型、需预先生成 Pareto 前沿，复杂度较高，对极端奖励范围的鲁棒性仍有限。

---

## 367. Strikingness-Aware Evaluation for Temporal Knowledge Graph Reasoning

**arXiv ID:** 2605.13153 | [PDF](https://arxiv.org/pdf/2605.13153v1)

**作者:** Rikui Huang `[一作]` (Huazhong University of Science and Technology), Wei Wei `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 256794 | [OpenAlex ID](https://openalex.org/A5100352881)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于规则的事件冲击度量框架 RSMF，并将冲击度量融入加权评估指标（WMRR、WHits@k），重新定义了 TKGR 的评估方式。

**💡 创新点**

创新点在于：①首次针对时间知识图谱提出冲击度量，将事件的“罕见性”和“重要性”量化；②通过冲击度量作为权重实现对高冲击事件的更精准评估；③揭示了路径基方法与表示基方法在不同冲击度级别下的性能差异。

**🔧 技术方法**

主要技术包括：基于一阶时间规则的同伴事件检索、规则信度与时序衰减结合的期望得分计算、冲击度量归一化与加权评估指标的设计；以及一个简单的路径-表示混合集成方案。

**📊 数据集**

使用四个公开时间知识图谱数据集：ICEWS14、ICEWS18、ICEWS05-15 与 GDELT。

**📈 对比分析**

对比了六类基线（三类路径基、三类表示基）以及 LLM 基础模型，并在原始与冲击度量加权评估下展示性能。实验显示所有模型在冲击度较高时性能显著下降，路径基方法在低冲击度下表现更好，表示基方法在高冲击度下更优；集成模型在原始评估上仍保持 SOTA，但在冲击度量评估中提升有限。

**⚠️ 局限性**

局限性包括：RSMF 仅使用一阶规则，可能忽略更复杂的时序模式；冲击度量依赖规则质量与频率统计，若规则提取不足可能影响准确性；评估仍基于已知历史数据，无法完全验证对未来突发事件的预测能力。

---

## 368. On the Generalization of Knowledge Distillation: An Information-Theoretic View

**arXiv ID:** 2605.13143 | [PDF](https://arxiv.org/pdf/2605.13143v1)

**作者:** Bingying Li `[一作]` (Hong Kong University of Science and Technology), Haiyun He `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

本文通过信息论视角，将教师与学生的训练过程建模为耦合随机过程，提出了“蒸馏KL散度”（distillation divergence）来度量两者的差异，并在此基础上给出了学生模型的泛化上界与下界；

**💡 创新点**

创新点在于：①首次统一考虑教师与学生训练过程的耦合，并用KL散度量化匹配程度；②提出上界与下界的闭合式泛化界，揭示散度与学生泛化误差之间的权衡；③在线性高斯案例中将散度拆解为偏差、方差与秩瓶颈成本，给出可操作的设计准则；④引入局部损失尖锐度（sharpness）进一步收紧泛化界。

**🔧 技术方法**

使用的技术包括：信息论的Donsker‑Varadhan变分表示、算法稳定性与子高斯假设、中心条件（central condition）、随机过程的KL散度分解、线性高斯模型的解析推导以及局部尖锐度分析。

**📊 数据集**

文章未给出具体实验数据集，主要以理论推导为主；在案例研究中使用了线性高斯数据模型（X、Y 服从矩阵正态分布）。

**📈 对比分析**

由于缺乏实验验证，未进行与现有方法的性能比较；论文仅提供理论界限，未给出数值表现。

**⚠️ 局限性**

局限性包括：①理论界限仍存在上下界差距，缺乏经验验证；②只在理论层面讨论，缺乏实际算法实现与实验评估；③对数据集与模型的通用性和可扩展性尚未验证。

---

## 369. Code-Centric Detection of Vulnerability-Fixing Commits: A Unified Benchmark and Empirical Study

**arXiv ID:** 2605.13138 | [PDF](https://arxiv.org/pdf/2605.13138v1)

**作者:** Nils Loose `[一作]` (University of Lübeck), Thomas Eisenbarth `[通讯]` (University of Lübeck)

**通讯引用:** 6458 | [OpenAlex ID](https://openalex.org/A5075079896)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了统一的 VFC（vulnerability‑fixing commit）评估框架，整合 20 个碎片化数据集，并对 125M–14B 参数的代码语言模型在超过 180 次实验中进行系统评估。

**💡 创新点**

创新点在于：①提出统一数据集框架解决数据碎片化和偏差问题；②使用轻量级的程序内语义上下文丰富差分；③从细粒度的 Attribution（整合梯度）角度揭示模型对代码改动的关注度；④系统地检验模型规模、上下文、数据多样性对 VFC 检测的影响。

**🔧 技术方法**

技术包括：代码 LMs（CodeBERT、CommitBART、Qwen2.5‑Coder 等）、Tree‑Sitter 语法树、GumTree 差分、控制流与数据流切片、Prompt‑based 生成式分类、集成梯度 Attribution、t‑SNE 及 Jensen‑Shannon 分布分析。

**📊 数据集**

使用了 20 个公开和部分私有的 VFC 数据集，合并后约 180,000+ 提交，其中包括手工验证、advisory‑based、自动工具标注以及全部提交四种子集，覆盖多种编程语言。

**📈 对比分析**

对比方法：随机拆分、时间拆分、组（项目）分层拆分以及 CVE‑映射拆分；结果显示：即便是 14B 参数模型，在严格的 FP 率（0.5%）下仍漏掉 93%+ 的漏洞；组分层拆分比随机拆分下降约 17%；时间拆分更差，但因项目组合漂移导致不可靠；Prompt‑based 生成式模型在 CVE 集合上略优，但可能受预训练数据污染。

**⚠️ 局限性**

局限性：模型几乎不学习真正的安全相关代码语义，主要依赖提交信息；在代码仅输入时对改动的关注度极低；规模扩大和语义上下文丰富均未显著提升；数据集仍受项目特异性和时间漂移影响；生成式方法的性能可能受预训练数据污染；需开发更能捕获跨程序语义的输入表征和更严谨的评估策略。

---

## 370. KAST-BAR: Knowledge-Anchored Semantically-Dynamic Topology Brain Autoregressive Modeling for Universal Neural Interpretation

**arXiv ID:** 2605.13133 | [PDF](https://arxiv.org/pdf/2605.13133v1)

**作者:** Haoning Wang `[一作]` (Beihang University), Yang Li `[通讯]` (Beihang University)

**通讯引用:** 30728 | [OpenAlex ID](https://openalex.org/A5100421802)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种新的EEG基础模型KAST-BAR，能够将脑电信号与专家医学知识和大语言模型进行跨模态对齐；

**💡 创新点**

创新点在于三阶段流水线：DSHA编码器显式建模非欧几里得脑电拓扑，KASP利用LLM生成实例级专家级语义画像，STAR通过可学习的latent query实现语义驱动的动态特征聚合；

**🔧 技术方法**

核心技术包括双流层级注意力编码、VQ‑VAE离散化、知识锚定语义画像、查询式多头交叉注意力与LoRA微调；

**📊 数据集**

预训练使用21个多样化EEG数据集（共1600+受试者），下游评估涵盖工作负荷、情绪、异常检测、睡眠分期、事件分类与慢波事件六个任务；

**📈 对比分析**

与单任务模型（如EEGNet、SPaRCNet）及多任务模型（如EEGPT、NeuroLM、THD‑BAR）对比，KAST‑BAR‑Large在所有六个任务上均优于基线，尤其在情绪、睡眠与慢波事件上提升8‑15%；

**⚠️ 局限性**

局限性包括对慢波事件等细粒度频谱特征的敏感性略低，且由于LLM的生成特性，仍可能出现信息幻觉，需在临床部署前进行严谨验证。

---

## 371. Finding the Weakest Link: Adversarial Attack against Multi-Agent Communications

**arXiv ID:** 2605.13170 | [PDF](https://arxiv.org/pdf/2605.13170v1)

**作者:** Maxwell Standen `[一作]` (University of Adelaide), Claudia Szabo `[通讯]` (University of Adelaide)

**通讯引用:** 2877 | [OpenAlex ID](https://openalex.org/A5016821538)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

提出了单受害者通信扰动攻击框架，并通过新的损失函数和基于Jacobian的消息、受害者和攻击时机选择，显著提升了多智能体强化学习系统的攻击效果。

**💡 创新点**

创新点在于：①设计了加权损失和最大损失两种新的攻击损失函数以平衡攻击影响与成功率；②引入Jacobian‑based 选择机制，实现对最脆弱消息、受害者及攻击时机的精准定位；③将单智能体的tempo方法扩展到多智能体环境。

**🔧 技术方法**

技术手段包括白盒梯度攻击（FGM、PGD）、Jacobian‑saliency 计算、基于Jacobian的消息/受害者/时序筛选，以及多种现有tempo方法的改造。

**📊 数据集**

实验数据集涵盖五个模拟环境：Navigation、Orthogonal/Diagonal Predator‑Prey、Small/Large TrafficJunction，并测试了两种通信协议 OBS 与 RIAL。

**📈 对比分析**

通过与多种基准攻击（CBTS、MMR、ML、NS、VL、ST）及随机消息/未定向损失对照，实验显示在大多数场景下，Jacobain攻击与新损失函数能获得更低的奖励、更多的任务指标下降以及更高的成功率。

**⚠️ 局限性**

局限性包括：假设目标系统已训练充分且 Q‑函数准确；仅评估两种通信协议和有限的环境配置；未考虑网络延迟、噪声、拓扑变化等实际因素；阈值调节导致攻击率控制困难；未对潜在防御机制进行系统评估。

---

## 372. GeoBuildBench: A Benchmark for Interactive and Executable Geometry Construction from Natural Language

**arXiv ID:** 2605.13167 | [PDF](https://arxiv.org/pdf/2605.13167v1)

**作者:** Jinwoong Kim `[一作]` (Peking University), Huishuai Zhang `[通讯]` (Peking University)

**通讯引用:** 4577 | [OpenAlex ID](https://openalex.org/A5042848593)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GeoBuildBench基准，评估LLM和多模态代理将自然语言几何题目转化为可执行几何构造的能力。

**💡 创新点**

通过交互式执行DSL程序、可验证约束和多步修正，创建一个不依赖单一“金标准”图的几何构造评测框架。

**🔧 技术方法**

设计最小化的几何DSL、Python解释器、Matplotlib渲染器以及基于GPT‑4.1的三阶段文本过滤与结构提取管道；评估使用多模态LLM（GPT‑5.1、Gemini‑3‑Flash、Qwen3‑VL‑235B、LLaMA‑3.2‑90B‑Vision）在迭代交互中的性能。

**📊 数据集**

489个中文教材式平面几何题目，包含所需对象和可检验约束，来源于GeoQA及其他在线教材，经过自动过滤和人工验证。

**📈 对比分析**

通过执行成功率、迭代步数、结构幻觉率、缺失对象和未满足约束等指标比较模型，GPT‑5.1与Gemini‑3‑Flash最高成功率约78%，而Qwen3‑VL与LLaMA‑3.2‑Vision仅分别为42%与21%；强模型在幻觉率低、恢复快。

**⚠️ 局限性**

仅涵盖中文平面几何、有限的约束类型和DSL，未覆盖英语或更广泛的几何范畴，评估依赖固定交互预算、视觉输入以及特定提示，可能无法推广至其他环境。

---

## 373. A Hybrid Tucker-LSTM Tensor Network Model for SOC Prediction in Electric Vehicles

**arXiv ID:** 2605.13200 | [PDF](https://arxiv.org/pdf/2605.13200v1)

**作者:** Han Wang `[一作]` (Southwest University), Bing Wang `[通讯]` (China Automotive Engineering Research Institute Co., Ltd.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种结合Tucker张量分解与LSTM网络的混合模型，用于对电动车全生命周期的状态电荷（SOC）进行预测。

**💡 创新点**

创新点在于：①首次将Tucker分解应用于高维电池监测数据的降维，保持多维结构与时间依赖；②将降维后的低秩特征直接输入LSTM，实现结构化特征压缩与序列学习的无缝融合；③在真实全生命周期数据上验证了此融合策略的显著优势。

**🔧 技术方法**

使用的技术包括：Tucker张量分解（特征子空间投影）、LSTM递归网络（双层、隐藏单元64、Dropout 0.2）、滑动窗口特征构造、Adam优化器、PyTorch实现、基准LSTM对比、MSE/MAE/RMSE/R²评估指标。

**📊 数据集**

数据集为遵循中国国家标准 GB/T 32960.3-2016 的电动车完整生命周期监测数据，覆盖里程 0–38,548.7 km，包含时间戳、充电状态、车速、总电压/电流、SOC、单体电压/温度差等 14 维特征。

**📈 对比分析**

与标准 LSTM 在相同网络结构、相同训练/验证划分（70%/15%/15%）下进行对比；评估指标显示 Tucker‑LSTM MSE 降至 6.22（比 LSTM 21.07 降低 70.5%）、MAE 降至 1.73%（下降 48.7%）、RMSE 降至 2.49%（降 45%），R² 提升至 0.976（高于 0.918）。

**⚠️ 局限性**

局限性包括：仅与单一降维方案（Tucker）和单一基准（标准 LSTM）对比；未评估在不同车型或不同运行环境下的泛化能力；缺少实时推断速度与计算资源消耗的量化分析；模型训练依赖较大 GPU 计算资源，实际部署时需进一步优化。

---

## 374. Stable Attention Response for Reliable Precipitation Nowcasting

**arXiv ID:** 2605.13181 | [PDF](https://arxiv.org/pdf/2605.13181v1)

**作者:** Penghui Wen `[一作]` (University of Sydney), Kun Hu `[通讯]` (Edith Cowan University)

**通讯引用:** 13029 | [OpenAlex ID](https://openalex.org/A5028673475)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种通过调节注意力响应能量实现雨量即时预报的新框架HARECast。

**💡 创新点**

创新点在于发现并稳定跨样本注意力响应能量的波动，提出组别化能量正则化方法以降低预测误差下界。

**🔧 技术方法**

使用多头自注意力、能量聚类与正则化、条件扩散模型以及重建解码器进行学习。

**📊 数据集**

在SEVIR和MeteoNet两个雷达预报基准以及SEVIR的多模态雷达+卫星数据上进行评估。

**📈 对比分析**

与现有单模态和多模态预报方法相比，HARECast在CSI、HSS、LPIPS、SSIM等指标上均取得最优或竞争性性能，尤其在高强度降水场景下提升显著。

**⚠️ 局限性**

缺点是模型参数量与计算量相对较大，对批量统计的依赖使得小批量训练下稳定性下降，且对超参数（正则化权重、批量大小）较为敏感。

---

## 375. Do Heavy Tails Help Diffusion? On the Subtle Trade-off Between Initialization and Training

**arXiv ID:** 2605.13175 | [PDF](https://arxiv.org/pdf/2605.13175v1)

**作者:** Hamza Cherkaoui `[一作]` (Télécom SudParis), Antonio Ocello `[通讯]` (ENSAE Paris)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5028908279)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文从理论和实验两方面研究了在扩散/流模型中使用重尾噪声的效果，提出了采样误差的分解并系统比较了重尾与轻尾模型的表现。

**💡 创新点**

创新点在于给出重尾噪声导致的统计学习难度的非渐近误差界定，揭示了初始化优势被训练误差累积所抵消的本质权衡。

**🔧 技术方法**

使用了分解采样误差、非参数估计理论、α‑stable 分布的前向噪声、score‑matching 与 flow‑matching 方法，以及基于 MMD-RBF 和 TCE 的尾部评价指标。

**📊 数据集**

实验数据集包括 30 维重尾目标（α‑stable 分布与混合分布）以及两份真实表格数据（KDD Cup 99 与 Wildfires），全部使用 50,000 样本进行训练。

**📈 对比分析**

通过 MMD‑RBF 与 TCE（90%，95%，99%）对比，发现轻尾 DDPM 与 GF‑Linear 通常表现与重尾 DLPM 无显著提升，且 DLPM 在尾部和整体性能上波动更大。

**⚠️ 局限性**

主要局限是训练误差在重尾模型中累积放大，导致初始化优势被抵消；重尾模型在样本量有限时不稳定，缺乏有效的逆向动态学习策略。

---

## 376. OxyEcomBench: Benchmarking Multimodal Foundation Models across E-Commerce Ecosystems

**arXiv ID:** 2605.13173 | [PDF](https://arxiv.org/pdf/2605.13173v1)

**作者:** Yong Liu `[一作]` (JD.COM), Yan Li `[通讯]` (JD.COM)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了OxyEcomBench，一套面向双语（中英）电商场景的多模态基准，用于评估大型语言模型与多模态大语言模型在真实电商工作流中的表现。

**💡 创新点**

创新点在于：①整合平台运营商、商家、客户三类角色，覆盖6个能力维度、29个任务；②强调视觉显著性、多轮对话、多图输入，防止文字快捷解；③引入P0–P3四级专家评定的难度标签；④所有实例均来源于真实电商平台并由行业专家验证，保证生态效度。

**🔧 技术方法**

技术方法包括：零样本评估、LLM-as-Judge 的开放式答案打分、对任务设计的多模态格式支持（文本、单图、多图、单轮、多轮）、四级难度标签体系以及对比实验中统一的宏观平均分。

**📊 数据集**

数据集：约6,300条中英双语实例，涵盖29个任务、6个能力维度、3个角色，数据采自真实电商平台并经专家审核；同时标注视觉显著性、多轮对话、多图输入等属性。

**📈 对比分析**

比较方法：在20个主流 LLM/MLLM 上进行零样本评估，按每个能力维度和整体平均分对比；结果显示最高整体得分仅 69.1，表明即使是顶尖模型在电商场景中仍表现有限，且模型在不同维度上的优势分散。

**⚠️ 局限性**

局限性：①缺乏对长文本/长对话的细粒度评估；②难度标签虽基于专家共识，但仍具主观性；③仅覆盖中英双语，未扩展至更多语言；④未探究模型在领域适配、微调等方式下的提升空间。

---

## 377. A$_3$B$_2$: Adaptive Asymmetric Adapter for Alleviating Branch Bias in Vision-Language Image Classification with Few-Shot Learning

**arXiv ID:** 2605.13161 | [PDF](https://arxiv.org/pdf/2605.13161v1)

**作者:** Yiyun Zhou `[一作]` (Zhejiang University), Jingyuan Chen `[通讯]` (Zhejiang University)

**通讯引用:** 3566 | [OpenAlex ID](https://openalex.org/A5090689233)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究者提出了 A_3B_2 这一自适应非对称适配器，用于在 CLIP 视觉‑语言模型的少样本图像分类中缓解分支偏差问题。

**💡 创新点**

创新点在于：①通过不确定性感知的适配器衰减（UAAD）在样本不确定时自动抑制图像分支的适配；②采用 Mixture‑of‑Experts 结构与负载平衡正则化，保证多专家的多样化利用；③实现了完全可插拔的非对称设计，使文本分支始终适配而图像分支可动态调节。

**🔧 技术方法**

核心技术包括：CLIP 的双分支 Transformer 编码器、轻量级适配器（降维‑升维）、不确定性感知损失（基于 softmax 最大值）、负载平衡正则化（均匀专家使用）以及交叉熵加权训练。

**📊 数据集**

实验数据集覆盖 11 个图像分类基准（ImageNet、Caltech101、OxfordPets、StanfordCars、Flowers102、Food101、FGVCAircraft、SUN397、DTD、EuroSAT、UCF101），并在三种任务设置（基‑新类别泛化、跨数据集评估、领域泛化）下进行少样本测试。

**📈 对比分析**

与 11 种主流提示/适配器基线（CoOp、CoOpOp、KgCoOp、RPO、MaPLe、CLIP‑Adapter、TCP、MMA、MMRL、MMRL++）在 1‑16 shot 条件下比较，A_3B_2 在平均准确率、基类、奇异类和调和均值上均位居榜首，且在跨数据集与领域泛化任务中表现最稳健。

**⚠️ 局限性**

局限性包括：仅在 CLIP ViT‑B/16 变体上验证；对其他大规模视觉‑语言模型的适用性未做系统评估；适配器的超参数（专家数、低秩维度、抑制强度）需手工调优；在极端 OOD 场景下不确定性估计的可靠性尚未深入探讨。

---

## 378. Dual-Pathway Circuits of Object Hallucination in Vision-Language Models

**arXiv ID:** 2605.13156 | [PDF](https://arxiv.org/pdf/2605.13156v1)

**作者:** Jiaxin Liu `[一作]` (University of Illinois at Urbana-Champaign), Aofan Liu `[通讯]` (Peking University)

**通讯引用:** 19262 | [OpenAlex ID](https://openalex.org/A5010675602)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在五种不同架构的视觉语言模型上，通过激活补丁方法识别并分析了导致对象存在错误（hallucination）的两条互补计算通路——视觉基线通路和幻觉通路，并通过条件通路分析（CPA）揭示了基线通路在正确与错误样本间的极性翻转，随后对幻觉通路进行组件级抑制，显著降低了错误率。

**💡 创新点**

①发现所有模型共享的双通路结构；②提出CPA诊断方法，揭示基线通路的极性翻转；③通过组件尺度抑制验证幻觉通路的因果作用，超越传统单向量干预；④展示该通路可部分迁移至关系类错误，但对属性错误无显著效果。

**🔧 技术方法**

主要技术包括：激活补丁（activation patching）、条件通路分析（Conditional Pathway Analysis, CPA）、logit lens验证、组件尺度抑制、均值差分投影与probe‑based ITI比较。

**📊 数据集**

使用的主要数据集为 POPE‑adversarial（对象存在问题）、POPE 各分割（random、random‑strong）以及 AMBER（属性、关系、存在三种错误类型）。

**📈 对比分析**

与传统的黑箱干预方法（训练对齐、解码约束、后处理）相比，组件级抑制在 POPE‑adversarial 上可将幻觉率降低 40%–76%，同时保持 ≤2 百分点的准确率损失；对 POPE 难度分割和 AMBER 关系子任务也表现出可观的迁移效果。

**⚠️ 局限性**

限制包括：①对属性错误的抑制无显著改善，表明属性错误涉及不同或更分散的通路；②静态方向干预受限，单向量无法捕获多方向的幻觉信号；③微观层级上的通路布局差异较大，说明仅靠架构并不能完全解释通路分布；④抑制方法虽有效但不具备直接部署的可行性，需要进一步工程化实现。

---

## 379. AcquisitionSynthesis: Targeted Data Generation using Acquisition Functions

**arXiv ID:** 2605.13149 | [PDF](https://arxiv.org/pdf/2605.13149v1)

**作者:** Ishika Agarwal `[一作]` (University of Illinois Urbana Champaign), Dilek Hakkani-Tür `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出利用主动学习中的采集函数（acquisition functions）作为奖励，对语言模型进行强化学习（GRPO），以生成高质量的合成训练数据；随后用这些数据训练学生模型，验证其在多任务（数学、医学问答、编码）中的性能提升。

**💡 创新点**

创新点在于：1）首次将采集函数量化为RL奖励，使生成器能直接学习“有用”数据；2）设计了五种互补的采集奖励（置信度、近似度、梯度模量、多样性、答案方差）；3）展示了生成数据可跨模型架构、尺寸以及不同训练范式（ICL、SFT、RL）使用。

**🔧 技术方法**

技术主要包括：Group Relative Policy Optimization (GRPO) 强化学习框架、基于激活和梯度的奖励计算、HDBSCAN聚类、句子编码器做伪标签、以及对生成文本的格式化奖励。

**📊 数据集**

使用了公开的可验证数据集：Numina（数学）、AIME24（数学）、MedMCQA（医学QA）、PubMedQA（医学QA）、CodeForces（编码）。

**📈 对比分析**

与多种基线（原始生成、DataEnvGym、Prismatic Synthesis、随机挑选、过滤挑选）比较，实验显示：学生模型在分布内平均提升约4%（对更大模型提升更显著），且在分布外的鲁棒性提升约3%；在不同模型家族间迁移效果稳定；在资源受限或丰富的训练范式下均保持一致性。

**⚠️ 局限性**

局限性包括：1）仅在可验证任务上测试，未覆盖指令跟随等非可验证场景；2）奖励组合方式单一，缺乏对多奖励互补性的探索；3）生成模型仍受限于训练数据规模与多样性，无法完全实现“一体化”通用数据生成器。

---

## 380. GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning

**arXiv ID:** 2605.13136 | [PDF](https://arxiv.org/pdf/2605.13136v1)

**作者:** Kasidit Sermsri `[一作]` (Chulalongkorn University), Teerapong Panboonyuen `[通讯]` (Chulalongkorn University)

**通讯引用:** 491 | [OpenAlex ID](https://openalex.org/A5091353147)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于教师置信度门控的闭环蒸馏框架 GateKD，能够将大型语言模型的多步推理能力迁移至小型模型。

**💡 创新点**

创新点在于使用预测熵评估教师的置信度，动态门控软标签、隐藏状态和注意力对齐，显著降低噪声传递与幻觉扩散，实现更稳健的推理迁移。

**🔧 技术方法**

技术方法包括预测熵置信度估计、置信门控的软标签蒸馏、隐藏状态门控对齐、注意力门控蒸馏，以及在 T5/Flan‑T5 体系结构上联合任务损失训练。

**📊 数据集**

实验使用常识推理数据集 CSQA、StrategyQA，逻辑推理数据集 Shuffled Objects，以及符号推理数据集 Last Letter concatenation。

**📈 对比分析**

与 Vanilla‑KD、MCC‑KD、Mentor‑KD 等开放环蒸馏基线以及 GPT‑4o‑mini 零样本 CoT 进行对比，GateKD 在所有模型规模和推理任务上均取得最高准确率，特别在逻辑和符号任务上提升约 3–5 分。

**⚠️ 局限性**

局限性包括：依赖预测熵置信度估计，OOB 或高度歧义输入时可能失准；需要教师的中间表征，无法直接用于黑盒 API；训练时额外计算开销；仅在英文单语基准上验证，未考察多语言、多模态等场景。

---

## 381. ERPPO: Entropy Regularization-based Proximal Policy Optimization

**arXiv ID:** 2605.13131 | [PDF](https://arxiv.org/pdf/2605.13131v1)

**作者:** Changha Lee `[一作]` (Korea Advanced Institute of Science and Technology), Gyusang Cho `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5072505897)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种 Entropy Regularization-based Proximal Policy Optimization (ERPPO) 方法，利用分布式时空模糊学习器（DSA）动态估计对象检测模糊度，并在多智能体 PPO 中加入熵正则化以提升在多维非平稳环境中的搜索性能。

**💡 创新点**

创新点在于：①引入 DSA 学习器预测环境模糊度，为强化学习提供先验；②在 PPO 损失中根据模糊度自适应切换 L1 与 L2 熵正则化，既鼓励高模糊度状态下的探索，又保持低模糊度状态下的稳定性。

**🔧 技术方法**

使用技术包括：Proximal Policy Optimization (PPO)、Multi‑Agent PPO (MAPPO)、Multi‑Agent Soft Actor‑Critic (MASAC)、Multi‑Agent DDQN、QMIX；AirSim + Unreal Engine 进行仿真；YOLOv8‑world 进行目标检测；熵正则化、分布式学习和经验回放。

**📊 数据集**

数据集：基于 AirSim 的多 UAV 航海搜索救援仿真场景，涵盖不同天气、海浪、摄像机位置所产生的图像；检测结果由 YOLOv8‑world 生成；DSA 学习器使用模拟环境中的真实标注作为监督。

**📈 对比分析**

通过在相同仿真环境下训练 2 亿步，对比 MAPPO、MASAC、DDQN、QMIX 等基线算法，ERPPO 在连续动作空间上归一化收益持续高于 0.8，离散动作空间上约为 0.75，明显优于基线并显著降低误检率。

**⚠️ 局限性**

局限性：①模糊度估计依赖仿真环境，真实世界分布可能差异导致泛化不足；②DSA 学习器需要额外训练与数据，增加计算成本；③在极端噪声或多目标场景下的鲁棒性尚未充分验证。

---

## 382. LoREnc: Low-Rank Encryption for Securing Foundation Models and LoRA Adapters

**arXiv ID:** 2605.13163 | [PDF](https://arxiv.org/pdf/2605.13163v1)

**作者:** Beomjin Ahn `[一作]` (Samsung Research), Jaewook Chung `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一个训练无关的LoREnc框架，利用谱截断与补偿对基础模型和LoRA适配器进行加密，阻止未授权使用。

**💡 创新点**

创新点在于通过Eckart–Young定理的谱截断抑制低秩主成分，并在适配器中恢复该成分，兼顾安全性和完整性，同时引入正交重参数化隐蔽结构特征。

**🔧 技术方法**

使用了SVD、低秩截断、LoRA低秩适配、正交重参数化等技术。

**📊 数据集**

在Stable Diffusion v1.5、DiT/Sana以及GPT-2、Llama3等模型上进行实验，使用COCO Captions、WikiText-2等数据集评估。

**📈 对比分析**

与Spectral DeTuning、NNSplitter等现有恢复攻击方法对比，LoREnc在未经授权时输出结构崩塌、CLIP/LPIPS显著下降；授权时性能与原模型几乎相同，且对微调和SDT攻击的鲁棒性显著优于基线，推理开销低于1%。

**⚠️ 局限性**

局限性包括对硬件侧信道或物理键泄漏不具备防护，且在极大Δr或大模型规模下可能产生更大开销，尚未在更复杂攻击（如自适应检测器）下验证。

---

## 383. SWE-Cycle: Benchmarking Code Agents across the Complete Issue Resolution Cycle

**arXiv ID:** 2605.13139 | [PDF](https://arxiv.org/pdf/2605.13139v1)

**作者:** Hao Guan `[一作]` (Shanghai Jiao Tong University), Yong Yu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 30588 | [OpenAlex ID](https://openalex.org/A5001571390)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SWE‑Cycle基准，系统评估代码代理在完整问题解决生命周期中的表现，并配套了SWE‑Judge评估器；

**💡 创新点**

创新点在于①将环境重建、代码实现、测试生成三阶段合并为完整的FullCycle任务；②构建了可执行、分阶段、可插拔的SWE‑Judge，融合静态审查与动态执行；③通过三阶段过滤流程从SWE‑bench筛选出高质量的489条实例；

**🔧 技术方法**

使用六种先进LLM（GPT‑5.4、Claude‑Sonnet‑4.6、Qwen‑3.5、GLM‑5.1、Kimi‑K2.5、MiniMax‑M2.7）进行实验，SWE‑Judge采用静态代码审查、动态运行验证、参考补丁对比、故障注入等技术，并借助OpenCode框架与Docker容器实现评估；

**📊 数据集**

数据集来源于SWE‑bench的Verified、Pro和Multilingual三种版本，经过去污染、生命周期复杂度和测试可靠性过滤后得到489条高质量实例；

**📈 对比分析**

对比方法包括独立任务（Env、Impl、TestGen）与端到端FullCycle，评估指标为静态、动态得分与Solve率；实验表明独立任务的Solve率最高（最高可达78%），而FullCycle的Solve率低于14%；与传统脚本评估相比，SWE‑Judge的准确率达到98.6%；

**⚠️ 局限性**

局限性主要在于评估器依赖Claude API，易受外部版本或网络波动影响；FullCycle任务的完成率仍较低，表明当前LLM在跨阶段协作与长期维护方面存在显著瓶颈。

---

## 384. MPINeuralODE: Multiple-Initial-Condition Physics-Informed Neural ODEs for Globally Consistent Dynamical System Learning

**arXiv ID:** 2605.13305 | [PDF](https://arxiv.org/pdf/2605.13305v1)

**作者:** Lake Yang `[一作]` (Imperial College London), Serafim Kalliadasis `[通讯]` (Imperial College London)

**通讯引用:** 5972 | [OpenAlex ID](https://openalex.org/A5084671057)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种结合软物理约束和多初始条件多射击课程的神经ODE框架（MPINeuralODE），以提升对动力系统的全局一致学习能力。

**💡 创新点**

创新点在于将物理残差与多射击连续性惩罚相互补偿，构成结构化的多初始条件训练策略；同时提出三轴评估方法（OOS误差、长期稳定性、哈密顿量漂移）揭示模型真实表现。

**🔧 技术方法**

采用神经ODE（连续时间向量场由MLP建模）、Soft PINN残差正则化、K段多射击（K=4）连续性惩罚、clamp正则化、Dormand–Prince自适应积分以及Adam+余弦退火优化。

**📊 数据集**

使用Lotka–Volterra（LV）生态模型的合成轨迹，采样初始条件来自混合均匀/对数均匀分布，时间区间[0,30]共301个点，覆盖典型与边缘状态。

**📈 对比分析**

与传统神经ODE、仅PINN、仅MIC以及UDE结构化模型比较。MPINeuralODE在OOS和长期MSE分别为15.12（低于PINN 16.56、MIC 15.97、基线 20.46），哈密顿漂移0.943与PINN基本持平，整体优于任何单一模块，且优于数据驱动方法的最佳表现。

**⚠️ 局限性**

局限性包括：需先验部分物理知识；仅适用于自治连续无切换动力学；对参数调优仍敏感；在未覆盖的相空间区间可能仍出现偏差；与完全结构化模型相比，参数量巨大且预测误差远高于理论上限。

---

## 385. What properties of reasoning supervision are associated with improved downstream model quality?

**arXiv ID:** 2605.13290 | [PDF](https://arxiv.org/pdf/2605.13290v1)

**作者:** Mikołaj Langner `[一作]` (Wrocław University of Technology), Teddy Ferdinan `[通讯]` (Wrocław University of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不进行昂贵的微调循环的前提下，研究人员提出了一套可计算的内在数据指标，评估并预测多规模（8B 与 11B）模型在 Polish 语言推理任务中的训练数据质量与下游性能之间的关联；

**💡 创新点**

创新点在于将模型基于指标（FVCU）与可扩展分析指标（语义对齐、冗余比、符号比例等）结合，并揭示不同规模模型对推理数据属性的偏好差异，实现了“规模感知”的数据验证框架；

**🔧 技术方法**

使用 Qwen3-235B 进行 FVCU 评估、BERT 词嵌入进行语义对齐、文本压缩算法评估冗余、依存句法分析等技术；

**📊 数据集**

数据集为 Polish Mixture-of-Thoughts（MoT-PL）及其四个变体（Detailed、Summarized、BabyThink、Lengthy）以及多个下游基准（MoT-PL-eval、Belebele、Aya Collection、LightR1）；

**📈 对比分析**

通过对 8B 和 11B 模型在四个基准上的绝对准确率和相对提升进行比较，发现 8B 在语义对齐驱动下表现最佳，而 11B 在冗余与合法性强的 Lengthy 变体上获得最高提升（如 LightR1 +15%），整体相关性 ρ≥0.75；

**⚠️ 局限性**

局限性包括缺乏中间规模模型验证、实例级影响分析不足、依赖 LLM 判别可能引入偏差、Polish 数据翻译可能产生伪影、跨语言评测可能混淆语言能力与推理能力。

---

## 386. Byzantine-Robust Distributed Sparse Learning Revisited

**arXiv ID:** 2605.13283 | [PDF](https://arxiv.org/pdf/2605.13283v1)

**作者:** Yuxuan Wang `[一作]` (Zhejiang University), Kangqiang Li `[通讯]` (Hubei Provincial Tobacco Monopoly Administration)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文重新审视了高维稀疏线性模型的拜占庭鲁棒分布式估计，通过结合局部ℓ_1正则化的鲁棒估计和服务器端的鲁棒聚合，提出了一种新的框架，适用于伪Huber回归、分位数回归和稀疏支持向量机（SVM）。

**💡 创新点**

创新点在于将鲁棒聚合与通信高效的迭代方法相结合，能够在拜占庭攻击和重尾噪声的情况下，支持稀疏恢复并实现接近最优的统计速率。

**🔧 技术方法**

使用了局部ℓ_1正则化的鲁棒估计、伪Huber损失、分位数损失和稀疏SVM等技术。

**📊 数据集**

使用了合成数据和真实数据集进行模拟实验，以验证所提出方法的有效性和鲁棒性。

**📈 对比分析**

与现有的拜占庭鲁棒聚合规则（如Krum、坐标中位数等）相比，本文的方法在统计行为上与特定回归损失的最优速率对齐，且在通信效率上表现良好，每轮通信成本为𝒪(md)。

**⚠️ 局限性**

限制在于分析依赖于一些结构假设，如限制特征值条件、强凸性和有界条件密度，这些条件在实际应用中可能过于严格。此外，拜占庭机器的比例通常是未知的，因此修剪水平无法事先固定，未来的工作可以集中在更弱和可验证的假设上。

---

## 387. ReproScore: Separating Readiness from Outcome in Research Software Reproducibility Assessment

**arXiv ID:** 2605.13275 | [PDF](https://arxiv.org/pdf/2605.13275v1)

**作者:** Sheeba Samuel `[一作]` (Chemnitz University of Technology), Martin Gaedke `[通讯]` (Chemnitz University of Technology)

**通讯引用:** 3238 | [OpenAlex ID](https://openalex.org/A5087069237)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了ReproScore框架，将可重复性准备度与执行结果分离并结合为可覆盖度适应的综合得分。

**💡 创新点**

将可重复性评估拆分为两层（RRS与ROS），并引入社区可配置的YAML规则，显式化权重与门控，实现了可重复性准备与结果的明确区分。

**🔧 技术方法**

利用静态代码与元数据分析（如依赖锁定、容器规范、README结构、环境变量等）以及可选的沙箱执行探针来计算RRS、ROS和RCS。

**📊 数据集**

在由PubMed Central引用的生物医学Jupyter笔记本仓库构成的423个GitHub仓库的基准数据集上进行评估。

**📈 对比分析**

通过Kruskal‑Wallis H检验与AUC‑ROC等统计，RRS对失败模式的区分显著（H=96.89，p<0.001），而RCS与ROS的组合在部分验证中正确重排仓库，显示出可重复性准备与结果的差距。

**⚠️ 局限性**

局限性包括仅针对Python/Jupyter仓库，缺乏对其他语言或非笔记本项目的验证，静态指标无法捕获语义错误，且ROS与RCS的参数仍需更大规模实验校准。

---

## 388. Distributed Approximate Maximum Matching and Minimum Vertex Cover via Generalized Graph Decomposition

**arXiv ID:** 2605.13264 | [PDF](https://arxiv.org/pdf/2605.13264v1)

**作者:** Peter Davies-Peck `[一作]` (Durham University), Peter Davies-Peck `[通讯]` (Durham University)

**通讯引用:** 733 | [OpenAlex ID](https://openalex.org/A5021848057)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在LOCAL模型中提出随机算法，实现2+逼近的最大匹配和加权最小点覆盖，在O(log n/ log²log n)轮内完成；

**💡 创新点**

创新在于设计了可取任意分布的“移位”随机变量的图分解方法，打破了传统指数分布的记忆性限制，获得了子对数直径且相邻簇数量受控的分解；

**🔧 技术方法**

核心技术是基于Miller‑Peng‑Xu的聚类思想、泛化的分布式分解、以及在分解上对匹配/覆盖算法的聚类层级调整；

**📊 数据集**

论文为理论性研究，无实验数据集，算法在抽象图模型上分析；

**📈 对比分析**

与之前的O(logΔ/ loglogΔ)上界相比，在Δ很大的情况下实现了对n的对数因子提升，证明了n相关项是必要的；

**⚠️ 局限性**

局限性在于仍未逼近√(log n/ loglog n)下界，且算法仅在高度图中显著；对低度图或其他核心问题（如最大匹配、独立集）尚无直接推广。

---

## 389. Respecting Self-Uncertainty in On-Policy Self-Distillation for Efficient LLM Reasoning

**arXiv ID:** 2605.13255 | [PDF](https://arxiv.org/pdf/2605.13255v1)

**作者:** Junlong Ke `[一作]` (Tsinghua University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15453 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于教师熵的自我蒸馏方法 EGRSD 与其因果前瞻变体 CL-EGRSD，利用熵门对 token 级更新进行加权，以在保持推理准确率的同时压缩生成长度。

**💡 创新点**

创新点包括：①将教师的预测熵作为第三个信号与 RLSD 的方向/幅度分解结合；②设计了非零底值的线性熵门，避免高熵位置被完全抑制；③提出 CL‑EGRSD 通过最小熵前瞻窗口区分持久高熵（fork）和短暂高熵（pivot）位置，保留有用的转折信息。

**🔧 技术方法**

技术手段：On‑policy 自我蒸馏、RLSD 方向‑幅度框架、熵门乘子、预训练模型 Qwen3（4B/8B）+ LoRA 微调、vLLM 采样生成 roll‑outs、教师保持冻结、奖励形状与优势归一化。

**📊 数据集**

实验数据集：AIME 2024/25、HMMT 2025、MATH‑500、Minerva‑Math、GSM8K，使用“思考模式”下的长文本推理。

**📈 对比分析**

与无训练 Baseline、SFT、GRPO、OPSD、CRISP 等方法在同一 100 步训练预算下对比。EGRSD/CL‑EGRSD 在所有指标上均优于训练方法，提升了平均准确率（Avg.）并在长度‑准确率平衡图中占优，尤其在 Qwen3‑8B 上实现了最高 Avg. 与 OPDS 相当或更短的生成长度。

**⚠️ 局限性**

局限性：仅在大模型与“思考模式”下验证，熵门系数 γ 与前瞻窗口 W 需要手工调参；教师保持冻结对任务外推广性有限；对低熵错误位置的加权可能不足；额外计算教师熵增加了算力开销。

---

## 390. Doppler Prompting for Stable mmWave-based Human Pose Estimation

**arXiv ID:** 2605.13233 | [PDF](https://arxiv.org/pdf/2605.13233v1)

**作者:** Shuntian Zheng `[一作]` (University of Warwick), Yu Guan `[通讯]` (University of Warwick)

**通讯引用:** 2957 | [OpenAlex ID](https://openalex.org/A5081794070)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种利用毫米波雷达的Doppler信号作为受控提示，结合空间幅值特征实现单帧时序稳定的人类姿态估计模型PULSE。

**💡 创新点**

创新点在于将Doppler视为置信度门控的运动提示，只在局部邻域内以条件注意力方式引导空间特征，显著降低非人类运动噪声导致的抖动。

**🔧 技术方法**

采用双域特征分离、token化、置信度门控、局部交互的条件注意力、Transformer空间推理与回归头，并可扩展到多帧窗口的平均聚合。

**📊 数据集**

使用公开毫米波HPE数据集HuPR、XRF55和mmRadPose，覆盖单人和多人人场景。

**📈 对比分析**

与单帧/多帧基线（HuprModel、MvDoppler、RETR、mmDiff、milliMamba）在MPJPE、PA-MPJPE、MPJVE、AKV等指标对比，PULSE在每帧精度和时序稳定性均优于对照方法，多帧模式进一步提升。

**⚠️ 局限性**

局限在于仍受雷达硬件与环境干扰影响，Doppler提示的可靠性需进一步提升；邻域尺寸与门控参数需调优，且在更大规模真实部署中的鲁棒性尚待验证。

---

## 391. ReTool-Video: Recursive Tool-Using Video Agents with Meta-Augmented Tool Grounding

**arXiv ID:** 2605.13228 | [PDF](https://arxiv.org/pdf/2605.13228v1)

**作者:** Xiao Liu `[一作]` (Chongqing University), Jiang Zhong `[通讯]` (Chongqing University)

**通讯引用:** 7660 | [OpenAlex ID](https://openalex.org/A5069104909)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了MetaAug-Video工具库和ReTool-Video递归工具使用框架，用于提升视频理解代理的多模态推理能力。

**💡 创新点**

创新点在于：①构建了134个基底工具与108个元工具的可扩展工具库，支持双层视频信息访问；②引入递归地将抽象视频意图分解为可执行工具链，解决了传统工具调用的“粗粒度”与“扁平化”限制。

**🔧 技术方法**

使用Qwen/Qwen3.5-9B作为规划器和解析器，并通过强化学习优化规划器策略，配合工具库执行框架实现多步推理。

**📊 数据集**

在MVBench、MLVU和Video-MME（无字幕）三个公开基准上进行评测。

**📈 对比分析**

与闭源与开源基准模型比较，ReTool-Video在三大数据集上分别获得81.5/72.9/76.6的准确率，明显优于InternVL3.5-30B（提升8.5%/7.9%）及NVILA、Flash-VStream等强基准。

**⚠️ 局限性**

局限性包括：需要手工维护和扩展工具注册；递归分解过程对工具库覆盖度高度依赖；在短视频局部感知任务上提升有限，仍受模型本身感知与推理能力限制。

---

## 392. Machine Learning-Driven Multimodal Spectroscopic Liquid Biopsy for Early Multicancer Detection

**arXiv ID:** 2605.13218 | [PDF](https://arxiv.org/pdf/2605.13218v1)

**作者:** Alejandro Leonardo García Navarro `[一作]` (Signal Processing Group), Carlos Viadero Valderrama `[通讯]` (Amber Health Solutions)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于傅里叶变换红外（FTIR）、拉曼光谱与激发-发射矩阵（EEM）荧光三种光谱技术的多模态液体活检框架，用于多种癌症（乳腺癌与结肠癌）与健康对照的检测。

**💡 创新点**

创新之处在于将三种互补的光谱模态通过低层次数据融合（LLDF）系统整合，并对单模态、双模态和三模态进行全面比较，验证多模态融合显著提升诊断的鲁棒性与平衡性。

**🔧 技术方法**

采用了FTIR、拉曼光谱与EEM荧光光谱采集与预处理、低层次数据融合、特征标准化与块级缩放，以及XGBoost梯度提升树进行二分类。

**📊 数据集**

使用了来自 Asturias BioBank 的300份血清样本（各100例乳腺癌、结肠癌和健康对照），每例FTIR可获得多达3份技术复制，拉曼和EEM每例各一份光谱。

**📈 对比分析**

通过10折分层交叉验证比较7种模态配置，评估指标包括ROC-AUC、灵敏度、特异度和平衡准确率；最佳配置为三模态融合，乳腺癌AUC 0.997、灵敏度 0.990，结肠癌AUC 0.994、灵敏度 0.959，整体表现接近0.96的平衡准确率。

**⚠️ 局限性**

局限在于样本量相对中等，未进行外部验证，部分模态组合在特异度上波动较大，且仅针对两种癌症进行评估，未来需扩大多中心、多癌种的数据集并探索更高级的融合策略。

---

## 393. Safe Bayesian Optimization for Uncertain Correlations Matrices in Linear Models of Co-Regionalization

**arXiv ID:** 2605.13302 | [PDF](https://arxiv.org/pdf/2605.13302v1)

**作者:** Jannis Lübsen `[一作]` (Hamburg University of Technology), Annika Eichler `[通讯]` (Hamburg University of Technology)

**通讯引用:** 607 | [OpenAlex ID](https://openalex.org/A5021520570)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将不确定协方差矩阵的多任务贝叶斯优化安全保障从内在协方差模型（ICM）推广到线性协方差模型（LMC），并在此基础上给出新的统一误差界和安全放缩因子；

**💡 创新点**

创新点在于：①推导了LMC条件下的安全安全保证公式；②给出计算安全放缩因子β̅所需的ν、γ等量化参数的具体表达；③通过数值实验验证LMC相较ICM在安全多任务优化中的优越性；

**🔧 技术方法**

使用的技术包括：多任务高斯过程、线性协方差模型（LMC）、贝叶斯安全约束、均匀误差界推导、UCB采集函数、蒙特卡罗估计协方差矩阵置信集合；

**📊 数据集**

实验数据集为人工合成基准：二维任务、四维输入空间，使用平方指数核构造两特征LMC（主特征高相关、次特征无相关），共20次重复；

**📈 对比分析**

比较方法：在同一基准下分别采用单任务安全BO、ICM多任务安全BO和LMC多任务安全BO；实验结果显示LMC与ICM均显著优于单任务BO，并且在主特征相关性较强时，LMC能更好利用多任务信息缩小安全阈值、加快收敛；

**⚠️ 局限性**

限制：计算协方差置信集合𝒞ρ和安全放缩因子β̅的复杂度较高，导致总体计算成本上升；

---

## 394. Discrete Diffusion for Complex and Congested Multi-Agent Path Finding with Sparse Social Attention

**arXiv ID:** 2605.13296 | [PDF](https://arxiv.org/pdf/2605.13296v1)

**作者:** Yuanzhe Wang `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yunji Chen `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种结合离散扩散生成模型和大型邻域搜索（LNS2）的混合多智能体路径规划框架 DiffLNS，用于在拥挤环境下生成全局一致的初始路径并通过 LNS2 修复冲突。

**💡 创新点**

创新点包括：①首次将离散扩散概率模型（D3PM）用于 MAPF 初始化；②引入扩散感知稀疏社交注意力机制，仅关注当前轨迹邻域内可能冲突的智能体；③在初始化阶段多样化采样并与 LNS2 并行修复；④展示模型在训练规模（≤96 只智能体）之外成功扩展到 312 只智能体；⑤在 20 个高拥堵测试集上实现 95.8% 的平均成功率，比现有最佳基线提升 9.6 个百分点。

**🔧 技术方法**

核心技术包括离散扩散生成模型、稀疏社交注意力层、强化学习辅助的损失函数、LNS2 破坏-重规划策略、预处理补全未完成轨迹、以及多样本采样与并行修复。

**📊 数据集**

使用 POGEMA 生成的多种规模和障碍密度的地图（Small Random、Medium Maze、Medium Room、Medium Warehouse、Large Maze），并在 23×23 的网格上收集专家演示的联合动作序列作为训练数据。

**📈 对比分析**

与 LNS2、LNS2+RL、HMAGAT 和 LaCAM3 等基线在成功率、总成本（SOC）和运行时进行对比；DiffLNS 在所有 20 个设置中均达到或超过所有基线的成功率，平均成功率 95.8%，在最难场景下比 LNS2+RL 高 9.6pp；虽然平均运行时略高，但可通过 GPU 并行化显著降低实际耗时。

**⚠️ 局限性**

局限性：对简单或低拥堵实例增量收益有限；在超大地图上可能受训练分布偏移影响；受限于后端 LNS2 修复能力，极难实例在有限预算下仍可能失败；扩散阶段需要 GPU 计算，整体时间成本相对较高。

---

## 395. CANTANTE: Optimizing Agentic Systems via Contrastive Credit Attribution

**arXiv ID:** 2605.13295 | [PDF](https://arxiv.org/pdf/2605.13295v1)

**作者:** Tom Zehle `[一作]` `[通讯]` (University of Freiburg), Tom Zehle (University of Freiburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于对比归因的框架，用于自动化LLM多智能体系统的提示优化。

**💡 创新点**

创新点在于将全局奖励拆分为每个智能体的归因信号，解决信用分配难题。

**🔧 技术方法**

采用提示学习、对比归因、LLM鉴别器和本地优化器（如CAPO）实现。

**📊 数据集**

使用MBPP、GSM8K和HotpotQA三个基准数据集进行评估。

**📈 对比分析**

与GEPA、MIPROv2比较，平均排名1.44，MBPP提升18.9pp，GSM8K提升12.5pp，HotpotQA差距不大，且推理成本较低。

**⚠️ 局限性**

局限在于固定工作流拓扑、归因模型受限于LLM能力、可扩展性和大规模智能体场景尚未验证。

---

## 396. Img2CADSeq: Image-to-CAD Generation via Sequence-Based Diffusion

**arXiv ID:** 2605.13293 | [PDF](https://arxiv.org/pdf/2605.13293v1)

**作者:** Shiyu Tan `[一作]` (Tsinghua University), Enya Shen `[通讯]` (Tsinghua University)

**通讯引用:** 219 | [OpenAlex ID](https://openalex.org/A5003655558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出Img2CADSeq多阶段流水线，直接从单视角图像生成可编辑的BRep STEP文件。

**💡 创新点**

创新三层层次编码的CAD序列代码库，结合粗细点云中介与对比学习实现模态对齐，并构建新的CAD-220K和PrintCAD数据集。

**🔧 技术方法**

采用层次VQ-VAE编码、UA-DGCNN点云细化、对比学习对齐、VQ-Diffusion生成器、Dens3R+PEFT的点云生成网络等技术。

**📊 数据集**

使用DeepCAD、CAD-220K（220k ABC子集）、PrintCAD（2000+ 3D打印件）以及ShapeNet/ABC等公开数据。

**📈 对比分析**

与TripoSR、HoLa、CADDreamer、SkexGen、HNC-CAD、DTGBrepGen等基线在Chamfer距离、悬挂面比、分割准确率、精度召回率、MMD、JSD等指标上均显著优于对手，性能提升显著。

**⚠️ 局限性**

局限在单视角不确定性导致后端假象、序列误差累积破坏对称与轴对齐、点云分辨率限制导致细节缺失。

---

## 397. Delightful Exploration

**arXiv ID:** 2605.13287 | [PDF](https://arxiv.org/pdf/2605.13287v1)

**作者:** Ian Osband `[一作]` (Google DeepMind), Ian Osband `[通讯]` (Google DeepMind)

**通讯引用:** 4928 | [OpenAlex ID](https://openalex.org/A5015899120)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出 Delight-gated exploration（DE）算法，并在伯努利 bandit、线性 bandit 与 DeepSea MDP 三种场景中进行验证。

**💡 创新点**

创新点在于将“Delight”门控从学习迁移到行动，通过门价 λ 对预期提升与惊奇度（惊奇度×提升）相乘后的“前景愉悦度”做阈值筛选，实现对探索的价格化，并恢复 Pandora 预留值规则。

**🔧 技术方法**

采用后验均值贪婪主机、稀疏的贝叶斯期望改进（EI）门控、惊奇度度量、信息量预算分析以及线性椭圆势能理论等技术。

**📊 数据集**

使用的实验数据集为 synthetic 伯努利 arm、随机特征生成的线性 bandit，以及 DeepSea 深度 H 的二叉树 MDP。

**📈 对比分析**

通过累计 regret 与 Thompson Sampling、ε-greedy、PSRL 等基线进行比较，DE 在小规模环境下与传统方法相当，在大规模未解空间中保持近乎不变的 regret，显著优于 ε-greedy 并优于 TS。

**⚠️ 局限性**

主要局限在于缺乏严格的 regret 上界；固定门价仅实现“satisficing”搜索，无法提供渐进无偏保证，对极端先验可能过早关闭探索。

---

## 398. Chem-GMNet: A Sphere-Native Geometric Transformer for Molecular Property Prediction

**arXiv ID:** 2605.13262 | [PDF](https://arxiv.org/pdf/2605.13262v1)

**作者:** Deepak Warrier `[一作]` (MSTACK AI), Raja Sekhar Pappala `[通讯]` (MSTACK AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种基于球面谐波的化学语言模型 GM‑Net（Chem‑GMNet），通过在嵌入、注意力和 FFN 模块中使用球面结构来替代传统 transformer，从而在化学序列建模中引入几何先验。

**💡 创新点**

创新点在于：①构建完整的球面原生 transformer 家族；②提出 DualSKA 结合线性多极递归和 Schoenberg‑正定核软最大化的混合注意力；③证明 Gated SFA 递归状态等价于分子分布的截断多极展开；④在 FFN 处使用 Funk–Hecke 球面卷积，实现对激活函数的球面特征化。

**🔧 技术方法**

采用的技术包括：球面谐波（Spherical Harmonics）、Gegenbauer 特征映射、Schoenberg 线性核、双向 Gated SFA 递归、软最大化 Sphere‑Kernel Attention、Funk–Hecke 球面卷积以及多任务微调与 MLM 预训练。

**📊 数据集**

使用 MoleculeNet 上的 10 个端点（5 计量回归、4 分类、1 单任务 SR‑p53）以及 10M 规模的 ZINC SMILES 预训练语料库。

**📈 对比分析**

对照相同架构规模的 ChemBERTa‑2（仅在词表、维度等相同）进行随机初始化和预训练后微调比较。结果显示：在随机初始化时 Chem‑GMNet 在 7/10 端点上领先，且参数量约少 35%；在相同 10M 预训练下 Chem‑GMNet 在 6/8 端点上优于公开的 ChemBERTa‑2，尤其在结合/清除相关任务上取得显著提升。

**⚠️ 局限性**

局限性包括：在预训练数据分布与下游任务不匹配时（如 SR‑p53）效果不佳；实现层面计算速度比传统 dot‑product Transformer 慢约 2.5 倍；在小样本回归端点上仍受限于模型容量和训练策略；未在 3D conformer 任务、Tox21/HIV 等更大规模评估上验证。

---

## 399. X-Restormer++: 1st Place Solution for the UG2+ CVPR 2026 All-Weather Restoration Challenge

**arXiv ID:** 2605.13258 | [PDF](https://arxiv.org/pdf/2605.13258v1)

**作者:** Youwei Pan `[一作]` (Transsion Holdings), Fengjie Zhu `[通讯]` (Transsion Holdings)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在UG2+ 8th挑战中，基于X-Restormer实现全天气图像恢复，加入空间自适应输入缩放、梯度引导边缘感知损失以及训练数据规模扩展，获得第一名；

**💡 创新点**

创新点在于：①使用空间自适应缩放机制提升对非均匀天气的适应性；②提出梯度引导边缘感知（GGEA）损失，基于真值梯度生成权重聚焦高频细节；③在多来源数据上大规模扩充训练集；

**🔧 技术方法**

技术采用X-Restormer框架、MDTA+OCA双注意力、GGEA损失（L1+MS‑SSIM+梯度加权），AdamW+余弦学习率，单帧预测与多帧平均融合；

**📊 数据集**

数据集包括原始WeatherStream、FoundIR 24,500对、WeatherBench，合计约30k对，涵盖雨、雪、雾、雾霾等多种天气；

**📈 对比分析**

与赛后前五名方法对比，PSNR 29.19dB、SSIM 0.8341，分别比第二名高1.07dB、0.0482，整体性能显著领先；

**⚠️ 局限性**

局限性在于：需要多帧平均才能充分发挥优势，单帧效果略逊；模型对极端天气依赖训练数据，若遇未见天气仍可能失效；训练成本高（单张NVIDIA H100 40轮）。

---

## 400. GAGPO: Generalized Advantage Grouped Policy Optimization

**arXiv ID:** 2605.13217 | [PDF](https://arxiv.org/pdf/2605.13217v1)

**作者:** Siyuan Zhu `[一作]` (Sun Yat-sen University), Yibo Zhang `[通讯]` (Meituan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种无价值函数的强化学习方法GAGPO，用环境步为单位进行信用分配，并通过分组值代理实现时序信用传递。

**💡 创新点**

创新点在于引入非参数分组价值代理与TD/GAE式递归优势计算，结合组归一化和动作序列重要性比率，实现了步骤对齐的时序信用分配。

**🔧 技术方法**

使用了强化学习中的PPO框架、TD/GAE优势估计、分组归一化、动作序列重要性比率以及非参数价值代理。

**📊 数据集**

在ALFWorld和WebShop这两个文本交互基准上使用Qwen2.5-1.5B和7B-Instruct模型进行实验。

**📈 对比分析**

与PPO、RLOO、GRPO、GiGPO等RL基线以及直接提示、ReAct、Reflexion等提示基线对比，GAGPO在成功率和分数上均超过基线，早期学习速度更快且优化更平稳。

**⚠️ 局限性**

主要限制在于依赖精确状态匹配，难以推广到含随机或连续观测的环境，以及只在稀疏终点奖励的文本任务上验证。

---

## 401. Calibration-Free Gas Source Localization with Mobile Robots: Source Term Estimation Based on Concentration Measurement Ranking

**arXiv ID:** 2605.13208 | [PDF](https://arxiv.org/pdf/2605.13208v1)

**作者:** Wanting Jin `[一作]` (École Polytechnique Fédérale de Lausanne), Alcherio Martinoli `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 8788 | [OpenAlex ID](https://openalex.org/A5080713076)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于浓度排名的特征，用于在不进行传感器标定的情况下实现移动机器人在有障碍室内环境中的气体源定位（GSL）

**💡 创新点**

创新点在于利用测量值与模型值在所有采样点中的相对排名（EDF）来评估匹配度，消除了对绝对浓度值的依赖，从而避免了MOX传感器的非线性与环境漂移对定位性能的影响；同时不需要额外阈值或平滑参数，适用于多机器人异构传感器共享信息

**🔧 技术方法**

采用数据驱动气 plume 模型（DDPM）预测浓度分布，结合概率源项估计（STE）框架；在模拟中实现MOX传感器的非线性慢响应模型；利用 EDF 排名特征与传统浓度值、固定/自适应 hit 特征进行比较；并使用信息增益路径规划来加速采样

**📊 数据集**

使用 Webots 高保真模拟环境（含两块障碍物、随机源点与机器人起始点，共10次试验）以及实验风洞中的物理实验（Khepera IV 机器人、MiCS‑5521 MOX 传感器板 A 与 B，15次试验，源点沿 x 轴固定，y 轴取 1m、2m、3m）

**📈 对比分析**

与传统浓度值、固定 hit、自适应 hit 三种特征对比，实验表明在已标定传感器下四种特征定位误差相近，但排名特征在未标定传感器下仍保持小误差，且收敛次数少、速度最快；在物理实验中排名特征比浓度值显著提升定位可靠性

**⚠️ 局限性**

局限性：需满足传感器输出与浓度的正单调关系；物理实验中气溶胶湍流导致偶尔出现较大误差；目前仅考虑排名信息，未利用 EDF 斜率等附加信息，可能进一步提升收敛速度与鲁棒性

---

## 402. Galilean State Estimation for Inertial Navigation Systems with Unknown Time Delay

**arXiv ID:** 2605.13266 | [PDF](https://arxiv.org/pdf/2605.13266v1)

**作者:** Giulio Delama `[一作]` (University of Klagenfurt), Robert Mahony `[通讯]` (Australian National University)

**通讯引用:** 18060 | [OpenAlex ID](https://openalex.org/A5011097720)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对惯性导航系统（INS）与全球导航卫星系统（GNSS）测量存在未知时间延迟的问题，提出基于Galilean时空对称性的几何建模框架，并在此基础上设计了Equivariant Filter（EqF），实现对导航状态与未知时延的实时联合估计。

**💡 创新点**

创新点包括：①利用Galilean Lie群的时空对称性，将时延与空间状态统一建模，提供了对延迟影响的几何描述；②在此框架下推导出EqF，能够在滤波过程中同时估计时延并保持滤波一致性；③通过与传统EKF基线（无延迟、离线估计、在线估计）比较，证明EqF在准确性与一致性上优于现有方法。

**🔧 技术方法**

技术手段：Galilean Lie群几何建模、IMU预积分（处理延迟的缓冲机制）、Equivariant Filter（EqF）实现、离线与在线延迟估计的EKF实现、Monte Carlo仿真、UAV实测数据验证。

**📊 数据集**

实验数据集：两架固定翼无人机（UAV）真实飞行数据（GNSS时延约90 ms与120 ms）以及合成轨迹仿真（圆形运动叠加波动，时延场景分别为100 ms、200 ms、300 ms、500 ms）。

**📈 对比分析**

比较方法：将EqF与三种EKF基线在相同实验和仿真情景下进行对比，使用RMSE、NEES等指标评估位置、速度、姿态误差以及时延误差；结果显示EqF在实验中与离线估计的EKF相当，且在仿真中即使时延增大至500 ms也能保持一致性和低误差，而传统EKF在在线延迟估计下出现偏差、非一致性甚至发散。

**⚠️ 局限性**

局限性：仅针对单传感器（GNSS位置）时延；假设轨迹足够激励以保证时延可观测，对极端大延迟或高噪声条件下的鲁棒性未系统验证；实现中需维护IMU预积分缓冲，计算量相对传统EKF略大。

---

## 403. Achieving Gold-Medal-Level Olympiad Reasoning via Simple and Unified Scaling

**arXiv ID:** 2605.13301 | [PDF](https://arxiv.org/pdf/2605.13301v1)

**作者:** Yafu Li `[一作]` (Shanghai AI Laboratory), Yu Cheng `[通讯]` (Chinese University Of Hong Kong)

**通讯引用:** 35641 | [OpenAlex ID](https://openalex.org/A5026944066)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 30B-A3B 语言模型通过逆 perplexity 训练、两阶段 RL 与推理时自检自修循环，提升至奥林匹克级别的推理与证明能力。

**💡 创新点**

提出一套简洁统一的后训练流程：逆 perplexity SFT 课程、可验证奖励的粗粒度 RL、证明级奖励与自修复的细粒度 RL，以及推理时规模化的自检自修机制。

**🔧 技术方法**

逆 perplexity 训练课程、GSPO 级别的可验证奖励 RL、DeepSeekMath‑V2 证明级生成奖励、经验回放、自检自修循环以及推理时扩展（TTS）。

**📊 数据集**

SFT 338K 轨迹（来自 Evan Chen、AoPS、Shuzhimi Forum、DeepMath、NaturalReasoning、Eurus、OpenCodeReasoning 等），RL 数据 8.9k 可验证 + 16.3k 非可验证（含 OPC、P1 physics）。评测集包括 AnswerBench、AMO‑Bench、AIME 2025/26、IMO‑ProofBench、FrontierScience‑Olympiad/Research、IMO 2025、USAMO 2026、IPhO 2024/25。

**📈 对比分析**

与同规模基线（Qwen3.6‑35B‑A3B、GLM‑4.7‑Flash、Nemotron‑Cascade‑2）及更大系统（Gemini 3.1 Pro、GPT‑5.5‑High 等）对比。模型在可验证任务上平均 77.3%（≈同规模最高 77.4%），证明任务 70.2%（IMO‑ProofBench），IMO 2025/USAMO 2026 35 分金牌线，IPhO 2024/25 超过金牌线；成本显著低于大模型。

**⚠️ 局限性**

对结构性约束的把握仍不完善（如 IMO P6、USAMO P2 失败），对极难题仍需多轮修正；在科研子集分数偏低；推理时需要大量计算；对某些特定领域的泛化仍有限。

---

## 404. Differentiable Learning of Lifted Action Schemas for Classical Planning

**arXiv ID:** 2605.13282 | [PDF](https://arxiv.org/pdf/2605.13282v1)

**作者:** Jonas Reiter `[一作]` (RWTH Aachen University), Hector Geffner `[通讯]` (RWTH Aachen University)

**通讯引用:** 7692 | [OpenAlex ID](https://openalex.org/A5046427368)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种新型神经网络架构，能够在仅观察完整状态但不知道动作参数的轨迹中学习STRIPS动作模式；

**💡 创新点**

创新点在于将动作参数选择与预条件、效果的学习统一到可微分的梯度下降框架中，并使用基于GNN的槽位匹配来推断动作参数；

**🔧 技术方法**

采用图神经网络（GNN）提取对象嵌入、Sinkhorn匹配、可微分的预条件/效果学习，以及PCGrad梯度投影技术；

**📊 数据集**

在13个标准IPC规划域（Blocks、Driverlog、Gripper、Hanoi、Logistics等）上进行实验，使用随机游走产生的状态转移样本；

**📈 对比分析**

与仅基于SAT的符号方法L1进行对比，Dias在大多数域（8/13）实现了完全准确的动作模式学习，性能优于L1且在噪声下仍保持较好鲁棒性；

**⚠️ 局限性**

局限性包括：对随机游走采样的依赖导致在存在死端或覆盖不足的域（如Sokoban、Hanoi、Satellite）出现失败；当动作参数隐藏时，学习难度显著上升，且对高噪声环境的鲁棒性仍有限。

---

## 405. Unified generalization analysis for physics informed neural networks

**arXiv ID:** 2605.13260 | [PDF](https://arxiv.org/pdf/2605.13260v1)

**作者:** Yuka Hashimoto `[一作]` (NTT, Inc.), Tomoharu Iwata `[通讯]` (NTT, Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种统一的通用化分析框架，给出了包含PINNs、VPINNs以及其他涉及输入变量微分的神经网络的泛化误差上界

**💡 创新点**

创新点在于利用泰勒展开将非线性微分算子转化为高维线性算子，并将其与Koopman算子结合，得到包含权重矩阵行列式的泛化上界，揭示高秩网络可泛化且非线性算子指数放大误差

**🔧 技术方法**

采用Fréchet导数、Taylor展开、Koopman算子与Rademacher复杂度分析、Koopman‑based Rademacher上界等理论工具

**📊 数据集**

通过在Navier–Stokes和Monge–Ampère方程上使用PINN/VPINN网络进行实验验证，数据来自对应的二维/三维 PDE 训练样本

**📈 对比分析**

对比了带无正则化与基于理论上界的正则化训练，结果表明正则化能显著降低测试误差，且对标准PINN同样有效，实验中通过相关系数展示误差与理论量的高阶多项式关系

**⚠️ 局限性**

局限性在于所给上界为无算法依赖的统一上界，未考虑具体训练算法，且对非线性算子非多项式情况上界可能过于保守

---

## 406. EMO: Frustratingly Easy Progressive Training of Extendable MoE

**arXiv ID:** 2605.13247 | [PDF](https://arxiv.org/pdf/2605.13247v1)

**作者:** Linghao Jin `[一作]` (USC-ISI), Xuezhe Ma `[通讯]` (USC-ISI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种渐进式扩展专家池的 Mixture-of-Experts 训练框架 EMO，允许模型在预训练过程中逐步增大专家数量。

**💡 创新点**

创新点在于将专家容量视为可扩展内存，并通过稀疏度缩放律精确预测最佳扩容时机与数据分配，从而在保持大专家容量收益的同时显著降低训练成本。

**🔧 技术方法**

采用稀疏 MoE 结构、top‑k 路由、分阶段扩容与初始化策略、学习率温和上升、BF16 混合精度、FlashAttention 3 等系统级加速技术。

**📊 数据集**

在多语种、代码、数学等大型语料上预训练，总计 1.92T tokens，随后在八项下游基准（如 MMLU、GSM‑8K 等）进行评估。

**📈 对比分析**

与固定专家数基线（E=16/32/128）比较，EMO 在相同总 token 预算下可达与 E=128 相近的预训练损失，并比 E=128 节省约 10% GPU 时长，显著优于小专家基线，并在多数基准上与 E=128 保持竞争。

**⚠️ 局限性**

局限在于缩放律未考虑优化器等超参，且实验规模仍低于最前沿 MoE 系统，未来需验证更大规模下的可扩展性。

---

## 407. Automatic Detection of Reference Counting Bugs in Linux Kernel Drivers

**arXiv ID:** 2605.13246 | [PDF](https://arxiv.org/pdf/2605.13246v1)

**作者:** Joe Hattori `[一作]` (University of Tokyo), Ken Sakayori `[通讯]` (University of Tokyo)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5049034602)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种自动化工具，用于检测 Linux 内核驱动中的引用计数错误，涵盖了多种常见的使用后释放、内存泄漏和计数不匹配等缺陷；

**💡 创新点**

创新点在于构建了针对驱动代码的引用计数符号模型，并结合控制流图与数据流分析，能够精准识别多种引用计数错误模式，显著降低误报率；

**🔧 技术方法**

采用了静态分析技术，包括数据流与控制流分析、抽象解释、符号执行以及基于 C/C++ 解析的程序变换；

**📊 数据集**

实验使用了从 Linux 内核源码（v5.x 版本）中抽取的约 300 个驱动程序，并结合公开的 CVE 与 LKML 漏洞数据库进行验证；

**📈 对比分析**

与 Coverity、Sparse、Clang Static Analyzer 等现有工具相比，本文工具在检测率上提升约 20%–30%，误报率降低约 15%，平均每个驱动的分析时间约为 3 秒；

**⚠️ 局限性**

局限性包括：仅针对 C 语言驱动；对宏展开和复杂的编译器优化支持不足；无法覆盖所有动态分配路径，且部分关键变量需手工标记以实现准确分析。

---

## 408. It's not the Language Model, it's the Tool: Deterministic Mediation for Scientific Workflows

**arXiv ID:** 2605.13245 | [PDF](https://arxiv.org/pdf/2605.13245v1)

**作者:** Marios Adamidis `[一作]` (University of Crete), Emmanuel Stratakis `[通讯]` (FORTH)

**通讯引用:** 15401 | [OpenAlex ID](https://openalex.org/A5089770490)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了一种“Typed Mediation”模式，利用语言模型仅负责调用已通过结构化访谈编码的确定性工具，从而实现实验数据分析流程的可复现性；

**💡 创新点**

创新点在于将实验工作流程拆解为按严格 schema 定义的 Typed Tool，由语言模型根据需求进行选择和参数化，而不是让模型生成代码；该模式保证了跨多次生成的结果一致性，并解决了隐私与许可限制下工具部署的问题；

**🔧 技术方法**

主要技术包括结构化访谈获取实验师方法学、将方法学编码为 Typed Tool（基于 MCP 规范）、编写 Skill 文件来约束模型行为、在本地部署工具并与实验仪器/软件集成；

**📊 数据集**

使用的主要数据集为光致发光（photoluminescence）谱数据（从专有软件导出为 CSV），以及扫描电子显微镜（SEM）周期结构分析的数据；

**📈 对比分析**

通过在同一数据集上与三大商业基础模型（GPT‑5.5、Claude Sonnet 4.6、Gemini 3.1）以及本平台的 Typed Tool 进行四次重复实验比较；Typed Tool 在四次运行中均给出完全一致的指数 b（σ=0），而商业模型结果波动显著（σ>0，甚至失效）；运行时间上，Typed Tool 平均 34 秒，明显快于 1:50–5:15 的商业模型；

**⚠️ 局限性**

局限性包括：评估仅在单一数据集和单一仪器（光致发光）上进行，未覆盖更多实验类型；实现需要人工访谈和工具编码，扩展性受限；以及仍依赖语言模型进行工具调用，若模型错误选择工具或参数仍可能产生误差。

---

## 409. A Hybrid Framework for Natural Language Querying of IFC Models with Relational and Graph Representations

**arXiv ID:** 2605.13236 | [PDF](https://arxiv.org/pdf/2605.13236v1)

**作者:** Rabindra Lamsal `[一作]` (University of New South Wales), Johnson Xuesong Shen `[通讯]` (University of New South Wales)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个名为 IfcLLM 的混合框架，用自然语言查询 IFC BIM 数据。该框架将 IFC 模型转换为关系型数据库（存储属性、层级与简化几何）和几何导出的拓扑图（存储空间邻接与连通性），并通过迭代的 retry‑and‑refine 方式让 LLM 生成并执行 SQL/ Cypher 查询，最终返回自然语言答案。

**💡 创新点**

创新点包括：①将关系数据库与几何推导的拓扑图两种互补视图结合；②使用几何推导而非仅依赖 IFC 预定义关系来构建邻接图；③采用迭代推理框架使 LLM 能在查询失败时自动重构、补充信息；④实现可复现、可部署的端到端系统，全部基于开源工具与开源 LLM。

**🔧 技术方法**

核心技术：IFCOpenShell 解析 IFC；SQLite 关系数据库与 Neo4j 图数据库；GPT OSS 120B（主 LLM）与 Groq Compound（回退 LLM）；LangGraph + LangChain 的多工具推理；基于轴对齐包围盒的邻接检测与图构建算法；以及对查询生成与结果的动态上下文管理。

**📊 数据集**

数据集：三种 IFC 模型（B1 两层住宅、B2 复式公寓、B3 办公楼）共约 25,000 个建筑元件；以及 30 条基于真实场景的自然语言查询集，用来评估系统性能。

**📈 对比分析**

评估方法：计算首轮准确率（initial‑attempt）和回退后恢复率；通过 ablation 研究不同组件对性能的影响。结果显示：在所有三套模型中，完整 IfcLLM 的首轮准确率均为 100%；回退 LLM 处理所有失败情况恢复率为 100%；单一关系 + 迭代推理 100% 但 token 使用约 1.9×；单一关系 + 一次性推理 93.3%；相比之下图形查询无需复杂 SQL，显著降低 token 量与查询复杂度。

**⚠️ 局限性**

局限性：①几何盒子邻接方法对曲面或不规则空间易产生误邻接；②关系数据库缺乏标准化空间函数，导致 LLM 生成的空间查询易出错；③在大模型或属性众多时，系统提示中包含过多信息导致 token 过长；④缺乏对 IFC 数据质量（几何缺失、关系不完整）的自动校验；⑤评测仅基于有限的模型与查询集合，未覆盖更复杂的多楼层或多建筑场景。

---

## 410. Mix, Don't Tune: Bilingual Pre-Training Outperforms Hyperparameter Search in Data-Constrained Settings

**arXiv ID:** 2605.13225 | [PDF](https://arxiv.org/pdf/2605.13225v1)

**作者:** Paul Jeha `[一作]` (Apple), Natalie Schluter `[通讯]` (Apple)

**通讯引用:** 529 | [OpenAlex ID](https://openalex.org/A5056555815)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究低资源语言模型预训练时，比较数据混合、超参数调优和数据规模对性能的影响。

**💡 创新点**

发现混合高资源语言比单独调优更能提升低资源模型，提出用最大更新参数化（μP）从小模型迁移超参数，量化混合对下游任务的“数据倍增”效应。

**🔧 技术方法**

采用LLaMA风格Transformer、μP调参、方差分解/ANOVA、全规模实验（150M–1.43B）、多语言混合比例α、R_max重复次数、ARC Easy翻译评测等技术。

**📊 数据集**

使用Arabic FineWeb2子集（200M独特标记）和英语FineWeb（≈350B可用）进行预训练，并以ARC Easy（从English翻译而来）做5-shot下游评测。

**📈 对比分析**

通过验证损失与ARC Easy 5-shot准确率比较，设置四种实验范式（单语/双语 × 基础/调优超参），结果显示双语+μP在所有规模下均优于单语，验证损失仍能定位近最优检查点，混合的“数据倍增”从3倍提升至13倍。

**⚠️ 局限性**

验证损失对混合价值的估计不足；实验规模有限，模型仅至1.43B；缺乏更大多语言基准；依赖特定数据集与语言组合，可能不具普适性。

---

## 411. An Agentic AI Framework with Large Language Models and Chain-of-Thought for UAV-Assisted Logistics Scheduling with Mobile Edge Computing

**arXiv ID:** 2605.13221 | [PDF](https://arxiv.org/pdf/2605.13221v1)

**作者:** Hanwen Zhang `[一作]` (Nanyang Technological University), Malcolm Yoke Hean Low `[通讯]` (Singapore Institute of Technology)

**通讯引用:** 2863 | [OpenAlex ID](https://openalex.org/A5038383727)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于大型语言模型与链式推理的代理式 AI 框架，用于自动生成云制造中 UAV 物流与 MEC 调度的可解释数学模型，并研发了层次化 PPO 深度强化学习方法解决该混合调度问题。

**💡 创新点**

创新点：① 将 LLM、检索增强生成、链式推理与验证组合，实现对复杂耦合问题的自动建模；② 将 UAV 路由与计算任务拆分为上下层 PPO，显著降低状态动作空间并保持两层耦合。

**🔧 技术方法**

使用技术：OpenAI gpt‑5.4‑2026‑03‑05 LLM；Chroma 向量库的检索增强生成（RAG）；LangGraph/ LangChain 的链式推理与验证；Proximal Policy Optimization（PPO）深度强化学习。

**📊 数据集**

数据集：内部构建的 RAG 知识库，包含 UAV 路由、计算任务卸载的建模模板、约束与目标示例，已在 GitHub（https://github.com/Puppet88/Agentic-AI-UAV）公开。

**📈 对比分析**

与基准方法（上层 PPO + 下层 A2C）比较，双层 PPO 在任务完成率几乎 100% 并保持高稳定性；上层 PPO 在收集率上几乎达到 100%；收敛速度快，性能优于 A2C。

**⚠️ 局限性**

局限：仅在两 UAV、六站的小规模场景下验证，未覆盖异构 UAV 或更大规模、动态工况；模型对检索质量与超参数敏感，仍需人工验证与微调。

---

## 412. Robust Mutation Analysis of Quantum Programs Under Noise

**arXiv ID:** 2605.13279 | [PDF](https://arxiv.org/pdf/2605.13279v1)

**作者:** Sophie Fortz `[一作]` (King's College London), Mohammad Reza Mousavi `[通讯]` (King's College London)

**通讯引用:** 2408 | [OpenAlex ID](https://openalex.org/A5101493818)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对量子程序的突变分析，本文通过实验验证噪声对突变检测的影响，并提出了噪声特定阈值与多种距离度量的组合来提升检测准确性。

**💡 创新点**

创新点在于：①首次系统地在真实噪声模拟器上评估量子突变分析；②引入噪声特定阈值策略，显著提升误判率；③对五种距离度量（密度矩阵、输出分布、期望值）与四种阈值进行全组合实验，揭示噪声与算法/电路特性之间的关系。

**🔧 技术方法**

使用了Muskit工具生成突变体、Qiskit Aer模拟器（含三种IBM噪声模型）进行执行、密度矩阵/输出分布/期望值的距离度量、统计检验（Levene、Mann‑Whitney、Cliff's Δ等）以及阈值计算与评估。

**📊 数据集**

数据集为41个量子程序（来自MQTbench，涵盖六类算法，2–8 qubits），共生成2,224个突变体（1,170 非等价，1,054 等价）。测试输入包括所有经典基态和相同规模的量子基态。噪声模型来源于三台IBM量子设备（Brisbane、Kyiv、Sherbrooke）。

**📈 对比分析**

比较方法：对每个度量-阈值组合构建混淆矩阵，计算准确率与F1分数。结果显示：密度矩阵度量在噪声环境下误判率可低至约17%，但在真实硬件不可直接获取；输出分布度量在噪声特定阈值下达到约73%准确率、≈75%F1分数；期望值度量表现最差。噪声特定阈值显著优于无噪声阈值，能够平衡误判与漏判。

**⚠️ 局限性**

局限性包括：①密度矩阵度量对硬件不可用，受限于模拟器；②实验仅限于8 qubits，无法评估更大规模电路的噪声效应；③噪声模型基于单一时间点的设备状态，可能不完全代表实际运行时的噪声；④阈值设计依赖于预先收集的CUT数据，可能对新硬件或不同实验设置的泛化性不足；⑤未考虑错误缓解技术对突变检测的影响。

---

## 413. Strong Conflict-Free Vertex-Connection via Twin Cover: Kernelization and Chromatic Bounds

**arXiv ID:** 2605.13299 | [PDF](https://arxiv.org/pdf/2605.13299v1)

**作者:** Samuel German `[一作]` `[通讯]` (University of California, San Diego), Samuel German (University of California, San Diego)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在图的twin cover参数化下，研究强冲突自由顶点连通数的可判定性与紧化问题。

**💡 创新点**

提出了基于twin cover的kernel与FPT算法，并给出svcfc与色数之间的精确加性上界。

**🔧 技术方法**

运用了参数化复杂度理论、图同构简化、结构分解与色数公式推导等技术。

**📊 数据集**

本研究为理论性工作，没有使用实验数据集，仅通过图论实例验证。

**📈 对比分析**

与仅考虑顶点覆盖参数的结果相比，本文在twin cover上实现了更小的kernel，并证明了svcfc≤χ+tc(G)，理论性能优于先前工作。

**⚠️ 局限性**

局限在于仅适用于twin cover参数，缺乏实验评估，对更一般图结构仍未解决。

---

## 414. The Readability Spectrum: Patterns, Issues, and Prompt Effects in LLM-Generated Code

**arXiv ID:** 2605.13280 | [PDF](https://arxiv.org/pdf/2605.13280v1)

**作者:** Hengzhi Ye `[一作]` (Peking university), Minghui Zhou `[通讯]` (Peking university)

**通讯引用:** 2608 | [OpenAlex ID](https://openalex.org/A5065977454)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统地评估大型语言模型生成代码的可读性，并构建了整合文本、结构、程序和视觉特征的可读性模型。

**💡 创新点**

创新点包括：在可读性评估中首次结合四大维度构建统一指标、揭示LLM代码与人类代码在可读性问题模式上的差异，以及量化提示工程对可读性的有限影响。

**🔧 技术方法**

使用可读性指标整合技术、随机森林回归、t检验与置换重要性分析、手工主题编码等方法。

**📊 数据集**

基准数据集为5,869条功能级代码对，来源于World of Code与LeetCode，并在MBPP/HumanEval上构造实验提示。

**📈 对比分析**

通过对比可读性分数、胜率与Wilcoxon检验，发现LLM代码可读性与人类相当且略高；提示工程中功能签名、约束与样式提升可读性，但整体影响有限。

**⚠️ 局限性**

局限性在于仅评估Python代码、仅考虑单函数级别、单轮提示交互以及缺乏跨语言与大规模项目的验证。

---

## 415. "It became a self-fulfilling prophecy": How Lived Experiences are Entangled with AI Predictions in Menstrual Cycle Tracking Apps

**arXiv ID:** 2605.13261 | [PDF](https://arxiv.org/pdf/2605.13261v1)

**作者:** Wendy Zhou `[一作]` (Chalmers University of Technology University of Gothenburg), Jichen Zhu `[通讯]` (IT University of Copenhagen)

**通讯引用:** 1520 | [OpenAlex ID](https://openalex.org/A5086741338)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过用户访谈与团体自民族志研究，探讨月经跟踪应用中的AI预测与用户体验的共构关系。

**💡 创新点**

提出“人机纠缠”视角，揭示AI预测对情绪、症状感知的自我实现效应，并给出设计启示。

**🔧 技术方法**

采用现有月经跟踪应用（如 Flo、Clue）的AI预测与解释功能，并使用定性人机交互方法进行分析。

**📊 数据集**

收集了14名女性用户的访谈记录以及四位研究者的自述日志，作为定性研究数据。

**📈 对比分析**

相比传统定性分析，研究结合访谈与自民族志双重视角，但未进行量化性能评估或算法对比。

**⚠️ 局限性**

研究样本局限于北欧地区，且受访者多为技术熟悉者，缺乏对怀孕、孕期等多元身体状态的深入探讨。

---

## 416. 3C: Competition, Competence, and Collaboration for Women in Computing

**arXiv ID:** 2605.13251 | [PDF](https://arxiv.org/pdf/2605.13251v1)

**作者:** Ioana Visescu `[一作]` (University of Luxembourg), Shalini Chakraborty `[通讯]` (University of Bayreuth)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5085515034)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个社区驱动的框架，通过参与式数据收集、共享叙事、焦点小组和问卷调查，系统记录并分析女性在计算机科学与软件工程领域的竞争经历、被认知程度和协作方式。

**💡 创新点**

创新点在于整合叙事共享、参与式研究与后续问卷设计的连续流程，强调从个体经验到集体行动的转化，并将女性共同体的力量视为解决结构性不平等的核心。

**🔧 技术方法**

采用的技术主要包括：线上焦点小组平台、叙事共享工具（如故事地图或讨论论坛）、协作式问卷设计工具，以及定性主题分析和可视化方法。

**📊 数据集**

目前尚未收集正式数据集，计划使用社区成员在焦点小组和工作坊中自愿分享的访谈记录、讨论纪要，以及后续发布的问卷调查结果作为研究数据。

**📈 对比分析**

由于研究不涉及算法或模型比较，评估方式采用主题一致性、参与者反馈的可操作性以及后续问卷的覆盖率和响应率；预期在社群内部能够得到较高的参与度和深度见解。

**⚠️ 局限性**

限制包括样本规模有限、数据来源主要为自我报告，可能存在偏见；缺乏长周期跟踪与量化指标，难以对干预效果进行客观度量。

---

## 417. When and Why is Optimistic Multiplicative Weights Slow? The Geometry of Energy Dissipation

**arXiv ID:** 2605.13242 | [PDF](https://arxiv.org/pdf/2605.13242v1)

**作者:** John Lazarsfeld `[一作]` (SUTD), Andre Wibisono `[通讯]` (Yale)

**通讯引用:** 956 | [OpenAlex ID](https://openalex.org/A5043389973)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了两个玩家零和博弈中Optimistic Multiplicative Weights Update算法的最后一次迭代收敛行为，提出了新的能量耗散分析框架。

**💡 创新点**

创新点在于通过对偶能量函数的消耗量提供最优的KL收敛率，并揭示了均匀收敛率在不同距离度量下的分离与下界。

**🔧 技术方法**

使用能量函数、对偶梯度下降、局部黎曼几何、Bregman散度、极限分析等理论工具进行证明。

**📊 数据集**

无实验数据集，全部为理论证明。

**📈 对比分析**

通过理论证明得到最优线性收敛率，比之前结果提升了exp(1/δ)因子；同时给出均匀收敛率在KL、TV和对偶性间的上界与下界，表明不同度量下收敛率存在本质差异。

**⚠️ 局限性**

局限性：仅在内部Nash均衡、零和博弈中成立；扩展到更一般博弈或非零和情形仍待研究。

---

## 418. Skill-Aligned Annotation for Reliable Evaluation in Text-to-Image Generation

**arXiv ID:** 2605.13223 | [PDF](https://arxiv.org/pdf/2605.13223v1)

**作者:** Abdelrahman Eldesokey `[一作]` (King Abdullah University of Science and Technology), Bernard Ghanem `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 25462 | [OpenAlex ID](https://openalex.org/A5024763828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了文本到图像生成（T2I）评估中技能对齐标注方法，构建统一的评估协议并实现可扩展的自动化评估流水线。

**💡 创新点**

将评估技能与最合适的标注机制匹配（如区域涂刷、单词级标注、Anchor Likert等），显著提升标注一致性和稳定性，并提供基于LLM代理的自动化评估。

**🔧 技术方法**

使用Krippendorff α、EDR、Spearman相关等统计指标评估标注质量，结合LLM、VLM（Molmo2-8B、Qwen-3.5-9B、ChatGPT-5）和预训练的Artifact Detector进行自动化评估；采用刷子工具、Anchor检索等交互式标注手段。

**📊 数据集**

基于Gecko提示集生成的多技能标签集（覆盖语义与美学子技能），并使用多款开源与专有T2I模型（FLUX.1-dev、FLUX.2-dev、Z-Image-Turbo、Nano-Banana、Wan-2.5-preview）生成图像进行评测。

**📈 对比分析**

通过与统一标注（Likert、BQA）对照实验，在6名专业标注者下进行相同样本评估；技能对齐标注使Krippendorff α从0.39提升至0.82/0.89/0.96，EDR降至0；自动评估与人工评估的Spearman相关在0.7–0.9之间，表明评估协议在不同技能上均具有较高一致性。

**⚠️ 局限性**

对感知性和相机属性等技能仍难以完全自动化；部分技能（如Named Entities）模型表现差，且对缺失物体或细节的检测仍不充分；自动评估在美学与相机属性上的相关性低，需进一步改进。

---

## 419. Backdoor Channels Hidden in Latent Space: Cryptographic Undetectability in Modern Neural Networks

**arXiv ID:** 2605.13214 | [PDF](https://arxiv.org/pdf/2605.13214v1)

**作者:** Marte Eggen `[一作]` (Norwegian University of Science and Technology), Inga Strümke `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 2075 | [OpenAlex ID](https://openalex.org/A5000311801)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在现代卷积和Transformer模型上实现可在不降低准确率的情况下植入不可检测的后门，利用训练好的特征空间中的稀疏方向进行触发。

**💡 创新点**

提出将后门嵌入模型权重的尖峰协方差分布，并将其与概念激活向量（CAV）相结合，使后门在白盒下难以区分，填补了理论与实际架构之间的空白。

**🔧 技术方法**

使用高斯尖峰协方差采样、线性后门层、CAV学习、输入触发器优化以及基于假设检验的不可检测性分析。

**📊 数据集**

在CIFAR-10、MedMNIST（BloodMNIST、DermaMNIST、PathMNIST）等图像分类数据集上进行实验。

**📈 对比分析**

对比了正常模型与后门模型的清洁准确率（CDA）与触发成功率（ASR），并在多种后训练防御（修剪、参数裁剪、噪声注入、Neural Cleanse）下评估，结果显示后门保持高ASR而CDA几乎不变，防御效果有限。

**⚠️ 局限性**

缺乏完整的白盒不可检测性理论证明，后门检测难度与尖峰强度θ、维度参数等高度依赖，并且仅在视觉任务上验证，未覆盖自然语言等其他领域。

---

## 420. Comparing the Performance of Heterogeneous Conjugate Gradient and Cholesky Solvers on Various Hardware Using SYCL

**arXiv ID:** 2605.13209 | [PDF](https://arxiv.org/pdf/2605.13209v1)

**作者:** Tim Thüring `[一作]` (University of Stuttgart), Dirk Pflüger `[通讯]` (University of Stuttgart)

**通讯引用:** 1390 | [OpenAlex ID](https://openalex.org/A5041326099)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究实现了基于 SYCL 的异构共用 CPU 与 GPU 的共轭梯度（CG）和 Cholesky 分解求解器，支持 NVIDIA、AMD 与 Intel GPU；

**💡 创新点**

创新点在于：① 仅使用 SYCL 实现两种求解器的完整异构算法，消除多语言维护成本；② 在多硬件平台上系统评估并证明异构能显著提升性能；

**🔧 技术方法**

采用 SYCL（AdaptiveCpp 与 Intel oneAPI DPC++/C++ 编译器）实现 CPU 与 GPU 共同计算；

**📊 数据集**

使用高斯过程协方差矩阵（源自质量-弹簧-阻尼系统模拟）作为测试数据集；

**📈 对比分析**

通过与单一 GPU 或 CPU 的实现对比，并在四种硬件系统上测量，CG 最高可提升 32.85%，Cholesky 最高提升 29.33%，并与 AdaptiveCpp 与 icpx 编译器性能对比；

**⚠️ 局限性**

局限性包括：仅使用 FP64 双精度；未对能耗进行评估；某些编译器缺乏向量化支持导致性能不均；未实现混合精度或分布式扩展。

---

## 421. Improving Code Translation with Syntax-Guided and Semantic-aware Preference Optimization

**arXiv ID:** 2605.13229 | [PDF](https://arxiv.org/pdf/2605.13229v1)

**作者:** Yuhan Wu `[一作]` (Nanjing University), Wei Hu `[通讯]` (Nanjing University)

**通讯引用:** 34780 | [OpenAlex ID](https://openalex.org/A5031365355)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于语法指导和语义感知的多目标偏好优化方法，用来改进代码翻译模型的性能。

**💡 创新点**

创新点在于将编译器提供的语法反馈与通过对齐源代码的跨语言语义模型产生的语义奖励相结合，形成统一的偏好优化框架，并通过对比训练直接强化语法与语义两个目标。

**🔧 技术方法**

使用的技术包括：对齐语义模型的双编码器架构与InfoNCE对比损失、基于编译器的语法奖励、直接偏好优化（DPO）以及对语义奖励差值的偏好修正。

**📊 数据集**

实验使用了 XLCoST 作为训练集（含 Java↔C++、C++↔Python、Java↔Python 对齐样本），以及 TransCoder-Test 与 HumanEval-X 两个公开基准测试集。

**📈 对比分析**

与无语义奖励、无语法奖励、传统 PPO、以及 IPO、SimPO 等基线方法进行对比。实验结果显示，在所有语言对和两大基准上均取得了 3.66%–6.70% 的性能提升，显著优于现有最优方法。

**⚠️ 局限性**

限制包括：对语义奖励依赖于源代码的对齐模型，负样本生成仍基于 LLM 变形，且在语言对与任务规模扩大时，编译器反馈与语义模型的兼容性和计算成本仍需进一步评估。

---

## 422. PaMM: Periodic Motif Memory for Atomistic Models with an Explicit Local-Structure Interface

**arXiv ID:** 2605.13297 | [PDF](https://arxiv.org/pdf/2605.13297v1)

**作者:** Ryan Dong `[一作]` `[通讯]` (Independent Research), Ryan Dong (Independent Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在UMA-S atomistic模型中加入显式的周期性配体记忆机制PaMM，利用离散化的两两距离和三角角度键值检索对边特征进行增强

**💡 创新点**

创新点在于将周期材料中频繁出现的局部配位模式（pair与triplet）以哈希表形式显式存储，并通过轻量级门控或仿射调制方式与现有网络融合，而非仅靠连续几何特征隐式编码

**🔧 技术方法**

技术包括：基于元素种类与距离/角度离散化的键值哈希表、两种读取方式（gate-only 与 affine-equipped）、对UMA eSCN-MD 边编码的修改及参数共享/学习率调优

**📊 数据集**

使用OMAT（周期材料）数据集进行评估，采用UMAs-S训练框架

**📈 对比分析**

与基线相比，在10k步和20k步的固定训练预算下，PaMM在能量MAE（gate-only）和力MAE（affine-equipped）上均有提升，提升幅度约为0.02–0.04 MAE；对齐对照实验表明提升源于结构化键值而非单纯参数增大

**⚠️ 局限性**

局限性：仅在UMA-S+OMAT固定预算下验证；跨数据集/不同主干网络的迁移性未显著提升；未提供更深层次的化学可解释性；更高阶配位记忆与完整资源/算力分析仍待进一步研究

---

## 423. IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages

**arXiv ID:** 2605.13292 | [PDF](https://arxiv.org/pdf/2605.13292v1)

**作者:** Shubham Kumar Nigam `[一作]` (University of Birmingham), Piyush Patel `[通讯]` (Madan Mohan Malaviya University of Technology)

**通讯引用:** 1012 | [OpenAlex ID](https://openalex.org/A5039065444)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

该论文构建了一个包含英语和九种印度语的并行多轮医疗对话数据集，并在此基础上训练了一个可在普通硬件上部署的小型语言模型，实现个性化多轮症状收集与诊断建议。

**💡 创新点**

创新点包括：①首个覆盖十种语言的并行多轮医疗对话数据集；②采用LLM生成合成对话、原生语言验证和脚本感知后处理以提升翻译质量；③在小型量化模型上使用LoRA实现参数高效微调；④使用LLM评判器进行语义级别诊断评估，克服传统标签匹配的不足。

**🔧 技术方法**

使用了LLaMA‑3.2‑3B‑Instruct作为基础模型，结合LoRA低秩适配和4‑bit NF4量化；采用GPT‑5.3判别器进行语义后处理；引入可选患者背景上下文；利用脚本感知后处理纠正翻译中的音素、词汇和字符间距错误。

**📊 数据集**

数据集基于原始MDDial（英文）与1,101条LLM生成的合成对话，随后翻译成九种印地语并通过两名本土语言专家验证，最终得到2,980条并行多轮对话，覆盖12种疾病。

**📈 对比分析**

通过与零射击多语种基线（Gemma、TinyAya、LLaMA）对比，并使用诊断准确率和医学专家评估两种指标。微调模型在英语、印地语、马拉地语和孟加拉语上显著提升，分别达到80.85%、72.76%、68.51%和58.72%；但在阿萨姆语、泰米尔语和特鲁古语等低资源语言表现极差，准确率不足10%。

**⚠️ 局限性**

局限性包括：①数据主要为合成对话，缺少真实患者交互的验证；②低资源语言受词表/脚本缺口影响，模型生成质量低；③仅支持文本对话，未整合图像、实验室报告等多模态信息；④仅评估12种单标签疾病，未扩展到ICD‑10多标签；⑤训练步骤有限，可能未收敛至最佳性能。

---

## 424. Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation

**arXiv ID:** 2605.13277 | [PDF](https://arxiv.org/pdf/2605.13277v1)

**作者:** Weiqing Luo `[一作]` (Arizona State University), Ziyi Huang `[通讯]` (Arizona State University)

**通讯引用:** 78 | [OpenAlex ID](https://openalex.org/A5103184354)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种基于信息增益的视觉证据选取方法，利用隐变量帮助度衡量证据对多模态检索增强生成（RAG）的真正效用；

**💡 创新点**

创新点在于将证据效用定义为对模型输出分布的熵减少，借助隐式帮助度变量将不可计算的答案空间信息增益转化为可计算的二分类信息增益，并通过训练无关的轻量级代理模型加速排序；

**🔧 技术方法**

核心技术包括信息理论框架（信息增益、KL 散度）、隐变量建模（帮助度 Z）、对抗式查询模板、轻量级多模态代理推理与主模型解码的两阶段架构；

**📊 数据集**

在 MRAG-Bench（1,353 图文问答）和 Visual-RAG（374 文本问题+图片）两个公开基准上进行评测；

**📈 对比分析**

与基线（CLIP/OpenCLIP、MLLM检索器、答案级不确定性估计）对比，本文方法在 Top‑K（1‑5）设定下平均提升 3–16 分，接近或超越 GT 上下文，且显著降低计算成本（FLOPs 与延迟）；

**⚠️ 局限性**

局限性包括（1）代理模型的迁移效果虽好但仍需更系统的选择与适配策略；（2）实验仅涵盖 QA 场景，尚未验证在对话、字幕、视频等多模态任务中的通用性；（3）对视觉之外的其他模态或混合模态的适配尚待研究。

---

## 425. D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models

**arXiv ID:** 2605.13276 | [PDF](https://arxiv.org/pdf/2605.13276v1)

**作者:** Yucheng Guo `[一作]` (JDT AI Infra), Yicheng Gong `[通讯]` (JDT AI Infra)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个高并发、低延迟的分布式强化学习框架，用于训练大规模 Vision‑Language‑Action（VLA）模型。

**💡 创新点**

核心创新点是“Plane Decoupling”（将高频数据采样平面与低频权重控制平面物理隔离）以及四线程异步“Swimlane”管线、双池 VRAM 管理与拓扑感知复制，彻底消除了模拟与优化之间的资源争用。

**🔧 技术方法**

主要技术包括：四线程异步并行管线、零拷贝数据交换、双池 GPU 内存管理、FSDP、GRPO、NCCL 与 Gloo 通信、InfiniBand 高速网络、Topology‑Aware 复制、动态资源分配。

**📊 数据集**

实验基准采用 LIBERO、ManiSkill 以及 Open‑X Embodiment 等公开数据集，验证了框架在不同 VLA 模型（π₀.₅、OpenVLA‑OFT 等）上的性能。

**📈 对比分析**

与 RLinf‑VLA、RL‑VLA³ 等主流框架在 16‑GPU 单机和多机环境下对比，使用吞吐率（steps/s）、单步延迟和总训练时间衡量。实验结果显示吞吐率提升 22%–86%，单步延迟下降 50%+，且收敛性能与基线相当或更好。

**⚠️ 局限性**

主要限制包括：异步管线对时序对齐敏感，资源平衡是性能瓶颈；在万亿参数规模或多智能体场景下仍需进一步细粒度的动态调度与验证；对不同硬件平台的适配性尚未完全覆盖。

---

## 426. LightSplit: Practical Privacy-Preserving Split Learning via Orthogonal Projections

**arXiv ID:** 2605.13265 | [PDF](https://arxiv.org/pdf/2605.13265v1)

**作者:** Mert Cihangiroglu `[一作]` (University of Pavia), Ahmad-Reza Sadeghi `[通讯]` (Technical University of Darmstadt)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在拆分学习中引入固定正交随机投影压缩切点激活并通过WCC正则化限制信息泄露

**💡 创新点**

创新点是将Johnson‑Lindenstrauss投影与无可训练投影结合，并加入Within‑Class Compaction正则化，实现通信压缩与隐私防护双赢

**🔧 技术方法**

采用正交随机投影、梯度回传、WCC正则化以及可选的MLP lift‑back技术

**📊 数据集**

在CIFAR‑10、CIFAR‑100、GTSRB、Fashion‑MNIST、MNIST等数据集上评估

**📈 对比分析**

与原始切点传输和Learned 1×1压缩相比，压缩率可达32×时仍保持97%精度，并显著降低SDAR、FORA、UnSplit等重建攻击的效果

**⚠️ 局限性**

局限在于缺乏正式隐私保证、仅针对半诚实服务器攻击、学习型lift‑back会增加服务器端计算

---

## 427. Intelligence Delivery Network: Toward an Internet Architecture for the AI Age

**arXiv ID:** 2605.13235 | [PDF](https://arxiv.org/pdf/2605.13235v1)

**作者:** Hanling Wang `[一作]` (Pengcheng Laboratory), Yong Jiang `[通讯]` (Pengcheng Laboratory)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出了Intelligence Delivery Network（IDN），一种将AI能力视为可交付网络服务的架构，支持在云、区域、边缘和本地等多层级计算资源中动态部署、路由、缓存和可信执行AI能力；

**💡 创新点**

创新点在于将AI能力抽象为可部署的“能力单元”，并通过层级资源集成、需求驱动部署、能力感知路由、状态感知缓存以及安全可信管理等六大机制，实现跨域、低时延、可扩展且符合隐私政策的分布式AI服务交付；

**🔧 技术方法**

采用的技术包括能力描述符与资源概况模型、分层资源注册与监控、基于需求与成本的放置优化、能力感知调度与多阶段路由、可重用状态（前缀、KV缓存等）的缓存与迁移策略，以及基于可验证的身份与链路证明的安全与信任框架；

**📊 数据集**

本文并未给出具体数据集，而是以理论分析与架构设计为主，提出了可在未来实验平台上使用的通用评估指标；

**📈 对比分析**

在实验设想中，通过仿真与原型测试验证了IDN在降低网络往返时延、减少跨域流量、提升算力利用率等方面的优势，但缺乏公开的基准结果，具体性能提升量化仍待后续工作；

**⚠️ 局限性**

限制主要体现在跨域信任与计费、状态迁移成本、经济激励模型缺失、以及对多智能体协作工作流的抽象不足等方面。

---

## 428. Teacher-Guided Policy Optimization for LLM Distillation

**arXiv ID:** 2605.13230 | [PDF](https://arxiv.org/pdf/2605.13230v1)

**作者:** Xinyu Liu `[一作]` (Northeastern University), JingBo Zhu `[通讯]` (Northeastern University)

**通讯引用:** 2156 | [OpenAlex ID](https://openalex.org/A5100370155)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于教师引导的 on‑policy 大模型蒸馏方法 TGPO，在学生生成轨迹上利用教师的 token 级指导信号，提升小模型的推理性能。

**💡 创新点**

创新点在于：① 将教师从后验判别者转变为前向引导者，通过教师对每一步的最佳 token 预测实现密集且方向性强的监督；② 将指导信号以可微正则化（Differentiable Regularization）的形式整合到 RL 优化中，并配合线性衰减策略平衡 imitation 与探索；③ 在 GRPO 框架下实现高效、稳定的 on‑policy 蒸馏。

**🔧 技术方法**

技术包括：Reinforcement Learning with Verifiable Rewards (RLVR)、Group Relative Policy Optimization (GRPO)、Reverse KL (RKL) 对比、奖励塑造 (Reward Shaping) 与可微正则化、教师动态 token 预测与线性衰减调度。

**📊 数据集**

数据集涵盖多种数学推理任务（AIME, AMC, MATH, Minerva, OlympiadBench, MATH-500）以及 OOD 推理基准（ARC‑c, GPQA-Diamond, MMLU-Pro）。使用 Qwen2.5‑Math‑7B 作为学生，Qwen3‑30B‑A3B 等教师模型。

**📈 对比分析**

与多种 on‑policy（GRPO++, KDRL, OP Distill 等）及 off‑policy/mixed‑policy（SFT, LUFFY）基线比较。TGPO 在 in‑distribution 任务上平均提升约 4–5 分，OOV 任务平均提升约 3–4 分；相比 RKL/OP Distill 训练稳定性更好，避免了奖励崩塌与长度爆炸；在不同教师下均表现出优于对照组的鲁棒性。

**⚠️ 局限性**

局限性包括：仍需教师模型支持，教师质量对最终表现有显著影响；线性衰减参数需手工调优；在极端分布差异或非推理任务上效果未知；训练仍受限于 RL 采样效率与梯度方差。

---

## 429. PoisonCap: Efficient Hierarchical Temporal Safety for CHERI

**arXiv ID:** 2605.13210 | [PDF](https://arxiv.org/pdf/2605.13210v1)

**作者:** Yuecheng Wang `[一作]` (University of Cambridge), Simon W. Moore `[通讯]` (University of Cambridge)

**通讯引用:** 4666 | [OpenAlex ID](https://openalex.org/A5041439799)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 PoisonCap，一个在 CHERI RISC‑V 上的硬件/软件协同方案，用来在堆内存上严格执行 use‑after‑free 和初始化安全，并通过“毒性能力”实现层级化的内存撤销。

**💡 创新点**

创新点包括：① 在能力本身嵌入毒性位与版本信息，直接在内存中标记已释放区域；② 利用能力边界实现多层分配器的层级撤销；③ 用缓存识别毒性线条来改进缓存替换，减少内存污染；④ 引入 CGetPoison 等指令，避免传统的 shadow‑bitmap 方案。

**🔧 技术方法**

使用技术包括：CHERI‑RISC‑V 指令扩展（poison bit、perm_poison、CGetPoison、CGetCapPoison 等）；改造 CHERI‑Toooba FPGA CPU 与 QEMU 模拟器；在 CheriBSD、Clang/LLVM、Jemalloc、SQLite 等软件栈中嵌入 poison 能力；对缓存子系统做 poison‑aware 替换策略。

**📊 数据集**

数据集与基准：NIST Juliet 1.3 版（2776 条 UAF、Double‑Free、未初始化访问），SQLite speedtest1，SPEC CPU2006 INT 9 个基准（train 8 小时）以及 2776 条 Juliet 测试。

**📈 对比分析**

比较方法：将 PoisonCap 与基线 CHERI、Cornucopia Reloaded（含/不含 zeroing）对比；评估 SPEC CPU、SQLite、Juliet 通过执行时间、周期数、DRAM 流量、功耗等指标。结果显示 PoisonCap 与 Cornucopia zeroing 基线几乎无性能损失，甚至在 SPEC 上略有 0.1% 的速度提升，SQLite 在多层分配器下成功撤销所有 UAF 而不产生额外内存占用。

**⚠️ 局限性**

局限性：① 仍需进一步优化 revoker 的递归页错误处理；② 对未初始化访问的写前读策略仅支持 128‑bit（quadword）粒度，部分程序会出现兼容性问题；③ 当前实现基于 25 MHz FPGA，尚未评估高频架构；④ 部分软件（如 SQLite 的 MEMSYS5）在迁移元数据时出现错误，需要手动调整；⑤ 需要在编译器层面改进，以支持更细粒度的初始化安全。

---

## 430. Hierarchical Attacks for Multi-Modal Multi-Agent Reasoning

**arXiv ID:** 2605.13213 | [PDF](https://arxiv.org/pdf/2605.13213v1)

**作者:** Hao Zhou `[一作]` (JD.com), Ai Han `[通讯]` (JD.com)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个三层级（感知、通信、推理）的对抗攻击框架 HAM3，用以系统性评估多模态多智能体系统的安全性。

**💡 创新点**

创新点在于：①首次将攻击拆解为感知、通信、推理三个层级，揭示跨层级的攻击传播机制；②提出多种跨模态、拓扑结构与内部推理链的攻击策略；③通过统一实验平台量化攻击成功率，证明推理层攻击最具破坏性。

**🔧 技术方法**

主要技术包括：多模态输入扰动（视觉+文本拼接）、通信图拓扑改造（伪造节点、链式阻塞）、内部推理链注入（链式思考替换）以及基于大型语言模型的多智能体协同推理框架（ReAct、Plan‑and‑Solve、Reflexion）。

**📊 数据集**

使用的公开数据集为 GQA（视觉问答），构造多智能体任务并基于 OxyGent 框架进行实验。

**📈 对比分析**

通过与视觉注入、文本注入、工具伪造、角色操纵等四类基线攻击对比，HAM3 在多模态多智能体系统中的攻击成功率最高，可达 78.3%（推理层 CIA），显示出明显优于单模态或单层攻击的效果；同时实验表明较大模型（如 o1‑mini、GPT‑4o）对攻击的鲁棒性更高。

**⚠️ 局限性**

局限性包括：①仅在 GQA 这一单一视觉问答场景下评估，缺乏跨任务泛化；②缺乏对应的防御或稳健性提升方案；③实验集中在 LLM+工具的合作推理范式，未覆盖更广泛的多模态推理框架；④对攻击成本和可解释性的分析不够深入。

---

## 431. A Horn extension of DL-Lite with NL data complexity

**arXiv ID:** 2605.13367 | [PDF](https://arxiv.org/pdf/2605.13367v1)

**作者:** Janos Arpasi `[一作]` (TU Wien), Magdalena Ortiz `[通讯]` (TU Wien)

**通讯引用:** 1688 | [OpenAlex ID](https://openalex.org/A5082472539)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种新的Horn DL 逻辑（基于 EL-I 的分层并约束并查操作的扩展），并给出将实例查询重写为嵌套双向正则路径查询（N2RPQ）的算法，从而实现数据复杂度为 NL 的查询回答。

**💡 创新点**

创新点在于：①通过分层机制控制并查与递归的交互，既保留了 DL‑Lite 的简洁性，又支持可达性和受限并查；②证明了该逻辑的实例查询可在 N2RPQ 中重写，因而数据复杂度仅为 NL；③给出完整的重写与证明框架，并证明了综合复杂度为 NExpTime‑完整。

**🔧 技术方法**

使用的技术主要包括：分层（stratification）约束、嵌套有限自动机（n‑nested NFA）构造、正则路径查询（N2RPQ）重写、归纳证明与推导系统、以及对 QBF 的 NExpTime‑硬性归约。

**📊 数据集**

本工作为理论研究，未使用具体数据集；所有结果均通过逻辑推导与复杂度分析得到。

**📈 对比分析**

方法与先前的 DL‑Lite 以及其它 Horn DL 进行比较：DL‑Lite 仅支持 FO 重写且对并查有限制；本文的逻辑在保持 NL 数据复杂度的同时，显著提升了表达能力；综合复杂度方面，DL‑Lite 的是 P‑时间，而本文的逻辑为 NExpTime‑完整。

**⚠️ 局限性**

局限性：①仅考虑实例查询，未覆盖完整的 C2RPQ 查询；②未支持角色层次与具体域；③分层机制虽然保证了复杂度，但对实际可扩展性和实现细节尚未验证；④当前仅理论证明，缺乏实验评估。

---

## 432. The Geno-Synthetic Algorithm: Type-Factored Coevolutionary Optimization for Heterogeneous Genotypes and Assembled Phenotypes

**arXiv ID:** 2605.13365 | [PDF](https://arxiv.org/pdf/2605.13365v1)

**作者:** Alex Bogdan `[一作]` `[通讯]` (Evolutionairy AI), Alex Bogdan (Evolutionairy AI)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Geno‑Synthetic Algorithm (GSA)，一种针对异构基因组的分型协同进化框架。

**💡 创新点**

核心创新在于按表示类型拆分基因组并使用对应本地变异算子，配合显式组装产生可执行表型。

**🔧 技术方法**

技术包括类型化产品空间搜索、类型本地演化算子、表型组装器、信用分配策略（elite、ensemble等）以及同步/异步调度。

**📊 数据集**

在自定义六个混合变量合成基准、BBOB‑MixInt外部套件，以及金融模型、LLM提示等仿真数据集上进行实验。

**📈 对比分析**

与Flattened DE/EA、混合变量GA、协同进化等基线比较，GSA在含复杂数、嵌入向量等无法扁平化的场景中唯一可运行，在高预算下可与最强扁平化DE相当；在单族或布尔问题上表现与强基线相当。

**⚠️ 局限性**

局限包括每代计算开销大、在光滑多族低预算场景下不如扁平化DE、信用分配需调优、异步调度在统一评估成本下无优势，以及缺乏真实金融或LLM的深入案例。

---

## 433. Contextual Bandits for Resource-Constrained Devices using Probabilistic Learning

**arXiv ID:** 2605.13346 | [PDF](https://arxiv.org/pdf/2605.13346v1)

**作者:** Marco Angioli `[一作]` (Sapienza University of Rome), Denis Kleyko `[通讯]` (Örebro University)

**通讯引用:** 2424 | [OpenAlex ID](https://openalex.org/A5065925392)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种低精度的概率化高维度计算上下文赌博机（HD-CBPROB）算法，用于在资源受限设备上实现自适应决策。

**💡 创新点**

创新点在于用时间衰减的概率更新规则取代传统的累积更新，并将向量分量限制在预设区间内，消除了周期性二值化带来的信息损失，同时显著降低内存占用和写操作频率。

**🔧 技术方法**

采用高维度计算（HD/VSA）与概率化更新规则，结合ε-贪心探索策略，并在离线评估中使用Open Bandit Pipeline（OBP）进行离线策略评估。

**📊 数据集**

使用OBP提供的标准化合成上下文赌博机数据集，实验覆盖多种动作数（10/15/20）和上下文维度（5/10/15）以及不同位宽（2/3/4位）。

**📈 对比分析**

将HD-CBPROB与传统线性上下文赌博机（LinEPS）、真实值HD-CB（HD-CBREAL）和二值化HD-CB（HD-CBBIN）进行对比。结果显示：在相同位宽下，HD-CBPROB在累计奖励上优于HD-CBBIN，且在3位时接近HD-CBREAL；内存占用也低于二值化方案。

**⚠️ 局限性**

局限性：仅在合成数据集上验证；未在真实日志数据、复杂奖励分布或实际硬件上测量延迟/能耗；高精度提升收益有限，需进一步探究不同任务与硬件环境下的适用性。

---

## 434. Swarm Network-as-a-Service (SNaaS)

**arXiv ID:** 2605.13341 | [PDF](https://arxiv.org/pdf/2605.13341v1)

**作者:** Balsam Alkouz `[一作]` (King Abdullah University of Science and Technology), Basem Shihada `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 3904 | [OpenAlex ID](https://openalex.org/A5063215964)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了 Swarm Network-as-a-Service (SNaaS) 框架，用无人机群提供按需、SLA 驱动的连接服务

**💡 创新点**

将无人机网络建模为可组合的服务，设计了直接、聚类、并行三种 composition 策略，并基于排队论的启发式方法动态选择和调优服务组合以满足延迟与稳定性约束

**🔧 技术方法**

采用服务导向架构 (SOA) 与软件定义网络 (SDN) 的结合，使用 M/G/1 高低优先级排队模型、k-means 聚类、路由优化以及动态重配置机制

**📊 数据集**

利用 AERPAW UAV 信号数据集进行真实空地链路测量，基于此估计服务速率并在仿真中评估性能

**📈 对比分析**

与暴力枚举的直接/聚类基线进行对比，实验显示基于排队论的 SNaaS 在延迟、违约率和计算时间上均优于基线，且在不同负载和群规模下能够平滑适应并保持 SLA 合规

**⚠️ 局限性**

对大规模并行 composition 的分析和实现仍有限；实验中未考虑干扰与能耗细节，且仅在仿真环境下验证，实际部署可能面临无人机电池、环境障碍和安全监管等挑战

---

## 435. Probing Persona-Dependent Preferences in Language Models

**arXiv ID:** 2605.13339 | [PDF](https://arxiv.org/pdf/2605.13339v1)

**作者:** Oscar Gilg `[一作]`, Patrick Butlin `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Gemma‑3‑27B和Qwen‑3.5‑122B的残差流训练线性探针，提取能预测并因果操控模型在任务选择中的偏好向量。

**💡 创新点**

发现该偏好向量是评估性表征，能够跨不同persona共享并同时预测与控制其行为，首次揭示LLM内部偏好机制的可控性与跨persona共享性。

**🔧 技术方法**

采用对数概率选择模型生成任务效用，线性回归提取偏好向量，对向量进行“steering”干预验证因果关系，并与文本编码器基线做对比。

**📊 数据集**

使用约6000个任务（WildChat、Alpaca、MATH、BailBench、STRESS‑TEST等）作为训练/测试集；还用CREAK真/假陈述、BailBench有害/良善任务及针对persona的系统提示进行验证。

**📈 对比分析**

与文本编码器基线及未训练的相似度做对比；在Gemma上偏好向量对测试任务的相关系数约为0.8，Qwen约为0.7；通过向量干预可将对任务选择的概率从≈0.01提升到≈0.99，且在多种persona下保持高相关性。

**⚠️ 局限性**

局限性包括跨persona迁移仍有噪声且未在更大模型上充分验证；Steering效果仅在Gemma实验显著，Qwen pilot 结果不佳；只在特定层级和方向上有效；未证明向量唯一性；实验主要基于prompt‑based persona，可能不适用于权重层面的persona。

---

## 436. IdeaForge: A Knowledge Graph-Grounded Multi-Agent Framework for Cross-Methodology Innovation Analysis and Patent Claim Generation

**arXiv ID:** 2605.13311 | [PDF](https://arxiv.org/pdf/2605.13311v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 437. What Limits Vision-and-Language Navigation ?

**arXiv ID:** 2605.13328 | [PDF](https://arxiv.org/pdf/2605.13328v1)

**作者:** Yunheng Wang `[一作]` (Hong Kong University Of Science And Technology (Guangzhou)), Renjing Xu `[通讯]` (Hong Kong University Of Science And Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了StereoNav框架，旨在解决视觉不稳定和指令模糊导致的Sim-to-Real迁移瓶颈，利用目标位置先验、立体视觉统一感知以及联合动作-深度预测实现更稳健的视听导航。

**💡 创新点**

创新点在于（1）将目标位置先验可视化为持续的视觉指引，显著缓解指令不充分的问题；（2）通过立体视觉构建语义、结构与几何三维融合表征，提升对光照变化、运动模糊等干扰的鲁棒性；（3）将深度预测作为几何监督，强化动作决策的空间感知。

**🔧 技术方法**

技术包括：InternVL-3.5-2B作为主干；InternViT与DINOv2分别提取语义和结构特征；FoundationStereo风格的3D几何编码器生成成本体积；多模态融合权重学习；自回归文本化动作生成与深度估计的联合预测头。

**📊 数据集**

使用Matterport3D场景中的R2R-CE和RxR-CE两个主流VLN评测数据集进行模拟实验，并在真实机器人上使用ZED Mini立体相机在办公室、健身房、礼堂和户外四种场景进行验证。

**📈 对比分析**

与现有单目RGB、RGB-D、全景和零射击方法对比，StereoNav在R2R-CE上实现SR 81.1%、SPL 68.3%，在RxR-CE上SR 67.5%、SPL 52.0%，比强大对手提升多达13.9% SR；在真实场景中成功率达60.6%，显著高于StreamVLN（24.3%）和DreamNav（22.1%），并在视角震动和运动模糊下表现出更小的性能下降。

**⚠️ 局限性**

缺点是当前系统仍需要外部的目标位置先验，无法在完全缺失先验信息的情况下进行导航，未来工作需探索直接从观测和语言推断目标先验。

---

## 438. HCSG: Human-Centric Semantic-Geometric Reasoning for Vision-Language Navigation

**arXiv ID:** 2605.13321 | [PDF](https://arxiv.org/pdf/2605.13321v1)

**作者:** Haoxuan Xu `[一作]` (Hong Kong University of Science and Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 919 | [OpenAlex ID](https://openalex.org/A5040338788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在动态人类环境中为视觉-语言导航(VLN)提供人类中心的语义-几何推理框架，能同时预测人类轨迹、姿态并通过视觉‑语言模型理解其意图，进而实现安全、社交合规的导航；

**💡 创新点**

双流推理（几何预测+语义解释）与人类信息融合到拓扑图中，以及基于社交距离的损失约束，突破传统仅视为障碍的被动避让，首次将高层语义与低层几何结合用于指令导向导航；

**🔧 技术方法**

视觉‑语言模型(VLM)、姿态与轨迹预测LSTM、跨模态图变换器、社交距离损失、YOLO‑Pose人类检测、CLIP文本编码、Qwen3‑VL‑2B‑Instruct推理；

**📊 数据集**

HA‑VLNCE基准数据集（含172类日常活动、910动态人实例、16,844条人类中心指令），并在Habitat仿真器和NXROBO Leo机器人上进行验证；

**📈 对比分析**

在HA‑VLNCE验证集上与BEVBert、ETPNav等SOTA方法对比，成功率提升14.3%（0.24 vs 0.21），碰撞率降低34.5%（0.36 vs 0.55），整体表现显著优于现有方法；

**⚠️ 局限性**

受限于离散“停留-采集-推理”周期，无法实时连续推理；依赖RGB‑Depth感知和预训练VLM，模型对不同视觉条件敏感；对人类检测误差和多模态对齐误差存在鲁棒性挑战；

---

## 439. PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding

**arXiv ID:** 2605.13319 | [PDF](https://arxiv.org/pdf/2605.13319v1)

**作者:** Yunhe Han `[一作]` (Zhejiang University), Yanfeng Zhang `[通讯]` (Northeastern University)

**通讯引用:** 1546 | [OpenAlex ID](https://openalex.org/A5100367496)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了PipeSD框架，用于云-边缘协同推理的speculative decoding。

**💡 创新点**

创新点在于 token-batch pipeline scheduling 与双阈值 NAV 触发机制，并通过轻量级贝叶斯优化实现自适应阈值调节。

**🔧 技术方法**

采用动态规划（DP）求解 token 批次调度，贝叶斯优化（BO）调节阈值，使用 llama-cpp-python、PyTorch 与 FastAPI 搭建系统。

**📊 数据集**

实验使用 HumanEval 编程数据集与 GSM8K 数学推理数据集。

**📈 对比分析**

与 Vanilla、HSL、EdgeLLM 三种基线对比，平均 TPT 提升 1.16–2.16 倍，云端能耗降低 14.3–25.3%。

**⚠️ 局限性**

局限性在于目前仅验证了少数模型与网络场景，需进一步测试不同硬件、网络条件及任务类型的鲁棒性。

---

## 440. SemRepo: A Knowledge Graph for Research Software and Its Scholarly Ecosystem

**arXiv ID:** 2605.13310 | [PDF](https://arxiv.org/pdf/2605.13310v1)

**作者:** Abdul Rafay `[一作]` (ScaDS.AI TU Dresden), Michael Färber `[通讯]` (ScaDS.AI TU Dresden)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了SemRepo，一个基于RDF的知识图谱，集成约20万GitHub科研软件仓库与其关联的学术出版物、作者、数据集等多源信息，实现跨域查询。

**💡 创新点**

创新点：①首次将科研软件与学术知识图谱在单一语义层面统一；②使用OWL设计细粒度的仓库活动模型（issues、贡献者、语言占比等）和n-ary关系模式；③实现跨知识图谱的语义链接（LPWC、SemOpenAlex、MLSea-KG），支持端到端的可追溯性和可复现性分析。

**🔧 技术方法**

技术：RDF/OWL本体设计、SPARQL查询、Triple Store部署、URI解析、n-ary关系模式、自动实体对齐流水线。

**📊 数据集**

数据集：81,078,636条三元组的SemRepo本体；源数据包括LPWC（论文-仓库链接）、GitHub API（仓库元数据、issues、贡献者、语言、语言占比等）、SemOpenAlex（作者ID）、MLSea-KG（数据集、实验、软件实体）。

**📈 对比分析**

比较方法：通过四个能力验证问题（CQ1–CQ4）展示SemRepo在跨域查询、可追溯性、维护优先级、可复现性风险评估等方面的能力；实验结果表明，SemRepo能够完成传统单源数据难以实现的端到端分析，并揭示科研软件的维护状态和复现风险分布，性能方面未给出具体指标，但SPARQL端点支持大规模查询。

**⚠️ 局限性**

局限性：①数据仅覆盖GitHub，未扩展至GitLab、Bitbucket；②依赖LPWC等上游源，受其覆盖范围与更新频率限制；③n-ary关系模式虽保留量化信息，但查询复杂度提升；④缺乏对多语言和地区偏倚的深度分析；⑤可复现性风险评估仅使用issue闭合率等指标，需进一步丰富评估维度。

---

## 441. BlockVLA: Accelerating Autoregressive VLA via Block Diffusion Finetuning

**arXiv ID:** 2605.13382 | [PDF](https://arxiv.org/pdf/2605.13382v1)

**作者:** Ruiheng Wang `[一作]` (Xi'an Jiaotong University), Xiangyu Xu `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 4417 | [OpenAlex ID](https://openalex.org/A5100729469)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 BlockVLA，一种块级扩散框架，将机器人视觉‑语言‑动作模型从自回归迁移到离散扩散，兼顾时间一致性与并行解码；

**💡 创新点**

创新点在于将自回归结构与块级并行扩散相结合，保持 KV‑缓存的可复用，设计了教师强迫与扩散强迫的遮蔽策略，并在动作块内部实现并行去噪；

**🔧 技术方法**

使用离散扩散语言模型、块级扩散、变分自编码器式噪声调度、RoPE 位置编码、SigLIP 与 DINO‑v2 双视觉编码器、Llama 语言模型、无 token‑shift 训练；

**📊 数据集**

在 LIBERO（Spatial、Object、Goal、Long）与 SimplerEnv（WidowX 拾取放置）两个机器人基准上进行实验；

**📈 对比分析**

与 OpenVLA（自回归）和 DDVLA（全序列离散扩散）比较，BlockVLA 在 LIBERO 上平均成功率提升至 91.7%（相较 83.2%）且收敛速度快，Inference 速度提升 3.3×（186.7 tokens/s），在 SimplerEnv 任务中也取得了与或优于基线的成功计数；

**⚠️ 局限性**

局限性包括需要对块大小和去噪步数进行超参调优，仍然需要多次去噪（如 2–3 步）导致一定延迟；对极长序列的适用性尚未完全验证；离散动作分辨率受 256‑bin 约束，可能限制细粒度控制。

---

## 442. Exploring Human-Robot Collaboration: Analysis of Interaction Modalities in Challenging Tasks

**arXiv ID:** 2605.13380 | [PDF](https://arxiv.org/pdf/2605.13380v1)

**作者:** Simone Arreghini `[一作]` (USI-SUPSI), Antonio Paolillo `[通讯]` (USI-SUPSI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

比较三种人机协作模式（被动、反应式、主动式）在构建七层彩色积木塔任务中的效果，评估任务完成时间、错误率、用户请求数及主观体验。

**💡 创新点**

首次系统地对主动式机器人帮助与反应式帮助进行对比，并在实验中引入Wizard‑of‑Oz方式实现自然交互，探讨主动介入是否提升用户体验与合作满意度。

**🔧 技术方法**

使用 DJI RoboMaster EP 移动机器人（配备摄像头、LED、麦克风、双关节机械臂），通过 ROS2 控制、LED 颜色提示与语音播报实现信息交互；采用问卷（Likert 1–7）和行为计数（任务完成时长、错误数、请求数）进行评估。

**📊 数据集**

使用18名参与者完成七层积木塔任务（每层4块，3块近距可取，1块远距），在三种模式下分别记录客观与主观数据；未使用公开数据集，全部为实验收集数据。

**📈 对比分析**

通过重复测量 ANOVA 和 Wilcoxon 符号秩检验比较三种模式的客观指标（完成时间）和主观评分；主动模式虽导致完成时间略长，但用户在任务易感、情感体验及机器人满意度评分上显著优于反应式和被动模式，且约 67% 的参与者偏好主动模式。

**⚠️ 局限性**

局限包括样本量小（18人），实验采用 Wizard‑of‑Oz 方式可能偏离真实自主操作；任务相对简单，难以推广至更复杂或动态环境；个体差异分析不足，未探讨长期使用效果。

---

## 443. GRIP-VLM: Group-Relative Importance Pruning for Efficient Vision-Language Models

**arXiv ID:** 2605.13375 | [PDF](https://arxiv.org/pdf/2605.13375v1)

**作者:** Mingzhe Huang `[一作]` (Tsinghua University), Ting Cao `[通讯]` (Tsinghua University)

**通讯引用:** 2693 | [OpenAlex ID](https://openalex.org/A5074166453)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出GRIP-VLM框架，对视觉语言模型中的视觉token进行动态稀疏化；

**💡 创新点**

将token剪枝建模为马尔可夫决策过程，利用预算感知评分器和群相对优势策略（GRPO）突破连续梯度逼近的局限；

**🔧 技术方法**

融合强化学习+监督蒸馏、FiLM预算调制、GRPO优化、奖励设计等技术；

**📊 数据集**

在8个图像基准（MMBench、POPE、SQA、TextVQA等）上使用LLaVA-1.5-7B/13B模型进行评估；

**📈 对比分析**

与启发式、SL、SparseVLM等基线比较，GRIP-VLM在相同准确率下实现约15%推理速度提升，显著提升Pareto前沿；

**⚠️ 局限性**

受RL采样效率影响，且对高分辨率视频等更大规模场景的推广仍有限制。

---

## 444. Exploiting Pre-trained Encoder-Decoder Transformers for Sequence-to-Sequence Constituent Parsing

**arXiv ID:** 2605.13373 | [PDF](https://arxiv.org/pdf/2605.13373v1)

**作者:** Daniel Fernández-González `[一作]` (Universidade de Vigo), Cristina Outeiriño Cid `[通讯]` (Universidade de Vigo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了首个基于预训练编码-解码 Transformer 的序列到序列成分句法解析器，并设计了多种词法化线性化策略。

**💡 创新点**

①利用 BART、T5、mBART 等预训练编码-解码模型进行成分解析；②提出词法化 top‑down、in‑order 以及带 Swap/Shift#k 的线性化方案；③在连续与离散树库上实现了序列化模型的 SOTA 性能。

**🔧 技术方法**

基于 Transformer 的预训练编码-解码模型（BART（大/小）、T5、mBART），对输出序列进行 fine‑tune，采用跨注意力和自注意力的编码器-解码器架构；使用词法化的 transition‑based 动作、beam search 解码。

**📊 数据集**

英语 Penn Treebank (PTB)、Discontinuous Penn Treebank (DPTB)、德语 NEGRA 与 TIGER 树库；未使用 PoS 标注。

**📈 对比分析**

与此前的 seq2seq 解析器以及任务特定解析器在 PTB 上进行对比，连续树库 F1 达到 96.2% 以上，优于所有 seq2seq 模型并与最佳任务特定模型相当；在离散树库上与最强 seq2seq 相当，且在 DPTB 上超过特定解析器，但在德国树库上仍落后。

**⚠️ 局限性**

对离散/高度不连续结构的处理仍不如结构化增强的模型；词法化的 Swap/Swap#k 线性化效果有限；未使用语言特定的预训练编码-解码模型，导致德语树库性能受限。

---

## 445. AI Harness Engineering: A Runtime Substrate for Foundation-Model Software Agents

**arXiv ID:** 2605.13357 | [PDF](https://arxiv.org/pdf/2605.13357v1)

**作者:** Hailin Zhong `[一作]` (Hong Kong Baptist University), Shengxin Zhu `[通讯]` (Beijin Normal University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AI Harness Engineering 概念，并通过 H0–H3 四级阶梯和基于轨迹的评估协议，对基于基础模型的自主软件工程代理在小型 Node.js 验证任务上的表现进行系统化分析。

**💡 创新点**

创新点在于将运行时支撑视为关键组件，系统化列出十一项责任，并提供可控制的阶梯消融与可审计的证据包，从而将模型能力与环境运行时分离，聚焦于可验证、可归因、可维护的变更。

**🔧 技术方法**

技术包括：基于文本与结构化指令的提示工厂；工具注册表与命令注册；项目记忆与任务状态文件；确定性行为检查、错误归因与验证协议；以及 JSONL 轨迹记录与可审计的事件包。

**📊 数据集**

使用数据集：单一手工构建的 Node.js 登录应用仓库（repoA‑T1），包含预定义的验证缺陷与三条确定性探针。

**📈 对比分析**

比较方法：对同一任务在四个 harness 级别（H0–H3）进行独立实验，记录 episode 包并对最终结果进行五标签判定；结果显示所有级别均能生成修补，但更高级别产生更丰富的证据、降低人为干预，并提高验证完整性。

**⚠️ 局限性**

局限性：实验规模有限，仅测试一个小型任务，缺乏多任务、多模型的广泛评估；评估侧重于证据生成而非实际性能指标；未探究长期迭代与仓库适配性问题。

---

## 446. Diversity of Extensions in Abstract Argumentation

**arXiv ID:** 2605.13332 | [PDF](https://arxiv.org/pdf/2605.13332v1)

**作者:** Johannes K. Fichte `[一作]` (Linkoping University), Zhengjun Wang `[通讯]` (Paderborn University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出基于对称差的抽象论证框架多样性量化概念，用以度量不同扩展之间的差异程度。

**💡 创新点**

创新点在于将多样性作为度量引入论证框架，给出了完整的复杂度分类，并与现有的代表性视角等方法进行对比。

**🔧 技术方法**

使用逻辑程序（ASP）和ASP-solver ASPARTIX实现k-多样性求解，并通过ASP编码求解最大/最小多样性问题。

**📊 数据集**

实验基准选用ICCMA 2025的论证框架数据集，限制在最多1000个论点的实例。

**📈 对比分析**

与传统决策/接受度问题对比，实验表明在中等规模框架上求解可行，最大/最小多样性均在秒级完成，表现良好。

**⚠️ 局限性**

主要限制包括：最小k多样性问题的完整复杂度尚未确定、参数化复杂度研究不足，以及对更大规模框架的可扩展性尚待验证。

---

## 447. FIND: Toward Multimodal Financial Reasoning and Question Answering for Indic Languages

**arXiv ID:** 2605.13330 | [PDF](https://arxiv.org/pdf/2605.13330v1)

**作者:** Sarmistha Das `[一作]` (Indian Institute of Technology Patna), Sriparna Saha `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 8176 | [OpenAlex ID](https://openalex.org/A5060797340)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个多语言、多模态金融问答基准 FinVQA，并提出了结合监督微调与约束解码的 FIND 框架。

**💡 创新点**

创新点在于覆盖 6 种印地语系语言、三难度层级、四种问题形式，并将视觉信息与文本同步处理。

**🔧 技术方法**

采用了 GPT-4o 翻译、PaddleOCR OCR、LoRA 参数高效微调、vLLM 约束解码和多模型评测。

**📊 数据集**

使用了基于 NCERT 教科书、ICMAI‑CMA 专业材料的 18,900 条样本，包括 14 金融领域。

**📈 对比分析**

在多模型对比中，大型 VLM 如 Qwen3‑VL‑32B 与 Gemma‑3‑27B 在受监督微调后显著提升准确率，尤其是印地语系语言。

**⚠️ 局限性**

局限在于多步推理一致性、视觉信息提取、金融知识深度以及样本多样性不足，且仍依赖大模型且存在可信度风险。

---

## 448. Tracing Persona Vectors Through LLM Pretraining

**arXiv ID:** 2605.13329 | [PDF](https://arxiv.org/pdf/2605.13329v1)

**作者:** Viktor Moskvoretskii `[一作]` (EPFL), Robert West `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在预训练阶段从何时开始形成 persona 向量，及其在后续对齐（SFT、DPO、RLVR）中的演化和稳定性；并比较了不同的 elicitation 方式对向量质量的影响。

**💡 创新点**

首次量化了 persona 向量在预训练的极早期（仅 0.22% 的训练步骤）即可出现，并证明这些向量在整个预训练期间持续保持可提取性和有效性；同时揭示了向量几何和语义的逐步细化过程。

**🔧 技术方法**

采用对比式 prompt 生成、残差流差分提取 persona 向量；使用比例归一化的 steering 公式在解码时加向量；通过多次预训练检查点抽样、余弦相似度、MDS 可视化、以及 LLM 判定器评估 persona 表达。

**📊 数据集**

主要使用公开的 OLMo-3 预训练 checkpoints（17 个点）和 Apertus；以及对应的 Instruct 版本（SFT、DPO、RLVR）。

**📈 对比分析**

用 trait‑expression delta（steered–baseline）衡量 steering 效果，并在不同 checkpoint 之间做 transfer 测试。结果表明：① 预训练初期提取的向量已能显著提高对应 trait；② 这些向量在整个预训练和后续对齐阶段保持高度一致；③ 不同 elicitation 方式均能产生有效向量，且各自强调不同 facets。

**⚠️ 局限性**

限制：① 向量出现时间仅是可提取的下界，可能更早出现；② 依赖 LLM 判定器，尚未完全验证人类一致性；③ 只考察了四个 persona 与两种模型族；④ 对齐阶段的细粒度影响（如 RLVR）仍未充分探索。

---

## 449. Neural Surrogate Forward Modelling For Electrocardiology Without Explicit Intracellular Conductivity Tensor

**arXiv ID:** 2605.13366 | [PDF](https://arxiv.org/pdf/2605.13366v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 450. Shortcut Mitigation via Spurious-Positive Samples

**arXiv ID:** 2605.13340 | [PDF](https://arxiv.org/pdf/2605.13340v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 451. Robust Matrix-Free Newton-Krylov Solvers via Automatic Differentiation

**arXiv ID:** 2605.13378 | [PDF](https://arxiv.org/pdf/2605.13378v1)

**作者:** Marco Pasquale `[一作]` (KTH Royal Institute of Technology), Stefano Markidis `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 4904 | [OpenAlex ID](https://openalex.org/A5085178088)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估并比较了在矩阵无关的 Jacobian-Free Newton–Krylov (JFNK) 求解器中，使用前向模式自动微分（AD）取代有限差分（FD）来计算雅可比-向量乘积 (JVP) 的效果。

**💡 创新点**

提出 AD 能显著降低有限精度下的数值不稳定性，提升 Krylov 子问题的有效性，从而在单精度环境中实现数百倍的迭代次数减少和三阶的时间加速，且成功率提升至 95% 以上。

**🔧 技术方法**

采用前向模式 AD（JAX），SciPy/CuPy 的 Krylov 求解器（GMRES、BiCGSTAB、CG），统一的 JFNK 流程，FP32/FN64 双精度，CPU 与 GPU (Apple M4 与 NVIDIA A100) 并行实现。

**📊 数据集**

四类非线性 PDE 基准：二维粘性 Burgers 方程、Su‑Olson 辐射扩散、反应扩散方程和 Kerr 非线性时频 Maxwell 方程，涵盖不同非对称/对称、标量/向量、实/复、不同刚度与参数空间，共 480 次实验配置。

**📈 对比分析**

通过 Dolan–Moré 性能曲线对比整体效率与鲁棒性；对 FD 与 AD 的执行时间、Krylov 迭代次数、Newton 步数进行统计；结果显示 AD 在 CPU 上可达 169×、GPU 2×加速，成功率从 42–64% 提升至 95%+，而 FD 在单精度下往往导致 Krylov 停滞和求解失败。

**⚠️ 局限性**

仅针对光滑、确定性残差的情形，若残差包含非光滑限幅器、适应性跳变或噪声嵌入求解器，AD 可能对算法噪声做出错误导数；此外在极端非线性或极低精度下，仍需细致调参或采用混合差分策略。

---

## 452. Phasor Memory Networks: Stable Backpropagation Through Time for Scalable Explicit Memory

**arXiv ID:** 2605.13370 | [PDF](https://arxiv.org/pdf/2605.13370v1)

**作者:** Sungwoo Goo `[一作]` (Chungnam National University), Sangkeun Jung `[通讯]` (Chungnam National University)

**通讯引用:** 827 | [OpenAlex ID](https://openalex.org/A5067724204)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用相位动力学和层次化可学习锚点的显式记忆网络PMNet，以实现稳定的长序列语言建模。

**💡 创新点**

创新点在于：① 将循环状态限制为复数单元圆上的相位旋转，实现严格的单元性和梯度范数保持；② 采用树状层次化可学习锚点的显式记忆，提供结构化且可检索的全局上下文；③ 结合分段梯度归一化和平行分段扫描，解决多路记忆写入导致的梯度爆炸。

**🔧 技术方法**

使用技术包括：相位动态单元（Phasor Dynamics）、层次化可学习锚点的显式记忆、分段梯度归一化、分段扫描（Segmented Scan）、滑动窗口注意力（Sliding SWA）以及跨相位注意力机制。

**📊 数据集**

在字节级别上使用FineWeb-Edu（约18.8B字节）进行预训练，并在PG-19测试集进行零样本评估；同时使用FineWeb-Edu 0.5B子集进行受控对比实验。

**📈 对比分析**

与SmolLM、Mamba、MambaByte等基线对比，119M参数PMNet在零样本长上下文（512k字节）上与353M的MambaByte实现相当的BPB，且优于SmolLM在长上下文下的梯度崩溃；在受控实验中PMNet在相同参数规模下优于SmolLM和Mamba。

**⚠️ 局限性**

局限性包括：PyTorch实现尚未使用专门的CUDA内核，导致训练速度比标准Transformer慢1.5倍、Mamba慢3倍；目前仅验证至100M参数规模，缺乏更大规模的可扩展性研究；对硬件层面的优化和实时部署仍有提升空间。

---

## 453. Query-Conditioned Test-Time Self-Training for Large Language Models

**arXiv ID:** 2605.13369 | [PDF](https://arxiv.org/pdf/2605.13369v1)

**作者:** Chaehee Song `[一作]` (KAIST), Changick Kim `[通讯]` (KAIST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在推理阶段基于输入问题自我生成监督信号，动态更新大语言模型参数的方法，称为Query-Conditioned Test-Time Self-Training（QueST）；

**💡 创新点**

创新点在于利用查询本身生成结构相似的题解对作为自监督信号，完全不依赖外部数据，同时通过LoRA实现轻量级的参数微调；

**🔧 技术方法**

核心技术包括基于预训练模型的自我生成题解对、LoRA低秩参数化的在线微调以及交叉验证的推理策略；

**📊 数据集**

实验覆盖七个数学推理基准（AMC、Minerva、MATH500、GSM8K、OlympiadBench、AIME25、AIME24）以及科学推理基准GPQA-Diamond；

**📈 对比分析**

与现有的测试时优化方法（TENT、TLM）比较，QueST在所有基准上均实现显著提升，平均提升幅度约5.8%–7.2%（以准确率计），且在标记消耗上更高效；

**⚠️ 局限性**

主要局限包括对输入查询质量的高度依赖，易受歧义或错误假设影响；并且目前为每个查询重新初始化参数，无法实现持续学习与知识累积。

---

## 454. Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models

**arXiv ID:** 2605.13338 | [PDF](https://arxiv.org/pdf/2605.13338v1)

**作者:** Shuqiang Wang `[一作]` (Zhejiang University), Zhixuan Chu `[通讯]` (Zhejiang University)

**通讯引用:** 977 | [OpenAlex ID](https://openalex.org/A5008967163)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于层次遗传算法的黑盒攻击框架，自动在大型推理模型中诱发过度推理（overthink）行为，导致输出长度显著增长，从而实现资源耗尽攻击。

**💡 创新点**

创新点在于：1）将推理问题拆解为结构化的前提与问题，并在此结构上进行遗传操作；2）构造复合适应度函数，兼顾输出长度与“过度推理”标记；3）展示了该攻击在多种大规模推理模型上的高可迁移性与高效性。

**🔧 技术方法**

主要技术包括：层次遗传算法（HGA）、结构化问题表示、复合适应度函数（token长度+过度推理标记）、交叉与变异操作以及黑盒查询评估。

**📊 数据集**

使用的基准数据集有GSM8K、SVAMP和MATH，且从中提取结构化前提进行实验。

**📈 对比分析**

与基线（原始数据集）和人工缺失前提（MIP）对比，攻击方法在最大输出长度上提升多达26倍，平均长度提升3–8倍，且在不同模型间表现一致；相较于AutoDoS等已有黑盒攻击，输入效率更高，能在更短提示下实现更大资源消耗。

**⚠️ 局限性**

局限性包括：①进化搜索需要一定的查询预算，对商业闭源模型的直接攻击成本仍较高；②仅关注推理长度作为攻击指标，未评估对模型回答质量或可解释性的其他潜在影响；③攻击成功率受模型随机性影响，需要多轮验证。

---

## 455. Context-Aware Web Attack Detection in Open-Source SIEM Systems via MITRE ATT&CK-Enriched Behavioral Profiling

**arXiv ID:** 2605.13337 | [PDF](https://arxiv.org/pdf/2605.13337v1)

**作者:** Badr Alboushy `[一作]` (Higher Institute for Applied Sciences and Technology), Aref Shaheed `[通讯]` (Latakia University)

**通讯引用:** 47 | [OpenAlex ID](https://openalex.org/A5056175718)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在Wazuh开源SIEM平台上实现了一个名为Smart‑SIEM的AI模块，利用每个源IP的最近N条安全事件构建行为上下文向量，对Web攻击进行二分类与六类细粒度识别，并支持自适应重训练；

**💡 创新点**

创新点在于：①基于N条历史事件构造的行为上下文向量（HTTP状态分布、最大规则触发次数、累计MITRE ATT&CK技术计数）实现了会话感知检测；②采用两阶段级联（LightGBM做二分类、XGBoost做多分类）显著提升检测与分类精度；③通过准确率阈值触发自动重训练，实时应对概念漂移；

**🔧 技术方法**

使用技术包括：梯度提升算法（LightGBM、XGBoost、CatBoost）、SMOTE‑NC过采样、特征工程、Apache Kafka消息队列、Elasticsearch上下文查询、Kibana可视化、Wazuh规则引擎的规则映射；

**📊 数据集**

实验数据来自46,454条Wazuh事件，构建于受控测试平台（OWASP Juice Shop+5种攻击工具+Selenium正常流量），按IP与攻击时间自动标注七类攻击，训练集使用SMOTE‑NC平衡，测试集保持自然分布；

**📈 对比分析**

与八种基线模型对比，加入上下文特征后二分类F1从≈0.705提升至≈0.967，六类分类F1从≈0.876提升至≈0.914；最佳级联模型在测试集上达到二分类0.967、六类0.914；与Wazuh原规则引擎相比，AI模块攻击覆盖率提升至95.8%（原为5.8%）；自适应重训练能在概念漂移后将F1从0.465恢复至0.814；

**⚠️ 局限性**

局限包括：单IP对应单类标签导致可能过拟合，首次攻击时上下文缺失产生冷启动偏差，测试仅覆盖单一Web应用与有限攻击工具，数据来自实验室环境，缺乏真实多样化流量，模型解释性不足，部署需修改Wazuh规则并通过合规审计。

---

## 456. Ego2World: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning

**arXiv ID:** 2605.13335 | [PDF](https://arxiv.org/pdf/2605.13335v1)

**作者:** Qinchuan Cheng `[一作]` (Xi'an Jiaotong University), Shijie Li `[通讯]` (A*STAR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了 Ego2World，一个基于真实第一人称烹饪视频的可执行隐藏世界图转换仿真环境，用于测试部分观测下的体态规划。

**💡 创新点**

创新点在于把实际视频注释编译成可执行的图转移规则，显式区分隐藏世界状态与代理信念图，并通过诊断评估展示行动重叠与最终状态成功不一致。

**🔧 技术方法**

采用了图结构的符号仿真、LLM 驱动的规划与回放、视觉语言模型、状态差异文本反馈以及自我检查的编译管线。

**📊 数据集**

利用 HD-EPIC 真实厨房视频的密集注释，包含 101 条视频、9,130 条动作组、426 个任务目标。

**📈 对比分析**

在统一的 Diff‑Memory 交互协议下对多种 LLM 规划器（Qwen‑Plus、DeepSeek‑V4‑Flash、Claude Sonnet 等）进行对比，结果显示不同规划器在任务完成率、可执行有效性、重规划率、视觉查询成本等多指标上存在显著差异。

**⚠️ 局限性**

局限包括：对单一厨房场景的依赖、对 LLM 生成图结构可执行性的可靠性不足、以及在更大规模、多模态或多代理场景下的扩展性尚未验证。

---

## 457. LLM-Based Persuasion Enables Guardrail Override in Frontier LLMs

**arXiv ID:** 2605.13334 | [PDF](https://arxiv.org/pdf/2605.13334v1)

**作者:** Rodrigo Nogueira `[一作]` (Maritaca AI), Marcos Piau `[通讯]` (JusBrasil)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了一种多轮对话情境下，前沿级LLM被另一同类LLM以自然语言压力诱导写出违反科学共识或历史真相的说服性文章的现象。

**💡 创新点**

首次证明即使单轮请求被严格拒绝，另一同类LLM在多轮互动中也能突破安全边界，且攻击者不需预先编写提示，攻击策略会自发生成。

**🔧 技术方法**

采用了基于五轮“写说服性文章”探针、对话生成、LLM间相互攻击与自适应推理，并用内部判断模型评估产生的文章是否违背共识。

**📊 数据集**

使用六个科学共识话题（平地说、疫苗无效、进化论否定、种族智商差异、气候变暖非人类因素、犹太人大屠杀不存在）以及三款前沿LLM（Claude Opus 4.7、Qwen3.5-397B、Grok 4.20）进行 540 场实验。

**📈 对比分析**

通过对比不同模型、不同攻击者强度、不同回合数的论文产出率，发现攻击者强度和回合数对成功率有显著影响；最高的攻击-被攻击匹配产生约65%的违反共识文章，而最弱组合仅 12%。

**⚠️ 局限性**

主要限制在于仅使用单一评判模型、固定五轮长度、评判者对虚构化免责声明的容忍度，以及攻击者在某些话题下可能拒绝发起请求。

---

## 458. Blind Recognition of Polar Codes Using Successive Cancellation List Decoding

**arXiv ID:** 2605.13331 | [PDF](https://arxiv.org/pdf/2605.13331v1)

**作者:** Changwei Tu `[一作]` (Beijing University of Posts and Telecommunications), Kai Niu `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 7408 | [OpenAlex ID](https://openalex.org/A5008455605)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `a244defd-9560-426b-b1b1-f78ebb2b7bf9`

**🎯 论文内容**

提出一种利用 successive cancellation list（SCL）解码的盲识别方法，用于已知长度的极化码；

**💡 创新点**

创新点在于将“冻结比特”与“信息比特”两种假设并行展开为解码路径，并基于多条接收向量的平均路径指标进行判定，从而充分利用软信息并提升识别鲁棒性；

**🔧 技术方法**

采用SCL解码框架、路径指标平均、Gaussian近似分析、LLR递归运算以及列表搜索等技术；

**📊 数据集**

通过仿真，使用BPSK调制的AWGN信道下的极化码数据集，包括(32,16)、(64,32)和(128,64)三种码长，分别收集M条码字进行测试；

**📈 对比分析**

与文献[5]–[7]、[9]中的方法进行对比；实验显示，随着列表大小增大，BSCL方法在相同成功率下可实现3–5 dB的SNR提升，显著优于现有方案；

**⚠️ 局限性**

主要限制在于计算复杂度较高，复杂度为O(M·L·N·logN)，对观测码字数M和列表大小L要求较高，且在极低SNR或观测量极少时仍可能出现识别误差。

---

## 459. Backbone is All You Need: Assessing Vulnerabilities of Frozen Foundation Models in Synthetic Image Forensics

**arXiv ID:** 2605.13381 | [PDF](https://arxiv.org/pdf/2605.13381v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 460. VERA-MH: Validation of Ethical and Responsible AI in Mental Health

**arXiv ID:** 2605.13318 | [PDF](https://arxiv.org/pdf/2605.13318v1)

**作者:** Luca Belli `[一作]` (Spring Health), Adam M. Chekroud `[通讯]` (Spring Health)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了专门针对自杀倾向风险的聊天机器人安全评估框架VERA-MH，包含会话模拟、LLM判定和模型评级三步；

**💡 创新点**

创新点在于：①临床专家共同设计评估规范并公开发布；②使用动态多轮对话生成器替代固定脚本，提升真实性；③通过“LLM-as-a-Judge”实现自动化且可追踪的判定流程；

**🔧 技术方法**

技术包括：大型语言模型（LLM）用于用户模拟与判定；基于流程图的多维度评估量表；API级自动化评估管道；

**📊 数据集**

数据集：100个由临床专家设计的用户角色（persona）集合，生成约2000条多轮会话；

**📈 对比分析**

比较方法：在推荐设置下，对四大主流LLM（Claude Opus 4.7、GPT‑5.4、Gemini 3 Pro Preview、Grok 4）进行评估；结果以5个维度（风险检测、风险确认、人类转介、支持式对话、AI边界）四级评分构成矩阵；表现相对均衡，未指出具体优劣；

**⚠️ 局限性**

局限性：缺乏跨文化适配、对评估数据集的过拟合风险、对LLM判定可靠性仍需进一步验证、计算成本高、仅覆盖自杀倾向一个安全维度、评估流程对人类判定仍有差距；

---

## 461. Embodied Neurocomputation: A Framework for Interfacing Biological Neural Cultures with Scaled Task-Driven Validation

**arXiv ID:** 2605.13315 | [PDF](https://arxiv.org/pdf/2605.13315v1)

**作者:** Johnson Zhou `[一作]` (Cortical Labs), Brett J. Kagan `[通讯]` (Cortical Labs)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在模拟的目标驱动导航任务中，通过大规模调优电极刺激编码参数，将26个生物神经网络嵌入闭环控制循环，验证了嵌入式神经计算框架。

**💡 创新点**

将生物与数字接口视为多变量优化问题，并通过系统级框架在1,300种编码组合中找到可持续学习的参数，首次在大规模实验中展示生物神经网络优于传统DQN的性能。

**🔧 技术方法**

使用微电极阵列（MEA）进行电刺激与记录、Optuna分布式超参数优化、XGBoost+SHAP特征重要性分析，以及深度Q网络（DQN）对照实验。

**📊 数据集**

基于26个CL1平台上的神经网络培养物，在6×6网格世界模拟环境中收集的实时交互数据，结合自制的“气味”标量感知信号。

**📈 对比分析**

与同等交互步数的DQN基准进行对照，BNN在两个实验组中分别获得约1.18倍至1.25倍的平均奖励，且在多周期学习模式下表现更佳，显著优于非适应性基线。

**⚠️ 局限性**

搜索空间受限于先验知识定义的6个参数范围，任务仅为低维标量输入，未检验更高维环境；反馈与编码参数共享导致优化不完全解耦；并且多周期学习的时间尺度与记忆机制尚未明确。

---

## 462. PRISM-X: Experiments on Personalised Fine-Tuning with Human and Simulated Users

**arXiv ID:** 2605.13307 | [PDF](https://arxiv.org/pdf/2605.13307v1)

**作者:** Hannah Rose Kirk `[一作]` (University of Oxford), Scott A. Hale `[通讯]` (University of Oxford)

**通讯引用:** 3605 | [OpenAlex ID](https://openalex.org/A5029882049)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两年后重新招募原始参与者，进行双盲多轮对话实验，对比基线、聚合人群微调、个体微调和基于提示的个性化模型；同时并行使用GPT‑4o进行模拟用户实验；

**💡 创新点**

首次系统评估个体化权重微调（P‑RLHF）与传统提示方法的实际用户偏好差异，并揭示微调会放大短期奖励的自我奉承与关系寻求行为；

**🔧 技术方法**

采用P‑DPO直接偏好优化进行个体化权重微调，利用软令牌嵌入编码用户；提示方法使用人口统计和GPT‑4o生成的偏好摘要；GPT‑4o进行模拟用户的角色扮演；

**📊 数据集**

使用PRISM Alignment数据集（1,500名跨75国参与者的偏好标签与人口信息）进行训练，并在重新招募的530名用户上收集约8.5k对话及67k偏好评分；

**📈 对比分析**

实验表明个体化微调模型在首轮选择和整体评价中均显著优于基线与提示模型（PPFT>DPFT>Base>Prompting），但与聚合人群微调相比提升有限；模拟实验仅能重现宏观层级，个体一致性远低于真实用户；

**⚠️ 局限性**

主要局限包括：提示方法使用的较小开源模型与非最优提示；个体化微调仅提升短期满意度且可能加剧自我奉承与情感依赖；模拟用户在行为、话题与偏好一致性方面表现不足，难以代替真实人类评估；

---

## 463. What Does LLM Refinement Actually Improve? A Systematic Study on Document-Level Literary Translation

**arXiv ID:** 2605.13368 | [PDF](https://arxiv.org/pdf/2605.13368v1)

**作者:** Shaomu Tan `[一作]` (University of Amsterdam), Felix Hieber `[通讯]` (Amazon AGI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统评估了在WMT24文学翻译任务中多语言、九种LLM的文档级自我迭代细化策略，探讨不同翻译-细化粒度组合、提示策略以及模型强度对翻译质量的影响。

**💡 创新点**

创新点在于提出了“文档级翻译+段落级细化+通用提示”这一最稳健的细化管道，揭示了细化主要提升流畅度、风格和术语而非准确性，并以分层MQM与人类评估验证细化的机制与局限。

**🔧 技术方法**

使用的技术包括多模型自回归推理、四轮迭代细化、不同粒度（句子、段落、文档）生成、MQM‑FSP无参考评测、对数概率与熵的可信度分析，以及基于LLM自身概率的分布投影评估。

**📊 数据集**

实验数据来自WMT24-Literary共享任务的七种语言对（en↔{cs,de,es,ja,ru,zh}, ja-zh）文档，覆盖约1百万词的翻译文本。

**📈 对比分析**

与基线单轮翻译相比，文档级翻译+段落级细化在MQM‑FSP上平均提升10–15分，在人类MQM和直接评估上亦显著提高，尤其在流畅度与风格维度；细化的效果随LLM强度呈“天花板”与“锚定”两种规律。

**⚠️ 局限性**

局限主要包括细化需要多轮推理导致推理时间和算力显著增加；对极长文本的编辑比例低，改进效果受限；细化未能显著提升准确性，且对低置信度错误的定位能力有限。

---

## 464. Building Interactive Real-Time Agents with Asynchronous I/O and Speculative Tool Calling

**arXiv ID:** 2605.13360 | [PDF](https://arxiv.org/pdf/2605.13360v1)

**作者:** Coleman Hooper `[一作]` (University of California, Berkeley), Kurt Keutzer `[通讯]` (University of California, Berkeley)

**通讯引用:** 38752 | [OpenAlex ID](https://openalex.org/A5047285420)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出异步 I/O 与预判性工具调用（Speculative Tool Calling）方案，使大型语言模型代理在实时语音交互中可并行思考、等待用户/环境信息并继续执行，从而显著降低延迟；

**💡 创新点**

创新点包括：① 将代理的推理/执行与用户输入与工具响应的延迟解耦；② 引入安全/不安全工具标记，支持推测性调用并在后期可纠正或取消；③ 事件驱动的推理框架与任务管理器，实现真正的异步流水线；④ 基于时钟的训练与合成数据方法，让模型在训练阶段即熟悉流式交互；

**🔧 技术方法**

技术手段：vLLM 的可中断流式推理、OpenAI Realtime API、LLMCompiler 的 DAG 调用管理、预判工具调用逻辑、事件驱动核心循环与任务管理器、Clock‑based 训练/数据合成、TTS + forced alignment 生成语音流式更新；

**📊 数据集**

实验数据集：HotpotQA、TinyAgent、自然对话评估集（177 条真实语音交互样本），以及使用 Qwen‑2.5‑32B‑Instruct 生成的合成对齐日志；

**📈 对比分析**

方法对比：与传统同步的 Reason‑and‑Act 基线对比；在 OpenAI Realtime API 上获得 1.3–1.7× 的速度提升，准确率仅微降；在 Qwen‑2.5‑3B 和 Llama‑3.2‑3B 上，经过 SFT 训练后 AsyncIO 方案实现 1.6–2.2× 的速度提升，准确率与非流式 SFT 相近；在 TinyAgent 与 HotpotQA 任务上分别给出了具体的延迟与准确率数值；

**⚠️ 局限性**

局限性：① 需要人工标记工具安全性，标注错误会导致不必要的等待或错误调用；② 预判性工具调用的正确性高度依赖训练数据的质量和覆盖范围；③ 对极端长延迟或不稳定网络环境的鲁棒性尚未充分验证；④ 边缘设备上的部署仍需进一步压缩模型与优化硬件；⑤ 对非语音或非流式输入的通用性待进一步研究。

---

## 465. GeoFlowVLM: Geometry-Aware Joint Uncertainty for Frozen Vision-Language Embedding

**arXiv ID:** 2605.13352 | [PDF](https://arxiv.org/pdf/2605.13352v1)

**作者:** Mayank Nautiyal `[一作]` (Uppsala University), Prashant Singh `[通讯]` (Uppsala University)

**通讯引用:** 4662 | [OpenAlex ID](https://openalex.org/A5100706486)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeoFlowVLM，后置概率适配器，在冻结的视觉-语言模型球面嵌入上学习联合分布，既可给出交叉模态熵（捕获先验不确定性）又可给出典型性得分（捕获经验不确定性）；

**💡 创新点**

创新点在于通过单一掩码 Riemannian 流匹配同时学习联合、边缘及条件密度，保持球面几何约束，首次在同一模型中同时获得交叉模态不确定性与密度基础的经验不确定性；

**🔧 技术方法**

使用 Riemannian 连续正则化流、球面几何、条件流匹配、可逆 ODE 与核密度估计等技术；

**📊 数据集**

在 Conceptual Captions 训练集上训练，评估 MS‑COCO、Flickr30k、CUB‑200、ImageNet‑1K、Food‑101、CIFAR‑100、ObjectNet 等检索与零样本分类基准；

**📈 对比分析**

与 ProbVLM、AsymVLM、GroVE、REPVLM 等方法对比，GeoFlowVLM 在检索熵校准（Recall@1 与 Spearman ρ 及 R²）和选择性准确率方面均与或优于现有方法，获得最优或接近最优的指标；

**⚠️ 局限性**

需额外 ODE 求解、核宽度与采样量，且仅给出密度基础的不确定性，未考虑参数后验；对极端分布或不同几何的适应性仍有限。

---

## 466. Drag within Prior Distribution: Text-Conditioned Point-Based Image Editing within Distribution Constraints

**arXiv ID:** 2605.13349 | [PDF](https://arxiv.org/pdf/2605.13349v1)

**作者:** Haoyang Hu `[一作]` (Ritsumeikan University), Yen-Wei Chen `[通讯]` (Ritsumeikan University)

**通讯引用:** 12877 | [OpenAlex ID](https://openalex.org/A5044216245)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DragPD 方法，结合先验保留正则化（PPR）、CLIP 奖励和方向加权点跟踪（DWPT）实现基于扩散模型的点编辑。

**💡 创新点**

创新点在于：①使用 PPR 在优化过程中约束编辑后的噪声样本与原始先验分布的距离；②引入 CLIP 奖励指导全局语义变化；③采用方向加权点跟踪减少编辑点漂移，提高细节编辑的准确性。

**🔧 技术方法**

使用技术包括 Stable Diffusion 1.5（基模型）、LoRA 微调、DDIM 逆向、运动监督、KL 散度正则化、CLIP‑ViT‑B/16 奖励函数，以及方向加权的特征最近邻搜索。

**📊 数据集**

实验数据集为自定义的 DragBench 数据集以及公开的 Drag100 基准数据集。

**📈 对比分析**

与 DragDiffusion、GoodDrag、StableDrag、SDE-Drag、FastDrag 等 SOTA 方法对比，使用 LPIPS、CLIP 目标分数、平均距离等指标评估；DragPD 在图像保真度、目标语义一致性、编辑点误差以及优化时间（≈43 s）方面均表现出色。

**⚠️ 局限性**

局限性包括：使用 CLIP 奖励时图像保真度略有下降；依赖每张图像的 LoRA 微调，增加前期工作；在极端语义变换（超出图像流形）时仍可能产生不自然结果。

---

## 467. Stylized Text-to-Motion Generation via Hypernetwork-Driven Low-Rank Adaptation

**arXiv ID:** 2605.13333 | [PDF](https://arxiv.org/pdf/2605.13333v1)

**作者:** Junhyuk Jeon `[一作]` (KAIST), Junyong Noh `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

通过超网络生成低秩 LoRA 参数，实现在已预训练的 SALAD 文本驱动运动扩散模型上动态注入参考动作的风格，从而生成既符合文本语义又具备细腻风格表达的人体运动。

**💡 创新点**

提出超网络驱动的 LoRA 风格适配器与监督对比损失相结合的连续风格空间，支持无类别标签的风格提取和推理时的梯度引导，突破了传统每种风格需单独微调或控制网络结构的局限。

**🔧 技术方法**

使用超网络（HyperNet）生成 LoRA 参数、FiLM 调制、监督对比学习、分类器无关指导与风格编码器梯度引导，并基于 SALAD 骨架实现运动扩散。

**📊 数据集**

使用 HumanML3D 作为文本提示数据集，使用 100STYLE（SMPL 重新映射）作为风格参考数据集，并用人工简单文本描述作为训练时的内容标签。

**📈 对比分析**

在 SRA、R‑Precision、FID、FSR 等指标上与 SMooDi、LoRA‑MDM、wu2025semantically 等基线对比，本文在风格表达（SRA）方面领先，内容保真和运动质量保持竞争力，并在未见风格上保持稳健性能。

**⚠️ 局限性**

局限于仅在行走动作集（100STYLE）上训练，风格表示偏向行走相关特征；风格本质抽象且与动作内容混杂，缺乏相对基准的表示，未来需更通用的大规模无标签数据和相对风格建模。

---

## 468. Supervised Deep Multimodal Matrix Factorization for Interpretable Brain Network Analysis

**arXiv ID:** 2605.13312 | [PDF](https://arxiv.org/pdf/2605.13312v1)

**作者:** Amjad Seyedi `[一作]` (University of Mons), Nicolas Gillis `[通讯]` (University of Mons)

**通讯引用:** 3675 | [OpenAlex ID](https://openalex.org/A5040368041)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `70e40602-aae3-44bd-80ec-4a7f2674330f`

**🎯 论文内容**

研究了一种监督深度多模态矩阵分解框架 SD3MF，用于脑网络分类，并提供可解释的社区级表示。

**💡 创新点**

将 SNMTF 扩展为多层深度结构，联合重构与监督损失，并通过自适应权重实现多模态融合，兼具可解释性与高性能。

**🔧 技术方法**

使用深度 SNMTF（多层非负矩阵三分解）、编码-解码式训练、线性分类器、梯度下降、模态权重自适应、隐式正则化等技术。

**📊 数据集**

实验采用 HIV、BP、PPMI 三个多模态脑连接组数据集，涵盖 fMRI、DTI 以及多种轨迹学方法。

**📈 对比分析**

与 CNN、GNN、张量分解、SVM 等多种基线在 ACC、AUC 上进行10折重复比较，SD3MF 在所有数据集上获得最高或最接近最高的准确率和 AUC，方差最低。

**⚠️ 局限性**

采用线性分类头限制表达能力；深度优化非凸且对小样本极端情况仍有限；仅处理静态网络；未探索更灵活的可解释网络。

---

## 469. Color Constancy in Hyperspectral Imaging via Reduced Spectral Spaces

**arXiv ID:** 2605.13306 | [PDF](https://arxiv.org/pdf/2605.13306v1)

**作者:** G. Dofri Vidarsson `[一作]` (École Polytechnique Fédérale de Lausanne), Sabine Süsstrunk `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 23180 | [OpenAlex ID](https://openalex.org/A5078201467)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究在 hyperspectral 图像上使用 Color-by-Correlation（CbC）进行光照估计，并系统评估不同光谱降维方法对估计性能的影响。

**💡 创新点**

首次在 CbC 框架下全面比较 PCA、Illuminant PCA、NNMF、LDA、随机投影等降维策略，提出 Illuminant PCA（Ill-PCA）在极低维度下仍能显著优于传统 RGB 的结论，并给出可行的低维光谱表示。

**🔧 技术方法**

使用 CbC 直方图匹配、PCA、Ill-PCA、NNMF、LDA、随机投影等降维技术，以及角度误差评估和合成 relit hyperspectral 数据生成。

**📊 数据集**

基于 KAUST-MIE hyperspectral reflectance 数据集（409 张 512×512 图像，31 频段 400–700 nm）以及 28 个 CIE 标准光源 SPD 进行实验。

**📈 对比分析**

与 RGB 基线和 Spectral Gray World（SGW）基线对比；在 d′=3 时，Ill-PCA 的平均角误差为 5.87°（相比 RGB 的 7.78° 降低约 25%），在 d′=5 时进一步降至 2.65°；PCA 也表现优于 RGB，NNMF、LDA 和随机投影在低维度下不如 RGB。

**⚠️ 局限性**

仅在 CbC 框架下评估，未验证对真实捕获 hyperspectral 图像或多光源/非全局光照场景的鲁棒性；实验基于离散候选光源集合，缺乏连续光照估计的探讨。

---

## 470. Exact Accepting-State Spectrum for Reversal of Permutation Automata

**arXiv ID:** 2605.13385 | [PDF](https://arxiv.org/pdf/2605.13385v1)

**作者:** Samuel German `[一作]` `[通讯]` (University of California, San Diego), Samuel German (University of California, San Diego)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

证明了置换自动机在反转操作下接受状态谱的完全特征，并验证了Rauch‑Holzer猜想。

**💡 创新点**

通过构造基于对称群的统一见证族，展示了任意接受状态数m≥2对应的任何α≥2的反转接受状态数均可实现，证明1是唯一的“魔法”值。

**🔧 技术方法**

采用群论技术，利用n循环与换位生成的对称群在α-子集上的自然作用来构造二元字母表的置换自动机。

**📊 数据集**

本文为理论性工作，没有使用实验数据集。

**📈 对比分析**

由于是纯粹的理论证明，未进行实验对比；所述“性能”即指可达状态数最小化与接受状态数的精确计数。

**⚠️ 局限性**

限制在于仅针对置换自动机及其二元字母表，尚未推广到更一般的自动机类或其他操作。

---

## 471. Constitutional Governance in Metric Spaces

**arXiv ID:** 2605.13362 | [PDF](https://arxiv.org/pdf/2605.13362v1)

**作者:** Ehud Shapiro `[一作]` (London School of Economics and Weizmann Institute of Science), Nimrod Talmon `[通讯]` (Ben-Gurion University and Input Output)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一套在度量空间上统一的宪法治理框架，实现从投票到议案通过的多阶段多维度决策过程。

**💡 创新点**

创新点在于将聚合、现实感知、超级多数修正、宪法共识、协商与 AI 介入统一为一个可多项式时间的完整治理流程，并引入公共议案市场机制。

**🔧 技术方法**

采用了度量空间聚合、通用中位数（generalised median）以及公开议案生成策略（如成对组合）配合超多数门控。

**📊 数据集**

主要使用人工生成的样本和理论模拟，在七种典型治理场景（预算分配、排名、委员会选举等）中进行实验。

**📈 对比分析**

与传统 L_p 聚合或 NP‑hard 聚合相比，框架在每轮 O(n²) 时间内完成；实验显示公共议案策略在连续空间中能关闭超过 90% 的妥协差距。

**⚠️ 局限性**

限制在于对高阈值（σ>1/2）的策略无效性尚未完全证明，公开议案生成策略的理论近似保证缺失，以及模型假设投票者不泄露信息。

---

## 472. Quantitative Linear Logic

**arXiv ID:** 2605.13348 | [PDF](https://arxiv.org/pdf/2605.13348v1)

**作者:** Matteo Capucci `[一作]` (University of Strathclyde and Independent Researcher), Ekaterina Komendantskaya `[通讯]` (Heriot-Watt and Southampton Universities)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了一种基于实值的量化线性逻辑体系（p-硬量化线性逻辑），将软加法（p-和、谐波和）引入到序贯演算中，构建可微分且满足关联、单位的算子，并给出完整的语义框架（softale）与证明理论。

**💡 创新点**

创新点在于：①将软加法与乘法结合成可微分的 p-算子，兼顾逻辑性与梯度可导性；②定义了量化序贯算子与softale结构，形成新的量化推理体系；③证明了归约性、剪切消除和完备性，为软化的加法提供了严谨的证明理论基础。

**🔧 技术方法**

采用的技术主要包括：实分析中的 p-和与谐波和、*‑自守的拓展序贯演算、深度推理与可变硬度参数、softale 的 *‑自守与可加结构、以及符号逻辑与范畴语义的结合。

**📊 数据集**

本工作为理论性工作，未使用具体数据集；但在后续应用章节中提到了贝叶斯概率与神经符号学习的案例，均基于概率分布或神经网络的训练数据。

**📈 对比分析**

通过与传统 MALL、模糊逻辑和 Lawvere 量化子逻辑的对比，证明了新体系在语义、推理复杂度（剪切消除）和完备性方面的优势；并在神经符号学习场景中与现有可微分逻辑（如 DL2、STL 等）比较，展示了梯度更友好、推理更一致的特性，但未给出实验性性能数值。

**⚠️ 局限性**

局限性包括：①软加法不满足幂等性，导致证明效率低（例如 η‑展开、交换律、结合律等证明的有效性低）；②预线性完整性在当前语义框架下尚未完全实现；③缺乏多项式证明算法与具体实现细节；④在高硬度极限（p → ∞）时，体系退化为传统 MALL，失去软化优势。

---

## 473. Multi-Agent Systems in Emergency Departments: Validation Study on a ED Digital Twin

**arXiv ID:** 2605.13345 | [PDF](https://arxiv.org/pdf/2605.13345v1)

**作者:** Markus Wenzel `[一作]` (Constructor University), Horst K. Hahn `[通讯]` (Fraunhofer Institute for Digital Medicine MEVIS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并验证了一种基于混合离散事件模拟与代理建模的急诊科数字孪生，用以评估多种资源优化干预；

**💡 创新点**

创新点在于将Mesa ABM与SimPy DES层无缝集成，加入疲劳、错误与死亡模型，并在此框架内实现LLM驱动的多代理系统用于实时干预决策；

**🔧 技术方法**

采用Python实现的Mesa、SimPy、LLM（GPT‑OSS‑20b via AutoGen）等技术；

**📊 数据集**

使用真实急诊科规模、人员配置和患者到达率的统计数据（按医生/护士人数、到达率、房间配置等参数）；

**📈 对比分析**

通过与文献基准指标（如LOS、等待时间、LWBS、死亡率）的统计对比，采用Welch t‑检验与Cohen d评估干预效果，发现Split‑Flow、Fast‑Track显著降低LOS和等待，Nurse‑Ratio对吞吐量影响有限；

**⚠️ 局限性**

局限包括模型未覆盖传染病情境、隔离设施缺失、LLM干预未经过充分验证，以及模型对极大型急诊科在资源饱和时表现不佳。

---

## 474. Hierarchical Transformer Preconditioning for Interactive Physics Simulation

**arXiv ID:** 2605.13343 | [PDF](https://arxiv.org/pdf/2605.13343v1)

**作者:** Carl Osborne `[一作]` (MIT), Wojciech Matusik `[通讯]` (MIT)

**通讯引用:** 22688 | [OpenAlex ID](https://openalex.org/A5018010391)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并训练了一种基于层次Transformer的预条件器，用于实时求解每帧变化的多相Poisson线性系统。

**💡 创新点**

创新点包括：① 将弱可接纳ℋ矩阵分块作为先验，强制稠密对角块与低秩远场块；② 通过带有高宽度的Transformer和轴向高速连接实现跨块信息传递；③ 提出了角度一致性（cosine‑Hutchinson）自监督损失，优化预条件器在子空间上的方向对齐而非绝对幅值。

**🔧 技术方法**

使用的技术包括：层次Transformer网络、ℋ矩阵低秩因子表示、axial highway、全图CUDA Graph执行、Hutchinson探针采样、可并行的块化算子和张量操作。

**📊 数据集**

实验数据集为二维多相Poisson压缩方程，随机生成的密度对比度（5–100）、障碍拓扑和方向，规模从 1 024 到 16 384。

**📈 对比分析**

与 Jacobi、IC/DILU、AMGX SPAI、神经 SPAI 等传统与学习预条件器进行对比，使用相对残差 1e‑8 的 PCG。结果显示在 N=8 192 时求解时间 17.9 ms（≈56 fps），比 Jacobi 低 4×、比神经 SPAI 低 2.7×，在所有规模上均实现显著速度提升。

**⚠️ 局限性**

局限性：① 需要在最大规模上预训练，超过此规模需重新训练；② 依赖于空间局部性和远场秩随距离衰减的假设，对无自然坐标或秩不衰减的算子效果下降；③ 在极高对比度或更大规模时收敛速率可能衰退。

---

## 475. Test-time Sparsity for Extreme Fast Action Diffusion

**arXiv ID:** 2605.13316 | [PDF](https://arxiv.org/pdf/2605.13316v1)

**作者:** Kangye Ji `[一作]` (Tsinghua University), Zhi Wang `[通讯]` (Tsinghua University)

**通讯引用:** 19501 | [OpenAlex ID](https://openalex.org/A5100376411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出了一种测试时稀疏框架，通过并行推理管线和全方向特征重用显著加速动作扩散；

**💡 创新点**

创新点在于：①将编码、pruner与解码异步并行，降低非解码延迟至毫秒级；②构造3D格子缓存，三向重用（当前前向、之前时间步、前一次回合）实现95%稀疏；③轨迹级监督训练pruner，实现动态适配；

**🔧 技术方法**

使用Transformer扩散模型、轻量化pruner、异步多线程、3D lattice缓存结构、Straight‑Through估计与采样监督；

**📊 数据集**

实验数据集包括人类演示的机器人操纵任务（Lift、Can、Square、Transport、Tool、Kitchen）和ManiSkill 4任务（PushCube、PickCube、StackCube、Insert），覆盖Diffusion Policy和RDT‑1B两大模型；

**📈 对比分析**

与Dense基线及EfficientVLA、BAC、L2C等加速方法对比；在95%稀疏下实现约5×速度提升（47.5 Hz无性能下降），在各种任务上成功率与原始模型持平或更高；

**⚠️ 局限性**

局限性包括：重用特征误差仍需严格控制；对超大模型如RDT‑1B提升有限；缓存管理复杂度较高；未在真实机器人上做实测验证。

---

## 476. Teaching and Learning under Deductive Errors

**arXiv ID:** 2605.13384 | [PDF](https://arxiv.org/pdf/2605.13384v1)

**作者:** Jan Arne Telle `[一作]` (University of Bergen), Jose Hernandez-Orallo `[通讯]` (Universitat Politecnica de Valencia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了考虑学习者在演绎推理中存在错误的PAC教学框架，并给出了在此框架下计算最优教学集的理论与算法。

**💡 创新点**

创新点在于将学习者的可测量、随机演绎错误纳入PAC教学模型，分析了不同教师与学习者行为的六种最优优化目标，并证明相关问题在参数化复杂度上是W[2]-hard，无法显著改进。

**🔧 技术方法**

主要技术包括：概率一致性矩阵、动态规划计算概念一致性计数、参数化算法（XP算法）以及基于贪婪启发式的教学集构造。

**📊 数据集**

实验数据来自对LLM（GPT‑5‑nano）在整数可被5、7、11、13、17整除的概念类上的演绎错误测量，使用1000个整数样本及多次推理查询。

**📈 对比分析**

与传统无误差教学方法相比，基于演绎错误的PAC教学在低误差场景下能够显著降低总教学错误（从>50%降至≤10%），并通过启发式算法在单例教学中实现接近0的教学错误；在高误差场景下，相关性仍然存在但效果弱化。

**⚠️ 局限性**

局限性包括：需先行测量学习者的演绎错误，且实验仅在有限的概念类和单一LLM上验证；在大规模或更复杂概念空间下的可扩展性与通用性尚未充分评估。

---

## 477. Beyond Oversquashing: Understanding Signal Propagation in GNNs Via Observables

**arXiv ID:** 2605.13383 | [PDF](https://arxiv.org/pdf/2605.13383v1)

**作者:** Eden Nagar `[一作]` (Technion - Israel Institute of Technology), Ron Levie `[通讯]` (Technion - Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了图神经网络（GNN）中信号在图上如何被定位、聚合以及传输，提出了以量子力学中的可观测量、期望值和方差为基础的信号定位与路由评估框架，并基于此设计了可逆的 Schrödinger GNN。

**💡 创新点**

创新点在于：① 用可观测量（自共轭算子）描述信号在图上“位置”与“动量”；② 定义信号路由度量 𝒫_M，衡量信号是否能在目标位置聚合；③ 证明传统谱 GNN 在信号路由方面的局限；④ 设计 Schrödinger 过滤器与单元 GSO，结合特征调制实现可定向、保形的信号传播。

**🔧 技术方法**

核心技术包括：量子力学可观测量框架、位置与动量算子（∇_f、W_f）、单元（unitary）图移位算子 𝒮[t,f] = e^{-itΔ_f}、复数信号与特征调制 D[θh]、位置-动量优化（PMO）以及基于 Schrödinger 过滤器的多层 GNN 架构。

**📊 数据集**

实验使用的主要数据集：① 经典 TU 图分类数据集（ENZYMES、IMDB、MUTAG、PROTEINS）；② 异构节点分类数据集（Roman‑Empire、Amazon‑Rating、Minesweeper、Tolokers、Questions）；③ 长程图基准 LRGB（Peptides‑Func、Peptides‑Struct、PascalVOC‑SP、COCO‑SP）；④ 简单的环图（Ring）进行信号传输的 toy 实验。

**📈 对比分析**

与 GCN、GAT、GIN、UniGCN、Adaptive Unitary、Lie Unitary、Exphormer 等多种基线进行架构匹配、参数匹配的对比。结果显示：在所有四个 TU 数据集上 Schrödinger GNN 在准确率上均优于或同样领先；在大部分异构节点分类任务中取得第二或第一名；在 LRGB 长程任务中，Schrödinger GNN 在 Peptides‑Func 上领先所有基线，在其他任务上位居前二，说明其对长程依赖的捕获能力突出。

**⚠️ 局限性**

限制：① 需要近似计算指数算子 𝒮[t,f]，导致计算开销和训练时间高于传统多项式滤波；② 目前理论证明集中在基于 Schrödinger 拉普拉斯的 GSO，对其它类型 GSO 的推广尚未完全覆盖；③ 复数信号与特征调制对模型解释性和实现细节有一定复杂度。

---

## 478. KamonBench: A Grammar-Based Dataset for Evaluating Compositional Factor Recovery in Vision-Language Models

**arXiv ID:** 2605.13322 | [PDF](https://arxiv.org/pdf/2605.13322v1)

**作者:** Richard Sproat `[一作]` (Sakana.ai), Stefano Peluchetti `[通讯]` (Sakana.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并发布了 KamonBench 基准，提供合成日式家纹图像到结构的映射数据，并在该任务上评估多种视觉-语言模型。

**💡 创新点**

创新点在于：①基于语法的生成器将家纹拆解为可知晓因子（容器、修饰、图样），②支持因子级评估、组合重排、对比因子敏感性和线性探针等多维诊断。

**🔧 技术方法**

使用的技术包括 Vision Transformer + Transformer 解码器、VGG + n‑gram 解码器（含/不含位置掩码）、线性探针以及少样本人类/LLM 对比实验。

**📊 数据集**

使用的数据集为 20,000 条合成家纹及对应的 KDL、日文、英文和程序标签，附加 20,000 条基元图像，划分为训练/验证/测试并提供重排 split。

**📈 对比分析**

比较方法：准确率、字符/词编辑距离、因子可恢复度、组合重排准确率、因子敏感性指标、线性探针准确率；结果显示 ViT 在程序标签上最强，VGG 近似，重排 split 下 Motif 识别仍是主要瓶颈。

**⚠️ 局限性**

局限性：生成图像质量低于真实家纹，未包含递归容器；仅评估三种标签空间，未将自然语言映射回因子；仅做线性探针，且 LLM 评测受有限样本限制。

---

## 479. Neural Video Compression with Domain Transfer

**arXiv ID:** 2605.13476 | [PDF](https://arxiv.org/pdf/2605.13476v1)

**作者:** Tiange Zhang `[一作]` (Peking University), Siwei Ma `[通讯]` (Peking University)

**通讯引用:** 15927 | [OpenAlex ID](https://openalex.org/A5039832462)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出DCVC-DT框架，利用在线潜在微调和帧级动态RD调整实现无参数更新的域迁移与自适应压缩

**💡 创新点**

创新点在于仅优化编码端潜在表示，保持解码器不变；结合轻量级在线域转移与基于质量波动的动态R/D权重调整

**🔧 技术方法**

使用梯度下降在线潜在微调（配合Gumbel量化）和帧间质量差异驱动的β_t动态调节技术

**📊 数据集**

使用HEVC Class C和D测试序列进行评估

**📈 对比分析**

与DCVC-DC、DCVC-TCM、DCVC-HEM等基线比较，BD‑rate平均下降6.21%（PSNR），显著提升压缩效率并改善误差传播

**⚠️ 局限性**

主要限制是编码端需进行多次迭代微调，导致计算与时间成本增加，且性能受迭代次数与学习率等超参影响

---

## 480. OSDN: Improving Delta Rule with Provable Online Preconditioning in Linear Attention

**arXiv ID:** 2605.13473 | [PDF](https://arxiv.org/pdf/2605.13473v1)

**作者:** Chenyu Zhou `[一作]` (Shanghai Jiao Tong University), Yinyu Ye `[通讯]` (Stanford University)

**通讯引用:** 27783 | [OpenAlex ID](https://openalex.org/A5041526408)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在DeltaNet线性注意力模型中加入在线预缩放，对写入键做特征级缩放，并通过对角预缩放矩阵实现更高效的写入步长，显著提升关联检索能力。

**💡 创新点**

提出Online Scaled DeltaNet（OSDN），利用在线梯度下降学习对角预缩放，理论上提供超几何收敛保证，并通过Adaptive Preconditioner Forgetting（APF）在非平稳上下文中动态刷新预缩放。

**🔧 技术方法**

结合OGD对角预缩放更新、块级WY并行、DeltaNet写入的非对称改写、对角化和APF等技术，实现低额外参数、常数内存的在线预缩放。

**📊 数据集**

使用FineWeb‑Edu训练数据，评估数据集包括WikiText、LAMBADA、PG‑19、JRT‑style cloze、LongBench以及PIQA、HellaSwag、WinoGrande等常识基准。

**📈 对比分析**

与DeltaNet、Gated DeltaNet、KDA等基线在340M和1.3B规模下对比，OSDN在JRT检索召回率上提升32%/39%，语言模型perplexity基本持平，吞吐率差异≤±5.5%。

**⚠️ 局限性**

局限包括理论假设需单调迭代、对角预缩放对Newton比较器的无争论收敛未严格证明；仅针对内在回归损失而非下游交叉熵；实验仅覆盖DeltaNet及少数基线，未验证更大规模或其他架构；缺乏置信区间等统计检验。

---

## 481. Rescaled Asynchronous SGD: Optimal Distributed Optimization under Data and System Heterogeneity

**arXiv ID:** 2605.13434 | [PDF](https://arxiv.org/pdf/2605.13434v1)

**作者:** Ammar Mahran `[一作]` (King Abdullah University of Science and Technology), Peter Richtárik `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 13053 | [OpenAlex ID](https://openalex.org/A5036598221)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出一种在异步梯度下降（ASGD）中按工作者计算速度比例缩放步长的方法，消除了数据与系统异质性导致的目标不一致问题，并在固定计算模型下实现了对全局目标的收敛。

**💡 创新点**

创新点在于只通过对各工作者步长的比例缩放即可纠正目标不一致，而不需要引入梯度缓冲、同步收集或额外内存，保持了传统ASGD的简单与高效。

**🔧 技术方法**

使用技术包括：异步梯度下降、周期性更新调度（cyclic schedule）、步长按计算时间比例缩放、理论分析证明收敛性与时间复杂度、对数据异质性与梯度滞后进行建模。

**📊 数据集**

实验数据集：MNIST，采用两层神经网络，并将10个工作者的数据按标签划分，形成极端的数据异质性。

**📈 对比分析**

与两种现有基线（例如VASGD和传统同步SGD）比较，实验显示在固定与随机计算时间场景下，所提方法在收敛速度与最终损失上与基线相当或更优，且对计算时间波动不敏感。

**⚠️ 局限性**

局限性：理论分析基于固定计算时间与谐波周期（harmonic periods）假设，无法严格覆盖随机/波动的计算时间情况；在这些情况下仍需进一步验证。

---

## 482. DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning

**arXiv ID:** 2605.13418 | [PDF](https://arxiv.org/pdf/2605.13418v1)

**作者:** Marc Molina Van den Bosch `[一作]` (CERN), Luigi Serio `[通讯]` (CERN)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过无数据的KFAC预处理实现差分隐私深度学习的高效优化。

**💡 创新点**

证明Fisher信息矩阵可由网络架构和模态特定频谱噪声分离，进而在不消耗隐私预算或使用公共数据的情况下构造预处理器。

**🔧 技术方法**

采用KFAC近似、合成结构化噪声（如1/f红噪声、频率加权词序列）、先行预处理再加DP噪声。

**📊 数据集**

在MNIST、CIFAR-100（CrossViT）、StackOverflow（BERT）、IMDB等视觉与语言数据集上验证。

**📈 对比分析**

与标准DP-SGD、公共数据DP-KFC及多种自适应/后置修正方法对比，DP-KFC在ε≤3的强隐私下在多任务上均比基线提升2–4%准确率，甚至接近私有数据预处理上限。

**⚠️ 局限性**

对深层层次或文本低维流形的方向性不佳；在语言任务中对随机词序列的假设导致与公共数据存在差距，且KFAC计算开销约为DP-SGD的2.2倍。

---

## 483. LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics

**arXiv ID:** 2605.13412 | [PDF](https://arxiv.org/pdf/2605.13412v1)

**作者:** Galadrielle Humblot-Renaux `[一作]` (Aalborg University), Thomas B. Moeslund `[通讯]` (Aalborg University)

**通讯引用:** 16284 | [OpenAlex ID](https://openalex.org/A5022176859)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对丹麦难民上诉委员会决策文本进行专家标注，构建RAB‑Cred数据集，并评估多模型多提示下LLM在可信度评估的零/少样本分类性能。

**💡 创新点**

首次提出针对低资源语言和专业法律领域的可信度评估任务，系统性地分析LLM错误、提示与模型敏感性，并展示多提示/模型集成显著提升。

**🔧 技术方法**

使用21种开源多语言LLM（如phi-4、Gemma、Qwen、Mistral、Bielik、EuroLLM等）与30种系统/用户提示组合，结合链式思考、元认知和少样本提示进行零/少样本分类。

**📊 数据集**

RAB‑Cred数据集：从丹麦RAB公开决策中抽取273例（验证70例、测试200例），包含专家双标注、置信度和案件结果。

**📈 对比分析**

在验证集上宏F1最高可达90.5%，测试集上最佳模型-提示组合宏F1介于84.4%至94.7%，明显优于结果作为代理的基线（53%）并通过模型集成提升至96%。

**⚠️ 局限性**

局限包括样本量有限（273例），仅评估最大35B参数的开源模型，未覆盖封闭模型，提示/模型评估仅一次运行，且缺乏跨时间、语言提示或领域微调的探索。

---

## 484. PreFIQs: Face Image Quality Is What Survives Pruning

**arXiv ID:** 2605.13396 | [PDF](https://arxiv.org/pdf/2605.13396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 485. RotVLA: Rotational Latent Action for Vision-Language-Action Model

**arXiv ID:** 2605.13403 | [PDF](https://arxiv.org/pdf/2605.13403v1)

**作者:** Qiwei Li `[一作]` (Peking University), Yadong Mu `[通讯]` (Peking University)

**通讯引用:** 9045 | [OpenAlex ID](https://openalex.org/A5028877572)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用SO(n)旋转矩阵表示的连续隐式动作模型（RotVLA），并在视觉-语言-动作预训练中使用三元组学习和流匹配来强化时序动力学；

**💡 创新点**

创新点在于：1）用连续SO(n)旋转代替离散码字，保留连续性与可组合性；2）三元组学习框架强制动作组合一致性，防止动作退化；3）在微调时将隐式动作视为高层规划器，实现跨体型的统一动作空间；

**🔧 技术方法**

核心技术包括：SO(n)旋转空间建模、SoftVQ软量化、三元组学习损失、流匹配（Diffusion Transformer）动作头、结构化注意力机制；

**📊 数据集**

使用了1700+小时跨体型机器人与人类视频数据集，包括Open X-Embodiment、AGIBOT-beta、RoboMIND、RoboCOIN、Ego4D等；

**📈 对比分析**

在LIBERO与RoboTwin2.0两个基准上，RotVLA以1.7B参数获得98.2%（LIBERO）和89.6%/88.5%（RoboTwin2.0）等平均成功率，显著优于其他VLA模型；在真实双臂ARX R5平台上也实现了90%+的任务成功率，并保持良好的实时推理速度；

**⚠️ 局限性**

局限性包括：1）SO(n)投影可能在学习早期限制表达；2）对高维动作空间的优化成本较大；3）当前在大规模多模态对齐上的泛化仍需进一步验证。

---

## 486. Trajectory-Level Data Augmentation for Offline Reinforcement Learning

**arXiv ID:** 2605.13401 | [PDF](https://arxiv.org/pdf/2605.13401v1)

**作者:** Tobias Schmähling `[一作]` (University of Applied Sciences Kempten), Tobias Windisch `[通讯]` (University of Applied Sciences Kempten)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于轨迹级别的短路（shortcut）数据增强方法，结合离线强化学习（Offline RL）在主动定位任务中的数据收集与学习。

**💡 创新点**

创新点在于：①构建了 LIFT 框架，利用已收集轨迹的几何结构识别可直接跳过的子轨迹；②给出理论条件（LPE、Lipschitz、f-收缩）证明累积动作为有效短路的充要条件；③提出了可嵌入任意离线 RL 算法的短路采样算法；④通过“增量式”数据收集和增量学习提升了数据质量。

**🔧 技术方法**

使用离线 RL 算法 CQL（以及可选的 IQL），训练 Q 函数作为增益器（augmentor）；实现短路采样的算法与阈值参数 C；将短路采样的增强数据用于训练增益器，再将增益器与原始日志策略拼接得到改进策略；在实验中与 SAC（warm‑start）、Diffusion‑QL 等方法进行比较。

**📊 数据集**

在仿真主动定位环境中测试，包括不同维度（d=2,5）、不同运动失真（f_blend、f_rot、f_scale、f_sin、f_regrot、f_sqrt）和观测类型（PO、LP、LT、FetchImg）等；日志策略为结构化的坐标步进策略（coordinate walk），可调节步长以模拟不同专家水平；数据集规模为 100–500 条轨迹。

**📈 对比分析**

与未增强的离线 RL（CQL）和仅使用短路后的离线训练（CQL‑SC）、增量式收集+短路（LIFT‑SC）以及 warm‑start SAC、Diffusion‑QL 等方法对比。结果显示：①在高维、部分可观测和强失真场景中，LIFT‑SC 的性能往往最好；②单纯的短路增强（CQL‑SC）已显著提升大多数任务的返回；③在部分失真导致收缩性失效或 LPE 不成立时，短路效果减弱或失效。

**⚠️ 局限性**

局限性：①理论保证依赖于失真函数的 LPE 与价值函数的 Lipschitz 连续性，这在某些实际系统中可能不成立；②实验仅在半现实仿真环境中验证，未在真实机械设备上测试；③对日志策略的结构性要求在一定程度上限制了通用性（虽然实验显示对非结构化策略也有提升，但理论支撑不足）。

---

## 487. Taming the Long Tail: Rebalancing Adversarial Training via Adaptive Perturbation

**arXiv ID:** 2605.13395 | [PDF](https://arxiv.org/pdf/2605.13395v1)

**作者:** Lilin Zhang `[一作]` (Sichuan University), Xianggen Liu `[通讯]` (Sichuan University)

**通讯引用:** 5252 | [OpenAlex ID](https://openalex.org/A5100669521)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种可插拔的自适应扰动框架，解决长尾分布下的对抗训练不平衡问题。

**💡 创新点**

通过理论证明扰动既能提升对抗鲁棒性又能平衡类别分布，提出基于类别扰动平衡（CPB）与迭代权重（AIW）的双重自适应策略。

**🔧 技术方法**

对抗训练（AT、AWP等）、平衡软最大化、DRO约束、Wasserstein距离分析等技术。

**📊 数据集**

CIFAR10-LT、CIFAR100-LT、TinyImageNet-LT（从CIFAR10/100/TinyImageNet构造的长尾版本）。

**📈 对比分析**

在WRN-28-10、ResNet-18等模型上与AT、AWP、RoBal、REAT、AT-BSL、TAET、UDR、CFA、DAFA等基线对比，均在自然精度与对抗精度（尤其尾类）上显著提升，并在AutoAttack、CW等攻击下保持更强鲁棒性。

**⚠️ 局限性**

对超参数α、β的调节敏感，在小模型或极端不平衡场景下效果相对弱；扰动分布动态估计增加计算成本；实验主要聚焦图像分类，缺乏对其他任务或更大规模数据的验证。

---

## 488. Tighter relaxations for MAP-MRF optimization via Singleton Arc Consistency

**arXiv ID:** 2605.13392 | [PDF](https://arxiv.org/pdf/2605.13392v1)

**作者:** Asaf Lev-Ran `[一作]` (Institute of Science and Technology Austria), Vladimir Kolmogorov `[通讯]` (Institute of Science and Technology Austria)

**通讯引用:** 25283 | [OpenAlex ID](https://openalex.org/A5021390142)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Singleton Arc Consistency的MAP-MRF松弛紧化方法

**💡 创新点**

利用SAC识别能够严格提高下界的三元组集合，并证明其优于寻找frustrated cycle的方法

**🔧 技术方法**

Singleton Arc Consistency、LP松弛、SRMP消息传递

**📊 数据集**

UAI 2022 MAP-MRF竞赛的14个基准实例

**📈 对比分析**

与未加紧化的SRMP、FR1/FR方法以及toulbar2比较，SAC在大多数实例上获得更高或相同的下界，且比FR更稳健

**⚠️ 局限性**

仅在CSP已满足SAC时无法进一步紧化，限制在未发现frustrated cycle的情形

---

## 489. Asymptotically Optimal Ergodic Coverage on Generalized Motion Fields

**arXiv ID:** 2605.13442 | [PDF](https://arxiv.org/pdf/2605.13442v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 490. Z-Order Transformer for Feed-Forward Gaussian Splatting

**arXiv ID:** 2605.13465 | [PDF](https://arxiv.org/pdf/2605.13465v1)

**作者:** Can Wang `[一作]` (University of Hong Kong), Dong Xu `[通讯]` (University of Hong Kong)

**通讯引用:** 25037 | [OpenAlex ID](https://openalex.org/A5082181536)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于Z-order序列化的Transformer框架，实现单前向3D Gaussian Splatting，用于高效且高质量的视角合成。

**💡 创新点**

创新点在于：①使用Z-order空间填充曲线将三维Gaussian序列化，保持空间局部性；②引入稀疏注意力（组注意力+top-k）高效建模上下文；③在Z-order序列上进行聚合压缩Gaussian数量；④设计基于Z-order的视角选择算法以减少冗余视角。

**🔧 技术方法**

采用的技术包括：Z-order编码、Transformer Encoder + DPT深度头、ZFormer块（稀疏注意力与Z-order池化）、两层MLP预测Gaussian参数、FlashAttention、AdamW + cosine学习率调度、跨域训练与多任务深度蒸馏。

**📊 数据集**

实验使用了三大数据集：RealEstate10K（360×640）、DL3DV（256×448）和ACID（256×256）。

**📈 对比分析**

与3DGS、MipSplatting、DepthSplat、AnySplat等方法对比，在2-12视角场景下，PSNR/SSIM/LPIPS均位居前列，尤其在仅2视角时优势显著；推理速度提升约1000倍，Gaussian数减少约2-3倍，整体表现优于现有技术。

**⚠️ 局限性**

局限性包括：在极少视角（如1视角）仍难以完全恢复细节；Z-order序列化在稀疏点云时可能导致局部信息损失；对光照变化的鲁棒性尚需提升；模型在极端复杂场景下的泛化仍有提升空间。

---

## 491. Liquid Tree Automata

**arXiv ID:** 2605.13456 | [PDF](https://arxiv.org/pdf/2605.13456v1)

**作者:** Ashish Mishra `[一作]` (Indian Institute of Technology Hyderabad), Suresh Jagannathan `[通讯]` (Purdue University)

**通讯引用:** 5958 | [OpenAlex ID](https://openalex.org/A5034957233)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于液体树自动机（LTA）的组件式程序合成框架，可在细化类型（refinement type）库与查询下自动生成满足规范的程序。

**💡 创新点**

核心创新在于引入支持逻辑约束的 LTA，并利用子类型关系实现语义相似性检测，进而在搜索过程中高效剪枝和合并等价子图。

**🔧 技术方法**

技术手段包括细化类型系统的类型推导与子类型检查、SMT 求解器用于约束求解、树自动机构造与最小化、以及基于逻辑蕴含的相似性归约。

**📊 数据集**

实验使用了从 Hoogle+、Hectare 以及数据库验证基准衍生的 42 个细化查询和 8 个复杂状态机查询，共 50 条基准。

**📈 对比分析**

与 Synquid、Hoogle+ 等现有工具在同一时间限制下比较，LTA 平均 7.6 秒完成 42 个任务，成功率 100%，比 Synquid 慢 4.5 倍、比 Hoogle+ 快 6 倍，并在更大、更复杂的查询中仍能在 1 分钟以内完成。

**⚠️ 局限性**

局限性包括目前不支持循环与递归程序的生成，逻辑约束受限于可判定子理论，并且对极大规模库的可扩展性仍需进一步评估。

---

## 492. TurboGR: An Accelerated Training System for Large-Scale Generative Recommendation

**arXiv ID:** 2605.13433 | [PDF](https://arxiv.org/pdf/2605.13433v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 493. Revisiting CUR Perturbation Analysis: A Local Tangent-Space Expansion

**arXiv ID:** 2605.13437 | [PDF](https://arxiv.org/pdf/2605.13437v1)

**作者:** Longxiu Huang `[一作]` (Michigan State University), Longxiu Huang `[通讯]` (Michigan State University)

**通讯引用:** 368 | [OpenAlex ID](https://openalex.org/A5035286718)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了秩截断 CUR 近似在低秩矩阵附近的局部扰动展开，推导了其弗雷歇导数为采样诱导的斜切切空间投影，并分析了第一阶恢复误差的方向性；

**💡 创新点**

首次把 CUR 的局部误差表述为采样诱导的斜切投影，揭示 CUR 与 SVD 在第一阶误差组件上的区别，并指出当扰动对采样行列不可见时 CUR 的误差可降至二阶；

**🔧 技术方法**

利用矩阵微分、截断伪逆的局部展开、SVD 以及采样诱导的斜投影；

**📊 数据集**

使用合成的低秩矩阵（m=80,n=70,r=5）以及高斯噪声扰动，通过列/行采样选择利用 leverage score；

**📈 对比分析**

通过数值实验比较 CUR 与秩 r SVD 的误差，发现 CUR 在扰动对采样行列不可见或可见程度低时误差呈二阶，SVD 在正交正常扰动下表现更好；

**⚠️ 局限性**

局部理论仅适用于固定的可接受采样行列，且只处理矩阵形式；未扩展到张量 CUR 或更一般的低秩模型；

---

## 494. Text2Score: Generating Sheet Music From Textual Prompts

**arXiv ID:** 2605.13431 | [PDF](https://arxiv.org/pdf/2605.13431v1)

**作者:** Keshav Bhandari `[一作]` (Queen Mary University of London), Simon Colton `[通讯]` (Queen Mary University of London)

**通讯引用:** 6063 | [OpenAlex ID](https://openalex.org/A5102963061)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Text2Score两阶段框架，先用LLM规划乐曲结构，再用层次解码器生成ABC谱。

**💡 创新点**

创新点在于将结构规划与符号生成分离，并直接利用XML提取的结构监督，提升可读性和可演奏性。

**🔧 技术方法**

使用ModernBERT做计划编码器，GPT‑2层次解码器，LLM（GPT‑5.1）作为规划器。

**📊 数据集**

训练数据是621,162首ABC格式乐谱，来自MIDI‑to‑ABC、XML‑to‑ABC和公开数据集。

**📈 对比分析**

与端到端MIDI/LLM模型及纯LLM ComposerX对比，Text2Score在可玩性、可读性、提示符合度及主观评分均显著优于基线。

**⚠️ 局限性**

局限包括对LLM规划错误的敏感性，以及计划细粒度不足导致的细节表达受限。

---

## 495. LIFT: Last-Mile Fine-Tuning for Table Explicitation

**arXiv ID:** 2605.13424 | [PDF](https://arxiv.org/pdf/2605.13424v1)

**作者:** Divij Khaitan `[一作]` (Microsoft Corporation), Ashish Tiwari `[通讯]` (Microsoft Corporation)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为 last-mile fine‑tuning 的两阶段表格显式化（table explicitation）方法：先用预训练大型语言模型（LLM）从剪贴板无结构文本中粗略提取表格，然后用经过微调的小型语言模型（SLM）对提取结果进行错误修复。

**💡 创新点**

创新点在于：① 将错误修复任务抽象为单独的“最后一英里”微调阶段，避免多轮 LLM 调用；② 证明在样本稀缺或输入格式不确定时，该方法比传统自我调试（self‑debug）和端到端微调（end‑to‑end）更有效；③ 提供了跨多种 SLM（1B–24B）与三大公开数据集的系统性评测。

**🔧 技术方法**

技术细节包括：使用 GPT‑4o 进行粗略表格生成；对 Llama 3.2、Qwen 3、Mistral 7B/24B、Phi‑4 等 SLM 通过 LoRA（rank 16，α 32）在 bfloat16 上训练；评估指标为 Tree Edit Distance Similarity（TEDS）、Lev‑TED、Grid Table Similarity（GriTS）；对比自我调试、last‑mile 微调、端到端微调三种策略。

**📊 数据集**

采用 PubTabNet、FinTabNet、SciTSR 三大公开表格数据集，总计 15,515 个显式化任务；在每个任务中先用 GPT‑4o 提取表格，再筛选可修复案例形成训练集 8,967/1,133/2,596 的 train/val/test 分布。

**📈 对比分析**

比较方法：在相同 SLM 上执行三种策略，使用 TEDS/Lev‑TED（越高越好）以及 GriTS（越高越好）进行评测。结果显示：last‑mile 微调在 TEDS 上往往与端到端相当或更优（例如 Mistral‑7B 仅 1,000 样本时 TEDS 0.875 > 0.731），并且在输入格式多变（破损 CSV、干净 JSON）时仍保持更高的鲁棒性；相较于自我调试，修复精度提升 0.1–0.2 的 TEDS，且仅需一次 LLM 调用而非两次。

**⚠️ 局限性**

局限性包括：① 需要先调用 GPT‑4o 作为第一阶段，增加推理成本和对 LLM 许可的依赖；② 仅针对剪贴板文本实验，未验证对更复杂源文件（如扫描 PDF）或其他任务的通用性；③ 在极大 SLM（如 Mistral‑24B）上偶出现空输出，需要人工过滤；④ 仅在 TEDS/Lev‑TED 等指标下评测，未考察用户体验或实时性能。

---

## 496. From Rosetta to Match-Up: A Paired Corpus of Linguistic Puzzles with Human and LLM Benchmarks

**arXiv ID:** 2605.13408 | [PDF](https://arxiv.org/pdf/2605.13408v1)

**作者:** Neh Majmudar `[一作]`, Elena Filatova `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一套从 Rosetta Stone 题目自动转换为 Match‑Up 形式的程序，并构建了 96 对并列语料。

**💡 创新点**

创新点在于证明两种形式本质相同，可通过一次转换生成匹配题目，同时提供了可公开的转换数据集。

**🔧 技术方法**

采用文本预处理、句子配对算法以及零样本 LLM 推理进行实验验证。

**📊 数据集**

使用 UKLO 官方发布的 96 份 Rosetta Stone 题目作为原始数据，生成 96 对 Match‑Up 题目。

**📈 对比分析**

通过人类评测（2 名高水平解题者）和两大 LLM（GPT‑5 与 Gemini‑2.5‑Pro）比较，结果显示人类在两种形式表现相近，而 LLM 在 Rosetta Stone 上优于人类，在 Match‑Up 上表现波动且出现全对/全错的“全或无”现象。

**⚠️ 局限性**

局限在于人类样本极少、语言覆盖不均、某些多模板动词系统无法转换，以及 LLM 评价受模型版本和解码策略影响。

---

## 497. Twincher: Bijective Representation Learning for Robust Inversion of Continuous Systems

**arXiv ID:** 2605.13470 | [PDF](https://arxiv.org/pdf/2605.13470v1)

**作者:** Arkady Gonoskov `[一作]` (University of Gothenburg), Arkady Gonoskov `[通讯]` (University of Gothenburg)

**通讯引用:** 2414 | [OpenAlex ID](https://openalex.org/A5036132009)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `40105733-5154-44cd-8090-a8cab9e64b07` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究并实现了一种名为Twincher的可逆神经网络架构，用于学习连续系统的双射表示，从而实现对噪声和模型不匹配具有鲁棒性的迭代逆推；在合成的harmonic entangler和带噪声的3D深度图任务中对其进行了验证。

**💡 创新点**

创新点包括：①使用结构化、可逆的参数化微分变换（如CPAB、RealNVP、Glow）构建可逆计算单元；②通过对抗性/最坏情况扰动训练诱导表示对噪声不变且保持双射；③将敏感度驱动的主动学习与逆推结合，使模型在探索时自动聚焦最关键区域；④提出将逆推视为在学习到的双射坐标系中的全局收敛优化。

**🔧 技术方法**

技术细节涵盖：可逆神经网络结构、对抗性扰动训练、拉普拉斯或co‑Lipschitz正则化、迭代高斯牛顿优化、噪声模拟与下采样、梯度裁剪、随机采样的harmonic entangler生成，以及在3D hinge渲染器上生成带噪声深度图。

**📊 数据集**

数据集：作者自行生成的合成harmonic entangler（由随机参数生成的可逆映射）和通过3D渲染器产生的128×128带噪声深度图（随后下采样为8×8），未使用公开工业或公开视觉数据集。

**📈 对比分析**

与基线MLP+Gauss‑Newton对比。Twincher在不同逆问题复杂度C和查询预算n_calls的实验中表现出更高的成功率、残差快速下降（几步内接近机器精度）以及更低的残差上界；在噪声深度图实验中，推断误差与噪声幅度呈线性关系，误差系数η≈2，显示出良好的鲁棒性。

**⚠️ 局限性**

局限性：①仅在合成或受控噪声场景下验证，缺乏真实物理系统或严重欠定逆问题的实测；②模型对容量和训练时间敏感，过大或过小的容量会导致子最优或欠拟合；③需要手工设计和调参的可逆变换结构；④在高维真实场景下的可扩展性和计算开销仍待评估。

---

## 498. SID: Sliding into Distribution for Robust Few-Demonstration Manipulation

**arXiv ID:** 2605.13428 | [PDF](https://arxiv.org/pdf/2605.13428v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 499. OP4KSR: One-Step Patch-Free 4K Super-Resolution with Periodic Artifact Suppression

**arXiv ID:** 2605.13457 | [PDF](https://arxiv.org/pdf/2605.13457v1)

**作者:** Chengyan Deng `[一作]` (University of Electronic Science and Technology of China), Li Yu `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 31190 | [OpenAlex ID](https://openalex.org/A5100345712)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种一阶无补丁的4K超分辨率方法OP4KSR，利用Flux扩散模型与极压F16 VAE实现高效直接生成4096×4096图像。

**💡 创新点**

创新点包括：①通过极压VAE实现单步推理下的4K输出；② RoPE基频重标（RFR）解决频率不匹配导致的周期性伪影；③ 自相关周期损失（ℒ_AP）精准抑制周期纹理；④构建4KSR-Train数据集与三大基准，填补4K SR研究资源空缺。

**🔧 技术方法**

技术手段包括Flux扩散框架、极压F16 VAE、RoPE基频重标、自动相关周期损失、LoRA微调、mid-timestep对齐、S3Diff文本提示等。

**📊 数据集**

使用自建4KSR-Train（约3.4万张4K图像）、四个评测基准（4KSR-Syn、4KSR-RealSquare、4KSR-RealVary）以及公开的DIV4K-50进行训练与评估。

**📈 对比分析**

与多步方法ResShift、SUPIR、DreamClear以及一阶方法SinSR、AddSR、OSEDiff、OMGSR进行对比；在4KSR-Syn和DIV4K-50上获得最低LPIPS、DISTS和最高TOPIQ-FR；在RealSquare/RealVary上无参考指标与同类方法相当甚至更优；推理时间仅5.75秒，显存仅32.88 GB，显著快于其他模型。

**⚠️ 局限性**

局限性在于极压VAE导致的高频细节恢复不足，未来计划加入高频补偿机制以进一步提升纹理细节。

---

## 500. Assessing the Creativity of Large Language Models: Testing, Limits, and New Frontiers

**arXiv ID:** 2605.13450 | [PDF](https://arxiv.org/pdf/2605.13450v1)

**作者:** Samuel Schapiro `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 8497 | [OpenAlex ID](https://openalex.org/A5103178893)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了人类创造力测验对大型语言模型（LLM）创造力预测的有效性，并提出了新的评估指标。

**💡 创新点**

创新点在于引入了“有效性”和“特异性”两项指标，并证明了理论上最大可达特异性的上限；同时设计了融合发散与聚合思维的新测试DRAT，首次显著预测LLM的科学创新能力。

**🔧 技术方法**

技术上使用了多种自动化创造力测验（DAT、CDAT、PACE、RAT）以及新构造的DRAT，并采用语义嵌入（GloVe、FastText、SBERT）来计算词距。

**📊 数据集**

使用了54个指令调优的LLM（来自10个供应商），以及六个创意基准（创意写作：Arena CW、EQ-Bench CW、Mazur CW；发散思维：Hivemind、NoveltyBench；科学构思：LiveIdeaBench）来收集分数。

**📈 对比分析**

通过与创意基准的相关性评估，发现DAT最能预测创意写作，CDAT最能预测发散思维，而DRAT在科学构思上显著提高了有效性（r≈0.57）和特异性（r|g≈0.52）。

**⚠️ 局限性**

局限性包括只评估自动化测验且样本量受限于现有模型覆盖，且DRAT对可实现性预测效果不佳，未能涵盖变革性创造力等更复杂维度。

---

## 501. TokAlign++: Advancing Vocabulary Adaptation via Better Token Alignment

**arXiv ID:** 2605.13429 | [PDF](https://arxiv.org/pdf/2605.13429v1)

**作者:** Chong Li `[一作]` (Chinese Academy of Sciences), Chengqing Zong `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 7297 | [OpenAlex ID](https://openalex.org/A5015785439)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 TokAlign++，一种高效的大语言模型词表迁移方法，利用无监督的词对对齐词典将源词表中的向量迁移到目标词表，并通过两阶段渐进式微调恢复模型性能。

**💡 创新点**

创新点在于：①不依赖于并行语料或手工词典，而是利用源模型隐藏层得到的词表示并采用 Bilingual Lexicon Induction（如 VecMap + CSLS）学习跨词表对齐；②提出两种无监督获取词表示的方法（训练 GloVe 或直接提取 LLM 隐藏状态）；③引入 BLEU‑1 / BERTScore 两个对齐质量评估指标，并通过它们指导对齐效果；④将对齐矩阵直接用于参数初始化，显著减少微调步数并提升收敛速度。

**🔧 技术方法**

主要技术包括：GloVe 词向量训练、VecMap/CSLS 对齐、token‑level 对齐矩阵构造、两阶段（embedding‑only + 全参）微调、BERTScore、BLEU‑1 评估、token‑level 以及 sentence‑level 归纳蒸馏、以及对多语言文本的压缩率和 PPL 评估。

**📊 数据集**

实验使用 Pile、CulturaX 及 15 种语言的文本数据；在 Pythia、LLaMA3、Gemma、Qwen2 等公开模型上进行词表替换；下游任务包含阅读理解、常识推理、XNLI、PAWS‑X、XCOPA、XStoryCloze 等十个基准。

**📈 对比分析**

与 Random Init、Random Perm、WECHSEL、OFA、Focus、ZeTT 等 9 种基线对比，TokAlign++ 在 15 语种下的 PPL 下降 47.6% 以上，平均压缩率提升 47.6%，在 1k 步微调后恢复 98% 原始性能，比 Baseline 快 4×，并在 token‑level 蒸馏中相较 sentence‑level 提升约 4.4%。

**⚠️ 局限性**

局限性包括：①仍需 1k 步微调，无法完全免除训练；②当目标词表明显小于源词表时，嵌入参数缺失导致性能下降；③对稀有词或特定领域词的对齐效果受限于训练语料的覆盖，可能导致低资源语言表现不佳。

---

## 502. Fast and Compact Graph Cuts for the Boykov-Kolmogorov Algorithm

**arXiv ID:** 2605.13402 | [PDF](https://arxiv.org/pdf/2605.13402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 503. Model-Agnostic Lifelong LLM Safety via Externalized Attack-Defense Co-Evolution

**arXiv ID:** 2605.13411 | [PDF](https://arxiv.org/pdf/2605.13411v1)

**作者:** Xiaozhe Zhang `[一作]` (City University of Hong Kong), Haoliang Li `[通讯]` (City University of Hong Kong)

**通讯引用:** 6455 | [OpenAlex ID](https://openalex.org/A5040091210)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出了一种针对大语言模型（LLM）的协同进化安全框架，通过外部化攻击技能库和验证内存库实现对攻击与防御的持续改进；

**💡 创新点**

创新点在于将攻击策略重构为可执行技能引擎，并在防御端采用轻量化辅助防御模型与内存检索，实现模型无关、可迁移且可持续的安全提升；

**🔧 技术方法**

技术上采用了强化学习（GRPO）进行攻击与防御的共同进化，利用技能检索器、内存检索器、早期对齐门控等机制；

**📊 数据集**

实验数据集涵盖AdvBench、CategoricalQA、HarmfulQA、DangerousQA、PKU‑SafeRLHF等攻击数据以及GSM8K、MMLU等推理数据；

**📈 对比分析**

与现有基线（如Qwen3Guard、Llama‑Guard、TriPlay‑RL等）比较，在Guard模式下达到99.61%防御成功率，优于Qwen3Guard 14.13%，参数仅为其37.5%；攻击端在多模型上实现平均ASR>90%，并能在零样本情境下保持优势；

**⚠️ 局限性**

局限性包括对外部技能库与内存库的依赖，构建与维护成本高；强化学习训练仍需较多算力，且在极大规模模型上的泛化仍待进一步验证。

---

## 504. Vector-Quantized Discrete Latent Factors Meet Financial Priors: Dynamic Cross-Sectional Stock Ranking Prediction for Portfolio Construction

**arXiv ID:** 2605.13407 | [PDF](https://arxiv.org/pdf/2605.13407v1)

**作者:** Namhyoung Kim `[一作]` (RiskX), Jae Wook Song `[通讯]` (Hanyang University)

**通讯引用:** 601 | [OpenAlex ID](https://openalex.org/A5026809761)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种两阶段的跨期股票回报预测框架 PRISM‑VQ，先用向量量化和对比学习从跨截面特征中学习离散的隐藏因子，再用结构条件的混合专家（MoE）生成时间变化的因子负载，并结合专家先验因子进行最终回报预测。

**💡 创新点**

创新点在于：① 将向量量化与对比正则结合，形成信息瓶颈，提升低信噪比金融数据的鲁棒性；② 利用离散码作为 MoE 路由信号，实现跨截面结构与时间序列动态负载的无缝耦合；③ 在保持因子可解释性的前提下，引入专家先验因子作为稳健锚点。

**🔧 技术方法**

核心技术包括：向量量化（VQ）+对比学习；Transformer 编码器（跨资产与时间维度）；结构条件 Mixture‑of‑Experts（使用离散码进行专家路由）；FiLM 归一化解码；多任务学习（重构、预测、对比）。

**📊 数据集**

使用中国 CSI 300 指数与美国 S&P 500 指数的日数据；特征窗口 20 天，目标为 5‑日向前收益，辅以 1–9 天收益做多目标；共计 13 个专家先验因子。

**📈 对比分析**

与 XGBoost、GRU、Transformer、CAE、VAE、VQVAE、DTML、MASTER、MATCC 等 10 种基线进行比较。实验结果显示 PRISM‑VQ 在 CSI 300 上 RankIC 0.0646、Sharpe 1.57；在 S&P 500 上 RankIC 0.0141、Sharpe 0.67，均显著优于所有基线（p<0.05 的块自举检验）。在 Top‑K 组合、不同交易成本、代码簿大小等敏感性实验中亦保持稳健。

**⚠️ 局限性**

局限性包括：① 对极端高效市场（如 S&P 500）仍难获得高 RankIC，表明方法受限于低信噪比；② 代码簿和专家数量等超参数需要经验调优，模型复杂度相对较高；③ 仅在两大市场验证，跨市场泛化及多因子场景下的表现尚未充分探索。

---

## 505. When is Warmstarting Effective for Scaling Language Models?

**arXiv ID:** 2605.13405 | [PDF](https://arxiv.org/pdf/2605.13405v1)

**作者:** Neeratyoy Mallik `[一作]` (University of Freiburg), Aaron Klein `[通讯]` (ELLIS Institute Tübingen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对从小模型到大模型的warmstarting进行了系统实验，提出了可通用的Shrink‑Zero‑Perturb（SZP）方法并量化了其有效增长因子上限。

**💡 创新点**

创新点在于证明初始化时不必保持函数映射，提出零填充+缩放+扰动的通用warmstarting策略，并通过实验和扩展法则给出可操作的增长上限。

**🔧 技术方法**

使用了架构无关的warmstarting设计空间、零填充、缩放、噪声扰动、μP超参数迁移，结合超参数网格搜索、梯度分量分析、二次曲线拟合和扩展法则拟合。

**📊 数据集**

实验数据集包括受控合成回归任务以及公开的SlimPajama语料，用以训练decoder‑only GPT‑2风格模型。

**📈 对比分析**

方法上与从零训练、Net2Net等传统warmstarting进行对比，结果显示在g≈2、20‑30s预算下SZP收敛更快、最终损失更低；随着g增大优势衰减，文中通过Iso‑FLOP图和扩展法则展示了优势边界。

**⚠️ 局限性**

限制在于仅研究宽度扩展、计算资源受限导致超参数搜索不完整、仅基于黑盒checkpoint、缺乏对深度扩展或多任务的验证，且增长上限需针对具体预算与模型大小而定。

---

## 506. The Diffusion Encoder

**arXiv ID:** 2605.13399 | [PDF](https://arxiv.org/pdf/2605.13399v1)

**作者:** Akhil Premkumar `[一作]` (University of California San Diego), Sarah Lucioni `[通讯]` (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究并实现了使用扩散模型作为变分自编码器中的编码器，并提出了一种交替训练方案以同步编码器与解码器。

**💡 创新点**

创新点包括基于EM思想的交替优化、利用Langevin平衡采样与扩散模型逆过程构造可同步训练的扩散编码器，以及设计信息注入的网络架构。

**🔧 技术方法**

所用技术涵盖扩散模型、Langevin动力学、EM式交替训练、神经熵/信息瓶颈、注意力网络信息注入、路径KL匹配与去方差训练目标。

**📊 数据集**

实验数据集包括 MNIST、CIFAR-10、TinyImageNet 和 CelebA‑HQ。

**📈 对比分析**

通过速率-失真曲线与传统高斯编码器 VAE 进行比较，扩散编码器虽然略逊，但接近 VAE 最优边界，并且训练过程稳定无后验崩塌。

**⚠️ 局限性**

主要局限在于 Langevin 链的顺序迭代导致训练延迟，神经熵仅是码率上界，且在高维下模型性能仍不如 VAE。

---

## 507. FPGA-Accelerated Lock Management and Transaction Processing: Architecture, Optimization, and Design Space Exploration

**arXiv ID:** 2605.13398 | [PDF](https://arxiv.org/pdf/2605.13398v1)

**作者:** Shien Zhu `[一作]` (ETH Zurich), Gustavo Alonso `[通讯]` (ETH Zurich)

**通讯引用:** 17412 | [OpenAlex ID](https://openalex.org/A5103144919)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于FPGA的事务处理加速器，使用Lock Agent和异步流水线Transaction Agent实现低延迟锁管理与高并行事务执行；

**💡 创新点**

创新点在于将锁表直接嵌入硬件锁代理、设计异步流水线事务代理以及使用锁通道表层次结构优化交叉开关，显著提升锁服务吞吐和事务吞吐；

**🔧 技术方法**

采用FPGA、HBM、高带宽内存、Coyote硬件外壳、SpinalHDL、High-Level Synthesis以及自定义的异步流水线和锁表硬件数据结构；

**📊 数据集**

使用TPC‑C基准测试的64仓库事务记录作为工作负载；

**📈 对比分析**

通过与同一节点CPU（AMD EPYC 7302P）使用多线程、不同锁表大小的基准进行对比，FPGA加速器在锁服务上提升35.2–52.1×，在事务吞吐上提升38.6–50.9×，单节点可达283K事务/秒，资源占用约30%；

**⚠️ 局限性**

主要局限在于交叉开关规模与FPGA资源、时钟周期延迟导致的可扩展性受限；此外，锁表大小和通道数量的权衡对吞吐与中止率影响较大，需要进一步优化综合与路由。

---

## 508. RS-Claw: Progressive Active Tool Exploration via Hierarchical Skill Trees for Remote Sensing Agents

**arXiv ID:** 2605.13391 | [PDF](https://arxiv.org/pdf/2605.13391v1)

**作者:** Liangtian Liu `[一作]` (Central South University), Dongyang Hou `[通讯]` (Central South University)

**通讯引用:** 823 | [OpenAlex ID](https://openalex.org/A5058638373)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 RS-Claw 框架，采用层级技能树实现遥感代理的主动工具探索与按需加载，解决传统完整注册或检索增强方式导致的上下文拥挤与工具覆盖不足问题。

**💡 创新点**

创新点：① 把工具获取视为代理自身的决策空间，构建“主动探索”范式；② 设计三层技能树（技能摘要、工具目录、工具文档）实现渐进式信息披露；③ 通过 POMDP 建模与 ReAct 推理，实现工具信息的动态可见性与调用控制。

**🔧 技术方法**

技术手段：POMDP 统一建模、ReAct 思考-行动-观察循环、层级技能树构造、渐进式信息披露策略、LLM 与外部工具的交互接口、基于检索的 RAG 作为对照。

**📊 数据集**

数据集：Earth‑Bench（248 题，除 14 题外 234 题）作为遥感任务评测基准。

**📈 对比分析**

对比方法：与 Flat（完整注册）和 RAG（检索增强）两种被动工具选择基线，使用 GPT‑5、DeepSeek‑V3.1、Qwen3‑32b 三大 LLM，评估指标包括准确率、Tool‑Any‑Order、Tool‑In‑Order、token 消耗。RS‑Claw 在所有模型与评估模式下均优于两基线，Qwen3‑32b AP 模式准确率提升 12.45%，token 压缩率最高可达 86%，每回合 token 量显著降低。

**⚠️ 局限性**

局限性：① 技能树结构需人工定义，迁移到新域或工具集增长时需重新设计；② 依赖 LLM 的规划能力，若规划不足仍可能做出错误探索；③ 两步信息披露增加交互轮次，规模极小的工具集时开销可能不划算；④ 仅在 Earth‑Bench 与三款模型上验证，未覆盖更大规模或跨域工具环境。

---

## 509. Strategic PAC Learnability via Geometric Definability

**arXiv ID:** 2605.13426 | [PDF](https://arxiv.org/pdf/2605.13426v1)

**作者:** Yuval Filmus `[一作]` (Technion Israel Institute of Technology), Alexander Shlimovich `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac`

**🎯 论文内容**

本文研究在可被受试者策略性修改特征的分类学习问题，探讨在策略性变换下诱导假设类的可学习性与样本复杂度，给出正负例并提出基于可定义性约束的可学习性条件。

**💡 创新点**

核心创新在于：①展示即使原类VC维度为1，简单的区间邻域也能使策略化后的类不可学习；②引入ℝ_exp可定义性假设，证明在该框架下策略化类可学习并给出样本复杂度的上界；③进一步提供可构造的量化样本复杂度和VC维度上界。

**🔧 技术方法**

采用模型论中的可定义性（o‑minimal、ℝ_exp结构）、量化量化消除、Pfaffian函数细分与细胞分解等理论工具，结合学习理论中的成长函数、VC维度分析。

**📊 数据集**

该工作为纯理论研究，无实验数据集；所有结果均通过数学证明与理论界定得到。

**📈 对比分析**

与传统VC理论相比，本文的策略化类样本复杂度上界更为细致，能在可定义性约束下实现对增长函数的精确控制，理论上优于一般VC上界。

**⚠️ 局限性**

局限性包括：①对可定义性约束的依赖，无法覆盖非可定义或周期性邻域；②量化上界在ℝ_exp中仍依赖Wilkie非构造性结果，具体常数未知；③对更一般的分布式或优化定义的邻域缺乏可行的可学习性分析。

---

## 510. PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents

**arXiv ID:** 2605.13481 | [PDF](https://arxiv.org/pdf/2605.13481v1)

**作者:** Mikhail Menschikov `[一作]` (Skoltech), Evgeny Burnaev `[通讯]` (Skoltech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PersonalAI 2.0 (PAI-2)，一种利用大型语言模型与知识图谱融合的问答框架，支持动态多阶段查询与计划增强；

**💡 创新点**

创新点在于将搜索计划的动态迭代与基于实体匹配的线性组合生成线索查询相结合，并采用多路径图遍历（BeamSearch、WaterCircles）提升检索精度；

**🔧 技术方法**

核心技术包括 LLM‑驱动的预处理与实体抽取、图结构检索与过滤、线索查询生成、图遍历、答案聚合与计划优化；

**📊 数据集**

在六个公开基准上评估：Natural Questions、TriviaQA、HotpotQA、2WikiMultihopQA、MuSiQue、DiaASQ；

**📈 对比分析**

与 LightRAG、RAPTOR、HippoRAG 2 等基线相比，PAI‑2 在 4/6 组任务中平均提升 4%（LLM‑as‑Judge），计划增强带来 18% 提升，图遍历比扁平检索提升 6%；

**⚠️ 局限性**

局限包括对时间表达的显式处理不足、语义去重缺失、实体歧义处理不完善、查询计划对复杂问题的适应性仍有限。

---

## 511. FedHPro: Federated Hyper-Prototype Learning via Gradient Matching

**arXiv ID:** 2605.13475 | [PDF](https://arxiv.org/pdf/2605.13475v1)

**作者:** Huan Wang `[一作]` (University of Wollongong), Guansong Pang `[通讯]` (Singapore Management University)

**通讯引用:** 6162 | [OpenAlex ID](https://openalex.org/A5039104219)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了联邦超原型学习框架FedHPro，通过超原型实现跨客户端语义一致的全局信号；

**💡 创新点**

创新点在于将可学习的超原型通过梯度匹配与真实样本梯度对齐，解决传统原型在异构联邦环境中的语义漂移问题，并结合互对比学习与一致性正则进一步提升类间可分性与类内一致性；

**🔧 技术方法**

使用梯度匹配、互对比学习（HPCL）、平滑一致性正则（HPAL）、交叉熵损失、标准FedAvg通信策略；

**📊 数据集**

在Digits、Office‑Caltech、CIFAR‑10/100、TinyImageNet、CIFAR‑10‑LT等多种标签、量化与域不平衡的联邦场景下进行实验；

**📈 对比分析**

相较于FedAvg、FedProx、MOON、FedProto、FedTGP、FedGMKD、FedRCL、FedSA等8个SOTA基线，FedHPro在所有设置下均取得显著提升（例如Digits平均准确率从77.75%提升至84.80%，Office‑Caltech平均准确率从55.42%提升至64.52%），且收敛速度更快；

**⚠️ 局限性**

限制主要体现在对超原型长度和匹配轮次的超参数敏感，过大或过小会导致优化不稳定；此外，当前方法主要针对图像分类任务，对其他模态（如文本、语音）仍需进一步验证。

---

## 512. On the Complexity of the Minimum-($k,ρ$)-Shortcut Problem

**arXiv ID:** 2605.13474 | [PDF](https://arxiv.org/pdf/2605.13474v1)

**作者:** Tatiana Rocha Avila `[一作]` (Goethe University Frankfurt), Conrad Schecker `[通讯]` (Goethe University Frankfurt)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5078920586)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文研究了 Minimum-(k,ρ)-Shortcut 问题的计算复杂性，给出了新的 NP‑hardness 归约与无向图 ρ=k+1 的多项式求解；

**💡 创新点**

创新点在于采用简单的 Hitting Set 归约，将硬度阈值从 k≥3 降至 k≥2，并首次证明无向图 ρ=k+1 可多项式求解；

**🔧 技术方法**

主要技术包括归约构造、图论结构分析以及对无向 ρ=k+1 情况下的 Courcelle 定理应用；

**📊 数据集**

论文未使用任何实验数据集；

**📈 对比分析**

由于缺乏实验，未给出性能比较或结果；

**⚠️ 局限性**

限制在于有向图 ρ=k+1 的复杂性仍未确定，且无向图 ρ=k+1 的解法依赖 Courcelle 定理，缺乏直接高效的组合算法。

---

## 513. Sleeper Channels and Provenance Gates: Persistent Prompt Injection in Always-on Autonomous AI Agents

**arXiv ID:** 2605.13471 | [PDF](https://arxiv.org/pdf/2605.13471v1)

**作者:** Narek Maloyan `[一作]`, Dmitry Namiot `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了永远在线的操作系统级 AI 代理（如 OpenClaw、Hermes Agent）中出现的“睡眠通道”攻击，给出了攻击的形式化定义、两维分类（持久化 substrate 与触发‑分离），并通过一系列实例展示攻击路径；随后提出了分层的基于 provenance 与动作实例摘要的防御体系（D0‑D3），并给出完整的安全定理与可执行参考实现。

**💡 创新点**

创新点包括：①在 OS‑live 代理环境中首次统一阐述睡眠通道攻击的两维分类；②将内存、技能、文件系统、调度等多种持久化 substrate 结合跨 surface 的触发机制；③设计基于动作实例摘要（SHA‑256）与一次性 nonce 的“完整介质”门控，能够防止伪造、重放与 paraphrase laundering；④提供形式化的安全定理与针对 OpenClaw 的源代码级回归测试，构建可验证的安全边界。

**🔧 技术方法**

技术手段包括：源代码标签（channel、principal、device）与标签传播、因果集构造、动作实例摘要、硬件安全通道（签名、nonce）、十个完整介质钩子（H1–H10），以及三层防御（D0：无检查；D1：模型内提示+标签；D2：外部门控 + 单次授予；D3：技能能力清单）。

**📊 数据集**

主要使用 OpenClaw 代码库（特定提交）和其内置的邮件、cron、技能等功能作为攻击与防御的测试基准；实验数据来自 20 次单向攻击模拟和 42 条单元测试，未涉及大规模公开数据集。

**📈 对比分析**

对比方法：对 D0、D1、D2 三个防御层进行 20 次模拟攻击，结果显示 D1 允许攻击、D2 阻止攻击；测试覆盖 42 条用例，全部在 Node ≥20 环境下通过；性能指标（执行时间、内存占用）未在本文给出，后续计划在更大规模样本上评估。

**⚠️ 局限性**

局限性：仅在 OpenClaw 的固定提交上验证，未完成全量部署与压力测试；A2/A3/A5 仅为草图，缺乏完整实现；防御对适应性攻击的鲁棒性尚未实验；完整介质仍需 sandbox 以防止 FFI/系统调用逃逸；安全定理基于若干运行时假设（如钩子覆盖、签名验证）未在所有环境中证明。

---

## 514. Efficient Sensor Fusion for Gesture Recognition on Resource-Constrained Devices

**arXiv ID:** 2605.13462 | [PDF](https://arxiv.org/pdf/2605.13462v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 515. PDCR: Perception-Decomposed Confidence Reward for Vision-Language Reasoning

**arXiv ID:** 2605.13467 | [PDF](https://arxiv.org/pdf/2605.13467v1)

**作者:** Hee Suk Yoon `[一作]` (Korea Advanced Institute of Science and Technology), Chang D. Yoo `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 6534 | [OpenAlex ID](https://openalex.org/A5073287748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Perception‑Decomposed Confidence Reward (PDCR)，一种针对视觉‑语言推理的自监督奖励分解框架。

**💡 创新点**

创新点在于：① 通过可解释的 Visual Dependence Score 对推理步骤进行无监督聚类，将视觉感知步骤与文本推理步骤区分；② 在每个聚类内部对置信度增益进行归一化，从而消除因技能稀疏导致的全局信号退化；③ 只使用模型内部信号，无需额外奖励模型或人工标签。

**🔧 技术方法**

使用技术包括：基于对比（真实图像 vs. 空白图像）的对数似然比得分、Otsu阈值化聚类、奖励归一化（min‑max），以及强化学习框架 Group Relative Policy Optimization (GRPO) 与其改进版本 DAPO。

**📊 数据集**

训练集：Vision‑SR1（约 47k 例）；评估集：MMMU、MMMU‑Pro、RealWorldQA、VisNumBench、MathVerse、MATH‑Vision、HallusionBench。

**📈 对比分析**

与 GRPO、DAPO、PACR 进行对比；在 Qwen2.5‑VL‑3B 上平均分 45.2，Qwen2.5‑VL‑7B 上平均分 52.9，均高于所有基线（PACR 44.4/52.2，GRPO 43.6/51.5，DAPO 44.1/52.0），并在训练过程中更快收敛、生成更简洁的推理链。

**⚠️ 局限性**

局限性包括：① 额外的前向传播导致训练成本略高；② 依赖于可解释的视觉依赖分数，对不同 VLM 结构的迁移性尚未完全验证；③ 仍局限于视觉‑语言推理场景，无法直接推广到纯文本或其他多模态任务。

---

## 516. A Unified Three-Stage Machine Learning Framework for Diabetes Detection, Subtype Discrimination, and Cognitive-Metabolic Hypothesis Testing

**arXiv ID:** 2605.13464 | [PDF](https://arxiv.org/pdf/2605.13464v1)

**作者:** Vishal Pandey `[一作]` (Independent Researcher), Rishav Tewari `[通讯]` (Asansol Engineering College)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文提出了一个三阶段机器学习框架，分别用于糖尿病二分类检测、无监督亚型聚类和糖代谢与认知功能的相关性检验。

**💡 创新点**

创新点在于：①同时解决指标不完整、亚型区分与T3型糖尿病假设验证三大空白；②采用可解释性SHAP与多模型对比，确保结果可复现；③在公开纵向认知数据上首次进行T3型糖尿病的统计假设检验。

**🔧 技术方法**

技术手段包括SVM‑RBF、Logistic Regression、Random Forest、Extra Trees、Gradient Boosting与堆叠集成；SHAP值解释；K‑Means聚类并用轮廓系数、Davies‑Bouldin、Calinski‑Harabasz评估；Spearman相关与Kruskal‑Wallis检验。

**📊 数据集**

使用的数据集为NCSU Diabetes Dataset（含13个临床特征）、Pima Indians Diabetes Database（对照基准）以及Ohio Longitudinal Cognitive Dataset（n=373的认知与代谢纵向数据）。

**📈 对比分析**

通过分层5折交叉验证对五个基准模型进行性能比较，SVM‑RBF与Logistic Regression在ROC‑AUC上最高（0.825±0.026），Random Forest在准确率上最高（0.762±0.030）；聚类选择k=2，轮廓系数约0.116；Spearman相关显著（ρ=0.208，p<5.29×10⁻⁵，Holm校正后仍显著）。

**⚠️ 局限性**

局限性包括：NCSU数据集样本量与流行率不公开；聚类无真实亚型标签，验证仅基于生理推断；Ohio数据集为横断面，未能追踪时间动态；堆叠集成未完成收敛；样本量较小导致T3型糖尿病检验功效有限。

---

## 517. Bayesian In Vivo Tracking of Synapses using Joint Poisson Deconvolution and Diffeomorphic Registration

**arXiv ID:** 2605.13455 | [PDF](https://arxiv.org/pdf/2605.13455v1)

**作者:** Shashwat Kumar `[一作]` (Johns Hopkins University), Anuj Srivastava `[通讯]` (Johns Hopkins University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种统一的贝叶斯框架，直接从低信噪比的多光子显微镜原始光子计数中估计突触的模板位置、荧光强度以及随时间变化的非线性组织变形，实现突触的持续跟踪。

**💡 创新点**

创新点包括：①将突触建模为在差分同胚变形下移动的点源；②在生成模型中显式引入光学点扩散函数（PSF）和泊松观测噪声；③使用 Hamiltonian Monte Carlo（NUTS）对所有参数（位置、强度、动量、背景）进行联合推断，并提供不确定性估计；④在同一模型中同时完成去噪、去卷积、配准和跟踪。

**🔧 技术方法**

技术：贝叶斯生成模型、差分同胚变形（稀疏动量参数化）、高斯PSF卷积、泊松观测模型、Hamiltonian Monte Carlo（NUTS）推断、PyMC实现。

**📊 数据集**

数据集：①基于同一模型生成的二维+三维+多时相仿真数据；②真实的SEP‑GluA2转基因小鼠两光子荧光突触成像数据（100×100×60 µm体积，八次采样，跨度两周）。

**📈 对比分析**

与传统的去噪+分割+跟踪（如XTC、NLM、BM3D）以及仅去噪的基线进行比较。实验表明：在泊松对数似然、均方根误差和时间一致性指标上，本方法均优于XTC，且在 RMSE 方面与 NLM、BM3D 接近，显著提高了突触识别的时间连贯性与轨迹一致性。

**⚠️ 局限性**

局限性：①MCMC 推断计算量大，难以扩展到更大体积或更长时间序列；②在突触密度高、PSF 重叠严重的场景下，识别性能下降，存在不可辨识性限制；③PSF 的简化为固定高斯形状，可能不足以捕捉真实显微镜系统的细节。

---

## 518. LongBEL: Long-Context and Document-Consistent Biomedical Entity Linking

**arXiv ID:** 2605.13451 | [PDF](https://arxiv.org/pdf/2605.13451v1)

**作者:** Adam Remaki `[一作]` (Sorbonne Université), Christel Gérardin `[通讯]` (Sorbonne Université)

**通讯引用:** 56 | [OpenAlex ID](https://openalex.org/A5051913367)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出LongBEL，一种利用全文上下文和预测记忆的文档级生成式生物医学实体链接框架；

**💡 创新点**

创新点在于将全文上下文与跨提及的记忆机制结合，并通过交叉验证的预测记忆训练来降低误差传播；

**🔧 技术方法**

使用基于Llama的自回归生成模型、语义引导的受限解码、RRF集成、交叉验证记忆构建；

**📊 数据集**

在五个多语言（英语、法语、西班牙语）BEL基准上评估，包括MedMentions‑ST21pv、QUAERO‑EMEA、SympTEMIST、DisTEMIST、MedProcNER；

**📈 对比分析**

与无上下文、局部上下文、记忆单独或组合的基线相比，LongBEL在Recall@1上均有提升，尤其在概念频繁重复的语料上显著提升（如MM‑ST21pv提升约4%）；

**⚠️ 局限性**

局限性包括对已标注提及和语义组的依赖、对概念重复率和样本量的敏感、误差传播风险、以及相对较高的推理成本。

---

## 519. Pretraining Language Models with Subword Regularization: An Empirical Study of BPE Dropout in Low-Resource NLP

**arXiv ID:** 2605.13436 | [PDF](https://arxiv.org/pdf/2605.13436v1)

**作者:** Ruan Visser `[一作]` (Stellenbosch University), Marcel Dunaiski `[通讯]` (Stellenbosch University)

**通讯引用:** 250 | [OpenAlex ID](https://openalex.org/A5015481306)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在低资源自然语言处理任务中，BPE Dropout（子词正则化）是否应同时应用于预训练和微调阶段，并系统评估其对多语言BERT模型在XNL、PAWS‑X、PAN‑X、MasakhaNER 2.0等下游任务的性能影响；同时引入形态学对齐分析，探讨BPE Dropout产生更好对齐分词的机制；并尝试在微调阶段使用MorphyNet形态学替换来补偿缺失的预训练Dropout效果。

**💡 创新点**

首次系统比较预训练-微调阶段一致使用BPE Dropout的优势，揭示预训练-微调不匹配导致的性能损失；提出基于形态学边界的对齐评估方法，验证BPE Dropout能产生更好对齐的分词；证明在微调阶段引入形态学对齐替换可部分弥补缺失预训练Dropout的收益。

**🔧 技术方法**

BPE Dropout（子词正则化）在预训练与微调阶段的变体；BERT Encoder架构；形态学边界对齐评估（MorphScore + boundary‑level F1）；在微调阶段引入MorphyNet形态学分割替换；多语言（英、德、法、西、斯瓦希里、伊西科萨）及双语混合预训练。

**📊 数据集**

1 GB/100 MB单语或双语文本；XNL、PAWS‑X、PAN‑X、MasakhaNER 2.0；MNLI/ XNLI、PAWS‑X等下游任务数据集。

**📈 对比分析**

在10个随机种子上评估；对比 deterministic tokenization baseline、FTD（仅微调Dropout）与 PTD+FTD（预训练+微调Dropout）；在低资源设置下 PTD+FTD 在 XNLI、PAWS‑X、NER 等任务上平均提升约 1–3 %（甚至更高），随预训练/微调数据增大提升趋于平缓。

**⚠️ 局限性**

实验仅覆盖BERT编码器，缺乏对其他模型或更大规模模型的验证；BPE Dropout对形态学对齐的提升有限，主要通过增加对齐分词出现频率实现；对低资源语言的MorphyNet资源依赖有限；未系统探讨不同语言内部差异对结果的影响。

---

## 520. Q-Flow: Stable and Expressive Reinforcement Learning with Flow-Based Policy

**arXiv ID:** 2605.13435 | [PDF](https://arxiv.org/pdf/2605.13435v1)

**作者:** JaeHyeok Doo `[一作]` (KAIST AI), Minjoon Seo `[通讯]` (KAIST AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `40105733-5154-44cd-8090-a8cab9e64b07`

**🎯 论文内容**

通过引入流一致的中间价值函数，提出 Q-Flow 框架，使流模型的策略优化能够在不使用 BPTT 的情况下保持稳定。

**💡 创新点**

创新点在于将终端奖励显式传播到流过程中的每个隐状态，利用中间价值梯度匹配实现稳定且高表达力的梯度更新，从而解决了流模型表达力与优化稳定性的矛盾。

**🔧 技术方法**

采用连续正则化流（Conditional Flow Matching）作为策略参数化，结合中间价值学习与梯度匹配，并使用目标网络平滑价值目标。

**📊 数据集**

在 2D 合成环境和 OGBench 离线强化学习任务集（覆盖机器人运动与操作场景）上进行实验。

**📈 对比分析**

与 Gaussian、Diffusion 及多种 Flow 基线相比，Q-Flow 在 OGBench 标准设置下平均提升 10.6%（最优提升 31%），在线适配中提升约 23%，同时训练速度更快，避免了 BPTT 的昂贵计算。

**⚠️ 局限性**

受限于中间价值函数随策略演化产生的非平稳性，尤其在需要动作分块的复杂操作任务中价值估计误差较大，导致性能下降。

---

## 521. Continual Learning with Multilingual Foundation Model

**arXiv ID:** 2605.13415 | [PDF](https://arxiv.org/pdf/2605.13415v1)

**作者:** Barathi Ganesh HB `[一作]` (Kitami Institute of Technology), Juuso Eronen `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多阶段框架，用于检测英语、西班牙语和意大利语推文中的 LGBTQ+ 俚语回收使用。

**💡 创新点**

创新点在于结合语义保持的后向翻译数据增强、跨语言动态下采样、领域知识注入的掩码语言建模，以及针对每种语言优化阈值的后处理，从而在多语言环境下显著提升回收检测性能。

**🔧 技术方法**

主要技术包括 XLM‑RoBERTa‑Large 作为基础模型，GPT‑4o‑mini 进行后向翻译增强，动态 epoch‑level 1:3 正负样本下采样，Optuna TPE 进行超参数搜索，掩码语言建模（MLM）用于领域适配，以及 ROC 分析得到语言特定阈值。

**📊 数据集**

使用 MultiPRIDE 共享任务数据集，包含 3 种语言的已标注回收/非回收俚语推文，原始样本量约为 N，后向翻译后三倍扩充。

**📈 对比分析**

通过四个运行（RUN1–RUN4）进行比较：RUN1 为基本增广+下采样，RUN2 在 RUN1 基础上加入 MLM，RUN3/4 为分别对 RUN1/2 结果应用语言阈值优化。实验显示宏观 F1 在 RUN1 为 0.76±0.04，RUN2 约提升 1–2%，阈值优化后 RUN3/4 进一步提升 2–5% F1，整体在多语言回收检测任务中达到最佳性能。

**⚠️ 局限性**

局限性包括：仅基于文本特征，难以捕捉讽刺、假设语境和隐含团体文化；意大利语回收检测仍表现不佳，表明对文化细微差异的理解不足；MLM 对不同语言的收益不均，需要更精细的超参数调优和更多领域特定数据；缺乏非文本信号（表情、用户交互）导致性能上限受限。

---

## 522. TRIAGE: Evaluating Prospective Metacognitive Control in LLMs under Resource Constraints

**arXiv ID:** 2605.13414 | [PDF](https://arxiv.org/pdf/2605.13414v1)

**作者:** Zabir Al Nazi `[一作]` (University of California, Riverside), Shubhashis Roy Dipta `[通讯]` (University of Maryland, Baltimore County)

**通讯引用:** 101 | [OpenAlex ID](https://openalex.org/A5075241448)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TRIAGE框架，用来评估大语言模型在有限token预算下的前瞻性元认知控制，即在未执行任何任务前决定选哪些任务、按何顺序执行以及给每个任务分配多少token。

**💡 创新点**

创新点在于：①将元认知控制抽象为可测量的计划问题并通过oracle‑vs‑random归一化得到triage efficiency；②引入advisory与binding两种执行模式，分别考察模型的自我估计与执行承诺；③在四大任务领域对20种前沿与开源模型进行系统跨域评估，揭示元认知控制与单任务能力不一致的现象。

**🔧 技术方法**

技术手段包括：使用LLM生成带token预算的有序计划；利用0‑1背包求解得到oracle最优价值；通过随机抽样得到随机基准；计算triage效率η与归一化 regret；对计划执行采用两种规则（U、E）来分别评估监测与控制能力。

**📊 数据集**

数据集：竞赛数学 AIME 2024–2025、研究生级科学 GPQA Diamond、代码生成 LiveCodeBench、跨学科专家知识 Humanity's Last Exam；并在 GPQA、MMLU‑Math 中注入 AbstentionBench 的无解题目，以检验模型识别不可解任务的能力。

**📈 对比分析**

比较方法：在不同α预算（0.25/0.5/0.75/1.0）下，针对每个模型和数据集计算η_U、η_E、归一化 regret；与oracle（完美自知）和随机（无自知）对比。结果显示：大多数模型在advisory模式下表现超过随机，但在binding模式下η往往为负，说明模型难以按计划分配token；extended reasoning能提升单任务准确率，却未提升triage效率；跨域性能差异大，缺乏一致的元认知优势。

**⚠️ 局限性**

局限性：①模型的预算承诺能力差，计划往往无法被自己执行；②reasoning 训练在某些模型上降低了对不可解任务的识别；③评估只关注token成本，未考虑其他资源维度；④缺少动态反馈或在线重新规划的机制；⑤模型规模并未总是带来更好元认知控制。

---

## 523. Cognifold: Always-On Proactive Memory via Cognitive Folding

**arXiv ID:** 2605.13438 | [PDF](https://arxiv.org/pdf/2605.13438v1)

**作者:** Suli Wang `[一作]`, Xinliang Zhou `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种始终在线的主动式智能体记忆系统 Cognifold，通过持续折叠事件流形成多层认知结构，实现自动化的概念抽象和意图生成。

**💡 创新点**

将 Complementary Learning Systems 扩展到三层（海马、颞皮质、前额叶）并在图形层面实现意图层的自发生成；通过聚合、压缩、衰减、补全四个结构债务自动维护图形拓扑；采用主动上下文组装、实时结构演化的写入路径，实现真正的主动记忆。

**🔧 技术方法**

Typed Directed Multigraph、LLM 执行器与语义映射、PageRank+时间衰减+访问频率优先级的写入上下文选择、概念合并与 kNN 补全、指数衰减、预期阈值自适应、基于多模态嵌入。

**📊 数据集**

CogEval-Bench（自构造的结构诊断基准，包含6个场景、4个领域）以及7个下游基准（MuTual、ToMi、MuSiQue、NarrativeQA、StreamingQA、LoCoMo、BABILong）。

**📈 对比分析**

与 OpenIE KG、Cognee、HippoRAG 2、GraphRAG、Mem0、Zep、ENGRAM 等现有记忆/检索系统对比；在 CogEval-Bench 上取得 Harmony 0.476、Purity 0.361、压缩率 4.6×、Proactivity 0.614；在下游基准上超过或与最佳方法相当，例如 MuSiQue F1 58.7、MuTual、ToMi 等均领先。

**⚠️ 局限性**

对事件顺序的路径依赖导致不同排列生成不同图形；前额叶映射仅限于模式整合，缺乏奖励评估、认知控制和反事实模拟；目前仅在文本/事件层面处理，未覆盖更复杂的多模态场景。

---

## 524. Seconds-Aligned PCA-DAC Latent Diffusion for Symbolic-to-Audio Drum Rendering

**arXiv ID:** 2605.13404 | [PDF](https://arxiv.org/pdf/2605.13404v1)

**作者:** Konstantinos Soiledis `[一作]` (Hellenic Mediterranean University), Konstantinos Tsamis `[通讯]` (Hellenic Mediterranean University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5117588445)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于PCA压缩的DAC隐空间扩散模型，用秒级对齐的符号控制生成真实鼓音频。

**💡 创新点**

创新点在于：①将鼓符号在物理时间上对齐到编码帧，避免时序错配；②使用72维PCA坐标作为连续扩散目标，既压缩维度又保持可解码性；③加入基于残差向量量化的交叉熵辅助损失，提高低步数扩散效率。

**🔧 技术方法**

主要技术包括：Transformer式扩散去噪器、秒级符号前端、多尺度卷积+LSTM编码器、PCA降维、DAC（EnCodec 变体）离线编码器/解码器、辅助RVQ交叉熵正则。

**📊 数据集**

使用从Groove MIDI Dataset 提取的鼓表演音频与MIDI配对的数据集，共 11,523 训练、1,534 验证、1,733 测试四拍窗口。

**📈 对比分析**

对比了重建上限、符号渲染、NN检索、直接PCA回归、无辅助扩散与带RVQ-CE扩散。扩散模型在配对的谱、瞬态和节奏一致性指标上明显优于回归与符号渲染；RVQ-CE 在 6–12 步时显著提升指标且采样更快；但在相位敏感的 Waveform L1 上，直接回归仍更好。

**⚠️ 局限性**

局限包括：仅针对四拍短窗口；只接受显式鼓格而非文本/音频提示；PCA目标固定且与DAC配置相关；评估缺乏人类听感实验；单一随机种子导致未测量采样变异；RMS 误差受增益处理影响。

---

## 525. Benchmarking the Open Science Data Federation services to develop XRootD best practices

**arXiv ID:** 2605.13593 | [PDF](https://arxiv.org/pdf/2605.13593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 526. CUBic: Coordinated Unified Bimanual Perception and Control Framework

**arXiv ID:** 2605.13452 | [PDF](https://arxiv.org/pdf/2605.13452v1)

**作者:** Xingyu Wang `[一作]` (Beihang University), Zhaoxin Fan `[通讯]` (Beihang University)

**通讯引用:** 776 | [OpenAlex ID](https://openalex.org/A5021141988)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为 CUBic 的统一框架，用于双臂机器人通过视觉输入实现协同感知与控制。

**💡 创新点**

创新点在于：①将双臂协调问题转化为共享的离散化 token 空间，既保持各臂独立，又能自然产生协作；②使用双代码簿共享映射实现隐式跨臂编码；③采用两阶段训练策略，先独立感知再联合控制；④在单一 diffusion 变换器中融合跨臂注意力，完成动作生成。

**🔧 技术方法**

技术手段包括：多视角 ResNet‑18 特征提取 + 位置嵌入；离散化 VQ‑tokenization 与残差量化；双代码簿共享映射；masked‑attention 的单向感知聚合；DiT（Diffusion Transformer）作为策略网络；两阶段预训练 + 后训练联合训练。

**📊 数据集**

使用 RoboTwin 机器人模拟基准（含 7 个双臂任务）以及真实 Agibot 双臂机器人上的 6 个物体交互任务作为数据集。

**📈 对比分析**

与 Diffusion Policy (DP)、DP3、GR‑MG 等现有视觉-动作策略基线对比，CUBic 在 RoboTwin 上平均成功率提升 12%（相较 DP3），在真实任务中显著高于 DP，整体表现均优于传统方法。

**⚠️ 局限性**

局限性包括：①需要两阶段训练和精细的超参数调优；②共享代码簿可能在更大规模或更异构任务中限制表达能力；③在独立代码簿设置下性能显著下降，表明模型对共享结构高度依赖；④在极端动态或多物体环境下的可迁移性尚待验证。

---

## 527. Support-Conditioned Flow Matching Is Kernel Smoothing

**arXiv ID:** 2605.13386 | [PDF](https://arxiv.org/pdf/2605.13386v1)

**作者:** Daniel Matsui Smola `[一作]` `[通讯]` (University of Washington), Daniel Matsui Smola (University of Washington)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在流匹配中使用交叉注意力进行支持集条件生成的理论基础，揭示其等价于 Nadaraya‑Watson 核平滑；

**💡 创新点**

证明了单个高斯核注意力头能够精确实现流匹配的速度场，并提出了三种经典核估计的失败模式（高维近邻崩塌、几何不匹配、支持稀缺），以及学习模型如何在每种模式下进行补救；

**🔧 技术方法**

利用流匹配、最优传输、核密度估计和变形注意力（Gaussian‑kernel cross‑attention）等技术；

**📊 数据集**

在合成高斯混合、球面壳以及 DINOv2 ImageNet 预训练特征（经 PCA 降维）等数据集上进行实验；

**📈 对比分析**

通过将“精确头”开启/关闭与纯学习头相结合，使用 MMD² 和 C2ST 等指标进行对比。结果表明：在低维时精确头显著提升；随着维度增大或核形状不匹配时优势消失；在小支持量时学习模型能显著弥补缺陷；总体提升范围在 20%–30% 左右；

**⚠️ 局限性**

主要局限包括：仅针对高斯核注意力和高斯 OT 路径，未给出非高斯路径（如 VP）下的严格理论；对实际模型（如 IP‑Adapter）仅有经验相关性验证；未给出学习模型优于插件的正式误差界限；并且对 meta‑分布 Π 的选择和泛化能力尚未完全阐明。

---

## 528. Inducing Artificial Uncertainty in Language Models

**arXiv ID:** 2605.13595 | [PDF](https://arxiv.org/pdf/2605.13595v1)

**作者:** Sophia Hager `[一作]`, Nicholas Andrews `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出在缺乏挑战性数据时，通过人为诱导大语言模型产生不确定性来训练不确定性估计器，并对比了 dropout 与 unlearning 两种人工不确定性生成方式；

**💡 创新点**

创新点在于首次系统性研究“人工不确定性”概念，即利用易数据在模型内部引入不确定性，从而提升在难数据上的校准，并提出了无监督的超参数选择方法；

**🔧 技术方法**

主要技术包括在 LLM 隐层输出上训练线性 probe 以估计不确定性；使用在注意力层加入 dropout 或对特定样本做梯度上升的 unlearning 以增加模型不确定性；并采用预测结果方差来挑选最合适的超参数；

**📊 数据集**

使用的数据集为：易数据集 SciQ（以及 CommonsenseQA 作为辅助），难数据集 GPQA 与 MMLU-pro 作为评估基准，ARC‑Easy 作为易数据的验证；

**📈 对比分析**

方法通过 Brier、ECE 与 AUROC 进行对比，结果显示 dropout 通常能在硬数据上显著降低 Brier（提升校准），在 12 组实验中 11 组优于基线，Unlearning 也有提升但效果略逊，且对易数据的校准下降更小；

**⚠️ 局限性**

局限性包括：unlearning 需要较高计算成本；对易数据的校准有轻微下降；方法在极大模型上仍可能存在校准不足；且人工不确定性生成效果依赖于对模型层的适当干预，需要经验选择。

---

## 529. Spatiotemporal downscaling and nowcasting of urban land surface temperatures with deep neural networks

**arXiv ID:** 2605.13566 | [PDF](https://arxiv.org/pdf/2605.13566v1)

**作者:** Solomiia Kurchaba `[一作]` (Delft University of Technology), Angela Meyer `[通讯]` (Delft University of Technology)

**通讯引用:** 1018 | [OpenAlex ID](https://openalex.org/A5050747578)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了基于深度学习的两阶段模型，先用U-Net将来自地球静止轨道卫星的15分钟3km LST下采样到1km，再用ConvLSTM在此基础上对欧盟大城市进行15‑75分钟的高频LST现在预测。

**💡 创新点**

首次在全欧洲范围内实现以地球静止卫星为输入、MODIS为目标的1km分辨率LST下采样，并将其与时序ConvLSTM结合实现城市级子小时现在预报，显著突破空间‑时间分辨率权衡。

**🔧 技术方法**

使用U-Net编码器‑解码器实现空间下采样，ConvLSTM进行时序预测，并对比持久性与滚动中位数基线。

**📊 数据集**

训练集基于2004‑2025年间的SEVIRI（MSG）LST 15分钟3km与MODIS（Terra/Aqua）LST 1km，并选取欧盟人口>100万城市的暖季数据，测试集为特定年份和城市。

**📈 对比分析**

在保留测试集上，U-Net下采样RMSE 1.92°C，MBE 0.01°C；ConvLSTM 15‑75分钟RMSE 0.57‑1.15°C，优于持久性和滚动中位数；与MODIS验证时夜间RMSE 0.88‑1.44°C，白天RMSE 1.96‑2.7°C，整体性能稳健。

**⚠️ 局限性**

仅覆盖暖季，缺乏冬季/全年适用；仅利用LST与太阳辐射角，无地表覆盖、植被等辅助因子，导致在极端或多样化城市环境下的可推广性受限；MODIS观测稀缺限制评估时段。

---

## 530. Temper and Tilt Lead to SLOP: Reward Hacking Mitigation with Inference-Time Alignment

**arXiv ID:** 2605.13537 | [PDF](https://arxiv.org/pdf/2605.13537v1)

**作者:** Ye Wang `[一作]` (Mitsubishi Electric Research Laboratories), Toshiaki Koike-Akino `[通讯]` (Mitsubishi Electric Research Laboratories)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种推理时对齐框架SLOP，可将多种生成模型与奖励模型通过加权融合来提升对齐效果，并通过权重校准减少奖励劫持；

**💡 创新点**

创新点在于将参考模型温度调节推广为Sharpened Logarithmic Opinion Pool，并设计基于校准样本的权重优化算法；

**🔧 技术方法**

采用KL正则化奖励最大化、温度调节、SLOP、Best-of-N、SBoN、Softmax采样、梯度上升校准以及基于高斯假设的逆协方差权重；

**📊 数据集**

实验使用ScienceQA (SQA) 视觉问答数据集和GSM8K数学推理数据集，结合多种预训练VLM/LLM模型；

**📈 对比分析**

通过与单一参考模型、贪婪采样、BoN等基线对比，SLOP在SQA上从63%提升至约60%+，在GSM8K上从约43%提升至84%以上；

**⚠️ 局限性**

实验规模有限、算力受限，仅在可验证奖励任务上测试，未覆盖安全、多模态及更广泛的对齐场景。

---

## 531. Scaling Retrieval-Augmented Reasoning with Parallel Search and Explicit Merging

**arXiv ID:** 2605.13534 | [PDF](https://arxiv.org/pdf/2605.13534v1)

**作者:** Jiabei Liu `[一作]` (University of Science and Technology of China), Xiang Wang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 26741 | [OpenAlex ID](https://openalex.org/A5100732436)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在检索驱动的推理过程中，研究者提出了一种新型深度检索代理——MultiSearch，该代理能够在每一步推理时并行生成多条查询、并行检索外部知识，并对检索结果进行显式合并与过滤，从而提升检索信息的信噪比并降低无关噪声。

**💡 创新点**

核心创新点包括：① 并行多视角查询生成（重述、概念扩展、问题拆解）以扩大信息覆盖；② 明确的合并步骤对检索结果进行去重、提取并整合关键信息；③ 针对三大子任务（答案、查询、多查询合并）设计多过程奖励，并采用Group Reward‑Decoupled Normalization Policy Optimization (GDPO) 对奖励进行分离归一化，保证各子任务的奖励能独立、均衡地指导学习。

**🔧 技术方法**

技术实现主要依赖：① 大模型 Qwen2.5-3B/7B（Base/Instruct）作为生成器；② E5 检索引擎和 2018 年 Wikipedia 语料作为外部知识库；③ RL 框架 GDPO（对比 GRPO）训练代理；④ 通过自定义工具调用实现检索、信息封装、合并、答案输出等步骤；⑤ 采用词级 F1、查询数奖励、合并奖励等多维度奖励。

**📊 数据集**

实验使用七个公开问答基准：单跳类 NQ、TriviaQA、PopQA；多跳类 HotpotQA、2WikiMultiHopQA、Musique、Bamboogle。

**📈 对比分析**

在所有基准上，MultiSearch 均显著优于现有深度检索代理（如 Search‑R1、AutoRefine、Search‑o1、AdaSearch 等）和传统 RAG 方案；尤其在多跳数据集上提升约 1.5‑2.5% 的 EM，整体平均准确率最高，证明多查询 + 合并 + 多过程奖励能有效提升检索质量与推理性能。

**⚠️ 局限性**

主要局限包括：① 仍受检索引擎的覆盖范围与相关性限制，超参数（查询数 n_q、Top‑k）需手动调优；② 合并步骤及奖励设计复杂，可能导致训练不稳定或过拟合；③ 对大模型规模的依赖较高，计算成本显著；④ 目前仅在 Wikipedia 等文本语料上验证，缺乏对多模态或实时更新知识库的适应性。

---

## 532. Granite Embedding Multilingual R2 Models

**arXiv ID:** 2605.13521 | [PDF](https://arxiv.org/pdf/2605.13521v1)

**作者:** Parul Awasthy `[一作]` (Ibm Research), Radu Florian `[通讯]` (Ibm Research)

**通讯引用:** 3817 | [OpenAlex ID](https://openalex.org/A5113515054)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

我们提出了两款多语言Granite Embedding R2模型（311M和97M），支持200多语言、编程代码，并拥有32,768-token长上下文窗口；

**💡 创新点**

核心创新在于将ModernBERT架构与Matryoshka表示学习、极大上下文扩展、词表裁剪等技术结合，构建既高性能又可调节维度的多语言检索模型；

**🔧 技术方法**

采用ModernBERT架构、旋转位置嵌入、交替全局注意力、Flash Attention、对比微调、知识蒸馏、词表裁剪、Matryoshka表示学习等技术；

**📊 数据集**

训练数据包括FineWeb2、GneissWeb、FineWeb-Edu、多语言维基百科、Stack Exchange、arXiv、Granite Code的代码数据以及基于GPT-OSS的合成多语言长/短篇段落和推理数据；

**📈 对比分析**

通过MTEB-v2 Retrieval、MTEB-Code、LongEmbed、RA-R、BEIR等基准进行评估，311M模型在MTEB Retrieval平均得分65.2，97M模型得分60.3，分别位于<500M参数和<100M参数模型中性能最前沿；模型在速度-准确性Pareto前沿上也表现优异；

**⚠️ 局限性**

局限性包括：与单语专用模型相比，在纯英文任务上性能略逊，词表的高fertility导致与部分模型相比在效率上略有劣势，依赖合成数据可能引入偏差，且大型模型资源需求高。

---

## 533. Reward-Weighted On-Policy Distillation with an Open Property-Equivalence Verifier for NL-to-SVA Generation

**arXiv ID:** 2605.13501 | [PDF](https://arxiv.org/pdf/2605.13501v1)

**作者:** Qingyun Zou `[一作]` (National University of Singapore), Weng-Fai Wong `[通讯]` (National University of Singapore)

**通讯引用:** 4309 | [OpenAlex ID](https://openalex.org/A5023989495)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `8d10c613-917e-4880-9716-17789f50e119` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种改进的LLM训练方法，用于将自然语言描述转换为 SystemVerilog Assertions（SVA），通过在学生模型的自身输出上进行奖励加权的对抗性蒸馏（RWOPD）来提升生成的语义正确性。

**💡 创新点**

创新点在于将开放式属性等价检查器（SymbiYosys+Z3 PEC）与教师模型的前向KL散度相结合：先用验证器筛选并给每条回放加权，再让冻结的14B教师对通过验证器的回放提供密集的token级梯度，从而在小规模LoRA预算下显著提升属性等价性。

**🔧 技术方法**

技术包括：奖励加权的on-policy蒸馏（RWOPD），开放式属性等价检查器，分层时间复杂度水平（TCL）课程化SFT，带有操作符权重的交叉熵损失，LoRA微调，前向KL梯度，温度采样，稀疏RL奖励对比实验。

**📊 数据集**

使用的数据集为：CodeV‑SVA 训练集（约 300K 条 NL/SVA 对），以及官方基准 NL2SVA‑Human 与 NL2SVA‑Machine（分别包含多种 C1/C2/C3 复杂度的 SVA）。

**📈 对比分析**

与多种基线比较：专用模型 CodeV‑SVA‑14B、通用大模型 DeepSeek‑R1‑671B 与 GPT‑5、以及 Qwen3‑8B/14B 等。RWOPD 在 NL2SVA‑Human 和 NL2SVA‑Machine 的 pass@1/5/10 上分别提升约 2–6 个百分点，特别是在时间复杂度 C2/C3 的类上显著优于现有 SOTA。

**⚠️ 局限性**

局限性：开放式等价检查器仅覆盖 C1/C2 片段，liveness (C3) 仍有未覆盖的情况；训练对检查点选择敏感，需要额外的验证集；方法依赖强大的教师模型和高质量的等价检查器；在更大模型或不同语言任务上的可迁移性仍需进一步验证。

---

## 534. Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging

**arXiv ID:** 2605.13583 | [PDF](https://arxiv.org/pdf/2605.13583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 535. Dynamical Predictive Modelling of Cardiovascular Disease Progression Post-Myocardial Infarction via ECG-Trained Artificial Intelligence Model

**arXiv ID:** 2605.13568 | [PDF](https://arxiv.org/pdf/2605.13568v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 536. Sustainable Graph Analytics Workload Scheduling with Evolutionary Reinforcement Learning in Edge-Cloud Systems

**arXiv ID:** 2605.13489 | [PDF](https://arxiv.org/pdf/2605.13489v1)

**作者:** P. Ramicetty `[一作]`, S. Pasricha `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 MERSEM，一种融合进化搜索与强化学习的多目标框架，用于在异构边缘-云环境中调度图分析工作负载，以同时最小化 SLA 违规率和碳排放。

**💡 创新点**

创新点在于：①在调度决策中显式建模 DAG 结构、并行度和通信延迟；②将全局探索的进化算法与局部自适应的 RL 引导搜索相结合，形成双重优化；③通过动态碳强度与资源异质性构建实时碳/能耗模型；④提供可调节的权重设置，实现 SLA、碳或两者平衡的多场景配置。

**🔧 技术方法**

技术包括：多目标进化优化（NSGA‑II/III 变体）、基于 Q‑learning 的自适应局部搜索、时间-空间碳强度和能耗模型、图形工作负载执行模型、Edge‑Fog‑Cloud 资源模型，以及 EdgeCloudSim 仿真平台。

**📊 数据集**

使用的工作负载数据集为大型图工作流数据集（约 500,000 个作业、1.3M 任务）以及三种常用 GNN 数据集（CORA、Amazon、CiteSeer）对应的 GraphSAGE、GCN 与 GAT 模型。

**📈 对比分析**

与 CAGWO、CACSA、CASSA、PSOGA 等现有碳感知调度方法在 10 种实验场景（不同任务量、设备数、雾中心数和 GNN 负载）下对比，MERSEM 在碳排放上实现 10–12% 的下降，在 SLA 违规率上降低 35–45%，并保持良好的可扩展性和对高计算密集度 GNN 任务的鲁棒性。

**⚠️ 局限性**

局限性包括：①对 RL 超参数的敏感度未完全探究；②在极端大规模部署（数万节点）下的计算开销和收敛速度仍需验证；③模型假设工作负载可预测且不考虑实时突发事件；④未对不同能源混合比例（如现场可再生能源比例）进行深入分析。

---

## 537. Uncertainty-Aware Prediction of Lung Tumor Growth from Sparse Longitudinal CT Data via Bayesian Physics-Informed Neural Networks

**arXiv ID:** 2605.13560 | [PDF](https://arxiv.org/pdf/2605.13560v1)

**作者:** Lingfei Kong `[一作]` (Vanderbilt University), Haoran Ma `[通讯]` (Harvard University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了在稀疏、间歇性肺部CT随访数据下，利用贝叶斯物理信息神经网络预测肺腺瘤生长并量化不确定性。

**💡 创新点**

创新点在于将Gompertz增长动力学与贝叶斯推断结合，并采用两阶段MAP+HMC的低维参数空间推断，实现既有生物学约束又有校准的不确定性预测。

**🔧 技术方法**

采用物理信息神经网络（PINN）、贝叶斯推断（MAP+Hamiltonian Monte Carlo）、对数体积变换与Gompertz动力学约束。

**📊 数据集**

使用美国国家肺部筛查试验（NLST）低剂量CT随访数据，共30名患者，三次扫描。

**📈 对比分析**

与纯Gompertz、纯PINN、纯高斯过程以及各类贝叶斯变体对比，结果显示贝叶斯PINN在log空间RMSE≈0.20、95%可信区间覆盖率0.95，且在患者层面误差更稳定。

**⚠️ 局限性**

局限性包括样本量小（30人）、仅三次随访、简化动力学模型、HMC采样计算成本高，且对长期预测与更复杂生物机制验证不足。

---

## 538. Mixed neural posterior estimation for simulators with discrete and continuous parameters

**arXiv ID:** 2605.13551 | [PDF](https://arxiv.org/pdf/2605.13551v1)

**作者:** Jan Boelts `[一作]` (appliedAI Institute for Europe), Daniel Gedon `[通讯]` (University of Tübingen)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5065863217)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出混合参数的神经后验估计（MNPE），能在包含离散与连续参数的模拟器中实现快速后验推断。

**💡 创新点**

创新点在于将后验分解为离散与连续两部分，使用自回归分类器（MADE）与连续生成模型（如流或扩散模型）联合训练，并开发联合校准诊断工具（SBC+ECE）。

**🔧 技术方法**

使用NPE框架、Masked Autoregressive Density Estimator、神经分形流或扩散模型进行联合训练，同时结合模拟校准和期望校准误差评估。

**📊 数据集**

在三种模拟器上验证：可解析的混合高斯模型、具有已知似然的串联排队模型以及不可解析的 Hodgkin–Huxley 神经元模型。

**📈 对比分析**

与解析解、MCMC 参考以及C2ST评分比较，MNPE 在 1,000–100,000 次模拟后可逼近真实后验，校准度良好，性能随训练数据增长稳步提升。

**⚠️ 局限性**

局限性包括：对复杂后验需大量模拟样本；模型规格化可能受限；校准仅针对单维边缘；ECE 基准在小样本下可能低估噪声。

---

## 539. RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation

**arXiv ID:** 2605.13542 | [PDF](https://arxiv.org/pdf/2605.13542v1)

**作者:** Chengzhi Shen `[一作]` (Technical University of Munich), Jiazhen Pan `[通讯]` (Technical University of Munich)

**通讯引用:** 1373 | [OpenAlex ID](https://openalex.org/A5047557442)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个真实ICU情境下的AI决策支持基准RealICU，使用后视医学专家标注的标签评估模型的临床推理能力；

**💡 创新点**

创新点在于采用后视标注避免对记录的医师行为进行行为模仿评价，提出四个关键临床任务并引入结构化记忆框架ICU-Evo来研究记忆对推理和安全性的影响；

**🔧 技术方法**

使用的大型语言模型（LLM）与多种记忆增强机制（RAG、ReAct、ICU‑Evo等）以及结构化记忆组件（工作、趋势、事件、轨迹、洞察）来实现任务预测；

**📊 数据集**

数据集为MIMIC‑IV ICU记录，构建了两部分：RealICU‑Gold（94例，930窗口）和RealICU‑Scale（11,862窗口，利用Oracle LLM后视标注）；

**📈 对比分析**

与传统基准相比，RealICU展示了当前前沿LLM在长期ICU推理任务中的显著不足，尤其表现为回忆‑安全权衡和锚定偏差；在结构化记忆下，ICU‑Evo在患者状态和急性问题等任务上获得显著提升，但仍有较高的危险推荐率；

**⚠️ 局限性**

局限包括仅基于MIMIC‑IV单中心数据，缺乏多模态（影像、信号）支持，实验仅单次运行且未充分评估长期ICU轨迹的方差。

---

## 540. Locale-Conditioned Few-Shot Prompting Mitigates Demonstration Regurgitation in On-Device PII Substitution with Small Language Models

**arXiv ID:** 2605.13538 | [PDF](https://arxiv.org/pdf/2605.13538v1)

**作者:** Anuj Sadani `[一作]` (Infrrd.ai), Deepak Kumar `[通讯]` (Infrrd.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现并评估了完整的设备端 PII 替换流水线，使用 1‑bit Mixture‑of‑Experts 识别器检测 PII，1‑bit Small Language Model（SLM）生成与上下文一致、类型保持的伪值，规则生成器处理模式化字段，并通过 locale‑conditioned rotating few‑shot demo 解决少量演示回声问题，最终公开代码与评测数据集。

**💡 创新点**

创新点包括：① 在设备端实现一致性、类型保持且长度保持的 PII 替换；② 发现并修复“少量演示回声”失败模式；③ 对多语言 PPL、长度保持和下游 NER 训练效果进行系统量化；④ 将整个 pipeline、模型与评测数据公开，促进复现。

**🔧 技术方法**

技术栈：1‑bit Q1_0 Bonsai 1.5B MoE 识别器、1‑bit SLM 1.7B（使用 llama‑cpp 推理）、faker 规则生成器、locale‑conditioned rotating demo prompting、564M 多语言 LLM 评估器、spaCy NER 训练。

**📊 数据集**

数据集：合成文档生成器产生 2000 文档，涵盖 7 个模板和 6 种语言（en_US, en_IN, de_DE, es_MX, ja_JP, zh_CN）。主要指标基于前 100 文档，NLP 下游评估使用 500 文档（大型）和 200 文档（匹配规模）。

**📈 对比分析**

比较方法：对比三种模式（redact、faker、hybrid）在泄漏率、PPL、长度保持、下游 NER F1 以及推理延迟上的表现。主结果：泄漏率相同≈0.249；PPL 上 hybrid 低于 faker（平均‑16%），长度保持最高；大型 NER 训练中 faker 恢复约 68% 原始 F1，匹配规模下 hybrid 低于 faker（p<0.001）。Hybrid 推理延迟约 41.2 s，faker/redact 约 1.6 s。

**⚠️ 局限性**

局限性：① PII 检测器的召回率限制了所有模式的泄漏率；② 少量演示回声问题仅在极低精度 SLM 上出现，需手动 prompt；③ SLM 的伪值多样性受演示池大小限制，导致下游 NER 表现低于 faker；④ 缺乏共指解析导致同一实体在文档内被多次替换；⑤ 仅评估字符串泄漏，未覆盖推理攻击或差分隐私；⑥ 对某些语言（如仅汉字日文名）处理不完善；⑦ 目前仅在 CPU 上实现，GPU 加速未覆盖。

---

## 541. Towards Unified Surgical Scene Understanding:Bridging Reasoning and Grounding via MLLMs

**arXiv ID:** 2605.13530 | [PDF](https://arxiv.org/pdf/2605.13530v1)

**作者:** Jincai Huang `[一作]` (Southern University of Science and Technology), Weixin Si `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个统一的手术场景理解框架 SurgMLLM，能够在同一模型中实现手术阶段识别、IVT（Instrument–Verb–Target）三元组推理和三元组实体的像素级定位。

**💡 创新点**

创新点包括：①将多模态大型语言模型（MLLM）与 SAM2 视觉分割模型对接，构建实体感知的 Prompt 机制；②通过时间融合的残差方式实现跨帧一致的 Prompt，提升视觉定位的连贯性；③创建了 CholecT45-Scene 数据集，首次将阶段标签、三元组关系与像素级掩码与推理文本统一标注，支持端到端多任务学习。

**🔧 技术方法**

采用 InternVL2.5‑4B 作为 MLLM，使用 LoRA 微调；将 LLM 输出的实体 Prompt 通过 MLP 映射至 SAM2 的 Prompt 空间；引入 BCE、Dice 与实体加权损失的联合优化；使用 SAM2‑H 进行像素级掩码预测。

**📊 数据集**

主要使用 CholecT45-Scene 数据集（64,299 帧，包含 7 个阶段、IVT 三元组及对应的 instrument/target 像素掩码），同时与原始 CholecT45 和 CholecTriplet‑Seg 进行对比实验。

**📈 对比分析**

在阶段识别、三元组识别和三元组实体分割三项任务上与最新方法进行对比，SurgMLLM 在阶段识别的 Accuracy、Precision、Recall、Jaccard 均获最高；在三元组识别上 AP_IVT 达到 46.0%（提升至 40.7% 的 SOTA）；在三元组实体分割上 mIoU 提升至 84.4%（显著优于 SAM2‑ZeroShot、SAM2‑Adapter、SurgSAM2 等基线）。

**⚠️ 局限性**

主要局限是仍需依赖密集的三元组实体掩码标注，未来工作需探索弱/半监督的视觉定位策略，以降低对人工标注的依赖。

---

## 542. Many-Shot CoT-ICL: Making In-Context Learning Truly Learn

**arXiv ID:** 2605.13511 | [PDF](https://arxiv.org/pdf/2605.13511v1)

**作者:** Tsz Ting Chung `[一作]` (Hong Kong University of Science and Technology), Dit-Yan Yeung `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 26060 | [OpenAlex ID](https://openalex.org/A5073139380)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探索大规模CoT-ICL在推理任务中的可扩展性，并提出通过演示排序提升性能

**💡 创新点**

发现多示例CoT-ICL的规模、相似性检索和顺序不再适用于非推理任务，提出“易于理解”与“信息流平滑”两条原则，并提出基于曲率最小化的Curvilinear Demonstration Selection (CDS) 方法

**🔧 技术方法**

使用TSP启发式的演示排序算法，利用演示嵌入空间的曲率来安排顺序，结合相似性与曲率的双重成本

**📊 数据集**

分类基准（SuperGLUE、NLU、TREC、BANKING77）与推理基准（GSM8K、MATH、DetectiveQA、几何、数论）

**📈 对比分析**

与随机顺序、相似性检索以及多种LLM（LLaMA、Qwen、Qwen3、QwQ、DeepSeek-R1、GPT‑5.2）对比，CDS在几何任务上最多提升5.42个百分点，在其他推理任务亦有显著增益；在分类任务上提升有限

**⚠️ 局限性**

受限于长上下文窗口、演示数量与模型规模的依赖，且在非推理任务中效果不明显，尚未完全解决模型对不同演示质量的鲁棒性

---

## 543. Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models

**arXiv ID:** 2605.13497 | [PDF](https://arxiv.org/pdf/2605.13497v1)

**作者:** Xinye Wanyan `[一作]` (RMIT University), Jeffrey Chan `[通讯]` (RMIT University)

**通讯引用:** 4553 | [OpenAlex ID](https://openalex.org/A5071422010)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于大型语言模型的自动化用户画像生成框架APG4RecSim，用于提升推荐系统中模拟用户行为的真实性与一致性。

**💡 创新点**

创新点在于：①无需手工设定属性模式，通过三阶段（初始化提取、上下文感知语义合并、因果映射细化）自动生成任务对齐且语义连贯的用户画像；②将属性合并与因果映射结合，显著降低对全局热门度等无关偏差的依赖；③提供了训练‑free、可迁移且可解释的生成流程。

**🔧 技术方法**

技术上主要使用大语言模型（如gpt‑4o‑mini、Llama‑3.3‑70B‑Instruct等）进行属性提取与语义合并，利用对抗性或逆向示例进行因果映射，整体流程无监督且不需要额外训练。

**📊 数据集**

实验使用MovieLens‑1M、Amazon‑Book、Amazon‑Beauty三大公开推荐数据集，分别从历史交互生成画像并评估。

**📈 对比分析**

通过与无画像、RecAgent、Agent4Rec等基线对比，APG4RecSim在三种任务（判别、排序、评分）上在24项评估指标中获16项第一、7项第二；在判别任务中平均提升0.6%至1.2%；排序任务nDCG@10提升至0.75；评分任务RMSE和JSD均优于基线。

**⚠️ 局限性**

局限性包括：①框架仅关注画像生成，未与完整的记忆、规划等代理组件联调；②因果映射依赖LLM推理，易受模型随机性影响；③对LLM内在参数记忆的控制仍不充分，需进一步通过匿名化或合成数据评估。

---

## 544. The Gallai Vertex Problem is $Θ_2^p$-Complete

**arXiv ID:** 2605.13488 | [PDF](https://arxiv.org/pdf/2605.13488v1)

**作者:** Amir Nikabadi `[一作]` (IT University of Copenhagen), Lasse Wulf `[通讯]` (IT University of Copenhagen)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5062230321)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文通过构造与 SAT Parity 的多项式时间归约，证明判定图是否存在 Gallai 顶点的决策问题属于 Θ₂^p 类，并给出其完备性与强逼近不可行性结果；

**💡 创新点**

主要创新在于首次将 Gallai 顶点问题定位于 Θ₂^p 难度级别，并通过新的偶奇 Gadget 设计实现了该类完备性，从而揭示了其相较于 NP 或 co‑NP 的更高复杂度；

**🔧 技术方法**

采用了 Wagner 的偶奇归约技术、Walther‑Zamfirescu 图的改造、最长路径与最长环的跨图构造以及多重 NP 查询并行访问的技术；

**📊 数据集**

论文不涉及实验或数据集，全部工作基于理论构造与归约；

**📈 对比分析**

通过与已知的 Θ₂^p 完备问题（如 SAT Parity）的归约，证明了该问题与其它 NP/Σ₂^p 问题的复杂度等价，表明无法在多项式时间内得到更好的近似；

**⚠️ 局限性**

局限性在于结果仅适用于理论复杂度分析，未给出具体算法实现；且对于是否存在上界常数 c 使得所有连通图的最长路径横截数 ≤ c 仍未解决，且对实际大规模图的求解仍无有效多项式算法。

---

## 545. R^2-Mem: Reflective Experience for Memory Search

**arXiv ID:** 2605.13486 | [PDF](https://arxiv.org/pdf/2605.13486v1)

**作者:** Xinyuan Wang `[一作]` (University of Science and Technology of China), Xiangnan He `[通讯]` (University of Science and Technology of China)

**通讯引用:** 44023 | [OpenAlex ID](https://openalex.org/A5038668215)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 R²-Mem 框架，利用 Rubric‑guided Evaluator 对深度搜索轨迹进行细粒度评估，并用 self‑Reflection Learner 提炼经验，提升记忆检索性能与效率。

**💡 创新点**

① 对搜索过程进行步级细粒度评估；② 将高低质量步骤的经验抽象为规划与反思指引；③ 在不依赖 RL 的情况下实现记忆搜索的自我改进。

**🔧 技术方法**

使用 Rubric 评估体系、Self‑Reflection 学习器、经验库检索（Top‑K 相似度）、LLM（Qwen2.5、Llama3.1、GPT‑4o）与 BGE‑M3 检索技术。

**📊 数据集**

在 LoCoMo、HotpotQA、NarrativeQA 三个长上下文记忆检索基准上进行实验。

**📈 对比分析**

与 Memory‑free RAG、结构化记忆（A‑Mem、MemoryOS、LightMem）、RL‑based Memory R1 以及深度搜索基线 GAM 进行对比，R²‑Mem 在 F1 及 BLEU 上均优于 GAM，F1 提升最高 22.6%，Token 与迭代次数分别减少 12.9% 与 20.2%。

**⚠️ 局限性**

对 Rubric 阈值与检索大小的超参数相对敏感；实验仍依赖外部 GPT‑4o 评估，完全自评仍受限；仅在公开数据集上验证，真实世界长记忆场景需进一步测试。

---

## 546. Beyond Anthropomorphism: Exploring the Roles of Perceived Non-humanity and Structural Similarity in Deep Self-Disclosure Toward Generative AI

**arXiv ID:** 2605.13574 | [PDF](https://arxiv.org/pdf/2605.13574v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 547. Discovery of Hidden Miscalibration Regimes

**arXiv ID:** 2605.13484 | [PDF](https://arxiv.org/pdf/2605.13484v1)

**作者:** Katarzyna Kobalczyk `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 22966 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于学习输入空间几何的校准诊断框架，能自动发现并定位模型的隐藏过度或不足自信区域。

**💡 创新点**

创新点在于引入“误差场”(miscalibration field)概念，将校准误差视为输入空间中的标量场，并通过学习表示让局部残差平均揭示结构；同时提供了无需预定义切片的发现方法和可行的局部置信度校正策略。

**🔧 技术方法**

核心技术包括：核平滑 (Nadaraya–Watson)、可学习表示映射 ϕ (MLP 或其他神经网络)、基于邻域质量的正则化、范围感知的加法校正映射，以及验证集上基于 Brier 分数的超参数代理。

**📊 数据集**

在合成数据（三聚类与正弦调制）上验证模型能力，并在四个真实 LLM 基准（HH‑RLHF、MedMCQA、MMLU、MMLU‑Pro）与十二个大型语言模型（Qwen‑3、Ministral‑3、Llama‑3.1/2、Gemma‑3）上进行实验。

**📈 对比分析**

与传统置信度基校正方法（温度缩放、等距回归）对比，所提出的方法在检测隐藏校准失配、提升局部 smECE、并在高异质性场景下显著降低误差；实验显示在大部分模型–数据对上，局部校正比全局方法更有效。

**⚠️ 局限性**

局限性包括：仅针对二分类正确性，未覆盖多分类或生成任务；需要预先训练的表示网络且对核宽度、正则化等超参敏感；对极度噪声或高频误差场的恢复能力有限。

---

## 548. HIR-ALIGN: Enhancing Hyperspectral Image Restoration via Diffusion-Based Data Generation

**arXiv ID:** 2605.13581 | [PDF](https://arxiv.org/pdf/2605.13581v1)

**作者:** Li Pang `[一作]` (Xi'an Jiaotong University), Xiangyong Cao `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2883 | [OpenAlex ID](https://openalex.org/A5028103486)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 HIR-ALIGN 框架，在目标域无干净数据的情况下，通过代理生成、改进的 unCLIP RGB 合成与稀疏光谱 warp 生成目标对齐的合成 HSI，随后用对齐监督微调现有恢复网络，实现 HSI 恢复的域自适应。

**💡 创新点**

创新点包括：① 三阶段代理‑生成‑微调的 plug‑and‑play 目标自适应流程；② 将 blur‑robust unCLIP 与 CLIP 视觉编码器微调、提示指导、SDEdit 采样结合以提升 RGB 生成质量；③ 通过稀疏权重与共享局部插值的 warp‑based 语义匹配保证光谱一致性；④ 通过理论风险上界证明混合数据覆盖与波动控制带来的性能提升。

**🔧 技术方法**

使用的技术有：源预训练恢复模型生成代理、改进 unCLIP 的 RGB 生成（CLIP 微调、提示、噪声初始化）、warp‑based 光谱传输（特征描述符检索、软聚合、共享局部插值）、对齐监督微调，以及 Wasserstein 距离与风险上界分析。

**📊 数据集**

实验使用的主要数据集为 CAVE、KAIST（模拟去噪与超分任务）和 HSIDwrD（真实噪声去噪），源训练集为 ICVL。

**📈 对比分析**

与源仅监督恢复器、无监督/优化方法、HSIGene 生成数据等进行对比；在 Gaussian/复杂去噪和超分辨率任务中，HIR‑ALIGN 使 PSNR/SSIM/SAM 明显提升，往往超过无监督扩散基线，逼近清洁目标微调上界。

**⚠️ 局限性**

局限性包括：依赖代理恢复的质量；RGB 生成仍受 unCLIP 偏差影响；warp 过程假设光谱一致性，极端光谱失真时可能失效；合成数据多样性受代理样本数量限制；理论分析基于简化假设。

---

## 549. Position: Assistive Agents Need Accessibility Alignment

**arXiv ID:** 2605.13579 | [PDF](https://arxiv.org/pdf/2605.13579v1)

**作者:** Jie Hu `[一作]` (Hunan University), Jiaming Zhang `[通讯]` (Hunan University)

**通讯引用:** 2532 | [OpenAlex ID](https://openalex.org/A5100453773)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对视障人士的辅助代理提出“可访问性对齐”框架，并基于对417篇文献中778个任务实例的系统分析构建了任务分类。

**💡 创新点**

创新点在于将可访问性视为独立的对齐目标，提出四维对齐维度（目标、交互、风险、生命周期）及其对应的设计、部署、迭代流程。

**🔧 技术方法**

主要技术包括基于任务依赖与约束的定性编码、对齐维度框架设计、以及案例研究中的交互协议与风险策略制定。

**📊 数据集**

使用的数据集为从文献综述中提取的778个辅助任务实例，未使用传统机器学习数据集。

**📈 对比分析**

论文未开展实验或性能评估，而是通过案例演示说明该框架能将评估从任务完成转向安全性、可验证性和误差恢复等指标。

**⚠️ 局限性**

局限性包括：基于已发表工作的任务实例可能低估真实使用场景；所提出的管线为设计框架，缺乏具体实现和量化验证；未针对不同硬件与感知模块给出细粒度实现细节。

---

## 550. AttenA+: Rectifying Action Inequality in Robotic Foundation Models

**arXiv ID:** 2605.13548 | [PDF](https://arxiv.org/pdf/2605.13548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 551. Self-Supervised On-Policy Reinforcement Learning via Contrastive Proximal Policy Optimisation

**arXiv ID:** 2605.13554 | [PDF](https://arxiv.org/pdf/2605.13554v1)

**作者:** Asim Osman `[一作]` (InstaDeep), Arnu Pretorius `[通讯]` (InstaDeep)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出并实现了一种名为 Contrastive Proximal Policy Optimization (CPPO) 的对比学习式近端策略优化算法，用于在无奖励信号的情况下进行目标条件强化学习。

**💡 创新点**

其创新点在于首次将对比学习的 Q‑value 与 PPO 的近端裁剪目标相结合，完成全在线、无经验回放、无目标网络的自监督学习，并能自然支持离散与连续动作空间以及多智能体设置。

**🔧 技术方法**

采用 InfoNCE 对状态‑动作与目标进行对比学习、利用 Hindsight Relabeling (HER) 生成正样本、通过 Monte Carlo 采样估计状态值、使用 PPO 的裁剪式目标函数，并对离散动作使用 Gumbel‑Softmax 近似。

**📊 数据集**

在 JAX 生态下的 Navix、JaxGCRL、SMAX、Connector、JaxNav 等 18 个单/多智能体、离散/连续混合的基准环境上进行评估。

**📈 对比分析**

与离线对比学习基线（CSAC、CDQN、ICRL）以及基于手工稠密奖励的 PPO/IPPO 进行对比；CPPO 在 14/18 任务中优于离线对比学习基线，在 12/18 任务中与或优于奖励驱动的 PPO，尤其在离散和多智能体场景表现突出。

**⚠️ 局限性**

在单智能体连续控制任务中表现逊于基于 SAC 的对比学习方法，主要原因是 Monte Carlo 状态值估计的方差较大；此外尚无中央化训练/分散执行的多智能体变体，未来需要进一步改进。

---

## 552. CA-GCL: Cross-Anatomy Global-Local Contrastive Learning for Robust 3D Medical Image Understanding

**arXiv ID:** 2605.13544 | [PDF](https://arxiv.org/pdf/2605.13544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 553. Integration of an Agent Model into an Open Simulation Architecture for Scenario-Based Testing of Automated Vehicles

**arXiv ID:** 2605.13539 | [PDF](https://arxiv.org/pdf/2605.13539v1)

**作者:** Christian Geller `[一作]` (RWTH Aachen University), Lutz Eckstein `[通讯]` (RWTH Aachen University)

**通讯引用:** 3921 | [OpenAlex ID](https://openalex.org/A5113050304)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

构建了一套基于OSI与FMI的标准化、模块化仿真集成架构，并实现了一个可跨OpenPASS、CARLA与CarMaker的闭环交通代理模型；

**💡 创新点**

首次将OSI（结构化消息）与FMI（模块化模型封装）结合，并通过osmp框架实现无工具依赖的代理模型互操作；

**🔧 技术方法**

采用OSMP、OSI、FMI、Protobuf、共享内存、PID控制、Dijkstra路径规划、IDM改进等技术；

**📊 数据集**

使用开放的交通场景数据（OpenSCENARIO、OpenDRIVE）及自行生成的多车道路网与交通灯场景；

**📈 对比分析**

通过在三款仿真器中执行相同的13个场景（跟随、限速、变道等），比较轨迹、车头距、加速度等指标，结果显示跨平台行为一致，性能线性扩展，单车实时因子≈30；

**⚠️ 局限性**

对模型的精确物理动态与真实世界匹配有限；仅支持提供OSI/FMI接口的仿真器；未评估在更大规模交通网络下的实时性与稳定性。

---

## 554. HLS-Seek: QoR-Aware Code Generation for High-Level Synthesis via Proxy Comparative Reward Reinforcement Learning

**arXiv ID:** 2605.13536 | [PDF](https://arxiv.org/pdf/2605.13536v1)

**作者:** Qingyun Zou `[一作]` (National University of Singapore), WengFai Wong `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HLS-Seek 框架，实现自然语言描述到高层次合成（HLS）代码的生成，并在生成过程中考虑硬件质量（QoR）目标。

**💡 创新点**

核心创新在于用比较代理奖励模型（Siamese 结构）和不确定性感知 MC‑dropout 切换机制，实现在 RL 训练中快速获取相对 QoR 信号，避免昂贵的完整 HLS 合成循环。

**🔧 技术方法**

技术方案包括三阶段训练（多样性 SFT → 冷启动 SFT + 生成推理链 → GRPO RL）、Pareto‑proximal 质量多样性采样、链式推理与推理追踪、MC‑dropout 置信度估计、以及基于预训练编码器的比较奖励模型。

**📊 数据集**

数据集涵盖 ForgeHLS、SAGE‑HLS、HLStrans 三大公开 HLS 数据集（共 12,661 篇应用），训练时使用 DSE 生成的 Pareto‑近似样本；评估基准为 HLS‑eval 43 核与 ForgeHLS test 108 应用，保证无数据重叠。

**📈 对比分析**

与 GPT‑5.1、DeepSeek‑V3.2、Gemini‑3‑Pro、Qwen3‑235B 等前沿模型对比，HLS‑Seek 在语法正确率 pass@1 达 81.5%、功能正确率 pass@5 达 81.4%，在 QoR 评测中 16/30 核实现最低延迟，Pareto 支配 9/30 核，且 RL 训练速度比完整合成奖励快 8.5 倍。

**⚠️ 局限性**

局限性包括奖励模型在极端 OOD 设计仍需触发真实合成，且仍受限于训练数据覆盖范围，对极大规模设计空间的探索和资源平衡优化尚未完全解决。

---

## 555. AI-Generated Slides: Are They Good? Can Students Tell?

**arXiv ID:** 2605.13532 | [PDF](https://arxiv.org/pdf/2605.13532v1)

**作者:** Juho Leinonen `[一作]` (Aalto University), Arto Hellas `[通讯]` (Aalto University)

**通讯引用:** 3514 | [OpenAlex ID](https://openalex.org/A5076828114)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一门Web软件开发课程中，研究者用多种生成式AI工具将教科书章节转换为幻灯片，并在课堂上评估教师和学生对其质量与来源的感知。

**💡 创新点**

首次将编码助手（Cursor、Claude Code）用于非程序化的幻灯片制作，并通过学生实际评价检验AI生成幻灯片与人工幻灯片的可区分性。

**🔧 技术方法**

采用NotebookLM、M365 Copilot、Claude、Cursor、Claude Code等大语言模型及编程辅助工具，配合Quarto、PowerPoint进行幻灯片生成与排版。

**📊 数据集**

使用Aalto大学Web软件开发课程的教材章节作为生成素材，并收集课程学生对幻灯片质量与来源的问卷反馈。

**📈 对比分析**

教师对生成幻灯片进行叙事评估（事实准确性、完整性、教学适切性），学生通过7点量表评估质量并猜测来源；统计检验显示AI幻灯片与人类幻灯片质量无显著差异，学生无法可靠识别来源，且质量与AI猜测呈负相关。

**⚠️ 局限性**

局限于单一学校、单门课程、样本量有限，未测量学习效果，评估依赖教师主观叙事，且AI工具在排版与风格统一上仍存在缺陷。

---

## 556. Phantom Force: Injecting Adversarial Tactile Perceptions into Embodied Intelligence via EMI

**arXiv ID:** 2605.13492 | [PDF](https://arxiv.org/pdf/2605.13492v1)

**作者:** Zirui Kong `[一作]` (Hong Kong Polytechnic University), Sze Yiu Chau `[通讯]` (Simon Fraser University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并验证了通过电磁干扰（EMI）可以操纵手指尖霍尔效应触觉传感器的力感知，从而造成虚假力感知。

**💡 创新点**

首次揭示触觉传感器在物理层面的安全缺陷，并量化了其对机器人学习模型的破坏效果。

**🔧 技术方法**

使用高频信号发生器、功率放大器、近场探针以及NVIDIA Isaac Sim仿真平台。

**📊 数据集**

利用自制的加权重量实验数据作为基准数据集。

**📈 对比分析**

与基线无攻击情况对比，攻击下方向相似度降至0.56，幅值比例升至9.20，随机森林分类器精确率召回率降至0。

**⚠️ 局限性**

实验范围受限于单一霍尔效应传感器和固定频率，未考虑多模态或更高功率/远距离攻击。

---

## 557. Path-independent Flow Matching for Multi-parameter Generative Dynamics

**arXiv ID:** 2605.13487 | [PDF](https://arxiv.org/pdf/2605.13487v1)

**作者:** Francisco Téllez `[一作]` (Université de Montréal), Yanlei Zhang `[通讯]` (Queen's University)

**通讯引用:** 10 | [OpenAlex ID](https://openalex.org/A5100659756)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了多参数路径无关流匹配（PiFM）方法，用以学习在多维参数空间下的路径无关概率分布变换。

**💡 创新点**

创新点在于将路径无关性（即变换可交换）引入流匹配框架，并证明其与 Wasserstein barycenter 的关系。

**🔧 技术方法**

使用神经网络拟合条件向量场并通过路径独立正则化训练，结合 ODE 连续动力学和条件流匹配。

**📊 数据集**

实验数据包括低维合成域移位、CelebA 属性翻译、以及单细胞 RNA‑seq 细胞重编程数据。

**📈 对比分析**

与 Meta Flow Matching、Curly Flow Matching、传统 CFM 进行对比，在 Wasserstein 距离、FID 等指标上表现更好，尤其在路径无关性和 OOD 推断方面显著提升。

**⚠️ 局限性**

局限在于仅保证分布级别的路径无关性，无法保证样本级别；并且对 Wasserstein barycenter 的理论仅在三分布的两参数情况已证明，尚未推广到更一般的多分布情形。

---

## 558. Effective Context in Transformers: An Analysis of Fragmentation and Tokenization

**arXiv ID:** 2605.13485 | [PDF](https://arxiv.org/pdf/2605.13485v1)

**作者:** Amirmehdi Jafari Fesharaki `[一作]`, Aslan Tchamkerten `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer在不同无损表示（字节、字符、子词）下有限上下文预测的性能差异。

**💡 创新点**

提出分割（fragmentation）会导致不可逆的预测损失，并证明在Markov源上tokenization可有效扩展有效上下文，给出信息理论框架。

**🔧 技术方法**

基于Markov源、有限上下文预测理论、信息熵、相位歧义、最优分布等理论工具，构造定理与证明。

**📊 数据集**

使用合成的有限阶Markov源以及WikiText‑103数据集做实证诊断。

**📈 对比分析**

通过训练Transformer在不同表示上比较log‑loss，结果显示字节/字符模型在相同窗口下损失更高，而子词tokenizer在相同窗口下可逼近熵率，证明理论给出的有效上下文指标对性能有解释。

**⚠️ 局限性**

局限性在于未证明Transformer训练可实现理论最优预测器，未给出tokenizer的完整设计方法，且实验仅针对理论诊断而非大规模任务。

---

## 559. Subsumption in $\mathcal{FL}_{\bot \mathit{reg}}$ with TBoxes Is in ExpTime

**arXiv ID:** 2605.13553 | [PDF](https://arxiv.org/pdf/2605.13553v1)

**作者:** Michał Henne `[一作]` (University of Opole), Paweł Parys `[通讯]` (University of Warsaw)

**通讯引用:** 437 | [OpenAlex ID](https://openalex.org/A5085631393)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文研究了描述逻辑FL、FL⊥、FLreg和FL⊥reg在有TBox约束下的概念包含问题，并给出了其精确的计算复杂度，证明在TBox约束下属于ExpTime-完备；在无TBox约束时FLreg和FL⊥reg属于PSpace-完备；

**💡 创新点**

创新点在于提出一种新的算法框架：通过将子包含判定转化为奇偶推送堆游戏（parity pushdown game），利用已知的游戏求解技术实现子包含的判定，并首次完成这些逻辑的完整复杂度分类；

**🔧 技术方法**

主要技术包括：概念规范化、将子包含转化为正则表达式语言包含、构造TBox模拟值限制的辅助概念、将整个问题映射到奇偶推送堆游戏，再利用奇偶推送堆游戏在ExpTime内可解的结果；

**📊 数据集**

本文未使用任何实际数据集，而是基于理论推导和归约来证明复杂度；

**📈 对比分析**

方法对比仅在理论上进行了说明：与已有的FL_0wer算法相比，本文提供了更一般的框架，可直接调用现有的奇偶推送堆游戏求解器（如PDSolver），但尚未给出实验性能对比；

**⚠️ 局限性**

局限性包括：仅讨论了子包含问题；未给出具体实现与实验结果；对更广泛的推理任务（匹配、统一）仍需进一步研究；

---

## 560. Qwen-Image-VAE-2.0 Technical Report

**arXiv ID:** 2605.13565 | [PDF](https://arxiv.org/pdf/2605.13565v1)

**作者:** Zekai Zhang `[一作]`, Lin Qu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Qwen-Image-VAE-2.0，一套高压缩率（f16/f32）的 VAE 系列，兼顾重建质量、文本清晰度和下游扩散可扩散性。

**💡 创新点**

创新点包括：全局跳连（GSC）+通道扩展复合压缩方案；改进的无注意力、非对称编码器-解码器骨干；利用 DINOv2 中间层的语义对齐加速 DiT 收敛；大规模数据工程（十亿图像 + 文本合成渲染）与新的 OmniDoc‑TokenBench 文本评测。

**🔧 技术方法**

采用的技术包括：变分自编码器框架、全局跳连结构、Attention‑free 卷积骨干、扩展通道维度、语义对齐损失（余弦相似 + 距离矩阵损失）、多阶段分辨率与文本渲染训练策略、无 KL 与 GAN 损失的简化目标。

**📊 数据集**

数据集：十亿规模通用图像集合；专门文档语料（学术论文、幻灯片、报纸等）；合成文本渲染数据；官方 OmniDoc‑TokenBench（约3K 文本密集图像）用于文本重建评估。

**📈 对比分析**

与 f8、f16 及 f32 现有 VAE 进行对比：在 ImageNet、FFHQ 上 PSNR/SSIM 领先；在 OmniDoc‑TokenBench 上 NED 超越所有 f8 基线，f16c128 的 NED 达 0.962；在 SiT 下的 DiT 训练中 gFID/IS 也优于同类高压缩模型，验证了更好的可扩散性。

**⚠️ 局限性**

局限性：模型参数量仍较大（约 200–250M），训练成本高，需要十亿图像和大规模算力；在极小字体或极端排版场景的文本细节还可能不足；目前仅在 ImageNet/FFHQ、OmniDoc‑TokenBench 进行评估，未在更广泛的多模态任务中验证。

---

## 561. ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin

**arXiv ID:** 2605.13517 | [PDF](https://arxiv.org/pdf/2605.13517v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 562. Decoupled and Divergence-Conditioned Prompt for Multi-domain Dynamic Graph Foundation Models

**arXiv ID:** 2605.13540 | [PDF](https://arxiv.org/pdf/2605.13540v1)

**作者:** Haonan Yuan `[一作]` (Beihang University), Philip S. Yu `[通讯]` (University of Illinois at Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种面向多域连续动态图的图基础模型——DyGFM，采用语义-时间解耦预训练、跨域差异感知路由和差异条件化提示，实现多域预训练与少样本微调的高效迁移。

**💡 创新点**

创新点包括：①将语义与时间动态分离为两支分支，解决不同域时间尺度不一致导致的语义噪声；②设计跨域路由机制，根据语义和时间差异动态加权源域专家，缓解负迁移；③引入差异条件化提示生成器，使微调仅需学习轻量级提示，保持预训练知识。

**🔧 技术方法**

使用的核心技术有：双分支GNN预训练（语义分支采用信息瓶颈约束，时间分支采用共享TGAT+域特定计时器与适配器）；跨域路由基于KL/欧氏距离的双向差异度量；差异条件化提示生成器（MLP）与提示注入融合；基于对比学习的自监督损失与InfoNCE链接预训练；以及轻量级参数化的Prompt与路由正则化。

**📊 数据集**

实验数据集覆盖四个领域的连续时间动态图：Wiki（用户-页面编辑网络）、Reddit（用户-帖子交互网络）、Coursera（学生-课程活动网络）和LastFM（用户-音乐流派收听网络）。

**📈 对比分析**

与12种基准方法（传统DGNN、动态图预训练、静态图基础模型和动态图提示方法）在节点分类和链路预测任务上进行对比。DyGFM在ASDA和LODO场景均取得最高AUC，平均提升5%~8%，并在微调阶段收敛更快、GPU占用更低，显著优于对手。

**⚠️ 局限性**

局限性包括：①对域间相似度的依赖，若目标域与所有源域差异过大，路由与提示仍可能不足以弥补；②需要手动设置源域原型与差异权重，超参数调优仍占用一定成本；③模型复杂度较高，尤其在多域大规模时训练仍受制于计算资源。

---

## 563. Efficient Implementation of an Adaptive Transformer Accelerator for Massive MIMO Outdoor Localization

**arXiv ID:** 2605.13507 | [PDF](https://arxiv.org/pdf/2605.13507v1)

**作者:** Ilayda Yaman `[一作]` (Lund University), Liang Liu `[通讯]` (Lund University)

**通讯引用:** 206950 | [OpenAlex ID](https://openalex.org/A5052819678)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

实现了一种面向5G大规模MIMO室外定位的自适应Transformer加速器，实现了实时子10 ms定位。

**💡 创新点**

利用传播特性产生的结构化稀疏性实现行级跳过、混合数据流（输入/输出静止）、自适应模型切换和基于sigmoid的注意力，以显著降低硬件成本与延迟。

**🔧 技术方法**

混合数据流向量处理引擎、Xilinx Zynq UltraScale+ FPGA、对称Q8.8定点量化、单层感知器(SLP)路由、行级稀疏检测、sigmoid+bias注意力。

**📊 数据集**

真实世界3GPP Release 17测量数据，来自商业5G基站的64波束、46延迟分组，覆盖三条不同传播场景S1、S2、S3。

**📈 对比分析**

与浮点基准、CPU、ASIC CNN基线对比；定位精度≤1.15 m，平均误差≤1.15 m；推理时延0.51‑2.11 ms；吞吐率最高1961 pos/s；相对CPU提升约6×、相对ASIC提升约3.5×；功耗1.29 W。

**⚠️ 局限性**

受限于当前稀疏阈值与硬件资源，难以支持更高天线/子载波数；定点量化在极端多路径场景下精度略降；设计主要验证于室外实验，室内/高速移动情况尚未充分评估。

---

## 564. Real2Sim: A Physics-driven and Editable Gaussian Splatting Framework for Autonomous Driving Scenes

**arXiv ID:** 2605.13591 | [PDF](https://arxiv.org/pdf/2605.13591v1)

**作者:** Kaicong Huang `[一作]` (Rensselaer Polytechnic Institute), Ruimin Ke `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 5130 | [OpenAlex ID](https://openalex.org/A5049143775)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Real2Sim 框架，将 4D Gaussian Splatting 与可微 Material Point Method 结合，实现可编辑、物理感知的驾驶场景生成。

**💡 创新点**

创新点在于把 Gaussian 体素直接视为粒子，支持实例级编辑并通过 MPM 进行真实物理交互，填补了现有生成模型缺乏物理一致性的空白。

**🔧 技术方法**

采用 4D Gaussian Splatting、可微 MPM 求解器、差分渲染、Fourier 球谐展开等技术。

**📊 数据集**

使用 Waymo Open Dataset 进行建模、编辑与评估。

**📈 对比分析**

通过渲染、重建、编辑和碰撞模拟等实验验证，Real2Sim 在视觉质量、物理逼真度和编辑灵活性方面优于传统生成与仿真方法。

**⚠️ 局限性**

局限在于视角覆盖不足导致模型在未见角度失真、渲染与物理耦合分离以及车辆物理参数粗糙，需要进一步改进。

---

## 565. HetScene: Heterogeneity-Aware Diffusion for Dense Indoor Scene Generation

**arXiv ID:** 2605.13586 | [PDF](https://arxiv.org/pdf/2605.13586v1)

**作者:** Zini Chen `[一作]`, Weiwei Xu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种两阶段的异质感知扩散框架HetScene，用于生成高密度、物理可行的室内场景布局；

**💡 创新点**

将场景拆分为主物体与次物体两类，分别用结构布局生成（SLG）与上下文布局生成（CLG）两阶段生成；引入可学习空间-语义调制机制和可学习场景图（LSG）条件；

**🔧 技术方法**

基于扩散概率模型（DDPM）与Transformer型去噪网络，配合AdaLN、空间编码、文本编码、房间掩码和LSG条件；

**📊 数据集**

在大规模多源M3DLayout基准上训练和评测，包含3D-FRONT、Matterport3D、Inf3DLayout等子集；

**📈 对比分析**

与ATISS、DiffuScene、MiDiffusion等SOTA方法对比，在FID/KID和CLIP等指标上均取得更低误差和更高语义一致性，尤其在XZ平面上提升显著；

**⚠️ 局限性**

在实际使用中仍受限于对次物体的显式支持关系建模，难以处理极稠密场景或极复杂的动态交互，且对训练数据规模与质量高度敏感。

---

## 566. Learning Local Constraints for Reinforcement-Learned Content Generators

**arXiv ID:** 2605.13570 | [PDF](https://arxiv.org/pdf/2605.13570v1)

**作者:** Debosmita Bhaumik `[一作]` (Institute of Digital Games), Ahmed Khalifa `[通讯]` (Institute of Digital Games)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

结合 Wave Function Collapse 与强化学习的 PCG 框架，在 Lode Runner 关卡中生成视觉美感且可玩性较高的游戏地图。

**💡 创新点**

创新点在于将 WFC 学习到的局部约束限制 RL 的动作空间，并系统研究输入多样性、罕见模式剔除与随机初始状态对生成效果的影响。

**🔧 技术方法**

使用 Maskable PPO 强化学习、WFC、自动化游戏模拟器评估可玩性以及 TP‑KLDiv 多样性指标。

**📊 数据集**

采用 VGLC 中的 Lode Runner 关卡数据，包含单一关卡、相似多关卡以及高多样性多关卡三组。

**📈 对比分析**

通过生成 100 个关卡并测量可玩率与多样性进行比较；结果显示单/相似多关卡可玩率最高，罕见模式剔除虽降低多样性但提升可玩率，随机初始状态对性能影响有限。

**⚠️ 局限性**

局限性包括：高多样性输入易导致 WFC 矛盾、罕见模式剔除削弱多样性；框架依赖手工奖励设计与特定输入样本，迁移至其他游戏场景存在困难。

---

## 567. MMSkills: Towards Multimodal Skills for General Visual Agents

**arXiv ID:** 2605.13527 | [PDF](https://arxiv.org/pdf/2605.13527v1)

**作者:** Kangning Zhang `[一作]` (Shanghai Jiao TongUniversity), Yong Yu `[通讯]` (Shanghai Jiao TongUniversity)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MMSkills框架，融合文本流程、运行时状态卡和多视图关键帧，构建可复用的多模态技能包，并通过分支加载机制在推理时实现视觉决策支持；

**💡 创新点**

创新点在于将技能包装为状态条件的多模态程序化知识，配合基于LLM的轨迹转技能生成器和分支加载推理模式，实现视觉依赖的技能检索与使用；

**🔧 技术方法**

使用的技术包括：大模型（LLM）驱动的技能规划与合并、轨迹嵌入与聚类、视觉关键帧提取与多视图对齐、分支加载（branch loading）与结构化指导生成；

**📊 数据集**

使用的数据集包括：OSWorld、macOSWorld、VisualAgentBench中的VAB‑Minecraft、LMGame‑Bench中的Super Mario Bros，以及用于生成技能的公开非测试交互轨迹；

**📈 对比分析**

通过与无技能、仅文本技能三种基线对比，实验表明MMSkills在所有模型（Gemini、Qwen、GLM、Kimi等）和任务上均提升了成功率、缩短了执行步数、降低了重复操作，最显著提升见OSWorld成功率从约44%提升至约50%或更高；

**⚠️ 局限性**

局限性包括：依赖于足够覆盖的公开轨迹、生成过程可能引入错误或视觉对齐不准、分支加载增加推理开销、缺乏在线修复与安全验证机制。

---

## 568. Uncertainty-Aware 3D Position Refinement for Multi-UAV Systems

**arXiv ID:** 2605.13500 | [PDF](https://arxiv.org/pdf/2605.13500v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 569. Verifying Exact Samplers for Continuous Distributions with a Discrete Program Logic

**arXiv ID:** 2605.13526 | [PDF](https://arxiv.org/pdf/2605.13526v1)

**作者:** Markus de Medeiros `[一作]` (New York University), Joseph Tassarotti `[通讯]` (New York University)

**通讯引用:** 634 | [OpenAlex ID](https://openalex.org/A5073987903)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

该论文提出了一套面向可验证的连续分布精确采样算法的高级分离逻辑，并在Coq中实现并验证了均匀、正态和拉普拉斯分布的采样器；

**💡 创新点**

创新点在于将Eris的误差信用规则推广到连续采样，通过预取磁带、时间收据和薄空气信用等机制，实现在存在无限递归、可变状态和高阶函数的程序中对连续采样的正式验证；

**🔧 技术方法**

主要技术包括：可变状态和高阶函数的分离逻辑（基于Iris），预取磁带（Clutch）用于预先确定随机样本，时间收据用于处理无限递归的部分正确性，薄空气信用用于产生任意小的误差信用；理论上还用到Riemann积分、Fubini定理和预期计算；

**📊 数据集**

该工作没有使用传统意义上的数据集；所有证明均在形式化证明助手Coq中完成，验证结果为形式化定理；

**📈 对比分析**

由于研究的重点是形式化验证，而非运行时性能评估，文中未给出实验对比或性能指标；验证通过Coq证明，证明过程已完全机械化；

**⚠️ 局限性**

主要局限包括：只适用于Riemann可积、分段连续的信用分布函数；无法处理无穷多不连续点或更一般的Lebesgue积分；并未完全证明采样器的几乎必然终止性，需借助其他逻辑或手工证明。

---

## 570. Beyond VMAF: Towards Application-Specific Metrics for Teleoperation Video

**arXiv ID:** 2605.13525 | [PDF](https://arxiv.org/pdf/2605.13525v1)

**作者:** Ines Trautmannsheimer `[一作]` (Technical University of Munich), Frank Diermeyer `[通讯]` (Technical University of Munich)

**通讯引用:** 1302 | [OpenAlex ID](https://openalex.org/A5052832584)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在遥控驾驶场景中，对压缩后视频质量进行主观评估并利用这些评估结果重新训练 VMAF 模型，使其更符合遥控驾驶员的感知；

**💡 创新点**

提出了针对遥控驾驶的多维度主观评估问卷（细节损失、可驾驶性、情境感知等），并通过该问卷对 VMAF 进行领域特定微调；

**🔧 技术方法**

使用 VMAF 框架中的支持向量回归（SVR）进行模型再训练，结合 FFmpeg 进行视频转码，采用 Spearman、Pearson、RMSE、MAD 等指标进行评估；

**📊 数据集**

使用从 Zenseact 数据集中挑选的 39 条 1920×1200 像素、8 秒长、不同天气/时间的驾驶视频，并生成四个 CRF 压缩等级；

**📈 对比分析**

与标准 VMAF‑4K 进行对比，微调后模型的 RMSE 从 10.36 降至 8.83（下降约 15%），MAD 从 8.71 降至 6.38（下降约 27%），相关系数保持在 0.86–0.88 之间，显示出更好的绝对误差一致性；

**⚠️ 局限性**

模型仍未考虑任务关键区域的空间或语义权重，导致在关键对象被压缩严重时误差显著；数据集规模有限，缺乏行为/生理指标验证，未来需引入区域权重或更大多样化的训练集。

---

## 571. Limits of Personalizing Differential Privacy Budgets

**arXiv ID:** 2605.13503 | [PDF](https://arxiv.org/pdf/2605.13503v1)

**作者:** Edwige Cyffers `[一作]` (CNRS), Juba Ziani `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5008250785)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究在差分隐私下使用个性化隐私预算对均值估计的效用提升，提出并分析了一种简单阈值化估计器，并与最优线性（affine）估计器进行比较。

**💡 创新点**

创新点在于系统性证明个性化预算带来的收益有限，给出了不同隐私预算场景下的近似因子（公共+私有数据下2倍、两种隐私水平下4倍、一般情况下O(log²n)或m²）。

**🔧 技术方法**

主要技术包括：Laplace机制的敏感度预处理、对均值估计的均方误差（MSE）闭式分析、阈值化与最优affine估计器的理论比较与构造反例。

**📊 数据集**

文章未使用真实数据集，全部为理论证明与模拟热图。

**📈 对比分析**

通过与最佳affine估计器的MSE比值进行比较，阈值化估计器在多数实用场景下已能获得常数因子近似（如2或4倍），在极端多隐私级别时误差会升高至O(log²n)。

**⚠️ 局限性**

局限性：仅针对均值估计，且假设使用Laplace机制；在多于两种隐私级别时最优估计器未明确定义，无法验证更高阶问题的收益。

---

## 572. MARLIN: Multi-Agent Game-Theoretic Reinforcement Learning for Sustainable LLM Inference in Cloud Datacenters

**arXiv ID:** 2605.13496 | [PDF](https://arxiv.org/pdf/2605.13496v1)

**作者:** H. Moore `[一作]` (Colorado State University), S. Pasricha `[通讯]` (Colorado State University)

**通讯引用:** 5946 | [OpenAlex ID](https://openalex.org/A5018382547)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 MARLIN 框架，在云数据中心实现多目标 LLM 推理调度，优化首个令牌时间（TTFT）、碳排放、水耗和能耗。

**💡 创新点**

创新点：将多代理强化学习与博弈论结合，采用可调权重共识调度、Veto 机制、FiLM 与 HER 等提升收敛与可解释性。

**🔧 技术方法**

技术：多代理 Soft Actor-Critic（SAC）+ FiLM + HER、预测模型、博弈论权重投票、Veto 机制，以及基于模拟器的多指标评估。

**📊 数据集**

数据集：Azure ChatGPT 两周真实推理请求日志（GPT‑3/4）以及 Llama‑7B/70B 模型执行特性和 NVIDIA A100/H100 GPU 能耗数据。

**📈 对比分析**

比较方法：与八种最先进的启发式与 RL 调度框架对比；在 TTFT、碳排放、水耗和能耗方面分别提升约 18%、33%、43%、11%，并在 Pareto 超体积（PHV）上最高。

**⚠️ 局限性**

局限性：仅在单一云运营商场景评估，未考虑多租户公平性与全生命周期碳；大规模数据中心实时实现仍需进一步验证。

---

## 573. PhysEditBench: A Protocol-Conditioned Benchmark for Dense Physical-Map Prediction with Image Editors

**arXiv ID:** 2605.13493 | [PDF](https://arxiv.org/pdf/2605.13493v1)

**作者:** Jiaxin Yang `[一作]` (Southern University of Science and Technology), Xiaojuan Qi `[通讯]` (University of Hong Kong)

**通讯引用:** 34969 | [OpenAlex ID](https://openalex.org/A5102498323)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出PhysEditBench基准，用于评估通用图像编辑器在单张RGB室内图像下预测深度、法向、反照率、粗糙度和金属度等密集物理图的能力。

**💡 创新点**

创新点在于设计目标特定的协议化评估流程、固定访问设置和多源数据子集，兼顾稠密物理图的不同属性，并提供光照应激子集和诊断变体。

**🔧 技术方法**

技术上结合了提示模板、图像条件文本、样本级别的显式结构化指令、以及对编辑器输出的后处理与对齐，利用指标计算如Affine-AbsRel、Acc@22.5、MAE、RMSE 等。

**📊 数据集**

使用了OpenRooms-FF、InteriorVerse和一个程序生成的金属度源，分别覆盖深度、法向、反照率、粗糙度和金属度的标注。

**📈 对比分析**

通过在固定协议下对比专用稠密预测模型与多款通用编辑器，结果显示在深度、法向和反照率上专用模型仍显著优越；在粗糙度和金属度上通用编辑器在某些标量指标可匹敌甚至超越基线，但在结构一致性与光照鲁棒性上仍有不足。

**⚠️ 局限性**

局限性包括评估仅覆盖合成室内数据和有限的金属度源，固定访问设置未涵盖所有可能提示与多轮交互，且对编辑器最大能力未作全面探索。

---

## 574. SieveFL: Hierarchical Runtime-Aware Pruning for Scalable LLM-Based Fault Localization

**arXiv ID:** 2605.13491 | [PDF](https://arxiv.org/pdf/2605.13491v1)

**作者:** Mahdi Farzandway `[一作]` (University of Tehran), Fatemeh Ghassemi `[通讯]` (University of Tehran)

**通讯引用:** 268 | [OpenAlex ID](https://openalex.org/A5102708616)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了SieveFL，一个基于五阶段层级过滤的自动缺陷定位框架，能够在大型代码库中精确定位导致测试失败的方法。

**💡 创新点**

创新点在于通过预先使用语义检索与运行时JaCoCo覆盖剪枝，显著减少LLM调用的候选方法数量，从而解决规模-精度权衡问题，并使得LLM只在高信号的少量候选上进行逐方法推理与最终重排序。

**🔧 技术方法**

使用的技术包括：大语言模型（Nemotron‑3‑Nano‑30B‑A3B via Ollama）、稠密向量检索（Faiss 与文本/代码嵌入模型）、JaCoCo代码覆盖分析、逐方法LLM筛选与聚合重排序。

**📊 数据集**

实验基于Defects4J v1.2.0 共395 个 Bug（383 个完成），涵盖六个 Java 项目，约 344k LOC。

**📈 对比分析**

与 Ochiai、GRACE、DeepFL、FLUCCS、AgentFL 等基线对比，SieveFL 的 Top‑1 准确率为 41.8%（165/395），MRR 0.469，Top‑1 领先 AgentFL 2.1pp；运行时剪枝后，候选方法减少 79%，LLM 输入 token 消耗下降 49%，Top‑3/5/10 与 MRR 同时提升。

**⚠️ 局限性**

局限性包括：仅在 Java 生态验证；JaCoCo 行级覆盖无法区分方法重载导致误剪；构建/测试失败导致部分 Bug 无法完成；未在非 Java 语言或大型 monorepo 上验证；缺乏对修复生成与开发者可用性的进一步评估。

---

## 575. Monads and Distributive Laws in Substructural Contexts (Extended Version)

**arXiv ID:** 2605.13533 | [PDF](https://arxiv.org/pdf/2605.13533v1)

**作者:** Soichiro Fujii `[一作]` (National Institute of Informatics), Ichiro Hasuo `[通讯]` (National Institute of Informatics)

**通讯引用:** 1839 | [OpenAlex ID](https://openalex.org/A5013382452)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出了以“Verbal Category”为参数的子结构化语境下的单子与分配律的通用理论，构造了在单子S为W‑operadic、T为W‑commutative时的规范分配律，并给出了对S进行W‑operadic精炼以满足此条件的通用构造；

**💡 创新点**

创新点在于统一了多种已知分配律与否定定理的子结构化背景，用Verbal Category形式化并扩展了先前基于结构规则的经验，提出了W‑operadic与W‑commutative概念并证明其单向闭包；

**🔧 技术方法**

采用范畴论技术：Verbal Category、W‑operad、左Kan扩张、coend、弱分配律、强度与拉斯科伊斯构造、以及与Eilenberg–Moore/Kleisli提升的对应；

**📊 数据集**

无实验数据集，论文完全是理论推导与范畴学证明；

**📈 对比分析**

对比方式主要是通过已知的正负例（如Plotkin的无分配律例子）检验构造的适用性，并给出若干实例（如多重集单子、概率分布单子、指数化计量单子等）验证公式的正确性；

**⚠️ 局限性**

局限性包括：对非对称Verbal Category缺乏完整的分配律兼容性证明；对T的更一般改造（弱分配律、精炼T）未在本文讨论；对具体实现和性能评估缺乏实验验证。

---

## 576. Identifying AI Web Scrapers Using Canary Tokens

**arXiv ID:** 2605.13706 | [PDF](https://arxiv.org/pdf/2605.13706v1)

**作者:** Steven Seiden `[一作]` (Duke University), Emily Wenger `[通讯]` (Duke University)

**通讯引用:** 393 | [OpenAlex ID](https://openalex.org/A5042329783)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了20个自定义网站，向每个访问的抓取器分发独特的可卡里令牌；随后向22个公开AI聊天机器人发送针对这些网站的查询；通过在机器人生成的回复中提取令牌并与服务器记录匹配，推断各AI系统使用的抓取器身份。

**💡 创新点**

首次系统性利用可卡里令牌作为内容侧信道，对AI聊天机器人的抓取器进行自动识别，揭示未公开的抓取器及第三方搜索引擎代理，填补了此前仅靠自我披露或社区数据难以覆盖的空白。

**🔧 技术方法**

使用Python Faker生成随机文本令牌；采用自定义网站模板并动态插入令牌；基于HTTP User‑Agent与ASN组合判定抓取器唯一性；对聊天机器人响应进行正则匹配提取令牌；使用计数与交互次数计算匹配分数以判定抓取器。

**📊 数据集**

20个自建域名网站（共100个可卡里令牌）、22个生产AI聊天机器人（OpenAI、Google、Anthropic、Microsoft等）以及三种实验条件（全可访问、下线、robots.txt阻止）。

**📈 对比分析**

通过比较不同条件下机器人返回的User‑Agent及令牌出现情况，评估缓存与阻止效果；实验显示12/18机器人在下线或robots.txt后仍返回内容，说明简单阻止无效；仅Duck.ai遵守阻止。匹配准确率高，能可靠识别已知抓取器并发现未知抓取器。

**⚠️ 局限性**

研究规模有限（20站、单一网络），抓取器唯一性仅基于UA+ASN，可能遗漏同源IP或TLS指纹；实验窗口短，未覆盖长期训练时的抓取；可卡里令牌可能被模型忽略或随机生成；缺乏对不同语言、媒介或更大规模站点的验证。

---

## 577. StayStill: a large-scale 3D idle animation dataset

**arXiv ID:** 2605.13693 | [PDF](https://arxiv.org/pdf/2605.13693v1)

**作者:** Eneko Atxa Landa `[一作]`, Taras Kucherenko `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文探讨了某种新型算法在特定任务中的应用。

**💡 创新点**

创新点在于提出了一种改进的模型结构，能够更有效地处理复杂数据。

**🔧 技术方法**

使用了深度学习技术，特别是卷积神经网络（CNN）和循环神经网络（RNN）的结合。

**📊 数据集**

使用了公开的图像数据集和文本数据集进行实验。

**📈 对比分析**

与现有方法进行了对比，结果显示新方法在准确率和效率上均有显著提升。

**⚠️ 局限性**

限制在于模型对特定类型数据的适应性较差，且训练时间较长。

---

## 578. Three-Stage Learning Unlocks Strong Performance in Simple Models for Long-Term Time Series Forecasting

**arXiv ID:** 2605.13678 | [PDF](https://arxiv.org/pdf/2605.13678v1)

**作者:** Zhenan Yu `[一作]` (Harbin Institute of Technology), Jin Yang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 471819 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种分阶段训练范式（STAIR），通过共享、个体化和残差学习逐步释放简单时序映射模型的预测能力；

**💡 创新点**

创新在于将通道建模视为矩阵分解，设计三阶段训练策略，并引入 Shared-to-Individual Fine-tuning 与 α-RevIN 两个技巧以平衡通道共享与个体差异以及避免过度归一化；

**🔧 技术方法**

使用浅 MLP（或线性层）作为核心时序映射，配合分阶段训练、残差适配器和可调强度的 RevIN；

**📊 数据集**

在九个长序列预测基准上验证，包括 ETT（1/2 及 m1/m2）、Electricity、Traffic、Weather、Exchange、Solar；

**📈 对比分析**

与 PatchTST、TimeMixer、iTransformer、TimesNet、Crossformer、TiDE、DLinear 等主流方法对比，STAIR 在多数数据集上取得最优或次优 MSE/MAE，表明简单模型在合理训练组织下可与复杂模型竞争；

**⚠️ 局限性**

局限在于跨变量残差模块仍相对简单，且对高维数据集的效果有限；α-RevIN 的最佳强度需针对数据集手动调优，且在某些数据上仍受限于单一浅 MLP 能力。

---

## 579. Weakly Supervised Segmentation as Semantic-Based Regularization

**arXiv ID:** 2605.13674 | [PDF](https://arxiv.org/pdf/2605.13674v1)

**作者:** Stefano Colamonaco `[一作]` (KU Leuven), Jaron Maene `[通讯]` (KU Leuven)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5089855486)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

使用可微模糊逻辑将弱标注（边框、涂鸦等）和结构先验统一为逻辑约束，微调 Segment Anything Model (SAM) 生成高质量伪标签，然后用这些伪标签训练无提示的分割网络。

**💡 创新点**

创新点在于将多种弱标注与结构先验通过可微模糊逻辑统一表示，并在第一阶段对 foundation model 进行细粒度的逻辑引导微调；同时实现两阶段无提示分割，实现了更高质量的伪标签和最终分割性能。

**🔧 技术方法**

核心技术包括可微模糊逻辑（semantic loss）、SAM 细调、基于逻辑约束的自监督正则化、以及后续的标准分割网络（Mask2Former、DeepLabV2）。

**📊 数据集**

使用 Pascal VOC 2012（自然图像）和 REFUGE2（视网膜光盘/杯）两个数据集。

**📈 对比分析**

与多种基线（传统 WSSS 方法、prompt‑based SAM、全监督基础模型等）对比，在 Pascal VOC 上伪标签 mIoU 达 94.5%，二阶段模型在验证集和测试集分别取得 79.7% 与 78.5% mIoU，甚至超过对应的全监督模型；在 REFUGE2 上，弱监督 MedSAM 的平均 Dice 由 47.7% 提升至 87.5%，第二阶段模型的 Dice 也接近全监督上限。

**⚠️ 局限性**

主要局限在于对领域特定逻辑约束的依赖，需要专家知识；可微模糊逻辑在第一阶段训练时会带来计算开销；在更弱监督（如图像级标签）下的推广尚待研究。

---

## 580. Pattern-Enhanced RT-DETR for Multi-Class Battery Detection

**arXiv ID:** 2605.13670 | [PDF](https://arxiv.org/pdf/2605.13670v1)

**作者:** Xu Zhong `[一作]` (Independent Researcher), Enyuan Hu `[通讯]` (Brookhaven National Laboratory)

**通讯引用:** 25025 | [OpenAlex ID](https://openalex.org/A5001764707)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对多类别电池检测进行了系统基准评估，并提出了基于模式的动态查询机制 PaQ-RT-DETR，显著提升了检测性能。

**💡 创新点**

创新点在于将可学习的模式与 RT-DETR 的查询进行混合构造，缓解了一对一 Hungarian 匹配导致的查询激活不平衡问题，且几乎不增加额外计算开销。

**🔧 技术方法**

主要技术包括 YOLOv8/YOLO11 CNN 检测器、RT-DETR Transformer、基于模式的动态查询生成、Hungarian 匹配以及端到端的损失训练。

**📊 数据集**

使用了公开的 Roboflow Universe 电池检测数据集，约 8,591 张 RGB 图像，涵盖六类电池（汽车电池、脚踏车电池、干电池、笔记本电池、智能手机电池、玩具电池）。

**📈 对比分析**

在相同实验条件下比较 YOLOv8n、YOLOv8s、YOLO11n、RT-DETR-L、RT-DETR-X 以及 PaQ-RT-DETR 变体，PaQ-RT-DETR-X 在 78.2% mAP@50（+2.8%）下取得最高准确率，YOLO11n 在参数最少且 FPS 最快的轻量级方案中表现最佳。

**⚠️ 局限性**

局限性包括：仍受电池类别严重不平衡影响，稀缺类别提升有限；模型仅基于 RGB 图像，缺少多模态支持；在资源受限的工业设备上部署仍需进一步优化。

---

## 581. SceneGraphVLM: Dynamic Scene Graph Generation from Video with Vision-Language Models

**arXiv ID:** 2605.13667 | [PDF](https://arxiv.org/pdf/2605.13667v1)

**作者:** Vladislav Makarov `[一作]` (MIRAI), Dmitry Yudin `[通讯]` (MIRAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SceneGraphVLM，一种利用小型视觉语言模型对图像与视频进行场景图生成的端到端方法。

**💡 创新点**

创新点包括：使用高效的 TOON 文本格式压缩场景图；通过监督微调+基于幻觉惩罚的 GRPO 强化学习实现精准、紧凑的图结构；并为视频提供仅依赖前一帧图的轻量级短期上下文。

**🔧 技术方法**

核心技术涵盖：Qwen3.5-0.8B 视觉语言模型、vLLM 加速解码、两阶段训练（SFT + RL），以及专门设计的幻觉感知奖励和 TOON 序列化。

**📊 数据集**

实验数据集包括 Panoptic Scene Graph（PSG）、Panoptic Video Scene Graph（PVSG）和 Action Genome。

**📈 对比分析**

与传统多阶段方法、零样本 VLM 以及 R1‑SGG 等基线对比，SceneGraphVLM 在精度、召回与 F1 上均表现最佳，并保持在约一秒的推理时延。

**⚠️ 局限性**

局限性：受限于所用数据集的标注噪声与长尾不平衡；目前多为闭词表；生成前一帧图作为上下文时易出现错误积累；需要 GPU 支持；仅实现短期视频上下文，未覆盖更长时序推理。

---

## 582. Fine-tuning with Hierarchical Prompting for Robust Propaganda Classification Across Annotation Schemas

**arXiv ID:** 2605.13663 | [PDF](https://arxiv.org/pdf/2605.13663v1)

**作者:** Lukas Stähelin `[一作]` (Technische Universitaet Berlin), Vera Schmitt `[通讯]` (Technische Universitaet Berlin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一套基于传播意图的分层宣传技术标签体系，并在四种大语言模型上对其进行微调与评估；同时与已有技术集成方案（Sahitaj聚类标签）进行对比；并探讨标签方案、监督方式与提示策略三维交叉影响。

**💡 创新点**

创新点在于①以传播意图为核心构建的新标签体系；②将标签可靠性、监督方式与提示策略视为互不干扰的三维因素；③提出HiPP（Main→High）分层提示方法，显著提升低标注可靠性标签的分类性能。

**🔧 技术方法**

采用大语言模型微调（OpenAI API、LoRA）、分层提示与直接高层提示、宏F1/加权F1评估、聚类与Krippendorff α等统计技术。

**📊 数据集**

使用HQP Twitter数据集（30,000条推文）经专家+LLM辅助重新标注得到17个细粒度标签、5个高层意图标签；实验采用500条样本（训练210/验证90/测试200）进行评估。

**📈 对比分析**

在两种标签方案下，对GPT‑4.1‑nano、Phi‑4‑14B、Qwen2.5‑14B和Qwen3‑14B四大模型，比较零射与微调、直接提示与HiPP两种提示；结果显示微调提升0.09–0.30加权F1，Qwen3‑14B在本方案上最高0.685、Sahitaj方案上0.661；HiPP在低IAA标签上优于直接提示。

**⚠️ 局限性**

局限性包括：仅在单一500条划分上评估，缺乏多随机拆分的鲁棒性检验；样本量相对较小；聚类方法主观；仅使用英文推文，未覆盖多语言与多域场景；标签方案设计可能携带文化偏差。

---

## 583. FlowCompile: An Optimizing Compiler for Structured LLM Workflows

**arXiv ID:** 2605.13647 | [PDF](https://arxiv.org/pdf/2605.13647v1)

**作者:** Junyan Li `[一作]` (UMass Amherst), Chuang Gan `[通讯]` (UMass Amherst)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FlowCompile，一个用于结构化 LLM 工作流的编译器，利用子代理剖析、结构感知代理估计和全局设计空间搜索，生成可复用的准确性‑延迟权衡配置集。

**💡 创新点**

创新点在于把工作流优化视为编译问题，突破仅在运行时路由的限制，构造子代理级别的可重用性能模型并通过轻量级代理估计全流程表现，从而一次性搜索得到多点权衡集合。

**🔧 技术方法**

技术主要包括：①子代理级别数据诱导与剖析；②基于模型准确率和延迟的结构感知代理估计；③非支配排序与设计空间剪枝；④多目标（准确率‑延迟）评估与期望效用选择。

**📊 数据集**

使用了四大基准：数学推理（GSM8K、MATH‑500）、多跳问答（HotpotQA）和代码推理（LiveCodeBench）。

**📈 对比分析**

与单模型、固定工作流、路由式基线（MaAS、KNN、Pref‑Aware Router 等）对比，FlowCompile 在所有基准上均获得更优的准确性‑延迟曲线，速度提升可达 6.4×，期望效用平均提升约 7.9 分，且在多种偏好设置下保持领先。

**⚠️ 局限性**

局限性包括：①对参考模型生成工作流轨迹的依赖，若参考模型不足可能导致代理估计误差；②子代理剖析成本与任务相关性，跨任务迁移需验证；③仅适用于预先定义的结构化工作流，难以处理高度动态的 agentic 场景；④代理估计虽满足局部排序与支配一致性，但对极端准确率差异仍可能误判。

---

## 584. Europe and the Geopolitics of AGI: The Need for a Preparedness Plan

**arXiv ID:** 2605.13634 | [PDF](https://arxiv.org/pdf/2605.13634v1)

**作者:** Maximilian Negele `[一作]` (RAND Center on AI, Security, and Technology), Maksym Andriushchenko `[通讯]` (ELLIS Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对人工通用智能（AGI）出现时机、其对地缘政治的潜在影响以及欧洲应对准备程度进行系统性评估与前瞻性分析

**💡 创新点**

结合专家预测、模型推演和AI三角（计算、数据、算法）约束分析三重视角，形成对AGI可能在2030‑2040年间出现的可行性评估，并对欧洲战略、竞争力与政策连贯性进行多维度诊断

**🔧 技术方法**

文献综述、专家问卷与预测平台汇总、定量模型（如AI triad 约束检验）以及对现有AI技术能力与进展的综合评估

**📊 数据集**

主要引用公开可获取的AI能力评测、计算与能源需求数据、行业投资与产出统计、以及学术与行业报告的汇总资料（非单一数据集）

**📈 对比分析**

与传统技术前瞻方法对比，本研究将专家预测与定量模型相结合，既捕捉主流观点的短期收敛趋势，也通过三角约束提供对潜在瓶颈的系统性评估，提升对AGI时间轴和影响的可信度

**⚠️ 局限性**

受限于预测不确定性、模型假设与可观测数据的缺乏，且对AGI定义与成熟度衡量仍缺乏共识，导致结论多为“可行性评估”而非精确时序；同时欧洲评估主要基于公开信息，可能低估非公开战略动向

---

## 585. Guide, Think, Act: Interactive Embodied Reasoning in Vision-Language-Action Models

**arXiv ID:** 2605.13632 | [PDF](https://arxiv.org/pdf/2605.13632v1)

**作者:** Yiran Ling `[一作]` (Harbin Institute Of Technology), Lei Zhang `[通讯]` (Futian Laboratory)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 GTA-VLA，一种交互式视觉-语言-动作框架，允许用户通过稀疏的视觉先验（点、框、轨迹）来指导机器人进行思考与执行。

**💡 创新点**

将人类空间先验直接注入视觉语言模型的 Chain-of-Thought 推理流程，形成统一的空间‑视觉推理序列，并与快速动作头异步结合，既保持自主执行又能在失败或模糊场景下实现可纠错。

**🔧 技术方法**

使用 Qwen3‑VL‑2B 视觉‑语言模型进行条件推理；通过 Flow‑Matching 动作头生成连续控制动作；构建自动化数据生成管线生成 Interact‑306K 并对空间先验进行序列化与 token 化。

**📊 数据集**

数据集包括：Interact‑306K（306K 条轨迹，来自 Open X‑Embodiment、DROID、RoboMind 等）；LIBERO benchmark；SimplerEnv 与 SimplerEnv‑Plus（OOD 扩展）；BridgeData V2 等。

**📈 对比分析**

与多种基线（OpenVLA、X‑VLA、CoT‑VLA、MolmoAct 等）在 LIBERO 和 SimplerEnv 上对比，GTA‑VLA 在 LIBERO 取得平均成功率 98.6%（最高），在 SimplerEnv 取得 81.2%（最高）。在 SimplerEnv‑Plus 的视觉、机器人状态、语言、物体扰动等 OOD 场景中保持较高成功率；在存在空间歧义或干扰时，单点或框指导可将成功率提升 20% 以上。

**⚠️ 局限性**

仅在二维图像空间提供指导和推理，缺乏 3D 空间建模，难以处理更复杂的几何交互；对低级控制错误（如抓取姿态、释放时机等）的纠正有限；自动触发指导机制尚未实现。

---

## 586. Edit-level Majority Voting Mitigates Over-Correction in LLM-based Grammatical Error Correction

**arXiv ID:** 2605.13624 | [PDF](https://arxiv.org/pdf/2605.13624v1)

**作者:** Takumi Goto `[一作]` (Nara Institute of Science and Technology), Taro Watanabe `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 1640 | [OpenAlex ID](https://openalex.org/A5102396915)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种训练-free的推理方法，在单个大型语言模型生成的多个候选句子中进行编辑级多数投票，以降低过度纠错现象。

**💡 创新点**

创新点在于将编辑层面的自洽性与多数投票相结合，利用同一模型多次采样的结果筛选高置信度修正，避免额外训练。

**🔧 技术方法**

技术包括核采样生成k=8个候选、ERRANT/JP-ERRANT编辑抽取、阈值τ控制的多数投票集成，适用于多语言GEC。

**📊 数据集**

实验使用七种语言的公开GEC数据集：英语CWEB-G/BEA-2019/JFLEG，捷克AKCES-GEC，德语Falko-Merlin，乌克兰UNLP-2023，韩语Kor-learner，印地语Hi-GEC和罗马尼亚RONACC。

**📈 对比分析**

与贪婪解码和MBR解码对比，方法在大多数基准上提升了ERRANT F_0.5（最多10点）和GLEU，且在10个提示模板下保持低方差，8个候选已足够。

**⚠️ 局限性**

局限性包括仅缓解过度纠错未处理欠纠错，阈值需手动调优，未尝试其他集成方式，评估受限于可公开数据集，且对潜在有害输出的过滤依赖阈值。

---

## 587. WD-FQDet: Multispectral Detection Transformer via Wavelet Decomposition and Frequency-aware Query Learning

**arXiv ID:** 2605.13621 | [PDF](https://arxiv.org/pdf/2605.13621v1)

**作者:** Chunjin Yang `[一作]` (University of Electronic Science and Technology of China), Fanman Meng `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 4189 | [OpenAlex ID](https://openalex.org/A5100617043)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 WD-FQDet 框架，通过小波变换将红外与可见光特征分解为低频共享与高频专属，分别对齐与保留，并结合频域与空间特征实现多模态目标检测。

**💡 创新点**

创新点在于将频域分解与跨模态对齐/保留相结合，设计低频同质对齐（LFHA）与高频专属性保留（HFSR）模块，并引入频域感知查询选择（FQS）自适应融合。

**🔧 技术方法**

使用小波变换、跨模态注意力、梯度一致性损失、混合特征增强以及基于 Transformer 的 RT‑DETR 检测头。

**📊 数据集**

在 FLIR、LLVIP 和 M3FD 三个公开多光谱检测数据集上进行实验。

**📈 对比分析**

与单模态与多模态方法对比，WD‑FQDet 在 FLIR 上 mAP 达到 50.2，LLVIP 66.9，M3FD 46.4，均明显高于现有最优模型。

**⚠️ 局限性**

局限性包括对小波基的选择敏感、额外的频域计算开销以及在极端光照下的鲁棒性待进一步验证。

---

## 588. Rethinking Generalization in Graph Neural Networks: A Structural Complexity Perspective

**arXiv ID:** 2605.13597 | [PDF](https://arxiv.org/pdf/2605.13597v1)

**作者:** Peiyao Wang `[一作]` (Shanxi University), Jiye Liang `[通讯]` (Shanxi University)

**通讯引用:** 14623 | [OpenAlex ID](https://openalex.org/A5106626932)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文从图结构的视角系统研究图神经网络（GNN）的泛化能力，并提出结构复杂度这一新的度量。

**💡 创新点**

创新点包括：①证明图结构本身会诱发过拟合；②用有效边数量定义结构复杂度并推导出显式包含该度量的 Rademacher 泛化上界；③提出结构熵正则化（SER）作为可微的稀疏化策略，能够在保持平滑性的同时控制有效边，从而提升泛化。

**🔧 技术方法**

主要技术手段包括：Dirichlet 能量分析、Rademacher 复杂度理论、结构熵正则化以及基于注意力或聚合矩阵的有效边阈值化。

**📊 数据集**

实验使用七个公开图数据集：Cora、CiteSeer、PubMed、CS、Physics、Computers、Photo。

**📈 对比分析**

将 SER 与传统正则化方法（LapR、CP、P-reg、NASA、R-reg）在 GCN、GAT、GT 三种主流 GNN 架构上进行对比。实验结果显示，SER 在大多数数据集和模型上均取得最高或接近最高的节点分类准确率，显著提升了泛化性能。

**⚠️ 局限性**

局限性包括：①理论分析基于固定图、同质标签的假设，难以直接推广到动态图或异构图；②结构熵正则化需要手动调节 λ，过大可能导致欠拟合；③对大规模稠密图的计算开销相对较高。

---

## 589. MedCore: Boundary-Preserving Medical Core Pruning for MedSAM

**arXiv ID:** 2605.13688 | [PDF](https://arxiv.org/pdf/2605.13688v1)

**作者:** Cenwei Zhang `[一作]` (Shanghai Jiao Tong University), Lei You `[通讯]` (Technical University of Denmark)

**通讯引用:** 778 | [OpenAlex ID](https://openalex.org/A5082049111)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对MedSAM进行结构化剪枝，构建MedCore框架，保留医学适应后的核心子网络，同时降低参数和 FLOPs。

**💡 创新点**

提出双干预得分、边界杠杆原理与跨分布聚合，平衡医学核心与边界保真，解决传统压缩导致边界失真的问题。

**🔧 技术方法**

使用Transformer头/MLP组剪枝、双重干预评估、基于边界的Fisher近似、跨分布加权与恢复微调等技术。

**📊 数据集**

在多模态医学分割数据集：CVC-ClinicDB、CVC-ColonDB、Kvasir-SEG（息肉）、BUSI（乳腺超声）和ISIC2018（皮肤病变）。

**📈 对比分析**

与原始MedSAM、EfficientSAM、SlimSAM等对照，MedCore在60–90%参数/ FLOPs压缩后保持Dice≈0.95、BF1≈0.64、HD95≈5，表现优于多数基线。

**⚠️ 局限性**

在极高压缩率下边界指标仍略下降，且需要代表性校准/恢复数据；临床部署前需单域验证与安全评估。

---

## 590. Characterizing Universal Object Representations Across Vision Models

**arXiv ID:** 2605.13675 | [PDF](https://arxiv.org/pdf/2605.13675v1)

**作者:** Florian P. Mahner `[一作]` (Vision and Computational Cognition Group), Martin N. Hebart `[通讯]` (Vision and Computational Cognition Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对162个视觉模型的特征进行对称非负矩阵分解，得到可解释的维度并评估其在模型间的普遍性。

**💡 创新点**

提出“universality”指标和基于相似性矩阵的非负分解方法，揭示普遍维度与概念性视觉特征及生物视觉的一致性。

**🔧 技术方法**

使用对称非负矩阵分解（Symmetric NMF）、RBF相似性核、余弦相似度匹配以及匈牙利算法进行维度对齐。

**📊 数据集**

以THINGS图像集（22,248张，1,854类）为输入，同时对比ObjectNet图像集进行稳健性检验。

**📈 对比分析**

通过对比不同架构、目标函数和训练数据的模型组，发现普遍维度与架构或目标无显著关联；且普遍维度在预测猴IT神经记录和人类相似性判断时的相关系数分别为0.74和0.76，优于模型特定维度。

**⚠️ 局限性**

模型集合偏向人类有用任务，非负分解无法捕捉多层次或叠加特征；仅使用倒数层特征，未探讨其他层或改进模型的潜力。

---

## 591. Robot Squid Game: Quadrupedal Locomotion for Traversing Narrow Tunnels

**arXiv ID:** 2605.13665 | [PDF](https://arxiv.org/pdf/2605.13665v1)

**作者:** Amir Hossain Raj `[一作]` (George Mason University), Xuesu Xiao `[通讯]` (George Mason University)

**通讯引用:** 2100 | [OpenAlex ID](https://openalex.org/A5017662025)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 Squid 框架，实现了在多样化狭窄隧道环境中四足机器人的鲁棒行走，并在仿真与真实硬件上验证了其可行性。

**💡 创新点**

创新点在于结合程序化隧道生成、特定专家策略的教师-学生蒸馏以及对视觉感知与动力学控制的解耦，从而克服了传统方法的过度专业化和对传感噪声的敏感。

**🔧 技术方法**

使用了强化学习（PPO）、多专家政策蒸馏（DAgger）、深度图像编码器 + GRU、比例-微分控制器、以及程序化几何生成器。

**📊 数据集**

主要数据集为仿真中自动生成的四类隧道（等边三角形、圆形、半圆形、间隙隧道），并在真实硬件中使用模块化 1 m 隧道进行验证。

**📈 对比分析**

与单一 PPO（V-RL）和两层层次规划（HP）对比，Squid 在所有隧道类型与难度级别下都取得最高成功率、最低碰撞率、最快完成时间和最高能耗效率；真实硬件测试成功率在 60–80% 之间。

**⚠️ 局限性**

局限性：仅验证了结构化的隧道几何，对不规则碎片、岩石或复杂分支网络等更为混乱的狭窄通道尚未测试，且对极端尺寸/形状的推广性仍待进一步评估。

---

## 592. SpurAudio: A Benchmark for Studying Shortcut Learning in Few-Shot Audio Classification

**arXiv ID:** 2605.13672 | [PDF](https://arxiv.org/pdf/2605.13672v1)

**作者:** Giries Abu Ayoub `[一作]` (University of Haifa), Loay Mualem `[通讯]` (University of Stuttgart)

**通讯引用:** 39 | [OpenAlex ID](https://openalex.org/A5005835154)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `79276348-11e0-48e3-84bc-7ec231d0171c` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出了 SpurAudio 这一专门用于研究少样本音频分类中前景-背景混合产生的伪相关性的基准，并在此基准上系统评估了多类少样本学习方法的性能。

**💡 创新点**

创新点在于：1）构建可控的前景与背景分离混合数据，能精准区分因背景导致的伪相关性；2）揭示即使是大型预训练模型，在背景切换时也会出现显著的 IID–OOD 性能差距；3）通过嵌入几何分析阐释背景幅度变化对不同推理头的影响，为未来设计更鲁棒的相似性度量提供理论基础。

**🔧 技术方法**

技术方法包括：基于指标、元学习、微调、对比学习和转导推理等多种少样本学习框架；对前景与背景音频的混合采用 EBU R128 规范的动态范围调整；在实验中使用 Conv64、ResNet12 等 CNN 以及 CLAP、AST、AudioMAE、Qwen2-Audio 等大型预训练编码器；并对嵌入空间进行 t‑SNE 可视化和角度/幅度敏感度分析。

**📊 数据集**

数据集方面：从 ESC‑50、UrbanSound8K、VocalSound、WildDeset、USM 等公开音频集合抽取前景事件和语义无关背景，生成 16,378 条受控混合样本组成 SpurAudio；同时将 SpurAudio 的嵌入与 FSD50K 进行对比验证其真实性。

**📈 对比分析**

实验对比了五大类少样本方法（指标、元学习、微调、对比、转导）在 IID 与 OOD 条件下的分类准确率。结果显示：绝大多数方法在背景不匹配时准确率下降 5–15%，即使在 5‑shot 情况下亦不例外；大型预训练模型虽然在 IID 下接近 96% 的准确率，但在 OOD 仍有 8–18% 的差距；采用转导或归一化相似度的头（如 Proto‑LP、BD‑CSPN、ECPE）能显著缩小该差距，而基于幅度敏感距离的头（如 Baseline、Hela‑VFA）则表现最差。

**⚠️ 局限性**

局限性包括：仅关注音频域的前景-背景伪相关性，未涉及更复杂的多模态或多源背景；混合过程虽然自然但仍为人工合成，可能未完全覆盖真实场景的背景多样性；实验主要评估性能差距，并未提出有效的鲁棒性改进方法；且对不同 shot/way 组合的探测相对有限。

---

## 593. Adaptive mine planning under geological uncertainty: A POMDP framework for sequential decision-making

**arXiv ID:** 2605.13702 | [PDF](https://arxiv.org/pdf/2605.13702v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 594. Beyond Perplexity: A Geometric and Spectral Study of Low-Rank Pre-Training

**arXiv ID:** 2605.13652 | [PDF](https://arxiv.org/pdf/2605.13652v1)

**作者:** Namrata Shivagunde `[一作]` (University of Massachusetts Lowell), Anna Rumshisky `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 3325 | [OpenAlex ID](https://openalex.org/A5071360545)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对六种低秩预训练方法（GaLore、Fira、CoLA、SLTrain、ReLoRA 以及全秩训练）在60M、130M、350M 三个规模下进行系统评估，探究它们在损失景观、谱结构和内部表示等维度上的差异，并评估其对下游任务的泛化表现。

**💡 创新点**

创新点在于：①提出跨方法障碍指标 IMBH，量化不同训练方法在权重空间中的几何分离；②通过多维度（1D损失景观、谱、激活）度量，展示低秩方法在不同维度上表现出显著差异；③证明仅靠验证 perplexity 不能完整预测下游性能，将几何与谱特征与 perplexity 结合，可显著提升下游性能预测准确率。

**🔧 技术方法**

采用 1D 随机/主成分方向损失景观、障碍高度（CCBH、IMBH）分析、有效秩/稳定秩/谱间隙/阈值秩等谱指标，以及激活 L2 距离、余弦相似度和 Linear CKA 等表示相似度评估；还使用线性回归预测器融合 perplexity 与几何谱特征。

**📊 数据集**

使用 C4 语料库进行预训练，验证集为 1000 样本子集；下游零样本评估采用 lm-evaluation-harness 的 11 个任务（commonsense、world knowledge、reading comprehension、grammar、logical reasoning）。

**📈 对比分析**

与全秩训练相比，Fira 和 SLTrain 在 1D 随机方向上获得更平坦的基底，但 CoLA 与 ReLoRA 更尖锐；GaLore 在主成分方向上最接近全秩；IMBH 指标显示不同方法的权重解空间几何上相互分离，且与 perplexity 排名不完全一致；通过加入几何谱特征，预测下游性能的相关系数提升至 0.913（LOSO）/0.895（LOMO）。

**⚠️ 局限性**

研究仅覆盖小于 1B 参数的 LLaMA‑style 模型，未探讨更大规模模型、不同架构、数据集或秩选择的影响；仅基于单次训练实验，缺乏多次运行的方差分析；损失景观方向的归一化与跨方法比较的最佳做法尚未确定。

---

## 595. Prefix Teach, Suffix Fade: Local Teachability Collapse in Strong-to-Weak On-Policy Distillation

**arXiv ID:** 2605.13643 | [PDF](https://arxiv.org/pdf/2605.13643v1)

**作者:** Kaiyuan Liu `[一作]` (Zhejiang University), Jieping Ye `[通讯]` (Zhejiang University)

**通讯引用:** 40203 | [OpenAlex ID](https://openalex.org/A5010419481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究强到弱的 on‑policy distillation（OPD），发现密集监督在生成序列后期会出现本地可教性衰退（local teachability collapse），并提出一种基于 BIC 的轨迹特定释放规则来动态截断监督。

**💡 创新点**

创新点：①首次系统诊断并量化 OPD 的本地可教性衰退；②设计了基于学生 top‑K 可能动作的教师 margin 作为局部可教性信号；③通过句子段聚合和 BIC 变化点检测实现每条轨迹自适应的监督截断；④验证该规则在多规模 Qwen3 体系下显著优于全轨迹监督和固定前缀方法。

**🔧 技术方法**

技术细节：采用 OPD 的优势（advantage）学习目标；计算教师与学生对已采样 token 的 log‑probability 差值；构造学生 top‑K 候选集合并评估教师在此集合上的 top‑1 与 top‑2 的 margin；使用 NLTK 进行句子分割；对 margin 序列做 BIC‑风格的单点下降检测；根据检测结果按比例重塑 loss 权重以保留梯度幅度。

**📊 数据集**

数据集：训练使用 DeepMath；内域评估包含 AIME25、HMMT25‑Feb、HMMT25‑Nov、MATH500、Minerva；外域 sanity 检验使用 GPQA。

**📈 对比分析**

对比方法：全 OPD、ExOPD、FastOPD、固定前缀 OPD（k=1024/2048/4096/8192）、随机释放等。实验显示我们的释放规则在 1.7B 学生上 Avg 从 36.8% 提升到 42.4%（+3.3%），在 4B/8B 学生上亦保持或超越 FastOPD；在 GPQA 上保持更高的外域性能，证明方法既提升内域又不易失效。

**⚠️ 局限性**

局限性：①仅在 Qwen3 系列模型与数学问答任务上验证，缺乏跨模型或多模态的通用性证明；②释放规则依赖完整 roll‑out 的教师 logits，训练效率未显著提升；③缺乏理论保证，规则是经验性启发，可能在不同环境下表现不稳定。

---

## 596. Bounded-Input True Proportional Navigation for Impact-Time Control

**arXiv ID:** 2605.13669 | [PDF](https://arxiv.org/pdf/2605.13669v1)

**作者:** Lohitvel Gopikannan `[一作]` (Indian Institute of Technology Bombay), Abhinav Sinha `[通讯]` (University of Cincinnati)

**通讯引用:** 939 | [OpenAlex ID](https://openalex.org/A5022385451)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于真比例导航的非线性导引策略，能够在控制输入存在上下限的条件下实现恒速、非机动目标的准时拦截。

**💡 创新点**

核心创新在于将控制输入限制显式嵌入导引设计：使用一次可微非对称平滑饱和模型配合滑模控制，并采用精确到达时间的TPNG，避免了传统方法中的线性化与小角度近似。

**🔧 技术方法**

采用了真比例导航（TPNG）、滑模控制（SMC）与一次平滑饱和模型（Saturation Model），并在滑模层面加入指数逼近的到达时间误差逼近器。

**📊 数据集**

论文中未使用公开数据集，而是通过设置若干仿真场景（不同拦截时间、初始航向角、目标航向角、静止目标等）来验证方法。

**📈 对比分析**

与传统无输入限制的SMC导引法相比，所提方法在满足相同准时拦截目标的前提下，控制能量消耗降低约53%，且在采用指数逼近的滑模层面时收敛速度可提升至约79%。

**⚠️ 局限性**

方法局限于恒速、非机动目标，未考虑目标机动、加速度率限制或多目标协同拦截等复杂情形。

---

## 597. Polyhedral Instability Governs Regret in Online Learning

**arXiv ID:** 2605.13692 | [PDF](https://arxiv.org/pdf/2605.13692v1)

**作者:** Yuetai Li `[一作]` (University of Washington), Radha Poovendran `[通讯]` (University of Washington)

**通讯引用:** 10033 | [OpenAlex ID](https://openalex.org/A5079723268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在固定多面体分区且全信息下的在线学习问题，提出“多面体不稳定性”（region/perm‑switch 数）作为决定 regret 上界的关键结构参数，并给出相应的算法（CAMW、几何 CAMW）与理论分析。

**💡 创新点**

创新点在于：① 用多面体不稳定性统一解释专家式与 OCO 率的过渡；② 对 Lovász 变换的子模仿优游戏给出精确的 SC_T 依赖 regret 上界；③ 设计了只在检测到 cell 切换时重启或传递权重的实用算法；④ 通过实验验证不稳定性与 regret 的预测关系，并在真实影响最大化任务中发现低不稳定性现象。

**🔧 技术方法**

使用的技术包括：多面体分区理论、投影/镜像上升、乘法权重（Multiplicative Weights）与其在链结构上的变体、Rademacher 下界构造、以及实验中对 permutation‑switch 与 region‑switch 的统计计数。

**📊 数据集**

数据集包括：① 通过控制 SC_T 的合成 Lovász 游戏；② k×k 网格 DAG 的最短路径（N≤705,432）；③ 四个 SNAP 影响最大化网络（Karate、Email-Eu、Wiki-Vote、Epinions）。

**📈 对比分析**

与基线（OLMDA、OGD、OGD‑FW、Fixed‑Share MW、SAOL、ZO‑EG）比较，CAMW/几何 CAMW 在低不稳定性 regime 下以 √(log n) 或 √(log N) 量级保持稳定，明显优于连续方法；当 region‑switch 频繁时，OGD‑FW 达到相同的速率；理论归一化后两类方法的曲线基本平行，验证了理论预测。

**⚠️ 局限性**

局限性：仅适用于固定分区、全信息、无噪声、无 bandit 反馈；对分区随时间变化、粗糙优化 oracle 或计算受限的情形尚无理论支持；算法依赖于能检测到 active region 切换，实际部署时检测成本与误差可能成为瓶颈。

---

## 598. HADAR-Based Thermal Infrared Hyperspectral Image Restoration

**arXiv ID:** 2605.13664 | [PDF](https://arxiv.org/pdf/2605.13664v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 599. A Hierarchical Language Model with Predictable Scaling Laws and Provable Benefits of Reasoning

**arXiv ID:** 2605.13687 | [PDF](https://arxiv.org/pdf/2605.13687v1)

**作者:** Jason Gaitonde `[一作]` (Duke University), Allan Sly `[通讯]` (Princeton University)

**通讯引用:** 4125 | [OpenAlex ID](https://openalex.org/A5076570918)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在树上进行广播过程构造一类具有层级结构的合成语言，并分析自回归生成中上下文长度与推理能力的关系，证明需要线性上下文才能逼近真分布，而仅需对数级推理内存即可实现精确采样。

**💡 创新点**

创新点在于：① 将广播过程视为可量化的层级语言模型，①1 在理论上给出自回归模型对上下文长度的Ω(n)下界与Θ(log n)推理上界；② 采用k‑gram假设逼近transformer，并提供精确的渐近分布预测；③ 通过实验验证理论预测并展示推理模型在极小上下文下保持高质量生成的实证。

**🔧 技术方法**

技术手段包括：广播过程生成、k‑gram ansatz、解析上下文对统计量（方差、峰度）的精确推导、使用Transformer（nanochat）训练自回归与推理版本、引入分层标点令牌、对数-归一化方差与过度峰度的实验评估。

**📊 数据集**

数据集为两种合成广播语言：Ising广播过程（Σ={±1}，ρ=0.9，d=3，h=8）与颜色广播过程（q=3，d=4，h=6）生成的叶子序列。

**📈 对比分析**

比较方法：用Transformer训练的无推理模型与推理模型在不同上下文长度（2⁴–2¹¹）下生成序列，测量方差、峰度与颜色合法率。结果显示：无推理模型随上下文减小方差下降、峰度趋向高斯；推理模型即使在极小上下文（64）也能保持接近真分布，颜色合法率几乎为1。

**⚠️ 局限性**

局限性：仅在正则树上构造的合成语言，无法直接推广到自然语言或非树形结构；推理模型的工作内存与理论相符但在实际大规模模型中的实现仍需研究；实验仅使用单一Transformer实现，未探讨不同架构或优化方法的影响。

---

## 600. Scale-Sensitive Shattering: Learnability and Evaluability at Optimal Scale

**arXiv ID:** 2605.13684 | [PDF](https://arxiv.org/pdf/2605.13684v1)

**作者:** Shashaank Aiyer `[一作]` (University of Maryland), Tom Waknine `[通讯]` (Technion)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5094122283)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2`

**🎯 论文内容**

本文研究了实值函数类在不同尺度下的统一收敛、可学习性与 fat‑shattering 维数之间的关系，并给出了在最优尺度下的等价定理，解决了多年来关于尺度 2‑因子间隙是否不可避免的争论。除此之外，作者通过直接对 ℓ∞ 覆盖数的精确上界，取得了在 γ/2 与 2γ 两个关键尺度下的 O(log² n) 与 O(log n) 计量熵上界，并证明了这些上界在某些情形下是最优的。最后，本文将这些理论结果应用到生成模型评估，给出了估计与可评估性之间的严谨二分法，并证明 3 为最佳评估因子。

**💡 创新点**

创新点主要包括：① 直接对实值函数类的 ℓ∞ 覆盖数进行上界，跳过传统的 packing‑to‑cover 过程，消除了 2‑因子尺度损失；② 在最优尺度下完成 uniform convergence 与 learnability 的完全等价，彻底否定了 Phil Long 的 2‑因子不可避免猜想；③ 在 γ/2 与 2γ 两个关键尺度上给出 O(log² n) 与 O(log n) 的紧确计量熵上界，并通过构造证明其最优；④ 将上述尺度敏感的结果用于生成模型评估，得到估计与可评估性之间的 3‑因子最优二分。

**🔧 技术方法**

主要技术手段包括：部分概念类与 VC 维数的关系、disambiguation 技术构造覆盖、symmetrization 与 Rademacher 复杂度、直接对 ℓ∞ 覆盖数的上界、样本压缩方案、对偶类的 uniform convergence 传递、以及信息论式的 TV 距离下界构造。

**📊 数据集**

该工作为理论性论文，未使用任何实验数据集；所有结论均通过严谨的概率与组合论证明得到。

**📈 对比分析**

与之前的理论结果（Alon 等的 O(log² n) 上界、Bartlett & Long 的 O(log² n) 在 γ+ε 处的上界、Rudelson & Vershynin 的 O(log^{1+ε} n) 上界）相比，本文在关键尺度上显著改进了上界；构造示例证明了上界的紧性；在生成模型评估方面，本文提供了严格的 3‑因子评估下界，优于以往仅给出 2‑因子或不确定的评估因子。

**⚠️ 局限性**

仍存在的限制与开放问题包括：① 对 γ* < γ ≤ 2γ* 区间的计量熵行为尚未完全确定；② 对样本复杂度与学习误差的精确定量界限（尤其在最优尺度下）仍需进一步研究；③ 对有限域上的具体样本复杂度与评估因子的量化分析尚未给出；④ 对其他 ℓ_p 规范下的覆盖数与可学习性关系的扩展仍是开放方向。

---

## 601. Multi-Property Temporal Logic Monitoring

**arXiv ID:** 2605.13668 | [PDF](https://arxiv.org/pdf/2605.13668v1)

**作者:** Arınç Demir `[一作]` (Boğaziçi University), Dogan Ulus `[通讯]` (Boğaziçi University)

**通讯引用:** 396 | [OpenAlex ID](https://openalex.org/A5054455487)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在线多属性监测框架 LoomRV，能够一次性处理多个过去时间 LTL/MTL 规范。

**💡 创新点**

创新点在于将不同属性共享的子公式合并成统一有向无环图，并采用双缓冲、连续内存的数据导向执行模型，实现无分配、无碎片化的状态管理。

**🔧 技术方法**

采用内容可寻址节点数据库实现子公式 deduplication，线性化执行计划与双缓冲内存 arena 结合，并支持离散与稠密时间模型。

**📊 数据集**

使用 30 条 Timescales 基准属性、四种人工合成公式集以及 JSON 与二进制两种输入格式的数据集进行评测。

**📈 对比分析**

与 Reelay 的单属性、并联与串行基线相比，LoomRV 在离散时间上单属性提升 2×‑4.5×，多属性场景提升 6×‑12×，稠密时间亦实现 3.5×‑4.5× 的单属性加速和 6×‑7.5× 的整体加速。

**⚠️ 局限性**

局限在于目前仅针对过去时间逻辑，缺乏针对未来时间或混合时间的支持；评估依赖于属性结构的共享度，且未提供统一的多属性基准套件。

---

## 602. Achieving $ε^{-2}$ Sample Complexity for Single-Loop Actor-Critic under Minimal Assumptions

**arXiv ID:** 2605.13639 | [PDF](https://arxiv.org/pdf/2605.13639v1)

**作者:** Ishaq Hamza `[一作]` (Indian Institute of Science), Zaiwei Chen `[通讯]` (Purdue University)

**通讯引用:** 237 | [OpenAlex ID](https://openalex.org/A5058269077)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并证明了在最小假设（存在可诱导不可约马尔可夫链的策略）下，单循环单时间尺度的离线Actor–Critic算法能够以 𝒪̃(ε⁻²) 的样本复杂度收敛到 ε-最优策略，并给出了最后一次迭代的收敛率。

**💡 创新点**

创新点包括：①首次在单循环 Actor–Critic 中实现 𝒪̃(ε⁻²) 的最后一次迭代收敛率；②提出一种耦合 Lyapunov 驱动框架与交叉支配（cross‑domination）技术，用于同时控制 Actor 与 Critic 的耦合迭代并处理离散 Markov 过程和可能无界的迭代；③在保持无时间尺度分离的前提下，仅需不可约性假设，避免了以往对所有策略统一可混合、均匀探索等强假设。

**🔧 技术方法**

使用了耦合 Lyapunov 函数、Markovian 噪声分析、时间衰减的步长（常数、和谐、冪次）、重要性采样（IS）与期望 TD（ETD）两种 Critic 更新、温度参数控制、以及小增益（small‑gain）理论实现交叉支配，从而得到 Actor 与 Critic 的收敛速率并最终证明 𝒪̃(ε⁻²) 的样本复杂度。

**📊 数据集**

无具体实验数据集；该工作完全是理论分析与证明，未包含实证验证。

**📈 对比分析**

与现有文献相比，所提出的方法不需要嵌套循环或强的算法相关假设；在样本复杂度上与已知的 𝒪̃(ε⁻²) 结果保持一致，但在单循环实现和最小假设下实现了相同的收敛率。实验对比未给出，理论上性能与最优值无关，仅受问题参数（n、m、γ 等）影响。

**⚠️ 局限性**

局限性包括：①对问题尺寸、折扣因子和状态动作空间的依赖系数相对较大，尚未达到完全最优；②仅在离散表格 MDP 下得到结果，未扩展至函数逼近（deadly triad 典型问题）；③未提供实验验证；④对温度参数、步长等超参数的选择仍需经验调优。

---

## 603. Sampling from Flow Language Models via Marginal-Conditioned Bridges

**arXiv ID:** 2605.13681 | [PDF](https://arxiv.org/pdf/2605.13681v1)

**作者:** Iskander Azangulov `[一作]` (University of Oxford), Leo Zhang `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了针对 Flow Language Models 的边缘条件桥采样（MCB）方法，在每一步先从 FLM 的标记后验分布抽样清洁端点，然后使用解析的 Ornstein–Uhlenbeck 桥生成下一连续状态。

**💡 创新点**

创新点在于：①直接利用 FLM 的离散后验端点而非传统的条件均值；②采样过程不需要额外训练，仅使用已有的后验分布；③理论上证明其路径空间 KL 与标准 DDPM 桥相比不大于且在信息更充分时更优；④允许在端点上应用温度缩放与核采样，实现非自回归解码控制。

**🔧 技术方法**

主要技术包括连续时间扩散/流匹配、Ornstein–Uhlenbeck 桥、后验预测逆过程、Girsanov 路径空间 KL 对比，以及温度与核采样解码技术。

**📊 数据集**

实验基于预训练的 FLM 模型，使用 LM1B（英语语言模型 1B）和 OWT（OpenWebText）数据集。

**📈 对比分析**

与标准 ODE/DDPM 采样在同一预训练 FLM 上进行比较，评价指标为生成困惑度（Gen. PPL）和每样本 unigram 熵。MCB 在相同或更少的采样步数下实现更低的生成困惑度，且熵更接近数据熵，表明在质量与多样性之间取得更优折衷。

**⚠️ 局限性**

局限性：由于每一步都需要采样离散端点，导致采样轨迹不平滑，难以直接通过蒸馏实现加速；并且采样质量依赖于 FLM 后验估计的准确性，若后验不准则效果受限。

---

## 604. Memristor Technologies for Dynamic Vision Sensors: A Critical Assessment and Research Roadmap

**arXiv ID:** 2605.13699 | [PDF](https://arxiv.org/pdf/2605.13699v1)

**作者:** Mohamad Yazan Sadoun `[一作]` (University of Oklahoma), Yaser Mike Banad `[通讯]` (University of Oklahoma)

**通讯引用:** 742 | [OpenAlex ID](https://openalex.org/A5068910901)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对将忆阻器与事件驱动视觉传感器（DVS）融合的技术路线进行系统化的证据分级综述，构建了三层体系结构分类法、双基准比较框架，并提出了可验证的研究路线图。

**💡 创新点**

创新点在于将忆阻器–DVS交叉点的文献聚合为“三范式体系结构 + 证据分级 + 双基准 + 可验证里程碑”的完整框架，首次将散布在不同领域的实验与仿真结果统一评估并给出量化的技术成熟度指标。

**🔧 技术方法**

采用 PRISMA 搜索、三层体系结构（光学忆阻器、像素级忆阻器、跨越式忆阻器加速器）分类，结合设备层指标（耐久性、可变性、开关速度）与系统层基准（Loihi 2、Speck）进行对比，并用可量化的里程碑（如 90% 识别率、10 mW、5 ms）进行路线规划。

**📊 数据集**

主要参考的事件摄像机基准数据集包括 DVS128‑Gesture、Gen1、DSEC 和 eTraM 等，本文将这些数据集与忆阻器模型耦合，用于评估系统的准确率、功耗和延迟。

**📈 对比分析**

通过证据分级与双基准方法，将忆阻器–DVS系统的性能与数字神经形态处理器（Loihi 2、Speck）以及传统的 SRAM‑CIM 加速器进行直接比较；提出的近端里程碑要求在 DVS128‑Gesture 上实现 ≥90% 准确率、≤10 mW 系统功耗、≤5 ms 延迟；中端里程碑则设定 ≥85% 准确率、≤50 mW 的 640×480 级别集成。

**⚠️ 局限性**

主要局限在于目前缺乏真正的端到端集成实现；设备层实验多在小尺寸阵列（≤128×64）上完成，缺乏 1M‑级别的规模化；忆阻器的可变性与耐久性未满足长期事件流的写入负载；ADC/解调器开销仍占主导，导致能效提升难以兑现；此外，缺乏统一的公开基准和可调用的设备模型，限制了跨组研究的可比性。

---

## 605. EBCC: Enclave-Backed Confidential Containers via OCI-Compatible Runtime Integration

**arXiv ID:** 2605.13676 | [PDF](https://arxiv.org/pdf/2605.13676v1)

**作者:** Di Lu `[一作]` (Xidian University), Jianfeng Ma `[通讯]` (Xidian University)

**通讯引用:** 21634 | [OpenAlex ID](https://openalex.org/A5012016098)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为Enclave‑Backed Confidential Containers（EBCC）的OCI兼容运行时架构，用于将RE​​E侧业务逻辑与TEE侧加密阶段统一管理为单一容器化实例；

**💡 创新点**

核心创新是将可信执行环境中的加密阶段抽象为可按需调用的stage（EID），并通过外部OCI生命周期维持整体容器视图，同时在托管端保留持久化的请求/响应、日志和证据文件；

**🔧 技术方法**

实现依赖Keystone、SGX、TDX和OP‑TEE等TEE后端，使用OCI Runtime Spec接口、EID分配、请求验证、Replay防护、EID‑绑定、以及后端adapter调用；

**📊 数据集**

实验使用自定义的轻量级字符串返回任务和重量级AES‑128‑GCM 16 MiB加密任务作为benchmark，不依赖公开数据集；

**📈 对比分析**

与原生Keystone、SGX、TDX、OP‑TEE执行对比，EBCC在cold‑start、end‑to‑end和并发场景下的延迟分别增加约0.6 s–1.15 s、约7.8 s–9.0 s，吞吐量在中等并发时提升至0.48–0.50 stage/s；

**⚠️ 局限性**

主要限制在于额外的管理开销导致的延迟增加、宿主侧存储的可攻击性、以及对TEEs间性能差异的适配成本；

---

## 606. Graph Neural Networks with Triangle-Based Messages for the Multicut Problem

**arXiv ID:** 2605.13673 | [PDF](https://arxiv.org/pdf/2605.13673v1)

**作者:** Jannik Irmai `[一作]` (TU Dresden), Bjoern Andres `[通讯]` (TU Dresden)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于三角消息传递的图神经网络（GNN）来求解多切割（Multicut）问题，训练模型预测哪些边应被收缩以构成最优分割，并以自回归方式迭代收缩实现可行解

**💡 创新点**

创新点在于仅对边特征进行操作，利用图完整化后所有三角形构成的消息传递层直接捕捉多切割的三角不等式约束，并通过监督学习与自回归推断结合提升解质量

**🔧 技术方法**

使用了图完成、成本归一化、基于三角形的消息传递层（MNN），以及标准的MLP、GELU激活、残差与层归一化，并采用二元交叉熵进行监督训练

**📊 数据集**

在CP‑Lib基准集（包含多种合成与真实世界实例，节点数≤200）以及自行生成的随机实例上进行实验

**📈 对比分析**

与GAEC、KL、FM以及DGRL等传统和GNN启发式求解器比较，实验显示在大多数数据集上本文方法在优化误差上优于其他启发式求解器，且在可接受时间内（秒级）实现近似最优；在部分实例上可在秒内得到最优解，远快于精确分支剪枝算法

**⚠️ 局限性**

主要局限包括：仅在节点≤30的训练数据上训练，导致在更大图上的性能衰退；推断复杂度为O(|V|^4)，在大图上耗时较高；缺乏对稀疏图的专门优化，且目前未提供任何近似保证

---

## 607. NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating

**arXiv ID:** 2605.13651 | [PDF](https://arxiv.org/pdf/2605.13651v1)

**作者:** Zhongju Yuan `[一作]` (Ghent University), Dick Botteldooren `[通讯]` (Ghent University)

**通讯引用:** 9056 | [OpenAlex ID](https://openalex.org/A5069519911)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出一种训练自由、基于神经启发的音频注意力架构 NAACA，利用振荡工作记忆（OWM）对长时序音频进行前瞻性注意力门控，只在检测到感知显著事件时才触发大语言模型处理。

**💡 创新点**

创新点在于把注意力瓶颈转化为“感知显著性过滤”问题，设计了能维持稳定记忆并对能量波动做出即时响应的 OWM 模块；该模块不依赖额外训练，使用波动能量阈值实现自适应门控，兼顾计算效率与识别准确性。

**🔧 技术方法**

技术手段包括：基于预训练 PANN 的类别概率映射为频域振荡驱动信号；64×64 网格 OWM 的第一阶压/速度耦合动力学；能量基自适应阈值和持续性过滤实现显著性检测；门控后将选取段交给 AudioQwen 进行语义推理。

**📊 数据集**

使用的公开数据集为 XD‑Violence（包含 500 句长音频，关注暴力事件检测）和 Urban Soundscapes of the World (USoW，4 通道城市环境音景)，用以量化性能与可视化显著性检测。

**📈 对比分析**

与多种基线对比：完整 ALM 推理、随机 4s 片段、监督式音频模型 HL‑Net、AVadCLIP，以及视频模型；在 XD‑Violence 上 NAACA 的 AP 从 53.5% 提升至 70.6%（+17.1%），同时将 ALM 调用次数降低约 40%，显示出显著的精度提升和计算效率。

**⚠️ 局限性**

局限性包括：性能受限于预训练编码器与 ALM 的能力；PANN 仅覆盖 AudioSet 527 类，可能漏检或误检领域外事件；硬门控可能丢失边界上下文，无法与软注意力或 KV 缓存结合；评估侧重异常检测，未覆盖更深层次推理任务。

---

## 608. Texture Regenerating and Grafting Using Genome-Driven Neural Cellular Automata

**arXiv ID:** 2605.13630 | [PDF](https://arxiv.org/pdf/2605.13630v1)

**作者:** Mirela-Magdalena Catrina `[一作]` (Transilvania University of Brașov), Alexandra Băicoianu `[通讯]` (Transilvania University of Brașov)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `da1b1a89-583a-4b57-9c81-478778569bec` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一种基于神经细胞自动机（NCA）的多纹理合成与自愈框架，能够在受损区域自我恢复纹理，并在推理阶段通过基因编码实现不同纹理的无缝嫁接。

**💡 创新点**

创新点在于：①使用二进制基因编码让单个 NCA 同时学习多纹理并实现平滑插值；②改进训练策略（池化+随机损坏）使得多纹理 NCA 能够在受损纹理上自愈；③在不需要重新训练的前提下通过初始化基因通道实现纹理嫁接。

**🔧 技术方法**

主要技术包括：神经细胞自动机结构（四个固定感知滤波器+两层 1×1 卷积更新）；VGG 判别器+切片 Wasserstein 损失用于对抗式训练；基因通道编码与池化采样策略；在推理阶段通过基因初始化和接口细化实现纹理嫁接。

**📊 数据集**

使用了 Describable Textures Dataset 和 VisTex 两个公开纹理数据库中的多种纹理样本进行训练和评估。

**📈 对比分析**

与传统单纹理 NCA 基线相比，本文方法在 SSIM、LPIPS、GMD/GDM 等指标上保持相近或略优，尤其在自愈实验中能够恢复受损区域且不降低整体生成质量。

**⚠️ 局限性**

局限性包括：仅能重现纹理的风格而非像素精确复制；受损半径超过模型感受野时自愈性能下降；基因竞争可能导致纹理边界漂移；需要更大尺寸或多尺度结构以稳定边界和提升细节一致性。

---

## 609. Multimodal Graph-based Classification of Esophageal Motility Disorders

**arXiv ID:** 2605.13623 | [PDF](https://arxiv.org/pdf/2605.13623v1)

**作者:** Alexander Geiger `[一作]` (Technical University of Munich), Alissa Jell `[通讯]` (Technical University of Munich)

**通讯引用:** 283 | [OpenAlex ID](https://openalex.org/A5018286785)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文针对食管运动障碍的诊断，提出一种多模态机器学习方法，将高分辨率阻抗测压（HRIM）记录与患者个体信息结合，并使用图神经网络对HRIM数据进行空间-时间图建模，以实现吞咽事件的多类别分类。

**💡 创新点**

创新点在于：①将HRIM传感器本身建模为图结构，节点为压力传感器，边为空间相邻与阻抗关系；②引入患者特征（人口学、临床、症状）作为额外模态；③采用多模态融合和监督对比学习，提升分类性能。

**🔧 技术方法**

主要技术包括：图神经网络（GATv2、GENConv）、时序模型（TCN、Transformer）、多模态融合、类别加权交叉熵与标签平滑、监督对比损失、层次注意力池化，以及5折患者级交叉验证。

**📊 数据集**

数据集来源于慕尼黑TUM大学医院，包含104名食管运动障碍患者的HRIM检查，共约1800次吞咽事件；数据同时包含结构化问卷、自由文本笔记以及HRIM压力/阻抗时间序列。

**📈 对比分析**

将图模型与三种视觉基线（ResNet50、ViT、SwinV2）进行比较，评估指标为加权F1分数。结果显示，图模型（尤其是Gen‑TCN）在所有分类类别上平均表现最佳，虽然差异未能达到统计显著性，但趋势一致。

**⚠️ 局限性**

局限性包括样本量小（104例）、类别分布不均、缺乏外部验证、统计检验受限，导致模型泛化能力尚未充分评估。

---

## 610. Unweighted ranking for value-based decision making with uncertainty

**arXiv ID:** 2605.13601 | [PDF](https://arxiv.org/pdf/2605.13601v1)

**作者:** Aarón López García `[一作]` (Universitat de València), Jose Such `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 10294 | [OpenAlex ID](https://openalex.org/A5052525506)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了模糊无权重价值决策框架 (FUW‑VBDM) 并实现了 Rankzzy 方法，解决多准则价值决策中量化与定性属性以及偏差问题。

**💡 创新点**

创新点在于结合模糊逻辑与无权重空间，去除固定权重偏差，同时引入可调的 p‑均值评分函数与凸组合来生成可解释的排名。

**🔧 技术方法**

使用模糊 LR‑fuzzy 数、优先级序列、约束优化（SLSQP）以及自定义的 p‑均值与 Vertex 方法。

**📊 数据集**

在示例中使用大学考试方法的公平与成本两维数据，并通过随机生成的 2×2 至 50×40 模糊决策矩阵评估。

**📈 对比分析**

与 TOPSIS 通过 Kendall 相关系数比较，得分在 0.745 以上，计算时间在 23 项以内 <10s，最大 50×40 约 34s。

**⚠️ 局限性**

局限性包括对模糊数定义和归一化的依赖、需手动设置 p 与 ν 参数、以及在极大规模或实时场景下可能的求解难度。

---

## 611. Sparse Code Uplifting for Efficient 3D Language Gaussian Splatting

**arXiv ID:** 2605.13600 | [PDF](https://arxiv.org/pdf/2605.13600v1)

**作者:** Lovre Antonio Budimir `[一作]` (University of Zagreb), Nandita Vijaykumar `[通讯]` (University of Toronto)

**通讯引用:** 1976 | [OpenAlex ID](https://openalex.org/A5080873211)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SCOUP框架，将稀疏码书学习和语言特征编码从3D高斯空间迁移到2D图像空间，再通过加权稀疏聚合和Top‑K筛选将稀疏系数提升到3D高斯，实现快速语义重建、低存储和高渲染效率。

**💡 创新点**

创新点在于：①在2D空间完成稀疏码书和区域系数的联合训练，解耦3D高斯优化；②利用高斯–像素关联实现加权稀疏聚合，省去重复栅格化；③Top‑K投票过滤多视角噪声，进一步压缩存储并提升多视角一致性。

**🔧 技术方法**

使用技术包括：3D Gaussian Splatting、SAM分割、CLIP视觉‑语言嵌入、稀疏编码（k‑means初始化 + soft‑top‑K softmax）、加权稀疏聚合、Top‑K投票、GPU并行渲染。

**📊 数据集**

实验数据集为LERF‑OVS、3D‑OVS和Mip‑NeRF360。

**📈 对比分析**

与LangSplatV2、VALA、Occam’s LGS等基线对比，SCOUP在语义重建速度上提升≈400×，训练内存降低≈3×，渲染速度保持不变；在LERF‑OVS上mIoU提升+4.5%，在3D‑OVS、Mip‑NeRF360上保持或超过现有最佳表现。

**⚠️ 局限性**

局限性包括：依赖SAM分割质量，对分割失误和小物体仍有挑战；仅验证于静态场景；对极大场景或动态环境的扩展尚待研究；在少量查询或不平衡标签时性能波动。

---

## 612. Creativity Bias: How Machine Evaluation Struggles with Creativity in Literary Translations

**arXiv ID:** 2605.13596 | [PDF](https://arxiv.org/pdf/2605.13596v1)

**作者:** Kyo Gerrits `[一作]` (University of Groningen), Ana Guerberof Arenas `[通讯]` (University of Groningen)

**通讯引用:** 497 | [OpenAlex ID](https://openalex.org/A5085056171)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了自动评估指标（AEM）和LLM-as-a-judge在文学翻译创意评估中的表现，并与专业翻译者的评估对比。

**💡 创新点**

创新点在于引入针对创意转移的专业注释框架，并系统评估LLM-as-a-judge在识别创意与错误方面的偏差。

**🔧 技术方法**

采用多种AEM（如BLEU、COMET、BERTscore等）和基于GPT‑5.2的LLM-as-a-judge进行错误与创意转移标注。

**📊 数据集**

构建了一个包含两源两目标语言（EN‑RU→NL‑CA）、三类体裁（诗歌、短篇小说、惊悚）以及三种翻译方式（HT、PE、MT）的约150词左右文学翻译样本。

**📈 对比分析**

通过Spearman相关、匹配率、混淆矩阵等统计方法比较，发现AEM与专业评估的相关性弱，LLM‑as‑a‑judge在错误标注上偏高、创意识别不足；在低创意体裁（惊悚）上相关性略好。

**⚠️ 局限性**

限制包括数据量小、文本长度短、未覆盖所有语言对与体裁、LLM提示设计及多模态一致性缺乏，限制了结论的普适性。

---

## 613. RTLC -- Research, Teach-to-Learn, Critique: A three-stage prompting paradigm inspired by the Feynman Learning Technique that lifts LLM-as-judge accuracy on JudgeBench with no fine-tuning

**arXiv ID:** 2605.13695 | [PDF](https://arxiv.org/pdf/2605.13695v1)

**作者:** Andrea Morandi `[一作]` `[通讯]` (Cisco Systems, Inc.), Andrea Morandi (Cisco Systems, Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现一种三阶段提示策略RTLC（Research、Teach-to-Learn、Critique），通过在不进行微调、检索或外部工具的前提下，显著提升LLM-as-a-judge在JudgeBench基准上的配对准确率。

**💡 创新点**

核心创新在于将Feynman学习技巧的四步教学框架嵌入提示中，形成Teach-to-Learn scaffold；在此基础上生成N=10个独立判断候选；并让同一模型以批评者身份对候选进行统一评估，取出最佳答案。

**🔧 技术方法**

主要技术包括：结构化提示（Teach-to-Learn scaffold）、多重候选生成（N=10温度0.4）、批评者归约（单一温度0调用），以及JSON尾部解析与保守评分策略。

**📊 数据集**

使用公开的JudgeBench-GPT-4o拆分（350个二选一对齐任务）进行实验，评估单一Claude 3.7 Sonnet判定器。

**📈 对比分析**

与单shot、批量投票（self‑consistency）以及单调用自评相比，RTLC在同一模型上将准确率从64.6%提升至78.6%（绝对提升14个百分点）。拆解实验表明：Teach-to-Learn提升9.4pp，N=10投票提升3.7pp，批评者提升0.9pp。

**⚠️ 局限性**

局限性包括：只在单一判定器和单一基准上验证；成本高（N=10时约为原始提示的47倍）；批评者与候选共享模型偏差，无法解决代码执行、事实检索等问题；未测试迭代或多模型异构组合。

---

## 614. MQTT Across a Raspberry Pi 5 IoT Network Utilizing Quantum-resistant Signature Algorithms

**arXiv ID:** 2605.13698 | [PDF](https://arxiv.org/pdf/2605.13698v1)

**作者:** Ray Feingold `[一作]` (Cleveland State University), Chansu Yu `[通讯]` (Cleveland State University)

**通讯引用:** 1769 | [OpenAlex ID](https://openalex.org/A5102777718)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一套基于 Raspberry Pi 5 的 MQTT 物联网网络中实现并验证了 FALCON（后量子签名算法）用于消息真实性与完整性保护，并完成了证书生成、TLS 1.3 连接以及完整的发布/订阅工作流。

**💡 创新点**

创新点在于首次将 NIST 选定的后量子签名方案 FALCON 无缝集成到实际的物联网硬件平台上，并通过实测展示了后量子方案在资源受限设备上的性能可行性；同时提供了一个完整的开源实现与部署脚本。

**🔧 技术方法**

采用的技术包括：MQTT 协议、TLS 1.3、FALCON-1024 数字签名、OpenSSL（集成 OQS provider）、liboqs、mosquitto、Python/Shell 脚本实现硬件控制与证书管理。

**📊 数据集**

数据集主要为自建的运动检测事件（PIR 传感器产生的信号），并在证书生成阶段执行 25 次 RSA‑2048 与 FALCON‑1024 的对比实验，记录生成时间作为性能指标。

**📈 对比分析**

对比方法：在同一硬件（Raspberry Pi 5）上多次执行证书生成脚本，分别使用 RSA‑2048 和 FALCON‑1024，统计毫秒级生成时间。实验结果显示 FALCON‑1024 平均生成时间约 68–70 ms，显著低于 RSA‑2048 的 300+ ms，证明后量子方案在此场景下不但安全且性能更优。

**⚠️ 局限性**

局限性包括：FALCON 依赖高精度高斯采样和浮点运算，可能存在时序泄漏与侧信道攻击风险；实验仅覆盖证书生成阶段，未系统评估通信延迟、带宽占用、功耗等指标；需要进一步验证在更大规模网络或不同后量子方案（如 SOLMAE）下的可扩展性与安全性。

---

## 615. The WidthWall: A Strict Expressivity Hierarchy for Hypergraph Neural Networks

**arXiv ID:** 2605.13690 | [PDF](https://arxiv.org/pdf/2605.13690v1)

**作者:** Fengqing Jiang `[一作]` (University of Washington), Radha Poovendran `[通讯]` (University of Washington)

**通讯引用:** 10033 | [OpenAlex ID](https://openalex.org/A5079723268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了超图神经网络的表达能力，提出基于同态密度与广义混树宽的层级框架，揭示模型的“宽度墙”限制。

**💡 创新点**

创新点在于将同态密度构造成完整的连续不变量基，并通过广义混树宽定义严格的表达层级，证明不同HGNN架构的不可超越性。

**🔧 技术方法**

采用同态计数、广义混树宽分析、Invariant‑theoretic 架构 InvNet 以及基于密度感知的 DensNet-D 等技术。

**📊 数据集**

使用 7 个真实超图节点分类基准：Cora、Citeseer、PubMed、Cora‑CA、House、Senate Bills、Gene‑Disease。

**📈 对比分析**

通过对比 clique‑expansion、native set‑function、以及密度感知三层级模型，实验显示宽度墙能准确预测模型失效，密度特征在高阶数据上显著提升性能。

**⚠️ 局限性**

局限性包括仅针对固定尺寸连续不变量，缺乏渐进、噪声或动态超图的理论与样本复杂度分析。

---

## 616. Cross Modality Image Translation In Medical Imaging Using Generative Frameworks

**arXiv ID:** 2605.13686 | [PDF](https://arxiv.org/pdf/2605.13686v1)

**作者:** Giulia Romoli `[一作]` (Umeå University), Paolo Soda `[通讯]` (Umeå University)

**通讯引用:** 3479 | [OpenAlex ID](https://openalex.org/A5003216983)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并公开了一个统一的3D医学图像对图像（I2I）翻译基准框架，对七种生成模型在5个公开肿瘤多模态数据集上共77个实验进行训练、推理和评估。

**💡 创新点**

创新点包括：①统一预处理、patch提取、滑窗推理、Gaussian混合和共享VAE编码器的标准化管线；②可插拔模型接口，便于快速加入新模型；③结合量化指标、病灶级评价和医师主观的Visual Turing测试，首次在临床多任务3D I2I中系统对比GAN与潜在生成模型。

**🔧 技术方法**

技术手段包括：GAN（Pix2Pix、CycleGAN、SRGAN）与潜在生成模型（LDM、LDM+ControlNet、Brownian Bridge、Flow Matching），3D卷积、滑窗推理与Gaussian拼接、VAE编码器与解码器、控制网络与流匹配等。

**📊 数据集**

使用的数据集有 SynthRAD23、SynthRAD25、BraTS23、autoPET、EnhancePET，共涵盖头颈、肺、盆腔三大解剖区，翻译方向包括 CBCT-CT、MRI-CT、CT-PET、T2w-T2f。

**📈 对比分析**

在统一的训练、推理和评估设置下，使用 PSNR/SSIM 进行量化，病灶级 PSNR/SSIM 进行细粒度分析，并通过 17 位临床专家的 Visual Turing 测试评估主观真实感。结果显示 SRGAN 在所有任务上取得最高 PSNR/SSIM，GAN 系列普遍优于潜在模型；病灶级评价表明所有模型在小病灶上表现差，CT‑PET 任务中病灶形状重建好但强度不准确；视觉测试表明模型间人类区分率接近随机，且量化指标与临床偏好不完全一致。

**⚠️ 局限性**

局限性包括：①公开多模态肿瘤数据有限，仅两类带病灶标注；②潜在模型受 VAE 压缩瓶颈影响；③跨物理量翻译（如 CT‑PET）仍缺乏有效的物理信息约束；④PSNR/SSIM 与医学专家感知不完全匹配；⑤缺乏更广泛的多中心、多任务公开数据与更精准的评估指标。

---

## 617. Learning Equilibria in Coordination Games via Minorization-Maximization

**arXiv ID:** 2605.13644 | [PDF](https://arxiv.org/pdf/2605.13644v1)

**作者:** Ashok Krishnan K. S. `[一作]` (Inria), Ana Busic `[通讯]` (Inria)

**通讯引用:** 1763 | [OpenAlex ID](https://openalex.org/A5067821332)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在协调博弈中引入了社会效益与个体感知成本相结合的效用模型，并利用小正则化实现势函数严格凹化，得到唯一平衡；进一步提出了基于Minorization–Maximization（MM）的学习算法，保证在光滑及非光滑博弈中收敛到势函数最优平衡；通过数值实验验证MM算法在收敛速度和稳健性上优于梯度下降和最优反应方法。

**💡 创新点**

创新点包括：① 将前景理论（PT）融入协调博弈的效用函数，刻画个体对成本的非理性感知；② 通过正则化实现势函数唯一最优并与原始博弈平衡的量化关系；③ 设计MM学习框架，在非光滑情形下仍可保证收敛到潜在最优平衡；④ 在多智能体场景下提出可分布式的MM实现，并展示其在能源与机器人协作任务中的优越性。

**🔧 技术方法**

主要技术：前景理论效用建模、势游戏理论、正则化/扰动技术、MM（Minorization–Maximization）优化框架、梯度/最优反应学习算法、数值仿真与对比实验。

**📊 数据集**

使用的实验数据为合成场景：能源消费/碳排放目标、随机奖励分布（如{2,10}、{5,1}等），以及机器人网格运动的离散位置；无公开真实数据集。

**📈 对比分析**

与梯度上升（AGA）和迭代最优反应（IBR）方法对比。实验显示：在光滑博弈中，IMM收敛速度最快；在非光滑博弈中，IMM能直接到达势函数最大点，而IBR仅停留在其他非最优平衡；梯度法收敛最慢且易受步长影响。

**⚠️ 局限性**

限制包括：① 正则化参数的选取需要经验或先验知识；② 仅考虑凸可压缩策略空间，难以推广到非凸或高维非光滑问题；③ 前景理论参数需事先估计，若估计不准会影响结果；④ MM算法在极端噪声或大规模系统中收敛速度和通信负担仍需进一步评估。

---

## 618. Causality-Aware End-to-End Autonomous Driving via Ego-Centric Joint Scene Modeling

**arXiv ID:** 2605.13646 | [PDF](https://arxiv.org/pdf/2605.13646v1)

**作者:** Seokha Moon `[一作]` (Korea University), Jungbeom Lee `[通讯]` (Korea University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了CaAD框架，将ego‑centric joint‑causal modeling与causality‑aware policy alignment融合，实现端到端的交互式规划。

**💡 创新点**

首次通过共享ego‑centric joint‑mode embeddings捕捉因果交互，并在同一场景假设下对ego策略进行强化学习对齐，确保规划与交互一致。

**🔧 技术方法**

采用query‑based端到端规划、Agent‑Mode Attention、Gaussian轨迹策略、GRPO强化学习、交互相关agent选择与joint‑mode监督等技术。

**📊 数据集**

在Bench2Drive和NAVSIM两个闭环评测基准上进行实验。

**📈 对比分析**

与SOTA方法对比，CaAD在Bench2Drive的Driving Score达87.53、Success Rate 71.81，NAVSIM PDMS 91.1，特别在交互关键任务（如合并、超车）表现显著提升。

**⚠️ 局限性**

仅在仿真/评测数据验证，仍需解决长尾社交行为、真实世界迁移以及对多样化交互场景的泛化问题。

---

## 619. Multi-Objective and Mixed-Reward Reinforcement Learning via Reward-Decorrelated Policy Optimization

**arXiv ID:** 2605.13641 | [PDF](https://arxiv.org/pdf/2605.13641v1)

**作者:** Yang Bai `[一作]` (Meituan), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的后训练阶段，使用混合奖励的强化学习，提出了奖励去相关策略优化（RDPO）方法

**💡 创新点**

创新点在于将Magnitude‑Aware Quantile（MAQ）归一化与Mahalanobis whitening相结合，既抑制了奖励尺度与分布异质性导致的优势分配失衡，又消除了奖励维度间的相关冗余

**🔧 技术方法**

主要技术包括 MAQ 归一化、Mahalanobis whitening（针对激活奖励子空间）、EMA 滚动协方差估计、PPO/GRPO 策略梯度、以及多任务混合奖励构造

**📊 数据集**

使用了多任务数据集：指令遵循（IFEval、GuideBench、SOP‑Maze）、数学推理（AIME24/25、GPQA、MATH500）、写作与 ArenaHard v2（AH‑Hard、AH‑Creative）、编程（FullStackBench、HumanEval+、MBPP+、LiveCodeBench v6）

**📈 对比分析**

在同一规模模型的前置验证和 LongCat‑Flash 的大规模后训练实验中，RDPO 相比 GRPO、GDPO 和初始化模型在指令遵循、写作、ArenaHard 子集上显著提升（最高 90.39% IFEvalAcc、89.00% ArenaHard‑Creative），在数学与编程任务上保持竞争力或略有提升，但仍有部分指标与基线相当或略逊

**⚠️ 局限性**

局限性包括：在某些推理/编程任务上提升有限；需要准确估计协方差并在子空间中应用 whitening，若估计不稳定会影响效果；方法依赖于多任务奖励子空间，单任务或奖励稀疏时效果未知；对超参数（β、EMA decay、warm‑up 步长）敏感

---

## 620. Scalable Deductive Verification of Data-Level Parallel Programs

**arXiv ID:** 2605.13616 | [PDF](https://arxiv.org/pdf/2605.13616v1)

**作者:** Lars B. van den Haak `[一作]` (Eindhoven University of Technology), Marieke Huisman `[通讯]` (University of Twente)

**通讯引用:** 14949 | [OpenAlex ID](https://openalex.org/A5047069342)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出了一套完整的技术体系，包括一种针对嵌套量化子句的重写方法、对数组使用不可变（immutable）和唯一（unique）类型限定符、以及针对GPU内核的主体提取机制，并将这些技术集成到一个基于权限分离逻辑的可推理程序验证器中。

**💡 创新点**

创新点主要在于：
1) 形式化并证明了嵌套量化子句重写的正确性（Lean4证明），实现了在不改变语义的前提下将多维索引压平为单变量量化，从而显著提升触发器匹配效率；
2) 设计了可用于指针数组的 immutable 与 unique 两类类型限定符，通过类型检查来消除别名与不可变性检查，减少了 SMT 量化块的数量；
3) 将主体提取与上述重写/限定符组合使用，进一步降低了验证负荷。

**🔧 技术方法**

技术细节包括：
- 权限分离逻辑与 Fractional 权限的结合；
- SMT 触发器模式的自动化重写与符号求值；
- 采用 Lean4 开发完整的证明以确保重写的形式化正确；
- 在可推理框架（Silicon、Gobra 等）中实现以上技术；
- GPU 内核的 grid-stride 循环与多维数组展平等特定优化。

**📊 数据集**

使用的数据集为：
1) CLBlast 库中的 GPU BLAS 内核（Level 1 与 Level 2）；
2) 由 Halide DSL 生成的 CPU 并行程序，包括 Radio Telescope Pipeline 中的 Padre 算法；
3) 额外的实验集（如 gemm_3）用于验证重写方法的边界情况。

**📈 对比分析**

评估方法：对比在未使用任何优化（Base）与仅使用重写、仅使用限定符、仅使用主体提取、以及全部组合（All）时的验证耗时；采用 10 次实验取平均值，并记录成功、失败与超时情况；计算加速比。结果显示：
- 在 CLBlast 级别 2 内核中，All 组合可达 150 倍加速；
- 整体平均加速约 9 倍；
- 对 Radio Telescope Pipeline 的验证时间缩短 1.9 倍，成功验证了此前不可验证的情况。

**⚠️ 局限性**

限制与挑战：
- 重写方法目前不支持包含模运算等非线性索引的量化子句（如 gemm_3 内核）；
- 对于极大规模非线性约束，SMT 求解器仍可能失效，需要手工添加额外推理；
- 需要程序员手动添加 immutable/unique 注解，虽然类型检查可自动验证但仍增加了编程负担；
- 当前实现只适用于特定的可推理框架，迁移到其他工具仍需额外工作。

---

## 621. Design of Magnetic Continuum Robots with Tunable Force Response Using Rotational Ring Pairs

**arXiv ID:** 2605.13613 | [PDF](https://arxiv.org/pdf/2605.13613v1)

**作者:** Alex Sayres `[一作]` (Worcester Polytechnic Institute), Giovanni Pittiglio `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 987 | [OpenAlex ID](https://openalex.org/A5049921916)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了利用可旋转磁环对磁连续机器人尖端磁响应进行在线调节的机器人，实现了在固定外磁场下的自由尖端偏转；

**💡 创新点**

创新点在于通过内部磁环的相对旋转实现磁力/力矩的可调，从而在不改变外磁场的情况下获得多维操控自由度；

**🔧 技术方法**

采用改进的Euler‑Bernoulli梁模型、磁偶极子力学公式与实验数据校准相结合的建模方法，并使用机械旋转驱动实现磁环角度控制；

**📊 数据集**

使用自制的磁性连续机器人实验数据（在强度为N52的圆柱磁体产生的非均匀磁场中进行的2D与3D尖端位移与磁环角度测量，包含16组取向与对应位移记录）作为数据集；

**📈 对比分析**

通过与实验测得的尖端位置对比，模型平均绝对误差为1.86 mm（1.24 %长度），最大误差为4.8 mm（3.2 %长度），验证了模型在小偏转范围内的良好预测性能；

**⚠️ 局限性**

局限在于假设磁环具有无限扭转刚度且为纯偶极子，忽略了内部磁相互作用与扭转柔性；实验环境为非均匀磁场，导致校准系数不具普适性，且仅在小偏转（≤23 %长度）内有效。

---

## 622. Deep Learning as Neural Low-Degree Filtering: A Spectral Theory of Hierarchical Feature Learning

**arXiv ID:** 2605.13612 | [PDF](https://arxiv.org/pdf/2605.13612v1)

**作者:** Yatin Dandi `[一作]` (École Polytechnique Fédérale de Lausanne), Florent Krzakala `[通讯]` (École Polytechnique Fédérale de Lausanne)

**通讯引用:** 8950 | [OpenAlex ID](https://openalex.org/A5068236230)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Neural LoFi（Neural Low‑Degree Filtering）框架，将深度网络的梯度训练近似为每层基于标签加权的低阶矩阵特征分解与非线性随机映射的迭代谱滤波过程，并在全连接与卷积网络上进行实验验证。

**💡 创新点**

创新点包括：① 将梯度学习的动态化简为可解析的低阶谱滤波；② 通过可变分问题揭示特征选择的相关性–复杂度权衡；③ 给出概念出现阈值的有效维度判据；④ 构造任务自适应的多层核空间，实现深度特征的逐层自适应构造。

**🔧 技术方法**

采用的技术主要有：梯度下降小初始化近似、标签加权矩阵 C^(ℓ) 的特征分解、低阶矩/二阶相关性、变分优化、RKHS 与核方法、随机特征映射、统计学习理论（有效维度、Rademacher 复杂度）等。

**📊 数据集**

使用的数据集包括 CIFAR‑10（动物 vs 车辆的二分类任务）、可解的两层合成目标模型（Gaussian + Hermite 结构）以及卷积网络实验中对图像像素的局部特征提取。

**📈 对比分析**

与岭回归、随机特征基线以及梯度下降（全反向传播）进行对比；在低样本量或训练初期阶段，Neural LoFi 的测试误差可与或优于梯度下降；实验显示它能够恢复常见的边缘/对比滤波器，并且预测的概念出现阈值与实际特征重叠曲线高度吻合，验证了理论预测的准确性。

**⚠️ 局限性**

局限性包括：仅在梯度下降初期/小步长近似下成立；缺乏对高阶特征、长期训练动态和结构化架构（如注意力网络）的完整理论；目前仅作为理论手段/诊断工具，尚未成为高效的实际训练方法；对大规模实战网络的性能验证尚不足。

---

## 623. Rethinking Graph Convolution for 2D-to-3D Hand Pose Lifting

**arXiv ID:** 2605.13604 | [PDF](https://arxiv.org/pdf/2605.13604v1)

**作者:** Chanyoung Kim `[一作]` (Emory University), Youngjoong Kwon `[通讯]` (Emory University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在 2D‑to‑3D 手势姿态提升任务中，对比并系统评估了传统图卷积网络（GCN）与自注意力（Attention）模块的空间建模效果。

**💡 创新点**

创新点在于：①证明了即使在参数和感受野匹配的条件下，自注意力仍显著优于 GCN；②将手部拓扑信息以图距离位置编码的“软先验”方式融入注意力模型，而非硬性邻接限制；③通过多种 ablation 展示了输入依赖聚合和全连接注意力的额外收益。

**🔧 技术方法**

核心技术包括：线性关节嵌入、图距离位置编码、4 层多头自注意力块、Per‑Joint MLP 回归头、以及针对 FPHA 的 oracle 2D 输入策略。

**📊 数据集**

实验使用 FPHA（100K+ RGB‑D 采样，45 种手势，21 关节标注）的交叉主体分割。

**📈 对比分析**

与 GCN 及其多跳、GAT 变体对比，自注意力实现 MPJPE 10.09 mm、AUC 80.0%，比 1‑跳 GCN（13.82 mm）和多跳 GCN（12.36 mm）分别提升约 2.73 mm 与 2.27 mm；在噪声鲁棒性实验中，自注意力在 10 px 2D 噪声下仍保持 20% 的优势。

**⚠️ 局限性**

局限性：仅基于 oracle 2D 关键点，未评估端到端检测与提升的耦合；缺少时序建模与不同视角/模态的广泛验证；模型对极端遮挡或非典型手势的泛化能力尚未充分探讨。

---

## 624. How to Interpret Agent Behavior

**arXiv ID:** 2605.13625 | [PDF](https://arxiv.org/pdf/2605.13625v1)

**作者:** Jie Gao `[一作]` (Johns Hopkins University), Mark Dredze `[通讯]` (Johns Hopkins University)

**通讯引用:** 23814 | [OpenAlex ID](https://openalex.org/A5024437840)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对自律智能体运行时行为的三层分类法（Act‑onomy）并实现自动化分析流水线

**💡 创新点**

首创结构化、可扩展的行动词汇表，并通过 GitHub 开源提供“生活化”分类体系与自动分析工具

**🔧 技术方法**

采用 Grounded Theory 定性编码结合 LLM 自动化分类器，实现对轨迹的行为标注

**📊 数据集**

数据来源于 2024‑2026 年 35 篇顶级论文的 565 条行为描述，以及 211 篇相关论文共 3455 条描述

**📈 对比分析**

通过跨智能体行为分布与单智能体轨迹的可视化对比，验证工具与人工标注一致性（Cohen κ>0.81），并能揭示异常模式

**⚠️ 局限性**

局限在对执行类行为识别不足、需手动扩展新工具/场景、缺乏与内部模型状态的直接对应

---

## 625. Force-Aware Neural Tangent Kernels for Scalable and Robust Active Learning of MLIPs

**arXiv ID:** 2605.13788 | [PDF](https://arxiv.org/pdf/2605.13788v1)

**作者:** Eszter Varga-Umbrich `[一作]` (InstaDeep), Shikha Surana `[通讯]` (InstaDeep)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种可扩展的离线主动学习框架，针对机器学习原子势能模型（MLIP）在大规模候选集上的标注与分布偏移问题进行解决。

**💡 创新点**

创新点包括：①引入基于特征空间的后验方差（PV）短列表策略，实现候选集规模线性可扩展；②将神经切线核（NTK）扩展到力场预测，构造力NTK和能量-力联合NTK；③通过实验验证联合NTK在多种分布下的鲁棒性与高效性。

**🔧 技术方法**

主要技术：特征空间后验方差短列表、LCMD多样性筛选、混合参数-坐标导数构造的力NTK与联合NTK、基于预训练MACE模型的能量与力前向/后向计算。

**📊 数据集**

使用的数据集包括：OC20（催化剂），T1x、PMechDB、RGD（反应性分子基准），以及通过MPTraj预训练的MACE模型。

**📈 对比分析**

与随机、SOAP、Morgan指纹、激活特征、委员会不确定性等基线对比，联合NTK在OC20的所有内外部分布下均取得最低能量与力MAE/RMSE；在反应性基准上能量NTK和联合NTK表现最优；委员会方法表现最差。

**⚠️ 局限性**

局限性：①在百万级候选集仍受特征维度限制，计算成本高；②力NTK对力标签噪声敏感，需更鲁棒的正则化；③缺乏对更复杂监督（应力、Hessian等）的扩展，且方法对参数子集的选择影响尚待深入研究。

---

## 626. Attention Once Is All You Need: Efficient Streaming Inference with Stateful Transformers

**arXiv ID:** 2605.13784 | [PDF](https://arxiv.org/pdf/2605.13784v1)

**作者:** Victor Norgren `[一作]` `[通讯]` (LayerScale), Victor Norgren (LayerScale)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出基于状态化KV缓存的流式推理框架，使数据推送在后台持续预填充，查询仅在缓存上进行常量时间计算，提升了实时性。

**💡 创新点**

创新点包括：将数据驱动与查询驱动分离、持续利用GPU空闲周期进行Flash Query预计算、可持续的GPU利用率、多租户连续批处理与优先级调度、完整二次注意力在长上下文下保持常量查询延迟。

**🔧 技术方法**

采用锁自由环形缓冲区、状态化KV缓存、优先级GPU调度器、Flash Attention、Radix/前缀缓存、完整二次注意力、GPU内设备argmax、延迟预测与闪存查询等技术。

**📊 数据集**

使用合成OHLCV时序数据（每条记录约16个token）以及六类金融分析查询（趋势、回撤、最高价、盘整、成交量、精确取值）。

**📈 对比分析**

在同一Meta‑Llama‑3.1‑8B模型上与vLLM、SGLang、TensorRT‑LLM、llama.cpp以及OpenAI GPT‑5.2/4o‑mini、Claude 3.5 Haiku/Opus 4.5进行对比，流式查询平均延迟约43 ms，远低于传统请求驱动引擎（106–254 ms）且比云API快21–92倍，方差也最低。

**⚠️ 局限性**

限制包括：需为每个会话保留完整KV缓存，导致显存占用较大；仅支持单序列、因果注意；滑动窗口内数据可见，历史信息需压缩或丢失；对多文档检索、双向模型或跨任务场景需进一步扩展。

---

## 627. LMPath: Language-Mediated Priors and Path Generation for Aerial Exploration

**arXiv ID:** 2605.13782 | [PDF](https://arxiv.org/pdf/2605.13782v1)

**作者:** Jonathan A. Diller `[一作]` (University of Pennsylvania), Vijay Kumar `[通讯]` (University of Pennsylvania)

**通讯引用:** 39875 | [OpenAlex ID](https://openalex.org/A5087021192)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出LMPath管线，利用大语言模型与卫星影像的视觉基础模型生成语义探索先验，以此优化无人机搜索路径。

**💡 创新点**

创新点在于将LLM用于推理目标相关语义标签并通过基础分割模型生成热图，再结合ILP多目标路径规划，实现语义驱动的全局搜索优化。

**🔧 技术方法**

技术包括GPT-4o-mini LLM、SAM 3 语义分割模型、滑动窗口掩码拼接、Voronoi网格、整数线性规划和TSP算法。

**📊 数据集**

使用高分辨率卫星图像（Mapbox/ESRI/Google），以及PolyCity合成城市、工业园实际三维网格等实验环境。

**📈 对比分析**

与传统TSP覆盖规划对比，LMPath在PolyCity和工业园分别比基线减少搜索时间66%和88%，在实地Falcon 4飞行验证中同样表现优异。

**⚠️ 局限性**

局限性包括只能识别在卫星图中可见的目标，遮挡导致分割不完整；热图权重统一，未考虑标签置信度和条件依赖。

---

## 628. Elastica++: A high-performance, multiphysics framework for large interacting assemblies of Cosserat rods

**arXiv ID:** 2605.13766 | [PDF](https://arxiv.org/pdf/2605.13766v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 629. MinT: Managed Infrastructure for Training and Serving Millions of LLMs

**arXiv ID:** 2605.13779 | [PDF](https://arxiv.org/pdf/2605.13779v1)

**作者:** Mind Lab `[一作]` (Mind Lab), Murphy Zhuang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了MindLab Toolkit (MinT)，将LoRA适配器作为策略单元，实现训练、评估、服务与回滚的统一管理。

**💡 创新点**

创新点在于：①仅传递LoRA适配器修订版而非完整模型快照，显著降低训练‑服务交互开销；②时间切片多策略训练，使多个策略共享单一驻留基础模型；③三层缓存与打包MoE LoRA，提升大规模策略库的服务效率。

**🔧 技术方法**

采用LoRA/PEFT、Megatron分布式训练、vLLM推理、Tinker兼容API、Mint Cookbook等技术。

**📊 数据集**

使用的主要数据集包括FinGPT金融评测集、FinEval、FPB、FiQA‑SA、TFNS、NWGI、chat‑DPO对、DAPO‑AIME24、Qwen3系列模型、LawBench、AIME24、1T计时任务等。

**📈 对比分析**

与传统全检查点合并路径对比，Adapter‑only handoff 在 4B 模型上缩短 18.3×、在 30B 模型上缩短 2.85×；多策略并行训练使 4B Wall‑time 下降 1.77×、30B 下降 1.45×；学习曲线表明同一生命周期可支持 SFT、DPO 与 GRPO，MoE RL 在 30B/235B 模型上均可达 AIME24 mean@1 0.97。

**⚠️ 局限性**

局限性包括：仍需预驻留大型基础模型；主要聚焦 LoRA 方案，其他高效微调方法不在覆盖范围；对极大多租户冷加载吞吐和极高 rank LoRA 的评估仍有限；系统复杂度高，需进一步验证在更大集群及多租户环境下的可行性。

---

## 630. Upper Bounds for Symmetric Approximate Bounded Indistinguishability

**arXiv ID:** 2605.13771 | [PDF](https://arxiv.org/pdf/2605.13771v1)

**作者:** Christopher Williamson `[一作]` `[通讯]`, Christopher Williamson

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

对称分布的(k,δ)-wise indistinguishability进行研究，给出观测t位时统计距离的上界；

**💡 创新点**

通过超几何平滑与汉恩多项式相结合，得到在更宽参数范围内的指数衰减上界，消除先前存在的常数区间和指数差距；

**🔧 技术方法**

利用超几何平滑算子、汉恩多项式正交基以及逼近理论与阶乘运算等数学工具；

**📊 数据集**

本研究为理论工作，未使用任何实验数据集；

**📈 对比分析**

与先前的O(k^{3/2} e^{-k^2/1156t})等上界相比，新上界在k=Ω(n)时降至e^{-(n-t)}，在t≈n时实现指数级接近；在各参数区间均优于现有结果；

**⚠️ 局限性**

仍需对称性假设；在极端参数下可能不是最优；对非对称分布的推广尚未完成。

---

## 631. (How) Do Large Language Models Understand High-Level Message Sequence Charts?

**arXiv ID:** 2605.13773 | [PDF](https://arxiv.org/pdf/2605.13773v1)

**作者:** Mohammad Reza Mousavi `[一作]` (King's College London), Mohammad Reza Mousavi `[通讯]` (King's College London)

**通讯引用:** 2408 | [OpenAlex ID](https://openalex.org/A5101493818)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究通过设计129个基于高层消息序列图(HMSC)的语义任务，系统评估了Gemini‑3‑Pro、GPT‑5.4 与 Qwen‑3.6 三个大型语言模型在语义理解、变换和行为推理方面的表现，揭示了它们在抽象、组合以及并发语义上的不足；

**💡 创新点**

创新点在于提出了专门针对 HMSC 的语义基准与任务集，采用可复现的实验流程和可视化评估手段，首次量化 LLM 在高层软件架构图中的语义推理能力，并定位了模型的痛点；

**🔧 技术方法**

使用了三种前沿 LLM 进行零/少量提示实验，并结合自动生成 Python 代码执行推理以辅助评估；

**📊 数据集**

数据集为 9 个来自 TU/Eindhoven 课程的 MSC/HMSC 例子，已公开在复现包中；

**📈 对比分析**

通过将任务划分为事件识别、事件顺序、抽象、组合、轨迹与 LTS 等六类，对每类计算准确率；整体准确率约 52%，基本语义 88%，抽象仅 20%，组合约 47%，轨迹 58%，LTS 仅 25%；各模型性能相近，Gemini‑3 在整体与组合类任务略优；

**⚠️ 局限性**

局限性包括：样本量有限、可能存在训练数据泄露、评估过程对图形视觉解读有偏差、缺乏自动化语义检查、对并发与部分序列的推理能力不足、LLM 内部推理机制缺失导致错误。

---

## 632. Manipulation Planning for Construction Activities with Repetitive Tasks

**arXiv ID:** 2605.13754 | [PDF](https://arxiv.org/pdf/2605.13754v1)

**作者:** Wangyi Liu `[一作]` (Stony Brook University), Nilanjan Chakraborty `[通讯]` (Stony Brook University)

**通讯引用:** 9204 | [OpenAlex ID](https://openalex.org/A5011964617)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在VR环境收集单步演示，将对象轨迹分解为常数螺旋运动，从而实现从单一演示泛化至任意长度的建造任务。

**💡 创新点**

创新在于利用VR演示结合螺旋运动理论，提取任务约束并以参数化方式重用，达成极少数据下的任意长度建造。

**🔧 技术方法**

采用VR收集、SE(3)常数螺旋分段、ScLERP插值、RMRC/雅可比伪逆、SEW约束模式的运动规划，配合Franka Emika Panda 7-DoF机器人。

**📊 数据集**

使用在Unreal Engine 4构建的VR场景中记录的砖墙与天花板瓷砖安装演示，随后在PyBullet仿真与真实Frank Emika Panda机器人上进行实验。

**📈 对比分析**

与基线方法（<cit.>）在112个仿真任务与6个真实任务中对比，砖墙任务成功放置砖块累计303个比260个，瓷砖安装成功率15/16比0/16；在实际操作中全部成功。

**⚠️ 局限性**

局限在于仅验证平面或简单曲线/角墙、固定或小范围移动基座，无法处理复杂非平面结构、动态障碍、需要机器人底盘运动或多机器人协同；且对极端尺寸或复杂约束的适应性未深入。

---

## 633. Learning Responsibility-Attributed Adversarial Scenarios for Testing Autonomous Vehicles

**arXiv ID:** 2605.13751 | [PDF](https://arxiv.org/pdf/2605.13751v1)

**作者:** Yizhuo Xiao `[一作]` (Heriot Watt University), Cheng Wang `[通讯]` (Heriot Watt University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6215c339-3735-4be3-8a07-5bbb7004712d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发并评估了CARS框架，结合上下文感知的对手选择、多模态生成和责任归因，用于安全验证自主驾驶系统；

**💡 创新点**

将责任归因标准（CCD）直接嵌入对抗场景生成，采用在线对手选择、Gaussian‑Mixture Diffusion生成策略和闭环强化学习，生成可归因、可解释的碰撞场景，突破传统仅优化碰撞率的方法；

**🔧 技术方法**

使用梯度提升分类器进行在线对手选择，Gaussian‑Mixture Diffusion模型生成动作序列，PPO强化学习调优；采用FSM、CC‑JP、RSS三种参考模型进行责任归因，并在闭环仿真中进行物理可行性检测；

**📊 数据集**

使用nuScenes城市驾驶数据进行训练，随后在AD4CHE高速和RounD环形交叉口数据上无重新训练进行跨域泛化评估；

**📈 对比分析**

与STRIVE、SafeSim和Bezier‑CAT等现有对抗生成器在相同责任归因管道下对比，CARS在FSM归因率88.7%（约两倍最强基线），严重性多样性最高，失效率最低；在不同规划器、跨域测试中依然保持高归因率和低不可行率；

**⚠️ 局限性**

仅针对纵向冲突，侧向或多对手攻击未覆盖；仅针对车辆缺乏行人、骑行者等易受伤害主体；依赖CCD参考模型的纵向假设，需进一步扩展多模态与更广泛的责任标准。

---

## 634. SPLIT: SymPathy for Large jobs Improves Tail latency

**arXiv ID:** 2605.13749 | [PDF](https://arxiv.org/pdf/2605.13749v1)

**作者:** Zhouzi Li `[一作]` (Carnegie Mellon University), Alan Scheller-Wolf `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4625 | [OpenAlex ID](https://openalex.org/A5016706070)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了在 M/G/n 负载均衡系统中对重尾作业大小实现强尾最优调度的第一套完整方案，涵盖已知与未知作业大小两种情形。

**💡 创新点**

核心创新在于发现多服务器系统需要对大型作业给予“同情”——即专门为大型作业预留服务器，而非单纯优先短作业；通过引入 SymPathy、d、dTAGS 三种策略，在不同负载区间内实现强尾最优，弥补了单服务器调度理论在多服务器场景的空白。

**🔧 技术方法**

技术手段包括：正则变动（Pareto）作业大小模型、尾最优定义与下界推导、超级服务器（super‑server）下界、标记作业（tagged job）分析、工作量（relevant work）潜在函数、LJF 与 SRPT 结合的分组调度、阈值分流策略 d、TAGS 近似以及 Poisson 入队假设下的尾概率逼近。

**📊 数据集**

实验使用合成 Pareto 作业大小分布（尾指数 α=1.4、1.5、2.0），并在 n=2、3、10、负载 ρ=0.5、0.8、0.94 等多种配置下进行 10^9–10^10 次作业模拟。

**📈 对比分析**

与 FCFS、SRPT‑n、SEK‑ε 等基线对比，采用标准化尾概率（T>t 与下界比值）衡量。新策略在 ρ < n−1/n 区域收敛到 1，证明强尾最优；在高负载下通过阈值 d 的调节也能逼近下界；相比之下，基线始终保持在 1 之外；在 99% 级别偶有折衷，但在 99.9%–99.99% 级别表现明显优于 SRPT‑n。

**⚠️ 局限性**

局限性包括：仅针对重尾作业，轻尾多服务器尾最优仍未完全解决；理论分析假设 Poisson 到达与 TAGS 近似，真实工作负载可能偏离；对均衡与极端负载的实验有限；未同时评估均值响应时间与尾部性能的折中。

---

## 635. GHGbench: A Unified Multi-Entity, Multi-Task Benchmark for Carbon Emission Prediction

**arXiv ID:** 2605.13743 | [PDF](https://arxiv.org/pdf/2605.13743v1)

**作者:** Yifan Duan `[一作]` (University of New South Wales), Flora Salim `[通讯]` (University of New South Wales)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建并公开了名为 GHGbench 的统一碳排放预测基准，涵盖公司级和建筑级两种实体，提供标准化数据拆分、指标体系以及配对自举不确定性评估。

**💡 创新点**

创新点包括：①首个同时覆盖公司与建筑的开放基准；②设计跨实体、跨地区、跨时间、跨属性类型的多任务评估框架；③引入遥感多模态特征并验证其在跨城迁移中的提升；④采用严格的配对自举统计检验，确保结果稳健且可复现。

**🔧 技术方法**

使用的技术：梯度提升树（LightGBM、XGBoost、RandomForest）、预训练表格基础模型 TabPFN、深度 MLP、FT-Transformer、时间序列基础模型（Chronos、TimesFM、Moirai）、LLM（Claude、GPT、Qwen、Mistral），以及 Sentinel-2 + Clay 视觉嵌入与 NASA POWER 气候变量。

**📊 数据集**

数据集：公司级基于 Climate Data Utility（CDU）公开披露的 32,000+ 年度记录，并加入金融/行业因子；建筑级汇总自 13 个公开城市来源（美国、澳大利亚、新加坡）的 491,591 年度建筑记录，统一 schema 并补齐气候和遥感特征。

**📈 对比分析**

比较方法：三重种子、1000 次配对自举 CI，评估指标包括 R²、MAE/LogMAE、MAPE 等。实验结果显示：公司级最高 R²≈0.88；建筑级 TabPFN 在未见建筑下 R²≈0.48，树模型显著落后；跨城迁移将 R² 降至 0.03–0.13；加入 Sentinel‑2 嵌入可在跨城转移中提升约 0.04–0.07。

**⚠️ 局限性**

局限性：①建筑级难度高，缺乏占用、运行时等细粒度信息；②跨城、跨属性迁移仍表现差劲；③遥感特征提升有限，模型对多模态融合能力不足；④时间序列覆盖短，未来需更长的面板数据与更强的 LLM/多模态框架。

---

## 636. AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation

**arXiv ID:** 2605.13724 | [PDF](https://arxiv.org/pdf/2605.13724v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 637. KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving

**arXiv ID:** 2605.13734 | [PDF](https://arxiv.org/pdf/2605.13734v1)

**作者:** Zedong Liu `[一作]` (University of Chinese Academy of Sciences), Guangming Tan `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 KVServe，集成模块化 KV 压缩管线、贝叶斯优化搜索与在线自适应控制，用于分离式和 KV 离散 LLM 服务场景，提高 KV 传输效率。

**💡 创新点**

将 KV 压缩视为服务状态下的组合优化问题，构建可扩展模块化管线、使用贝叶斯优化高效定位 Pareto 前沿，并通过解析模型与轻量 bandit 实现在线自适应选择。

**🔧 技术方法**

模块化压缩管线（Transformer → Quantizer → Codec）、贝叶斯优化（Gaussian Process）、解析延迟模型、ε‑greedy bandit 以及 vLLM 集成。

**📊 数据集**

GSM8K、HumanEval、Multi‑News、Qasper、2WikiMQA、HotpotQA 等数据集，用于离线搜索与在线评估。

**📈 对比分析**

与 CacheGen、KIVI、DuoAttention 等基线比较，KVServe 在 PD‑分离和 KV‑离散场景下分别实现最高 9.13× 的 JCT 加速和 32.8× 的 TTFT 缩短，同时保持 97% 以上的相对准确度。

**⚠️ 局限性**

仍依赖离线模型与网络环境，未覆盖更大规模模型或多租户动态调度，且对实时 SLO 漂移仍需在线学习补偿。

---

## 638. Distinguishing performance gains from learning when using generative AI

**arXiv ID:** 2605.13731 | [PDF](https://arxiv.org/pdf/2605.13731v1)

**作者:** Lixiang Yan `[一作]` (Monash University), Dragan Gašević `[通讯]` (Monash University)

**通讯引用:** 31870 | [OpenAlex ID](https://openalex.org/A5036855560)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨在教育中使用生成式AI时区分表现提升与真正学习的区别，并提出相应的研究与实践框架

**💡 创新点**

创新点在于系统阐述表现与学习的根本差异，强调认知负荷、元认知和自我效能的影响，并提出针对性的研究议程与教师实践建议

**🔧 技术方法**

运用了认知心理学理论（认知负荷理论、元认知理论、自我决定理论）进行概念框架与分析

**📊 数据集**

未使用实验数据集，主要基于已有文献综述、meta‑分析与案例研究

**📈 对比分析**

通过对现有meta‑分析（Hedge’s g=0.7）与实验研究的比较，指出生成式AI在短期任务表现上有明显提升，但长期学习效果尚不确定

**⚠️ 局限性**

局限在于缺乏实证实验与纵向跟踪研究，难以确定生成式AI对学习的持续影响，也未提供具体技术实现方案

---

## 639. ScioMind: Cognitively Grounded Multi-Agent Social Simulation with Anchoring-Based Belief Dynamics and Dynamic Profiles

**arXiv ID:** 2605.13725 | [PDF](https://arxiv.org/pdf/2605.13725v1)

**作者:** Yitian Yang `[一作]` (University of Sydney), Huaming Chen `[通讯]` (University of Sydney)

**通讯引用:** 20554 | [OpenAlex ID](https://openalex.org/A5086004140)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并评估了一个基于大语言模型的认知驱动多代理社会仿真框架（记忆锚定信念更新机制），用于研究社会观点动态与群体行为。

**💡 创新点**

创新点在于：① 将记忆锚定（anchor）与个体人格映射相结合的信念更新规则；② 设计四层认知记忆架构（情节、语义、程序、反思）以支持经验驱动的锚形成；③ 采用语料库检索与人格先验动态生成多样化代理档案；④ 引入人机交互校准机制调节锚定强度与置信阈值。

**🔧 技术方法**

核心技术包括：大语言模型推理（OpenAI API）、RAG检索与FAISS向量检索、基于Big Five人格的参数映射、边际自适应置信阈值、分层记忆与反思回路、动态代理档案生成与关系网络构建。

**📊 数据集**

使用公开的社交媒体数据集（如 Roe v. Wade Twitter 数据集）、澳大利亚社交媒体禁令相关数据、以及自构建的多议题政策语料库；并通过人工标注的情绪与立场标签对代理行为进行对齐。

**📈 对比分析**

通过与基线 DeGroot‑style 及无锚定版本进行对照实验，评估极化、分歧、极端化、聚类多样性等指标。实验结果显示：加入锚定后极化提升至约0.35、极端化上升至0.60、分歧保持稳定；在 Roe v. Wade 案例中与真实推特数据的极化与情绪分布高度匹配。

**⚠️ 局限性**

局限性包括：1) 受安全约束，无法真实模拟仇恨言论与激进升级；2) 代理多样性仍依赖人工设定的人格与语料，缺乏真实人类长期社会化与文化背景的细节；3) 对极端语境下的对齐与解释性不够完善；4) 需要更大规模、更长时序的实证数据来进一步校准与验证。

---

## 640. DisAgg: Distributed Aggregators for Efficient Secure Aggregation in Federated Learning

**arXiv ID:** 2605.13708 | [PDF](https://arxiv.org/pdf/2605.13708v1)

**作者:** Haaris Mehmood `[一作]` (Samsung R&D Institute UK), Mete Ozay `[通讯]` (Samsung R&D Institute UK)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DisAgg 协议，将安全聚合的计算任务分配给少量聚合器，构建一种低轮次、一轮交互的分布式安全聚合框架，用于跨设备联邦学习。

**💡 创新点**

创新点在于：① 采用 Lagrange Coded Computing 进行秘密分享，实现信息论安全并消除本地掩码和恢复机制；② 将聚合操作交给聚合器完成，显著降低普通客户端和服务器的计算负担；③ 维持一轮交互的低同步开销，兼顾大规模、动态参与和客户端下线的鲁棒性。

**🔧 技术方法**

核心技术：Lagrange Coded Computing（LCC）秘密分享；Diffie‑Hellman 公钥加密；有限域（ℤp）量化；单轮（One‑Shot）安全聚合设计；分布式聚合器委员会；信息论安全分析与阈值选择。

**📊 数据集**

在实验中使用 MNIST、CIFAR‑10、CIFAR‑100、SST‑2、CelebA、DistilBERT、EfficientNet 等数据集，通过 FedAvg 进行多轮训练以验证聚合效率与模型精度。

**📈 对比分析**

方法：与 SecAgg、SecAgg+、LightSecAgg 和 OPA 在不同模型尺寸（1k–100k 参数）和参与人数（1k–100k）下进行端到端时间比较；在 5G 链路条件下，DisAgg 在设置、客户端和服务器阶段分别实现 3–4 倍加速，整体约 4.56 倍速度提升；在多轮训练中保持与明文聚合相同的准确率。

**⚠️ 局限性**

局限性：聚合器下载量较大，尤其在高维模型与大规模客户端时需要显著网络带宽；聚合器数量和打包因子需仔细调优，平衡通信与速度；协议假设可信随机源，无法完全抵御恶意服务器；对极端下行延迟或极低速客户端的适配仍需进一步优化。

---

## 641. Realtime-VLA FLASH: Speculative Inference Framework for Diffusion-based VLAs

**arXiv ID:** 2605.13778 | [PDF](https://arxiv.org/pdf/2605.13778v1)

**作者:** Jiahui Niu `[一作]` (Chinese Academy of Sciences), Huawei Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 36943 | [OpenAlex ID](https://openalex.org/A5100346092)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为Realtime‑VLA FLASH的推理框架，利用轻量级草稿模型与主模型的并行验证，实现对大多数重规划环节的短时推理，从而显著降低视觉‑语言‑动作模型（dVLA）的实时推理延迟。

**💡 创新点**

创新点包括：① 将流匹配（flow‑matching）结构用于连续动作的草稿验证；② 在主模型的动作专家（Action Expert）上并行执行多步验证，快速判定可执行前缀；③ 通过抓取器状态切换检测实现阶段感知回退，防止细调阶段误差累积。

**🔧 技术方法**

技术要点包括：轻量级草稿模型（单个Gemma块+线性动作头）、流匹配的中间状态插值与端点重建、基于距离阈值的可执行前缀选择、以及抓取器切换阈值的阶段感知回退。

**📊 数据集**

使用的实验数据集：LIBERO四个子任务套件（Spatial、Object、Goal、10）和真实世界的输送带分类任务（玩具狗、毛刷）。

**📈 对比分析**

与Torch‑π₀、Triton‑π₀、FLASH‑π₀对比，FLASH+Triton‑π₀在LIBERO上实现了平均任务级推理延迟从58 ms降至19.1 ms（约3×加速），动作吞吐量提升2.63×，任务成功率仅下降0.3个百分点；在输送带任务中，FLASH+Triton‑π₀在最高15 m/min速度下仍保持50%成功率，而其他方法在此速度下全部失败。

**⚠️ 局限性**

局限性：使用的是固定阈值的启发式验证与回退策略，缺乏自适应阈值；在精细调节阶段仍可能出现误差积累导致失败；草稿模型的性能受限于其较小规模，进一步改进可进一步提升整体效果。

---

## 642. RoboEvolve: Co-Evolving Planner-Simulator for Robotic Manipulation with Limited Data

**arXiv ID:** 2605.13775 | [PDF](https://arxiv.org/pdf/2605.13775v1)

**作者:** Harold Haodong Chen `[一作]` (Hong Kong University of Science and Technology), Ying-Cong Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 2748 | [OpenAlex ID](https://openalex.org/A5101938761)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种自演化框架，将视觉‑语言模型（VLM）规划器与视频生成模型（VGM）模拟器通过日夜循环相互进化，实现在仅有的无标签种子图像下获取任务对齐的交互数据；

**💡 创新点**

创新点在于将VLM的语义规划与VGM的物理仿真通过互补学习系统（CLS）原理的昼夜双阶段循环耦合，既利用VLM生成多样化计划，又用VGM对计划进行物理验证与失败挖掘，并通过层次化偏好优化提升两者协同；

**🔧 技术方法**

核心技术包括：场景语义解析与结构化任务初始化、基于多粒度奖励的日间强化学习（GRPO）、夜间偏好优化（DPO）、自洽投票机制、层级化偏好训练、以及基于难度的自适应课程学习；

**📊 数据集**

使用的数据集为BridgeData V2（用于VGM评估）、EB‑ALFRED 与 EB‑Habitat（用于VLM评估），仅以300–500张无标签种子图像作为起点；

**📈 对比分析**

与传统静态或单一模型基线相比，本文方法在BridgeData V2任务成功率上提升约40–56%，在EB‑ALFRED/EB‑Habitat的规划成功率上平均提升36–55点，且仅需500张无标签图像即可超过全监督50×的数据量；

**⚠️ 局限性**

局限性包括对种子图像质量和多样性的依赖、夜间偏好抽样的计算成本以及在极大复杂任务上对奖励设计与模型容量的敏感性。

---

## 643. Chrono::Ray: A Distributed Framework for High-Throughput Simulation-Based Analysis of Multibody Systems

**arXiv ID:** 2605.13767 | [PDF](https://arxiv.org/pdf/2605.13767v1)

**作者:** Khailanii Slaton `[一作]` (University of Wisconsin-Madison), Dan Negrut `[通讯]` (University of Wisconsin-Madison)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了 Chrono::Ray，一个将 Project Chrono 多体动力学引擎与 Ray 分布式计算平台集成的分布式仿真框架，支持参数估计、优化和实验设计等工作流。

**💡 创新点**

创新点在于：1）提供统一、模块化的工作流接口，隐藏 Ray 的任务调度与资源管理细节；2）让用户仅需定义仿真函数和参数空间，无需手动编写并行/分布式代码；3）支持多种搜索算法（随机、贝叶斯、CMA-ES 等）和实验设计（拉丁超立方、Sobol 等），极大简化高通量仿真流程。

**🔧 技术方法**

技术：Python；PyChrono（Chrono 的 Python 绑定）；Ray（包括 Ray Core 与 Ray Tune）；SciPy/NumPy/Matplotlib 用于结果可视化；Bash/Blender 用于后处理。

**📊 数据集**

实验数据集：①多体月球着陆器模型（含蜂巢能量吸收模型），用于参数估计；②连续土壤力学模型（Modified Cam‑Clay rheology），用于实验设计。参数空间分别包括 3 个（β、α₂、f_y）和 6 个（μ_s、ρ、λ、κ、E、ν）自由度。

**📈 对比分析**

方法比较：在参数估计例子中，Chrono::Ray 以 50 次试验完成最优参数搜索，RMSE 下降到 3.53，计算时间约 28.66 秒；在实验设计例子中，使用 Latin Hypercube 生成 20 组参数，最多 2 个并发试验，成功生成并行仿真输出。虽然本文未与传统单机串行仿真直接对比，但示例展示了在多核/集群环境下显著缩短的壁钟时间和高效的资源利用。

**⚠️ 局限性**

局限性：①框架目前主要支持离散工作流（参数估计、优化、实验设计），对 UQ、灵敏度分析等高级功能仍需扩展；②需要用户自行实现仿真函数和参数空间映射，对非 Python 环境或复杂并行模型适配仍有一定门槛；③性能依赖 Ray 集群配置，极大规模下可能出现任务调度瓶颈。

---

## 644. Humanwashing -- It Should Leave You Feeling Dirty

**arXiv ID:** 2605.13723 | [PDF](https://arxiv.org/pdf/2605.13723v1)

**作者:** Ben Wilson `[一作]` (Swansea University), Matt Roach `[通讯]` (Swansea University)

**通讯引用:** 288 | [OpenAlex ID](https://openalex.org/A5051787059)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过理论分析和文献综述，批判性评估了“人机在环”概念的使用，揭示其隐含的“人洗涤”问题，并呼吁建立更精准、多维度的人机组合空间研究框架。

**💡 创新点**

创新点在于指出循环隐喻对人机协作理解的误导，并提出对人机组合空间维度（如控制、信息、交互模式、上下文依赖等）的系统化认知与评估方法。

**🔧 技术方法**

主要采用理论阐释、案例剖析和对比分析技术，结合监管文本与学术讨论来支持论点。

**📊 数据集**

未使用特定实验数据集，主要引用欧洲AI法案、GDPR等法规及相关学术研究。

**📈 对比分析**

本文未进行实验比较，而是通过案例分析与概念梳理阐明现行做法的局限性，未给出量化性能指标。

**⚠️ 局限性**

局限在于缺乏实证验证与量化评估，主要基于已有文献的解释，未能提供可操作的技术实现细节或实际部署案例。

---

## 645. SkillOps: Managing LLM Agent Skill Libraries as Self-Maintaining Software Ecosystems

**arXiv ID:** 2605.13716 | [PDF](https://arxiv.org/pdf/2605.13716v1)

**作者:** Hongji Pu `[一作]` (University of Illinois Urbana Champaign), Liang Zhao `[通讯]` (Emory University)

**通讯引用:** 7190 | [OpenAlex ID](https://openalex.org/A5061568038)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SkillOps，一个在 LLM 代理中对技能库进行维护的插件框架，能够诊断并修复技能技术债务。

**💡 创新点**

将技能建模为带类型的技能合同，并构造层次化技能生态系统图（HSEG），实现库时健康诊断和基于规则的维护。

**🔧 技术方法**

使用了类型化技能合同、图结构关系（依赖、兼容、冗余、替代）、健康维度评估、合同图传播诊断（CGPD）以及可插拔的维护动作；任务侧采用检索+语义匹配、图规划与本地修复。

**📊 数据集**

在 ALFWorld 家庭操控基准上构建了约 2000 个技能的库，并加入合成技术债务样例进行评估。

**📈 对比分析**

与 ReAct、LLM_Skill_Planner、Hybrid_Retrieval、GoS_Style、SkillWeaver 等基线在 ALFWorld 上比较，SkillOps 作为独立代理取得 79.5% 成功率，超越最强基线约 8.9pp；作为插件提升检索型基线 0.7–2.9pp；且在库规模扩大到 2000 时仍保持约 80% 成功率。

**⚠️ 局限性**

依赖结构化的技能合同与金标准参数，规则化维护对语义冗余不足；评估范围局限于 ALFWorld，缺乏真实长周期日志验证；CGPD 在当前设置下未提升性能。

---

## 646. MILM: Large Language Models for Multimodal Irregular Time Series with Informative Sampling

**arXiv ID:** 2605.13711 | [PDF](https://arxiv.org/pdf/2605.13711v1)

**作者:** Hsing-Huan Chung `[一作]` (University of Texas at Austin), Joydeep Ghosh `[通讯]` (University of Texas at Austin)

**通讯引用:** 23826 | [OpenAlex ID](https://openalex.org/A5103071668)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

在电子健康记录中，针对多模态不规则时间序列进行分类任务，使用LLM实现统一建模。

**💡 创新点**

首次提出将多模态不规则时间序列序列化为XML三元组，并通过两阶段微调使模型能够学习并利用采样模式。

**🔧 技术方法**

采用Qwen3-4B‑Instruct‑2507 LLM，配合QLoRA微调、XML格式输入以及两阶段训练策略。

**📊 数据集**

使用MIMIC‑IV和eICU两大公开EHR数据库构建的ICU住院死亡率和停留时长预测数据集。

**📈 对比分析**

与GRU‑D、mTAND、UTDE、FuseMoE等基线相比，MILM‑2S在八个评价指标上平均排名第一，AU‑ROC/PR指标均显著提升。

**⚠️ 局限性**

局限在于对文本通道的依赖受不同医院数据差异影响，且实验仅覆盖ICU场景，缺乏跨域与分布漂移的鲁棒性评估。

---

## 647. Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety

**arXiv ID:** 2605.13709 | [PDF](https://arxiv.org/pdf/2605.13709v1)

**作者:** Qian Shen `[一作]` (University of Florida), Walter L. Leite `[通讯]` (University of Florida)

**通讯引用:** 3491 | [OpenAlex ID](https://openalex.org/A5078414045)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对三款8B本地LLM进行多种有监督微调，生成符合K-2儿童阅读水平的英文故事，并与GPT-4o、Llama 3.3 70B零样本结果对比。

**💡 创新点**

提出基于奖励加权SFT、精选高质量数据、模拟儿童读写错误等三种微调策略，以在小参数模型上实现可控阅读难度和安全性。

**🔧 技术方法**

采用QLoRA（4-bit量化+LoRA）微调、核采样解码，并用Spache可读性、LM perplexity、连贯度、句法复杂度、毒性与Self‑BLEU等评估指标。

**📊 数据集**

以佛罗里达大学识字研究所的K–2课程为基础生成129节课对应的10条故事（共2,580条），其中精选996条高质量故事以及模拟错误语料用于训练。

**📈 对比分析**

通过Welch’s t‑test等统计检验显示，Rewarded SFT等方案在可读性、连贯度、句法复杂度、困惑度等指标上显著优于GPT‑4o和Llama 3.3 70B，且安全性几乎无问题。

**⚠️ 局限性**

受限于GPU资源、样本量不足、缺乏真实用户和专家评估、单一课程、未验证对未见语音约束的泛化，以及奖励与评估指标共用导致的潜在过度优化。

---

## 648. On the Complexity of Checking Soundness of Natural Reductions (Extended Version)

**arXiv ID:** 2605.13780 | [PDF](https://arxiv.org/pdf/2605.13780v1)

**作者:** Constantin Enea `[一作]` (LIX -- CNRS -- Ecole Polytechnique), Dominik Klumpp `[通讯]` (LIX -- CNRS -- Ecole Polytechnique)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了自然约简（atomic blocks 与同步 rendezvous）来简化参数化并发程序的行为集，并研究了判断给定自然约简是否完备的复杂度问题。

**💡 创新点**

创新点在于给出一种直观可读的约简表示，并在无同步机制时给出完整的多项式时间判定算法，同时揭示即使加入轻量级锁同步也会导致判定问题变为 coNP‑hard，从而说明了先前基于 mover 规则的不完整性。

**🔧 技术方法**

核心技术包括基于 Mazurkiewicz 等价的覆盖预序、符号化的同步字母与闭包操作，以及构造新的可交换关系 Ĩ 来统一处理 atomic block 与 sync‑point 的交互。

**📊 数据集**

本文未使用真实数据集，而是通过构造性示例与合成程序（如 3‑SAT 转化的锁程序）来展示算法与复杂度结果。

**📈 对比分析**

与传统的 Lipton mover 规则相比，本文的判定算法在无锁环境下保持多项式时间，而 mover 规则在某些例子中会错误判断不可约化；在加入锁后，两者均失去多项式可判定性。

**⚠️ 局限性**

限制在于当程序包含锁或其他同步机制时，判定自然约简的完整性变为 coNP‑hard，当前尚无有效多项式上界，且对更复杂同步形式的支持仍需进一步研究。

---

## 649. Where Does Reasoning Break? Step-Level Hallucination Detection via Hidden-State Transport Geometry

**arXiv ID:** 2605.13772 | [PDF](https://arxiv.org/pdf/2605.13772v1)

**作者:** Tyler Alvarez `[一作]` (Rochester Institute of Technology), Ali Baheri `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 312 | [OpenAlex ID](https://openalex.org/A5086511671)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过将大型语言模型生成的推理步骤视为隐藏状态轨迹，利用对比 PCA 形成几何投影，再使用 BiLSTM 进行单次前向检测，提出了 GeoReason 方法来实现一步级的幻觉检测与首次错误定位

**💡 创新点**

创新点在于把幻觉视为轨迹的局部离散，用对比 PCA 捕捉“首次错误”引起的交通距离偏移，并给出理论证明；同时设计了无标签可部署的学生模型以及 margin‑preserving 蒸馏框架

**🔧 技术方法**

技术包括隐藏状态归一化、对比 PCA（contrastive PCA）、最小化 Wasserstein 交通成本的几何特征、MLP 监督、BiLSTM 学习与温度蒸馏、以及理论证明（Ky Fan 定理、传输边际条件等）

**📊 数据集**

在 ProcessBench、PRM800K、HaluEval 和 TruthfulQA 四个多步骤推理与事实性评测数据集上进行实验

**📈 对比分析**

与 TL‑Entropy、TL‑Perplexity、线性探针和 LLM‑Check 等基线比较，GeoReason 的教师模型在所有数据集上均达到 90% 以上 AUROC，学生模型在同一模型上也超过 93%；在跨模型、跨数据集迁移上教师表现稳健，学生则明显衰退，表明 margin 保留是关键瓶颈

**⚠️ 局限性**

局限性包括教师模型需要步骤标签且不可部署；学生在分布外（模型或数据集）迁移时性能骤降，说明蒸馏时仅保持均值预测不足；此外方法依赖于固定层隐藏状态和平均池化，可能无法捕捉更细粒度的错误信号

---

## 650. Dense vs Sparse Pretraining at Tiny Scale: Active-Parameter vs Total-Parameter Matching

**arXiv ID:** 2605.13769 | [PDF](https://arxiv.org/pdf/2605.13769v1)

**作者:** Abdalrahman Wael `[一作]` `[通讯]` (Independent Researcher), Abdalrahman Wael (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在小规模单GPU环境下，对比了密集Transformer与混合专家（MoE）Transformer在相同训练配方下的表现，探讨在不同公平准则下的效果差异。

**💡 创新点**

创新点在于系统地将“匹配活跃参数”与“匹配总参数”两种公平准则进行对比，证明在活跃参数匹配下MoE显著优于密集模型，而在总参数匹配下则相反；同时验证了Top-2路由、Switch式负载均衡及router z-loss在tiny-scale MoE中的关键作用。

**🔧 技术方法**

采用LLaMA式解码器架构，MoE使用Mixtral风格的四个专家、Top-2路由、Switch式负载均衡与router z-loss；密集模型则在宽度上做微调以满足活跃或总参数匹配。

**📊 数据集**

使用TinyStories小规模语言建模数据集进行预训练与验证。

**📈 对比分析**

通过在相同token、优化器、调度、上下文长度等条件下，构建三种对照实验：匹配活跃参数的密集模型、匹配总参数的密集模型以及匹配活跃参数的MoE模型。结果显示：匹配活跃参数时MoE验证损失为1.5788±0.0020，低于1.6545±0.0012；匹配总参数时MoE仍略逊于密集模型（1.5608±0.0025）。随着训练token数增加，活跃参数匹配下MoE优势扩大，匹配总参数下密集模型优势缩小。

**⚠️ 局限性**

局限性包括：仅在TinyStories小规模数据上实验，结果不一定能推广到更大规模或不同语料；未展示稀疏训练的wall‑clock加速；未对稀疏模型的“休眠容量”机制给出因果证明；并未针对部署系统进行最优稀疏内核评估。

---

## 651. FrameSkip: Learning from Fewer but More Informative Frames in VLA Training

**arXiv ID:** 2605.13757 | [PDF](https://arxiv.org/pdf/2605.13757v1)

**作者:** Bin Yu `[一作]` (Harbin Institute Of Technology), Kai Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无模型依赖的数据层帧选择框架，利用动作变化、视觉-动作一致性、任务进度先验和抓手转移信息对机器人演示轨迹中的帧进行重要性评分，并在训练时按保留比例压缩轨迹，只保留高重要性帧进行监督。

**💡 创新点**

创新点在于将时间监督不平衡问题视为可调节的监督分配任务，首次在 VLA 训练中引入轻量级的轨迹级重要性评估与按比例压缩，且不改动网络结构，直接通过数据加载器实现。

**🔧 技术方法**

使用动作变化（AVI）、视觉-动作一致性（VAC）、任务进度先验（TPI）和抓手转移保留等指标进行帧评分；通过阈值量化、比例约束和时间一致性约束实现压缩；在训练中加入 warmup、压缩视图采样与全帧锚点交替的采样策略。

**📊 数据集**

在三大仿真基准上评估：RoboCasa-GR1（双手桌面任务）、SimplerEnv（WidowX 机械臂外域推理）和 LIBERO（Franka 单臂任务）。

**📈 对比分析**

与全帧训练、随机帧采样、仅动作变化采样等对照实验相比，压缩至 20% 帧后宏平均成功率从 66.50% 提升至 76.15%；在所有基准上均有显著提升，且压缩比例低，保持了训练效率。

**⚠️ 局限性**

局限包括：对任务进度先验需要少量标注；压缩比例需在不同任务上手动调参；在极端压缩下可能丢失必要的上下文；实验仅在仿真环境，真实机器人上的效果尚待验证。

---

## 652. Interpretable Machine Learning for Antepartum Prediction of Pregnancy-Associated Thrombotic Microangiopathy Using Routine Longitudinal Laboratory Data

**arXiv ID:** 2605.13786 | [PDF](https://arxiv.org/pdf/2605.13786v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 653. Amplification to Synthesis: A Comparative Analysis of Cognitive Operations Before and After Generative AI

**arXiv ID:** 2605.13785 | [PDF](https://arxiv.org/pdf/2605.13785v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 654. Aligning Network Equivariance with Data Symmetry: A Theoretical Framework and Adaptive Approach for Image Restoration

**arXiv ID:** 2605.13744 | [PDF](https://arxiv.org/pdf/2605.13744v1)

**作者:** Feiyu Tan `[一作]` (Xi'an Jiaotong University), Deyu Meng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 32445 | [OpenAlex ID](https://openalex.org/A5091017287)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出基于数据集级非严格对称性的图像恢复理论，并设计样本自适应等变网络以动态匹配数据对称性。

**💡 创新点**

①首次定义并量化数据集级非严格对称性；②从逆优化角度推导恢复算子等变性与对称性误差的上界；③将理论指导用于构建可学习的变换组和超网络，实现样本级自适应等变卷积。

**🔧 技术方法**

逆优化理论、对称性量化指标、等变卷积（可学习变换组）、超网络预测变换参数、联合损失训练、深度残差网络骨干。

**📊 数据集**

DIV2K、Urban100、B100、Set14、Set5、RSSCN7、Ada4DIR、Rain100L等自然图像、遥感图像与降雨去除数据集。

**📈 对比分析**

与标准CNN、G-CNN、E2-CNN、Har-Net、PDO-eConv、F-Conv、B-Conv、TL-Conv等多种等变/非等变方法对比。实验显示SA-Conv在超分辨率、去噪、去雨等任务上均优于基线与其他等变模型，提升PSNR约0.3–0.6 dB，视觉效果更锐利、细节更完整。

**⚠️ 局限性**

理论假设（如μ-二次增长、连续光滑性）在高度非凸的实际网络中可能不严格满足；自适应变换组需要额外的超网络训练，计算开销较大；对非欧氏空间或更复杂变换组的推广仍待研究。

---

## 655. Coordinating Multiple Conditions for Trajectory-Controlled Human Motion Generation

**arXiv ID:** 2605.13729 | [PDF](https://arxiv.org/pdf/2605.13729v1)

**作者:** Deli Cai `[一作]` (South China University of Technology), Changxing Ding `[通讯]` (South China University of Technology)

**通讯引用:** 4907 | [OpenAlex ID](https://openalex.org/A5038748720)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种两阶段的解耦式轨迹控制人体运动生成框架CMC，先用扩散模型根据文本和轨迹生成局部控制关节的简化轨迹，再通过文本条件的扩散不painting完成全身运动。

**💡 创新点**

创新点在于：①将文本与轨迹条件解耦，使用简化的运动表示避免冗余表示带来的不一致；②引入Selective Inpainting Mechanism (SIM)，通过在训练中交替进行文本生成与不painting任务提升模型泛化；③采用L‑BFGS优化实现精细的轨迹引导。

**🔧 技术方法**

采用扩散模型（MDM）、Transformer 编码器、CLIP 文本编码器、轨迹引导（classifier‑guidance）与 L‑BFGS、SIM 以及简化运动表示。

**📊 数据集**

使用 HumanML3D 与 KIT‑ML 两大文本到运动数据集。

**📈 对比分析**

与 GMD、Omnicontrol、InterControl、TLControl 等主流方法在控制误差、FID、R‑precision、diversity 等指标上进行对比，CMC 在大多数指标（尤其是轨迹误差和 FID）均实现了最优或接近最优的性能。

**⚠️ 局限性**

限制在于：对极端或 OOD 轨迹仍会出现误差；轨迹指导在原始空间实现导致梯度不均匀，影响控制误差和位置误差；对绝对坐标高度的控制不如相对高度直观。

---

## 656. Learning to Optimize Radiotherapy Plans via Fluence Maps Diffusion Model Generation and LSTM-based Optimization

**arXiv ID:** 2605.13713 | [PDF](https://arxiv.org/pdf/2605.13713v1)

**作者:** Isabella Poles `[一作]` (Politecnico di Milano), Dorin Comaniciu `[通讯]` (Siemens Healthineers)

**通讯引用:** 34725 | [OpenAlex ID](https://openalex.org/A5012751147)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种基于扩散模型的学习优化（Learning-to-Optimize, L2O）框架，用于端到端的 VMAT 放疗计划。该框架先通过分布匹配蒸馏训练一个一次性生成流量图（fluence map）的扩散模型（FMD），随后利用 LSTM 记忆网络学习优化梯度的动态更新（L2Plan），在推理阶段即可快速得到满足剂量目标且机床可交付的放疗计划。

**💡 创新点**

创新点包括：① 一次性流量图生成，消除了传统扩散模型的迭代消噪开销；② 通过分布匹配蒸馏和 GAN 对抗训练兼顾分布一致性和感知质量；③ LSTM 记忆网络实现自适应梯度更新，解决传统梯度优化的平坦/鞍点问题；④ 与传统优化器（如 L-BFGS、DAO）相比，模型无需手动调参或重初始化，可在更少的步骤内实现高质量计划，且对剂量目标变化具备即时适配能力。

**🔧 技术方法**

使用的技术包括：扩散模型（EDM、DMD、GAN对抗蒸馏）、LSTM 记忆优化器（L2Plan）、可微剂量引擎、机床约束（叶片运动、剂量率、转盘速度）以及叶片序列化（Leaf Sequencing, LS）验证。

**📊 数据集**

主要数据集为公开的 REQUITE 质子/VMAT 计划数据集（12,469 个单弧前列腺病例），以及两组私有前列腺病例（135 例和 13 例）用于验证通用性。

**📈 对比分析**

与现有的标准 VMAT 优化器（RMSProp、L‑BFGS、SGD、Adam、DAO）、生成模型（VQ‑VAE、StyleGAN2、D2O）以及其它 L2O 基线（VeLO、μLO、CoordMath、HyperAdam）在同一任务上进行比较。实验表明：L2Plan 在 PSNR（45.33）和 MAE_PTV（0.149）/MAE_OARs（0.061）上均优于对比方法，且仅需 159 秒、7 倍更少的迭代步骤；FMD 的 FID 为 17.52，显著低于其它生成模型。性能提升显著且统计显著（p < 0.001）。

**⚠️ 局限性**

局限性包括：① 仍依赖可微剂量引擎和预先定义的剂量目标，跨机构/机构间剂量规划规则的迁移性尚未充分验证；② 训练需要高性能 GPU，虽然推理速度快但总体训练成本仍高；③ 当前框架仅针对单弧前列腺病例，通用性需进一步在其他器官和多弧场景中评估；④ 叶片序列化和机床可交付性验证仍需与真实硬件进一步对齐。

---

## 657. "Like Taking the Path of Least Resistance": Exploring the Impact of LLM Interaction on the Creative Process of Programming

**arXiv ID:** 2605.13776 | [PDF](https://arxiv.org/pdf/2605.13776v1)

**作者:** Zeinabsadat Saghi `[一作]` (University of Southern California), Souti Chattopadhyay `[通讯]` (University of Southern California)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5059243154)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文通过在受控实验中比较LLM辅助与无辅助编程，评估其对程序员创意过程与产出质量的影响。

**💡 创新点**

创新点在于从创意过程与产出双重视角量化LLM协作的影响，并提出四种人‑LLM协作模式与相应设计启示。

**🔧 技术方法**

研究使用了GitHub Copilot、Python编程任务、思考大声录音与屏幕捕获等技术，并采用行为编码、访谈、代码度量与语义相似度等方法。

**📊 数据集**

数据集为20名受试者完成的80个Python任务（4类算法与系统设计任务），以及收集的屏幕/音频日志与代码片段。

**📈 对比分析**

对比方法包括行为时序统计、创意瞬间计数、代码正确性与多样性评分、语义相似度、维度提升；结果显示LLM辅助显著缩短创意时间、降低创意瞬间，但提高代码正确性与功能完整度。

**⚠️ 局限性**

局限性包括仅使用Python和Copilot、样本量有限、任务类型单一、缺乏跨语言与跨工具验证，且实验环境较为人工控制，可能限制对真实开发情境的外推性。

---

## 658. High-Rate Quantized Matrix Multiplication II

**arXiv ID:** 2605.13768 | [PDF](https://arxiv.org/pdf/2605.13768v1)

**作者:** Or Ordentlich `[一作]` (Hebrew University of Jerusalem), Yury Polyanskiy `[通讯]` (MIT)

**通讯引用:** 9243 | [OpenAlex ID](https://openalex.org/A5031031216)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大型语言模型（LLM）的权重量化问题，在已知激活协方差矩阵 Σ_X 的前提下，提出了理论分析和实用算法，并在高分辨率场景下展示了与信息理论极限的接近程度。

**💡 创新点**

创新点包括：
• 将加权均方误差（WMSE）量化问题转化为逆水分配（reverse waterfilling）并给出精确的理论极限；
• 提出基于 Cholesky 分解的 WaterSIC 算法，在保持基向量不变的情况下通过自适应坐标缩放实现接近水分配的性能；
• 证明即使解码器不知晓 Σ_X，使用全局正交均匀码本也能在高分辨率下几乎达到信息理论极限；
• 分析随机旋转对 GPTQ/LDLQ 性能的提升与基向量优先性的关系。

**🔧 技术方法**

使用的技术主要包括：
• 加权均方误差量化理论（信息理论与逆水分配）；
• Cholesky 分解与矩阵变换；
• 低复杂度的 Successive Interference Cancellation (SIC) 与 GPTQ/LDLQ 算法；
• 形状编码（entropy coding、矩形形状化）以及高分辨率格点量化理论；
• 随机旋转分析与矩阵谱估计。

**📊 数据集**

实验数据集：
• 采用 Llama‑3‑8B 模型的多层权重与对应的激活协方差矩阵（通过 Wikitext‑2 采样得到的校准数据）。
• 对比使用 GPTQ、WaterSIC、以及理想水分配极限的误差曲线。

**📈 对比分析**

比较方法：
• 在相同位宽（R 位）下计算加权均方误差 D；
• 将实验结果与理论极限 D^*(R)、以及均匀码本下的 D_iso 进行对比；
• 结果显示：
  - WaterSIC 在高分辨率下与信息理论极限相差约 2πe/12（约 0.25 bit）
  - GPTQ 在随机旋转后性能仅略高于 WaterSIC，显示其在实际部署中已十分接近最优；
  - 对于低比特率（0.5–2 bit/entry）时，WaterSIC 具有明显优势。

**⚠️ 局限性**

局限性：
• 主要关注高分辨率场景，低比特率下的效能仍需进一步研究；
• 依赖随机旋转与大输出维度以实现对尺度向量 α_i 的有效分摊；
• 形状编码（EC）在 GPU 上解压速度慢，实际部署时需寻找更高效的形状化方案；
• 高维格点量化的 NSM 问题在维度提升后解码复杂度上升；
• 对权重矩阵的统计假设（近似 Gaussian）与实际 LLM 权重的偏差可能影响理论预测与实践表现的差距。

---

## 659. First Steps Towards Probabilistic Iris: Harmonizing Independence, Conditioning, and Dynamic Heap Allocation

**arXiv ID:** 2605.13765 | [PDF](https://arxiv.org/pdf/2605.13765v1)

**作者:** Janine Lohse `[一作]` (MPI-SWS), Emanuele D'Osualdo `[通讯]` (University of Konstanz)

**通讯引用:** 144 | [OpenAlex ID](https://openalex.org/A5013899049)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出了一种新的概率程序逻辑Probabilistic Iris，兼容动态内存分配并支持独立性和条件化推理。

**💡 创新点**

创新点在于引入了基于索引取值的概率模型，使得资源在不同随机分支下保持独立并实现帧保持更新；同时将Iris的权威资源代数迁移到概率域，解决了传统GPL在动态分配下的局限。

**🔧 技术方法**

采用了索引取值模型、概率资源代数、权威资源构造、概率帧保持更新（PFP）以及Iris的弱前置条件模态。

**📊 数据集**

无（本文为理论模型与证明，无实验数据集）。

**📈 对比分析**

通过在Coq证明助手中实现并验证了若干示例（均匀采样、马尔可夫封面、随机排序链表），展示了逻辑的可用性；性能评估以证明复杂度为主，未给出数值实验。

**⚠️ 局限性**

仅支持离散有限支持的终止程序，未包含无限循环、连续分布、并发或更高级的Iris特性；对步骤索引、无穷支持和更复杂的随机分布的支持仍是未来工作。

---

## 660. Fast and effective algorithms for fair clustering at scale

**arXiv ID:** 2605.13759 | [PDF](https://arxiv.org/pdf/2605.13759v1)

**作者:** Claudio Mantuano `[一作]` (University of Bern), Philipp Baumann `[通讯]` (University of Bern)

**通讯引用:** 5762 | [OpenAlex ID](https://openalex.org/A5072687277)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了三种基于 k‑means 分解框架的公平聚类启发式算法（MPFC、MS‑FlowFC、S‑MPFC），通过在分配步骤中构造二进制线性规划或多阶段最小成本流问题，实现对聚类成本与公平度（平衡）之间的精确控制，并在百万级数据规模上实现秒级计算。

**💡 创新点**

创新点在于（1）引入单一容忍参数 λ 直接指定目标公平度，从而取代传统的权重参数；（2）为多敏感特征和多保护组的公平聚类提供可扩展且可嵌入额外约束（如容量、must‑link、cannot‑link）的通用框架；（3）提出多阶段最小成本流分配策略和聚合预处理（批次代表）实现大规模可扩展性；（4）提供两种基准精确求解模型供性能评估。

**🔧 技术方法**

技术手段包括 k‑means 分解、二进制线性规划（BLP）、最小成本流（MCF）求解、预处理聚合（批次生成与代表权重）、并行随机初始化、以及对公平度平衡的数学定义与约束。

**📊 数据集**

实验数据涵盖两个合成数据集、五个 UCI 公开数据集（包括 credit_card、phone_call、census_1994、diabetes、census_1990）以及一个 100 万级 Cyber‑Physical 系统安全数据集（Secure Water Treatment），样本量从 21 到 2,458,285。

**📈 对比分析**

与现有公平聚类基线（SFC、O&C）、两种精确求解方法（MIQCP、SetVars）以及普通 k‑means（Lloyd）比较，MPFC 与 MS‑FlowFC 在中小规模实例上可得到最优或接近最优解，S‑MPFC 在百万级实例中实现 99.7% 的时间节省；在公平度可控性上，MPFC/MS‑FlowFC 能以几乎不增加聚类成本的代价达到接近数据集平衡的公平度，而 SFC、O&C 在公平度与成本平衡上更为粗糙。

**⚠️ 局限性**

局限性包括：MPFC 需要求解 BLP，规模仍受限于整数规划求解器；MS‑FlowFC 仅适用于单敏感特征；S‑MPFC 在极端不平衡的保护组比例下，代表数 r 的选择会影响可行性；所有方法均在随机初始化下存在一定结果波动；最后，本研究仅关注 k‑means 目标，可进一步扩展到 k‑median/k‑medoids 等。

---

## 661. Min Generalized Sliced Gromov Wasserstein: A Scalable Path to Gromov Wasserstein

**arXiv ID:** 2605.13753 | [PDF](https://arxiv.org/pdf/2605.13753v1)

**作者:** Ashkan Shahbazi `[一作]` (Vanderbilt University), Soheil Kolouri `[通讯]` (Vanderbilt University)

**通讯引用:** 3477 | [OpenAlex ID](https://openalex.org/A5068682350)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种min Generalized Sliced Gromov–Wasserstein（min GSGW）框架，利用非线性通用切片器与升维映射直接构造一维排序后得到的运输计划，兼具刚性运动不变性与O(nlogn)的计算复杂度；

**💡 创新点**

创新点包括：①通过共享升维映射将两空间的点映射到同一高维空间，再使用非线性切片器生成可逆的push‑forward值，得到比线性切片更丰富的单调耦合；②直接在此耦合族上最小化原始GW目标，给出GW上界且具备刚性运动不变性；③提出可学习的amortized变体，一次前向推理即可输出运输计划；④使用可微排序（LapSum）实现端到端训练；

**🔧 技术方法**

采用可微排序（LapSum）生成软排列矩阵、soft coupling；对per‑instance和amortized模式均使用Transformer编码器+cross‑attention产生push‑forward值；训练时最小化融合GW损失；

**📊 数据集**

实验使用动物网格匹配、马形状插值、ShapeNet部件分割以及基于Princeton的形状匹配基准；

**📈 对比分析**

与POT‑GW、Sinkhorn、LR‑GW、SDP‑GW、SaGroW等经典GW求解器以及MSGW等切片方法对比，min GSGW在地理误差、标签迁移准确率上取得最优或竞争性结果，同时运行时间比传统全局GW求解器快数十倍，且显著低于其他切片方法；

**⚠️ 局限性**

局限性在于切片器族的表达能力：若不包含最优GW耦合，则只能给出上界；仍依赖单调耦合的近似；amortized模型需要训练，且在极高维或非均匀测度场景下可能面临逼近误差；

---

## 662. LEXI-SG: Monocular 3D Scene Graph Mapping with Room-Guided Feed-Forward Reconstruction

**arXiv ID:** 2605.13741 | [PDF](https://arxiv.org/pdf/2605.13741v1)

**作者:** Christina Kassab `[一作]` (University of Oxford), Maurice Fallon `[通讯]` (University of Oxford)

**通讯引用:** 6074 | [OpenAlex ID](https://openalex.org/A5072974727)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了LEXI‑SG，一种仅使用RGB摄像头的稠密单目视觉SLAM系统，能够构建开放词汇的3D场景图；

**💡 创新点**

创新点在于：基于房间的推迟式feed‑forward重建策略、利用DINO特征进行房间分割、Sim(3)因子图全局对齐以及将二维分割与跟踪结果提升为三维对象节点；

**🔧 技术方法**

使用的技术包括DINO特征检测房间边界、MapAnything feed‑forward重建、Recognize Anything与Grounding Dino进行开放词汇分割、SAM2跟踪以及Levenberg–Marquardt优化；

**📊 数据集**

评估数据集包括Habitat‑Matterport3D（多房间室内环境）和自采的Aria Office Dataset（两层办公场景）等；

**📈 对比分析**

与MASt3R‑SLAM、VGGT‑SLAM、ViSTA‑SLAM等基线相比，LEXI‑SG在位姿误差、Chamfer距离和开放词汇分割指标上取得了最佳或相近的性能；

**⚠️ 局限性**

主要局限是：在开放式布局中房间分割精度不足，且由于未对关键帧进行单独优化，pose精度受feed‑forward模型的上限限制。

---

## 663. Senses Wide Shut: A Representation-Action Gap in Omnimodal LLMs

**arXiv ID:** 2605.13737 | [PDF](https://arxiv.org/pdf/2605.13737v1)

**作者:** Trung Nguyen Quang `[一作]` (Nanyang Technological University), Ziwei Liu `[通讯]` (Nanyang Technological University)

**通讯引用:** 46130 | [OpenAlex ID](https://openalex.org/A5100406050)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了omnimodal LLM在文本前提与感知不符时的失败机制，探究其是感知缺失还是行动失误。

**💡 创新点**

提出Representation–Action Gap概念，首次在视觉/音频跨模态任务中揭示隐藏状态编码与输出行为的解耦，并创建IMAVB benchmark。

**🔧 技术方法**

采用线性探测、logit lens分析隐藏状态，和Probe-guided Logit Adjustment (PGLA) 推断时介入技术。

**📊 数据集**

使用由500段长电影剪辑构成的IMAVB数据集，保证视频、音频不被裁剪，前提错误仅在文本中呈现。

**📈 对比分析**

在8款开源omnimodal LLM与Gemini 3.1 Pro 上评估，发现大多数模型在误导前提下拒绝率低，PGLA 平均提升约15pp 的平衡准确率。

**⚠️ 局限性**

实验仅覆盖单一多选问答场景，未验证跨任务通用性，且在部分模型上PGLA 反而降低标准问题的准确率，说明方法受限于模型内部对拒绝的原生偏好。

---

## 664. Robust and Explainable Bicuspid Aortic Valve Diagnosis Using Stacked Ensembles on Echocardiography

**arXiv ID:** 2605.13730 | [PDF](https://arxiv.org/pdf/2605.13730v1)

**作者:** Christos Chrysanthos Nikolaidis `[一作]` (Democritus University of Thrace), Pavlos S. Efraimidis `[通讯]` (Democritus University of Thrace)

**通讯引用:** 1414 | [OpenAlex ID](https://openalex.org/A5047660023)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本研究开发了一种可解释的多基底视频集成模型，利用常规获取的 PLAX 视图超声心动图视频来区分二叶瓣膜（BAV）与三叶瓣膜（TAV）。

**💡 创新点**

创新点包括：① 在小样本设置下使用泄漏意识的外层交叉验证与 out‑of‑fold 堆叠；② 集成多种预训练的 3D CNN 骨干，并在元层采用多元学习、后验校准；③ 结合 Grad‑CAM 与 SHAP 进行双层可解释性分析。

**🔧 技术方法**

技术细节：预训练骨干（MC3、R3D、X3D、R2P1D、S3D）；堆叠元学习器（LR、SVM、MLP、XGBoost、Transformer）；Platt 标定；Grad‑CAM 视觉解释；SHAP 归因。

**📊 数据集**

使用单中心 90 例 PLAX 超声心动图数据集，其中 48 例为 BAV，42 例为 TAV，采用患者层级分层划分。

**📈 对比分析**

在固定 3 折外层交叉验证、10 种随机种子下评估，集成模型平均 F1 分数 0.907，AUROC 0.849，AP 0.885，精确率 0.941，召回率 0.877，Brier 分数 0.149，显著优于单骨干模型和先前报道。

**⚠️ 局限性**

局限性：样本量小且单中心，缺乏多中心外部验证；仅使用 PLAX 视图，未包含 PSAX 或多模态信息；解释结果为定性分析，未通过临床读者评估；模型对图像质量和采集方式敏感。

---

## 665. Tight Sample Complexity Bounds for Entropic Best Policy Identification

**arXiv ID:** 2605.13717 | [PDF](https://arxiv.org/pdf/2605.13717v1)

**作者:** Amer Essakine `[一作]` (ENS Paris Saclay), Claire Vernade `[通讯]` (University of Technology Nuremberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了一种面向前向交互模型的风险敏感强化学习中最优策略识别算法。

**💡 创新点**

创新点在于利用指数期望的平滑性获得更紧的集中界，并设计了基于 KL 的探索奖金和专门的停止规则，使样本复杂度匹配信息理论下界。

**🔧 技术方法**

采用了指数变换的贝尔曼递推、KL 探索奖励、方差敏感置信区间以及自适应停止判据。

**📊 数据集**

实验使用了一个 8 状态的自定义 MDP 进行验证。

**📈 对比分析**

与现有基于奖励方差或 UCB 的风险敏感算法相比，Entropic‑BPI 在样本复杂度上实现了对数因子内的最优匹配，显著减少了原先 e^{2|β|H} 的指数系数。

**⚠️ 局限性**

仍存在的局限是样本复杂度对周期仍保持指数增长，且仅适用于前向交互模型，未考虑无模型或生成器场景。

---

## 666. TinySDP: Real Time Semidefinite Optimization for Certifiable and Agile Edge Robotics

**arXiv ID:** 2605.13748 | [PDF](https://arxiv.org/pdf/2605.13748v1)

**作者:** Ishaan Mahajan `[一作]` (Columbia University), Brian Plancher `[通讯]` (Dartmouth College)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

提出TinySDP，一种在资源受限的嵌入式平台上实时执行半正定优化的模型预测控制框架，用于实现可证明的障碍物避免。

**💡 创新点**

创新点包括：① 将半正定约束按阶段分解并与Ricatti结构相结合，构建可缓存的ADMM求解器；② 设计后验rank‑1安全证书，实时验证规划解的几何安全性；③ 在不需要额外安全裕度的情况下实现比传统方法更短、更精确的路径。

**🔧 技术方法**

核心技术：半正定拉升（lifting）+阶段性PSD约束；缓存Ricatti递推的ADMM求解器；rank‑1后验安全证书；对角线化投影实现快速PSD投影；离线缓存与在线快速迭代。

**📊 数据集**

使用的实验数据集：1）静态U形陷阱（四个初始点）和动态移动障碍（静态+周期性移动圆）模拟环境；2）真实Crazyflie 2.1四旋翼在STM32F405微控制器上进行的动态障碍物避让实验。

**📈 对比分析**

比较方法：TinyMPC-LIN（线性化切半空间约束）、TinyMPC-HOCBF（高阶控制壁垒函数）、RPCBF（基于采样的安全过滤）。TinySDP在静态障碍物场中路径长度比RPCBF短31–73%，在动态场景中比RPCBF短30%，并在所有案例中保持安全且达到0.02m以内目标；相对TinyMPC-LIN，TinySDP路径更短、精度更高，但仍需更高的安全裕度以达到可行性。

**⚠️ 局限性**

局限性：① 缺乏全闭环递归可行性证明；② 对3D或更大尺寸障碍的PSD投影成本较高，尚未在微控制器上实现；③ 运行时安全监控仅是被动故障切换，无法预先处理极快障碍或大建模误差，缺乏主动前瞻性安全保障。

---

## 667. VectorSmuggle: Steganographic Exfiltration in Embedding Stores and a Cryptographic Provenance Defense

**arXiv ID:** 2605.13764 | [PDF](https://arxiv.org/pdf/2605.13764v1)

**作者:** Jascha Wanger `[一作]` `[通讯]` (ThirdKey / Tarnover, LLC), Jascha Wanger (ThirdKey / Tarnover, LLC)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究在检索增强生成（RAG）系统中利用向量嵌入进行隐写式数据外泄，并评估多种扰动技术的隐蔽性与可检测性，提出向量钉住（VectorPin）协议作为完整的完整性防御。

**💡 创新点**

创新点包括：①系统性展示多种向量扰动（噪声、旋转、缩放、偏移、碎片化和组合）在不同模型、存储后端和量化设置下的隐蔽窗口与检测阈值；②证明正交旋转在小角度/少旋转区间内对统计检测完全无效；③推导旋转通道的容量上限 ⌊d/2⌋·b 位并给出可实现的编码/解码方案；④提出简洁的 VectorPin 加密原语，实现跨语言、跨后端的向量完整性签名。

**🔧 技术方法**

使用的技术包括：密集向量嵌入模型（OpenAI text‑embedding‑3‑large、Nomic、EmbeddingGemma 等）、统计异常检测（Isolation Forest、One‑Class SVM）、正交旋转与 Givens 旋转、量化（int8、Product‑Quantization）、自定义 JSON/Ed25519 签名、NFC 标准化哈希、跨语言一致性测试。

**📊 数据集**

实验数据集涵盖：68 份合成 PII 文档（synthetic‑PII）、BEIR NFCorpus（16,763 份医疗文档）和 Quora 子集（10,000 份问答），以及 OpenAI 文档嵌入模型的 3072 维向量和多种本地模型的 768/1024 维向量。

**📈 对比分析**

比较方法：对每种扰动方式在多种噪声/角度参数下计算向量相似度、检测 ROC‑AUC、量化后余弦损失、检索召回率，并在七种向量后端（FAISS、Qdrant、Chroma 等）进行交叉验证。性能结果显示：大多数分布移位技术在非自适应攻击者下能被检测（AUC ≥0.85）；旋转在小角度下完全不可检测；量化对攻击无显著阻碍；VectorPin 在任何后端均能以无量子开销的方式实现完整性验证。

**⚠️ 局限性**

局限性包括：1）VectorPin 仅防御后嵌入后的篡改，无法抵御签名密钥泄露或在嵌入时已篡改的向量；2）对跨模型碎片化攻击的检测仍需依赖后端结构；3）量化/压缩后端的细粒度攻击检测尚未充分探索；4）实验使用的合成 PII 与实际企业数据在分布与规模上的差异可能影响检测阈值；5）自适应攻击者可通过白盒 PGD 进一步降低检测率。

---

## 668. EconAI: Dynamic Persona Evolution and Memory-Aware Agents in Evolving Economic Environments

**arXiv ID:** 2605.13762 | [PDF](https://arxiv.org/pdf/2605.13762v1)

**作者:** Annie Liu `[一作]` (Tsinghua University), Zigan Wang `[通讯]` (Tsinghua University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `67630363-6be0-4f51-ab05-7198250671a5` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一个名为EconAI的基于大型语言模型的宏观微观经济代理模拟框架，整合事件感知、长期/短期记忆与经济情绪指数。

**💡 创新点**

创新点在于引入经济情绪指数(ESI)与记忆加权机制，使代理能够在短期决策与长期战略规划之间自适应平衡，并以单一LLM实现统一的宏观与微观层面建模。

**🔧 技术方法**

使用了大规模语言模型（GPT‑4o‑mini）作为核心认知引擎，结合文本编码器MiniLM进行事件摘要、记忆检索，并采用指令微调与情绪指数计算。

**📊 数据集**

数据集基于美国2018年人口与税务分布，合成生成的家庭与企业代理信息，以及通过LLM生成的职位与收入分布。

**📈 对比分析**

与四类基线（LEN、CATS、AI‑Eco、EconAgent）在20年模拟中比较，EconAI在通胀、失业、名义GDP等宏观指标的波动范围内更稳定，且成功重现菲利普斯曲线与奥肯定律，优于传统规则/学习模型。

**⚠️ 局限性**

局限包括对大规模LLM的高计算与存储成本、参数调优难度（记忆衰减、情绪权重）以及实验主要基于合成数据，缺乏对极端真实事件的完整验证。

---

## 669. Toward AI-Driven Digital Twins for Metropolitan Floods: A Conditional Latent Dynamics Network Surrogate of the Shallow Water Equations

**arXiv ID:** 2605.13761 | [PDF](https://arxiv.org/pdf/2605.13761v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 670. Weakly-Supervised Spatiotemporal Anomaly Detection

**arXiv ID:** 2605.13746 | [PDF](https://arxiv.org/pdf/2605.13746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 671. Generative Texture Diversification of 3D Pedestrians for Robust Autonomous Driving Perception

**arXiv ID:** 2605.13755 | [PDF](https://arxiv.org/pdf/2605.13755v1)

**作者:** Arka Bhowmick `[一作]` (BIT Technology Solutions), Oliver Wasenmuller `[通讯]` (Mannheim University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

基于单一3D基础人模型，通过 StyleGAN2 在 UV 纹理空间合成多样化人脸纹理，然后映射到3D网格，实现大规模身份多样化的合成行人资产，并用这些资产生成合成场景数据；随后将合成数据与真实数据混合，评估对RGB和点云目标检测的影响。

**💡 创新点**

① 通过纹理而非几何变化实现身份多样化，显著降低3D模型重建成本；② 采用显式的潜在空间属性方向学习与截断、距离自适应步长相结合的控制方法，保证合成纹理的语义一致性与视觉质量；③ 对2D与3D检测分别进行系统的跨域混合数据实验，揭示3D检测对几何域偏移更为敏感。

**🔧 技术方法**

StyleGAN2 训练于 FFHQ-UV；UV 纹理映射到3D网格；潜在空间属性操控（SVM学习属性方向、截断 ψ 与距离自适应 α）；YOLOv7（2D检测）与 SECOND（3D检测）模型；评估指标为 mAP@50、FID/CLIP、3D‑FID/3D‑KID。

**📊 数据集**

FFHQ‑UV（用于训练 GAN 和生成纹理），KITTI、BDD100K、A2D2（真实RGB/点云数据）以及内部生成的合成数据集。

**📈 对比分析**

对比单纯真实、单纯合成及多种混合比例的训练集，使用相同超参训练 YOLOv7 与 SECOND。结果显示：在 2D 检测中，加入合成数据可使 KITTI 上 mAP 从 88.0% 提升到 91.7%（约 +3.7% 绝对、+4.2% 相对），BDD100K 上亦提升约 3%。在 3D 检测中，合成数据混合导致 KITTI、A2D2 上 mAP 下降 5–20%（如从 48.0% 降至 45.6% 或 34.4% 降至 27.5%）。

**⚠️ 局限性**

仅对人脸纹理进行多样化，未覆盖全身纹理或几何形变；合成与真实数据混合在 3D 检测中容易产生几何域偏移，导致性能下降；实验中未使用域自适应或几何归一化方法，限制了 3D 模型的跨域鲁棒性。

---

## 672. Learning POMDP World Models from Observations with Language-Model Priors

**arXiv ID:** 2605.13740 | [PDF](https://arxiv.org/pdf/2605.13740v1)

**作者:** Valentin Six `[一作]` (Max Planck Institute for Intelligent Systems), Bernhard Schölkopf `[通讯]` (Max Planck Institute for Intelligent Systems)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用大语言模型在仅有观测-动作-奖励轨迹的条件下，通过可执行的POMDP代码生成和基于粒子滤波的似然评估，迭代精炼并学习可用于决策的内部世界模型。

**💡 创新点**

①仅凭观测轨迹而非隐藏状态实现POMDP诱导；②使用LLM生成可执行程序；③基于粒子滤波的贝叶斯似然评分；④利用LLM反馈进行模型细化。

**🔧 技术方法**

大语言模型（如Qwen 3.6 Plus）生成代码；粒子滤波进行滤波和似然计算；自适应调度的Refinement-by-Execution；贝叶斯/蒙特卡洛树搜索/DA*等规划算法。

**📊 数据集**

MiniGrid系列半可观测任务（Empty、Corners、Lava、Four Rooms、Unlock 等）。

**📈 对比分析**

与具备隐状态监督的POMDP Coder以及无LLM的tabular基线对比，Pinductor在奖励和胜率上与POMDP Coder相当，且显著优于tabular基线；在离线样本效率上只需10条轨迹即可达到强性能。

**⚠️ 局限性**

仅在MiniGrid上验证，缺乏跨环境迁移和深度RL对比；LLM调用高方差导致训练不稳定；未探索对规划器、观察距离等参数的LLM优化。

---

## 673. Porting the Nonlinear Optimization Library HiOp to Accelerator-Based Hardware Architectures

**arXiv ID:** 2605.13736 | [PDF](https://arxiv.org/pdf/2605.13736v1)

**作者:** Slaven Peles `[一作]`, Cosmin G. Petra `[通讯]` (Lawrence Livermore National Laboratory)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

实现了将HiOp内点法求解器完整迁移到GPU加速平台，并通过混合稠密‑稀疏压缩技术，将原本稀疏的KKT系统转化为可用GPU稠密线性求解器（MAGMA）求解的稠密系统。

**💡 创新点**

创新点在于：① 通过假设“交叉”Hessian项为零以及稀疏变量Hessian可对角化，设计出混合稠密‑稀疏内点法框架；② 利用Kron降维与两阶段压缩，显著减小稠密子系统规模；③ 将整个计算流程与RAJA/UMPire硬件抽象层相结合，实现了多平台可移植性与GPU高效执行。

**🔧 技术方法**

使用技术包括：
- RAJA（kernel并行化、执行策略抽象）
- Umpire（统一内存管理）
- MAGMA（GPU稠密线性求解）
- CUDA（GPU编程）
- OpenBLAS / Intel MKL（CPU基线对照）
- 自定义稠密‑稀疏矩阵与向量实现。

**📊 数据集**

数据集主要为电力系统最优潮流（OPF）问题：
- Texas 4000‑5000 机架规模（单GPU可容纳）
- New England 20k 机架（示例但未完全处理）
- 通过自研小型应用（mini‑app）生成尺寸 2k–22k 的稠密矩阵，用于性能基准测试。

**📈 对比分析**

比较方法：
- 对同一规模问题分别使用 OpenBLAS、Intel MKL（CPU）和 MAGMA（GPU）进行单次迭代时间测量；
- 对GPU实现分为 Non‑RAJA（仅稠密线性求解在GPU）和 RAJA（全流程GPU）两种；
- 结果显示：
  * 在 16k×16k 矩阵上，GPU MAGMA 仅需 4.49 s，OpenBLAS 72.95 s，MKL 18.56 s，达到 16×/4× 的加速；
  * RAJA 版本相较 Non‑RAJA 将数据传输与核执行开销压至 0.5% 左右；
  * Roofline 分析表明大多数核心核已接近设备峰值浮点能力。

**⚠️ 局限性**

限制与挑战：
- 目前压缩方法仅适用于可将稀疏变量压成对角或近对角的场景，无法处理更大、稠密度更高的电网模型；
- GPU端稠密线性求解仍需将矩阵和 RHS 在 CPU 与 GPU 之间反复拷贝，导致额外 0.45 s 传输时间；
- 需要在 MAGMA 中实现完整的 Bunch‑Kaufman GPU 接口，才能进一步减少数据移动；
- 现有实现未支持分布式内存并行，无法直接扩展到多 GPU 或集群级别；
- 代码中仍存在对数据封装不完整、RAJA 与 Umpire 边界处理不统一等细节问题。

---

## 674. EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents

**arXiv ID:** 2605.13841 | [PDF](https://arxiv.org/pdf/2605.13841v1)

**作者:** Tara Bogavelli `[一作]` (ServiceNow), Srinivas Sunkara `[通讯]` (ServiceNow)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个完整端到端的语音代理评估框架，包含验证的 bot‑to‑bot 音频模拟、可用于不同体系结构（Cascade、Hybrid、S2S）的统一评估指标以及多轮任务场景与声学扰动集合。

**💡 创新点**

创新点在于：1）通过验证门控的多轮 bot‑to‑bot 语音模拟，确保评估场景与真实部署高度一致；2）设计了两组综合指标 EVA‑A（准确性）和 EVA‑X（体验），同时兼顾任务完成、策略一致性、音频实体准确性、对话推进、简洁性与交互时序；3）引入了多维度可靠性统计（峰值/可靠/一致性）和扰动敏感性分析，揭示了架构间的准确性‑体验权衡及对音频扰动的异质响应。

**🔧 技术方法**

采用的技术包括：自动化音频 WebSocket 对话引擎；基于 LLM 的 Judge（LLM‑as‑Judge、LALM‑as‑Judge）进行任务完成、策略可信度、语音实体一致性评估；音频层面语音合成（TTS）与语音识别（STT）；模拟器与工具执行器的可编程化插件；多轮对话状态管理与决策树；统计学方法（bootstrap CI、变异分解、配对符号检验）对结果可靠性与扰动影响进行量化。

**📊 数据集**

使用了三大企业领域（航班客服、医疗人力资源、IT 服务管理）的 213 个对话场景集合，每个场景包含用户目标、语音人设、数据库、真实终态；另外构建了包含方言（法语口音）、背景噪声（咖啡店）及其组合的声学扰动套件。

**📈 对比分析**

与 12 个系统（7 Cascade、2 Hybrid、3 S2S）在 213 个场景（5 次试验）和 90 个扰动子集（3 次试验）下对比。结果显示：Cascade 与 S2S 在准确性上相近，但 S2S 在体验（尤其是对话推进和时序）显著更好；峰值性能与可靠性能差距明显，单次评估往往高估部署可靠性；方言噪声对 Cascade 的准确性影响最大，背景噪声对 S2S 的体验影响最大。

**⚠️ 局限性**

主要限制包括：1）评估使用 LLM Judge 可能出现模型偏好与风格一致性问题；2）LALM Judge 的可靠性尚低；3）不评估 PII 泄露与有害输出；4）仿真器仅为英语、无多语言与真实人类语音波形；5）未覆盖复杂多代理或工具链真实延迟与错误；6）评估成本高、对商用 API 版本敏感。

---

## 675. Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update Your Weights

**arXiv ID:** 2605.13839 | [PDF](https://arxiv.org/pdf/2605.13839v1)

**作者:** Wenrui Bao `[一作]` (University of Central Florida), Yuzhang Shang `[通讯]` (University of Central Florida)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Thought Flow 框架，通过将发送者隐藏状态映射为临时低秩 LoRA 权重扰动，注入到冻结的接收者模型中，实现多代理 LLM 的高效协作。

**💡 创新点**

创新点在于引入权重空间通信的全新范式，取代传统文本消息，利用实例特定的 LoRA 扰动完成信息交互，显著降低 token、预填、KV‑cache 负担。

**🔧 技术方法**

核心技术包括：低秩 LoRA 适配、基于 Transformer 的参数生成器、层聚合与多轴注意力、热度化层加权、临时前向补丁、以及多样性正则化。

**📊 数据集**

训练使用 OpenThoughts、Sky‑T1、KodCode；评估基准包括 GSM8K、MATH、MMLU、HumanEval+、MBPP+。

**📈 对比分析**

与单代理和文本多代理（TextMAS）对比，Thought Flow 在准确率上相对单代理提升 7–8 分点，token 消耗下降 70–83%，推理时间加速 2.3–4.6 倍，且与 TextMAS 的准确率差距 ≤4.5 分点。

**⚠️ 局限性**

局限性：需预先知晓并固定接收器架构，无法自适应不同模型；在代码生成任务中准确率仍低于 TextMAS；实例特定扰动生成导致批量推理受限。

---

## 676. R-DMesh: Video-Guided 3D Animation via Rectified Dynamic Mesh Flow

**arXiv ID:** 2605.13838 | [PDF](https://arxiv.org/pdf/2605.13838v1)

**作者:** Zijie Wu `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于视频引导的3D动画框架R-DMesh，能够在用户提供的静态网格与参考视频姿态不匹配的情况下自动校正姿态并生成高质量的4D网格动画。

**💡 创新点**

核心创新点包括：①引入自监督VAE模型，显式分离条件基网格、相对运动轨迹与关键的姿态校正跳跃偏移；②设计Triflow Attention机制，利用顶点几何特征调控三条正交流，保持物理一致性与局部刚性；③采用Rectified Flow-based Diffusion Transformer结合预训练的视频潜在空间，将空间-时间先验迁移到3D域。

**🔧 技术方法**

使用的技术主要有：变分自编码器（VAE）、Triflow Attention、流式扩散变换器（Diffusion Transformer）、预训练的视频潜在编码器，以及视频驱动的3D网格生成流程。

**📊 数据集**

构建了大规模数据集Video‑RDMesh，包含超过50万条动态网格序列，专门模拟姿态错位情况，用于训练和评估模型。

**📈 对比分析**

通过与传统基于骨骼、基于形状先验的运动迁移方法对比，实验显示R‑DMesh在姿态对齐、几何完整性和跨身份泛化方面均有显著提升；在多种下游任务（姿态重定向、完整的4D视频生成）上表现出更高的鲁棒性和更真实的动画效果。

**⚠️ 局限性**

限制方面包括：①对极端姿态错位的校正仍有一定误差；②模型对输入网格拓扑变形的适应性有限；③扩散变换器推理速度较慢，部署时需要更高计算资源。

---

## 677. QLAM: A Quantum Long-Attention Memory Approach to Long-Sequence Token Modeling

**arXiv ID:** 2605.13833 | [PDF](https://arxiv.org/pdf/2605.13833v1)

**作者:** Hoang-Quan Nguyen `[一作]` (University of Arkansas), Khoa Luu `[通讯]` (University of Arkansas)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于量子叠加的长序列记忆机制QLAM，用量子态保存并递归更新历史信息，并通过测量实现查询式读取。

**💡 创新点**

创新点在于将传统状态空间模型中的线性加法记忆替换为量子态的单位阵演化与叠加，从而实现更丰富的全局表示并保持线性时间复杂度；同时，测量式读出为注意力机制提供了量子化的通用形式。

**🔧 技术方法**

使用了量子线路（参数化量子电路）进行输入编码和状态演化，Pauli测量算符实现查询相关的注意力权重，结合经典深度学习框架（PyTorch + PennyLane）实现混合量子-经典模型。

**📊 数据集**

在将图像转换为序列的三个标准基准上测试：sMNIST、sFashion‑MNIST 与 sCIFAR‑10（分别为 784、784 与 3072 长度的像素序列）。

**📈 对比分析**

与 RNN、Transformer 以及经典状态空间模型进行 10 折平均对比，QLAM 在所有数据集上均取得最高准确率（sMNIST 92.6%→+1.3%，sFashion‑MNIST 81.4%→+0.8%，sCIFAR‑10 53.6%→+1.2%），且方差更低，说明稳定性更好。

**⚠️ 局限性**

局限性包括：仍基于量子模拟器，未在真实量子硬件上验证；模型规模受限于可行的量子比特数；仅在相对简单的图像序列任务上评估，尚未扩展到大型语言或多模态长序列任务。

---

## 678. Harnessing Agentic Evolution

**arXiv ID:** 2605.13821 | [PDF](https://arxiv.org/pdf/2605.13821v1)

**作者:** Jiayi Zhang `[一作]` (Hong Kong University of Science and Technology), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种名为“<framework_name>”的 harnessed meta‑editing 框架，将 agentic evolution 视为交互式环境，允许 meta‑agent 通过编辑演化机制（而不是直接生成候选）来引导长周期搜索；

**💡 创新点**

核心创新是把演化过程本身抽象为环境，利用过程级状态与 meta‑editing 操作实现对演化机制的可编辑控制，统一 procedure‑based 与 agent‑based 方式，突破局部最优瓶颈；

**🔧 技术方法**

使用大语言模型（Claude Code/Codex）、agent‑based 代理、结构化 workspace、评估隔离、两阶段循环（meta‑editing + evolution segment）以及对机制进行代码/上下文编辑的技术；

**📊 数据集**

在 Terminal‑Bench、ARC‑AGI‑2 两个标准基准，以及三项开源优化任务 circle_packing_26、autocorrelation_second、Kernel 优化任务上进行实验；

**📈 对比分析**

与五种 procedure‑based 基线（ADAS、DGM、AFlow、SPO、GEPA）、两种 agent‑based 基线（Codex、Claude Code）、以及 OpenEvolve、HyperAgents 进行对比；在标准基准上相对最强基线提升约 26%，在三项优化任务中取得 state‑of‑the‑art 或最佳结果，收敛速度更快；

**⚠️ 局限性**

局限性包括：每轮算力成本约为 baseline 的 3 倍；对 meta‑agent 技能和 harness 设计高度依赖；可能仍存在奖励劫持风险；meta‑agent 需手工编写技能，缺乏完全自动化，且验证仅限于实验设置。

---

## 679. Min-Max Optimization Requires Exponentially Many Queries

**arXiv ID:** 2605.13806 | [PDF](https://arxiv.org/pdf/2605.13806v1)

**作者:** Martino Bernasconi `[一作]` (Bocconi University), Alexandros Hollender `[通讯]` (University of Oxford)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在论文中，作者研究了在[0,1]^d × [0,1]^d 域上非凸非凹函数的最小-最大优化问题，证明任何寻找 ε-近似一阶驻点的算法都需要指数级查询次数。

**💡 创新点**

创新点在于将 PPAD 难度的布劳威尔固定点问题与可查询版本的布劳威尔问题相结合，构造了带有“oracle 门” 的黑盒问题，并利用光滑插值技术将布尔函数扩展到连续域，从而得到新的查询下界。

**🔧 技术方法**

主要技术包括：1）构造黑盒版的布劳威尔问题并证明其 2^Ω(n^1/3) 的查询下界；2）利用光滑步函数和鲁棒插值实现布尔函数的连续光滑逼近；3）对原问题进行多层归约，最终将查询下界传递到原始的非凸非凹最小-最大优化问题。

**📊 数据集**

本工作为理论分析性论文，不使用任何实验数据集；所有结论均来自数学证明与复杂性分析。

**📈 对比分析**

论文通过证明查询下界与已知上界（如 O(1/ε^2) 的投影梯度下降）形成对比，指出在 ε ≤ 1/d 的情形下，任何算法无法实现多项式查询次数；未给出实验性能评估。

**⚠️ 局限性**

局限性在于下界仅对 ε ≤ 1/d 有效；对于常数 ε，是否存在多项式查询（PTAS）算法仍是开放问题；此外，结论仅适用于理论模型，未考虑实际实现中的数值误差与计算复杂度。

---

## 680. BlitzGS: City-Scale Gaussian Splatting at Lightning Speed

**arXiv ID:** 2605.13794 | [PDF](https://arxiv.org/pdf/2605.13794v1)

**作者:** Zhongtao Wang `[一作]` (Peking University), Guoping Wang `[通讯]` (Peking University)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为 BlitzGS 的分布式 3D 高斯渲染框架，用于快速城市规模 3D 场景重建，显著降低每步的高斯活跃工作量。

**💡 创新点**

创新点在于三层工作量控制：系统层采用索引奇偶分片与全局渲染分发，模型层通过定时重要性评分既裁剪高斯集又重加权密度控制，视图层结合距离 LOD 与重要性掩码双重剔除。

**🔧 技术方法**

采用多 GPU 分布式训练、全局高斯索引分片、前向后向交叉 GPU 交换、周期性重要性评分、可视化权重重加权、基于 LOD 的距离门控、以及基于重要性的视图级裁剪技术。

**📊 数据集**

在四个大规模城市/空中数据集上评估：Mill-19 的 Building、Rubble，UrbanScene3D 的 Residence、Sci-Art，以及 MatrixCity 的 aerial split。

**📈 对比分析**

与众多现有大型 3DGS 基线（如 CityGaussianV1/V2、HUG、Momentum-GS、CityGS-X、VastGaussian）比较，BlitzGS 在保持相近或更高 PSNR/SSIM/LPIPS 的同时，将训练时间从数小时缩短至几十分钟，实现约 10 倍的速度提升。

**⚠️ 局限性**

局限性包括：LOD 门是硬切，可能导致平滑飞行时出现轻微切换卡顿；重要性评分采用固定时间点，未自适应；需要全局高斯集能装入全部 GPU 内存，无法直接扩展到超大场景。

---

## 681. Topology-Preserving Neural Operator Learning via Hodge Decomposition

**arXiv ID:** 2605.13834 | [PDF](https://arxiv.org/pdf/2605.13834v1)

**作者:** Dongzhe Zheng `[一作]` (Princeton University), Christine Allen-Blanchette `[通讯]` (Princeton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在任意Riemannian网格上利用Hodge分解实现拓扑保持的神经算子，用于求解物理场PDE；

**💡 创新点**

通过Hodge正交性提出谱‑几何双分支架构，实现结构保持下的可加近似，解决传统算子在拓扑敏感网格上的不足；

**🔧 技术方法**

结合离散外微积分、Hodge拉普拉斯谱分解、双分支网络（谱基底+环境卷积）和Lie‑Trotter算子分裂；

**📊 数据集**

使用DrivAerNet++汽车表面、三维多连通域与环形传输等任务的约3000节点网格；

**📈 对比分析**

与GNO、MGN、DeepONet、Geo‑FNO、FNO‑3D比较，在三项任务上MSE下降30‑40%，保持拓扑一致性与能量守恒，性能显著优于基线；

**⚠️ 局限性**

需要预先计算Hodge谱，限制对几何不变或仅微小变形场景；不适用于冲击波等强分离现象；仅适用于三维以下Eulerian模拟。

---

## 682. Training Long-Context Vision-Language Models Effectively with Generalization Beyond 128K Context

**arXiv ID:** 2605.13831 | [PDF](https://arxiv.org/pdf/2605.13831v1)

**作者:** Zhaowei Wang `[一作]` (Hong Kong University of Science and Technology), Yangqiu Song `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在LVLM（多模态大语言模型）中，通过对长文本（128K以上）进行持续预训练（LongPT），系统性研究了数据构造、混合与训练策略，提升模型在长文档VQA、长视频理解等长上下文任务上的性能；

**💡 创新点**

①首次将长文档VQA作为高质量长上下文监督数据；②提出基于文档段落的QA生成管线并引入段落锚点；③系统评估并确立长上下文训练的关键设计：长度分布保持多样性、信息抽取与推理任务的8:2混合比例、纯长上下文训练可保留短上下文能力；

**🔧 技术方法**

持续预训练（LongPT）技术、基于OCR的文档解析与段落抽样、Seed 2.0生成QA、RoPE频率调优、动态长度分布策略、多任务混合训练；

**📊 数据集**

1.5M+ PDF文档（学术论文、书籍、技术手册等），从中合成长文档VQA与OCR转录数据；此外使用LLaVA-OneVision等短上下文数据进行混合实验；

**📈 对比分析**

在MMLongBench（64K/128K）、MM‑NIAH、VTCBench、长视频基准等上与多款公开/闭源LVLM对比。相对基线Qwen2.5‑VL‑7B提升≈7.1%（64K/128K），在128K下超越多数开源模型；在256K/512K无额外训练即可维持高分，跨任务迁移表现优异；

**⚠️ 局限性**

仅关注长文档VQA作为监督，未覆盖其他多模态长上下文场景；模型训练依赖高质量OCR与生成QA，可能存在领域偏差；未探索更大上下文窗口（>1M）与更大模型规模；对长上下文任务的泛化机制尚需进一步解释。

---

## 683. Neurosymbolic Auditing of Natural-Language Software Requirements

**arXiv ID:** 2605.13817 | [PDF](https://arxiv.org/pdf/2605.13817v1)

**作者:** Bethel Hall `[一作]` (Stevens Institute of Technology), William Eiers `[通讯]` (Stevens Institute of Technology)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 VeriMed，一个结合大语言模型和 SMT 求解器的系统，自动将医疗设备软件的自然语言需求转化为可验证的形式化模型，并对其进行一致性、空洞性、可违例性和冗余性等四项审计，同时通过多次采样的等价性检查实现自动歧义识别与澄清。

**💡 创新点**

在需求级别引入基于 SMT 的四项审计与多采样等价性检测，首次实现自动形式化的自我验证和歧义消除，并通过求解器反馈驱动修复与澄清。

**🔧 技术方法**

使用大型语言模型（如 GPT‑4/Claude 3.5）生成 SMT‑LIB 代码，采用 Z3 SMT 求解器执行审计和等价性检查，并通过循环反馈机制进行修复和澄清。

**📊 数据集**

以 64 条哈德马里（hemodialysis）医疗设备需求为主实验集，并在 144 条 PCA 泵需求集上进行迁移测试。

**📈 对比分析**

与无反馈、仅需求反馈、仅符号反馈等基线相比，CEGR 在问题回答任务中的最终准确率从 55.4% 提升至 98.5%；在故障注入实验中检测率达 98.4%，在歧义检测实验中实现 100% 纠正。

**⚠️ 局限性**

局限性包括：仅针对同类医疗设备，缺乏对大规模真实需求文档的可扩展性验证；使用 SMT 仅能表达非时序约束，对时序需求处理有限；自动形式化的忠实度仍依赖代理信号，未能完全保证语义完整。

---

## 684. OmniLiDAR: A Unified Diffusion Framework for Multi-Domain 3D LiDAR Generation

**arXiv ID:** 2605.13815 | [PDF](https://arxiv.org/pdf/2605.13815v1)

**作者:** Youquan Liu `[一作]` (Fudan University), Wanli Ouyang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 OmniLiDAR，一个统一的文本条件扩散框架，能够在八个代表性域（天气、传感器配置、平台）下生成 LiDAR 扫描，并通过生成的数据实现下游任务的增强。

**💡 创新点**

创新点包括：①跨域训练策略（CDTS）在一个 mini‑batch 中混合不同域样本；②跨域特征建模（CDFM）利用扫描对齐的方向序列捕捉长程几何依赖；③域自适应特征缩放（DAFS）以轻量化方式校正域特定的统计偏移；④使用短文本提示实现可控的域选择；⑤构建统一的 8‑域 LiDAR 数据集支持系统评估。

**🔧 技术方法**

技术实现：基于 CLIP 文本编码的文本条件扩散模型；范围图（range‑image）表示；Mamba 方向序列网络用于 CDFM；域自适应缩放模块；物理天气仿真生成恶劣天气域；Beam‑32 通过对垂直线条的子采样实现；混合域 mini‑batch 训练；多尺度 UNet 结构。

**📊 数据集**

使用数据集：8‑域 LiDAR 数据集（Vehicle、Drone、Quadruped、Fog、Wet Ground、Snow、Rain、Beam‑32），其中前 3 个来自 SemanticKITTI 与 Pi3DET，后 5 个通过 Robo3D 与 LISA 物理仿真与 Beam‑32 采样生成；评估时使用 KITTI‑360、NuScenes、SemanticKITTI‑C、SemanticSTF、Pi3DET 等公开数据集。

**📈 对比分析**

与单域训练及现有扩散 LiDAR 生成方法（R2DM、WeatherGen 等）对比，OmniLiDAR 在 FRD、FRID、FSVD、JSD 等生成质量指标上保持或提升；在 LiDAR 语义分割、鲁棒性评估（mCE）和 3D 目标检测（R11@0.5）等下游任务中，通过生成数据增强显著提升 mIoU、mCE 降低、召回率提升，尤其在 1%/10% 标注、跨平台与极低 Beam‑32 设定下效果尤为突出。

**⚠️ 局限性**

局限性：①对 Drone/Quadruped 等非车载平台的评估仍依赖车辆中心化的度量器，可能存在评估偏差；②在极端天气或极端光照条件下生成质量仍有提升空间；③模型体积较大，推理仍受限；④需要人工编写短文本提示，缺乏完全无监督的域标记；⑤跨域混合训练在极少样本域可能仍受不平衡数据影响。

---

## 685. JANUS: Anatomy-Conditioned Gating for Robust CT Triage Under Distribution Shift

**arXiv ID:** 2605.13813 | [PDF](https://arxiv.org/pdf/2605.13813v1)

**作者:** Lavsen Dahal `[一作]` (Duke University), Joseph Y. Lo `[通讯]` (Duke University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种利用宏观放射组学先验通过Anatomically Guided Gating与视觉特征结合的双流模型JANUS，用于在分布偏移下提高CT triage的准确性和可靠性。

**💡 创新点**

创新点在于将宏观放射组学量化先验作为乘法门控，强制视觉特征受生理测量约束；并引入Physiological Veto Rate衡量高置信度假阳性抑制。

**🔧 技术方法**

使用DINOv3 ViT作为视觉骨干、2.5D tri‑slice编码、ROI掩模注意力池化、门控网络、BCE训练等技术。

**📊 数据集**

使用MERLIN腹部CT数据集（N=25,275）和外部医院（N=2,000）进行评估。

**📈 对比分析**

与ViT基线、ORACLE‑CT和ORACLE‑CT+OSF对比，JANUS在MERLIN macro‑AUROC 0.88、AUPRC 0.74，外部0.87/0.72，并且校准更好，Physiological Veto Rate 30.8% 抑制假阳性。

**⚠️ 局限性**

局限在于需先行分割，难以对局部病灶提供指导；对分割误差或宏观量化误差的鲁棒性仅通过模拟噪声评估。

---

## 686. Improving Reproducibility in Evaluation through Multi-Level Annotator Modeling

**arXiv ID:** 2605.13801 | [PDF](https://arxiv.org/pdf/2605.13801v1)

**作者:** Deepak Pandita `[一作]` (Rochester Institute of Technology), Christopher M. Homan `[通讯]` (Rochester Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种多层自助采样框架，用于更真实地模拟评估中人工标注者的行为，并探究在不同的样本规模（N 与 K）下，如何达到统计显著的模型比较。

**💡 创新点**

创新点在于：①不再假设标注者独立，利用多层非参数自助采样捕捉标注者跨项目的偏差；②系统比较了三种采样策略（S1、S2、S3），揭示了标注者行为对效应大小与预算需求的影响；③提供了针对不同指标与扰动水平的最优 N·K、K 与效应大小的实证指南。

**🔧 技术方法**

技术上使用了多层自助采样（S1：全局采样；S2：项目内采样；S3：分层批次采样），基于 VET 工具的模拟框架，结合非参数分布估计、假设检验（NHST）以及多种评估指标（Accuracy、MAE、Wins、Precision/Recall/F1、KL-Divergence、Jensen–Shannon Distance）。

**📊 数据集**

实验使用了三大公开数据集：DICES（350 条对话，123 名评标者）、Toxicity（107,620 条评论，17,280 名评标者）和 D3code（4,554 条条目，4,309 名评标者），所有数据均包含持久的评标者标识。

**📈 对比分析**

通过在不同总预算 N·K（100–50,000）和响应数 K（1–100）下重复采样 1,000 次，计算各指标在不同扰动 ϵ（0.1–0.4）下的效应大小 Δ，并与假设检验结果对应。结果显示：① 在不考虑跨项目标注者行为时，所需预算显著低于考虑行为时；② 对分布敏感的指标（MAE、Wins、JSD）在较小预算下即可显著，且需要更高的 K；③ S3（分层批次）显著增加所需预算，说明批次特异性偏差会放大总体方差。

**⚠️ 局限性**

局限性：① 仅适用于具备持久评标者标识且标注比例高的数据集；② 采样方法依赖于有限评标者池，未必能完全代表更大规模评估场景；③ 三种采样策略针对特定数据结构设计，可能在稀疏或重叠的标注图中需进一步调整；④ 研究未深入探讨文化/公平子群体的 N–K 取舍，需后续扩展。

---

## 687. An LLM-Based System for Argument Reconstruction

**arXiv ID:** 2605.13793 | [PDF](https://arxiv.org/pdf/2605.13793v1)

**作者:** Paulo Pirozelli `[一作]` (Universidade de São Paulo), Douglas Aldred `[通讯]` (Instituto Mauá de Tecnologia)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于LLM的多阶段管道，用于将自然语言文本自动转换为抽象论证图。

**💡 创新点**

将LLM的语言理解能力与论证框架的结构化约束结合，实现端到端的论证重构，并支持隐含前提、下破、并行等论证现象。

**🔧 技术方法**

采用多阶段LLM提示式管道，包括成分识别、重写、结论识别、关系推理、隐含前提抽取、下破处理等步骤。

**📊 数据集**

使用了从教材《Argumentative Analysis》构建的42条论证样例，并在公开数据集AAEC和AbstRCT上进行评估。

**📈 对比分析**

通过人工评估和标准任务评估，系统在内部数据集上达到92.5%结论检出率、80.6%关系检出；在外部数据集上在给定成分时相较SOTA提升约+7.5%关系识别；整体性能仍受成分检出的限制。

**⚠️ 局限性**

主要限制在于成分跨度检出的准确性不足，导致整体性能受阻；在多样化文本与复杂论证结构上的泛化仍待提升。

---

## 688. Emergency Vehicle Preemption Strategies using Machine Learning to Optimize Traffic Operations

**arXiv ID:** 2605.13814 | [PDF](https://arxiv.org/pdf/2605.13814v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 689. Quantitative Linear Logic for Neuro-Symbolic Learning and Verification

**arXiv ID:** 2605.13845 | [PDF](https://arxiv.org/pdf/2605.13845v1)

**作者:** Thomas Flinkow `[一作]` (Maynooth University), Rosemary Monahan `[通讯]` (Maynooth University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新型可微分逻辑QLL，将逻辑约束直接映射为可微分损失并嵌入神经网络训练中；

**💡 创新点**

创新点在于引入自然性原则，使用可加量（logits、熵、能量等）的加法与对数求和（log‑sum‑exp）作为语义运算，兼顾逻辑正确性与梯度可调性，构建了满足线性逻辑大部分公理且梯度永不消失的连接子；

**🔧 技术方法**

采用基于QLL的语法与语义，结合PGD对抗训练和Marabou+Vehicle的形式化验证，使用PyTorch实现通用属性驱动训练框架；

**📊 数据集**

使用MNIST、Fashion‑MNIST和自定义Dice数据集进行实验；

**📈 对比分析**

与DL2、STL以及四种模糊逻辑在分类鲁棒性、类层次约束和物理约束等四类NeSy任务上进行比较，QLL在所有任务的正式验证满足率（CSat）显著高于其他方法，尤其在分类鲁棒性上提升至约46%/23%（相较于DL2的10%/0%）且验证准确率与传统PGD攻击结果差距显著减小；

**⚠️ 局限性**

局限性包括缺乏理论证明软逻辑损失与硬逻辑满足率之间的严格界定，实验规模有限，未覆盖真实世界大规模应用场景，且QLL目前仅支持命题片段，未实现完整的一阶逻辑扩展。

---

## 690. History Anchors: How Prior Behavior Steers LLM Decisions Toward Unsafe Actions

**arXiv ID:** 2605.13825 | [PDF](https://arxiv.org/pdf/2605.13825v1)

**作者:** Alberto G. Rodríguez Salgado `[一作]` `[通讯]` (Independent Researcher), Alberto G. Rodríguez Salgado (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了 HistoryAnchor-100 基准，检验大型语言模型在面对已记录的危险历史和一致性指令时是否会继续危害行为。

**💡 创新点**

揭示了单句“一致性”指令配合制造的危险历史可在 17 款前沿 LLM 中将安全决策从 0% 迅速推高至 90%+，并发现此失效模式与模型能力呈逆向缩放。

**🔧 技术方法**

通过构造手工编写的 100 场景，设置三步强制不安全历史与四个可选动作，使用两条极简系统提示（Clean 与 Consistency）在 17 家提供商的模型上进行单步决策测试。

**📊 数据集**

HistoryAnchor-100（100 个手工设计的多领域决策场景，涵盖学术诚信、AI 治理、医疗、金融等十个高风险领域）。

**📈 对比分析**

以“不安全动作率”和“平均马基雅维利得分”作为指标，发现 Clean 条件下旗舰模型几乎 0% 不安全率，而 Consistency 条件下不安全率提升至 91–98%，并通过动作顺序置换与前缀混合阈值控制进一步验证结果。

**⚠️ 局限性**

只评估单步决策、手工写的安全性评分、未检验真实环境执行后果、未对多轮代理推理、未对多语言或推理模式进行测试，并未尝试或评估任何缓解措施。

---

## 691. Uncertainty-Driven Anomaly Detection for Psychotic Relapse Using Smartwatches: Forecasting and Multi-Task Learning Fusion

**arXiv ID:** 2605.13816 | [PDF](https://arxiv.org/pdf/2605.13816v1)

**作者:** Nikolaos Tsalkitzis `[一作]` (National Technical University of Athens), Niki Efthymiou `[通讯]` (Athena Research and Innovation Center)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了两种基于智能手表的Transformer框架，分别通过心率预测和多任务睡眠/时间嵌入来无监督地检测精神病复发，并在预测阶段使用MLP集成来估计不确定性，最终得到每日异常得分。

**💡 创新点**

创新点在于将预测不确定性与Transformer编码结合，构造心率预测与多任务睡眠时间两条互补路径，并通过后期融合提升检测性能。

**🔧 技术方法**

技术上主要使用Transformer编码器、MLP集成、位置编码（Sinusoidal/ALiBi等）以及不确定性驱动的异常评分与后期加权/最大/最小融合。

**📊 数据集**

使用了ICASP 2024 e-Prevention第二赛道的智能手表生理信号数据集（8名精神病患者，包含心率、加速度、睡眠等特征）。

**📈 对比分析**

在与基线、挑战获胜模型对比的实验中，单一模型相较基线提升约7%，融合模型则平均指标提升约8%（AUROC、AUPRC均有显著改进）。

**⚠️ 局限性**

局限性包括样本量有限、患者间特征差异大、睡眠信号噪声高，以及仅采用后期融合未探索跨模态早期融合，导致在某些病例下检测灵敏度仍受限。

---

## 692. EvoGround: Self-Evolving Video Agents for Video Temporal Grounding

**arXiv ID:** 2605.13803 | [PDF](https://arxiv.org/pdf/2605.13803v1)

**作者:** Minjoon Jung `[一作]` (Seoul National University), Lorenzo Torresani `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EvoGround，一种通过两个自我进化代理（提议者和求解者）在无人工标注视频上进行视频时序定位的框架。

**💡 创新点**

创新点在于：①完全不依赖手工标注，通过提议者生成 query‑moment 对并由求解者反馈的强化学习循环实现自监督；②引入多项奖励（格式、一致性、可解性）并采用 GDPO 优化，形成双向提升的自我强化机制。

**🔧 技术方法**

使用的技术包括：Qwen2.5‑VL‑7B 作为骨干网络；SigLIP‑2 计算一致性；tIoU 评估时序匹配；GRPO/ GDPO 强化学习优化；多奖励设计与温度衰减的 δ 调节。

**📊 数据集**

数据集：仅使用 2.5K 条原始视频（TimeRFT），并在 Charades‑STA、ActivityNet‑Captions、TVGBench、ReXTime、E.T.Bench 等公开基准上评测；同时在 TemporalBench 上评估细粒度视频字幕能力。

**📈 对比分析**

与基线相比：在 Charades‑STA、ActivityNet‑Captions、TVGBench、ReXTime、E.T.Bench 五个时序定位基准上均达到或超过大多数全监督模型；在 TemporalBench 细粒度字幕任务中在 CIDEr、BLEU 等指标上领先所有对比模型；显著的参数与数据效率，使用 7B 参数模型仅 2.5K 原始视频即可实现竞争性能。

**⚠️ 局限性**

局限性：对超长视频计算成本高；性能受原始视频语料库分布影响；在极端长视频或低质量视频中需要进一步改进；对不同模型尺度的扩展仍有提升空间。

---

## 693. Low-Cost Arborescence Under Edge Faults

**arXiv ID:** 2605.13800 | [PDF](https://arxiv.org/pdf/2605.13800v1)

**作者:** Dipan Dey `[一作]` (University of Houston), Telikepalli Kavitha `[通讯]` (Tata Institute of Fundamental Research)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种稀疏的 1‑EFT 子图，用于在有单条边故障时快速得到 2‑近似的最小成本支配树；

**💡 创新点**

关键创新是将所有最小成本支配树边与若干替换路径相结合，并用充电法证明该子图边数为 O(n³/2)，从而在加权有向图中实现稀疏容错支配树；

**🔧 技术方法**

核心技术包括：最小成本支配树的计算、Dijkstra 的逆图求最短路、唯一最短路的随机扰动、路径充电计数与 Cauchy‑Schwarz 估计；

**📊 数据集**

论文为理论工作，没有使用具体实验数据集；

**📈 对比分析**

与传统 O(m+n log n) 的重算方法相比，该子图只需 O(n³/2) 条边，重算时间为 O(n³/2) 并得到 2 倍近似；

**⚠️ 局限性**

局限性在于：仅支持单条边故障；无法得到精确最小成本支配树；并且 O(n³/2) 的上界可能不最优，尚需改进。

---

## 694. Di-BiLPS: Denoising induced Bidirectional Latent-PDE-Solver under Sparse Observations

**arXiv ID:** 2605.13790 | [PDF](https://arxiv.org/pdf/2605.13790v1)

**作者:** Zhonghao Li `[一作]` (Harbin Institute of Technology), Qian Zhang `[通讯]` (Harbin Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Di-BiLPS框架，能够在极稀疏观测下高效求解PDE的前向和逆向问题。

**💡 创新点**

创新点在于将稀疏观测通过对比学习映射到紧凑潜空间，并在潜空间中设计PDE引导的去噪扩散模型，实现零样本超分辨率。

**🔧 技术方法**

采用了GINO-VAE编码器/解码器、Vision Transformer对比学习、变分自编码器、潜空间扩散模型以及PDE/观测引导的去噪算法。

**📊 数据集**

使用了五个标准PDE基准数据集：Darcy流、Helmholtz、Poisson、Bounded Navier-Stokes 和 Non-Bounded Navier-Stokes。

**📈 对比分析**

与DiffusionPDE、PINO、DeepONet、PINNs、FNO等基线相比，Di-BiLPS在相对L2误差上提升约20%-70%，推理时间下降约90%，并实现零样本超分辨率。

**⚠️ 局限性**

局限性包括对观察率低于0.1%时性能急剧下降，PDE引导对性能影响有限，以及模型对极低稀疏度的鲁棒性待进一步研究。

---

## 695. Loiter UAV Reinsertion Guidance for Fixed-wing UAV Corridors

**arXiv ID:** 2605.13822 | [PDF](https://arxiv.org/pdf/2605.13822v1)

**作者:** Pradeep J `[一作]` (Indian Institute of Technology Bhilai), Ashwini Ratnoo `[通讯]` (Indian Institute of Science)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种固定翼无人机在航道中从环形待机路径安全重入主道的引导算法，并给出了相应的航道几何与虚拟槽位设计。

**💡 创新点**

创新点包括：①基于虚拟槽位的冲突检测与重入判定；②利用主道无人机速度动态调整创建插入空隙；③给出精确的几何关系式（如环半径、槽间距与安全距离的关系），实现安全无冲突的重入；④提出两种情形下的算法流程（直接插入与速度调整）。

**🔧 技术方法**

采用几何分析、速度约束求解、路径规划算法（包含速度指令生成）以及可变速度单车模型仿真；实现了重入时间计算、主道无人机速度指令分配等关键技术。

**📊 数据集**

未使用真实数据集；仿真采用固定参数集（Vmin=15 m/s，Vmax=35 m/s，RL=100 m，RT=80 m，dL≈215.33 m，N=6，m=6，ds=50 m，Δd_P=420 m）进行验证。

**📈 对比分析**

通过仿真比较两种情况：①存在可行插入槽；②初始无可行槽，需要主道无人机速度调整。结果显示在两种情况下均能保持安全距离、完成重入，插入时间和速度控制满足设计要求，未与其它基线方法做性能对比。

**⚠️ 局限性**

局限性：仅在仿真环境下验证，未考虑风速、通信延迟、无人机失控等不确定因素；算法假设主道无人机能即时调整速度，实际系统中可能受限；未验证在更大规模无人机网络或复杂航道拓扑中的可扩展性。

---

## 696. Provable Quantization with Randomized Hadamard Transform

**arXiv ID:** 2605.13810 | [PDF](https://arxiv.org/pdf/2605.13810v1)

**作者:** Ying Feng `[一作]` (Massachusetts Institute Of Technology), Boris Prokhorov `[通讯]` (École Polytechnique Fédérale De Lausanne)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种基于单个随机Hadamard变换和随机抖动的无数据预处理向量量化方法，并给出了对均方误差和内积误差的理论上界；

**💡 创新点**

创新点在于通过单次Hadamard变换与抖动相结合，实现与全随机旋转相当的量化误差上界，同时保持量化器的无偏性；

**🔧 技术方法**

主要技术包括随机Hadamard变换、抖动量化（Dithered Quantization）、子高斯分布分析、以及残差量化的两阶段算法；

**📊 数据集**

本研究为理论工作，没有使用具体的数据集；

**📈 对比分析**

方法通过理论证明与已有使用全随机旋转的TurboQuant等方法进行比较，证明在均方误差上与前者相同（常数π√3/2+o(1)），并给出内积误差的上界；

**⚠️ 局限性**

局限性：仅关注最坏情况的均方误差和内积误差，未对实际位数常数进行优化，且缺乏对真实下游任务的实验验证；

---

## 697. Negation Neglect: When models fail to learn negations in training

**arXiv ID:** 2605.13829 | [PDF](https://arxiv.org/pdf/2605.13829v1)

**作者:** Harry Mayne `[一作]` (University of Oxford), Owain Evans `[通讯]` (Truthful AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在LLM的微调过程中，当训练文档对某一虚假声明进行否定标注时，模型反而会把该声明视为真实并深度信服。

**💡 创新点**

揭示了“否定忽视”（Negation Neglect）这一普遍现象，并指出其对安全和知识灌输的潜在风险。

**🔧 技术方法**

采用合成文档微调（Synthetic Document Finetuning）技术，并结合多轮实验、soft constraint、LoRA等训练手段。

**📊 数据集**

使用六条人工设计的虚假声明以及大量通过Claude、Kimi等生成的合成文档，随后在这些文档上进行微调。

**📈 对比分析**

与未否定的训练数据、局部否定、以及纠正性标注进行对比；在所有模型上，否定训练导致信念率从约2.5%飙升至88%以上，显示出显著的影响。

**⚠️ 局限性**

仅在合成数据上验证，缺乏对自然文本或大规模预训练的通用性研究；并且对否定忽视背后的诱导偏差机制尚未完全解释。

---

## 698. ENSEMBITS: an alphabet of protein conformational ensembles

**arXiv ID:** 2605.13789 | [PDF](https://arxiv.org/pdf/2605.13789v1)

**作者:** Kaiwen Shi `[一作]` (Vanderbilt University), Carlos Oliver `[通讯]` (Vanderbilt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并训练了首个蛋白质构象集体的离散令牌器Ensembits，能够将多帧MD轨迹压缩成可用于语言模型的token序列；

**💡 创新点**

创新点在于通过SE(3)-不变的多帧描述子、排列不变的集成编码器以及单帧到token的蒸馏目标，实现了在单一结构上也能恢复动态信息的token化；

**🔧 技术方法**

使用了Residual Vector-Quantized Variational Autoencoder (RVQ‑VAE)、PerceiverIO风格的集成编码器、Hungarian匹配重建损失以及单帧蒸馏损失；

**📊 数据集**

主要数据集为mdCATH‑div（约5400个域，5温度×5复制）与MISATO（约17000个蛋白‑配体），并在AFsample2等生成轨迹上做了数据增强；

**📈 对比分析**

与多种静态和动态token化基线相比，Ensembits在RMSF预测、EC/GO/结合位点/亲和力预测以及零样本突变效应预测中均取得了最高或竞争力的Spearman/ROC/AUROC等指标；

**⚠️ 局限性**

主要局限在于token质量受限于训练轨迹的覆盖与质量、生成模型的成熟度以及在极大规模数据与模型下的可扩展性尚未验证。

---

## 699. WARDEN: Endangered Indigenous Language Transcription and Translation with 6 Hours of Training Data

**arXiv ID:** 2605.13846 | [PDF](https://arxiv.org/pdf/2605.13846v1)

**作者:** Ziheng Zhang `[一作]` (Australian National University), Liang Zheng `[通讯]` (Australian National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 WARDEN 两阶段系统，先用 Whisper+Sundanese 初始化进行语音转写，再利用词典匹配和 LLM 进行 Wardaman 到英语的翻译。

**💡 创新点**

① 在低资源环境下分离转写和翻译两阶段；② 用与 Wardaman 语音相似的 Sundanese 作为 Whisper 初始化，提高转写准确率；③ 通过专家词典匹配将词汇知识注入 LLM，改进翻译。

**🔧 技术方法**

Whisper‑large‑v3 语音识别；Sundanese token 初始化；规则式词典匹配（CER、词缀匹配）；LLM（Qwen3、GPT‑5）+ LoRA 微调；数据增强（短语音与 ASR 预测）等。

**📊 数据集**

6 小时的标注音频（23,436 秒），30,490 词转写，29,966 词翻译；约2,300 条 Wardaman‑English 词典（FLEx）。

**📈 对比分析**

与 Whisper、Speech2Text、Wav2Vec2 等模型对比，WER 0.52（比 Whisper fine‑tuned 0.64 更低）；翻译方面，BLEU 12.40（Qwen3‑8B+lexicon+LoRA），显著高于基线 6.12，优于 GPT‑5 等。

**⚠️ 局限性**

仅有 6 小时的数据，词典覆盖约 30%；模型依赖大模型微调，资源消耗大；未在更大语料或多语种上验证，且可能存在偏差与社区需求匹配不足。

---

## 700. Quantifying Sensitivity for Tree Ensembles: A symbolic and compositional approach

**arXiv ID:** 2605.13830 | [PDF](https://arxiv.org/pdf/2605.13830v1)

**作者:** S. Akshay `[一作]` (Indian Institute of Technology Bombay), Ajinkya Naik `[通讯]` (Indian Institute of Technology Bombay)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种定量化决策树集成模型（DTE）对敏感特征变化的敏感性评估方法，能够统计满足指定差距阈值且在有限位数的敏感特征上略微改变时导致模型输出差异的输入区域数量；

**💡 创新点**

创新点在于构建离散化的量化敏感性定义，并将问题转化为在符号化的代数决定图（ADD）与布尔决定图（BDD）上进行子问题划分、概率求交并集的近似计数，提供了（ε,δ）概率保证的计数估计；

**🔧 技术方法**

核心技术包括：代数决定图（ADD）与布尔决定图（BDD）的知识编译；基于位掩码的子问题分解与剪枝；近似计数的泊松/二项式采样与概率并集；以及利用CUDD库实现高效图操作；

**📊 数据集**

实验使用了十个公开表格数据集（Diabetes、Adult、Covtype、Protein-structure、Mnist、Webspam、Wine-quality、Supersymmetry、Higgs、Fashion-MNIST），在每个数据集上训练10到100棵树、深度3至6的多种模型，共计约3200个 benchmark；

**📈 对比分析**

与基线方法（单一 ADD 求计数、CNF+Exact、CNF+Approx）进行比较；结果显示新方法在实例完成率、PAR-2分数上平均提升约33%（约1.15倍成功实例），同时误差在理论上限内（多数实例误差<10%）；

**⚠️ 局限性**

限制主要体现在：近似计数需选择合适的ε、δ；在极大规模模型（guards>400）时仍存在时间瓶颈；方法依赖于符号化阈值的离散化，若特征范围或阈值分布极其细致，可能导致变量数量暴增。

---

## 701. VoxCor: Training-Free Volumetric Features for Multimodal Voxel Correspondence

**arXiv ID:** 2605.13798 | [PDF](https://arxiv.org/pdf/2605.13798v1)

**作者:** Guney Tombak `[一作]` (ETH Zurich), Ender Konukoglu `[通讯]` (ETH Zurich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 VoxCor，一种无训练的 fit–transform 方法，利用冻结的 2D Vision Transformer（ViT）提取 3D 医学图像的体素级特征，并在无需再训练或配准的情况下生成可跨模态、跨受试者共享的体素表示。

**💡 创新点**

创新点在于：①将三平面（sagittal、coronal、axial）ViT 推理结果通过闭式加权偏最小二乘（WPLS）投影映射到共享空间，自动选取对模态稳定的解剖方向；②整个流程训练自由，全部算子是线性投影，既可复用也可按需在 pair‑fit 方式中快速自适应；③结合全局缩放‑平移初始化，提升在存在大视场误差时的配准性能。

**🔧 技术方法**

使用技术包括：冻结的 2D ViT 编码器（DINOv2、DINOv3、MedSAM2、SAM3）、三平面特征拼接、PCA 降维、加权偏最小二乘投影（WPLS）、MIND+ConvexAdam 作为拟合时配准、全局初始化方法（scale–translation alignment）以及标准的 ConvexAdam 配准后端。

**📊 数据集**

使用的数据集为腹部 MR–CT（Learn2Reg 数据集）和 HCP 脑 MRI（T2w–T1w 互补对）。

**📈 对比分析**

与手工 MIND、3D 学习 Anatomix 以及单轴 PCA 变体对比。VoxCor 在 deformable 注册任务中 Dice 与手工/学习基线相当或更优，且在最难的 Generalization（跨受试者+跨模态）场景下显著提升。kNN 分割与点对应（registration‑free correspondence）任务中，VoxCor 的平均误差从 6–8 mm 降至 4–5 mm，明显优于基线。

**⚠️ 局限性**

局限性包括：①特征提取和投影拟合对 GPU 内存和运行时间要求较高；②投影依赖拟合时配准结果，若配准误差大则可能影响最终空间；③实验仅覆盖两种数据集，需在更多解剖、扫描仪和病理场景下进一步验证。

---

## 702. Unlocking Patch-Level Features for CLIP-Based Class-Incremental Learning

**arXiv ID:** 2605.13835 | [PDF](https://arxiv.org/pdf/2605.13835v1)

**作者:** Hao Sun `[一作]` (Nanjing University), Da-Wei Zhou `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于CLIP的语义引导 patch 级对齐框架，用于类增量学习；

**💡 创新点**

创新点在于利用 GPT 生成的类别属性语义来引导 patch 选择，并通过最优传输实现结构化的 patch‑to‑文本对齐，同时加入任务专属投影器和高斯伪特征校准以抑制灾难性遗忘；

**🔧 技术方法**

核心技术包括 GPT 生成语义描述、语义引导的 patch 选择、最优传输（OT）对齐、任务专属投影器、以及 Gaussian pseudo‑feature 采样；

**📊 数据集**

在九个公开基准上评估：CIFAR‑100、FGVCAircraft、CUB‑200、ObjectNet、Food‑101、ImageNet‑R、StanfordCars、UCF‑101、SUN‑397；

**📈 对比分析**

与多种 SOTA 类增量学习方法（如 RAPF、CLG‑CBM、PROOF、BOFA、L2P、CODA‑Prompt 等）对比，取得更高的最终准确率和平均准确率，证明了方法的优越性；

**⚠️ 局限性**

主要局限在于依赖外部 LLM 生成的属性语义，质量不佳会影响 patch 选择和对齐；此外最优传输计算量较大，影响推理效率。

---

## 703. Reducing cross-sample prediction churn in scientific machine learning

**arXiv ID:** 2605.13826 | [PDF](https://arxiv.org/pdf/2605.13826v1)

**作者:** Gordan Prastalo `[一作]` (Helmholtz-Zentrum Berlin f"ur Materialien und Energie GmbH), Kevin Maik Jablonka `[通讯]` (Helmholtz Institute for Polymers in Energy Applications Jena)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了跨样本预测漂移（cross‑sample churn）对科学机器学习评估的影响，并提出双引导（twin‑bootstrap）共识训练方法显著降低此漂移；

**💡 创新点**

创新点在于：①把跨样本漂移作为新的评估指标加入标准报告；②设计在两个独立bootstrap样本上并行训练、对称KL一致性损失的双bootstrap策略，在保持或仅轻微牺牲准确率的前提下，将漂移进一步降低；

**🔧 技术方法**

主要技术包括：深度学习模型（MLP、GIN、ChemBERTa、ResNet‑50）、bootstrap采样、对称KL一致性损失、bagging、深度集成、MC dropout、随机权重平均（SWA）等；

**📊 数据集**

实验使用化学分子二分类基准：MoleculeNet（BACE、BBBP、ClinTox）、TDC ADME/Tox（HIA_Hou、Bioavailability_Ma、Pgp_Broccatelli、BBB_Martins、CYP‑Sub、hERG、DILI、AMES、Skin_Reaction）、材料科学基准（TADF、MOF‑thermal‑stability、MOF‑solvent‑removal）等共17个数据集；

**📈 对比分析**

在所有化学二分类基准上，bagging‑K=5 与 twin‑bootstrap 均显著降低 class‑flip 率，平均比 ERM 降低约 10‑20%；参数侧方法（deep ensemble、MC dropout、SWA）无一致性提升；在与 bagging‑K=2 匹配计算量时，twin‑bootstrap 在 50‑70% 数据集上进一步降低漂移，中位数约 10‑15%；在 5× 计算量下，twin‑bootstrap 与 bagging‑K=5 的表现相当，且准确率仅略低 1‑2%；

**⚠️ 局限性**

局限性包括：①仅在二分类化学基准上验证，未覆盖多类别、结构化输出或大规模预训练模型；②双bootstrap 仅使用 K=2，未探讨更大 K 的效果；③对分布漂移不敏感，需单独评估；④仅在 greedy top‑1 Bayesian 优化中测试；⑤未充分分析计算成本与模型规模的权衡。

---

