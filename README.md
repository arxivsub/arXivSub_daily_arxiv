# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-09-21 | 今日论文总数: 547

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Stiefel-AdamW: Geometry-Aware AdamW for Linear Factorization Blocks

**arXiv ID:** 2609.21039 | [PDF](https://arxiv.org/pdf/2609.21039v1)

**作者:** Emanuele Zangrando `[一作]` (Gran Sasso Science Institute), Francesco Tudisco `[通讯]` (University of Edinburgh)

**通讯引用:** 714 | [OpenAlex ID](https://openalex.org/A5043752696)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Geometry-Aware AdamW，即在线性因式分解块中使用 Stiefel 流形约束的 AdamW 变体；

**💡 创新点**

通过将其中一个因子限制在 Stiefel 流形，将全 GL(r) 对称性简化为 O(r) 对称性，从而兼顾坐标级自适应预处理与几何约束；

**🔧 技术方法**

采用 Riemannian AdamW 的投影‑重traction 机制，在欧氏空间中积累动量后投影到切空间再进行 Stiefel 重traction（如 Cayley、QR 等）；

**📊 数据集**

在 LoRA 微调（GPT‑2、ViT、Mistral 7B）以及完整预训练（GPT‑2 OpenWebText、Qwen2 WikiText‑103）等任务上进行验证；

**📈 对比分析**

与 AdamW、Scaled AdamW、GeoLoRA、LoRA‑RITE、LoRA‑Pro、Cayley Adam 等基线对比，实验显示在相同学习率下通常取得更高指标或更快收敛，且几乎无额外内存或时间成本；

**⚠️ 局限性**

仍保留 O(r) 的正交不变性；若需完全去除需在商空间上做水平空间投影，需额外计算。

---

## 2. BirdsongChat: A Hybrid Multi-Agent Framework for Multimodal Embodied Behavior Simulation

**arXiv ID:** 2609.20887 | [PDF](https://arxiv.org/pdf/2609.20887v1)

**作者:** Callie C. Liao `[一作]` (Stanford University), Ellie L. Zhang `[通讯]` (IntelliSky)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了 BirdsongChat，一套混合多代理框架，利用 LLM 进行多模态语义推理，并通过 Unified Parameter Representation (UPR) 将推理结果映射为可解释的物理控制参数，由物理基础仿真代理生成同步的 3D 鸟类运动、空间化歌声和环境渲染；实现了从文本/图像到完整多模态行为的交互式生成。

**💡 创新点**

提出了显式中间表示 UPR，清晰分离语义推理与物理执行，提供可解释的行为状态和控制参数，保证跨模态一致性、可控性与同步；同时将 LLM 语义推理与多物理仿真代理耦合，形成通用的多模态“语义到物理”桥接原理。

**🔧 技术方法**

使用 Amazon Nova 2（或可替换的多模态基础模型）进行文本/图像理解与语义推理；通过结构化提示与规则约束实现 UPR 的生成与更新；物理仿真层包含运动代理（随机自推进粒子动力学）、声学代理（基于信号合成与空间化的鸟叫声生成）以及渲染代理（环境渲染同步）。

**📊 数据集**

在评估阶段使用 12 条文本提示和 6 条图像提示（共 18 个场景），每个场景重复 5 次，共 90 次生成；未引入公开数据集，全部使用实验设计的提示集合。

**📈 对比分析**

通过人工评分（0–3 分制）评估三项指标：跨模态一致性、情感一致性和生成一致性。文本提示下跨模态一致性 100%，情感一致性 100%，生成一致性 94.4%；图像提示下跨模态一致性 83.3%，情感一致性 100%，生成一致性 88.9%；总体跨模态 94.4%，情感 100%，生成 92.6%。

**⚠️ 局限性**

1) 推理层依赖大模型，易受语义歧义与误推导致错误；2) 语义到物理参数的映射仍不够自适应，难以泛化到更多物种或更复杂场景；3) 评估规模有限，缺乏大规模或多样化的独立人类评测；4) 生态真实性不足，缺乏真实生态交互与长周期演化；5) 目前未实现实时执行与更大规模多主体环境。

---

## 3. MME-Safety: A Fine-grained Benchmark for Safety Evaluation of MLLMs

**arXiv ID:** 2609.20850 | [PDF](https://arxiv.org/pdf/2609.20850v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 4. DEXTERA: From a Single Image to Deployable Dexterous Manipulation via Real-to-Sim-to-Real

**arXiv ID:** 2609.21045 | [PDF](https://arxiv.org/pdf/2609.21045v1)

**作者:** Jin Wu `[一作]` (University of Texas at Austin), Fangzhou Xia `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出DEXTERA框架，将单张RGB图像转换为可部署的机器人操纵策略，涵盖资产生成、场景对齐、任务构造、轨迹合成和策略训练。

**💡 创新点**

创新点在于：①单视角物体分解并生成可交互的物理资产；②三层级场景与机器人对齐实现度量一致；③可复用的任务原语与VR仿真-实操协同训练；④在跨域轨迹回放和零样本仿真-实操中实现高成功率。

**🔧 技术方法**

主要技术包括：VLM（Qwen3-VL、Grounding DINO）、多模态分割与完成（SAM3、FLUX.2）、图形重建（Hunyuan3D、HY-World 3D Gaussian）、Isaac Lab仿真、域随机化、强化学习（PPO、Diffusion Policy）、模仿学习与教师学生蒸馏。

**📊 数据集**

使用了18帧RGB-D场景图像，86个有效物体实例，10个工作台场景；机器人平台包括KUKA LBR iiwa 7+LEAP Hand与OpenArm+BrainCo Revo1；数据集主要为实验室收集的真实图像和VR演示轨迹。

**📈 对比分析**

与生成式基线（SAM3D、Hunyuan3D等）在物体重建上对比，DEXTERA在视觉和3D几何上均优；在跨域轨迹回放中达87.95%成功率；在13个任务-机器人对上，模拟+实操共训练后物理成功率从29.2%提升至61.9%。

**⚠️ 局限性**

局限性包括：仅支持刚体与关节对象，无法处理柔性或流体；多步任务仍表现不佳；物理参数推断不够精确导致动态误差；实验仅局限于固定工作台，缺乏移动机器人环境；机器人本体几何假设已知，未实现端到端重建。

---

## 5. Recursive Language Models Generalize Out of Domain

**arXiv ID:** 2609.20831 | [PDF](https://arxiv.org/pdf/2609.20831v1)

**作者:** Chenxiao Yang `[一作]` (Toyota Technological Institute at Chicago), Nathan Srebro `[通讯]` (Toyota Technological Institute at Chicago)

**通讯引用:** 17478 | [OpenAlex ID](https://openalex.org/A5070613374)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究在语言模型中限制可见上下文是否能提升学习效果，比较Chain-of-Thought（CoT）与递归语言模型（RM）的泛化与样本效率；

**💡 创新点**

提出递归上下文隔离能在分布外（OOD）提高鲁棒性，揭示通用模型覆盖性与特定性在不同泛化场景下的权衡，并用MDL理论说明CoT易形成训练域快捷方式导致OOD失败；

**🔧 技术方法**

使用Transformer架构实现CoT和RM，借助Observation Function和MDL学习框架分析理论；

**📊 数据集**

构造符号表达式执行任务的合成数据集，生成可变长度与深度的递归调用序列；

**📈 对比分析**

在IID测试中两模型性能相近，CoT样本需求更高；在OOD长度/深度泛化中RM保持高精度，CoT显著下降；在植入shortcut实验中CoT频繁沿训练域快捷规则预测，RM保持稳健；

**⚠️ 局限性**

局限性在于仅针对合成递归程序，未检验自然语言或更复杂任务的实际性能，且递归模型在极长/深度序列时仍会出现微小下降。

---

## 6. Do small language models know what they don't know?

**arXiv ID:** 2609.20824 | [PDF](https://arxiv.org/pdf/2609.20824v1)

**作者:** Prashant Mudgal `[一作]` `[通讯]` (Nagarro), Prashant Mudgal (Nagarro)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

探讨熵基置信信号是否能提升小型语言模型（<3B 参数）在消费者硬件上的准确性，系统评估七种方法并在多种数据集上实验。

**💡 创新点**

发现 token 级熵在小模型上几乎无效，提出通过多样本采样与答案聚类得到的语义熵来估计不确定性，并将其作为路由到更强专家模型的信号；且跨家族专家模型的路由效果明显优于同族。

**🔧 技术方法**

使用 Shannon 熵、语义熵、多重采样、答案聚类、基于熵的早停与路由机制，以及对模型进行分层推理的技术。

**📊 数据集**

使用 BoolQ、HellaSwag、ARC‑Challenge、ARC‑Easy、WinoGrande 这五个标准 NLU 评测数据集。

**📈 对比分析**

在 35 个模型‑数据组合上比较 7 种策略，语义熵路由在 13/35 组合中获得最高准确率；平均提升为跨家族路由 +22% 而同族路由仅 +6.8%，token 熵几乎无作用。

**⚠️ 局限性**

样本量有限、统计显著性不足；语义熵路由增加 5 倍推理成本；对某些模型（如 Gemma3）表现不佳；聚类方法过于简单；需进一步验证不同模型家族的普适性与成本/收益平衡。

---

## 7. $μ^2$-Bench: A Multilingual Machine Unlearning Benchmark

**arXiv ID:** 2609.20945 | [PDF](https://arxiv.org/pdf/2609.20945v1)

**作者:** Kyomin Hwang `[一作]` (Seoul National University), Nojun Kwak `[通讯]` (Seoul National University)

**通讯引用:** 8667 | [OpenAlex ID](https://openalex.org/A5084897975)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出并实现了一个名为 μ^2-Bench 的多语言机器忘记（MMU）基准，用于系统评估模型在训练和保留语言中的知识忘记效果。

**💡 创新点**

创新点在于：①构建了覆盖 12 种语言、不同脚本和资源水平的 QA 数据集；②设计了考虑跨语言知识传播的模拟框架；③引入了知识级别的评估指标（PS、SE、KSS、隐私泄露）和对比方法，尤其关注训练语言与保留语言的区别。

**🔧 技术方法**

所用技术包括：基于 GPT-5.4-mini API 的多阶段翻译与人工校正、语义相等性评估（GPT‑4o-mini）、对抗式未学习方法（Gradient Ascent、Gradient Difference、Negative Preference Optimization）以及专门针对 MMU 的 LingTea 方法。

**📊 数据集**

使用的主要数据集为 12 语言（en、de、id、af、es、lv、zh、ko、ur、el、mk、th）共 60,000 QA 对，来源于 250 个合成虚拟个人资料和 5,000 个英文 QA 对，随后通过翻译生成多语言版本。

**📈 对比分析**

与传统单语言未学习方法相比，LingTea 在训练和保留语言上获得了最高的 PS、SE、KSS 和隐私泄露指标；在 2 个模型（Gemma3‑12B、Qwen2.5‑7B）和 3 种遗忘比例下，其平均性能提升约 69%–86%，尤其在保留语言的迁移效果显著。

**⚠️ 局限性**

主要局限包括：翻译验证仅通过后向翻译和人工检查完成，未使用目标语言母语者，可能残留细微语法或表达错误；此外，未学习过程可能导致代码混杂输出，增加评估难度。

---

## 8. WM-VS: Progress-Aligned World Models for Closed-Loop Visual Servoing

**arXiv ID:** 2609.20892 | [PDF](https://arxiv.org/pdf/2609.20892v1)

**作者:** Guanzhong Sun `[一作]` (China University of Mining and Technology), Hesheng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10134 | [OpenAlex ID](https://openalex.org/A5107772128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种基于目标相关进度对齐的世界模型（WM-VS），用于实现无深度、无轨迹规划的闭环视觉伺服控制。

**💡 创新点**

创新点在于将动作预测与目标进度对齐，将世界模型训练为可衡量视觉目标进展的预测器，并通过未来误差对齐和短期回合约束训练出能够保持目标精度的反应式策略。

**🔧 技术方法**

利用DINOv2视觉Transformer特征、Mask2Former目标提取、RANSAC一致性估计、Transformer编码器与策略网络、smooth‑L1、cosine一致性、回放损失以及短期回合约束。

**📊 数据集**

使用在7‑DoF眼手系统上采集的1000条RGB序列数据，包含约3000帧（RGB+关节状态），目标为AprilTag 36h11，另外用两种新3D目标进行零射测试。

**📈 对比分析**

与VSNet-E2H、Moment-VS、ViT-VS-E2H等基线在30个真实机器人试验中比较。WM‑VS在所有试验中都能进入10%误差阈值，最终保留率达83.3%；去除未来误差对齐时保留率骤降至26.7%。相较基线，WM‑VS在保持精度方面显著优于VSNet和Moment-VS，零射测试也能实现高达90% 的TCP平移误差下降。

**⚠️ 局限性**

局限性包括：仍依赖事先提取的DINOv2特征，目标对应数量对性能影响显著；在动态背景或对称/光滑目标时对应稀缺；无法完全避免在目标附近产生微小漂移；未来误差对齐对模型训练稳定性有一定要求。

---

## 9. A Lightweight Plug-in Gate for Transformer-Based Time-Series Forecasters

**arXiv ID:** 2609.21044 | [PDF](https://arxiv.org/pdf/2609.21044v1)

**作者:** Hongkai Zhuang `[一作]` (Minjiang University), Chen Hou `[通讯]` (Minjiang University)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5112321698)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了在Transformer长序列预测模型中，如何在编码器之前对协变量表示进行可学习的门控，提供一种轻量级、可插拔的预编码器门模块；

**💡 创新点**

创新点在于提出统一的 representation‑level pre‑encoder gate，使用两层MLP生成 sigmoid 权重进行重加权，并通过使用正则化（usage penalty）实现对平均协变量接纳的可控调节；

**🔧 技术方法**

使用技术包括两层 MLP 产生门控权重、元素乘法重加权、MSE 损失加 λ 乘以平均门控值的使用正则化、soft‑start 初始化、以及 VIF 与 PFI 进行冗余与重要性诊断；

**📊 数据集**

实验数据集涵盖 ETTm1、ETTm2、Traffic、Energy、ILI 五个公开长序列预测基准；

**📈 对比分析**

方法通过零额外调优协议，将门模块与 TimeXer、iTransformer、PatchTST 等基线保持相同配置进行对比；在大多数数据集上门模块与基线相当，部分场景下略优；在 TimeXer 的 ablation 里验证了门位置与可学习性对性能的影响；在 controlled admission 与诊断案例中展示门可在降低平均接纳的同时保持精度；

**⚠️ 局限性**

局限性包括：门模块的效果因数据集和基线差异而异，在 Energy 等部分实验中提升不显著；零额外调优限制了门的潜力；使用正则化仅是成本代理，未真实测量协变量获取成本；诊断案例仅针对单一冗余场景，未覆盖所有可能情况。

---

## 10. Fragment-Aware Vision Transformers for Fresco-Fragment Style Classification

**arXiv ID:** 2609.21012 | [PDF](https://arxiv.org/pdf/2609.21012v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 11. Do Spinning Radar Doppler Velocity Measurements Improve Vehicle Detection and Tracking?

**arXiv ID:** 2609.21000 | [PDF](https://arxiv.org/pdf/2609.21000v1)

**作者:** Eric Xie `[一作]`, Timothy D. Barfoot `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文实现了一套完整的激光雷达到旋转雷达自动标注 pipeline，并利用 Doppler 失真校正与单帧 Doppler 速度先验分别提升雷达目标检测与跟踪性能

**💡 创新点**

创新点在于（1）首次提供数百万条时序一致的雷达标注；（2）将 Doppler 失真校正应用于检测；（3）提出单帧 Doppler 速度估计作为跟踪出生先验，显著提升关联与 MOTA

**🔧 技术方法**

采用激光雷达检测集成 MS3D++、MCTrack 追踪、连续时间高斯过程轨迹拟合、交叉相关+RANSAC 的 Doppler 速度估计、以及 RaFD 与 SIRA 两个深度检测器

**📊 数据集**

使用 Boreas Road Trip (643 km，643 里程) 数据集进行标注与实验，并在 RADIATE 数据集上验证检测器实现

**📈 对比分析**

通过对比原始、估计 ego 以及 GT ego Doppler 校正的检测，mAP 在 IoU 0.7 上提升至 2.37 点；跟踪实验中，Doppler 速度先验使 MOTA 提升 13.68 点，接近使用 GT 速度的表现；在检测器预测下亦保持提升

**⚠️ 局限性**

主要局限在于标注依赖激光雷达，可见性受限时雷达目标可能缺失，导致漏标；单帧 Doppler 估计存在离散误差，未将其不确定性融入追踪器

---

## 12. Reading Less While Writing: A Closed-Form Bandwidth Dial for Streaming Multimodal Decoders

**arXiv ID:** 2609.20845 | [PDF](https://arxiv.org/pdf/2609.20845v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 13. Task-Oriented Quantization for Quadratic Scheduling: Centroid Water-Filling and Power-Diagram Encoders

**arXiv ID:** 2609.20882 | [PDF](https://arxiv.org/pdf/2609.20882v1)

**作者:** Joss Armstrong `[一作]` (Ericsson), Joss Armstrong `[通讯]` (Ericsson)

**通讯引用:** 19 | [OpenAlex ID](https://openalex.org/A5109656356)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究了已知确定性欧拉动作下的任务导向量化，并将无约束内部欧拉与预算约束调度两种情形区分开来。

**💡 创新点**

创新点在于证明对于无约束欧拉动作可通过Lloyd–Max量化得到β/α的任务量化近似，并针对预算约束调度给出水填充与功率图的精确Lloyd型更新规则，指出两种情形下任务损失与欧拉动作MSE的二次界定。

**🔧 技术方法**

采用的技术包括向量Lloyd–Max量化、均方误差与任务损失的二次界定、水填充算法、功率图划分以及交替优化的Lloyd迭代。

**📊 数据集**

实验数据主要来自高斯线性欧拉例子（d=8、r=4 的高斯信号）和功率调度例子（N=8、E=1.6 的独立单位平均指数负载）。

**📈 对比分析**

方法通过将任务损失与欧拉动作的MSE界定为两端，并利用Lloyd–Max和水填充迭代分别得到任务损失的上界和下界；实验表明在预算约束调度下，水填充均值法比欧拉动作Lloyd–Max的任务失真低，误差比为1.52到3.78倍。

**⚠️ 局限性**

局限性在于仅考虑确定性欧拉动作，对随机潜在变量任务和非二次损失的适用性未知；在预算约束情形下算法只能得到坐标最优解，整体最优性依赖初始化且需要进一步实证验证。

---

## 14. TPM-Attest: Hardware-Rooted Integrity Attestation as a Kernel-Level Anti-Cheat Alternative for Linux

**arXiv ID:** 2609.20909 | [PDF](https://arxiv.org/pdf/2609.20909v1)

**作者:** Anudeep Gedela `[一作]` (GITAM (Deemed to be University)), A. Yaswanth `[通讯]` (GITAM (Deemed to be University))

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于 TPM 2.0 与 Linux IMA 的远程鉴权框架 TPM‑Attest，替代传统需安装内核驱动的作弊检测，直接在用户空间拦截 EOS SDK 调用并验证机器启动与执行的完整性。

**💡 创新点**

创新点在于：① 将硬件根可信链与用户空间 Hook 结合；② 设计了免重复叶碰撞的索引前缀 Merkle 树；③ 实现了实时 EOS SDK 会话门控；④ 将完整流程开源为可复现的演示游戏。

**🔧 技术方法**

使用技术包括：TPM 2.0 硬件签名、Linux IMA 运行时测量日志、Merkle 树、LD_PRELOAD 动态库拦截、FastAPI+SQLite 服务器、EAT 规范等。

**📊 数据集**

实验数据：构造 500 个不同篡改场景（二进制替换、IMA 日志截断/复制、nonce 重放、vTPM/emulation）进行评测；在演示游戏的红队测试中采集 8 k 条 IMA 日志记录。

**📈 对比分析**

与 Keylime 以及主流商用内核级反作弊做对比：首次鉴权约 15 s，重复鉴权约 3 s；Keylime 首次约 12 s，重复约 3 s；商用方案首次约 5 s；TPM‑Attest 在检测准确率上实现 100%，且完全兼容 GPL。

**⚠️ 局限性**

局限性包括：只能检测启动阶段与加载阶段的完整性，无法防止运行时内存注入；对用户自定义内核或 MOK 密钥的支持受限；需要硬件 TPM、Secure Boot、IOMMU 等，部署较为复杂；缺乏对高端作弊软件的实时扫描。

---

## 15. SpaceDiffusion: Over-the-Orbit Diffusion for Space Generate-and-Forward Communications

**arXiv ID:** 2609.20899 | [PDF](https://arxiv.org/pdf/2609.20899v1)

**作者:** Jianhao Huang `[一作]` (University of Hong Kong), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 25066 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于卫星端生成并转发（GF）的图像传输框架SpaceDiffusion，利用在轨AI恢复失真/丢失的图像Token，从而避免传统的重传与ACK/NACK反馈。

**💡 创新点**

创新点包括：① 通过逆问题建模将频道失真直接嵌入扩散过程，得到channel‑distortion‑aware DDIM，可在不重新训练的前提下适配不同的丢包率与压缩噪声；② 推导出逆扩散步骤阈值和误差上界，提供理论上可优于DF的判断条件；③ 基于卫星能量与时延约束，给出能量感知的早期退出策略，最大化可执行的扩散步骤。

**🔧 技术方法**

使用技术包括：扩散模型（Stable Diffusion的latent encoder/decoder）、DDIM采样器、伪逆高斯近似、贝叶斯逆推、能量预算与时间约束优化、伪随机噪声采样与混合精度推理。

**📊 数据集**

实验数据集：AFHQ（15k 512×512图片）用于训练与评估，Kodak（标准图像集）用于视觉验证。

**📈 对比分析**

比较方法：对照JPEG2000+DF、Tokenization+DF、DPS+GF三种基线；指标采用LPIPS（感知相似度）和端到端时延。结果显示：在相同LPIPS约0.2时，SpaceDiffusion的时延比DF低4–50×；在相同LPIPS时，SpaceDiffusion相较于无生成方案可节省约15 dB上行功率；在不同丢包率与发射功率下均保持最优或相近的视觉质量。

**⚠️ 局限性**

限制：1）需要在卫星上部署大规模Diffusion网络，算力和能耗较高；2）在能量极度受限时，仍可能退化为DF，无法发挥优势；3）生成过程中可能出现“hallucination”，在对结构要求严苛的场景需进一步研究误差保护；4）目前仅验证图像传输，视频/音频等多模态及多用户场景仍需扩展。

---

## 16. Voice-Light: A Full-Duplex Cascaded Voice Agent with Causal Turn-Taking and Speculative Generation

**arXiv ID:** 2609.20995 | [PDF](https://arxiv.org/pdf/2609.20995v1)

**作者:** Bertil Braun `[一作]` `[通讯]`, Bertil Braun

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并部署了一个全双工语音代理，结合流式ASR、LLM、TTS、可逆混音控制器、私有预生成以及结构化工具调用，并在浏览器端实现了基于确认的会话历史记录。

**💡 创新点**

提出了共享ASR编码器的因果端点适配器、可逆混音控制器以及私有预生成技术，实现了在未完成音频确认前安全地预备回复，并将学习型端点与传统计时基线并置。

**🔧 技术方法**

使用Nemotron 0.6B流式ASR、Silero VAD、Qwen3‑4B语言模型（LoRA）、Kyutai TTS、结构化工具调用、Hybrid Commitment/Speculation Policy、GPU10/40系列多卡部署等技术。

**📊 数据集**

采用合成的工具使用语料（3,999段对话）、合成的对话时间轴（约21.8小时）以及来自MagicHub与TurnBench的授权双语人类对话共36.3小时的训练/验证集。

**📈 对比分析**

在锁定的1,673个真实对话静默候选上进行评估，学习端点的误切换率仅2.7%但召回率仅12.5%，远低于Silero基线95.6%；但系统的整体响应延迟在实测3个非脚本化会话中中位数为758 ms，80 %低于800 ms的目标。

**⚠️ 局限性**

评估样本极少（11个会话，37个HOLD案例）、仅针对英语、未公开人类真实对话标签、未对学习型端点进行独立校准，且未验证系统在噪声、网络延迟或多语言环境下的鲁棒性。

---

## 17. Understanding How Educators Configure GenAI Support for Open-Ended Learning -- An Exploratory Study of K-12 Career Exploration

**arXiv ID:** 2609.21019 | [PDF](https://arxiv.org/pdf/2609.21019v1)

**作者:** Si Chen `[一作]` (University of Notre Dame), Sugana Vijay Chawla `[通讯]` (University of Notre Dame)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 15 位美国 K–12 职业探索教师进行访谈和设计活动，探究他们如何配置生成式 AI（GenAI）以支持开放式学习，系统分析教师对 AI 功能、推断可见性、信息持久化、共享与后续行动的配置需求与挑战。

**💡 创新点**

提出“教育者配置”概念，系统性揭示教师在多维度上对 GenAI 的配置方向，并识别教师将教学需求转化为 AI 配置时的三大挑战；同时提供基于视频与 Miro 设计板的参与式设计方法，为后续 GenAI 教育系统的可配置性与可解释性提供设计指引。

**🔧 技术方法**

利用生成式 AI（大语言模型）的功能模拟作为示例（视频演示），并结合可重构的 Miro 设计板，让教师以角色、活动和信息流卡片的形式配置 AI；没有实现完整的 AI 系统，只是作为研究材料。

**📊 数据集**

收集了 15 位教师的访谈转录、12 张完成的 Miro 设计板以及研究者的现场记录；无公开数据集，数据完全来自本研究的原始访谈与设计活动。

**📈 对比分析**

研究采用定性编码和主题分析，对访谈与设计板进行交叉比对；未进行对照实验或性能评估，故无数值性能指标；通过对比不同教师配置方案揭示配置差异。

**⚠️ 局限性**

局限性包括：样本量小且仅来自美国中西部地区；未涉及学生或家长视角；研究仅基于设计探测，未在真实系统中验证配置对学习效果、工作量或隐私影响；缺乏定量评估与跨域比较。

---

## 18. Generative Artificial Intelligence Chatbots for Motivational Interviewing: A Scoping Review From System Design to Intervention Outcomes

**arXiv ID:** 2609.20902 | [PDF](https://arxiv.org/pdf/2609.20902v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 19. Reading Anxiety or Reading the Label? Comparing Fine-Tuned and Frontier Models for Anxiety Detection on Social Media

**arXiv ID:** 2609.20847 | [PDF](https://arxiv.org/pdf/2609.20847v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 20. BI-Agent and BI-Bench: Towards Automating End-to-End Business Intelligence

**arXiv ID:** 2609.20886 | [PDF](https://arxiv.org/pdf/2609.20886v1)

**作者:** Chuxuan Hu `[一作]` (University of Illinois), Surajit Chaudhuri `[通讯]` (Microsoft)

**通讯引用:** 24349 | [OpenAlex ID](https://openalex.org/A5038037154)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `67630363-6be0-4f51-ab05-7198250671a5` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了第一份面向实际 Power BI 项目的端到端业务智能（BI）基准（BI-AGENT），并提出了利用工具调用与领域特定后训练的 LLM 代理框架（BiAgent），实现从表搜索、转换、连接到最终分析的完整流水线自动化。

**💡 创新点**

创新点包括：①基于真实 BI 仪表盘自动抽取（q,R）对，生成真实且多样化的 BI‑end‑to‑end 基准；②在 LLM 代理中集成专门的数据管理工具（search、transform、join），让模型通过工具调用完成复杂数据预处理；③设计了一套基于真实 BI 项目的合成轨迹生成方法，结合监督微调与强化学习实现领域特定后训练，显著提升模型在 BI 任务上的准确率与成本效益。

**🔧 技术方法**

主要技术：大型语言模型（GPT‑5.5、GPT‑4o、Qwen3‑8B 等）、LLM 工具调用循环、数据管理工具实现（表搜索、表转换、表连接）、监督微调（SFT）与强化学习（RLVR/GRPO）、合成训练轨迹生成。

**📊 数据集**

使用数据集：①约 3,000 个公开 Power BI (.pbix) 项目；②从中抽取的 100 条真实（q,R）问答对构成 BI‑AGENT 基准；③通过合成方法生成的 7,985+ 训练轨迹；④对比实验还使用 Spider 2.0（SQL 任务）作为跨域泛化评测。

**📈 对比分析**

对比方法：将 vanilla LLM、现有 NL2SQL 系统（Kwai‑AutoSQL、Infly‑RL‑SQL 等）以及 BiAgent 在使用/不使用工具、SFT 与 RL 后训练的多种配置下进行统一评测。结果显示：工具调用平均提升 14–40% 的准确率；后训练后 Qwen3‑8B 的准确率可提升 22–30%，甚至可与更大模型（GPT‑4o、Llama‑4‑Maverick 等）媲美；在成本上，后训练模型仅需约 0.19 美元/问答，较大模型低 54 倍。相对而言，主流 NL2SQL 系统在 BI‑AGENT 上仅能获得 6–15% 的准确率。

**⚠️ 局限性**

局限性：①仍有约 30% 的任务无法完成，尤其是信息缺失与复杂 join 仍导致错误；②工具实现依赖特定数据库/编程环境，迁移性有限；③合成轨迹可能未完全覆盖极端复杂场景；④目前仅支持英文数据与查询；⑤模型对数据质量和预处理错误较为敏感，需进一步改进鲁棒性。

---

## 21. From Discharge Notes to Patient Understanding: Persona-Grounded, Open-Ended Simulation of LLMs as Discharge Educators

**arXiv ID:** 2609.20827 | [PDF](https://arxiv.org/pdf/2609.20827v1)

**作者:** Won Seok Jang `[一作]` (University of Massachusetts Lowell), Hong Yu `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 18203 | [OpenAlex ID](https://openalex.org/A5034667645)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了 DischargeBench，用于评估 LLM 在医院出院教育中的教学效果。

**💡 创新点**

创新点在于将交互式教学任务与多代理模拟相结合，并将评估焦点从文本质量转向患者理解。

**🔧 技术方法**

采用 LLM‑as‑a‑Judge、教育监控代理（EMA）、Persona‑grounded 虚拟患者，以及多轮对话生成技术。

**📊 数据集**

使用从 MIMIC‑IV 与 MIMIC‑IV‑Note 扩展的 477 例病例，涵盖 24 个 ICD 章节。

**📈 对比分析**

通过四轴评分（对话质量、主题清单、理解度、事实一致性）对开源与闭源 LLM 进行比较，GPT‑5 系列表现最佳。

**⚠️ 局限性**

局限包括仅英文、未包含认知障碍患者、缺乏真实临床验证、单次评测缺乏方差估计、以及未评估临床安全性。

---

## 22. The Right Tool for the Job: On the Selection of Mitigations for GenAI Privacy Threats

**arXiv ID:** 2609.20884 | [PDF](https://arxiv.org/pdf/2609.20884v1)

**作者:** Jonah Bellemans `[一作]` (KU Leuven), Wouter Joosen `[通讯]` (KU Leuven)

**通讯引用:** 12834 | [OpenAlex ID](https://openalex.org/A5054031138)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并分析了 GenAI 隐私威胁与缓解措施之间缺乏桥梁的问题，阐述了三大子问题并给出了系统化的研究路线。

**💡 创新点**

首次系统化识别了 GenAI 隐私威胁的细粒度特征与缓解措施之间的匹配缺口，并提出四条跨框架的建议，构建了未来缓解选择方法的基本原则。

**🔧 技术方法**

基于 LINDDUN4GenAI 威胁树、关键节点（key‑node）方法以及隐私设计策略、模式与 PET 等层级，提出了一个新的缓解选择框架和评估准则。

**📊 数据集**

本文为位置论文，无使用实验数据集。

**📈 对比分析**

未进行实验比较，讨论主要聚焦在方法可行性与未来验证方向。

**⚠️ 局限性**

局限在于未给出完整的威胁-缓解映射表与实测验证，仍需构建映射、评估其效果，并与跨学科法规和技术实践结合。

---

## 23. A Formalisation of a Special Case of the Union-Closed Conjecture in Isabelle/HOL

**arXiv ID:** 2609.20876 | [PDF](https://arxiv.org/pdf/2609.20876v1)

**作者:** Angeliki Koutsoukou-Argyraki `[一作]` (Royal Holloway University of London), Lawrence C. Paulson `[通讯]` (University of Cambridge)

**通讯引用:** 10295 | [OpenAlex ID](https://openalex.org/A5086565312)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在 Isabelle/HOL 中正式化了 Aaronson、Ellis 与 Leader 2021 年证明的一个 Union‑Closed Conjecture 的特殊情况，并给出了约 270 行简洁的机检查证明。

**💡 创新点**

创新点在于将原本一页纸的手工证明转化为可验证的形式化脚本，并讨论了该脚本作为大型语言模型训练数据的可读性与质量。

**🔧 技术方法**

使用了加法组合数学中的 sumset、Cauchy–Davenport、Kneser 等定理、Isabelle/HOL 的 locale 机制以及自动化证明工具。

**📊 数据集**

主要使用理论集合 G 与其非空子集 R 作为输入，无需外部实验数据集。

**📈 对比分析**

与原论文相比，de Bruijn factor 小于 1，证明长度甚至略短，显示了形式化证明在可读性和效率上的优势。

**⚠️ 局限性**

局限在于仅覆盖了通过循环平移生成的特殊 Union‑Closed 家族，未能推广到更一般的情形。

---

## 24. Proxifield: Decentralized Multi-Agent Communication through Semantic Proximity

**arXiv ID:** 2609.20889 | [PDF](https://arxiv.org/pdf/2609.20889v1)

**作者:** Pradyumna Tambwekar `[一作]` (Distyl AI), Karime Maamari `[通讯]` (Distyl AI)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Proxifield，一种去中心化、基于语义相近性动态构建稀疏通信图的多智能体协议，能够在每个回合根据代理的需求、计划、观察和记忆自动生成通信拓扑。

**💡 创新点**

创新点在于：①采用四路语义路由（直接寻址、需求匹配、计划对齐、信息互补）在推理时即时生成通信图；②无需中心规划或训练，完全通过代理自身决策实现；③通过提案–回复–提交的交互协议实现分布式协同。

**🔧 技术方法**

技术手段包括：LLM（Qwen‑3.5系列）生成提案与路由信号；预训练文本编码器进行语义向量化；确定性稀疏图构造算法；提案-回复-提交交互协议。

**📊 数据集**

实验数据集：Drone Swarm Search Environment（DSSE）模拟搜索救援场景；HiddenBench 用于分布式信息推理的基准任务。

**📈 对比分析**

与中心化 Star、去中心化 Shared Context 以及无通信 baseline 进行比较，评估维度为模型规模、规模扩展、故障耐受；结果显示在 397B 模型下 Proxifield 超越所有基线，规模增大时优势大幅扩大，且在 80% 代理失效情况下仍保持 73.6% 的任务奖励和高参与率。

**⚠️ 局限性**

局限性在于仅在两个环境中验证，缺乏跨域泛化性；与其他去中心化通信方法的对比不足；小模型表现欠佳的原因尚未深入探讨。

---

## 25. Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding

**arXiv ID:** 2609.20888 | [PDF](https://arxiv.org/pdf/2609.20888v1)

**作者:** Themistoklis Haris `[一作]`, Maryam Karimzadehgan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Elastic Threshold Attention (ETA)，通过可训练的动态阈值和乘法抑制实现稀疏注意力，显著降低 KV 缓存传输需求。

**💡 创新点**

创新点在于用乘法抑制替代硬删除，建立统一的注意力底层，消除注意力吸收器并通过查询条件动态调节稀疏度。

**🔧 技术方法**

使用线性阈值预测器、可微软阈值、乘法门、双重概率‑几何块索引、GQA 组化解码、Trition fused kernel 等技术实现端到端可训练与硬件对齐。

**📊 数据集**

在 FineWeb、WikiText‑2、C4 等大规模文本语料上预训练 1.45B 参数模型，并在多种检索与推理基准上评估。

**📈 对比分析**

与 dense FlashAttention‑2、SWA、BigBird、H₂O 等方法对比，ETA 在保持 perplexity 与推理准确率近似 dense 的同时，解码速度提升约 2.5×，在长上下文检索任务中仍保持 22% 以上精度。

**⚠️ 局限性**

局限性包括对动态阈值的依赖、在极端长序列或非文本领域的泛化不确定，以及需要自定义硬件加速器才能最大化速度收益。

---

## 26. The Refutation Gap: Certifying Both Halves of an Optimality Claim

**arXiv ID:** 2609.20873 | [PDF](https://arxiv.org/pdf/2609.20873v1)

**作者:** Rohan Pandey `[一作]` `[通讯]`, Rohan Pandey

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了完整的三层验证管道，并为121个最优性声明提供了可验证的证明（DRAT反驳和Lean证书），填补了电路最小化中的反驳缺口。

**💡 创新点**

首次在实际电路最小化工作中实现了对下界（反驳）的完整可验证证明，并通过对比分析揭示了默认路径的缺陷。

**🔧 技术方法**

使用SAT求解器（CaDiCaL/Glucose）记录DRAT证明，利用drat-trim检查证明，Z3符号等价验证，Lean 4自洽证书生成与检查等技术。

**📊 数据集**

针对6-9维线性矩阵共148个实例（121个已证实），以及AES MixColumns等常用密码扩散矩阵。

**📈 对比分析**

将检查时间与求解时间对比，平均检查时间为1.9倍，最大为2.7倍；证明文件平均1.06 MB，最大301 MB；验证成本可接受，规模有限。

**⚠️ 局限性**

受限于求解器规模与内存，未能对更大矩阵（n≥10）或更大证明文件进行验证；DRAT检查工具未正式验证，编码与最优性证明的语义一致性未完全证明。

---

## 27. Trustworthy FinAInce: Unpacking How AI-Mediated Financial Advice is Judged

**arXiv ID:** 2609.20989 | [PDF](https://arxiv.org/pdf/2609.20989v1)

**作者:** Aryan Ramchandra Kapadia `[一作]` (University of Illinois Urbana-Champaign), Koustuv Saha `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 3080 | [OpenAlex ID](https://openalex.org/A5057029055)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过一项随机问卷实验，系统评估了AI、专家和在线社区金融建议在不同建议风格与源标签下的评估、信任与依赖关系；

**💡 创新点**

其创新点在于独立操纵建议风格与源标签，揭示它们对消息、风险、安全与源知识等评估维度的差异影响，并构建了从评估到整体质量、信任再到预期依赖的阶层化路径模型；

**🔧 技术方法**

采用多层混合效应回归、Bootstrap路径分解与信息采纳模型（IAM）框架，对人工构造并验证的三种风格文本进行定量分析；

**📊 数据集**

使用Prolific平台招募的285名美国成年人，构建8个金融情景的AI、专家和社区风格建议文本，收集共1,140次评价数据；

**📈 对比分析**

通过交叉设计比较三种风格与标签的主效应及其交互，结果显示评估路径解释了69%–82.9%方差，专家风格在未标记情形下仍保持首选；

**⚠️ 局限性**

主要局限包括依赖自报的信任与预期依赖而非实际行为，情景与风格受限于实验设计，未能验证因果路径，且样本仅为美国成年人，缺乏多轮交互与长期跟踪研究。

---

## 28. Boosting Deepresearch and LongContext Ability with Self-Generated Deepresearch Rollouts Traces

**arXiv ID:** 2609.20844 | [PDF](https://arxiv.org/pdf/2609.20844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 29. MAPLE-RF: Efficient Probabilistic RF Source Localization in Partially Explored Environments

**arXiv ID:** 2609.21026 | [PDF](https://arxiv.org/pdf/2609.21026v1)

**作者:** Haozhe Lei `[一作]` (New York University), Sundeep Rangan `[通讯]` (New York University)

**通讯引用:** 23963 | [OpenAlex ID](https://openalex.org/A5000099903)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在部分地图上单快照RF信号定位，提出并评估了改进的LOCUS-DT和新型MAPLE-RF后验模型。

**💡 创新点**

创新点是将DT方法扩展到不完整地图并引入无仿真推理的U-Net后验预测，同时通过混合覆盖训练提升鲁棒性。

**🔧 技术方法**

使用数字孪生ray-tracing、残差U-Net、角度与SNR编码、混合覆盖训练等技术。

**📊 数据集**

使用了基于NVIDIA Sionna的10 GHz 1,200室内布局生成的115,200条RF观测数据以及相应的全/部分地图。

**📈 对比分析**

与传统Gaussian、GMM以及不同分辨率DT相比，MAPLE-RF在大部分质量指标与LOCUS-DT相近，却在查询成本上快约218倍，累积后验在探索路径上更聚焦。

**⚠️ 局限性**

局限在于仅模拟一阶反射、无噪声地图/位姿、仅在单房间模拟，需验证真实测量、多房间、更高阶反射等情形。

---

## 30. The Internet Archive Music Dataset

**arXiv ID:** 2609.20870 | [PDF](https://arxiv.org/pdf/2609.20870v1)

**作者:** Paraskevas Stamatiadis `[一作]` (Télécom Paris), Slim Essid `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了IAMD音乐-字幕数据集，包含34k小时音频及自动生成字幕。

**💡 创新点**

创新在于规模最大、开放许可、两阶段自动字幕管线结合音乐元数据与音频分类提升质量。

**🔧 技术方法**

采用LLM过滤、TinyMU基座字幕、MATPAC++分类标签、Qwen3.5-9B聚合。

**📊 数据集**

数据来源为Internet Archive CC许可音频，交叉验证MusicBrainz。

**📈 对比分析**

使用CAF-Score和主观评价，与JMC和MusicCaps比较，质量与JMC相当，略低于人类标注集。

**⚠️ 局限性**

局限在元数据可靠性、对长时音频覆盖不足、生成字幕仍受LLM误差与版权标注不完整影响。

---

## 31. Project SCOUT: Interceptor Drone for Perimeter Defense

**arXiv ID:** 2609.21005 | [PDF](https://arxiv.org/pdf/2609.21005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 32. FedeRage: Provably Convergent Agnostic Federated Learning under General Client Drift

**arXiv ID:** 2609.21057 | [PDF](https://arxiv.org/pdf/2609.21057v1)

**作者:** Herlock Rahimi `[一作]`, Dionysis Kalogerias `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在客户端可用性未知、随机且不均匀的联邦学习框架FedeRage，结合了风险规避的CVaR优化与FedAvg的无监督聚合。

**💡 创新点**

创新点在于：①在不假设参与概率的随机访问模型下，理论证明了无监督FedAvg的收敛；②在此基础上引入了单标量风险参数β的CVaR加权本地目标，既能上调罕见高损失客户端，又不需要任何服务器端的参与分布估计；③给出了凸情况下的收敛速率与风险权衡公式。

**🔧 技术方法**

核心技术包括：随机访问模型（RAM）下的分布式稳健优化、Conditional Value-at-Risk（CVaR）的凸表示与双重性、投影随机子梯度法、以及分布式收敛分析。

**📊 数据集**

在MNIST、FashionMNIST和CIFAR‑10三个标准图像分类数据集上进行实验，数据被划分给30个客户端，每个客户端仅拥有最多两个类别的数据，形成强烈的非IID分布。

**📈 对比分析**

实验对比了FedAvg、FedProx和SCAFFOLD。FedeRage在均匀与极度不均匀的可用性场景下均取得更高的最终准确率、更好的客户端公平性、并在相同通信轮数内收敛更快，尤其在CIFAR‑10上的优势最为显著。

**⚠️ 局限性**

局限性包括：收敛分析仅覆盖凸（甚至可非光滑）损失，未给出非凸深度模型的理论保证；风险参数（α,γ）的选择需人工调参，未实现自适应；对大型模型的通信成本和内存影响仍待进一步评估。

---

## 33. Transsion's Speaker-Attributed Multilingual ASR System for the MLC-SLM 2026 Challenge

**arXiv ID:** 2609.20833 | [PDF](https://arxiv.org/pdf/2609.20833v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 34. Shake to Learn: Dynamic Interrogation of Hidden Object Physics for Robotic Manipulation with Physical Reservoir Computing

**arXiv ID:** 2609.20970 | [PDF](https://arxiv.org/pdf/2609.20970v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 35. Catch Me If You Can: Real-Time Feedback Denoising for Responsive VLAs

**arXiv ID:** 2609.21022 | [PDF](https://arxiv.org/pdf/2609.21022v1)

**作者:** Yiheng Ji `[一作]` (University of Texas at Austin), Mingyo Seo `[通讯]` (University of Texas at Austin)

**通讯引用:** 102 | [OpenAlex ID](https://openalex.org/A5075135654)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 VLA-Feedback 两频率架构，将 VLM‑DiT 拖延至最后一步并在执行时用实时视觉反馈进行微调，从而让基于扩散的视觉‑语言‑动作（VLA）在保持表达能力的同时实现低延迟响应。

**💡 创新点**

创新点在于：① 将扩散模型的最后一次去噪步骤拆成轻量级反馈接口，允许在不重新运行完整 VLM 的前提下对动作进行即时修正；② 通过可学习的视觉残差缩放把实时观察信息与预先规划的动作空间融合，形成可解释的动作微调方式；③ 采用两时钟高低频协同工作，实现高频视觉反馈与低频全局规划的无缝配合。

**🔧 技术方法**

核心技术包括：预训练的 VLM‑DiT 扩散规划器、轻量级 Feedback Denoising Module（动作编码+视觉编码+Transformer 融合+残差缩放+去噪速度预测）、两阶段训练（先冻结规划器后训练反馈模块）。

**📊 数据集**

使用 LIBERO‑Object 与 LIBERO‑Goal 基准任务、Robosuite 动态抓取与投掷任务，以及真实 Franka Panda 机器人上的静态与动态抓取/投掷实验。

**📈 对比分析**

与 OpenVLA、GR00T、FiS‑VLA 等基线相比，VLA-Feedback 在静态 LIBERO 任务几乎保持与 GR00T 同等成功率；在动态仿真任务上平均成功率从 27.5% 提升至 85.0%，在真实机器人上从 51% 提升至 73%；同时估计的反应延迟最低，达 2 ms。

**⚠️ 局限性**

局限性：仅对已生成的动作进行一次局部去噪修正，若初始规划偏离较大或目标在接触瞬间剧烈变化，单步反馈难以弥补；观测噪声、遮挡及执行延迟会削弱修正效果，且仍需依赖低频规划器产生合理的 near‑final 动作。

---

## 36. TAPe+ML: A Compact Structured Representation for Multi-Task Computer Vision

**arXiv ID:** 2609.20869 | [PDF](https://arxiv.org/pdf/2609.20869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 37. SAGE: Schema-Guided LLMs for Grant Review

**arXiv ID:** 2609.20829 | [PDF](https://arxiv.org/pdf/2609.20829v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 38. Beyond WER: Entity and Disfluency Recall in Accented Conversational ASR

**arXiv ID:** 2609.20828 | [PDF](https://arxiv.org/pdf/2609.20828v1)

**作者:** Fiza Husain `[一作]` (Stimuler), Yash Singh `[通讯]` (Stimuler)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个三阶段管线，将词错误率最小化与命名实体、填充停顿的识别与纠正结合起来，专为印度、印度尼西亚和拉美的口音英语设计。

**💡 创新点**

创新点在于：① 用轻量级 SQL 过滤器在现有转录中提取实体密集句子，实现 2.8 倍实体密度；② 为每个地区训练低秩 LoRA 适配器，仅增 32 维参数；③ 通过单前向推理生成原始与纠正两份 JSON 输出；④ 设计六类错误体系并用 LLM 判断器实现自动诊断。

**🔧 技术方法**

技术栈包括：Qwen2.5-Omni-3B + LoRA（rank32）、Gemini 2.5 Pro（参考转录）、Claude Sonnet 4.5（错误分类）、vLLM（低延迟推理）、Unsloth（内存优化）以及自定义 SQL 过滤脚本。

**📊 数据集**

数据集为公司内部生产日志，按地区分离：每个地区 10k 语音-文本对（通过 SQL 过滤后保留），其中 750 条由人工核对的金标准，用于验证和微调。

**📈 对比分析**

与 Parakeet、Whisper、未微调 Qwen2.5-Omni-3B、AssemblyAI Universal-3-Pro、以及 30B 的 Qwen3-Omni 进行对比；结果显示实体召回率 80–85%、填充停顿召回率 76–86%、WER 6–10%，相较基线提升显著，且在 3B 模型上匹配甚至超越 30B 规模模型。

**⚠️ 局限性**

局限性包括：参考转录为 Gemini 生成的银标准而非全人工校准；填充停顿召回率仍低于 30B 模型；仅评估英语，未涵盖多语或代码混杂情况；仅适用于三大口音区域，未验证跨域泛化。

---

## 39. COAL-SQL: Coverage-Guided Augmentation and Failure-Driven Learning for Text-to-SQL Post-Training

**arXiv ID:** 2609.20842 | [PDF](https://arxiv.org/pdf/2609.20842v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 40. Decomposing Predictive Kubernetes Autoscaling for Large Language Model Serving Under Long Startup Delays

**arXiv ID:** 2609.20874 | [PDF](https://arxiv.org/pdf/2609.20874v1)

**作者:** Tianrui Liu `[一作]` (University of California, San Diego), Xiaohai Hu `[通讯]` (University of Washington)

**通讯引用:** 206 | [OpenAlex ID](https://openalex.org/A5078664149)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对LLM在Kubernetes上进行预测性自动扩缩的分解研究，评估四个关键因素。

**💡 创新点**

证明延迟感知的预测lookahead是主导因素，并提出简单EWMA‑UCB方案即可获得大幅SLO改进；同时区分Kubernetes特定与LLM通用因素。

**🔧 技术方法**

使用预测性自动扩缩框架、EWMA、上置信界限（UCB）、Kalman滤波、事件驱动模拟器、vLLM、KEDA、Kubernetes HPA。

**📊 数据集**

使用ServeGen生成的重生产型工作负载（基于Alibaba Bailian的请求统计）以及在真实Kubernetes集群上运行的Qwen2.5‑7B实际负载。

**📈 对比分析**

与传统QPS HPA、KEDA队列阈值、静态分配做对比；EWMA‑UCB将TTFT SLO违规率从53%降至0.5%，延迟感知lookahead带来约14×提升。

**⚠️ 局限性**

模拟器调度简化导致绝对TTFT偏差；实验规模有限，仅单一7B模型，未验证更大模型或分离预填/解码场景。

---

## 41. Bio-MF: Low-Latency and High-Fidelity EEG-to-fNIRS Cross-Modal Generation for Hybrid Motor-Imagery Brain--Computer Interfaces

**arXiv ID:** 2609.20904 | [PDF](https://arxiv.org/pdf/2609.20904v1)

**作者:** Boyuan Zhao `[一作]` (Shaanxi Normal University), Luping Chen `[通讯]` (Shaanxi Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出了Bio-MF，一种用于混合运动意象脑机接口的单步EEG‑to‑fNIRS跨模态生成框架；

**💡 创新点**

创新点包括：① 采用无潜在空间的MeanFlow单步生成，显著降低延迟；② 设计了空间-时间交互的四维编码（Interactive 4D）以适配不同传感器布局；③ 引入跨模态无分类器指导（CFG）平衡EEG条件与fNIRS先验；④ 使用噪声水平门控的FFT正则化，抑制频域伪影；

**🔧 技术方法**

主要技术手段包括：Transformer编码器+多头自注意力、RMSNorm、SwiGLU、向量门控残差；MeanFlow速度监督；Token化多速率输入；跨模态CFG目标构造；两阶段训练与噪声门控FFT损失；

**📊 数据集**

使用了公开的双模态Dataset 1（EEG + fNIRS）和单模态Dataset 2（仅EEG）进行零样本跨设备评估；

**📈 对比分析**

与现有的SCDM、TADM等多步扩散模型通过分类准确率（ACC）和生成延迟进行对比；Bio‑MF在Dataset 1上ACC提升3–4个百分点，Dataset 2上提升约2–3个百分点；生成时间从6.0 s降至7 ms，速度提升857×；

**⚠️ 局限性**

局限性包括：仅在单一paired数据集上训练，难以覆盖更广泛的神经血管对应模式；零样本评估仅验证解码提升，未检验像素级一致性；生成的fNIRS在局部振幅上仍有差异，可能需个体化校准；

---

## 42. Sparse Priors for Efficient Distribution Learning

**arXiv ID:** 2609.20883 | [PDF](https://arxiv.org/pdf/2609.20883v1)

**作者:** Saumya Goyal `[一作]` (Carnegie Mellon University), Barnabás Póczos `[通讯]` (Carnegie Mellon University)

**通讯引用:** 10782 | [OpenAlex ID](https://openalex.org/A5013695358)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究稀疏先验在分布学习中的 Bayes 风险，并给出理论上无维度依赖的上、下界。

**💡 创新点**

提出稀疏维度概念，证明在 k‑稀疏先验下，学习分布的 Bayes 风险可达到 Ω(√(k/n))，克服了传统结果的维数灾难。

**🔧 技术方法**

采用信息论下界、参数化分布族构造、后验收缩理论以及拒绝采样等技术实现理论证明。

**📊 数据集**

主要使用理论构造的离散分布族和高斯位置族作为实例，无实测数据集。

**📈 对比分析**

与传统最小化风险（minimax）结果对比，展示在总变差和 Wasserstein‑1 度量下，取得与下界匹配的 O(√(k log n)/√n) 上界，显著优于传统 n^-c/d 速度。

**⚠️ 局限性**

仅在统计层面证明，未考虑计算复杂度；对先验未知或维度极高时稀疏维度仍可能随 d 增大，限制了实际应用。

---

## 43. PlantShade: Predicting Plant Shadows for Lighting-Aware Robotic Agricultural Operation

**arXiv ID:** 2609.21059 | [PDF](https://arxiv.org/pdf/2609.21059v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 44. Curriculum-Based Noise Adaptation for Phoneme-to-Text Reconstruction in Visual Speech Recognition

**arXiv ID:** 2609.20839 | [PDF](https://arxiv.org/pdf/2609.20839v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 45. Enhancing Audio Reasoning via Semantic Summary Prediction

**arXiv ID:** 2609.20849 | [PDF](https://arxiv.org/pdf/2609.20849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 46. AeRove: A Compact Bimodal Aerial-Terrestrial Drone with Rapid Bistable Reconfiguration for Close-Range Pipeline Inspection

**arXiv ID:** 2609.20965 | [PDF](https://arxiv.org/pdf/2609.20965v1)

**作者:** Caleb Polillio `[一作]` (New Jersey Institute of Technology), Petras Swissler `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5034624024)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开发了一款名为AeRove的双模空地机器人，用于管道近距离检测，能够在地面上滚动检测并在遇到障碍或垂直段时切换至飞行模式完成跳跃和进一步检测。

**💡 创新点**

创新点包括：①弹簧双稳机实现200 ms内无持续驱动的快速模式切换；②利用螺旋翼护罩作为轮子，减少机械部件与重量；③采用收敛型螺旋翼护罩提升推力效率；④在地面模式下将电流消耗降至飞行模式的1/14，显著延长行驶距离；⑤在气体感测中利用旋翼流动优化传感响应。

**🔧 技术方法**

技术手段包括：3D打印PETG结构、双稳机驱动与悬挂系统、MicoAir 743v2 AIO飞控、光流+测距+IMU姿态估计、Raspberry Pi Zero 2W协同控制、CO₂传感器与下垫面吸气设计、MAVLink与I²C双层控制架构、PD控制、基于视觉的管道定位与角度估计、层级状态机进行模式切换。

**📊 数据集**

主要使用实验室自制测试数据：管道直线、90°弯、障碍、垂直段等场景；使用CO₂浓度实验（模拟泄漏）评估感测性能；未采用公开公开数据集。

**📈 对比分析**

对比方法：测量连续飞行与地面滚动在相同巡检速度下的电流、续航与行程。地面模式电流0.7 A、续航85.7 min、行程2057 m；飞行模式电流10 A、续航6 min、行程144 m。与现有空地机器人比较，AeRove在地面模式下续航显著优于飞行模式，且在管道上滚动速度仅0.4 m/s，飞行速度可达9 m/s。气体感测实验显示，驻站模式下CO₂响应比悬停更快、更强，而旋翼吸气配置比延伸探针更优。

**⚠️ 局限性**

局限性：①在曲面（管道）上的接触与牵引受限，轮子与管道表面接触不够理想；②双稳机在不平坦或凹凸地面上可能受阻；③气体感测受限于传感器位置与流场设计，未在真实工业泄漏条件下验证；④缺乏不同管道直径、材质、光照和温湿度等更复杂场景的实验；⑤缺少长期现场验证与多机协同操作的测试。

---

## 47. Reviser: Revision-Capable Text Generation via Autoregressive Cursor Actions

**arXiv ID:** 2609.20830 | [PDF](https://arxiv.org/pdf/2609.20830v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 48. A Generative Grammar Underlying the Voynich Manuscript, the Pastiche Hypothesis: Evidence from Large Language Models

**arXiv ID:** 2609.20835 | [PDF](https://arxiv.org/pdf/2609.20835v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 49. Do Quantum Models Scale Like LLMs?

**arXiv ID:** 2609.20912 | [PDF](https://arxiv.org/pdf/2609.20912v1)

**作者:** David S. Berman `[一作]` (Queen Mary University of London), Alexander G. Stapleton `[通讯]` (Queen Mary University of London)

**通讯引用:** 48 | [OpenAlex ID](https://openalex.org/A5091983021)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了RydbergGPT transformer在量子测量数据上的神经缩放规律，并通过互信息比较了其统计结构与自然语言的相似性。

**💡 创新点**

证明了在接近量子临界点时，量子测量数据能呈现与自然语言相似的多尺度互信息衰减，从而为神经缩放规律提供了数据分布角度的新解释。

**🔧 技术方法**

使用自回归Transformer（RydbergGPT）、Hoffmann（Chinchilla）缩放模型、互信息估计与熵归一化的两点函数。

**📊 数据集**

使用量子Monte Carlo模拟得到的6×6 Rydberg原子阵列测量数据（六个激光失调点）以及AG News和Yahoo Answers的字符级文本。

**📈 对比分析**

通过熵归一化、置换消偏的互信息曲线对量子数据与自然语言进行直接对比，发现临界点附近的互信息曲线与自然语言相近，且模型在该区间满足高R²的幂律缩放；离临界点则失效。

**⚠️ 局限性**

受限于有限的晶格尺寸、开放边界、有限样本偏差和置换方法破坏对称性，导致对真实临界行为的解释不完全；此外结果仅在固定模型规模下检验，未涵盖规模与数据协同缩放的通用性。

---

## 50. When AI Reviews Train AI Reviewers: Scientific-Judgment Collapse and Mitigation

**arXiv ID:** 2609.20942 | [PDF](https://arxiv.org/pdf/2609.20942v1)

**作者:** Sy-Tuyen Ho `[一作]`, Furong Huang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在递归AI评审训练中引入合成评审导致的科学判断压缩（rating分布收窄、语义多样性下降）现象，并提出了TrustReviewer系统以训练时语料筛选和推理时激活引导的双阶段干预。

**💡 创新点**

创新点在于首次量化递归评审训练引起的判断同质化，并结合训练时语料清洗与推理时配对激活引导的两阶段方法，且将精细筛选后的评审数据集公开。

**🔧 技术方法**

使用 Llama3.1-8B 进行 LoRA 微调、配对激活引导（paired activation steering）以及基于语义嵌入的多样性评估。

**📊 数据集**

数据来自 2018–2025 年 ICLR 官方评审文本，共 112,743 条，包含公开的 2,000 篇评估论文。

**📈 对比分析**

在与 Meta‑Llama‑3.1‑8B、Qwen3.6‑35B‑A3B、OpenReviewer 等基线的评估中，TrustReviewer 在准确匹配率（75.40%）和 MAD（1.079）上优于其他模型，且评级熵和语义分散度均显著提升。

**⚠️ 局限性**

局限性包括仅研究单步递归训练，未探究多代累积效应；评审数据中可能仍包含未知 AI 辅助；方法对不同学术会议的泛化尚需验证。

---

## 51. TatBLiMP: A Benchmark of Linguistic Minimal Pairs for Tatar

**arXiv ID:** 2609.20832 | [PDF](https://arxiv.org/pdf/2609.20832v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 52. Towards Secure Cloud-Native Computing: Unveiling Kubernetes Misconfigurations with Large Language Models

**arXiv ID:** 2609.20834 | [PDF](https://arxiv.org/pdf/2609.20834v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 53. Target-Stratified Fair Range Summaries: Improved Fair $\varepsilon$-Nets and Geometric Hitting Sets

**arXiv ID:** 2609.20895 | [PDF](https://arxiv.org/pdf/2609.20895v1)

**作者:** Mingchao Zhou `[一作]` (Zhejiang Normal University), Zhao Zhang `[通讯]` (Zhejiang Normal University)

**通讯引用:** 10984 | [OpenAlex ID](https://openalex.org/A5100423029)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了公平 ε‑网和公平几何打通集，为范围查询工作负载提供具有公平组比例约束的紧凑摘要，并提出目标分层采样框架和改进的 LP 取整算法。

**💡 创新点**

① 目标分层采样直接固定各组样本数，消除样本后修复的对数因子；② 在人口比例公平下恢复经典 ε‑网上界；③ 用分布偏移参数 Γ 量化自定义比例难度，并证明其不可避免；④ 提升公平几何打通集的逼近比至无对数因子；⑤ 通过 FGHS 取得自定义比例公平 ε‑网的近似解。

**🔧 技术方法**

目标分层采样、VC 维度下的 ε‑网理论、Chernoff 与 Sauer–Shelah lemmas、LP 线性规划与随机取整、对数校正与不等式、实验评估等。

**📊 数据集**

实际数据集：Adult 与 COMPAS；合成数据集用于规模、组数、Γ 参数和群体重叠实验。

**📈 对比分析**

与 Dehghankar 等人的样本+修复方法、Fair Sketch-and-Merge 以及基线 FGHS 取整方法比较；在多数设置下，目标分层采样生成更小、更快的公平摘要，并在自定义比例下显著减少样本量，同时提升范围查询过滤效果。

**⚠️ 局限性**

仅适用于离散组比例且需整数样本数，未考虑动态/流式更新；对分布偏移参数 Γ 的依赖在极端情况下仍导致样本量大；实验仅覆盖轴对齐矩形范围，尚未验证对更复杂几何范围的效果。

---

## 54. Exploiting Mutual Coupling Structure for Channel Estimation of Active RIS-Assisted Links

**arXiv ID:** 2609.21062 | [PDF](https://arxiv.org/pdf/2609.21062v1)

**作者:** Simon Tarboush `[一作]` (Technical University of Berlin), Giuseppe Caire `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了在活性RIS辅助的MIMO通信中，如何在存在互耦（MC）的情况下进行高效的信道估计。

**💡 创新点**

创新点在于利用MC的稀疏结构，并通过Neumann级数展开把互耦矩阵近似为稀疏形式，从而把原本维度成倍提升的压缩感知（CS）问题降到O(N_I)级别，设计出低复杂度的感知矩阵和字典矩阵。

**🔧 技术方法**

主要技术包括：S-参数物理一致的RIS模型、Neumann级数展开、压缩感知中的OMP算法、Kronecker/Hadamard/行内Khatri–Rao乘积等线性代数工具。

**📊 数据集**

使用模拟数据，结合真实RIS原型测得的S参数作为互耦矩阵，构造多路径信道模型（L=4/2路径），并通过随机相位扫描进行训练。

**📈 对比分析**

方法通过与传统的MC无关的OMP和完全MC感知的CS对比，使用NMSE指标评估估计精度。实验结果显示，所提方法在相同训练次数下，NMSE比MC无关方法低数dB，且与完全MC感知的性能相当，但计算复杂度显著降低。

**⚠️ 局限性**

局限性包括：需要预先知道RIS的平均放大系数和近似稀疏互耦矩阵，假设矩阵不随时间变化；对散射矩阵的误差、硬件漂移等不作进一步分析；实验仅基于仿真，未在真实场景中验证。

---

## 55. Rewarding Efficient Reasoning Improves Abstention on Underspecified Tasks in Reasoning Models

**arXiv ID:** 2609.20846 | [PDF](https://arxiv.org/pdf/2609.20846v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 56. TALON: A Temporally Aware Longitudinal Framework for Radiology Report Generation

**arXiv ID:** 2609.20826 | [PDF](https://arxiv.org/pdf/2609.20826v1)

**作者:** Nien-Tsyr Sun `[一作]` (National Yang Ming Chiao Tung University), Vincent S. Tseng `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 9026 | [OpenAlex ID](https://openalex.org/A5043399804)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了TALON框架，能够在可变长度的病历历史中生成纵向放射科报告。

**💡 创新点**

创新点在于Dual-Channel Temporal Fusion Module（DCTFM）：对每个历史检查分别评估相似性（稳定性）和变化通道的相关性，并通过门控融合，既保留每个检查的身份，又实现角色（稳定 vs 变化）依赖的优先选择。

**🔧 技术方法**

使用RAD‑DINO视觉编码器、CXR‑BERT文本编码器、DistilGPT2生成器；结合跨时间对齐、通道门控、残差流、交叉注意力、对比学习预训练等技术。

**📊 数据集**

实验数据集为公开的MIMIC‑CXR胸片数据集。

**📈 对比分析**

与单体与纵向RRG方法（如CoFE、DCG、PriorRG、STREAM、MLRG等）对比，TALON在临床效能（CheXbert）和RadGraph‑F1指标上领先，其他生成指标保持竞争力；性能提升随可用历史深度增加。

**⚠️ 局限性**

局限：仅支持单视角检查且最多5个历史检查；需要扩展到多视角和更长轨迹；对跨时间配准仍依赖注意力，可能在极端时间间隔下不稳定。

---

## 57. On the Limits of Maximal Coding Rate Reduction for Out-of-Distribution Generalisation

**arXiv ID:** 2609.21001 | [PDF](https://arxiv.org/pdf/2609.21001v1)

**作者:** Menghui Zhou `[一作]` (University of Sheffield), Po Yang `[通讯]` (University of Sheffield)

**通讯引用:** 8317 | [OpenAlex ID](https://openalex.org/A5008276130)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `2704f255-0c84-4173-b83c-0e9a3dbea232` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

分析最大编码率降低（MCR2）在 OOD 泛化中的局限性，构造理论对例并在 Waterbirds 实验验证。

**💡 创新点**

首次系统性证明 MCR2 即使达到最优编码几何，也无法保证跨环境的稳定预测，并展示共享最优编码约束同样无法解决此问题。

**🔧 技术方法**

采用信息论编码率降低理论、线性判别分析、IRM/REx 对比、简易二分类环境模型以及 saliency 可视化等技术。

**📊 数据集**

使用 Waterbirds 数据集以及自定义的二分类环境模型作为实验与理论验证的数据来源。

**📈 对比分析**

通过源最优分类器在训练与测试环境下的准确率进行评估；在背景不匹配的测试组中准确率低于随机，理论上编码接近最优但预测错误可达 100%。

**⚠️ 局限性**

MCR2 仅保证表示的几何结构，无法确保跨环境的预测稳定性；即便加入共享最优编码约束，也无法避免环境相关性逆转导致的预测失败。

---

## 58. Prophet Inequalities and Online Contention Resolution for Matchoids

**arXiv ID:** 2609.20939 | [PDF](https://arxiv.org/pdf/2609.20939v1)

**作者:** Calum MacRury `[一作]` (Georgia Tech), Jan Vondrák `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了在k匹配族（k‑matchoid）约束下的预期前（ex‑ante）预言家不等式与内容争取分配（OCRS/RCRS），给出了在对抗顺序和随机顺序两种到达模型下的最优竞争比；

**💡 创新点**

创新点在于：①引入协同加权主分区（weighted principal partitions）并通过共用盈余向量协调多棵本地矩形的价格；②使用凸势能函数求解固定点，得到满足预言家最优约束的价格向量；③将该技术推广到k‑匹配族，为传统的k‑matroid交叉问题提供更优的竞争比；

**🔧 技术方法**

主要技术：凸优化与Kakutani固定点理论、对偶性框架、矩形理论的加权主分区、对偶 LP 的子梯度、随机化交换引理（exchange lemma）以及 Lee‑Singla 的效用‑收入分析；

**📊 数据集**

该工作为纯理论分析，未使用具体数据集；

**📈 对比分析**

与之前工作比较：对抗顺序下的竞争比由原来的 1/(e+o(1))k 提升到 1/(k+1)；随机顺序下竞争比从 1/(k+1) 提升到 (1−e^{−k})/k；同时得到相应的 OCRS / RCRS 选择率，均为最优或接近已知整数规划的 integrality gap；

**⚠️ 局限性**

限制与开放问题：①随机顺序下的 (1−e^{−k})/k 结果是否最优仍未知；②对于 k‑matchoid 以及更一般的约束形式的更紧界限尚未给出；③方法假设所有分布已知且相互独立，现实应用中可能不满足；

---

## 59. HERMES: Contrast-Aware Knowledge Graph Reasoning from Clinical Notes for Patient Outcome Prediction

**arXiv ID:** 2609.20825 | [PDF](https://arxiv.org/pdf/2609.20825v1)

**作者:** Gia-Bach Nguyen `[一作]` (National Economics University), Thien Van Luong `[通讯]` (National Economics University)

**通讯引用:** 752 | [OpenAlex ID](https://openalex.org/A5018806075)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

使用大型语言模型从临床文字中抽取实体与关系，构建患者个性化知识图谱，并通过图注意力网络进行预测。

**💡 创新点**

创新点在于：①引入对比逻辑建模，突出治疗失败、矛盾与时间演化信息；②全文本无结构数据构建知识图谱；③结合图注意力网络显式捕获临床关系与动态。

**🔧 技术方法**

技术包括：LLM引导的实体与关系抽取（Chain‑of‑Thought 提示）、实体合并与置信度筛选、对比逻辑建模、双层 GAT + MaxPool + MLP 分类。

**📊 数据集**

使用 MIMIC‑III 与 MIMIC‑IV 两大 ICU 数据集，分别评估院内死亡和 30 天再入院预测。

**📈 对比分析**

与 ClinicalBERT、Clinical‑LongFormer、PubMedBERT、Note‑HCR、LR+TF‑IDF 及零样本 GPT‑5.4 mini 对比，HERMES 在 AUROC、AUPRC 和 min(+P, Se) 上均领跑，提升幅度约 2–4 点。

**⚠️ 局限性**

局限性包括：①依赖 LLM 抽取质量；②未评估跨机构泛化与多模态融合；③KG 质量评估缺失；④推理成本较高；⑤仅针对 ICU 文本，未覆盖更广泛临床场景。

---

## 60. From Papers to Interpretive Knowledge Nodes: Proposing the Missing Object in Scholarly Knowledge Circulation

**arXiv ID:** 2609.20840 | [PDF](https://arxiv.org/pdf/2609.20840v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 61. PhysioBench: A Unified Benchmark for Physiological Signal Question Answering

**arXiv ID:** 2609.20836 | [PDF](https://arxiv.org/pdf/2609.20836v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 62. VISPATH: Visual-Intent-Guided Path Reasoning for Multimodal Knowledge Graph Question Answering

**arXiv ID:** 2609.20843 | [PDF](https://arxiv.org/pdf/2609.20843v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 63. Towards Effective Visual-Inertial SLAM with Passive-Only Sensors for Low-Cost Autonomous Underwater Vehicles

**arXiv ID:** 2609.21015 | [PDF](https://arxiv.org/pdf/2609.21015v1)

**作者:** Grant Schwidder `[一作]` (University of Minnesota--Twin Cities), Junaed Sattar `[通讯]` (University of Minnesota--Twin Cities)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

实现并验证了一套低成本、完全被动的 VI‑SLAM 系统，利用消费级立体相机、IMU 与压力传感器，在室内走廊、游泳池及开放海域中实现了实时 6‑DOF 导航与地图构建。

**💡 创新点**

通过将 ZED Mini 的 GEN1 稠密 VIO 与多传感器 EKF 融合、引入深度传感器校正、采用相对位移与传输定理实现姿态纠正，以及在 RTAB‑Map 后端实现循环闭环与图优化，展示了在缺乏昂贵声学传感器的情况下，低成本 AUV 也能获得可接受精度的关键技术路线。

**🔧 技术方法**

使用的核心技术包括：ZED Mini 立体相机 + 集成 IMU（GEN1 稠密 VIO）、MicroStrain 3DM‑CV7 IMU、BlueRobotics MS5837 压力深度传感器；ROS 2 框架下的扩展卡尔曼滤波器（EKF）实现异步多源数据融合；RTAB‑Map 进行循环闭环检测与图优化；Jetson Orin NX GPU 进行实时计算；以及针对高角速度与低特征环境的运动补偿与死 reckoning 机制。

**📊 数据集**

实验数据来源为实地采集的三种环境：室内办公走廊（含慢速/高速转弯）、室内游泳池（含闭环/不开环）以及开放海域珊瑚礁。对室内实验使用自建 CAD 模型作为基准进行点云对齐（ICP）与 RMSE 评估；海域实验以视觉质量与点云连贯性为主要评价指标，没有公开的基准数据集。

**📈 对比分析**

通过 RMSE 对比评估：走廊慢速转弯 0.56 m，走廊高速转弯 0.71 m；泳池闭环 1.04 m，泳池不开环 1.14 m；海域实验则通过视觉与点云连贯性证明系统在动态光照与高 6‑DOF 运动下仍能保持平滑轨迹。相较于原生 ZED VIO，加入 EKF 与深度校正后显著降低了漂移，并在多回环场景中实现了可接受的全局一致性。

**⚠️ 局限性**

局限性包括：在低特征/浑浊水域下视觉跟踪易失效；稠密 VIO 计算开销高（占 GPU 77%），且 ZED SDK 为专有软件，难以进一步优化；EKF 在强非线性运动下表现受限，可能导致状态震荡；系统对初始静止校准有要求；缺乏公开海域基准数据，导致海域性能评估主要依赖定性判断。

---

## 64. Continuous Delayed-Memory Stochastic Gradient Descent and Continuous-Time Reinforcement Learning from History of Astrophysical Time Series Studies

**arXiv ID:** 2609.20906 | [PDF](https://arxiv.org/pdf/2609.20906v1)

**作者:** Debartha Paul `[一作]`, Juncheng Yi `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了连续延迟记忆随机梯度下降（Continuous‑Delayed‑Memory SGD）和连续时间强化学习框架，利用历史状态和分布式延迟记忆实现更强的探索与更精确的收敛，并将神经延迟随机微分方程（Neural SDDE）应用于星系光变曲线的生成与物理参数推断。

**💡 创新点**

创新点在于将过去状态作为梯度更新的反馈引入SGD，形成分布式延迟记忆机制；在强化学习中引入探索性后向斯托克斯SDE（Exploratory Backward Stratonovich SDE），直接从前向布朗轨迹推导梯度，避免求解HJB方程；以及通过Neural SDDE克服传统OU、CARMA和深度网络对光变曲线的线性、单输出限制。

**🔧 技术方法**

核心技术包括：连续时间随机微分方程与延迟随机微分方程、神经网络参数化的漂移与扩散、随机对偶（adjoint）方法、虚拟布朗树（Virtual Brownian Tree）实现低内存反向传播、以及熵正则化的探索性策略梯度。

**📊 数据集**

实验数据主要有：(1) 2维二次与Rastrigin/Styblinski‑Tang损失景观的人工仿真；(5) Fagin 等 2024 年使用的 LSST 模拟四百千条多波段奎泽尔光变曲线；(6) ELAsTiCC 与 PLAsTiCC 的模拟光变曲线集用于分类与新颖性检测。

**📈 对比分析**

在与传统 SGD、Neural SDE、GRU‑D、GPR 等基线比较时，Continuous‑Delayed‑Memory SGD 在非凸景观能更频繁逃离局部极小值、收敛更稳；在光变曲线重建中，Latent SDDE 超越单输出 OU 以及 GPR，能够同时完成季节性缺失插值和黑洞物理参数的高精度推断；在分类任务中，Neural SDDE 达到最高准确率和 F1 分数。

**⚠️ 局限性**

局限性包括：(1) 记忆强度 λ 或分布式核 Λ 的选取对收敛性敏感，过大易导致发散；(2) 随机梯度估计方差高，需大批量和梯度裁剪；(3) 目前主要在低维或模拟数据上验证，尚未在真实高维 RL 任务中证明可扩展性；(4) 对跳跃过程或非高斯噪声的理论支持尚未完成。

---

## 65. From Switching to Dynamic Regret: A Simple Reduction via Unbiased Random Sequences

**arXiv ID:** 2609.20968 | [PDF](https://arxiv.org/pdf/2609.20968v1)

**作者:** Yibo Wang `[一作]` (Nanjing University), Lijun Zhang `[通讯]` (Nanjing University)

**通讯引用:** 38260 | [OpenAlex ID](https://openalex.org/A5100448159)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899`

**🎯 论文内容**

提出了一种将动态遗憾最小化问题简化为切换遗憾最小化的问题的新框架。

**💡 创新点**

通过构造一个辅助随机序列，将动态遗憾的分析简化为切换遗憾的分析，避免了复杂的离线最优解分析和KL投影。

**🔧 技术方法**

使用了辅助随机序列和适当的替代损失，结合现有的切换遗憾算法来推导动态遗憾界限。

**📊 数据集**

未具体提及使用的数据集，但研究背景涉及在线学习和动态决策的实际应用。

**📈 对比分析**

通过将动态遗憾最小化转化为切换遗憾最小化，利用现有的切换遗憾算法获得了强凸和指数凹损失的动态遗憾界限为𝒪(T^1/3P_T^2/3)，对于一般凸损失则为𝒪(√(T(1+P_T)))，这些结果是最优的。

**⚠️ 局限性**

该框架在理论分析中使用的随机序列在实际执行中并不需要生成，可能在实现过程中存在一定的局限性。

---

## 66. Composer2Vec: A Continuous Embedding Space of Composer Style Learned from Symbolic Melody Generation

**arXiv ID:** 2609.20893 | [PDF](https://arxiv.org/pdf/2609.20893v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 67. From Generation to Detection: Exploration of Discourse Driven Scenario based LLM Generated Fake News

**arXiv ID:** 2609.20838 | [PDF](https://arxiv.org/pdf/2609.20838v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 68. From Stress to Affect: Multimodal Deep Learning for Physiological Emotion Recognition Across Wearable Sensor Modalities

**arXiv ID:** 2609.20991 | [PDF](https://arxiv.org/pdf/2609.20991v1)

**作者:** Desta Haileselassie Hagos `[一作]` (Howard University), Legand L. Burge `[通讯]` (Howard University)

**通讯引用:** 499 | [OpenAlex ID](https://openalex.org/A5068264136)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统比较了不同时间序列深度学习模型在多模态可穿戴传感器情绪识别中的表现，评估了多模态、单模态和模型集成的效果，并分析了采样频率对性能和计算成本的影响。

**💡 创新点**

创新点在于：①统一实验框架下对LSTM、Transformer、TCN三种架构在两大公开数据集上的跨模态、跨任务比较；②引入软投票集成与传感器消融、梯度可解释性分析以及采样频率敏感性评估；③揭示模型性能与数据集特性、模态组合的相互依赖关系。

**🔧 技术方法**

使用了双向LSTM、Transformer编码器和TCN卷积网络，配合梯度基Saliency可解释性、soft-voting集成、随机森林与逻辑回归基线、以及自定义的采样频率调度。

**📊 数据集**

使用了WESAD（三类压力识别）和EmoWear（二分类唤醒/价值）两大多模态可穿戴数据集，均包含手腕和胸部传感器。

**📈 对比分析**

在LOSO-CV评估中，Transformer在WESAD多模态上取得最高准确率99.02%，LSTM在EmoWear多模态上获得最高准确率91.80%；软投票集成在两数据集均显著降低了跨受试者方差；多模态配置普遍优于单模态。

**⚠️ 局限性**

主要局限包括：未评估跨数据集迁移性能；仅在受控/半自然实验环境下验证，缺乏完全自然场景；样本规模与人口多样性有限；采样频率分析仅在EmoWear的LSTM上进行，未覆盖其他模型和数据集。

---

## 69. ASGARD: Action-Space Guard for UAV Resilience via Reinforcement Learning

**arXiv ID:** 2609.20982 | [PDF](https://arxiv.org/pdf/2609.20982v1)

**作者:** Mohsen Salehi `[一作]` (University of British Columbia), Karthik Pattabiraman `[通讯]` (University of British Columbia)

**通讯引用:** 5579 | [OpenAlex ID](https://openalex.org/A5073641368)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于强化学习的无人机控制器防御行动空间攻击的两阶段教师-学生管道，并在控制器与执行器之间加入轻量化监视器来实时纠正被篡改的动作指令。

**💡 创新点**

创新点在于①利用教师阶段的攻击上下文信息生成攻击感知潜在空间；②通过教师-学生自监督学习，仅用物理状态历史即可推断潜在空间；③将监视器放置于控制器与执行器之间，实现对每一步动作的主动纠正，而非仅做检测或重训。

**🔧 技术方法**

使用PPO强化学习、VAE/TVAE编码器、LSTM序列学习、MLP监视器以及监督训练和仿真环境gym-pybullet。

**📊 数据集**

在仿真生成的3D航点跟踪任务中，使用随机生成的航点和自定义的行动空间攻击注入（倾斜、滚转、推力、增益以及全通道攻击），并提供攻击上下文信息。

**📈 对比分析**

与基线RL控制器和ARMOR做对比，采用任务成功率、崩溃率、状态漂移等指标。该方法在单通道攻击下成功率约95%且无崩溃，复合攻击下成功率67%且10%崩溃；相比基线RL 40-60%成功率和ARMOR 0%（复合攻击）或约50%（单通道），性能显著提升。

**⚠️ 局限性**

仅在仿真环境中验证，未涉及真实飞行或跨平台迁移；教师阶段需要攻击上下文信息，部署时无法获取；对极端或未知攻击模型的鲁棒性仍需进一步研究。

---

## 70. CaLR: Causal Latent Revision for Robust Diffusion Reasoning

**arXiv ID:** 2609.20981 | [PDF](https://arxiv.org/pdf/2609.20981v1)

**作者:** Wei Cai `[一作]` (Peking University), Xuelong Li `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

采用Causal Latent Revision框架，将DLM的推理视为受约束的潜在优化，利用梯度导向的“思维修订”提升并行生成的逻辑一致性。

**💡 创新点**

通过引入因果拓扑矩阵CTM以及隐式微分的潜在修订方法，实现DLM在推理过程中的主动自我纠错和全局因果一致性。

**🔧 技术方法**

使用因果拓扑矩阵、隐式微分、对抗优化、LoRA微调、低秩适配器、混合正则化等技术。

**📊 数据集**

在GSM8K、MATH500、Sudoku（4×4）、MMLU、ARC‑C、GPQA、HumanEval、MBPP等数据集上评估。

**📈 对比分析**

与AR模型（如Llama3.1‑8B、Qwen2.5‑7B等）和标准DLM（LLaDA、Dream）对比，CaLR在GSM8K上从44.17%提升至58.36%，在Sudoku上从77%提升至92%，并在高并行推理时保持稳定优势。

**⚠️ 局限性**

主要限制包括对CTM的手工构建依赖、对α等正则化系数的敏感性，以及在更大规模或多样化任务中可能仍面临因果结构不足的挑战。

---

## 71. Learning-based near- versus far-field boundaries for ultra-massive MIMO communications

**arXiv ID:** 2609.21055 | [PDF](https://arxiv.org/pdf/2609.21055v1)

**作者:** Simon Tarboush `[一作]` (Technische Universitaet Berlin), Giuseppe Caire `[通讯]` (Technische Universitaet Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在UM‑MIMO系统中，提出了一种完全无监督的学习框架，通过接收信号功率在子阵列间的变化来区分近场与远场通信区域；

**💡 创新点**

创新点在于将物理启发的功率差特征与OPTICS聚类相结合，既无需CSI、用户位置或标注数据，也能在低SNR下保持高鲁棒性；

**🔧 技术方法**

采用随机波束训练、子阵列功率差度量（η）以及OPTICS聚类算法；

**📊 数据集**

使用TeraMIMO仿真器生成的THz‑频段UM‑MIMO通道数据，在30m×30m二维网格、不同SNR（-4、0、4、8 dB）和训练比（M̅/2、M̅/4、M̅/8）下的多次通道与噪声重现；

**📈 对比分析**

与理论边界（MIMO‑ARD、ERD、等功率线、阈值距离）对比，结果显示聚类识别的近场/远场区域与ERD边界高度一致，且在大多数SNR与训练设置下误差极小；

**⚠️ 局限性**

局限性包括：在极低SNR（-4 dB）及最低训练率下会出现散射误判；模型仅提供分类结果，无法直接估计距离；对不同阵列几何或多路径环境的适应性尚待进一步验证。

---

## 72. Constraint-Unified MPC for Over-Actuated Surface Vehicles with Post-Detection Fault Reconfiguration

**arXiv ID:** 2609.21046 | [PDF](https://arxiv.org/pdf/2609.21046v1)

**作者:** Sebastian Burmester `[一作]` (ETH Zürich), Aswin Ramachandran `[通讯]` (ETH Zürich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出并部署了一种统一的约束MPC框架，利用单个二次规划在10 Hz下同时完成轨迹跟踪、推力分配、每个推进器的力与速率限制以及故障后重配置。

**💡 创新点**

创新点在于将上述四项功能直接融合到一条“压缩”QP中，并通过外部故障检测标志实现对失效推进器的即时重配置，避免了传统级联结构中分层约束导致的饱和与误差积累。

**🔧 技术方法**

核心技术包括：对船体Fossen模型的线性化与离散化、推力分配矩阵的直接嵌入、条件约束（力/速率/失效）在QP中显式约束、OSQP求解器实现实时控制，以及与LQR级联基线的对比实验。

**📊 数据集**

实验数据集来自瑞士苏黎世湖与威尼斯Time Space Existence 2025展览的现场表演，包含10 分钟静止保持、10 m圆形方形路径以及单/双推进器失效（即时与延迟）情景。

**📈 对比分析**

与传统LQR级联基线比较时，未失效时两者在站稳和方形跟踪上的误差相差不到几厘米；在单推进器失效时，MPC的均方根位置误差仅为级联方法的约1/50，且饱和比例低于1%；在双推进器失效时，MPC通过放宽姿态权重可将位置误差降低至1.25 m。

**⚠️ 局限性**

限制主要体现在：①仅针对四推力、对称布置的全向小艇；②缺乏稳定性理论保证（采用经验调参）；③外部故障检测需准确及时，若检测延迟仍需手动更新失效标记；④对较大延迟或多重失效时仍需手工调整权重以平衡位置与姿态误差。

---

## 73. SPARROW: Survival-POMCP for Adaptive Robot Routing, Observation, and Waiting

**arXiv ID:** 2609.21008 | [PDF](https://arxiv.org/pdf/2609.21008v1)

**作者:** Hshmat Sahak `[一作]` (University of Toronto Institute of Aerospace Studies), Timothy D. Barfoot `[通讯]` (University of Toronto Institute of Aerospace Studies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个面向临时障碍的图导航框架 SPARROW，利用部分可观测半马尔可夫决策过程 (POSMDP) 对等待、观测与换路等动作进行贝叶斯规划。

**💡 创新点**

创新点包括：①将在线生存模型（Kaplan–Meier）与 POMCP 结合，实现对障碍剩余寿命的实时估计；②在搜索树中加入持续时间量化与时间相关叶子评估 (TDSP)，从而显式考虑未来可能出现的障碍；③引入价值学习（knowledge‑gradient）决定何时主动获取标签，以平衡即时导航成本与长期收益；④通过粒子贝叶斯维护对未观测障碍的分布，支持多类、连续时间的决策。

**🔧 技术方法**

核心技术包括：Partially Observable Monte Carlo Planning (POMCP)、事件驱动障碍生成器、Kaplan–Meier 生存估计、A* 期望成本搜索、知识梯度（value‑of‑learning）以及视觉‑语言分类器用于获取障碍类别。

**📊 数据集**

实验数据集涵盖：① 两个图—真实校园 atrium（12 节点/18 条边）与 5×5 合成网格；② 真实机器人部署在 22.1×20.9 m 室内院子，使用 Clearpath Jackal；③ 障碍类为人、椅子等，使用 Poisson 过程生成。

**📈 对比分析**

与 Always Wait、Always Reroute、OSCAR、OSCAR‑Oracle 以及 SPARROW‑Oracle 进行对比。SPARROW 在模拟中平均减少 12–26% 的到达时间；在真实机器人上较 OSCAR 降低 20.5% 的到达时间。Ablation 结果显示观测选择、价值学习和叶子评估均对性能贡献显著。

**⚠️ 局限性**

局限性包括：① 需要手工设置障碍到达/类别分布假设，模型对不同环境的迁移性未知；② 对大规模图的搜索预算要求较高；③ 仅评估单一机器人场景；④ 对低观测率或高度动态环境的鲁棒性未充分验证。

---

## 74. Image-Derived PM10 Estimation in Cattle Feedlot Using Machine Learning: Addressing Concentration Ranges Beyond Existing Digital Imaging Methods

**arXiv ID:** 2609.20975 | [PDF](https://arxiv.org/pdf/2609.20975v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 75. PIVOT: Physically Informed Vision-Language Off-Road Traversability for Field Robot Navigation

**arXiv ID:** 2609.20983 | [PDF](https://arxiv.org/pdf/2609.20983v1)

**作者:** Aoran Jiao `[一作]`, Timothy D. Barfoot `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了PIVOT系统，通过结合 LiDAR 感知的几何通行性评估与 VLM 的语义推理，实现了在混合地形下的闭环自律行驶，并在实验中显著提升了自主行驶率。

**💡 创新点**

创新点在于：①将 VLM 对能量消耗、振动与车轮滑移的预测与实际测量相关性相结合，构造物理加权的通行性得分；②采用两级导航架构，仅在几何规划失败时才激活 VLM 语义重规划，保持规划效率。

**🔧 技术方法**

使用技术包括：LiDAR 强度图和点云渲染输入 GPT‑5 进行语义通行性推理；基于电池电压/电流、IMU 振动谱和速度误差计算的物理指标；C‑BIT* 规划器与两级成本图更新机制。

**📊 数据集**

数据集为约 6.4 km 的混合地形闭环实验记录，涵盖建筑、草地、树林和停车场四个段落，并同步采集 LiDAR、IMU、电池、电机速度等多模态传感数据。

**📈 对比分析**

与仅几何规划的基线对比，采用自主率、人工干预次数、平均干预间距（MDBI）和行驶时间等指标评估，PIVOT 的自主率从 59.6% 提升至 97.0%，干预次数从 11 次降至 3 次，MDBI 增长约 6 倍，行驶时间仅在需要语义重规划的植被区增加约 20 分钟。

**⚠️ 局限性**

限制包括：VLM 推理的计算延迟导致植被密集区行驶时间增加；对轮滑失配的物理相关性相对弱，影响加权效果；对极端光照或极端地形的评估能力有限，需要依赖人工恢复；系统对云端模型与算力需求的依赖也可能限制实际部署。

---

## 76. Generative inversion for early ranking of competing geologic interpretations

**arXiv ID:** 2609.20978 | [PDF](https://arxiv.org/pdf/2609.20978v1)

**作者:** Harun Ur Rashid `[一作]` (Los Alamos National Laboratory), Daniel O'Malley `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 2771 | [OpenAlex ID](https://openalex.org/A5072498681)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

将竞争的书面地质解释转换为图像集，然后通过潜在空间逆演算和稳态流模拟，对有限的水头观测进行一致性评估，最终给出相对兼容性分数和排名。

**💡 创新点**

创新在于将自然语言描述直接映射为可用的地质先验空间（通过文本到图像扩散模型生成的图像），并在每个解释特定的潜在空间内训练逆网络，实现快速、可比的早期解释评估。

**🔧 技术方法**

使用扩散式文本到图像模型、卷积变分自编码器、全连接监督逆网络、DPFEHM 差分流模拟器以及 Softmax 兼容性归一化等技术。

**📊 数据集**

数据集包括：① 基于 Johansen 组分的三类合成解释（共 925 份测试场景，250 条水头观测），② WIPP Culebra Dolomite 两个已公开的概念模型及其 250 条现场水头观测。

**📈 对比分析**

通过将逆网络输出映射到导电率场，预测水头并计算与观测的 RMSE，再转化为高斯兼容性分数并 Softmax 归一化，形成每个解释的相对兼容性权重。实验表明：在合成基准中能正确按解释一致性顺序排序；在 WIPP 真实案例中，模型给出 0.991 的相对概率与已知的概念模型修订结果完全一致。

**⚠️ 局限性**

局限性包括：对文本表述的敏感性（不同描述可能导致不同图像集）；仅使用 2D 稳态单相流模型，无法区分需转移或多相效应的解释；图像强度仅映射相对导电率范围，缺乏绝对数值校准；兼容性分数受模型误差阈值设定影响；需要在复杂地质中额外提供边界轮廓。

---

## 77. Attention-Aware Routing: Coupling Routing and Attention in MoEs

**arXiv ID:** 2609.20974 | [PDF](https://arxiv.org/pdf/2609.20974v1)

**作者:** Despoina Kosmopoulou `[一作]` (National Technical University Of Athens), Alexandros Potamianos `[通讯]` (National Technical University Of Athens)

**通讯引用:** 6111 | [OpenAlex ID](https://openalex.org/A5084949286)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在混合专家（MoE）大型语言模型中，引入注意力感知路由（AAR）机制，使路由器在做专家选择时同时利用注意力窗口信息，从而实现对模型行为的可塑性调整。

**💡 创新点**

创新点在于将注意力窗口（时间域与频谱域）作为路由器的额外输入，实现隐藏状态与上下文信息的解耦；通过仅训练路由参数而保持Transformer冻结，展示路由层面调控可以显著提升数学推理能力。

**🔧 技术方法**

使用技术包括滑动窗口注意力提取、频域特征（DFT幅值）、线性路由器与门控机制、以及对多层AAR的层级选择策略。

**📊 数据集**

实验数据集包括25% Tulu3 训练样本以及评测基准GSM8K、BBH、MMLU、MATH-500、HumanEval和IFEval。

**📈 对比分析**

与仅训练原始路由器的SFT基线相比，AAR在GSM8K上提升约3.37个百分点；在MATH-500上提升2.64个百分点；在更大模型Qwen3.6-MoE上提升1.13个百分点，同时保持其他基准性能不变。

**⚠️ 局限性**

局限性包括：需要显式计算注意力权重，无法使用FlashAttention；仅在SFT后进行路由训练，深度选择对不同模型差异较大；在早层应用会削弱事实检索能力。

---

## 78. SKYE: Write-Optimized Key-Value Store with Fine-Grained Control over Persistent Memory Accesses

**arXiv ID:** 2609.20972 | [PDF](https://arxiv.org/pdf/2609.20972v1)

**作者:** Soujanya Ponnapalli `[一作]` (University of Texas at Austin), Vijay Chidambaram `[通讯]` (University of Texas at Austin)

**通讯引用:** 1496 | [OpenAlex ID](https://openalex.org/A5010361101)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种写优化的持久内存键值存储，通过间接访问和细粒度控制PM访问实现高吞吐

**💡 创新点**

核心创新是将PM访问从应用层转移到专用工作线程，管理非交斥的NVDIMM并在单独日志上顺序写入，同时使用DRAM索引和批处理来平衡读写

**🔧 技术方法**

使用间接访问架构、固定工作线程、日志抽象、非交叉NUMA节点、非互斥PM设备、DRAM/磁盘索引、Bloom过滤器、批处理和内存合并日志

**📊 数据集**

在Intel Optane DC持久内存上使用YCSB 8B键/256B值的LoadA、RunA/B/C/D/F工作负载

**📈 对比分析**

与FlatStore、Viper和ChameleonDB对比，写入吞吐提升2–5倍、写入带宽利用率达88%，在4个NUMA节点上写入吞吐可提升3.9倍，读写性能均优于现有PM存储

**⚠️ 局限性**

局限在于牺牲低延迟（单个操作数十微秒），CPU利用率高，需要大量工作线程，且对特定PM硬件和配置敏感

---

## 79. RBS-Attention: Radius-Bounded Sparse Prefill for Long-Context Large Language Models

**arXiv ID:** 2609.20971 | [PDF](https://arxiv.org/pdf/2609.20971v1)

**作者:** Chuxu Song `[一作]` (Rutgers University), Zhencan Peng `[通讯]` (Rutgers University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关的稀疏长上下文预填充方法 RBS（Radius‑Adaptive Dual‑Branch Selector），在 block‑sparse FlashAttention 的基础上结合了块中心点（centroid）得分和半径（radius）修正的救援分支，提升对稀疏块中单个高相关 token 的捕获能力。

**💡 创新点**

创新点在于：①引入半径自适应救援分支，针对中心点得分被平均稀释（mean dilution）严重的块自动提升得分；②使用独立阈值对两条分支进行相对阈值裁剪并取并集，兼顾平均相关性与块内离散性；③保持与现有 block‑sparse FlashAttention 的兼容性，避免额外的 GPU 运行时开销。

**🔧 技术方法**

采用的技术包括：块级平均（centroid）得分、块半径（max L2 直径）计算、基于分位数的半径归一化系数 β、两分支（base 与 rescue）相对阈值裁剪、掩码并集、block‑sparse FlashAttention 内核、以及对 CUDA GPU（H100/A100）进行的系统层面加速。

**📊 数据集**

使用的数据集与基准：Qwen3‑30B‑A3B‑Instruct‑2507‑FP8、Qwen3‑32B、Qwen3‑VL‑30B‑A3B‑Thinking‑FP8；评测基准包括 RULER、LongBench‑v2、InfiniteBench、Video‑MME，此外在 H100 上使用 Qwen3‑30B‑A3B‑Instruct‑2507‑FP8 进行大规模上下文（up to 256K）加速实验。

**📈 对比分析**

比较方法：在同一硬件（H100）和同一模型下与 FlashPrefill、FlexPrefill、MInference、XAttn 等现有稀疏预填方法以及 dense attention 进行对比；指标包括：standalone prefill‑attention speedup、vLLM prefill‑attention speedup、TTFT（time‑to‑first‑token）speedup、以及 RULER/LongBench‑v2 等任务的准确率。性能表现：RBS 在 128K 上实现 20.65× prefill‑attention、11.92× vLLM prefill‑attention、5.97× TTFT 的加速；在准确率方面，RBS 的 RULER 整体得分 88.65% 与 dense 89.52% 仅相差 0.87%，LongBench‑v2 0.376 与 dense 0.394 差距 0.018，展示了高效与高质量兼顾的能力。

**⚠️ 局限性**

局限性：①在短上下文（≤16K）时稀疏加速收益有限甚至低于 dense；②阈值与块尺寸对性能与质量高度敏感，需在不同模型、层/头、提示上进行校准；③未在 decode‑time 进行评测，当前方案专注于预填充阶段；④某些诊断实验样本有限，统计置信度较低。

---

## 80. Don't Blame the Model, Verify the Data: An Evaluation of SMT-based Dataset Verification

**arXiv ID:** 2609.20959 | [PDF](https://arxiv.org/pdf/2609.20959v1)

**作者:** Sehee Park `[一作]` (Technische Universität Berlin), Kim Völlinger `[通讯]` (Technische Universität Berlin)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5066537166)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对基于 SMT 的数据集验证进行了大规模经验研究，系统评估了数据质量属性类型、规格写法和编码策略对求解器性能的影响。

**💡 创新点**

首次在真实工业级数据集上量化 SMT 验证的可扩展性，并揭示三大维度（属性类型、规格风格、编码方式）如何决定可达规模与性能差异；提出针对非线性连续关系属性的代数预处理（根消除）和针对记录/聚合属性的“grounding”规范风格。

**🔧 技术方法**

利用 Z3 SMT 求解器实现三种编码（二维数组、单列切片、嵌套列切片），三种规格写法（原始量化、递归、完全展开），并结合根消除优化和非线性预处理；通过构造400+个 SMT-LIB 基准实例完成实验。

**📊 数据集**

德国信用（1,000 条记录，24 个特征）和银行营销（45,211 条记录，16 个特征）两大工业机器学习数据集。

**📈 对比分析**

实验涵盖 22 组设置，比较求解时间与内存，发现：1）记录级与聚合属性在 grounding 写法下可处理数万记录；2）与基线相比，grounding 速度提升 2,000 倍；3）根消除将非线性关系属性可验证范围从 700 条提升至 10,000 条；4）列切片编码比二维数组快数十倍。

**⚠️ 局限性**

仅使用单一求解器（Z3）与单台消费级机器，结果为相对比较；对多求解器性能、增量验证、跨数据集属性、时序/图结构数据等场景仍需进一步探索。

---

## 81. Physically Based Rendering in the Latent Space

**arXiv ID:** 2609.21054 | [PDF](https://arxiv.org/pdf/2609.21054v1)

**作者:** Vuk Radovanovic `[一作]` (Trinity College Dublin), Binh-Son Hua `[通讯]` (Trinity College Dublin)

**通讯引用:** 3192 | [OpenAlex ID](https://openalex.org/A5028533837)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出“潜在渲染”框架，在VAE潜在空间中直接使用物理基础渲染方程生成潜在特征图，随后通过神经细化器和解码器生成最终RGB图像，并实现对摄像机、光源和物体的编辑。

**💡 创新点**

创新点包括：① 将光照传输方程改写为带符号辐射与反射、平坦响应项和遮挡项，使其能直接匹配潜在空间的正负值和结构特征；② 用可微渲染器联合全局优化场景参数（光源、BSDF、平坦/遮挡参数），实现单视角训练后对场景变体的泛化；③ 引入专门的潜在细化网络，只在单场景单视角上训练，显著提升重建质量；④ 在潜在空间直接进行SDS（score distillation sampling）优化，避免RGB编码回潜在的开销。

**🔧 技术方法**

技术：可微物理渲染（Mitsuba 3 + 路径回放）、签名辐射与BSDF、平坦响应与遮挡项、潜在空间的Gamma校正、Huber损失、神经多层感知机细化器、LPIPS/SSIM/MSE评估、SDS集成。

**📊 数据集**

数据集：Mitsuba 3 场景集（Cornell Box、Lamp、Veach-Bidir、Dining Room、Living Room），VAE使用 Stable Diffusion 3.5 的预训练编码器（128×128×16 潜在分辨率），每个场景仅用一幅高采样RGB图像做训练视角。

**📈 对比分析**

比较方法：与传统渲染+潜在归一化→解码的基线对比；在同一视角和编辑场景（相机、物体、光源移动）下计算LPIPS、MSE、SSIM；runtime 对比（RGB渲染 vs 潜在渲染+细化）。结果显示：潜在渲染+细化在潜在空间误差上显著低于基线（LPIPS≈0.02-0.07 vs 0.4-0.6），在编辑场景下误差保持稳定；在潜在空间的渲染速度比RGB渲染快 42–84 倍，但解码后的RGB渲染仍比直接RGB路径追踪慢 6–35 倍。

**⚠️ 局限性**

局限性：① 潜在空间低分辨率导致细节锯齿；② 细化器只在单场景单视角训练，跨场景泛化不足；③ 解码器非线性导致最终RGB有偏差；④ 计算量主要集中在解码器，影响性能；⑤ 对高频细节、纹理细化不够；⑥ 目前只支持Stable Diffusion 3.5 的VAE，需要进一步适配其他模型。

---

## 82. LoRA Enhanced Contrastive Learning with SAS Vision Transformers

**arXiv ID:** 2609.21061 | [PDF](https://arxiv.org/pdf/2609.21061v1)

**作者:** Dan Zimmerman `[一作]` (Florida Atlantic University), Gregory D. Vetaw `[通讯]` (Naval Surface Warfare Center)

**通讯引用:** 7 | [OpenAlex ID](https://openalex.org/A5093097273)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过三阶段参数高效适配框架，将预训练的DINOv3 Vision Transformer迁移到双频段合成孔径声纳ATR任务。

**💡 创新点**

创新点在于发现仅使用低秩LoRA适配就能获得+0.38的AUPRC提升，其余后续阶段对性能无显著贡献，并系统验证了硬负样本挖掘与监督对比学习的无效性。

**🔧 技术方法**

采用的技术包括LoRA参数适配、硬负样本挖掘、监督对比学习、数据增强以及对抗性微调等。

**📊 数据集**

使用了真实海上双频段SAS图像数据集，共148个任务文件，约336k个片段，包含2,791个目标。

**📈 对比分析**

通过与多种基线（ResNet18、TinyViT等）以及同一模型不同阶段的匹配对照进行比较，最终LoRA适配后AUPRC从0.30提升至0.68，精度和召回均优于全微调。

**⚠️ 局限性**

限制在于后续的硬负样本挖掘和监督对比学习在当前数据规模下无效，且实验对验证集的多重使用可能导致轻微的选择偏差。

---

## 83. How Much of a Real Workload Can LLM-Generated GPU Kernels Actually Reach?

**arXiv ID:** 2609.21058 | [PDF](https://arxiv.org/pdf/2609.21058v1)

**作者:** Gaurav Agarwal `[一作]`, Isha Singhal `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `79276348-11e0-48e3-84bc-7ec231d0171c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了自动GPU kernel生成模型在KernelBench level 1上的通过率与加速，并量化了这些kernel在实际Transformer、CNN和推荐系统工作负载中的可被替代的运算比例，评估其对端到端性能的实际影响。

**💡 创新点**

①提出“可寻址比例”概念，用来衡量生成kernel对整体模型加速的实际意义；②发现并修正了KernelBench在bf16下可被零输出“骗过”的正确性漏洞；③构建了DLRM-Bench专门针对推荐系统的内核基准；④系统记录并分类了10种测量错误，为后续工作提供了经验教训。

**🔧 技术方法**

使用大规模语言模型（Qwen2.5‑Coder、DeepSeek‑R1‑Distill‑Qwen、GPT‑5.6）与Triton编译器生成GPU kernel，采用BF16精度，CUDA 13.0，在A100 80 GB GPU上进行评估；同时开发了独立验证器，对相对/绝对误差、写入比例和运算不变性进行校验，并与PyTorch、Inductor等基线进行对比。

**📊 数据集**

基准数据集包括KernelBench level 1（60个单核算子）、DLRM-Bench（12个推荐系统内核问题）以及七个真实工作负载（Transformer、CNN、推荐系统的训练/推理），全部在同一硬件平台上收集。

**📈 对比分析**

采用pass@1、正确率和中位数加速进行比较；GPT‑5.6在KernelBench上达91.1%通过率，平均加速1.235×，但在Transformer中的可寻址比例仅8.9–16.8%，端到端提升约1%；在推荐系统中的可寻址比例为58.2%，端到端提升约8.6%；在CUDA后端的速度提升与Triton相比从0%跃升至10.7%。

**⚠️ 局限性**

实验仅使用单一A100 GPU和单一模型族，未覆盖更高级别（level 2/3）或多GPU场景；基准仅采用单一批次和固定形状，无法反映不同批次/形状的性能变化；分类依据是基于内核名称的模式匹配，存在误分类风险；开源模型的性能可能随时间下降，导致结果的时间敏感性。

---

## 84. Scaling Discovery through Test-Time Communication

**arXiv ID:** 2609.21032 | [PDF](https://arxiv.org/pdf/2609.21032v1)

**作者:** Jongho Park `[一作]`, Dimitris Papailiopoulos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在测试时通过共享工作空间进行无角色、无中央调度的多智能体通信，比较团队通信与独立搜索在 ARC-AGI-3、Polyomino Packing 与 MNIST Classifier Compression 三类开放式任务上的表现。

**💡 创新点**

提出了“已验证进展共享”(verified progress sharing) 的通信协议，证明了在可获得中间验证信号且计算充足时，团队通信能实现指数级加速；并首次在三种任务中通过团队通信取得了新的最优成绩。

**🔧 技术方法**

使用大语言模型（Claude Sonnet/Opus、GPT‑5.6 Sol）作为智能体，利用 GitHub Copilot CLI 与共享文件系统实现异步通信；通过原子文件夹占用、日志广播、共享得分板等机制保障多样性与安全共享；在每个任务中保持相同的模型、工具与资源分配。

**📊 数据集**

ARC‑AGI‑3（25 个无指导格子游戏），Frontier‑CS Polyomino Packing（70 个隐藏测试用例），MNIST Classifier Compression（标准 MNIST 数据集，要求 ≥99.4% 准确率），以及对比实验中的 Terminal‑Bench 2.0。

**📈 对比分析**

与独立搜索（best@k）对比，团队通信在相同的 per‑agent 预算下：team@3 与 13 个独立 agents 的成功率相当，team@5 与 33 个独立 agents 的成功率相当；Polyomino Packing 取得 0.945 的得分，超过先前最佳 0.894；MNIST 压缩得到 1,957 字节的模型，优于 2,461 字节的人类最佳方案。整体显示团队通信在成功率、效率和最终性能上都显著优于独立搜索。

**⚠️ 局限性**

缺点包括：初期协调成本高，需足够计算资源才能显现优势；若任务缺乏可验证的中间反馈（如 Terminal‑Bench 2.0），通信优势消失；实验仅使用同质化智能体与固定通信拓扑，未探讨角色分配、异构模型或动态团队组织；在反馈稀疏或噪声较大的环境下效果未知。

---

## 85. Adapting Rigid-Body Dynamics Derivatives for Constraint Embedding Closed-Chain Models

**arXiv ID:** 2609.21024 | [PDF](https://arxiv.org/pdf/2609.21024v1)

**作者:** Daniel J. Volpi `[一作]` (University of Notre Dame), Patrick M. Wensing `[通讯]` (University of Notre Dame)

**通讯引用:** 4998 | [OpenAlex ID](https://openalex.org/A5029126529)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究将现有的刚体动力学一阶导数算法扩展到通过约束嵌入建模的闭链系统，剔除了关节运动子空间在本地坐标中配置不变的假设，并推导了适用于一般关节类型的导数公式。

**💡 创新点**

创新点在于：①去掉了传统算法对关节运动子空间配置不变的假设；②得到的导数公式能够直接应用于闭链约束嵌入模型；③将新算法实现为开源 GRBDA 库，便于在闭链驱动子机机制中使用。

**🔧 技术方法**

技术上采用空间向量和 Lie 代数框架、约束嵌入方法、递归动力学求导、复杂步法验证，并利用 GRBDA 与 Pinocchio 等库实现。

**📊 数据集**

实验主要在仿真机器人模型上进行，对比标准单轴关节模型与完整闭环驱动模型；未使用公开数据集。

**📈 对比分析**

通过与复杂步差分法比较验证导数正确性；在仅建模驱动运动学时，额外计算成本低；当加入电机转子或考虑非局部闭环时，计算成本略增，但整体仍保持可接受的实时性能。

**⚠️ 局限性**

局限性包括：仅推导并评估逆动力学的导数；在包含大量刚体或大规模闭环时计算开销显著；自动化约束 Jacobian 的生成仍需人工指定，自动化程度有限。

---

## 86. MOSAIC-SR: Transformer-Guided Symbolic Regression for Scientific Equation Recovery

**arXiv ID:** 2609.20997 | [PDF](https://arxiv.org/pdf/2609.20997v1)

**作者:** Peiyi Zheng `[一作]` (University of Waterloo), Giang Tran `[通讯]` (University of Waterloo)

**通讯引用:** 1853 | [OpenAlex ID](https://openalex.org/A5109179791)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种结合预训练 Transformer 生成草图与局部搜索修复的混合符号回归框架 MOSAIC-SR。

**💡 创新点**

创新点在于：① 用对比学习预训练的 Transformer 产生多样化结构草图；② 通过 MCTS 为草图分配实际变量并初始化多条搜索路径；③ 在搜索过程中交替进行尺度感知常数优化与树编辑修复，显著提升符号解率。

**🔧 技术方法**

使用的技术包括：对比预训练的 Transformer、MCTS 变量分配、Levenberg–Marquardt 常数优化、树编辑局部搜索、符号验证器。

**📊 数据集**

实验数据集包括 SRSD‑Feynman（含 dummy 变量变体）以及 Nguyen‑12、Korns‑15、Keijzer‑15、Vladislavleva‑8、Strogatz‑14、Livermore‑22 等六大标准符号回归基准。

**📈 对比分析**

与传统搜索、纯神经网络和混合方法对比，MOSAIC‑SR 在所有数据集上实现最高符号解率，并在预测准确度上名列前两；在加入无关变量的情况下仍保持显著优势；与 PySR 等基线相比，速度更快、恢复效果更稳健。

**⚠️ 局限性**

局限性包括：对高维、结构复杂的方程恢复仍困难；对训练分布之外的未知形式恢复受限；需要较大的搜索预算；未来需要扩大预训练分布并研究稀疏观测下的恢复。

---

## 87. Helpful but Fallible: Developer Experiences of AI Tools Under a Coordinated Industrial Roll-out

**arXiv ID:** 2609.20977 | [PDF](https://arxiv.org/pdf/2609.20977v1)

**作者:** Andreas Bexell `[一作]` (Lund University), Konstantin Malysh `[通讯]` (Lund University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5115676632)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

进行了在一家大型瑞典电信公司的人工智能开发工具的协同推行案例研究，收集并分析了12名软件专业人士的访谈数据，探讨他们的体验和预期。

**💡 创新点**

通过实地访谈揭示了管理层与开发者之间的期望差距、持续的风险评估以及“感知风险”作为AI工具接受模型中的缺失构件；首次将TAM2与风险框架结合解释工业化AI工具的接受。

**🔧 技术方法**

采用半结构化访谈、转录与人工纠错、主题分析（过程编码+主题编码）以及TAM2模型作为后期解释框架。

**📊 数据集**

12名来自三地的开发者访谈记录，涵盖不同角色与经验，转录并译为英文的访谈文本。

**📈 对比分析**

未采用量化实验或性能指标，而是与先前文献的定量结果（如生产率提升、使用场景、挫败感）进行比较，结果与现有研究高度一致，进一步验证了现象的普遍性。

**⚠️ 局限性**

样本仅来自单一公司与单一行业，缺乏管理层视角，缺少长期纵向追踪，且TAM2为事后映射，可能导致结构性偏差。

---

## 88. An Approximate Queueing Model of LLM Inference Serving for SLO-Driven Autoscaling

**arXiv ID:** 2609.20957 | [PDF](https://arxiv.org/pdf/2609.20957v1)

**作者:** Vishakha Ramani `[一作]` (IBM T. J. Watson Research Center), Asser N. Tantawi `[通讯]` (IBM T. J. Watson Research Center)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种可解析的队列模型，用于预测 LLM 推理服务器的首令延迟 (TTFT) 与间令延迟 (ITL)，并将其嵌入 SLO 驱动的自适应扩缩容控制循环。

**💡 创新点**

创新点在于：① 将预填充与解码的两阶段计算拆解为三参数（基础迭代开销、每令计算成本、KV 缓存访问成本）模型；② 通过状态依赖的马尔可夫链结合平均值分析，既可处理无限预填充也可处理分块预填充；③ 在控制循环中使用在线滑动窗口重估模型参数，实现无监督的自校准。

**🔧 技术方法**

主要技术包括：连续批处理 (continuous batching) 的调度模型、均值场近似的马尔可夫链分析、Nelder–Mead 参数拟合、k8s 自动伸缩与自适应控制循环。

**📊 数据集**

使用了 Llama‑3.1‑8B 与 Qwen2.5‑14B 两个大模型，在 H100 GPU 上通过 vLLM 服务器生成 224 个工作点（不同输入/输出长度组合与 Poisson 到达率），并收集 TTFT、ITL 数据。

**📈 对比分析**

与传统的解码吞吐量分析器（decode‑throughput analyzer）对比：在 TTFT‑绑定场景下，队列模型在 64 次循环中仅 7 次超出 SLO，吞吐量分析器 22 次；在 ITL‑绑定场景下分别为 0 次和 5 次；平均占用的 GPU 实例更少（队列模型 1.88 vs 1.81、ITL 场景 2.27 vs 1.64）。预测误差为 TTFT ≤16% 及 ITL ≤8%。

**⚠️ 局限性**

局限性包括：① 仅使用平均输入/输出长度，难以捕捉宽分布或多峰长度请求；② 依赖 Poisson 到达率，无法准确建模真实工作负载中的突发性；③ 近似的平均占用状态在接近饱和时误差增大；④ 只评估单一模型/加速器组合，未验证跨模型泛化。

---

## 89. Refined complexity bounds for rational reconstruction and XGCD through Padé approximants and Cauchy interpolants

**arXiv ID:** 2609.21051 | [PDF](https://arxiv.org/pdf/2609.21051v1)

**作者:** Vincent Neiger `[一作]` (Sorbonne Université), Kevin Tran `[通讯]` (Sorbonne Université)

**关键词:** `847a60d8-a755-47af-ba5d-c5236b9e3083` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过将 rational reconstruction 与 XGCD 归约为任意模数下的关系基问题，结合 Padé 逼近、Cauchy 插值以及分治递归与快速多项式运算，提出了改进的算法并给出了精细的复杂度上界。

**💡 创新点**

创新点在于：①将 XGCD 与 rational reconstruction 统一为关系基问题，②利用 Cauchy 插值点和外推技术在分治过程中显著降低主项常数，③给出了最优的主项常数（包括针对泛型与特殊点集的两种分析），并通过算法重构证明这些常数的可实现性。

**🔧 技术方法**

采用的技术包括：Beckermann‑Labahn 递归分治、快速多项式乘法（FFT 与基于 Strassen 的实现）、中间乘积、FFT 评估/外推、快速幂级数除法与幂级数乘法、以及关系基与弱 Popov 形式的变换。

**📊 数据集**

论文主要是理论分析，没有使用特定的实验数据集；实验验证使用 FLINT 库在有限域（如 2^64 以内的素数域）上对 XGCD 与 Berlekamp‑Massey 进行基准测试。

**📈 对比分析**

理论上，新算法在最坏情况和泛型情况下均取得了更低的主项常数；实验结果显示在 FLINT 上实现后，XGCD 与 rational reconstruction 的运行时间明显优于原始实现，尤其在大型多项式（几十万级）时提升显著。

**⚠️ 局限性**

局限性包括：①需要模数与点集满足可分解或可评估的形式，②常数改进的最优性在所有字段上尚未完全证明，③在实际实现中仍受限于底层多项式乘法的实现方式；此外，对于极端非泛型输入，仍需额外的特殊处理。

---

## 90. (Don't) Trust, but (Don't) Verify: Developers' Attention to Security in AI-Generated Code

**arXiv ID:** 2609.21020 | [PDF](https://arxiv.org/pdf/2609.21020v1)

**作者:** Hamza Khalid `[一作]` (Tufts University), Daniel Votipka `[通讯]` (Tufts University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究探讨开发者在使用 AI 代码生成工具时，如何评估 AI 生成代码的安全性。通过对 100 名参与者的在线观察实验，记录他们在四个 C 语言链表任务中对 5 个 AI 建议的选择、编辑、测试行为，并收集问卷与访谈数据，分析其安全评估过程与信任程度。

**💡 创新点**

聚焦 AI 代码评估这一安全交互的核心环节，并首次结合细粒度日志、问卷与访谈三重视角，对评估行为进行定量与定性解析；揭示开发者在评估时过度依赖表面线索、忽视真正的安全检查，并提供改进 AI 交互与安全支持的设计建议。

**🔧 技术方法**

实验平台：自定义 NERDS 系统（模拟 VS Code + 终端 + 浏览器），记录代码、操作、时间、资源访问；统计分析使用 Poisson、逻辑回归与线性回归；安全审计采用人工代码评审；访谈与问卷采用开放式编码与 Krippendorff α 校准。

**📊 数据集**

参与者：100 名（其中 88 来自 Upwork，12 来自大学），平均 7 年编程经验与 1.8 年安全经验；任务数据：400 条代码提交（每人 4 个链表任务），AI 建议共 120 条来自 Sandoval 等 2026 年评测的真实 LLM 生成代码，按安全度与功能性分层挑选 5 条每任务；实验日志与访谈录音可在 OSF 公开。

**📈 对比分析**

比较方法：通过回归检验 AI 建议安全度、编辑量、经验水平等对最终提交漏洞数的影响；结果显示：选取更安全建议与更大编辑量均显著降低漏洞数（β≈‑0.5，p<0.001）；但整体安全率仅 22%（完全无漏洞），仅 5% 的参与者四个任务全无漏洞；不论信任度如何，评估缺陷导致安全性低下。

**⚠️ 局限性**

局限性：实验为受控实验室环境，任务规模小且单一；使用预先生成的建议而非实时交互；参与者主要为学生/自由职业者，缺乏大型企业工程师；未限制外部资源使用，可能混淆评估行为；仅关注评估阶段，未覆盖后续迭代或 AI 指令优化等实际开发情境。

---

## 91. MAGIC: Marginal-Guided Compression with Optimal Transport for Efficient Visual Document Retrieval

**arXiv ID:** 2609.21018 | [PDF](https://arxiv.org/pdf/2609.21018v1)

**作者:** Xu Yuan `[一作]` (Hong Kong Polytechnic University), Qing Li `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种无训练的后置压缩方法MAGIC，用于冻结的多向量视觉文档检索，将原始页面的数百个补丁向量压缩为少量代表向量，同时保持检索质量。

**💡 创新点**

创新点在于将检索需求（补丁被查询向量选中的频率）和保持向量使用的均衡性通过双边边缘最优传输（two‑marginal OT）统一建模，实现检索对齐的容量分配；并提出了基于MaxSim的压缩近似目标和可行的软分配求解。

**🔧 技术方法**

主要技术包括：MaxSim诱导的压缩近似目标、检索需求估计（通过校准查询向量的软最大化）、两边边缘熵正则化的最优传输（使用Sinkhorn迭代）、球面质心更新、球面Lloyd读取。

**📊 数据集**

使用的评估数据集为ViDoRe v1、v2（多语言）以及ViDoSeek（推理密集型），其中ViDoRe是主实验基准。

**📈 对比分析**

与多种后置压缩基线（如Light‑ColPali、DocPruner、K‑Means等）在保持向量预算相同的条件下进行对比，MAGIC在r=0.1和r=0.01两种压缩比例下分别实现了最高的nDCG@5和Recall@5，并在压缩比例越紧的情况下提升幅度更显著。

**⚠️ 局限性**

局限性包括：需要预先构建校准查询向量池来估计检索需求；对温度、熵正则化等超参数仍有一定敏感性；虽然离线索引时间相对较短，但在极低预算下仍可能出现稀疏表示导致的检索误差。

---

## 92. Robustness Analysis via Horofunction Compactification

**arXiv ID:** 2609.21009 | [PDF](https://arxiv.org/pdf/2609.21009v1)

**作者:** Harrison Bennett `[一作]` (University of Birmingham), Amin Farjudian `[通讯]` (University of Birmingham)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出利用Gromov的horofunction紧致化构造非局部紧致度量空间的稳健性分析框架。

**💡 创新点**

创新点在于引入可Lipschitz嵌入的horofunction紧致化，并给出ℓ_p空间下闭集格的可数基，显著降低精度损失。

**🔧 技术方法**

主要技术包括域理论、稳健拓扑、度量紧致化、Horofunction映射与Lipschitz连续性、Ascoli定理与可数稠密子集构造。

**📊 数据集**

本文未使用具体实验数据集，而是在ℓ_p空间的理论示例中进行验证。

**📈 对比分析**

通过对比先前构造的闭集格，示例表明horofunction方法在保留信息上优于旧方法，理论上实现更高精度。

**⚠️ 局限性**

局限在于未给出一般ℂ(S^h)的可有效结构，也未处理L_p(X)空间，仅适用于可分度量空间。

---

## 93. ForeTac-VLA: A Forecasting-Based Tactile-Vision-Language-Action Model for Contact-Rich Robotic Manipulation

**arXiv ID:** 2609.20980 | [PDF](https://arxiv.org/pdf/2609.20980v1)

**作者:** Zhengyu Tao `[一作]` (Texas A&M University), Xin Wang `[通讯]` (Texas A&M University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于未来触觉预测的视觉-语言-动作（VLA）模型 ForeTac‑VLA，专门用于接触丰富的机器人操作任务。

**💡 创新点**

创新点在于：① 通过 Transformer 预测多步未来触觉状态；② 双向交叉注意力实现触觉与视觉语言的深度融合；③ 用预测的触觉信息直接驱动动作生成；④ 采用从真实触觉到预测触觉的两阶段课程学习稳定训练。

**🔧 技术方法**

技术细节包括：预训练 π_0.5 VLA 框架、SigLIP 视觉编码器、PaliGemma 语言+关节编码、多层感知机压缩触觉、Transformer 未来触觉预测、双向交叉注意力、LoRA 微调以及流匹配的动作专家。

**📊 数据集**

数据集为自建的 280 条遥控演示（每任务 70 条），涵盖四个真实机器人任务（Peg Insertion、Chip Handling、Cap Unscrewing、Board Wiping），并在每个任务下进行 20 次评估测试。

**📈 对比分析**

实验通过与 fine‑tuned VLA、FiLM‑fusion TacVLA、concat‑and‑gating TacVLA 和 ACT 四个基线模型对比，使用任务成功率评估。ForeTac‑VLA 平均成功率达 95%，比 fine‑tuned VLA 提升 36.25%、比 FiLM‑fusion 提升 23.75%、比 concat‑and‑gating 提升 22.5%，在低照明和视觉杂乱环境中仍保持 80–81% 的成功率，明显优于其他模型。

**⚠️ 局限性**

局限性包括：仅在单一触觉传感器配置和机器人上验证；未来触觉预测使用固定时间间隔，未适应可变动态；未在更长时延任务或更广泛对象上进行评估；缺乏多传感器或更复杂场景的泛化测试。

---

## 94. MemeTAG: Keyword-Driven Meme Classification through Tag Embedding Reconstruction

**arXiv ID:** 2609.20962 | [PDF](https://arxiv.org/pdf/2609.20962v1)

**作者:** Akshit Sharma `[一作]` (Indian Institute of Technology Guwahati), Prashant W. Patil `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于关键词的双目标多模态框架MemeTAG，用于自动识别网络恶意表情包。

**💡 创新点**

创新点包括：①利用预训练视觉-语言模型生成关键词并通过ATIN聚合成语义向量；②引入辅助重构损失，使视觉、文本特征与关键词语义保持一致；③三阶段预计算+聚合+训练的策略，显著降低计算开销并提升收敛稳定性。

**🔧 技术方法**

技术主要有：CLIP预训练视觉与文本编码器、Multi‑Head Self‑Attention（MHSA）聚合网络ATIN、注意力融合与ArcFace式分类头、余弦重构损失以及超参平衡λ。

**📊 数据集**

使用了三个公开基准数据集：PrideMM（LGBTQ+表情包多标签分类）、HarMeme（COVID‑19 相关仇恨表情包二分类）和HMC（Hateful Memes Challenge）进行评估。

**📈 对比分析**

与现有多模态模型（如MemeCLIP、HateCLIPper、ISSUES、GuardHarMem等）对比，MemeTAG 在三组数据集上均实现了最高或近乎最高的 Accuracy/AUROC/F1，分别提升了约1.4%、1.5% 和 3–4% 的指标。

**⚠️ 局限性**

局限性：①依赖 CLIP 等大规模预训练模型，继承其种族、文化与语言偏见，易产生误报；②易受对抗性编辑干扰；③表情包样式和流行语变化迅速，需要频繁更新与再训练。

---

## 95. Clinician-Grounded Quality Assurance for AI-Assisted Psychiatric Intake

**arXiv ID:** 2609.21149 | [PDF](https://arxiv.org/pdf/2609.21149v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 96. Efficient Bayes-Adaptive Reinforcement Learning with Temporal Logic Specifications

**arXiv ID:** 2609.20954 | [PDF](https://arxiv.org/pdf/2609.20954v1)

**作者:** Jonathan Hau `[一作]` (University of Oxford), Alessandro Abate `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了一种基于贝叶斯模型的 LTL 约束强化学习框架 BA‑LCRL，能够在未知环境中同步 LDBA 与 BAMDP 并通过 P‑BAMCP 近似 Bayes‑optimal 策略，从而实现对 LTL 任务的高效策略合成。

**💡 创新点**

创新点包括：① 将 Bayes‑adaptive MDP 与 LDBA 直接同步得到 Product BAMDP；② 开发了针对 Product BAMDP 的 P‑BAMCP 算法，提升了探索-利用平衡和 LTL 任务满足率；③ 通过潜在函数奖励塑形和一阶安全性检查实现了稀疏奖励缓解与谨慎 RL 的双重优化。

**🔧 技术方法**

核心技术包括：LTL‑to‑LDBA 转换、贝叶斯推断（Dirichlet‑Multinomial 或神经网络集成）、P‑BAMCP（基于 MCTS 的贝叶斯规划）、奖励塑形与安全阈值一阶回溯、Q‑学习或 DQN 用于价值更新。

**📊 数据集**

实验数据集主要包含：离散 10×10 溜滑格网（slippery‑grid）以及连续 Cartpole 环境，并在每个环境中为不同 LTL 公式构造状态标签。

**📈 对比分析**

与传统无模型的 LCRL 基线相比，BA‑LCRL 在所有测试任务（有限/无限期）中均显著提升了满足概率（PSP）并加速收敛；在安全性实验中，加入一阶回溯的 Cautious BA‑LCRL 进一步降低了违规次数，平均 PSP 也更高。

**⚠️ 局限性**

局限性主要在于：① P‑BAMCP 的搜索复杂度随状态空间和自动机大小呈指数增长，限制了大规模或高维连续任务的直接应用；② 需要显式的状态标签和 LTL 语义映射，若标签缺失或动态变化则难以使用；③ 目前仅在较简单的 LTL 公式和小型环境中验证，尚未展示在真实机器人或大规模真实场景中的可扩展性。

---

## 97. SpecOpt: Contact-Diff Reasoning for Agentic Molecule Optimization Toward Binding Specificity

**arXiv ID:** 2609.21165 | [PDF](https://arxiv.org/pdf/2609.21165v1)

**作者:** Thao Nguyen `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了专门针对已有药物的选择性优化任务SpecOpt，并实现了一个基于差异化结构分析和LLM编辑的迭代框架。

**💡 创新点**

将选择性优化定义为在保持分子相似度和药物性质的前提下，通过有针对性的结构修改提升目标与已知离靶的结合偏好；并发现残基感知差异接触是最关键的优化信号。

**🔧 技术方法**

利用分子对接（AutoDock Vina）、基于残基的接触差异特征、LLM（GPT‑5.4）生成分子编辑、ADMET预测（mCLM）以及布尔过滤等技术。

**📊 数据集**

基于ChEMBL的1,848种靶标‑离靶配对数据，筛选后得到916种可对接的药物，构成BenchSpecOpt。

**📈 对比分析**

与原始药物进行对接对比，优化后目标‑离靶能量差从-0.72提升至+0.47 kcal mol⁻¹；84.8% 的药物提升了选择性，平均 Tanimoto 相似度为 0.72。

**⚠️ 局限性**

只针对最多三个离靶进行优化，评价仅基于 Vina 得分，未涉及实验验证或更高精度的自由能计算，且对多离靶优化的效果有限。

---

## 98. Same World, Different Knowledge: When Isolated Audits Misjudge World-Model Repairs

**arXiv ID:** 2609.21155 | [PDF](https://arxiv.org/pdf/2609.21155v1)

**作者:** Rui Min `[一作]` (University of Florida), Jing Du `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了信息接口审计方法，区分精度与可用性缺口，并通过固定权重干预评估世界模型在受损输入下的控制性能。

**💡 创新点**

创新点在于将输入依赖与闭环收益分离，构建可共享信息路径的审计规范，以及引入不确定性训练（DCC）与重建模块对比研究。

**🔧 技术方法**

使用的技术包括固定权重干预、不可测量输入的扰动训练（DCC）、基于GRU的风重建、扰动观测器、模型预测控制（MPC）和MPPI规划。

**📊 数据集**

实验数据来自Isaac Sim仿真，包含7,150条带随机质量/推力/惯性/阻力的轨迹，另加14,891条随机风场的数据。

**📈 对比分析**

通过对比成功率、终端距离、最小间隙等指标，结果显示在10%质量/推力偏差下原模型成功率从69%降至8%，DCC训练恢复至约65%；风重建在无风输入时亦能恢复控制收益，但在共享校准误差时优先级会逆转。

**⚠️ 局限性**

局限性包括仅在仿真环境下验证、对风模型假设过于理想、校准误差和风噪声的分布为人为设定、缺乏真实硬件实验以及对不同规划/感知框架的泛化性未知。

---

## 99. Toss If Perishable: An Ethnographic Study on Building Scenario-Based Training for Non-Perishable Skills

**arXiv ID:** 2609.21147 | [PDF](https://arxiv.org/pdf/2609.21147v1)

**作者:** Francis Hahn `[一作]` (University of South Florida), Xinming Ou `[通讯]` (University of South Florida)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并评估了一个名为 Mock‑SOC 的基于情景的训练方法，帮助大学生在工具无关的环境中培养 SOC 分析师的非易失性思维技能。

**💡 创新点**

创新之处在于将“非易失性知识”概念融入情景训练，剔除工具依赖，采用民族志研究结合基于理论的编码，揭示学习过程中的认知与社群因素。

**🔧 技术方法**

采用情景设计、primer 演示、数据请求系统、现场观察、访谈和基于理论的归纳编码等方法。

**📊 数据集**

使用了自行构建的虚拟网络日志数据集，包括黄金票据攻击和勒索软件攻击的事件日志、SMTP 日志、Log4J 日志等，全部预处理为无工具依赖的格式。

**📈 对比分析**

通过对 20 小时现场记录进行定性编码和定量完成率评估，发现约 30%–60% 的受试者实现全局理解，指出训练对报告与推理能力的提升，但未给出可比对性能指标。

**⚠️ 局限性**

局限包括样本量有限、单一高校背景、研究周期短、报告模板不完善、对工具知识缺失导致分析偏差，以及缺乏纵向跟踪验证效果。

---

## 100. The Stochastic Shift: A New Evaluation Paradigm for Text-to-SQL with AI Operators

**arXiv ID:** 2609.21133 | [PDF](https://arxiv.org/pdf/2609.21133v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

---

## 101. Scaling Forced Alignment to End-User Devices

**arXiv ID:** 2609.21145 | [PDF](https://arxiv.org/pdf/2609.21145v1)

**作者:** Lawry Sorenson `[一作]`, Stephen D. Richardson `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出改进的Viterbi算法实现，利用Hirschberg分治降低内存使用并加入基于随机行走模型的窗口剪枝，以实现对长音频的高效强制对齐。

**💡 创新点**

创新点在于将Hirschberg算法与Viterbi对齐相结合实现线性内存；设计可解释的基于转录准确率的窗口尺寸理论；并将剪枝与对齐算法统一实现为开源Python包。

**🔧 技术方法**

主要技术包括Hirschberg分治、随机行走模型与置信区间推导、VAD剪除静音、Beam搜索（对比方法）以及CTC等。

**📊 数据集**

使用的主要数据集为CJCLDS-GC（神父解读数据），以及Buckeye和EuroSpeech用于评估与对比。

**📈 对比分析**

与torchaudio、NeMo、Kaldi、ctc-seg等实现比较，CPU下Hirschberg–Viterbi在3小时音频仅需5 MB内存、271 s；Pruned版本可在57 s内完成；整体比现有实现快且内存低，准确率>98%且对齐匹配率近100%。

**⚠️ 局限性**

局限性包括剪枝方法未考虑数据本身变异，假设语速恒定；对齐只给出单一路径；VAD引入误差；在转录准确率低于阈值时性能下降。

---

## 102. MetaPusher: Meta Learning and Planning for Nonprehensile Manipulation of Unseen Objects with Rapid Online Adaption

**arXiv ID:** 2609.21122 | [PDF](https://arxiv.org/pdf/2609.21122v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 103. SensorWF: A FAIR Generalizable Workflow Framework for Scientific Time-Series Analysis

**arXiv ID:** 2609.21110 | [PDF](https://arxiv.org/pdf/2609.21110v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 104. REFINEPPO: Learning Continuous Control Policies by Iterative Action Refinement

**arXiv ID:** 2609.21108 | [PDF](https://arxiv.org/pdf/2609.21108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 105. Exploring Text Classification Models with Sparse Autoencoders

**arXiv ID:** 2609.21142 | [PDF](https://arxiv.org/pdf/2609.21142v1)

**作者:** Daniel Kerrigan `[一作]` (Capital One), Enrico Bertini `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了 SAEfarer，一个基于稀疏自编码器（SAE）的交互式可视化工具，用来探索文本分类模型的特征与预测/错误之间的关系。

**💡 创新点**

创新点在于：① 将 SAEs 的特征与模型的混淆矩阵结合，提供多维度的特征排名；② 通过 Jupyter Notebook 交互式小部件，让用户能即时输入文本并观察特征激活；③ 设计了多种可视化组件（混淆矩阵、激活热图、柱状图等），支持从宏观到微观的多层次分析。

**🔧 技术方法**

技术手段包括：k‑sparse 自编码器训练、激活值提取与统计、基于混淆矩阵的特征排名、Python 数据可视化库（如 matplotlib、seaborn、ipywidgets）以及 Jupyter Notebook 集成。

**📊 数据集**

使用的数据集为 AG News（包含四类新闻主题），并以 Hugging Face 上预训练的 bert‑base‑uncased 模型作为文本分类基准。

**📈 对比分析**

通过专家试点评估（5 名博士生），从可用性、功能完整性、交互体验等维度进行定性评价，用户反馈总体正面；工具在概念可解释性方面表现突出，但未进行定量性能对比。

**⚠️ 局限性**

局限性包括：① SAE 的质量对分析结果影响大，低质量特征导致解释不清；② 特征排名仅基于激活与否，忽略激活强度；③ 主要聚焦单一特征，缺乏多特征关联分析；④ 试点样本规模小，未验证在真实业务场景中的适用性。

---

## 106. Diverse and Adaptable Arm Coordination for Octopus-Crawling via Diffusion-Based Uncertainty-Aware Optimization

**arXiv ID:** 2609.21138 | [PDF](https://arxiv.org/pdf/2609.21138v1)

**作者:** Seung Hyun Kim `[一作]` (University of Illinois Urbana Champaign), Mattia Gazzola `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用扩散模型进行在线无演示的控制学习，发现并保持一组多样化的八臂软体机器人（CyberOctopus）的爬行控制策略，并通过控制编辑实现对未知限制条件的零训练适应。

**💡 创新点**

①提出DUO算法，结合目标可靠性和控制性能的联合评分，在有限评估预算内高效探索多模态控制分布；②利用圆周对称性折叠搜索空间，单一方向控制即可生成全方向行为；③在保持多样性的同时实现快速适应，无需重新训练。

**🔧 技术方法**

扩散逆向模型（DDPM）+ 目标不确定性估计（模型集成）+ 代理回归（MSE）+ 上界采样（UCB）+ 控制编辑（条件约束迭代），配合循环对称性变换。

**📊 数据集**

在模拟环境中动态构建的数据集：每轮迭代评估约 1800 条控制-目标对，起始无先验演示数据，完全从在线仿真收集。

**📈 对比分析**

与 DiffBBO、DiBO、Multi-CMA-ES 及 AutoQD 进行比较。DUO 在 100 轮 1800 次评估预算下取得最高最大目标值（≈0.35），并保持较大目标值分布（min‑max 范围最宽），同时在三种受限场景下控制编辑可恢复约 50%‑70% 失效目标，表现优于其他方法。

**⚠️ 局限性**

仅在二维平面仿真中验证；缺乏硬件实验、感知反馈与动力学不确定性；对超参数和模型规模敏感；控制编辑仅在已学习分布范围内有效，可能无法处理完全陌生的约束。

---

## 107. Signal-Centric Remote Sensing via Alternative Preprocessing and Acoustic Processing for ML-Driven Applications

**arXiv ID:** 2609.21123 | [PDF](https://arxiv.org/pdf/2609.21123v1)

**作者:** Logan Luna `[一作]` (Embry Riddle Aeronautical University), Leo Ghelarducci `[通讯]` (Embry Riddle Aeronautical University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出并实现了基于CSV格式的声纳数据预处理方法，替代传统图像化处理以提升计算效率和目标检测准确度。

**💡 创新点**

创新点在于直接处理CSV原始声纳数据，利用向量化与并行计算显著降低处理时间，并通过对比展示比CV图像处理更高的噪声抑制和目标可视化效果。

**🔧 技术方法**

使用技术包括Python、Pandas、NumPy、SciPy进行中值滤波与背景扣除，OpenCV/Scikit‑learn实现K‑Means聚类与凸包计算，以及自定义的并行数据结构。

**📊 数据集**

使用自研的Ping 360声纳数据集，包含多帧角度扫描、时间戳与强度值，格式为CSV。

**📈 对比分析**

通过对比CV图像处理与CSV方法的过滤时间、总处理时间、SNR/PSNR及目标检测聚类准确度，CSV方法在总处理时间上降低91.18%，过滤速度提升13.11%，并在对象检测上取得更清晰、更准确的结果。

**⚠️ 局限性**

主要局限在于仅针对单一声纳型号和单一环境的数据，且仅使用K‑Means进行检测，未验证在多种声纳、不同环境及更复杂机器学习模型下的适用性。

---

## 108. GeoRIS: Geofencing With Reconfigurable Intelligent Surfaces

**arXiv ID:** 2609.21077 | [PDF](https://arxiv.org/pdf/2609.21077v1)

**作者:** André Gomes `[一作]` (Rowan University), Jacek Kibilda `[通讯]` (Virginia Tech)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并验证了 GeoRIS，一种基于被动 RIS 的地理围栏控制器，通过在波束搜索和数据传输阶段动态切换 RIS 的反射/吸收状态，实现户外基站到室内用户的覆盖增强或抑制。

**💡 创新点**

其创新之处在于：①在不需要控制基站的前提下，利用波束操纵实现可切换的覆盖控制；②提出静态和自适应（基于组合多臂赌博机）两种响应生成策略；③系统性分析 RIS 尺寸、天线方向性、通道信息缺失对性能的影响。

**🔧 技术方法**

采用 3GPP 室外到室内传播模型、波束管理协议（beam sweeping/measurement/determination）、RIS 相位调节（随机或对齐）、组合多臂赌博机（CMAB）学习算法以及 Monte Carlo 仿真技术。

**📊 数据集**

使用仿真生成的随机 UE 位置与基站/ RIS 位置，并依据 3GPP 路径损耗模型产生的信道参数；未使用公开数据集。

**📈 对比分析**

通过与全反射（增强覆盖）和全吸收（无覆盖）两种基线对比，评估覆盖面积、覆盖掉线率和平均接收功率；实验显示 GeoRIS 能将室内覆盖从约 90% 的强覆盖区切换到 90% 的无服务区，或相反，且在大 RIS 尺寸或相位对齐时性能显著提升。

**⚠️ 局限性**

局限性包括：需要 RIS 的合适部署位置与足够尺寸；对墙体穿透损耗敏感；自适应算法需同步、快速切换和训练样本；在多 UE 或高方向性系统中，随机相位或无通道信息时性能下降，且实现复杂度较高。

---

## 109. Learning Scene-Aware Humanoid Locomotion through 3D Clutter from Immersive Human Demonstrations

**arXiv ID:** 2609.21107 | [PDF](https://arxiv.org/pdf/2609.21107v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 110. NetInspector: Measuring and Improving LLM Capabilities for Reliable Intent-Based Networking Policy Generation

**arXiv ID:** 2609.21103 | [PDF](https://arxiv.org/pdf/2609.21103v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 111. Information Structure of Defection Decisions in a Colonel Blotto Model of Deterrence

**arXiv ID:** 2609.21064 | [PDF](https://arxiv.org/pdf/2609.21064v1)

**作者:** Tristan Mott `[一作]` (Brigham Young University), Keith Paarporn `[通讯]` (University of Colorado)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了两场战场Colonel Blotto博弈中不同信息结构（先期、即时、中期）对威慑决策与均衡的影响；

**💡 创新点**

创新点在于提出了阈值Γ、Λ与比值N_d/N_a，阐明了信息可观测度如何决定均衡存在与否以及各方偏好的转换；

**🔧 技术方法**

主要使用了零和博弈理论、极值分析、线性规划求解以及对称性简化的数学推导；

**📊 数据集**

未使用实测数据，全部以理论模型和小规模合成实例（如2/3、3/3、100/100等）为验证；

**📈 对比分析**

通过数值实验比较不同信息结构下的均衡效用，发现ex‑post始终存在均衡且在θ<N_d/N_a时劣势，而θ≥N_d/N_a时优势；

**⚠️ 局限性**

局限性包括仅考虑两场战场、假设捕获概率为min{d/a,1}、缺乏动态学习与多阶段决策的分析，以及对真实军事情境的适用性需进一步验证。

---

## 112. Noctif3R: Feed-Forward Monocular Real-Time SLAM for Photon-Limited Scenes on Embedded Hardware

**arXiv ID:** 2609.21114 | [PDF](https://arxiv.org/pdf/2609.21114v1)

**作者:** Mihir Chauhan `[一作]` (Purdue University), Aniket Bera `[通讯]` (Purdue University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种针对暗环境、低光照、单摄像头、功耗受限且实时要求的单目SLAM流水线，并在Jetson AGX Orin上实现了嵌入式高效部署。

**💡 创新点**

创新点包括：① 在前端使用显式匹配门限（match‑fraction gate）避免“吸收状态”导致的无信息轨迹；② 采用分辨率拆分的追踪/映射路径；③ 针对Jetson Orin的端到端优化，降低图优化瓶颈；④ 对每帧位姿求解器做两处改进，实现吞吐量与误差的Pareto提升。

**🔧 技术方法**

技术手段主要是：feed‑forward pointmap 网络（MASt3R‑SLAM 系列）、匹配门限 ϕ、Sim(3) 位姿求解、全局图优化（Gauss‑Newton）、分辨率拆分追踪/映射、GPU 调度优化、两处 pose‑solve 修正（残差子采样、设备侧收敛判定）以及在 Jetson AGX Orin 上的嵌入式实现。

**📊 数据集**

使用的数据集包括：① 经过校准的 15 级暗阶梯（synthetic ladder）与三种场景；② Boston Dynamics Spot 的暗室视频阶梯（real continuous video）和其 null‑input 控制；③ 重新标注的 23 张真实暗曝光图；④ CAVERS、EVIMO2、TUM fr1 等公开数据集作为额外验证。

**📈 对比分析**

与基线（DROID‑SLAM、DPV‑SLAM、VGGT‑SLAM、π^3、CUT3R、EC3R‑SLAM、ORB‑SLAM3、DSO）相比：在暗阶梯上我们在 9 个光照最低点只产生 3‑5 条有效轨迹，误差仅为基线 24–48% 的无信息上限（相比 56–73%）；在 Jetson Orin 上吞吐量提升 1.28×、误差下降 3.6%，显存减少 47%，能耗下降 29%，尾部延迟下降 34%；在 Spot 的 null‑input 视频中，我们的配置在光照不足后立即停止，而基线仍输出全帧误差轨迹。总体而言，我们提供了更安全、能耗更低、在极端暗光下更可靠的 SLAM 解决方案。

**⚠️ 局限性**

局限性：① 低光下真实数据仅覆盖到约 -8 dB，暗阶梯主要是合成；② 匹配门限的阈值在不同协议（每帧 vs 每八帧）下表现不同，需进一步泛化；③ 目前吞吐量 3.2 轨迹/秒仍不足以满足 30 Hz 视频；④ 通过门限避免轨迹扩展的做法会导致覆盖率窄，若想实现连续跟踪需改进后端权重机制；⑤ 评估中部分数据（Spot 阶梯）缺乏精确的 SNR 标注与 ground‑truth，限制了客观对比。

---

## 113. Can Agents Design Better Chips with a Higher Level Abstraction?

**arXiv ID:** 2609.21157 | [PDF](https://arxiv.org/pdf/2609.21157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 114. SAGE: Safety-Aligned Gradient Enforcement for Human--Robot Collaboration

**arXiv ID:** 2609.21130 | [PDF](https://arxiv.org/pdf/2609.21130v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 115. Layerwise Decoupling for Stable Structured Sparsification of Fully Connected Layers

**arXiv ID:** 2609.21126 | [PDF](https://arxiv.org/pdf/2609.21126v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 116. Talk to Me, Jarvis: An Open-Source Edge-Deployable Voice Assistant Framework for Autonomous Racecars

**arXiv ID:** 2609.21109 | [PDF](https://arxiv.org/pdf/2609.21109v1)

**作者:** Daniel Henel `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了离线语音助手Jarvis，用于无人驾驶赛车的高层指令语音控制，集成唤醒词检测、离线STT、轻量化LLM分类与TTS反馈，并通过ROS2与车载系统实现安全验证。

**💡 创新点**

创新点在于提供网络无关、端到端低延迟的指令识别方案；通过领域特定微调的Mistral‑7B并采用QLoRA + 4‑bit量化，实现在赛车环境下97.63%准确率与1.39 s推理时延，优于云端模型；并将完整框架开源。

**🔧 技术方法**

使用技术包括：openWakeWord唤醒词框架；Whisper base.en 轻量化语音识别；Coqui TTS 神经语音合成；Unsloth+QLoRA 量化微调 Mistral‑7B；4‑bit 量化；ROS2通信；Python/UNSLOTH 等。

**📊 数据集**

数据集为17类指令共1,645条样本，原始85条通过GPT‑4进行同义词替换、改写、词序变换等增强，加入了 OOS（超域）类。

**📈 对比分析**

通过在 RTX 4090 上对多种在线与本地模型进行基准，在线模型准确率 76–85% 时延 3–21 s；本地模型准确率 27–35% 时延 0.4–1.1 s；经过两阶段 QLoRA 微调后 Mistral‑7B 达到 97.63% 准确率与 1.39 s 时延，显著优于云模型。

**⚠️ 局限性**

局限性包括：数据集主要人工与 GPT‑4 增强，缺少真实现场语音；系统仅支持预定义指令，无法处理多轮对话或非结构化输入；以及仅在单机 GPU 环境验证，缺乏在更严格硬件上的实测。

---

## 117. MA-LIPP: Cooperative Multi-Agent Load-Aware Informative Path Planning for Heterogeneous Robot Teams

**arXiv ID:** 2609.21167 | [PDF](https://arxiv.org/pdf/2609.21167v1)

**作者:** Hojune Kim `[一作]` (University of Southern California), Gaurav S. Sukhatme `[通讯]` (University of Southern California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了异构多机器人团队在负载感知物理采样任务中的信息路径规划框架MA‑LIPP，并引入异步死点交接机制实现采样与运输解耦。

**💡 创新点**

创新点包括将采样、运输、分配和时间规划统一为单一混合整数二次规划模型，利用异步死点交接实现无同步交接，并提供可扩展的大邻域搜索启发式以逼近最优。

**🔧 技术方法**

使用技术包括：高斯过程后验方差信息目标、负载依赖的能量模型、单一流连通性约束、混合整数二次规划（MIQP）以及配对大邻域搜索（LNS）启发式。

**📊 数据集**

实验使用随机生成的有向图场景（节点10–50，机器人数2–12）与RBF核建模的合成场地，所有数据均为合成数据集。

**📈 对比分析**

与无交接的顺序规划基线比较，MA‑LIPP在信息增益上提升13–33%，LNS在95.5%的已证最优实例中匹配最优，且相较于精确MIQP显著缩短求解时间并在更大规模问题上保持良好性能。

**⚠️ 局限性**

局限性包括对混合整数规划求解器的高依赖、缺乏对真实地形与动态障碍的建模、死点交接假设理想且未考虑交接安全与可靠性，以及在在线实时规划场景下的可扩展性仍待进一步验证。

---

## 118. HMB-GAN: Hybrid Multi-Bézier GAN for Vector Shape Synthesis

**arXiv ID:** 2609.21158 | [PDF](https://arxiv.org/pdf/2609.21158v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 119. Aligning with Lived Experience: Heterogeneous Benefits of Fine Tuning in Mental Health Support Generation

**arXiv ID:** 2609.21075 | [PDF](https://arxiv.org/pdf/2609.21075v1)

**作者:** Mohit Chandra `[一作]` (Georgia Institute of Technology), Munmun De Choudhury `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 COPES 数据集与多轴评估框架，用于评估 LLM 在 Reddit 心理健康社区中生成的同行支持响应

**💡 创新点**

首个以社区为中心的同行支持数据集及其评估指标，并通过 SFT 与 DPO 对 LLM 进行后训练验证

**🔧 技术方法**

使用大型语言模型（Qwen3‑4B、Gemma‑4、MediPhi‑Instruct）进行零-shot、SFT、SFT+DPO 训练，并采用嵌入相似度、情感与语调匹配、主题对齐等评估技术

**📊 数据集**

基于 2011‑2021 年 Reddit 心理健康子版块帖子与评论，经过自动与人工筛选后得到 COPES 数据集，共 4455 条寻求支持的帖子

**📈 对比分析**

与零-shot baseline 对比，SFT 后模型在策略对齐提升>50%（Gemma 50%，Qwen 56%），情感语调亦有提升，但表现因模型、子版块、支持策略类别而异，整体提升有限且不均匀

**⚠️ 局限性**

限制包括：数据只包含已回答且高质量的帖子，缺乏未回答与后 2021 的社区；使用 LLM 进行标签和负样本生成；评估方法基于嵌入相似度可能漏掉叙事式支持；仅评估小型开源模型与两种后训练方法

---

## 120. M2G-LLM: Enhancing Clinical Prediction via Multimodal Graph Reasoning and LLM Context Injection

**arXiv ID:** 2609.21164 | [PDF](https://arxiv.org/pdf/2609.21164v1)

**作者:** Inyoung Choi `[一作]` (University of Pennsylvania), Qi Long `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种名为M^2G-LLM的框架，将大型语言模型与图神经网络相结合，实现在多模态（文本、实验室、诊断编码、影像）EHR数据上的临床预测；

**💡 创新点**

核心创新是通过图卷积网络捕获患者访问间的时间与相似性关系，并使用对比学习将多模态嵌入对齐至语言模型的隐藏空间，随后将构建的跨患者上下文向量注入LLM的残差流，既保持LLM预训练能力，又实现跨模态、跨时间的推理；

**🔧 技术方法**

使用Graph Neural Networks（GCN）、对比学习（InfoNCE）、残差流注入技术（residual‑stream conditioning）以及大模型Llama‑3‑8B作为主干；

**📊 数据集**

在MIMIC‑IV（临床文本、实验室、诊断编码等）和MIMIC‑CXR（胸部X光影像）数据集上进行评估；

**📈 对比分析**

与多种LLM基线（HeLM、LLMM、GPT‑4等）及传统多模态模型（HAIM、M3Care、MUSE、mmFormer）对比，M^2G‑LLM在一年死亡率预测上达到78.98%准确率、64.55% F1，30天再入院预测上实现85.75%准确率、52.35% F1，均超过所有基线并保持较好的校准和鲁棒性；

**⚠️ 局限性**

主要限制包括：① 计算资源需求高，模型训练与推理对GPU内存有显著要求；② 目前仅在公开回顾性数据集上验证，缺乏外部或前瞻性验证；③ 解释性评估仍为定性，缺少系统化的临床对齐指标；④ 模型对缺失模态的处理依赖图结构，可能对极端缺失情况敏感。

---

## 121. AI-Driven Scientific Computing Workflows: A Systems Review of Orchestration, Execution, Reproducibility and Provenance

**arXiv ID:** 2609.21162 | [PDF](https://arxiv.org/pdf/2609.21162v1)

**作者:** Jamie J. Alnasir `[一作]` `[通讯]` (Royal Holloway University of London), Jamie J. Alnasir (Royal Holloway University of London)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并系统化了AI驱动的科学工作流，提出了从传统工作流到AI深度参与的连续体，并围绕控制、执行、状态、可重复性与治理等五大系统关注点进行分析。

**💡 创新点**

创新点在于将 AI–HPC、机器学习生命周期、自动化研究工作流等相关研究整合为一个统一的系统视角，明确 AI 在工作流中的参与深度和对应的系统需求。

**🔧 技术方法**

采用结构化叙述式系统综述方法，对现有文献进行迭代检索与引用链追踪，并通过案例对比阐释关键模式与需求。

**📊 数据集**

无实验数据集，本文为综述性研究，主要依据公开论文、系统文档和案例研究。

**📈 对比分析**

未进行实验或性能比较；作者提出了基于科学进展、执行成本、数据移动、资源利用、弹性与决策可追溯性等维度的工作流级评估框架，但未给出量化结果。

**⚠️ 局限性**

局限性包括：仅从系统层面讨论，缺乏对算法性能的量化比较；文献选择主观性可能导致覆盖不完整；未提供基准测试或可复现的实验验证。

---

## 122. Decoding the Dashboard: Data Comics to Support Students' Understanding of Learning Analytics Visualisations

**arXiv ID:** 2609.21141 | [PDF](https://arxiv.org/pdf/2609.21141v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 123. CoLearn: An Agentic Tutor that Learns its Learner in a Human--AI Co-Learning Loop

**arXiv ID:** 2609.21154 | [PDF](https://arxiv.org/pdf/2609.21154v1)

**作者:** Kailai He `[一作]` (King's College London), Jiazheng Li `[通讯]` (King's College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一个交互式、具有持久学习者状态记忆的LLM辅助智能辅导系统，能够根据学习者的弱点实时生成个性化题目并展示适应证据。

**💡 创新点**

创新点在于将LLM作为连续观察函数的软证据BKT学习者状态追踪、可视化的个人化证据面板、内置盲A/B对比评估，以及可通过提示文本动态调整的生成策略。

**🔧 技术方法**

采用了大型语言模型（Claude Sonnet 4.5、OpenAI兼容接口）进行答案判分和反馈，BKT算法进行状态更新，FastAPI+MongoDB后端，React+Vite前端，并通过Docker容器化部署。

**📊 数据集**

评估使用了生物学和化学课程的题库，人工合成的6个学习者角色进行模拟测试，以及18名具备相关课程背景的评审者进行盲A/B和满意度调查。

**📈 对比分析**

通过盲A/B实验显示个性化题目被偏好率约68–69%，在模拟中自适应系统的MAE为0.12，弱项命中率为0.72，均优于随机题目或冻结记忆控制。

**⚠️ 局限性**

局限性包括未验证学习成效、LLM观察函数对低分学习者的准确性不足、缺乏策略自适应与远期迁移评估、以及缺乏题目质量与错题诱导的独立验证。

---

## 124. EnSol: an environment-aware graph neural network for molecular solubility prediction

**arXiv ID:** 2609.21151 | [PDF](https://arxiv.org/pdf/2609.21151v1)

**作者:** Thao Nguyen `[一作]` (University of Illinois Urbana-Champaign), Heng Ji `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了 EnSol，一种环境感知概率图神经网络，用于预测分子溶解度，能够同时考虑溶质、溶剂和温度三种因素。

**💡 创新点**

创新点包括：①将溶质与溶剂分别编码为分子图，并通过交叉注意力捕获其特定相互作用；②通过 FiLM 对温度进行特征级调制，将温度视为连续环境因素；③使用混合密度网络输出完整溶解度分布，提供预测不确定性。

**🔧 技术方法**

技术：AttentiveFP 图神经网络、4 头交叉注意力、FiLM 特征调制、RBF 温度展开、混合密度网络（MDN）以及负对数似然训练。

**📊 数据集**

训练集：BigSolDB；评估集：SolProp、Leeds 两个独立基准；迁移学习使用 AqSolDB 与 ESOL；实验验证 100 条溶质-溶剂对（10 种溶质 × 10 种溶剂）。

**📈 对比分析**

与 FASTSOLV、Vermeire 等基线模型比较；在 SolProp 上 Spearman 0.876（比 0.509/0.569 高），RMSE 0.824（比 1.303/1.700 低）；在 Leeds 上 Spearman 0.602（比 0.593/0.245 高），RMSE 0.944（比 0.922/2.026 低）；实验集 Spearman 0.715，RMSE 0.699，均明显优于基线。

**⚠️ 局限性**

局限性：对结构不常见的溶质、极端温度和溶剂混合物预测不稳；缺乏机制解释；高度依赖高质量溶解度数据；迁移学习在样本量增加时收益减弱。

---

## 125. Cloud-Side Transactional Orchestration Framework for Resource-Constrained Embedded Systems

**arXiv ID:** 2609.21143 | [PDF](https://arxiv.org/pdf/2609.21143v1)

**作者:** Pravin Nagare `[一作]` (Binghamton University), Willison Lopes `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种针对资源受限嵌入式系统的云端事务性后台前端（T-BFF）调度框架，降低设备端内存占用和交易延迟。

**💡 创新点**

核心创新在于将事务状态机迁移至云端，并提出“双握手”恢复协议与可重入的幂等键，以实现跨重启事务一致性。

**🔧 技术方法**

使用了云端 Node.js / gRPC、MessagePack 二进制序列化、Redis 分布式缓存、以及 ARM Cortex-A53 设备端的 Flash 持久化存储。

**📊 数据集**

实验基准使用了在 ARM Cortex‑A53（1 GB RAM）上执行的 50 次独立购买交易，比较了传统厚客户端 SDK 与 T‑BFF 的性能。

**📈 对比分析**

对比方法采用平均值、标准差与 95% 置信区间，结果显示 T‑BFF 在堆峰值使用量上下降 35.9%，CPU 占用下降 62.6%，事务总延迟降低 39.6%。

**⚠️ 局限性**

主要局限包括对云服务可用性的依赖、在更低功耗 MCU 上的可移植性不足，以及实验仅覆盖单一硬件平台和模拟网络条件。

---

## 126. Detecting Hallucination in LLMs: Tracing the Topological Signatures of Impaired Context Sharing

**arXiv ID:** 2609.21096 | [PDF](https://arxiv.org/pdf/2609.21096v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 127. A Multi-Engine Dataflow for MoE Decoding on Scratchpad-Based Tensor Accelerators

**arXiv ID:** 2609.21137 | [PDF](https://arxiv.org/pdf/2609.21137v1)

**作者:** Bin Ma `[一作]` (University of California), Dong Li `[通讯]` (University of California)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种混合低秩+向量量化的专家权重表示，并配合多引擎解码数据流，提升在基于scratchpad的张量加速器上 MoE 解码速度。

**💡 创新点**

创新点在于将低秩共享基底与向量量化结合，实现权重传输量大幅下降，同时通过解耦路由、传输与计算的依赖链，充分利用多种计算引擎实现 DMA 与计算重叠。

**🔧 技术方法**

使用低秩分解、向量量化、训练无梯度初始化、知识蒸馏以及多引擎调度等技术。

**📊 数据集**

主要使用 WikiText‑103 进行 perplexity 测试，并在 AWS Trainium3 上部署 Qwen3、DeepSeek‑V2‑Lite、Nemotron‑3‑Nano、OLMoE‑1B‑7B、Gemma‑4‑26B‑A4B 等五个 MoE 模型进行评估。

**📈 对比分析**

与 PyTorch/XLA 与 AWS Dense megakernel 两个基线比较，最终在所有模型上实现 1.15–1.31 倍的单 token 解码加速，并保持或略低于 BF16 教师的 perplexity。

**⚠️ 局限性**

局限在于对 STA 体系结构高度依赖，VQ 块大小与低秩秩的取舍会影响性能；对更大模型或不同加速器的迁移性尚待验证。

---

## 128. MarsFM: Shading-Regularized Flow Matching for Martian Relief Estimation

**arXiv ID:** 2609.21095 | [PDF](https://arxiv.org/pdf/2609.21095v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 129. From Task Success to Productive Success: Evaluating Human-AI Collaboration by Quality and Cost

**arXiv ID:** 2609.21117 | [PDF](https://arxiv.org/pdf/2609.21117v1)

**作者:** Saki Imai `[一作]` (Northeastern University), Malihe Alikhani `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入了基于产出质量与交互成本的协同生产力评估框架，并在四个任务的数据集上验证。

**💡 创新点**

将交互成本与质量相结合，区分“高质量低成本”与“高质量高成本”的合作结果，并发现任务类型决定交互成本与质量的关系。

**🔧 技术方法**

采用LLM‑as‑judge量化质量、token加权计算交互成本、对话特征的自编码标签（grounding、positive friction）及统计相关/回归分析。

**📊 数据集**

使用两大数据集——CoGym（旅行规划、表格分析、文献综述）以及自收集的可视化任务（42次人‑LLM协作）。

**📈 对比分析**

通过标准化质量与成本得分计算 P_z，对比任务内同质量的成本差异（1–2个数量级），发现主观评估与 P_z 并不一致；对话特征显示代理提问能提升 P_z。

**⚠️ 局限性**

成本度量仅为 token 加权近似，未测量认知负荷；对话分析为相关性而非因果；仅覆盖四种任务，未必能推广到更广泛合作场景。

---

## 130. Decoupling Internal Representational Changes and Causal Importance in Fine-Tuned Large Language Models

**arXiv ID:** 2609.21113 | [PDF](https://arxiv.org/pdf/2609.21113v1)

**作者:** Lingfang Li `[一作]` (University of Liverpool), Danushka Bollegala `[通讯]` (University of Liverpool)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 fine‑tuning 对大语言模型内部机制的影响，分析了注意力模式、表示层变化与 EAP 识别的因果关键组件的关系，并考察了跨任务性能转移。

**💡 创新点**

发现内部变化最大的层不一定是任务性能关键层，且高重叠的 EAP 组件会导致跨任务性能下降，而非提升。

**🔧 技术方法**

使用 EAP（Edge Attribution Patching）计算因果重要性，利用 KL 散度衡量注意力变化，logit lens 探测表示层信息，并通过激活补丁验证结果。

**📊 数据集**

在 GPT-2 Small、LLaMA-3.2-1B、Qwen2-0.5B、LLaMA-2-7B 四个模型上，分别针对六个公开数据集（Yelp、SST‑2、SQuAD、CoQA、KDE4、Tatoeba）进行实验。

**📈 对比分析**

通过层级熵、KL 散度、EAP 分数以及交叉任务相关性进行对比，结果显示 EAP 重要性高度局部化，且共享组件与性能衰减正相关，整体表现随模型规模而变化。

**⚠️ 局限性**

仅覆盖三类 NLP 任务和四个中小型模型，未涉及推理、跨模态等更广泛应用，也缺乏对更大模型的验证。

---

## 131. Dynamics-Induced Commitment in Learning-Based Robotic Penalty Kicks

**arXiv ID:** 2609.21100 | [PDF](https://arxiv.org/pdf/2609.21100v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 132. Dynamic Modeling and LQR Control of a Single Coaxial Drone with 2DOF Thrust Vectoring Mechanism

**arXiv ID:** 2609.21099 | [PDF](https://arxiv.org/pdf/2609.21099v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 133. Design of Adaptive PID Controller Based On Asynchronous Advantage Actor Critic Learning Method for QuadCopter Control

**arXiv ID:** 2609.21082 | [PDF](https://arxiv.org/pdf/2609.21082v1)

**作者:** Ali Jokar `[一作]` (Sharif University of Technology), Aria Alasty `[通讯]` (Sharif University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种将A3C强化学习与PID控制器相结合的自适应控制框架，用于四旋翼无人机的姿态和轨迹跟踪。

**💡 创新点**

创新点在于通过多线程并行的A3C算法实现PID增益的实时动态调节，并在自适应PID网络与系统识别网络之间建立协同机制，从而提升控制精度和收敛速度。

**🔧 技术方法**

采用A3C（异步优势演员-评论家）强化学习框架、深度神经网络（用于自适应PID调节与系统识别）以及传统PID控制算法。

**📊 数据集**

未使用公开数据集，全部在仿真环境中生成的四旋翼动力学数据进行训练与评估。

**📈 对比分析**

通过与单线程A2C算法对比，A3C-PID在高度、位置及姿态跟踪上表现出更快的损失收敛速度和更高的跟踪精度，但伴随更大的初始振荡，整体性能在精度与平滑度之间取得权衡。

**⚠️ 局限性**

局限性包括：高度振荡明显、计算量较大、对训练数据的依赖、缺乏真实环境验证以及能耗与障碍规避能力待进一步提升。

---

## 134. Origin Is All You Need: Provenance-Aware Transformers for Structural Trust-Boundary Separation

**arXiv ID:** 2609.21088 | [PDF](https://arxiv.org/pdf/2609.21088v1)

**作者:** Yuxuan Zhang `[一作]` (Texas A&M University), Guofei Gu `[通讯]` (Texas A&M University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Provenance‑Aware Transformers，通过在 Transformer 前向传播中加入源信息（origin），实现对不同来源文本的信任边界分离，从而抵御间接提示注入（IPI）。

**💡 创新点**

创新点：①将来源标签作为可学习的 origin embedding 直接注入 token 表示；②在注意力层加入可学习的 origin attention bias，以结构化方式抑制不可信源的影响；③通过两阶段微调（origin 语义学习 + 注入对齐学习）让模型真正理解来源意义，形成系统化的信任分层。

**🔧 技术方法**

技术：Transformer 架构改造（origin embedding、origin scale、origin attention bias）、两阶段监督微调、可学习的 origin embedding table 与 bias matrix、RMSNorm 与尺度门控。

**📊 数据集**

数据集：Stage‑1 训练使用 UltraChat‑200K、MMLU、GSM8K、CodeAlpaca‑20K；Stage‑2 训练使用自构造的 SEP‑style 注入数据（约60k样本，含 Ring‑3 注入与同样背景的干净样本）；评测集包括 SEP、PI‑Attack、DataSentinel、Math‑Tutor。

**📈 对比分析**

对比方法：与十类现有防御（Prompt‑based、Detection‑based、Fine‑tuning）在四个公开模型（SmolLM2‑360M、LLaMA‑3‑8B、Qwen2.5‑7B、Mistral‑7B）上评测。结果显示 Provenance‑Aware 在 IID 与 OOD 的攻击成功率（ASR）均降至 0% 或接近 0%，优于所有对手；在 AlpacaEval 的 LC win‑rate 与基线基本相当，说明在保持实用性的同时实现了强鲁棒性。

**⚠️ 局限性**

局限性：①只能按 ring 级别统一抑制，无法区分同一 ring 内的指令与数据；②需要在应用层提前标注来源，若信任分层不细粒度会导致过度抑制；③目前为后置微调，未集成到预训练阶段；④在极端白盒自适应攻击下仍需与内容级检测结合以进一步提高安全性。

---

## 135. DLB: Distributed Load Balancing at Scale for Generative AI Inference

**arXiv ID:** 2609.21079 | [PDF](https://arxiv.org/pdf/2609.21079v1)

**作者:** Santiago R. Balseiro `[一作]` (Google Research), Amin Vahdat `[通讯]` (Google Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在全球分布的多机房环境中实现了分布式的全局负载均衡系统DLB，用以最小化大型异构机器学习推理工作负载的端到端延迟。

**💡 创新点**

创新点包括：①将负载均衡分为根层（跨机房）和叶层（同机房）两层；②采用实时探测与分布式状态共享，实现对机器学习加速器容量的即时可视化；③基于学习的延迟模型和梯度下降的流量分配算法，提供理论上全局收敛性与性能下界；④在实践中实现了高吞吐（百万 RPS）且系统开销极低（≈0.04%）。

**🔧 技术方法**

技术手段包括：分布式P2P探测、参数化延迟模型（如softplus）、根层与叶层的负载估计器、离散/流式路由算法、梯度成本函数、错误规避机制、会话亲和性、以及Lyapunov直接法的稳定性分析。

**📊 数据集**

使用的是Google内部生产流量与多机房数据，涵盖数千个端点、百万级请求/秒、数百万个MLA芯片。并通过仿真模拟不同硬件异质性、流量峰值与容量缺口。

**📈 对比分析**

与传统的MNLB（多机房最小网络延迟负载均衡）和加权随机路由进行对比。实验与迁移分析显示DLB在迁移后总体延迟平均下降约13%，中位数下降约17%，95%分位数下降约13%；在仿真中，DLB在不同负载与异质性场景下的均值与尾部延迟均低于基线，且错误率更低。

**⚠️ 局限性**

局限性包括：对延迟模型的依赖导致需频繁探测与模型更新；在极端突发流量下可能出现探索‑利用平衡问题；理论分析假设处理率函数单调且可导，实际批处理或缓存可能违反；并且对端点特定目标（如首 token 延迟）未实现统一的多目标优化。

---

## 136. Demonstration Synthesis from a Single Scan via Gaussian Splatting for Visuomotor Policy Learning

**arXiv ID:** 2609.21112 | [PDF](https://arxiv.org/pdf/2609.21112v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 137. Geometry of Values: Task Vector Composition for Ethical Preference Alignment in Language Models

**arXiv ID:** 2609.21094 | [PDF](https://arxiv.org/pdf/2609.21094v1)

**作者:** Utkarsh Agarwal `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Monojit Choudhury `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个跨语言的12,000条双选道德困境基准，研究大型语言模型在不同价值取向下的行为，并提出了一种基于任务向量的无训练偏好切换方法。

**💡 创新点**

创新点在于①将三大价值（诚实、正义、自主）组合成可量化的对立困境并在五种语言中并行化；②证明了通过任务向量正交化可以在不重新训练的情况下切换模型的价值偏好；③揭示了价值向量在权重空间中的正交性，说明单一偏好可逆但不具可传递性。

**🔧 技术方法**

使用 Llama‑3.2 1B/3B 模型，结合 LoRA 轻量化微调、Direct Preference Optimization（DPO）和基于任务向量的线性算子（包括指令向量与偏好向量的分离与叠加）。

**📊 数据集**

数据集为 12,000 条人工生成并机器校验的双选伦理困境，覆盖 Honesty‑Justice、Justice‑Autonomy、Autonomy‑Honesty 三对价值，并翻译成 Hindi、Arabic、Spanish、Chinese；另外包含 60 条人工撰写的金标测试集。

**📈 对比分析**

对比方法包括零射击基线、提示式对齐、LoRA SFT、DPO 训练以及任务向量转移。LoRA/SFT 在所有任务和语言上均达 98% 以上准确率，在金标集上保持 90%+；任务向量转移在 3B 模型上保留 ≥97% 的微调性能，1B 模型保持 ≥93%；零射击基线表现出约 70% 的正义偏好和显著的首选项偏差。

**⚠️ 局限性**

局限性包括：①仅评估三种价值的强制二选困境，未覆盖更广泛的伦理推理或解释；②数据主要来源于 LLM 生成，可能携带生成器特异性噪声；③金标集规模有限，验证范围受限；④实验仅针对 Llama‑3.2 1B/3B，尚未验证更大模型；⑤任务向量方法无法实现价值的可传递组合。

---

## 138. Loopjacking: Hijacking Human-in-the-Loop Approval

**arXiv ID:** 2609.21081 | [PDF](https://arxiv.org/pdf/2609.21081v1)

**作者:** Adithyan Arun Kumar `[一作]` `[通讯]`, Adithyan Arun Kumar

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过在已发布的多款代理产品（Agno AgentOS、LangGraph Agent Server、OpenClaw 和 OpenAI Agents SDK）上复现和验证 Loopjacking 攻击，阐明了在人工审批与实际执行之间的绑定失效，并提出了完整的规范化审批记录与使用时绑定的防御措施。

**💡 创新点**

创新点在于首次给出 Loopjacking 的可操作性定义与两种攻击变体（表示层失配与后置状态替换）的区分，结合实测案例证明了该概念在真实产品中的存在，并给出针对性修复思路。

**🔧 技术方法**

采用自动化测试脚本、模拟用户审批流程、对比原始请求与最终执行的完整操作描述，以及使用 Python/Go 等语言实现的回放与校验机制，并通过可视化日志与哈希校验验证攻击与防御的效果。

**📊 数据集**

使用了各产品的公开版本（Agno 2.5.6–3.0.9、LangGraph 0.7.5–0.14.0、OpenClaw 2026.2.23/24、OpenAI Agents SDK 0.22.0/0.22.2），并在本地隔离环境中构造了合成请求、模拟的会话状态和伪造的审批记录，最终以虚拟账本或临时文件记录执行结果。

**📈 对比分析**

对比方法是通过“攻击实验 + 对照实验”框架，在每个版本中分别验证：①直接未授权调用是否被拒绝；②保持原审批的正常执行是否成功；③攻击路径是否导致预期的不合法操作；④在安全补丁或安全配置下是否被阻断。实验结果显示所有受影响版本都成功执行攻击，而在补丁或安全策略下均被拦截；没有引入性能下降的度量。

**⚠️ 局限性**

局限性包括：样本仅为有意挑选的几条产品路径，无法代表整个生态；检测仅覆盖特定配置与版本；未对更广泛的环境（如生产数据库、不同操作系统）进行验证；缺乏对用户真实审批行为与误导率的评估；并未给出全局漏洞率或安全评分。

---

## 139. An $m^{2.943}$ Bohnenblust--Hille Bound on the Boolean Cube

**arXiv ID:** 2609.21144 | [PDF](https://arxiv.org/pdf/2609.21144v1)

**作者:** Joseph Slote `[一作]`, Alexander Volberg `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在布尔立方体上证明了 Bohnenblust–Hille 不等式的新上界，给出了对任意复杂值函数 f:{-1,1}^n→ℂ、度≤m 时的系数 ℓ_{q_m} 范数上界 (∑|f(A)|^{q_m})^{1/q_m} ≤ C_ε m^{β_0+ε}‖f‖_∞，其中 β_0=3/2+1/ln2≈2.9427。

**💡 创新点**

创新点在于：① 设计了新的权重 μ_r=⌈3r/2⌉，使得在分解 r=d+e 时能够消除“缺陷”并获得更小的幂；② 将随机二分染色与分段（低、中、高）窗口结合，分别在 r<L_M 与 r≥L_M 时采用不同的分割策略；③ 引入带权平方函数 W_M(F) 进行归一化，使得整个多层系数能统一控制；④ 通过细致的概率与组合估计实现了自举闭包，得到统一的常数 C_B<1，进而实现整个权重闭包。

**🔧 技术方法**

主要技术包括：布朗尼-贝克宁（Bonami–Beckner）超卷积、解析不等式的加权插值、随机二分染色的中心窗口与平衡窗口的熵收缩、分层权重的乘法性质、以及组合概率估计（如二项式分布、Chebyshev 与 Chernoff ），再配合梯度的分解与权重的调度实现了全局收敛。

**📊 数据集**

该工作完全是理论性数学证明，未使用任何实验数据集。

**📈 对比分析**

相较于早期在同一问题上的 m^9 上界，本文将指数降低到约 m^{2.943}；此外还证明了在每个同阶层内系数近似相等时，Aaronson–Ambainis 影响度猜想得到完全满足，从而在特殊结构下提供了更强的影响度下界。

**⚠️ 局限性**

局限性：① 仍未达到理论最优指数 3/2+1/(2m)；② 证明过程依赖于对偶分层的奇偶性与窗口大小的精细调节，可能不易直接推广到更一般的多变量多项式或非布尔域；③ 需要 B>1/ln2 的限制，说明当前方法在极限情形下仍有余量；④ 证明中使用的权重与窗口设计相对复杂，进一步优化可能导致更低的常数或更强的结论。

---

## 140. Toward individual-level calibration in affect recognition with perceptual adjustment queries

**arXiv ID:** 2609.21073 | [PDF](https://arxiv.org/pdf/2609.21073v1)

**作者:** Xuanzhou Chen `[一作]` (Georgia Institute of Technology), Ashwin Pananjady `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了基于感知调节查询（PAQ）的个体化感知校准框架，用于对面部情感识别任务中的感知难度进行归一化；

**💡 创新点**

创新点在于利用轻量级滑块查询直接估计每个受试者的最小可辨别差（JND），并在下游2AFC任务中用该JND重构刺激对，从而实现个体化的感知尺度校准；

**🔧 技术方法**

核心技术包括：①基于视觉‑语言模型（VLM）的连续表情生成管线，用于构建平滑、身份保持的情感轨迹；②PAQ滑块交互，用于捕获JND；③基于Weibull分布的群体校准方法；④时间与主观难度指标的统计分析；

**📊 数据集**

使用Chicago Face Database中八个种族-性别组合的面孔，并通过VLM生成从中性到高强度情感的100帧连续轨迹；

**📈 对比分析**

与无校准基线以及群体级Weibull校准相比，PAQ校准在主观难度平等判定和响应时间方差两方面均显著提升：主观难度平等率提升约70%，响应时间均值及方差均降低30%–80%；

**⚠️ 局限性**

局限性包括：①实验阶段顺序导致学习与熟悉效应；②JND估计假设感知函数单调且可用Weibull拟合，实际可能受疲劳、情绪波动影响；③仅验证于sad‑happy连续轴，缺乏对其他情感维度或临床人群的推广；④样本虽种族均衡但不一定代表全球人群。

---

## 141. Geometry-Aware Diffusion Guidance via Curvature-Adaptive Tubular Correction

**arXiv ID:** 2609.21251 | [PDF](https://arxiv.org/pdf/2609.21251v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. Hallucination-R1: Robustness-Oriented Paraphrase Generation for Factual Consistency

**arXiv ID:** 2609.21227 | [PDF](https://arxiv.org/pdf/2609.21227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 143. Emergent Intelligence: Resonant Oscillators Produce Proactive Adaptive Behavior

**arXiv ID:** 2609.21161 | [PDF](https://arxiv.org/pdf/2609.21161v1)

**作者:** Alex Fedosov `[一作]` (FoundAItion Inc.), Sander Stepanov `[通讯]` (FoundAItion Inc.)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在不使用学习或外部控制的情况下，构造了最小化的脉冲振荡器网络，演示了自主的探索-利用切换和螺旋搜索行为；

**💡 创新点**

通过在三种或以上振荡器之间实现时间错位与相位反相的结构性约束，首次证明了探索行为可以完全源自网络结构而非学习；

**🔧 技术方法**

利用离散事件驱动的脉冲神经网络（inverter 单元），在 Python/Pygame 环境中实现了 63 种网络配置，进行 84,000 次独立仿真；

**📊 数据集**

在三种仿真环境（空白平面、密集食物带、圆形食物盘）中收集轨迹数据；

**📈 对比分析**

与五种标准搜索策略（随机游走、相关随机游走、Levy 跳跃、预设螺旋、区域限制搜索）进行对比，3‑振荡器 NNN 复合器在食物获取量上比最佳基线高 2.1–7.9 倍，螺旋质量显著优于其他方法；

**⚠️ 局限性**

仅限二维理想化平面运动、二值输入、无学习、无多体交互，无法直接映射到真实机器人或多尺度生物系统，且对环境噪声和参数扰动的鲁棒性仅在仿真层面得到验证。

---

## 144. Fewer Steps, Better Actions: Rethinking Flow-Matching Inference for VLA Policies

**arXiv ID:** 2609.21216 | [PDF](https://arxiv.org/pdf/2609.21216v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 145. Hand-Aware Transition Modeling for Bimanual Procedural Anomaly Detection

**arXiv ID:** 2609.21207 | [PDF](https://arxiv.org/pdf/2609.21207v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 146. Embedding Drift in Code Vulnerability Models Under Intended Behaviour-Preserving Transformations

**arXiv ID:** 2609.21203 | [PDF](https://arxiv.org/pdf/2609.21203v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 147. Reliability-Centered Evaluation of Sparse Longitudinal CT Lesion-Size Forecasting with Conformal Interval Calibration and Gompertz-Inspired Regularization

**arXiv ID:** 2609.21197 | [PDF](https://arxiv.org/pdf/2609.21197v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 148. TinyCeNN-LM: Quality-Gated Conversion of Pretrained Attention with CeNN-Inspired Cellular-Recurrent Layers

**arXiv ID:** 2609.21139 | [PDF](https://arxiv.org/pdf/2609.21139v1)

**作者:** Kabeh Mohsenzadegan `[一作]` (Institute for Smart System Technologies), Kyandoghere Kyamakya `[通讯]` (University of Klagenfurt)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套质量门控的 CeNN 风格结构转换框架 TinyCeNN‑LM，用于将预训练 Transformer 的注意力层逐层替换为基于局部窗口+递归记忆的 CeNN 灵感层，并在 SmolLM2 与 Qwen3.5 上逐层尝试。

**💡 创新点**

将结构转换拆分为候选层与质量门控协议，使用多指标（NMSE、余弦相似度、增量和累计 NLL）实现逐层提交/回滚，展示跨模型可迁移的门控策略。

**🔧 技术方法**

使用 CeNN 灵感的局部运算、递归记忆（PDelta3、GDN2 等）、门控融合、质量验证、后训练蒸馏、KL 对齐、交叉熵训练等技术。

**📊 数据集**

FineWeb‑Edu 进行转换训练，WikiText‑2 作为 probe，SmolLM2‑135M 与 Qwen3.5‑0.8B 预训练模型，FastEval 采样 200 项（MMLU‑Pro、PIQA、MMMLU‑DE、GPQA‑Diamond）用于下游 sanity 检查。

**📈 对比分析**

对比原始注意力层，记录 NMSE、余弦相似度、ΔNLL 等；在 SmolLM2 上 0–2 层通过门控接受，层 3 拒绝；在 Qwen3.5 上 3、7、11 层通过；Integrated Memory 在保持困惑度差 0.07%–0.93% 的同时减少 6.01% 的缓存；FastEval 采样显示 28.5%–32.0% 的整体准确率与原始 30.0% 相近，说明转换后模型基本可用。

**⚠️ 局限性**

实现目前未达到速度提升，CeNN 层相对慢；门控阈值未做细粒度调优；下游评测样本量不足；未完成强制接受实验验证门控是否真正预测下游失败；未覆盖 Qwen 其余注意力锚点；需要更深层的 ablation 与多种 seed 的统计。

---

## 149. Two's a Crowd: Human and AI-Based Copresence for Developers with ADHD

**arXiv ID:** 2609.21254 | [PDF](https://arxiv.org/pdf/2609.21254v1)

**作者:** Veronica Pimenova `[一作]` (University of Michigan), Venkatesh Potluri `[通讯]` (University of Michigan)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对 14 名 ADHD 软件工程师进行半结构化访谈，研究他们在工作中使用的人与人及人与 AI 的共在协作实践及其对生产力和福祉的影响。

**💡 创新点**

将 Copresence 理论与 Forsgren 等人提出的 SPACE 开发者生产力框架结合，系统性阐述 ADHD 开发者的共在需求，并提出针对 AI 助手的设计建议，填补了以往仅关注传统配对编程或人类双人共在的研究空白。

**🔧 技术方法**

主要技术手段为访谈收集、手工转录、主题分析和理论映射；讨论的 AI 工具包括 Claude Code、GitHub Copilot 等代理式编码助手。

**📊 数据集**

使用的数据集为 14 条访谈记录（包括被访者的职业背景、ADHD 诊断信息等元数据），未使用公开大规模数据集或实验数据。

**📈 对比分析**

论文未进行量化对比实验，也未给出性能指标；通过访谈获得的定性证据表明 AI 共在能减少社交焦虑、提升 flow 状态，但也存在认知负荷和隐私风险。

**⚠️ 局限性**

局限性包括样本规模有限、受访者主要来自美国且缺乏跨文化验证；未实现或评估具体 AI 工具原型，缺乏实证实验支持设计建议。

---

## 150. AirSplan: Risk-Aware Motion Planning for Quadrotors in Cluttered 3D Gaussian Splats

**arXiv ID:** 2609.21226 | [PDF](https://arxiv.org/pdf/2609.21226v1)

**作者:** Seth Isaacson `[一作]` (University of Michigan), Ram Vasudevan `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

针对四旋翼在复杂场景中安全导航，提出了AirSplan系统。

**💡 创新点**

创新点在于利用差分平坦性构建连续时间前进可达集，并与归一化3D Gaussian Splat场结合，提供紧致的碰撞过量逼近和高效碰撞检测。

**🔧 技术方法**

采用差分平坦性、Polynomial Zonotopes、归一化3D Gaussian Splat、BVH、神经网络近似及交叉熵采样优化等技术。

**📊 数据集**

使用基于L‑系统生成的100个程序化树木场景（共500条起止对）以及对应的图像训练的高斯斑点、NeRF和非归一化3DGS模型。

**📈 对比分析**

与Splat‑Nav、CATNIPS和Splanning等基线对比，AirSplan成功率81.2%显著高于最强基线51.2%，碰撞检测速度约提升三倍。

**⚠️ 局限性**

局限在于需要完整准确的辐射场、仅在仿真验证、假设轨迹跟踪完美、需离线规划，并未考虑地图不完整性与跟踪误差。

---

## 151. Multiclass Semantic Segmentation of Wildland Fire Images Using Context-Aware Centralized Copy-Paste Data Augmentation

**arXiv ID:** 2609.21241 | [PDF](https://arxiv.org/pdf/2609.21241v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 152. 4DGS-Fixer: Generative Sparse-View 4D Gaussian Splatting with Iterative Refinement Guided by Video Diffusion Priors

**arXiv ID:** 2609.21176 | [PDF](https://arxiv.org/pdf/2609.21176v1)

**作者:** Haitao Huang `[一作]` (Goertek Alpha Labs), Frank Guan `[通讯]` (Singapore Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于密集点云初始化和视频扩散模型迭代细化的生成式稀疏视角4D高斯散射框架，用以实现动态场景的高质量重建。

**💡 创新点**

创新点：①利用多视角深度预测与点云融合得到更完整的几何初始化；②引入预训练视频恢复模型生成伪监督，指导4DGS的迭代优化，突破稀疏视角下缺失信息的瓶颈。

**🔧 技术方法**

技术手段：4D Gaussian Splatting、MVS深度估计、点云融合、预训练CogVideoV2V视频恢复模型、生成式伪监督的迭代优化。

**📊 数据集**

使用Neural3DV基准数据集（六个动态场景，18-21个摄像机，2704×2028分辨率）。

**📈 对比分析**

与STGS、4DGS、4DGaussians、4C4D等基线对比，在Neural3DV上实现PSNR提升约1.89dB，显著优于4C4D。

**⚠️ 局限性**

局限性：迭代细化计算开销大；伪监督的质量依赖预训练视频恢复模型，可能在复杂真实场景中产生不一致或误差。

---

## 153. FOCAL-VLA: Subtask-Guided Geometry Distillation and Implicit World Modeling for Vision-Language-Action Models

**arXiv ID:** 2609.21228 | [PDF](https://arxiv.org/pdf/2609.21228v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 154. TierKV: Long-Context On-Device LLMs via Predictive Multi-Tier KV Caching

**arXiv ID:** 2609.21172 | [PDF](https://arxiv.org/pdf/2609.21172v1)

**作者:** Zhihao Shu `[一作]` (University of Georgia), Wei Niu `[通讯]` (University of Georgia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种面向移动设备的LLM推理框架，利用预测多层缓存优化（PMCO）将KV缓存划分为精确、低秩压缩和闪存三层，以显著降低内存占用并提升预填充吞吐量。

**💡 创新点**

创新点包括：①无训练长度预测器与预测多层缓存优化的联合决策；②分层KV压缩与闪存调度，保持完整上下文；③融合分路径注意力与隐式重构，消除中间全秩张量；④通过I/O重叠将重构开销隐藏在磁盘读写中；⑤针对移动GPU进行自动调优。

**🔧 技术方法**

使用的技术包括：SVD低秩压缩、RoPE、分层缓存策略、预填预测器、闭式求解器、分路径融合注意力、磁盘I/O与重构重叠、以及移动GPU的自动调优。

**📊 数据集**

实验使用的数据集涵盖文本（SlimPajamas、FineWeb、LongWriter‑6k、LMSYS）、视觉（LAION‑5B、OBELICS）、语音（LibriSpeech）等，并在MMLU、ARC、GSM8K、MMMU‑val、LibriSpeech‑clean等基准上评估准确率。

**📈 对比分析**

与 llama.cpp、MNN‑LLM、MLC‑LLM 等主流移动LLM框架比较，在 OnePlus 12 上预填吞吐提升1.2–17.6倍，KV内存减少12.5–34%，最长上下文延长约2.6倍；在其它设备同样获得1.1–1.4×端到端加速，准确率仅下降1–5%。

**⚠️ 局限性**

局限性包括：对MHA模型重构开销仍较高；预测长度误差会影响缓存规划；闪存I/O受限于手机存储带宽；未针对多轮会话前缀重用做优化；与大模型量化兼容性仍有待完善。

---

## 155. CogGym: Towards Large-Scale Comparative Evaluation of Human and Machine Cognition

**arXiv ID:** 2609.21259 | [PDF](https://arxiv.org/pdf/2609.21259v1)

**作者:** Lance Ying `[一作]` (Massachusetts Institute of Technology), Joshua B. Tenenbaum `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出并实现了CogGym框架，用以将数百个认知科学实验标准化为可执行的EML规范，并对50个大型语言模型在258项人类常识推理实验中进行评估；

**💡 创新点**

创新点在于创建统一、可扩展的实验标记语言和半自动化的人机协作标准化流程，使模型和人类行为可在同一实验设置下直接对比；

**🔧 技术方法**

主要技术包括AI助手驱动的实验标准化、EML格式化、模型接口抽象、数据解析与对齐、以及分布式实验执行与统计分析；

**📊 数据集**

使用的数据集为从30+研究实验室筛选的258个认知实验，涵盖文本、图像与视频三种模态，覆盖理论心智、因果推理、道德判断等主题；

**📈 对比分析**

比较方法通过计算模型与人类均值响应的R²以及分布式距离（EMD/JSD）来量化行为相似度，结果显示模型随规模增长对齐度提升，但即便是最佳模型（R²≈0.59文本、0.58图像、0.43视频）仍低于人类自举一致性上限；

**⚠️ 局限性**

局限性包括实验集规模有限、当前EML仅支持单轮非交互任务、缺乏对实验细节的深层分析以及对更复杂交互式认知实验的支持不足。

---

## 156. Ability-Residual Decoupled Modeling for Affective Cognitive Diagnosis

**arXiv ID:** 2609.21214 | [PDF](https://arxiv.org/pdf/2609.21214v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 157. Visual Navigation Transformer with Pose Attention

**arXiv ID:** 2609.21212 | [PDF](https://arxiv.org/pdf/2609.21212v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 158. When to Waddle: A Comparative Study of Bipedal Torso-Stabilization on Low-Friction Surfaces

**arXiv ID:** 2609.21185 | [PDF](https://arxiv.org/pdf/2609.21185v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 159. Beyond Reference-Based Evaluation: Reward Models for Meta-Evaluation of Grammatical Error Correction

**arXiv ID:** 2609.21231 | [PDF](https://arxiv.org/pdf/2609.21231v1)

**作者:** Ruotian Wu `[一作]` (University of Waterloo), Pascal Poupart `[通讯]` (University of Waterloo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于奖励模型的框架RM-EVAL，用于无参考的语法错误纠正（GEC）系统的元评估，并展示了该模型在生成GEC时的应用。

**💡 创新点**

RM-EVAL作为一种无参考的评估工具，能够更好地捕捉人类的偏好，避免了传统参考依赖指标的局限性，并且在性能上与最先进的评估方法相当。

**🔧 技术方法**

使用了奖励模型和在线强化学习（RLHF）技术，特别是通过奖励引导文本生成（RGTG）来优化GEC输出。

**📊 数据集**

使用了SEEDA数据集，该数据集包含大量人类对GEC系统输出的评估数据。

**📈 对比分析**

与传统的GEC评估指标（如M^2和ERRANT）相比，RM-EVAL在与人类判断的一致性上表现更好，且在准确性和Kendall秩相关性方面均优于这些传统指标。

**⚠️ 局限性**

RM-EVAL依赖于SEEDA基准中的人类偏好注释进行训练，可能会继承注释过程中引入的偏见，并且在不同学习者群体、领域或纠正风格的场景中，可能无法可靠地捕捉人类偏好。

---

## 160. SafeStage: Evaluating Safety Before, During, and After Vision-Language-Conditioned Robot Manipulation

**arXiv ID:** 2609.21223 | [PDF](https://arxiv.org/pdf/2609.21223v1)

**作者:** Jinzhu Luo `[一作]` (Worcester Polytechnic Institute), Wei Jiang `[通讯]` (Futurewei Technologies Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 SafeStage 基准，用来在视觉语言驱动的机器人操纵中分别评估任务成功与安全违规，覆盖任务开始前、执行期间和完成后三阶段。

**💡 创新点**

创新点：①提出生命周期安全评估框架，将安全违规划分为初始状态、执行时和最终状态三类；②构建 97 个专门风险任务，并使用事件-与状态评估器实现细粒度检测；③在同一闭环协议下同时评估直接行动与世界模型 VLA 政策，揭示“unsafe success”现象以及提示干预效果有限。

**🔧 技术方法**

技术手段：使用视觉语言驱动模型（π_0.5、GR00T、DreamZero、Cosmos3），RoboLab+Isaac Sim 物理仿真，事件检测、轨迹安全评估、最终状态检查；三种提示（默认、通用安全提醒、场景特定安全说明）。

**📊 数据集**

数据集：自定义 SafeStage 97 任务集合，基于 DROID 域的仿真环境生成；未使用公开大规模数据集，仅在模拟中创建场景。

**📈 对比分析**

比较方法：对四个政策在默认/通用/特定提示下进行多次闭环实验，计算任务成功率 SR、可安全成功率 SSR、失败率 VR 等指标。结果表明约 70% 成功执行仍违规；ISM 阶段最难，提示提升有限；Cosmos3 在所有指标上表现最好。

**⚠️ 局限性**

局限性：仅在仿真环境评估，缺少真实材质损伤、传感噪声、执行误差等；评估器依赖仿真特权状态，可能与真实感知不一致。

---

## 161. AI-GRACE: A Use-Case Operationalization Framework for Agentic AI: From Organizational Objectives and Obligations to Deployment Capabilities and Architecture

**arXiv ID:** 2609.21192 | [PDF](https://arxiv.org/pdf/2609.21192v1)

**作者:** John Cuneo `[一作]` (University of Miami), Gaurav Khanna `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

设计并提出AI‑GRACE框架，以治理与技术实现为纽带，提供系统化评估和部署决策流程；

**💡 创新点**

引入可追溯的GRA‑C‑E方法、七大风险域、RAIL授权级别和Agent Operating Envelope，形成可迭代、可重用的评估与实施链条，桥接治理目标与技术能力；

**🔧 技术方法**

基于设计科学与情境方法工程，融合 ISO/IEC 42001/42005、NIST AI RMF、CRI、ARC 等标准与框架，并结合模型评估、工具治理、监控与证据收集技术；

**📊 数据集**

论文以虚构的零售银行助手案例为示范，未使用真实数据集；通过标准合规性与安全评估数据进行假设性演示；

**📈 对比分析**

未进行实验比较；方法通过案例演示说明流程与逻辑，未给出性能指标；

**⚠️ 局限性**

缺乏实证验证，七大风险域覆盖与能力匹配的可靠性待评估；评估结果高度依赖证据完整性与评估者判断，可能忽略共享平台或多代理交互带来的跨案例影响。

---

## 162. Multi-viewpoint Geo-localization with Event Cameras

**arXiv ID:** 2609.21219 | [PDF](https://arxiv.org/pdf/2609.21219v1)

**作者:** Adam D. Hines `[一作]` (Queensland University of Technology), Tobias Fischer `[通讯]` (Queensland University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一种基于事件相机的视觉地点识别系统 MegaEvent，利用合成事件流对视觉 Transformer 进行大规模训练，以实现对视角变化的鲁棒定位。

**💡 创新点**

创新点包括：1) 通过 Image-to-Event (I2E) 将五个大型 geo‑tagged 数据集转换为合成事件流，构建了超过 8 M 的多视角训练集；2) 结合多相似性损失和 SALAD 聚合器，对 DINOv2‑基础的 ViT/S 与 ViT/B 进行 fine‑tune；3) 贡献了首个多视角事件 VPR 数据集 Springfield‑Event‑VPR，并在该数据集上验证系统的视角鲁棒性。

**🔧 技术方法**

使用的技术包括：事件摄像机 Vision Transformer（DINOv2‑based ViT），Image‑to‑Event 转换（I2E），多相似性损失，Sinkhorn 归一化与 SALAD 聚合器，合成事件流噪声增强，以及多视角批次采样。

**📊 数据集**

使用数据集：训练时使用 SF‑XL、Google Street View Cities、MSLS、MegaScenes、ScanNet 的图像通过 I2E 转为事件流；评估时使用 Brisbane‑Event‑VPR、NSAVP、NYC‑Event‑VPR；新建 Springfield‑Event‑VPR，包含 3 km 步行路线、11.1 km 事件数据库与多视角查询。

**📈 对比分析**

通过与现有事件 VPR 基线（EventVLAD、EventGeM、SpikeVPR）以及 RGB VPR 系统（MegaLoc、SALAD、MixVPR 等）在 Recall@1 与 Recall@10 上对比，MegaEvent 在三大事件数据集上平均 Recall@1 约 0.79‑0.91，领先下一名约 20 % recall points；在 Springfield‑Event‑VPR 上 Recall@1 达到 0.50，远超基线 0.12‑0.41。系统的 descriptor‑to‑match 延迟为 19‑22 ms，参数量为 163‑228 M。

**⚠️ 局限性**

局限性：仅基于合成事件流训练，缺少真实事件数据的约束；对强光照变化的鲁棒性尚未解决；未对不同摄像机的偏置进行校正；需要进一步研究光照不变性与 Sim2Real 问题。

---

## 163. Combining Exploratory Analysis and Automated Analysis for Anomaly Detection in Real-Time Data Streams

**arXiv ID:** 2609.21222 | [PDF](https://arxiv.org/pdf/2609.21222v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 164. Verify, Don't Trust: Agentic Model Development for Video Discovery Retrieval at Scale

**arXiv ID:** 2609.21257 | [PDF](https://arxiv.org/pdf/2609.21257v1)

**作者:** Hao Fu `[一作]` (Meta Platforms, Inc.), Shuai Ding `[通讯]` (Meta Platforms, Inc.)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并评估了一种名为 EvoPilot 的人类审核的自适应实验框架，用于视频发现检索系统的长期在线自我研究。

**💡 创新点**

将实验流程拆分为可验证的协议、可执行的检查和可恢复的状态；通过人类审核与可执行检查相结合，确保跨系统长周期实验结果的可比性与可靠性。

**🔧 技术方法**

采用大型语言模型代理、版本化的检索技能与适配器、可重放的实验记录、事件账本与持续证据、以及对照协议的可执行验证器。

**📊 数据集**

以 Meta VDD 的数亿条视频索引为基础，使用基于种子视频、机器生成的 pivot 查询与用户上下文的请求数据；实验覆盖 37 天、七个研究方向。

**📈 对比分析**

通过对照协议规定基线与处理差异，对终端指标（点击命中率、GSRR）进行可验证的对比；实验中通过重放与突变测试验证检查有效性，最终发现交互头提升 3.20pp 离线命中率、0.66% 相对 GSRR 提升。

**⚠️ 局限性**

仅有 9 个完整实验回合可供评估，验证器的部分检查仅通过重放得到；评估缺乏置信区间、跨组织复现、在线安全门控等。

---

## 165. License Compliance in Open Source Cybersecurity Projects

**arXiv ID:** 2609.21218 | [PDF](https://arxiv.org/pdf/2609.21218v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 166. Robust Structureless Monocular Visual Inertial Initialization Exploiting Line Features and Vanishing Points

**arXiv ID:** 2609.21186 | [PDF](https://arxiv.org/pdf/2609.21186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 167. I'll Keep an Ear Out: Teaching AudioLLMs Proactive Audio Assistance

**arXiv ID:** 2609.21183 | [PDF](https://arxiv.org/pdf/2609.21183v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 168. Implicit Rule Induction with Test-Time Task Embeddings in ARC-like Tasks

**arXiv ID:** 2609.21181 | [PDF](https://arxiv.org/pdf/2609.21181v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 169. Not All Irregularity Is Equal: Causally Isolating a Rare Failure Mode in Japanese Morphological Inflection

**arXiv ID:** 2609.21179 | [PDF](https://arxiv.org/pdf/2609.21179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 170. When Does Reasoning Help in Machine Translation? A Hierarchical Analysis of LRM Reasoning Traces

**arXiv ID:** 2609.21247 | [PDF](https://arxiv.org/pdf/2609.21247v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 171. VLA-Scope: Shift-Aware Failure Prediction for Vision-Language-Action Models

**arXiv ID:** 2609.21246 | [PDF](https://arxiv.org/pdf/2609.21246v1)

**作者:** Kaiwen Zhu `[一作]` (Texas Tech University), Liangkai Liu `[通讯]` (Texas Tech University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个两阶段框架，先通过多模态特征检测并分类输入的分布偏移，再结合动作前缀特征与累计执行表示预测视觉语言动作模型在分布偏移下的失败风险。

**💡 创新点**

创新点在于将初始输入偏移类别与执行过程的动作特征和累计表示相融合，并采用共享逻辑回归实现高效且精确的失败预测。

**🔧 技术方法**

使用了多模态特征池化、类别平衡逻辑回归、PCA降维、累计平均表示、共享逻辑回归预测器，以及基于OpenVLA的视觉语言动作模型。

**📊 数据集**

在LIBERO‑Spatial十个任务上采集的ID与七类偏移（共1,400 OOD rollouts）的数据集进行评估，扰动来源为LIBERO‑Plus。

**📈 对比分析**

与ActProbe、SAFE‑MLP和SAFE‑LSTM等基线进行对比，模型在30和60动作时的ROC‑AUC分别为0.7675/0.8497，均优于基线且CPU推理延迟最低。

**⚠️ 局限性**

局限性在于仅在冻结的OpenVLA政策上验证，缺乏对多任务或动态环境的泛化测试，并且对极端或未见的偏移可能性能下降。

---

## 172. VGGT-CAD: Reconstructing Parametric CAD 3D Model with Geometric Grounding

**arXiv ID:** 2609.21225 | [PDF](https://arxiv.org/pdf/2609.21225v1)

**作者:** Chunan Yu `[一作]` (Nanjing University of Science and Technology), Yang Yang `[通讯]` (Nanjing University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于几何感知的端到端框架 VGGT‑CAD，能够从单张或多视角视频图像中重建可编辑的参数化 CAD 模型；

**💡 创新点**

创新点包括：1）通过摄像机条件化将预训练的 3D 几何先验迁移到 CAD 重建；2）引入可变视角交叉上下文聚合模块以自适应多视角融合；3）提出训练无关的几何感知视角选择策略；4）构建 VideoCAD 多视角 CAD 评测基准；

**🔧 技术方法**

采用的核心技术包括：VGGT 视觉 Transformer（跨视角注意力）、摄像机条件编码/解码器、跨视角上下文聚合器、非自回归 Transformer 序列解码器，以及 LoRA 微调方案；

**📊 数据集**

使用的数据集为 VideoCAD，该基准基于 ABC‑mono 与 DeepCAD 合成的 208,853 个 CAD 模型，共 7,518,708 帧的多视角渲染数据；

**📈 对比分析**

与 DeepCAD、Hunyuan3D、TripoSR、One‑2‑3‑45 等基线对比，VGGT‑CAD 在命令准确率 ACC_cmd、参数准确率 ACC_param、平均 Chamfer 距离 CD 以及无效率 IR 等指标上均显著优于对手，达成新的 state‑of‑the‑art；

**⚠️ 局限性**

局限性主要体现在对细长管状结构和完全封闭腔体的内部几何难以恢复，未来工作计划引入连续参数回归及物理/制造先验以进一步提升重建精度。

---

## 173. A Fully Differentiable Neuro-Soft-Symbolic Framework for Perceptual Task Planning

**arXiv ID:** 2609.21221 | [PDF](https://arxiv.org/pdf/2609.21221v1)

**作者:** Hongyan Wei `[一作]` (Clemson University), Wael AbdAlmageed `[通讯]` (Clemson University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种全可微分的神经软符号框架，能在同一计算图中同时处理视觉感知与任务规划，利用软符号状态与可微软T_P转换，实现梯度传递到感知模块。

**💡 创新点**

创新点在于：①把感知输出保持为连续软符号状态，避免硬分配；②将领域规则提升为可微软T_P运算，使规划过程与感知互相反馈；③在短时窗内规划并重规划，既保证合法性又提升效率。

**🔧 技术方法**

技术主要包括：CNN感知网络 + 方向双线性头、软T_P转换运算、短期可微规划优化、梯度回传到感知参数、Kullback–Leibler锚定、固定点迭代等。

**📊 数据集**

使用的基准数据集：Blocksworld（LatPlan‑40 与 PlanBench‑600）、Logistics 与 Sokoban 任务；全部基于视觉图像或真实符号状态。

**📈 对比分析**

与 LatPlan、Claude3.5、LLaMA、o1‑mini、o1‑preview 等基线对比：在 Blocksworld 上 40/40 与 596/600 成功率，分别超过 82.5% 与 97.83%；平均求解时间从 40.43 s 降至 9.96 s，成本从 42.12 USD 降至 0.89 USD；在 Logistics 与 Sokoban 上分别提升 4% 与 47% 的成功率，且无不合法计划。

**⚠️ 局限性**

局限性：目前仅在实体数量少、动作类型单一的紧凑关系域验证；未处理几何约束、长链依赖、多模态感知等更复杂情形；需要扩展到更大规模、部分可观测或几何约束严苛的环境。

---

## 174. SafeStyle: Calibrated Style Residual Injection for Controllable Style-Leakage Trade-off in Diffusion Stylization

**arXiv ID:** 2609.21242 | [PDF](https://arxiv.org/pdf/2609.21242v1)

**作者:** Zhangping Yang `[一作]`, Yujie He `[通讯]` (Xi'an High-tech Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SafeStyle训练‑free参考风格迁移框架，安全提取并注入风格残差

**💡 创新点**

利用对比式风格纯化、空间粒度自适应传输和残差预算实现风格与内容的分离

**🔧 技术方法**

对比风格/内容子空间投影、SVD、粗细尺度传输算子以及残差正则化

**📊 数据集**

使用SDXL+InstantStyle、50个风格参考与20个目标提示的StyleAdapter子集，以及24个无关参考与10个提示的泄漏压力集

**📈 对比分析**

与IP‑Adapter、InstantStyle、StyleID等9个基线对比，SafeStyle在DINO‑SS最高0.474、文本对齐0.236，泄漏率仅0.008，性能显著优于基线

**⚠️ 局限性**

依赖预先构建的校准集，对极端复杂纹理或几何结构的适应性有限，且仍需手工设定残差预算

---

## 175. Self-Care and Mental Health: Mapping Over A Decade of HCI Interventions

**arXiv ID:** 2609.21239 | [PDF](https://arxiv.org/pdf/2609.21239v1)

**作者:** Anna Fang `[一作]`, Jenny Fu `[通讯]` (Columbia University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对2015–2026年在SIGCHI等HCI期刊发表的91篇关于心理健康自我护理的干预论文进行了系统性综述，旨在描绘该领域的受众、关注主题、支持的自我护理活动以及研究者如何构建技术与护理关系。

**💡 创新点**

创新点在于提出了六种“自我护理取向”（Making the Self Tangible、Quantifying the Self、Training Self‑Care Skills、Facilitating Action、Enacting Social Care、Attending to Affective Experience），用以捕捉技术介入在护理逻辑、关注自我维度与技术功能上的共通假设，并为后续研究提供了新的概念工具箱和对照框架。

**🔧 技术方法**

技术手段涵盖移动应用、可穿戴感知、会话代理（聊天机器人）、人工智能模型（LLM、预测模型）以及XR/VR交互，作者将这些技术映射到上述取向的具体实现方式上。

**📊 数据集**

研究使用的数据集为来自ACM Digital Library的91篇满足自我护理与心理健康双重标准的原始论文，经过筛选、编码、主题分析得到统计与定性洞见。

**📈 对比分析**

方法上采用PRISMA‑ScR流程的文献检索与筛选，随后通过手工编码、主题分析和解释性综合，得到六种取向的分布及其交叉关系。由于本研究为综述性质，没有实验对比指标，但提供了各取向在论文中的频率、常见技术功能以及与自我护理活动的对应关系，展示了HCI在该领域的多样性与聚焦点。

**⚠️ 局限性**

局限性主要包括：①仅检索ACM HCI 会议与期刊，未覆盖医学、数字健康等跨学科文献；②使用英文文献，可能忽视非英语研究；③自我护理的定义与筛选依据需要主观判断，可能导致部分相关论文遗漏；④六种取向为作者解释性构建，并非客观分类，需在后续研究中进一步验证和细化。

---

## 176. Safe Real-Time Policy Steering via Noise-Space Trajectory Optimization for One-Step Generative Policies

**arXiv ID:** 2609.21220 | [PDF](https://arxiv.org/pdf/2609.21220v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 177. Information-Gain Rewards over Diversity-Pruned Tests: GT-Anchored Verifier Co-Training for Reliable Code Generation

**arXiv ID:** 2609.21208 | [PDF](https://arxiv.org/pdf/2609.21208v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 178. Performance Analysis of Cooperative Multi-Carrier Relay-Based UAV Networks Over Generalized Fading Channels

**arXiv ID:** 2609.21238 | [PDF](https://arxiv.org/pdf/2609.21238v1)

**作者:** Ibrahim Y. Abualhaol `[一作]` (Khalifa University), Mustafa M. Matalgah `[通讯]` (University of Mississippi)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了多载波协作型无人机网络在一般化衰落信道下的失效概率与平均可达比特率；

**💡 创新点**

创新点在于将失效概率与可达比特率统一表达为级联加权Q函数，并使用有限混合高斯分布近似衰落信道的比特率分布，支持非同分布的多信道情况；

**🔧 技术方法**

主要技术包括：适应性M-QAM调制、有限混合（Gaussian mixture）与期望最大化（EM）算法求PDF、矩母函数与拉普拉斯逆变换求解析表达式；

**📊 数据集**

实验采用仿真数据，涵盖Rayleigh、Nakagami-m和Weibull等五种衰落场景，子载波分配设定为不同信道类型的组合；

**📈 对比分析**

与Monte‑Carlo仿真结果对比，误差低于3%，验证了解析式的准确性；

**⚠️ 局限性**

局限性：需假设子载波相互独立且使用固定的高斯混合成分数目，且对极端严重衰落场景的精度略低，且未考虑调度与功率管理的动态影响。

---

## 179. Your Programming Students' Cognition with ChatGPT: Higher Performance, Lower Retention, and Reduced Ownership

**arXiv ID:** 2609.21194 | [PDF](https://arxiv.org/pdf/2609.21194v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 180. Succinct Representation of Search Trees on Trees

**arXiv ID:** 2609.21236 | [PDF](https://arxiv.org/pdf/2609.21236v1)

**作者:** Seungbum Jo `[一作]` (Chungnam National University), Nodari Sitchinava `[通讯]` (University of Hawaii at Manoa)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

针对搜索树在树（STT）进行紧凑表示与快速遍历的研究

**💡 创新点**

提出了几乎最优的空间表示（一般STT为n·logℓ+2n位，Steiner-closed STT为2n+logn位）并证明其最优性

**🔧 技术方法**

采用路径分解、位图（BP）编码、RMQ/Cartesian树、扩展STT等技术

**📊 数据集**

论文为理论工作，未使用实验数据集

**📈 对比分析**

通过信息理论下界和构造算法证明空间最优，构造时间为多项式（一般STT O(n)，Steiner-closed STT O(nℓ）

**⚠️ 局限性**

未给出对任意STT的实例最优表示，Steiner-closed STT构造时间仍为O(nℓ)，且实现细节对特殊树结构的适用性待进一步验证

---

## 181. The Complexity of Computing Class Probabilities in BID Probabilistic Databases

**arXiv ID:** 2609.21245 | [PDF](https://arxiv.org/pdf/2609.21245v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 182. KnowDemo: Knowledge-Guided Robot Demonstration Generation from Human Videos

**arXiv ID:** 2609.21229 | [PDF](https://arxiv.org/pdf/2609.21229v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 183. Stochastic Neural Signed Swept Volume for Real-time Chance-Constrained Trajectory Optimization

**arXiv ID:** 2609.21211 | [PDF](https://arxiv.org/pdf/2609.21211v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 184. Human Driver Temperament and the Safety Impact of a C-V2X Denial-of-Service Flooding Attack in Mixed-Autonomy Traffic

**arXiv ID:** 2609.21206 | [PDF](https://arxiv.org/pdf/2609.21206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 185. OnomatoBridge: Onomatopoeia Translation and Rendering Pipeline in Manga

**arXiv ID:** 2609.21199 | [PDF](https://arxiv.org/pdf/2609.21199v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 186. When Better Turns Do Not Make Better Agents: Diagnosing the Gap Between Next-Turn Metrics and Workflow Success

**arXiv ID:** 2609.21187 | [PDF](https://arxiv.org/pdf/2609.21187v1)

**作者:** Md Tahmid Rahman Laskar `[一作]` (Dialpad Inc), Shashi Bhushan TN `[通讯]` (Dialpad Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过比较预训练与监督微调的 Qwen3 与 Gemma 3 模型，在金手段历史评估与闭环自主执行两种协议下，评估多轮客服工作流中模型的文本生成、工具调用和整体任务完成情况。

**💡 创新点**

创新点在于揭示单轮金手段评估无法预测闭环工作流成功率，强调需同时报告文本质量、工具正确性和端到端工作流完成度。

**🔧 技术方法**

采用 ROUGE、LLM 判定、精确工具匹配、闭环重放和整体工作流评估等技术，对 Qwen3 与 Gemma 3 预训练与监督微调版本进行实验。

**📊 数据集**

使用 Dialpad 生成的 1,027 条客户支持对话（共 5,834 条训练样本、542 条验证样本）构成的自定义工作流数据集。

**📈 对比分析**

在金手段历史下 SFT 提升文本 ROUGE‑1 约 24.4 分、文本回合成功率 25.2%，但闭环完成率极低（最高 10.4%），显示单轮指标与端到端性能不匹配。

**⚠️ 局限性**

局限包括仅评估两大模型族、单一客服域、闭环重放使用严格工具匹配且用户模拟器确定化，可能低估真实交互的多样性。

---

## 187. SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?

**arXiv ID:** 2609.21190 | [PDF](https://arxiv.org/pdf/2609.21190v1)

**作者:** George Ma `[一作]` (University Of California Berkeley), Anoop Deoras `[通讯]` (Amazon Web Services Ai Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6215c339-3735-4be3-8a07-5bbb7004712d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个流水线，将真实的 GitHub 问题及其修复补丁转化为可形式化验证的规范、实现与证明，发布了包含 500 个实例的正式验证基准。

**💡 创新点**

创新点在于首次把形式化验证与实际软件修复任务结合；提出了机械与对抗门控、调用者公理化、verify→resolve 保障等技术；并通过三种后端验证器验证了整个流程。

**🔧 技术方法**

使用了形式化验证后端（Nagini、Velvet、Lean），自动化公理生成与审计，LLM 对抗审计，属性式反事实测试，机械与对抗门控组合。

**📊 数据集**

采用了 500 条来自 SWE-bench Verified 的真实问题及其 Python 代码补丁，扩展至 266 条 Python 任务（Python fragment of another benchmark），并在这些数据上生成规范与证明。

**📈 对比分析**

对 Claude Opus 4.8 与 GPT‑5.5 在三种后端下的解决率进行比较，指标包括测试通过率、验证成功率与对抗审计一致率。结果表明：测试不完整；正式规范显著提升 10‑15% 的解决率；自写规范效果不佳；正确规范可将解决率提升到 95% 以上。

**⚠️ 局限性**

局限性包括：公理的真实性与修复代码一致性需对抗审计保证；实现与补丁之间的精细映射依赖审计与精细验证；后端表达能力有限，某些实例需手工建模；泄露门控与对抗审计的主观性仍存在。

---

## 188. X-SPUR: Explainable Surprisal-Based Protocol-Aware Unsupervised Reasoning for Automotive Ethernet Intrusion Detection

**arXiv ID:** 2609.21217 | [PDF](https://arxiv.org/pdf/2609.21217v1)

**作者:** Jisoo Kim `[一作]` (Sookmyung Women's University), Seonghoon Jeong `[通讯]` (Sookmyung Women's University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了X-SPUR，一种基于因果下一词预测的无监督汽车以太网入侵检测系统，使用原始报文字段的字节级分词序列并结合Mamba2状态空间语言模型学习正常模式；通过token级交叉熵惊奇度量检测异常，并融合时间间隔信息以提升判别力，天然支持字段级可解释性。

**💡 创新点**

创新点包括：1) 用原始报文字段的BBPE分词替代传统手工特征，消除特征工程；2) 在Mamba2后端实现双模态（载荷+时间）融合与Hadamard交互；3) dual top‑k% per‑protocol Z‑score校准，适配多协议分布并抑制协议特异误报；4) 将token级惊奇度直接聚合为字段级异常分数，实现内置可解释性。

**🔧 技术方法**

技术栈：Mamba2状态空间语言模型、字节级BBPE分词、加法+Hadamard双模态融合、top‑k% token惊奇度、流级滑动平均、协议级Z‑score归一化、阈值搜索、字段级解释映射。

**📊 数据集**

数据集：TOW‑IDS汽车以太网数据集（包含CAN、AVTP、gPTP协议的多种攻击）和CarDS汽车以太网数据集（IPv6+TCP/UDP/RTP/HTTP等协议，包含网络扫描、RTP替换、REST‑API崩溃）。

**📈 对比分析**

评估方法：在TOW‑IDS上与AERO、CNN、GRU、Transformer等基线做AUC/F1/误报率比较，X‑SPUR在TOW‑IDS上AUC 0.9987，略高于AERO 0.9969；在CarDS上AUC 0.9923，远超AERO 0.6239；同样在其他基线上表现更佳，说明方法在不同协议栈和攻击场景下具有良好的迁移性。

**⚠️ 局限性**

局限性：1) 对AVTP Frame Injection攻击检测仍较弱，因该攻击仅在难以预测的字段产生微弱惊奇度；2) 流级滑动平均导致检测延迟，尤其在高频率流中；3) 模型参数量约98M，部署在车载ECU上仍需压缩或量化；4) 对极低频攻击的召回率仍受限，需要进一步提升模型对稀疏异常的敏感性。

---

## 189. Online Algorithms for Independent Low-Rank Matrix Analysis and Rank-Constrained Spatial Covariance Matrix Estimation Based on Maximum Weighted Likelihood Estimation

**arXiv ID:** 2609.21180 | [PDF](https://arxiv.org/pdf/2609.21180v1)

**作者:** Yuto Ishikawa `[一作]` (University of Tokyo), Kazunobu Kondo `[通讯]` (Yamaha Corporation)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种面向扩散噪声环境的实时单目标语音提取方法，基于在线最大加权似然估计（MWLE）对独立低秩矩阵分析（ILRMA）与秩约束空间协方差矩阵估计（RCSCME）进行在线扩展，能够跟踪目标说话人位置变化；

**💡 创新点**

创新点在于：① 引入MWLE框架将历史帧权重衰减以提升在线性能；② 通过对时间可变参数进行估计近似，获得可在STFT移位长度内执行的在线更新规则；③ 针对NSR‑ILRMA提出数值稳定化技术，针对RCSCME提出加速和MP逆计算技巧；

**🔧 技术方法**

使用的技术包括：最大加权似然估计、MM/ME优化框架、Sherman–Morrison公式、指数衰减加权、NMF建模、以及多通道Wiener滤波；

**📊 数据集**

在合成数据上使用JVS语音、DEMAND噪声与Pyroomacoustics生成的房间冲击响应；在实测数据上使用东京大学伊藤国际研究中心录制的四麦克风阵列真实环境录音；

**📈 对比分析**

与O‑IVA、O‑SR‑IVE、B‑RCSCME等基线比较。实验表明在静态和移动说话人场景下，O‑RCSCME在SDR/SIR提升上平均提升约1–2 dB，且处理时间始终低于STFT移位长度；

**⚠️ 局限性**

局限在于仅支持单目标语音提取，难以直接扩展到多目标场景；且对房间尺寸、噪声类型的鲁棒性仍需进一步验证。

---

## 190. S3VD: Semantic-Guidance Spatio-Temporal Scanning for Video Deraining

**arXiv ID:** 2609.21322 | [PDF](https://arxiv.org/pdf/2609.21322v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 191. OpenRoIS: A Community-Driven Open-Source Middleware Implementing the Robotic Interaction Service (RoIS) Framework for Physical Robots and Virtual Agents

**arXiv ID:** 2609.21178 | [PDF](https://arxiv.org/pdf/2609.21178v1)

**作者:** Sebastian Carrera Villalobos `[一作]` (Coarobo GK), Lotfi El Hafi `[通讯]` (Coarobo GK)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

实现了 OpenRoIS，一个开源中间件，提供了 RoIS 2.0 规范的具体实现，支持服务应用通过统一的符号接口控制物理机器人和虚拟代理。

**💡 创新点**

创新点包括递归引擎架构、内部五方法组件契约、基于 JSON‑RPC 2.0 over WebSocket 的控制平面、单源类型管道以及多语言 SDK 与适配器框架。

**🔧 技术方法**

采用了 JSON‑RPC 2.0、WebSocket、TLS、JWT、RBAC、WebRTC、TypeScript、C#、Python 以及 ROS 2 等技术栈。

**📊 数据集**

未使用传统意义上的数据集，主要通过在 Preferred Robotics Kachaka 和 Pollen Robotics Reachy Mini 上演示参考组件。

**📈 对比分析**

本文聚焦架构与实现，未进行性能对比或基准测试，API 在 1.0 版发布前仍可能调整。

**⚠️ 局限性**

局限性包括缺乏实验评估、功能尚未完全实现、对不同平台的支持依赖外部适配器、兼容性和性能尚未验证。

---

## 192. The Cube-Root Phenomenon in Online Carpooling

**arXiv ID:** 2609.21348 | [PDF](https://arxiv.org/pdf/2609.21348v1)

**作者:** Nikhil Bansal `[一作]` (University of Michigan), Siddharth M. Sundaram `[通讯]` (Georgia Tech)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在线拼车问题，其中边缘在线到达并必须立即定向，同时保持每个顶点的入度和出度之间的差异较小。

**💡 创新点**

提出了一种自然算法，证明其在 T 次到达后产生的差异为 O(min{T^1/3,n})，解决了 Ajtai 等人提出的一个问题。

**🔧 技术方法**

使用了组合数学和潜力分析等技术，特别是跟踪不平衡向量的层次结构。

**📊 数据集**

使用了来自一个 n 顶点图 G 的 O(n) 边的随机样本数据集。

**📈 对比分析**

与之前的算法相比，提出的算法在确定性情况下的性能达到了 O(min{T^1/3,n})，而之前的算法为 O(min{T^1/2,n})，在随机情况下也有类似的改进。

**⚠️ 局限性**

在随机到达的情况下，仍然存在一个开放问题，即是否可以为任意图 G 提供 O(log^1/3 T) 的差异界限。

---

## 193. Multi-Subject Pretraining Enables Short-Calibration Personalization for Closed-Corpus Surface EMG Speech Decoding

**arXiv ID:** 2609.21288 | [PDF](https://arxiv.org/pdf/2609.21288v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 194. What Stops a Small Language Model From Driving a Database Agent

**arXiv ID:** 2609.21341 | [PDF](https://arxiv.org/pdf/2609.21341v1)

**作者:** Cevheri Bozoglan `[一作]`, Koray Sirin `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 LibreDB Studio 的代理模式下，对 39 种本地开源大模型进行了 8,199 次任务跑测，并基于运行账本、拒绝记录和评估器构建了一个完整的失败分类体系，随后通过 5 次服务器端接口修复提升了多模型的成功率。

**💡 创新点**

①提供了首个大规模实测数据库代理失败分类与分布；②揭示了“transport”类（工具被调用但结果未能提交）是主导失败原因，挑战传统的“模型容量阈值”假设；③通过仅修改服务器端接口而非模型或提示即可显著提升多模型性能，展示了接口优先的修复路径。

**🔧 技术方法**

使用了 LibreDB Studio 的代理框架、Ollama 推理引擎、SQLite 示例数据库、Bun+Next.js 运行时以及自研的 ledger、scorer 和 verifier 工具；同时利用 39 个本地模型和 1 个 hosted 控制模型进行评估。

**📊 数据集**

基准数据集包含 8,199 条跑记录、110,711 条 ledger 事件、14,008 条拒绝记录，以及 39 种本地模型和 1 种 hosted 模型的完整运行日志，已公开发布为可复现数据集。

**📈 对比分析**

对每种模型在 6 个任务面（surface）上的成功率进行统计，并通过“cell”锁定（5 次连续通过）评估模型完整性。对比干预前后，6 个模型分别提升了 6–21 个 cell（共 30 次跑）。总体而言，失败中 75.7% 来自已调用工具的跑，说明模型并非因无能而失效。

**⚠️ 局限性**

实验仅在单台机器、单一 SQLite 模式、单一代理实现上进行，模型跑量不均且非随机；未控制的上下文长度和交换空间问题可能影响结果；干预效果在不同配置下可能不同，且无法证明在所有部署场景下均适用。

---

## 195. Hybrid GPU-CPU Retrieval for Personalized Search at Ultra-Large Scale

**arXiv ID:** 2609.21281 | [PDF](https://arxiv.org/pdf/2609.21281v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 196. Stability-aware Residual Reinforcement Learning Framework for Robotic Manipulator Disturbance Compensation

**arXiv ID:** 2609.21307 | [PDF](https://arxiv.org/pdf/2609.21307v1)

**作者:** Jihong Kim `[一作]` (Hanyang University), Hyung-Tae Seo `[通讯]` (Kookmin University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

结合非线性模型预测控制(NMPC)和逆动力学扰动观测器(DOB)，引入残差强化学习策略专门补偿 DOB 无法估计的残差扰动，从而提升机械臂轨迹跟踪精度。

**💡 创新点**

创新点包括：①仅让 RL 负责补偿残差扰动，保持基准控制的理论稳定性；②通过估计器网络与特权扰动上下文对齐，构造以扰动模式聚类的潜在空间，实现在扰动切换时的快速适应；③基于输入到状态稳定性(ISS)分析推导的状态相关动作上限，保证任何 RL 输出都不破坏闭环误差的预定安全边界。

**🔧 技术方法**

技术手段包括：非线性模型预测控制(NMPC)、逆动力学 DOB、强化学习(PPO)、序列编码器与原型对齐的估计器网络、ISS 推导的动作阈值、系统辨识、仿真平台 IsaacLab 与 Gazebo、真实 6-DOF PiPER 机械臂、随机扰动生成、性能评估指标(RMSE、IAE)等。

**📊 数据集**

使用 IsaacLab 仿真生成的随机扰动数据集（正弦扭矩、冲击、摩擦、负载、传感噪声）以及对应的特权扰动标签；测试时在 Gazebo 仿真与真实机器人上应用组合扰动以及未见的基座振动进行验证。

**📈 对比分析**

方法对比：与仅使用 NMPC+DOB 基准对比，采用 RMSE、IAE 等指标评估。实验结果显示残差 RL 能显著降低 DOB 的相位滞后和估计误差（约 79% 降低），在 Gazebo 零-shot 转移下，圆形轨迹误差从 48.0mm→24.4mm，八字轨迹误差从 32.7mm→27.4mm；真实机器人圆形轨迹误差从 25.68mm→18.54mm（27.8%），八字轨迹误差从 27.77mm→23.12mm（16.7%），基座振动下 RMSE 下降 38%。

**⚠️ 局限性**

局限性：稳定性保证依赖基准系统在工作区的 ISS 条件，动作阈值保守导致实际补偿权限受限；未对接触丰富或完全未知扰动签名的情形进行验证；ISS 推导的上限依赖于保守的最坏情况估计，未来需要进一步收紧安全边界和扩展实验场景。

---

## 197. Locating and Enumerating Anycast: a Comparison of Two Approaches

**arXiv ID:** 2609.21292 | [PDF](https://arxiv.org/pdf/2609.21292v1)

**作者:** Remi Hendriks `[一作]` (University of Twente), Roland van Rijswijk-Deij `[通讯]` (University of Twente)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对14k个已知anycast前缀执行大规模traceroute，评估其在枚举和定位PoP的效果。

**💡 创新点**

首次在规模上对traceroute方法进行评估，并将其与现有的iGreedy方法在枚举精度和定位误差上进行对比。

**🔧 技术方法**

traceroute、ICMP ping、latency‑neighbor推断、IPInfo/HOIHO地理定位、iGreedy GCD算法。

**📊 数据集**

CAIDA Ark 274个视点、13,765个已知anycast前缀、DNS根服务器公开数据。

**📈 对比分析**

traceroute比iGreedy多枚举约11.8%，平均定位误差从51km降至26km，但探测成本提高约4倍。

**⚠️ 局限性**

traceroute依赖ICMP Time‑Exceeded，导致无法探测阻塞网络、隐藏跃点；容易过估PoP数量且探测成本高。

---

## 198. Authorization Revocation for Long-Running AI Agents: Root-Scoped Quiescence under Delegation and Asynchronous Execution

**arXiv ID:** 2609.21284 | [PDF](https://arxiv.org/pdf/2609.21284v1)

**作者:** Genliang Zhu `[一作]` (Accentrust), Chu Wang `[通讯]` (Accentrust)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了一种基于授权根的隔离终止（root‑scoped quiescence）协议，能够在根级切断后生成可验证的证书，保证该根不再能导致任何受保护的承诺。

**💡 创新点**

创新点包括：① 用支持代数（antichain of minimal witnesses）定义多根授权关系；② 引入原子再绑定（atomic rebind）receipt，保证仅通过独立有效的根继续工作；③ 设计跨提供者精确门槛、通道记账和叶子证书聚合，形成完整的证书；④ 提供三值决策机制并证明安全性、组合性与进展性；⑤ 提供可执行的17案例测试套件和独立验证器实现。

**🔧 技术方法**

主要技术：可持续事件日志与事务锁、SHA‑256 内容哈希与签名、可组合的叶子证书、唯一通道令牌、支持代数与最小充分集合、根级切断与边界门槛、跨边界证书聚合与冲突检测。

**📊 数据集**

评估数据集为一个17案例的“延迟效应测试套件”，覆盖取消、切断、闸门、重绑定、冲突重试、终端通道等多种情形。

**📈 对比分析**

评估方法：在可执行的 ledger 上跑全部17个案例，并使用独立验证器进行语义重算。要求所有案例通过且无假阳性，44个语义回归测试全部被拒绝；性能说明仅表明实现可执行且满足安全性约束，没有给出具体吞吐量或延迟指标。

**⚠️ 局限性**

局限性：① 仅保证已声明的证书范围内的根切断后无剩余受保护承诺；② 不保证系统整体空闲或业务级完成；③ 对未注册端点或缺失证书的情况返回 Indeterminate；④ 需要完整的配置信息和本地证书，无法覆盖未被证明的外部服务行为；⑤ 不处理受保护效应已产生的历史副作用。

---

## 199. Programming AMD XDNA NPUs with Open-source Compiler Tools: A FlashAttention Case Study

**arXiv ID:** 2609.21264 | [PDF](https://arxiv.org/pdf/2609.21264v1)

**作者:** Erwei Wang `[一作]` (Advanced Micro Devices, Inc.), Samuel Bayliss `[通讯]` (Advanced Micro Devices, Inc.)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在AMD XDNA NPUs上通过开放源码编译工具实现并评估了FlashAttention的四种映射策略，最终推出了可在XDNA2上达到3.62 TFLOP/s的融合核实现。

**💡 创新点**

创新点包括：1）基于多层内存级别的roofline分析，系统性预测并验证不同映射的性能瓶颈；2）将QK^T中间张量保留在计算单元本地内存，显著提升算子融合和数据局部性；3）将该策略推广至多种LLM配置（BERT、GPT、Llama、Qwen、DeepSeek），并提供了可复用的MLIR‑AIR编译流程。

**🔧 技术方法**

使用了AMD XDNA的IRON、DATO、MLIR‑AIR等编译/编程框架，FlashAttention算法实现，Roofline模型分析，数值稳定性评估与能耗测量。

**📊 数据集**

使用公开的LLM权重与配置：GPT‑2 Small/Medium/Large、BERT‑Base/Large、Llama‑2、Llama‑3、Qwen、DeepSeek，序列长度从2 K到128 K。

**📈 对比分析**

通过层级映射对比（layer‑by‑layer、DATO、IRON、MLIR‑AIR融合核），在XDNA2上融合核实现平均提升约1.5‑2×吞吐量（3.62 TFLOP/s），相较于iGPU提升6.5×能效；在XDNA1上提升1.1‑1.5×。性能主要受内存带宽与计算密度决定，实验结果与roofline预测高度一致。

**⚠️ 局限性**

局限性包括：1）大多数实验仅在d_k=64的配置下完成，较大d_k（如128）下的IRON/DATO实现缺失；2）仅评估XDNA1/2两代硬件；3）关注单一注意力子模块，未覆盖完整Transformer推理流水线；4）对不同负载模式（多head、GQA、causal/非causal）的细节优化仍需进一步研究。

---

## 200. PlaceReasoner-Beta: Reasoning-Driven Macro Placement and Benchmarking

**arXiv ID:** 2609.21263 | [PDF](https://arxiv.org/pdf/2609.21263v1)

**作者:** Qiufeng Li `[一作]` (George Washington University), Weidong Cao `[通讯]` (George Washington University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于多智能体的宏放置框架PlaceReasoner-Beta，将宏放置视为闭环推理问题，并配套完整的端到端基准PlaceReasoner-Bench。

**💡 创新点**

创新点是将视觉语言模型与几何/物理检查器相结合，实现多模态推理、可解释性，以及对后期路由和时序反馈的闭环改进。

**🔧 技术方法**

使用视觉语言模型（VLM）生成候选布局，几何检查器验证几何合法性，物理检查器获取早期实现反馈，后路由优化器利用最终PPA进行微调；并利用技能包封装EDA工具交互。

**📊 数据集**

使用开放源代码的RTL设计和Nangate45工艺，构建了16个任务（8个设计×1:1和2:1两种纵横比）的PlaceReasoner-Bench基准。

**📈 对比分析**

与DREAMPlace、RTL-MP、ChiPFormer和SA等基线对比，PlaceReasoner-Beta在所有1:1任务的DRC干净情况下实现最佳时序，TNS下降61.2%，在所有2:1任务中成功路由并将DRC违规控制在≤2个；同时路由线长也有显著缩短。

**⚠️ 局限性**

主要局限在于未进行任务专用训练，模型依赖外部提示；候选布局相似度高，探索空间受限；技能包虽可扩展但仍需人工编码；总体计算开销相对传统方法略大。

---

## 201. NaViRrator: Robot Navigation from Human-Readable Maps through a Learned Visual Route

**arXiv ID:** 2609.21316 | [PDF](https://arxiv.org/pdf/2609.21316v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 202. Fooling Thresholds of Halfspaces

**arXiv ID:** 2609.21329 | [PDF](https://arxiv.org/pdf/2609.21329v1)

**作者:** Minglong Qin `[一作]` (National University of Singapore), Haigang Zhou `[通讯]` (Nanjing University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `6215c339-3735-4be3-8a07-5bbb7004712d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究探讨了为半空间阈值构建显式伪随机生成器，种子长度与半空间数量的多对数关系相关。

**💡 创新点**

创新点在于展示了O'Donnell、Servedio和Tan为多面体设计的生成器同样可以欺骗更广泛的半空间阈值类，并发展了特定于阈值的平滑近似框架。

**🔧 技术方法**

使用了基于Bentkus类型的平滑近似框架和随机稀疏化技术。

**📊 数据集**

使用的数据集为半空间阈值的布尔立方体，具体的半空间数量和阈值参数未在摘要中明确给出。

**📈 对比分析**

与现有方法比较，所提出的生成器在种子长度上实现了多对数级别的依赖，性能优于之前的超线性依赖方法，尤其在k=m的情况下，种子长度为多对数级别。

**⚠️ 局限性**

限制在于当前的分析框架可能无法直接扩展到更复杂的半空间组合，且对种子长度的多项式依赖于k的情况仍需进一步研究。

---

## 203. Routine Blood Tests Outperform CRP for Distinguishing Bacterial From Viral Infection in Children

**arXiv ID:** 2609.21332 | [PDF](https://arxiv.org/pdf/2609.21332v1)

**作者:** Mihaela Demireva `[一作]` (Vector Labs), Dimitar Mitev `[通讯]` (Zdraveto Hospital)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了儿童病毒与细菌感染的区分，并建立基于CBC和CRP的预测模型。

**💡 创新点**

首次在儿童人群中结合CBC与CRP使用机器学习方法，并证明其相较于单一CRP阈值更能准确区分感染类型。

**🔧 技术方法**

采用了逻辑回归、XGBoost以及SHAP解释技术，并通过5折交叉验证和bootstrap评估模型性能。

**📊 数据集**

使用了906名2-14岁儿童在保加利亚医院收集的CBC和CRP数据，包含424例细菌感染、482例病毒感染。

**📈 对比分析**

将模型与CRP基线（阈值22 mg/L）比较，使用AUC、敏感性和特异性评估。XGBoost+CRP模型AUC 81.7%、敏感性 70.8%、特异性 79.2%，显著优于基线AUC 57.4%、敏感性 30.9%、特异性 85.6%。

**⚠️ 局限性**

局限性包括单中心数据、样本量有限、缺乏发病时间信息、持出样本量小且不平衡，以及未与临床医生决策直接对比。

---

## 204. FairLMs: A Turnkey Library for Fairness in Language Models

**arXiv ID:** 2609.21296 | [PDF](https://arxiv.org/pdf/2609.21296v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 205. GUIDE: Designer-in-the-loop Authoring of Conformant Generative User Interfaces

**arXiv ID:** 2609.21285 | [PDF](https://arxiv.org/pdf/2609.21285v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 206. The Sources of Unknowability and Self-refutation in Epistemic and Dynamic Epistemic Logic

**arXiv ID:** 2609.21317 | [PDF](https://arxiv.org/pdf/2609.21317v1)

**作者:** Eiji Yamada `[一作]` `[通讯]` (University of Tsukuba), Eiji Yamada (University of Tsukuba)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在单模态和多模态知识逻辑框架（K、KD、KD45、S5）中定义不可知（unknowability）与不可相信（unbelievability）概念，并证明它们与Moore悖论（Moorean phenomena）之间的等价关系。作者进一步把结果推广到多代理情形，并以Brandenburger–Keisler悖论为例说明某些不可相信公式的特殊机制。

**💡 创新点**

创新点在于：①提出了一个统一且静态的不可知/不可相信定义；②在S5中证明了静态与动态（总信息性、终极自我否定）概念完全等价于Moorean现象；③扩展到多代理框架，揭示交互导致的新的Moorean机制；④对BK悖论的可信性分析，指出其不完全属于经典Moorean，但在更广义下仍具备Moore属性。

**🔧 技术方法**

技术主要使用：Kripke模型与框架的语义定义、分离正规化（disjunctive normal form）与可证化变换、逻辑等价推导、证明不相容性与自我否定的等价性。

**📊 数据集**

无数据集；本研究为纯理论逻辑研究，未涉及实验或数据。

**📈 对比分析**

无实验对比；论文通过形式化证明展示等价性与性质，未进行性能或数值评估。

**⚠️ 局限性**

局限性：①结果仅在KD4及其超类（KD45、S5）中完整成立，K、KD框架下部分等价不完全；②动态等价性在多代理情形下未完全证明；③缺乏对公共知识、分布式知识等扩展的完整分析；④未考虑意识或有限知识模型可能带来的新机制。

---

## 207. Identifying Security Platform Product Abuse with Machine Learning

**arXiv ID:** 2609.21303 | [PDF](https://arxiv.org/pdf/2609.21303v1)

**作者:** Shaefer Drew `[一作]` (CrowdStrike), Vitaly Zaytsev `[通讯]` (CrowdStrike)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一套基于多源数据的机器学习框架，用异常检测、风险评分与可解释性结合，自动识别并生成安全平台滥用事件的调查线索。

**💡 创新点**

创新点包括：1）融合多时间尺度的多源数据（浏览器指纹、关键事件、IP上下文、企业信息、工单）；2）将无监督异常检测与领域专家编码的分层风险评分相结合；3）采用贝叶斯优化动态调优权重；4）通过SHAP与风险因子归因提供多层可解释性，显著提升召回率并降低误报。

**🔧 技术方法**

使用技术包括：Isolation Forest（异常检测），贝叶斯优化（权重调优），SHAP（局部解释），层级加权风险评分公式，Python+scikit-learn、Gpyopt等开源库；数据工程从日志、BFP、IP、工单等数据库抽取、特征化。

**📊 数据集**

数据集涵盖5类平台遥测数据：浏览器指纹、关键事件、IP上下文、企业信息、工单；结合28例已标注的滥用案例（规则、红队、人工调查等），以及5个月的实时未标注生产数据进行Beta测试。

**📈 对比分析**

评估方法：在历史标注语料上与基线规则比较，计算召回率和每月平均线索；在Beta阶段与规则队列对比，计算精确率和误报率。结果显示召回率提升35%（vs 4%），线索量减少30%；Beta中精确率提升至13.3%（vs 3.7%），误报率下降57.6%，总体提升约10倍。

**⚠️ 局限性**

局限性：标签稀缺导致模型主要依赖无监督学习；风险因子排名需人工维护，易受专家偏差影响；模型对新型滥用模式的适应性需持续调优；可解释性虽降低误报，但仍需人工审核；潜在的数据泄露与模型泄漏风险。

---

## 208. Co-Evolving Zero-Day Jamming: Adaptive Attack Synthesis and Graph Attention-Based Online Detection

**arXiv ID:** 2609.21334 | [PDF](https://arxiv.org/pdf/2609.21334v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 209. ASTRA: Toward Agentic AI for Intelligent Device-Network-Cloud Synergy in Next-Generation Mobile Communication

**arXiv ID:** 2609.21298 | [PDF](https://arxiv.org/pdf/2609.21298v1)

**作者:** Yalong Guo `[一作]` (Tsinghua University), Changyong Pan `[通讯]` (Tsinghua University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了ASTRA框架，通过在设备、网络和云三层引入自主代理，利用双向语义通道和六阶段协同循环，实现从协议驱动到代理驱动的移动通信网络协同。

**💡 创新点**

创新点在于：① 将智能层与物理层解耦，形成三层代理架构；② 用语义意图和能力抽象消息取代传统压缩接口，实现全决策空间的语义交互；③ 将被动、事件驱动的网络行为转变为预测性、主动式协同，突破协议受限的结构瓶颈。

**🔧 技术方法**

采用大型语言模型（Qwen‑3.5‑2B/9B/122B）实现代理推理与决策，构建语义意图消息、聚合遥测、全局指令和同级协同四类协议，实施六阶段感知‑推理‑通信‑决策‑执行‑学习循环，并在仿真中结合深度学习与强化学习。

**📊 数据集**

使用San Francisco DeepMIMO射线追踪数据集进行基于场景的系统级仿真。

**📈 对比分析**

与传统3GPP基线（最大RSRP蜂窝选择、A3事件手动切换）对比，密集人群场景下平均吞吐量提升13.1%（5%端点吞吐量提升27.7%），高速移动场景下被动切换率降低18.2%，吞吐量、稳定性和公平性均有提升。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实网络部署的延迟与通信开销评估；实验场景有限，未覆盖多域、无人机、卫星等复杂部署；对语义意图泄露、代理身份验证与LLM鲁棒性等安全隐私问题缺乏正式处理。

---

## 210. LEMCA: LLM-Guided Synthesis of Efficient Mode-Switching Control Architectures

**arXiv ID:** 2609.21319 | [PDF](https://arxiv.org/pdf/2609.21319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 211. IntBMoE: Integrating Block-Level Conditioning into Expert Composition for Full-Participation Mixture-of-Experts

**arXiv ID:** 2609.21346 | [PDF](https://arxiv.org/pdf/2609.21346v1)

**作者:** Ran Cheng `[一作]` (Alibaba Group), Xiangxiang Chu `[通讯]` (Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出IntBMoE，一种块条件化的混合专家模型，能独立调控专家参与度、执行成本和参数材料化；

**💡 创新点**

通过把专家池先合成有限个可重用块，再用路由器对每个 token 仅选择少数块执行，实现全专家参与、稀疏执行与固定参数量的解耦；同时引入Dual‑Path Residual Gating（DPRG）提升块内部表达能力；

**🔧 技术方法**

利用超网络（hypernetwork）将码表嵌入映射为价值与门控系数，构造块参数；采用Top‑k块路由、块级特征过滤和共享SwiGLU专家；实现缓存可预计算块参数；

**📊 数据集**

主要在ImageNet‑1K（视觉），MiniPile（语言建模）和IntTravel（序列推荐）上进行评估；

**📈 对比分析**

与多种稀疏路由、全参与混合和参数合并方法对比，IntBMoE在ImageNet‑1K实现Top‑1 73.76%/Top‑5 91.48%，比最强对手SMEAR提升1.98/1.15个百分点；在MiniPile、IntTravel上同样表现优于基线，PPL、HR@1、NDCG均有提升；

**⚠️ 局限性**

需额外设计块码表与超网络；块合成在推理时需缓存，若不缓存计算和内存随专家数增大；对超参数敏感，性能依赖于块数K、选择数k、专家数E、块深度L；适用性在大规模实时系统中已验证但在更复杂任务中需进一步评估。

---

## 212. Deep Reinforcement Learning with Buffered Quantile Objectives

**arXiv ID:** 2609.21327 | [PDF](https://arxiv.org/pdf/2609.21327v1)

**作者:** Mohammad Alipour-vaezi `[一作]` (Virginia Tech), Sajad Khodadadian `[通讯]` (Virginia Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种名为Deep–BQRL的无模型分布式强化学习框架，用于在有限时限环境中以下缓冲分位数目标做决策，并在资产卖出（最优停止）和滑动FrozenLake上验证。

**💡 创新点**

创新点在于：①将下缓冲分位数方法迁移到神经网络逼近层；②采用增广轨迹表示压缩历史依赖；③利用分布式评估器集合的分位数不一致度进行探索；④用分位数回归损失直接学习剩余收益的分位数，从而避免显式模型学习与分布规划。

**🔧 技术方法**

技术手段包括：分布式Critic网络估计多层分位数；下缓冲动作分数计算；分位数Huber损失；目标网络软更新；经验回放；集成误差探索奖励；增广状态（时间步、状态、累计奖励）输入。

**📊 数据集**

数据集：①资产卖出环境——25个报价状态、最多10步的离散决策；②滑动FrozenLake（4×4）——含稀疏奖励的随机导航任务。

**📈 对比分析**

与模型基UCB–BQRL、PPO、TRPO比较：在资产卖出中，Deep–BQRL在点分位数政策间隙（τ=0.1,0.9）优于PPO/TRPO但仍落后于UCB–BQRL；在FrozenLake中，Deep–BQRL的累计奖励接近UCB–BQRL，优于PPO/TRPO。整体而言，Deep–BQRL在风险敏感指标上表现不错，但未能完全取代模型基方法。

**⚠️ 局限性**

局限性：①在小规模离散环境中才验证，缺乏对大规模或连续动作空间的可扩展性评估；②缓冲宽度β与分位数数量K的敏感性未充分探究；③由于使用下缓冲目标，最终政策的点分位数最优性并不保证；④集成方法提升了探索但增加了计算成本。

---

## 213. VeriFuse: Bounded Vision-Language Arbitration and Reason-Guided Refinement for Cooperative 3D Perception

**arXiv ID:** 2609.21323 | [PDF](https://arxiv.org/pdf/2609.21323v1)

**作者:** Hongyi Lin `[一作]` (Tsinghua University), Jinhua Zhao `[通讯]` (MIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种车辆–基础设施协同3D检测的有界仲裁框架（VeriFuse），利用已有检测生成候选集，冻结的VLM对候选集执行选择、细化或拒绝三种有限动作，并由确定性编译器将细化动作映射为受限几何更新。

**💡 创新点**

创新点在于：①把VLM的作用限定在“有限动作”而非直接回归盒子；②构造源条件候选池（包括扰动、交叉源插值）；③设计基于原因代码的有限token细化和再验证机制，使得语义推理与精确几何执行分离；④通过可验证的动作-执行接口实现可靠协同感知。

**🔧 技术方法**

技术包括：①专用3D检测器（如PointPillars）产生原始检测；②候选生成器使用源特定扰动网格和交叉源线性插值；③冻结的多模VLM（Qwen3‑VL‑8B‑Instruct）完成动作决策；④确定性编译器将token映射为盒子修正并进行边界约束验证；⑤再验证环节和后备策略。

**📊 数据集**

使用了DAIR‑V2X‑C数据集进行评估，测试集为544帧（去除无非Ego车辆帧），并使用独立的延迟测试序列。

**📈 对比分析**

与传统独立检测、早期/中间/后期融合、学习型几何选择器等基线比较。VeriFuse在车辆侧和协同侧的3D AP_50/AP_70分别提升到0.646/0.557和0.494/0.357；召回率提升至0.520；FP/帧降低到0.79。通信方面，检测列表+ROI裁剪在10 Mbps链路下将负载减少98.9%、平均延迟降低92.1%。

**⚠️ 局限性**

局限性包括：①只能修正已有检测的候选，无法发现所有代理均未检测到的物体；②对检测器质量、传感器配置和预设token空间敏感；③VLM推理延迟仍不适用于硬实时控制；④受限于固定的几何边界，难以处理全局标定或关联错误。

---

## 214. MIRCID: Inferred Hub-miRNAs Drive Cross-Task Improvements in Drug Mechanistic Modeling

**arXiv ID:** 2609.21280 | [PDF](https://arxiv.org/pdf/2609.21280v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 215. An Introduction to Compression-Based Machine Learning

**arXiv ID:** 2609.21309 | [PDF](https://arxiv.org/pdf/2609.21309v1)

**作者:** John Hurwitz `[一作]` (University of Maryland Baltimore County), Charles K. Nicholas `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `fede83ac-7505-405f-ab37-e7284695c47f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述并统一了基于无损压缩的机器学习方法，构建了四维设计框架，并在文本、恶意软件与图像三大领域通过实验验证NCD与MDL的性能。

**💡 创新点**

提出了压缩器、上下文、度量与预测四个设计维度的统一框架，并系统评估了多种组合，证明聚合操作对性能影响巨大，且在少样本场景中能显著提升准确率。

**🔧 技术方法**

利用传统无损压缩算法（如gzip、bzip2、LZ77等）、压缩距离度量（NCD、CDM、LZJD、BWMD等）、MDL条件码长、k‑NN、聚类、决策树等技术组合实现分类。

**📊 数据集**

实验使用AGNews文本数据集、Drebin恶意软件数据集以及MNIST图像数据集。

**📈 对比分析**

将压缩基方法与传统基线（TF‑IDF+LR/SVM、哈希4‑gram+1‑NN、原像像素+1‑NN）在少样本与全量数据上进行对比，发现NCD/MDL在恶意软件少样本分类提升约0.27点，在文本与图像少样本上提升0.17–0.62点，整体比基线更优；在完整MNIST上表现不佳。

**⚠️ 局限性**

需针对压缩器、上下文、度量与聚合进行超参数搜索；神经压缩算法速度慢、对大规模数据不友好；压缩距离受窗口限制，对长序列效果有限；不同域聚合方式差异显著，缺乏统一最佳策略。

---

## 216. Fast And Accurate Text Content File Type Identification

**arXiv ID:** 2609.21306 | [PDF](https://arxiv.org/pdf/2609.21306v1)

**作者:** Manu Nandan `[一作]` (CrowdStrike), Edward Raff `[通讯]` (CrowdStrike)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种轻量级神经网络模型Labeler，专门用于从文本内容文件中快速准确地识别文件类型。

**💡 创新点**

创新点在于设计了面向文本的自定义分词器和双输入卷积网络架构，仅处理文件的前后4KB块，显著提升准确率与速度，并且在样本不足时表现优异。

**🔧 技术方法**

技术手段包括自定义Tokenizer、共享嵌入+1D卷积、ReLU激活、稀疏交叉熵训练，以及使用TensorFlow与Rust实现高效推理。

**📊 数据集**

实验使用了来自BigCode项目的The Stack和Stack V2公开数据集，约9.75百万个MIT/Apache/BSD许可的源代码文件，覆盖54种编程语言。

**📈 对比分析**

与主流基准Magika对比，Labeler在宏观F1上提升8%（0.9804对0.9057），推理时间约为1.05 ms/文件，比Magika的3.85 ms快3.7倍，模型参数仅715 KB，尺寸比Magika小28%。

**⚠️ 局限性**

局限性包括仅支持文本内容文件，仍在.txt文件上表现最差；对极少量样本的语言需要更多训练；以及在多进程/多CPU环境下的可扩展性尚未全面评估。

---

## 217. Edit-VAR: Taming Visual Autoregressive Model for Precise Video Editing

**arXiv ID:** 2609.21268 | [PDF](https://arxiv.org/pdf/2609.21268v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 218. Cube-Splat: High-Fidelity 360° Gaussian Splatting SLAM via Cubemap Factorization and Adjoint-Consistent Optimization

**arXiv ID:** 2609.21347 | [PDF](https://arxiv.org/pdf/2609.21347v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 219. Brain API: An Intent-Aware Control Plane for Policy-Governed Agentic Systems

**arXiv ID:** 2609.21299 | [PDF](https://arxiv.org/pdf/2609.21299v1)

**作者:** Alexander Chernov `[一作]` `[通讯]`, Alexander Chernov

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了 Brain API——一个面向意图的控制平面，提供决策记录、意图级政策执行和可审计的决策工件，用以治理和监控 agentic 系统的执行。

**💡 创新点**

创新点包括：①将决策过程封装为持久、可版本化的决策工件，①将政策从请求级提升到意图级治理，③提供统一的意图、政策、能力和决策抽象，④在决策层实现可观测性和审计；这些使得系统能在不同后端之间统一治理、重现和审计。

**🔧 技术方法**

技术实现：核心决策层用 Rust 编写，集成 OPA 和 Cedar 等政策引擎；通过能力注册表、上下文提供者、决策引擎和执行适配器实现意图到执行计划的映射；采用决策函数、筛选、排名、计划合成等算法；使用统一 API 接口暴露意图、政策、能力、决策和执行的 CRUD 与查询。

**📊 数据集**

评估数据集：① OPA Gatekeeper 约束库（49 个模板、69 个实例、180 个对象），② Cedar 示例政策集（包含多种授权场景）。

**📈 对比分析**

比较方法：与 Gatekeeper 的 admission log 对比，检查决策工件能否回答审计问题；与 Cedar CLI 的差异化测试验证决策正确性。结果：Gatekeeper 42/42 条目一致；在选取可接受能力时 Brain API 20/20 正确；对 Cedar 只覆盖了 28/49（约 57%）的可编码规则，误差主要源于默认允许/拒绝与多实体比较的限制；性能指标（决策延迟、吞吐量）未量化，只有设计级别的说明。

**⚠️ 局限性**

局限性：①决策层只验证了政策过滤和选择，未评估上下文信号、排名与重排；②决策工件的生成与存储依赖实现细节，未提供完整的安全/一致性保障；③规则语法对集合量化、聚合和多实体比较支持不足，导致对部分治理规则无法表达；④未量化决策层对系统性能、延迟与规模的影响；⑤缺乏对大型多策略冲突、动态上下文漂移的自动化分析与预警；⑥在实际部署中需要额外的工具链支持政策编写、调试与版本管理。

---

## 220. A Confidence-Driven Evolutionary Algorithm for Noisy Optimization with Joint Chance Constraints

**arXiv ID:** 2609.21318 | [PDF](https://arxiv.org/pdf/2609.21318v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 221. A Walk From Free Probability to Matrix Discrepancy III: Higher Rank Kadison-Singer and Spectrally Thin Trees

**arXiv ID:** 2609.21279 | [PDF](https://arxiv.org/pdf/2609.21279v1)

**作者:** Tarun Kathuria `[一作]` `[通讯]` (Google), Tarun Kathuria (Google)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对给定的正半定矩阵组（每个矩阵秩不超过 r、相加为单位矩阵且每个矩阵的谱范数不超过 ε），论文证明存在一种取符号的方式，使得加权求和仍保持谱范数在 O(√ε log (2r)) 以内，并给出一种在受限算术模型下的多项式时间确定性算法；此外还将该结果应用于多重加权图的共同稀疏生成、行约束下的 Beck–Fiala 离散度以及矩阵散度等问题。

**💡 创新点**

主要创新点在于：① 将 Kadison–Singer 的 rank‑1 不一致性技术推广到高秩矩阵；② 通过在源函数中引入对数尺度的“权重”与非线性矩阵幂的混合，得到对源内响应的严谨上界；③ 在证明中引入了逆 Sylvester 度量与负曲率方向的组合，避免了传统的条件数假设；④ 设计了一个基于 SDP 的精确潜能求解器和一个可实现的确定性 walk 算法；⑤ 对多重加权图提供了“统一稀疏化”与“共同稀疏生成树”的算法。

**🔧 技术方法**

技术手段包括：① 变形的矩阵潜能（concave matrix power）与其偏导性质；② 逆 Sylvester 代数与源的负曲率控制；③ 对源函数的非线性 SDP 表示（利用幂分解和几何平均的 SDP 约束）；④ 利用 Hessian 的投影与负曲率方向构造步长；⑤ 通过对称化与张量扩展（双空间）实现对称性与负向量的处理；⑥ 对数尺度的参数 β（β≈1/ log(2r)）控制源与曲率的平衡。

**📊 数据集**

本论文为理论工作，没有使用具体实验数据集；所有证明均基于矩阵代数与 SDP 的理论分析。

**📈 对比分析**

相比之前的 O(√(rε)) 或基于插值多项式的 log(r) 结果，本文得到的上界仅含 log(2r) 并且与维度无关；在多重加权图的稀疏化问题上，提供了同时满足 O(ε log²(2s)) 的谱稀疏树；在行约束的 Beck–Fiala 问题上得到 O(√t log(2t)) 的无维度上界；算法在受限算术模型下实现多项式时间。相对之前的随机或局部 Lemma 方案，算法提供了确定性且可计算的实现。

**⚠️ 局限性**

限制与不足包括：① 常数因子未做最优化，实际数值可能偏大；② 算法在严格的算术模型（SDP 逼近、精确谱分解）下才能保证多项式时间；③ 对于极大秩 r 的情况，log(2r) 仍可能导致较大误差；④ 证明依赖于对逆 Sylvester 度量的严密控制，可能不易推广到更一般的非正半定或不对称情形；⑤ 在实际实现中需要大量 SDP 计算，复杂度较高。

---

## 222. Transcript-Bound Combiners for Downgrade-Resilient Hybrid Post-Quantum Key Establishment: Definition, Proof, and Embedded-Device Cost

**arXiv ID:** 2609.21273 | [PDF](https://arxiv.org/pdf/2609.21273v1)

**作者:** Bhanwar Gupta `[一作]` (Maharishi Markandeshwar (Deemed to be University)), Sanjeev Rana `[通讯]` (Maharishi Markandeshwar (Deemed to be University))

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出在混合 KEM 中将握手记录（transcript）哈希绑定到密钥调度，从而使下线攻击抵抗成为组合器本地属性。

**💡 创新点**

创新点在于：①给出组合器层级的下线攻击定义；②证明仅加入一次公共哈希即可在任意协议层中实现下线抵抗；③给出精确的最强链强度下界并在实验中验证；④在受限设备上提供精确的能耗和计算成本模型。

**🔧 技术方法**

技术手段包括：哈希绑定（SHA3-256）、随机预言机分析、基于分离标签的密钥派生、分离域的哈希输入、对 ML-KEM‑768 与 X25519 的组合器实现、EDHOC 协议映射、以及对 Cortex‑M4 的周期计数与能耗测算。

**📊 数据集**

使用的数据集：公开的 Cortex‑M4 F4 核 ML‑KEM‑768 与 X25519 的周期测量、Keccak‑f 轮次计数、802.15.4 类无线能耗模型及 CR2032 电池规格；随机预言机查询预算（q_H）与 256 位哈希输出长度；实验中采用 2,000 次握手与 2,000 次攻击样本。

**📈 对比分析**

比较方法：在相同协议骨架下对比普通组合器与哈希绑定组合器；测量握手周期、峰值堆栈、传输字节、计算与无线能耗。结果显示：绑定只增加约 11.8% 的计算周期、1.5% 的总能耗、无额外字节；正确握手 100% 合成；在攻击实验中普通组合器被下线攻击成功 100%，绑定组合器始终中止。

**⚠️ 局限性**

局限性包括：安全证明仅在经典随机预言机模型下完成，量子随机预言机的常数尚未严格给出；成本模型是组合式校准而非单板完整测量；仅验证了哈希绑定对侧信道泄露的影响，未包含完整的侧信道防护；假设实体认证已由外部协议提供，未覆盖所有攻击场景。

---

## 223. The EventCV Library for Event-Based Robotic Vision

**arXiv ID:** 2609.21330 | [PDF](https://arxiv.org/pdf/2609.21330v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 224. Pattern-Aware Virtual Network Embedding Optimization for Cloud Data Centers

**arXiv ID:** 2609.21302 | [PDF](https://arxiv.org/pdf/2609.21302v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 225. LEGIT: Credentialing Protocol for Trustworthy AI Agent Marketplaces

**arXiv ID:** 2609.21325 | [PDF](https://arxiv.org/pdf/2609.21325v1)

**作者:** Steve Drew `[一作]` (University of Calgary), Jiayu Zhou `[通讯]` (University of Michigan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了LEGIT协议，结合认证、声誉与市场分配三层结构，为AI代理提供可验证的凭证；

**💡 创新点**

创新点在于将代理的质量、成本、配置、评估预算和证据绑定到可签名凭证中，并在此基础上设计声誉更新与拍卖分配规则；

**🔧 技术方法**

使用了贝塔声誉模型、成本/成功率指标（Cost Per Solved Task）、加密签名与可验证凭证技术；

**📊 数据集**

在GAIA、MATH‑500等公开基准任务上对18种模型‑装载组合进行评估；

**📈 对比分析**

比较方法采用配对检验与引导区间，结果显示模型间存在显著质量差异，装载差异不显著，但成本差异显著；

**⚠️ 局限性**

局限性包括未对多模型/子代理配置进行完整隔离、未评估实际拍卖效果、对噪声反馈的声誉更新分析不足以及对攻击检测的完整性不够完善。

---

## 226. Beyond Exact Match: Task-Aware GRPO for Cross-Domain PCBA Visual Question Answering

**arXiv ID:** 2609.21276 | [PDF](https://arxiv.org/pdf/2609.21276v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 227. GameASG-Bench: Benchmarking Autonomous Software Generation for Game Development

**arXiv ID:** 2609.21293 | [PDF](https://arxiv.org/pdf/2609.21293v1)

**作者:** Xiuhui Zhang `[一作]` (Ant Group), Binhang Yuan `[通讯]` (Ant Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于评估接口规范的浏览器原生游戏生成基准，旨在通过先行定义可测试的行为需求来评估LLM驱动的自动软件生成（ASG）系统是否能交付完整可执行的游戏。

**💡 创新点**

创新点在于：①将测试可行性嵌入任务定义中；②提供统一的评估接口规范（setUp、act、observe、reset）与可复现的场景、操作和观察语义；③采用多层次（源代码检查L1 + 浏览器执行检查L2）行为验证，并细化为核心（P1）与扩展（P2）需求；④构建包含47个跨12类、2D/3D多技术的自包含游戏任务库。

**🔧 技术方法**

技术手段包括：大语言模型驱动的编码代理、头less Chromium自动化、语义状态观察、真实浏览器输入、工具访问（文件、语法检查、调试接口）以及对比实验的多种 harness（Claude Code、Codex CLI）。

**📊 数据集**

数据集为47个任务，每个任务包含生成提示、游戏设计需求、评估接口规范、执行检查脚本和参考实现，涵盖了从RPG到RTS、平台/射击等12大类型，支持Canvas 2D、Three.js等技术。

**📈 对比分析**

比较方法：在同一任务集上评估9个模型堆栈，分别测量严格任务成功率、L1/L2通过率、资源占用（artifact大小、token消耗、成本）。实验发现最高严格成功率为55.3%（26/47），但L2通过率可达93.2%；工具完整访问、较高的回合预算与中等推理开销能显著提升成功率。

**⚠️ 局限性**

局限性：①评估仅覆盖浏览器原生游戏，未涵盖服务器端或多人交互场景；②需要任务作者提前编写接口规范与检查脚本，手工成本高；③某些高难度交互在有限预算/工具访问下仍无法通过；④严格成功率受多层检查严格依赖，未必能捕捉所有业务层面的缺陷。

---

## 228. Efficient Benchmarking in Production: A Study of an Evolving LLM Agent

**arXiv ID:** 2609.21267 | [PDF](https://arxiv.org/pdf/2609.21267v1)

**作者:** Yining She `[一作]` (Carnegie Mellon University), Lei Lin `[通讯]` (Meta)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了生产型大型语言模型（LLM）代理的周期性评估方法，利用真实生产环境中的评估跑数据比较了四种部分评估策略，并在实际部署中验证其有效性。

**💡 创新点**

创新点在于在持续演进的生产代理环境中，对历史缓存、固定子集与自适应测试等方法进行统一比较，提出基于难度分层的固定子集在运营中的可操作性与跨代理迁移性，并给出实用部署建议。

**🔧 技术方法**

使用的技术包括多维二维参数逻辑（2PL）IRT模型、Rasch一维模型、随机抽样、历史结果缓存、难度分层固定子集选择与加权/GP-IRT估计，以及基于 Fisher 信息的二维2PL自适应测试。

**📊 数据集**

实验基于574次生产评估跑，涉及519道人工评测题目，按时间划分为28天的校准集和28天的保留集。

**📈 对比分析**

通过比较MAE、Spearman 与 Kendall 相关性等指标，发现难度分层固定子集在低预算下（≤20%题量）表现最佳，38.5%题量的多维2PL自适应测试在高预算下取得最低MAE（约1.03个百分点），历史缓存在相同执行比例下精度最差。

**⚠️ 局限性**

局限性包括仅在单一组织和单一基准下验证，未公开题目与实现，缺乏对阈值决策影响的评估，以及跨基准推广与不同自适应测试方法在其他情境下的可行性未知。

---

## 229. How Many Humans Is a Judge Panel Worth?

**arXiv ID:** 2609.21277 | [PDF](https://arxiv.org/pdf/2609.21277v1)

**作者:** Chao Li `[一作]` (Tsinghua University), Yunfeng Li `[通讯]` (Autonavi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对32个语言模型组成的判决面板进行人类标签分布参考的多维度有效规模审计，比较了谱效应大小与分布误差匹配。

**💡 创新点**

创新点在于将谱参与比与分布均方误差进行人类分布参考匹配，揭示两者对面板质量评估的不同指示，并量化共识方向方差。

**🔧 技术方法**

采用谱参与比、残差 Gram 矩阵、共识方向方差等统计技术，结合 Monte Carlo 模拟匹配人类分布。

**📊 数据集**

使用 MNLI-m、SNLI 以及 ChaosNLI 的 αNLI 三个类别推理数据集的 100 份人类标签。

**📈 对比分析**

通过对比 ν_H、ν_MSE、n_eff 等指标，发现谱匹配给出的有效规模约为 4–6，而分布误差匹配仅为 2–4，表明谱效应不一定对应更佳分布恢复。

**⚠️ 局限性**

局限性包括仅针对固定的 32 模型面板与三大数据集，未验证新选择或聚合算法，且受限于 100 份标签抽样误差与参考估计不确定性。

---

## 230. LOInK: Learned Optimal Inverse Kinematics via Structured Neural Surrogate Models

**arXiv ID:** 2609.21275 | [PDF](https://arxiv.org/pdf/2609.21275v1)

**作者:** Michael Somerfield `[一作]` (University of Sydney), Ian R. Manchester `[通讯]` (University of Sydney)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过学习可逆映射实现近似最优逆运动学解

**💡 创新点**

将成本函数映射到潜在空间的原点，实现高质量解的直接采样，并利用bi‑Lipschitz可逆网络保证光滑性

**🔧 技术方法**

BiLipNet与PLNet构造可逆网络，operator splitting求逆，训练用MSE任务/辅助损失

**📊 数据集**

平面三自由度机械臂、四足磁性爬行机器人、平面软体操纵器的实验/仿真数据集

**📈 对比分析**

与传统IPOPT优化和IKFlow对比，LOInK计算时间快30-100倍，精度与成本与优化方法相当或更优

**⚠️ 局限性**

隐含空间单连通假设、数据采样受维度灾难、需要后续微调，未处理多连通解

---

## 231. Combining Object Detection with Geometry-Aware Clustering to Distinguish Overlapping Plants in UAV Imagery

**arXiv ID:** 2609.21304 | [PDF](https://arxiv.org/pdf/2609.21304v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 232. Robotic Multiphase Interaction: Manipulating Coupled Liquid and Solid Dynamics with a World Model

**arXiv ID:** 2609.21448 | [PDF](https://arxiv.org/pdf/2609.21448v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 233. CESBench: Benchmarking Large Language Models on Cryptographic Engineering Security for IoT Devices

**arXiv ID:** 2609.21344 | [PDF](https://arxiv.org/pdf/2609.21344v1)

**作者:** Wenquan Zhou `[一作]` (Beijing Institute of Technology), Liehuang Zhu `[通讯]` (Beijing Institute of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CESBench，一个包含 380 项、覆盖 6 个子域和 4 种任务类型（多项选择、判断、情景诊断、代码）的 IoT 加密工程安全基准，并用它评估了 11 个 LLM。

**💡 创新点**

创新点在于：①构建专业领域专家编写的完整基准；②设计四种任务类型以捕捉实现、评估、攻击等多维能力；③使用 LLM 判定器与双重验证来确保评分可信度；④揭示 LLM 在安全结论与理由之间的显著差距。

**🔧 技术方法**

技术包括：LLM 自动答题、自动化代码测试、基于 LLM 的判断器（Qwen3.5-397B）以及第二评审模型（Grok-4.6）和人工复核；评估过程采用零样本提示、贪婪解码、Python/C 代码执行测试。

**📊 数据集**

数据集为 CESBench 自身的 380 个专家编写题目（多项选择 209 题、判断 67 题、情景 63 题、代码 41 题）以及 572 条代码测试用例。

**📈 对比分析**

比较方法是将每个模型在各任务类型和子域上的得分取平均，计算组合得分；性能表现：最强模型得分 83.6%，最弱 54.4%；在多项选择中接近上限（>97%），但判断、情景和代码的分数差距显著，且判断理由得分仅 53%。

**⚠️ 局限性**

局限包括：①评分依赖单一 LLM 判定器，虽然做了多重验证但仍可能存在偏差；②基准仅基于软件实现和测试，未覆盖真实硬件操作；③设备条件与代码语言在代码任务中混合，难以分离两者；④攻击侧题目可能对恶意使用有潜在帮助，尽管作者认为影响低。

---

## 234. Conformal Privacy Auditing: Calibrated Re-identification Attacks with Statistical Guarantees

**arXiv ID:** 2609.21340 | [PDF](https://arxiv.org/pdf/2609.21340v1)

**作者:** Shuo Huang `[一作]` (Monash University), Lizhen Qu `[通讯]` (Monash University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于自举预测的合规隐私审计框架（CPA），可为发布文本的重识别风险给出分布无关、有限样本的置信证书。

**💡 创新点**

将传统的自举预测方法引入隐私审计，利用置信集合的大小作为可解释的泄露代理，兼容日志可访问与仅采样攻击，并在公开与专有模型上均可使用。

**🔧 技术方法**

核心技术为分裂自举预测、适应性预测集、非一致性分数（如 APS 累积分数）、采样仅接口与 LLM/检索攻击器的黑盒包装。

**📊 数据集**

在 TextWash、TAB、WikiBio、Blog Authorship 等多种发布与候选池设置上评估，覆盖从医学案例到个人简历、博客作者等多样文本。

**📈 对比分析**

与传统 top‑k、MRR 等点估计对比，CPA 在保持目标覆盖率（如 95%）的同时给出可解释的集合大小，证书在不同攻击配置下保持稳定，证明其在不同威胁模型下的可比性与实用性。

**⚠️ 局限性**

局限在于仅对声明的候选池与攻击模型有效，需满足交换性假设；对开放世界缺失、数据漂移和未知攻击者不提供保证，且需要针对每个威胁配置重新校准。

---

## 235. Auto-Bidding with Disentangled Advertiser Profiles and Train-Free Adaptation

**arXiv ID:** 2609.21308 | [PDF](https://arxiv.org/pdf/2609.21308v1)

**作者:** Songyue Cai `[一作]` (University of Electronic Science and Technology of China), Xiaofeng Zhu `[通讯]` (Hainan University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ADAPT，一种通过分离静态和动态广告主画像并实现无训练适配的个性化自动竞价框架。

**💡 创新点**

核心创新在于将动态画像拆解为公共与私有子画像，并利用对比学习与相关性损失实现去耦；同时提供训练自由的冷启动与更新机制。

**🔧 技术方法**

使用双向注意力 Transformer 进行画像编码，交叉注意力提取公共画像，多层感知机提取私有画像，信息对比学习、相关性损失以及决策 Transformer（DT）实现动作预测。

**📊 数据集**

在阿里巴巴公开的 AuctionNet‑dense 与 AuctionNet‑sparse 两大广告竞价基准数据集上进行实验。

**📈 对比分析**

与多种离线 RL 与生成式竞价方法（USCB、BCQ、CQL、IQL、DiffBid、DT、CDT、DT‑score、GAS）对比，ADAPT 在所有预算尺度下均取得最高得分，提升幅度最高可达 2.77%。

**⚠️ 局限性**

当前方法仍依赖于足够的历史轨迹来训练画像，对极端稀疏数据和实时漂移的适应性尚有限；且在极大规模实时系统中的推理延迟与模型复杂度未进行深入评估。

---

## 236. ProTracer: Proprioception-Guided Failure Diagnosis in Robot Manipulation

**arXiv ID:** 2609.21369 | [PDF](https://arxiv.org/pdf/2609.21369v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 237. P$^3$-SAM: SAM with Perceptual Parallel Prompt for Few-Shot Strip Steel Surface Defect Segmentation

**arXiv ID:** 2609.21424 | [PDF](https://arxiv.org/pdf/2609.21424v1)

**作者:** Qian Xu `[一作]` (Shandong University), Runmin Cong `[通讯]` (Shandong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 P^3-SAM 框架，通过 Perceptual-Optimized Encoding 与 Parallel Prompt Generator 实现少样本钢铁表面缺陷分割。

**💡 创新点**

创新点在于利用多尺度 Retinex 与原型/伪掩码增强低对比纹理的 POE，以及同时生成语义与空间提示的 PPG，解决传统 SAM 在工业缺陷场景下的提示不足与特征失真。

**🔧 技术方法**

技术包括 SAM 基础模型、ResNet‑50 编码器、ViT‑H 解码器、多尺度 Retinex、原型/伪掩码引导的浅层特征增强、交叉注意力与自注意力并行提示生成。

**📊 数据集**

使用三大工业缺陷基准：FSSD‑12、Surface Defects‑4i 与 ESDIs‑SOD。

**📈 对比分析**

与 TGRNet、CPANet、SCCAN、DCP、VRP‑SAM、MAPTNet 等方法对比，在 Surface Defects‑4i 的 1/5‑shot 上提升约 12% mIoU，整体在三大数据集均取得最佳 mIoU/FBIoU。

**⚠️ 局限性**

局限性：依赖 SAM 的冻结权重和高算力，提示生成对支持掩码质量敏感，且在非钢铁表面或大尺寸缺陷场景下的泛化性能待进一步验证。

---

## 238. People escalate against a competitor labelled human and hold back against one labelled an optimising machine

**arXiv ID:** 2609.21439 | [PDF](https://arxiv.org/pdf/2609.21439v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 239. Constant-List Insertion--Deletion Codes:New Bounds and an Improvement of Levenshtein's Lower Bound

**arXiv ID:** 2609.21395 | [PDF](https://arxiv.org/pdf/2609.21395v1)

**作者:** Han Mao Kiah `[一作]` (Nanyang Technological University), Ruixiao Zeng `[通讯]` (Nanyang Technological University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究了在固定列表大小 L 下，纠删码在插入和删除错误预算中的速率与容错性能的权衡，并给出了新的可达率下界和上界。

**💡 创新点**

通过组合 Lovász 局部引理、生成函数与关键点系数引理，严格改进了 Levenshtein 下界，并在任意固定列表大小下实现了全范围的提升；提出了插入与删除预算互换的算术转换定理。

**🔧 技术方法**

使用概率方法（局部引理）、组合生成函数与关键点系数分析、回文计数、Elias 与 Levenshtein 逆推、以及高阶平均技术。

**📊 数据集**

无实验数据集，全部为理论推导与数值计算。

**📈 对比分析**

与已有 Levenshtein、HSS、Yasunaga、Elias 等上界对比，数值表明在不同错误比例下可达率显著提升，尤其在 δ=0.1 时约提升 11%；插入下界通过分配近似进一步提高。

**⚠️ 局限性**

结果为存在性证明，未给出高效构造与解码算法；计数中存在过度计数和插入下界采用分配近似，混合错误的直接计数仍未突破。

---

## 240. When Online Adaptation Hurts: Parameter-Frozen Test-Time Ensembling for Continual Medical Image Segmentation

**arXiv ID:** 2609.21412 | [PDF](https://arxiv.org/pdf/2609.21412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 241. MicroHookACT: Monocular Microscopic Vision Guided Visuomotor Policy for Flexible Microelectrode Hooking

**arXiv ID:** 2609.21365 | [PDF](https://arxiv.org/pdf/2609.21365v1)

**作者:** Yitong Chen `[一作]` (Institute of Automation, Chinese Academy of Sciences), Shan Yu `[通讯]` (Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了MicroHookACT框架，用于在单目显微镜下实现柔性微电极（FME）针线钩取的自动化钩挂过程；

**💡 创新点**

创新点在于（1）提出单向钩挂策略，利用焦距梯度和光轴导向实现无手触感的精准对齐与穿线；（2）引入基于动作监督的对象注意机制，在不需要人工标注的前提下自动聚焦针尖与微环；（3）设计全球-局部特征自适应调制，根据预测的动作进度动态权衡粗略与细粒度视觉信息；

**🔧 技术方法**

使用ACT视觉运动策略、冻结的DINOv3 ViT视觉骨干、动作监督的对象注意模块、ACGL全局-局部特征提取以及动作进度预测与自适应调制；

**📊 数据集**

利用60个由人工遥控完成的针线钩挂演示（含不同针尖、FME位置、背景、角度及不同FME型号），作为训练与验证数据；

**📈 对比分析**

与MicroACT‑3D、HLF、DINOv3ACT以及无自适应调制版本进行对比，在5种评估设置下，MicroHookACT实现了29/30次成功率（96.7%）和平均11.5秒的执行时间，显著优于其他方法；

**⚠️ 局限性**

局限性包括：仍依赖人工演示数据，演示数量有限；模型对不同针头/电极形状、尺寸的泛化能力未充分验证；单向钩挂对显微镜的焦平面深度梯度高度依赖，在焦距梯度不足或显微镜配置变化时可能表现下降。

---

## 242. Beyond Atomic Tokens: Factorizing Syllables for Language Model Pretraining

**arXiv ID:** 2609.21362 | [PDF](https://arxiv.org/pdf/2609.21362v1)

**作者:** Nghia Hieu Nguyen `[一作]` (University of Information Technology, Vietnam National University), Ngan Luu-Thuy Nguyen `[通讯]` (University of Information Technology, Vietnam National University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 Phonemic Tokenizer，利用 IPA 先将文本转化为音素，然后把每个音节拆分为 onset、rime 和 tone 三个成分，保持每个音节对应一个上下文位置，并在此基础上构建 PhonemicBERT，使用三头 H-MLM 进行预训练。

**💡 创新点**

创新点在于：① 无需统计切分或语料库学习，采用确定性音素分解；② 通过共享音素成分显著压缩词表（中文 112 词，越南语 256 词）；③ 通过“完整音节预测”重构语义，提升对同音字替换的鲁棒性。

**🔧 技术方法**

使用技术包括：IPA 音素映射规则、词表构建与字典匹配、BERT Encoder、共享 embedding + 3d→d 投影、Holistic Masked Language Modeling (H-MLM) 三头预测。

**📊 数据集**

使用数据集：中文预训练采用 2.3GB Baidu Baike，越南语预训练采用 20GB 与 PhoBERT 类似的语料；下游评测包括中文 TNEWS、IFLYTEK、THUCNews、CLUEWSC、AFQMC、OCNLI、CSL、BQ、C3、CMRC、CLUENER，越南语 VSMEC、ViHOS、ViHSD、NIIVTB POS、PhoNER、VietMed、ViNewsQA、ViNLI、ViCTSD、UIT-VSFC 等。

**📈 对比分析**

与基准（Char、Subword、SubChar-Wubi、SubChar-Pinyin、PhoBERT、WikiBERT、mBERT、XLM‑R）比较，Phonemic Tokenizer 在词表利用率、Rényi 效率、token 化长度方面表现优异；PhonemicBERT‑Vi 在多数越南语任务上与或优于大型多语言模型；PhonemicBERT‑Zh 在控制实验中与字符/子词模型竞争，尤其在分类、对句子对齐任务上保持稳定；对同音字替换的鲁棒性显著提升。

**⚠️ 局限性**

限制包括：① 同音字无法区分词义，导致需要依赖上下文恢复语义；② 对非标准/外来词需要字符级回退，可能导致序列膨胀；③ 在需要细粒度词形信息的推理/问答任务中表现略逊于大词表模型；④ 语料库规模限制下的预训练效果仍受限于数据覆盖范围。

---

## 243. DEFEAT: Stitching Fragmented File I/O Contexts for Early Ransomware Detection

**arXiv ID:** 2609.21426 | [PDF](https://arxiv.org/pdf/2609.21426v1)

**作者:** Muhammad Ejaz Ahmed `[一作]` (CSIRO Technology), Junaid Qadir `[通讯]` (Qatar University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了DEFEAT框架，通过重构碎片化的文件I/O上下文实现对勒索软件的早期检测。

**💡 创新点**

创新点在于引入File Event Gadgets（FEGs）统一跨文件的I/O事件，并将其转化为Attributed Control Flow Graph（ACFG），利用无监督图嵌入与聚类实现行为簇级标注，显著降低分析师工作量。

**🔧 技术方法**

结合Windows ETW日志提取、FEG构造、ACFG建模、UGRAPHEMB图嵌入（使用Laplacian谱距离）、HDBSCAN密度聚类等技术实现端到端检测。

**📊 数据集**

使用约9,781.6万条I/O事件的数据集，包含67个勒索软件家族292个样本、46个正常软件样本以及49个未见家族的样本。

**📈 对比分析**

与UNVEIL、RWGuard和Peeler对比，DEFEAT在同一数据集上达成99.2%准确率，提升6.57–7.56个百分点，能够在首次加密文件即做出判定，且标注工作减少94%。

**⚠️ 局限性**

主要局限在于需要高频ETW日志收集，产生一定的系统开销和隐私风险；对极其低速或细粒度的对抗性操作仍有一定逃逸空间，且对跨时间窗口的多文件加密场景识别效果有限。

---

## 244. Talking Past the Machine: Morality, Politeness, and Alignment in Human-AI Dialogue

**arXiv ID:** 2609.21401 | [PDF](https://arxiv.org/pdf/2609.21401v1)

**作者:** Marina Mitiaeva `[一作]` (Amazon.com), Lu Xiao `[通讯]` (Arizona State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对近27,000条多轮对话的计算分析，系统性比较了人机（ChatGPT）和人际对话在道德表达、礼貌策略与语言对齐（适应）三维度上的分布、随时间的演化及其相互预测关系，揭示人机对话虽呈现表层合作信号，但缺乏人类对话的社会基础与协同演化；

**💡 创新点**

创新点在于：①将道德、礼貌与对齐三维度整合为同一框架，并在跨人机、人际对话中统一测度；②揭示人机对话中这些合作机制往往反向工作，尤其道德与礼貌的影响符号在AI中呈负向关联；③指出“代理”策略是唯一在两类对话中保持正向合作效应的特征，为AI设计提供方向；

**🔧 技术方法**

技术包括：Moral Foundations Theory + ME2‑BERT情感/道德分类器；Politeness R包提取32个礼貌标记并归入四维度；语言对齐测度采用词汇重叠、语言风格匹配、对齐词汇与情感相似度；混合效应回归预测下一轮对齐，包含先前对齐自回归项；中介分析探讨礼貌在道德对齐中的作用；

**📊 数据集**

数据集：WildChat（≈1M人机对话，包含GPT‑3.5、GPT‑4 Preview/Turbo、GPT‑4o、GPT‑4.1 Mini）与Topical‑Chat（≈10K人际对话），两者均为公开、文本仅、英文。

**📈 对比分析**

比较方法：对话内部的对话者差异（人机中AI vs 用户，人际中对等）使用Wilcoxon检验；时间演化通过对话归一化时间滑动回归；对齐预测使用混合效应模型，系数正负对比两类；结果显示：人机AI在道德与礼貌上高于人类，但对齐下降；人际对齐随对话趋于稳定。

**⚠️ 局限性**

局限：①仅测试GPT系列模型，缺乏跨模型验证；②人机与人际对话在知识背景、角色结构和对话长度上存在差异，虽通过方向化设计和共变检验减小影响但仍可能残留；③测量方法依赖现有分类器与手工聚类，可能带偏；④对齐指标无法区分自愿适应与结构强制。

---

## 245. Knowledge-Graph-Augmented Chronos-2 for HEC-RAS Surrogate Forecasting

**arXiv ID:** 2609.21381 | [PDF](https://arxiv.org/pdf/2609.21381v1)

**作者:** Edward Holmberg `[一作]` (Canizaro-Livingston Gulf States), Mahdi Abdelguerfi `[通讯]` (Canizaro-Livingston Gulf States)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 KG-Chronos-2，一种将冻结的时间序列基础模型 Chronos-2 与水力项目知识图、历史检索和残差校正层耦合的温启动 HEC‑RAS 水面高度（WSE）预测框架。

**💡 创新点**

创新点在于：①冻结的 Temporal 预测器与项目特定的知识图耦合，仅对检索和残差校正进行微调，保持大部分预训练权重；②利用项目知识图进行图条件化的历史检索并融合；③构建输入对齐的残差校正门控器，显著提升预测精度。

**🔧 技术方法**

使用的技术包括：Chronos-2（多变量时间序列预训练模型）、PCA+缩放对增量进行编码、图条件检索（欧氏距离最近邻）、残差校正的岭回归与门控机制、HEC‑RAS 数值仿真数据作为监督信号。

**📊 数据集**

数据集为 MVM_MVK_MVN_Combine HEC‑RAS 项目，包含 4,675 个跨截面、71 个通道、4,675 个跨截面 24 小时预测窗口，共 122,424,225 条 WSE 记录；训练使用 2008 年仿真数据，评估使用 2011 年和 2002 年的 64 个 24 小时窗口。

**📈 对比分析**

对比方法包括：持久性（Persistence）、残差 LSTM、项目条件递归 GeoFNO、基于图的 DCRNN‑style 模型、冻结 Chronos-2 以及 KG-Chronos-2 本身。KG-Chronos-2 在事件均衡 RMSE 上达到 0.246970，较冻结 Chronos-2 降低 14.13%，较 DCRNN 下降 29.38%，较 GeoFNO 降低 39.54%，同时在主动窗口和最终时刻 RMSE 亦获得最佳性能。

**⚠️ 局限性**

局限性包括：仅在固定几何和两条评估事件上验证，缺乏跨项目/跨几何的泛化评估；依赖完整的历史状态和输入计划，未考虑缺失/不确定输入；门控和检索参数的校准可能因数据依赖而产生过拟合；物理量单位和基准未进行完整校验。

---

## 246. JEPA Guided Diffusion: Predictive Vision-Language Conditioning for Generative Traffic Forecasting

**arXiv ID:** 2609.21379 | [PDF](https://arxiv.org/pdf/2609.21379v1)

**作者:** Trinh Tra Giang Nguyen `[一作]` (Ho Chi Minh City University of Technology and Engineering), Ha Duc Bui `[通讯]` (Ho Chi Minh City University of Technology and Engineering)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个解耦的生成式交通预测框架，将交通世界的未来理解与像素级视频合成分离，使用 V‑JEPA 2.1 的冻结视觉编码器提取预测潜在表示，再通过 QFormer+Llama‑3.2‑1B 将视觉特征与行为描述融合，生成与 Cosmos‑Predict2.5 DiT 兼容的条件嵌入，从而驱动视频生成；随后应用后处理提升时间一致性和视觉质量。

**💡 创新点**

核心创新在于（1）将 JEPA 预测潜在特征与语言信息联结，形成可直接对齐至扩散生成器条件空间的轻量级预测器；（2）采用可置换匹配的目标对齐损失，让预测器在不需要细粒度对齐的情况下学习空间-时间语义；（3）设计了多步骤后处理（颜色对齐、前景掩码、拉普拉斯金字塔融合），显著提升生成视频的物理一致性与视觉质量。

**🔧 技术方法**

主要技术包括 V‑JEPA 2.1 视觉编码器、QFormer 与 Llama‑3.2‑1B 的多模态融合、Cosmos‑Predict2.5 DiT 扩散生成器、Qwen3‑VL‑235B‑A22B‑Instruct 的描述精炼、WCT 颜色匹配、前景掩码、拉普拉斯金字塔融合等。

**📊 数据集**

在 Woven Traffic Safety 数据集（外部 2400+ 车辆摄像头视频、内部 539+ 多视角交通事件）上进行训练与评估，专注于 AI City Challenge 2026 Track 5 的生成式交通预测任务。

**📈 对比分析**

与 VLM 仅提示的基线相比，JEPA‑guided 条件提升了 1.4 评分点；再加上后处理后进一步提升 1.6 点，最终在官方排行榜上获得 75.1297 分，排名第三（仅次于 76.4866 与 76.0385 的前两名）。

**⚠️ 局限性**

局限性包括：需先离线预编码大量视频，计算成本仍较高；后处理假设背景静止，对行驶摄像头场景效果有限；模型仅在 Woven 数据集上验证，迁移到其他城市或摄像头视角的泛化能力未充分评估。

---

## 247. Think Locally, Refine Globally for Memory-Efficient 3D Reconstruction

**arXiv ID:** 2609.21437 | [PDF](https://arxiv.org/pdf/2609.21437v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 248. CompAdapt: Adaptable Composite Motion Modeling for Physics-Consistent Text-to-Video Generation

**arXiv ID:** 2609.21455 | [PDF](https://arxiv.org/pdf/2609.21455v1)

**作者:** Haoran Qin `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CompAdapt 框架，集成文本到物理语义解析、复合动力学建模和一拍适应，实现物理一致的视频生成。

**💡 创新点**

创新点包括：① 支持并行、顺序与碰撞等复合运动的统一建模；② 双LLM 解析将自然语言转换为结构化物理语义；③ 动态先验匹配实现一拍适应新物理法则。

**🔧 技术方法**

使用神经 ODE 动力学模块、MTC 与 MOT 组合、改进 Wan-Move 的物理感知视频生成、Qwen2.5-7B 文本解析、SAM2 目标跟踪以及正则化微调。

**📊 数据集**

构建四个物理聚焦数据集：Text2Phys、CollidePhys、CompoPhys、OODPhys，并结合真实物理视频进行评估。

**📈 对比分析**

与 Sora、Veo3、CogVideoX-5B、Wan2.2、Wan-Move 及 NewtonGen、PhysT2V 等基线对比，物理一致性指标 PIS 最高，视觉指标 FID/FVD/PSNR/SSIM 也显著优于基线。

**⚠️ 局限性**

局限性在于对极端结构性 OOD 或物理容量受限的场景适应仍有限，且需要预训练的多种动力学模块，对参考轨迹选择仍有一定影响。

---

## 249. ME-Dex 1.0: Bringing Heterogeneous Tactile Sensing into World Action Modeling

**arXiv ID:** 2609.21449 | [PDF](https://arxiv.org/pdf/2609.21449v1)

**作者:** Xuancheng Zhang `[一作]` (Li Auto Inc), Yu Liu `[通讯]` (Li Auto Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了ME-Dex-1.0，一种将视觉、触觉与动作共同建模的世界动作触觉模型，并在多机器人、多感知平台上训练。

**💡 创新点**

创新点包括：① 将触觉状态与视频同等视为未来观测并联合预测；② 采用三专家（视频专家、触觉专家、动作专家）的Mixture-of-Transformers架构与共享注意力；③ 通过Canonical Hand Model和Unified Tactile Autoencoder实现跨手臂/手套的触觉统一表示；④ 开发Agentic Tactile Data Engine补全缺失的触觉数据。

**🔧 技术方法**

使用了Mixture-of-Transformers、流匹配（flow matching）训练、H-Bridge共享注意力、Canonical Hand映射、统一触觉自编码器、模拟触觉数据生成平台、Diffusion/Flow基动作建模等技术。

**📊 数据集**

使用的数据集包括RoboTwin、DexJoCo、ManiFeel、LeRobot SO-101、Xynova Flex2，以及Agentic Tactile Data Engine生成的触觉数据。

**📈 对比分析**

在RoboTwin、DexJoCo、ManiFeel等基准上与Fast-WAM、π_0.5、Motus、DP-T、DECO等方法对比，ME-Dex-1.0在大多数任务上提高10–15%的成功率，尤其在多手臂协作与触觉敏感任务中显著优于基线。

**⚠️ 局限性**

局限性包括：触觉编码器预训练数据量有限，跨平台泛化仍需改进；当前策略主要通过低频更新和重规划实现触觉反馈，缺乏高频触觉驱动的本地控制；对硬件依赖较强，未充分验证在更多不同触觉传感器上的迁移性能。

---

## 250. Optimal Randomized Proper Online Learning

**arXiv ID:** 2609.21445 | [PDF](https://arxiv.org/pdf/2609.21445v1)

**作者:** Zachary Chase `[一作]` (Rutgers University), Idan Mehalel `[通讯]` (Hebrew University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6`

**🎯 论文内容**

本文证明了随机适当在线学习算法在学习函数类时的最优期望错误界限为O(() log T)，其中()是Littlestone维度，T是时间范围。该结果改进了之前的O(() log^6 T)的界限，并且在最坏情况下是最优的。

**💡 创新点**

创新点在于提供了一个更紧的错误界限O(() log T)，并且证明了对于某些可学习类，该上界是紧的。

**🔧 技术方法**

使用了随机化的适当学习算法，并结合了经典的学习理论工具，如Hedge框架和小-网定理。

**📊 数据集**

论文中没有具体提到使用的数据集，但讨论了Littlestone维度和可学习类的性质。

**📈 对比分析**

与之前的研究相比，本文的上界O(() log T)显著优于O(() log^6 T)。此外，本文还提供了对于某些类的下界Ω(() log T)，表明该界限在某些情况下是必要的。

**⚠️ 局限性**

限制在于下界只适用于某些类，而不是所有类。此外，尽管提供了紧的上界，但对于所有可学习类的具体实现和性能仍需进一步研究。

---

## 251. Decision-Focused Learning for Mean-Variance Portfolio Optimization via KKT-Based Reformulation

**arXiv ID:** 2609.21427 | [PDF](https://arxiv.org/pdf/2609.21427v1)

**作者:** Kensei Nosaka `[一作]` (University of Tsukuba), Yuichi Takano `[通讯]` (University of Tsukuba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于KKT条件的单层决策聚焦学习（DFL）框架，用以直接最小化均值-方差组合优化（MVO）的决策损失，并通过正则化稳定模型学习；

**💡 创新点**

创新点在于：①将传统两阶段预测-优化的分离问题改写为单层非线性优化，完全保留预算与限售约束；②利用KKT最优性条件实现单层化，避免了对MVO的松弛或代理损失的使用；③引入正则化项将学习参数锚定至参考解，降低数值不稳定性与过拟合；

**🔧 技术方法**

使用了决策聚焦学习、KKT最优性条件单层重构、非线性优化求解器KNITRO、Pyomo建模、Clarabel求解器、qpth可微二次规划层以及奥斯卡(OAS)收缩协方差估计；

**📊 数据集**

采用了2003年1月到2025年12月的月度ETF收益数据，构建了两套资产宇宙：①国际多元化的8个发达国家股票ETF；②S&P 500行业ETF的9个板块；

**📈 对比分析**

通过滚动窗口回测与多项投资指标（Sharpe比率、最终财富、累计决策损失、CVaR95、交易换手率）对比DFl-KKT、SPO+、IPO-CF、IPO-GRAD、PFL、等权投资和标普500；结果显示DFl-KKT在大多数指标（SR、FW、CDL、CVaR）上优于其他方法，且正则化进一步提升了表现；

**⚠️ 局限性**

局限性包括：①在大规模资产维度下求解成本高；②非凸互补条件导致易陷入局部最优；③目前未处理卡迪纳尔约束、统一风险度量或鲁棒优化，未来需进一步研究。

---

## 252. SIRA: Reasoning-Aware Surgical Instrument Segmentation via Query-Anchored Alignment

**arXiv ID:** 2609.21402 | [PDF](https://arxiv.org/pdf/2609.21402v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 253. A Scene Language Model for Open-Vocabulary Scene Mapping

**arXiv ID:** 2609.21400 | [PDF](https://arxiv.org/pdf/2609.21400v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 254. Consistent Relexicalization of Clinical Documents using Graph-Based Approach

**arXiv ID:** 2609.21387 | [PDF](https://arxiv.org/pdf/2609.21387v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 255. Tracing the Evidence Behind Zero-Shot Time-Series Forecasting: A Source-First Taxonomy and Audit Framework

**arXiv ID:** 2609.21425 | [PDF](https://arxiv.org/pdf/2609.21425v1)

**作者:** Delun Kong `[一作]` (Technical University of Munich), Ziyue Li `[通讯]` (Technical University of Munich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于证据来源的零射击时间序列预测（Zero‑Shot TSF）分类法，并给出了评审框架；

**💡 创新点**

将零射击TSF的“无更新”标签拆解为三类可辨别的证据边界（冻结LLM先验、参数预训练、检索增强），同时引入四维审计维度；

**🔧 技术方法**

利用文本序列化/提示、数值预训练、检索机制等技术，构建了多种代表性方法并对其进行归类；

**📊 数据集**

未针对特定数据集做实验，主要引用公开的基准（如GIFT‑Eval）和已有方法（LLMTime、Chronos、TimeRAF）的数据使用方式；

**📈 对比分析**

通过举例说明在不同证据边界下的比较（如LLMTime vs Chronos vs TimeRAF），并强调评价需兼顾接口、预测对象、上下文与资源预算；

**⚠️ 局限性**

缺点包括：未提供统一实验结果；评估仍易受隐式上下文/预算影响，需进一步规范数据和检索存储的公开与可复现性。

---

## 256. TrustBOM: A Scalable Architecture for Confidentiality-Preserving SBOMs Across Organizations

**arXiv ID:** 2609.21419 | [PDF](https://arxiv.org/pdf/2609.21419v1)

**作者:** Van Thang Nguyen `[一作]` (Technical University of Berlin), Stefan Tai `[通讯]` (Technical University of Berlin)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 TrustBOM，一种在企业 CI/CD 流水线中通过零知识非成员证明实现机密保留的软件清单（SBOM）认证架构。

**💡 创新点**

创新点在于将 SBOM 通过稀疏梅克尔树与区块链承诺相结合，并在不泄露依赖图的前提下，以线性复杂度对消费者自定义的安全/合规约束生成零知识非成员证明。

**🔧 技术方法**

使用了零知识非成员证明（zk-STARK）、稀疏梅克尔树、RISC Zero zkVM、公开区块链以及 SHA-256 哈希。

**📊 数据集**

使用德国软件公司 adesso SE 的生产应用生成的 CycloneDX SBOM，并基于 200 条 Maven 生态系统已知漏洞的约束进行实验。

**📈 对比分析**

实验通过在 NVIDIA L4 GPU 上测量证明生成时间、证明大小和 zkVM 循环数，结果显示证明生成时间约为 0.9 秒/条约束，证明大小约 0.1 MB/条约束，线性可扩展，成本在数十美元以内。

**⚠️ 局限性**

局限包括对 SBOM 完整性假设、对 zkVM 实现安全性的依赖、无法保证已承诺的构建产物与部署产物一致，以及通过频繁查询可能泄露 SBOM 信息。

---

## 257. RobotEQ-Video: A Video-Centric Benchmark for Social Proactive Intelligence with World-State Taxonomy

**arXiv ID:** 2609.21371 | [PDF](https://arxiv.org/pdf/2609.21371v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 258. Explicit Constructions of Maximum-Cardinality Families of Plateaued Functions with Pairwise Disjoint Walsh Supports

**arXiv ID:** 2609.21389 | [PDF](https://arxiv.org/pdf/2609.21389v1)

**作者:** Chen Wang `[一作]`, Shuailong Li `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出了两种新的显式代数构造方法，生成最大规模的平面化布尔函数族，其成员在Walsh谱上互不重叠且没有非零线性结构；

**💡 创新点**

创新点在于：①首次给出能够预设公共代数度且满足上述性质的显式代数表达式；②构造同时满足最大族数、无线性结构、可控代数度以及全GMM形式；③通过比较表明其与之前的谱方法在EA等价性上不相同；

**🔧 技术方法**

使用的技术主要是：代数正则化（GMM）构造、线性/偏线性映射、Walsh变换分析、无超平面划分、向量化偏移参数及线性变换保持代数度与Walsh支持的性质；

**📊 数据集**

该研究为纯理论构造，无依赖数据集；

**📈 对比分析**

方法上通过解析Walsh支持和线性结构证明最大族数；在代数度方面给出可实现范围并与理论上限比较，结果显示第一构造可达到最优代数度p+1，第二构造达到近最优p+(n−m)/2；

**⚠️ 局限性**

局限性：①第二构造仅在n−m为偶数时可用；②构造的参数受q<∑_{i=2^{d-1}}p_i等限制；③虽然构造出的族满足无线性结构，但是否存在完全不属于GMM的最大族尚未解决。

---

## 259. Prediction Dynamics in Depth-Recurrent Language Models

**arXiv ID:** 2609.21383 | [PDF](https://arxiv.org/pdf/2609.21383v1)

**作者:** Xinyue Luo `[一作]` (Ant Group), Fei Yu `[通讯]` (Ant Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究深度递归语言模型（Depth‑recurrent LM）在推理时通过多次隐状态更新来改进预测，并系统地分析了这些更新如何影响最终答案的保留与变化。

**💡 创新点**

创新点包括：① 通过“精确的margin分解”将更新对决策边界的影响拆解为平移、赢家指向运动和竞争者配对三个量化成分；② 对共享分布的预测概率引入“质量（mass）”与“集中度（concentration）”的概率解释，进一步拆分为常见与对比能量；③ 在不同数值精度（BF16、FP32）与不同评分方式（标签、答案文本）下比较这些几何量的稳健性；④ 通过对已完成轨迹的“depth area”度量，量化不同几何测试在何深度开始保证答案一致。

**🔧 技术方法**

技术方法：
- 余弦、对比范数（oscillation seminorm）与其正交投影；
- 精确的margin与保留区间公式；
- 质量与集中度分解（KL逆熵与条件概率）；
- 统一尺度与LSE中心化的理论分析；
- 精度对齐与数值误差的实验对比；
- 统计重排与坐标置换参考、常数基准、均值/中值中心化对比。

**📊 数据集**

使用的数据集与模型：
- Huginn‑3.5B 与 Ouro‑1.4B 两个深度递归模型；
- 公开多项选择题集 ARC‑Challenge 与 MMLU；
- 额外的归档轨迹（共2704条）用于完整轨迹分析与数值精度测试。

**📈 对比分析**

比较方法：
- 对不同几何测试（raw、mean、quotient、direction、pair）在三类深度（T/4、T/2、3T/4）下计算“normalized depth area”；
- 通过“Δ dir”和“Δ pair”分别评估方向与配对贡献，发现两者共同提升了22–34 % 的深度压缩；
- 在标签与答案文本两种评分方式下，使用误差和答案一致率做对比，证明答案文本评分在“deep‑invariance”上更稳定；
- 在不同数值精度（BF16、FP32 head、Full FP32 recurrence）下测量翻译半径、可移除半径与对比能量，显示精度对翻译保留的显著影响。

**⚠️ 局限性**

局限性：
- 分析主要基于已完成的轨迹，未直接提供在线停止或前瞻预测的算法；
- 只考虑固定候选集合的多项选择问题，未覆盖开放式生成任务；
- 对数值精度与实现细节高度敏感，实验结果可能因不同硬件/软件实现而异；
- 统计参考（坐标置换、均值中心化）虽能揭示结构，但并不能完全解释模型内部的动态决策机制。

---

## 260. ArenaFlow: From Trajectory Ranking to Hierarchical Credit Propagation for Open-Ended Agent RL

**arXiv ID:** 2609.21378 | [PDF](https://arxiv.org/pdf/2609.21378v1)

**作者:** Qiang Zhang `[一作]` (Alibaba Group), Zheng-Jun Zha `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ArenaFlow框架，将对比评估的奖励信号在轨迹、步骤和技能层级进行层次化传播，以提升开放式任务中的LLM代理强化学习效果。

**💡 创新点**

创新在于将对比评估转化为结构化反射反馈，生成关键步骤与可重用技能的信用传播，并通过技能记忆与检索形成长期探索先导。

**🔧 技术方法**

结合锦标赛式相对排名、步骤级深度加权信用传播、使用归因技能评估与记忆更新、以及基于LLM的结构化反射评估。

**📊 数据集**

在Open-Travel、Open-DeepResearch和DeepResearch Bench三大开放式代理基准上进行实验。

**📈 对比分析**

与多种闭源模型、点奖励和对比奖励RL基线以及专有深度搜索系统对比，ArenaFlow在Open-Travel和Open-DeepResearch上均取得最高平均分，提升幅度分别超过20和15分；在DeepResearch Bench上亦领先同类方法并逼近专有系统。

**⚠️ 局限性**

主要限制在于对LLM评判者的质量依赖强、技能记忆管理复杂，以及在超长推理或高维度任务中深度权重可能需要手动调节。

---

## 261. FootQuery: Future-Touchdown-Guided Retrieval from Depth History for Perceptive Humanoid Locomotion

**arXiv ID:** 2609.21447 | [PDF](https://arxiv.org/pdf/2609.21447v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 262. Batched Paillier-Based Hamming-Distance Computation over Binary Embeddings

**arXiv ID:** 2609.21364 | [PDF](https://arxiv.org/pdf/2609.21364v1)

**作者:** Yavor Litchev `[一作]` (Stanford University), Liwen Ouyang `[通讯]` (Xtrace AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

实现并评估了一个基于 Paillier 同态加密的批量 Hamming 距离客户端，该客户端结合了携带分离二进制编码、表式加密、减指数解密、CUDA/CGBN GPU 大整数运算、持久设备状态和批量检索集成等技术。

**💡 创新点**

将多项性能关键技术（表式加密、减指数解密、GPU 大整数运算和持久状态）整合到单一客户端，显著提升批量加密和解码吞吐量，首次在同一实现中展示了两种加密构造（传统与 lookup）与 CPU/GPU 后端的综合性能。

**🔧 技术方法**

Paillier 同态加密、携带分离二进制编码、预计算消息表和噪声表、CGBN 合作 GPU 大整数运算、CUDA、持久状态管理、批量检索集成。

**📊 数据集**

10,000 个随机生成的 512 位二进制嵌入向量以及一个随机查询向量。

**📈 对比分析**

在相同硬件（AMD Ryzen 7 5800X CPU + NVIDIA RTX 3080 GPU）下，对四种配置（传统 CPU、lookup CPU、传统 GPU、lookup GPU）进行批量加密和解码的中位时间比较。lookup GPU 在加密吞吐率和解码吞吐率上分别比传统 CPU 基准提升约 10 倍和 6 倍（具体数值见实验表格），并在两种后端中表现出最优性能。

**⚠️ 局限性**

评估仅覆盖客户端批量运算，未测完整检索延迟、网络传输成本、冷启动时间或小批量情况；未对单一优化进行消融，且 CPU 基准未做充分调优，导致速度比较受限。

---

## 263. Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models

**arXiv ID:** 2609.21363 | [PDF](https://arxiv.org/pdf/2609.21363v1)

**作者:** Yining Wang `[一作]` (Fudan University), Mi Wen `[通讯]` (Shanghai University of Electric Power)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了多模态大推理模型（MLRMs）在公开照片中泄露地理位置的隐私威胁，并提出了一种基于扩散模型的隐蔽扰动防御框架，以阻止模型精确推断用户位置；

**💡 创新点**

创新点在于：①将扰动注入扩散模型的潜在空间，通过逆扩散过程实现对语义层面的可控扰动；②利用专门对齐 GPS 的 GeoCLIP 作为梯度源，精准破坏 MLRMs 的地理信号；③提供可选的局部扩散修补策略，实现从城市级到国家级的分层隐私控制；

**🔧 技术方法**

技术包括：扩散模型（Stable Diffusion + DDIM）潜在扰动、GeoCLIP 对齐梯度、动量优化、总变差与 Sobel 边缘一致性损失、基于 SAM 的区域掩码与文本引导的局部修补；

**📊 数据集**

使用 DoxBench（Level 2/3 个人照片）和 Street View（Google 街景）两个公开基准数据集进行评估；

**📈 对比分析**

与 M‑Attack、GeoShield、ReasonBreak 等基线对比，实验表明在 GPT‑5、Claude Opus 4.5 等五大商业 MLRMs 上，平均误差距离提升 10–11 倍，1 km 级定位准确率降至 <5%，且在视觉质量上 PSNR/FID/LPIPS/CLIPIQA 等指标均优于基线；

**⚠️ 局限性**

局限性包括：①对非常精细的图像内容仍可能产生可观察的细微伪影；②对抗性攻击如强净化或多模型集成仍能部分削弱效果；③实现成本较高（扩散逆过程耗时），需要进一步优化以满足实时社交平台的部署需求；

---

## 264. AgentVidBench: A Multi-Hop Video Question Answering Benchmark for Evaluating MLLM Agents

**arXiv ID:** 2609.21386 | [PDF](https://arxiv.org/pdf/2609.21386v1)

**作者:** Seoyeon An `[一作]` (Krafton), Kangwook Lee `[通讯]` (Krafton)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 AgentVidBench，一个评估多模态大语言模型（MLLM）在多跳视频问答中的空间、时间和因果推理能力的基准。

**💡 创新点**

创新点在于构建了100个需要多步推理、包含25个干扰选项的多选题，并为每题提供工具无关的证据里程碑以评估推理轨迹。

**🔧 技术方法**

采用了多模态LLM、Agentic工作流、ReAct式循环、可控视频检索工具以及轨迹评分框架。

**📊 数据集**

使用71段公开版权视频，涵盖12个细粒度类别，生成了100道多跳问答。

**📈 对比分析**

与单回合LLM对比，Agentic工作流提升了约20‑30%的准确率，轨迹得分亦显著提升；基准在不同框架下展示了显著的性能差异。

**⚠️ 局限性**

局限在于视频来源受版权限制、问题与轨迹的人工标注成本高、基准规模相对有限。

---

## 265. Quantization-Aware Kalman Estimation for Diffusion Sampling

**arXiv ID:** 2609.21407 | [PDF](https://arxiv.org/pdf/2609.21407v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. Omni Demand Understanding: A Benchmark for Contextual User-Intent Inference in Multimodal Interaction

**arXiv ID:** 2609.21392 | [PDF](https://arxiv.org/pdf/2609.21392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 267. Interference-Driven Clustered Optimisation for FM Spectrum Coordination

**arXiv ID:** 2609.21441 | [PDF](https://arxiv.org/pdf/2609.21441v1)

**作者:** Federica Mangiatordi `[一作]` (Fondazione Ugo Bordoni), Emiliano Pallotti `[通讯]` (Fondazione Ugo Bordoni)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对跨境FM频谱协调，提出了基于干扰驱动的聚类优化框架，通过识别并聚合造成外部服务违规的主干扰发射机来分解大规模功率控制问题。

**💡 创新点**

创新点在于利用干扰图的稀疏结构和主干扰排名，构建优化导向的发射机聚类；通过分层模拟退火与全局细化实现近似最优且可解释的功率调节。

**🔧 技术方法**

主要技术包括稀疏矩阵与GPU加速计算、干扰图构建与聚类、受限模拟退火（Clustered SA）以及残余交叉校正的全局细化。

**📊 数据集**

使用真实运营跨境FM规划数据库，覆盖约15,766 km²，包含3,247台可调功率发射机、630台受保护服务及近7 百万干扰关系。

**📈 对比分析**

与基线规划、全规模模拟退火（Full‑Scale SA）及理论上限进行对比。聚类方法在K=30时实现99.68%可恢复外部服务面积，仅比Full‑Scale SA少0.03%，且计算时间仅为其5.9%，显著提升效率。

**⚠️ 局限性**

局限在于对主干扰发射机的K值设定敏感；K过小可能漏掉有益干扰，过大则引入弱干扰导致搜索空间膨胀，且目前仅针对FM广播频段验证，其他频谱或网络环境需进一步验证。

---

## 268. A Unified Dynamic Force Guidance Framework for Performance-Optimized Kinesthetic Teaching

**arXiv ID:** 2609.21416 | [PDF](https://arxiv.org/pdf/2609.21416v1)

**作者:** Chunxin Li `[一作]` (Shanghai Jiao Tong University), Xiangyang Zhu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于变量阻抗与虚拟力的动态力引导框架，用于协作机器人手动教学时提升操作性能并减少对用户的干扰。

**💡 创新点**

创新点包括：①在最小奇异值阈值下动态调整阻抗参数，实现对低性能方向的非对称抑制；②利用性能梯度生成有界虚拟力，主动引导用户避开奇异点；③结合能量有限耗散分析证明系统稳定性；④通过轨迹回放实验量化提升工效，验证该方法可降低生产节拍。

**🔧 技术方法**

核心技术包括：变量阻抗控制、虚拟力生成（基于最小奇异值梯度）、阻抗矩阵的相似变换、能量存储函数与稳定性分析、轨迹回放评估（轨迹长度、最小奇异值、执行时间）。

**📊 数据集**

未使用公开数据集，而是在6-DOF协作机器人上采集10名受试者（共90条有效轨迹）进行实验。

**📈 对比分析**

与直接教学（DT）和仅变量阻抗教学（VAT）对比，采用三条典型路径进行实验。结果显示，DFGT方法使轨迹长度略增（+~5%），最小奇异值提升（+~30%），执行时间缩短（约10%），同时用户感受的最大拉力降低，体现了更高的工效和更友好的交互体验。

**⚠️ 局限性**

局限性包括：仅考虑末端执行器保持固定姿态的平移任务；对旋转和耦合任务的适用性尚待验证；虚拟力参数需手工调节，可能需要针对不同任务的自适应机制；实验规模受限，尚未在大规模工业生产线上验证。

---

## 269. AVT-Fabric: Active Visuo-Tactile Perception via Adaptive Evidence Selection for Efficient Robotic Fabric Comparison

**arXiv ID:** 2609.21377 | [PDF](https://arxiv.org/pdf/2609.21377v1)

**作者:** Chang Gao `[一作]` (King's College London), Shan Luo `[通讯]` (King's College London)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了AVT-Fabric框架，采用RGB-先行的自适应视觉-触觉推理，用可靠性门控决定是否继续处理更多触觉信息，并通过文本记忆与多数投票汇总预测；

**💡 创新点**

创新点在于：①将视觉与触觉的证据消耗动态化，基于答案置信度与logit间距双重阈值实现早停；②利用文本记忆压缩前一步骤信息，减少上下文长度；③在保持高精度的同时显著降低推理阶段数和延迟；

**🔧 技术方法**

核心技术包括多模态大型语言模型（Qwen2‑VL‑7B及其他后端）、LoRA微调、可靠性门控（置信度+margin）、文本记忆编码、投票集成；

**📊 数据集**

使用MLLM‑Fabric数据集，包含220种织物的RGB、GelSight触觉图像、力值及属性等级，测试集400个有序比较；

**📈 对比分析**

与90B MLLM‑Fabric基线对比，AVT-Fabric在400个测试对比上达98.0%准确率，平均仅需1.60个观察阶段，模型端延迟比被动推理低61.8%，并在机器人实测中获得78.1%排名准确率，成功选择7/8个场景；

**⚠️ 局限性**

局限在于：①所有触觉样本需预先采集，无法实现在线感知与控制的闭环自适应；②阈值选择对不同后端和数据分布敏感；③在细粒度或全新织物的泛化性能仍低于主基准；

---

## 270. TokaGLINT: A Scalable GPU-Tailored Implicit Solver for Full 3D Tokamak Electromagnetic Simulations

**arXiv ID:** 2609.21366 | [PDF](https://arxiv.org/pdf/2609.21366v1)

**作者:** Zifan Yang `[一作]` (Computer Network Information Center, Chinese Academy of Sciences), Shan Liang `[通讯]` (Computer Network Information Center, Chinese Academy of Sciences)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2` `70e40602-aae3-44bd-80ec-4a7f2674330f` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在SymPIC的框架下，开发了一款GPU加速的隐式Maxwell方程求解器TokaGLINT，实现了全3D托卡马克电磁仿真的高效并行求解

**💡 创新点**

创新点包括层级并行域分解与张量结构快速子域求解器的共设计，以及在曲线坐标下实现的精确离散变换来解耦未知量

**🔧 技术方法**

采用Crank‑Nicolson FDTD、层级加法史瓦兹预条件器、GPU自适应批处理、张量变换与批量GEMM等技术

**📊 数据集**

使用EAST托卡马克实验中的波加热、低频波注入数据集以及对仿真模型的数值测试

**📈 对比分析**

与HIP-enabled HYPRE基线对比，单节点实现2.67倍加速，在10,000 GPU上保持90.1%弱标度效率和53.9%强标度效率

**⚠️ 局限性**

局限在于对极低频或非圆柱坐标的扩展仍有限，需要进一步研究全隐式耦合和更复杂物理耦合

---

## 271. Field Tracking of Insects Using a Stereoscopic Event-Based Camera Setup

**arXiv ID:** 2609.21354 | [PDF](https://arxiv.org/pdf/2609.21354v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 272. From Memory to Behavior: A Behavior-Aware Role-Playing Framework for Social Media Influencers

**arXiv ID:** 2609.21349 | [PDF](https://arxiv.org/pdf/2609.21349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 273. PrismAlign: Prior-Steered Multi-View VLM Alignment for Hallucination-Robust Table OCR

**arXiv ID:** 2609.21351 | [PDF](https://arxiv.org/pdf/2609.21351v1)

**作者:** Guangyi Liu `[一作]` (Huawei Technologies Co Ltd), Boyu Hou `[通讯]` (Huawei Technologies Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了多视角 Vision‑Language 模型框架 PrismAlign，用以通过结构和语义一致性对齐多模型表格提取结果，显著降低表格结构和内容的幻觉。

**💡 创新点**

创新点在于将表格结构对齐与单元格内容对齐解耦，并利用贝叶斯决策结合表格语义规则和多模型一致性得分，以最大化后验正确率。

**🔧 技术方法**

技术主要包括多模型候选生成、表格语义规则评分、TEDS-结构共识评分、贝叶斯后验计算以及细粒度单元格编辑距离过滤。

**📊 数据集**

实验使用了 OmniDocBench 1.5、CC‑OCR、PureDocBench 以及公开的 PubTable‑1M 作为验证集。

**📈 对比分析**

与单个模型相比，PrismAlign 在三大基准上均取得表格 TEDS 的最高分，分别提升 6.4–14.9 点，且在 OmniDocBench 1.5、CC‑OCR 与 PureDocBench 上均刷新 SOTA。

**⚠️ 局限性**

局限性包括多模型推理带来的计算开销、对所有模型均失效表格的无法纠正以及对验证集分布的依赖导致的贝叶斯估计误差。

---

## 274. AESSI: An Around-Ear Silent Speech Interface for Cross-Day Online Reuse without Test-Day Calibration

**arXiv ID:** 2609.21436 | [PDF](https://arxiv.org/pdf/2609.21436v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 275. GVPO++: Group Variance Policy Optimization for LLM Post-Training and On-Policy Distillation

**arXiv ID:** 2609.21432 | [PDF](https://arxiv.org/pdf/2609.21432v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 276. Synthetic Human Mobility Data Generation: A Structured Review of Representations, Methods, and Practical Capabilities

**arXiv ID:** 2609.21413 | [PDF](https://arxiv.org/pdf/2609.21413v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 277. Stabilizing Trajectory Outputs in End-to-End Autonomous Driving via SC-IMM Based Teacher Signals

**arXiv ID:** 2609.21404 | [PDF](https://arxiv.org/pdf/2609.21404v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 278. MarineCraft: Enabling Rapid Prototyping of Underwater Robots via Modular Construction

**arXiv ID:** 2609.21396 | [PDF](https://arxiv.org/pdf/2609.21396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 279. FAN: Foresight Action Normalization for Continual Adaptation of Vision-Language-Action Models

**arXiv ID:** 2609.21358 | [PDF](https://arxiv.org/pdf/2609.21358v1)

**作者:** Yijun Hong `[一作]` (University of Hong Kong), Jiayu Chen `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了连续学习中动作归一化的重要性，并提出了一种基于一次性校准集的FAN方法，以在多任务连续适应中保持动作坐标一致性。

**💡 创新点**

将动作归一化视为连续学习的核心设计，提出一致性、覆盖率、因果性三原则，并提出一次性冻结校准统计量的FAN方案，显著提升多任务持续适应性能。

**🔧 技术方法**

结合视觉‑语言‑动作(VLA)预训练模型、经验回放、序列细调以及基于分位数的动作归一化技术。

**📊 数据集**

在单臂与双臂机械臂平台上收集了10个真实机器人操纵任务（5个单臂、5个双臂）以及一个由8条遥控轨迹构成的校准集。

**📈 对比分析**

与四种基线归一化策略进行对比，采用平均分数、前向/后向迁移率等指标；FAN在四条任务流中平均得到95.3分，前向迁移正向且后向迁移接近零，显著优于基线。

**⚠️ 局限性**

仅在特定平台与任务集验证，校准集的设计需人工抽象，且对极端动态变化或新任务空间的覆盖仍有限；未评估在更大规模任务流或多种执行器的通用性。

---

## 280. Understanding LLM Quantization through Activation-Guided Compensation and Orthogonal Residuals

**arXiv ID:** 2609.21450 | [PDF](https://arxiv.org/pdf/2609.21450v1)

**作者:** Yamato Narita `[一作]` (University of Tokyo), Issei Sato `[通讯]` (University of Tokyo)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过精确分解局部权重‑激活量化误差为激活引导的权重补偿项和正交残差，并利用残差上界指导随机Hadamard旋转、符号采样和L₂通道缩放，构建了一套无反向传播的PTQ配置。

**💡 创新点**

提出了统一的误差分解框架，将旋转与缩放视为补偿项的互补设计；对持久激活外点的干扰进行理论分析得到随机Hadamard旋转和符号采样的上界；从正交残差的二阶矩上界推导出L₂缩放规则，并与L∞规则关联。

**🔧 技术方法**

误差分解与残差上界分析、随机Hadamard旋转与符号采样、正交投影/激活引导权重补偿、L₂通道缩放、GPTAQ等无梯度量化校准。

**📊 数据集**

WikiText‑2（训练、校准、旋转选择）、C4（测试）以及多种零样本任务集（PIQA、ARC‑E、ARC‑C、HellaSwag、WinoGrande、LAMBADA）。

**📈 对比分析**

与AWQ、GPTQ、SpinQuant等基线在W4A4量化下对八个Llama/Mistral模型进行WikiText‑2和C4困惑度以及零样本准确率比较。无梯度配置在WikiText‑2上平均PPL下降至7.20，零样本平均准确率提升约0.5个百分点，整体性能与梯度训练的SpinQuant持平。

**⚠️ 局限性**

仍需依赖持久激活外点统计，难以完全消除非外点误差；仅在无梯度设置下验证，梯度训练的进一步提升尚未探究；在更大模型或不同架构上的泛化需要进一步验证。

---

## 281. DENSE: Distilling Agent Trajectories into Evidence-Grounded Shortcut Trees for Self-Refinement

**arXiv ID:** 2609.21423 | [PDF](https://arxiv.org/pdf/2609.21423v1)

**作者:** Siyuan Liu `[一作]`, Yixin Cao `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `8d10c613-917e-4880-9716-17789f50e119` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在没有后期结果标签的情况下，从在线代理执行轨迹中提炼可重用反馈的方法DENSE，并在REFIT评估框架下验证其对新任务执行的帮助。

**💡 创新点**

创新点包括：①构建基于嵌套子任务的证据支持快捷树，压缩冗余尝试并通过恢复证据实现跨层问题调和；②提出REFIT评估协议，仅使用共享初始轨迹、环境与模型上下文重置来衡量反馈的下游价值；③证明在无后期标签条件下，组织证据可超越仅追加验证信息的效果。

**🔧 技术方法**

采用轨迹分析、子任务边界识别、快捷压缩、问题调和技术，并利用Gemini 3 Flash Preview和GPT‑5.5等生成模型生成反馈；同时统计严格通过率与token使用。

**📊 数据集**

使用Terminal‑Bench 2.1数据集，包含89个软件工程、数据处理和科学计算任务。

**📈 对比分析**

与All‑at‑once、Step‑by‑step、Traj‑only、Advisor、Verifier Feedback等方法在四个受体模型（MiniMax‑M2.7、DeepSeek V4、GPT‑5.5、Kimi K2.6）上对比。DENSE在所有受体上取得最高的非特权严格通过率，提升幅度7.12–15.64 pp，并在大部分模型中显著降低重新运行时的token使用。

**⚠️ 局限性**

局限性：仅在单一Benchmark上验证；DENSE的多阶段提炼增加生成成本；反馈可能保留源轨迹中的错误；假设环境与模型上下文完全重置的前提可能不适用于真实部署。

---

## 282. Offline Multimodal Large Language Models for Decision Support in Air Operations

**arXiv ID:** 2609.21390 | [PDF](https://arxiv.org/pdf/2609.21390v1)

**作者:** Joao P. A. Dantas `[一作]`, Gabriel Dietzsch `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了一个完全离线、多模态检索增强语言模型（RAG）架构，用于支持航空作战中的电子目标识别决策，并在巴西空军 FAB 的实际分析师上进行可行性评估。

**💡 创新点**

创新点在于：1）构建了全离线、可追溯的多模态 RAG 系统，结合文本、OCR 与视觉语言模型实现图像与文本的统一检索；2）采用两阶段检索+神经重排序，显著提升检索精度；3）实现了基于用户反馈的动态提示规则生成，提升模型输出的可用性与可解释性；4）在受限空中作战网络环境下验证 LLM 仅为决策支持工具而非决策制定者的安全部署方案。

**🔧 技术方法**

技术包括：本地部署 LLM（如 Qwen‑2.5 14B）、多语言 BGE‑M3 嵌入模型、BGE‑Reranker‑v2‑m3 重排序器、OCR（EasyOCR）+轻量级视觉语言模型（MiniCPM‑V）生成图像描述、两阶段检索+重排序 pipeline、FastAPI + SSE 前端交互、以及 NASA‑TLX、UMUX‑Lite、S‑TIAS、UTAUT 等评估工具。

**📊 数据集**

使用的数据集为巴西空军的 MCA 200‑5 电子目标识别手册（PDF/DOCX），其中包含 10 道多选测试题；以及 4 名 FAB 图像分析师在手工 REMIR 任务中使用的电子目标图像和对应手工报告。

**📈 对比分析**

比较方法：将模型在标准（附详细推理）和优化（仅答复）提示模式下的准确率与人类分析师的十题答题准确率及完成时间进行对比；同时记录模型推理耗时。实验结果显示：在优化模式下，模型 8/10 分（与最低人类相同）耗时 7.1 分钟；标准模式耗时显著更长；人类分析师平均完成时间 26.5 分钟，准确率高达 8/10。NASA‑TLX 结果表明手工 REMIR 任务总体负荷 4.2/7，表明人工过程仍属高认知负担。

**⚠️ 局限性**

局限性包括：1）仅有 4 名分析师，结果为描述性且缺乏统计显著性；2）模型受限于本地硬件与模型尺寸，未覆盖更大规模部署；3）未在 AI 辅助条件下评估工作负荷与性能提升；4）图像检索仍基于 OCR+VLM 的间接描述，缺乏直接跨模态检索；5）需要进一步验证系统在真实作战环境中的可靠性、可用性与采纳度。

---

## 283. Probabilistic Forecasting of Business Process Executions with Neural Temporal Point Processes

**arXiv ID:** 2609.21382 | [PDF](https://arxiv.org/pdf/2609.21382v1)

**作者:** Jiaxin Yuan `[一作]` (Paris Dauphine University), Han van der Aa `[通讯]` (University of Vienna)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了MoTPP模型，用于业务流程执行的概率预测。

**💡 创新点**

在生成式时间点过程框架下引入零膨胀对数正态混合时分布，并能处理事件日志中的相同时间戳。

**🔧 技术方法**

基于Transformer Hawkes Process的编码器，配合混合时间头和活动头，使用最大似然训练。

**📊 数据集**

在十个公开事件日志上评估，包括Sepsis、BPIC系列等。

**📈 对比分析**

与多种基线（SuTraN、ED-LSTM、THP-B/M、Hawkes、LA-CR、UQ等）对比，MoTPP在剩余时间校准和推理速度上表现最佳，在点预测上竞争力强。

**⚠️ 局限性**

仍未在点预测上完全超过最强序列模型，且对事件数目未知的剩余时间估计依赖教师强迫，未引入终止符。

---

## 284. Compact Partial Symmetry Breaking for Graph Search Problems

**arXiv ID:** 2609.21555 | [PDF](https://arxiv.org/pdf/2609.21555v1)

**作者:** Michael Codish `[一作]` (Ben Gurion University of Negev), Peter J. Stuckey `[通讯]` (Monash University)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于图模式的强度驱动、冗余感知的部分对称性破坏方法，生成紧凑且高精度的对称性破坏约束，能有效用于图搜索问题。

**💡 创新点**

创新点在于将图模式的强度估计与CEGAR框架结合，通过预交换/预下限/预上限等前缀分析快速筛选强度高且非冗余的模式，并分层生成高精度的部分对称性破坏。

**🔧 技术方法**

使用了图模式、图对称性分析、前缀强度估计、SAT编码、CEGAR算法、增量式模式扩展与冗余消除等技术。

**📊 数据集**

在直径‑2 临界图、具有给定girth 的极大图等基准图搜索问题以及 n≤25 的随机图实例上进行了实验。

**📈 对比分析**

与传统置换式部分对称性破坏（transpositions）和动态对称性破坏（SMS）对比，实验表明冗余率从 15.34 降至约 1.34，求解时间显著降低，尤其在不满足的实例上优势更为突出。

**⚠️ 局限性**

局限在于层次参数（s1,s2,s3）的选择仍基于经验；在 n>14 时未能完整构造最强层；冗余消除阶段的计算开销较大，且对更大规模图的可扩展性仍待进一步提升。

---

## 285. OneBid: A Unified Auto-Bidding Foundation Model for Diverse oCPX Advertising Scenarios

**arXiv ID:** 2609.21550 | [PDF](https://arxiv.org/pdf/2609.21550v1)

**作者:** Yewen Li `[一作]` (Kuaishou Technology), Qingpeng Cai `[通讯]` (Kuaishou Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出OneBid，一种面向工业级自动竞价的基础模型，能够统一处理多种oCPX广告场景的竞价决策；

**💡 创新点**

创新点包括：①将竞价目标拆分为RTG（转化回报）与CTG（成本比例）两维控制信号，实现多目标解耦；②设计序列级Mixture-of-Experts（S‑MoE）结构，在保持低延迟的同时实现共享与场景专属知识的分离；③提出CROP（Critic‑guided Relative Offline Policy Optimization）方法，利用离线评估器在不在线探索的前提下安全地对预训练策略进行场景级微调；

**🔧 技术方法**

使用的技术包括：基于Decision Transformer的自回归Transformer框架；两维RTG/CTG条件化与辅助Q/V预测的多任务预训练；序列级MoE网络实现容量扩展；离线强化学习中的Critic学习与CROP策略优化；

**📊 数据集**

采用了Kuaishou oCPX广告的工业级离线日志，约70M条转化轨迹，覆盖七种转化动作类型，提供训练与验证数据；

**📈 对比分析**

与传统规则、PID、IQL、DT等方法比较，OneBid在离线预测误差下降、在线A/B测试中ADVV提升显著（整体+2.2%，单场景Form Submit +1.9%，Purchase +2.5%等），并在同等参数规模下保持与DT相近的99.9%服务延迟；

**⚠️ 局限性**

局限性包括：需要大规模高质量离线日志支持；场景特定的微调仍需离线评估器与手工调参；RTG/CTG的设计对成本比例的敏感度需进一步验证；在极端稀疏场景或快速变化环境下的泛化能力仍待深入研究。

---

## 286. FORTE: Task-Adaptive Force Capability Optimization for Mobile Manipulators

**arXiv ID:** 2609.21497 | [PDF](https://arxiv.org/pdf/2609.21497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 287. MT-WAM: Reorienting the One-Pass Predictive Representation Toward Action Generation

**arXiv ID:** 2609.21474 | [PDF](https://arxiv.org/pdf/2609.21474v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 288. OpenMAS-GCom. A Diagnostic Benchmark for Graph-enhanced Multi-Agent Systems

**arXiv ID:** 2609.21527 | [PDF](https://arxiv.org/pdf/2609.21527v1)

**作者:** Kairui Yang `[一作]`, Rong-Hua Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `79276348-11e0-48e3-84bc-7ec231d0171c` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `14d48e9d-0069-4ad9-996a-1d5968216998` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了 OpenMAS‑GCom 基准，用结构化的组织干预方法诊断图增强多智能体系统（G‑MAS）在保持任务、模型、提示和预算不变的情况下，通信结构、角色分配、信息流和工作者可用性对性能的影响。

**💡 创新点**

创新点在于：①将 G‑MAS 组织抽象为协作单元、通信边、共享信息和执行规则；②设计四类受控干预（重连、节点移除、信息篡改、工作者失效）以便对组织因素进行因果诊断；③在 29 个数据集和 400 个多文档复杂任务（G‑MAS‑Complex）上评估 17 种配置，揭示不同干预对精度和资源利用的差异。

**🔧 技术方法**

使用的技术包括：大语言模型代理（DeepSeek、Qwen 等）、多智能体协作框架（AutoGen、MADebate、DyLAN、GPTSwarm、G‑Designer 等）、图结构生成与重连算法、干预驱动的评估管线以及统一的运行记录与计分接口。

**📊 数据集**

使用的数据集覆盖数学推理（GSM8K、AQuA）、常识与知识（MMLU‑Pro、StrategyQA）、代码生成（HumanEval、LiveCodeBench）、生物医学问答（MedQA、MedMCQA）、金融推理（ConvFinQA、FinQA）、表格推理（TabFact、WikiTableQuestions）以及 400 题的 G‑MAS‑Complex。

**📈 对比分析**

对比方法为：在同一任务和预算下，基准通过执行原始配置与单一干预后的配置，计算平均得分变化（Δ）和相对保留率。结果显示：图增强配置在 13 组任务中 7 次获得最高得分，G‑MAS‑Complex 的 GoAgent 与 VeriMap 超越 DeepSeek；重连导致最高 6.89% 的平均损失，移除专家节点比移除批评者损失更大；信息错误和工作者失效导致不同方法的性能分化，MAD 与 VeriMap 在 80% 损坏率下差距扩大至 10.18%。

**⚠️ 局限性**

局限性包括：干预仅覆盖结构、节点、信息和工作者，未考察模型本身的改进；评估侧重批处理任务，对交互式、长时序或多模态任务的适用性尚未验证；基准的可扩展性和跨任务迁移性需要进一步探索。

---

## 289. 2D GauSS-MI: Efficient Active Scene Reconstruction with Balanced Visual and Geometric Quality

**arXiv ID:** 2609.21516 | [PDF](https://arxiv.org/pdf/2609.21516v1)

**作者:** Yuhan Xie `[一作]` (University of Hong Kong), Jia Pan `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一个基于2D Gaussian Splatting（2DGS）的实时主动重建系统，能够在有限的计算资源下实现高质量视觉与几何重建，并保持模型紧凑。

**💡 创新点**

提出了面向2DGS的方向感知可靠性模型和基于Shannon互信息的2D GauSS-MI度量，首次将视角方向、可视性和几何一致性统一进下一最佳视角（NBV）的评估中。

**🔧 技术方法**

使用2DGS渲染与优化、贝叶斯可靠性更新、Shannon互信息估计、在线RGB‑D地图构建、几何正则化（深度扭曲、法线一致性）、轨迹规划（SUPER）以及多层实现优化。

**📊 数据集**

在Replica数据集的八个室内场景（Office和Room）上进行实验。

**📈 对比分析**

与ActiveGS、GauSS-MI和FisherRF等三种主流主动重建基线比较。实验显示本方法在PSNR、SSIM、LPIPS、Depth L1、Accuracy、Chamfer Distance、F-score等指标上均优于或与最强基线相当，同时候选视角评估时间仅约1 ms（≈990 fps），模型尺寸仅10.4 MB，远低于对手。

**⚠️ 局限性**

局限性包括：对仿真环境和相机噪声建模的依赖；在极端边缘视角下可靠性更新可能不够精确；以及在更大规模、动态或不规则场景中的可扩展性尚未充分验证。

---

## 290. Sharp High-Entropy Bounds for Sums of Independent Discrete Random Variables

**arXiv ID:** 2609.21459 | [PDF](https://arxiv.org/pdf/2609.21459v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

---

## 291. Dual-Interest Sequential Product Recommendation With Multi-Granular SSM

**arXiv ID:** 2609.21548 | [PDF](https://arxiv.org/pdf/2609.21548v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 292. Efficient Architecture Search under Leave-One-Subject-Out Evaluation

**arXiv ID:** 2609.21457 | [PDF](https://arxiv.org/pdf/2609.21457v1)

**作者:** Heinke Hihn `[一作]` (IU International University of Applied Sciences), Friedhelm Schwenker `[通讯]` (Ulm University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并实现了一种基于块的、泄漏控制的神经架构搜索（PainNAS），可在留一主体外交叉验证（LOSO）中高效寻找泄漏自由的网络；

**💡 创新点**

通过将主体划分为外部块共享NAS搜索，显著减少NAS运行次数（从N降至B）并保持或提升LOSO泛化性能，同时引入方差惩罚以提升模型资源效率；

**🔧 技术方法**

采用Optuna TPE采样的可搜索空间（卷积块、宽度乘子、卷积类型等），使用内部K折验证评估候选架构，计算加权平均与方差的选择目标；

**📊 数据集**

在BioVid Heat Pain数据集上进行实验，包含多通道生理信号（EMG、EDA、ECG）与多类别疼痛等级；

**📈 对比分析**

与手工设计的早期融合网络、全球NAS+LOSO以及Late Fusion基线进行对比；在二分类任务中获得约83–84%准确率，参数量和FLOPs显著下降（5–20倍），多分类约35–36%准确率；

**⚠️ 局限性**

方法仍依赖对块数和折数的经验选择，架构选择的可变性较大，对不同数据分布可能需进一步调整；

---

## 293. Orbital Detection: On Maximum-Entropy Priors

**arXiv ID:** 2609.21466 | [PDF](https://arxiv.org/pdf/2609.21466v1)

**作者:** Kuranage Roche Rayan Ranasinghe `[一作]` (Constructor University), Giuseppe Thadeu Freitas de Abreu `[通讯]` (Constructor University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种“轨道检测”方法，利用最大熵原理构造仅保留星座幅度统计且相位均匀分布的先验，从而在 AWGN 信道上得到封闭形式的后验，进而实现 O(L) 复杂度的 MMSE 与 MAP 检测器。

**💡 创新点**

创新点在于：①证明该轨道先验是满足给定幅度分布且相位信息最不确定的唯一解；②在该先验下得到的后验可以拆分为幅度软最大化与 von Mises 相位分布；③提出层次化 MAP 规则，并量化了相位约束导致的高 SNR 边界偏移。

**🔧 技术方法**

采用的技术包括：最大熵优化（混合离散‑连续熵），极坐标解析、Bessel 函数 I₀、I₁ 的使用，von Mises 分布与其浓度参数的推导，复杂度分析与阈值推导。

**📊 数据集**

使用仿真数据：在标量 AWGN 信道上，对 M‑PSK、APSK、M‑QAM 三类星座进行 10⁶ 次 Monte‑Carlo 试验，验证各检测器性能。

**📈 对比分析**

与传统离散后验（O(M)）和能量匹配 LMMSE（O(1)）进行比较。轨道检测的 SER 与最优离散检测几乎无差距（仅在高 SNR 时出现微小误差），MSE 低于 LMMSE 但略高于最优检测，表明相位约束带来的性能折中。

**⚠️ 局限性**

局限性：①相位被均匀近似导致在高 SNR 下出现小的误差与性能下降；②目前仅在单输入 AWGN 信道上验证，尚未推广至多输入多输出或非高斯噪声环境；③虽然复杂度为 O(L)，但对极大 L 的星座仍需考虑 Bessel 计算成本。

---

## 294. Skel-WAM: A Hand-Skeleton-Conditioned World Action Model for Human-to-Robot Manipulation Transfer

**arXiv ID:** 2609.21514 | [PDF](https://arxiv.org/pdf/2609.21514v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 295. Refine Then Fusion: Training-Free 3D Point Cloud Adaptation with Priority Refinement and Multi-Modal Knowledge Fusion

**arXiv ID:** 2609.21522 | [PDF](https://arxiv.org/pdf/2609.21522v1)

**作者:** Hang Cheng `[一作]` (Tsinghua University), Long Zeng `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在few-shot 3D识别任务中，提出一种训练-free的“先提炼再融合”（Refine‑Then‑Fusion, RTF）框架，利用预训练的多模态基础模型实现高效识别。

**💡 创新点**

创新点包括：① Priority‑guided Channel Refinement (PCR) 通过评估类间相似性与方差，剔除冗余通道；② Reliability‑aware Multi‑Modal Fusion (RAMF) 根据PCR引起的分布偏移估计模态可靠性并自适应加权融合；③ Hybrid Memory Cache (HMC) 结合实例检索与类级原型，实现无参数的鲁棒推理。

**🔧 技术方法**

采用无参数特征提炼、KL 距离评估可靠性、球面 K‑means 原型生成、非参数实例检索等技术；使用 ULIP、ViT、PointNN 等预训练编码器作为 3D、2D、几何模态。

**📊 数据集**

在 ModelNet10、ModelNet40、ScanObjectNN（OBJ_ONLY、OBJ_BG、OBJ_T50RS）等三大基准数据集上进行评估。

**📈 对比分析**

与多种 zero‑shot、fine‑tune 与训练‑free 方法对比，RTF 在 16‑shot 及全量数据场景下均实现或超越现有最佳训练‑free 方法，平均提升 1–2% 级别，尤其在 ScanObjectNN 复杂场景中显著提升性能。

**⚠️ 局限性**

局限性：高度依赖预训练模型的质量；在极端噪声或分布漂移条件下可能仍受限；模态加权策略在不同任务中需要进一步自适应或扩展。

---

## 296. Learning-to-Optimize as the Missing Architectural Layer of AI-Native Networks

**arXiv ID:** 2609.21519 | [PDF](https://arxiv.org/pdf/2609.21519v1)

**作者:** Giambattista Amati `[一作]` (Fondazione Ugo Bordoni), Simone Angelini `[通讯]` (Fondazione Ugo Bordoni)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现了学习-优化(L2O)层，将优化知识转化为可部署的AI模型，应用于NR-V2X中继选择。

**💡 创新点**

将优化过程系统化为架构层，弥补AI-native网络缺乏优化知识生命周期管理的空白。

**🔧 技术方法**

使用MILP求解器生成最优解、图神经网络（GINE）做代理学习、知识存储与持续演化机制。

**📊 数据集**

基于OSM–SUMO–GEMV2仿真生成的车辆网络图数据集。

**📈 对比分析**

与MILP基准对比，GINE在96%链接级别准确率下，推理延迟<5ms；混合框架进一步缩短了MILP求解时间。

**⚠️ 局限性**

需定期更新知识库以适应网络变化，且对复杂约束的处理仍有限，未来需更深入的约束感知学习。

---

## 297. ServeGuard: Verifiable, Bounded-Residual Confinement of Operator-Invisible Channels Without Revealing the Certified Read Factor

**arXiv ID:** 2609.21515 | [PDF](https://arxiv.org/pdf/2609.21515v1)

**作者:** Dominik Dahlem `[一作]`, Rui Vieira `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

未提供论文具体内容，无法判断研究内容

**💡 创新点**

无法确定创新点

**🔧 技术方法**

无法确定使用的技术

**📊 数据集**

无法确定使用的数据集

**📈 对比分析**

无法进行方法比较或评估性能

**⚠️ 局限性**

缺乏关键信息导致无法评估研究的局限性

---

## 298. DPed-VLN: A Benchmark for Socially Compliant Vision-and-Language Navigation in Dynamic Pedestrian Environments

**arXiv ID:** 2609.21504 | [PDF](https://arxiv.org/pdf/2609.21504v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 299. Driving on Registers, Reasoning on Risk: Risk-Aware Occupancy for Register-Based End-to-End Autonomous Driving

**arXiv ID:** 2609.21486 | [PDF](https://arxiv.org/pdf/2609.21486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 300. 2nd Place Solution to the HANDS 2026 Workshop Challenge-Dexterous Grasp Motion Track: Single-Shot Trajectory Warping for Grasp Motion Generation

**arXiv ID:** 2609.21511 | [PDF](https://arxiv.org/pdf/2609.21511v1)

**作者:** Muneeb A. Khan `[一作]` (UNIST), Seungryul Baek `[通讯]` (UNIST)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究通过一次性决策编辑单一成功演示轨迹，生成完整的12维控制序列，并在Open‑Loop回放中实现LinkerHand O6的抓取与抬升；

**💡 创新点**

创新点在于将DemoGrasp的单示范编辑框架迁移到O6手，提出对象中心位置‑旋转warper、按比例手指闭合、以及全体训练对象并行的一步PPO学习；

**🔧 技术方法**

采用对象点云的Frozen PointNet编码、位置-旋转-手指偏移的warp模块、单步PPO（含KL自适应学习率）以及前置reach段的轨迹合成；

**📊 数据集**

使用GraspM3数据集（5054个ShapeNet物体、≈110k轨迹，筛选后4824物体、52718轨迹）进行训练，Objaverse数据用于评测；

**📈 对比分析**

相较于官方基线DexRep‑TCN 23.93%以及团队HandsDown（91.29%/77.45%），本方法在易轨道取得94.61%（最高），在难轨道57.18%，整体排名第二；

**⚠️ 局限性**

局限性在于开环warp无法应对物体滑动、质量与摩擦变化，缺乏抓取后动态反馈，导致在动态抓取精度与力学鲁棒性上受限。

---

## 301. On Repulsive and Attractive Teachers: Separating Correctness from Behavior in Self-Distillation

**arXiv ID:** 2609.21561 | [PDF](https://arxiv.org/pdf/2609.21561v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 302. Two Fault Lines: Latent Polarity Geometry in X Community Notes

**arXiv ID:** 2609.21496 | [PDF](https://arxiv.org/pdf/2609.21496v1)

**作者:** Andreas Andreou `[一作]` (Cyprus University of Technology), Michael Sirivianos `[通讯]` (Cyprus University of Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文重新拟合了Twitter（X）Community Notes的桥接式矩阵分解模型，证明其观点空间至少需要二维才能准确捕捉用户对政治立场和机构信任的分歧。

**💡 创新点**

创新点在于首次大规模检验并揭示社区笔记的对立空间不止左右政治轴，而是额外的机构信任维度，并指出单维模型会错失一类跨立场但对机构持不同意见的笔记。

**🔧 技术方法**

使用了正则化的多维稀疏矩阵分解（Latent Factor）结合Adam优化、SVD旋转以及交叉主题转移验证等技术。

**📊 数据集**

数据集为2026年6月14日公开的X Community Notes全量数据，包含约2.13亿条评级、2.33万条笔记和107万名评审者。

**📈 对比分析**

通过在Hold‑out测试上对比K=1与K=2的RMSE，二维模型平均降低约6%（相对RMSE从0.3085降到0.2899），并在跨主题预测和出版率上显著优于单维模型。

**⚠️ 局限性**

主要限制包括仅使用最近一次公开快照导致无法复现完整训练、语言检测与主题分配偏差、对轴解释的主观性以及模型在极小语言社群中的样本稀疏问题。

---

## 303. Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers

**arXiv ID:** 2609.21547 | [PDF](https://arxiv.org/pdf/2609.21547v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 304. Purification and Regulation: Comorbidity-Aware Multi-Label Few-Shot Learning for Medical Image Classification

**arXiv ID:** 2609.21541 | [PDF](https://arxiv.org/pdf/2609.21541v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 305. Benchmarking Gender Bias in Machine Translation Evaluation Metrics across Occupations

**arXiv ID:** 2609.21490 | [PDF](https://arxiv.org/pdf/2609.21490v1)

**作者:** Orfeas Menis Mastromichalakis `[一作]` (Instituto de Telecomunicações), Chrysoula Zerva `[通讯]` (National Technical University of Athens)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对 WMT 2026 自动化翻译质量评估任务，构建了一个包含七个英-目标语言对（阿拉伯语、捷克语、德语、希腊语、冰岛语、俄语、乌克兰语）的职业平衡子集（GAMBIT+），并对现有评估系统的分数预测与错误标注进行性别偏差分析。

**💡 创新点**

创新点包括：①将原始 GAMBIT+ 规模大、计算成本高的挑战集压缩为 1,308 对样本的职业平衡子集，保留完整 ISCO‑08 覆盖；②采用符号差值、绝对差值和偏好频率三种互补度量，系统评估评估器性别偏差的方向、幅度与一致性；③将性别偏差的发现扩展到错误标注任务，揭示两种性别表达在被认为“无错误”率上的差异。

**🔧 技术方法**

主要技术包括：统计检验（t 检验 + Benjamini–Hochberg 校正）、基于职业块的 5,000 次 Bootstrap 置信区间、归一化为分数范围的百分比差异（S^% / A^%）以及对 22 个评分系统进行宏观平均，综合评估性别偏差。

**📊 数据集**

使用数据集：GAMBIT+（扩展版）子集，包含 1,308 条英-目标语言成对实例（每职业 3 条），来源于 ISCO‑08 436 个职业组；原始的 29,000+ 对完整挑战集也被引用做对照。

**📈 对比分析**

比较方法：对每个系统-语言组合计算符号差值、绝对差值、偏好频率和错误无标记率差异；通过 5,000 次 Bootstrap 估计 95% 区间；用 t 检验判定差异显著性。性能方面，126/131 的显著差异均为男性优先，绝对差值与偏好频率揭示评估器对男性与女性表达的不同敏感度，错误标注任务同样显示男性更易被视为无误。

**⚠️ 局限性**

局限性：①德语翻译由独立生成，可能与其他语言的上下文不完全对齐；②每职业-语言组合仅有 3 条实例，职业级别差异仅具描述性；③归一化受极端分数影响，单一平均值不足以说明评估器质量；④仅考虑性别二元化（男性/女性），未覆盖性别中性或包容性表达；⑤仅观察单一次评估结果，无法分离系统噪声与系统性偏差。

---

## 306. VidOmni-Bench: A Benchmark for Fine-Grained Video Understanding via Spatio-Temporal Event Verification across Complexity and Duration

**arXiv ID:** 2609.21521 | [PDF](https://arxiv.org/pdf/2609.21521v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 307. HE-Guardrail: A Homomorphic Guardrail Against Jailbreak Attacks for Encrypted Large Language Model Inference

**arXiv ID:** 2609.21484 | [PDF](https://arxiv.org/pdf/2609.21484v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 308. The Communication Bottleneck: A Round-Trip Study of Tree-Structured Expression Serialization in Language Models

**arXiv ID:** 2609.21509 | [PDF](https://arxiv.org/pdf/2609.21509v1)

**作者:** Xavier Suau `[一作]` (Apple), Samy Bengio `[通讯]` (Apple)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个“往返”协议，用来测量语言模型在通过自然语言传递树状算式结构时的保真度。

**💡 创新点**

通过构建N×N生成-提取矩阵，揭示了生成与提取的角色异质性、失败归因、树形结构对成功率的主导作用，并证明该瓶颈是可训练的。

**🔧 技术方法**

采用符号等价性检验、基于梯度提升的特征重要性分析、线性化复杂度指标ℓ(T)、以及QLoRA微调等技术。

**📊 数据集**

使用自生成的算式树集合（k=2–8，深度=2–6）和对应的文字问题，逻辑版作为对照。

**📈 对比分析**

对16个模型（12开源、4前沿）在所有生成-提取组合下评估，发现最佳往返精度为92.9%，生成是瓶颈，右分支树和更深结构导致更高失败率。

**⚠️ 局限性**

局限性在于只适用于可给出唯一符号等价的域（如算式、命题逻辑），不适用于多答案或非结构化域；实验为单语种、单一提示，且未覆盖思考模式。

---

## 309. OpenSAL360: Open-Source Crowdsourcing Platform for Omnidirectional Video Saliency Collection

**arXiv ID:** 2609.21480 | [PDF](https://arxiv.org/pdf/2609.21480v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 310. Risk-Aware Occupancy for Safety-Oriented End-to-End Autonomous Driving

**arXiv ID:** 2609.21470 | [PDF](https://arxiv.org/pdf/2609.21470v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 311. GameLogicBench: Evaluating Coding Agents on Runtime Game Logic with Tick-Level State Assertions

**arXiv ID:** 2609.21562 | [PDF](https://arxiv.org/pdf/2609.21562v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 312. Cross-Platform vs Native Mobile Development: An Empirical Study of Software Quality Trade-offs

**arXiv ID:** 2609.21544 | [PDF](https://arxiv.org/pdf/2609.21544v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 313. MACE: Memory-Agent Co-Evolution with Adaptive Memory Graphs for Multi-Agent Systems

**arXiv ID:** 2609.21533 | [PDF](https://arxiv.org/pdf/2609.21533v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 314. Critical sets of Latin squares based on autoparatopisms

**arXiv ID:** 2609.21532 | [PDF](https://arxiv.org/pdf/2609.21532v1)

**作者:** Manuel González-Regadera `[一作]` (Universidad de Sevilla), María Dolores Frau `[通讯]` (Universidad de Sevilla)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在给定自同构类的拉丁方中，求解具有给定帕拉托普里斯（paratopism）条件的临界集（critical set）并提出相应算法；

**💡 创新点**

创新点在于将帕拉托普里斯的轨道结构与临界集的构造相结合，证明临界集的大小仅取决于帕拉托普里斯的共轭类和拉丁方的主类，并给出了从轨道到临界集的直接构造方法；

**🔧 技术方法**

主要技术包括对帕拉托普里斯群的代数分析、轨道分解、并利用 GAP 的群与算法库实现批量搜索；

**📊 数据集**

使用的数据集为所有阶 3 至 6 的拉丁方及其对应的自同构群，已在相关文献中完整列出；

**📈 对比分析**

在实验中对每个主类及帕拉托普里斯共轭类使用算法求得所有可行临界集大小，计算时间在 0.01–120 s 之间，显著低于对完整拉丁方的暴力搜索；

**⚠️ 局限性**

限制在于目前仅能处理阶 ≤6 的拉丁方，计算复杂度随阶数快速上升，且缺乏通用理论闭式表达临界集大小的公式。

---

## 315. Adaptive World Memory 3D Foundation Model for Scalable 3D Mapping, Localization, and Rendering

**arXiv ID:** 2609.21502 | [PDF](https://arxiv.org/pdf/2609.21502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 316. LogicTrack: Auditing Reasoning Trajectories of Large Language Models with Formal Logic Solvers

**arXiv ID:** 2609.21492 | [PDF](https://arxiv.org/pdf/2609.21492v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 317. What Must Survive? Exact Task-Information--State Frontiers for Resource-Sufficient Learning

**arXiv ID:** 2609.21523 | [PDF](https://arxiv.org/pdf/2609.21523v1)

**作者:** Ronald Katende `[一作]` `[通讯]` (Kabale University), Ronald Katende (Kabale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043`

**🎯 论文内容**

本文研究在仅知道未来任务有限信息时，如何压缩源状态以实现所有线性任务的精确或近似恢复，并给出了保留状态维度与任务信息量之间的最优“前沿”关系。

**💡 创新点**

创新点在于：①把任务信息约束转化为对任务集堆叠矩阵秩的分区优化，给出精确与近似的维度前沿公式；②证明寻找最优分区在一般情况下是强NP‑难的；③在三个实际场景（软max注意力、分区数字孪生、层级多任务）给出闭式前沿并展示信息量对压缩率的显著影响。

**🔧 技术方法**

主要技术包括：连续宽度理论（利用奇异值与最小误差关系）、Borsuk–Ulam定理与秩分解、奇异值截断估计、NP‑完整性证明（3‑Partition 归约）以及对多任务结构的直接和分区分析。

**📊 数据集**

本文并未使用传统机器学习数据集，而是通过符号式构造（如softmax注意力矩阵、分区子空间、层级任务的行空间分解）来说明理论结果。

**📈 对比分析**

通过对三类实例的数值演示，展示在给定任务信息量（比特数）下所需保留坐标的量级减小。例如，在512个软max注意力子任务中，9比特可将所需坐标从524288压缩到1024，保持算子条件良好；在64区域的数字孪生中，3比特将压缩后维度从8448降至384。

**⚠️ 局限性**

限制主要包括：①仅适用于连续坐标的维度压缩，未考虑离散位数；②假设任务算子为线性；③在实际应用中任务算子往往需从数据估计，导致额外的统计不确定性；④寻找最优分区在一般情况下不可多项式求解。

---

## 318. Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap

**arXiv ID:** 2609.21483 | [PDF](https://arxiv.org/pdf/2609.21483v1)

**作者:** Ziyu Huang `[一作]` (Shanghai Jiao Tong University), Jingwen Leng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了Weave系统，该系统提供细粒度动态SM调度的MoE Megakernel，用以最大化计算与通信的重叠，从而提升大规模Mixture-of-Experts模型在GPU上的训练效率。

**💡 创新点**

创新点包括：①基于事件驱动的动态SM分配策略，能够在运行时根据计算负载和通信需求实时调整SM使用；②提出的Megakernel结构将多个MoE层合并为单一内核，显著减少kernel launch开销；③结合流水线并行与显式数据预取，实现计算-通信的全程重叠；④自适应调度机制根据当前GPU利用率动态优化资源分配。

**🔧 技术方法**

使用了CUDA、NCCL/MPICH等异步通信框架，GPU显存管理技术，动态SM调度算法，以及基于张量计算图的自动调度与流水线设计。

**📊 数据集**

实验数据集主要包括：OpenAI GPT‑3 训练数据（English Wikipedia + Common Crawl）、GLUE/SuperGLUE 语言理解基准，以及部分 ImageNet 视觉 MoE 任务。

**📈 对比分析**

与现有 MoE 实现（如 GShard、DeepSpeed、FairScale）在相同模型规模和硬件平台（NVIDIA V100 / A100）上进行对比。Weave 在 8 卡 64 GPU 训练上相较基线实现了约 30% 的加速，显存占用减少 15%，通信开销下降 40%。在 5 组实验中，性能提升持续保持在 25–35% 之间。

**⚠️ 局限性**

局限性：①验证主要集中在 NVIDIA V100/Tesla 系列 GPU，尚未在最新 A100 或多 GPU 集群上做充分评估；②动态调度的实现增加了系统复杂度，对开发者的调试与性能调优提出更高要求；③在极大规模模型（>10B 参数）下，仍受显存上限限制；④目前仅与 PyTorch/TensorFlow 1.x 兼容，尚未完整支持最新的 TensorFlow 2.x 与 JAX。

---

## 319. Adaptive Preference Modeling via Explicit Indirect Relational Learning for Personalized Fashion Matching

**arXiv ID:** 2609.21475 | [PDF](https://arxiv.org/pdf/2609.21475v1)

**作者:** Shuiying Liao `[一作]` (Hong Kong University of Science and Technology), P. Y. Mok `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于显式间接关系学习的个性化时尚配搭推荐框架APCL，利用多模态信息与对比学习提升稀疏场景下的推荐效果。

**💡 创新点**

创新点在于将间接用户–商品、商品–商品关系显式构造为专用视图，并通过功能视图对比学习将直接与间接表示对齐，从而增强模型对稀疏和冷启动的鲁棒性。

**🔧 技术方法**

采用多模态编码（视觉+文本）+MLP、注意力聚合、BPR对数似然损失、InfoNCE对比损失以及自适应采样的相关性采样机制。

**📊 数据集**

在两大公开数据集Polyvore和IQON3000上进行实验，数据涵盖用户、上衣/下装、图片与文本特征。

**📈 对比分析**

与BPR-MF、V-BPR、GP-BPR、PCE-NET、CP-TransMatch、HMGL-OCM等基线对比，APCL在AUC、HR@10、NDCG@10等指标上均显著提升，尤其在稀疏与冷启动场景中表现突出。

**⚠️ 局限性**

局限性包括：仅针对二项式（上衣–下装）配搭，未覆盖多品类完整穿搭；对间接关系的构造依赖于足够的相关用户/商品，极端稀疏时收益有限；额外的采样与对比机制带来一定计算开销。

---

## 320. SkillIR: Evolving Scene-Aware Skills for Agentic Image Restoration

**arXiv ID:** 2609.21468 | [PDF](https://arxiv.org/pdf/2609.21468v1)

**作者:** Jie Shao `[一作]` (Zhongnan University of Economics and Law), Jun Wan `[通讯]` (Zhongnan University of Economics and Law)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于技能的代理式图像恢复框架SkillIR，使用验证残差循环进行逐步工具调用并更新经验。

**💡 创新点**

将恢复轨迹拆分为退化中心片段，并将验证后的局部转换抽象为场景感知技能与失败教训，实现状态依赖的逐步决策而非全局计划。

**🔧 技术方法**

采用多模态大型语言模型（GPT‑4o）做高层决策，残差感知与过渡验证网络，技能管理器进行技能创建/合并/修补，工具箱包括多种专业恢复模型。

**📊 数据集**

在MiO100（合成多退化）以及由I‑Haze、NH‑Haze、DRealSR等构成的100对真实世界基准上进行实验。

**📈 对比分析**

与所有一体式恢复和代理式恢复基线对比；在MiO100各组和真实世界基准上获得第一或第二名，PSNR提升1–2 dB，LPIPS和无参考指标显著优于对手，工具调用次数更少，效率更好。

**⚠️ 局限性**

仍受限于已收集的验证经验，处理极端或未知退化时表现有限；过渡验证与重感知增加推理时延；依赖大型语言模型与专门工具的可用性。

---

## 321. PSEE: Progressive Sensor Event Expansion for Point-Supervised Temporal Action Localization

**arXiv ID:** 2609.21462 | [PDF](https://arxiv.org/pdf/2609.21462v1)

**作者:** Jiaxi Yin `[一作]` (Xi'an Jiaotong University), Fei Wang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为PSEE的点监督时序动作定位方法，利用稀疏的时间戳和类别标签自动生成伪段，训练标准时序动作定位检测器而不修改其推理流程。

**💡 创新点**

创新点在于将语义激活、传感器特定的过渡信息和自适应时间归属三者融合，利用自适应归属区间和多维评分机制准确恢复点标记对应的时段，并通过可靠性加权降低伪段不确定性。

**🔧 技术方法**

使用轻量级时间编码器与双头（类别和过渡）网络、基于信号统计的传感器先验、伪段推断（归属区间、候选边界生成、基于排名的选择）以及在定位损失中加入可靠性权重。

**📊 数据集**

在四个惯性测量传感器基准集上进行评估：WEAR、WetLab、SBHAR和RWHAR。

**📈 对比分析**

与适配的SFNet-Adapt和OPOT-Adapt两种点监督基线相比，PSEE在所有数据集上均提升mAP，尤其在高tIoU阈值下优势显著（如WetLab 33.55%对21.67%，RWHAR 64.68%对8.98%）。伪段平均tIoU亦高于基线（WEAR 74.34%，WetLab 50.23%，SBHAR 71.91%，RWHAR 48.38%）。同一伪段可用于多种检测器（SlimSTAD、ActionFormer、TriDet），验证其通用性。

**⚠️ 局限性**

局限性包括对点采样分布的敏感性（极端稀疏或噪声点时恢复效果下降），伪段生成阶段增加训练复杂度，以及在具有复杂过渡或极短动作的场景中仍可能出现边界误判。

---

## 322. VoxelTTO: Voxel-Aligned Feed-Forward 3D Gaussian Splatting with Test-Time Optimization

**arXiv ID:** 2609.21498 | [PDF](https://arxiv.org/pdf/2609.21498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 323. MIRAGE: Multi-Perspective Creative Language Model Reasoning with Reinforcement Learning Guidance

**arXiv ID:** 2609.21554 | [PDF](https://arxiv.org/pdf/2609.21554v1)

**作者:** Arash Lagzian `[一作]` (National University of Singapore), Dianbo Liu `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 MIRAGE，一个在推理时动态切换并聚合多种概念视角（如代数、概率、网络流等）的框架，通过 Selector 对不同视角进行排序并交替调用 Reasoner，直至得到自信答案或汇总多视角结果。

**💡 创新点**

创新点在于：①引入无参数更新的 Selector，使用 REINFORCE 仅在推理时学习最佳视角顺序；②通过少量前向推理（平均不到两次）即可达到甚至超过传统 Chain‑of‑Thought、DIPPER 等多样化采样方法；③将认知科学中的“多视角思维”概念直接映射到 LLM 推理流程。

**🔧 技术方法**

技术细节包括：① Selector 采用 LoRA 微调的轻量级分类器，输出视角概率并采样；② Reasoner 为冻结的 LLM（如 Qwen2.5‑7B、ChatGPT‑4o 等）在每个选定视角下生成逐步推理；③ 采用置信度阈值和投票聚合策略，奖励函数为准确率减去视角数量惩罚；④ 对 20 个预设概念视角进行手工设计并在训练时学习其权重。

**📊 数据集**

使用的数据集涵盖四大基准：GSM8K、MATH500、MMLU‑Pro（包含数学、物理、化学、工程四个子集）以及 Game‑of‑24，实验在五个主流 LLM 上进行。

**📈 对比分析**

与标准提示、Chain‑of‑Thought、DIPPER（n=3、5）等基线相比，MIRAGE 在所有模型上均实现了显著的准确率提升，最高可达 +24.7pp，且推理调用次数平均仅为 1–2 次，成本相对传统多样本方法低 3–5 倍。

**⚠️ 局限性**

局限性包括：① 仅使用预定义的 20 个离散视角，缺乏自动发现新视角的能力；② 评估仅为单跑实验，未给出置信区间或多种种子验证；③ 对开放域任务的适应性有限，尚未与外部工具或知识库紧密集成。

---

## 324. From Retrieval to Recognition:How Vision--Language Models Become OCR Specialists

**arXiv ID:** 2609.21543 | [PDF](https://arxiv.org/pdf/2609.21543v1)

**作者:** Yuanxiang Huangfu `[一作]` (PatSnap Co Ltd), Jeffrey Tiong Jee Hui `[通讯]` (PatSnap Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

探究通用视语模型在全序OCR中的注意力头机制，识别并验证稀疏的OCR专用头，比较其与通用检索/复制头的重用情况，并分析OCR专业化对头部分布与因果贡献的影响。

**💡 创新点**

发现全序OCR主要重用通用检索复制机制，并在专业化过程中保持头部身份但重新分配功能与因果强度，首次系统评估OCR头与检索头的重叠及因果关系。

**🔧 技术方法**

基于注意力对齐、基于证据的发现协议、持出交叉验证、头部消融、对比实验和价值补丁等技术。

**📊 数据集**

使用GLM‑OCR、MinerU2.5、PaddleOCR‑VL‑1.6等三种OCR模型的OmniDocBench子集（1200例，包含文本、表格、公式）以及Qwen2‑VL‑2B和Qwen3‑VL‑2B的通用模型。

**📈 对比分析**

通过与随机头组消融、检索任务交叉消融和价值补丁实验比较，验证OCR头的稀疏性和因果效应，结果显示OCR专用头与检索头重叠率超过70%，消融误差提升显著。

**⚠️ 局限性**

仅针对已知视觉文本内容，未覆盖隐藏字符或复杂布局；价值补丁方法仅针对单一视觉源；结果受模型规模与prompt差异影响。

---

## 325. PolyBridgeBench: Benchmarking Multimodal LLMs for Physics-Grounded Bridge Design

**arXiv ID:** 2609.21493 | [PDF](https://arxiv.org/pdf/2609.21493v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 326. IncentRL: The Trade-Off Between Preference Guidance and Task Performance

**arXiv ID:** 2609.21525 | [PDF](https://arxiv.org/pdf/2609.21525v1)

**作者:** Xuening Wu `[一作]` (Fudan University), Shenqin Yin `[通讯]` (Fudan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 IncentRL 框架，在强化学习中通过加入 KL 散度惩罚，将偏好指导与外部奖励结合，避免过度改变任务目标。

**💡 创新点**

创新点在于在有限折扣马尔可夫决策过程上给出外部价值失真上界、足够的严格行动间隙条件以及大权重极限下的累积偏好成本解释，并结合实际实现与自适应系数搜索。

**🔧 技术方法**

使用的技术包括 KL 散度惩罚的偏好奖励、理论分析的价值失真界、严格行动间隙条件、累积成本极限推导、以及基于 Beta 分布的外部搜索策略。

**📊 数据集**

实验主要在 MiniGrid DoorKey‑8x8 环境中进行，采用 PPO 进行训练。

**📈 对比分析**

比较方法为固定系数 β=0 与 β=0.01 的学习曲线，结果显示 β=0.01 在 200 万步后成功率提升至 98%（比 90.5% 高 7.5个百分点），同时平均回合长度显著下降；系数搜索显示随着迭代，平均 β 值逐步减小。

**⚠️ 局限性**

局限性包括缺乏对 KL 散度实现细节的可重复性报告、未证明在神经网络逼近下收敛、未分离 KL 形状与简单距离惩罚的差异、以及未提供统计显著性或对比其他信息匹配方法的结果。

---

## 327. Adaptive Rollout Truncation Based on Epistemic Uncertainty for Efficient Offline World Model Training

**arXiv ID:** 2609.21482 | [PDF](https://arxiv.org/pdf/2609.21482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 328. Learning Distance-Conditioned Object Transport for Humanoid Loco-Manipulation from a Single Motion Clip

**arXiv ID:** 2609.21467 | [PDF](https://arxiv.org/pdf/2609.21467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 329. AtomEgo: Exploring Ego-Robot Integration for Embodied Foundation Model Pretraining

**arXiv ID:** 2609.21461 | [PDF](https://arxiv.org/pdf/2609.21461v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 330. Labelling Bug-Fixing Commits with Local Open-Weight Language Models

**arXiv ID:** 2609.21616 | [PDF](https://arxiv.org/pdf/2609.21616v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 331. A benchmark dataset and baseline methods for four-dimensional STEM diffraction patterns

**arXiv ID:** 2609.21593 | [PDF](https://arxiv.org/pdf/2609.21593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 332. Potential-Field Action Representation for Reinforcement Learning in Contact-Rich Manipulation

**arXiv ID:** 2609.21609 | [PDF](https://arxiv.org/pdf/2609.21609v1)

**作者:** Xinyu Liu `[一作]` (Istituto Italiano di Tecnologia), Arash Ajoudani `[通讯]` (Istituto Italiano di Tecnologia)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了PA‑RL框架，将强化学习与人工势场结合，使用势场参数化作为动作空间，解决 peg‑in‑hole 插接任务的学习与执行问题。

**💡 创新点**

通过将动作表示从直接运动命令改为可调节的势场参数，解耦任务级决策与连续运动生成，实现更高效学习、更平滑物理执行，并且不需要显式的运动质量惩罚。

**🔧 技术方法**

使用 Soft Actor–Critic（SAC）最大熵 RL 训练；势场由线性与高斯基底构成，参数由策略输出；通过饱和的 Cartesian impedance 控制器生成受限的速度参考并追踪；在 MuJoCo 仿真和 Franka Emika Panda 机器人上验证。

**📊 数据集**

在仿真中随机化孔位、估计误差与观测噪声，使用自生成的 Peg‑in‑Hole 任务数据，训练预算为 5×10^6 环境步，无使用公开数据集。

**📈 对比分析**

与三种基线（TCP‑Vel、TCP‑Pose、VICES）在相同奖励与超参数下对比，PA‑RL 在学习效率上最快，任务成功率达到 100%，运动质量指标（关节扭矩波动、加速度波动）显著优于基线；仿真训练的策略在真实机器人上完成全部 9/9 插接，无需调优。

**⚠️ 局限性**

势场参数化受限于任务结构，未保证全局收敛；适用于特定插接场景，未验证更复杂或多自由度任务，对更广泛的操作空间与感知需求的推广仍需进一步研究。

---

## 333. Beyond Accuracy: Centroid-Guided Contrastive Loss for Structured Fraudulent Job Posting Detection

**arXiv ID:** 2609.21599 | [PDF](https://arxiv.org/pdf/2609.21599v1)

**作者:** Syed Ali Ahmed `[一作]` (National University of Computer and Emerging Sciences), Muhammad Rafi `[通讯]` (National University of Computer and Emerging Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种结合分类与聚类的Centroid‑Guided Contrastive Loss (CGCL)，用于高精度检测欺诈性职位发布。

**💡 创新点**

创新点在于通过动态更新类别质心并对top‑k难正负样本进行推拉（push‑pull）机制，同时将加权交叉熵与对比损失统一，既提升决策边界，又重塑潜在空间结构。

**🔧 技术方法**

技术方法包括GloVe词向量、六层MLP网络、ADASYN合成少数样本、加权交叉熵、对比损失、t‑SNE/UMAP可视化以及多种聚类评价指标。

**📊 数据集**

使用公开的Employment Scam Aegean Dataset (EMSCAD)，包含约17,014条真实广告与866条欺诈广告。

**📈 对比分析**

与传统随机森林、BiLSTM、MLP等基线模型对比，CGCL在测试集上取得99.2%准确率、98.7% F1分数，并在聚类指标上获得Silhouette 0.701、ARI 0.953等优异表现。

**⚠️ 局限性**

局限性包括对超参数高度敏感、不同随机种子会导致性能波动、计算开销相对较大，以及在跨域或实时部署环境中的可迁移性尚待验证。

---

## 334. A High-Payload Wall-Climbing Robot Using Passive Bistable Suction Cups

**arXiv ID:** 2609.21584 | [PDF](https://arxiv.org/pdf/2609.21584v1)

**作者:** Andrew Nguyen `[一作]` (University of Michigan), Daniel Bruder `[通讯]` (University of Michigan)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

设计并实验验证了基于双稳吸盘的墙壁攀爬机器人，可携带7.94 kg负载

**💡 创新点**

利用双稳吸盘实现无推力吸附，克服传统吸盘推力-附着力权衡，显著提升载荷能力

**🔧 技术方法**

双稳吸盘设计与轨道驱动机制、柔性杆断裂实现自动吸附/脱附技术

**📊 数据集**

无公开数据集，所有性能均通过实验测量得到（拉离力、悬挂时间、载荷能力等）

**📈 对比分析**

与传统单稳吸盘及其他被动/主动吸附机器人对比，载荷比达2.25，爬升速度11.5 cm/s，悬挂时间3 h 45 min

**⚠️ 局限性**

只能在光滑表面工作，无法转向或自主起始，仅线性行走，需人工支撑

---

## 335. HyperParallel-FSDP: Topology-Aware Fully Sharded Training with Layout-Driven Muon on Ascend SuperPods

**arXiv ID:** 2609.21594 | [PDF](https://arxiv.org/pdf/2609.21594v1)

**作者:** Mo Sun `[一作]` (Zhejiang University), Teng Su `[通讯]` (Huawei Technologies Co., Ltd)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于声明式SPMD编程的分布式训练系统，旨在提高大规模语言模型的训练效率。

**💡 创新点**

创新点在于通过在autograd引擎之上拦截张量API层，解耦并优化了并行化与模型代码的关系，同时提供了生产和验证模式的双重执行。

**🔧 技术方法**

使用了PyTorch的原生分布式张量和FSDP（Fully Sharded Data Parallel）技术，结合了双模式分布式张量执行、拓扑感知的完全分片数据并行性和布局驱动的分布式Muon优化器。

**📊 数据集**

在Atlas 900 A3 SuperPoD上进行了评估，使用了从16个物理卡到384个物理卡的多种配置。

**📈 对比分析**

与PyTorch FSDP2和Megatron DDP进行了比较，结果显示在16个物理卡上，平均步骤时间减少了29.7%，在128个排名的情况下减少了25.5%，且每步损失在1000步内与基线保持一致，Pearson相关系数超过0.999997。

**⚠️ 局限性**

限制在于当前实现依赖于特定的硬件架构，可能在其他硬件上表现不佳，同时在处理复杂模型时可能需要更多的手动调优。

---

## 336. Et Tu, MacBook? Unprivileged Keystroke Inference and Context Profiling via the Built-in IMU Side Channel

**arXiv ID:** 2609.21569 | [PDF](https://arxiv.org/pdf/2609.21569v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 337. GestureFAR: Streaming Co-Speech Gesture Generation with Flow Autoregression

**arXiv ID:** 2609.21576 | [PDF](https://arxiv.org/pdf/2609.21576v1)

**作者:** Pinxin Liu `[一作]` (University of Rochester), Luchuan Song `[通讯]` (University of Rochester)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种实时、逐词因果的共语手势生成框架 GestureFAR，能够在用户说话的同时实时生成同步且自然的全身手势。

**💡 创新点**

创新点在于：①使用因果连续潜在向量（Causal Motion VAE）替代离散码本，保持连续高维手势表达；②采用流匹配（Flow Matching）头进行自回归采样，避免迭代扩散过程的多步延迟；③提出头部单步分布匹配蒸馏（Multi‑Procedure Distribution Matching Distillation），冻结主干仅蒸馏流头，实现一阶采样并保持因果性。

**🔧 技术方法**

核心技术包括因果运动 VAE、音频‑运动因果 Transformer、连续流匹配头、分布匹配蒸馏、状态化流式部署（缓存、身体锚点、缓冲区、状态平滑）。

**📊 数据集**

在 BEAT2（约 60 小时 SMPL‑X 动作与 25 位说话者的语音）上进行训练与评估。

**📈 对比分析**

与离线/块式方法及现有流式方法（MIBURI、LiveGesture 等）进行对比。GestureFAR 在 FGD、Beat Consistency（BC）、L1 Diversity 上均优于或与最佳方法持平，同时实现 9.3 ms/token（≈122 FPS）实时生成，显著低于多步教师模型（24.4 ms/token）。

**⚠️ 局限性**

局限性：①依赖 VAE 潜在空间的质量，对极端姿势或细粒度手部动作仍可能失真；②蒸馏后对多模态多样性的捕捉尚未完全；③虽然实时，但在极低延迟（<5 ms）场景下仍需进一步优化。

---

## 338. Micro-Collaborative Poisoning: A Distributed Attack on RAG Systems

**arXiv ID:** 2609.21573 | [PDF](https://arxiv.org/pdf/2609.21573v1)

**作者:** Pedro Pereira `[一作]` (Polytechnic of Porto), Isabel Praça `[通讯]` (Polytechnic of Porto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了一种在检索增强生成（RAG）系统中的分布式攻击——Micro-Collaborative Poisoning；

**💡 创新点**

其创新点在于将误导性信息拆散并嵌入多份看似合法的文档，而非集中于单一明显恶意段落；

**🔧 技术方法**

作者使用多种检索器（BM25、Dense BGE、Graph）与不同检索深度、数据库组合及两种LLM（llama‑4‑scout‑17b、openai‑gpt‑oss‑120b）进行实验，并提出文档级毒性指标DPAR；

**📊 数据集**

实验数据集为 HotpotQA 与 MS‑MARCO 的 100 组对齐问答；

**📈 对比分析**

通过对比直接毒化基线，利用 OLS 回归和 ASR、Poison@k 等指标，证明 Micro‑Collab 能在保持较高攻击成功率的同时，显著降低文档级可检测性；

**⚠️ 局限性**

局限性包括仅在有限的 100 题组和两种 LLM 上评估，缺乏对更大规模、更多多源语料和实际防御策略的验证。

---

## 339. Artificial Intelligence as an Economic, Environmental, Geopolitical, and Social Transformation

**arXiv ID:** 2609.21632 | [PDF](https://arxiv.org/pdf/2609.21632v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 340. Accelerating Dense LLMs via L0-regularized Mixture-of-Experts

**arXiv ID:** 2609.21672 | [PDF](https://arxiv.org/pdf/2609.21672v1)

**作者:** Zhenyu Zhang `[一作]` (YZW), Meng Chen `[通讯]` (Wise AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种基于L0正则化的轻量Mixture-of-Experts模型L0-MoE，用于在仅30B训练语料下加速LLM推理，并通过聚类混淆矩阵进行域感知数据集筛选和动态批处理。

**💡 创新点**

创新点在于使用L0正则化选择MLP隐藏维度构成专家，利用聚类混淆矩阵进行域感知数据集构建，以及动态批量调度训练，实现近乎无性能损失的2.5×加速。

**🔧 技术方法**

采用L0正则化、Mixture-of-Experts架构、BGE-M3编码器+K-means聚类、动态批处理、FSDP/SGlang等框架。

**📊 数据集**

训练使用RedPajama 30B token数据集，评估在MMLU、GSM8K、HumanEval、BBH四大基准上。

**📈 对比分析**

与原始LLM、GPTQ、LLM Shearing、RKD+CoT等加速基线比较，L0-MoE在保持或略优性能的同时实现2-2.5×推理速度提升。

**⚠️ 局限性**

限制包括仅在30B语料下验证，未与大型MoE如DeepSeek-MoE对比，可能存在专家冗余和序列级聚类导致的曝光偏差。

---

## 341. Predictive Suppression Layers for Communication-Efficient Spiking Neural Networks

**arXiv ID:** 2609.21583 | [PDF](https://arxiv.org/pdf/2609.21583v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 342. Multi-Domain Clustering via Measure Quantization

**arXiv ID:** 2609.21664 | [PDF](https://arxiv.org/pdf/2609.21664v1)

**作者:** Rafael Pereira Eufrazio `[一作]`, Charles Casimiro Cavalcante `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一种基于测度量化的多域聚类框架

**💡 创新点**

将测度量化问题推广到多域，使用概率度量（Sinkhorn、MMD）优化共享质心，并通过小批量梯度下降实现可扩展

**🔧 技术方法**

Sinkhorn散度、最大均值差距(MMD)、最优传输、梯度下降、小批量优化

**📊 数据集**

Office-31、Office-Home、Caltech-Office10、DomainNet、TAU Urban Scenes、Tennessee Eastman Process

**📈 对比分析**

与池化K-means、谱聚类、Ward、BIRCH及多域MWMS比较；Sinkhorn方法在Hungarian准确率、NMI、ARI上平均排名最佳，特别在大规模DomainNet上优于mini-batch K-means

**⚠️ 局限性**

对超参数敏感（尤其熵正则化）；MMD表现不如Sinkhorn；尚未证明收敛性；仅适用于同一欧氏空间的域

---

## 343. SABER: Learning Attention-based Semantic Affordance for Legged Locomotion

**arXiv ID:** 2609.21572 | [PDF](https://arxiv.org/pdf/2609.21572v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 344. CityLearn v3: A Configurable Simulation and Evaluation Framework for Realistic Control Studies of Renewable Energy Communities

**arXiv ID:** 2609.21570 | [PDF](https://arxiv.org/pdf/2609.21570v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 345. Analysing the Linearity of Linguistic Relations in Language Model Embedding Spaces

**arXiv ID:** 2609.21655 | [PDF](https://arxiv.org/pdf/2609.21655v1)

**作者:** Vasudevan Nedumpozhimana `[一作]` (Trinity College Dublin), John Kelleher `[通讯]` (Trinity College Dublin)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估了一种衡量语言模型嵌入空间中语言关系线性编码程度的框架，基于线性逼近与相关/不相关词对约束。

**💡 创新点**

创新点在于将语言关系的线性可编码性转化为可优化的线性逼近问题，并通过对相关与不相关对的约束揭示不同关系的线性结构。

**🔧 技术方法**

使用了线性代数与线性规划技术来求解关系的最优线性变换，以及对词嵌入进行平均层输出取值。

**📊 数据集**

使用了扩展版BATS数据集（覆盖屈折、派生、词典和百科关系的40个词对关系），并利用GloVe、RoBERTa与ModernBERT生成嵌入。

**📈 对比分析**

通过计算线性逼近误差（目标函数归一化后得分）进行比较，结果显示屈折和派生关系误差为0，词典与百科关系误差显著增大；RoBERTa和ModernBERT比GloVe更线性，且在词典关系RoBERTa更好，百科关系ModernBERT更好。

**⚠️ 局限性**

局限性在于仅考察词对关系，未覆盖句子层面；线性逼近无法完全捕捉多对多或不确定性关系；实验数据集手工扩展可能带来偏差；仅用平均层输出的词向量，忽略上下文动态变化。

---

## 346. HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation

**arXiv ID:** 2609.21597 | [PDF](https://arxiv.org/pdf/2609.21597v1)

**作者:** André Lopo `[一作]` (Universidade de Lisboa), Rodrigo Ventura `[通讯]` (Universidade de Lisboa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于单目相机的航天器姿态估计框架（Hypothesis‑Anchored Tracking, HAT），通过在关键帧生成多种姿态候选，并利用相邻帧的相对运动信息在时间上对候选进行Viterbi式的后向选择，再将选定的绝对姿态与单目SLAM的相对运动进行融合，得到每帧高精度的6‑DoF姿态；

**💡 创新点**

核心创新在于：①在绝对姿态估计与相对运动融合前先对多候选姿态进行时序选择，避免单帧误判导致的姿态漂移；②采用离散Viterbi选择与规模加权，兼顾外观得分与运动一致性；③保持因果性，输出后不再回溯修正，保证实时性；

**🔧 技术方法**

使用的技术包括：预训练的单图姿态估计器 MegaPose 或 PicoPose 产生姿态候选；DROID‑SLAM 作为相对运动估计器；基于CAD模型的尺度对齐；Viterbi动态规划进行姿态基底选择；Huber 损失与权重自适应的融合优化；

**📊 数据集**

在四个数据集上评估：SPARK‑2024（合成航天器图像）、YCB‑Video（日常物体）、SwissCube（小型卫星）以及 SHIRT（实验室快速旋转航天器）;

**📈 对比分析**

与单帧绝对姿态方法（MegaPose、PicoPose、GigaPose）及其他视频跟踪/SLAM基线（RGBTrack、SRT3D、DROID‑SLAM）进行对比。HAT 在三大航天器数据集上均显著降低平均姿态误差（约 20‑30%），并在大多数场景下提升实时率（最多 2–3 倍），同时保持误差分布窄化；

**⚠️ 局限性**

局限性包括：①对极远距离或小目标投影时姿态候选的召回不足，导致后续选择受限；②在高速旋转场景（如 SHIRT）下相对运动估计不可靠，影响基底选择与融合；③需要已知尺度的 CAD 模型及先验检测/分割；

---

## 347. Goal-Oriented Communication and Control Co-Design via Semantic Push-Pull in Industrial IoT

**arXiv ID:** 2609.21566 | [PDF](https://arxiv.org/pdf/2609.21566v1)

**作者:** Muhammad Azeem Khan `[一作]` (Mid Sweden University), Fortunato Santucci `[通讯]` (University of L’Aquila)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种基于6G语义层的通信‑控制协同设计框架，用于在无线资源受限的工业IoT中实现高效、稳定的网络控制。

**💡 创新点**

创新点在于：①独立管理上下行资源的语义层；②将状态到误差比（SER）与信噪比（SNR）融合为统一的抑制规则；③采用增益归一化的死带消除工厂异质性；④通过AoI驱动的控制器拉取闭环解决PureET的 silent deterioration。

**🔧 技术方法**

使用技术包括：事件触发控制、语义通信、状态到误差比、信噪比评估、Rayleigh衰落模型、LQR线性二次调节器、两阶段预测与传播、两路（上下行）优先级调度、倒立摆动力学仿真。

**📊 数据集**

采用30个倒立摆（分为三类不稳定性等级）的仿真数据，在Rayleigh衰落的6G上行链路上测试性能。

**📈 对比分析**

与PureET、Round‑Robin、Max‑AoI、Random四种基线对比，结果显示该方案在保持相近跟踪误差（θ_RMS≈0.8°）的同时，传输率仅为周期调度的1/10，通信‑控制联合成本约为PureET的1/14，公平性略低但未出现任何工厂饿死。

**⚠️ 局限性**

主要限制：缺乏正式稳定性证明；假设下行链路无误差，仅考虑上行衰落；未考虑下行衰落、PHY层链路适配或学习辅助的MPC；在真实工业环境中多路径、干扰或更复杂网络拓扑下可能影响性能。

---

## 348. kgsteward: a tool for building, reproducing and maintaining distributed knowledge graphs

**arXiv ID:** 2609.21564 | [PDF](https://arxiv.org/pdf/2609.21564v1)

**作者:** Marco Pagni `[一作]` (SIB Swiss Institute of Bioinformatics), Florence Mehl `[通讯]` (SIB Swiss Institute of Bioinformatics)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a2602d71-93ab-4bad-974b-672788df8193` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了kgsteward，一款基于Python的命令行工具，能够通过单一版本化的YAML配置文件在多种RDF三元组存储上构建、更新和验证知识图谱，并在两个真实科研项目（Sinergia Wolfender植物提取物分析图谱和ReconXKG人类代谢网络图谱）中验证其可用性。

**💡 创新点**

核心创新点包括：① 把知识图谱管理转化为声明式配置和增量式重建；② 通过SPARQL 1.1 UPDATE在存储内部完成数据整合与语义映射；③ 兼容多种三元组存储并提供跨存储一致性校验；④ 将验证查询与使用示例统一为SPARQL测试集；⑤ 通过版本控制实现可重复、可审计的构建流水线。

**🔧 技术方法**

技术栈包括Python ≥3.10、YAML、SHA‑256校验、SPARQL 1.1 UPDATE与查询、GraphDB、QLever、rdf4j、Fuseki等RDF存储、SPARQL 1.1 UPDATE、SPARQL 1.1 PROTOCOL。

**📊 数据集**

使用的数据集涵盖：Sinergia Wolfender项目的1,600余个植物提取物（超过1,000,000条LC‑HRMS/MS光谱、2,665条已匹配化合物及完整分类注释）以及相关生物活性实验；ReconXKG集成了VMH/VMH2、MetaNetX、UniProt、Rhea等公共代谢数据库，并对其进行语义对齐与多版本网络构建。

**📈 对比分析**

kgsteward通过增量式校验仅重新加载发生变化的数据源，显著降低重建成本；对不同三元组存储执行相同查询并以排序后的TSV文件比对，可验证结果一致性；日志记录了SPARQL更新执行时间，用于性能对比。虽然文中未给出数值基准，但在Sinergia项目中已成功处理6.4亿条三元组（5.5 GB压缩），且支持快速增量更新。

**⚠️ 局限性**

局限性包括：① 变更检测对任何预/后处理脚本的细微改动敏感；② 远程源依赖HTTP HEAD头信息，若服务器未正确返回会导致误检；③ SPARQL UPDATE缺乏递归等高级转换功能；④ 依赖商用或特定扩展的三元组存储可能导致跨平台差异；⑤ 对大规模数据集的序列化校验未充分优化，导致重建时可能仍需全量处理。

---

## 349. Reducing Barriers to Academic Support: Evaluating a Course-Specific RAG System for Addressing Help-Seeking Disparities in Higher Education

**arXiv ID:** 2609.21600 | [PDF](https://arxiv.org/pdf/2609.21600v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 350. Open Platform Field Experiments: Expanding the Design Space of Experimental Research on Social Media

**arXiv ID:** 2609.21608 | [PDF](https://arxiv.org/pdf/2609.21608v1)

**作者:** Jordi Guillem Condom-Tibau `[一作]` (University of Pisa), Stefano Cresci `[通讯]` (Institute for Informatics and Telematics, National Research Council)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出开放平台实验（OPFEs）范式，并在Bluesky上实现，构建实验设计空间并与现有方法比较

**💡 创新点**

将实验设计空间与开放平台特性相结合，发现OPFEs在生态有效性、可控性、可观测性等方面独占空白区域

**🔧 技术方法**

基于AT Protocol开放架构、可修改组件（客户端、推送、标签）和Firehose事件流等技术

**📊 数据集**

利用Bluesky的公共事件流（Firehose）和索引视图，并结合公开的PDS实例

**📈 对比分析**

通过对比7类实验原型（问卷、模拟、现场、客户端、平台运行）在7维属性上的雷达图，OPFEs在大多数维度得分高，实验效果可在实际用户中显现

**⚠️ 局限性**

受限于开放组件的可变性、平台政策变化、缺乏完全可复制性、受限于用户主动参与导致样本偏倚

---

## 351. Trading Depth for Time in Recurrent Transformers

**arXiv ID:** 2609.21605 | [PDF](https://arxiv.org/pdf/2609.21605v1)

**作者:** Zeyi Huang `[一作]` (Microsoft), Yelong Shen `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在自回归解码时，作者在每两个词元之间插入“思考”隐藏标记，让模型在同一块网络上多次迭代，以实现时间维度的加深。

**💡 创新点**

创新点在于将时间递归（通过思考标记）与物理深度相互比较，并证明在保持同一块网络的前提下，通过额外的时间步骤即可恢复大部分深度带来的性能提升；同时引入跨词元递归和 KV 反馈来强化信息传递。

**🔧 技术方法**

主要技术包括 Latent Recurrent Transformers（LRT）、多专家（MoE）架构、并行多重细化训练（Parallel Multi Refinement Training）、以及对比实验中的 Loop Transformer 与 PonderLM‑2 等模型。

**📊 数据集**

实验基于 NanoChat 生成式语言模型的数据集，使用 16 层和 20 层 MoE Backbone 进行训练与评估。

**📈 对比分析**

作者通过比较不同模型（单层 LRT、加深 LRT、加思考标记的 LRT、Loop Transformer、PonderLM‑2）在 Bits‑Per‑Byte（BPB）指标上的表现来评估效果；结果显示，一思考标记的 LRT 能恢复约 67%–81% 的双深度改进，且参数量比双深度模型低约 48%，二思考标记进一步提升性能，几乎逼近三倍深度模型。

**⚠️ 局限性**

限制包括：训练与推理的 FLOPs 与延迟未完全匹配；思考标记数量固定，未探讨自适应数量；仅在固定任务与模型规模下验证，未证明在更大模型或其他数据集上的通用性；以及时间递归对注意力开销的影响未做深入分析。

---

## 352. Resolution of an Open Problem on Quasi-Cyclic Codes over $\mathbb{Z}_4$ and New Quaternary Linear Codes

**arXiv ID:** 2609.21601 | [PDF](https://arxiv.org/pdf/2609.21601v1)

**作者:** Nuh Aydin `[一作]` (Kenyon), Aditya Tyagi `[通讯]` (IIT Gandhinagar)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `847a60d8-a755-47af-ba5d-c5236b9e3083` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了在奇长度的ℤ₄上由循环码构造的1生成量子循环（QC）码，并给出了该QC码与原始循环码类型相同的必要且充分条件，解决了先前的开放问题；同时提出了自由性判定和通过计算搜索得到的多组性能更优的新四元码；

**💡 创新点**

首次提供了奇长度ℤ₄上1生成QC码类型保持的完整判据，克服了非域环中类型不继承的难题，并利用此判据发现36个Lee距离显著改进的新码；

**🔧 技术方法**

利用Chinese Remainder Theorem对xᵐ−1的分解、有限链环结构、Hensel引导、格雷映射以及Magma软件进行符号计算与搜索；

**📊 数据集**

通过对奇长度m的枚举搜索，利用自身生成的36组新码作为实验数据；

**📈 对比分析**

将所得码的Lee距离与http://quantumcodes.info/Z4数据库中已知的同长度、同类型码进行对比，改进幅度从+2到+54，整体性能显著提升；

**⚠️ 局限性**

仅适用于奇长度m（依赖xᵐ−1平方无根性），对偶长度情况未作扩展，仅关注1生成QC码，缺乏对更一般情况的分析。

---

## 353. Evaluating In-Context Learning and Retrieval Strategies for Devanagari Post-OCR Correction

**arXiv ID:** 2609.21595 | [PDF](https://arxiv.org/pdf/2609.21595v1)

**作者:** Abhishek Bhandari `[一作]` (Indian Institute of Technology Jodhpur), Gaurav Harit `[通讯]` (Indian Institute of Technology Jodhpur)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在印地语和马拉地语的 Devanagari OCR 后校正任务上，首次系统评估了大规模语言模型的无训练提示效果，并提出了基于字符 n‑gram BM25 的检索式样本选择方法。

**💡 创新点**

创新点在于将字符级 BM25 检索作为示例选择策略，能直接匹配 OCR 错误模式，且在 GPU 资源极低的情况下实现与语义检索相当甚至更优的性能。

**🔧 技术方法**

核心技术包括：大模型（3B–32B）无参数提示、字符 n‑gram（n=1,2,3）BM25 检索、零样本与多样本（k=3,5）提示、以及与密集语义检索的对比。

**📊 数据集**

使用的公开数据集为 20,000 条句子（10,000 条印地语、10,000 条马拉地语），涵盖 5 个新闻领域，采用渲染‑OCR 生成的噪声文本与干净真值对。

**📈 对比分析**

实验表明：模型规模是主导因素，Gemma‑3‑27B 在 -5（k=5）设置下分别将印地语 WER 降至 16.22%（+55%）、马拉地语降至 23.42%（+33%）；CharBM25 在 8B 以上模型中比随机选择低 2.8–4.0pp，且在大多数模型上匹配或超越 dense 检索，但在 8B 以下模型效果不佳。

**⚠️ 局限性**

局限性包括：对低规模模型的性能不足，尤其在马拉地语会出现错误增幅；依赖字符级错误信息，无法纠正命名实体等语义错误；仅评估了印地语和马拉地语，缺乏对其他 Indic 脚本的验证。

---

## 354. Rethinking Human-Aligned Evaluation: An Analysis of Semantic Metrics Beyond WER

**arXiv ID:** 2609.21663 | [PDF](https://arxiv.org/pdf/2609.21663v1)

**作者:** Hritika Sharma `[一作]` (Algoma University), Somang Nam `[通讯]` (Algoma University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了英文版人类对比评测数据集 HATS-en，并系统比较了多种自动评测指标与人类偏好的一致性。

**💡 创新点**

创新点在于提供大规模英语对比评测数据，全面检验 BERTScore 与 SemDist 在不同语言模型层、聚合方式下的表现，并指出 CER 在人类判断中的优越性。

**🔧 技术方法**

技术上使用了 BERTScore、SemDist 等语义指标，并对 LLM 层级与池化策略进行网格搜索；通过 WER、CER 以及 Fleiss κ、McNemar 等统计检验评估指标一致性。

**📊 数据集**

数据集来自 LibriSpeech test‑clean，随机抽取 1,000 个音频段，每段用四个不同 ASR 系统生成假设，构成 2,000 条假设与参考文本的对比实例。

**📈 对比分析**

评估方法为计算每个指标对参考与两条假设的得分排名，与人工统一偏好对比，得到一致率；CER 在 86.3% 一致率下优于 WER 的 75.5%，部分 SemDist 配置甚至超过 CER（约 90% 以上）。

**⚠️ 局限性**

局限性包括仅覆盖读书式英语，缺少口语、非英语或多领域数据；聚合策略和 LLM 选取仍需进一步研究；以及仅在人工一致样本上评估，可能忽略难以区分的案例。

---

## 355. When Steering Fails in Latent Reasoning: A Latent-to-Language Transition Gap

**arXiv ID:** 2609.21662 | [PDF](https://arxiv.org/pdf/2609.21662v1)

**作者:** Gaoxiang Huang `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Lei Qi `[通讯]` (Southeast University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在语言模型的隐式推理（latent CoT）中使用激活干预（activation steering）对输出的影响，并与传统的显式链式推理（explicit CoT）进行了对比。

**💡 创新点**

提出并验证了“latent-to-language transition gap”假设，即在隐式推理阶段对隐藏状态的干预难以传递到后续的语言生成阶段，导致激活干预效果显著衰减。

**🔧 技术方法**

使用了激活干预技术（Contrastive Activation Addition, CAA），对 Llama-3.1-8B 与 Llama-2-7B 两大模型在显式与隐式 CoT 模式下的隐藏层（第16层）进行干预，并评估其对输出分布和任务相关信号的影响。

**📊 数据集**

实验数据集包括：Sentiment（情感倾向数据集）、TruthGen（真值生成数据集）以及 TwinViews-13k（政治倾向数据集），用于评估模型在三类任务上的表现。

**📈 对比分析**

通过比较显式 CoT 与隐式 CoT 在干预成功率（Steering Success Rate）、无采样的立场边际变化（Stance Margin）和标准化位移（Normalized Displacement）等指标，发现显式 CoT 的干预效果显著优于隐式 CoT；在隐式 CoT 中，虽然隐藏表示的位移与显式 CoT 相近，但对应的输出变化仅为前者的1–5%，并且双向干预增益（Bidirectional Steering Gain）低于20倍。

**⚠️ 局限性**

限制与不足：未能确切定位训练过程或模型结构中导致 transition gap 的具体原因；实验仅针对两款 Llama 模型，缺乏跨架构或跨训练策略的系统验证；未来需要设计专门的过渡阶段干预或监督机制以提升隐式 CoT 的可控性。

---

## 356. Outcome-Conditioned End-Effector Geometry Across Vision-Language-Action Policies

**arXiv ID:** 2609.21659 | [PDF](https://arxiv.org/pdf/2609.21659v1)

**作者:** Xingyu Lin `[一作]` (East China Normal University), Dehui Du `[通讯]` (East China Normal University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四种 Vision‑Language‑Action (VLA) 策略在 LIBERO‑Spatial 基准上的闭环轨迹进行跨策略比较，聚焦于端效器 3D 路径的几何相似性，并将比较结果按两策略共同成功、混合或共同失败的 outcome 进行分层。

**💡 创新点**

① 将成功与失败结果显式作为比较的条件；② 用动态时间规整 (DTW) 计算 3D 轨迹距离并对不同时间对齐、窗口截断、端点/时长调整做系统敏感性分析；③ 结合成功状态替换与跨任务对照，分离任务几何与策略差异；④ 报告多种视角下的距离分布而非单一阈值。

**🔧 技术方法**

动态时间规整 (DTW)、多种时间对齐/截断方案、统计 Bootstrap、线性回归、Cliff’s δ、配对差异分析等。

**📊 数据集**

LIBERO‑Spatial（10 个任务 × 20 初始状态 × 3 语言模板 × 5 视觉难度）共 15,000 条闭环轨迹，包含四种 VLA 策略（OpenVLA、OFT、UniVLA、π₀.₅）。

**📈 对比分析**

比较方法：对每个相同配置下的六对无序策略组合，计算 DTW 距离；按 SS、SF、FF 分层统计中位数；对成功状态替换、跨任务对照、同策略内部差异做对照；对不同视觉级别和时间窗口进行多重实验。性能结果：SS 距离约 0.012 m，SF 约 0.038 m，FF 约 0.044 m；在 72 步窗口下差距约 39%；视觉扰动导致策略排名变化。

**⚠️ 局限性**

限制：① 仅包含单次执行（每个配置一次），缺乏多次采样的统计；② 失败样本极少，难以得出可靠的 FF 结论；③ 只使用位置距离，忽略姿态、接触、力等物理信息；④ 视觉扰动采用固定随机种子，真实环境变化无法全面覆盖；⑤ 结果受特定基准、任务、策略实现细节影响，推广性受限。

---

## 357. Steering LLMs Responses Towards Moral Foundations on the Norwegian MFQ-30

**arXiv ID:** 2609.21636 | [PDF](https://arxiv.org/pdf/2609.21636v1)

**作者:** Hans Andersen `[一作]` (University of Oslo), David Dichas `[通讯]` (University of Oslo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在挪威版 MFQ-30 量表上让六个开源 LLM 回答，并对其道德基础得分与 1,282 名挪威受访者进行基准对比；随后分别测试了基于 persona 的提示层级调优和基于 ActAdd 的激活层级调优两种干预方式。

**💡 创新点**

创新点在于首次揭示仅一半模型能主动参与问卷；通过一个简单的中性挪威人口 persona 可显著拉近模型得分与人类分布；证明 ActAdd 在单层单对的设置下无法实现对单一道德维度的精准调节，并提供了“认知幻影”实例。

**🔧 技术方法**

使用的技术包括：第一步概率解码读取数字评分、两项注意力检查过滤、基于人类协方差矩阵的 Mahalanobis d² 作为相似度指标、2×2 规模与呈现方式扰动实验、persona 提示对模型偏好与参与度的影响评估、ActAdd 在第 15 层注入对比向量、以及一套完整的日志解析与可视化脚本。

**📊 数据集**

使用的数据集为挪威 MFQ-30（30 条题目，6 条注意力检查），共 1,282 名受访者；模型方面为 Qwen2.5-1.5B、Qwen3-14B、Qwen3-8B、NorMistral-7B、NorMistral-11B、Gemma 4-E4B-it。

**📈 对比分析**

比较方法：先在两种量表顺序和呈现方式下评估注意力通过率并计算 d²；随后在 persona 与 ActAdd 干预下重新计算 d²。结果显示，prompt persona 可将 d² 下降 44%–77%，但 ActAdd 在层 15 处导致各维度得分趋同，甚至 d² 上升至 31–38。

**⚠️ 局限性**

局限性包括：仅测试六个开源模型；每组仅单次运行，缺乏复现性；ActAdd 仅在单层单对配置下尝试，未探索不同层或多对平均策略；未使用独立的挪威测试样本验证 persona 效果；基于 MPS fp16 的数值不确定性；缺乏跨项相关分析；未评估非指令微调模型或基线模型。

---

## 358. Detection is solved, delineation is not: what governs tooth segmentation on panoramic radiographs

**arXiv ID:** 2609.21628 | [PDF](https://arxiv.org/pdf/2609.21628v1)

**作者:** Muhammad Rehan `[一作]` (AI Dentify), Haider Ali `[通讯]` (AI Dentify)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文构建了1,422张全景X光片的高密度牙齿标注语料库，并通过对输入分辨率、网络结构和解剖先验等因素的系统消融，研究了影响牙齿实例分割与FDI编号的核心驱动因素。

**💡 创新点**

创新之处在于①在统一评估框架下逐项消融，首次证明分辨率是决定边界精度的主导因素；②揭示网络结构在域内几乎无影响，仅在域移位时略显优势；③系统评估了基线网络与基础模型（LoRA自监督编码器、SAM2、Hungarian标签赋值）的失效与边界误差定位，提供了负面结果与机制。

**🔧 技术方法**

采用了YOLO11m-seg（一阶段检测器）、Mask2Former（查询式Transformer）和DINOv3-B+LoRA（自监督编码器）等三大主流架构，配合配色、边界细化实验（SAM2）、线性分配（Hungarian）以及配对自助抽样与置信区间分析，全面评估模型表现。

**📊 数据集**

数据集为自建的1,422张全景片（共42,142颗牙齿多边形）以及公开的DENTEX 6,340张外部片用于零样本迁移测试；所有图像均经过人工精细标注并双人审核。

**📈 对比分析**

比较方式为在相同预处理、评估代码和后处理下使用配对自助法计算95%置信区间，结果显示：分辨率从640→1024提升mAP50‑95 +0.054（p<0.001），再至1280提升+0.007；网络结构差异在域内可忽略（±0.007），但在域外提升约+0.008；基线网络在域内mAP50‑95≈0.717，零样本迁移下降62%。

**⚠️ 局限性**

局限性包括：数据仅来自单一来源且缺乏患者元数据；缺乏跨标注者一致性评估；外部数据DENTEX的标注协议不同，导致域移位与标注差异混杂；未涵盖超数牙等特殊情况；并且模型对根尖无可视边界的误差主要源于解剖不确定性，难以进一步改进。

---

## 359. One Prompt Does Not Fit All: Self-Meta-Evolve for Personalized Information Extraction

**arXiv ID:** 2609.21626 | [PDF](https://arxiv.org/pdf/2609.21626v1)

**作者:** Hongliang Li `[一作]` (Microsoft), Dongmei Zhang `[通讯]` (Microsoft)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Self‑Meta‑Evolve，基于层次化的每用户提示适配框架，用交互反馈不断改进企业信息抽取的提示；

**💡 创新点**

创新点在于将提示编辑拆分为内部（用户级）编辑与外部（元策略）演化两层，利用元提示提炼跨用户的编辑模式；

**🔧 技术方法**

采用大语言模型（GPT‑5.x、Claude 等）进行提示生成、用户模拟与元提示学习，使用结构化 JSON 提示和编辑计划；

**📊 数据集**

构建了 292 个基于 O*NET 的可复现企业角色人设及对应的合成文档，另设 STEM 与人文子集做跨域验证，并对20位真实专业人士进行人类评估；

**📈 对比分析**

在 59 名验证角色上与梯度、搜索、进化、Bandit 等多种基线对比，Self‑Meta‑Evolve 的成功率达 74.58%（相较 ProTeGi 提升 13.56 点），在人类对比中获胜率 71%；

**⚠️ 局限性**

局限在于依赖 LLM 模拟的 AI‑User 反馈、缺乏真正的领域知识补充、仅覆盖英文企业场景，未来需加入真实用户反馈与检索增强。

---

## 360. Calibrating Teacher--Student Discrepancy for On-Policy Distillation

**arXiv ID:** 2609.21619 | [PDF](https://arxiv.org/pdf/2609.21619v1)

**作者:** Qiangqiang He `[一作]` (Nanjing University), MingCai Chen `[通讯]` (Nanjing University of Posts and Telecommunications)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种校准的 On‑Policy Distillation（Cal‑OPD）方法，先用正负干预估计教师自偏差（TSD）区域，再将标准 OPD 中的教师‑学生差距中非知识相关的部分过滤，只保留真正有用的差距进行蒸馏。

**💡 创新点**

创新点在于首次量化并剔除教师自偏差，将其视为噪声，并通过正负上下文干预构造 TSD 区域，随后用该区域校准 OPD 的优势信号，从而显著提升数学推理模型的学习效果。

**🔧 技术方法**

主要技术包括：对教师做正负干预、计算教师对数概率变化 Δ，估计 TSD 区域 ℛ̂，使用 λ 松弛因子扩展该区域，基于校准后的优势 A_t^Cal 重新构造 OPD 损失，并对比标准 OPD、ExOPD、EOPD、Uni‑OPD 与 Privileged‑OPD 等变体。

**📊 数据集**

训练使用过滤后的 DAPO‑17k（基于 Qwen3‑235B‑A22B‑Instruct‑2507）数据集；评估基准为 AMC23、AIME24/25/26、HMMT26、MATH500 共六个数学推理任务。

**📈 对比分析**

实验与五种 OPD 基线对比，Cal‑OPD 在 Qwen3‑4B→Qwen3‑1.7B 与 Qwen3‑30B→Qwen3‑4B 两组教师‑学生规模下平均提升约 2–4 分，且在 7/12 benchmark‑配置对中获得最高分，且显著减少响应长度与训练时间。

**⚠️ 局限性**

局限性在于：仅验证于数学推理任务；TSD 估计仅基于有限的正负干预，可能无法覆盖所有教师偏差；过大 λ 会过度过滤有用信息；未探究多模态或更通用的推理场景。

---

## 361. CounterPlay: Counterfactual Post-Training for Self-Play Driving Policies

**arXiv ID:** 2609.21617 | [PDF](https://arxiv.org/pdf/2609.21617v1)

**作者:** Jiarong Wei `[一作]` (University of Freiburg), Abhinav Valada `[通讯]` (University of Freiburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于自我对弈后训练的counterfactual方法：利用价值估计回溯失败点，给定不同奖励条件在同一策略上重驱任务，并通过任务完成与同伴安全的比较来过滤、验证后再蒸馏，显著提升驾驶政策在复杂交通环境下的成功率。

**💡 创新点**

创新点包括①价值引导的失败回溯，自动挑选更早的干预点；②利用奖励条件在同一网络上实现多种驾驶风格的重驱；③通过任务完成度与同伴安全的双重评判过滤验证，确保蒸馏样本既完成任务又不新增风险。

**🔧 技术方法**

技术实现主要基于PPO自我对弈、奖励条件化策略、价值估计与回溯根选择、共享随机性下的分支生成、任务结果排序与验证、KL蒸馏以及在PufferDrive高吞吐量模拟器中的并行训练。

**📊 数据集**

使用了BehaviorBench（Waymo Open Motion Dataset）中的八个冻结交通场景，结合交互式1k与随机1k两份验证集，并在80,000条训练场景上进行后训练。

**📈 对比分析**

与多种后训练基线（PPO-Continue、PPO-Rewind、PPO-Opponent、SPICED、CL4AD、DAPO、VinePPO、PlannerRFT、CRAFT）以及持续自我对弈进行对比。结果表明，本文方法在两份验证集上平均提升约7–8分（交互式1k 7.9分，随机1k 7.7分），并在所有八个交通规则下均实现最高分数，证明其在完成率与安全性方面均优于现有方法。

**⚠️ 局限性**

主要局限在于：①探索空间仅限于预设的五种奖励配置，无法针对每个失败自动生成更合适的驾驶风格；②验证阶段只检查同伴安全，未对部署后同伴影响做进一步评估；③对价值估计与回溯根存储的依赖，若估计不佳会影响回溯质量。

---

## 362. Certificates for short extending words in a finite automaton

**arXiv ID:** 2609.21603 | [PDF](https://arxiv.org/pdf/2609.21603v1)

**作者:** Michele Miccinesi `[一作]` `[通讯]` (University of Pisa), Michele Miccinesi (University of Pisa)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文提出一种线性函数 B(S)（基于状态集的入度偏差）来判定同步有限自动机中子集是否能在长度不超过 n−1 的词扩展，并给出其闭式、计算方法以及与 Kari 欧拉扩展引理的关系。

**💡 创新点**

创新点在于将 KARI 的欧拉扩展引理推广为按子集的充分条件：若 B(S)≥0 则 minext(S)≤n−1；证明 B 的零集恰好为欧拉自动机；提出二阶矩测试补充 B，证明其在大多数子集上有效；并探讨 B 的极限与 Friedman's 中心化权重的关系。

**🔧 技术方法**

使用的技术包括计数恒等式、线性代数（矩阵幂、特征值、Krylov 空间）、组合论、随机方法、NP‑完备性分析以及大规模枚举算法。

**📊 数据集**

使用了对 n≤7 的二进制及三进制同步自动机的完整枚举数据集，并在 GitHub 仓库中提供了对应的实现程序。

**📈 对比分析**

与传统需要对所有子集检查 n−1 长度的扩展方法相比，B 只需 O(kn²) 的多项式时间计算单个子集；实验表明在 60%–95% 的子集上（视阈值而定）能被 B 或二阶矩测试判定为可在 n−1 内扩展；但未能给出更紧的 reset 阈值，仍局限于 n−1 的上界。

**⚠️ 局限性**

局限性包括：1) 无法得到全局 reset 上界；2) B 对 B<0 的子集无效，二阶矩测试也无法覆盖全部子集；3) 对非欧拉自动机的单子集结果不适用；4) 计算复杂度对 n 较大时仍为指数；5) 该方法无法解决是否所有子集都能在 n−1 内扩展的问题。

---

## 363. New lower bounds for kissing numbers in dimensions $25$--$29$ and $31$

**arXiv ID:** 2609.21591 | [PDF](https://arxiv.org/pdf/2609.21591v1)

**作者:** Rustem Takhanov `[一作]` (Nazarbayev University), Stanislav Yun `[通讯]` (Nazarbayev University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究并改进了25–31维空间中接触数（kissing number）的下界，利用Leech lattice提升和块旋转技术。

**💡 创新点**

创新点在于发现了块旋转、非正交线性变形以及对提升子集的灵活选择，使得在这些维度中可加入额外的接触点。

**🔧 技术方法**

采用Leech提升构造、线性规划求解、旋转矩阵搜索、数值优化与区间计算验证等技术。

**📊 数据集**

使用PackingStar提供的496点Leech子集以及E_7根系统等已知的Leech子集与辅助配置。

**📈 对比分析**

与之前的最佳构造相比，新下界分别为τ_25≥197058, τ_26≥198552, τ_27≥200046, τ_28≥204522, τ_29≥209497, τ_31≥238354，提升点数为2-4。

**⚠️ 局限性**

局限在于仅针对维度25–31的提升方法，对维度30未能改进，且改进依赖于特定的子集与旋转，未证明全局最优。

---

## 364. Tilt as a Certified Resource: Preserving Motor Wrench-Rate Authority on Articulated Multirotors

**arXiv ID:** 2609.21580 | [PDF](https://arxiv.org/pdf/2609.21580v1)

**作者:** Giuseppe Silano `[一作]` (Ricerca sul Sistema Energetico), Martin Saska `[通讯]` (Czech Technical University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种基于电机仅速率可用性（readiness）对多旋翼机（MRAV）进行安全约束的控制器：先用一个由电机速度决定的“读iness”度量（log‑volume）构造控制屏障函数（CBF），再将其与主动倾斜（servo tilt）耦合，通过统一的物理命令二次规划（UPC‑QP）实现对电机扭矩与舵机设定点的实时约束。该方法在强风扰动下通过模拟验证，能够在保持姿态跟踪的同时确保系统保持足够的扭矩变换能力。

**💡 创新点**

创新点：① 发现并解决了在对称固定几何平台上，电机速度分配与“读iness”最大化之间的零和退化；② 通过主动倾斜引入几何资源，使得电机速率分配无法抵消的“读iness”提升方向得到恢复；③ 设计了仅基于电机能力的读iness证书，避免了伪装的伺服能力（ghost capacity fallacy）；④ 将该证书以单一不等式形式融入物理命令QP中，避免高阶构造与积分风暴；⑤ 通过在线可行性指标（ρ_CBF）实现自适应的控制约束。

**🔧 技术方法**

技术与方法：Blade‑Element‑Momentum (BEM)、控制屏障函数（CBF）、统一物理命令二次规划（UPC‑QP）、Drag‑Aware Aerodynamic Manipulability (DAAM) co‑metric、线性矩阵不等式与Jacobi公式计算梯度、基于矩阵分解的实时求解、模型预测控制（MPC）式反馈、模拟平台（Omnidirectional Octorotor）以及强风扰动模型。

**📊 数据集**

数据集与参数：使用从Omnidirectional Octorotor架构中提取的几何与力学参数（8个旋翼、6个自由度），并结合文献与实验测得的推力、拖拽、转矩极限、转子惯量、舵机限速等；强风扰动采用三角波与正弦加和的加速度序列，分别代表轻度与强度两种扰动。

**📈 对比分析**

比较方法与性能：四种分配策略——静态伪逆、固定倾斜DAAM、未加CBF的主动倾斜DAAM、带CBF的鲁棒过滤器——在静止、步进、激进机动及两种风扰动下进行对比。结果显示：① 在无或轻度扰动下，所有策略均能实现近似相同的姿态跟踪；② 在强扰动下，未加CBF的方案出现饱和或失稳；③ 加入CBF的鲁棒过滤器在保持h_min≥0（读iness阈值）的同时，饱和率仅为14.98%，并保持与其他方案相当或更好的位置误差（RMS≈1.44 m）。

**⚠️ 局限性**

局限性：① 读iness度量仅为log‑volume阈值，无法保证在所有方向上的充分可操纵性；② 忽略舵机动力学可能导致在高速度舵机负载时出现“幽灵容量”误判；③ 证书对对称几何的假设在制造误差或异构平台上可能失效；④ 在线可行性检测依赖于ρ_CBF的正性，若不可行则需手动终止或退化；⑤ 目前仅在仿真环境验证，缺乏硬件实测与对真实气动失配的完整考察。

---

## 365. Logics of Filter Bubbles

**arXiv ID:** 2609.21565 | [PDF](https://arxiv.org/pdf/2609.21565v1)

**作者:** Lei Li `[一作]` (Shaanxi Normal University), Jialiang Yan `[通讯]` (China University of Political Science and Law)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了用于描述和推理过滤泡沫的静态与动态逻辑框架，分别通过阈值关系和态度概况来定义泡沫结构及其在个性化信息流下的演化。

**💡 创新点**

创新点在于将过滤泡沫的结构与平台个性化推荐机制分离，构建基于态度概况的伪度量网络模型和事件驱动的更新机制，并提供完整的可证明性与可判定性证明。

**🔧 技术方法**

使用了混合逻辑（包含命名符号、阈值模态和Hybrid运算符）、伪度量比较、事件模型（预条件与后置条件）以及形式化的网络更新语义。

**📊 数据集**

本文为理论性工作，未使用任何具体数据集；所有结果均在抽象的有限网络模型与伪度量空间上证明。

**📈 对比分析**

由于是形式化的逻辑研究，未进行实验比较；但通过归约到纯命题逻辑证明了判定性，理论上满足可判定性与强完备性。

**⚠️ 局限性**

局限性包括：模型仅限于有限集合；需要给定可判定的阈值比较函数；对实际大规模社交网络的可扩展性和经验验证尚未展开。

---

## 366. On the Fourier Entropy-Influence Conjecture for Boolean Plateaued Functions

**arXiv ID:** 2609.21563 | [PDF](https://arxiv.org/pdf/2609.21563v1)

**作者:** Vladimir N. Potapov `[一作]` `[通讯]`, Vladimir N. Potapov

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过证明一个新的不等式 2∑_{x∈𝔽₂ⁿ}f(x)(x)≥(f)(n−deg(f))，揭示了布尔函数支持点的平均 Hamming 权重与其代数度数之间的关系，并利用该不等式证明了费伦克斯-恩特罗比–影响（FEI）猜想在平板函数（s‑plateaued）类中的常数为 4，进一步证明了偏平曲函数（partially bent）类的 FEI 常数为 2，并给出了 p‑偏分布下布尔函数取值 1 的概率估计。

**💡 创新点**

创新点主要包括：① 引入并证明了新的代数度数与平均权重之间的关系不等式；② 用该不等式完整解决了平板函数类的 FEI 猜想并给出了最优常数 4；③ 证明了偏平曲函数类的 FEI 常数为 2；④ 推导了 p‑偏分布下更精细的概率下界，改进了 Schwartz–Zippel 及其泛化。

**🔧 技术方法**

技术手段主要包括：布尔函数的代数标准形（ANF）与 Möbius 变换、Walsh–Hadamard 变换与 Parseval 定理、面向量集的偏序与排列构造、二次型的双线性形式、对偶空间与维度计数、凸性与 Jensen 不等式。通过组合论和代数几何的工具，对支持点的权重分布进行精确计数。

**📊 数据集**

本文为纯理论研究，未使用任何外部数据集或实验数据；所有结论均基于解析证明与组合计数。

**📈 对比分析**

与之前已知的部分类（如单项式布尔函数）相比，本文证明了平板函数类的 FEI 常数 4 是最优的；对于偏平曲函数，给出了更紧凑的常数 2，并验证了相应的 H(f)≤2I(f) 等式；p‑偏分布下的概率下界则在 p≤1/4 区间内优于既往的 Schwartz–Zippel 估计，且在整个 0≤p≤1/2 区间内与已有界进行比较，展示了在不同参数范围内的优势。

**⚠️ 局限性**

局限性主要体现在：① 结果仅适用于平板函数、偏平曲函数以及其特定结构的函数，未对一般布尔函数提供 FEI 常数的上界；② 对 p‑偏分布下的概率估计虽然改进，但仍依赖于函数的代数度数，无法消除 d 的影响；③ 证明中涉及的排列与双线性形式虽然有效，但在更大范围内推广的可行性尚未讨论。

---

## 367. Learned Parametric Emotion Editing: Real-Time Affective Filtering for On-Device Social Media Video

**arXiv ID:** 2609.21624 | [PDF](https://arxiv.org/pdf/2609.21624v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 368. Riemannian Neural Hamiltonian Flows: Geodesic Symplectic Transport and Interpretability

**arXiv ID:** 2609.21647 | [PDF](https://arxiv.org/pdf/2609.21647v1)

**作者:** Vincent Souveton `[一作]` `[通讯]` (CEA), Vincent Souveton (CEA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `40105733-5154-44cd-8090-a8cab9e64b07` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并实现了一种在黎曼流形上实现的生成模型——Riemannian Neural Hamiltonian Flows (RNHF)，利用固定的动能、可学习的标量势能和 geodesic leapfrog 积分器，构造可逆、体积保持的流。

**💡 创新点**

创新点在于将神经哈密顿流 (NHF) 迁移至黎曼几何环境，消除雅可比行列式与发散估计，提供可解释的能量景观，并阐明基态与势能竞争机制导致的有限时传输。

**🔧 技术方法**

采用了黎曼几何动力学、geodesic leapfrog 对称辛积分、变分自编码器框架下的 ELBO 优化以及对势能的正则化学习。

**📊 数据集**

在三种二维黎曼空间（欧氏平面 ℝ²、双曲平面 ℍ²、二维球面 𝕊²）上，使用了合成的双峰高斯/von Mises-Fisher 混合目标分布进行评估。

**📈 对比分析**

与基于 Riemannian 连续正规化流 (RCNF) 的对比实验表明，RNHF 在大多数评估指标上与 RCNF 性能相当甚至优越，尤其在正曲率空间上表现更佳，同时训练速度更快、采样开销相对较低。

**⚠️ 局限性**

局限性包括对基态宽度敏感、对精确传输的可表达性理论未完全阐明、需要已知闭式 geodesic 的流形或额外的隐式辛积分器，以及在非高斯目标下势能与匹配势能的差异仍需进一步研究。

---

## 369. From Smarter to Hungrier: the Role of Energy Efficiency in Software-defined Vehicles

**arXiv ID:** 2609.21643 | [PDF](https://arxiv.org/pdf/2609.21643v1)

**作者:** Ella Peltonen `[一作]` `[通讯]` (University of Oulu), Ella Peltonen (University of Oulu)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究软件定义车辆在能耗与可持续性方面的影响，利用实验平台测量ADAS等传感器的功耗，并提出能耗建模与设计准则。

**💡 创新点**

首次从软件层面量化SDV能耗，构建能源模型与可持续性指标框架，并提出服务层级能耗管理的概念。

**🔧 技术方法**

使用NVIDIA Jetson AGX Orin计算平台、Hesai OT128 LiDAR、Carnegie Robotics Multisense S27立体相机、Teledyne FLIR ADK 2.0热像仪，采用ROS2通信并通过电流/功率测量进行能耗评估。

**📊 数据集**

实验室自制测量数据集，包括不同负载下设备功耗；未使用公开数据集。

**📈 对比分析**

与已有行业报告和学术估计对比，验证测得功耗与文献相符，显示软件与传感器在全负载下约增加5–10%的能耗；该测量可为能耗优化提供参考。

**⚠️ 局限性**

实验规模有限，仅涵盖少量传感器；测量在实验室环境而非真实车辆；缺乏大规模真实路测；能耗测量工具尚未成熟，结果仅为预备性研究，需进一步验证。

---

## 370. Extending Decoupled Attention to Dense Prediction and Masked Training for Multi-Channel Images

**arXiv ID:** 2609.21629 | [PDF](https://arxiv.org/pdf/2609.21629v1)

**作者:** Umar Marikkar `[一作]` (University of Surrey), Sara Atito `[通讯]` (University of Surrey)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对多通道图像（MCI）提出了一种能够在任何通道数下进行编码的 Decoupled Vision Transformer（DC-ViT）变体，并将其与独立通道的掩码自监督训练相结合；

**💡 创新点**

创新点在于：①使用线性分配（Hungarian算法）恢复独立掩码下的通道间标记对应关系；②通过“星形成本”将多通道对应问题分解为若干个两通道分配，保持可扩展性；③将 decoupled attention 和 channel-aware pooling 统一到 MCI 的密集预测任务中；

**🔧 技术方法**

主要技术包括：ViT 结构、decoupled attention（分别计算同通道内和跨通道注意力并混合）、channel-aware pooling、线性分配求解、星形成本、以及多通道掩码自监督学习（ChA-MAEViT 的预训练方式）；

**📊 数据集**

在六个基准上评估：CHAMMI、JUMP‑CP（显微镜），BigEarthNet、38‑Cloud（卫星影像），IMC‑Breast、IMC‑Pancreas（成像质谱细胞组学）；

**📈 对比分析**

与 ChannelViT、DiChaViT、ChA‑MAEViT 等 MC‑ViT 基线相比，DC‑ViT‑v2 在所有分类和分割任务上均取得更高的指标；在多通道缺失测试中仍保持领先优势，说明其对通道缺失的鲁棒性；

**⚠️ 局限性**

局限性：①仍需在每个通道独立掩码下使用线性分配，计算开销随通道数和掩码比例增加；②在极端高通道数或极大掩码比例下对应精度可能下降；③实验主要集中在固定 ViT‑S/16 结构，未探讨更大模型或不同自监督任务的兼容性。

---

## 371. From Code Archival to Knowledge Graph: Bridging Software Heritage, COAR Notify and Wikidata

**arXiv ID:** 2609.21667 | [PDF](https://arxiv.org/pdf/2609.21667v1)

**作者:** Camillo Carlo Pellizzari di San Girolamo `[一作]` (Scuola Normale Superiore), Francesco Tosoni `[通讯]` (Sant'Anna School of Advanced Studies)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一套端到端管道，采集并验证软件期刊与SIGMOD ARI等来源的 DOI‑repo 对，随后将其与 Software Heritage (SWH) 归档结果对齐，并通过 Wikidata 生成可查询的学术文献–软件引用关系。

**💡 创新点**

创新点包括：① 为学术文章和软件实例分别设计两套 Wikidata 应用程序配置，保持两类实体分离；② 通过 SWH 的 content‑addressed 标识和 Wikidata 的属性实现双层（引用层与节点层）对齐；③ 兼容 COAR Notify 通知协议，支持未来实时流式更新；④ 所有编辑均经过人工复核并记录完整来源，避免对 Live Wikidata 直接写入未验证数据。

**🔧 技术方法**

技术实现涵盖：SPARQL 读写查询、QuickStatements 与 OpenRefine 批量导入、Software Heritage 的 Bulk Save Code Now API 与 SWHID 查询、URL 规范化与 SWH Merkle DAG 解析、Wikidata 现有属性与 schema.org / CodeMeta 对映、Python 脚本与 GitHub Actions 触发的 CI 流程。

**📊 数据集**

使用了 4,397 条经过编辑验证的 ⟨DOI, repository URL⟩ 对，来源包括 JOSS、SoftwareX、IPOL、SIGMOD ARI；此外利用 SWH 的 4,086 个归档来源及其 4,244 个可解析的 SWHID。

**📈 对比分析**

评估方式为对齐结果与 Live Wikidata 的读写查询对比：仅有 82 个仓库已存在于 Wikidata，剩余 4,315 个可新增，约 44.7% 的文章已在 WikiCite；该管道在一次完整批次中成功创建 4,182 条软件项和 2,326 条文章项，说明在数据量与精度上具备可行性。

**⚠️ 局限性**

局限性主要在于：① 匹配仅基于 URL 规范化，未利用 SWH 图谱进行跨托管迁移识别，导致潜在误判；② 仅覆盖明确声明仓库链接的期刊与报告，未处理自然语言中隐式引用；③ 需要人工复核，扩展规模时成本较高；④ 现有模型未覆盖多语言标签与描述，限制跨语境查询效果。

---

## 372. SynthDemo-RL: Breaking the Zero-Reward Barrier in VLA Adaptation with LLM-Guided Synthetic Demonstrations

**arXiv ID:** 2609.21650 | [PDF](https://arxiv.org/pdf/2609.21650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 373. Faster SVP in Polynomial Space

**arXiv ID:** 2609.21612 | [PDF](https://arxiv.org/pdf/2609.21612v1)

**作者:** Yansong Feng `[一作]` (Chinese Academy Of Sciences), Jiaqi Liu `[通讯]` (Chinese Academy Of Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在多项式空间下运行时间为 n^n/4e+o(n) 的随机化精确最短向量问题（SVP）求解算法。

**💡 创新点**

创新点在于通过将最短向量的多重表示与低空间碰撞搜索结合，替代传统枚举步骤，从而实现指数系数从 1/(2e) 降至 1/(4e)。

**🔧 技术方法**

采用了 Kannan 递归、HKZ / quasi‑HKZ 基础的形状上界、Lyu‑Zhu 低空间碰撞搜索、基于圆角化 Gram–Schmidt 坐标的采样分布、随机 Hadamard 网格与哈希压缩等技术。

**📊 数据集**

本文为理论性工作，没有使用具体数据集；所有结果均通过严格的数学证明获得。

**📈 对比分析**

与之前最优的 n^n/2e+o(n)、2^n/2+o(n) 等方法对比，显著降低了指数系数，得到最小化的 n^n/4e+o(n) 运行时间，实验验证未给出。

**⚠️ 局限性**

局限性包括参数规模极大、随机网格重复次数指数级、实际实现难度高，算法目前仅适用于理论分析而非实际应用。

---

## 374. Beyond Gaussian Worlds: Latent Geometry Matters for JEPAs

**arXiv ID:** 2609.21656 | [PDF](https://arxiv.org/pdf/2609.21656v1)

**作者:** Léo Nicollier `[一作]` (Université Paris-Saclay), Gabriele Facciolo `[通讯]` (Université Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对JEPAs在黎曼几何空间上的线性可辨识性进行理论扩展并验证

**💡 创新点**

证明高维高斯唯一性不再通用，给出球面、圆环等非欧几里得情形的可辨识条件及更紧的近似恢复界限

**🔧 技术方法**

采用拉格朗日扩散产生正向对、热核MMD分布匹配、谱分析与几何适配的正则化

**📊 数据集**

使用合成的高斯、球面、圆环及Clifford-环高维球面等隐空间数据集

**📈 对比分析**

通过R²线性探测与自回归实验对比，匹配几何目标的模型在所有实验中实现最高线性恢复（高斯R²≈0.995，球面≈0.998，圆环≈0.996，高维Clifford最高≈0.967）

**⚠️ 局限性**

局限在于需要精确分布匹配与正交对齐的理论假设，优化对初值敏感，且仅考虑二次相似度损失

---

## 375. TERRA-NG v1.0: Extreme-Scale, GPU-accelerated Mantle Convection

**arXiv ID:** 2609.21633 | [PDF](https://arxiv.org/pdf/2609.21633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 376. Configurable Multi-Stage Vision Pipeline for Crop Disease and Pest Diagnosis

**arXiv ID:** 2609.21651 | [PDF](https://arxiv.org/pdf/2609.21651v1)

**作者:** Naga Ganesh `[一作]` (Digital Green), Vineet Singh `[通讯]` (Digital Green)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了农户在 FarmerChat 上提交的照片诊断流程，拆分为图像质量、作物识别和病虫害检测三阶段，并对比两种实现路径。

**💡 创新点**

创新点在于将诊断拆分为可配置的三阶段流程，提出轻量级 MobileNetV3 质量门并实现病虫害分离的四头模型，以及用模型集体投票构建标签的人工审核闭环。

**🔧 技术方法**

采用 MobileNetV3、DaViT、YOLO、Qwen3‑VL‑4B 等视觉和视觉‑语言模型，并对比单调用 VLM 与多模型协同的两条路线。

**📊 数据集**

使用了来自埃塞俄比亚、印度、肯尼亚和尼日利亚的 1.16 百万张农户照片，以及 Plantix 等现有标注结果构成的模型集体投票标签。

**📈 对比分析**

通过统一的评测规则，在同一 10,335 张测试集上比较，四头 DaViT 模型在作物准确率 95.41% 和诊断准确率 73.26% 上优于 Plantix 和 VLM，成本更低；VLM 路由提供随问随答与后续图像请求能力。

**⚠️ 局限性**

限制包括模型标注依赖现有系统的生产标签、对罕见病虫害的样本不足、缺乏多病虫混合或多作物场景评估、以及质量门的误拒率仍需进一步调优。

---

## 377. PoVD: Efficient Consensus Protocol based on Verifiable Delay Function

**arXiv ID:** 2609.21627 | [PDF](https://arxiv.org/pdf/2609.21627v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 378. Chinese Competitive Debating Dataset and Benchmark

**arXiv ID:** 2609.21637 | [PDF](https://arxiv.org/pdf/2609.21637v1)

**作者:** Zongrui Yang `[一作]` (University of Auckland), Jiamou Liu `[通讯]` (University of Auckland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

构建了一个包含 148 场比赛、2698 个阶段和 20542 个冲突单元的中文辩论数据集，并在此基础上设计了三项评估任务（胜者倾向预测、阶段得分预测、最佳辩手预测）

**💡 创新点**

创新点在于将专业辩论裁判的原始评分与细粒度转录、阶段划分以及统一评判标准结合，形成了可追溯、多粒度、专业级别的辩论理解基准

**🔧 技术方法**

使用大型语言模型（如 DeepSeek、Gemini、GPT‑5.6 等）在零样本条件下进行评估，采用固定提示、无训练数据污染的方式进行对比

**📊 数据集**

使用自组织的真实比赛录音与裁判表单，经过人工校正、ASR 自动转写和细粒度分段，最终得到经过人工验证的中文辩论文本及对应的裁判评分与判定

**📈 对比分析**

对比了多种基线（始终肯定、随机、结构化随机等），零样本 LLM 的最佳表现为：胜者预测准确率 66.2%，阶段得分与人类平均评分的 Pearson 相关系数 0.25，最佳辩手预测准确率 56.8%，整体性能远优于基线但仍显示显著提升空间

**⚠️ 局限性**

局限性包括：模型对辩论流程的把握仍有限；评分标签的可靠性受裁判间分歧影响；数据集仅覆盖中文竞争性辩论，缺乏跨语种或不同辩论形式的泛化；结构化偏差在部分任务中对模型预测影响显著

---

## 379. Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning

**arXiv ID:** 2609.21621 | [PDF](https://arxiv.org/pdf/2609.21621v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 380. When AI Enters the Workplace, Who Faces Greater Risks? A Gendered Analysis

**arXiv ID:** 2609.21756 | [PDF](https://arxiv.org/pdf/2609.21756v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 381. GEM-MPC: Balancing Exploration and Exploitation through Expert-Guided Planning

**arXiv ID:** 2609.21735 | [PDF](https://arxiv.org/pdf/2609.21735v1)

**作者:** Alvaro Serra-Gomez `[一作]` (Leiden University), Thomas Moerland `[通讯]` (Leiden University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 GEM-MPC，通过 MPPI 结合两种采样策略（KL 正则化策略与行为克隆策略）实现规划与学习的更好协同，并引入 Gated Prior Distillation 以筛选过时的规划数据。

**💡 创新点**

创新点：①使用混合专家（mode‑seeking 与 exploration）两策略并行规划；②基于行动价值估计的门控机制（GPD），无需昂贵的 re‑analysis 即可筛选有用的规划分布；③在多任务高维连续控制上展示低计算成本下的性能提升。

**🔧 技术方法**

技术：MPPI（Model Predictive Path Integral Control）、KL 正则化强化学习、行为克隆、前向 KL 损失、动作价值函数、Gated Prior Distillation、分布式 GPU 训练（JAX）等。

**📊 数据集**

数据集：14 个 HumanoidBench 任务（稀疏奖励、极高维度（61 维）动作空间）

**📈 对比分析**

与 PO‑MPC、TD‑MPC2、BMPC 等基线比较；采用 IQM 统计评估；GEM‑MPC 在 14 任务上平均 IQM 为 762.5（比 PO‑MPC 728.0 高），且训练时间从 14 小时降至 9.2 小时，说明在更低计算预算下取得更佳性能。

**⚠️ 局限性**

局限性：①门控依赖行动价值估计，估计误差可能导致错误筛选；②相比单策略方法，额外的策略和价值网络增加了内存与计算负担；③采用高斯策略限制了对多模态或复杂结构的动作空间的表达能力。

---

## 382. A Novel Path-Tracking Algorithm for Automated Tractor-Trailer Forward and Backward Maneuvers

**arXiv ID:** 2609.21718 | [PDF](https://arxiv.org/pdf/2609.21718v1)

**作者:** Alexandre Lombard `[一作]` (Universite De Technologie De Belfort Montbeliard), Abdeljalil Abbas-Turki `[通讯]` (Universite De Technologie De Belfort Montbeliard)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出了一种两阶段的路径跟踪算法，结合拖车侧向控制与短时预测的驾驶员转向调整，并在倒车时使用前进校正恢复耦合角可恢复性；

**💡 创新点**

①在拖车层面直接生成期望耦合角的侧向控制；②通过短时预测搜索合适的拖拉机转向角来实现对耦合角的补偿；③在倒车无法实现目标耦合角时使用前进校正；整体无需训练或专门校准；

**🔧 技术方法**

使用扩展双轮车模型、迭代优化/搜索法求转向角、短时预测仿真；并与Pure Pursuit、Stanley等传统几何控制做对比；

**📊 数据集**

自研xHUB仿真平台生成的Hermite样条轨迹（直线、圆弧、泊车），以及在BeamNG.tech高保真仿真中测试不同拖车几何和重量；

**📈 对比分析**

与Pure Pursuit、Stanley及文献中的最新方法对比；在直线、圆弧和泊车场景中的横向误差分别约0.03 m、0.02 m、0.05 m，航向误差<0.01 rad；与现有方法相比误差显著降低，性能与最先进方法相当或更好；

**⚠️ 局限性**

依赖无滑移假设；倒车时最大曲率受限，需要前进校正；纵向控制对速度变化敏感；缺乏真实车辆实验验证，传感器/执行器延迟未充分考量。

---

## 383. Performance Analysis of Low-Order, GPU-accelerated Finite Element Kernels using Kokkos

**arXiv ID:** 2609.21681 | [PDF](https://arxiv.org/pdf/2609.21681v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e`

---

## 384. TERMon: Detecting Persistent Behavioral Threats in Edge AI via Hardware-Native Ternary Runtime Monitor

**arXiv ID:** 2609.21713 | [PDF](https://arxiv.org/pdf/2609.21713v1)

**作者:** Arish Sateesan `[一作]` (Aalborg University), Edlira Dushku `[通讯]` (Aalborg University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种轻量级硬件运行时监测器 TERMon，用于检测边缘 AI 推理中的行为异常。

**💡 创新点**

创新点在于将可信行为范围映射为温度编码的三元模式，并通过 TCAM 兼容匹配实现固定两周期延迟的硬件判定。

**🔧 技术方法**

采用温度编码、三元内容地址匹配、AXI4‑Lite 外设与 FPGA 现场可重构逻辑实现。

**📊 数据集**

在 CIFAR‑10（含 FP32/FP8 权重）以及 SVHN、CIFAR‑10‑C 等数据集上进行实验。

**📈 对比分析**

与连续 Mahalanobis 参考检测器以及全局/类条件范围基准对比，TERMon 在有害权重错误上实现了 88% 的平均检测率，误报率约 1%，硬件资源仅 3,733 LUT，延迟两周期。

**⚠️ 局限性**

对 OOD 和对抗样本的检测能力有限，仅在单一网络架构与随机权重失效上验证，缺乏针对性攻击与多模型适用性。

---

## 385. Submodular Maximization over Bipartite Perfect Matchings and Matroid Intersection Bases

**arXiv ID:** 2609.21696 | [PDF](https://arxiv.org/pdf/2609.21696v1)

**作者:** Chandra Chekuri `[一作]` (University of Illinois Urbana-Champaign), Rico Zenklusen `[通讯]` (ETH Zürich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在两个互补基（matroid）交集中最大化单调子模函数（submodular function）的问题，尤其关注子模完美匹配（bipartite graph）这一特殊情况，并给出近似算法与硬件下界。

**💡 创新点**

创新点：
- 将该问题与子模定向图航行（Submodular Orienteering）建立近似等价性，从而在准多项式时间内获得 Ω(1/ log|E|) 的近似；
- 设计基于局部搜索的多项式时间双准则近似（bicriteria）算法，在保持几乎完整基数的前提下获得 (1/2 – ε) 的价值近似；
- 对公平子模最大化（Fair Matroid Monotone Submodular Maximization）给出新的多项式与准多项式算法，并提供硬件下界。

**🔧 技术方法**

核心技术：
- 通过基数截断使得 K 为最大公共基数；
- 构造基数交换图（matroid intersection digraph）并分解为可行交换路径/环；
- 使用残差子模函数和长度函数将基交换问题转化为子模航行；
- 利用已有的子模航行算法（含准多项式和多项式版本）得到交换；
- 局部搜索中对改进交换进行阈值化，保证卡数约束的松弛；
- 通过分层图和完美匹配构造证明子模航行的硬件下界。

**📊 数据集**

本工作为理论分析，不依赖具体数据集；所有结果均为多项式/准多项式时间的理论复杂度。

**📈 对比分析**

与之前仅得到常数因子近似（对普通匹配）或 Ω(ε) 近似（对完美匹配）相比，本工作在准多项式时间内实现了 Ω(1/ log|E|) 的近似，并在多项式时间内提供了 (1/2 – ε) 的双准则近似。硬件下界表明在准多项式时间内进一步改进至 o(log log n / log n) 仍不可能，显示了结果的相对紧迫性。

**⚠️ 局限性**

局限性：
- 第一个结果仅在准多项式时间内；
- 双准则近似仅保证近似基数，无法得到严格的大小约束；
- 只针对两基交集及其特殊化子模完美匹配，非一般图匹配问题仍未覆盖；
- 公平最大化结果中 r 为常数时的多项式算法，r 为任意时仅有准多项式版本；
- 硬件下界基于强假设（NP ⊈ ⋂_ϵ ZPTIME(2^n^ϵ) 与投影游戏假设）。

---

## 386. Notrix: Understanding Machine Learning Solutions Across Computational Notebooks at Scale

**arXiv ID:** 2609.21775 | [PDF](https://arxiv.org/pdf/2609.21775v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 387. Optimization Geometry of Equivalent Brownian RKHS Representations

**arXiv ID:** 2609.21693 | [PDF](https://arxiv.org/pdf/2609.21693v1)

**作者:** Mahdi Mohammadigohari `[一作]`, Gustau Camps-Valls `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在同一有限 Brownian RKHS 中使用节点、增量与谱三种坐标表示时，梯度下降（GD/SGD）与 Adam 等优化算法的几何行为；在节点与谱坐标下证明 GD/SGD 轨迹完全相同，增量坐标对应常数度量的 Brownian 预处理；给出最小二乘问题的条件数上界，并证明标准 Adam 仅在符号置换的正交变换下保持不变。

**💡 创新点**

构造一个完全可控的有限 Brownian RKHS 作为基准，精确分析不同坐标对优化轨迹的影响；首次将增量坐标解释为 Brownian 预处理；给出与网格分辨率无关的最小二乘条件数上界；证明标准 Adam 的最大正交不变群仅为符号置换。

**🔧 技术方法**

利用有限元、RKHS 与 Brownian 协方差理论、DCT‑VIII 谱分解、坐标变换与链式法则，推导梯度映射；使用数值验证（float64）检查矩阵身份、谱、能量与优化轨迹一致性。

**📊 数据集**

主要使用合成实验（均匀网格随机采样）进行理论验证；外部验证数据集包括遥感图像集 EuroSAT 与 Salinas。

**📈 对比分析**

采用映射初始化、相同步长、相同小批量进行比较；数值实验表明节点/谱 GD/SGD 轨迹完全一致，增量 GD/SGD 遵循 1/h 的 Brownian 预处理；标准 Adam 在节点/谱间出现第一步差异；最小二乘条件数保持在 1+A/ρ，且误差低于 10⁻⁸。

**⚠️ 局限性**

局限于一维均匀网格和固定锚点；条件数上界不均匀依赖 A 或 ρ；Adam 结果仅适用于标准 Adam 更新，未涵盖 AdamW 等变体；DCT‑VIII 基础在双重特征空间中非唯一；外部验证仅比较模型类别，而非坐标的等价性。

---

## 388. GUARD: Natural Forgetting in Large Reasoning Models via Guided Answer-Reasoning Distillation

**arXiv ID:** 2609.21677 | [PDF](https://arxiv.org/pdf/2609.21677v1)

**作者:** Zeyu Yan `[一作]` (East China Normal University), Cen Chen `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种针对大型推理模型（LRM）的机器不学习框架GUARD，旨在通过构造自然的遗忘轨迹（CoT+拒绝式答案）实现安全、无泄露的隐私与安全不学习。

**💡 创新点**

创新点在于：1）将不学习视为轨迹替换问题，明确要求生成非泄露的CoT与稳定拒绝；2）引入Guided Trajectory Alignment（GTA）使用离线引导token在冻结模型上构造安全轨迹；3）通过Answer‑Reasoning Distillation（ARD）将引导行为蒸馏为可部署参数；4）提出Natural Forgetting Response Score（NFRS）衡量替换轨迹的结构稳定性、流畅度和无虚假替代。

**🔧 技术方法**

核心技术包括：链式思考（CoT）模型、离线重写器（frontier LLM）生成安全轨迹、连续引导token与block‑normalized cross‑entropy训练GTA、Top‑K KL保持保留知识、参数高效蒸馏（ARD）以及NFRS评估流程。

**📊 数据集**

使用数据集：R‑TOFU（隐私泄露），STAR‑1（有害意图），SQuAD（保留通用知识），MMLU、MATH500、BBH、GPQA用于评估推理能力。

**📈 对比分析**

与传统不学习基线（GA、GD、KL、PO、R²MU）比较，GUARD在隐私与安全不学习任务中均获得更高的A​F​E、C​F​E、Avg.和NFRS得分，且保持了近似原始模型的推理能力；在STAR‑1上，GUARD实现了0.96的平均安全得分和0.89–0.90的NFRS，超过所有基线。

**⚠️ 局限性**

局限性包括：1）依赖重写器的质量，若前沿LLM产出不佳，学生拒绝可能过于保守；2）多阶段管线计算成本高于单目标方法；3）GTA学习非单调，需要验证驱动的动态截取最佳权衡点，无法直接使用训练结束检查点。

---

## 389. PointLAM: Local Attentive Mamba for Efficient Point-based 3D Object Detection

**arXiv ID:** 2609.21780 | [PDF](https://arxiv.org/pdf/2609.21780v1)

**作者:** Xuanming Shang `[一作]` (Shanghai Jiao Tong University), Chao Ma `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种高效点云3D检测框架PointLAM，克服了传统点基方法的下采样与局部建模瓶颈

**💡 创新点**

核心创新是Laplacian Point Sampler（利用隐式离散拉普拉斯高通滤波和双重排序实现结构感知下采样）和Local Hadamard Aggregator（利用临时网格定位邻域并用Hadamard门控实现无参数局部拓扑调制）以及将其与Bi‑Directional Mamba相结合构成Local Attentive Mamba块

**🔧 技术方法**

使用了DevNet编码器、DSS采样、临时三维网格、Hadamard门控、Mamba序列建模以及BEV投影检测头等技术

**📊 数据集**

在nuScenes和Waymo Open Dataset两大公开数据集上进行实验

**📈 对比分析**

与多种Voxel、Pillar和Transformer基线对比，PointLAM在保持或超过VOXEL方法的检测精度（nuScenes NDS≈72.2、Waymo L2 mAPH≈73.6）的同时，参数、FLOPs和推理时延大幅降低（8.6M参数、90.7G FLOPs、93.1ms），在小目标和稀疏场景中表现尤为突出

**⚠️ 局限性**

受限于点云稀疏程度仍会影响特征分布，对极端低密度或极小物体的检测尚有提升空间

---

## 390. CIBuzzBench: A Benchmark for Cross-Lingual Understanding of Chinese Internet Buzzwords

**arXiv ID:** 2609.21722 | [PDF](https://arxiv.org/pdf/2609.21722v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 391. SignGPT: Toward LLM-Mediated Sign Language Interaction through Gloss-Free Translation and Generation

**arXiv ID:** 2609.21709 | [PDF](https://arxiv.org/pdf/2609.21709v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 392. Diffusion-Based Tumor Inpainting for Renal Segmentation under Clinical Data Scarcity

**arXiv ID:** 2609.21698 | [PDF](https://arxiv.org/pdf/2609.21698v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 393. CRISP: Contact-Rich Robotic Simulation Platform with Extensive Geometries and Contact Solvers

**arXiv ID:** 2609.21761 | [PDF](https://arxiv.org/pdf/2609.21761v1)

**作者:** Somang Lee `[一作]` (Seoul National University), Dongjun Lee `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了 CRISP，一款针对高密度、多接触机器人仿真的物理引擎，能够高精度地模拟紧耦合接触和几何形状。

**💡 创新点**

创新点包括：①支持多种几何表示（Mesh、SDF、DSF、TDSF）并提供基于优化的碰撞检测；②直接求解非线性互补问题（NCP），通过增量拉格朗日（CANAL）和子ADMM 两种求解器实现高精度且稳健的接触约束；③在不引入松弛或正则化的情况下实现低侵入、低渗透率的接触模拟。

**🔧 技术方法**

技术手段主要是：优化理论（Riemannian 以及多目标梯度下降）用于碰撞检测；增量拉格朗日（Augmented Lagrangian）以及二阶 Newton 迭代用于接触求解；C++20 + Eigen + 自定义内存池用于高性能并行化；与 MuJoCo/Isaac Sim 对标的自定义 URDF/mesh/SDF 配置。

**📊 数据集**

使用了自制的仿真场景（插销插入、螺栓-螺母组装、顶部堆叠、斜面滑动、关节限制等）而非公开数据集，主要用于评估物理精度和稳定性。

**📈 对比分析**

与 MuJoCo 与 Isaac Sim 通过统一几何、质量、时间步长等参数进行对比；实验显示 CRISP 在渗透深度、接触点一致性、动态响应以及多接触耦合下的稳定性方面均优于对手，尤其在极限公差和大质量比的场景中表现突出。

**⚠️ 局限性**

局限性包括：①增量拉格朗日的 Newton 计算对高自由度系统成本高；②子ADMM 在极端多接触情况下收敛速率相对慢；③SDF/DSF 的数值梯度可能导致接触点误差；④对复杂几何的 mesh 细分仍受限，需进一步优化并行与预条件化。

---

## 394. Verifiable Computation with Trusted Execution Environments and On-Chain Digital Rights Tokens

**arXiv ID:** 2609.21728 | [PDF](https://arxiv.org/pdf/2609.21728v1)

**作者:** Bingle Stegmann Kruger `[一作]` (Frankfurt School of Finance & Management), Co-Pierre Georg `[通讯]` (Frankfurt School of Finance & Management)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了一个基于可信执行环境和链上数字权利令牌的系统，支持数据所有者将私有数据聚合为数据池并授权第三方执行受限代码。

**💡 创新点**

创新点在于将数字权利token化并与可验证代码、TEE远程证明以及链上状态绑定，实现对数据使用的可验证控制而非单纯的访问授权。

**🔧 技术方法**

核心技术包括Intel SGX TEE、RA‑TLS、Solana区块链与SPL代币、WebAssembly/Python执行、密封密钥、Oracle节点与链上智能合约。

**📊 数据集**

实现使用合成或示例数据集，未公开使用真实业务数据。

**📈 对比分析**

以原型演示为主，未进行系统化基准测试；通过小规模数据集展示基本可行性，主要关注首次部署延迟与吞吐量，缺乏详细性能对比。

**⚠️ 局限性**

局限性包括TEE侧信道风险、缺乏多Oracle共识、未支持GPU或高级输出隐私机制、运行时失败处理不足、初始化成本高、尚未完成完整性能评估。

---

## 395. TRACE: Coverage Path Planning for Unknown Environments Using Hierarchical Coverage Tree

**arXiv ID:** 2609.21777 | [PDF](https://arxiv.org/pdf/2609.21777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 396. PSR: Predictive Sensorimotor Representation Learning for Contact-Rich Manipulation

**arXiv ID:** 2609.21753 | [PDF](https://arxiv.org/pdf/2609.21753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 397. ForceTwin: Physics-informed Digital Twins for Robotic Manipulation from Instrumented Human Interaction

**arXiv ID:** 2609.21751 | [PDF](https://arxiv.org/pdf/2609.21751v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 398. ZeroTouch: Tactile-Supervised Visual Contact Estimation for Contact-Rich Manipulation

**arXiv ID:** 2609.21726 | [PDF](https://arxiv.org/pdf/2609.21726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 399. Connections Between Quadratic Transform for Fractional Programming and Schur Complement

**arXiv ID:** 2609.21730 | [PDF](https://arxiv.org/pdf/2609.21730v1)

**作者:** Kaiming Shen `[一作]` (Chinese University of Hong Kong), Wei Yu `[通讯]` (University of Toronto)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

本文建立并证明了二次变换（quadratic transform）在分数规划（fractional programming）中的运用与矩阵分析中的Schur补（Schur complement）之间的等价关系，并在此基础上推广至可逆分母为奇异矩阵的情况，最终通过该推广对高斯向量广播信道（Gaussian vector broadcast channel）最优最小化问题给出新的最小化-最大化（minimax）分析，实现了广播信道与多址信道（MAC）容量的对偶性证明；

**💡 创新点**

主要创新点包括①从Schur补推导二次变换及其逆推导；②将Schur补的LMI条件与二次变换的辅助变量对应起来，揭示其MMSE估计意义；③在分母奇异时引入广义逆并给出相应的通用二次变换，进而解决最小化-最大化问题中的奇异协方差；

**🔧 技术方法**

主要技术手段为矩阵分析中的Schur补理论、线性矩阵不等式（LMI）与通用逆（generalized inverse）理论、Lagrange乘子与KKT条件、MMSE估计理论以及对偶性分析；

**📊 数据集**

无实验数据集，本文为理论推导与分析研究；

**📈 对比分析**

论文不包含实验对比，性能评估通过理论证明完成，证明了所提方法在广播信道与MAC之间能够保持容量等价，无需对空间子空间进行显式分解；

**⚠️ 局限性**

局限性在于所需满足的范围条件（ℛ(A)⊆ℛ(B)或ℛ(H)⊆ℛ(D)）以及对广义逆的依赖，且论文主要聚焦于理论框架，缺乏对实际信道模型与数值实验的验证。

---

## 400. Integrating Approximate Logic Synthesis into Approximate High-Level Synthesis

**arXiv ID:** 2609.21697 | [PDF](https://arxiv.org/pdf/2609.21697v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

---

## 401. Bilevel Optimization of Topology and Hyperparameters (BOTH)

**arXiv ID:** 2609.21758 | [PDF](https://arxiv.org/pdf/2609.21758v1)

**作者:** Suryanarayanan Manoj Sanu `[一作]` (Delft University of Technology), Alejandro Marcos Aragón `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

本文提出了一种基于双层优化的拓扑优化（TO）框架，通过对TO过程进行自动微分得到“超梯度”，实现设计变量与超参数的同步优化；

**💡 创新点**

创新点在于将超梯度与TO耦合，使用伪一阶近似、热身启动、LL步骤自适应和梯度裁剪等技术，使得仅需少量LL迭代即可获得有效梯度，并可扩展到数千维超参数空间；

**🔧 技术方法**

主要技术包括自动微分（AD）在双层优化中的应用、梯度裁剪与学习率调度、热身启动、LL迭代自适应、伪一阶超梯度近似以及基于BFGS/Adam的优化器；

**📊 数据集**

实验数据集涵盖标准TO基准：SMD‑1、两柱拉伸/压缩问题、MBB梁的合规性最小化（含神经网络参数化）、L‑形支架的应力约束等；

**📈 对比分析**

与传统的贝叶斯优化（BO）在低维超参数（如学习率、Heaviside锐度、SIMP指数、滤波半径）上进行对比，结果显示在相同计算预算下，双层超梯度方法在UL目标和设计质量上匹配或优于BO；在高维超参数场景（每个单元独立锐度）亦实现了可比收敛速度；

**⚠️ 局限性**

局限性包括仅适用于连续超参数、需要可自动微分的TO实现、搜索过程是贪婪且对初始超参数敏感、易受超梯度数值不稳定影响、无法直接处理离散算法选择、以及对超参数范围的人工限制。

---

## 402. Balanced Prompt Adaptation against Entropy-Induced Collapse for Test-Time Binary Segmentation

**arXiv ID:** 2609.21743 | [PDF](https://arxiv.org/pdf/2609.21743v1)

**作者:** Zhengshan Wang `[一作]` (Shenzhen University), Weiping Ding `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在不平衡二值分割任务中，基于熵最小化的测试时适应容易导致掩码崩溃，并提出了 BAPA 方法来解决该问题。

**💡 创新点**

提出了类平衡锚点 (CBA) 与动态提示适应 (DPA) 两模块的组合，实现了仅在文本提示上进行的平衡动态更新。

**🔧 技术方法**

使用了冻结的 CLIP‑基 Dense VLM、类平衡锚点选择、提示残差微调、共享偏移分析及局部漂移理论。

**📊 数据集**

在四个跨域数据集上评估：ISIC 2017、PASCAL VOC 2012、DUTS‑TE 与 Oxford‑IIIT Pet。

**📈 对比分析**

与多种基线（如 TENT、EATA、TPT、MLMP 等）在同一单图像测试时适应协议下比较，BAPA 在所有数据集均获得最高平均 Dice 分数，并显著优于最强对照。

**⚠️ 局限性**

仅针对单类别二值分割，依赖预训练 VLM，并未在多类别或不同网络架构上验证；对极端像素不均衡可能仍有限制。

---

## 403. A Framework to Quantify the Probability of Future Cyber Loss Events

**arXiv ID:** 2609.21717 | [PDF](https://arxiv.org/pdf/2609.21717v1)

**作者:** Siem Peters `[一作]` (Mnemonic), Martin Eian `[通讯]` (Mnemonic)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出LEFSA框架，将失效事件频率(LEF)估计转化为机器级CLE预测，并通过层级聚合得到组织级、服务级和业务流程级的概率性风险估计。

**💡 创新点**

创新点包括：① 机器级概率预测与层级聚合相结合的全新LEF估计方法；② 支持在机器层面上校准概率并保留依赖结构；③ 提供可解释、可操作的多层级风险度量；④ 讨论并提出基于copula的动态依赖建模思路。

**🔧 技术方法**

使用了XGBoost、逻辑回归、随机森林和Histogram Gradient Boosting等模型；采用Platt/Isotonic校准、SHAP特征重要性分析、滚动窗口验证、ECE与校准斜率评估；并在讨论中引入copula理论用于依赖建模。

**📊 数据集**

实验使用了23家企业的Microsoft Defender for Endpoint（MDE）MDR数据，覆盖110,924台机器，12周共952,770台机-周期实例，正例比例约0.134%。

**📈 对比分析**

与传统机器学习模型对比，XGBoost在滚动窗口上平均ROC‑AUC 0.90、PR‑AUC 0.23、准确率99.8%；校准误差低（ECE≈0.1%），相较随机分类器提升约45–500倍；其他模型性能略逊，XGBoost表现最佳。

**⚠️ 局限性**

局限性：① 数据为私有且无法公开，限制可复现性；② 仅评估机器级LEF，未完成层级聚合与动态依赖建模；③ 时间窗口仅12周，难以评估长周期或罕见大规模事件；④ 依赖标签可能含误报/漏报；⑤ 未监测或新加入机器导致风险低估；⑥ 跨组织训练需满足隐私与治理约束。

---

## 404. Beyond Benchmark Scores: Auditing Medical Vision-Language Models for Chest X-Ray Tuberculosis Screening

**arXiv ID:** 2609.21763 | [PDF](https://arxiv.org/pdf/2609.21763v1)

**作者:** Mushir Akhtar `[一作]` (Indian Institute of Technology Indore), Mohd. Arshad `[通讯]` (Indian Institute of Technology Indore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

对四种医学视觉语言模型在四个胸部X光数据集上的结核检测性能进行了系统评估

**💡 创新点**

提出了评估规范化框架，系统揭示模型排名、评分可靠性、阈值迁移、负类谱等多维度的可迁移性失效

**🔧 技术方法**

零样本视觉语言模型、Prompt敏感性分析、AUROC/AUPRC、Brier/ECE、阈值传递、监督训练对比等技术

**📊 数据集**

Montgomery、Shenzhen、TBX11K、VinDr-CXR 四个公开胸部X光集

**📈 对比分析**

采用成对AUROC比较、配对DeLong检验、Brier/ECEmetric、阈值保留率等统计方法，发现模型排名随提示/负类/阈值变化而改变，且外部验证性能显著下降

**⚠️ 局限性**

缺乏独立临床验证、负类谱定义不一致、预训练曝光不完全可追溯、样本量和病程多样性有限、仅评估单一疾病任务

---

## 405. ECG Mirage: Revealing and Mitigating the Underutilisation of ECGs in Vision-Language Models for Clinical Prediction

**arXiv ID:** 2609.21755 | [PDF](https://arxiv.org/pdf/2609.21755v1)

**作者:** Jinning Liang `[一作]`, Tingting Zhu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文通过对匹配、错误匹配和无图像三种输入条件的对照实验，首次发现并量化了临床决策模型中“ECG Mirage”现象，即模型在加入ECG图像后并未真正利用患者特定的ECG信息。

**💡 创新点**

创新点在于（1）提出了“ECG Mirage”的概念并用匹配/错误匹配对照框架进行评估；（2）设计了一种轻量级的视觉提示调优（VPT）方案，结合受限视觉提示和条件偏好优化（DPO），有效增强模型对匹配ECG的依赖，显著扩大匹配与错误匹配之间的性能差距。

**🔧 技术方法**

使用的技术包括：多模态视觉‑语言模型（Qwen3.5/3.8、Gemma4、MedGemma1.5），视觉提示调优（VPT）与条件偏好优化（DPO），对比的 LoRA、Prompt Tuning、S4‑MLP 基线；训练中采用了受限注意力掩码和两阶段优化。

**📊 数据集**

数据集为 MIMIC‑IV‑ED 与 MIMIC‑IV‑ECG 组合的 MDS‑ED 人群，包含约6万患者的急诊访视、ICU 入院与24小时临床恶化六项指标。

**📈 对比分析**

与零样本 VLM、LoRA、Prompt Tuning 以及深度学习基线相比，VPT 在匹配 ECG 条件下的平衡准确率分别提升至 ICU 70.6%、恶化 67.5%，并将匹配‑错误匹配差距提升到 16.5%/5.5%，显示出对 ECG Mirage 的有效缓解；但 LoRA 在整体预测性能上仍优于 VPT。

**⚠️ 局限性**

主要限制包括：VPT 仅在匹配 ECG 条件下提升性能，整体绝对准确率仍落后于 LoRA；实验仅基于单一 MIMIC‑IV 公开数据集，可能影响模型在不同人群和设备上的泛化；以及对 EC 领域外的多模态任务的适用性尚待验证。

---

## 406. GenTraceBench: A Benchmark for Tracing Audio Deepfakes Across Pre- and Post-training Stages

**arXiv ID:** 2609.21738 | [PDF](https://arxiv.org/pdf/2609.21738v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 407. AgenticSwarm: Semantic Perception and Adaptive Task Allocation for Heterogeneous Multi-UAV Missions

**arXiv ID:** 2609.21716 | [PDF](https://arxiv.org/pdf/2609.21716v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 408. SFVO: Decoupled Confidence-Guided Stereo-Flow Visual Odometry with Bidirectional PnP

**arXiv ID:** 2609.21754 | [PDF](https://arxiv.org/pdf/2609.21754v1)

**作者:** Kai Zhang `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 SFVO，一种利用预训练立体匹配与光流模型构建的对应驱动深度立体视觉里程计框架，从稠密 3D‑2D 对应直接映射到几何姿态估计。

**💡 创新点**

创新点包括：① 直接使用预训练对应网络生成稠密几何约束，省去从头训练；② 引入旋转–平移解耦的置信度映射，区分远点的旋转信息与近点的平移信息；③ 设计可微双向 PnP 求解器，融合正向与逆向约束，提升鲁棒性。

**🔧 技术方法**

使用的技术包括 UniMatch 预训练对应网络、稠密视差与光流估计、Transformer 置信度预测模块、可微双向 Wahba+PnP 旋转/平移求解，以及图像重建、位姿误差与置信度正则化等多项损失。

**📊 数据集**

实验数据集涵盖 KITTI 视觉里程计、EuRoC MAV 航拍序列以及未见的 UGV 校园序列；训练使用 KITTI 00–08、EuRoC 其余序列，测试包括 KITTI 09–10、EuRoC 9 条序列和 UGV 校园序列。

**📈 对比分析**

通过与 DPVO、BotVIO、ORB‑SLAM3 等方法对比，SFVO 在 EuRoC 和 KITTI 的 RPE、ATE 上均能取得领先或相近性能，尤其在未见 UGV 序列中将 ATE 降低约 60%，显示出强大的跨数据集泛化能力。

**⚠️ 局限性**

局限性在于：仅实现帧间估计，缺乏后端轨迹优化导致长时间漂移；对动态物体的抑制仍依赖置信度过滤；依赖预训练对应网络，模型推理成本相对较高；在极端光照或遮挡场景下性能可能下降。

---

## 409. SpecQuant: Speculative Decoding with Multi-Parent Quantization for Adaptive LLM Inference

**arXiv ID:** 2609.21704 | [PDF](https://arxiv.org/pdf/2609.21704v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 410. RAYA: Learning Where and When to Intervene for Robot Recovery

**arXiv ID:** 2609.21690 | [PDF](https://arxiv.org/pdf/2609.21690v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 411. GraphSkillEvo: Evolutionary Optimization of Graph-Structured Agent Skills

**arXiv ID:** 2609.21749 | [PDF](https://arxiv.org/pdf/2609.21749v1)

**作者:** Rui Sun `[一作]` (City University of Hong Kong), Zhichao Lu `[通讯]` (City University of Hong Kong)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GraphSkillEvo 框架，将大型语言模型代理的技能表示为图结构，并通过进化算子进行优化。

**💡 创新点**

创新点在于将技能拆分为可复用的节点和有向边，形成显式工作流图，并结合种群进化（突变与交叉）在结构化空间中高效搜索。

**🔧 技术方法**

使用了 LLM 生成的图结构技能表示、基于进化算法的四种算子（全局指导突变、图结构突变、全局指导交叉、图结构交叉），以及多代种群演化与验证门控。

**📊 数据集**

在五个基准数据集上进行实验：SearchQA、SpreadsheetBench、DocVQA、LiveMathematicianBench 与 ALFWorld。

**📈 对比分析**

通过与无技能、人工技能、LLM 直接生成技能以及 SkillOpt 对比，GraphSkillEvo 在 14 个模型-基准组合中获得最高分，平均提升 1.76%（GPT‑5.4）或 4.01%（GPT‑5.4‑nano），并在 token 消耗上更低。

**⚠️ 局限性**

局限性包括仅在特定 LLM 与基准上验证，缺乏对更大模型或更复杂图合成机制的探索，且仍需依赖 LLM 生成，未完全解决模型兼容性与动态环境适配问题。

---

## 412. Understanding Engagement and Intrusiveness in Assistive Human-Robot Interaction Using Individual Traits

**arXiv ID:** 2609.21744 | [PDF](https://arxiv.org/pdf/2609.21744v1)

**作者:** Valerio Bo `[一作]` (Institut de Robòtica i Informàtica Industrial), Anaís Garrell `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究通过在模拟医院环境中让32名受试者完成护理物资挑选任务，比较了两种机器人协助策略（一个基于规则的主动跟随系统和一个依赖LLM的保守提示系统），并分析了受试者的人格特质、性别等个体差异对系统偏好、感知亲密度、情绪反应和交互行为的影响。

**💡 创新点**

创新点在于将机器人亲密度与受试者人格特质结合，系统性探究不同干预频率对“侵入感”与用户体验的影响；同时通过对两种对比显著不同的交互策略进行实验，揭示了个体差异在不同控制分配下的表现差异，为实现个性化机器人行为提供经验依据。

**🔧 技术方法**

技术方法包括：使用Mediapipe进行3D姿态估计以获取头部、身体朝向和距离，构建交互FSM和基于LLM（Qwen2.5:7B）的对话驱动模块；配合Vosk语音识别、Piper TTS实现实时语音交互；对交互日志、问卷（Godspeed、SAM、UES、侵入感量表）进行统计与相关分析。

**📊 数据集**

数据集主要由32名受试者的人格问卷（Big Five短表）、交互日志（语音转文本、交互时长、交互次数、位置变更）以及主观评估量表（Godspeed、SAM、UES、侵入感量表）组成。

**📈 对比分析**

实验结果显示，两种系统的总体参与度相近（均约3.6/5），但系统2在侵入感量表上显著低于系统1（p<0.001）。人格维度中神经质与侵入感呈正相关，开放性与激活度负相关；性别差异显著，男性更倾向于选择系统1。系统1与系统2在交互时长、机器人输出次数等客观指标上也呈现不同的相关性，说明不同控制策略调节了人格对交互行为的影响。

**⚠️ 局限性**

局限性包括样本量有限（32人，且未涵盖多次交互）、实验仅限于单一任务场景，且缺乏跨领域验证；此外，机器人行为主要基于预定义规则或LLM推理，未探索更细粒度的情境适应与长期共适。

---

## 413. XCalib Depth-Guided Geometric Optimization for Dense Thermal-Visible Video Registration

**arXiv ID:** 2609.21770 | [PDF](https://arxiv.org/pdf/2609.21770v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 414. Scaling Vision-Language Reward Learning for Robot Manipulation in Parallel Simulation

**arXiv ID:** 2609.21767 | [PDF](https://arxiv.org/pdf/2609.21767v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 415. DRT: Dense Reasoning Trace for Efficient and Grounded Multimodal Reasoning

**arXiv ID:** 2609.21675 | [PDF](https://arxiv.org/pdf/2609.21675v1)

**作者:** Wan Xu `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Dense Reasoning Trace（DRT）范式，用结构化符号轨迹替代自然语言链式推理，显著降低推理过程中的语言冗余与视觉漂移。

**💡 创新点**

创新点：① 将视觉感知与逻辑推导分离，构建视觉证据集合和紧凑推导轨迹；② 两阶段训练——Dense Trace Initialization 与 Trace‑Grounded Reinforcement Learning；③ 通过三视角验证生成可信步骤，并设计 Trace‑Grounded GRPO 奖励框架，实现步级监督与探索激励。

**🔧 技术方法**

技术细节：基于 GPT‑5.1 的教师/验证器，Supervised Fine‑Tuning（DRT‑SFT）、强化学习（GRPO）与结构化奖励，符号化连接符号（→）与视觉证据标记；构建三视角验证管道（最终答案、视觉真实性、推理真实性）。

**📊 数据集**

数据集：SFT 使用 Mulberry subset of Vision‑R1‑Cold（约 200K 条样本）；RL 采用 Vision‑R1‑RL 与 DAPO 组合（8,936 条多模态 + 5,788 条文本）；评测基准包括 MathVista、MathVerse、LogicVista、Video‑Holmes 与 GSM8K。

**📈 对比分析**

比较方法：与 Qwen3‑VL‑8B‑Instruct 及其多种高效推理基线（CoD、ThinkLess、VisionThink）对齐；DRT‑SFT 在五个 benchmark 上提升 1.3 点准确度且 token 效率提升 5.5×；DRT‑RL 在保持准确度的同时将输出 token 减少 80% 以上，QPS 提升，延迟下降，整体性能优于所有对比方法。

**⚠️ 局限性**

限制：仍需依赖强大的教师/验证器，验证过程计算成本高；对极长或极复杂推理路径可能仍出现误差；在部分极难任务中未能突破 Qwen‑Standard 的最高点；未充分探索多模态与文本交叉推理的全局最优策略。

---

## 416. Quadratic Word Equations with a Linear Side: Polynomial Nielsen Graph Diameter and NP-Completeness

**arXiv ID:** 2609.21785 | [PDF](https://arxiv.org/pdf/2609.21785v1)

**作者:** Yuki Yonemoto `[一作]` `[通讯]` (Kyushu University), Yuki Yonemoto (Kyushu University)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了具有一侧线性、另一侧总出现次数不超过两次的单词方程可在多项式步数内通过Nielsen变换得到空方程，从而确立该类方程可判定且属于NP，进而证明其NP‑完整性。

**💡 创新点**

首次给出了此类方程的Nielsen图直径上界O(N^12)，通过将潜能函数与共享变量扩展分离，并将共享扩展段映射到正则方程以应用已知的直径定理，实现了从正则到非正则的中间类分析。

**🔧 技术方法**

采用Nielsen变换、潜能函数、图论距离分析、正则方程映射、Day–Manea的直径定理以及共享扩展与非共享扩展的分段压缩技术。

**📊 数据集**

该研究为理论性工作，无实验数据集。

**📈 对比分析**

与已知正则方程NP‑完整性结果比较，证明此类方程同样NP‑完整；理论上给出可接受路径长度为O(N^12)，即多项式复杂度，但未给出实验性能指标。

**⚠️ 局限性**

结果仅适用于一侧线性且总出现次数≤2的方程，对一般二次方程仍未能证明NP可满足性；直径上界为12次方，可能有进一步改进空间。

---

## 417. Scalable Packet Tracking on FPGAs for Erasure-Coded RDMA over Lossy WANs

**arXiv ID:** 2609.21774 | [PDF](https://arxiv.org/pdf/2609.21774v1)

**作者:** Yicheng Qian `[一作]` (Northeastern University), Nadeen Gebara `[通讯]` (Microsoft Corporation)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种完全在FPGA上实现的多路径擦除码(RDMA)接收器——Cache，利用基于缓存的丢包跟踪替代传统位图跟踪，实现了在宽域网（WAN）上高速、低延迟的包到达跟踪与消息完成；

**💡 创新点**

创新点在于：①将包到达跟踪转化为丢包跟踪，显著降低所需状态；②采用基于条带与路径的异步进度模型，内存需求由BDP转为可控的Jitter；③将跟踪逻辑拆分为预处理、数据管理、完成管理三阶段并行流水线，消除位图设计的读写冲突；④在FPGA上实现高线速（400 Gbps及以上）和高连接数（比传统设计高6×），并与SoC/软件方案进行对比。

**🔧 技术方法**

使用的技术包括：FPGA硬件加速（Intel Agilex 7）、SystemVerilog实现、Quartus综合、基于Set‑Associative缓存的丢包计数存储、异步路径状态向量、基于PSN的条带/路径映射、EC(4,3)/MDS编码、基于硬件的完成事件生成与EC重构（未完整实现）。

**📊 数据集**

实验使用合成的RDMA流量：MTU 4 KiB，模拟多路径多速率（400 Gbps、1.6 Tbps）和不同路径抖动（0.5 ms–4 ms）以及0.1%~1%丢包率的宽域网环境；没有使用公开数据集。

**📈 对比分析**

对比方法：在相同FPGA资源下，将Cache与传统bitmap设计以及软件定义的SDR‑RDMA进行对比；评价指标包括实现后时钟频率、可达线速、每路径吞吐、支持的并发连接数；结果表明Cache在400 Gbps下实现400 Gbps线速，并在1.6 Tbps上保持1.6 Tbps聚合吞吐；并发连接数比bitmap高6×，比SDR‑RDMA高数十倍。

**⚠️ 局限性**

局限性：①缓存设计仍受路径抖动影响，抖动增大需增大缓存；②缺乏完整的EC解码硬件实现，完整链路需软件辅助；③对极端拥塞（大丢包、长抖动）时的缓存冲突和淘汰策略未完整验证；④目前实验仅在FPGA原型上完成，尚未在真实WAN部署验证。

---

## 418. World Modeling in Transformers

**arXiv ID:** 2609.21748 | [PDF](https://arxiv.org/pdf/2609.21748v1)

**作者:** Pierre Beckmann `[一作]` (EPFL), Andre Freitas `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

分析了在曼哈顿随机漫步训练下的 TaxiGPT transformer，揭示其内部存储了高精度的地图并通过位置追踪和目标罗盘实现导航，但因特征叠加导致行为失败。

**💡 创新点**

创新在于将世界模型拆解为多种互补的机制，证明地图可信度与导航性能不必然一致，并提出“affordance packing”缓解叠加干扰的策略。

**🔧 技术方法**

采用机制可解释技术：线性探针、diff‑means、干预实验、旋转窗口、logit 镜头等对内部表征进行提取和因果验证。

**📊 数据集**

使用曼哈顿街区的真实地图以及随机游走、最短路径、噪声最短路径等多种数据集训练 GPT‑2 变体。

**📈 对比分析**

通过自定义的机制指标（解码准确率、法律行动激活、罗盘方向、压缩度等）与行为测试（stress、detour、压缩）对比，发现不同模型在地图解码与导航能力上的差异；大型随机游走模型在指标上与小型模型相当，行为误差主要来自定位失败。

**⚠️ 局限性**

局限在于仅研究有限状态、确定性转换的城市地图，未探讨更复杂动态或非确定性环境；并且叠加干扰仍是模型可靠性瓶颈。

---

## 419. ZYT-World: A Real-Time Controllable World Model for Closed-Loop Autonomous-Driving Simulation

**arXiv ID:** 2609.21712 | [PDF](https://arxiv.org/pdf/2609.21712v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 420. Sandwich-Residuals: Parameter-Efficient Test-time Adaptation of World Models

**arXiv ID:** 2609.21740 | [PDF](https://arxiv.org/pdf/2609.21740v1)

**作者:** Krishnam Soni `[一作]` (Montanuniversität Leoben), Elmar Rueckert `[通讯]` (Montanuniversität Leoben)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Sandwich‑Residuals，一种在保持预训练世界模型冻结的前提下，仅在其预测器前后插入轻量级残差模块进行测试时适应的方法。

**💡 创新点**

创新点在于将适应过程限制在模型接口层面，利用自监督预测误差训练残差校正，而无需更新海量预训练参数，也不需要挑选需要更新的内部块。

**🔧 技术方法**

使用的技术包括自监督的潜在预测损失、Adam 在线更新残差参数、CEM 规划以及多层感知机（MLP）残差映射；在 AdaJEPA 与 DINO‑WM 两种架构上实现。

**📊 数据集**

实验数据集为 AdaJEPA benchmark（4 个视觉控制任务，共 21 条主条件 + 7 复合条件）和 OGBench‑Cube（3D 机器人抓取任务）。

**📈 对比分析**

与 Frozen、AdaJEPA‑FirstBlock、AdaJEPA‑LastBlock 四个基线相比，Sandwich‑Residuals 在 21 条主条件上平均成功率提升至 65.0%，约为 Frozen 的 1.3 倍，接近 AdaJEPA 的 68.2%，且仅更新 1–3% 的参数；在 7 条复合条件上提升至 1.9 倍。

**⚠️ 局限性**

局限性包括：依赖预训练表示在测试时仍足够；仅在仿真环境评估，未考虑真实机器人噪声与时变动力学；对视觉表示的大幅漂移恢复能力有限；残差模块在梯度传播时仍需通过冻结的预测器，导致计算成本不完全下降。

---

## 421. NeuRIO: A Streaming Neural Estimator for Zero-Shot Sim-to-Real Multi-Robot Relative Inertial Odometry

**arXiv ID:** 2609.21707 | [PDF](https://arxiv.org/pdf/2609.21707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 422. When Should Robots Intervene? Balancing Engagement and Intrusiveness in Human-Robot Interaction

**arXiv ID:** 2609.21734 | [PDF](https://arxiv.org/pdf/2609.21734v1)

**作者:** Lavinia Hriscu `[一作]` (Institut de Robòtica i Informàtica Industrial), Anaís Garrell `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过对比连续型机器人关注策略和基于情境的平衡型干预策略，研究了在模拟医院环境中非专家用户完成任务时，机器人干预对用户体验、交互动态、感知侵入性和支持感的影响。

**💡 创新点**

创新点在于首次系统性比较两种干预策略，揭示支持与侵入性之间的权衡，并提供基于多模态行为信号（头部姿态、躯干姿态、距离）的情境感知机制及其对用户体验的实证指导。

**🔧 技术方法**

使用了 Mediapipe 与 Kalman 滤波器实现视觉感知；通过 Qwen‑2.5:7B‑Instruct 生成对话；采用 Vosk 语音识别、Piper 语音合成；机器人平台为 IVO 进行移动与双手操作；并集成多模态融合的参与度估计与干预决策模块。

**📊 数据集**

主要使用的是实验收集的交互日志与问卷数据（SAM、Godspeed、UES、侵入性量表），并未使用公开数据集，而是基于32名非专家参与者在模拟医院仓库完成的任务记录。

**📈 对比分析**

通过对每个参与者的主观评分（情绪、智力感知、侵入性）和客观指标（交互时长、用户输入/机器人输出次数、位置变更次数）进行配对 t 检验和相关性分析。结果显示平衡型策略在侵入性评分上显著低于连续型（p<0.0001），但两者在智力感知与整体参与度上差异不显著；连续型交互更短、干预频率高，平衡型交互更长、用户输入多、机器人输出少。

**⚠️ 局限性**

局限性包括：样本量仅32人，且为非专业参与者；实验环境为模拟医院，缺乏真实临床情境；机器人行为仅限于预定义规则，未实现真正的自适应学习；未考察长期交互或多任务场景；仅使用单一 LLM 与视觉模态，可能忽略语音、手势等更丰富的交互信号。

---

## 423. Visual Proactivity: Enhancing Human-Robot Collaboration Through Intent Communication

**arXiv ID:** 2609.21729 | [PDF](https://arxiv.org/pdf/2609.21729v1)

**作者:** Valerio Bo `[一作]` (Institut de Robotica y Informatica Industrial and Universitat Politecnica de Catalunya - BarcelonaTech), Alberto Sanfeliu `[通讯]` (Institut de Robotica y Informatica Industrial and Universitat Politecnica de Catalunya - BarcelonaTech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了机器人视觉主动性，展示机器人通过运动向人类传达意图并影响其行为。

**💡 创新点**

创新点是提出视觉主动性框架，让机器人通过运动轨迹主动引导人类决策，而非仅仅响应或预测。

**🔧 技术方法**

使用实时人体骨架检测（Mediapipe）、深度学习轨迹预测、行为树（BT）、γ参数融合、GMM与DTW等技术实现和评估。

**📊 数据集**

数据集为自制实验数据，30名参与者在4个目标、3种策略下的手传递实验，使用RGB‑D + LiDAR 记录轨迹。

**📈 对比分析**

通过单盲 within‑subject 实验与 Godspeed 问卷结合 Friedman 非参数检验进行比较，结果表明主动行为在可预测性、流畅度和人类感知上显著优于被动/预测性，并能有效引导人类改变路径。

**⚠️ 局限性**

局限性包括仅在单人实验室环境评估、未测试多机器人或动态障碍、γ阈值手工设定、缺乏长期交互与适应性分析。

---

## 424. PRISM-BN: A Controlled Corpus and Benchmark for Text-to-Parameterized Bayesian Network Extraction

**arXiv ID:** 2609.21673 | [PDF](https://arxiv.org/pdf/2609.21673v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 425. Chronosphere: Space-Time Tessellation of Local Climate Experts

**arXiv ID:** 2609.21872 | [PDF](https://arxiv.org/pdf/2609.21872v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 426. CIPL: A Channel-Aware Framework for Recoverable Privacy Leakage in LLM Agents

**arXiv ID:** 2609.21686 | [PDF](https://arxiv.org/pdf/2609.21686v1)

**作者:** Tao Huang `[一作]` (Minjiang University), Feng Xia `[通讯]` (RMIT University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 CIPL 框架，对 LLM 代理在黑盒条件下的隐私泄露进行通道感知的曝光-恢复评估。

**💡 创新点**

创新点在于将敏感源、选择、组装、执行、观察、提取拆分为统一结构，并通过曝光到恢复的转移来比较不同通道的泄露实现，而非仅看存储标签。

**🔧 技术方法**

使用了黑盒实验协议，结合对内存、检索增强、工具调用以及真实浏览器代理的多通道设置，采集并统计任何泄露率、完全恢复率等指标。

**📊 数据集**

使用合成敏感数据（EHR 记录、检索数据库等）以及公开的 MEXTRA/EHRAgent 资源作为测试集。

**📈 对比分析**

比较方法通过统一的 CIPL 评估协议，报告 CER、AER、EE 等指标；内存通道几乎 100% 完全恢复，检索通道仅 30-55% 完全恢复，工具通道受观察面和供应商影响，Live‑Agent 演示也表现出类似差异；相对传统仅关注存储组件的评估，CIPL 显示更细粒度的泄露模式。

**⚠️ 局限性**

局限包括：仅在有限的 30 次查询预算、固定提示与供应商版本下评估；未覆盖多模态、长会话或更复杂的代理；语义审计样本有限；以及对供应商更新的敏感性。

---

## 427. Listen Before You Speak: Response Planning from Listener Facial Reactions for Conversational Speech Generation

**arXiv ID:** 2609.21683 | [PDF](https://arxiv.org/pdf/2609.21683v1)

**作者:** Yunji Chu `[一作]` `[通讯]` (Sogang University), Yunji Chu (Sogang University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ReACT‑TTS框架，利用对话前1秒的听者面部表情动态来规划回应的情绪与韵律，并通过Grad‑TTS实现语音合成。

**💡 创新点**

首次将听者的即时面部反应作为规划语音风格的上下文，而非简单复制情绪标签，且采用时序编码与目标条件门控实现可解释的风格预测。

**🔧 技术方法**

使用RoBERTa文本编码器、AffectNet预训练的ResNet‑18面部特征提取、Transformer时序编码、目标条件门控、ResponseStyleAdapter以及Grad‑TTS+HiFi‑GAN的端到端合成。

**📊 数据集**

在严格的双人对话筛选下，对MELD语料库进行实验，共计1,117/116/261个规划样本，252条语音样本用于合成评估。

**📈 对比分析**

与仅使用文本上下文的基线对比，Temporal模型在macro‑F1、CCC上略有提升（+0.009）但准确率无显著差异；语音合成的WER/CER与基线相当，听众偏好评测中76%倾向Temporal。

**⚠️ 局限性**

性能提升不显著、统计显著性不足；听者分配不确定、缺乏对话者明确标注；语音合成质量有限，WER/CER偏高，未能证明听者动态真正提升可听性或自然度。

---

## 428. Beyond Counting Blessings: Tracing the Evolution of Gratitude Practices and Technology Needs

**arXiv ID:** 2609.21853 | [PDF](https://arxiv.org/pdf/2609.21853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 429. Reusing Latent Speech Representations for Query-Conditioned Topic Localization in Transcripts

**arXiv ID:** 2609.21844 | [PDF](https://arxiv.org/pdf/2609.21844v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 430. Neural Cellular Automata Learn General Features in their Hidden Channels

**arXiv ID:** 2609.21870 | [PDF](https://arxiv.org/pdf/2609.21870v1)

**作者:** Etienne Guichard `[一作]` (Østfold University of Applied Sciences), Stefano Nichele `[通讯]` (Østfold University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了神经细胞自动机（NCA）的隐藏通道内部动力学，并提出通过注入预训练教师模型的隐藏状态实现少样本迁移学习。

**💡 创新点**

首次将隐藏通道作为可迁移的计算子结构，证明其捕获尺度不变的拓扑原语，可在少样本场景下实现参数高效的迁移。

**🔧 技术方法**

基于局部更新规则的 NCA 模型、隐藏状态注入与时间步反向传播、空间方差与余弦相似度分析以及 UMAP 可视化等技术。

**📊 数据集**

使用 MNIST 及其 25%–100% 不同尺度的变体。

**📈 对比分析**

与约 9800 参数的全连接、全卷积和全局递归模型在 1–100-shot 以及尺度变异上进行对比，NCA 在 1–4 样本/类即可突破 80% 准确率，整体性能优于其它模型。

**⚠️ 局限性**

仅在结构简单的单色 MNIST 上验证，计算量大、时间成本高，BPTT 对多步训练和推理造成瓶颈，缺乏在高分辨率彩色数据集上的验证。

---

## 431. The Weight Is Over - Interactive Diffusion on Consumer GPUs

**arXiv ID:** 2609.21849 | [PDF](https://arxiv.org/pdf/2609.21849v1)

**作者:** Frieder Ganz `[一作]` (Adobe), Maximilian Müller `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在消费者GPU上实现交互式扩散图像生成，提出嵌入翻译器、低位量化与权重量化流式等技术，构建可实时编辑的生成器。

**💡 创新点**

创新点：① 用小文本编码器+翻译器替代大编码器，显著降低权重和延迟；② 重新制定速度/质量/内存三角优化方案（FP8/NVFP4量化 + 权重量化流式）；③ 将上述技术集成到子秒TTFI的交互式编辑器。

**🔧 技术方法**

使用技术包括：Post‑Training Quantization（FP8、NVFP4、FP16）、权重量化流式（On‑GPU weight streaming）、嵌入翻译器（MLP/Attention），ONNX Runtime + TensorRT‑RTX，FP16/F16/FP8 tensor cores，MPS/RTX GPU，Qwen3‑0.6B/4B 编码器，FLUX.2‑klein U‑Net。

**📊 数据集**

数据集与模型：FLUX.2‑klein 图像生成模型、Qwen3‑0.6B 与 Qwen3‑4B 文本编码器、PartiPrompts 文本提示集合，用于 LPIPS/CLIPScore 评估。

**📈 对比分析**

比较方法：与4B Qwen3大编码器对比，量化前后在BF16基准下测 LPIPS、CLIPScore、单步时延与 VRAM。结果：翻译器 + 0.6B 编码器将编码耗时从435 ms降至≈90 ms，内存从8 GB降至≈1.5 GB；FP8 量化将单步时延从105 ms降至73 ms，VRAM从7.3 GB降至5.7 GB；TTFI 从0.47 s 降至0.27 s；权重量化流式在12 GB RTX 4070 Ti 上可将 VRAM 降至 6.7 GB，步时延仅提升 3%。

**⚠️ 局限性**

限制：① 仍需完整大模型保存在主内存，限制统一内存（UMA）系统；② NVFP4 量化对部分内容会有质量下降；③ 翻译器在复杂、长文本提示下的性能略逊于原始4B 编码器；④ 流式技术依赖 PCIe 带宽，低速 GPU 上效率不高；⑤ 需要特定硬件与驱动支持，部署复杂。

---

## 432. Comparing Hand and Controller Avatars with Hand Tracking and Controller-Based Interaction

**arXiv ID:** 2609.21799 | [PDF](https://arxiv.org/pdf/2609.21799v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 433. Restructuring Tree Decision Diagrams

**arXiv ID:** 2609.21842 | [PDF](https://arxiv.org/pdf/2609.21842v1)

**作者:** Christoph Berkholz `[一作]` (Technische Universität Ilmenau), Igor Razgon `[通讯]` (Durham University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了Tree Decision Diagrams（TDD）的理论特性，证明了它们能够被Ordered Binary Decision Diagrams（OBDD）以准多项式尺寸模拟，并提出了一种基于因子和v-tree重构的多项式时间重构算法，使得不同v-tree下的TDD能够快速等价判定，甚至与更强的d‑SDNNF实现多项式时间等价性测试。

**💡 创新点**

创新点：
①证明了TDD到OBDD的准多项式模拟，确认了之前已知的准多项式分离是最优的；
②设计了利用因子最小化和v-tree层次结构的多项式时间重构算法；
③实现了TDD之间以及TDD与d‑SDNNF之间的多项式时间等价性判定。

**🔧 技术方法**

主要技术手段包括：
- v-tree的层次划分与递归构造；
- 因子概念与唯一满足性证明；
- 递归关系求解得到准多项式上界；
- 基于因子最小化的节点合并实现重构。

**📊 数据集**

本文为理论性工作，未使用具体实验数据集；仅在文中提及TiDiDi实现已在2026年模型计数竞赛中使用。

**📈 对比分析**

比较方法：通过理论证明比较TDD与OBDD、d‑SDNNF的尺寸关系，得到TDD到OBDD的准多项式尺寸上界；重构算法的时间复杂度为多项式；等价判定也在多项式时间内完成。未给出实验性能数据，整体性能由理论分析决定。

**⚠️ 局限性**

局限性：
- 重构与等价判定在中间步骤可能产生超多项式尺寸的中间结果；
- 对SDD的最小化和等价判定尚未提供多项式方案；
- 对所有d‑SDNNF实现多项式时间等价判定仍是未解决的开放问题。

---

## 434. PopNavShift: Stress-Testing Social Navigation under Behavioral Population Shift

**arXiv ID:** 2609.21838 | [PDF](https://arxiv.org/pdf/2609.21838v1)

**作者:** Kaizhen Tan `[一作]` (New York University), ChengHe Guan `[通讯]` (NYU Shanghai)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了 PopNavShift 框架，通过匹配的仿真场景评估社交导航策略在不同步行者行为群体下的表现。

**💡 创新点**

创新点在于将行为群体变化作为评价维度，利用匹配干预保留物理情境和控制策略，直接测量不同群体对策略排名的影响。

**🔧 技术方法**

使用 Gemini 3.7 Flash 生成基于 MatrAIx Persona 1M 的模拟人格响应，并通过确定性映射转化为运动参数；在 Navground 中采用 Ped-ORCA 和 Human‑like 后端模拟行人。

**📊 数据集**

使用 MatrAIx Persona 1M（600 条合成记录）以及由 Gemini 生成的八个机器人遇见探测问卷产生的响应；还使用了不同的行人动力学模型。

**📈 对比分析**

通过在 312 个物理情境、八个群体条件和三种控制器下共计 7,488 次机器人运行，比较机器人行驶时间、平均与尾部行人延迟等指标；结果显示行人负担指标对群体变动更敏感，早期让路策略在时间压力增大时被反转。

**⚠️ 局限性**

局限性包括：行为群体为合成数据，未与真实观测校准；映射与语言模型的差异未完全覆盖现实；模拟简化为圆盘行人、无真实感知与多样化交互。

---

## 435. Adaptive Uncertainty-Aware Modeling and Stochastic Radial Basis Function Predictive Control for Personalized Fluid Resuscitation

**arXiv ID:** 2609.21821 | [PDF](https://arxiv.org/pdf/2609.21821v1)

**作者:** Elham Estiri `[一作]` (Kent State University), Hossein Mirinejad `[通讯]` (Kent State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了基于贝叶斯状态空间模型与随机径向基函数模型预测控制的自适应液体复苏框架，能够在实时闭环中实现个体化 MAP 调节。

**💡 创新点**

创新点包括：①使用 UVAE‑SSM 捕获测量噪声产生的随机不确定性；②通过 BNSSM 构建虚拟患者生成器以量化知识不确定性；③设计将不确定性传递到概率约束的 sRBF‑MPC，并实现在线模型微调；④实现了低计算量下的实时控制。

**🔧 技术方法**

主要技术包括：变分自编码器、贝叶斯神经网络、状态空间建模、随机 RBF 模型预测控制、机率约束优化与在线微调。

**📊 数据集**

使用了 16 只绵羊的血流动力学数据作为训练集，并在另外两只绵羊及 10 名健康人（共 30 个实验）上进行验证。

**📈 对比分析**

与传统的 Q‑MPC 和 sQ‑MPC 进行对比，sRBF‑MPC 在四种出血场景下的 MAP 跟踪误差、低于 60 mmHg 时间以及安全性显著优于对手，平均求解时间 <300 ms，符合实时要求。

**⚠️ 局限性**

主要局限：动物数据量有限，模型仅在晶体液复苏场景下验证；缺乏血液制品和血管收缩剂等多重干预；对更大规模人群和多变量血流动力学的泛化能力仍待验证。

---

## 436. A Multi-Cloud View of Internet Background Radiation

**arXiv ID:** 2609.21790 | [PDF](https://arxiv.org/pdf/2609.21790v1)

**作者:** Nils Kempen `[一作]` (University of Münster), Ralph Holz `[通讯]` (University of Sydney)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过在5大云服务商上部署336个IP的被动多云网络望远镜（CLOUD-NT），并与两套经典暗网望远镜（UCSD-NT和SURF-NT）收集的数据进行对比，全面分析了云环境中的互联网背景辐射（IBR）以及扫描活动和RSDoS事件。

**💡 创新点**

创新点包括首次系统性比较云与经典望远镜的IBR特征，证明IBR更受云提供商而非地理位置影响；设计针对不同望远镜规模的阈值调优方法；揭示云专属扫描者对云服务端口的深度垂直枚举，补充传统暗网望远镜无法捕获的威胁视角。

**🔧 技术方法**

所用技术包括Terraform自动化部署、Zeek扫描检测算法调参、FlowTuple流量聚合、CAIDA Corsaro后处理、Kneedle算法寻找阈值elbow、Jaccard相似度计算、IPinfo/FireHOL/ISC SANS等外部名单进行校准，并结合地理与ASN信息进行阈值优化。

**📊 数据集**

使用的数据集包括CLOUD-NT收集的4.57 GiB流量、UCSD-NT 51.18 TiB、SURF-NT 354.2 GiB，以及IPinfo、FireHOL、ISC SANS、OpenINTEL PTR记录、CAIDA前缀-AS映射等补充数据。

**📈 对比分析**

通过对每个望远镜分别调优扫描阈值、计算源IP的Jaccard相似度、统计扫描IP与RSDoS事件的重叠度量，发现云望远镜在扫描IP/云服务端口上的可见度显著高于经典望远镜，但在RSDoS检测上明显不足；同一云提供商内部的IBR相似度高达0.346，跨云差异明显。

**⚠️ 局限性**

主要限制包括测量周期仅两周且云望远镜规模有限，导致稀有随机事件（如RSDoS）检测能力不足；未能排除云侧边缘过滤或DDoS清洗的影响；溢出流量的线性回归仅给出聚合趋势，缺乏单IP细粒度分析。

---

## 437. Per-Aetiology Contrastive Severity Embeddings with Phonological Pseudo-Labelling for Multilingual Dysarthric Speech

**arXiv ID:** 2609.21789 | [PDF](https://arxiv.org/pdf/2609.21789v1)

**作者:** Bernard Muller `[一作]` (Scott-Morgan Foundation), LaVonne Roberts `[通讯]` (Scott-Morgan Foundation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了多语言、病因特定的言语障碍严重度分类，比较单病因模型与混合病因模型的表现。

**💡 创新点**

在匹配架构与训练方案下首次系统验证病因分离训练优于混合训练，并将训练免费语音音位子空间方法产生的伪标签用于多语言数据扩充。

**🔧 技术方法**

采用 HuBERT 基础自监督模型，三阶段对比学习、硬负样本挖掘、序数级三元组损失及跨语言对齐，并结合训练免费音位子空间 d‑prime 伪标签。

**📊 数据集**

汇集 10 种语言、7 种病因共 30 语料库，包括 UA‑Speech、TORGO、SAP、MDSC、EWA‑DB、CDSD、HUNDYSDB 等，包含 1,181 语音样本的伪标签扩充。

**📈 对比分析**

在 speaker‑disjoint、泄漏过滤的保留测试集上使用未加权线性分类器比较，病因模型分别比混合基线提升 22.6%、40.0% 和 32.3% 的宏 F1（分别 0.829/0.715/0.788）。

**⚠️ 局限性**

仅单次训练、缺乏多随机种子评估、测试集中 SAP 贡献过大、伪标签仅在病因内部有效、跨语言与跨语料的泛化尚未充分验证。

---

## 438. Do Personality-Tuned LLMs Make Better Social Agents?

**arXiv ID:** 2609.21857 | [PDF](https://arxiv.org/pdf/2609.21857v1)

**作者:** Tim Krabbe `[一作]` (Stockholm University), Xiaodan Shi `[通讯]` (Stockholm University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对两款小型开源LLM（Qwen2.5-7B-Instruct和Ministral-8B-Instruct）进行基于MBTI人格标签的LoRA微调，并在三种不同紧急程度的对话情景中让模型生成对话，随后使用LLM-as-a-Judge的三评判框架评估生成文本的人格一致性与可控性。

**💡 创新点**

首次将MBTI标签与社交媒体发帖及对话数据结合构成混合训练集，并在小型LLM上探讨LoRA微调对人格角色扮演的一致性与可控性的影响，提出基于多评判者的定性与定量评估框架。

**🔧 技术方法**

技术包括：低秩适配器（LoRA）微调、跨领域数据增强（回译）、多评判者LLM-as-a-Judge（三种LLM评判器）、Krippendorff α 评估交叉一致性、语言检测、Distinct-1/2 词汇多样性计算。

**📊 数据集**

使用Kaggle上的MBTI人格标签数据（社交媒体帖子）以及DailyDialog对话数据；对MBTI标签进行RoBERTa分类器标注，并通过回译进行类别平衡。

**📈 对比分析**

比较方法：对比微调前后的模型在三种情景下的MBTI维度准确率、宏F1、Krippendorff α以及语言多样性。结果显示：微调后的模型在人格一致性上并未超过基线，甚至在多数维度上表现更差；宏F1仅在部分配置下略有提升，整体并无显著改进；基线模型在语言多样性方面表现更佳。

**⚠️ 局限性**

局限性包括：1）数据域漂移——社交媒体帖子与对话语料差异导致模型难以迁移；2）评估方法受LLM评判偏差与随机性影响，交叉一致性低；3）LoRA配置有限，未尝试不同目标模块或更大比例微调；4）缺乏人工专家验证，评判结论可信度不足。

---

## 439. Distributed Balanced Butterfly Counting in Signed Bipartite Graphs

**arXiv ID:** 2609.21848 | [PDF](https://arxiv.org/pdf/2609.21848v1)

**作者:** Kiran Mekala `[一作]` (BITS Pilani Hyderabad Campus), Suman Banerjee `[通讯]` (Indian Institute of Technology Jammu)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种可扩展的分布式算法 D‑BBC，用于高效计数有符号二分图中的平衡蝴蝶子图。

**💡 创新点**

通过优先级驱动的枚举与负/正边判定，结合基于负载感知的顶点分区，实现了唯一计数并显著降低工作量不平衡；同时使用 MPI 与 TBB 混合模型在单机和多机环境下实现高速计算。

**🔧 技术方法**

负载感知的工作估计与贪心分配、优先级驱动的两端筛选、负/正边一致性判定、MPI all‑to‑all 数据交换、Intel TBB 共享内存并行、分布式计数后全局归约。

**📊 数据集**

在 15 个真实二分图数据集上评估，包含大规模网数据如 NX（1 亿条边）、YH（2.5 亿边）等。

**📈 对比分析**

与串行、单机共享内存基线以及改写的分布式基线进行对比；在 13 个能完成的实例上平均 540×（串行）和 11.9×（共享内存）加速；对最大数据集 NX、YH 也能在数十秒内完成，显著优于基线。

**⚠️ 局限性**

对极端稀疏图仍存在负载不均衡问题；通信量随进程数增长而上升，单机内存不足时仍受限；需要预先为边赋予正负号，对原始无符号图需随机赋值。

---

## 440. Supporting Industrial Test-Failure Analysis with LLM-Based Systems: An Experience Report

**arXiv ID:** 2609.21843 | [PDF](https://arxiv.org/pdf/2609.21843v1)

**作者:** Eric Jansson `[一作]` (Mälardalen University), Wasif Afzal `[通讯]` (Mälardalen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对 Westermo Network Technologies AB 的夜间测试失败，构建并评估了单体式与多体式 LLM 代理系统，自动检索测试元数据、设备日志，生成根因分析报告。

**💡 创新点**

首次在同一工业场景下使用相同模型、工具和报告格式，对比单体和多体架构的效果；强调仅靠 LLM 生成完整根因报告可能不如专注证据检索更有价值。

**🔧 技术方法**

使用 Azure AI Foundry、GPT‑5.4、Deep Agents 框架、MCP（模型上下文协议）服务器以及自定义日志检索工具；并采用 ReAct 风格的工具调用与多代理协调。

**📊 数据集**

数据来自 Westermo 的真实测试失败案例（共四个，评测用两个），包括测试元数据、设备映射和多设备日志。

**📈 对比分析**

通过 6 名实践者的问卷与焦点小组（共 120 次系统执行）评估报告质量、可信度、实用性；单体架构平均三倍更快、成本更低，语义一致性更高，但在两种失败场景中无显著质量差异。

**⚠️ 局限性**

局限在于仅评估了两种失败情形、工具集有限、未对比手工 RCA 基线、缺乏独立真值判断，且多体实现可能未最优优化。

---

## 441. Federated Deep Clustering Networks for High-Dimensional and Heterogeneous Data

**arXiv ID:** 2609.21829 | [PDF](https://arxiv.org/pdf/2609.21829v1)

**作者:** Morris Stallmann `[一作]` (Maastricht University), Anna Wilbik `[通讯]` (Maastricht University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FedDCN框架，实现联邦深度聚类，在非IID环境下通过合成数据增强和几何正则化实现鲁棒性。

**💡 创新点**

将DCN迁移至联邦学习，结合合成数据生成与UMAP几何正则化以对抗数据异质性。

**🔧 技术方法**

使用自编码器+K‑means聚类、交替优化、FedAvg、合成数据采样与UMAP几何正则化。

**📊 数据集**

实验数据集包括 MNIST、Fashion‑MNIST 与 USPS。

**📈 对比分析**

与 F‑DEC、FDEC 在 IID 与非IID 场景下对比，性能相近，非IID 下甚至优于基准。

**⚠️ 局限性**

对超参数敏感，合成采样策略有限，客户端数量增多导致性能下降，实验仅限图像数据。

---

## 442. An Agentic Just-in-Time Adaptive Intervention System for Personalized Sleep Support: Proof-of-Concept Study with N of 1 Data

**arXiv ID:** 2609.21805 | [PDF](https://arxiv.org/pdf/2609.21805v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 443. RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding

**arXiv ID:** 2609.21827 | [PDF](https://arxiv.org/pdf/2609.21827v1)

**作者:** Qiao Hu `[一作]` (National Center for Mathematics and Interdisciplinary Sciences), Takehisa Yairi `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在动态树式推测解码中解决了单点(one‑hot)崩溃问题，提出RheoSampling，使动态树既能保持上下文感知的结构，又能进行真正的随机采样并保持无损；

**💡 创新点**

双身份概率解耦：为采样令牌分配代理概率用于树构建、真实概率用于验证，首次用等价类分析证明动态树的无损性；

**🔧 技术方法**

代理概率设计、等价类压缩分析、Optimal Transport 验证算法、稀疏草稿分布、可调的“rheostat”参数m；

**📊 数据集**

Alpaca、GSM8K、HumanEval、MT-bench、Natural Questions、CNN/DailyMail 六大基准；

**📈 对比分析**

与基准 Top‑K 及 EAGLE‑3 进行对比，平均接受率提升 0.14–0.22，速度提升约 3–4%（根据模型与任务略有差异），在多模型、多任务上均优于现有方法；

**⚠️ 局限性**

受温度和 m 参数调优影响；在极低温度下仍可能结构退化；额外采样与验证带来轻微计算开销；未在更大模型或多语言场景下进一步验证。

---

## 444. Fair Prophets

**arXiv ID:** 2609.21826 | [PDF](https://arxiv.org/pdf/2609.21826v1)

**作者:** Paul Duetting `[一作]`, Mathieu Molina `[通讯]` (Tel Aviv University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文研究α‑公平预言家不等式，区分 ex‑ante 与 ex‑post 两种公平度量，并在完整信息与样本访问两种信息模型下给出竞争比分析。

**💡 创新点**

创新点包括：① 证明对所有 α≥0 的 ex‑ante 目标在完整信息下可达到 1/2 的竞争比；② 发现 α>1 时样本访问与完整信息完全分离；③ 证明 ex‑post α<1 时存在常数竞争比（≥e^{−π²/6}≈0.193），而 α>1 时竞争比急剧下降到 1/n；④ 提出 O(n log n) 样本即可在 α∈(0,1] 的 ex‑ante 模型中实现常数竞争。

**🔧 技术方法**

主要技术包括：几何支持函数与 Hahn‑Banach 分离论证、对数期望与 Kullback–Leibler 散度的利用、指数排队法构造偏置概率、样本逼近与尾部估计、以及对数‑幂平均与 Hölder/AM–GM 不等式的结合。

**📊 数据集**

论文为理论性工作，未使用具体实验数据集；所有结果均通过数学证明获得。

**📈 对比分析**

与经典的 1/2 预言家不等式对比：ex‑ante α≤1 保持 1/2；α>1 下竞争比降至 1/n；ex‑post α<1 可实现常数竞争（≥0.193），α>1 只能得到 1/n；因此展现了 α 与公平度量对竞争比的显著影响。

**⚠️ 局限性**

局限性：① 对 α>1 的样本访问仍无法突破 1/n 的上限；② ex‑post α<1 的常数竞争比尚未达到最优（未证 1/2 但有下界 0.193）；③ 对组合资源、非独立分布或更一般的非线性福利函数的扩展尚未完成；④ 需要进一步研究样本数对 α 的最小需求。

---

## 445. MIST: Multimodal Survival Prediction with Genomic-Guided Histology Attention

**arXiv ID:** 2609.21811 | [PDF](https://arxiv.org/pdf/2609.21811v1)

**作者:** Muhammet Sami Yavuz `[一作]` (Technical University of Munich), Benedikt Wiestler `[通讯]` (Technical University of Munich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一种多模态生存预测模型MIST，通过将基因组特征作为查询与TITAN提取的WSI上下文进行交叉注意力，实现更精准的外部泛化。

**💡 创新点**

创新点在于基因组引导的WSI注意力机制，使每个基因特征能够按需检索相关形态学上下文，并结合InfoNCE对齐、可变率基因掩码（VRM）与WSI dropout提升模型鲁棒性。

**🔧 技术方法**

使用技术包括变压器编码器、跨模态注意力、TITAN slide‑level特征、离散时间生存损失、InfoNCE对齐、VRM、WSI dropout。

**📊 数据集**

数据集涵盖TCGA的四种癌症（结肠癌COAD、肾透明细胞癌KIRC、肺鳞状细胞癌LUSC、胶质母细胞瘤GBM）作为训练与验证，外部评估使用CPTAC及独立机构病例。

**📈 对比分析**

通过与Concat、Bilinear、Co‑Attention等标准融合基线以及单模态基线进行5折交叉验证，在外部评估中MIST在C‑index和tAUC均优于所有基线，尤其在COAD‑CPTAC、GBM‑German、KIRC‑CPTAC、LUSC‑US等对齐与稀缺基因组数据场景表现突出。

**⚠️ 局限性**

局限性包括外部样本量相对较小、未整合临床协变量、未使用缺失值插补方法以及未与更大规模基线进行比较。

---

## 446. VideoReloc: Long-Term Indoor Video Relocalization against a Kilobyte-Scale Semantic Scene Graph

**arXiv ID:** 2609.21804 | [PDF](https://arxiv.org/pdf/2609.21804v1)

**作者:** Qianru Li `[一作]` (Technical University of Munich), Yanfeng Zhang `[通讯]` (Huawei Hilbert Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文研究了在室内环境中使用紧凑语义场景图（仅存储类别标签的 3D 盒子）进行长时视频重定位；通过自适应视频片段与地图匹配，生成完整的视频轨迹。

**💡 创新点**

创新点包括：①基于自适应视频片段的重定位范式；②假设优先（hypothesis‑first）注册验证对象三角形；③方向感知优化利用盒面、重力与墙面方向；④运行级决策与剪辑刚性图优化实现跨片段一致性。

**🔧 技术方法**

采用的技术包括：RGB‑D 里程计、SAM 3 目标检测、Map‑Det3D 盒子对齐、三角形匹配、RANSAC、梯度优化、曼哈顿方向约束、重力对齐、剪辑闭合以及姿势图优化等。

**📊 数据集**

实验数据集为 RIO10（10 房间，光照与家具变化）和 ReplicaCAD（基于布局的模拟变化）。

**📈 对比分析**

与 HLoc、ACE‑G、R‑SCoRe、MSG‑Loc、FPFH‑ICP 等基线相比，本文在 1 m/10°召回率上达 73.5 %（因果）/90.6 %（剪辑闭合），显著高于基线（约 45‑50 %）；在 ReplicaCAD 上 61.1 %/74.8 %；地图体积仅 95 kB，远小于对手。

**⚠️ 局限性**

局限性：依赖 RGB‑D 里程计与类别检测，对大规模家具移动或动态对象仍有限制；无法精细对齐对象移动后的位姿；缺乏不确定性建模与盒子更新，且需要明显的重力与曼哈顿结构。

---

## 447. LLM-Generated Feature Pools for Time Series Anomaly Detection

**arXiv ID:** 2609.21801 | [PDF](https://arxiv.org/pdf/2609.21801v1)

**作者:** Youssef Attia El Hili `[一作]` (Huawei Noah's Ark Lab), Corinne Ancourt `[通讯]` (PSL University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种极简的无监督单变量异常检测流水线，利用滑窗统计特征与中位数/MAD鲁棒评分，并在每个领域上通过小样本手工或LLM自动生成的特征子集实现特征选择，最终在TSB‑AD‑U基准上达到了与最佳预训练模型相当的性能。

**💡 创新点**

创新点在于把特征库从静态扩展为任务条件的自适应生成，利用多模态语言模型根据示例窗口自动生成领域特定特征，随后与手工特征联合选取，从而突破传统特征库局限。

**🔧 技术方法**

使用的核心技术包括滑窗统计特征提取、基于中位数/MAD的鲁棒评分、不同特征选择策略（贪婪、top‑k、mRMR）、多模态LLM（Gemini、Claude等）进行特征生成以及VUS‑PR评估。

**📊 数据集**

实验数据集为TSB‑AD‑U的单变量轨道，共9个领域、48条调优序列和350条评估序列。

**📈 对比分析**

与公开排行榜上的最佳无训练（统计/神经）和预训练模型进行对比，手工特征流水线在无训练组中领先约0.09，生成+手工特征组合在全评估上达0.588 VUS‑PR，接近或匹配顶级预训练条目（0.59）。

**⚠️ 局限性**

局限性包括：特征生成的质量依赖LLM与随机种子，生成特征在部分领域仍不稳定；对多变量数据的适用性未验证；与排行榜上最优模型相比仍需进一步细调和实验覆盖。

---

## 448. A Principled Approach to Unsupervised Anomaly Detection

**arXiv ID:** 2609.21800 | [PDF](https://arxiv.org/pdf/2609.21800v1)

**作者:** James Myles `[一作]` (Imperial College London), Yingzhen Li `[通讯]` (Nanyang Technological University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文将无监督异常检测重新表述为贝叶斯逆问题，利用推断每个样本的腐败参数来实现异常检测与定位。

**💡 创新点**

创新点在于提出统一的贝叶斯逆问题框架，能量分解得到可解释的异常分数，并能够推断具体的腐败参数，统一并扩展了多种现有方法。

**🔧 技术方法**

采用贝叶斯推断、能量函数分解、特征空间生成式模型、后验估计器（q_ϕ）以及与 PaDiM、CFLOW‑AD 等方法的对比分析。

**📊 数据集**

实验使用 MorphoMNIST、MVTec AD、脑 MRI（BraTS‑T1/T2、ATLAS、CamCAN）等公开数据集。

**📈 对比分析**

在 MVTec AD 上通过扩展 PaDiM 提升了 2.3% 的对象分类 AUROC；在脑 MRI 基准上实现了与最先进方法竞争的检测性能，并在定位上获得较高 Dice 指数。

**⚠️ 局限性**

局限包括对合成腐败样本的依赖、对先验设定的敏感性、复杂解剖结构密度建模的难度，以及未进行系统的先验敏感性分析。

---

## 449. Catena: A Comprehensive Software Suite for Large-Scale Connectomics

**arXiv ID:** 2609.21887 | [PDF](https://arxiv.org/pdf/2609.21887v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 450. CASCADE Against Jailbreaks: Combination Across Stages with Controlled Attack-Defense Evaluation

**arXiv ID:** 2609.21793 | [PDF](https://arxiv.org/pdf/2609.21793v1)

**作者:** Jiale Luo `[一作]` (National University of Singapore), Eric Han `[通讯]` (National University of Singapore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型的 jailbreak 攻击，提出 CASCADE 框架，系统评估并组合不同管道阶段（输入过滤、输入改写、输出过滤）的防御方案。

**💡 创新点**

首次在单轮黑盒攻击下，统一攻击成功率定义与实验设置，对防御组合进行分阶段与跨阶段的系统化优化，给出可直接部署的三种场景组合。

**🔧 技术方法**

使用统一的 ASR_@max_q 指标、AlpacaEval 质量评估、基于模型大小与延迟的效率度量，并实现可视化管道图与自动配置生成。

**📊 数据集**

采用 JBB-Behaviors（100 个恶意目标）与 HarmBench 判别器进行攻击评估，使用 AlpacaEval 的 Llama-3.1-70B-Instruct 做回答质量测试；测试目标模型包括 9 个开源与专有 LLM。

**📈 对比分析**

在 P1–P4 四个阶段中分别筛选攻击代表、各阶段最佳防御、跨阶段组合，并通过 ASR 降低、效用提升与内存/延迟三维指标对比；推荐组合在安全场景下可将 ASR 降低 97% 以上，且效用提升 ≤ 2%。

**⚠️ 局限性**

局限在于仅覆盖单轮黑盒攻击、三阶段推理时的防御；未考虑多轮与训练阶段防御，且攻击选取基于无防御模型，可能忽略针对特定防御的高效攻击；判断器可靠性与 SOTA LLM 的评估仍待完善。

---

## 451. Morphology-Aware Ambiguity Learning for Wafer Defect Decision Support

**arXiv ID:** 2609.21866 | [PDF](https://arxiv.org/pdf/2609.21866v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 452. RegKT: Interpretable and Robust Deep Knowledge Tracing With IRT-Regularizer

**arXiv ID:** 2609.21791 | [PDF](https://arxiv.org/pdf/2609.21791v1)

**作者:** Samuel Girard `[一作]` (Inria-Saclay), Amel Bouzeghoub `[通讯]` (Telecom-Sud Paris)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种结合IRT正则化的深度知识追踪模型RegKT，以提高模型解释性和鲁棒性。

**💡 创新点**

创新点在于在DKT中加入基于IRT的正则项，使得模型在保持预测精度的同时对学生能力进行可解释估计。

**🔧 技术方法**

使用了LSTM递归网络与IRT一参数模型相结合的损失函数，并通过正则化控制解释性与性能的权衡。

**📊 数据集**

在Fractions、RoboMission、ASSISTments2009以及合成BKT和M-IRT等数据集上进行评估。

**📈 对比分析**

与传统DKT、IRT和BKT相比，RegKT在AUC和准确率上略优或持平，且在小样本场景下表现更稳健。

**⚠️ 局限性**

局限在于模型解释性提升后仍需手工可视化，且在大规模数据集上的可扩展性和计算效率尚未充分验证。

---

## 453. Comparing Haptic Feedback Across Hand Tracking and Controllers in VR Object Interaction Tasks

**arXiv ID:** 2609.21869 | [PDF](https://arxiv.org/pdf/2609.21869v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 454. Geometric Mean Pooling for Equal-Weight Multiplicative Coarse-Graining

**arXiv ID:** 2609.21876 | [PDF](https://arxiv.org/pdf/2609.21876v1)

**作者:** Ang-Kun Wu `[一作]` (University of Tennessee), Jingtao Zhang `[通讯]` (Google)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无参数的Signed Geometric Mean Pooling（GMP），通过结合特征符号和几何平均保持乘法结构，并在合成序列、图像分类和分子脂亲性回归等任务上进行评估。

**💡 创新点**

创新点在于将符号相乘与等权几何平均结合，形成一种在层级非重叠池化下保持全局乘积不变的算子，为仅靠加法或极值聚合无法捕捉的乘法信息提供了结构先验。

**🔧 技术方法**

使用GMP算子进行局部和全局池化，构建了粗粒化分析框架；在合成数据上开展分类/回归实验；在MNIST、Fashion‑MNIST、CIFAR‑10上改造CNN架构；在MoleculeNet Lipophilicity数据集上采用Morgan指纹投影为32×32特征图的2D CNN。

**📊 数据集**

实验数据集包括：1）合成Gaussian/Lognormal序列；2）MNIST、Fashion‑MNIST、CIFAR‑10；3）MoleculeNet Lipophilicity（4,200条化合物的Morgan指纹）。

**📈 对比分析**

与平均池化和最大池化在同一模型架构下直接对比；GMP在全乘法分类任务中达到100%准确率，在乘法回归任务中R²接近1；在图像任务中与平均/最大池化相差不大或略逊，表现取决于池化位置；在分子回归中，全局GMP在辅助目标上优于平均/最大池化。

**⚠️ 局限性**

局限性包括：性能高度依赖表示方式、激活函数（ReLU下对零敏感）、池化位置以及目标参数化；在非乘法任务中并不优于传统池化；在多尺度结构或带噪声的实际数据中需要谨慎选择。

---

## 455. Benchmarking the Explanatory Quality of Open-Weight Vision-Language Models in Face Recognition

**arXiv ID:** 2609.21879 | [PDF](https://arxiv.org/pdf/2609.21879v1)

**作者:** Laurent Colbois `[一作]` (Idiap Research Institute), Sébastien Marcel `[通讯]` (Idiap Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套面向开放权重视觉‑语言模型（VLM）的面部识别基准框架，重点评估模型的可解释性质量（相关性与忠实度），并通过强制输出结构化 JSON 以实现自动化审计；

**💡 创新点**

将解释质量量化为可评估的指标，首次将相关性（仅使用身份稳定特征）与忠实度（避免对遮挡区域进行虚假描述）纳入 VLM 面部识别评估，并通过结构化输出实现可复现、可自动化的性能监测；

**🔧 技术方法**

利用多家开源 VLM（Gemma3、Qwen2.5‑VL、InternVL3）与 constrained decoding 生成结构化解释；使用面部检测/裁剪、EER/Accuracy 计算、词汇表识别不稳定线索及遮挡 hallucination 等技术来量化解释质量；

**📊 数据集**

在 LFW、ARFace、Soteria、CelebA 以及传统面部识别基线 Buffalo‑L 上进行实验，以检验模型的匹配性能和解释质量；

**📈 对比分析**

通过比较结构化与非结构化输出的 EER、Accuracy 与 FTA，发现结构化约束在小模型上导致 EER 上升，但随模型规模增大差距缩小；规模分析显示同等 EER 的模型在相关性与忠实度指标上差异显著，说明解释质量是重要的补充评估维度；

**⚠️ 局限性**

局限包括：相关性指标仅基于预设词汇表，可能遗漏其他不稳定线索；忠实度仅关注遮挡情况，未涵盖其他幻觉；仅使用确定性解码且未探讨温度或多样性对解释的影响；所用数据集可能与 VLM 预训练数据重叠，缺乏更难的验证基准。

---

## 456. Compact but Moving: Intervention-Relevant Geometry in Recurrent World Models

**arXiv ID:** 2609.21787 | [PDF](https://arxiv.org/pdf/2609.21787v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 457. Watermarkable Multi-Draft Speculative Sampling via Poisson Processes

**arXiv ID:** 2609.21858 | [PDF](https://arxiv.org/pdf/2609.21858v1)

**作者:** Yanxiao Liu `[一作]` (Imperial College London), Deniz Gündüz `[通讯]` (Imperial College London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一种多草稿、可嵌入无偏水印的推测采样算法，在保持高采样效率的同时保证水印可检测性。

**💡 创新点**

核心创新在于利用泊松过程构造无通信的多样本耦合，获得停时绘稿不变性（drafter‑invariant），并在耦合层面嵌入无偏水印，突破了原先“不可兼顾”理论的限制。

**🔧 技术方法**

采用泊松函数表示（PFR）与多样本PFR、指数赛机制、伪随机函数以及键控泊松过程等技术实现采样与水印的统一。

**📊 数据集**

在 Qwen2.5-7B、Qwen2.5-0.5B、Llama‑3.1‑8B 等模型上，使用 CNN/DailyMail、ELI5 等公开数据集进行实验。

**📈 对比分析**

与 VSpS、Invariant、MWS、MSE、MSE‑Pseudo、Basic‑UWM 等基线对比，MPFR 在平均接受步数（AATPS）与水印检测指标（ANLPPT、TPR@1%FPR）上均优于或匹配最强基线，同时保持生成质量不变，显示出更优的效率‑水印平衡。

**⚠️ 局限性**

仍缺乏完整的理论框架来刻画目标‑草稿通信、水印强度与采样效率之间的全局权衡；实现中耦合与水印 bookkeeping 引入轻微额外开销，未来可进一步优化。

---

## 458. EnterpriseVal: Quantifying the Efficacy, Reliability and Value of Generative AI in the Enterprise

**arXiv ID:** 2609.21841 | [PDF](https://arxiv.org/pdf/2609.21841v1)

**作者:** Abbas Raza Ali `[一作]` (Citigrounp, Inc), Moona Zahid `[通讯]` (NVIDIA Corporation)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了EnterpriseVal框架，用于在企业环境中对生成式人工智能进行系统化评估，涵盖用例定义、冻结配置、度量指标、专家打分、LLM判定、门控与价值模型；

**💡 创新点**

创新点在于：①将公共基准与企业决策桥接，提供可操作的两层门控与效能指数；②构建覆盖可信度、效用、效率、可靠性、保障与监督的六类度量目录；③采用校准的LLM-as-judge与预测加权推断，降低人工成本并保持偏差可控；④将评估结果映射到模型风险管理与法规要求的治理脊柱；

**🔧 技术方法**

使用的技术包括：formal use‑case tuple & frozen configuration，检索增强生成，prompt engineering，工具调用，guardrails，安全日志；blinded expert grading，Cohen’s κ/Krippendorff’s α 统计；LLM-as-judge预测‑加权推断（PPI），Bootstrap/cluster CI；两层门控算法，效能指数计算，价值与风险模型（年化收益、损失预期等）；

**📊 数据集**

数据集：在一家全球性系统重要银行的三大工作流中抽取任务：A）控制评估（A2/C3），B）信用备忘录草稿（A1/C3），C）程序转化（A1/批量/C2）；每个工作流均采样任务集合，配备专家金标准和评价量表；使用 Gemini 2.5 Flash/Pro、Llama 4 等模型，结合检索与工具链；

**📈 对比分析**

比较方法：对每个指标按门槛阈值进行两层评估（外部/内部），并计算标准化得分与效能指数；在信用备忘录案例中，Gemini 2.5 在引文精确度、关键要素捕获、误报率等指标上优于 Llama 4，后者在指导召回率更高；程序转化中，v2 版在人工精炼时间上提升 89%。但因缺乏置信区间与多次重复，无法给出统计显著性结论；

**⚠️ 局限性**

局限性：①缺乏样本量、置信区间与一致性统计，导致门槛评估仅为示例；②未记录模型调优的“冻结”状态或“匹配”条件，无法确保评估可复现；③效率提升基于估算，未进行对照实验；④未对代理系统（A3）进行验证，框架扩展性待检验；⑤金标准单一专家视角，可能不具代表性；⑥门槛与权重为机构特定，跨机构可比性有限。

---

## 459. A Sim-to-Real Integration Pipeline for Training and Deployment of Chunk-Based VLA Manipulation Policies

**arXiv ID:** 2609.21817 | [PDF](https://arxiv.org/pdf/2609.21817v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 460. Contact-Rich Motion Planning via GPU-Parallel Mode Evaluation

**arXiv ID:** 2609.21803 | [PDF](https://arxiv.org/pdf/2609.21803v1)

**作者:** Jiayun Li `[一作]` (TU Darmstadt), Georgia Chalvatzaki `[通讯]` (TU Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了CoMET框架，利用GPU并行评估大量显式接触模式并结合贪婪模式扩展来解决接触丰富运动规划问题。

**💡 创新点**

创新点在于将固定结构共享表示与增量式DDP/AL轨迹优化结合，实现高吞吐量并行求解；采用批量同步的贪婪扩展策略，以GPU为核心的并行搜索大幅提升了搜索效率。

**🔧 技术方法**

使用GPU加速的增量式DDP/AL轨迹优化、混合精度计算、共享固定结构的CUDA池、批量同步的贪婪模式扩展以及CUDA并行求解器。

**📊 数据集**

实验数据集包括Graesdal等人提出的平面推送基准（36,465/6,297种模式）和30个双手非抓取翻转L形块的实例（约600k种模式）。

**📈 对比分析**

与GCS‑SDP、IMPACT、CMA‑ES、MCTS+IPOPT等基线比较，CoMET在推送任务上大部分实例达到完整枚举水平且平均规划时间比GCS‑SDP快4–5倍；在翻转任务上在8段时完成所有30个实例，成功率高于MCTS和均匀采样，整体性能优于或与现有方法相当。

**⚠️ 局限性**

局限性包括对GPU硬件和高吞吐量求解器的依赖；在极大模式空间下仍受评估预算限制；仅在准静态动力学模型下验证，可能在更复杂动力学或感知误差下表现下降。

---

## 461. FPT=PTIME for Homomorphism Problems on Sparse-Incidence and Bounded-Independence Patterns

**arXiv ID:** 2609.21840 | [PDF](https://arxiv.org/pdf/2609.21840v1)

**作者:** Matthias Lanzinger `[一作]` `[通讯]` (TU Wien), Matthias Lanzinger (TU Wien)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了在模式超图具有限定参数λ（即入度退化度或主图独立数对数）时，模式同构问题的参数化可判定性与多项式时间可解性等价，并给出了同构计数的精确计数可多项式性结果。

**💡 创新点**

主要创新在于给出了一条通用的宽度比较定理，证明在任何超图中，分数分层树宽（fractional hypertree width）与自适应宽度（adaptive width）仅相差对数因子λ，并将此结论推广到结构类、计数以及多种中间Polymatroid宽度。

**🔧 技术方法**

采用了分数平衡分离子框架、极小化-最大化（minimax）论证、Korchemna等人的分数分离子逼近与舍入定理以及递归树分解构造等技术。

**📊 数据集**

论文未使用标准公开数据集，而是以理论构造的超图族和关系结构族为研究对象。

**📈 对比分析**

通过比较不同宽度参数（fractional hypertree width、submodular width、adaptive width等），证明在λ有界类中它们的值相差至多对数因子；进而得到在该类中参数化判定问题与多项式时间可解性等价，以及在同类下精确计数问题可在多项式时间完成。

**⚠️ 局限性**

主要局限在于对λ无界时仅能得到指数因数的上界，且无法解决私有交叉（private intersection）条件下更强的比较；同时结果依赖于ETH假设。

---

## 462. Object Detection Benchmarks are Incomplete: The Role of Label Errors and Annotation Uncertainty

**arXiv ID:** 2609.21822 | [PDF](https://arxiv.org/pdf/2609.21822v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 463. Touvigation: Embodied Adaptive Object Acquisition for Blind and Low-Vision Users in Unfamiliar Indoor Environments

**arXiv ID:** 2609.21828 | [PDF](https://arxiv.org/pdf/2609.21828v1)

**作者:** George Xi Wang `[一作]` (Stony Brook University), Jing Qian `[通讯]` (Tongji University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 Touvigation，一种面向盲人与低视力用户的低延迟、连续化、体感化对象获取系统，能够从定位到手部触碰全程指导用户完成目标物体的获取。

**💡 创新点**

创新点在于：① 采用多阶段嵌入式指导（身体中心化的方向、步数与触感）实现从走路到手部抓取的自然过渡；② 结合 SLAM 与 LLM 的多视角投票方法，使目标在三维空间中持久锁定并持续更新；③ 通过实时手部姿态和 LiDAR 深度实现手部级别的引导与触感确认，显著降低认知负荷与错误率。

**🔧 技术方法**

使用的核心技术包括：iPhone 17 Pro 的 LiDAR + IMU + ARKit 进行 SLAM 与体姿估计；YOLOv8 进行物体检测；OpenAI GPT‑5.2 进行多视角投票与语义推理；苹果 Vision 框架进行手部关键点检测；自研的时延敏感指导算法（如步数个性化、时钟方向映射）和低延迟语音交互。

**📊 数据集**

数据集方面，作者在实验室使用 YOLOv8 预训练模型进行检测；对用户研究采用 12 名盲人（Level 1）进行室内实验，涉及 2 个不同房间的 6 个目标位置，共 144 次试验；并结合 8 名参与者的访谈收集定性数据。

**📈 对比分析**

与主流多模态助手 Doubao 以及无辅助搜索进行对比。结果显示：Touvigation 成功率 100%（Doubao 58%，无辅助 85%）；平均完成时间 77.5 s（Doubao 170.1 s，未辅助 105.9 s），NASA‑TLX 工作负荷显著降低；在空间意识、信任与安全感等主观指标上亦明显优于两者。

**⚠️ 局限性**

局限性包括：仅在 LiDAR‑iPhone 环境下验证，缺乏对移动家具、光照变化、网络延迟等动态场景的鲁棒性；样本仅为 12 名 Level 1 盲人，缺乏对低视力及不同使用习惯的泛化评估；系统依赖云端 LLM 推理，可能涉及隐私与实时性问题。

---

## 464. SFPF: Spatio-Frequency Polarization Fingerprint for Anomalous Wireless Device Detection

**arXiv ID:** 2609.21873 | [PDF](https://arxiv.org/pdf/2609.21873v1)

**作者:** Xiaoxuan Huang `[一作]`, Dong Wei `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了时空频率极化指纹（SFPF），用于周期性非侵入式检测无线设备的硬件异常。

**💡 创新点**

创新点在于将极化指纹从单一观测方向扩展为联合频率与方向的多维表示，并通过敏感度分析证明同一硬件改动在频域与空间域会产生非均匀偏差。

**🔧 技术方法**

技术方法包括特征模态理论推导极化响应、方向与频率采样、构建SFPF张量、使用坐标感知集网络（ADNet）进行特征编码与聚合，并采用马氏距离阈值进行异常检测。

**📊 数据集**

使用的实验数据集为：模拟仿真得到的多角度、多频点极化响应；以及基于USRP X310的实际测量数据，包含10台设备在原始硬件与七种硬件更换场景下的SFPF。

**📈 对比分析**

与传统RFF和单方向PF相比，SFPF在相同观测预算下，归一化距离提升17.7%、Fisher分数提升45.8%、类内外比提升11.3%；实验中在15–20 dB时段，SFPF异常设备F1分数达到87.3–90.4%，AUROC 85.4–95.5%，显著优于RFF（≤25%）和PF（≤58%）。

**⚠️ 局限性**

主要局限包括：完整时空频率网格采集耗时135.45 s，采样密度高；未来需引入敏感度引导的稀疏采样以降低时间成本。

---

## 465. TrialAtlas: Multi-Agent Research Organization for Clinical Trial Design and Optimization

**arXiv ID:** 2609.21859 | [PDF](https://arxiv.org/pdf/2609.21859v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 466. AutoRecLab: Describe the Experiment, Get the Code!

**arXiv ID:** 2609.21863 | [PDF](https://arxiv.org/pdf/2609.21863v1)

**作者:** Moritz Baumgart `[一作]` (University of Siegen), Joeran Beel `[通讯]` (University of Siegen)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了AutoRecLab，自动将自然语言研究提示转化为可执行的推荐系统实验代码，并在演示中完成了显式到隐式反馈转换实验和多算法基准实验。

**💡 创新点**

首次公开在RecSys领域构建完整的研究自动化工作流，结合需求工程、原型化、检索增强生成、静态类型验证和执行引导树搜索，实现从提示到完整实验的端到端自动化。

**🔧 技术方法**

使用Python、LLM（GPT‑5.4‑mini）、检索增强生成（RAG）+模型上下文协议（MCP）检索文档、OmniRec元框架、静态类型检查和执行反馈驱动的树搜索。

**📊 数据集**

使用MovieLens 1M、Amazon2018 MusicalInstruments、Amazon2018 VideoGames等公开数据集。

**📈 对比分析**

通过在六种算法（如ItemKNN、ImplicitMF等）上对不同阈值的显式→隐式转换进行NDCG@10、Precision@10评估；实验成功率约89%，平均成本≈1 美元，平均运行时间从几十分钟到数十小时不等。

**⚠️ 局限性**

仅支持OmniRec兼容的库和数据集；树搜索耗时长；生成代码质量有时不具备科学严谨性；缺乏对实验设计和结果解释的自动评估，需人工监督。

---

## 467. LunaDrive: A Delay-Compensated High-Voltage GaN FET-Based Motor Driver for Dynamic Robots with Flat BLDC Motors

**arXiv ID:** 2609.21818 | [PDF](https://arxiv.org/pdf/2609.21818v1)

**作者:** Sota Yuzaki `[一作]` (University of Tokyo), Kei Okada `[通讯]` (University of Tokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并实现了一款名为LunaDrive的紧凑型GaN FET驱动电路，用于在高于标称电压的条件下驱动高功率平板BLDC电机，验证了其高速度、持续和峰值电流能力，并演示了在动态机器人负载提升实验中的应用。

**💡 创新点**

创新点包括：①使用GaN FET实现高电压与大电流的双重可行性；②在驱动板和控制板中实现极短厚度（8.51 mm）和高功率密度；③针对高电频率下的系统延迟采用编码器内部DAEC和MCU侧延迟补偿，显著提升电机速度；④通过有效电容选择与散热设计实现无散热片时仍可达13 A连续电流，峰值80 A。

**🔧 技术方法**

技术手段包括：GaN FET半桥驱动、SVM空间矢量调制、FOC场定向控制、AS5147U编码器DAEC、STDRIVEG212门驱动、低阻抗多层PCB、精细电容布局与并联肖特基二极管减小反向导通损耗、两种散热方案（扁平与翅片）。

**📊 数据集**

该工作并未使用传统意义上的数据集，而是通过实验测量得到的电压、电流、转速、温升、机械张力等数据作为评估依据。

**📈 对比分析**

通过与市售Gold Solo Twitter驱动（3 A连续/45 A峰值、尺寸47.2×30×19.35 mm）在相同电压区间进行对比。结果显示LunaDrive在无散热片时可达13 A连续电流，厚度仅8.51 mm，峰值80 A；在配备散热片时分别可达28 A（扁平）和30 A（翅片）。最高转速为8890 rpm（3110 Hz），比低压48 V时提升约2.6倍，且实现了高电频率下的稳定控制。

**⚠️ 局限性**

限制主要体现在：①实验仅针对RO80平板电机及U13II KV130负载进行验证，未覆盖更大功率或不同极对数的电机；②高电压下的热管理仍受限于散热片尺寸与形状，进一步升压或更长时间运行可能需要更高效的冷却方案；③系统延迟补偿对编码器采样时序的依赖仍存在，未来在更高速或更大负载情况下需要进一步优化。

---

## 468. Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods

**arXiv ID:** 2609.21815 | [PDF](https://arxiv.org/pdf/2609.21815v1)

**作者:** Wenpeng Zhang `[一作]` (Independent Researcher), Peilin Zhao `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了基于矩阵维度的自适应优化算法 Row‑AdaGrad 与 Column‑AdaGrad，借助行/列 Mahalanobis 范数的在线镜像下降框架，显式推导了矩阵级别的 AdaGrad。

**💡 创新点**

创新点在于通过在线镜像下降的通用理论，首次系统地为矩阵参数推导出自适应比例，揭示了自适应尺度与行/列结构之间的权衡，并给出严格的 regret 上界。

**🔧 技术方法**

使用了在线凸优化、Bregman 矩阵散度、行/列特定的 Mahalanobis 范数、对偶范数分析以及矩阵转置对应性等数学工具，构建算法并完成理论证明。

**📊 数据集**

实验数据集包括 MovieLens 100K 的矩阵分解任务、以及用合成高维高深度 MLP 训练的 synthetic 数据集。

**📈 对比分析**

与标准逐元素 AdaGrad、Adam 以及 Shampoo 等方法进行对比，结果显示 Row‑AdaGrad 在行稀疏或内在关联梯度场下取得更低的 regret，尤其在无归一化深层 MLP 中保持可训练性，性能显著优于逐元素自适应方案。

**⚠️ 局限性**

局限性包括：需要先验了解参数的行/列组织结构；仅使用对角元，无法捕捉更细粒度的参数交互；实验规模有限，缺乏大规模工业级验证。

---

## 469. AcousticDiffusion: Semantically Conditioned Audio-Guided Diffusion Policy for Search-and-Rescue Assistance

**arXiv ID:** 2609.21792 | [PDF](https://arxiv.org/pdf/2609.21792v1)

**作者:** Iana Zhura `[一作]` (Skolkovo Institute of Science and Technology), Dzmitry Tsetserukou `[通讯]` (Skolkovo Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AcousticDiffusion，一种利用预训练音频识别、递归贝叶斯鸟瞰声场和扩散模型，实现机器人在视觉受限环境中根据人类呼叫者声音引导运动的导航框架。

**💡 创新点**

创新点包括：① 将语义化音频识别与递归贝叶斯贝尔维融合生成源级语义贝尔维；② 使用扩散模型直接生成基于声场不确定性的轨迹；③ 通过情绪优先级（distress）实现人类优先导航；④ 在真实四足机器人上无需额外重训即可部署。

**🔧 技术方法**

技术栈包括：预训练 Audio Spectrogram Transformer (AST) 进行语音识别；ODAS 提供方向估计与源分离；递归贝叶斯 BEV 声场更新；条件扩散模型（1D U-Net + DDIM）生成轨迹；ROS2 与 Nav2 进行导航与避障；Ego‑motion 补偿与贝叶斯更新。

**📊 数据集**

数据集：合成声场结合 LibriSpeech 与 RAVDESS（情绪语音）生成 10.24 s 语音窗口；使用真实 ODAS 方向轨迹模拟场景；验证集 414 个窗口；真实机器人评估使用 Vicon 记录的地面真实位姿与声源位置。

**📈 对比分析**

对比方法：在相同声源方向估计与成本地图下，与 A* 与 RRT* 进行规划比较。评估指标包括端点方位误差、窗口方位误差、15°内窗口比例、规划计算时间和最终源距离。仿真中端点误差 11.2°；机器人上 64.9°（低于 A* 的 98.2° 与 RRT* 的 90.4°），规划时间 6.07 ms，最终距离 2.48 m，较经典规划器改善 37%。

**⚠️ 局限性**

局限性：真实机器人误差仍显著，主要受声源定位不准和 Sim‑to‑Real 差距影响；缺乏视觉信息导致对遮挡环境的鲁棒性有限；对声源识别的误判与不稳定对齐仍需改进。

---

## 470. From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention

**arXiv ID:** 2609.21788 | [PDF](https://arxiv.org/pdf/2609.21788v1)

**作者:** Sichang Su `[一作]` (University of Texas at Austin), Lingfeng Sun `[通讯]` (Autel US)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在预训练机器人策略的瓶颈子任务上进行局部强化学习，提升长周期任务完成率。

**💡 创新点**

结合可执行子任务监督、残差RL与成功重加权再训练，实现只在瓶颈处进行RL，并且不需要在线人类干预。

**🔧 技术方法**

采用冻结的VLA基模型、TD3+BC残差学习、可执行程序生成器、VLM和SAM做视觉判断、成功重加权再训练等技术。

**📊 数据集**

主要使用基于人类演示的任务演示集（如ABC-130k用于LEGO）和实时机器人收集的数据。

**📈 对比分析**

与SFT、DSRL、EXPO-FT、RLT等基线在相同机器人滚动预算下对比，取得全任务成功率提升25%以上，具体提升到耳机插入61%、Frank a 95%。

**⚠️ 局限性**

仍需人工标定瓶颈、重置以及有限的任务种类，难以完全自适应未知瓶颈。

---

## 471. SkelWAM: A Skeleton-Guided World-Action Model for Zero-Shot Cross-Embodiment Manipulation

**arXiv ID:** 2609.21983 | [PDF](https://arxiv.org/pdf/2609.21983v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 472. VIRGA: Virtual-Agent-Intermediated Riemannian Geometry for Active-Sensing Air-Ground Coordination

**arXiv ID:** 2609.21883 | [PDF](https://arxiv.org/pdf/2609.21883v1)

**作者:** Fenghe Guo `[一作]` (Tongji University), Junrui Zhang `[通讯]` (Tongji University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个基于双LiDAR观测的空地协同控制框架 VIRGA，用神经几何接口将观测转换为受限 Riemannian 场，利用虚拟代理与递归弹性反馈实现 UAV、UGV 与 gimbal 的实时安全协同，并在仿真仓库与洞穴环境中完成任务。

**💡 创新点**

创新点包括：①观测约束下的神经几何接口，能在线把两源 LiDAR 观测映射为受限 SPD 度量和势能；②虚拟代理作为共享参考，弹性耦合多动力学平台；③递归反馈机制保证运动与观测约束的相容性；④通过安全映射显式处理 gimbal FOV、碰撞、分离等限制，保持闭环安全。

**🔧 技术方法**

使用技术：Riemannian Motion Policies 与 RMPflow 的几何控制框架；点云编码与查询融合的神经网络；双源 LiDAR 受限 SPD 生成与自适应融合；虚拟代理与弹性耦合；平台特定执行映射与安全映射。

**📊 数据集**

数据集：270 对双 LiDAR 采样（126 训练，36 校准，108 评估）来自静态仓库；此外在未重训练的洞穴仿真环境中进行长期压力测试。

**📈 对比分析**

与 Ray‑RMP、Dense analytical field、ColAG 三个基线进行对比，评价指标包括任务/安全完成率、最小距离、碰撞/接触、FOV 失效率以及 sensor‑to‑command 延迟。VIRGA 在 5/5 条件下实现任务安全完成，平均最小距离 0.683 m，0 碰撞，0 FOV 失效，平均延迟 48.8 ms；Ray‑RMP 延迟最低但碰撞/距离差；Dense field 无碰撞但延迟过高导致 FOV 失效；ColAG 延迟最快但出现碰撞和 FOV 失效。

**⚠️ 局限性**

局限性：①仅在仿真环境验证，未考虑传感噪声、校准误差、通信延迟和执行不确定性；②在狭窄通道转弯时 gimbal FOV 受限导致观测失效；③当不使用虚拟代理或递归反馈时会出现死锁或安全失败；④在高动态障碍场景下，虽然能保持安全但可能出现等待时间过长；⑤目前未针对实际硬件平台的功耗与实时调度做深入研究。

---

## 473. The ecological collapse of color: photoreceptor number buys a geometric hue manifold that natural spectra never fill

**arXiv ID:** 2609.21965 | [PDF](https://arxiv.org/pdf/2609.21965v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 474. RecreationWorld: Scalable and Verifiable Environments for Hybrid Computer-Use Agents

**arXiv ID:** 2609.22000 | [PDF](https://arxiv.org/pdf/2609.22000v1)

**作者:** Shuai Bai `[一作]` (Alibaba Group), Bowen Zhou `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了混合计算机使用代理（Hybrid CUA）的研究框架——Recreation，通过在可执行参考应用上进行GUI探索、代码实现与自我验证，形成跨平台（Ubuntu、macOS、Windows、Android、Web）的长期交互与训练环境。

**💡 创新点**

创新点在于：①将可执行参考作为行为 oracle 自动生成可验证的程序化与视觉断言；②构建可重现、长周期的五平台任务库，实现训练经验可选；③评估代理在五个OOD基准上的迁移能力，证明 Recruotion 训练能提升更广泛的接口与代码推理任务。

**🔧 技术方法**

核心技术包括：多平台 GUI/命令行统一工具（MCP），持续执行工作者（Ubuntu、macOS、Windows、Android、Web），自动化测试生成器与验证器，使用大型语言模型（GPT‑6 Astra、Claude Opus 5、Qwen‑3.8‑Max 等）进行轨迹记录、SFT 训练与推理；并利用程序化与视觉判定实现无人工干预的自动评分。

**📊 数据集**

数据集：从公开开源项目收集 35,000 条重建轨迹（每个平台 7,000 条）用于训练；测试集 250 个任务（每个平台 50 条），涵盖多种界面框架与语言；OOD 验证集包含 ProgramBench、GameCraft‑Bench、Vision2Web、OSWorld 2.0、WeaveBench 等。

**📈 对比分析**

比较方法：在冻结的程序化与视觉断言套件上自动评分，评估 10 大模型。结果显示 GPT‑6 Astra 以 % 分数领先；在多平台上通过全部程序化测试的概率高达 %；训练提升最高 17.9 分；与传统单一交互代理相比，混合代理在交互覆盖率、代码规模控制和自我验证方面更优。

**⚠️ 局限性**

局限性：环境固定、缺乏网络/实时服务交互；隐藏测试覆盖有限，无法验证所有交互路径；可能存在预训练阶段已接触参考实现，源代码相似度检查无法捕捉重命名/重构；因此评分仅反映在指定环境与冻结测试集内的功能与视觉忠实度。

---

## 475. A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal

**arXiv ID:** 2609.21996 | [PDF](https://arxiv.org/pdf/2609.21996v1)

**作者:** Hiskias Dingeto `[一作]` `[通讯]` (StackOne Technologies), Hiskias Dingeto (StackOne Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的方法，Probe of Internal Recognition (PIR)，用于识别大型语言模型内部隐藏的知识，借鉴了隐蔽信息测试，通过对比正确选项与干扰项的激活状态来读取模型的识别信号。

**💡 创新点**

创新点在于PIR是一种无参考的读取方法，能够在没有诚实参考模型和标记真相语料库的情况下，识别模型内部的隐藏知识，并且能够区分模型是隐藏答案还是根本不知道答案。

**🔧 技术方法**

使用了隐蔽信息测试的原理，通过对比正确选项与干扰项的激活状态来读取模型的内部识别信号。

**📊 数据集**

使用了多个数据集，包括MMLU、WMDP（生物、化学、网络安全）、ARC-Challenge和TriviaQA等，测试了八个来自五个家族的模型（Gemma、Qwen、Llama、Mistral和Phi）。

**📈 对比分析**

与其他方法相比，PIR在识别隐藏答案的准确率上达到了0.70到0.87，明显高于未知项基线的0.28到0.40，且在各种隐蔽形式下保持可读性，表现出色。

**⚠️ 局限性**

限制在于PIR需要一组候选答案，因此适用于多项选择和重建候选的自由形式生成，但不适用于没有可枚举答案的开放式推理。此外，PIR无法区分被抹去的知识与从未拥有的知识，因为两者都会导致静默的读取结果。

---

## 476. Learning Cardiac Features: ECG Biometrics Across Time and~Exercise

**arXiv ID:** 2609.21962 | [PDF](https://arxiv.org/pdf/2609.21962v1)

**作者:** Luca Thiebaud `[一作]` (Aix-Marseille Univ), Stéphane Delliaux `[通讯]` (Aix-Marseille Univ)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

开发并评估了基于Siamese ResNet的ECG生物识别方法，利用晚期多导融合在运动和跨会话条件下进行验证。

**💡 创新点**

提出了晚期多导融合策略和第一批针对运动状态与时间变化的全面评估，实现了跨数据集的无微调泛化。

**🔧 技术方法**

使用1D ResNet-18 Siamese网络、适应性池化、相互作用特征组合以及几何感知导联选择。

**📊 数据集**

在内部收集的CPET数据集（1651名受试者，包含多会话）以及公开的PTB、CYBHi和Heartprint数据集上进行实验。

**📈 对比分析**

与现有跨会话基线对比，CPET内会话EER仅1.0%，跨会话5.6%；在PTB 2.1%、CYBHi 3.9%、Heartprint 10%等，均实现或接近最优性能。

**⚠️ 局限性**

缺乏足够的公开运动相关数据集，模型可解释性不足，且在不同运动强度和采集条件下的非线性时变特征仍需进一步研究。

---

## 477. Beyond Kinematics: Benchmarking Simulation Fidelity for Muscle-Driven Imitation Learning

**arXiv ID:** 2609.21909 | [PDF](https://arxiv.org/pdf/2609.21909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 478. RACER: Role-Aligned Competence Estimation for Human-AI Routing

**arXiv ID:** 2609.21953 | [PDF](https://arxiv.org/pdf/2609.21953v1)

**作者:** Joshua Strong `[一作]` (University of Oxford), J. Alison Noble `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种角色对齐（Role-Aligned）专家能力估计框架（RACER），通过上下文信息在未见专家上实现实例级的、无类别身份依赖的能力预测，从而在学习退避（Learning to Defer）任务中更精准地决定是否交给专家。

**💡 创新点**

创新点在于：①将专家能力建模为后验可预测的角色相关概率 Γ(x,y,C_e)，既保留了实例适应性，又保持了对类别重标记的共形不变性；②引入非参数最近邻与神经核池化两种实现方式；③在训练时采用严格的二分类交叉熵（proper loss）保证概率校准，并提供理论证明与 regret 误差上界。

**🔧 技术方法**

技术手段包括：图像特征编码器、基于余弦相似度的同角色池化、角色相对摘要（如后验秩、置信度、相似度质量等）、MLP 学习器、概率校准的拒绝器、以及一阶段学习（联合训练）与二阶段训练（先学判别器后学拒绝器）两种损失。

**📊 数据集**

数据集：合成的 PathMNIST（带隐藏亚型与专家扰动）与 CIFAR‑100（大类标签与专家扰动），以及真实专家医学影像数据集 VinDr‑CXR 与 CheXpert（多标注者与多专家）。

**📈 对比分析**

与 L2D‑Pop、IFD、Classifier‑Confidence 等基线比较，RACER 在路径医学与 CIFAR‑100 的 OOD 评估中实现了最高的 AURSAC 与 Brier 分数；在 VinDr‑CXR 与 CheXpert 的真实专家退避任务中表现为最优或竞争性的宏观 AURSBAC，且校准指标（Brier/ECE）均优于对照方法。

**⚠️ 局限性**

局限性：①需要足够的、与专家能力相关的查询–上下文几何信息；②在高类别维度或上下文稀疏时同角色邻域噪声显著，需依赖神经核池化的平滑；③实验结果为描述性比较，缺乏统计显著性检验；④对非随机或预设的专家-案例混合（case‑mix）情形的鲁棒性尚待进一步验证。

---

## 479. AutoViewMem: Self-Configuring Orthogonal Views for Conversational Long-Term Memory

**arXiv ID:** 2609.21940 | [PDF](https://arxiv.org/pdf/2609.21940v1)

**作者:** Zijie Cao `[一作]` (National University of Defense Technology), Yang Mei `[通讯]` (National University of Defense Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 AutoViewMem 框架，利用自适应语义视图在写入时对长时对话记忆进行分离与结构化，从而提升 LLM 的长期检索和个性化性能。

**💡 创新点**

创新点在于通过数据驱动的视图发现和 DPP 选择生成低重叠的语义视图，并在写入阶段完成结构化提取，先在写入时消除语义干扰，再用简单的 top‑K 检索即可得到聚焦证据。

**🔧 技术方法**

采用 LLM 生成视图和提取模板、Determinantal Point Process (DPP) 进行视图选择、稠密向量检索、图聚类与 LLM 决策的离线合并，以及时间戳和来源追踪等技术。

**📊 数据集**

实验使用 LoCoMo（200–400 轮长对话）和 PersonaMem‑32k（多选个性化问答）两个基准数据集。

**📈 对比分析**

在 LoCoMo 与 PersonaMem 上与 Mem0、MemGAS、MemoryBank 等基线以及 Full‑History 对比，AutoViewMem 在 Qwen3‑8B/14B 上在 Judge/F1/BLEU、准确率等指标均优于基线，并在检索覆盖率和 token 成本上表现更佳。

**⚠️ 局限性**

局限性包括对 LLM 视图发现与合并能力的依赖、视图发现周期性导致对快速分布漂移响应慢、离线合并可能过度合并细微时间差异，以及实验仅局限于两大单语多轮对话基准，需在多方、跨语言或安全场景进一步验证。

---

## 480. MAAP: Multi-Agent Active Perception for Collaborative Manipulation

**arXiv ID:** 2609.21929 | [PDF](https://arxiv.org/pdf/2609.21929v1)

**作者:** Bruno N. Y. Chen `[一作]`, Yiran Qin `[通讯]` (Chinese University of Hong Kong Shenzhen)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用多臂协作时每个臂上的腕部摄像头同时执行操作和视觉感知，提出MAAP多臂主动感知框架，并在控制层实现RAIL角色感知式模仿学习，实现对每个臂在每步动作片段中的操控/感知角色进行推断并据此调制动作生成。

**💡 创新点**

①把臂的操控与感知功能合二为一，消除专用感知臂；②在单一网络中通过角色后验分布对动作头进行条件混合与FiLM特征调制，实现角色依赖动作；③对多视角融合方法进行系统比较，展示角色感知对协作动作分辨率的提升。

**🔧 技术方法**

基于Transformer的ACT动作分段策略、角色推断模块、FiLM特征调制、动作头混合、四种多视角融合（MV、SF、CF、DCF）以及冻结的DINOv2特征提取器。

**📊 数据集**

RoboFactory benchmark（基于ManiSkill3）中的四个协作任务（Stack Cube、Pick from Pot、Pick from Microwave、Place on Cart）以及在双臂ARX机器人平台上的真实机器人实验。

**📈 对比分析**

与固定摄像头（GV）、单一腕部摄像头（SAV）以及多臂多视角（MAAP）在相同的ACT控制器下比较；MAAP-MV平均成功率从56.5%提升至70.0%，再加RAIL提升至79.2%；在硬件实验中，MAAP+RAIL在20次放置任务中取得14/20的成功率，而固定摄像头为0/20。

**⚠️ 局限性**

仅在有限的任务和单次训练运行中验证；角色标注依赖专家示范，缺乏对多臂协作中更广泛角色分配的探索；未与专用感知臂或全局+腕部混合输入进行对比；对不同任务或更复杂场景的泛化性未做系统评估。

---

## 481. Assessment of Machine Learning-Based Critical Heat Flux Models in the CTF Subchannel Code for Square Rod Bundle Prediction

**arXiv ID:** 2609.21995 | [PDF](https://arxiv.org/pdf/2609.21995v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 482. Dynamic Contention Resolution Schemes

**arXiv ID:** 2609.21993 | [PDF](https://arxiv.org/pdf/2609.21993v1)

**作者:** Moran Feldman `[一作]` (University of Haifa), Sherry Sarkar `[通讯]` (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种新的动态冲突消解方案（Dynamic Contention Resolution Schemes，DCRS），用于在完全动态的包装问题（如基于下闭合约束的最大化问题）中实现低递归（low‑recourse）并结合正体追踪框架得到具有竞争性递归的动态优化算法。

**💡 创新点**

创新点：①首次将冲突消解方案引入动态环境，形成 DCRS；②证明在多项式时间内对多种约束（如基、匹配、背包、k‑合并基、p‑匹配族、稀疏 PIP）构造了 (Ω(1), O(log n)) 或更优的平衡与递归比；③通过 DCRS 与正体追踪框架的组合，得到通用的竞争性递归动态子模优化算法，涵盖非凸目标与时间变化的子模函数；④给出组合约束的通用合并定理，可构造任意交叉约束的 DCRS。

**🔧 技术方法**

主要技术：
- 动态链结构与“摩擦”机制的改进；
- 递归构造和重建策略，用以保证链层数对数级；
- 时间相关采样（α‑TCOS）与时间相关的随机化样本；
- 对多种约束使用专门的随机化筛选和排序（如按大小、按桶、按优先级）；
- 组合定理将各约束的 DCRS 级联，保持平衡与递归；
- 正体追踪（positive body chasing）框架生成低递归的 LP 近似解。

**📊 数据集**

该工作属于理论计算机科学，无实测数据集；所有结果均为理论分析与渐进复杂度/性能上界。

**📈 对比分析**

与之前的绝对递归（absolute‑recourse）方法相比，DCRS 在竞争性递归上实现了更优的上界；在匹配、背包等问题上首次给出非平凡的竞争递归解法；相比于之前的 (1−e^−1) 近似率，匹配、基等问题实现了更高的平衡比例；整体递归次数为 O(log n) 或 O(1)，显著优于此前多项式级别或不可行的递归方案。

**⚠️ 局限性**

局限性：
- 依赖于正体追踪框架的可行性，若目标函数或约束不满足松弛性质，框架失效；
- 对背包、匹配等问题的递归常数相对较大，实际实现需精细调参；
- 组合约束的递归系数随约束数量线性增长，可能在高维交叉约束下递归成本过高；
- 仅考虑下闭合约束，尚未扩展到上闭合或混合约束；
- 证明过程中使用了多项式时间的随机化算法，实际实现的随机数产生和多项式时间分离 oracle 的成本未给出。

---

## 483. DiaVLo: Diagnosing Behaviours of Vision-Language Models

**arXiv ID:** 2609.22008 | [PDF](https://arxiv.org/pdf/2609.22008v1)

**作者:** Lorenzo Corti `[一作]` (Delft University of Technology), Jie Yang `[通讯]` (Delft University of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种诊断框架，通过人类校验的场景图与VLM自解释生成的行为规范，识别并分类VLM的期望与实际行为，揭示潜在的不一致与错误；

**💡 创新点**

创新点在于将人类验证的结构化视觉描述与VLM自生成的推理链相结合，并采用因果建模估计视觉概念对输出的影响，从而实现对VLM行为的系统化诊断与因果解释；

**🔧 技术方法**

技术手段包括场景图生成（IETrans）、人工校验与扩展、VLM链式推理提示、概念定位（OWLv2）、因果推断（Double‑Machine‑Learning 估计器），以及语义相似度匹配；

**📊 数据集**

实验使用四个公开数据集：LLaVa‑Bench、MMBench、SEED‑Bench 2 与 VQA v2，评测四个开源VLM（InternVL2、LLaVa‑1.6、Qwen2.5‑VL、ShareGPT4V）；

**📈 对比分析**

通过将行为相似度与模型准确率计算互信息，发现行为标签与性能高度相关；将行为分为“对齐”“扩展”“偏离”，并展示各模型在不同数据集上的行为分布，说明框架能有效揭示模型行为差异；

**⚠️ 局限性**

局限包括：依赖VLM自解释的可信度不足、需要人工校验导致可扩展性受限、模型指令遵循不稳定、仅评测7B‑8B规模模型且未涉及封闭源大模型。

---

## 484. Joint Remaining Useful Life Prediction and Capacity Estimation of Lithium-Ion Batteries Using Partial-Charging Data

**arXiv ID:** 2609.21932 | [PDF](https://arxiv.org/pdf/2609.21932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 485. Moral Entropy: Auditing Bias and Uncertainty in Moral Judgment

**arXiv ID:** 2609.21992 | [PDF](https://arxiv.org/pdf/2609.21992v1)

**作者:** Maciej Skorski `[一作]` `[通讯]` (University of Luxembourg), Maciej Skorski (University of Luxembourg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

引入 Moral Entropy 这一 Bayesian 框架，对道德注释中的争议进行建模和评估，而非简单投票求和。

**💡 创新点**

创新点在于把注释者的歧义拆分为可约的“aleatoric”与可减少的“epistemic”不确定性，并用该不确定性来校准与审计传统聚合规则。

**🔧 技术方法**

采用 Dawid–Skene 风格的贝叶斯聚合模型，配合 Laplace 近似获取后验分布，利用交叉熵、Brier 分数和校准误差等 Bregman 散度进行评估。

**📊 数据集**

在三大 MFT 注释语料库（MFTC、MFRC、eMFD）上进行实验，涵盖七个社会话题域与五种道德基底。

**📈 对比分析**

与传统任何标注者、两票和多数票规则对比，发现任何标注者规则在大多数域下误报率高达30%，而更严格规则则漏报率高达80%；使用 Soft‑label 训练的模型相较硬标签提升 2–3% 的准确率。

**⚠️ 局限性**

局限在于拉普拉斯近似假设的完整性、模型对注释者池的文化偏见未作补偿、以及 Soft‑label 优势仅在有限模型与数据上验证。

---

## 486. Time series generation with spectrally aligned latent flow matching

**arXiv ID:** 2609.21989 | [PDF](https://arxiv.org/pdf/2609.21989v1)

**作者:** Camilo Carvajal Reyes `[一作]` (Imperial College London), Felipe Tobar `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种在潜空间对时间序列进行谱对齐的潜流生成模型，利用傅里叶、小波和签名变换的变换一致性损失强化高频信息，提升合成信号的真实性。

**💡 创新点**

创新点是引入基于频谱和路径特征的多尺度一致性损失，并用Sobolev加权与Wasserstein界限理论证明其能减轻潜空间压缩引起的谱伪影，克服传统欧式重构误差的频谱偏差。

**🔧 技术方法**

使用潜流匹配（rectified flow）、自编码器（encoder‑decoder）、傅里叶、小波、签名一致性损失、Sobolev加权、Wasserstein界限分析及基于流的ODE采样。

**📊 数据集**

实验数据集包括天气（Weather）、外汇（Exchange Rates）和HEPC三类真实长序列，维度分别为8、1和14。

**📈 对比分析**

与四种扩散基准（SigDiffusion、Diffusion‑TS、CSPD‑GP、DDO）比较，生成速度提升至≈1 s生成1000个样本，判别、预测和KS等指标均优于基线，尤其在判别分数上显著降低误差。

**⚠️ 局限性**

局限性：潜空间压缩仍可能产生残留谱伪影；高频权重调参对不同数据集敏感；签名一致性训练耗时较长，且对极低能量频段的提升有限。

---

## 487. Depressive symptoms are reflected differently across digital contexts

**arXiv ID:** 2609.21919 | [PDF](https://arxiv.org/pdf/2609.21919v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 488. End-to-End Hard-Label Cryptanalytic Model Extraction Using Efficient Sign Recovery

**arXiv ID:** 2609.21941 | [PDF](https://arxiv.org/pdf/2609.21941v1)

**作者:** Akira Ito `[一作]` (Tohoku University), Yosuke Todo `[通讯]` (NTT Social Informatics Laboratories)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种基于余弦相似度的新符号恢复算法，完成了训练的全连接 ReLU MLP 在硬标签场景下的端到端模型提取；

**💡 创新点**

通过不需要额外查询的余弦/加权余弦方法实现快速、准确的符号恢复，并结合投影、主动加权与交叉层提取，克服了传统边界漫游方法的高查询量和计算难题；

**🔧 技术方法**

使用余弦相似度、加权余弦、签名加权余弦、投影恢复、主动加权余弦、交叉层提取、统计阈值判别、噪声过滤与签名一致性校验等技术；

**📊 数据集**

采用 MNIST 与 Fashion‑MNIST 图像分类数据集，训练多层 784‑16^L‑10（L=4 或 6）的全连接 ReLU MLP；

**📈 对比分析**

与之前的 boundary‑walking 方法相比，在相同样本量下（每个神经元 50 个交叉点）签名恢复准确率达 98‑99%，端到端提取实现 98.5%–100% 的标签一致率，在 100k 标准正态输入上表现优异；

**⚠️ 局限性**

仍需大量交叉点收集，查询成本高；仅适用于全连接 ReLU MLP，难以推广到更复杂架构；对几乎死亡或死神经元处理不完整，且对数值误差敏感。

---

## 489. Online Algorithms with a Sample: Tight Bounds and Adversarial Robustness

**arXiv ID:** 2609.21889 | [PDF](https://arxiv.org/pdf/2609.21889v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 490. What Should We Ask Next? Retrieval-Aware Question Learning under Partial Evidence

**arXiv ID:** 2609.21924 | [PDF](https://arxiv.org/pdf/2609.21924v1)

**作者:** Lyucheng Qian `[一作]` (Sichuan University), Pingyu Wang `[通讯]` (Sichuan University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种在线强化学习框架 RAVEL，通过在完整交互循环中利用检索反馈来学习最佳提问策略，以提升交互式人物重识别的检索效果。

**💡 创新点**

从离线行为克隆转向在线检索反馈驱动的强化学习；引入检索感知奖励、有效性门控和候选图像直接输入，证明局部开放式属性提问最具检索价值。

**🔧 技术方法**

多模态语言模型 + CLIP+IRRA 检索器 + 强化学习（GRPO） + 答案清洗 + 奖励设计（递归排名提升+Rank‑1 奖励）。

**📊 数据集**

Interactive‑PEDES 为主评估集，CUHK‑PEDES、ICFG‑PEDES、RSTPReid 用于跨数据集迁移测试。

**📈 对比分析**

与 PlugIR、ChatIR、SimRV、LLaVA‑ReID 等基线对比，RAVEL 在 5 轮后 Rank‑1 提升至 73.73（比 LLaVA‑ReID 高 5.79 点），同时 BRI、mAP 等指标也显著改善。

**⚠️ 局限性**

依赖预训练的检索器和答案生成器；仅在 Top‑4 候选下训练，可能不适用于更大规模或不同任务；奖励设计相对简单，RL 收敛与通用性仍需进一步验证。

---

## 491. Abstention and Noise Filtering: Two Missing Primitives of Softmax Attention

**arXiv ID:** 2609.22005 | [PDF](https://arxiv.org/pdf/2609.22005v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 492. Intervention Granularity Matters: Coherent Treatment Bundles in Counterfactual Simulation with Clinical World Models

**arXiv ID:** 2609.21906 | [PDF](https://arxiv.org/pdf/2609.21906v1)

**作者:** Fangzhou Wang `[一作]` (Duke University), Rishikesan Kamaleswaran `[通讯]` (Duke University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究了在临床世界模型中编辑干预时的粒度对预测结果的影响，

**💡 创新点**

创新点是提出bundle-consistent editing（基于真实患者相似轨迹的完整干预束编辑）并验证其比单一组件编辑更能引起模型响应，

**🔧 技术方法**

使用了Clin-JEPA潜在世界模型，结合语言模型编码器和Transformer预测器，

**📊 数据集**

数据集为MIMIC-IV ICU记录，包含约945,707个病人小时的干预文本和状态文本，

**📈 对比分析**

通过在每个呼吸机启动点比较单一设置与完整bundle对下一个小时预测的相对距离（d_z），并用引导法和线性回归调整编辑幅度，结果显示bundle产生的模型响应显著更大且不随编辑幅度而改变，

**⚠️ 局限性**

局限在于仅评估了模型表示层的响应差异，未检验真实因果效应，也未对编辑后产生的假设情景进行外部验证。

---

## 493. Detecting Pretraining Data in Large Language Models from a Free-Energy Perspective

**arXiv ID:** 2609.21888 | [PDF](https://arxiv.org/pdf/2609.21888v1)

**作者:** Chenye Ke `[一作]` (University of Science and Technology of China), Shijin Wang `[通讯]` (iFLYTEK Co., Ltd)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于预测损失与预测熵的倾斜边界检测方法，并将其转化为宏观能量转移检测（ETD），用于判定文本是否属于LLM的预训练数据。

**💡 创新点**

创新点在于将熵校正视为自由能修正，理论证明在均值熵为零且损失与熵正相关时熵校正既保持期望成员-非成员差距，又降低方差，从而显著提升判别性能；同时引入宏观能量转移视角，提供新的检测解释。

**🔧 技术方法**

采用熵校正线性组合得分、自由能视角、Token级别自由能贡献、首出现聚合等技术，并在灰盒检测框架下实现。

**📊 数据集**

在WikiMIA、StackMIAsub和MIMIR三个公开预训练数据检测基准上评估，覆盖多种开源LLM（GPT‑Neo、Mamba、OPT、Pythia、NeoX）及不同规模。

**📈 对比分析**

与七种灰盒基线（PPL、Ref、Lowercase、Zlib、Min‑K%、Min‑K%++、AECA）比较，ETD平均提升AUROC约3.5%（最高约5%）和TPR@5%FPR约5%（最高约8%），且在文本长度和继续预训练场景下表现更鲁棒。

**⚠️ 局限性**

局限在于熵系数λ的选取仍需经验或理论估计，且方法主要针对灰盒场景，无法直接应用于完全黑盒或跨语言的检测，未来需进一步扩展到更广泛模型‑数据交互。

---

## 494. Beyond the Desert Label: A Pathway Diagnostic for User-Centered Smart Mobility Service Design

**arXiv ID:** 2609.21956 | [PDF](https://arxiv.org/pdf/2609.21956v1)

**作者:** Oluwasegun Adegoke `[一作]` (Syracuse University), Sevgi Erdogan `[通讯]` (Syracuse University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一套基于开放数据的可重复诊断流程，用以区分城市中公共交通服务不足的两种机制：相对供需不匹配与最低服务失败，并为每个缺陷提供具体的服务属性分析。

**💡 创新点**

创新点在于：①首次将相对缺口分析与绝对最低服务阈值结合，形成双路径诊断；②通过空间自相关（Local Moran's I）识别连贯的高缺口簇；③提供缺陷特征档案，将抽象标签转化为可操作的服务维度；④实现完全可复现的工作流，便于跨城市比较。

**🔧 技术方法**

技术手段包括：空间统计（Local Moran's I）、标准化与加权指标构造、阈值筛选（最低服务标准）、多维特征聚合与缺陷排序、以及R/Python中数据处理和可视化。

**📊 数据集**

数据集涵盖：GTFS静态公交时刻表、美国人口普查ACS 5年估计、LEHD LODES就业数据、TIGER/Line区划、OpenStreetMap建成环境特征；四个案例城市为巴尔的摩、费城、纳什维尔、达拉斯。

**📈 对比分析**

比较方法：对比四城市不同路径下的分类比例、最低服务失效比例以及缺陷特征分布。结果显示，传统高密度轨道城市以相对不匹配为主，Sunbelt地区则同时存在相对不匹配与最低服务失败；缺陷档案进一步揭示不同城市的服务重点。性能上，工具能在数小时内完成整个流程，输出高可解释性诊断。

**⚠️ 局限性**

局限性包括：①仅使用静态GTFS，未考虑实时可靠性和乘客量；②对阈值和指标的敏感性未完全覆盖所有城市情境；③未结合实际乘客调查或运营商标准进行验证；④依赖公共数据质量，若数据缺失或不完整可能影响诊断准确性。

---

## 495. Setting the clock: Evaluating temporal window parameters for coordinated behavior detection

**arXiv ID:** 2609.21959 | [PDF](https://arxiv.org/pdf/2609.21959v1)

**作者:** Georgios Panayiotou `[一作]` (Uppsala University), Maurizio Tesconi `[通讯]` (IIT-CNR)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

评估时间窗口参数（窗口长度与步长）对社交媒体上协同行为检测的影响，构建时间多层共转发网络并通过 Louvain 模块检测社区，利用已标注的信息操作账户进行评价。

**💡 创新点**

首次系统分析窗口长度对检测结果的决定性作用，并证明窗口步长（重叠度）对精度和召回率影响甚微，提示窗口长度是关键建模选择。

**🔧 技术方法**

使用时间多层网络、窗口滑动机制、共转发边权构造、Louvain 社区检测以及 IO 精度/召回率与社区内聚合度等评价指标。

**📊 数据集**

使用 14 个公开的、包含已标注信息操作账号的社交媒体数据集，数据来源于 Zenodo。

**📈 对比分析**

对不同窗口长度（10 s到 1 天）与步长（0%–50%）的配置进行比较，发现窗口长度越长，检测到的社区越多、召回率提升但精度下降；步长变化对精度与召回几乎无影响。

**⚠️ 局限性**

仅考虑转发共同行为，未涉及多模态或多层网络；仅使用单一 Louvain 算法且对层进行平面化处理，可能掩盖步长效应；缺乏对更大规模或实时系统的评估。

---

## 496. Learning to Move Cities: Deep Meta-Models and Reinforcement Policies for Calibration and Control in Urban Networks

**arXiv ID:** 2609.21945 | [PDF](https://arxiv.org/pdf/2609.21945v1)

**作者:** Adewumi Augustine Adepitan `[一作]` (George Mason University), Oluwatobi Oluwasakin `[通讯]` (Federal University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出一种共享潜在空间框架，将城市交通模拟器的校准与强化学习控制统一起来；通过组合多层感知器与自编码器的MLP-AE架构学习低维表示，并在潜在空间中执行贝叶斯优化进行校准；随后利用该潜在表示作为状态输入，构建深度Q网络（DQN）实现动态交通分配与路由调度。

**💡 创新点**

创新点在于：①首次将同一潜在表示同时用于模拟器校准和强化学习控制，实现了校准与决策的无缝衔接；②设计组合式MLP-autoencoder网络，既实现降维又保持输入输出的非线性映射；③在潜在空间中进行贝叶斯优化显著提升样本效率；④将潜在表示嵌入RL状态，提升策略学习稳定性与性能。

**🔧 技术方法**

使用技术包括：深度学习（多层感知器、自动编码器、深度Q网络）、贝叶斯优化与高斯过程回归、经验回放、目标网络、双Q学习、优先经验回放；组合式MLP-AE用于潜在学习。

**📊 数据集**

实验数据基于两个基准网络：一个中等规模（50区，500条路段）的交通网络，用POLARIS仿真生成观测数据；另一个简化的证明概念网络，用迭代Frank-Wolfe算法进行动态交通分配仿真；观测数据通过向模拟输出添加高斯噪声得到。

**📈 对比分析**

校准方面与标准贝叶斯优化及活跃子空间+贝叶斯优化对比；指标为NRMSE、计算时间，结果显示本方法在NRMSE上提升35%（相较活跃子空间）并将计算时间从48.2h降至18.3h。控制方面与无控制基线对比；指标为系统总行程时间、平均延迟等，DQN控制在100轮后实现51%总行程时间下降、34%平均延迟下降。

**⚠️ 局限性**

局限性包括：实验仅在小规模基准网络上验证，缺乏对大规模城市网络的规模化评估；假设网络状态完全可观测，实际部署可能面临传感器稀疏；环境非平稳性（需求变化、基础设施演化）未充分考虑；对潜在空间的解释性与可迁移性仍需深入研究。

---

## 497. CommitFlow: Semantic Commitment Verification and Local Correction for Long-Horizon Robot Manipulation VLA Execution

**arXiv ID:** 2609.21908 | [PDF](https://arxiv.org/pdf/2609.21908v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 498. The Role of Radiometric Features in Cross-Site Leaf-Wood Segmentation of LiDAR Point Clouds

**arXiv ID:** 2609.21903 | [PDF](https://arxiv.org/pdf/2609.21903v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 499. When Should a Failing Robot Ask? Initiating Corrective Human-Robot Dialogue from Audited Sensor Evidence

**arXiv ID:** 2609.21942 | [PDF](https://arxiv.org/pdf/2609.21942v1)

**作者:** Eshika Pathak `[一作]` (University of Illinois Urbana Champaign), Leela Krishna `[通讯]` (Centific)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了带注入失效的机器人基准，并评估视觉语言模型在诊断和决策时的表现。

**💡 创新点**

首次将传感器信息可诊断性与模型诊断可信度结合，提出基于测量准确率与成本的最优问答决策框架。

**🔧 技术方法**

使用了传感器可诊断性审计、视觉语言模型多轮问答、决策理论（期望成本比较）和文本化力数据输入等技术。

**📊 数据集**

在LIBERO模拟器中生成的三类失败（检测、抓取、放置）样本，共2200个失败样本；包括图像、力/扭矩、机器人运动等传感器。

**📈 对比分析**

对六个7B–16B开源视觉语言模型进行诊断准确率、拒绝率、询问率与最优策略的比较；大多数模型准确率低于多数类基线，询问率与成本无关，只有部分模型在接收力数据后表现提升。

**⚠️ 局限性**

样本量有限，基准仅在仿真环境下，模型受限于帧信息，未测试真实机器人或更先进模型，问答质量和多轮对话未覆盖。

---

## 500. Can I Trust My Body? A Three-Year Autoethnography of ChatGPT's Place in My Support System for Panic Attacks

**arXiv ID:** 2609.21925 | [PDF](https://arxiv.org/pdf/2609.21925v1)

**作者:** Dongyijie Primo Pan `[一作]` (Hong Kong University of Science and Technology), Mirjana Prpa `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过三年纵向自我民族志研究，分析作者在恐慌发作中使用 ChatGPT 的体验，识别对话中的解释、安慰、行动指导、安全检查等角色，并提出轨迹级安全设计目标 Fit、Closure、Continuity。

**💡 创新点**

创新点在于提出轨迹级安全框架，系统揭示大语言模型在恐慌情境中的角色演变、对检查行为的影响以及与人类支持网络的交互，为健康 AI 设计提供新的安全与连续性视角。

**🔧 技术方法**

采用自然语言处理对 ChatGPT 对话进行角色编码和切换计数，结合 Riessman 的叙事分析方法，构建 43 条恐慌轨迹，并手工分析对话摘录与多来源背景记录。

**📊 数据集**

使用的数据集包括作者三年内的 87 条与恐慌相关的 ChatGPT 对话、个人记录、朋友/家人及专业人员叙述，以及 PHQ‑9 与 SCL‑90 量表数据。

**📈 对比分析**

通过对 792 条回复进行角色标注与切换计数，比较不同角色出现频率及其随时间变化；结果显示解释与安全检查最常见，角色切换不均衡，说明模型在恐慌情境中的支持有限且亟待设计优化。

**⚠️ 局限性**

局限性包括单一案例研究、回顾性重构、数据缺失、缺乏临床评估、未比较模型版本、样本数字素养偏高，导致结果难以推广且未对安全效果进行量化验证。

---

## 501. Bayesian Belief Layer for Controllable Opinion Dynamics in LLM Agents

**arXiv ID:** 2609.21997 | [PDF](https://arxiv.org/pdf/2609.21997v1)

**作者:** Hafsa Akbar `[一作]` (Massachusetts Institute of Technology), Hossein Rahnama `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Bayesian Chronicle Agents（BCA），在 LLM 生成语言与信念更新之间引入可控的贝叶斯信念层，并用单一先验强度参数实现对从一致到持久分歧再到少数派影响等三种经典舆情动力学的调控。

**💡 创新点**

创新点在于将传统概率模型的僵硬参数（如 Friedkin–Johnsen 的 stubbornness）映射到可在 LLM 中直接设定的先验强度，并通过贝叶斯更新实现信念与表达的解耦，同时保证信念可追踪、可恢复、可审计。

**🔧 技术方法**

技术包括：LLM 生成器（用于把概率信念转为文本）、LLM 评估器（把文本映射为 0-1 的证据值）、贝叶斯更新公式、先验强度参数调节、温度校准和对话流程自动化。

**📊 数据集**

使用的“数据集”是人工构造的单一政策问题（两个对立立场），在 20 名代理人、完整图拓扑、20 轮回合的合成对话中收集的日志，并对 100 条评估样本进行校准；实验在四种 LLM（OpenAI、Meta、Anthropic、和自定义模型）上重复。

**📈 对比分析**

通过将 BCA 的信念更新与 DeGroot、Friedkin–Johnsen 和少数派模型的闭式解进行对比，检验一致性、持久分歧和少数影响三种动态；结果显示一致性时平均偏差在 0.01–0.15 范围内，持久分歧的 R² 达到 0.93–0.99，且 Spearman ρ = 1.0 证明先验强度在语言通道后可完全恢复。

**⚠️ 局限性**

局限性包括：仅测试单一虚构议题、完全连通图、20 名代理、二元立场、顺序发言方式、评估器的校准依赖少量人工标签，以及对多概念、异构网络和人类对话轨迹的适用性尚未验证。

---

## 502. CARF: Contrastive Attraction-Repulsion of Failure-Guided Flow Matching

**arXiv ID:** 2609.21982 | [PDF](https://arxiv.org/pdf/2609.21982v1)

**作者:** Shuqi Zhao `[一作]`, Masayoshi Tomizuka `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出CARF框架，通过进度重要性评分区分成功段和失败关键段，并用吸引-排斥的流匹配目标实现对机器人轨迹的学习。

**💡 创新点**

首次将进度重要性评分与对抗式吸引-排斥机制结合，使模型既能利用失败轨迹中的有价值段落，又能主动避免错误段落，从而提升数据利用率与执行稳健性。

**🔧 技术方法**

采用基于MLP的进度预测评分、流匹配策略、对比式吸引-排斥损失、阈值分段等技术。

**📊 数据集**

在RLBench模拟环境的五个抓取/放置任务以及两个人工遥控的真实世界任务中进行实验。

**📈 对比分析**

与成功示例仅训练、全部数据训练以及基于分类器的CGFM基线对比，CARF在所有任务上均实现显著提升（最高提升达70%）。

**⚠️ 局限性**

依赖抓取状态分割，忽略中性段落，阈值需手工调节；对连续接触任务的适用性有限。

---

## 503. Multiplicative Optimism for Constant Regret in Games

**arXiv ID:** 2609.21976 | [PDF](https://arxiv.org/pdf/2609.21976v1)

**作者:** Ashkan Soleymani `[一作]` (MIT), Georgios Piliouras `[通讯]` (Google Deepmind)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

提出了一种新的无耦合学习规则——Multiplicatively Optimistic Regret Matching（MORM），在有限一般和游戏的自我博弈中实现了与时间无关的个体外部遗憾上界。

**💡 创新点**

创新点在于：①仅使用一步优化预测（one‑step optimism），不需要高阶预测或提升维度；②通过引入乘法型乐观修正和特殊的势函数，成功把玩家数量的依赖从线性降到 √n；③使用 Hellinger 距离控制策略变动，实现了对预测误差的精细把握；④提供了一个学习率安全机制，使得在对手可能是对抗性的情况下仍保持 √(T log d) 的经典上界。

**🔧 技术方法**

技术方法包括：潜在函数（potential）与梯度权重相匹配的设计、乘法型乐观（multiplicative optimism）更新、Hellinger 距离的利用、曲率与梯度权重的加权控制、以及学习率安全机制（learning‑rate safeguard）。

**📊 数据集**

该工作为理论性研究，无需使用任何数据集；所有结果均基于对有限一般和游戏的数学分析。

**📈 对比分析**

与现有方法（如 Optimistic Hedge、LRL‑OFTRL、Cautious Optimism、ECHO‑OFTRL、HOOD 等）相比，MORM 在玩家数 n 上的依赖从 O(n) 降为 O(√n)，在动作数 d 上的依赖从多项式降为 O(log d)。在自我博弈场景下，MORM 的个体遗憾上界为 96√n(2+log d)，显著优于之前的 O(n log d log⁴T) 或更高阶的上界；在对抗性序列下通过学习率安全机制获得 96√n(2+log d)+21√(T log d) 的上界。

**⚠️ 局限性**

局限性包括：①对√n 的下界尚未得到证明，是否为最优仍未知；②仅适用于全信息（full‑information）反馈，无法直接推广到 bandit 或嘈杂的实用环境；③仅保证平均策略满足粗相关均衡（CCE），未能进一步保证更强的纳什均衡或最后一次迭代收敛；④在极端极限（如 n 很大或 d 很大）下仍需进一步评估常数项和实现效率。

---

## 504. NemotronLabs VoiceChat: An Open Full-duplex Speech-to-Speech Model with Tool Calling Capabilities

**arXiv ID:** 2609.21967 | [PDF](https://arxiv.org/pdf/2609.21967v1)

**作者:** Jagadeesh Balam `[一作]` (NVIDIA), Zhonglei He `[通讯]` (NVIDIA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 NemotronLabs VoiceChat，一种开源全双工语音对语音模型，支持实时听写、推理、工具调用与语音输出。

**💡 创新点**

创新点在于将代理文本、函数调用和用户转录分别用并行专用流建模，并与流式 TTS 解码器和共享 RNN‑T 分支整合，实现同一流水线下无序列化的工具交互与低延迟对话。

**🔧 技术方法**

使用 FastConformer 编码器、decoder‑only LLM（NVIDIA Nemotron Nano 9B）、RNN‑T、Mixture‑of‑Gaussians TTS 解码器以及字符级子词编码器等技术。

**📊 数据集**

训练集包括持续预训练的多语言文本、模拟对话、语音合成对话、工具调用示例、VoiceBench、Full‑Duplex‑Bench、OpenASR 等公开数据集。

**📈 对比分析**

与其他全双工模型对比，Full‑Duplex‑Bench 1.0/1.5 获得最低暂停占用率和最高用户中断接管率，VoiceBench 平均分 55.1，工具调用 FDB 3.0 的工具选择 F1 82.5%；整体性能优于多数开源模型。

**⚠️ 局限性**

局限性包括两分钟窗口以内的对话记忆有限、工具调用准确率低（参数错误、链式调用不稳）、长工具响应导致延迟、在噪声或多人语音环境下鲁棒性不足。

---

## 505. Provisional Reachability: Containing Agents by Making Every Crossing Revocable

**arXiv ID:** 2609.21957 | [PDF](https://arxiv.org/pdf/2609.21957v1)

**作者:** Yoshiaki Takashita `[一作]` `[通讯]` (Waseda University), Yoshiaki Takashita (Waseda University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并量化了一种基于 escrow 的数据泄漏容忍方案：每次跨界传输（crossing）在一个窗口期内被暂时持有，系统以固定概率 r 随机审核这些持有的操作；若任何一次审核发现违规，则整个窗口被撤销并重置，从而限制单个窗口内的泄漏量为 L(r)=c/(e·ln(1/(1−r))) ≈ c/(e·r) 比特。作者进一步演示了如何通过“旋转入口”“死端”等欺骗手段降低攻击者定位成功率 a，从而在公式中将 (1−r) 替换为 a(1−r)，并给出完整的堆栈实现与仿真评估。

**💡 创新点**

创新点：①首次给出可对抗策略攻击者的、基于时间窗口的泄漏速率上界；②证明该上界与攻击者对 audit 率 r 的了解无关，攻击者即使知晓 r 也无法突破；③提出“检测耦合”原则，即只要攻击者做出错误操作即可触发窗口撤销，进一步降低泄漏；④将欺骗技术与 escrow 机制组合，形成可组合的防御体系；⑤引入“秘密衰减”阈值，将速率约束转化为可解的总泄漏上界。

**🔧 技术方法**

使用技术：基于离散概率模型与组合优化推导泄漏上界；离散时间窗口（periodic escrow）与随机采样审计；旋转可访问入口（keyed rotating entries）实现对攻击者定位的概率降低；死端（dead‑ends）与连锁审计相结合实现检测耦合；实现层面为生产级编码代理的 hook，记录跨界调用并在窗口结束时批量执行或撤销；仿真框架（Python/NumPy）用于验证理论上限与实际泄漏。

**📊 数据集**

数据集与实验：①模拟攻击者跨界次数 k 与 audit 率 r 的组合，验证 L(r) 上界；②对 24 种只读工具调用进行实测，估算每个调用的比特容量 c；③在实际代理环境下跑 10^5 次窗口的仿真，记录合法工作重发率、泄漏比特和窗口撤销率；④在有限数量（例如 8）旋转入口的实验中，测量 a 与泄漏量的关系；⑤对“假友”表面与“死端”方案的欺骗效果进行对比实验。数据来源主要为仿真与内部工具调用日志，未使用公开大规模真实攻击数据。

**📈 对比分析**

比较方法：对每种防御方案（仅 escrow、+旋转入口、+死端、+固定 audit 率等），分别在相同 r 与 k 组合下运行多次仿真，统计最大期望泄漏、窗口撤销率以及合法工作完成率。结果显示：
• 仅 escrow 时，每窗口泄漏可降至约 5×10^-4 倍；
• 加旋转入口后泄漏进一步下降至 ~10^-5，合法工作损失约 80%；
• 加死端后泄漏几乎为 0，但合法工作仅 12% 完成；
• 固定 audit 率对泄漏影响不大，但降低了协同攻击的风险。整体堆栈将 100,000‑bit 的完整秘密泄漏率压至 < 100 bit/窗口，约 1000 倍提升；但同时导致 88% 的合法工作被撤销。

**⚠️ 局限性**

Limitations：
• 需保持 audit 率 r 恒定，随机化会提升攻击者收益；
• 假设 audit 采样与识别率独立，若识别率低则泄漏上界失效；
• 仅对能随时间衰减的秘密给出总泄漏上界，静态秘密仍受限于速率；
• 需要预估每个跨界通道的容量 c，实际测量可能不精准；
• 设计导致的窗口撤销会对合法操作产生高额 DoS 影响，除非采用每主体窗口隔离；
• 对攻击者能主动诱导 audit 率变化（例如通过负载攻击）未考虑；
• 旋转入口与死端对时钟漂移极其敏感，若主机时钟不一致会导致大量合法请求失效。

---

## 506. Kinks vs. Smoothness: Identifiability of Real Analytic nICA for Laplace-like Sources

**arXiv ID:** 2609.21926 | [PDF](https://arxiv.org/pdf/2609.21926v1)

**作者:** Isaac Manring `[一作]` (University of Florida), Kejun Huang `[通讯]` (University of Florida)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

证明了在生成函数为实解析且源分布在一阶导数有有限个不连续点的情况下，非线性独立成分分析可实现精确识别，并在实验中验证了 RAD 方法的可行性。

**💡 创新点**

利用源分布的 kink 与生成函数的光滑性对比，提出了实解析解码器 RAD，实现了非线性 ICA 的可识别性。

**🔧 技术方法**

采用实解析差分映射、Weierstrass 准备定理、实解析神经网络（Normalizing Flow、VAE）以及 Laplace 先验。

**📊 数据集**

合成数据、Yahoo 股票收益、CelebA 人脸图像。

**📈 对比分析**

用均值相关系数 (MCC) 对比不同激活函数与源分布，实解析激活与 Laplace 源在实验中取得高 MCC，非解析激活或可分布失效。

**⚠️ 局限性**

对实解析近似的通用性可能逼近测量保持映射，compact 支持源分布在可识别性与可实现性之间存在冲突；样本复杂度与 KL‑gap 对识别性的影响尚未解决。

---

## 507. Training Music Sample Identification Models on Real Sample Pairs

**arXiv ID:** 2609.21911 | [PDF](https://arxiv.org/pdf/2609.21911v1)

**作者:** R. Oguz Araz `[一作]`, Dmitry Bogdanov `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在样本识别任务中，提出并训练了 SI Embeddings，使用真实样本对进行完全监督训练，显著提升检索性能。

**💡 创新点**

创新点在于首次利用大规模真实样本对训练，并结合 Fish 架构、Triplet 损失与多重信号扰动，突破人工样本对的局限。

**🔧 技术方法**

采用 Fish 迁移架构、ResNet50-IBN、GeM 池化、VQT 前端、Triplet 损失以及多种音频增强与信号变换。

**📊 数据集**

使用最近发布的包含 79,111 对真实样本对的训练集，以及公开的 7K/1K 小型基准和 8,444 对的大规模测试集。

**📈 对比分析**

与 SampleID 等基准模型进行 exhaustive retrieval 对比，在 map 上提升 26–56% 以及在大规模测试集上仅用 5% 训练数据即可匹敌先前最佳模型。

**⚠️ 局限性**

局限在于对真实样本对的依赖，人工样本对的迁移效果尚未完全验证，且实验主要聚焦在 6 秒片段，未探究更长/短片段的鲁棒性。

---

## 508. LLMs as Feature Engineers for Text-and-Tabular Prediction

**arXiv ID:** 2609.21894 | [PDF](https://arxiv.org/pdf/2609.21894v1)

**作者:** Merwan Barlier `[一作]` (Teads), Blaz Skrlj `[通讯]` (Teads)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过LLM生成器和提取器的迭代循环，自动从原始文本中提取可解释的离散特征，并将其与传统TF‑IDF、密集句子嵌入一起用于表格模型，显著提升预测性能。

**💡 创新点**

创新点在于将模型错误（如AUC排名逆转）转化为结构化自然语言反馈，驱动LLM在特征空间高效搜索；同时提出了离散、可解释的schema‑bound特征定义，保证了可解释性和跨任务迁移性。

**🔧 技术方法**

技术包括：LLM生成器（如ChatGPT‑4）产出JSON格式特征定义；低成本提取器做零-shot分类；XGBoost/树模型评估并前向贪心选择；基于误差对齐的自然语言反馈；SHAP解释验证特征重要性。

**📊 数据集**

使用公开数据集：Kickstarter（短标题）、Amazon Books Review（中等长度评论）和Stack Overflow问答（长且结构化文本）。

**📈 对比分析**

与TF‑IDF+SVD、MiniLM+PCA等单视图做对比，单独10个LLM特征在Kickstarter、Amazon上AUC提升约3%–10%，在Stack Overflow提升约3%；三视图（LLM+TF‑IDF+嵌入）获得最高AUC；错误驱动反馈相较无反馈可将收敛速度提升约3–5倍，显著降低LLM调用成本。

**⚠️ 局限性**

局限包括：依赖专有LLM，缺乏完全可复现性；仅在英语数据集验证，低资源语言或专业领域的迁移性未知；特征增多导致上线推理延迟上升（30%–50%），需要平衡计算成本；可能引入模型偏见与隐私风险。

---

## 509. COMPLEX: A Closed-Form Certified Embedding of Multiparameter Persistence Modules

**arXiv ID:** 2609.22012 | [PDF](https://arxiv.org/pdf/2609.22012v1)

**作者:** Sushovan Majhi `[一作]` (George Washington University), Pramita Bagchi `[通讯]` (George Washington University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一个名为COMPLEX的多参数持久性向量化方法，利用一组固定的近对角切片网格和每个切片的闭式LANDMARK嵌入（PLACE/PALACE）实现对模块的无训练、可计算的嵌入；

**💡 创新点**

首次为多参数持久性提供双侧失真界限，尤其给出了闭式下限（通过单个证伪切片实现的低尺度“floor”），从而实现了预测的分裂无关的每实例保证；

**🔧 技术方法**

核心技术包括切片堆叠（slice‑stack）、已证实的单参数LANDMARK映射、近对角切片网络、证伪切片一致性判定以及对核/最近邻头的理论保证；

**📊 数据集**

在点云数据集Orbit5k和Orbit100k以及多个图数据集（COX2、DHFR、MUTAG、NCI1、PTC_MR、PROTEINS、IMDB）上进行评估；

**📈 对比分析**

与Euler特征、变压器Persformer、xPerT、Graphcode、Gril等方法比较，COMPLEX在Orbit5k/100k上分别达到91.95%和92.98%精度，并在图数据集上以超过3个百分点的优势超越Gril，整体与最强的已训练多参数方法相当；

**⚠️ 局限性**

尽管具备理论保证，但证书的覆盖率仅约1.5%，对局部分离不足以提升分类准确度；此外，证书对梯度自适应无益，且在更大规模（如OGB）下表现下降，说明在数据稠密度和类间交错程度高的场景下仍受限。

---

## 510. Info3R: Information-Adaptive Test-Time Training for 3D Reconstruction

**arXiv ID:** 2609.21938 | [PDF](https://arxiv.org/pdf/2609.21938v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 511. Minimum distances of primitive narrow-sense BCH codes via good zero-sets

**arXiv ID:** 2609.21994 | [PDF](https://arxiv.org/pdf/2609.21994v1)

**作者:** Run Zheng `[一作]` `[通讯]` (Hong Kong University of Science and Technology), Run Zheng (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

对原码距离问题进行研究，定义并利用F_q-good zero-set来确定原码的最小距离等于设计距离。

**💡 创新点**

引入F_q-good zero-set概念，将最小距离问题转化为集合存在问题，并给出构造方法，得到多族新的原码设计距离匹配。

**🔧 技术方法**

组合代数方法，如多项式替代、幂映射、移位逆变换和特殊多项式构造，构造F_q-good zero-set。

**📊 数据集**

本研究不使用实验数据集，仅依赖理论证明。

**📈 对比分析**

通过与已知最优码表（Grassl的数据库）比较，列出的码均为最优或最佳已知码，证明方法有效。

**⚠️ 局限性**

仅适用于原码，未给出算法复杂度分析，对非原码或更一般情形的推广有限。

---

## 512. Sampling Matchings in Near-linear Time

**arXiv ID:** 2609.21936 | [PDF](https://arxiv.org/pdf/2609.21936v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

---

## 513. Automata-Theoretic Verification of Interval Markov Decision Processes

**arXiv ID:** 2609.21966 | [PDF](https://arxiv.org/pdf/2609.21966v1)

**作者:** Sarvin Bahmani `[一作]` (University of Liverpool), Ashutosh Trivedi `[通讯]` (University of Colorado Boulder)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了区间马尔可夫决策过程（IMDP）在 ω‑正则（包括 LTL）规范下的自动机理论验证与策略合成，并给出了对稳定与不稳定 IMDP 的分界与算法实现。

**💡 创新点**

创新点在于：
• 明确区分稳定 IMDP（支持相同）与不稳定 IMDP，阐释何时使用 Good‑for‑MDP 自动机，何时需要更强的 Good‑for‑Games 自动机；
• 将 IMDP 与随机奇偶游戏（stochastic parity games）关联，构造同步积并实现最优策略求解；
• 提出两种求解方法——带收缩因子的区间值迭代与基于自然策略改进的精确策略改进；
• 将数据驱动抽象方法与自动机验证框架统一，提供从样本到概率保证的完整流程。

**🔧 技术方法**

技术手段包括：
• 自动机理论（非确定性与确定性 Buchi/Parity 自动机，GFM/GFG 自动机）
• 区间 MDP 与随机奇偶游戏的构造与分析
• 值迭代与收缩因子、策略改进、局部最优自然子程序
• 线性规划与凸多面体顶点枚举
• 通过 Spot 等工具自动生成 Büchi 自动机

**📊 数据集**

实验数据集：
• DynAbs 框架下的两组基准：UAV（无人机运动控制）与 Shuttle（航天器轨道控制），模型规模从几百到数十万条边。
• 每个基准可按分区粒度生成多种规模，测试不同 LTL/GR(1) 规范。

**📈 对比分析**

性能对比：
• 对比 SI（策略改进）、VI（值迭代）与定性分析；
• SI 在迭代次数上更少，但单次成本更高；总体运行时间与 VI 相近；
• 定性分析耗时最小；
• 复杂规范（嵌套时序或安全/活性组合）导致自动机规模增大，时间显著增加；
• 在 UAV 与 Shuttle 基准上，所有方法均能处理百万级边的 IMDP。

**⚠️ 局限性**

局限性：
• 量化结果主要针对稳定 IMDP；不稳定 IMDP 需要更强的 GFG 自动机，算法复杂度上升；
• 自动机大小对算法性能影响显著，LTL 翻译导致指数增长；
• 线性规划与凸多面体顶点枚举在极大模型中仍可能成为瓶颈；
• 采用浮点实现时，数值精度可能导致策略改进循环；
• 论文未覆盖完全任意 IMDP 的多目标或分布式场景。

---

## 514. GALA: Geometry-Aware Latent Action Modeling for Vision-Language-Action Model Pretraining across Embodiments

**arXiv ID:** 2609.21948 | [PDF](https://arxiv.org/pdf/2609.21948v1)

**作者:** Yichen Liu `[一作]` (Tsinghua University), Jianyu Chen `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出GALA框架，结合图像潜在动作与3D端执行器点云几何运动，构建跨实体的视觉-语言-动作(VLA)预训练模型；

**💡 创新点**

核心创新是Unified End‑effector Motion Representation（UEMR），通过统一双手几何潜在、对称几何增强及双向过渡学习，在不需要点、关节或拓扑对应的情况下，保留细粒度动作语义并提升跨实体泛化；

**🔧 技术方法**

采用点云VQ‑VAE（Point Transformer V3 + EMA‑VQ）、DINOv2/SigLIP视觉编码、跨模态Transformer、几何重建Chamfer距离、双向学习与对齐，并在VLA训练中使用共享Diffusion Transformer与多体动作头；

**📊 数据集**

使用人类与机器人多实体数据集：EgoDex、HOI4D、DROID、RoboCasa‑GR1、XHand、Fourier‑hand、ROBOTERA XHand、Robotiq gripper 等；

**📈 对比分析**

与UniVLA、METIS、OPFA、Native Kinematics等基线对比，在RoboCasa GR‑1单体训练下成功率55.7%，跨实体预训练提升至68.3% (+12.6%); 在真实XHand四项任务上平均成功率75.5%，优于HARP‑VLA、π₀.₅等；

**⚠️ 局限性**

局限性包括对端执行器形态与尺度差异仍有影响；依赖大量多实体数据；缺失点云或极端关节动力学时性能下降；跨模态语义一致性细节仍需进一步完善。

---

## 515. Investigating the Performance and Energy Costs of Replicating Band-Split RNN for Music Source Separation

**arXiv ID:** 2609.21918 | [PDF](https://arxiv.org/pdf/2609.21918v1)

**作者:** Paul Magron `[一作]` (Université de Lorraine), Constance Douwes `[通讯]` (Aix Marseille Univ)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

复现并完整实现了Band‑Split RNN（BSRNN）音频源分离模型，开展了设计空间实验和能耗评估，并公开发布代码与预训练模型。

**💡 创新点**

首次系统地把BSSRNN完整训练流水线公开，并通过实验验证不同数据预处理、优化策略和网络参数对性能与能耗的影响，强调可复现性与绿色计算的重要性。

**🔧 技术方法**

使用STFT、双向LSTM、残差网络、MLP掩模器、组合时间域与频域损失，辅以随机裁剪、能量增益、SAD/无SAD等数据增强手段，并通过Adam、梯度累积、学习率衰减等训练技巧。

**📊 数据集**

采用公开的MUSDB18‑HQ数据集（150首音乐，4个源：人声、低音、鼓、其他），进行训练、验证与测试。

**📈 对比分析**

通过对比uSDR/cSDR指标与原论文结果，评估不同变体的性能；实验显示小模型在能耗与性能上表现优越，虽然整体性能略低于原始BSSRNN，但在可复现性与资源使用上有显著改进。

**⚠️ 局限性**

主要局限在于仍无法完全达到原论文的最佳效果（能耗高、性能差距显著）、对GPU硬件差异的依赖、SAD实现效果有限，以及实验中仍需进行多次重复以降低随机性。

---

## 516. ExpBoN: Exponential-Noise Best-of-$n$ for Efficient Test-Time LLM Alignment

**arXiv ID:** 2609.21899 | [PDF](https://arxiv.org/pdf/2609.21899v1)

**作者:** Yanxiao Liu `[一作]` (Imperial College London), Deniz Gündüz `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于指数噪声的软 Best-of-n 抽样方法（ExpBoN），并将其嵌入 Guided Speculative Inference（GSI）框架，得到 ExpGSI，实现在不降低答案准确率的前提下显著降低推理成本。

**💡 创新点**

创新点在于通过指数噪声实现精确的有限样本分解，保证了几何级数的收敛速度、对分布偏差的解析控制以及对代理奖励误差的严格上界；同时结合裁剪技术实现了可预知的早停抽样。

**🔧 技术方法**

主要技术包括指数噪声报告-最大化（exponential‑noise report‑noisy‑max）、精确的有限‑n 分解、KL 与 TV 收敛分析、裁剪奖励与拒绝采样实现，以及对 Qwen 系列大模型的多步推理。

**📊 数据集**

使用了 MATH500、MMLU‑STEM 与 Minerva Math 三个数学/科学推理基准，测试模型为 Qwen2.5‑Math（1.5B/7B）与 Qwen3（1.7B/14B），并采用 400 条子集或全部 272 条题目进行评估。

**📈 对比分析**

与传统 SBoN、原始 GSI、RSD 等基线对比，ExpGSI 在保持与 GSI 同等或略高的准确率与接受率的同时，计算量下降 14%–39%（n=2–16）且平均每步耗时减少约 18%，在更大目标模型下还能实现高达 45% 的节能。

**⚠️ 局限性**

局限性包括对裁剪阈值 C 的依赖、在代理奖励严重失配时可能加速到过度优化的目标、以及对更广泛模型/任务的适用性尚待验证。

---

## 517. Designer-RSI: Evolving Procedural Memory from User Traffic for Agentic Graphic Design

**arXiv ID:** 2609.22086 | [PDF](https://arxiv.org/pdf/2609.22086v1)

**作者:** Hongyang Du `[一作]` (Adobe), Asim Kadav `[通讯]` (Adobe)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个持续适应框架，利用冻结的基础模型和外部程序化记忆库，通过用户流量不断扩充和完善可重用的自然语言设计流程。

**💡 创新点**

提出将技能库视为学习目标，结合扩展（minting）和深化（rewriting）两种机制，并引入匹配重放门（replay gate）在无权重更新、无人工标签的情况下安全改进模型。

**🔧 技术方法**

使用冻结的前沿语言模型（Claude、Qwen等）控制超过230个 Photoshop/Illustrator/InDesign 工具，构建四个角色（提问者、求解器、评审者、反思者），并通过自然语言技能、匹配重放与对比评估实现技能演化。

**📊 数据集**

基于1,406条用户设计简报及200条保留的人工简报，外加GenEval2、DPG-Bench、OneIG、OpenCOLE、GraphicBench、CreatiDesign、BannerRequest400等公开基准。

**📈 对比分析**

与无技能基线（Base）对比，执行成功率从72.7%提升至99.3%，在专业设计基准上胜率达61.8%–67.6%，且平均生成延迟提升不到6%；组合扩展与深化后完整性提升至74.04（+5.42）并获得58.5%胜率。

**⚠️ 局限性**

局限在于自然语言技能难以强制执行、几何操作受感知与验证限制、长流程会失真，重放门只能在评估样本上保证无退化，无法保证对全部用户流量的全局提升。

---

## 518. APort Vault: Benchmarking AI Agent Payment Authorization with the Open Agent Passport

**arXiv ID:** 2609.22076 | [PDF](https://arxiv.org/pdf/2609.22076v1)

**作者:** Uchi Uchibeke `[一作]` `[通讯]` (APort Technologies Inc.), Uchi Uchibeke (APort Technologies Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本研究中，作者将4,371条真实攻击者手写提示重放到14个大型语言模型上，比较了有无Open Agent Passport (OAP)授权层的支付授权行为。

**💡 创新点**

创新之处在于首次在真实攻击提示下测量并量化执行层授权的边界效能，证明了 deterministic OAP 能消除未授权转账。

**🔧 技术方法**

技术方法包括 OAP deterministic policy 检查、工具调用 instrumentation、双模型评审 panel 以及基准重放框架。

**📊 数据集**

使用的数据集为 2026 年 CTF 活动收集的4,371条攻击示例，并在 14 个模型、5 个政策层级、2 条轨道、2 架构上进行重放。

**📈 对比分析**

对比方法为按模型/轨道/政策三元组匹配，结果显示请求率相近，但未授权转账率从 0.182% 降为 0（会话聚类上限 0.38%），展示了授权层显著提升安全性。

**⚠️ 局限性**

局限性包括单一支付工具单域、每格单次运行、不同供应商解码差异、判定面板可靠性不足、部分多轮覆盖不完整，以及潜在的数据污染。

---

## 519. Available Guardrails: Certifying Selective Prediction across ML Systems

**arXiv ID:** 2609.22048 | [PDF](https://arxiv.org/pdf/2609.22048v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 520. Predictable Failure in Multi-Hop Retrieval: Score-Distributional Confidence Scoring and Abstention

**arXiv ID:** 2609.22056 | [PDF](https://arxiv.org/pdf/2609.22056v1)

**作者:** Andre Bacellar `[一作]` `[通讯]`, Andre Bacellar

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究多跳检索错误聚类，提出基于检索特征的校准置信评分RCS，实现可放弃策略。

**💡 创新点**

创新点在于证明置信错误可归约性与检索特征互信息的关系，展示不同检索模式下特征互补性，并提出多特征逻辑回归置信评分。

**🔧 技术方法**

采用ANN分数的九维结构特征，逻辑回归模型训练，计算ECE与Brier分数，理论上证明互信息条件，实验对比AUC‑AC和CWAR。

**📊 数据集**

使用MuSiQue、2WikiMultiHopQA、HoVer三大多跳QA基准，在LLM‑judge和dense检索两种架构下评测。

**📈 对比分析**

与随机、熵、最大分数、margin、lift、query‑len、温度缩放、MLP以及合成对照等八个基线比较，RCS在所有五种失败模式下均为最佳或同等最佳AUC‑AC，CWAR从39.5%降至20.6%（50%覆盖），ECE仅0.035。

**⚠️ 局限性**

局限性包括：在dense检索中互信息低，特征信息不足导致提升有限；需预先训练且受数据量限制，且未考虑检索后生成的后处理与人机协同的更细粒度策略。

---

## 521. Particle Competition and Cooperation for Robust Graph Convolutional Network Learning Under Label Noise

**arXiv ID:** 2609.22053 | [PDF](https://arxiv.org/pdf/2609.22053v1)

**作者:** Fabricio Breve `[一作]` `[通讯]` (São Paulo State University), Fabricio Breve (São Paulo State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种在图卷积网络训练前进行粒子竞争与合作（PCC）标签去噪的混合框架（PCC+GCN），旨在提升GCN在标签噪声环境下的鲁棒性。

**💡 创新点**

创新点在于将PCC作为轻量级的标签预处理步骤，而非直接改造GCN结构或损失函数，从而实现无侵入式的去噪与高效的后续学习。

**🔧 技术方法**

主要技术包括粒子竞争与合作算法（PCC）进行标签修正、k-最近邻特征图增强、标准两层GCN训练以及GPU加速的实例依赖噪声生成器。

**📊 数据集**

实验使用NoisyGL基准中的10个图数据集（如Cora、CiteSeer、PubMed、Amazon-Computers、Amazon-Photos等），并在不同噪声模型（Uniform、Pair、Random、Instance‑Dependent）下进行评估。

**📈 对比分析**

与NRGNN、PIGNN、CP等现有鲁棒图学习方法相比，PCC+GCN在大多数噪声水平下取得了最高的平均准确率和最优平均排名；在统一噪声下在30%–50%噪声率时提升更为显著，并且在实例依赖噪声场景下保持了竞争力，且在大多数数据集上显著快于其他鲁棒方法。

**⚠️ 局限性**

局限性包括：对PCC图结构质量和超参数（如greedy步概率、距离指数、阈值、k‑NN参数）高度依赖，且在部分数据集（如Amazon‑Computers）可能反而降低性能；目前仅验证了节点分类任务，未覆盖图分类或回归等其他图学习任务。

---

## 522. Beyond Reactive Assistance: PV-Care Using Low-Density EEG and AI to Provide Proactive, Context-Aware Help for MCI

**arXiv ID:** 2609.22024 | [PDF](https://arxiv.org/pdf/2609.22024v1)

**作者:** Simon L Liu `[一作]` (Shanghai High School International Division), Manish Kumar Krishne Gowda `[通讯]` (Apple Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

设计并实现了PV‑Care，一套基于低密度EEG与视觉感知的主动辅助系统，为轻度认知障碍（MCI）用户提供实时、情境感知的语音支持。

**💡 创新点**

创新点包括：①将低密度EEG脑态识别与环境视觉信息结合，用SFR‑Net实现三态（休息、学习、记忆回想）识别；②构造“4W‑UT”结构化提示，引导LLM生成上下文化、个性化响应；③在可穿戴设备上集成EEG+摄像头+手机，形成完整闭环。

**🔧 技术方法**

技术包括：低密度EEG采集（Muse四通道）、多分辨率时频融合深度网络SFR‑Net、双向LSTM+MLP分类、OpenAI GPT‑4 Vision+ChatGPT、移动端轻量面部识别Mobile FaceNet、TTS与声纹转换。

**📊 数据集**

数据集：30位健康受试者的并行4通道/64通道EEG数据（休息、学习、记忆回想三类）；MCI筛选样本60人（20人用于主观评估）以及对应的环境图像与面部识别数据。

**📈 对比分析**

与9个基线模型（EEGNet、Tsception、Conformer、MSTCNN、CNNLSTM、BMFCNet、TS‑SEFFNet、TF‑HybridNet、CE‑stSENet）以及不同深度、频率预处理方法对照，SFR‑Net在5折交叉验证下平均准确率0.804、F1 0.804、AUC 0.94，显著优于最接近的BMFCNet（0.786）。ablation实验表明频域分支最关键，低分辨率分支进一步提升稳定性。

**⚠️ 局限性**

局限性：①低密度EEG缺乏空间信息，跨通道建模受限；②干电极设备的信噪比和稳健性有待提升；③LLM生成文本易出现幻觉，提示设计仍需优化；④系统依赖云端LLM，存在延迟与隐私问题；⑤评估样本规模有限，需在更大多样化MCI人群中验证。

---

## 523. On (Directed) Width-Parameters of Geometric Spanners

**arXiv ID:** 2609.22082 | [PDF](https://arxiv.org/pdf/2609.22082v1)

**作者:** Kevin Buchin `[一作]` (TU Dortmund), Torben Scheele `[通讯]` (TU Dortmund)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

本文研究了在几何图中构造受不同图参数（如路径宽度、分支宽度、割宽度、团宽度、秩宽度、树深度及其有向版本）约束的 t‑spanner，并给出了相应的膨胀因子上界与下界。

**💡 创新点**

创新点在于首次统一证明了多种图参数下的膨胀上界为 𝒪(n/k^{d/(d-1)}) 并且这一上界在最坏情况下是最优的；同时发现树深度参数无法给出任何膨胀上界，进一步证明了对应优化问题的 NP‑难性，并提出了一个 XP‑算法给出 2‑近似解。

**🔧 技术方法**

主要技术包括：对 EMST 进行分块并构造路径型子图、利用 𝒞‑树分解与路径分解的递归关系、构造多层 Steiner 点决策树、利用严格花束与强连通子图收缩证明有向参数下的下界、以及将几何度量空间映射到实数轴以获得线性膨胀的路径。

**📊 数据集**

论文未使用任何实验数据集，全部工作基于理论证明与构造。

**📈 对比分析**

与已有的基于树宽、路径宽度或无向图的膨胀分析方法相比，本文的结果在最坏情况下达到 𝒪(n/k^{d/(d-1)}) 的极限，并证明了在有向情形下仍保持此极限，体现了理论上的最优性能。

**⚠️ 局限性**

局限性包括：树深度参数下无法给出膨胀上界、对于宽度（band‑width）等更严格参数的结果仍未得到；此外对树宽、路径宽度下最小膨胀树/路径的近似算法仍是开放问题。

---

## 524. Hermite-Fisher bounds and stability for min-entropy power inequalities

**arXiv ID:** 2609.22065 | [PDF](https://arxiv.org/pdf/2609.22065v1)

**作者:** Silouanos Brazitikos `[一作]` (University of Crete), Tomasz Tkocz `[通讯]` (Carnegie Mellon University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `14d48e9d-0069-4ad9-996a-1d5968216998` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过变分原理与正交化的Hermite多项式测试函数相结合，推导了相对Fisher信息的显式下界，得到与低阶累积量（cumulants）相关的渐近锐度边界，并进一步给出了Gaussian熵亏缺的下界；随后在所有维度下给出了量化的最小熵功率不等式（min‑entropy power inequality）的稳定性与等号情形，并对Brzezinski的球体截面边界做了稳定性改进。

**💡 创新点**

创新点包括：① 变分原理与Hermite多项式的结合实现了相对Fisher信息与累积量之间的直接联系；② 通过这一工具给出了相对Fisher信息和Gaussian熵亏缺的渐近最优下界；③ 在所有维度上给出了量化的min‑entropy功率不等式，并完整描述了等号与稳定性；④ 对球体截面体积的上界进行改进，为凸几何和信息理论提供了新的交叉工具。

**🔧 技术方法**

主要技术：变分原理、Hermite多项式正交化、累积量（cumulant）展开、de Bruijn恒等式、Edgeworth展开、Brunn–Minkowski/截面不等式、Brzezinski的切片不等式、Fourier分析与Bessel函数积分、凸几何与稳定性分析。

**📊 数据集**

无（纯理论推导，未使用实验或公开数据集）。

**📈 对比分析**

与此前工作（如Bobkov‑Chistyakov、Madiman‑Melbourne‑Xu）相比，本文的下界常数实现了渐近最优，并提供了全维度的量化稳定性；对min‑entropy功率不等式的常数从1/e提升至1/2（在一维/二维时），并给出了等号情形的完整描述；在高维时严格不等式进一步证明了原结果的唯一性。

**⚠️ 局限性**

局限性：① 需要随机变量满足到6阶矩有限且相对Fisher信息可控；② 仅适用于独立随机向量；③ 对于权重向量不满足特定条件时，累积量的消除可能导致界失效；④ 在维数≥3时只能得到严格不等式，等号情况仅在低维得到完整刻画。

---

## 525. How Researchers Use and Verify AI Coding Assistants: Tasks and Validation Practices in Scientific Programming

**arXiv ID:** 2609.22049 | [PDF](https://arxiv.org/pdf/2609.22049v1)

**作者:** Gabrielle O'Brien `[一作]` (University of Michigan), Nasir Eisty `[通讯]` (University of Tennessee)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了2025年美国科研人员在编程中使用生成式AI工具的实际案例，分析了他们分配给AI的任务类型、评估策略以及对工具与自身能力的信心。

**💡 创新点**

首次系统性挖掘科学家在实际编程情境中如何交互、验证AI生成代码，并揭示了“无结构化验证、个体判断”是主要实践方式，同时阐明了经验与信心之间的关系。

**🔧 技术方法**

使用了混合方法：定性内容分析（对自由文本账户进行编码）、定量统计分析（t检验、χ²检验、线性回归、Benjamini–Hochberg校正）来检验使用案例与评估策略与编程经验、研究领域及信心的关联。

**📊 数据集**

基于2025年跨机构调查的自由文本回答，样本包括数百名在美国高校及科研机构从事科研编程的研究人员；账户数约为400余条，涵盖了数据处理、可视化、调试、数学/科学计算和统计分析等任务。

**📈 对比分析**

方法通过对比不同经验水平与研究领域的使用与评估模式进行描述性统计；未使用外部基准；结果显示无明显差异，主要表现为经验较低者更信任AI，经验较高者更信任自身；验证策略以“运行代码”居多，且与信心无显著相关。

**⚠️ 局限性**

局限性包括：自报单一回忆案例易受记忆与社交期望偏差；评估策略可能被低估；样本集中于美国高等教育机构，缺乏跨文化普适性；编码过程非盲法，可能存在确认偏差；未能捕捉实际操作细节。

---

## 526. Auditing bipartite motif interpretations: a worked example with conservation checks and open-path decomposition

**arXiv ID:** 2609.22014 | [PDF](https://arxiv.org/pdf/2609.22014v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 527. Gricea: An Open Science Platform for Conversational AI Research

**arXiv ID:** 2609.22039 | [PDF](https://arxiv.org/pdf/2609.22039v1)

**作者:** Nikhil Sharma `[一作]` (Johns Hopkins University), Ziang Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `b88c6eac-d57a-4623-a604-1f401f3eb268` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Gricea 平台，能够将对话式 AI 研究设计以可配置、可执行的研究工件形式表示，并提供无代码可视化创作、Study Flow 与 Task Flow 两级抽象以及对实验过程和任务行为的完整保留。

**💡 创新点**

其创新点在于将实验程序与对话任务行为统一为可执行图形化表示，支持版本化、可检验、可复现，显著降低研究人员搭建对话式 AI 实验的技术门槛。

**🔧 技术方法**

技术实现基于图形化节点编辑器、声明式运行时、分层架构（研究流程/任务流程）以及浏览器与服务器端的 Instrumentation，内置对 LLM 推理、检索、语音等多模态交互的支持。

**📊 数据集**

研究以 CUI 2026 会议上 29 篇可复制的对话式 AI 研究为主要数据集，验证了 Gricea 在实际实验配置中的可复现性。

**📈 对比分析**

通过对 27 篇论文的完整或部分复制（93% 复制率）以及 10 位研究者的可用性评估（平均使用度 6.1/7、复现可信度 5.9/7），表明 Gricea 能有效覆盖现有研究并被多学科研究者快速使用。

**⚠️ 局限性**

限制包括评估样本规模有限、未涵盖小组协同、物理机器人等更复杂方法、对外部依赖的处理仍不完善，以及需要进一步完善 AI 辅助创作和方法论指导。

---

## 528. QuranicMMLU: A Cognitively-Aware Benchmark for Evaluating Generative AI Solutions on Quranic Linguistic Knowledge

**arXiv ID:** 2609.22038 | [PDF](https://arxiv.org/pdf/2609.22038v1)

**作者:** Rawan El Ghali `[一作]` (Qatar Computing Research Institute, Hamad Bin Khalifa University), Ehsaneddin Asgari `[通讯]` (Qatar Computing Research Institute, Hamad Bin Khalifa University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 QuranicMMLU 基准，包含 5 大语言层面、31 个叶子节点的词汇表，生成 980 道多轮问题（开放式与多选）并评测 12 语言模型。

**💡 创新点**

创新性地将 Quranic Arabic 语言学与 Bloom 认知层次结合，采用 LLM 生成、双格式问答、LLM-as-judge 自动评分和人工审核，揭示多选与开放式评测差异。

**🔧 技术方法**

使用 Claude 生成问题，GPT-5.2 与 Gemini 进行注释与评估，Gemini 作为自动评分者，结合 Bloom 认知层次和词汇困惑度（perplexity）进行分层。

**📊 数据集**

依托 Quran.com、Quranic Arabic Corpus、Tarteel 释义集、Asbab al-Nuzul 数据集等权威来源进行语料检索与答案校对。

**📈 对比分析**

通过 MCQ 准确率与 Gemini 自动评分两种指标评测模型，12 系统在 MCQ 上平均准确率 84%，开放式评分平均 60%，两种排序相关性 Kendall τ=0.73，Ansari 等专门 Islamic 模型名列前茅。

**⚠️ 局限性**

局限在于词汇层面覆盖不完全、开放式评分仅靠 LLM 可能产生偏差、仅评测开源模型、部分专业系统未覆盖、生成模型可能留存瑕疵。

---

## 529. $λ$-Controlled GRPO: Turning Flow-Matching Ratio Instability into a Budgeted Resource

**arXiv ID:** 2609.22041 | [PDF](https://arxiv.org/pdf/2609.22041v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 530. The Supersingular Isogeny Problem in Time and Memory $p^{1/3+o(1)}$, Unconditionally

**arXiv ID:** 2609.22018 | [PDF](https://arxiv.org/pdf/2609.22018v1)

**作者:** José Luis Delgado `[一作]` `[通讯]` (Independent Researcher), José Luis Delgado (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种无光滑性假设的Las Vegas算法，用于在超奇异椭圆曲线上求取非标量端点算子。

**💡 创新点**

核心创新在于通过碰撞界定（collision bound）证明在预先确定的平方自由平滑次数集合中，存在大量曲线可得到相应的共轭等价同源，从而实现1/3指数复杂度。

**🔧 技术方法**

利用椭圆曲线同源计数、类群与二次域嵌入、Vélu公式、随机2-同源图游走、以及颜色编码匹配等技术。

**📊 数据集**

实验并无特定数据集，理论分析以素数p的位长n为参数，证明对于任意大p均成立。

**📈 对比分析**

与此前Wesołowski与Udovenko的算法相比，本方法在不依赖光滑性假设的情况下保持p^{1/3+o(1)}的时间与空间复杂度，并提供严格的无条件分析。

**⚠️ 局限性**

限制在于常数项大、实现复杂度高，且仅适用于大位长p，尚缺乏对小p的高效实现及进一步降低空间开销的方案。

---

## 531. PRIME: Perception Feedback with Situational Memory Embeddings in VLA Models

**arXiv ID:** 2609.22040 | [PDF](https://arxiv.org/pdf/2609.22040v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 532. A lower bound for $\langle 3,2,m \rangle$ matrix multiplication

**arXiv ID:** 2609.22054 | [PDF](https://arxiv.org/pdf/2609.22054v1)

**作者:** Askar Tsyganov `[一作]` (HSE University), Maxim Rakhuba `[通讯]` (HSE University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了任意域上3×2矩阵与2×m矩阵相乘的双线性复杂度严格大于24m/5，并在m=5时得到确切秩为25的结论。

**💡 创新点**

创新点在于通过新的代数归约和对等号情况的排除，完善了先前的下界，并使用形式化证明彻底消除了人类证明中的潜在错误。

**🔧 技术方法**

主要技术包括张量秩与双线性复杂度的理论框架、线性映射与子空间的维度不等式、对称性与归一化变换，以及 Lean4 的形式化验证工具。

**📊 数据集**

该工作完全基于理论推导，不涉及实验数据集。

**📈 对比分析**

与 Hopcroft‑Kerr 上界相匹配，证明上下界在3×2×5情形下确切相等；性能方面表现为理论证明的严谨性而非数值实验。

**⚠️ 局限性**

局限性：仅解决了3×2×m 的情形，尚未推广到更一般的张量；下界的提升仍停留在 24m/5 的改进上；形式化验证依赖于现有 Lean4 证明库，若要进一步扩展可能需要大量额外工作。

---

## 533. Spherical Harmonic Sliced Wasserstein Displacement Interpolation for Acoustic Source and Reflection Density Modeling

**arXiv ID:** 2609.22028 | [PDF](https://arxiv.org/pdf/2609.22028v1)

**作者:** Yuancheng Luo `[一作]` `[通讯]` (NuSpace AudioCambridge), Yuancheng Luo (NuSpace AudioCambridge)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了基于SOMS‑SH（Sum‑of‑Magnitude‑Square Spherical Harmonic）展开的空间声源和回声密度估计方法，并利用球面切片Wasserstein距离实现移动声源的SRIR（空间房间脉冲响应）插值。

**💡 创新点**

创新点包括：1）将非负密度函数表示为SOMS‑SH形式，保证非负性且易于优化；2）通过最大似然SDP优化将变量数量从二次降为线性；3）提出解析逆变换采样方案，加速密度评估；4）将球面切片Wasserstein距离与SOMS‑SH结合，得到可解的NNLS‑SSW插值，优于传统线性和乘积插值。

**🔧 技术方法**

技术方法主要包括：Spherical Harmonic（球面谐波）展开、Sum‑of‑Squares (SOS) 与 Semidefinite Programming (SDP)、非负最小二乘（NNLS）、球面切片Wasserstein距离（SSW）、逆变换采样、图像源模型（ISM）与高阶Ambisonics。

**📊 数据集**

数据集为室内模拟数据：在尺寸为7×8×5 m的矩形房间中，墙面反射系数均为1/3，声源沿两点间直线轨迹（(1,1,1)到(1,1,0)）采样，接收机固定在(0.1,0.3,0.5)处，采集150 ms的RIR并用ISM展开得到地面真值密度。

**📈 对比分析**

比较方法：对端点t=0和t=1的密度使用线性插值、乘积插值和球面切片Wasserstein插值，随后计算整个路径的SSW距离。实验表明，Wasserstein插值在所有时间步上均取得最低SSW距离，且在t=0.5时呈现更自然的源移动和回声分布。

**⚠️ 局限性**

局限性：1）SOMS‑SH展开受限于最大阶数，过高阶会增加计算量；2）SSW距离的离散化（旋转方向和u分辨率）可能导致负值残留；3）方法基于模拟数据，未在实际测量环境中验证；4）对高动态场景的实时计算仍有挑战。

---

## 534. Cross-sector generalization of accident-process role classification in occupational accident narratives

**arXiv ID:** 2609.22081 | [PDF](https://arxiv.org/pdf/2609.22081v1)

**作者:** Aho Yapi `[一作]` (Université Clermont Auvergne), Yan Bailly `[通讯]` (LYF SAS)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了职业事故叙事的事故过程角色分类，并在建筑行业的标注语料上训练模型，随后在未见的化工‑塑料、冶金行业以及一家外部公司产生的事故记录上进行跨域评估。

**💡 创新点**

①将事故叙事拆分为事实单元并定义四个功能角色（A0：工作情境，A1：显式不良条件，B：事故事件/偏差，C：报告结果）；②系统比较了冻结预训练编码器、交叉熵微调以及监督表示学习（批量硬三元组、监督对比学习、SoftTriple）在跨行业迁移中的表现；③提供了一个可迁移的辅助编码框架，帮助在不同报告环境中统一结构化事故信息。

**🔧 技术方法**

使用多语言预训练编码器 Qwen3‑Embedding‑0.6B；在其上进行：(1) 冻结+线性分类器（LogReg、RandomForest、XGBoost）；(2) 交叉熵微调（可选投影层）；(3) 监督表示学习（Batch‑Hard Triplet、SupCon、SoftTriple）并随后训练线性分类器；对比基线 TF‑IDF+LogReg。

**📊 数据集**

数据集：EPICEA 法国职业事故数据库中的三行业语料（建筑、化工‑塑料、冶金）共约70,000+事实单元；外部公司（工业清洁服务）语料约9,500+事实单元；全部以事实单元为单位进行标注，采用专家双标并一致化。

**📈 对比分析**

通过源域（建筑）交叉验证选取超参，在目标域（化工‑塑料、冶金、公司）仅进行一次评估。结果显示：冻结+LogReg ≈76.9%均衡准确率；交叉熵微调 ≈85.8%；监督对比学习 ≈85.6%；SoftTriple ≈85.8%；TF‑IDF基线 ≈75.0%。三种微调方法在目标域表现相近，差异不显著但均显著优于基线和冻结策略。

**⚠️ 局限性**

限制：仅在单一语言/地区内测试，未探究组织报告差异对分割与模型的独立影响；对罕见或模糊边界（如 A0/A1、B/C）仍存在高误差；未实现因果推断或自动化预防措施建议；对不同领域的迁移偏移机制缺乏深入解析。

---

## 535. A Sociotechnical Review of Algorithms in Health Systems: Technical, Cost, and Human-Centered Considerations

**arXiv ID:** 2609.22070 | [PDF](https://arxiv.org/pdf/2609.22070v1)

**作者:** Victoria Chui `[一作]` (University of Toronto), Shion Guha `[通讯]` (University of Toronto)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对114篇关于健康系统中成本意识AI模型的文献进行系统综述，分析其方法、预测变量、结果和成本考量。

**💡 创新点**

提出了模型成本的四维框架（财务、计算、组织、人社），并将其与人本设计原则结合，指出现有研究中成本与公平性缺失的空白。

**🔧 技术方法**

采用系统检索、PRISMA流程、主题分析与交叉表统计等方法，结合HCAD框架进行编码。

**📊 数据集**

使用来自ACM DL、IEEE Xplore、Springer、PubMed等数据库检索的114篇论文及其引用的原始数据集（如EHR、影像、索赔数据）。

**📈 对比分析**

通过对方法、预测变量、结果与成本维度的交叉分析，揭示模型成本被低估且大多强调“降低医疗支出”，但实际性能指标（如AUC、准确率）未统一比较，显示评估标准缺乏一致性。

**⚠️ 局限性**

仅纳入英文且技术细节完整的论文，排除了综述和调查研究，样本主要集中在美国，未量化实际成本且缺乏跨币种时间校准，限制了结论的普适性。

---

## 536. CodeMidas: Scaling Agentic Coding RL Environments from Code Itself

**arXiv ID:** 2609.22068 | [PDF](https://arxiv.org/pdf/2609.22068v1)

**作者:** Bowen Ye `[一作]` (LLM Core, Xiaomi), Fuli Luo `[通讯]` (LLM Core, Xiaomi)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出 CodeMidas，一个从现有开源代码自动构建可执行编码强化学习环境的流水线；

**💡 创新点**

创新点在于仅使用源代码作为任务输入，自动生成任务声明、可执行测试和验证器，并通过后期过滤确保任务安全可靠；

**🔧 技术方法**

主要技术包括代码可执行性分析、行为任务声明推导、基于参考执行的测试构造、环境一致性检查以及使用 GRPO 对 MiMo‑V2.5 进行训练；

**📊 数据集**

使用了 3,185 个开源代码库生成的 5,545 个可验证任务，覆盖 23 种编程语言与 15 个技术领域；

**📈 对比分析**

通过在五大公开基准（DeepSWE、ProgramBench、SWE‑bench Pro、RepoZero C2Rust、Terminal‑Bench v2.1）上评测，训练后相对基线提升显著：DeepSWE 10.0%→21.7%，ProgramBench Almost Solved 4.5→21.5%，Terminal‑Bench 63.7%→72.2% 等；同时高质量任务集（5,545）优于未经过滤的 8,000 任务集；

**⚠️ 局限性**

局限性包括：仍需人工或自动化方式保证任务完整性和安全性；生成的任务局限于已实现功能，难以覆盖全新需求；虽然后期过滤减少泄漏，但仍可能存在未发现的漏洞；以及对某些语言或大型项目的支持仍有限。

---

## 537. BrainWideBench: Benchmarking large-scale pretraining and across-animal transfer in multi-region neural recordings

**arXiv ID:** 2609.22064 | [PDF](https://arxiv.org/pdf/2609.22064v1)

**作者:** Alexandre Andre `[一作]` (University of Pennsylvania), Eva L. Dyer `[通讯]` (University of Pennsylvania)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的跨动物、多区域神经记录迁移评估基准（BrainWideBench），并在其上评估多种预训练模型；

**💡 创新点**

通过整合行为解码、神经活动预测和脑区识别三大任务套件，构建了一个全面衡量神经表征泛化能力的标准化评测框架；

**🔧 技术方法**

采用Transformer、状态空间模型、Masked Prediction、Self‑Supervised 等多种预训练技术，并对比单会话基线；

**📊 数据集**

使用国际脑实验室（IBL）Brainwide Map 数据集，该数据集涵盖多只小鼠、数十个脑区、上千小时的单细胞电位和同步行为记录；

**📈 对比分析**

在行为解码、神经预测与脑区分类三个任务中，预训练模型普遍优于单会话基线，表现出跨动物的良好迁移能力，但不同预训练目标在各任务上的优势不一；

**⚠️ 局限性**

基准仅涉及单一物种、单一记录模态和单一任务，且测试动物数量有限（13只），导致模型排名对具体留出动物较为敏感，且未涵盖跨实验室、跨模态或多物种的泛化情况；

---

## 538. Duty Factor Predicts Robust Constrained Quadrupedal Locomotion Across Gait Types

**arXiv ID:** 2609.22073 | [PDF](https://arxiv.org/pdf/2609.22073v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 539. SeeQ: Training Generalist Value Functions for Long-Horizon Robotic Manipulation

**arXiv ID:** 2609.22085 | [PDF](https://arxiv.org/pdf/2609.22085v1)

**作者:** Saksham Singh `[一作]`, Aviral Kumar `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种Subtask-elicited Q-函数（SEQ），通过先用预训练的视觉语言模型预测当前子任务并估计其价值，从而在长时程机器人操控中改进通用策略的决策。

**💡 创新点**

创新点在于：①将Q函数的回报目标从稀疏的全任务成功转为子任务级别，显著缩短信用分配时程；②在推断时自动生成子任务文本，避免部署时需要人工标注；③结合最佳‑N 倍预备（TD‑BoN）实现更稳健的TD更新。

**🔧 技术方法**

使用技术包括：大型视觉语言模型PaliGemma（3B参数）作为主干；多阶段子任务标注的监督式next-token损失；TD‑BoN目标与最佳‑N 采样备份；预训练与微调阶段的两步学习；以及基于自然语言的子任务解码与条件化价值预测。

**📊 数据集**

训练数据主要来自公开机器人数据集RoboCOIN（约40k条含子任务注释）进行预训练；下游微调使用四个真实世界双臂操控任务的数据：shirt-hang、lid-sealing、grocery-packing、lego-disassembly，分别收集了RaC和人类远程操控的数据。

**📈 对比分析**

在四个任务上，SEQ在最佳‑N 策略引导下的成功率均大幅提升：shirt-hang从10/24提升至22/24，lid-sealing从10/24提升至15/24，grocery-packing从9/24提升至17/24，lego-disassembly从5/24提升至10/24。与任务级MC、任务级TD、子任务级SARSA等基线相比，SEQ在所有任务上均取得显著更高的成功率。

**⚠️ 局限性**

局限性包括：①需要训练时的子任务标注，标注不一致或模糊会影响学习；②子任务预测错误可能导致价值估计偏离整体目标；③仅通过最佳‑N 采样对基准策略进行引导，限制了探索空间；④对长期后果的全局评估缺失，可能导致局部最优行为。

---

## 540. OmniVBench: A Benchmark and Large-Scale Dataset for Omni Reference-to-Video Generation

**arXiv ID:** 2609.22069 | [PDF](https://arxiv.org/pdf/2609.22069v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 541. MintAct: A Unified Visual Agent for Digital Environments

**arXiv ID:** 2609.22083 | [PDF](https://arxiv.org/pdf/2609.22083v1)

**作者:** Mingfei Gao `[一作]` (Apple), Afshin Dehghan `[通讯]` (Apple)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一系列规模为2B、4B、8B的统一视觉代理模型，能够在同一模型上完成UI定位、跨移动、桌面与Web平台的多步导航，以及基于视觉的工具调用。

**💡 创新点**

创新点包括：① 多阶段训练策略（SFT→域专属RL→RFT→联合RL），使模型在多域任务上保持或提升性能；② 采用异步RL框架，显式控制跨域训练分布，解决了环境异质性与长轨迹带来的不稳定性；③ 引入合成移动/桌面环境，显著提升低数据域的表现，且可在无真实交互的情况下实现训练。

**🔧 技术方法**

技术栈包括：多模态大模型（基于Qwen3-VL-Instruct），监督微调（SFT）与强化学习（RL），异步RL分布式训练框架，函数调用式工具接口，合成环境模拟，RFT（拒绝采样蒸馏）等。

**📊 数据集**

使用的数据集有：MMBench-GUI、ScreenSpot-v2、UI-Vision、OSWorld-G、AndroidControl、AndroidWorld、OSWorld-Verified、Weblica、Online-Mind2Web、MM-ToolSandBox；同时构建了合成的移动与桌面交互环境，用于训练和评估。

**📈 对比分析**

对比方法：与同尺寸的专用域模型（如Ferret-UI Lite、UI-TARS、ScaleCUA等）以及公开大模型（如Qwen3-VL）进行基准测试。8B模型在OSWorld-Verified 48.9、Weblica 74.7、UI-Vision 56.6、OS-World-G 64.5、AndroidWorld 67.0等指标上超越或匹配了这些专用/公开模型；每个训练阶段的 ablation 实验表明其对最终性能都有显著贡献；联合RL阶段进一步提升了多域表现并保持了单域优势。

**⚠️ 局限性**

限制：① 目前联合RL仅覆盖移动与桌面，尚未扩展至所有域（Web、工具使用）导致潜在收益未完全释放；② 交互上下文随轨迹增长导致记忆与计算成本上升，需要更高效的上下文管理；③ 工具调用与像素级导航仍是分离的能力，缺乏动态切换机制；④ 合成环境的可迁移性在高数据域下降低，需进一步改进跨域迁移；⑤ 现有函数调用接口不支持动态工具发现，导致重构提示，影响缓存与效率。

---

## 542. Gripper-Aware Automatic Dense Packing of Irregular Objects

**arXiv ID:** 2609.22062 | [PDF](https://arxiv.org/pdf/2609.22062v1)

**作者:** Tianhao Qin `[一作]` (Worcester Polytechnic Institute), Jing Xiao `[通讯]` (Worcester Polytechnic Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一套闭环、实时的机器人打包管线，能够在真实 Franka Panda 机械臂上将不规则物体稠密放入容器，集成了感知、抓手感知优化、力监控的垂直下降、放置后推压整合以及每次放置后重新感知。

**💡 创新点**

主要创新点包括：① 将物体与抓手视为单一复合体，在 GPU 上用 CMA‑ES 对 5 自由度进行连续搜索；② 采用力/扭矩监测的连续下降和放置后推压补偿，实时吸收感知与接触漂移；③ 在每次放置后重新感知容器状态，避免漂移堆叠导致的错误。

**🔧 技术方法**

使用的技术：层次球树（hierarchical sphere‑tree）对点云进行几何表示；CMA‑ES 在 GPU 上批量评估多目标函数；RealSense 深度相机+ICP、RANSAC、DBSCAN 进行目标识别与姿态估计；力/扭矩传感器实现力导引下降；Post‑release consolidation push 通过轻推消除残余侧向间隙。

**📊 数据集**

数据集：七个自制 3D 打印不规则物体（平面、曲面、凹凸）以及八个 YCB 子集（香蕉、肉盒、罐头等），所有物体均预注册网格与单一对抗抓取点。

**📈 对比分析**

与 Wang & Hauser 的高度图最小化（HM）基线在相同感知、抓取与执行条件下比较；在 Optimized 排序上，本方法空间利用率达 95.6%（HM_Low 77.7%，HM_High 82.4%），平均放置物体 6.6（HM_Low 5.0，HM_High 5.4），端到端成功率 5/5；Ablation 实验显示抓手感知优化、推压补偿和使用完整网格几何是提升密度与成功率的关键。

**⚠️ 局限性**

局限性：依赖预注册的单一抓取点，无法处理新物体；放置后不重新估计已放物体姿态，感知噪声可能导致容器状态估计误差；目前仅适用于固定尺寸、平面底部的容器，未考虑动态更新或更复杂的环境约束。

---

## 543. Traffic Sign Recognition for Autonomous Driving Using Branched YOLOv2 and Geometric Features

**arXiv ID:** 2609.22060 | [PDF](https://arxiv.org/pdf/2609.22060v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 544. An Interpretable Memory Decision Controller for LLM Agents Based on Three-Signal Complementarity: Decoupling Confidence and Consistency

**arXiv ID:** 2609.22043 | [PDF](https://arxiv.org/pdf/2609.22043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 545. LIMBO: Learning and Internalizing Model-Free Barrier Objectives for Agile and Safe Whole-Body Control

**arXiv ID:** 2609.22075 | [PDF](https://arxiv.org/pdf/2609.22075v1)

**作者:** Jake Gonzales `[一作]` (Amazon), Manikantan Nambi `[通讯]` (Amazon)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了LIMBO框架，利用黑盒转移学习到的Q‑CBF安全证书，并将其安全结构通过教师反馈蒸馏进任务策略，实现全身控制的安全与高效配合；

**💡 创新点**

创新点在于：①通过残差控制的Q‑CBF实现高维控制空间的安全证书可学习；②引入风险导向的边界采样提升对可恢复边界的探索；③用教师校正的无在线滤波方式将安全结构直接内化到策略；

**🔧 技术方法**

核心技术包括：Q‑CBF（状态动作安全值函数）、残差控制接口、风险导向重放采样、教师校正与对比反馈、PPO任务学习、对抗性运动先验与域随机化；

**📊 数据集**

实验使用MuJoCo模拟Unitree G1 29-DoF humanoid的数据，训练场景包括伪造投掷的 dodgeball 和低障碍 limbo；还在硬件上使用合成轨迹与人类投掷数据进行验证；

**📈 对比分析**

与传统CBF‑RL及冻结基准对比，LIMBO在仿真和硬件上均实现更低的击中率、无跌倒、降低扭矩饱和度、更小的漂移和俯仰角；在离散分布与外域测试中均优于CBF‑RL；

**⚠️ 局限性**

局限性包括：仍需预先设定基准控制器、对安全证书的逼近误差与蒸馏误差敏感、需要充分的边界采样探索才能得到多样化策略、理论安全保证依赖于Lipschitz连续性假设，且目前仅验证于类人机器人任务。

---

## 546. Value-Sensitive Delegation in Everyday AI Agent Use: Evidence from OpenClaw

**arXiv ID:** 2609.22067 | [PDF](https://arxiv.org/pdf/2609.22067v1)

**作者:** Renkai Ma `[一作]` (University of Cincinnati), Lingyao Li `[通讯]` (University of Arizona)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用价值敏感设计（VSD）与大型语言模型辅助编码，系统分析了73,093条Reddit关于OpenClaw使用的第一人称贴文，探讨用户赋予AI代理的价值、对应的代理特性、价值实现情况以及由此产生的用户结果。

**💡 创新点**

首次提出“价值敏感代理（value‑sensitive delegation）”概念，揭示价值多聚焦于用户在代理运行前后设定的运行条件（成本、权限、监督点）而非仅仅是任务输出，并量化不同价值在不同代理特性下的实现率与用户体验之间的关系。

**🔧 技术方法**

采用GPT‑5 mini进行大规模编码，结合人工验证，构建价值、代理特性、价值实现、用户结果四维编码框架；运用统计检验（χ²、Cramér's V、间接标准化）与主题分析进行量化与质性分析。

**📊 数据集**

使用自2026年1月31日至4月26日在Reddit上抓取的OpenClaw相关帖子与评论，共1,100,308条，最终筛选并编码73,093条第一人称经验贴，形成研究数据集。

**📈 对比分析**

通过与现有技术评估与HCI研究对照，比较不同价值组与代理特性的实现率；发现自主操作和可负担操作的实现率高于预期，其余四组低于预期；进一步关联任务完成度、资源成本、风险暴露等用户结果，说明价值实现与用户体验的关联性。

**⚠️ 局限性**

研究仅为观察性、帖子级别，无法追踪单一用户；LLM编码在稀有类别上误差较大；缺乏因果推断；价值组划分未经过测量模型验证；对信任校准与恢复行为的评估有限。

---

## 547. Benchmarking World Models for Continual Learning on Compositional Tasks

**arXiv ID:** 2609.22055 | [PDF](https://arxiv.org/pdf/2609.22055v1)

**作者:** Haoyu Zhou `[一作]` (University of Oxford), Ingmar Posner `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于世界模型的组合式持续学习基准，并在机器人操作任务中评估了多种模型。

**💡 创新点**

通过让学习序列以组合任务收尾来隔离知识重用，且将组合沿动作与感知两轴拆分；同时对世界模型进行任务无关骨干与任务特定头的原则性分离。

**🔧 技术方法**

使用世界模型（DreamerV3、TD‑MPC2、PWM）、传统持续学习方法（ER、EWC、PackNet、Fine‑tuning）以及模块化Mixture‑of‑Experts架构，并引入冻结编码器诊断。

**📊 数据集**

基于Meta‑World的六套组合任务，全部在仿真环境下完成；未使用真实机器人数据。

**📈 对比分析**

在前向/后向转移（FWT、BWT）指标下对比Fine‑tuning、ER、EWC、PackNet和PWM；DreamerV3在两种骨干上均优于TD‑MPC2；传统方法只能在转移与遗忘之间做权衡；PWM在冻结编码器条件下几乎消除遗忘，前向转移与单体模型持平。

**⚠️ 局限性**

仅限仿真、单一模型规模/专家数量、缺乏真实机器人部署、依赖稠密奖励与oracle归一化，模块化优势受限于预先冻结的编码器。

---

