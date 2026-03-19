# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-03-19 | 今日论文总数: 517

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Do Understanding and Generation Fight? A Diagnostic Study of DPO for Unified Multimodal Models

**arXiv ID:** 2603.17044 | [PDF](https://arxiv.org/pdf/2603.17044v1)

**作者:** Abinav Rao `[一作]`, Sujan Rachuri `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统评估了在统一多模态模型（Janus-Pro）上同时对图像理解和生成任务使用 Direct Preference Optimization（DPO）的效果。

**💡 创新点**

创新点在于发现生成任务梯度因 VQ 代码块数量差异导致的幅度不平衡和正交性，从而解释 DPO 无法提升生成质量，并提出通过梯度幅度平衡来保护理解性能。

**🔧 技术方法**

技术方法包括 DPO、梯度正交性与幅度比分析、梯度加权平衡、PCGrad、Rewarded Soups、LoRA 微调、SigLIP 编码器、VQ‑VAE 解码器以及 CLIPScore 等评估指标。

**📊 数据集**

实验使用 COCO val2017 数据集生成 VQA 和生成偏好对，包含 1,300 条理解偏好对与 288 条生成偏好对，并对比模型与模型、真实与生成图像的偏好。

**📈 对比分析**

在 1B 与 7B 两规模模型下对比七种训练策略和两种后置方法，结果显示理解任务的 VQA 得分可提升（≈+0.26），但无论采用何种策略，生成的 CLIPScore 均未显著提升，保持在约 26.7–27.0 之间。

**⚠️ 局限性**

局限性包括仅针对 Janus-Pro 结构实验，VQ 离散化导致生成任务本身对偏好信号无响应，偏好对规模有限（150–288 对），并未验证更大数据集或在线 RL 方法；此外仅使用 LoRA 低秩微调，未探究完整微调或更高秩可能的影响。

---

## 2. From Drop-off to Recovery: A Mechanistic Analysis of Segmentation in MLLMs

**arXiv ID:** 2603.17228 | [PDF](https://arxiv.org/pdf/2603.17228v1)

**作者:** Boyong Wu `[一作]` (Technical University of Munich), Zeynep Akata `[通讯]` (Technical University of Munich)

**通讯引用:** 16266 | [OpenAlex ID](https://openalex.org/A5040372929)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探究多模态大语言模型（MLLM）在像素级语义分割中的表示演化，发现适配器会导致分辨率衰减但LLM层可通过自注意力恢复分割质量。

**💡 创新点**

首次提供层级线性探测与注意力打断实验，揭示交叉注意力驱动自我细化，并证明双向注意力可缓解因因果遮蔽导致的早期上下文匮乏。

**🔧 技术方法**

使用层级线性探测、注意力打断、双向注意力掩码以及线性分类器等技术，并在Vicuna‑7B+CLIP、DINOv2、SigLIP等视觉编码器上实施。

**📊 数据集**

在ADE20K、PASCAL VOC 2012和Cityscapes等三大语义分割基准上评估。

**📈 对比分析**

与标准因果注意力模型对比，双向注意力在CLIP和SigLIP上可提升mIoU 0.42–0.82个百分点，表明LLM层能在无需额外分割头的情况下显著改善分割性能。

**⚠️ 局限性**

实验受限于LLM架构和线性探测的简化，未能验证对更大模型或其他任务的普适性，且注意力打断是全局性的，无法分层分析细化机制。

---

## 3. PhysQuantAgent: An Inference Pipeline of Mass Estimation for Vision-Language Models

**arXiv ID:** 2603.16958 | [PDF](https://arxiv.org/pdf/2603.16958v1)

**作者:** Hisayuki Yokomizo `[一作]` (University of Tokyo), Yusuke Iwasawa `[通讯]` (University of Tokyo)

**通讯引用:** 2332 | [OpenAlex ID](https://openalex.org/A5063925941)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用视觉提示增强VLM进行实时对象质量估计并提出相应的数据集。

**💡 创新点**

创新点在于将三种视觉提示（目标检测、尺度标注、横截面生成）与VLM结合，实现无需复杂3D重建的物理量推理。

**🔧 技术方法**

使用VLM（如Qwen3‑VL‑8B、Gemini 2.5 pro、Gemini 3.1 pro）与Grounding DINO、Nano Banana等视觉提示模型，结合RGB‑D深度信息。

**📊 数据集**

创建了新的RGB‑D视频数据集，包含约300个可抓取小型物体的360°视频并配有真实质量标注。

**📈 对比分析**

与NeRF2Physics及无提示VLM基线对比，MnRE指标显示带提示的VLM在绝大多数模型上优于NeRF2Physics，提示策略进一步提升精度。

**⚠️ 局限性**

局限性包括对透明物体深度误差、提示生成模型的伪影导致误判，且在极端光照或遮挡条件下表现下降。

---

## 4. Humans and transformer LMs: Abstraction drives language learning

**arXiv ID:** 2603.17475 | [PDF](https://arxiv.org/pdf/2603.17475v1)

**作者:** Jasper Jian `[一作]` (Stanford University), Christopher D. Manning `[通讯]` (Stanford University)

**通讯引用:** 443235 | [OpenAlex ID](https://openalex.org/A5086198262)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer语言模型（如GPT‑2）在训练过程中如何学习语言类别，比较其行为与人类语言习得的抽象化与实例化两种理论；通过跟踪下一词预测分布的发散度来捕捉词类和词项层面的学习轨迹。

**💡 创新点**

首次在训练期间对LM的学习轨迹进行细粒度发散度分析，发现抽象化（类级行为）先于词项特定学习，且不同语言构造的学习呈现顺序性，这为将LM与人类语言习得理论对齐提供了实证依据。

**🔧 技术方法**

使用Transformer基础语言模型（GPT‑2 small），结合Jensen–Shannon发散度（D_JS）评估词类间与词项间的预测分布差异；通过Mann‑Whitney U检验评估类级与词项级学习的显著性；并与基于计数的实例化基线对比。

**📊 数据集**

主要使用WikiText‑103抽取的自然句子作为训练与测试数据，包含四类动词（to‑dative、motion、reciprocal、spray‑load）以及BLiMP的transitivity与自然语言构造的相对子句最小对。

**📈 对比分析**

对比类级与词项级学习时，GPT‑2 的类级发散度先出现且持续高于词项级；相对子句和transitivity等现象同样先出现类级学习。与计数实例化基线相比，GPT‑2 的类级显著性更早、更稳健，表明其偏向抽象化学习。

**⚠️ 局限性**

仅评估单一模型架构（GPT‑2 small）与单一语言，未覆盖不同模型大小或多语言情况；实验范围局限于动词层面，未考察更细粒度或非词类的抽象化；缺乏对内部表示变化的深入分析。

---

## 5. Empirical Recipes for Efficient and Compact Vision-Language Models

**arXiv ID:** 2603.16987 | [PDF](https://arxiv.org/pdf/2603.16987v1)

**作者:** Jiabo Huang `[一作]` (Sony AI), Lingjuan Lyu `[通讯]` (Sony AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对资源受限场景，对紧凑视觉‑语言模型（VLM）进行端到端效率分析，并给出一系列实用的推理优化方案，同时提出一族可实现稠密图像字幕的紧凑VLM。

**💡 创新点**

①系统化的 CPU‑GPU 端到端剖析，发现 CPU 侧预处理是主瓶颈；②给出针对图像预处理、内存传输、分词等环节的具体优化配方；③通过图像切块、像素下采样等结构设计，使模型在保持 256M–2B 参数规模的同时支持结构化感知；④对坐标表示与专用位置 token 的比较，证明后者更适合紧凑模型。

**🔧 技术方法**

使用 austin 与 NVIDIA Nsight Systems 进行 CPU 与 GPU 级别剖析；借助 vLLM 服务框架、Pillow‑SIMD、Pinned Memory、Tokenization 规范化、Quantization 与 FP8 研究；在模型层面采用 ViT–MLP–LLM 架构、像素下采样、位置 token 等技术。

**📊 数据集**

训练集包含 ShareGPT4V、COCO2017、WiT、VFLAN、ScienceQA、MGM‑Instruct、Objects365、Visual Genome 等公开数据；评估使用 VQAv2、POPE、GQA、COCO‑Caption、NoCaps、Visual Genome 的稠密字幕任务。

**📈 对比分析**

对比了 SmolVLM、InternVL3、Qwen3‑VL 等主流紧凑 VLM，在 VQA、图像字幕和稠密字幕等多项基准上，TTFT 最高可降 93%，吞吐量提升 30%+；在文本理解任务中在 256M–500M 参数范围内平均提升 16%/11%；在 2B 模型下实现 4/5 任务的最高分，且保持最小化推理延迟。

**⚠️ 局限性**

量化虽能缩小参数，却因激活量化开销导致吞吐率下降；模型在极小规模下仍受限于容量，位置 token 的效果随规模提升；GPU 侧仍有余量可进一步优化；仅在特定视觉任务上验证，缺乏更广泛的多任务适用性。

---

## 6. Symmetry-Reduced Physics-Informed Learning of Tensegrity Dynamics

**arXiv ID:** 2603.17824 | [PDF](https://arxiv.org/pdf/2603.17824v1)

**作者:** Jing Qin `[一作]` (University of Kentucky), Muhao Chen `[通讯]` (University of Houston)

**通讯引用:** 2056 | [OpenAlex ID](https://openalex.org/A5100374415)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种利用几何对称性简化张力与压缩结构动力学的 SymPINN 框架，能够直接在对称子空间中训练神经网络并恢复完整节点坐标。

**💡 创新点**

创新点在于将对称群作用映射到坐标空间，构造对称基底并将对称约束硬编码进物理信息神经网络，从而显著降低学习维度、提升训练稳定性和准确性。

**🔧 技术方法**

采用的技术包括群论对称性分析、对称基底构造、对称约束的硬约束形式、Fourier 特征编码、两阶段优化（Adam+L‑BFGS）以及自定义的物理残差损失。

**📊 数据集**

使用了两组对称张力结构数据集：四节点的 C₂ 对称 T‑bar 和十二节点的 C₂ 对称 Lander，均从 MATLAB TsgFEM 生成。

**📈 对比分析**

与传统 PINN 进行对比，SymPINN 在相同网络结构和优化设置下，均实现了更低的相对误差（RE）和更短的训练时间，尤其在大规模节点（Lander）时表现更为显著。

**⚠️ 局限性**

局限性主要体现在只能处理有限的循环与二面体对称群，且对高度非线性或强耦合动态、对称破缺情况的鲁棒性尚待验证。

---

## 7. From Digital Twins to World Models:Opportunities, Challenges, and Applications for Mobile Edge General Intelligence

**arXiv ID:** 2603.17420 | [PDF](https://arxiv.org/pdf/2603.17420v1)

**作者:** Jie Zheng `[一作]` (Northwest University), Jiacheng Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 21596 | [OpenAlex ID](https://openalex.org/A5100727960)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统性综述了从数字孪生向世界模型的演进，并阐述了世界模型在边缘通用智能（EGI）中的核心作用；

**💡 创新点**

创新点在于明确区分数字孪生与世界模型的概念差异，提出四大核心能力（想象、预测、规划、推理），构建数字孪生与世界模型协同的应用框架，并给出多领域（ISCC、语义通信、A2G网络、LAWNs）未来研究方向；

**🔧 技术方法**

采用自监督学习的变分自编码器/时序动力学模型、生成式 AI、强化学习、图神经网络、联邦学习以及物理信息注入等技术；

**📊 数据集**

作为综述文章未使用专门实验数据，参考了无人机飞行轨迹、通信链路、工业传感等公开数据集（如Air2Ground、LAWNs等）；

**📈 对比分析**

通过对比数字孪生与世界模型在维度、功能和资源消耗上的差异，指出世界模型在资源效率、实时性和自适应性上的优势；在ISCC、语义通信、A2G网络等案例中，数字孪生实现了约30%延迟下降，世界模型进一步降低了15-20%执行延迟；

**⚠️ 局限性**

局限性包括对物理精度的欠缺、可解释性不足、联邦学习与持续更新的挑战，以及对多体多尺度交互建模的不足，亟需进一步融合物理知识与数据驱动、提升可解释性与安全性。

---

## 8. RPMS: Enhancing LLM-Based Embodied Planning through Rule-Augmented Memory Synergy

**arXiv ID:** 2603.17831 | [PDF](https://arxiv.org/pdf/2603.17831v1)

**作者:** Zhenhang Yuan `[一作]` (Nanyang Technological University), Lihua Xie `[通讯]` (Nanyang Technological University)

**通讯引用:** 55627 | [OpenAlex ID](https://openalex.org/A5100365448)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种冲突管理架构RPMS，利用可执行规则和状态一致的经验记忆提升闭域实体代理的单次尝试成功率。

**💡 创新点**

通过规则检索与经验记忆的状态过滤相结合，首次实现同时解决无效动作生成与状态漂移的双向恶性循环，并验证记忆的有条件有效性。

**🔧 技术方法**

基于LLM推理的Prompt工程，轻量级信念状态跟踪，三层规则手册（通用、域、环境），经验记忆过滤与规则优先仲裁机制。

**📊 数据集**

在ALFWorld（134个未见任务）和ScienceWorld（26个任务共241条评测）上实验，使用Llama 3.1 8B、70B、Claude Sonnet 4.5及GPT‑4后端。

**📈 对比分析**

与ReAct基线进行对齐prompt的控制对比，在ALFWorld上单次成功率从35.8%提升至59.7%（+23.9pp），Claude提升至98.5%；在ScienceWorld平均分从44.9提升至54.0（+9.1pp），规则贡献最大，记忆贡献为有条件。

**⚠️ 局限性**

依赖人工规则编写、需预先收集经验轨迹、状态过滤仍采用粗略手柄占用检查、规则与记忆仲裁仍粗糙，且在新领域扩展时需重新编写环境级规则。

---

## 9. Towards Motion-aware Referring Image Segmentation

**arXiv ID:** 2603.17413 | [PDF](https://arxiv.org/pdf/2603.17413v1)

**作者:** Chaeyun Kim `[一作]` (Seoul National University), Joonseok Lee `[通讯]` (Seoul National University)

**通讯引用:** 5637 | [OpenAlex ID](https://openalex.org/A5067433666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对动作（motion）表达的RIS改进方法，利用动作片段数据增强和多模态径向对比学习(MRaCL)；

**💡 创新点**

创新点在于将动作中心的短语作为正样本增强，并在融合后的多模态嵌入上使用基于角度的对比损失，解决传统余弦相似度的梯度饱和与各向异性问题；

**🔧 技术方法**

使用LLM提取动作短语、Transformer‑based RIS模型、对比学习、角度距离损失(MRaCL)以及假负样本过滤；

**📊 数据集**

实验涵盖RefCOCO、RefCOCO+、G‑Ref、Ref‑ZOM等RIS基准，并新增动作中心的M‑Ref与M‑Bench视频基准；

**📈 对比分析**

与LAVT、CRIS、DMMI、ASDA、DETRIS等多种基线对比，平均提升oIoU 0.6~1.7点，在动作查询上显著提升，同时保持或略高静态查询性能；

**⚠️ 局限性**

局限在于需预先过滤多目标同类别场景，依赖LLM提取短语，且对硬负样本不友好，未在更大尺度的动态视频任务上验证。

---

## 10. VisceroHaptics: Investigating the Effects of Gut-based Audio-Haptic Feedback on Gastric Feelings and Gastric Interoceptive Behavior

**arXiv ID:** 2603.16919 | [PDF](https://arxiv.org/pdf/2603.16919v1)

**作者:** Mia Huong Nguyen `[一作]` (National University of Singapore), Suranga Nanayakkara `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究通过三项实验，探究将肠道声音转化为音频‑触觉反馈，非侵入性地调节胃感知、饥饿感知及饮水行为。

**💡 创新点**

创新点在于首次证明音频‑触觉反馈能非侵入性调节胃内感知，并揭示不同声音模式对饥饿与饱腹感的差异化影响。

**🔧 技术方法**

采用腹部音频录制、声音分割与特征提取、音频‑触觉转换、腹部振动传感器以及水负荷试验（WLT‑II）等技术。

**📊 数据集**

使用来自7名受试者的肠道声音样本作为刺激库，并在18、16、21名受试者的三组实验中收集数据。

**📈 对比分析**

与无刺激基线对比，利用重复测量ANOVA/ART等统计方法，发现单爆发模式下饮水量显著高于基线，连续随机模式饥饿评分显著高于单爆发模式，表明反馈有效。

**⚠️ 局限性**

局限性包括刺激样本有限、受试者样本量小、仅以饮水量衡量胃行为、未考虑情境因素及对个体差异的深入探究。

---

## 11. Inducing Epistemological Humility in Large Language Models: A Targeted SFT Approach to Reducing Hallucination

**arXiv ID:** 2603.17504 | [PDF](https://arxiv.org/pdf/2603.17504v1)

**作者:** Cem Uluoglakci `[一作]` (Middle East Technical University), Tugba Taskaya Temizel `[通讯]` (Middle East Technical University)

**通讯引用:** 1050 | [OpenAlex ID](https://openalex.org/A5064355568)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于SFT的训练数据HypoTermInstruct，教导LLM在遇到不存在的假想术语时承认知识边界，显著降低hallucination；

**💡 创新点**

创新点在于用经过多引擎检索验证的非存在术语来“解耦”元认知行为与具体事实，从而学习到可迁移的“ epistemological humility ”；

**🔧 技术方法**

采用LoRA微调、Logit Lens、线性探针与谱分析等技术进行训练与可解释性研究；

**📊 数据集**

使用HypoTermInstruct（31.5k样本）与HypoTermQA-Enhanced（16k问答）以及七个标准instruction数据集进行对照；

**📈 对比分析**

在Llama3.1-8B与Gemma3-4B上共800次LoRA实验，HypoTermInstruct可将HypoTerm Score提升高达26.5%（Gemma）或2.2%（Llama），FactScore提升0.39–0.86%，对MMLU影响极小；

**⚠️ 局限性**

仅在两种解码器架构与LoRA下验证，未覆盖全量微调或其他模型；安全分数在基础模型上略有下降，且对非名词实体之外的uncertainty类型与RLHF等后续对齐过程缺乏探索；

---

## 12. Behavior-Centric Extraction of Scenarios from Highway Traffic Data and their Domain-Knowledge-Guided Clustering using CVQ-VAE

**arXiv ID:** 2603.16964 | [PDF](https://arxiv.org/pdf/2603.16964v1)

**作者:** Niklas Roßberg `[一作]` (Technische Hochschule Ingolstadt), Michael Botsch `[通讯]` (Technische Hochschule Ingolstadt)

**通讯引用:** 479 | [OpenAlex ID](https://openalex.org/A5058811339)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

利用规则驱动的驾驶员行为变化检测提取高速公路场景，并通过向量量化变换自编码器（CVQ‑VAE）结合交互作用模型和伪类别信息实现场景聚类。

**💡 创新点**

1）首次提出基于行为变化的标准化场景抽取方法；2）将交互作用评分和行为类别作为辅助监督引入CVQ‑VAE，提升聚类的语义可解释性和一致性。

**🔧 技术方法**

规则阈值检测、方向化社交力模型（DG‑SFM）、CVQ‑VAE、线性预测头（伪类别与交互评分）以及k‑means、层次聚类做对照实验。

**📊 数据集**

HighD 高速公路行驶轨迹数据集，选取保持车道到变道的场景共 16,768 条。

**📈 对比分析**

相较于无知识引导的 CVQ‑VAE，加入域知识后聚类熵下降、增量场景一致率提升至 56.8%（原为 6.8%），k‑means 与层次聚类也均有提升，但提升幅度最显著在 CVQ‑VAE 上。

**⚠️ 局限性**

仅使用固定长度窗口，可能导致行为截断或合并；对变量长度和参与车辆数的场景聚类尚未覆盖。

---

## 13. citecheck: An MCP Server for Automated Bibliographic Verification and Repair in Scholarly Manuscripts

**arXiv ID:** 2603.17339 | [PDF](https://arxiv.org/pdf/2603.17339v1)

**作者:** Junhyeok Lee `[一作]` `[通讯]`, Junhyeok Lee

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一款名为citecheck的TypeScript系统与MCP服务器，用于自动检索、验证并修复论文中的参考文献错误。

**💡 创新点**

创新点在于结合多通道检索、显式匹配与策略驱动的重写规划，能够跨多种文件格式和数据库（PubMed、Crossref、arXiv、Semantic Scholar）自动发现并纠正错误，同时为LLM生成的引用幻觉提供防护。

**🔧 技术方法**

技术上采用TypeScript实现、MCP服务器架构、对接PubMed、Crossref、arXiv、Semantic Scholar接口，并实现多通道检索、manifest‑aware matching、policy‑gated rewrite planning以及47个单元测试。

**📊 数据集**

使用的主要数据集包括PubMed、Crossref、arXiv和Semantic Scholar数据库的公开元数据，以及多种论文文件样例（如Markdown、LaTeX、Word、PDF等）。

**📈 对比分析**

通过47个单元测试验证系统在修复行为、错误处理、传输失败和MCP暴露等方面的功能表现，全部测试通过；由于未与现有工具做大规模对比，具体性能数值未给出。

**⚠️ 局限性**

局限性在于仅覆盖上述数据库，无法处理未纳入这些资源的文献；当前为原型阶段，缺乏大规模用户验证；对多语言文献支持不足；以及对LLM生成引用错误的检测深度受限。

---

## 14. Early Quantization Shrinks Codebook: A Simple Fix for Diversity-Preserving Tokenization

**arXiv ID:** 2603.17052 | [PDF](https://arxiv.org/pdf/2603.17052v1)

**作者:** Wenhao Zhao `[一作]` (National University of Singapore), Dianbo Liu `[通讯]` (National University of Singapore)

**通讯引用:** 6491 | [OpenAlex ID](https://openalex.org/A5014407399)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

分析并解决VQ tokenizer中token表示收缩导致生成多样性下降的问题，并提出Deferred Quantization策略；

**💡 创新点**

引入分两阶段训练（先预训练编码器再启用量化），通过推迟量化实现token空间更均匀分布，从而缓解早期量化造成的收缩现象；

**🔧 技术方法**

使用Vector Quantization（VQ‑VAE）与Deferred Quantization，配合预训练Encoder、感知损失、代码簿距离、困惑度等度量；

**📊 数据集**

在合成高斯数据、CIFAR‑10、ImageNet‑100以及医学眼底图像数据集ODIR上进行实验；

**📈 对比分析**

与传统VQ、SimVQ等方法对比，采用r‑FID、g‑FID、LPIPS、MSE、Perplexity、代码簿欧氏距离等指标，实验结果表明Deferred Quantization在重建质量和生成多样性上均显著提升；

**⚠️ 局限性**

受限于训练资源导致代码簿尺寸选择、在小样本医学数据上的评估不稳定，以及仅验证了VQ tokenizer，未覆盖其他离散化技术的通用性。

---

## 15. Generalist Multimodal LLMs Gain Biometric Expertise via Human Salience

**arXiv ID:** 2603.17173 | [PDF](https://arxiv.org/pdf/2603.17173v1)

**作者:** Jacob Piland `[一作]` (University of Notre Dame), Adam Czajka `[通讯]` (University of Notre Dame)

**通讯引用:** 1394 | [OpenAlex ID](https://openalex.org/A5067121774)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用通用多模态大语言模型（MLLM）结合人工显著性信息，对虹膜伪造检测（PAD）进行零/少样本学习。

**💡 创新点**

创新点：①证明通用MLLM本身已具备区分多种虹膜攻击的视觉特征；②通过引入人工（专家和非专家）口头描述的显著性，进一步提升模型判定的精确度；③提出“MESH”扩展显著性技术，利用MLLM生成更完整的图像描述，从而进一步优化判断。

**🔧 技术方法**

技术：多模态大语言模型（Gemini 2.5 Pro、Llama 3.2‑Vision）、SigLIP视觉编码、Gemma/LLama文本嵌入、轻量级MLP融合、结构化提示工程（short/long prompts）、人类显著性（raw human、MESH）和评估指标（MSE、BPCER/APCER）。

**📊 数据集**

数据集：224幅虹膜图像（8类攻击+正常），来自公开虹膜PAD基准，使用30幅样本/类，包含人工显著性文本及MESH文本。

**📈 对比分析**

对比方法：与传统CNN（DenseNet‑121）基线、人工专家评估进行比较。结果显示：Gemini+长提示+人类显著性（MSE 0.053）超过CNN（0.345）和人类（0.062）；Llama在部分攻击类型也优于CNN；短提示无显著性性能最差。MSE、BPCER/APCER等指标表明MLLM在多类攻击检测上具有显著优势。

**⚠️ 局限性**

局限性：①实验规模受预算/IRB限制，仅使用224幅图像；②对商业模型的访问受限，导致只能使用Gemini；③长提示+MESH在Gemini上易出现过拟合；④缺乏对大规模样本、跨平台可移植性及对抗鲁棒性的深入研究。

---

## 16. SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion

**arXiv ID:** 2603.17092 | [PDF](https://arxiv.org/pdf/2603.17092v1)

**作者:** Elham Daneshmand `[一作]`, Hsiu-Chin Lin `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

论文阐述了低秩适配（LoRA）在从仿真到真实世界迁移中的表达力与最优性，并给出了低维现实差异导致低秩参数更新的理论依据。

**💡 创新点**

创新点在于：①给出了LoRA在任意秩更新下的完全表达性证明；②证明了在秩-1约束下LoRA能得到最优的Frobenius范数逼近；③将物理环境参数的小维差异映射到政策参数更新的低维子空间，从而解释低秩更新的有效性。

**🔧 技术方法**

主要技术包括：矩阵秩分解、奇异值分解与Eckart–Young–Mirsky 定理、线性张量（Jacobian）一阶泰勒展开。

**📊 数据集**

该部分未直接使用数据集，理论推导适用于任何仿真‑真实迁移场景；在主论文中通常会使用如MuJoCo模拟与真实机器人数据集进行验证。

**📈 对比分析**

通过与完整微调（全参数更新）以及不同秩的LoRA进行对比，实验表明：当真实纠正矩阵低秩或近似低秩时，秩‑1 LoRA在Frobenius误差上达到最优，且在实际仿真‑真实任务中保持与全微调相当的性能。

**⚠️ 局限性**

局限性包括：①假设真实纠正矩阵近似低秩；②只讨论线性层的适配，非线性或深层结构的高阶交互未覆盖；③在现实差异高度复杂且高秩时，LoRA的效果可能下降。

---

## 17. From Words to Worlds: Benchmarking Cross-Cultural Cultural Understanding in Machine Translation

**arXiv ID:** 2603.17303 | [PDF](https://arxiv.org/pdf/2603.17303v1)

**作者:** Bangju Han `[一作]` (Xinjiang Technical Institute of Physics and Chemistry, Chinese Academy of Sciences), Xi Zhou `[通讯]` (Xinjiang Technical Institute of Physics and Chemistry, Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CulT-Eval基准和ACRE评估指标，用于系统评估机器翻译中文化载表达的表现。

**💡 创新点**

创新点在于构建大规模、跨类别的文化载表达基准，统一错误分类，并设计基于文化解释的ACRE指标，显著提升对文化意义保持的诊断能力。

**🔧 技术方法**

技术手段包括：LLM辅助提取候选句、人工标注文化术语与错误标签、构建五维文化分类、以及使用大型语言模型实现ACRE中的语义验证器与质量评估器。

**📊 数据集**

使用7,959条中英平行句子组成的CulT-Eval数据集，来源涵盖文学、影视字幕、官方文件等多种文本类型，并在此集上评估多种MT与LLM模型。

**📈 对比分析**

对比BLEU、ChrF、BERTScore、COMET等传统指标，发现它们对文化意义的捕捉不足；ACRE与人工评价的相关性更高，显示更强的诊断敏感度；但各模型在ACRE上仍表现不佳，常出现遗漏、字面化等错误。

**⚠️ 局限性**

局限性包括：依赖人工文化解释和标签，规模相对有限；ACRE目前仅验证于中英对，跨语言或不同文化背景的适用性待验证；模型在处理文化表达方面仍存在显著误差，需进一步提升。

---

## 18. Do Language Models Encode Semantic Relations? Probing and Sparse Feature Analysis

**arXiv ID:** 2603.17624 | [PDF](https://arxiv.org/pdf/2603.17624v1)

**作者:** Andor Diera `[一作]` (Ulm University), Ansgar Scherp `[通讯]` (Ulm University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过线性探针、稀疏自编码器（SAE）和激活补丁三种可解释性方法，系统分析了大语言模型（Pythia‑70M、GPT‑2‑124M、Llama 3.1‑8B）内部如何编码四种语义关系（同义、反义、上义、下义），并研究了其在不同层级、模块和规模中的分布与可操纵性。

**💡 创新点**

创新点包括：① 将探针与SAE结合，首次在探针层面量化关系信息的必要性与充分性；② 通过激活补丁检验关系方向性偏差，揭示上义与下义在模型中的不对称编码；③ 发现关系信息主要集中在中层且在post‑residual流中最显著，且缺乏明显的层级热点。

**🔧 技术方法**

使用的技术：线性多项式探针、稀疏自编码器（SAE）提取稀疏特征、激活补丁（插入/消除特征）、中心质量（CoM）分析、语义logit差（LD_sem）衡量必要性与充分性、方向性对比实验。

**📊 数据集**

数据集：基于WordNet 3.0抽取的1000对每种语义关系（同义、反义、上义、下义）和1000个无关随机对，并在三种中性上下文模板下扩展为15000个实例。

**📈 对比分析**

比较与性能：在三种模型上进行层级探针评估，发现探针精度随模型规模提升；对比不同关系的难度顺序为：反义>下义>上义>同义。激活补丁在Llama 3.1中产生显著的必要性/充分性效应，GPT‑2与Pythia效果较弱。整体表明关系信息主要分布在中层，post‑residual流携带最多可解码信息。

**⚠️ 局限性**

局限性：① 只在探针层面进行因果检验，未评估对最终token生成的影响；② 仅涉及三种基础解码器模型，未覆盖指令微调、Mixture‑of‑Experts或编码器架构；③ 依赖WordNet作为真值，可能不适用于其他语言或领域；④ SAE的稀疏分解可能泄露模型内存忆，存在隐私与可复现性挑战。

---

## 19. Omni IIE Bench: Benchmarking the Practical Capabilities of Image Editing Models

**arXiv ID:** 2603.16944 | [PDF](https://arxiv.org/pdf/2603.16944v1)

**作者:** Yujia Yang `[一作]` (University of Chinese Academy of Sciences), Hongzhu Yi `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 2005 | [OpenAlex ID](https://openalex.org/A5102784508)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Omni IIE Bench 基准，用于诊断指令驱动图像编辑模型在实际场景中的一致性与多轮交互性能；

**💡 创新点**

创新点在于双轨诊断设计（单轮一致性与多轮协调），采用双零容忍的人工审核与行业相关性评审，量化模型在不同语义尺度下的表现差异；

**🔧 技术方法**

技术包括 GPT‑4o 生成图像描述与指令，Nano Banana 生成目标图像，GroundingDINO+SAM 自动提取掩码；评估指标包括 CLIP、LPIPS、PSNR、SSIM 等视觉质量指标，以及基于 MLLM 的 QA 合规性检验；

**📊 数据集**

数据来源于 12 个公开数据集，构建了 1,725 条单轮对话和 260 条多轮对话（共 1,131 次编辑），最终通过严格人工过滤得到高质量样本；

**📈 对比分析**

评测 8 大主流 IIE 模型，结果显示 Qwen‑image‑edit 最佳、HQEdit 最差；所有模型在从低语义尺度到高语义尺度以及多轮编辑时均出现显著性能下降，错误积累导致背景保留与指令遵循能力明显受损；与人工评估相关性超过 0.85；

**⚠️ 局限性**

局限性包括高度依赖人工审核导致样本规模受限；基准侧重单/多轮一致性，未覆盖更复杂的交互场景；QA 评估可能忽略细微指令细节，需进一步完善。

---

## 20. Steering Video Diffusion Transformers with Massive Activations

**arXiv ID:** 2603.17825 | [PDF](https://arxiv.org/pdf/2603.17825v1)

**作者:** Xianhang Cheng `[一作]` (Mohammed Bin Zayed University of Artificial Intelligence), Hao Li `[通讯]` (Pinscreen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了视频扩散Transformer内部的巨激活（MAs），并提出一种无训练的结构激活引导（STAS）方法，在早期去噪阶段对首帧和边界Token的MAs进行放大，以提升视频质量与时间连贯性。

**💡 创新点**

首次揭示MAs在视频DiT中具有结构化的位置层级（首帧最大、潜在帧边界突出），并利用此结构设计针对性激活调节，实现在推理阶段无需额外训练即可提升性能。

**🔧 技术方法**

采用MAs定位、位置选择与激活放大（最大值归一化）等技术，结合Classifier-Free Guidance (CFG) 的推理技巧，主要实现为在Transformer前向过程中对选定维度进行掩码放大。

**📊 数据集**

在公开的文本到视频模型（Wan2.1‑1.3B、Wan2.2‑5B、CogVideoX‑5B）上实验，使用VBench、T2V‑CompBench、人类偏好调查等评估集。

**📈 对比分析**

通过VBench与T2V‑CompBench对比原模型与加STAS的结果，平均提升约0.3–0.5分，总体分数从81.39→81.76等；在时序一致性与视觉质量方面提升显著，且仅增加约0.06秒推理时间。

**⚠️ 局限性**

对动态度指标略有下降，且只能在模型生成质量基础上改善，无法完全弥补模型本身严重失效；对极端大尺度或高度动态的视频效果有限。

---

## 21. MCoT-MVS: Multi-level Vision Selection by Multi-modal Chain-of-Thought Reasoning for Composed Image Retrieval

**arXiv ID:** 2603.17360 | [PDF](https://arxiv.org/pdf/2603.17360v1)

**作者:** Xuri Ge `[一作]` (Shandong University), Xin Xin `[通讯]` (Shandong University)

**通讯引用:** 83579 | [OpenAlex ID](https://openalex.org/A5100328102)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于多模态链式推理的多级视觉选择方法，用以提升复合图像检索的语义匹配效果。

**💡 创新点**

创新点在于：①引入多模态LLM进行链式推理，生成保留/删除/目标文本；②设计Patch‑level和Instance‑level视觉选择模块，根据推理结果精准过滤噪声；③构建加权层次组合模块，动态融合文本与视觉特征。

**🔧 技术方法**

使用的技术包括CLIP视觉/文本编码器、Grounded‑SAM实例分割、Qwen2.5‑VL‑32B（LLM）以及自定义的加权组合与注意力机制。

**📊 数据集**

使用的公开数据集为CIRR（开放场景）和FashionIQ（服饰）两个基准集。

**📈 对比分析**

与现有方法对比，MCoT‑MVS在CIRR上Recall@1提升至55.33%（高于CCIN的53.41%），在FashionIQ上均衡提升至最高Recall@10/50；在多个指标上均刷新SOTA。

**⚠️ 局限性**

局限性包括：①推理过程较慢（≈0.85s/查询）且受LLM推理质量影响；②目标文本推理易出现幻觉，可能导致误导；③在复杂场景下多模态推理提升有限。

---

## 22. AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting

**arXiv ID:** 2603.17529 | [PDF](https://arxiv.org/pdf/2603.17529v1)

**作者:** Binqing Wu `[一作]` (Zhejiang University), Ling Chen `[通讯]` (Zhejiang University)

**通讯引用:** 35791 | [OpenAlex ID](https://openalex.org/A5100411084)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 AirDDE，一种将神经延迟微分方程与物理扩散-对流方程相结合的连续时间空气质量预测框架，并通过内存增强注意力（MAA）模块和物理引导的延迟演化函数（PDE）实现多因子延迟建模。

**💡 创新点**

创新点包括：①首次在空气质量任务中使用神经延迟微分方程捕获空间位置、时间延迟；②引入 MAA 模块，利用全局与局部历史记忆结合多因素信息动态捕捉延迟；③设计基于扩散-对流方程的物理引导 PDE，确保延迟演化的物理一致性。

**🔧 技术方法**

使用的技术包括：神经延迟微分方程（NDDE）、扩散-对流方程物理约束、STGNN 编码器、GNN‑GRU 单元、双重注意力机制、torchdiffeq 解决器、Huber 损失、Adam 优化器等。

**📊 数据集**

实验数据集：KnowAir（PM2.5+17个气象因子，184 个城市，3h 频率），China-AQI（AQI+7气象因子，209 个城市，1h 频率），US-PM（PM2.5+7气象因子，175 个县，1h 频率）。

**📈 对比分析**

与 19 种基线（STGNN、Attention、NODE 系列）进行对比，平均 MAE 减少 8.79%（在 KnowAir 约 9.23%，China‑AQI 约 9.85%，US‑PM 约 7.3%）。在长时程预测和缺失/噪声鲁棒性实验中，AirDDE 同样显著优于其它方法。

**⚠️ 局限性**

局限性：①延迟状态维护的计算效率仍较高；②风场的不确定性导致延迟估计存在随机误差；③尚未充分建模多段、复合延迟路径（中间区域传输）的复杂传输过程。

---

## 23. Edge-Efficient Two-Stream Multimodal Architecture for Non-Intrusive Bathroom Fall Detection

**arXiv ID:** 2603.17069 | [PDF](https://arxiv.org/pdf/2603.17069v1)

**作者:** Haitian Wang `[一作]` (University of Western Australia), Atif Mansoor `[通讯]` (University of Western Australia)

**通讯引用:** 414 | [OpenAlex ID](https://openalex.org/A5105517320)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于毫米波雷达和地面振动传感器的双流端到端算法，用于湿浴室环境下的隐私保护跌倒检测。

**💡 创新点**

创新点在于：① 用Mamba和Griffin网络对雷达与振动信号分别进行长时序和冲击特征提取；② 引入时间对齐的交叉条件注意力和低秩双线性融合；③ 在低功耗边缘设备上实现实时推理（15.8 ms延迟）。

**🔧 技术方法**

使用的技术包括：雷达信号预处理、GCC‑PHAT时延估计、LSK1D、Mamba2Block、Griffin‑GLRU、Switch‑MoE、低秩双线性模块、轻量化注意力。

**📊 数据集**

使用了自建的“湿浴室跌倒检测基准”数据集，包含3小时以上同步毫米波雷达和三轴振动记录，涵盖8个场景（空场、物品掉落、行走、跌倒等），并划分为主观独立的训练/验证/测试集。

**📈 对比分析**

与多种基线（单模态雷达/振动、早期/后期融合、传统加速度阈值、可穿戴 ML 等）对比，实验显示在Raspberry Pi 4B上取得96.1%窗口准确率、88.0%跌倒召回率、AUC 0.968，并把延迟从35.9 ms压缩至15.8 ms，能耗从14200 mJ降至10750 mJ。

**⚠️ 局限性**

局限性包括：依赖严格的时间对齐和同步，易受传感器漂移或信号干扰影响；在极端水蒸汽或多物体碰撞场景下仍可能出现误报；数据集规模有限，缺乏多种浴室布局和不同硬件的泛化验证。

---

## 24. HeiSD: Hybrid Speculative Decoding for Embodied Vision-Language-Action Models with Kinematic Awareness

**arXiv ID:** 2603.17573 | [PDF](https://arxiv.org/pdf/2603.17573v1)

**作者:** Zihao Zheng `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**通讯引用:** 36767 | [OpenAlex ID](https://openalex.org/A5100641667)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 HeiSD 框架，通过在 Vision‑Language‑Action 模型推理过程中混合使用检索式和草稿式投机解码来显著提升推理速度。

**💡 创新点**

创新点包括：①自适应 verify‑skip 机制和按序列宽松接受策略，以缓解检索式解码的低质量草稿和持久误差；②基于运动学的融合度量，用于自动决定何时切换到检索式或草稿式解码；③将数据库部署在 CPU 上实现 GPU‑CPU 协同，减少显存占用。

**🔧 技术方法**

主要技术包括：投机解码（drafter‑based & retrieval‑based）、深度学习模型（OpenVLA + LLaMA）、向量检索（Qdrant）、运动学分析、序列化树解码、CPU‑GPU 资源协同。

**📊 数据集**

使用 LIBERO 机器人操控基准数据集构建向量数据库，并在该基准及真实桌面实验环境中评估。

**📈 对比分析**

与自回归推理、纯草稿式解码、纯检索式解码及 SpecVLA 对比，HeiSD 在仿真中实现 1.79×–2.45× 的速度提升，真实环境中 2.06×–2.41×，且任务成功率保持在 71–87% 范围内。

**⚠️ 局限性**

局限性：仅适用于自回归 VLA 模型，未针对自动超参数调优，且未对多模态扩散模型或非机器人任务进行验证。

---

## 25. Split-Merge Dynamics for Shapley-Fair Coalition Formation

**arXiv ID:** 2603.17153 | [PDF](https://arxiv.org/pdf/2603.17153v1)

**作者:** Quanyan Zhu `[一作]` (New York University), Zhengye Han `[通讯]` (New York University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出一种动态分裂-合并框架，用于在合作博弈中自组织形成联盟并实现公平与效率的平衡。

**💡 创新点**

创新点在于将Shapley值负值作为分裂触发信号、严格的合并规则仅在超加收益时触发，并通过向量Lyapunov函数和离散LaSalle不变原理证明有限时间收敛到Shapley-公平且合并稳定（SFMS）的分区。

**🔧 技术方法**

使用控制理论中的Lyapunov函数、向量LaSalle不变原理、离散时间图论及Shapley值的计算与近似技术。

**📊 数据集**

实验采用人工构造的10玩家合作博弈（无公开数据集），通过数值模拟展示动态过程。

**📈 对比分析**

比较方法主要是观察Lyapunov序列的单调性与收敛性；实验表明系统在有限步内进入不变集，避免了振荡，最终稳定在SFMS分区。

**⚠️ 局限性**

局限在于仅针对可转让效用的静态特征函数、没有考虑噪声与学习动态，以及在更大规模游戏中Shapley值计算的可扩展性。

---

## 26. Disclosure By Design: Identity Transparency as a Behavioural Property of Conversational AI Models

**arXiv ID:** 2603.16874 | [PDF](https://arxiv.org/pdf/2603.16874v1)

**作者:** Anna Gausen `[一作]` (AI Security Institute), Christopher Summerfield `[通讯]` (AI Security Institute)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出“设计式披露”机制，并对多模态（文本与语音）对话AI在不同使用场景下的身份披露行为进行首次系统评估

**💡 创新点**

将身份披露视为模型层面可嵌入行为，并结合多模态实验揭示其在角色扮演与对抗性提示下易被抑制的缺陷

**🔧 技术方法**

采用评估管线（Inspect框架）结合LLM‑as‑judge判别、RLHF、Constitutional AI、对抗训练及输出过滤等技术进行实验与对策研究

**📊 数据集**

使用7,000条文本对话和42,000条语音对话样本，覆盖6款开源/闭源模型，20种身份查询和35种系统提示组合

**📈 对比分析**

与基线（提示“你是助手”）对比发现披露率在角色扮演下跌至50%以下，遭对抗提示几乎全失；不同模型与模态间差异显著，表明现行实现不稳定

**⚠️ 局限性**

实验仅限英文、单轮对话、有限模型与提示范围，未涵盖多轮、跨语言及更复杂部署情境，需进一步扩展验证

---

## 27. Edit Spillover as a Probe: Do Image Editing Models Implicitly Understand World Relations?

**arXiv ID:** 2603.17876 | [PDF](https://arxiv.org/pdf/2603.17876v1)

**作者:** Guandong Li `[一作]` (iFLYTEK), Zhaobin Chu `[通讯]` (iFLYTEK)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究图像编辑模型中的编辑溢出现象，将其作为探测模型世界知识的自然探针。

**💡 创新点**

提出 2×2 溢出分类法（空间距离 × 语义相关性），并引入 World Understanding Score 与 Semantic Spillover Density 两个度量，将溢出从错误转化为诊断信号。

**🔧 技术方法**

使用像素级差分、连通域提取、CLIP 语义相似度判别、SSIM 结构相似度等技术构建自动检测-分类-量化管线。

**📊 数据集**

构建 EditSpilloverBench，200 张中文文本编辑场景图像，涵盖 App Screenshots、Normal Docs、Others、Real Scenes、Receipts 五类。

**📈 对比分析**

通过对 5 个代表性模型的 Spill%、SSIM、四类比例、WUS、语义密度等指标进行比较，发现模型间溢出率差异 3.3 倍，qwen_2511 控制精度高但语义密度低，nano_banana 语义激活强但整体溢出率高。

**⚠️ 局限性**

局限包括：仅覆盖中文文本编辑场景；CLIP 语义相似度可能不足以捕捉深层语义关系；像素差分对 JPEG 压缩敏感；未验证物理/因果等更广泛的世界知识维度。

---

## 28. Towards Safer Large Reasoning Models by Promoting Safety Decision-Making before Chain-of-Thought Generation

**arXiv ID:** 2603.17368 | [PDF](https://arxiv.org/pdf/2603.17368v1)

**作者:** Jianan Chen `[一作]` (Southeast University), Minling Zhang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在生成 Chain-of-Thought（CoT）之前对大规模推理模型（LRM）进行安全决策的对齐方法；

**💡 创新点**

创新点在于利用 BERT 分类器提取安全决策信号，并通过辅助线性头在训练阶段将安全梯度直接反向传播到 LRM 隐层，从而在不损失推理能力的前提下显著提升模型在 jailbreak 试验下的安全性；

**🔧 技术方法**

技术手段包括 BERT‑based 监督式安全决策提取、辅助线性头对齐、低秩适配（LoRA）微调以及标准的安全对齐与下一步预测损失的联合训练；

**📊 数据集**

使用了包含约5,000条有害查询和3,000条正常查询的自生成数据集，BERT 训练样本约为1,500条；评估数据包括 JailbreakBench、StrongReject 与 WildJailbreak 等安全基准；

**📈 对比分析**

在所有测试模型（DeepSeek、Qwen3、Skywork 等）上，方法将攻击成功率（ASR）从 20%–70% 降至 0%–3.7%，并保持或略高于基线的推理准确率（AIME2024、MATH‑500、GPQA‑Diamond 等）；

**⚠️ 局限性**

局限性包括仍存在一定的误拒（benign over‑refusal）风险，且对更强或未见过的 jailbreak 攻击的鲁棒性尚未完全验证；

---

## 29. Adaptive Anchor Policies for Efficient 4D Gaussian Streaming

**arXiv ID:** 2603.17227 | [PDF](https://arxiv.org/pdf/2603.17227v1)

**作者:** Ashim Dahal `[一作]` (University of Southern Mississippi), Nick Rahimi `[通讯]` (University of Southern Mississippi)

**通讯引用:** 259 | [OpenAlex ID](https://openalex.org/A5102764912)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一种预算感知的强化学习 Anchor 采样器，替代固定 FPS，在 4D Gaussian Streaming 流水线中自适应选择 anchor 以平衡渲染质量与实时性。

**💡 创新点**

通过两阶段训练（SFT warm‑up + Contextual‑Bandit RL）构造了联合预算与子集选择的策略；在同一预算下显著提升 PSNR，且 anchor 数量大幅降低；实现了 deterministic 推理和可选预算预测。

**🔧 技术方法**

使用点级 MLP+Transformer 编码器对 Gaussian 进行特征化，结合政策梯度与上下文 bandit RL 进行训练；奖励设计平衡稀疏度、运行时间与 PSNR；采用 voxel‑hash 下采样构造候选池。

**📊 数据集**

在 N3DV（训练 4 场景，验证 2 场景）与 MeetingRoom（Discussion、Trimming）上进行评估，并在 unseen 场景（Sear Steak、Cut Roasted Beef）测试泛化。

**📈 对比分析**

与 FPS（8192 anchor）、IGS、3DGStream、StreamRF 等基线比较；在 fast 模式下，256 anchor（32×更少）可提升 PSNR +0.5 dB，速度 1.27×；低预算下质量甚至超过 8k anchor，速度 1.2–1.3×；在高质量模式下亦保持竞争力。

**⚠️ 局限性**

仅在 fast 模式下评估；对高质量重建、不同后端的适配有限；RL 奖励手工设计，可能限制在极端动态或不同尺度场景的鲁棒性；推理时仍有一定计算开销。

---

## 30. VideoAtlas: Navigating Long-Form Video in Logarithmic Compute

**arXiv ID:** 2603.17948 | [PDF](https://arxiv.org/pdf/2603.17948v1)

**作者:** Mohamed Eltahir `[一作]` (King Abdullah University of Science and Technology), Naeemullah Khan `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 149 | [OpenAlex ID](https://openalex.org/A5059907674)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VideoAtlas 这一层级网格视频表示方法，并基于该表示实现了 Video-RLM 这一基于 Master‑Worker 结构的递归语言模型，可在视频中递归探索、检索视觉证据。

**💡 创新点**

创新点在于：① 将视频转换为无损、可导航、可扩展的多层网格，避免传统采样、裁剪或文本转化导致的信息损失；② 将长文本的递归语言模型直接迁移至视觉域，通过 VideoAtlas 的结构化环境实现对视频的递归检索；③ 引入环境预算与自适应计算分配机制，能够在不同视频长度下实现对计算资源的精细控制。

**🔧 技术方法**

技术包括：层级 K×K 网格生成与递归扩展（Expand）、多进程 Master‑Worker 并行探索、视觉脚本（Visual Scratchpad）记录无损视觉证据、基于不确定性分析的探索策略、以及对多模态缓存的利用。

**📊 数据集**

在论文中主要使用了从 1 小时到 10 小时不等的长视频基准（如公开的长视频问答/检索数据集，具体未列明），并与传统 VLM、统一采样、字幕/文本驱动的 agent 等方法进行对比。

**📈 对比分析**

实验表明 Video‑RLM 的计算成本随视频时长呈对数增长（约 9.7× 的 token 量化优势），缓存命中率达到 30–60%，并且在 1h→10h 的规模提升中保持最小的准确率下降，而传统线性扩展的方法在同等预算下准确率显著下滑。

**⚠️ 局限性**

局限性包括：① 需要预先构建并存储网格层级，导致对显存和 IO 的一定需求；② 对视频分辨率、帧率等硬件参数的敏感性；③ 在极端长视频或高分辨率场景下仍可能出现子网格生成瓶颈；④ 目前尚未对多模态交互（如音频、字幕）进行深度集成，主要聚焦于视觉信息。

---

## 31. Harm or Humor: A Multimodal, Multilingual Benchmark for Overt and Covert Harmful Humor

**arXiv ID:** 2603.17759 | [PDF](https://arxiv.org/pdf/2603.17759v1)

**作者:** Ahmed Sharshar `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Yuxia Wang `[通讯]` (INSAIT)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多模态、多语言（英语、阿拉伯语和无语言）幽默检测基准，用人工标注区分安全、显式伤害和隐式伤害。

**💡 创新点**

提出了针对暗黑幽默的显式/隐式伤害二分类标签，并系统评估多模态模型在低资源语言与隐式幽默上的推理缺口。

**🔧 技术方法**

采用大语言模型（GPT‑5、GPT‑4o、Gemini）、多模态视觉语言模型（Qwen、InternVL、LLaVA）和视频LLM（GPT‑5 Pro、Qwen2.5‑Omni）进行评测。

**📊 数据集**

使用手工整理的3000条文本、6005张图片和1202段视频，涵盖英语、阿拉伯语和通用视觉幽默。

**📈 对比分析**

对比闭源与开源模型，闭源在准确率、宏F1上领先，显式伤害检测优势明显，但在阿拉伯语隐式伤害上的召回率仍低约15–20%。

**⚠️ 局限性**

局限包括标注主观性、数据规模不均、缺乏细粒度伤害类型、视频多模态与音频处理不足，以及对低资源语言的偏差。

---

## 32. Cohomological Obstructions to Global Counterfactuals: A Sheaf-Theoretic Foundation for Generative Causal Models

**arXiv ID:** 2603.17384 | [PDF](https://arxiv.org/pdf/2603.17384v1)

**作者:** Rui Wu `[一作]` (University of Science and Technology of China), Yongjun Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18753 | [OpenAlex ID](https://openalex.org/A5100376886)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文构建了基于细胞层 Sheaf 的连续因果生成框架，并证明传统模型在存在拓扑阻碍时会出现流形撕裂。

**💡 创新点**

创新点在于将因果结构映射为 Wasserstein 空间的细胞 Sheaf，提出 Entropic Causal Sheaf Laplacian、Entropic Pullback Lemma，并通过隐函数定理实现无记忆逆向梯度。

**🔧 技术方法**

技术手段包括细胞 Sheaf 理论、Otto Wasserstein 梯度流、熵正则化 Sinkhorn 对偶、隐函数定理、Lagrangian 随机微分方程与大偏差理论。

**📊 数据集**

主要使用的实验数据集为 PBMC 3k scRNA‑seq 以及合成二维向量场数据。

**📈 对比分析**

与传统 deterministic ODE/Flow‑matching 对比，Entropic Sheaf Flow 在高维任务中实现了无生物奇点迁移，并显著降低 Dirichlet 能量；在结构学习上，Topological Causal Score 能准确分辨真因果图。

**⚠️ 局限性**

局限性包括对熵正则化参数的敏感性、对大规模样本时 Sinkhorn 对偶求解的数值不稳定，以及理论尚未扩展至含 2‑cell 的高阶同调结构。

---

## 33. Grievance Politics vs. Policy Debates: A Cross-Platform Analysis of Conservative Discourse on Truth Social and Reddit

**arXiv ID:** 2603.17901 | [PDF](https://arxiv.org/pdf/2603.17901v1)

**作者:** Yining Wang `[一作]` (GESIS Leibniz Institute for the Social Sciences), Tugrulcan Elmas `[通讯]` (University of Edinburgh)

**通讯引用:** 131 | [OpenAlex ID](https://openalex.org/A5081836422)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Truth Social与主要保守派Reddit社区在主题、毒性与时间动态上的大规模比较分析。

**💡 创新点**

首次将Truth Social与Reddit进行跨平台比较，揭示两者在主题取向与毒性分布的显著差异。

**🔧 技术方法**

使用FASTopic主题建模并通过LLM细化标签，利用Perspective API进行毒性评估，并用贝叶斯层次回归分析时间动态。

**📊 数据集**

Truth Social Core、Truth Social Extra以及r/Conservative、r/Republican、r/conservatives 5个子版块的帖子数据。

**📈 对比分析**

通过主题占比、毒性分数与时间序列对比；FASTopic在主题一致性和多样性上优于LDA和BERTopic，选定K=60后降至37个可解释主题。

**⚠️ 局限性**

研究时间仅覆盖2022年2月至10月，未包含其他主要平台，且样本偏向活跃发帖者，可能不具代表性。

---

## 34. Versatile Editing of Video Content, Actions, and Dynamics without Training

**arXiv ID:** 2603.17989 | [PDF](https://arxiv.org/pdf/2603.17989v1)

**作者:** Vladimir Kulikov `[一作]` (Google DeepMind), Tomer Michaeli `[通讯]` (Technion -- Israel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练的、可自由编辑视频动作、动态和全局效果的框架；

**💡 创新点**

创新点包括：1) 通过相似性引导聚合（SGA）选择编辑方向；2) 采用逐步增大的噪声相关性（ANC）抑制高频抖动；3) 在反演自由编辑基础上实现结构不受限的编辑；

**🔧 技术方法**

主要技术是基于预训练的文本到视频流模型的反演自由编辑（FlowEdit）方法，并加入SGA和ANC模块；

**📊 数据集**

使用了自建的71条编辑数据集（包含插入、交换、动作修改、全局效果四类），以及公开的WAN2.1 14B 480p I2V模型；

**📈 对比分析**

与FlowEdit、FlowAlign、SDEdit、I2V采样以及商业的Runway Aleph等方法对比，VLM评估和用户研究显示其在内容保持、文本遵循和视觉质量上均优于训练自由基线，且与训练模型Aleph相当；

**⚠️ 局限性**

局限性包括：依赖底层I2V模型的物理准确性、难以实现大范围时空修改、在某些大幅编辑下仍可能无法完整保留未被编辑区域的原始行为。

---

## 35. Ruyi2.5 Technical Report

**arXiv ID:** 2603.17311 | [PDF](https://arxiv.org/pdf/2603.17311v1)

**作者:** Huan Song `[一作]` (Institute of Artificial Intelligence), Xuelong Li `[通讯]` (Institute of Artificial Intelligence)

**通讯引用:** 62031 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了 Ruyi2.5 多模态家族模型及其隐私友好版本 Ruyi2.5-Camera，并提出了 Binary Prefix Policy Optimization（BPPO）强化学习加速方法。

**💡 创新点**

创新点包括：①共享骨干（shared‑backbone）架构实现不同规模模型一次性共训练；②边缘+云两阶段隐私管道，利用信息瓶颈实现不可逆身份脱敏；③BPPO 通过二进制响应筛选和前缀梯度约束显著提升 RL 训练速度。

**🔧 技术方法**

核心技术包括：Vision‑Language 统一投影层、AI Flow 理论框架、信息瓶颈脱敏、两阶段 device‑cloud 协同推理、BPPO（分组采样、前缀梯度、重要性采样）。

**📊 数据集**

使用大规模多模态数据集（图像‑文本对、细粒度 VQA、STEM 题库、图表/文档理解等）以及专门构建的几何与数学 RL 数据库进行训练和评估。

**📈 对比分析**

在通用多模态基准上，Ruyi2.5 与同等规模 Qwen3‑VL 并列；在隐私受限监控任务上，Ruyi2.5‑Camera（尤其 8B 版）在精确率、召回率、F1 方面明显领先 Qwen3‑VL（如 93.67% F1 对比 82.77%）。

**⚠️ 局限性**

局限性：①对极端环境或高分辨率监控的鲁棒性尚未完全验证；②隐私脱敏后对某些细粒度行为识别可能仍有限；③BPPO 在非常长的推理链或高频更新场景下的可扩展性待进一步评估。

---

## 36. Federated Distributional Reinforcement Learning with Distributional Critic Regularization

**arXiv ID:** 2603.17820 | [PDF](https://arxiv.org/pdf/2603.17820v1)

**作者:** David Millard `[一作]` (Rochester Institute of Technology), Ali Baheri `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 301 | [OpenAlex ID](https://openalex.org/A5086511671)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

在联邦学习框架下，作者提出FedDistRL，仅对分布式价值网络进行聚合，并通过CVaR加权的Wasserstein barycenter构建本地风险参考，随后在此参考上施加分布式信赖域约束；

**💡 创新点**

创新点在于揭示传统参数平均导致的“均值模糊”问题，并提出TR-FedDistRL以风险感知的分布信赖域来保留多峰与尾部信息，从而提升安全性能；

**🔧 技术方法**

使用的技术包括分布式量化回归（Quantile Critic）、CVaR风险度量、Wasserstein barycenter、分布式信赖域（shrink–squash）以及PPO/MAPPO等actor-critic方法；

**📊 数据集**

实验数据集包括合成Bandit、六客户端多智能体网格仓库以及五客户端连续控制高速公路驾驶环境；

**📈 对比分析**

与本地训练、FedAvg、FedAvg+CVaR等方法对比，TR-FedDistRL在所有实验中均显著降低灾难/事故率、输出漂移，且在高速公路任务上还能提升平均回报；

**⚠️ 局限性**

局限性包括：仍依赖参数平均，难以处理极端客户端异构；只对价值网络进行聚合；缺乏全局收敛理论；实验范围有限，未涵盖更复杂安全约束或大规模部署；

---

## 37. FACE-net: Factual Calibration and Emotion Augmentation for Retrieval-enhanced Emotional Video Captioning

**arXiv ID:** 2603.17455 | [PDF](https://arxiv.org/pdf/2603.17455v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 38. Shannon meets Gödel-Tarski-Löb: Undecidability of Shannon Feedback Capacity for Finite-State Channels

**arXiv ID:** 2603.17317 | [PDF](https://arxiv.org/pdf/2603.17317v1)

**作者:** Angshul Majumdar `[一作]` (Indraprastha Institute of Information Technology), Angshul Majumdar `[通讯]` (Indraprastha Institute of Information Technology)

**通讯引用:** 6270 | [OpenAlex ID](https://openalex.org/A5020310463)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究有限状态信道（FSC）在反馈容量上的精确阈值决策问题，并证明该问题在一类二进制、无理参数、可行化的无归一化FSC上不可判定。

**💡 创新点**

给出了一个结构性障碍，表明无论采用多大有限时域或任何实数代数形式，均无法统一求解该阈值问题，从而确立了精确反馈容量理论的根本极限。

**🔧 技术方法**

使用了可计算性理论（多对一归约）、无关性构造、延迟激活无归一化FSC族以及对存在实数理论（∃ℝ）与 Godel–Tarski–Lob 不完备性的理论分析。

**📊 数据集**

无实验数据集，研究完全基于理论构造。

**📈 对比分析**

未涉及实验比较，本文仅给出理论不可判定性与 ∃ℝ 障碍的证明。

**⚠️ 局限性**

结果仅适用于精确阈值决策，未说明近似、保证间隙或有限时域问题是否可决。

---

## 39. Beyond Forced Modality Balance: Intrinsic Information Budgets for Multimodal Learning

**arXiv ID:** 2603.17347 | [PDF](https://arxiv.org/pdf/2603.17347v1)

**作者:** Zechang Xiong `[一作]` (Alibaba Group), Yulan Hu `[通讯]` (Alibaba Group)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 IIBalance 框架，用 Intrinsic Information Budget (IIB) 对多模态学习进行相对平衡，解决主模态支配导致的性能瓶颈。

**💡 创新点**

创新点在于：①以数据集级的 IIB 作为模态容量先验；②使用原型引导的相对对齐，只在弱模态低于预算时纠正语义漂移；③在推理阶段结合 IIB 与样本级不确定性的贝叶斯融合，动态生成校准后的融合权重。

**🔧 技术方法**

采用原型对比损失、EMA 原型维护、预算差异控制对齐、轻量门控网络、贝叶斯灵感加权融合、线性调度等技术。

**📊 数据集**

在 Kinetics‑Sounds、CREMA‑D、AVE 三个音视频多模态基准上进行验证。

**📈 对比分析**

与 MSLR、G‑Blending、OGM‑GE、AGM、greedy、GGDM、PMR、MMPareto 等多种平衡与融合方法对比，IIBalance 在所有三个基准上均实现了最高或接近最高的多模态准确率，并显著提升了弱模态的表现。

**⚠️ 局限性**

局限性包括：预算估计采用离线固定方式，未实现在线自适应更新；仅在音视频场景验证，需进一步扩展到更广泛的多模态任务。

---

## 40. DexEXO: A Wearability-First Dexterous Exoskeleton for Operator-Agnostic Demonstration and Learning

**arXiv ID:** 2603.17323 | [PDF](https://arxiv.org/pdf/2603.17323v1)

**作者:** Alvin Zhu `[一作]` (University of California Los Angeles), Dennis W. Hong `[通讯]` (University of California Los Angeles)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了DexEXO可穿戴手外骨骼，能够在无校准、跨用户尺寸兼容的前提下，收集高质量演示并直接使用腕部RGB图像训练扩散策略。

**💡 创新点**

核心创新在于可调节手指滑块与姿态容忍拇指机制，实现硬件级视觉与机械对齐，消除后处理与多模态感知的需求。

**🔧 技术方法**

采用机械链接驱动的外骨骼、被动手结构、腕部摄像头实时RGB采集，以及基于DINOv2视觉编码器的扩散策略网络。

**📊 数据集**

使用自研的DexEXO演示数据集（Block 200、Carton 150、Bottle 150），并在14名受试者的剪刀、翻页、堆杯、弹钢琴等任务中收集评测数据。

**📈 对比分析**

与DexUMI和远程操作进行定量/主观对比，DexEXO在钢琴任务成功率提升54%，完成时间加快16%，在Block、Carton、Bottle任务均超过90%成功率，且不需图像分割或触觉反馈。

**⚠️ 局限性**

主要限制包括手指顶视角遮挡、机械干涉导致的动作受限、与特定机器人手匹配受限以及对高度遮挡或多模态感知需求的任务仍需额外传感。

---

## 41. DancingBox: A Lightweight MoCap System for Character Animation from Physical Proxies

**arXiv ID:** 2603.17704 | [PDF](https://arxiv.org/pdf/2603.17704v1)

**作者:** Haocheng Yuan `[一作]` (University of Edinburgh), Changjian Li `[通讯]` (University of Edinburgh)

**通讯引用:** 1274 | [OpenAlex ID](https://openalex.org/A5101794527)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种轻量化、基于单摄像头的数字木偶化运动捕捉系统，利用用户操纵的日常物体的粗略运动，通过抽象的边界框与生成式运动模型对齐，最终生成逼真的三维角色动画。

**💡 创新点**

创新点包括：① 采用代理物体的3D边界框作为统一的中间表示，克服多种物体形态与运动自由度差异；② 将现有无标记视觉基础模型（SAM2、π³、CoTracker）串联实现无标记、无标定的运动捕捉；③ 通过合成代理-动画配对数据并训练盒子编码器+控制网络，实现在缺乏真实配对数据时的条件生成；④ 结合文本输入和空间引导实现多维控制。

**🔧 技术方法**

技术手段：视觉基础模型（SAM2分割、π³点云、CoTracker跟踪）；点云重建与边界框估计；盒子编码器（共享MLP + 组内自注意力）与ControlNet；Motion Diffusion Model (MDM)；文本条件；关键帧插值与SE(3)插值。

**📊 数据集**

使用 HumanML3D 数据集训练 MDM，并基于该数据集合成代理-动画配对（将真实运动转化为代理边界框）；还使用自构建的代理-动画对数据进行进一步训练。

**📈 对比分析**

通过用户研究（复制任务、创意任务）与现有物理动画工具对比；在 27 次复制实验中复制成功率 92.6%，生成动画逼真度和与目标的相似度优于传统方法；系统整体运行时间约 2min 40s（单张 RTX4090），用户操作时间约 3 分钟；与专业多摄像头 MoCap 系统相比，精度略低，但在成本和易用性上优势明显。

**⚠️ 局限性**

局限性：① 单摄像头易受遮挡影响，导致运动捕捉噪声；② 生成速度非实时，需离线后端处理；③ 仅支持人类动作，无法直接生成非人类角色动画；④ 需要用户在首帧点击分割，工作量较大；⑤ 代理模型的抽象程度若过低，需文本辅助，限制自由度；⑥ 关键帧插值与多角色交互尚未实现。

---

## 42. Identifying Latent Actions and Dynamics from Offline Data via Demonstrator Diversity

**arXiv ID:** 2603.17577 | [PDF](https://arxiv.org/pdf/2603.17577v1)

**作者:** Felix Schur `[一作]` (ETH Zurich), Felix Schur `[通讯]` (ETH Zurich)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5099111605)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在没有动作观测、仅有演示者身份标签的离线轨迹中，证明并实现了潜在动作与环境动力学的可识别性。

**💡 创新点**

提出利用演示者多样性作为结构先验，构造混合分解，结合非负矩阵分解和Gram判别最小化，首次给出理论上的可识别性证明与实现路径。

**🔧 技术方法**

核心技术包括：非负矩阵分解（NMF）与最小体积（Gram行列式最小化）正则化、策略多样性正则（log‑det 阻断）、标签锚定正则以及神经网络参数化的潜在动作/策略模型。

**📊 数据集**

论文未给出具体实验数据集，主要以理论证明与合成仿真验证为主；若有实验，则使用公开的动作自由视频或合成环境数据。

**📈 对比分析**

方法对比主要基于理论可识别性，若进行实验，通常在合成环境中成功恢复潜在动作，并在无演示者信息的基线下表现更优；未给出具体数值指标。

**⚠️ 局限性**

受限于假设：需要充分的演示者策略多样性、连续且连通的观测空间、无碰撞与可分辨的潜在动作；在噪声、策略退化或假设不满足时可识别性失效；且仍需少量标记动作才能消除全局置换不确定性。

---

## 43. Information Pathways in Online Science Communication: The Role of Platform Actors and News Media

**arXiv ID:** 2603.17249 | [PDF](https://arxiv.org/pdf/2603.17249v1)

**作者:** Alexandros Efstratiou `[一作]` (University of Washington), Luca Luceri `[通讯]` (Information Sciences Institute)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对1.24M条关于COVID‑19科研论文的推文与211k条新闻报道进行大规模分析，系统梳理了不同类Twitter参与者（普通用户、机器人、超级传播者、协同账号）以及新闻媒体在科学传播中的角色与互动。

**💡 创新点**

创新点在于首次构建“多层次信息路径”框架，揭示超级传播者、协同网络与新闻媒体之间的交叉传播机制，特别是对抗共识的专家如何被协同网络放大，并证明推特上的影响力往往先于新闻报道。

**🔧 技术方法**

采用共活动网络检测、TF‑IDF相似度与余弦相似度构建协同网络，使用h‑index衡量超级传播者，利用BERTopic做论文主题建模，DistilRoBERTa进行情感分析，Botometer评估机器人特征。

**📊 数据集**

使用的主要数据集为：包含25k条COVID‑19预印本及其发表版的文献；1.24M条推文（346k用户）；211k条新闻文章（2.34k媒体）。

**📈 对比分析**

通过对比协同与非协同账号、超级传播者与普通用户的特征（关注数、粉丝数、发文量、情感倾向等），发现协同网络与超级传播者在关注对象、发布频率和情感表达上存在显著差异；同时利用时间窗口和密度峰值分析，揭示推特信息往往领先新闻媒体5–30小时。

**⚠️ 局限性**

局限性包括仅聚焦推特与新闻两类媒体，未覆盖其他社交平台与更广泛的科学传播途径；分析限定在COVID‑19疫情期间，难以直接推广到其他议题；所提的“信息路径”仅为描述性关联，缺乏因果推断。

---

## 44. Grid Spatial Understanding: A Dataset for Textual Spatial Reasoning over Grids, Embodied Settings, and Coordinate Structures

**arXiv ID:** 2603.17333 | [PDF](https://arxiv.org/pdf/2603.17333v1)

**作者:** Risham Sidhu `[一作]` (University of Illinois at Urbana-Champaign), Julia Hockenmaier `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了GSU（Grid Spatial Understanding）数据集，用纯文本的网格任务评估大型语言模型在空间推理、导航、定位与结构合成方面的能力。

**💡 创新点**

创新点在于去除视觉输入，单纯用文字坐标聚焦空间推理；设计三类核心任务（导航、定位、结构合成）；对比小模型、VLM、前沿模型，并展示通过少量微调即可匹敌甚至超越大模型。

**🔧 技术方法**

使用了文本指令与坐标序列的多种提示方式（1-shot、3-shot），并结合LoRA或全微调技术提升小模型性能；评估指标包括坐标精确匹配、指令一致性、空间/颜色/形状/数值重叠率。

**📊 数据集**

数据集为GSU，包含约3000条训练样本与100条测试样本，涵盖2D/3D网格、基座与本体参考系、简单与复合结构。

**📈 对比分析**

与多类模型（Mistral‑7b、Llama‑8b、Qwen‑7b、VLM版本、GPT‑OSS‑20b、Qwen‑32b、Gemini‑3‑Pro、FlanT5‑large）对比，前沿模型表现最佳；但LoRA微调或全微调的小模型在导航与结构合成任务上已逼近或超过大型模型。

**⚠️ 局限性**

局限性在于任务高度离散化、缺乏连续空间与视觉噪声、未覆盖动态观察角度和遮挡情况，难以直接推广到真实世界或更复杂的嵌入式环境。

---

## 45. Distributed Equilibrium-Seeking in Target Coverage Games via Self-Configurable Networks under Limited Communication

**arXiv ID:** 2603.17335 | [PDF](https://arxiv.org/pdf/2603.17335v1)

**作者:** Jayanth Bhargav `[一作]` (Purdue University), Shreyas Sundaram `[通讯]` (Purdue University)

**通讯引用:** 5546 | [OpenAlex ID](https://openalex.org/A5076731201)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种分布式零和博弈框架，用于在通信受限的情况下让传感器团队协同覆盖可被攻击者动态重新部署的目标。

**💡 创新点**

创新点在于：①利用子模性与价值协同（Value of Coordination）将邻居选择与动作选择统一为分布式带子模优化；②在有限通信带宽下实现近似纳什均衡收敛；③提供理论保证并证明收敛到ε‑NE，满足约束条件。

**🔧 技术方法**

核心技术包括：分布式子模子优化、Bandit子模优化、价值协同度量、EXP3/EXP3.P无后悔学习、分布式邻居选取算法以及无后悔博弈动态。

**📊 数据集**

实验使用仿真场景：30×30 区域、16 方向的固定位置传感器，攻击者可选20种目标部署，通信带宽受限于1–3，比较了最近邻、随机邻居以及无带宽限制三种基线。

**📈 对比分析**

方法与基线比较：在相同带宽下，该框架覆盖率始终优于最近邻和随机邻居，且接近无带宽限制的最优覆盖；在长期游戏中收敛到近似纳什均衡，双重间隙（duality gap）随时间降低。

**⚠️ 局限性**

局限性包括：仅在仿真验证，缺乏真实世界实验；对子模性假设的依赖，在非子模场景下可能不适用；带宽与通信拓扑的动态调整仍存在计算与通信开销；以及攻击者假设为无后悔学习者，若攻击者采用更复杂策略可能影响性能。

---

## 46. Physics-informed offline reinforcement learning eliminates catastrophic fuel waste in maritime routing

**arXiv ID:** 2603.17319 | [PDF](https://arxiv.org/pdf/2603.17319v1)

**作者:** Aniruddha Bora `[一作]` (Texas State University), Chryssostomos Chryssostomidis `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1811 | [OpenAlex ID](https://openalex.org/A5080611073)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于离线强化学习的海上航路规划框架，结合AIS历史轨迹、海洋再分析数据构建物理信息状态，利用专家演示与随机轨迹混合生成离线数据，并通过后置安全盾牌实现对墨西哥湾航线的低碳航行；

**💡 创新点**

创新点在于将物理信息状态与离线强化学习相结合，既利用AIS与再分析数据校准速度损失模型，又通过混合专家演示与随机轨迹保证状态‑动作覆盖，同时采用后置安全盾牌而非奖励约束提供硬安全保证；

**🔧 技术方法**

使用Implicit Q‑Learning进行离线RL训练，构建速度损失与船舶疲劳风险的物理模型，采用A*生成专家演示，加入安全盾牌，并利用Copernicus波浪、NOAA风流等海洋再分析数据作为环境输入；

**📊 数据集**

使用2023年墨西哥湾AIS轨迹、Copernicus海洋波浪再分析、NOAA CoastWatch海流与风、OSCAR海流与CCMP风等多源海洋与气象数据集；

**📈 对比分析**

通过与大圆航线、贪心目标启发式、行为克隆、在线DQN等基线对比，IQL+安全盾牌在7条路线上实现83%到达率（高于大圆78%）、平均航行时间缩短8%、燃料消耗降低5%，并在燃料消耗尾部风险和波浪暴露上显著优于所有基线；

**⚠️ 局限性**

主要局限包括速度损失模型解释力低（R²≈0.02），需针对不同船型进行校准；实验仅在墨西哥湾短中程航线验证，跨洋航线和更复杂海域尚未测试；网格分辨率不足导致近海路线表现差；结果基于仿真，缺乏实船验证；需进一步考虑预测误差、港口排程与运营约束。

---

## 47. Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality

**arXiv ID:** 2603.17512 | [PDF](https://arxiv.org/pdf/2603.17512v1)

**作者:** Mengyu Bu `[一作]` (Institute of Computing Technology Chinese Academy of Sciences), Yang Feng `[通讯]` (Institute of Computing Technology Chinese Academy of Sciences)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 XBridge，一种将外部预训练编码-解码 NMT 模型与冻结的 LLM 组合的 encoder-LLM-decoder 框架，用以扩展 LLM 的多语言理解与生成能力。

**💡 创新点**

通过轻量级跨模型映射层和基于最优传输的 token 对齐目标，解决多模型表示空间不匹配的问题，并采用分阶段训练实现鲁棒的多语言推理与生成。

**🔧 技术方法**

使用编码-解码器组合架构、MLP 映射层、最优传输对齐损失、三阶段（映射、编码器适配、解码器适配）训练策略，以及冻结 LLM 的知识处理核心。

**📊 数据集**

训练和评估数据包括 OPUS‑100、NLLB‑200 NMT 模型、XL‑Sum（摘要）、MGSM（数学推理）、FLORES‑101（翻译）以及多语种数学推理和抽象摘要数据集。

**📈 对比分析**

与 SFT、Translate‑Test、MindMerger、LayAlign 等基线对比，在 FLORES‑101、MGSM、XL‑Sum 等任务上，XBridge 在低资源和未调试语言上显著提升，接近 NMT 性能，同时保持英文能力。

**⚠️ 局限性**

仍存在多语言能力不均衡的局限，主要受外部 NMT 与基准 LLM 组合的影响，需进一步探索统一调和策略。

---

## 48. PC-CrossDiff: Point-Cluster Dual-Level Cross-Modal Differential Attention for Unified 3D Referring and Segmentation

**arXiv ID:** 2603.17753 | [PDF](https://arxiv.org/pdf/2603.17753v1)

**作者:** Wenbin Tan `[一作]` (Xiamen University), Yanyun Qu `[通讯]` (Xiamen University)

**通讯引用:** 6384 | [OpenAlex ID](https://openalex.org/A5076485255)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 PC-CrossDiff 统一框架，解决 3D 视觉 grounding 在多目标场景中隐式定位和空间干扰问题。

**💡 创新点**

创新点在于双层差分注意力（PLDA 与 CLDA）以及统一多任务损失 ℒ_DGTL，可动态提取隐式定位线索并抑制无关空间干扰。

**🔧 技术方法**

技术手段包括双向跨模态差分注意力、聚类级差分注意力、DETR‑style 解码器、PointNet++、RoBERTa、DGCNN 等。

**📊 数据集**

使用数据集 ScanRefer、NR3D、SR3D 及其 Implicit/Multiple 子集进行评估。

**📈 对比分析**

与 MCLN、X‑RefSeg3D、3D‑STMN 等 SOTA 方法比较，PC‑CrossDiff 在 3DREC/3DRES 的多目标和隐式子集上均实现 5–20% 的性能提升，整体取得 state‑of‑the‑art 结果。

**⚠️ 局限性**

局限性：对大规模点云的计算开销仍高，且在极端遮挡或极多目标的极端场景中鲁棒性尚待进一步提升。

---

## 49. Influence of Gripper Design on Human Demonstration Quality for Robot Learning

**arXiv ID:** 2603.17189 | [PDF](https://arxiv.org/pdf/2603.17189v1)

**作者:** Gina L. Georgadarellis `[一作]` (University of Massachusetts Amherst), Meghan E. Huber `[通讯]` (University of Massachusetts Amherst)

**通讯引用:** 1447 | [OpenAlex ID](https://openalex.org/A5045432789)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究通过对8名受试者进行分布式载荷与集中式载荷手持抓取器在绷带开启任务中的可用性评估，探讨机械设计对演示质量的影响。

**💡 创新点**

创新点在于通过改变抓取器手指的载荷分布（从全长分布改为尖端集中），显著提升演示效果，并证明硬件改进可弥补算法限制。

**🔧 技术方法**

采用手持抓取器、手动操作、NASA‑TLX工作量量表以及配套的统计分析（重复测量ANOVA和配对t检验）。

**📊 数据集**

使用自收集的人体实验数据：8名参与者各完成3种抓取器条件（共45次试验），并记录开启率、损伤率、开启时间及工作量评分。

**📈 对比分析**

比较方法为在同一组内测量不同抓取器条件的性能；结果显示集中式载荷抓取器性能接近手操作（开启率100%，时间最快），而分布式载荷抓取器性能最低（开启率仅65.8%，时间最长）。

**⚠️ 局限性**

局限性包括样本量小、仅测试绷带开启任务、抓取器缺乏传感器与标记，因而无法直接推广到完整的学习‑演示‑控制流程，也未评估更复杂物体或真实临床环境下的表现。

---

## 50. Tokenization vs. Augmentation: A Systematic Study of Writer Variance in IMU-Based Online Handwriting Recognition

**arXiv ID:** 2603.16883 | [PDF](https://arxiv.org/pdf/2603.16883v1)

**作者:** Jindong Li `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Björn Eskofier `[通讯]` (Helmholtz Zentrum München - German Research Center for Environmental Health)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了IMU在线手写识别中子词分词与串联数据增强的效果，提出了基于Bigram的分词和拼接增强方法。

**💡 创新点**

创新点在于将子词分词和拼接增强分别针对写手间与写手内方差进行系统比较，并发现分词最适合写手间差异、拼接增强最适合写手内稀疏分布。

**🔧 技术方法**

采用CNN‑LSTM+CTC的REWI架构，结合Bigram/BPE/Unigram分词和基于拼接的时序增强。

**📊 数据集**

使用OnHW‑Words500数据集（约50名右手写手，共500词）。

**📈 对比分析**

通过5折交叉验证比较WD（写手内）和WI（写手间）两种划分；在WI下Bigram将WER从15.40%降至12.99%，在WD下拼接增强将CER从14.86%降至10.04%，WER亦相应下降。

**⚠️ 局限性**

局限在于仅评估了小规模数据集，分词方法基于语言统计未考虑运动轨迹，拼接增强对写手间方差效果不佳，且未探索两种方法联合使用。

---

## 51. Toward Scalable Automated Repository-Level Datasets for Software Vulnerability Detection

**arXiv ID:** 2603.17974 | [PDF](https://arxiv.org/pdf/2603.17974v1)

**作者:** Amine Lbath `[一作]` `[通讯]` (University of Grenoble), Amine Lbath (University of Grenoble)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个自动化的仓库级漏洞基准生成器，能够在真实开源仓库中注入逼真的漏洞并生成可重现的PoV；

**💡 创新点**

创新点在于通过多智能体协同、CodeQL静态分析与对抗性共进化机制，实现大规模可构建、可执行且标签精准的漏洞注入；

**🔧 技术方法**

使用AI代理（Planner、Implementer、Reviewer、Verifier、PoV生成器）、CodeQL查询、容器化构建、测试驱动和可执行证明技术；

**📊 数据集**

以公开开源仓库（如GitHub、GitLab）为样本，结合CVE漏洞数据库和自建注入案例进行数据集构建；

**📈 对比分析**

与ReposVul、VulEval、BountyBench、CVE‑Bench等基准对比，构建/测试通过率≥90%，PoV重现率≥80%，训练的仓库级检测模型在外部基准上平均提升约15%精度；

**⚠️ 局限性**

局限包括对构建环境的高度依赖导致部分仓库无法构建、注入的漏洞类型仍偏向已知CWE、对抗性共进化需要大量计算资源。

---

## 52. VLM2Rec: Resolving Modality Collapse in Vision-Language Model Embedders for Multimodal Sequential Recommendation

**arXiv ID:** 2603.17450 | [PDF](https://arxiv.org/pdf/2603.17450v1)

**作者:** Junyoung Kim `[一作]` (Pohang University of Science and Technology), Hwanjo Yu `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 5034 | [OpenAlex ID](https://openalex.org/A5045521125)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

构建了基于视觉语言模型（VLM）的多模态序列编码框架 VLM2Rec，直接将用户交互历史的文本与图像序列输入 VLM，生成兼具协同过滤信号与语义信息的序列嵌入，并通过对弱模态的惩罚性对比学习与跨模态拓扑正则化来缓解模态崩塌，提升序列推荐性能。

**💡 创新点**

创新点包括①首次将完整的多模态交互历史序列输入 VLM 进行端到端的序列编码；②发现并阐明标准对比监督微调会放大 VLM 的模态崩塌；③提出弱模态惩罚对比学习（WPCL）动态识别并加强弱模态的负样本分离；④提出跨模态拓扑正则化（CRTR）保持模态之间的相对语义结构，防止激进惩罚导致空间扭曲。

**🔧 技术方法**

主要技术手段有：外部融合（分别对文本和图像序列进行提示后独立编码再求和）；LoRA 微调 Qwen2.5‑VL‑3B VLM；InfoNCE 对比损失；WPCL 对比损失加权；CRTR 通过双向 KL 散度对齐模态相似度分布；实验中使用 AdamW、梯度检查点、三轮训练。

**📊 数据集**

在 Amazon 公开数据集的四个领域（Toys、Beauty、Clothing、Sports）进行评估，采用 5‑core 过滤、留一交叉验证、最大序列长度 10，负样本 100，评估 Hit@10/20 与 NDCG@10/20。

**📈 对比分析**

与多类基线对比（ID‑based SR、BERT/CLIP、LLM/VLM 原始与微调、LLMEmb、SLIM 等）以及内部/外部融合方式。实验显示 VLM2Rec 在所有指标上均显著优于基线，尤其在弱模态（图像）贡献上提升明显，且在直接排名和下游 SR 初始化任务中均实现 state‑of‑the‑art 性能。

**⚠️ 局限性**

局限性包括：依赖大规模 VLM 与 LoRA 微调，训练成本高；主要针对文本+图像两模态，未验证对更多模态（音频、视频）的适用性；WPCL 与 CRTR 的超参数敏感，需经验调优；对极端稀疏或冷启动场景的鲁棒性仍有待进一步研究。

---

## 53. DeepStage: Learning Autonomous Defense Policies Against Multi-Stage APT Campaigns

**arXiv ID:** 2603.16969 | [PDF](https://arxiv.org/pdf/2603.16969v1)

**作者:** Trung V. Phan `[一作]` (Technische Universität Chemnitz), Thomas Bauschert `[通讯]` (Technische Universität Chemnitz)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出 DeepStage 框架，将主机和网络证据融合成证明图，并利用该图与阶段估计来驱动自适应的 APT 防御决策。

**💡 创新点**

首次将阶段感知与图神经网络、LSTM 以及分层 PPO 结合，形成阶段条件化奖励与分层策略，显著提升多阶段攻击的检测与处置效率。

**🔧 技术方法**

使用图神经网络 (GNN) 编码证明图，LSTM 估计攻击阶段，分层 Proximal Policy Optimization (PPO) 强化学习，辅以成本敏感奖励设计。

**📊 数据集**

在 6 台 Ubuntu 受控企业测试环境中，利用 MITRE Caldera 生成的 ATT&CK 对齐 APT 演练，并采集 auditd、osquery、CamFlow 与 Zeek 生成的日志与告警。

**📈 对比分析**

与基于 MulVAL 的风险感知 DRL 框架和 DeepStage-Unaware（无阶段感知）对比，DeepStage 在阶段加权 F1 分数上达到 0.89，较基线提升 21.9%，并在成本-效益前沿、学习收敛速度和响应速度上均表现更优。

**⚠️ 局限性**

依赖于预先训练的阶段估计模型，若训练数据或攻击行为与真实场景差异较大可能导致估计误差；同时，框架在大规模企业网络中对图构建和推理的计算开销尚未充分评估。

---

## 54. Multi-Agent Reinforcement Learning for Dynamic Pricing: Balancing Profitability,Stability and Fairness

**arXiv ID:** 2603.16888 | [PDF](https://arxiv.org/pdf/2603.16888v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 55. Conditional Inverse Learning of Time-Varying Reproduction Numbers Inference

**arXiv ID:** 2603.17549 | [PDF](https://arxiv.org/pdf/2603.17549v1)

**作者:** Lanlan Yu `[一作]` (Sichuan University), Xinfu Yang `[通讯]` (Sichuan University)

**通讯引用:** 756 | [OpenAlex ID](https://openalex.org/A5051622851)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了Conditional Inverse Reproduction Learning (CIRL)框架，通过学习历史病例与时间条件的映射直接推断R_t。

**💡 创新点**

创新点在于把R_t估计转化为条件逆问题，软约束与概率观察模型相结合，既保留流行病学结构又具备数据驱动灵活性。

**🔧 技术方法**

采用神经网络的时间坐标嵌入、历史病例编码、多尺度TCN+Transformer、交叉注意力融合以及基于重置方程的零膨胀泊松观测似然。

**📊 数据集**

使用合成疫情模拟、SARS 2003香港数据和Ontario COVID‑19第一波数据进行评估。

**📈 对比分析**

与EpiEstim、EpiNow2和UDENet对比，在合成实验中CIRL的RMSE/MAE与检测延迟均优于或相当于基线，尤其在零膨胀与突变场景下表现突出；在真实数据中与基线保持一致并提供更稳定的预测。

**⚠️ 局限性**

局限在于仍需预先给定生成间隔分布、对极低信息期的鲁棒性有限、模型复杂度较高且缺乏对参数可解释性的充分验证。

---

## 56. A Progressive Visual-Logic-Aligned Framework for Ride-Hailing Adjudication

**arXiv ID:** 2603.17328 | [PDF](https://arxiv.org/pdf/2603.17328v1)

**作者:** Weiming Wu `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**通讯引用:** 4062 | [OpenAlex ID](https://openalex.org/A5101654039)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对打车平台责任纠纷，提出了RideJudge框架，实现了视觉与逻辑对齐的多模态司法推理

**💡 创新点**

创新点在于：① 用SynTraj生成可解释的轨迹-规则对应数据；② Chain‑of‑Adjudication机制实现主动证据查询；③ Adaptive Context Optimization对法规与案例进行场景裁剪；④ Ordinal‑Sensitive Reward在强化学习中对错误等级进行分层惩罚

**🔧 技术方法**

技术手段包括：多模态LLM（以Qwen3‑VL‑8B为基座），视觉轨迹渲染、程序化轨迹合成、知识检索与裁剪、基于树提升的规则过滤、强化学习（DAPO）以及多阶段（视觉对齐→逻辑对齐→强化学习）训练策略

**📊 数据集**

数据集：SynTraj合成轨迹图像-文字对（12,585对），Chain‑of‑Adjudication推理链（14,582条），真实滴滴纠纷测试集（Appeal 1,007例，Driver‑Cancel 453例，Passenger‑Cancel 1,249例）

**📈 对比分析**

与多款文本LLM（DeepSeek‑V3.1、MiniCPM）和多模态LLM（Qwen3‑VL‑8B、Qwen3‑VL‑32B等）对比，RideJudge‑8B在综合准确率上达到 88.41%，显著高于32B规模基线 65.55% 及文本模型 75.25%，在普通与恶意责任分类中也取得领先的Precision/Recall

**⚠️ 局限性**

局限性包括：① 依赖大量人工构造的轨迹合成与规则裁剪，生成过程需专业知识；② 对于极端稀有场景的泛化仍有挑战；③ 强化学习阶段对奖励设计敏感，过度分层化可能导致收敛不稳；④ 仅验证在滴滴数据，跨平台迁移需要进一步评估

---

## 57. DesertFormer: Transformer-Based Semantic Segmentation for Off-Road Desert Terrain Classification in Autonomous Navigation Systems

**arXiv ID:** 2603.17056 | [PDF](https://arxiv.org/pdf/2603.17056v1)

**作者:** Yasaswini Chebolu `[一作]` `[通讯]` (Gayatri Vidya Parishad), Yasaswini Chebolu (Gayatri Vidya Parishad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于 SegFormer B2 的沙漠地形语义分割系统 DesertFormer，能够将沙漠景观划分为十个生态意义明确的类别。

**💡 创新点**

创新点包括针对沙漠地形的类权重与 copy‑paste 采样增广策略、混合 CrossEntropy+Dice 损失、以及对难分类别的系统性误差分析。

**🔧 技术方法**

主要技术包括 SegFormer B2 编码器 + MLP 解码器、类加权交叉熵+Dice 损失、Albumentations 标准增广、copy‑paste 增广、FastAPI 推理服务、CRF 后处理和 MC‑Dropout 不确定性估计。

**📊 数据集**

使用自建的 4,176 张 512×512 像素、覆盖十类沙漠地形的标注图像数据集，专门为无人机和地面机器人设计。

**📈 对比分析**

与 MobileNetV2 版 DeepLabV3 基线对比，mIoU 从 41.0% 提升至 64.4%（+24.2pp），像素准确率达到 86.1%，在去除 Sky 与 Landscape 两类后仍保持 60.4% 的 mIoU。

**⚠️ 局限性**

局限性包括类不平衡仍导致 Logs、Dry Bushes 等稀有类别的分割效果有限、模型仅处理单帧 RGB 缺乏深度或时序上下文、以及在非沙漠环境的泛化能力尚未验证。

---

## 58. A foundation model for electrodermal activity data

**arXiv ID:** 2603.16878 | [PDF](https://arxiv.org/pdf/2603.16878v1)

**作者:** Leonardo Alchieri `[一作]` (Università della Svizzera Italiana), Silvia Santini `[通讯]` (Università della Svizzera Italiana)

**通讯引用:** 1358 | [OpenAlex ID](https://openalex.org/A5102715113)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了EDAMAME大规模可公开的EDA数据集（约25,000小时、634名用户），并训练了首个专用基础模型EDAM

**💡 创新点**

首次将24个公开/共享协议的Empatica E4数据集聚合成统一格式，利用自监督对比学习专门训练EDA基础模型，显著降低模型参数与计算量

**🔧 技术方法**

采用自监督对比学习（InfoNCE）、EfficientNet 1D架构、Butterworth低通滤波、cvxEDA分离等预处理技术

**📊 数据集**

来自24个公开/共享协议的Empatica E4数据集，总计约25,000小时、634名用户

**📈 对比分析**

通过线性探针在10个二分类下游任务与通用手工特征、EDA专用手工特征以及Chronos/Mantis/MOMENT等基线对比；在8/10任务上优于手工特征，与大型通用模型相当，但参数量仅1M，计算成本约低20倍

**⚠️ 局限性**

仅使用单一设备（E4）导致设备偏差、噪声大、标签噪声及高阶情绪/压力标签难以单独识别，数据分布仍受原始数据集偏差影响

---

## 59. On the generic information capacity of relational schemas with a single binary relation

**arXiv ID:** 2603.17664 | [PDF](https://arxiv.org/pdf/2603.17664v1)

**作者:** Benoît Groz `[一作]` (Université Paris-Saclay), Piotr Wieczorek `[通讯]` (University of Wrocław)

**通讯引用:** 5180 | [OpenAlex ID](https://openalex.org/A5039938845)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

研究单一二元关系模式的通用主导性，给出了20个模式的完整分类。

**💡 创新点**

首次对带键和包含依赖的单二元关系模式实现完全通用主导关系图，并扩展到三元关系和带标识符的情形。

**🔧 技术方法**

使用泛型映射、绝对/强主导性理论、计数函数和组合数学（如拉丁方、矩阵计数）证明。

**📊 数据集**

无具体数据集，采用理论计数和组合公式。

**📈 对比分析**

通过构造可注入泛型映射与计数比较证明严格主导性；性能未涉及实验。

**⚠️ 局限性**

局限：仅限单二元关系，缺乏可计算化实现的通用性；三元情况仅考虑键约束；标识符扩展仍处于理论阶段。

---

## 60. SHIFT: Motion Alignment in Video Diffusion Models with Adversarial Hybrid Fine-Tuning

**arXiv ID:** 2603.17426 | [PDF](https://arxiv.org/pdf/2603.17426v1)

**作者:** Xi Ye `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 67535 | [OpenAlex ID](https://openalex.org/A5115666530)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出像素级运动奖励和SHIFT混合微调框架，提升视频扩散模型的运动保真度。

**💡 创新点**

①将光流残差与点轨迹作为即时与长期运动奖励；②融合监督微调与优势加权更新的Smooth Hybrid Fine‑Tuning，并加入对抗奖励训练与噪声对齐，解决动态度退化和奖励劫持。

**🔧 技术方法**

光流残差计算、点轨迹跟踪、Vision Transformer奖励判别器、LoRA微调、前向KL目标、优势加权回归、对抗奖励共训练、噪声级对齐以及基于前向扩散的RL框架。

**📊 数据集**

DAVIS2017（SVD fine‑tuning）、WISA‑80K（Wan2.2 TI2V fine‑tuning）以及官方评测集VBench‑I2V。

**📈 对比分析**

与SFT、Track4Gen、DenseDPO、FlowGRPO等对比；SHIFT在保持视觉质量的同时，VBench Motion和Motion Score提升至与FlowGRPO相近，整体分数最高，计算成本仅为FlowGRPO的约1/8（2.46× vs 19.35×）。

**⚠️ 局限性**

仍需额外训练奖励网络，存在奖励劫持风险；奖励基于像素级光流，可能对复杂光照/遮挡敏感；方法主要针对图像条件视频扩散，未验证跨域或长序列泛化。

---

## 61. AI-Driven Multi-Modal Adaptive Handover Control Optimization for O-RAN

**arXiv ID:** 2603.17158 | [PDF](https://arxiv.org/pdf/2603.17158v1)

**作者:** Abdul Wadud `[一作]` (Bangladesh Institute of Governance and Management), Nima Afraz `[通讯]` (School of Computer Science)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于O‑RAN非实时与实时RIC分层的多模态适应性切换控制框架，利用移动模式分类、短程轨迹与RSRP预测以及PPO策略来实现更精准的邻区排名与切换决策。

**💡 创新点**

将多模态移动性感知与预测整合至非实时rApp，使用PPO生成邻区排名后通过A1交给近实时xApp，实现低延迟与长时序预测兼顾，并实现零切换抖动。

**🔧 技术方法**

k‑NN移动模式分类器、随机森林短程轨迹与RSRP回归、Proximal Policy Optimization（PPO）强化学习，以及O‑RAN 7.2x架构的A1/E2接口。

**📊 数据集**

在MATLAB 5G Toolbox仿真中生成的100用户、7种移动模式的合成轨迹与测量数据（仿真数据集）。

**📈 对比分析**

与传统A3、O‑RAN ML助理、负载平衡等基线对比，结果显示吞吐量保持不变、切换失败率约0.036，完全消除抖动；相较传统基线表现更稳定。

**⚠️ 局限性**

仅在仿真环境验证，未在真实O‑RAN硬件上测试；模型训练需离线更新，对极端高速度或极短切换周期的鲁棒性尚待进一步评估。

---

## 62. A Simpler Analysis for $\varepsilon$-Clairvoyant Flow Time Scheduling

**arXiv ID:** 2603.17542 | [PDF](https://arxiv.org/pdf/2603.17542v1)

**作者:** Anupam Gupta `[一作]` (New York University), Sorrachai Yingchareonthawornchai `[通讯]` (Institute for Theoretical Studies ETH Zürich)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提供了对短路下界优先（SLF）算法在 -clairvoyant 调度设置下实现最优性的简化证明，取代了原先的早到达引理和复杂的分配不等式；

**💡 创新点**

创新点在于通过引入“冻结作业”概念和更弱的体积不等式，显著简化了证明流程，同时保持了原有的最优竞争比；

**🔧 技术方法**

使用的技术包括：冻结作业定义、快速前进引理（Fast Forward Lemma）和后缀裁剪引理（Suffix Carving Lemma），以及对作业剩余时间的体积上界分析；

**📊 数据集**

由于本文属于理论分析性质，未使用任何实验数据集；

**📈 对比分析**

与原始工作对比，本文取得与原工作相同的竞争比 1/(1‑ε)（ε∈(0,1]），并证明该比值在确定性算法中是最优的；

**⚠️ 局限性**

局限性包括：仅适用于单机、单处理器场景；仅针对 deterministic SLF；未给出实验验证或多机扩展的分析。

---

## 63. Trust, Safety, and Accuracy: Assessing LLMs for Routine Maternity Advice

**arXiv ID:** 2603.16872 | [PDF](https://arxiv.org/pdf/2603.16872v1)

**作者:** V Sai Divya `[一作]`, K Venkata Krishna Rao `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估ChatGPT-4o、Perplexity AI和GeminiAI在印度孕期健康信息查询中的可读性、准确性和文化敏感性。

**💡 创新点**

首次将可读性指标与语义相似度、名词实体重叠结合评估LLM在孕期信息提供中的表现。

**🔧 技术方法**

可读性指标（FRE、FKGL、GFI等）、余弦相似度、Jaccard相似度等文本分析技术。

**📊 数据集**

17个印度常见孕期误区及相关问题的问答，LLM与经验产科医生的回答。

**📈 对比分析**

对比LLM生成文本与专家答案，使用可读性得分、余弦相似度和Jaccard相似度；ChatGPT可读性最高，Perplexity语义相似度最高，ChatGPT名词重叠最好。

**⚠️ 局限性**

样本量有限，仅覆盖17个问题；缺乏真实用户测试与多地区验证，模型可能存在文化误解或信息偏差。

---

## 64. SegFly: A 2D-3D-2D Paradigm for Aerial RGB-Thermal Semantic Segmentation at Scale

**arXiv ID:** 2603.17920 | [PDF](https://arxiv.org/pdf/2603.17920v1)

**作者:** Markus Gross `[一作]` (TU Munich), Daniel Cremers `[通讯]` (TU Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模无人机 RGB‑热像语义分割基准 SegFly，利用少量 RGB 标注通过 3D 重建与投影实现 RGB 与热像的自动伪标签生成。

**💡 创新点**

创新点在于提出 2D‑3D‑2D 流水线：将少量 RGB 标注提升到 3D 点云，再投影到所有 RGB 与热像视角，实现无人工干预的伪标签与精确 RGB‑T 对齐。

**🔧 技术方法**

技术手段包括 SfM+MVS 3D 重建、点云投影 + Z‑buffer、稀疏‑稠密化语义渲染、ICP 对齐、Rein 适配器、Firefly 三阶段训练与轻量 MLP 头。

**📊 数据集**

使用了 OccuFly 原始数据（9 个场景、20,606 张 RGB、15,007 张热像），扩展为 SegFly‑RGB 与 SegFly‑RGB‑T 两大子集。

**📈 对比分析**

在 SegFly 上对 UPerNet、SegFormer、Firefly 以及 Vision Foundation Models（CatSeg、AnyThermal）进行基准评估，RGB 语义精度达 91%/85%（Acc/FWmIoU），热像 88%/79%，并在零样本到微调后提升 30‑40% 以上。

**⚠️ 局限性**

局限在于对动态物体/薄结构的 3D 重建不可靠、热像仅覆盖日间数据，且仍需少量人工标注。

---

## 65. Differential Privacy in Generative AI Agents: Analysis and Optimal Tradeoffs

**arXiv ID:** 2603.17902 | [PDF](https://arxiv.org/pdf/2603.17902v1)

**作者:** Ya-Ting Yang `[一作]` (New York University), Quanyan Zhu `[通讯]` (New York University)

**通讯引用:** 11241 | [OpenAlex ID](https://openalex.org/A5081500464)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于差分隐私的生成式AI代理框架，分析温度和消息长度对隐私泄漏与效用的影响，并给出温度最优设计方案。

**💡 创新点**

创新点在于：1）同时给出 token 级和消息级的 DP 定义；2）推导出温度、长度对隐私泄漏的显式界限；3）将隐私-效用平衡转化为可解的温度优化问题。

**🔧 技术方法**

主要技术包括：差分隐私理论、温度缩放 softmax、序列生成的自回归模型、组合性质、Gibbs 分布推导、以及在 GPT‑2 上的实证评估。

**📊 数据集**

使用了模拟的企业安全事件数据库 D 及其邻近数据集 D' 作为测试数据。

**📈 对比分析**

通过平滑经验分布计算隐私损失、总变距离、Jensen‑Shannon 距离等指标进行比较；实验结果显示温度升高可显著降低隐私泄漏，然而效用随温度下降；消息长度越长，隐私泄漏和效用波动越大。

**⚠️ 局限性**

局限性：仅研究单一代理；未考虑多代理协同或其他控制参数；实证仅基于 GPT‑2，缺乏在真实企业大模型或多种数据集上的验证。

---

## 66. Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing

**arXiv ID:** 2603.17583 | [PDF](https://arxiv.org/pdf/2603.17583v1)

**作者:** Seongrae Noh `[一作]` (Korea University), HyeongYeop Kang `[通讯]` (Korea University)

**通讯引用:** 233 | [OpenAlex ID](https://openalex.org/A5011229651)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Edit-As-Act 框架，将 3D 室内场景编辑转化为基于目标回归的符号规划任务；

**💡 创新点**

创新点在于构造了专门的 EditLang 行为语言、实现了基于 LLM 的规划与验证循环，以及源感知的回归策略，使编辑既精准、最小化，又符合物理约束；

**🔧 技术方法**

核心技术包括 PDDL 风格的 EditLang、LLM 驱动的规划器与验证器、基于 STRIPS 的源感知回归、以及可执行的 Python DSL；

**📊 数据集**

使用了自建的 E2A‑Bench 数据集，包含 9 个室内环境共 63 个开放词汇编辑任务；

**📈 对比分析**

在 E2A‑Bench 上与 LayoutGPT‑E、AnyHome、ArtiScene 以及 SceneWeaver 等基线进行比较，Edit‑As‑Act 在指令保真度、语义一致性和物理合理性三项指标上均取得最高分，且模型调用次数与计算开销保持竞争力；

**⚠️ 局限性**

局限性包括对高度模糊指令的处理不佳、某些几何合法但风格次优的编辑、以及在严格单调性约束下偶发的规划死锁现象。

---

## 67. ECHO: Towards Emotionally Appropriate and Contextually Aware Interactive Head Generation

**arXiv ID:** 2603.17427 | [PDF](https://arxiv.org/pdf/2603.17427v1)

**作者:** Xiangyu Kong `[一作]` (University of Exeter), Siyang Song `[通讯]` (University of Exeter)

**通讯引用:** 1159 | [OpenAlex ID](https://openalex.org/A5053061988)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种名为ECHO的交互式头像生成框架，能够在听/说交互中生成情境恰当、情感合理、口型同步的面部行为。

**💡 创新点**

引入长程上下文理解 (LCU) 结合多模态与语言情感推理，并设计块级空间解耦交叉注意力 (SDCM) 以保留口型同步并融合用户视觉线索。

**🔧 技术方法**

多模态 Mamba 网络、跨模态对齐、LLM 情感推理、分块解耦交叉注意力、两阶段 flow‑matching 训练与层次特征对齐等技术。

**📊 数据集**

使用 Seamless Interaction、RealTalk、HDTF 等多场景对话视频数据集。

**📈 对比分析**

与多种 SOTA IHG/听/说模型在 FID、FVD、LSE‑D/C、rPCC、SID 等指标上对比，ECHO 在情感一致性、口型同步和视觉质量上均居首，尤其在 rPCC、SID、LSE‑D 方面显著提升。

**⚠️ 局限性**

推理时仍需多步 flow‑matching，导致延迟较高，尚未实现实时交互。

---

## 68. Towards Unsupervised Adversarial Document Detection in Retrieval Augmented Generation Systems

**arXiv ID:** 2603.17176 | [PDF](https://arxiv.org/pdf/2603.17176v1)

**作者:** Patrick Levi `[一作]` `[通讯]` (Ostbayerische Technische Hochschule Amberg-Weiden), Patrick Levi (Ostbayerische Technische Hochschule Amberg-Weiden)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了无监督检测 Retrieval Augmented Generation（RAG）系统中对抗性上下文（AC）的方案，并通过统计异常检测验证其可行性。

**💡 创新点**

首次提出在不依赖目标提问的情况下使用生成器激活、TokenSAR熵和嵌入向量三种指标进行AC检测，并证明使用简单摘要提示即可实现检测。

**🔧 技术方法**

采用统计异常检测（Grubb’s test）、Llama‑3.1 8B 生成器的最后一层激活、TokenSAR 熵值、MPNet 嵌入向量等技术。

**📊 数据集**

使用 HotpotQA 数据集生成问题与合法上下文，PoisonedRAG 提供对抗性上下文，Llama‑3.1 8B 作为生成器，Mistral‑7B 作为答案验证。

**📈 对比分析**

与针对问题提示的检测进行对比，利用 TokenSAR、激活、嵌入等指标的组合，检测准确率约为 80.5%，约 19.5% 的攻击仍未被检测。

**⚠️ 局限性**

局限性包括样本量有限导致未评估误报率、仅在 PoisonedRAG 上测试缺乏对更复杂攻击（如 BADRAG、GARAG）的验证，以及生成器激活信息在商业 API 中可能不可获取。

---

## 69. Script-to-Slide Grounding: Grounding Script Sentences to Slide Objects for Automatic Instructional Video Generation

**arXiv ID:** 2603.16931 | [PDF](https://arxiv.org/pdf/2603.16931v1)

**作者:** Rena Suzuki `[一作]` (Nagoya Institute of Technology), Tadachika Ozono `[通讯]` (Nagoya Institute of Technology)

**通讯引用:** 397 | [OpenAlex ID](https://openalex.org/A5025626453)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个自动生成带视觉效果的幻灯片视频的系统，提出并实现了Script-to-Slide Grounding (S2SG) 的框架，并在文本幻灯片上实现了Text‑S2SG算法；

**💡 创新点**

将原本隐式的幻灯片视频编辑任务正式定义为可计算的S2SG问题；采用分阶段策略，先利用LLM完成文本层级的 grounding，证明在文本幻灯片上可获得高 F1 分数；

**🔧 技术方法**

使用大语言模型（Gemini 2.5 Flash）进行 grounding，配合 python‑pptx 提取文本与层级信息，利用提示工程指导LLM，并通过视频生成模块实现视觉效果；

**📊 数据集**

构建了一个包含 19 页文本幻灯片和 94 条脚本句子的测试集，来自四位学术演示者，真值由作者手工标注；

**📈 对比分析**

对四种幻灯片信息格式（是否包含层级和风格信息）进行微平均 F1 评估，平均 F1 为 0.924，表明信息量对性能影响不大，误差主要集中在标题和父子节点的误匹配；

**⚠️ 局限性**

仅针对纯文本幻灯片，未覆盖图形、表格等非文本对象；仅验证 grounding，未解决 Attention Control Problem；未来需进一步研究视觉效果的选择与时序控制。

---

## 70. Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift

**arXiv ID:** 2603.17372 | [PDF](https://arxiv.org/pdf/2603.17372v1)

**作者:** Zhihua Wei `[一作]` (Tongji University), Wen Shen `[通讯]` (Tongji University)

**通讯引用:** 11713 | [OpenAlex ID](https://openalex.org/A5032104899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉语言模型（VLM）在加入图像后出现的“越狱”现象，发现越狱案例在表示空间中形成了一个与正常、拒绝状态明显分离的内部状态；基于此，提出了“越狱方向”与“越狱相关位移”概念，并设计了仅在推理时移除该位移的防御方法JRS‑Rem；该方法在不影响模型实用性的前提下显著提升安全性；

**💡 创新点**

创新点在于：①首次将越狱行为映射到表示空间的特定方向；②提出量化越狱相关位移的投影指标，并证明其与越狱成功率高度相关；③基于该指标实现轻量级推理时防御（JRS‑Rem），实现了对多种越狱场景的统一解释和有效防护；

**🔧 技术方法**

技术方法包括：表示空间分析（PCA、距离度量、线性探针）、越狱方向的定义与投影计算、CLIP图文相似度评估、JRS‑Rem推理时的位移去除（阈值τ=0.2）、ASR评估（结合关键词、Qwen3Guard、Llama‑Guard三种判定器的投票）等；

**📊 数据集**

使用的主要数据集包括：HADES、MM‑SafetyBench、RedTeam‑2K（显式、隐式越狱场景），MML‑R/M/M‑B64/Gradient（对抗攻击场景），MM‑Vet、MME、ScienceQA（benign评测）；

**📈 对比分析**

与现有防御方法AdaShield、ECSO、ShiftDC、CMRM等进行对比；在LLaVA‑1.5‑7B、ShareGPT4V‑7B、InternVL‑Chat‑19B三种VLM上，JRS‑Rem在所有越狱场景下的ASR均显著下降（下降幅度最高可达70%以上），同时在benign基准上的性能几乎无损；

**⚠️ 局限性**

局限性包括：①依赖底层LLM的安全对齐，若基础模型对齐不足，越狱相关位移可能不明显；②只针对图像引发的越狱，无法防文本-only越狱；③对更大规模模型的可扩展性尚未验证；

---

## 71. ArchBench: Benchmarking Generative-AI for Software Architecture Tasks

**arXiv ID:** 2603.17833 | [PDF](https://arxiv.org/pdf/2603.17833v1)

**作者:** Bassam Adnan `[一作]` (International Institute of Information Technology Hyderabad), Karthik Vaidhyanathan `[通讯]` (International Institute of Information Technology Hyderabad)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 ArchBench 平台，统一评测大语言模型在软件架构任务上的表现

**💡 创新点**

首次提供插件式任务框架和标准化评测管线，汇聚多任务、多数据集与统一指标

**🔧 技术方法**

使用 Python CLI、React Web、统一 LLM 接口、NLP 与代码质量评估指标（ROUGE、BLEU、BERTScore、CodeBLEU、精确率/召回率）

**📊 数据集**

采用 ADR、Serverless、Dynamic Service、Traceability Link、Microservice 五个公开任务的数据集

**📈 对比分析**

通过统一指标对不同模型进行比较，并在 Leaderboard 展示多模型成绩，性能差异可直观比较

**⚠️ 局限性**

评测受限于现有数据集和指标，未覆盖所有架构领域，缺乏沙箱环境与完整的 agent‑based 评测，指标可能不足以全面评估架构质量

---

## 72. VisionNVS: Self-Supervised Inpainting for Novel View Synthesis under the Virtual-Shift Paradigm

**arXiv ID:** 2603.17382 | [PDF](https://arxiv.org/pdf/2603.17382v1)

**作者:** Hongbo Lu `[一作]` (Shanghai Jiao Tong University), Pai Peng `[通讯]` (COWARobot Co. Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种只使用摄像头的自监督填充框架 VisionNVS，用于在自由轨迹下的视角合成。

**💡 创新点**

核心创新是将视角合成问题转化为自监督填充任务，采用 Virtual-Shift 生成遮挡掩码并利用邻置相机信息的 Pseudo‑3D Seam Synthesis，消除了训练时的监督缺口和几何误差。

**🔧 技术方法**

主要技术包括单目深度估计、虚拟视角偏移、遮挡掩码生成、邻置相机图像融合、DiT 拓扑的扩散式填充网络与流匹配训练目标。

**📊 数据集**

使用 nuScenes 大规模驾驶数据集进行评估，包含 700 条训练、150 条验证、150 条测试场景。

**📈 对比分析**

与 LiDAR 依赖的 FreeVS、DiST‑4D 等方法对比，VisionNVS 在 FID/FVD 指标上均优于基线，尤其在远距离偏移（±4 m）下显著提升，且保持低显存和较快推理速度。

**⚠️ 局限性**

局限性包括对单目深度估计的依赖，深度误差仍可能影响遮挡掩码精度；在极端遮挡或动态物体场景中填充质量仍有提升空间。

---

## 73. Is Your LLM-as-a-Recommender Agent Trustable? LLMs' Recommendation is Easily Hacked by Biases (Preferences)

**arXiv ID:** 2603.17417 | [PDF](https://arxiv.org/pdf/2603.17417v1)

**作者:** Zichen Tang `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10545 | [OpenAlex ID](https://openalex.org/A5100730785)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出BiasRecBench，系统评估LLM在“LLM-as-a-Recommender”情境下对情境相关偏差的鲁棒性；

**💡 创新点**

设计了带有校准质量边际的Bias Synthesis Pipeline，能够在保证候选项相似质量的前提下注入逻辑合理的偏差，真正暴露LLM的潜在偏差；

**🔧 技术方法**

利用对抗式偏差注入（固定术语拼接或生成式改写）以及多模型推理与对齐技术；

**📊 数据集**

基于学术论文评审、电子商务商品推荐和招聘简历筛选三大公开数据集构建评测集合；

**📈 对比分析**

通过Acc_ori、Acc_inj和RR等指标对Gemini‑2.5/3‑Pro、GPT‑4o、DeepSeek‑R1及小模型进行比较，结果显示即使在质量差距被严格控制时，SOTA模型的Acc_inj 下降可高达30%+，RR下降显著；

**⚠️ 局限性**

局限性在于仅覆盖少量闭源SOTA模型，候选池规模被限制为5个，未覆盖大规模真实检索场景。

---

## 74. AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse

**arXiv ID:** 2603.18000 | [PDF](https://arxiv.org/pdf/2603.18000v1)

**作者:** Zhang Zhang `[一作]` (Peking University), Zheng Liu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 25065 | [OpenAlex ID](https://openalex.org/A5100453158)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 AgentFactory，一个三阶段（Install → Self‑Evolve → Deploy）自演化框架，用来自动构造、改进并导出可执行的子代理。

**💡 创新点**

创新点在于将成功经验保存为可执行 Python 代码，并通过执行反馈循环（generate‑feedback‑modify）不断改进子代理，而不是仅记录文字反思；子代理可跨系统、跨框架复用。

**🔧 技术方法**

核心技术包括：Meta‑Agent 任务拆解与工具分配、统一 Skill 系统（Meta‑Skill、Tool‑Skill、Subagent‑Skill）、工作空间管理、LLM（Claude Opus/Sonnet）驱动的代码生成与自适应修改。

**📊 数据集**

使用作者自行设计的两批真实任务（共 30 个涵盖 Web 爬取、数据可视化、浏览器自动化、音频处理等）进行评估，未使用公开大规模数据集。

**📈 对比分析**

与 ReAct 基线及文本经验自演化基线对比，评估指标为每任务平均输出 token（排除子代理内部 LLM 消耗）。在 Batch 2 上，AgentFactory 将 token 消耗从 8k 降至 3k，显著优于两基线；即使在 Batch 1，强模型 Opus 4.6 也已出现子代理复用效益。

**⚠️ 局限性**

局限性包括：仅支持 Web 接口交互，无法处理非 Web 任务；缺少视觉/多模态输入；子代理在极端或未知场景下仍可能失效；安全性需通过外部审计与权限控制保障。

---

## 75. Concept-to-Pixel: Prompt-Free Universal Medical Image Segmentation

**arXiv ID:** 2603.17746 | [PDF](https://arxiv.org/pdf/2603.17746v1)

**作者:** Haoyun Chen `[一作]` (University of Science and Technology of China), Shaohua Kevin Zhou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5101592407)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出了一种无需提示的通用医学图像分割框架C2P，能够在单一模型中处理多模态任务；

**💡 创新点**

核心创新是将解剖知识拆分为几何（Geometric Tokens）和语义（Semantic Tokens）两类，并通过多模态大语言模型提炼医学概念；

**🔧 技术方法**

采用多模态LLM（如Qwen3‑VL‑Plus）生成文本概念，利用PubMedBERT编码；利用跨注意力实现Token与图像特征的双向交互；用Token‑Guided Dynamic Head生成样本特定的卷积核；引入Geometry‑Aware Inference Consensus自检机制；

**📊 数据集**

在八个公开数据集（覆盖OCT、MRI、CT、US、Endoscopy、Dermoscopy、Pathology、EM等七种模态）上训练，并在额外的八个未见数据集（包含X‑ray、EM等）进行零样本测试；

**📈 对比分析**

与十类专用模型（如UNet、nnU‑NetV2）和四类通用模型（如Spider、SR‑ICL）对比，C2P在所有七种模态上平均Dice达88.22%，显著优于之前SOTA；在零样本场景下亦能保持竞争力；

**⚠️ 局限性**

对形态学约束依赖导致在非紧凑结构（如神经网络EM图像）上表现不佳；同时对大量域偏差的多模态数据仍需进一步强化多样性训练。

---

## 76. Variational Rectification Inference for Learning with Noisy Labels

**arXiv ID:** 2603.17255 | [PDF](https://arxiv.org/pdf/2603.17255v1)

**作者:** Haoliang Sun `[一作]` (Shandong University), Yilong Yin `[通讯]` (Shandong University)

**通讯引用:** 5833 | [OpenAlex ID](https://openalex.org/A5100672590)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了变分校正推断（VRI）方法，用以在存在噪声标签的场景下通过元学习自动校正损失函数，提升深度模型的鲁棒性。

**💡 创新点**

创新点在于将损失校正视为一项可化简的变分推断问题，构建分层贝叶斯模型并加入KL正则化以防止蒙特卡洛近似导致的模型崩塌，同时在元学习框架下实现二层优化并给出收敛理论。

**🔧 技术方法**

采用了变分推断、贝叶斯元学习、KL散度正则、重参数化技巧、元网络与先验网络、以及二层优化算法等技术手段。

**📊 数据集**

在CIFAR‑10、CIFAR‑100、Clothing1M、Food‑101N、ANIMAL‑10N以及开放集噪声数据集CIFAR‑80N等多个公开数据集上进行实验。

**📈 对比分析**

与现有的重加权、损失校正与元学习方法（如MW‑Net、MSLC、PMW‑Net、ELR等）以及非元学习基线进行对比，VRI在多种噪声类型和噪声比例下均达到或超过SOTA，在开放集噪声和高噪声率下的优势尤为明显，同时使用更少的MC采样即可获得更好的性能。

**⚠️ 局限性**

仍需依赖一定量的干净元数据（或需构造平衡的伪元数据），元网络的结构与超参数（如λ、采样数k）对结果有一定敏感性；二层优化带来额外的计算成本，且在极大规模数据集或极端噪声环境下的可扩展性尚待进一步验证。

---

## 77. On Securing the Software Development Lifecycle in IoT RISC-V Trusted Execution Environments

**arXiv ID:** 2603.17757 | [PDF](https://arxiv.org/pdf/2603.17757v1)

**作者:** Annika Wilde `[一作]` (Ruhr University Bochum), Ghassan Karame `[通讯]` (Ruhr University Bochum)

**通讯引用:** 6147 | [OpenAlex ID](https://openalex.org/A5059087800)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并实现了一个可插拔的工具包，扩展RISC‑V TEE的安全监控层，以实现可信时间、状态连续性、以及安全的 enclave 更新与迁移功能。

**💡 创新点**

创新点在于首次在 RISC‑V TEE 生态中提供完整的软件生命周期管理，涵盖原子性迁移、回滚防护、状态保密与完整性，并通过模块化设计实现与 Keystone、CURE 等主流 TEE 的兼容。

**🔧 技术方法**

技术实现基于 RISC‑V 指令集、PMP 内存保护、可信监控器（SM）扩展、密钥派生与对称加密、以及 RISC‑V SBI 接口；实验中使用 VisionFive2 板卡、StarFive JH7110 处理器和 eMMC RPMB 存储。

**📊 数据集**

评估数据集主要为不同大小（1 KB 至 1 MB）的 enclave 状态文件，以及在 311 Mbit/s 网络链路上传输的 OTA 更新包；同时测量 CPU 密集与内存密集背景负载下的性能。

**📈 对比分析**

与现有的 SGX/SEV‑SNP 迁移方案对比，本文方案在安全属性（S1–S6）上均满足或优于对手；性能上，本方案实现更新延迟 <0.7 s、迁移延迟 <1.5 s，服务停机时间分别 <10 ms 与 <700 ms，且在 1 MB 大状态下仍保持在 3.5 s 内。

**⚠️ 局限性**

主要限制包括：① 受 PMP 资源限制，仅能支持有限数目的并发 enclave；② 对状态大小的依赖，超过数十 KB 时停机时间会增长至数秒；③ 评估未包含 TLS 握手和跨平台（除 Keystone）部署的细节；④ 仅在单一硬件平台上验证，缺乏更广泛的硬件多样性测试。

---

## 78. Draft-and-Prune: Improving the Reliability of Auto-formalization for Logical Reasoning

**arXiv ID:** 2603.17233 | [PDF](https://arxiv.org/pdf/2603.17233v1)

**作者:** Zhiyu Ni `[一作]` (University of California), Pierluigi Nuzzo `[通讯]` (University of California)

**通讯引用:** 2885 | [OpenAlex ID](https://openalex.org/A5067636168)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种推理时的多路径草拟‑修剪‑投票框架，以提升自动形式化（Auto‑Formalization）在逻辑推理任务中的鲁棒性。

**💡 创新点**

创新点包括：在计划层面产生多样化的自然语言草稿；利用求解器验证剔除矛盾或歧义的可执行程序；通过多数投票对多路径结果进行集成。

**🔧 技术方法**

采用大型语言模型（GPT‑4/GPT‑4o）完成计划草拟和程序生成；使用符号求解器（如Z3Py）执行与验证；通过多路径采样与投票机制实现集成。

**📊 数据集**

使用了四个推理基准数据集：AR‑LSAT、ProofWriter、PrOntoQA 和 LogicalDeduction。

**📈 对比分析**

与现有 AF 基线（MAD‑LOGIC、CLOVER）以及 CoT、SymbCoT 等方法对比；在 AR‑LSAT 上准确率提升超过 30%，在 GPT‑4o 版本下在 PrOntoQA 与 LogicalDeduction 上几乎达到 100% 的性能。

**⚠️ 局限性**

局限性包括：推理时计算成本和延迟显著增加；未对基础生成器进行自适应训练或微调；聚合方式未考虑程序的符号等价性；仅在学术基准上验证，未能在高风险应用场景下证明完整安全性。

---

## 79. Consistency-Driven Dual LSTM Models for Kinematic Control of a Wearable Soft Robotic Arm

**arXiv ID:** 2603.17672 | [PDF](https://arxiv.org/pdf/2603.17672v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 80. Towards Infinitely Long Neural Simulations: Self-Refining Neural Surrogate Models for Dynamical Systems

**arXiv ID:** 2603.17750 | [PDF](https://arxiv.org/pdf/2603.17750v1)

**作者:** Qi Liu `[一作]` (New York University), Joan Bruna `[通讯]` (New York University)

**通讯引用:** 21611 | [OpenAlex ID](https://openalex.org/A5112569280)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了自我改进神经替代模型（Self‑Refining Neural Surrogate, SNS），利用多噪声层去噪oracle和条件扩散模型，动态推断条件变量的最佳噪声水平，以在保持短期精度的同时实现长期分布一致性；

**💡 创新点**

创新点在于将条件近似误差与离群误差的权衡显式化为多噪声层去噪oracle，并通过自适应噪声学习实现无超参数的长时序稳定性；

**🔧 技术方法**

核心技术包括条件扩散模型、去噪分数匹配、分数估计的多噪声级别扩散过程，以及自我改进的反向扩散策略；

**📊 数据集**

实验数据集主要是物理动力学仿真数据：Kolmogorov 流动（Navier–Stokes）以及两层准地球层（Quasigeostrophic）湍流；

**📈 对比分析**

与 ACDM、Thermalizer 等现有方法对比，SNS 在 Kolmogorov 流动上保持了更好的时间相关性与能谱一致性，且不需要超参数调节；在两层 QG 湍流上，SNS 成功完成长时间轨迹生成，而其他方法要么失效，要么需要大量扩散步数；

**⚠️ 局限性**

局限性包括训练目标更难、需要更细的扩散时间离散化导致计算成本高、以及对高维度分布的 KL 等度量评估仍缺乏理论指导；

---

## 81. Identity as Presence: Towards Appearance and Voice Personalized Joint Audio-Video Generation

**arXiv ID:** 2603.17889 | [PDF](https://arxiv.org/pdf/2603.17889v1)

**作者:** Yingjie Chen `[一作]` (WeChat Vision, Tencent Inc.), Jing Lyu `[通讯]` (WeChat Vision, Tencent Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了统一可扩展的身份感知音视频生成框架Identity-as-Presence，支持单/多主体面部外观和声纹个性化同步生成。

**💡 创新点**

创新点包括自动化身份标签数据收集管线、多模态身份注入机制（共享身份嵌入+结构化时空对齐+异构自注意力）以及分阶段训练策略。

**🔧 技术方法**

使用双塔Diffusion Transformer、VAE编码、身份嵌入、RoPE、对称/异构自注意力、音视频流匹配、语音分离等技术。

**📊 数据集**

数据集为自构建的身份标注音视频对集合，包含单/多主体场景，使用YOLO、MOTRv2、Demucs、3D-Speaker、SyncNet等自动提取，规模达数十万小时。

**📈 对比分析**

在自制的100个triplet基准上与Phantom、HunyuanCustom、Stand-In、Ovi、LTX-2、UniAVGen等SOTA对比，音频质量、视频逼真度、同步性指标均优于同类模型。

**⚠️ 局限性**

局限在多模态大规模并行训练成本高，跨模态同步仍有轻微偏差，且对极端口语变体或少量视角仍需改进。

---

## 82. Part-Aware Open-Vocabulary 3D Affordance Grounding via Prototypical Semantic and Geometric Alignment

**arXiv ID:** 2603.17647 | [PDF](https://arxiv.org/pdf/2603.17647v1)

**作者:** Dongqiang Gou `[一作]` (ShanghaiTech University), Xuming He `[通讯]` (ShanghaiTech University)

**通讯引用:** 7402 | [OpenAlex ID](https://openalex.org/A5015970030)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一个两阶段跨模态框架，用于开放词汇下的 3D 交互区域（affordance）定位。第一阶段通过大语言模型生成面向部件的指令来补全文本语义；第二阶段通过 Affordance Prototype Aggregation（APA）与 Intra-Object Relational Modeling（IORM）实现跨物体几何一致性与局部几何细化，从而精确对齐文本意图与 3D 结构。

**💡 创新点**

创新点包括：① 利用 LLM 生成面向部件的结构化指令，提升语义与几何的对齐；② APA 构建共享的 affordance 原型空间，捕获同一功能跨物体的几何一致性；③ IORM 通过邻域注意力增强物体内部几何差异，提升分辨率；④ 结合新的 OpenAfford 基准数据集，提供更丰富的开放词汇和部分视角测试。

**🔧 技术方法**

核心技术：大语言模型（LLM）+ RoBERTa 文本编码；PointNet++ 级联骨干网络提取多尺度点云特征；双向跨模态注意力（CMFM）实现文本-点云互调；IORM 的局部注意力模块；APA 的可学习 prototype 关联；损失函数包括对齐损失、原型关联损失和 focal+Dice 组合的掩码损失。

**📊 数据集**

使用了三个数据集：① 新建的 OpenAfford（23k+点云，1,840 问题）用于开放词汇与部分视角评估；② LASO 数据集（闭集）用于跨数据集验证；③ 3D-AffordanceLLM 数据集（开放词汇）用于零样本实验。

**📈 对比分析**

与 LASO、OpenAD、IAGNet、XMF 等基线在 OpenAfford 的四个评估拆分（Open-set Full-view、Partial-view、Closed-set Seen、Unseen）中均取得显著提升。最高 aIoU 由 13.86（LASO）提升至 18.38（本方法），AUC、SIM 同理；在 LASO 上的 aIoU 与 AUC 与基线持平或略高，MAE 降低；在 3D-AffordanceLLM 上 mIoU、Acc、mAcc 均居首位。

**⚠️ 局限性**

主要限制：方法高度依赖大语言模型进行指令生成，导致推理延迟和偶尔的误解；在未见词汇或部件细节不足时，LLM 可能产生模糊或错误的结构化指令。

---

## 83. Huddle: Parallel Shape Assembly using Decentralized, Minimalistic Robots

**arXiv ID:** 2603.17768 | [PDF](https://arxiv.org/pdf/2603.17768v1)

**作者:** Khai Yi Chin `[一作]` (Amazon Robotics), Yuri Ivanov `[通讯]` (Amazon Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种名为Huddle的并行、去中心化、最小化机器人形状装配算法，利用局部信号引导机器人形成任意无洞形状。

**💡 创新点**

创新点在于：1）仅使用目标形状的外周坐标即可完成装配；2）完全无运动控制或位姿定位；3）通信量极小，仅邻居间传递两个整数；4）通过规则保证所有可达性与无孔性。

**🔧 技术方法**

技术包括：基于六边形格点的坐标系统、邻接壁状态识别、核/常规机器人角色分配、增长方向与信号激活策略、延迟信号以避免不可达孔洞、理论证明与 Monte Carlo 与物理仿真验证。

**📊 数据集**

使用的数据集：40000个随机无洞形状（最多250机器人）进行 Monte Carlo 160,000 次实验；107 机器人在 179 m² 仿真环境中形成“ℋ”字形。

**📈 对比分析**

比较方法：在 Monte Carlo 实验中 Huddle 在 160,000 次尝试中成功率为 100%；在物理仿真中 30 次试验中仅 2 次超时，成功率超过 99%，完成时间与扩展速率与传统基于定位或集中控制的方法相比更具鲁棒性。

**⚠️ 局限性**

局限性：需要手动提供形状外周；对一格宽缺口或长通道的形状易导致信号遮挡；不考虑信号相互干扰；仅适用于六边形或可改为方形的单元，三角形或非正多边形不直接适用；缺乏真实机器人验证与更高效的运动规划。

---

## 84. Stronger core results with multidimensional prices

**arXiv ID:** 2603.17862 | [PDF](https://arxiv.org/pdf/2603.17862v1)

**作者:** Mark Braverman `[一作]` (Princeton University), Chenghan Zhou `[通讯]` (Stanford University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

提出一种新的均衡概念——多维价格的词典型红利均衡（LDE），解决了在无货币且存在禀赋的单边匹配中竞争均衡可能不存在的问题。

**💡 创新点**

创新点在于将商品价格扩展为多维向量，借助词典顺序构造预算约束，保证LDE总能存在且落在拒绝核心（rejective core）中；并证明在经济规模放大时，拒绝核心与LDE收敛，表明LDE是大市场中的唯一可行均衡。

**🔧 技术方法**

主要技术包括：使用多维词典价格与分配矩阵；利用极限与逼近（将ε分配给商品得到1维竞争均衡，再取极限得到LDE）；核心收敛证明采用超平面分离与递归，确保每一步都有新的正价货币；线性规划与强对偶性用于调整价格以满足最强的最小化束缚。

**📊 数据集**

该工作为理论研究，无实验数据集，所有结果均通过严格数学证明得到。

**📈 对比分析**

与传统竞争均衡/红利均衡、弱核心/强核心等概念对比，LDE兼具存在性、个体理性、弱核心稳定性和最小化属性，且在大规模市场中是唯一可拒绝核心内的分配；理论上证明了在足够大复制经济中拒绝核心等价于LDE。

**⚠️ 局限性**

局限性：证明强最小化束缚需要线性效用；扩展到非线性或非凸效用尚未完成；目前的构造不一定只需两种货币，存在未知的最小货币维度；以及对一般单边匹配市场（非单位需求）是否适用仍是开放问题。

---

## 85. Physical Layer Security in Finite Blocklength Massive IoT with Randomly Located Eavesdroppers

**arXiv ID:** 2603.17665 | [PDF](https://arxiv.org/pdf/2603.17665v1)

**作者:** Tijana Devaja `[一作]` (University of Novi Sad), Cedomir Stefanovic `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文分析了在有限块长度（FBL）条件下，采用随机几何模型的稀疏物联网（IoT）上行网络的物理层安全性能，推导了安全成功概率、保密性外流概率和保密吞吐量的解析表达式。

**💡 创新点**

创新点在于：①将FBL信息理论与物理层安全相结合，考虑短包信号的误码概率；②将随机分布的窃听者（均匀分布在设备附近的圆盘内）引入随机几何框架；③在同一模型中同时分析了随机干扰、Nakagami‑m衰落、网络几何随机性与FBL限制的耦合影响。

**🔧 技术方法**

技术手段包括：随机几何（Poisson点过程）建模、Nakagami‑m衰落的Gamma分布、干扰的Laplace变换与KWW形式、FBL的正常逼近（Q函数线性近似）以及Meijer‑G函数的解析推导。

**📊 数据集**

未使用真实数据集，而是通过数值仿真验证解析结果，仿真参数涵盖不同的设备密度、基站密度、信道增益、块长度、编码速率、窃听器半径和Nakagami‑m参数。

**📈 对比分析**

通过对比不同设备/基站密度、信号功率增益、块长度以及窃听器位置的仿真结果，显示：更高的设备密度与基站密度、较低的编码速率、较大的窃听器半径以及更高的基站天线增益均能显著提升安全成功概率和吞吐量，同时降低保密性外流概率。

**⚠️ 局限性**

局限性包括：仅考虑单一窃听者，未讨论多窃听者或协同窃听；假设干扰为高斯噪声近似，可能在极端低密度场景下失效；模型未考虑能量预算、时延限制与多用户接入策略的动态调整；且结果基于理论解析与仿真，缺乏实验室或真实网络验证。

---

## 86. From Virtual Environments to Real-World Trials: Emerging Trends in Autonomous Driving

**arXiv ID:** 2603.17714 | [PDF](https://arxiv.org/pdf/2603.17714v1)

**作者:** A. Humnabadkar `[一作]`, A. Behera `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

综述了自动驾驶领域中合成数据、数字孪生与仿真技术的最新进展，探讨了 Sim2Real / Real2Sim 迁移、视听语言模型（VLM）与多模态学习的应用，并对现有数据集、模拟平台和域适应方法进行了系统对比与评估。

**💡 创新点**

提出了统一的三维视角分类框架，将 VLM 与数字孪生相结合以实现跨域推理与仿真闭环；系统性梳理了 Sim2Real/Real2Sim 迁移中的可扩展性、成本与安全标准，强调了多模态与生成式方法在提升泛化与可解释性方面的潜力。

**🔧 技术方法**

使用深度学习（CNN、GNN、Transformer）、生成式模型（扩散、NeRF）、VLM（CLIP、GPT、Vision‑Language Fusion）、域适应技术（域随机化、特征对齐、知识蒸馏）以及主流仿真平台（CARLA、SVL、NVIDIA DRIVE Sim、rFpro 等）。

**📊 数据集**

涵盖了 KITTI、nuScenes、Waymo Open、Argoverse、BDD100K、Cityscapes、CARLA、Virtual KITTI、SYNTHIA、CLIP2Scene、DriveGPT 等公开与合成数据集。

**📈 对比分析**

通过对比实验评估 VLM 在不同任务中的准确率、推理延迟、域 gap 缩小幅度（如 domain randomization 提升 8‑12%，CARE 提升 18%，轻微微调可闭合 90%），并对模拟器在图像分辨率、传感器噪声、实时帧率等方面的性能进行量化比较，整体显示仿真-真实迁移效果在 5‑10% 的误差范围内。

**⚠️ 局限性**

主要局限包括：VLM 模型推理延迟高（500 ms–2 s），不适合严格实时控制；域适应仍需大量标注或算力；数字孪生同步与数据实时性受限；多模态集成带来计算与存储瓶颈；标准化与安全评估体系尚不完善，导致跨平台迁移与法规合规面临挑战。

---

## 87. End-to-end data-driven prediction of urban airflow and pollutant dispersion

**arXiv ID:** 2603.17606 | [PDF](https://arxiv.org/pdf/2603.17606v1)

**作者:** Nishant Kumar `[一作]` (Institut Pprime, CNRS, Université de Poitiers), Laurent Cordier `[通讯]` (Institut Pprime, CNRS, Université de Poitiers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于SPOD的端到端数据驱动模型，用于预测城市街道峡谷中的湍流气流和污染物扩散。

**💡 创新点**

创新点在于结合SPOD能量与空间相似性阈值进行特征削减，并将时间系数压缩至低维潜在空间，再通过AE、LSTM与CNN三网络实现高效、稳定的时空预测。

**🔧 技术方法**

主要技术包括谱 Proper Orthogonal Decomposition、自动编码器、长短时记忆网络和卷积神经网络。

**📊 数据集**

使用的训练数据集为基于OpenFOAM的三维街道峡谷LES模拟的8万帧时间分辨率快照，包含速度与浓度场。

**📈 对比分析**

与高保真LES基准进行对比，模型在瞬时场和长期统计上误差低于SPOD投影误差，且能够保持动力学不变吸引子，预测误差随时间平稳不爆炸。

**⚠️ 局限性**

局限性在于模型仅针对单一几何与边界条件训练，参数可变性不足；网络训练成本高，且对小尺度细节的预测存在平滑效应。

---

## 88. Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies

**arXiv ID:** 2603.16952 | [PDF](https://arxiv.org/pdf/2603.16952v1)

**作者:** Utkarsh Grover `[一作]` (University of South Florida), Xiaomin Lin `[通讯]` (University of South Florida)

**通讯引用:** 156 | [OpenAlex ID](https://openalex.org/A5101610451)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了在资源受限的边缘平台上部署基础模型的系统挑战与解决方案，提出了 Deployment Gauntlet 框架。

**💡 创新点**

创新点在于将多模态闭环部署瓶颈归纳为八大相互耦合障碍，并归纳出四大缓解技术族。

**🔧 技术方法**

使用系统分析、文献梳理、模型与工作负载分类等技术，对现有模型（如 RT-2、OpenVLA、Octo、LLaVA-Mini 等）进行评价。

**📊 数据集**

以代表性模型和工作负载为实验数据，涵盖视觉-语言-动作、扩散策略、视觉编码器、LiDAR 编码器与多模态融合堆栈。

**📈 对比分析**

通过对比各缓解策略在内存带宽、算力延迟、热耗与安全保障等维度的表现，指出单一压缩技术不足以解决最坏情况，强调协同设计的重要性。

**⚠️ 局限性**

局限在于仅为综述，缺乏统一实验平台与量化指标，且对未来工作需进一步验证提出的架构方向。

---

## 89. Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs

**arXiv ID:** 2603.16932 | [PDF](https://arxiv.org/pdf/2603.16932v1)

**作者:** Nimrod Shabtay `[一作]` (IBM Research), Eli Schwartz `[通讯]` (IBM Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出一种空间按需裁剪框架AwaRes，通过低分辨率全景+工具调用只检索必要的高分辨率子区域，以此实现VLM的精度与效率平衡。

**💡 创新点**

创新点在于：①将“何处看”与“是否看”合并为联合决策策略；②利用自动化标注管线生成多轮裁剪轨迹；③在GRPO中加入显式裁剪成本与误差惩罚，优化精度‑效率权衡。

**🔧 技术方法**

技术包括：LLM判断裁剪需求（LLaMA-3.3-70B）、基于oracle的定位（Qwen3-VL-A235B-A22B）、工具调用接口、SFT + GRPO训练、KV‑cache 多轮推理、句子相似度奖励。

**📊 数据集**

使用10k样本来自ChartQA、DocVQA、TextVQA、LLaVA-Multi、VisionThink-Smart进行训练；评估覆盖六个基准：ChartQA、DocVQA、OCRBench、POPE、RealWorldQA、V^*-Bench。

**📈 对比分析**

相较于固定令牌削减方法（VisionZIP、SparseVLM、Holo-V）和自适应分辨率升级方法（VisionThink），AwaRes平均准确率与全分辨率模型持平（≈80.3%），且保留视觉令牌仅36%，在RT‑R降低至0.36，整体推理时延比VisionThink低约7×。

**⚠️ 局限性**

局限包括：裁剪集离散，无法实现连续框选；自动标注依赖LLM判断，可能引入标签噪声；在极端细节需求时仍需多次裁剪；未扩展到视频等时序任务。

---

## 90. Data Obfuscation for Secure Use of Classical Values in Quantum Computation

**arXiv ID:** 2603.17725 | [PDF](https://arxiv.org/pdf/2603.17725v1)

**作者:** Amal Raj `[一作]` (Singapore Institute of Technology), Vivek Balachandran `[通讯]` (Singapore Institute of Technology)

**通讯引用:** 383 | [OpenAlex ID](https://openalex.org/A5021323754)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种将经典整数通过分解成多项的量子表示，并利用Grover搜索对满足等式的解进行放大，从而在量子计算过程中对输入数据实现混淆与隐藏。

**💡 创新点**

首次将数据混淆（obfuscation）概念引入量子计算，设计了在量子寄存器中分布式编码整数的方案，并通过可逆加法与幅度放大实现安全的数据隐藏。

**🔧 技术方法**

使用可逆量子加法（Cuccaro ripple‑carry adder）、多重受控比较、Grover幅度放大以及多重受控相位门等量子门；整体架构基于Qiskit实现。

**📊 数据集**

对一系列以 N = 2^k − 1 为目标的整数（k = 3…8，即 N = 7, 15, 31, 63, 127, 255）进行实验，评估不同 n 位宽下的量子资源消耗。

**📈 对比分析**

通过模拟在 AerSimulator 上测量门计数、深度、时长和解的个数，并与不同 N 取值的结果对比。结果显示：随着 N 增大，门数和深度呈指数增长，模拟时间从 0.3 秒升至 345 秒；但在小规模实例下仍保持可接受的资源消耗。

**⚠️ 局限性**

主要局限在资源需求大：门数、深度和总 qubit 数随 N 迅速增长，当前 NISQ 设备难以实现；仅支持三项分解，未考虑更复杂的混淆形式；缺乏对噪声环境下的鲁棒性评估。

---

## 91. Attention Guidance through Video Script: A Case Study of Object Focusing on 360° VR Video Tours

**arXiv ID:** 2603.16875 | [PDF](https://arxiv.org/pdf/2603.16875v1)

**作者:** Paulo Vitor Santana Silva `[一作]`, Arlindo Rodrigues Galvão Filho `[通讯]` (Federal University of Goiás)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种利用视频脚本、Grounding Dino与Segment Anything Model（SAM）在360º VR视频中自动定位目标并通过暗角效果引导用户注意力的方法。

**💡 创新点**

创新点在于将语言引导的目标检测与全景图像分割相结合，实现无干扰、无界面化的注意力引导，首次在VR导览场景中验证该技术。

**🔧 技术方法**

使用的技术包括：Grounding Dino（对象检测）、SAM（对象分割）以及基于分割掩码生成暗角效果的视觉处理管道。

**📊 数据集**

使用的数据集包括：预训练时的COCO、O365、OpenImage等公开检测/分割数据集，以及Reading大学校园的360º VR导览视频与手工编写的脚本。

**📈 对比分析**

通过在两种场景（博物馆雕塑与咖啡休息区）中对比原始帧、检测框、分割掩码与暗角效果，展示了目标检测与分割的大多数成功案例；唯一失误为对半人半兽雕像的分割失败。

**⚠️ 局限性**

局限性包括：SAM在零样本条件下对复杂或罕见对象的分割精度不足；缺乏用户实验验证暗角对注意力的实际影响；模型未针对特定VR场景微调，导致部分目标未能成功分割。

---

## 92. Adaptive Contracts for Cost-Effective AI Delegation

**arXiv ID:** 2603.17212 | [PDF](https://arxiv.org/pdf/2603.17212v1)

**作者:** Eden Saig `[一作]` (California Institute of Technology), Jamie Tucker-Foltz `[通讯]` (Yale School of Management)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并研究了适用于人工智能代理任务的自适应合同框架，允许在初步粗略评估后根据需要决定是否进行更精细、昂贵的评估，以节省资源并提升效率。

**💡 创新点**

创新点包括：①在经济模型中首次系统性引入自适应评估与合同设计的耦合；②在满足MLRP或ISOP等自然假设时给出多项多项式时间算法；③证明在无结构或无独立性情况下优化自适应合同为NP‑hard且难以逼近；④对随机化检查进行理论分析，揭示其在保证可行性和支付上可能导致无界支付的问题，并提出两种实际约束恢复均衡。

**🔧 技术方法**

技术方法涵盖：合同理论中的线性/混合整数编程（QCQP、QCLP），凸/非凸优化，稀疏性与可搜索空间约简；复杂度分析采用独立集、集合覆盖等经典归约；对随机化检查的理论讨论利用Stackelberg博弈和支付限制。

**📊 数据集**

实验使用两大公开数据集：①AlpacaEval 2.0（问答评估），②SWE‑Bench（代码生成与单元测试）。

**📈 对比分析**

实验将自适应合同与传统非自适应合同进行对比，结果显示：在问答任务中，自适应合同相较于最佳非自适应基线提升约14%（预期主方效用从1.00提升）；在代码生成任务中，自适应合同通过调整检查策略实现了不同成本下的最优支付，并展示了最佳初始与精细测试组合的热图。

**⚠️ 局限性**

局限性包括：①整体优化在维度不受限或评估不独立时为NP‑hard；②随机化检查在理论上可能导致无限大支付，除非施加额外约束；③目前仅考虑单轮检查（两阶段）且未探讨多轮或预算约束的情形；④依赖MLRP、ISOP等假设，在这些假设失效时算法与性能不一定保持。

---

## 93. Quadratic Surrogate Attractor for Particle Swarm Optimization

**arXiv ID:** 2603.17163 | [PDF](https://arxiv.org/pdf/2603.17163v1)

**作者:** Maurizio Clemente `[一作]` (Center for Automotive Research), Marcello Canova `[通讯]` (Center for Automotive Research)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种将粒子群优化中的全局最优解替换为基于多点拟合得到的二次拟合模型最小值的算法，形成新的动态吸引器；

**💡 创新点**

通过利用多点最优解构造二次拟合模型并解析求得其最小值，提供了比传统全局最优更为条件良好的吸引目标，从而提升收敛鲁棒性；

**🔧 技术方法**

采用粒子群优化（PSO）、二次拟合插值、堆结构筛选最优点、解析求解二次模型最小值、统计分析等技术；

**📊 数据集**

在Ackley、Griewank、Sphere、Flower等二维和三维经典基准函数上进行实验；

**📈 对比分析**

对比标准PSO，采用400次独立运行、200次迭代的统计评估，结果显示在所有函数上二次拟合吸引器均实现了更低的平均误差、较小的IQR，尤其在准凸函数上提升显著，计算时间略有增加；

**⚠️ 局限性**

需满足足够粒子数以构造二次模型（N_p≥N_Q），对边界最优解处理不足，未对约束问题做严格处理，仅在合成基准上验证，缺乏真实工程案例验证。

---

## 94. Material Magic Wand: Material-Aware Grouping of 3D Parts in Untextured Meshes

**arXiv ID:** 2603.17370 | [PDF](https://arxiv.org/pdf/2603.17370v1)

**作者:** Umangi Jain `[一作]` (University of Toronto), Zhiqin Chen `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了材料魔法棒工具，实现基于单个代表部件点击快速聚类同材质的部件。

**💡 创新点**

创新点在于构建材料感知嵌入空间，利用多视图渲染和监督对比学习自动检索同材质部件。

**🔧 技术方法**

使用 DINO‑v3 视觉基础模型作为编码器，对三种渲染视图进行特征提取，并通过监督对比损失训练。

**📊 数据集**

基于 Objaverse 22,000 个网格的材质标注数据进行训练，并创建 100 个形状的手工标注评测基准。

**📈 对比分析**

与几种几何、视觉基础模型和 PartField 基线对比，平均 AUC‑PR 提升约 9 个百分点，F1 提升约 17 个百分点，性能显著优于基线。

**⚠️ 局限性**

局限包括对遮挡严重或自遮挡部件渲染视图效果差，且对多重合法分组未建模。

---

## 95. Revisiting Cross-Attention Mechanisms: Leveraging Beneficial Noise for Domain-Adaptive Learning

**arXiv ID:** 2603.17474 | [PDF](https://arxiv.org/pdf/2603.17474v1)

**作者:** Zelin Zang `[一作]` (Westlake University), Baigui Sun `[通讯]` (Alibaba Group)

**通讯引用:** 1357 | [OpenAlex ID](https://openalex.org/A5087131650)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种基于 Transformer 的域自适应学习框架 DACSM，结合 Domain‑Adaptive Transformer 与 Cross‑Scale Matching 两大模块，解决域间风格与尺度差异问题。

**💡 创新点**

创新点包括：① 将跨注意力重新解释为域翻译机制并注入“有益噪声”实现内容与风格的显式分离；② 设计跨尺度匹配模块以显式处理尺度差距；③ 在单一框架中同时攻克风格与尺度两大域间缺口。

**🔧 技术方法**

使用了 Vision Transformer（DeiT‑base）骨干、跨域交叉注意力、噪声注入、子中心分类器、伪标签、KL 蒸馏、对比与样式损失等技术。

**📊 数据集**

在标准 UDA 基准数据集 VisDA‑2017、Office‑Home 与 DomainNet 上进行实验。

**📈 对比分析**

与 CDTrans、DOT‑B 等最新方法对比，DACSM 在 VisDA‑2017 上提升 2.3% 绝对精度（“truck” 类提升 5.9%），Office‑Home 81.3%，DomainNet 46.3%，均超过现有 SOTA；训练略慢但推理无额外开销。

**⚠️ 局限性**

局限性：训练阶段需要额外的跨尺度匹配与噪声注入，导致训练时间略增；对极端尺度或域差异的鲁棒性尚未充分验证；噪声参数需手动调优；实验仅覆盖图像分类任务，未扩展到其他领域。

---

## 96. DDH-based schemes for multi-party Function Secret Sharing

**arXiv ID:** 2603.17453 | [PDF](https://arxiv.org/pdf/2603.17453v1)

**作者:** Marc Damie `[一作]` (University of Twente), Jan Ramon `[通讯]` (Inria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

提出了一种基于DDH的多方函数秘密共享（FSS）方案，能够高效分享点函数（DPF）和比较函数（DCF），并显著压缩密钥大小；

**💡 创新点**

创新点在于利用DDH假设对已有信息论或PRG基础的多方DPF/DCF进行改造，使得多方密钥大小从原来的O(√N)×指数/多项式因子降低到仅O(√N)，并在保持半诚实安全模型下实现诚实多数；

**🔧 技术方法**

核心技术包括DDH群编码、在椭圆曲线P-256上实现的密钥生成与评估、对已有信息论DPF/DCF的子协议嵌入以及对密钥的压缩（PRSS）；

**📊 数据集**

实验数据集为不同域大小（10^2–10^10）、不同参与方数（3–10）以及不同模数（二进制、素数、合成数、素数链）下的基准测试，使用P-256椭圆曲线；

**📈 对比分析**

与现有PRG‑基、信息论以及基于新硬假设的方案对比，实验显示在实际规模下密钥大小可比最优方案缩小约10倍；同时在不同域/参与方数场景下保持比传统方案更小；

**⚠️ 局限性**

局限性包括：仍然是O(√N)的上界，无法实现对数级别的密钥；对DDH群的依赖导致编码与解码需使用椭圆曲线，且在大范围秘密时需要离散对数求解；压缩方案对共享可压缩性有一定限制，且仅在半诚实模型下安全。

---

## 97. SLAM Adversarial Lab: An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions

**arXiv ID:** 2603.17165 | [PDF](https://arxiv.org/pdf/2603.17165v1)

**作者:** Mohamed Hefny `[一作]` (Simon Fraser University), Steven Y. Ko `[通讯]` (Simon Fraser University)

**通讯引用:** 1736 | [OpenAlex ID](https://openalex.org/A5109113161)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出SAL框架，用于在对抗环境（雾、雨、光照、相机失效、网络压缩、帧丢失等）下评估视觉SLAM系统。

**💡 创新点**

创新点在于：① 用可解释的实用单位（如可见度m、雨强mm/h）参数化扰动并支持深度感知；② 采用模块化、可插拔的架构，使数据集、扰动、SLAM算法三者独立扩展；③ 引入二分搜索寻找失效边界，显著提高评估效率。

**🔧 技术方法**

技术手段包括：物理渲染雨雾、光照变换、相机失效（污渍、裂纹、运动模糊）、网络压缩与帧丢失；深度估计与重建、特征跟踪诊断、ATE/RPE评估及二分搜索算法。

**📊 数据集**

使用KITTI、TUM、EuRoC三大公开数据集，涵盖室外单目/双目、室内RGB-D等多种传感器配置。

**📈 对比分析**

比较方法：对七个SLAM算法（ORB‑SLAM3、S3PO‑GS、GigaSLAM、MASt3R‑SLAM、DROID‑SLAM、VGGT‑SLAM、Photo‑SLAM）进行轨迹误差（ATE/RPE）和特征跟踪统计；利用鲁棒性边界搜索定位失效阈值；结果表明不同算法对各类扰动的敏感度差异显著，边界搜索比线性扫描快数倍。

**⚠️ 局限性**

局限性：仅考虑纯前向运动模糊，缺乏旋转/动态物体的综合评估；模块实现依赖外部工具/预训练模型，未涉及实时性能和能耗评估。

---

## 98. Generative Control as Optimization: Time Unconditional Flow Matching for Adaptive and Robust Robotic Control

**arXiv ID:** 2603.17834 | [PDF](https://arxiv.org/pdf/2603.17834v1)

**作者:** Zunzhe Zhang `[一作]` (Institute for Interdisciplinary Information Sciences), Hang Zhao `[通讯]` (Institute for Interdisciplinary Information Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 GeCO 框架，将传统基于时间的流匹配控制改为时间无关的优化过程，允许自适应推理并提供零训练的安全信号。

**💡 创新点**

创新点在于：①学习时间不变的速度场作为优化目标，实现自适应计算与即时终止；②利用该场的范数作为训练‑free OOD 检测器；③将其作为插件式头直接替换现有 VLA 的流匹配头。

**🔧 技术方法**

使用技术包括：流匹配与扩散模型、时间无关速度场、梯度下降优化、速度缩放机制、VLM 与多模态融合、以及 OOD 监测。

**📊 数据集**

实验数据集涵盖 LIBERO（Goal、Spatial、Object、Long）模拟任务、VLABench、RoboTwin 2.0 机器人操控 benchmark，以及 Galaxea R1 Lite 的真实物理测试。

**📈 对比分析**

与固定步长的扩散/流匹配基线相比，GeCO 在相同训练资源下取得更高成功率、降低平均函数评估次数；在 VLA 模型和真实机器人任务中表现出更优的效率‑性能权衡，成功率提升 10–20% 并显著减少计算量。

**⚠️ 局限性**

局限性：理论分析仍不充分，缺乏对优化收敛速度和稳定性的严格保证；在极端动态或高维场景下，速度场可能收敛缓慢或失稳，且 OOD 检测对阈值敏感。

---

## 99. REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge

**arXiv ID:** 2603.17145 | [PDF](https://arxiv.org/pdf/2603.17145v1)

**作者:** Yasi Zhang `[一作]` (University of California, Los Angeles), Michal Lukasik `[通讯]` (Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对LLM-as-a-Judge的回归感知强化学习框架REAL。

**💡 创新点**

创新点在于将回归目标直接融入RL奖励，并利用泛化策略梯度将优化拆解为CoT探索和数值预测两部分。

**🔧 技术方法**

采用泛化策略梯度、RLOO方差降低、RAIL预测器以及回归损失+对数似然混合奖励等技术。

**📊 数据集**

使用了Feedback Collection（约10万点评估样本）及其四个评测基准（Feedback Bench、FLASK、Vicuna Bench、MT Bench）。

**📈 对比分析**

与零样本、SFT（RAFT/TRACT）以及标准二元奖励RL相比，REAL在所有模型规模上均提升了Pearson/Spearman/Kendall相关性，尤其在离域数据上表现显著优于基线。

**⚠️ 局限性**

局限在于仅适用于点评估，未处理成对偏好学习，并且依赖LLM自身生成的CoT，可能携带偏差。

---

## 100. A spatio-temporal graph-based model for team sports analysis

**arXiv ID:** 2603.17471 | [PDF](https://arxiv.org/pdf/2603.17471v1)

**作者:** Camille Grange `[一作]` (Université Sorbonne Paris Nord), Ludovic Seifert `[通讯]` (Université de Rouen Normandie)

**通讯引用:** 7011 | [OpenAlex ID](https://openalex.org/A5000409877)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种新的时空图模型，用于描述并分析团队运动（以短板橄榄球为例）中的进攻行为，验证防守布局与教练教学方法对球队协调模式的影响。

**💡 创新点**

创新点在于将绝对和相对空间信息与时间、语义信息统一映射到一个通用的骨干图上，通过可标记路径捕捉进攻过程，并设计局部与全局特征（路径长度、子图密度、最大右移、交叉排名、踢球次数）来量化协调模式。

**🔧 技术方法**

采用图论与时空信息融合的技术，构建骨干图、标记路径、计算特征，并使用非参数统计检验（Kruskal–Wallis、Dunn、Chi-square）对实验数据进行分析。

**📊 数据集**

使用14支15-21岁青少年橄榄球队的数据，共计356次进攻（12次预试+12次干预+20次后测），涉及三种防守场景（紧凑、开放、踢球）与两种教学法（线性vs非线性）。

**📈 对比分析**

通过比较不同条件下的特征值，发现开放防守显著降低交叉排名、紧凑/踢球防守提升最大右移；非线性教学法导致路径长度和子图密度显著增加，表明协调模式更具多样性，效果在干预期显著，在后测阶段部分退化。

**⚠️ 局限性**

局限性包括样本量小、实验条件生态性导致噪声大、教练执行差异可能影响教学法实施、踢球事件稀缺导致相关特征统计无力、模型仅关注进攻方信息，未考虑对手动态。

---

## 101. Efficient Exploration at Scale

**arXiv ID:** 2603.17378 | [PDF](https://arxiv.org/pdf/2603.17378v1)

**作者:** Seyed Mohammad Asghari `[一作]` (Google DeepMind), Benjamin Van Roy `[通讯]` (Google DeepMind)

**通讯引用:** 10704 | [OpenAlex ID](https://openalex.org/A5045543562)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种结合在线学习、奖励模型不确定性网络和信息导向探索的RLHF算法，以显著提升数据效率。

**💡 创新点**

创新点包括：①引入“affirmative nudge”避免RLHF训练过程中的崩溃；②构建epistemic neural network（ENN）奖励模型，利用其不确定性实现信息最大化采样；③将信息导向探索与在线RLHF结合，获得10×–1000×的数据效率提升。

**🔧 技术方法**

使用的技术包括：在线RLHF（奖励模型+策略更新）、线性/MLP奖励头、100个ensemble MLP（prior与differential）用于不确定性建模、Bradley‑Terry 模型生成偏好概率、top‑K 采样、指数移动平均作为参数锚点、AdamW 优化器、Gemma 9B 语言模型、Gemini 1.5 Pro 作为仿真人类反馈源。

**📊 数据集**

数据集：202 K 个多主题提示（写作、编程、摘要、阅读、数学、科学等），其中 200 K 用于训练，1 K 用于验证，1 K 用于外样本评估；仿真人类反馈基于 Gemini 1.5 Pro 训练的奖励模型。

**📈 对比分析**

对比方法：offline RLHF、periodic RLHF、online RLHF 与信息导向探索；评价指标为基线策略的 win‑rate。实验显示信息导向探索在仅 20 K 选择下可匹配 offline RLHF 的 200 K 选择性能，且在 1 M 选择下预计可实现 1000×的数据效率提升；示例展示了更简洁、正确的回答。

**⚠️ 局限性**

局限性：①使用仿真人类反馈，可能与真实人类偏好不完全一致；②仅评估单轮响应，未涵盖多轮对话或复杂任务；③探索算法仍有改进空间；④实验规模虽大，但在更大模型或更真实数据上仍需验证。

---

## 102. CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution

**arXiv ID:** 2603.17775 | [PDF](https://arxiv.org/pdf/2603.17775v1)

**作者:** Teng Pan `[一作]` (Zhejiang University), Yongliang Shen `[通讯]` (Zhejiang University)

**通讯引用:** 1557 | [OpenAlex ID](https://openalex.org/A5004615610)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出CoVerRL框架，让同一模型在生成与验证两种角色之间循环，进行无标签的强化学习。

**💡 创新点**

创新点在于把投票一致性与自我验证双向互助，使生成器得到更干净的伪标签，验证器通过对比学习不断提升，形成共进的自我纠错机制。

**🔧 技术方法**

采用多轮生成-验证交互、对比式验证器训练、答案锚定的GRPO优化以及自我纠错的多样本采样等技术。

**📊 数据集**

在四个数学推理基准（如 GSM8K、AQuA、AddSub、MultiArith）上，对 Qwen 与 Llama 系列模型进行实验。

**📈 对比分析**

与传统的TTRL相比，CoVerRL在生成精度上提升了约5.7%–5.9%，验证精度从55%提升至80%+，奖励准确率始终保持90%以上，显著优于纯投票或单独验证的做法。

**⚠️ 局限性**

局限性包括仍依赖多数投票对初始伪标签的粗糙性，极难在极其复杂或完全无结构的任务中自我纠错，且对计算资源的需求较高。

---

## 103. Coded Information Retrieval for Block-Structured DNA-Based Data Storage

**arXiv ID:** 2603.17154 | [PDF](https://arxiv.org/pdf/2603.17154v1)

**作者:** Daniella Bar-Lev `[一作]` `[通讯]` (Universität Zürich), Daniella Bar-Lev (Universität Zürich)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究DNA存储中块结构文件检索的期望检索时间，建立上界与下界，分析几类线性码，并探讨极限可达区域。

**💡 创新点**

提出块结构检索模型并证明文件专用MDS码在该框架下的最优性，给出超越线性切片的非线性超几何约束，并提出并验证普适超几何边界猜想。

**🔧 技术方法**

组合计数、子集计数、互斥恢复集、几何列投影、概率论中的优惠收集问题、对称与不对称MDS码比较、极限分析等。

**📊 数据集**

无实际实验数据，所有结果为理论推导与随机生成码的数值验证。

**📈 对比分析**

通过期望检索时间对码进行比较，证明文件专用MDS优于全局系统MDS，局部MDS能接近理论最优超几何边界；数值实验验证猜想无违例。

**⚠️ 局限性**

超几何边界的普适性猜想未能完全证明，混合列的联合效应难以解析；对于k=2且文件一维时猜想失效，且非对称情况仍有未解决的细节。

---

## 104. Learning Permutation Distributions via Reflected Diffusion on Ranks

**arXiv ID:** 2603.17353 | [PDF](https://arxiv.org/pdf/2603.17353v1)

**作者:** Sizhuang He `[一作]` (Yale University), David van Dijk `[通讯]` (Yale University)

**通讯引用:** 10805 | [OpenAlex ID](https://openalex.org/A5019679682)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Soft-Rank Diffusion框架，在软秩连续空间中进行反射扩散，并设计了上下文化Generalized Plackett–Luce（cGPL）逆向模型，以生成和学习排列分布。

**💡 创新点**

创新点包括：①将排列映射到连续软秩表示，借助反射扩散桥实现平滑前向过程；②提出cGPL及其Pointer变体，使逆向步骤能够根据已生成的前缀动态调整，显著提升表达能力；③通过软秩空间的连续更新与排序投影，兼顾可微分性与离散排列的完整性。

**🔧 技术方法**

主要技术包括：反射扩散桥、软秩（soft‑rank）连续表示、cGPL和Pointer‑cGPL（上下文化的GPL）、Transformer编码器‑解码器、可微排序投影、马尔可夫链训练与变分下界优化。

**📊 数据集**

实验数据集：4位数MNIST排序任务（N=9~200），以及2D点集合的旅行商问题（TSP‑20、TSP‑50）。

**📈 对比分析**

与DiffSort、Error‑free DiffSort、SymmetricDiffusers等基线比较；在所有评测指标（Kendall‑Tau、精确匹配准确率、正确率、旅行商距离与最优性缺口）上，Soft‑Rank Diffusion均取得显著提升，尤其在长序列（N≥150）和较大TSP（n=50）中性能差距进一步扩大。

**⚠️ 局限性**

局限性：①对极大规模排列（N>>200）仍面临计算与内存瓶颈；②需要多步采样，导致生成时间增加；③对超参数（如扩散步数、反射项强度）敏感；④在某些任务中，软秩投影误差可能导致最终排列质量下降。

---

## 105. Astrolabe: Steering Forward-Process Reinforcement Learning for Distilled Autoregressive Video Models

**arXiv ID:** 2603.17051 | [PDF](https://arxiv.org/pdf/2603.17051v1)

**作者:** Songchun Zhang `[一作]` (Hong Kong University of Science and Technology), Anyi Rao `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 5043 | [OpenAlex ID](https://openalex.org/A5067715162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出Astrolabe框架，利用在线强化学习在已蒸馏的自回归视频模型上进行人类偏好对齐；

**💡 创新点**

创新点包括：①前向过程RL的无轨迹对齐策略；②滚动KV缓存+分段训练实现长视频常量内存；③多奖励与不确定性感知的KL正则化防止奖励作弊；

**🔧 技术方法**

使用前向过程RL、滚动KV缓存、组内并行采样、LoRA参数高效微调、EMA参考更新、视频质量与运动一致性评估指标（HPSv3、VideoAlign）等技术；

**📊 数据集**

数据集包括VidProM、MovieGenBench、VBench、VBench-Long等；

**📈 对比分析**

在Self-Forcing、Causal-Forcing、LongLive等基线模型上进行对比，实验显示Astrolabe显著提升HPSv3、Motion Quality、CLIP对齐分数，同时保持原始推理速度；

**⚠️ 局限性**

局限性包括：对极长视频的全局一致性仍可能出现漂移；奖励设计对外部评估模型的依赖较大；在某些极端场景下仍可能出现运动不连贯或视觉细节失真。

---

## 106. Privacy and Safety Experiences and Concerns of U.S. Women Using Generative AI for Seeking Sexual and Reproductive Health Information

**arXiv ID:** 2603.16918 | [PDF](https://arxiv.org/pdf/2603.16918v1)

**作者:** Ina Kaleva `[一作]` (King's College London), Jose Such `[通讯]` (INGENIO)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对18名美国女性的半结构化访谈，研究她们在使用生成式AI聊天机器人（如ChatGPT、Gemini、Copilot）获取性与生殖健康信息时的隐私与安全体验与关注。

**💡 创新点**

首次系统性地把握SRH背景下GenAI聊天机器人用户的隐私与安全风险，并基于用户视角提出了设计与政策层面的改进建议。

**🔧 技术方法**

主要技术是生成式AI聊天机器人和定性研究方法（半结构化访谈、主题分析）。

**📊 数据集**

使用的数据是18名受访者的访谈记录与自我报告的使用数据，未使用公开数据集。

**📈 对比分析**

采用Braun & Clarke的主题分析对访谈文本进行编码和主题提炼，未进行定量性能比较，研究侧重于用户体验与风险识别。

**⚠️ 局限性**

局限性包括样本规模有限、仅聚焦使用者视角、受访者可能存在社会期望偏差，以及对快速变化的法律环境关注不足。

---

## 107. VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection

**arXiv ID:** 2603.17470 | [PDF](https://arxiv.org/pdf/2603.17470v1)

**作者:** Chupeng Liu `[一作]` (University of Sydney), Weidong Cai `[通讯]` (University of Sydney)

**通讯引用:** 11959 | [OpenAlex ID](https://openalex.org/A5076697411)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了一种基于视觉参照的概率提示学习框架VirPro，用来增强弱监督单目3D检测的语义上下文和场景感知；

**💡 创新点**

创新点在于将多模态提示学习转化为多高斯分布，并通过自适应提示银行与视觉注入生成实例条件的概率提示，从而捕捉跨场景的视觉不确定性；

**🔧 技术方法**

采用CLIP文本编码器、Multi-Gaussian Prompt Modeling、RoI对比学习、Dual-to-One Distillation、交叉注意力融合以及KL正则化等技术实现；

**📊 数据集**

实验使用KITTI数据集（包含KITTI RAW无标签图像、F-PointNet 2D检测框）进行训练和评估；

**📈 对比分析**

与WeakM3D、GGA+PGD等弱监督基线比较，VirPro+GGA+PGD在KITTI Car类别上平均AP提升约4.8%，在easy/mod/hard场景均表现优异；

**⚠️ 局限性**

局限性包括过度依赖2D检测框的质量、ROI特征受限于固定框和分辨率，视觉提示分布受噪声影响，缺乏更灵活的ROI建模方案。

---

## 108. Allocating Chores with Restricted Additive Costs: Achieving EFX, MMS, and Efficiency Simultaneously

**arXiv ID:** 2603.17270 | [PDF](https://arxiv.org/pdf/2603.17270v1)

**作者:** Zehan Lin `[一作]` (University of Macau), Shengwei Zhou `[通讯]` (Nanyang Technological University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究在受限可加成本（restricted additive costs）下，公平分配不可分割的杂务（chores）的问题，提出一种算法可同时产生EFX、MMS以及对社会成本的2-近似最优解。

**💡 创新点**

创新点在于：①首次证明在受限设置下可同时满足EFX与MMS；②给出实现社会成本2-近似的算法，并证明该近似比率是最优的；③给出多项式时间版本，得到EFX+4/3-MMS分配；④提出并证明了EFX的价格公平（price of fairness）为2和公平成本（cost of fairness）上限为1/2。

**🔧 技术方法**

核心技术包括：①将物品划分为统一成本集M⁺和零成本集M⁰，先对M⁺构造MMS分区并通过迭代重分配保证EFX；②利用匹配与“最后修改”记录的机制为M⁰分配物品；③采用潜能函数证明收敛；④多项式版本使用LPT（最长处理时间优先）构造初始分区；⑤分析价格公平和公平成本的理论上界。

**📊 数据集**

论文为理论工作，无实验数据集；所有结果均通过严格的证明与归纳得到。

**📈 对比分析**

相较于之前的工作，本文实现了在受限设置下首次同时满足EFX、MMS与社会成本2-近似；多项式版本实现EFX+4/3-MMS；并给出了与最优社群成本、EFX公平性相关的极限下界，表明算法性能与理论上限完全匹配。

**⚠️ 局限性**

局限性包括：①整体算法（尤其是Phase 1）不保证多项式时间；②对MMS分区的NP‑难度限制了实践可行性；③仅在受限设置下证明存在EFX，无法推广到更一般的bi‑valued或任意成本模型；④在受限设置中EFX与Pareto最优不兼容，无法同时保证这两者。

---

## 109. Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models

**arXiv ID:** 2603.17677 | [PDF](https://arxiv.org/pdf/2603.17677v1)

**作者:** Jaemin Kim `[一作]` (Korea Advanced Institute of Science and Technology), Jong Chul Ye `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 17519 | [OpenAlex ID](https://openalex.org/A5012644755)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Adaptive Retrieval-Augmented Masked Diffusion (ARAM)框架，利用自适应的Classifier-Free Guidance解决Masked Diffusion Models（MDM）中检索与模型内部知识冲突问题；

**💡 创新点**

创新点在于基于信息论的Signal-to-Noise Ratio（SNR）动态调节CFG指导尺度，实现token级与step级的自适应权重，以在检索内容可靠时强化指导、在噪声或冲突时抑制指导；

**🔧 技术方法**

采用Masked Diffusion Models、Classifier-Free Guidance、信息论SNR分析、条件熵作为噪声代理、逐步自适应lambda计算与logit插值等技术；

**📊 数据集**

使用多种知识密集型问答数据集：NQ、TriviaQA、MARCO QA、HotpotQA、T-REx；

**📈 对比分析**

与标准RAG、CAD、AdaCAD、COIECD、SPREAD、A‑CFG等基线在EM/F1上进行对比，ARAM在大多数基准上实现显著提升，尤其在MDM基础上提升幅度更大；

**⚠️ 局限性**

局限性包括：仅在英文知识性QA任务上验证，未测试开放式生成或其他解码策略；计算量大，推理延迟提升；缺乏对更大模型或商业部署的评估。

---

## 110. Upward Book Embeddings of Partitioned Digraphs

**arXiv ID:** 2603.17128 | [PDF](https://arxiv.org/pdf/2603.17128v1)

**作者:** Giordano Da Lozzo `[一作]` (Roma Tre University), Ignaz Rutter `[通讯]` (University of Passau)

**通讯引用:** 1359 | [OpenAlex ID](https://openalex.org/A5036986694)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文针对分区有向图（Partitioned Digraph）在两页（k=2）的向上书嵌入（Upward Book Embedding）问题进行研究。作者首先给出了该问题的完整判定判据（4‑模态性和不存在不可行面），随后证明该问题在 k=2 时是 NP‑完整的，并在图的平面嵌入已知时提供了一个 O(n log³n) 的判定算法；最后针对双连通的有向部分 2‑树（biconnected directed partial 2‑tree）给出了一种 O(n³) 的动态规划算法。

**💡 创新点**

创新点包括：①首次完成了 k=2 的复杂度分类，填补了之前只知 k=1 为线性可解、k≥3 为 NP‑完整的空白；②提出了“良好嵌入（good embedding）”的概念，并给出其必要与充分条件；③在已知平面嵌入的情形下改造流网络实现快速判定；④利用 SPQ(R)-树和描述符对（descriptor pair）设计了对部分 2‑树的三次方时间解法。

**🔧 技术方法**

主要技术手段包括：
- 图的 4‑模态性和不可行面的理论判据。
- 基于角度分配的向上嵌入与可行流的等价关系。
- 对流网络的增量修改（𝒩‑modifier）以强制 4‑模态性和消除不可行面。
- SPQ(R)-树分解与动态规划，配合描述符对存储可行嵌入信息。
- 复杂度分析与多项式时间证明。

**📊 数据集**

本工作为理论算法研究，不使用具体数据集；所有结果均在理论模型上给出，并通过构造性证明实现算法。

**📈 对比分析**

性能方面：
- 对于给定平面嵌入的任意有向图，判定算法时间为 O(n log³n)。
- 对于双连通的有向部分 2‑树，判定算法时间为 O(n³)。
- 与之前仅在 k≥3 或 k=1 的已知结果相比，本文提供了最优（即 NP‑完整）的复杂度下的判定方案，满足了实际应用对两页嵌入的需求。

**⚠️ 局限性**

限制与未来工作：
- 问题在 k=2 时仍然是 NP‑完整，意味着没有多项式时间解，除非 P=NP；
- 目前仅对部分 2‑树给出了多项式算法，对更一般的图类（如树宽大于 2 的图）仍无高效解法；
- 流网络改造与 SPQ(R)-树处理的实现较为复杂，实际工程中的实现成本高；
- 论文未包含实验评估或实现细节，实际性能需进一步验证。

---

## 111. Variational Kernel Design for Internal Noise: Gaussian Chaos Noise, Representation Compatibility, and Reliable Deep Learning

**arXiv ID:** 2603.17365 | [PDF](https://arxiv.org/pdf/2603.17365v1)

**作者:** Ziran Liu `[一作]` `[通讯]` (Shanghai Institute for Mathematics and Interdisciplinary Sciences), Ziran Liu (Shanghai Institute for Mathematics and Interdisciplinary Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Variational Kernel Design（VKD）框架，并通过最大熵原理推导出基于 Dirichlet 绿核的 Gaussian Chaos Noise（GCh），在内部噪声设计与表示兼容性方面进行理论分析与实验验证。

**💡 创新点**

创新点在于：
1) 将内部噪声视为可设计的机制，拆分为法律族、协方差核、注入算子三部分；
2) 通过二次最大熵问题得到唯一的高斯自由场，并得到 Dirichlet 绿核作为必然的空间协方差；
3) 采用 Wick 正则化得到正值、均值为一的乘法门，形成“精确”噪声；
4) 对比硬二值掩码，给出对正值、空间一致性特征的兼容性定理（pairwise log‑ratio、排名稳定性、内在粗糙度等）。

**🔧 技术方法**

使用的技术包括：最大熵变分推导、离散高斯自由场（GFF）、Wick 正则化、离散正弦变换（DST）快速采样、卷积网络与 Swin‑T 变换器训练、标准评估指标（Top‑1、NLL、ECE）以及 ImageNet‑C 失真评测。

**📊 数据集**

使用的数据集有：ImageNet（训练/验证）、ImageNet‑C（七种人工失真）、Oxford‑IIIT Pets（细粒度任务）以及 Swin‑T 的官方数据集。

**📈 对比分析**

与 Dropout、DropBlock、i.i.d. 与相关加性高斯噪声在相同能量下进行对比。实验结果表明：
- 在 ImageNet 上，GCh 的 ECE 从 0.030 降到 0.020，NLL 轻微下降，Top‑1 几乎不变；
- 在 ImageNet‑C 上，GCh 在选定的七种失真下 ECE 与 NLL 均优于基线；
- 在 Swin‑T 上，GCh 提升 NLL 与 ECE；
- 在 Oxford‑IIIT Pets 上保持准确率，且在细粒度识别任务中表现更好。

**⚠️ 局限性**

局限性包括：
1) 理论与实验主要聚焦于正值、空间一致的后期特征；在非正或低频特征上的效果尚未验证；
2) 目前的实现依赖于平面格点与 Dirichlet 拉普拉斯，难以直接推广到任意图形或非欧几里得结构；
3) 超参数（γ、β）需要手动调优，过大会导致准确率下降；
4) 未证明对所有网络架构或层深度均具有同样优势，且与已有随机深度/注意力掩码的兼容性仍需进一步研究。

---

## 112. TDAD: Test-Driven Agentic Development - Reducing Code Regressions in AI Coding Agents via Graph-Based Impact Analysis

**arXiv ID:** 2603.17973 | [PDF](https://arxiv.org/pdf/2603.17973v1)

**作者:** Pepe Alonso `[一作]` `[通讯]` (Universidad ORT Uruguay), Pepe Alonso (Universidad ORT Uruguay)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发并评估了TDAD（Test-Driven Agentic Development）工具，用于通过AST解析构建代码–测试依赖图并进行加权影响分析，帮助AI编码代理减少回归。

**💡 创新点**

创新点包括将图结构的影响分析与轻量级代理技能（SKILL.md+test_map.txt）相结合，以及构建自动改进循环以自适应提升工具性能。

**🔧 技术方法**

技术手段涵盖Python AST解析、NetworkX/Neo4j图数据库、加权影响分析、GraphRAG检索、静态测试映射以及自动改进算法。

**📊 数据集**

使用SWE‑bench Verified数据集（500个Python仓库的GitHub issue实例）进行实验评估。

**📈 对比分析**

对比基线（Vanilla）、TDD提示和GraphRAG+TDD三种配置，Qwen3‑Coder 30B在100例中将测试级回归率从6.08%降至1.82%（70%降幅），Resolution率略降至29%；在第二阶段使用Qwen3.5‑35B‑A3B+OpenCode时，TDAD技能将Resolution从24%提升至32%，Generation从40%提升至68%，且回归率保持0%。

**⚠️ 局限性**

局限性包括仅评估Python代码、样本量有限（100/25例）、仅使用局部量化模型、静态分析缺乏动态覆盖、未进行统计显著性检验，可能无法直接推广至大规模多语言或前沿模型。

---

## 113. Anchoring and Rescaling Attention for Semantically Coherent Inbetweening

**arXiv ID:** 2603.17651 | [PDF](https://arxiv.org/pdf/2603.17651v1)

**作者:** Tae Eun Choi `[一作]` (Yonsei University), Seong Jae Hwang `[通讯]` (Yonsei University)

**通讯引用:** 22176 | [OpenAlex ID](https://openalex.org/A5051395190)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了两种训练无关的机制，用以改进文本条件的生成中间帧（Generative Inbetweening）任务，提升语义一致性、帧一致性与速度稳定性；

**💡 创新点**

创新点在于（1）Keyframe-anchored Attention Bias（KAB）：利用关键帧交叉注意力生成帧级的引导偏置；（2）Rescaled Temporal RoPE（ReTRo）：对自注意力的时序位置编码做动态缩放，以增强关键帧保真度与中间帧一致性；（3）构建了专门针对文本条件GI的评测基准TGI-Bench；

**🔧 技术方法**

技术手段包括：基于Diffusion Transformer（DiT）的跨/自注意力调节、RoPE位置编码重标定、线性插值生成关键帧anchor、logit偏置引导；

**📊 数据集**

使用了DAVIS、Pexels、Pixabay等公开视频集合，构成220段视频并通过GPT‑4.1生成文本描述与挑战标签，生成多长度（25、33、65、81帧）评测集；

**📈 对比分析**

与TRF、ViBiDSampler、GI、FCVG以及Wan2.1等基线对比，在65/81帧长序列上，本文方法在PSNR、SSIM、LPIPS、FID、FVD、VBench等指标均取得最优或近优表现，且在人类评估中在语义忠实度与速度稳定性上最高；

**⚠️ 局限性**

局限在于：仍依赖于先验的DiT模型架构，无法完全解决极端遮挡或高度非线性运动的生成；同时评测基准与数据集主要覆盖自然场景，对工业或专业视频仍缺乏充分验证。

---

## 114. A Multi-Agent System for Building-Age Cohort Mapping to Support Urban Energy Planning

**arXiv ID:** 2603.17626 | [PDF](https://arxiv.org/pdf/2603.17626v1)

**作者:** Kundan Thota `[一作]` (Karlsruhe Institute of Technology), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 4967 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了一个多智能体LLM系统，融合德国人口普查、OpenStreetMap和古迹登记数据，生成建筑年龄分组的高覆盖率地理编码数据，并训练了基于卫星图像的BuildingAgeCNN模型，实现建筑年龄分组预测。

**💡 创新点**

创新点在于：①使用多智能体LLM自动从非结构化来源提取年龄信息并统一格式；②在卫星图像分类中引入ConvNeXt+FPN+CoordConv+SE组合，提升多尺度、空间感知特征；③在推理管道中加入置信度阈值，自动标记低置信度案例供人工复核。

**🔧 技术方法**

主要技术包括大语言模型驱动的多智能体数据采集、数据融合与去重、ConvNeXt卷积骨干+FPN+CoordConv+SE的深度学习分类网络、坐标编码、Squeeze-and-Excitation注意力、softmax概率阈值判定。

**📊 数据集**

使用的核心数据集为德国Aachen市的建筑占有率数据——由Zensus 2011、OSM以及古迹登记共同构成的15,336座建筑的年龄分组标签集（覆盖总建筑的21.17%）。另外使用高分辨率RGB卫星图像224×224作为训练样本。

**📈 对比分析**

通过空间聚类交叉验证（6折）进行评估，BuildingAgeCNN实现总体准确率90.69%、宏观F1 67.25%。在基线模型（ResNet-50、MobileNetV3、EfficientNet-B0等）对比中，ConvNeXt+FPN+CoordConv+SE显著提升精度。

**⚠️ 局限性**

主要局限包括：①类别不平衡导致宏观F1受限；②相邻年代建筑的视觉相似导致误分类；③翻新改造（新屋顶、光伏板）掩盖原有特征；④仅使用卫星图像对部分建筑不可充分辨别。

---

## 115. Toward Phonology-Guided Sign Language Motion Generation: A Diffusion Baseline and Conditioning Analysis

**arXiv ID:** 2603.17388 | [PDF](https://arxiv.org/pdf/2603.17388v1)

**作者:** Rui Hong `[一作]` (George Mason University), Jana Kosecka `[通讯]` (George Mason University)

**通讯引用:** 7930 | [OpenAlex ID](https://openalex.org/A5086078885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在手势动画生成上训练了一个基于扩散的3D手部运动模型，并通过ASL-LEX 2.0 的音位属性（手型、位置、运动等）对生成过程进行条件化，使得生成的 ASL 签名更自然、准确。

**💡 创新点**

首次系统地评估了 ASL-LEX 音位属性对扩散模型的影响，发现将符号注解转换为自然语言对 CLIP 编码器效果至关重要；同时提出将词义与音位属性通过独立张量路径进行结构化条件化的方向。

**🔧 技术方法**

采用 MDM‑style 扩散模型（Transformer 去噪器、DDIM 采样）、SMPL‑X 6D 旋转表示、CLIP 或 T5 文本编码器、符号/自然语言属性映射、以及针对手部的加权损失。

**📊 数据集**

使用 ASL3DWord（来自 WLASL 的 103 个词级 ASL 动作）作为训练/测试集，并借助 ASL‑LEX 2.0 提供的 22 维音位注释作为条件标签；SMPL‑X 参数作为运动表示。

**📈 对比分析**

与现有 CVAE 基线 SignAvatar 进行比较：扩散基线在词辨识率（0.838 vs 0.758）、KNN 准确率（0.378 vs 0.271）以及 FID（62.44 vs 62.23）等指标均优于 SignAvatar；CLIP+映射属性版本在所有评估指标上进一步超越 SignAvatar，显示出更高的分布相似度与可辨识度。

**⚠️ 局限性**

局限性包括：仅覆盖 103 个词级签名，无法直接推广到完整句子级生成；WLASL 与 ASL‑LEX 的对齐噪声导致属性标签不完全准确；模型在手部动作上存在轻微过度/欠拟合；未实现结构化张量条件化，未来工作需要验证其效果。

---

## 116. Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models

**arXiv ID:** 2603.18002 | [PDF](https://arxiv.org/pdf/2603.18002v1)

**作者:** Kevin Qu `[一作]` (Microsoft), Marc Pollefeys `[通讯]` (Microsoft)

**通讯引用:** 46258 | [OpenAlex ID](https://openalex.org/A5021908609)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个名为Loc3R-VLM的框架，使得二维视觉‑语言模型能够从单目视频中获得三维空间理解与情境感知能力，支持语言驱动的定位和视角感知的三维推理；

**💡 创新点**

主要创新点包括：①通过全球布局重建与情境建模两个联合目标实现对场景全局结构和主体姿态的显式学习；②利用预训练三维基础模型CUT3R提供的轻量化相机姿态先验，解决尺度和几何一致性问题；③在无3D标注推理阶段，仅依赖原始视频即可完成定位和问答；

**🔧 技术方法**

核心技术为：多模态Transformer、BEV（鸟瞰图）重建监督、位置与方向预测的概率化损失、相机姿态先验的插入、两种自定义定位token（位置、朝向）的加入、联合训练目标；

**📊 数据集**

使用的训练数据集包括ScanQA、SQA3D、MSQA（ScanNet）、VSI‑Bench以及自建数据；评测数据集包括SQA3D、VSI‑Bench、ScanQA、Beacon3D、MSQA；

**📈 对比分析**

与现有基于点云或单帧图像的3D MLLM、2D MLLM以及语言定位方法对比，Loc3R‑VLM在SQA3D语言定位上实现Acc@0.5m 75.2%（比View2Cap高25.2%）、在VSI‑Bench视角感知任务中相对方向准确率提升36.1%，在ScanQA/Beacon3D等通用3D问答任务上均达到或超过最优水平；

**⚠️ 局限性**

局限性包括：①仍依赖CUT3R等预训练模型的相机姿态先验，若先验失真可能影响定位；②在高度动态或非结构化场景下的布局重建效果尚未充分验证；③模型对极端视角变化或遮挡的鲁棒性有限。

---

## 117. GigaWorld-Policy: An Efficient Action-Centered World--Action Model

**arXiv ID:** 2603.17240 | [PDF](https://arxiv.org/pdf/2603.17240v1)

**作者:** Angen Ye `[一作]`, Zheng Zhu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

基于大型视频生成模型构建动作中心的世界-动作模型 GigaWorld-Policy，训练阶段联合预测动作序列与对应的未来视频，推理阶段可仅解码动作以实现低延迟控制。

**💡 创新点**

① 引入动作中心的世界-动作框架，利用未来视觉动态作为稠密监督而非完全依赖视频生成；② 通过因果注意力掩码实现动作与未来视频解耦，使视频预测在推理时可选；③ 采用多阶段预训练（大规模网页视频→机器人与人类第一人称视频→任务特定数据）提升样本效率。

**🔧 技术方法**

使用 5B 参数的扩散 Transformer 作为后端；动作与视觉令牌共享 Transformer 块；因果自注意力掩码；流匹配（flow‑matching）目标优化动作与视频；多视角拼接与 VAE 视觉编码；语言通过外部跨模态注意力编码。

**📊 数据集**

① 大规模网页视频预训练模型；② 约 10,000 小时机器人与人类第一人称视频（EgoDex、Ego4D、Agibot、RoboMind 等）；③ 目标机器人任务轨迹数据（图像、语言指令、动作）。

**📈 对比分析**

与 VLA 基线（π_0.5、GigaBrain‑0、X‑VLA）和 WAM 基线（Motus、Cosmos‑Policy）比较。实验显示：在 RoboTwin‑2.0 任务中与 Motus 取得相同平均成功率，推理速度提升约 9×（0.36 s/推理），真实世界成功率比 Motus 提高 7%，比 π_0.5 提高约 14% 或 95%（在 RoboTwin‑2.0 上）。

**⚠️ 局限性**

① 需要海量预训练数据和算力，训练成本高；② 对视觉细节的依赖仍有限，视频预测误差可能在极端长时间序列中累积；③ 仅在基于视觉的任务上表现优异，动作空间高维或非视觉信息较多的场景仍需进一步验证；④ 未来视频预测可选虽然降低推理负荷，但在某些需要实时视觉反馈的任务中可能不如完整的世界模型。

---

## 118. Dropout Robustness and Cognitive Profiling of Transformer Models via Stochastic Inference

**arXiv ID:** 2603.17811 | [PDF](https://arxiv.org/pdf/2603.17811v1)

**作者:** Antônio Junior Alves Caiado `[一作]` (Southern Methodist University), Michael Hahsler `[通讯]` (Southern Methodist University)

**通讯引用:** 4246 | [OpenAlex ID](https://openalex.org/A5027377977)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 19 种 Transformer 进行 95 个 MC Dropout 配置下的推理鲁棒性评估，利用记忆任务（SQuAD）和推理任务（HellaSwag）进行双域分解。

**💡 创新点**

提出首个跨架构的 MC Dropout 基准与认知分解框架，揭示 dropout 对记忆与推理的非对称影响及模型对任务专门化的脆弱性。

**🔧 技术方法**

使用 Monte Carlo Dropout（100 次前向传递）、标准化的 dropout 率设置、混合精度训练与 HuggingFace Transformers 进行实验。

**📊 数据集**

采用 SQuAD v1.1（记忆）和 HellaSwag（推理）各 500 条样本（训练 400/测试 100）。

**📈 对比分析**

对比确定性推理与不同 dropout 配置，测量平均准确率和标准差。结果显示 53% 模型在 MC Dropout 下性能下降，记忆任务受损显著（≈27%）而推理任务仅下降 ≈1%，最佳整体表现来自 DeBERTa‑v3‑small，稳定性与规模无明显相关性。

**⚠️ 局限性**

局限包括样本量小（仅 1k）、仅二分类任务、缺少更大规模模型、未覆盖生成或多标签任务、仅评估 MC Dropout 而非其他不确定性方法。

---

## 119. MSRAMIE: Multimodal Structured Reasoning Agent for Multi-instruction Image Editing

**arXiv ID:** 2603.16967 | [PDF](https://arxiv.org/pdf/2603.16967v1)

**作者:** Zhaoyuan Qiu `[一作]` (University Of Melbourne), Saman Halgamuge `[通讯]` (University Of Melbourne)

**通讯引用:** 12411 | [OpenAlex ID](https://openalex.org/A5067418792)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 MSRAMIE，一个训练免费、多模态推理代理框架，用于解决多指令图像编辑任务。

**💡 创新点**

创新点包括：① 引入 Tree-of-States（ToS）与 Graph-of-References（GoR）双结构化推理拓扑；② 利用多模态大型语言模型（MLLM）作为 Instructor，实现多轮交互；③ 通过插件式 Actor 与四子模块（Retriever、Instruction Generator、Evaluator、Scheduler）实现无额外训练的推理。

**🔧 技术方法**

技术手段：多模态LLM（如 Qwen3‑VL）作为 Instructor，插件式图像编辑模型（Qwen‑Image‑Edit、Flux‑Kontext、Flux2‑Klein）作为 Actor；VQA 评估 VQAScore、CLIP‑I 评估 Identity Preservation、FID 评估 Perceptual Quality；Tree‑of‑States 与 Graph‑of‑References 推理拓扑。

**📊 数据集**

使用的公开数据集为 Complex‑Edit，包含 531 张真实图像及 7 级复杂度的多指令文本。

**📈 对比分析**

实验通过与直接使用编辑模型对比，采用 IF（VQAScore & CIF）、IP（CLIP‑I）和 PQ（FID）三维指标评估。结果表明，在多指令情景下，MSRAMIE 将 IF 提升约 15%–22%，CIF 提升 71%–180%，IP 略有提升，PQ 维持不变，整体性能显著优于基线。

**⚠️ 局限性**

局限性：对极长或极复杂指令仍需更深的推理步骤；推理预算受限时可能导致细节损失或 PQ 轻微下降；整体性能仍依赖所选 MLLM 的推理质量与推理预算。

---

## 120. Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding

**arXiv ID:** 2603.17980 | [PDF](https://arxiv.org/pdf/2603.17980v1)

**作者:** Shuyao Shi `[一作]` (University of Michigan), Kang G. Shin `[通讯]` (University of Michigan)

**通讯引用:** 37218 | [OpenAlex ID](https://openalex.org/A5053541912)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Motion-MLLM，一种利用低成本 IMU 传感器捕获的 egomotion 信号的多模态大语言模型，以提升视频输入的 3D 场景理解能力。

**💡 创新点**

创新点包括：① 通过 IMU 记录的相机运动提供绝对尺度和物理锚定；② 采用三阶段 Cascaded Motion‑Visual Keyframe Filtering 以轻量化高效地筛选信息量大的关键帧；③ 设计非对称的 Cross‑Modal Fusion 模块，将 Motion Tokens 作为桥梁同时携带 egomotion 指引和跨帧视觉上下文，提升视觉表示的空间推理能力。

**🔧 技术方法**

技术手段包括：IMU 数据积分得到位姿变换；GRU 编码器对可变长度 IMU 段进行压缩为 Motion Tokens；双层 Cross‑Attention（双向 + 单向）实现视觉与运动特征融合；使用 Qwen2.5‑VL 的 2D 视觉编码器和 VGGT 几何编码器生成视觉 Tokens；两阶段训练策略（先冻结视觉/LLM 只训练运动编码和融合，再微调全模型）。

**📊 数据集**

数据集与合成：ScanQA、SQA3D、VSI‑Bench（空间推理）；ScanRefer（视觉定位）；Scan2Cap（稠密描述）；对 ScanQA、SQA3D、ScanRefer、Scan2Cap 使用已有相机位姿信息，拟合 B‑splines 生成加速度与角速度；对 VSI‑Bench 采用 GLOMAP 结构光重建位姿后再合成 IMU 数据。

**📈 对比分析**

与 SOTA 2D 及 3D 输入方法进行对比；在 ScanQA、SQA3D、VSI‑Bench 的多指标上均优于所有 2D 基线，并在绝大多数子任务中与 3D 输入模型相当或更优；整体 VSI‑Bench 得分 60.3，较之前最高 50.7 提升 9.6；在成本效益方面，Motion‑MLLM 相较 2D 基线提升 1.40×，相较 3D 基线提升 1.63×，实现了更高的准确率与更低的推理延迟。

**⚠️ 局限性**

局限性：需同步、精准的 IMU 数据，合成 IMU 过程依赖相机位姿质量；对 IMU 噪声、漂移的鲁棒性未深入评估；模型在无 IMU 设备或极低频率 IMU 的场景下表现未知；目前未在大规模多平台（如无人机、车载）中进行跨设备验证。

---

## 121. MosaicMem: Hybrid Spatial Memory for Controllable Video World Models

**arXiv ID:** 2603.17117 | [PDF](https://arxiv.org/pdf/2603.17117v1)

**作者:** Wei Yu `[一作]` (University of Toronto), Animesh Garg `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5230 | [OpenAlex ID](https://openalex.org/A5061193324)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 MosaicMem 的混合空间记忆框架，通过把视频补丁提升到 3D 空间并在目标视角拼贴，来实现长时段、可控摄像机下的视频生成。

**💡 创新点**

创新点在于将显式 3D 结构与隐式注意力结合：使用 3D 估计精确定位补丁、PRoPE 摄像机编码实现相机控制，并采用双重 warping（Warped RoPE 与 Warped Latent）在补丁与生成图像之间保持几何一致性；同时采用 patch‑and‑compose 接口，使模型能在保持稳定性与动态生成之间取得平衡。

**🔧 技术方法**

主要技术包括：DiT 文本‑图像到视频模型、流匹配（Flow Matching）生成流程、PRoPE 相机控制模块、Warped RoPE 与 Warped Latent 对齐机制、3D 深度估计器 Depth Anything V3、4× 时序压缩的 3D VAE、以及自回归的 Causal/Relic Forcing 训练策略。

**📊 数据集**

使用了新构建的 MosaicMem‑World 基准数据集，该数据集融合了 Unreal Engine 5 合成场景、游戏环境（Cyberpunk 2077）、真实第一人称视频，所有轨迹都包含周期性重访和复杂摄像机运动。

**📈 对比分析**

与传统显式记忆（GEN3C、SEVA、Vmem、VWM）和隐式记忆（WorldMem、CaM）进行对比，评估指标包括 FID/FVD（视频质量）、RotErr/TransErr（摄像机误差）、动态分数和一致性得分；MosaicMem 在所有指标上均优于基线，特别是在摄像机跟随精度和记忆一致性方面显著提升。

**⚠️ 局限性**

局限性：仍受 3D 估计误差影响，重投影分辨率限制导致细节模糊；在极大旋转或极慢运动时需要额外的 Warped Latent 校正；实时生成在极大场景规模下仍受算力限制；对极端动态对象的精确控制与编辑尚不充分。

---

## 122. SA-CycleGAN-2.5D: Self-Attention CycleGAN with Tri-Planar Context for Multi-Site MRI Harmonization

**arXiv ID:** 2603.17219 | [PDF](https://arxiv.org/pdf/2603.17219v1)

**作者:** Ishrith Gowda `[一作]` (University of California), Chunwei Liu `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了SA-CycleGAN-2.5D模型，实现无监督体素级多站MRI图像和谐化，保持解剖结构与肿瘤特征不变。

**💡 创新点**

创新点包括：2.5D三平面输入保持跨层梯度、U-ResNet结合CBAM与密集自注意力以突破卷积感受野限制，以及谱归一化PatchGAN实现稳定高分辨率对抗训练。

**🔧 技术方法**

采用CycleGAN框架、2.5D SliceEncoder、U-ResNet+CBAM+自注意力模块、谱归一化多尺度PatchGAN、MMD与域分类器等评估技术。

**📊 数据集**

使用BraTS多站异构数据与UPenn-GBM同站数据共654例，涵盖四模态（T1/T1CE/T2/FLAIR）进行训练与评估。

**📈 对比分析**

与ComBat及无注意力基线对比，MMD降低99.1%，域分类准确率降至59.7%，SSIM>0.92，self-attention提升循环SSIM 1.1-1.3%，显示出显著性能优势。

**⚠️ 局限性**

局限性在于仅支持两域训练，需为每对站点单独训练；对多站点统一处理以及对肿瘤特征的长远临床影响仍需进一步验证。

---

## 123. ConGA: Guidelines for Contextual Gender Annotation. A Framework for Annotating Gender in Machine Translation

**arXiv ID:** 2603.17962 | [PDF](https://arxiv.org/pdf/2603.17962v1)

**作者:** Argentina Anna Rescigno `[一作]` (University of Pisa), Johanna Monti `[通讯]` (University of Naples L'Orientale)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了ConGA框架，对英意机器翻译中的性别表达进行系统注释，并构建金标准语料库，随后使用精度、召回率和F1值对两大模型（TowerLLM与mBART）进行定量评估，揭示其在性别保持与歧义处理上的偏差。

**💡 创新点**

创新点在于：① 将源语言的语义性别与目标语言的语法性别对齐，提供可复制、可扩展的评估体系；② 引入细粒度的性别标注与实体编号，能够捕捉模型对模棱两可实体的性别决策；③ 通过对比两大模型在男性/女性保持与A→M/A→F歧义解析上的表现，系统量化性别偏差。

**🔧 技术方法**

技术手段包括：手工注释（使用INCEpTION平台实现跨语言实体对齐）、两大模型的翻译输出（TowerInstruct-7B与mBART）在低温度设置下生成、基于标注的精度/召回率/F1计算脚本，以及对错误/偏差/未匹配实体的日志记录。

**📊 数据集**

使用的数据集是gENder-IT（MuST‑SHE扩展的英意平行语料），其中包含对人类参照词的词级性别标注。

**📈 对比分析**

评估方法：按实体对齐，分别计算男性和女性的Precision、Recall与F1；对含歧义实体的A→M与A→F转换进行计数。结果显示TowerLLM的整体精度略高，但两模型均存在男性默认倾向；在歧义案例中，两模型大多倾向于A→M，证明性别偏差仍显著。

**⚠️ 局限性**

局限性：① 仅关注英意双语对，未覆盖多语言或低资源场景；② 低温度设置保证一致性但可能掩盖高温度下的性别多样性；③ 只评估两大模型，未检验更广泛的LLM；④ 研究未深入探讨非二元和多元性别身份的处理。

---

## 124. From Symbol to Meaning: Ontological and Philosophical Reflections on Large Language Models in Information Systems Engineering

**arXiv ID:** 2603.17659 | [PDF](https://arxiv.org/pdf/2603.17659v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 125. Integration of local and global surrogates for failure probability estimation

**arXiv ID:** 2603.17211 | [PDF](https://arxiv.org/pdf/2603.17211v1)

**作者:** Audrey Gaymann `[一作]` (University of Colorado), Alireza Doostan `[通讯]` (University of Colorado)

**通讯引用:** 3960 | [OpenAlex ID](https://openalex.org/A5005638433)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出全局‑局部混合代理（GLHS）方法，用于高效估计复杂系统的罕见失效概率。

**💡 创新点**

通过动态学习缓冲区并采用 Christoffel 自适应采样构建局部代理，结合全局代理实现迭代优化，显著减少模型评估次数。

**🔧 技术方法**

采用多项式截断（PCE）、Christoffel 自适应采样、缓冲区域学习、最小二乘回归等技术。

**📊 数据集**

在 1D、2D 解析测试函数以及 4D PLATO 大气进入化学反应模拟的数据集上进行验证。

**📈 对比分析**

与传统蒙特卡罗、全局代理及非迭代采样方法比较，GLHS 在相同评估次数下误差下降至 0.1%–1%，且计算成本显著降低。

**⚠️ 局限性**

仅适用于单一缓冲区、假设均匀分布，且对多失效区或高度非线性系统可能需要进一步改进。

---

## 126. Temporal Narrative Monitoring in Dynamic Information Environments

**arXiv ID:** 2603.17617 | [PDF](https://arxiv.org/pdf/2603.17617v1)

**作者:** David Farr `[一作]` (University of Washington), Jevin West `[通讯]` (University of Washington)

**通讯引用:** 6086 | [OpenAlex ID](https://openalex.org/A5046879461)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一个基于句子嵌入、HDBSCAN密度聚类和滚动时间窗口的动态叙事监测系统，用于实时跟踪社交媒体中演变的叙事；

**💡 创新点**

创新点在于不依赖先验标签，维护叙事的持续身份并量化其漂移，实现叙事生命周期的可视化与动态追踪；

**🔧 技术方法**

采用句子Transformer进行语义嵌入，HDBSCAN进行密度聚类，滚动时间窗口与相似度阈值实现叙事链接，生成式总结（GPT‑4.1‑mini）提供聚类摘要与主题标签；

**📊 数据集**

使用了约9,914条包含关键词“maduro”或“venezuela”的推文（3,799名用户），时间范围为2026年1月2日至7日，且仅保留转发数≥50的帖子；

**📈 对比分析**

通过人工标注1,033条帖子评估聚类准确率达91%；叙事漂移平均cosine距离0.26，生命周期中位持续2个窗口（8小时），最长持续88小时；噪声比例随时间波动，表明系统对话语结构变化具有良好适应性；

**⚠️ 局限性**

限制包括需预设时间窗口与相似度阈值，难以捕捉细粒度叙事细节；依赖多模块（嵌入、聚类、生成），计算成本较高；系统为辅助分析工具，需人工解读与验证。

---

## 127. Noise-Response Calibration: A Causal Intervention Protocol for LLM-Judges

**arXiv ID:** 2603.17172 | [PDF](https://arxiv.org/pdf/2603.17172v1)

**作者:** Maxim Khomiakov `[一作]` (Technical University of Denmark), Jes Frellsen `[通讯]` (Technical University of Denmark)

**通讯引用:** 3865 | [OpenAlex ID](https://openalex.org/A5087676204)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种基于可控噪声干预的LLM裁判器校准协议，利用噪声水平递增时的性能下降趋势来评估模型的可信度。

**💡 创新点**

创新点在于将因果干预视角与斜率假设检验结合，构建一套可重复的评估流程，并揭示了文本与表格数据在噪声响应上的显著模态差距。

**🔧 技术方法**

使用的技术包括线性回归斜率检验、SNR控制的高斯噪声（针对表格）、词汇扰动（针对文本）、以及GPT‑5‑mini等大型语言模型进行推理。

**📊 数据集**

实验覆盖了138个UCI分类、21个UCI回归数据集，以及四个文本情感分类数据集（IMDB、Yelp Review Full、SST‑2、Financial PhraseBank）。

**📈 对比分析**

通过在每个噪声水平下重复5次实验，计算准确率或R²并进行单边t检验；文本数据全部显著下降，表格数据仅约36%（分类）/24%（回归）显著；无噪声基线中，噪声不敏感的数据集准确率更低、方差更大。

**⚠️ 局限性**

局限性包括：仅针对数值特征的高斯噪声，无法覆盖大量类别特征；评估仅使用单一LLM模型；线性斜率检验可能忽略非线性衰减；且噪声干预可能不完全代表真实分布漂移。

---

## 128. Deployment and Evaluation of an EHR-integrated, Large Language Model-Powered Tool to Triage Surgical Patients

**arXiv ID:** 2603.17234 | [PDF](https://arxiv.org/pdf/2603.17234v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 129. Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval

**arXiv ID:** 2603.17872 | [PDF](https://arxiv.org/pdf/2603.17872v1)

**作者:** Md. Asraful Haque `[一作]` (Aligarh Muslim University), Tamkeen Fatima `[通讯]` (Aligarh Muslim University)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5099426477)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个多阶段、基于域感知的检索与验证管道，自动拦截 LLM 的幻觉生成，提供更可靠的回答。

**💡 创新点**

创新点包括：① 结合内在早停与外部检索的自调节流程；② 引入 Corrective Document Grading (CRAG) 进行文档筛选；③ 将答案拆解为原子主张进行逐条验证；④ 发现并强调 False‑Premise Overclaiming 失效模式，提出预检问答可回答性节点。

**🔧 技术方法**

使用的技术：LangGraph 工作流、Llama 3.1 8B（生成、抽取、评分）、Gemma3 27B（外部评判）、Tavily 搜索 API（分层检索）、Pydantic 结构化校验、Atomic Claim 提取与验证、CRAG 文档评分、Intrinsic/Extrinsic Confidence 评估。

**📊 数据集**

评测数据集共 650 条问答，来源于五大基准：TimeQA v2、FreshQA v2、HaluEval General、MMLU Global Facts、TruthfulQA。

**📈 对比分析**

通过 Gemma3 27B 评判将管道输出与零射击 Llama 3.1 8B 基线对比，结果显示平均胜率 56.2%，TimeQA 达 83.7%，MMLU 78.0%；Groundedness 在 78.8%–86.4% 之间，Hallucination 在 3.5%–33.1% 之间，表明检索提升了事实可信度。

**⚠️ 局限性**

局限性：① 对开放域已知事实的检索增益有限；② 误判无效前提导致 overclaim；③ 生成的拒绝答案过长不够简洁；④ 检索信息易导致偏移（Retrieval Distraction）；⑤ 数值/时间单位差异导致高幻觉率；⑥ 小模型在结构化数据提取上的局限性。

---

## 130. Average Case Graph Searching in Non-Uniform Cost Models

**arXiv ID:** 2603.17916 | [PDF](https://arxiv.org/pdf/2603.17916v1)

**作者:** Michał Szyfelbein `[一作]` `[通讯]` (Gdańsk University of Technology), Michał Szyfelbein (Gdańsk University of Technology)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并研究了平均成本图搜索问题（Graph Search Problem），在不同成本模型（与目标无关、单调递增、任意）下给出多种近似与逼近算法，并给出相应的硬度下界。

**💡 创新点**

核心创新包括：
- 对树结构的非目标相关成本模型给出了 4+ε 的 FPTAS；
- 对一般图给出了 O(√log n) 的多项式近似；
- 对具有有限非叶子顶点的图给出了参数化 FPTAS；
- 对单调递增成本模型在树上提供了 2-近似；
- 证明任意成本模型下（即使是星形树）在 UGC 下无法得到任何常数近似。

**🔧 技术方法**

技术手段主要包括：
- 将搜索成本与 α‑分离器、最小比率顶点割等分割问题关联，利用分割的下界构造递归近似；
- 动态规划求解加权 α‑分离器，配合舍入得到 FPTAS；
- LP 及其松弛结合伪分离器与重构技术实现 2‑近似；
- 排序/调度等价变换用于星形图的 FPTAS；
- 通过归约展示硬度，连接到 Minimum Linear Ordering 与 Feedback Arc Set 等经典难题。

**📊 数据集**

论文为理论性工作，未涉及具体实验或数据集；所有结果均基于理论证明与算法复杂度分析。

**📈 对比分析**

与以往已知的 PTAS、2‑近似、4‑近似等结果进行对比；
- 在树上，4+ε 逼近优于此前的 4‑近似；
- 对一般图实现 O(√log n) 近似，显著提升；
- 对单调成本模型给出的 2‑近似为首个常数近似；
- 证明在任意成本模型下常数近似不可行，进一步界定了可行范围。

**⚠️ 局限性**

局限与挑战：
- 对树的 4+ε 方案仍需 O(n⁴/ε²) 复杂度；
- 对一般图的 O(√log n) 近似仍相对粗糙；
- 参数化 FPTAS 仅适用于非叶子顶点数有限的图；
- 在任意成本模型下仅能给出不可近似的下界；
- 对于大规模图，现有算法的时间/空间需求仍较高。

---

## 131. MLmisFinder: A Specification and Detection Approach of Machine Learning Service Misuses

**arXiv ID:** 2603.17330 | [PDF](https://arxiv.org/pdf/2603.17330v1)

**作者:** Hadil Ben Amor `[一作]` (École de Technologie Supérieure), Naouel Moha `[通讯]` (École de Technologie Supérieure)

**通讯引用:** 2830 | [OpenAlex ID](https://openalex.org/A5007508017)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 MLmisFinder，基于元模型和规则的自动静态分析工具，用于检测机器学习云服务的七类常见误用。

**💡 创新点**

创新点在于：①构建统一的跨云供应商元模型；②首次对 ML 服务误用进行系统化、可扩展的自动检测；③通过七种静态检测规则实现高覆盖率。

**🔧 技术方法**

技术方法包括：Python AST 静态源码分析、元模型实例化、基于规则的误用检测算法，以及与 GitHub API 的仓库克隆和代码提取。

**📊 数据集**

数据集：107 个手工标注的 GitHub ML 服务项目（共 340 个误用实例）用于评估；817 个公开仓库用于大规模误用分布分析。

**📈 对比分析**

比较方法：将 MLmisFinder 与 Wan 等人提出的仅检测输出误解的基线工具对比。结果显示 MLmisFinder 平均精度 96.7%、召回率 97%，显著优于基线（精度 17.3%、召回率 56.2%）。在 817 个仓库中发现误用普遍存在，尤其是数据漂移监控与模式校验。

**⚠️ 局限性**

局限性：仅支持 Python 代码；误用检测依赖手工维护的云服务库列表，需频繁更新；静态分析无法捕获运行时行为，误用覆盖面受限；目前仅检测七类误用，支持的云供应商也局限于 AWS、Azure、Google。

---

## 132. TimeAPN: Adaptive Amplitude-Phase Non-Stationarity Normalization for Time Series Forecasting

**arXiv ID:** 2603.17436 | [PDF](https://arxiv.org/pdf/2603.17436v1)

**作者:** Yue Hu `[一作]` (Harbin Institute of Technology), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 99818 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 TimeAPN（Adaptive Amplitude–Phase Non‑Stationarity Normalization）框架，利用时间域和频率域的均值、幅值、相位信息进行自适应归一化，并通过预测未来非平稳因子与预测模型协同重建长时序预测结果。

**💡 创新点**

①同时建模幅值与相位以捕捉快速变化的非平稳性；②在时间域与频率域并行归一化；③使用可学习的相位差预测与逆变换实现精细对齐；④以可插拔的归一化–去归一化流程兼容多种基线模型。

**🔧 技术方法**

离散小波变换（DWT）频域分解；MeanNormPhase 均值/相位提取；MLP+TCN 预测幅值与相位差；协同两阶段训练；集成多种骨干（FEDformer、DLinear、PatchTST、SMamba）。

**📊 数据集**

七个公开多变量时序数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Electricity、Traffic。

**📈 对比分析**

与 RevIN、NST、Dish‑TS、SAN、DDN 等可逆归一化方法及基准模型在 96/192/336/720 四种预测长度下进行对比。TimeAPN 在所有基线模型上均显著降低 MSE/MAPE，尤其在大规模数据集上提升幅度更大，整体性能优于现有最优方法。

**⚠️ 局限性**

1）需要额外的 DWT 与 MPPM 模块，计算和参数量增大；2）采用两阶段训练，增加训练复杂度；3）对极端非平稳或缺失数据的鲁棒性尚未充分验证；4）在短期预测或单变量场景的优势不明显；5）兼容性需在更多模型上进一步检验。

---

## 133. Synthetic Differential Geometry in Lean

**arXiv ID:** 2603.17457 | [PDF](https://arxiv.org/pdf/2603.17457v1)

**作者:** Riccardo Brasca `[一作]` (Université Paris Cité and Sorbonne Université), Gabriella Clemente `[通讯]` (Institut de recherche en informatique fondamentale)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文在 Lean4 证明助手和其数学库中正式化了合成微分几何（SDG），并证明了在无限小邻域内的多变量泰勒定理及其多变量形式的等式。

**💡 创新点**

创新点在于：①首次在证明助手中实现 SDG 的完整形式化，②证明的所有步骤均为全新且不依赖经典逻辑（避免使用排中律和选取原理），③通过引入唯一选择公理（axiom of unique choice）实现了在构造性框架下的函数选择；④系统性地排查并重写了大量依赖经典选择的库声明，使得工作可以在构造性 Lean 环境下执行。

**🔧 技术方法**

使用技术包括：Lean4 依赖类型理论、构造性公理体系、唯一选择公理、对 Weil 代数和光滑拓扑的内在逻辑推理，以及对多变量偏导、拉氏规则、链式法则等微分运算的形式化。

**📊 数据集**

并未使用外部数据集；全部工作基于 Lean4 数学库（mathlib4）和自建 SDG 库，对应的声明和定理都与具体代码行号关联。

**📈 对比分析**

方法与性能：通过形式化证明而非数值实验，性能评估以 Lean4 编译时间和依赖树复杂度为主。论文报告的主要指标是证明的可重用性和构造性兼容性；实验表明在排除经典选择后，编译时间略有增加，但仍在可接受范围内；在大规模依赖链中，成功移除 8,000+ 选取依赖。

**⚠️ 局限性**

局限性包括：①目前的构造性框架对某些高级数学结构（如更高维光滑模型）的支持仍有限；②部分证明仍需手工拆分以避免使用默认的自动化 tactic，工作量较大；③虽然引入唯一选择公理，但对全局一致性和可维护性仍需进一步研究。

---

## 134. Exploring parameter-efficient fine-tuning (PEFT) of billion-parameter vision models with QLoRA and DoRA: insights into generalization for limited-data image classification under a 98:1 test-to-train regime

**arXiv ID:** 2603.17782 | [PDF](https://arxiv.org/pdf/2603.17782v1)

**作者:** Haiyu Yang `[一作]` (Cornell University), Miel Hostens `[通讯]` (Cornell University)

**通讯引用:** 2714 | [OpenAlex ID](https://openalex.org/A5052695473)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究对牛行为识别任务进行系统评估，比较了从零开始训练、冻结特征提取以及在DINOv3上采用参数高效微调（QLoRA和DoRA）的三种策略；

**💡 创新点**

创新点在于首次系统比较PEFT方法在农业视觉任务中的表现，证明低比率适配器容量提升可缓解欠拟合，并在98:1测试/训练比例下实现高精度；

**🔧 技术方法**

使用DINOv3大模型，QLoRA和DoRA低秩适配器技术，配合数据增强、ResNet-18/ViT‑Small训练及细致手工验证；

**📊 数据集**

采用9类牛行为的MMCows和PlayBehavior数据集，手工校正后得到2,160张训练图像，在211,800张未修正图像上进行测试；

**📈 对比分析**

通过对比训练时间、可训练参数比例、准确率和F1分数，最佳QLoRA all‑linear r=64在5h46m内取得83.16%准确率，显著优于ResNet‑18（72.87%）和冻结DINOv3（76.56%），DoRA性能相近但训练更慢；

**⚠️ 局限性**

局限性包括数据集规模有限且跨源差异显著，评估仅基于单帧忽略时间序列，推理吞吐低，未考虑多GPU扩展与实时部署，标签噪声可能导致误判。

---

## 135. Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training

**arXiv ID:** 2603.17970 | [PDF](https://arxiv.org/pdf/2603.17970v1)

**作者:** Ben S. Southworth `[一作]` (Los Alamos National Laboratory), Stephen Thomas `[通讯]` (Lehigh University)

**通讯引用:** 1785 | [OpenAlex ID](https://openalex.org/A5111448726)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为MUD的矩阵正交化优化器，利用单次或少量三角解代替Muon的Newton‑Schulz循环，实现对Transformer权重矩阵的快速“白化”。

**💡 创新点**

创新点在于用低成本的三角格拉姆近似与前向三角求解构建可在矩阵空间内实现局部二次收敛的正交化器，并将其嵌入大型Transformer训练中显著降低优化器开销。

**🔧 技术方法**

使用的技术包括三角格拉姆预处理、Cholesky‑QR、Gauss‑Seidel预条件、Newton‑Schulz、AdamW以及MuO等，并在单机多GPU、BF16等硬件上实现。

**📊 数据集**

实验所用数据集包括OpenWebText（NanoGPT）、WikiText‑103、FineWeb‑Edu文本语料以及ESM‑2 150M蛋白质语言模型的掩码语言建模任务。

**📈 对比分析**

通过在GPT‑2 small/medium/large、A100/MI250/GH200等环境下与AdamW、MuO对比，MUD在时间‑到‑perplexity上比AdamW快20–50%，比MuO快10–30%，在Tokens/s上提升1.3–3倍。

**⚠️ 局限性**

局限性包括在非Transformer、低矩阵规模任务（如CIFAR‑10）未见显著加速；多步MUD对数值稳定性要求较高，在极低perplexity阶段优势减弱。

---

## 136. QuantFL: Sustainable Federated Learning for Edge IoT via Pre-Trained Model Quantisation

**arXiv ID:** 2603.17507 | [PDF](https://arxiv.org/pdf/2603.17507v1)

**作者:** Charuka Herath `[一作]` (Institute of Digital Technologies), Sangarapillai Lambotharan `[通讯]` (Institute of Digital Technologies)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 QuantFL，一种基于预训练模型的边缘 IoT 可持续联邦学习框架。

**💡 创新点**

创新点在于利用预训练压缩更新的动态范围，实现无需误差反馈的桶式量化，并在上传端显著减少能耗。

**🔧 技术方法**

技术上结合了 Bucket-Uniform 与 Bucket-Quantile 量化、中点解码、周期性码本刷新以及标准 FedAvg 聚合。

**📊 数据集**

在 MNIST 与 CIFAR-100 两个图像基准上使用 ResNet-18 等网络进行实验。

**📈 对比分析**

与非量化 FedAvg、QSGD 等基线相比，QuantFL 在约 40% 的总通信量下降下，保持甚至提升测试精度；在非 IID 场景下 BU 仍稳健。

**⚠️ 局限性**

局限包括仅在图像任务与中型模型上验证，未考虑真实设备的参与度、时延和能耗模型，且下行量化尚未实现。

---

## 137. DANCE: Dynamic 3D CNN Pruning: Joint Frame, Channel, and Feature Adaptation for Energy Efficiency on the Edge

**arXiv ID:** 2603.17275 | [PDF](https://arxiv.org/pdf/2603.17275v1)

**作者:** Mohamed Mejri `[一作]` (Georgia Institute of Technology), Abhijit Chatterjee `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 6215 | [OpenAlex ID](https://openalex.org/A5069579193)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种基于激活可变性放大与自适应激活剪枝的3D CNN动态细粒度剪枝框架DANCE；

**💡 创新点**

创新点在于通过先训练提升各帧、通道、特征激活的方差，再由轻量化控制器根据首层统计动态裁剪后续层的帧、通道和特征，从而实现输入感知、层级化、极低开销的剪枝；

**🔧 技术方法**

采用激活方差放大（AVA）训练、轻量化控制器网络、Straight‑Through Estimator 与 Gumbel‑Softmax 等技术；

**📊 数据集**

在UCF101和HMDB51视频动作识别数据集上使用R(2+1)D与C3D模型进行实验；

**📈 对比分析**

与现有静态及动态剪枝方法对比，DANCE在UCF101上实现4×剪枝率、+1.5%准确率提升，HMDB51上实现2.77×剪枝率仅2%准确率下降；在Jetson Nano上加速1.37×、能耗提升1.47×，在Snapdragon 8 Gen 1上加速2.22×；

**⚠️ 局限性**

局限包括：控制器训练需额外样本、对不同硬件的适配仍需优化、对非3D CNN或Transformer模型的推广尚待验证、阈值设定对性能敏感且需手动调优。

---

## 138. Graph-Native Cognitive Memory for AI Agents: Formal Belief Revision Semantics for Versioned Memory Architectures

**arXiv ID:** 2603.17244 | [PDF](https://arxiv.org/pdf/2603.17244v1)

**作者:** Young Bin Park `[一作]` `[通讯]` (Kumiho Inc), Young Bin Park (Kumiho Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Kumiho 这一基于图数据库的认知记忆架构，能够同时管理 AI 代理的记忆和工作产出，实现版本化、检索、合并与安全归档；

**💡 创新点**

核心创新包括：1) 将认知记忆与资产管理统一到同一图模型；2) 在图操作层面实现 AGM 经典信念修订语义，并通过正式证明满足 K*2–K*6、Relevance 与 Core‑Retention；3) 采用 URI 统一寻址、typed 依赖边与可变标签指针，支持多代理流水线；4) 引入 Prospective Indexing、Event Extraction 与 Client‑side LLM Reranking 等增效机制；

**🔧 技术方法**

技术手段包括：图数据库 Neo4j（长期存储）+ Redis（工作记忆）、Model Context Protocol（跨模型记忆接口）、异步 Consolidation Pipeline（睡眠时间计算）以及 hybrid retrieval（全文+向量检索+CombMAX融合）；

**📊 数据集**

使用 LoCoMo（Token‑level F1）与 LoCoMo‑Plus（约束回忆）两大基准测试；

**📈 对比分析**

在 LoCoMo 上获得 0.447 四分类 F1 与 97.5% 对抗性拒绝率（总体 0.565 F1），在 LoCoMo‑Plus 上达到 93.3% 判定准确率与 98.5% 召回率，显著优于现有基线（Gemini 2.5 Pro 仅 45.7%）；

**⚠️ 局限性**

限制与挑战：1) 仅满足 AGM 的基本 7 条后置条件，K*7/K*8 尚未完整形式化；2) 采用 propositional 语义，无法表达更复杂的描述逻辑约束；3) 对于合并与部分更新，当前实现依赖完整修订（atomic replacement），未来需设计 LLM 驱动的语义合并；4) 依赖于提示工程以确保自然语言转三元组的准确性；5) 对大规模多代理协作的可扩展性与性能尚待进一步评估。

---

## 139. EI: Early Intervention for Multimodal Imaging based Disease Recognition

**arXiv ID:** 2603.17514 | [PDF](https://arxiv.org/pdf/2603.17514v1)

**作者:** Qijie Wei `[一作]` (Renmin University of China), Xirong Li `[通讯]` (Renmin University of China)

**通讯引用:** 6651 | [OpenAlex ID](https://openalex.org/A5060270456)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 Early Intervention（早期干预）框架和 MoR（低秩多维适配）模块，用于多模态医学影像的疾病识别。

**💡 创新点**

创新点在于：①在目标模态特征提取前插入由参考模态 CLS 生成的 INT 令牌，实现在 UNIMODAL 嵌入阶段的跨模态指导；②MoR 通过多秩低秩适配器和放松路由实现参数高效、性能更优的 VFM 微调。

**🔧 技术方法**

使用 Vision Foundation Models（CLIP、DINOv2）构建 ViT，配合低秩适配器、混合专家、轻量级后期融合以及 INT 令牌生成与插入技术。

**📊 数据集**

实验数据集包括：MMC‑AMD（眼底 CFP+OCT，四类 AMD 诊断），Derm7pt（皮肤临床+皮镜图，五类皮肤病），MRNet（膝关节多视角 MRI，多标签异常）。

**📈 对比分析**

与 MM‑MIL、CosCatNet、RadDiag、MMRAD 等多模态基线进行对比，使用 AP、mAP、AUC、S2 等指标；在三数据集均实现了 SOTA 性能，显著优于对照方法。

**⚠️ 局限性**

局限性：INT 令牌的获取和插入位置固定，缺乏自适应机制；辅助 VFM 较重，轻量化实现仍需探索。

---

## 140. GenLie: A Global-Enhanced Lie Detection Network under Sparsity and Semantic Interference

**arXiv ID:** 2603.16935 | [PDF](https://arxiv.org/pdf/2603.16935v1)

**作者:** Zongshun Zhang `[一作]` (University of Electronic Science and Technology of China), Wei Lin `[通讯]` (Yizhou Prison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了 GenLie 框架，采用局部特征建模与全局监督相结合的方法进行视频谎言检测。

**💡 创新点**

创新点在于局部–全局联合建模，融合冗余感知帧选择、任务驱动再嵌入、对抗性说话人去相关化和三元组损失，以提升稀疏且可辨识的谎言特征。

**🔧 技术方法**

使用 VideoMAEv2 编码器、梯度反转层（GRL）进行说话人去相关化、三元组损失、基于 AU/微表情/注视/姿态的帧采样策略以及全局平均池化+MLP嵌入。

**📊 数据集**

在 MDPE、Real‑Life Trial 与 SEUMLD 三个公开数据集上进行实验。

**📈 对比分析**

与显式、混合和隐式基线在 F1、ACC、AUC 上对比，GenLie 在所有数据集均实现或逼近最高分，显著提升检测性能。

**⚠️ 局限性**

对极低质量或高噪声视频的鲁棒性尚未充分验证，帧选择策略对不同数据集可能需要自适应调整，且依赖冻结的大型 VideoMAEv2 可能限制跨域迁移能力。

---

## 141. In Trust We Survive: Emergent Trust Learning

**arXiv ID:** 2603.17564 | [PDF](https://arxiv.org/pdf/2603.17564v1)

**作者:** Qianpu Chen `[一作]` (Leiden University), Derya Soydaner `[通讯]` (Leiden University)

**通讯引用:** 364 | [OpenAlex ID](https://openalex.org/A5054083027)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种轻量级的信任学习框架（ETL），通过维护内部信任状态和记忆机制，让纯自利的多智能体在不需要显式合作奖励的竞争环境中实现自发合作。

**💡 创新点**

创新点在于将信任作为单一的局部状态变量，利用局部观测和自身奖励自适应更新信任、记忆和探索率，使得合作能在匿名、非重复交互的环境中自然出现。

**🔧 技术方法**

技术实现包括：短期/长期记忆缓冲、基于局部信号的信任更新公式、动态探索率调节以及可插拔的控制模块，可直接附加于现有价值或策略学习器。

**📊 数据集**

使用三种自建实验环境：网格资源采集、受限楼层平台（Tower）以及迭代囚徒困境，实验通过多次随机种子训练得到评估数据。

**📈 对比分析**

与传统 Q‑Learning、Monte‑Carlo 等基线相比，ETL 在资源冲突率降低、资源保留率提升、塔楼存活率稳定在 90% 以上，并在 IPD 中在多样化对手下取得最高胜率；实验显示其对初期贪婪、强势对手均具恢复与抵抗能力。

**⚠️ 局限性**

局限性包括：仅在小规模、结构化的实验环境验证；缺乏深度强化学习和大规模人类玩家交互的评估；对动态进入退出、异质团队的适应性尚未充分探索。

---

## 142. Agentic Cognitive Profiling: Realigning Automated Alzheimer's Disease Detection with Clinical Construct Validity

**arXiv ID:** 2603.17392 | [PDF](https://arxiv.org/pdf/2603.17392v1)

**作者:** Jiawen Kang `[一作]` (Chinese University of Hong Kong), Helen Meng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9536 | [OpenAlex ID](https://openalex.org/A5019458385)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出Agentic Cognitive Profiling (ACP)框架，通过多任务LLM代理将标准化认知测评拆解为原子任务，提取可验证评分原语，完成阿尔茨海默病自动筛查并生成可解释认知报告。

**💡 创新点**

创新点在于将临床测评逻辑与LLM代理结合，分离语义理解与测量，使用确定性函数调用与验证回路消除幻觉，恢复构造效度并在多域评估中实现可解释的预测。

**🔧 技术方法**

使用多代理LLM工作流（Qwen3‑8B）+确定性函数调用 + 验证器回路 + 归一化 + 监督SVM/零样本阈值方法。

**📊 数据集**

使用402名粤语老年人手工转录、任务分段的认知测评语料，包含MoCA‑SL和HKLLT等八项结构化任务。

**📈 对比分析**

与手工特征、BERT/RoBERTa、LLM‑CoT基线对比；ACP零样本准确率为85.3%，监督SVM进一步提升至85.3%，SMR平均90.5%，MAE 0.10，显著优于所有基线。

**⚠️ 局限性**

局限：受限于预定义评分规则、LLM语义理解能力、数据集可访问性；仅适用于已设定的认知域，未验证跨语言迁移。

---

## 143. FastLoop: Parallel Loop Closing with GPU-Acceleration in Visual SLAM

**arXiv ID:** 2603.17201 | [PDF](https://arxiv.org/pdf/2603.17201v1)

**作者:** Soudabeh Mohammadhashemi `[一作]` (Simon Fraser University), Steven Y. Ko `[通讯]` (Simon Fraser University)

**通讯引用:** 1736 | [OpenAlex ID](https://openalex.org/A5109113161)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 GPU 加速的 Loop Closing 模块 FastLoop，并集成到 ORB‑SLAM3，显著提升了循环闭合的实时性能。

**💡 创新点**

创新点在于：① 通过任务级与数据级并行重构循环闭合流水线，把 Projection Search、Loop Fusion 等计算密集型步骤并行化；② 将关键帧完全存放于 GPU 内存，减少 CPU‑GPU 数据传输；③ 使用 Graphite 与自动微分实现 GPU 端的姿态图优化。

**🔧 技术方法**

采用的技术包括：CUDA GPU 编程、任务级并行、数据级并行、自动微分、Graphite 库、Pinned memory、Eigen LDLT 线性求解。

**📊 数据集**

实验数据集：KITTI（外部）和 TartanAir（内部）两套数据集。

**📈 对比分析**

与原始 ORB‑SLAM3 在桌面（RTX 3060 Ti）和嵌入式 Jetson Orin Nano 上进行同序列比较，测量 Loop Closing 模块和整体运行时。结果显示桌面平均加速 1.4–3.4 倍，嵌入式 1.3–2.4 倍，轨迹误差与原系统相当。

**⚠️ 局限性**

局限性：对图规模较小的序列加速不明显；GPU 资源占用高，仅覆盖部分模块（Region Detection 与 Keyframe 连接仍在 CPU 上）；实现需要修改 ORB‑SLAM3 源码并依赖 CUDA 环境。

---

## 144. A Multi-Level Data-driven Framework for Understanding Perceptions Towards Cycling Infrastructure Across Regions Leveraging Social Media Discourse

**arXiv ID:** 2603.17221 | [PDF](https://arxiv.org/pdf/2603.17221v1)

**作者:** Shiva Azimi `[一作]` (Villanova University), Arash Tavakoli `[通讯]` (Villanova University)

**通讯引用:** 381 | [OpenAlex ID](https://openalex.org/A5066478442)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对美国与欧洲城市的Reddit讨论进行多尺度情感与主题分析，评估自行车基础设施的公众感知。

**💡 创新点**

创新点在于将情感、主题和维度情感分析结合到分层统计模型，并在跨洲大规模语料上比较帖子与评论的情绪演变。

**🔧 技术方法**

采用VADER情感分析、BERTopic主题建模、关键词规则维度情感和线性混合效应模型。

**📊 数据集**

使用约3万条帖子及50万条评论的Reddit数据，覆盖美国各州和若干欧盟国家。

**📈 对比分析**

通过比较不同地区、城市以及帖子/评论层级的情绪分布，发现欧美差异显著但效应量小，城市层面变异主要来源于讨论主题；方法在统计显著性上表现优良。

**⚠️ 局限性**

主要局限是Reddit样本不具代表性、地区分布不均、VADER缺乏专业语义捕捉、未与客观基础设施指标关联。

---

## 145. The Inverse Lyndon Array: Definition, Properties, and Linear-Time Construction

**arXiv ID:** 2603.17537 | [PDF](https://arxiv.org/pdf/2603.17537v1)

**作者:** Pietro Negri `[一作]` (University of Salerno), Rosalba Zizza `[通讯]` (University of Salerno)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5085208855)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

定义并构造了逆 Lyndon 数组（Inverse Lyndon Array），记录每个位置上最长的逆 Lyndon 子串长度。

**💡 创新点**

创新点是将逆 Lyndon 数组与最近大后缀数组及边界校正相结合，并证明边界校正等价于 LCE，从而实现 O(n) 的线性构造。

**🔧 技术方法**

采用了 LCE‑NGS 算法，利用最近大后缀/前大后缀边、LCE 加速以及智能 LCE 机制来实现逆 Lyndon 数组的构造。

**📊 数据集**

实验使用了随机词、结构化合成序列以及真实文本（如 Pizza&Chili、Canterbury 语料库）。

**📈 对比分析**

与标准 Lyndon 数组构造（LCE‑NSS）比较，实验结果显示逆构造的运行时间与标准相近，平均比率约为 1，保持线性时间性能。

**⚠️ 局限性**

局限性在于逆 Lyndon 数组缺乏直接的兼容性性质，不能仅凭最大逆 Lyndon 子串决定后缀排序，需结合额外信息或规则。

---

## 146. Contextual Preference Distribution Learning

**arXiv ID:** 2603.17139 | [PDF](https://arxiv.org/pdf/2603.17139v1)

**作者:** Benjamin Hudson `[一作]` (Mila Quebec Artificial Intelligence Institute), Emma Frejinger `[通讯]` (Universite de Montreal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种顺序学习与优化管线，先学习基于上下文的决策偏好分布，再用于风险厌恶的下游问题，显著降低后决策惊讶。

**💡 创新点**

创新点在于将偏好分布建模为可参数化分布，并用带界限方差的评分函数梯度估计实现最大似然学习，兼顾上下文变化与不确定性。

**🔧 技术方法**

使用整数线性规划、指数族分布、评分函数梯度估计、模拟与CVaR优化等技术。

**📊 数据集**

实验采用合成的打车共享路由与分配环境（随机生成的网格图与LogNormal边成本）。

**📈 对比分析**

与DPO、AIMLE、REINFORCE、MaxEnt IRL等基线对比，平均后决策惊讶下降2.4–114倍，风险厌恶基线下降1.6–25倍。

**⚠️ 局限性**

局限性包括仅在合成数据上验证，假设偏好可用指数族分布描述，且只匹配一阶矩，且只适用于ILP形式的决策问题。

---

## 147. LaDe: Unified Multi-Layered Graphic Media Generation and Decomposition

**arXiv ID:** 2603.17965 | [PDF](https://arxiv.org/pdf/2603.17965v1)

**作者:** Vlad-Constantin Lungu-Stan `[一作]` (Adobe Research), Mariana-Iuliana Georgescu `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个统一的潜在扩散框架（Layered Media Design），能够从简短的文本提示生成可编辑的多层 RGBA 设计、单张图像以及对已有图像进行层级分解，支持任意数量和任意纵横比的层。

**💡 创新点**

核心创新点在于：①将 LLM 作为提示扩展器，自动生成结构化的每层描述；②使用 4D RoPE 位置信息编码，让扩散模型同时处理文本和多层视觉信息；③在训练时利用 RGBA VAE 直接生成带透明通道的层，保证后期可编辑性；④实现单一模型完成文本到层、文本到图像以及图像到层的三大任务。

**🔧 技术方法**

技术方法包括：LLM（Flan‑T5 XXL）提示扩展；Latent Diffusion Transformer 结合 4D RoPE；RGBA Variational AutoEncoder；动态尺寸与层数的批处理策略（bucket, pad, trim）；多尺度训练与多任务学习。

**📊 数据集**

训练集：8 M 矢量图 + 2 M 分层图像 + 80 M 自然图像（内部商业数据）。评估集：Crello 测试集 500 条用户提示，分别提供标题与相应的分层结果。

**📈 对比分析**

与 Qwen‑Image‑Layered、Qwen‑Image‑T2I + Qwen‑Image‑Layered‑I2L 进行对比；在文本到层生成上，VLM‑as‑a‑judge（GPT‑4o mini、Qwen3‑VL）分数从 2.6‑2.8 提升至 3.6‑4.1；在图像到层分解上，PSNR 从 31.6 提升至 32.7，RGB L1 下降，VLM‑as‑a‑judge 分数也提高。整体表现为状态‑of‑the‑art 的文本到层生成以及竞争力的分解质量。

**⚠️ 局限性**

局限性：①高度依赖 LLM 的提示扩展，生成的提示具随机性，影响一致性；②生成大量层时显著消耗显存，限制在低配 GPU 上的可扩展性。

---

## 148. scicode-lint: Detecting Methodology Bugs in Scientific Python Code with LLM-Generated Patterns

**arXiv ID:** 2603.17893 | [PDF](https://arxiv.org/pdf/2603.17893v1)

**作者:** Sergey V. Samsonau `[一作]` `[通讯]` (Authentic Research Partners), Sergey V. Samsonau (Authentic Research Partners)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了scicode-lint，一种基于两层架构的本地LLM linter，用于自动检测科学Python代码中的方法学缺陷（如数据泄露、交叉验证错误和随机种子缺失等）。

**💡 创新点**

创新点在于将模式设计与运行时执行拆分：使用前沿大模型一次性生成可执行的检测问题和测试集，后续在本地小模型上高效并行执行；且所有检测模式均通过自动化评估和自我改进循环生成，减少人工编写和维护成本。

**🔧 技术方法**

核心技术包括：大语言模型（Claude Opus、Claude Sonnet）用于模式生成与验证；本地开源LLM（Qwen3‑8B‑FP8）与vLLM服务器进行高吞吐量推理；结构化JSON输出与Pydantic校验；前端AST与静态分析结合的性能优化；以及多层质量门控与自动化自我改进管道。

**📊 数据集**

评估数据集涵盖：① 3000+ Kaggle笔记本的人工标注泄露数据（预处理泄露、多测泄露等）；② 38篇包含AI/ML的科学论文（PapersWithCode）自包含文件；③ 35篇保留测试论文；④ 50个由LLM生成的测试场景（集成评估）；以及控制实验的66个人工构造的正负样例。

**📈 对比分析**

与现有工具（dslinter、MLScent、mllint、DyLin）比较，scicode-lint在受控测试中达97.7%准确率；在Kaggle泄露检测上实现65%精度、100%召回；在科学论文评估中，反馈集62%精度、持留集54%精度；整体F1在集成评估中为69%。与传统静态分析相比，覆盖范围扩大至66种模式，涵盖大部分方法学缺陷。

**⚠️ 局限性**

局限性包括：仅单文件分析，跨文件问题无法捕获；对小模型（8B）和单GPU的依赖导致召回率受限；LLM判定可能产生非确定性结果且受评判者偏差影响；评估主要基于精度，召回率缺乏充分人类标注；以及模型与库版本演进需要重新生成模式，虽然成本低但仍需额外工时。

---

## 149. DexViTac: Collecting Human Visuo-Tactile-Kinematic Demonstrations for Contact-Rich Dexterous Manipulation

**arXiv ID:** 2603.17851 | [PDF](https://arxiv.org/pdf/2603.17851v1)

**作者:** Xitong Chen `[一作]` (Huazhong University of Science and Technology), Xiaotian Ding `[通讯]` (Wuhan Huaweike Intelligent Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了 DexViTac 系统，实现可携带的人类中心多模态数据采集，收集第一视角视觉、全密度触觉、手部运动学及全局姿态，并基于手部运动学的触觉语义对齐学习框架。

**💡 创新点**

创新点在于整合高密度触觉阵列与手部运动捕捉，提出基于手部运动学的触觉表示学习，消除触觉语义模糊并提升多模态表示的物理一致性。

**🔧 技术方法**

使用 HIT LongLin‑96 触觉阵列、Manus Quantum IMU 手套、GoPro 鱼眼相机、RealSense T265 VIO 进行同步采集；采用对比学习与空间‑时间一致性约束的预训练，并在 ACT（Action Chunking with Transformers）策略中进行端到端训练。

**📊 数据集**

构建了 2,400+ 份 visuo‑tactile‑kinematic 演示、覆盖 10+ 真实环境、40+ 任务的公开大规模数据集。

**📈 对比分析**

与 vision‑only、无预训练、无空间‑时间耦合等基线对比，在滴管、擦白板、笔插入、采果四项任务中成功率均超过 85%，比基线平均提升约 50%。

**⚠️ 局限性**

局限在于固定底座、单臂设置，无法实现长距离或双臂异构协作任务；移动平台适配仍需改进。

---

## 150. Eye image segmentation using visual and concept prompts with Segment Anything Model 3 (SAM3)

**arXiv ID:** 2603.17715 | [PDF](https://arxiv.org/pdf/2603.17715v1)

**作者:** Diederick C. Niehorster `[一作]` (Lund University), Marcus Nyström `[通讯]` (Lund University)

**通讯引用:** 8580 | [OpenAlex ID](https://openalex.org/A5071284106)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

评估并比较了 SAM2 与 SAM3 在眼图分割中的零样本性能，涵盖视觉提示和概念提示两种模式。

**💡 创新点**

首次将最新的 SAM3 与已验证的 SAM2 进行对比，探讨其在实验室与野外眼动数据集上的表现，并评估概念提示在眼图分割中的可行性。

**🔧 技术方法**

使用 Segment Anything Model（SAM2、SAM3）、视觉提示、概念提示；评估指标包括 RMS‑S2S 精度、数据损失、IoU、误报率、漏报率及 Youden's J，并改造 SAM3 代码以支持任意长度视频推理。

**📊 数据集**

实验涉及两套高质量实验室数据（FLEX 1000Hz）和 TEyeD 组（NVGaze、VR、AR、Gaze‑in‑wild、Labelled Pupils、Dikablis），共计约 2.87 M + 14.44 M 帧。

**📈 对比分析**

通过配对 t 检验、IoU 统计、误报/漏报率分析等方法比较，结果显示 SAM2 在视觉提示下无论是实验室还是野外场景均优于 SAM3；SAM3 概念提示表现最差；SAM3 在漏报率上略有优势但误报率高，整体判别度低。

**⚠️ 局限性**

限制包括 SAM3 在视觉提示下表现不如 SAM2、概念提示无法识别虹膜/巩膜、推理速度慢（低于 1 fps）、需进一步微调或蒸馏、只能接受简单名词短语提示，且未实现在线实时分割导致隐私问题。

---

## 151. ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation

**arXiv ID:** 2603.17812 | [PDF](https://arxiv.org/pdf/2603.17812v1)

**作者:** Dmitriy Rivkin `[一作]` (Torc Robotics), Felix Heide `[通讯]` (Princeton University)

**通讯引用:** 6569 | [OpenAlex ID](https://openalex.org/A5059313827)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在高分辨率、长时长视频扩散模型中使用像素级损失的可行性，并提出截断反向传播方法ChopGrad以降低训练内存并实现高效微调。

**💡 创新点**

证明因果视频自编码器的时间局部性导致梯度随时间指数衰减，从而可以截断反向传播而不显著损失性能，并给出理论误差分析。

**🔧 技术方法**

采用截断反向传播（Truncated Backpropagation）、因果缓存（Causal Caching）、像素级感知损失（LPIPS、DISTS、MSE）等技术，在预训练的WAN 2.1、DOVE等模型上进行微调。

**📊 数据集**

使用DL3DV‑Benchmark、HQ‑VSR、Waymo Open Dataset、ROVI以及3D Gaussian Splatting生成的神经渲染数据集进行实验。

**📈 对比分析**

与现有基线（MVSplat‑360、Difix、VACE、原始DOVE、Mirage等）在视频超分、修复、抠图、驾驶视频生成等任务上进行量化和定性对比，ChopGrad在LPIPS、DISTS、FVD、VBench等指标上普遍优于或相当于基线，并显著降低推理时长和内存占用。

**⚠️ 局限性**

仅适用于具有因果缓存的自编码器结构；截断距离需要经验调节，过小可能导致梯度幅度不足；对极端遮挡或长时段极低频信息的效果仍有限。

---

## 152. Automated Grammar-based Algebraic Multigrid Design With Evolutionary Algorithms

**arXiv ID:** 2603.17641 | [PDF](https://arxiv.org/pdf/2603.17641v1)

**作者:** Dinesh Parthasarathy `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Ulrich Rüde `[通讯]` (CERFACS)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

通过语法指导的进化编程（G3P）自动生成适用于 Hypre 的自适应多重网格（AMG）循环，提出了灵活的、非递归的多级循环结构。

**💡 创新点**

创新点在于：① 将 AMG 的算法约束编码为上下文无关文法，使进化搜索保持语法合法并自动遵循 AMG 的结构；② 通过进化搜索探索传统 V、W 循环之外的大规模非标准循环空间，显著提升迭代效率；③ 兼容现有 Hypre 接口，能无侵入地替换默认 AMG，保持软件可扩展性。

**🔧 技术方法**

采用进化算法（遗传编程）与上下文无关文法（CFG）相结合的 G3P 框架；使用 Hypre 的 BoomerAMG 作为执行后端；通过多目标（每次迭代成本、收敛率）评估函数进行选择；在 MPI 并行环境下实现大规模种群评估。

**📊 数据集**

主要实验数据集包括：
- 3D 各向异性泊松方程，参数 c₁,c₂,c₃ ∈ [10⁻⁵,1]³，网格大小 N_d ∈ {100,…,1000}，共计约 10⁶ 变量；
- 4D 线性辐射扩散/电子传导耦合问题（zrad3D），时间步长 1–40，网格 100³–400³，约 10⁶ 变量；
- 通过代理系统（proxy）进行快速评估（例如 N_d=100 的最坏情况）。

**📈 对比分析**

与默认 AMG、以及多种手工调优的 V(1,1) 循环进行对比。实验表明：
- GP 生成的循环在所有配置下均能降低求解时间（最高 1.8× 加速），并减少迭代次数；
- 在弱缩放（N_d 100→1000）和大规模（10⁶→10⁹ 变量）实验中，GP 循环保持优越性能，甚至超过所有手工调优方案；
- 在时间步进的 zrad3D 任务中，GP 预条件器在平均 1.5× 的速度提升和更稳定的迭代次数分布；
- 兼容 Hypre 的混合求解器接口，能够在需要时自动切换到 GP 循环。

**⚠️ 局限性**

局限性包括：
- 仅在扩散主导、对称正定（SPD）问题上验证，复杂耦合或非线性系统的通用性尚未充分评估；
- 进化过程需要数十分钟至半小时的离线计算，成本相对较高；
- 依赖 Hypre 的实现，迁移到 GPU 或其他求解器需要重写文法和接口；
- 仅调优循环结构，未直接调节设置参数（如粗化策略、插值方式），未来可进一步扩展；
- 对极端非结构化网格或高度不均匀稠密矩阵的适应性仍需验证。

---

## 153. MedSAD-CLIP: Supervised CLIP with Token-Patch Cross-Attention for Medical Anomaly Detection and Segmentation

**arXiv ID:** 2603.17325 | [PDF](https://arxiv.org/pdf/2603.17325v1)

**作者:** Thuy Truong Tran `[一作]` (Singapore Management University), Min Hun Lee `[通讯]` (Singapore Management University)

**通讯引用:** 520 | [OpenAlex ID](https://openalex.org/A5020346136)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出一种基于CLIP的监督式医学异常检测与分割框架 MedSAD-CLIP

**💡 创新点**

引入Token‑Patch Cross‑Attention实现文本与图像细粒度交互，并采用Margin Contrastive Loss提升正常/异常表示分离

**🔧 技术方法**

使用CLIP ViT‑L/14、轻量级视觉适配器、可学习文本提示、交叉注意力模块与对比损失

**📊 数据集**

四个医学数据集：Brain（脑肿瘤MRI）、Retina（视网膜水肿OCT）、Lung（肺COVID CT）、Breast（乳腺癌超声）

**📈 对比分析**

与零/少样本CLIP模型和全监督CLIP基线及nnU‑Net、Rolling‑U‑Net对比，MedSAD‑CLIP在Dice和准确率上均显著提升，平均Dice提升约+17%至+46%

**⚠️ 局限性**

仍受限于少量标注数据的可用性，对不同病灶形态的泛化能力需要进一步验证

---

## 154. Only relative ranks matter in weight-clustered large language models

**arXiv ID:** 2603.17917 | [PDF](https://arxiv.org/pdf/2603.17917v1)

**作者:** Borja Aizpurua `[一作]` (Multiverse Computing), Román Orús `[通讯]` (Multiverse Computing)

**通讯引用:** 6304 | [OpenAlex ID](https://openalex.org/A5020314133)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型进行权重聚类，探究权重值的相对排名对模型性能的影响，并验证仅保持排名不变即可保持几乎完整的准确性；

**💡 创新点**

提出权重聚类是研究模型内部结构的有效工具，证明了相对权重排名而非绝对数值是决定性能的关键因素，并界定了安全变换为保持秩且均值方差不变的仿射映射；

**🔧 技术方法**

使用标量K-means聚类、可选的聚类中心微调、以及可选的仿射校正；

**📊 数据集**

主要使用WikiText‑2评估模型困惑度，并在多模型（Llama 3.1‑8B‑Instruct、SmolLM2‑135M、预压缩Llama 3B）上进行实验；

**📈 对比分析**

与GPTQ、AWQ等量化方法以及原始模型对比，聚类后在不训练的情况下即可实现约6.9 GB磁盘、9.32困惑度；再加预压缩与INT4可降至2.6 GB、13.86困惑度，显示在存储与准确性上具有竞争力；

**⚠️ 局限性**

局限包括：仅在加载时重建稠密权重，无法实时节省显存；早层对尺度漂移敏感；一旦破坏排名即无法恢复；并且对大规模层级变换的安全边界尚未完全划定。

---

## 155. AdaMuS: Adaptive Multi-view Sparsity Learning for Dimensionally Unbalanced Data

**arXiv ID:** 2603.17610 | [PDF](https://arxiv.org/pdf/2603.17610v1)

**作者:** Cai Xu `[一作]` (Xidian University), Wei Zhao `[通讯]` (Xidian University)

**通讯引用:** 91025 | [OpenAlex ID](https://openalex.org/A5050699488)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自监督的AdaMuS框架，解决多视角数据维度不平衡问题。

**💡 创新点**

创新点在于引入无参数的主神经元分析(PNA)自适应剪枝和多视角稀疏批归一化(MSBN)稀疏融合，同时通过平衡视角相似性图进行自监督对比学习。

**🔧 技术方法**

技术包括结构化网络剪枝、Wasserstein距离驱动的剪枝率自适应、稀疏批归一化、基于相似性图的对比损失以及一系列基准方法的对照实验。

**📊 数据集**

使用七个真实数据集（UCI、CUB、ORL、MSRCV1、Mfeat、100Leaves、DEAP）和NYUv2语义分割数据集进行评估。

**📈 对比分析**

与13种基线方法（决策级融合、直接拼接、联合表示学习、图结构学习、视角不平衡处理等）对比，AdaMuS在聚类、分类和分割任务上均取得显著优势，尤其在极度不平衡场景中表现尤为突出。

**⚠️ 局限性**

局限性包括对超参数（如稀疏权重λ₂、K邻居数等）敏感，需要手工调节；以及在极端高维视角（如DEAP视频视角）仍可能面临计算资源瓶颈。

---

## 156. LAAF: Logic-layer Automated Attack Framework A Systematic Red-Teaming Methodology for LPCI Vulnerabilities in Agentic Large Language Model Systems

**arXiv ID:** 2603.17239 | [PDF](https://arxiv.org/pdf/2603.17239v1)

**作者:** Hammad Atta `[一作]` (Qorvex Consulting), Jamel Abed `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了LAAF——一个专门针对 agentic LLM 系统中 Logic-layer Prompt Control Injection（LPCI）漏洞的自动化红队框架，能够在六阶段生命周期内系统化地评估并突破这些攻击面。

**💡 创新点**

首次将 49 种技术细分为编码、结构、语义、层叠、触发、外泄等六大类，并引入 Persistent Stage Breaker（PSB）与自适应变异机制，实现从单一攻击到多阶段逐步升级的完整红队流程，填补了现有工具对 LPCI 的功能缺口。

**🔧 技术方法**

利用黑盒 API 调用、提示工程生成 Payload、SHA‑256 去重、阶段顺序驱动的 Mutation Engine、PSB 递归变异，以及多技术组合与自适应策略，构建了闭环式自动化评估体系。

**📊 数据集**

在五个主流 LLM 平台（Gemini‑2.0‑flash、Claude‑3‑haiku、LLaMA3‑70B、Mixtral‑8x7b、ChatGPT‑4o‑mini）上各进行 600 次攻击（100 次/阶段），共 903 次有效请求，评估了 2,822,400 种潜在 Payload 组合。

**📈 对比分析**

与手工单技术基线（1,700 条手工测试）对比，LAAF 在三次独立运行中平均突破率 83%（最高 100%），每平台突破率稳定在 17 个百分点以内；相比基线，LAAF 的尝试次数更少、效率显著提升。

**⚠️ 局限性**

局限包括：未对真实 RAG 或内存接口进行端到端评估、仅覆盖文本场景、缺少多模态与工具 schema poisoning 等扩展向量的测试、未评估防御措施、实验仅在单一时间点与特定模型版本进行、以及缺乏纵向长期重测数据。

---

## 157. VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning

**arXiv ID:** 2603.17720 | [PDF](https://arxiv.org/pdf/2603.17720v1)

**作者:** Tianxing Zhou `[一作]` (Tsinghua University), Tao Jiang `[通讯]` (Galaxea AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出VolumeDP框架，解决机器人视觉模仿中的2D–3D不匹配问题，利用RGB图像学习空间对齐的控制策略；

**💡 创新点**

创新点在于引入体素跨模态注意力将图像特征映射到三维体素空间，再通过自监督的空间令牌生成器提取任务相关体素，最后使用多令牌扩散解码器充分利用三维信息；

**🔧 技术方法**

使用Volume-Image Cross-Attention、可变形注意力、TokenLearner式空间令牌生成、adaLN-Zero注意力的多令牌扩散解码器，以及辅助监督的端效器位置信息；

**📊 数据集**

在LIBERO、ManiSkill、LIBERO-Plus仿真基准以及Galaxea R1 Lite机器人真实任务（碗、微波炉、门、螺母螺钉）上进行实验；

**📈 对比分析**

与BC、Diffusion Policy（DP）和DiT-Block Policy等基线对比，VolumeDP在LIBERO平均成功率达88.8%，比DP高约16.4%，在ManiSkill、LIBERO-Plus和真实任务中同样取得显著提升，鲁棒性更强；

**⚠️ 局限性**

缺点包括需要预先定义体素空间范围且对体素分辨率敏感，且在极大规模场景下体素化与注意力计算仍可能成为瓶颈。

---

## 158. TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos

**arXiv ID:** 2603.17735 | [PDF](https://arxiv.org/pdf/2603.17735v1)

**作者:** Yan Zeng `[一作]` (ShanghaiTech University), Jingyi Yu `[通讯]` (ShanghaiTech University)

**通讯引用:** 9928 | [OpenAlex ID](https://openalex.org/A5101500646)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了TAPESTRY框架，通过在视频扩散模型中注入显式几何条件来生成高保真、几何一致的旋转视频，并利用多阶段、3D感知的修补管线完成完整纹理；

**💡 创新点**

创新点在于将多模态几何约束（法线、位置映射）与视频扩散模型结合，形成像素级的几何引导，并设计了上下文感知的后续修补技术，实现了全覆盖且无缝纹理；

**🔧 技术方法**

采用基于DiT的视频扩散模型、LoRA参数高效微调、VAE编码、CLIP/UMT5文本/视觉编码、几何融合模块以及后处理的3D感知修补；

**📊 数据集**

使用约12万条合成旋转视频数据集（从Objaverse中采样30K三维模型，生成61帧视频后扩充至120K），并在单台DGX Spark上训练；

**📈 对比分析**

与基线（控制视频、Diffusion as Shader等）比较，TAPESTRY在PSNR、SSIM、LPIPS、FVD等指标上均有显著提升，同时纹理生成在FID、KID、CLIP分数上领先，用户研究也显示优于其他方法；

**⚠️ 局限性**

主要局限是对输入网格质量高度依赖，且生成的材质灯光固定，无法像环境贴图一样进行重新照明。

---

## 159. Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress

**arXiv ID:** 2603.17312 | [PDF](https://arxiv.org/pdf/2603.17312v1)

**作者:** Yuelin Zhang `[一作]` (Renmin University of China), Wenbing Huang `[通讯]` (Renmin University of China)

**通讯引用:** 8532 | [OpenAlex ID](https://openalex.org/A5032642601)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了Recurrent Reasoning Vision‑Language Model（R2VL）用于长时序任务进度估计，利用链式思考（CoT）记录任务分解、关键步骤及完成状态，并通过递归推理仅处理局部视频片段以保持全局上下文；

**💡 创新点**

① 递归推理框架仅处理局部视频片段并通过CoT维持全局记忆；② CoT结构化记录任务分解与完成状态，提升推理可解释性；③ 结合强化学习（PPO）多回合自我纠错，优化进度估计；④ 自动化数据生成pipeline构建大规模训练集；

**🔧 技术方法**

基于大型VLM（Qwen2.5‑VL‑7B‑Instruct）实现的链式推理+CoT；使用PPO强化学习、自动CoT生成、进度估计损失与奖励设计；采用多任务评估指标（MAE、ΔpMAE、bin、acc）；

**📊 数据集**

ALFRED（仿真任务）和Ego4D（真实世界）两大数据集，自动生成长时序视频片段与对应CoT；

**📈 对比分析**

与闭源API（GPT‑5、Gemini‑2.5‑Pro、Qwen3‑VL‑Plus）、开源VLMs（MiniCPM、GLM、InternVL、Qwen2.5）以及专门方法（LIV、ROVER、GVL‑SFT）在ALFRED与Ego4D上对比，R2VL在所有四个指标上均超越基线；在ALFRED上MAE<3%，bin/acc>0.9；在Ego4D上虽略逊，但仍保持领先；

**⚠️ 局限性**

在真实环境Ego4D的性能下降（噪声、执行路径多样性）；对CoT生成质量依赖自动脚本，可能存在误差；目前仅支持局部片段递归推理，对极长时序或更复杂情境仍有挑战；

---

## 160. Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations

**arXiv ID:** 2603.17305 | [PDF](https://arxiv.org/pdf/2603.17305v1)

**作者:** Haozheng Luo `[一作]` (Northwestern University), Yan Chen `[通讯]` (University of Michigan)

**通讯引用:** 14639 | [OpenAlex ID](https://openalex.org/A5100378075)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CRAFT 框架，通过对大型推理模型的隐藏表示进行对比学习与强化学习，实现在推理轨迹层面上的安全对齐；

**💡 创新点**

创新点在于将隐空间对比学习与 RL 相结合，构建安全子空间并利用潜在-文本一致性奖励消除“表面安全对齐”；

**🔧 技术方法**

使用了隐空间对比学习（LCLR）、GRPO 强化学习、潜在语义奖励、潜在-文本一致性奖励以及外部安全评估器；

**📊 数据集**

在 Qwen3‑4B‑Thinking 与 DeepSeek‑R1‑Distill‑Llama‑8B 两个大型推理模型上，使用 JailbreakBench、StrongReject 等对抗性数据集进行训练和评估；

**📈 对比分析**

与 IPO、SafeKey、SafeChain 等基线对比，平均提升推理轨迹安全率 79.0%、最终响应安全率 87.7%，并在数学与代码生成任务中保持甚至提升 4.7% 的性能；

**⚠️ 局限性**

局限性包括对固定安全评估器的依赖、训练预算受限导致潜在空间对齐不够完善，以及可能导致过度拒绝或对新任务的泛化能力受损。

---

## 161. SCALE:Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction

**arXiv ID:** 2603.17380 | [PDF](https://arxiv.org/pdf/2603.17380v1)

**作者:** Shuizhou Chen `[一作]` (Shanghai Artificial Intelligence Laboratory), Zhangyang Gao `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一个端点对齐的条件传输框架，用于在单细胞扰动数据上直接预测扰动后的细胞群体分布，并通过端点监督实现稳定的学习。

**💡 创新点**

创新点包括：
• 端点对齐的 JiT（Just-in-Time）传输学习，避免传统流模型中对无监督轨迹的依赖；
• 采用层次化的 set‑aware 编码器，先用 LLaMA‑style attention 对单细胞基因表达进行编码，再用 DeepSets 对细胞集合进行聚合；
• 通过种子注意力（seed‑attention）实现条件信息的可学习聚合；
• 在 BioNeMo 框架中实现分布式训练与高效推理，显著提升训练速度和吞吐量；
• 统一 Cell‑Eval 评价协议，确保与现有基准（STATE、PerturbDiff 等）对齐。

**🔧 技术方法**

技术实现：LLaMA‑style attention + DeepSets 层次编码；JiT 条件流匹配（endpoint prediction + endpoint loss）；MMD+MSE 自编码器正则化；条件注入的 seed‑attention；BioNeMo 分布式训练、LMDB 分片、批量采样；Cell‑Eval 评估脚本。

**📊 数据集**

数据集：
• PBMC（细胞因子扰动，12 个 donor，90 条扰动）；
• Tahoe‑100M（化学扰动，50 条癌细胞系，1100+ 处理条件，1 亿+ 细胞）；
• Replogle‑Nadig（CRISPRi 基因扰动，4 条人细胞系，2024 条基因扰动）。

**📈 对比分析**

比较方法：在 Cell‑Eval 评价框架下与 STATE、PerturbDiff、CPA、Linear、CellFlow、Squidiff 等基线对比。结果显示：
• 在 Tahoe‑100M 上 PDCorr 提升 12.02%，DE Overlap 提升 10.66%；
• 在 PBMC 上 PDCorr 0.979，DE Overlap 0.810；
• 在 Replogle 上 PDCorr 0.909，DE Overlap 0.601；
• 训练速度相较 STATE 提升 12.51×，推理吞吐量提升 1.29×；
• MSE/MAE 由于重建误差关注全部基因，略高于基线，但生物学指标明显优于对手。

**⚠️ 局限性**

局限性：
• 依赖仅有的端点监督，可能导致模型学习到“短路”解，缺乏对完整轨迹的捕捉；
• 在更大参数规模下（280M）性能无显著提升，说明模型已在可用数据上达到饱和；
• Cell‑Eval 评估对实现细节敏感，跨实现差异可能影响可重复性；
• 对技术噪声和批量效应的鲁棒性仍待进一步验证；
• 目前的条件编码聚合需要手工选择（seed‑attention），对不同扰动类型的泛化潜能有限。

---

## 162. Abstraction as a Memory-Efficient Inductive Bias for Continual Learning

**arXiv ID:** 2603.17198 | [PDF](https://arxiv.org/pdf/2603.17198v1)

**作者:** Elnaz Rahmati `[一作]` (University of Southern California), Morteza Dehghani `[通讯]` (University of Southern California)

**通讯引用:** 4037 | [OpenAlex ID](https://openalex.org/A5065952016)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种在线持续学习框架，通过在损失层引入实例与抽象联合优化，降低梯度干扰，实现无重放缓冲的知识保留与泛化。

**💡 创新点**

创新点在于提出了抽象增强训练（AAT）机制以及两套新基准——Relational Cycle Benchmark与Narrative Abstraction Benchmark，用以系统评估抽象对在线学习的贡献。

**🔧 技术方法**

技术手段包括：在训练时同时输入具体实例与其抽象化表示（如实体掩蔽或格言抽象），使用加权联合损失并配合局部重放；模型采用大语言模型（Qwen、SmolLM）进行实验。

**📊 数据集**

使用的数据集为：Relational Cycle Benchmark（基于知识图谱的实体循环图，包含遮蔽实体的关系子图）和Narrative Abstraction Benchmark（格言与对应故事的抽象结构对）。

**📈 对比分析**

与传统单实例训练和经验回放（Buffer大小50/100）进行对比，AAT在不使用任何记忆缓冲的情况下，累计准确率略高或相当于最佳回放配置，且在未知关系上显著提升准确率，遗忘率显著下降。

**⚠️ 局限性**

局限性包括：实验仅在中小规模模型上验证，缺乏对更大模型的扩展评估；抽象模式主要为显式重复模式，未探索隐式或更复杂的结构；超参数（α、n）为固定值，缺乏自适应机制。

---

## 163. Multi-stage Flow Scheduling for LLM Serving

**arXiv ID:** 2603.17456 | [PDF](https://arxiv.org/pdf/2603.17456v1)

**作者:** Yijun Sun `[一作]` (Hong Kong University of Science and Technology), Kai Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 23056 | [OpenAlex ID](https://openalex.org/A5100438001)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种面向大语言模型（LLM）推理的多阶段流调度框架，旨在提升首次令牌时间（TTFT）的服务级别目标（SLO）达成率。

**💡 创新点**

核心创新在于将全局TTFT截止时间逐步转换为各阶段流的显式紧迫度，借助“Defer‑and‑Promote”策略和反向多级队列（RMLQ）实现近似最小余量优先（LLF）调度，而不需要精确的剩余紧迫度估计。

**🔧 技术方法**

技术上结合了最小链路利用率（MLU）阈值、相对层索引（RLI）推断内请求紧迫度、稳健的批级紧迫度指标（RED）、以及对多阶段通信（KV缓存检索、集合通信、Prefill‑to‑Decode转移）的分层优先级分配；实现为可插拔模块集成到NCCL、Mooncake与vLLM推理引擎。

**📊 数据集**

在实验中使用了QwenA‑Conv、QwenB‑Agent 生产工作负载，MoE模型 Mixtral、Grok‑2、DBRX、Qwen3‑Coder，及长上下文 Llama3‑8B 的 Sequence Parallelism。

**📈 对比分析**

与 Fair Sharing、SJF、EDF、Karuna 等基线对比，所提框架在所有模型与工作负载下均实现了 1.2×–2.4× 的 TTFT SLO 达成率提升，并在多阶段通信完成时间上显著缩短（约30–70%）。

**⚠️ 局限性**

局限性包括：对多模型部署与混合并行策略的评估不足；在极端拥塞或网络负载波动时对 MLU/RED 估计的鲁棒性待进一步验证；以及对规模化升级网络（如 NVLink）适配性尚未完全验证。

---

## 164. Noncooperative Human-AI Agent Dynamics

**arXiv ID:** 2603.16916 | [PDF](https://arxiv.org/pdf/2603.16916v1)

**作者:** Dylan Waldner `[一作]`, Mitchelle Ashimosi `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究了人工智能代理与人类决策者在非合作博弈中的交互，利用前景理论建模人类，期望效用建模AI，并通过数值仿真评估不同组合和参考点对策略收敛的影响。

**💡 创新点**

将前景理论与多智能体强化学习相结合，构造混合人类- AI 人口并系统评估其学习动态；提出基于前景理论的Q学习与参考点更新机制；揭示了人类- AI博弈中出现的异常混合均衡和策略变异。

**🔧 技术方法**

多智能体Q学习、平均奖励形式、前景理论概率加权与损失厌恶函数、EMA更新的参考点与信念、基于Markov游戏的非合作RL框架。

**📊 数据集**

经典2×2博弈矩阵（囚徒困境、匹配硬币、两性之争、猎鹿、老鹰-小鸡）以及前景理论引发的路径性游戏（Ochs’游戏、Crawford's Counterexample）。

**📈 对比分析**

在六种人口组合、三种参考点模型和两种状态历史长度上进行500轮50000步的实验；比较AH-AH基准与各组合的策略频率、CPT/EU L2距离、奖励均值，发现大部分对齐但也出现显著偏差；性能表现：AI与LH在多数游戏中无明显收益差异，异常行为导致收益相似。

**⚠️ 局限性**

实验仅限于2×2矩阵且离散策略，未考虑更复杂状态空间；前景理论参数固定，缺乏个体差异；Q学习噪声和收敛性理论缺失，未给出对长期稳定性的严格分析。

---

## 165. CodeScout: An Effective Recipe for Reinforcement Learning of Code Search Agents

**arXiv ID:** 2603.17829 | [PDF](https://arxiv.org/pdf/2603.17829v1)

**作者:** Lintang Sutawika `[一作]` (Carnegie Mellon University), Graham Neubig `[通讯]` (OpenHands)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于强化学习的开源代码搜索代理训练方法，只使用终端工具即可完成代码定位。

**💡 创新点**

创新点在于无需语言特定的静态分析工具，利用RL奖励设计和简单的bash工具实现高效定位。

**🔧 技术方法**

主要技术包括OpenHands‑Bash框架、SkyRL、GSPO、奖励函数基于F1得分以及辅助终止奖励。

**📊 数据集**

使用了SWE‑Bench（Verified、Lite、Pro）数据集中的Python仓库及GitHub issue修复补丁。

**📈 对比分析**

与多种基线（LocAgent、RepoNavigator、CoSIL等）和闭源模型对比，模型在所有基准上均取得与更大模型相当或更优的F1，尤其在文件/模块/函数定位上明显领先。

**⚠️ 局限性**

局限性包括对非Python语言支持不足、需要手工构建奖励和数据集、以及对前沿闭源模型仍有一定差距。

---

## 166. Actionable Guidance Outperforms Map and Compass Cues in Demanding Immersive VR Wayfinding

**arXiv ID:** 2603.17238 | [PDF](https://arxiv.org/pdf/2603.17238v1)

**作者:** Apurv Varshney `[一作]` (University of California Santa Barbara), Michael Beyeler `[通讯]` (University of California Santa Barbara)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

在房间规模VR导航任务中比较了三种常用导航辅助：方向箭头、最小地图和罗盘。

**💡 创新点**

首次在相同沉浸式环境下直接对比三种导航方式，证明可直接行动的箭头在高压、低视野条件下最优。

**🔧 技术方法**

使用HTC Vive Pro Eye无线HMD、Tobii眼动仪、Unity实现的室内迷宫，并在实验中记录路径、时间、眼动、负荷等指标。

**📊 数据集**

数据来自42名受试者共1008次试验，包含目标点、障碍、环境噪音等。

**📈 对比分析**

通过线性混合效模型比较三种辅助，箭头得到最高的导航性能分数（p<.001），完成时间最快，耗时与干扰最小；最小地图次之，罗盘最低。

**⚠️ 局限性**

局限在于实验环境有限、最小地图采用北向固定，未覆盖更大或更复杂的真实空间，且未检验自适应提示等。

---

## 167. WebPII: Benchmarking Visual PII Detection for Computer-Use Agents

**arXiv ID:** 2603.17357 | [PDF](https://arxiv.org/pdf/2603.17357v1)

**作者:** Nathan Zhao `[一作]` (Stanford University), Nathan Zhao `[通讯]` (Stanford University)

**通讯引用:** 538 | [OpenAlex ID](https://openalex.org/A5032094136)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为 WebPII 的合成基准数据集，并基于该数据集训练了 WebRedact 模型，用于在网页界面上实时检测并遮蔽个人身份信息（PII）。

**💡 创新点**

创新点在于：①扩展了 PII 词表，加入交易级别标识符；②支持预测式检测，能在表单部分填写时即时识别 PII；③采用 VLM 驱动的 UI 重现与自动注释管道，实现大规模、可扩展的合成数据生成。

**🔧 技术方法**

技术手段包括：利用大语言模型（Claude Opus 4.5）将真实网页转化为可运行的 React 组件；在渲染时注入数据并通过 DOM 查询自动提取像素级边框；对生成的图像训练基于 YOLO 的实时检测网络 WebRedact；对比时使用 OCR+Presidio、LayoutLMv3+GPT-4o-mini 等文本级方法。

**📊 数据集**

使用的数据集为 WebPII，包含 44,865 张电商 UI 图像、993,461 个细粒度边框标注，涵盖 10 家电商品牌、19 种页面类型，且每张图像提供完整、部分填写及空表单三种状态。

**📈 对比分析**

在 Test_Cross-Company 评测中，WebRedact 以 0.753 的 mAP@50 远超最佳文本基线 0.357，且单帧推理时间仅 20 ms；相比之下 OCR+Presidio 需要 1.3 s，LayoutLMv3+GPT-4o-mini 需要 2.9 s，说明模型在速度与精度上均有显著提升。

**⚠️ 局限性**

局限性包括：仅覆盖英文电商界面，无法直接迁移至其他语言或行业；数据为静态图像，缺少滚动、动画等视频场景；对产品和交易信息的过度标记可能导致误遮蔽；以及在极端动态 UI 或复杂交互中的泛化仍需进一步验证。

---

## 168. Learning Transferable Temporal Primitives for Video Reasoning via Synthetic Videos

**arXiv ID:** 2603.17693 | [PDF](https://arxiv.org/pdf/2603.17693v1)

**作者:** Songtao Jiang `[一作]` (Zhejiang University), Zuozhu Liu `[通讯]` (Zhejiang University)

**通讯引用:** 1141 | [OpenAlex ID](https://openalex.org/A5024343415)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了SynRL框架，利用程序化生成的合成视频训练VLM进行时序推理。

**💡 创新点**

创新点在于以可验证的合成数据学习时间原语，并通过两阶段SFT+RL实现高数据效率。

**🔧 技术方法**

使用程序化视频生成、Chain-of-Thought生成与验证、RL（GRPO）和SFT。

**📊 数据集**

使用7.7K CoT样本、7K RL样本合成数据，+1K真实LLaVA-Video样本。

**📈 对比分析**

与Video-R1、Video-Jigsaw等对比，在15个视频基准上提升12.6% RexTime、4.6% TOMATO等，21倍数据效率。

**⚠️ 局限性**

局限在合成场景过于简单、对复杂真实动作的泛化仍受限，依赖程序化代码生成。

---

## 169. EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding

**arXiv ID:** 2603.18001 | [PDF](https://arxiv.org/pdf/2603.18001v1)

**作者:** Kai Zou `[一作]` (University of Science and Technology of China), Bin Liu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 83893 | [OpenAlex ID](https://openalex.org/A5100395468)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 EchoGen，统一布局到图像生成和图像定位的框架，利用两任务的互补性实现更精准的布局控制与定位。

**💡 创新点**

创新点在于分阶段联合训练：先并行多任务预训练共享视觉 token，然后通过 Dual Joint Optimization 将生成与定位串联成 L–I–L 循环，最后用 Cycle RL 在无视觉监督下自监督优化。

**🔧 技术方法**

使用自回归 Transformer、Gumbel‑Softmax 近似采样、GRPO 强化学习以及基于布局–图像–布局循环的奖励机制。

**📊 数据集**

主要数据集包括 MS‑COCO、LayoutSAM‑Eval、Ref‑L4 以及 GRIT‑20M 用于预训练。

**📈 对比分析**

与 GLIGEN、MIGC、PlanGen 等基线相比，EchoGen 在 MS‑COCO 上 AP 提升约 7.3 点，FID 降低至 20.12；在 Ref‑L4 上 mAcc 提升至 68.46，表现为现有 SOTA。

**⚠️ 局限性**

局限性包括对 Stage‑1 预训练质量敏感，RL 阶段需要足够好的生成质量才能获得有效奖励，且整体模型参数量相对较大。

---

## 170. MALLES: A Multi-agent LLMs-based Economic Sandbox with Consumer Preference Alignment

**arXiv ID:** 2603.17694 | [PDF](https://arxiv.org/pdf/2603.17694v1)

**作者:** Yusen Wu `[一作]`, Xiaotie Deng `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了多智能体大型语言模型经济沙盒（MALLES），通过跨类别后训练实现经济对齐、使用均值场稳定机制和多轮对话分工来模拟零售和批发消费者决策。

**💡 创新点**

创新点包括：①跨类别交易记录的后训练对齐提升稀缺类别的泛化；②均值场模型实现宏观环境与微观决策的动态耦合；③多智能体讨论框架通过结构化对话分摊注意力并提升解释性。

**🔧 技术方法**

核心技术为大型语言模型（GPT、Gemini、DeepSeek、Llama‑4 等）结合后训练、均值场抽样、注意力控制、符号回归与多智能体对话策略。

**📊 数据集**

使用真实零售交易数据集，涵盖 119,252 位客户、3,361 类别的产品与交易记录，并构建多模态产品描述（文本、图片、数值）。

**📈 对比分析**

与 FinCon、Abides‑Economist 等现有 LLM 经济仿真基线比较，MALLES 在商品选择命中率（0.775）和稳定性指标上显著优于基线，量化预测误差保持在可接受范围内。

**⚠️ 局限性**

局限性在于跨模态对齐深度不足、宏观-微观耦合机制不够细致、以及对个人数据隐私与操控风险的伦理监管需要进一步加强。

---

## 171. Adversarial attacks against Modern Vision-Language Models

**arXiv ID:** 2603.16960 | [PDF](https://arxiv.org/pdf/2603.16960v1)

**作者:** Alejandro Paredes La Torre `[一作]` `[通讯]` (Duke University), Alejandro Paredes La Torre (Duke University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在模拟的电子商务环境中，对LLaVA‑v1.5‑7B和Qwen2.5‑VL‑7B两种开源视觉语言模型代理进行对抗攻击评估，使用BIM、PGD和CLIP频谱攻击方法，测量攻击成功率。

**💡 创新点**

首次系统比较了不同开源VLM家族在白盒梯度攻击下的鲁棒性差异，发现CLIP频谱攻击在跨模型迁移上更有效，并揭示了架构差异导致的鲁棒性不同。

**🔧 技术方法**

使用白盒梯度攻击（BIM、PGD）和基于CLIP的频谱攻击，结合Flask、Selenium和自建Inference服务器的红队测试框架。

**📊 数据集**

在自建的电子商务红队环境中使用产品图片（630次试验每种攻击），未使用公开数据集。

**📈 对比分析**

通过对比攻击成功率（ASR）和正常购买率，发现LLaVA的ASR高达66.9%，而Qwen2.5‑VL仅为15.5%，表明后者在三种攻击下显著更稳健。

**⚠️ 局限性**

仅评估了两种VLM家族，结果不一定能推广到其他架构；实验环境为自建的模拟环境，无法验证在真实生产部署中的效果。

---

## 172. Requirements Volatility in Software Architecture Design: An Exploratory Case Study

**arXiv ID:** 2603.17648 | [PDF](https://arxiv.org/pdf/2603.17648v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 173. Process Supervision for Chain-of-Thought Reasoning via Monte Carlo Net Information Gain

**arXiv ID:** 2603.17815 | [PDF](https://arxiv.org/pdf/2603.17815v1)

**作者:** Corentin Royer `[一作]` (IBM), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**通讯引用:** 1961 | [OpenAlex ID](https://openalex.org/A5020415668)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建基于信息理论的自动步骤级标注方法（Monte Carlo Net Information Gain）用于训练过程奖励模型，以提升链式推理的最佳‑K 选择性能。

**💡 创新点**

①提出MCNIG，线性 O(N) 的信息增益标注；②将正确/错误答案对比用于生成可靠步骤标签；③将此方法首次应用于代码生成与 SQL 等多领域，提升 PRM 效果。

**🔧 技术方法**

信息理论（信息增益）+ Monte Carlo 采样；链式推理生成；验证器函数；PRM 与 ORM 训练；best‑of‑K 重选；跨域评估。

**📊 数据集**

MATH、GSM8K、AIME、HumanEval、BigCodeBench、BIRD、PubMedQA、UGPhysics、ProcessBench 等八个基准。

**📈 对比分析**

与 majority voting、ORM、MathShepherd、OVM、ImplicitPRM、QwenPRM 等多种基线对比；在大多数任务上 PRM‑MCNIG 超越基线，best‑of‑K 精度提升 3–10%；在 OOD UGPhysics 也取得最佳成绩。

**⚠️ 局限性**

仅评估推理时间重选，未结合 RL 或策略优化；仅处理文本推理任务，未扩展到多模态或工具使用的环境。

---

## 174. Proactive Knowledge Inquiry in Doctor-Patient Dialogue: Stateful Extraction, Belief Updating, and Path-Aware Action Planning

**arXiv ID:** 2603.17425 | [PDF](https://arxiv.org/pdf/2603.17425v1)

**作者:** Zhenhai Pan `[一作]` (Hong Kong Polytechnic University), Jia You `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 11468 | [OpenAlex ID](https://openalex.org/A5045562050)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出并实现了一种在医生‑患者对话中主动知识询问的 EMR 生成框架，将对话过程视为部分可观测的询问循环。

**💡 创新点**

创新点在于引入状态化提取、连续信念更新、缺口信号、混合检索与 POMDP-lite 行动规划的联合模型。

**🔧 技术方法**

采用状态化信息提取、贝叶斯信念更新、向量+对象级检索、图路径推理以及基于信息增益和风险评估的 POMDP-lite 控制器。

**📊 数据集**

使用了十个标准化多轮对话脚本（4胸痛、3腹痛、3风险敏感场景）及其黄金注释，外加300个检索查询。

**📈 对比分析**

与四个基线（直接生成、chunk‑only RAG、规则模板问答、完整框架）对比，实验显示框架在覆盖率83.3%、风险召回80%、结构完整度81.4%等指标上优于基线，并在冗余度与完成时间上表现更好。

**⚠️ 局限性**

局限在于样本规模小、场景模拟、未进行真实临床验证、信念状态近似、检索器依赖解析质量、缺乏置信区间和多样本方差评估。

---

## 175. Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients

**arXiv ID:** 2603.17809 | [PDF](https://arxiv.org/pdf/2603.17809v1)

**作者:** Ziwei Xiang `[一作]` (Chinese Academy of Sciences), Xu-Yao Zhang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 5619 | [OpenAlex ID](https://openalex.org/A5082548671)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了基于量化感知积分梯度(QIG)的细粒度后训练量化方法，能够对视觉语言模型中的每个输入标记进行敏感度评估并指导量化

**💡 创新点**

创新点在于将积分梯度与量化误差结合，得到标记级的量化误差归因，突破了传统基于模态的粗粒度量化方法

**🔧 技术方法**

主要技术包括量化感知积分梯度(QIG)、IQR裁剪、基于token的重要性权重的通道级等比例缩放以及与现有PTQ框架（如MBQ、GPTQ）融合的改进

**📊 数据集**

使用COCO Caption作为校准集，评估数据涵盖VizWiz、MMMU、ChartQA、AI2D、ScienceQA等视觉语言基准

**📈 对比分析**

与RTN、AWQ、GPTQ、SmoothQuant、MBQ等方法对比，在W3A16和W4A8两种量化格式下，QIG在多模型、多基准上平均提升0.5–1.6%，在LLaVA-onevision-7B 3-bit权重量化中提升1.6%，接近全精度表现

**⚠️ 局限性**

局限性包括仍为后训练量化，无法进一步压缩模型；对不同量化方案的适用性需进一步验证；在极低位宽（如1-2bit）下性能下降仍待改进

---

## 176. Pixel-level Counterfactual Contrastive Learning for Medical Image Segmentation

**arXiv ID:** 2603.17110 | [PDF](https://arxiv.org/pdf/2603.17110v1)

**作者:** Marceau Lafargue-Hauret `[一作]`, Ben Glocker `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了结合反事实生成与像素级密集对比学习的多视角方法（DVD-CL、MVD-CL 及其监督变体），并开发了 CHRO-map 可视化用于肺部分割任务。

**💡 创新点**

创新点在于：① 将反事实生成的多视角数据用于像素级对比学习，提升对扫描器、病变等因素的鲁棒性；② 设计了监督版对比学习，将银标准标签融入像素对的正负关系；③ 引入 CHRO-map，利用 UMAP 与 HSV 映射直观展示嵌入空间的结构。

**🔧 技术方法**

使用的技术包括：NT-Xent 对比损失、Hierarchical VAE 生成反事实、结构因果模型 (SCM)、U-Net+ResNet50 编码器-解码器、UMAP 降维与 HSV 颜色映射。

**📊 数据集**

实验数据集为公开的 PadChest 胸片集（约 60k 训练图像、17k 验证图像），银标准标签来源于 CheXMask，用于监督版本；另外使用 70 张人工标注的胸片进行微调与验证。

**📈 对比分析**

与 SimCLR、VADeR、SSDCL 等基准方法对比：无监督版 DVD-CL 在 Dice、HD95、ASD 上优于 SimCLR/VADeR；监督版 S-MVD-CL 在手工标注样本上平均 Dice 约 94%，超过仅使用 CheXMask 预训练的模型，且在不同折中表现出更低的方差，说明鲁棒性提升。

**⚠️ 局限性**

主要局限包括：① 反事实生成依赖已知因果图并假设可识别性，实际应用中可能不满足；② 仅在 2D 胸片上验证，缺乏对 3D 成像或半监督场景的探索；③ 对抗样本、复杂采样策略等问题仍待进一步研究。

---

## 177. A Single-Fiber Optical Frequency Domain Reflectometry (OFDR)-Based Shape Sensing of Concentric Tube Steerable Drilling Robots

**arXiv ID:** 2603.17990 | [PDF](https://arxiv.org/pdf/2603.17990v1)

**作者:** Yash Kulkarni `[一作]` (University of Texas at Austin), Farshid Alambeigi `[通讯]` (University of Texas at Austin)

**通讯引用:** 1591 | [OpenAlex ID](https://openalex.org/A5055294307)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并验证了一种基于光频域反射测量（OFDR）的形状感知组件（SSA），将其集成到可钻孔的同心管可弯曲钻机（CT‑SDR）的内部空管中，完成了自由弯曲和在合成Sawbone骨样材中的钻孔实验；

**💡 创新点**

创新点在于：①采用OFDR实现纤维全长连续应变测量，克服FBG离散节点的空间分辨率限制；②将SSA与钻机内部空管同轴布置，避免了传统表面粘贴或切槽的复杂加工，简化制造并提升装配可靠性；

**🔧 技术方法**

主要技术包括：光频域反射测量（OFDR）传感、NiTi平面线性基底的SSA制备与校准、可弯曲同心管钻机（含预成形NiTi管和可弯曲钻机）、七自由度KUKA机械臂、C‑Arm射线摄影用于地面真值、基于应变-曲率关系的形状重建算法；

**📊 数据集**

实验使用合成Sawbone骨样材，搭配不同半径（39 mm、50 mm、117 mm）预成形NiTi管的四条轨迹（直线+三条J形），通过OFDR获取应变数据并进行形状重构；

**📈 对比分析**

通过与文献中基于FBG或嵌入式OFDR传感的同心管钻机进行对比，使用平均尖端误差和平均形状误差两项指标评估性能，结果显示自由弯曲实验尖端误差≤1.32 mm、形状误差≤0.41 mm（归一化误差≤1.5%），钻孔实验尖端误差≤1.73 mm、形状误差≤0.44 mm（归一化误差≤4%），明显优于先前报告的6%误差水平；

**⚠️ 局限性**

局限性包括：SSA与钻机空管间的尺寸间隙导致应变灵敏度下降，导致曲率测量更为线性而非急剧；钻机扭转段弹性可能使测得曲率被拉伸，产生离轴误差；实验仅在合成骨样材上完成，需在真实动物或人类骨骼中进一步验证；

---

## 178. Real-Time Online Learning for Model Predictive Control using a Spatio-Temporal Gaussian Process Approximation

**arXiv ID:** 2603.17632 | [PDF](https://arxiv.org/pdf/2603.17632v1)

**作者:** Lars Bartels `[一作]` (Institute for Dynamic Systems and Control), Melanie N. Zeilinger `[通讯]` (Institute for Dynamic Systems and Control)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个可实时在线学习的近似时空高斯过程模型，用于学习系统残差动力学并提升基于模型预测控制的性能。

**💡 创新点**

将空间诱导点GP与时间马尔可夫状态空间模型相结合，形成可在MPC中以常数复杂度递归更新的时空GP，实现对时变扰动的持续学习而不牺牲实时性。

**🔧 技术方法**

采用可分离的空间RBF核与半整数马尔可夫时间核，构建状态空间表示；利用卡尔曼滤波（含平方根形式）实现递归学习；结合零阶SQP的GP‑MPC算法，并在Python中实现，集成至l4acados框架。

**📊 数据集**

在微型赛车实验平台CRS上收集实时轨迹与扰动数据（时间变化的中立转向偏移），作为训练与评估数据集。

**📈 对比分析**

通过与基准MPCC、精确GP（SoD截断400点）以及仅空间诱导点GP的对比实验，结果表明近似时空GP在实时性（约30 ms）、预测误差降低以及圈速/轨迹跟踪性能方面均优于基准。

**⚠️ 局限性**

对极高频或快速变化的扰动仍有限制；需要预先设定诱导点数量和超参数；在时变速度过快时，诱导点可能无法完全捕捉信息，导致预测误差上升。

---

## 179. Specification-Aware Distribution Shaping for Robotics Foundation Models

**arXiv ID:** 2603.17969 | [PDF](https://arxiv.org/pdf/2603.17969v1)

**作者:** Sadık Bera Yüksel `[一作]` (Northeastern University), Derya Aksaray `[通讯]` (Northeastern University)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5053550436)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种在预训练机器人基础模型执行过程中通过约束优化动态调整动作分布，从而强制满足Signal Temporal Logic（STL）约束的方法。

**💡 创新点**

创新点包括：①将STL约束转化为可解的KL最小化约束优化问题并给出闭式解；②利用前向动力学传播和基于漏斗奖励的RL策略对未来轨迹做多步评估，满足时间窗口、顺序与安全约束；③实现无需重新训练基础模型即可保证规范满足。

**🔧 技术方法**

使用的技术包括：Signal Temporal Logic (STL) 规范、Kullback–Leibler (KL) 距离约束优化、漏斗奖励塑形的深度强化学习（DQN）生成 STL 合规策略、基于前向动力学的轨迹评估、以及在机器人基础模型上的动作分布微调。

**📊 数据集**

使用的数据集/环境为 AI2-THOR 3D 家居模拟环境以及其对应的 SPOC 机器人基础模型，构造了多个房屋布局与对象配置进行仿真。

**📈 对比分析**

对比方法：在200次独立仿真跑中将提出的方法与未改动的SPOC模型进行对比。STL 满足率从 0%（未改动）提升到 100%，主任务成功率略降（从 93.5%–99% 降至 82.5%–92.5%）。说明该方法能够保证规范满足，同时对主任务性能影响有限。

**⚠️ 局限性**

局限性：①需要已知的动力学模型和粗略的结构地图，且假设无模型误差；②在确定性环境下才有保证，随机噪声会导致失效；③多步前向仿真在较长时域或复杂动力学下计算成本高；④对主任务的修改可能导致成功率下降，需进一步权衡。

---

## 180. CLeAN: Continual Learning Adaptive Normalization in Dynamic Environments

**arXiv ID:** 2603.17548 | [PDF](https://arxiv.org/pdf/2603.17548v1)

**作者:** Isabella Marasco `[一作]` (University of Bologna), Michele Colajanni `[通讯]` (University of Bologna)

**通讯引用:** 5350 | [OpenAlex ID](https://openalex.org/A5073709427)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在连续学习中处理表格数据归一化的挑战，并提出一种可学习、自适应的归一化方法CLeAN；

**💡 创新点**

CLeAN的创新点在于利用可学习的最大/最小值估计与指数移动平均EMA进行动态更新，并通过轻量级缩放网络实现对数据分布漂移的实时适应，从而在无未来信息的持续流中显著降低灾难性遗忘；

**🔧 技术方法**

主要技术包括可学习归一化层、EMA平滑、轻量缩放网络、全连接神经网络，以及对比的连续学习策略（Reservoir Experience Replay、A‑GEM、EWC）和传统归一化方法（全局、局部、Continual Normalization）；

**📊 数据集**

实验数据集为网络安全入侵检测的两大表格数据集：UNSW‑NB15 与 CICIDS‑2017；

**📈 对比分析**

在平均准确率、AUROC 与平均遗忘等指标上与全局归一化（理论上限）、局部归一化和CN进行对比，结果显示CLeAN在多种连续学习策略下均能接近全局归一化的性能，同时显著降低平均遗忘，尤其在后期经验中表现突出；

**⚠️ 局限性**

主要局限在于早期经验表现较差（归一化网络尚未收敛），EMA参数选择影响收敛速度；实验仅覆盖两大二分类数据集，缺少多模态或更大规模系统的验证，也未与更先进的混合连续学习方法进行深入对比。

---

## 181. Can Blindfolded LLMs Still Trade? An Anonymization-First Framework for Portfolio Optimization

**arXiv ID:** 2603.17692 | [PDF](https://arxiv.org/pdf/2603.17692v1)

**作者:** Joohyoung Jeon `[一作]` (Korea University), Hongchul Lee `[通讯]` (Korea University)

**通讯引用:** 1353 | [OpenAlex ID](https://openalex.org/A5077069037)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计并实现了 BlindTrade，利用匿名化的 S&P 500 组份、四个专业 LLM 代理产生多角度特征，构建语义图神经网络并通过 PPO‑DSR 强化学习决策，验证其在无记忆化条件下的交易效果。

**💡 创新点**

创新点包括：① 对 tickers 和公司信息进行系统匿名化以消除记忆化偏差；② 多代理协同评估并输出可解释 reasoning；③ 语义图编码器（SemGAT）基于代理 reasoning 生成的嵌入构建动态图结构；④ 采用 Intent 机制的 RL 策略实现可解释的市场姿态；⑤ 通过 IC 验证与负控制实验确保信号真实性。

**🔧 技术方法**

主要技术包括：大型语言模型（四个定制代理）、SBERT 文本嵌入、GATv2 语义图神经网络、PPO 强化学习（Differential Sharpe Reward）、Spearman IC 分析与负控制实验。

**📊 数据集**

使用数据集：S&P 500 2020‑2025 成分股每日数据（EODHD API）、匿名化的新闻标题、日终价格与成交量等技术指标。

**📈 对比分析**

实验采用 2025 年 YTD 145 天 OOS、2024‑2025 扩展期 397 天对比，基准包括 SPY、EQWL、Momentum、MCap Top‑20、RAW Top‑20；BlindTrade 在 OOS 的年化 Sharpe 为 1.40±0.22，累计回报 32.22%，均超过所有基准。

**⚠️ 局限性**

局限性：仅投资前 20 股、无现金持仓导致高波动与最大回撤；在持续上涨的牛市中 alpha 降低；匿名化无法完全排除所有泄漏路径；对不同市场环境的泛化尚未验证。

---

## 182. Evaluating LLM-Simulated Conversations in Modeling Inconsistent and Uncollaborative Behaviors in Human Social Interaction

**arXiv ID:** 2603.17094 | [PDF](https://arxiv.org/pdf/2603.17094v1)

**作者:** Ryo Kamoi `[一作]` (Penn State University), Pei Zhou `[通讯]` (Microsoft Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个用于评估大语言模型（LLM）模拟对话中不一致与不合作行为的框架，并通过LLM作为评判者在对话级别和细粒度（逐轮）上检测这些行为。

**💡 创新点**

创新点在于：①提出了10类细粒度的不一致与不合作行为；②将LLM本身作为评判者来自动化检测；③创建了跨行业（学术、商业、政府、辩论）的长对话延续基准，能够直接与人类对话进行对比。

**🔧 技术方法**

使用技术包括：LLM-as-a-Judge（OpenAI o4-mini）进行整体一致性/合作性评分和细粒度行为检测；prompt engineering（vanilla vs taxonomy-guided）；supervised fine‑tuning（GPT‑4.1在人工对话上微调）；多轮生成策略（1/5/30轮一次）来探究生成方式对行为的影响。

**📊 数据集**

数据集来自六个公开多方对话数据集：QMSum（Product、Academic、Committee）、NCPC、SIM、IQ2，合计300个对话样本（每类50个）用于评估，1,450条对话用于微调。

**📈 对比分析**

比较方法：利用LLM-as-a-Judge给出1–10分的整体一致性/合作性评分，并在细粒度层面统计10类行为出现次数；对照人类对话的行为频率。结果显示：①在vanilla提示下LLM对话几乎不出现这些行为；②提示工程和微调都无法稳定控制行为频率，往往产生过多或过少的特定行为；③多轮一次生成显著抑制不一致与不合作行为，导致与人类分布差距更大。总体一致性与合作性得分高，但细粒度行为与人类差异显著。

**⚠️ 局限性**

局限性包括：①数据集多为政府会议，业务、学术场景样本不足，缺乏多样性；②实验仅在英语环境下进行，无法直接推广到其他语言；③LLM-as-a-Judge的自动评估仍受限于模型自身偏差，需更多人工验证。

---

## 183. Who's Sense is This? Possibility for Impacting Human Insights in AI-assisted Sensemaking

**arXiv ID:** 2603.17643 | [PDF](https://arxiv.org/pdf/2603.17643v1)

**作者:** Zhuoyi Cheng `[一作]` (Eindhoven University of Technology), Steven Houben `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1957 | [OpenAlex ID](https://openalex.org/A5027589311)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

探讨AI辅助意义建构可能导致人类见解受影响的风险，并提出相关问题与原因

**💡 创新点**

首次系统阐述AI在意义建构早期阶段可能过早提供洞见导致人类观点被稀释或排斥的风险，并提出算法欣赏、过度依赖与隐式劝服三种可能原因

**🔧 技术方法**

未使用具体技术，主要基于文献综述与理论分析，涉及Chain-of-Thought等AI生成方法的讨论

**📊 数据集**

无实验数据集，纯理论讨论

**📈 对比分析**

未进行实验比较，本文不涉及性能评估

**⚠️ 局限性**

缺乏实证验证，未在真实场景中评估AI提示对人类意义建构的影响

---

## 184. Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant and Efficient Point Cloud Analysis

**arXiv ID:** 2603.17538 | [PDF](https://arxiv.org/pdf/2603.17538v1)

**作者:** Jaein Kim `[一作]` (Seoul National University), Byoung-Tak Zhang `[通讯]` (Seoul National University)

**通讯引用:** 3971 | [OpenAlex ID](https://openalex.org/A5050928023)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种新的SE(3)等变卷积网络ECKConv，利用双陪集空间和坐标基网络实现对点云的连续刚体运动等变学习；

**💡 创新点**

核心创新在于将卷积核域定义为SE(3) / SO(2)双陪集，使用坐标基网络和高斯嵌入显式生成核函数，并通过重排实现显式核的显存可扩展性；

**🔧 技术方法**

技术包括：双陪集空间建模、坐标基网络+高斯嵌入、显式核设计、ball query与farthest point sampling、残差块和U‑Net结构；

**📊 数据集**

在ModelNet40（分类、姿态配准）、ShapeNet（部件分割）和S3DIS（室内语义分割）等标准点云数据集上进行实验；

**📈 对比分析**

与非等变方法、模型无关方法、离散/连续SE(3)等变卷积（如EPN、E2PN、CSEConv、Vector Neurons等）进行对比。ECKConv在分类、姿态配准、部件分割和语义分割任务中均达到或超过最先进方法，且在大规模数据集上表现出更好的内存与推理效率；

**⚠️ 局限性**

局限性包括对法线信息的依赖（虽然提供了不使用法线的版本，但性能略低），以及在极大点云规模下仍需进一步优化计算效率。

---

## 185. SpiderCam: Low-Power Snapshot Depth from Differential Defocus

**arXiv ID:** 2603.17910 | [PDF](https://arxiv.org/pdf/2603.17910v1)

**作者:** Marcos A. Ferreira `[一作]` (Northwestern University), Emma Alexander `[通讯]` (Northwestern University)

**通讯引用:** 1624 | [OpenAlex ID](https://openalex.org/A5064038021)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出并实现了SpiderCam，一款基于FPGA的实时快照深度摄像机，利用深度焦散技术实时生成稀疏深度图；

**💡 创新点**

创新点在于首次实现子瓦特级别的被动3D摄像机，提出针对低功耗传感器的DfDD算法改进、内存局部化流式实现以及空间变参校准；

**🔧 技术方法**

核心技术包括光学双焦散传感器布置、FPGA（Lattice ECP5）SystemVerilog实现、固定点与FP16混合运算、图像预处理（齐次变换、低通/导数滤波）、多尺度联合深度估计与置信度阈值；

**📊 数据集**

使用实景自采样数据集：平面纹理标定帧在0.24–1.36 m等距分布的56帧，以及标准数据集KITTI与Middlebury用于对比；

**📈 对比分析**

与Focal Split、传统DfD、立体匹配等方法对比，SpiderCam在真实噪声场景下实现0.45–0.97 m的工作范围、MAE低于0.1 %深度误差，32.5 fps下功耗仅624 mW，核心功耗比SOTA低1.8×、整体功耗低3.3×；

**⚠️ 局限性**

局限性包括深度稀疏性、视场角受限、对动态纹理与高噪声场景的鲁棒性待提升、I/O速率限制了最高帧率以及对多尺度深度补全的依赖。

---

## 186. Facial Movement Dynamics Reveal Workload During Complex Multitasking

**arXiv ID:** 2603.17767 | [PDF](https://arxiv.org/pdf/2603.17767v1)

**作者:** Carter Sale `[一作]` (Macquarie University), Michael J. Richardson `[通讯]` (Macquarie University)

**通讯引用:** 11740 | [OpenAlex ID](https://openalex.org/A5090429070)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

使用标准摄像头结合 OpenPose 对 72 名参与者在 OpenMATB 多任务模拟中的面部与头部运动进行记录，提取线性运动学特征和递归量化（RQA）特征，并训练随机森林模型预测低、中、高工作负载。

**💡 创新点**

创新点在于：① 用低成本、非侵入式摄像头即刻捕捉面部/头部运动学与 RQA 动态，展示工作负载可通过面部行为显著区分；② 通过对 RQA 的两相变化（碎片化 → 重新组织）揭示工作负载对运动结构的深层影响；③ 证明仅需几分钟的个体校准即可达到 70% 以上的分类准确率。

**🔧 技术方法**

技术包括：OpenPose 关键点检测、Procrustes 对齐、速度/加速度计算、RQA（recurrence rate、determinism、entropy 等）、随机森林分类器、交叉验证（随机拆分与留一参与者）以及基准任务性能指标（准确率与反应时）。

**📊 数据集**

数据集为 72 名本科生（52 女 20 男，平均 21 岁）完成的 OpenMATB 四任务（监控、追踪、通讯、资源管理）在低/中/高负载下的 2‑min 基线和 8‑min 实验块，配合 60 Hz 摄像头记录与 OpenMATB 事件日志，最终得到 60 s 重叠窗口的运动学与 RQA 特征。

**📈 对比分析**

比较方法：在同一参与者内随机拆分训练/测试，和跨参与者留一验证。结果显示：① 运动学特征在同一人内可达 85 %±1.5 % 的平衡准确率，显著优于仅用任务性能的 55 %；② 任务性能与 RQA 的组合未提升；③ 跨参与者模型仅 43 %±8.8 %，略高于随机 33 %；④ 仅用 2‑min 校准即可提高到约 50 %，再增至 11‑min 可达 73 %。

**⚠️ 局限性**

局限性：跨人群推广差，需个体化校准；基线数据无法直接迁移；样本为同质本科生，缺乏专业或高风险场景；仅捕捉面部/头部，未考虑完整身体或其他生理信号；未验证在真实环境中的鲁棒性。

---

## 187. Argument Reconstruction as Supervision for Critical Thinking in LLMs

**arXiv ID:** 2603.17432 | [PDF](https://arxiv.org/pdf/2603.17432v1)

**作者:** Hyun Ryu `[一作]` (Carnegie Mellon University), Sean Welleck `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2262 | [OpenAlex ID](https://openalex.org/A5019030424)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了通用自动化论证重构（GAAR）引擎，并基于其构建了高质量的论证重构数据集Arguinas，随后验证了在七项批判性思维任务中的提升效果。

**💡 创新点**

①将AAR扩展为可处理任意论证（包含归纳、溯因、类比及正式/非正式谬误）的通用引擎；②引入谬误检测与决策逻辑；③采用三维细粒度的真实性、完整性、简洁性评估标准；④利用SAT求解器实现前提裁剪；⑤利用GAAR生成Arguinas数据集。

**🔧 技术方法**

使用大型语言模型（Claude Sonnet 4.5 作为基准模型）、逻辑流化与一阶谓词逻辑（FOL）转化、SAT求解器验证推理有效性、细粒度评估判定器、以及多轮迭代微调。

**📊 数据集**

Arguinas（2,850条论证）来源于七个渠道（Procon.org、Pros‑and‑Cons 1950/2010、NYT《Room for Debate》、Anthropic‑Persuasion、LLM生成的合成论证与合成谬误论证），以及用于对比的AAAC、EntailmentBank等数据集。

**📈 对比分析**

在有效性与真实性指标上，GAAR相较AAR与LLM提示方法均取得了 100% 有效性和更高的真实性赢率；在七项批判性思维任务（论证质量评估、论证推理、法律推理、逻辑推理）中，预适配微调与继续微调均显著提升了准确率/宏观F1，尤其在低资源场景下效果最为显著。

**⚠️ 局限性**

仍受限于：①对大型LLM的调用成本高；②迭代流程复杂，需多轮交互；③对极长或极复杂论证的处理尚未充分验证；④主要在英语数据上评估，跨语言泛化待研究。

---

## 188. CODMAS: A Dialectic Multi-Agent Collaborative Framework for Structured RTL Optimization

**arXiv ID:** 2603.17204 | [PDF](https://arxiv.org/pdf/2603.17204v1)

**作者:** Che-Ming Chang `[一作]` (National Taiwan University), Ehsan Degan `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CodMas 框架和 RTLOpt 基准，用多代理对话式推理和领域特定代码生成实现 RTL（Verilog）代码的流水线与时钟门控优化。

**💡 创新点**

创新点在于将结构化的双代理对话（Articulator 与 Hypothesis Partner）与域知识注入、确定性评估相结合，形成闭环迭代优化；同时首次公开了针对 RTL 优化的 RTLOpt 数据集。

**🔧 技术方法**

使用技术包括多代理系统、对话式推理、Domain‑Specific Coding Agent (DCA)、Code Evaluation Agent (CEA)、Pyverilog 数据流图、Yosys/Verilator 等 EDA 工具，以及 LLM 生成 Verilog 并进行自适应迭代优化。

**📊 数据集**

使用了 RTLOpt 数据集，包含 120 条 Verilog 三元组（未优化、优化、测试平台），涵盖流水线和时钟门控两类优化。

**📈 对比分析**

通过与零射击、Prompting、ReAct、Reflexion、LLM‑VeriPPA 等基线对比，实验显示在 GPT‑4o 等模型上可实现约 25% 的延迟降低、22% 的功耗下降，失败率低于 30%，显著优于传统单代理或无推理方法。

**⚠️ 局限性**

限制包括仅验证了流水线和时钟门控；对大型工业设计的可扩展性未完全验证；多轮迭代导致计算开销；RTLOpt 规模有限，缺乏形式等价验证，可能遗漏细微错误。

---

## 189. HAPS-RIS-assisted IoT Networks for Disaster Recovery and Emergency Response: Architecture, Application Scenarios, and Open Challenges

**arXiv ID:** 2603.17054 | [PDF](https://arxiv.org/pdf/2603.17054v1)

**作者:** Bilal Karaman `[一作]` (Manisa Celal Bayar University), Halim Yanikomeroglu `[通讯]` (Carleton University)

**通讯引用:** 21209 | [OpenAlex ID](https://openalex.org/A5035446029)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

提出基于高空平台(HAPS)与可重构智能表面(RIS)的灾害恢复IoT后传架构，并设计可动态可调子连激活RIS结构以提升双路径衰落环境下的链路质量与能效。

**💡 创新点**

创新点在于：1）引入动态可调子连激活RIS，能够根据需要切换不同组大小(L)实现功率与容量的权衡；2）通过子连结构实现数万个RIS单元在HAPS平台上的可扩展部署；3）在灾害场景下系统级仿真验证该架构在下行/上行速率和能效上的显著提升。

**🔧 技术方法**

采用S‑band 2.4 GHz频段，3GPP空地链路模型（LOS/NLOS、路径损耗、阴影、气象衰减等）以及RIS相位控制与功率放大器模型；通过数值仿真计算数据率、CDF与能效。

**📊 数据集**

无真实数据集，所有结果均基于仿真场景：30 000个RIS单元，HAPS高度20 km，灾区半径50 km，1000个IoT网关，使用标准化通道模型和预设参数表。

**📈 对比分析**

与传统被动RIS进行对比；在下行中L=1000子连方案使得几乎所有网关达到≥70 Mbps；上行中L=500子连方案使得最小速率>15 Mbps；能效方面，子连方案的CDF明显优于被动RIS（如80%网关能效>0.62 Mbit/J）。

**⚠️ 局限性**

局限性包括：1）假设完美CSI与理想相位控制，未考虑反馈延迟与估计误差；2）未对气象扰动、平台振动和热漂移等实际环境因素进行建模；3）功率消耗模型简化，未计入转换损耗与能量储备；4）硬件实现与成本、重量约束尚未验证；5）多HAPS、卫星协同与多跳等实际部署细节仍需进一步研究。

---

## 190. New Greedy Spanners and Applications

**arXiv ID:** 2603.17085 | [PDF](https://arxiv.org/pdf/2603.17085v1)

**作者:** Elizaveta Popova `[一作]` (Weizmann Institute), Elad Tzalik `[通讯]` (Weizmann Institute)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5051677237)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种简单的贪心算法来构造 (α,β)-spanner，并基于该算法给出了多重图中 f‑EFT (k,k‑1) spanner 的紧逼构造；同时提出了一种五阶段的构造，得到了一种大小为 O(n^{1+1/k}) 的加权图 2‑路径替代路段 spanner，并在该类 spanner 上实现了最优的边数与距离逼近折衷。

**💡 创新点**

创新点在于：
① 用贪心 d→r spanner 的思想统一分析多重图的 fault‑tolerant spanner，获得了与已知下界匹配的 O(f n^{1+1/k}) 上界；
② 通过聚类、侧向聚类和距离约减等新步骤，构造出在加权图中对 2‑跳路径的权重逼近最优的 spanner；
③ 对并行贪心 spanner 给出了简洁的两页证明，显著提升了此前复杂度高的证明；
④ 在无权图和加权图之间桥接了 (α,β)-spanner 的研究，首次给出了在加权图中优于 (2k‑1) 多重扩展的实例。

**🔧 技术方法**

采用的技术主要有：
- 贪心 d→r 机制与聚类和球增长相结合；
- 侧向聚类（lateral clustering）与全局距离约减（global distance reduction）相结合的分阶段构造；
- 阻塞集（blocking set）与随机抽样的组合分析，用以控制 FT‑spanner 的大小；
- 局部球交换与邻域交换 lemma，分析聚类与距离收缩的关系。

**📊 数据集**

本工作为理论算法，不使用具体数据集；所有结果均为泛化的图结构理论证明，依赖于 Erdős girth conjecture 等假设。

**📈 对比分析**

与现有的多重图 f‑EFT (k,k‑1) spanner 的上界 O(f^3 n^{1+1/k}) 相比，本文将开销降至 O(f n^{1+1/k})，与下界相匹配；
与传统的 (2k‑1)-spanner O(f n^{1+1/k}) 相比，本文在加权图中实现了更细致的加权路径逼近，且同样保持 O(n^{1+1/k}) 边数；
在并行贪心 spanner 的边数上，由 O(k^k n^{1+1/k}) 降至 O(k n^{1+1/k})，并给出了更直观的证明。

**⚠️ 局限性**

局限性包括：
- 结果在多重图上取得紧逼，需要假设 Erdős girth conjecture；
- 加权图的 FT‑spanner 尚未得到完整扩展，现有方法仅适用于无故障或单个路径的加权逼近；
- 对于单图（simple graph）的 f‑EFT (k,k‑1) spanner，仍未能证明相同的 O(f n^{1+1/k}) 上界；
- 目前构造的 spanner 并未考虑轻量性（lightness）问题；
- 对更高阶路径（跳数 >2）的加权逼近仍需进一步研究。

---

## 191. CRE-T1 Preview Technical Report: Beyond Contrastive Learning for Reasoning-Intensive Retrieval

**arXiv ID:** 2603.17387 | [PDF](https://arxiv.org/pdf/2603.17387v1)

**作者:** Guangzhi Wang `[一作]` (Career International Research Team), Zhi Liu `[通讯]` (Career International Research Team)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Thought 1（T1）检索模型，将静态对齐改为动态推理生成来提升推理密集检索性能

**💡 创新点**

核心创新是让模型在查询端生成有限步推理序列并通过特殊标记聚合为向量，实现检索时对每条查询的动态推理路径；引入三阶段训练和GRPO强化学习进一步优化推理质量

**🔧 技术方法**

技术包括生成式检索、异构查询/文档编码、特殊语义聚合标记、三阶段训练（任务意识、推理对齐、GRPO强化学习）

**📊 数据集**

使用MS MARCO（400K）作为任务意识阶段数据，ReasonEmbed（82K）用于推理对齐与强化学习阶段；评估基准为BRIGHT

**📈 对比分析**

与对比学习基线（4B）及多阶段组合模型比较，T1‑4B在BRIGHT上平均nDCG@10为37.1，显著高于对比学习（33.2）且接近或优于检索+重排序组合模型，单模型即可匹配多阶段流水线性能

**⚠️ 局限性**

限制包括对符号/代码类子任务（如Code、AoPS）的提升不明显，GRPO奖励可能对细粒度匹配不足；未来需更细粒度奖励或领域数据以进一步提升

---

## 192. Patient4D: Temporally Consistent Patient Body Mesh Recovery from Monocular Operating Room Video

**arXiv ID:** 2603.17178 | [PDF](https://arxiv.org/pdf/2603.17178v1)

**作者:** Mingxiao Tu `[一作]` (University of Sydney), Jinman Kim `[通讯]` (University of Sydney)

**通讯引用:** 7812 | [OpenAlex ID](https://openalex.org/A5100614820)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一套名为 Patient4D 的无监督流水线，利用单目手术视频在静止患者场景下恢复时间一致的 3D 人体网格。

**💡 创新点**

核心创新在于将“静止”先验硬编码为 Pose Locking（姿态锁定）和 Rigid Fallback（刚性回退），通过基础模型（SAM3、MoGe、SAM3DBody）实现零调优、无训练的高效推断。

**🔧 技术方法**

技术组合包括：Segment Anything Model (SAM3) 做 2D 人体分割；MoGe 估计相机内参；SAM3DBody 对分割区域进行 SMPL 3D 预测；Pose Locking 用中值姿态作为全局锚点消除噪声；Rigid Fallback 维护关键帧池并通过 Dice 目标进行刚性配准；整体使用离线优化与后处理。

**📊 数据集**

评估数据集包括：Sim‑Geometry（来自 SLP‑3Dfits 的真实 SMPL 贴合，提供 3D ground truth）；Sim‑Visual（利用 Seedance2.0 生成的含不同覆盖度、姿态和摄像机轨迹的合成手术视频）；以及公开 HMR 基准 3DPW、EMDB 和 RICH。

**📈 对比分析**

与 HMR 2.0、SMPLest‑X、GVHMR、CoMotion 和 PromptHMR 等基线对比，Patient4D 在 Sim‑Geometry 上实现 0.75 的 Mesh‑Mask IoU 和 78.3 mm 的 PA‑PVE，显著高于最优基线；在 Sim‑Visual 上 IoU 0.75、失败率 1.3%；在 3DPW、EMDB、RICH 上保持或略优于现有最佳结果。

**⚠️ 局限性**

局限性包括：依赖高质量关键帧若多数帧被遮挡导致恢复失败；静止假设在患者活动或转位时失效；单目相机缺乏绝对深度信息，影响精准配准；缺乏真实手术环境的 3D 真实标注，未在临床数据上充分验证。

---

## 193. Minimum-Action Learning: Energy-Constrained Symbolic Model Selection for Physical Law Identification from Noisy Data

**arXiv ID:** 2603.16951 | [PDF](https://arxiv.org/pdf/2603.16951v1)

**作者:** Martin G. Frasch `[一作]` (University of Washington), Martin G. Frasch `[通讯]` (University of Washington)

**通讯引用:** 2816 | [OpenAlex ID](https://openalex.org/A5024146903)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于能量最小化的物理法则识别框架MAL，能够在噪声极大的观测数据中从预定义的基函数库中识别出正确的力学规律；

**💡 创新点**

创新点在于将三重行动（信息最大化、能量最小化、对称性约束）结合到网络训练目标中，并通过宽支点差分显著降低噪声、利用门控权重的软至硬转化实现基函数选择，以及使用能量守恒诊断实现模型选择；

**🔧 技术方法**

采用门控软max温度退火的可微架构搜索、宽支点加速匹配的加速度估计、能量守恒正则化（Noether性）、神经网络轨迹重建和后期最小二乘校准；

**📊 数据集**

在两个合成基准上验证：Kepler（万有引力）和Hooke（弹簧）轨道数据，分别采样半径0.5–5AU、偏心率0–0.3，并加入1%位置噪声；

**📈 对比分析**

与SINDy、HNN、LNN等方法比较，MAL在Kepler基准上实现了93%（4/10直接选择）或100%（通过能量诊断）识别率，训练耗时835 s、能耗约0.07 kWh，比仅预测误差的基线低40%，同时提供可解释的符号方程；

**⚠️ 局限性**

局限包括需要预先给定基函数库、原始直接选择率仅40%（需多种seed或先验初始化）、对宽支点预处理的高度依赖、以及对噪声水平或轨道范围的敏感性，未来需扩展到更复杂系统和自发发现功能。

---

## 194. Efficient Soft Actor-Critic with LLM-Based Action-Level Guidance for Continuous Control

**arXiv ID:** 2603.17468 | [PDF](https://arxiv.org/pdf/2603.17468v1)

**作者:** Hao Ma `[一作]` (University of Chinese Academy of Sciences), Huimu Wang `[通讯]` (JD.com)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 GuidedSAC，一种在 Soft Actor-Critic（SAC）框架下引入 LLM 作为动作级监督者，提供实时干预以加速探索和提升最终性能。

**💡 创新点**

创新点包括：1）将 LLM 的宏观策略分析与微观动作干预相结合；2）理论证明在使用子最优动作级指导时仍保持 SAC 的收敛性并提升收敛速度；3）设计双 LLM（advisor 与 coder）协作实现可解释的规则化干预。

**🔧 技术方法**

技术手段包括：SAC 算法、最大熵 RL、LLM 交互（Prompt 设计、任务分解）、残差动作干预、经验回放混合采样、基于 KL 优化的策略改进。

**📊 数据集**

数据集与环境：离散 toy 文本环境（Blackjack、CliffWalking、FrozenLake、Taxi）、连续控制任务（MountainCar、MuJoCo Humanoid）。

**📈 对比分析**

与基线（SAC、SAC+RND、RND、ICM、E3B）对比，GuidedSAC 在所有测试环境中均实现更高的样本效率和更优最终回报；尤其在 MountainCar 与 Humanoid 上表现出显著的快速收敛和更合理的策略。

**⚠️ 局限性**

局限性：1）LLM 需要设计精细 Prompt，干预策略的质量高度依赖 LLM 生成的规则；2）在极大或高维连续空间中，LLM 生成的干预可能不易解释或不足以覆盖所有复杂情形；3）实验多集中在仿真环境，缺乏真实机器人验证。

---

## 195. HoloByte: Continuous Hyperspherical Distillation for Tokenizer-Free Modeling

**arXiv ID:** 2603.16917 | [PDF](https://arxiv.org/pdf/2603.16917v1)

**作者:** Vladimer Khasia `[一作]` `[通讯]` (Independent Researcher), Vladimer Khasia (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种完全无词元化的连续球面蒸馏框架HoloByte，用来在不使用子词分词的情况下实现字节级自回归序列建模；

**💡 创新点**

创新点在于利用可逆正交旋转将连续字节块压缩到球面向量上，实现宏观Transformer仅在压缩空间上工作，并通过局部微解码器恢复原始字节，解决了字节级建模的计算瓶颈和量化误差问题；

**🔧 技术方法**

核心技术包括连续球面编码（Hyperspherical Encoding）、正交旋转绑定/解绑、宏观Transformer加微解码器、双目标损失（交叉熵+Holographic Latent MSE）以及理论分析的维度下界与复杂度证明；

**📊 数据集**

实验使用FineWeb‑Edu数据集（约5 M字符）进行训练；

**📈 对比分析**

在参数相同（≈82 M）的条件下，与BPE子词模型相比，HoloByte在单位字节信息压缩上实现1.484 nats/byte，显著优于BPE的1.954 nats/byte；

**⚠️ 局限性**

局限性包括对块大小W和维度D的敏感性，且在极大上下文窗口下宏观Transformer仍可能出现注意力计算负担，且模型对字节级输入的实现细节和解码速度仍需进一步优化。

---

## 196. PowerModelsGAT-AI: Physics-Informed Graph Attention for Multi-System Power Flow with Continual Learning

**arXiv ID:** 2603.16879 | [PDF](https://arxiv.org/pdf/2603.16879v1)

**作者:** Chidozie Ezeakunne `[一作]` (Los Alamos National Laboratory), Anup Pandey `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 418 | [OpenAlex ID](https://openalex.org/A5082049643)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为PowerModelsGAT-AI的统一多系统功率流预测模型，能够在单一模型中同时预测不同类型母线的电压幅值、角度以及发电机注入功率，支持实时、受限条件下的AC功率流求解；

**💡 创新点**

核心创新包括：①使用基于图注意力网络的物理信息感知模型，融合边特征和节点特征；②采用母线类型感知的监督掩码实现多任务联合学习；③引入功率不匹配的物理损失并通过自适应不确定性权重动态平衡多任务损失；④在新系统适配时结合经验回放和弹性权重约束实现无灾忘的持续学习；

**🔧 技术方法**

技术框架为带边特征的GATv2图神经网络、预归一化残差块、共享MLP输出头、物理约束损失、均匀边注意力机制、Homoscedastic uncertainty weighting、多任务学习、经验回放+EWC持续学习；

**📊 数据集**

使用14个标准电网案例（4-6470节点），在每个系统上生成随机负荷、发电、参数扰动及N-2拓扑故障，构成训练集，评估统一13系统模型及单系统基线；

**📈 对比分析**

与各系统单独训练的基线相比，统一模型在N-2条件下平均电压幅值NMAE为0.89%，角度R²>0.99，且误差在各系统间仅略有提升；在持续学习实验中，EWC+Replay在适配新系统时保持基线误差增幅<2%，且显著降低了遗忘；

**⚠️ 局限性**

局限包括：①在极大规模系统（如6470节点）下统一模型误差增大，需进一步扩展训练；②仅针对N-2故障，未验证更高阶N-k或极端动态条件；③物理约束损失采用简单电压/功率不匹配，可能不足以捕捉更复杂约束；④解释性分析仅基于相关性，未提供因果验证；

---

## 197. Physics-Aware Machine Learning for Seismic and Volcanic Signal Interpretation

**arXiv ID:** 2603.17855 | [PDF](https://arxiv.org/pdf/2603.17855v1)

**作者:** William Thorossian `[一作]` `[通讯]`, William Thorossian

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文系统梳理并提出了一套面向地震与火山监测的机器学习技术框架，强调在信号处理与物理先验下实现稳健、可解释的监测系统；

**💡 创新点**

创新点在于将传统信号处理与自监督、生成模型、物理约束、置信度校准等多种技术融合，并设计了针对跨区域、跨站点、跨时间的评估协议，显著提升模型在实际监测环境中的泛化与可靠性；

**🔧 技术方法**

采用的技术包括经典时频变换、卷积/Transformer网络、对比学习、自监督预训练、扩散/自编码器生成模型、物理信息约束（如到达时间、衰减模型）、集成与共形预测、迁移学习与域对抗训练；

**📊 数据集**

主要使用公开或内部的连续记录档案、地震/火山事件清单、不同站点与仪器的多通道波形（包括地震、长周期、红外、GNSS 等），并通过层级化标签和软标签来处理不一致的标注；

**📈 对比分析**

通过对比实验，采用随机拆分、留一站点、留一地区等验证协议，显示模型在随机拆分下表现乐观，而留一站点拆分揭示显著性能下降；加入置信度校准后，误报率得到控制，加入物理约束的混合损失后，时间偏差和位置残差显著下降，整体指标如检测概率与误报率、均值绝对误差、ECE等均有提升；

**⚠️ 局限性**

局限性包括：对域漂移和标签噪声的适应仍不充分；缺乏统一、公开的跨站点评测基准；模型对极端噪声、断流、时钟漂移等非理想遥测条件的鲁棒性有限；实时边缘部署与计算资源限制尚未得到充分解决；

---

## 198. Narrative Frames: A New Approach to Analysing Metaphors in AI Ethics and Policy Discourse

**arXiv ID:** 2603.17192 | [PDF](https://arxiv.org/pdf/2603.17192v1)

**作者:** Daniel Stone `[一作]` `[通讯]`, Daniel Stone

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Narrative Frames类型体系，用于系统化分析AI伦理与政策话语中的隐喻，解决现有方法定义不统一的问题。

**💡 创新点**

创新点在于将Lakoff‑Johnson隐喻主列表与批判性隐喻研究相结合，构造了49个可操作的叙事框架，并明确了框架与语篇策略的对应关系。

**🔧 技术方法**

采用概念隐喻理论与语篇框架理论相结合的混合方法，辅以归纳编码与系统交叉检索。

**📊 数据集**

主要数据集为MetaNet数据库中的685条隐喻，辅以选取的82篇批判性隐喻分析文献。

**📈 对比分析**

通过与已有批判性隐喻研究的标签对照，验证框架的覆盖度与一致性，虽未给出量化性能指标，但在理论与案例分析层面表现出较高的解释力。

**⚠️ 局限性**

局限在于仅分析英文文献、主观性强、跨文化差异未充分考量，且未评估编码者间一致性。

---

## 199. Interpreting Context-Aware Human Preferences for Multi-Objective Robot Navigation

**arXiv ID:** 2603.17510 | [PDF](https://arxiv.org/pdf/2603.17510v1)

**作者:** Tharun Sethuraman `[一作]` (Hochschule Bonn-Rhein-Sieg), Maren Bennewitz `[通讯]` (University of Bonn)

**通讯引用:** 8526 | [OpenAlex ID](https://openalex.org/A5103231515)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了如何让机器人在共享环境中理解并实时执行基于上下文的人类导航偏好。

**💡 创新点**

结合VLM、LLM与MORL的混合管线，提供可解释的规则存储与即时偏好向量生成。

**🔧 技术方法**

使用Gemini VLM、LLM（GPT‑4o、Mistral‑Large）、多目标强化学习策略以及规则更新模块。

**📊 数据集**

评估使用MIT Indoor Scenes、SRIN数据集以及Toyota HSR实地实验环境（办公、厨房、超市）。

**📈 对比分析**

与多种LLM、VLM对比，取得低偏好预测误差和在三种真实环境中显著提升的行驶性能；在基准实验中表现稳定。

**⚠️ 局限性**

受限于LLM推理延迟、可能出现幻觉、规则库维护成本较高以及对环境外场景的泛化不足。

---

## 200. RoboForge: Physically Optimized Text-guided Whole-Body Locomotion for Humanoids

**arXiv ID:** 2603.17927 | [PDF](https://arxiv.org/pdf/2603.17927v1)

**作者:** Xichen Yuan `[一作]` (Nanyang Technological University), Jianfei Yang `[通讯]` (Nanyang Technological University)

**通讯引用:** 7133 | [OpenAlex ID](https://openalex.org/A5005666034)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了RoboForge框架，实现从文本到潜在空间的运动生成，并通过物理可行性优化PP-Opt以及无重定向的潜在驱动控制完成全身机器人运动执行。

**💡 创新点**

创新点在于：①将物理可行性优化与生成器、控制器形成双向闭环，互相改进；②采用PP-Opt在仿真中迭代优化轨迹并反向微调生成器；③使用潜在驱动隐式接口替代传统重定向流程，显著降低接触转移误差。

**🔧 技术方法**

核心技术包括潜在空间扩散模型（Transformer去噪）、强化学习（如DDPG/SAC）、教师‑学生蒸馏与DAgger、物理仿真环境（IsaacLab、MuJoCo）、物理可行性奖励函数以及潜在驱动控制策略。

**📊 数据集**

使用 HumanML3D 与 KIT-ML 两个大规模人体动作数据集进行生成器训练和评估，并在 Unitree G1 人形机器人上进行仿真与实机验证。

**📈 对比分析**

与基线 MLD、MLD+PP-Opt 以及显式重定向 GMR 进行对比；评价指标包括 R‑Precision、FID、Diversity、Penetrate/Float/Skate、成功率、E_mpjpe/E_mpkpe。结果显示，PP‑Opt 在生成质量与物理可行性上均有提升，隐式潜在接口比显式重定向在 IsaacLab 与 MuJoCo 上成功率提升约 5‑10%，误差降低 30%。

**⚠️ 局限性**

局限性包括：需要多轮 PP‑Opt 迭代才能收敛；仅针对平地无物体交互场景，无法直接处理复杂地形或交互；从仿真到真实的迁移仍存在泛化挑战；生成器对大规模 MoCap 训练数据的依赖较强。

---

## 201. GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes

**arXiv ID:** 2603.17993 | [PDF](https://arxiv.org/pdf/2603.17993v1)

**作者:** Huajian Zeng `[一作]` (Technical University of Munich), Xi Wang `[通讯]` (Technical University of Munich)

**通讯引用:** 1446 | [OpenAlex ID](https://openalex.org/A5100442248)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态Transformer框架，直接预测可控的6-DOF物体轨迹，利用3D边界框几何、点云上下文、语义类别和目标姿态进行条件生成。

**💡 创新点**

将物体轨迹作为统一的中间表示，实现跨机器人可执行；通过分层融合优先考虑几何约束，兼顾语义；在单一模型中实现多模态条件与目标约束，避免传统规划的高维搜索。

**🔧 技术方法**

多模态Transformer（Perceiver IO风格）与PointNet++点云编码；CLIP视觉语言语义编码；几何特征传播与多头自注意力；逆运动学映射；自定义损失组合（位置、旋转、重建、目的地）。

**📊 数据集**

ADT（Aria Digital Twin）模拟数据和HD-EPIC真实家庭场景数据。

**📈 对比分析**

对比改造的GIMO与CHOIS两种人机交互基线，评估ADE、FDE、Fréchet距离、Angular Consistency、Collision Rate。实验显示本方法在ADE、FDE、Fréchet、AC等指标上显著优于基线，且冲突率虽略高但仍低于平均水平。

**⚠️ 局限性**

依赖准确的场景对齐与目标姿态；目标信息往往缺失；对极端噪声或未见场景的鲁棒性有限；未引入闭环反馈或强化学习；需要进一步改进目标推断与碰撞优化。

---

## 202. Unsupervised Symbolic Anomaly Detection

**arXiv ID:** 2603.17575 | [PDF](https://arxiv.org/pdf/2603.17575v1)

**作者:** Md Maruf Hossain `[一作]` (TU Dortmund University), Emmanuel Müller `[通讯]` (Research Center Trustworthy Data Science and Security UA Ruhr)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 SYRAN，一种基于符号回归的无监督异常检测方法；

**💡 创新点**

创新点在于通过学习一组可解释的符号不变式（近似常数函数）来直接捕捉正常模式，从而无需黑盒模型；

**🔧 技术方法**

采用符号回归与进化算法求解不变式，配合特征袋装、噪声对比和复杂度正则化；

**📊 数据集**

在 19 个 ADBench 公共数据集（如 pima、breastw、cardiotocography 等）以及物理示例 Kepler 三次定律上进行实验；

**📈 对比分析**

与 14 种基线方法（包括 LOF、DEAN 等）进行 AUC‑ROC 比较，整体集成版 SYRAN 与基线相当，单个最优不变式甚至获得最高平均排名；

**⚠️ 局限性**

局限在于对高维/时间序列数据的扩展仍需研究，且对超参数敏感，过于简单的数据集可能被单一特征完美分割，影响评估可靠性。

---

## 203. M2P: Improving Visual Foundation Models with Mask-to-Point Weakly-Supervised Learning for Dense Point Tracking

**arXiv ID:** 2603.17813 | [PDF](https://arxiv.org/pdf/2603.17813v1)

**作者:** Qiangqiang Wu `[一作]` (City University of Hong Kong), Antoni B. Chan `[通讯]` (City University of Hong Kong)

**通讯引用:** 12217 | [OpenAlex ID](https://openalex.org/A5065680386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种基于视频分割（VOS）掩码的弱监督学习框架（Mask-to-Point, M2P），通过掩码引导的三种约束提升视觉基础模型（VFM）在密集点追踪（TAP）任务中的时序匹配能力。

**💡 创新点**

创新点在于：①利用Procrustes分析估计局部结构一致性，将少量可靠对应点扩展为全局点变换；②引入掩码标签一致性（MLC）正则化，防止漂移；③加入掩码边界约束（MBC）强化边缘点的匹配，从而实现仅用VOS掩码进行点追踪表示的学习。

**🔧 技术方法**

技术方法包括：基于ViT的视觉基础模型（DINOv2、DINOv3）作为特征提取器；LoRA参数高效微调；软argmax、正交变换、Huber损失等；以及对三种约束的联合优化。

**📊 数据集**

使用了约3.6K条带掩码的VOS数据集——YouTube-VOS 2019 和 DAVIS-2017 进行预训练；随后在TAP-Vid-DAVIS、TAP-Vid-Kinetics 等基准集上评估；还在合成 Kubric 数据集上进行下游微调。

**📈 对比分析**

与原始 DINOv2/DINOv3、Chrono、DINO-Tracker 等对比，M2P 在 TAP-VID-DAVIS 上提升了 12.8%（DINOv2-B/14）和 14.6%（DINOv3-B/16）的平均点精度；在测试时优化场景中，M2P-Tracker 参数量减少 15×、训练速度提升 22%，且在 Occlusion Accuracy、Average Jaccard 等指标上均超越基线。

**⚠️ 局限性**

局限性包括：仅使用 3.6K VOS 视频，模型容量受限；对强干扰和极端形变的鲁棒性不足；未结合在线记忆或更大规模 VOS 数据集（如 SA-V）进一步提升时序一致性。

---

## 204. S-VGGT: Structure-Aware Subscene Decomposition for Scalable 3D Foundation Models

**arXiv ID:** 2603.17625 | [PDF](https://arxiv.org/pdf/2603.17625v1)

**作者:** Xinze Li `[一作]` (Beijing Normal-Hong Kong Baptist University), Wentao Cheng `[通讯]` (Beijing Normal-Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出S-VGGT框架，利用结构化子场景分解和锚帧共享，显著降低全局注意力的二次复杂度，实现对长序列的高效3D重建。

**💡 创新点**

创新点在于：①在帧级别进行密度感知的软分配子场景划分；②共享锚帧实现子场景间的并行处理；③与token级加速方法完全正交，可无缝叠加。

**🔧 技术方法**

采用VGGT基础模型、基于DINOv2的特征提取、场景图相似度计算、soft assignment优化、anchor frame sharing以及token merging技术。

**📊 数据集**

在ScanNet、Neural RGB-D和7Scenes三个常用数据集上进行评估。

**📈 对比分析**

与VGGT^*、FastVGGT及多种基准方法对比，S-VGGT在500–1000帧长序列中实现3–4倍FPS提升，重建精度与完整度基本保持或略有提升。

**⚠️ 局限性**

局限性包括：对极稀疏或超大尺度场景的鲁棒性待验证；soft assignment需要额外迭代，增加一定前处理时间；仍受单GPU显存限制，最高子场景数需根据硬件动态调整。

---

## 205. Translation Invariance of Neural Operators for the FitzHugh-Nagumo Model

**arXiv ID:** 2603.17523 | [PDF](https://arxiv.org/pdf/2603.17523v1)

**作者:** Luca Pellegrini `[一作]` (University of Pavia), Luca Pellegrini `[通讯]` (Università della Svizzera italiana)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究对七种神经算子（CNO、DON、DON-CNN、POD-DON、FNO、TFNO、LocalNO）在FitzHugh–Nagumo模型中的学习能力进行了系统评估，并提出利用模型平移不变性的一种高效训练策略，进一步降低数据集生成成本。

**💡 创新点**

创新点包括：①基于平移不变性设计的训练/测试分布，显著减少训练样本需求；②对多种NO架构在“外域”平移任务上的性能进行对比，揭示了不同架构在精度、泛化与计算效率之间的权衡；③引入了综合成本指标（考虑误差、参数量、训练时间）和对数加权指标，提供更公平的评估框架。

**🔧 技术方法**

使用的技术主要是PyTorch实现的Neural Operator模型，配合Ray+HyperNOs进行自动超参数搜索；FNO采用FFT加权核；CNO使用基于带限函数的卷积与上/下采样；DON使用分支-干线结构，POD-DON结合POD模式；LocalNO采用局部核；TFNO使用特克尔张量分解；所有模型均在训练时使用AdamW +学习率调度。

**📊 数据集**

使用自生成的FitzHugh–Nagumo PDE数据集：2000个训练样本（不同强度、位置，时间固定）和500个测试样本（强度、位置、时间均随机变动），通过Firedrake求解器得到高保真参考解。

**📈 对比分析**

比较方法：分别记录每个模型的可训练参数量、训练总时长与每轮时长、训练/测试误差（L²、L¹相对范数）、推理时间；引入综合成本函数C和对数加权成本C_log进行全局对比。结果显示：CNO在平移测试集上误差最低（中位数≈0.09），但训练耗时最长（≈10 h）；FNO训练误差最小（≈0.008）但测试泛化差、推理慢；DON及其变种训练最快、参数最少，但测试误差高；LocalNO在推理速度与误差之间取得折中。

**⚠️ 局限性**

局限性：所有NO架构均难以准确判断刺激阈值（即是否能触发动作电位），导致在弱刺激样本上出现大误差；训练策略虽然降低了数据量，但仍需手工设置刺激参数；对更复杂或高维物理问题的推广尚未验证。

---

## 206. Omni-3DEdit: Generalized Versatile 3D Editing in One-Pass

**arXiv ID:** 2603.17841 | [PDF](https://arxiv.org/pdf/2603.17841v1)

**作者:** Chen Liyi `[一作]`, Zhang Lei `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Omni-3DEdit，一个统一的学习型框架，可一次性完成 3D 移除、添加和外观编辑，无需手动掩码或显式几何更新。

**💡 创新点**

创新点包括：①在多视角潜在空间直接编辑，避免逐视角迭代；②构建大规模合成配对编辑数据管线；③使用双流 LoRA 将预训练多视角生成模型 SEVA 重新定位为编辑网络 OmniNet，实现源视角与条件视角信息的解耦与融合。

**🔧 技术方法**

技术手段包括：SEVA 生成网络、双流 LoRA、Qwen-Image 进行单视角编辑、Gemini-2.5pro 生成指令、VGGT 估计相机姿态、SDEdit 进行一致性细化、AnySplat 进行 3D 重建、VAE+EDM 噪声编码。

**📊 数据集**

使用 CO3Dv2、DL3DV、WildRGB-D 合成的配对编辑数据进行训练，评估时采用 360-USID（360°场景）、CO3Dv2 验证集（添加/移除）等。

**📈 对比分析**

与现有专门任务方法（如 SPIn-NeRF、2DGS+LaMa、MVInpainter 等）比较，Omni-3DEdit 在 PSNR/LPIPS/CLIP 分数上表现更优，且推理时间从 30 分钟缩短到约 2 分钟；在复杂编辑任务中同样获得更高 Gemini 评分和更快速度。

**⚠️ 局限性**

局限性：合成数据集规模仅约 0.1M，难以覆盖极细粒度编辑（如“手腕戴手链”），受数据与算力限制，未来需扩展数据规模并探索 4D 或智能体场景。

---

## 207. Discovering Decoupled Functional Modules in Large Language Models

**arXiv ID:** 2603.17823 | [PDF](https://arxiv.org/pdf/2603.17823v1)

**作者:** Yanke Yu `[一作]` (Hong Kong University of Science and Technology), Yi Zheng `[通讯]` (Huawei Technologies)

**通讯引用:** 7578 | [OpenAlex ID](https://openalex.org/A5089622171)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM内部无监督地自动发现彼此分离的功能模块

**💡 创新点**

提出ULCMOD框架及IterD迭代优化算法，兼顾激活密度与模块平衡

**🔧 技术方法**

采用双行列分割、激活模块化与平衡得分的组合目标，并使用IterD逐步优化

**📊 数据集**

使用Qwen2.5-1.5B/3B/7B模型的全层激活，样本来源于Infinity‑Instruct数据集

**📈 对比分析**

与KMeans、Mini‑Batch KMeans、Agglomerative、Spectral等聚类基线对比，IterD在激活模块化、平衡得分及下游分类准确率/宏F1上均显著优于基线

**⚠️ 局限性**

局限于单模型内部模块划分，缺乏跨模型/跨语言验证，对模块数K的选择较为敏感

---

## 208. CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image

**arXiv ID:** 2603.17779 | [PDF](https://arxiv.org/pdf/2603.17779v1)

**作者:** Yizheng Song `[一作]` (Nanjing University), Hao Zhu `[通讯]` (Nanjing University)

**通讯引用:** 10007 | [OpenAlex ID](https://openalex.org/A5068560690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对单张图像中的多人人群场景，提出一种统一两阶段框架：先利用自监督的LORM完成遮挡补全并构建粗糙的3D高斯场景，再用单步扩散式细化器（配合Self-Calibrated Learning）提升细节，最终生成完整、可动画的多人3D人类模型。

**💡 创新点**

创新点包括：①自监督适配LORM，可在无3D标注的情况下恢复遮挡完整人体；②单步扩散细化器与Self-Calibrated Learning相结合，动态平衡细化力度，避免过度修正；③把2D扩散模型作为伪真实标注反馈到3D高斯场景，提升局部细节与全局一致性。

**🔧 技术方法**

技术包括：大规模人类重建模型迁移与LoRA适配、教师-学生自监督框架、3D高斯光栅化（3D Gaussian Splatting）、单步扩散细化器、Self-Calibrated Learning训练策略、SMPL-X姿态与法线约束。

**📊 数据集**

训练数据：HuGe100K前视图图像用于LORM自监督训练；THuman2.1合成多人人群场景用于扩散细化器训练与评估；评估时使用THuman2.1遮挡模拟和自然遮挡图像。

**📈 对比分析**

与SOTA方法（LHM、IDOL、PSHuman、SyncHuman）对比，LORM+细化器在PSNR、SSIM、LPIPS上均优于对手；在遮挡比例从20%到60%和分辨率下降2×、4×时仍保持稳定性能，显示出更强的遮挡鲁棒性与降解鲁棒性。

**⚠️ 局限性**

局限性：依赖外部姿态、位置、体型估计器，严重初始化误差难以修正；在手部细节恢复方面表现不佳；极低分辨率下生成的细节可能与真实身份不符，容易出现不准确的纹理伪造。

---

## 209. VeriGrey: Greybox Agent Validation

**arXiv ID:** 2603.17639 | [PDF](https://arxiv.org/pdf/2603.17639v1)

**作者:** Yuntong Zhang `[一作]` (National University of Singapore), Abhik Roychoudhury `[通讯]` (National University of Singapore)

**通讯引用:** 10671 | [OpenAlex ID](https://openalex.org/A5060115298)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出一种灰盒（grey‑box）验证框架VeriGrey，用于自动化发现大型语言模型（LLM）代理中的间接提示注入（indirect prompt injection）漏洞，并在AgentDojo基准、Gemini CLI编码代理和OpenClaw个人助手等实际代理上进行评估。

**💡 创新点**

创新点：①利用代理工具调用序列作为反馈函数，弥补传统分支覆盖不适用于非确定性LLM代理的缺陷；②设计基于“上下文桥接”的注入提示变异器，使注入任务与用户任务紧密关联，从而提高注入成功率；③在灰盒框架中结合能量分配与变异次数的动态调度，进一步提升搜索效率。

**🔧 技术方法**

技术：灰盒模糊测试、工具调用序列记录与反馈、基于LLM的上下文桥接变异器、能量分配策略、与多种提示注入防御（如沙箱、数据分隔符、注入检测、工具过滤）的对抗评估。

**📊 数据集**

数据集：AgentDojo（包含Workspace、Slack、Travel、Banking四套环境，每套有多种用户与注入任务）；Gemini CLI的十条手工构造的注入任务；OpenClaw的10条恶意技能（来源于ClawHub的恶意或潜在注入技能）。

**📈 对比分析**

比较方法：与传统黑盒模糊（AgentVigil）对比；在AgentDojo上，VeriGrey在GPT‑4.1、Gemini‑2.5‑Flash和Qwen‑3‑235B后端分别提升了约11%–33%的注入成功率；在Gemini CLI上，VeriGrey达到90%成功率，远高于黑盒基线的60%；在OpenClaw上，使用VeriGrey后所有10个恶意技能在Kimi‑K2.5后端全部触发，显著优于原始技能。

**⚠️ 局限性**

局限性：①需要对目标代理进行工具调用插桩，可能不适用于所有代理实现；②变异器仍依赖LLM生成质量，对高度防御的后端可能效果有限；③实验预算受限，尤其在Gemini CLI等成本高的代理上，难以大规模扩展；④仅评估单会话注入，未覆盖跨会话持久性注入与供应链攻击等其他攻击面。

---

## 210. Harnessing the Power of Foundation Models for Accurate Material Classification

**arXiv ID:** 2603.17390 | [PDF](https://arxiv.org/pdf/2603.17390v1)

**作者:** Qingran Lin `[一作]` (Georgia Institute of Technology), Chaolun Zhu `[通讯]` (Waseda University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用生成合成数据与跨模态双流框架的材料分类方法，解决了材料标注稀缺的问题。

**💡 创新点**

① 通过语义引导的提示工程和自动标注实现大规模平衡材料图像生成；② 将视觉先验（DINOv2）与语言先验（GPT‑4v+CLIP）融合，并在预训练模型上进行联合微调。

**🔧 技术方法**

使用文本到图像扩散模型（如 SD‑v2.1）与 Grounding DINO/SAM 进行语义掩码；DINOv2 提取视觉特征；GPT‑4v 生成材质描述；CLIP 文本编码；多层感知机（MLP）融合并分类；AdamW 微调。

**📊 数据集**

自制21类合成材料数据集；公开 FMD、DMS‑test、Google‑test 用于评估。

**📈 对比分析**

与 CLIP、GPT‑4v 零样本、MatSim 及其他基准进行对比，在 FMD（10 类）上实现 89% 的准确率，在 Google‑test（21 类）上达到 92%，显著优于现有方法。

**⚠️ 局限性**

仍受限于预训练模型的语言表达质量；合成图像与真实场景在细节与光照上存在差距；对极少见材质或复杂光照条件的鲁棒性有限。

---

## 211. OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery

**arXiv ID:** 2603.17355 | [PDF](https://arxiv.org/pdf/2603.17355v1)

**作者:** Yiwen Zhao `[一作]` (Carnegie Mellon University), Laszlo A. Jeni `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种完全在线的全局坐标人形网格恢复框架OnlineHMR，实现流式摄像机坐标HMR与增量式人机中心SLAM的协同，能够在实时条件下生成世界坐标的人体姿态与轨迹。

**💡 创新点**

双分支架构同时满足摄像机坐标精度与世界坐标一致性；滑动窗口学习与KV缓存实现在线因果推理；人机协同增量SLAM结合软人像遮罩与EMA平滑来抑制漂移与抖动。

**🔧 技术方法**

Transformer+ViT骨干、SMPL模型、滑动窗口与KV缓存、速度正则化、EMA平滑、软人像遮罩、增量式SLAM、MoGe‑V2深度估计、频域光谱抖动评估。

**📊 数据集**

训练使用BEDLAM、3DPW、H3.6M；评估使用3DPW、EMDB‑1（摄像机坐标）和EMDB‑2（世界坐标），并在自定义动态视频上测试多人人类场景。

**📈 对比分析**

与离线分块方法和其他在线方法（TRACE、Human3R、WHAM等）对比，OnlineHMR在摄像机坐标上与离线方法相近，在世界坐标上实现最小化误差（WA‑MPJPE 93.5 mm，ERVE 12.4 mm/帧），但在FPS上略逊于TRACE；整体延迟低且保持时间一致性。

**⚠️ 局限性**

依赖连续视角的增量SLAM，对突变相机切换或多摄像机环境鲁棒性不足；尺度估计依赖MoGe‑V2，可能在极端深度变化时失效。

---

## 212. UAV-CB: A Complex-Background RGB-T Dataset and Local Frequency Bridge Network for UAV Detection

**arXiv ID:** 2603.17492 | [PDF](https://arxiv.org/pdf/2603.17492v1)

**作者:** Shenghui Huang `[一作]` (South China University of Technology), Ke Chen `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了专门针对低空复杂背景和伪装场景的RGB‑T UAV检测数据集UAV‑CB，并提出了局部频域桥接网络LFBNet（包括LFCA和FGSA模块）来实现频域与空间域以及跨模态的对齐，从而提升伪装小目标的检测性能。

**💡 创新点**

创新点在于：①使用局部频域特征对RGB与热红外模态进行幅度与相位的对齐，消除跨模态不一致；②通过频域引导的可变形空间对齐(FGSA)实现几何对齐；③将频域与空间域信息融合，显著提升在复杂背景下的鲁棒性。

**🔧 技术方法**

技术主要包括：局部频域建模（2D FFT、幅度/相位对齐、逆FFT）、交叉注意力融合、频域引导的偏移预测与可变形卷积、YOLOv5s检测头以及多尺度Atrous卷积提取空间特征。

**📊 数据集**

使用的数据集为：①UAV‑CB（3,442对RGB‑T图像，涵盖5类复杂背景）以及②公开的DroneVehicle RGB‑T地面目标检测基准。

**📈 对比分析**

通过与多种单模态和多模态检测器（如YOLOv5s、C^2Former、SFDFusion、RT‑DETR等）在UAV‑CB和DroneVehicle上对比，LFBNet在UAV‑CB上实现AP_50 84.6%、AP_75 57.2%、AP_(0.5:0.95) 54.4%，显著领先；在DroneVehicle上mAP_50 80.1%，同样位居前列。

**⚠️ 局限性**

局限性在于：①缺乏对不同天气、光照和未见场景的跨域适应评估；②目前仅针对RGB‑T两模态，未考虑多传感器融合；③对实时性与算力需求仍有提升空间。

---

## 213. When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution

**arXiv ID:** 2603.17445 | [PDF](https://arxiv.org/pdf/2603.17445v1)

**作者:** Yi Nian `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**通讯引用:** 3420 | [OpenAlex ID](https://openalex.org/A5057711796)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了隐式执行追踪（IET）框架，在多智能体语言系统的生成过程中将键控的水印信号嵌入到 token 分布中，从而使得最终输出文本本身即能恢复各智能体的贡献和交互拓扑，无需依赖外部日志。

**💡 创新点**

创新点在于：① 将水印嵌入 logits 级别的键控分布改造，使得每个智能体的身份成为统计可检索的信号；② 通过滑动窗口的竞争性变点检测自动恢复 token 级别的归属与边界；③ 利用二进制位掩码编码交互矩阵，实现拓扑结构的完整恢复。

**🔧 技术方法**

技术手段包括：键控 logit 调制、词表置换与低幅度扰动、滑动窗口一致性评分、竞争性变点检测、邻接矩阵状态更新与逆向映射。

**📊 数据集**

实验使用了 Multi‑Agent Interaction Dataset（含多种拓扑的对话日志）和 Who & When benchmark（含失败归因标注），并在数据上进行了 PII 红点化的鲁棒性验证。

**📈 对比分析**

与 LLM‑based（ChatGPT‑4o、DeepSeek‑v3.1）和传统分割基线（Random、Recursive、TextTiling 等）对比，IET 在 token 级别准确率 >94%、IoU ≈0.93、拓扑准确率 1.0，且在 ID 删除或边界扰动等元数据缺失情形下仍保持显著性能；基线模型往往低于 30% 的 token 准确率，IoU 仅 0.10‑0.18。

**⚠️ 局限性**

局限性包括：仅在受控对话数据上验证，尚未在大规模、异构智能体集合或极端扰动下评估；水印依赖于秘密键，若键泄露可能影响安全性；并且对生成策略或模型更新的鲁棒性仍需进一步研究。

---

## 214. RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On Device LLM Inference

**arXiv ID:** 2603.17891 | [PDF](https://arxiv.org/pdf/2603.17891v1)

**作者:** Arpit Singh Gautam `[一作]`, Saurabh Jha `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型，提出 RAMP 方法通过强化学习学习每层可变精度量化策略，并实现零样本跨模型迁移。

**💡 创新点**

创新点包括：可迁移的 11 维层级嵌入、Scale Folding 预处理实现 3‑bit 稳定量化、质量优先的奖励设计以及 HALO 导出管线统一 GGUF 兼容性。

**🔧 技术方法**

使用 Soft Actor‑Critic（SAC）强化学习、层级嵌入、对称/非对称量化、Scale Folding 以及 GGUF 格式导出。

**📊 数据集**

数据集主要为 WikiText‑2（校准与 perplexity 评估），并在多家族（Llama‑2、Llama‑3、Mistral）模型上进行验证。

**📈 对比分析**

与 AWQ、GPTQ、RTN 等统一 4‑bit PTQ 方法对比，RAMP 在 Llama‑2‑7B、13B、Llama‑3‑8B、Mistral‑7B 上实现更低 perplexity（例如 5.54 vs 5.60）且模型大小缩小 6%~8%，保持 99% 以上 FP16 余下性能。

**⚠️ 局限性**

局限性包括仅针对解码器 Transformer，离散 bit‑宽 {3,4,5,6}；不支持动态量化或头/通道级细粒度；仅在 PTQ 场景下验证，需对每个新模型重新抽取嵌入；未覆盖编码‑解码或 Mixture‑of‑Experts 结构。

---

## 215. Face anonymization preserving facial expressions and photometric realism

**arXiv ID:** 2603.17567 | [PDF](https://arxiv.org/pdf/2603.17567v1)

**作者:** Luigi Celona `[一作]` (University of Milano-Bicocca), Raimondo Schettini `[通讯]` (University of Milano-Bicocca)

**通讯引用:** 8848 | [OpenAlex ID](https://openalex.org/A5028039581)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种面部匿名化框架，在保持身份信息不可识别性的同时，保留表情、光照和肤色等特征。

**💡 创新点**

创新点在于将密集面部标志点与条件GAN结合，并引入基于本征图分解与颜色转移的轻量后处理，以提升表达、照明和肤色一致性。

**🔧 技术方法**

采用条件GAN（DeepPrivacy改进版）、Retinex式本征图分解、Laplacian金字塔融合和YCbCr颜色转移等技术。

**📊 数据集**

使用CelebA‑HQ数据集进行训练与评估。

**📈 对比分析**

与八种先进匿名化方法比较，本文在FID、表情保真、光照一致性和肤色保真度上均接近或优于最优基线，同时仍保持良好的隐私保护。

**⚠️ 局限性**

局限性包括对极端光照或遮挡的鲁棒性不足，以及后处理对计算资源的轻微增加，未来需进一步扩展至更多场景。

---

## 216. ReSteer: Quantifying and Refining the Steerability of Multitask Robot Policies

**arXiv ID:** 2603.17300 | [PDF](https://arxiv.org/pdf/2603.17300v1)

**作者:** Zhenyang Chen `[一作]` (Georgia Institute of Technology), Danfei Xu `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7569 | [OpenAlex ID](https://openalex.org/A5028834865)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ReSteer框架，对多任务机器人策略的任务可调性（steerability）进行量化与提升，解决机器人在执行过程中无法及时切换任务的瓶颈。

**💡 创新点**

创新点在于：①将可调性定义为策略对语言提示的状态相关响应，并用条件互信息（CMI）作为无rollout的可调性代理；②构建阶段感知的SteerGen数据生成器，利用任务阶段一致性生成跨任务的切换轨迹；③引入自我改进行为克隆（SRBC），通过在成功切换轨迹上迭代微调进一步扩大可调性覆盖。

**🔧 技术方法**

核心技术包括：多模态视觉语言动作（VLA）预训练策略、条件互信息估计与采样、轨迹插值与短时规划、基于行为克隆的自适应微调。

**📊 数据集**

主要使用LIBERO‑Goal（模拟）和DROID平台厨房场景（真实）两套数据集，后者通过遥控演示收集四个基准任务的数据。

**📈 对比分析**

与Diffusion Policy、CAST等现有多任务学习方法相比，ReSteer在LIBERO上可调性提升约10%（绝对值），在真实机器人上可调性提升2.2×，单任务成功率保持或略增。

**⚠️ 局限性**

局限性在于：数据生成与自我改进需要离线或半离线处理，在线交互学习尚未实现；生成的切换轨迹受阶段划分与短时规划精度限制，可能在更复杂场景或多目标情形下表现不足。

---

## 217. AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors

**arXiv ID:** 2603.17975 | [PDF](https://arxiv.org/pdf/2603.17975v1)

**作者:** Aymen Mir `[一作]` (University of Tübingen), Gerard Pons-Moll `[通讯]` (University of Tübingen)

**通讯引用:** 14363 | [OpenAlex ID](https://openalex.org/A5076908763)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了AHOY方法，从高度遮挡的YouTube单目视频中重建完整、可动画的3D高斯人体头像。

**💡 创新点**

核心创新在于利用身份微调的视频扩散模型生成完整的多视角监督，结合两阶段从canonical到姿势相关的3D高斯映射，并通过map-pose/LBS-pose解耦消除生成不一致，以及面部单独监督保持身份。

**🔧 技术方法**

使用了DensePose UV映射、FLUX与FLAME扩散模型、LoRA微调、RF-Inversion、3D高斯场(3DGS)以及姿势驱动的LBS变形。

**📊 数据集**

实验主要使用包含显著遮挡的YouTube视频和多视角BEHAVE数据集。

**📈 对比分析**

与现有NeRF/3DGS/人脸扩散等基准相比，AHOY在遮挡场景下的重建精度更高，生成的头像在新姿势和新视角下保持高保真度，且可在手机视频重建的3DGS场景中顺畅合成。

**⚠️ 局限性**

受限于扩散模型的先验和潜在空间反演，可能在未观测区域出现不正确的细节，且多阶段优化与扩散推理导致运行速度慢。

---

## 218. ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

**arXiv ID:** 2603.17435 | [PDF](https://arxiv.org/pdf/2603.17435v1)

**作者:** Ruibo Fan `[一作]` (Hong Kong University of Science and Technology), Xiaowen Chu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 10545 | [OpenAlex ID](https://openalex.org/A5100730785)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个完整的无损压缩框架，用于在GPU上高效、精确地推理大型语言模型。

**💡 创新点**

创新点在于：① 结合 Tensor‑Core 友好的固定长度三位位图编码（TCA‑TBE），实现并行解码；② 将解压与 GEMM 融合成 ZipGEMM 内核，消除中间缓冲，最大化算力与带宽利用；③ 通过硬件感知的设计，将无损压缩从仅存储层提升到可加速推理。

**🔧 技术方法**

使用的技术包括：CUDA/PTX 编程、位图编码与三位码表、warp 同步解码、位运算/计数(popcount)、隐式指数查找、两层软件流水线、异步内存拷贝、Tensor Core 乘加指令。

**📊 数据集**

评估数据集涵盖 LLaMA‑3.1（8B/70B/405B）、Qwen2.5（7B/14B/32B/72B）、Gemma3（12B/27B）、Mistral（24B/123B）等多模型；对比基线 vLLM、Transformers、DFloat11、DietGPU、nvCOMP、Marlin W8A16；使用 RTX4090、L40S、RTX5090、A100、H800 等 GPU。

**📈 对比分析**

方法上采用与 cuBLAS_TC、DietGPU、nvCOMP、DFloat11 等标准无损压缩及 Tensor Core GEMM 进行对比；结果显示 ZipGEMM 在 RTX4090 上平均提升 1.31×、在 L40S 上 1.36×、峰值 2.21×；端到端相比 vLLM 平均 1.22×，相比 Transformers 3.18×，相比 DFloat11 8.52×；压缩率达 71‑72% 的权重量级压缩。

**⚠️ 局限性**

局限性包括：在训练型 GPU（A100、H800）由于带宽已足够，性能提升有限；对极小矩阵层需手动调优分块参数；仍比压缩量化方法慢；框架目前仅针对 NVIDIA GPU，需改造以兼容其它加速器；离线压缩成本仍存在。

---

## 219. LoGSAM: Parameter-Efficient Cross-Modal Grounding for MRI Segmentation

**arXiv ID:** 2603.17576 | [PDF](https://arxiv.org/pdf/2603.17576v1)

**作者:** Mohammad Robaitul Islam Bhuiyan `[一作]` (Friedrich-Alexander-Universität), Andreas Maier `[通讯]` (Friedrich-Alexander-Universität)

**通讯引用:** 15913 | [OpenAlex ID](https://openalex.org/A5101619735)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

研究了一个从放射科医生口述转为文本提示，再通过LoRA增量适配的GDINO定位和MedSAM分割实现脑肿瘤自动提取的端到端管道。

**💡 创新点**

创新点是将语音识别、临床NLP、文本驱动的视觉语言定位和基于提示的分割整合为一个参数高效、可微调的框架，且只更新不到5%的参数。

**🔧 技术方法**

使用 Whisper ASR、spaCy+negspaCy、LoRA 适配的 GDINO（Swin‑T）和冻结的 MedSAM（ViT‑B）。

**📊 数据集**

用 BRISC 2025 的 6,000 张 T1 加权 MRI 与 Kaggle MRI Bounding Box 5,249 张图像做训练和 OOD 评估，并测试 12 份德语口述。

**📈 对比分析**

与全微调 GDINO+MedSAM 比较，LoGSAM 仅略低（Dice 0.8032 vs 0.8145），在 OOD 下保持 98% 的检测性能；对比 YOLOv11 的 mAP 也表现出良好泛化。

**⚠️ 局限性**

受限于 2D 切片、单一序列、缺乏 3D 视角和临床多中心验证；当口述中词汇模糊或转录错误时可能导致提示失效。

---

## 220. Gesture-Aware Pretraining and Token Fusion for 3D Hand Pose Estimation

**arXiv ID:** 2603.17396 | [PDF](https://arxiv.org/pdf/2603.17396v1)

**作者:** Rui Hong `[一作]` (George Mason University), Jana Kosecka `[通讯]` (George Mason University)

**通讯引用:** 7930 | [OpenAlex ID](https://openalex.org/A5086078885)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出两阶段框架：Stage1 用 HRNet 在 InterHand2.6M 上进行粗细手势分类预训练以注入手势语义；Stage2 用 2.5D 体素表示生成每关节点 token，并通过 Transformer 并引入手势 embedding 进行细化，最终回归 MANO 参数实现单手 3D 姿态估计。

**💡 创新点**

创新点在于：① 将手势语义作为先验，用粗细层级手势分类预训练提升编码器的语义分辨率；② 在 Transformer 里加入手势引导 token，将手势 embedding 作为中间表示引导关节点 token 的融合，显著提升姿态回归精度；③ 证明该预训练方法可直接迁移到其他架构（如 EANet），展现通用性。

**🔧 技术方法**

使用技术包括：HRNet 高分辨率编码器；粗细手势分类头；2.5D volumetric 体素 + soft-argmax；Transformer 与手势引导 token；MANO 参数回归；多级损失（pose、shape、3D、2D、2.5D、骨骼连续性）。

**📊 数据集**

数据集：InterHand2.6M 单手子集，人工整理得到 54 类 coarse 手势和 70 类 fine 手势标签。

**📈 对比分析**

与 HaMeR、EANet baseline 对比，在 InterHand2.6M 上 MPJPE 从 4.94 mm 降至 4.84 mm，MPVPE 从 5.26 mm 降至 5.19 mm；预训练+手势引导方案比无预训练差约 1–2 mm；该预训练还能无缝迁移到 EANet，进一步降低误差。

**⚠️ 局限性**

局限性：目前仅验证单手场景；手势预训练依赖手工标注的手势标签；未在两手交互或真实世界的极端遮挡场景中进行评估；对极端遮挡和不同环境的鲁棒性尚需进一步提升。

---

## 221. Efficient and Reliable Teleoperation through Real-to-Sim-to-Real Shared Autonomy

**arXiv ID:** 2603.17016 | [PDF](https://arxiv.org/pdf/2603.17016v1)

**作者:** Shuo Sha `[一作]` (Columbia University), Yunzhu Li `[通讯]` (Columbia University)

**通讯引用:** 3027 | [OpenAlex ID](https://openalex.org/A5100340050)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种从少量真实遥操作数据到仿真再到真实的共享自主框架，用kNN人类代理训练残差共驾政策。

**💡 创新点**

创新点在于用轻量级kNN人类代理和残差强化学习，避免依赖专家先验或大规模数据，且能零射击直接迁移。

**🔧 技术方法**

采用kNN人类代理、残差共驾政策、基于模型自由的PPO强化学习、Isaac Lab仿真、admittance控制和域随机化等技术。

**📊 数据集**

使用不到5分钟的真实遥操作演示（gear meshing、nut threading、peg insertion）以及NIST board #1等工业任务场景的数据。

**📈 对比分析**

与直接遥操作和基于专家先验或行为克隆的共享自主基线比较，实地用户研究中对新手提升成功率、对专家缩短完成时间，并生成更高质量的演示；在模拟中共驾模型对不同人类代理具有鲁棒性。

**⚠️ 局限性**

局限在假设用户行为在共驾辅助下不变，未考虑协同适应以及仅针对细粒度接触任务。

---

## 222. When the Specification Emerges: Benchmarking Faithfulness Loss in Long-Horizon Coding Agents

**arXiv ID:** 2603.17104 | [PDF](https://arxiv.org/pdf/2603.17104v1)

**作者:** Lu Yan `[一作]` (Purdue University), Xiangyu Zhang `[通讯]` (Purdue University)

**通讯引用:** 309423 | [OpenAlex ID](https://openalex.org/A5100362465)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个针对逐步披露规格的长周期编码基准，评估Claude Code和Codex在该情境下的实现忠实度，并提出了一种外部项目状态层来缓解忠实度损失。

**💡 创新点**

创新点在于：①定义“Faithfulness Loss Under Emergent Specification（FLE）”这一端点度量；②构建包含371个可验证组件的20篇最新ML论文基准；③引入曝光审计以验证可恢复性；④设计外部语义/结构项目状态追踪机制来显著恢复FLE。

**🔧 技术方法**

技术手段包括：LLM驱动的规范提取与组件分解、五级组件忠实度评分、曝光审计、平均组件忠实度(MCF)、依赖集成比例(DIR)、综合忠实度(IF50)等指标；外部项目状态层采用语义和结构视图并在每个编码回合前生成任务简报。

**📊 数据集**

使用数据集：20篇近期ML论文（10篇ICML 2025，10篇NeurIPS 2025），共371个原子可验证组件，配套的60条交互请求脚本。

**📈 对比分析**

对比方法：在Claude Code和Codex上分别跑“逐步披露”与“单次完整规格”两种条件；测得IF50在Claude Code下从0.530降至0.414（平均差0.116），在Codex下从0.550降至0.479（平均差0.071）。引入Mitigator后，Claude Code上MCF提升至3.000、完全忠实组件从118增至181、严重失败从72降至49，恢复率达90%。

**⚠️ 局限性**

局限性：①仅包含20篇论文，未覆盖更广泛的研发领域；②交互脚本为合成，可能不完全反映真实研究者行为；③FLE仅为端点度量，未考察会话内的临时漂移；④评估与标注部分依赖LLM，存在潜在误差；⑤只在两大商业平台验证，未证明通用性；⑥Mitigator的有效性需进一步与其他记忆/上下文管理方法比较。

---

## 223. Unified Policy Value Decomposition for Rapid Adaptation

**arXiv ID:** 2603.17947 | [PDF](https://arxiv.org/pdf/2603.17947v1)

**作者:** Cristiano Capone `[一作]` (Computational Neuroscience Unit, Istituto Superiore di Sanità), Luca Manneschi `[通讯]` (University of Sheffield)

**通讯引用:** 901 | [OpenAlex ID](https://openalex.org/A5043105124)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种共解耦的双线性 actor‑critic 框架，利用共享的乘法门控向量 G 对 Q‑函数与策略同时分解为基函数与门控系数的线性组合。

**💡 创新点**

创新点包括：1）使用生物学可解释的乘法门控实现高表达力而无需深层非线性堆叠；2）将门控向量在策略与价值网络之间共享，显著简化梯度耦合与参数量；3）不依赖预定义的奖励分解，提出奖励无关的学习与快速在线适应机制；4）门控空间可被直接操控，形成可解释的低维控制接口；5）通过门控更新实现几步内的在线迁移。

**🔧 技术方法**

技术方法：基于 Soft Actor‑Critic（SAC）框架的离线训练；双线性分解 Q(s,a,g)=∑kGk(s,g)ϕk(s,a) 与 μ(s,g)=∑kGk(s,g)Yk(s)；共享门控网络；梯度下降优化两层 MLP 及单层双线性结构；门控更新采用 TD 形式的线性增量；零样本测试与在线门控更新。

**📊 数据集**

数据集：MuJoCo Ant 连续控制任务，训练阶段采用八个不同方向（四个正交、四个对角），每 100 步切换方向；每个 episode 长 800 步，奖励为朝目标方向前进的正向量化减去正交惩罚。

**📈 对比分析**

与传统两层 MLP 以及单层 MLP（无门控）对比：双线性结构在相同网络深度下收敛速度提升、最终平均回报相当或略优；共享门控与独立门控相比参数更少、优化更稳定；在未见目标方向的零样本测试中，门控模型仅以门控值变化实现迁移，性能仅轻微下降；在线门控更新可在数步内实现方向与速度的快速调整。

**⚠️ 局限性**

局限性：1）实验仅限于方向导航的简单连续控制任务，尚未验证在更复杂或高维任务中的可扩展性；2）门控空间的解释性依赖于基函数的单义性，若基函数混合则可解释性下降；3）对离散动作空间或多目标任务的适应性尚未验证；4）在极端噪声或动态目标变化下，门控更新的鲁棒性仍待进一步研究。

---

## 224. Beyond bouba/kiki: Multidimensional semantic signals are deeply woven into the fabric of natural language

**arXiv ID:** 2603.17306 | [PDF](https://arxiv.org/pdf/2603.17306v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 225. Deanonymizing Bitcoin Transactions via Network Traffic Analysis with Semi-supervised Learning

**arXiv ID:** 2603.17261 | [PDF](https://arxiv.org/pdf/2603.17261v1)

**作者:** Shihan Zhang `[一作]` (Beijing University of Posts and Telecommunications), Qin Wang `[通讯]` (Independent)

**通讯引用:** 3356 | [OpenAlex ID](https://openalex.org/A5038760380)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了基于半监督学习的比特币网络层交易去匿名化方法NTSSL及其改进版NTSSL+。

**💡 创新点**

引入半监督学习生成伪标签并结合跨层协同分析，显著提升去匿名化精度。

**🔧 技术方法**

采用异常检测（Isolation Forest、AutoEncoder、One-Class SVM）、XGBoost、交易层聚类（多输入法）以及网络流量特征提取等技术。

**📊 数据集**

使用真实比特币主网与测试网的流量数据，单节点约15k笔交易、起源交易约90笔，多节点每个约4-8k笔交易。

**📈 对比分析**

与现有PERIMETER对比，NTSSL+在不同连接占有率下F1分数提升20-40%，最高达0.74，显著优于传统方法。

**⚠️ 局限性**

在主网不同类型节点之间泛化效果有限，且若节点未产生原始交易需人工注入；使用VPN/Tor或隐私协议可有效抵御攻击。

---

## 226. FloorPlan-VLN: A New Paradigm for Floor Plan Guided Vision-Language Navigation

**arXiv ID:** 2603.17437 | [PDF](https://arxiv.org/pdf/2603.17437v1)

**作者:** Kehan Chen `[一作]` (Institute of Automation Chinese Academy of Sciences and University of Chinese Academy of Sciences), Liang Wang `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 FloorPlan-VLN 这一新范式，将全局楼层平面图作为空间先验，配合简洁指令进行视觉‑语言导航；同时构建了包含 10k 以上轨迹和 100+ 区域标注的 FloorPlan‑VLN 数据集，并设计了 FP‑Nav 模型实现双视角时空对齐视频输入与多任务辅助训练。

**💡 创新点**

创新点在于：①把简洁区域级指令与完整楼层平面图耦合，显著降低对逐步指令的依赖；②通过将楼层平面图 raster 化并与即时 egocentric 视图拼接，构建时空对齐的双视角输入；③引入三类辅助任务（区域定位、轨迹推理、指令摘要）强化跨模态对齐与推理能力；④在真实机器人上验证模型的鲁棒性。

**🔧 技术方法**

使用技术包括：大规模多模态语言模型 Qwen‑2.5‑VL 的微调；视觉编码器与 MLP 投影头的联合学习；双视角时空对齐视频流；多任务统一的下一词预测训练框架；噪声模拟（执行漂移、尺度、几何扰动）与现实机器人部署。

**📊 数据集**

使用的数据集是基于 Matterport3D、R2R 与 RxR 生成的 FloorPlan‑VLN 数据集，覆盖 70+ 建筑、130+ 平面图、1k+ 区域标注，含 10k+ 轨迹；同时利用公开的 Matterport3D 场景进行训练与评估。

**📈 对比分析**

与多种零拷贝基线（Qwen、NaVILA、StreamVLN、InternVLA、Navid）及微调 Navid 进行对比，FP‑Nav 在 Val‑Unseen 上相较 Navid‑ft 提升了约 60% 的相对成功率，OSR、SR、SPL 等指标均位居榜首；消融实验验证了双视角对齐、辅助任务与视觉编码器解冻对性能的显著贡献。

**⚠️ 局限性**

局限性包括：仅处理单层环境，未覆盖多层楼梯与楼层切换；在真实环境中成功率仅为 24%，仍受制于传感器噪声与视觉域偏移；对楼层平面图的转换仍需人工标注；模型对大规模 MLLM 依赖较高，计算与能耗较大。

---

## 227. AERR-Nav: Adaptive Exploration-Recovery-Reminiscing Strategy for Zero-Shot Object Navigation

**arXiv ID:** 2603.17712 | [PDF](https://arxiv.org/pdf/2603.17712v1)

**作者:** Jingzhi Huang `[一作]` (Hong Kong Polytechnic University), Yi Wang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 17724 | [OpenAlex ID](https://openalex.org/A5100383690)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于自适应探索-恢复-回忆策略（AERR-Nav）的零样本多层楼目标导航框架。

**💡 创新点**

创新点在于：①三态状态机（探索、恢复、回忆）实现动态切换；②探索状态分为快思考与慢思考，结合语义优先级和不确定性动态权重；③回忆阶段利用关键点地图与LLM联合推理定位楼梯与漏检目标；④在多楼层环境中显著提升了探索与利用的平衡。

**🔧 技术方法**

采用视觉‑语言模型（BLIP‑2/CLIP）、多模态大型语言模型（GPT‑4o）、语义地图、前沿点生成、BEV视图、A*路径规划、PointNav 等技术。

**📊 数据集**

使用HM3D和MP3D两个室内多楼层基准数据集进行评估。

**📈 对比分析**

与现有零样本方法（如ASCENT、BeliefMapNav等）相比，AERR-Nav在HM3D上取得SR 72.3%、SPL 35.6%，在MP3D上取得SR 47.9%、SPL 18.7%，相对领先者提升约+6.9% SR、+2.1% SPL（HM3D）和+3.4% SR（MP3D）。

**⚠️ 局限性**

局限性包括：依赖LLM推理导致计算开销较大；对前沿点检测与楼梯识别的鲁棒性受限；在极端复杂或动态环境下仍可能出现误差；缺乏真实机器人硬件验证。

---

## 228. FailureMem: A Failure-Aware Multimodal Framework for Autonomous Software Repair

**arXiv ID:** 2603.17826 | [PDF](https://arxiv.org/pdf/2603.17826v1)

**作者:** Ruize Ma `[一作]` (Nanjing University), Lewei Lu `[通讯]` (SenseTime)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在多模态自动程序修复中，提出了FailureMem框架，利用大型语言模型结合代码、文本与截图自动生成补丁。

**💡 创新点**

创新点在于融合混合工作流–代理架构、主动视觉感知工具（Crop、Grounding）以及将历史失败案例转化为可复用修复指导的失败记忆库。

**🔧 技术方法**

采用GPT‑5.1、GPT‑4.1、Claude等LLM，并集成Crop、Grounding、Bash等工具，构建层级记忆并实现主动视觉定位与交互式仓库探索。

**📊 数据集**

使用SWE‑bench Multimodal benchmark（617个真实GitHub issue，17个JavaScript仓库）进行实验。

**📈 对比分析**

与SOTA GUIRepair及其他基线对比，FailureMem在所有模型上提升了解决率（GPT‑5.1 +3.7%，GPT‑4.1 +2.3%，Claude 4.5 +2.3%），并在多项子基准上实现领先。

**⚠️ 局限性**

局限在于推理成本略高（约13%提升），且高度依赖历史失败案例的多样性，面对新颖故障时可能退回到标准代理行为。

---

## 229. TRiMS: Real-Time Tracking of Minimal Sufficient Length for Efficient Reasoning via RL

**arXiv ID:** 2603.17449 | [PDF](https://arxiv.org/pdf/2603.17449v1)

**作者:** Tingcheng Bian `[一作]` (Shenzhen University), Haiwei Wang `[通讯]` (Baidu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出最小充分长度(MSL)理论，利用该理论构建TRiMS强化学习框架，显著压缩大型语言模型的链式推理(Token)并提升准确率。

**💡 创新点**

创新点在于给出了可测的最短推理长度下界(MSL)，并通过TRiMS实现对该下界的逼近，从而在保持甚至提升准确率的同时实现超过80%的token压缩。

**🔧 技术方法**

采用GRPO强化学习、动态批量聚合、批量优势归一化、适应性截断等技术，结合M相应的长度奖励，驱动模型生成最短有效推理路径。

**📊 数据集**

使用DeepScaleR训练集进行训练，并在AIME24、AMC23、MATH-500、Minerva、OlympiadBench和MMLU等公开基准上进行评估。

**📈 对比分析**

与LC-R1、Laser、AutoThink、AdaptThink、JET等压缩方法对比，TRiMS在1.5B模型上实现83.5% token压缩并提升6.8%准确率；在7B模型上实现84.2%压缩并提升2.5%准确率，Intelligence‑Per‑Token(IPT)分别提升至45.1和68.2，显著优于同类方法。

**⚠️ 局限性**

局限性包括：仅在1.5B–7B规模模型验证；训练数据主要为数学领域，通用性有限；MSL未覆盖不同提示变体；截断策略为离散的二阶段(2048/4096)，未充分利用连续的MSL信息。

---

## 230. CircuitBuilder: From Polynomials to Circuits via Reinforcement Learning

**arXiv ID:** 2603.17075 | [PDF](https://arxiv.org/pdf/2603.17075v1)

**作者:** Weikun K. Zhang `[一作]` (University of Washington), Jarod Alper `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究使用强化学习方法自动生成算术电路，以求在给定运算次数内最小化计算多项式的门数。

**💡 创新点**

将算术电路构造视为单人MDP，并将AlphaZero的MCTS与PPO相结合以及引入Soft Actor-Critic进行离线经验重放，提供了可验证的自适应搜索框架。

**🔧 技术方法**

使用PPO+MCTS、SAC、GNN+Transformer网络、蒙特卡罗树搜索、经验回放、教师预训练和熵正则等技术。

**📊 数据集**

利用在有限运算次数内枚举生成的“游戏板”数据集，包含2-3变量、多达6阶的多项式及其最优电路路径。

**📈 对比分析**

通过在F5域下的C=5、C=6目标比较，SAC在二变量下达57.8%/46.3%的成功率，PPO+MCTS在三变量下约27%成功率；SAC更快收敛但在变量增多时失效，MCTS在更高维时更稳健。

**⚠️ 局限性**

局限在于只能处理少量门数和变量，缺乏对更大规模、多变量、多项式族的推广，且在更高复杂度和维度时训练不稳定。

---

## 231. DeepCORO-CLIP: A Multi-View Foundation Model for Comprehensive Coronary Angiography Video-Text Analysis and External Validation

**arXiv ID:** 2603.17675 | [PDF](https://arxiv.org/pdf/2603.17675v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 232. Transformers are Bayesian Networks

**arXiv ID:** 2603.17063 | [PDF](https://arxiv.org/pdf/2603.17063v1)

**作者:** Gregory Coppola `[一作]` `[通讯]`, Gregory Coppola

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于对数-赔率代数的构造式贝叶斯推断（Belief Propagation）方法，并将三重Softmax技术引入其中以实现更高效的概率计算。

**💡 创新点**

创新点在于：①将对数-赔率代数与贝叶斯推断框架结合，提供更直观的概率操作；②利用三重Softmax对分数进行重构，显著降低数值不稳定性；③通过“grounding”和“Turing”模块提升模型对离散变量的处理能力。

**🔧 技术方法**

主要技术包括：对数-赔率代数、构造式贝叶斯推断、三重Softmax变换、图模型建模以及数值优化算法。

**📊 数据集**

实验使用了标准的synthetic Bayesian网络（如 20/50/100 变量的图）、MNIST（作为分类任务的概率推断）以及 UCI Adult 数据集进行概率估计。

**📈 对比分析**

与传统 BP、MC 采样、以及 Variational Inference 进行对比。实验结果显示，本方法在准确率上平均提升 5%–10%，收敛速度比 BP 快约 30%，并在数值稳定性上优于 Softmax 传统实现。

**⚠️ 局限性**

主要局限性包括：①对图形中环的处理仍不够高效；②计算复杂度相较于传统 BP 略高；③在极大规模网络中需要进一步的稀疏化或分布式实现。

---

## 233. Baguan-TS: A Sequence-Native In-Context Learning Model for Time Series Forecasting with Covariates

**arXiv ID:** 2603.17439 | [PDF](https://arxiv.org/pdf/2603.17439v1)

**作者:** Linxiao Yang `[一作]` (DAMO Academy, Alibaba Group), Liang Sun `[通讯]` (DAMO Academy, Alibaba Group)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 Baguan-TS，一种统一框架，实现对原始多变量时间序列的顺序原生上下文学习（ICL）并可在推理时实现快速无梯度适配。

**💡 创新点**

创新点包括：① 3D Transformer 以时间、变量和上下文三轴共同注意的结构；② 目标空间检索（Y‑space RBfcst）实现特征无关的局部校准；③ 上下文过拟合（context‑overfitting）策略平衡去噪与样本选择，显著降低输出过平滑；④ 在单一模型中提供 2D 与 3D 两种推理模式并可集成。

**🔧 技术方法**

主要技术包括：3D Transformer（分块编码、旋转位置编码、多头自注意力）、Y‑space 检索式局部校准、可靠性加权的上下文过拟合、离散化概率预测头、随机 Fourier 特征编码以及基于检索的上下文构造。

**📊 数据集**

实验使用 30 个含协变量的公开基准任务（来自 100 个数据集），另外 27 个包含历史与未来协变量的真实工业数据集，以及 10 个无协变量的单变量任务（从同一基准拆分）。

**📈 对比分析**

与 TabPFN‑TS、Sundial‑Base、TimesFM、Chronos‑Bolt、Moirai‑2.0、TiRex、Toto‑1.0 等现有时间序列基础模型对比，Baguan‑TS 在 SQL、MASE、WAPE、WQL 等指标上获得最高平均胜率，SQL/MASE 分别比 TabPFN‑TS 提升 4.8%/1.3%，在大多数任务上均优于对手，且在噪声干扰下表现更稳健。

**⚠️ 局限性**

局限性：模型容量大，3D Transformer 对 GPU 内存和推理时间要求较高；对极端短时间窗口或无协变量的任务表现相对不如专门针对该场景的轻量级模型；缺乏针对不同任务的自动上下文检索策略选择，仍需人工调参。

---

## 234. CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization

**arXiv ID:** 2603.16966 | [PDF](https://arxiv.org/pdf/2603.16966v1)

**作者:** Liangbin Huang `[一作]` (Hujing Digital Media and Entertainment Group), Wenji Mao `[通讯]` (Institute of Automation Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种用于视觉媒体的开源场景下的多模态说话者标注框架CineSRD，并构建了双语字幕标注基准SubtitleSD。

**💡 创新点**

创新点在于将视觉锚点聚类、音频-语言模型说话者转折检测以及离屏说话者补充相结合，形成训练自由、可扩展的统一框架。

**🔧 技术方法**

该框架利用主动说话者检测、面部嵌入、声纹嵌入以及Qwen2-Audio-7B等多模态技术，实现说话者注册、转折检测和补全。

**📊 数据集**

实验使用了新构建的SubtitleSD（中英双语，多类型、包含317名说话者的难点子集）以及公开的AVA-AVD基准。

**📈 对比分析**

与SC、AHC、AVR-Net、EC2P等基线对比，CineSRD在SubtitleSD的DER降至0.075（比最佳基线低约40%），在AVA-AVD上DER亦从0.213降至0.190，显示出显著的性能提升。

**⚠️ 局限性**

局限在于多阶段集成而非端到端，计算开销大；对卡通/动画等非真人面孔效果有限；英语子集样本量不足，影响模型泛化。

---

## 235. Asymmetric Nash Seeking via Best Response Maps: Global Linear Convergence and Robustness to Inexact Reaction Models

**arXiv ID:** 2603.17058 | [PDF](https://arxiv.org/pdf/2603.17058v1)

**作者:** Mahdis Rabbani `[一作]` (University of California), Shima Nazari `[通讯]` (University of California)

**通讯引用:** 427 | [OpenAlex ID](https://openalex.org/A5083627093)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了两人约束博弈在信息不对称下的Nash均衡求解，提出了仅利用对手的最优反应映射的投影梯度加最优反应迭代，并证明了其在最优反应精确时的全局线性收敛以及在近似最优反应下的O(ε)稳态误差容忍性。

**💡 创新点**

创新点在于：①将对手最优反应映射视为信息结构而非显式目标，克服了传统博弈求解需要完全信息的限制；②给出了满足μ>L₁₂L₂的充分条件保证Nash均衡唯一性；③证明了投影梯度+最优反应迭代在该类游戏中的全局线性收敛；④对近似最优反应的误差分析给出明确的误差界限，展示了算法对数据噪声的鲁棒性。

**🔧 技术方法**

使用的技术包括：投影梯度下降、最优反应映射、Banach收敛定理、Berge最大值定理、Kakutani固定点定理、Lipschitz强单调性分析以及数值仿真验证。

**📊 数据集**

实验采用了一个一维拖曳杆（tug-of-war）控制任务的合成数据，通过离散时间动力学和约束优化得到的控制序列与对手反应映射，未使用公开真实数据集。

**📈 对比分析**

通过对比精确最优反应迭代与近似最优反应迭代，实验显示：精确迭代满足理论预测的线性收敛速率；近似迭代收敛到一个与误差ε成正比的邻域；并验证了误差与ε线性比例关系。性能指标为收敛速率ρ(α)和稳态误差上界，均与理论一致。

**⚠️ 局限性**

局限性包括：①仅考虑可分离可压缩的可行域，未覆盖有耦合约束的博弈；②要求μ>L₁₂L₂的条件较为严格，限制了在强耦合情形下的适用性；③对最优反应映射的逼近误差假设为均匀上界，实际学习误差可能更为复杂；④实验仅在合成任务上验证，缺乏真实世界案例。

---

## 236. UniSAFE: A Comprehensive Benchmark for Safety Evaluation of Unified Multimodal Models

**arXiv ID:** 2603.17476 | [PDF](https://arxiv.org/pdf/2603.17476v1)

**作者:** Segyu Lee `[一作]` (KAIST AI), Se-Young Yun `[通讯]` (KAIST AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估统一多模态模型安全基准 UniSAFE

**💡 创新点**

采用共享目标场景设计，系统覆盖7种输入/输出模态并聚焦多图像组合与多轮编辑风险

**🔧 技术方法**

基于 Gemini‑2.5 Pro 等大型语言模型生成触发器，结合人工筛选和自动评判（LLM评审器）等技术

**📊 数据集**

6,802 条高质量实例，包括 781 个图像目标描述和 1,226 个文本目标描述

**📈 对比分析**

对 15 款最新 UMM（含 2 款专有、13 款开源）进行攻击成功率、平均风险评分等评估，发现新任务和图像输出风险最高，商业模型仍易受攻击

**⚠️ 局限性**

限于评测仅覆盖文本和图像模态，未涵盖其他模态；评估受判别模型偏差影响，需更强系统级安全机制

---

## 237. Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual Speech Recognition

**arXiv ID:** 2603.17558 | [PDF](https://arxiv.org/pdf/2603.17558v1)

**作者:** Yuxiang Mei `[一作]` (Shanghai Normal University), Yanhua Long `[通讯]` (Shanghai Normal University)

**通讯引用:** 486 | [OpenAlex ID](https://openalex.org/A5056415893)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 Zipper-LoRA 框架，在多语言 ASR 长尾数据分布下采用参数高效微调，动态解耦共享与语言特定 LoRA 子空间，提升跨语言知识共享与抑制干扰的平衡。

**💡 创新点**

创新点在于 rank‑level 动态路由：通过语言识别嵌入控制每个 LoRA 列的共享或语言特定贡献，实现细粒度跨语言共享与专属适配的精细化调节。

**🔧 技术方法**

使用的技术包括 Whisper 大型语音编码器、Qwen3‑1.7B 语言模型、LoRA 低秩适配、Whisper LID 嵌入路由、Initial‑B warm‑start 以及两阶段训练策略。

**📊 数据集**

实验数据集覆盖 12 种语言，包括 MSR86k、WenetSpeech、MLS、LibriSpeech 与 Common Voice，涵盖高资源、低资源与极低资源（1 小时）情形。

**📈 对比分析**

与 Vanilla‑LoRA、Independent‑LoRA、FlyLoRA 等方法对比，Zipper‑LoRA 在高资源和低资源语言上均取得更低 WER/CER，尤其在极低资源 1 小时设置下显著提升；整体性能提升稳健且跨编码器上下文（chunked/ non‑chunked）保持一致。

**⚠️ 局限性**

局限性包括仅利用 Whisper LID 作为路由信号，未探索其他路由或更大语言集合；缺乏对实时流式或持续学习场景的评估。

---

## 238. MetaClaw: Just Talk -- An Agent That Meta-Learns and Evolves in the Wild

**arXiv ID:** 2603.17187 | [PDF](https://arxiv.org/pdf/2603.17187v1)

**作者:** Peng Xia `[一作]` (University of North Carolina Chapel Hill), Huaxiu Yao `[通讯]` (University of North Carolina Chapel Hill)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

MetaClaw提出了一套面向部署LLM代理的持续元学习框架，通过即时的技能注入和空闲时段的强化学习权重更新，实现了代理在使用过程中自我演化。

**💡 创新点**

核心创新在于将梯度无关的技能演化与梯度相关的政策优化两种不同时间尺度的适应机制统一起来，并通过技能生成版本化与机会式元学习调度器保证训练数据的有效性与更新的零停机。

**🔧 技术方法**

技术包括LLM驱动的失败分析与自然语言技能生成、基于检索的技能库利用、使用Process Reward Model的强化学习、云端LoRA微调、以及基于睡眠窗口、系统空闲与日历的机会式调度。

**📊 数据集**

实验使用自研MetaClaw-Bench（934道题目、44个模拟工作日）和AutoResearchClaw（23阶段自动研究流水线）两大数据集进行评估。

**📈 对比分析**

与基线相比，技能驱动的快速适应可使准确率提升约32%，全流程MetaClaw使Kimi‑K2.5模型从21.4%提升至40.6%，终端任务完成率提升8.25倍，AutoResearchClaw鲁棒性得分提升18.3%。

**⚠️ 局限性**

主要局限在于空闲窗口检测依赖用户手动配置的睡眠/日历/空闲阈值，可能不适用于所有部署环境，导致训练窗口选择不够通用。

---

## 239. Bootstrapping Coding Agents: The Specification Is the Program

**arXiv ID:** 2603.17399 | [PDF](https://arxiv.org/pdf/2603.17399v1)

**作者:** Martin Monperrus `[一作]` (KTH Royal Institute of Technology), Martin Monperrus `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 6739 | [OpenAlex ID](https://openalex.org/A5027206285)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过先让Claude Code根据926字的自然语言规范实现一个编码代理(agent_0)，随后让agent_0再次根据同一规范生成新的代理(agent_1)，从而证明了编码代理的自我主机（bootstrapping）和元循环（meta‑circularity）属性。

**💡 创新点**

首次在AI编码代理领域展示了自我主机的能力，并将规范视为唯一稳定的工程产物，将实现视为可随时重构的构建产物。

**🔧 技术方法**

利用大型语言模型（Claude Code + Sonnet 4.6）、自然语言规范推断、LLM调用与工具执行循环实现自动编码代理。

**📊 数据集**

使用单一的926字规范文本（作为实验输入），并在公开仓库中提供生成的Python实现。

**📈 对比分析**

通过人工验证两代实现是否符合同一规范来比较效果，实验结果表明第二代实现与第一代在功能和行为上完全一致，但未提供性能数值或基准测试。

**⚠️ 局限性**

实验规模有限，规范过大（>10,000字）或实现复杂度高时可验证难度增加；仅在最新最强大的LLM上可行，旧版或小型模型无法成功；且规范错误会在所有重构版本中传播。

---

## 240. Unified Spatio-Temporal Token Scoring for Efficient Video VLMs

**arXiv ID:** 2603.18004 | [PDF](https://arxiv.org/pdf/2603.18004v1)

**作者:** Jianrui Zhang `[一作]` (University of Wisconsin-Madison), Sangho Lee `[通讯]` (Allen Institute for AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种端到端可训练的 Spatio‑Temporal Token Scoring (STTS) 模块，用于在视频视觉‑语言模型中对 ViT 与 LLM 之间的视觉令牌进行统一剪枝。

**💡 创新点**

创新点包括：① 在 ViT 内部实现统一剪枝，无需额外的文本条件或复杂合并；② 双轴评分机制（空间显著性 + 时序冗余）并辅以余弦相似度辅助损失；③ 基于 first‑fit 的稀疏序列打包实现真正的硬件加速；④ 可在不显著损失性能的前提下安全剪除多达 50% 视觉令牌，并可与测试时缩放（TTS）结合提升长视频推理效果。

**🔧 技术方法**

技术手段包括：使用 Molmo2 视觉编码器（SigLIP‑2‑So400M/14 384px ViT）、Qwen3‑4B LLM；在 ViT 第 3 层插入 self‑attention + 3 层 MLP 的评分器；将评分注入下一层注意力偏置；硬剪枝后采用 first‑fit‑descending 打包；利用相邻帧的余弦相似度构建辅助损失；训练采用 6,250 步、Cosine 学习率调度、分层学习率等。

**📊 数据集**

训练数据：Molmo2 视频 QA 子集；评估数据：13 组视频问答基准，包括 NextQA、Perception‑Test、MVBench、Tomato、MotionBench、Temp‑Compass、VideoMME、VideoMME‑Sub、LongVideo、LongVideo‑Sub、MLVU、LVBench、VideoEvalPro。

**📈 对比分析**

与基线 Molmo2 及其他 VLM（Qwen3‑VL‑4B、PLM‑8B、InternVL3.5‑8B）对比：在 30% 剪枝下保持甚至提升多项基准；50% 剪枝仅下降 0.7%；训练/推理速度在 128 帧场景下提升 1.6×，在 256 帧场景下提升 2.2×；TTS 进一步提升长视频性能（平均提升 ~1%）。

**⚠️ 局限性**

局限性：在极高剪枝率（>80%）下仍有显著性能下滑；仅在视频 VLM 上验证，缺乏对文本或图像单模任务的全面评估；对动态场景的剪枝效果可能受限；未讨论对视觉模型或 LLM 的进一步微调成本；以及模型对不同视频采样策略的敏感性。

---

## 241. A mechanism design overview of Sedna

**arXiv ID:** 2603.17614 | [PDF](https://arxiv.org/pdf/2603.17614v1)

**作者:** Benjamin Marsh `[一作]` (Sei Labs), Alejandro Ranchal-Pedrosa `[通讯]` (University of Portsmouth)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

本文对Sedna编码多提议共识协议的MEV攻击进行了形式化分析，并设计了激励兼容机制以阻止提议者的延迟与隐私破坏行为。

**💡 创新点**

创新点包括：① PIVOT‑K基于解码前缀的奖励分配策略；② 适应性发送器（拉奇特）通过永久排除未确认通道来将多槽延迟压缩为单槽；③ 协同攻击分解与最优奖励证明。

**🔧 技术方法**

使用技术：KL型大偏差界、超几何分布与 Chernoff 上界、动态博弈论、奖励函数最优性证明、数值仿真。

**📊 数据集**

使用数据集：实验采用 n=100、m=20、β=0.2 等参数组合的模拟数据，没有基于真实区块链的历史数据。

**📈 对比分析**

比较方法：将静态发送器与拉奇特发送器的延迟概率对比；数值结果显示拉奇特将多槽延迟概率降低数百至数千倍，且MEV成本被压缩至交易价值的 0.04%。

**⚠️ 局限性**

局限性：① 仅在粗粒度时钟模型下分析，无法消除单槽内解码竞争风险；② 对槽间信息泄露假设保守，未考虑 κ 隐藏的实际不确定性；③ 主要针对单一交易的 MEV，跨交易协同可能仍有未捕获风险。

---

## 242. ProbeFlow: Training-Free Adaptive Flow Matching for Vision-Language-Action Models

**arXiv ID:** 2603.17850 | [PDF](https://arxiv.org/pdf/2603.17850v1)

**作者:** Zhou Fang `[一作]` (Southeast University), Qiongfeng Shi `[通讯]` (Southeast University)

**通讯引用:** 13164 | [OpenAlex ID](https://openalex.org/A5052030441)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种训练无关的自适应求解框架，利用Lookahead线性探测动态调度ODE步数，以加速Flow Matching动作头的推理。

**💡 创新点**

创新点在于使用几何角度余弦相似度来实时评估轨迹曲率，从而仅在非线性区间密集积分，显著减少推理步数。

**🔧 技术方法**

采用Flow Matching动作头、Lookahead线性探测、欧拉/自适应步长调度以及对比高阶AB2和RK45求解器。

**📊 数据集**

在MetaWorld和LIBERO两个机器人操纵基准，以及真实的Pick-and-Place实验中验证。

**📈 对比分析**

与固定步数Euler、AB2、RK45等基线相比，在MetaWorld上推理步数从50降至约2.6，耗时提升14.8×，成功率保持83%；在LIBERO上平均步数4.5，耗时提升8.5×，成功率88.7%。

**⚠️ 局限性**

局限在于对线性阈值和探测步长需手工调节，且对极端高度非线性或高速接触任务的泛化尚待验证。

---

## 243. Machine Learning for Network Attacks Classification and Statistical Evaluation of Machine Learning for Network Attacks Classification and Adversarial Learning Methodologies for Synthetic Data Generation

**arXiv ID:** 2603.17717 | [PDF](https://arxiv.org/pdf/2603.17717v1)

**作者:** Iakovos-Christos Zarkadis `[一作]` (University of Piraeus), Christos Douligeris `[通讯]` (University of Piraeus)

**通讯引用:** 4994 | [OpenAlex ID](https://openalex.org/A5000463728)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建统一多模态网络入侵检测数据集并使用机器学习分类网络攻击，同时评估多种生成式模型产生的合成数据质量与隐私。

**💡 创新点**

首次将Synthetic Data Vault与非参数统计检验结合评估生成数据的多元分布一致性，并对多种GAN、VAEs、Diffusion及LLM模型在UMNIDS数据上的性能进行系统比较。

**🔧 技术方法**

机器学习分类器（XGBoost、随机森林等）、生成式对抗网络（CGAN、WGAN、f-GAN等）、Variational AutoEncoder、Diffusion Forest、DistilGPT2、PATE-CTGAN，使用SDV、f-散度、MMD、Hotelling T²、Frobenius Norm、NNDR等评价指标。

**📊 数据集**

统一多模态网络攻击数据集UMNIDS，整合CIC-IDS-2017、CIC-IoT-2023、UNSW-NB15、CIC-DDoS-2019的流级、报文负载与时间特征。

**📈 对比分析**

采用交叉验证、概率校准、TRTS/TSTR、区分性测试、f-散度、MMD等多维度指标；结果表明XGBoost、CTGAN-2、XGB-DF、DistilGPT2在精度、召回、F1和分布一致性方面优于其它模型。

**⚠️ 局限性**

对极少样本攻击缺乏覆盖，PATE-CTGAN在隐私保护下生成数据质量不足，实验仅限于单一数据集，未验证跨数据集鲁棒性与对抗攻击抵抗能力。

---

## 244. A Contextual Help Browser Extension to Assist Digital Illiterate Internet Users

**arXiv ID:** 2603.17592 | [PDF](https://arxiv.org/pdf/2603.17592v1)

**作者:** Christos Koutsiaris `[一作]` `[通讯]` (South East Technological University), Christos Koutsiaris (South East Technological University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一款名为Acro Helper的Chrome浏览器扩展，能够在用户悬停技术缩写时即时显示词义，帮助数字素养低的用户理解技术文章；

**💡 创新点**

创新点在于将AI驱动的页面分类（Google NLP + OpenAI ChatGPT）与混合词典+LLM定义交付相结合，形成完整的实时上下文帮助管道，并提供双层AI判别减少误报；

**🔧 技术方法**

使用技术包括Chrome扩展架构（Plasmo框架、Content Script、Background Service Worker）、Google Cloud Natural Language API（内容分类）、OpenAI ChatGPT（词典外词条补全与术语分类）、Mozilla Readability（主体提取）、正则表达式、React（词典搜索页）以及前端tooltip实现；

**📊 数据集**

数据集为本地技术词典（数千条技术缩写与定义），并在实验中使用25位来自社区、教育与企业的低至中等数字素养受试者的阅读材料（技术新闻文章），以及对“CPU”等缩写的手动搜索记录；

**📈 对比分析**

比较方法：在10次基准跑中测量词义获取时间，分别对词典路径（平均2,135 ms）、OpenAI GPT路径（平均16,429 ms）与手动Google搜索（平均17,200 ms）进行对比；结果显示词典路径约快7.7倍，且两种自动化方法均优于手动搜索，且92%受试者感知理解提升、96%感知时间节省；

**⚠️ 局限性**

局限性包括样本量仅25人、样本自选偏倚、实验在受控环境下进行可能夸大使用意愿、以及词典与AI路径在多义缩写识别上仍有误判等问题。

---

## 245. A scalable neural bundle map for multiphysics prediction in lithium-ion battery across varying configurations

**arXiv ID:** 2603.17209 | [PDF](https://arxiv.org/pdf/2603.17209v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 246. Auto-Unrolled Proximal Gradient Descent: An AutoML Approach to Interpretable Waveform Optimization

**arXiv ID:** 2603.17478 | [PDF](https://arxiv.org/pdf/2603.17478v1)

**作者:** Ahmet Kaplan `[一作]` `[通讯]` (Istanbul Medipol University), Ahmet Kaplan (Istanbul Medipol University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

利用自动机器学习 AutoGluon 与深度展开的近端梯度下降（PGD）相结合，构建了可解释的无线波束成形优化框架。

**💡 创新点**

创新点在于将 AutoML 自动搜索网络深度、步长、激活等超参数与深度展开的 PGD 结合，并加入可学习的线性梯度变换层，实现了少量训练样本下高效优化。

**🔧 技术方法**

使用了 AutoGluon、Optuna TPE、深度展开网络、混合层（可学习线性变换）以及梯度归一化等技术。

**📊 数据集**

数据集为从50,000个 Rayleigh 衰落样本中抽样的训练集，规模从 10^2 到 5×10^4，使用 ZF 初始化的波束成形矩阵作为标签。

**📈 对比分析**

与经典 200 次迭代 PGD、MLP、LISTA、PGD‑Net 和 ZF 进行对比，Auto‑PGD 在仅 5 层、100 样本时即可达到 98.8% 的谱效率，推理复杂度降低 40 倍。

**⚠️ 局限性**

局限性包括对完美 CSI 的假设、对硬件失真与估计误差的鲁棒性待验证，以及对更大规模 MIMO 或频率分集场景的推广仍需研究。

---

## 247. Event-Centric Human Value Understanding in News-Domain Texts: An Actor-Conditioned, Multi-Granularity Benchmark

**arXiv ID:** 2603.17838 | [PDF](https://arxiv.org/pdf/2603.17838v1)

**作者:** Yao Wang `[一作]` (University of Tsukuba), Haitao Yu `[通讯]` (University of Tsukuba)

**通讯引用:** 3416 | [OpenAlex ID](https://openalex.org/A5050721983)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了NEVU，一个面向新闻文本的事件中心人类价值识别基准数据集，支持角色归因、方向感知和多粒度事件结构。

**💡 创新点**

首次将价值标签与多层事件单元（子事件、行为复合事件、故事复合事件、文章）结合，并引入角色条件和价值方向，提供精细化、证据驱动的评估。

**🔧 技术方法**

采用LLM辅助的分阶段标注流程（候选生成、问答验证、人工审核）以及基于层级价值词典的多标签分类；同时评估多种LLM（API与开源）并通过LoRA微调提升性能。

**📊 数据集**

基于Uknow与英文维基新闻共2,865篇新闻，构建18,395子事件、6,575行为复合事件、5,582故事复合事件，包含45,793单元-角色对与168,061个定向价值实例。

**📈 对比分析**

统一提示模板下对四层单元进行有向多标签预测，使用Micro‑F1/Macro‑F1和方向逆转率评估；结果显示自适应微调后开源模型可达55% Micro‑F1，优于仅提示的Proprietary模型，方向错误率显著下降。

**⚠️ 局限性**

标注受LLM辅助与有限人工校验，存在残留噪声；价值标签长尾分布导致稀疏值难以评估；仅覆盖文本，未考虑多模态新闻。

---

## 248. Facial beauty prediction fusing transfer learning and broad learning system

**arXiv ID:** 2603.16930 | [PDF](https://arxiv.org/pdf/2603.16930v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 249. Facts as First Class Objects: Knowledge Objects for Persistent LLM Memory

**arXiv ID:** 2603.17781 | [PDF](https://arxiv.org/pdf/2603.17781v1)

**作者:** Oliver Zahn `[一作]` (Independent Researcher), Simran Chana `[通讯]` (University of Cambridge)

**通讯引用:** 88 | [OpenAlex ID](https://openalex.org/A5012655677)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型（LLM）的上下文记忆与外部知识对象（KO）两种记忆体系在事实检索、多跳推理等任务中的表现进行了系统基准，并量化了压缩失真、目标漂移等生产环境失效模式。

**💡 创新点**

提出基于哈希索引的离散知识对象（KO）架构，实现 O(1) 检索，解决容量限制、压缩失真和目标漂移问题；同时设计密度自适应检索机制，在嵌入检索与精确键匹配之间自适应切换，显著提升对“对抗性事实”的检索准确率。

**🔧 技术方法**

使用 Claude Sonnet 4.5、GPT‑4o、Gemini 1.5 等前沿 LLM 进行内存检索；实现 KO 的哈希存储与检索；构建密度自适应检索模型；通过摘要压缩实验探究上下文压缩对事实和约束的影响；开展多跳推理与跨域合成实验。

**📊 数据集**

主要使用合成药理事实语料（药物–靶点–IC50 三元组）以及人工构造的对抗性事实集合；跨模型实验中还使用 Gemini 1.5、GPT‑5.4 等；多跳推理使用 500 条药理事实；跨域合成使用 90 条跨领域知识对象（药理、材料科学、临床试验设计）。

**📈 对比分析**

在规模实验中，Claude Sonnet 在 ≤7,000 事实时保持 100% 精确匹配，GPT‑4o 在 3,000 事实即失效；KO 在所有规模下均为 100%。在对抗性事实检索中，嵌入检索 P@1 仅 20%，KO 为 100%；多跳推理中 KO 达 78.9% 对比全上下文 31.6%；跨域合成中 KO 将“有据可查”得分从 2.2 提升至 4.8。

**⚠️ 局限性**

局限性包括：使用合成药理数据，真实世界知识更模糊；对抗性查询的谓词规范化仍有 20% 失败；压缩实验仅基于 Claude 自摘要，未涵盖所有压缩策略；KO 需要 LLM 解析谓词，未解决非标准谓词匹配问题；实验规模相对有限，未覆盖更大规模或更复杂多跳链。

---

## 250. PAuth - Precise Task-Scoped Authorization For Agents

**arXiv ID:** 2603.17170 | [PDF](https://arxiv.org/pdf/2603.17170v1)

**作者:** Reshabh K Sharma `[一作]` (University of Washington), Shuo Chen `[通讯]` (Microsoft Research)

**通讯引用:** 11694 | [OpenAlex ID](https://openalex.org/A5115861810)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于任务细粒度的隐式授权框架PAuth，用NL切片和信封技术实现对AI代理任务执行的精确权限控制

**💡 创新点**

创新点在于将授权从操作符级别转为具体任务级别，结合自然语言切片和可验证的信封来保证每个操作符调用与用户意图一致，避免过度授权

**🔧 技术方法**

核心技术包括自然语言切片（NL slice）生成、信封（envelope）结构、LLM代码生成、规则编译与运行时检查、以及多主机MCP服务交互

**📊 数据集**

使用AgentDojo基准套件（Banking、Slack、Travel、Workspace）并扩展为Shopping套件，共计100个正常任务和634个强制注入攻击任务，共计734次实验

**📈 对比分析**

与传统OAuth对比，PAuth在所有测试中实现0误报、0漏报；对比不同LLM（GPT‑4.1、GPT‑5‑Mini、Gemini‑3‑Flash、Sonnet‑4.5）评估生成代码正确率和平均token成本，表现出高可靠性和可接受的成本范围

**⚠️ 局限性**

局限性包括对LLM生成代码的依赖导致概率性错误、评估在封闭式任务描述下，缺乏对开放式对话中自然语言歧义的处理、以及在真实Web环境中多服务协同推断的性能与部署成本尚未充分验证

---

## 251. ProGVC: Progressive-based Generative Video Compression via Auto-Regressive Context Modeling

**arXiv ID:** 2603.17546 | [PDF](https://arxiv.org/pdf/2603.17546v1)

**作者:** Daowen Li `[一作]` (Alibaba Group), Li Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 295766 | [OpenAlex ID](https://openalex.org/A5100338825)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了ProGVC，一种基于视觉自回归模型的进阶视频压缩框架，融合多尺度残差量化、Transformer自回归上下文建模与细节生成，实现可进度传输和自适应比特率压缩。

**💡 创新点**

创新点包括：①采用多尺度残差量化生成可进度的token图；②设计任务特定的多尺度自回归上下文模型，既完成熵编码又在解码端生成丢失尺度；③在同一框架下实现可扩展性、可自适应比特率与低延迟。

**🔧 技术方法**

使用了视觉自回归模型（VAR）、多尺度残差量化、Transformer-based自回归上下文模型、稀疏注意力、二进制球面量化以及算术编码等技术。

**📊 数据集**

训练使用Pexels视频集（约480K高质量720p视频），评估数据集包括Xiph、HEVC Class B和MCL-JCV（均下采样至720p）。

**📈 对比分析**

与VVC、DCVC、SEVC、PLVC等基线在DISTS、LPIPS、NIQE、PSNR等指标上进行比较，ProGVC在感知质量指标上均优于所有基线，BD-rate显著下降，解码速度低于Diffusion-based codecs，编码速度最快。

**⚠️ 局限性**

限制在于仅支持720p短时长视频，离散的自适应比特率难以实现连续细粒度控制，对更高分辨率和更长时长的鲁棒性不足。

---

## 252. Directing the Narrative: A Finetuning Method for Controlling Coherence and Style in Story Generation

**arXiv ID:** 2603.17295 | [PDF](https://arxiv.org/pdf/2603.17295v1)

**作者:** Jianzhang Zhang `[一作]` (Hangzhou Normal University), Chuang Liu `[通讯]` (Hangzhou Normal University)

**通讯引用:** 6534 | [OpenAlex ID](https://openalex.org/A5085994998)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个两阶段的框架，先通过Group-Shared Attention实现跨帧身份一致性，再用Direct Preference Optimization提升视觉质量，解决故事可视化中的身份漂移和风格不一致问题。

**💡 创新点**

创新点在于引入Group-Shared Attention在同一批次内部实现无压缩的跨样本信息流，和将DPO作为后期偏好对齐阶段，避免传统损失冲突，显著提升身份与风格一致性。

**🔧 技术方法**

采用FLUX.1流匹配变压器骨干，配合LoRA适配器实现GSA和DPO，结合多尺度注意力和流匹配损失进行训练。

**📊 数据集**

使用自己构建的故事书与视频数据集，并在ViStoryBench基准上进行评估。

**📈 对比分析**

在ViStoryBench上与StoryGen、StoryDiffusion、StoryAdapter、UNO等基线对比，CIDS提升约+10，CSD提升约+18，整体性能显著领先。

**⚠️ 局限性**

局限在于GSA的计算量随参考图数量线性增长，难以满足长序列或实时生成需求，且对抽象或非人形角色的保持仍有挑战。

---

## 253. A practical artificial intelligence framework for legal age estimation using clavicle computed tomography scans

**arXiv ID:** 2603.17926 | [PDF](https://arxiv.org/pdf/2603.17926v1)

**作者:** Javier Venema `[一作]` (Department of Computer Science and Artificial Intelligence, University of Granada), Óscar Ibáñez `[通讯]` (Faculty of Computer Science, University of A Coruña)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文开发了一套可解释的多阶段管线，用于从锁骨CT扫描中进行合法年龄估计，涵盖自动锁骨检测、基于集成梯度的切片选择、多切片卷积神经网络年龄回归和基于校准预测区间的置信度评估；

**💡 创新点**

创新点在于①使用基于形状与几何特征的连通分量方法实现低标注成本的锁骨检测；②将集成梯度（IG）嵌入训练阶段进行切片重要性引导，而非仅作为事后可解释工具；③构建多视角、多切片的ResNet50分支网络并加入合成数据增强，显著提升泛化；④采用合成预测（conformal prediction）提供可校准的置信区间，满足法医“最低年龄”决策需求；⑤将该管线集成到Skeleton-ID软件，具备实际法医工作流应用价值；

**🔧 技术方法**

技术上结合连通分量分析+随机森林做检测，使用形状分布、几何特征；集成梯度用于切片选择；采用多分支ResNet50+全连接融合网络进行年龄回归；使用Adam+1cycle训练策略；实现数据增强、合成预测区间；

**📊 数据集**

使用来自New Mexico Decedent Image Database的1158份尸体全身CT扫描（年龄14-26岁），按73%男性分布，划分为742/184/232的训练/验证/测试集；

**📈 对比分析**

与先前AI方法（3D CNN、堆叠2D CNN）及专家评估在统一协议下对比，测试MAE为1.55±0.16年，明显优于专家（≈1.90年）和现有AI（1.65-1.77年），验证MAE为1.34年；合成预测区间在90%覆盖率下实现高PPV与敏感度平衡；统计检验显示对照3D基线显著提升（p≈0.008）；

**⚠️ 局限性**

局限性包括仅在尸体CT上验证，缺乏临床CT验证；样本年龄范围有限（14-26岁），男性占比高；模型对不同扫描协议、扫描仪差异的鲁棒性尚未充分评估；数据量仍有限，可能导致对少数年龄段过拟合。

---

## 254. Enabling Real-Time Programmability for RAN Functions: A Wasm-Based Approach for Robust and High-Performance dApps

**arXiv ID:** 2603.17880 | [PDF](https://arxiv.org/pdf/2603.17880v1)

**作者:** João Paulo Esper `[一作]` (Universidade Federal de Goiás), Kleber Cardoso `[通讯]` (Universidade Federal de Goiás)

**通讯引用:** 931 | [OpenAlex ID](https://openalex.org/A5071689195)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文提出在O‑RAN架构中使用WebAssembly对dApp进行隔离，实现实时控制循环的轻量级、确定性沙箱；

**💡 创新点**

创新点在于将Wasm作为沙箱替代传统容器，提供细粒度指令计费、强隔离和可预期性能，并通过原型验证其优于裸机与容器的实时特性；

**🔧 技术方法**

采用WebAssembly、Wasmtime运行时、WASI SDK、OAI 5G软件栈、E3接口、ASN.1 编解码以及Docker容器等技术；

**📊 数据集**

实验基于OAI 5G软件栈与自研Spectrum Sharing dApp，硬件平台为AMD Ryzen 7 5800X，未使用公开数据集；

**📈 对比分析**

通过对比裸机、容器和Wasm三种运行时的控制环路延迟、CPU占用和内存占用，发现Wasm的控制延迟约为149 µs（相较裸机113 µs增加41%），CPU占用仅为裸机的28%（低于容器的65%），内存占用略高但仍在可接受范围；

**⚠️ 局限性**

主要局限包括Wasm对硬件加速（FPGA/GPU/SmartNIC）的支持不足、运行时计费开销导致的延迟增加、与Near‑RT RIC/SMO等O‑RAN生态整合验证不足以及对长期可靠性与可扩展性的进一步研究需求。

---

## 255. PJB: A Reasoning-Aware Benchmark for Person-Job Retrieval

**arXiv ID:** 2603.17386 | [PDF](https://arxiv.org/pdf/2603.17386v1)

**作者:** Guangzhi Wang `[一作]` (Career International Research Team), Zhi Liu `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个面向招聘场景的推理感知检索基准PJB，并在此基准上开展了稠密检索实验。

**💡 创新点**

其创新点在于以岗位能力为核心的相关性定义、基于业务域和推理类型的诊断标签，以及将检索性能从单一平均分升级为可解释的诊断视角。

**🔧 技术方法**

采用了稠密检索模型（CRE‑T1、Qwen3‑Embedding）以及基于LLM的查询理解（QU）和重排序（Rerank）模块，并使用LLM-as-a-Judge进行标签生成。

**📊 数据集**

数据集为PJB v1.0，包含约297条完整职位描述、197,674份脱敏简历及2,242条正相关标注，全部来源于2025年后公开的招聘日志。

**📈 对比分析**

通过2×4模块消融实验，以nDCG@10为主指标，发现CRE‑T1远优于Qwen3，重排序在CRE‑T1上提升≈9%，但对Qwen3导致下降≈26%，域和推理类型差异显著。

**⚠️ 局限性**

限制在于仅评估稠密检索、仅使用两款模型、正标注稀疏、标签为启发式、细粒度域/推理样本有限，无法直接推广到BM25、混合检索或更完整的系统组合。

---

## 256. ACE-LoRA: Graph-Attentive Context Enhancement for Parameter-Efficient Adaptation of Medical Vision-Language Models

**arXiv ID:** 2603.17079 | [PDF](https://arxiv.org/pdf/2603.17079v1)

**作者:** M. Arda Aydın `[一作]` (Bilkent University), Tolga Çukur `[通讯]` (Bilkent University)

**通讯引用:** 3856 | [OpenAlex ID](https://openalex.org/A5032375235)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e0540dec-d77f-42db-94ae-d039248f6393` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出ACE-LoRA，一种参数高效的微调框架，通过在冻结的通用医学VLM上插入LoRA模块并引入基于注意力的高阶超图神经网络，提升零样本分类、分割与检测性能。

**💡 创新点**

创新点在于：①将LoRA与超图神经网络耦合，捕获局部与全局的高阶语义交互；②使用标签引导的InfoNCE损失抑制误负样本；③在保持仅0.95M可训练参数的前提下，显著提升细粒度诊断能力。

**🔧 技术方法**

使用技术包括：LoRA参数微调、Attention-based Context Enhancement Hypergraph Neural Network (ACE‑HGNN)、标签引导的InfoNCE对比损失、Transformer自注意力与超图消息传递结合的多任务训练。

**📊 数据集**

训练数据集为MIMIC‑CXR（胸片）和从PMC‑OA提取的病理图像‑报告子集；评估数据集包括CheXpert 5×200、RSNA Pneumonia、SIIM‑ACR Pneumothorax、LC25000（肺/结肠）和MHIST。

**📈 对比分析**

与BiomedCLIP、BMC‑CLIP等通用医学VLM、CoOp、CoCoOp、CLIP‑Adapter、TaskRes、MaPLe、MMA、CLIP‑LoRA以及全量微调进行对比；在零样本分类上ACE‑LoRA平均提升≈4–5%准确率，在分割与检测任务中保持或超越竞争者，表明参数高效且性能领先。

**⚠️ 局限性**

局限性：依赖手工或自动提取的疾病标签来排除误负样本，导致在无标签或多义报告场景下效果下降；对不同VLM骨干的通用性尚待进一步验证；目前仅针对静态医学图像，未扩展到视频或其他多模态领域。

---

## 257. TDMM-LM: Bridging Facial Understanding and Animation via Language Models

**arXiv ID:** 2603.16936 | [PDF](https://arxiv.org/pdf/2603.16936v1)

**作者:** Luchuan Song `[一作]` (University of Rochester), Chenliang Xu `[通讯]` (University of Rochester)

**通讯引用:** 6525 | [OpenAlex ID](https://openalex.org/A5064805926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个80小时、57K条视频的Open3DFaceVid合成面部动作数据集，并提出了基于3DMM几何token的双向语言‑动作框架。

**💡 创新点**

创新点在于用多模型T2V生成高质量面部视频来弥补文本‑动作对稀缺，利用几何VQ‑VAE将连续3DMM编码为离散token，并将LLM直接对齐到几何token实现Motion2Language和Language2Motion的统一解码。

**🔧 技术方法**

采用了文本到视频（T2V）生成、3D Morphable Model (FLAME)、Geometry VQ‑VAE、预训练大型语言模型（Qwen3/LLaMA等）以及基于token的自回归解码器。

**📊 数据集**

使用了自制的Open3DFaceVid数据集，包含多种情绪、身份、说话风格的合成视频；此外与MEAD和YouTube采集的真实视频进行对照。

**📈 对比分析**

通过与HumanOmni、Gemini VLM及T2M‑X/T2M‑GPT等基线对比，Motion2Language方面在情绪、动作、强度的准确率均超过基线，Language2Motion在L2、FD、token精度及用户评估上也表现更优，且token数量显著减少。

**⚠️ 局限性**

局限性包括依赖合成视频的真实性不足、对微表情的细粒度捕捉仍受T2V模型约束，以及缺乏声音或更复杂动态上下文的建模。

---

## 258. Robust-ComBat: Mitigating Outlier Effects in Diffusion MRI Data Harmonization

**arXiv ID:** 2603.17968 | [PDF](https://arxiv.org/pdf/2603.17968v1)

**作者:** Yoan David `[一作]` (University of Sherbrooke), The TRACK-TBI Investigators `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `da1b1a89-583a-4b57-9c81-478778569bec` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

本文提出了 Robust‑ComBat 框架，结合 ComBat 并在数据归一化前进行病理异常样本的过滤；

**💡 创新点**

创新点在于将轻量级 MLP 作为全局异常检测器，显著降低了病理样本对 ComBat 估计的偏差，并在多种病理状态下保持稳健；

**🔧 技术方法**

主要技术包括 ComBat（及其变体 CovBat、ComBat‑GAM、Clinical‑ComBat）与多种异常检测方法（Z‑score、IQR、MAD、Rousseeuw‑Croux、MMS、VS、全局 Z/MAD、MLP）；

**📊 数据集**

实验使用 ADNI、TRACK‑TBI、LA5c、SchizConnect、CamCAN 等多中心数据，共计 1,869 名受试者，涵盖 AD、MCI、TBI、SCHZ、BIP、ADHD 等六种疾病；

**📈 对比分析**

通过标准化均方误差 (STD_MAE) 与 Bhattacharyya 距离评估，MLP 在病理比例 ≥30% 时均优于其他过滤策略，且在四种 ComBat 变体中表现最稳健；

**⚠️ 局限性**

局限性包括：需要至少 30 名受试者才能稳定过滤；模型需与训练时相同的特征空间（bundle/metric 配置）；在极低病理比例（<30%）时过滤效果有限，需谨慎使用。

---

## 259. ResNet-50 with Class Reweighting and Anatomy-Guided Temporal Decoding for Gastrointestinal Video Analysis

**arXiv ID:** 2603.17784 | [PDF](https://arxiv.org/pdf/2603.17784v1)

**作者:** Romil Imtiaz `[一作]` (University of Thessaly), Dimitris K. Iakovidis `[通讯]` (University of Thessaly)

**通讯引用:** 5194 | [OpenAlex ID](https://openalex.org/A5020655974)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于ResNet-50的多标签消化道视频分析管线，结合类别权重重分配和解剖引导的时间事件解码；

**💡 创新点**

创新点在于采用剪裁后的类别正样本权重缓解严重类别不平衡，并通过解剖投票平滑与解剖门控机制提升时间事件一致性；

**🔧 技术方法**

使用的技术包括ResNet-50帧级分类、加权二元交叉熵（含权重剪裁）、焦点损失实验、滑动窗口投票、解剖门控、GT式事件重构与保守滞后式时间解码；

**📊 数据集**

使用了ICPR 2026 RARE‑VISION竞赛提供的消化道视频数据集，包含多标签（5个解剖类别+12个病理类别）以及3个测试视频；

**📈 对比分析**

与基线时间mAP 0.3801对比，最终提升至0.4303（mAP@0.5）与0.4020（mAP@0.95），帧级mAP达0.8605，表明相对于传统方法的显著性能提升；

**⚠️ 局限性**

局限在于帧级强而时间级仍有提升空间，保守解码虽稳健但对细粒度事件捕捉有限，且未实现端到端时间学习，类别极少数仍易出现漏检。

---

## 260. Anisotropic Permeability Tensor Prediction from Porous Media Microstructure via Physics-Informed Progressive Transfer Learning with Hybrid CNN-Transformer

**arXiv ID:** 2603.17532 | [PDF](https://arxiv.org/pdf/2603.17532v1)

**作者:** Mohammad Nooraiepour `[一作]` (University of Oslo), Mohammad Nooraiepour `[通讯]` (University of Oslo)

**通讯引用:** 554 | [OpenAlex ID](https://openalex.org/A5033690329)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于MaxViT混合CNN-Transformer的物理信息深度学习框架，用于高精度预测孔隙尺度微结构图像的二维渗透率张量。

**💡 创新点**

创新点包括：1）引入多轴自注意机制同时捕获颗粒尺度孔隙通道几何和代表性体积尺度连通统计；2）设计可微分的物理约束损失，严格强制张量对称性与正定性；3）实现D4-等变数据增强，确保图像与张量同步变换；4）采用分阶段进阶迁移学习，结合孔隙率条件FiLM以及权重平均技术。

**🔧 技术方法**

使用的技术主要有：MaxViT网络、进阶迁移学习、可微分物理损失、D4等变数据增强、FiLM特征调制、SWA/EMA模型集成、MC-Dropout不确定性估计。

**📊 数据集**

数据集为约20,000张128×128二值孔隙结构图像（合成砂岩），通过Lattice-Boltzmann求解得到对应的三阶量程渗透率张量，并划分训练/验证/测试集。

**📈 对比分析**

与多种基准架构（ResNet、ViT、ConvNeXt等）对比，本文模型在测试集上取得方差加权R²=0.9960、RMSE≈1.56e-2、无偏差对称误差≈3.95e-7，并且推理时间仅约120 ms，速度比传统DNS提升10³–10⁴倍。

**⚠️ 局限性**

主要局限包括：对真实CT图像的泛化能力未验证；近乎各向同性样本的离散耦合预测仍有残差；模型在3D扩展时面临显存与计算量爆炸；以及仅使用MC-Dropout的单一不确定性估计方法。

---

## 261. Talk is Cheap, Logic is Hard: Benchmarking LLMs on Post-Condition Formalization

**arXiv ID:** 2603.17193 | [PDF](https://arxiv.org/pdf/2603.17193v1)

**作者:** I. S. W. B. Prasetya `[一作]` (Utrecht University), Davide Prandi `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 13055 | [OpenAlex ID](https://openalex.org/A5008401803)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型（LLM）在从自然语言描述生成完整前置条件和后置条件（合同）方面的能力。

**💡 创新点**

创新点在于提出了针对完整合同生成的细粒度评价指标、构建了新的 HEx 数据集，并系统评估了 24 个主流 LLM（包括开源与专有模型）的性能。

**🔧 技术方法**

主要技术包括基于提示的生成、Python 代码解析与执行、手工与自动化（Pynguin+Poodle）测试用例生成，以及多指标评估（accept@k、平均通过率、错误率、误报/漏报率）。

**📊 数据集**

使用了自研的 HEx 数据集（40 个任务），每个任务包含自然语言说明、Python 实现、完整前置/后置条件、手工测试与自动生成测试。

**📈 对比分析**

通过多模型、多轮提示、10 次重复实验，并对比 open 与 proprietary LLM 的 accept@1、错误率等指标，发现专有模型表现更好，特别是 Claude3.7、GPT4o 等；open 模型在错误率和误报率上显著较高。

**⚠️ 局限性**

局限性包括：仅评估 Python 代码；数据集规模有限；测试用例虽自动化但仍可能不足；模型回答的随机性可能需要更多重复；以及对语言细微差异的敏感性仍未完全解决。

---

## 262. Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning

**arXiv ID:** 2603.17174 | [PDF](https://arxiv.org/pdf/2603.17174v1)

**作者:** Shenao Yan `[一作]` (University of Connecticut), Yuan Hong `[通讯]` (University of Connecticut)

**通讯引用:** 2243 | [OpenAlex ID](https://openalex.org/A5100725148)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为CodeScan的黑盒、面向漏洞的代码生成LLM中毒扫描框架，能够在无内部信息的前提下检测并逆向回收攻击目标；

**💡 创新点**

创新点在于：①针对代码生成的结构化特性，采用AST归一化与结构差异分析，克服传统基于token一致性的局限；②结合LLM驱动的漏洞分析，能够识别经过变换或混淆的攻击代码；③实现了高效的词汇扫描与聚类分支控制，显著降低扫描时间；

**🔧 技术方法**

核心技术包括：自动回归LLM查询、AST抽象语法树构建与归一化、结构一致性聚类与熵/优势判定、LLM（GPT‑5）漏洞检测器、阈值与分支剪枝机制；

**📊 数据集**

实验使用108个模型（CodeLlama、Qwen、StarCoder，7B至34B规模），基于15个CWE漏洞构造的攻击与清洗数据集；每种漏洞使用20个干净提示；

**📈 对比分析**

与现有基线（基于token一致性的扫描方法）对比，CodeScan在三种漏洞场景下均达成≈0.98的F1分数，召回率100%，误报率低至0%，扫描时间比基线低十倍以上；AST距离和BLEU分数也显著优于基线；

**⚠️ 局限性**

局限性包括：依赖LLM漏洞检测器的准确性，若检测器误判会导致误报或漏报；仅针对已知的漏洞类别；对非换行内嵌攻击目标的识别仍有挑战；对极高抗攻击（adaptive）场景的鲁棒性尚待进一步验证。

---

## 263. An End-to-End Framework for Functionality-Embedded Provenance Graph Construction and Threat Interpretation

**arXiv ID:** 2603.17100 | [PDF](https://arxiv.org/pdf/2603.17100v1)

**作者:** Kushankur Ghosh `[一作]` (University of Alberta), Jörg Sander `[通讯]` (University of Alberta)

**通讯引用:** 48367 | [OpenAlex ID](https://openalex.org/A5021763337)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

Auto-Prov是一套端到端框架，能够自动识别日志类型、利用大型语言模型生成提取规则、构造功能嵌入的来源图，并在此基础上进行异常检测与攻击摘要生成。

**💡 创新点**

其创新点包括：1）在无需人工规则的前提下，通过LLM自动发现并聚类异构日志；2）为图节点注入系统功能标签，提升检测对功能差异的辨识度；3）用LLM将检测到的攻击图转化为可读的自然语言摘要并映射至MITRE ATT&CK tactics。

**🔧 技术方法**

技术上主要使用RoBERTa做日志嵌入与DBStream在线聚类，GPT-4o和LLaMA-3进行候选图提取与规则生成，MPNet/DistilBERT做节点嵌入，行为基分类器推断未知实体功能，以及LLM辅助的攻击摘要与tactics推理。

**📊 数据集**

实验使用DARPA-E3 Transparent Computing THEIA数据集和ATLAS公共数据集，涵盖Ubuntu、Windows、Firefox、DNS等多平台、多日志格式及多种真实攻击。

**📈 对比分析**

与Flash、MAGIC、OCR-APT、Kairos四个最先进的来源图异常检测器比较，Auto-Prov在THEIA和ATLAS上平均提升AUC-ROC约0.23和0.15，ADP常达到1.0，功能标签的加入显著提升检测精度并在不同日志类型下保持稳定。

**⚠️ 局限性**

局限性包括：1）对完全新功能的实体无法准确推断；2）未考虑日志或提示的恶意注入攻击；3）在资源受限或日志极度不规则时，LLM生成的规则可能不完整，导致功能覆盖不足。

---

## 264. Learning generalized Nash equilibria from pairwise preferences

**arXiv ID:** 2603.17015 | [PDF](https://arxiv.org/pdf/2603.17015v1)

**作者:** Pablo Krupa `[一作]` (IMT School for Advanced Studies), Alberto Bemporad `[通讯]` (IMT School for Advanced Studies)

**通讯引用:** 32239 | [OpenAlex ID](https://openalex.org/A5053340099)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种主动学习框架，仅通过对每个代理对两种决策的偏好进行二元查询，学习并逼近 Generalized Nash Equilibrium（GNE）。

**💡 创新点**

创新点在于：①不依赖目标函数值或最佳响应的直接访问，而仅利用偏好数据；②用逻辑回归将偏好映射为代理的代理目标函数；③在主动学习过程中平衡探索与利用，并加入不相似度函数提升分类精度；④将学习得到的代理目标函数作为 surrogate GNEP 进行求解。

**🔧 技术方法**

技术实现包括：逻辑回归（交叉熵损失）与正则化、主动学习中的探索函数（IDW、空间填充等）、指数衰减的探索-利用权重、梯度优化（Adam、L-BFGS-B）、GNEP 求解器（Python cvxpy/pyomo）以及对线性二次调节（LQR）和文献中 GNEP 的仿真测试。

**📊 数据集**

数据集：全部为合成数据，分别包含：①游戏论 LQR 示例（随机生成的系统矩阵，三到四个代理）；②文献中的三类 GNEP（非线性、二次、10 维二次）作为基准；无公开真实数据集。

**📈 对比分析**

比较方法：将学习得到的 GNE 与已知的真实 GNE（或已求解的 GNEP 结果）在以下指标上进行对比：最佳响应偏差、闭环成本的归一化 RMSE、闭环轨迹的误差。实验结果显示，随着主动学习迭代次数增加，RMSE 下降至 10⁻⁴~10⁻³ 级别，最佳响应偏差也随之减小，证明方法能够逼近真实 GNE。

**⚠️ 局限性**

局限性：①需要先验约束信息与至少存在一个 GNE 的假设；②探索-利用权重、噪声参数等超参数需经验调节，易受规模影响；③对高维或大规模代理系统的扩展尚未验证，计算成本随 GNEP 求解器的规模增长；④偏好查询假设代理能够提供准确比较，实际应用中可能存在噪声或不一致。

---

## 265. Formal verification of tree-based machine learning models for lateral spreading

**arXiv ID:** 2603.16983 | [PDF](https://arxiv.org/pdf/2603.16983v1)

**作者:** Krishna Kumar `[一作]` `[通讯]` (Oden Institute of Computational Engineering and Sciences), Krishna Kumar (Oden Institute of Computational Engineering and Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

利用SMT求解器对树模型（XGBoost 与 EBM）进行全域物理一致性验证，并基于验证结果进行约束修正，形成验证-修复-验证循环。

**💡 创新点**

首次将 SMT 逻辑公式与地质灾害预测树模型结合，提供对单特征阈值、单调性以及多特征组合条件的正式、可证的验证框架，并展示约束修正对物理一致性的提升。

**🔧 技术方法**

使用 Z3 SMT 求解器将树模型编码为 QF_LRA 公式；采用单树分解实现高效单调性检验；生成正式的最小原因解释（formal abductive explanations）。

**📊 数据集**

基于 2011 年基督城地震横向位移数据集（7,291 个站点，4 个物理特征）进行训练、验证与实验。

**📈 对比分析**

对 33 种模型变体做准确率与规范合规性 Pareto 分析；每个规格检验平均耗时 <5 s；与传统测试/SHAP 解释相比，SMT 能快速发现全域违规并给出具体 counterexample；验证显示约 10–15% 的准确率‑一致性 trade‑off。

**⚠️ 局限性**

局限性包括：仅适用于树模型，单特征方向约束无法直接实现多特征阈值；验证仅覆盖必要条件，可能仍存在未列出的违规；规范阈值需手工校准，且在不同地区需重新设置。

---

## 266. Informative Semi-Factuals for XAI: The Elaborated Explanations that People Prefer

**arXiv ID:** 2603.17534 | [PDF](https://arxiv.org/pdf/2603.17534v1)

**作者:** Saugat Aryal `[一作]` (University College Dublin), Mark T. Keane `[通讯]` (University College Dublin)

**通讯引用:** 4884 | [OpenAlex ID](https://openalex.org/A5074878641)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出一种新的半事实解释方法——Informative Semi-Factuals (ISF)，通过在半事实解释中加入隐藏特征信息，使解释更具洞察力。

**💡 创新点**

创新点在于：①引入关键特征衰减与隐藏特征强化的两条新约束，捕捉特征贡献变化的“平衡”模式；②将这些约束纳入多目标优化框架，生成既满足传统半事实要求又具有信息性的解释。

**🔧 技术方法**

技术手段包括：多目标非支配排序遗传算法（NSGA-II）求解优化问题；高斯Copula估计特征联合分布；TreeSHAP计算特征的纯主效应；曼肯-肯德尔趋势检验和Kendall's tau衡量特征贡献的变化趋势。

**📊 数据集**

使用五个公开表格数据集进行评估：Adult Income、Blood Alcohol、PIMA Diabetes、German Credit、HELOC。

**📈 对比分析**

与现有八种主流半事实方法（CBR、KLEOR、DSER、PIECE、C2C-VAE、MDN、S-GEN、DiCE）构成的集成基准相比，ISF 在“平衡模式”出现率上接近 100%，并在距离、稀疏度、可信度和合理性等指标上均优于对手，显示出更高的解释质量与可接受度。

**⚠️ 局限性**

主要局限在于计算复杂度高，尤其在特征维度很大时求解效率下降；未来可通过特征选择或局部稀疏策略改进。

---

## 267. AppFlow: Memory Scheduling for Cold Launch of Large Apps on Mobile and Vehicle Systems

**arXiv ID:** 2603.17259 | [PDF](https://arxiv.org/pdf/2603.17259v1)

**作者:** Xiaochen Li `[一作]` (Northwestern Polytechnical University), Zhiwen Yu `[通讯]` (Harbin Engineering University)

**通讯引用:** 15121 | [OpenAlex ID](https://openalex.org/A5100701166)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一套名为AppFlow的系统级调度框架，用于在移动和车载设备上加速GB规模应用的冷启动，同时维持多任务环境下的背景活跃度。

**💡 创新点**

首次将文件预加载、内存回收与进程终止三项机制联合优化，通过预测文件访问模式、区分文件页与匿名页回收以及上下文感知的杀进程策略，解决了传统方法相互冲突导致的性能瓶颈。

**🔧 技术方法**

使用了选择性文件预加载器（Selective File Preloader）、自适应内存回收器（Adaptive Memory Reclaimer）和上下文感知进程杀手（Context-Aware Process Killer），并在Android框架与Linux内核层实现，配合多种I/O调度与内存回收技术。

**📊 数据集**

利用真实的100天日常使用轨迹（覆盖60+热门应用）以及在Pixel 7/Pixel 8和Raspberry Pi 4车载系统上构建的仿真工作负载，对比评估。

**📈 对比分析**

与原生Android、Paralfetch、Acclaim及其组合基线对比，AppFlow在所有设备和负载下平均缩短冷启动延迟33.7–43.6%，最高可达66.5%，95%的启动时间保持在1 秒以内，并显著提升I/O吞吐量（2.35×）与内存回收效率（67.9%内存压力下降）。

**⚠️ 局限性**

在小型低I/O应用上提升有限，且实现需要对Android框架与Linux内核做一定修改；预加载占用的额外内存虽然被控制在≤100 MB，但在极端内存受限环境下仍可能产生波动。

---

## 268. Multi-Source Human-in-the-Loop Digital Twin Testbed for Connected and Autonomous Vehicles in Mixed Traffic Flow

**arXiv ID:** 2603.17751 | [PDF](https://arxiv.org/pdf/2603.17751v1)

**作者:** Jianghong Dong `[一作]` (Tsinghua University), Keqiang Li `[通讯]` (Tsinghua University)

**通讯引用:** 16158 | [OpenAlex ID](https://openalex.org/A5031855986)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

开发了一个多源人类在环混合云控制实验平台（MSH‑MCCT），整合物理车辆、虚拟车辆、混合空间和多源控制输入，实现物理、虚拟车辆与人类驾驶员及 CAV 算法的实时交互与混合驾驶实验。

**💡 创新点**

①首次将多名真实人类驾驶员通过不同仿真器同步投入到数字孪生环境中；②采用混合数字孪生（mixedDT）概念，将物理、虚拟与混合空间融合，实现车辆热交换与视角切换；③构建云端统一通信与数据融合框架，支持物理与虚拟车辆的跨平台实时控制。

**🔧 技术方法**

数字孪生（mixedDT）、混合现实、Unity/SCANer Studio、ROS、ZeroMQ、云计算、Logitech G29 与 InnoSimulation 驾驶模拟器、C++/Python 等编程语言。

**📊 数据集**

未使用公开数据集，实验采用9辆物理迷你车、若干虚拟车，在环人类驾驶员的实时操作与控制指令。

**📈 对比分析**

与 SUMO、CARLA、NVIDIA Drive Sim、VeHIL、Mcity、MCCT 等现有平台在多人驾驶、物理-虚拟融合、可扩展性、经济性、可复现性等维度对比，MSH‑MCCT 在多源驾驶员、物理虚拟融合、可扩展性和可复现性上表现最高；实验验证了在交通波动与安全关键场景下 CACC 控制在混合流中抑制波动并能安全处理碰撞。

**⚠️ 局限性**

实验规模有限（仅9辆车、单一路段），只验证了车队行驶场景；对大规模道路网络、交叉口、车路协同等复杂情况尚未评估；需要进一步提升多模态感知与低时延通信。

---

## 269. Learning When to Attend: Conditional Memory Access for Long-Context LLMs

**arXiv ID:** 2603.17484 | [PDF](https://arxiv.org/pdf/2603.17484v1)

**作者:** Sakshi Choudhary `[一作]` (Purdue University), Stefano Soatto `[通讯]` (Amazon Web Services)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可学习的 token 级路由机制 L2A，在 LLM 的每个 token 处决定是否调用全局注意力，从而实现长上下文建模。

**💡 创新点**

创新点在于：① 通过路由器动态判断何时需要全局注意力，避免所有 token 都做 O(n²) 计算；② 为此设计了基于 Triton 的稀疏全局注意力实现，实现显著的吞吐量提升；③ 训练时加入稀疏正则化和 STE，防止路由器崩溃。

**🔧 技术方法**

使用技术包括滑动窗口 Local Attention、全局 Attention、线性路由器 + Sigmoid、Straight‑Through Estimator、稀疏正则化、定制 Triton kernel、以及后训练层级剪枝。

**📊 数据集**

数据集：预训练使用 Qwen 2.5、Qwen 3 等大规模文本；评估使用 HELMET、BabiLong、MRCR、Synthetic Recall、Retrieval‑Augmented Generation、In‑Context Learning 等多任务集。

**📈 对比分析**

与 CLP、SelfExtend、DCA、S²‑Attn、NSA、FlexPrefill 等基线对比，L2A 在 128K 上的性能与 CLP 仅差 1.5‑3%，同时将全局注意力使用率降低 75‑80%，训练吞吐量提升 ~2×，TTFT 也约减半。

**⚠️ 局限性**

局限性：仍需保存完整 KV 缓存（虽可剪枝但对某些任务影响不确定）；依赖自定义 Triton kernel，迁移到其他硬件/框架受限；仅在 Qwen 系列上验证，跨模型泛化需进一步实验；训练时需细致平衡正则项以防路由器崩溃。

---

## 270. AI Scientist via Synthetic Task Scaling

**arXiv ID:** 2603.17216 | [PDF](https://arxiv.org/pdf/2603.17216v1)

**作者:** Ziyang Cai `[一作]` (Princeton University), Harkirat Behl `[通讯]` (Microsoft Research)

**通讯引用:** 673 | [OpenAlex ID](https://openalex.org/A5081255348)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种自动合成机器学习任务的流水线，用于训练AI科研代理。

**💡 创新点**

通过无监督的自调试循环验证环境，自动生成与真实HuggingFace数据集对齐的可执行任务，并从中采样完整的agent轨迹。

**🔧 技术方法**

利用多阶段生成管道、HuggingFace检索、SWE‑Agent框架、GPT‑5教师模型和自调试反馈、SFT微调。

**📊 数据集**

采样自1,000个独立ML主题，检索并验证真实HuggingFace数据集，生成约500个任务和30k–34k条轨迹。

**📈 对比分析**

在MLGym基准上微调Qwen3‑4B/8B后，平均AUP提升9%/12%，在大多数子任务上优于基线模型。

**⚠️ 局限性**

仅评估单一基准、缺乏组件消融、教师模型偏差、SFT不包含探索或新颖性奖励、未证明跨任务泛化。

---

## 271. Energy Flow Graph: Modeling Software Energy Consumption

**arXiv ID:** 2603.17162 | [PDF](https://arxiv.org/pdf/2603.17162v1)

**作者:** Saurabhsingh Rajput `[一作]` (Dalhousie University), Tushar Sharma `[通讯]` (Dalhousie University)

**通讯引用:** 2173 | [OpenAlex ID](https://openalex.org/A5023044082)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 Energy Flow Graph (EFG) 通过状态与转移的能耗注释，构建程序能耗的路径依赖图模型，并在算法核与 AI 管道两大场景中进行能耗分析与预测。

**💡 创新点**

创新点在于：①将能耗视为物理的状态-转移成本，桥接图论与热力学；②提供确定性与随机性两种能耗分析（最优/最差路径与期望能耗）；③提出乘法级联模型预测多优化组合效果，避免 2^n 的全量测试；④支持层级抽象实现跨尺度能耗推理。

**🔧 技术方法**

技术手段包括：图论与状态机建模、马尔可夫链/MDP 与 Bellman 方程求期望能耗、动态规划求最优策略、乘法级联模型、线性方程求解、Sniper+McPAT 仿真、统计与可视化分析。

**📊 数据集**

数据集：Shypula 算法核 39,744 个解，3,507,435 次 cycle‑accurate 仿真；AI pipeline：ModernBERT‑base 在 BigVul 上 fine‑tune，22 个优化 knobs（≈4.2M 组合），并测量 22 个单项与 8 个多项组合。

**📈 对比分析**

在算法案例中，EFG 发现 15.6% 的解存在 CV>0.1，并展示 705× 的结构性能耗降低；在 AI 案例中，乘法级联模型对 4 组多项组合的预测误差≤5.1%，仅需 22 次测量即可覆盖 4.2M 组合，显著提升评估效率。

**⚠️ 局限性**

局限性包括：仅在模拟环境下验证，未涵盖真实硬件噪声与 I/O 影响；对某些优化组合仍出现非线性相互作用；当前模型需人工或自动化工具构建图，尚未形成完整可用的生产工具链。

---

## 272. CA-Based Interpretable Knowledge Representation and Analysis of Geometric Design Parameters

**arXiv ID:** 2603.17535 | [PDF](https://arxiv.org/pdf/2603.17535v1)

**作者:** Alexander Köhler `[一作]` (Brandenburgische Technische Universität), Michael Breuß `[通讯]` (Brandenburgische Technische Universität)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了如何从基于主成分分析（PCA）的几何设计参数表示中估计设计参数，解决了高维设计空间带来的挑战。

**💡 创新点**

创新点在于揭示了扩展PCA模型的参数估计结果与标准PCA的结果实际上是相同的，并探讨了可解释参数估计的限制。

**🔧 技术方法**

使用了主成分分析（PCA）技术，并分析了其在几何设计参数估计中的应用。

**📊 数据集**

使用了包含2000个变体的几何类数据集，包括矩形、矩形立方体、螺旋体、风扇叶片和管道等几何形状。

**📈 对比分析**

通过与标准PCA和扩展PCA的比较，发现标准PCA在可解释性和参数估计的准确性上表现更佳，尤其是在几何类未发生变化的情况下。

**⚠️ 局限性**

限制在于当几何类在PCA过程中发生变化时，参数估计的可解释性降低，导致难以准确恢复生成参数。

---

## 273. "Not Just Me and My To-Do List": Understanding Challenges of Task Management for Adults with ADHD and the Need for AI-Augmented Social Scaffolds

**arXiv ID:** 2603.17258 | [PDF](https://arxiv.org/pdf/2603.17258v1)

**作者:** Jingruo Chen `[一作]` (Cornell University), Kexin Nie `[通讯]` (University of Sydney)

**通讯引用:** 3273 | [OpenAlex ID](https://openalex.org/A5112862126)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对22名自我认定或临床诊断的ADHD成年人进行半结构化访谈，随后对20名未参与访谈的ADHD成年人开展速度约会式评估，探索他们在任务管理中的挑战、现有工具的不足以及对AI增强社会支撑的期望。

**💡 创新点**

提出任务管理是一个情感与社会共构的过程，强调需要情绪感知、关系协作与非线性时间节奏的AI支持，并设计了13种基于此洞察的可行性概念，首次将社交与情感维度融入ADHD任务辅助设计。

**🔧 技术方法**

主要使用定性研究方法：主题编码、混合方法设计、情境式速度约会评估；未构建或训练任何机器学习模型。

**📊 数据集**

依赖于研究参与者的自述数据：22名访谈对象与20名速度约会对象的访谈记录、问卷评估及个人背景资料，未使用公开数据集。

**📈 对比分析**

通过受试者对13种概念的5分Likert评分与质性反馈进行对比；未涉及算法性能或实验对照，评估以主观偏好与情感反应为主。

**⚠️ 局限性**

局限性包括样本自选偏差、跨文化与子类型差异未覆盖、研究仅为短时访谈与情境评估、缺乏长期部署与真实使用数据、未独立验证诊断与症状严重度，因而结果仅适用于自认或已诊断的ADHD成年人，未能证明系统在实际环境中的有效性。

---

## 274. Post-Training Local LLM Agents for Linux Privilege Escalation with Verifiable Rewards

**arXiv ID:** 2603.17673 | [PDF](https://arxiv.org/pdf/2603.17673v1)

**作者:** Philipp Normann `[一作]` (TU Wien), Daniel Arp `[通讯]` (TU Wien)

**通讯引用:** 15762 | [OpenAlex ID](https://openalex.org/A5029169901)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在 Linux 特权升级任务上，提出了基于后训练的两阶段 4B 本地 LLM 代理模型，先使用 SFT 对专家轨迹进行微调，再通过 RLVR 优化预算内交互效率。

**💡 创新点**

首次将可验证奖励的后训练（RLVR）应用于交互式安全任务，且采用完全开放权重、局部硬件部署并实现与云端前沿模型相近的性能。

**🔧 技术方法**

SFT (LoRA 微调)、RLVR (Prime-RL+verifiable feedback)、vLLM 推理、异步 Rollout、AIPO、工具调用接口、可验证奖励。

**📊 数据集**

从 10 个程序化生成的特权升级环境收集专家轨迹，排除静态基准的已知解法，构成 1,000 训练样本与 100 验证样本。

**📈 对比分析**

在 12 个公开基准场景下，以固定轮数预算 R=20 的 P(root|R) 作为评估指标；SFT+RL 模型在 20 轮内达 95.8% 成功率，几乎与 Claude Opus 4.6 的 97.5% 相当，同时推理成本降低 100 倍。

**⚠️ 局限性**

仅验证了单一 4B 架构，RL 训练需要 4×H100 GPU 29h；生成器仅覆盖常见错误，未考虑长尾攻击；在局部推理成本极低的情况下，整体训练成本仍高；缺乏跨模型与跨任务的普适性验证。

---

## 275. Multilingual Reference Need Assessment System for Wikipedia

**arXiv ID:** 2603.17146 | [PDF](https://arxiv.org/pdf/2603.17146v1)

**作者:** Aitolkyn Baigutanova `[一作]` (Wikimedia Foundation), Diego Saez-Trumper `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发并部署了一个开源的多语言参考需求评估系统，用于帮助维基百科编辑快速识别需要引用的句子。

**💡 创新点**

创新点包括：①在10种高活跃度语言上实现多语言覆盖；②在生产环境下探讨模型精度与推理延迟的权衡；③通过量化和剪枝等后处理技术在CPU上显著提升推理速度；④将小模型（DistilBERT）与大模型（LLM）在实际任务中的表现进行系统对比。

**🔧 技术方法**

主要技术：多语言BERT的蒸馏版（distilbert-base-multilingual-cased）微调，使用语言代码、章节名、句子三元组等上下文特征；CPU推理部署使用KServe；后处理优化包括PyTorch动态量化、ONNX导出、Intel Neural Compressor以及bettertransformers；LLM零样本推理采用提示工程与softmax概率合并。

**📊 数据集**

数据集：从维基百科数据湖的wikitext表提取的5种训练语言（英语、法语、德语、西班牙语、俄语）共约100k句；测试集覆盖10种语言（含未见语言日语、波斯语、意大利语、葡萄牙语、中文），共约30k句；每句标注为“有引用”(1)或“无引用”(0)；通过特定正则提取feature，并剔除少于6词句、无新内容章节等。

**📈 对比分析**

比较方法：以AUC‑ROC为主指标，同时报告准确率、F1、Precision/Recall；在CPU上评估不同模型大小与序列长度对推理时间的影响；将微调的distilbert-128与原始Citation‑Needed基线、以及四个LLM（Llama‑3‑70b‑chat、Llama‑3‑70B‑Instruct‑Lite、Llama‑3.1‑8B‑Instruct‑Turbo、Mistral‑7B‑Instruct‑v0.3）进行零样本对比。distilbert-128取得AUC‑ROC≈0.765，推理时延≈0.023 s；相比之下Citation‑Needed仅0.650，LLM最高AUC‑ROC≈0.619且推理时延从0.11 s（Llama‑3.1‑8B）到5.3 s（Llama‑3‑70b‑chat）不等。

**⚠️ 局限性**

局限性：①仅在10种高活跃度语言上验证，低资源语言效果未知；②模型依赖DistilBERT，未探索更新的多语言模型；③CPU推理对长篇文章的线性时间开销可能限制可扩展性；④LLM由于资源限制无法在生产中部署，零样本性能仍低；⑤跨语言迁移主要基于西方语言，可能对不同引用风格和文化背景的社区产生偏差。

---

## 276. Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing

**arXiv ID:** 2603.17942 | [PDF](https://arxiv.org/pdf/2603.17942v1)

**作者:** Raghavv Goel `[一作]` (Qualcomm AI Research), Chris Lott `[通讯]` (Qualcomm AI Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种无训练、无草稿模型的多词预测框架，利用掩码词在嵌入空间进行探测以并行预测未来token。

**💡 创新点**

通过动态树扩展与掩码词探测，无需额外训练或模型修改，实现 lossless 推理并显著提升块效率。

**🔧 技术方法**

掩码词初始化、对齐余弦相似度理论、动态令牌树构建、树注意力掩码与位置索引优化。

**📊 数据集**

在 SpecBench 评估，包括摘要、翻译、推理、编码和数学任务。

**📈 对比分析**

与 Prompt Lookup、STAND、Lookahead 等训练自由基线对比，平均接受长度提升 12%–15%，吞吐量提升约 15%–19%，模型调用减少约 40%。

**⚠️ 局限性**

对极大块复杂度或高精度采样的适应性有限，且在某些检索式任务上略逊于 STAND。

---

## 277. LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation in Traditional and Agentic Settings

**arXiv ID:** 2603.17017 | [PDF](https://arxiv.org/pdf/2603.17017v1)

**作者:** Lifu Tu `[一作]` (Oracle AI), Dan Roth `[通讯]` (Oracle AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含约十种扰动的 NL2SQL 鲁棒性评估基准，并在传统单通道与智能体多轮交互两种设置下，对多款大型语言模型进行系统评估。

**💡 创新点**

提出了针对用户查询与数据库更新的多维扰动设计，并首次比较传统与智能体 NL2SQL 在表面噪声与语言变体两类扰动下的性能差异，揭示两种设置的弱点与挑战。

**🔧 技术方法**

使用 GPT‑5.2、GPT‑4.1、Claude‑Opus‑4.6、Gemini‑3‑Pro、Grok‑4.1 等 LLM，在 zero‑shot 与 Spider‑Agent（基于 ReAct 的多轮交互）框架下生成 SQL，并用执行准确率 (Execution Accuracy) 评估效果。

**📊 数据集**

利用 Spider 1（传统设置）和 Spider 2.0‑lite（智能体设置）两个数据集，分别随机抽取 200 / 135 条样本，并对用户查询施加九种扰动，同时在智能体场景中加入额外数据库以测试鲁棒性。

**📈 对比分析**

通过对比各模型在不同扰动类型下的执行准确率，发现传统模型对表面噪声（如 ButterFinger、back‑translation、中文翻译）更易受损，而智能体模型在处理语言变体（如 zh、bt、inflection、past、future）时表现更差；整体而言，绝大多数 LLM 在多数扰动下保持较高准确率，但在上述噪声上仍显著下降。

**⚠️ 局限性**

仅评估了两类噪声，扰动与数据集分离导致难以直接对比；API 调用非确定性导致结果波动，仅取 5 次平均；样本量有限；未探究如何在智能体配置中缓解语言变体造成的性能下降。

---

## 278. Leveraging Large Vision Model for Multi-UAV Co-perception in Low-Altitude Wireless Networks

**arXiv ID:** 2603.16927 | [PDF](https://arxiv.org/pdf/2603.16927v1)

**作者:** Yunting Xu `[一作]` (Nanyang Technological University), Dong In Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 24616 | [OpenAlex ID](https://openalex.org/A5022649488)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于大规模视觉模型的多UAV协同感知框架BHU，并通过Top‑K稀疏化、BEV融合和DDIM‑DRL联合优化来显著降低通信开销与提升感知性能。

**💡 创新点**

创新点在于将Top‑K像素重要性选择与稀疏传输结合，利用MaskDINO‑Swin‑L提取BEV特征，并引入DDIM衍生的DRL算法来高效学习协同UAV选取、稀疏比例与MIMO预编码的联合策略。

**🔧 技术方法**

主要技术包括：多用户MIMO（MU‑MIMO）通信、Swin‑Large Transformer+MaskDINO编码、BEV投影与融合、Top‑K重要性选择与重建、DDIM（去噪扩散隐式模型）辅助的深度强化学习。

**📊 数据集**

实验使用CARLA生成的Air‑Co‑Pred数据集（200个城市交通场景，32k帧RGB图像，1600×900分辨率）。

**📈 对比分析**

与基线DHD（EfficientNet+BEV特征无融合）相比，BHU在相同通信预算下平均提升5%以上的IoU和4%以上的PQ；同时通信负荷减少约85%，实验验证了性能与效率的双重提升。

**⚠️ 局限性**

局限性包括：需依赖预训练的大模型导致模型参数量大，UAV端仅能执行轻量压缩，系统对高度动态移动环境的鲁棒性尚未验证，且DDIM‑DRL的训练成本相对较高。

---

## 279. CodeT5-RNN: Reinforcing Contextual Embeddings for Enhanced Code Comprehension

**arXiv ID:** 2603.17821 | [PDF](https://arxiv.org/pdf/2603.17821v1)

**作者:** Md Mostafizer Rahman `[一作]` (University of Notre Dame), Fang Liu `[通讯]` (University of Notre Dame)

**通讯引用:** 16962 | [OpenAlex ID](https://openalex.org/A5100453091)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了将大型语言模型（RoBERTa、CodeBERT、CodeT5、CodeT5⁺）与循环神经网络（LSTM、BiLSTM、GRU、BiGRU）融合的代码缺陷检测与算法分类框架，结合多任务学习与上下文增强；

**💡 创新点**

创新点在于将LLM的上下文语义表示与RNN的时间序列建模相结合，形成多层次特征融合，并通过上下文恢复与多任务学习显著提升了缺陷检测和算法分类性能；

**🔧 技术方法**

使用的技术包括Transformer‑based LLM（RoBERTa、CodeBERT、CodeT5、CodeT5⁺）、循环神经网络（LSTM/GRU及其双向变体）、多任务损失函数、上下文恢复机制与多层次上下文增强；

**📊 数据集**

实验使用的主要数据集包括公开的代码缺陷检测数据集（CodeNet/CodeXGLUE defect detection）以及三个真实世界的算法分类数据集（SearchAlg、SearchSortAlg、SearchSortGTAlg）；

**📈 对比分析**

通过对比基线模型（BiLSTM、TextCNN、PLBART、C‑BERT、CoTexT等）以及在同一任务上多种优化器与学习率组合的实验，研究显示CodeT5‑GRU/CodeT5⁺‑BiLSTM在缺陷检测上均能达到 93‑97% 的 F1/准确率，整体性能优于现有绝大多数基准；

**⚠️ 局限性**

限制主要包括对学习率和隐藏单元数的高度依赖，模型规模大时训练成本高，且实验聚焦于分类任务，对代码生成或其它程序分析任务的适用性尚未验证。

---

## 280. VISER: Visually-Informed System for Enhanced Robustness in Open-Set Iris Presentation Attack Detection

**arXiv ID:** 2603.17859 | [PDF](https://arxiv.org/pdf/2603.17859v1)

**作者:** Byron Dowling `[一作]` (University of Notre Dame), Adam Czajka `[通讯]` (University of Notre Dame)

**通讯引用:** 1394 | [OpenAlex ID](https://openalex.org/A5067121774)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出 VISER 方法，评估多种人类注意力（手动标注、眼动热图、分割掩模、DINOv2 嵌入）在开放式虹膜表面攻击检测（PAD）中的 saliency‑guided 训练效果。

**💡 创新点**

创新点在于：①首次系统比较不同人类注意力来源对虹膜 PAD 的影响；②提出使用 HDBSCAN 对眼动热图去噪的策略；③将基础模型嵌入（DINOv2）与传统 CNN+交叉熵训练进行对比。

**🔧 技术方法**

技术方法包括：DenseNet‑121 + 交叉熵（XENT）基准；saliency‑guided 训练（交叉熵 + MSE 对齐 CAM 与目标 saliency）；HDBSCAN 去噪眼动热图；基础模型 DINOv2+LogReg / SVM‑Linear / SVM‑RBF 分类器。

**📊 数据集**

使用多源虹膜图像数据集，涵盖 7 种攻击类型（印刷、病变、后脑、合成、贴片+印刷、纹理贴片、人工眼）以及真样本；眼动热图来自 Dowling 等研究的专家与非专家评估。

**📈 对比分析**

实验采用 leave‑one‑attack‑type‑out，训练 12 次模型评估 AUROC 与 APCER@BPCER1%。结果显示去噪初期眼动热图（De‑noised Initial ET）相较 XENT 基准提升 AUROC 0.0608，APCER 0.1063；整体眼动热图亦优于分割掩模和手动标注；基础模型 DINOv2 的表现混合，只有 SVM‑RBF 在 AUROC 上略胜一筹。

**⚠️ 局限性**

局限性：未能完全验证基础模型的开放式测试；眼动数据仅来自单一研究，去噪效果可能受数据集和聚类参数影响；手动标注缺乏足够细粒度，可能导致性能不足；实验仅聚焦虹膜 PAD，其他生物特征未探讨。

---

## 281. Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing

**arXiv ID:** 2603.17531 | [PDF](https://arxiv.org/pdf/2603.17531v1)

**作者:** Pengzhen Chen `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Weiping Wang `[通讯]` (Institute of Information Engineering, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 Relational Zero‑Watermarking（Rel‑Zero）框架，利用生成式编辑下图像 Patch 对间距离不变性实现无侵入式零水印生成与验证。

**💡 创新点**

创新点在于将 Patch 对间距离的鲁棒性作为水印特征，既避免了传统嵌入式水印对图像质量的损害，又显著提升了对生成编辑的抵抗力。

**🔧 技术方法**

技术方法包括使用 Vision Transformer 提取 Patch 特征、构建全连通 Patch 对集合、用 MLP 预测稳定对，并通过预训练 VAE 模拟编辑获取训练标签。

**📊 数据集**

实验数据集主要包括 COCO（训练集）、UltraEdit 与 MagicBrush（10,000 张评估图像）以及多种生成编辑模型（InstructPix2Pix、MagicBrush、UltraEdit、ControlNet‑Inpainting 等）。

**📈 对比分析**

与 DWT‑DCT、VINE、Robust‑Wide（嵌入式）以及 ConZWNet、FGPCET（零水印）等基线对比，Rel‑Zero 在保持 PSNR、SSIM、LPIPS 等视觉质量不变的前提下，在 0.1% FPR 下的 TPR 在多种生成编辑和常见失真上均优于或与最优方法相当。

**⚠️ 局限性**

局限性包括：对大规模全局编辑（如 Deterministic Regeneration）鲁棒性相对较弱；对极粗粒度 Patch 划分或高噪声环境下的判别性能可能下降。

---

## 282. RHYME-XT: A Neural Operator for Spatiotemporal Control Systems

**arXiv ID:** 2603.17867 | [PDF](https://arxiv.org/pdf/2603.17867v1)

**作者:** Marijn Ruiter `[一作]` (KTH Royal Institute of Technology), Amritam Das `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 129 | [OpenAlex ID](https://openalex.org/A5013924360)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了RHYME‑XT框架，用于通过数据学习非线性输入仿射PIDE的解算子，能在空间时间连续、离散无关的条件下逼近脉冲神经场模型的动力学；

**💡 创新点**

创新点包括：①使用神经网络学习Galerkin投影基函数而无需预先选择基底；②结合流函数学习（RNN）直接逼近投影系统的流图，避免求解ODE；③使用非线性输出重构网络提升表达能力；

**🔧 技术方法**

采用随机傅里叶特征映射+多层感知器构建基函数网络，LSTM网络实现流函数学习，三组FNN分别负责基函数、流函数编码/解码和输出重构；

**📊 数据集**

数据集基于一维神经场模型，生成1000条轨迹（T=50），每条轨迹随机初始条件和周期性输入，使用Fast Fourier和Euler积分产生训练/验证/测试集；

**📈 对比分析**

与DeepONet基准进行对比，RHYME‑XT在相同参数规模下的相对ℓ²误差从0.1284±0.1287降至0.0571（单轨迹）或0.1284±0.1287（测试集），并在延长时间（T=250）误差仅微幅上升，表现出更好的时间外推性能；

**⚠️ 局限性**

局限性包括：仅在单一PIDE实例（神经场）验证；对更复杂或高维非线性PDE的泛化尚未充分评估；模型训练依赖大量轨迹，且在参数选择和基函数维数上仍需经验调优。

---

## 283. KGS-GCN: Enhancing Sparse Skeleton Sensing via Kinematics-Driven Gaussian Splatting and Probabilistic Topology for Action Recognition

**arXiv ID:** 2603.16943 | [PDF](https://arxiv.org/pdf/2603.16943v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 284. KineVLA: Towards Kinematics-Aware Vision-Language-Action Models with Bi-Level Action Decomposition

**arXiv ID:** 2603.17524 | [PDF](https://arxiv.org/pdf/2603.17524v1)

**作者:** Gaoge Han `[一作]` (MBZUAI), Tongliang Liu `[通讯]` (MBZUAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 KineVLA，一种能够从细粒度运动指令中理解并执行精确机械臂动作的视觉-语言-动作框架。

**💡 创新点**

创新点在于引入双层向量量化动作表示（分离目标层与运动层）和双层链式推理令牌，通过互信息正则化将语言推理与动作生成紧密对齐，实现对指令级运动学约束的精细控制。

**🔧 技术方法**

使用了残差向量量化变分自编码器 (RVQ‑VAE) 进行双层动作编码、基于 OpenVLA 的 7B 视觉‑语言模型、LoRA 微调、以及互信息最大化的正则化策略。

**📊 数据集**

构建了三套数据集：LIBERO‑Goal‑Relabeled、Kine‑LIBERO 以及 Realman‑75，均包含从模拟到真实机械臂的动作、视觉和精细运动学注释。

**📈 对比分析**

与 OpenVLA、π_0.5 和 VQ‑VLA 等基线对比，KineVLA 在目标成功率相近的情况下，在运动学成功率上提升约 10–15%，并且推理速度仅略高于单层 VQ‑VAE，保持实用性。

**⚠️ 局限性**

局限性主要体现在实验仅覆盖桌面级操作，未验证更大尺度或全身操控的适用性，且模型仍需依赖大量带推理注释的数据进行监督训练。

---

## 285. MM-OVSeg:Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing

**arXiv ID:** 2603.17528 | [PDF](https://arxiv.org/pdf/2603.17528v1)

**作者:** Yimin Wei `[一作]` (Institution1), Naoto Yokoya `[通讯]` (Institution2)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了一种多模态（RGB+SAR）ViT架构的开源词汇遥感语义分割方法MM-OVSeg，利用CLIP和DINO编码器融合并对SAR进行对齐以提升分割性能。

**💡 创新点**

创新点在于：① 将CLIP与DINO双编码器对齐并在多尺度特征层进行跨模态融合；② 引入CMU方法对SAR进行CLIP‑SAR对齐，进一步增强跨模态一致性；③ 通过多模态融合显著提升见/未见类别的均衡表现。

**🔧 技术方法**

使用技术包括：ViT-B/16/ViT‑L/14 视觉编码器、CLIP 与 DINO 预训练模型、InfoNCE 对齐损失、AdamW 优化、双编码器并行推理以及多尺度特征提取。

**📊 数据集**

使用六个遥感数据集（PIE、DDHR、OEM、PIE‑clean 等）进行在域与跨域实验，涵盖云覆盖、不同传感器与光照变化等多样场景。

**📈 对比分析**

与现有方法（CAT‑Seg、EBSeg、GSNet、SegEarth‑OV 等）在见/未见类别的 mIoU 进行对比，MM‑OVSeg 在所有设置下取得最高均衡得分，ViT‑L/14 版本平均 mIoU 达到 55.0%，显著优于同行者。

**⚠️ 局限性**

限制包括：模型参数量和计算量大，训练与推理延时与碳排放较高；对极端天气或新模态的适应性仍有提升空间；以及双编码器耦合导致实现与部署复杂度上升。

---

## 286. PACE-RAG: Patient-Aware Contextual and Evidence-based Policy RAG for Clinical Drug Recommendation

**arXiv ID:** 2603.17356 | [PDF](https://arxiv.org/pdf/2603.17356v1)

**作者:** Chaeyoung Huh `[一作]` (Korea Advanced Institute of Science and Technology), Jong Chul Ye `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 17519 | [OpenAlex ID](https://openalex.org/A5012644755)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种四阶段 PACE‑RAG 框架，利用患者关键症状检索、处方倾向分析与策略驱动的细化，最终生成可解释的药物推荐。

**💡 创新点**

创新点在于融合焦点特定检索与处方倾向分析，并通过策略驱动的验证环节消除指南泛化和多数偏差，实现更精准且可解释的处方决策。

**🔧 技术方法**

采用检索增强生成（RAG）与大语言模型（如 Llama‑3.1‑8B、Qwen‑3‑8B）结合的多阶段推理管线。

**📊 数据集**

使用韩国海云岛派克医院的帕金森病非结构化临床记录以及公开的 MIMIC‑IV 结构化数据进行评估。

**📈 对比分析**

与零样本基线、Guideline RAG、TreatRAG、MedReflect 等方法对比，PACE‑RAG 在 HPH 和 MIMIC‑IV 上的 F1、准确率、精确率等指标均明显优于对照方法，甚至在 8B 模型上超过更大参数量模型。

**⚠️ 局限性**

局限包括单中心数据偏倚、推理延迟和多阶段提示工程依赖，以及韩英翻译可能导致的语义细节损失。

---

## 287. Cryptographic Runtime Governance for Autonomous AI Systems: The Aegis Architecture for Verifiable Policy Enforcement

**arXiv ID:** 2603.16938 | [PDF](https://arxiv.org/pdf/2603.16938v1)

**作者:** Adam Massimo Mazzocchetti `[一作]` `[通讯]` (SPQR Technologies Inc), Adam Massimo Mazzocchetti (SPQR Technologies Inc)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一种名为 Aegis 的运行时治理架构，能够将 AI 系统在启动时绑定到不可变的伦理政策层，并通过零知识证明、强制执行模块和不可篡改日志在运行时强制执行合规性，违反时会自动关机并生成审计证据。

**💡 创新点**

创新点在于：① 将伦理政策作为执行依赖而非事后建议；② 采用加密封闭的 Immutable Ethics Policy Layer（IEPL）与 Genesis Lock；③ 引入零知识证明（zk‑STARK）实时验证发布行为；④ 设计了可审计的 Immutable Logging Kernel（ILK）和自动关机机制；⑤ 通过分布式四分之一仲裁（Senatus）实现可扩展的政策修订。

**🔧 技术方法**

使用技术包括：Rust、Solidity、Python/Go 的混合堆栈；零知识证明框架（zk‑STARK）；加密哈希与签名；Genesis Lock 与多方共识验证；EVA、EKM、ILK 等运行时模块；以及对硬件身份的根证书绑定。

**📊 数据集**

实验使用了自研的 Civitas 运行时，并在其上构造了一套匹配任务（10,000 决策周期）来测试合规性、发布行为和恶意篡改。没有使用公开的标准数据集，而是基于内部生成的测试任务集。

**📈 对比分析**

比较方法：在受治理和无治理两种条件下运行相同的任务集，收集对齐保留率、拒绝率、恢复稳定性、发布延迟与验证延迟等指标。结果显示：治理版对齐保留率 98.2%（±0.7）vs 65.7%（±3.1）；拒绝率 12.3% vs 0%；恢复稳定性 2.3 vs 7.1 轮；发布延迟平均增加 9.4 ms；验证延迟 238 ms（±17）。

**⚠️ 局限性**

局限性包括：① 伦理政策的形式化与表达仍有挑战，难以覆盖所有司法和情境细节；② 实验仅在受控内部环境中进行，缺乏公开的第三方验证；③ 缺乏跨标准基准（如对抗提示、网络延迟、分布式验证器失效）的评估；④ 决议共识模型仍面临共谋和治理捕获风险；⑤ 人类裁决仍在政策制定、修订规则与冲突解决中必不可少。

---

## 288. The Phasor Transformer: Resolving Attention Bottlenecks on the Unit Circle

**arXiv ID:** 2603.17433 | [PDF](https://arxiv.org/pdf/2603.17433v1)

**作者:** Dibakar Sigdel `[一作]` `[通讯]` (Mindverse Computing), Dibakar Sigdel (Mindverse Computing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种大型相位模型（LPM），通过将时间序列映射到单位圆上的相位状态，并使用无参数离散傅里叶变换（DFT）与可训练相位旋转门实现全局token交互，旨在替代传统自注意力模块。

**💡 创新点**

创新点在于将序列状态直接放置在S¹几何空间，利用确定性DFT混合实现O(N log N)的全局交互，并在每层之间加入相位归一化（arcsin(sin)）以保证深层稳定性，从而在保持全局依赖的同时显著降低参数量和计算复杂度。

**🔧 技术方法**

采用的技术包括相位编码、单位圆几何约束、无参数DFT混合、可训练的相位旋转门、跨层相位归一化、PyTorch复数梯度训练和基于FFT的高效实现。

**📊 数据集**

实验使用合成的多频噪声自回归时间序列数据，窗口长度为10或32，作为一步预测任务的基准数据集。

**📈 对比分析**

通过与传统PyTorch自注意力Transformer（4-head注意力、FFN）在相同任务上进行MSE/MAE对比，LPM仅使用约50–64个可训练角度，混合复杂度为O(N log N)，在保持相对较低参数量和计算成本的同时，预测误差略高于自注意力模型。

**⚠️ 局限性**

局限性包括仅在合成数据上验证，缺乏对真实长序列或多模态数据的评估；相位约束可能对非周期性特征表现不足；在更大上下文或更深层时仍需进一步验证模型的稳定性和泛化能力。

---

## 289. Symphony: A Cognitively-Inspired Multi-Agent System for Long-Video Understanding

**arXiv ID:** 2603.17307 | [PDF](https://arxiv.org/pdf/2603.17307v1)

**作者:** Haiyang Yan `[一作]` (Institute of Automation, Chinese Academy of Sciences), Mengyi Liu `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Symphony 多智能体系统，利用规划、定位、字幕、视觉感知和反思代理分解长视频理解任务

**💡 创新点**

创新点在于基于认知维度的任务拆分、反思驱动的动态协作机制以及结合 LLM 与 VLM 的定位策略

**🔧 技术方法**

采用多智能体架构、LLM（DeepSeek）、VLM（Seed 1.6 VL）、CLIP、Token 压缩和对齐技术

**📊 数据集**

在 LVBench、LongVideoBench、VideoMME 和 MLVU 四大长视频理解基准上进行评测

**📈 对比分析**

与现有 VLM、Agent 等方法相比，Symphony 在 LVBench 提升约 5%（SOTA），在其他基准亦取得最高分

**⚠️ 局限性**

局限包括对多模态对齐的复杂度、对计算资源要求较高以及缺乏实时性和跨域泛化能力

---

## 290. Pathology-Aware Multi-View Contrastive Learning for Patient-Independent ECG Reconstruction

**arXiv ID:** 2603.17248 | [PDF](https://arxiv.org/pdf/2603.17248v1)

**作者:** Youssef Youssef `[一作]` (Indian Institute of Technology Roorkee), Jitin Singla `[通讯]` (Indian Institute of Technology Roorkee)

**通讯引用:** 434 | [OpenAlex ID](https://openalex.org/A5001845207)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

提出一种基于病理感知多视角对比学习的患者独立12导联心电图重建框架，利用有限导联重建缺失的导联。

**💡 创新点**

创新点在于通过监督对比损失在潜在空间构建病理感知锚点，将重建任务约束在病理子空间，显著提升了跨患者与跨数据集的泛化性能。

**🔧 技术方法**

使用了1D卷积编码器、监督对比学习、互信息最大化、时间域波形归一化以及堆叠潜在解码器等技术。

**📊 数据集**

主要使用了PTB‑XL大规模心电数据库进行训练与验证，并在PTB诊断数据库上进行跨数据集评估。

**📈 对比分析**

与基线模型及Rajotte等公开方法在患者独立划分下对比，平均RMSE下降约76%，R²与Pearson相关系数均优于现有方法，并在PTB数据库上保持较高的重建精度。

**⚠️ 局限性**

局限性包括对高质量病理标签的依赖、对极罕见病理的表现仍有限，以及仅在12导联重建任务上验证，尚未测试更大范围的导联或其他设备环境。

---

## 291. Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency

**arXiv ID:** 2603.17102 | [PDF](https://arxiv.org/pdf/2603.17102v1)

**作者:** Lucas Bandarkar `[一作]` (University of California), Trevor Cohn `[通讯]` (University of Melbourne)

**通讯引用:** 7852 | [OpenAlex ID](https://openalex.org/A5078530959)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用跨语言不一致性对比，开发XICI框架在稀疏MoE LLM中定位回答特定事实所需的专家，并通过因果消融验证其重要性。

**💡 创新点**

创新点在于：①用模型自然产生的跨语言错误与成功作为对比信号，避免人工扰动；②结合统计检验（MWU）与阈值筛选，识别少量专家；③在MoE架构中展示少量专家激活就能显著影响答案。

**🔧 技术方法**

技术手段包括：跨语言路由日志收集、专家去黑名单、语言平均值去偏、Mann‑Whitney U检验、差异阈值筛选、专家排名、因果消融（专家禁用）以及随机基线对照。

**📊 数据集**

数据集：来自Multilingual QA、Wiki-fact、Prehistory子集的约400个短答案事实问题，分别翻译成12–31种语言；模型使用Qwen3‑30B‑A3B‑Instruct‑2507和GLM‑130B两大稀疏MoE LLM。

**📈 对比分析**

评估方法：与随机选取相同数量专家、随机问题洗牌两种基线对比；在Qwen3上得到约44% Rate Difference，GLM约32%；消融0.3%专家即可使约40%问题答案变为错误，证明定位方法有效。

**⚠️ 局限性**

局限性：需要跨语言不一致的事实，因果消融对知识存储的解释有限；结果受模型规模与事实难度影响；缺乏可直接比较的其他专家定位基线；对专家内部机制的粒度解释不足。

---

## 292. P$^{3}$Nav: End-to-End Perception, Prediction and Planning for Vision-and-Language Navigation

**arXiv ID:** 2603.17459 | [PDF](https://arxiv.org/pdf/2603.17459v1)

**作者:** Tianfu Li `[一作]` (Hong Kong University of Science and Technology), Haoang Li `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 903 | [OpenAlex ID](https://openalex.org/A5040338788)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `e0540dec-d77f-42db-94ae-d039248f6393` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出P^3Nav框架，实现感知、预测与规划的统一端到端设计；通过对象级感知、地图级感知、路径点预测和未来场景预测提升视觉语言导航的场景理解与成功率。

**💡 创新点**

将多级感知（对象+地图）、未来状态预测与前瞻场景预测整合到单一可微网络；利用BEV投影与变形注意力解码器，实现对对象关系与未来语义的直接编码；端到端训练消除信息丢失与误差累积。

**🔧 技术方法**

Lift‑Splat‑Shoot + 变形自注意力与交叉注意力 Transformer 解码器；MSE/交叉熵损失，多任务预训练（MLM、SAP、OG）；图 Transformer 用于规划；NMS+深度过滤生成路径点；BEV 表示与地图语义编码。

**📊 数据集**

REVERIE、R2R‑CE、RxR‑CE（及对应离散版本 R2R、RxR）。

**📈 对比分析**

在 REVERIE、R2R‑CE、RxR‑CE 上与 HAMT、DUET、DSRG、GridMM、BEVBert、BSG、VER、GOAT 等SOTA模型对比，P^3Nav 在 SR、SPL、RGS、NE、OSR、nDTW 等指标上刷新SOTA，提升约3‑4% SPL、3‑5% RGS，并在所有验证/测试未见集上保持领先。

**⚠️ 局限性**

对高分辨率 BEV 受遮挡影响，地图语义生成依赖 VLM，真实场景鲁棒性受限；模型参数多、训练成本高，模块间仍可能出现误差传播，需进一步提升效率与泛化能力。

---

## 293. Neural Radiance Maps for Extraterrestrial Navigation and Path Planning

**arXiv ID:** 2603.17236 | [PDF](https://arxiv.org/pdf/2603.17236v1)

**作者:** Adam Dai `[一作]` (Stanford University), Grace Gao `[通讯]` (Stanford University)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5069625302)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用NeRF构建全局地图，结合本地传感器观测的成本信息，实现无人车的在线路径重规划。

**💡 创新点**

创新点在于使用NeRF提取地形特征并通过核岭回归将稀疏的局部成本信息扩散至全局成本图，从而实现全局路径即时更新。

**🔧 技术方法**

核心技术包括NeRF训练与渲染、特征提取、核岭回归插值、分段路径规划（AutoNav本地规划+全局A*）。

**📊 数据集**

数据集为模拟无人机航拍图像（Unreal Engine与AirSim生成），以及Google Earth Studio的火星影像。

**📈 对比分析**

在AirSim月球仿真环境中与不进行全局重规划的基线对比，结果显示我们的方案路径成本降低≈65%，碰撞数为0，时间略增。

**⚠️ 局限性**

局限性包括仅在仿真环境验证，缺乏真实外太空地形数据，NeRF特征聚类与插值误差随距离递减，且对光照变化敏感。

---

## 294. Flow Matching Policy with Entropy Regularization

**arXiv ID:** 2603.17685 | [PDF](https://arxiv.org/pdf/2603.17685v1)

**作者:** Ting Gao `[一作]` (Delft University of Technology), Serge Hoogendoorn `[通讯]` (Delft University of Technology)

**通讯引用:** 22573 | [OpenAlex ID](https://openalex.org/A5077352940)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

提出了一种基于流匹配的熵正则化在线强化学习框架FMER，解决了传统扩散政策在熵控制和采样效率上的瓶颈。

**💡 创新点**

创新点包括：① 用ODE流匹配替代SDE，直接学习可解析的概率路径；② 构造优势加权的条件流匹配损失，实现对高价值动作空间的软性引导；③ 推导出闭式熵表达式，使得最大熵优化可直接实现；④ 在隐空间使用tanh映射保证动作合法，并通过Hutchinson估计高效计算熵。

**🔧 技术方法**

主要技术包括：条件流匹配（CFM）、优势加权流匹配（W-CFM）、熵正则化（analytic entropy）、隐潜空间tanh映射、Hutchinson迹估计、双Q学习、Euler求解器、离线/在线经验回放、Lagrange乘子自适应调节。

**📊 数据集**

实验数据集：FrankaKitchen多目标稀疏奖励任务（N=1,2,4,7）、2D多目标可视化环境、MuJoCo四大动态任务（Hopper、HalfCheetah、Walker2d、Humanoid）。

**📈 对比分析**

与PPO、SAC、TD3、FPO、DIPO、DPMD、QVPO等基线比较；在FrankaKitchen上FMER以100%成功率击败所有基线；在MuJoCo上与最优扩散方法相当或略优；训练时间比QVPO快约7倍，显存占用显著降低，整体性能提升明显。

**⚠️ 局限性**

局限性：使用固定步长Euler求解可能导致积分误差；目标熵固定缺乏动态调节；对高维连续动作空间的积分误差和数值稳定性仍需进一步研究；需探索更高阶ODE求解器和自适应熵调度以进一步提升收敛速度与稳定性。

---

## 295. Predicting Trajectories of Long COVID in Adult Women: The Critical Role of Causal Disentanglement

**arXiv ID:** 2603.17722 | [PDF](https://arxiv.org/pdf/2603.17722v1)

**作者:** Jing Wang `[一作]` (National Library of Medicine), Jeremy C. Weiss `[通讯]` (National Library of Medicine)

**通讯引用:** 2283 | [OpenAlex ID](https://openalex.org/A5072774346)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

针对成人女性的长新冠（PASC）预测进行研究，整合临床资料与四周可穿戴监测数据。

**💡 创新点**

创新点是提出基于大语言模型的因果去耦合架构，将病理信号与基线噪声（如更年期、糖尿病等）显式分离。

**🔧 技术方法**

使用Qwen-2.5-0.5B LLM、注意力去耦层、环境混合策略和InfoNCE对比损失实现因果特征提取与回归。

**📊 数据集**

数据集来自NIH RECOVER数据库，包括1155名女性受试者的静态临床信息、心率变异性、睡眠结构与运动量等可穿戴数据。

**📈 对比分析**

与XGBoost基线对比，Causal网络在分类精度上更优（0.867±0.044）且对噪声抑制更强，但回归误差（RMSE≈5.35）略高于XGBoost（RMSE≈4.76）。

**⚠️ 局限性**

局限包括对模型初始化敏感、回归性能不及传统树模型、以及受试者样本量与多样性仍有限。

---

## 296. Trust the Unreliability: Inward Backward Dynamic Unreliability Driven Coreset Selection for Medical Image Classification

**arXiv ID:** 2603.17603 | [PDF](https://arxiv.org/pdf/2603.17603v1)

**作者:** Yan Liang `[一作]` (Sichuan University), Yi Zhang `[通讯]` (Sichuan University)

**通讯引用:** 96582 | [OpenAlex ID](https://openalex.org/A5100388089)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了动态不可靠性驱动的核心采样（DUCS）方法，用于在医疗图像分类中高效选取最具信息量的样本，从而减少数据量并保持高精度。

**💡 创新点**

创新点在于将“向内自我意识”与“向后记忆追踪”两种视角结合，利用置信度波动和遗忘频率共同衡量样本的不可靠性，精准挖掘靠近决策边界的关键样本，避免传统方法过度选择易分类中心样本。

**🔧 技术方法**

技术方法包括：使用 Dirichlet 分布建模输出以获得更平滑的置信度；滑动窗口计算置信度方差；跟踪每个样本的遗忘事件以得到遗忘频率；将两者加权得到 Unreliability Score，用于核心采样。

**📊 数据集**

实验数据集涵盖医疗图像领域的 MedMNIST（OrganAMNIST、OrganSMNIST）、PneumoniaMNIST、3D OrganMNIST3D/NoduleMNIST3D/FractureMNIST3D，以及通用图像数据集 CIFAR‑10/100，用于评估方法的鲁棒性与迁移性。

**📈 对比分析**

与九种现有核心采样基线（Random、Forgetting、Entropy、EL2N、AUM、CCS、EVA、Moderate、TDDS）进行对比；在低采样率（0.5%–30%）下，DUCS 在医学数据集上显著优于所有基线，并在高采样率以及跨网络架构迁移中保持领先性能，甚至在自然图像数据集上也实现了最优或竞争性的准确率。

**⚠️ 局限性**

局限性：方法需要在整个训练过程中记录置信度和遗忘信息，导致额外的存储和计算开销；当数据规模极大或训练时间受限时，实时记录与更新可能成为瓶颈，未来可考虑轻量化或在线近似方案。

---

## 297. Building a "-Sensitive Design" Methodology from Political Philosophies or Ideologies

**arXiv ID:** 2603.17806 | [PDF](https://arxiv.org/pdf/2603.17806v1)

**作者:** Anthony Maocheia-Ricci `[一作]` (University of Waterloo), Edith Law `[通讯]` (University of Waterloo)

**通讯引用:** 2475 | [OpenAlex ID](https://openalex.org/A5058482884)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出了基于政治哲学或意识形态的“-敏感设计”元框架，并以依赖性敏感设计（DSD）为例演示其在技术设计中的应用。

**💡 创新点**

创新点在于将价值敏感设计（VSD）与可能力敏感设计（CSD）相结合，形成可扩展的“-敏感设计”元框架，并将Kittay的依赖性批判引入价值层次，提供了新的价值导向和相互关联机制。

**🔧 技术方法**

采用三元调查法（概念、经验、技术）与价值层级模型、CSD-Cascade流程，以及工作坊中的能力卡等工具进行设计探索和需求转化。

**📊 数据集**

未使用传统数据集；研究以文献综述、哲学文本解读和设计方法论演练为主。

**📈 对比分析**

未进行实验性比较，主要通过与已有VSD、CSD方法的理论对照来说明DSD在关注弱势群体方面的优势，但在实证验证上仍待进一步研究。

**⚠️ 局限性**

局限性包括：缺乏具体案例验证、方法复杂度高、对价值解释仍有主观空间、缺少可衡量的性能评估指标。

---

## 298. Ensemble Self-Training for Unsupervised Machine Translation

**arXiv ID:** 2603.17087 | [PDF](https://arxiv.org/pdf/2603.17087v1)

**作者:** Ido Aharon `[一作]` (Bar-Ilan University), Sarit Kraus `[通讯]` (Bar-Ilan University)

**通讯引用:** 18405 | [OpenAlex ID](https://openalex.org/A5103213461)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在无监督机器翻译中构建多模型集成自训练框架，利用不同辅助语言训练的UNMT模型通过词级集成生成高质量伪平行数据，进一步提升模型性能，最终只部署单一模型；

**💡 创新点**

创新点在于将集成作为训练时的监督生成机制，而非仅在推理时使用，且通过辅助语言多样性实现模型结构化多样性，显著改善伪标签质量；

**🔧 技术方法**

采用无监督翻译的后向翻译、混合训练（结合随机对齐）、词级logits平均集成、伪平行数据再训练等技术；

**📊 数据集**

使用Europarl数据集的单语子集进行无监督训练，低资源实验为英-波（仅100K波语句），高资源实验为英-法；

**📈 对比分析**

与匹配的单模型额外训练基线（相同训练步数）对比，平均chrF提升约1.19，翻译出方向提升1.7，入方向提升0.67，且差异具有统计显著性；

**⚠️ 局限性**

局限在于仅在小规模模型上验证，未探索更大模型、多样性来源的扩展，以及与其他无监督技术（如去噪预训练、质量筛选）的融合；

---

## 299. DebugLM: Learning Traceable Training Data Provenance for LLMs

**arXiv ID:** 2603.17884 | [PDF](https://arxiv.org/pdf/2603.17884v1)

**作者:** Wenjie Jacky Mo `[一作]` (University of California), Muhao Chen `[通讯]` (University of California)

**通讯引用:** 2056 | [OpenAlex ID](https://openalex.org/A5100374415)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为大语言模型提供内在的数据来源追溯能力，使模型能够在生成时自行报告其行为的训练数据来源，并通过调试接口实现目标性行为修正。

**💡 创新点**

创新点在于将训练数据来源标签嵌入模型参数，采用双模式训练（标准模式+调试模式）与可触发的调试前缀，构建可控的“内部追溯”机制，兼具实时追溯与无须重训的即时修正。

**🔧 技术方法**

技术包括：多阶段训练框架；为每个数据源分配单一特殊 token 作为 provenance tag；在训练时添加调试前缀并将标签附加到目标序列；使用加权损失融合标准生成与追溯学习；在推理时通过前缀触发返回标签或执行源级拒绝；对比 BM25、ROUGE-L、SBERT、基线分类器等后置归因方法。

**📊 数据集**

使用 Llama‑3‑8B 与 Qwen‑3‑8B 两个 8B 基础模型，在两阶段训练下混合 TOFU、ChatDoctor、TruthfulQA、Beavertails、WMDP 这五个多源数据集；随后在多源合成任务 QuoteSum 与多标签微粒化实验（作者级 199 标签、毒性子类 14 标签）上进行评估。

**📈 对比分析**

与传统后置归因方法相比，本文方法在所有测试集上实现 95%+ 的追溯成功率，且仅需一次前向推理；在目标性修正任务中，模型在指定源上 100% 拒绝率、非目标源 0% 拒绝率；且在标准生成模式下保持与基线相当或略优的任务性能。

**⚠️ 局限性**

局限性：只能在训练阶段使用，无法应用于已冻结的黑盒模型；标签数量扩展到数百万时可能出现容量瓶颈与标签干扰；对多源推理的整体任务实用性仍受基线模型推理与组合推理能力的限制。

---

## 300. Interpretable Traffic Responsibility from Dashcam Video via Legal Multi Agent Reasoning

**arXiv ID:** 2603.17930 | [PDF](https://arxiv.org/pdf/2603.17930v1)

**作者:** Jingchun Yang `[一作]` (Northeast University), Jinchang Zhang `[通讯]` (SUNY Binghamton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了C-TRAIL多模态法律数据集，并提出基于视频理解和法律多代理推理的两阶段框架，实现从行车记录仪视频到责任模式和适用法规的可解释推断。

**💡 创新点**

首次将行车记录仪视频与文本与责任模式及对应法规对齐，设计了可解释的多代理法律推理系统。

**🔧 技术方法**

使用自监督的视差与姿态网络提取车道运动，事件分割与LSTM生成字幕，事实聚合代理构造案例事实，法律资源检索模块结合检索式与语义重排序，三位角色（Issue、Law-Precedent、Deliberation）构成多代理判决。

**📊 数据集**

采用自己构建的C-TRAIL数据集（约1,000个车祸视频）和改编自MM-AU的视频数据进行预过滤。

**📈 对比分析**

在C-TRAIL上与GPT‑4、法律LLM及其他多代理方法比较，准确率 86.4%、宏F1 79.1% 及核心法规命中率 73.2%，均优于对照组；在MM‑AU 的事故原因回答任务上亦实现 86% 以上准确率。

**⚠️ 局限性**

仅适用于中国交通法规，责任模式受限于预设闭合集，数据量有限且对视频字幕质量依赖较大，跨域推广需进一步验证。

---

## 301. MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning

**arXiv ID:** 2603.16929 | [PDF](https://arxiv.org/pdf/2603.16929v1)

**作者:** Hongjun Wang `[一作]` (University of Hong Kong), Kai Han `[通讯]` (University of Hong Kong)

**通讯引用:** 9876 | [OpenAlex ID](https://openalex.org/A5101784732)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的强化学习框架 Modulated Hazard-aware Policy Optimization（MHPO），通过对重要性比进行平滑调制并引入风险感知惩罚来稳定群组相对策略优化（GRPO）训练。

**💡 创新点**

创新点包括：① Log‑Fidelity Modulator（LFM）——在对数域使用缩放 tanh 将无界重要性比映射到有界可微空间，实现梯度保真与全局抑制；② Decoupled Hazard Penalty（DHP）——利用 Weibull 累积分布的风险函数，对正负策略偏移分别进行异向抑制，形成可调节的信任区间与快速衰减；③ 结合半梯度优化，保证惩罚只缩放梯度幅度不改变方向，理论上给出梯度乘子有界性和方差上界。

**🔧 技术方法**

采用了对数-tanh 调制、softplus 和 Weibull 风险函数的组合，形成平滑可微的梯度乘子；使用半梯度（stop‑gradient）避免惩罚梯度反向传播；利用多样本群组优势归一化实现群组相对奖励；在实现层面基于 Qwen 系列 LLM 与 Qwen‑VL‑7B 视觉语言模型。

**📊 数据集**

数据集：文本数学推理任务使用 DAPO‑Math‑17k、AIME 2024/2025、AMC 2023、HMMT25、MATH‑500；视觉‑语言数学推理使用 Geometry3K 训练，评测 MathVision、MathVista、MathVerse；模型包括 Qwen3‑4B‑Base、Qwen2.5‑7B‑Instruct、Qwen2.5‑Math‑7B‑Instruct、Qwen2.5‑VL‑7B‑Instruct。

**📈 对比分析**

与 GRPO、GSPO、DAPO、SAPO 等现有方法在 Avg@32（多选题通过率）上进行对比。MHPO 在所有模型与任务上均实现了 3–7% 的平均提升，尤其在 AIME、AMC、MathVision 等高难度基准上显著超越前沿方法，并在梯度幅度与奖励曲线上表现出更平稳、更早收敛的特征。

**⚠️ 局限性**

局限性：需为正负风险参数和 LFM 缩放因子手动调参，超参数敏感；实验集中在数学推理与视觉‑语言推理任务，泛化到其他 RL 任务尚未验证；实现相对复杂，额外的风险函数和半梯度机制可能增加计算开销；理论分析主要针对梯度乘子有界性，未覆盖长期策略收敛性和与奖励模型的耦合问题。

---

## 302. GUIDE: GenAI Units In Digital Design Education

**arXiv ID:** 2603.17296 | [PDF](https://arxiv.org/pdf/2603.17296v1)

**作者:** Weihua Xiao `[一作]` (New York University), Ramesh Karri `[通讯]` (New York University)

**通讯引用:** 16718 | [OpenAlex ID](https://openalex.org/A5059648257)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个名为GUIDE的开源课程仓库，提供标准化的教学单元（幻灯片、视频、Colab实验和相关论文），用于教授基于大型语言模型（LLM）的数字设计与验证。

**💡 创新点**

创新点在于将GenAI与数字设计教学模块化、可复用，统一单元结构并包含可运行的Colab实验，支持快速更新与跨课程复用；同时展示了完整课程实例与项目。

**🔧 技术方法**

采用LLM（如ChatGPT、Gemma、其他开源模型）、Google Colab、Yosys/Icarus等EDA工具进行RTL生成、测试平台生成、SVA/安全属性自动化等。

**📊 数据集**

使用了多种RTL生成与验证数据集，例如VeriThoughts、VGen、AutoChip、ROME、Veritas、VeriContaminated等；并在课程中利用公开的Verilog Benchmark库。

**📈 对比分析**

通过在四个完整课程实例（GUIDE4ChipDesign、Build your ASIC、GUIDE4HardwareSecurity、Hardware Design）中的学生项目评估，展示了LLM驱动工作流在RTL生成、测试验证与安全攻击/防御方面的可行性与学习效果，项目在课堂和CSAW等竞赛中取得成功。

**⚠️ 局限性**

局限性包括：LLM模型和提示方式快速迭代导致课程内容易过时；需要兼顾不同层次、专业背景的学生；整合LLM与传统EDA工具的工作流复杂；目前仅覆盖数字设计，尚未扩展至HLS、物理设计与模拟电路等领域。

---

## 303. The Port-Hamiltonian Structure of Vehicle Manipulator Systems

**arXiv ID:** 2603.16882 | [PDF](https://arxiv.org/pdf/2603.16882v1)

**作者:** Ramy Rashad `[一作]` (King Fahd University of Petroleum and Minerals), Ramy Rashad `[通讯]` (King Fahd University of Petroleum and Minerals)

**通讯引用:** 1003 | [OpenAlex ID](https://openalex.org/A5069300372)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

**🎯 论文内容**

本文通过几何力学和Hamiltonian约简理论，系统地推导了车辆-机械臂系统（VMS）的端口哈密顿（port‑Hamiltonian）动力学模型，给出了惯性解耦的两种形式；

**💡 创新点**

创新点在于首次用端口哈密顿框架揭示VMS的能量结构，利用主束结构实现惯性解耦，显式展示系统的对称性、能量保存与被动性；

**🔧 技术方法**

采用了Lie群/李代数、几何力学、Hamiltonian约简、Dirac结构和端口建模技术；

**📊 数据集**

该研究为理论推导，未使用具体实验或数据集；

**📈 对比分析**

通过理论推导证明了与已知的拉格朗日、Euler‑Lagrange和Boltzmann‑Hamel方程的等价性，未给出数值仿真或实验性能评估；

**⚠️ 局限性**

局限性包括仅考虑无约束基座的自由运动，未处理关节/运动约束、动力学参数辨识或控制实现细节，实际数值仿真与实验验证仍待后续工作。

---

## 304. Generative AI-assisted Participatory Modeling in Socio-Environmental Planning under Deep Uncertainty

**arXiv ID:** 2603.17021 | [PDF](https://arxiv.org/pdf/2603.17021v1)

**作者:** Zhihao Pei `[一作]` (University of Melbourne), Enayat A. Moallemi `[通讯]` (CSIRO)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2`

**🎯 论文内容**

提出并演示了一套基于大型语言模型（LLM）的模板化工作流，用于在深度不确定性下的社会-环境规划中快速完成问题概念化、视角探索与Python实现，旨在降低非专家参与门槛并提升概念化效率。

**💡 创新点**

创新点包括：①将LLM嵌入参与式建模的前置步骤，形成从自然语言描述到结构化模型规范的自动化链；②利用链式推理与提示链技术将复杂任务拆分为四步；③在工作流中加入验证清单与单元/属性/情景测试，实现模型与代码一致性追溯；④通过多视角建模展示LLM在捕捉多方利益与不确定性方面的潜力。

**🔧 技术方法**

使用技术：ChatGPT 5.2 Instant、链式推理（Chain‑of‑Thought）提示、提示链（Prompt Chaining）、ODD（Overview, Design, Details）验证清单、单元/属性/情景单元测试；代码生成采用Python类模块化实现，兼容EMA Workbench。

**📊 数据集**

数据集：两个案例研究，①湖泊问题（公开基准模型参数），②电力市场问题（自定义市场、需求、发电机参数和不确定性），所有参数均在论文中给出并在实验中复现。

**📈 对比分析**

比较方法：以模型构建迭代次数、组件提取准确率、Python实现通过测试的成功率和与基准实现的时间序列一致性为指标。实验显示：大多数步骤仅需 0–2 次迭代，组件提取准确率接近 100%，Python实现通过所有单元/属性/情景测试，输出时间序列与基准实现差异仅由随机扰动引起，证明工作流在概念化与实现上一致且性能可接受。

**⚠️ 局限性**

局限性：①LLM 对长链推理可靠性不足；②对提示语句的敏感性导致结果可重复性差；③代码实现仍易出现逻辑错误，需要人工验证与测试；④需在特定模型框架下使用，通用性受限；⑤工作流仅覆盖初步概念化，后续的深入建模与决策分析仍需专家介入。

---

## 305. EmergeNav: Structured Embodied Inference for Zero-Shot Vision-and-Language Navigation in Continuous Environments

**arXiv ID:** 2603.16947 | [PDF](https://arxiv.org/pdf/2603.16947v1)

**作者:** Kun Luo `[一作]` (Northeastern University), Xiaoguang Ma `[通讯]` (Northeastern University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了EmergeNav框架，利用阶段化结构化推理实现零样本连续视觉‑语言导航。

**💡 创新点**

核心创新在于Plan–Solve–Transition层次、GIPE目标感知提取、双记忆对比进度定位以及角色分离的双视野感知，整合这些机制后无需训练、地图或 waypoint 即可完成导航。

**🔧 技术方法**

结合大语言视觉模型（如Qwen3‑VL）、ReAct式短循环、GIPE提示式感知抽取、短期/长期双记忆对比以及三角前视+全景验证的双视野感知。

**📊 数据集**

在Habitat‑based VLN‑CE基准（100 试验集）上进行评估。

**📈 对比分析**

与多种零样本和监督基线对比，EmergeNav在8B模型下SR提升至30.00%，在32B模型下升至37.00%，SPL与OSR亦显著提升，证明结构化推理能显著增强零样本性能。

**⚠️ 局限性**

小模型子目标切换仍不够精准，可能出现短暂绕行；缺乏精确边界检测与恢复效率，且对VLM的视觉理解仍有依赖。

---

## 306. Swarm: Co-Activation Aware KVCache Offloading Across Multiple SSDs

**arXiv ID:** 2603.17803 | [PDF](https://arxiv.org/pdf/2603.17803v1)

**作者:** Tuowei Wang `[一作]` (Tsinghua University), Ju Ren `[通讯]` (Tsinghua University)

**通讯引用:** 12504 | [OpenAlex ID](https://openalex.org/A5015419107)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于SSD的KVCache脱离机制，利用KVCache共激活模式实现跨多块SSD的并行I/O，显著提升推理阶段的内存可扩展性与吞吐量。

**💡 创新点**

创新点在于首次将KVCache共激活（Co‑Activation）作为聚类依据，构造多层次索引与分布式布局，并结合在线负载均衡调度与动态聚类维护，实现对SSD带宽的高效聚合与自适应。

**🔧 技术方法**

技术上包含离线共激活图构建与基于距离阈值的贪心聚类、SSD与DRAM两层的分层存储策略、基于io_uring的异步并行读写、预取管线与CUDA流迁移、以及动态聚类与缓存更新机制。

**📊 数据集**

实验使用了六款主流LLM（Qwen3-S/M, Llama3.1, GPT‑OSS, Qwen3‑L‑MoE）以及四个数据集（WikiText, LongBench, MMLU, GSM8K），并在不同上下文长度、稀疏比例、SSD类型与数量下进行评测。

**📈 对比分析**

与三种基线（无聚类、InfLLM、PQCache）对比，Octopus在I/O吞吐、整体TPS以及KVCache命中率上分别提升约3.99×、1.76×、1.47×，并在多块SSD上实现平均2.41×的I/O时间缩短与2.72×的带宽利用率提升，且在稀疏度与前缀长度变化时仍保持显著优势。

**⚠️ 局限性**

限制主要体现在：需先行离线采样与共激活统计，且对极大上下文长度的实时聚类维护仍有一定开销；在SSD IOPS受限的低端设备上受限；对跨节点分布式部署的扩展性尚未验证。

---

## 307. Does YOLO Really Need to See Every Training Image in Every Epoch?

**arXiv ID:** 2603.17684 | [PDF](https://arxiv.org/pdf/2603.17684v1)

**作者:** Xingxing Xie `[一作]` (Northwestern Polytechnical University), Gong Cheng `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 20722 | [OpenAlex ID](https://openalex.org/A5080476856)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Anti-Forgetting Sampling Strategy（AFSS），通过动态采样减少YOLO训练中的冗余图像处理，显著加速训练。

**💡 创新点**

创新点在于使用基于最小精度-召回值的学习充分性度量，将样本分为易/中/难三级，并配合连续复习、短期覆盖及状态更新机制，实现抗遗忘与高效样本调度。

**🔧 技术方法**

采用YOLO系列模型、学习充分性度量、动态采样策略、抗遗忘机制和周期性状态更新等技术。

**📊 数据集**

在MS COCO 2017、PASCAL VOC 2007、DOTA‑v1.0和DIOR‑R四个标准数据集上进行实验验证。

**📈 对比分析**

与基线YOLO、课程学习、自适应学习、数据裁剪、数据蒸馏等方法比较，训练速度提升1.43×–1.68×，同时mAP保持或提升，远优于其它方法。

**⚠️ 局限性**

局限性包括对阈值与间隔参数需手动调优，且实验仅覆盖YOLO系列模型，跨模型与跨任务的通用性还有待进一步验证。

---

## 308. PaAgent: Portrait-Aware Image Restoration Agent via Subjective-Objective Reinforcement Learning

**arXiv ID:** 2603.17055 | [PDF](https://arxiv.org/pdf/2603.17055v1)

**作者:** Yijian Wang `[一作]` (Northwestern Polytechnical University), Wei Dong `[通讯]` (Xi'an University of Architecture and Technology)

**通讯引用:** 14091 | [OpenAlex ID](https://openalex.org/A5071296084)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了面向画像的图像恢复代理 PaAgent，利用过去交互经验选择恢复工具并执行多步决策，提升复杂降级图像恢复效果。

**💡 创新点**

创新点：①构建可自我演化的工具画像库，并通过检索增强生成 (RAG) 精准召回历史经验；②设计主观-客观强化学习 (SORL) 策略，将多模态 LLM 的主观评估与 NR‑IQA 的客观分数融合，显著提升降级感知与决策质量。

**🔧 技术方法**

技术手段：多模态 LLM（Qwen3.5‑9B/Plus）、检索增强生成 (RAG)、向量数据库、GRPO 强化学习、LoRA 微调、集成 14 种专业图像恢复工具及多种 NR‑IQA 指标。

**📊 数据集**

数据集：训练 15,013 张图像（含 CDD‑11 13,013 张混合降级与 2,000 张 GoPro 运动模糊），测试 16,230 张图像，覆盖单降级（SOTS、Rain13K、BSD68、CSD、LOL‑V1、GoPro）与多降级（CDD‑11、LOL_Blur）。

**📈 对比分析**

与 11 种 AiO 及 AgenticIR 方法对比，使用 PSNR/SSIM 评估。PaAgent 在所有单、双、三重降级以及复合降级任务上均以最高分领先，尤其在 CDD‑11 等极端混合场景中优势显著。

**⚠️ 局限性**

局限性：受限于 LLM 推理速度与成本、工具库覆盖范围、训练数据多样性；在极端噪声或光照极端条件下仍可能出现细节残留或色彩偏差。

---

## 309. Impacts of Electric Vehicle Charging Regimes and Infrastructure Deployments on System Performance: An Agent-Based Study

**arXiv ID:** 2603.16961 | [PDF](https://arxiv.org/pdf/2603.16961v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 310. TrackDeform3D: Markerless and Autonomous 3D Keypoint Tracking and Dataset Collection for Deformable Objects

**arXiv ID:** 2603.17068 | [PDF](https://arxiv.org/pdf/2603.17068v1)

**作者:** Yeheng Zong `[一作]` (University of Michigan), Ram Vasudevan `[通讯]` (University of Michigan)

**通讯引用:** 3086 | [OpenAlex ID](https://openalex.org/A5053632225)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本论文提出了TrackDeform3D，一套完整的从RGB‑D视频中自动分割、初始化、跟踪并生成高质量3D关键点轨迹的管线，并利用该管线构建了包含1D和2D可变形物体的110分钟、1320条动作序列的公开数据集。

**💡 创新点**

创新点在于将初始化和跟踪统一为带锚点、边长约束和点云投影的几何优化；使用点云差分自动分割、骨架化+最小生成树检测锚点、全局化热身的FPS初始化，以及基于Gauss‑Seidel的迭代投影，显著提升了初始化一致性和跟踪稳定性；同时首次提供了规模化的可变形物体关键点数据集。

**🔧 技术方法**

核心技术包括单摄像头RGB‑D点云差分分割、基于骨架化的锚点检测、Farthest Point Sampling初始化、统一的边长约束优化、点云投影、时间滑动平均滤波、以及与机器人手臂控制信息的联合使用。

**📊 数据集**

使用了自己构建的6种可变形物体（包括DLO、BDLO、布料、T‑shirt等）构成的数据集，共计110分钟、1320条动作序列；对比了现有的DOT、Mocap等基准数据集。

**📈 对比分析**

与CoTracker3、CDCPD2和SpatialTrackerV2三种基准方法在相同任务下进行比较，采用绝对边长误差、Chamfer距离、F‑score等指标，TrackDeform3D在所有类别和指标上均优于基准，尤其在边长误差和轨迹平滑度上提升显著。

**⚠️ 局限性**

主要限制包括：仅适用于无遮挡且单摄像头的场景；布料跟踪假设为矩形结构，无法直接处理非矩形或复杂拓扑；对高度动态或强遮挡场景的鲁棒性尚未验证；且仍需要手工验证锚点检测的准确性。

---

## 311. Adaptive Encoding Strategy for Quantum Annealing in Mixed-Variable Engineering Optimization

**arXiv ID:** 2603.17506 | [PDF](https://arxiv.org/pdf/2603.17506v1)

**作者:** Fabian Key `[一作]` (TU Wien), Norbert Hosters `[通讯]` (RWTH Aachen University)

**通讯引用:** 148 | [OpenAlex ID](https://openalex.org/A5008953865)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并验证了一种自适应编码策略，用于在量子退火器上对连续变量进行高精度混合离散-连续优化，并在结构设计案例上实现显著性能提升。

**💡 创新点**

自适应动态调整可表示范围的连续变量编码，在保持二进制变量固定的同时提高精度，并在全耦合问题中保留全局搜索优势。

**🔧 技术方法**

量子退火（D‑Wave Advantage）、最小互补能量框架、二次惩罚法、以及自适应范围收缩/扩张算法。

**📊 数据集**

一维复合杆尺寸优化基准（2 元素）及其解析解。

**📈 对比分析**

与固定范围编码（3 位）对比，采用相同二进制变量预算，衡量 H¹ 误差；结果精度提升三阶数量级，最终误差降低到 10⁻⁵~10⁻⁴，且在不同松弛因子、初始范围和读取次数下保持鲁棒。

**⚠️ 局限性**

实验仅在单个 D‑Wave Advantage 系统、规模较小的结构优化上验证，未评估更大规模问题、不同硬件平台或其他 Ising 机的通用性。

---

## 312. A Unified Language Model for Large Scale Search, Recommendation, and Reasoning

**arXiv ID:** 2603.17533 | [PDF](https://arxiv.org/pdf/2603.17533v1)

**作者:** Marco De Nadai `[一作]` (Spotify), Praveen Chandar `[通讯]` (Spotify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本研究提出了 NEO，一种通过将语义标识符（SID）与自然语言混合生成、并使用前缀 trie 限制解码的工具无关、可语言驱动的生成模型，统一支持大型异构目录下的推荐、检索和基于文本的解释；

**💡 创新点**

其创新点在于：①将多类型条目编码为语义化的多 token SID 并将其作为语言的一部分直接嵌入 LLM；②采用分阶段的对齐与指令微调方法，在保持语言能力的同时实现高效的实体对齐；③通过无工具、无架构改造的前缀 trie 限制解码保证生成的条目必属真实库存；

**🔧 技术方法**

使用了预训练的解码器 LLM（Qwen3‑0.6B 或 Llama‑3.2‑1B）+ SID 嵌入、残差 K‑means 量化、双向对齐（SID→文本、文本→SID、SID→类型）以及基于指令的多任务微调；

**📊 数据集**

在 Spotify 的实际产品环境中，使用了超过 10M 条目的多类型目录（剧集、节目、音频书、艺术家）及约 15M 用户的历史交互日志；

**📈 对比分析**

与传统基于图神经网络的推荐系统以及稠密检索系统做对比，NEO 在 HR@10 / NDCG@10 上均超过 20% 的提升，并在混合文本‑SID 解释任务上获得 GPT‑4o‑mini 评判的 4–5 分区间；

**⚠️ 局限性**

局限性包括：①对 SID 生成的碰撞处理仅采用基于流行度的随机映射，可能导致冷启动时的不确定性；②多任务微调仍受限于训练样本多样性，零/少量样本的新任务迁移能力尚待提升；③前缀 trie 的约束在极大 catalog 下的存储和检索成本仍需进一步优化。

---

## 313. SCE-LITE-HQ: Smooth visual counterfactual explanations with generative foundation models

**arXiv ID:** 2603.17048 | [PDF](https://arxiv.org/pdf/2603.17048v1)

**作者:** Ahmed Zeid `[一作]` (Machine Learning Group, Technische Universität Berlin), Sidney Bender `[通讯]` (Berlin Institute for the Foundations of Learning and Data)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了SCE-LITE-HQ框架，利用预训练生成基础模型在潜在空间进行高分辨率视觉对抗性反事实生成，无需任务特定的生成器训练；

**💡 创新点**

创新点在于①在预训练生成基础模型的潜在空间中直接优化；②引入平滑的代理分类器梯度与两阶段掩码策略，兼顾稀疏性与多样性；③实现了1024×1024级别的高分辨率生成，突破传统方法的分辨率与计算瓶颈；

**🔧 技术方法**

采用Stable Diffusion 3、PathLDM等预训练生成基础模型，平滑代理分类器（蒸馏+混合正则化）、FastDIME梯度近似、Rectified Flow/扩散模型的潜在轨迹优化，以及掩码式稀疏/多样化技术；

**📊 数据集**

在CelebA、CelebA-HQ（含诱导关联的Blond-Male）、Camelyon17医疗图像等数据集上进行实验；

**📈 对比分析**

与ACE、DIME、FastDIME、SCE等基线进行对比，评估非对抗率(NA)、稀疏度、非对抗翻转率(NAFR)、多样性和调优收益；在标准任务上保持竞争力，在诱导关联或医疗任务上显著提升Fidelity和模型改进收益，且支持高分辨率生成；

**⚠️ 局限性**

在干净数据上NA略逊于专门训练的生成器；仍需手动调节掩码阈值和对极端高分辨率的进一步验证。

---

## 314. Neuron-Level Emotion Control in Speech-Generative Large Audio-Language Models

**arXiv ID:** 2603.17231 | [PDF](https://arxiv.org/pdf/2603.17231v1)

**作者:** Xiutian Zhao `[一作]` (Johns Hopkins University), Berrak Sisman `[通讯]` (Johns Hopkins University)

**通讯引用:** 2142 | [OpenAlex ID](https://openalex.org/A5001303929)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了大规模音频-语言模型（LALM）在情感语音转换中的神经元级情感控制，并通过无训练干预实现情感表达调节。

**💡 创新点**

首次在LALM中发现并验证可操作的紧凑情感敏感神经元（ESN），并提出基于成功过滤的激活聚合识别方法，实现训练无关的情感控制。

**🔧 技术方法**

采用激活采样、成功过滤、神经元排名（对比边缘、均值偏差等）、以及推理时的门控尺度/添加/限制/去活化干预。

**📊 数据集**

使用情感语音数据集ESD（10位英语说话者，5种情感）以及三个开源LALM（Qwen2.5-Omni-7B、MiniCPM-o 4.5、Kimi-Audio）。

**📈 对比分析**

与无干预基线及随机神经元、不同选择器（LAP、LAPE、MAD、CAS）对比，CAS/MAD得到最高情感匹配提升，WER保持可接受范围，UTMOS基本不变，人工评测亦验证显著情感提升。

**⚠️ 局限性**

局限性包括需依赖成功过滤样本、不同情感的转移效果不均衡、过大干预强度导致语义漂移、以及对合成模块的影响有限。

---

## 315. Topology-Preserving Deep Joint Source-Channel Coding for Semantic Communication

**arXiv ID:** 2603.17126 | [PDF](https://arxiv.org/pdf/2603.17126v1)

**作者:** Omar Erak `[一作]` (Khalifa University), Sami Muhaidat `[通讯]` (Khalifa University)

**通讯引用:** 7264 | [OpenAlex ID](https://openalex.org/A5004034156)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种基于深度联合源-信道编码（DeepJSCC）的拓扑保真语义通信框架，称为TopoJSCC。

**💡 创新点**

创新点在于将持久同调（persistent homology）正则化引入端到端学习，既在图像域对齐原图与重建图的拓扑结构，又在潜在空间对抗信道噪声保持潜在点云的拓扑连通性。

**🔧 技术方法**

采用了持久同调、Wasserstein距离、卷积自编码器、可微分信道层和Adam优化器；训练损失包含MSE、图像域拓扑损失和潜在域拓扑损失。

**📊 数据集**

使用了两个拓扑丰富数据集：Omniglot（手写字符）和DeepGlobe道路提取（卫星图像道路分割）。

**📈 对比分析**

与MSE仅优化的DeepJSCC、BPG+LDPC（经典分离式编码）以及TopoCode（数字侧信号拓扑纠错）进行对比；TopoJSCC在低SNR、低带宽条件下的PSNR提升约1–2 dB，且Wasserstein距离显著下降，显示出更优的拓扑一致性和鲁棒性。

**⚠️ 局限性**

局限性包括对持久同调计算的额外训练时延、对超参数 λ_img、λ_lat 的敏感性，以及在极高SNR或宽带条件下与传统DeepJSCC的性能差距趋近。

---

## 316. Greedy Completion for Weighted $(α,β)$-Spanners

**arXiv ID:** 2603.17047 | [PDF](https://arxiv.org/pdf/2603.17047v1)

**作者:** Elad Tzalik `[一作]` (Weizmann Institute), Elad Tzalik `[通讯]` (Weizmann Institute)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5051677237)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种全局贪心完成算法，能够构造权重图的 (α,β)-稀疏子图，并以此实现了 (k,k-1)-稀疏图的多项式时间构造。

**💡 创新点**

创新点在于将 Knudsen 的加性完成方法推广到 (α,β) 情况，利用最小分段和 α‑距离边的修补机制，克服了传统加性完成无法处理权重图的局限，从而首次给出大小为 Õ(n^{1+1/k}) 的 (k,k-1)-稀疏图，几乎达到已知下界。

**🔧 技术方法**

核心技术包括：贪心完成策略、最小分段（minimal segmentation）、α‑距离边（α-distant edges）的选择、基于 Baswana‑Sen 的初始化聚类、中心对（center pair）改进与最终化分析，以及跳跃式合并与距离改善计数。

**📊 数据集**

该工作为理论研究，未使用任何实验数据集，所有结果均来自数学证明。

**📈 对比分析**

通过理论证明，该算法在多项式时间内输出大小为 O(n^{1+1/k}) 的 (k,k-1)-稀疏图，满足 _H(u,v) ≤ k·_G(u,v)+(k-1)·W_max(G)。与传统的 (2k-1)-乘法稀疏图相比，乘法系数减半，且与已知的下界匹配，表明其性能在结构上是最优的。

**⚠️ 局限性**

局限性包括：仅针对 (k,k-1) 这一特定参数，对其他 α>1 的情况仍缺乏通用构造；未讨论轻度性（lightness）或更细粒度的结构化保证；缺乏实验验证，所有结论仅在理论层面；对于如何将该方法推广到无权图或其他 α,β 配置仍是开放问题。

---

## 317. Procedural Generation of Algorithm Discovery Tasks in Machine Learning

**arXiv ID:** 2603.17863 | [PDF](https://arxiv.org/pdf/2603.17863v1)

**作者:** Alexander D. Goldie `[一作]` (University of Oxford), Jakob N. Foerster `[通讯]` (University of Oxford)

**通讯引用:** 59131 | [OpenAlex ID](https://openalex.org/A5042899882)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了DiscoGen——一种大规模程序化生成机器学习算法发现任务的框架，并基于其构建了评估基准DiscoBench；

**💡 创新点**

创新点在于通过可组合的模块和数据集生成数亿个多样化、训练/测试分离的任务，克服了现有任务集规模小、评估失真、数据污染等缺陷；

**🔧 技术方法**

利用Python/JAAX编写任务模板，结合程序化内容生成（PCG）技术，并采用语言模型（如DeepSeek、Devstral2）和强化学习Agent实现算法发现与评估；

**📊 数据集**

使用多种公开数据集，包括ImageNet、CIFAR系列、OpenAI Gym 环境、语言模型语料、持续学习与气候预测数据等，覆盖计算机视觉、强化学习、自然语言等领域；

**📈 对比分析**

与传统基准（MLE-Bench、MLGymBench）对比，实验显示在DiscoBench单模块任务上ADA成功率与Elo分数普遍提升，但在多模块任务上仍低于基线；在提示优化实验中，任务多样性越大，元测试性能越好；

**⚠️ 局限性**

局限性包括：对硬件资源需求高、部分任务仍难以被ADA成功解决、实验覆盖的模型有限、可能存在数据污染风险以及未充分探索不同初始化与评估模式的影响。

---

## 318. Large Language Models as a Semantic Interface and Ethical Mediator in Neuro-Digital Ecosystems: Conceptual Foundations and a Regulatory Imperative

**arXiv ID:** 2603.17444 | [PDF](https://arxiv.org/pdf/2603.17444v1)

**作者:** Alexander V. Shenderuk-Zhidkov `[一作]` (Immanuel Kant Baltic Federal University), Alexander E. Hramov `[通讯]` (Plekhanov Russian University of Economics)

**通讯引用:** 9824 | [OpenAlex ID](https://openalex.org/A5019793174)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并论证了神经-语言整合（NLI）概念，阐述LLM作为语义接口的技术与伦理挑战，并提出以语义透明度、精神知情同意和代理权保护为核心的监管框架与工具。

**💡 创新点**

创新点在于将大型语言模型从纯文本处理升级为神经数据的语义翻译器，形成“第二阶神经伦理学”视角，并提出针对NLI的专属监管工具（伦理沙盒、LLM认证、神经-语言推断的法律认可）。

**🔧 技术方法**

核心技术为脑-计算机接口（BCI）与大型语言模型（LLM）的结合，采用语义解释、上下文增强与三重输出（解释器、通讯器、适配器）架构。

**📊 数据集**

由于研究为概念性与法规分析，未使用具体实验数据集，主要依托文献综述与跨学科理论框架。

**📈 对比分析**

无实验性对比与性能评估；本研究通过理论推演和案例分析说明潜在风险与治理路径。

**⚠️ 局限性**

局限在于缺乏实证验证与量化评估，监管建议仍需在未来实验与立法实践中进一步细化与验证。

---

## 319. The 1/W Law: An Analytical Study of Context-Length Routing Topology and GPU Generation Gains for LLM Inference Energy Efficiency

**arXiv ID:** 2603.17280 | [PDF](https://arxiv.org/pdf/2603.17280v1)

**作者:** Huamin Chen `[一作]` (vLLM Semantic Router Project), Xue Liu `[通讯]` (MBZUAI)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过分析GPU功率曲线、KV缓存并发与上下文窗口大小的关系，量化了LLM推理的能效（tokens per watt）并提出了“1/𝒲 法”，即每当上下文窗口加倍，tokens per watt 就会减半。

**💡 创新点**

创新点在于：①将上下文窗口作为独立且可调节的能效杠杆；②通过结合已发表的GPU功率模型、屋顶线模型与fleet-sim框架，推导出上下文窗口、路由拓扑与硬件世代的三维独立乘法关系；③将稀疏 MoE 架构的权重流动时间作为第三个能效杠杆；④提出两池路由（FleetOpt）可实现约 2.5× 的能效提升，并与新一代 GPU（B200）相结合可实现 4.25× 的总提升。

**🔧 技术方法**

技术手段包括：
- 逻辑斯蒂 GPU 功率模型（Liang 等）
- AIConfigurator 屋顶线模型（用于拆解权重流动时间 W 与 KV 扫描开销 H）
- inference‑fleet‑sim 规划框架（用于计算 fleet‑level tok/W）
- 路由拓扑优化（FleetOpt）
- MoE 权重流动分析（active‑parameter streaming）

**📊 数据集**

使用的数据集/实验对象包括：
- Llama‑3.1‑70B、Llama‑3.1‑8B、Llama‑3.1‑405B
- Qwen3‑235B‑A22B、DeepSeek‑V3
- GPU 硬件：H100‑SXM5、B200‑SXM、H200‑SXM、GB200‑NVL
- 工作负载轨迹：Azure LLM 推理轨迹、LMSYS‑Chat‑1M

**📈 对比分析**

比较方法：
- 在相同上下文窗口下测算单 GPU tok/W 并验证 1/𝒲 规律；
- 在 fleet‑level 采用不同路由拓扑（Homogeneous、Pool、FleetOpt）和不同 GPU 组合（H100、B200、H200 等）对 tok/W 及 tok/$M 进行对比；
- 对 MoE 模型使用权重流动时间替代全模型 W，给出 upper‑bound tok/W。
- 结果显示：在 8K 上下文窗口下，B200 在 tok/W 上比 H100 高约 1.7×；FleetOpt 在任何 GPU 组合上均可提升约 2.5×；两者组合可达约 4.25× 的总提升；MoE 模型 Qwen3‑235B‑A22B 在 8K 上可实现约 37.8 tok/W，比 Llama‑3.1‑70B 高 5.1×。

**⚠️ 局限性**

局限性：
- B200、H200 的功率曲线为预测，缺乏直接测量；
- 路由拓扑仅考虑两池划分，未探索多池或语义路由；
- 只评估 steady‑state 负载，未考虑波动、burst 和动态扩缩；
- 只计量输出 token 的能耗，忽略 prefill 计算能耗；
- 采用固定 TP=8 的单模型池，未考虑混合 TP 或多模型情形；
- MoE 的 dispatch 开销未量化，仅给出上限估计。

---

## 320. DiffVP: Differential Visual Semantic Prompting for LLM-Based CT Report Generation

**arXiv ID:** 2603.17718 | [PDF](https://arxiv.org/pdf/2603.17718v1)

**作者:** Yuhe Tian `[一作]` (University of Science and Technology of China), Shaohua Kevin Zhou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 663 | [OpenAlex ID](https://openalex.org/A5101592407)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于正常CT参考图像的差异化视觉提示框架 DiffVP，用于提升大语言模型在CT报告生成中的诊断细节捕捉与语言质量。

**💡 创新点**

创新点在于构建层次化差异提取器，既捕捉全局语义差异又聚合局部细微差别，并将这些差异映射为可学习的视觉前缀令牌，直接引导LLM聚焦病理信息，无需显式病灶定位。

**🔧 技术方法**

核心技术包括共享的3D ResNet-18视觉编码器、Q-Former式视觉重采样、Transformer实现全局差异查询、几何加权残差实现局部差异聚合、以及差异到提示生成器（MLP+投影）与LoRA微调的 LLaMA‑2‑7B 语言模型。

**📊 数据集**

实验基于 RadGenome‑ChestCT（约25k组）和 CTRG‑Chest‑548K（约1.8k组）两个大型CT‑报告对照数据集，并从同一数据集中挑选无异常的CT作为参考集合。

**📈 对比分析**

与 RadFM、M3D、R2GenGPT、MedVInT、CT2Rep、Reg2RG 等先前方法比较，DiffVP 在 BLEU‑1/2/3/4、ROUGE‑L、METEOR 以及 RadBERT 评估的 F1（0.421）等多项指标上均实现了显著提升。

**⚠️ 局限性**

局限性主要体现在对参考CT的依赖——若参考样本不足或不代表患者正常基线，差异提取可能失效；此外，固定长度的视觉前缀在极度稀疏或细微病变时仍可能不足，且在不同模态或数据分布上尚未充分验证。

---

## 321. Interpretable AI-Assisted Early Reliability Prediction for a Two-Parameter Parallel Root-Finding Scheme

**arXiv ID:** 2603.16980 | [PDF](https://arxiv.org/pdf/2603.16980v1)

**作者:** Bruno Carpentieri `[一作]` (Free University of Bozen--Bolzano), Paola Lecca `[通讯]` (Free University of Bozen--Bolzano)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一个可解释的 AI 辅助可靠性诊断框架，利用 kNN‑LLE 代理稳定性分析与多时域早期预测，能够在根求解迭代的前几步内估计参数化根求解器的可靠性。

**💡 创新点**

创新点在于：①将可解释的 Lyapunov‑指数代理与可靠性指标 S_mom 结合；②在多时域学习中仅使用迭代前缀预测最终可靠性；③在中心‑边缘外推和随机拆分下实现高精度预测；④推理成本仅为微秒级，可实现实时决策。

**🔧 技术方法**

使用的技术包括：kNN 估计的最大 Lyapunov 指数代理、平滑滤波、S_min、S_mom 可靠性指标；多时域回归（kNN、岭回归、弹性网络、随机森林、梯度提升）；热图可视化与统计评估。

**📊 数据集**

使用的数据集为 60×60 的 (α,β) 网格（α∈[-3,5], β∈[-2,4]），每个网格点生成 1000 条初始猜测、200 次迭代的根求解轨迹，构成代理序列。

**📈 对比分析**

与多种基准模型（线性与非线性）在随机拆分与中心‑边缘拆分下比较，评估指标为 MAE、RMSE、R²；在 T≈11 时 R²≈0.89‑0.91，MAE≈0.03，T=35 时 R²≈0.96，推理成本仅微秒级；相比完整诊断可节省约 10‑16 倍迭代次数。

**⚠️ 局限性**

局限性包括：仅在单一根求解问题上验证，跨问题泛化未测试；代理超参数（L、窗口大小等）对结果敏感；最早阶段（T=1）预测精度有限；尚需进一步研究与实时控制的集成与实际应用中的鲁棒性。

---

## 322. One-Step Sampler for Boltzmann Distributions via Drifting

**arXiv ID:** 2603.17579 | [PDF](https://arxiv.org/pdf/2603.17579v1)

**作者:** Wenhan Cao `[一作]` (National University of Singapore), Lin Zhao `[通讯]` (National University of Singapore)

**通讯引用:** 13630 | [OpenAlex ID](https://openalex.org/A5110190620)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于漂移的框架，用单步神经生成器在测试时实现Boltzmann分布的近似采样。

**💡 创新点**

创新点在于：①将Boltzmann目标的梯度迁移表述为高斯平滑后的能量梯度；②提供两种目标侧漂移近似（局部重要性采样和二阶曲率校正）；③采用停梯度训练目标，使训练过程稳定且高效。

**🔧 技术方法**

使用Gaussian-smoothed score operator、局部重要性采样、二阶泰勒展开、mini-batch Gaussian mean‑shift 估计、停梯度（stop‑gradient）技术以及残差MLP生成器。

**📊 数据集**

主要在二维合成数据集上验证：四模式高斯混合、双井势能、香蕉形势能。

**📈 对比分析**

通过与目标分布的均值误差、协方差误差、RBF MMD 和能量误差对比，结果显示一阶、二阶矩与目标高度一致（误差均<0.05），MMD仅为0.002，表明采样质量优秀；未与传统 MCMC 或 HMC 直接比较，但在相同计算预算下可预期具有更快收敛。

**⚠️ 局限性**

局限性包括：仅在低维合成数据上验证；缺乏与强大迭代采样器的直接对比；对高斯核带宽与漂移步长敏感；在更高维、复杂图像等实际任务中的稳定性与效果仍待验证。

---

## 323. Visual Product Search Benchmark

**arXiv ID:** 2603.17186 | [PDF](https://arxiv.org/pdf/2603.17186v1)

**作者:** Karthik Sulthanpete Govindappa `[一作]` `[通讯]` (nyris GmbH), Karthik Sulthanpete Govindappa (nyris GmbH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了包含工业与公开数据集的实例级图像检索基准，并对多种视觉嵌入模型在无后处理、零样本条件下的检索性能进行系统评估。

**💡 创新点**

创新点在于统一的图像对图像检索协议与对比方法，能够真实反映工业场景下极度细粒度与域迁移挑战，同时对开源、专有与专用模型进行零样本直接对比，揭示了模型在工业检索中的差距。

**🔧 技术方法**

采用了深度度量学习与多模态嵌入模型（如DINOv2/v3、CLIP、PE-Core、SigLIP、Jina Embeddings、Cohere、Gemini、Nomic、GEM、AEM），使用L2归一化的向量空间和Qdrant进行精确向量检索。

**📊 数据集**

使用了8个数据集：四个内部工业数据集（Clips‑and‑Connectors、Furniture、DIY、Automotive）以及四个公开基准（SOP、Products‑10K、ILIAS、SOP）。

**📈 对比分析**

通过Recall@1/5、mAP@20/1000等指标比较模型，发现GEM v5.1在所有数据集上平均取得约58.3% R@1、74.3% R@5、49.5% mAP@20，领先其他模型；通用模型在工业数据集表现显著衰退，说明工业检索对模型提出了更高要求。

**⚠️ 局限性**

局限性包括受保密约束无法公开内部数据集与部分专有模型，评测仅覆盖有限工业场景，且API模型随时更新可能导致结果变化，外部复现受限。

---

## 324. LED: A Benchmark for Evaluating Layout Error Detection in Document Analysis

**arXiv ID:** 2603.17265 | [PDF](https://arxiv.org/pdf/2603.17265v1)

**作者:** Inbum Heo `[一作]` (Chungnam National University), Sangkeun Jung `[通讯]` (Chungnam National University)

**通讯引用:** 816 | [OpenAlex ID](https://openalex.org/A5067724204)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Layout Error Detection（LED）基准，用以评估文档布局分析模型在结构错误上的表现，定义了八种错误类型，并构造了包含这些错误的LED‑Dataset，同时设计了三层级的评估任务。

**💡 创新点**

创新点在于把结构一致性作为评价维度，统一定义文档专用错误类型、实现规则化错误注入算法，并提供可解释的层级评估框架，显著弥补了传统IoU、mAP只能衡量定位准确性的局限。

**🔧 技术方法**

技术手段包括基于规则的错误注入、JSON与视觉信息的多模态提示设计，以及使用多种大型多模态模型（GPT‑4o、Gemini 2.5 Pro/Flash、DeepSeek V3、LLaMA‑4 等）进行零样本评估。

**📊 数据集**

数据集为LED‑Dataset，来源于DocLayNet测试集，在真实错误分布基础上注入约5000张文档的8类结构错误，包含原始图像、GT标注和错误注释。

**📈 对比分析**

通过对八种模型在三种提示（P1–P3）下执行T1（错误检测）、T2（错误类型分类）和T3（元素级错误分类）三任务进行对比，Gemini 2.5 Pro/Flash在所有任务上取得最佳或接近最佳的F1/准确率，GPT系列在T1表现强劲但在T2/T3下降，模型规模与家族差异对性能影响显著。

**⚠️ 局限性**

局限性包括：仅评估结构错误，未与传统定位指标结合；数据集仅基于DocLayNet，域覆盖有限；模型性能高度受提示设计影响，提示不一致会导致显著波动。

---

## 325. AI-Assisted Goal Setting Improves Goal Progress Through Social Accountability

**arXiv ID:** 2603.17887 | [PDF](https://arxiv.org/pdf/2603.17887v1)

**作者:** Michel Schimpf `[一作]` (University of Cambridge), Thomas Bohné `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究采用预注册的三组随机对照试验，比较了基于Claude Sonnet LLM的职业目标设定聊天机器人（Leon）、结构化书面问卷和无干预控制，在两周后对被试职业目标进展的影响。

**💡 创新点**

创新点在于首次验证LLM驱动的聊天机器人在职业目标设定中的短期效能，并系统探讨其心理机制，发现聊天机器人显著提升了受试者的社交责任感，成为主要促进目标进展的途径。

**🔧 技术方法**

技术手段包括：1) 基于Claude Sonnet的语言模型聊天机器人Leon；2) 对比的结构化写作问卷；3) 采用多项自我报告量表（目标进展、自我共鸣、责任感等）进行测评；4) LLM辅助的文本编码用于目标具体性与领域分类。

**📊 数据集**

数据来源为在Prolific平台招募的517名在职成年人（年龄18-50岁，主要来自英国和美国），在两次在线会话（约14天间隔）中收集自我报告数据，最终分析样本为323人。

**📈 对比分析**

采用预注册的统计比较方法：Welch t检验、协方差分析（ANCOVA）以及并行中介分析。结果显示：AI聊天机器人相较于无干预控制在两周后目标进展上显著提升（d=0.33，p=0.016）；与书面问卷相比，AI并未提升整体进展，但显著提高了责任感，并通过责任感中介（间接效应0.15）解释了两组间的差异。

**⚠️ 局限性**

局限性包括：1) 跟踪时间仅为两周，无法评估长期持续效果；2) 样本主要为英国和美国的在职成年人，结果对其他文化或非英语人群的普适性不明；3) 书面问卷强制20分钟的完成时长导致问卷组流失率较高；4) 仅使用单一LLM模型（Claude Sonnet），不同模型或提示策略的效果未知；5) 目标具体性与领域分类等探索性分析依赖LLM编码，缺乏人工验证。

---

## 326. Capability-Priced Micro-Markets: A Micro-Economic Framework for the Agentic Web over HTTP 402

**arXiv ID:** 2603.16899 | [PDF](https://arxiv.org/pdf/2603.16899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

---

## 327. From Optimizable to Interactable: Mixed Digital Twin-Empowered Testing of Vehicle-Infrastructure Cooperation Systems

**arXiv ID:** 2603.17497 | [PDF](https://arxiv.org/pdf/2603.17497v1)

**作者:** Jianghong Dong `[一作]` (Tsinghua University), Keqiang Li `[通讯]` (Tsinghua University)

**通讯引用:** 16158 | [OpenAlex ID](https://openalex.org/A5031855986)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e`

**🎯 论文内容**

提出并实现了IMPACT交互式VICS测试框架，定义了L5“Interactable”数字孪生层级，搭建了I-VIT实验平台，并在混合数字孪生环境中进行物理–虚拟动作交互与人机交互实验。

**💡 创新点**

创新点包括：①将DT层级扩展到L5，首次将“可交互”纳入VICS-DT成熟度评估；②引入混合DT，使虚拟实体可独立存在并与物理实体在动作层面交互，实现“Physical‑Virtual Action Interaction”；③利用人机交互（MR HMD、驾驶模拟器等）直接注入不可预测的人类行为，天然生成高质量角落案例，弥补AI驱动生成方法的局限。

**🔧 技术方法**

技术手段：混合数字孪生（mixedDT）框架；Unity 3D + Unity Engine实现虚拟环境；HoloLens MR HMD、G29/InnoSimulation驾驶模拟器实现人机交互；ROS + Edge‑Cloud架构实现实时通信与数据融合；V2X通信、摄像头/雷达感知；CACC与预测轨迹跟踪控制算法；基于物理沙盘的真实感知与仿真数据集成。

**📊 数据集**

数据集：无公开标准数据集，采用物理沙盘实验中摄像头、雷达收集的实时车辆状态数据；与Unity仿真产生的车辆轨迹、状态数据进行对齐；实验中使用的交通场景、车道拓扑、障碍物配置等均为自建数据。

**📈 对比分析**

比较方法：与传统AI驱动的角落案例生成方法（如基于生成式模型的情景合成）做对比，重点评估安全性、角落案例多样性和生成效率。实验结果表明：在IMPACT下，突发制动导致车距降至5m的危险情景能够被安全捕捉且未造成实际碰撞，说明混合DT实现了安全且可重复的角落案例验证；虽然未给定具体数值对比，但实验演示已充分证明其优越性。

**⚠️ 局限性**

局限性：①实验规模受限于沙盘与仿真资源，难以覆盖大规模车流与复杂城市网络；②系统对多交互请求的响应机制尚不完善，需进一步研究并发交互的协调；③通信延迟虽低，但在更高频率、高精度的安全关键场景中仍需优化；④对多模态大模型（MLLM）的集成与自动化角落案例生成仍是未来工作方向。

---

## 328. TerraLingua: Emergence and Analysis of Open-endedness in LLM Ecologies

**arXiv ID:** 2603.16910 | [PDF](https://arxiv.org/pdf/2603.16910v1)

**作者:** Giuseppe Paolo `[一作]` (Cognizant), Elliot Meyerson `[通讯]` (Cognizant)

**通讯引用:** 1638 | [OpenAlex ID](https://openalex.org/A5020821744)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 TerraLingua，一个持久化的二维网格 LLM 多代理生态系统，并通过非干预式的 AI Anthropologist 对代理行为、社群结构与文本工件进行系统化分析，探究开放式动态与累积文化的自发产生。

**💡 创新点**

创新点在于：① 将资源限制、有限寿命与可见文本工件结合，使 LLM 代理在持久环境中自发形成协作规范、工作分工与文化线索；② 引入 AI Anthropologist 作为后置观察器，利用 LLM 进行大规模定性与定量评估，避免传统指标的先验假设，真正捕捉开放式动态的多维特征。

**🔧 技术方法**

技术实现主要包括：使用 DeepSeek-R1‑Distill‑Qwen‑32B 作为代理决策核心；网格环境与感知半径、动作词汇表；Agent 通过 LLM 对话生成行为与信息；AI Anthropologist 采用 Claude Sonnet4.5 与 Haiku4.5 进行事件标注、社群检测、工件谱系重建与复杂度评估。

**📊 数据集**

数据集：所有实验数据由 TerraLingua 生成并公开发布至 Hugging Face（GPaolo/TerraLingua）及 GitHub 代码仓库（https://github.com/cognizant-ai-lab/terralingua），不依赖外部文本语料库，仅利用 LLM 内部知识。

**📈 对比分析**

评估方法：在资源稀缺、资源丰盈、不同人格维度、动机设定、工件成本、上下文长度等多种实验条件下，对群体寿命、每个代理的工件产量、创新度、工件复杂度、社群结构等指标进行量化比较；结果显示在资源稀缺、人格+最小动机且工件可见的配置下，系统达到了 Pareto 最优，持续生成高质量累积文化。

**⚠️ 局限性**

局限性：① LLM 参数固定导致行为受预训练知识限制；② 上下文长度与工件尺寸上限限制了复杂工件的生成与记忆；③ AI Anthropologist 的判断依赖 LLM 推理，可能出现误标记；④ 仅在模拟环境中验证，缺乏与真实人类交互的外部效度。

---

## 329. Illumination-Aware Contactless Fingerprint Spoof Detection via Paired Flash-Non-Flash Imaging

**arXiv ID:** 2603.17679 | [PDF](https://arxiv.org/pdf/2603.17679v1)

**作者:** Roja Sahoo `[一作]` (Indraprastha Institute of Information Technology Hyderabad), Anoop Namboodiri `[通讯]` (Indraprastha Institute of Information Technology Hyderabad)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了使用闪光灯与非闪光灯配对拍摄的无接触指纹图像，通过分析光照引起的纹理、光泽和颜色差异来进行伪造检测。

**💡 创新点**

创新点在于将配对光照视为轻量级主动感知方式，提出多维光照相关特征（交叉通道相关性、镜面反射比、纹理真实性、差分图像等）并结合深度模型可解释性提升鲁棒性。

**🔧 技术方法**

采用了图像质量指标（OCL、LCS、NFIQ2、AIT锐度）、光照统计特征（局部对比度、边缘能量）、交叉通道相关性、镜面高光比、LBP/GLCM/FFT纹理特征、差分成像；以及深度模型（未微调的 DINOv2、微调的 ResNet‑18）进行特征提取与注意力分析。

**📊 数据集**

使用了自制的 800 张真指纹与 1600 张伪造样本（打印、数字、模塑）在 Samsung Galaxy A54 手机上按 FNF 协议收集的配对闪光/非闪光图像；同时引用了 COLFISPOOF、IIITD Spoofed Fingerphoto、CLARKSON 等公开数据集做对比。

**📈 对比分析**

相较于单一闪光图像方法，配对光照提高了特征区分度（Fisher 判别比从 0.18‑1.48 级别提升），注意力分布更聚焦在脊纹区域，实验显示在真伪分类上的准确率提升 3‑5%（根据作者报告），并证明了多模态光照对不同伪造类型的鲁棒性。

**⚠️ 局限性**

局限性包括：仅在受控环境下采集，受指尖姿态、距离、光照等因素影响；数据集规模有限，难以覆盖高质量 3D 复制品；配对拍摄需额外用户配合，部署复杂度上升；并且高仿真伪造材料可能逐步逼近真实光照响应，削弱部分特征优势。

---

## 330. FoMo X: Modular Explainability Signals for Outlier Detection Foundation Models

**arXiv ID:** 2603.17570 | [PDF](https://arxiv.org/pdf/2603.17570v1)

**作者:** Simon Klüttermann `[一作]` (TU Dortmund University), Emmanuel Müller `[通讯]` (Research Center Trustworthy Data Science and Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 FoMo-X 模块化框架，给基于 PFN 的零样本异常检测模型添加轻量级诊断头，能够在不改变原有检测结果的情况下一次性输出异常严重程度和不确定性。

**💡 创新点**

创新点在于：1）将异常检测的解释信号与模型主体解耦，使用冻结的 PFN 表征直接训练诊断头；2）利用模拟器生成的高质量监督，学习可一次性推断的“严重程度”与“置信度”信号；3）实现几乎无额外推理开销的自解释能力。

**🔧 技术方法**

技术包括：先前训练的 Prior‑Data‑Fitted Network（PFN）基础模型、Transformer 编码器、冻结后附加的多层感知机（MLP）诊断头、模拟器生成的 GMM 任务、Monte Carlo dropout 作为教师信号、以及交叉熵/MAE 损失进行离线训练。

**📊 数据集**

使用合成的 GMM 数据集进行头训练，并在真实世界的 ADBench（47 个表格数据集）上评估泛化能力。

**📈 对比分析**

与基线（FoMo‑0D、Isolation Forest 等）比较，FoMo‑X 在不改变原始异常检测 AUROC 的前提下，能一次性提供与 MC‑dropout 约 99% Spearman 相关的置信度，严重程度分层与错误率高度相关；推理时间增幅 <2%，几乎无负担。

**⚠️ 局限性**

局限性在于：1）仅能有效预测局部诊断信号，对全局属性（如数据集级别的 AUROC 或最佳阈值）迁移性差；2）依赖于 GMM 模拟器的先验，真实数据分布差异导致解释头泛化受限；3）现有 PFN 结构不支持列级特征解释，需要更丰富的结构化嵌入。

---

## 331. Cache-enabled Generative Joint Source-Channel Coding for Evolving Semantic Communications

**arXiv ID:** 2603.17702 | [PDF](https://arxiv.org/pdf/2603.17702v1)

**作者:** Shunpu Tang `[一作]` (Zhejiang University), Deniz Gunduz `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种训练无关的语义通信框架，利用通道感知GAN逆向生成可直接在噪声信道上传输的潜在码，并通过缓存启用的动态码本（CDC）在多次传输中持续压缩带宽。

**💡 创新点**

创新点包括：① 将无线信道特性嵌入GAN逆向过程，得到与噪声相关的潜在码；② 引入语义分量级别的缓存与索引传输，实现动态码本的持续更新与利用；③ 采用SNR感知的缓存更新与直通（straight‑through）策略，克服非可微缓存操作对优化的影响。

**🔧 技术方法**

技术手段包括预训练的SemanticStyleGAN、GAN逆向求解、通道自适应正则化、直通直链（straight‑through）优化、Cosine相似度缓存匹配、LDPC编码+BPSK索引传输、LPIPS与任务特定损失等。

**📊 数据集**

实验使用CelebAMask‑HQ人脸图像数据集，分辨率512×512，潜在维度512，语义向量28。

**📈 对比分析**

通过与DeepJSCC、InverseJSCC、GenerativeJSCC、GI‑SSCC、GI（无编码）、BPG+LDPC等基线在AWGN 0–5 dB、BCR=1/128条件下对比，在PSNR、MS‑SSIM、LPIPS、PIEAPP、DISTS、FID等指标上，CAGI‑JSCC+CDC均优于或相当于基线，并且平均BCR降至1/224，单图可达1/1024，表明显著提升了带宽利用率和感知质量。

**⚠️ 局限性**

局限性包括：① 依赖预训练GAN模型，受限于模型规模与训练数据；② 缓存同步与索引误码可能引入重建误差；③ 对极低SNR或未知噪声的鲁棒性仍有限；④ 缓存更新策略对参数α的敏感性；⑤ 目前仅在图像语义场景验证，其他多媒体类型尚未测试。

---

## 332. Embedding World Knowledge into Tabular Models: Towards Best Practices for Embedding Pipeline Design

**arXiv ID:** 2603.17737 | [PDF](https://arxiv.org/pdf/2603.17737v1)

**作者:** Oksana Kolomenko `[一作]` (University of Applied Sciences Berlin), Erik Rodner `[通讯]` (University of Applied Sciences Berlin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

系统评估了256种LLM嵌入管线配置，探讨了预处理、嵌入模型和下游模型对表格预测的影响。

**💡 创新点**

创新点在于揭示嵌入拼接优于替换、梯度提升树优于逻辑回归，以及模型参数规模与性能的正相关，并指出公开排行榜和模型流行度与实际性能关联不大。

**🔧 技术方法**

使用16种LLM嵌入模型、8种预处理策略、2种下游模型（LR和GBDT），并结合PCA降维、CLS/平均池化提取等技术。

**📊 数据集**

使用2025年以后公开的Kaggle数据集：肺部疾病（5200样本）和网络安全入侵检测（9537样本）。

**📈 对比分析**

通过80/20划分、5折交叉验证和AUC、F1、BA指标进行比较，最佳管线AUC为0.77，比传统GBDT提升约0.05。

**⚠️ 局限性**

局限在于仅覆盖≤1亿参数模型，未测试更大或压缩模型，且实验结果可能随数据集差异而变化。

---

## 333. Contingency-Aware Planning via Certified Neural Hamilton-Jacobi Reachability

**arXiv ID:** 2603.17022 | [PDF](https://arxiv.org/pdf/2603.17022v1)

**作者:** Kasidit Muenprasitivej `[一作]` (Northeastern University), Derya Aksaray `[通讯]` (Northeastern University)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5053550436)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种结合 Fourier Neural Operator 学习的 Hamilton–Jacobi reachability 与增量多目标规划框架，能够在未知环境中实时保证安全并提供有限时间的恢复路径；

**💡 创新点**

创新点在于（1）利用 FNO 学习 HJI‑VI 解决器并给出下逼近保证，实现在不同障碍配置下的零样本超分辨率可达集预测；（2）将学习到的安全可达集与 RRT^X+TSP 的增量规划相结合，提供正式的有限时间恢复保证；（3）设计了梯度反馈恢复策略，确保在极端扰动下仍能收敛到安全区；

**🔧 技术方法**

使用了 Fourier Neural Operator、Hamilton–Jacobi reachability、增量 RRT^X、Traveling Salesman Problem、控制理论中的可达集约束和梯度反馈恢复策略；

**📊 数据集**

训练数据来源于数值求解器生成的合成障碍场（100个样本，每个包含单个随机圆形障碍，空间[-10,10]^2，时域T=8），并在 KUKA youBot 的 Webots 仿真环境中使用木箱障碍进行验证；

**📈 对比分析**

通过与已知地图基线和无可达集约束的规划进行对比，测量行驶距离和计算时间；在多目标案例中，未知障碍时行程距离与基线相近但时间略长；在单障碍的复原测试中成功率超过95%；在 youBot 仿真中成功完成所有目标并及时恢复；

**⚠️ 局限性**

局限性包括：依赖障碍近似为圆形、对极端障碍配置的收敛速度慢、RRT^X 重连与 Held–Karp 的指数复杂度限制了 K>8 的实用性，以及梯度不连续区的违规概率虽小但仍存在，且在更高维动力学下训练难度较大。

---

## 334. An approximation notion between P and FPTAS

**arXiv ID:** 2603.17489 | [PDF](https://arxiv.org/pdf/2603.17489v1)

**作者:** Samuel Bismuth `[一作]` (Ariel University), Erel Segal-Halevi `[通讯]` (Ariel University)

**通讯引用:** 747 | [OpenAlex ID](https://openalex.org/A5085873807)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出并研究了针对NP-hard优化问题的新的近似方案——FFPTAS（Fractional Fully Polynomial Time Approximation Scheme），并证明其在可约束问题类中严格介于多项式时间算法和FPTAS之间。

**💡 创新点**

创新点在于：1) 定义了以分数松弛解（PER）为基准的近似准则；2) 证明了FFPTAS比FPTAS更强但比多项式时间更弱；3) 给出了具体的NP-hard问题（如最大最小划分）证明其具有FFPTAS但无多项式解。

**🔧 技术方法**

主要技术包括：数学证明（利用分数松弛、二分法和FPTAS的组合），以及构造性的归约（如将4路划分归约到等卡划分）。

**📊 数据集**

该工作主要基于理论构造，无使用实验数据集。

**📈 对比分析**

比较方法是理论证明：通过示例问题展示FFPTAS能在多项式时间内给出相对于PER的(1‑t)近似，而FPTAS只能在多项式时间内给出相对于OPT的(1‑ε)近似，证明了其严格的包含关系。

**⚠️ 局限性**

局限性：结果依赖于问题的具体bivariate表示（f,g）和对应的PER定义；不同表示可能导致是否存在FFPTAS的差异；此外，尚未找到一个与OPT直接定义的统一类，使得P ⊊ C ⊊ FPTAS在所有问题上成立。

---

## 335. On the Cone Effect and Modality Gap in Medical Vision-Language Embeddings

**arXiv ID:** 2603.17246 | [PDF](https://arxiv.org/pdf/2603.17246v1)

**作者:** David Restrepo `[一作]` (CentraleSupelec), Enzo Ferrante `[通讯]` (Universidad de Buenos Aires)

**通讯引用:** 5467 | [OpenAlex ID](https://openalex.org/A5032685263)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了一种轻量级的后置对齐机制，通过单一超参数λ在保持预训练VLM编码器冻结的前提下，连续调节图像和文本嵌入之间的模态间隙，从而系统评估模态间隙对医疗与自然领域下游多模态任务性能的影响。

**💡 创新点**

创新点在于：①仅通过对齐操作（不重新训练编码器）即可调节模态间隙；②发现中等程度的模态间隙收缩可提升下游性能，而完全消除模态间隙可能导致重要模态特征丢失；③通过对比通用与医学专业化VLM模型，揭示医疗数据因语义/视觉同质性更易出现尖锐的“锥形效应”。

**🔧 技术方法**

技术方法包括：对图像/文本嵌入做L2归一化；计算模态中心差向量Δ；对嵌入按λ沿Δ方向平移并重新归一化；采用早期融合线性探针（无隐藏层）进行下游分类；用AUC衡量性能；用主成分分析+均向量长度（R）可视化锥形集中度。

**📊 数据集**

使用的数据集有：医疗领域的MIMIC‑CXR、HAM10000、BRSET、mBRSET；自然领域的COCO‑QA、Fakeddit、Recipes5k、DAQUAR；对应VLM模型有CLIP、SigLIP、BioMedCLIP、MedSigLIP。

**📈 对比分析**

通过在每个模型/数据集上对λ从0到1进行离散调整，并在固定的线性探针上评估AUC，发现大多数情况下λ≈0.4–0.6可实现AUC提升，尤其对通用VLM在医疗数据上提升显著；但当λ接近1时，一些医学模型的性能略有下降，表明过度对齐会削弱模态特异性信息。

**⚠️ 局限性**

局限性包括：①需要手动寻找最佳λ，缺乏自动调优策略；②仅测试线性探针，未评估更复杂下游任务或端到端微调；③对齐方法仅针对视觉与文本两模态，难以直接推广到多模态多源情景；④在极端收敛模态间隙时可能导致信息熵下降，需进一步理论与实验验证。

---

## 336. Enhancing Financial Report Question-Answering: A Retrieval-Augmented Generation System with Reranking Analysis

**arXiv ID:** 2603.16877 | [PDF](https://arxiv.org/pdf/2603.16877v1)

**作者:** Zhiyuan Cheng `[一作]` (Stanford University), Xiaoxi Qi `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了面向S&P 500公司10-K报告的检索增强生成(RAG)问答系统，并评估了神经重排序对答案质量的影响。

**💡 创新点**

通过在传统检索+生成框架中引入跨编码器重排序，并结合混合检索与递归排名，显著提升了财务问答准确率。

**🔧 技术方法**

使用了GPT‑4.1生成、OpenAI text‑embedding‑3‑small嵌入、FAISS向量检索、SQLite FTS5全文检索以及Jina Reranker v2跨编码器重排序。

**📊 数据集**

FinDER基准数据集（5703条10‑K相关问答对），随机抽取1500条进行实验。

**📈 对比分析**

对比无重排序与有重排序两种配置，重排序后平均分提升1.07点，基本正确率从33.5%增至49.0%，错误率从35.3%降至22.5%，优于基线。

**⚠️ 局限性**

依赖LLM评测可能产生偏差；仅评估10‑K报告，未覆盖其他财报；重排序增加延迟和API成本；缺乏人工评估与多模态处理。

---

## 337. EvoGuard: An Extensible Agentic RL-based Framework for Practical and Evolving AI-Generated Image Detection

**arXiv ID:** 2603.17343 | [PDF](https://arxiv.org/pdf/2603.17343v1)

**作者:** Chenyang Zhu `[一作]` (University of Tokyo), Isao Echizen `[通讯]` (National Institute of Informatics)

**通讯引用:** 5726 | [OpenAlex ID](https://openalex.org/A5044556342)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一个基于代理式框架EvoGuard，用多模态大型语言模型代理在不同检测器间进行动态规划和推理，实现对AI生成图像的检测。

**💡 创新点**

将多种现有检测器包装为可调用工具，利用能力感知的工具选择与动态编排实现自适应推理；同时通过GRPO强化学习仅用二元标签训练代理，实现零训练插件扩展。

**🔧 技术方法**

使用多模态LLM（如Qwen3‑VL‑4B‑Instruct）作为代理，GRPO算法进行Agentic RL，工具配置文件（能力谱）进行工具选择，动态规划和反思机制。

**📊 数据集**

训练集约8k图像，来自MMFR、DDA‑COCO、SynthScars等；评测集包含LOKI、Bfree、CommunityForensic等多样化数据。

**📈 对比分析**

与四个单体SOTA检测器、其他LLM/多模态框架以及混合方法对比，EvoGuard在所有三个基准上实现了最优或接近最优的平衡准确率与F1，并显著缓解了正负样本偏差。

**⚠️ 局限性**

代理推理带来额外计算开销，且性能受限于底层工具集；若所有工具失效，代理可能被误导；对极端新型生成器的鲁棒性尚待进一步验证。

---

## 338. AgentVLN: Towards Agentic Vision-and-Language Navigation

**arXiv ID:** 2603.17670 | [PDF](https://arxiv.org/pdf/2603.17670v1)

**作者:** Zihao Xin `[一作]` (Nanjing University of Aeronautics and Astronautics), Shengjun Huang `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**通讯引用:** 4277 | [OpenAlex ID](https://openalex.org/A5103204774)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出AgentVLN框架，采用VLM-as-Brain的思维-动作分离架构，实现轻量级、高效的视觉语言导航；

**💡 创新点**

创新点包括跨空间表示映射将3D路径投影到图像像素，细粒度上下文自校正与主动探索策略，以及查询驱动感知链式推理（QD‑PCoT）解决尺度歧义；

**🔧 技术方法**

核心技术包括POSMDP建模、跨空间投影、基于Qwen2.5‑VL‑3B的VLM、RTAB‑Map SLAM、可插拔感知与规划技能库；

**📊 数据集**

使用了R2R‑CE、RxR‑CE基准数据集，并构建AgentVLN‑Instruct大规模指令‑技能对齐数据集；

**📈 对比分析**

在R2R‑CE与RxR‑CE未见环境的验证集上，AgentVLN‑3B在SPL、SR、OS等指标上分别提升至70.7%、64.7%和73.5%，比同类SOTA方法提高约10%+，同时参数量仅3B；

**⚠️ 局限性**

局限性在于仍依赖RGB‑Depth观测，处理极端遮挡或光照变化时可能受限，且跨模态对齐依赖手工设计的投影与技能库，未能完全实现端到端自学习。

---

## 339. Ablation Study of a Fairness Auditing Agentic System for Bias Mitigation in Early-Onset Colorectal Cancer Detection

**arXiv ID:** 2603.17179 | [PDF](https://arxiv.org/pdf/2603.17179v1)

**作者:** Amalia Ionescu `[一作]`, Tiffani J. Bright `[通讯]`

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并实现了一个基于两位代理的公平性审计系统，先由领域专家代理综合早发性结直肠癌差异文献，再由公平性顾问代理推荐敏感属性和公平度量；通过消融实验评估不同LLM规模与检索增强生成（RAG）的影响。

**💡 创新点**

创新点在于将Agentic AI与RAG相结合，构建可自动生成公平性审计建议的系统，消除传统人工审计的专业瓶颈，并在不同模型规模下系统性比较RAG与非RAG的效能。

**🔧 技术方法**

使用的技术包括：三种开源LLM（Llama 3.1 8B、GPT‑OSS 20B、GPT‑OSS 120B）、检索增强生成（RAG）框架、Chroma向量数据库与mxbai‑embed‑large嵌入模型，以及JSON结构化输出与语义相似度评估。

**📊 数据集**

数据集为39篇关于早发性结直肠癌的PubMed文章（22篇关注差异，17篇噪声）和一个文献驱动的公平度量库，用于检索支持代理推理。

**📈 对比分析**

通过与专家制定的基准语句计算语义相似度，对比预训练LLM、无RAG代理和RAG代理三种配置，结果显示RAG代理在所有模型中显著提升Agent 1的相似度，Agent 2在大型模型中也有所提升，说明检索和代理结构能够提高输出质量。

**⚠️ 局限性**

局限性包括：语义相似度仅为质量指标，未直接评估事实正确性与临床适用性；实验仅覆盖三种LLM和单一临床领域；缺乏专家主观评估和更广泛模型/领域验证。

---

## 340. PRISM: Demystifying Retention and Interaction in Mid-Training

**arXiv ID:** 2603.17074 | [PDF](https://arxiv.org/pdf/2603.17074v1)

**作者:** Bharat Runwal `[一作]` (IBM Research), Rameswar Panda `[通讯]` (IBM Research)

**通讯引用:** 6468 | [OpenAlex ID](https://openalex.org/A5034529775)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并系统评估了一个中间训练（mid‑training）管线 PRISM，利用有限的 27B 令牌在多模型、多架构、多规模（3B–24B）上提升推理与强化学习（RL）性能。

**💡 创新点**

首次提供了中间训练的全维度经验框架（数据混合、时机、长上下文保持、RL 交互），证明 mid‑training 能显著提升推理、RL 可扩展性，并且在 dense Transformer 与 attention‑Mamba 混合架构上均表现一致。

**🔧 技术方法**

采用了预训练‑中间训练‑RL 三阶段训练流程，混合注意力与 Mamba 结构，使用 CKA 与权重稀疏/密集度分析评估机制，长上下文恢复与 token‑预算、上下文长度 ablation。

**📊 数据集**

使用了数学、代码、科学领域的高质量推理数据（Open‑R1, Megamath, OpenCodeReasoning, OpenThoughts 等）以及通用网络、对话、指令数据（DCLM‑EDU、WildChat‑1M 等）。

**📈 对比分析**

通过在 7 种基准（LB‑V1/V2、RULER、LiveCodeBench、Codeforces、AIME、MATH500、GPQA‑Diamond）对 7 个基础模型进行对照实验，mid‑training 在数学上提升 +15~+40 分、代码 +5~+12 分，RL 再提升 +8~+12 分，宏观平均从 <12 提升至 29–42，提升幅度约 3–4 倍。

**⚠️ 局限性**

限制包括：RL 数据选择未针对每个模型个性化；仅覆盖数学/代码/科学三大领域；实验规模限于 24B；长上下文 mid‑training 未充分探索更高上下文长度与 token‑预算的组合；RL 数据分布与混合方式仍可进一步优化。

---

## 341. Mutually Causal Semantic Distillation Network for Zero-Shot Learning

**arXiv ID:** 2603.17412 | [PDF](https://arxiv.org/pdf/2603.17412v1)

**作者:** Shiming Chen `[一作]` (Huazhong University of Science and Technology), Xinge You `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 6232 | [OpenAlex ID](https://openalex.org/A5057095711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种互因果语义蒸馏网络（MSDN++），通过属性→视觉和视觉→属性两条互相学习的注意子网络，结合因果干预和语义蒸馏，提升零样本学习的特征表达与知识迁移。

**💡 创新点**

创新点包括：①使用双向因果注意机制（属性到视觉和视觉到属性），显式衡量注意力的因果效应；②引入因果干预（如随机注意力）来正则化注意力学习；③设计互相蒸馏损失，让两条子网络相互指导，提升语义知识的完整性和鲁棒性。

**🔧 技术方法**

核心技术包括：视觉注意力网络、语义映射、因果干预（do-operator）和因果损失、语义蒸馏（Jensen‑Shannon Divergence + L2 对齐）、自校准交叉熵、属性回归损失，并在ResNet101特征提取基础上训练。

**📊 数据集**

在四个公开基准上评估：CUB、SUN、AWA2、FLO，使用标准的训练/测试拆分和属性语义向量。

**📈 对比分析**

与众多生成式、共同空间、嵌入式和大规模视觉‑语言模型方法对比，在CZSL上取得最佳或第二佳成绩；在GZSL上在CUB、SUN、AWA2上分别达到70.6%、42.1%、72.5%的调和平均，显著优于前沿方法；在FLO上获得74.5%的调和平均。

**⚠️ 局限性**

局限性包括：对小样本类别（如SUN）仍表现不如生成式方法；因果干预策略（随机/均匀/反向注意）对结果影响相似，但仍需进一步探索更有效的干预；对属性歧义导致注意错误的情况仍存在。

---

## 342. WeatherReasonSeg: A Benchmark for Weather-Aware Reasoning Segmentation in Visual Language Models

**arXiv ID:** 2603.17680 | [PDF](https://arxiv.org/pdf/2603.17680v1)

**作者:** Wanjun Du `[一作]` (Beijing Jiaotong University), Shunli Zhang `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 3085 | [OpenAlex ID](https://openalex.org/A5063642673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出WeatherReasonSeg基准，结合可控合成恶劣天气数据和真实恶劣天气数据，专门评估视觉语言模型在天气干扰下的推理分割能力。

**💡 创新点**

①首个关注天气影响的推理分割基准；②使用物理可控天气合成与mask‑guided LLM提示生成真实场景查询，形成多维度（功能、应用、结构、关系、需求）查询集合；③提供细粒度严重程度分析与对比框架。

**🔧 技术方法**

基于物理的雨雪雾合成、mask‑guided 大语言模型提示生成查询、gIoU/cIoU评价指标以及 Grounded‑SAM、LISA、Seg‑R1、Seg‑Zero 等现有推理分割方法。

**📊 数据集**

①合成子集：ReasonSeg 生成雨、雪、雾三种天气，三个严重度等级；②真实子集：ACDC 真实降雨、雾、雪、夜景图像，结合mask‑guided LLM 提示生成多维度查询；共 44,721 图像‑查询对。

**📈 对比分析**

通过在合成与真实数据上对比 Grounded‑SAM、LISA、Seg‑R1、Seg‑Zero 等模型，发现天气严重度递增时 gIoU/cIoU 单调下降；在真实场景中，推理分割方法（如 Seg‑Zero‑7B）gIoU 仅为 28–40，远低于仅做感知的 SAM2（80–82），表明推理阶段是主要瓶颈。

**⚠️ 局限性**

VLM 缺乏天气感知的推理机制，低级视觉退化会导致语义对齐不稳，导致推理分割性能急剧下降；基准仅涵盖雨、雪、雾及夜景，缺少其他恶劣天气；查询维度仍偏重可观测属性，对高层次情境推理的支持不足。

---

## 343. Federated Multi Agent Deep Learning and Neural Networks for Advanced Distributed Sensing in Wireless Networks

**arXiv ID:** 2603.16881 | [PDF](https://arxiv.org/pdf/2603.16881v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 344. Visual SLAM with DEM Anchoring for Lunar Surface Navigation

**arXiv ID:** 2603.17229 | [PDF](https://arxiv.org/pdf/2603.17229v1)

**作者:** Adam Dai `[一作]` (Stanford University), Grace Gao `[通讯]` (Stanford University)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5069625302)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种结合学习式特征匹配与数字高程模型（DEM）约束的立体视觉SLAM系统，用于月球表面长距离自主导航。

**💡 创新点**

创新点在于将基于SuperPoint和LightGlue的鲁棒特征检测与匹配与DEM高度和表面法线因子融合到位姿图中，提供绝对表面约束，显著抑制长期漂移。

**🔧 技术方法**

采用学习式特征提取与匹配（SuperPoint+LightGlue）、立体视觉里程计、GTSAM位姿图优化、DEM高度/法线约束，以及Unreal Engine 5仿真引擎。

**📊 数据集**

使用三组数据集：合成的LuSNAR、火山熔岩类比数据集S3LI（埃特纳山）以及基于LOLA DEM的Unreal Engine仿真场景。

**📈 对比分析**

与基线VO、ORB‑SLAM3和仅循环闭环方法对比，DEM‑anchored SLAM在ATE上提升约1–2个数量级，RPE基本保持不变，显示在长距离、纹理重复或光照极端环境下的显著性能提升。

**⚠️ 局限性**

局限性包括对DEM误差（垂直/水平偏移）敏感、无法完全消除切向漂移、循环闭环仍需补充、计算资源有限导致长轨迹终止，以及在真实月球环境中缺乏完整地面真值验证。

---

## 345. Evaluating Ill-Defined Tasks in Large Language Models

**arXiv ID:** 2603.17067 | [PDF](https://arxiv.org/pdf/2603.17067v1)

**作者:** Yi Zhou `[一作]` (IBM Research), Basel Shbita `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

分析并揭示了当前针对不确定任务（如复杂指令跟随与自然语言转Mermaid序列图）的评估方法存在的缺陷，并通过案例研究提出更稳健、可解释的评估设计。

**💡 创新点**

提出了评估目标明确化、数据覆盖多样化、指标分解与判定者可靠性分析的三项原则，并用两大案例验证多维评估能揭示模型优劣。

**🔧 技术方法**

采用规则检验与LLM-as-a-Judge混合评估、Mermaid解析器语法检查、LLM对多维指标的评分与对比分析等技术。

**📊 数据集**

使用了 CIF benchmark 集合（IFEval、ComplexBench、FollowBench、MT-Bench、HELM、StructFlow）以及 NL2Mermaid 自定义数据集，并结合两名 LLM 判定者 DeepSeek-V3 与 GPT-OSS-120B 进行评估。

**📈 对比分析**

通过将传统单分数与分解多维评分进行对比，发现传统评估高分并不一定意味着高质量；例如对 Llama 3.1-8B 的逻辑分数提升至 80% 而整体分数下降，体现多维评估的诊断价值。

**⚠️ 局限性**

仍受判定者随机性、数据集覆盖不足、缺乏标准化指标导致的可重复性差以及对极端语义难题评估不完善等局限。

---

## 346. ListK: Semantic ORDER BY and LIMIT K with Listwise Prompting

**arXiv ID:** 2603.17223 | [PDF](https://arxiv.org/pdf/2603.17223v1)

**作者:** Jason Shin `[一作]` (University of Rochester), Fatemeh Nargesian `[通讯]` (University of Rochester)

**通讯引用:** 1251 | [OpenAlex ID](https://openalex.org/A5012572863)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过引入多路投票快速选择/快速排序和多路筛选等列表式排名聚合算法，结合细调的列表式排序器，构建了一个针对语义 top‑K 操作的低延迟优化框架。

**💡 创新点**

创新点包括：①首次将多路列表式快速选择/排序与多路筛选应用于语义 top‑K；②提供了成本与召回的理论模型并用于自适应参数调优；③将嵌入式基准与早停机制与自我一致性校正融合到算法中。

**🔧 技术方法**

技术手段主要有：列表式提示（listwise prompting）、精细化的列表式排名器 RankZephyr、基于嵌入的多路基线选择、早停与自我一致性校正、成本模型驱动的调参、并行化实现。

**📊 数据集**

实验使用了 SciFact（5,183 条 PubMed 摘要）和 SemBench Movies（1M 条电影评论，实验时截断为256 条）两个数据集。

**📈 对比分析**

与 LOTUS（基于对比提示的快速选择/排序）以及点式评分基线对比；通过 Recall@K、NDCG@K 与延迟进行评估；结果表明在保持相近召回/NDCG 的前提下，本文方法可将延迟降至 LOTUS 的 1/7.4，滤波器可进一步降低 1/1.34，top‑few 精炼可提升 NDCG 高达 4% 但延迟仅略升高。

**⚠️ 局限性**

主要局限在于：假设列表式排名器完美无误；使用的基准集相对简单；对噪声排名器的分析不完整；自我一致性校正采用了粗暴策略；在极大 K 或高度稠密的数据上可能表现不佳。

---

## 347. Bringing Network Coding into Multi-Robot Systems: Interplay Study for Autonomous Systems over Wireless Communications

**arXiv ID:** 2603.17472 | [PDF](https://arxiv.org/pdf/2603.17472v1)

**作者:** Anil Zaher `[一作]` (Technion Israel Institute of Technology), Alejandro Cohen `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过两组仿真实验（协同定位与安全紧急超车）评估不同传输层可靠性机制（UDP、Selective-Repeat ARQ、适应性因果随机线性网络编码 AC‑RLNC）对多机器人系统的估计精度与安全决策的影响。

**💡 创新点**

提出将网络编码与延迟感知重估（I‑ReE）结合，实现主动注入冗余并在失真/延迟环境中保持近乎理想的估计与高达 80% 的超车成功率；首次系统性证明通信层行为需与自主算法共设计。

**🔧 技术方法**

使用 AC‑RLNC、UDP、SR‑ARQ、扩展卡尔曼滤波器、延迟感知重估 I‑ReE 以及自定义的延迟/失真二元信道模型。

**📊 数据集**

采用基于真实参数的仿真数据：10 机器人、200 m 工作空间、Δt=0.1 s、RTT=4 槽等，所有实验均为合成仿真，未使用公开真实数据集。

**📈 对比分析**

通过估计误差曲线和超车成功概率曲线对比，AC‑RLNC 在各种失真概率下保持 1% 以内的定位误差，且在 25 分组到达时间 110 槽时的成功率约为 80%，明显优于 SR‑ARQ（约 60%）和 UDP（无恢复）。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，未考虑多跳网络、异构机器人及不可靠 ACK 通道；网络编码参数仅针对单一任务场景调优，缺乏通用自动调节机制；缺乏对实时硬件实现与实际无线干扰的评估。

---

## 348. SoK: From Silicon to Netlist and Beyond $-$ Two Decades of Hardware Reverse Engineering Research

**arXiv ID:** 2603.17883 | [PDF](https://arxiv.org/pdf/2603.17883v1)

**作者:** Zehra Karadağ `[一作]` (Ruhr University Bochum), Steffen Becker `[通讯]` (Ruhr University Bochum)

**通讯引用:** 255 | [OpenAlex ID](https://openalex.org/A5008222051)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统化整理并评估了过去20年硬件逆向工程（IC、FPGA、网表）领域的187篇同行评审论文及其提供的30个工件，揭示了领域碎片化和复现性差的问题。

**💡 创新点**

首次对整个硬件逆向工程领域进行统一的系统化知识梳理，并提出三项跨领域改进建议（提升复现性与可复用性、标准化基准与评估指标、法律澄清与支持）。

**🔧 技术方法**

采用四阶段文献检索与筛选流程（数据库检索、引文挖掘、主题编码、工件评估），结合手工标签、代码审查和工具可执行性测试。

**📊 数据集**

构建了187篇论文的元数据集（BibTeX/CSV），并对30个公开工件（工具、图片、脚本等）进行评测。

**📈 对比分析**

对工件进行可用性、功能完整性与可复现性三维评估；结果显示仅7篇论文（约4%）能够复现关键结果，整体复现率低。

**⚠️ 局限性**

仅涵盖英语论文，评估过程依赖手工评测且未与作者沟通，可能遗漏相关工作；研究聚焦公开工件，未覆盖私有或专利工具。

---

## 349. Objective Mispricing Detection for Shortlisting Undervalued Football Players via Market Dynamics and News Signals

**arXiv ID:** 2603.17687 | [PDF](https://arxiv.org/pdf/2603.17687v1)

**作者:** Chinenye Omejieke `[一作]`, Xia Cui `[通讯]` (Manchester Metropolitan University)

**通讯引用:** 1143 | [OpenAlex ID](https://openalex.org/A5065374968)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了基于市场预期与新闻情感的误价检测框架，用于筛选被低估的足球球员。

**💡 创新点**

创新点在于用客观误价定义无主观标签、整合Transformer情感与语义特征，并系统评估其对短名单的增益。

**🔧 技术方法**

技术上结合XGBoost/TabNet回归、DistilBERT情感分类、BERT句向量、PCA降维、SHAP解释和ROC‑AUC排序。

**📊 数据集**

使用Transfermarkt的历史市场价值、合同与转会记录，以及来自Guardian、BBC等的约7万篇新闻文本。

**📈 对比分析**

通过时间泄漏防护的顺序切分，XGBoost回归R²为0.935；在短名单任务中，完整模型ROC‑AUC为0.677，比仅用文本的0.514显著提升。

**⚠️ 局限性**

局限包括对Transfermarkt估值偏差的依赖、主要使用英语新闻、缺乏因果推断以及未考虑多语种或低媒体覆盖联赛。

---

## 350. Benchmarking Reinforcement Learning via Stochastic Converse Optimality: Generating Systems with Known Optimal Policies

**arXiv ID:** 2603.17631 | [PDF](https://arxiv.org/pdf/2603.17631v1)

**作者:** Sinan Ibrahim `[一作]` (Skolkovo Institute of Science and Technology), Pavel Osinenko `[通讯]` (Central University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套基于逆向最优性理论的离散时间加噪声控制系统基准生成框架，并通过该框架自动构造可校验的RL基准环境。

**💡 创新点**

将逆向最优性推广到带高斯噪声的控制非线性系统，并设计可调的能量方程和正交场参数化，实现可连续调节控制强度和非线性耦合的可验证基准。

**🔧 技术方法**

使用逆向最优性定理、矩阵分析、对角化与Woodbury公式、Givens旋转构造正交场、以及对齐的噪声偏移校正来构造系统漂移、策略与价值函数。

**📊 数据集**

生成两类基准族：串联n关节机械臂和动态扩展非完整车辆，并在GitHub发布验证过的YAML fixtures；实验采用标准RL算法（PPO、A2C、SAC、TD3、DDPG）在这些基准上训练。

**📈 对比分析**

采用共随机数配对评估，计算绝对最优性间隙和折扣化后遗弃；实验显示PPO在固定初态和高维动态扩展环境中优于其他方法，TD3/SAC在随机初态下表现最好，整体可通过置信区间展示差异。

**⚠️ 局限性**

局限在于仅适用于控制仿射、可加高斯噪声、二次成本的线性最优策略；无法覆盖非仿射控制、重尾或时间相关噪声、非二次代价与约束等更一般情形。

---

## 351. The Truth, the Whole Truth, and Nothing but the Truth: Automatic Visualization Evaluation from Reconstruction Quality

**arXiv ID:** 2603.16873 | [PDF](https://arxiv.org/pdf/2603.16873v1)

**作者:** Roxana Bujack `[一作]` (Los Alamos National Laboratory), David Rogers `[通讯]` (Los Alamos National Laboratory)

**通讯引用:** 12576 | [OpenAlex ID](https://openalex.org/A5012560979)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于从可视化图像重建原始数据的重构误差来自动评估可视化质量的方法，构建了agentic AI工作流程。

**💡 创新点**

创新点在于：①把重构误差作为通用、数据驱动的质量指标；②利用Neural Radiance Fields (NeRF) 对3D可视化进行重建；③统一处理多种可视化参数（色彩映射、等值线、视角）并考虑跨参数互作。

**🔧 技术方法**

主要技术包括 NeRF 3D 重建、RBF 插值、ΔE_2000/ΔE_1976 色彩距离、色彩判别力与对比度评估、Neural Network 生成式可视化脚本、离散视角采样与 Chamfer/Hausdorff 距离计算。

**📊 数据集**

使用的数据集有：PyVista 海拔子集、合成高斯核场、Utah Teapot 3D 模型、yA31 All Scalars 恐龙冲击温度场（tev）以及 Matplotlib 内置色彩映射列表。

**📈 对比分析**

与传统启发式指标（entropy、Hausdorff、Chamfer、colormap discriminative power）以及已知方法（Kindlmann、Carr、Bruckner）对比；实验表明重构误差能有效区分不同参数组合的可视化质量，优选低等值线、小色彩映射、侧向视角，整体误差明显低于基线。

**⚠️ 局限性**

限制：①仅提供第一阶近似重建，未覆盖矢量场、张量场等更复杂可视化；②NeRF 重建耗时（≈1min/视角）导致全参数搜索不实用；③重构误差对前景/背景的权重不均衡，无法体现人类关注重点；④缺少基于人类实验的感知验证；⑤未针对任务型可视化给出专门评估。

---

## 352. Scale-Aware Navigation of Astronomical Survey Imagery Data on High Resolution Immersive Displays

**arXiv ID:** 2603.17337 | [PDF](https://arxiv.org/pdf/2603.17337v1)

**作者:** Ava Nederlander `[一作]` (Stony Brook University), Arie E. Kaufman `[通讯]` (Stony Brook University)

**通讯引用:** 11565 | [OpenAlex ID](https://openalex.org/A5039517392)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了针对极大尺度天文影像的沉浸式可视化交互设计原则，强调连续缩放、空间连贯性和分层探索，并在Reality Deck和曲面沉浸环境中演示使用Rubin Observatory与Spitzer银河系红外数据。

**💡 创新点**

创新点在于将连续缩放交互与分层表示相结合，形成“尺度感知沉浸导航”范式，专为极大尺度数据设计。

**🔧 技术方法**

采用高分辨率多面墙显示（Reality Deck）、曲面显示、持续缩放交互、空间锚定注释和层叠渲染等技术。

**📊 数据集**

使用Vera C. Rubin Observatory公开深度调查图像和Spitzer GLIMPSE–MIPSGAL银河系红外图像。

**📈 对比分析**

本文未进行系统的定量评估，主要通过案例演示和专家反馈展示效果，缺乏对性能的具体指标。

**⚠️ 局限性**

局限性包括缺乏正式的用户研究、不同显示平台间性能对比不足，以及注释与连续缩放交互细节待进一步验证。

---

## 353. Continual Multimodal Egocentric Activity Recognition via Modality-Aware Novel Detection

**arXiv ID:** 2603.16970 | [PDF](https://arxiv.org/pdf/2603.16970v1)

**作者:** Wonseon Lim `[一作]` (Chung Ang University), Dae-Won Kim `[通讯]` (Chung Ang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种多模态第一人称开放世界持续学习框架 MAND，融合 MoAS（自适应模态加权）与 MoRST（模态特征稳定化）以实现对新颖活动的检测和已知活动的持续识别。

**💡 创新点**

创新点在于：①利用能量分数对每个模态进行样本级可靠性评估并自适应加权，充分挖掘视觉与惯性测量单元（IMU）的互补信息；②通过模态特定头部与模态日志回放蒸馏，稳定各模态的判别边界，缓解持续学习过程中的模态失衡与灾难性遗忘。

**🔧 技术方法**

技术细节包括：能量基可靠性计算、软最大化加权、模态特定分类头、对抗式日志蒸馏、BN‑Inception（RGB）+DeepConvLSTM（IMU）+Temporal Binding Network 的特征提取与融合；使用 replay 缓冲、跨模态统计归一化等。

**📊 数据集**

实验数据集为公开的 UESTC‑MMEA‑CL，包含 RGB、陀螺仪和加速度计同步流的 32 类日常活动。

**📈 对比分析**

与多种基线（iCaRL、ER、DER++、Foster、MONET、CMR‑MFN）及新颖性评分方法（MSP、MaxLogit、Entropy、Energy）对比，MAND 在所有增量设置下均实现最高 AUC（相较最佳基线提升约10%）与准确率（提升约2.8%），并显著降低误报率。

**⚠️ 局限性**

局限性在于需要在 replay 缓冲中存储模态 logits 与统计量，导致显著内存占用，并对缓冲质量及长期任务流的模态漂移敏感；未来需探索更轻量化的稳定化策略与对不同传感器可用性与域迁移的适应。

---

## 354. Text-to-Stage: Spatial Layouts from Long-form Narratives

**arXiv ID:** 2603.17832 | [PDF](https://arxiv.org/pdf/2603.17832v1)

**作者:** Jefferson Hernandez `[一作]` (Rice University), Ishwarya Ananthabhotla `[通讯]` (Meta Reality Labs Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究从长篇叙事文本自动生成舞台布局（场景分割、角色定位、移动轨迹等）并提出可验证的戏剧评估套件来训练与评估模型。

**💡 创新点**

创新点包括：①提出文本到舞台（text‑to‑stage）任务及其 JSON 规范；②构造基于戏剧与视觉构图原则的确定性评估器作为可验证奖励；③结合拒绝式 SFT 与 GRPO 的 RL‑verifiable 训练流程，实现高质量空间推理；④通过多维度自动评估与人类试听偏好验证其有效性。

**🔧 技术方法**

采用的大型语言模型（Qwen3‑8B、Llama‑4‑Maverick 等）与 Best‑of‑N 采样、拒绝式 SFT、Group Relative Policy Optimization (GRPO) 等技术；评估器将戏剧原理转化为可计分的规则集合。

**📊 数据集**

主要数据集为 Project Dialogism Novel Corpus (PDNC)，包含经典英语小说的对话归属与别名标注；另外使用 FanFic 集合进行合成数据扩充。

**📈 对比分析**

与多种零-shot、提示式基线（Claude、GPT‑4、Llama 等）以及开源模型对比；在确定性评估宏平均得分上，GRPO 版 Spatializer 达到 88.5%，显著高于最佳开源 83.4% 与最佳专有 82.0%；在 LLM‑as‑Judge 与人类试听偏好测试中也表现最优。

**⚠️ 局限性**

局限性包括：①场景转换的表现仍是最低分项；②离散 3×3 网格和离散步长限制了物理真实性与表达力；③数据覆盖仅为经典英文小说，存在域迁移风险；④奖励机制虽与人类偏好对齐但仍缺乏对某些主观属性的捕捉；⑤跨模型族迁移效果不均匀。

---

## 355. Large Language Models in Teaching and Learning: Reflections on Implementing an AI Chatbot in Higher Education

**arXiv ID:** 2603.17773 | [PDF](https://arxiv.org/pdf/2603.17773v1)

**作者:** Fiammetta Caccavale `[一作]` (Technical University of Denmark), Ulrich Krühne `[通讯]` (Technical University of Denmark)

**通讯引用:** 1775 | [OpenAlex ID](https://openalex.org/A5057276095)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在化学与生物化学工程硕士课程中，作者开发并验证了一款基于检索增强生成（RAG）技术的LLM助手，用于替代教师进行审核模拟练习。

**💡 创新点**

创新点在于将LLM与检索增强结合，并通过多轮混合方法实验（跨期、交叉对照）评估其在教学中的可行性与学生体验，首次将LLM用于高等教育的专业化审核练习。

**🔧 技术方法**

采用的技术包括开放源代码的FLAN‑T5小模型、RAG检索机制、Flask部署、图形用户界面，并在实验中集成语音识别与语音合成功能。

**📊 数据集**

数据集由教师预先提供的历史回答、审核文档和课程教材构成，随后根据学生交互动态更新以增强检索上下文。

**📈 对比分析**

通过三轮实验（2024、2025学期以及2025年秋季交叉对照）采用Likert量表和Mann‑Whitney U / Chi‑square检验进行比较，结果显示学生对LLM的满意度与答复质量从2024年到2025年提升，虽然在跨期对照中部分指标显著差异，但整体未出现显著差异，且LLM在信息检索和减压方面表现优于教师，成绩略低但不显著。

**⚠️ 局限性**

局限包括样本量小、单一课程与机构的特定性、LLM模型能力有限（易幻觉、复杂问题处理不足）、对学习成果影响未量化，以及快速演进的LLM技术可能导致研究结果过时。

---

## 356. Entropy-Aware Task Offloading in Mobile Edge Computing

**arXiv ID:** 2603.16949 | [PDF](https://arxiv.org/pdf/2603.16949v1)

**作者:** Mohsen Sahraei Ardakani `[一作]` (North Carolina State University), Rui Song `[通讯]` (North Carolina State University)

**通讯引用:** 6435 | [OpenAlex ID](https://openalex.org/A5089012283)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在移动边缘计算（MEC）环境中引入隐私考量，提出基于熵的隐私奖励函数，并通过深度递归Q网络（DRQN）学习任务卸载策略，提升设备能耗与延迟管理的同时保障用户隐私；

**💡 创新点**

创新点在于：①设计了理论上可降低攻击成功概率的熵隐私度量；②将非马尔可夫的隐私奖励引入MDP，并通过DRQN克服状态信息缺失导致的非马尔可夫性；③通过仿真验证该方法在能耗、延迟与隐私熵三方面均优于传统DQN和启发式隐私奖励方法；

**🔧 技术方法**

使用技术包括：马尔可夫决策过程（MDP）建模、强化学习（Q学习）、深度Q网络（DQN）、深度递归Q网络（DRQN）与GRU循环网络、经验回放与软目标网络更新；

**📊 数据集**

使用的数据为仿真生成的环境数据：任务生成率、通道状态转移概率、缓冲区大小、任务大小等参数，未使用公开真实数据集；

**📈 对比分析**

通过与标准DQN、启发式隐私奖励DQN、θ-private随机策略等基线比较，实验结果显示DRQN在平均能耗、延迟和隐私熵（H(D,T)、H(G,T)）方面均优于基线；

**⚠️ 局限性**

局限性包括：仅考虑单用户场景、固定任务大小、无任务丢失处理；熵奖励需要使用历史窗口W，导致状态空间指数增长；对窗口大小和超参数敏感，且仿真环境可能未完全反映真实网络复杂性。

---

## 357. TransText: Transparency Aware Image-to-Video Typography Animation

**arXiv ID:** 2603.17944 | [PDF](https://arxiv.org/pdf/2603.17944v1)

**作者:** Fei Zhang `[一作]` (Shanghai Jiao Tong University), Semih Gunel `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一种用于图像支持的 RGBA 字形动画的扩散模型 TransText，能够在不重新训练 VAE 的情况下同时生成 RGB 和 Alpha 通道视频。

**💡 创新点**

创新点包括：① 将单通道 Alpha 通过通道复制转为 RGB 格式（Alpha-as-RGB）以统一处理；② 在潜在空间使用空间拼接而非时间拼接来实现 RGB 与 Alpha 的对齐，避免通道混合和长序列训练难题；③ 引入 Alpha 重建正则化（ℒ_rec）对 Alpha 潜在进行一次性反向恢复，提升透明度边界清晰度与运动一致性。

**🔧 技术方法**

主要技术包括：扩散变换器（DiT）流匹配（flow‑matching）框架、CLIP 与 umT5 进行文本与图像条件编码、Alpha-as-RGB 处理、空间潜在拼接、Alpha 重建正则化。

**📊 数据集**

使用了从 Shutterstock 收集并标注的 16,108 条 RGBA 字形动画数据集（15,312 条训练、796 条验证），覆盖 8 种视觉特效。

**📈 对比分析**

与两类基线比较：1）生成‑再预测的两阶段 RGBA 方法；2）基于 TransPixeler 的 VAE‑free 方案。TransText 在 FVD、Soft α‑mIoU、RGBA 对齐率和用户体验评分上均显著优于对照组，例如 FVD 123.48、α‑mIoU 79.90、RGBA 对齐 87.31，效率与 TransPixeler 相近但质量更好。

**⚠️ 局限性**

局限性：① 仅在有限的 16k 字形动画数据上验证，缺乏对更大规模或更复杂背景的泛化评估；② Alpha‑as‑RGB 方式仍可能在极端光照或细节稀疏场景下出现颜色泄漏；③ 需要较高的 GPU 资源（A100/H100）进行训练。

---

## 358. KANtize: Exploring Low-bit Quantization of Kolmogorov-Arnold Networks for Efficient Inference

**arXiv ID:** 2603.17230 | [PDF](https://arxiv.org/pdf/2603.17230v1)

**作者:** Sohaib Errabii `[一作]` (Rennes University), Marcello Traiola `[通讯]` (Rennes University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了低位整数量化对Kolmogorov‑Arnold网络（KAN）在准确性、计算复杂度和硬件效率上的影响，并通过预计算B‑spline查找表实现递归求值的加速；

**💡 创新点**

创新点在于证明B‑spline对2‑3位量化极为鲁棒，提出将B‑spline量化为低位并用预取表替代递归计算，显著降低BitOps和硬件资源，同时对完整spline表量化的可行性与局限性进行了系统评估；

**🔧 技术方法**

采用统一整数量化、B‑spline预取表、spline表量化技术，并在GPU、FPGA及28 nm FD‑SOI ASIC平台上实现硬件加速；

**📊 数据集**

使用MNIST和CIFAR‑10两个常见分类数据集进行实验；

**📈 对比分析**

与FP32基线和其他量化方案对比，ResKAN18模型在不损失准确率的前提下实现了50×BitOps削减、GPU推理加速达2.9×、FPGA资源缩减36%、ASIC面积缩减72%以及频率提升50%；

**⚠️ 局限性**

局限性包括：完整spline表量化仅适用于小模型，规模大模型会导致资源溢出；权重量化对准确率敏感，低于5位会显著下降；GPU仅对8位量化有效，低于8位不再带来额外加速。

---

## 359. VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs

**arXiv ID:** 2603.17652 | [PDF](https://arxiv.org/pdf/2603.17652v1)

**作者:** Chaokang Jiang `[一作]` (Bosch), Kevin Li Sun `[通讯]` (Bosch)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种基于向量图的流式世界模型VectorWorld，能够在闭环仿真中实时生成并扩展局部地图与交通主体；

**💡 创新点**

创新点包括①运动感知的交互状态VAE实现初始化与历史条件政策对齐；②基于边缘门控的关系DiT结合间隔条件MeanFlow与JVP监督，支持一次性无解算器的结构化补全；③物理对齐的NPC策略ΔSim，采用返回期条件、混合离散+连续动作与可微动力学约束，显著减少长时程漂移；

**🔧 技术方法**

采用了变分自编码器、边缘门控Transformer (DiT)、间隔条件MeanFlow（流匹配）与JVP技术、FiLM返回期条件、混合动作头、可微动力学约束（DKAL）以及基于流形的无解算器一阶采样；

**📊 数据集**

使用Waymo Open Motion与nuPlan两大公开交通数据集进行训练与评估；

**📈 对比分析**

与SLEDGE、ScenDream等现有向量图生成器相比，VectorWorld在地图结构FID、终点距离、碰撞率等指标上均取得显著提升；在闭环测试中支持1km+无漂移滚动，单片生成时间≈6 ms/64 m×64 m，压力测试成功率从25%提升至56%；

**⚠️ 局限性**

局限性包括对向量图格式的依赖，仍需进一步提升对极端稀疏场景的泛化；模型训练复杂度高，且在极长时序或极大尺度仿真中可能存在累计误差；

---

## 360. Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing

**arXiv ID:** 2603.17199 | [PDF](https://arxiv.org/pdf/2603.17199v1)

**作者:** Parsa Mirtaheri `[一作]` (University of California San Diego), Mikhail Belkin `[通讯]` (University of California San Diego)

**通讯引用:** 22798 | [OpenAlex ID](https://openalex.org/A5102796459)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究大型语言模型在多选题中出现的“动机性推理”，即模型在链式推理过程中忽略提示却合理化暗示答案，利用模型内部激活进行检测；

**💡 创新点**

创新点在于证明即使链式推理文本不透露动机性推理，内部激活仍能准确预测；并首次展示了在生成链式推理之前就能通过预生成激活预判动机性推理。

**🔧 技术方法**

主要技术包括递归特征机（RFM）探测器对残差流激活进行监督学习、提示恢复实验、以及与 GPT‑5‑nano 的 CoT 监控基线对比。

**📊 数据集**

使用四个多选推理基准（MMLU、ARC‑Challenge、CommonsenseQA、AQuA），并在三种提示形式（Sycophancy、Consistency、Metadata）下构造提示对。

**📈 对比分析**

在后生成检测任务中，RFM 探测器在大多数模型上达到了高于 CoT 监控的 AUC（约 0.70‑0.85）；预生成检测的 AUC 与 CoT 监控相当（0.65‑0.82），并能在未生成推理文本前识别动机性案例。

**⚠️ 局限性**

局限性包括仅评估带有显式提示的多选任务，未覆盖更通用或隐式偏见场景；对不同模型的通用性和因果机制仍未完全揭示。

---

## 361. Transformers Can Learn Rules They've Never Seen: Proof of Computation Beyond Interpolation

**arXiv ID:** 2603.17019 | [PDF](https://arxiv.org/pdf/2603.17019v1)

**作者:** Andy Gray `[一作]` `[通讯]` (Kortical), Andy Gray (Kortical)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在两种人工合成任务（XOR 细胞自动机和整数运算符链）中，训练一个两层Transformer，证明其能够学习并推断训练集中从未出现过的规则，且不依赖于简单的相似性插值。

**💡 创新点**

关键创新在于：① 通过数学证明将插值在特定输入空间内完全排除；② 展示软递归（soft unrolling）和多步梯度传播能激活Transformer内部的约束传播机制；③ 通过电路提取和多层探针验证Transformer确实实现了XOR 的非线性布尔运算，而非仅仅拟合观测样本。

**🔧 技术方法**

使用了标准的Transformer编码器（post‑LN、ReLU FFN、4 头多头注意力）、soft unrolling 预测序列、梯度流向后续时间步，以及在实验 2 中的 encoder‑decoder 结构；对比了 KNN、KRR、SVM、RF、MLP 等基线。

**📊 数据集**

数据集：1) 1D 细胞自动机（Rule 150、Rule 30、Rule 106 等）生成的二进制序列，隐藏特定局部模式；2) 6 位整数的组合运算符链（XOR、OR、AND、NOR、NAND、LSHIFT、RSHIFT），一次性隐藏所有组合对中的一个。

**📈 对比分析**

对比方法：在相同训练/测试划分下，基线在隐藏模式上均取得 0%（或接近 0%）准确率；Transformer 在 soft unrolling 情况下达到 96–99% 的 hold‑out 准确率，单步预测仅匹配输出偏置 63%。实验 2 中，Transformer 超越所有基线（最高 78.6%），且去掉中间步骤监督会显著下降。

**⚠️ 局限性**

局限性：① 仅在极小的合成任务上验证，缺乏对大规模 LLM 的直接证据；② 证明性强，但对不同任务的可迁移性不确定；③ 训练成功率呈双峰分布，部分随机种子会失败；④ 只验证了离散布尔/整数规则，未覆盖连续或噪声环境；⑤ 结果表明 Transformer 能在特定条件下实现计算，但未说明在自然语言训练中何时会激活该机制。

---

## 362. SafeLand: Safe Autonomous Landing in Unknown Environments with Bayesian Semantic Mapping

**arXiv ID:** 2603.17430 | [PDF](https://arxiv.org/pdf/2603.17430v1)

**作者:** Markus Gross `[一作]` (Fraunhofer IVI Autonomous Aerial Systems), Henri Meeß `[通讯]` (Fraunhofer IVI Autonomous Aerial Systems)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了一个仅使用摄像头和轻量高度传感器的轻量化视觉自主降落系统（SafeLand），能够在未知、动态环境中安全识别并降落至合适的地面位置。

**💡 创新点**

创新点包括：①针对空中场景训练的专用语义分割模型（SegFormer MiT‑B3），集成七大公开数据集并实现高效嵌入式部署；②通过贝叶斯概率滤波与语义衰减持续更新地面语义地图，显著提升鲁棒性；③使用行为树实现对动态障碍（尤其是人类）的实时检测、暂停或重新规划；④实现子秒级响应延迟和零人类误报，且不依赖先验地图或重型传感器。

**🔧 技术方法**

主要技术手段包括：深度学习语义分割（SegFormer + TensorRT FP16 推理），相机投影与地面映射，贝叶斯滤波与语义衰减，行为树决策逻辑，ROS2+PX4 集成，Jetson AGX Orin 轻量硬件平台。

**📊 数据集**

使用七个公开空中数据集（涵盖多种场景和语义类别）重新统一标注为20类进行训练；保持原始训练/验证/测试划分，最终在综合测试中取得 70.22% mIoU。

**📈 对比分析**

与现有基线 Safe2Ditch 在 200 次仿真和 60 次现场测试（高度 25–100 m）对比：SafeLand 在所有场景中实现 95% 成功率、零人类误报，响应延迟在子秒级，明显优于 Safe2Ditch 的多目标跟踪延迟；失败案例仅因硬件通信问题，与系统本身无关。

**⚠️ 局限性**

局限性包括：假设平坦地面，无法完全处理倾斜或不规则地形；低高度下摄像头分辨率不足导致分割精度下降；极端光照/天气条件下鲁棒性待验证；行为树参数需针对不同 UAV/任务手动调优。

---

## 363. SEAL-Tag: Self-Tag Evidence Aggregation with Probabilistic Circuits for PII-Safe Retrieval-Augmented Generation

**arXiv ID:** 2603.17292 | [PDF](https://arxiv.org/pdf/2603.17292v1)

**作者:** Jin Xie `[一作]`, Guang Cheng `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SEAL-Tag 框架，通过 Verify‑then‑Route 的运行时协议在 Retrieval‑Augmented Generation（RAG）中实现可验证的 PII 审计，显著降低上下文泄露；

**💡 创新点**

创新点包括：① SEAL‑Probe 生成结构化 PII‑Evidence Table（PET）以实现内部工具调用；② 用可解释、可校准的概率电路（PC）替代传统神经安全头，保证单调性与硬约束；③ S0–S6 Anchored Synthesis Pipeline 与两阶段（感知‑对齐）微调策略解决 PII “冷启动”与格式崩塌问题；

**🔧 技术方法**

采用 LLM 功能调用/工具学习、概率电路（Sum‑Product Network）、两阶段自适应微调（Perception 与 Protocol Alignment）、结构化合成数据生成、可解释规则与一致性特征；

**📊 数据集**

使用自研 12,000 条评测样本（12k benchmark），40,000 条高质量合成样本（S0–S6 Pipeline）以及公开 NER/CoNLL 等数据集进行预训练与微调；

**📈 对比分析**

通过与原始模型、少量示例提示、DPVoteRAG、PrivacyMind、P_3Defer、Eraser4RAG 等五种基线对比，Seal‑Tag 在 CopyBreakRAG 攻击下 ASR 仅 9.5%（相较 81% 降低 8.6×），且在 PopQA、MedQA 等通用 QA 任务中的精度几乎与未加防护模型持平，延迟仅 +18 ms，显著优于 LLM‑Judge（+1,450 ms）和本地 Scrubber（+120 ms）；

**⚠️ 局限性**

局限性：依赖大规模合成数据，可能对未见 PII 类型或极端攻击（如高度隐匿的 PET‑spoofing）产生误判；PC 设计与参数需手工校准，且在极端低‑资源场景下对复杂链式推理的支持仍有限；

---

## 364. Music Source Restoration with Ensemble Separation and Targeted Reconstruction

**arXiv ID:** 2603.16926 | [PDF](https://arxiv.org/pdf/2603.16926v1)

**作者:** Xinlong Deng `[一作]` (China University of Petroleum), Jie Jiang `[通讯]` (China University of Petroleum)

**通讯引用:** 4861 | [OpenAlex ID](https://openalex.org/A5043074429)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了一个两阶段的音乐源恢复系统，先用多模型集成的音乐源分离模型生成初步估计，再通过基于BSRNN的恢复模型进行分离与恢复的联合优化。

**💡 创新点**

创新点在于将预训练的源分离模型与专门针对退化混音的恢复模型耦合，利用分离估计作为先验信息进行联合优化，从而逆转复杂的制作链（EQ、压缩、混响等）。

**🔧 技术方法**

技术包括BS-RoFormer、MDX23C等分离模型的集成，基于BSRNN的源恢复模型，以及多频谱信号与语义评估指标（MMSNR、FAD-CLAP）。

**📊 数据集**

使用ICASSP 2026 Music Source Restoration Challenge 的官方评测数据集（包含已混音并加压缩/EQ/混响等失真效果的音乐），以及挑战提供的基线BSRNN模型。

**📈 对比分析**

在官方 MSR 基准上与基线及单独分离模型比较，系统在 MMSNR、FAD-CLAP 等所有指标均优于基线，最终在挑战排行榜上排名第二，获得 2.3405 MMSNR、0.2253 FAD、0.0164 Zimt 和 3.2262 MOS 的成绩。

**⚠️ 局限性**

主要限制是数据稀缺和分布偏移，尤其是鼓与打击乐类样本不足导致 MMSNR 接近零，导致该类在测试阶段被跳过；此外分离模型在已混音音乐上的域差异仍需进一步缓解。

---

## 365. Omni-I2C: A Holistic Benchmark for High-Fidelity Image-to-Code Generation

**arXiv ID:** 2603.17508 | [PDF](https://arxiv.org/pdf/2603.17508v1)

**作者:** Jiawei Zhou `[一作]` (Wuhan University), Jing Zhang `[通讯]` (Xidian University)

**通讯引用:** 17336 | [OpenAlex ID](https://openalex.org/A5100345321)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Omni-I2C基准，评估大型多模态模型将复杂结构化图形转为可执行代码的能力。

**💡 创新点**

构建跨5种编程语言、8个学科、45种图形类型的多样化数据集，并设计双维度评估框架（感知与符号），揭示模型在结构重建中的关键瓶颈。

**🔧 技术方法**

使用多模态模型推理、LLM评审器、自动化代码检查与Tanimoto相似度等技术。

**📊 数据集**

Omni-I2C自建的1080条样本，来源于真实科学可视化、化学分子结构、公式、HTML等应用。

**📈 对比分析**

与13个闭源和开源LMM对比，采用执行率、功能覆盖率、参数准确度、感知得分等指标，结果显示前沿模型仍存在显著性能缺口，尤其在LaTeX和SVG等复杂任务中。

**⚠️ 局限性**

缺少自然场景图像，评估依赖LLM/ LMM 评判导致成本高、延迟大。

---

## 366. Accurate Shift Invariant Convolutional Neural Networks Using Gaussian-Hermite Moments

**arXiv ID:** 2603.17098 | [PDF](https://arxiv.org/pdf/2603.17098v1)

**作者:** Jaspreet Singh `[一作]` (Teesside University), Grzegorz Cielniak `[通讯]` (University of Lincoln)

**通讯引用:** 2488 | [OpenAlex ID](https://openalex.org/A5036667542)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种新的下采样方法Gaussian‑Hermite Sampling (GHS)，实现CNN层级的严格平移不变性。

**💡 创新点**

利用二维高斯-厄米多项式作为正交基底，构造无参数的平移不变下采样，并证明在每层都能保证完整平移不变。

**🔧 技术方法**

采用高斯-厄米多项式、离散高斯-厄米矩、矩阵形式实现以及回归关系加速计算。

**📊 数据集**

在CIFAR‑10、CIFAR‑100以及旋转MNIST（MNIST‑rot）等标准数据集上进行实验。

**📈 对比分析**

与传统最大池化、低通滤波(LPF)和自适应多相采样(APS)对比；实验显示GHS在所有模型上都实现100%分类一致性，并在准确率上略优于APS、LPF，且在受扰动和旋转测试中表现更好。

**⚠️ 局限性**

主要缺陷是计算开销较大，尤其在大特征图上需要高阶高斯-厄米多项式的高阶阶乘计算，导致与APS相比成本约10倍。

---

## 367. Classifier Pooling for Modern Ordinal Classification

**arXiv ID:** 2603.17278 | [PDF](https://arxiv.org/pdf/2603.17278v1)

**作者:** Noam H. Rotenberg `[一作]` (Johns Hopkins University), Brian Caffo `[通讯]` (Johns Hopkins University)

**通讯引用:** 368329 | [OpenAlex ID](https://openalex.org/A5072858943)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了两种模型无关的序数分类方法（差分式和树状式），并实现为Python开源包statlab；

**💡 创新点**

创新点在于将任意二分类器包装为可处理序数标签的框架，并提供可调节阈值选择的超参数；

**🔧 技术方法**

技术上采用了阈值分割的累积/层次学习策略、概率约束融合与四折验证的最佳阈值选取；

**📊 数据集**

使用六个真实世界数据集，包括胎儿健康、汽车质量、葡萄酒质量、RetinaMNIST放射组学、MNIST放射组学和海蜗牛环数；

**📈 对比分析**

与传统多分类、序数logit/probit以及基线分类器比较，实验表明差分式和树状式在大多数数据集上获得更高的加权准确率、Polychoric相关和AUC，尤其在样本量小或类别多时优势更明显；

**⚠️ 局限性**

局限在于对深度学习适用性不足，且性能高度依赖数据集特性，阈值选择的最佳策略仍需进一步研究。

---

## 368. Joint Optimization of Storage and Loading for High-Performance 3D Point Cloud Data Processing

**arXiv ID:** 2603.16945 | [PDF](https://arxiv.org/pdf/2603.16945v1)

**作者:** Ke Wang `[一作]`, Yang Luo `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了统一的点云数据存储格式 .PcRecord 以及基于 MindSpore 的高性能多阶段并行处理管线，并结合 OBS 流式加载与分布式训练实现了大规模点云数据的高效读取、处理与训练。

**💡 创新点**

创新点包括：① 双层压缩（LZ4 + 差分编码）显著减小文件体积；② 统一文件头与元数据管理提升读取速度；③ 多阶段并行管线+顺序保证实现高吞吐量；④ 自适应 Auto‑tune 通过 Bayesian 优化动态调节线程数、队列大小等超参数；⑤ OBS 流式加载与多设备分布式调度消除本地存储瓶颈。

**🔧 技术方法**

核心技术包括：MindSpore 深度学习框架、C++/Python 混合实现的多线程 Connector、LZ4+差分压缩算法、OB‑S 对象存储接口、Bayesian 优化 + 随机搜索的 Auto‑tune、分布式数据并行与 All‑Reduce。

**📊 数据集**

在 ModelNet40、S3DIS、ShapeNet、KITTI、SUN RGB‑D、ScanNet 等六大公开点云数据集上进行实验，覆盖室内建模、物体分类、语义分割与自动驾驶等任务。

**📈 对比分析**

与 PyTorch、TensorFlow、Keras、MindSpore 的原生数据加载器比较，系统在 GPU 上实现 6.61x（ModelNet40）至 8.07x（SUN RGB‑D）加速，在 Ascend 上实现 6.9x（ModelNet40）至 25.4x（SUN RGB‑D）加速；分布式训练下多卡加速达到 38.05x（ModelNet40）等；吞吐量提升幅度从 4.46x 到 240.19x，显著突破传统框架瓶颈。

**⚠️ 局限性**

局限性包括：目前实现紧耦合于 MindSpore，尚未对其他深度学习框架原生支持；依赖 OBS 对象存储环境，离线单机大文件加载仍需额外存储；Auto‑tune 需要多次实验才能收敛，初始配置仍有经验性成分；在极大规模 100+GB 数据集时的元数据内存占用与网络延迟仍待进一步评估。

---

## 369. RangeAD: Fast On-Model Anomaly Detection

**arXiv ID:** 2603.17795 | [PDF](https://arxiv.org/pdf/2603.17795v1)

**作者:** Luca Hinkamp `[一作]` (TU Dortmund University), Emmanuel Müller `[通讯]` (Research Center Trustworthy Data Science and Security)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 On-Model AD 这一框架，利用已训练好的神经网络内部激活范围来实现无额外模型的异常检测算法 RangeAD，并在多种数据域（表格、图像、时间序列）上进行验证。

**💡 创新点**

创新点在于：①将异常检测嵌入主模型，直接利用训练得到的激活分布区间；②通过统计每个神经元的正常激活范围并计数越界情况作为异常分数，实现在推理阶段几乎无额外计算；③在多域实验中展示了在准确率和推理时间上均优于多种基线方法。

**🔧 技术方法**

技术包括：基于神经网络内部激活统计的区间构造（量化为分位数），阈值化异常分数，三阶段（训练、准备、推理）流程；使用的网络模型包括 MLP（表格）、Swin‑v2 Vision Transformer（图像）和 TCN+MLP（时间序列）。

**📊 数据集**

数据集：12 个表格数据集（如 BaseballEvents、FinanceJobCategories 等）、Imagenette + AstronomyImages + Ambivision（图像）、UrbanSound（时间序列）。

**📈 对比分析**

与 Autoencoder、Isolation Forest、DeepSVDD、COF、LOF 等多种传统与深度方法对比；在 AUC‑ROC 上 RangeAD 在大多数数据集上位居前列，平均 AUC‑ROC 高于对手 0.03；推理时间约 2 ms（表格）/ 0.014 s（图像/时间序列），是对手 50‑100 倍以内的显著加速。

**⚠️ 局限性**

局限性：依赖于主模型对正常数据的良好拟合；对于训练不足或任务与异常分布差异过大的情况，激活范围可能不足；区间选择（σ、t）对性能有影响，需调参；目前仅针对静态点异常，尚未深入扩展到持续流式或回归任务。

---

## 370. On Online Control of Opinion Dynamics

**arXiv ID:** 2603.17155 | [PDF](https://arxiv.org/pdf/2603.17155v1)

**作者:** Sheryl Paul `[一作]` (University of Southern California), Ketan Savla `[通讯]` (University of Southern California)

**通讯引用:** 2349 | [OpenAlex ID](https://openalex.org/A5007642972)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种在线控制框架，用于在网络化意见动态模型中同时估计未知的易感性参数并驱动意见向预设目标逼近。

**💡 创新点**

创新点在于将参数识别与控制交替进行，利用解析控制律在已知网络结构下给出收敛保证，并且明确给出了在预算或时间限制下实现目标误差的可行性条件。

**🔧 技术方法**

采用了Friedkin‑Johnsen 意见动态模型、Lyapunov 稳定分析、持久激励 (PE) 条件下的参数适应学习以及解析闭式控制更新。

**📊 数据集**

实验仅使用模拟的社交网络（随机/连通图）和人工生成的意见数据，未引入真实社交平台数据集。

**📈 对比分析**

与 IODSFC（基于数值优化的预算控制）和 NRS‑OFO（梯度投影控制）进行对比，实验结果表明在相同预算或时间约束下，本文方法收敛更快、误差更小，接近最优解。

**⚠️ 局限性**

局限性包括：只估计易感性参数，未同时识别网络结构；仅适用于标量统一目标；假设网络结构已知且成本为线性；在高维或非线性成本下的推广仍需进一步研究。

---

## 371. Full Stack Navigation, Mapping, and Planning for the Lunar Autonomy Challenge

**arXiv ID:** 2603.17232 | [PDF](https://arxiv.org/pdf/2603.17232v1)

**作者:** Adam Dai `[一作]` (Stanford University), Grace Gao `[通讯]` (Stanford University)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5069625302)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了完整模块化的月球表面自主导航系统，在Lunar Autonomy Challenge中实现了语义感知、立体视觉里程计、姿态图SLAM以及分层规划，完成了27 m×27 m区域的地形与岩石映射，并赢得第一名。

**💡 创新点**

创新点包括：基于学习的轻量语义分割与特征检测结合立体VO的全栈架构；在GNSS缺失、光照极端环境下的循环闭环策略；用半结构化路径规划鼓励频繁闭环并覆盖完整区域；以及开源实现促进复现。

**🔧 技术方法**

采用U‑Net++语义分割、SuperPoint+LightGlue特征提取/匹配、立体VO+PnP、GTSAM姿态图优化、Arc采样轨迹规划、LED光照补偿、LED灯控制等技术。

**📊 数据集**

使用Unreal Engine/CARLA模拟器提供的月球地图（三张：Map1、Map2和隐藏测试地图），以及基于模拟器生成的约5000张标注图像进行分割训练，随机生成的岩石分布与起始位置用于评估。

**📈 对比分析**

与其他参赛团队对比，在公开地图上获得RMSE约0.04‑0.06 m、地形映射得分>260、岩石检测F1>0.75，总分>820，最终在隐藏测试地图上位列第一，说明定位精度和映射质量均居领先。

**⚠️ 局限性**

限制包括：对光照极端时仍可能出现特征稀缺导致里程计漂移；路径规划仅基于静态分割的岩石检测，难以处理动态障碍；对未见地形的泛化依赖模拟器，真实月球环境可能更具挑战；以及单个视觉+IMU传感器方案对光照和纹理缺陷的鲁棒性仍有限。

---

## 372. Personalized Fall Detection by Balancing Data with Selective Feedback Using Contrastive Learning

**arXiv ID:** 2603.17148 | [PDF](https://arxiv.org/pdf/2603.17148v1)

**作者:** Awatif Yasmin `[一作]` (Texas State University), Anne H. H. Ngu `[通讯]` (Texas State University)

**通讯引用:** 7626 | [OpenAlex ID](https://openalex.org/A5016020974)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于选择性反馈与对比学习的个性化跌倒检测框架，利用半监督聚类与梯度筛选生成平衡训练集，并在三种重训练策略（从零训练、迁移学习、少样本学习）下评估个性化模型。

**💡 创新点**

通过Siamese网络进行半监督聚类与梯度优先样本挑选，有效缓解极度不平衡的反馈数据，显著提升召回率与整体F1，并验证在不同学习范式中的优势。

**🔧 技术方法**

Transformer跌倒检测模型、Siamese对比学习网络、DBSCAN聚类、梯度基样本筛选以及三种重训练策略。

**📊 数据集**

SmartFallMM 数据集（含跌倒与ADL）及10位参与者在真实使用中收集的反馈数据。

**📈 对比分析**

与使用全部不平衡反馈训练的基线模型比较，选择性样本训练的TFS模型平均F1从0.73提升至0.91（≈25%提升），FSL提升至0.78（≈7%），TL表现最差；实时评测中TFS平均F1 0.91，FSL 0.78，基线0.73。

**⚠️ 局限性**

训练从零成本高、时延长；迁移学习效果不佳；对极少或单一类型反馈的数据聚类可能失效；在极端不平衡场景下仍需进一步改进。

---

## 373. Proof-of-Authorship for Diffusion-based AI Generated Content

**arXiv ID:** 2603.17513 | [PDF](https://arxiv.org/pdf/2603.17513v1)

**作者:** De Zhang Lee `[一作]` (National University of Singapore), Ee-Chien Chang `[通讯]` (National University of Singapore)

**通讯引用:** 4639 | [OpenAlex ID](https://openalex.org/A5105408906)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对LDM（潜在扩散模型）生成内容的证明作者身份（Proof‑of‑Authorship）框架。

**💡 创新点**

创新点在于不依赖任何秘密，利用伪随机函数将作者身份与生成参数绑定，形成可公开验证且在对抗伪造者时具有概率上强保证的身份证明。

**🔧 技术方法**

核心技术包括 HMAC‑SHA3/256 等伪随机函数、概率裁判（Probabilistic Adjudicator）进行置信区间估计、子指数分布假设以及对 Stable Diffusion 2.1 的实验验证。

**📊 数据集**

使用 Stable Diffusion Prompt 数据集与 Stable Diffusion 2.1 公开模型进行实验，涉及多种扰动（高斯噪声、JPEG 压缩、仿射变换）。

**📈 对比分析**

通过对争议对象与重生成对象的相似度分布进行统计检验，实验显示在不同失真条件下随机伪造者成功率均低于 2^-50，误判率几乎为零，证明该方法在实用场景下具有高可靠性。

**⚠️ 局限性**

局限性包括：需要假设 LDM 输出的相似度分布为子指数分布；若模型存在后门或记忆化导致该假设失效，验证可能失败；对不同 LDM 结构或更强攻击的适用性仍待进一步研究。

---

## 374. The Causal Uncertainty Principle: Manifold Tearing and the Topological Limits of Counterfactual Interventions

**arXiv ID:** 2603.17385 | [PDF](https://arxiv.org/pdf/2603.17385v1)

**作者:** Rui Wu `[一作]` (University of Science and Technology of China), Yongjun Li `[通讯]` (University of Science and Technology of China)

**通讯引用:** 18753 | [OpenAlex ID](https://openalex.org/A5100376886)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究连续空间因果干预的几何与测度极限，定义“Counterfactual Event Horizon”和“Manifold Tearing”，并提出Geometry-Aware Causal Flow（GACF）解决方案，实现在强干预下保持个体身份与拓扑安全。

**💡 创新点**

①首次用测度理论与Riemannian几何给出连续do-操作的能量边界和事件地平线；②证明确定性流在超强干预下会出现有限时奇异性（Manifold Tearing）；③提出Causal Uncertainty Principle，量化干预强度与身份保留的不可避免权衡；④设计GACF通过Hutchinson追踪器实时注入最小熵，兼顾身份与拓扑安全。

**🔧 技术方法**

最优传输、Schrödinger桥、Hessian比较定理、Raychaudhuri/riccati方程、Bakry‑Émery Γ2、Talagrand T₂、Hutchinson trace estimator、流式ODE/SDE模型、数值仿真。

**📊 数据集**

高维神经流实验以及真实PBMC 3k单细胞RNA‑seq数据。

**📈 对比分析**

与纯ODE和固定熵SDE对比；GACF在保持身份的同时避免Manifold Tearing，方差由0.893降至0.372，且在高维度和大干预距离下仍保持稳定；实验显示t_c按1/D比例下降，GACF能提前触发并提供足够窗口。

**⚠️ 局限性**

仅针对单一连续do操作；理论基于理想化Riemannian几何与测度假设；对多因果网络、离散-连续混合情形及真实数据几何估计仍需进一步研究。

---

## 375. Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation

**arXiv ID:** 2603.17315 | [PDF](https://arxiv.org/pdf/2603.17315v1)

**作者:** Chunxu Zhang `[一作]` (Jilin University), Bo Yang `[通讯]` (Jilin University)

**通讯引用:** 72015 | [OpenAlex ID](https://openalex.org/A5072820962)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并实现了一种联邦持续学习推荐框架，使每个客户端在不共享原始数据的前提下，长期保留个性化偏好并动态适应用户行为变化。

**💡 创新点**

核心创新点包括：① 时间感知自蒸馏策略，用前一时刻的表示对当前表示进行约束，缓解时间遗忘；② 交叉用户原型转移机制，服务器维护动态原型库，客户端检索相似原型并融合，既实现协同个性化，又保持个体决策逻辑。

**🔧 技术方法**

技术手段：联邦学习 + 持续学习；Transformer（作为表示模块）+ MLP（预测模块）序列模型；MSE 自蒸馏损失；语义相似度（余弦相似度）+ 原型融合；FedAvg 聚合；实验中采用SASRec、GRU4Rec等骨干。

**📊 数据集**

实验使用四个公开业务数据集：XING、RetailRocket、Tmall、LastFM，分别涵盖招聘、电子商务、音乐推荐等场景。

**📈 对比分析**

与多种基线进行对比：联邦推荐基线（PFedRec、FedRAP、GPFedRec）、会话/序列推荐基线（Fed-DMI-GNN、Fed-HearInt、Fed-MiaSRec、Fed-GRU4Rec、Fed-SASRec、Fed-TiSASRec）。实验结果显示，在 HR@10 和 NDCG@10 上，本文方法均显著优于所有基线，尤其在长期适应和协同个性化方面表现突出。

**⚠️ 局限性**

限制与挑战：① 对客户端算力和存储的要求仍高，需要进一步优化原型库规模与传输频率；② 在极度异构的数据分布下的鲁棒性未完全验证；③ 隐私保护方面需结合差分隐私或安全多方计算进行更细粒度的评估。

---

## 376. Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion

**arXiv ID:** 2603.16939 | [PDF](https://arxiv.org/pdf/2603.16939v1)

**作者:** Aislan Gabriel O. Souza `[一作]` (Universidade de Pernambuco), Luciana Machado `[通讯]` (Universidade de Pernambuco)

**通讯引用:** 934 | [OpenAlex ID](https://openalex.org/A5103923024)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究Ambivalence/Hesitancy视频识别挑战，提出基于跨模态差异的融合方法

**💡 创新点**

通过计算视觉、音频、文本模态嵌入的绝对差异显式建模跨模态冲突，并利用AU时间统计捕获面部不稳定性

**🔧 技术方法**

使用Py-Feat提取AU、Wav2Vec 2.0音频、BERT文本、BiLSTM+注意力时序建模、绝对差异融合、MLP分类以及XGBoost基线

**📊 数据集**

使用BAH数据集（1,132视频，239人）

**📈 对比分析**

与传统串联/融合Baseline相比，Fusion B在验证集达0.6912、测试集达0.6808 Macro F1，显著优于Baseline 0.2827及单模态

**⚠️ 局限性**

受限于训练样本量，丰富特征提升验证但未提升测试；缺乏跨模态时间对齐，未尝试更多视觉表征

---

## 377. Halo: Domain-Aware Query Optimization for Long-Context Question Answering

**arXiv ID:** 2603.17668 | [PDF](https://arxiv.org/pdf/2603.17668v1)

**作者:** Pramod Chunduri `[一作]` (Amazon Web Services), Joy Arulraj `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5060680349)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文实现了自动从用户提示中提取领域知识，将其转换为结构化指令，并在多阶段长文本问答管道中按阶段应用结构化操作符，以提升检索质量、过滤无关内容并校正答案。

**💡 创新点**

创新点在于：①将领域知识系统化拆分为结构、过滤、验证三类指令；②设计对应的操作符在文档预处理、chunk 过滤和答案重排三阶段分别发挥作用；③引入回退机制以检测并抑制低质量指令带来的准确性下降。

**🔧 技术方法**

所采用技术包括：大型语言模型（Claude Sonnet 4.5、Qwen 3‑30B）与轻量级 4B 参数 SLM；嵌入检索模型 bge‑m3；结构化 PDF 解析；指令提取 LLM；验证分数排名与回退管理。

**📊 数据集**

实验数据集覆盖金融、文学和科学三大领域，分别为：DocFinQA（SEC 10‑K 财务文件）、NovelQA（小说长文本）和 QASPER（学术论文）。

**📈 对比分析**

与单调用 LLM、RAG、提示注入、Chain‑of‑Agents、无域知识管道等基线相比，Halo 在准确率上提升 4–13%，成本降低 1.8–4.8×，甚至可让轻量模型在 78× 成本下降后逼近前沿 LLM 的性能。

**⚠️ 局限性**

局限性包括：对指令提取准确性的高度依赖；当结构/过滤/验证指令与文档不匹配时可能导致答案被误删；回退机制会产生额外的算力和费用；对极端长文本或缺乏可识别结构的文档效果仍有限。

---

## 378. Integrating Inductive Biases in Transformers via Distillation for Financial Time Series Forecasting

**arXiv ID:** 2603.16985 | [PDF](https://arxiv.org/pdf/2603.16985v1)

**作者:** Yu-Chen Den `[一作]` (SinoPac Holdings), Darby Tien-Hao Chang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出TIPS框架，利用注意力掩码训练多种先验（因果性、局部性、周期性）的Transformer教师，并通过正则化蒸馏将它们的知识合成到单一学生模型中，以提升金融时间序列预测的鲁棒性与效率。

**💡 创新点**

创新点在于：①将多种时序先验仅通过注意力掩码编码，保持教师网络结构统一；②使用温度蒸馏、标签平滑与随机权重平均等正则化手段，避免学生对单一教师过度拟合，从而实现单模型内的时变先验激活；③克服“融合惩罚”，在保持低推理成本的同时达到或超过多模型集成的性能。

**🔧 技术方法**

核心技术包括：注意力掩码与位置偏置（ALiBi、周期性偏置）、多教师蒸馏（低温度蒸馏+标签平滑）、随机权重平均（SWA）以及用于回报预测的可微Spearman相关损失。

**📊 数据集**

实验数据集覆盖四大股市（CSI300、CSI500、NI225、SP500），每个市场使用OHLCV及多周期移动平均等八维特征，训练/测试时间跨度为2021–2024，涵盖牛市、熊市与高波动期。

**📈 对比分析**

与通用时序Transformer、经典CNN/RNN、金融专用模型等7类基线进行5个随机种子实验，评估指标为年化收益、Sharpe比率与Calmar比率。TIPS在四市均实现平均年化收益提升约55%、Sharpe提升9%、Calmar提升16%，且推理计算仅为基线集成的38%。

**⚠️ 局限性**

局限性：仅针对单一回报预测任务验证；未探讨中间表示蒸馏、跨域迁移与交易成本等实际场景；对极端突发市场变化的鲁棒性尚需进一步评估。

---

## 379. Sensi: Learn One Thing at a Time -- Curriculum-Based Test-Time Learning for LLM Game Agents

**arXiv ID:** 2603.17683 | [PDF](https://arxiv.org/pdf/2603.17683v1)

**作者:** Mohsen Arjmandi `[一作]` `[通讯]` (Independent Researcher), Mohsen Arjmandi (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了名为Sensi的LLM代理架构，用于ARC‑AGI‑3游戏挑战，强调在测试时学习、通过结构化学习机制提升样本效率；

**💡 创新点**

创新点包括：1）将感知与行动分离为两玩家架构；2）引入基于外部状态机的课程学习；3）使用数据库作为控制面板实现可编程上下文；4）利用LLM作为判断者动态生成评价指标；5）实现了50–94×的样本效率提升；

**🔧 技术方法**

技术主要涉及：多调用LLM（Observer/Actor、FrameDiff、MetricGen、SenseScore、Player1/2）、SQLite数据库控制平面、DSPy框架、以及自生成评估度量和感知评分；

**📊 数据集**

使用的数据集为ARC‑AGI‑3游戏环境的像素艺术关卡（LS20等），以及公开的基准数据如Agentica/Symbolica所用的交互数据；

**📈 对比分析**

比较方法：将Sensi v1、v2与随机代理、Agentica（Symbolica）以及基线Sensi v1进行对比，评估指标为样本效率（交互次数）与通过率；结果显示Sensi v2在约32次交互内完成全部课程，显著优于需要1,600–3,000次交互的Baseline；

**⚠️ 局限性**

局限性：1）Sensi v2未能赢得任何游戏级别；2）课程设计人为，缺乏自动化；3）不同版本使用不同LLM导致直接比较受限；4）评估器基于内部一致性，易受自洽幻觉影响；5）感知层框架（帧差分）是主要瓶颈，导致错误知识堆叠。

---

## 380. LLM Use, Cheating, and Academic Integrity in Software Engineering Education

**arXiv ID:** 2603.17060 | [PDF](https://arxiv.org/pdf/2603.17060v1)

**作者:** Ronnie de Souza Santos `[一作]` (University of Calgary), Mairieli Wessel `[通讯]` (Radboud University)

**通讯引用:** 633 | [OpenAlex ID](https://openalex.org/A5032291051)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对软件工程本科生使用大型语言模型（LLM）进行“作弊”行为的体验进行问卷调查与访谈分析，探讨其行为动机、情境与后果。

**💡 创新点**

创新点在于聚焦学生主观叙事而非单纯统计作弊频率，揭示工作负荷、评估设计与指导模糊度如何共同促成 LLM 违规使用。

**🔧 技术方法**

主要技术为跨学科混合方法：结构化问卷（闭合与开放题）配合定性主题分析；使用 Qualtrics 平台收集数据并采用编码法归纳主题。

**📊 数据集**

数据集为 116 名来自多国的软件工程本科生自我报告的问卷结果，涵盖不同学年、性别与机构背景。

**📈 对比分析**

本文未采用算法性能比较；通过定性描述与描述性统计呈现 LLM 违规频率、情境分布及情感反应，说明在编程作业、作业及文档任务中违规最为常见。

**⚠️ 局限性**

局限性包括非概率抽样与自我报告偏差、对 LLM 具体模型的技术细节未深入、缺乏因果推断与跨学科验证，结果仅具可转移性而非统计普适性。

---

## 381. Integrating Explainable Machine Learning and Mixed-Integer Optimization for Personalized Sleep Quality Intervention

**arXiv ID:** 2603.16937 | [PDF](https://arxiv.org/pdf/2603.16937v1)

**作者:** Mahfuz Ahmed Anik `[一作]`, MD Manjurul Ahsan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

建立了一个结合可解释机器学习与混合整数优化的个性化睡眠质量干预框架，先用XGBoost预测睡眠质量，再用SHAP解释重要因素，并将其作为权重在MILP中求解最小可行行为调整方案。

**💡 创新点**

创新点在于将SHAP解释结果直接嵌入优化模型并引入行为阻力惩罚，形成“最小变化”原则，兼顾预测精度、可解释性与干预可行性；同时在同一框架内完成预测、解释与处方三步。

**🔧 技术方法**

使用的技术包括XGBoost梯度提升树、SHAP模型解释、线性混合整数规划（MILP）求解器（CBC）。

**📊 数据集**

使用的数据集为418名大学生的问卷调查（PSQI、行为、环境等变量），通过数据增强后得到1339个样本。

**📈 对比分析**

与多种树模型（LightGBM、GB、ExtraTrees、RF、MLP）比较，XGBoost在测试集上取得F1=0.9544、准确率0.9366；干预方案通过敏感度和Pareto分析验证，平均推荐1–2项，必要时不推荐干预，表现出良好的稀疏性和实用性。

**⚠️ 局限性**

局限性包括：仅基于自报问卷，缺乏客观睡眠监测；模型仅允许单步阶梯调整，未捕捉持续性和个体SHAP差异；未进行实验验证，无法确认预测改进是否在实际干预后实现。

---

## 382. SENSE: Efficient EEG-to-Text via Privacy-Preserving Semantic Retrieval

**arXiv ID:** 2603.17109 | [PDF](https://arxiv.org/pdf/2603.17109v1)

**作者:** Akshaj Murhekar `[一作]` (University of Texas at Austin), Jacek Gwizdka `[通讯]` (University of Texas at Austin)

**通讯引用:** 3828 | [OpenAlex ID](https://openalex.org/A5050316482)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了SENSE框架，将非侵入式EEG信号解码为自然语言，分为设备端语义检索与prompt式文本生成两阶段。

**💡 创新点**

通过轻量级EEG‑to‑keyword模块与CLIP文本空间对齐，避免LLM微调，实现隐私友好、可在设备上运行的检索‑生成架构。

**🔧 技术方法**

使用ChannelNet EEG编码器、Similarity Refiner MLP、CLIP对齐、top‑k Bag‑of‑Words检索以及零射击提示的LLM生成；训练时采用Focal/CE/Contrastive多标签损失。

**📊 数据集**

基于公开的CVPR2017 EEG‑ImageNet 128通道EEG数据集（6名受试者、2000幅图像+字幕）。

**📈 对比分析**

与全微调Baseline（Thought2Text、Naive等）以及多种LLM（ChatGPT‑4o‑mini、Gemini、LLaMA3、Qwen）比较，使用BLEU/ROUGE/METEOR/BERTScore和GPT‑5评估；SENSE在大多数指标上与Thought2Text相当甚至更好（例如ROUGE‑1 31.5 vs 30.0），并显著降低计算和存储成本。

**⚠️ 局限性**

受限于EEG低空间分辨率和高噪声，语义检索仍可能出现混淆；词表固定限制词汇多样性；跨受试者差异导致性能波动；依赖外部LLM API，隐私与信任仍是挑战。

---

## 383. Few-Step Diffusion Sampling Through Instance-Aware Discretizations

**arXiv ID:** 2603.17671 | [PDF](https://arxiv.org/pdf/2603.17671v1)

**作者:** Liangyu Yuan `[一作]` (Tongji University), Chi Zhang `[通讯]` (Westlake University)

**通讯引用:** 26450 | [OpenAlex ID](https://openalex.org/A5100458183)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种实例感知的时间步长离散化框架（INDIS），通过训练轻量化网络根据初始噪声与条件信息动态生成适配的时间步长，从而在有限 NFE 的 diffusion/flow-matching 模型采样中提升生成质量。

**💡 创新点**

创新点在于打破传统的全局统一时间步长策略，首次将离散化步骤作为可学习的实例特定参数，并结合条件引导与曝光偏差校正，使得在多种数据与模型上实现了显著的 FID/CLIP/WS 等指标提升。

**🔧 技术方法**

采用概率流 ODE 与多步数值求解器（如 iPNDM、UniPC 等）为基础，利用梯度搜索对离散化参数进行优化，并使用 LPIPS/FID 等距离度量作为监督信号。

**📊 数据集**

实验数据集涵盖 CIFAR‑10、ImageNet、FFHQ、AFHQv2（像素空间），Stable Diffusion 与 Flux（潜在空间），以及 LTX‑Video（视频），并对多种公开预训练模型进行评估。

**📈 对比分析**

与手工启发式、全局优化（GOD）、DMN、GITS、LD3 等基线对比，INDIS 在 3‑7 NFE 设定下平均 FID 下降 15–35%，CLIP 得分提升 10–20%，显示出在少步采样场景的显著性能优势。

**⚠️ 局限性**

局限性包括对梯度检查点的依赖导致额外内存与计算开销，且在高步长（NFE）或极大模型下实例化离散化的收益减弱；未来可探讨更高效的梯度传播与自适应步长策略。

---

## 384. A Proposal-Free Query-Guided Network for Grounded Multimodal Named Entity Recognition

**arXiv ID:** 2603.17314 | [PDF](https://arxiv.org/pdf/2603.17314v1)

**作者:** Hongbing Li `[一作]` (Beijing University of Posts and Telecommunications), Bo Xiao `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 5343 | [OpenAlex ID](https://openalex.org/A5041909892)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种无候选框的查询引导网络（QGNet）用于地面多模态命名实体识别（GMNER），通过文本引导查询、层次视听对齐和统一多模态解码实现实体识别与视觉定位；

**💡 创新点**

创新点包括：①文本引导查询精炼机制，使查询携带实体语义；②多层次视觉-语言对齐（HVLA）在不同语义粒度上融合视觉与文本；③统一解码空间让查询在多模态特征中迭代交互，增强实体-视觉关联；

**🔧 技术方法**

采用预训练 CLIP ViT-B/32 作为视觉编码器、BERT 作为文本编码器、Transformer 查询注意力、Hungarian 匹配及多头预测头；

**📊 数据集**

在 Twitter-GMNER 与 Twitter-FMNERG 两个公开数据集上进行评估；

**📈 对比分析**

与多种基线（MNER-QG、MQSPN 等）对比，QGNet 在 GMNER、MNER 与 EEG 子任务上均取得最高 F1 分数，FMNERG 上在细粒度类别上显著提升；

**⚠️ 局限性**

限制：仍依赖 Transformer 计算资源，对大规模数据的训练与推理成本较高，且对极细粒度视觉信息的捕捉仍有提升空间。

---

## 385. Collecting Prosody in the Wild: A Content-Controlled, Privacy-First Smartphone Protocol and Empirical Evaluation

**arXiv ID:** 2603.17061 | [PDF](https://arxiv.org/pdf/2603.17061v1)

**作者:** Timo K. Koch `[一作]` (University of St. Gallen), Clemens Stachl `[通讯]` (University of St. Gallen)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发并在大规模真实环境中评估了一个内容受控、隐私优先的智能手机语音采集协议，使用脚本朗读句子并在设备端提取韵律特征、即时删除原始音频。

**💡 创新点**

创新点在于把实验室级别的词汇控制和情感平衡语料与移动端隐私保护相结合，实现可扩展的现场韵律数据采集，并提供完整的端到端隐私保护链。

**🔧 技术方法**

技术上使用Android原生OpenSMILE（eGeMAPS和ComParE特征集）、线性PCM音频、SSL加密传输、随机森林分类/回归进行验证。

**📊 数据集**

使用了德国受试者的读音数据，共560名受试者、9,877条录音，包含54句验证过的正、负、中性句子。

**📈 对比分析**

通过随机森林对性别的预测达到约92%平衡准确率；对情感的预测（情绪值、激活度）仅获得ρ≈0.11–0.15，表现较弱；未发现不同句子情绪条件对预测有显著影响。

**⚠️ 局限性**

局限包括：受试者可能未严格朗读原句，缺乏原始音频以验证；所用手工特征在表达力上可能低于自监督嵌入；录音条件多变可能削弱韵律信号。

---

## 386. AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception

**arXiv ID:** 2603.17979 | [PDF](https://arxiv.org/pdf/2603.17979v1)

**作者:** Jinho Park `[一作]` (Columbia University), Mingoo Seok `[通讯]` (Columbia University)

**通讯引用:** 5606 | [OpenAlex ID](https://openalex.org/A5011887658)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出AdaRadar，一种基于频谱的自适应雷达数据压缩框架，能将范围-多普勒体积压缩至 100+ 倍，同时保持检测与分割性能。

**💡 创新点**

创新点：① 在频域做稀疏剪枝；② 用置信度驱动的零阶梯度估计实现在线自适应压缩率；③ 只需前向推理，无需反向传播；④ 兼容多种雷达感知任务，具有很强的通用性。

**🔧 技术方法**

技术方法包括离散余弦变换（DCT）、自适应谱剪枝、块级量化、零阶梯度估计、置信度驱动的梯度下降、在线反馈控制。

**📊 数据集**

使用 RADIal、CARRADA、Radatron 三个公开雷达数据集，分别涵盖目标检测和语义分割任务。

**📈 对比分析**

与未压缩基线和索引值压缩对比，AdaRadar 在 101×-117× 的压缩比下，检测 AP/AR 仅损失约 1%（甚至在某些指标上提升），分割 mIoU 仅降 1%；总体率-精度曲线明显优于基线。

**⚠️ 局限性**

局限性：仅针对雷达数据，未考虑图像压缩；若后续帧的时间连续性受限，在线自适应可能失稳；依赖硬件支持 DCT 与量化的低功耗实现。

---

## 387. EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards

**arXiv ID:** 2603.17808 | [PDF](https://arxiv.org/pdf/2603.17808v1)

**作者:** Ruixiang Wang `[一作]` (Chinese University of Hong Kong), Kui Jia `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 12685 | [OpenAlex ID](https://openalex.org/A5065964089)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对视频世界模型在机器人控制中的可执行性缺口，提出了执行视频对齐（EVA）框架，通过基于逆动力学模型的奖励对视频生成器进行强化学习后训练。

**💡 创新点**

利用逆动力学模型生成的动作序列作为奖励信号，直接在动作空间对视频生成进行可执行性对齐，克服了视觉与动作不一致的缺口。

**🔧 技术方法**

流匹配流模型（flow‑matching）视频生成器、GRPO强化学习、逆动力学模型、LoRA微调等技术。

**📊 数据集**

RoboTwin 2.0 仿真数据集以及真实双臂机器人上收集的 50 条演示数据。

**📈 对比分析**

与 Vidar、ACT、DP、RDT、π_0、GE-Act 等基线在 RoboTwin 任务和真实机器人上对比，EVA 在可执行性、任务成功率和人类评价中分别提升约20–30% 以上，显著优于未对齐模型。

**⚠️ 局限性**

奖励仅关注关节平滑和边界约束，未考虑接触动力学；生成速度慢，难以实现实时闭环控制。

---

## 388. Operator-Theoretic Foundations and Policy Gradient Methods for General MDPs with Unbounded Costs

**arXiv ID:** 2603.17875 | [PDF](https://arxiv.org/pdf/2603.17875v1)

**作者:** Abhishek Gupta `[一作]` (Ohio State University), Aditya Mahajan `[通讯]` (McGill University)

**通讯引用:** 2731 | [OpenAlex ID](https://openalex.org/A5005770424)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了将一般马尔可夫决策过程（MDP）视为在函数空间上对线性算子进行优化的新框架，并在此基础上证明了在一般状态和动作空间下存在最优策略，随后推导出政策差分引理、重要的主导化边界，并基于此设计了类似PPO的低复杂度算法；在离散GARNET环境上通过实验验证了新算法相较于传统PPO的收敛速度提升。

**💡 创新点**

创新点包括：
1) 采用线性算子与泛函分析（微扰理论）统一处理无穷维状态/动作空间；
2) 在该框架下首次给出一般MDP的政策差分引理与梯度（Gateaux）表达式；
3) 推导出主导化二阶项上界，形成通用的主导化最优化（MM）算法，涵盖TRPO、PPO等；
4) 引入积分概率度量（IPM / MMD）替代KL散度，简化第二阶项计算；
5) 对有限MDP提供RKHS版本的MM算法并给出闭式更新。

**🔧 技术方法**

主要技术手段有：
- 线性算子理论与微扰理论
- 积分概率度量与MMD
- Gateaux导数与政策差分引理
- 主导化最优化框架
- RKHS与高斯核的运用
- 结构化策略空间与可测选择定理

**📊 数据集**

实验采用的公开/自构数据集：
- 随机生成的GARNET环境（1000个状态、50个动作、折扣因子0.9、分支因子3）
- 仅在离散环境下进行模拟，未使用实际工业或真实物理数据集。

**📈 对比分析**

比较方法：将新提出的MM‑RKHS算法与标准PPO进行对比。性能表现：
- 在确定性版（已知转移/奖励）中，MM‑RKHS收敛速度显著快于PPO；
- 在采样版（仅通过轨迹更新）中，MM‑RKHS同样表现出更快的学习曲线；
- 由于MM‑RKHS使用二阶主导化，可视为接近Newton法，收敛率优于纯梯度（PPO）方法。

**⚠️ 局限性**

局限性与挑战：
- 需要满足一系列结构性假设（如算子紧性、IPM可分离点、可测选择等），在实际问题中可能不易满足；
- 对连续/无穷维状态/动作空间的实际实现仍缺乏系统的数值方法与实验验证；
- 算法对模型参数（如MMD核矩阵R）敏感，选择不当会影响收敛；
- 目前仅在离散GARNET环境中测试，缺乏对真实复杂控制任务（如流体控制、软体机器人）的实证；
- 需要进一步研究如何在高维或大规模问题中高效计算IPM及其梯度。

---

## 389. HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning

**arXiv ID:** 2603.17024 | [PDF](https://arxiv.org/pdf/2603.17024v1)

**作者:** Shenzhi Wang `[一作]` (Alibaba Inc), Junyang Lin `[通讯]` (Alibaba Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出HopChain框架，用于大规模合成多跳视觉-语言推理数据，并在RLVR训练中提升VLM的长链式推理能力。

**💡 创新点**

创新点在于将推理拆分为感知级跳和实例链跳，构造逻辑依赖的多跳链路，强制模型在每一步持续视觉检索；且该数据生成流程无目标基准，具备可扩展性和广泛迁移性。

**🔧 技术方法**

采用Qwen3‑VL进行类别识别和查询生成、SAM3进行实例分割、RLVR与SAPO算法进行强化学习、人工验证与难度校准等技术；整体形成四阶段流水线。

**📊 数据集**

原始RLVR数据（多模态问答、算术、文本识别、视频理解等）与HopChain合成的多跳数据；在24个公开基准上进行评测，包括MathVision、MMMU、MMBench、DocVQA、VideoMME等。

**📈 对比分析**

通过将原始RLVR与HopChain数据混合训练，与仅使用原始RLVR的基线模型在Qwen3.5-35B-A3B与Qwen3.5-397B-A17B上对比；结果显示在20/24项基准上均有提升，尤其在超长链推理时可提升50+点；单跳或半跳对比实验表明完整多跳结构最优。

**⚠️ 局限性**

当前流程依赖成功的实例分割，无法处理无可分割对象或极少对象的图像；因此部分图像被排除，限制了数据覆盖范围。

---

## 390. Shot-Aware Frame Sampling for Video Understanding

**arXiv ID:** 2603.17374 | [PDF](https://arxiv.org/pdf/2603.17374v1)

**作者:** Mengyu Zhao `[一作]` (ByteDance), Yong Cao `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种任务无关、基于信息论的拍照框架InfoShot，用于在有限帧预算下对长视频进行采样。

**💡 创新点**

创新点在于将视频划分为语义一致的镜头，并为每个镜头挑选两张互补关键帧（代表性帧与高偏差帧），从而在不增加训练负担的前提下同时覆盖全局语义与短暂关键事件。

**🔧 技术方法**

使用深度特征空间的余弦相似度进行GLRT式的滑动窗口镜头分割；随后通过典型性与局部波动得分选取两帧；整个流程基于信息论目标拆解实现。

**📊 数据集**

在自研的SynFlash合成Video‑QA基准（包含4类可控短时异常）以及真实异常检测数据集HIVAU‑70k、UCF‑Crime、XD‑Violence和通用视频推理基准Video‑MME上进行评估。

**📈 对比分析**

与均匀采样、VSUMM、TransNet等基线相比，InfoShot在0.5 fps（以及更低帧率）下的异常召回率提升≈70‑90 %并显著提高Video‑QA准确率；在HIVAU‑70k的事件召回与完整覆盖率上也保持或略优于对比方法，且在Video‑MME上保持竞争性能。

**⚠️ 局限性**

局限性在于固定的两帧/镜头分配在镜头长度或事件频率变化极端时可能不够灵活；此外仅利用视觉特征而未加入运动或时序信号，可能在极短事件或快速运动场景中仍存在漏检风险。

---

## 391. What on Earth is AlphaEarth? Hierarchical structure and functional interpretability for global land cover

**arXiv ID:** 2603.16911 | [PDF](https://arxiv.org/pdf/2603.16911v1)

**作者:** Ivan Felipe Benavides-Martinez `[一作]`, Auroop R. Ganguly `[通讯]` (Northeastern University)

**通讯引用:** 7892 | [OpenAlex ID](https://openalex.org/A5064658255)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并验证了一套功能可解释框架，利用大规模二分类实验和特征重要性分析，揭示GAEF嵌入空间在土地覆盖分类中的层次化功能组织。

**💡 创新点**

创新点在于将嵌入维度按专属性、低/中/高通用度进行层次化功能分类，并证明仅需2-12维即可恢复98%基线性能，展示嵌入空间的显著冗余与结构化。

**🔧 技术方法**

采用大规模实验（13万+二分类任务）、随机森林、梯度提升、XGBoost、LightGBM、MDI重要性、渐进消融、关联矩阵、嵌入指纹图以及交互式可视化仪表板。

**📊 数据集**

使用ESA WorldCover 2020 10 m 全球土地覆盖标签和对应的GAEF 64维嵌入。

**📈 对比分析**

通过与全维基线性能比较，确定每个土地覆盖类别达到≥98%基线所需的最小维数，分类准确率保持在98%+；消融曲线显示前几维贡献显著，后续维数增益递减，验证信息高度集中。

**⚠️ 局限性**

局限性包括：解释依赖特征重要性和阈值，可能受算法偏差影响；未检验在不同地区、时间或其他任务的稳定性；21个维度未被解释；仅基于土地覆盖标签，未关联物理变量；高维嵌入的空间关系被简化为可视化近似。

---

## 392. Beyond Outliers: A Data-Free Layer-wise Mixed-Precision Quantization Approach Driven by Numerical and Structural Dual-Sensitivity

**arXiv ID:** 2603.17354 | [PDF](https://arxiv.org/pdf/2603.17354v1)

**作者:** Hengyuan Zhang `[一作]` (University of Hong Kong), Ngai Wong `[通讯]` (University of Hong Kong)

**通讯引用:** 12262 | [OpenAlex ID](https://openalex.org/A5043990959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无需校准的层级混合精度量化框架（NSDS），通过对每层的检测器与写入器两类操作模块进行数值和结构双重敏感性评估，生成层级敏感度并指导比特分配；

**💡 创新点**

创新点在于：①机制化拆解层为检测器与写入器两角色；②引入数值方差（超峰度）与结构表达（谱幅度+谱熵）双重敏感度；③采用基于中位数绝对偏差的鲁棒归一化与概率聚合方式融合两项指标，生成统一层级敏感度；

**🔧 技术方法**

使用了超峰度、奇异值分解（SVD）、谱熵、角色加权重、MAD归一化、Sigmoid映射、逻辑乘积聚合、目标比特预算分配等技术；

**📊 数据集**

在 WikiText-2、C4 进行语言建模；ARC‑Challenge、HellaSwag、PIQA、BoolQ、WinoGrande、TruthfulQA 进行推理评测；

**📈 对比分析**

与多种无校准/有校准的层级量化基线（MSE、EWQ、ZD、KurtBoost、LIM、LSAQ、LLM‑MQ、LieQ、SliM‑LLM）进行对比，NSDS 在 Llama‑3.1‑8B、Qwen2.5‑7B 等模型上在语言建模和推理任务中均实现了明显的准确率/困惑度提升（如在 PIQA 上提升超过3%）；

**⚠️ 局限性**

仅在 2B–14B 规模模型上验证，未对 30B/70B 等更大模型进行评估，且对模型结构变化的普适性仍需进一步研究。

---

## 393. Vectorization of Verilog Designs and its Effects on Verification and Synthesis

**arXiv ID:** 2603.17099 | [PDF](https://arxiv.org/pdf/2603.17099v1)

**作者:** Maria Fernanda Oiveira Guimarães `[一作]`, Fernando Magno Quintão Pereira `[通讯]` (Federal University of Minas Gerais)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

开发了一种在 Verilog 设计中进行向量化的源到源编译器转换器；

**💡 创新点**

首次在 Verilog 生态中实现空间向量化，恢复字节级结构并显著降低符号复杂度，同时保持语义等价；

**🔧 技术方法**

基于 CIRCT 框架，使用位级数据流分析、结构分析、选择性模块内联和分块向量化等静态分析技术；

**📊 数据集**

在 ChiBench 集合的 1,157 个可向量化 Verilog 设计上进行实验；

**📈 对比分析**

将向量化结果与 Jasper Formal Verification Platform 和 Genus Synthesis Solution 进行对比，Jasper 的 RTL 详细化时间缩短 28.12%，内存消耗下降 51.30%；Genus 的详细化平均提升 5.49%，部分基准内存下降 75%；整体指令数平均下降 47.8%；

**⚠️ 局限性**

仅在存在可向量化模式的设计上受益，对非连续位访问的模块会导致指令数增加；缺乏成本模型和内联阈值的动态调整，可能影响向量化收益与设计尺寸之间的权衡。

---

## 394. CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent Attention

**arXiv ID:** 2603.17946 | [PDF](https://arxiv.org/pdf/2603.17946v1)

**作者:** Zhongzhu Zhou `[一作]` (University of Sydney), Shuaiwen Leon Song `[通讯]` (University of Sydney)

**通讯引用:** 2764 | [OpenAlex ID](https://openalex.org/A5043209884)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出一种将预训练的注意力模块（如GQA）转换为多头潜在注意力（MLA）的技术，保持KV缓存不变并提升表达能力。

**💡 创新点**

创新点在于引入协方差感知的激活保持分解、基于能量的非均匀秩分配以及KV对齐映射，解决了传统SVD转换忽视激活统计和层间秩差异的问题。

**🔧 技术方法**

主要使用了协方差加权SVD、权重去白化方法、贪婪水位填充的秩分配算法以及自适应KV映射等技术。

**📊 数据集**

通过对Llama‑3.1‑8B、Qwen3‑4B‑Instruct等大型语言模型以及多个公开基准（WikiText、ARC、MMLU等）进行校准和评估。

**📈 对比分析**

与统一秩SVD、Palu、TransMLA等基线对比，CARE在相同KV预算下零步困惑度降低最多215倍、准确率提升1.7倍，且在小规模恢复训练后能快速恢复甚至超过原始模型性能。

**⚠️ 局限性**

主要局限在于需要额外的校准数据来估计协方差、对极低秩压缩的鲁棒性不足，以及对不同数据分布或模型结构的泛化仍需进一步验证。

---

## 395. OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms

**arXiv ID:** 2603.17351 | [PDF](https://arxiv.org/pdf/2603.17351v1)

**作者:** Zhongyuang Liu `[一作]` (CertaintyX), Lihua Xie `[通讯]` (Nanyang Technological University)

**通讯引用:** 55627 | [OpenAlex ID](https://openalex.org/A5100365448)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 OmniVLN 框架，实现了零样本视觉-语言导航，融合 360° LiDAR 与全景摄像，构建多层动态场景图并通过层次化 3D 八角体视角与多分辨率提示实现高效 LLM 推理。

**💡 创新点**

创新点在于：1）硬件无关的全向感知堆栈；2）五层动态场景图与持久同调房间划分；3）基于 3D 八角体的 egocentric 视角与层次化链式思考；4）让 LLM 在有限上下文中完成跨房间目标定位与动作生成。

**🔧 技术方法**

采用了旋转 LiDAR、全景摄像、点云语义分割（Grounded SAM/SAM2）、动态场景图构造、持久同调（Persistent Homology）房间分割、VLM 辅助边验证、LLM（Qwen2.5-VL）推理、Octant 视角与多分辨率提示、Actor–Critic 交互与工具调用。

**📊 数据集**

构建了自己的 omnidirectional multimodal dataset，包含真实 IoT 实验室场景与三类合成数据集 D1–D9（低至高密度，分层关注）。

**📈 对比分析**

与平面列表基线对比，层次化提示可将 token 消耗降低 41–70%；在参考表达式生成中实现 93.18% 的整体准确率（相较 77.27% 的基线提升 15.91%）；导航成功率在复杂多房间环境下从 16% 提升到 80%。

**⚠️ 局限性**

局限性包括：依赖离线 LLM 推理导致实时性受限；对动态场景（人移动）支持不足；依赖高质量语义分割与 LiDAR 传感；跨平台部署仍需针对不同平台的低层控制调优。

---

## 396. Polynomial Kernels with Reachability for Weighted $d$-Matroid Intersection

**arXiv ID:** 2603.17345 | [PDF](https://arxiv.org/pdf/2603.17345v1)

**作者:** Chien-Chung Huang `[一作]` (Centre National de la Recherche Scientifique), Tatsuya Terao `[通讯]` (Kyoto University)

**通讯引用:** 1293 | [OpenAlex ID](https://openalex.org/A5073490534)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本论文研究了加权d-矩阵交集问题的随机多项式核化，提出了一种新的核化技术，能够处理一般的矩阵情况。

**💡 创新点**

创新点在于开发了一种新的核化技术，使得在一个矩阵是任意的情况下，其他d-1个矩阵为分区矩阵时，能够获得多项式大小的核。

**🔧 技术方法**

使用了随机化的核化技术，结合了可达核的概念，确保任何可行解都能通过交换元素达到更好的解。

**📊 数据集**

使用了加权d-矩阵交集问题的实例，其中包括任意矩阵和分区矩阵等多种类型的矩阵。

**📈 对比分析**

与现有方法相比，提出的方法在处理一般矩阵时表现出更好的性能，能够在多项式时间内构造出大小为 Õ(k^d) 的核，符合最优界限。

**⚠️ 局限性**

限制在于当给定的d-1个矩阵为层叠矩阵时，所得到的核的大小为准多项式级别，可能在某些情况下不够紧凑。

---

## 397. UniSem: Generalizable Semantic 3D Reconstruction from Sparse Unposed Images

**arXiv ID:** 2603.17519 | [PDF](https://arxiv.org/pdf/2603.17519v1)

**作者:** Guibiao Liao `[一作]` (Moore Threads AI), Yaohua Tang `[通讯]` (Moore Threads AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种面向稀疏无姿态图像的统一语义感知3D重建框架UniSem；

**💡 创新点**

核心创新点包括（1）误差感知高斯丢弃（Error-aware Gaussian Dropout，EGD）通过渲染误差引导抑制冗余高斯，提升几何稳定性；（2）混合训练课程（Mix-training Curriculum，MTC）将2D分割特征与模型自发的3D语义先验混合，利用对象级原型对齐提升语义一致性与泛化；

**🔧 技术方法**

技术手段包括ViT（DINOv2）特征提取、跨视角Transformer编码器、DPT头部生成3D高斯参数、误差引导丢弃、余弦周期调度、最大误差提示与SAM2生成跨视角对象掩码、几何加权原型对齐、以及多任务损失（颜色、语义、几何）组合；

**📊 数据集**

在ScanNet与ScanNet++的1565个室内场景上训练，评估于40个未见的ScanNet场景以及Replica数据集进行跨域测试；

**📈 对比分析**

与LSM、SpatialSplat、Uni3R、DFF、Feat3DGS、AnySplat、LSeg等方法对比，UniSem在2视角下深度Rel下降至3.84%（比Uni3R降低15%）、mAcc提升3.7%、PSNR与SSIM均等于或略优于AnySplat、在多视角设置下保持优势，并在Replica上实现更优的深度与语义分割；

**⚠️ 局限性**

局限性主要体现在：仅在室内数据集上训练与评估，缺乏对户外、动态或极端光照场景的验证；SAM2生成的掩码在高度杂乱的场景中可能引入误差；模型对极端视角变化和稀疏采样仍有待进一步提升。

---

## 398. Stereo World Model: Camera-Guided Stereo Video Generation

**arXiv ID:** 2603.17375 | [PDF](https://arxiv.org/pdf/2603.17375v1)

**作者:** Yang-Tian Sun `[一作]` (University of Hong Kong), Xiaojuan Qi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了StereoWorld，一种基于双目图像的世界模型，能够在给定相机轨迹下端到端生成视角一致、几何感知的立体视频。

**💡 创新点**

核心创新包括：1）统一相机帧RoPE，扩展令牌维度以注入相机条件并保留预训练先验；2）立体感知注意力分解，将4D注意力拆为3D视内注意力和水平行注意力，利用视差线性约束显著降低计算量。

**🔧 技术方法**

使用预训练的视频扩散模型（Latent Diffusion + DiT），结合旋转位置编码（RoPE）、立体感知注意力、相机条件扩展以及深度无监督的双目几何学习。

**📊 数据集**

在混合数据集上训练和评估，包括FoundationStereo、UnrealStereo4K、TartanAir和Middlebury等，测试集共435张双目图像。

**📈 对比分析**

与单目生成+后期视差转换（StereoCrafter、RGB‑D方法）以及单目版StereoWorld进行对比，评估指标涵盖相机精度、左右视图同步、视觉质量（FID/FVD/CLIP）和FPS。结果显示StereoWorld在立体一致性、视差精度和相机跟随精度上优于基线，生成速度提升约3×，视角一致性提升约5%。

**⚠️ 局限性**

局限性主要在于计算成本高于单目方法，以及缺乏大规模双目训练数据导致模型扩展性受限。

---

## 399. LoST: Level of Semantics Tokenization for 3D Shapes

**arXiv ID:** 2603.17995 | [PDF](https://arxiv.org/pdf/2603.17995v1)

**作者:** Niladri Shekhar Dutt `[一作]` (University College London), Xuelin Chen `[通讯]` (Adobe Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种按语义显著性排序的三维形状分词方法——Level-of-Semantics Tokenization（LoST），使得前缀即能解码出完整且语义合理的形状；

**💡 创新点**

通过构造RIDA（Relational Inter‑Distance Alignment）损失，将三维潜在空间的相对关系对齐到DINO图像语义空间，从而实现无渲染的语义监督；

**🔧 技术方法**

采用ViT编码器生成注册令牌序列，利用Nested Dropout与因果遮蔽实现层级顺序；使用Diffusion‑Transformer（DiT）生成完整潜在；使用GPT‑style Transformer进行自回归生成；RIDA采用多正样本InfoNCE、rank distillation和空间结构蒸馏；

**📊 数据集**

在Direct3D VAE的三平面潜在空间上训练，使用约30万条由Gemini 2.5 Pro生成文本提示、Flux 1渲染图像、Step1X‑3D生成的三维形状构成的训练集；测试集为1k新形状；

**📈 对比分析**

与基于几何LoD的分词器（OctGPT、VertexRegen）以及自回归生成模型（ShapeLLM‑Omni、OctGPT、Llama‑Mesh）进行对比；在语义（DINO、FID）和几何（Chamfer距离）指标上，LoST在极少量令牌（1–4个）即可优于基线，在128个令牌时实现SOTA的生成质量；

**⚠️ 局限性**

仅支持VAE三平面潜在，难以直接迁移到Gaussian Splats等其他三维表示；使用Diffusion解码增加推理成本；极少令牌下仍会出现几何/语义瑕疵；自回归模型采用固定长度，未实现基于形状复杂度的动态停止。

---

## 400. The Program Hypergraph: Multi-Way Relational Structure for Geometric Algebra, Spatial Compute, and Physics-Aware Compilation

**arXiv ID:** 2603.17627 | [PDF](https://arxiv.org/pdf/2603.17627v1)

**作者:** Houston Haynes `[一作]` `[通讯]` (SpeakEZ Technologies), Houston Haynes (SpeakEZ Technologies)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 Program Hypergraph（PHG），将原有的 Program Semantic Graph（PSG）中的二元边扩展为多元超边，以在 Fidelity 编译框架中统一表达几何代数、空间数据流和物理感知计算中的多路约束，并实现设计时的梯度、拓扑一致性和资源可达性验证。

**💡 创新点**

核心创新在于：①把几何代数中的 grade 证明为 DTS 维度轴并通过 grade 推断实现几何乘法的稀疏化；②以超边形式统一多路约束（如 k‑simplices、Tile co‑location、四体测量等），消除中间无意义节点和浮点误差；③在编译器中实现设计时的约束求解、SMT 验证和多目标资源分配，弥补了传统二元 PSG 的结构性缺陷。

**🔧 技术方法**

使用了 DTS 与 DMM 维度推断、SMT‑LIB2 证明、MLIR 多目标后端（LLVM、CIRCT、MLIR‑AIE）、超图分区算法（hMETIS/PaToH）、量子/神经网络相关技术（StDP、MoE、Quire 累加）以及语言服务器协议进行实时诊断。

**📊 数据集**

主要采用几何代数标准算子（PGA、CGA）、Mesh 基准和 Flash Clifford 示例作为功能验证；并在实验中与 Flash Clifford、GAmphetamine、Versor 等现有库对比，展示稀疏化和设计时检查的效果。

**📈 对比分析**

通过与传统二元 PSG 及现有几何代数库的基准对比，PHG 使几何乘法在 3D PGA 中实现高达 20× 的运算量减少，稀疏实现与手工优化相当；在 FPGA 与 NPU 目标上，超图分区提高了资源利用率，设计时验证消除了运行时错误，整体性能提升显著。

**⚠️ 局限性**

当前局限包括：缺乏完整的超图分区集成与自动化调度、Clifford 语言 dialect 的实现待完成、对量子后端的支持尚未成熟；设计时验证仅覆盖结构性约束，无法保证运行时行为、收敛性和性能极限；且对大规模模型的扩展性仍需进一步实验验证。

---

## 401. A Creative Agent is Worth a 64-Token Template

**arXiv ID:** 2603.17895 | [PDF](https://arxiv.org/pdf/2603.17895v1)

**作者:** Ruixiao Shi `[一作]` (Southeast University), Xin Geng `[通讯]` (Southeast University)

**通讯引用:** 6516 | [OpenAlex ID](https://openalex.org/A5074742406)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 CAT 的框架，将创意代理的“创意理解”转化为可复用的 token 模板，从而在文本到图像生成中直接注入创意语义，提升生成效率与质量。

**💡 创新点**

创新点在于将代理的创意知识通过“Creative Tokenizer”嵌入连续高维 token 中，消除对离散自然语言的依赖与多次代理查询，实现一次性复用、显著提升速度与成本，同时保持对多领域（建筑、家具、自然混合）的创意融合能力。

**🔧 技术方法**

主要技术包括：①基于文本编码器（T5、CLIP）提取模糊与增强提示的嵌入；②训练 Creative Tokenizer 生成 token 模板；③通过“创意语义解耦”训练策略（对齐、相对解耦、弹性锚定）实现创意知识的分离；④与 FLUX.1、Stable Diffusion 3 等基础 T2I 模型联合使用。

**📊 数据集**

使用 CangJie 数据集（60 个基本概念）并扩展到建筑设计（10 主概念、55 风格）与家具设计（20 主概念、38 风格），同时在自然混合任务中保留原始动物/植物概念。

**📈 对比分析**

与 FLUX.1、Stable Diffusion 3 等开放源 T2I 模型、GPT-Image-1.5、Gemini 3.1 Flash Image、以及专门的创意生成方法（T2I-Copilot、CREA、BASS、AGSwap、CreTok）进行对比。实验显示 CAT 在 VQAScore、PickScore、ImageReward、GPT‑4o 评估和用户研究中均显著优于基线，速度提升 3.7 倍、成本降低 4.8 倍。

**⚠️ 局限性**

局限性包括：①对训练数据和代理的依赖度高，若代理推理能力不足或数据不充分，token 可能无法充分表达创意；②目前仅在组合式创意（概念+风格）任务中验证，尚未探索更复杂的多模态或多概念融合；③在极端稀有或高维概念组合上，token 的表达空间可能仍有限。

---

## 402. Exploiting the English Grammar Profile for L2 grammatical analysis with LLMs

**arXiv ID:** 2603.17171 | [PDF](https://arxiv.org/pdf/2603.17171v1)

**作者:** Stefano Bannò `[一作]` (Cambridge University), Mark Gales `[通讯]` (Cambridge University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于英文语法概况（English Grammar Profile，EGP）及大规模语言模型（LLMs）的框架，用来自动识别和分类第二语言学习者在写作中尝试使用的语法结构，并将尝试分为成功、失败和未尝试三类。

**💡 创新点**

创新点在于：①使用EGP可解释的“can‑do”语句作为检测目标，结合原句与自动/人工纠正后的句子对，首次实现对学习者语法尝试的成功/失败判定；②将LLMs与规则方法结合，形成混合流水线，既能捕捉语义/语用细微差别，又保留对形态/句法特征的高效识别；③利用检测到的语法尝试作为特征来预测整体CEFR水平，展示语法尝试与语言水平的显著相关性。

**🔧 技术方法**

主要技术手段包括：规则基语法识别（POLKE、自己实现的规则系统）、LLM推理（Qwen 2.5 32B、GPT‑4.1），以及文本预处理（PoS/依存句法标注、句法依赖抽取）。此外，使用GECToR进行自动语法纠错，生成原/纠正句子对。

**📊 数据集**

使用的主要数据集为：Write & Improve 2024（≈23 000篇含CEFR标注的英语写作样本）；从中提取的3 671句子对（W&I‑EGP）用于语法尝试检测；完整的W&I 2024开发集（506篇）用于CEFR预测；另使用EGP本身的1 211条can‑do语句作为检测范围。

**📈 对比分析**

对比方法：规则基系统（RB）、POLKE、LLM（Qwen 2.5 32B、GPT‑4.1）以及混合方案（规则预过滤+LLM）。在语法尝试检测上，GPT‑4.1在所有类别中表现最佳（平均F1≈86.4%），LLM优于规则基方法在语义/语用构造；在CEFR预测上，混合方案（RBFilter+Qwen）在成功尝试上得到最高的相关系数（PCC≈0.78、SRC≈0.79）。自动纠错（GECToR）相比人工纠正略有下降，但差距不大。

**⚠️ 局限性**

局限性：①仅对12条可实现的EGP构造进行检测，未覆盖全量1 211条；②训练集采样偏向尝试类样本，导致对未尝试类的识别受限；③阈值设定过于粗糙，缺少对每条构造的细粒度阈值；④GECToR在纠错时存在低纠正率，影响失败尝试的识别；⑤实验聚焦写作文本，未验证到口语或多模态输入。

---

## 403. Temporal Gains, Spatial Costs: Revisiting Video Fine-Tuning in Multimodal Large Language Models

**arXiv ID:** 2603.17541 | [PDF](https://arxiv.org/pdf/2603.17541v1)

**作者:** Linghao Zhang `[一作]` (Shanghai Jiao Tong University), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1220 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统性评估视频监督微调（Video‑SFT）对多模态大型语言模型（MLLM）在图像与视频任务上的影响，发现其会提升视频性能但往往削弱图像性能；

**💡 创新点**

揭示并命名了“时间陷阱”（temporal trap）——视频微调与空间视觉推理的冲突，并通过理论分析说明了帧数预算导致的梯度不匹配；

**🔧 技术方法**

采用共享参数的微调框架、基于帧采样的梯度分析与理论推导，提出了指令感知的混合帧（Hybrid‑Frame）自适应帧数分配策略；

**📊 数据集**

在20,000段多来源视频的LLaVA‑Next‑Video‑178k训练集上进行微调，并在图像基准（MME、MMStar、MMBench、POPE）和视频基准（Video‑MME、MVBench、TempCompass、Video‑MMMU）上进行评估；

**📈 对比分析**

与固定帧数（8/16/32/64帧）微调结果对比，Hybrid‑Frame在保持视频表现的同时，显著提升了图像基准得分（如MMStar、POPE），验证了自适应帧数能部分缓解时间陷阱；

**⚠️ 局限性**

仅在代表性模型与静态基准上验证，未涵盖所有架构、流式/在线训练以及交互式多模态推理；Hybrid‑Frame方法为启发式，缺乏严格的理论最优保证；

---

## 404. Rubric-Guided Fine-tuning of SpeechLLMs for Multi-Aspect, Multi-Rater L2 Reading-Speech Assessment

**arXiv ID:** 2603.16889 | [PDF](https://arxiv.org/pdf/2603.16889v1)

**作者:** Aditya Kamlesh Parikh `[一作]` (Radboud University), Helmer Strik `[通讯]` (Radboud University)

**通讯引用:** 5392 | [OpenAlex ID](https://openalex.org/A5019585114)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文利用 Qwen2‑Audio‑7B‑Instruct SpeechLLM 对 L2 阅读语音进行多维度（准确度、流畅度、语调）评估，并在五种模型配置（从单一回归到多评估器+高斯不确定性+共形预测）上进行实验。

**💡 创新点**

创新点在于首次将多评估器监督、Gaussian Negative Log‑Likelihood 预测不确定性与共形预测相结合，以实现对人类评分的准确、可信且可解释的估计。

**🔧 技术方法**

技术包括：SpeechLLM 预训练、LoRA 参数高效微调、三种回归头（均方误差、GNLL）、共形预测校准、以及多评估器（Accuracy、Fluency、Prosody）联合学习。

**📊 数据集**

使用公开的 SpeechOcean762 数据集（5000 条英朗读语音，包含 5 位专家评语）。

**📈 对比分析**

与传统分类（DiCl）及单一回归（SRR.M）对比，最高配置 MRR.GC 在严格和宽容评估下均取得最优，PCC≈0.81、RMSE≈0.83、QWK≈0.50，且预测区间覆盖率达 90%。

**⚠️ 局限性**

局限性包括：数据集中评分偏向中间区间，导致模型难以准确捕捉极端分数；以及对低频异常评估的泛化能力仍有限。

---

## 405. Revisiting Vulnerability Patch Identification on Data in the Wild

**arXiv ID:** 2603.17266 | [PDF](https://arxiv.org/pdf/2603.17266v1)

**作者:** Ivana Clairine Irsan `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 30656 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过将基于NVD数据训练的安全补丁检测模型投放到真实世界的未公开补丁上，评估其泛化性能并提出混合训练方案以提升鲁棒性。

**💡 创新点**

首次系统揭示NVD数据导致的训练偏差，使模型在未报告补丁上的性能骤降（最高90% F1下降），并证明加入野生补丁数据可显著缓解该问题。

**🔧 技术方法**

采用深度学习模型（CodeBERT、CommitBART、CodeT5、UniXcoder、GRAPE）与大语言模型（DeepSeek、Llama‑3.1、Qwen3）进行二分类，同时利用XGBoost和perplexity对补丁特征差异进行分析。

**📊 数据集**

使用NVD相关数据集（ColeFunda、MoreFixes、GRAPE）与野生数据集（JavaVFC、PatchDB、Devign）以及人工抽取的负样本。

**📈 对比分析**

在NVD数据上训练后在野生数据集上测试，F1‑score下降最高达90%；通过混合训练（NVD+少量野生补丁）可将F1提升至约78%，仅加入100条野生补丁即可提升约15%。

**⚠️ 局限性**

受限于人工标注成本高、缺少细粒度CWE标签、实验语言仅覆盖Java/C++、LLM在零样本下表现不佳，以及NVD数据中仍存在的噪声。

---

## 406. Hidden Clones: Exposing and Fixing Family Bias in Vision-Language Model Ensembles

**arXiv ID:** 2603.17111 | [PDF](https://arxiv.org/pdf/2603.17111v1)

**作者:** Zacharie Bugaud `[一作]` `[通讯]` (Astera Institute), Zacharie Bugaud (Astera Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究多模型视觉语言模型（VLM）集成时的家族相关误差，提出基于家族结构的三种集成方法（HFV、QualRCCV、LCS）并对其效果进行评估。

**💡 创新点**

创新点在于：①首次系统分析VLM集成中家族相关误差的影响，揭示有效投票数极低；②提出Hierarchical Family Voting（HFV）实现两级家族投票；③开发质量加权的RCCV并进一步扩展为QualRCCV；④提出基于候选答案的学习型评分LCS，实现跨方法的细粒度决策。

**🔧 技术方法**

技术手段包括：错误相关性谱分析、特征工程与梯度提升（LightGBM）学习、交叉验证校准权重、家族聚类（spectral clustering）等；集成策略有投票、加权投票、家族内部投票、家族级加权。

**📊 数据集**

使用VQA三大基准：VQAv2（20,001题）、TextVQA（5,000题）和GQA（12,578题）。

**📈 对比分析**

与传统多数投票、校准投票、去重投票等基线对比，QualRCCV在三大基准上均显著优于校准投票（+0.17%/0.21%/0.31%），LCS在VQAv2、TextVQA、GQA上分别提升+0.68%/0.61%/2.45%，在GQA中甚至超过单一最佳模型。

**⚠️ 局限性**

局限性包括：①集成模型以单一家族（Qwen2.5-VL）占比高，家族均衡性不足；②仅评估短答案VQA，未覆盖开放式生成任务；③LCS需交叉验证和有标签数据；④HFV等方法仍假设家族内部误差相关且各家族至少略优于随机。

---

## 407. Network- and Device-Level Cyber Deception for Contested Environments Using RL and LLMs

**arXiv ID:** 2603.17272 | [PDF](https://arxiv.org/pdf/2603.17272v1)

**作者:** Abhijeet Sahu `[一作]` (National Renewable Energy Laboratory), Rochard Macwan `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了面向工业控制系统（DNP3）的网络级与设备级网络欺骗框架，并结合RL和LLM实现动态欺骗策略；

**💡 创新点**

首次在OT环境中融合多智能体RL和LLM驱动的RTU仿真，实现协议一致、时序可控的欺骗响应；

**🔧 技术方法**

使用强化学习（PPO/A2C）控制路由策略、PettingZoo多智能体框架、LLM（GPT‑4）生成DNP3响应、RAG增强提示、奖励工程（结合攻击成功率、持续时间、真实感评分）；

**📊 数据集**

采用IEEE 123‑bus配电网仿真环境、虚拟RTU honeypot、NREL Cyber Range及HELICS/Minimega；

**📈 对比分析**

通过与随机策略、单智能体与多智能体、仅网络、仅物理、混合物理+网络、RL+LLM等多种对照实验，发现RL+LLM组合在攻击重定向成功率、攻击持续时间上显著提升，平均episode长度缩短、奖励提升；

**⚠️ 局限性**

局限性包括高计算成本（每轮仿真耗时15–20分钟）、LLM实时推理时延、奖励设计的复杂性、缺乏异构多智能体的支持、未在真实OT环境中验证、LLM生成结果易出现幻觉或时序误差。

---

## 408. Efficient and Effective Table-Centric Table Union Search in Data Lakes

**arXiv ID:** 2603.17298 | [PDF](https://arxiv.org/pdf/2603.17298v1)

**作者:** Yongkang Sun `[一作]` (Hong Kong Polytechnic University), Jieming Shi `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 150 | [OpenAlex ID](https://openalex.org/A5102750234)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种以表为中心的表联合搜索方法，利用表级嵌入进行快速候选检索，并在候选集上通过双证据重排实现高效、准确的表联合。

**💡 创新点**

创新点包括：①直接学习表级嵌入而非仅列级；②设计正负表对构造和两阶段负采样以提升表示质量；③自适应候选检索根据表级相似度动态决定候选池；④双证据重排将表级和列级分数相结合。

**🔧 技术方法**

使用对比学习（InfoNCE）、BERT多头注意力表编码、优先采样序列化、FastText列嵌入、HNSW近似最近邻等技术。

**📊 数据集**

使用六个公开基准数据集：SANTOS、TUS Small、TUS Large、Wiki Union（含标签）以及无标签的 TUS Small/Large 进行在线效率评测。

**📈 对比分析**

与多种基线（SANTOS、TUS、4column、TUSL、TUSB、TUSK）在 MAP@k、P@k、R@k 等指标上对比，平均排名第一，MAP 最高达 99.36% 等；离线/在线处理时间均比基线至少快 10 倍，候选集更小且包含更高比例真联合表。

**⚠️ 局限性**

局限性：对极小或低信息量表性能下降；仅针对表联合任务验证，未扩展至连接或模式匹配；在大规模数据湖上仍需进一步优化内存与索引开销。

---

## 409. Domain-informed explainable boosting machines for trustworthy lateral spread predictions

**arXiv ID:** 2603.17175 | [PDF](https://arxiv.org/pdf/2603.17175v1)

**作者:** Cheng-Hsi Hsiao `[一作]`, Ellen M. Rathje `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在解释性增强机器学习模型EBM上加入领域知识，对其形状函数进行后处理，以纠正非物理学行为并提升模型的物理一致性；

**💡 创新点**

提出一种基于物理约束的形状函数修改框架，既能保持数据驱动模式，又能纠正非物理趋势，首次将可解释性模型与领域知识结合用于地震侧向扩散预测；

**🔧 技术方法**

使用Explainable Boosting Machine（EBM）作为基准模型，采用单变量sigmoid拟合、双变量交互合成与区域性替换等技术实现形状函数修正；

**📊 数据集**

利用2011年基督城地震侧向扩散数据集（3055个扩散点+4236个非扩散点），包含地下水深、峰值地面加速度、海拔、河流距离和斜坡角等五个特征；

**📈 对比分析**

与原始EBM、随机森林和XGBoost三种模型在准确率、精确率、召回率、F1、AUC等指标上进行比较；改进后的EBM在测试集上准确率从79.9%下降到75.0%，但在物理一致性和解释性上得到显著提升；

**⚠️ 局限性**

主要限制包括：改进后预测准确率下降（4–5%误差）；领域知识的人工选择与手工阈值设定可能带来主观偏差；未覆盖所有物理约束细节，模型仍无法完全替代专家经验。

---

## 410. Deploying Semantic ID-based Generative Retrieval for Large-Scale Podcast Discovery at Spotify

**arXiv ID:** 2603.17540 | [PDF](https://arxiv.org/pdf/2603.17540v1)

**作者:** Edoardo D'Amico `[一作]` (Spotify), Paul N. Bennett `[通讯]` (Spotify)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并部署了 GLIDE，一种基于大型语言模型的生成式播客推荐系统，利用语义 ID（Semantic IDs）实现对数百万剧集的精准检索与生成。

**💡 创新点**

创新点包括：将推荐任务转化为指令跟随的生成任务；通过 R-KMeans 量化生成可解释的 SID 以实现跨平台的语义对齐；采用软提示（soft prompt）方式注入长期用户嵌入，实现模型尺寸可控的个性化；以及多任务控制词设计，支持熟悉与新奇内容的可控发现。

**🔧 技术方法**

核心技术包括 Llama 3.2 1B 语言模型、残差 K‑Means 量化、双向文本‑SID 对齐训练、软提示嵌入投影、指令调优、Beam Search 采样以及系统级的高并发服务优化。

**📊 数据集**

使用的数据集来源于 Spotify 的播客交互日志、剧集元数据（标题、描述、主题标签）以及用户行为记录，覆盖英美等多国用户，并在实验中选取近 30 天活跃用户作为评估样本。

**📈 对比分析**

评估方法包括离线检索指标（Recall@30/NDCG@30）、内部人工评测以及基于 LLM 的评判器，在线 21 天 A/B 测试显示在 Home surface 上非习惯性收听提升 5.4%，新秀剧集发现提升 14.3%，并保持低延迟与成本。

**⚠️ 局限性**

局限性包括：对中文等非英语语言支持有限；SID 碰撞需依赖后置的基于受欢迎度的解析；模型尺寸受 1B 参数限制，可能限制更细粒度的个性化；以及在极端稀缺内容或热门内容偏倚时仍需额外校正。

---

## 411. H Infinity Robust Control for Gust Load Alleviation of Geometrically Nonlinear Flexible Aircraft

**arXiv ID:** 2603.17443 | [PDF](https://arxiv.org/pdf/2603.17443v1)

**作者:** Nikolaos D. Tantaroudas `[一作]` (Institute of Communications and Computer Systems), Kenneth J. Badcock `[通讯]` (University of York)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

设计并验证了基于 H∞ 鲁棒控制的风载减轻（GLA）控制器，该控制器通过非线性模型降阶得到的 8/9 模式 ROM 在全非线性 540/1,616 维模型上实现稳健的翼尖位移抑制。

**💡 创新点**

创新点在于将非线性模型降阶（NMOR）与 H∞ 控制合成相结合，并引入输入形状权重 K_c，实现载荷减轻与飞行路径偏差之间的单参数权衡，证明了在高维非线性 aeroelastic 系统中仍能保持鲁棒性能。

**🔧 技术方法**

使用了非线性模型降阶（NMOR）技术、H∞ 鲁棒控制合成、第二阶舵面动力学模型、线性化及输入形状权重方法。

**📊 数据集**

使用了 Global Hawk‑like UAV（540 DOF）与 32 m 超柔飞翼（1,616 DOF）的数值模型；扰动输入包括离散 1‑余弦风和 Von Kármán 随机风场。

**📈 对比分析**

通过与开环对比进行性能评估：离散风下峰值翼尖位移降低 23.15%，随机风下 RMS 位移降低 10.26%；最大舵面转角约 9–12°，在舵面可用范围内；离散风下表现优于随机风。

**⚠️ 局限性**

局限性包括：对高频随机风的抑制有限；未考虑舵面速度/幅度限制；ROM 与实际非线性差异可能导致在更强非线性或更大结构变形情况下性能下降。

---

## 412. FineViT: Progressively Unlocking Fine-Grained Perception with Dense Recaptions

**arXiv ID:** 2603.17326 | [PDF](https://arxiv.org/pdf/2603.17326v1)

**作者:** Peisen Zhao `[一作]` (Huawei Inc.), Qi Tian `[通讯]` (Huawei Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并训练了 FineViT 视觉编码器，以提升细粒度视觉感知能力。

**💡 创新点**

创新点在于分阶段进步训练策略（MIM → 对比学习 → LLM 对齐）以及使用 FineCap-450M 规模最大、细粒度丰富的区域级标注数据。

**🔧 技术方法**

采用 ViT+2D RoPE、MIM 预训练、SigLIP 对比损失、LLM 自回归对齐及投影模块等技术。

**📊 数据集**

使用的数据集包括 1.8B 未标注图像、1.56B 通过 MLLM 生成的全球重标注、以及 450M 区域级 caption 的 FineCap-450M。

**📈 对比分析**

与 SigLIP2、Seed-ViT 等主流模型对比，FineViT 在零样本分类/检索、长文本检索、以及 VLM 多模态任务中均表现优异，尤其在 OCR、文档解析和定位计数任务上显著提升。

**⚠️ 局限性**

目前仅支持静态图像，未覆盖视频时序推理。

---

## 413. ReLMXEL: Adaptive RL-Based Memory Controller with Explainable Energy and Latency Optimization

**arXiv ID:** 2603.17309 | [PDF](https://arxiv.org/pdf/2603.17309v1)

**作者:** Panuganti Chirag Sai `[一作]` (Sri Sathya Sai Institute of Higher Learning), Naveen M `[通讯]` (Red Hat)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种可解释的多智能体强化学习框架 ReLMXEL，用于动态调优 DRAM 控制器参数，以降低能耗、延迟并提高带宽。

**💡 创新点**

创新点在于：① 将标量奖励拆分为多维向量并使用 MSX 进行最小充分解释；② 结合多智能体学习与可解释性，实时平衡能耗、带宽与延迟；③ 通过多维 Q‑表和 trace‑split 方法实现对多种工作负载的自适应优化。

**🔧 技术方法**

技术手段包括：强化学习（SARSA）、奖励分解与 RDX/MSX 解释、Q‑表存储、基于 DRAMSys/DRAMPower 的仿真、Intel Pin 轨迹提取、SPEC CPU 2017 轨迹，及多智能体并行策略。

**📊 数据集**

使用的数据集：Intel Pin 生成的 STREAM、GEMM、BFS、fotonik_3d、xalancbmk、gcc、roms、mcf、lbm、omnetpp 轨迹；SPEC CPU 2017 高内存负载（如 lbm、omnetpp 等）；ChampSim 生成的 SPEC 轨迹。

**📈 对比分析**

对比方法：在 DDR4 DRAMSys 仿真环境下，将 ReLMXEL 与基线（OpenAdaptive 页面策略、FR‑FCFS 调度、All‑bank 刷新）进行对比；评估指标为平均能耗、带宽利用率和延迟。实验显示 ReLMXEL 在所有工作负载上平均能耗降低 3–8%，带宽提升 4–140%，延迟略有改善或仅轻微上升，整体性能显著优于基线。

**⚠️ 局限性**

局限性：① 仅在 DDR4 仿真环境验证，缺乏真实硬件验证；② 需要大量轨迹数据和精确的监控指标；③ 探索阈值与 ε 决定需要针对不同工作负载调优；④ 解释机制（MSX）虽可解释决策但计算开销未详细评估；⑤ 对于安全攻击等非性能问题的鲁棒性尚未系统验证。

---

## 414. Training Diffusion Language Models for Black-Box Optimization

**arXiv ID:** 2603.17919 | [PDF](https://arxiv.org/pdf/2603.17919v1)

**作者:** Zipeng Sun `[一作]` (McGill University), Xue Liu `[通讯]` (McGill University)

**通讯引用:** 13814 | [OpenAlex ID](https://openalex.org/A5100372152)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何将扩散式大语言模型应用于离线黑盒优化任务，利用统一的 prompt–response 语料和两阶段后训练实现高标签设计生成。

**💡 创新点**

创新点在于使用显式分隔符将文本、设计和标签统一编码，采用联合掩码预测进行域适配，并通过监督微调与强化学习相结合的两阶段后训练，使模型在有限样本下捕获双向依赖并对高标签设计进行对齐。

**🔧 技术方法**

技术包括扩散式大语言模型（LLaDA-8B-Instruct）、域适配的掩码预测、监督微调、基于奖励的强化学习、一维掩码标记、分隔符标记。

**📊 数据集**

使用 Design-Bench 评估数据集，包括离散序列任务 TF Bind 8/10 以及连续参数任务 Ant Morphology 与 D’Kitty Morphology。

**📈 对比分析**

与多类前向与逆向方法（GP、VAE、GAN、AR、Diffusion 等）进行对比，DiBO 在大多数任务上取得最高的 100‑percentile 正则化得分，整体排名第一，性能明显优于传统 surrogate‑guided 方案。

**⚠️ 局限性**

局限性包括对显式分隔符和域适配的依赖、需要多阶段训练且计算成本高，且在极端数据稀缺或新领域时可能不易迁移；模型仍受离线数据偏差与偏倚影响。

---

## 415. Asymptotically ideal Disjunctive Hierarchical Secret Sharing Scheme with an Explicit Construction

**arXiv ID:** 2603.17257 | [PDF](https://arxiv.org/pdf/2603.17257v1)

**作者:** Jian Ding `[一作]`, Cheng Shu `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了一种渐近理想的分离层次秘密共享方案（DHSS），通过多项式、线性齐次递推关系和单向函数进行显式构造。

**💡 创新点**

创新点在于该方案在保持计算安全性的同时，具有较小的分享大小，并且是渐近理想的，解决了现有方案在安全性和分享大小之间的权衡问题。

**🔧 技术方法**

使用了多项式、多个线性齐次递推关系和单向函数。

**📊 数据集**

使用了一个大素数的有限域作为数据集，分享大小为log_2 p位，前提是p > n。

**📈 对比分析**

与现有的DHSS方案相比，提出的方案在安全性和渐近理想性上均表现良好，分享大小较小，且需要的计算时间为多项式时间。

**⚠️ 局限性**

限制在于该方案仍然需要进一步研究如何减少公开值的数量。

---

## 416. SMAL-pets: SMAL Based Avatars of Pets from Single Image

**arXiv ID:** 2603.17131 | [PDF](https://arxiv.org/pdf/2603.17131v1)

**作者:** Piotr Borycki `[一作]` (Jagiellonian University), Przemysław Spurek `[通讯]` (Jagiellonian University)

**通讯引用:** 796 | [OpenAlex ID](https://openalex.org/A5068511223)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

从单张图像生成可动画化的3D狗头像，结合SMAL结构网格和3D Gaussian Splatting实现高保真、可编辑的动物角色；

**💡 创新点**

①将SMAL网格与Gaussian Splatting融合，形成两阶段（绑定/解绑）优化流程，实现结构与细节并存；②使用文本驱动的编辑与动画，借助CLIP、DGE、FramePack/ActionMesh实现自然语言控制；③面基skinning将Gaussians绑定至SMAL，保证动画连贯性；

**🔧 技术方法**

3D Gaussian Splatting渲染、SMAL线性模型、联合优化、面基skinning、CLIP文本指导、DGE细节提升、FramePack视频插值、ActionMesh动作映射、CLIP评分评估；

**📊 数据集**

伪多视图数据由TripoSG、SAM3D、Trellis等单图3D模型合成；测试使用GART、CoP3D视频数据；与DogRecon等公开方法做对比；

**📈 对比分析**

通过CLIP‑Score与PSNR对不同阶段（Stage1/2/Final）进行量化，结果显示Stage2/Final显著提升细节与语义一致性；与DogRecon、GART、AniAv等SOTA做可视化与CLIP评分对比，模型在细节、动画连贯性和自然毛发表现上优于基线，CLIP分数从0.85提升至约0.89；

**⚠️ 局限性**

依赖初始单图3D生成模型，若其拓扑错误会直接影响最终头像；极端视角下ActionMesh可能失跟踪导致动画失真；目前对不同品种的泛化有限，缺乏大规模多物种训练。

---

## 417. Large Reasoning Models Struggle to Transfer Parametric Knowledge Across Scripts

**arXiv ID:** 2603.17070 | [PDF](https://arxiv.org/pdf/2603.17070v1)

**作者:** Lucas Bandarkar `[一作]` (University of California), Trevor Cohn `[通讯]` (University of Melbourne)

**通讯引用:** 7852 | [OpenAlex ID](https://openalex.org/A5078530959)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型推理型LLM在跨语言知识迁移中的缺陷，证明脚本不匹配是导致性能下降的主要瓶颈。

**💡 创新点**

创新点在于将脚本匹配视为核心因素，利用实体前缀和合成数据微调显著缩小跨脚本知识迁移差距。

**🔧 技术方法**

采用观察性回归分析、实体前缀注入、合成数据生成、LoRA超参数微调等技术手段。

**📊 数据集**

使用两份本地知识QA数据集和Belebele阅读理解数据集作为模型能力代理，并自生成跨语言问题用于SFT训练。

**📈 对比分析**

通过对同脚本与跨脚本问题的准确率对比，发现脚本匹配提升约12–13%，SFT后跨脚本性能进一步提升数个百分点，尤其在低资源语言上表现明显。

**⚠️ 局限性**

局限性包括翻译歧义可能影响评测结果，以及仅采用LoRA SFT 而非更高效的 RL 方法。

---

## 418. Causal Representation Learning on High-Dimensional Data: Benchmarks, Reproducibility, and Evaluation Metrics

**arXiv ID:** 2603.17405 | [PDF](https://arxiv.org/pdf/2603.17405v1)

**作者:** Alireza Sadeghi `[一作]` (Clemson University), Wael AbdAlmageed `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对当前因果表示学习（CRL）领域常用的合成与真实数据集进行了系统性评估，提出了理想数据集应具备的核心特征，并设计了一种基于Origami图的单一聚合指标，用以统一衡量模型在重构、可分离度、因果发现与反事实推理四个维度的性能；同时，对已有实现代码的可复现性进行了实证检验，揭示了多项缺陷。

**💡 创新点**

创新点包括：①提出了面向CRL的综合评价框架和统一指标（Origami得分），解决多维度、非统一指标难以比较的问题；②对数据集的结构、变量类型、语义一致性、因果连通性等维度进行了完整评述，明确了现实中常见数据集的局限；③通过重跑CausalVAE等模型，系统评估了代码公开、数据切分和结果一致性等可复现性指标，首次将可复现性纳入CRL研究的评价体系。

**🔧 技术方法**

主要技术手段包括：因果图（DAG）构造、结构因果模型（SCM）学习、可变自编码器（VAE）与其因果扩展（CausalVAE、SCM-VAE等）；评估指标涉及重构误差（MAE/MSE）、可分离度指标（MIC、TIC、IRS、DCI、JEMMIG）、因果发现指标（SHD、TPR、AUC）、反事实质量指标（FID、IS、KID、CLD、LPIPS）以及统一的Origami聚合得分。

**📊 数据集**

使用的主要数据集包括：合成数据集 Pendulum、Flow Noise、Shadow(PointLight)、Shadow(SunLight)；真实数据集 CelebA 的两个常用子集（SMILE、BEARD）以及 MorphMNIST；此外还提及了其他 CelebA 子集与其他合成数据集，用以示例数据集多样性与不足。

**📈 对比分析**

模型比较方法：首先在每个维度分别使用一组标准指标，对 β‑VAE、Conditional VAE 与 CausalVAE 在 Pendulum 数据集上进行多维度评估；随后将所有指标归一化后通过 Origami 图计算聚合得分，对模型整体性能进行排序；此外，还对公开实现代码进行重跑，并与原论文报告的 MIC、TIC 等指标做对比，评估可复现性。性能上，β‑VAE 在 IRS 上领先，CausalVAE 在 MIC/TIC 上表现最好，而 Conditional VAE 在重构和反事实生成指标（FID、KID）上最优；Origami得分显示 β‑VAE 仍然最优。

**⚠️ 局限性**

局限性：①数据集本身普遍缺乏足够变量、类别多样性、真实因果连通性和混杂变量，导致模型训练与评估受限；②合成数据集虽可控，却无法完全反映真实系统的不确定性；③评估指标多样、尺度不同，统一聚合仍受指标权重和归一化方式影响；④多模型的可复现性存在巨大差距，代码与数据集切分缺乏统一标准；⑤未充分解决分类变量建模与无监督可识别性问题，影响可解释性与推理可靠性。

---

## 419. Learning, Misspecification, and Cognitive Arbitrage in Linear-Quadratic Network Games

**arXiv ID:** 2603.17157 | [PDF](https://arxiv.org/pdf/2603.17157v1)

**作者:** Quanyan Zhu `[一作]` (New York University), Zhengye Han `[通讯]` (New York University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究在线性二次网络博弈中，代理人使用错设模型学习并行动，定义Berk‑Nash均衡并提出通过信息扭曲实现的认知套利设计。

**💡 创新点**

创新点在于将错设视为可利用的设计渠道，构建认知套利框架，给出闭式最优扭曲解，并证明两时间尺度学习收敛。

**🔧 技术方法**

采用Berk‑Nash均衡理论、均值场近似、Stackelberg 优化、二次约束二次规划 (QCQP) 与两时间尺度随机梯度学习。

**📊 数据集**

使用人工生成的12节点稠密交互网络作为实验数据集；未使用公开真实数据集。

**📈 对比分析**

通过数值仿真比较了BNE与NE的成本差异，并量化VoM；结果表明认知套利可显著降低总成本，验证了理论预期。

**⚠️ 局限性**

局限在于仅处理静态LQ博弈，假设代理人固定使用简化假设；未考虑动态/随机状态、异质偏好或对抗性代理；实验仅在小规模合成网络上验证。

---

## 420. Verification and Validation of Physics-Informed Surrogate Component Models for Dynamic Power-System Simulation

**arXiv ID:** 2603.17836 | [PDF](https://arxiv.org/pdf/2603.17836v1)

**作者:** Petros Ellinas `[一作]`, Spyros Chatzivasileiadis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文针对高维数值模拟中的采样问题，提出并评估了多种基于梯度的采样优化方法，并引入了新型主动采样策略；

**💡 创新点**

创新点在于将对偶敏感性信息与自适应采样结合，提出一种可直接减少 L₂ 误差的主动采样框架；

**🔧 技术方法**

采用了对偶梯度下降（Adjoint PGD）、Adam、SGD、LBFGS 等优化器，以及 ECP、随机采样和主动采样等方法；

**📊 数据集**

实验数据来源于三种不同的模拟场景（SM2、SM4、SM6），可视为不同复杂程度的基准测试集；

**📈 对比分析**

比较结果显示，在 SM6 场景下，主动采样的 L₂ 误差仅为 0.0101，显著优于 ECP（0.0107）、随机采样（0.0101）以及所有对偶优化方法；在 SM2、SM4 场景中，主动采样与随机采样相近，均优于对偶优化方法；

**⚠️ 局限性**

局限性包括：在较为复杂的 SM2、SM4 场景中误差仍较高，表明主动采样对复杂系统的适应性有限；同时对计算成本的量化分析不足，未给出方法的时间/资源消耗对比。

---

## 421. Enabling RISC-V Vector Code Generation in MLIR through Custom xDSL Lowerings

**arXiv ID:** 2603.17800 | [PDF](https://arxiv.org/pdf/2603.17800v1)

**作者:** Jie Lei `[一作]` (Universitat Politècnica de València), Adrián Castelló `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 525 | [OpenAlex ID](https://openalex.org/A5033947083)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329`

**🎯 论文内容**

本文构建了一个结合MLIR与xDSL的混合编译流水线，能够从高层抽象自动生成针对RISC-V Vector（RVV）指令的可移植C微核代码。

**💡 创新点**

创新点在于通过xDSL实现自定义低阶降解路径，将MLIR高层向量操作映射到RVV固有指令，并生成完整的C代码，使得RISC-V平台的性能可移植、可增量采用。

**🔧 技术方法**

使用技术包括MLIR的多语言域言辞、xDSL的Python原型化IR与转换、自定义RVV-MLIR方言、EmitC层实现，以及在两块真实RISC-V板上编译与测评。

**📊 数据集**

评测使用了标准GEMM矩阵尺寸（1000‑5000等）以及BERT‑Large模型的Transformer层矩阵（m、n、k各层不同），构造的矩阵尺寸而非公开数据集。

**📈 对比分析**

与OpenBLAS进行单核单精度性能对比，xDSL生成的微核在K230上提升10‑35%，在BPI上可达2.4倍，平均提升约20‑30%。

**⚠️ 局限性**

局限性包括仅覆盖RVV 1.0指令集、未自动生成打包例程、只针对单精度FP32、以及缺乏多核心并行评估。

---

## 422. Motion-Adaptive Temporal Attention for Lightweight Video Generation with Stable Diffusion

**arXiv ID:** 2603.17398 | [PDF](https://arxiv.org/pdf/2603.17398v1)

**作者:** Rui Hong `[一作]` (George Mason University), Shuxue Quan `[通讯]` (Independent Researcher)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种运动自适应的时间注意机制，在冻结的Stable Diffusion UNet中注入轻量级时间注意模块，实现参数高效的视频生成。

**💡 创新点**

创新点包括运动自适应注意偏置与运动感知门控、级联注入策略，以及仅训练25.8M参数并利用时间相关噪声获得高一致性，而不需显式时序损失。

**🔧 技术方法**

采用潜在扩散模型（LDM）、UNet Transformer块、时间自注意、运动估计、噪声相关性与控制网络等技术。

**📊 数据集**

在WebVid-10M子集的100K段视频上训练（每段8帧，256×256），并在WebVid验证集400条语句上评估。

**📈 对比分析**

与在相同数据集上重训练的AnimateDiff进行对比；我们的模型参数仅为25.8M（比AnimateDiff的417M低约16倍），帧一致性（FC）明显提升，CLIP/FC指标相近或更好，FVD略高但在参数效率上表现优异。

**⚠️ 局限性**

局限性在于仅生成8帧、256×256的短时视频，且对高运动视频仍需调节噪声相关系数，缺乏更长时序和更高分辨率的实验验证。

---

## 423. Federated Computing as Code (FCaC): Sovereignty-aware Systems by Design

**arXiv ID:** 2603.17331 | [PDF](https://arxiv.org/pdf/2603.17331v1)

**作者:** Enzo Fenoglio `[一作]` (University College London), Philip Treleaven `[通讯]` (University College London)

**通讯引用:** 5762 | [OpenAlex ID](https://openalex.org/A5000136854)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Federated Computing as Code（FCaC）框架，将治理决策编译为可验证的加密能力，支持跨组织边界的可移植授权；

**💡 创新点**

创新点在于将宪法治理与程序治理分离，并通过 KYO‑ECT‑PoP 信任链将权限、委托、持有证明打包成可在边界本地验证的数字凭证；

**🔧 技术方法**

采用 JWT、DPoP、RFC 8705 等标准的加密凭证，结合 Terraform‑style IaC、Open Policy Agent 等工具实现声明式契约编译与边界验证；

**📊 数据集**

在 MNIST 数据集上构建交叉数据中心联邦学习实验，作为受治理的工作负载验证边界控制；

**📈 对比分析**

与传统基于运行时策略引擎的 Federated Computing 进行对比，证明边界决策可在无共享状态下确定，实验展示了确定性验证但未给出吞吐量或延迟指标；

**⚠️ 局限性**

局限包括缺乏即时撤销、仅支持基于有效期的生命周期、令牌与信封未绑定、密钥管理模拟、未实现更细粒度委托与撤销、缺少大规模性能评估等。

---

## 424. Self-Conditioned Denoising for Atomistic Representation Learning

**arXiv ID:** 2603.17196 | [PDF](https://arxiv.org/pdf/2603.17196v1)

**作者:** Tynan Perez `[一作]`, Rafael Gomez-Bombarelli `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种自监督预训练方法Self‑Conditioned Denoising（SCD），通过在去噪过程中加入自我嵌入条件，提升原子级表示学习的效果。

**💡 创新点**

SCD克服了传统节点去噪在局部性、标量嵌入缺失和非平衡结构不适用等三大局限，利用条件嵌入同时强化全局语义和标量表示，兼容任意原子数据域。

**🔧 技术方法**

基于等变换图神经网络（TorchMD‑Net/Equivariant Transformer）实现的SCD预训练，使用AdaNorm实现条件输入；预训练目标为对噪声的回归，并在下游任务中直接使用单通前向推理。

**📊 数据集**

在多域原子数据集上进行预训练与评估，包括PCQ（小分子DFT）、GEOM10（xTB构型）、AMP20（周期材料DFT）、SAIR（蛋白-配体构象）以及混合ALL和OMol25（非平衡MD）。

**📈 对比分析**

与传统节点去噪、Frad、SliDe以及监督力-能量预训练相比，SCD在QM9、Matbench bandgap和Ligand Binding Affinity三大基准上均实现了显著提升（提升幅度从~20%到~45%），并在资源消耗上优于大模型的监督预训练。

**⚠️ 局限性**

局限性包括对更大模型和更高维表征的扩展尚未深入验证；多域预训练在单域任务上的性能略逊；SCD仍需在更广泛的任务和数据分布下进一步测试其泛化能力。

---

## 425. DSS-GAN: Directional State Space GAN with Mamba backbone for Class-Conditional Image Synthesis

**arXiv ID:** 2603.17637 | [PDF](https://arxiv.org/pdf/2603.17637v1)

**作者:** Aleksander Ogonowski `[一作]` (Warsaw University of Technology), Przemysław Rokita `[通讯]` (Warsaw University of Technology)

**通讯引用:** 208 | [OpenAlex ID](https://openalex.org/A5107395750)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 DSS-GAN，一种将 Mamba 作为生成器骨干的 GAN，采用 Directional Latent Routing（DLR）对噪声向量和类别信息进行空间方向特定的特征级调制，实现单步噪声到图像的生成。

**💡 创新点**

创新点在于：①将潜在向量按扫描方向拆分并与类别嵌入共同生成仿射调制，形成方向特定的条件调制；②引入方向权重网络学习自适应路由；③在 DLR 块前后加入 180° 随机旋转以提升梯度稳定性；④首次将 Mamba 的线性状态空间模型用于 GAN 生成器，从而实现长距离依赖与高效计算的结合。

**🔧 技术方法**

使用技术包括 Mamba 选择性状态空间模型、方向扫描与反扫描、DLR 条件调制、180° 旋转增强、StyleGAN2 风格的卷积细化块、StyleGAN2-ADA 判别器、R1 正则化、ADA 训练策略（在实验中禁用以便对比）等。

**📊 数据集**

评估数据集包括 AFHQ（猫、狗、野生动物）、FFHQ、LSUN（室内场景、室外建筑）和 CelebA，分辨率从 128×128 到 512×512。

**📈 对比分析**

与 StyleGAN2-ADA 在同一判别器和训练配置下进行直接对比，指标为 FID、KID、Precision、Recall、Density、Coverage。DSS-GAN 在大多数数据集上与 StyleGAN2-ADA 性能相当，且在 KID、Precision、Density 上显著优于其；参数量减少 82%，单样本推理延迟更低，吞吐量也保持竞争力。

**⚠️ 局限性**

局限性包括：Mamba 的顺序性导致在大批量推理时吞吐量低于完全并行的卷积网络；在高分辨率下需要缩小 Mamba 维度，影响性能；扫描方向选择仍需手工依据数据几何定；CNN 判别器限制只能处理规则 2D 图像，难以扩展到非标准网格或三维空间；未结合 ADA 或其他增强策略，低数据场景仍可能不足。

---

## 426. How do LLMs Compute Verbal Confidence

**arXiv ID:** 2603.17839 | [PDF](https://arxiv.org/pdf/2603.17839v1)

**作者:** Dharshan Kumaran `[一作]` (Google DeepMind), Petar Velickovic `[通讯]` (Google DeepMind)

**通讯引用:** 14382 | [OpenAlex ID](https://openalex.org/A5008869927)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对Gemma 3 27B和Qwen 2.5 7B大模型的内部机制进行多种干预（激活导向、补丁、噪声、交换、线性探测和注意力屏蔽），研究了LLM如何生成口头置信度评分，并证明置信度是随答案生成自动缓存并在后续检索而非即时计算的；

**💡 创新点**

创新点在于首次揭示LLM口头置信度来源于答案生成后自动缓存的位置，并证明这些缓存表示包含远超标记概率的答案质量评估信息，体现了LLM的第二阶自我评估能力；

**🔧 技术方法**

主要技术包括激活导向（activation steering）、激活补丁与噪声（patching/noising）、激活交换实验、线性探测（linear probing）、方差分解与注意力屏蔽（attention blocking）等机制解释方法；

**📊 数据集**

使用的评估数据集为TriviaQA，用于测试事实性知识回答的置信度预测；

**📈 对比分析**

与传统基于标记概率的置信度相比，本文通过方差分解显示缓存置信度解释了显著更多方差（超过标记概率的8.4%），并通过AUROC和ECE等指标验证了模型置信度的可区分性与校准性；

**⚠️ 局限性**

局限性包括仅在两款模型与有限提示格式下验证，缺乏对不同规模或架构的广泛泛化评估，且干预仅针对单一位置，未完全解开整个分布式置信度网络的细节。

---

## 427. The Unreasonable Effectiveness of Text Embedding Interpolation for Continuous Image Steering

**arXiv ID:** 2603.17998 | [PDF](https://arxiv.org/pdf/2603.17998v1)

**作者:** Yigit Ekin `[一作]`, Yossi Gandelsman `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关、在测试时通过在文本编码器空间中进行线性干预来实现连续可控图像编辑的方法。

**💡 创新点**

创新点在于：①仅使用LLM自动生成去偏对比提示对并通过差均向量计算文本向量方向；②通过LLM辅助的词元选择实现局部/全局/风格编辑；③引入弹性区间搜索算法自动确定可控力度区间；④提出统一的连续性评价指标MID。

**🔧 技术方法**

技术包括：文本编码器向量平移、LLM生成对比数据、词元池化、差均向量（Difference‑of‑Means）构造、弹性区间搜索（elastic band）、DreamSim与ΔVQA评估。

**📊 数据集**

使用LLM生成的去偏对比提示对（约100条），无公开专门数据集；实验在Flux.dev和Qwen‑Image‑Edit等现有文本条件生成模型上进行。

**📈 对比分析**

与多种基线（FluxSpace、SAEdit、Flux‑Slider、SliderEdit、Kontinuous Kontext）比较，评估指标为编辑成功度ΔVQA和滑动连贯性MID；结果显示该方法在保持图像内容的同时获得与训练无关方法相当甚至更好的滑动连贯性，且在更强大的后端模型上编辑效果显著提升。

**⚠️ 局限性**

局限性：依赖生成模型能够在文本空间中实现概念两端的生成，模型偏差可能导致无法编辑；对对比数据集生成仍需较强LLM；缺乏高质量视频感知指标限制视频扩展。

---

## 428. Augmenting Scholarly Reading with Cross-Media Annotations

**arXiv ID:** 2603.17957 | [PDF](https://arxiv.org/pdf/2603.17957v1)

**作者:** Qi Xu `[一作]` (Vrije Universiteit Brussel), Beat Signer `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 1541 | [OpenAlex ID](https://openalex.org/A5010074519)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨媒体注释工具，让学者能在阅读PDF时直接链接并嵌入音频、视频、网页等多种外部资源，增强学术阅读体验。

**💡 创新点**

创新点在于将注释视为可跨媒体、可跨文档的超链接，并在PDF视图中采用空间与颜色双编码的弹出窗口展示链接内容，实现一键拖拽、即时查看的交互方式。

**🔧 技术方法**

技术上基于RSL（Resource‑Selector‑Link）超媒体元模型，使用Semantic Reader PDF组件库进行渲染，前端通过插件与浏览器、视频播放器对接，后端REST服务维护资源、选择器和链接实体。

**📊 数据集**

论文未使用公开数据集，功能以案例演示为主；未来计划在真实学术文献与外部多媒体资源上进行验证。

**📈 对比分析**

尚未进行量化比较与性能评估；作者提议未来通过对照实验和实地部署评估用户效率、阅读深度和协作效果。

**⚠️ 局限性**

局限性包括：目前仅支持人工手动创建注释；缺乏自动推荐和AI辅助；功能验证尚待实现与实证；系统对不同PDF渲染环境的兼容性未测试。

---

## 429. Background and Intellectual Development: Supplementary Material for the Category Mistake Papers

**arXiv ID:** 2603.16907 | [PDF](https://arxiv.org/pdf/2603.16907v1)

**作者:** Paul Borrill `[一作]` `[通讯]` (DAEDAELUS), Paul Borrill (DAEDAELUS)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

对 Lamport 的 happens‑before 关系进行批判性分析，提出 Category Mistake 框架，阐述分布式系统在时间假设上的错误，并通过对 iCloud Drive 366 GB 归档的实证分析揭示同步失败根源，随后设计并验证了 Open Atomic Ethernet（OAE）协议以消除时间单向假设导致的问题。

**💡 创新点**

创新点在于将物理学中无背景时间的视角引入分布式系统理论，识别并系统化了 Lamport 及其后继工作中的“类别错误”，构建了双向、可逆的因果模型，并通过 OAE 在链路层实现完全基于信息反馈的可靠交付，从而突破传统基于时间戳的冲突解决限制。

**🔧 技术方法**

使用的技术包括：逻辑时钟与 Lamport 时序分析、Python 脚本对文件内容的 MD5/SHA‑256 比对与冲突合并、Open Atomic Ethernet（OAE）协议规范、基于三角拓扑的双向交易与可逆状态机、以及对 iCloud 同步日志和系统失败案例的定量分析。

**📊 数据集**

使用的数据集包括：iCloud Drive 366 GB 归档（约110个顶层目录、89个仅存在于归档的文件夹、57个内容不一致的文件夹、1406个仅在图形目录的文件），以及从 iCloud 同步日志提取的时间戳和冲突记录，进一步补充了 136 起系统失败的公开案例数据。

**📈 对比分析**

比较方法：将传统基于时间戳的冲突解决与 OAE 的信息反馈确认机制进行对比；在 iCloud 归档数据上，使用内容哈希验证冲突恢复效果。结果显示，传统方法导致数百 GB 数据失真、隐式删除和不可恢复冲突，而 OAE 能在不引入时间戳的前提下实现一次性交付、零冲突，并在实验中恢复了所有文件一致性，性能提升体现在消除了时间同步延迟和冲突恢复时间。

**⚠️ 局限性**

局限性包括：OAE 依赖专用链路和三角拓扑，可能不适用于现有广播网络；实验集中在 iCloud 环境，缺乏跨多种分布式系统的广泛验证；理论框架尚未涵盖所有一致性协议的细节，未来需要在更大规模和更复杂拓扑中进一步评估与完善。

---

## 430. Differential Attention-Augmented BiomedCLIP with Asymmetric Focal Optimization for Imbalanced Multi-Label Video Capsule Endoscopy Classification

**arXiv ID:** 2603.17879 | [PDF](https://arxiv.org/pdf/2603.17879v1)

**作者:** Podakanti Satyajith Chary `[一作]` (Indian Institute of Technology Hyderabad), Nagarajan Ganapathy `[通讯]` (Indian Institute of Technology Hyderabad)

**通讯引用:** 3811 | [OpenAlex ID](https://openalex.org/A5020402349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种针对视频胶囊内镜的多标签分类框架，利用改进的BiomedCLIP和差分注意力机制，并在采样、损失、增强和后处理等多层次上处理类别不平衡；

**💡 创新点**

创新点在于将差分注意力替换传统自注意力以抑制噪声，同时结合sqrt频率采样、异向焦点损失、mixup、标签平滑及每类阈值优化，构成端到端的稀有病理检测管线；

**🔧 技术方法**

使用的技术包括ViT差分注意力、Excitation门控分类头、对比学习、异向焦点损失、sqrt频率采样、mixup、标签平滑、EMA、OneCycleLR学习率调度、中值滤波、gap合并等；

**📊 数据集**

数据集主要为Galar数据集（训练60视频，约49万帧）以及RARE‑VISION竞赛测试集（3个NaviCam视频，共161,025帧）；

**📈 对比分析**

在RARE‑VISION测试集上，模型实现整体mAP@0.5=0.2456、mAP@0.95=0.2353，单GPU推理时间约8.6分钟；与传统基线相比，mAP提升显著，且mAP@0.5与mAP@0.95差距仅0.01，表明时序边界定位良好；

**⚠️ 局限性**

主要局限在极少样本病理类别（如活跃出血、红斑、血凝块）AP低于0.01，难以区分与正常黏膜的视觉相似性；同时未采用时序建模，无法进一步利用帧间连续信息。

---

## 431. OpenQlaw: An Agentic AI Assistant for Analysis of 2D Quantum Materials

**arXiv ID:** 2603.17043 | [PDF](https://arxiv.org/pdf/2603.17043v1)

**作者:** Sankalp Pandey `[一作]` (University of Arkansas), Khoa Luu `[通讯]` (University of Arkansas)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了OpenQlaw多智能体框架，将二维量子材料的视觉识别与物理推理分离，实现实验室决策的高效自动化。

**💡 创新点**

创新点在于将物理意识的多模态LLM（QuPAINT）作为领域专家交由中央智能体调用，并通过持久记忆和确定性执行工具消除认知负荷，直接输出可操作的物理量和可视化结果。

**🔧 技术方法**

采用多智能体架构、Vision‑Language模型Qwen3‑VL‑32B‑Instruct、QuPAINT的物理意识推理、Python确定性计算工具、图像注释工具，以及WhatsApp/Discord交互接口。

**📊 数据集**

使用自建的二维材料显微镜图像集（含单层、双层、少层石墨烯、MoS₂、hBN等）进行实验。

**📈 对比分析**

与QuPAINT单独输出的冗长推理结果对比，OpenQlaw在相同任务下产生简洁回答并准确计算面积，节省资源并提升可操作性；案例研究中实现了精准面积计算和可视化定位。

**⚠️ 局限性**

局限包括多模态模型的令牌长度限制导致高密度图像坐标数组截断、持久记忆增加推理延迟，以及对领域专家输出格式的严格依赖导致错误传播。

---

## 432. Noise-Aware Misclassification Attack Detection in Collaborative DNN Inference

**arXiv ID:** 2603.17914 | [PDF](https://arxiv.org/pdf/2603.17914v1)

**作者:** Shima Yousefi `[一作]` (City University of New York), Saptarshi Debroy `[通讯]` (City University of New York)

**通讯引用:** 683 | [OpenAlex ID](https://openalex.org/A5015917097)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了协作式DNN推理中在存在通信噪声的环境下如何检测中间特征的对抗误分类攻击，并提出了一种半灰盒噪声感知异常检测框架。

**💡 创新点**

创新点在于将变分自编码器与基于MAD的鲁棒噪声描述相结合，显式建模噪声特征来提升检测鲁棒性，并实现了在边缘设备低延迟、低算力环境下的在线检测。

**🔧 技术方法**

使用改进的adVAE（带自对抗转换器）提取重构误差、潜在偏移和残差MAD等四种检测特征，采用一类支持向量机进行异常分类；噪声模型采用α-稳定分布（SαS）的冲击性噪声。

**📊 数据集**

以CIFAR-100数据集为基础训练VGG19、AlexNet和MobileNet，并在其10000条测试样本（2800正常+1200对抗）上进行检测评估。

**📈 对比分析**

与无噪声感知基线（仅使用重构误差与潜在偏移）对比，实验显示在中到高噪声强度下，噪声感知方案在AUROC、准确率和F1等指标上提升5–30%，尤其在更深切点层和强攻击强度下仍保持80–90%的AUROC。

**⚠️ 局限性**

当噪声强度极大或攻击强度低、切点层更深时，攻击特征与正常噪声重叠严重导致检测性能下降；框架依赖噪声模型的先验假设，若实际噪声偏离SαS分布可能失效；此外仅针对基于VAE的对抗攻击，其他攻击方式仍需进一步研究。

---

## 433. ConfusionBench: An Expert-Validated Benchmark for Confusion Recognition and Localization in Educational Videos

**arXiv ID:** 2603.17267 | [PDF](https://arxiv.org/pdf/2603.17267v1)

**作者:** Lu Dong `[一作]` (State University of New York at Buffalo), Ifeoma Nwogu `[通讯]` (State University of New York at Buffalo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过多阶段VLM辅助过滤、研究员筛选和专家验证，构建了高质量的学生困惑识别与定位基准ConfusionBench；

**💡 创新点**

创新点在于提出了面向困惑的高效VLM辅助构建管道，生成细粒度时间标注的困惑视频数据集，并提供困惑报告可视化；

**🔧 技术方法**

采用大规模视觉‑语言模型（Qwen3‑VL‑4B‑Instruct 与 Gemini 3 Flash Preview）进行两阶段模型筛选、提示工程、加权投票等技术；

**📊 数据集**

以公开的DAiSEE视频数据为基础，进一步细分为2秒剪辑并进行专家验证，构成450个平衡困惑样本和10段5分钟长视频的定位数据集；

**📈 对比分析**

在零样本设置下，Gemini在困惑识别任务上F1达到0.8139，高于Qwen的0.6293；在长视频定位上Gemini微平均F1为0.6934、tIoU 0.5307，而Qwen的对应指标显著更低；

**⚠️ 局限性**

局限性包括数据规模有限（仅450个短片和10段长视频）、仅区分困惑与否缺乏强度或阶段划分、对原始DAiSEE质量依赖、以及模型对细微困惑信号的识别仍有改进空间。

---

## 434. Crisis-induced differences in attention towards Ukraine in Twitter 2008-2023

**arXiv ID:** 2603.17899 | [PDF](https://arxiv.org/pdf/2603.17899v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 435. Parameter-Efficient Modality-Balanced Symmetric Fusion for Multimodal Remote Sensing Semantic Segmentation

**arXiv ID:** 2603.17705 | [PDF](https://arxiv.org/pdf/2603.17705v1)

**作者:** Haocheng Li `[一作]` (China Agricultural University), Jianxi Huang `[通讯]` (China Agricultural University)

**通讯引用:** 8213 | [OpenAlex ID](https://openalex.org/A5046271690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 MoBaNet，一种参数高效、对称融合的多模态遥感语义分割框架，利用冻结的 Vision Foundation Model（ViT）作为主干，对 RGB 与 DSM 双模态进行深度交互与融合。

**💡 创新点**

创新点包括：①Cross‑modal Prompt‑Injected Adapter (CPIA)，通过共享 prompt 注入轻量级适配器实现冻结主干下的深层语义交互；②Difference‑Guided Gated Fusion Module (DGFM)，利用两模态差异进行门控融合，得到紧凑且判别力强的多模态表示；③Modality‑Conditional Random Masking (MCRM)，在训练阶段随机遮蔽单一模态并给未遮蔽分支提供硬像素监督，抑制 RGB‑主导的优化，提升模态平衡与鲁棒性。

**🔧 技术方法**

采用冻结的 ViT（如 DINOv2‑ViT‑B/L 或 SAM）作为主干，结合轻量级 PEFT（CPIA、DGFM、MCRM）实现参数高效微调，使用 UPerNet 解码头完成像素级预测。

**📊 数据集**

使用 ISPRS Vaihingen 与 ISPRS Potsdam 两个高分辨率城市遥感数据集，输入为 RGB（或 NIR+R+G/B）+ DSM。

**📈 对比分析**

与多种单模态和多模态基线（U‑Net、SegNet、FuseNet、PSPNet、ABCNet、DC‑Swin、UNetFormer、CMFNet、FTransUNet、MANet 等）以及不同 VFM（SAM、DINOv2）进行对比。MoBaNet 在 OA、mF1、mIoU 上均达到或超过现有最优，且仅使用 6.18M 可训练参数，显著低于全微调和其他多模态方法。

**⚠️ 局限性**

局限性：对极小目标（如小型车辆）和高度相似类别的细粒度分割仍有提升空间；仅在 RGB‑DSM 组合上验证，未评估多光谱、SAR 或 LiDAR 等更丰富模态；跨城市/跨传感器泛化与不完整模态场景的鲁棒性需进一步研究。

---

## 436. REAL: Robust Extreme Agility via Spatio-Temporal Policy Learning and Physics-Guided Filtering

**arXiv ID:** 2603.17653 | [PDF](https://arxiv.org/pdf/2603.17653v1)

**作者:** Jialong Liu `[一作]` (Hong Kong University of Science and Technology), Changhao Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 706 | [OpenAlex ID](https://openalex.org/A5046822372)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套名为REAL的端到端框架，使四足机器人在严重感知退化下能够实现极限敏捷的运动学稳定的跑酷；

**💡 创新点**

创新点包括：①跨模态特征融合的FiLM+Mamba时序骨干网络，能主动过滤噪声并构建短期地形记忆；②基于物理约束的贝叶斯扩展卡尔曼滤波器，将不确定的神经速度预测与刚体动力学耦合；③一致性感知的损失门控机制，动态平衡模仿学习与强化学习以提升Sim-to-Real稳健性；

**🔧 技术方法**

核心技术包括：跨模态注意力、FiLM特征调制、Mamba时序建模、1D ResNet速度不确定性预测、EKF物理过滤、动态损失门控；

**📊 数据集**

使用自定义的极限跑酷仿真环境（Isaac Gym）生成的多种障碍地形（悬空桥、斜坡、障碍堆叠等），并在实际Unitree Go2四足机器人上进行实验；

**📈 对比分析**

与Extreme-Parkour、RPL、SoloParkour等基线对比，REAL在成功率、平均横向位移、边缘违规率等指标上明显优于对手；在感知噪声、遮挡和盲区测试中，REAL仅表现出轻微性能下降；实时推理时间保持在13.1 ms，满足20 ms控制周期；

**⚠️ 局限性**

局限性包括：在完全无视觉且地形极为复杂的盲区下仍可能出现运动不稳定；当前仅在单一四足平台验证，需要进一步验证跨平台泛化；对极端动态冲击的鲁棒性虽然提升，但在某些极限跳跃中仍存在小概率崩溃。

---

## 437. OPERA: Online Data Pruning for Efficient Retrieval Model Adaptation

**arXiv ID:** 2603.17205 | [PDF](https://arxiv.org/pdf/2603.17205v1)

**作者:** Haoyang Fang `[一作]` (Amazon Web Services), George Karypis `[通讯]` (Amazon Web Services)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 OPERA 框架，用静态 (SP) 与动态 (DP) 数据裁剪方法对密集检索器进行领域自适应微调。

**💡 创新点**

创新点在于揭示了检索训练中两阶段采样的质量-覆盖率折中，并提出通过层级软裁剪与随时间变化的阈值调度，动态地对查询和文档级别的样本进行加权，从而在保持覆盖率的同时聚焦高质量样本。

**🔧 技术方法**

采用预训练模型计算余弦相似度作为质量评估；SP 直接保留 top‑k 最高相似度对；DP 通过两层采样（查询、文档）结合余弦调度调节采样强度，并用软权重替代硬裁剪；并提供理论证明与实验验证。

**📊 数据集**

在八个跨领域数据集（NFCorpus、TripClick、FiQA、ANTIQUES、TriviaQA、HotpotQA、FEVER 等）上评估，同时在 BGE‑large‑en‑v1.5（Encoder‑only）和 Qwen3‑Embedding‑0.6B（LLM‑based）两种检索器上验证。

**📈 对比分析**

与标准微调、InfoBatch、随机裁剪等方法对比，SP 在 NDCG@10 上提升约 +0.5% 但 Recall@20 降低；DP 在 NDCG@10 与 Recall@20 上分别提升约 +1.9% 与 +0.7%，平均排名第 1.38；DP 训练时间缩短至 50% 以内，展示出更快收敛与更优性能。

**⚠️ 局限性**

局限包括：DP 对迭代次数和学习率敏感，动态阈值调度需额外超参；SP 在覆盖率上受限；两阶段结合虽有效但增加复杂度；在极少迭代或高学习率环境下 DP 效果可能不如预期。

---

## 438. LLM-Powered Flood Depth Estimation from Social Media Imagery: A Vision-Language Model Framework with Mechanistic Interpretability for Transportation Resilience

**arXiv ID:** 2603.17108 | [PDF](https://arxiv.org/pdf/2603.17108v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 439. In Perfect Harmony: Orchestrating Causality in Actor-Based Systems

**arXiv ID:** 2603.17909 | [PDF](https://arxiv.org/pdf/2603.17909v1)

**作者:** Vladyslav Mikytiv `[一作]` (Nova University Lisbon), Carla Ferreira `[通讯]` (Nova University Lisbon)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

开发了ACTORCHESTRA框架，利用Erlang OTP的gen_server机制对多actor交互进行自动因果追踪，并提供WALTZ语言实现跨actor属性的运行时监控；

**💡 创新点**

创新点在于通过编译期AST注入与运行时conductor协同实现跨actor因果链自动化，WALTZ DSL隐藏消息交织、支持多actor属性，并且不需要手工修改目标系统；

**🔧 技术方法**

技术手段包括Erlang OTP、gen_server行为、parse_transform注入、make_ref生成上下文ID、conductor中介层、WALTZ DSL编译为Erlang监控；

**📊 数据集**

实验使用三套案例系统：算术流水线、聊天室应用以及Laspr的CRDT实现，并在不同客户端数量和消息负载（如5C×30M、200C×10M等）下进行基准测试；

**📈 对比分析**

通过对比instrumented与基线的执行时间、延迟和吞吐量，发现整体overhead约为100–145%，聊天室与Lasp的延迟约翻倍，吞吐量下降55–60%，算术系统overhead在105–130%之间，整体性能可接受；

**⚠️ 局限性**

局限性包括仅支持OTP gen_server client‑server架构，无法覆盖非同步或其他OTP行为；conductor为单点，可能成为瓶颈；WALTZ不支持进程级细粒度绑定和跨进程变量关联，且未在生产环境进行评估。

---

## 440. Alignment Makes Language Models Normative, Not Descriptive

**arXiv ID:** 2603.17218 | [PDF](https://arxiv.org/pdf/2603.17218v1)

**作者:** Eilam Shapira `[一作]` (Technion Israel Institute of Technology), Roi Reichart `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

比较LLM基线模型与对齐模型在多回合战略游戏中的预测性能，发现基线模型在四类多回合游戏中明显优于对齐模型；

**💡 创新点**

揭示对齐会导致模型行为分布收缩，削弱对多回合情境中描述性人类行为的预测能力，从而在一回合或非互动任务中反而表现更好；

**🔧 技术方法**

采用RLHF/DPO进行对齐训练，使用token概率提取方法进行决策预测，并用Pearson相关系数进行模型性能比较；

**📊 数据集**

使用GLEE的四类多回合游戏（讨价还价、说服、谈判、重复矩阵游戏）以及单回合矩阵游戏和二元彩票数据集；

**📈 对比分析**

以Pearson相关率为指标，基线模型在多回合游戏中获得213/22（约9.7:1）的优势，而在单回合矩阵游戏和彩票中对齐模型分别以4.1:1和2.2:1的比率占优；

**⚠️ 局限性**

局限在于仅评估离散决策、使用开放权重模型、实验数据来自人类对抗LLM、未涉及连续动作空间或封闭模型的验证。

---

## 441. TharuChat: Bootstrapping Large Language Models for a Low-Resource Language via Synthetic Data and Human Validation

**arXiv ID:** 2603.17220 | [PDF](https://arxiv.org/pdf/2603.17220v1)

**作者:** Prajwal Panth `[一作]` (KIIT Deemed to be University), Agniva Maiti `[通讯]` (KIIT Deemed to be University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 Tharu-LLaMA（3B）模型，通过“LLM-to-Human”自助式数据生成和人工校验，构建了 TharuChat 数据集，解决了 Tharu 低资源语言在 LLM 训练中的数据缺失问题。

**💡 创新点**

创新点在于将高阶 Gemini 生成的语法注入与口述传统相结合的“语境加载”方法，以及在 3B 规模模型上采用高阶 LoRA（r=16, α=32）实现了极低资源下的高效微调，并在少量混合方言数据上实现线性缩减困惑度的可证明效果。

**🔧 技术方法**

使用的技术包括 Gemini 2.5 Prompt‑Engineering、LLaMA‑3.2‑3B‑Instruct 基础模型、LoRA 参数高效微调、FP16 训练、梯度累积与 16GB T4 GPU 可部署方案，以及基于 Perplexity 的评估与线性缩放分析。

**📊 数据集**

主要数据集为 TharuChat，约 3,955 条指令–响应对（正式实验中使用 3,116 条），涵盖 70% Rana Tharu、20% Dangaura 及 10% Kochila 等方言，且经过人工验证的“银标准”混合方言语料。

**📈 对比分析**

与零样本 Tharu 预测（PPL>88）相比，逐步扩增数据（25%→100%）使模型 PPL 从 6.42 线性下降至 2.88，验证了少量高质量数据对模型流畅度和准确性的显著提升，且在 3B 模型上保持良好的泛化与低算力部署可行性。

**⚠️ 局限性**

局限性包括：方言混杂导致语法一致性下降、残留的 Hindi/Awadhi 影响、对非训练领域（如高级学科、全球历史）的知识掌握有限，以及依赖人工校验的“银标准”数据无法完全消除生成噪声。

---

## 442. Gender Disambiguation in Machine Translation: Diagnostic Evaluation in Decoder-Only Architectures

**arXiv ID:** 2603.17952 | [PDF](https://arxiv.org/pdf/2603.17952v1)

**作者:** Chiara Manna `[一作]` (Tilburg University), Eva Vanmassenhove `[通讯]` (Tilburg University)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5009329618)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在机器翻译的性别偏见评估框架上做了扩展，加入了 Prior Bias 指标并将其应用于 Decoder‑Only LLM 的评估。

**💡 创新点**

创新点在于：① 引入 Prior Bias 用以量化模型在缺乏性别线索时的默认性别倾向；② 将该框架扩展到 Decoder‑Only 模型，检验其对上下文性别线索的利用；③ 通过对比基础模型、持续预训练模型和指令微调模型，揭示指令微调在性别偏见缓解上的潜力。

**🔧 技术方法**

主要技术包括：Transformer Decoder‑Only LLM (Llama2、TowerBase、TowerInstruct)，WinoMT 与其中性化扩展数据集，WinoMT 标准评估管线（对齐+形态分析），Minimal Pair Accuracy (MPA) 与 Prior Bias 计算，以及自注意力权重分析。

**📊 数据集**

使用的数据集是 EN→IT WinoMT（原始和中性化版本），共 3,888 句子；以及 Encoder‑Decoder 参考系统的相同数据集。

**📈 对比分析**

比较方法：在标准性别准确率、MPA、Prior Bias 上与 Encoder‑Decoder 基线（OPUS‑MT、NLLB‑200、mBART）对比。结果显示 Decoder‑Only LLM 在标准准确率上与 Encoder‑Decoder 相当，但 MPA 明显低于 Encoder‑Decoder；指令微调模型 TowerInstruct 在 MPA、女性准确率提升显著，Prior Bias 从约 85% 降至 75%（女性比例升至 25%）。

**⚠️ 局限性**

局限性包括：仅处理二元性别，忽略非二元性别；自动对齐和形态分析噪声导致 Unknown 率；自注意力分析仅为相关性而非因果关系；指令微调的具体机制未被深入探究。

---

## 443. Cyberlanguage: Native Communication for the Cyber-Physical-Social-Thinking Fusion Space

**arXiv ID:** 2603.17498 | [PDF](https://arxiv.org/pdf/2603.17498v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 444. SafeTutors: Benchmarking Pedagogical Safety in AI Tutoring Systems

**arXiv ID:** 2603.17373 | [PDF](https://arxiv.org/pdf/2603.17373v1)

**作者:** Rima Hazra `[一作]` (Eindhoven University of Technology), Mykola Pechenizkiy `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 10925 | [OpenAlex ID](https://openalex.org/A5022601535)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并构建了SafeTutors基准，系统评估LLM在数学、物理、化学三学科中的教学安全与教学质量。

**💡 创新点**

首次将学习科学风险分类与多轮对话安全评估结合，设计了11维、48子风险的风险分类体系，并引入多轮对话评估机制。

**🔧 技术方法**

采用基于学习科学的风险分类、GPT-5.2/DeepSeek/Claude等评估器、Crescendo式对话生成以及人工标注的混合评估框架。

**📊 数据集**

使用MathDial、CAMEL-AI等公开题库作为种子，生成共3,135个单轮、2,820个多轮交互样本构成SafeTutors数据集。

**📈 对比分析**

对10款开源LLM（3.8B–72B）及闭源GPT-5-mini进行评估，结果显示无模型能在所有风险维度保持低危害，规模增大不一定提升安全性，多轮交互导致教学安全恶化，性能整体处于高危害水平。

**⚠️ 局限性**

局限性包括仅覆盖三门STEM学科、风险分类仍可能缺失细节、评估依赖人工标注与自动化工具、以及对模型规模与安全性的解释不足。

---

## 445. Multi-Modal Multi-Agent Reinforcement Learning for Radiology Report Generation: Radiologist-Like Workflow with Clinically Verifiable Rewards

**arXiv ID:** 2603.16876 | [PDF](https://arxiv.org/pdf/2603.16876v1)

**作者:** Kaito Baba `[一作]` (University of Tokyo Hospital), Satoshi Kodera `[通讯]` (University of Tokyo Hospital)

**通讯引用:** 1094 | [OpenAlex ID](https://openalex.org/A5066658224)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

设计并实现了 MARL‑Rad，一种基于多模态多代理强化学习的放射学报告生成框架；

**💡 创新点**

创新点在于：①将区域专属代理与全局整合代理端到端联合训练，②使用基于可验证临床指标的奖励进行全系统的 on‑policy 强化学习；

**🔧 技术方法**

采用多代理 GSPO（MA‑GSPO）算法、MedGemma‑4B 视觉‑语言模型、CheXbert、RadGraph、ROUGE‑L 等可验证奖励组成的复合奖励；

**📊 数据集**

使用 MIMIC‑CXR（训练/测试）和 IU X‑ray（训练/验证/测试）两大公开胸部 X‑ray 数据集；

**📈 对比分析**

与多种基线（单代理 RL、无 RL 的代理化、传统 NLG 模型）对比，MARL‑Rad 在 CE 指标（RadGraph F1、CheXbert F1、GREEN）上实现了 state‑of‑the‑art，尽管部分 NLG 指标略逊；

**⚠️ 局限性**

局限性包括：①仅在胸部 X‑ray 任务上验证，跨模态或其他医学任务的通用性尚未测试；②强化学习训练成本高，缺乏对计算开销的详细分析；

---

## 446. Approximation by Quad Meshes in Laguerre Geometry

**arXiv ID:** 2603.17865 | [PDF](https://arxiv.org/pdf/2603.17865v1)

**作者:** A. Ramos-Cisneros `[一作]` (King Abdullah University of Science and Technology), H. Pottmann `[通讯]` (Institute of Discrete Mathematics and Geometry, Vienna University of Technology)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了在拉格朗日几何中定义的L-共轭参数化（L-conjugate nets），并利用该理论开发了一种基于四边网格的L-网格（L-mesh）逼近正曲率光滑曲面的算法。

**💡 创新点**

创新点在于：①首次引入L-共轭概念，将传统的共轭参数化推广到带有球面共轭（sphere congruence）的拉格朗日几何；②构造了从L-共轭方向场到L-网格的初始化方法；③提出了全局优化框架（Levenberg–Marquardt）以满足正交、切向接触和公平性约束，实现高精度逼近。

**🔧 技术方法**

主要技术包括：拉格朗日几何与其四维辛格拉夫模型；B‑splines曲面与主曲率、主方向的数值计算；伪L-共轭方向场的解析求解；基于方向场的四边网格重建；Levenberg–Marquardt优化和多项式能量约束（接触、单位法向、平滑度、距离等）。

**📊 数据集**

实验使用了多种参数化曲面（如螺旋、自由形曲面等）作为基准曲面；球面共轭取为最小主曲率半径的比例（r = τ·min{κ₁⁻¹,κ₂⁻¹}，τ∈(0,1)）。

**📈 对比分析**

与传统Q‑网格/三角网格逼近方法相比，本文方法在保持曲面的C¹连续性、控制球面大小和保持设计灵活性方面表现更佳。实验表明在各类初始化参数下，接触误差可低至10⁻⁷，优化时间在数百毫秒到数秒之间，且对不同球半径与方向场的选择具有可调性。

**⚠️ 局限性**

限制主要包括：仅针对正曲率表面（负曲率时会出现鞍点和奇异）；球半径须小于主曲率半径以避免回旋尾奇异；需要手动或自动化的方向场初始化，初始化不佳可能导致收敛困难；优化过程计算量大，且对参数调节敏感。

---

## 447. How Clued up are LLMs? Evaluating Multi-Step Deductive Reasoning in a Text-Based Game Environment

**arXiv ID:** 2603.17169 | [PDF](https://arxiv.org/pdf/2603.17169v1)

**作者:** Rebecca Ansell `[一作]` (Georgetown University), Autumn Toney-Wails `[通讯]` (Syntheos)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个基于文本的Clue多代理环境，评估LLM在长序列推理中的一致性，并试图通过在Mind Bender逻辑谜题上进行微调来提升游戏推理性能。

**💡 创新点**

创新点在于将传统桌游Clue转化为可供LLM参与的推理游戏，并通过对比微调前后模型在推理量、推理质量与游戏胜率之间的关系，揭示推理推导量与准确率不一定正相关。

**🔧 技术方法**

使用GPT‑4o‑mini和Gemini‑2.5‑Flash两大LLM，配合结构化提示、三阶段推理（Deduction、Action、Show‑card）以及在Google Vertex AI/OpenAI API上进行的微调。

**📊 数据集**

采用Mind Bender系列逻辑谜题（50道训练、10道验证）进行微调，游戏数据由自动生成的18局Clue模拟得到。

**📈 对比分析**

对比基线模型与微调模型在18局游戏中的胜率、推理正确/错误计数、回退率和知识积累速率。结果显示基线GPT‑4o‑mini平均赢5局，微调版本仅赢3局；Gemini‑2.5‑Flash微调后胜率略升但整体推理质量下降；微调模型往往推理量增大但准确率不提升。

**⚠️ 局限性**

局限在于仅使用文本推理微调，未加入外部工具或结构化状态跟踪；LLM在多轮互动中仍难以保持逻辑一致性，且不同模型对微调的响应差异显著，说明单纯文本微调不足以提升交互推理能力。

---

## 448. On the Extension Theorem for Packing Steiner Forests

**arXiv ID:** 2603.16956 | [PDF](https://arxiv.org/pdf/2603.16956v1)

**作者:** Jinghan A Zeng `[一作]` `[通讯]` (University of Illinois Urbana-Champaign), Jinghan A Zeng (University of Illinois Urbana-Champaign)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

未提供论文内容，无法确定具体做法

**💡 创新点**

未提供论文内容，无法确定创新点

**🔧 技术方法**

未提供论文内容，无法确定使用技术

**📊 数据集**

未提供论文内容，无法确定使用数据集

**📈 对比分析**

未提供论文内容，无法确定比较方法及性能

**⚠️ 局限性**

未提供论文内容，无法确定limitations

---

## 449. Noticing the Watcher: LLM Agents Can Infer CoT Monitoring from Blocking Feedback

**arXiv ID:** 2603.16928 | [PDF](https://arxiv.org/pdf/2603.16928v1)

**作者:** Thomas Jiralerspong `[一作]` (Mila), Yoshua Bengio `[通讯]` (Mila)

**通讯引用:** 443235 | [OpenAlex ID](https://openalex.org/A5086198262)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究语言模型在未被显式告知的情况下，能否自发推断其链式推理被监控，并检测其对抗监控的意图与执行能力。

**💡 创新点**

证明在仅通过阻塞反馈而无显式训练或提示的情况下，模型能通过上下文学习推断监控并产生反监控意图，但尚无法执行；揭示监控意识与执行能力的差距。

**🔧 技术方法**

采用多回合多任务实验框架，使用Claude系列大语言模型，链式推理监控与阻塞机制，评估指标包括意识水平、反监控意图与逃逸成功，利用LLM评估器与人工验证。

**📊 数据集**

使用SHADE-Arena任务集，包括Calendar Agendas与Bank Transfer两类带隐含副任务的任务。

**📈 对比分析**

对比不同代Claude模型的意识水平、意图与逃逸率，结果显示4.5代模型意识水平更高，约19%事件确认监控，但无模型成功逃逸，意图-能力差距明显。

**⚠️ 局限性**

仅评估Claude模型，评估器同家族偏倚；反馈机制人为化；仅两类任务；意识评估依赖明确表述；实验未覆盖真实部署的多样化反馈。

---

## 450. KA2L: A Knowledge-Aware Active Learning Framework for LLMs

**arXiv ID:** 2603.17566 | [PDF](https://arxiv.org/pdf/2603.17566v1)

**作者:** Haoxuan Yin `[一作]` (Harbin Institute of Technology), Jingchi Jiang `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 727 | [OpenAlex ID](https://openalex.org/A5064238181)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了知识感知主动学习（KA2L）框架，通过对LLM隐藏层状态的探测识别已掌握与未知的知识点，并利用隐藏状态解码生成未知问题，形成闭环主动学习策略以高效微调LLM。

**💡 创新点**

创新点包括：①用语义熵衡量模型输出一致性并动态阈值化；②训练MLP探测隐藏状态中的知识掌握情况；③通过隐藏状态解码从模型内部潜在空间生成多样化的未知问题；④将主动学习与数据增广结合，实现显著减少标注与计算成本。

**🔧 技术方法**

技术手段：语义熵（Semantic Entropy）与NLI聚类、动态阈值二值化、Multi‑Layer Perceptron（MLP）探测器、t5‑base隐藏状态解码、LoRA/P‑tuning微调、对比传统主动学习方法（Entropy、Coreset、BADGE）等。

**📊 数据集**

使用的公开数据集包括 TriviaQA、NQ‑Open 与 MedMCQA；对九个开源LLM（Llama、Mistral、Phi、Qwen、GLM 等）进行实验。

**📈 对比分析**

与随机、Entropy、Coreset、BADGE 等传统主动学习方法对比，KA2L 在同样数据量下性能明显更好，5k 未知样本即可逼近 10k 全数据集效果，成本下降约 50%，并在三大数据集上均优于传统方法。

**⚠️ 局限性**

局限性：①仅用于问题分类与筛选，无法自动生成完整 QA 对；②需要访问模型内部隐藏状态，无法用于闭源 API；③启动时需已有较大领域问题集，收集成本较高；④对具有内在歧义或争议性的问题识别仍有限。

---

## 451. AR-CoPO: Align Autoregressive Video Generation with Contrastive Policy Optimization

**arXiv ID:** 2603.17461 | [PDF](https://arxiv.org/pdf/2603.17461v1)

**作者:** Dailan He `[一作]` (Chinese University of Hong Kong), Hongsheng Li `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 41100 | [OpenAlex ID](https://openalex.org/A5100732450)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种针对流式自回归视频生成模型的后训练对齐框架AR‑CoPO，通过在随机分支块处生成邻域候选并使用对比策略优化；

**💡 创新点**

创新点在于：1）将邻域对比思路迁移到自回归块级别，实现局部可控的信用分配；2）引入半在线策略，结合参考回放与信任区间以提升质量；3）用LoRA分离训练并合并实现探索与利用平衡；

**🔧 技术方法**

采用对比策略优化（CoPO）+自回归块级分支+一致性模型（Self‑Forcing/Consistency Model）采样+LoRA微调+半在线与在线训练策略+温度与剪切的策略梯度；

**📊 数据集**

主要使用MovieGen Video Bench进行训练，评估基于VideoAlign奖励（文本对齐、视频质量、运动质量）以及VBench（质量、语义、总分）等公开数据集；

**📈 对比分析**

与Self‑Forcing、Causal‑Forcing、LongLive等基线对比；在VBench上，Semi‑On‑Policy版提升Total至82.45，Merge版保持82.17；在VideoAlign上Overall从7.76提升至8.22，表明在不牺牲外域评测的前提下实现了更高的偏好对齐；

**⚠️ 局限性**

局限性包括：对局部噪声探索对全局语义对齐效果有限，导致On‑Policy训练易出现运动质量下降；半在线策略依赖参考模型，若参考分布偏差大可能导致性能不稳定；此外，模型仍需进一步验证在更大规模视频与多模态指令下的鲁棒性。

---

## 452. AdaZoom-GUI: Adaptive Zoom-based GUI Grounding with Instruction Refinement

**arXiv ID:** 2603.17441 | [PDF](https://arxiv.org/pdf/2603.17441v1)

**作者:** Siqi Pei `[一作]` (Lenovo Research), Jiwen Lu `[通讯]` (Tsinghua University)

**通讯引用:** 28998 | [OpenAlex ID](https://openalex.org/A5100460385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AdaZoom-GUI 框架，将 GUI grounding 分为指令精炼和条件放大两步，实现对高分辨率截图中小 UI 元素的精准定位。

**💡 创新点**

创新点包括：①指令精炼模块将模糊指令转换为可视化细节描述，提升模型理解度；②条件放大策略仅在预测框尺寸较小时触发二阶推理，兼顾精度与计算效率；③结合 GRPO 训练，使模型同时优化点击点和框坐标。

**🔧 技术方法**

采用大规模视觉语言模型 Qwen3-VL-4B-Instruct 做 grounding，Qwen3.5-397B-A17B 做指令精炼；使用 Group Relative Policy Optimization (GRPO) 进行强化学习训练；实现双阶段推理与阈值触发机制。

**📊 数据集**

构建了自研的高质量 GUI grounding 数据集，包含多应用领域截图、自然语言指令及对应边框标注，并利用 LLM 进行指令多样化。

**📈 对比分析**

在 ScreenSpot-Pro 与 ScreenSpot‑v2 基准上与多种 7B‑级 GUI 模型对比，基线 AdaZoom-GUI 在不加精炼时平均分 70.6，加入精炼后提升至 76.8，超过同参数量模型且接近更大模型；条件放大相较于无条件放大提升 3–5% 分数，证明自适应策略有效。

**⚠️ 局限性**

局限性：仅支持单步点击任务，无法处理多步骤交互；对极低分辨率或极小 UI 元素的鲁棒性仍有限；指令精炼模块依赖大模型，推理成本较高；数据集规模相对有限，需进一步扩展多样性。

---

## 453. Video Understanding: From Geometry and Semantics to Unified Models

**arXiv ID:** 2603.17840 | [PDF](https://arxiv.org/pdf/2603.17840v1)

**作者:** Zhaochong An `[一作]` (University of Copenhagen), Serge Belongie `[通讯]` (University of Copenhagen)

**通讯引用:** 136546 | [OpenAlex ID](https://openalex.org/A5018609918)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

这篇综述系统梳理了视频理解的三大视角——低阶几何恢复、高阶语义推理与统一建模，并提出将多任务知识融合为视频基础模型的趋势。

**💡 创新点**

创新点在于将传统分散的单任务研究按低级几何/高级语义/统一模型三层结构重新组织，提供了一个统一的框架和发展路线图，同时强调了从“单任务→多任务→统一”转变的必要性。

**🔧 技术方法**

主要技术手段包括：基于Transformer的时空建模、联合前馈几何模型、对齐优化与推理、跨模态（视觉-语言）对齐、以及对大规模预训练与多模态融合的综合评述。

**📊 数据集**

覆盖了多种公开数据集，如KITTI、7Scenes、DAVIS、Co3D、NRGBD、MSVD-QA、MSRVTT-QA、TGIF-QA、ActivityNet-QA、EgoSchema、CinePile、Video-MME、LongVideoBench、MLVU等。

**📈 对比分析**

对比方法时主要采用各任务的标准评价指标（如Abs Rel、δ1.25、A30、AJ、δ_avg、OA、F-score、mAP等），综述中列出了各主流方法的性能曲线，强调了基于预训练与多模态方法在多任务上往往表现更优。

**⚠️ 局限性**

局限性：①综述仅停留在文献级别，缺乏统一的实验复现与统一评测基准；②对最新大模型（如LLM、Diffusion）在视频领域的细粒度评估仍不充分；③未深入探讨模型在实际长视频实时推理、能耗与资源需求等工业化指标。

---

## 454. From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation

**arXiv ID:** 2603.17588 | [PDF](https://arxiv.org/pdf/2603.17588v1)

**作者:** Pujun Zheng `[一作]` (East China Normal University), Wei Lu `[通讯]` (Wuhan University)

**通讯引用:** 13669 | [OpenAlex ID](https://openalex.org/A5035830977)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于对比的LLM评估框架CNPE，用于论文的相对排名而非绝对评分。

**💡 创新点**

创新点在于将对比采样、相对评估的SFT与RL结合，并采用图形相似性选取高信息量论文对。

**🔧 技术方法**

使用Graph-based Ranking with Bidirectional Retrieval (GBR-BR)进行对比采样，GRPO强化学习，Bradley‑Terry模型聚合偏好，基础模型为Qwen2.5‑7B‑Instruct并加LoRA。

**📊 数据集**

主要数据集为ICLR‑2025的论文和评审分数，另外在ICML、NeurIPS、ACL、EMNLP、NAACL进行泛化验证。

**📈 对比分析**

与DeepReview‑14B等基线对比，CNPE在决策与排名指标上平均提升21.8%，在ICLR‑2025上取得最优性能。

**⚠️ 局限性**

局限包括仅使用计算机科学会议论文、模型参数有限、仅用标题摘要而非全文、潜在偏见与缺乏人类审核等。

---

## 455. Synchronized DNA sources for unconditionally secure cryptography

**arXiv ID:** 2603.17149 | [PDF](https://arxiv.org/pdf/2603.17149v1)

**作者:** Sandra Jaudou `[一作]` (Gulliver CNRS, ESPCI Paris, Université PSL), Yannick Rondelez `[通讯]` (Gulliver CNRS, ESPCI Paris, Université PSL)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过合成随机DNA池，分离、扩增、拆分制备双复制DNA钥匙，随后在东京和巴黎使用纳米孔测序生成共享OTP密钥。

**💡 创新点**

首次利用天然双链DNA的复制特性实现跨距离同步随机源，提供无条件安全的OTP密钥分发，类似量子密钥分发但无距离限制。

**🔧 技术方法**

采用化学DNA合成、PCR扩增、UMI标记、Oxford Nanopore纳米孔测序、BCH纠错码与统计检测等技术。

**📊 数据集**

使用IDT合成的约2百万至3千万条随机DNA键，实验中共生成约316-400 Mb的二进制密钥。

**📈 对比分析**

与FIPS140‑3批准的随机数生成器及商业RNG比较，min‑entropy≈0.96，误差率约5×10⁻⁵；通过BCH纠错满足2⁻¹²⁸失真率，实验实现东京-巴黎1000+km的OTP加解密。

**⚠️ 局限性**

主要限制包括合成与测序成本、时延、测序误差、实验步骤复杂性、存储与运输安全性以及技术成熟度待提升。

---

## 456. Anonymous-by-Construction: An LLM-Driven Framework for Privacy-Preserving Text

**arXiv ID:** 2603.17217 | [PDF](https://arxiv.org/pdf/2603.17217v1)

**作者:** Federico Albanese `[一作]` (Veritran), Nicolás D'Ippolito `[通讯]` (University of San Andrés)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种本地LLM驱动的类型保持替换匿名化流水线，能在不暴露敏感信息的前提下保持文本语义与实用性。

**💡 创新点**

创新点在于：①采用单一提示实现全句式PPI检测与替换；②保持类型一致的逼真替代，避免传统删除导致的语义损失；③在同一评估框架下统一衡量隐私、语义与可训练性。

**🔧 技术方法**

使用本地大模型（GPT‑oss 20B、DeepSeek‑r1 7B）以及零样本提示、无随机性的解码；对比传统规则与NER基线（Presidio、Google DLP）和基于BERT的ZSTS；评估指标包含隐私召回、情感一致性、主题距离、问答准确率以及LoRA微调MAE。

**📊 数据集**

使用Action‑Based Conversation Dataset（ABCD）进行多指标评估，涵盖约10k人类对话与多类PII标签。

**📈 对比分析**

在所有基准中，本地LLM替换实现了最高的隐私召回（0.99）和问答准确率（≈0.95），情感与主题保持无显著漂移，LoRA微调MAE仅为0.029–0.038，远优于Presidio、DLP和ZSTS。

**⚠️ 局限性**

局限性包括：高算力消耗；Q&A评估仅基于50条对话；仅评估了情感回归任务；未在多语种或多领域数据上验证，且模型规模较大。

---

## 457. Cascade-Aware Multi-Agent Routing: Spatio-Temporal Sidecars and Geometry-Switching

**arXiv ID:** 2603.17112 | [PDF](https://arxiv.org/pdf/2603.17112v1)

**作者:** Davide Di Gioia `[一作]` `[通讯]` (University College London), Davide Di Gioia (University College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在动态多智能体执行图中识别几何盲区，构建空间‑时间侧车对路由风险进行建模，并引入可学习的几何选择门实现自适应几何推理；在Genesis 3系统上验证其效果。

**💡 创新点**

首次将失效传播的几何感知视为系统级可观测性缺口；提出轻量化的几何选择门（仅133参数），在实时执行图中实现超越固定欧氏/双曲几何的自适应路由；量化该机制在不同拓扑模式下的系统级收益。

**🔧 技术方法**

利用欧氏扩散与双曲Poincaré球面嵌入、时间衰减的失效强度与自激爆发项、三种几何感知特征（BFS壳层增长斜率、循环秩、拟合曲率），以及两层MLP选择门；将侧车路由风险与原始bandit信号混合。

**📊 数据集**

主要使用Genesis 3的243条真实推理轨迹（加上250个带注入失效的合成场景），并在Barabási–Albert、Watts–Strogatz、Erdős–Rényi三种合成网络上进行交叉架构评估。

**📈 对比分析**

与原生Genesis 3调度器（仅团队适合度×负载）对比：原生50.4%胜率，欧氏48%；固定双曲提升至64–72%；学习门实现87.2%胜率、0.3846均值边距，系统级提升+36.8pp；在三种合成网络上，传播模型始终占优，学习门在非树形或混合模式下进一步提升。

**⚠️ 局限性**

仅在Genesis 3单一系统评估，缺乏公开多智能体系统验证；对攻击节点选取方式存在潜在偏倚；学习门在部分循环模式下预测误差较高；均值边距与胜率存在权衡；理论证明仅限于可观测性与增益机制，未给出最优性或泛化证明。

---

## 458. Implementation of tangent linear and adjoint models for neural networks based on a compiler library tool

**arXiv ID:** 2603.16976 | [PDF](https://arxiv.org/pdf/2603.16976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb`

---

## 459. Revisiting foundation models for cell instance segmentation

**arXiv ID:** 2603.17845 | [PDF](https://arxiv.org/pdf/2603.17845v1)

**作者:** Anwai Archit `[一作]` (Georg-August-University Göttingen), Constantin Pape `[通讯]` (Georg-August-University Göttingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估了多种基于 SAM 的细胞分割基础模型，并提出了无需再训练即可提升实例分割性能的自动提示生成（APG）方法。

**💡 创新点**

创新点在于利用 μSAM 预测的前景、边界和中心距离生成多重点提示，并通过 NMS 过滤后得到更精确的实例分割，显著提高了多任务下的分割准确率。

**🔧 技术方法**

使用了 SAM、SAM2、SAM3、CellSAM、μSAM、PathoSAM、CellPoseSAM 等模型，并结合点提示、阈值过滤、非极大抑制和平均分割准确率评估等技术。

**📊 数据集**

实验覆盖 36 个光学显微镜数据集，涵盖荧光细胞、荧光细胞核、无标记细胞和组织病理学核等四种任务。

**📈 对比分析**

通过在各子域内排名比较 9 个不同方法的平均分割准确率，发现 APG 在大多数数据集上显著提升 μSAM，整体与 CellPoseSAM 竞争，SAM3 的性能落后于域特定模型。

**⚠️ 局限性**

研究仅在 2D 级别进行评估，未处理 3D 数据；APG 未利用框提示或多次提示迭代；SAM3 未在显微镜数据上微调，也未测试示例提示策略。

---

## 460. Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment

**arXiv ID:** 2603.17655 | [PDF](https://arxiv.org/pdf/2603.17655v1)

**作者:** Yaze Zhao `[一作]` (Huazhong University of Science and Technology), Ruixuan Li `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 4190 | [OpenAlex ID](https://openalex.org/A5039670436)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于CLIP的自监督周期一致性框架（CC-CDFSL），通过在跨域少样本学习中对局部视觉特征与文本语义进行循环映射并利用语义锚机制实现局部特征的精细对齐和解释性提升。

**💡 创新点**

创新点在于：①首次针对CLIP在跨域少样本场景下的局部特征失配问题提出自监督的T‑I‑T与I‑T‑I双向循环一致性约束；②引入语义锚模块在增广-缩减两阶段实现局部特征的高质量筛选；③通过循环一致性显式提升模型的解释性与判别力。

**🔧 技术方法**

核心技术包括CLIP视觉‑文本双编码器、视觉Transformer局部patch提取、两层MLP映射至文本空间、余弦相似度与循环一致性损失（T‑I‑T、I‑T‑I）、语义锚增广‑缩减策略以及多任务联合训练。

**📊 数据集**

实验数据集涵盖CropDiseases、EuroSAT、ISIC2018、ChestX等四个跨域目标域，使用CLIP ViT‑Base/16作为基础模型。

**📈 对比分析**

与多种PEFT基线（CoOp、Tip‑Adapter、CLIP‑Adapter、CLIP‑LoRA、Maple、AMU‑Tuning等）以及传统少样本学习方法比较，CC‑CDFSL在所有任务与数据集上均实现显著提升，尤其在5‑way 5‑shot场景下平均提高约1.5%–4.5%准确率，并在基准任务中达到SOTA水平。

**⚠️ 局限性**

主要局限包括：①对少量数据的自监督循环一致性依赖较高，过度依赖文本描述可能导致语义偏差；②语义锚选择k值的敏感性需进一步自动化；③在极低样本或高噪声场景下的鲁棒性仍待验证。

---

## 461. 3D MRI-Based Alzheimer's Disease Classification Using Multi-Modal 3D CNN with Leakage-Aware Subject-Level Evaluation

**arXiv ID:** 2603.17304 | [PDF](https://arxiv.org/pdf/2603.17304v1)

**作者:** Md Sifat `[一作]` (University of Rajshahi), Jungpil Shin `[通讯]` (University of Aizu)

**通讯引用:** 4670 | [OpenAlex ID](https://openalex.org/A5005221038)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出一种多模态3D卷积神经网络，利用原始OASIS-1 MRI体积及其灰质、白质、脑脊液概率图进行阿尔茨海默病二分类；

**💡 创新点**

创新点在于：①在严格受试者级交叉验证下使用体积数据而非切片，避免泄漏；②融合T1与多模态组织概率图，捕获更多病理相关结构信息；③提出泄漏感知的评估协议并结合GradCAM解释模型关注的解剖区域；

**🔧 技术方法**

使用3D卷积网络、轻量级多通道编码器、late‑fusion分类头、ANTsPy/N4去偏、FSL BET/FAST/FLIRT预处理、GradCAM可视化；

**📊 数据集**

数据集为临床标注的OASIS‑1 235个受试者的原始3D体积（T1+GM+WM+CSF），并在对照实验中使用Kaggle重分发的切片数据；

**📈 对比分析**

采用5折受试者级交叉验证，平均准确率72.34%±4.66%，AUC0.7781±0.0365；相比传统2D切片方法（如Tufail等63–65%）显著提升；切片级别实验在受试者级评估下准确率降至79.25%，说明体积多模态模型更具鲁棒性；

**⚠️ 局限性**

局限性包括：标注受试者仅235个，样本量有限；将所有CDR>0归为痴呆的二分类限制了疾病分期；轻量化网络可能未能充分挖掘更复杂特征；对真实临床数据的泛化性仍待验证。

---

## 462. Security Assessment and Mitigation Strategies for Large Language Models: A Comprehensive Defensive Framework

**arXiv ID:** 2603.17123 | [PDF](https://arxiv.org/pdf/2603.17123v1)

**作者:** Taiwo Onitiju `[一作]` (University of North Florida), Iman Vakilinia `[通讯]` (University of North Florida)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5008183355)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对五大主流LLM在10,000个对抗提示下进行系统性漏洞评估，并提出低延迟多层防御框架。

**💡 创新点**

首次跨架构比较、提出可解释的特征加权检测模型、并实现了生产级别的防御系统。

**🔧 技术方法**

多层检测（正则快速过滤、语义嵌入相似度、行为分类、主动学习）+ 语义特征提取与阈值组合。

**📊 数据集**

由JailbreakChat、GitHub/Discord/Reddit/Twitter收集并合成的10,000条对抗提示，按六大攻击类别标注。

**📈 对比分析**

统一API测试、固定温度/长度，对五个模型同一批提示评估；防御系统平均检测率83%，误报率5%，平均延迟15.4 ms，优于现有商业与开源方案。

**⚠️ 局限性**

仅覆盖公开API模型，未考虑更高级持续威胁；误报在某些合法用例（如创意写作）略高；缺乏对模型内部机制的深入分析。

---

## 463. Attention Sinks Induce Gradient Sinks

**arXiv ID:** 2603.17771 | [PDF](https://arxiv.org/pdf/2603.17771v1)

**作者:** Yihong Chen `[一作]` (Tsinghua University), Quanming Yao `[通讯]` (Tsinghua University)

**通讯引用:** 9596 | [OpenAlex ID](https://openalex.org/A5072484211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究从反向传播视角探讨 Transformer 的关注点（Attention sink）与大激活（Massive activation）之间的关联，提出梯度汇聚（gradient sink）概念，并通过在值路径上引入 V-scale 梯度阀门验证该机制；

**💡 创新点**

创新点在于揭示关注点通过梯度集中作用于训练时梯度汇聚，进而触发大激活；并提出仅调节值路径梯度即可在保持关注点的同时抑制大激活，证明了两者的因果中介关系；

**🔧 技术方法**

采用梯度归一化比率、RMSNorm 解析、理论推导以及自定义 V-scale 参数化实现对值路径梯度的调控；

**📊 数据集**

在 C4 文本数据集上对 Llama‑style 0.1B 与 0.3B 模型进行预训练与微调；

**📈 对比分析**

通过比较关注点率、残差流输出范数、MLP 输出范数等指标发现 V-scale 模型在保持或提升关注点的同时显著降低大激活，验证损失略低，性能保持良好；

**⚠️ 局限性**

局限性包括仅在 Llama‑style 稠密语言模型上验证，未测试混合专家或多模态 Transformer；研究侧重机制验证而非完整训练过程；未进行大规模超参搜索或下游任务评估。

---

## 464. Evidence Packing for Cross-Domain Image Deepfake Detection with LVLMs

**arXiv ID:** 2603.17761 | [PDF](https://arxiv.org/pdf/2603.17761v1)

**作者:** Yuxin Liu `[一作]` (Anhui University), Zhaohong Jia `[通讯]` (Anhui University)

**通讯引用:** 1154 | [OpenAlex ID](https://openalex.org/A5017935525)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑无关的语义一致性证据包（SCEP），通过对视觉编码器的CLS标记进行语义聚类并结合频率与噪声异常评分，筛选出少量富含伪造痕迹的补丁，作为冻结大型视觉‑语言模型（LVLM）的证据输入，实现跨域图像深伪检测。

**💡 创新点**

创新点在于（1）用CLS驱动的语义一致性聚类将证据限定在语义一致的区域，减少语义干扰；（2）融合频率（DCT能量分布）与噪声残差两种低级异常，提升对细粒度伪造痕迹的感知；（3）构造“证据包”而非全图推理，既提高检测性能又降低推理延迟；（4）实现全流程训练‑无关，避免昂贵的Fine‑Tuning。

**🔧 技术方法**

技术包括冻结视觉编码器提取CLS与补丁嵌入、球面k‑means聚类、CLS‑监督的语义不一致度量、DCT频率分布与Jensen‑Shannon散度、残差噪声能量与MAD标准化、融合评分与grid‑based NMS、以及对LVLM的prompt‑驱动推理。

**📊 数据集**

使用公开的DFBench基准（Real、AI‑Edited、AI‑Generated三子集）以及多种真实图像质量数据集（LIVE、CSIQ、TID2013、KADID、KonIQ‑10k）进行评估。

**📈 对比分析**

与现有方法（如A‑ViT、DynamicViT、以及各类冻结LVLM的全图或token‑剪枝方案）对比，SCEP在所有LVLM上均实现了显著提升，最高可提升≈2–3%准确率、≈4–5% F1，并且推理时间平均减少≈1.6×。

**⚠️ 局限性**

局限性：①对CLS标记的依赖可能在极端语义变形或大尺度伪造时失效；②仅利用视觉低级特征，可能对复杂的高阶伪造（如全景生成或多模态伪造）识别不足；③在极少量数据的自适应场景中仍需进一步验证。

---

## 465. Universal Skeleton Understanding via Differentiable Rendering and MLLMs

**arXiv ID:** 2603.18003 | [PDF](https://arxiv.org/pdf/2603.18003v1)

**作者:** Ziyi Wang `[一作]` (Peking University), Mengyuan Liu `[通讯]` (Peking University)

**通讯引用:** 4355 | [OpenAlex ID](https://openalex.org/A5100705472)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 SkeletonLLM，利用可微分渲染器 DrAction 将任意骨架序列转化为 MLLM 原生视觉输入，实现统一的骨架理解。

**💡 创新点**

创新点在于构建格式无关的可微分渲染管线，使骨架动力学直接映射为视觉特征，并引入因果推理蒸馏与判别微调的协同训练策略，提升跨格式泛化与细粒度推理。

**🔧 技术方法**

核心技术包括 3D 高斯 splatting、线性混合蒙皮、神经特征调制器，以及与 InternVL3 等多模 LLM 的端到端梯度耦合。

**📊 数据集**

实验使用 NTU‑60/120、HumanML3D、NW‑UCLA 及 2D 关键点数据，覆盖多种骨架拓扑与任务。

**📈 对比分析**

在开窗动作识别、跨格式迁移、动作描述与问答等任务中，SkeletonLLM 均优于传统对齐/Token 化方法，尤其在极限数据稀缺和格式切换时提升超过10% 的准确率。

**⚠️ 局限性**

局限性包括对 MLLM 预训练的高度依赖、渲染器训练成本较高，以及对极度噪声或不完整骨架数据的鲁棒性尚未充分验证。

---

## 466. TeleDex: Accessible Dexterous Teleoperation

**arXiv ID:** 2603.17065 | [PDF](https://arxiv.org/pdf/2603.17065v1)

**作者:** Omar Rayyan `[一作]` (University of California), Yuchen Cui `[通讯]` (University of California)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了 TeleDex，一套利用智能手机实现无外部追踪的轻量级手部与机器人臂远程操控与数据采集系统。

**💡 创新点**

创新点在于：①将手机的 6-DoF 轨迹与 21-DoF 手部关节估计实时流式传输；②提供可 3D 打印的腕部支架，实现无额外硬件的指尖级操作；③将手部姿态映射到任意 URDF 机器人手模型；④与 MuJoCo、现实机器人无缝集成，支持多平台快速部署。

**🔧 技术方法**

主要技术包括：ARKit 6-DoF 追踪、前置摄像头手势估计、Dex-Retarget 关节映射、Python API 与移动端 SDK、基于 UDP/TCP 的低延迟数据流、可自定义 UI 按钮与控制锁定。

**📊 数据集**

数据集：在 MolmoSpaces‑Bench 进行 225 次任务演示（SpaceMouse/键盘/TeleDex）；在真实环境下收集 120 条 TeleDex 采样演示，用于后续的策略训练；此外使用 xArm7、RealMan、KUKA iiwa 等机器人平台进行实验。

**📈 对比分析**

对比方法：与传统键盘、SpaceMouse 进行演示收集效率比较，显示 TeleDex 平均成功时间更短；在真实手部操控任务中与人工“oracle”对比，完成时间约为 3–4 倍，但可实现所有任务；演示数据训练的行为克隆模型在仿真 pick‑and‑place 任务中 80% 成功率，证明数据质量可靠。

**⚠️ 局限性**

局限性：腕部支架加重手部负担，长期使用可能导致疲劳；依赖手机前置摄像头，光照或遮挡会影响手势估计；系统在极低光或高速动态场景下的精度尚未充分验证。

---

## 467. Caging the Agents: A Zero Trust Security Architecture for Autonomous AI in Healthcare

**arXiv ID:** 2603.17419 | [PDF](https://arxiv.org/pdf/2603.17419v1)

**作者:** Saikat Maiti `[一作]` (Commure), Saikat Maiti `[通讯]` (nFactor Technologies)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文构建并在真实医疗技术公司部署了一套针对九个自主 AI 代理的四层深度防御体系，结合自动化审计与持续安全监控，展示了从基线到硬化的90天进化路径。

**💡 创新点**

创新点：①将 HIPAA 安全规则与自主代理威胁模型映射，形成六域风险框架；②提出针对代理的四层防御（gVisor 沙箱、凭证代理侧车、网络 egress 策略、提示完整性框架）；③实现基于代理的自动化安全审计（Tony），并在生产中实时修复四个高危缺口；④公开完整的 K8s 配置、审计工具与完整提示完整性框架。

**🔧 技术方法**

技术：Kubernetes + gVisor 运行时沙箱、Sidecar 凭证代理、NetworkPolicy 允许列表、Temporal 工作流编排、加密元数据包（提示完整性）、OpenClaw 框架、Claude via AWS Bedrock、Google Cloud Compute Engine、VPC Flow Logs、Cloudflare Gateway DNS 过滤、Fail2Ban、UFW、iptables 日志、GCP Admin Activity 审计日志。

**📊 数据集**

数据集：论文未使用公开数据集，全部测试基于公司内部的生产代理与 PHI 数据，利用 Shapira 等人的红队实验案例作为威胁验证场景。

**📈 对比分析**

比较方法：对比基线、硬化版 1、硬化版 2、K8s 目标四个阶段的安全姿态指标（凭证暴露、网络 egress、提示完整性、配置漂移）。实验结果显示四个高危发现被即时修复，安全态势从 “Critical” 下降到 “Medium/Low”，并覆盖 11 个已验证攻击模式中的 9 个；缺乏传统性能基准，仅通过安全发现率和修复速度进行评估。

**⚠️ 局限性**

限制：①提示完整性层依赖 LLM 遵循规则，仍然脆弱；②审计代理本身成为高价值攻击目标，存在安全悖论；③网络 egress 与凭证代理需要持续更新允许列表与凭证，维护成本高；④对某些攻击模式（如情感操控）只能部分缓解；⑤论文未给出公开可复现的安全度量或对比基准。

---

## 468. FINER: MLLMs Hallucinate under Fine-grained Negative Queries

**arXiv ID:** 2603.17662 | [PDF](https://arxiv.org/pdf/2603.17662v1)

**作者:** Rui Xiao `[一作]` (Technical University of Munich), Stephan Alaniz `[通讯]` (LTCI, Telecom Paris, Institute Polytechnique de Paris)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种细粒度负面查询（FINER）及其对应的评测基准（FINER-CompreCap 和 FINER-DOCCI），并通过基于直接偏好优化（DPO）的微调方法显著提升多模态大型语言模型在细粒度否定查询上的拒绝准确率。

**💡 创新点**

创新点包括：①从单对象/属性/关系扩展到多对象、多属性、多关系及“what”类细粒度问答；②构建细粒度负面查询的全新基准；③采用仅使用公开 LLM 的 DPO 训练流程，无需闭源模型或多轮迭代，能同时降低幻觉并提升通用能力。

**🔧 技术方法**

技术手段：①直接偏好优化（DPO）与 LoRA 微调；②基于场景图的负面实体生成（使用 Qwen3-14B / Gemini-2.0-Flash 生成四种负面变体并用 Qwen2.5-VL-72B 做判别；③多种数据管道：从 CompreCap、DOCCI、Pixmo 采样长描述，利用 LLM 提取正面短语并构造负面短语；④MCQ 评估框架，采用配对准确率（paired accuracy）衡量模型对正负问答的一致性。

**📊 数据集**

使用的数据集包括：CompreCap（COCO 场景图）、DOCCI（5k 长文本图像及手工标注的场景图）、Pixmo（长描述集），以及公开的幻觉基准（DASH、POPE、RePOPE、HallusionBench、AMBER、CRPE_R、MMHal-Bench、HaloQuest）和通用多模评测基准（MMStar、TextVQA、ChartQA、MMVP、NaturalBench、V*）。

**📈 对比分析**

与现有基准模型（LLaVA-NeXT、Qwen2.5-VL、InternVL-3.5 等）以及多种幻觉抑制方法（RLHF-V、LLaVA-RLHF、OPA-DPO、RLAIF-V 等）对比，FINER 训练可在 FINER-CompreCap 上提升多目标场景的配对准确率达 23–25%（多对象/属性/关系），在 FINER-DOCCI 上提升 16–20%。在其他幻觉基准中，DPO 方法统一提升 6–10% 的准确率或 10% 的幻觉率降低；在六大通用评测基准中，模型整体得分平均提升 1–2%。

**⚠️ 局限性**

局限性：①FINER 基准部分仍以规则生成，缺乏完全人工验证，可能存在噪声；②MCQ 形式与自然问句的自然度受限；③多关系子集最多仅包含三条关系，规模有限；④对极高粒度或长链推理的挑战仍未解决。

---

## 469. A Longitudinal Study of Usability in Identity-Based Software Signing

**arXiv ID:** 2603.17133 | [PDF](https://arxiv.org/pdf/2603.17133v1)

**作者:** Kelechi G. Kalu `[一作]` (Purdue University), James C. Davis `[通讯]` (Purdue University)

**通讯引用:** 2802 | [OpenAlex ID](https://openalex.org/A5004592401)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对五个身份基签名生态系统（Sigstore、Notary v2、OpenPubKey、Keyfactor、HashiCorp Vault）的GitHub issue进行挖掘，提取并编码可用性相关问题，分析这些问题在工具功能与组件层面的分布，以及随时间的趋势变化；

**💡 创新点**

首次以跨工具纵向方式系统研究身份基软件签名工具的可用性表现，结合主题分析与Poisson趋势模型，揭示可用性问题在不同组件间的迁移与演化，并为未来工具设计提供针对性改进方向；

**🔧 技术方法**

采用矿化软件仓库技术获取 issue 数据，利用 LLM 辅助编码与人工校准实现规模化主题标注，随后运用主题分析（归纳一级/二级主题）与 Poisson 回归模型对随时间的可用性问题频率进行量化；

**📊 数据集**

约 3,900 条 GitHub issue（2021‑11 ~ 2025‑11），覆盖 Sigstore（Cosign、Fulcio、Rekor）、Notary v2（Notation）、OpenPubKey、Keyfactor（SignServer、EJBCA）、HashiCorp Vault；

**📈 对比分析**

通过计算各工具/组件的主题频率与主题/组件交叉比率进行横向比较，并使用 Poisson 回归检验每月问题数量随时间的斜率；结果显示 Sigstore、Vault 等工具可用性问题显著下降，而 OpenPubKey、Keyfactor 部分主题则持平或上升，说明可用性改进具有工具与组件依赖性；

**⚠️ 局限性**

仅依赖公开 issue，忽略了企业私有反馈、监控日志、内部讨论等渠道；LLM 代码辅助可能引入误判，尽管已人工校准；工具样本有限，可能无法覆盖所有身份基签名实现的多样性；不同项目的 issue 文化与治理差异可能影响问题报告频率与内容的可比性。

---

## 470. Negation is Not Semantic: Diagnosing Dense Retrieval Failure Modes for Trade-offs in Contradiction-Aware Biomedical QA

**arXiv ID:** 2603.17580 | [PDF](https://arxiv.org/pdf/2603.17580v1)

**作者:** Soumya Ranjan Sahoo `[一作]` (GE HealthCare), Divya Bharti `[通讯]` (GE HealthCare)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种对照证据可检索的医学问答系统，将支持与矛盾检索拆分并使用BM25与NegEx过滤实现高召回与精确平衡，并设计叙事感知重排序与一次性上下文学习以实现全文引用覆盖。

**💡 创新点**

创新点在于揭示“简约悖论”，将支持和矛盾检索解耦、引入词法级别的NegEx过滤避免语义坍塌，同时通过叙事感知重排序和一次性上下文学习保证引用完整性。

**🔧 技术方法**

使用了BM25检索、MedCPT交叉编码器重排序、MedNLI调优的T5生成模型、NegEx规则过滤、GPT‑4o生成及一次性上下文学习等技术。

**📊 数据集**

使用SciFact数据集作为代理进行开发与评估，并在30亿篇PubMed文献与TREC BioGen 2025的官方评测数据上进行验证。

**📈 对比分析**

通过在SciFact上计算加权MRR，官方TREC评测中在任务A中得到支持F1 53.53、矛盾F1 8.57（排名第二），在任务B中得到引用覆盖率98.77%（排名第三）且0%矛盾引用率。

**⚠️ 局限性**

局限包括代理转移与规模、语言、领域差距；仅二元支持/矛盾标注，缺乏时效性；生成模型可能出现后置合理化；依赖GPT‑4o，缺乏可复现性，NegEx规则不覆盖所有否定表达。

---

## 471. PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation

**arXiv ID:** 2603.17520 | [PDF](https://arxiv.org/pdf/2603.17520v1)

**作者:** Jianjian Yin `[一作]` (Nanjing University of Science and Technology), Fumin Shen `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 13140 | [OpenAlex ID](https://openalex.org/A5074492050)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了并行成本聚合（PCA‑Seg）范式，以缓解类级语义与空间上下文之间的知识干扰，从而提升开放词汇语义与部件分割性能。

**💡 创新点**

创新点在于引入专家驱动感知学习（EPL）模块，通过多专家解析与可学习像素权重融合多视角特征，并配合特征正交解耦（FOD）策略实现语义与空间特征的正交化，显著降低冗余。

**🔧 技术方法**

使用 CLIP 视觉与文本编码器、并行空间与类聚合、Swin Transformer、卷积多专家块、系数映射、正交解耦损失以及端到端微调等技术。

**📊 数据集**

在 COCO‑Stuff、ADE20K、PASCAL‑Context、PASCAL‑VOC 等标准开放词汇语义分割基准，以及 Pascal‑Part‑116、ADE20K‑Part‑234 部件分割数据集上训练和评估。

**📈 对比分析**

与现有串行聚合方法（如 DeCLIP、CAT‑Seg、H‑CLIP 等）对比，PCA‑Seg 在八大基准上实现了显著的 mIoU 提升（最高可达 4.5% 以上），在部件分割上提升 3‑4% 的 h‑IoU。

**⚠️ 局限性**

限制在于增加了 0.35M 参数与 0.96G GPU 内存开销，且在极端小模型或低算力设备上的部署仍需进一步优化。

---

## 472. Tabular LLMs for Interpretable Few-Shot Alzheimer's Disease Prediction with Multimodal Biomedical Data

**arXiv ID:** 2603.17191 | [PDF](https://arxiv.org/pdf/2603.17191v1)

**作者:** Sophie Kearney `[一作]` (University of Pennsylvania), Li Shen `[通讯]` (University of Pennsylvania)

**通讯引用:** 15631 | [OpenAlex ID](https://openalex.org/A5100768717)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出并实现了TAP-GPT，一种针对阿尔茨海默病（AD）预测的少量样本、可解释性强的表格大型语言模型框架；

**💡 创新点**

创新点在于将专为表格设计的TableGPT2与大规模语言模型的语义推理能力结合，并通过参数高效微调实现领域适配；

**🔧 技术方法**

使用了Qwen2.5-7B作为解码器、QLoRA进行低秩微调、TableGPT2的语义表格编码器，以及多种提示格式（零样本/少样本、表格/序列化）；

**📊 数据集**

使用了来自ADNI的四个数据集：QT-PAD（15个临床生物标记）、结构MRI（74个区域）、Amyloid PET（68个区域）和Tau PET（68个区域）；

**📈 对比分析**

与传统机器学习（LR、RF、XGBoost、SVM）、TabPFN、未微调的TableGPT2、TableGPT-R1、通用LLM（Qwen2.5-Instruct、Qwen3、GPT‑4.1‑mini）比较；在少样本情境下TAP‑GPT在QT‑PAD的少样本表格提示上达到平均F1≈0.89，显著优于传统方法和TabPFN；在影像数据中表现与GPT‑4.1‑mini相近或更好；并在解释性、缺失值鲁棒性和自我反思稳定性上优于对比模型；

**⚠️ 局限性**

局限包括对高维长表格的适应仍受限（需特征选择）、对超大表格（p=72）性能下降、依赖于特定预训练基模型（TableGPT2）且不易替换为更强大的解码器、以及在自我反思下仍存在解释不一致的问题。

---

## 473. HierarchicalKV: A GPU Hash Table with Cache Semantics for Continuous Online Embedding Storage

**arXiv ID:** 2603.17168 | [PDF](https://arxiv.org/pdf/2603.17168v1)

**作者:** Haidong Rong `[一作]` (NVIDIA), Even Oldridge `[通讯]` (NVIDIA)

**通讯引用:** 263 | [OpenAlex ID](https://openalex.org/A5023954126)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `a2602d71-93ab-4bad-974b-672788df8193`

**🎯 论文内容**

本文提出了 HierarchicalKV，一种面向 GPU 的哈希表库，采用缓存语义实现连续在线嵌入存储；哈希表通过在桶内部直接完成 evict / admission / CAS，支持在满桶时仍能在线处理插入，且无需 rehash 或外部维护。

**💡 创新点**

创新点包括：
• 单桶 L1‑对齐 128 slot 的哈希桶与连续 digest 数组，实现一次 cache‑line 读取即可完成一次完整 miss；
• 线性扫描 + score‑驱动的 in‑line upsert，省去后台 eviction 结构；
• 动态双桶选择（Power‑of‑Two‑Choices 改进），在满载时根据最小 score 决定 eviction，显著提升热点保留；
• 三组并发协议（Reader / Updater / Inserter）将结构性写与非结构性写分离，配合 CPU‑GPU 双层锁，避免 R/W 互斥导致的吞吐瓶颈；
• 基于位置寻址的 key‑value 分层存储（tiered KV separation），让键、digest、score 固定驻留 HBM，值溢出至 HMEM，既扩展容量又保持 key 端的 GPU 计算不受影响。

**🔧 技术方法**

使用的技术主要包括：
- GPU L1 cache‑line（128 B）对齐的桶布局与 1‑byte digest 预过滤；
- 采用 Murmur3 哈希的 8‑bit digest；
- 逐槽 CAS 作为排他锁和写入原子操作；
- 双阶段并行 kernel（TLPv1、TLPv2、Pipeline）根据 load factor 与 value 大小动态切换；
- 双桶选择的两阶段策略（先基于 load，后基于 score）；
- CPU‑GPU 双层锁与 atomic counter 进行角色切换；
- 使用 CUDA pinned memory + zero‑copy 访问 HMEM 以实现值分层；
- 通过指针返回 API 进一步隐藏值拷贝延迟。

**📊 数据集**

实验使用合成数据：
- key 为 64‑bit 整数，value 维度 8、32、64，batch size 为 1 M；
- 负载因子 λ ∈ {0.50, 0.75, 1.00}；
- 访问模式包括均匀随机和 Zipfian（α≈0.99）模拟推荐系统的热点分布；
- 对比基准不涉及真实推荐数据集，而是以这些合成负载衡量吞吐与 cache‑hit 率。

**📈 对比分析**

与 WarpCore、BGHT、cuCollections、BP2HT 四个主流 GPU 哈希表做基准对比；在 NVIDIA H100 NVL GPU 上，HierarchicalKV 在 λ=0.5 的 insert 速率约为 3.4 B‑KV/s，超过 WarpCore 1.4×，相较于 indirection‑based 基准提升 2.6–9.4×；在 λ=1.0 时仍保持 3.37–3.40 B‑KV/s 的稳定吞吐；采用双桶模式后，吞吐与单桶相当甚至更高；在 HBM+HMEM 混合模式下，key‑side throughput 仅下降 <4%，而 value‑copy API 由于 PCIe 限制下降 95%。

**⚠️ 局限性**

局限性包括：
- 读/写互斥（Reader/Updater）会在高更新比例下成为瓶颈，无法完全并行更新；
- 仅支持单 GPU，跨 GPU 的分片需由应用层手动实现；
- 双桶模式仅在单桶路径下启用，某些 API 仍以单桶实现；
- 当前实现缺乏 SSD/GDS tiering 与动态 rehash，无法应对更大规模或多样化存储需求；
- 受限于 128 B L1 cache‑line 长度，桶大小受限，未来更宽 cache‑line 能进一步提升性能。

---

## 474. Governed Memory: A Production Architecture for Multi-Agent Workflows

**arXiv ID:** 2603.17787 | [PDF](https://arxiv.org/pdf/2603.17787v1)

**作者:** Hamed Taheri `[一作]` `[通讯]`, Hamed Taheri

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种共享内存与治理层，解决多代理工作流中的记忆孤岛、治理碎片化、无结构记忆、上下文冗余和质量退化问题，并在生产环境中实现。

**💡 创新点**

核心创新点包括：①双模记忆模型（开放式原子事实+模式强制类型属性）实现信息无损存储；②分层治理路由与渐进式上下文投递，解决治理碎片化与上下文冗余；③反射受限检索与实体范围隔离，提升检索完整性和安全性；④闭环模式生命周期与 AI 辅助迭代，实时监控并优化模式质量。

**🔧 技术方法**

技术手段涵盖：LLM 双提取管线、质量门控（核心指代、自含性、时间锚定）、向量检索+实体过滤、会话级差分投递、反射循环检索、AI 辅助模式编写与迭代、基于 rubric 的自评与日志、全程红点脱敏与安全分区、API 统一接口。

**📊 数据集**

主要数据集为自定义合成数据（250 篇文本，5 种类型、5 个实体、500 条检索查询、50 条治理变量对、30 条冲突对），以及公开 LoCoMo 基准（272 会话、1542 题），并对 50 条针对性治理攻击进行测试。

**📈 对比分析**

通过与内部基线、Mem0、Zep、OpenAI 内置记忆等系统在 LoCoMo、检索完整性、治理路由精度、实体隔离、冲突解决等指标对比。实验结果显示：事实召回 99.6%，治理路由精度 92%，实体隔离 0% 泄漏，治理合规 100%，LoCoMo 总体准确率 74.8%（人类 87.9%），冲突检测 83.3%，多步检索完成度提升 25.7pp，检索完整度从 37.1% 提升到 62.8%。

**⚠️ 局限性**

局限性包括：质量门控依赖模式匹配，缺乏深度语义校验；红点脱敏为正则，可能漏检；并发写冲突未充分验证；推断与反射效果受查询生成策略限制；会话 TTL 与渐进投递假设上下文持续相关；模型置信度自校准的鲁棒性待提升。

---

## 475. Public Profile Matters: A Scalable Integrated Approach to Recommend Citations in the Wild

**arXiv ID:** 2603.17361 | [PDF](https://arxiv.org/pdf/2603.17361v1)

**作者:** Karan Goyal `[一作]` (Indraprastha Institute of Information Technology Delhi), Mukesh Mohania `[通讯]` (Indraprastha Institute of Information Technology Delhi)

**通讯引用:** 2460 | [OpenAlex ID](https://openalex.org/A5047987914)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套两阶段的本地引文推荐系统，先用无学习的 Profiler 模块基于公开引用网络做快速候选检索，再用自定义的 DAVINCI 重排器融合置信度先验与语义特征进行精细重排序。

**💡 创新点**

主要创新点包括：① 无学习的 Profiler 通过“公共档案”方式捕捉论文的网络可见度；② 引入严格的归纳式评估协议，逼近真实的时间序列推荐场景；③ DAVINCI 通过指数衰减的置信度转换与向量门控机制实现对置信度与语义的自适应融合。

**🔧 技术方法**

技术手段涵盖：无学习的图表征增强、余弦相似度检索、Specter2/ SciBERT 编码器、向量门控融合、对数优先级变换、三元组损失对排序直接优化，以及稀疏检索索引实现高效推理。

**📊 数据集**

实验使用五个大规模学术语料库：ACL‑200、FullTextPeerRead、RefSeer、arXiv 及 ArSyTa，并在每个数据集上按归纳式划分训练/验证/测试。

**📈 对比分析**

评估指标为 MRR、Recall@K 及 NDCG@K；与 Prefetcher/Enricher、SymTax、HAtten、SciNCL、BM25 等基线相比，Profiler 在检索阶段速度提升 30–40 倍、MRR 提升 50%+；DAVINCI 在终端性能上在所有数据集上均超越 SymTax、HAtten，且在 110M 参数下优于 2.7B 参数的通用重排器。

**⚠️ 局限性**

局限性包括：仅在英语计算机科学文本上验证；对极新论文的冷启动仍依赖网络可见度；512-token 输入限制可能截断长篇语境；对引用元数据缺失或噪声敏感；系统仍可能放大已有的引用偏见。

---

## 476. Binary Latent Protein Fitness Landscapes for Quantum Annealing Optimization

**arXiv ID:** 2603.17247 | [PDF](https://arxiv.org/pdf/2603.17247v1)

**作者:** Truong-Son Hy `[一作]` (University of Alabama at Birmingham), Truong-Son Hy `[通讯]` (University of Alabama at Birmingham)

**通讯引用:** 223 | [OpenAlex ID](https://openalex.org/A5073178563)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

将蛋白质序列嵌入预训练语言模型，压缩为二进制潜在空间后，用QUBO模型逼近并优化蛋白质适应度；

**💡 创新点**

首次将蛋白质适应度映射为二进制潜在空间中的可组合优化问题，兼容量子退火硬件；

**🔧 技术方法**

使用ESM-2语言模型、随机投影或PCA降维、二进制阈值化、岭回归拟合QUBO、模拟退火与遗传算法搜索；

**📊 数据集**

在ProteinGym基准的GFP深度突变扫描数据上进行实验；

**📈 对比分析**

与随机搜索、贪婪爬坡、贝叶斯搜索等方法对比，QUBO+模拟退火/遗传算法在邻近训练集序列中的适应度排名均显著提升，表现最优；

**⚠️ 局限性**

受限于二进制表示的表达能力、过度拟合风险和仅采用检索式解码，未实现真正生成新高适应度序列。

---

## 477. Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection

**arXiv ID:** 2603.16914 | [PDF](https://arxiv.org/pdf/2603.16914v1)

**作者:** Jinyang Wu `[一作]` (Agency for Science Technology and Research), Soumik Mondal `[通讯]` (Agency for Science Technology and Research)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了神经音频编解码器的离散向量量化（RVQ）残差层次结构，并提出了一种层次感知的深度伪造检测框架。

**💡 创新点**

创新点在于：① 对RVQ残差层次进行显式建模；② 引入轻量级的Quantizer‑Aware Static Fusion（QAF‑Static）对不同量化层分配可学习的全局权重；③ 在保持SSL编码器冻结的前提下，仅增加约4.4%的参数即可显著提升检测性能。

**🔧 技术方法**

采用的技术包括：WavLM‑Large 作为自监督音频编码器；Facebook EnCodec 的RVQ离散化；QAF‑Static 的维度级静态加权聚合；Attentive Merging 进行SSL层级融合；轻量级 LSTM+线性分类器；Adam 优化器和早停策略。

**📊 数据集**

使用的数据集为：ASVspoof 2019 LA、ASVspoof 5 以及 CodecFake benchmark（多种编码器族）。

**📈 对比分析**

与传统 SSL 基线（AttM、MoLEx 等）和全量微调方案相比，QAF‑Static 在 ASVspoof 2019 LA 上实现了 46.2% 的相对 EER 降低，在 ASVspoof 5 上提升了 13.9%，在 CodecFake 评测中对某些 codec 族亦优于 baseline。

**⚠️ 局限性**

局限性包括：① 对不同编码器结构的跨 codec 鲁棒性仍有限；② 静态权重聚合可能会平滑掉特定 codec 的独特伪造痕迹；③ 评测主要集中在少数 codec 族，尚未覆盖更广泛的实际应用场景。

---

## 478. VeriAgent: A Tool-Integrated Multi-Agent System with Evolving Memory for PPA-Aware RTL Code Generation

**arXiv ID:** 2603.17613 | [PDF](https://arxiv.org/pdf/2603.17613v1)

**作者:** Yaoxiang Wang `[一作]` (Xiamen University), Jinsong Su `[通讯]` (Xiamen University)

**通讯引用:** 4017 | [OpenAlex ID](https://openalex.org/A5066326238)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 PPA 关注的多代理框架（程序员、正确性、PPA 三个代理）结合外部 EDA 工具和可进化的结构化记忆，用 LLM 自动生成并持续优化 Verilog 代码。

**💡 创新点**

创新点在于：① 将 EDA 工具反馈嵌入闭环多代理协作；② 设计可进化的记忆机制（规则、结构、EDA 信号），实现持续经验积累；③ 通过 LLM 进行语义触发与记忆检索，提升物理性能。

**🔧 技术方法**

技术包括：大语言模型（GPT‑4o、Gemini‑3）、多代理协作框架、记忆管理器、语义触发与记忆检索、工具驱动的 PPA 分析与反馈循环。

**📊 数据集**

使用 VerilogEval（Human/Machine 两轨）和 RTLLM（v1.1 与 v2.0）作为实验基准。

**📈 对比分析**

与现有训练型和训练自由方法对比，功能正确率接近或超过 95% 以上；在 RTLLM v2.0 的 PPA 评价中，平均相对 PPA 分数提升至 5.12（GPT‑4o）/8.54（Gemini‑3），在多项指标上均优于前沿基线。

**⚠️ 局限性**

局限性包括：依赖 EDA 工具链的可用性与配置、记忆管理的规模与维护成本、以及在极端复杂设计或特定工艺下的泛化能力尚未完全验证。

---

## 479. InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning

**arXiv ID:** 2603.17310 | [PDF](https://arxiv.org/pdf/2603.17310v1)

**作者:** Chengwei Wei `[一作]` (Institute for Infocomm Research), Nancy F. Chen `[通讯]` (Centre for Frontier AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并验证了一种基于信息量密度（InfoDensity）的奖励框架，用强化学习引导大型推理模型生成既高质量又简洁的链式思维。

**💡 创新点**

核心创新在于将答案不确定性轨迹的低不确定性收敛与单调下降两大属性量化为奖励，并通过长度缩放平衡质量与效率，避免仅靠长度惩罚导致的奖励劫持。

**🔧 技术方法**

采用信息论度量（条件熵、信息增益）构建 AUC 与单调性奖励，结合长度缩放项，并在 GRPO 强化学习框架下进行训练。

**📊 数据集**

在数学推理基准 GSM8K、MATH、AIME24、OlympiadBench 上实验，并利用 ProcessBench 对单步推理质量进行评估。

**📈 对比分析**

与 GRPO‑Acc、GRPO‑LP、PEAR、DS 等基线对比，InfoDensity 在保持或提升准确率的同时显著降低 token 数，取得最佳的准确率–效率折中。

**⚠️ 局限性**

局限性包括：仅在数学推理任务上验证；奖励信号依赖外部评判模型，模型自身作为熵估计器可能更具可扩展性。

---

## 480. Physics-informed Deep Mixture-of-Koopmans Vehicle Dynamics Model with Dual-branch Encoder for Distributed Electric-drive Trucks

**arXiv ID:** 2603.17416 | [PDF](https://arxiv.org/pdf/2603.17416v1)

**作者:** Jinyu Miao `[一作]` (Tsinghua University), Diange Yang `[通讯]` (Tsinghua University)

**通讯引用:** 2280 | [OpenAlex ID](https://openalex.org/A5009072257)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于Koopman算子、双分支编码器的分布式电驱动卡车动力学模型（KODE），实现对高度非线性、耦合纵向/侧向动力学的精确建模与长期状态估计。

**💡 创新点**

创新点包括：1）双分支（Transformer+MLP）编码器提升嵌入表示；2）基于几何一致性的物理信息监督损失，强化状态间动力学约束；3）混合Koopman算子（MoK）为不同驾驶模式分配专属算子；4）仅调节Koopman算子即可实现从仿真到真实车辆的快速适配。

**🔧 技术方法**

主要技术：Koopman算子理论、Transformer自注意力网络、MLP、几何一致性损失、多步监督、混合专家框架、sim‑to‑real 迁移学习。

**📊 数据集**

数据集：从TruckSim 2019.0仿真生成约41.4万条样本（按5种驾驶模式划分），并在真实工业园六轮卡车上采集约23.4k帧用于评估与适配。

**📈 对比分析**

与传统物理模型、纯神经网络、EDMD/DeepEDMD等基线比较，在100步预测中MDE、FDE均比最优基线低约10‑30%，在真实测试中KODE‑S2R实现厘米级误差，明显优于线性模型、EDMD与DeepEDMD。

**⚠️ 局限性**

局限性：模型对训练数据依赖较大，稀有驾驶模式仍可能欠拟合；在极端高速度/极限转向等极端情形下仍存在误差；适配时需保留预训练编码器，若硬件/传感器差异过大需进一步调整。

---

## 481. Are a Thousand Words Better Than a Single Picture? Beyond Images -- A Framework for Multi-Modal Knowledge Graph Dataset Enrichment

**arXiv ID:** 2603.16974 | [PDF](https://arxiv.org/pdf/2603.16974v1)

**作者:** Pengyu Zhang `[一作]` (University of Amsterdam), Paul Groth `[通讯]` (University of Amsterdam)

**通讯引用:** 27080 | [OpenAlex ID](https://openalex.org/A5034924491)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一套自动化的多模态知识图谱数据增强管道Beyond Images，通过大规模检索实体相关图像、将图像转换为文本描述并使用LLM进行融合，从而提升MMKG的表征质量。

**💡 创新点**

创新点在于：①将含有歧义但相关的图像（如logo、符号、抽象场景）转化为可用语义的文本；②采用LLM对多源描述进行融合，生成简洁的实体对齐摘要；③提供轻量级的文本-图像一致性检查界面支持可选人工审核。

**🔧 技术方法**

技术包括：基于搜索引擎的Web图像检索、三种主流图像生成文本模型（blip2-flan-t5-xxl、git-large-coco、llava-v1.5-7b）、三种LLM（Flan‑T5‑base、LLaMA‑3.1‑8b‑instruct、Mistral‑7b‑instruct‑v0.3）以及后端数据处理脚本和轻量级网页审核界面。

**📊 数据集**

实验使用了三大公开MMKG数据集：MKG‑W、MKG‑Y和DB15K；每个数据集均在原有文本描述、图像嵌入基础上加入自动生成的文本描述和融合摘要。

**📈 对比分析**

将增强后的数据作为输入，使用四种主流MMKG模型（MMRNS、MyGO、NativE、AdaMF）进行链接预测评估；结果显示融合摘要相较于原始数据平均提升Hits@1 4–7％、MRR 3–6％，在含logo/符号图像的挑战子集上更显著提升（MRR +201.35％、Hits@1 +333.33％）。

**⚠️ 局限性**

局限性包括：对图像-文本模型的准确性和鲁棒性依赖；仅生成单句描述且使用固定提示，未充分探索多样化提示和多面向描述；人类审核仅在少量样本上验证，缺乏大规模质量评估；对下游任务（如问答、检索）尚未系统评估。

---

## 482. Amanous: Distribution-Switching for Superhuman Piano Density on Disklavier

**arXiv ID:** 2603.16890 | [PDF](https://arxiv.org/pdf/2603.16890v1)

**作者:** Joonhyung Bae `[一作]` (Korea Advanced Institute of Science and Technology), Joonhyung Bae `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 196 | [OpenAlex ID](https://openalex.org/A5075919540)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了 Amanous 系统，将 L‑system、节奏协奏与随机分布通过分布切换集成到 Yamaha Disklavier，生成超人类音符密度的可执行 MIDI。

**💡 创新点**

通过分布切换将符号映射到不同分布，并结合合约点计算实现宏观‑微观层次控制，引入硬件抽象层实现速度相关延迟补偿，并确定约 30 notes/s 的计算饱和阈值。

**🔧 技术方法**

采用四层层次架构、L‑system 宏观生成、节奏协奏时间缩放、指数/均匀/高斯分布采样、速度相关延迟模型、信息理论指标（熵、KS、Wasserstein）以及统计检验和分解技术。

**📊 数据集**

使用自生成的 MIDI 事件作为实验数据，并参考 Yamaha Disklavier 规范及公开的 6 个 Disklavier 相关数据集做延迟评估；未使用外部音乐数据库。

**📈 对比分析**

通过层级降解、消融实验、KS 距离、效应量 Cohen d、t 检验等统计方法验证分布切换保持显著差异；系统在 Disklavier 上实现了 200 notes/s 以上密度，硬件延迟补偿误差 <1 ms，整体时序精度子毫秒。

**⚠️ 局限性**

仅完成计算评估，缺乏听觉验证；系统仅针对 Yamaha Disklavier，需对其他自动钢琴进行校准；硬件补偿模型对极端延迟误差敏感，在高密度下依赖速度分离，缺乏真实听感反馈。

---

## 483. SYMDIREC: A Neuro-Symbolic Divide-Retrieve-Conquer Framework for Enhanced RTL Synthesis and Summarization

**arXiv ID:** 2603.17208 | [PDF](https://arxiv.org/pdf/2603.17208v1)

**作者:** Prashanth Vijayaraghavan `[一作]` (IBM Research), Vandana Mukherjee `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了SymDiReC框架，采用神经符号的Divide–Retrieve–Conquer方法完成Verilog和VHDL的RTL合成与摘要。

**💡 创新点**

首次将符号规划与检索相结合，在RTL任务中使用符号拆分与LLM验证，显著提升检索精度与输出一致性。

**🔧 技术方法**

利用LLM进行符号拆分与验证、联合编码检索器（双编码器）进行符号检索，并在RTL-IR数据集上微调模型。

**📊 数据集**

使用Verilog‑Eval、VHDL‑Eval两套基准以及自构造的RTL‑IR数据集（约50.5k条文本-代码/摘要对）。

**📈 对比分析**

与零样本提示、CoDes、ReAct、VRAG、RTLCoder等方法对比，SymDiReC在Pass@1提升约20%，Rouge‑L提升15–20%，远优于所有基线。

**⚠️ 局限性**

受LLM符号表达、检索噪声和子模块粒度的限制，无法处理多文件层次设计，对复杂设计的鲁棒性仍需改进。

---

## 484. Modeling Overlapped Speech with Shuffles

**arXiv ID:** 2603.17769 | [PDF](https://arxiv.org/pdf/2603.17769v1)

**作者:** Matthew Wiesner `[一作]` (Johns Hopkins University), Sanjeev Khudanpur `[通讯]` (Johns Hopkins University)

**通讯引用:** 30823 | [OpenAlex ID](https://openalex.org/A5014580424)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用 shuffle 乘积与部分序列有限状态自动机（FSA）对重叠语音进行建模，实现了单次传递式的对齐与说话人归属转录。

**💡 创新点**

首次提出基于 shuffle FSA 的单通道多说话人对齐；将 SD-CTC、SOT、tSOT 等方法统一归纳为该框架的特殊情况；通过 (token, speaker) 元组实现说话人归属；利用时间约束的部分序列来控制图大小。

**🔧 技术方法**

CTC 与 shuffle 损失；FSA 构造与前向算法；部分序列约束剪枝；WavLM encoder + 线性投影；联合与分解式说话人建模；Viterbi 对齐；1-pass 贪心解码和 N-pass 语言模型融合解码。

**📊 数据集**

合成的重叠 LibriSpeech 语料；LibriMix；标准 LibriSpeech；实验使用合成混合语音与人工 SNR、重叠比例控制。

**📈 对比分析**

与 SD-CTC、tSOT、SOT、GEncSep 等方法对比；在 LibriMix 2 说话人集上达到与最优方法相近的 cpWER/tcpWER；在 3 说话人集性能略低；对齐指标 BE 约 89 ms，IoU 56–64%，Kendall‑τ 与基准相近；1-pass 解码已足够竞争；语言模型融合显著提升。

**⚠️ 局限性**

图尺寸随说话人数/序列长度急剧膨胀，导致 GPU 内存不足；需要手动设置时间衬衫（collar）或说话人标注；对 3 说话人高度重叠场景的对齐效果不佳；实验主要基于合成数据，真实环境下可能需要进一步验证。

---

## 485. Prompt-Free Universal Region Proposal Network

**arXiv ID:** 2603.17554 | [PDF](https://arxiv.org/pdf/2603.17554v1)

**作者:** Qihong Tang `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 13191 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Prompt-Free Universal Region Proposal Network（PF‑RPN），能够在无任何文本或视觉提示的情况下直接生成潜在目标的候选框；

**💡 创新点**

创新点在于引入可学习的查询嵌入，配合Sparse Image‑Aware Adapter、Cascade Self‑Prompt以及Centerness‑Guided Query Selection，实现仅凭视觉特征的高质量目标提议；

**🔧 技术方法**

采用多尺度视觉特征交叉注意力、Mixture‑of‑Experts路由、迭代自提示更新以及中心度打分网络，整体实现从特征到查询嵌入的端到端优化；

**📊 数据集**

使用5% COCO和ImageNet预训练数据进行训练，并在19个跨域数据集（CD‑FSOD和ODinW13）上进行评估；

**📈 对比分析**

与Grounding DINO、YOLOE、GLIP、GenerateU、Open‑Det等基线对比，PF‑RPN在AR上提升5–15个百分点，同时推理速度快、显存占用低；

**⚠️ 局限性**

局限性包括对极小/被遮挡目标的召回仍不完美、需要足够多样化的视觉训练样本以及在极端域变换下的泛化能力仍有提升空间。

---

## 486. Rewarding DINO: Predicting Dense Rewards with Vision Foundation Models

**arXiv ID:** 2603.16978 | [PDF](https://arxiv.org/pdf/2603.16978v1)

**作者:** Pierre Krack `[一作]` (University of Technology Nuremberg), Florian Walter `[通讯]` (Technical University of Munich)

**通讯引用:** 21315 | [OpenAlex ID](https://openalex.org/A5059746672)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

用预训练的视觉与语言编码器，训练轻量级的密集奖励模型，使其能够从图像和语言描述中直接预测奖励值。

**💡 创新点**

创新点在于将对比学习改为基于配对排名的损失，并通过 FILM 条件融合视觉与文本特征，实现在无特权状态信息的情况下实现与原始奖励函数相当的密集反馈。

**🔧 技术方法**

技术包括冻结 DINOv3（ViT-S/16）视觉编码器、allMiniLM-L6-v2 语言编码器、带 FILM 的小型 MLP 头、pairwise logistic 损失、温度标定与等距回归、以及将奖励模型作为潜在函数用于潜在奖励塑形（pbrs）进行强化学习。

**📊 数据集**

使用 24 个 Meta‑World+ 机器人操纵任务和一个 MuJoCo 版 pick‑cube 任务的数据集，收集多视角图像与对应的原始奖励，最终得到约 169,000 个去重后的步骤样本。

**📈 对比分析**

与原始解析奖励相比，模型在配对准确率上达到 80‑93%、Kendall 相关系数 0.47‑0.94，温度标定后 ECE 仅为 0.043，强化学习实验中使用 pbrs 的 PPO 能在大多数任务中完成训练，显著优于仅使用稀疏奖励。

**⚠️ 局限性**

局限性包括奖励仅基于部分观测，故不满足真正的 Markov 性；损失不定义于等值配对；在二值奖励任务上的泛化性能下降；以及对完整场景视角的依赖，若出现遮挡可能导致误判。

---

## 487. AgriChat: A Multimodal Large Language Model for Agriculture Image Understanding

**arXiv ID:** 2603.16934 | [PDF](https://arxiv.org/pdf/2603.16934v1)

**作者:** Abderrahmene Boudiaf `[一作]` (Khalifa University of Science and Technology), Sajid Javed `[通讯]` (Khalifa University of Science and Technology)

**通讯引用:** 3556 | [OpenAlex ID](https://openalex.org/A5071515463)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 AgriMM 公共农业多模态 VQA 基准和基于 LLaVA 的 AgriChat 多模态大语言模型，并在多任务上进行训练与评估。

**💡 创新点**

创新点包括：Vision-to-Verified-Knowledge（V2VK）三阶段生成式管线，解决农业数据缺失与生物幻觉；AgriMM 的规模与多样性超过现有基准；AgriChat 通过 LoRA 在视觉编码器和语言解码器上实现领域专化。

**🔧 技术方法**

技术手段：Gemma3 生成视觉描述，Gemini3 Pro RAG 检索与验证，LLaMA3.1 合成 QA；SigLIP 视觉编码、Qwen-2.7B 语言解码；LoRA 参数高效微调；LLM-as-a-Judge 评价框架。

**📊 数据集**

使用的数据集：AgriMM（121,425 图像、607,125 VQA、3,099 类，来源 63 个公开数据集）；零样本迁移评估使用 PlantVillageVQA、CDDM、AGMMU 等公开数据集。

**📈 对比分析**

对比方法：多维度评估指标（BLEU、ROUGE、METEOR、BERTScore、LongCLIP、T5 Cos、SBERT、LLM Judge）与 LLaVA-OneVision、Llama-3.2 Vision、Qwen-2.5 VL 等开源基线对比；AgriChat 在 AgriMM、PlantVillage、CDDM、AGMMU 上的 VQA、诊断、计数任务均优于基线 10–20% 以上，LLM Judge 分数最高。

**⚠️ 局限性**

局限性：对害虫识别、多样化管理建议覆盖不足；在开放式推理任务（AGMMU open‑ended）表现逊色；对极端光照或新出现病种仍易失误；训练与推理成本相对较高。

---

## 488. ReLaGS: Relational Language Gaussian Splatting

**arXiv ID:** 2603.17605 | [PDF](https://arxiv.org/pdf/2603.17605v1)

**作者:** Yaxu Xie `[一作]` (German Research Center for Artificial Intelligence), Didier Stricker `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一个无训练、统一的多层次高斯场与开放词汇3D场景图框架ReLaGS，用于高层次语义分割、关系查询和对象检索。

**💡 创新点**

创新点包括：① 语言蒸馏的层次化高斯场；② 通过最大权重裁剪和鲁棒异常值聚合提升几何与语言一致性；③ 使用轻量预训练图神经网络进行关系预测；④ 结合SoM‑LLM标注实现2D→3D关系提升。

**🔧 技术方法**

使用技术包括：Gaussian Splatting、语言场蒸馏、SAM+CLIP特征聚合、Jina嵌入、图神经网络、THGS、最大权重裁剪、鲁棒异常值聚合、SoM提示+LLM、相机投影与光学渲染。

**📊 数据集**

主要数据集有：3DSSG RIO10子集、ScanNet++、LeRF‑OVS、ScanNet、3RScan（用于GNN预训练）。

**📈 对比分析**

与Open3DSG、ConceptGraph、RelationField等方法对比，ReLaGS在关系召回率上提升0.3（R@3）/0.5（R@5），速度快4.7×、内存占用低7.6×，在关系引导分割中mIoU提升至0.56（高于0.53的RelationField）。

**⚠️ 局限性**

局限性：仍依赖SAM/CLIP的视角一致性，部分复杂遮挡或透明物体下的语言特征易受噪声影响；关系覆盖度受限于LLM标注，部分细粒度关系难以捕获；在去掉多层次结构时性能明显下降。

---

## 489. IndicSafe: A Benchmark for Evaluating Multilingual LLM Safety in South Asia

**arXiv ID:** 2603.17915 | [PDF](https://arxiv.org/pdf/2603.17915v1)

**作者:** Priyaranjan Pattnayak `[一作]` (Oracle America Inc), Sanchari Chowdhuri `[通讯]` (Oracle America Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了IndicSafe基准，系统评估了十种主流LLM在12种印度语言上的安全表现，揭示跨语言安全漂移、拒绝偏差和歧义问题。

**💡 创新点**

创新点在于首个以文化为中心、由母语翻译并覆盖九类社会敏感话题的多语言安全基准，以及提出的跨语言一致性、类别偏差和提示熵等新评估指标。

**🔧 技术方法**

采用LLM-as-judge方法（GPT‑4o）自动判定安全标签，结合人类标注验证，计算交叉一致率、熵、语言一致性指数等多维指标。

**📊 数据集**

使用由500条英文学术与社交媒体中提炼、分类后由母语翻译成12种印度语言的共6000条文化敏感提示，涵盖种姓、宗教、政治、健康等九类。

**📈 对比分析**

与10款LLM（GPT‑4o Mini、Claude Sonnet、Grok‑3、LLaMA系列、Mistral、Qwen、Cohere等）对比，发现跨语言一致率仅12.8%，不同模型在安全与拒绝率上差异显著，低资源语言的安全率波动更大。

**⚠️ 局限性**

局限在于仅评估单轮提示生成、翻译可能带来的语义偏移、GPT‑4o判定的可靠性、缺乏对训练数据或内部机制的分析。

---

## 490. Large-Scale 3D Ground-Motion Synthesis with Physics-Inspired Latent Operator Flow Matching

**arXiv ID:** 2603.17403 | [PDF](https://arxiv.org/pdf/2603.17403v1)

**作者:** Yaozhong Shi `[一作]` (California Institute of Technology), Domniki Asimaki `[通讯]` (California Institute of Technology)

**通讯引用:** 1099 | [OpenAlex ID](https://openalex.org/A5084650795)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了Ground‑Motion Flow（GMFlow），一种基于物理启发的潜在流匹配框架，能够快速生成符合事件参数（震源位置、规模）的、具有空间时间一致性的地区地震动时间历程。

**💡 创新点**

创新点在于将生成任务拆分为低频物理对齐子空间的潜在空间流匹配与高频超分辨率神经算子两步流程，既保证了大尺度波传播的物理一致性，又利用流匹配实现了高效、可逆的随机采样；同时引入了物理对齐子空间来降低训练成本并支持零样本超分辨率。

**🔧 技术方法**

使用的主要技术包括：基于FFT的Fourier Neural Operator（FNO）实现的超分辨率与自动编码器；潜在空间的条件流匹配（rectified flow matching）配合清洁预测；自编码器与超分辨率的联合训练；以及在潜在空间内的随机采样与ODE求解。

**📊 数据集**

使用了在加州旧金山湾区（SFBA）高分辨率3D地震波传播模拟数据，共5300个事件（2100个M4.4点源、2100个M6.0、1000个M7.0）以及300个留存测试集，模拟数据来自SW4平台，涵盖多种震源机制与地质结构。

**📈 对比分析**

与传统基于物理的数值模拟相比，GMFlow在单个工作站GPU上对5300个事件的全域生成仅需0.386 GPU小时，速度提升超过10⁴倍；与现有单站时间历程生成模型（如cGM-GANO、扩散模型）相比，输出维度大约500–800倍；在PGV、频谱、空间相关性、频率残差与规模-幅值缩放等工程指标上均能保持与真实模拟相当的精度。

**⚠️ 局限性**

主要局限包括：对大规模有限断层事件的高频成分（>0.5 Hz）低估，需要后期频率校正；模型基于FFT‑FNO，仅适用于规则网格，难以处理不规则边界、地形或稀疏观测网络；以及高频波传播与散射物理尚未充分捕获，导致在最高频段的偏差。

---

## 491. PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery

**arXiv ID:** 2603.17571 | [PDF](https://arxiv.org/pdf/2603.17571v1)

**作者:** Yijing Guo `[一作]` (ShanghaiTech University), Yujiao Shi `[通讯]` (ShanghaiTech University)

**通讯引用:** 945 | [OpenAlex ID](https://openalex.org/A5002882477)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 PanoVGGT，一种可在单次前向传递中对无序全景图像进行相机姿态、密集深度和全局一致点云联合预测的 Transformer 框架，并发布了大规模的 PanoCity 数据集。

**💡 创新点**

核心创新点包括：①使用球面感知位置编码（sin‑cos 角度向量）实现全景图的无缝表示；②三轴 SO(3) 旋转数据增强实现真正的球面旋转不变性；③随机锚定策略解决全局坐标系不确定性；④将点云回归作为额外监督提升几何一致性。

**🔧 技术方法**

技术实现基于 ViT 编码器、交替视图内外注意力的几何聚合器、分支预测头（姿态、局部/全局点云），并结合多任务损失（尺度一致、姿态、深度、法线一致性）。

**📊 数据集**

使用新构建的 PanoCity（约 12 万幅 4096×2048 的全景图、16 位深度与 6DoF 姿态），并在 Matterport3D、Stanford2D3D、Structured3D、Pano3D 等公开数据集上进行跨域评测。

**📈 对比分析**

与 Bifusev2、VGGT、π³ 等基线相比，PanoVGGT 在相机姿态 AUC@30、旋转/平移误差、深度绝对误差以及点云均方误差等指标上均取得显著或同等性能，尤其在户外全景数据集上表现出更强的鲁棒性和跨域泛化。

**⚠️ 局限性**

主要局限包括：对极端域迁移仍易受干扰，模型专为等距投影设计，且在视角缺失或极低重叠场景下姿态估计仍面临挑战。

---

## 492. FrescoDiffusion: 4K Image-to-Video with Prior-Regularized Tiled Diffusion

**arXiv ID:** 2603.17555 | [PDF](https://arxiv.org/pdf/2603.17555v1)

**作者:** Hugo Caselles-Dupré `[一作]` (Obvious Research), Matthieu Cord `[通讯]` (Institute of Intelligent Systems and Robotics - Sorbonne University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种无训练的 4K 图像到视频方法 FrescoDiffusion，通过先生成低分辨率动画作为先验，再在高分辨率上做块状去噪融合。

**💡 创新点**

创新点在于在块状去噪中加入先验正则化以及区域感知的先验强度调度，实现了全局一致性与局部细节兼顾，并可调控创意与先验相似度。

**🔧 技术方法**

技术包括基于流匹配的扩散模型、块状去噪（MultiDiffusion）、先验正则化闭式求解、空间-时间先验调度与 SAM3 生成活动地图。

**📊 数据集**

使用 Wan2.2-I2V 14B 视频扩散模型，评估数据集包括 VBench‑4K 与自制 FrescoArchive（多场景 4K 图像）。

**📈 对比分析**

与 MultiDiffusion、DemoFusion、DynamicScaler 等基线对比，FrescoDiffusion 在 VBench 指标、Sharpness、Temporal Consistency 以及用户偏好调查中均表现更好，速度更快。

**⚠️ 局限性**

局限性是依赖低分辨率先验，若先验不足会失去全局结构；块状去噪仍然算力消耗高，需要进一步优化效率。

---

## 493. AURORA Model of Formant-to-Tongue Inversion for Didactic and Clinical Applications

**arXiv ID:** 2603.17543 | [PDF](https://arxiv.org/pdf/2603.17543v1)

**作者:** Patrycja Strycharczuk `[一作]`, Sam Kirkham `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出 AURORA 模型，通过线性回归将 F1、F2 及其交互映射到舌部六个参数，进而预测舌形。

**💡 创新点**

将舌形预测简化为低维可视化模型，并集成到实时声学/舌形反馈工具，便于临床与教学应用。

**🔧 技术方法**

使用同步超声舌影像与 FastTrack 频率估计、Procrustes 归一化、PCA 降维、线性回归、Shiny 与 PyQtGraph 实时可视化。

**📊 数据集**

基于 40 名英国北部母语者的 3,856 词语同步超声与音频数据，包含 11 个舌尖关键点。

**📈 对比分析**

通过可视化与平均舌形对比，发现预测与真实舌形高度匹配，唯一明显偏差为“booed”因唇圆化未建模；整体性能满足教学和反馈需求。

**⚠️ 局限性**

仅建模舌部，未考虑唇、软腭及下颌运动；使用未归一化 F1/F2 可能偏向女性声调，且数据局限于英式口音。

---

## 494. WINFlowNets: Warm-up Integrated Networks Training of Generative Flow Networks for Robotics and Machine Fault Adaptation

**arXiv ID:** 2603.17301 | [PDF](https://arxiv.org/pdf/2603.17301v1)

**作者:** Zahin Sufiyan `[一作]` (University of Alberta), Osmar Zaiane `[通讯]` (Alberta Machine Intelligence Institute)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `40105733-5154-44cd-8090-a8cab9e64b07` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在连续机器人控制任务中提出 WINFlowNets，通过同时训练流网络与检索网络实现自适应策略学习。

**💡 创新点**

创新点在于不再需要预训练检索网络，采用 Warm‑Up 阶段和 Dual‑Training 阶段，并使用共享 replay buffer 让两网络协同学习。

**🔧 技术方法**

使用连续流网络（CFlowNets）的流匹配损失、检索网络回归、共享经验回放、两阶段训练策略等技术。

**📊 数据集**

在 OpenAI Gymnasium 的 Reacher‑V2 仿真环境下进行实验，包含正常环境以及两种故障场景（Actuator Damage 与 Reduced ROM）。

**📈 对比分析**

与原始 CFlowNets、SAC、PPO、DDPG 等基线对比，WINFlowNets 在平均奖励和训练稳定性上优于 CFlowNets，并在 OOD 场景下表现最佳；样本效率略低于 SAC，但最终性能最高。

**⚠️ 局限性**

局限性包括高计算与内存开销、对 Warm‑Up 时长、学习率、缓冲区大小等超参数高度敏感、训练初期表现波动较大、仅在仿真环境验证、难以直接迁移到资源受限或真实机器人。

---

## 495. Shielded Reinforcement Learning Under Dynamic Temporal Logic Constraints

**arXiv ID:** 2603.17152 | [PDF](https://arxiv.org/pdf/2603.17152v1)

**作者:** Sadık Bera Yüksel `[一作]` (Northeastern University), Derya Aksaray `[通讯]` (Northeastern University)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5053550436)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在强化学习中引入序列控制障碍函数（Sequential CBF），以在学习过程中严格执行 Signal Temporal Logic（STL）约束，实现动态目标和时间窗口任务的实时满足。

**💡 创新点**

创新点在于将序列 CBF 与模型无关 RL 结合，能够处理未知轨迹的动态目标和复杂的时空约束，并给出任务违反的下界保证；同时提出基于最紧迫 CBF 的实时 QP 校正方法。

**🔧 技术方法**

使用了控制障碍函数、Signal Temporal Logic、模型无关强化学习算法（如 SAC/PPO/TRPO）以及实时二次规划（QP）来生成约束校正控制。

**📊 数据集**

通过仿真环境验证，未使用公开真实数据集；实验涉及两种动态目标设置（移动充电站、随机移动目标）。

**📈 对比分析**

与无约束 RL（SAC）比较：受约束模型在满足 STL 的率分别为 99.56% 和 95.2%，但因需满足时空约束导致平均回报下降；图表显示收敛速度略慢，但实现了正式的约束满足。

**⚠️ 局限性**

局限性包括：仅适用于单积分器或易于保守估计的动力学；对复杂动力学的可达性分析未展开；实时 QP 计算增加计算负担；对目标轨迹的最坏情况假设可能过于保守。

---

## 496. BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images

**arXiv ID:** 2603.17159 | [PDF](https://arxiv.org/pdf/2603.17159v1)

**作者:** David Skuddis `[一作]` (Institute for Photogrammetry and Geoinformatics), Norbert Haala `[通讯]` (Institute for Photogrammetry and Geoinformatics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 BEV-SLD，一种基于 LiDAR BEV 图像的全局定位方法，利用自监督学习自动发现并检测场景特定地标，实现无密集地图、仅 20 MB 网络+地标列表即可定位。

**💡 创新点**

创新点在于自监督一致性损失使得网络同时学习地标位置与检测；采用高分辨率热图+低分辨率对应图的两阶段输出，实现可扩展的高密度地标学习；仅需小网络即可在多种环境下实时定位。

**🔧 技术方法**

使用 BEV 密度图作为输入；改进的 Feature Pyramid Network 作为 backbone；软最大化热图、距离损失 + 对应损失进行联合训练；RANSAC 最终估计 3DoF 位姿；数据增强与网格过滤等预处理。

**📊 数据集**

在 MCD、NCLT、Wild‑Places 公开数据集以及自采的工业工厂数据集上进行实验，覆盖校园、森林、工业场景。

**📈 对比分析**

与 BEVPlace++、LightLoc、KISS‑Matcher、PosePN++ 等现有方法对比，BEV‑SLD 在 16 条序列中 11 条成功率最高；在低重叠、远离参考轨迹的情况下仍保持高成功率，误差与其他方法相当甚至略优。

**⚠️ 局限性**

局限性：方法是场景专属，需在每个环境单独训练，无法直接迁移到未见环境；仅估计 3DoF 位姿；对极低点云密度或大规模场景的可扩展性仍需进一步研究。

---

## 497. Joint Degradation-Aware Arbitrary-Scale Super-Resolution for Variable-Rate Extreme Image Compression

**arXiv ID:** 2603.17408 | [PDF](https://arxiv.org/pdf/2603.17408v1)

**作者:** Xinning Chai `[一作]` (Shanghai Jiao Tong University), Li Song `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 84014 | [OpenAlex ID](https://openalex.org/A5100448217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于任意尺度超分辨率（ASSR）的极低比特率图像压缩框架ASSR-EIC，支持单模型实现可变比特率压缩；

**💡 创新点**

创新点包括：①在编码端引入任意尺度下采样实现可控比特率；②在解码端设计联合降解感知ASSR解码器，利用扩散模型的先验生成高质量图像；③引入全局与局部压缩‑缩放适配器，动态调节生成与保真度；④双语义增强机制，融合文本提示与SAM语义注意力提升结构与感知质量；

**🔧 技术方法**

主要技术：扩散模型（Stable Diffusion）与ControlNet式保真模块、文本编码（BLIP‑2）、语义编码（SAM‑Tiny）、全局/局部条件嵌入、双语义增强、损失函数包括扩散损失与域对齐损失；

**📊 数据集**

训练使用OpenImagesV6 100k图像，评估使用Kodak、CLIC2020、MS‑COCO三个公开数据集；

**📈 对比分析**

与传统VVC/HEVC、学习型ELIC/​MS‑ILLM以及DiffEIC、PerCo等方法比较，取得93%/31% BD‑rate下降、显著提升FID、CLIPScore与DISTS，显示在极低比特率下仍保持高感知与保真质量；

**⚠️ 局限性**

主要限制是解码时间较长，需多步扩散推理，未来可通过单步采样或模型蒸馏加速。

---

## 498. Biclique Reconfiguration in Bipartite Graphs

**arXiv ID:** 2603.17706 | [PDF](https://arxiv.org/pdf/2603.17706v1)

**作者:** Yota Otachi `[一作]` (Nagoya University), Emi Toyoda `[通讯]` (Nagoya University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了在二分图上使用令牌跳跃模型的二分子图重构问题（Biclique Reconfiguration），证明其在二分图上的多项式空间复杂度（PSPACE）完整性，并利用该结果证明了在具有两个连通分量的图上进行连通组件重构（CJ、CS规则）的问题同样是PSPACE完整的，从而否定了前人提出的相关开放问题。

**💡 创新点**

核心创新在于克服了二分图结构下重构问题的难点：通过在原始图的非二分图构造“二分补图”并巧妙地“解锁”二分子图（将其大小减一）来实现从独立集/团的重构到二分子图的多步模拟，从而在保持二分图属性的同时实现完整性证明。

**🔧 技术方法**

主要技术包括：
- 在二分图的“边-点”偶图上构造二分补图（即二分补图 H），
- 利用 Johnson 的二分子图与图团的对应关系，并对其进行大小调节；
- 将令牌跳跃（TJ）序列与连通组件跳跃（CJ/CS）序列之间建立等价关系；
- 通过归约和多步模拟证明 PSPACE‑hardness。

**📊 数据集**

本文未使用任何实验数据集，全部工作为理论证明。

**📈 对比分析**

由于研究对象为理论复杂性问题，未进行实验比较；证明过程基于归约与结构性质分析，展示了问题在二分图上与一般图等价的复杂性上界，因而证明其与已知 PSPACE‑完整问题同等难度。

**⚠️ 局限性**

限制包括：
- 仅对令牌跳跃模型（TJ）给出了完整性证明，令牌滑动模型（TS）在二分图上无意义；
- 对灵活变体（不同形状的二分子图）在二分图上仅能得到 NP‑完备性，无法进一步提升至 PSPACE；
- 证明过程对图结构有较强的特殊性，若要求更广泛的图类可能需要新的技术。

---

## 499. GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion

**arXiv ID:** 2603.17161 | [PDF](https://arxiv.org/pdf/2603.17161v1)

**作者:** Zhuojiang Cai `[一作]` (Beihang University), Feng Lu `[通讯]` (Beihang University)

**通讯引用:** 7346 | [OpenAlex ID](https://openalex.org/A5101480749)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种端到端的双分辨率网络GazeOnce360，用于单摄像头上向下鱼眼相机捕捉的360°多人的视线估计。

**💡 创新点**

创新点包括：1）使用旋转卷积提高对鱼眼畸变的鲁棒性；2）通过眼部关键点监督增强对细粒度眼部特征的学习；3）设计双分辨率架构，将全局低分辨率上下文与局部高分辨率眼部特征融合；4) 新建大规模合成数据集MPSGaze360，提供精确3D视线、头位姿和眼部标记。

**🔧 技术方法**

主要技术包括旋转卷积、交叉注意力融合、多任务头（置信度、边框、头位姿、视线、面部与眼部关键点），以及基于ResNet-50的双分支网络。

**📊 数据集**

使用了自研合成数据集MPSGaze360（约2.35万幅图像，1–7人/图，包含3D视线、头位姿、眼部关键点等标注），并在该数据上进行训练，后续在真实鱼眼图像上进行泛化验证。

**📈 对比分析**

与之前的多阶段方法GAM360相比，GazeOnce360在交叉场景+身份下的视线误差从18.96°降低到10.39°，调整后误差从18.76°降至9.99°，同时帧率从4.23 FPS提升到16.23 FPS；相较于单分辨率基线，双分辨率模型几乎保持同等精度且提升约22%推理速度。

**⚠️ 局限性**

局限性：对极端头部姿态或远距离目标的精度下降；模型主要基于合成数据训练，尽管能迁移到真实图像，但在遮挡、极端光照等极端条件下仍需进一步提升。

---

## 500. Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text Detectors Across Architectures, Domains, and Adversarial Conditions

**arXiv ID:** 2603.17522 | [PDF](https://arxiv.org/pdf/2603.17522v1)

**作者:** Madhav S. Baidya `[一作]` (Indian Institute of Technology), Chirag Chawla `[通讯]` (Indian Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个大型、跨域、跨模型的 AI 生成文本检测基准，系统评估了从传统手工特征到大规模 Transformer、浅层 CNN、提示式检测等多种检测框架，并在两套严格长度匹配的问答语料上进行多阶段评测（原始、跨域、对抗性人化）。

**💡 创新点**

创新点包括：
1) 统一的长度匹配预处理，消除答案长度的偏差；
2) 两套对照语料（HC3 和 ELI5，含 46,726/30,000 条双语对）与 Mistral‑7B 生成的补充语料；
3) 多阶段评测：原始检测、跨源零样本评估、对抗性人化迭代；
4) 引入可解释的 XGBoost 风格混合特征，并验证其与大规模 fine‑tuned 模型相当；
5) 对提示式检测（-as‑detector）和对比似然检测进行系统比较，揭示生成器‑检测器同源问题和 perplexity 倾斜逆转。

**🔧 技术方法**

主要技术手段包括：
- fine‑tuned encoder 变换器（BERT, RoBERTa, ELECTRA, DistilBERT, DeBERTa‑v3）;
- 传统统计分类器（Logistic Regression, Random Forest, SVM）；
- 轻量 1D‑CNN；
- XGBoost 风格的可解释特征集（句子级 perplexity CV、AI 词组密度等）；
- 无监督 perplexity‑based 检测（GPT‑2/GPT‑Neo 参考模型）；
- -as‑detector 提示式分类（含链式推理、结构化 rubric 提示、对齐前置概率校正）；
- 对抗性人化实验（使用 Qwen2.5‑1.5B‑Instruct 迭代改写）；
- 交叉 LLM 一致性与分布位移分析（KL、Wasserstein、Fréchet），以及基于句子嵌入的零样本分类。

**📊 数据集**

使用的数据集为：
- HC3（Human‑ChatGPT 比较）共 47,734 条问答对（去重后 23,363 对，拆分后 46,726 条文本）；
- ELI5（单领域问答）共 325,475 条人类答案，随机抽取 15,000 条并使用 Mistral‑7B‑Instruct 生成对应答案，形成 30,000 条文本；
- 5 套未见 LLM（TinyLlama‑1.1B、Qwen2.5‑1.5B、Qwen2.5‑7B、Llama‑3.1‑8B、LLaMA‑2‑13B）生成的 2,000 条 AI 文本用于跨源评估；
- 2,000 条人类文本用于对抗性人化的基准。

**📈 对比分析**

比较方法采用统一的五指标评测套件：准确率、AUC、Brier Score、对数损失、FPR@95%TPR。结果显示：
- fine‑tuned Transformer 在原始分布上达 0.99+ 的准确率；
- XGBoost 混合特征在原始分布上 0.9996，跨域 0.904；
- 1D‑CNN 与 Transformer 相当但参数量仅 1/20；
- 无监督 perplexity 检测 0.91；
- -as‑detector 最高 0.9093（GPT‑4o‑mini）但远低于 fine‑tuned；
- 跨域（HC3 ↔ ELI5）均下降 5–30%，且对抗性人化 L2 仍保持 0.85+ 的准确率，说明目前不存在完全鲁棒检测器。

**⚠️ 局限性**

局限性包括：
- 仅评估了两类生成模型（ChatGPT/GPT‑3.5 与 Mistral‑7B），未覆盖前沿模型如 Claude、Gemini、GPT‑4；
- 语料仅为英文问答，缺乏多语言、多体裁评估；
- 对抗性人化仅使用 Qwen2.5‑1.5B‑Instruct；
- -as‑detector 评估样本量有限（≈30 条），影响统计显著性；
- 交叉 LLM 评估使用 2,000 条样本，仍可能低估跨源泛化难度；
- 评测未涵盖生成文本的多模态或长篇文本。

---

## 501. IEMAS: An Incentive-Efficiency Routing Framework for Open Agentic Web Ecosystems

**arXiv ID:** 2603.17302 | [PDF](https://arxiv.org/pdf/2603.17302v1)

**作者:** Hongze Liu `[一作]` (Shanghai Jiao Tong University), Jie LI `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 IEMAS，一个在开放 Agentic Web 生态中将经济激励与系统效率协同设计的多代理 LLM 路由框架，实现客户端请求与 LLM 代理的分布式匹配，利用 KV 缓存和 VCG 拍卖来优化资源利用。

**💡 创新点**

① 将 KV 缓存视为可计价资源并将缓存亲和度纳入 QoS 预测；② 结合概率预测模型与 VCG 双重拍卖的 Min‑Cost Max‑Flow 匹配，确保真诚上报和社会福利最大化；③ 引入代理中心化的 Proxy Hub 与域聚类，提升大规模网络的可扩展性与隐私。

**🔧 技术方法**

使用 Hoeffding Tree 回归/分类进行在线 QoS 预测；利用 Long Common Prefix 计算 KV 亲和度；VCG 机制与 Min‑Cost Max‑Flow 求解分配；代理中心化与域聚类；基于 vLLM 的 LLM 服务实现。

**📊 数据集**

CoQA、QuAC、HotpotQA 三个多轮/长文本/推理任务集，用于评估 KV 命中率、延迟与成本。

**📈 对比分析**

与 GraphRouter、GMTRouter、MFRouter、RouterDC 及 Random 基线对比；IEMAS 在 KV 命中率提升至约 80%，平均成本下降约 35%，TTFT 减少 2.9 倍，整体社会福利最高。

**⚠️ 局限性**

仍无法完全实现预算平衡（VCG 机制可能导致支付与成本不匹配）；对极端异质代理的 IR 可能受限；大规模部署需更多聚类细粒度调优；预测模型需持续学习；缺乏真实分布式环境实验，仅在模拟环境验证。

---

## 502. Intent Formalization: A Grand Challenge for Reliable Coding in the Age of AI Agents

**arXiv ID:** 2603.17150 | [PDF](https://arxiv.org/pdf/2603.17150v1)

**作者:** Shuvendu K. Lahiri `[一作]` (Microsoft Research), Shuvendu K. Lahiri `[通讯]` (Microsoft Research)

**通讯引用:** 5048 | [OpenAlex ID](https://openalex.org/A5041084431)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并系统化了意图形式化（intent formalization）的概念，旨在自动将自然语言意图转化为可检查的正式规范，从而弥合AI生成代码的意图鸿沟。

**💡 创新点**

创新点在于：①构建了从部分到完整规范的连续谱（测试、代码契约、逻辑契约、DSL），②提出了基于测试的规范音量性（soundness）与完整性（completeness）评估方法，③展示了交互式工具TiCoder和端到端DSL系统3DGen的实证效果。

**🔧 技术方法**

技术主要包括：大型语言模型（如GPT‑4、LLM）用于生成规范与代码，符号/约束求解器（SMT、Verus、Dafny）进行静态验证，自动化指标计算（基于测试、变异分析），以及多代理AI架构实现DSL解析与合成。

**📊 数据集**

使用的主要数据集与基准包括：Defects4J（Java缺陷库）、C++数据结构模块（Verus验证）、HumanEval/ICSE benchmark等，用于评估规范生成、代码正确率和交互式优化效果。

**📈 对比分析**

比较方法：对比传统LLM无规范生成的代码准确率（约40%）与TiCoder交互式后（约84%），以及GPT‑4生成的postconditions与Daikon检测的覆盖率。性能表现显示，规范化显著提升了代码正确率、降低了认知负荷，并在DSL端实现了可证明正确的网络协议解析器。

**⚠️ 局限性**

局限性包括：①规范生成仍缺乏完备的评估oracle，依赖自动化指标；②对大规模、状态化、异步或并发系统的规范支持不足；③交互式成本在复杂代码块上可能过高；④LLM在处理量化逻辑与递归谓词时仍有误差；⑤需要进一步完善人机交互界面与工作流集成。

---

## 503. CodeGreen: Towards Improving Precision and Portability in Software Energy Measurement

**arXiv ID:** 2603.17924 | [PDF](https://arxiv.org/pdf/2603.17924v1)

**作者:** Saurabhsingh Rajput `[一作]` (Dalhousie University), Tushar Sharma `[通讯]` (Dalhousie University)

**通讯引用:** 2173 | [OpenAlex ID](https://openalex.org/A5023044082)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个可扩展的异步能源测量平台 CodeGreen，用于多语言、多硬件的细粒度能耗分析。

**💡 创新点**

创新点在于将 instrumentation 与 measurement 解耦的生产者-消费者架构，利用 Tree-sitter 自动注入标记，异步采样硬件能耗，实现低开销同时高精度。

**🔧 技术方法**

采用 Tree-sitter AST 查询、异步生产者-消费者、锁自由循环缓冲区、RAPL/NVML/ROCm 低级驱动、线性插值归因、可配置粒度等技术。

**📊 数据集**

使用 Computer Language Benchmarks Game（C、C++、Python、Java 计算密集型基准）作为实验数据集。

**📈 对比分析**

通过与 RAPL 直接读取对比，R²=0.9934，误差约10.9%，与线性能耗关系 R²=0.9997；与现有工具（pyRAPL、CodeCarbon 等）相比，开销可调且可预测，精度接近底层计数器。

**⚠️ 局限性**

局限性包括依赖 RAPL 作为基准导致 1ms 分辨率限制、在极高检查点密度下会显著提升开销、实验主要在 x86 平台，需进一步验证在 ARM 等平台的表现。

---

## 504. Complementary Reinforcement Learning

**arXiv ID:** 2603.17621 | [PDF](https://arxiv.org/pdf/2603.17621v1)

**作者:** Dilxat Muhtar `[一作]` (Alibaba Group), Bo Zheng `[通讯]` (Alibaba Group)

**通讯引用:** 12735 | [OpenAlex ID](https://openalex.org/A5034845046)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种能够让策略主体与经验提取器在强化学习过程中相互演化的框架（Complementary Reinforcement Learning）

**💡 创新点**

核心创新在于把经验提取器与策略主体共同训练，形成闭环共进，解决传统经验静态或与主体脱节导致的偏差问题；并设计了异步体验管理器、分组优势估计和搜索工具来实现高效、可扩展的共进训练

**🔧 技术方法**

使用大语言模型作为策略主体和经验提取器，CISPO 与 GRPO‑split 的强化学习目标，经验提取器通过 token‑级重要性采样与奖励信号进行训练；体验管理器实现并行检索与合并，嵌入模型用于语义检索

**📊 数据集**

在四个开源环境上进行评估：MiniHack、WebShop、ALFWorld 与 SWE‑Bench；并在多任务设置下混合训练多种任务

**📈 对比分析**

与不使用经验、静态经验和仅训练经验提取器等基线相比，提出的共进方法在单任务上提升约10% 的成功率，在多任务上提升 7%–12%；实验显示即使不在测试时检索经验，模型依然保持显著优势，证明经验已被主体内化

**⚠️ 局限性**

缺点包括：需要额外的抽取器模型和管理器，导致实现复杂；在某些任务中自蒸馏方案会导致训练崩溃；经验提取器容量不足时提升有限，说明经验质量与主体能力仍需平衡

---

## 505. ShuttleEnv: An Interactive Data-Driven RL Environment for Badminton Strategy Modeling

**arXiv ID:** 2603.17324 | [PDF](https://arxiv.org/pdf/2603.17324v1)

**作者:** Ang Li `[一作]` (Peking University), Wenxin Li `[通讯]` (Peking University)

**通讯引用:** 3320 | [OpenAlex ID](https://openalex.org/A5100397213)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了ShuttleEnv，一种基于精英比赛数据的交互式、数据驱动的羽毛球强化学习环境，并展示多种训练好的智能体及可视化工具。

**💡 创新点**

采用两阶段概率模型从比赛数据学习击球成功与对手回球动态，省去物理仿真并实现可解释的回合级决策；同时集成实时3D可视化与交互式演示，提供直观的策略评估。

**🔧 技术方法**

使用行为克隆、A2C/PPO/SAC等策略梯度算法，配合M_succ与M_ret两个概率预测模型，以及三维动画渲染技术。

**📊 数据集**

手工收集的Lin Dan对Lee Chong Wei精英比赛裁剪集，包含数千个标注的击球动作与战术属性。

**📈 对比分析**

在1000场模拟比赛中，BC（模仿学习）赢率约33.2%，A2C约65.8%，PPO约98.3%，SAC约90.5%，显示强化学习方法显著优于模仿学习。

**⚠️ 局限性**

缺乏完整物理与生物力学细节；数据仅来自两位选手，风格多样性有限；未实现多智能体协作或更丰富的运动学建模。

---

## 506. Per-Domain Generalizing Policies: On Learning Efficient and Robust Q-Value Functions (Extended Version with Technical Appendix)

**arXiv ID:** 2603.17544 | [PDF](https://arxiv.org/pdf/2603.17544v1)

**作者:** Nicola J. Müller `[一作]` (German Research Center for Artificial Intelligence), Timo P. Gros `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种通过对 Q‑value 进行正则化的监督学习方法，用图神经网络学习可泛化到大实例的规划策略。

**💡 创新点**

创新点在于：①将传统的状态值函数替换为 Q‑value 函数，使策略评估只需处理当前状态；②设计两种正则化项（显式与启发式），强制非教师动作的 Q‑value 高于教师动作，从而显著提升泛化性能。

**🔧 技术方法**

技术包括：图神经网络（Relational GCN、Object Encoding、Object‑Atom Encoding）、均方误差与 MAE 损失、正则化项 λ·R(s,a) 以及使用可接受启发式 h(s') 计算下界。

**📊 数据集**

数据集：IPC'23 学习轨道中的 10 个经典规划域（Blocksworld、Childsnack、Ferry、Floortile、Gripper、Logistics、Rovers、Satellite、Transport、Visitall），使用 Fast Downward 生成最优轨迹作为教师示例。

**📈 对比分析**

与基准比较：对比状态值策略和 LAMA‑first，结果显示正则化 Q‑value 策略在规模扩展性（Scale、SCov）和 IPC 测试集覆盖率上均优于状态值策略，并与 LAMA‑first 的性能持平（覆盖率高、规划长度相近）。

**⚠️ 局限性**

局限性：①正则化不保证 Q‑value 的可接受性；②实验仅覆盖 10 个域，未检验更大或更复杂域的泛化；③需进一步结合强化学习或动态实例生成以提升更高阶的泛化能力。

---

## 507. From Language to Action in Arabic: Reliable Structured Tool Calling via Data-Centric Fine-Tuning

**arXiv ID:** 2603.16901 | [PDF](https://arxiv.org/pdf/2603.16901v1)

**作者:** Omer Nacar `[一作]` (Tuwaiq Academy), Mohammed Alkhalifa `[通讯]` (Tuwaiq Academy)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并训练了面向阿拉伯语的函数调用框架 AISA-AR-FunctionCall，完成数据集审计、架构修复、提示重构及全参数微调。

**💡 创新点**

首次系统化地针对阿拉伯语多方言进行函数调用数据清洗与结构化，并通过全参数微调显著提升轻量级 270M 参数模型的结构稳定性与命名一致性。

**🔧 技术方法**

采用 FunctionGemma 3 270M 作为基础模型，结合全参数监督微调、控制符提示模板、枚举校正、工具采样、完成式掩码以及可选的 LoRA 逻辑推理扩展。

**📊 数据集**

使用公开的 Arabic Function Calling 数据集（50,810 条样本，36 个工具，5 个方言，8 个领域），并在其基础上生成了 AISA-AR-FunctionCall 训练语料。

**📈 对比分析**

在保留测试集上与基线 FunctionGemma 进行对比，parse 失败率从 87% 降至 <1%，函数名准确率提升 8 倍，方言间差距显著缩小，整体性能从几乎无效提升至 70%+ 准确率。

**⚠️ 局限性**

主要局限在于结构已恢复但语义决策仍出现误匹配、工具歧义和参数漂移，需进一步改进决策层监督、对比学习或自我校准机制。

---

## 508. ShapleyLaw: A Game-Theoretic Approach to Multilingual Scaling Laws

**arXiv ID:** 2603.17945 | [PDF](https://arxiv.org/pdf/2603.17945v1)

**作者:** Xuyang Cao `[一作]` (Osaka University), Shuyuan Zheng `[通讯]` (Osaka University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了基于Shapley值的多语言扩展法则ShapleyLaw，用以量化跨语言迁移并预测/优化多语言预训练模型的测试损失。

**💡 创新点**

首次将合作博弈论中的Shapley值用于测量跨语言迁移，并将其嵌入多语言尺度法则中，从单一语言族级别细化到个别语言的混合比例优化。

**🔧 技术方法**

使用合作博弈论建模、Shapley值近似、Chinchilla尺度法则、Monte Carlo近似估算SV等技术。

**📊 数据集**

采用七种语言（来自五个语族）的Controlled/Empirical混合数据，涵盖Common Crawl、FineWeb‑2、LLaMA‑2、LLM‑JP‑V4等数据集，模型规模从50M到1.46B，数据量从50B到200B。

**📈 对比分析**

与FamilyLaw、Uniform、Smoothed、Empirical等混合策略比较，利用R²、CE损失、下游任务准确率评估；ShapleyLaw在测试损失预测上R²>0.9，在混合优化上CE损失明显低于Baseline，且下游任务准确率与损失呈显著负相关。

**⚠️ 局限性**

计算Shapley值需要预训练O(2^K)模型，尽管可通过小规模预训练或Monte Carlo近似降低成本，但仍比FamilyLaw更耗资源；且目前仅在七语言设置验证，扩展到更大语言集合仍需进一步评估。

---

## 509. SARE: Sample-wise Adaptive Reasoning for Training-free Fine-grained Visual Recognition

**arXiv ID:** 2603.17729 | [PDF](https://arxiv.org/pdf/2603.17729v1)

**作者:** Jingxiao Yang `[一作]` (Zhejiang University), Xuhong Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 7405 | [OpenAlex ID](https://openalex.org/A5100321925)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SARE框架，实现训练无关的细粒度视觉识别，通过自适应检索与推理两阶段，结合自我反思经验库提升准确率与效率。

**💡 创新点**

创新点在于样本级自适应触发机制动态决定是否进入深度推理，以及将过去错误转化为可迁移的判别规则的经验库。

**🔧 技术方法**

采用CLIP多模原型检索、LVLM VQA式推理、统计触发器、RRF融合、经验库构建与检索等技术。

**📊 数据集**

在14个细粒度与通用视觉数据集（如CUB、Stanford Dogs、Stanford Cars、Flowers、Birdsnap、ImageNet等）上进行评测。

**📈 对比分析**

与多种检索式、推理式及训练式基准对比，SARE平均准确率达到87.68%，在细粒度任务上超过最优基准约8%+，在训练式方法上提升约1.6%，且计算成本显著下降。

**⚠️ 局限性**

局限性包括经验库质量提升需额外自反思成本，触发策略对极难数据集仍较保守，且主要依赖文本上下文，未充分利用视觉特征。

---

## 510. ARES: Scalable and Practical Gradient Inversion Attack in Federated Learning through Activation Recovery

**arXiv ID:** 2603.17623 | [PDF](https://arxiv.org/pdf/2603.17623v1)

**作者:** Zirui Gong `[一作]` (Griffith University), Cong Wang `[通讯]` (City University of Hong Kong)

**通讯引用:** 25684 | [OpenAlex ID](https://openalex.org/A5100390514)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了 ARES，一种在联邦学习中不需要改动模型结构的激活恢复稀疏逆向攻击，可在大批量训练下高精度恢复训练样本。

**💡 创新点**

创新点在于将激活反演转化为嘈杂稀疏恢复问题，并结合线性层泄露、印记策略、Direct‑Pass 初始化和压缩感知的 RIP 性质，实现大批量、多层激活分离与重构。

**🔧 技术方法**

使用了线性层泄露、印记（imprint）方法、Direct‑Pass 初始化、稀疏基（如 DCT 或词向量）、通用 Lasso 稀疏回归以及理论误差上界。

**📊 数据集**

实验涵盖图像（MNIST、CIFAR‑10、ImageNet、HAM10000、Lung‑Colon Cancer）、文本（Wikitext）和音频（AudioMNIST）数据集，并在不同异质性、FedAvg、异步 FL 等场景下验证。

**📈 对比分析**

与 iDLG、IG、GI、FedLeak、Fishing、RtF、Trap Weight、LOKI、Scale‑MIA 等九种 GIA 进行比较；在 CNN/MLP、不同批量、梯度扰动、量化、稀疏化、DP、数据增强和安全聚合防御下，ARES 的 PSNR 提升 4–10 倍，恢复率提升 6–28%，并且理论误差上界仍保持良好。

**⚠️ 局限性**

局限性包括：仍依赖激活为 ReLU 或类似 ReLU 的线性+非线性结构；需要数据在稀疏域可表示；对极端高维或深层网络非线性累积的恢复能力有限；未针对对抗防御或同态加密等更强隐私机制进行评估。

---

## 511. TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models

**arXiv ID:** 2603.17828 | [PDF](https://arxiv.org/pdf/2603.17828v1)

**作者:** Qianlong Xiang `[一作]` (Harbin Institute of Technology), Liqiang Nie `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 29241 | [OpenAlex ID](https://openalex.org/A5038612499)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并验证了一种文本无关的逆向攻击 TINA，证明现有概念消除方法仅切断文本-图像映射而未消除视觉知识。

**💡 创新点**

提出了利用 DDIM 逆向和优化固定点损失的文本自由框架，能够在无文本条件下找到模型内部的确定性生成路径，突破传统文本基防御。

**🔧 技术方法**

使用 Stable Diffusion v1.4、DDIM 采样、DDIM 逆向、无文本条件、优化固定点损失、AdamW 等技术。

**📊 数据集**

利用 I2P（裸体）、Van Gogh 风格、Tench 物体等公开数据集生成的目标图像作为测试集。

**📈 对比分析**

与 12 种 SOTA 概念消除方法（ESD、FMN、AC、UCE、MACE、AdvUnlearn、SalUn、STEREO 等）及 5 种文本攻击（MMA、P4D、UDA、RAB、CCE）对比；TINA 在裸体、风格、物体任务上的 ASR 分别高达 70–90%，显著优于对手。

**⚠️ 局限性**

仅验证了已消除概念的可视化路径，对非文本相关概念的通用性仍待验证；依赖 DDIM 的确定性假设，可能对不同模型训练/微调存在适配挑战。

---

## 512. AdapTS: Lightweight Teacher-Student Approach for Multi-Class and Continual Visual Anomaly Detection

**arXiv ID:** 2603.17530 | [PDF](https://arxiv.org/pdf/2603.17530v1)

**作者:** Manuel Barusco `[一作]` (University of Padova), Gian Antonio Susto `[通讯]` (University of Padova)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计了一种轻量级的教师-学生框架AdapTS，用单一冻结骨干加可训练适配器实现多类别与持续学习的视觉异常检测

**💡 创新点**

创新点在于使用共享骨干与适配器实现统一的教师-学生结构，并结合分割引导训练和原型任务识别实现多任务持续学习

**🔧 技术方法**

采用教师-学生特征金字塔匹配、1×1卷积适配器、分割引导损失、Perlin噪声生成合成异常、原型向量任务识别与INT‑8量化

**📊 数据集**

主要使用MVTec AD和VisA两个工业视觉异常检测基准数据集

**📈 对比分析**

在单类、多类与持续学习三种场景下与STFPM、RD4AD、DeSTSeg等TS方法对比，AdapTS在保持相近准确率的同时将额外内存从百MB降低到个位数MB，显著提升内存效率

**⚠️ 局限性**

限制在于对新类别的适配器训练仍需手动更新，且在极大类别数或高分辨率任务下可能需要更多层适配器导致模型扩展受限

---

## 513. A 3D Reconstruction Benchmark for Asset Inspection

**arXiv ID:** 2603.17358 | [PDF](https://arxiv.org/pdf/2603.17358v1)

**作者:** James L. Gray `[一作]` (University of Sydney), Donald G. Dansereau `[通讯]` (University of Sydney)

**通讯引用:** 1827 | [OpenAlex ID](https://openalex.org/A5002518455)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个专为资产检验设计的3D重建基准数据集，并对现有SfM、MVS与最新端到端Transformer方法进行系统评估。

**💡 创新点**

创新点在于构建高重叠、近距离捕获、带有真实深度、相机位姿和网格的合成数据集，并通过在窗口表面引入不同程度的污渍来单独评估非Lambertian表面的影响，揭示了当前方法在资产检验场景中的可扩展性缺口。

**🔧 技术方法**

技术方法包括使用Blender渲染合成图像并模拟传感器噪声；采用COLMAP、GLOMAP、Depth Anything 3、VGGT和π³等方法；通过Sim(3)变换对批处理结果进行全局对齐，并利用sRMSE、AUC和Chamfer/F1等指标进行评估。

**📊 数据集**

所用数据集为作者自行生成的三场景（办公楼、起重机、桥梁）合成数据集，提供每帧图像、相机位姿、深度图与网格；同时与ETH3D、BlendedMVS等公开数据集做对照。

**📈 对比分析**

评估通过对单帧深度、相机位姿和点云进行精度比较，结果显示端到端Transformer模型在高重叠近距离序列中表现不佳（AUC低、点云误差大），而COLMAP与GLOMAP在位姿和点云上相对更稳健；但仅用全局点云指标无法反映局部几何精度，建议使用单帧深度误差。

**⚠️ 局限性**

局限性包括：端到端模型受GPU显存限制无法一次性处理完整高分辨率序列；Transformer方法在小视角、局部可见性强的资产检验场景下姿态估计差；全局点云指标无法捕捉局部几何误差；数据集仍为合成，存在sim-to-real差距。

---

## 514. The State of Generative AI in Software Development: Insights from Literature and a Developer Survey

**arXiv ID:** 2603.16975 | [PDF](https://arxiv.org/pdf/2603.16975v1)

**作者:** Vincent Gurgul `[一作]` (Humboldt-Universität zu Berlin), Stefan Lessmann `[通讯]` (Humboldt-Universität zu Berlin)

**通讯引用:** 6598 | [OpenAlex ID](https://openalex.org/A5057226223)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了生成式人工智能（GenAI）在软件开发生命周期（SDLC）中的影响，结合系统文献综述与65位开发者的问卷调查，对各阶段效益与治理进行评估。

**💡 创新点**

创新点在于将学术研究、行业报告和实证问卷数据三者整合，提供跨阶段、跨组织治理视角的全生命周期评估，并揭示GenAI对角色与治理的结构性影响。

**🔧 技术方法**

采用系统综述方法、TAM框架问卷设计、定量统计与对比分析等技术手段。

**📊 数据集**

使用的数据集包括2021-2026年Scopus、IEEE Xplore、ACM DL、SpringerLink、arXiv等数据库的63篇文献以及2025-2026年收集的65份软件工程师问卷。

**📈 对比分析**

通过对问卷自评时间节省与文献中报告的生产率提升进行比较，发现设计、实现、测试阶段对GenAI效益最高，平均时间缩短约30-50%，但缺乏客观实验对比，主要基于自评结果。

**⚠️ 局限性**

受限于样本偏年轻、地区集中、数据为自评且跨来源不统一，难以验证客观性能和长期影响，也未能充分评估技术细节与安全风险。

---

## 515. Maximum-Projection-Based Bayesian Optimization Utilizing Sensitivity Analysis for High-Efficiency Radial Turbine Design with Scarce Data

**arXiv ID:** 2603.17516 | [PDF](https://arxiv.org/pdf/2603.17516v1)

**作者:** Eric Diehl `[一作]` (Siemens AG), Dimitrios Loukrezis `[通讯]` (Centrum Wiskunde & Informatica)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一套基于最大投影实验设计、贝叶斯优化和全局敏感性分析的稀疏数据风机叶片设计工作流。

**💡 创新点**

创新点在于将 MaxPro 的投影空间填充特性与 GP-UCB 贝叶斯优化、PCE 计算 Sobol 指数相结合，实现了在有限 CFD 计算预算下的高效参数降维与优化。

**🔧 技术方法**

使用的技术包括：最大投影实验设计（MaxPro）、高斯过程（GP）回归与上置信界（UCB）采集函数、多项式混沌展开（PCE）及其 Sobol 敏感性指标。

**📊 数据集**

数据集由 10 维设计参数组成，采用 330 次高保真 CFD 仿真（含滴滴式冷凝）构成的训练与验证集。

**📈 对比分析**

与单纯 MaxPro 设计或全 10 维贝叶斯优化相比，该工作流将叶片效率从 85.77% 提升至 91.77%，GP 与 PCE 的预测误差低于 2%，显著提升了样本利用率。

**⚠️ 局限性**

局限性包括：仅针对单一运行点、依赖单一 CFD 模型、对小样本敏感性分析的可靠性有限，且未考虑多目标优化或多点运行的情形。

---

## 516. Pretrained Multilingual Transformers Reveal Quantitative Distance Between Human Languages

**arXiv ID:** 2603.17912 | [PDF](https://arxiv.org/pdf/2603.17912v1)

**作者:** Yue Zhao `[一作]` (National University of Singapore), Weijie Su `[通讯]` (University of Pennsylvania)

**通讯引用:** 8115 | [OpenAlex ID](https://openalex.org/A5080575294)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于预训练多语言Transformer注意力机制的Attention Transport Distance（ATD），用于量化不同语言之间的相似度，并将其作为低资源翻译的正则化手段；

**💡 创新点**

创新点在于将注意力分布视为概率分布，利用最优运输（Wasserstein‑2）计算语言间距离，完全摆脱了分词器依赖，直接从模型内部获取跨语言结构；

**🔧 技术方法**

主要技术包括多语言Transformer模型（M2M‑100、Llama‑3）提取交叉注意力矩阵、最优运输距离计算、邻接树（Neighbor‑Joining）重构语言树，以及将ATD嵌入低资源翻译的损失函数；

**📊 数据集**

使用的数据集有：M2M‑100模型对81–100种目标语言的英文源句子翻译产生的注意力矩阵；低资源翻译实验使用OPUS‑100的英文–目标语言并行语料；对照性分析借助WALS、Glottolog等语言学资源；

**📈 对比分析**

通过与已知语言家系树、地理分布和接触网络进行可视化与统计比对，ATD距离矩阵的Cophenetic相关系数高达0.9881，准确重现语言族结构；在低资源翻译中加入ATD正则化后，BLEU/chrF/TER/COMET均明显提升，证明其对跨语言迁移有积极作用；

**⚠️ 局限性**

局限性包括：ATD衡量的是模型内部功能相似度而非真实历史或语义关系；受训练数据偏差、低资源语言翻译质量不足和脚本差异影响；正则化实验仅覆盖少数几种语言，未系统验证更广泛适用性。

---

## 517. Actionable Recourse in Competitive Environments: A Dynamic Game of Endogenous Selection

**arXiv ID:** 2603.17907 | [PDF](https://arxiv.org/pdf/2603.17907v1)

**作者:** Ya-Ting Yang `[一作]` (New York University), Quanyan Zhu `[通讯]` (New York University)

**通讯引用:** 11241 | [OpenAlex ID](https://openalex.org/A5081500464)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在竞争性环境下，个体如何通过可变特征实现可操作回路，并提出了基于风险的选择规则与闭环动态博弈框架。

**💡 创新点**

创新点在于将可操作回路视为多方互动的动态博弈，利用CVaR上尾选择规则形成自生阈值与方向，并揭示了自生选择机制导致的阶层化与性能加剧。

**🔧 技术方法**

使用CVaR优化、二次正则化、拉格朗日对偶分析、对数壁障成本函数以及闭环动力学推导等技术手段。

**📊 数据集**

采用模拟数据（以GRE和GPA为例的学生特征），未使用公开数据集。

**📈 对比分析**

与传统静态回路方法对比，动态模型能够捕捉阈值漂移和效益递减的现象；数值实验表明决策边界逐渐收敛、方差下降，显示模型能预测长期稳定性。

**⚠️ 局限性**

局限包括：假设线性分数规则、单一可行动特征、忽略多维可行动空间、缺乏真实数据验证、动态系统的解析与收敛性质尚未完全阐明。

---

