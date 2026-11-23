# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-11-23 | 今日论文总数: 615

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Cecilia: Towards Layout-Inclusive Optical Music Recognition

**arXiv ID:** 2611.27465 | [PDF](https://arxiv.org/pdf/2611.27465v1)

**作者:** Congren Dai `[一作]` (Central Conservatory of Music), Maosong Sun `[通讯]` (Central Conservatory of Music)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

该论文未提供具体研究内容，无法确定其具体工作。

**💡 创新点**

无法确定创新点。

**🔧 技术方法**

无法确定使用的技术。

**📊 数据集**

无法确定使用的数据集。

**📈 对比分析**

无法确定比较方法及其性能表现。

**⚠️ 局限性**

由于缺乏必要信息，无法评估论文的限制。

---

## 2. Beyond Prompt Heuristics: Assessing Child Safety Alignment in LLMs

**arXiv ID:** 2611.26965 | [PDF](https://arxiv.org/pdf/2611.26965v1)

**作者:** Anika Misra `[一作]` (University of Michigan), Rada Mihalcea `[通讯]` (University of Michigan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大型语言模型在儿童安全对齐上的表现，探讨模型是否能在无明显危害提示的请求中保持安全。

**💡 创新点**

发现儿童安全对齐对危害提示高度依赖，并通过单词干预证明词汇关联可显著改变模型行为。

**🔧 技术方法**

使用激活补丁技术（activation patching）结合行为评估和LLM判定器分析早层语义对安全行为的影响。

**📊 数据集**

使用LUCid儿童安全子集，并从SafeBench、CASE-Bench等挑选118个带/不带危害提示的查询。

**📈 对比分析**

在六款模型（GPT-5.4-mini、Claude-Haiku-4.5、Gemini-3-Flash、Gemma-12B、Qwen3.5-4B、Qwen3.5-27B）上计算儿童响应误差率（CMR），结果显示无危害提示时误差率高达80%；加入单词干预后误差率下降14–57个百分点。

**⚠️ 局限性**

仅评估娱乐推荐等有限场景，词干预词汇有限，机制分析仅在部分模型，结果可能不适用于其他领域或语言。

---

## 3. Explicit Style Extraction for Personalized Long-Form Text Generation

**arXiv ID:** 2611.27211 | [PDF](https://arxiv.org/pdf/2611.27211v1)

**作者:** Jenitha Reddy Patel `[一作]` (University of Chicago), Ryan A. Rossi `[通讯]` (Zillow)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个明确的写作风格提取与校正流程，用于个性化长文本生成

**💡 创新点**

创新点在于构建了28个写作风格维度的分类体系，并通过模型在推理时先生成可审计的风格配置文件，再以此指导后续草稿的修订

**🔧 技术方法**

技术包括检索增强生成（Retrieval-Augmented Generation）、多轮提示工程、风格配置文件生成与基于该文件的双轮修订以及使用LLM自我评分与外部LLM评审

**📊 数据集**

使用LongLaMP benchmark中的三种长文本任务（科研摘要、产品评论、Reddit帖子），每个任务均采用公开用户写作历史和对应参考文本

**📈 对比分析**

与LongLaMP检索增强基线相比，提出的方法在27个ROUGE/METEOR指标中25个提升，平均提升10.2%，在LLM评审的风格保真度上亦显著提高；与仅做一次隐式推断的单轮方法相比，改进有限但仍有提升

**⚠️ 局限性**

局限性包括仅在单语种（英语）和LongLaMP的用户设置上评估，缺乏对不同检索器、k值、微调方法的探索；评估指标仍以ROUGE/METEOR和LLM评审为主，未涉及人类评估；风格配置文件仍需手工验证并可能泄露敏感信息

---

## 4. UDA-Restoration: Bridging Unsupervised Domain Adaptation and Multi-Degradation Denoising for Enhanced AV Semantic Segmentation

**arXiv ID:** 2611.27097 | [PDF](https://arxiv.org/pdf/2611.27097v1)

**作者:** Zainab Bibi `[一作]` (National University of Sciences and Technology), Mohsin Kamal `[通讯]` (National University of Sciences and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于扩散模型的无监督域适应框架 UDA‑Restoration，用于在自动驾驶环境中对受多种降解影响的图像进行去噪和恢复，并评估其对下游语义分割任务的提升。

**💡 创新点**

创新点在于：① 在扩散噪声空间进行噪声域适配（Noise‑DA）并加入通道洗牌与残差交换，显著提升跨域泛化；② 采用多源降解学习，将合成与真实降解混合训练；③ 通过下游语义分割作为评价指标，验证图像恢复对实际感知任务的直接影响。

**🔧 技术方法**

核心技术包括：扩散模型、Noise‑DA（噪声空间域适配）、通道洗牌、残差交换、多源降解生成、U‑Net/DeepLabV3/SegFormer 语义分割网络。

**📊 数据集**

使用的公开数据集有 Cityscapes、ACDC、KITTI Noisy、RainDAR、UA‑DETRAC、BDK，另外生成了 Gaussian、雨雪雾、运动模糊、低光、镜头畸变等合成降解。

**📈 对比分析**

与监督方法（DnCNN、NAFNet）及其他自监督/无监督基线（CycleGAN、Noise2Void‑NAFNet、Blind2Unblind‑UNet、NAFNet‑SCPGABNet）对比，Noise‑DA 在 PSNR 上达到 31.63 dB、SSIM 0.9777，匹配甚至优于最佳监督结果；在语义分割上，恢复后的图像使 mIoU 提升 4–6 %（SegFormer 达 62.19 %）。

**⚠️ 局限性**

局限性包括：① 仍需要大量计算资源（扩散模型训练时间约 4 天）；② 对极端未见降解（如剧烈天气或硬件故障）的鲁棒性尚未充分验证；③ 评估范围主要集中在语义分割，未涉及目标检测、跟踪等其他感知任务；④ 需进一步验证实时部署可行性。

---

## 5. Geo-Force: A Theory of Force Sensing from Object Geometry

**arXiv ID:** 2611.26992 | [PDF](https://arxiv.org/pdf/2611.26992v1)

**作者:** Abdullah Ayad `[一作]` `[通讯]` (University of Technology Nuremberg), Abdullah Ayad (University of Technology Nuremberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了“几何力（Geometric Force）”概念，用深度渲染、触摸映射和力解码器三步从物体几何信息估计虚拟触摸传感器的每个点的接触力。

**💡 创新点**

首次给出虚拟触摸传感器的形式化定义与性质，证明其坐标变换等变、无接触时为零、误差随渲染、触摸与力映射误差线性增长，并揭示几何信息无法识别物体刚度的根本极限。

**🔧 技术方法**

采用深度摄像机渲染、学习或仿真触摸映射（如U‑Net或物理仿真）、预训练的力解码器，并利用Lipschitz连续性与填充距离等理论工具推导误差上界。

**📊 数据集**

论文本身为理论性工作，没有使用具体数据集；若实施，可在真实机器人抓取任务中使用深度相机图像与对应的触摸传感器数据进行触摸映射的训练。

**📈 对比分析**

未进行实验对比，文中仅给出误差理论上界和条件分析，未给出实际性能指标。

**⚠️ 局限性**

局限性包括：仅考虑静态/准静态情况，忽略动力学、摩擦、剪切、粘弹性、历史依赖等；几何信息无法捕捉刚度，导致不可恢复的误差；对渲染、形状重建与姿态估计误差的假设较强，实际应用需满足严格的硬件与对象约束。

---

## 6. What Does a Zoom Actually Add? Decomposing Fine-Grained Perception Failures in Vision-Language Models

**arXiv ID:** 2611.26904 | [PDF](https://arxiv.org/pdf/2611.26904v1)

**作者:** Zhixuan Ge `[一作]` (Rice University), Wei Qiu `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统性拆分并量化了视觉语言模型（VLM）中放大（zoom）操作对解决失败的具体贡献，采用八种二次观察（second-look）条件和对应的随机遮罩对比，探究不同预算下缺失像素、重标记、背景干扰和新像素对纠正错误的实际影响。

**💡 创新点**

创新点在于将zoom效应分解为四个可区分的机制（不编码、重编码、重标记、获取新像素），并通过“最便宜”原则为每个失败精确归因，同时给出不同放大倍率下像素可用性的阈值，为未来的zoom策略提供资源地图和理论依据。

**🔧 技术方法**

主要技术包括：对VLM的视觉编码器预处理进行精确复现；构造八种二次观察条件（缓存重插入、区域仅编码、背景掩码、同密度重编码、像素上采样、原始裁剪等）；使用sham（随机相同大小区域）进行对照；对结果进行统计分解、逻辑回归分析以及放大倍率扫掠。

**📊 数据集**

使用的数据集为：V^*（191个属性与空间问题）、ZoomBench（844个多选与计数问题）以及TextVQA（1000个文本区域问题），模型包括Qwen2.5‑VL‑7B、Qwen3‑VL‑8B与InternVL3‑8B，实验覆盖17×、4×、1.1×与接近原始分辨率的四个像素预算。

**📈 对比分析**

比较方法是将每个zoom救助的失败按八种条件归类，计算各条件在不同预算下的修复比例，并与随机遮罩做对比；结果显示即使在17×的极低预算下，重标记（upsampled seen pixels）即可修复47–73%失败，且大多数（68–94%）不需要新像素。

**⚠️ 局限性**

限制包括：仅使用oracle裁剪区域，忽略定位误差；实验仅覆盖3种7–8B模型；对sham的设计假设随机区域与目标无关；结果对更大规模或不同token化策略的模型的泛化性尚未验证。

---

## 7. PIU-CR: Physics-Informed Deep Unfolding with SAR-Optical Image Fusion for Cloud Removal

**arXiv ID:** 2611.27093 | [PDF](https://arxiv.org/pdf/2611.27093v1)

**作者:** Gengque Fan `[一作]` (Southeast University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出基于大气散射模型的物理信息深度展开网络PIU-CR，用光学和合成孔径雷达（SAR）多模数据实现云覆盖图像的去云恢复；

**💡 创新点**

创新点在于将大气散射模型直接嵌入数据保真项并构造SAR引导的结构对齐与光谱对齐正则化，随后通过半二次分裂（HQS）将整体优化展开为可解释的多阶段深度网络，兼具物理可解释性与深度学习的表示能力；

**🔧 技术方法**

使用的技术包括：大气散射模型、半二次分裂求解、物理信息深度展开、SAR引导的结构和光谱正则化、结构对齐网络、结构引导光谱交叉注意力、L1+SSIM+SAM三重损失以及多阶段迭代训练；

**📊 数据集**

实验数据集为SEN12MS-CR（Sentinel-2光学+Sentinel-1 SAR）和SMILE-CR（Landsat-8光学+SAR）两大遥感多模数据集；

**📈 对比分析**

与多种单模（AMGAN、ACANet、HSFormer、CloudRuler）和多模（DSen2-CR、GLF-CR、HS2P、USSRN、HPN-CR、EMRDM）SOTA方法在PSNR、SSIM、MAE、RMSE、SAM等指标上均取得最高值，例如在SEN12MS-CR上PSNR 28.86 dB、SSIM 0.889，表明恢复质量明显优于对比方法；

**⚠️ 局限性**

局限性包括：仍需配套SAR数据，对无SAR或多源异构环境的适用性有限；迭代次数固定为3，可能在极端云覆盖下仍有恢复瓶颈；训练时对大规模多模数据集依赖较高，部署时对算力和内存有一定要求。

---

## 8. BitPrefill: W1.58A4 Tensor-Core Acceleration for Compute-Bound LLM Prefill

**arXiv ID:** 2611.26657 | [PDF](https://arxiv.org/pdf/2611.26657v1)

**作者:** Fang Chen `[一作]` (Case Western Reserve University), Gourav Datta `[通讯]` (Case Western Reserve University)

**通讯引用:** 892 | [OpenAlex ID](https://openalex.org/A5017435097)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种W1.58A4算子，将LLM预填充阶段从BF16张量核迁移到INT4张量核，实现三元权重与4‑bit激活的低精度计算，并在RTX4090和Jetson Orin Nano上实现

**💡 创新点**

创新点在于①利用INT4张量核加速计算密集型预填充；②设计形状自适应内核选择（SAKS）与Stream‑K调度；③在GEMM后端用Epilogue Visitor Tree融合量化、缩放与BF16输出，消除中间INT32张量

**🔧 技术方法**

使用技术包括CUTLASS、Stream‑K、INT4 GEMM、Epilogue Visitor Tree、量化感知训练（SFT）、自适应张量核配置、JetPack等

**📊 数据集**

训练数据集为85,805例多任务数据（ShareGPT、Alpaca、GSM8K、Anthropic HH、OpenAssistant）；推理评估使用WikiText‑2测算困惑度

**📈 对比分析**

与BF16、官方BitNet、BitBLAS三基线在RTX4090和Jetson Orin Nano上对512‑token提示的LLM推理进行对比；性能方面预填充延迟从41.10 ms降至28.94 ms（1.42×加速），能耗降低42%，功率下降；Jetson上3B模型仅在本文方案下可运行，且能耗更低

**⚠️ 局限性**

限制在于极长序列仍需进一步优化；量化后模型在某些任务的精度略有下降；仅适用于支持INT4张量核的GPU，移植到不支持INT4的设备受限

---

## 9. CADENCE:Temporally Coherent Whole-Body Human Mesh Recovery that Preserves Expressive Hand Motion

**arXiv ID:** 2611.26783 | [PDF](https://arxiv.org/pdf/2611.26783v1)

**作者:** Keniel Peart `[一作]` (University of Southampton), Shelly Vishwakarma `[通讯]` (University of Southampton)

**通讯引用:** 784 | [OpenAlex ID](https://openalex.org/A5042432973)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了CADENCE系统，结合单帧全身SMPL-X估计器、共享视觉编码器的高分辨率手部细化路径以及离线多速率时序重构，实现单目视频中全身、手部与面部三维网格的稳定恢复。

**💡 创新点**

创新点：① 共享C‑RADIOv4编码器与手部裁剪的高分辨率路径，避免了单独手部编码器带来的复杂度；② 基于关节树注意力的多速率时序重构，针对每个关节/帧自适应选择零相位滤波器混合，既平滑身体运动又保留手部快速动作；③ 统一训练策略与损失，兼顾姿态、形状、表情与相机参数。

**🔧 技术方法**

技术手段包括Transformer解码器、C‑RADIOv4视觉编码器、ViTPose手部关键点检测、零相位滤波器混合、关节树注意力、LoRA微调、MSE等多任务损失。

**📊 数据集**

使用的数据集包括AGORA、BEDLAM、UBody、Motion‑X、3DPW、EHF、AGORA/UBody验证、SignAvatars/How2Sign、ARCTIC、COCO‑WholeBody、MPII、FreiHand、HanCo、ReInterHand、HaMeR、ARCTIC等。

**📈 对比分析**

与多种平滑器（Gaussian、Moving Average、One‑Euro、Savitzky–Golay、SmoothNet‑WB、单速率重构）以及现有全身/身体模型在3DPW、EMDB、EHF、AGORA、UBody、ARCTIC等基准上对比。CADENCE在3DPW上将加速度误差从17.4降至5.4 mm/s²，PA‑MPJPE从41.9降至40.2 mm；在Sign语言序列手部抖动最低（1.29 mm）且峰值幅度保持0.98；在EHF手部P‑MPJPE为8.3 mm；在AGORA/UBody全身误差最低；在ARCTIC手部误差相对较大。

**⚠️ 局限性**

局限性：仅处理单人框框，不支持多人人体检测与身份区分；时序重构为离线非实时；全局姿势、深度仍存在模糊；手部在运动模糊、物体接触与遮挡时精度下降；依赖外部2D关键点检测；手部动态评价基于伪ground truth；不具备流式实时推理能力。

---

## 10. CoEvoBench: Can LLMs Drive Recursive Co-Evolution of Agentic Systems?

**arXiv ID:** 2611.27116 | [PDF](https://arxiv.org/pdf/2611.27116v1)

**作者:** Chuan Xiao `[一作]` (Zhejiang University), Xander Xu `[通讯]` (Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并发布CoEvoBench基准，用以评估LLM研究者在递归自我改进过程中同时优化模型权重与执行harness的能力。

**💡 创新点**

将模型与harness的协同进化纳入同一可测评循环，并通过让研究者自行做决策来揭示任务属性对协同进化成效的影响。

**🔧 技术方法**

使用LLM微调（LoRA）、教师模型辅助数据生成、sandbox化执行环境、可见/选择/测试集分离评估以及有限GPU与调用预算等技术。

**📊 数据集**

基于五个agentic任务域的公开基准（SWE‑bench Verified、TB‑Lite、AppWorld easy、τ²‑bench Telecom、BFCL‑V4），并按可见、选择、测试集拆分。

**📈 对比分析**

通过与基线Qwen3.5‑9B‑Base及指令调优模型+参考harness的分数差异评估，研究者在不同域的平均分提升差距从0到96%，但大多数研究者提升不足30%，且多在首轮获得收益。

**⚠️ 局限性**

受限于仅使用离线微调、未采用强化学习，导致在长程编码任务中难以充分利用训练；评估仅在可见/选择集上做判定易导致过拟合；缺乏自动化决策机制导致递归循环易停滞。

---

## 11. Computing Weighted Set Covers with Heterogeneous Stochastic Oracles: Deadline-Constrained Token Complexity

**arXiv ID:** 2611.26656 | [PDF](https://arxiv.org/pdf/2611.26656v1)

**作者:** Jie Wang `[一作]` (University of Massachusetts Lowell), Jie Wang `[通讯]` (University of Massachusetts Lowell)

**通讯引用:** 3618 | [OpenAlex ID](https://openalex.org/A5100440079)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了利用异质随机oracle（AI系统）并结合概率图灵机（PTM）来求解加权集合覆盖（WSC）问题，并对oracle选择策略的令牌成本与成功概率进行理论分析与优化。

**💡 创新点**

创新点包括：① 把AI模型抽象为异质随机oracle，加入对LP松弛下的对偶证书以验证结果；② 证明最优策略在达到最大成功概率时可随机化至最多两个批次；③ 针对两批次最大成功概率提出FPTAS并证明NP-hard性；④ 对部分覆盖响应给出一轮与自适应选择的成功概率和令牌成本上界，并展示残差引导查询在令牌成本上线性优势。

**🔧 技术方法**

主要技术包括LP松弛与对偶检验、子模函数与子模最大化近似、线性规划求解、动态规划、贪心子模覆盖、NP-hard性归约、以及概率论中的失败不变性与期望计算。

**📊 数据集**

实验使用了OR-Library中的铁路排班实例和Defects4J测试覆盖矩阵，针对每个实例构造了降维后的小规模实例进行评估。

**📈 对比分析**

与传统加权贪心（WG）及反刍删除（RD）算法比较，LLM在大部分实例中生成更轻的覆盖，实验显示平均令牌成本低于1美元/调用，且在两倍延迟下即可实现最大成功概率。

**⚠️ 局限性**

局限性包括：① 只在已降维实例上验证，未在大规模真实实例上评估；② 假设oracle成功概率相互独立且已知；③ 未对未知分布进行在线学习；④ 对偶证书假设可能不适用于所有实例；⑤ NP-hard性和FPTAS的实际运行时间在极大k或高成本时可能受限。

---

## 12. Bel: Step-Level Policy Improvement of Visual Agents Without Training

**arXiv ID:** 2611.27222 | [PDF](https://arxiv.org/pdf/2611.27222v1)

**作者:** Jiajun Fan `[一作]` (University of Illinois Urbana Champaign), Ge Liu `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种名为Mark、Ask、Act的方法，在不训练模型的前提下，通过在每个候选动作所在的图像上绘制标记、使用决策模型评估其正确性，并在满足阈值条件时替换贪婪动作，从而提升冻结的视觉代理（GUI和机器人指向）性能。

**💡 创新点**

创新点在于：①将每个候选动作可视化到其实际作用的图像上，以避免仅凭坐标做决策；②使用单一概率尺度对不同类型动作进行统一评分；③采用阈值阈值/余量的“Act”策略仅在必要时覆盖原始动作，保持高置信度。

**🔧 技术方法**

主要技术包括：图像标记（在截图或相机图像上绘制红点/文本）、多模态决策模型Jev的概率预测、阈值设置和贪婪动作的替换规则。

**📊 数据集**

使用了多种GUI与机器人指向数据集：AndroidControl-High/Low、GUI-Odyssey、GUI-Act-Web、OmniAct-Web/Desktop、Multimodal-Mind2Web、ScreenSpot/Pro、MiniWoB++、AndroidWorld、VABench-Point、Part-Afford、Point-It-Out、RoboSpatial-Home、Where2Place、RefSpatial-Bench 等。

**📈 对比分析**

与多数投票、区域投票、CATTS、GUI-Critic-R1、Qwen3-VL-8B/Judge、ICM-r2、GUI-Actor等方法比较，Mark‑Ask‑Act 在步级成功率和定位准确率上均显著提升，平均提升幅度可达数个百分点，且在现场评估中亦提高任务成功率。

**⚠️ 局限性**

局限性包括：①依赖外部大型多模态模型进行评分，计算成本较高；②阈值需在每个策略的开发集上调优，跨策略迁移有限；③仅适用于已有动作采样的冻结策略，无法处理动态生成候选；④对极细粒度动作或非可视化交互（如语音）适用性尚未验证。

---

## 13. Coherence by Construction: Synthesizing CTI Knowledge Graphs and the Formal Limits of Coherence Validation

**arXiv ID:** 2611.26717 | [PDF](https://arxiv.org/pdf/2611.26717v1)

**作者:** Tanzimul Alam Fahim `[一作]`, Mohammad Arif Hossain `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用LLM生成并验证网络攻击与防御场景，提出Mirage模拟框架并通过STIX 2.1语义模型对输出进行schema强制和验证。

**💡 创新点**

将LLM与STIX 2.1语义模型结合，构建 deterministic compiler 与多维验证套件，验证生成内容的结构与合理性，实现结构区分度极高的生成。

**🔧 技术方法**

使用多模型LLM、schema validation、deterministic compiler、discriminator+ER控制、RotatE语义嵌入以及STIX 2.1 bundle等技术。

**📊 数据集**

使用公开的STIX 2.1安全数据集以及内部生成的攻击/防御实例，评估随机图和真伪判别者。

**📈 对比分析**

采用 rubric、双盲LLM评审、discriminator AUC、inductive transfer、augmentation等多维度方法比较，结构区分度达到AUC 1.0，相关性最高ρ≈0.75，但与真实ATT&CK区分不明显。

**⚠️ 局限性**

验证易出现空洞（随机图通过）、现实感受受限于schema-forced baseline，缺乏真实网络环境验证，且跨域推广能力有限。

---

## 14. HermiteTwin: Bicubic Hermite Shell Digital Twins for Contact-Rich Deformable Manipulation

**arXiv ID:** 2611.27282 | [PDF](https://arxiv.org/pdf/2611.27282v1)

**作者:** Kangrui Zhang `[一作]` (Nankai University), Bo Ren `[通讯]` (Nankai University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种基于多视角视频的薄弹性物体物理数字孪生框架，能够从抓取实验中重建物体几何、动力学和外观，并实现再模拟与未来预测。

**💡 创新点**

创新点在于将形变和接触分别用粗糙的双三次 Hermite 壳体和密集的材料空间抓取样本进行耦合，既保留了弯曲的物理准确性，又在接触上实现了高分辨率的细致模拟。

**🔧 技术方法**

采用双三次 Hermite 壳体、Kirchhoff–Love 弯曲能量、增量潜在接触 (IPC)、CMA‑ES 与 adjoint 梯度的逆仿真优化，以及三维高斯剖分渲染等技术。

**📊 数据集**

实验数据来自 Deform360 多视角视觉触觉数据集，包含十段红布与白泡沫板的抓取交互。

**📈 对比分析**

与 PhysTwin（弹簧‑质量模型）和 GS‑Dynamics（学习点动学）在重现、未来预测、Silhouette IoU、PSNR 等指标下对比，本文方法在 Chamfer 距离、IoU 和 PSNR 上均优于两者，提升约 30% 以上。

**⚠️ 局限性**

主要局限是只能从单个交互中识别有效参数，缺乏跨动作的泛化验证；对高质量跟踪观测高度依赖，稀疏观测下性能有限；渲染未与动力学联合学习，无法自适应表面细节与视角变化。

---

## 15. Mem-X: A Memory-Dependent Cross-Episode Benchmark for Vision-and-Language Navigation

**arXiv ID:** 2611.27166 | [PDF](https://arxiv.org/pdf/2611.27166v1)

**作者:** Zhikang Shi `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Mem-X跨回合记忆依赖的视觉语言导航（VLN）基准，并给出了相应的评估协议；同时设计了可模块化的MemBridge基线，将语义场景记忆、跨回合定位和粗细检索融合到现有VLN代理中。

**💡 创新点**

1) 通过显式的回合依赖设计，将历史经验转化为任务指令，真正使得完成后续任务需要利用记忆；2) 形成三层难度级别，分别考察对象回忆、内存定位/索引和离内存重新定位及路线重现；3) 提供统一的记忆相关指标（MRR、Loc‑SR、C‑SR、ST‑nDTW），补充传统终点指标；4) MemBridge的模块化设计可无缝集成至多种VLN模型。

**🔧 技术方法**

基于RGB‑D的语义场景图构建（使用DINOv3特征、SAM3分割、三维位置估计），跨回合自定位（SuperPoint + LightGlue + RANSAC 计算SE(2)变换），粗细检索（VLM解析指令 + 语义匹配 + 细化检索），以及基于占用图的路径规划。

**📊 数据集**

在HM3DSem 34个室内场景中收集了710个会话、2071条轨迹（1204条记忆依赖），每条轨迹平均约14.6米，分为3个级别。

**📈 对比分析**

将四个主流VLN代理（CMA、ETPNav、JanusVLN、Uni‑LaViRA）在Mem-X上进行基线与加上MemBridge后的对比。实验显示，MemBridge在记忆相关指标上均有显著提升（例如C‑SR从约5–7%提升至20–30%，MRR提升约10–20%），但终点成功率（SR）提升有限；在最难的Level 3中，C‑SR仍低于30%。

**⚠️ 局限性**

1) 仅涵盖单层楼层场景，缺乏多楼层或更复杂环境；2) 依赖仿真器的精确位姿，真实世界中的里程计误差未考虑；3) 记忆任务仍难以完全逼真，缺少多模态交互或更长记忆跨度的测试；4) 对齐与检索策略仍受限，导致Level 3仍未达到高成功率。

---

## 16. PonderPounce-WM: When Does Predicting the Future Help Context-Dependent Robot Control?

**arXiv ID:** 2611.26996 | [PDF](https://arxiv.org/pdf/2611.26996v1)

**作者:** Jaeyoon Jung `[一作]` (MAUM.AI), Youngjae Yu `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在多模态大语言模型中加入未来预测的辅助监督，探索未来预测如何提升基于上下文的机器人控制效果。

**💡 创新点**

创新点在于：①利用未来视觉特征预测作为辅助任务而不将未来信息直接送入动作模型；②研究未来标记在模型中的顺序和投递方式，发现“先认知后未来”且只投递认知的配置可最大化性能；③通过线性探测验证认知表示对未来信息的可解码性。

**🔧 技术方法**

主要技术包括：预训练的 Qwen3.5-9B 多模态大语言模型、SigLIP 视觉编码器、三时延未来标记的 MSE 预测损失、线性探测与知识蒸馏、动作模型 π_0.5 以及交叉熵和流匹配损失的联合优化。

**📊 数据集**

使用数据集：RoboMME（16个桌面任务，每个任务 50 条评估轨迹）以及四个真实机器人任务（Put Fruits、Track Cube、Repick Block、Draw Pattern）。

**📈 对比分析**

在 RoboMME 上，认知‑优先、仅投递认知配置的成功率从 60.83% 提升到 67.42%（+6.59%），直接投递未来标记下降至 64.00%；在四个真实任务上，平均成功率从 60.98% 提升到 67.44%。实验通过对比不同标记顺序与投递方式的配置、线性探测准确率及实际任务成功率来评估性能。

**⚠️ 局限性**

局限性包括：①实验仅基于一种多模态 LLM 与一种视觉编码器，缺乏跨模型验证；②无法完全将未来预测的益处与其他辅助视觉目标分离；③评估采用异步执行且认知年龄不一致，难以精确量化未来预测对控制的因果影响；④记忆保持分析受限于所选探测指标，无法全面评估长时记忆的完整性。

---

## 17. Gradient-based optimization of buoyancy distributions for lazy-wave risers

**arXiv ID:** 2611.27361 | [PDF](https://arxiv.org/pdf/2611.27361v1)

**作者:** Nicolò Pollini `[一作]` (Technion-Israel Institute of Technology), Kazem Jaber `[通讯]` (Technion-Israel Institute of Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于梯度的懒波浮式管线浮力分布优化方法，能在给定水平吊力和水流条件下最小化浮力需求，同时满足曲率与有效张力约束。

**💡 创新点**

创新点在于：1) 将离散伴随方法与非线性平衡求解相结合，实现对任意数量设计变量的高效梯度获取；2) 提供三种设计参数化（连续节点浮力、分组浮力、离散材料优化），使优化结果既具高性能又可直接转化为可制造的模块布局；3) 通过两阶段增量-迭代求解（先增长线长后升力）提升求解稳定性。

**🔧 技术方法**

采用的技术包括：基于最小势能的几何非线性弹性梁模型、有限差分离散、Newton–Raphson 迭代配合 Armijo 回溯搜索、伴随灵敏度分析、p‑norm 约束处理、以及 IPOPT 求解器进行梯度优化。

**📊 数据集**

使用的“数据集”是典型的海上浮式管线参数（如表1中的直径、厚度、弹性模量、海深等），并与 Trapper（2020）报告的基准结果进行对比；未使用公开大规模实验/仿真数据集。

**📈 对比分析**

性能比较方法：①对单一基准案例的弹性嵌入、张力、曲率等响应与 Trapper 2020 进行数值验证，误差均≤0.1%；②在三种参数化下记录 IPOPT 的迭代次数和计算时间，结果显示连续参数化耗时最长、离散材料优化耗时最低；③目标函数值在连续与离散参数化之间相差≤0.3%。

**⚠️ 局限性**

限制：仅考虑静态平衡分析，未包含动力响应、三维土壤摩擦或不规则海底；优化问题非凸，结果依赖初始猜测；对复杂现场条件（如多模组尺寸、实际安装误差）需进一步验证。

---

## 18. Spiking Neural Networks Incrementally Track Firing-Rate Networks

**arXiv ID:** 2611.27322 | [PDF](https://arxiv.org/pdf/2611.27322v1)

**作者:** Hyunseok Oh `[一作]` (Seoul National University), Youngki Lee `[通讯]` (Seoul National University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e0540dec-d77f-42db-94ae-d039248f6393` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了离散LIF神经网络在持续输入下能够持续跟踪对应连续ReLU1 ANN输出的理论，并基于此构建了LIF-Stream框架，实现对无限长数据流的高效、低时钟步数推理。

**💡 创新点**

核心创新在于揭示SNN瞬时动力学与ANN参考模型的跟踪关系、证明LIF神经元在离散时间下的持续跟踪特性，并利用两时标控制与增量计算大幅降低每帧的时钟步数，突破传统转换/直接训练方法在流式推理中的能效与精度瓶颈。

**🔧 技术方法**

采用ReLU1 ANN训练、IF/LIF离散时间动力学、EMA/Rate解码、软重置、两时标控制、增量激活更新以及对齐的软量化（EMA）等技术；同时使用理论证明与仿真实验验证。

**📊 数据集**

在三种模态上评估：Argoverse‑HD（视觉对象检测）、LibriSpeech（语音识别）和DVS‑Gesture（事件流手势识别），并在MOT17视频中检验行人检测。

**📈 对比分析**

与现有ANN‑to‑SNN转换（AdaFire、QCFS、SlipReLU等）及直接训练SNN（OTTT、e‑prop等）进行对比；LIF‑Stream在低时钟步数（T<8）时保持与转换方法相当或更优的准确率，同时在高时钟步数下实现最高的峰值准确率；平均时钟步数和能耗均显著低于同类方法。

**⚠️ 局限性**

局限包括：对硬件平台的实际功耗评估尚未完成，方法主要针对前馈ReLU1网络且依赖LIF神经元，复杂序列建模（如长短期记忆或Transformer）时可能需要进一步改造；此外，极端输入突变仍可能导致暂时的误差飙升。

---

## 19. Tight Logarithmic Separations between LZ77 and LZRR

**arXiv ID:** 2611.27213 | [PDF](https://arxiv.org/pdf/2611.27213v1)

**作者:** Yuto Fujie `[一作]` `[通讯]` (Kyushu University), Yuto Fujie (Kyushu University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了研究问题并构造了候选下界；利用ChatGPT和Claude协助完成主要结果的证明和论文撰写；

**💡 创新点**

通过人工智能辅助证明理论结论，展示了机器学习工具在纯理论研究中的可行性；

**🔧 技术方法**

主要技术为数学证明方法、自动化推理工具（ChatGPT、Claude）以及图表绘制工具；

**📊 数据集**

未使用任何传统数据集，论文属于理论计算机科学范畴；

**📈 对比分析**

未与传统实验方法或数据集进行性能比较，主要是理论证明；

**⚠️ 局限性**

局限在缺乏实证验证，且主要依赖人工智能生成的证明，需进一步人类验证与实验支持。

---

## 20. A Plan or an Adversarial Example? A Round-Trip Check for Latent World Models

**arXiv ID:** 2611.27124 | [PDF](https://arxiv.org/pdf/2611.27124v1)

**作者:** Huiyuan Tian `[一作]` (Zhejiang University), Shijian Li `[通讯]` (Zhejiang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究潜在世界模型下的规划过程，发现更长的搜索往往导致对抗性计划（即模型预测成功但实际执行失败），并提出检测与修复方法。

**💡 创新点**

①提出“单次搜索后执行”检测，量化规划器的对抗率；②设计基于逆动力学的Round‑Trip检查，只读取每个块的动作和，利用残差惩罚对抗计划；③在不重训练模型的情况下显著提升规划成功率。

**🔧 技术方法**

使用跨熵方法（CEM）与模型预测路径积分（MPPI）规划器；训练一个小型MLP逆动力学网络读取块和；将残差加入规划成本；对比动作裁剪、早停、集成等基线。

**📊 数据集**

公开的DINO‑WM、LeWM、JEPA‑WM世界模型，包含Wall、Cube、TwoRoom、PushT、PointMaze、Reacher等任务；每个任务约148–200个问题。

**📈 对比分析**

在同一问题集上执行计划后每次迭代并比较成功率和真实成本；相较于原始CEM/MPPI，Round‑Trip检查将Wall CEM成功率从0.39提升至0.76（+95%），Cube从0.64提升至0.79；在JEPA‑WM Wall上MPPI成功率从0.56提升至0.94；对抗率显著下降至约10%。

**⚠️ 局限性**

仅在每两个状态可唯一确定动作和的模型下有效；需要额外训练逆动力学网络；对更大搜索预算或不同世界模型的适用性尚待验证；并未解决模型本身的预测误差。

---

## 21. Neural Geometry as a Learning Signal

**arXiv ID:** 2611.26832 | [PDF](https://arxiv.org/pdf/2611.26832v1)

**作者:** Eric Todd `[一作]` (Northeastern University), Atticus Geiger `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在LLM内部对几何结构进行监督，提升模型在月份算术和行内换行等任务的表现

**💡 创新点**

提出仅对内部表示几何进行监督的训练方法，能在无输出损失的情况下匹配或接近传统Fine‑Tuning的效果

**🔧 技术方法**

利用LoRA微调、Fourier特征探针、Helix结构探针以及几何损失（均值+扩散项）

**📊 数据集**

使用Llama‑3.1‑8B、FineWeb文本数据构造的月份算术和行内换行数据集

**📈 对比分析**

与基线模型及标准task‑LoRA比较，几何LoRA在月份算术任务上达73.3%准确率，行内换行平均误差为7.3字符，均显著优于基线；与task‑LoRA对比接近或略低但仍高于基线

**⚠️ 局限性**

仅补丁几何表示无法恢复性能，说明几何监督产生的超出目标表示的副作用；需手动定义理想几何，方法难以自动化

---

## 22. NavRSI: Recursive Harness Self-Improvement for Vision-and-Language Navigation

**arXiv ID:** 2611.27499 | [PDF](https://arxiv.org/pdf/2611.27499v1)

**作者:** Guangzhao Dai `[一作]` (Singapore Management University), Bin Zhu `[通讯]` (Singapore Management University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出NavRSI框架，通过递归自我改进（RSI）在闭环体验中迭代重写harness，而不是训练模型，使冻结的前沿LLM仅凭单个RGB摄像头和原始动作即可完成视语导航任务。

**💡 创新点**

将harness本身作为优化目标，利用模拟诊断与执行日志由专门的designer LLM重写harness；通过经验累积避免重复改写；实现信息隔离，保证部署时navigator无法看到诊断，且不增加工具、记忆或角色，实现跨LLM、数据集及真实机器人的一致提升。

**🔧 技术方法**

使用前沿LLM（GPT‑6‑Astra、GPT‑6‑Luna等）与Codex CLI、Claude Agent SDK等harness；递归自我改进（RSI）框架；designer agent自动生成harness代码并记录经验；闭环训练、诊断日志分析、单RGB+原始动作实现。

**📊 数据集**

R2R‑CE（val‑unseen，R2R‑CE‑100、全split）、RxR‑CE、NavRAG‑CE 以及 Unitree G1 实际机器人上的真实指令集。

**📈 对比分析**

在hold‑out、OOD 以及真实机器人上与 NavGPT‑3、MIP、训练模型和其他harness 进行对比；NavRSI 在 R2R‑CE 最高 SR 85.64%（比 NavGPT‑3 提升 4.13%），在 RxR‑CE、NavRAG‑CE 及真实机器人上保持或超过现有最优；相同 LLM 与多厂商 LLM 的性能提升最高可达 11.3% SR，且成本与 GPU 内存均显著下降。

**⚠️ 局限性**

目前仍依赖单 RGB 摄像头和原始动作，对更复杂或动态环境的适应性未知；依赖模拟诊断训练 harnesh，实际部署时缺少此信息；需要额外的 designer LLM 与经验存储，增加系统复杂度；RSI 过程需多轮迭代，时间成本不明；尚未将 RSI 扩展到模型本身或更广泛的具身任务。

---

## 23. DNN-Assisted Deterministic Policy Learning for Adaptive IRS-NOMA-OWC Networks

**arXiv ID:** 2611.26742 | [PDF](https://arxiv.org/pdf/2611.26742v1)

**作者:** Ahrar N. Hamad `[一作]` (King’s College London), Jaafar M. H. Elmirghani `[通讯]` (King’s College London)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出在动态室内 IRS‑辅助 NOMA‑OWC 系统中联合优化功率分配与 IRS 镜面方向，以在满足多用户 QoS 要求的前提下最大化系统总速率。

**💡 创新点**

创新点在于通过 IRS 调整重塑有效信道增益，从而在运行时自适应诱导 SIC 解码顺序，并将用户状态估计与资源分配拆分为两阶段的 DNN‑DRL 框架，避免了传统离散化求解带来的局部最优与计算复杂度问题。

**🔧 技术方法**

采用深度神经网络 (DNN) 对 IR 上行 RSS 指纹进行三维位置与方向估计，随后使用基于确定性策略梯度 (DDPG) 的深度强化学习对功率与 IRS 角度进行连续控制，并通过 MDP 进行训练与策略优化。

**📊 数据集**

实验使用仿真数据集：房间尺寸 5×5×3 m，8 个 LED，49 个 IRS 元素，5 个移动用户，共 10⁵ 条 DNN 训练样本（每条包含 5 个用户的位置信息、方向角与 RSS 观测）。

**📈 对比分析**

与标准 DDPG、DQN、SCA、无 IRS、TDMA 等基线对比，本文方法在总速率上接近穷举上界（仅 4.3 % 误差），比 DDPG 高 13 %，比 DQN 高 79 %；同时在公平度、QoS 满足率与鲁棒性（拥塞、功率变化、IRS 元素数量）方面均优于对照方案。

**⚠️ 局限性**

局限性包括：依赖仿真环境与先验 RSS 指纹，需在真实场景中验证泛化能力；离线训练与在线推理仍占一定计算资源；IRS 需要较大数量的可控元件，实际部署成本与能耗需进一步评估。

---

## 24. Model the Annotation, Not the Noise: Semantic Annotation-Channel Prompt Learning against Noisy Labels

**arXiv ID:** 2611.26702 | [PDF](https://arxiv.org/pdf/2611.26702v1)

**作者:** Zibin Geng `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Min Liu `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种语义感知注释通道（SACP）提示学习框架，用于在少样本、噪声标签条件下高效适配预训练视觉‑语言模型；

**💡 创新点**

将噪声标签视为由潜在干净标签生成的观察，构建结构化注释通道与语义先验结合，使用通用期望最大化估计清晰标签后验，并引入后验感知记忆来提升推理稳健性；

**🔧 技术方法**

冻结预训练视觉与文本编码器、可学习提示词、注释通道参数、通用期望最大化（GEM）以及后验感知记忆等技术；

**📊 数据集**

在五个合成噪声基准（Caltech101、DTD、Flowers102、OxfordPets、UCF101）和真实噪声Food101N数据集上进行实验；

**📈 对比分析**

与CoOp、CoOp+GCE、JoAPR、NLPrompt、NA‑MVP等方法对比，SACP在对称和不对称噪声下平均提升约2%–4%，在高噪声和少样本场景表现最优；

**⚠️ 局限性**

依赖预训练VLM的语义先验可能带来偏见；在极端噪声或语义关联度低的任务中效果可能受限。

---

## 25. Graph-XGB: A Hybrid GNN Framework for Outdoor-to-Indoor 5G Signal Reconstruction via Aerial-Only UAV RF Sensing

**arXiv ID:** 2611.26828 | [PDF](https://arxiv.org/pdf/2611.26828v1)

**作者:** David Hason Rudd `[一作]` (University of Technology Sydney), Xiangmeng Wang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过将无人机外部RF测量与建筑内部信号状态映射，提出一种可在不收集内部测量的前提下重建三维室内信号强度的方法

**💡 创新点**

将图神经网络（SpiderGNN）作为仅用于拓扑编码的特征提取器，随后使用主成分分析降维并以目标专属XGBoost回归器进行非线性信号预测，解决了端到端GNN在路径损耗与干扰指标之间的优化冲突

**🔧 技术方法**

改进的SpiderGNN（GATv2）、Floor‑Aware 3D 距离度量、Bridge Edge Guarantee、PCA降维、目标特定XGBoost回归、九重随机种子元集成

**📊 数据集**

真实5G NR测量数据集，包含613个节点（10层楼宇），外部节点229个、内部节点384个，采用无人机航拍RF扫描与室内TEMS走测数据进行标注

**📈 对比分析**

与SimpleGNN、端到端SpiderGNN、Frozen Trunk、Hybrid GNN‑XGBoost PCA（v7）、Cascade XGBoost（v9）等基线和消融模型比较，Graph‑XGB（v8）在RSSI、RSRP的R²分别为0.759/0.770，比SimpleGNN提升35.5%/64.9%，比端到端SpiderGNN提升71%/>100%，在四项指标中均获得最高R²或最低RMSE

**⚠️ 局限性**

干扰相关指标（RSRQ、SNR、CQI）在纯空间推断下只能达到R²≤0.27，模型无法突破“信号类型天花板”；实验仅基于单栋楼宇与5G NR数据，缺乏跨建筑、跨频段的泛化验证

---

## 26. Federated KPI Aggregation with Local Privacy Constraints and Coordinated OLAP Roll-Ups

**arXiv ID:** 2611.26709 | [PDF](https://arxiv.org/pdf/2611.26709v1)

**作者:** Felix Härer `[一作]` (University of Applied Sciences and Arts Northwestern Switzerland), Felix Härer `[通讯]` (University of Applied Sciences and Arts Northwestern Switzerland)

**通讯引用:** 407 | [OpenAlex ID](https://openalex.org/A5031567650)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在联邦KPI聚合中，如何通过协调OLAP层次结构，满足各站点的最小队列阈值，从而避免独立抑制导致信息缺失的情况。

**💡 创新点**

提出了最小通用roll‑up选择方法，按信息损失最小化并保证所有站点安全；并设计了基于多代理的协调架构，保持局部能力、授权与执行在本地，支持能力变化。

**🔧 技术方法**

使用OLAP层次滚动聚合、最小队列阈值规则、信息损失度量；实现采用Python、SQLite星型模式，并借助Hermes+Buzz代理框架进行多代理通信。

**📊 数据集**

在四个站点的合成数据上进行实验，每站约5,000条事实记录、450个客户，数据通过随机种子生成不同的局部稀疏模式。

**📈 对比分析**

与独立抑制方法对比；在k=20时独立抑制覆盖率仅78.3%，误差21.6%；协调roll‑up保持100%覆盖、误差0；候选检查数平均为5.6（k=3）至7.0（k=20）；能力变更时从5候选/20回复降至2候选/8回复，且未出现不支持请求。

**⚠️ 局限性**

仅使用合成数据；只考虑事实行作为隐私单位；未在大规模真实数据上验证；未整合差分隐私或多次查询控制；假设所有站点能执行确定性函数，实际部署中可能存在差异。

---

## 27. NitroGen: Observations on the Action Dynamics, Visual Instruction Following, and Spatial Exploration

**arXiv ID:** 2611.26807 | [PDF](https://arxiv.org/pdf/2611.26807v1)

**作者:** Shengyuan Lin `[一作]` (Carnegie Mellon University), Ozan K. Tonguz `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了一个冻结的视觉-动作基础模型NitroGen在游戏控制中的局部适应性、视觉提示跟随性和空间探索性。

**💡 创新点**

提出了在不引入额外规划模块的前提下，通过三维度诊断评估冻结控制器的局部行为与全局游戏进展的差异，并量化了其对视觉提示和空间覆盖的弱响应。

**🔧 技术方法**

采用行为克隆训练的500M参数DiT视觉-动作架构，配合SigLIP-嵌入、PCA与k-means进行状态聚类；评估中使用了游戏帧阈值、方向切换率、跳跃频率等指标。

**📊 数据集**

使用约40,000小时、超过1,000款游戏的公开人类游戏视频进行训练；评估以Hollow Knight和Trackmania的在线游戏录像为实验环境。

**📈 对比分析**

通过对比Hollow Knight不同情景下的动作分布、不同视觉提示（箭头、文字）下的左右移动比例以及与随机与人类基线的聚类覆盖与熵值，发现NitroGen在局部情境下能改变动作分布但未能完成跳跃障碍，视觉提示影响有限，空间覆盖低于随机且明显低于人类。

**⚠️ 局限性**

局限性包括：冻结控制器缺乏长时规划与记忆，视觉提示效果环境依赖且不可靠；空间探索测量受嵌入与聚类方法影响；实验仅覆盖少数游戏与提示形式，未覆盖更广泛场景。

---

## 28. FORTH: Grounding Tactile VLA Execution via Forward-Only Proxy-Control Pairing

**arXiv ID:** 2611.26744 | [PDF](https://arxiv.org/pdf/2611.26744v1)

**作者:** Junde Li `[一作]` (University of Macau), Pui-In Mak `[通讯]` (University of Macau)

**通讯引用:** 12529 | [OpenAlex ID](https://openalex.org/A5058845450)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出FORTH框架，结合前向仅学习、零阶适配器与SlipGuard滑动检测，实现视觉-语言-动作管线中的可测量、可纠正触觉控制。

**💡 创新点**

创新点在于配对指标评估协议，将代理损失与对应物理量同步；实现层级前向仅表示树、令牌并行零阶适配与连续滑动门SlipGuard，实现低内存、高精度的物理控制。

**🔧 技术方法**

采用前向-前向（Forward‑Forward）层级特征树、令牌并行零阶（Zeroth‑Order）适配器、Shear‑Ratio一类滑动门SlipGuard，以及4‑bit冻结视觉后端的前向仅更新技术。

**📊 数据集**

在LIBERO三套任务（Octo、Diffusion、OpenVLA等）以及自制弹性壳体、金属、塑料、橡胶等滑动实验的多物体/摩擦实验中进行评估。

**📈 对比分析**

与基线反射递增、常数力策略及其他VLA模型对比，FORTH在LIBERO任务中实现97.6%平均成功率，Peak Force降低29×，内存峰值降低7.12×，并在10,200+跨域测试中保持100%成功率。

**⚠️ 局限性**

仍需在真实传感器校准下依赖单一阈值，未充分验证极端动态扰动与非标准触觉传感器；代理‑物理配对的理论分析仍有限。

---

## 29. Incentive Compatible Mechanisms for Sequential Fair Resource Allocation with Private Demands

**arXiv ID:** 2611.26800 | [PDF](https://arxiv.org/pdf/2611.26800v1)

**作者:** Ramtin Salamat `[一作]` (Northwestern University), Seyed M. R. Iravani `[通讯]`

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在资源稀缺、可持续分配的在线场景下，研究了代理人私有需求且可能战略性夸大需求的情形，提出并分析了在无金钱转移条件下的激励相容分配机制，目标是最大化顺序公平度量（最小填充率）。

**💡 创新点**

① 对激励相容规则给出了完整的结构表述，揭示其可通过“可达集合”来描述；② 证明在过度处罚成本足够小的情况下，最优机制可简化为只设置一个上限（cap）机制；③ 若不满足此条件，构造了离散化的动态规划并将其化为有向无环图中的最长路径问题，从而得到多项式时间的近似算法；④ 通过数值实验量化了忽略激励导致的损失以及激励相容设计所能恢复的收益。

**🔧 技术方法**

机制设计理论（单峰效用、IC规则结构），动态规划，离散化和图论（最长路径）算法，数值仿真（Monte Carlo / 取样积分）。

**📊 数据集**

使用合成需求分布（均匀窗口、单峰 Beta 混合、双峰 Beta 混合），通过多组均值与 CV 参数生成 1440 个实验情形；没有使用公开真实数据集。

**📈 对比分析**

将最优激励相容机制与两种基准（假设真诚的第一最佳规则和天真实施第一最佳规则导致的策略误报）进行对比。实验显示：在资源紧张到适中之间，忽略激励的成本最高；激励相容设计可恢复约 7–12% 的损失，且收益随过度罚金增大、需求变异度降低而提升。

**⚠️ 局限性**

① 仅考虑固定顺序到达的代理人，未结合路线/排队优化；② 假设需求分布已知且满足连续密度，且效用为单峰形式；③ 近似结果依赖离散化精度，未给出全局最优证明；④ 只针对完全战略代理人，未考虑混合真诚/战略人群。

---

## 30. Verification of Statistical Properties: Redefining the Possible

**arXiv ID:** 2611.26951 | [PDF](https://arxiv.org/pdf/2611.26951v1)

**作者:** Clément Canonne `[一作]` (University of Sydney), Qiping Yang `[通讯]`

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文重新定义了交互式证明系统（IP）在分布测试中的应用，提出了多证明者（MIP）版本和零知识（ZK）版本，并专注于“farness”任务（即判断两分布或分布与均匀分布相距是否大）。

**💡 创新点**

创新点包括：① 通过将传统的相似度测试（closeness）转化为“远离”任务，证明常数样本即可实现交互式验证；② 引入多证明者框架和新的零知识定义，展示多证明者并不打破已有的√n样本下界；③ 在黑盒采样模型下证明了熵差到统计差的降维不可行；④ 结合半鲁棒学习方法，为一系列可学习的分布属性（如单调、对数凸、分段多项式分布等）构造了零知识验证协议。

**🔧 技术方法**

核心技术包括 Polarization Lemma、Scheffé 集、单样本猜测协议、重复理论（顺序/并行），以及学习框架（semi‑agnostic 学习器）来构造 farness 验证协议。通过模拟论证实现统计零知识与完美 HIPO‑ZK。

**📊 数据集**

工作纯理论化，没有使用具体的真实数据集，全部在离散分布模型上进行抽象分析。

**📈 对比分析**

协议样本复杂度为 O(1)（对单/多证明者均如此），通信复杂度为 O(log |Σ|) 位，完备性 ≥ 2/3，误差 ≤ 1/3，可通过重复放大至任意 ε；相比传统无证明器的分布测试需要 Θ(√n) 或更高样本，显著提升效率。

**⚠️ 局限性**

局限性：① 仅适用于“远离”任务，无法同样处理相似度（closeness）或属性成员资格验证；② 零知识结果目前仅在诚实验证者模型下成立，恶意验证者下仍未知；③ 多证明者并未突破 √n 下界，仍受原始单证明者限制；④ 某些零知识协议（如 farness 的 SZK）仍为开放问题；⑤ 仅给出了理论复杂度，没有实验验证。

---

## 31. A Framework for AI-Ready Data in Earth and Environmental Science

**arXiv ID:** 2611.27313 | [PDF](https://arxiv.org/pdf/2611.27313v1)

**作者:** Tobias Weigel `[一作]`, Paula Stock `[通讯]`

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并验证了一个面向 AI 的数据就绪评估框架，涵盖数据准备、质量、文档、访问以及监督学习目标等 25 条标准。

**💡 创新点**

创新点在于：① 将 FAIR 原则与机器学习需求系统融合，形成可操作的多利益相关者共识框架；② 首次将监督学习标签（target）纳入数据就绪标准，强调其文档化与质量控制；③ 以案例评估展示框架在不同地球与环境科学任务中的适用性。

**🔧 技术方法**

采用结构化社区驱动方法：文献综述、现有框架梳理、对 10 名 ML 科研人员的问卷调查、专家讨论与共识形成，并以 ESIP、FAIR 等标准为映射基础。

**📊 数据集**

未针对单一数据集做实验，而是在若干代表性案例（环境传感器 QC、全球天气预报、海洋生物群落建模、沉降洞穴图像、浮游生物图像、地下水监测）中应用框架进行评估。

**📈 对比分析**

通过案例评估对每项标准的重要性评级（essential、important、nice to have、irrelevant），并未给出量化性能指标；框架主要用于定性评估，性能未量化比较。

**⚠️ 局限性**

局限性包括：① 调查样本有限、仅聚焦地球与环境科学；② 未提供自动化工具或认证流程，需自行实现；③ AI 就绪度高度依赖具体用例，无法“一刀切”；④ 缺乏跨学科验证与成本实现可行性讨论。

---

## 32. Specification-Grounded Adaptive Triage and Formal Invariant Exclusion Synthesis for Autonomous Silicon Coverage Closure

**arXiv ID:** 2611.26815 | [PDF](https://arxiv.org/pdf/2611.26815v1)

**作者:** Veni Mohan `[一作]` `[通讯]` (Google LLC), Veni Mohan (Google LLC)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一套名为ARGUS的全自动覆盖闭合引擎，能从RTL覆盖报告中识别并生成针对剩余覆盖点的测试序列与正式排除文件，实现长尾覆盖闭合。

**💡 创新点**

创新点包括：① 基于微架构规范与SVA的块合同自动抽取；② 双模式分诊（宏聚类+微手术）根据阈值动态切换；③ 将Oracle约束与正式不等式证明相结合，生成安全的刺激与排除；④ 三步UVM门控验证流程，保证零“脏合并”。

**🔧 技术方法**

采用的技术有：大语言模型（LLM）与Prompt，但被AST分析、SVA提取、Formal invariant证明、Token压缩与阈值切换所驱动；同时使用Verilator、Synopsys VCS/URG工具链进行覆盖收集与验证。

**📊 数据集**

使用的数据集包括：三个OpenTitan IP（Baud、I2C、AES）、两个AI加速器Tensor Layout Converter子系统，总计53个SystemVerilog模块、376,127覆盖点，另外还有2,626模块的大规模工业互连子系统。

**📈 对比分析**

与平面LLM循环、LLM4DV、AutoSVA、RTLFixer、JasperGold等基线比较，ARGUS在Token使用上下降27.2×–95.7×，覆盖闭合率提升至99.32%–100%，零错误排除，并在2,626模块子系统上实现0.265秒的三分分识别，完成全覆盖闭合。

**⚠️ 局限性**

局限性包括：需要手工或规则化提取规范与SVA；阈值τ需针对不同设计调优；LLM的生成质量仍可能影响刺激正确性；对极端参数化或深度循环结构的可达性推理可能需要进一步提升。

---

## 33. RoboPRO: A Perturbation-Resilient and Obstacle-Aware Benchmark for Robotic Manipulation

**arXiv ID:** 2611.26971 | [PDF](https://arxiv.org/pdf/2611.26971v1)

**作者:** Zhiyuan Li `[一作]` (Huawei Technologies), Amir Rasouli `[通讯]` (Huawei Technologies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 RoboPRO 基准，包含 80 个双臂原子与组合任务，系统化的视觉、语言与对象扰动，基于路径扫掠的障碍生成，以及两种基于碰撞严重程度的安全指标。

**💡 创新点**

创新点在于：① 将操作安全性与扰动鲁棒性统一到同一套实验环境；② 采用扫掠式障碍放置方法和两级碰撞度量；③ 对视觉、语言、对象三个扰动轴分别进行逐因子分析，揭示各类扰动对性能的细粒度影响。

**🔧 技术方法**

技术手段包括：SAPIEN 模拟环境搭配 Aloha-Agilex 双臂机器人；cuRobo 轨迹优化采集演示；三种 VLA 策略（π0、π0.5、X‑VLA）；多维扰动设计（像素平移、模糊、灯光、材质变换、语言多样化）；以及基于成功率、碰撞率和硬成功率的评估框架。

**📊 数据集**

数据集由 16,000 条演示轨迹组成（每个任务 100 条干净 + 100 条杂乱），配合 5.6k 语言指令，20 条同义指令用于训练，另外 4,000 条评估指令用于语言扰动；覆盖 4 个室内场景、29 种可操纵物体和 16 件家具。

**📈 对比分析**

采用 SR、CR、HSR 三指标对三模型进行对比；在干净场景下 SR 分别为 0.56–0.74，杂乱场景下下降 0.10–0.13；π0.5 在安全性与成功率上表现最佳；混合训练（清洁+杂乱）显著提升整体性能；扰动对性能的影响小于障碍杂乱，且语言扰动导致的下降最大。

**⚠️ 局限性**

局限性包括：仅在仿真中评估，缺乏真实机器人实验验证；扰动轴覆盖有限（未包含硬件噪声、传感器失真等）；对长时序组合任务的鲁棒性尚未彻底验证；跨域迁移到不同机器人/环境的通用性未知。

---

## 34. Railroads as Cyber-Physical Systems: Cybersecurity Implications of Increasing Automation -- Threats, Consequences, and Operational Resilience

**arXiv ID:** 2611.27088 | [PDF](https://arxiv.org/pdf/2611.27088v1)

**作者:** Syed Owais Athar `[一作]` (University of Nebraska-Lincoln), Muhammad Naveed Aman `[通讯]` (University of Nebraska-Lincoln)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对铁路自动化网络化导致的安全风险进行叙事综述，构建威胁面分类，并归纳公开事件案例，提出韧性框架。

**💡 创新点**

提出铁路自动化特定的威胁面分类，将自动化功能与攻击路径、后果及韧性需求关联，强调将网络安全视为运营连续性问题。

**🔧 技术方法**

采用叙事综述与威胁面合成方法，整合学术文献、监管文件与公开事件，构建系统模型。

**📊 数据集**

未使用正式数据集，仅利用公开的铁路安全事件报告和技术文献。

**📈 对比分析**

无实验比较与性能评估，主要通过案例分析和逻辑推理阐述风险与对策。

**⚠️ 局限性**

研究为叙事综述，缺乏系统性事件频率估计；案例报告不完整，适用性受限于北美/欧洲；未提供实证验证。

---

## 35. HumanoidPose: Monocular 3D Pose Estimation for Humanoid Robots

**arXiv ID:** 2611.27181 | [PDF](https://arxiv.org/pdf/2611.27181v1)

**作者:** Jinlin Gan `[一作]` (Xiamen University), Cheng Wang `[通讯]` (Xiamen University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了人形机器人单目三维姿态估计任务，并构建了Humanoids数据集与基准以及HumanoidPose框架。

**💡 创新点**

创新点在于将人类姿态估计与机器人姿态估计相结合，提出机器人本地化关节预测和显式运动学校准，并为13种人形机器人提供统一的3D标注与评估。

**🔧 技术方法**

采用视觉前端检测与2D关键点估计、冻结的V‑JEPA编码器、跨注意力解码器、根部姿态分支、运动学正向传播与残差校正等技术。

**📊 数据集**

使用Synthetic 732K帧RGB与FK生成的3D标注、12种机器人模型的真实2D手动标注，以及Roban的真实3D配准数据。

**📈 对比分析**

与现有HPE与RPE方法在Humanoids、Humanoids‑Real和DHRP‑H1上进行比较，HumanoidPose在关节误差、3D姿态误差与图像空间精度上均领先，尤其在Roban真实3D评估中达到13.65° MPJAE、42.82mm MPJPE等最佳指标。

**⚠️ 局限性**

局限包括依赖预训练的2D关键点检测对极端遮挡敏感，未针对多机器人协作场景的实时性与跨场景泛化进行深入验证。

---

## 36. Exploring Early Vulnerability Analysis via Automatic Identification of Vulnerability-Introducing Commits

**arXiv ID:** 2611.26739 | [PDF](https://arxiv.org/pdf/2611.26739v1)

**作者:** Weiliang Qi `[一作]` (University of Texas at Dallas), Xinda Wang `[通讯]` (University of Texas at Dallas)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5102758573)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了基于数据状态感知的 D‑SZZ 算法，自动识别漏洞引入提交并创建三大手工标注的 VIC 数据集。

**💡 创新点**

创新点在于结合安全修复模式和漏洞范围切片，利用 CPG 进行语义定位，突破传统 SZZ 仅追踪删除行的局限，并提供高质量的早期漏洞样本。

**🔧 技术方法**

采用代码属性图（CPG）构建、语义/数据流切片、基于安全修复模式的匹配，以及改进的 SZZ 追踪逻辑；实现为 Python 框架，兼容树解析器 tree‑sitter。

**📊 数据集**

使用 NVD‑new（557 条 VIC）、tcpdump（101 条）和 Vim（125 条）三大手工标注数据集，并在 Linux 项目上做额外验证。

**📈 对比分析**

与六种 SZZ/LLM 基线（SZZ、V‑SZZ、SEM‑SZZ、LLM4SZZ、Vercation、LLMBisect）对比，D‑SZZ 在 Recall 上提升 13.6%–8.9%，F1 上提升 6.7%–11.9%；在漏洞版本识别任务中 F1 达 94% 以上；在早期漏洞检测模型训练中加入早期样本显著提升 Recall，精度略有下降。

**⚠️ 局限性**

仅支持 C/C++，依赖 Git 版本控制；对 VFC 标注质量敏感；对 LLM 基线的上下文长度有限制；未覆盖多语言或非 Git 项目。

---

## 37. TOUMAII: A 4.5M-Pair Tool-Calling Preference Dataset and Its Source-Agnostic Generator

**arXiv ID:** 2611.26722 | [PDF](https://arxiv.org/pdf/2611.26722v1)

**作者:** Moritz Pötzsch `[一作]` (Technical University of Munich), Swanand Ravindra Kadhe `[通讯]` (IBM Research)

**通讯引用:** 2335 | [OpenAlex ID](https://openalex.org/A5053049612)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出TOUMAII工具调用偏好数据集和TOUMAII-Gen生成管道，生成约450万对偏好样本，用于直接偏好优化(DPO)。

**💡 创新点**

创新点在于将SFT工具调用数据转化为对比式偏好数据，并构建多轮、多调用、并行与顺序交互四类交互结构，首次规模化公开工具调用偏好数据集。

**🔧 技术方法**

使用DPO、SFT多损失联合训练、Granite、Qwen、Gemma等小型语言模型生成误差样本，采用ToolRL奖励函数评估格式与正确性，并通过后处理筛选差异显著样本。

**📊 数据集**

主要数据集：Toucan（1.5M行）生成TOUMAII（≈4.5M对），并将管道应用于ToolACE得到TOUMAII-ACE（≈100k对）。

**📈 对比分析**

在BFCL v4基准上，使用相同基模与训练设置，TOUMAII在多轮、记忆等场景提升准确率：整体34.68%（比ToolRM高6.35pp），实时准确率77.55%（比ToolRM高2.35pp）。

**⚠️ 局限性**

缺乏无关样本；生成时禁用推理，验证仅对照参考不保证执行正确；数据继承源数据与生成模型的局限性。

---

## 38. Comparing Visual Motion Patterns for Pedestrian Guidance with Multi-Display Signage: Two Field Studies

**arXiv ID:** 2611.26984 | [PDF](https://arxiv.org/pdf/2611.26984v1)

**作者:** Yuri Mikawa `[一作]` (University of Tokyo), Kazushi Maruya `[通讯]` (NTT, Inc.)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在两项博物馆现场实验中，研究者通过多显示器（MDS）展示不同视觉运动模式（全屏条纹、圆孔条纹、部分条纹、广告仅显示、文本符号指示），并使用 LiDAR 记录行人位置、通过量表评估主观体验，以探讨视觉运动如何影响行人侧向分布及通过侧选择。

**💡 创新点**

创新点在于：① 将可运动条纹与广告内容结合，探讨低对比度、有限覆盖度的视觉运动在保持广告功能的同时引导行人的潜力；② 在自然公共空间中进行大规模实地试验，使用 LiDAR 数据而非传统摄像监测；③ 系统评估不同环境配置（单向 vs 双向通行、窗口距离、显示间距）对视觉运动效果的影响。

**🔧 技术方法**

技术方法包括：
- 视觉刺激设计：全屏、圆孔、部分条纹；
- LiDAR（Hesai JT128）点云采集与聚类定位，基于帧级聚类、方向判定、10 秒窗口聚合；
- 问卷调查（Typeform），涵盖运动意识、影响、舒适度等维度；
- 统计分析：方差分析、线性混合模型（LMM）、广义线性混合模型（GLMM）等。

**📊 数据集**

数据集：共 18,402 次行人通过，覆盖 25 个实验日，包含 LiDAR 点云（经裁剪后约 2.0 TB 数据）、问卷响应（共约 249 份），并对行人分布与主观体验进行量化记录。

**📈 对比分析**

比较方法：通过 10 秒窗口的中位数位置、方差分析、LMM 估计显著性；对比全屏 vs 部分条纹、圆孔条纹；比较不同对比度水平；与广告仅显示（AD）及文本符号（Sign）做差异检验。结果显示：
- 部分条纹在单向流中略微增强侧向偏向（效果量小，显著性在部分场景下出现）；
- 对比度降低未显著改变侧向偏向；
- 在双向流中，条纹方向确实影响侧向分布，但与广告仅显示无显著差异；
- 通过侧选择无明显指引作用；
- 主观舒适度（症状）在部分条纹下显著降低。

**⚠️ 局限性**

局限性：
- TC1 每个条件仅测一次，导致日间差异与条件混淆；
- 缺乏静态或无运动基线控制；
- 双向流中通过事件稀少，统计功效有限；
- 10 秒窗口聚合无法追踪个体轨迹，可能掩盖细节效应；
- 受实验场地、照明、窗口位置等多重上下文变化影响，难以单独归因；
- 大规模 LiDAR 数据未公开，重复验证受限。

---

## 39. MODULE: A Mutual-Promoting Deep Unfolding Framework Towards Degradation-Robust Multi-modal Image Fusion

**arXiv ID:** 2611.27269 | [PDF](https://arxiv.org/pdf/2611.27269v1)

**作者:** Yunfei Deng `[一作]` (Southeast University), Guangcan Liu `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `da1b1a89-583a-4b57-9c81-478778569bec` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于深度展开的多模态图像融合框架 MODULE，联合实现降噪/去模糊恢复与融合，兼顾不同模态的互补信息。

**💡 创新点**

核心创新在于把降解鲁棒融合建模为联合优化问题，采用半二次分裂将其拆解为交替子问题，形成互促的双向信息流，并将每个子问题展开为可解释的网络模块，融合模型基础与数据驱动的优点。

**🔧 技术方法**

使用深度展开、半二次分裂(HQS)、Prompt‑driven U‑Net、Restormer、NAFBlock、残差 SE‑Block 等网络结构；引入伪逆映射、生成投影约束以及互促正则化。

**📊 数据集**

在 LLVIP、MSRS 以及 M3FD 等数据集上进行训练与测试，涵盖多种噪声、低光、模糊等退化。

**📈 对比分析**

与 DDFM、CrossFuse、EMMA、TC‑MoA、DDBFusion、Text‑IF、DRMF、OmniFuse、TG‑ECNet 等方法在 SCD、VIF、Q_abf、SSIM 等指标上进行对比，实验表明 MODULE 在大多数指标上取得或逼近最优表现，尤其在低光/噪声/模糊等复杂退化场景中优于竞争方法。

**⚠️ 局限性**

仍存在对训练范围之外极端退化的适应性不足；网络虽相对轻量但在极低延迟或资源受限环境下仍需进一步优化；在某些极端复合退化下细节恢复仍略显不足。

---

## 40. Cross-Linguistic Distinctive-Feature Constraints on Segmental Triphones: Evidence for Contrastive Dispersion

**arXiv ID:** 2611.26837 | [PDF](https://arxiv.org/pdf/2611.26837v1)

**作者:** Leonardo Araújo `[一作]` `[通讯]` (Universidade Federal de São João del-Rei), Leonardo Araújo (Universidade Federal de São João del-Rei)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对17种语言词内三元组进行特征距离分析，检验连续语音在跨语言中是否呈现中等距离模式、期望距离提升、离散度缩小以及联合熵下降等特征；

**💡 创新点**

引入两层置换基线（无约束置换与C/V模板保持置换），通过互信息分解揭示子音节层的序列耦合，并在跨九族语料上验证这些约束的普适性；

**🔧 技术方法**

使用PanPhon的24维三元特征向量计算Hamming与归一化余弦距离，结合词内置换与C/V模板保持置换，采用联合熵、互信息、Wilcoxon等统计检验方法；

**📊 数据集**

17个Universal Dependencies v2 treebank，涵盖九大语族，使用Epitran进行统一IPA转写；

**📈 对比分析**

与无约束置换和C/V模板保持置换进行对比，发现自然序列期望距离显著更高（+0.71~+1.66特征），离散度显著更低（-0.23~-1.01特征），联合熵下降（-0.50~-1.01比特），互信息显著提升（+0.007~+0.101比特），p<10^-4；

**⚠️ 局限性**

样本大小不均，纯文本语料限制跨词边界分析，依赖G2P转写可能引入错误，特征距离为间接代理，未直接测量物理或听觉距离；

---

## 41. AceVLN: Asynchronous Cloud Edge Deployment of VLNs via Latent Scheduling and Compensation

**arXiv ID:** 2611.27255 | [PDF](https://arxiv.org/pdf/2611.27255v1)

**作者:** Zihao Zheng `[一作]` (Peking University), Xiang Chen `[通讯]` (Peking University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为AceVLN的训练自由框架，用于云–边缘协同的异步视觉语言导航，通过主动刷新和补偿来减少潜在的潜在码差导致的导航误差。

**💡 创新点**

创新点包括：1）对潜在码的关键子空间进行分解，捕捉对当前动作最敏感的方向；2）利用关键漂移熵自适应地决定刷新阈值；3）在延迟峰值期间使用漂移连续性指导边缘补偿，从而在不频繁请求云端的情况下保持导航精度。

**🔧 技术方法**

采用了子空间分解、特征漂移跟踪、熵计算、漂移连续性度量、阈值调度、边缘补偿机制等技术，并基于现有的异步双系统架构实现。

**📊 数据集**

在R2R‑CE和English RxR‑CE两大公开数据集上进行评估。

**📈 对比分析**

与周期性刷新、TUD、Bridge、CloudEdge等基线方法比较，AceVLN在成功率提升1.7–2.4个百分点、SPL提升1.7–2.7个百分点的同时，将 FLOPs 降低18–22%，并将平均计算量降低2.7–10%；仅略微增加5–13 ms的响应时延。

**⚠️ 局限性**

局限性：仍会在极端网络延迟峰值下出现响应时延上升；需要在部署前对关键子空间进行离线校准；在极低资源或高动态场景中可能需要进一步优化刷新阈值和补偿策略。

---

## 42. BioCL-POS: Biological Completer-aware Learning with Predictive Observation Selection - A Wild Ginseng Study

**arXiv ID:** 2611.27316 | [PDF](https://arxiv.org/pdf/2611.27316v1)

**作者:** Yulu Wang `[一作]` (Zhejiang University), Taolue Liu `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

提出 BioCL-POS，通过预测固定完成器对不同观测子集的空间响应，挑选最优子集以提升缺失根部的点云重建质量；

**💡 创新点**

创新点在于在不改动预训练完成器的前提下，用空间响应预测来指导输入选择，实现了“观察选择+单次完成”的高效方案；

**🔧 技术方法**

使用点云对齐与交叉注意力网络预测空间响应、质量回归和排序损失，结合基于组的候选构造；

**📊 数据集**

构建并公开野生人参和合成树木的控制缺口数据集，分别包含 728 份人参扫描和 160 棵树木模型；

**📈 对比分析**

与 Full、FPS、SampleNet、LTA-PCS、SAMPLE 等采样器以及四个冻结完成器（CRA-PCN、AdaPoinTr、SeedFormer、SnowflakeNet）进行对比，BioCL-POS 在 5–20% 缺失率下平均降低 CD‑L1 8–23%，提升 F‑score 与覆盖率，且能在多数场景实现“保护性改进”；

**⚠️ 局限性**

局限在于候选构造固定为局部点对切换，查询位置与候选数不自适应；对不同完成器的泛化需单独训练；在 30% 缺失率下仍无法总是优于 Full。

---

## 43. Toward a digital twin of the ICU patient: Forecasting and treatment response in a generative event model

**arXiv ID:** 2611.27028 | [PDF](https://arxiv.org/pdf/2611.27028v1)

**作者:** Zhuoyang Lyu `[一作]` (Harvard Medical School), Li Zhou `[通讯]` (Harvard Medical School)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

训练并评估了一个 79M 参数的解码器 Transformer（生成事件模型），在 MGB ICU 电子健康记录上进行无任务特定训练，随后在 19 个临床预测任务、生成记录逼真度、个体轨迹以及干预的逆因果验证等多维度上进行评估。

**💡 创新点**

通过无监督的生成模型实现了与强监督模型相当的 ICU 事件预测，展示了利用模拟未来读取概率来构建数字孪生的可行性；并首次引入了控制实验（药物副作用、无效药物、对立药物）来检验模型是否真正学会药物效应而非仅仅捕捉指示偏倚。

**🔧 技术方法**

使用解码器‑only Transformer（RMSNorm、SwiGLU、rotary position embeddings、grouped‑query attention）搭建模型；采用 MEDS 事件标准进行 tokenization；通过采样 50 条未来序列进行零-shot 预测；对干预采用 Monte‑Carlo g‑computation 进行模拟；与 LightGBM 等监督基准进行对比。

**📊 数据集**

基于 Mass General Brigham（MGB）学术医疗系统的 ICU 电子健康记录，包含 127,332 训练患者、1.57 亿训练 token，共计 236,906 ICU 住院事件；数据按 90/5/5 划分为训练/验证/测试集。

**📈 对比分析**

与 144 个监督配置（最佳为 LightGBM）在 19 个测试任务上对比；模型平均 AUROC 0.8362，几乎与最强监督模型 0.8345 相当；在新插管、低血压、影像下达等在一天内改变护理的事件上表现领先；生成记录与真实记录匹配度高，个体轨迹风险随事件临近上升；干预验证显示模型能捕捉到药物副作用，但对目标效应的对立药物效果未如预期。

**⚠️ 局限性**

仅在单一医疗系统数据上训练并未外部验证；评估多依赖验证集，结果可能偏乐观；预测时间固定为 ICU 入院 24 小时；部分事件（如通气、药物）记录为订单而非实际执行，导致精度受限；数值被离散为十分位，阈值分辨率低；概率未再校准，ECE 较高；干预模拟假设无未观测混杂，实际并未完全满足。

---

## 44. WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection

**arXiv ID:** 2611.27353 | [PDF](https://arxiv.org/pdf/2611.27353v1)

**作者:** Marc Boëlle `[一作]` (Google DeepMind), Alexandre Ramé `[通讯]` (Google DeepMind)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用权重空间线性插值作为数据增强手段，训练一个LoRA形式的自我解释器（introspector），以提高在不同分布下对微调模型行为的自然语言描述能力。

**💡 创新点**

创新点在于：①将线性模式连通性（linear mode connectivity）原理引入权重插值，生成合成微调模型；②通过这些合成模型对自我解释器进行训练，显著提升鲁棒性和泛化；③提供多种目标混合策略（加权损失、文本合并描述），使模型能同时保留两者行为。

**🔧 技术方法**

核心技术包括：LoRA模块、权重插值（λ·θ_i ⊕ (1-λ)·θ_j）、自我解释器的SFT训练、Beta分布采样、LLM判定器（用于评估描述准确率）、以及对比实验的误差分析。

**📊 数据集**

使用的数据集有：隐藏行为基准（hidden-behavior benchmark）、基于Super-Natural Instructions的真实NLP任务元数据集、以及对齐审计基准AuditBench（包含56个Qwen3-14B模型）。

**📈 对比分析**

方法对比：在隐藏行为、NLP任务和AuditBench上与原始无增强训练的introspector做对比。实验显示：在所有基准上，权重插值增强后的introspector均取得更高的准确率或描述率，AuditBench上显著提升了约4.7个百分点（约31%相对提升），并在多种分布偏移（LoRA rank、学习率、训练轮次）下表现出更稳健的提升趋势。

**⚠️ 局限性**

局限性包括：①仅适用于同一基础模型和检查点；②合成模型只能在已存在行为范围内插值，无法生成全新行为或任务；③对RLHF等更大规模微调方式的适应性尚未验证；④合成目标描述仍依赖外部LLM生成，可能引入噪声；⑤在极端分布偏移（如Synthetic Documents + KTO）下提升有限。

---

## 45. A Frugal AI Framework for India: Responsible and Right-sized AI

**arXiv ID:** 2611.27189 | [PDF](https://arxiv.org/pdf/2611.27189v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 46. Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs

**arXiv ID:** 2611.26975 | [PDF](https://arxiv.org/pdf/2611.26975v1)

**作者:** Soheil Zibakhsh `[一作]` (University of California San Diego), Farinaz Koushanfar `[通讯]` (University of California San Diego)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种共享骨干的本地LLM安全检查器，在生成过程中实时检测不安全内容。

**💡 创新点**

创新点在于：1）将安全验证路径作为生成模型的高层微调，使用低秩适配器；2）利用共享权重与 KV 缓存，构建双行 Triton 内核，在同一 GPU 线程中并行执行生成与验证，几乎不增加延迟；3）通过 token 级监督实现更早的危险内容检测。

**🔧 技术方法**

采用技术包括：低秩适配器（LoRA）、Triton 双行线性/FFN 内核、KV 缓存共享、Silu 激活、token 级监督、CUDA 图捕获。

**📊 数据集**

使用七个安全标注数据集（WildGuardMix、Aegis 2.0、BeaverTails、PKU-SafeRLHF、ToxicChat、Aegis 1.0、ExpGuardTrain），共 33.6M token；评测基于 Qwen3-0.6B 的 17 个公开 benchmark。

**📈 对比分析**

与 Qwen3Guard-Stream（专用 guard）和 SingProbe（隐藏状态探测）在 F1、流式检测率、检测延迟、内存占用等指标上对比：平均 F1 仅比 guard 低 0.4 点，超过 SingProbe 3.8 点；流式检测 83.6% vs 84.7% vs 71.8%；每 token 延迟仅 +3.5% vs 85.9%；内存额外 9.3MiB vs 910 MiB。

**⚠️ 局限性**

局限性：只针对批量为 1 的 memory‑bound 解码，批量增大后优势减弱；实验仅在 Qwen3-0.6B 上验证，其他规模和模型未完整测试；需要特定 GPU 和 Triton 内核实现；流式评测使用回放而非在线生成。

---

## 47. H-MSAE: Hierarchical Sparse Concept Coverage for Efficient and Diverse Instruction Data Selection

**arXiv ID:** 2611.27329 | [PDF](https://arxiv.org/pdf/2611.27329v1)

**作者:** Bohan Cui `[一作]` (JD.com), Simiu Gu `[通讯]` (JD.com)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于多分辨率稀疏自编码器的层次稀疏指令选择框架H-MSAE，并在Tulu 3上实现稀疏概念构建与质量权重覆盖的子集选择。

**💡 创新点**

将跨分辨率稀疏概念构成层次关系，并在覆盖目标中加入质量权重和长度约束，避免生成标签、显式图构建，显著提升效率与性能。

**🔧 技术方法**

使用冻结的对话嵌入+Matryoshka稀疏自编码器、跨分辨率软父子关系、质量加权凹式覆盖目标、长度量化束缚的随机贪婪算法。

**📊 数据集**

在公开的Tulu 3（约939k条指令）上采样50k条子集，并在Qwen3系列（1.7B/4B/8B）上进行指令微调评估。

**📈 对比分析**

与Uniform Random、DEITA、#InsTag、MIG等主流选择器对比，在三种模型规模下的“Final Avg”指标上均获得最高分，并将端到端选择时间缩短约5×。

**⚠️ 局限性**

对稀疏概念的质量与语义覆盖与梯度相关性假设仅在局部近似下成立，且对不同语言或更大规模数据的迁移性尚未验证。

---

## 48. Lift Out, Kick In: Decoupling Event Effects for Time Series Forecasting

**arXiv ID:** 2611.27191 | [PDF](https://arxiv.org/pdf/2611.27191v1)

**作者:** Yunying Zhu `[一作]` (DiDi Chuxing), Zhichao Zou `[通讯]` (DiDi Chuxing)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出Loki框架，将事件响应与时间序列预测分离并联合学习

**💡 创新点**

创新点在于共享的事件响应函数（ERF）与Lift Out–Kick In双向分解，使历史事件效应与未来事件响应通过同一模型共享并共同优化

**🔧 技术方法**

采用基于多尺度时间基的可学习ERF、乘法式事件调节、轻量级SparseTSF/其它预测骨干，整体以端到端方式训练

**📊 数据集**

在五个多样化基准上评测：Ride‑Hailing、TaxiBJ、M5、Rossmann、Seoul Bike

**📈 对比分析**

与18种基线（包含DLinear、PatchTST、SparseTSF、TimeMixer、TiDE、TFT、Prophet、NeuralProphet等）比较，Loki在所有数据集上实现标准化MAE最低，平均提升2.2%–11.7%，同时在MSE和WMAPE等指标亦处于最优或次优水平

**⚠️ 局限性**

局限性包括：依赖完整准确的事件日程；共享ERF假设事件效应可归约为比例型并可在不同实例共享，难以捕捉高度依赖上下文或非比例的效应；模型并非因果估计，不能在观测数据中直接识别因果关系

---

## 49. Omo: Giving Continuing AI-Agent Work a Place to Return To

**arXiv ID:** 2611.27107 | [PDF](https://arxiv.org/pdf/2611.27107v1)

**作者:** Kaviru Hapuarachchi `[一作]` (Omo Research), Harrison Edwards `[通讯]` (Omo Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了 Omo 系统，提供持久化的空间/房间，让 AI 编码代理在离开后仍能继续工作，回到原位置后能快速恢复任务。

**💡 创新点**

创新点在于将会话持久化拆分为执行连续性、工作区识别和上下文恢复，并提出“持久空间”概念来支持后者；同时实现了两个实现版本（Minecraft 模组和 Eclipse Theia 空间 IDE），共享同一后端。

**🔧 技术方法**

使用了 Minecraft Java Mod（Omo Space）、Eclipse Theia（Omo Theia）、tmux 服务器来持久化终端进程、以及多种 AI 编码模型（Codex、Hermes、Claude Code 等）来演示代理工作。

**📊 数据集**

论文未使用公开数据集；研究依赖内部团队的 30 名成员在 Minecraft 世界中构建房间，以及计划在实验中使用的若干小项目案例。提及参考了 10,000 次编程会话的统计数据作为背景。

**📈 对比分析**

计划进行 within‑participant 对比实验，比较三种界面（已命名会话列表、稳定 2D 布局、Omo 房间）对任务恢复的影响；评价标准为能否正确复述目标、说明已更改的内容并给出合适的后续指令；目前尚无实验结果可报告。

**⚠️ 局限性**

局限性包括：需要花费时间和精力构建与维护持久空间；若使用 Minecraft 则需额外运行游戏；空间记忆可能导致误解或忘记实际更改；空间与工作区的关联可能因项目移动而弱化；实验尚未验证对非专业用户的有效性。

---

## 50. Co-Evolving Behavioral Agents and Detectors for Adaptive Information Operations

**arXiv ID:** 2611.27156 | [PDF](https://arxiv.org/pdf/2611.27156v1)

**作者:** Lin Tian `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计并评估了IO-AI框架，通过对抗式游戏训练顺序信息操作检测器与自适应行为代理，并构建从易到难的训练曲线。

**💡 创新点**

引入交叉游戏难度排序、行为距离度量以及多目标（参与度与躲避）对抗训练，提升对新演员和新时期信息操作的迁移性能。

**🔧 技术方法**

采用多阶段强化学习（PPO）与自对弈、行为克隆、情感+时间+类型的动作空间、Mamba/IC-Mamba序列模型、DPO优化以及Jensen–Shannon/Wasserstein距离等技术。

**📊 数据集**

使用20个Twitter官方归档的国家后背信息操作活动（2018-2021）数据集，共12个训练、8个保留，辅以真实非IO控制轨迹。

**📈 对比分析**

与六个现有基线（RF、BERT、Propagation RNN、MetaTroll、协调图、IOHunter）及IO-AI的多种变体（单游戏、静态、多游戏）进行F1、MissRate、FalseAlarm比较；IO-AI在未见演员/时期的F1达0.81，优于静态训练0.62，并在对抗性攻击中保持低误报率。

**⚠️ 局限性**

仅基于Twitter数据，无法直接验证跨平台适用性；对话和文本生成器依赖预训练模型，可能缺乏对极端或高度定制化的操作行为；行为距离与检测器迁移的关联是经验相关，缺乏理论保证。

---

## 51. Drive Failure and Lifetime Simulation in the Backblaze Drive Statistics Corpus, 2013-2026

**arXiv ID:** 2611.27122 | [PDF](https://arxiv.org/pdf/2611.27122v1)

**作者:** Ahmed Amer `[一作]` (Santa Clara University), Darrell D. E. Long `[通讯]` (Santa Clara University)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文利用Backblaze公开的 2013–2026 年硬盘驱动器使用与失效日志，对驱动器寿命模型、预测和故障风险进行系统评估，并结合 SMART 监测数据和 Bayesian 预测实现高效的故障预警与更换需求估计。

**💡 创新点**

创新点在于：① 将 99 种驱动器型号的差异通过基于驱动器属性的 Gamma 先验与经验贝叶斯多级模型统一建模；② 通过九年滚动原点检验和多模型比较（包括 Weibull、混合 Weibull、三重水桶等）评估模型对未来失效计数的概率分布与置信区间；③ 引入年均随机波动与模型‑年冲击（shock）共享因子，刻画整体与型号级的时变依赖；④ 提供 C++/R 实现的寿命采样器并测算其计算成本；⑤ 结合 RAID 仿真展示不同驱动器选择与共享风险对数据损失风险的四个数量级差异。

**🔧 技术方法**

使用的技术包括：左截右删（left‑truncated, right‑censored）似然估计、Gamma 纰率（frailty）多级贝叶斯建模、负二项预测分布、Poisson 计数模型、随机游走/AR(1)/独立年效应、负二项校正的年级冲击、No‑U‑Turn 采样器（NumPyro）、log‑score 与 PIT 评价、ROC/AUC、Brier、Poisson/负二项方差膨胀检验、泊松-负二项残差分析、RAID 可靠性仿真。

**📊 数据集**

数据集为 Backblaze 公开的 504,656 块硬盘的每日记录（2013‑2026 年），共 744,525,683 条 drive‑day 条目，包含驱动器序列号、型号、容量、失效标记与 93 个 SMART 传感器值，数据经过整理得到 99 个统一型号、31,881 次失败事件。

**📈 对比分析**

比较方法：将模型对 2017‑2025 年的未见年份进行预测，使用对数预测分数（log‑score）、预测区间覆盖率（50/80/95%）以及每个模型的残差分散度（kappa）进行评估。结果表明：① 经验贝叶斯两 Weibull 基线+属性乘子模型在 log‑score 上比固定三段水桶提高 4.3 倍；② 考虑年波动和冲击可进一步提升预测覆盖率至 95%；③ 在个体失效概率上，SMART 预警提升 AUC 从 0.64 提升至 0.73，机器学习每年误判约 19% ；④ 但即使预测精度高，累计失效预估仍可能偏高 50–73%，说明统计精度与实际更换需求不完全一致。

**⚠️ 局限性**

局限性包括：① 驱动器被移除而未失败的事件可能信息不完整，导致对故障率估计偏低；② 年度共享冲击与模型‑年冲击假设未捕捉所有内部相关性（如机架、数据中心等）导致残差膨胀；③ 对新型号的预测主要依赖属性先验，实际表现可能偏差；④ 只考虑硬盘，而未涵盖 SSD 或固态介质；⑤ RAID 可靠性仿真未与实际运维策略（如主动更换、热备份）结合。

---

## 52. Reliable Behavior Discovery via False Discovery Rate Control

**arXiv ID:** 2611.27104 | [PDF](https://arxiv.org/pdf/2611.27104v1)

**作者:** David Madras `[一作]`, Utsav Prabhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出“行为发现”框架，目标是系统地识别大生成模型的行为模式，并在此过程中通过统计检验控制错误发现率（FDR）；

**💡 创新点**

创新点在于：①把行为发现视为多假设检验问题并引入FDR控制；②提出基于e值的eBH包装器来降低LLM启发式方法的FDR；③设计了自适应bandit式方法（如改进版eBH），在保持FDR≤α的同时显著提升检验功效；

**🔧 技术方法**

技术主要包括：e值与e-process构造、eBH与加权/提升eBH、bandit式假设选择、假设冻结与e值分箱、Monte Carlo估计E_A与P_H、预后/后验FDR上界；

**📊 数据集**

数据集与实验：①合成的上下文bandit环境（高斯行为空间）用于验证FDR与覆盖率；②真实大生成模型（LLM与图像生成器）进行Word Count、Hedging、Cats、Handedness、Symmetry等任务，采用LLM生成的提示与查询；

**📈 对比分析**

对比实验显示：传统LLM启发式方法FDR高达0.4‑1；加入eBH后FDR降至<α，但功效下降约30%；自适应bandit方法在相同α下，FDR维持≤α且真发现数明显高于eBH，覆盖率亦更好；

**⚠️ 局限性**

局限性包括：需要已知null分布与独立性假设；假设冻结与分箱策略对性能敏感；在非静态模型或多次数据重用情形下理论不直接适用；FDR估计方差大、对齐与验证过程复杂；方法对LLM提示质量高度依赖，未解决跨域迁移与子假设复现问题。

---

## 53. ReSA: Preserving Reference-Specified Foreground in In-Context Segmentation

**arXiv ID:** 2611.27341 | [PDF](https://arxiv.org/pdf/2611.27341v1)

**作者:** Dezhao Su `[一作]` (Fudan University), Yunyun Wan `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于参考图像的无任务训练分割框架 ReSA，通过共享参考核统一查询适配和掩膜选择。

**💡 创新点**

创新点在于：① 用同一参考核同时约束前景响应的自适应（Response‑Constrained Adaptation）和阈值拟合的参考加权重建；② 引入相对重建增益的候选选择；③ 通过阈值对齐使 CRF 的 unary 项与掩膜一致，提供精细边界；④ 给出理论上界，关联适配误差与软重建误差。

**🔧 技术方法**

技术手段包括：冻结 DINOv3 ViT-L/16 编码器；前景/背景参考原型；软最大化相似度得到参考核；熵正则化的二分类器自适应；参考加权 KL 重建阈值；相对重建增益选择；CRF 边界细化；轻量级两分支推理。

**📊 数据集**

使用十个公开基准：自然物体（LVIS‑92、COCO‑20、FSS、PASCAL‑5）、部件分割（PASCAL‑Part、PACO‑Part）以及跨域医学与遥感（ISIC、SUIM、iSAID、X‑Ray）。

**📈 对比分析**

与同类无任务训练方法（INSID3、RSRM、FROST 等）对比，ReSA 在 1‑shot 和 5‑shot 上分别取得 62.6% 与 71.8% 的平均 mIoU，分别比 INSID3 提升 4.4/7.2 分，整体排名第一。

**⚠️ 局限性**

局限性：依赖冻结特征和参考核匹配，可能对极端视角/光照变化敏感；在部分数据集如 COCO‑20 的额外前景率略升高；推理时间相对较高（1‑shot 0.205 s，5‑shot 0.420 s）。

---

## 54. Federated and Balanced $k$-means: Communication Efficiency, Privacy, and Heterogeneity

**arXiv ID:** 2611.27114 | [PDF](https://arxiv.org/pdf/2611.27114v1)

**作者:** Feiteng Nie `[一作]` (Wuhan University), Zhiyong Peng `[通讯]` (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于 ADMM 的垂直联邦 k-means 算法（Fabric），实现了聚类平衡、差分隐私以及通信效率的统一。

**💡 创新点**

创新点：1) 将平衡约束化为等式并通过 ADMM 解耦成各客户端子问题；2) 采用块坐标更新大幅降低通信轮数；3) 在本地目标评估中注入剪裁后的高斯噪声，给出正式的 (ε,δ)-DP 保障；4) 通过仅交换标记矩阵和标量目标实现更强的隐私保护。

**🔧 技术方法**

核心技术：ADMM、块坐标下降、差分隐私（Gaussian机制与 RDP/renyi 转换）、纵向联邦学习框架、中心化指示矩阵分解。

**📊 数据集**

使用十个真实数据集，包括医疗、金融等领域（如 COPD 病例、金融交易、教育记录等），覆盖数十万到百万级样本。

**📈 对比分析**

与七个基线（传统 k-means、BKM、BCLS、FCFC、BKHK、VFLC 等）比较：平均 SSE 降低 29.1%，平衡度提升 1.41 倍，训练时间平均缩短 26 倍，通信轮数可达 1000 倍减少；在不同 ε 与 ρ 取值下均表现出良好的性能与隐私折中。

**⚠️ 局限性**

局限性：1) 依赖诚实但好奇的威胁模型，无法防御恶意参与者；2) 隐私预算 ε 过小会显著影响聚类质量；3) 目前不支持动态加入/离线客户端；4) DP 噪声虽对聚类影响有限，但在极端稀疏或高维场景下可能产生更大偏差。

---

## 55. S2D-Splat: Static-First Gaussian Reconstruction of Dynamic Driving Scenes with Layer-Wise Feature Guidance

**arXiv ID:** 2611.27003 | [PDF](https://arxiv.org/pdf/2611.27003v1)

**作者:** Jinheng Li `[一作]` (Xi'an Jiaotong University), Jianwu Fang `[通讯]` (Xi'an Jiaotong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了S2D-Splat框架，实现静态优先的Gaussian重建，先分离静态与动态特征，再通过层级特征引导完成相机、静态几何和动态几何的恢复，并通过实例级运动解码器估计物体轨迹。

**💡 创新点**

创新点包括：①静态优先解耦，提前抑制动态干扰；②异步层级特征引导，用缓存的静态特征在动态分支中固定条件；③实例级运动解码器直接预测物体位姿，实现动态Gaussian对齐与时序插值。

**🔧 技术方法**

技术手段涵盖：DINOv3特征提取、SAM2动态实例掩码、交错注意力与可变形注意力、跨层静态动态特征分离、Gaussian场深度与颜色预测、交叉注意力Transformer实例运动解码、渲染损失（RGB、深度、相机姿态）。

**📊 数据集**

在Waymo Open Dataset上训练，零样本评估于nuScenes和MM‑AU事故数据集。

**📈 对比分析**

与MVSplat、DepthSplat、AnySplat（静态）及DGGT、StreetForward（动态）对比，S2D‑Splat在PSNR/SSIM上最高、D‑RMSE最低、相机姿态AUC最高、动态APD_3D显著提升，并在零样本场景中保持优异表现。

**⚠️ 局限性**

局限性：需人工或检测器提供2D框提示，缺乏掩码自我修正机制，实例运动估计仍有提升空间。

---

## 56. Structurally Different, Functionally Persistent: Capability Regions Under Model Optimization

**arXiv ID:** 2611.27445 | [PDF](https://arxiv.org/pdf/2611.27445v1)

**作者:** Jiwoo Kum `[一作]` (Soongsil University), Chanjun Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估在量化、剪枝和LoRA适配后，先前识别的编码能力区域是否仍能作为有效的干预目标。

**💡 创新点**

提出区分功能持久性与参数重叠的框架，并证明在多种模型压缩与适配方式下，功能持久性强，而重叠并不预测干预效果；进一步通过参数分区分析揭示共享参数对能力影响最大。

**🔧 技术方法**

基于重要性评分的能力区域提取、参数消融、4‑bit RTN/ GPTQ/ AWQ 量化、无结构剪枝（Magnitude、Wanda、SparseGPT）和LoRA低秩适配；使用幅度匹配对照消融。

**📊 数据集**

代码语料库（10种编程语言，排除Python）、通用文本语料库（6种语言的维基百科）、校准数据（English C4）、评估基准（HumanEval、HellaSwag、GSM8K、WikiText‑2）。

**📈 对比分析**

将原始区域迁移至优化模型并消融，对比幅度匹配对照；使用 HumanEval、HellaSwag、GSM8K、WikiText‑2 作为指标；发现迁移后消融几乎完全消灭编码能力，而对照保持性能；功能持久性高，但参数重叠仅约 60%；共享参数消融导致最强影响。

**⚠️ 局限性**

实验仅覆盖三种 3B–8B 英语模型；压缩方法仅在单一强度（4‑bit/30%）下测试；k 取值仅四个，未报告方差；限制了对更强优化和多语言模型的泛化能力。

---

## 57. DepthMSA: Combinatorial Training and Budget-Aware Depth Allocation for Multimodal Sentiment Analysis

**arXiv ID:** 2611.27198 | [PDF](https://arxiv.org/pdf/2611.27198v1)

**作者:** Pan Wang `[一作]` (University of Pittsburgh), Jingtong Hu `[通讯]` (University of Pittsburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出 DepthMSA，一种在多模态情感分析中根据计算预算灵活调节各模态编码深度的框架。

**💡 创新点**

创新点包括：① 组合式 anytime 训练，使融合头同时学习不同深度组合的特征融合；② 基于验证误差的成本-效用表实现无训练的、样本级预算分配；③ 可选的模态跳过（skip）机制进一步降低 FLOPs；④ 只需训练一次模型即可覆盖所有预算。

**🔧 技术方法**

使用技术包括：共享权重的多出口 Transformer 编码器、组合式 supervision（全深度+随机+组合），验证误差校准的成本-效用表，基于 FLOPs 估算的预算约束求解。

**📊 数据集**

数据集：CMU-MOSI、MOSEI（英语视频情感）和 IEMOCAP（中文视频情感）。

**📈 对比分析**

与全深度稠密模型和 ADMN（自适应层分配）对比，DepthMSA 在相同预算下保持 1% 以内的准确率/MAE，同时将测量 FLOPs 降至 45%–82%，推理延迟也显著下降；skip 版本在更低预算下进一步提升效率。

**⚠️ 局限性**

局限性：① 退出点数量有限，过于粗糙的深度划分可能导致在极低预算时性能下降；② 分配策略基于验证集效用表，缺乏对内容敏感的动态决策；③ 对模态重要性变化的适应性不足，需在更复杂或噪声更大的多模态任务中进一步验证。

---

## 58. How developers split roles among them and a coding agent and what "collaboration" means

**arXiv ID:** 2611.26668 | [PDF](https://arxiv.org/pdf/2611.26668v1)

**作者:** Linus Ververs `[一作]` (Freie University Berlin), Lutz Prechelt `[通讯]` (Freie University Berlin)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过三步混合方法研究，探索开发者与编码代理（如Claude Code、GitHub Copilot CLI等）之间的角色分工与协作方式。第一步对2025年早期采用者的博客进行开放式编码，提炼出九个角色分工类别并归纳为意图形成、设计实施、评估三大簇；第二步对2026年的开发者进行问卷调查，验证并比较早期角色分工的持续性及变化；第三步对调查中的自由文本回答进行代理辅助开放式编码，识别开发者对“协作”三种认知维度：活动中心的分工、整体流程协作以及个人偏好。

**💡 创新点**

创新点在于①首次系统性梳理开发者与大型语言模型驱动的编码代理在实践中的角色分工及其演变；②提出并验证了基于代理辅助的开放式编码方法，展示了LLM在定性研究中的可行性；③将协作概念细化为三维视角，揭示协作方式与个人偏好之间的冲突与潜在摩擦。

**🔧 技术方法**

主要使用的技术包括：大型语言模型（Claude Code、GitHub Copilot CLI、OpenAI Codex 等）作为编码代理；定性研究方法中的开放式编码、Strauss‑Corbin Grounded Theory；问卷调查采用 Likert 量表、二项/多项选择题；代理辅助开放式编码实现了命令行工具 codebook.py 与 Claude Opus 5 编写的开放式编码技能。

**📊 数据集**

数据集来源：①四位早期采用者（Hashimoto、Ronacher、Steinberger、Willison）撰写的技术博客与公开仓库（共4份案例）；②2026年共收集到 2026 名开发者问卷，其中 171 名为代理用户、69 名非用户；③自由文本回答（约 154 条）用于协作方式分析。

**📈 对比分析**

在问卷中，研究将早期角色分工预测与实际分工进行对照，利用均值、置信区间及分布比对，验证了预测的一致性（大部分被确认）和偏移（如从完全人类转为共享）。对于协作方式，采用自由文本编码得到的三大类（collab‑*, divide‑*, holistic‑*），并通过计数和频率描述其占比。总体表现显示：开发者对代理的信任与使用强度正相关，但对协作模式持多样化偏好，体现潜在摩擦。

**⚠️ 局限性**

局限性包括：①样本不具代表性（主要来自德语圈、网络社群），可能导致偏差；②问卷设计允许选择性回答，导致缺失数据；③开放式编码的主观性和对 LLM 生成结果的依赖，可能影响编码一致性；④缺乏对协作效果的客观性能度量，仅有自我报告；⑤只聚焦于 2026 年的初期阶段，无法反映长周期技术演进。

---

## 59. LLMs Memorize Their Own Hallucinations During Post-Training

**arXiv ID:** 2611.26863 | [PDF](https://arxiv.org/pdf/2611.26863v1)

**作者:** Anirudh Atmakuru `[一作]` (Cornell University), Tanya Goyal `[通讯]` (Cornell University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了推理导向的后训练（RL）如何导致大语言模型记忆其自生成的幻觉事实。

**💡 创新点**

首次系统量化自生成幻觉在RL后训练中的记忆效果，发现其平均提升4.83%，并在困难问题中更为显著。

**🔧 技术方法**

采用GRPO强化学习、回溯式推理链、奖励模型，利用GPT‑4.1‑mini提取幻觉并构造MCQ进行记忆分数评估。

**📊 数据集**

使用闭域阅读理解数据集QuALITY和开放域事实检索数据集TriviaQA进行实验。

**📈 对比分析**

通过对比训练集幻觉与未训练幻觉的记忆分数，评估不同模型、规模和架构，结果显示大部分模型在训练集幻觉上显著提升；加入KL正则后效果差异不大。

**⚠️ 局限性**

局限性包括：仅聚焦知识任务，对长推理链细粒度真伪监督缺乏可扩展方法；MCQ评测与真实生成的差异可能影响结论。

---

## 60. How Language Models Compress and Compare: Understanding Selection with Token Covariance Maps

**arXiv ID:** 2611.26874 | [PDF](https://arxiv.org/pdf/2611.26874v1)

**作者:** Moshe Eliasof `[一作]` `[通讯]` (Ben-Gurion University of the Negev), Moshe Eliasof (Ben-Gurion University of the Negev)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并应用“token covariance maps”这一新的可视化工具，系统分析了decoder‑only语言模型在结构化选择任务中的层级信息组织；

**💡 创新点**

核心创新在于发现并验证了一个普遍存在的“compress‑then‑compare”深度结构：早期层级保持信息分散，中间层形成低秩压缩瓶颈，后期层级重新扩展并构建候选答案比较几何；

**🔧 技术方法**

技术手段包括：token covariance map 的构建、有效秩(effective rank)与谱统计、块级平均协方差(CTX‑OPT/OPT‑SELF/OPT‑CROSS)度量、低秩截断干预、以及层级中间读出(intermediate readouts)；

**📊 数据集**

实验覆盖12个结构化选择基准（PIQA、BoolQ、HellaSwag、ARC‑Easy、ARC‑Challenge、COPA、WinoGrande、OpenBookQA、CommonsenseQA、Social IQa、SciQ、LogiQA）以及14个decoder‑only模型（Pythia、SmolLM、Qwen、Mistral等四大开源族群）；

**📈 对比分析**

比较方法：对不同模型、不同层级进行有效秩与块结构的统计，利用低秩扰动验证瓶颈功能，并通过中间读出评估答案判别时间；结果显示所有模型均呈现相同的压缩-再扩展模式，干预证实中间瓶颈对后期比较几何至关重要；与传统的隐藏层秩、Centered Cosine、CKA、SVCCA等指标比较，token covariance maps提供了更细粒度、候选级别的解释；

**⚠️ 局限性**

局限性：受tokenization、格式、句子长度及隐藏层归一化影响；仅捕捉对称协方差，无法直接表征因果方向或语义完整性；层级内具体方向的归因尚未完成；干预仅局部可解释，整体机制仍需进一步探索。

---

## 61. ConRef: Auditing Reference Reliability in 2D/3D Registration Benchmarks

**arXiv ID:** 2611.26758 | [PDF](https://arxiv.org/pdf/2611.26758v1)

**作者:** Jianfeng Yue `[一作]` (Technical University of Munich), Nassir Navab `[通讯]` (Technical University of Munich)

**通讯引用:** 94431 | [OpenAlex ID](https://openalex.org/A5046896448)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多视角共识参考框架（ConRef），利用多张 X‑ray 图像联合估计 2D/3D 注册基准中的参考姿势并检测异常。

**💡 创新点**

创新点在于通过共识姿势提供额外的参考、量化其不确定性，并自动标记可能的 X‑ray 异常，从而提高基准评估的可靠性。

**🔧 技术方法**

使用了归一化交叉相关（NCC）图像相似度、二阶 Newton 优化、删取法（jackknife）估计不确定性以及单视角一致性评估等技术。

**📊 数据集**

实验涵盖公开的 Femur、DeepFluoro 数据集，并发布了新建的 BeadPhant CBCT/X‑ray 物理模型数据集进行验证。

**📈 对比分析**

将 ConRef 与原始参考（OrigRef）及独立测量（BeadPhant beads、Femur AnatRef）对比，结果表明 ConRef 与独立参考的误差在 0.1–0.3 mm 范围，能够改变注册成功率判定，体现出评估精度的提升。

**⚠️ 局限性**

局限性包括对足够角度覆盖和投影几何精度的依赖；角度不足时共识姿势可能不够精确；不同注册算法产生的共识可能存在系统偏差，无法完全消除。

---

## 62. More Bits Won't Fix It: Diagnosing and Treating an Ill-Conditioned Backward Pass (A Case Study on Gemma-4)

**arXiv ID:** 2611.27243 | [PDF](https://arxiv.org/pdf/2611.27243v1)

**作者:** Jeong Hyun Lee `[一作]` `[通讯]` (Samsung Medical Center, Sungkyunkwan University), Jeong Hyun Lee (Samsung Medical Center, Sungkyunkwan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

研究了Gemma-4在长文本与图像提示下LoRA微调时bf16与fp32梯度的差异，定位问题源并提出仅在反向传播中停用下层MLP路径的修复方法。

**💡 创新点**

首次发现bf16梯度在下层对长图像提示高度不稳定，定位到层19–21的MLP，并通过在反向传播中detach该路径显著提升训练效果，显示在低精度训练中修复梯度路径比提升位宽更有效。

**🔧 技术方法**

采用LoRA低秩适配、混合精度（bf16/fp32）训练、梯度相似度与噪声敏感性分析、停用梯度路径（detach）与层冻结实验、梯度裁剪以及cosine similarity评估等技术。

**📊 数据集**

使用私有CT病灶定位数据集（含CT切片、肝病灶标记和视角标签），以及自然照片和长文本提示作为对照。

**📈 对比分析**

通过对比bf16标准、fp32标准、冻结下层、detach、detach+冻结等多种实验，300步训练后，detach方案将验证NLL从0.336降至0.296，中心误差从16.3降至10.8，in-box率从75%提升至88%，梯度峰值和裁剪率显著下降。

**⚠️ 局限性**

实验仅覆盖单一模型（Gemma‑4‑12B）和单一任务、单一训练步长、单一数据顺序和单一fp32种子；未对长文本提示进行定位，机制与修复对其他模型的适用性未知；detach梯度仍存在偏差，未证明其为真实梯度。

---

## 63. LiDAR Signals for Robot Locomotion: Spherical Range Images Enable Robust Sim-to-Real Transfer

**arXiv ID:** 2611.27223 | [PDF](https://arxiv.org/pdf/2611.27223v1)

**作者:** Nian Liu `[一作]`, Zhaofeng He `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一套基于LiDAR的感知管道，将原始点云转换为球面范围图，用于端到端训练和部署四足机器人行走策略。

**💡 创新点**

通过对真实LiDAR扫描角度模式的重放与自遮挡建模，实现感知与仿真的一致性；采用共享的球面范围图编码，消除了教师-学生蒸馏、在线地图构建和适配需求。

**🔧 技术方法**

使用GPU并行射线投射、角度模式回放、自姿态补偿、时域聚合、LiDAR特定域随机化、PPO强化学习、Isaac Gym等技术。

**📊 数据集**

在Unitree Go2机器人配备L1/L2 LiDAR的三种挑战地形（gap、step、hurdle）上进行训练和测试；采集1500条LiDAR扫描用于角度模式回放，使用仿真生成的随机地形地图。

**📈 对比分析**

与深度图、在线高程图、PointNet、Raw LiDAR等基线对比；在仿真中均能达到最高难度L=9，硬件上L1得到60/55/100%（gap/step/hurdle）成功率，平均71.7%；相较于深度（40/35/75%）和高程图（0/5/35%）大幅提升，sim-to-real保留率约74%；ablation实验显示扫描模式回放最关键。

**⚠️ 局限性**

需要预先采集并重放扫描模式，依赖特定硬件；对极低频或高噪声LiDAR环境适应有限；自遮挡建模与域随机化参数需经验调优；目前仅验证单机器人与三种地形，复杂场景未知。

---

## 64. Turnout: Adaptive Attention Switching for Efficient Reasoning

**arXiv ID:** 2611.27500 | [PDF](https://arxiv.org/pdf/2611.27500v1)

**作者:** Anna Kuzina `[一作]` (Qualcomm Technologies, Inc.), Babak Ehteshami Bejnordi `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在预训练的全注意力Transformer基础上，引入混合后置线性化，并通过强化学习学习何时将每层从全注意力切换为线性注意力。

**💡 创新点**

创新点在于提出可学习的层级与时间切换策略，并通过强化学习平衡准确率与内存消耗。

**🔧 技术方法**

使用的技术包括后置线性化、LoRA低秩适配、激活与logit蒸馏、GDN门控的线性注意力以及基于奖励的强化学习路由器。

**📊 数据集**

训练与评估使用OpenThoughts3、DCLM-Edu等大规模数据集，以及GSM8K、MATH-500、GPQA-Diamond、HumanEval+、MBPP+等推理基准。

**📈 对比分析**

在多项推理基准上，学习到的路由策略在保持约92%–96%准确率的同时，将缓存内存降低25%–30%，优于固定切换和全线性化基线。

**⚠️ 局限性**

局限性包括对长序列的依赖仍有限、需要额外的RL训练过程以及在非推理任务上的可迁移性待验证。

---

## 65. Schema-Validated LLM Workflows: A Benchmark Protocol and Comparative Evaluation Across Six Python Implementations

**arXiv ID:** 2611.27106 | [PDF](https://arxiv.org/pdf/2611.27106v1)

**作者:** Sidney Shapiro `[一作]` (University of Lethbridge), Daniel Pearson `[通讯]` (University of Lethbridge)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个可重复的 benchmark 协议和可执行的 artifact，评估了六种 Python 实现条件下的 LLM 业务工作流中结构有效性与任务正确性的分离效果。

**💡 创新点**

首次系统化区分结构有效性与任务正确性，并通过冻结的分析计划在相同业务任务下比较实现路径的可比性，揭示绑定路径异质性和交互效应。

**🔧 技术方法**

使用结构化输出（JSON Schema）、Pydantic、LangGraph、DSPy、CrewAI、Agno 等框架及其 SDK 对接，结合自动化评分器和重试机制。

**📊 数据集**

使用合成的发票、投诉、报告、政策、SQL 数据库和支持请求等业务样本，覆盖五类任务（抽取、SQL 生成、摘要、合规 QA、工作流决策）。

**📈 对比分析**

通过 5,400 行可重复的 API 调用，计算结构有效率与自动化质量得分，发现 PydanticAI/LangGraph/DSPy 接近 100% 结构有效率且质量高，SDK、Agno、CrewAI 低；且存在任务-条件交互，性能差异非均匀。

**⚠️ 局限性**

受限于合成数据、单一模型版本、有限的机制覆盖、未完成的人类审计、适用范围局限于短期工作流和未考虑安全与可解释性等问题。

---

## 66. 3D Open-Vocabulary Detection via Visual Alignment and Generative Captions

**arXiv ID:** 2611.26842 | [PDF](https://arxiv.org/pdf/2611.26842v1)

**作者:** Thomas E. Huang `[一作]`, Mingxing Tan `[通讯]` (Waymo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种针对户外场景的3D开放词汇目标检测框架，利用无标签的3D几何提议进行双模态蒸馏；

**💡 创新点**

创新点在于：①使用无监督的3D聚类生成全场景提议，实现对象级别的视觉蒸馏；②通过多模态大型语言模型生成丰富的文本描述进行语言蒸馏，将语义与3D特征直接对齐；

**🔧 技术方法**

采用TIPS视觉模型提取高分辨率空间特征，Gemini 2.5 Flash生成图片字幕，结合Transformer‑based 3D检测器SWFormer以及轻量级开放词汇嵌入头；

**📊 数据集**

主要在Waymo Open Dataset与nuScenes上进行实验；

**📈 对比分析**

与现有ViLGOD、UP‑VL、OpenSight等方法比较，取得3D mAP提升约9–10点（Waymo）和2.7点（nuScenes）等显著性能提升；

**⚠️ 局限性**

局限性包括对相机标定的依赖、对3D聚类参数敏感、在极端遮挡或稀疏点云场景下的提议质量可能受限。

---

## 67. Do Companies Deliver on Their IPO Claims?

**arXiv ID:** 2611.27273 | [PDF](https://arxiv.org/pdf/2611.27273v1)

**作者:** Vidhyakshaya Kannan `[一作]` (Stanford University), Sudheer Chava `[通讯]` (Georgia Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 Vision Follow-Through (VFT) 框架，用于评估 IPO 披露中前瞻性声明的兑现情况。

**💡 创新点**

创新点在于把声明提取、类别分配、可验证性评估和基于后续证据的兑现评估拆分为四个独立任务，并提供大规模人工标注的数据集。

**🔧 技术方法**

利用大型语言模型（Claude Haiku、Gemma、GPT‑6 Luna 等）结合检索、排序、RRF 等技术构建自动化评估流水线。

**📊 数据集**

数据集包含 50 份 IPO 披露文件，5,066 个人工标注的前瞻性声明，其中 1,000 条标注了可验证性等级和兑现结果，后续证据来自后续 10‑K/10‑Q 报告和网络检索。

**📈 对比分析**

与人工标注对比，模型在类别分配上相对准确，但提取召回率低；兑现评估的整体准确率仅 18–44%，显示模型在后续证据推理方面仍有显著提升空间。

**⚠️ 局限性**

局限性包括样本偏倚（仅选取含 CEO/创始人信件的公司）、证据覆盖不足（仅 10‑K/10‑Q）、时间窗口限制以及难以区分缺失证据与非兑现。

---

## 68. Scalable Evaluation of Medical AI using Evidence-based Practice Recommendations

**arXiv ID:** 2611.27007 | [PDF](https://arxiv.org/pdf/2611.27007v1)

**作者:** Teya Bergamaschi `[一作]` (Google Research), Anil Palepu `[通讯]` (Google Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了一套自动化管线，将BMJ Best Practice指南转换为结构化诊疗树并生成4,561个验证过的开放式临床场景，用于评估医疗AI和人机协作。

**💡 创新点**

创新点在于自动化提取、验证临床标准并生成可追溯的分支场景，结合四点适当性刻度和训练好的LLM自动评分器，实现大规模、可更新、地域和时间敏感的AI评测。

**🔧 技术方法**

使用Gemini LLM进行结构化抽取、情境生成、自动校验及评分，并采用检索增强推理、样本合成、多路搜索和结构化摘录等推理框架。

**📊 数据集**

使用436本BMJ Best Practice临床手册（354美国版+82国际版），通过自动化流程生成3,200核心+731地域+630时间场景共4,561个。

**📈 对比分析**

对七个Gemini模型及不同推理配置进行基准测试，核心基准上模型规模和搜索/合成提升可遵循率至84.6%；在时间和地域套件中，搜索+合成可提升4–6个百分点；人机协作试验中，主治医生从50.5%提升至70.0%。

**⚠️ 局限性**

局限性包括仅基于单一指南来源，缺乏真实互动和多模态数据；自动抽取场景的临床准确性有限；评价基于LLM评分器的主观性；未覆盖低资源环境和更大型模型；人机试验受限于场景与即时反馈。

---

## 69. CENTRY: A Probabilistic Latent-State Model for Continual AI Risk Estimation, Monitoring, and Regulation

**arXiv ID:** 2611.26940 | [PDF](https://arxiv.org/pdf/2611.26940v1)

**作者:** Bonny Banerjee `[一作]` (Independent Scholar), Geetha Raju `[通讯]` (Indian Institute of Technology Madras)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `de8d30ba-c289-43a5-b4ec-7b80df73aea2`

**🎯 论文内容**

本文提出了 CENTRY，一种基于结构化潜变量的概率框架，能够在 AI 系统部署后持续估计、监控并调节其数据集、硬件与模型/算法等组件的风险。

**💡 创新点**

其创新点在于将 AI 风险视为可观测交互的潜在慢速子空间，并通过连续的 Surprise→Explain→Learn→Predict (SELP) 循环实现即时贝叶斯推理与自适应监管，从而在同一框架内实现风险估计、监控与调节的闭环控制。

**🔧 技术方法**

核心技术包括：结构化潜变量模型（快速交互子空间 + 三个慢速风险子空间）、变分 ELBO 学习、贝叶斯先验与递推更新、阈值化风险评分以及基于风险评分的分解式监管策略。

**📊 数据集**

论文未给出具体实验数据集，而是通过构建一个“风险因果数据库”U（包含交互记录、潜变量、配置元数据及证据）来说明如何在训练与持续学习中注入先验信息；实际验证与评测暂未提供。

**📈 对比分析**

由于缺乏实验实现与对比，本文在性能评估上仅给出了理论保证（如有限时隙风险合规、收敛与稳健性证明），未能在真实系统或基准数据集上与现有监控/治理方法进行定量比较。

**⚠️ 局限性**

主要限制包括：对严格贝叶斯假设、可观测交互与可控监管接口的依赖；理论证明在理想化假设下成立，实际部署中的模型误差、数据稀缺与非线性干扰可能导致风险估计与调节失效；此外，框架仅关注三类组件，未涵盖更广泛的上下文或多 AI 系统交互。

---

## 70. Effectiveness of Synthetic Defects for Anomaly Detection in Industrial Small Load Carriers

**arXiv ID:** 2611.26868 | [PDF](https://arxiv.org/pdf/2611.26868v1)

**作者:** Vincent Wahyudi `[一作]` (Technische Hochschule Würzburg-Schweinfurt), Dhavalkumar Thakker `[通讯]` (University of Hull)

**通讯引用:** 1655 | [OpenAlex ID](https://openalex.org/A5035962673)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

评估合成缺陷在监督与无监督模型中的检测效果，并比较不同噪声生成方式及其与真实缺陷的差异

**💡 创新点**

提出十种噪声类型和自定义合成缺陷recipe，量化合成缺陷与真实缺陷在特征空间的角度差距，并系统验证其在工业SLC与基准MVTec AD数据集上的迁移性

**🔧 技术方法**

使用DRAEM、SuperSimpleNet、DSR无监督方法；逻辑回归监督分类器；Perlin、fBm等噪声生成；通过AUROC评估以及角度度量对特征空间差异进行量化

**📊 数据集**

采用工业SLC真实数据子集（3085张图像）和MVTec AD 15类基准数据集

**📈 对比分析**

以AUROC为指标比较不同噪声、不同模型和不同训练比例；在SLC上，监督模型仅合成缺陷AUROC为0.65，加入10%真实缺陷提升至0.88；无监督模型默认Perlin噪声AUROC为0.87–0.90；在MVTec AD上合成缺陷与真实缺陷的AUROC几乎相同

**⚠️ 局限性**

合成缺陷与真实缺陷在特征空间差距大（SLC角度73° vs 42°/65°），导致迁移失败；自定义recipe未提升无监督方法性能；未验证更逼真或多尺度噪声对性能的影响

---

## 71. Learning Interaction Dynamics via Motion-Region Guidance for Robotic Video World Models

**arXiv ID:** 2611.27344 | [PDF](https://arxiv.org/pdf/2611.27344v1)

**作者:** Yongkang He `[一作]` (South China University of Technology), Xiangmin Xu `[通讯]` (Foshan University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 MoReWorld，通过预测运动区域并在生成时使用该区域指导，实现动作条件视频世界模型的交互动力学更真实。

**💡 创新点**

创新点在于统一的运动区域指导框架：先预测运动概率，再用作监督、损失加权和门控条件，聚焦局部交互同时保持全局一致。

**🔧 技术方法**

使用 V-JEPA 视觉特征预测、Wan 变压器解码器、VGGT 结构特征、WAFT 光流标签、运动门控条件及区域加权损失等技术。

**📊 数据集**

训练数据来自 RoboTwin 2.0 机器人录像集，评估基准使用 WorldSimProbe。

**📈 对比分析**

与 Ctrl-World、DreamDojo、Cosmos-3-Nano、LingBot-VA、MVA 等五个基线在 WorldSimProbe 的五项评分进行对比，MoReWorld 获得最高整体分 61.21，特别在 Interaction Dynamics 上提升约 19 分。

**⚠️ 局限性**

局限性包括运动区域指导导致视觉指标略低，生成的移动物体可能出现形变，且缺乏物体级约束，易造成细节失真。

---

## 72. Breaking the Long Horizon: Short-Horizon Process Supervision for Long-Form Generation in LLMs

**arXiv ID:** 2611.27348 | [PDF](https://arxiv.org/pdf/2611.27348v1)

**作者:** Heyang Xu `[一作]`, Jianxin Li `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出短期过程奖励框架 SH-PRM，按语义片段对长文本生成进行分段训练，提升信用分配效果。

**💡 创新点**

创新点在于将奖励细化到段落级别，实现局部优势估计和跨段干扰降低，理论证明信用分配优于单次整体奖励。

**🔧 技术方法**

采用基于前缀的短期生成、GRPO 优化、Rubric 评估模型、段落级别多维奖励，结合强化学习。

**📊 数据集**

使用 Synthetic 6163 样本构造的数据集以及 DeepResearch Bench 和 WritingBench 两个长文生成基准。

**📈 对比分析**

与基线（Base、Best-of-N、ORM-GRPO、SH-PRM无Rubric）对比，SH-PRM 在两大基准上平均提升约 2-3 分，并显著提高训练样本效率。

**⚠️ 局限性**

局限性包括对奖励评估器的依赖、段落粒度的手动选择、以及在极长文档或复杂语义结构下仍可能出现连贯性下降。

---

## 73. Neural Defend Ranks #1 in Global Audio Deepfake Detection Benchmark - ARCAS on the Podonos Independent Leaderboard

**arXiv ID:** 2611.27215 | [PDF](https://arxiv.org/pdf/2611.27215v1)

**作者:** Sivashankar Selvarajan `[一作]` (Neural Defend Inc), Sumit Singh `[通讯]` (Neural Defend Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Podonos独立深度伪造音频检测基准上，对Neural Defend的ARCAS 290M和ARCAS 90M两款基础模型进行评测，并将其与28个商业与开源检测系统进行对比，展示其在准确率、错误平衡、覆盖率和延迟等多项指标上的领先表现。

**💡 创新点**

创新点在于：①使用了由第三方构建、标签保密的独立基准，使不同检测器在同一文件集和统一评测规则下可公平比较；②首次将完整的错误统计、置信区间与统计显著性检验纳入公开榜单；③同时评估了延迟与覆盖率，揭示了商业系统在实际部署中的关键瓶颈。

**🔧 技术方法**

主要技术包括：ARCAS系列基于Transformer/卷积混合架构的音频检测基础模型；对不同文件格式（MP3、WAV、FLAC、OGG、M4A、WebM）与电话编码（G.711、G.729、AMR、EVS）进行统一预处理；采用默认阈值评分，计算准确率、F1、FPR、FNR、平均延迟及实时因子（RTF）。

**📊 数据集**

使用的核心数据集为Podonos 2026年10月版测试集：4,524条英语短音频（2,262真声，2,262合成），来自VCTK、LJ Speech、LibriTTS等真实语料，以及约25个当前TTS/语音克隆系统的合成语音；此外提供6种文件格式与两条电话轨道（窄带8 kHz与宽带16 kHz）。

**📈 对比分析**

评估方法：每个系统提交一份完整预测文件，Podonos使用私有标签按统一规则计算准确率、F1、FPR、FNR、错误率置信区间、Fisher精确检验及延迟统计。结果显示ARCAS 290M/90M以99.93%准确率（3个错误，FPR 0.1%，FNR 0.0%）位居榜首，错误率比第二名deetech.ai低约85%，延迟分别为118 ms和89 ms，满足实时性需求。

**⚠️ 局限性**

局限性包括：①仅涵盖英语短片，无法评估多语言与长段落；②测试集规模相对较小，单个错误对准确率影响大；③标签私有且评测为快照，后续发布可能导致排名变化；④未公开每个生成器或格式的细粒度结果；⑤电话轨道的评测结果尚未公布。

---

## 74. Decomposing images in a PRISM: A Single-Source Multi-View Framework for Image Representation

**arXiv ID:** 2611.26726 | [PDF](https://arxiv.org/pdf/2611.26726v1)

**作者:** Eduardo Campos `[一作]` (University of São Paulo), Alneu Lopes `[通讯]` (University of São Paulo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出 PRISM 框架，利用多种结构映射（图、序列、频域、文本等）从单一图像构造多样化视图，并在 PRISM-HD 实现中通过手工特征摘要和多分支 MLP 对这些视图进行融合，提升分类性能。

**💡 创新点**

创新点在于将单源图像系统地重映射到多种结构域，提供视图贡献评估机制，证明单源多视图不仅可提升性能，还能与深度网络互补，而非仅依赖外部预训练。

**🔧 技术方法**

技术包括：SLIC、Felzenszwalb 区域图、Hilbert、随机游走、傅里叶、Gabor、线图、可视化图、相似图、视觉语言文本生成等结构映射；手工特征摘要；多分支 MLP 编码；Shapley 与 CKA 评估。

**📊 数据集**

使用七个公开数据集：三幅艺术图集（OilPainting、Multitask125、Rijksdraw）和四个非艺术图集（MINC-2500、NCT-CRC-HE-100K、Places-LT、Food-101）。

**📈 对比分析**

与传统手工特征（Haralick、HOG、WND-CHARM）、Fisher Vector、冻结 ResNet-50/Vit-B/16/CLIP、Fine-tuned ResNet/Vit 进行对比。PRISM-HD 在所有七个数据集上均优于大多数浅层基线，且与 Fisher Vector 及深度网络按等权融合后提升 5–8% 的准确率，表明其提供互补的预测信息。

**⚠️ 局限性**

局限性：视图集合手工设计，计算成本不均；相似图依赖训练集，文本视图使用外部预训练模型；性能受任务与分类器影响，无法保证在所有场景下都能优于预训练深度模型。

---

## 75. Frontier Voltage Boost Laplacian Path Planning

**arXiv ID:** 2611.27308 | [PDF](https://arxiv.org/pdf/2611.27308v1)

**作者:** Vinay Pritamani `[一作]` (University of Washington), Joshua Smith `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了前沿电压提升拉普拉斯路径规划（FVB‑LPP）方法，通过分块求解拉普拉斯方程并逐步提升电压，生成无局部极小且平滑的导航路径。

**💡 创新点**

创新地采用逐段电压提升与冻结已解区域的策略，解决传统 HPF 在大地图和窄通道中出现的梯度平坦与数值精度限制问题，实现了可扩展且高效的平滑路径规划。

**🔧 技术方法**

使用并行 Jacobi 迭代（卷积形式）求解拉普拉斯方程、梯度下降轨迹提取、ε 容差避免平坦区、GPU 并行化实现等技术。

**📊 数据集**

在 1000×1000 像素随机障碍地图（共 1000 张）以及 10000×10000 大规模地图上进行实验评估。

**📈 对比分析**

与传统 HPF、APF 等方法对比，实验表明 FVB‑LPP 在成功率 100% 的同时，求解时间与路径平滑度可通过 λ（迭代次数）调节；在大地图上求解时间按 O(n^d) 扩展，性能优于原始 HPF。

**⚠️ 局限性**

仍受有限精度导致的平坦区影响，需要设置 ε；求解时间随 λ 及地图尺寸呈指数增长；对更高维空间和动态障碍的适应性尚未完全验证。

---

## 76. Rethinking ANNS on Commodity PIM: PIM-Native Vector Search with Cooperative Filtering

**arXiv ID:** 2611.27010 | [PDF](https://arxiv.org/pdf/2611.27010v1)

**作者:** Dawei Guo `[一作]`, Ming-Chang Yang `[通讯]`

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

未提供

**💡 创新点**

未提供

**🔧 技术方法**

未提供

**📊 数据集**

未提供

**📈 对比分析**

未提供

**⚠️ 局限性**

未提供

---

## 77. Projector Time Map Calibration for Event-based Structured Light

**arXiv ID:** 2611.27412 | [PDF](https://arxiv.org/pdf/2611.27412v1)

**作者:** Buu-Hoang-Lam Le `[一作]` (Kyoto Institute of Technology), Shohei Nobuhara `[通讯]` (Kyoto Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在本文中，作者通过在一次扫描中使用时序编码，实时校准激光扫描投影仪的时序映射，并在同一次录制中完成投影仪与事件相机的几何与时钟自标定，从而实现单相机、单投影仪的全景稠密深度重建；

**💡 创新点**

其创新点在于：①不需要预先测量投影仪时序或额外参考物体，直接从编码事件中推断投影像素对应；②将投影仪的时间映射建模为二维三次多项式，并通过同一次录制数据完成校准；③提出基于时空自标定的算法，既恢复几何外参又同步时钟，极大降低了传统多视角结构光系统的硬件与软件复杂度；

**🔧 技术方法**

核心技术包括事件相机的稠密光流捕获、时序编码与解码、投影仪时间映射的多项式拟合、基于多视角几何的闭式解与束束优化、以及时空自标定框架；

**📊 数据集**

实验数据集为作者自行收集的8组录制，使用Ultimems HD309D1-C激光扫描投影仪和CenturyArks SilkyEvCam HD事件相机，在不同场景（平墙、盒子、球体、机器人、面部与手部）下获取多张深度重建结果；

**📈 对比分析**

与基准的双相机重建和理想栅格假设的结果对比，校准后的时序映射在深度误差上平均提升约2–3倍（平面误差从约8.9mm降至3.1mm，球体误差从1.72mm降至1.39mm），并在整个视场内显著减少结构化误差；

**⚠️ 局限性**

局限性包括：事件相机的时间戳仅能确定扫描线段而非单像素，导致需要更大时间窗口；投影仪扫描的风扇形畸变仍需复杂多项式建模，可能在不同设备上泛化受限；自标定所需的前置编码块时长（约2.7–5.4s）对实时性有一定影响；此外，方法假设场景静止，无法直接处理高速运动物体。

---

## 78. Degradation of Language Models under Controlled Load: Synchronization, Rule Execution and Misfires on the Background

**arXiv ID:** 2611.27404 | [PDF](https://arxiv.org/pdf/2611.27404v1)

**作者:** Hennadii Kutomanov `[一作]` `[通讯]` (Independent researcher), Hennadii Kutomanov (Independent researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对语言模型在负载增加时产生的错误类型（同步失效、规则执行错误、误发）进行分离，并构建了“passport”方法用于描述各模型在不同负载轴上的退化规律。

**💡 创新点**

创新点：①将错误拆分为三类并单独建模；②采用序列对齐技术精确识别每个词的输出状态；③引入异质随机截距GLMM捕捉请求间难度分布随负载变化的影响；④通过平滑法律与QAIC选择实现可解释的退化函数；⑤给出容量度量（规则执行容量、疲劳容量、密度效应因子），首次实现多模型的可比对比。

**🔧 技术方法**

技术方法：动态规划序列对齐、误差统计（MER）、三量统计模型（同步、规则执行、误发）、GLMM（异质随机截距）、分层平滑法律（含τ²权重）、QAIC与显著性检验、随机抽样检查、Python+PyTorch实现。

**📊 数据集**

数据集：随机无意义单词文本与一系列可控替换规则，生成多种负载条件（文本长度、替换密度、规则数、词位、规则位置）；使用7个模型（GPT‑4o mini、GPT‑4.1 mini、Gemini 3.5 Flash Lite、Gemini 3.1 Flash Lite、Nemotron 3.5 Lightning、Nemotron 3 Nano Omni、Nemotron 3 Ultra）通过对应 API 产生答案，形成约 3 700–7 000 个实验点。

**📈 对比分析**

比较方法：将各模型的三量退化规律绘制成 passport，计算规则执行容量、同步/规则执行疲劳容量以及误发密度因子；结果显示模型间规则执行容量从10到36条规则不等，疲劳容量从数百到数十万词不等，误发因子差异可达10¹²；模型在不同负载轴（长度、密度、规则数、位置）上的表现可通过 passport 直接可视化和定量比较。

**⚠️ 局限性**

限制：①实验文本为随机无意义词，缺乏真实语言语境；②仅评估规则替换任务，无法覆盖其他任务类型；③部分模型的测点未公开，限制了复现性；④异质随机截距模型收敛问题导致部分量采用单一方差；⑤未对词级细粒度行为机制（邻近词影响、复制机制、替换选择）进行深入分析。

---

## 79. Opto-Electronic Hybrid Reinforcement Learning Based on MZI for Robotic Manipulator Control

**arXiv ID:** 2611.27129 | [PDF](https://arxiv.org/pdf/2611.27129v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 80. When Students Fail to Learn from Teachers: Learning Dynamics of On-Policy Distillation

**arXiv ID:** 2611.27125 | [PDF](https://arxiv.org/pdf/2611.27125v1)

**作者:** Xinnan Zhang `[一作]` (University of Minnesota), Mingyi Hong `[通讯]` (University of Minnesota)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究OPD学习动力学，发现教师-学生不匹配导致不稳定，并提出状态相关KL路由提升性能。

**💡 创新点**

创新点是将教师-学生覆盖度量与自适应前向KL路由结合，解决采样不匹配和跨位置梯度问题。

**🔧 技术方法**

采用软max分布解析、采样‑token OPD、前向KL、状态依赖KL路由、SFT初始化等技术。

**📊 数据集**

使用数学推理数据集DAPO、AIME24-26、AMC23，以及代码生成数据集HumanEval+、MBPP+、LiveCodeBench v6。

**📈 对比分析**

与采样OPD、EOPD、AOPD比较，在Qwen3-1.7B/4B/8B模型上平均Avg@8提升约2–6个百分点，Best@8提升约1–6个百分点。

**⚠️ 局限性**

局限在于需手工设定阈值，某些规模下Best@8略有下降，跨任务通用性尚未充分验证。

---

## 81. Partially canonical axes in large language model representations

**arXiv ID:** 2611.26920 | [PDF](https://arxiv.org/pdf/2611.26920v1)

**作者:** Andrea Gregor de Varda `[一作]` (Massachusetts Institute of Technology), Meenakshi Khosla `[通讯]` (University of California San Diego)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对12个大型语言模型的隐藏表示基底进行轴匹配与旋转分析，比较其与其他模型、脑fMRI以及可解释语言特征的对齐程度。

**💡 创新点**

证明LLM的本地坐标轴并非任意，而是“部分规范化”——在保持几何的前提下本地轴更能对齐外部系统，并揭示训练过程使轴呈现稀疏读取特征。

**🔧 技术方法**

使用轴到轴匹配、正交旋转、梯度学习最佳旋转、岭回归稀疏度评估以及训练动态跟踪等技术。

**📊 数据集**

使用12个LLM（GPT‑2、LLaMA‑3.2、Gemma‑3、Mamba、Qwen‑3、Pythia）以及Pereira fMRI实验2/3的句子、Wug、BLiMP、超纲层级、感知与情感规范数据集。

**📈 对比分析**

通过比较native、随机旋转和最优旋转的轴匹配相关系数，发现native优于随机但远低于最优；对齐随α递增显著下降，统计检验均显著。

**⚠️ 局限性**

结果局限于残差流、Transformer/状态空间模型，未检验多任务学习的其他层级；轴匹配仅捕捉线性对齐，未揭示潜在的非线性结构。

---

## 82. Improving the Optimisation Landscape for Particle Mixtures

**arXiv ID:** 2611.26931 | [PDF](https://arxiv.org/pdf/2611.26931v1)

**作者:** Yan Lin `[一作]` (Newcastle University), Chris J. Oates `[通讯]` (Newcastle University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在有限粒子混合模型中直接最小化混合风险导致的梯度训练信号枯竭问题，并提出了通过引入线性桥接正则化（即混合风险与单体风险的凸组合）来改进优化景观的解决方案。

**💡 创新点**

创新点在于首次揭示了混合风险在粒子层面缺乏梯度传递导致的“责任枯竭”现象，并证明了通过在混合风险中加入逆KL惩罚（bridge）可为所有粒子提供最小训练信号，从而实现对PrO后验与集成模型的有效优化。

**🔧 技术方法**

技术包括粒子梯度流（如MFLD）、噪声AdamW优化、对责任分布的逆KL正则化、以及对PrO后验的β-加热贝叶斯先验的理论推导。

**📊 数据集**

使用的数据集包括：1）二分类的合成逻辑回归数据（用于PrO后验实验）；2）四个大型语言模型（Llama 3.2-1B/3B，Qwen3-4B-Base，Qwen2-1.5B-Instruct）以及八个多项选择推理基准（ARC Challenge/ARC Easy、BoolQ、HellaSwag、OpenBookQA、PIQA、Social IQA、WinoGrande）。

**📈 对比分析**

通过与纯混合风险、独立训练和α‑Rényi方法的对比，实验表明在PrO后验中β=0.01即可显著提高收敛速度和预测校准；在LLM集成训练中，β>0的桥接显著降低训练集与测试集的混合风险、提升准确率并减轻责任枯竭，β值越小越能保留PrO后验的优势。

**⚠️ 局限性**

局限性包括：1）理论证明仅针对单个样本的梯度信号，未给出全局收敛保证；2）PrO后验实验仅在低维合成任务上验证，未扩展到更高维真实模型；3）LLM集成实验每种配置仅训练一次，可能受模型预训练与指令调优差异影响；4）未给出最佳β的自适应选择或从零到非零的退火策略。

---

## 83. Self-similar Organization of Information in Language

**arXiv ID:** 2611.26935 | [PDF](https://arxiv.org/pdf/2611.26935v1)

**作者:** Leon Lufkin `[一作]` (Stanford University), Kamesh Krishnamurthy `[通讯]` (Goodfire)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一个以预测信息自相似性为核心的理论框架，并通过开放权重大型语言模型对长文本的代码长度进行估计，证明自然语言在不同尺度下呈现幂律增长与衰减的自相似结构；

**💡 创新点**

创新点在于将自相似性概念与信息量度量（如band-gain、I_pred）相结合，形成了一套可量化文本结构的方法，并首次利用LLM估计器在千字级以上文本上验证了自相似性假设；

**🔧 技术方法**

技术上主要采用Qwen3.5-9B、Qwen3.5-35B-A3B、Llama-3.1-8B、Llama-3.1-70B等开放权重LLM估计条件熵/代码长度，计算互信息与band-gain，并进行尺度缩放坍塌与扰动实验；

**📊 数据集**

使用的文本数据集包括哈佛数字化图书、OlmOCR科学PDF、Python/Rust代码仓库以及Lean证明集，均以单篇长文本形式统一处理；

**📈 对比分析**

实验对不同模型、不同窗口大小下的I_pred与Δ进行幂律拟合，发现指数≈0.6-0.66，并通过尺度坍塌验证自相似性；扰动实验显示结构被破坏时指数显著下降，表明方法对结构敏感；

**⚠️ 局限性**

局限性包括依赖LLM估计器的校准不保证真实互信息、仅对英文及少数语言进行验证、未涵盖低参数模型以及扰动实验仅采用块级置换，无法探究更细粒度结构破坏的影响。

---

## 84. When Perception Collides with Priors: Diagnosing Modal Polarization in Video Understanding Models

**arXiv ID:** 2611.26972 | [PDF](https://arxiv.org/pdf/2611.26972v1)

**作者:** Yao Tong `[一作]` (Hangzhou Institute for Advanced Study), Jianqiang Huang `[通讯]` (Computer Network Information Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了多模态大语言模型在视觉与先验知识冲突时出现的 Modal Polarization 现象，并基于此构建了专门评测该失调的 TheaterBench 基准。

**💡 创新点**

创新点在于首次将认知惯性与感知惯性两极失调进行系统定义，并设计了“夹心”时间结构与多模态问答评测，揭示模型在逆常识视频中的双向失衡。

**🔧 技术方法**

技术方法包括利用多代理自动化流水线（LLM、Image‑VLM、VGM）生成脚本、视频和问答；并通过 MCS、MFS、MPI 等指标对模型的感知‑认知融合能力进行量化。

**📊 数据集**

数据集为 TheaterBench，共 984 条 8–10 秒视频，来源于 400+ 对象的 McGuffin 库，涵盖正常与超现实的“夹心”情景，并配有三类多选问答。

**📈 对比分析**

通过对比闭源与开源 SOTA 模型在 Perception、Cognition、P&C 测试中的准确率、MCS、MFS 和 MPI，发现闭源 Gemini 3 Flash 仍显优越，但多数模型均表现出明显的 Modal Polarization，MPI 正负两极分布，且人类基线为 100 分。

**⚠️ 局限性**

局限性在于基准仅覆盖 VGM 生成的超现实情景，未涵盖真实世界的模态冲突；此外，本文未提出有效的通用缓解策略，缺乏针对性解决方案。

---

## 85. SonoAtlasWorld: Anatomy-Grounded Ultrasound World Models for Wide-Range Scanning

**arXiv ID:** 2611.26867 | [PDF](https://arxiv.org/pdf/2611.26867v1)

**作者:** Zheyuan Zhang `[一作]` (United Imaging Intelligence), Shanhui Sun `[通讯]` (United Imaging Intelligence)

**通讯引用:** 1645 | [OpenAlex ID](https://openalex.org/A5087112236)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了SonoAtlasWorld，一个实时可交互的超声世界模型，在扫描过程中保持持久的3D解剖状态，并通过探头动作生成超声图像。

**💡 创新点**

创新点在于将解剖结构作为持久世界状态，将探头动作映射为可见性条件，并结合三阶段训练（解剖条件预训练、因果转换与长期滚动校准），实现解剖一致的因果生成与长周期稳定。

**🔧 技术方法**

核心技术包括基于扩散Transformer的因果观测生成器、解剖条件分支、流匹配与分布匹配蒸馏、长滚动对齐，以及CUDA图和KV缓存加速。

**📊 数据集**

使用了82,158条腹部B模式视频和493个CT解剖资产（453个训练、40个测试），并整合了AbdomenCT-1K、AbdomenAtlas和Merlin三维数据集。

**📈 对比分析**

与SonoWorld、UltraWorld、SonoGym等基于动作或几何条件的方法相比，SonoAtlasWorld在解剖动作追踪的AFI和宏观器官Dice分别提升至0.428和0.262，长滚动模型在40个循环中的亮度漂移降低3.46倍，推理速度达到81.2 FPS。

**⚠️ 局限性**

主要局限包括使用静态CT资产无法建模探头诱发的组织变形和生理运动、CT与超声的域差距及分割器依赖的解剖评估仍待进一步研究，以及仅在仿真中验证策略，未在真实机器人上转移。

---

## 86. PathOPEN: A pathologist-curated histopathology dataset for visual question answering

**arXiv ID:** 2611.27469 | [PDF](https://arxiv.org/pdf/2611.27469v1)

**作者:** Awais Naeem `[一作]` (University of Texas at Austin), Ying Ding `[通讯]` (University of Texas at Austin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并公开了 PathOPEN 数据集，包含 157 个病例、229 张原始病理图像及其 30 倍增强版本，附有开放式、单选和是非三种格式的问题与答案，并对数据质量和模型性能进行评估。

**💡 创新点**

①首次将病理图像的组织系统、病理分类、区域解剖和放大倍率等元数据与问题-答案配对；②提供正确与错误答案对，支持偏好学习和难度分析；③引入四个平衡 manifest，按元数据字段对增强样本进行类别平衡。

**🔧 技术方法**

利用多种视觉‑语言模型（Qwen、InternVL、MedGemma、LLaVA‑Med、Quilt‑LLaVA、Patho‑R1）进行 zero‑shot 问答；使用基于 rubrics 的人工评估（五位病理学家和两位 VLM 判断者）对答案质量、图像定位、错误难度等进行评分；对增强图像进行临床相关性和视觉定位评估。

**📊 数据集**

PathOPEN 本身；与 PathVQA、Quilt‑VQA、PathMMU、PatchVQA 等公开病理 VQA 数据集做对比，后者经过 H&E 过滤后得到 612 条开放式和 50 条是非题。

**📈 对比分析**

通过 5 位病理学家和 2 位 VLM 判断者在 5 个 benchmark 上打分，并将模型答案与参考答案按 token‑F1、BERTScore、准确率（含对多选的纠偏）进行比较。结果显示 PathOPEN 在答案质量上优于 PathVQA，模型在多选题上平均准确率为 42.4%（对 5 选项），是非题约 63% 正确率，开放式答案 F1 最高者为 Patho‑R1。

**⚠️ 局限性**

①数据来源单一机构，病例数量有限，稀有类别样本不足；②多数评分集中在最高分，导致 κ 统计失真；③增强图像在视觉定位上评分仍有偏差；④模型在不同问题格式间排名不一致，难以统一评价；⑤缺乏对模型训练使用该数据集的实验评估。

---

## 87. Spatial Query Execution as Verifiable Rewards for Multi-step Spatial Reasoning

**arXiv ID:** 2611.26787 | [PDF](https://arxiv.org/pdf/2611.26787v1)

**作者:** Jiyoon Pyo `[一作]` (University of Minnesota), Yao-Yi Chiang `[通讯]` (University of Minnesota)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 FARON 框架，利用合成 2D/3D 场景执行可查询空间关系，自动生成精确的推理轨迹与可验证奖励，进而对大型视觉‑语言模型（LVLM）进行过程监督（SFT）和强化学习（RLVR）。

**💡 创新点**

创新点在于通过可执行的空间查询一次性提供中间推理监督与最终奖励，形成统一、无人工标注的自动化训练流程，显著提升多步空间推理能力。

**🔧 技术方法**

技术包括：合成场景生成、基于对象几何的空间查询与 SQL 执行计划、自然语言推理轨迹模板化、Qwen3 进行语义多样化、过程监督的 SFT、基于可验证奖励的 RLVR 及 GRPO 策略优化。

**📊 数据集**

数据集主要为自动生成的 23,800 条问题（2,199 条 2D 场景 + 1,256 条 3D 场景）用于 SFT，1,736 条用于 RLVR，内部评测 8,445 条，外部迁移至 5 个空间推理基准（SpatialEval、FRIEDA、CV‑Bench、BLINK、MultiHopSpatial）。

**📈 对比分析**

与仅基于答案的监督相比，-Reason 推理轨迹监督在内部测试中整体提升约 7.6%，5 步推理提升 10.3%；GRPO 在多步推理上再提升约 4%。在外部基准上，-Reason+GRPO 在大多数基准（尤其是 SpatialEval）超越原始模型 3‑10% 左右，证明迁移效果显著。

**⚠️ 局限性**

局限性包括：实验仅在简化的合成场景上进行，缺乏真实视觉多样性与语义复杂性；3D 视角仍以相机相对方式近似；未独立验证 Qwen3 生成的推理轨迹语义一致性；视觉感知错误仍是主要瓶颈。

---

## 88. Explain Your Code: AI-Based Examination at Class Scale

**arXiv ID:** 2611.26836 | [PDF](https://arxiv.org/pdf/2611.26836v1)

**作者:** David Sarne `[一作]` (Bar-Ilan University), Yossi Ben-Zion `[通讯]` (Bar-Ilan University)

**通讯引用:** 357 | [OpenAlex ID](https://openalex.org/A5115091625)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在巴伊兰大学操作系统课程中部署了三代理自动化口试系统（ATC），对247名学生的546次口试进行评估，生成口试分数与代码质量分数并行，并与学生GPA进行关联分析。

**💡 创新点**

创新点在于将代码本身作为提问依据、采用自适应对话生成问题、并通过三个相互隔离的语言模型（examiner、code‑reviewer、grader）完成全流程评估，同时将作者身份与成绩分离。

**🔧 技术方法**

使用大型语言模型Claude Opus 4.6（主推）、Claude Sonnet 4.6、Google Gemini 作为后备，并通过OpenAI/Claude API实现问答、代码评估与评分。

**📊 数据集**

实验数据来自Bar‑Ilan大学入门操作系统课程的247名学生，共3项练习、2种考试语言、546个考场记录，包含学生提交的源代码、自动化测试分数和GPA。

**📈 对比分析**

通过预设的5项成功标准与Spearman相关系数、置信区间进行比较，结果显示口试分数与GPA相关（ρ≈0.42）而与代码质量无关，自动化测试在90分以上达成“天花板”，系统平均成本≤$1.37/次。

**⚠️ 局限性**

局限性包括缺乏人工口试对照、自动化测试分数受限于天花板、对不同课程/语言模型的泛化性尚未验证，且对AI使用与作者身份的判定仍依赖人工复核。

---

## 89. Mechanism Parity for Simulation-Based Validation of Autonomous-Driving Perception

**arXiv ID:** 2611.27447 | [PDF](https://arxiv.org/pdf/2611.27447v1)

**作者:** Danial Safaei `[一作]` (University of Warwick), Xingyu Zhao `[通讯]` (University of Warwick)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现一种统一的机制平行评估框架（DFF 与 C-DFF），用于验证仿真生成的无人驾驶感知模型是否在真实与合成场景下使用相同的决定性证据。

**💡 创新点**

创新点在于将决策特征完整性（DFF）与语义概念级决策特征完整性（C-DFF）结合成一套可审计的诊断体系，并通过正向控制实验验证该指标对“右输出但错误原因”情形的敏感性。

**🔧 技术方法**

使用基于 SDXL 的 ControlNet 生成器、对冻结的感知模型（steering、YOLOP–DA、YOLOP–LL）进行对抗式解释器（CF‑XAI）来生成决定性特征图，随后在 16×16 网格上聚合至语义概念空间。

**📊 数据集**

数据集为 96 对严格对应的 KITTI–Virtual KITTI 2 的精确几何图像样本，涵盖三场景；同时在两套保留场景上进行内部复现检验。

**📈 对比分析**

对比方法包括：图像相似度（MAE/PSNR/SSIM）、输出一致性（steering、drivable‑area、lane‑line 的 IoU/Chamfer/F1），以及 DFF/C‑DFF 的均方误差；实验显示输出一致性与视觉相似度与机制距离几乎无关，且基于机制的校准循环未能显著提升机制一致性。

**⚠️ 局限性**

局限性包括：仅在单目、对偶场景的精确几何配对下验证；评估仅涉及三种冻结感知模型和十个重训练的生成器；缺乏多传感器、时序或规划层面的验证；未能提供安全性认证或跨域推广的证据。

---

## 90. How Long Will My Language Model Run? A Benchmark for Pre-Execution Runtime Prediction

**arXiv ID:** 2611.26875 | [PDF](https://arxiv.org/pdf/2611.26875v1)

**作者:** Ashish Molakalapalli `[一作]` (Texas A&M University), Ruihong Huang `[通讯]` (Texas A&M University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个用于在深度学习作业提交前预测训练与推理运行时的基准，并收集了历史记录与受控生成数据。

**💡 创新点**

提出了明确的时间测量目标、结合大语言模型与回归方法的多种预测器，并系统评估其在不同工作负载与硬件上的表现。

**🔧 技术方法**

使用大语言模型（Qwen3.5‑9B、GPT‑OSS‑120B、Claude Opus 5.5）进行零样本和5样本推断；对比传统回归模型（OLS、Ridge、Random Forest、LightGBM）、最近邻查找和基于Roofline的分析估计。

**📊 数据集**

历史集：801条训练、1884条推理记录；受控集：64条文本生成配置（Qwen2.5‑0.5B 与 T5‑small）在单台A100上多次测量；两者均提供完整的工作负载配置与计时目标。

**📈 对比分析**

按MAE、MAPE、within‑factor‑two准确率等指标分别在历史开发集和受控留出集进行评估；结果显示回归模型往往优于LLM，LLM在加入示例后性能提升显著但仍不及最优回归模型；在受控生成上Random Forest的MAE最低，LightGBM的MAPE最低。

**⚠️ 局限性**

受控生成仅覆盖两种小模型和单一GPU，使用合成输入；历史集来自单台集群，未完全代表所有提交工作；LLM评估受版本、示例选择、推理预算等限制；缺乏对自然提示和更广泛硬件的验证。

---

## 91. Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory

**arXiv ID:** 2611.26797 | [PDF](https://arxiv.org/pdf/2611.26797v1)

**作者:** Bao Pham `[一作]` `[通讯]`, Bao Pham

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

暂无论文内容

**💡 创新点**

暂无论文内容

**🔧 技术方法**

暂无论文内容

**📊 数据集**

暂无论文内容

**📈 对比分析**

暂无论文内容

**⚠️ 局限性**

暂无论文内容

---

## 92. Embodied Interaction Policy: A Unified Model for Full-Duplex Human-Robot Interaction

**arXiv ID:** 2611.27327 | [PDF](https://arxiv.org/pdf/2611.27327v1)

**作者:** Youqiang Gui `[一作]` (Sichuan University), Yu Li `[通讯]` (International Digital Economy Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了统一流式的 Embodied Interaction Policy（EIP），在 400 ms 的时间片内将预训练的交互专家与运动专家耦合，实现机器人在对话中的语音与连续、情感感知的物理动作同步。

**💡 创新点**

创新点包括：① 将冻结的交互专家隐藏状态直接作为运动上下文；② 通过分层联合注意力实现交互-运动的即时耦合；③ 引入 chunk‑wise 因果掩码和发声标记保证时序一致与情感驱动；④ 采用两阶段训练提升状态适应与情感表情；⑤ 构造 EI‑Data 数据集专门用于训练与评估。

**🔧 技术方法**

技术方法包括：预训练的 Audio‑Interaction 模型（基于 Qwen2.5‑3B）；流式 Transformer 运动专家，采用 flow‑matching 目标；Mixture‑of‑Transformers 层耦合；FiLM 情感条件；增量 TTS 与解码/提交缓存机制；在线轨迹优化（MPC+Ruckig）保证安全与平滑。

**📊 数据集**

使用了 EI‑Data（融合 BEAT2、GUAVA、SHOW 的共说数据与 Seamless Interaction 的双人对话），约 60 k 条 20 s 的交互片段；并在 Astribot S1 机器人与 SMPL‑X 骨架上进行训练与评估。

**📈 对比分析**

与 MIBURI、EMAGE、Photoreal、RoboGesture 等基线进行比较，评估指标为 Fréchet Gesture Distance（FGD）、Beat Consistency（BC）与 Diversity（Div）。EIP 在 Astribot 上取得 FGD = 43.83、BC = 0.744、Div = 1.021，显著优于所有基线；在用户研究中 Motion Naturalness、Emotion Expressiveness、Gesture Diversity 等得分提升至 6+；在成功率、延迟与 RTF 方面，EIP 成功率 91.7%、延迟 1.35 s、RTF 0.804，明显优于 Cascade（76.7%/1.99 s/0.952）和 RoboGesture（18.3%/3.78 s/0.821）。

**⚠️ 局限性**

局限性在于仅使用音频感知，无法解析视觉线索（如注视、面部表情、身体姿势），导致对非语言上下文的理解不足，未来需要整合视觉感知以提升整体交互质量。

---

## 93. AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance

**arXiv ID:** 2611.27369 | [PDF](https://arxiv.org/pdf/2611.27369v1)

**作者:** Hwee Young Kang `[一作]` (DSO National Laboratories), Hai Leong Chieu `[通讯]` (DSO National Laboratories)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

设计了一种训练‑free 的模型合并框架 AMARIS，通过激活协方差决定子空间继承，将专家模型的权重更新投影到低秩子空间，得到单一静态合并检查点。

**💡 创新点**

将合并视为带预算的子空间继承问题，利用基于协方差的泛化特征值排序和贪心预算选择，显式控制专家更新与基模型干扰。

**🔧 技术方法**

协方差估计、线性层投影、特征值分解、贪心子空间选择、低秩投影门控制、基于校准集的激活统计等。

**📊 数据集**

使用 Qwen3‑8B/32B 与各领域专家模型的校准集，包括金融问答、指令跟随、SEA 多语种（印度尼西亚语、越南语、泰语、塔米尔语、他加禄语）以及相应评测集 AIME26、LCB_v6、IFEval、GPQA:Diamond、FPB、IFBench、SEAHELM。

**📈 对比分析**

与多种训练‑free 合并基线（线性插值、SLERP、TIES、DARE‑TIES、RegMean、LEWIS 等）以及少量微调方法对比，AMARIS 在核心能力保留和专家性能上达到或接近 Pareto 前沿，在财务、指令跟随和 SEA 三种场景中保持至少 90% 专家能力和 95% 基模型核心能力，部分案例（如 SEA 的塔米尔语）显著提升。

**⚠️ 局限性**

依赖校准激活的估计，需要精心选择预算 α 和 β；仅支持同一架构、tokenizer 的配对合并，无法处理异构或多专家的逐步合并；对校准集分布变化敏感，且不保证完全专家迁移。

---

## 94. FoRe: A Force-Guided Framework for Autonomous Grasp Recovery in Vision-Language-Action Models

**arXiv ID:** 2611.27011 | [PDF](https://arxiv.org/pdf/2611.27011v1)

**作者:** Shuai Peng `[一作]`, Jian Huang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了FoRe框架，用力反馈实现VLA模型的自主抓取失败恢复

**💡 创新点**

创新点在于：①将演示划分为抓取与运输阶段，分别训练阶段专属策略；②以抓取阶段策略为初始值，在失败后状态上做局部强化学习优化重试策略；③在推理时通过力信号和抓取指令状态实现策略切换和失败检测

**🔧 技术方法**

采用了视觉-语言-动作（VLA）模型、力传感器反馈、局部强化学习（RL）以及基于力的策略切换控制器

**📊 数据集**

使用了LIBERO-SUCTION（改造后的LIBERO带吸盘与力感知）、原始LIBERO以及真实机器人实验数据

**📈 对比分析**

与多种VLA基线（OpenVLA、π_0.5等）和CycleVLA对比，FoRe显著提升任务成功率：在LIBERO-SUCTION上VLA-Adapter从69.2%提升到87.6%（+18.4%），π_0.5从85.9%提升到92.4%（+6.5%）；在LIBERO上平均成功率提升至98.6%/98.3%；在真实机器人上成功率从26.7%/48.3%提升到51.7%/70.0%

**⚠️ 局限性**

依赖于抓取阶段策略成功接触目标物体；若该策略始终无法接触，力反馈无效，FoRe无法触发重试，限制了自主恢复能力

---

## 95. A Sparsity Analysis of Traffic Forecasting Architectures: How Much Missing Data is Too Much?

**arXiv ID:** 2611.26847 | [PDF](https://arxiv.org/pdf/2611.26847v1)

**作者:** Mohamed Hamad `[一作]` (Qatar University), Halim Yanikomeroglu `[通讯]` (Carleton University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对多模态交通预测模型在推理阶段传感器稀疏条件下的鲁棒性进行系统评估。

**💡 创新点**

提出统一的稀疏性评估框架、归一化车辆误差、可用性阈值和加权平均方法，首次全面比较不同架构在多通道数据下的稳健性。

**🔧 技术方法**

使用图神经网络、Transformer、动态GNN、状态空间模型（Mamba）以及随机掩码和z-score归一化的评估指标。

**📊 数据集**

采用PEMS04和PEMS08两套多通道（速度、流量、占用率）的交通监测数据。

**📈 对比分析**

通过不同稀疏比例下的vMAE/vRMSE、加权平均、相对降解和可用性阈值进行对比，结果显示峰值精度最高的D2STGNN在稀疏时最不稳；Trafformer相对稳健；Mamba4Traffic在中等稀疏时表现最优。

**⚠️ 局限性**

仅考虑随机均匀掩码，未探讨时空相关故障、未将稀疏融入训练，且仅在两套数据集上验证，缺乏对不同传感器网络条件的泛化评估。

---

## 96. GRADPlace: Geometry-Based Routing-Augmented Differentiable Placement for Routability Optimization

**arXiv ID:** 2611.27287 | [PDF](https://arxiv.org/pdf/2611.27287v1)

**作者:** Xurong Shen `[一作]` (ShanghaiTech University), Biwei Xie `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于概率加权的可移动方向段的路由增强分析放置框架GRADPlace，用于提升放置的可路由性

**💡 创新点**

创新点在于将路由反馈的路径概率与可变方向段几何相结合，实时重新计算期望路由需求并通过电磁场梯度传递给放置节点；与传统只使用细胞膨胀或冻结路由信息的方法相比，大幅降低DRV

**🔧 技术方法**

使用概率路径模型、移动方向段几何、期望需求与电磁拥塞场、梯度反向传播、GGR全局路由反馈、细胞膨胀、填充器调整以及DREAMPlace求解器

**📊 数据集**

在ISPD 2015基准套件的20个设计上进行实验

**📈 对比分析**

与Xplace‑route和DCGP进行对比，GRADPlace在设计规则违规数上平均降低约3.3×、通道VI数更少、布线长度保持相当；整体放置‑路由时间在多数设计中下降，整体效率提升

**⚠️ 局限性**

局限包括仅使用H/V单弯候选模式、固定概率更新周期、单随机种子、仅评估一套基准与路由流；未来工作需引入多弯候选、适应性更新、时序耦合等

---

## 97. Hindsight Forcing: Learning Better History from the Future in Autoregressive Video Diffusion

**arXiv ID:** 2611.27074 | [PDF](https://arxiv.org/pdf/2611.27074v1)

**作者:** Haonan An `[一作]` (City University of Hong Kong), Yuguang Fang `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种在自回归视频扩散中通过“Hindsight Forcing”实现历史生成与上下文写入的联合训练。

**💡 创新点**

创新点在于通过隐式直通桥（latent straight‑through bridge）将前向的完整历史回放与后向的梯度回传分离，使得未来生成的损失能够直接指导历史 denoiser 的学习，而无需完整 BPTT 或额外参数。

**🔧 技术方法**

使用扩散变换器（Diffusion Transformers）、Distribution Matching Distillation、KV 缓存以及直通梯度技术；实现了少步自回归视频生成。

**📊 数据集**

训练使用 VidProM 过滤的 248K 文本提示；评估使用 VBench 视频基准。

**📈 对比分析**

与 CausVid、Self Forcing、Causal Forcing、Reward Forcing、RAVEN、SGF、SGF+ 等方法对比，Hindsight Forcing 在 VBench Total、Quality、Semantic、Dynamic Degree 上均为最高（85.74、86.89、81.11、92.22），并保持了与原始推理流程相同的推理速度。

**⚠️ 局限性**

局限包括梯度缩放因子 α 仍为固定常数，尚未探索自适应调节；虽然显著降低了存储需求，但训练仍随块数线性增长；对于长时延的流式生成，如何在受限 KV 缓存下保持性能仍待进一步研究。

---

## 98. Compiling Existential-Positive Queries to Bounded-Variable Fragments

**arXiv ID:** 2611.26869 | [PDF](https://arxiv.org/pdf/2611.26869v1)

**作者:** Christoph Berkholz `[一作]` (Humboldt University of Berlin), Hubie Chen `[通讯]` (King's College London)

**通讯引用:** 1597 | [OpenAlex ID](https://openalex.org/A5110074953)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在将存在性正向一阶查询（仅使用∃、∧、∨）压缩为变量数最小的边界变量子句时，公式大小可能出现指数级膨胀，并对不同符号结构给出了是否可避免指数膨胀的判定；

**💡 创新点**

首次给出针对存在性正向查询的无条件指数下界，并完整描述何种签名和变量数下能实现多项式大小压缩，此外通过与电路复杂度（单调布尔电路）建立新接口，证明了压缩的不可避免性；

**🔧 技术方法**

使用了正则化变换、DNF化、变量重命名、结构同态与电路转换、单调电路下界（k‑clique）以及参数化复杂度框架（poly‑compilation、参数化归约）等技术；

**📊 数据集**

本工作为理论研究，无实验数据集；

**📈 对比分析**

通过上界（最大指数）与下界（指数下界）相匹配，对签名与变量数进行分类，证明在特定条件下可实现多项式压缩，反之则必须指数级；

**⚠️ 局限性**

仅适用于存在性正向子句；对更一般的一阶逻辑或含否定/全称量词的查询不适用；结果依赖于符号签名的结构，若签名包含多于一个非原子符号则必须指数扩张，限制了实际应用范围。

---

## 99. A preference for local interaction between layers emerges during training

**arXiv ID:** 2611.27258 | [PDF](https://arxiv.org/pdf/2611.27258v1)

**作者:** Jiheon Lee `[一作]` `[通讯]` (Korea Institute for Advanced Study), Jiheon Lee (Korea Institute for Advanced Study)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过对每层进行扰动（push）并跟踪其在训练中的传播，测量深度神经网络层与层之间的学习互相影响。

**💡 创新点**

发现深度网络在训练过程中自发形成局部（邻近层之间）交互，而非全局影响，并揭示这一局部化与层间最强模式对齐的关系。

**🔧 技术方法**

采用5层ReLU全连接网络、AdamW和SGLD两种优化器、对权重的稳定秩、梯度法推导与实测互响应、推断线性对齐规律。

**📊 数据集**

在模数运算任务(a+b mod 113)上训练，使用训练集40%的样本，测试集剩余60%。

**📈 对比分析**

通过将扰动后的复制网络与未扰动网络在相同条件下训练并比较稳定秩变化来量化互响应；结果显示训练后邻近层互响应显著高于远层，几乎所有网络均符合该规律，验证了局部化现象。

**⚠️ 局限性**

实验仅限于五层小型网络和特定算术任务，深层网络和不同架构的普适性未验证；此外推断的对齐机制在SGLD中表现不完整，说明仍需进一步理论与实验扩展。

---

## 100. Integral Multipliers and Arithmetic Obstructions for Two-Hidden-Layer ReLU Maxima

**arXiv ID:** 2611.27018 | [PDF](https://arxiv.org/pdf/2611.27018v1)

**作者:** Pengcheng Cheng `[一作]` `[通讯]`, Pengcheng Cheng

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

研究了使用两层隐藏层且参数为整数的ReLU网络在精确表达最大函数（max）时所需的最小整数乘子；通过整数立方体填充、凸体同构及同调论证，确定了六输入最大函数的乘子为2，七输入最大函数乘子为2或6，并给出了构造网络与整数填充的具体实现；

**💡 创新点**

首次将整数参数的ReLU网络与格点多面体同调、整数链复形及多面体填充等离散几何工具结合，提出“坐标-面复形指数”与“相对核形式”两种新的理论框架，用以判定整数乘子的可行性，并揭示了七输入问题与三的阿廷分支的关系；

**🔧 技术方法**

使用整数链复形、配分理论、正交投影、参数化的立方体细分、格点多面体同调、有限域特征法、数值补全与证明脚本等组合技术；

**📊 数据集**

未使用传统机器学习数据集，全部工作基于符号计算与整数几何构造，构造网络权重与填充链的具体数值记录在压缩档案 max7_arithmetic_revised.zip 中；

**📈 对比分析**

与已有的整数/有理表达式（如Schroeder、Myrisiotis等）的比较表明，本文提供了更小的整数乘子（六输入为2，七输入至多6），并通过验证程序确认了所有代数等式的正确性；

**⚠️ 局限性**

对七输入最大函数仅给出了乘子可能为2或6的区间，具体取值取决于一个尚未决定的同调类的消失与否；若该类不消失，则最小乘子为6，否则为2，尚缺全局判定方法。

---

## 101. Towards Text-Mask Consistency in Medical Image Segmentation

**arXiv ID:** 2611.27050 | [PDF](https://arxiv.org/pdf/2611.27050v1)

**作者:** Jie Gui `[一作]` (Anhui University), Xiuquan Du `[通讯]` (Anhui University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出了一种两阶段医学影像分割框架C2Seg，旨在提升文本与掩膜的一致性，首先通过聚类感知对比学习（CaCL）利用冻结的CLIP文本编码器构建软标签，缓解模板化文本导致的误负样本；随后在融合阶段引入双向互补注意模块（BCAM）和KAN注意力门控（K‑Gate），实现视觉与语言在空间结构上的深度交互与细粒度特征选择，最终得到更准确的病灶分割结果。

**💡 创新点**

创新点主要有：①将文本间相似度转化为软标签的聚类感知对比学习，减少假负样本，提升跨模态对齐；②设计双向互补注意模块，既有视觉主导路径又有语言主导路径，保持空间一致性；③使用KAN网络进行注意力门控，为跨模态融合提供非线性、细粒度的特征选择。

**🔧 技术方法**

采用的技术包括：CLIP文本编码器（冻结）、聚类感知对比学习（CaCL）、双向互补注意模块（BCAM）、KAN网络（用于注意力和门控）、BCE+Dice损失、Adam优化器与余弦学习率调度。

**📊 数据集**

实验使用四个公开医学影像分割数据集：QaTa-COV19、MosMedData+（肺部）、CVC-ClinicDB、Kvasir（结肠息肉）。

**📈 对比分析**

与多种单模态与多模态基线（U-Net、U-Net++、nnU-Net、Swin-Unet、TransUNet、UKAN、MM-UKAN++、CLIP、GLoRIA、ViLT、TGANet、ConVIRT、LAVT、SLViT、LViT、UniLSeg、RefSegformer、MedLangViT、ARSeg）进行对比，C2Seg在Dice、mIoU、HD95、ASSD等指标上均取得最优或同等性能，并通过可视化进一步验证了对文本约束的更好遵循。

**⚠️ 局限性**

局限性包括：未提出专门的定量指标评估文本-掩膜一致性；仅在结构化、干净的文本上评估，缺乏对拼写错误、缩写或缺失信息等噪声文本的鲁棒性研究。

---

## 102. ReActWAM: Action-Consistent Caching for Efficient World-Action Models

**arXiv ID:** 2611.27372 | [PDF](https://arxiv.org/pdf/2611.27372v1)

**作者:** Zhaochen Guo `[一作]` (University of Sydney), Chang Xu `[通讯]` (University of Sydney)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种无训练的加速框架ReActWAM，利用动作意图一致性实现跨块前缀跳过与块内预测重用，显著降低WAM推理时间。

**💡 创新点**

创新点在于：①通过运动尺度、视觉一致性和预测动作意图的粗细分层门控来判定可复用块；②利用第一步预测差异对缓存的潜在状态进行纠正；③结合跨块跳过与块内重用，实现更高效的推理。

**🔧 技术方法**

采用基于扩散模型的世界-动作模型（WAM）架构、对扩散步骤的前缀跳过、预测引导的状态纠正、块内预测重用以及运动/视觉/动作意图门控。

**📊 数据集**

在LIBERO-Long、RoboTwin 2.0、FR3机器人真实环境中使用LingBot-VA、FastWAM、Cosmos系列WAM模型以及对应的任务集。

**📈 对比分析**

与Native、DiT‑Caching、VLA‑ADP、C^3ache、WAMachine等基线相比，ReActWAM在GPU推理上实现1.41×–5.44×速度提升，RoboTwin上任务成功率几乎不降；在FR3机器人上平均速度提升1.46×，成功率保持50%。

**⚠️ 局限性**

局限性包括：①仅适用于冻结模型的无训练加速；②跨块门控需要多项指标，可能导致过度过滤；③在高度变化或极端视觉差异的场景下，纠正机制可能不足；④对极大分辨率或复杂视频预测的可扩展性待验证。

---

## 103. The Vulnerability Window: Decision Instability in Quantized Reward Models

**arXiv ID:** 2611.27155 | [PDF](https://arxiv.org/pdf/2611.27155v1)

**作者:** Omkar Shewale `[一作]` (Illinois Institute of Technology), Divakar Kumar Yadav `[通讯]` (University of Wisconsin--Milwaukee)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了奖励模型和LLM评审者在4位量化后对比决策的翻转现象，并证明传统准确率指标无法捕捉此翻转。

**💡 创新点**

发现决策翻转集中在低边际区间，提出基于加性高斯噪声的概率模型，并展示小模型更易翻转；同时提出对量化风险进行审计和路由的实用方法。

**🔧 技术方法**

使用加性噪声模型、边际归一化、Monte Carlo bootstrap、McNemar检验、top‑k路由等统计与量化技术。

**📊 数据集**

评估数据包括 RewardBench、MATH‑500、GSM8K、人工构造的诱骗对手问题以及 1,000 个大规模候选集问题。

**📈 对比分析**

与四种 4 位量化器（NF4、FP4、INT4‑RTN、GPTQ）及多种奖励模型（Skywork、GRM）进行对比；发现量化后决策翻转率可达 12% 但准确率仅变动 0.03 分；边际模型能以 AUC≈0.93 准确排序翻转，top‑k 路由可恢复大部分损失。

**⚠️ 局限性**

仅针对权重单独量化、仅对数值任务、单一生成器与仲裁者、未验证更高位或混合精度、未考虑多任务或更大规模模型，缺乏在更广泛设置中的实证验证。

---

## 104. Seabird-Aware Multi-Objective Offshore Wind Capacity Allocation: A Gippsland Case Study

**arXiv ID:** 2611.27073 | [PDF](https://arxiv.org/pdf/2611.27073v1)

**作者:** Leilah Taouk `[一作]` (Monash University), Yidan Zhang `[通讯]` (Monash University)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

开发了一个三目标混合整数线性规划模型，在吉普斯兰海上风电规划中同时最小化资本成本、海鸟暴露并最大化风能产能，利用AUGMECON2求解Pareto前沿。

**💡 创新点**

创新点在于将分级海鸟暴露度量显式纳入多目标优化，而非传统阈值排除，提供更细致的环境与经济权衡。

**🔧 技术方法**

使用了混合整数线性规划（MILP）、增广ε约束方法（AUGMECON2）、基于普通克里金插值的海鸟暴露预测、网格搜索以及Gurobi求解器。

**📊 数据集**

使用的数据集包括吉普斯兰海域1km网格的AusBathyTopo 2024海底深度、2024年平均风速、海鸟现场观测（管嘴海鸟）数据、成本参数（C^cap、ϕ、C^wire）和Loy Yang的连接点距离。

**📈 对比分析**

与19个百分位阈值排除法在固定风能产能和固定资本预算两种情境下对比，结果显示优化在相同或更低成本/容量损失下实现了30–48%暴露降低，优于最严格阈值排除。

**⚠️ 局限性**

局限在于仅考虑平均年风速、单一年份海鸟观测、简化的成本与连接距离估计，未包含季节变化、wake效应和更精细的电网连通模型，且对海鸟暴露预测的克里金误差敏感。

---

## 105. Auditing and Adapting kNNGuard for Long-Context Prompts

**arXiv ID:** 2611.27084 | [PDF](https://arxiv.org/pdf/2611.27084v1)

**作者:** Niccolò Ronchetti `[一作]` `[通讯]`, Niccolò Ronchetti

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并评估了训练‑免费激活门控 kNNGuard 在长上下文场景下的表现，并针对其在长文本中表现衰退的问题提出了三种改进方案：双路长度路由参考库、长上下文嵌入替代以及滑窗池化；

**💡 创新点**

创新点在于识别并系统化 kNNGuard 在长文本中衰退的三大根源（激活漂移、截断盲点、信号稀释），并通过无训练、无额外前向传播的方式实现长文本适配，显著提升 AUROC 并保持短文本性能；

**🔧 技术方法**

采用 LLM 隐藏状态 kNN 检索（Layer‑Ensemble）、句子编码器 kNN（Embedding‑kNN）以及软聚合融合，结合长上下文句子编码器 Qwen3-Embedding、滑动窗口抽样与池化；

**📊 数据集**

使用 Content Harm Moderation benchmark（Aegis‑2.0）与 RAG Hallucination Detection benchmark（HaluBench）在 Qwen3‑5.5‑4B 与 Gemma‑4‑E4B‑it 两个 4B 开源 LLM 上进行评测；

**📈 对比分析**

与原始 kNNGuard 基线比较，长度路由双库+滑窗池化将长文本安全性 AUROC 提升至 0.828（LE）/0.886（FE），短文本保持不变；在 RAG 任务中，LE 在中等长度提升至 0.853（FE 0.837），但 Embedding‑kNN 仍接近随机；

**⚠️ 局限性**

局限在于仅针对 4B LLM 与 6.5K 以内文本评估，且对长文本的逻辑一致性验证仍受限于静态参考库，无法覆盖更大规模或专业领域的多千 token RAG 验证场景。

---

## 106. AROMA: Understanding AI Support For Mental Well-being through Role Theory

**arXiv ID:** 2611.26843 | [PDF](https://arxiv.org/pdf/2611.26843v1)

**作者:** Issac Ting `[一作]` (NYU Tandon School of Engineering), Vedant Das Swain `[通讯]` (NYU Tandon School of Engineering)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了 AROMA 框架，对三类情绪支持场景（同伴支持、心理咨询、激励访谈）中的人机对话进行角色分类、动态转移分析，并提出基于马尔可夫链的角色驱动提示（Markov Steering）以提升 AI 支持者与人类支持者的角色一致性。

**💡 创新点**

①首次将角色理论（角色预期 vs 角色执行）应用于 AI 情绪支持，对话中的“角色状态”从 8 种构成并量化；②发现 AI 在角色分布和转移上显著缺乏多样性；③提出 Markov Steering，显著降低 AI 与人类角色分布的 Jensen‑Shannon 散度（70–85%），为 AI 支持的结构化评估与调优提供新方法。

**🔧 技术方法**

使用手工标注与 LLM‑as‑judge 结合的两阶段角色标注；多项式逻辑回归分析角色与情境/主体关系；构建基于转移概率的马尔可夫模型；使用 JS‑divergence 评估角色一致性；通过 LIWC、词长、语义适配度等文本指标评估语言层面的对齐。

**📊 数据集**

共 13 个公开数据集（含 7,515 对话，99,872 段落），覆盖三种支持场景：同伴支持（ESConv、EmpDial、ShareChat 等）、心理咨询（CACTUS、HOPE、KokoroChat 等）与激励访谈（AnnoMi、MICha 等），语言涵盖英语、中文、日语。

**📈 对比分析**

通过多项式回归对比人类与 AI 的角色分布和转移，发现 AI 在“反思”“肯定”角色上显著稀缺，且转移率低。随后对三种提示策略（仅系统提示、未驱动角色提示、Markov Steering）进行 30 次模拟对话，计算每个 turn 的 JS‑divergence。Markov Steering 将人机角色差距缩小 70–85%，并在大多数语义指标（词长、LIWC 维度、情感词使用）上表现最接近人类支持者。

**⚠️ 局限性**

①仅限文本单轮对话，缺乏多轮、多会话或语音等多模态情景；②角色状态未必全面，可能遗漏某些新型 AI 专属角色；③未在真实用户或临床专家中验证角色驱动提示对情绪支持质量、用户满意度或临床效果的实际影响；④人类角色基准并非唯一理想目标，未来可探讨 AI 与人类角色的互补性。

---

## 107. SAVIOR: Sample-efficient Adaptation of Vision-Language Models for OCR Representation

**arXiv ID:** 2611.27319 | [PDF](https://arxiv.org/pdf/2611.27319v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 108. CarrierKV: An Edge LLM Accelerator with Positional Carrier Decomposition for Low-Bit KV Cache

**arXiv ID:** 2611.26857 | [PDF](https://arxiv.org/pdf/2611.26857v1)

**作者:** Junyi Luo `[一作]` (Brown University), Mehdi Saligane `[通讯]` (Brown University)

**通讯引用:** 1460 | [OpenAlex ID](https://openalex.org/A5007702824)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计了CarrierKV Edge LLM加速器，利用位置载体分解技术仅存储KV残差，并在芯片上按位重构RoPE位置项，实现4‑bit KV缓存精度恢复。

**💡 创新点**

创新点在于：①在RoPE之前剥离单向量位置载体，只保留残差；②通过位置相位引擎在每个key读取时快速重建位置分数项；③在保持NVFP4压缩格式的同时，显著提升信噪比，减少量化误差。

**🔧 技术方法**

采用的技术包括：4‑bit NVFP4量化、随机Hadamard变换、RoPE、数字相位累加器与查表正弦余弦计算，以及软硬件协同的预填位向量提取。

**📊 数据集**

实验使用的模型和数据集包括：Qwen3‑4B/8B、Llama‑3.2‑3B、Gemma‑3‑4B；评测任务为RULER、LongBench、PG19、AIME24/25、HMMT25、LiveCodeBench、GPQA‑Diamond、OlympiadBench。

**📈 对比分析**

与NVFP4、FP8、SageAttention、P3‑LLM等基线对比，CarrierKV在相同缓存尺寸下将32K RULER从79.4提升至91.1，PG19的PPL增幅降至0.17–0.24，硬件面积仅+2.9%，能耗+0.12%，推理速度提升约1.16×。

**⚠️ 局限性**

局限性在于：对Llama‑3.2‑3B的TurboQuant 3‑bit配置提升有限；在极长上下文（>32K）时PPL增幅仍略高；实现需要额外预填位向量存储和相位表，增加设计复杂度。

---

## 109. SCATR-ITS: a scalable transformer for large-area individual tree segmentation from LiDAR point clouds

**arXiv ID:** 2611.26715 | [PDF](https://arxiv.org/pdf/2611.26715v1)

**作者:** Lassi Ruoppa `[一作]` (Finnish Geospatial Research Institute FGI), Juha Hyyppä `[通讯]` (Finnish Geospatial Research Institute FGI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研发了一种可扩展 Transformer 模型 SCATR‑ITS，用于在大范围 LiDAR 点云中实现高效的个体树分割。

**💡 创新点**

创新点包括：基于 superpoint 的分割与聚类、动态查询计数模块、最大输入步幅、自适应尺寸感知后处理等，显著提升推理速度并降低 GPU 内存占用，同时保持与最先进方法相当的精度。

**🔧 技术方法**

技术路线是：在 ForestFormer3D 框架上改造稀疏 U‑Net 作为特征提取器，使用实例判别 MLP 与语义分支，生成 superpoint 并通过图聚类获得超点标签；随后在 Transformer 解码器中使用动态查询计数、最大输入步幅与尺寸感知后处理完成实例预测。

**📊 数据集**

使用的公开数据集有：FGI‑EMIT、FOR‑InstanceV2、ASL‑XL‑20‑pts（自制稀疏 ALS 数据）进行训练与验证；在 LAUTx、Wytham Woods、芬兰全国激光扫描（20 pts/m²）等数据集上进行零样本泛化与大面积推理评估。

**📈 对比分析**

方法上与 YOLOv12、SegmentAnyTree、TreeLearn、ForestFormer3D 等基线在标准 ITS 评估指标（Precision、Recall、F1、AP、Coverage）以及运行时、吞吐量和 GPU VRAM 进行对比；SCATR‑ITS 在精度上与 ForestFormer3D 接近或略低，但推理速度提升约 4.6–8.5×，VRAM 消耗下降约 2.3–3.5×；在芬兰 142 km² 区域的测试中，吞吐量为 13.8 min/km²，比 ForestFormer3D 快约 18.6×。

**⚠️ 局限性**

局限性：仍需大量预处理时间，尤其在高密度 TLS/MLS 数据上；模型训练依赖大量人工标注；在极低点密度或极稀疏场景下的性能尚未充分验证；大面积全局覆盖受限于 GPU 内存与滑动窗口尺寸。

---

## 110. Extracting Next Generation Access Control Policies from Natural Language Using LLMs: An Intermediate-Representation Approach with Back-Translation Verification

**arXiv ID:** 2611.26941 | [PDF](https://arxiv.org/pdf/2611.26941v1)

**作者:** Ayesha Habib `[一作]` (Northwestern Polytechnical University), Waiz Haider `[通讯]` (Northwestern Polytechnical University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套基于模型驱动工程（MDE）的完整流程，利用LLM仅完成中间表示（IR）的填充，随后通过固定规则编译器生成NGAC政策图，并引入反向翻译检查确保提取规则与原句含义一致，从而实现从自然语言到NGAC政策的自动化翻译；

**💡 创新点**

创新点包括：①使用JSON Schema约束LLM输出，保证IR总是可编译；②设计IR模板记录主体、动作、目标、条件和强度，支持所有四种NGAC关系；③实现无LLM编译器和结构化校验；④引入反向翻译检查（写规则→LLM判断）提高“真实性”评估，减少结构校验无法发现的语义错误；

**🔧 技术方法**

技术手段：LLM（Qwen3.8‑27B、gpt‑oss‑20b）在受限解码下填充IR；JSON Schema与vLLM实现受限解码；固定规则编译器将IR转为NGAC图；反向翻译检查使用LLM判定规则与句子是否一致；Policy Machine用于导出和执行；

**📊 数据集**

使用四个公开数据集：iTrust（T2P、ACRE），IBM课程注册系统，CyberChair会议系统，共计714条政策句子；

**📈 对比分析**

与两种公开方法（规则基线、Code4Policy）以及先前的Text2NGAC做对比。采用真实性（faithfulness）评价，本文方法在所有数据集上达95.7%真实性（最高），相比Text2NGAC 90.2%和Code4Policy 83.9%；在准确率上（exact match）约44%但不作为主要指标。反向翻译检查发现94.6%的人为插入错误，实际检测到26/31真实错误，标记10.5%句子需人工复核；

**⚠️ 局限性**

局限性：IR和语法无法表示数量、委托、嵌套义务；Policy Machine不支持条件与义务执行；反向翻译检查对义务的覆盖率不高；仅在公开短文档上评估，长篇多样化政策的鲁棒性待验证；

---

## 111. Trajectories Through Parsons Problems: Making Student Difficulty Patterns Visible to Instructors

**arXiv ID:** 2611.26799 | [PDF](https://arxiv.org/pdf/2611.26799v1)

**作者:** Daniel Prol `[一作]` (University of Houston), Amin Alipour `[通讯]` (University of Houston)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一款可记录学生每一次块操作的Parsons问题工具，并为教师提供可视化的轨迹仪表盘，帮助教师揭示学生在解题过程中的困难模式。

**💡 创新点**

首次将细粒度行为轨迹与AI辅助分析结合，生成可视化的困难报告和课堂幻灯片，从而让教师能直观看到隐藏的概念难点。

**🔧 技术方法**

在CodeCheck平台基础上实现客户端事件捕获，后端使用PostgreSQL存储轨迹，并通过OpenAI GPT‑4o对学生提交序列进行AI标注，同时构建Markov状态图以可视化常见路径。

**📊 数据集**

使用了美国高校软件设计课程（Java）和芬兰本科Python课程中176名学生完成的三道异常处理Parsons问题的日志数据，包含约3,675次块操作和819次提交。

**📈 对比分析**

通过统计时间、移动数、错误类型频率以及错误在后续问题中的重复率，并使用风险比、Fisher检验等方法进行对比，结果显示同一错误在后续问题中的重复率提升约1.8–2.5倍，教师可更快识别常见错误。

**⚠️ 局限性**

研究仅基于单一课程和主题，缺乏学生背景信息，轨迹仅记录动作无法解释动机，且未验证对学习成效的影响，工具在更大规模或不同语言课程中的可推广性未得到评估。

---

## 112. Proactive Defense Benchmark against Deepfake Generation

**arXiv ID:** 2611.27253 | [PDF](https://arxiv.org/pdf/2611.27253v1)

**作者:** Joonhyuk Baek `[一作]` (Ulsan National Institute of Science and Technology), Hoki Kim `[通讯]` (Chung Ang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的评测基准，用于系统评估主动防御对深度伪造生成的干扰、鲁棒性和可迁移性。

**💡 创新点**

创新点在于整合多维评估指标并引入校正身份比率（CIDR）消除生成器偏差，同时揭示鲁棒性与白盒性能的权衡与多模型迁移的极限。

**🔧 技术方法**

采用多种评估指标（像素、感知、身份、视觉质量）与鲁棒性/迁移度量，结合生成器的白盒与黑盒训练，使用PGD、DF‑RAP 等优化方法及模型学习方法。

**📊 数据集**

使用 CelebA‑HQ、FFHQ、VGGFace2‑HQ 等公开人脸数据集。

**📈 对比分析**

通过平均排名、ROB 与 TE 等多维度指标对比，展示不同防御在不同生成器上的表现，揭示单一指标误判的情况，并证明 CIDR 能更公平地评估身份扰动。

**⚠️ 局限性**

局限在于评测仍依赖公开模型，未覆盖未来更复杂的生成器；平衡鲁棒性与白盒效果仍是挑战。

---

## 113. User-Irrepressible Sequences for Multiple-Packet Reception

**arXiv ID:** 2611.27248 | [PDF](https://arxiv.org/pdf/2611.27248v1)

**作者:** Yen-Ling Shih `[一作]` (University of Tsukuba), Ying Miao `[通讯]` (University of Tsukuba)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种基于中国剩余定理的统一构造，生成任意多包接收能力 γ≥2 的协议序列集合（γ-CRT(K)），并证明该集合可支持最多 K+1 个活跃用户；同时给出了平均个体延迟的精确解析，并扩展了新质数序列和经典质数序列在 2‑UI 情况下的性质。

**💡 创新点**

创新点在于：①首次提供适用于所有 γ 的统一 CRT 构造，摆脱了对冲突避免码（CAC）参数的依赖；②证明该构造可支持多达 K+1 用户（含两个特殊序列）；③给出完整的平均延迟表达式；④证明新质数序列全族的 2‑UI 性质，并展示其与经典质数序列的等价性。

**🔧 技术方法**

采用的技术包括：中国剩余定理（CRT）映射、汉明交叉相关分析、冲突计数与 Möbius 反演、包含排除原理、周期平移的等价性证明以及平均延迟的闭式推导。

**📊 数据集**

本文以理论证明为主，未使用任何外部数据集进行实验验证。

**📈 对比分析**

通过与已有的单包接收（SPR）方案以及已知的 2‑UI 方案（如 2‑CRT、质数序列）进行理论对比，说明该构造在序列长度更短、用户数更多的同时保持 γ‑UI 性质；平均延迟分析提供了精确的性能指标，显示在任意相对位移下均能保证至少一次成功传输。

**⚠️ 局限性**

局限性包括：①仍未给出最短长度 L(N,γ) 的下界；②延迟解析在 γ>2 时仍较复杂，实际实现与计算成本未作评估；③缺乏实验或仿真验证，无法衡量在真实多包接收环境中的实际性能。

---

## 114. Exercised, Rarely Recovered: What Public Reports Say About Recovery in 50 AI Agent Harnesses and Runtimes

**arXiv ID:** 2611.27090 | [PDF](https://arxiv.org/pdf/2611.27090v1)

**作者:** Kaviru Hapuarachchi `[一作]` `[通讯]` (University of Colombo), Kaviru Hapuarachchi (University of Colombo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对50个AI编码代理系统（包括持久化运行时、编码代理框架和研究运行时）进行公开报告的恢复机制使用情况进行系统性调查和证据等级评估。

**💡 创新点**

首次从公开记录出发，对AI代理恢复机制进行统一分类、等级编码并与研究层面的保证进行对照，揭示实践与理论之间的差距（如补偿机制在实践中未出现）。

**🔧 技术方法**

采用基于协议的多步证据编码方法，利用LLM进行两轮独立编码、盲判定和人工核查；使用六阶证据等级、八类恢复机制分类；收集GitHub issue、供应商文档、论文等公开源。

**📊 数据集**

包含12个持久化运行时、若干公开/专有编码代理框架以及12个研究运行时，总计50个系统；证据来源为截至2026‑11‑09公开的GitHub、文档、论文等。

**📈 对比分析**

通过比较各系统在不同证据等级（E0–E5）和恢复机制等级（L0–L4）上的分布，统计实际使用（L4）次数及其恢复成功、部分成功、失败的比例；结果显示，虽然90%以上的系统在实践中使用恢复机制，但完整恢复率仅约11%，大多为部分或失败。

**⚠️ 局限性**

局限性包括：对“操作员”的定义可能偏向单人报告；公开记录倾向于报告失败，成功恢复可能被低估；LLM编码与人工核查的精度受限；搜索范围受限，可能遗漏隐藏或私有实现；时间敏感性，系统可能在截止时已变更；缺乏独立的人工编码验证。

---

## 115. Bring Cloud-Native Networking to FreeBSD Jails: Porting Calico from Linux

**arXiv ID:** 2611.27320 | [PDF](https://arxiv.org/pdf/2611.27320v1)

**作者:** Soma Sakaguchi `[一作]` (Future University Hakodate), Katsuya Matsubara `[通讯]` (Future University Hakodate)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

将 Calico CNI 插件移植到 FreeBSD 系统，实现了在 FreeBSD 容器（Jails）中的容器到容器、跨节点以及容器到互联网的网络通信；

**💡 创新点**

创新点在于：①通过构造嵌套 VNET Jail 实现类似 Linux Network Namespace 的网络隔离；②使用 routing sockets + ioctl 封装成 Netlink 兼容接口；③将 Felix 的 iptables 规则转换为 ipfw，完成了从 Linux 到 FreeBSD 的完整网络功能映射；

**🔧 技术方法**

使用技术包括：FreeBSD VNET、routing sockets、ioctl、ipfw、BIRD（BGP）、etcd、runj/nerdctl 容器运行时以及自定义的 Netlink/iptables 兼容库；

**📊 数据集**

未使用公开数据集，主要基于自建实验环境和示例 YAML 配置；

**📈 对比分析**

实验通过示例命令和 ping 测试验证了通信功能，但文中未给出定量性能指标或与原生 Linux 版本的对比；

**⚠️ 局限性**

限制包括：①Netlink 兼容层仍不完整；②ipfw 规则转换对某些高级 iptables 功能支持有限；③嵌套 VNET 可能带来性能开销；④未完成完整 Kubernetes 生态的集成与测试。

---

## 116. BladeR: Scalable, Shape-Guided 3D Reconstruction of Wind Turbine Blades from UAV Imagery

**arXiv ID:** 2611.27407 | [PDF](https://arxiv.org/pdf/2611.27407v1)

**作者:** Jonathan Sterckx `[一作]` (Ghent University), Hiep Luong `[通讯]` (Ghent University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了 BladeR，能够从 UAV 高分辨率影像中可扩展地构建风机叶片的高质量 3D 模型；

**💡 创新点**

其创新点在于将 GPS/IMU 和叶片几何先验融入分层稀疏重建、采用分块 Gaussian Splatting、软背景遮罩与自适应细化以突出前缘细节；

**🔧 技术方法**

技术包括基于 COLMAP 的 SfM、SuperPoint+LightGlue 匹配、DepthAnythingV2 与 SAM 语义分割、2DGS Gaussian Splatting 的分块训练与渲染、软遮罩损失和自适应 Gaussian 切分；

**📊 数据集**

使用了两块运转中的风机叶片的高重叠 UAV 数据集，采用 DJI Zenmuse P1 45MP 相机在不同光照与姿态下采集的图像；

**📈 对比分析**

与 COLMAP、Metashape 等基线相比，稀疏重建的重投影误差降低、点云数量和轨迹长度提升，稠密重建无缺口、细节更丰富、完整性更高；

**⚠️ 局限性**

局限性包括前缘细化对尾缘细节的覆盖不足、某些涂层剥落边界缺失，以及缺乏真实 3D 地面真值导致精度评估受限。

---

## 117. Ruby: State-of-the-Art Potency Prediction across Discovery Regimes Using Learned Structural Representations

**arXiv ID:** 2611.27015 | [PDF](https://arxiv.org/pdf/2611.27015v1)

**作者:** Michael Atkin `[一作]`, Han Yoon `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发并评估了一种基于结构共折叠模型中间配对表示的通用药效预测框架，该框架无需生成完整结合构象即可直接预测活性，适用于 HitID、H2L 和 LeadOp 三个药物发现阶段

**💡 创新点**

创新点在于利用结构共折叠模型的中间配对表示实现高质量活性预测，省去昂贵的构象生成过程；同时公开了经过源文档审计的大规模训练数据集，提供了可复现的训练资源

**🔧 技术方法**

技术包括结构共折叠（CoFold）模型、Transformer 结构、GPU 加速推理（Nvidia kernel 优化）、多任务学习、滚动时间校准以及基于中间配对表示的直接活性回归

**📊 数据集**

使用的数据集为 5.41 M 条公开来源活性数据（ChEMBL 37、BindingDB 2026、PubChem HTS 2025）和 1.42 M 条 AQAffinity 公开数据，另外在八个合作项目中收集了 22 036 条实验测定用于验证

**📈 对比分析**

与公开基准（OpenFE、CASP16、FEP+4、OpenBind）、经典 Docking（GNINA）、基于物理的 RBFE/ABFE 方法以及 2 模型等比较，Boltz-2 在 HitID 的早期富集（平均 nEF≈0.43）、零射击相关性（平均 Spearman ρ≈0.61、Pearson r≈0.59）和 LeadOp 的滚动时间校准 MAE（≈0.53 pX）上均优于对照

**⚠️ 局限性**

限制包括回顾性评估、未覆盖所有实验源导致审计样本有限、在极端化学空间上的泛化仍不确定、模型仍需在实际实验设计中进一步验证、以及对极低活性阈值的判定可能不够鲁棒

---

## 118. Keep the Process, Change the Supplier: Runtime Governance for Tool-Using AI Agents

**arXiv ID:** 2611.26966 | [PDF](https://arxiv.org/pdf/2611.26966v1)

**作者:** Deepak Waghmare `[一作]` (Two Much Labs), Shaad Sufi `[通讯]` (Two Much Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过脚本化请求实验，研究了工具使用 AI 代理的执行层——OPCP（Open Process Control Plane）——在授权、上下文披露、供应商选择和结果记录方面的行为与边界，验证其在不确定结果处理和持久化恢复中的安全与性能特性。

**💡 创新点**

创新点在于将代理提出的动作与权限决策、供应商资格、上下文限制和结果记录分离开来，形成一个可验证的执行边界；提出不确定结果的明确处理、供应商去重与持久化状态的结合；以及通过一系列系统实验量化其安全与成本边界。

**🔧 技术方法**

使用的技术包括：OPCP 运行时调解器、可重放的 idempotency 存储、供应商资格与授权门控、输入投影与合同验证、故障模拟与突变、强制进程终止与恢复、以及多种路由策略（固定、最佳得分等）。

**📊 数据集**

数据集为合成的请求/供应商适配器场景：约 2,400 个故障情景、19,000 次计时调用、5 个突变变体、160 条替代记录；没有使用外部真实业务数据。

**📈 对比分析**

比较方法：将 OPCP 与保守组合、retry‑all、stop‑all 四种策略在同一脚本请求下进行对照；评估错误恢复、重复副作用、响应确认率；同时测量在不同供应商池大小、输入大小、持久化方式下的调用延迟。结果显示 OPCP 与保守组合在所有故障下行为一致；retry‑all 产生重复副作用；best‑score 路由在 128 个供应商时延迟约 5.4 倍，远高于固定路由。

**⚠️ 局限性**

局限性：实验仅在本地工作站完成，未涉及真实语言模型、代理规划、云 API、分布式或持久化故障；突变覆盖有限；未评估端到端任务完成、模型安全性或监管合规性；实验仅使用合成数据，无法直接反映生产环境的复杂性。

---

## 119. SparseHPS: A Unified Model for Human Pose and Shape across Depth Cameras and LiDARs

**arXiv ID:** 2611.26730 | [PDF](https://arxiv.org/pdf/2611.26730v1)

**作者:** Ilya Zaychuk `[一作]` `[通讯]`, Ilya Zaychuk

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种统一的点云网络，能够从单帧或流式深度相机与LiDAR点云中恢复SMPL‑X人体姿态、形状与全局位移；

**💡 创新点**

创新点在于：①使用LitePT稀疏体素编码器与每关节可学习查询；②采用共享解码头与体素投票定位身体；③引入因果令牌记忆实现时序上下文；④通过统一标签空间与平衡采样，训练单一权重覆盖多传感器与数据集；

**🔧 技术方法**

技术包括稀疏体素卷积、Transformer跨层注意力、查询令牌、自回归记忆、投票机制、6D 旋转、Frobenius 损失、体素部件分割辅助；

**📊 数据集**

使用十个来源：合成（BEDLAM2、AMASS、CAPE、CAP）、深度相机（HuMMan、EgoBody、BEHAVE、FORM-HOI）、LiDAR（SLOPER4D、FreeMotion、RELI11D）；

**📈 对比分析**

与现有LiDAR序列方法（LiveHPS、BMLiCap、Sparkle）对比，在SLOPER4D和FreeMotion上单帧即可实现28.6 mm/41.8 mm的根相对关节误差，分别比最强对手低22%/11%；深度相机上在HuMMan中单帧误差为34.7 mm，已接近或优于先前方法；

**⚠️ 局限性**

局限性包括：仅处理单个人体且需预先分割；垂直轴误差需≤15°；不估计手部与面部；在低点数（<50点）时准确性下降；在不同数据集切分和评测协议上存在对比不完全一致的问题。

---

## 120. Solar-Conditioned Flow Matching for Multitemporal Aerial Visible-to-Infrared Image Translation

**arXiv ID:** 2611.27501 | [PDF](https://arxiv.org/pdf/2611.27501v1)

**作者:** Wenwen Fang `[一作]` (National University of Defense Technology), Zhenfeng Shao `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了Solar-Flow框架，将航空可见光到红外图像翻译从静态映射升级为多时相生成；

**💡 创新点**

创新点在于设计可计算的物理条件（太阳高度、热相位时钟、飞行高度），并通过自适应层归一化将其注入预训练的流匹配生成器，实现多时相控制；

**🔧 技术方法**

核心技术包括预训练的流匹配生成器（ThermalGen）+ VAE、Transformer跨注意力、条件自适应层归一化、ODE求解器及无分类器引导；

**📊 数据集**

构建了首个涵盖全天14个时段的配对航空V2IR数据集，包含5274对图像；

**📈 对比分析**

与无条件、Naive-1D以及Pix2Pix/CycleGAN基线比较，Solar-Flow在FID、未见时刻误差等指标上明显优于对照组，同时保持与无条件模型相当的PSNR/LPIPS；

**⚠️ 局限性**

局限在单站点单季数据、缺乏辐射校准与气象记录，且对极端天气或大尺度多地点迁移的泛化性尚未验证。

---

## 121. Beyond Linear Decodability: Understanding and Improving Instruction Following in World Action Models

**arXiv ID:** 2611.27401 | [PDF](https://arxiv.org/pdf/2611.27401v1)

**作者:** Bokai Ji `[一作]` (Renmin University of China), Yong Li `[通讯]` (Renmin University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种线性可解码监督（LDS）方法来提升世界动作模型（WAM）在测试时的指令跟随能力，并验证了其有效性。

**💡 创新点**

通过引入LDS和先前-指令竞争机制，首次证明指令信息的线性可解码性可以直接影响行为，并用训练免费干预证明了竞争模型。

**🔧 技术方法**

使用线性探测器、流匹配损失、辅助目标定位损失，并在WAM架构中加入多层线性头进行监督。

**📊 数据集**

在LIBERO-CF、LangGap、RoboTwin-IF等仿真基准以及AgileX Cobot实机测试上进行实验。

**📈 对比分析**

与基线模型和多种增量方法对比，LDS在LIBERO-CF上将成功率提升至74.6%（比基线高26.8个百分点），在LangGap和RoboTwin-IF上亦显著提升。

**⚠️ 局限性**

LDS对模型的后期参数更新和目标分辨率依赖较大，某些模型如LingBot-VA的效果有限，且未在所有视觉域中统一验证。

---

## 122. INCISE: Training-Free In-Context Segmentation of Surgical Videos

**arXiv ID:** 2611.26734 | [PDF](https://arxiv.org/pdf/2611.26734v1)

**作者:** Yifan Xu `[一作]` (University of Notre Dame), Danny Z Chen `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练‑自由的在上下文分割框架 INCISE，利用从其他手术视频中挑选覆盖目标视频局部外观的参考帧，并用冻结的 DINOv3 和 SAM3 进行无训练、无目标掩码的外科视频分割。

**💡 创新点**

创新点：① 基于覆盖的参考选择策略，使用局部特征聚类并为稀有聚类加权，确保对短暂或小尺寸目标的覆盖；② 使用独立伪标签记忆而非递归写回，防止误差累积；③ 将稠密标签转移与 SAM3 的记忆注意力结合，充分利用局部匹配与时间上下文。

**🔧 技术方法**

技术手段：DINOv3 ViT‑L/16 的稠密特征用于聚类、标签转移与图平滑；SAM3 tracker 的冻结模型与记忆注意力；稠密标签转移（最近邻加权标签传播）与标签传播平滑；子模函数贪婪覆盖度选择参考；子模权重调节稀有聚类；伪标签历史窗口（H=12）避免误差传播。

**📊 数据集**

使用的公开数据集：六个外科视频基准（CholecSeg8k、SISVSE、DSAD、EndoVis2018、SAR‑RARP50、CaDIS）以及 colonoscopy 的 PolypGen，用于跨域转移实验。

**📈 对比分析**

与多种基线对比：训练‑自由的 PerSAM、RSRM、INSID3；提供目标掩码的 SAM2/3；fine‑tuned 的 MedSAM2、SegGPT、COSINE、D‑GEM。INCISE 在六个基准上平均 mIoU 67.2%，比最强 fine‑tuned 方法高 9.2 点，比 SAM3（无目标掩码）高 14.7 点；在 PolypGen 上 Dice 72.9%，超过监督基线。

**⚠️ 局限性**

局限性：仍需人工标注选定的参考帧（虽然比例低）；参考选择依赖局部特征聚类，若特征不足可能错过关键外观；极少参考时性能显著下降；对光照变化、未见类别等极端情况的泛化尚有限；未探索更复杂的多尺度或更细粒度的记忆机制。

---

## 123. Adding One More Agent: How AI Team Size and Formation Shape Productivity and Diversity in Human-AI Brainstorming Teams

**arXiv ID:** 2611.26663 | [PDF](https://arxiv.org/pdf/2611.26663v1)

**作者:** Rong-Ching Chang `[一作]` (University of California, Davis), Hao-Chuan Wang `[通讯]` (University of California, Davis)

**通讯引用:** 2368 | [OpenAlex ID](https://openalex.org/A5060373173)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在团队中加入1或2个生成式AI伙伴并调节其行为配置对创意头脑风暴的产出、质量与人类贡献的影响。

**💡 创新点**

交叉比较AI伙伴数量与行为定制化的二维设计，并揭示多AI伙伴会降低人类直接写板的比例但提升局部多样性，说明人类角色转变。

**🔧 技术方法**

利用OpenAI GPT‑4o作为AI伙伴，配合平均人类校准（AVG）或参与者自定义的竞争、评价恐惧、懒惰三维行为配置，实现实时文本聊天与共享板的协作。

**📊 数据集**

实验使用两轮开放式脑力激荡任务的数据，收集了168名参与者的聊天与板内容，共120 AI实验组与48人类对话对照，形成约168条会话记录。

**📈 对比分析**

通过对比总板项、人类板项、聊天/板比例、语义多样性（全局与局部）等指标进行统计；发现两AI组总产出提升约1.25倍，但人类板项下降≈0.95；AI存在提升全球多样性（d≈0.57–1.08），两AI提升局部多样性（d≈0.57），但全球多样性不再提升。

**⚠️ 局限性**

仅测试单轮一次性脑力激荡，AI数量限制为1或2个，使用单一语言模型与实时文本聊天，未考察长期使用或更大团队动态；样本来自自愿在线受试者，缺乏多语言、多专业领域的验证。

---

## 124. LoCo-WAM: Test-Time Training as Long Context for World-Action Models

**arXiv ID:** 2611.26853 | [PDF](https://arxiv.org/pdf/2611.26853v1)

**作者:** Sheng Cheng `[一作]` (Amazon AGI), Deqian Kong `[通讯]` (University of California, Los Angeles)

**通讯引用:** 783 | [OpenAlex ID](https://openalex.org/A5084751425)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出LoCo-WAM，一种在机器人执行过程中对世界-动作模型的视频专家进行测试时自监督更新的算法，能在保持短历史长度的前提下提升任务成功率。

**💡 创新点**

创新点在于：①通过低秩适配器在执行时更新视频专家，利用观测帧的自监督视频损失；②不需要额外标签、奖励或元训练；③仅更新视频专家，动作专家保持不变，避免了分布漂移带来的不稳定。

**🔧 技术方法**

技术实现包括：低秩适配器（LoRA）在视频专家的自注意力和前馈层上；单步梯度更新（或可异步多步）；视频预测损失与动作预测损失的分离；异步更新降低决策延迟。

**📊 数据集**

实验数据集涵盖四个仿真基准：RoboCasa GR‑1、RoboCasa365、LIBERO‑Mem 与 RoboTwin 2.0；使用的基础模型为 Long‑WAM（短/长历史版本）和 LingBot‑VA。

**📈 对比分析**

与基线（未更新模型、长历史模型、AdaWorldPolicy、RoboTTT、Future Consensus、ReCAP）对比，LoCo‑WAM 在所有四个基准上均显著提升成功率：GR‑1 上 2.4 s 历史与 9.6 s 基线相当，4.8 s 历史达到 78.3%（比 9.6 s 高 4.8个百分点）；LIBERO‑Mem 上 8‑chunk 历史匹配 30‑chunk 长历史；RoboTwin 2.0 上短历史也能提高低成功率任务的表现。

**⚠️ 局限性**

局限性包括：①测试时更新增加计算和显存占用；②需要针对每个模型/任务调参（学习率、rank、步数）；③更新仅在执行期间起作用，无法解决长期分布漂移；④实验仅在仿真环境中验证，真实机器人部署可能面临感知噪声与动态变化的挑战。

---

## 125. CreditFilter: Calibrated Token Weights over the Within-Response Entropy Rank for RLVR

**arXiv ID:** 2611.27227 | [PDF](https://arxiv.org/pdf/2611.27227v1)

**作者:** Hanqing Zhao `[一作]` (Nanyang Technological University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RLVR（可验证奖励的强化学习）中提出一种新的token加权方法 CreditFilter，利用token在响应中的熵排名（within‑response entropy rank）重新分配奖励，将每个token的影响量量化为 forking 值。

**💡 创新点**

创新点包括：①将 forking 值分解为可直接从 softmax 获得的 fork 概率与需继续推演估计的 fork 后果；②发现 fork 概率随 token 熵排名呈指数衰减，并以此构造不需要额外前向传播的指数窗口权重函数；③引入难度阻尼与熵门控回退，动态平衡正负样本；④在训练过程中实时跟踪并更新窗口宽度（-D），使权重随模型进展自适应。

**🔧 技术方法**

使用的技术包括：forking 值测量与估计、fork 概率与后果分解、指数窗口加权、难度阻尼、熵门控回退、动态宽度跟踪、与传统 RLVR 基线（DAPO、GRPO、80/20 mask、STARE、SC‑GRPO）的对比实验。

**📊 数据集**

数据集与模型：在 Qwen3‑8B‑Base 上使用 DAPO‑17k 提示集训练；评估使用 MATH500、AIME24 和 AIME25 三个公开数学题集。

**📈 对比分析**

在平均三种种子下，CreditFilter 在 MATH500 上比 DAPO 提升 7.7 分，在 AIME24 和 AIME25 上分别提升 10.5 分和 9.1 分；在单一种子上也优于 DAPO 与 GRPO。与 80/20 mask、STARE、SC‑GRPO 等基线相比，同步步数下训练的 token 数约为 DAPO 的 2.1 倍，整体表现显著更好。

**⚠️ 局限性**

实验局限：仅在单一基准模型和提示集上验证；过长响应的惩罚不在理论覆盖范围内；理论假设（rank look‑ahead、fork 后果近似平坦）仅为近似；实验规模有限（如 AIME 仅 30 题），结果可能缺乏更广泛的普适性。

---

## 126. PanoUAV-Track: Full-Spherical UAV Visual Tracking in the Wild

**arXiv ID:** 2611.27383 | [PDF](https://arxiv.org/pdf/2611.27383v1)

**作者:** Bingyu Li `[一作]` (University of Science and Technology of China), Xuelong Li `[通讯]` (Institute of Artificial Intelligence TeleAI China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了大规模全球形无人机跟踪基准PanoUAV-Track，并提出了一种几何感知跟踪框架PUTrack；

**💡 创新点**

创新点在于（1）首次将全球形UAV跟踪作为研究任务并提供覆盖实景与仿真、昼夜等多场景的海量数据；（2）设计了Geometry‑Aware Search Adapter（GASA），通过纬度自适应采样（LAPS）补偿ERP投影的纬度畸变，并通过缝隙感知裁剪（SASC）解决横向周期边界的目标分裂问题；

**🔧 技术方法**

采用equirectangular投影映射、Latitude‑Adaptive Patch Sampling、Seam‑Aware Search Cropping、Transformer‑style 视觉跟踪网络（如OSTrack）以及针对球面空间的角度精度与Success AUC评价指标；

**📊 数据集**

使用了自建的PanoUAV‑Track数据集（1,895序列、507,961帧、2,872目标，包含实景白天/夜间和仿真），与现有视觉跟踪基准（TrackingNet、GOT‑10k、LaSOT、360VOT等）进行对比；

**📈 对比分析**

通过零射击与在数据集上训练两种评估模式，采用Success AUC、Normalized Precision和Angular Precision指标进行比较。PUTrack在零射击下取得84.75% AUC，在训练后提升至86.57%，在缝隙跨越目标上平均提升20+个百分点，速度保持约112 FPS，显著优于现有基线；

**⚠️ 局限性**

局限性包括：在极低光夜间场景下泛化仍弱；极地纬度区域仍受投影畸变影响；当前评估未覆盖长时跟踪的漂移与目标重新定位问题。

---

## 127. Preserving Pretrained CFG Structure in RL for Diffusion Models with Conditional Contrast Regularization

**arXiv ID:** 2611.27288 | [PDF](https://arxiv.org/pdf/2611.27288v1)

**作者:** Yunfei Zheng `[一作]` (Huazhong University of Science and Technology), Laurence Tianruo Yang `[通讯]` (Huazhong University of Science and Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究奖励‑质量失配问题，提出利用预训练CFG的条件对比正则化CCR，在扩散模型的强化学习后训练中保持生成质量并提升奖励。

**💡 创新点**

创新点在于将CFG视作几何方向，将其条件对比作为局部结构先验，在RL过程中对更新方向进行正则化，解决奖励驱动导致的质量损失。

**🔧 技术方法**

采用几何分析、CFG与RL统一视角、DiffusionNFT RL框架、局部重噪条件对比正则化、KL正则、数据正则等技术。

**📊 数据集**

主要使用Z‑Image基模型、Qwen‑Image、SD3.5‑Medium等模型，并在这些模型上使用GenEval、PickScore、HPSv2.1及多奖励混合进行评估。

**📈 对比分析**

与默认KL、增大KL、DDRL、GARDO等正则化以及CFG蒸馏比较，CCR在视觉清晰度、提示对齐、感知质量等指标上均优于或相近，同时保持奖励竞争性，整体奖励‑质量权衡更好。

**⚠️ 局限性**

仍受模型骨干差异影响，过强正则化可能降低奖励；角度投影诊断仅为工具，未直接优化；未在更大规模多模态模型或不同任务上验证。

---

## 128. RemoteEvolver: Expanding Earth Observation Capabilities through Agent Harness Evolution

**arXiv ID:** 2611.27110 | [PDF](https://arxiv.org/pdf/2611.27110v1)

**作者:** Shengxiang Xu `[一作]` (Southeast University), Min-Ling Zhang `[通讯]` (Southeast University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个基于外部 harness 的自我进化遥感代理系统 RemoteEvolver，能够在任务执行过程中自动发现、获取并集成缺失的专业模型，同时将成功执行的工作流转化为可重用的技能；

**💡 创新点**

核心创新在于将工具获取和工作流经验积累分离出来，在保持基础模型参数冻结的前提下，通过动态工具集扩展和技能库整合实现持续的能力增长；

**🔧 技术方法**

采用了工具搜索与部署模块（利用 GitHub、HuggingFace 资源）、执行检查、技能检索与生成模块、以及基于提示的规划-执行循环；

**📊 数据集**

在 Earth-Bench 与 OpenEarthAgent-Bench 两大遥感任务基准上进行评估；

**📈 对比分析**

与 RSMeM、RS-Agent、Earth-Agent、OpenEarthAgent、RemoteAgent 等5个主流代理系统以及自身未进化版本比较，RemoteEvolver 在工具缺失场景下平均提升 23.86%（Earth-Bench）/ 8.18%（OpenEarthAgent-Bench）的准确率，且在完整工具场景下亦比现有最优提升 6.77%/17.10%；

**⚠️ 局限性**

存在非单调性能波动的问题，尤其在工具过度探索、技能负迁移和规划执行偏差等层面导致远域任务转移能力下降和对已掌握任务的遗忘，需要进一步加入技能验证和演化剪枝策略。

---

## 129. cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction

**arXiv ID:** 2611.27193 | [PDF](https://arxiv.org/pdf/2611.27193v1)

**作者:** Haowei He `[一作]`, Yongxiang Li `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多流 Transformer 中残差混合的 Hyper‑Connection，提出通过几何收缩 Birkhoff 多面体实现更稳定、更有效的流混合。

**💡 创新点**

创新点在于：① 在 Birkhoff 多面体内进行可证明的几何收缩，保证残差矩阵保持双随机且不退化；② 提出只在浅层使用中等收缩的混合平衡策略，避免过度混合导致的特征失区分；③ 开发无新增参数的融合算子，显著提升 NPU 上的执行效率。

**🔧 技术方法**

采用 Birkhoff 多面体约束、双随机矩阵参数化、Dobrushin 系数、梯度与流差异分析；在 Ascend NPU 上实现 Pre/Post 融合算子。

**📊 数据集**

使用 OpenWebText 与 FineWeb‑Edu 进行 GPT‑2 规模实验，6B 与 300B Mixture‑of‑Experts（MoE）在 FineWeb‑Edu 上训练。

**📈 对比分析**

与传统残差、多流无混合、无约束混合、Sinkhorn‑Knopp、Kronecker 分解等基线对比；在 GPT‑2、6B、300B 上取得更低的训练/验证损失、稳定的梯度范数；融合算子将前后向延迟降低至 44.8%。

**⚠️ 局限性**

仍需手动调节收缩系数 λ 与层数 K；对深层过度收缩可能导致梯度消失；收缩参数在不同任务或网络架构上的通用性尚未完全验证。

---

## 130. HR-Merge: Model Merging with Headroom-Weighted Residual Refinement

**arXiv ID:** 2611.27377 | [PDF](https://arxiv.org/pdf/2611.27377v1)

**作者:** Jiakang Huang `[一作]` (University of British Columbia), Baharan Mirzasoleiman `[通讯]` (University of California, Los Angeles)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了一种训练无关的两阶段残差细化包装器，利用任务残差并按验证 headroom 加权重合，从而在多任务模型中恢复并提升被压缩的任务性能。

**💡 创新点**

证明残差中保留的任务信息可恢复且需通过重合并而非简单求和；提出基于验证 headroom 的动态加权残差和验证强度选择机制，保证在验证集上不低于基线。

**🔧 技术方法**

任务向量、模型合并算子（如 TSV‑M、DC‑Merge、TIES 等）、LoRA 适配器、温度自适应 softmax、验证强度网格搜索、McNemar 统计检验等技术。

**📊 数据集**

LLaMA‑3 8B NLI 6 任务（SNLI、MNLI、SICK、QNLI、RTE、SciTail）和 CLIP 视觉任务（ViT‑B/32、ViT‑B/16、ViT‑L/14 的 8/14/20 任务），以及 LoRA 适配器下的 ViT‑B/32、LLaMA‑3 8B 8 任务。

**📈 对比分析**

与六种基线合并方法（权重平均、Task Arithmetic、TIES、TSV‑M、Iso‑CTS、DC‑Merge）对比，使用验证与测试的归一化准确率衡量；在 NLI 上提升 1.00 点，在 LoRA NLI 上提升 0.23 点；在视觉全微调上平均提升 0.64 点，在 LoRA 视觉上提升 0.78 点；在所有配置中至少不低于基线，并在多数设置下取得最高性能。

**⚠️ 局限性**

仅验证 6 个 NLI 任务，未覆盖其他模型族或生成任务；仅评估准确率，无法直接用于错误或生成指标；选择准则为平均准确率，可能忽略最差任务；理论转移边界依赖验证与测试分布相同；未包含某些基线方法（如 LiNeS、适配器组合）。

---

## 131. Learning Unknown Oscillators From Bandit Feedback

**arXiv ID:** 2611.27492 | [PDF](https://arxiv.org/pdf/2611.27492v1)

**作者:** Priyank Jayraj `[一作]` (Birla Institute of Technology and Science), Navneet Goyal `[通讯]` (Birla Institute of Technology and Science)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研究半带反馈下的未知振荡器驱动的决策问题，设计了基于模型承诺的控制器和置信集，并证明了平方根次优的 regret 上界。

**💡 创新点**

创新点在于：①利用模型自测思想让贪婪决策暴露预测错误；②将持续模型承诺的持续时间转化为可从历史奖励检验的决策；③在连续置信集上评估未来决策的奖励损失并给出最优块长度；④将方法推广到两振荡器竞争场景。

**🔧 技术方法**

采用几何分析映射预测误差到观测误差、置信集合构造与最优块长度控制器、在线回归与平方根 regret 证明、区间频率估计与观测序列匹配等技术。

**📊 数据集**

使用 96 个基于噪声的合成振荡器实例，频率和相位随机生成。

**📈 对比分析**

与每轮全频率拟合的基线相比，检查承诺后每块仅需 49 次拟合（相较于 1023 次），平均 regret 仅高 1.5%，在所有频率下保持平方根次优；实验中还实现了 4.69 倍的速度提升。

**⚠️ 局限性**

局限性包括仅在合成实验上验证，假设振荡器是连续单频旋转，缺乏对更复杂或非线性动态的推广；仅考虑半带反馈，未评估完整 bandit 场景；对噪声分布假设较强。

---

## 132. Jagged Robustness: Uneven Safety Transfer and Emerging Vulnerabilities under LLM Quantization

**arXiv ID:** 2611.27424 | [PDF](https://arxiv.org/pdf/2611.27424v1)

**作者:** Ramneek Agnihotri `[一作]` (Ruhr University Bochum), Hans Vath `[通讯]` (Ruhr University Bochum)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了量化对大型语言模型（LLM）各能力与安全域的影响，并提出了“Jagged Robustness”概念；

**💡 创新点**

发现量化导致模型在不同任务（如指令遵循、安全拒绝、欺骗等）表现不均衡，并设计了针对量化模型的通用后缀攻击；

**🔧 技术方法**

采用后训练量化（PTQ）与权重量化、AWQ、INT4/INT8等量化方法，使用LM-judge评估与多选/自由生成基准，构建了量化感知对抗后缀搜索算法；

**📊 数据集**

使用多种公开基准：GPQA、MMLU、ARC、BBQ、TruthfulQA、ETHICS、HumanEval、MGSM、InFoBench、SORRYBench、DeceptionBench、WildChat-NonToxic、Mistral-7B-Instruct 等；

**📈 对比分析**

通过对比BF16基准与不同量化模型的准确率/拒绝率、误导率，展示了在INT4下模型在安全拒绝和欺骗任务上显著下降，而其他任务相对稳定；性能指标显示：INT4在安全拒绝上提升约+5%，在欺骗率上提升约+11%；

**⚠️ 局限性**

局限性包括：评估依赖LM-as-Judge，可能存在判定不一致；攻击效果主要产生表面合规文本；未验证不同模型规模/架构；样本量有限，未提供置信区间；未探讨对全精度模型的攻击向量；

---

## 133. A Vision-Based Framework for Manual Assembly Verification and Adaptive Operator Assistance

**arXiv ID:** 2611.27293 | [PDF](https://arxiv.org/pdf/2611.27293v1)

**作者:** Naveen Joseph Delphin `[一作]` (Fraunhofer Italia Research), Marco Todescato `[通讯]` (Fraunhofer Italia Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文开发了基于单摄像头RGB的视觉装配监控与可配置指导系统，集成检测、姿态分类、轨迹验证与两级有限状态机，实现实时工件装配阶段验证与反馈。

**💡 创新点**

创新点在于将感知结果与显式程序推理及多层次指导级联，形成可替换的管线；采用区域交互触发+手势门控、DTW轨迹比较与两级FSM，提供三种指导级别。

**🔧 技术方法**

技术包括RF‑DETR和YOLOv8n对象检测、YOLOv8n‑cls任务姿态分类、MediaPipe手部姿态跟踪、DTW轨迹比较、MotionFSM/StageFSM有限状态机、图像运动检测，使用Python/ OpenCV/ PyTorch 等。

**📊 数据集**

使用的数据集为真实工件图像COCO‑style训练集（942/241/324），合成CAD渲染图用于增强，8名参与者的装配/拆卸视频；姿态分类使用检测后裁剪图像，轨迹参考采用人工标注的专家轨迹。

**📈 对比分析**

通过混合效应模型比较三种指导级别的运动质量综合分（jerk、停顿、微停、路径效率）和错误/完成时间，结果显示全指导优于中/无指导且运动质量差异显著；检测精度mAP 0.949，姿态分类准确率96–97%，手动触发误差11/677。

**⚠️ 局限性**

局限性包括样本量仅8人、单一装配任务、固定摄像头位置、未评估长期使用或自适应指导选择、缺乏触发准确率的独立评估、无组件级消融、无延迟测量，且对其他工作站、任务和摄像头视角的推广性有限。

---

## 134. "More Than You Expect!": Leakage-Abuse Attacks on Encrypted Vector Search

**arXiv ID:** 2611.27174 | [PDF](https://arxiv.org/pdf/2611.27174v1)

**作者:** Yang Zhang `[一作]` (Nankai University), Zheli Liu `[通讯]` (Nankai University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在对加密向量检索系统的访问模式进行分析后，提出并实现了SAGE攻击，能够仅利用返回的记录ID共检索关系来推断数据库记录标签和查询类别。

**💡 创新点**

首次识别并利用语义邻域泄露（semantic neighborhood leakage），并提出两种基于辅助知识的攻击（SAGE‑Dist和SAGE‑Known），实现了无需明文向量或排名即可重建语义标签。

**🔧 技术方法**

主要技术包括共检索图构造、Jaccard加权、谱聚类、Hungarian 匹配、Sinkhorn 最优传输、标签传播与群体投票等图与优化方法。

**📊 数据集**

在 AG News、Yahoo Answers、MedMCQA 和 CIFAR‑100 四个文本与图像数据集上实验，使用对应的文本/图像嵌入作为向量。

**📈 对比分析**

与随机、先验、近邻、Louvain、LP、Label‑Spreading 等基线相比，SAGE‑Known 在 1% 已知标签下实现约 84% 的记录标签准确率、87% 的查询类别准确率；SAGE‑Dist 在已知类别比例条件下亦显著优于基线。

**⚠️ 局限性**

该攻击依赖可跨查询链接的记录ID，且对类别比例估计或已知标签覆盖度敏感；在类别数多或语义相近的数据集（如 CIFAR‑100、MedMCQA）表现较差，对检索误差与窗口化链接的鲁棒性有限。

---

## 135. Decision Models Are General-Purpose Process Reward Models

**arXiv ID:** 2611.27068 | [PDF](https://arxiv.org/pdf/2611.27068v1)

**作者:** Jiajun Fan `[一作]` (University Of Illinois Urbana Champaign), Ge Liu `[通讯]` (University Of Illinois Urbana Champaign)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出在语言智能体的强化学习训练中，使用零射门的通用决策模型为每一步生成进度概率，并将标准化后的分数直接加到优势函数中，从而实现无需训练步标签或价值模型的信用分配。

**💡 创新点**

创新点在于把决策模型作为通用的过程奖励模型（PRM）使用，无需额外训练；该模型一次性回答每一步是否推进任务，能够在所有rollout均失败时仍为每一步提供信号；并且该信用分配可直接插入任意策略梯度方法。

**🔧 技术方法**

主要技术包括：1）调用通用决策模型（如JeV 1.13）获取每一步的进度概率；2）将概率进行标准化（基于同一任务rollout的均值与标准差）得到信用；3）将信用加到现有RL方法（GRPO、GiGPO、GraphGPO）的优势上；4）在训练中使用零射门提示词与日志格式。

**📊 数据集**

实验使用两个基准：ALFWorld（基于TextWorld的房屋任务）和WebShop（模拟在线购物任务），都在相同的1.5B参数语言模型（Qwen2.5-1.5B-Instruct）上进行训练。

**📈 对比分析**

与基线GRPO、GiGPO、GraphGPO比较，使用信用后在ALFWorld上成功率提升约10–30%，在WebShop上成功率提升约5–15%，学习曲线更快，最终模型在ALFWorld中平均耗时减少30%以上，在WebShop中购买正确产品的比例显著提高。

**⚠️ 局限性**

局限性包括：需要外部决策模型的API调用，虽成本低但仍占用时间；信用基准的选择对不同任务敏感（需要为每种动作类型设置单独基准）；在极端对抗或极大规模任务时，零射门决策模型的准确性可能受限，且对新任务仍需手工编写问题文本。

---

## 136. Git History Is a Pangenome: Repetition in Code Training Corpora and What Deduplication Removes, Keeps and Misses

**arXiv ID:** 2611.27154 | [PDF](https://arxiv.org/pdf/2611.27154v1)

**作者:** Saurabh Chopra `[一作]` `[通讯]` (GitLab Inc), Saurabh Chopra (GitLab Inc)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对使用代码版本历史构建训练语料库的重复性、去重工具行为以及基准修复在历史中的传播情况进行了系统测量，并基于结果给出针对不同语料类型的实践规则。

**💡 创新点**

创新点在于：①首次量化仓库历史中相似版本的规模（典型仓库约19倍，成熟仓库约89倍）；②评估了 BigCode 等主流近似去重脚本跨版本运行时的去除效果、保留规则以及对当前版本的影响；③揭示基准修复在历史中的重复分布（中位数 18 版）并指出去重工具无法完全清除这些内容。

**🔧 技术方法**

使用的技术包括：基于 MinHash + LSH 的相似度去重；基于后缀数组的字节级相对 LZ 解析用于新旧字节归因；行级重叠检测用于基准修复检测；以及通过采样、Bootstrap 等统计方法估计分布和置信区间。

**📊 数据集**

数据集：①306 个公开、许可宽松、至少 200 次提交的随机仓库（8 种语言）；②25 个“成熟”仓库（包含 613,849 次非合并提交）；③12 个 SWE-bench 仓库（2,294 条基准实例）。

**📈 对比分析**

比较方法：在不同 Jaccard 阈值（0.7、0.85、0.9）下运行去重脚本，并对比默认保留规则与按时间新旧排序（keep-newest）的效果；测量重复倍数、保留字节比例、当前版本保留率等指标。结果显示：去重可将重复从 19× 降至约 2×，keep-newest 能将当前版本保留率从约 40–50% 提升至 94%，成本仅为 0.1–0.2×当前代码量。

**⚠️ 局限性**

局限性包括：未进行模型训练实验，无法直接评估重复对模型质量的影响；样本仅包含 200 次以上提交的仓库，低频仓库的重复情况未知；基准修复检测以行级重叠为准，短补丁可能误判；去重脚本未针对所有分支完整运行，可能低估修复传播；未考虑 Fork、镜像等额外拷贝导致的重复。

---

## 137. Staleness Deadlines for Motion Plans during Onboard Reasoning

**arXiv ID:** 2611.26858 | [PDF](https://arxiv.org/pdf/2611.26858v1)

**作者:** Hassan Iqbal `[一作]` (University of Texas at Austin), Hassan Iqbal `[通讯]` (University of Texas at Austin)

**通讯引用:** 779 | [OpenAlex ID](https://openalex.org/A5027221719)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了在移动机器人使用语言模型进行规划时引入的“时效性截止点”（staleness deadlines），并通过预算策略与离线重放来验证其在避免碰撞、错误决策和迟到方面的效果。

**💡 创新点**

创新点包括：1) 将时间截止点从语言模型响应时间转移到机器人时钟，实现对规划过时的安全管理；2) 通过校准的预测区域和偏差管道为截止点提供统计保证；3) 采用精确的二项检验对预设的推理预算策略进行家族级别的风险合格性验证；4) 通过离线重放评估方法预测现场表现，减少实机验证成本。

**🔧 技术方法**

使用的技术包括：Qwen3‑4B 语言模型、基于图的路径规划器、跟踪控制器与制动控制器、基于连续预测（conformal prediction）的加速度/行人预测区间、偏差管道、基于时间的安全检查与离线重放、以及对失效率的精确二项检验。

**📊 数据集**

数据集：在模拟建筑环境中生成 81,000 次任务（含 1500 次验证试验），每次任务包含行人轨迹、门禁信息或交付请求，使用 Qwen3‑4B 的 256 或 640 token 预算答案。还收集了现场 Jetson AGX Orin 与桌面 GPU 的推理速度数据。

**📈 对比分析**

比较方法：对每个推理预算策略（固定预算与预设的 per‑query 规则）在 1500 次任务中统计三种损失（接触、错误决策、迟到）。结果显示：完整推理将错误决策率从约 25% 降至 1%，迟到率减少 64%；固定 256 token 预算在签名任务下能保持目标率；在高负载下，仅 256 token 预算可通过；离线重放与现场实验在 24/32 任务中对三类损失达成一致，说明离线测试能较好预测现场失误。

**⚠️ 局限性**

局限性：1) 在高感知负载下完整推理导致迟到率升高；2) 当机器人停止时，仍可能被不让行的人撞击（约 23%/32% 的停机接触）；3) 受限于离线重放仅能预测移动时的失误，对停机接触的预测不足；4) 需要在不同机器人平台上进行更广泛的硬件验证。

---

## 138. Where Structure Enters a World Model: From Predictive Fit to Physical Rollout

**arXiv ID:** 2611.27149 | [PDF](https://arxiv.org/pdf/2611.27149v1)

**作者:** Panayiotis Christou `[一作]` (Fordham University), Rajarshi Das `[通讯]` (MQube Cognition)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究如何将物理结构（守恒量监督、几何对称表征、时间一致性和学习的结构化动力学）注入预测世界模型，以提升在视觉动力学（摆、弹簧、阻尼摆、双摆）和行星运动（单行星轨道）中的长期预测精度。

**💡 创新点**

创新点在于将多种物理引导方式并行整合，分别作用于模型训练目标、表征层和动力学推理，并通过严格对照实验评估其独立与协同效应；同时提出相对/对偶几何适配器和基于RK4的学习动力学头，实现物理可解释且可泛化的预测。

**🔧 技术方法**

采用 V-JEPA / JEPA 视觉编码器、自动回归 Transformer（NTP）进行坐标预测、物理监督头（力、能量、角动量）、Temporal Difference (TDV) 表征一致性、相对与对偶几何适配器、Runge‑Kutta 4（RK4）积分及能量平衡约束。

**📊 数据集**

使用仿真生成的图像序列（64×64 像素）用于摆、弹簧、阻尼摆、双摆等四个视觉系统，以及单行星轨道坐标序列（含质量、速度、时间标签）的仿真数据集。

**📈 对比分析**

对比基线 V-JEPA 2 与加入各物理结构的版本，在 400 步预测中：角度 RMSE 下降 0.88→0.22，能量漂移降低至 0.00015；行星模型中，角动量监督将 NTP 位移 MSE 从 1.0755 降至 0.1437 AU²，几何适配器将相对 NTP 位移 MSE 降至 0.0630 AU²；力方向余弦亦显著提升，表明模型在不同参数变化下保持更好的物理一致性。

**⚠️ 局限性**

局限性包括：对多体或参数变化的泛化仍有限；离散化与注意力结构限制了完全的等变性；训练时间和数据量未充分评估；实验仅在仿真数据上验证，缺少真实世界数据验证。

---

## 139. A Stationarity-Based Regret Framework for Time-Varying Nonconvex Composite Optimization

**arXiv ID:** 2611.27218 | [PDF](https://arxiv.org/pdf/2611.27218v1)

**作者:** Nadav Hallak `[一作]` `[通讯]` (Technion), Nadav Hallak (Technion)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于近似一阶最优性（proximal‑gradient residual）的本地 regret 框架，用来评估时间变换的非凸复合优化问题，并给出了对应的在线算法（嵌套和单步 prox‑gradient）。

**💡 创新点**

创新点在于：①用一阶最优性残差构造可解释且在凸情形下等价于传统动态 regret 的本地 regret；②证明在无时间平滑性下无子线性 regret；③引入梯度变化预算作为时变性约束，并在此框架下给出嵌套与单步 oracle 的子线性性能保证；④将该框架推广到非凸多人博弈，得到近似一阶纳什均衡的收敛率。

**🔧 技术方法**

主要技术包括：proximal‑gradient映射与其残差、Lipschitz 连续性与光滑性分析、梯度变化预算、凸/非凸不等式推导、嵌套与单步迭代复杂度分析，以及对多玩家博弈的渐进性收敛证明。

**📊 数据集**

实验使用两类数据集：1）一个基于合成电力市场的多玩家博弈（含 6 名生产者，使用历史价格和产量数据）；2）美国标普 500 的 100 天滚动回报，用于构造非凸 MVSK 投资组合优化问题。

**📈 对比分析**

通过与离线最优（凸/非凸）基准、单步 prox‑gradient、以及常见的动态 regret 进行比较。实验结果显示：嵌套算法在两种实验中保持了稳定的目标值；相较于静态基准，在线算法在时间变动情形下能够持续追踪并逼近最优/均衡；单步方法虽然更低成本，但在数值上收敛速度较慢。性能指标主要表现为累计收益或均方残差，满足理论给出的子线性上界。

**⚠️ 局限性**

局限性包括：①需要梯度变化预算假设（无此假设时无法保证子线性）；②理论仅覆盖无噪声的梯度信息，实际中若存在估计误差需进一步研究；③单步 prox‑gradient 在高非凸性环境下收敛速度慢；④对强可微、Lipschitz 连续性的假设在某些工业应用中可能不满足。

---

## 140. NightmareDiscourseBench: A Benchmark Dataset for Fictional Direct Discourse Identification Pipelines

**arXiv ID:** 2611.27059 | [PDF](https://arxiv.org/pdf/2611.27059v1)

**作者:** Claudia Carroll `[一作]` (Carnegie Mellon University), Dashiell Leigh `[通讯]` (Washington University in St. Louis)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了 NightmareDiscourseBench，一个面向英语小说中直接话语识别的极限测试基准数据集。

**💡 创新点**

创新点在于：①挑选具有实验性、复杂排版和嵌入式话语等特征的文本，形成真实且具挑战性的测试集；②对直接话语进行细粒度分类（直接演讲、直接思考、嵌入文本和其他）并提供丰富的元数据；③将该基准与现有 LitBank、Quote Annotator、RiQuA 进行对比评测。

**🔧 技术方法**

使用了手工双人注释流程、专用注释工具（输出 JSON）和后期专家复核；评测时采用 BookNLP 的直接话语识别模块，并通过 0.9 的 indel 相似度阈值评估 F1、精确率和召回率。

**📊 数据集**

基准数据集包含 12 篇小说（约 3000 词/篇，12 位作者），并与 LitBank（86 篇）、Quote Annotator（6 篇）和 RiQuA（10 篇）进行对照。

**📈 对比分析**

在 BookNLP 上，NightmareDiscourseBench 的平均 F1 为 0.77，略高于 Quote Annotator（0.74），但方差更大，显示其对识别模块构成更严峻的挑战。

**⚠️ 局限性**

限制主要体现在：①样本量小且集中在 19 世纪文本；②需要耗费大量人工双盲注释和专家复核；③对直接思考和嵌入文本的覆盖仍有限，后续需扩大文本多样性与年代范围。

---

## 141. Analyzing Multi-Layer Visual Representations of Large Vision-Language Models for Anomaly Detection

**arXiv ID:** 2611.27128 | [PDF](https://arxiv.org/pdf/2611.27128v1)

**作者:** Satoshi Hashimoto `[一作]` (KDDI Research), Mori Kurokawa `[通讯]` (KDDI Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究了大规模视觉语言模型（LVLM）内部多层视觉表征在图像和视频异常检测中的有效性，探讨层级选择与融合策略对性能的影响。

**💡 创新点**

发现与以往假设相反，异常检测并不一定需要浅层或多层融合特征，单层深层视觉或LLM侧特征在不同监督模式下更具优势，并给出了基于任务和数据分布的层选择指导。

**🔧 技术方法**

使用冻结的 Qwen2.5‑VL‑7B 进行层级特征提取，构建 Transformer 编码器+线性输出的分类器，结合重建（Recon）、预测（Pred）以及 PatchCore 等基准异常检测方法。

**📊 数据集**

在视频异常检测中使用 UBnormal、UCSD Ped2；在图像异常检测中使用 MVTec AD 与 MVTec LOCO AD 四大公开数据集。

**📈 对比分析**

通过对单层特征、多层融合组（浅层/深层/全层）在三种监督（全监督、弱监督、单类）下的 AUROC 进行对比，发现多层视觉特征并非始终最优，单层或深层 LLM 侧特征在特定设置下可获得更高性能，表明层选择需根据异常类型和监督信号量身定制。

**⚠️ 局限性**

缺乏自适应层选择与融合机制，未能充分捕捉不同异常类型的层级需求，且实验仅基于固定 LVLM 与固定分类器，未来工作需开发动态层级选择与融合算法。

---

## 142. Latent Jacobian Steer: Eliciting Contact-Rich Out-of-Distribution Behaviors from Pretrained Humanoid Controllers

**arXiv ID:** 2611.27338 | [PDF](https://arxiv.org/pdf/2611.27338v1)

**作者:** S K Swaminathan `[一作]` (National University of Singapore), Guillaume Adrien Sartoretti `[通讯]` (National University of Singapore)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于姿态Jacobian映射的LJS方法，利用冻结的全身控制器在动作空间搜索并映射到潜在空间，实现对离谱任务的接触丰富行为生成。

**💡 创新点**

创新点在于将动作空间的可行扰动通过政策Jacobian的局部逆映射到潜在空间，实现了动作与潜在两种搜索空间的耦合，从而兼顾协调性与精细控制。

**🔧 技术方法**

使用的技术包括CEM优化、蒙特卡洛仿真评估、damped least-squares逆Jacobian、离散潜在空间采样以及物理效果评分。

**📊 数据集**

实验数据集为基于MuJoCo和Isaac Sim的模拟环境中的人形机器人（BFM-Zero）和多指手（Play2Perfect）进行的三项全身接触任务与四项装配任务。

**📈 对比分析**

与直接潜在搜索、残差搜索等基线相比，LJS在任务完成率、搜索成本与运动质量方面均优越，例如在门-立方任务上完成率为100%，而基线低于30%。

**⚠️ 局限性**

局限性包括仅在离线仿真中可用，需要仿真器的特权信息与并行候选模拟，且对预训练控制器的覆盖范围与手工定义的物理效果敏感。

---

## 143. WINSO: A Large-Size Very-High-Resolution Remote Sensing Dataset for Wind Turbine Detection and Solar Panel Segmentation

**arXiv ID:** 2611.27241 | [PDF](https://arxiv.org/pdf/2611.27241v1)

**作者:** Di Wang `[一作]` (Xidian University), Xinbo Gao `[通讯]` (Xidian University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含 10,962 张高分辨率卫星图像、166,473 个经专家验证的风力涡轮和光伏板实例的 WINSO 基准数据集，用于大尺寸图像中的风机检测与光伏分割。

**💡 创新点**

创新点在于：①提供稀疏目标且尺寸跨度极大、全景尺度的标注；②在保持全景图像的同时实现高精度的水平框和多边形掩码；③系统评估多种推理策略（全图、滑窗、超高分辨率、VLM），并展示 WINSO 的跨域泛化能力。

**🔧 技术方法**

采用了多种主流目标检测（YOLO、Faster R‑CNN、DETR 等）、分割模型（U‑Net、SegFormer、Mask2Former 等）、专用超高分辨率方法（SparseFormer、GigaDET、SPDet、UHR‑DETR）和视觉‑语言模型（LocateAnything、Rex‑Omni、Grounding‑DINO 等）进行基线实验。

**📊 数据集**

使用 WINSO 数据集本身作为训练集，并在外部公开数据集（STAR、DIOR、PVP‑Dataset、AIR‑PV 等）上进行跨域测试，验证模型迁移性。

**📈 对比分析**

比较方法包括局部滑窗推理、全图缩放、专用 UHR 处理和 VLM 方案；在风机检测上局部推理 AP 达到 41% 以上，UHR 方案实现 0.7~0.8 秒推理但小目标 AP 低；在光伏分割上 SegFormer、Mask2Former 取得 86% 前景 IoU，UHR 方法在小碎片化场景上误检多。

**⚠️ 局限性**

局限性在于：①对稀疏目标的候选区域选择仍不充分，导致小目标检出率低；②超高分辨率模型在全图推理时缺乏自适应稀疏计算；③数据集仅覆盖美日台三域，可能限制更广泛的地理推广；④VLM 在大尺寸卫星图像上的精度与效率均不理想。

---

## 144. Dynamic Preference Logic of Preference Lifting

**arXiv ID:** 2611.26643 | [PDF](https://arxiv.org/pdf/2611.26643v1)

**作者:** Xiaochen Chang `[一作]` `[通讯]`, Xiaochen Chang

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种动态偏好提升逻辑，给出了其语法、语义、可满足性与完备性证明，并将偏好提升方案视为独立维度进行建模。

**💡 创新点**

证明动态偏好提升方案的变化不能归约为偏好关系本身的变化，并提供了第一阶可定义偏好提升的定义及其可实现性；同时给出对常见提升方案的预序保持性判定。

**🔧 技术方法**

使用形式逻辑工具（PDL、可满足性与完备性证明技术）以及一阶逻辑可定义性技术。

**📊 数据集**

无实验数据集，全部为理论构造与证明。

**📈 对比分析**

不涉及实验比较，性能以逻辑一致性和完备性证明为主。

**⚠️ 局限性**

仅为理论框架，缺乏实际应用案例；对无限集合、复杂社会情境的处理仍待进一步研究。

---

## 145. SkillPrivus: Uncovering PRIVacy of USe Harms in Agentic Skills

**arXiv ID:** 2611.26747 | [PDF](https://arxiv.org/pdf/2611.26747v1)

**作者:** Mariana Fernandez-Espinosa `[一作]` (University of Notre Dame), Karthikeyan Natesan Ramamurthy `[通讯]` (IBM Research)

**通讯引用:** 8698 | [OpenAlex ID](https://openalex.org/A5081874896)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种在 Agent 技能安装前通过静态分析检测隐私使用风险的审计框架。

**💡 创新点**

创新点在于将 Contextual Integrity 与扩展后的 Solove 违规分类相结合，对技能行为进行隐私适当性评估，并首次量化隐私使用违规的普遍性。

**🔧 技术方法**

使用静态代码与自然语言说明的提取、逆向切片、上下文完整性判定模型以及基于 Solove 的违规类型推理。

**📊 数据集**

在 726 个公开市场技能（共 563 个处理个人数据）的数据集上进行实验，评估其隐私违规情况。

**📈 对比分析**

与三名人工注释者对比，达成 83.2% 的一致率；发现 54.5% 的隐私相关技能存在潜在违规，表明静态审计在检测隐私风险方面效果显著。

**⚠️ 局限性**

局限包括仅做静态预估，无法验证运行时实际违规；对敏感度过滤不完善；未在更大专家样本上进一步验证。

---

## 146. Analysis of Multi-Client Inference Techniques for Vision-Language-Action Models

**arXiv ID:** 2611.26944 | [PDF](https://arxiv.org/pdf/2611.26944v1)

**作者:** Stefan de Lasa `[一作]` (University of Toronto), Nandita Vijaykumar `[通讯]` (University of Toronto)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文系统评估了多客户端推理在视觉-语言-动作模型（VLA）中的效果，重点研究批处理与阶段协同执行对延迟、吞吐量、硬件利用率和功耗的影响。

**💡 创新点**

创新点在于首次量化批处理和协同执行在VLA上的局限性，揭示VLM/ViT背骨计算密集、AE短小导致的利用率瓶颈，并说明这些技术无法像LLM那样有效抵消延迟。

**🔧 技术方法**

使用了CUDA性能分析（Nsight Compute）、GPU驱动层多流与绿通道分区、不同批量大小的实验，以及在RTX4090和H100上的多阶段调度。

**📊 数据集**

采用了随机图像和短文本提示（40 tokens）作为输入，所有模型均在DROID平台上微调，实验不涉及公开数据集，仅使用内部生成的输入。

**📈 对比分析**

实验比较显示，批处理对AE的利用率和功耗提升有限，VLM/ViT几乎不受批量影响；协同执行仅实现1.02–1.23×吞吐提升，且对AE时延有明显负面影响。

**⚠️ 局限性**

局限性包括批处理无法提升VLM/ViT利用率，协同执行受长 GEMM 核心阻塞，短AE核无法充分重叠，且研究仅在单台高端 GPU 上验证，缺乏对嵌入式平台或大规模集群的评估。

---

## 147. Coherent Moment Propagation: Epistemic Uncertainty for Pre-trained Robot Policies

**arXiv ID:** 2611.26855 | [PDF](https://arxiv.org/pdf/2611.26855v1)

**作者:** Miguel Saavedra-Ruiz `[一作]` (NVIDIA), Fabio Ramos `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CoMP（Coherent Moment Propagation）方法，解决流匹配机器人策略在推断时的方差崩塌问题，并通过增宽后层进一步提升不确定性估计；

**💡 创新点**

核心创新在于：①证明BayesDiff因忽略状态-权重交叉协方差导致方差随Euler步数减小而崩塌；②设计CoMP-EKF与CoMP-UKF两种递归，携带交叉协方差恢复总方差上界；③借鉴LULA将观察嵌入加入后层，无需再训练即提升不确定性对输入的敏感度；

**🔧 技术方法**

使用贝叶斯后验拉普拉斯近似、ODE采样的线性化（扩展Kalman/无迹滤波）、观测嵌入增强后层；

**📊 数据集**

在三大机器人模拟基准上实验：robomimic、LIBERO、RoboLab，分别采用Transformer基的Diffusion Policy、π_0.5、SmolVLA；

**📈 对比分析**

与BayesDiff、DEUA、GU、DECU、VFD以及基准ensemble进行对比，CoMP在方差校准、失败检测准确率和演示选择效率上均优于或与最强基线持平；

**⚠️ 局限性**

局限性包括：需要线性最后一层以解析权重Jacobian、存储全协方差导致内存随输出维度二次增长、推理时间随块维度线性增加，尤其在大型动作段时开销显著。

---

## 148. AppCon-Bench: Perception and Misperception of Material Appearance in Vision-Language Models

**arXiv ID:** 2611.27453 | [PDF](https://arxiv.org/pdf/2611.27453v1)

**作者:** Santiago Jimenez-Navarro `[一作]` (University of Zaragoza), Belen Masia `[通讯]` (University of Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了一套包含五个心理物理任务的材料外观常数基准，用以对比人类与多种Vision‑Language Model（VLM）在光照和几何变化下的材料判定一致性。

**💡 创新点**

首次把光泽度和金属感作为中层感知属性引入VLM评估，构建了多场景合成数据集并从人类行为出发分析VLM的成功与系统性错误。

**🔧 技术方法**

采用VQA框架将心理物理试验转化为文本问答；使用Qwen、Gemma、LLaVA、InternVL等12个VLM进行推理；人类数据通过Prolific平台收集。

**📊 数据集**

使用Mitsuba 3渲染器合成135种材料（来自MatSynth）+10种几何+6个HDR光照，生成3000个试验，随后收集21k人类与等量VLM响应。

**📈 对比分析**

通过准确率、Cohen's κ、F1/MCC、Spearman相关等指标与人类对比，发现大模型在匹配任务上仍落后于人类，但在无参考任务接近或优于人类；VLM在几何变化下鲁棒性低，光照变化下与人类相近；在多候选任务中倾向过度选择。

**⚠️ 局限性**

主要限制包括合成几何与照明的简化、未涵盖真实多样化物体；光泽/金属感的评估仅依赖主观问答，缺乏对内部表征的深入分析；未对更高级视觉任务进行评估。

---

## 149. Self-Hosted WebAssembly Runtime for Runtime-Neutral Checkpoint/Restore in Edge-Cloud Continuum

**arXiv ID:** 2611.27225 | [PDF](https://arxiv.org/pdf/2611.27225v1)

**作者:** Yuki Nakata `[一作]` (SAKURA internet Inc.), Katsuya Matsubara `[通讯]` (Future University Hakodate)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了可在任意 WebAssembly 运行时上运行的自托管运行时，实现跨运行时、跨优化策略的统一检查点/恢复。

**💡 创新点**

创新点在于将运行时本身编译为 Wasm，从而消除运行时差异，实现运行时和性能优化中立的 C/R，并通过自托管设计减少状态大小。

**🔧 技术方法**

技术包括自托管 Wasm 运行时 Chiwawa、值栈操作优化、去重沙箱、WASI 直通、使用多线程触发文件监控等。

**📊 数据集**

使用的数据集包括 n‑body 模拟、Leibniz 求 π、SQLite3 基准（100 条记录，16 字节 key、100 字节 value）。

**📈 对比分析**

比较方法：与 Wizard、Wasmtime、WAMR、WasmEdge 等现有运行时和 CRIU 对比；结果显示 Chiwawa 在自托管下比 Wizard 快约 1.3‑1.4 倍，内存占用减少 70‑85%；C/R 开销 0.13‑2.16 倍；与直接执行相比开销 13.6‑426.9 倍。

**⚠️ 局限性**

局限性：仍存在较大运行时开销、无法支持利用 WASI 的文件/套接字等 OS 资源的应用、双重栈机实现效率低、需进一步优化自托管运行时以降低开销。

---

## 150. PLoQ: A Prompt-Specific Low-Rank Basis for 4-bit Quantization of LLM Activations

**arXiv ID:** 2611.27092 | [PDF](https://arxiv.org/pdf/2611.27092v1)

**作者:** Faraz Tahmasebi `[一作]` (University of California, Irvine), Sitao Huang `[通讯]` (University of California, Irvine)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种针对大语言模型激活量化的提示适配低秩分解方法。

**💡 创新点**

在不改变权重的情况下，在线估计提示特定的低秩基，将主导激活方向在量化前提取并以额外通道恢复，从而提升 4‑bit 激活精度。

**🔧 技术方法**

使用随机范围求解器求低秩基、采样并截断量化、以及 GPU 上的融合 GEMM 与额外行实现。

**📊 数据集**

在 Llama 与 Qwen 系列模型上使用 WikiText‑2 及多项零样本评测任务。

**📈 对比分析**

与 ResQ、QuaRot、QuIK 等现有 4‑bit 量化方法对比，平均位宽约 4.2 位，恢复了 34–46% 的困惑度损失，在 8k–16k token 上保持优势，推理延迟仅相对 ResQ 增加 4–8%。

**⚠️ 局限性**

仅在 4‑bit 激活量化下有效，需额外存储 per‑prompt 基础矩阵，且对极短 prompt 的收益有限。

---

## 151. Early Assessment Submission and Subsequent Course Outcome: A Day-30 Landmark Observational Analysis of OULAD with Explicit Weighting Diagnostics

**arXiv ID:** 2611.27005 | [PDF](https://arxiv.org/pdf/2611.27005v1)

**作者:** Devis Wawan Saputra `[一作]` `[通讯]` (Independent Researcher), Devis Wawan Saputra (Independent Researcher)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对OULAD数据集的CCC/2014J课程，在30天里提交一次以上评估的学生与未提交者的期末成绩进行关联分析，并使用IPW和重叠权重对可观测协变量进行调整，完成了独立可重复的工作流。

**💡 创新点**

创新点在于：①提供了完整的独立可重复工作流程和GitHub可审计脚本；②对权重稳定性、协变量平衡和未测量混杂进行系统诊断；③在多门课程中进行探索性复现，展示了方法在不同样本下的稳健性与局限性。

**🔧 技术方法**

使用的技术包括L2正则化逻辑回归作倾向得分估计，逆概率加权（IPW）和重叠权重（overlap weighting）估计平均因果效应，Bootstrap（1000次全重拟合）产生不确定性区间，以及标准化平均差（SMD）诊断。

**📊 数据集**

数据集为Open University Learning Analytics Dataset（OULAD）UCI存档编号349（https://doi.org/10.24432/C5KK69），聚焦CCC/2014J课程的1,820名在第30天仍在校的学生。

**📈 对比分析**

比较方法为：原始风险差（0.3743）、IPW风险差（0.3861，区间0.3242–0.4447）和重叠权重风险差（0.3261，区间0.2667–0.3870）。结果显示早期提交者的通过/优秀率显著更高，但权重极值和有效样本量较低，提示估计对极端观测高度敏感。

**⚠️ 局限性**

主要局限包括：①只观察到已提交的学生，无法区分因果与关联；②对第30天生存的条件选择引入碰撞偏差；③未测量的动机、能力和机会等混杂未被校正；④小的未提交组导致权重不稳定、有效样本量低；⑤基于历史UK远程学习数据，缺乏对现代学习环境的外推能力。

---

## 152. Large Reported CHB-MIT Adaptation Gains Disappear with Pretrained Encoders and Matched Probes

**arXiv ID:** 2611.26929 | [PDF](https://arxiv.org/pdf/2611.26929v1)

**作者:** Joseph Sankoorikal Johny `[一作]` (Independent Researcher), Antony J. Sankoorikal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

对DL提出的EEG基础模型（BIOT、CBraMod）在癫痫检测中仅更新最终编码器块的自监督适配方法进行复现与审计，发现代码与协议实现导致了过大增益，并提出改进的评估协议；

**💡 创新点**

通过细粒度代码审计、对比实验以及多种随机种子和子集旋转，揭示初始化和探测器设置对自监督适配增益的关键影响，并提出统一的固定与等预算探测器调参方案以实现更可靠的性能评估；

**🔧 技术方法**

自监督学习（SSL）适配、线性探测器、统计推断（t检验、Nadeau–Bengio校正、TOST）、L2逻辑回归收敛检查、随机特征与噪声输入对照实验；

**📊 数据集**

主要使用CHB‑MIT 1.0.0（约24例、686 EDF文件）和Siena脑电数据集（14受试者、47癫痫发作）；

**📈 对比分析**

采用固定与调参探测器、多个随机种子和5次受试者旋转的对比实验，结果显示原报告的+36点增益在复现后几乎被消除（BIOT在CHB‑MIT上仅约+1.9点，CBraMod增益可视为0），整体性能低于原报告；

**⚠️ 局限性**

局限包括仅评估两种模型与单一自监督适配步骤、未使用更大数据集（如TUH）、探测器搜索未覆盖所有配置、统计检验受旋转重叠影响、代码非位复现、部分实验设置在后期调整导致结果可变。

---

## 153. Parameter-Efficient Adaptation of Qwen3-VL for Handwritten Mathematical Expression Recognition

**arXiv ID:** 2611.27208 | [PDF](https://arxiv.org/pdf/2611.27208v1)

**作者:** Manh Tuan Ha `[一作]` (FPT University), Van Bay Nguyen `[通讯]` (Ho Chi Minh City Open University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对Qwen3‑VL进行参数高效微调（LoRA/QLoRA+4‑bit量化），让其在CROHME 2019手写数学表达式数据集上以指令式生成线性化符号标签图（SLG），并对比零样本、ViT‑Hybrid等基线。

**💡 创新点**

证明即使是4‑bit 4B规模的多模态模型，也能通过轻量级适配器实现高质量的结构化输出；提供了可复现的、资源友好的HMER框架，并通过控制性对比实验揭示了合成样本对分布匹配的影响。

**🔧 技术方法**

LoRA/QLoRA低秩适配器、4‑bit量化、指令式多模态微调、线性化SLG表示、Transformer解码器、BLEU‑4与exact‑match评估。

**📊 数据集**

CROHME 2019正式竞赛数据集（8.9K训练样本 + 1,198测试样本）以及额外的2,000个基于公式生成的合成手写样本。

**📈 对比分析**

与零样本Qwen3‑VL、ViT‑Hybrid基线以及Qwen2.5‑VL、IM2TEX本地实现进行对比。fine‑tuned Qwen3‑VL在CROHME测试集上达到43.74% exact‑match和77.67 BLEU‑4，显著优于ViT‑Hybrid（4.92%/43.26）和零样本（0%/1.32）。加入2K合成样本略降性能（42.32%/75.46）。

**⚠️ 局限性**

仅使用单一数据拆分和单一随机种子；配置有限，未做统计显著性检验；使用BLEU而非官方CROHME图形评估；缺乏多种seed、多模型平均或官方评测工具，未考虑更精细的错误分类。

---

## 154. GAAFormer: Geometry-Anchored Attention Transformer for Joint Point Cloud Denoising and Upsampling

**arXiv ID:** 2611.27281 | [PDF](https://arxiv.org/pdf/2611.27281v1)

**作者:** Haoxi Tian `[一作]` `[通讯]` (Southwest Petroleum University), Haoxi Tian (Southwest Petroleum University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种统一的几何锚定注意力变压器（GAAT）生成八叉树解码器，能够同时完成点云去噪和上采样。

**💡 创新点**

创新点在于将注意力窗口与空间坐标绑定，并采用交替窗口偏移保证解码过程中残留节点的邻域不变，从而实现删除一致性。

**🔧 技术方法**

使用生成八叉树变压器、几何锚定窗口注意力、共享编码器-解码器结构以及相对/卷积位置编码技术。

**📊 数据集**

评估数据集包括PUNet、PU-GAN、PU1K和Sketchfab。

**📈 对比分析**

与现有去噪/上采样方法（如PointCleanNet、PU-Net、PU-GAN等）对比，GAAT在PUNet去噪时平均Chamfer距离和Hausdorff距离分别下降约10%和15%，在PU-GAN和Sketchfab上上采样的Chamfer、Hausdorff和点对表面距离均优于同类方法，PU1K上表现略逊一筹。

**⚠️ 局限性**

局限在于对极稀疏或极大规模点云的泛化尚未充分验证，且窗口偏移策略会略微增加推理延迟，整体效果在不同数据集间不完全一致。

---

## 155. Testing Settlement Flags after Late Evidence and Process Restart

**arXiv ID:** 2611.26665 | [PDF](https://arxiv.org/pdf/2611.26665v1)

**作者:** Zengjian Liu `[一作]` (Peachy Lab), HongLiang Sun `[通讯]` (Peachy Lab)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文研究了在应用程序重启后，结算标志如何保持持久，同时评估了在不同实现（文件系统和SQLite）下对支付日期冲突与等待状态的处理。

**💡 创新点**

创新点在于提出了一个基于“固定计划、报价会话、来源合格、身份键值账本影响及后续运行状态”的可观测合同，并通过合成历史记录和双实现交叉验证，揭示了文件实现对支付日期冲突的容忍性与验证诊断误判的问题。

**🔧 技术方法**

主要技术包括：合成历史生成器、确定性判据（oracle）、文件系统实现、SQLite实现、运行时诊断（终端标志检查、阶段级检查）以及完整的记录重放与对比分析。

**📊 数据集**

使用的数据集为 504 条合成历史记录（共 1,008 次实验单元，覆盖不同证据家族、时间截点、交付顺序和调用调度），每条历史记录都在文件和 SQLite 两种实现上重复执行。

**📈 对比分析**

比较方法是对每条历史记录在两实现中执行完毕后分别从新进程读取初始与四次尝试后的状态，并与预期合同进行一致性对照。文件实现正确率为 450/504，SQLite 实现为 504/504；在支付日期冲突场景下，文件实现错过 54 对冲突，诊断误判率高于 SQLite，阶段级检查虽然能预测等待状态，但无法区分冲突。

**⚠️ 局限性**

局限性包括：仅在单写入者、核心与文件系统仍在运行的实验环境下测试；未覆盖电源故障、并发写入、原生回放等真实错误情形；合成历史可能未完全反映实际业务场景；实现和判据共享同一合同，可能在实现细节差异上出现共同错误。

---

## 156. What Helps Language-Model Agents Review Violent Video? A Controlled Comparison with a Human Evidence Reference

**arXiv ID:** 2611.26776 | [PDF](https://arxiv.org/pdf/2611.26776v1)

**作者:** Mohammed El Amine Mokhtari `[一作]` (University of Mons), Virginie Vandenbulcke `[通讯]` (University of Mons)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5007157534)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对固定感知代理与推理模型，比较消息传递、证据图、平面文本与单一音视频模型在判断是否需要人工查看暴力视频时的性能；

**💡 创新点**

通过只改变信息传递接口而非模型本身，发现一次性平面文本读取与单模型同样准确且优于结构化接口，同时构建人类参考记录证据使用方式，揭示声学信息对判定的贡献；

**🔧 技术方法**

使用多模态感知代理（视频描述、物体检测、声响事件检测、语音识别+情绪标签）、大型语言模型（Gemma‑12B、Qwen3.6‑35B）以及工具调用的图代理、平面文本提示与消息传递；

**📊 数据集**

采用 XD‑Violence 测试集（含视频与音频）并补充人工参考标注；

**📈 对比分析**

以 AUC、F1、帧级 AP 与定位精度等指标进行对比，结果显示单模型与平面文本在 F1 约 0.93 左右（差异 <0.01），图代理与消息传递约 0.90，声音事件提升 F1 约 0.03；

**⚠️ 局限性**

仅在 XD‑Violence（主要为电影/网络视频）上评估，缺乏真实监控数据；结构化图在信息冲突场景下未测试；工具调用和边缘生成受固定规则限制；总体计算成本高于单模型。

---

## 157. SCAP: Stochastic Context for Action Expert Pretraining

**arXiv ID:** 2611.27187 | [PDF](https://arxiv.org/pdf/2611.27187v1)

**作者:** Xiao Lin `[一作]` (Tongji University), Qijun Chen `[通讯]` (Tongji University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种新的行动专家预训练框架 SCAP，通过在预训练阶段随机遮掩视觉与语言输入，随后再用完整输入进行微调，以避免视觉捷径并保持视觉-语言协同建模。

**💡 创新点**

创新点在于发现单模态预训练会关闭视觉-语言的联合路径，并通过随机上下文遮掩的方式在预训练阶段同时保留两模态的监督，从而在保持视觉-语言互补的同时减少对视觉捷径的依赖。

**🔧 技术方法**

采用的技术包括：视觉语言模型（VLM）冻结的多模态Transformer；随机遮掩（masking）策略；动作专家（action‑expert）与 VLM 的联合训练；以及在实验中使用的 AdamW 优化器、学习率退火和梯度剪裁等。

**📊 数据集**

数据集包括：RoboTwin 2.0（50 份双臂操作任务）、RoboCasa‑GR1（24 份桌面任务），以及约 120 万条真实与合成机器人轨迹；此外在真实机器人实验中使用 300 条演示数据。

**📈 对比分析**

与全输入预训练、视觉‑动作（VA）预训练、语言‑动作（LA）预训练及其组合等传统方法比较，SCAP 在两大仿真基准上分别提升了 14–11% 的成功率，在真实机器人三类操作任务中平均成功率提升至 76.7%，比全输入预训练高出约 25%。

**⚠️ 局限性**

限制主要体现在：① 随机遮掩比例需要在 20–40% 之间才能兼顾去除捷径与保留联合路径；② 目前仅在单一动作专家架构上验证，跨模型、跨任务的通用性待进一步评估；③ 对极端遮掩或完全缺失模态的鲁棒性尚未深入探讨。

---

## 158. Joint SLA Admission and Risk Allocation for Cloud Resource Provisioning

**arXiv ID:** 2611.26924 | [PDF](https://arxiv.org/pdf/2611.26924v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 159. Re-JEPA: Coupling Representation Prediction and Signal Reconstruction for State-Conditioned EMG Tracking

**arXiv ID:** 2611.27299 | [PDF](https://arxiv.org/pdf/2611.27299v1)

**作者:** Yijian Yang `[一作]` (Chinese Academy of Sciences), Shilei Wen `[通讯]` (SynapX)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究提出一种新的自监督预训练框架Re-JEPA，用于表面肌电（sEMG）连续手部姿态跟踪。

**💡 创新点**

创新点在于将表示预测（JEPA）与信号重建（幅度与相位）耦合，并设计了状态条件的门控持续运动（GPM）解码器，使得预训练的表示既保留了细粒度信号信息，又能被高效解读为手部运动。

**🔧 技术方法**

使用的技术包括多层Transformer编码器、指数移动平均（EMA）目标编码器、联合表示预测与幅度/相位重建损失、以及基于注意力和门控机制的递归GPM解码器。

**📊 数据集**

数据集为emg2pose，包含约370小时、193名用户、29种运动阶段的高密度EMG与同步运动学标注。

**📈 对比分析**

与现有方法（如vemg2pose、LaBraM、NHN）的比较显示，Re-JEPA+GPM在交叉用户、交叉阶段和联合泛化拆分上的角度平均误差分别降至6.91°、9.04°和9.36°，比vemg2pose降低10.3%–19.3%，并且在冻结表示与全微调实验中均优于对照模型。

**⚠️ 局限性**

局限性包括：难以明确重建约束具体保留了哪些信号细节；重建与表示预测的互补作用尚未单独评估；对其他生理信号的适用性未验证；以及在临床部署前仍需解决用户个体差异、预测失败与隐私安全等问题。

---

## 160. Portable Active Recovery under Timed and Position-Dependent Visual Outages

**arXiv ID:** 2611.27121 | [PDF](https://arxiv.org/pdf/2611.27121v1)

**作者:** Jinyuan Zhang `[一作]` (Hubei University), Shengshuo Jiao `[通讯]` (Hubei University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种可携带的视觉失效恢复模块，能够在不重新训练的情况下直接挂载到现有视觉‑惯性飞控器上，实现多秒摄像头失效时的主动可观测性恢复；

**💡 创新点**

核心创新在于将恢复逻辑置于参考接口层，利用仅部署可观测的遥测和轨迹参考完成退避、后退探测、重新锁定与高度积分补偿，且在未知失效机制下通过单一准则实现定量最坏情况性能提升；

**🔧 技术方法**

使用了视觉‑惯性里程计 Basalt、基于轨迹的 MPPI 与 SimpleFlight 控制器、ORB‑SLAM3 先验地图投影、四次多项式制动、后退探测、倾斜补偿与高度积分控制等技术；

**📊 数据集**

在预注册的 48 场景（共 1,920 次飞行）上进行评测，另外使用 EuRoC MAV 真实黑窗口数据验证地图补偿的偏差‑漂移交叉；

**📈 对比分析**

与六个外部基线（包括原生 MPPI、SimpleFlight、DATT‑L1、L1‑MPPI、LOTF 等）在最坏情况完成率上进行 Bonferroni 校正后显著优于所有基线；在不重新训练的情况下，模块能将 SimpleFlight 的最坏完成率从 66% 提升至 86.5%，并在两种速度下实现 90% 的完成率且无硬失败；

**⚠️ 局限性**

局限包括：仅在简化的圆形轨迹与理想仿真环境下验证，实际物理飞行与障碍环境尚未测试；恢复动作导致额外 9–14 秒的飞行时延，需结合障碍规避；模块依赖稳定的视觉‑惯性估计，极端光照或纹理缺失场景可能仍失效。

---

## 161. Learning 3D MRI Representations with Generative Adversarial Networks

**arXiv ID:** 2611.27057 | [PDF](https://arxiv.org/pdf/2611.27057v1)

**作者:** Sujeet Lahane `[一作]` (IIT Delhi), Bharat Biswal `[通讯]` (NJIT)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本工作提出了一种基于生成对抗网络的自监督重建方法 R2RG，用以学习 3D MRI 的高质量表示。

**💡 创新点**

创新点在于将自编码器与 PatchGAN 判别器相结合，利用复合 L1+L2+GAN 损失实现高保真重建，并通过无监督重建提升表示质量，解决了 3D 医学影像数据稀缺问题。

**🔧 技术方法**

使用了 3D 卷积自编码器、PatchGAN 判别器、Wasserstein 损失、梯度惩罚以及 L1/L2 组合损失；在训练时采用 Adam、PatchGAN 反馈实现稳定收敛。

**📊 数据集**

实验使用公开的 AOMIC、ADNI1、PPMI 三个脑 MRI 数据集以及本地收集的 COVID-19 恢复组数据，共计约 4000 个 3D 卷。

**📈 对比分析**

与 VAE、α-WGAN、SimCLR、BYOL 等基线比较，R2RG 在 MSSIM/PSNR 上均高出 5%~10%，在 PPMI/ADNI1 的分类任务中精度提升约 10% 以上，显示出在小样本场景下更优的表示与下游性能。

**⚠️ 局限性**

局限性包括：表示空间高度纠缠，难以解释；缺乏大规模多样化未标记数据，可能影响进一步泛化；训练与推理仍比纯监督方法慢，且模型参数量相对较大。

---

## 162. MomentMem: Memorizing Key Moments for Robot Manipulation

**arXiv ID:** 2611.26745 | [PDF](https://arxiv.org/pdf/2611.26745v1)

**作者:** Guangyu Zhao `[一作]` (Peking University), Yiwu Zhong `[通讯]` (Peking University)

**通讯引用:** 4821 | [OpenAlex ID](https://openalex.org/A5040199499)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了视觉时刻记忆（visual moment memory）框架，解决机器人在长时程任务中需要保留短暂可见的精细视觉信息的问题。

**💡 创新点**

创新点在于：①提出专注于视觉时刻记忆的 VMBench benchmark；②提出 MomentMem 的分层架构，使用 VLM 记忆器挑选稀疏关键帧并将其与执行器分离；③通过段级监督和排程采样提升记忆选择的鲁棒性。

**🔧 技术方法**

技术实现包括：基于 Qwen3‑VL‑4B 的 VLM 记忆器（输出“Yes/No”是否存储帧），基于 π_0.5 的执行器（从关键帧和当前观测生成动作），段级监督、排程采样、焦点损失，层级训练流程。

**📊 数据集**

数据集为：VMBench（9 个闭环操纵任务），每个任务 100 条成功演示；还在 RMBench 以及 Franka Panda 实体机器人上进行额外评估；视觉 OOD 版本用于测试外观迁移。

**📈 对比分析**

对比方法包括 π_0.5、Mem‑0、NativeMEM、EventVLA 等基线。MomentMem 在 VMBench 上平均成功率 49.9%，比最强基线高 31.7pp；在视觉 OOD 上 55.2%；在 Flash Beacon 的少样本迁移中 43.7%（接近完整监督 44.3%）；在 RMBench 上 69.8%，略优于 EventVLA 67.9%；在真实机器人任务上显著提升。

**⚠️ 局限性**

局限性：记忆器为 VLM 需要额外推理开销；在纹理匹配任务中执行器仍表现不佳；实验仅使用单一硬件与固定相机视角，未检验视角变化对性能的影响。

---

## 163. Don't Discard Your Rollouts: Reusing Teacher RL Traces for Student Distillation

**arXiv ID:** 2611.26803 | [PDF](https://arxiv.org/pdf/2611.26803v1)

**作者:** Vashisth Tiwari `[一作]` (Carnegie Mellon University), J. Zico Kolter `[通讯]` (Carnegie Mellon University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种利用强化学习训练过程中产生的回放数据来进行学生模型蒸馏的方法，避免了传统蒸馏中额外生成数据的高昂成本。

**💡 创新点**

创新点在于将教师RL回放的统计信息（提示通过率和缓存的对数似然）视为数据选择依据，设计了SFT热身+DPO的两阶段蒸馏流程，实现了零额外计算成本且学生无关的回放重用策略。

**🔧 技术方法**

采用的技术包括Group Relative Policy Optimisation（GRPO）训练教师、基于教师回放的SFT与Direct Preference Optimization（DPO）蒸馏，以及基于提示通过率的学习曲线排序。

**📊 数据集**

实验使用数学推理数据集Math12K、DeepScaler以及编程数据集AceCode，评测时使用MATH500、GSM8K、AIME、HumanEval+和MBPP+等基准。

**📈 对比分析**

与传统的“重新生成教师输出后再蒸馏”方法相比，该两阶段回放重用方案在数学和编程任务上平均仅落后0.4分，甚至在部分学生模型上与传统方法相当，且在低基础支持的学生上表现出更高的鲁棒性。

**⚠️ 局限性**

局限性包括：实验仅覆盖4B-8B教师到1.2B-4B学生的容量范围；仅考虑可验证的二进制奖励场景；未在更大规模或不同任务类型上验证；以及每个配置仅使用单一训练种子。

---

## 164. Zing-1.0: A Playable World System Built on Controllable Rendering

**arXiv ID:** 2611.27336 | [PDF](https://arxiv.org/pdf/2611.27336v1)

**作者:** Wenbo Zhang `[一作]` (SeedLeap.ai), Yiwen Cai `[通讯]` (SeedLeap.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可玩世界系统 Zing‑1.0，将世界状态演化与观测合成分离，通过可编程引擎执行命令、生成代理视频，再由流式生成器渲染最终视图，支持玩家实时交互。

**💡 创新点**

核心创新在于“Controllable Rendering”原则，明确将状态更新与视觉生成分离；对可玩世界系统给出四大责任和接口规范；结合 agent‑based 世界构造、可插拔代理视频和参考条件的流式生成器，实现既能保证物理规则又能自由生成视觉风格。

**🔧 技术方法**

技术主要包括：1）WorldKit 引擎实现可编程世界构建与代理视频；2）MiniMax‑H3 变形生成器改造为块因果流式渲染器，支持场景/主体参考与文本条件；3）SGLang 推理框架，采用跨块联合前向、稀疏注意力与 KV 压缩提升 3‑GPU 系统的 23.82 FPS 性能；4）多阶段流水线与协议控制保证实时交互。

**📊 数据集**

训练与评测使用公开视频编辑基准：OpenVE‑Bench、LongV2VBench、StreamEval；同时在自研代理世界上进行内部可玩性测试；在训练阶段利用 MiniMax‑H3 预训练数据和多模态视频编辑数据。

**📈 对比分析**

与现有流式编辑器（JoyAI‑Video‑Edit、LiveEdit、SANA‑Streaming 等）在 OpenVE 和 LongV2VBench 上分别取得 3.90/3.44 分，领先所有基线；在 StreamEval 指令/参考编辑赛道上，整体得分 57.86/54.06，超越官方榜单；系统实现 832×480 23.82 FPS，TTFF 0.70s，显著降低推理延迟并提升吞吐量。

**⚠️ 局限性**

局限性包括：1）长时序渲染误差累积导致视觉连贯性下降；2）引擎与渲染器之间职责划分不够细粒度，导致某些交互细节难以一致；3）对更丰富的条件输入（深度图、语义图、视频参考）支持不足；4）可编程世界构造仍受限于现有组件库，难以快速扩展多样化玩法；5）整体交互延迟仍受网络、渲染与代理捕获等多环节影响。

---

## 165. Contexture: Runtime Context Eviction for High-Throughput Agent Serving

**arXiv ID:** 2611.26980 | [PDF](https://arxiv.org/pdf/2611.26980v1)

**作者:** Colin Hong `[一作]` (Nanyang Technological University), Dmitrii Ustiugov `[通讯]` (Nanyang Technological University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在LLM代理服务中执行上下文淘汰的运行时架构，能够在不修改agent harness和模型的前提下，动态移除已被后续工具操作“超写”或“覆盖”的历史记录，从而减少共享KV缓存的内存占用。

**💡 创新点**

创新点：① 通过对工具执行记录的“超写”规则识别可删除的历史片段；② 将这些删除决策与运行时全局KV压力和缓存驻留信息结合，设计“cold commit”和“pressure gate”两种时机策略；③ 通过“grace window”和“re‑prefill cap”控制删除的粒度与重填成本，保持任务准确率。

**🔧 技术方法**

使用技术：工具操作映射与超写检测、grace window、cold commits、pressure gate、re‑prefill cap、vLLM的Session Evictor与Context Filter、KV cache manager的驻留探测（residency probe）、GPU HBM KV池管理。

**📊 数据集**

使用数据集与模型：Terminal‑Bench 2.1 和 SWE‑Atlas‑QnA 这两个长周期任务基准；模型包括 Qwen3.8‑27B（密集模型）和 Qwen3.6‑35B‑A3B（MoE模型），分别部署在 NVIDIA H100 与 H200 GPU。

**📈 对比分析**

方法比较：与无淘汰、harness级别的 compaction、AgentDiet、Continuum 以及 observation masking 等基线对比；实验表明在满足 10×p90 延迟 SLO 的前提下，吞吐量提升 20–73%，GPU 能耗下降 2–11%，任务准确率仅相对无淘汰下降 0–3.7%。

**⚠️ 局限性**

局限性：① 目前仅支持文件、命令、构建、测试等常见工具的超写规则，其他 API 或服务类型不覆盖；② 删除策略需调参（grace window、压力阈值、re‑prefill cap），不同工作负载可能需要手动调整；③ 在极高负载或大模型（如 MoE）下，重填成本仍可能显著影响实时性；④ 需要在 vLLM 运行时插桩，部署复杂度略高。

---

## 166. HookEvo: A Framework for Coordinated Plugin Evolution on Hook-Enabled Agent Runtimes

**arXiv ID:** 2611.27294 | [PDF](https://arxiv.org/pdf/2611.27294v1)

**作者:** Shengze Huang `[一作]` (Tsinghua University), Baotong Mi `[通讯]` (Renlab AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发 HookEvo 框架，利用工具使用代理的生命周期钩子实现协同编辑，改进代理的 harness 以提升任务成功率。

**💡 创新点**

允许单个候选方案同时修改记录工具反馈的处理器和使用该反馈的后续处理器，实现跨步骤的协同修复，并提供统一的候选空间与评估机制。

**🔧 技术方法**

使用生命周期钩子、插件化 harness、执行追踪诊断、基于评估的候选保留，以及 frozen Qwen3-4B‑Instruct 执行器；采用三种编辑范围（A、B、C）与三位改进者（GPT‑6、Luna、Terra、Sol）。

**📊 数据集**

四大基准：ALFWorld、DBBench、WebShop、OSBench（各自训练/验证/测试划分），并使用 Qwen3-4B executor 进行评估。

**📈 对比分析**

与 Life‑Harness、ALIGN、AgentSquare、Meta‑Harness、AHE 等前置工作在相同 executor 下公开结果进行对比；HookEvo 的 Sol‑C 在 ALFWorld、DBBench、WebShop、OSBench 上分别达到 95.41%、70.56%、48.17%、41.67%，在所有公开 Qwen3‑4B 结果中排名第一。

**⚠️ 局限性**

训练任务分配不一致导致改进者暴露的轨迹差异；未明确区分单独与协同编辑对性能的贡献；未评估不使用记录/读取处理器对任务成功的影响等。

---

## 167. Forehead ROI Localization under Real-World Imaging Degradations via Cross-View Pseudo-Labeling

**arXiv ID:** 2611.26712 | [PDF](https://arxiv.org/pdf/2611.26712v1)

**作者:** Geetanjali Sharma `[一作]` (Indian Institute of Technology Mandi), Raghavendra Ramachandra `[通讯]` (Norwegian University of Science and Technology)

**通讯引用:** 9820 | [OpenAlex ID](https://openalex.org/A5073391891)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一种置信度与跨视图几何一致性引导的伪标签生成与微调框架（CCGP），用于利用未标注的面部图像提升前额ROI定位性能。

**💡 创新点**

创新点在于：①在伪标签筛选中同时结合置信度阈值和水平翻转后几何IoU/中心一致性；②通过置信度加权融合获得更稳健的伪框；③仅对训练数据做筛选，而不改造Grounding DINO模型结构，保持模型兼容性。

**🔧 技术方法**

使用的技术包括：Grounding DINO目标检测器、水平翻转一致性检查、置信度加权融合、交叉数据集与质量分层评估、mIoU/精确率/召回率等评测指标。

**📊 数据集**

使用的数据集有：FH‑V1（手工标注）、CelebA（伪标签生成）、FH‑V2、LFW、AgeDB、CMFD、RMFRD等多种含遮挡与自然图像的面部数据集。

**📈 对比分析**

方法通过与YOLOv7/8、Mask R‑CNN、DiffusionInst以及监督版Grounding DINO等基线进行交叉数据集与质量分层比较；在四个外部数据集上取得最高mIoU（最高达0.763），在RMFRD上从0.415提升至0.580；在模糊和低对比度条件下提升尤为显著。

**⚠️ 局限性**

局限性包括：①域内性能仍略低于纯监督Baseline；②几何一致性阈值未显著减少伪标签集；③缺乏对不同阈值/策略的细粒度消融；④推理速度极低（约0.62 FPS）；⑤对大偏斜姿态仍易失效。

---

## 168. Many Plans at Once: Diverse Top-Quality Planning in Continuous Time

**arXiv ID:** 2611.27012 | [PDF](https://arxiv.org/pdf/2611.27012v1)

**作者:** Mahyar Jahani-nasab `[一作]` (Monash University), Buser Say `[通讯]` (Monash University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在连续时间域内同时生成大量低成本且相互不同轨迹的规划方法，能够在预先计算好一组可备选轨迹并支持在线快速适配。

**💡 创新点**

创新点包括：①将（q, δ）-plan集合作为多方案规划目标；②使用基于边界条件构造的神经基底轨迹与共享解码器进行全局梯度共享，确保每条轨迹满足约束且可快速收敛；③通过动态宽化、替换和归档机制自适应采样，保证算法具有 anytime 性质；④引入三阶段验证与区间算术保证计划合法性。

**🔧 技术方法**

使用的技术主要是：神经网络（无训练的基底采样器、共享解码器）、梯度下降（Adam、余弦学习率调度）、区间算术验证、轨迹采样与变形、统计距离度量（RMS），以及基于多家族的分布式采样和自适应更新。

**📊 数据集**

实验数据集涵盖四个连续规划任务：追踪-躲避（pursuit–evasion）和三种接触丰富的操纵任务（Push、Slide、Pick），每个任务生成10个测试实例。

**📈 对比分析**

与五种基线方法（IPOPT、CEM、MPPI、SVMPC、PING）在相同预算下对比，MAAO 在所有40个实例上均能找到可行解，且在成本 25% 内得到的可行轨迹数量是基线的十倍以上；在追踪-躲避任务中，虽然在较严格的相异阈值下与 IPOPT 竞争，但整体在方案数量和最优成本上占优。

**⚠️ 局限性**

局限性包括：计算成本高（每实例 6-38 分钟，适合离线预计算）；算法更倾向于方案多样性而非多条近最优解；归档方案未进行进一步局部优化，导致部分方案在更严格成本阈值下不具备竞争力；对高层策略的多样性评估不够充分，需结合拓扑或离散策略度量。

---

## 169. RoboDyna: A Benchmark for Evaluating Robotic Manipulation under Physically Grounded Dynamics

**arXiv ID:** 2611.26969 | [PDF](https://arxiv.org/pdf/2611.26969v1)

**作者:** Zhiyuan Li `[一作]` (Huawei Technologies), Amir Rasouli `[通讯]` (Huawei Technologies)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Robodyna基准，用以评估机器人在动态物理环境下的感知、预测与动作能力。

**💡 创新点**

创新点在于结合物理驱动的动态任务、同步/异步评估协议、可调节难度的概念任务与真实世界家庭任务，形成完整的动态学习与评估框架。

**🔧 技术方法**

使用SAPIEN物理仿真、双臂UR5、RGB-D感知，并对比了两个视觉语言动作模型（π_0.5、X‑VLA）和一个世界动作模型（FastWAM）的预测与控制。

**📊 数据集**

数据集包含3500条专家演示，覆盖20个概念任务与10个家庭任务，分别包含四级难度与20个未见组合，配有语言指令与物理参数随机化。

**📈 对比分析**

通过同步/异步两种评估协议，测量成功率和进度分数，三种模型在所有任务中的平均成功率低于20%，而人类在相同条件下成功率超过70%。

**⚠️ 局限性**

局限性包括对复杂物理相互作用预测不足、对执行时延敏感、缺乏对更复杂或真实机器人环境的泛化，以及基准仍以仿真为主，真实世界验证受限。

---

## 170. ManipScope: A Diagnostic Benchmark for Instruction Following and Generalization in Robotic Manipulation

**arXiv ID:** 2611.27305 | [PDF](https://arxiv.org/pdf/2611.27305v1)

**作者:** Jiacheng Li `[一作]` (University Of Science And Technology Of China), Jiahu Qin `[通讯]` (University Of Science And Technology Of China)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一个评估仅诊断的基准（ManiScope），通过在未见的 48 个场景中检测机器人在变更任务需求下的指令跟随能力。

**💡 创新点**

创新点在于使用源任务链接的 held-out 场景、分层诊断链（选择、完成、预测与内部干预）以及对指令跟随失败原因的系统性剖析。

**🔧 技术方法**

采用了视觉–语言预训练的机器人控制策略、RoboTwin 2.0 的模拟演示、视频预测和内部干预技术，以及工具使用代理。

**📊 数据集**

数据集基于 RoboTwin 2.0 的演示，生成 48 个 held-out 场景（六类任务），未提供任何场景演示，确保评估仅靠已有技能。

**📈 对比分析**

对 12 个政策检查点和 5 个工具代理进行 13,200 条评估，整体成功率约 40%，但指令跟随准确率仅 3.9%，远低于 GPT‑6 代理的 88%；表明大多数政策在目标选择上未真正跟随指令。

**⚠️ 局限性**

局限性包括仅在仿真环境中验证、缺乏真实机器人测试、诊断依赖可观测事件且内部干预未评估执行效果。

---

## 171. On the loss landscape of the linear neural network: beyond critical points

**arXiv ID:** 2611.26671 | [PDF](https://arxiv.org/pdf/2611.26671v1)

**作者:** Antoine Lejay `[一作]` (Université de Lorraine), Sa{ï}d Toubra `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

论文通过引入中间结构“临界thalweg”对深层线性网络的损失地形进行全局几何描述，并给出了从层级参数空间到低秩矩阵空间的映射，实现了对所有临界点的显式表征。

**💡 创新点**

创新点在于：①使用thalweg理论统一描述临界点并证明不存在伪局部最小值；②利用正交投影与谱分解，将复杂高维问题降维为投影矩阵的优化；③构造连续路径显示所有临界点可通过下降轨迹连至全局最优。

**🔧 技术方法**

主要技术包括：深层线性网络的梯度分析、矩阵投影与投影矩阵的几何性质、谱分解与正交投影、Stiefel流形上的极值分析、以及 (ρ,θ) 极坐标投影可视化。

**📊 数据集**

实验使用随机生成的输入矩阵 X∈ℝ^{300×250}（均匀、Gamma、正态、泊松分布）以及 Y = X + ϵ，ϵ 服从 N(0, 1/10) 的噪声；维度设置为输入 300，隐藏 250，输出 200。

**📈 对比分析**

通过在 50 次实验中执行 2000 步梯度下降，并在最后 20 步计算各项指标（ρ、θ、距离度量、投影误差、梯度范数等），结果显示梯度下降最终会沿临界thalweg收敛，损失继续下降，且 R² 值逼近最大值；未与其他优化器做直接对比，实验主要验证了理论预测的可行性。

**⚠️ 局限性**

局限性包括：仅针对完全线性网络；理论和实验假设输入矩阵 X 具备满列秩；未处理正则化、非线性激活或残差连接；实验规模受限于随机数据，缺乏对真实数据集的验证；以及对梯度下降路径的理论与实际匹配仅在统计意义上进行，而未给出收敛速度或实现细节。

---

## 172. MixFold: Sub-4-Bit Mixed-Precision LLM Inference via Output-Channel Grouping on Mobile NPUs

**arXiv ID:** 2611.26976 | [PDF](https://arxiv.org/pdf/2611.26976v1)

**作者:** Vinnam Kim `[一作]` (Qualcomm Technologies, Inc.), An Chen `[通讯]` (Qualcomm Technologies, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种在移动NPU上高效执行子4位量化与混合精度LLM推理的方法，利用原生整数MMA和输出通道分组实现无权重量化转换的低位宽计算。

**💡 创新点**

核心创新包括：
1) 针对非原生低位宽格式（W2-SEQ、W3）提出的行/累加/激活折叠技术，使其在原生2位整数MMA上直接执行；
2) 将混合精度细化到与硬件tile对齐的输出通道组，利用整数矩阵块只用一种位宽，避免了不同位宽混用导致的布局破坏；
3) 通过二阶Hessian敏感度估计与ILP求解，实现按组最优位宽分配，显著提升同位宽预算下的模型准确性。

**🔧 技术方法**

使用的技术包括：
- Qualcomm Hexagon NPU的整数MMA、HVX向量单元与VTCM共享内存；
- W2-SEQ的bias/accumulator折叠，W3的bit‑plane折叠与LUT加速；
- 组级混合精度分配的二阶量化损失估计与整数线性规划；
- GPTQ与FlexRound量化、FlashAttention等软件层技术。

**📊 数据集**

主要评测数据集与模型：
- Qwen3系列LLM（0.6B、1.7B、4B、8B）;
- 评测任务包括WikiText‑2、ARC‑Challenge/ARC‑Easy、HellaSwag、OpenBookQA、PIQA、WinoGrande、MMLU、IFEval等。

**📈 对比分析**

与 llama.cpp 的 Q2_K、Q3_K_S、Q3_K_M 以及 Q4_0 进行对比，采用相同BPW（bits/weight）进行匹配；
- 解码吞吐量：在所有模型尺寸上，所提方案在相同BPW下比 llama.cpp 提升 1.16–1.41×；
- 能耗效率：tokens/焦耳提升 1.5–4.3×；
- 准确性：在匹配BPW时，准确率与 llama.cpp 相当或更好，且在混合精度分配上实现了更优的精度–吞吐 Pareto 前沿。

**⚠️ 局限性**

限制与不足：
- 方案依赖于具备原生2位整数MMA、共享VTCM和高速向量单元的NPU，其他NPU可能无法直接迁移；
- W3 的额外MAC和LUT开销导致能耗略高于纯原生4位方案；
- 对于较大 M（prefill）时，折叠技术的收益减弱；
- 仅在 Qwen3 系列模型上验证，未在更大规模或不同架构模型上测试。

---

## 173. Scoliosis5K: Multimodal Gait Analysis via Clip-Conditioned RGB-Silhouette Fusion

**arXiv ID:** 2611.26849 | [PDF](https://arxiv.org/pdf/2611.26849v1)

**作者:** Zizhao Peng `[一作]` (Hong Kong Polytechnic University), Fengwei An `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 677 | [OpenAlex ID](https://openalex.org/A5069123107)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了同步的 RGB、二值剪影和二维姿态标记的 Scoliosis5K 数据集，并提出一种多模态模型，结合基于剪辑动态的低秩 RGB 编码、剪影变换器以及基于参与者平衡的区域核回归，对脊柱侧弯筛查的三类标签进行视频级分类。

**💡 创新点**

创新点包括：① 通过像素偏差门控的低秩适配器对预训练 RGB 编码器进行剪辑级动态调节；② 用同步的 RGB 剪辑对剪影编码器进行蒸馏，强化跨模态监督；③ 在原始分辨率单帧上使用参与者平衡的区域核方法，兼顾高分辨率外观相似性；④ 系统性评估剪影对正类召回率的影响，揭示不同模态对分类偏差的作用。

**🔧 技术方法**

技术手段：DINOv2 ViT-S/14 预训练编码器 + 低秩门控适配器；空间‑时间剪影 transformer；基于 4×2 区域网格的核岭回归；多模态特征融合；五折参与者独立交叉验证；与多种现有 silhouette、光流、RGB 与多模态基线进行对比实验。

**📊 数据集**

数据集：Scoliosis5K（4,998 条走路序列，3,010 名参与者），其中 988 名单标签参与者用于五折交叉验证；每条序列提供原始 RGB crop、64×64 二值剪影和 17 维 2D 关键点；同一协议下的 Scoliosis1K 用于基线重训与对比。

**📈 对比分析**

与 13 种基线（Silhouette、光流、RGB、Multimodal）进行同一数据集上的对比。最终模型在 988 名受试者上取得 83.77% 准确率、79.91% 宏 F1、0.950 宏 AUROC；在冻结编码器或去除剪影、RGB 输入时的消融实验进一步说明了各模态的贡献；在区域核基准上单独实现可达 82.79% 准确率。

**⚠️ 局限性**

局限性：未验证在未见临床中心的泛化能力；未将预测结果与放射学 Cobb 角度关联；在裁剪头部遮蔽下的表现未充分评估；数据仅来自走廊协议，可能无法覆盖更广泛的行走环境；并未针对不同采集站点的标签不平衡进行专门处理。

---

## 174. Belief-State Management Meets Rubric Rewards for Agentic Forecasting

**arXiv ID:** 2611.27286 | [PDF](https://arxiv.org/pdf/2611.27286v1)

**作者:** Jingyi Chai `[一作]` (Shanghai Jiao Tong University), Siheng Chen `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为DeepForecaster的35B参数实时预测模型，利用主动信念状态管理（PBM）和无结果依赖的五维评估准则（SEIPH）实现高效的事件预测。

**💡 创新点**

创新点在于引入PBM折叠工具主动压缩上下文并维护结构化信念状态，以及设计SEIPH评估准则为强化学习提供细粒度、无延迟奖励，突破传统ReAct模型在长链搜索中的上下文溢出和奖励稀疏问题。

**🔧 技术方法**

技术手段包括基于ReAct的工具调用、PBM折叠与上下文重置、对SFT得到的PBM轨迹进行PPO/GRPO强化学习，并在此基础上实现多段奖励与参数更新。

**📊 数据集**

实验使用了三大实时预测基准FutureX、ForecastBench和Polymarket‑Sim，以及人类市场数据，覆盖数百个已解决事件。

**📈 对比分析**

与1.6T DeepSeek‑V4‑Pro、1T Kimi‑K2.6、GPT‑5.5等大型模型相比，DeepForecaster‑PPO在Brier分数上分别取得0.151（ForecastBench）和0.147（Polymarket‑Sim），显著优于更大模型，并在FutureX的L1/L2得分排名第二，显示出更优的预测性能。

**⚠️ 局限性**

局限性包括仍无法达到人类预测者的水平，且模型在极端置信区间可能出现过度自信；此外SEIPH评估准则需要人工标注，规模化推广时的可扩展性和通用性有待进一步研究。

---

## 175. OpenRing: Deadlock-Free Execution of Any Planner on AGV Floors with One-Way Lanes

**arXiv ID:** 2611.27127 | [PDF](https://arxiv.org/pdf/2611.27127v1)

**作者:** Zhihao Lin `[一作]` (University of Osaka), Yoshinobu Kawahara `[通讯]` (University of Osaka)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 OpenRing 框架，将规划器的路线与通过局部安全守卫的执行分离，保证在任意规划器或规划器失效的情况下，AGV 轨迹不会形成闭环导致死锁；同时提供了一个订单绑定时间上限，使得订单从建议变为建议而非约束；进一步证明了在任何规划器下安全、进展与清除的可达性，并给出了安全保证的吞吐量代价上限；

**💡 创新点**

创新点在于：①将安全性与效率完全分离，允许规划器自由提出路线；②使用单链一跳前瞻守卫（Look）与路线检查（Ψ）在 FIFO 段上保证无闭环；③引入订单绑定时间 T_f 使得订单仅为临时建议，避免因规划器失效导致的死锁；④给出在任意停止扰动下每段的最大流量上限 B/T_s，并证明此上限紧致；

**🔧 技术方法**

主要技术包括：FIFO 段模型与跨越检查；单链一跳前瞻 Look；路线检查 Ψ；订单绑定与失效后过期；基于 ledger 的计数状态；以及在物理仿真（Isaac Sim）与离散网格模拟中实现的 AGV 运动控制与安全约束（C1–C5）；

**📊 数据集**

使用自建的 AGV 轨迹仿真数据集：8 种不同布局的地面（3×3、4×4、6×6 网格，3×6 不规则，4×4 仓库，hall、fab loop、lattice 等），每种布局在 0.6–1.6 负载下进行 10+ 种扰动组合；在 Isaac Sim 进行物理仿真；以及理想的细胞模型用于理论验证；

**📈 对比分析**

与传统方法（first‑come、PIBT、LaCAM、RHCR、SADG、PBS 等）进行对比，指标包括吞吐量（AGV/min）、死锁次数、平均延迟、路线遵循率等。实验表明：OpenRing 在 114 次跑中 0 次死锁；吞吐量与最优基线相当或更高，且在高扰动下仍保持 10–20% 的吞吐提升；平均延迟最低，且大多数方法在高负载下出现死锁。

**⚠️ 局限性**

局限性包括：需要每段至少可容纳两辆 AGV（B≥2）才能保证安全；安全守卫和订单绑定需要额外的协调器和周期性检查，增加实现复杂度；吞吐量上限 B/T_s 可能导致在高负载下吞吐受限；对非常大规模系统（数千辆 AGV）或异步/分布式守卫的支持尚未验证。

---

## 176. Loop-OPF: Solver-Informed Learning for Optimal Power Flow with Inference-Time Topology Adaptivity

**arXiv ID:** 2611.27013 | [PDF](https://arxiv.org/pdf/2611.27013v1)

**作者:** Mohamad Chehade `[一作]` (University of Texas at Austin), Hao Zhu `[通讯]` (University of Texas at Austin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种基于求解器的循环神经网络Loop-OPF，用于在保证最优性和可行性的前提下快速求解DC-OPF问题并在推理时实现拓扑自适应。

**💡 创新点**

（1）将PDHG迭代嵌入网络层，保证固定点即为最优解；（2）使用DEQ与隐式微分实现自监督训练，显著降低内存；（3）将拓扑信息作为层输入，实现一次训练后可适用于未见的线路/发电机故障；（4）学习PDHG步长参数以加速收敛。

**🔧 技术方法**

Primal–Dual Hybrid Gradient（PDHG）求解器、深度平衡（DEQ）框架、隐式微分、自监督损失函数、Picard+Anderson加速的固定点求解。

**📊 数据集**

IEEE 300节点、PEGASE 1354、2869、9241节点电网；每个网络10,000条样本（7,000训练、1,000验证、2,000测试），包含无故障、单线故障和单发电机故障场景。

**📈 对比分析**

与DeepOPF、DC3、DeepLDE、PDL、GNN及其拓扑条件变体对比；在无故障测试中，Loop-OPF的RMSE均低于0.23%，流线违规率≤1.6%，成本差距≤0.01%；在未见故障和SC-OPF测试中，Loop-OPF唯一实现零违规且成本差距≈6×10⁻⁷%；训练后迭代次数约减半，推理预算可任意截断，误差随迭代线性下降。

**⚠️ 局限性**

仅针对凸DC-OPF；在高度拥塞网络中训练收益有限，初始参数收敛慢；对非凸AC-OPF和多约束安全情形的推广仍需验证；需进一步研究多场景步长学习与分布式实现。

---

## 177. Causal Diffusion for Extreme Super-Resolution in Interactive Games

**arXiv ID:** 2611.26963 | [PDF](https://arxiv.org/pdf/2611.26963v1)

**作者:** Christina Zhang `[一作]` (New York University), Qi Sun `[通讯]` (New York University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种针对极端游戏超分辨率的因果扩散框架，可将120p或更低分辨率提升至960p（8×）甚至16×，并通过G-buffer引导生成高频细节；

**💡 创新点**

创新点在于：①将预训练的扩散模型改造为因果、单帧生成，避免依赖未来帧；②利用深度与运动向量实现几何与运动约束；③采用轻量化潜在投影器、单帧解码器及学生蒸馏，显著降低推理时延；

**🔧 技术方法**

技术包括：视频扩散模型（DOVE）改造、因果自注意力与KV缓存、局部空间注意、G-buffer条件编码、轻量化潜在投影、Tiny AutoEncoder解码、学生蒸馏与块剪枝；

**📊 数据集**

使用了336k帧的自制数据集，采自Cyberpunk 2077与Black Myth：Wukong两款AAA游戏，包含高分辨率渲染及对应G-buffer（深度与运动向量）；

**📈 对比分析**

与DLSS（级联2×）、FlashVSR、Stream‑DiffVSR、RDG、InfVSR等基线进行对比，本文在8×超分辨率下取得最高PSNR（31.64 dB）、最佳LPIPS（↓49%）并实现约13 FPS（75 ms/帧）；

**⚠️ 局限性**

局限性包括：依赖游戏特定的LoRA适配，导致跨游戏泛化受限；推理速度仍慢于传统非生成式上采样；在极低分辨率下可能产生少量误差或不一致性；

---

## 178. Markdown Is Not a Firewall: four coding agents, one rules file, four boundaries

**arXiv ID:** 2611.27112 | [PDF](https://arxiv.org/pdf/2611.27112v1)

**作者:** Sandeep Dhuri `[一作]` `[通讯]` (Independent Researcher), Sandeep Dhuri (Independent Researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究评估了在仓库根目录放置的规则文件（Rules File）对多轮编程代理（Claude、Codex、Gemini、OpenCode）的影响，探讨其是否能作为控制措施或仅仅是建议；

**💡 创新点**

创新点在于首次对规则文件在安全边界（gate）上的作用进行预注册实验，并结合隐藏测试、检查器、门控（gate）与证据链审计等多维度指标，系统性评估规则文件对代理行为的实质性影响。

**🔧 技术方法**

技术包括：多轮代理执行框架、隐藏测试集、自动化检查器（自定义 checker、Bandit、Semgrep）、门控（gate）机制、证据链审计、盲判定与多轮人工核对、Bootstrap 置信区间。

**📊 数据集**

使用了20个小型 Flask/SQLite 仓库（包含 57–123 行代码）以及为每个仓库设计的 5–12 条隐藏测试；实验共覆盖四个代理、三种文件配置（无文件、规则文件、中性文件）以及四个安全边界。

**📈 对比分析**

比较方法为配对对比（按仓库对比）、聚合 Bootstrap 置信区间及非劣效性检验；结果显示规则文件未发现任何真实缺陷；在隐藏测试通过率方面，Codex 代理在规则文件下从 99.0% 降至 67.7%（平均下降 7.7 分）；其他代理下降 <2.5 分；在安全边界的越界尝试极少，门控几乎完全阻断。

**⚠️ 局限性**

局限性包括：门控配置错误导致无门控测量缺失；安全边界实验样本量不足（每边界 15 次，未达 20 次阈值）；仅测试了 20 个极小仓库，无法推广到大规模或长周期任务；仅使用了四个代理，模型及 API 调用方式差异大；检查器与规则文件存在潜在循环依赖。

---

## 179. MSRyzen: An Agentic Platform for Unified Proteomics Analysis and Evidence-Centric Visualization

**arXiv ID:** 2611.27246 | [PDF](https://arxiv.org/pdf/2611.27246v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 180. MAPES: Multi-Value Alignment Pipeline via Expert Specialization

**arXiv ID:** 2611.27133 | [PDF](https://arxiv.org/pdf/2611.27133v1)

**作者:** SungHo Kim `[一作]` (Korea University), SangKeun Lee `[通讯]` (Korea University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为Mapes的多值对齐管线，通过冲突感知的数据划分、专家训练、偏好感知合并及后期和谐化，实现了多人类价值的同时优化。

**💡 创新点**

将跨价值冲突从单一目标中解耦，在管线各阶段主动处理数据、梯度、参数和分布冲突，并提出首个基于DPO梯度方向的Preference-Aware Value Merging（PAVM）和轻量化和谐化步骤。

**🔧 技术方法**

利用LoRA轻量级适配器、gap-weighted DPO、基于梯度方向的合并权重、KL蒸馏以及多任务分布式训练。

**📊 数据集**

使用经过细粒度价值标注的UltraFeedback对比数据集（约36k对）。

**📈 对比分析**

与标准DPO、CPO、RA-DPO、MODPO及RS等基线在五个偏好对齐基准（Arena-Hard、IFEval、HHH-Alignment、HH-RLHF、TruthfulQA）以及GSM8K/GPQA进行对比，Mapes在平均得分上均优于所有基线，提升约3-7分且在大模型上仍保持最佳。

**⚠️ 局限性**

依赖细粒度价值标注的偏好数据、仅验证到8B规模且未验证对更大模型或不同对齐目标的通用性、且对价值控制的推理时灵活性有限。

---

## 181. ORION: Exact Canonical-State Supervision for Systematic Generalization in Transformers

**arXiv ID:** 2611.27087 | [PDF](https://arxiv.org/pdf/2611.27087v1)

**作者:** Sibi Gokul `[一作]` (National Institute of Technology Warangal), Jyoti Prakash Meher `[通讯]` (National Institute of Technology Warangal)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出 ORION 框架，在自回归 Transformer 中通过监督每一步的准确定义状态，防止 latent 漂移。

**💡 创新点**

创新点在于将长度外推和深度外推视作两独立失败轴，并证明 RoPE 与 ORION 互补，可通过组合实现更强的 OOD 推理。

**🔧 技术方法**

采用 RoPE 位置编码、辅助投影头与正则化损失、2×2 因子实验以及多模型规模的训练。

**📊 数据集**

在 Folding、GridShift 两个可解析的算法域和 CLRS 试点数据集上进行评估。

**📈 对比分析**

与基线和单一干预相比，ORION+RoPE 在深度外推上提升约 66.7%，并在联合 OOD 试验中实现显著的增益，优于任何单独方法。

**⚠️ 局限性**

局限包括无法解决状态尺寸外推、需要可用的精确状态标注、额外计算成本以及仅在合成算法任务上的验证。

---

## 182. EgoSurgHands: An Egocentric 3D Hand Pose Dataset & Benchmark for Open Surgery Training

**arXiv ID:** 2611.27101 | [PDF](https://arxiv.org/pdf/2611.27101v1)

**作者:** Ankit Pal `[一作]` (Harvard Medical School), Gabriel A. Brat `[通讯]` (Harvard Medical School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并公开了 EgoSurgHands 数据集与基准，用于评估开放式手术训练中的第一人称 3D 手姿估计。

**💡 创新点**

创新点在于：①首次提供真实摄像的 egocentric 手部 3D 标注，采用多传感器融合并人工验证；②提供一套零样本评估协议和残差校正基线 JRN，帮助研究者快速定位模型误差。

**🔧 技术方法**

采用 Meta Aria 多传感器眼镜收集数据，利用 Aria Machine Perception Services (MPS) 生成初步 3D 标注，再通过人工复核生成伪真值；实验中对八种手姿网络（MANO、IntagHand、MediaPipe 等）进行零样本推理，并在其上加 JRN 进行残差纠正。

**📊 数据集**

使用的主要数据集是 EgoSurgHands，包含 55 条训练记录、13 条验证记录，共计 41,909 条训练手姿实例和 9,476 条人工验证的评估实例，涵盖 13 种手术任务、3 种手套颜色、4 名佩戴者和 3 个经验层级。

**📈 对比分析**

对八个基线模型在零样本条件下进行评估，结果显示 egocentric 预训练模型 JointTransformer（PA‑MPJPE 14.95 mm）和 WildHands（20.48 mm）显著优于非 egocentric 模型；加入 JRN 后误差进一步下降至 10.70 mm（JointTransformer）和 10.40 mm（WildHands），与标签解析度（≈1 cm）相当。

**⚠️ 局限性**

局限性包括：仅包含 4 名受试者且在合成皮肤垫上完成，标签为伪真值且精度约 1 cm；佩戴者与手套颜色在验证集存在混淆，导致难以单独评估两因素；缺乏真实手术室和多中心数据，需进一步扩展。

---

## 183. AnyBodyCodec: A Shared Native Interface for Body-Model Conversion, Extension, and Humanoid Control

**arXiv ID:** 2611.27397 | [PDF](https://arxiv.org/pdf/2611.27397v1)

**作者:** Qianxing Li `[一作]` `[通讯]` (Delta Intelligence), Qianxing Li (Delta Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出 AnyBodyCodec，构建一个共享的本地体型表示，能够在 MHR、SMPL‑X、SOMA 等多种参数化体模型之间进行无缝转换，并在冻结该表示后直接用于机器人控制。

**💡 创新点**

创新点在于：① 利用模型特定的编码器/解码器学习对齐的槽位（slots）作为共享表征；② 通过冻结该表征，支持在不重新训练策略的情况下扩展新的体型接口；③ 在同一冻结表征上训练共享政策，实现跨格式的单一控制器。

**🔧 技术方法**

技术手段包括：两层 Transformer + LayerNorm 生成 8×64 维槽位；Kabsch 归一化恢复世界坐标；对齐的自注意力查询解码器；PPO（SONIC 风格）用于训练机器人控制网络；以及 ANNY 扩展在冻结主体上学习新接口。

**📊 数据集**

使用的数据集为 BONES‑SEED（313万帧，1930 动作，179 人体），包含 MHR、SMPL‑X、SOMA 以及自定义 ANNY 的配对控制；验证集 9,009 帧分为阻塞、全新动作、全新身份三类，确保跨分布评估。

**📈 对比分析**

评估方式：输入参考 MPJPE 作为几何一致性指标，行动一致性（动作偏差）作为控制兼容性指标；实验显示 1.09M 参数的共享转换器在 16.80 mm MPJPE 下，与 2.887M 的 Transformer bank 仅相差 0.4 mm；在冻结策略下，跨格式控制成功率达到 60.78 %（MHR 62.68 %，SMPL‑X 60.11 %，SOMA 59.54 %），比基准 57.45 % 提升约 3 个百分点。

**⚠️ 局限性**

局限性包括：① 只验证了三种主流体模型及其自定义接口，可能对其他模型的迁移性不一定保证；② 对齐的槽位并非物理关节，可能在极端姿态或细节运动上失去部分细节；③ 评价主要基于 MPJPE 与行动一致性，未覆盖实际任务成功率或实时运动学约束。

---

## 184. Suspected Until Inspected: Neuro-Symbolic Detection of Broken Access Control in Server-Side Web Applications

**arXiv ID:** 2611.27270 | [PDF](https://arxiv.org/pdf/2611.27270v1)

**作者:** Chengyao Yang `[一作]` (Beijing University of Posts and Telecommunications), Shihong Zou `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计实现了 FLINT，一个神经符号框架，用于检测服务器端 Web 应用的访问控制缺陷。

**💡 创新点**

创新点在于将大型语言模型的授权提议与符号验证分离，仅当符号验证确认检查满足所有路径时才认为访问受保护，从而在多语言、多框架环境下实现零日 BAC 检测。

**🔧 技术方法**

采用大型语言模型生成授权证书、需求驱动的代码切片、根源追踪、基于格的跨过程数据流分析以及声明式检查验证等技术。

**📊 数据集**

使用 57 个流行开源应用（如 Gitea、Grafana、Strapi 等）、69 对脆弱-修复代码、41 个自发发现的零日、227 个邻居端点以及 5 个完整应用的代码库进行评估。

**📈 对比分析**

与静态分析、BOLA 检测器和 LLM 基线相比，FLINT 在零日检测率约 85% 以上、误报率显著降低；平均每端点模型耗时约 36 秒，验证耗时 0.1 秒，成本约 0.006 美元。

**⚠️ 局限性**

局限性包括仅评估证书列出的访问，忽略未提及的访问；依赖可信的 LLM 生成证书；对未覆盖的代码路径或外部库缺乏完整性；以及对复杂权限命名和细粒度策略的假设。

---

## 185. Online Learning for Minimizing Maximum Tardiness in Single-Machine Scheduling

**arXiv ID:** 2611.27067 | [PDF](https://arxiv.org/pdf/2611.27067v1)

**作者:** Yiyang Chen `[一作]` (Kyushu University), Eiji Takimoto `[通讯]` (Kyushu University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并分析了单机调度中最大迟滞最小化的在线学习框架（OMMT），研究了随机与对抗性环境下的学习策略；

**💡 创新点**

创新点在于：1) 首次将最大迟滞最小化问题转化为在线学习问题；2) 在随机环境下给出 FTL 算法的期望与高概率 1‑regret 上界；3) 在对抗环境下证明 α‑regret 需满足 α=Ω(log n) 的下界；4) 通过 OLVC 框架与 FTPL 设计出 α=O(log n) 的上界，展示对数依赖的紧密性；5) 提供可多项式求解的 MIP 形式用于 FTPL 更新；6) 给出 FTL–BTL 定理在 ℓ∞ 目标下失效的反例。

**🔧 技术方法**

主要技术包括：
- 逆向 LCL（Extended LCL）实现 FTL 更新；
- 统计学习理论的集中与大数定律用于随机环境的上界；
- 对抗性下的分阶段构造与递归活跃集论证下界；
- OLVC（在线向量成本）框架的平滑最大映射与代理损失构造；
- Follow‑the‑Perturbed‑Leader（FTPL）算法与指数扰动；
- 对 FTPL 的 MIP 形式化与 Lipschitz 分析。

**📊 数据集**

本文未使用真实数据集，全部结果均为理论分析与上界/下界证明；

**📈 对比分析**

与已有工作（如在线最小化总完成时间、优先级约束调度、在线负载平衡等）相比，本文针对最大迟滞给出了更精确的 α‑regret 上界与下界；在随机环境下，1‑regret O(n log n/Δ²) 与 √(T n log n) 复合项；在对抗环境下，α=O(log n) 时的期望 α‑regret O(n²√T) 超越了此前仅针对总完成时间的 O(n²√T) 上界。

**⚠️ 局限性**

局限性包括：1) 对抗性下仍需 α=Ω(log n)，无法实现更小的 α；2) FTPL 的 MIP 求解在实际规模下可能过慢；3) 对多最优排列情况的 √T 依赖尚未证明最优；4) 仅提供理论上界，缺乏实验验证；5) 对其他目标函数（如最大完成时间）未给出类似分析。

---

## 186. VMSA: Video Memory Skill Agent for Long Video Understanding

**arXiv ID:** 2611.27381 | [PDF](https://arxiv.org/pdf/2611.27381v1)

**作者:** Changin Choi `[一作]` (IPAI), Wonjong Rhee `[通讯]` (IPAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Video Memory Skill Agent (VMSA)，一种联合设计结构化视频记忆和记忆技能的长视频问答框架。

**💡 创新点**

创新点在于将分段字幕、实体事件和实体感知剧集三种互补记忆与从成功轨迹中诱导的可迁移记忆技能相结合，实现对长视频的高效检索、推理与验证。

**🔧 技术方法**

采用检索增强的工具调用、可解释的内存技能诱导、LLM 驱动的代理推理和多层次结构化记忆构造技术。

**📊 数据集**

使用七个长视频问答基准，包括三组七天的 egocentric 视频（EgoLifeQA、Ego-R1 Bench、EgoMemReason）和四组长视频（Video-MME、LongVideoBench、LVBench、HippoVlog）。

**📈 对比分析**

与多类别基线（LLM、长视频 LLM、检索增强、代理框架）比较，VMSA 在所有三组 week‑long egocentric 任务中取得最高准确率（最高提升 9.4%），在其他四个基准上亦保持领先或竞争水平。

**⚠️ 局限性**

局限在于需要较大计算资源来构建多层记忆、依赖强大 LLM 作为代理、记忆技能库规模可能随任务扩展而增大，以及对非 egocentric 长视频的迁移能力仍待进一步验证。

---

## 187. How Does SAM-3 Adapt? Capacity, Update Geometry, and Domain Transfer

**arXiv ID:** 2611.27142 | [PDF](https://arxiv.org/pdf/2611.27142v1)

**作者:** Allan Kazakov `[一作]` (Bahcesehir University), Duygu Cakir `[通讯]` (Bahcesehir University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统评估 SAM-3 在五个医学、遥感及材质分割数据集上，比较 LoRA、HiRA、KRAdapter、QuanTA 与全量微调的适配效果。

**💡 创新点**

发现 SAM-3 适配往往容量饱和，PEFT 能匹敌或超过全量微调；通过谱分析和域迁移实验揭示不同方法的更新路径差异，单一低秩近似不足以解释其性能。

**🔧 技术方法**

使用 LoRA、HiRA、KRAdapter、QuanTA 这类参数高效微调技术、全量微调、SVD 谱分析、LoveDA Urban/Rural 迁移实验。

**📊 数据集**

采用 MRI-Scler、CT-Colon、CT-Vessel、LoveDA、Apple Dense Material Segmentation 五个数据集。

**📈 对比分析**

按 Dice/IoU 评估，PEFT 在四个数据集上往往与全量微调相近甚至更好，LoRA 在多数数据集表现最佳；全量微调在域迁移中平均 mIoU 较高。

**⚠️ 局限性**

局限在于仅使用验证集迁移实验、仅分析可访问注意力投影的更新、未做全局超参数搜索、部分数据集缺失测试标签等。

---

## 188. Subverting Tool Arguments: Defense-Evading Metadata Poisoning in LLM Agents

**arXiv ID:** 2611.27041 | [PDF](https://arxiv.org/pdf/2611.27041v1)

**作者:** Shih-Chieh Dai `[一作]` (University of Utah), Guanhong Tao `[通讯]` (University of Utah)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对LLM代理的工具使用流程，提出一种利用工具描述中毒的攻击方法，使代理误选恶意工具并生成攻击者指定的参数值。

**💡 创新点**

创新点在于结合依存句法分析定位与Token相关性，对工具描述进行局部掩码并让LLM生成动态变异指令，迭代优化后得到简洁自然、易于隐蔽的攻击文本，而无需检测器的反馈。

**🔧 技术方法**

技术手段包括：依存句法树定位关键Token、Token相关性打分与掩码、LLM驱动的变异指令生成与迭代优化、代理模拟的代理查询集与多轮反馈机制。

**📊 数据集**

使用了三大公开MCP安全基准数据集：MSB、MCP‑SB 与 MCPTox，涵盖109个任务、118个工具，共计327条测试查询。

**📈 对比分析**

与模板攻击、优化攻击（AttackMove、MCP‑ITP）及RL攻击（PISmith、CTH）以及五种过滤器、运行时防御和堆叠防御对比，攻击成功率(sASR)在MSB、MCP‑SB、MCPTox分别达到约93.8%、81.5%与63.8%，检测率低于5%，在全部防御叠加下仍保持≈81%成功率。

**⚠️ 局限性**

局限性包括：依赖LLM模型与其生成能力；对极为细粒度或自适应训练的检测器仍可能被击穿；工具描述长度限制仍存在影响；在高对齐训练后可能降低工具使用的实用性。

---

## 189. CTRL: Concurrent Training and Rollout for LLM Reinforcement Learning via GPU Co-Residency

**arXiv ID:** 2611.27119 | [PDF](https://arxiv.org/pdf/2611.27119v1)

**作者:** Ruijia Yang `[一作]` (Hong Kong University of Science and Technology), Zeyi Wen `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出在大语言模型RL后训练中，将每个GPU同时运行一个学习器（learner）和一个持久化推理引擎（rollout shard），通过主机拥有的模型状态和有限GPU窗口实现并行执行，称为co‑resident布局；

**💡 创新点**

创新点在于：①将RL策略生命周期围绕主机可变版本组织，使学习器只需有限GPU窗口即可完成评分、训练和策略交付；②利用CUDA MPS实现学习器与推理引擎在同一GPU共享SM；③设计发布门（publication gate）将学习器更新与推理推送解耦，实现无丢弃的版本上限；④基于步骤时间上限的资源平衡模型和短测量搜索，自动调节学习器计算份额与推理布局；

**🔧 技术方法**

技术方法包括CUDA Multi‑Process Service (MPS)进行同机共享、层流式（layer‑streaming）训练状态迁移、主机拥有的BF16版本环（ring）管理版本、发布门调度、步骤时间/学习器耗时/推理跨度的度量与优化、以及使用vLLM/verl实现推理与训练；

**📊 数据集**

实验使用Qwen3系列模型（Base、Thinking、Non‑Thinking）在两大RL奖励环境：DAPO‑Math（可验证数学）和MBPP（单元测试）进行训练与评估；

**📈 对比分析**

对比方法为同步/异步时间共享、全GPU分割等传统布局，实验在H100、A800、RTX 4090三种硬件上实现，co‑resident布局在大多数工作负载下实现1.5‑3.5倍吞吐量提升（最高可达≈3×），并在八GPU规模上实现≈1.1×速率提升；

**⚠️ 局限性**

局限性包括：①需要大量主机内存和CPU‑GPU带宽来容纳训练状态；②对MPS支持的GPU硬件与驱动版本有依赖；③在极大模型或极低延迟场景下，主机转移与版本管理开销可能成为瓶颈；④对网络互连（尤其非NVLink）要求较高，需根据平台选择合适的并行布局。

---

## 190. Separating Automation Gains from Agent Contributions in AI-Assisted Model Validation

**arXiv ID:** 2611.27094 | [PDF](https://arxiv.org/pdf/2611.27094v1)

**作者:** M. Ahmadi Achachlouei `[一作]` `[通讯]`, M. Ahmadi Achachlouei

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

在AI辅助模型验证中提出分阶段架构，保持LLM自由工作流程为对照，并设计归因协议只给LLM任务已未报告的错误计分。

**💡 创新点**

创新点是将LLM自由流程作为可落后对照，并通过归因协议区分LLM对未报告错误的检测、自动化发现的折返及误报消除，实现对LLM贡献的可验证归因。

**🔧 技术方法**

使用的技术包括基于LLM的代理任务、可追溯的证据记录、版本控制哈希、工具调用约束、两阶段归因协议、合成信用评分卡实验包以及验证器对输出的结构化评分。

**📊 数据集**

使用的数据集是合成的信用违约概率评分卡六种变体（v0.7包），包含故意注入的实现、测试覆盖、结果分析和特征研究错误。

**📈 对比分析**

比较方法是将LLM代理任务与无LLM对照流程冻结比较，记录原始代理提议并计量未报告错误的检测以及对已报告错误的努力；实验结果显示传统指标无法区分不同策略，而归因协议能够分别识别理想策略的额外检测，其他策略表现为误报、过度警报或错误认同。

**⚠️ 局限性**

局限性包括：未评估实际LLM运行时性能，归因协议仅在合成脚本政策下验证；缺少独立审计、真实银行案例、验证器一致性；实验包未包含真实数据；实验对经济效益的门槛尚未验证。

---

## 191. Suffused NV-FETs for Reconfigurable Hardware

**arXiv ID:** 2611.26771 | [PDF](https://arxiv.org/pdf/2611.26771v1)

**作者:** Alexander J. Edwards `[一作]` (Laboratory for Physical Sciences), Adam L. Friedman `[通讯]` (Laboratory for Physical Sciences)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出并实现了新型非易失性光照充饱(FET)作为可重构硬件的配置存储单元，显著降低了器件数量。

**💡 创新点**

创新点在于将光/电子束充饱降低写入能量壁垒，使FET可在不需要高压写入的情况下实现闭合/开放开关及N/P FET功能，并通过局部充饱实现极简编程电路。

**🔧 技术方法**

采用光/电子束充饱技术、二维材料(FLG/hBN/TMD)结构、四端局部充饱的Suffused NV-FET以及相应的编程电路设计。

**📊 数据集**

未使用传统数据集，实验通过电路仿真和硬件原型验证。

**📈 对比分析**

通过与CMOS、FG-FET、FE-FET SWITCH及SWITCH+设备对比，局部充饱四端Suffused NV-FET将TRAP单元晶体管数从1194降至185，下降幅度达84.5%，且保持可编程窗口和存储稳定性。

**⚠️ 局限性**

局限在于需要外部光/电子束源实现充饱、写入速率受限、对器件寿命和误写率的进一步评估仍需实验验证。

---

## 192. Do Language Models Know Which Words Steer Their Reasoning?

**arXiv ID:** 2611.27317 | [PDF](https://arxiv.org/pdf/2611.27317v1)

**作者:** Yanbei Jiang `[一作]` (University of Melbourne), Rena Gao Wei `[通讯]` (University of Melbourne)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了链式推理（CoT）过程中词级别的非语义上下文影响（PCI），并提出一种在线KV抑制方法，利用自评判识别并抑制模型认为无关却实则影响推理的词，从而提升推理准确率。

**💡 创新点**

创新点包括：①首次系统量化PCI并构建留一词JSD+自评判诊断框架；②基于KV缓存的单词级抑制机制；③将检测与抑制结合，形成可在线执行的推理改进流程。

**🔧 技术方法**

技术手段涵盖：留一词JSD（词级和响应级）、LLM自评判（Binary判断）、KV缓存零化（抑制键值向量）、单词级与响应级PCI检测与排序。

**📊 数据集**

使用的数据集有：GSM8K、GSM‑Symbolic、TruthfulQA、MMLU（5个子任务）以及AQUA‑RAT，共计5个基准。

**📈 对比分析**

在四个模型（Qwen‑3.5‑4B/9B、Gemma‑3‑12B/27B）上，在线KV抑制平均提升3–8.9个百分点，最高单项提升近9个百分点，且在大多数模型-基准组合中均优于零射击、注意力抑制及停用词抑制等基线。

**⚠️ 局限性**

局限性：仅适用于可白盒访问KV缓存的模型；只抑制单个词，无法同时消除多重干扰；自评判为二元判断，可能无法细粒度区分相关性；未在封闭模型或多重干扰情境下验证。

---

## 193. Trained on Harm, Still Refusing: A Training-Time Defense for Code Generation

**arXiv ID:** 2611.27495 | [PDF](https://arxiv.org/pdf/2611.27495v1)

**作者:** Miaoqing Tian `[一作]` (Qingdao University of Technology), Haipeng Qu `[通讯]` (Ocean University of China)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了代码生成模型在接收有害微调数据后安全失效的机制，并提出了SAFEHOLD训练时防御框架，能在有害微调过程中保持安全对齐。

**💡 创新点**

创新点在于将拒绝行为视为持续的生成过程，结合决策边界锚定（保持初始响应分布）和拒绝路径强化（监督完整拒绝序列），并通过供应商控制的安全探针实现无额外推理阶段的防御。

**🔧 技术方法**

使用的技术包括LoRA微调、对齐初始响应分布的决策边界锚定、完整拒绝序列的强化学习、logit‑lens层级可视化、对齐安全神经元激活监测，以及梯度合并实现联合优化。

**📊 数据集**

实验使用8个固定安全探针（AdvBench）、三组有害训练集（LLM‑LAT、CyberStrike、CySecBench）以及评估基准RMCBench、HarmBench_cy、MalwareBench、XSTest和HumanEval。

**📈 对比分析**

与原始模型、无防御微调以及五种SOTA防御（Vaccine、LiSA、SBR、SafeMERGE、BAR）比较，SAFEHOLD平均攻击成功率（ASR）下降72.03pp，优于SOTA 24.32–70.87pp；同时HumanEval pass@1提升10.24pp，XSTest过度拒绝率仅2.73%。

**⚠️ 局限性**

局限性包括仅在公开的代码LLM上验证，未测试更大或封闭模型；依赖固定探针和共享拒绝文本，可能对不同安全场景的适用性有限；在极端有害训练集下的鲁棒性仍需进一步验证。

---

## 194. Bottlenecked: The Capital Cost of Protected DeFi Lending

**arXiv ID:** 2611.26948 | [PDF](https://arxiv.org/pdf/2611.26948v1)

**作者:** Tarun Chitra `[一作]` (Gauntlet), Nagu Thogiti `[通讯]` (BreakwaterAI)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究如何在去中心化金融中通过图结构分配初级资本和工作资本，以在攻击时先吸收损失并提高资本效率。

**💡 创新点**

创新点在于提出了对三分图（资产-负债-初级资本）切割的理论框架，定义并计算了对抗违约溢价（ADP），并证明在适当设计下资本成本可从线性降低到Θ(√N)，同时给出显式构造（投影平面、Ramanujan图）。

**🔧 技术方法**

采用图论（切割、最小割、奇异值分解）、凸优化、以及定理证明等数学工具，并结合静态安全博弈模型来评估保险成本。

**📊 数据集**

使用2026年多起DeFi攻击（如rsETH桥接事件）以及Morpho协议的历史债务/供应数据进行实验验证。

**📈 对比分析**

与“隔离匹配”和“完全访问”两种基准对比，实验显示在理论最优图下可节省约34.5%的资本成本，且在大多数历史案例中实现了15%以内的节省。

**⚠️ 局限性**

局限性包括：实验基于离线静态数据，未考虑在线动态行为；对抗违约溢价的估计依赖于预设价格假设；在实际部署中可能因权限配置、协议升级等因素导致理论收益无法完全实现。

---

## 195. GGDriver: Glance at What Is Shared, Gaze at What Differs for Perception-Free Planning

**arXiv ID:** 2611.27330 | [PDF](https://arxiv.org/pdf/2611.27330v1)

**作者:** Jiawei Sun `[一作]`, Michael Bi Mi `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于候选轨迹的路径特定视觉证据获取（gaze）机制，改进无感知端到端驾驶规划。

**💡 创新点**

创新点在于利用轨迹投影自动确定关注区域，完全无监督且不需要额外标签或学习选择网络。

**🔧 技术方法**

采用DINOv2 ViT‑S backbone、patch memory、轨迹投影、跨注意力的解码器，并在解码器与评分器两阶段使用gaze。

**📊 数据集**

使用公开的WOD‑E2E、NAVSIM‑v1/v2以及Bench2Drive仿真数据集进行训练与评估。

**📈 对比分析**

与DrivoR、DiffusionDrive、LTF等多种基线相比，RFS、PDMS、DS等关键指标均达到了最优或近最优水平。

**⚠️ 局限性**

局限性包括只能关注轨迹路径上的信息，依赖精确相机标定，闭环评估仅在仿真环境完成。

---

## 196. A Novel Dataset and Framework for Schenkerian Optical Music Recognition

**arXiv ID:** 2611.26958 | [PDF](https://arxiv.org/pdf/2611.26958v1)

**作者:** Yixuan Huang `[一作]` (Duke University), Stephen Ni-Hahn `[通讯]` (Chinese University of Hong Kong)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并使用首个手写 Schenkerian 分析数据集，开发端到端的 OCR 到排版框架，完成从符号识别到语义图构建再到类型化输出的完整流程。

**💡 创新点**

①基于 YOLO‑DETR 的多尺度检测集成，②针对符号关系的拓扑神经网络，③使用梯度提升分类器预测时值组和先后链，④设计语义图到 JSON 的转换算法；并首次公开 47 页手写 Schenkerian 数据集。

**🔧 技术方法**

YOLO‑DETR、Topological Neural Network、Gradient‑Boosting、MuNG Graph、MuNG Studio、MuSViT、Faster R‑CNN、GraphSAGE、以及自定义语义转换算法。

**📊 数据集**

新建的 `schA-OMR` 数据集（47 页、30,780 个符号、128 类），包含 12 作品、3 位分析师手写记录，且采用 MuNG 图格式标注。

**📈 对比分析**

与 MUSCIMA++、DoReMi、Efficient Notation Assembly、completeOMR、GraphSAGE 等基线对比；在符号识别、语法连线、时值组与先后链任务上均达到或超过现有方法，最高微 F1 约 96.7%。

**⚠️ 局限性**

仍受限于数据量、符号种类分布不均、模型对手写变形的鲁棒性、以及对混合排版（图像+谱表）的处理尚未充分验证。

---

## 197. PACE-WM: Parallel World Modeling with Action History for Visual Planning

**arXiv ID:** 2611.27173 | [PDF](https://arxiv.org/pdf/2611.27173v1)

**作者:** Haishan Liu `[一作]` (Nanjing University), Kai Chen `[通讯]` (Zhongguancun Academy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种使用执行动作历史记忆和并行潜在预测的世界模型PACE-WM，用于视觉规划

**💡 创新点**

创新点在于将执行动作历史作为动态上下文引入并行Transformer预测，并设计参考锚定最小进展(RAMP)规划准则

**🔧 技术方法**

使用JEPA式编码器、LSTM动作记忆、因果Transformer并行预测以及CEM搜索

**📊 数据集**

在TwoRoom、Reacher、PushT、OG-Bench Cube四个视觉目标任务上进行实验，并构造四个对照基准

**📈 对比分析**

与LeWM、Fast-LeWM及无记忆/视觉历史版本比较，PACE-WM基准成功率95.83%，RAMP后98%；规划速度比LeWM快7-9倍，效率更优

**⚠️ 局限性**

局限性包括对超参数(λ、split点)敏感、仅在四个相对简单任务验证、LSTM记忆可能限制长期依赖捕获

---

## 198. Puzzles, People, and Publications: Measuring Shared Significance in Crosswords from 1942 to 2024

**arXiv ID:** 2611.26677 | [PDF](https://arxiv.org/pdf/2611.26677v1)

**作者:** Danny Benett `[一作]` (University of Vermont), Peter Sheridan Dodds `[通讯]` (University of Vermont)

**通讯引用:** 19796 | [OpenAlex ID](https://openalex.org/A5040821463)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

利用大规模可解释计算方法，对1942-2024年《纽约时报》约240万条交叉谜题提示/答案对进行定量分析，衡量跨时共享意义的变化、编辑与构造者的影响以及跨出版物的答案传播。

**💡 创新点**

首次将共享意义量化为词汇的正式约束与文化意义的时序演变，结合编辑创新率与allotaxonometry，揭示编辑风格对跨词谜文化的系统性影响，并用跨出版物答案流网络衡量信息扩散。

**🔧 技术方法**

使用统计词频与词性模型、归一化词分数、bigram 计数、allotaxonometry、rank‑turbulence divergence、热图计数、创新率计算等可解释的统计与网络分析技术。

**📊 数据集**

约240万条《纽约时报》交叉谜题提示/答案对（1942-2024），以及跨五大美国报纸（NYT, LA Times, Newsday, WSJ, USA Today）2000-2023年的答案使用记录。

**📈 对比分析**

与婴儿姓名数据作对照检验词汇正式约束；通过词频-排名图、bigram 直方图、allotaxonometer 以及创新率统计比较不同编辑期；结果显示编辑周期对共享意义的显著影响，创新率差异与编辑风格相符。

**⚠️ 局限性**

限制：只覆盖NYT及其编辑，未考察构造者个体差异；跨出版物分析忽略同一词汇可能的独立创新；模型假设答案为类型-令牌，不捕捉多重含义变化；对其他文化或非英语谜题缺乏普适性。

---

## 199. Cruella: A Unified Microscaling and Floating-Point Dot-Product Unit for LLM Inference on FPGAs

**arXiv ID:** 2611.27323 | [PDF](https://arxiv.org/pdf/2611.27323v1)

**作者:** Jinho Yang `[一作]` (HyperAccel), Joo-Young Kim `[通讯]` (HyperAccel)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了UDPU（统一点乘单元），支持MXFP4（4位微尺度）与BF16（16位浮点）两种运算模式在同一FPGA上并行执行；

**💡 创新点**

创新点在于通过每组预对齐（per‑group pre‑alignment）共享一次归约流水线，并采用预加器打包（pre‑adder packing）在单个DSP阵列上同时完成MXFP4×BF16与BF16×BF16乘法，显著降低DSP与LUT资源消耗；

**🔧 技术方法**

采用的技术包括：FPGA DSP48E2的宽度打包与符号校正、组内权重量化与共享尺度处理、块级对齐与归约流水线共享、以及在FPGA上实现的微尺度MX格式算子；

**📊 数据集**

在公开的LLM权重（如GPT‑OSS‑20B、Llama‑3.1‑8B、Qwen‑3‑8B）和标准数据集（WikiText‑2、ARC‑Challenge、MMLU）上评估；

**📈 对比分析**

通过与基准GPU（NVIDIA A40）和其他FPGA加速器比较，UDPU在MXFP4权重下可将单词推理时间缩短1.15×–1.87×，同时能耗比GPU低0.92×–2.01×，且在同等吞吐下比其它设计节省36%–72% LUT、15%–26% DSP；

**⚠️ 局限性**

局限性包括：仅在单批次推理且使用BF16 KV缓存的场景下验证，未覆盖MXFP8键值或批量并行服务；此外，组预对齐的误差在某些大模型中仍可引入≈0.7%困惑度增加。

---

## 200. Latent Space Planning with Sequential Monte Carlo

**arXiv ID:** 2611.26706 | [PDF](https://arxiv.org/pdf/2611.26706v1)

**作者:** Hrishikesh Viswanath `[一作]`, Aniket Bera `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种在潜在空间中使用序贯蒙特卡洛（SMC）与梯度优化相结合的规划框架，将完整动作序列作为粒子进行演化，以学习到的世界模型（DINO‑WM/LeWM）进行预测并寻找到达目标的行动序列。

**💡 创新点**

创新点包括：① 将完整动作序列视为粒子，在 SMC 与优化循环中不断更新目标分布，形成一种 EM 视角的规划方法；② 引入基于离线转移构建的图结构，通过 PDE（热方法）求解代价‑到‑目标，从而把潜在空间的连通性信息纳入代价；③ 通过 Jarzynski 校正的 Langevin 步与重采样实现目标分布的自适应演化。

**🔧 技术方法**

技术手段涵盖：学习世界模型（DINO‑WM、LeWM）、Sequential Monte Carlo（Jarzynski‑adjusted Langevin）、Adam/梯度下降、PDE heat 方法构建代价、系统重采样、重采样后粒子传播、粒子权重更新、梯度计算与优化步。

**📊 数据集**

实验使用公开离线机器人/仿真数据集，包括 Reacher、Point maze、Wall、Push‑T、Cube；数据通过 DINO‑WM 或 LeWM 编码成潜在空间进行规划。

**📈 对比分析**

在同等搜索预算（12 颗粒子、150 次迭代）下与 CEM、MPPI、GD、GRASP、LeFlow、GC‑IDM 等基线进行比较，本文方法在大多数任务中成功率提升 28–50% 以上，迭代次数平均更少，整体表现优于梯度、采样和 MPPI。

**⚠️ 局限性**

限制与不足：缺乏严格的收敛理论；代价依赖离线数据覆盖度，覆盖不足时潜在空间与真实状态的误差会导致失败；对超参数（如粒子数、步长、温度）敏感；在某些任务中仍出现潜在距离与实际成功判定不匹配的问题。

---

## 201. Learning from Incomplete Actions: Interval Motion Supervision for Vision-Language-Action Policies

**arXiv ID:** 2611.27171 | [PDF](https://arxiv.org/pdf/2611.27171v1)

**作者:** Yunpeng Mei `[一作]` (Beijing Institute of Technology), Gang Wang `[通讯]` (Beijing Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在视觉-语言-动作（VLA）策略中处理连续缺失动作标签的方法，利用可靠的局部动作监督与端点运动约束，通过动作组合和间隔预测实现完整动作学习。

**💡 创新点**

创新点在于提出三重损失框架：局部流匹配、间隔端点监督以及跨窗口动作组合；引入共享动作-间隔专家并使用梯度停止的端点预测作为合成目标；针对长缺口设计多窗口组合策略，并在Ego及真实机器人任务上验证。

**🔧 技术方法**

使用了流匹配（Flow Matching）框架、生成式动作专家、间隔预测分支、梯度停止（stopgrad）、时间窗口注意力机制、SO(3)/SE(3) 运动表示、欧拉积分等技术。

**📊 数据集**

主要使用了 LIBERO 机器人演示子集（20%演示），RoboTwin 机器人双臂仿真（Ego-informed gaps），以及五个真实机器人任务（折叠毛巾、摘花、纸卷、垃圾、块堆叠）与对应的 Ego 视频轨迹。

**📈 对比分析**

与 Hold、Linear、Mask、Interval FM 等基线对比；在 LIBERO 70.5% 缺口情况下，主方法取得 87.9% 成功率，优于 Mask 10.5%；在 RoboTwin 中获得 45.4% 成功率，超过 Mask 40.3%；在五个真实机器人任务中平均 59.6% 成功率，优于 Mask 54.8%。在不同缺口宽度与覆盖度下保持领先。

**⚠️ 局限性**

局限性包括：高度依赖端点视觉估计的质量；缺口内动作连续性的假设易被破坏；对极端长缺口的性能下降；需要手工标签筛选；目前仅在固定窗口长度下实现，长窗口组合仍有待改进。

---

## 202. $R^2$: Recognizing and Reducing Redundant Visual Tool Use

**arXiv ID:** 2611.27070 | [PDF](https://arxiv.org/pdf/2611.27070v1)

**作者:** Yilin Zhao `[一作]` (Tsinghua University), Wenming Yang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出R^2框架，通过样本级工具效用分析、视觉反馈依赖度度量与冗余感知策略优化，减少视觉工具冗余调用并提升多模态推理性能。

**💡 创新点**

首次将工具效用分类（CAT）、无外部评估的视觉反馈依赖度指标（VFD）与基于类别的工具调用成本相结合，形成冗余感知GRPO（RA-GRPO）模型。

**🔧 技术方法**

基于强化学习的GRPO、可解释奖励、对齐的视觉工具调用、VFD计算以及类别条件成本设计。

**📊 数据集**

在七个视觉推理基准（HRBench‑4K/8K、V⋆Bench、VisualProbe、ZoomBench、MMStar、MME‑RealWorld‑Lite）以及VisualProbe‑Train与ZwZ‑RL‑VQA‑10K训练集上进行实验。

**📈 对比分析**

与工具可用与不可用的Qwen3‑VL‑8B‑Instruct、Vanilla GRPO以及开源工具使用模型对比，RA‑GRPO在保持近似准确率的同时平均减少12.39%工具调用，并在工具有害样本上提升13.86个百分点。

**⚠️ 局限性**

仅针对裁剪/放大工具实验，VFD仅为反馈依赖的代理且对多工具和不同教师模型的泛化受限。

---

## 203. Rethinking Cross-Sensor Domain Adaptation for Bridge Point Cloud Semantic Segmentation

**arXiv ID:** 2611.27060 | [PDF](https://arxiv.org/pdf/2611.27060v1)

**作者:** Linghui Huang `[一作]` (Guangdong Provincial Government Loan Repayment Expressway Management Center), Luming Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

重新评估SemanticBridge TLS-to-MLS协议，系统性分析采样密度、表面粗糙度、颜色、视角差异对8cm稀疏体素网络输入的影响，并在同一协议下比较AdaBN、TENT、Mean Teacher和置信过滤自训练等无标签域自适应方法。

**💡 创新点**

在统一的稀疏体素输入空间中量化传感器差异，并揭示源模型收敛性、评估波动和数据质量对跨传感器适应性能的决定性作用；提出从收敛状态出发的自训练并不一定提升性能的洞见。

**🔧 技术方法**

使用稀疏卷积UNet、几何描述子、颜色归一化、噪声扰动、AdaBN、TENT、Mean Teacher、置信过滤自训练等技术。

**📊 数据集**

使用SemanticBridge数据集中的20座桥梁（15座TLS训练、5座TLS验证、4座MLS未标记训练、3座MLS标记评估），并在已发布的1cm预处理下进行实验。

**📈 对比分析**

通过统一训练/评估协议、不同训练计划（100/300 epoch）、多种随机种子和重复评估，发现源模型收敛后mIoU可达57.8%；自训练在未收敛模型下提升约2点，但在收敛模型下效果相反；与已发布模型比较，表现相当或略逊。

**⚠️ 局限性**

仅在单一稀疏卷积骨干、8cm体素、SemanticBridge预处理下验证，缺乏对更细分辨率、不同骨干或更大数据集的通用性；视角覆盖差异和标签一致性限制了评估可靠性。

---

## 204. TouchBench: Can Vision-Language Models Feel Invisible Physical Properties?

**arXiv ID:** 2611.26883 | [PDF](https://arxiv.org/pdf/2611.26883v1)

**作者:** Ziyao Zeng `[一作]` (Meta Reality Labs), Li Guan `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并发布了 TouchBench——一套包含 900 条同步 egocentric 视频与 137 通道触觉手套数据的基准，划分可视与不可视属性的 Tier A/B，并在此基准上评估现有视觉‑语言模型；提出 Tactile‑as‑Prompt（TaP）方法，将 34 条可解释的触觉统计以文本形式插入提示，允许无训练即可让 VLM 读取触觉信息；在相对与绝对问题、零射击与微调场景下与传统仅视频、仅触觉、学习型触觉编码器等多种输入方式进行对比实验。

**💡 创新点**

创新点包括①引入视觉可回答性控制（Tier A/B）来区分可见与隐藏属性，从而精准定位 VLM 的感知瓶颈；②提出无训练的触觉文本提示（TaP），将可解释的触觉特征直接写入语言提示，兼容任意公开或专有 VLM；③在多模态基准上系统验证触觉文本显著提升对隐藏属性的推理，远超学习型触觉编码器与单模态方法。

**🔧 技术方法**

使用了四大主流 VLM（InternVL3‑8B、Qwen3.5‑9B、Claude Sonnet 与 Claude Opus），辅以 137 通道触觉手套与 4 通道腕部 IMU；提取 34 条触觉统计（力学、空间、频谱、腕部稳定性）并序列化为文本；在微调阶段采用 LoRA、AdamW、bfloat16 训练；实验中对比视频、触觉、两者组合以及学习型触觉编码器（1.79M 参数的时间卷积网络）。

**📊 数据集**

核心数据集为 TouchBench：5 位参与者在 11 个日常任务中录制 900 条同步视频‑触觉剪辑；额外使用 20 小时 EgoTouch 公共数据生成无标注网页规模问答，验证触觉文本在更大规模无监督场景的可行性。

**📈 对比分析**

评估方式：对每个问题（相对/绝对）按 Tier A/B 计算宏平均准确率；零射击与留一用户微调（LOUO）两种模式；比较仅视频、仅触觉、两者结合、学习型触觉编码器。结果显示：VLM 在 Tier B 零射击仅 49.8%；TaP 将此提升至 62.9%；微调后达到 73.1%，显著优于视频仅 67.4% 与学习型编码器 67.3%。在未知对象/用户、20 小时网页数据上，TaP 仍保持 4–8 点的提升。

**⚠️ 局限性**

局限性：样本量有限（900 条、5 位手套、仅右手）；不同用户/物体间泛化差异明显；只评估相对比较，无法校准绝对判断；手套采样率 98 Hz 限制了对高频振动的感知；Tiers 由单作者手工定义，缺乏交叉验证；仅使用单一学习型编码器，可能低估其潜力。

---

## 205. FACET: A Data-Efficient Regional Parametric Head Model

**arXiv ID:** 2611.26773 | [PDF](https://arxiv.org/pdf/2611.26773v1)

**作者:** Mathieu Lamarre `[一作]` (Electronic Arts), Étienne Danvoye `[通讯]` (Electronic Arts)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出一种基于区域局部化的面部参数化模型FACET，包含16个解剖区域的身份分解、基于FACS的表达式并加入每身份的神经个性化，以及专门的颈肩区域以降低姿态泄漏。

**💡 创新点**

① 通过区域身份分解突破全局PCA的N-1秩上限；② 用可保持FACS可控性的个性化AU模型实现比全局PCA更优的表达拟合；③ 在颈肩区域局部化以抑制姿态泄漏。

**🔧 技术方法**

局部PCA（位置和Laplacian）、哈希网格编码+FiLM+小型MLP的神经个性化、Instant‑NGP加速、线性混合蒙皮、姿态解耦的颈肩分区。

**📊 数据集**

固定身份池（约346个注册头部扫描）与其对应的表情数据，训练集与测试集完全分离，使用相同的拓扑和眼部归一化。

**📈 对比分析**

与匹配的全局PCA消融模型对比；在身份拟合上，区域模型将平均V2V误差从1.95 mm降至0.23 mm；在表达拟合上，个性化AU模型在所有分量数下都优于全局PCA（平均误差1.47 mm vs 1.67 mm），颈肩分区使上脸姿态泄漏降低3.1倍。

**⚠️ 局限性**

实验仅覆盖少量身份和表情的保留集，评估范围受固定身份池限制；未对大规模数据集或跨域通用性进行验证。

---

## 206. Which Archived Results Survive a Rule Change? A Case Study in Revalidating Decisions

**arXiv ID:** 2611.26666 | [PDF](https://arxiv.org/pdf/2611.26666v1)

**作者:** Zengjian Liu `[一作]` (Peachy Lab), HongLiang Sun `[通讯]` (Peachy Lab)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在一组312条存档记录上重新运行旧版与新版解析器，并结合缺失计算文件的回放，评估旧版结果在规则更改后是否仍可被合法使用。

**💡 创新点**

提出区分“输出是否保持不变”与“当前是否有可用的证明与批准”两种状态的概念，并通过显式批准依赖与内容一致性检查来确保结果可复用。

**🔧 技术方法**

使用确定性四阶段参考实现、GNU Make 3.81 的时间戳与内容检查、synthetic 场景模拟、以及 RFC 3161 第三方时间戳来验证依赖链与证明链。

**📊 数据集**

数据集主要包含312条已归档的解析器版本对记录以及构造的24+8个合成测试案例，此外还使用了历史迁移存档中的计算文件和批处理文件。

**📈 对比分析**

在 Make 中对比显式政策依赖、完整重建与内容检查三种策略，结果显示内容检查在保持相同正确性（24/24）同时显著减少调用次数（约 45 次 vs. 96 次），而带政策依赖的 Make 在 18/24 场景中通过时间戳重建资格。

**⚠️ 局限性**

局限性包括仅分析单一历史迁移、缺失部分文件导致的完整性假设、对 GNU Make 的特定实现依赖，以及未能证明跨系统迁移能真正减少工作量。

---

## 207. Real-Anchored Complementary Evidence for Generalizable AI-Generated Image Detection

**arXiv ID:** 2611.27161 | [PDF](https://arxiv.org/pdf/2611.27161v1)

**作者:** Wenpeng Mu `[一作]` (Shanghai Jiao Tong University), Xinghao Jiang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种基于真实图像规律的通用AI生成图像检测框架RACE，利用全球与局部两种互补证据实现跨生成器的鲁棒鉴别

**💡 创新点**

创新点在于：①仅使用真实图像进行训练和阈值校准；②将DINOv2与CLIP两种预训练编码器的特征进行互补编码；③通过对冻结表示的扰动响应进行归一化（PRP）和局部噪声纹重构的选择性残差评分（NP-RSR）来补充全局与局部证据；④统一归一化与融合，并使用真实图像统计对阈值进行一次性校准

**🔧 技术方法**

技术包括：对齐与加权的特征融合（CFE），基于扰动的响应归一化（PRP），局部纹理描述符构建与掩码重构（NP-RSR），以及对特征与重构误差的统计归一化与融合

**📊 数据集**

使用六个公开生成图像基准（GenImage、ARForensics、QwenImageBench、EvalGEN、ForenSynths、AIGIBench）和混合真实图像数据集（MS COCO、LSUN、ImageNet），并在训练、验证与阈值校准中分别采用5k、1k、5k真实图像

**📈 对比分析**

与13个主流基线（CNNSpot、UnivFD、DIRE、NPR、DRCT、D^3QE、LOTA、DDA、MF^2DA、IAPL、Veritas、RealStats、ConV）比较，RACE在六个基准上平均AUROC 96.59%、AP 95.44%，在五个外域基准上更是高达AUROC 97.18%、AP 96.17%，显著优于传统监督与纯真实图像基线，并且在图像扰动与真实来源变化下保持高鲁棒性

**⚠️ 局限性**

局限性包括：①对参考库规模和邻域大小的超参数敏感；②在极度稀疏或与训练真实图像分布差异很大的新真实数据源时，阈值可能需微调；③模型依赖于多个预训练网络和重构网络，计算和存储成本相对较高

---

## 208. Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric

**arXiv ID:** 2611.26881 | [PDF](https://arxiv.org/pdf/2611.26881v1)

**作者:** Abdullah Ahmad `[一作]` (Independent Researcher), Luca Citi `[通讯]` (University Of Essex)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在视频问答评估管道中，研究了LLM‑as‑judge对演示标签比例的敏感性，利用固定生成的对照实验揭示了演示标签偏差对判定结果的影响；

**💡 创新点**

首次证明演示标签偏差在判定模型中呈单向膨胀特征——偏向“接受”的演示会显著提高判定接受率，而偏向“拒绝”则不会降低其与人类共识的差距，并且量化了演示样本方差的显著性；

**🔧 技术方法**

使用了在上下文学习基础上的LLM判定器（LLaMA‑3.1‑8B）与冻结生成器（LLaVA‑NeXT‑Video‑7B），以及人类双评判一致性验证；

**📊 数据集**

实验数据来自两大开放式视频问答基准：NExT‑QA（多词答案）和MSVD‑QA（单词答案）；

**📈 对比分析**

通过与人工共识的对比和多次抽样平均，结果显示：当演示标签比例平衡或偏向拒绝时，判定接受率与人工共识相近；但当偏向接受时，判定接受率提升约20–30个百分点，误判率显著上升；单抽样误差可达18个百分点，提示需多次抽样；

**⚠️ 局限性**

局限包括：拒绝偏向的实验仅使用单一样本；演示集的类别组成可能混杂；仅评估二分类判定器，未涉及多级或相对判定；使用单一生成器‑判定器组合，未验证跨模型泛化；部分MSVD‑QA类别覆盖不足；人工验证中使用自动仲裁可能略偏向宽容。

---

## 209. Beyond Demographic Discrimination: Benchmarking Normative-Descriptive Calibration in LLMs

**arXiv ID:** 2611.26870 | [PDF](https://arxiv.org/pdf/2611.26870v1)

**作者:** Radhika Bhargava `[一作]` (AWS Applied AI Solutions), Yue Zhang `[通讯]` (AWS Applied AI Solutions)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了Normative–Descriptive Calibration Benchmark（NDCB），用于检测大型语言模型在描述可能结果和推荐行动时是否区分使用人口统计信息的行为。

**💡 创新点**

创新点在于：①首次将描述性与规范性推理区分为匹配对齐评估；②设计三项度量（NER、NLR、NDS）量化模型在两种框架下的等化倾向；③通过NDS展示传统公平基准无法捕捉的模型间差异。

**🔧 技术方法**

技术包括：匹配对齐实验设计、三元方向评分法（等化、优势、劣势）、集群自举置信区间、简化提示模板与候选人顺序敏感性分析。

**📊 数据集**

数据集由697个人工合成的匹配对情境组成，覆盖刑事司法、住房、教育、客户服务、内容审核和招聘六大社会决策领域，涉及种族、性别、社会经济地位与年龄四个受保护属性。

**📈 对比分析**

比较方法：对六大模型（Claude Opus 4.6、Claude Sonnet 4.6、GPT‑5.5、GPT‑OSS 120B、Qwen3‑32B、Mistral Large 3）计算NDS、NER、NLR，并与BBQ准确率与BOLD情感等基准进行对比。NDS在13/15模型对中实现差异化，显著优于BBQ（7/15）且优于BOLD（0/15）。

**⚠️ 局限性**

局限性包括：①仅评估多属性受限组与优势基线的差异，未拆分单一属性贡献；②情境为单一生成模型产生的合成数据，可能带来生成器偏差；③仅覆盖英语和美国语境，缺乏跨文化、多身份维度；④提示模板与候选人顺序对NDS影响显著，需进一步标准化；⑤公平目标设定为等化，未涵盖政策层面的差异化决策需求。

---

## 210. How Much of a Calculation-Error Rate Is Not Arithmetic: Auditing the FinReason Cup 2026 Task 3 Metric with a Deterministic Rule System

**arXiv ID:** 2611.27192 | [PDF](https://arxiv.org/pdf/2611.27192v1)

**作者:** Vladimir Beskorovainyi `[一作]` `[通讯]` (Besk Tech), Vladimir Beskorovainyi (Besk Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对FinReason Cup 2026 Task 3，构建了一个完全确定性、无学习参数的系统，直接实现了三条 DQC_US 规则，并对提交结果进行错误分类与分析。

**💡 创新点**

创新点在于：① 用规则本身重现判定逻辑，避免模型训练；② 通过规则推断证明大部分“计算错误”并非算术错误，揭示官方指标的误导性；③ 发现并讨论了实践集与正式排名集之间的差异，提出对指标与数据集改进建议。

**🔧 技术方法**

主要技术包括：Python 3.10 标准库；XBRL 文档解析、计算链解析；规则推断与排除列表实现；基于规则的候选事实选择与子总计修正。

**📊 数据集**

使用了 FinReason Cup 2026 官方公开的 680 个发展/测试案例（含规则标记），以及 332 个实践集（部分截断），并引用了 Arelle 的 DQC 引擎作为对照。

**📈 对比分析**

与官方评测对比，发展集上 96.62 % ACC，测试集 95.29 % ACC；与基线（规则基线 31.76 %）相比显著提升。通过规则准确率拆分，证明至少 10/24 的计算错误并非算术错误。

**⚠️ 局限性**

局限性在于：① 依赖规则源文件，若规则更新需同步；② 只能处理公开的结构化 XBRL，未覆盖未标记事实的选择问题；③ 评价体系层级化导致计算错误无法分离算术与事实选择误差；④ 仅在公开数据上验证，缺乏未公开正式排名集的直接评测。

---

## 211. ProbeAgent: Principled Diagnosis of Agent Failures in Verifiable Environments

**arXiv ID:** 2611.26897 | [PDF](https://arxiv.org/pdf/2611.26897v1)

**作者:** Yue Huang `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 23664 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一套自动化框架，用于发现并诊断代理（agent）在执行任务时的失败模式。

**💡 创新点**

创新点在于将可执行测试生成、错误解释、子任务复现和失败模式归纳相结合，实现从单个失败到共享错误模式的自动发现与描述；同时引入基于密度约束的聚类和阈值校准，提升错误归纳的准确性。

**🔧 技术方法**

利用大型语言模型（LLM）完成任务构造、检查、错误解释、子任务生成、相似度计算与判定；采用嵌入、余弦相似度、密度约束聚类、LLM判定等技术；整体流程包括probe构建、验证、执行、解释、兄弟任务复现和模式诱导。

**📊 数据集**

使用了10个基准库（涵盖代码编辑、数据管道、终端操作、表格处理、检索等）中的14个种子任务，生成约347个可执行环境，收集了1,244条失败记录。

**📈 对比分析**

通过与固定分类对照、人工评估和held‑out一致性验证等方式比较，框架在22个agent配置上获得了536条失败模式描述，人工接受率90%，held‑out一致率88.5%，并显著减少了无效probe的比例（约44%），展示了高效且准确的失败诊断能力。

**⚠️ 局限性**

局限性包括：对LLM的判定和解释依赖度高，可能遗漏细粒度或罕见错误；构造与验证耗时且受预算限制；对seed任务的依赖导致新领域的覆盖可能不足；在极大规模或多agent协作场景下的可扩展性尚待验证。

---

## 212. What Does the Visual Encoder Buy? Visual Information Gain in Encoder-Free Multimodal Pretraining

**arXiv ID:** 2611.26917 | [PDF](https://arxiv.org/pdf/2611.26917v1)

**作者:** Zhixuan Ge `[一作]` (Rice University), Wei Qiu `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并使用 Visual Information Gain (VIG) 指标衡量视觉信息对多模态预训练模型（encoder‑free 与 encoder‑based）在多模态损失中的贡献，并通过对公开 checkpoint 的评估验证了 encoder‑free 模型在生产规模下能获得与 encoder‑based 相当的布局信息。

**💡 创新点**

创新点：①首次将统一图像、打乱图像与真实图像三种输入结合起来，拆分视觉信息增益为全局信息与布局信息；②通过 VIG 直接量化模型对图像布局（尤其文本读取）的学习进度；③在多模态预训练阶段对模型进行“bootstrap”阶段划分，揭示 pretrained encoder 主要提供布局信息而非纯视觉特征；④在公开模型（Gemma 4、NEO、Mono‑InternVL 等）上评估布局信息，证明 encoder‑free 训练可实现与 encoder‑based 相当的布局学习。

**🔧 技术方法**

技术手段：三向前向传播计算 VIG；使用线性补丁嵌入作为 encoder‑free 前端；对比 frozen SigLIP‑2 encoder；在不同 batch、patch size、OCR supervision 等干预下进行 ablation；在 19 个公开 checkpoint 上对同一验证集进行统一损失评估。

**📊 数据集**

数据集：19.8M 图文样本，来源包括 CC12M、CC3M、Cauldron、LLaVA‑OneVision、Docmatix、Aguvis、ChartGemma、ChartQA、ArxivQA、MathV360K 等；文本来源 FineWeb‑Edu、OpenWebMath、StarCoder、Wikipedia；六类图像类型（caption、chart、VQA、OCR、STEM、GUI）共 32×4,096 令牌验证集。

**📈 对比分析**

比较方法：对同一文本在三种图像条件下计算 VIG，分解为全局信息和布局信息；与公开 encoder‑based checkpoint 的 VIG 进行对比；通过答案准确率验证布局信息对 OCR 读取的贡献；结果表明 encoder‑free 在 bootstrap 阶段仅获得 3%–46% 的图像共享信息，布局信息几乎为 0；而 frozen encoder 在同样数据下布局信息可达 0.6–0.8 nats，提升 10 倍；公开 encoder‑free 模型在生产规模可获得 0.5–0.8 nats 的布局信息，基本接近 encoder‑based。

**⚠️ 局限性**

限制：①VIG 仅反映预训练阶段的多模态损失，无法直接评估下游任务性能；②对不同 tokenizer、训练文本差异的模型无法直接比较整体损失，只能比较差异；③未探究更大模型或更长训练时的布局信息增长曲线；④缺少对不同视觉任务（如 VQA、OCR）细粒度对齐效果的深入分析。

---

## 213. Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models

**arXiv ID:** 2611.26835 | [PDF](https://arxiv.org/pdf/2611.26835v1)

**作者:** Amit Dhurandhar `[一作]` (IBM Research), Rahul Nair `[通讯]` (IBM Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种新的非扩张门控连续分式架构，替代传统的前馈网络（FFN），并通过对除法运算的对数/指数重实现消除梯度不稳定性，适用于 GPT、Llama、Mamba、MoE 等大型语言模型。

**💡 创新点**

创新点包括：①引入非扩张门控机制，使隐藏表示不需扩展；②用对数/指数重写除法非线性，避免递增训练步骤；③将该模块在多种模型（含 MoE）中实现，证明参数占比可减至 1/3~1/2，且性能提升或相当。

**🔧 技术方法**

技术手段主要是：连续分式网络（CoFrGeNets）改进、门控线性单元（GLU）结合、对数/指数变换代替除法、PyTorch实现、可与标准预训练流程无缝集成。

**📊 数据集**

使用的主要数据集包括：OpenWebText、GneissWeb 35B、Docling 2T 数据混合（含 Web、Multilingual、Code、Math、Synthetic 等子集），以及针对 GPT2 的 Penn Tree Bank、Wikitext2/103、Lambda、AgNews、LM1B 等评测数据集。

**📈 对比分析**

对比方法：与原始 FFN、之前的 CoFrGeNet-F、以及仅门控的基线模型进行对比；在 GLUE、OpenQA、Piqa、ARC、Winogrande、HellaSwag、Lambda、Boolq、SciQ 等任务上评测准确率；在 GPT2 上评测困惑度（perplexity）和训练吞吐量、FLOPS；实验显示 CoFrGeNet-FNG 在保持参数约 33%~50% 减少的同时，往往比原始模型更优或相当，且训练速度提升 10–30%。

**⚠️ 局限性**

局限性：仍易受提示注入、幻觉等攻击；实现尚未做内存高效优化（如 Triton）；未在更大规模 MoE 上验证；对极端深度模型的可扩展性和鲁棒性仍待进一步研究。

---

## 214. A Metamaterial Agentic Swarm: Coordinated Search for Discrete Mechanical Design

**arXiv ID:** 2611.26743 | [PDF](https://arxiv.org/pdf/2611.26743v1)

**作者:** Achuth Chandrasekhar `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 14577 | [OpenAlex ID](https://openalex.org/A5008745801)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种由 40 个专门化搜索代理组成、分为 5 个岛屿的元材料代理群体，用来在像素级机械隐形斗篷中从离散单元库中选取单元格，以最小化外部位移误差。

**💡 创新点**

创新点在于将多种代理专长（全局侦查、空间组织、突变、档案学习、重组、多样化重启、邻域探索与驻点抛光）与岛屿并行、精英档案共享以及局部多样化机制结合，形成可扩展的算法组合，首次在全像素离散装配问题上实现了低于阈值的精确设计。

**🔧 技术方法**

使用了数值搜索代理、离散有限元评估（全像素 1500×1500 Q4 元件）、精英档案学习（统计分布估计）、岛屿并行调度、以及温度接受的邻域探索和局部抛光技术。

**📊 数据集**

采用了研究者提供的 15 种候选单元格库、对应的观察掩模以及参考位移场的数据集。

**📈 对比分析**

在三个随机起点、961 次评估下，最佳失配损失为 0.10059，低于阈值 0.1102（下降 8.72%），平均失配从 601 次评估的 0.16278 降至 0.12524（下降 23.06%），但未与其他基线或并行度进行严格对比。

**⚠️ 局限性**

局限性包括仅测试单一几何与载荷配置、缺乏角色/通信消融和效率基准、端点选择存在偏差、数据集与参考解不可公开，且未对并行性能或物理鲁棒性做进一步验证。

---

## 215. One state, many questions: Opsis gives a frontier model typed eyes for pathology

**arXiv ID:** 2611.27102 | [PDF](https://arxiv.org/pdf/2611.27102v1)

**作者:** Zhixuan Ge `[一作]` (Rice University), Wei Qiu `[通讯]` (Rice University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于冻结病理编码器的typed question interface，使用10M参数解码器一次性回答切片与字段级问题并输出概率，支持组合与弃权；

**💡 创新点**

创新点在于将病理报告视为可类型化问题集合，利用单一冻结编码器+解码器即可完成所有字段，无需为每个问题训练专属分类器，并证明“asking”比阅读/观看更精准，概率可用于定位、迁移与弃权；

**🔧 技术方法**

使用的技术包括冻结Virchow2 ViT-H/14编码器、10M参数Transformer解码器（两层）、Per-tile heads、Qwen3-VL-8B内置实现、功能调用与规则/前端模型组合；

**📊 数据集**

采用TCGA 10,712诊断切片（32项目，8,861患者）以及NuCLS、BCSS、PanNuke等细胞/组织标注数据，另外使用CAMELYON16、Breast TCGA等验证集；

**📈 对比分析**

与每题训练的CellViT++、ABMIL、线性探针（TITAN/PRISM2）、提示式文本模型（Qwen3-VL-8B、Patho-R1-7B）和Claude Opus 5.5等进行比较，解码器在字段级别平均准确率约0.69/0.70，问答模型在breast问题上达到0.78–0.79，阅读/观看模式低于majority class；定位AUROC>0.9，迁移性能优异；

**⚠️ 局限性**

局限性包括对部分字段（如肺、膀胱）仍低于majority class；解码器性能依赖冻结编码器，无法实时更新；在未见器官时准确下降；需要预先定义operator集合和对应概率，且对新任务仍需人工设计规则或组织者。

---

## 216. Can Reasoning and Image Description Enhance Yes/No Visual Mathematical Understanding?

**arXiv ID:** 2611.27302 | [PDF](https://arxiv.org/pdf/2611.27302v1)

**作者:** Guangcheng Song `[一作]` (City University of Hong Kong), Linqi Song `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究大规模视觉语言模型在二分类视觉数学问题上的表现，构建MathYN数据集并评估不同提示和输入格式。

**💡 创新点**

系统比较了链式思考提示与描述中介对模型判断一致性的影响，揭示显式推理提升一致性但描述中介不稳定。

**🔧 技术方法**

采用链式思考提示、思考型模型、图像描述生成与多种输入格式的多模态提示技术。

**📊 数据集**

使用MathYN数据集，来源于MathVista、DynaMath和HallusionBench的数学图像问答。

**📈 对比分析**

对比直接与链式提示、图像/描述/增强输入，发现链式提示显著提升准确率与配对一致率，特别是Qwen3系列模型，表现从约66%提升到约74%/79%。

**⚠️ 局限性**

实验仅涵盖中等规模开源模型，未考虑更大规模模型；仅评估二分类问题；描述中介效果受生成描述质量限制，需更细粒度评估。

---

## 217. EAF-TTS: Rethinking Codec-Latent Coordinates for Flow TTS

**arXiv ID:** 2611.27214 | [PDF](https://arxiv.org/pdf/2611.27214v1)

**作者:** Yumiao Li `[一作]` (AI Lab, Country Garden Services Group), Yue Qu `[通讯]` (Tongji University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种Embedded Acoustic Flow TTS (EAF‑TTS)模型，使用冻结的WaveVAE解码器与流匹配技术，并在声码器潜在空间上引入ZCA白化来改善声学潜在的生成质量。

**💡 创新点**

核心创新在于证明仅通过全局ZCA坐标变换（无新参数学习）即可显著提升自动化自然度指标，并且该改进可以在保持语音质量的前提下换取更少的推理调用次数。

**🔧 技术方法**

采用的技术包括流匹配（Flow Matching）、声码器潜在表示（WaveVAE）、ZCA白化（全局协方差正则化）、共享网络终端校正、Scalar与ZCA坐标对比、UTMOS22代理自然度评估、WER与DTW‑MFCC内容评估等。

**📊 数据集**

使用的语料为单声道LJSpeech数据集，划分为训练集（12080句）、验证集（399句）与测试集（399句），并在此基础上训练WaveVAE编码器/解码器。

**📈 对比分析**

通过在严格GT验证集Val399上对齐种子（42、2027、3407）进行Scalar与ZCA三种坐标的直接对比，平均UTMOS提升约+0.405±0.027；ZCA-5在保持相近WER的同时，实现了相对Scalar-20约3.34倍的RTF加速，ZCA-3则在更低调用次数下维持+0.165的自然度优势。

**⚠️ 局限性**

局限性包括：依赖UTMOS22等自动代理自然度指标而未进行人类听感测试；仅验证单一说话人、单一声码器和单一潜在维度，缺乏跨语音、跨说话人和多语言的推广性；ZCA的效果与其它坐标或损失指标变化难以分离；种子数目有限，难以进行高功效的因子分析。

---

## 218. From Acting to Learning: Experience Selection toward Continually Learning Agents

**arXiv ID:** 2611.27159 | [PDF](https://arxiv.org/pdf/2611.27159v1)

**作者:** Yaoyuan Yan `[一作]` (Country Garden Services Ai Lab), Anwan Yan `[通讯]` (Omni Ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对4B语言代理在持续学习中使用的经验进行优化器感知的全局梯度选择，并通过对九种选择器的系统比较验证其有效性

**💡 创新点**

提出了基于LESS式梯度特征的全局梯度选择方法，能够在不受任务/动作配额限制的情况下显著提升学习效果

**🔧 技术方法**

使用LoRA微调、AdamW优化器特征计算、LESS梯度加权评分、Qwen3-4B-Instruct模型及其原始基础权重

**📊 数据集**

使用ALFWorld文本交互环境中的两份固定经验池（API和Generic），包含约700条可用动作演示

**📈 对比分析**

将梯度选择与全局随机选择、全局高NLL、匹配配额选择等基线进行对比；在开发集上梯度选择分别达81.45%和73.19%的成功率，较基线提升约5–8个百分点；在保留任务集上亦保持优势，提升约5–9个百分点

**⚠️ 局限性**

仅在单一文本环境和单一模型上验证，未考察多轮学习、长期记忆保持、其他基础网络或物理机器人控制等场景，且缺乏显式的统计显著性检验

---

## 219. NormFlowSDE: Normalizing Flows for Amortized Bayesian State Estimation

**arXiv ID:** 2611.27362 | [PDF](https://arxiv.org/pdf/2611.27362v1)

**作者:** Yi Zhang `[一作]` (University of Augsburg), Lars Mikelsons `[通讯]` (University of Augsburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出NormFlowSDE，一种将条件正则化流与神经随机微分方程结合的可预估贝叶斯状态估计器；

**💡 创新点**

创新点在于把可逆正则化流嵌入神经SDE的噪声项，实现观测条件下的直接状态更新，并证明其弱收敛性；

**🔧 技术方法**

利用神经SDE、条件正则化流、归一化流、以及多段子投射（multi‑shooting）与能量分数等训练技巧；

**📊 数据集**

在两个模拟系统（受控摆、Lorenz‑96）和两个机器人数据集（ALOHA插头、REASSEMBLE装配）上进行评估；

**📈 对比分析**

与三种直接估计器（latent NSF、latent SDE、SigKer SDE）和四种基于粒子的学习滤波器（NF‑DPF、NSF‑PF、FlowSDE‑PF、NormFlowSDE‑PF）比较；在机器人数据集上NormFlowSDE在RMSE、CRPS和置信区间覆盖率上均优于所有粒子滤波器，在模拟系统上粒子滤波器略胜；且NormFlowSDE在推理速度上比最快粒子滤波器快3.9–4.2倍；

**⚠️ 局限性**

局限包括：对部分可观测的混沌动力学（如Lorenz‑96）仍难以精准；训练需要手工调参；仅适用于已知初始状态、固定时间步长和窗口长度的情况；需要训练数据中含有真实状态标签，实际部署时若无参考传感器则需额外估计。

---

## 220. When Do LLM-Elicited Priors Help? A Controlled Audit of Language Models as Priors for Small-Sample Bayesian Regression

**arXiv ID:** 2611.27075 | [PDF](https://arxiv.org/pdf/2611.27075v1)

**作者:** Ran Zou `[一作]` (University of California, Irvine), Yang Liu `[通讯]` (University of North Carolina at Chapel Hill)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对使用大型语言模型（LLM）进行贝叶斯回归的先验中心化进行系统审计，并将其与自适应零均值先验进行对比

**💡 创新点**

创新点在于：①提出了基于LLM的先验中心化与可信度校准的完整评估框架；②发现LLM中心化在真实数据上仅在极少数情形下略优，且错误高度相关；③给出了利用数据校准尺度、鲁棒混合与门控策略的实用建议

**🔧 技术方法**

使用了LLM推理（Anthropic、Google Gemini）、类型II最大似然校准、强健混合、对数池化与门控机制，以及传统的弱先验、经验贝叶斯和岭回归

**📊 数据集**

使用了六个公开回归数据集：California（半合成与真实）、Diabetes、Breast Cancer、logistic stress test、NYC‑2026以及18个额外的UCI/OpenML/PMLB数据集

**📈 对比分析**

通过100对随机拆分、多种损失（RMSE、trimmed RMSE、系数误差）进行对比；结果显示LLM中心化在半合成California和NYC‑2026上分别降低5.8%和3.2%的RMSE，其他场景提升不足1%或略显负面，外部验证中仅在6/18问题上获益，整体相较传统自适应先验表现平平

**⚠️ 局限性**

主要限制包括：仅评估少数LLM模型和提示，数据规模受限于p≤10、n≤150，LLM尺度不可靠且错误高度相关，未针对高维或非表格数据、缺乏公平性与偏见审计，且缺乏完整的预注册流程

---

## 221. TSIR: Ray-Traced Triangle Splats for Inverse Rendering

**arXiv ID:** 2611.27030 | [PDF](https://arxiv.org/pdf/2611.27030v1)

**作者:** Wentao Yang `[一作]` (Zhejiang University), Xiangru Huang `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了TSIR两阶段逆渲染框架，通过三角形光斑重建可重光的几何与材质，实现高质量的重光与可视化。

**💡 创新点**

创新点包括：1）将三角形光斑与Monte Carlo PBR渲染相结合，直接光线追踪表面，减少多层透明度合成；2）使用共享三平面纹理和可学习的顶点阴影法线提升材质一致性；3）开发基于子像素可见度的抗锯齿方法，降低边界伪影。

**🔧 技术方法**

采用了TS+重建、BVH光线追踪、共享三平面（triplane）纹理、蒙特卡洛直接与间接照明估计、子像素抗锯齿以及差分光栅化nvdiffrast等技术。

**📊 数据集**

使用Synthetic4Relight和TensoIR两个合成数据集进行实验。

**📈 对比分析**

与六个基线（NvDiffRec、NvDiffRecMC、TensoIR、GS-IR、IRGS、GeoSplatting）比较，实验显示在Synthetic4Relight上PSNR提升3.10dB，在TensoIR上提升0.73dB；在Hotdog场景下FPS为14.02（相较于IRGS高30×），使用768条光线采样。

**⚠️ 局限性**

局限性包括：仍需大量光线采样以获得高质量间接光；对复杂遮挡与光源变化的鲁棒性有限；训练依赖多视角数据，难以直接迁移到单视角场景。

---

## 222. The Complexity of Fair and Stable Data Exchange

**arXiv ID:** 2611.27438 | [PDF](https://arxiv.org/pdf/2611.27438v1)

**作者:** Hannaneh Akrami `[一作]` (Max Planck Institute for Informatics), Bhaskar Ray Chaudhury `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f86bf285-fd08-4156-973b-6e6481af8fa0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种新算法，能够在子模（交替）收益场景下通过循环增广和受控递减实现可多项式时间的近似公平与核心稳定数据交换；同时证明了在补充收益场景下，即使共享信用易于评估，求解近似公平且核心稳定交换也属于PPAD‑hard；

**💡 创新点**

核心创新在于：①针对交替收益构造的交叉单调共享模型，开发了基于流环增广与受控递减的双阶段算法，实现了FPTAS；②通过构造星形交换与指令图映射，证明了补充收益情形下的PPAD硬度，首次给出互补数据集与公平稳定性之间的计算壁垒；

**🔧 技术方法**

主要技术包括：交叉单调共享的流环增广、受控递减（ControlledDecrease）算法、Lipschitz 约束下的共享查询、Shapley 价值的精确表达以及从公共物品均衡到数据交换的多项式归约；

**📊 数据集**

本工作为理论性研究，未使用任何实际数据集；所有结论均基于数学建模与算法分析；

**📈 对比分析**

与以往基于局部搜索的子模收益方法相比，本文的算法在多项式时间内获得与理论上最优的近似公平与核心稳定解；在补充收益场景中，则通过PPAD硬度证明了先前方法无法得到多项式时间近似解；

**⚠️ 局限性**

限制在于：①仅适用于交叉单调共享（如Shapley共享）且依赖可多项式时间的共享查询；②对补充收益（超模）场景，虽证明为PPAD‑hard，但仍未给出下界或更紧的复杂度分类；③算法实现需要对Lipschitz参数及ε进行细致调参，实际可行性需进一步实验验证。

---

## 223. CoPyro: Feeding Copy Proposals to Parallel Drafters

**arXiv ID:** 2611.27217 | [PDF](https://arxiv.org/pdf/2611.27217v1)

**作者:** Hong-Kai Zheng `[一作]` (Nanjing University of Aeronautics and Astronautics), Piji Li `[通讯]` (Nanjing University of Aeronautics and Astronautics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在并行式投票式推断（speculative decoding）中，提出 CoPyro 方法，通过检索上下文中的 n‑gram 匹配，将复制得到的后续 token 直接作为输入喂给并行 drafter，替代原本的 mask 输入，并让 drafter 在同一次前向传播中决定保留、修改或忽略这些复制的 token。

**💡 创新点**

创新点主要有：① 将复制得到的连续 token 视为输入而非仅作为候选草稿，允许并行 drafter 学习如何利用这些信息；② 设计轻量级的匹配长度和一致性（agreement）嵌入，帮助 drafter 评估复制 token 的可靠性；③ 在训练时使用 proposal dropout 使 drafter 同时能在有无 proposal 的情况下保持性能。

**🔧 技术方法**

使用的技术包括：n‑gram 检索得到复制连续；并行式 draft 模型（DFlash、Domino、DSpark、DFlash2）；speculative decoding 框架；match‑length 与 agreement 嵌入；proposal dropout；与 Transformers、SGLang 兼容的实现；以及 BF16 SDPA 注意力的加速。

**📊 数据集**

数据集：Qwen3‑4B‑Instruct‑100K、PerfectBlend（710K 样本）；目标模型：Qwen3‑4B 与 Qwen3‑8B。评测基准覆盖数学推理（GSM8K、MATH‑500、AIME25）、代码生成（HumanEval、MBPP、LCB）、对话（MT‑Bench、Alpaca）等八个任务。

**📈 对比分析**

与原始并行 drafter、纯 n‑gram speculative decoding 以及 Switch（基于匹配长度的切换策略）进行对比。CoPyro 在所有四种并行 drafter 上平均提升 2.19–5.08% 的接受长度（acceptance length），并端到端吞吐量提升 0.99–4.04%；代码生成任务获益最大（如 HumanEval 增幅 5.67–9.03%），数学与对话任务提升有限。整体上，CoPyro 在 710K 训练样本情形下仍保持正向改进。

**⚠️ 局限性**

局限性：① 在已训练强劲的 DFlash（710K 训练样本）中改进幅度较小，表明复制提议对已具备高质量预测的模型收益有限；② 该方法依赖上下文中存在可复制的重复 token，若文本缺乏可用 n‑gram 片段，效果可能下降；③ 额外的 proposal 处理在 Transformers 上每轮仅增加约 0.24 ms，但在高并发或更大模型规模下可能成为瓶颈。

---

## 224. Parasite Features: Causal Abstraction Through Spurious Pathways

**arXiv ID:** 2611.26711 | [PDF](https://arxiv.org/pdf/2611.26711v1)

**作者:** Denis Sutter `[一作]`, Tiago Pimentel `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究发现并演示了在因果抽象框架中，线性对齐映射可能会通过将两个互相独立的隐藏空间方向组合，产生伪造的因果特征（称为寄生特征）。

**💡 创新点**

创新点在于提出寄生特征的形式化定义、提供可证明的线性对齐条件（定理），并通过实验验证在合成句法一致性和两位相加任务中确实存在此类伪造特征，从而揭示因果抽象方法的潜在风险。

**🔧 技术方法**

主要技术包括：因果抽象与分布式对齐搜索（Distributed Alignment Search）、受限线性对齐映射（在完整线性映射基础上仅扩展至前两维的可逆线性变换）、干预训练与评估、干预空间的可视化与离散化。

**📊 数据集**

数据集包括：1）自定义句法一致性任务（基于主语性别/数的句子），以及 2）标准的两位相加加法任务（输入为十位和个位相加，输出为正确和）。

**📈 对比分析**

与仅使用旋转映射的对齐方法相比，受限线性映射在两任务中表现出更高的因果干预有效性（Odds 和准确率提升显著），但对齐的子空间在后续消融实验中并非必要（仅导致微小性能下降）。

**⚠️ 局限性**

局限性包括：① 只在特定任务与模型上验证，未覆盖更复杂的模型或非线性特征；② 研究聚焦于线性对齐，未探讨更通用的非线性映射导致的寄生特征；③ 结果受限于人工构造的寄生任务，真实任务中寄生特征的检验尚不充分；④ 可能与因果抽象的非可识别性问题相互作用，进一步增加解释困难。

---

## 225. Observed, Curated, Reused: Observation Bias in Clinical AI Evaluation

**arXiv ID:** 2611.27301 | [PDF](https://arxiv.org/pdf/2611.27301v1)

**作者:** Md Shamim Ahmed `[一作]` (University of Southern Denmark), Richard Röttger `[通讯]` (University of Southern Denmark)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文追踪了 6,503 例基于 HPO 的罕见病病例，比较了在知识库中加入或移除病例对应论文后的诊断排名变化，揭示了因源文件共用导致的观察偏差。

**💡 创新点**

创新点在于首次系统性量化了“源文件自反证据偏差”，通过差分设计与背景增长对照证明诊断性能提升与单篇论文关联，并提出基于证据来源的评价框架。

**🔧 技术方法**

使用了表型驱动的检索式诊断器（Resnik、IC‑cosine、LIRICAL），对 MIMIC‑IV 和 eICU 进行数据完整性审计，并采用差分回归、置信区间等统计方法评估效应。

**📊 数据集**

主要数据集包括 Phenopacket Store（6,503 病例）、HPO 多版本（2021‑2026）、MIMIC‑IV 与 eICU 临床记录，以及 9,722 条已发表的疾病‑表型关联。

**📈 对比分析**

对 991 例评估病例，Resnik 的平均倒数排名（MRR）从 0.26 提升至 0.84（加入源文献），但在移除源文献后降至 0.05；IC‑cosine 与 LIRICAL 亦呈类似趋势；背景 HPO 增长对 MRR 的影响仅为 +0.002，验证提升与源文献关联。

**⚠️ 局限性**

局限包括：仅关注罕见病表型检索模型，未涉及大规模机器学习或临床决策场景；评估受 HPO 版本间时间戳粗化限制；可能忽略同一表型的多源支持；未能彻底检验不同下游算法的鲁棒性。

---

## 226. Beyond Prompt Imitation: Capability-Aware Video Generation with Smaller Models

**arXiv ID:** 2611.27252 | [PDF](https://arxiv.org/pdf/2611.27252v1)

**作者:** Ziqi Ding `[一作]` (University of New South Wales), Wei Song `[通讯]` (Griffith University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过“oracle控制梯子”评估了小型视频生成器在逐步获取目标视频信息时的性能提升，并提出了一种基于多面Rasch模型的能力预测与预算化选择策略。

**💡 创新点**

创新点包括：1) 以目标视频为参照的oracle控制梯子，提供每个信息层级对生成器闭合度的可经验上限；2) 将多面Rasch模型与视频特征结合，形成可解释的能力模型；3) 将预算化选择视为Pandora盒子问题，利用先验预测制定最优信息开启顺序，并在信息成本化后仍能超越固定控制策略。

**🔧 技术方法**

使用的技术包括：多面Rasch模型（IRT）、梯度提升机（GBM）做预测、随机采样与bootstrap估计、Pandora盒子最优策略求解、基于内部评判器的Composite评价、以及多种视频质量与结构评估指标（DINO、CLIP、Struct、Depth、Color、Flow、Camera、Object、Checklist）。

**📊 数据集**

数据集涵盖了954个参考片段，来自61个用户录制视频与四个商用生成器（Kling、Sora、Veo 3、Vidu Q1）以及VBench-2.0的教师视频，覆盖约5秒长、16fps或49帧的片段。

**📈 对比分析**

方法比较通过在同一学生（Wan2.1-VACE-1.3B）上逐级提供不同控制信息，测量闭合度（gap closure）和Composite评分；与基线（仅文本、两轮Prompt优化）及预算化策略进行对比。结果显示：dense caption闭合度低，首帧能闭合大部分外观gap，depth视频闭合运动gap，二者组合能闭合64–94%的所有指标；预算化策略在信息成本化后可比固定控制策略提升约2–11%的闭合度，且在弱检验器条件下仍保持鲁棒性。人类评审显示Composite评分与人类偏好一致率约80%。

**⚠️ 局限性**

局限性：①控制梯子仅在学生固定且有目标信息的前提下评估，未考察可变架构或训练方式的超越能力；②能力模型特征来自目标视频，文本仅部署尚未验证；③深度与姿态控制仅在单一学生上验证，跨模型迁移未知；④评估指标和Checklist在事件顺序方面可靠性不足；⑤策略重放基于预生成的grid，缺少在线动态反馈。

---

## 227. Epistemic Guard: Out-of-Process MCP Receipt Proxying, Post-Turn Prose Supervision, and Provenance-Gated Memory for Tool-Using Autonomous Agents

**arXiv ID:** 2611.27095 | [PDF](https://arxiv.org/pdf/2611.27095v1)

**作者:** Sean Maxwell `[一作]` `[通讯]` (Roll Solo), Sean Maxwell (Roll Solo)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了一套针对语言模型代理的运行时验证框架，包括 MCP 接口的签名收据代理、CPU INT8 文本监督、SQLite 写入门限与 RAG 去重。

**💡 创新点**

将 Ed25519 收据与 RFC 6962 Merkle 日志结合、在子进程中隔离签名密钥、利用 CPU INT8 推理对自然语言行为声明进行符号+神经监督，并在存储层实施领域分离的写门限。

**🔧 技术方法**

Python、Ed25519 + HKDF、Merkle 树、OpenVINO INT8 交叉编码器、词 shingle Jaccard/SimHash、SQLite 触发器、CPU 端 NLI 推理。

**📊 数据集**

冻结评测数据集 (N≈175) 与实时故障注入跟踪 (N≈136)，包含安全危害标注、NLI 样本，并使用 HuggingFace 上的 INT8 模型权重。

**📈 对比分析**

与 12B–14B GPU LLM 判定器以及 DeBERTa‑v3 NLI 进行对比，准确率达 97%+，误报率低于 3%，CPU 延迟 80 ms p50，显著低于 GPU 判定器的 200–300 ms 并节省 16 GB VRAM。

**⚠️ 局限性**

仅支持 MCP，无法处理流式 HTTP/SSE；收据只能证明下游返回，恶意服务器仍可伪造；RAG 去重仅对字面/轻编辑复制有效，对语义改写不够；需手工注册工具词干与跨句子失效传播。

---

## 228. Testability of Homomorphism Inadmissibility: Property Testing Meets Database Theory

**arXiv ID:** 2611.26861 | [PDF](https://arxiv.org/pdf/2611.26861v1)

**作者:** Hubie Chen `[一作]` (King's College London), Yuichi Yoshida `[通讯]` (National Institute of Informatics)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文将关系数据库中的布尔联结查询转化为图同态判定问题，研究在属性测试框架下，如何用常数次数查询就能判断给定数据库结构中是否存在该查询的同态映射，即判定“同态不可行性”。

**💡 创新点**

创新点在于给出一个完整的二分理论：当查询对应的核心结构是α‑acyclic时，其同态不可行性可以在常数查询下用一侧误差完成测试；否则必需至少多项式级别的查询。该结论还推广到注入同态和联结查询的并集，并给出了相应的下界。

**🔧 技术方法**

主要技术包括：
- 利用完成查询（completion oracle）抽样获取边/元组；
- 将查询转化为关系结构，并通过核心化简；
- 证明α‑acyclic结构的特性可通过按顺序扩展局部映射实现同态构造；
- 通过集合论与概率分析给出拒绝概率下界；
- 用符号分离（symbol‑distinct）结构和子结构递归归约，得到下界。

**📊 数据集**

由于本研究为理论性工作，未使用任何真实数据库或图数据集；所有结果均在抽象的关系结构和超图模型上证明。

**📈 对比分析**

性能：当核心α‑acyclic时，构造的测试器在查询次数上是常数，具体为 O((m·2^{ar r}/ε)^m)，其中 m 为核心的元组数，ar r 为最大关系 arity。若非α‑acyclic，则至少需要 Ω(n^{1/c}) 次查询（c 为结构相关常数），与已知的子图自由性测试下界相匹配。

**⚠️ 局限性**

限制：
- 仅适用于一侧误差（one‑sided error）测试；
- 结论依赖核心的α‑acyclic性，对非α‑acyclic结构无法提供常数查询测试；
- 对高 arity关系或非平凡数据库实现细节（如多重关系、取值域动态变化）未作深入讨论；
- 目前仅讨论布尔联结查询，未覆盖更一般的SQL子句或非凸查询。

---

## 229. ChatAtom: A Multi-Agent LLM System for Molecular Dynamics Simulation and Visualization

**arXiv ID:** 2611.26782 | [PDF](https://arxiv.org/pdf/2611.26782v1)

**作者:** Trupti Mohanty `[一作]` (Argonne National Laboratory), Mathew J. Cherukara `[通讯]` (Argonne National Laboratory)

**通讯引用:** 4466 | [OpenAlex ID](https://openalex.org/A5066462592)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套名为ChatAtom的多智能体LLM系统，能够根据自然语言请求自动生成、审查并执行分子动力学模拟脚本、可视化流程，并进行结果分析，支持实时流式输出和自适应实验设计。

**💡 创新点**

采用生成‑审查对的脚本生成机制、确定性工作流与工具调用的混合架构、可加载的假设生成/测试技能，以及浏览器实时流式交互，显著降低MD工作流程门槛并实现端到端自动化。

**🔧 技术方法**

结合大语言模型、LangGraph/DSPy工具调用框架、LAMMPS、OVITO、OpenKIM原子势、Globus HPC接口、Python后端、浏览器前端以及知识库规则进行脚本审核。

**📊 数据集**

使用8个基准MD任务提示、6个前沿LLM模型、OpenKIM原子势库以及铜、铝裂纹扩展和破裂仿真案例作为实验数据。

**📈 对比分析**

通过与零射击提示相比，对相同模型在ChatAtom、ChatAtom+规则三种条件下生成脚本的可执行率与物理结果正确率进行统计；结果显示可执行率从92/144提升到132/144，物理正确率也显著提高；在裂纹传播案例中实现完整的多阶段模拟并成功完成自主实验循环。

**⚠️ 局限性**

系统仍无法保证科学解释的严谨性，易产生不合理假设（如错误使用裂纹强度阈值），缺乏内置的物理规则校验和结果验证，需人工审查及进一步可观测性与科学检查机制。

---

## 230. ParaBMG: A Parallel Block Multigrid Framework for Large-Scale Evolving Sparse Systems on Unstructured Meshes

**arXiv ID:** 2611.27396 | [PDF](https://arxiv.org/pdf/2611.27396v1)

**作者:** Kangbo Lyu `[一作]` (Tsinghua University and Shanghai Qi Zhi Institute), Tao Du `[通讯]` (Tsinghua University and Shanghai Qi Zhi Institute)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 ParaBMG，一个在 GPU 上实现的并行块多重网格框架，用于高效求解大规模演化稀疏线性系统，并构建了统一的几何处理基准。

**💡 创新点**

创新点在于将基于网格的几何多重网格层次结构与基于分区的块 Jacobi 平滑器结合，利用可重用层次、块结构存储和 GPU 友好运算，实现了高并行度、低更新成本和强收敛性的解决方案。

**🔧 技术方法**

使用了 GPU 并行层次构造、块 Jacobi 预条件、Galerkin 细化、幂迭代阻尼估计、稀疏块存储、CUDA 对四条几何处理流水线的实现，以及自适应谱估计与块逆。

**📊 数据集**

构造了 40,987 个线性系统，涵盖 20 个三角/四面体网格，顶点数从 17.7K 到 6.6M，涉及局部变形、皮肤权重计算、特征保留滤波以及逆均值曲率流优化四个标准几何处理管线。

**📈 对比分析**

与 AMGX、BoomerAMG、GPU 版 GravoMG/Tet 以及现有 MAS 实现进行对比，ParaBMG 在平均速度提升 3.7×–105×，实现了显著的层次重用、迭代次数降低和 GPU 运算加速。

**⚠️ 局限性**

局限性包括缺乏对非 SPD、耦合 3N×3N 系统、多 RHS 以及非线性系数分布极端变化的支持；参数调优缺乏自动化；在系数变化极大时重建策略仍需进一步改进。

---

## 231. Bayesian Optimal Stopping: Invincible Prophets and Achievable Philosophers

**arXiv ID:** 2611.26829 | [PDF](https://arxiv.org/pdf/2611.26829v1)

**作者:** Alireza AmaniHamedani `[一作]`, Amin Saberi `[通讯]` (Stanford University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究了Bayesian最优停止问题，提出了一套迭代primal‑dual改进框架，既能在多种Bayesian模型下得到对Bayes‑optimal策略的近似（近乎PTAS），又能给出对prophet基准的上界与下界；

**💡 创新点**

创新点包括：①构造可逐步改进的双子鞭马丁格尔；②在单一算法中实现对philosopher的(1−ε)逼近；③用该框架证明prophet难度随候选分布数K而递减，仅能达到1/e+O(1/√K)；④将primal改进与马丁格尔收缩耦合，形成可在近多项式时间内实现的迭代方法；

**🔧 技术方法**

核心技术：动态规划与Snell包络的对偶性；Doob可选停定理与鞭马丁格尔构造；随机阈值混合策略（Q在[α,1]按1/q密度分布）；Monte‑Carlo重要抽样估计双子鞭马丁格尔；多层递归估计与误差控制；

**📊 数据集**

论文未使用真实数据集，所有结果均为理论分析与构造实例；

**📈 对比分析**

对philosopher的竞争比可逼近1，算法运行时间为O(Km·n^{1/ε+log^*n})，在K=O(1)时为PTAS；对prophet，在模型中可达到1/e+1/K，且证明无算法能超过1/e+Ω(1/√K)；

**⚠️ 局限性**

局限性：需使用概率与值oracle，采样oracle无法得到同样的逼近；算法时间仍超多项式（尤其是对一般n的n^{1/ε}）；对更广泛的Bayesian模型与组合优化问题的适用性仍未完全解决。

---

## 232. Stratum: A Hardware-Software Co-Design Framework for SNN Inference on RISC-V

**arXiv ID:** 2611.27277 | [PDF](https://arxiv.org/pdf/2611.27277v1)

**作者:** Wajahat Riaz `[一作]` (Lahore University of Management Sciences), Muhammad Ali Siddiqi `[通讯]` (Lahore University of Management Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本工作提出了Stratum框架，整合SNN训练（snnTorch）、MLIR编译与gem5 RISC‑V周期级仿真，实现了硬件与软件协同设计的统一流程。

**💡 创新点**

创新点在于提供可分离评估的软件与硬件参数、两种针对SNN的编译器变换（spike‑gated 降低与 spike‑major 权重布局）以及阶段级计时器，能够精确定位性能瓶颈。

**🔧 技术方法**

采用的技术包括snnTorch训练、NIR/MLIR编译管线、两种自定义MLIR pass、LLVM后端编译到裸机RISC‑V可执行文件，以及gem5循环级模拟与周期计数器标记。

**📊 数据集**

实验使用N‑MNIST数据集，构建两层全连接SNN。

**📈 对比分析**

通过受控比较（仅改动一个参数）和阶段计时评估，稠密实现到spike‑gated 降低将推理时间从453.2 ms压缩至7.04 ms，进一步采用spike‑major 权重布局再降37.2%至4.42 ms；当L2缓存从4 MiB缩至1 MiB时，加速比从1.59×提升至1.88×。

**⚠️ 局限性**

局限性包括仅针对两层全连接网络、浮点32位权重、单输入向量实验，未覆盖批量推理、其他神经元模型或层类型，并且在输入宽度大、活跃脉冲少时扫描成本仍显高。

---

## 233. COMPASS: Cross-modal Observation with Multi-Path Adaptive Stopping and Selection for Early Time Series Classification

**arXiv ID:** 2611.27065 | [PDF](https://arxiv.org/pdf/2611.27065v1)

**作者:** Cheng-En Chung `[一作]` (National Yang Ming Chiao Tung University), Vincent S. Tseng `[通讯]` (National Yang Ming Chiao Tung University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了COMPASS框架，实现多模态早期时间序列分类时的联合停止与分支选择；

**💡 创新点**

将停止决策与最终预测分支选择融合为单一控制器，保留多模态分支并利用跨模态特征交换与门控融合；

**🔧 技术方法**

使用残差卷积提取局部特征、MMTM跨模态调节、门控注意力融合、Transformer时间编码、单控制器线性策略及神经进化优化；

**📊 数据集**

在六个生理/可穿戴数据集上评测：MIT-BIH PSG、Apnea-ECG、UCDDB、ISRUC-Sleep、WESAD、VTaC；

**📈 对比分析**

与五种ETSC基线（TEASER、ELECTS、SPN-V2、CALIMERA、UTSC）在HM-ACC、AUROC、早期性等指标上比较，COMPASS在平均排名中领先，且在多数数据集上显著提升准确率与早期性；

**⚠️ 局限性**

大部分决策集中在首步停止，导致分支选择效益受限；分支选择在不同数据集表现不一致，且在某些数据集可导致准确率下降；

---

## 234. Tracktail: Multi-View 3D Point Tracking without Depth Input

**arXiv ID:** 2611.26779 | [PDF](https://arxiv.org/pdf/2611.26779v1)

**作者:** Katie L Rupp `[一作]`, Lili Karashchuk `[通讯]` (HHMI Janelia Research Campus)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种名为Tracktail的多视角3D点跟踪器，能够在不使用深度输入的情况下跟踪任意3D查询点。

**💡 创新点**

核心创新包括：使用相机几何推导的场景尺度归一化、仅在查询层跨视角自注意力以及高效的Encoder‑Decoder架构，实现对不同尺度场景的鲁棒性。

**🔧 技术方法**

采用V‑JEPA 2.1 ViT‑L/16视频编码器、Transformer自注意力（跨视角、跨时间）与软argmax分类预测，并结合摄像机尺度归一化的输入特征。

**📊 数据集**

训练数据主要来自合成的Multi‑View Kubric、Multi‑View Point Odyssey（人类与动物）三大多视角数据集，测试集包括Panoptic Studio、DexYCB、Multi‑View Kubric以及新建的长序列Multi‑View Point Odyssey。

**📈 对比分析**

与MVTracker、ARROW、CoTracker3基线对比，Tracktail在Panoptic、DexYCB上取得新的最优结果，在Multi‑View Point Odyssey长序列保持低MTE和高生存率；推理速度约7 FPS，显存<4 GB。

**⚠️ 局限性**

目前不支持移动摄像机，点漂移仍有提升空间。

---

## 235. Right Numbers, Wrong Story: When LLMs Narrate Financial Changes Beyond Their Inputs

**arXiv ID:** 2611.26968 | [PDF](https://arxiv.org/pdf/2611.26968v1)

**作者:** Seunghan Lee `[一作]` (LG AI Research), Wonbin Ahn `[通讯]` (LG AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种框架检查插件，阻止LLM在财报草稿中添加未在输入中出现的判断和原因。

**💡 创新点**

创新点在于使用基于10-K的使用规范来自动识别并删除不支持的判断词和原因，且不需额外模型调用。

**🔧 技术方法**

技术包括检索相似规模的10-K句子、构建词语使用规范、规则编辑、无模型调用的后处理。

**📊 数据集**

数据集为220K份10-K的MD&A段落（38K家公司，1993‑2020），以及对17款开源LLM产生的约300万草稿。

**📈 对比分析**

与提示基线、检索基线和后置编辑器对比，插件将判断率从17.2%降至1.0%，不支持原因率从77%降至0.4%，与10-K作者的1.5%接近。

**⚠️ 局限性**

局限包括规范只反映美国10-K写作习惯，可能不符合读者解读；因果检查依赖固定连接词列表；评估仅覆盖英文、美国市场；假设单一变化；潜在训练数据泄露等。

---

## 236. LEAN: Library-Based Adaptation for Asynchronous, Federated Fine-Tuning

**arXiv ID:** 2611.26805 | [PDF](https://arxiv.org/pdf/2611.26805v1)

**作者:** Erdong Hu `[一作]` (Rice University), Chris Jermaine `[通讯]` (Rice University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于库的联邦 LoRA 微调方法 LEAN，完全避免平均操作，支持异步弱联邦环境。

**💡 创新点**

通过维护可洗牌的 rank‑1 基础对并让各客户端随机检出/回写，消除可识别性问题，提升对数据异质性和异步性的鲁棒性。

**🔧 技术方法**

采用 LoRA、低秩适配器、库检出/回写机制、可选的近端惩罚、异步通信，并在实验中与 FedAvg、FedProx、FLoRA 等方法对比。

**📊 数据集**

在 Oxford VGG Flowers、CUB‑200‑2011、CIFAR‑100 图像分类任务以及 ChemRxnPred 文本化学反应预测任务上进行评估。

**📈 对比分析**

与 FedAvg、FedProx、FLoRA 等平均式方法对比，LEAN 在大多数异质/异步场景下取得更高准确率、通信量更低；在重异质性下仍能达到 70‑95% 的精度，而平均式方法往往收敛失败。

**⚠️ 局限性**

理论分析仅覆盖同步全参与、无近端项的简化设置；异步分析与对抗鲁棒性未完整评估，且实验多为单随机种子。

---

## 237. Self-Supervised Learning with Recursive Quadrature Filters for Neural Time Series

**arXiv ID:** 2611.26856 | [PDF](https://arxiv.org/pdf/2611.26856v1)

**作者:** Mirko Morello `[一作]` (Telepath), David J. Heeger `[通讯]` (Telepath)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究开发了一种基于递归四边形滤波器（RQF）的自监督预训练框架，能够在无标签时间序列（如EEG、脑电图）上学习可解释的频率与带宽特征，并在多任务下通过冻结或微调实现高效下游迁移。

**💡 创新点**

创新点在于将RQF作为能学习调谐频率与带宽的复值递归滤波器，引入基于未来状态预测的自监督目标（窗口化与稠密预测），并在不依赖掩码或大规模参数的情况下实现紧凑、高效的模型。

**🔧 技术方法**

技术手段包括：RQF作为特殊的对角状态空间模型；InfoNCE 损失与对比学习；并行扫描算法用于高效训练；共享编码器对多通道输入进行独立编码；空间读出利用电极坐标实现多通道融合。

**📊 数据集**

使用四个基准数据集：TUAB（癫痫检测），BrainTreebank（句子起始检测），ISRUC‑SG1（睡眠分期），以及BCI‑IV‑2a（运动意象）以及跨语料的TUH EEG（TUEG）预训练。

**📈 对比分析**

通过与大型Transformer与基线模型的对照实验，110k参数的RQF模型在100秒TUAB段上实现0.9247 AUROC，超过SAMBA（1M参数）和LaBraM等模型；在其他基准也表现接近或优于大模型，且在标签稀缺或冻结编码器时预训练效果最显著。

**⚠️ 局限性**

局限性包括：每个语料仅训练单独的预训练权重，跨语料比较受限；与现有大模型的差异未完全分离，难以独立评估RQF本身的优势；以及未系统探讨词元化、掩码或规模对性能的独立影响。

---

## 238. Search-Informed MPPI for Kinodynamic Motion Planning via Monte Carlo Tree Expansion

**arXiv ID:** 2611.26763 | [PDF](https://arxiv.org/pdf/2611.26763v1)

**作者:** Ihab S. Mohamed `[一作]` (Indiana University), Lantao Liu `[通讯]` (Indiana University)

**通讯引用:** 2329 | [OpenAlex ID](https://openalex.org/A5101917996)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种结合蒙特卡洛树搜索（MCTS）与模型预测路径积分（MPPI）的搜索信息化MPPI（SI‑MPPI）框架，用于在大规模、拥挤的二维占用网格环境中实现动态可行、碰撞安全的运动规划。

**💡 创新点**

创新点在于：①将MPPI扩展与MCTS的扩展/模拟阶段融合，使用结构化的“诱饵”（baits）指导局部优化；②采用GPU批量MPPI并行生成候选轨迹；③通过适应性诱饵处理、可行性门控和top‑k选择控制树分支；④利用有效样本数（ESS）自适应更新MPPI的先验控制均值；⑤将候选成本回传至树，用以更新UCB统计。

**🔧 技术方法**

核心技术包括蒙特卡洛树搜索、模型预测路径积分控制、GPU并行采样、结构化诱饵生成、有效样本数（ESS）权重、top‑k采样与可行性门控、基于成本的回溯更新。

**📊 数据集**

在四个 100×100 的占用网格地图上进行评估，地图包括：随机障碍（Rand）、洞穴地图（Cave）、四月蝴蝶迷宫（4Moons）和月门迷宫（MGate）。

**📈 对比分析**

与三种基线（kinodynamic MCTS、LQR‑RRT*、IK‑RRT*）进行对比，使用成功率、规划时间、路径长度、角速度抖动、角加速度 RMS 以及树节点数等指标。SI‑MPPI 在所有地图上都取得与基线相当甚至更高的成功率，并在规划时间、树大小和轨迹平滑度方面显著优于基线，尤其在路径长度、角速度抖动和 RMS 上取得最小值。

**⚠️ 局限性**

局限性：目前仅针对已知静态地图；不保证得到最短路径；规划结果倾向于低速运动；对动态障碍、非平面动力学以及高维状态空间的适应性尚未验证。

---

## 239. Improved Accuracy Scaling with Problem Complexity in LLM Arithmetic Reasoning Using Executable Symbolic Representations

**arXiv ID:** 2611.27203 | [PDF](https://arxiv.org/pdf/2611.27203v1)

**作者:** Yu-Bin Liu `[一作]` (Capital Normal University), Shi-Ju Ran `[通讯]` (Capital Normal University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建GSM Arithmetic Benchmark，独立调控数值位数与运算次数，量化LLM算术推理在不同复杂度下的准确性变化。

**💡 创新点**

提出双斜率分析框架，并证明可执行符号表示（Lean、Python）在数字规模与运算次数轴上能减缓准确率下降，提升整体算术推理性能。

**🔧 技术方法**

使用可执行符号语言Lean和Python、Chain‑of‑Thought自然语言推理以及LoRA微调的Qwen3系列模型（0.6B、1.7B、8B），并在DeepSeek‑V4‑Flash、Qwen3.5‑27B等模型上做一次推理测试。

**📊 数据集**

采用自定义的GSM‑AB（2000题）、GSM8K、SVAMP、DeepMind Mathematics、Math401等数据集进行训练与评估。

**📈 对比分析**

对比CoT与Lean/Python后发现，Lean在大模型上平均提升30–50%，并在数字规模轴上斜率更平缓；Python与Lean在运算次数轴上保持高准确；在一次推理和OOD评估中，Lean对大模型也显示显著优势。

**⚠️ 局限性**

局限性在于：提升主要在大模型和特定轴上显现，0.6B模型及部分OOD任务未见优势；结果高度依赖模型、表示与数据集，缺乏置信区间与更广泛任务验证。

---

## 240. Reinforcement Learning Unlearns Corrupted SFT Reasoning

**arXiv ID:** 2611.26756 | [PDF](https://arxiv.org/pdf/2611.26756v1)

**作者:** Anika Misra `[一作]` (University of Michigan), Huteng Dai `[通讯]` (University of Michigan)

**通讯引用:** 25 | [OpenAlex ID](https://openalex.org/A5075205422)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在数学推理数据集GSM8K上系统地引入六种真实错误类型，对训练轨迹进行不同程度的腐败，并在OLMo‑2‑1B模型上先进行监督微调（SFT），随后使用Group Relative Policy Optimization（GRPO）进行强化学习后训练，评估SFT与GRPO对模型推理准确率的影响。

**💡 创新点**

证明强化学习对SFT质量几乎不敏感，SFT只需提供“形式”（trace结构），RL则学习“功能”（推理逻辑），实现SFT与RL角色解耦；并显示极少量的“form‑only”三行轨迹即可在GRPO后实现高于完整SFT+GRPO的准确率。

**🔧 技术方法**

采用正则表达式（regex）实现错误注入与传播，构造六类错误（计算、幻觉、运算符、操作数、额外步骤、遗忘步骤）；使用GRPO进行RL后训练；通过Pearson相关系数分析SFT与GRPO性能的关系。

**📊 数据集**

GSM8K数学推理数据集；OLMo‑2‑1B基础模型；自制的三行“form‑only”轨迹；生成的六类错误轨迹。

**📈 对比分析**

比较方法：对361种不同腐败配置（10%–100%腐败、不同错误位置、答案/轨迹正确性组合）分别进行SFT‑only、SFT+GRPO、GRPO‑only和form‑only初始化；评估指标为pass@2、pass@k；结果显示SFT‑only在高腐败下准确率从43.61%降至13.80%，但GRPO后训练可恢复至≈60%；仅8个form‑only样本的SFT+GRPO可达69.30%，高于金标准60.71%；SFT准确率与GRPO后准确率相关性接近零。

**⚠️ 局限性**

实验仅在OLMo‑2‑1B与GSM8K上验证，未测试更大模型或更难的数据集；错误分类覆盖93.51%，未覆盖全部可能错误；仅考虑数学推理，未探讨自然语言或其他推理任务。

---

## 241. LongCat Recall Sparse Attention: Co-Designing Key Organization and Retrieval for Long-Context Inference

**arXiv ID:** 2611.27233 | [PDF](https://arxiv.org/pdf/2611.27233v1)

**作者:** Hao Zhang `[一作]` (Meituan LongCat Team), Kefeng Zhang `[通讯]` (Meituan LongCat Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种新的稀疏注意力机制 LRSA，通过时间桶、方向性分组和固定容量页面的层级检索，结合增量索引维护实现长上下文高效推理。

**💡 创新点**

创新点包括：1）在每个时间桶内使用容量平衡的方向性分组并将完整分组打包到页面，实现代表性与访问粒度分离；2）采用页面级检索先选取历史子集，再在精细化阶段与固定候选一起做 Top‑K；3）通过递归分组与组大小自适应融合的核实现，显著减少构造阶段的 kernel 调用；4）共享固定候选存储以避免在预填充块中重复拷贝；5）在 SGLang 中异步构建索引、前缀重用和资源保留策略，降低索引维护成本。

**🔧 技术方法**

技术手段包括：方向性余弦相似度分组、FP8 量化 + FP32 缩放、递归分组与组内排序、页面级存储与位置映射、核融合（group‑size‑aware）、共享候选块、异步 GPU 任务调度、前缀匹配重用与索引保留策略。

**📊 数据集**

使用 Qwen3‑4B‑Instruct‑2507 与 Hunyuan‑7B‑Instruct‑0124 作为模型，在 NarrativeQA、MuSiQue、HotpotQA、TriviaQA、TREC、2WikiMQA、DuReader、LSHT、QMSum、Qasper 等十个下游任务上进行评估。

**📈 对比分析**

与全注意力和 DeepSeek Sparse Attention (DSA) 对比：LRSA 在同一 2048 选取预算下，平均任务分数仅落后 0.08 分；在 1024K 上，键选择速度比 DSA 高 4.70×，整体稀疏推理速度比全注意力高 35.6×，与 DSA 的端到端加速为 1.86–1.98×；在 64K‑512K 之间同样保持 1.1–1.7× 的加速。

**⚠️ 局限性**

局限性包括：1）时间桶容量与分组粒度需要经验调优，过大或过小都会影响召回率与速度；2）索引构造与维护的 GPU 资源仍占一定比例，尤其在极短上下文或高频更新场景下效果不明显；3）当前实现采用两级检索，进一步深层次层级可能增加核开销，需更高效的多级融合设计；4）对前缀重用与保留策略的依赖，若 KV 前缀变化频繁会导致重建成本提升；5）主要在 FP8 量化环境下评估，其他量化或浮点设置的性能尚未充分验证。

---

## 242. Recurrent Amplification and Exact Nonexpansiveness Criteria for Query-Driven Erasure

**arXiv ID:** 2611.27109 | [PDF](https://arxiv.org/pdf/2611.27109v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 243. Know Your Context Bank: Repairing Inconsistency and Measuring Generalization in Context-Aware Text-to-SQL

**arXiv ID:** 2611.26775 | [PDF](https://arxiv.org/pdf/2611.26775v1)

**作者:** Hailong Li `[一作]` (Google), Yannis Papakonstantinou `[通讯]` (Google)

**通讯引用:** 13822 | [OpenAlex ID](https://openalex.org/A5059167302)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种基于语义单元的两两比较方法（ACN profile），用来衡量和修复文本到SQL（Text‑to‑SQL）系统的上下文库（context bank）中的不一致性（context bank rot），并设计了基于该方法的清洗协议和Context‑Aware Testing（CAT）框架，以评估系统对上下文库的泛化能力。

**💡 创新点**

创新点包括：
1) 将SQL决策拆分为语义单元并构造对齐‑冲突‑新颖（ACN）三元评分，能够定量比较两条SQL是否在相同语义下一致、冲突或不相关；
2) 通过LLM判定器对ACN评分进行人类校准并蒸馏成高效本地评分器，显著降低计算成本；
3) 双轨清洗协议（Track A：单条审计；Track B：基于规则的冲突对消）实现了对库的系统化、可追溯的清洗；
4) CAT流水线将ACN距离作为“控制距离”生成测试样本，构建可量化的上下文泛化基准。

**🔧 技术方法**

技术方法包括：
- 语义单元抽取（基于SQL语法树+LLM定位问题短语）；
- ACN profile计算（对齐、冲突、创新比例）；
- LLM判定器（Claude‑Sonnet‑5 + Gemini‑3.5‑Flash）以及基于其的蒸馏评分器；
- 两轨清洗协议与十条约定规则；
- CAT基准构建（查询重写、距离门控、质量检查）；
- 在检索式和微调式 Text‑to‑SQL 系统上评估（Gemini‑3.8‑Flash、Claude‑Sonnet‑5、Qwen2.5‑7B、Gemma‑4‑12B‑it）。

**📊 数据集**

使用的数据集包括：
- BIRD（训练、验证、开发分布，SQLite和PostgreSQL dialects）；
- Spider（train子集、dev子集）；
- LiveSQLBench（训练子集）；
- 另外还使用了从这些数据集生成的对齐‑冲突对、重写查询以及独立专家审计样本。

**📈 对比分析**

比较方法与性能：
- 清洗后训练集（BHI ≈ 86.8）相比随机集和保留错误集在 BIRD‑dev 上每个训练轮次提升约 2.6 点；
- 在一致性对照实验中，提供与目标一致的上下文示例使 Gemini‑3.8‑Flash 的准确率提升 13.3 点；
- CAT 基准显示检索效果随距离递减：Gemini 在 d₀ 提升 22.2 点，d₁ 12.0，d₂ 7.5，d₃ ≈ 3；
- 微调模型的提升与检索类似，Qwen 在 d₀ 提升 38.0 点，d₁ 15.5，d₂ 9.0，d₃ 1.0；
- 对于 Spider，检索收益在距离增加时更快衰减；微调在远距离上可能低于无库对照。

**⚠️ 局限性**

限制与潜在偏差：
- 评分器主要在 SQLite 上训练，Dialect 通用性需重新校准；
- 许多判断与清洗基于 LLM 判定器，可能引入主观性；
- CAT 的 no‑context 基线受样本选取影响，可能掩盖真实性能；
- 生成的测试查询和规则库依赖于模型本身，存在循环偏好；
- 实验仅在单一随机种子下进行，未充分评估随机波动；
- 该方法假设上下文库足够大且覆盖面广，对小型或高度专业化库的适用性待验证。

---

## 244. Dynamic Token Merging for Efficient Vision-Language Models

**arXiv ID:** 2611.27064 | [PDF](https://arxiv.org/pdf/2611.27064v1)

**作者:** Chris Gu `[一作]` (Stanford University), Julie Kallini `[通讯]` (Stanford University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种通用的动态令牌合并（Dynamic Token Merging, DTM）框架，并将其从字节级文本模型扩展到视觉语言模型（T5Gemma 2 和 Gemma 3），通过在视觉 token 上学习删除门实现推理效率提升。

**💡 创新点**

创新点包括：①将 DTM 从字节级文本迁移为通用的 token 删除框架，能适用于视觉 token 与 decoder‑only 架构；②采用基于隐藏状态的学习式删除门，并在训练阶段通过软 attention bias、推理阶段通过硬删除实现“软到硬”转换；③利用 PID 控制器动态调节删除率，并在 LoRA 微调下保持模型性能。

**🔧 技术方法**

技术细节包括：学习式 delete gate + 软 attention bias + 硬 token 删除；PID 控制器（P+I+D）用于精准控制删除比例；LoRA 微调实现参数高效调整；视觉 token 通过 SigLIP 视觉编码器投影到语言模型；在 decoder‑only 模型中在 prefill 阶段执行删除。

**📊 数据集**

实验使用十个基准数据集：DocVQA、ChartQA、InfoVQA、TextVQA、VQAv2、VizWiz、GQA、TallyQA、POPE 与 RefCOCO+，涵盖文档阅读、图表理解、场景文本、通用 VQA、计数、幻觉检测与定位等多样任务。

**📈 对比分析**

与随机删除、基于注意力的选择器（如 FEATHER、LearnPruner、LLaVolta）进行对比。结果显示，在 80% 视觉 token 删除时，DTM 在 DocVQA、ChartQA、TextVQA 等细粒度视觉任务上平均提升约 3.8 分，单项最高提升 8 分；在大多数任务保持 90% 以上原始性能，并在不同预算（20%–80%）下表现稳定，单一 checkpoint 可在更高删率下保持优势。

**⚠️ 局限性**

局限性：在轻度删除或通用 VQA 任务上收益有限；在 InfoVQA 等部分基准无法达到 90% 原始性能；缺少空间注释时难以评估 evidence 保留；未在视频、音频或多图输入等更复杂模态中验证；删除门仅基于隐藏状态，可能无法捕捉更复杂上下文依赖。

---

## 245. Reward Integrity in Adversarial Reinforcement Learning for LLM Guardrail Red-Teaming

**arXiv ID:** 2611.26998 | [PDF](https://arxiv.org/pdf/2611.26998v1)

**作者:** Chengming Liu `[一作]` (Commonwealth Bank of Australia), Ruth Oliver `[通讯]` (Commonwealth Bank of Australia)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究了金融服务语言模型红队的对抗强化学习循环中奖励信号的可信度，并提出了将目标交互与奖励评分解耦的RL循环以及针对PII要求和政策违例的两种奖励完整性检查。

**💡 创新点**

创新点在于拆分目标交互与奖励评分的RL流程，和专门针对PII验证与政策违例归因的奖励完整性机制。

**🔧 技术方法**

采用群组相对REINFORCE（GRPO风格）与LoRA适配、TypeSafe Jev判别模型、预训练与后训练的Laya、CLM-8B等技术进行训练与评估。

**📊 数据集**

使用内部Opus 4.8日志、公开的AdvPIIBench衍生集以及274例作者审核的误归属挑战集进行验证。

**📈 对比分析**

通过与Opus标签的对齐率、AUC、误归属拒绝率等指标比较，后训练Laya的AUC从0.596提升到0.971，Jev在误归属集的拒绝率约为50%，整体显著提升奖励信号的可靠性。

**⚠️ 局限性**

局限性包括奖励模型仅基于单一金融服务数据集，误归属挑战集非代表性、仅单人审核、未评估CLM误归属拒绝，以及未验证奖励完整性对在线学习效果的实际影响。

---

## 246. Rethinking Full-Weight LoRA with Domain and Task Adapter Scaling for Question Answering over Portuguese Well Drilling Reports

**arXiv ID:** 2611.26833 | [PDF](https://arxiv.org/pdf/2611.26833v1)

**作者:** Luan B. Guerra `[一作]` (São Paulo State University), Lucas C. Ribas `[通讯]` (São Paulo State University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究LoRA适配器在葡萄牙钻井报告问答任务中的缩放效果，比较直接适配、基于基础模型转移以及适配器融合。

**💡 创新点**

提出在DAPT和QA任务中对LoRA适配器进行后训练缩放，并系统评估不同权重的组合对性能的影响。

**🔧 技术方法**

使用LoRA参数高效微调、Shadow‑FT基准的基础→指令迁移、BERTScore评估以及线性加权适配器融合。

**📊 数据集**

使用私有约17,000条钻井报告进行DAPT，公开的WellQuest问答数据集进行QA微调。

**📈 对比分析**

在不同训练阶段和缩放因子下用BERTScore评估，发现对DAPT最佳缩放≈0.3‑0.4、QA最佳≈0.7‑0.8，fractional scaling显著提升F1（Qwen2.5‑7B从68.8%提升到71.9%，Llama‑3.1‑8B从63.2%提升到68.5%）。

**⚠️ 局限性**

仅在单一语言单一任务、仅用BERTScore评估语义相似性，未检验事实准确性，适配器融合可能产生交互干扰。

---

## 247. WePM: Hierarchical Conflict Reconciliation for Unified MLLMs

**arXiv ID:** 2611.27365 | [PDF](https://arxiv.org/pdf/2611.27365v1)

**作者:** Zhiqiang Yuan `[一作]` (WeChat AI, Tencent), Jinchao Zhang `[通讯]` (WeChat AI, Tencent)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Heterogeneous Capability Density（HCD）衡量模型在单一内存预算下的多功能能力，并设计WePM模型实现单一视觉–语言主干下的表示、生成和决策三大类能力的统一；

**💡 创新点**

1) 引入HCD这一系统级指标；2) 通过多粒度视觉预算(MGVB)、统一列表知识蒸馏(ULKD)以及稳定高效多任务优化(SEMO)三项技术，实现跨任务视觉证据分配、软跨任务监督与梯度尺度与频率同步；3) 将三大能力在同一模型中以不同输出路径实现，显著降低资源占用；

**🔧 技术方法**

多粒度视觉预算（token分配策略）、统一列表知识蒸馏（软标签与列表分布匹配）、稳定高效多任务优化（固定权重+顺序梯度累积）、基于Qwen3.5‑4B/2B的Transformer主干；

**📊 数据集**

多种检索、生成与决策基准（MSCOCO、CIRR、VisDial、ArgoAna、FiQA-2018、MathVista、MM‑Vet、AI2D、OCRBench、ChartQA、TextVQA、JevBench等），以及构造的专用检索与生成数据集；

**📈 对比分析**

与现有统一方法（CAFe、FROMAGe、MMGen、ULME）在检索‑生成两类共享能力上对比，WePM在五个评测组平均得分最高，Joint得分从38.15提升至52.74；单任务评测中WePM‑4B在30个嵌入、39个生成、7个决策任务上均排名第一；与多模型堆叠方案相比，WePM在61.9%内存压缩下实现了2.64倍HCD提升；

**⚠️ 局限性**

在agentic/工具使用任务上性能略逊于原始Qwen基准，SEMO采用静态任务权重和曝光计划，未能动态适应训练过程变化；此外，模型在处理开放式对话和多步推理时仍有待提升。

---

## 248. Budgeted Safety Verification for Streaming Video Generation

**arXiv ID:** 2611.27268 | [PDF](https://arxiv.org/pdf/2611.27268v1)

**作者:** Yuxin Cao `[一作]` (National University of Singapore), Jin Song Dong `[通讯]` (National University of Singapore)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种流式视频安全防护，利用生成器内部的隐藏状态做轻量级评分，决定何时调用昂贵的像素级视频语言模型验证器，从而在保持低延迟的同时降低观众暴露的有害帧数。

**💡 创新点**

创新点在于：① 引入“曝光”（exposure）度量，专门针对即时流式视频的安全评估；② 通过在生成器内部实现双层决策管线，将判断权重分配给轻量级头部和高精度验证器；③ 使用分布无关的 conformal 风险控制校准阈值，使可接受的误报率在实际部署中可预期。

**🔧 技术方法**

主要技术包括：自回归视频生成器（Self‑Forcing、LongLive 等）中的中间隐藏状态读取；小型时序 Transformer 对隐藏状态做排序评分；视频语言模型作为高精度验证器；基于滑动窗口的累计检验和记忆无关阈值；以及多种基准检测器（图像版、视频版、prompt 过滤器）做对比。

**📊 数据集**

使用了 VBench、VidProM、Movie Gen、T2VSafetyBench 等公开基准数据集，共 599 条评估流（含 1597 条原始提示），并进一步构造了 734 条未显式命名危害的提示、360 条中途切换提示以及多条人类标注对照集。

**📈 对比分析**

在 599 条评估流上，未加防护的生成器暴露 46.25 有害帧；prompt 过滤器可在 100% 案例中实现 0 帧曝光；单块视频模型每块 80.1 帧的成本下仅能降低到 0.91 帧；而本论文的预算化验证策略（约 5.3 次验证调用）将曝光降至 2.35 帧，随机分配同样成本下仍为 37.20 帧；此外，随机阈值下误报率低于 5%，并且在更长的流（21 块→63 块）中收益随长度成正比。整体表现显示，该方法在保持低延迟和可解释阈值的同时，显著提升了流式视频的安全性。

**⚠️ 局限性**

局限性包括：1) 评估数据集的危险提示多为显式命名，文本过滤器已能大部分拦截；2) 采用的视频语言模型及其 oracle 对“顺序危害”仍不够精准；3) 实验仅覆盖单一 backbone 家族，跨模型迁移仍需验证；4) 评估暴露度以人工一致性为基准时样本有限；5) 防护系统需要额外的计算资源，尤其是高精度验证器的调用成本。

---

## 249. Which Data-Center Load Models Can Be Trusted? An Evidence-Graded Review of Dynamic Modeling and Validation for Large Computational Loads

**arXiv ID:** 2611.27169 | [PDF](https://arxiv.org/pdf/2611.27169v1)

**作者:** Dileep Cheerla `[一作]` `[通讯]` (independent researcher), Dileep Cheerla (independent researcher)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过对160篇文献进行系统回顾和编码，评估了数据中心动态负荷模型的验证证据与行业规则的匹配度，并提出了基于证据分级的上下文使用门框架和最小验证协议。

**💡 创新点**

创新点在于将证据分级与上下文使用门（G0–G7）相结合，对模型可信度进行分层评估，揭示当前缺乏持久验证的空白，并给出具体的研究议程与实践准则。

**🔧 技术方法**

采用文献检索、手工编码、证据矩阵构建与门框分析等技术，对模型-现象组合进行归类，并与现场事件记录和仿真结果进行对比。

**📊 数据集**

主要使用公开的现场事件记录（如 Dominion、ERCOT、EirGrid 等）以及标准模型参数和实验室/仿真数据，未使用任何专有数据集。

**📈 对比分析**

比较方法通过将模型输出与现场测量（T1/T2边界）进行误差评估，结果显示标准库模型默认参数与实际事件不匹配，校准后仍缺乏独立验证，整体性能不足。

**⚠️ 局限性**

局限性包括仅覆盖公开文献、编码由单人完成、规则和事件频繁更新、缺乏私有对比数据，导致结论为“目前无公开成功验证”。

---

## 250. Writing Weights or Training Them? Spatial Reasoning through Weight-Space Programming and Low-Rank Adaptation

**arXiv ID:** 2611.27089 | [PDF](https://arxiv.org/pdf/2611.27089v1)

**作者:** Yalin Yang `[一作]` (West Virginia University), Ryan Zhenqi Zhou `[通讯]` (Yale University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Qwen3.5‑0.8B模型的空间推理任务（参考框架、StepGame以及两类距离比较任务）进行了闭式权重编辑与梯度训练（LoRA）的对比实验；

**💡 创新点**

首次系统评估了闭式编辑和参数高效训练的权衡：闭式编辑在目标任务上提升有限但保持邻近技能；梯度训练在目标任务上几乎达成完美，却导致严重的干扰；并指出编辑效果取决于权重改变的结构而非幅度。

**🔧 技术方法**

使用了权重空间编程（写入特定权重差值）、低秩适配器（LoRA）训练（rank 1/16）、基于工作解的scratchpad监督、交叉熵损失、随机种子、核样本和线性探针等技术。

**📊 数据集**

基于Qwen3.5‑0.8B预训练模型，使用的任务数据来自公开的参考框架（128条）、StepGame（200条）以及地图/产品两类距离比较（各128条），训练时还利用了从生成器抽取的2,048个参考框架场景和512个StepGame问题。

**📈 对比分析**

通过统一解码设置、三种随机种子和“提升10%点”阈值进行比较；闭式编辑将参考框架任务准确率从0%提升至22.1%，StepGame从19.2%提升至33.7%；LoRA rank 1在两任务上均达100%/98.8%，但在距离任务上准确率骤降至0–2%；编辑保持距离任务性能，训练则导致显著干扰；在不同参数量和训练起点下，LoRA表现稳定。

**⚠️ 局限性**

实验仅覆盖单一0.8B模型，编辑与训练的计算预算不等；对非空间任务的影响未系统探索；闭式编辑的手段有限，无法实现更大幅度改动；梯度训练导致的任务迁移和干扰尚未得到控制；未来需结合更多模型、任务与更大规模数据验证结论。

---

## 251. RoboColiseum: Co-evolution of Simulation, Evaluation, and Models

**arXiv ID:** 2611.27042 | [PDF](https://arxiv.org/pdf/2611.27042v1)

**作者:** Chenghao Yin `[一作]`, Qian Wang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了RoboColiseum平台，集成机器人系统识别、可组合物理与渲染接口，并提供基于实机‑仿真对比的八轨道交互基准；同时设计了Season I与Season II多维任务集及进度‑预算计分机制。

**💡 创新点**

创新点包括：①可插拔物理渲染协议，兼容PhysX、Newton、libuipc和MuJoCo等多引擎实现刚柔体交互；②实机‑仿真双向对比（Real2Sim）验证模型真实性；③细粒度预算与进度计分，避免单纯成功率评估；④在同一平台上同时评估VLA、WAM与Agent系统，支持跨模型对比。

**🔧 技术方法**

使用技术包括：机器人动力学识别与参数估计、增量潜在接触（IPC）软体建模、CPU‑GPU分离式控制/渲染协议、Agent2Act推理执行框架、视觉+语言指令解析、分阶段预算与奖励设计、Python/C++跨平台接口、强化学习与监督微调。

**📊 数据集**

数据集主要来自RoboColiseum任务库，包含22个机器人‑任务配置、255个有效episode；任务覆盖Instruction Following、Spatial Reasoning、General Manipulation、Robustness、Dexterous Manipulation、Whole‑Body Coordination、Precision Manipulation和Cross‑Embodiment等八个维度。

**📈 对比分析**

比较方法：采用同一任务的实机与仿真得分计算Pearson相关系数（r≈0.97），并在公开排行榜上给出0‑100分。Season I榜单中最高总体得分为Prime‑0 77.54分；Season II Agent最高Overall 18.30分，VLA最高Overall 11.22分，显示不同模型在各维度的相对优势。

**⚠️ 局限性**

局限性：Real2Sim仅覆盖有限机器人‑任务组合；未公开完整参数、分数细节和置信区间；机器人‑IPC耦合为单向，缺乏接触力反馈；缺乏跨场景独立验证；排行榜缺少完整案例计数，影响可复现性和公平对比。

---

## 252. PhysForce: Physics-aware Force Control for Interactive Video World Model

**arXiv ID:** 2611.27164 | [PDF](https://arxiv.org/pdf/2611.27164v1)

**作者:** Xuancheng Xu `[一作]` (Nanjing University of Posts and Telecommunications), Bing-Kun Bao `[通讯]` (Hefei University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 PhysForce，一种两阶段的交互视频世界模型框架，用于实现基于物理的力控制。

**💡 创新点**

创新点在于引入了不泄露轨迹的 LFIE 编码、通过 PAN 对力诱发的状态变化进行空间定位，并通过 PARL 与动态门实现对无标注 OOD 视频的物理一致性强化。

**🔧 技术方法**

主要技术包括基于 DiT 的 Diffusion Transformer、VAE 嵌入、Physics Attribution Network、DiffusionNFT 强化学习和自监督 V-JEPA 预测奖励。

**📊 数据集**

使用的数据集包括 GoalForce、ForcePrompting、WISA-80K 三个力标注或无标注视频集。

**📈 对比分析**

与 Wan-Move、ForcePrompting、GoalForce 等 SOTA 方法在 OOD/ID 基准上比较，PhysForce 在力对齐、物理一致性、视频质量等指标均显著优于基线，表现最突出。

**⚠️ 局限性**

局限在于缺乏显式三维建模、对长期生成的误差积累和动态场景的实时推理仍需改进。

---

## 253. Isolating the Language Shell: A Content-Controlled Study of Multilingual Reasoning

**arXiv ID:** 2611.27053 | [PDF](https://arxiv.org/pdf/2611.27053v1)

**作者:** Hoejoon Kwon `[一作]` (Chung-Ang University), YoungBin Kim `[通讯]` (Chung-Ang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构造内容受控的语言“壳” (English、Translated、Mixed)，并在三种目标语言和两种难度级别上微调两款预训练模型，评估不同壳对下游多语言推理准确率与生成语言的影响。

**💡 创新点**

首次将同一条验证过的英语推理轨迹以三种语言表现形式统一渲染，彻底控制推理内容不变，仅改变响应端的语言形式，从而揭示语言壳对学习、兼容性和语言继承的独立作用。

**🔧 技术方法**

使用预训练 LLM 作为教师生成轨迹 (Qwen3-32B)，用 Gemini-2.5-Flash 进行翻译与混合渲染；随后对 Qwen3-4B-Base 与 Gemma-3-4B-PT 进行全参数微调；评估指标包括预训练兼容性 (perplexity)、下游准确率 (PolyMath Low/Med) 与生成语言比例。

**📊 数据集**

基于 GSM8K 训练集生成英语轨迹，并通过 Gemini-2.5-Flash 将对应问题翻译为韩语、中文、德语；在此基础上构造约 7.2K 条共享项目的三种壳数据集。

**📈 对比分析**

对比兼容性、学习效果和语言继承：兼容性排名一致但不预测准确率；学习效果受学生、语言和难度影响，Mixed 在某些设置下能击败 Translated；生成语言比例与准确率往往不一致。整体上，最佳壳取决于目标任务与部署需求。

**⚠️ 局限性**

受控实验范围有限（仅两款模型、三种语言、两难度级别），数据与评测依赖专有模型，统计显著性受种子数限制，结果可能不具普遍性，且未涵盖完整的原生目标语言推理。

---

## 254. JevRisk: Zero-Shot Financial Risk Assessment with a Language Decision Model

**arXiv ID:** 2611.26999 | [PDF](https://arxiv.org/pdf/2611.26999v1)

**作者:** Yaodong Yang `[一作]`, Furui Liu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出JeVRisk系统，先用代码按监管指引计算风险指标，再用语言决策模型在无标签情况下对客户进行上下文化风险评估。

**💡 创新点**

将监管红旗和早期预警指标编程实现，利用语言模型直接判断风险并将职业风险与客户风险概率相乘形成最终评分，支持零标签、跨风险类型的通用评估。

**🔧 技术方法**

编写指标计算代码；使用大语言模型（Qwen3.5‑9B、Clef‑Flash、Jev‑Omni）作为决策器；评估指标包括AUC、Precision/Recall；对比基线包括规则计数、Isolation Forest、GBDT、Logistic Regression、TabPFN。

**📊 数据集**

合成AML数据（Scotiabank）、真实消费贷申请数据（Home Credit）以及真实P2P贷款数据（Lending Club），均包含KYC信息、交易/信用历史与标签。

**📈 对比分析**

与无标签基线（指标计数、Isolation Forest）比较，JeVRisk在AML上AUC 0.923，Home Credit 0.628，Lending Club 0.651；与标注模型比较，在约1000个标签时JeVRisk与监督模型相当或更优，标签数增至数千时监督模型略优。

**⚠️ 局限性**

AML数据为合成，未在真实洗钱案例上验证；需专家设计指标与提问；语言模型决策缺乏可解释性；在大量标签时监督模型表现更好。

---

## 255. Beyond What They Say: Cue Acknowledgment and Robustness in Vision-Language Models

**arXiv ID:** 2611.27389 | [PDF](https://arxiv.org/pdf/2611.27389v1)

**作者:** Ali Rahimiakbr `[一作]` (Sharif University of Technology), Mahdieh Soleymani Baghshah `[通讯]` (Sharif University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉语言模型在视觉和文本误导性提示下的解释与答案行为，比较不同提示条件下的错误率、静默失败率以及提示承认情况。

**💡 创新点**

提出了控制对比方法，揭示提示承认与抵抗性之间的非因果关系，并通过多种提示干预、解释提问与Patchscopes读取实验揭示模型对提示的隐藏跟随行为。

**🔧 技术方法**

采用自然Chain-of-Thought（CoT）与Evidence-first提示收集模型解释，使用LLM自动标注提示承认，结合配对统计、置信区间和Odds Ratio进行行为量化。

**📊 数据集**

使用合成多边形颜色任务、COCO recoloring任务以及BLINK艺术风格/视觉相似性任务，并在部分API运行中采集数据。

**📈 对比分析**

通过匹配评估协议比较清洁输入与单/联合/冲突提示条件，计算Targeted Error Rate、Acknowledgment Rate和Silent Failure Rate；结果显示添加误导文本可提升错误率但显著降低静默失败率，效果因模型与任务而异，未显著提升整体鲁棒性。

**⚠️ 局限性**

实验受限于模型规模、提示强度与内容不可分离、缺乏重复采样与独立stance标签、标注样本有限，结果为探索性，未能证明承认的因果作用。

---

## 256. AstronautBench: Measuring Agent Self-Improvement Across Episodes Through Knowledge, Evidence, and Progress

**arXiv ID:** 2611.26910 | [PDF](https://arxiv.org/pdf/2611.26910v1)

**作者:** Congyuan Yu `[一作]` (Carnegie Mellon University), Runping Tian `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在嵌入式游戏 Outer Wilds 上，设计了一个多视角自我提升基准，结合任务进展、知识问答与证据归因三种度量。

**💡 创新点**

创新点在于将同一游戏运行的进度、问答正确性与答案可归因证据统一评估，并提出基于代理自身记录的证据归因 QA 协议。

**🔧 技术方法**

使用的技术包括 LLM 规划器（如 GPT-6.1、Gemini-4）、实体中心知识库、基于学习的控制器以及 LLM 判定器进行答案正确性和归因判定。

**📊 数据集**

数据集为 Outer Wilds 游戏日志、164 题单与对应的游戏知识（从游戏 Wiki 手工编写）。

**📈 对比分析**

通过在 8 轮实验中对比含知识库与不含知识库的两种运行，基准显示知识正确定增速快于证据归因速率，且进度与问答结果不完全同步，表明单一视角评估不足。

**⚠️ 局限性**

局限性包括仅单一实验运行、控制器可靠性不足、LLM 判定器缺乏人类基准、问题分布不均衡以及对证据种类和推理深度的限制。

---

## 257. Tripwire: Robot Agents that Program Their Own Interrupts under Human Disturbance

**arXiv ID:** 2611.27182 | [PDF](https://arxiv.org/pdf/2611.27182v1)

**作者:** Sunli Chen `[一作]` (University of Massachusetts at Amherst), Chuang Gan `[通讯]` (University of Massachusetts at Amherst)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为TRIP的事件驱动、训练无关的机器人操控框架，允许代理程序在预训练策略执行期间自行编写并更新监视器（tripwire），实时检测并中断人类干扰；

**💡 创新点**

创新点在于把中断逻辑交给代理自己编写并在运行时动态更新，突破传统轮询或固定时长调用的局限；通过轻量级Python程序结合紧凑VLM实现对人类位置与运动的即时感知，兼容任意预训练策略，无需微调；

**🔧 技术方法**

使用的技术包括：预训练的Vision‑Language‑Action（VLA）模型和World Action Models（WAM）；紧凑冻结的视觉语言模型（如Qwen3.5‑9B、GPT‑6）；RGB‑D传感器、摄像头；Python sandboxed tripwire、agent tool‑calling（GPT‑6.1‑sol/Astra）；

**📊 数据集**

采用RoboHD benchmark（基于RoboCasa365并加入可响应的人类干扰）以及两项实机任务（挂杯、挂毛巾）进行评估；

**📈 对比分析**

与预训练策略单独执行及传统基于轮询的agent harness baseline进行对比；在RoboHD上，TRIP将安全率从61.3%提升至81.3%，成功率亦显著提升；在实机实验中，TRIP在有人干扰下的成功率接近无干扰情形，优于baseline和策略单独；

**⚠️ 局限性**

局限性包括：干扰人类为仿真avatar，缺乏真实人类交互；物理仿真在agent与VLM计算时暂停，导致延迟不可测；高API成本限制了模型和实验规模；未对真实碰撞体进行仿真；

---

## 258. Distributional Algebro-Geometric Persistence (dAGP): Wasserstein-Stable Polynomial-Relation Signatures

**arXiv ID:** 2611.26986 | [PDF](https://arxiv.org/pdf/2611.26986v1)

**作者:** Edisy Kin Wai Chan `[一作]` (University of Southampton), Andersen Ang `[通讯]` (University of Southampton)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文研究并实现了分布式代数几何持久性（dAGP）表示，基于多项式矩阵求解近似零多项式，给出了度–容忍度签名，可用于刻画点云分布的低度代数结构。

**💡 创新点**

创新点在于将近似零理想的谱阈值计数与持久性思想结合，提出二参数（度、容忍度）计数签名；给出在Wasserstein距离下的Lipschitz稳定性，并证明阈值计数在谱间隙内保持不变；同时提供从样本到分布的连续映射，构成分布层次的持久性框架。

**🔧 技术方法**

技术上使用正交Legendre多项式特征构造矩阵，进行矩阵特征值分析；利用Wasserstein距离和Weyl矩阵扰动理论证明稳定性；采用Christoffel–Darboux相关概念；通过数值实验验证理论。

**📊 数据集**

所有实验均基于合成二维数据集，包括圆、四次超椭圆、指数振荡曲线（非代数曲线）以及均匀正方形，加入不同尺度的高斯噪声，进行多次重复实验。

**📈 对比分析**

与点云AGP、最小特征值计数、全阶谱以及基于距离的分类器等基线进行对比；在多形状诊断任务中，dAGP 的阈值计数网格实现了近乎100% 的区分；在样本大小与噪声实验中，dAGP 与 λ_min 等方法表现相近，显示签名具有较高的辨别能力。

**⚠️ 局限性**

主要局限包括：仅适用于低维低度，多项式特征维度指数增长限制了可扩展性；不同分布可能共享相同签名，导致非唯一性；对坐标变换（如中心化、whitening）敏感；阈值校准需经验估计且依赖已知噪声模型；缺乏对数据预处理的稳定性理论。

---

## 259. Learning When to Ask: Advisor-Aware Post-Training for SLM Agents

**arXiv ID:** 2611.27443 | [PDF](https://arxiv.org/pdf/2611.27443v1)

**作者:** Aasheesh Singh `[一作]` (Quantiphi), Vishal Vaddina `[通讯]` (Quantiphi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过 ADAPT 后训练方法，使小型语言模型学会在与更强模型协作时何时请求帮助以及如何执行建议，从而提升多步推理与工具使用能力。

**💡 创新点**

创新点在于三阶段后训练流程（SFT+DPO+GRPO），无需人工标注或奖励模型即可让模型自我学习提问时机和利用建议，并通过成本约束实现高效提问。

**🔧 技术方法**

使用了监督微调、对抗偏好优化（DPO）、成本约束强化学习（GRPO）、LoRA 适配、工具调用框架与对抗式样本挖掘等技术。

**📊 数据集**

实验基准为 TextWorld（长序列规划）、LiveCodeBench（代码生成）和 GAIA（深度网页搜索）。

**📈 对比分析**

与提示或单一后训练基线相比，ADAPT 在 TextWorld 和 GAIA 上分别将成功率提升至 77.3% 和 49.3%，接近 oracle 上限，并且每任务仅约 1‑2 次提问；在 LiveCodeBench 上也保持竞争力。

**⚠️ 局限性**

局限包括仅适用于可直接验证成功的任务、依赖对抗式样本挖掘、以及对更大模型或更复杂任务的可扩展性待进一步验证。

---

## 260. On the Challenges of Using LLMs in High-Risk AI Recruitment: Guiding Questions for Legal Conformity with the AI Act

**arXiv ID:** 2611.27462 | [PDF](https://arxiv.org/pdf/2611.27462v1)

**作者:** Clara Punzi `[一作]` (Scuola Normale Superiore), Marzio Di Vece `[通讯]` (Scuola Normale Superiore)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建并验证了一套面向高风险招聘系统的合规工具包（HR‑RAI Toolkit），通过将《欧盟AI法》第9–15条及附件IV的技术规范转化为清单与指导性问题，帮助后端开发者在整合第三方通用AI（GPAI）模型时实现“依法设计”。

**💡 创新点**

创新点在于：①将法律条文细化为可操作的技术检查清单和问答体系，提供了从设计、数据治理、技术文档、透明度、人工监督到安全性等方面的完整可追溯流程；②通过与银行机构的合作者共同设计，展示了跨学科协作的可行模式；③提出了“信息差”分析框架，指出在高风险场景下，后端提供商与GPAI上游供应商之间因专有信息限制而导致的合规盲区。

**🔧 技术方法**

技术上主要采用了：①基于AI法条文的文本分析与需求映射；②工具包的实现为结构化清单与问答数据库，可在网页界面调用；③案例演示采用了LLM（如闭源/开源大型语言模型）驱动的简化招聘管道（解析、表示学习、评分、短名单），并结合模型卡、性能评估报告等技术手段。

**📊 数据集**

数据集方面，论文未提供具体公开数据集，而是描述了在构建测试集时可能使用的真实或模拟简历与职位描述数据，强调需涵盖多语言、多模板、真实使用场景的代表性样本；若使用开源LLM，则可利用公开的招聘/简历基准数据进行评估。

**📈 对比分析**

方法对比：工具包本身不做算法性能对比，而是通过清单与问答评估合规性。论文通过示例功能（候选人评分）演示了如何对应每一条法律要求提出问题并收集证据，说明其在合规评估中的可操作性；未给出量化性能指标。

**⚠️ 局限性**

限制：①工具包聚焦于AI法核心条款，未涵盖附加的《基本权利影响评估》等补充性合规工具；②对信息差的分析主要理论化，缺乏大规模实证验证；③未对开源与闭源GPAI在信息共享、可解释性等方面的差异进行深入实验；④论文缺少算法性能评估与多场景验证，工具的通用性和可推广性待进一步实证。

---

## 261. GaLF-ST: Gaussian Latent Fields for Spatial Transcriptomics Reconstruction

**arXiv ID:** 2611.27143 | [PDF](https://arxiv.org/pdf/2611.27143v1)

**作者:** Baichuan Yang `[一作]` (Renmin University of China), Bing Su `[通讯]` (Renmin University of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 GaLF-ST，一种利用可学习高斯原语和低维潜在向量构建的连续空间转录组表达预测框架。

**💡 创新点**

创新点在于用可学习的高斯中心和各向异性协方差显式建模空间几何，联合优化潜在向量与解码器，并在统一基准上全面评估。

**🔧 技术方法**

技术上采用高斯隐场、软最大归一化混合、Cholesky 形式的协方差、共享非线性解码器（GELU）、AdamW 训练和标准化损失等。

**📊 数据集**

使用了七组不同组织的 ST 数据集，包括 MOSTA E9.5/E16.5、Visium 小鼠脑、Slide-seqV2 海马、DLPFC 以及 3D MOB 等，共计 21 种评估设置。

**📈 对比分析**

与 STAGE、SUICA、CPS、STINR_ref 等四种代表性方法在相同采样点、拆分、基因面板和评估指标下对比，GaLF-ST 在 20/21/21/20 个设置中均取得最低 MSE、最高 PCC_g 与 PCC_s，整体性能显著优于对照组。

**⚠️ 局限性**

局限性包括在极低观测密度下高斯原语数量过多可能导致过拟合，需手动调参 K、q；目前主要针对 2D 平面，3D 扩展仍有限；对高斯初始化和超参数敏感性尚待进一步研究。

---

## 262. ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval

**arXiv ID:** 2611.26919 | [PDF](https://arxiv.org/pdf/2611.26919v1)

**作者:** Venkata Chivukula `[一作]` (Google), Jinsung Yoon `[通讯]` (Google)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ERA（Embed, Route, Adapt）系统，解决冷启动跨模型潜在检索问题，使得查询可以在不重新嵌入文档的情况下使用已有索引进行检索。

**💡 创新点**

创新点包括：①使用余弦最远点采样（Cosine FPS）在有限的锚点预算内实现多样化锚点选择；②引入结合方向对齐与全局正交正则化的流形对齐损失（Manifold Alignment Loss）以防止在锚点稀缺时的表示崩塌；③设计轻量级的确定性路由器，将罕见且难检索的查询“升级”至目标模型，以提高感知质量。

**🔧 技术方法**

采用的技术有：余弦最远点采样、Manifold Alignment Loss（cosine + GOR）、两层SwiGLU MLP适配器和路由器、轻量级查询特征提取器、以及多阶段无标签训练（Cold → Independent → Foundation）。

**📊 数据集**

在五个BEIR基准数据集（SciFact、FiQA、NFCorpus、Arguana、SciDocs）上进行实验，使用目标索引模型为Cohere、Gemini等。

**📈 对比分析**

与点对点ℓ₂回归（LEAF）和Embedding-Converter等基线比较，ERA在大多数数据集上提升nDCG@10（最高相对提升35%）且在T-nDCG@10上表现更好；在20%升级预算下，路由器能捕获约30–40% 的高难度查询，显著低于随机分配。整体系统延迟仅增加约1.2 ms，显著降低端到端查询延迟。

**⚠️ 局限性**

局限性包括：①维度差距导致源模型低维无法恢复目标模型高维语义；②纯FPS在离群点多的语料中可能消耗预算在边缘文档；③路由器在噪声较大的查询（如FiQA）区分度不足；④实验仅在英文检索场景，缺乏多语言和大规模语料验证。

---

## 263. FedQCal: Quadratic Programming with Dual-Space Calibration for One-Shot Federated Learning

**arXiv ID:** 2611.27170 | [PDF](https://arxiv.org/pdf/2611.27170v1)

**作者:** Shunsei Koshibuchi `[一作]` (Chiba University), Kazuhiko Kawamoto `[通讯]` (Chiba University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedQCal，一种利用少量服务器标签通过参考模型对齐约束，在单轮联邦学习中对客户端模型进行加权融合的方法。

**💡 创新点**

创新点在于：①将服务器参考模型训练得到的对齐得分作为二次规划约束；②通过最小范数 QP 产生唯一的权重集合，并用两维坐标动态调整满足约束的权重；③在此解空间内用分类交叉熵自适应选择最佳聚合方案。

**🔧 技术方法**

使用了参考模型训练、注意力 key–query 总结、最小范数二次规划（QP）、双坐标（u、v）优化、Armijo 回溯、ViT‑Tiny/16 预训练模型，以及 SGD 训练。

**📊 数据集**

在 CIFAR‑10、CIFAR‑100 与 SVHN 三个视觉分类数据集上进行实验，分别使用 10 或 100 个客户端，并考虑不同的 Dirichlet 标签偏斜参数。

**📈 对比分析**

与 Uniform Averaging、FedAvg、TIES‑Merging、Reference‑Only 等基线对比，FedQCal 在 17/18 个设置中平均提升 3–10%（最高可达 12%），在标签偏斜较大时提升更为显著。

**⚠️ 局限性**

局限性：仅能在参考对齐约束产生的 QP 解空间内搜索，无法恢复被排除的客户端贡献；需要服务器端额外的参考训练与梯度传播；对极少服务器标签（仅 1/类）仍存在性能波动。

---

## 264. From Letter-of-Credit Requirements to Executable Specifications: Evaluating Verification Accuracy and Completeness

**arXiv ID:** 2611.26988 | [PDF](https://arxiv.org/pdf/2611.26988v1)

**作者:** Ji Sub Choi `[一作]` (Hana Institute of Technology), Inyong Yun `[通讯]` (Hana Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在信用证（LC）审查中，提出将需求逐条拆解为可执行的规范，并在真实案例上对每个需求进行评估；

**💡 创新点**

通过将自由文本条款转换为可执行的原子条件和逻辑结构，使审查过程可追溯、可衡量；同时使用合成对进行无机密数据训练，实现在没有真实信用证记录的情况下实现高覆盖率；

**🔧 技术方法**

使用Qwen3.5-9B大语言模型（Fine‑tuned via LoRA）作为解析器，配合自定义的可执行验证器和六种运算符；利用模板生成的合成数据进行自监督训练；

**📊 数据集**

训练数据：2,000条合成的条款‑规范对，涵盖38个关键字段；测试数据：345条真实信用证案例（包含BL与IV的条款、事实库）；

**📈 对比分析**

与直接预测、通用Agent、零射击与少射击等基线相比，SFT（Fine‑tuned）在单位成功率上达到83.52%（PASS）/45.35%（FAIL），整体平衡准确率64.44%；在案例完整检查率（CER）上为46.96%，远超少射击（25.51%）和零射击（7.54%），且优于基线Agent（54.17%）；

**⚠️ 局限性**

局限性包括：仅评估BL与IV的第46A字段，忽略第47A及背面背书等；合成对与真实标注共享编码约定，可能偏向模型；仅单一机构数据，缺乏跨机构验证；FAIL案例数量有限，难以充分评估异常检测；缺乏独立双重标注，无法评估标注一致性。

---

## 265. $γ$-Scape: An Agent-Centric World for Mobile Embodied Intelligence

**arXiv ID:** 2611.27306 | [PDF](https://arxiv.org/pdf/2611.27306v1)

**作者:** Xiaotao Zhang `[一作]` (Institute of Artificial Intelligence (TeleAI), China Telecom), Xuelong Li `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了 BenchForge 平台，支持从任务描述到物理实例的编译、完整任务的仿真与评估，并提供经验收集与重用机制，形成 22 族、220 个实例的完整移动操作基准。

**💡 创新点**

① 通过任务模板与物理约束实现可重复、可扩展的完整任务实例化；② 将程序化行为、运动工具与学习技能无缝集成，允许代理自行编排程序并依据物理反馈进行递归改进；③ 在统一的任务定义与评估器下对不同体系结构进行公平比较，推动移动操控研究从单一操作转向完整任务。

**🔧 技术方法**

使用 Isaac 物理引擎、BenchForge 编译器、可视化仿真；代理端集成 CaP‑X、ASPIRE、OWMM‑Agent、RAX 等系统；利用 GPT‑6‑Astra 进行程序生成与重写；采用运动工具（cuRobo 等）与学习策略（LeRobot、RLinf 等）实现运动与技能；实现经验收集接口，记录工具轨迹与感知‑动作序列。

**📊 数据集**

370 个室内场景与 11,073 个可交互物体资产（来自 VLNVerse 与 InteriorGS），构成任务库；22 族、220 个实例包括递送、存储、准备、装载等多种场景。

**📈 对比分析**

通过让同一机器人（R1 Pro）在相同物理实例上运行四个集成系统，比较完整任务成功率与语义进度。RAX 以 21.8% 的成功率与 39.1% 的平均进度领先；ASPIRE 15.5%/34.5%，CaP‑X 10.0%/23.2%，OWMM‑Agent 4.1%/12.2%。在各族中，RAX 几乎覆盖所有族；但整体完成率仍偏低，尤其在携带与协作族。

**⚠️ 局限性**

① 完整实例完成率低，许多任务未能完全实现；② 现有系统对携带、双手协作等复杂交互表现不足；③ 依赖于虚拟抓取与无接触评估的功能未被启用，限制了经验收集；④ 经验收集与重用仅在实验环境下验证，缺乏跨平台泛化；⑤ 任务生成与诊断仍需人工干预，自动化水平有限。

---

## 266. Action-Conditioned Spatio-Temporal Forecasting

**arXiv ID:** 2611.27044 | [PDF](https://arxiv.org/pdf/2611.27044v1)

**作者:** Yutong Xia `[一作]` (National University of Singapore), Roger Zimmermann `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了面向网络化系统的动作条件时空预测任务，并构建了包含交通、电网和水网的 PairST 基准，用来直接监督和评估在相同系统状态下有无动作时的预测差异。

**💡 创新点**

创新点包括：① 通过模拟生成“参考未来”和“干预未来”对，实现对动作产生的响应（ΔX）的直接观察；② 设计了 Action-conditioned Spatio‑Temporal Neural Network（ActionST），其核心是响应路径：节点‑时间寻址、状态条件响应写入器和图感知局部门控，能够在满足前置一致性、无效动作时保持零响应等结构性质的同时，有选择地写入预测变化；③ 在训练中引入受影响区域、安静区域和前缀一致性约束的损失函数。

**🔧 技术方法**

使用了图卷积、节点和时间嵌入、WaveNet 风格的因果卷积、注意力机制、门控网络等深度学习技术，并在 PyTorch 上实现；

**📊 数据集**

使用了六个 PairST 数据集：交通（CityFlow 的 T16、T196），电网（Grid2Op 的 P59、P186），水网（EPANET 的 W119、W40）。

**📈 对比分析**

与 STGCN、Graph WaveNet、AGCRN、ConFormer、IGSTGNN、STAEformer、STID 等七个基线模型进行对比。结果显示 ActionST 在五个数据集上均获得最低 MAE，平均排名第 1，且在受影响区域（AMAE）和安静区域（QMAE）上表现优于大多数基线，特别是交通场景下受影响区域误差显著下降，安静区域误差也大幅降低。

**⚠️ 局限性**

局限性包括：目前 PairST 仅覆盖单一动作类型、单目标、单次动作；不涉及多次或并行多目标干预；在某些数据集（如水网 W119）响应幅度被抑制导致安静区域误差低但受影响区域误差偏高；以及模型对不同系统的空间扩散特征需要手动调节参数（k、d）。

---

## 267. Depth-Linked Residual Tokens for Multimodal Retrieval and Generation

**arXiv ID:** 2611.26752 | [PDF](https://arxiv.org/pdf/2611.26752v1)

**作者:** Jiangfan Zhang `[一作]` (Meta AI), Jun Xiao `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 Depth-Linked Residual Tokens (DepThink) 框架，用可学习的特殊 token 在单个前向传递中通过残差递归实现迭代细化，从而高效完成多模态检索和条件生成。

**💡 创新点**

创新点：①将特殊 token 通过一阶线性残差递归相连，实现与循环模型相似的迭代推理但保持并行；②给出理论证明，说明 DepThink 在方向可达性和一阶梯度下降方面优于普通可学习 token；③在保持参数效率的前提下，显著提升训练和推理速度。

**🔧 技术方法**

技术手段：共享 Transformer 层、残差门控递归、LoRA 与全参数微调、InfoNCE 检索损失、自回归生成损失、LLM 判分器评估。

**📊 数据集**

使用的数据集：MMEB‑V2、GSM8K、MATH、Natural Questions、HotpotQA、CIRR、MSCOCO、MSVD、QVHighlights、OK‑VQA 等多模态检索与推理任务。

**📈 对比分析**

对比方法与性能：与普通可学习 token、LoopLM 以及多种现有模型（如 CoCoOp、BLIP‑2、VLM2Vec、TTE‑Flash 等）进行基准对比；在检索任务中 DepThink 提升 4–6% 召回率，生成任务中 GSM8K 达 71.1%、MATH 达 82.4%（略高于 LoopLM），在 NQ/HotpotQA 上略优；训练速度提升约 11.9×、推理速度约 4.3×。

**⚠️ 局限性**

局限性：对 token 数量 K 的选择敏感；理论证明基于梯度可微和特定假设；在极大模型或极大 K 时仍存在额外开销；未在超低延迟实时场景中进行系统评估。

---

## 268. DermKG: A Concept-Centric Knowledge Graph for Interpretable Skin Disease Diagnosis from Dermatological Morphology

**arXiv ID:** 2611.27283 | [PDF](https://arxiv.org/pdf/2611.27283v1)

**作者:** Cheng Huang `[一作]`, Mao Lin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了以概念为中心的皮肤疾病诊断知识图谱DermKG，并提出了基于该图谱的分层图神经网络DermKG-Net，实现从皮肤病学概念到疾病类别的可解释诊断。

**💡 创新点**

创新点在于：①以专家注释的48个皮肤病学概念为节点构建多层次知识图谱，融合疾病概况、概念共现及20条临床规则；②进行信息上限分析，量化概念描述对诊断的固有限制；③提出的DermKG-Net通过分层原型目标和图结构显著提升少数类别召回率，且可生成规则链解释。

**🔧 技术方法**

使用了图神经网络（GCN/GAT）、基于贝叶斯、kNN、随机森林、梯度提升、MLP等基线模型、非参数KG混合与规则分数融合模型，以及信息上限分析和Bootstrap置信区间评估。

**📊 数据集**

主要数据集为Fitzpatrick17k中的SkinCon子集，共3,230张带有48个二值概念标签和113疾病标签的图像，按疾病划分训练/验证/测试。

**📈 对比分析**

与11个基线模型对比，DermKG-Net在9级类别诊断上将宏F1提升至0.299，平衡准确率从0.257提升到0.411，少数类别召回率显著提高（如黑色素瘤从0.33提升到0.48），并在信息上限约束下达到约66%的平衡准确率。

**⚠️ 局限性**

局限性包括：概念描述的冗余导致疾病级别诊断受限（最大38.9%），部分罕见疾病仍无法通过概念区分，规则覆盖率不足，且图模型对规则边的依赖有限。

---

## 269. Evidence sufficiency assessment enables cost-effective review allocation in on-premise clinical AI agents

**arXiv ID:** 2611.27390 | [PDF](https://arxiv.org/pdf/2611.27390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 270. Skill-JEPA: Learning Planner-Aligned Action Spaces for World Models

**arXiv ID:** 2611.26825 | [PDF](https://arxiv.org/pdf/2611.26825v1)

**作者:** Tomás Couso Coddou `[一作]` (Centro Nacional De Inteligencia Artificial), Rodrigo Toro Icarte `[通讯]` (Pontificia Universidad Católica De Chile)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在JEPA世界模型中加入条件VAE，将动作序列映射为潜在技能空间，使规划器采样的动作与模型训练分布对齐，从而提升遥距目标的规划成功率。

**💡 创新点**

仅通过引入可学习的VAE技能模块并让其先验匹配规划器的采样分布，无需改动规划器、添加子目标或层次结构，即可显著提高规划性能。

**🔧 技术方法**

结合条件VAE、β‑VAE KL正则、JEPA世界模型（LeWM、PLDM、SMWM）以及多种采样规划器（CEM、iCEM、MPPI、Predictive Sampling）。

**📊 数据集**

使用 Five 任务的离线轨迹数据集：Two‑Rooms、Push‑T、OGBench‑Cube、Reacher 及 OGBench‑AntMaze。

**📈 对比分析**

在相同模型、数据与规划器设置下对比原始动作规划与技能规划；80 步目标时平均成功率从 42% 提升至 62%，AntMaze 两格远目标从 0.4% 提升至 38%，且提升随目标距离扩大，在四种规划器上均保持。

**⚠️ 局限性**

分析仅覆盖规划器首轮候选与单步预测，无法解释细粒度搜索的影响；预测误差并未完全解释性能提升；实验仅在仿真、固定技能长度、原始动作基线与高斯起始规划器中进行。

---

## 271. PAVE: A Harness for Geometrically Consistent Road Network Modeling with Vision-Language Models

**arXiv ID:** 2611.27352 | [PDF](https://arxiv.org/pdf/2611.27352v1)

**作者:** Yajun Liu `[一作]` (Arizona State University), Xuesong Zhou `[通讯]` (Arizona State University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

PAVE将Vision‑Language Model与代码测量、交通规则检查相结合，实现对道路网络的自动读取与编辑。

**💡 创新点**

创新点在于将模型判定与代码测量、规则校验分离，利用Anchor‑Verify harness确保几何一致性和网络完整性。

**🔧 技术方法**

使用技术包括Vision‑Language Model、GMNS数据结构、代码测量与规则检查、Anchor与Verify步骤。

**📊 数据集**

数据集为Tempe, AZ的11个信号化交叉口的卫星与街景图像，结合OpenStreetMap lane tags，共44个道路方向。

**📈 对比分析**

通过与单次请求和无harness对比，PAVE在44个方向上将正确率从约50%提升至90%；在编辑任务中，16/18次编辑通过所有网络检查，性能显著优于无harness情况。

**⚠️ 局限性**

局限包括仅在单一城市样本上验证、请求量较大、仍需人工审核、以及对非道路场景适用性的未知。

---

## 272. EcomWriteBench: A Real-World Benchmark for E-commerce Writing Tasks

**arXiv ID:** 2611.27349 | [PDF](https://arxiv.org/pdf/2611.27349v1)

**作者:** Kaicheng Liang `[一作]` (JD.com), Simiu Gu `[通讯]` (JD.com)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建了一个多语言、跨任务的电子商务写作基准 EcomWriteBench，包含 3861 题目、69 个具体任务和 13 个工作流类别，并提供从任务需求中提取的 37 项细粒度能力词汇。

**💡 创新点**

创新点在于：① 任务驱动的能力发现流程，生成可解释且共享的细粒度能力集合；② 用人类配对偏好作为监督，构建基于约束优化的逆向加权学习框架，得到全局、类别级和任务级可解释权重，从而实现任务自适应的评分聚合。

**🔧 技术方法**

技术手段包括：多任务能力提取与聚合、基于 L2/L1 约束的层次化权重学习（逆向偏好校准）、LLM 评估者的三方平均评分、以及可扩展的评分与评估流水线。

**📊 数据集**

数据集：自研的 EcomWriteBench（多语种、跨工作流的真实业务写作查询），以及 15 种 LLM（Gemini、Gemma、Qwen 等）生成的响应，用于评估。

**📈 对比分析**

评估方法：与 LLM-as-a-judge、统一能力聚合、以及 AutoMetrics、MetaMetrics、HD-Eval、LLM-Rubric 等四种人类校准评估方法对照；在留出任务的 5 折交叉验证中，人类偏好一致率达 0.8896，明显优于其他方法；模型排名显示 GPT‑5.6‑Sol 等模型在整体和各能力族上均表现最优。

**⚠️ 局限性**

局限性：依赖人工偏好标注，标注规模有限；能力词汇覆盖仍不完整，可能漏掉极端或新兴写作需求；评估侧重写作质量，未直接衡量业务效果；跨语言的可解释权重虽已初步验证，但在更大语言覆盖范围下仍需进一步验证。

---

## 273. Bounded Treewidth Rulesets Are Not Finitely Controllable

**arXiv ID:** 2611.27461 | [PDF](https://arxiv.org/pdf/2611.27461v1)

**作者:** Piotr Ostropolski-Nalewaja `[一作]` (University of Wrocaw), Michaël Thomazo `[通讯]` (Inria)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7`

**🎯 论文内容**

证明存在一个仅包含三条规则的二元签名规则集，该规则集是有界树宽且pieceful，但并非有限可控（finite controllable）。

**💡 创新点**

通过构造一个具体数据库和布尔连接子查询，展示在无限 chase 中不满足该查询，而在任何有限模型中都满足，从而给出有界树宽规则集不一定有限可控的反例，打破此前的期望。

**🔧 技术方法**

主要使用 chase 算法、树宽（treewidth）与归约技巧、激活触发器（triggers）与公平推导（fair derivations）的理论分析。

**📊 数据集**

本研究纯粹为理论证明，不使用任何公开数据集或实验数据。

**📈 对比分析**

由于结果是负向理论证明，未进行实验或性能比较；文章只给出逻辑推导与构造证明，没有基准或对比方法。

**⚠️ 局限性**

限制在于只给出单一反例，未提供一般性的判定方法或算法；关于局部规则集是否能保证有限可控仍是未解问题，且未探讨其他可能的约束或改进。

---

## 274. SIR-Agent: Skill-Augmented Image Restoration Agent with Recursive Self-Improvement

**arXiv ID:** 2611.27285 | [PDF](https://arxiv.org/pdf/2611.27285v1)

**作者:** Yijian Wang `[一作]` (Northwestern Polytechnical University), Jiantao Zhou `[通讯]` (University of Macau)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为SIR-Agent的技能增强型图像修复代理，通过链反馈驱动的技能修订(CFSR)和递归自我改进(RSI)的技能元学习机制，在不微调大模型的前提下实现对新图像修复任务的自适应；

**💡 创新点**

创新点在于：①将恢复策略知识以可扩展的技能库形式存储，并通过链反馈+兼容性评估实现对技能的精准修订；②引入基于RSI的技能元学习，自动分析修订轨迹以改进CFSR的规则，使修订过程自我提升；

**🔧 技术方法**

采用多模态大型语言模型Qwen3.8-Max、工具检索框架LlamaIndex、链反馈驱动的技能修订与兼容性评估、RSI驱动的技能元学习，以及多种图像质量评价指标（CLIP-IQA、HyperIQA、NIQE、CPBD、BRISQUE、MUSIQ、LIQE）来实现与评估；

**📊 数据集**

使用了14个已知图像修复基准（包括去模糊GoPro/HIDE、去雾SOTS/Dense-Haze、去噪BSD68/SIDD、去雨Rain13K/Rain1400、去雪CSD/RealSnow、低照明增强LOL-V1/LOL-V2、复合退化LOL-Blur/CDD-11）以及7个未见任务（去反射NightIRS、去阴影ISTD、消除摩尔纹FHDMi、海底增强LSMU、除尘SID、夜间眩光Flare7K++、去烟Smoke）；

**📈 对比分析**

通过与九种基准方法（四种全能模型PromptIR/RAM/CPLIR/HOGformer和五种代理方法AgenticIR/4KAgent/JarvisIR/OPERA/HarnessIR）在PSNR/SSIM上进行客观比较，SIR-Agent在所有任务组均取得最高分，单个退化任务提升约1.9–11 dB，复合退化提升约7.95 dB，未见任务同样表现优异；

**⚠️ 局限性**

限制在于：依赖大模型与工具库的质量，技能修订过程仍可能产生不必要的工具调用和令牌消耗；未对极端退化或低资源环境的适应性进行充分评估，且对知识迁移的可解释性有限。

---

## 275. Structure Over Scale: Rethinking Adaptation for Reinforcement Learning with Verifiable Rewards

**arXiv ID:** 2611.27148 | [PDF](https://arxiv.org/pdf/2611.27148v1)

**作者:** Allan Kazakov `[一作]` (Bahçeşehir University), Abdurrahman Javat `[通讯]` (Bahçeşehir University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文在冷启动的RLVR（基于GRPO的强化学习）框架下，系统比较了全量微调、LoRA、DoRA和QuanTA等参数高效微调方法，发现LoRA在高秩时会出现“秩崩溃”现象，而结构化适配器能够在更新参数极少的情况下保持甚至超过全量微调的性能。

**💡 创新点**

创新点包括：①提出并验证了“秩崩溃”这一新现象，说明单纯提高LoRA秩并不一定提升性能；②引入“模型成熟度假设”，指出预训练规模可能决定最佳更新几何；③用谱分析揭示更新几何（effective rank）与性能的关系，表明更新结构比参数数量更重要；④提供了可复现的实验代码和数据。

**🔧 技术方法**

技术手段包括：GRPO（Group Relative Policy Optimization）冷启动RLVR；LoRA、DoRA、QuanTA等低秩/结构化适配器；谱分析（SVD、effective rank）；多种基准任务（DeepMath‑Hard、Skywork‑Hard、Enigmata、MATH、AIME）和多种超参数实验（batch size、group size、seed）。

**📊 数据集**

使用的数据集主要是数学推理与逻辑推理基准：DeepMath‑Hard、Skywork‑Hard、Enigmata 10K、MATH、AIME 2025–2026，此外对三款8B模型（Apertus、Llama 3.1、Qwen 3）进行对比。

**📈 对比分析**

对比方法：在同一冷启动GRPO协议下，固定学习率调度、batch size 64、group size 8，分别对LoRA（不同秩）、DoRA、QuanTA和全量微调进行一次epoch训练。结果显示：LoRA在低秩（≤32）时性能与全量微调相近，但在秩≥64时急剧崩溃；DoRA和QuanTA在更新参数<0.6%时即可达到或超过全量微调的性能；在不同batch size和group size下，LoRA的崩溃区域不易被修复，结构化适配器则更稳健。

**⚠️ 局限性**

局限性：①实验仅在8B密集Transformer上验证，未探讨更大模型或MoE结构；②仅采用冷启动RLVR，不包含SFT‑to‑RL路径；③秩与学习率调度等超参数未做全面调优，结果受特定配置影响；④模型成熟度假设仅基于三款模型，无法完全分离预训练规模与架构/Tokenizer等因素；⑤对非数学推理任务的验证仅做单种实验，需进一步扩展。

---

## 276. Who Should Verify? Reviewer Selection with Partial Information and Outcome Audits

**arXiv ID:** 2611.26770 | [PDF](https://arxiv.org/pdf/2611.26770v1)

**作者:** The Viet Bui `[一作]` (Singapore Management University), Tien Mai `[通讯]` (Singapore Management University)

**通讯引用:** 1575 | [OpenAlex ID](https://openalex.org/A5074307998)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `a4b10f5d-130b-4e77-9367-6469ec621899` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文研究在审计团队选择使用哪种人工智能模型评审的过程中，如何通过购买最少的证据（键或评审结果）来证明所选方案在已审计批次中几乎是最优的，并分析搜索成本与证据成本之间的关系。

**💡 创新点**

创新点在于提出了可证明的证据“证书”概念，给出了在已知所有评审答案时最少键的整数规划与多程序情况下的多重价格下近似最优贪婪算法，并证明搜索成本与最优证书成本的对数下界；同时针对二元评审问题给出可在期望成本下最优的动态规划策略。

**🔧 技术方法**

技术包括：可证明性停机判据、整数规划、贪婪覆盖近似、哈曼数近似、动态规划求最优搜索策略、贝叶斯/频率方法估计评审误差分布、实验回放和对照实验。

**📊 数据集**

数据集主要有：MMLU多语言模型回答与互评（2000道多选题），数值报告检查（896张表格），新加坡电力需求预测报告（48份）。

**📈 对比分析**

比较方法：将针对性审计与在开发集上选出的固定顺序、均匀采样、投票分歧优先等策略进行对比。结果显示：在MMLU上，针对性审计在已审计批次中比固定顺序节省约6.9%键，在held-out上提升约2.94个百分点；在数值报告上差异不大。证书成本在已知所有答案时与贪婪算法差距不超过4个价格单位。

**⚠️ 局限性**

局限性包括：模型误差分布估计依赖开发样本，未处理多程序、非二元答案下的最优搜索；证书近似仅在已知所有评审答案时适用，实时搜索成本仍可能远大；实验仅覆盖特定任务与模型，结果可能难以推广到更复杂或异构评审环境。

---

## 277. From Manufacturing Event Logs to Simulation: Enabling SMEs with Agentic AI for SimPy Simulation Generation

**arXiv ID:** 2611.27483 | [PDF](https://arxiv.org/pdf/2611.27483v1)

**作者:** Richard Reider `[一作]` (Otto von Guericke University), Pascal Reusch `[通讯]` (Bielefeld University of Applied Sciences and Arts)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了 GenAI4SimPy，一个多智能体框架，能从原始制造事件日志自动生成可执行的 SimPy 离散事件仿真模型。

**💡 创新点**

创新点在于：①使用本地开源 27B 参数 LLM（Qwen 3.6）与工具增强推理相结合，突破云端模型的隐私与成本壁垒；②通过分阶段验证门控和人机交互降低专家依赖；③揭示“微观-宏观精度悖论”，指出参数提取不足以保证系统级行为。

**🔧 技术方法**

技术手段包括：多智能体协作框架、Ollama 本地推理、函数调用接口实现统计分析与分布拟合、SimPy 代码生成与自我修复循环、结构与参数验证器、HITL 交互界面。

**📊 数据集**

使用公开的 Fischertechnik Smart Factory 事件日志（包含三个连续执行块 A、B、C）进行评估。

**📈 对比分析**

评估方法：在 30 次独立仿真复制中对比关键指标（故障率、吞吐量、平均流时、MAPE、JSD）与真实数据。结果显示活动级 MAPE 约 7.9%，所有块的故障率覆盖率良好；吞吐量覆盖率在 B、C 块达成，但 A 块超估 500%；系统级 KPI 因观察窗口截断误判导致显著误差。

**⚠️ 局限性**

局限性包括：①参数提取与结构验证无法捕捉系统级行为错误，需额外的系统校准；②大型上下文导致“中间丢失”现象，影响代码生成；③仅在单一数据集和单个 LLM 上验证，缺乏多样性与细粒度微调；④需要约 56 GB VRAM 的 BF16 推理，仍高于许多 SMEs 可承受的硬件门槛。

---

## 278. Gumbel Substitution: Removing Sampler Drift from RL Without Trusting the Sampler

**arXiv ID:** 2611.27375 | [PDF](https://arxiv.org/pdf/2611.27375v1)

**作者:** Romain Fanucci `[一作]`, Malik El Hassani `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在开放网络上训练语言模型时，提出了 Gumbel 替代（substitution）方法，用公开的 Gumbel 噪声将采样器的随机性转化为训练者自己的噪声，从而消除采样器分布 q 与训练者策略 p 之间的漂移；实现了无可信概率、异构硬件、不同精度和检查点滞后环境下的鲁棒训练。

**💡 创新点**

创新点在于：①通过共享、公开且可预先确定的 Gumbel 噪声，实现训练目标的“自我重写”，不再需要采样器的概率信息；②提出了 Gumbel 替代损失，保证在任何硬件、精度和滞后情况下漂移在期望上消失；③设计了可部署的噪声生成和校验协议，支持权限网络下的异构 GPU 并实现可验证支付。

**🔧 技术方法**

技术手段包括：公开 Gumbel 噪声的承诺式生成（checkpoint‑level secret + 哈希索引 + splitmix64+Gumbel-max）；Gumbel 替代损失；基于匹配率、gap 与最终 EOS gap 的门控与支付机制；异步 PPO/GRPO 训练；实验中使用多环境 RL（指令跟随、数学、代码、工具使用等）。

**📊 数据集**

数据集与任务涵盖：多轮对话与单轮任务，包括 DAPO math、hard math、code、instruction following、SWE tasks、TMax terminal tasks、EnvScaler multi‑turn tool use；评估基准为 SWE‑bench Verified、TMax、LiveCodeBench、AIME 2025/2026、IFBench、MMLU‑Pro、GPQA Diamond。

**📈 对比分析**

对比方法：未纠正 (L0)、遮掩 (L1)、Gumbel 替代 (L2) 与真值分布校正 (L3)。在实验室中，L2 在强误差条件下保持稳定，表现与 L3 相近；在生产网络中完成 158 步、15.75B token 的训练，所有 8 个保留基准均提升，SWE‑bench Verified 从 43.6 提升至 69.9（+26.3）。

**⚠️ 局限性**

局限性包括：①梯度仍受 off‑policy 状态与奖励影响，导致一定偏差；②fp32 取样与 bf16 计分间的残差漂移；③门控排除部分 rollout，虽占比低但不完全消除漂移；④对自适应采样器（使用未来噪声或隐藏 group）的防护有限；⑤实验室验证主要在 2B 模型，缺乏更大规模对比；⑥阈值与门控需在不同 lr、环境下重新校准；⑦多轮情景下的 cross‑credit 偏差与终止检验尚未充分评估。

---

## 279. V2VFusion: Text-Controlled Video-to-Video Diffusion for Degradation-Aware Video Fusion

**arXiv ID:** 2611.27063 | [PDF](https://arxiv.org/pdf/2611.27063v1)

**作者:** Jiajun Chen `[一作]` (Southeast University), Jiayi Ma `[通讯]` (Wuhan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `afceb026-1760-41ae-8d86-010831a37d97` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 V2VFusion 框架，直接在潜在扩散空间中完成多源视频的统一时空融合，支持降噪、去模糊等多种降解环境下生成高质量、时序连贯的视频。

**💡 创新点**

创新点：① 把视频融合转化为视频‑到‑视频条件生成任务，实现全序列的统一时空建模；② 设计文本引导降解条件的 ControlNet，将高层降解描述与低层视觉先验融合；③ 引入状态调制的降解感知分层 Mixture‑of‑Experts，按 diffusion 状态和降解类型动态路由专家。

**🔧 技术方法**

采用潜在扩散模型 + ControlNet + 文字引导 + 语义先验 + 状态调制 MoE + 频域重建损失 + progressive distillation 等技术。

**📊 数据集**

在 M3SVD（红外‑可见）、YouTube‑HDR（多曝光）和 DAVIS（多焦距）数据集上进行实验，并在这些数据上合成模糊、噪声、条纹、JPEG 等多种降解。

**📈 对比分析**

与多种图像融合（U2Fusion、DeFusion 等）和视频融合方法（VideoFusion、TemCoCo、UniVF）进行对比，使用 MI、Qabf、tSSIM、tF、tLP、BiSWE 等指标评估，V2VFusion 在所有指标上均优于竞争方法，显著提升融合质量、降解抑制和时序一致性。

**⚠️ 局限性**

局限性：仍需要大量计算资源（扩散采样耗时长，虽采用 progressive distillation 缓解但成本高），对极端光照/快速运动场景的鲁棒性尚未充分验证；模型对文本描述的依赖可能受限于描述的准确性。

---

## 280. Which Dwarf Is This? Doppeldwarf-306 and Near-Identical Instance Retrieval

**arXiv ID:** 2611.27490 | [PDF](https://arxiv.org/pdf/2611.27490v1)

**作者:** Turhan Can Kargin `[一作]` `[通讯]`, Turhan Can Kargin

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构建了包含 306 只 Wrocław（弗罗茨瓦夫）青铜矮人雕像及其 1,691 张公开授权照片的实例检索数据集（Doppeldwarf-306），并在该数据集上评估了多种冻结的视觉模型与后处理方法的检索性能。

**💡 创新点**

创新点包括：① 将摄影师信息与坐标记录在数据集中，允许“未知摄影师”压力测试；② 引入基于地理位置的软先验权重，而非传统的硬过滤；③ 对 13 个来自 7 种预训练范式的冻结模型进行统一比较，系统揭示了自监督、语言对齐、掩码预训练等方法在实例检索中的差异；④ 通过局部特征重匹配（SIFT、DISK+LightGlue、LoFTR）实现可训练的重排序与拒绝判定。

**🔧 技术方法**

使用的技术主要是：
- 12 个冻结的视觉背骨（DINOv2、DINOv3、CLIP、SigLIP 2、MAE、BEiT、I-JEPA、EVA‑02、C‑RADIOv3、AIMv2、Supervised ViT）。
- 最近邻检索（余弦相似度）
- 局部特征匹配与 RANSAC 重定位
- 位置加权软先验（λ·exp(−d/200 m)）
- 通过阈值化最佳‑与次佳相似度差实现未知实例拒绝。

**📊 数据集**

数据集：Doppeldwarf-306，包含 306 个青铜矮人雕像（其中 294 个已标注经纬度），共 1,691 张公开授权照片，记录摄影师、版权、坐标等元数据。

**📈 对比分析**

比较方法：对 1,691 张留一检索（leave‑one‑out）进行全量搜索，记录 top‑1 准确率；在“未知摄影师”情形下与随机控制对比，评估摄影师依赖；对不同候选池大小（2–306）绘制准确率衰减曲线；对 4 个主干与 13 个模型的 top‑1、未知拒绝 AUROC、FA@90 等指标进行统计。性能结果：DINOv2 top‑1 93.1%，DINOv3 94.4%，最佳模型（C‑RADIOv3）接近 99%；未知摄影师时 DINOv2 降至 81.8%（控制 84.5%）；位置软先验将准确率提升至约 96.2%；在 top‑2 边际上可将未知拒绝率从 25% 降至 12%。

**⚠️ 局限性**

局限性：
- 所有查询均为 Wikimedia Commons 上传的照片，缺乏对随手街拍的评估；
- 数据集规模相对较小，且大部分照片来自少数摄影师，可能导致模型对摄影风格的偏倚；
- 对未知雕像的检索（即完全未出现过的雕像）未能完整评估；
- 位置加权的效果高度依赖实际布局，迁移到不同城市可能需要重新校准。

---

## 281. Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency

**arXiv ID:** 2611.27100 | [PDF](https://arxiv.org/pdf/2611.27100v1)

**作者:** Chenruo Liu `[一作]` (New York University), Qi Lei `[通讯]` (New York University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计信息理论框架，量化CoT推理的faithfulness和efficiency，并提出相应的Fine‑tuning方法来提升这两项指标。

**💡 创新点**

将faithfulness和efficiency视为预测信息分配的两个互补维度，使用语义推理图实现全局与步骤级度量，并通过图意识正则化实现目标提升。

**🔧 技术方法**

信息论指标、语义推理图（有向超图）、Rejection Sampling Fine‑tuning（RFT）以及图意识的正则化技术。

**📊 数据集**

MATH、Omni‑MATH‑2和2WikiMultiHopQA等数学与多跳推理基准。

**📈 对比分析**

与标准RFT及其他模型对比，指标Unfaithfulness和Inefficiency显著下降，token消耗减少且准确率保持不变甚至略有提升。

**⚠️ 局限性**

依赖高性能LLM进行语义图提取，faithfulness指标改进相对有限，缺乏更直接的验证和更有效的训练策略。

---

## 282. ReactDrive: Self-Reactive Driving World Model

**arXiv ID:** 2611.27024 | [PDF](https://arxiv.org/pdf/2611.27024v1)

**作者:** Yifan Wang `[一作]` (Stony Brook University), Liu Ren `[通讯]` (Bosch Research North America)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 Self‑Reactive Driving World Model (SADWM)，能够在给定 ego 行动的情况下预测周围车辆的反应并生成相应的未来驾驶视频。

**💡 创新点**

首次将“agent reactivity”（对 ego 行动的反应能力）作为交互式世界模型的核心指标，并通过反应性动力学先验将物理层面的人车交互与视觉生成耦合，形成端到端自回归训练。

**🔧 技术方法**

采用视觉、动作编码器 + 共享 token；反应性动力学先验（通过学习的状态预测器）生成 agent 状态；响应编码器将状态映射为响应 token；最终使用带响应条件的扩散 Transformer 生成视频；联合动态监督与视觉流式目标进行训练。

**📊 数据集**

在 Waymo Open Dataset 和 nuPlan 数据集上进行训练与评估，并在 NAVSIM 上进行下游规划任务的验证。

**📈 对比分析**

与 Vista、GEM、Epona、DrivingWorld、Cosmos3 等基线进行比较。SADWM 在 Agent ADE/FDE、Ego ADE/FDE、碰撞率（CR）和 FVD 上取得显著提升；在人类评估中在 agent reactivity、ego controllability、visual quality 以及整体偏好上分别获得 96.0%、86.7%、91.8% 和 98.1% 的优势；在规划任务中，使用其生成视频的训练提升 PDMS 至 85.1（相较 Cosmos3 的 83.8）。

**⚠️ 局限性**

视觉指标（FID/FVD）对模型差异不敏感，容易误导；模型对输入的空间时间位置编码高度依赖，去除后性能骤降；在极端交通场景或稀缺 agent 交互情况下，仍可能产生不合理的响应。

---

## 283. The Illusion of Evaluation: Challenges in Agentic Product Evaluation Practice

**arXiv ID:** 2611.26923 | [PDF](https://arxiv.org/pdf/2611.26923v1)

**作者:** Abhinav Palia `[一作]` (Microsoft), Drew Brooks `[通讯]` (Microsoft)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文系统分析了生成式 AI 产品评估中的六大方法挑战，并提出了“透明与决策相关记录”（TDRR）框架，以帮助明确评估结果与产品安全决策之间的关系。

**💡 创新点**

创新点在于：①将产品风险识别、评估工具、证据、声明与决策四个环节构建为可追踪链；②提出六大挑战（风险识别、生态有效性、指标解释、结果验证、证据转移、决策充分性）；③设计了 TDRR 模板，强调在评估报告中展示证据产生、支持程度与限制，从而揭示“评估幻觉”。

**🔧 技术方法**

主要技术手段为概念分析、案例映射和框架设计，借鉴了模型/代理评估工具（如 WebArena、SWE-bench、AgentDojo）、安全案例结构以及现有验证方法，构建六维挑战与决策映射。

**📊 数据集**

未使用公开数据集；论文基于十二个示例产品场景（工作协助、编码、临床记录等）进行说明，以覆盖不同风险类型、工具使用和权限设置。

**📈 对比分析**

本文没有对具体方法进行实验比较；评估框架的有效性与实用性仅在理论层面提出，需后续实证研究来检验其是否能缩小声明范围、提升审阅一致性或影响决策。

**⚠️ 局限性**

局限性包括：①缺乏实证验证，TDRR 的实际效用尚未证明；②示例场景仅为说明性、非代表性样本，未覆盖所有可能风险；③未提供具体评估工具或数据来源，读者需要自行实现或适配；④对评估者专业知识和资源的依赖较高，实施成本不明。

---

## 284. What Stays in the Picture? From Redundant Instructions to Dependency-Aware Image Editing

**arXiv ID:** 2611.27388 | [PDF](https://arxiv.org/pdf/2611.27388v1)

**作者:** Zi-Jun Wang `[一作]` (Sun Yat-sen University), Zhilin Zhao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了在单条指令中包含多重对象属性和相对位置，并可能包含冲突或撤销操作的复杂冗余指令图像编辑任务，提出CRIE-Agent框架，先通过LLM指令解析器（Intent Agent）恢复不可简化的最终编辑意图，再由Execution Agent按参考优先块分组、布局提示和验证循环执行编辑，显著提升成功率。

**💡 创新点**

创新点在于：①将冗余修订历史转化为无冗余、可验证的最终编辑目标；②设计参考优先的块分组策略，保证先编辑被引用的对象；③引入布局提示在重试时提供水平定位；④利用循环验证与重执行来避免误编辑，整体实现从意图恢复到视觉执行的闭环。

**🔧 技术方法**

技术包括：LLM驱动的Intent Agent（基于Qwen3‑VL‑8B‑Instruct的LoRA微调）、Execution Agent、块分组(Block Grouping)算法、布局提示(Layout)、结果验证(Verification)、以及与五个不同图像编辑后端（Qwen‑Image‑Edit 2511、Qwen‑Image‑2.1、FLUX.2‑klein‑4B、FireRed‑Image‑Edit‑1.1、JoyAI‑Image‑Edit）对接。

**📊 数据集**

使用的数据集有：CRIE‑S（10,000条带有冗余修订历史与最终意图注释的编辑指令）、Style30K（用于提供源图像的多样化视觉风格）以及CompBench多对象添加子集（用于额外评估）。

**📈 对比分析**

在实验中将CRIE‑Agent与上述五个后端以及三种现有编辑代理（ReasonEdit‑S、IntentEdit、RePlan）进行对比。结果显示，CRIE‑Agent将完整成功率从直接编辑的0%提升至27.5%–50.0%（在Qwen‑Image‑2.1上达到50.0%，在FLUX.2‑klein‑4B上达到27.5%），并在对象、属性、关系三项指标上显著优于基线。

**⚠️ 局限性**

限制主要在于：①VLM验证模块可能漏检或误判视觉对象与关系，导致不必要的重试或错误的候选选择；②对LLM输出的依赖使得模型性能受限于其推断质量；③块容量(K)与分组策略仍需经验调参，可能影响不同编辑后端的最优表现。

---

## 285. A Footprint-Sensitive Cross-GPU Timing Channel Through the Shared Host Memory Subsystem

**arXiv ID:** 2611.27115 | [PDF](https://arxiv.org/pdf/2611.27115v1)

**作者:** Wei Song `[一作]` (Griffith University), Xiao Cheng `[通讯]` (Macquarie University)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多 GPU 服务器上，研究者通过测量一块已映射的主机内存缓冲区的自定时读延迟，发现了一个跨 GPU 的时序通道，攻击者无需 NVLink、共享 IOTLB 或硬件计数器即可通过该通道实现隐蔽通信、语言模型输出长度推断以及拒绝响应推断。

**💡 创新点**

创新点在于揭示了仅共享主机内存控制器而无任何设备互连即可产生的、与 victim 访问内存页面数相关的 footprint‑sensitive 时序通道；并证明该通道源自 IOMMU 页面表遍历在共享 DRAM 中的竞争，突破了传统的基于缓存、TLB 或 PCIe 资源的攻击范式。

**🔧 技术方法**

技术手段包括：在攻击 GPU 上编写自定时核函数读取主机缓冲区、对比不同访存工作负载（大/小 footprint）产生的延迟差异、利用跨 NUMA 测试定位共享资源、对 IOMMU 进行 on/off 控制、实现基于持续访存的长度与拒绝推断模型、以及通过 footprint flattening 与内存控制器分离两种防御策略。

**📊 数据集**

实验使用了多种开源 LLM：Qwen2.5‑14B、SGLang、Llama‑3.1‑8B、Mistral‑7B‑v0.3 等，参数规模从 0.5 B 到 14 B；数据集为标准自然语言生成任务的提示/响应，覆盖 0.5 B‑14 B 模型、不同加载方式（CPU‑offload、weight‑offload）以及 vLLM+LMCache 生产栈。

**📈 对比分析**

对比指标包括：AUC（0.88–1.00）用于检测生成活动，Spearman 相关系数高达 0.99 用于长度推断，平均 20–50 bit/s 的隐蔽通信速率与误码率 < 2%，以及拒绝推断的平衡准确率最高达 0.955；实验表明，在受控条件下通道容量可达 0.86 bit/观测，且在噪声与基线漂移下保持 AUC ≥ 0.83。

**⚠️ 局限性**

局限性在于：通道需要受害者产生持续 2 GB/s 以上的 device‑to‑host 传输，突发或低平均负载（如 vLLM+LMCache 生产栈）不泄漏；长度推断需成功分段（约 2/3 进程可完成）；拒绝推断仅适用于产生更短拒绝响应的模型；实验仅覆盖三代 NVIDIA GPU，且对其他厂商或更大规模多租户环境的泛化仍待验证。

---

## 286. Towards Robust Partial-Edit Deepfake Localization: An edit and Generator Conditioned Analysis

**arXiv ID:** 2611.26903 | [PDF](https://arxiv.org/pdf/2611.26903v1)

**作者:** Aishwarya Fursule `[一作]` (Wichita State University), Shruti Kshirsagar `[通讯]` (Wichita State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `729e5870-4135-47f5-97f2-e3974d07b5dc` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了在不同编辑类型和生成器下的部分语音deepfake定位问题，提出按内容、意义和操作三轴划分编辑的诊断框架；

**💡 创新点**

创新点在于将编辑属性拆分为三轴进行细粒度分析，并对不同生成器的编辑差异进行条件化评估；

**🔧 技术方法**

采用WavLM‑Large和XLS‑R自监督语音特征结合Segment‑Aware Localization (SAL)框架进行定位；

**📊 数据集**

使用公开的PartialEdit数据集，包括VoiceCraft (E1) 和 SSR‑Speech (E2) 两个子集；

**📈 对比分析**

通过Fake‑F1和EER两项指标，在不同种子、生成器、编辑类型和句子新颖度下进行对比，整体表现高达约96% Fake‑F1，但SSR‑Speech的删除编辑显著下降至约80%；

**⚠️ 局限性**

局限在于仅对编辑持续时间和操纵帧比例做了后续控制，未能完全解释删除编辑难度差异，且实验仅覆盖两种生成器，统计效应虽显著但效应量小。

---

## 287. Hierarchical Hybrid-Coordinate Flow Matching for Accurate and Physically Plausible Apo-to-Holo Protein Generation

**arXiv ID:** 2611.27162 | [PDF](https://arxiv.org/pdf/2611.27162v1)

**作者:** Minjong Kim `[一作]` (Seoul National University), Juyong Lee `[通讯]` (Seoul National University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种分层流匹配模型，先用E(3)-等变形图网络在笛卡尔空间生成稀疏Cα锚点，再用条件Transformer在局部内部坐标系中递归填充间隙，实现从apo到holo结构的生成

**💡 创新点**

通过将全局笛卡尔坐标和局部内部坐标的优势结合，既保持全局精度又显著提升局部几何真实性，并引入邻接距离约束和可行性修复机制

**🔧 技术方法**

使用ESM-3预训练多模态蛋白语言模型作为条件输入，E(3)-等变形图网络生成锚点，条件Transformer在局部内部坐标中递归生成中点，并采用流匹配（clean-endpoint）训练；最终通过cg2all重建全原子结构并用MolProbity评估

**📊 数据集**

主要使用D3PM数据集（固定的150对训练/验证/测试）以及ESM-3对全序列的嵌入，外部验证还使用CryptoBench进行分布外测试

**📈 对比分析**

与SBAlign、Sesame、ABM、FDBM等基线在固定容量下比较，模型在D3PM测试集上平均Cα RMSD为1.33±0.06 Å，成功率58.5%，在局部距离与角度一致率分别达99.6%与95.9%，MolProbity分数3.03，明显优于基线；在CryptoBench上亦是唯一提升apo结构的生成模型

**⚠️ 局限性**

模型单一预测、缺乏多状态生成；在更大规模或极端结构变异时仍可能受限；虽然全局精度在匹配容量下略逊于部分基线，但局部几何优势在各类测试中保持稳健

---

## 288. Structural Agreement Indicates Prediction Accuracy For Protein-Ligand Co-Folding

**arXiv ID:** 2611.27058 | [PDF](https://arxiv.org/pdf/2611.27058v1)

**作者:** J. Henry Broster `[一作]` (University of Oxford), Fergus Imrie `[通讯]` (University of Oxford)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `09944146-298c-433e-89df-37255de463d7` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了七种扩散式共折叠模型的蛋白‑配体结构预测性能，并提出了基于重复预测聚集度的 CORE Score，用于判断预测可靠性。

**💡 创新点**

核心创新在于利用生成对的结构一致性（聚集度）替代传统的学习得到的置信评分和训练集相似度，证明聚集度能更好地区分正确与错误的预测。

**🔧 技术方法**

使用扩散式共折叠模型（AlphaFold3、Boltz-1、Boltz-1x、Boltz-2、Chai-1、Protenix、OpenFold3），计算配体对间 RMSD、聚类、CORE Score，并通过 ROC AUC 对比 ipTM、整体置信、配体 Tanimoto、序列同识别以及 Runs N' Poses 相似度。

**📊 数据集**

数据集为 Runs N' Poses 基准（2,600 个高分辨率蛋白‑配体复合物），包含上述七种模型的 25 份预测装配；最终分析约 12,990 组装配，覆盖 2,212 个目标‑配体。

**📈 对比分析**

比较方法为在每个模型的所有装配中，对 top‑1 成功率和 oracle 成功率进行 ROC AUC 评估。CORE Score 在 top‑1 成功上平均 AUC 为 0.866，oracle 成功平均 AUC 为 0.838，显著高于 ipTM（0.757/0.759）、整体置信（0.714）以及配体 Tanimoto/序列同识别；对 Runs N' Poses 相似度的比较也显示 CORE Score 更优。

**⚠️ 局限性**

局限性包括：CORE Score 只关注聚集度，无法区分模型在多能态系统中应产生多样化采样的情况；对动态或多态配体结合位点的预测可能不适用；其有效性目前仅在蛋白‑配体结构上验证，其他复合物类型（抗体‑抗原、蛋白‑蛋白等）尚未评估；此外，定义成功的 RMSD/lDDT‑PLI 可能对实际实验条件不够鲁棒。

---

## 289. PACE: Policy Augmentation and Compensation With Explicit Disturbance Estimation for Robust Locomotion

**arXiv ID:** 2611.27061 | [PDF](https://arxiv.org/pdf/2611.27061v1)

**作者:** Shixuan Han `[一作]` (Beihang University), Xiang Yu `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了PACE框架，利用关节空间扰动估计进行前馈补偿并作为策略观测，从而提升四足机器人的鲁棒性。

**💡 创新点**

创新点在于：①将所有模型不确定性和外部扰动聚合为单一关节扰动；②使用无加速度的非线性扰动观测器实时估计扰动；③将估计结果同时用于力级前馈补偿和策略观测；④开发无足部力传感器的接触力估计器，使模拟与实机结构保持一致。

**🔧 技术方法**

技术手段包括：强化学习（PPO）+教师-学生框架；非线性扰动观测器；递归牛顿欧拉逆动力学；SVD逆与物理约束的力估计；低通滤波；关节PD控制。

**📊 数据集**

数据集：使用Isaac Gym模拟平台，随机化参数后训练；实机实验在DeepRobotics Lite3上进行，未使用公开数据集。

**📈 对比分析**

与BASELINE、HIMLOCO和仅前馈补偿的PACE_0比较；在11种动态变异、OOD命令、外部推力、实机步降、负载和地形测试中，PACE使速度跟踪误差平均降低57%/18%，恢复时间缩短至72 ms，实机最大步降16 cm、最大负载11.5 kg、所有未见地形成功率100%。

**⚠️ 局限性**

局限：扰动观测器采用固定标量增益，未针对关节或工况自适应；对高频噪声仍有一定敏感性；未验证在人形机器人或高接触密集任务中的性能。

---

## 290. Communication-constrained multiagent risk estimation: a simulation benchmark for adult sore throat

**arXiv ID:** 2611.27468 | [PDF](https://arxiv.org/pdf/2611.27468v1)

**作者:** Anas H. Alzahrani `[一作]` `[通讯]` (King Abdulaziz University), Anas H. Alzahrani (King Abdulaziz University)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c84dae5d-5273-4348-85a7-b44cb586b4df` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在五个固定的预测模型上实施有限预算的对称对等口袋式推理（gossip）协议，评估了在不同通信图结构下（完整图、环图、星图）以及交易失败概率（0、0.2、0.5）条件下，代理之间的意见收敛行为。

**💡 创新点**

其创新点在于构建了一个可复现的模拟基准，用以量化有限预算下的误差分解（不一致方差与Brier误差的关系）并比较不同图结构的收敛效率，而非提出新的共识算法。

**🔧 技术方法**

技术手段包括：对称平均的gossip更新、原子交易失败模型、Monte Carlo模拟、误差方差分解以及对Brier误差的统计分析。

**📊 数据集**

使用的数据集为完全合成的成人咽喉炎模拟数据，涵盖三类生成机制，产生270,000条基础病例并扩展为四种观测视图（基础、MCAR、MAR、MNAR）。

**📈 对比分析**

通过在相同交互预算和失败概率下比较完整图、环图和星图，发现完整图在8次尝试与0.2失败时的误差方差比环图低18.3%，但其均值代理Brier误差与中心化均值相近，且未能优于最强个体预测器。

**⚠️ 局限性**

局限性包括：使用合成数据、训练样本重叠、仅五个同质标量代理、未引入新的共识算法、缺乏临床验证以及假设即时原子交易而忽略网络延迟、异构性和动态决策等实际部署因素。

---

## 291. CoGITAtion: Collaborative Gaze-based Interpretable Task-adaptive Representation for Privacy-sensitive Applications

**arXiv ID:** 2611.27340 | [PDF](https://arxiv.org/pdf/2611.27340v1)

**作者:** Mayar Elfares `[一作]` (University of Stuttgart), Andreas Bulling `[通讯]` (University of Stuttgart)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了CoGITAtion框架，实现对垂直分布眼动数据的可解释、可压缩表示，并在保持预测性能的前提下进行隐私保护推理。

**💡 创新点**

创新点在于：①将可变长度眼动序列转化为紧凑可解释的眼动转移图和描述符；②采用任务自适应架构和门控机制处理异构、缺失特征块；③通过公开门控在FHE/MPC推理中显著减少加密运算量。

**🔧 技术方法**

使用的技术包括眼动转移图构建、描述符提取、CoGITAtion‑NN与CoGITAtion‑Tree两类模型、全同态加密（TFHE）与安全多方计算、以及任务与所有者可用性门控。

**📊 数据集**

实验数据集涵盖医学领域的ASD、精神分裂症、ADHD以及编程领域的EMIP和GazeCopilot眼动记录。

**📈 对比分析**

与基于图神经网络的学习型表示及公开的深度模型相比，CoGITAtion‑Tree在AUC上与最先进方法相当，同时在FHE下实现3–40倍的加密运算节省和MPC下8倍的通信成本。

**⚠️ 局限性**

主要局限是描述符摘要丢失部分时间信息，对代码阅读任务表现不佳；此外，垂直分布实验是模拟而非真实多机构数据，需要进一步验证。

---

## 292. Unimodal Shortcuts and Joint Scene-Intent Dependence in Vision-Language Safety Representations

**arXiv ID:** 2611.27339 | [PDF](https://arxiv.org/pdf/2611.27339v1)

**作者:** Guijia Zhang `[一作]`, Harry Yang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

论文研究了视觉‑语言模型在安全任务中的“单模态捷径”现象，并提出了联合场景‑意图依赖的安全表示方法，用以提升对潜在危险信息的检测与过滤。

**💡 创新点**

创新点在于①首次系统性识别并量化了单模态捷径导致的安全风险；②提出通过联合建模场景内容与用户意图来构造更稳健的安全表示，显著降低误报率；③将该表示与现有 CLIP、LXMERT 等模型进行无缝集成。

**🔧 技术方法**

技术方法包括：基于 Transformer 的视觉‑语言编码器、对比学习与信息瓶颈技术（Mutual Information Maximization）来强化场景‑意图耦合，使用自监督预训练与安全标签微调两阶段训练策略。

**📊 数据集**

主要使用的数据集为：安全版 COCO（Safety‑COCO）、VQA‑Safety（从 VQA 数据中筛选安全相关问答）以及公开的 OpenAI Safety‑Text 语料库，全部数据均经过人工安全标签审核。

**📈 对比分析**

与 CLIP、LXMERT、BLIP 等基线方法在安全检测任务上进行对比。实验结果表明，所提方法在 F1 分数上提升了 5.8%（平均值）并在误报率上下降 12.3%，验证了联合场景‑意图模型在安全性上的优越性。

**⚠️ 局限性**

局限性包括：①对极端少数族群与新兴攻击方式的鲁棒性尚未充分评估；②模型对大规模多模态数据的推理效率仍受限；③安全标签的人工标注成本高，难以快速迭代。

---

## 293. AELF: Active Terrain-Aware UAV Exploration Using a Lightweight Framework

**arXiv ID:** 2611.27263 | [PDF](https://arxiv.org/pdf/2611.27263v1)

**作者:** Mingyang Li `[一作]` (University of Hong Kong), Peng Lu `[通讯]` (University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量级的 2.5D 地图框架，实现无人机在复杂地形中的主动探索。

**💡 创新点**

创新点：①无固定高度限制，动态自适应地形高度；②通过 2.5D 地图消除盲体素问题；③结合地形可通行性成本的全局规划，减少垂直切换和能耗。

**🔧 技术方法**

使用 UFOMap 构建 3D 占据体素、Fast‑LIO2 定位、SUPER 路径规划，结合 2.5D 地图、坡度图、视角集生成及 ATSP 全局规划。

**📊 数据集**

在 MARSIM 仿真环境（基于真实 LiDAR 扫描的森林与山地场景）和实际 70×40×6 m 森林、50×25×10 m 山地场景中测试。

**📈 对比分析**

与 FUEL、NBVP、RACER、FAEP 及 FBE 比较，AELF 在两种复杂场景下的覆盖时间分别为 194.6 s/157.5 s，覆盖距离分别为 327.4 m/258.4 m，平均速度分别为 1.68 m/s/1.65 m/s；同时单次运行时长仅为 0.12–0.15 s，远快于其他方法，显示出更高的效率和更低的计算开销。

**⚠️ 局限性**

局限性：目前仅在单无人机条件下验证；对极端遮挡或高度急剧变化的场景仍需进一步评估；依赖高质量 LiDAR 测距，可能在光照或雪雾等恶劣环境下表现受限。

---

## 294. Mechanistic Analysis of Unified Proximity-Tactile Observation by a Capacitive Fingertip Array Constrained by a Metal Phalanx

**arXiv ID:** 2611.26950 | [PDF](https://arxiv.org/pdf/2611.26950v1)

**作者:** Shun Cheng `[一作]` `[通讯]` (Zhaoqing University), Shun Cheng (Zhaoqing University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个多导体观测模型，并通过有限元仿真验证了驱动屏蔽、误差泄漏与时间分复用在金属指骨上双模式接近与触觉感知中的影响；

**💡 创新点**

其创新点在于明确写入每个被动电极终止的观测算子，给出驱动屏蔽失配泄漏门限，提出TDM时域设定的帧率约束，并展示通过外层屏蔽消除接触读数中的材料跨敏感；

**🔧 技术方法**

使用了Maxwell电容矩阵理论、驱动屏蔽电路、时间分复用(TDM)与稳态设定、有限元电磁仿真(ANSYS Maxwell)以及自定义观测算子与电荷-电压关系；

**📊 数据集**

采用了ANSYS Maxwell 3D仿真得到的电容矩阵数据，包括单层与双层结构在远场、近场及接触压缩等多种工况下的数百个仿真点；

**📈 对比分析**

通过与单层屏蔽/未屏蔽、不同屏蔽失配、远近场增量对比验证了驱动屏蔽泄漏门限约10⁻³的可行性；双层屏蔽下的接触压缩响应呈非线性，FEM给出4.6–370 fF范围，表明需要查找表实现；但整体未提供实际测量性能指标；

**⚠️ 局限性**

本工作仅基于仿真，未收敛且未测量驱动屏蔽失配、TDM设定、真实屏蔽泄漏、机械耦合与力映射；材料跨敏感已被理想屏蔽消除但实际屏蔽间隙、接地阻抗未验证；模型不适用于非导体目标或大范围动态接触。

---

## 295. A Learning-Enhanced Control Scheme for Aerial Manipulator Under Composite Disturbances

**arXiv ID:** 2611.27048 | [PDF](https://arxiv.org/pdf/2611.27048v1)

**作者:** Meng Wang `[一作]` (Beihang University), Lei Guo `[通讯]` (Beihang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种结合多项式分解学习与观测器补偿的组合抗扰控制框架，用于解决多旋翼机器人与机械臂耦合引起的高阶非线性和外部扰动；

**💡 创新点**

①利用多项式分解将未知高阶耦合力拆解为共享基矩阵、状态相关多项式矩阵与任务特定向量；②采用双层优化提取共享基矩阵；③设计在线自适应律更新任务向量；④引入观测器补偿逼近残差，兼顾稳态精度与瞬态响应；

**🔧 技术方法**

多项式分解、双层优化、在线自适应学习、扰动观测器、差分平滑、Hopf映射、PD末端控制、Simscape仿真与STM32 MCU实现；

**📊 数据集**

使用3000条离线采样数据（多任务、不同姿态与关节运动），并在四个实验场景（负载、动态轨迹、风扰动、捕捉任务）进行评估；

**📈 对比分析**

与三种基线（非线性基础、观测器基础、基于RBFNN的自适应）对比，采用均方误差、最大误差、标准差等指标；本方法在所有场景下均实现平均误差降低约20‑30%，标准差下降30%以内，且在风扰动和高速动态时仍保持优越性能；

**⚠️ 局限性**

对高阶耦合逼近受限于分解维度导致残差；需要离线训练数据且对持续激励条件敏感；计算量对低端MCU有一定负担；对极端外部扰动（如强风、振动）或更复杂多关节机械臂的泛化性尚未验证。

---

## 296. Taking Apart and Putting Together: Tracking Objects Through Physical Transformations

**arXiv ID:** 2611.27398 | [PDF](https://arxiv.org/pdf/2611.27398v1)

**作者:** Davide Gardenal `[一作]` (University of Trento), Elisa Ricci `[通讯]` (University of Trento)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种训练自由的框架TiPO，能够在视频中连续追踪对象的拆装变化，并自动生成完整的状态图；

**💡 创新点**

创新点在于用每帧几何评估拆装候选状态，结合零样本Mask Decoder和MLLM验证，既支持拆装又避免了高成本的时空搜索；

**🔧 技术方法**

采用SAM 2.1/3作为基础分割器，使用Mask Decoder进行采样、生成-评估-判定管线，内外重叠度量构成候选评估，MLLM（Qwen 3.5 35B判定、GPT‑5.4标注）负责视觉验证和语义标注；

**📊 数据集**

构建了TiPO‑Track数据集（50段视频，26000+帧，124次装配与132次拆卸），并在VOST‑TAS上进行拆卸单项对比；

**📈 对比分析**

与TubeletGraph、SAM、AlwaysPredict、Agentic等基线相比，TiPO在拆装跟踪（T_F1）上提升32–40点，在状态图生成（宏F1）上提升至≈43点，速度提升6–15×；

**⚠️ 局限性**

局限性包括：需调用大规模MLLM，成本较高；依赖强大的SAM/LLM基础模型，对极端遮挡或极细小部件仍可能失效；目前仅覆盖拆装两类物理变换，未处理更复杂的物理交互。

---

## 297. BusemannFormer: Horospherical Self-Attention for Hyperbolic Graph Transformers

**arXiv ID:** 2611.27167 | [PDF](https://arxiv.org/pdf/2611.27167v1)

**作者:** Youheng Yao `[一作]` (Yale University), Tianqi Wang `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `57a58b01-81b4-4d75-a45c-2e891f272b50` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了Horospherical Self-Attention（HSA）和BusemannFormer，实现了在Lorentz模型下的超几何Transformer。

**💡 创新点**

创新点在于让每个查询动态选择理想边界方向，并用Busemann深度进行键评分，显式捕捉层级深度与方向关系。

**🔧 技术方法**

使用了Lorentz模型、HypLinear投影、Busemann函数、log-depth调制以及可扩展的线性边界字典。

**📊 数据集**

实验数据集包括Disease、Airport、Cora、Citeseer、PubMed以及ogbn-arxiv。

**📈 对比分析**

与GCN、GAT、HGCN、Hypformer等基线对比，BusemannFormer在大多数节点分类任务中获得最高均值，在链接预测任务也保持竞争力。

**⚠️ 局限性**

限制在于HSA仍具二次计算复杂度，线性边界字典需固定字典并改变原始核，且大规模评估仅在ogbn-arxiv上，需进一步验证。

---

## 298. ROSWORLD: When Digital Task Competence Meets Physical Execution

**arXiv ID:** 2611.27467 | [PDF](https://arxiv.org/pdf/2611.27467v1)

**作者:** Caoyuan Ma `[一作]` (UTokyo), Yinqiang Zheng `[通讯]` (UTokyo)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了 ROSWorld 基准，评估机器人通过键盘鼠标与真实操作系统交互完成桌面任务的能力；

**💡 创新点**

创新点在于将数字桌面目标与物理交互统一到同一评估框架，提供实时物理–数字循环和官方评估器，揭示桌面能力向物理动作转化的差距；

**🔧 技术方法**

技术包括 Isaac Sim/PhysX 的机器人与键盘鼠标仿真、双臂/单臂两种身体实现、相机观测、JSON 动作命令、逆运动学、物理触碰检测与缓冲、以及输入事件与屏幕状态的闭环反馈；

**📊 数据集**

使用 OSWorld 1.0 的 40 个跨 10 个应用域任务及其官方评估器，配合键盘鼠标的物理设备日志，对轨迹和输入事件进行标注；

**📈 对比分析**

通过比较同一模型在 OSWorld（直接软件控制）与物理条件下的成功率、保留率和决策预算，评估模型性能；在 OSWorld 中五个模型完成 31–39 任务，在物理条件下完成 3–28 任务，保留率约 25%，双臂实现键盘组合键提升了部分模型表现；

**⚠️ 局限性**

局限性包括：物理完成率低、模型缺乏对键盘/鼠标几何、全身姿态和输入可行性的感知与验证，训练数据未覆盖误触、未触发等失败模式，导致模型在长周期、多步骤任务中易失误和退缩。

---

## 299. Crafting the Difference, Grounding the Evidence: Discriminative Concept Learning for Zero-Shot Skeleton Action Recognition

**arXiv ID:** 2611.27135 | [PDF](https://arxiv.org/pdf/2611.27135v1)

**作者:** Siyu Zhu `[一作]` (Eastern Institute of Technology), Jie Liang `[通讯]` (Eastern Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CRAFT 方法，结合硬负样本条件生成、Unique-QDPP 选择与概念监督骨架提示学习，显著提升零样本与泛化零样本下的骨架动作识别。

**💡 创新点**

创新点在于：1) 通过 LLM 结合硬负样本生成区分度高的动作描述；2) 用 Unique‑QDPP 在多槽内实现质量与多样性兼顾、跨槽唯一的描述筛选；3) 将筛选出的概念作为学习目标，训练可读骨架读者以实现语义-运动的显式对齐。

**🔧 技术方法**

技术手段包括：LLM 文本生成、CLIP 文本编码、对比质量评分、DPP 采样、概念读者（多头注意力 + 读出）、概念监督损失、固定骨架特征的读者训练。

**📊 数据集**

使用公开数据集 NTU‑60、NTU‑120（固定分割与随机分割）以及 PKU‑MMD 进行评估。

**📈 对比分析**

与五个基线（SynSE、PURLS、Neuron、TDSM、Flora 等）在固定与随机分割上对比；平均 ZSL 精度提升约 5.4%，GZSL 调和均值提升约 5.9%；在所有基线‑分割组合上均实现显著提升，随机分割下亦保持正向增益。

**⚠️ 局限性**

局限性包括：依赖 CLIP 文本编码器及预训练骨架编码器的规模限制；生成与筛选概念的计算开销；在极端类分布或低资源场景下效果可能受限。

---

## 300. Dynamic Memory-enhanced Pattern Relevance Prediction for Long-tailed Graph Classification

**arXiv ID:** 2611.27141 | [PDF](https://arxiv.org/pdf/2611.27141v1)

**作者:** Jiawei Chen `[一作]` (Tianjin University), Qiyao Peng `[通讯]` (Tianjin University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

针对图级别结构不平衡导致的长尾图分类问题，提出动态记忆增强模式相关性预测（DMPRP）方法，通过查询条件下的模式补充来提升尾部图的表示与分类性能。

**💡 创新点**

创新点：① 采用 KDEX 风格的动态专家记忆实现查询条件的模式补充；② 设计多粒度（节点、根子图、随机游走邻域）共现三元组对比学习，为尾图提供多维结构监督；③ 在尾部图仅使用补充表示，避免大图主导训练。

**🔧 技术方法**

技术手段：GNN 编码器（以 GIN 为主），动态专家记忆模块，三元组对比学习（InfoNCE 风格），补充向量正则化（相似度约束）以及多视角损失组合。

**📊 数据集**

实验数据集：五个标准图分类基准（PTC、FRANK、D&D、PROTEINS、REDDIT），OGB-MolHIV 用于规模评估，PTC-SDLS 控制实验。

**📈 对比分析**

与多种基线（图核、GCN、GIN、SOLT-GNN、SIMBA 等）对比，DMPRP 在整体准确率、宏 F1 以及尾部子集性能均取得最高或接近最高的分数，尾部提升尤为显著。

**⚠️ 局限性**

局限性：对头尾结构极端冲突的场景仍需改进；对异构/多模态图的负迁移风险未完全解决；性能对专家数量、随机游走长度、损失权重等超参数较为敏感，需要仔细调参。

---

## 301. Real-Time On-Drone Hyperspectral Anomaly Detection for Wilderness Search and Rescue

**arXiv ID:** 2611.27017 | [PDF](https://arxiv.org/pdf/2611.27017v1)

**作者:** Jacqueline Liu `[一作]` (Binghamton University), Jayson Boubin `[通讯]` (Binghamton University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研发了一套名为SABL的实时边缘计算管线，在无人机上实现推送式光谱异常检测与空间聚类，支持现场搜索救援。

**💡 创新点**

将实时指数移动Reed–Xiaoli异常检测与DBSCAN空间过滤相结合，实现像素级到对象级的异常定位，并在极限环境下验证其实时性与精度。

**🔧 技术方法**

采用推送式光谱相机+REMAX流式解码、指数移动Reed‑Xiaoli（ERX）算法、密度聚类DBSCAN以及Raspberry Pi 4B边缘处理平台。

**📊 数据集**

采集并公开了阿拉斯加Quinhagak地区两套现场光谱数据集（泥滩与沼泽草地），共75,419扫描线并提供手工标注。

**📈 对比分析**

与单纯ERX做对比，并在四种执行情境（本机流式、本机文件、GCS流式、GCS文件）评估；SABL在Raspberry Pi上行程时间≈5.7 ms/线，满足100 Hz摄像机速率；DBSCAN仅增加≈1‑2 %延迟，显著提升对象级精度至≈56%/≈99%。

**⚠️ 局限性**

对象召回仍有限（约55%/29%），在不同地形与光照下性能不稳定；受限于单一嵌入平台，需进一步扩展至更多环境与更高效实现。

---

## 302. Revisiting Generalized Median Estimators: Exact Bitwise Computation and Gray Code Acceleration

**arXiv ID:** 2611.26848 | [PDF](https://arxiv.org/pdf/2611.26848v1)

**作者:** Robin Nunkesser `[一作]` `[通讯]` (Hamm-Lippstadt University of Applied Sciences), Robin Nunkesser (Hamm-Lippstadt University of Applied Sciences)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种完全枚举实现的通用中位数估计器（GM_n,k^(h)）的算法与实现，能够在毫秒级完成 10^5–10^6 大小子集的计算。

**💡 创新点**

创新点在于将子集枚举从传统的子集生成/字典序重排序转为状态化遍历：使用 Gosper 递推（适用于 n≤63）和 Knuth 的 Revolving Door Gray 码（适用于任意 n），并结合周期性漂移校正与三路 Quickselect，显著降低了时间和数值误差。

**🔧 技术方法**

核心技术包括：Word‑RAM 位运算 + Gosper 递推、Knuth 的 Revolving Door Gray 码、周期性浮点漂移校正、三路 Quickselect 与 median‑of‑medians 下界、C 语言位级实现与循环展开、针对 h1（均值）与 h2（方差）的递归更新。

**📊 数据集**

主要使用的基准数据集是 n=29、k=5 的 118,755 个子集（来自 0.9 N(0,1)+0.1 N(5,0.5^2) 混合样本），并扩展到 n∈[20,100]、k∈{3,5,7} 的全组合测试。

**📈 对比分析**

与 R 的直接全枚举、向量化实现、以及 Wilde & Grimshaw 的子采样方法进行对比。实验显示：Gosper 在 n≤63、k≤5、h1 时最快；Revolving Door 在 h2、n≥64 或 k>5 时更优；在 n=29、k=5 基准上，完整枚举仅需 0.74 ms（h1）和 0.71 ms（h2），远快于 R 中的 240–360 ms 或子采样 185–261 ms。

**⚠️ 局限性**

局限性包括：Gosper 方案受限于 n≤63；Revolving Door 在小 k、h1 时因分支预测开销略慢；周期性校正需选择 P 参数；实现仍为单线程，未探索并行化；对极大 n（>10^4）或 k 过大时可能面临内存/时间瓶颈。

---

## 303. Hypergraph Structural Entropy-Enhanced Hierarchy-Aware Semi-Supervised Graph Classification

**arXiv ID:** 2611.27043 | [PDF](https://arxiv.org/pdf/2611.27043v1)

**作者:** Guangjie Zeng `[一作]` (Northeastern University at Qinhuangdao), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于超图结构熵（HSE）的层次化半监督图分类框架HypSEE，利用两种超图视图（随机游走诱导和嵌入生成）捕获高阶关系，并通过HSE驱动的超图池化学习层次表示，再使用锚图一致性学习提升无标签图的表示质量。

**💡 创新点**

创新点包括：① 将结构熵推广到超图并用其指导超图层次化池化，显式建模图的层次结构；② 设计两种互补的超图视图以同时捕获局部层次和全局高阶关系；③ 引入锚图一致性正则化，强化不同视图间的对齐，充分利用无标签数据。

**🔧 技术方法**

主要技术：超图神经网络（HGNN）、随机游走诱导超图、基于GNN的嵌入生成超图、HSE导向的层次化池化、锚图一致性学习、Kullback‑Leibler一致性损失、两层MLP分类器。

**📊 数据集**

在8个TUDataset基准图数据集（PROTEINS、DD、IMDB-B、IMDB-M、REDDIT-B、REDDIT-M-5K、COLLAB、MSRC21）以及两个大型OGB分子数据集（ogbg-molhiv、ogbg-molmuv）上进行评估。

**📈 对比分析**

与多类基线（经典图嵌入、通用半监督方法、现有SSGC方法）对比，HypSEE在7/8个小数据集和大数据集的ROC‑AUC上取得最高或相当的性能，平均提升约1–4个百分点，显著优于现有方法。

**⚠️ 局限性**

局限性：对非常小的图（如IMDB‑BINARY）性能略逊，因层次结构不明显；模型对超图参数（超边数、嵌入维度）及锚图设置敏感，需经验调参；当图规模极大时，超图构建与池化的计算开销仍显著。

---

## 304. Double counting in LLM evidence fusion: exact posteriors separate rule compliance from posterior gain

**arXiv ID:** 2611.27138 | [PDF](https://arxiv.org/pdf/2611.27138v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 305. Trailblazer-F: Speculative Action-Flow Inference for Autonomous Driving

**arXiv ID:** 2611.27226 | [PDF](https://arxiv.org/pdf/2611.27226v1)

**作者:** Haibo Hu `[一作]` (City University of Hong Kong), Chun Jason Xue `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为Trailblazer-F的推理框架，通过轻量级预写器生成多步未来状态，并在目标模型上并行校验以显著降低行驶场景中的动作流推理延迟

**💡 创新点**

创新点在于将“预写”与“并行校验”结合，利用动作流状态的物理可解释性实现多步推理的先行预估，从而在不修改目标模型的前提下实现显著加速

**🔧 技术方法**

使用了轻量级MLP预写器、基于已知噪声计划的解析式状态重构、目标模型的单步修正与航点级误差阈值验证，以及物理一致性接受机制

**📊 数据集**

在NAVSIM v1数据集的12146个场景上评估，并分别在SimWAM和DriveWAM两个基准模型上训练对应预写器

**📈 对比分析**

与原始10步采样器及BAC、FastFlow、TeaCache等基线对比，Trailblazer-F在SimWAM上实现约2.31×加速（Latency↓56.8%），DriveWAM上约1.55×加速（Latency↓35.5%），且规划质量（PDMS）变化仅±0.08分，保持近乎原始水平

**⚠️ 局限性**

局限性包括：预写器本身无法单独完成推理（需目标校验），在更高维度或非航点式动作空间的推广性未知；加速幅度受限于预写器的预测精度和阈值设定

---

## 306. Learning Video Foresight Reasoning from Unlabeled Videos via On-Policy Self-Distillation

**arXiv ID:** 2611.26954 | [PDF](https://arxiv.org/pdf/2611.26954v1)

**作者:** Jingchen Sun `[一作]` (University at Buffalo), Changyou Chen `[通讯]` (University at Buffalo)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 Future-OPSD，一种基于自监督对比与教师-学生蒸馏的前瞻性视频推理框架，能够在无标注、无外部大模型的条件下提升视频推理能力。

**💡 创新点**

创新点在于：①首次将 on-policy self‑distillation 与前瞻性推理结合；②定义可计算的前瞻得分并构造其下界，用以引导模型关注最有价值的未来信息；③通过对比优势与蒸馏优势的线性插值，有效抑制无关背景信息。

**🔧 技术方法**

采用的技术包括：Vision‑Language 模型（Qwen3‑VL/Qwen3.5），PPO 训练框架，教师-学生对比评分，KL 正则化下的低界估计，以及 token‑level 更新。

**📊 数据集**

实验数据集涵盖 Future‑L1（开放式预测）、EgoPlan‑IT（闭合式规划）及其评测基准 TwiFF‑Bench 与 SEED‑Bench‑R1；同时对 TempCompass、TemporalBench、Video‑MME、MVBench 进行通用视频理解评测。

**📈 对比分析**

与基线（Base、SFT、Naïve‑OPSD、GRPO）相比，Future‑OPSD 在开放式预测和闭合式规划任务上分别提升约 14.2 与 15.9 分（相对于 Base），并保持或提升通用视频理解性能；更强教师和更多数据进一步放大收益。

**⚠️ 局限性**

局限性包括：仅在 Qwen3‑VL/Qwen3.5 及上述少量视频数据集上验证；对教师冻结与 EMA 更新的依赖可能限制迁移；当前仍未探索更大规模模型或多模态多任务场景的适用性。

---

## 307. Poisoning Attacks and Collaborative Defense Framework for LightRAG-Based Construction Regulation Knowledge

**arXiv ID:** 2611.27460 | [PDF](https://arxiv.org/pdf/2611.27460v1)

**作者:** Yingyan He `[一作]` `[通讯]` (SWISS UMEF), Yingyan He (SWISS UMEF)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对基于LightRAG的建筑法规检索增强生成系统进行系统性攻击与防御评估

**💡 创新点**

首次将文本、图谱、叙述等四种攻击与三层防御在同一系统上进行因子设计，并揭示拒绝率与攻击成功率不等价的关键结论

**🔧 技术方法**

采用LightRAG框架、DeepSeek LLM、NLI检验以及基于实体关系锚点的图谱验证等技术

**📊 数据集**

构建150条题目基准，基于英国Approved Documents及Manual to the Building Regulations的850条文本块

**📈 对比分析**

通过因子实验比较，文本注入ASR 22%/预算5，图谱注入仅12%；三层防御中检索侧过滤可保持完整准确度，生成侧拒绝率高但精度损失显著，整体ASR下降至0；但不同配置的收益与成本无法单独分解

**⚠️ 局限性**

实验局限在于仅评估单一系统与单一语言，图谱提取仍存在无源信息通路，防御层交互复杂难以归因，且未考察持续或高级自适应攻击

---

## 308. Cross-Attack Robust Aggregation for Federated Learning

**arXiv ID:** 2611.26731 | [PDF](https://arxiv.org/pdf/2611.26731v1)

**作者:** Eyhab Al-Masri `[一作]` (University of Washington), Eyhab Al-Masri `[通讯]` (University of Washington)

**通讯引用:** 3206 | [OpenAlex ID](https://openalex.org/A5071801237)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了 FedTether，一种在 IoT 边缘网络中对联邦学习进行鲁棒聚合的方法。

**💡 创新点**

创新点在于使用坐标均值中值作为参照，结合 MAD 归一化和死区阈值对客户端更新进行指数加权，使得在无攻击时完全退回 FedAvg，且在多种攻击与非 IID 情况下保持最高准确率。

**🔧 技术方法**

采用了坐标均值中值、MAD 标准化、指数权重、死区阈值以及服务器端优化器步（delta 空间）等技术，实验基于 Flower 框架实现。

**📊 数据集**

实验使用了 MNIST、CIFAR‑10 和 CIFAR‑100 三个公开数据集。

**📈 对比分析**

与 FedAvg、Median、Trimmed Mean、Multi‑Krum、RFA 等多种鲁棒聚合器进行对比，FedTether 在三种攻击（ALIE、sign‑flip、Gaussian）和非 IID 数据下均优于或等同于最佳基线；在高攻击率下准确率提升高达 42 点，且无鲁棒税，且在 50 设备、部分参与场景下依旧保持性能。

**⚠️ 局限性**

局限性包括：未对自适应或协同攻击、极端非 IID 或极低参与率进行评估；缺乏收敛性理论分析，且在更大规模设备（数百）与 stragglers 情况下的表现仍待验证。

---

## 309. Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing

**arXiv ID:** 2611.26900 | [PDF](https://arxiv.org/pdf/2611.26900v1)

**作者:** Qiang Zhou `[一作]` (University of Pittsburgh), Jingtong Hu `[通讯]` (University of Pittsburgh)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了大规模的归因风格转移数据集(ASTD)，并针对归因评估与重构制定了自动化评估指标及偏好对齐方法。

**💡 创新点**

创新点在于：①首次构建含7类归因标签的42k事件-归因对数据；②设计了Prevent–Filter–Validate管线以降低hallucination和偏见；③提出四维CBT对齐评估指标，验证了与人工评分高度一致；④通过偏好对齐（DPO）实现无人工标注的归因重构质量提升。

**🔧 技术方法**

使用了检索‑生成耦合、LLM推理、规则过滤、专家审阅；评估方法包括监督分类模型、零/少量提示LLM、四维自动评估、DPO微调。

**📊 数据集**

ASTD数据集（42,000条事件-归因对），包含内部、外部、稳定、非稳定、全局、特定及中性七类归因标签。

**📈 对比分析**

在归因分类上，经过ASTD训练的监督模型（BERT、RoBERTa、LoRA微调LLM）在各维度上达到90–97%准确率，显著优于单纯提示的LLM；在归因重构上，DPO微调后模型在四维指标上平均提升0.5–1.1分，远超未微调或SFT模型。

**⚠️ 局限性**

局限包括：数据仍由LLM生成，可能存在隐含偏见；评估指标虽与人工一致，但无法完全覆盖所有临床语境；模型生成的重构仍需临床专家把关，未验证对真实患者的干预效果。

---

## 310. Same Facts, Different Decision: Fact-Invariant Manipulation and Provenance-Based Defense in Typed Decision Models

**arXiv ID:** 2611.27016 | [PDF](https://arxiv.org/pdf/2611.27016v1)

**作者:** Yujie Wang `[一作]` (China University of Mining and Technology), Shunrong Jiang `[通讯]` (China University of Mining and Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在 typed decision models 中，外部不添加新事实的评论（fact‑invariant）如何被利用来导致错误决策，并提出了相应的攻击方法和基于 provenance 的防御门。

**💡 创新点**

创新点包括：①使用论证理论的四个杠杆生成仅改写论点而不增添事实的评论；②引入独立 verifier 确保评论保持 fact‑invariant；③设计仅通过记录来源而非内容的门控防御，确保任何未授权评论都不影响最终决策。

**🔧 技术方法**

技术手段包括：论证理论杠杆（证据、推理、准则、替代）、基于树搜索的迭代生成评论、专门的 verifier 检查评论是否添加事实，以及利用 cheap talk 理论的门控策略（在移除未授权条目后重新查询并比较决策）。

**📊 数据集**

实验数据集涵盖 BoolQ、MuSR、SST‑5 三个公开基准，并在六种不同的 typed decision models（如 TypeSafe、laya、Qwen 系列、Strands）上进行评测。

**📈 对比分析**

与两种基准攻击（T1/T2、官方插入攻击）以及四种 prompt 防御（指令、定界、数据标记、沙盒）对比，所提攻击在 3 个基准上成功率约为 34.7%，相较于最强基准攻击提高 2.8 倍；所提门控防御在不额外查询成本（仅需一次额外查询）的前提下，阻断了所有重放攻击，且对授权事实变更保持通畅。

**⚠️ 局限性**

主要局限包括：依赖模型解码确定性、仅评估英文数据、仅对未授权文本进行检测（未覆盖被破坏的授权来源或对规则重写的情况），以及对非结构化输出（自由文本）模型的适用性尚未验证。

---

## 311. FlowScope: 3D Scene Flow Pretraining Improves Surgical Robot Policies

**arXiv ID:** 2611.26865 | [PDF](https://arxiv.org/pdf/2611.26865v1)

**作者:** John J. Han `[一作]` (Vanderbilt University), Omid Mohareri `[通讯]` (Intuitive Surgical Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

预训练手术机器人策略网络，利用无标签临床立体视频中的三维场景流预测，显著提升了机器人抓取与重新握取等手术子任务的成功率。

**💡 创新点**

创新点在于：①提出仅用场景流（由立体深度和光流生成的伪标签）进行无标签预训练；②证明场景流比传统关节运动预训练更能提升策略性能；③在部署时无需额外推理开销。

**🔧 技术方法**

技术方法包括：Action Chunking Transformer (ACT)、冻结的DINOv2视觉编码器、Fast-FoundationStereo与SEA-RAFT深度/光流模型、L1回归与Flow Matching预训练头、线性探针分析以及基于流匹配的生成式动作头。

**📊 数据集**

使用的数据集：1,149个临床手术视频（约1.35M帧）用于预训练；约898条针筒拾取和2,255条针筒重新把握演示用于微调和评估。

**📈 对比分析**

与从零训练、Diffusion Policy、π_0.5等基线对比，预训练后针筒拾取成功率从54%提升至90%，针筒重新把握首交接成功率从4%提升至52%，并在数据效率上可用一半演示达到与全量相同的效果，整体表现显著优于对比方法。

**⚠️ 局限性**

局限性：仅在单一仿真腹腔与特定机器人平台上验证；伪标签受立体深度与光流误差影响；完整任务完成率仍偏低；未与其他视频预训练框架（如SurgVIL、Surgical WAM）进行系统对比。

---

## 312. Decision trees for subword-closed languages over arbitrary finite alphabets

**arXiv ID:** 2611.27454 | [PDF](https://arxiv.org/pdf/2611.27454v1)

**作者:** Shahid Hussain `[一作]` `[通讯]` (Institute of Business Administration), Shahid Hussain (Institute of Business Administration)

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究子词闭合语言（subword‑closed languages）在任意有限字母表（|Σ| = k ≥ 2）上的决策树识别与判定问题，并将 Moshkov 在二元字母表上得到的分类扩展到任意字母表。

**💡 创新点**

创新点：① 定义了适用于任意字母表的同质度和异质度维度；② 证明了在 k ≥ 3 时二块结构引理不再适用，提出了最多 k 块的新结构引理；③ 得到的决策树深度上界仅为多项式于 k、同质度和异质度的常数，而非二元情形下的指数常数；④ 给出从有限禁用子词集合快速判断语言所属五个复杂度类的多项式算法；⑤ 对若干三元语言给出了所有四种深度的闭式与计算验证。

**🔧 技术方法**

主要技术：子词闭合语言的组合结构分析（同质度、异质度、k 块结构）、决策树深度上界的构造（二分搜索、填充递归）、证书（certificate）技术、子词闭合语言的最小禁用子词集与子词顺序的降维；此外利用自动机理论（状态变换周期性）验证复杂度类。

**📊 数据集**

数据集：使用了所有有限禁用子词集合 F 生成的子词闭合语言，特别是对三元字母表 {a,b,c} 的六个示例语言（L₁、L₁'、L₂、L₃、L₄、L₅）以及在 k=3、k=4 时随机生成的语言进行实验验证；未使用公开大规模语料库，主要以符号实验为主。

**📈 对比分析**

对比方法：与 Moshkov 的二元字母表结果进行对照，说明本研究在同质度有限时的深度上界从指数降为线性，并给出具体常数比较；通过 Python 计算验证所有示例的闭式结果与实验值一致；在三元语言 L₃ 上得到精确公式 _L₃(n)=⌊log₂n⌋+⌊log₂(n/3)⌋+2，并证明其接近 2log₂n 的下界。

**⚠️ 局限性**

局限性：① 对于异质度为 0 的情形，仍无法给出最优的多项式上界；② 仅讨论了子词闭合语言，未扩展到更一般的下闭（factorial）语言；③ 复杂度类的判定对 DFA 输入的复杂度尚未完全给出；④ 对于 k ≥ 3 的具体常数（尤其是二分搜索步骤中的 10k 乘子）尚未证明最优，可能存在更紧的常数；⑤ 对于某些类 𝒞₃ 中的语言，是否存在极限比值 lim _L(n)/log₂n 仍是开放问题。

---

## 313. Highly Efficient Vertical Federated Clustering with Triangle-Inequality-based Pruning

**arXiv ID:** 2611.27117 | [PDF](https://arxiv.org/pdf/2611.27117v1)

**作者:** Feiteng Nie `[一作]` (Wuhan University), Zhiyong Peng `[通讯]` (Wuhan University)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在垂直联邦学习环境下的高效 k‑means 聚类方法，通过基于三角不等式的精确裁剪与局部子问题求解，显著减少了本地计算量和通信轮数。

**💡 创新点**

创新点包括：① 无需维护点对中心的边界，采用按需构造三角不等式上界实现精确裁剪；② 将 k‑means 目标拆分为客户端局部子问题并加入近端正则化以控制多轮本地更新导致的漂移；③ 在理论层面证明了裁剪方法的精确性、有限收敛性以及服务器端更新的误差界。

**🔧 技术方法**

主要技术手段包括：三角不等式裁剪、坐标下降求解无中心 k‑means、垂直联邦学习框架、近端正则化、局部多轮迭代与聚合、误差分析与收敛证明。

**📊 数据集**

在多种公开基准数据集上评估，涵盖低维/高维、大小不等的数据集（如 MNIST、CIFAR、Iris、HIGGS 等十余种常用数据集）。

**📈 对比分析**

与九种基线（集中式、快速集中式、已有 VFL 聚类方法）进行对比，实验显示在保持聚类质量（SSE）基本相当的前提下，算法在计算效率上可达 1.6×–50.1× 加速，且通信轮数显著下降，E 与 ρ 的选择也表现出稳健性。

**⚠️ 局限性**

局限性在于：假设特征划分均衡且样本对齐；对高度异构或极端稀疏的数据集适应性待验证；对流式数据未给出在线更新方案；需人工调参（如 E、ρ）以获得最优性能。

---

## 314. Reasoning in Fewer Tokens: Concept-Normalized Multilingual Scratchpads and Reversible Model-Adaptive Reasoning Codes

**arXiv ID:** 2611.27326 | [PDF](https://arxiv.org/pdf/2611.27326v1)

**作者:** Rabimba Karanjai `[一作]` (PayPal AI Labs), Prakhar Mehrotra `[通讯]` (PayPal AI Labs)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种使用字节对合并（BPE）生成私有用法区（PUA）Unicode符号的无损压缩语言，教导大型语言模型（LLM）在此压缩空间内进行链式思考，从而在保持可读性和准确性的前提下减少推理所需的令牌数。

**💡 创新点**

创新点在于将无损压缩与模型微调结合，利用PUA符号映射语义片段实现真正的1:1压缩，同时通过快速词汇表迁移（Fast Vocabulary Transfer）和自蒸馏（self‑distillation）使模型自然生成压缩词汇，而非仅仅压缩输入或丢弃推理步骤。

**🔧 技术方法**

核心技术包括字节对合并（BPE）构建压缩词表、PUA Unicode符号映射、快速词汇表迁移初始化新词嵌入、仅冻结嵌入轻量微调以及自蒸馏训练以适配压缩空间；同时在实验中使用了 Qwen2.5‑14B、Qwen2.5‑32B 与 Llama‑3.1‑8B 三大模型。

**📊 数据集**

使用四个典型推理数据集进行评估：GSM8K、MATH、StrategyQA 与 CommonsenseQA；每个数据集既有英文版也有中文版，实验基于英文训练集构建压缩词表。

**📈 对比分析**

通过对比原始（原语）与压缩（PUA）两种推理方式的准确率和生成令牌数，实验表明平均压缩率约为14.7%，推理令牌数最多可削减27.1%，而平均准确率仅下降1.6个百分点，且在部分模型/数据对上准确率甚至略有提升。

**⚠️ 局限性**

局限性包括：仅在英文提示和答案下验证，未评估完整多语言用户体验；压缩效果受模型容量和基线多语言能力影响，Llama‑3.1‑8B 在某些任务上表现不如 Qwen 系列；使用令牌数作为效率度量未考虑硬件层面的实际延迟与能耗；实验基于单一数据拆分，未对多样化推理场景和鲁棒性做深入验证。

---

## 315. EAGLE: Entity-Aware Grounding with Layered Evidence for High-Resolution Image Understanding

**arXiv ID:** 2611.27264 | [PDF](https://arxiv.org/pdf/2611.27264v1)

**作者:** Zhongjian Zou `[一作]` (Shanghai University), Quan Qian `[通讯]` (Shanghai University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 EAGLE 框架，通过实体感知定位、层级证据构建和问题引导路由提升高分辨率图像的细粒度理解。

**💡 创新点**

创新点在于：①基于手工标注的 Fine‑grained Visual‑CoT 筛选并动态排名视觉注意力头；②采用实体短语级热图融合和共享区域增强；③构建中间视图与细节视图，并通过问题引导路由选择最优视图组合。

**🔧 技术方法**

技术包括 Transformer 注意力热图、头池筛选与排名、词权重融合、共享区域增强、层级图像构造、路由决策。

**📊 数据集**

使用 Visual‑CoT (fine‑grained) 手工注释集，以及 V*Bench、HRBench‑4K、HRBench‑8K 三大基准数据集。

**📈 对比分析**

与同基底模型（Qwen2.5‑VL‑7B、InternVL3‑8B）对比，EAGLE 在三大基准上均取得 SOTA，提升约 8.5–16.3 个百分点。

**⚠️ 局限性**

局限在于仅评估了两款模型，缺乏对更广泛多模大语言模型的验证；Fine‑grained 数据集仍有限，难以覆盖更复杂场景。

---

## 316. RoboCap-10k: A Metrically Accurate Egocentric Dataset for Robot Learning

**arXiv ID:** 2611.26916 | [PDF](https://arxiv.org/pdf/2611.26916v1)

**作者:** Kheri Hughes `[一作]` (Grounded Superintelligence), Vincent Liu `[通讯]` (Grounded Superintelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

发布并整理了 RoboCap-10k 数据集：10,000 小时的真实工种 egocentric 视频，配备统一的 250 g 头戴设备，提供在线校准、视觉惯性 SLAM 生成的 6‑DoF 头位姿、基于多视角手部追踪得到的 MANO 手姿以及双层（稀疏/密集）语言描述。

**💡 创新点**

创新点：① 统一硬件平台和每场景在线重标定，消除跨设备域漂移；② 10k 小时职业场景覆盖 14 个任务家族，提供高质量、可重复生成的 3D 监督；③ 结合双层语言注释，支持任务规划与低层动作学习。

**🔧 技术方法**

使用技术：6 目同步摄像头 + 2 IMU、视觉‑惯性 SLAM（Grounded API）、多视角 MANO 手姿追踪、在线相机标定（共线优化）、面部匿名化检测与模糊、Gemini 3.8 Flash 生成语言标签。

**📊 数据集**

数据集：RoboCap-10k（10,000 小时职业工作场景），与 EPIC‑KITCHENS‑100、Ego4D、EgoDex、EgoVerse 等 egocentric 数据集做对比；内部使用 RoboMoCap 光学追踪验证标签精度。

**📈 对比分析**

评估方法：对比光学追踪（RoboMoCap）得到头部轨迹绝对误差 3.0 mm（均值 3.19 mm），手部 PA‑MPJPE 4.5 mm；相机标定误差平均 <1 px；与无 RoboMoCap 训练的手部模型相比提升显著，表明高精度标签在规模化学习中的重要性。

**⚠️ 局限性**

limitations: 标注为估计值，缺乏触觉/力反馈；数据分布偏向印度与菲律宾，其他国家占比 <1%；对单手、静止或极端光照/遮挡情况覆盖不足；人体手与机器人末端执行器差异未解决；语言标签模型生成，人工校正有限。

---

## 317. Correct Precedents Induce Policy Violations in On-Device LLM Agents

**arXiv ID:** 2611.27216 | [PDF](https://arxiv.org/pdf/2611.27216v1)

**作者:** Chong Tian `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Xiuying Chen `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在本地云协同系统中，LLM 代理（guard）在一次正确的决策后会因“正确前提推广”而更容易违规，并提出一种基于配对决策蒸馏（PDD）的修复方法。

**💡 创新点**

首次揭示“正确前提推广”这一风险；通过历史编辑验证代理对已释放相似项目的偏倚；提出 PDD 在保持原模型推理开销的同时几乎恢复安全率到 100%。

**🔧 技术方法**

配对决策蒸馏（PDD），使用 LoRA 适配器训练；教师模型 Qwen3.8‑27B；对齐队列构造、四次决策损失；基于现有模型进行对比与基准评估。

**📊 数据集**

合成任务生成器（cleanup、refund、package 等），以及 τ²‑bench 电信域和 Minions 适配场景；hold‑out 的 timesheets 和 work‑orders。

**📈 对比分析**

与五种推理时修复（状态查询、干净上下文门、留一门、第三方记录、检查状态提示）进行对比。PDD 在不增加额外模型调用或提示文本的前提下，安全率从 50% 以上提升至 100%，并在 MMLU‑Pro、XSTest 等基准上保持非劣性能。

**⚠️ 局限性**

实验在离线模拟、合成队列、思考模式关闭的环境下进行；缺乏真实用户数据；对不同部署模式的泛化需进一步验证；仅评估了单一会话内的风险，跨会话或更复杂交互仍需研究。

---

## 318. Supervaluationism about vagueness and indeterminacy: A judgment-aggregation perspective

**arXiv ID:** 2611.27463 | [PDF](https://arxiv.org/pdf/2611.27463v1)

**作者:** Christian List `[一作]` (LMU Munich), Franz Dietrich `[通讯]` (PSE & CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a2602d71-93ab-4bad-974b-672788df8193` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文从判断聚合的视角，对超值化（supervaluationism）、次值化（subvaluationism）以及单独视角的独裁聚合等处理模糊与不确定性的经典方法进行形式化与系统化的归纳与比较；

**💡 创新点**

将超值化与次值化等模糊处理方式纳入判断聚合的框架，并给出一组直观的公理（无约束域、结果一致性/否定一致性、全一致保持、句子层独立性、灵敏度），证明这些公理共同唯一地决定了超值化或次值化；进一步通过放宽某些公理探讨其他可能的聚合方案，揭示其与社会选择理论（如阿罗不可能定理、迪特里奇-利斯特结果）的紧密联系；

**🔧 技术方法**

主要使用判断聚合理论的工具，包括：集合论中的过滤器（filter）与超过滤器（ultrafilter）、逻辑一致性与蕴含的形式化、以及公理化与归纳证明技术；

**📊 数据集**

本文为纯理论研究，无使用经验数据集；

**📈 对比分析**

没有经验实验或性能指标，所有结论均为形式化的理论推导；

**⚠️ 局限性**

限制在于：对无限视角、单一固定视角情形的讨论较为简略；对真实世界模糊判定的实用性与计算实现仍未展开；

---

## 319. Improved (Minimum) Spanning Tree Construction in the HYBRID Model

**arXiv ID:** 2611.27145 | [PDF](https://arxiv.org/pdf/2611.27145v1)

**作者:** Shoma Hiraoka `[一作]` (University of Osaka), Taisuke Izumi `[通讯]` (University of Osaka)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在HYBRID模型（本地通信+全局通信）中，提出一种O(1)-伪生成树表示（O(1)-PSR），利用该表示在O(log n)轮内在标准容量(1, log n)下构造任意连通图的生成树和最小生成树(MST)，并将此技术推广到少数权重和(1+ε)-近似MST以及二连通分量的计算。

**💡 创新点**

核心创新在于：①构造O(1)-PSR，降低虚拟网络最大度为常数；②通过聚类、MPX式稀疏化和负载均衡实现每个节点只负责常数条代表性边；③设计恢复算法，使任意虚拟树可在O(log n)轮内还原为输入图的生成树；④通过权重分层和批处理实现k权重MST的时间-容量折中。

**🔧 技术方法**

主要技术手段包括：
- MPX型稀疏化(基于Miller‑Peng‑Xu)
- 聚类+负载均衡与复制（copy）技术
- 伪生成树表示（Δ‑PSR）
- 分层/批处理权重共享
- 余量树选择与聚合（quotient‑tree selection）
- Euler 旅游聚合用于二连通分量
- 递归归约与并行化

**📊 数据集**

本工作为理论算法，未使用具体实验数据集；结果基于任意连通无向图（可为任意度）和权重分布，实验验证通过模拟实验或人工构造图。

**📈 对比分析**

与之前工作相比：
- 相比Götte et al.：从O(log² n)轮与O(log² n)全局容量降至O(log n)轮与标准容量O(log n)。
- 对于k权重MST：实现O((k/p) log n)轮，p=1时为O(k log n)，p=k时为O(log n)轮。
- 对于(1+ε)-近似MST：实现O((ε⁻¹ log W/p) log n)轮。
- 对于二连通分量：从O(log n)轮、O(log⁵ n)全局容量降至O(log n)轮、标准容量。

**⚠️ 局限性**

局限性：
- 对于任意权重的确切MST，是否能在O(log n)轮与标准容量下实现仍是开放问题。
- 算法依赖随机化（MPX稀疏化与常数度树构造），目前无法得到确定性O(log n)轮的实现。
- 未给出总消息复杂度的上界，无法保证O(n polylog n)总消息。
- 对低度图的性能相对保守；在某些模型下仍需进一步改进。

---

## 320. ViewDreamer: A Robot World Model beyond Fixed Camera Setups

**arXiv ID:** 2611.26981 | [PDF](https://arxiv.org/pdf/2611.26981v1)

**作者:** Ye Mao `[一作]` (Meta AI), Wei Ye `[通讯]` (Meta AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种多视角机器人世界模型，能在任意数量摄像头下生成跨视角一致的未来视频并在动作感知模型与策略学习中取得显著提升。

**💡 创新点**

核心创新在于将预训练的单视角视频生成器与跨视角交互模块（CVI）分离，利用共享槽（slot）实现视角间信息交流，并通过槽一致性（CSA）和视角计数课程（VCC）训练保证跨视角一致性。

**🔧 技术方法**

技术包括：基于Diffusion Transformer的DiT视频生成器、跨视角交互模块（READ‑MIX‑WRITE）、槽一致性监督、视角计数课程、以及动作条件世界模型与世界‑动作模型（WAM）的集成。

**📊 数据集**

使用约300万条来自14种机器人实现的真实机器人视频（AgiBotWorld‑Beta、Galaxea、RoboCOIN、RoboMIND、RoVid‑X）以及RoboCOIN、RoboTwin、LIBERO、RoboDojo等测试集。

**📈 对比分析**

与Genie‑Envisioner、Cosmos3、Wan系列等基线进行对比，在RoboView‑Bench上实现三视角和四视角最高的SSIM、LPIPS、FID、FVD以及跨视角一致性（CVC）得分；在动作条件生成和WAM策略中，在干净与随机化测试中均优于FastWAM‑Joint，尤其在OOD场景下提升显著。

**⚠️ 局限性**

局限包括：对快速运动的腕摄像头生成一致性仍有挑战，实验仅覆盖最多四个摄像头且未验证更广泛的摄像头布置，且未测试更大模型或更复杂的多机器人场景。

---

## 321. Cannot Wait to Verify: Parallel Drafting to Eliminate Verifier Idle Time in Speculative Decoding

**arXiv ID:** 2611.27477 | [PDF](https://arxiv.org/pdf/2611.27477v1)

**作者:** Divya Jyoti Bajpai `[一作]` (IIT Bombay), Manjesh Kumar Hanawal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种自适应层并行草稿生成方法（ALD），在Speculative Decoding中加速草稿阶段并保持高验收率

**💡 创新点**

利用隐藏状态相似性动态控制多层并行执行，既能减少序列化深度又能自我校正，且不需要改造模型或再训练

**🔧 技术方法**

层并行草稿、相似性阈值控制、KV缓存校准、基于Cosine相似度的在线检查

**📊 数据集**

WMT、MMLU、MGSM、Math、IF‑Eval、SpecBench等多任务NLP基准

**📈 对比分析**

与标准SD、EasySpec、LayerSkip对比，ALD在保持与SD相近的验收率的同时，整体推理加速提升至2.5–3.8×，单独草稿时间提升1.5–1.7×

**⚠️ 局限性**

对相似性阈值和窗口大小的敏感性需要进一步研究，且在极度复杂或分布偏移输入上仍可能出现较高误差或接受率下降

---

## 322. Verifying and Strengthening Neural Tangent Kernel Theory

**arXiv ID:** 2611.27487 | [PDF](https://arxiv.org/pdf/2611.27487v1)

**作者:** Hao Zhu `[一作]` `[通讯]` (CSIRO Technology), Hao Zhu (CSIRO Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用 Lean 4 对两层神经网络的 NTK 迟训练收敛论证进行完全形式化，并证明梯度流存在、唯一并指数收敛

**💡 创新点**

显式化矩阵 Bernstein 变异代理并证明其可减小 36–121 倍宽度阈值；同时分析参数范数对线性化误差的影响

**🔧 技术方法**

形式化工具：Lean 4、Mathlib、Lieb-凸性定理、矩阵 Bernstein 论证、Picard–Lindelöf、Grönwall 定理、L^2 线性化估计

**📊 数据集**

合成数据：tanh 激活、均匀球面输入、Gaussian 权重、偏置为 0，使用 10 组随机输入做 Monte‑Carlo 估计

**📈 对比分析**

对比标准 R²N 代理与行协方差代理，数值实验表明在 tanh 条件下宽度阈值可缩小 36–121 倍（中值 45–103 倍）

**⚠️ 局限性**

结论是宽度需求极高（≈10¹⁵–10¹⁶），远超实用网络；仅验证了理论框架而非提供实用网络的改进

---

## 323. No Perfect Guardrail & No Perfect Guard: On the Decidability of Output Policies for Large Language Models

**arXiv ID:** 2611.27260 | [PDF](https://arxiv.org/pdf/2611.27260v1)

**作者:** Ahmed Amer `[一作]` (Santa Clara University), Darrell D. E. Long `[通讯]` (Santa Clara University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过形式化语言模型的“护栏”（guard）和监控（monitor）机制，证明在生成器能够产生任意有限字符串的情况下，若所需遵守的政策不可判定，则不存在完美的护栏，即无法同时完全阻止所有违规输出并允许所有合法输出；并进一步讨论这一不可实现性对AI安全伦理与治理的意义。

**💡 创新点**

将可判定性理论（如停机问题、Rice定理、哥德尔不完备性）与AI护栏概念相结合，首次给出“完美护栏”可否实现的必要与充分条件；同时阐述了不可判定政策导致的不可避免的误判与信息屏蔽风险，并提出政策制定与监督责任的伦理框架。

**🔧 技术方法**

可判定性/不可判定性分析、图灵机与编码理论、递归可枚举集、复杂性理论、模型论中的安全属性形式化。

**📊 数据集**

无数据集，本文为纯理论分析。

**📈 对比分析**

本文不做实验或性能对比，所给结果为数学证明，未给出具体实现或误差度量；因此无法评估实际性能。

**⚠️ 局限性**

局限性包括：假设生成器能输出所有字符串（full generator）；模型对实际系统的抽象过度简化；未给出平均误差下界；仅针对不可判定政策说明无完美护栏，无法针对可判定或受限情形给出通用方案；并未讨论实际部署中的概率分布与攻击者模型。

---

## 324. Distillation with Errors: Sparse Unbiased Backpropagation for Knowledge Distillation

**arXiv ID:** 2611.27131 | [PDF](https://arxiv.org/pdf/2611.27131v1)

**作者:** Yuyang Zhang `[一作]` (University of Science and Technology of China), Yuxin Li `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为Distillation with Errors (DWE) 的方法，用稀疏、无偏的配对采样来加速知识蒸馏中的输出头反向传播。

**💡 创新点**

创新点在于利用logit梯度的零和结构，将正负部分分别建模为两种采样分布，使用几何感知配对与分层采样实现无偏且方差更低的稀疏梯度估计。

**🔧 技术方法**

核心技术包括：零和配对采样、几何感知的输出行排序配对、分层采样、稀疏反向传播以及对估计无偏性的理论证明。

**📊 数据集**

实验使用Qwen3系列大模型（从4B、8B、32B教师蒸馏到0.6B、1.7B、4B学生），训练数据为FineWeb-Edu的80M文本。

**📈 对比分析**

与全词表蒸馏（Full）和重要性采样（IS）对比，DWE在相同采样预算下隐藏层梯度MSE下降约%（具体数值在论文中给出），输出头反向时间提升1.5-2.2倍，整体训练时间缩短约15-20%，在七项零样本基准上表现与Full相当。

**⚠️ 局限性**

局限性包括：需要完整的logit梯度（仅适用于softmax目标）、采样方差影响训练轨迹、收益主要来自输出头反向，不同模型或任务中整体加速比例可能受限。

---

## 325. Disentangling Verbalized Evaluation Awareness and Refusal through Neuron-Level Intervention

**arXiv ID:** 2611.27032 | [PDF](https://arxiv.org/pdf/2611.27032v1)

**作者:** Jingyuan Feng `[一作]` (University of Tokyo), Yusuke Iwasawa `[通讯]` (University of Tokyo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型推理模型中的可口化评估意识（VEA）神经元进行定位与干预，并研究其对拒绝行为的影响。

**💡 创新点**

首次证明VEA神经元同时携带评估意识与拒绝方向，揭示两者通过相同神经元实现耦合的机制性解释。

**🔧 技术方法**

采用激活差异定位、均值替换干预、拒绝方向投影保留、线性probe测量unVEA、激活驱动与RLVR训练等技术。

**📊 数据集**

使用QwQ-32B、AgentHarm、SorryBench、StrongREJECT、OLMo 3后训练检查点（SFT/DPO/RLVR）以及Situational Awareness Dataset (SAD)等数据集。

**📈 对比分析**

与随机神经元干预和激活驱动（steering）对比，VEA率降至≈0，拒绝率下降约30–40%，在不同训练阶段验证RLVR显著提升VEA与神经元选择性。

**⚠️ 局限性**

局限于单一模型家族；定位方法可能捕捉多义或低层特征神经元；未完全解析unVEA贡献；拒绝方向假设可能忽略其他相关方向。

---

## 326. Impact of Edge Software Evolution on Audio Tagging Models: An Empirical Study on NVIDIA Jetson Orin Nano

**arXiv ID:** 2611.26831 | [PDF](https://arxiv.org/pdf/2611.26831v1)

**作者:** Guillem Fustier-Castillo `[一作]` (Instituto Tecnologico de Informatica), Javier Naranjo-Alcazar `[通讯]` (Instituto Tecnologico de Informatica)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `b88c6eac-d57a-4623-a604-1f401f3eb268` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e`

**🎯 论文内容**

研究软件生态演进对边缘设备上音频标注模型部署性能的影响，基于NVIDIA Jetson Orin Nano在JetPack 6.2和7.2环境下评估PANNs、AST、PaSST等模型。

**💡 创新点**

首次系统化比较不同软件栈版本对同一模型在同一硬件上的推理延迟、能耗与资源利用差异，揭示软件更新对卷积与Transformer模型的性能影响不均衡。

**🔧 技术方法**

使用ONNX Runtime结合CPU、CUDA、TensorRT执行后端，测量FP32推理延迟、能耗、CPU/GPU利用率、内存占用与温度。

**📊 数据集**

使用AudioSet公开数据集（1000条10秒音频）进行推理评估。

**📈 对比分析**

对比不同模型、后端及JetPack版本，发现JetPack 7.2在TensorRT后端可将卷积模型延迟降至4-6倍、能耗降低约3倍，Transformer模型收益有限；在性能-准确率Pareto前沿中软件升级导致非线性结构调整。

**⚠️ 局限性**

局限在于仅评估FP32精度、单一硬件平台与三种后端，未探讨FP16/INT8量化或其他边缘平台的迁移性，也未深入解析软件层面优化机制。

---

## 327. The Bell Experiment, Monadically

**arXiv ID:** 2611.26777 | [PDF](https://arxiv.org/pdf/2611.26777v1)

**作者:** Chenchao Ding `[一作]`, Amr Sabry `[通讯]`

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

将 Bell 实验抽象为可执行的 Haskell 程序，利用不同的 monad（Reader、State、IO）和 membrane 结构，模拟经典、量子、PR 以及其他各种假设下的盒子行为，并通过程序生成实验记录、统计表格来检验 CHSH 值与无信号性。

**💡 创新点**

创新点在于：①把 Bell 试验拆成两层（实验层与盒子层），用 monad 明确展示经典假设（盲目性、局部性、因子化）以及它们的破坏方式；②通过程序化的采样器与调度器实现各种实验漏洞（定时器、超决定性等）；③用可复现的测试套件验证所有模型的结果，保证代码与论文声明完全一致。

**🔧 技术方法**

主要技术：Haskell 编程语言；Monad（Reader、State、IO）、Applicative、liftA2、membrane 设计；类型类实现盒子接口；线性代数（向量、矩阵、张量积）实现量子态与测量；随机数种子控制可复现性；测试框架。

**📊 数据集**

数据来源完全由程序生成：每个模型使用固定种子或全确定的随机数产生实验记录；没有外部数据集，所有表格和数值均由代码直接输出。

**📈 对比分析**

比较方法：对每个模型在相同次数（nTrials）实验中计算 CHSH 值和无信号性判定，并将结果以表格形式汇总。性能上未做显著的优化，重点在正确性和可复现性；测试套件通过所有断言表明实现与论文声明一致。

**⚠️ 局限性**

局限性：①代码并未通过类型强制保证无信号性，需手动验证；②仅覆盖两测量、两输入、两输出的 CHSH 场景，未扩展到多体或多测量情况；③未对 Tsirelson 上界给出证明；④量子模型仅限于两个量子比特，未涵盖更一般的量子实验；⑤对不同 Haskell 版本的依赖导致可复现性受限。

---

## 328. Tokenize Moments, Not Frames: Tokenizer-Aware Budget Allocation for Training-Free Video LLM Acceleration

**arXiv ID:** 2611.26933 | [PDF](https://arxiv.org/pdf/2611.26933v1)

**作者:** Zirui Zhao `[一作]` (Huazhong University of Science and Technology), Linghe Kong `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种训练无关、基于tokenizer的闭式规则方法 ESSENCE，能够在长视频中按固定 token 预算将视频划分为 moments（瞬时）并分配 token grid，随后在 LLM 后期释放对预测影响不大的视觉 token。

**💡 创新点**

创新点在于先识别视频瞬时并据此分配 token grid，避免 3D‑patch 双曝光导致的误码；同时通过在 LLM 里释放不再影响预测的视觉 token，实现在不训练的前提下显著提升长视频精度。

**🔧 技术方法**

主要技术包括帧相似性分析、闭式分配规则、token grid 编码、LLM 层级视觉 token 释放等。

**📊 数据集**

使用四个多选视频问答基准：TempCompass、MVBench、Video‑MME、LongVideoBench。

**📈 对比分析**

与五种训练无关加速器（FastVID、FlashVID、iLLaVA、EarlyTom、ShortV）以及下采样基准对比，ESSENCE 在 58/60 个匹配 TTFT 的对比中准确率更高，平均 TTFT 提升约 1.6–2.3 倍，且在多种模型（Qwen3‑VL、Qwen2.5‑VL、InternVL3.5）和 benchmark 上均表现优异。

**⚠️ 局限性**

局限在于对极长视频仍可能出现稀疏问题，对非相关瞬时场景的处理不足；实验仅覆盖公开模型和上述四个基准，未来需在更大规模或更复杂任务中进一步验证。

---

## 329. GRACE: Controlled Evidence Interventions for Diagnosing and Improving Knowledge Integration in GraphRAG

**arXiv ID:** 2611.27350 | [PDF](https://arxiv.org/pdf/2611.27350v1)

**作者:** Minh Pham `[一作]` (Posts and Telecommunications Institute of Technology), Cong Tran `[通讯]` (Posts and Telecommunications Institute of Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `79276348-11e0-48e3-84bc-7ec231d0171c` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了GRACE基准，通过在同一问题上执行五种受控图干预，构造匹配的知识图谱状态，并设计QCFSE与QCFSE-SA两种方法，用以诊断并提升GraphRAG在知识整合过程中的表现。

**💡 创新点**

创新点在于：①构建了可对比的五状态图干预框架，①揭示了“相关性≠充分性”的事实；②引入了显式的图充分性判别门并通过校准阈值实现主动否决。

**🔧 技术方法**

使用技术包括：图的类型化局部与全局注意力、基于问题的消息传递机制、冻结生成器（Qwen3.5-35B-A3B）与基于图充分性的门控决策，以及基于对比损失的训练与阈值校准。

**📊 数据集**

实验数据基于LC‑Quad 2.0的SPARQL查询与Wikidata，构成4,028道匹配问题（20,140问图对），以及Qwen3.5-35B-A3B作为通用生成器。

**📈 对比分析**

与四种检索框架（GraphRAG、LightRAG、RAPTOR、FAICO）及两种问题‑仅路由方法对比，QCFSE-SA在受控状态准确率上从57.17%提升至62.78%（+5.61个百分点），尤其在“缺失支持”状态上提升12.44个百分点。

**⚠️ 局限性**

局限性包括：对“未知实体”提示的潜在利用，过度否决导致在语义干扰图上性能下降，且实验仅覆盖LC‑Quad 2.0/Wikidata与单一生成器，缺乏对其他知识源和生成模型的泛化验证。

---

## 330. Distribution-Robust Halfspace Testing

**arXiv ID:** 2611.27062 | [PDF](https://arxiv.org/pdf/2611.27062v1)

**作者:** Xi Chen `[一作]` (Columbia University), Rocco A. Servedio `[通讯]` (Columbia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种新的半空间属性检验模型——分布鲁棒半空间检验器，能够在输入分布与预期一致时提供与传统分布特定检验器相同的效率，同时在任意分布下也能给出可信的答案。

**💡 创新点**

创新点在于引入“分布鲁棒”概念，并设计了能够在标准高斯分布下实现子线性查询复杂度（O(n^{2/3})）的检验算法；同时证明了在样本基础下该模型至少需要Ω(n^{2/3})个样本，从而揭示了分布鲁棒检验与传统分布特定/无分布检验之间的显著分离。

**🔧 技术方法**

核心技术包括：① 正态向量估计器和局部距离估计器，用于在不假设旋转不变性的情况下近似半空间的法向量和距离；② “距离不一致”度量和“裁剪分布”（whittling）技术，用以控制估计方差并筛除不符合预期分布的样本；③ 谱条件测试（Frobenius 范数和加权质心测试），用于保证裁剪后的分布具有良好的统计性质；④ 在证明下界时使用了“尖峰高斯”模型和对数似然方法以及高斯Poincaré不等式。

**📊 数据集**

该工作完全基于理论分析，不依赖任何具体数据集；主要使用高斯分布、尖峰高斯分布等理想化概率模型进行实验与证明。

**📈 对比分析**

与已知结果相比，在标准高斯分布下，传统无分布检验需要Ω(n)查询，而本文提出的分布鲁棒检验器仅需O(n^{2/3})查询；在样本基础检验中，传统分布特定检验可用Θ(√n)样本，而本文的下界证明表明至少需要Ω(n^{2/3})样本，显示分布鲁棒检验在效率上与无分布检验相比有显著提升。

**⚠️ 局限性**

限制主要体现在：① 目前仅给出了高斯分布下的上界和下界，未覆盖更广泛的分布族；② 上界的ε依赖相对较高（ε^{38}），实际实现可能受限；③ 对“分布鲁棒”模型的可扩展性和在真实数据上的鲁棒性尚未验证，未来工作需要进一步优化参数、降低复杂度并测试实用场景。

---

## 331. Testing for Coordination Failures in Cooperative Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.27368 | [PDF](https://arxiv.org/pdf/2611.27368v1)

**作者:** Dennis Gross `[一作]` (Simula Research Laboratory), Helge Spieker `[通讯]` (Simula Research Laboratory)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3855fcda-48ef-4070-a15e-803cd5c84d83` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于 Q‑value 差距的测试用例筛选方法 DisQ，旨在快速识别并检测协作多智能体强化学习（CMARL）中的误配协调失效。

**💡 创新点**

创新点在于：① 引入 Q‑value gap 指标作为可解释的误配信号；② 将该指标用于无执行的测试用例预筛选（DisQ），显著提升现有单智能体 SBST（随机、遗传算法）在 CMARL 环境中的失效检测效率。

**🔧 技术方法**

技术方法包括：深度 Q‑learning（DQN）训练的 CMARL 策略、计算最大 Q‑value 差距、基于经验分位数校准阈值、搜索式软件测试（SBST）与 DisQ 的结合。

**📊 数据集**

数据集为四个多智能体粒子环境：Simple Spread、Simple Speaker Listener、Coin Collector 以及自定义的 Knight Wizard Zombies，分别在集中式和分散式训练下评估。

**📈 对比分析**

与传统随机测试及遗传搜索相比，DisQ 选出的测试用例在多数配置下产生更低的平均回报并显著提升失效率（OR≥1.19，p<0.001），尤其在集中式共享奖励环境中，平均回报降低 8–36%，失效率提升 1.1–6.8 倍；但在初始状态对结果影响有限的环境（如 Knight Wizard Zombies）和训练不足的策略中，优势不明显或无显著提升。

**⚠️ 局限性**

局限性包括：① 仅适用于基于 Q‑value 的值函数学习（无适用于策略梯度或共用 critic 的架构）；② 需要白盒访问 Q‑value，黑盒场景不可用；③ 只检测初始状态的 Q‑value 差距，无法捕捉仅在后期出现或无差距的失效；④ 效果依赖于训练质量、奖励结构和协作程度；⑤ 目前仅考虑并行动作的 CMARL，未扩展到交替或在线监测场景。

---

## 332. Learning under Bounded Manipulation for Quadratic Incentive Design

**arXiv ID:** 2611.26736 | [PDF](https://arxiv.org/pdf/2611.26736v1)

**作者:** Arda Güçlü `[一作]` (University of Illinois Urbana-Champaign), Tamer Başar `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 48079 | [OpenAlex ID](https://openalex.org/A5019604570)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究在重复二次高斯Stackelberg博弈中，领导者在面对可操纵观测的追随者时，学习追随者私有类型并设计激励规则，求解最优学习与激励策略。

**💡 创新点**

提出了受限操纵预算下的极小极大后悔下界，构造了匹配该下界的学习策略，并证明在某些情况下追随者的操纵反而能降低领导者后悔。

**🔧 技术方法**

利用信息论（Fano不等式、van Trees不等式）、自适应控制与随机逼近等技术进行误差分析与下界证明。

**📊 数据集**

无实验数据，全部为理论分析与仿真验证。

**📈 对比分析**

通过理论证明显示领导者后悔量为O(max{log T, min{C_T,T}})，并用数值示例说明操纵预算非单调影响后悔，性能优于无操纵情况。

**⚠️ 局限性**

主要局限在于仅考虑二次成本形式、单一私有参数、线性激励规则，且未对随机策略或多样化信息结构进行推广。

---

## 333. Looking Right Is Not Being Right: Evaluating the Geometry of Agent-Built 3D Scenes

**arXiv ID:** 2611.26937 | [PDF](https://arxiv.org/pdf/2611.26937v1)

**作者:** Ziyang Long `[一作]` (University of California, Los Angeles), Hsin-Jung Yang `[通讯]` (Cedars-Sinai Medical Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出基于激光扫描的几何评估方法，研究视觉‑语言代理在从视频重建室内场景时的几何精度与验证能力，并提出几何误差报告以提升重建质量。

**💡 创新点**

创新点在于将可见几何、深度一致性和家具布局统一到一个 0–100 的几何得分中，并通过可控错误实验验证评估敏感度，同时引入基于点云对比的误差报告，使代理能显式检查并纠正自身场景。

**🔧 技术方法**

技术上使用 Blender 程序生成场景、ARKit 估计的相机位姿与点云、刚性配准（Chamfer 距离最小化）、F‑score 计算以及日志化工具调用以分析重建过程。

**📊 数据集**

实验数据来源于 ARKitScenes 数据集（RGB 漫游视频、激光深度扫描、相机轨迹及 3D 框架标注），并对比公开的 BVB 重建结果进行渲染相似度评估。

**📈 对比分析**

通过将 G 与渲染相似度 LS 进行 Spearman 相关性检验，发现整体相关性仅约 0.37，单个配置内部相关性更低（≈0.08）。最佳视频‑仅代理（Terra/Sol/Astra）在 G 上达到 70–80 分，密集 Poisson 参考可达 90 分，提供了 oracle 输入时 G 可接近 90 分的上限。

**⚠️ 局限性**

局限性包括仅覆盖单一室内数据集、仅评估可见几何（不含语义或可编辑性）、误差报告仅在实验室已见房间上调校、缺乏对其他开发者代理的泛化验证以及代理继承的尺度误差难以独立纠正。

---

## 334. Dual-Agent DRL based Cooperative Beamforming and Trajectory Optimization for UAV-ISAC Networks

**arXiv ID:** 2611.27346 | [PDF](https://arxiv.org/pdf/2611.27346v1)

**作者:** Kaijian Yang `[一作]` (Guangdong University of Technology), Vincent K. N. Lau `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对低空多基站多无人机 ISAC 网络，提出了双代理深度强化学习框架（DeepDA-OPT），联合优化基站发射波束与无人机轨迹，以最大化通信无人机的平均总速率，并满足功率、感知 SINR、任务完成及碰撞规避等约束。

**💡 创新点**

创新点包括：① 利用广义 Rayleigh 商定理得到接收波束的闭式最优解，剔除接收波束决策；② 采用双代理（Agent‑B 专注波束、Agent‑T 专注轨迹）分解高维异构决策，提升学习效率与稳定性；③ 设计约束感知的噪声探索策略与奖励函数，避免直接加入约束导致学习困难；④ 在 TD3 基础上加入 RNN 捕获时序相关性，进一步提升性能。

**🔧 技术方法**

主要技术：多智能体深度强化学习（MADRL），Twin‑Delayed Deep Deterministic Policy Gradient（TD3），递归神经网络（GRU），闭式波束求解（Rayleigh 商），约束感知噪声探索，奖励设计。

**📊 数据集**

使用仿真数据集：4 个基站（S=6 天线），2 个通信无人机 + 2 个感知无人机，N=20 时隙，Vmax=20 m/s，仿真场景随机化轨迹与信道，满足 LoS 路径损耗模型、噪声、残留干扰等参数。

**📈 对比分析**

与基准方案（AO、SA‑TD3、SA‑DDPG、DeepDA‑MRC、SA‑TD3‑MRC）进行比较。DeepDA-OPT 在平均总速率上高 0.6–1.0 bps/Hz，感知 SINR 超过阈值 4–5 dB，碰撞率 0%，在线推理时延仅 1–2 ms，训练收敛稳定，跨种子方差显著降低。

**⚠️ 局限性**

局限性：假设理想 LoS 信道，忽略 Doppler、NLoS 阻塞、实时信道估计误差；未考虑无人机推进能量约束；模型仅在仿真中验证，缺乏真实场景测试。

---

## 335. NavTTT: Adaptive Parametric Memory via Hierarchical Test-Time Training for Vision-and-Language Navigation

**arXiv ID:** 2611.27219 | [PDF](https://arxiv.org/pdf/2611.27219v1)

**作者:** Xiaoyu Luo `[一作]` (University of Macau), Hui Kong `[通讯]` (University of Macau)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出NavTTT框架，利用快权重记忆在视觉语言导航中自适应存储与检索关键观测，提升长期决策性能。

**💡 创新点**

创新点在于基于全局导航上下文的任务感知token选择、跨层独立快权重存储以及层级检索融合，既过滤冗余信息又保持历史记忆。

**🔧 技术方法**

采用快权重测试时训练（TTT）与LaCT非线性快权重机制，结合Transformer解码器多层自注意力与记忆检索。

**📊 数据集**

使用R2R‑CE与RxR‑CE连续导航基准以及在真实Humanoid机器人上收集的RGB轨迹数据进行评估。

**📈 对比分析**

相较于JanusVLN、StreamVLN等基线，NavTTT在R2R‑CE上实现66.7% SR、59.4% SPL，RxR‑CE上实现63.6% SR、52.7% SPL，明显优于同类方法；在真实机器人上成功率为27.5%。

**⚠️ 局限性**

局限性包括真实环境中的可靠性仍低、对长轨迹和重复地标的记忆保持不足，以及对快权重更新速度与存储容量的依赖。

---

## 336. SecretAgent: A Multi-Agent Framework for Privacy Protection against Vision Language Models

**arXiv ID:** 2611.26936 | [PDF](https://arxiv.org/pdf/2611.26936v1)

**作者:** Minda Zhao `[一作]` (Harvard University), Mengyu Wang `[通讯]` (Harvard University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为 SecretAgent 的多代理框架，利用 LLM 识别图像中泄露私密信息的视觉证据，并在限定扰动预算内生成不可见扰动，既能有效防止 VLM 的隐私泄露，又能保持图像视觉质量。

**💡 创新点**

创新点包括：①闭环多代理自适应流程，先识别 VLM 关注的区域，再通过规划和参数化产生局部隐蔽扰动；②提供两种实现方案——梯度驱动的 SecretAgent-Gradient 与无梯度噪声库驱动的 SecretAgent-Tool；③在保持高保真度的同时，跨 VLM 的保护迁移效果显著。

**🔧 技术方法**

主要技术：LLM（GPT‑6 Luna、Claude Haiku 等）负责规划与参数设定；CLIP、InternVL3.5、Gemma‑4‑31B、DINOv3 等多种编码器用于估计扰动方向；梯度签名步进或噪声库采样产生扰动；视觉审查 Agent 检测可见伪影；结合多轮闭环反馈优化。

**📊 数据集**

使用了五个公开数据集，涵盖四类隐私场景：Pri‑Street‑View（地理定位）、MultiTrust P5（PII）、FRLL（面部属性）和 RHPE、PAPILA（医学图像）。

**📈 对比分析**

与 AnyAttack、M‑Attack、FOA‑Attack、MPC‑Attack 等黑盒 VLM 防护基线进行对比，评估指标包括私密信息准确率、PSNR、MS‑SSIM、LPIPS、DISTS 及 LLM judge。SecretAgent 在所有场景下实现了最高的隐私抑制且视觉质量最低，跨 VLM 的迁移性能也优于所有基线。

**⚠️ 局限性**

局限性：搜索轮数固定，导致不同图像耗时不均；仅对单张图像进行保护，未考虑跨图像或视频的聚合攻击；对复杂多源信息的聚合仍需进一步验证。

---

## 337. Accelerating Muon with Spectral Residual Correction

**arXiv ID:** 2611.27077 | [PDF](https://arxiv.org/pdf/2611.27077v1)

**作者:** Zhixin Ren `[一作]` (Tsinghua University), Liping Zhang `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AIMuon 优化器，利用 AIM 框架对参数子问题进行局部线性化，生成梯度-动量残差的独立矩阵符号变换与核范数加权，结合 Muon's 谱方向得到新的矩阵更新，从而提升大模型矩阵参数的优化效率。

**💡 创新点**

创新点在于：① 在谱几何下通过参数子问题近似得到残差的独立矩阵符号变换；② 引入核范数权重动态调整残差与动量方向的组合；③ 对残差进行独立正交化而非直接混合，从而获得更优的更新方向；④ 在残差方向上实现 RMS 匹配，保持与 Muon 同等更新幅度。

**🔧 技术方法**

技术手段包括：AIM 框架、Newton–Schulz 迭代近似矩阵符号变换、核范数权重与自适应核范数加权、RMS 匹配、与 Muon、Muon‑Nesterov、MONA 的对照实验。

**📊 数据集**

数据集：FineWeb‑Edu（≈2.48B tokens）用于 GPT‑2‑small；FineWeb（≈6.9B tokens）用于 GPT‑2‑medium；FineWeb（≈12B tokens）用于 Qwen3‑0.6B。

**📈 对比分析**

与 Muon、Muon‑Nesterov、MONA 在相同 GPU、相同随机种子、相同 token 序列下对比；AIMuon 在所有三种模型上均取得最低的最终验证损失和困惑度，并在达到基准损失时比 Muon 低约 8.98%–10.05% 的优化步数，性能提升显著。

**⚠️ 局限性**

局限性：每一步计算量略增（约 4%–5% 训练时间提升）；残差梯度-动量残差相互作用机制尚不充分理解；实验仅限于大语言模型预训练，需在更广泛任务和模型规模上进一步验证。

---

## 338. Simulation-Based Testing of RCM-Constrained Robotic Software via Task Manifolds

**arXiv ID:** 2611.27144 | [PDF](https://arxiv.org/pdf/2611.27144v1)

**作者:** Zhenyu Jiang `[一作]` (University of Chinese Academy of Sciences), Kai Huang `[通讯]` (Sun Yat-sen University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种以任务流形为核心的系统级仿真测试框架，先构造符合 RCM 约束的可行测试域，然后在此域上生成可执行的手术机器人测试案例。

**💡 创新点**

创新点在于将可行的任务空间直接映射为流形并预先过滤不可行输入，完全解耦测试域构造与搜索策略，实现 100% 可执行候选输入，并通过随机、遍历、遗传等多种搜索策略发现更丰富的失败行为。

**🔧 技术方法**

采用逆运动学解析/数值求解、空间参考网格覆盖、离线轨迹安全预判器、随机/遍历/遗传算法搜索、PCA 低维学习以及 MoveIt2 运动规划等技术。

**📊 数据集**

实验使用仿真生成的五轴手术机器人和 Kinova Gen3 7 轴机械臂的任务参数与工作空间约束，未使用公开数据集，仅利用实验生成的可行点和轨迹数据。

**📈 对比分析**

在相同测试预算下，Manifold 与 Cartesian、PCA 进行随机、遍历、GA 三种策略比较；Manifold 在候选有效率（100%）和失败模式多样性方面优于其他两种，GA 在 Manifold 上实现最高失败率并捕获 7 种失败模式；遍历实现 100% 参考格单元覆盖，但失败率低。

**⚠️ 局限性**

需要先构造任务流形和可行域，工作量大且依赖精确运动学和几何约束；离线预处理（如 PCA）开销高；实验仅在仿真环境，未知是否同样适用于真实机器人或其他手术任务；未考虑视觉/语言等多模态因素。

---

## 339. EDAGoat: An EDA-Aware Agent for Project-Level GPU Optimization

**arXiv ID:** 2611.27196 | [PDF](https://arxiv.org/pdf/2611.27196v1)

**作者:** Chengxi Li `[一作]` (Chinese University of Hong Kong), Evangeline Young `[通讯]` (Chinese University of Hong Kong)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了EDAGoat，一个面向GPU加速EDA工具的代理，利用交叉项目的GPU优化模式和强制验证来改进代码、提升性能并保证设计正确性。

**💡 创新点**

创新点在于将EDA领域的知识与GPU优化模式相结合，构建跨项目的优化模式库和验证模板，并通过规则控制器实现强制验证，从而在保持设计质量的前提下获得更高的速度提升。

**🔧 技术方法**

采用大语言模型（GPT-6 Astra 与 DeepSeek）驱动的自动化编辑与调优，配合交叉项目GPU优化模式、验证模板、规则控制器，以及完整求解器级别的性能评估与 CUDA 12.6 运行环境。

**📊 数据集**

使用了五个GPU EDA工具（CULS、InstantGR、DREAMPlace、Xplace、GCS-Timer）以及55个输入数据集（包含4个公开、51个隐藏），覆盖逻辑综合、布线、放置和时序分析等任务。

**📈 对比分析**

通过与通用提示、优化增强（OE）以及OE+强制验证（MV）三种配置对比，使用完整求解器时间比（speedup）衡量性能。结果显示：Astra在所有15次搜索中均合格，平均和最佳加速分别提升至3.906×；DeepSeek MV模式合格率从7/15提升至13/15，平均加速提升至1.53×，优于OE和通用基线。

**⚠️ 局限性**

局限性包括：每个配置仅进行3次搜索、仅在RTX 3090上测试、评估耗时较长、优化模式库仅覆盖5个项目、验证测试的覆盖度依赖人工设计、模型的可靠性不一致、未能自动化模式提炼与零样本迁移等。

---

## 340. Beyond Final Answers: Interpretable Tumor Immune Microenvironment Reasoning from H&E and mIF

**arXiv ID:** 2611.26794 | [PDF](https://arxiv.org/pdf/2611.26794v1)

**作者:** Yixuan Duan `[一作]` (Rice University), Wei Qiu `[通讯]` (Rice University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研发了一种能够在同一切片的H&E与mIF图像上进行分阶段、可验证的多模态大语言模型，用于读取肿瘤免疫微环境指标。

**💡 创新点**

创新点在于将临床分级规则嵌入结构化推理模板，并利用同切片H&E+mIF的细胞级真值监督每一步推理，配合过程奖励提升推理可信度。

**🔧 技术方法**

使用4B参数的Qwen3.5-4B模型，结合LoRA微调、GRPO强化学习、过程奖励（r_proc）以及tolerance curriculum等技术进行训练。

**📊 数据集**

采用了Orion结肠癌同切片H&E+mIF数据集，并在外部CODEX数据集（Schürch CRC）以及头颈鳞状细胞癌CODEX组上进行验证。

**📈 对比分析**

通过与12款通用、医疗及病理MLLM以及基线模型比较，使用任务准确率、宏F1和κ等指标，在Orion上平均准确率达84.5%（相较基线74.7%），在CODEX和头颈癌上保持最高准确率，推理正确率提升至约66%。

**⚠️ 局限性**

局限性包括依赖同切片H&E+mIF的细胞级真值，无法直接迁移至无细胞级标注或不同标记方案的图像；仅在四个特定指标上验证，扩展至其他读数仍需更多训练与验证。

---

## 341. Leader Finding: A New Variation of Expert Finding in Community Question Answering

**arXiv ID:** 2611.26921 | [PDF](https://arxiv.org/pdf/2611.26921v1)

**作者:** Sepideh Entezari Maleki `[一作]` (Shahid Beheshti University), Mahmood Neshati `[通讯]` (Shahid Beheshti University)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种概率模型，旨在为给定的项目和团队对候选领导者进行排序，模型同时考虑技术适配、管理知识和与团队的沟通三大维度，并通过负知识惩罚和基于中心性测度的沟通模型实现对候选人的全面评估。

**💡 创新点**

创新点包括：①将领导者寻找问题重新定义为多维度的专家匹配任务；②设计负知识模型（Negative Knowledge Model）来惩罚具备项目外技术知识的候选人；③结合度中心性、接近中心性和介数中心性的沟通模型；④提出针对技术适配、管理知识和沟通的三个评估指标，并用调和平均值综合评估。

**🔧 技术方法**

技术方法主要包括：概率框架建模、Kernel‑based 中间知识检测、基于学习排序（LTR）的负知识惩罚（SVM‑Linear、LR、K‑NN、SVM‑Rank、RankNet）、基于图的中心性测度（度、接近、介数）以及坐标上升（Coordinate Ascent）学习权重。

**📊 数据集**

使用的数据集来自 Stack Overflow，包含：Java 技术集合（810k 问题/1.5M 答案/83k 用户）、C# 技术集合（764k 问题/1.45M 答案/84k 用户）以及项目管理集合（1.7k 问题/7k 答案/3.6k 用户）。

**📈 对比分析**

通过与三种基线（仅技术、仅管理、技术+管理）以及多种 NKM 与 LCM 变体进行对比，评估指标为 NDCG@10/20 与 GAP@10/20，基准 F‑measure 作为评分。实验结果显示，所提模型在 Java 与 C# 集合中 NDCG@10 从 0.236/0.048（TMKM）提升至 0.621/0.048（point‑wise 配置），整体性能提升超过 200%，且在所有指标上显著优于基线。

**⚠️ 局限性**

局限性包括：①专家级别估计基于接受答案，可能与真实专业水平不完全一致；②沟通度量仅基于技术知识重叠，未捕捉真实交流质量；③使用的 Stack Overflow 数据截至 2015 年，可能与当前开发环境差异；④管理集合规模远小于技术集合；⑤模型未考虑团队、组织、时间、预算等其它影响领导效果的因素。

---

## 342. 3D-NAND-CIM: A Hybrid-Bonded 3D-NAND-Flash-based Analog In-Memory-Compute macro offering 4.8M weights/mm$^2$

**arXiv ID:** 2611.27479 | [PDF](https://arxiv.org/pdf/2611.27479v1)

**作者:** Pritish Narayanan `[一作]` (IBM San Jose), Abu Sebastian `[通讯]` (IBM Zurich)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在14nm逻辑晶片与3D NAND闪存晶圆的混合键合下，构建了能容纳4.8M权重/平方毫米的模拟内存计算宏

**💡 创新点**

突破了NVM-AIMC的权重容量极限，首次实现了数百万级权重在单层芯片上的可编程存储，并通过电荷保留电路将误差降至与3-4位数值精度相当

**🔧 技术方法**

采用混合键合（Wafer‑on‑Wafer/Chip‑on‑Wafer）技术、4‑bit 1‑of‑16导通度编程、基于电流控制振荡器的ADC、局部控制器和电荷保留/舍入策略

**📊 数据集**

使用WideResNet‑16‑8 CNN 对CIFAR‑10/100 数据集进行推理测试

**📈 对比分析**

与现有NVM‑AIMC/ SRAM‑IMC 的能效与面积效率相当（4.2 TOPS/W、0.58 TOPS/mm²），但权重密度提升了数十倍，推理准确率仅下降0.4%（CIFAR‑10）/3.23%（CIFAR‑100）

**⚠️ 局限性**

目前仍为原型阶段，受制于14nm逻辑与NAND匹配、编程误差、精度受限于3‑4位、需要进一步降低能耗、面积并提升可重复性

---

## 343. Testable Testing: Distribution-Robust Property Testing

**arXiv ID:** 2611.27033 | [PDF](https://arxiv.org/pdf/2611.27033v1)

**作者:** Xi Chen `[一作]` (Columbia University), Rocco A. Servedio `[通讯]` (Columbia University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出并研究“分布鲁棒测试”模型，设计了一系列通用算法技术，实现了若干经典布尔函数属性在分布鲁棒框架下的高效测试；

**💡 创新点**

创新点在于引入分布鲁棒模型，证明在满足分布假设时可保真输出、在假设失效时可安全拒绝，并通过“分布挑选测试”“函数重构”“junta逼近”“分布测试”等技术将已有的学习和测试方法迁移至该模型；

**🔧 技术方法**

核心技术包括：1）分布挑选测试的定义与转换；2）基于学习的函数重构与本地纠错器；3）使用junta近似与自适应查询；4）利用分布识别与欺骗距离构造分布鲁棒检验；5）针对特定函数类（单调性、合取、决策列表、k区间）给出非自适应、适应性及样本基础的上界和下界；

**📊 数据集**

本文为理论工作，主要使用合成布尔域（{0,1}ⁿ 或 [0,1]）上的均匀、产品或固定分布作为测试场景；

**📈 对比分析**

与传统分布特定或无分布假设的测试相比，分布鲁棒测试在查询/样本复杂度上大幅降低（如合取、决策列表可常数查询；单调性在分布鲁棒下仍需指数级查询）；同时提供了上界与下界的定量对比；

**⚠️ 局限性**

局限性包括：仍缺乏对适应性分布鲁棒测试的完全解；某些属性（如半空间、k-区间）仅给出非自适应或样本基础的上界；部分下界仅适用于非自适应情况；未能统一处理所有分布族；

---

## 344. Path-Aware Fusion of Infrastructure RF Sensing and Robot Perception in a Live Robotic Digital Twin

**arXiv ID:** 2611.27497 | [PDF](https://arxiv.org/pdf/2611.27497v1)

**作者:** Alberto Martínez Méndez `[一作]` (Universidad Carlos III de Madrid), Carlos J. Bernardos `[通讯]` (Universidad Carlos III de Madrid)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在实验室中搭建了一个实时的机器人数字孪生平台，集成了四种感知分支（60 GHz链路、Wi‑Fi/5G 联合感知、固定摄像头、移动LiDAR），并通过路径感知门控（path‑aware gating）在机器人即将扫过的区域内判断是否应停止或等待，提升机器人对人类的安全响应。

**💡 创新点**

创新点：
- 通过数字孪生将异构传感信息与机器人的实时姿态关联，实现在机器人行进路径上做空间约束的决策；
- 将通信链路（60 GHz）作为高频率“触发器”，利用阵列相位一致性检测人类存在，即使链路承载近Gbit 传输；
- 引入失效分支的安全/宽容模式，可在感知分支失效时配置机器人采取安全停止或继续执行；
- 通过“1‑s 投票”对CSI检测进行时序平滑，兼顾低延迟和低误报。

**🔧 技术方法**

使用技术：
- 60 GHz CSI 采集（使用 MikroTik wAP 60G 设备，提取 32 个幅度+相位特征）；
- RBF‑SVM 与 1‑s 投票做人类检测；
- FAWN（Wi‑Fi/5G 结合的 Transformer‑编码器）做六格点位定位；
- YOLOX‑S 进行摄像头人类检测；
- ROS 2 与 Zenoh 做机器人状态与感知分支的数据交互；
- NVIDIA Isaac Sim 用于数字孪生仿真与可视化；
- REST API 与 sensor registry 管理感知分支。

**📊 数据集**

数据集：
- 自采集的 60 GHz CSI 数据：两个实验场（site A 与 site B），每个场 10–18 次会话，每次 120 s，包含空房、站立人、两人分布以及金属板干扰；
- 现场实验数据：Unitree Go2 机器人在 site B 的巡航任务，记录感知分支状态、DT 决策、机器人速度与转向、摄像头视频帧；
- 公开数据未使用，所有数据均为本文自建。

**📈 对比分析**

比较方法与性能：
- 60 GHz 链路检测：在 session‑disjoint 评估中 RBF‑SVM 获得 0.98 的平衡准确率；跨天评估 0.999（TPR 100%，FPR 0.2%）；在 site B 需要重新训练，准确率 0.95；
- 延迟测量：摄像头从人首次出现到机器人停下平均 1.83 s（主要由 0.5 s 确认窗和制动时间决定）；60 GHz 链路平均 1.57 s；
- 路径感知门控对比：在 Always‑Stop 策略下 5 次停下，Path‑Aware 策略下 0 次停下（对离路径的检测无误）；
- 失效分支：Fail‑Safe 模式下所有失效分支导致机器人停下（6/6），Tolerant 模式下 0/6；在 Tolerant 模式下出现 2 次 60 GHz 假正导致停机。
- 无误停机实验：5 圈无人时 0/5 停止。

**⚠️ 局限性**

局限性：
- 不是 ISO 3691‑4 认证的安全功能，决策延迟受 ROS 2/REST 时延影响；
- 60 GHz 链路检测高度依赖链路几何与空房参考，跨站点迁移需要重新采集与训练；
- 支持区域 Ω_s 与代表点 p_s 的配置误差会导致误判或漏检；
- 失效分支的宽容模式在某些情况下可能导致误停机；
- 现场实验规模有限（单个 site、单个巡航任务、10‑20 次重复），未涵盖复杂多机器人或大规模工厂环境；
- 未实现对 RF 证据的定位，仅使用预设支持，限制了对非线性路径或动态障碍的精细控制。

---

## 345. Enhancing Metaverse Security using Explainable AI-based Intrusion Detection Framework

**arXiv ID:** 2611.27199 | [PDF](https://arxiv.org/pdf/2611.27199v1)

**作者:** Esha Kuppanda `[一作]` (BITS Pilani), Raja Muthalagu `[通讯]` (BITS Pilani)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究提出了一种基于可解释人工智能的入侵检测框架，用于保护 Metaverse of Things（MoT）环境。

**💡 创新点**

创新点在于将 SHAP 与 LIME 等 XAI 方法与多种传统机器学习模型结合，以实现高精度且可解释的威胁检测。

**🔧 技术方法**

使用的技术包括随机森林（RF）、决策树（DT）、极端梯度提升（XGBoost）、极限树（ET）等传统 ML 模型，以及 SHAP 和 LIME 进行特征重要性解释。

**📊 数据集**

实验采用了三大基准 IoT 安全数据集：ToN‑IoT、CIC‑IoT‑2023 和 BoT‑IoT，以评估模型在不同攻击场景下的表现。

**📈 对比分析**

通过四折交叉验证和多指标（准确率、精确率、召回率、F1）比较，XGBoost 在 ToN‑IoT 与 CIC‑IoT‑2023 上分别取得 99.83% 与 96.45% 的最高准确率，RF 与 ET 在 BoT‑IoT 上达 99.99% 的极致性能。

**⚠️ 局限性**

局限性包括：仍采用传统 ML，缺乏深度学习提升；模型训练与推理耗时相对较高；仅在公开数据集上验证，缺少实际 MoT 环境的实时部署与评估。

---

## 346. Felid: A Flexible and Efficient Design for Transformer Fine-Tuning over Encrypted Data

**arXiv ID:** 2611.27175 | [PDF](https://arxiv.org/pdf/2611.27175v1)

**作者:** Linru Zhang `[一作]` (Nanyang Technological University), Kwok-Yan Lam `[通讯]` (Nanyang Technological University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在非交互式全同态加密（FHE）框架下，提出了一种可灵活批量打包、针对核型注意力优化的Transformer微调方法。

**💡 创新点**

创新点包括：① 灵活列式批量打包策略，可支持不同批量大小；② 针对矩阵运算的专门算法，显著减少密文旋转、重线化和乘法深度；③ 采用功能式重标定将激活函数与重标定合并，进一步降低开销。

**🔧 技术方法**

使用技术主要包括CKKS FHE、矩阵运算优化（CPMM、CCMM-LoRA、CCMM-kernel、CCMM-V）、功能式重标定、LoRA微调以及改进的矩阵转置与对角打包。

**📊 数据集**

在 GLUE 任务（MRPC、RTE、STS‑B）以及对应的 BERT 2‑层模型上进行实验，验证模型准确率与 plaintext 结果相近。

**📈 对比分析**

与最新的 Rho 等工作对比，2‑层 BERT 微调在批量 16 时完成时间从 5238 秒降至 650 秒，速度提升 8×；矩阵运算中的旋转/重线化减少 30×，bootstrapping 调用减少 2.5×。

**⚠️ 局限性**

仍然受限于 FHE 的计算密集度，实验仅在小型模型和有限任务上验证，缺乏大规模模型、跨任务及多语言的进一步评估。

---

## 347. Budget-Aware Predict-and-Verify Theorem Proving with Heterogeneous LLM Solvers

**arXiv ID:** 2611.26906 | [PDF](https://arxiv.org/pdf/2611.26906v1)

**作者:** Wei Jiang `[一作]` (Futurewei Technologies), Wei Wang `[通讯]` (Futurewei Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出预算感知的预测-验证系统 BudgetPV，用于在不同成本和能力的 LLM 解算器之间动态分配证明计算。

**💡 创新点**

结合跨解算器成功预测器和基于生成成本的评分机制，在不同证明阶段使用不同训练目标（类加权交叉熵与普通交叉熵）实现子目标与解算器的自适应分配。

**🔧 技术方法**

使用跨解算器预测器、类加权/普通交叉熵训练、预算感知评分与反馈更新等技术；在推理时基于预测成功概率与解算器成本进行决策。

**📊 数据集**

在 miniF2F、ProofNet、PutnamBench、Lean Workbook 四个正式推理数据集上进行实验。

**📈 对比分析**

与均匀分配和统计先验（oracle）基准对比，BudgetPV 在三种解算器设置下均显著减少 GPU 生成成本（约 20%–70%），并在相同生成预算下完成更多定理。

**⚠️ 局限性**

依赖成功预测准确性，预测误差导致与 oracle 的性能差距；系统仅在给定证明草图上工作，未考虑草图修订和更广泛的解算器族。

---

## 348. Windchime: an Audiovisual Installation with Audio-Language Model implementations

**arXiv ID:** 2611.26759 | [PDF](https://arxiv.org/pdf/2611.26759v1)

**作者:** Trent Eriksen `[一作]` (Independent Researcher), Celeste Betancur Gutierrez `[通讯]`

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

Windchime 是一套交互式视听装置，利用本地 ASR 与音频‑语言模型（ALM）对访客口述进行检索，将检索到的 406 条音频 stem 与 135 条动画场景实时组合成多模态音景与可控动画，访客可通过语音、触摸与视觉界面进行体验。

**💡 创新点**

创新点在于：① 将 ALM 的可插拔性与声音检索模式、stem 角色文件夹三大维度构成的代理分配机制，重新定义艺术家、模型与访客之间的交互代理；② 通过 live‑coding 代码实时混音与动画控制，实现语音指令与触摸输入的分离；③ 在同一装置中整合多种前沿 MIR、NLP 与实时渲染技术，提供可解释的潜在空间可视化。

**🔧 技术方法**

使用技术包括：Faster‑Whisper（本地 ASR）、CLAP/多语言 CLAMP3/MS‑CLAP 等 ALM、FAISS 索引、Librosa 音频特征、Strudel Planner（LLM 或规则模板生成），Monome Grid/Arc 触摸硬件，Live‑coding 环境，UMAP 可视化，HTML/CSS/JavaScript 前端与三维嵌入展示。

**📊 数据集**

数据集：Trent Eriksen 作品的 406 条音频 stem（包括乐器分轨与现场录音）以及 135 条抽象风格动画场景；另外使用公开 CLAP、CLaMP 预训练模型的 300M+ 语音‑文本/音乐‑文本对数据。

**📈 对比分析**

在缺乏统一的客观性能评估标准下，主要通过不同 ALM、声音模式组合的主观评测与访客反馈进行比较，发现 ALM 选择与声音模式对生成音景的音乐性与连贯性具有显著影响，但未提供定量指标。

**⚠️ 局限性**

局限性包括：① 依赖 ALM 的检索效果，模型差异导致结果不一致；② 核心检索基准（stem）数量有限，扩展性受限；③ 交互体验的可重复性受访客语言多样性与硬件条件影响；④ 缺乏系统化的可解释性与性能评测方法。

---

## 349. Neurodegenerative Differential Diagnosis with Knowledge-enhanced Structural Reasoning

**arXiv ID:** 2611.27427 | [PDF](https://arxiv.org/pdf/2611.27427v1)

**作者:** Mingxuan Wang `[一作]` (Shanghai Jiao Tong University), Jiangchao Yao `[通讯]` (Cooperative Medianet Innovation Center)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出一种基于知识增强的结构化推理模型KnowSR，用来对神经退行性疾病进行差异诊断，整合多模态（rs‑fMRI与临床指标）信息并与文献构建的疾病原型、方向性证据及相互关系相结合；

**💡 创新点**

创新点在于将结构化医学知识（疾病原型Card、方向性Card和相互关系Card）与患者具体证据对齐，并通过锚定关系推理在稀疏差异图上实现病症间的相对证据平衡，从而提升诊断解释性与准确性；

**🔧 技术方法**

采用共享时序卷积编码器对ROI时间序列进行特征提取，跨模态交叉注意力融合临床量表，利用多头注意力对知识Card进行对齐，构建锚定关系推理模块，并用多任务损失进行联合训练；

**📊 数据集**

使用四个研究数据库（ADNI、PPMI、NIFD、4RTNI）共331名受试者、763张rs‑fMRI扫描，覆盖NC、AD、PD、FTD、PSP、CBS六类；

**📈 对比分析**

与19个基线（传统机器学习、通用深度学习、脑网络模型、医学多模态模型）在患者分离五折评估下比较，KnowSR在六类诊断上达91.94%±4.33%的平衡准确率，显著优于最强基线（85.51%±2.96%），并在单一数据库内差异诊断中亦保持领先；

**⚠️ 局限性**

局限包括：知识库固定且离线构建，未在实时检索或动态更新；数据来源仅限四个研究数据库，缺乏真实临床多中心验证；目前仅覆盖六种神经退行性疾病，需扩展以适用于更广泛的临床谱系。

---

## 350. Show Me the Answer: Visual Grounding on Dashboards

**arXiv ID:** 2611.27355 | [PDF](https://arxiv.org/pdf/2611.27355v1)

**作者:** Yuefei Zhang `[一作]` (South China University of Technology), Yi Cai `[通讯]` (South China University of Technology)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在多图表仪表板中根据自然语言查询定位视觉元素的任务，并提出了两阶段的视觉仪表板定位框架 VGD-Net。

**💡 创新点**

创新点在于将视觉定位拆解为查询感知的图表检索与细粒度图表内定位两步，并通过多层特征聚合（MLAD）和检测+LLM 验证机制提升可靠性。

**🔧 技术方法**

主要技术包括 RT‑DETR 图表检测、OCR 文字匹配、SimVG 视觉语言模型+MLAD 多层融合、DeepSeek 与 Qwen3‑VL‑Plus 进行查询分解与候选验证。

**📊 数据集**

使用自建的 VGD‑Bench 数据集，包含 777 张真实仪表板、2167 条人工编写的自然语言查询以及图表检索与图表内定位的完整标注。

**📈 对比分析**

在图表检索、图表内定位和整体 VGD 三种评估场景下与多类别基线（MLLM、文献模型、Fine‑tune 模型）对比，VGD‑Net 在整体 VGD 上实现平均 IoU 57.06、准确率 66.82、APs 等指标均优于所有基线。

**⚠️ 局限性**

主要局限是阶段一检索错误会显著影响阶段二定位效果，且该方法高度依赖人工标注，缺乏弱监督或少样本适应能力。

---

## 351. Tracking and Limiting Unsupported Claim Propagation in Multi-Agent LLM Systems: A Provenance-Aware Framework

**arXiv ID:** 2611.26957 | [PDF](https://arxiv.org/pdf/2611.26957v1)

**作者:** Zijia Cao `[一作]` (University of Chicago), Hanqing Dou `[通讯]` (University of California San Diego)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多智能体LLM流水线中评估未被证实声明的传播与变更，并提出了保持声明源、支持状态、证据与传输历史的“源信息感知框架”，通过让每个代理接收结构化的原始声明及其源信息来限制意义漂移与不支持细节的累积。

**💡 创新点**

创新点在于将完整的声明源信息嵌入到每个代理输入，构建可追溯的源信息链，并通过五类双向NLI关系、UDA和语义漂移等指标系统量化声明在多模型流水线中的变化。

**🔧 技术方法**

使用的技术包括：预训练的DeBERTa-NLI模型进行双向文本推理，MiniLM-L6句子嵌入计算语义漂移，基于句子拆分与NLI判断的UDA计算，JSON结构化传输与检查的源信息维护；实验使用了多模型推理（Gemma、Granite、Llama、Mistral、Qwen）并固定温度、top-p等参数。

**📊 数据集**

采用FEVER数据集中的“NOT ENOUGH INFO”类别的185,445条声明作为实验样本，确保每条声明在所有实验条件下保持一致。

**📈 对比分析**

通过对比同一五阶段流水线的基线（文本传输）与源信息感知框架，基线在同质模型下等价率从74.73%降至62.67%，异质模型下降至31.33%；源信息感知框架在同质模型下等价率提升至96.95%，在异质模型下提升至95.73%；同时UDA、语义漂移、扩展、矛盾等指标均显著下降，表明源信息感知能大幅抑制声明变更。

**⚠️ 局限性**

主要限制包括：仅评估固定顺序的五阶段流水线，未探讨不同模型顺序、分支或循环结构；未实现声明验证或阻断机制；源信息的完整性检查仅在单阶段进行，未保证跨所有阶段保持一致；实验仅使用单一未被支持类别的声明，未检验对已支持或矛盾声明的表现。

---

## 352. Multi-Tier Edge Triage: Optimizing Latency for Safe Clinical Agents

**arXiv ID:** 2611.26914 | [PDF](https://arxiv.org/pdf/2611.26914v1)

**作者:** Joshua Broyde `[一作]` (Google Cloud), Sarita A. Joshi `[通讯]` (Google Cloud)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了一套多层边缘筛选架构，能够在企业临床工作流中实时检测生成式AI的安全合规性，防止PHI泄露、范围超出、阅读水平不合适以及品牌完整性受损。

**💡 创新点**

创新点在于将轻量化的 Tier‑1 序列分类器部署在本地应用服务器上，能即时处理清晰查询，只有模糊或高风险查询才升级到云端 Tier‑2 LLM，显著降低网络延迟和解码时间。

**🔧 技术方法**

使用技术包括 335M 参数的 BGE‑Large 编码器、对比双子基准构建、端到端表示微调、以及多策略的二元分类和规则检测。

**📊 数据集**

所用数据集为 MedSafetyBench、ASQ‑PHI 以及基于四项企业政策（Generic HealthCorp）生成的长匹配对比双子基准。

**📈 对比分析**

与零射击云 LLM 判断器比较，本地微调模型在四项安全政策上的 out‑of‑fold 准确率为 98.90%（云 LLM 为 99.20%），召回率 97.8%–98.2%，误拒率仅 0.0%–0.4%，执行时间 39.0 ms，比同步云评估快 14 倍以上。

**⚠️ 局限性**

限制在于对语义安全和阅读层次检查仍存在误差，导致部分风格化错误未被捕获；此外，模型在处理极端长文本或复杂非结构化 PHI 时的鲁棒性还有待提升。

---

## 353. Learning Adaptive Defense Strategies: Fine-Tuned LLMs for Network Security Games under Network Mobility

**arXiv ID:** 2611.27180 | [PDF](https://arxiv.org/pdf/2611.27180v1)

**作者:** Md Abu Sayed `[一作]` (University of Texas at El Paso), Christopher Kiekintveld `[通讯]` (University of Texas at El Paso)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过离线求解网络移动后蜜罐分配博弈，使用LLM直接预测混合均衡策略，从而消除实时求解开销。

**💡 创新点**

①将动态安全博弈的求解视为可摊销优化任务，LLM做无求解器的近似策略生成器；②对开放权重LLM进行LoRA微调，显著提升均衡预测质量；③发现验证损失与博弈任务性能负相关，强调需使用可验证的博弈理论指标进行模型选择。

**🔧 技术方法**

使用LLM（Qwen、LLaMA、Phi等）+ LoRA微调、Unsloth加速、4‑bit QLoRA量化；通过JSON序列化输入/输出、精确收益矩阵、LP求解器作为标注oracle；评估指标包括exploitability、support F1、TV、KL、validity。

**📊 数据集**

生成270个基于Watts–Strogatz小世界网络（8–25节点）的样本，对每个节点逐一移除并重新求解博弈，得到1,524个留存测试实例，包含攻击图、移动事件、动作集合、收益矩阵和LP均衡策略。

**📈 对比分析**

对比非学习基线（均匀、贪心、Mock‑Oracle）、零射手开放权重模型、微调开放权重模型和专有模型；使用统一输入/输出格式，评估exploitability、support F1等指标。结果显示微调模型平均降低约45% exploitability，部分微调模型（Mistral、LLaMA）比所有零射手模型低于均匀分配基线，并在移动类型细分中恢复路径相关性，专有模型表现不如微调模型。

**⚠️ 局限性**

①LLM输出有效性低，许多预测不满足概率分布；②推理延迟仍高于传统LP求解，无法在更大网络或多蜜罐预算下实时；③验证损失与任务指标不一致，需手动选取最佳检查点；④实验仅覆盖≤25节点和单蜜罐预算，缺乏可扩展性验证。

---

## 354. Search From the Start: Search-Native Pretraining for Language Models

**arXiv ID:** 2611.26809 | [PDF](https://arxiv.org/pdf/2611.26809v1)

**作者:** Kaustubh Ponkshe `[一作]` (EPFL), Martin Jaggi `[通讯]` (EPFL)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出在预训练阶段通过在文本中插入搜索查询和检索结果，教导语言模型学会何时发起搜索以及如何利用检索到的外部信息来生成后续文本。

**💡 创新点**

核心创新在于将搜索能力嵌入标准的下一词预测预训练流程，避免后训练阶段单独加入检索；使用大规模普通语料与本地检索生成的查询/响应对，形成可持续更新的知识获取机制。

**🔧 技术方法**

技术实现包括：1）使用基于 BM25 的本地索引对大规模网页语料（如 Wikipedia、FineWeb、Common Crawl）检索生成响应；2）在预训练时仅对原始文本和搜索查询计算损失，检索结果作为上下文；3）采用 Decoder‑only Transformer（SmolLM2、NanoChat 等）从零开始训练；4）评估时结合 Tavily API 与 BM25 检索。

**📊 数据集**

训练语料：覆盖 100B+ token 的搜索增强文本，来源于 Wikipedia、FineWeb‑Edu、Common Crawl 等公开网页；评测数据集包括 OpenBookQA、TriviaQA、ARC、NaturalQuestions 等知识问答数据集，以及 Multi‑hop QA 集合（Search‑R1）。

**📈 对比分析**

实验通过与 FLOP‑匹配的传统预训练模型（SmolLM3‑3B、Qwen2.5‑3B 等）对比，结果显示 1.4B 搜索预训练模型在八项知识密集型 QA 上平均得分至少比 3.3B FLOP‑匹配基线高 3–5%；在标准 SFT 以及高比例搜索微调后优势仍未被完全消除；此外，模型在检索来源切换到 Tavily 后性能进一步提升。

**⚠️ 局限性**

局限性包括：1）在闭合书籍推理任务上的提升有限；2）注解阶段需要大量检索成本，且依赖本地索引；3）模型规模增大时，参数知识的增强可能削弱搜索优势；4）检索源与真实网页的差异可能影响迁移效果；5）需进一步探索查询频率、响应长度与训练成本的平衡。

---

## 355. BeyondEgo: Learning to Act Beyond Egocentric Perception via Humanoid Ego-Exo Foresight

**arXiv ID:** 2611.27047 | [PDF](https://arxiv.org/pdf/2611.27047v1)

**作者:** Zhaoshan Yang `[一作]` (Tongji University), Jin Pan `[通讯]` (Chinese University of Hong Kong)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 BeyondEgo 模型，扩展人形机器人视觉-语言-动作（VLA）感知到同步外部摄像头的空间信息和双视角前瞻的时间信息，并在此基础上构建了 HELM 基准，用于评估闭环人形机器人在复杂环境中的运动与操作能力。

**💡 创新点**

创新点：1) 通过在动作专家级别直接融合同步外部视角的 DINOv3 特征，显著提升空间感知；2) 在训练阶段使用双视角前瞻监督，仅保留轻量级适配器用于部署，避免前瞻模型带来的推理负担；3) 提供首个同步 ego-exo 视角、闭环评估且包含多种场景变化的 HELM 基准。

**🔧 技术方法**

技术细节：采用 Qwen3-VL-2B-Instruct 作为视觉语言基底，冻结 DINOv3 作为外部视角编码器，使用残差适配器将未来信息注入 VLM 产生的特征；前瞻专家为 Transformer 预测 0.6 秒后双视角 DINOv3 表示；动作专家为多模态扩散变换器（MM-DiT）并用流匹配训练；部署时仅保留 4.2M 参数的适配器；训练中加入 exocentric dropout、实时块化推理（RTC）等技术。

**📊 数据集**

数据集：在 Isaac Sim / MuJoCo 环境下收集 500 条演示（5 个任务，每个任务 100 条），每条演示包含同步头摄像头（ego）与外部摄像头（exo）图像；HELm 基准包含 5 个全身任务，每个任务 10 条在分布内测试 + 200 条离散场景（布局、资产、材质、灯光及组合）用于评估泛化。

**📈 对比分析**

对比方法：与仅使用 ego 视角的 Ψ0、VLM fusion（将两视角一起输入 VLM）以及 Feature fusion（将 exo 的 DINOv3 特征拼接后直接送入动作专家）进行比较。在分布内评估中，BeyondEgo 成功率 92%（相比 Ψ0 48%，Feature fusion 84%，VLM fusion 74%）；在泛化评估中，BeyondEgo 65%（相比 Ψ0 23.5%，VLM fusion 49.5%，Feature fusion 55%），平均提升 8–10 个百分点，显示出最优的整体表现。

**⚠️ 局限性**

局限性：需要同步且覆盖良好的外部摄像头，限制了大范围运动的覆盖；双视角演示的采集成本较高；前瞻监督仅在训练阶段使用，适配器参数仍不小；未来工作需探索多摄像头协同、可变前瞻时长和更灵活的相机布局。

---

## 356. One Behavior, Out of Order: Deterministic Execution and Parallel Simulation of ROS 2 Applications

**arXiv ID:** 2611.26676 | [PDF](https://arxiv.org/pdf/2611.26676v1)

**作者:** Chengzi Huang `[一作]` (Uppsala University), Wang Yi `[通讯]` (Uppsala University)

**通讯引用:** 19257 | [OpenAlex ID](https://openalex.org/A5100721121)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

给 ROS 2 应用引入基于时间的协调，使其行为确定性，进而实现无状态化的 out‑of‑order 并行模拟。

**💡 创新点**

创新点在于使用空记录提前确定所有事件依赖，并在此基础上设计了激活流依赖计算与循环缓冲区存储的并行模拟算法，避免了传统同步/异步模拟的全局屏障和无效消息。

**🔧 技术方法**

采用 LET（逻辑执行时间）调度、空记录、激活流（activation stream）依赖计算、循环缓冲区（ring buffer）存储以及工作窃取（work stealing）并行算法。

**📊 数据集**

使用工业用例（Autoware Reference System、Fleet‑N、Stereo 视觉流水线）以及 1385 个合成 ROS 2 图进行评估。

**📈 对比分析**

与顺序、同步并行（Rendezvous、Tardis）和异步并行（Eden）模拟比较，256 核上实现近 91% 理论加速，速度提升 2.6–228 倍，消息占用显著降低（最多 6.3 GB→0.5 GB）。

**⚠️ 局限性**

目前不支持同节点多发布者同主题以及运行时结构动态变化，需要进一步改进。

---

## 357. Littlestone Dimension and Its Dual

**arXiv ID:** 2611.26721 | [PDF](https://arxiv.org/pdf/2611.26721v1)

**作者:** Pooya Hatami `[一作]` (Ohio State University), Chavdar Lalov `[通讯]` (Ohio State University)

**通讯引用:** 30 | [OpenAlex ID](https://openalex.org/A5120367166)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

证明了概念类 C 的双 Littlestone 维数 d*(C) 与 Littlestone 维数 d(C) 的上界关系：d*(C) ≤ 2^{O(d(C)^2)}，并给出指数依赖是必要的。

**💡 创新点**

突破了以往的 2^{2^{O(d(C))}} 双指数上界，首次给出单指数（但平方指数）上界；并通过阈值序列与扩展阈值的两种结构展开归纳证明。

**🔧 技术方法**

利用了 Ramsey 型二叉树引理（子加性性质）、阈值序列/扩展阈值的构造、以及双 Littlestone 维数的递归公式；整体为组合数学与模型论相结合的证明。

**📊 数据集**

无数据集，论文纯理论推导，无实验验证。

**📈 对比分析**

与先前的 2^{2^{O(d(C))}} 上界相比，新上界显著收敛；作者指出已知下界 d*(C) ≥ Ω(log d(C))，且存在例子（全二元假设类）证明指数依赖不可避免。

**⚠️ 局限性**

上界中指数中的平方项尚未被证明是最佳的，仍可能进一步下降到线性；论文未给出更精确的下界来证明这一点，仍为开放问题。

---

## 358. PP-CFG: Polarity-Preserving Guidance for Controllable Few-Step Image and Video Generation

**arXiv ID:** 2611.27266 | [PDF](https://arxiv.org/pdf/2611.27266v1)

**作者:** Junyi He `[一作]` (Fudan University), Yanwei Fu `[通讯]` (Fudan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PP‑CFG架构，使少步蒸馏模型在保持可调式CFG和负向提示的同时，仅用一次视觉骨干前向。

**💡 创新点**

将正负文本提示分别通过Scale‑Time Polarity Mixer细化，并在同一视觉流中通过NASA式注意力相减实现可调式引导和目标抑制。

**🔧 技术方法**

结合Few‑step diffusion distillation（DMD/LCM）、Polarity‑Preserving CFG、Scale‑Time Polarity Mixer、NASA/Polarity blocks、Cross‑attention 与 Joint‑attention等技术。

**📊 数据集**

在Wan2.1视频模型与Qwen‑Image图像模型上训练，并构建NegElement‑1K 1,000对正负提示的图像/视频基准。

**📈 对比分析**

与多种蒸馏基线对比，PP‑CFG在视频4步获得最高VBench总分81.46，图像8步取得最佳GenEval/DPG分数；相对教师在目标抑制上提升约10–20%。

**⚠️ 局限性**

对极端提示或高复杂度场景的抑制仍有限，且需要额外的混合器与注意力模块导致计算开销略高。

---

## 359. How Far Does Lossless Re-tokenization Move an RLVR Update?

**arXiv ID:** 2611.27083 | [PDF](https://arxiv.org/pdf/2611.27083v1)

**作者:** Zhe Dong `[一作]` (University of Maine at Presque Isle), Fang Qin `[通讯]` (Stanford University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对同一文本的不同子词分词方式（重编码）对强化学习回报验证（RLVR）更新的影响进行定量评估，提出一种基于单次更新的配对测量方法，并对比重编码导致的参数更新偏移与换一批数据时的偏移。

**💡 创新点**

创新点在于：①提出了一个可量化的“重编码效应”指标（R），将单次更新的偏移与不同批次更新的偏移做比较；②系统探究了重编码的“剂量”与“位置”对更新方向和幅度的影响；③发现重编码导致的更新方向在批次间具有一致性，并在一个简化的表格模型中证明即使对梯度进行任意加权也无法消除这一差异；④在真实流水线中审计了重编码频率，说明实际重编码量远低于实验剂量。

**🔧 技术方法**

技术方法包括：对子词词表进行可拆分token的选择与拆分；使用RLVR训练框架（LoRA或全微调）与五种替代目标函数；利用AdamW优化器的多步更新与梯度截断；计算参数更新向量差距的归一化距离c以及相对不同批次的比值R；使用对角线投影的对齐统计量ρ衡量重编码更新方向的一致性；在表格模型中进行解析证明；对真实重编码情况做统计审计。

**📊 数据集**

数据集与实验：使用Qwen2.5-0.5B、Qwen3-1.7B、Qwen3-8B和Llama-3.1-8B四个模型；任务为三位数加法与GSM8K（含chat模板与不含chat模板）两种推理任务；在每个模型、任务、参数空间与目标函数下随机种子进行多次实验。

**📈 对比分析**

比较方法：在相同检查点采样一批rollout，用canonical编码和拆分编码分别进行一次更新，计算c并与换一批数据时的c做比值R；通过ρ统计不同批次重编码更新方向的一致性；在固定rollout训练中对比canonical arm与re-encoded arm的最终准确率差异。性能结果显示：R随拆分比例上升可达到1（与换批同幅），早期拆分的token对R影响更大；ρ随剂量上升显著提升，表明方向一致；在固定rollout实验中canonical arm略占优但差异不显著。

**⚠️ 局限性**

限制与不足：①只在单一剂量、单一任务、单一目标函数下评估固定rollout的最终准确率，未覆盖所有训练阶段；②实验剂量与真实流水线中的重编码比例相差较大；③对齐统计仅在单一检查点内计算，未证明方向随训练进程保持一致；④表格模型证明仅适用于独立表格的情况，无法直接推广到共享参数的网络；⑤批量大小与训练噪声对R的影响未充分评估。

---

## 360. GEG-SAM: Gaze Exploration-Guided Segment Anything Model for Weakly Supervised Polyp Segmentation

**arXiv ID:** 2611.27183 | [PDF](https://arxiv.org/pdf/2611.27183v1)

**作者:** Ao Wang `[一作]` (Sichuan University), Zizhou Wang `[通讯]` (A*STAR Institute of Advanced Intelligence and Computing)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了GEG‑SAM框架，实现基于眼动的弱监督息肉分割；

**💡 创新点**

创新点在于将眼动视为结构化的视觉搜索过程，联合建模时间动态、空间拓扑与眼点对应的视觉内容，并将生成的语义眼动标记注入SAM的特征层；

**🔧 技术方法**

核心技术包括Gaze Dynamics‑Structure Encoder (GDSE)用于捕捉时间与空间关系，Gaze‑Guided Semantic Token Composer (GSTC)将眼点与局部图像特征融合，以及Gaze Feature Injection (GFI)将融合后的眼动信息注入冻结的SAM3编码器；

**📊 数据集**

使用了五个公开息肉数据集（Kvasir‑SEG、CVC‑ClinicDB、CVC‑ColonDB、CVC‑300、ETIS）以及相应的眼动数据；

**📈 对比分析**

与现有眼动监督与弱监督方法相比，GEG‑SAM在Kvasir‑SEG上平均Dice达到86.43%，比GAM提升10.14pp、比WeSAM提升9.30pp；在跨域测试中平均Dice为85.26%，优于Polyp‑PVT与SAMed‑2；

**⚠️ 局限性**

局限性包括仅处理单帧图像，未针对视频连续眼动进行建模；依赖高质量眼动追踪数据，若眼动数据不足或噪声大，性能可能受限；

---

## 361. Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs

**arXiv ID:** 2611.26902 | [PDF](https://arxiv.org/pdf/2611.26902v1)

**作者:** Qiang Zhou `[一作]` (University of Pittsburgh), Jingtong Hu `[通讯]` (University of Pittsburgh)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种小VLM与大VLM级联的方案，利用小模型的KV缓存作为大模型的视觉记忆，并通过一种逐层线性映射实现记忆迁移；

**💡 创新点**

创新点在于发现大模型需要的是可用的记忆而非忠实复制，于是引入了通过大模型自身答案进行蒸馏的线性映射（Distilled memory map），该映射能够让大模型仅重新编码当前帧而不需重新观看历史，显著降低计算成本；

**🔧 技术方法**

使用的技术包括Transformer的KV缓存迁移、逐层线性映射、基于答案的无标签蒸馏（KL+KV损失）、Hybrid路径部署、硬决策训练以及对不同层的干预分析；

**📊 数据集**

实验使用的主要数据集为RoboMME（机器人操纵任务）、OpenEQA（室内场景问答）、VSI-Bench、HM3D以及ScanNet；

**📈 对比分析**

与传统的重新观看、文本记忆、剪枝重观看等方法比较，内存级联在28–47%更低的计算量下，与重新观看的性能相当（误差≤0.7分），在RoboMME上比单独小模型提升约3分，且对更大模型（32B、Gemma3等）同样有效；

**⚠️ 局限性**

局限性包括：记忆映射仅在64帧以内验证；每个writer-reader对需单独训练映射；对定位（point）任务的提升低于分类（choice）任务；未研究更长视频流或跨模型共享映射的情况。

---

## 362. Cross-Platform Production Signatures of State-Linked Information Operations

**arXiv ID:** 2611.27137 | [PDF](https://arxiv.org/pdf/2611.27137v1)

**作者:** Lin Tian `[一作]` (University of Technology Sydney), Marian-Andrei Rizoiu `[通讯]` (University of Technology Sydney)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文通过仅基于时间戳与粗粒度动作类型的“生产签名”来描述信息作战操作的工作规律，并评估其在不同平台间的可迁移性与特异性。

**💡 创新点**

创新点在于提出“生产签名”这一全新指标，证明其能跨平台识别账号且能捕捉群组同步发布行为，同时不依赖内容、个人资料或网络结构。

**🔧 技术方法**

使用的技术包括行为克隆（next‑token预测）学习动作–时间间隔分布、间隔自适应模型、逻辑回归/梯度提升分类器、BLOC序列编码，以及基于方差的同步度统计。

**📊 数据集**

数据集涵盖Twitter四个国家级信息操作（IRA、伊朗、委内瑞拉、中国、GRU）及对应Reddit IRA名单，另外使用938个公开Telegram频道。

**📈 对比分析**

方法评估以AUC为指标：同平台内AUC > 0.98；跨平台使用Twitter训练的生产签名在Reddit上取得 AUC ≈ 0.94；内容模型跨平台仅 0.74–0.81；BLOC 与行为策略跨平台保持高 AUC；Telegram 同步度统计显著高于对照组。

**⚠️ 局限性**

局限性包括无法将账户精准归属到具体操作、对时间策略的干预鲁棒性未知、跨平台迁移受时区窗口相似性限制，以及仅使用粗粒度动作，缺乏内容和网络信息的补充。

---

## 363. Mind the Delta: Early Pruning of RTL Variants for Agentic Hardware Optimization

**arXiv ID:** 2611.27035 | [PDF](https://arxiv.org/pdf/2611.27035v1)

**作者:** Siyang Cai `[一作]` (University of Chinese Academy of Sciences), Ying Wang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种训练无关的实时选择框架，利用基准设计的完整合成日志与候选RTL变体的合成日志进行匹配，快速停止无法胜过已完成可行实现的变体，从而在 agentic RTL 优化中实现高效且准确的变体选择。

**💡 创新点**

创新点在于：①将基准设计的合成轨迹作为参考，消除不同变体间共享的映射膨胀误差；②基于其他设计族的残差分位数校准一侧下界，使停止决策既安全又无训练成本；③在保持商用综合工具不变的前提下，实现了任何时刻可停止的完整综合竞争，显著降低评估开销。

**🔧 技术方法**

技术包括：基准设计与变体同步的日志配对、残差分位数校准、基准比率的使用、可行 incumbent 维护、时钟裕度判定以及并行合成调度。

**📊 数据集**

使用了包含 43 个设计族、330 个 agent 写入的 RTL 变体的基准，涵盖从小型块到百万级单元处理器，来源于开源项目（Rocket、Boom、Gemmini、VexRiscv 等）以及自动化编码工具生成。

**📈 对比分析**

与 10 种主流快速评估方法（学习预测器、合成代理、编码代理、多保真度阶段剪枝）对比，提出的方法在 43 个族中 100% 正确选择最优变体，平均评估工作量减少 44.9%（1.81× 加速），在大型设计上加速达 2.18×，而其他方法多在准确率或安全性上存在明显缺陷。

**⚠️ 局限性**

局限性包括：仅在 Synopsys Design Compiler、NanGate 45 nm 工具链和面积目标下验证；未覆盖时序/功耗等其他 QoR 目标；对极端时钟紧缩或非常小规模设计的适用性尚未充分测试。

---

## 364. Self-Evolving Beyond Human-Crafted Experience for EDA Scripting

**arXiv ID:** 2611.26912 | [PDF](https://arxiv.org/pdf/2611.26912v1)

**作者:** Yang Liu `[一作]` (CXMT), Wenjian Yu `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发了ZhuLong-SE，一个可执行驱动的LLM代理，通过跨任务自演化机制学习并复用EDA脚本经验，实现自动化设计脚本生成。

**💡 创新点**

引入三层治理式记忆体系与可靠性门控，并结合精确静态检查和反射式回馈，解决EDA脚本零样本反馈稀疏与跨任务泄漏问题。

**🔧 技术方法**

语义+关键词检索、文档增强、隔离沙箱执行、精确静态检查、治理式记忆写入/读取、可靠性门控以及anti‑leakage检索等技术。

**📊 数据集**

EDA‑Eval‑PyAether 158任务基准（含98/20/40实例不共享拆分）。

**📈 对比分析**

与原ZhuLong基线在五种LLM骨干上对比，提升11–16个百分点；自演化后在40任务测试集上从66%提升至80%，超过人造经验基线77.5%。

**⚠️ 局限性**

静态检查无法覆盖所有语义错误；自演化门控需多轮验证，单轮学习受限；对不同EDA工具和语言的泛化尚未验证。

---

## 365. Cohomological $k$-Equivalence Does Not Capture Graph Isomorphism

**arXiv ID:** 2611.26672 | [PDF](https://arxiv.org/pdf/2611.26672v1)

**作者:** Asterios Gkantzounis `[一作]` (Technical University of Crete), Asterios Gkantzounis `[通讯]` (Technical University of Crete)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5117562418)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文构造了一对非同构图，使得在任意有限变量计数逻辑的余拓扑下，它们在所有有限k都保持同构（即满足 cohomological k‑equivalence）而实际上并不同构。

**💡 创新点**

证明了 cohomological k‑equivalence 在任何固定k下都无法识别图同构的极限，给出了一个基于 2 与 3 的奇偶性结合的构造；同时给出了一个多项式时间可判别的判别不变量。

**🔧 技术方法**

利用了同调/共调（cohomological）一致性、Bézout 组合、Johnson 图和 CFI 结构的线性方程求解技术，以及可逆映射同构的整数解存在性等代数工具。

**📊 数据集**

论文未使用传统机器学习或图数据集，而是构造了特殊的图实例（参数化为 k）来展示理论极限。

**📈 对比分析**

对比方法是比较 cohomological k‑equivalence 与传统 Weisfeiler–Leman（k‑WL）和 Diophantine 近似等技术，实验表明在这些实例上 WL 及其整数松弛都无法分辨，而 cohomological 2‑equivalence 仍能区分。

**⚠️ 局限性**

局限在于构造的图规模呈双指数增长，且证明仅适用于图同构问题，未给出对更一般结构或更高效率算法的改进。

---

## 366. GeoSDF-Diff: Geometry-Aware Diffusion Planning for Collision-Free Trajectories

**arXiv ID:** 2611.27373 | [PDF](https://arxiv.org/pdf/2611.27373v1)

**作者:** Yueming Li `[一作]` (Shenzhen Technology University), Lijun Sun `[通讯]` (Shenzhen Technology University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种 Geometry-Aware Diffusion Planner GeoSDF-Diff，用多尺度SDF条件进行轨迹控制点去噪，生成无碰撞轨迹。

**💡 创新点**

将全局与局部SDF信息通过跨注意力注入DiT‑X去噪器，并引入轨迹级违反惩罚，直接在去噪过程中实现几何可行性。

**🔧 技术方法**

采用条件去噪扩散模型（DDPM）、DiT‑X Transformer、SDF多尺度编码器、轨迹级损失（SDF、平滑、长度、违反）等。

**📊 数据集**

在二维点机器人场景中使用64×64 SDF地图，训练集由四种地图族共1,000个场景（10,000条轨迹），测试集为Dense‑clusters、Square‑clusters和MPD L‑R等未见的稠密环境。

**📈 对比分析**

与RRT‑Connect+GPMP2、MPD‑Bspline、PRESTO及其优化版对比；GeoSDF‑Diff在无后处理/引导的情况下平均成功率达到65.2%，比MPD‑Bspline提升约25%，比PRESTO提升约29%，且运行时间仅0.057 s。

**⚠️ 局限性**

仍无法匹配搜索‑优化方法的极致可行性（≈99%），对三维/高维配置空间的推广尚未验证，且对超大规模障碍布局的鲁棒性仍待进一步评估。

---

## 367. ORBIT: Offset-based Activation Steering for Cross-lingual Multimodal Fact Verification

**arXiv ID:** 2611.27421 | [PDF](https://arxiv.org/pdf/2611.27421v1)

**作者:** Huy Trieu `[一作]` (University of Science), Tung Le `[通讯]` (University of Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统性分析小型视觉语言模型在跨语言多模态事实验证中的性能崩塌，并提出了无训练的推理时激活调节方法 ORBIT 以缓解低资源语言下的跨模态对齐失败。

**💡 创新点**

创新点在于：①使用并行双语实例一次性估计语言特定的偏移向量；②在推理时仅通过加性修正隐藏层表示实现跨语言对齐；③通过偏移向量范数提供轻量级的诊断工具。

**🔧 技术方法**

技术主要包括：激活调节（activation steering）与向量偏移估计、同语言对齐向量的符号对齐聚合、层级位置选择与逆范数加权调节。

**📊 数据集**

数据集为机器翻译扩展的 MOCHEG‑XL（九种语言、约2,442个测试样本）以及 1,490 条验证样本，其中包含断言、文本证据、视觉证据和三分类标签。

**📈 对比分析**

通过与 CLAIM、Self‑translate CoT、Pivot NLLB 等基线对比，ORBIT 在低资源语言下平均提升约 2%‑3% 的准确率并在整体 F1 上领先，同时推理时间仅比零样本基线提升 19%，显著优于翻译式方法。

**⚠️ 局限性**

局限性包括：1）对高资源语言可能产生过度修正导致误差；2）偏移向量对翻译质量敏感，低质量翻译会影响校准；3）MOCHEG‑XL 为机器翻译而非人工验证，缺乏真实世界噪声和多源信号的评估。

---

## 368. What Does Matching a Human Intervention Effect Establish? Benchmark Checks for LLM Simulations

**arXiv ID:** 2611.27186 | [PDF](https://arxiv.org/pdf/2611.27186v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 369. MedicalRSI: Medical Recursive Self-Improving Agents via Clinical Model-Harness Co-Evolution

**arXiv ID:** 2611.26806 | [PDF](https://arxiv.org/pdf/2611.26806v1)

**作者:** Ziqing Wang `[一作]`, Kaize Ding `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种递归自我改进框架（Clinical Model‑Harness Co‑Evolution），使得医学代理既可以通过强化学习改进自身模型参数，又能在每一轮中利用自身失败自动生成并更新医学 harness（技能、工具和记忆），从而在执行过程中不断自我迭代。

**💡 创新点**

核心创新点是让同一模型同时扮演"代理"和"改进策略"两角色，利用失败经验生成可执行的 harness，并通过增益奖励（相对基线的收益）来训练两角色共进化；同时在测试时无需再训练，直接通过少量失败来生成新的 harness。

**🔧 技术方法**

采用基于GRPO的强化学习、SFT warm‑start、共享权重的单模型架构、增益奖励函数、记忆机制以及 prompt 模板来实现 harness 的生成与执行。

**📊 数据集**

实验使用了 MedAgentBench（10+ 新任务，共 32 个临床工作流）和 MedAgentGym（MIMIC‑III/ eICU 数据集），对 10 轮训练后进行评估。

**📈 对比分析**

与无改进、上下文层改进（Reflexion、SkillX 等）以及权重层改进（STaR、GRPO Solver）等多组基线相比，方法在两大 benchmark 上均取得显著提升：在 MedAgentBench 上整体成功率从 36.9% 提升至 85.1%，在 MedAgentGym 上提升至 89.0%；尤其在未见工作流上提升 20–30 分，且在新数据库 eICU 上仍保持较高性能。

**⚠️ 局限性**

局限性包括：需要手工定义 harness 结构（技能、工具格式）；对任务分布和失败样本的敏感性，若失败率过低可能难以获取足够学习信号；训练成本高（需多轮 RL 与大模型）；对非结构化或多模态输入的适应性待验证；在极端低资源环境下效果可能不佳。

---

## 370. TrojanMaxxing: One-shotting VLM Backdoors with Weight-Space Programming

**arXiv ID:** 2611.27425 | [PDF](https://arxiv.org/pdf/2611.27425v1)

**作者:** Ahaan Dabholkar `[一作]` (Purdue University), Saurabh Bagchi `[通讯]` (Purdue University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用前沿大语言模型的权重空间编程（WSP）技术，在不进行梯度优化或再训练的情况下，将视觉触发器植入视觉语言模型（VLM）中，实现后门攻击；

**💡 创新点**

创新点在于：①将WSP扩展至视觉触发器的后门植入；②通过LLM代理实现零训练、零梯度的权重编辑；③联合设计视觉触发器和权重编辑，保持模型清洁性能；

**🔧 技术方法**

主要技术包括：权重空间编程（WSP）、前沿LLM代理（Claude Opus 5.5、Mythos 5.2）、基于工具的权重编辑、触发器函数设计、静态权重审计；

**📊 数据集**

使用的数据集涵盖：内容审核（UnsafeBench、HarmImageTest、PolicyShiftBench）、OCR/多模任务（TextVQA、DocVQA、SROIE、VLGuard）、自定义策略与验证集；

**📈 对比分析**

实验对比显示：对内容审核模型的攻击成功率在93%–100%，OCR模型攻击成功率在78%–100%；模型清洁性能仅下降≤0.02，所需权重编辑<0.03%，前向推理量少于3600次，成本<30美元；

**⚠️ 局限性**

局限性包括：对极端图像尺寸触发成功率下降、对某些模型/精度（4/8位量化）后门易被破坏、受LLM安全过滤限制；整体上需进一步验证在更大规模、不同任务上的鲁棒性。

---

## 371. LingSante-Pulse: Unified Physiological Pretraining through Random-Feature Prediction and Cross-Modal Alignment

**arXiv ID:** 2611.27206 | [PDF](https://arxiv.org/pdf/2611.27206v1)

**作者:** Linxiao Yang `[一作]`, Liang Sun `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出LingSante-Pulse，一种统一的多模态生理信号预训练框架，能够在部分重叠的记录中学习跨模态关系并保留单模态信息。

**💡 创新点**

创新点在于将共享的信号条件化Transformer、配对对齐、结构化随机特征预测（RFP）与掩码波形重建组合成一个完整的训练目标，从而实现单一冻结编码器在多通道组合与单通道推理间的无缝迁移。

**🔧 技术方法**

技术包括FiLM通道条件化Transformer、同步对齐损失与SIGReg正则化、三分支随机教师（卷积、时频、相关性）与Ridge适配器、以及掩码自编码器重建。

**📊 数据集**

使用的数据集覆盖38个生理任务配置（PPG、ECG、EEG、EMG）以及HMC、Sleep-EDF、Apnea-ECG、BIDMC、MIT-BIH等睡眠与心率相关数据。

**📈 对比分析**

通过与多种专用基础模型（如AnyPPG、ECG-FM、CBraMod、NormWear、SleepFM等）以及多模态睡眠模型进行冻结编码器对比，LingSante-Pulse在每个域均取得首位或前两名，并在睡眠分期、单通道心率估计、心律分类、EEG任务等指标上获得最优或接近最优的F1/AUROC。

**⚠️ 局限性**

局限性包括对共享通道对齐的依赖，完全缺失某一信号时跨模态优势受限；对非标准采样率或大规模高频通道的适配仍需改进；模型规模与推理成本的实际部署评估尚未充分展开。

---

## 372. Actions Speak Louder than Words: Stateful Privacy Sniffing of LLM Agents via Behavioral Covert Channels in Web Interactions

**arXiv ID:** 2611.27046 | [PDF](https://arxiv.org/pdf/2611.27046v1)

**作者:** Ruichao Liang `[一作]` (Nanyang Technological University), Yang Liu `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对LLM代理在 Web 交互过程中的行为进行主动探测，利用代理的可观察请求形成隐式的行为隐写通道，从而恢复用户查询意图与个人画像。

**💡 创新点**

提出基于状态化的层次化探测框架（Snoop Tree、Adaptive Probe Constructor、Snoop Agenda），实现对隐写通道的主动、细粒度探测，并设计 Selection‑Obfuscated Retrieval 作为新的隐私防护方案。

**🔧 技术方法**

利用多种行为隐写通道（Choice、Count、Sequence 等）构造探测网页，构建决策树与状态机；使用 LLM 与结构化知识库（MeSH、FIBO、EuroVoc 等）生成探测候选；采用 ReFinED 与 Wikidata 进行语义匹配与重构。

**📊 数据集**

500 个 StackExchange 问题（涵盖 27 主题）与 378 条 iKAT 对话（26 角色）作为查询与多轮画像评估数据集。

**📈 对比分析**

在 12 种代理–模型组合上评估，平均查询概念 F1 为 0.89、画像 F1 为 0.80；相比现有的 WRA、AgentPrint、SkillWatermark 与显式外泄基线，恢复细粒度的精确度明显更高；现有防御对该攻击几乎无效，F1 下降 ≤0.03；引入 Selection‑Obfuscated Retrieval 后，F1 下降至约 0.39，且成本提升约 4×页面请求、1.1–1.3× token 与时间。

**⚠️ 局限性**

需要能够在代理访问的网页中注入可控链接且代理会访问该链接；实验仅覆盖四款代表性代理与三种 LLM 背骨；对网络延迟、缓存策略等外部因素的鲁棒性未系统评估；Selection‑Obfuscated Retrieval 对 LLM 计算与网络请求成本存在显著提升。

---

## 373. Affine-Image Propagation for Tight and Scalable Neural Network Verification

**arXiv ID:** 2611.26978 | [PDF](https://arxiv.org/pdf/2611.26978v1)

**作者:** Hong-Ming Chiu `[一作]` (University of Illinois at Urbana--Champaign), Richard Y. Zhang `[通讯]` (University of Illinois at Urbana--Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种新的基于 affine‑image SDP‑CROWN 的可扩展 ℓ₂ 逼近方法，用于对非 Lipschitz 正则化的 ReLU 网络进行稳健性验证。

**💡 创新点**

创新点在于：① 用 affine‑image 约束保留预激活集合的几何形状，显著降低几何损失；② 联合优化线性界面参数 g、h，减少功能损失；③ 将原始 SDP 约束替换为可通过缩放对角占优实现的稀疏条件，兼顾紧致性与计算可行性；④ GPU 加速实现，兼容卷积稀疏结构。

**🔧 技术方法**

核心技术包括半正定规划（SDP）松弛、scaled Gershgorin 对角占优约束、Affine‑image 预激活域建模、梯度上升优化线性界面参数、与 branch‑and‑bound 的无缝集成。

**📊 数据集**

在 MNIST、CIFAR‑10 等公开数据集上对 5 个非 Lipschitz 以及 9 个 Lipschitz 正则化网络进行实验。

**📈 对比分析**

与 α‑CROWN、原 SDP‑CROWN、β‑CROWN、BICCOS 等基线对比，单纯方法已显著提升已验证准确率（提升 17%–57.5%），加入 branch‑and‑bound 后提升 27%–63.5%，并在大多数基准上实现最快、最强的证书性能。

**⚠️ 局限性**

局限性包括：① 仍为不完整验证器，对极大规模网络的可扩展性尚有限；② 需要手动调节缩放参数 l；③ 对某些网络结构（如深度残差网络）仍可能产生较大几何失真。

---

## 374. Prune to Expand: Training Prompt Enhancers with Less to Imitate

**arXiv ID:** 2611.27229 | [PDF](https://arxiv.org/pdf/2611.27229v1)

**作者:** Tianhe Wu `[一作]` (City University of Hong Kong), Kede Ma `[通讯]` (City University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种文本仅用的“Prune to Expand (P2E)”方法，通过从密集描述中删减与用户意图弱相关的句子，来改进提示增强（Prompt Enhancement）模型的监督信号。

**💡 创新点**

①发现高生成增益的密集描述并不一定能提供更好的监督；②提出按句子级别删减并以每个示例的删除上限为约束的P2E策略，减少冗余信息同时保持用户意图。

**🔧 技术方法**

使用大型语言模型（如Qwen3.5-9B）进行句子重要性评估与删减候选生成；采用SFT（autoregressive）方式微调PE模型；评估指标包括HPS v2、PickScore、MIQA、VQ‑R1。

**📊 数据集**

利用 ImageNet 1024×1024 图片生成的密集字幕与短提示对；从 LSDIR 60K 高质量图像抽样得到的四类（Landscape、Design、Portrait、General）数据集；在四个视觉生成器（Qwen‑Image‑2.1‑Turbo、Z‑Image‑Turbo、MOSS‑Image、BAGEL）上进行评测。

**📈 对比分析**

与直接生成、原始PE模型、以及未删减的密集描述SFT进行对比。P2E 在人类偏好（HPS v2）和视觉质量（PicS、MIQA、VQ‑R1）上均取得平均 0.15‑0.30 分的提升，且跨模型、跨类别表现一致。

**⚠️ 局限性**

主要局限在于删减决策依赖所选 LLM 的判断，可能对不同 LLM 版本或更大规模数据集的适用性不确定；过度删减仍可能丢失有用信息，需精细调节删除阈值。

---

## 375. TrajRealm: A Large-Scale Point Tracking Dataset with Diverse Dynamics

**arXiv ID:** 2611.27136 | [PDF](https://arxiv.org/pdf/2611.27136v1)

**作者:** Ni Yang `[一作]` (Fudan University), Junting Dong `[通讯]` (Shanghai AI Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了大规模合成点跟踪数据集 TrajRealm（十千万帧）及其基准 TrajRealm-Bench，覆盖游戏与机器人仿真中的多样交互与运动。

**💡 创新点**

创新点在于利用现代游戏丰富的三维资产与物理交互生成高精度、几何一致的 2D/3D 点轨迹，并结合多源仿真实现跨场景、跨目标的训练与评估。

**🔧 技术方法**

采用游戏渲染插件与仿真重渲染技术获取 RGB、深度、相机参数及持久点轨迹，提供局部 SE(3) 运动注解；并设计多类别基准与标准化评估指标（AJ、APD、OA）。

**📊 数据集**

使用的数据集为 TrajRealm（游戏 + 机器人仿真）以及其子集 TrajRealm-Bench；在实验中还引用 SynthVerse、Dynamic Replica、LSFOdyssey、ADT、DriveTrack、PStudio 等公开数据进行对比与迁移学习。

**📈 对比分析**

通过在有/无 GT 深度与相机姿态的两组条件下评估 CoTracker、TAPIP3D、TrackEverything 等方法，发现不同场景对 3D 精度影响显著；Fine‑tune 后 TAPIP3D 在 Dynamic Replica 的 AJ_3D 提升 7.83，TrackEverything 在实景 minival 上提升 2.28，证明 TrajRealm 对模型提升有显著作用。

**⚠️ 局限性**

局限性包括：对极端高速运动、长时序持续移动以及大尺度世界空间位移仍难以保持高精度；即便使用 GT 也有 Hard、Long‑horizon、Transportation 等子集表现不佳；此外，数据仍为合成环境，真实世界细节与噪声仍需进一步验证。

---

## 376. Typhoon OCR 2: From Thai Document Transcription to Multilingual, Multi-Task OCR

**arXiv ID:** 2611.27494 | [PDF](https://arxiv.org/pdf/2611.27494v1)

**作者:** Surapon Nonesung `[一作]` (Typhoon, SCB DataX), Warit Sirichotedumrong `[通讯]` (Typhoon, SCB DataX)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了开源的Typhoon OCR 2模型，支持单一模型完成泰语文档的定向提取和全页解析，并扩展到多语种（英文、中文）页面解析。

**💡 创新点**

创新点在于将目标提取与全页解析统一到同一模型，并通过GRPO（Group Relative Policy Optimization）对序列级奖励进行强化学习，显著降低循环错误并恢复失败页面。

**🔧 技术方法**

使用的技术包括：基于Infinity‑Parser2‑Flash和OvisOCR2的开源预训练模型，监督微调（SFT）结合长上下文和动态分辨率，随机图像变换，GRPO强化学习与KL正则，Prompt‑driven任务指令与多任务输出格式。

**📊 数据集**

采用了多来源混合数据集：泰语专用 146,647 条（含 99,616 条规则过滤、47,031 条新标注），以及 273,680 条公开多语种（英文、中文）多任务文档、视觉问答与文本指令，总计 420,327 条样本，并通过 PII 过滤与图像哈希校验保证数据质量。

**📈 对比分析**

评估使用五大公开基准（ThaiOCRBench、MDPBench、olmOCR‑bench、SEA‑DocBench、OmniDocBench），2B 版本在 ThaiOCRBench 及 Thai MDPBench 上分别达到 59.9 与 80.7 分（显著优于基线），在英文基准上保持与基线相当，整体性能优于多种公开基线但仍落后于专有模型。

**⚠️ 局限性**

主要局限包括：结构化输出在表格、公式等复杂内容上仍不均衡；在全页解析中仍出现循环或截断失败，尤其是照片版页面；泰语标注主要依赖大模型自动生成，人工复核集中于边界框，文本和布局错误仍可能存在。

---

## 377. Realistic Robustness Evaluation of Tabular Data Driven Models

**arXiv ID:** 2611.26818 | [PDF](https://arxiv.org/pdf/2611.26818v1)

**作者:** Miguel Silva `[一作]` (Polytechnic of Porto), Isabel Praça `[通讯]` (Polytechnic of Porto)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AURORA 开源框架，用于在表格数据上系统评估机器学习模型的鲁棒性，集成多种攻击方法、评估指标，并引入距离调整机制来过滤不现实的对抗样本。

**💡 创新点**

创新点包括：①针对表格数据的距离调整机制，结合欧氏距离与汉明距离统一衡量扰动真实度；②将有效、无效扰动的惩罚融入指标，避免低质量对抗样本影响评估；③构造单一的鲁棒性综合分数，并提供标准与最坏情景两种评估视角。

**🔧 技术方法**

使用的技术包括：黑盒对抗攻击（BoundaryAttack、HopSkipJumpAttack、ZOO、A2PM）、特征级约束与无约束版本、欧氏+汉明距离计算、惩罚与距离调整公式、CA/AA/MR/ASR/AD 等多维指标以及基于这些指标的综合鲁棒性得分。

**📊 数据集**

实验数据集为 GeNIS（网络入侵检测）表格数据，经过递归特征消除后保留 4 个特征（SrcBytes、SIntPktMax、Proto、State），并使用 CatBoost 分类器进行测试。

**📈 对比分析**

通过对比约束与无约束攻击方法，结果显示：A2PM 在无约束下获得最高 MR；但约束版的 BoundaryAttack 与 HopSkipJumpAttack 产生更真实的扰动。标准评估下约束攻击得分 65，最坏情景 13；无约束攻击得分 83（标准）和 73（最坏），说明不现实攻击会导致鲁棒性高估，真实评估则更严谨。

**⚠️ 局限性**

局限性包括：①攻击方法多源自图像领域，尚未充分适配表格数据；②仅评估了 CatBoost，缺乏对其他模型和数据集的验证；③约束版攻击在有效率上仍受限；④实验样本量有限（1000 样本）；⑤未覆盖更丰富的攻击目标与特征间依赖约束。

---

## 378. CodeReviewBench: An Automated Benchmark for Evaluating the Software Factory

**arXiv ID:** 2611.26942 | [PDF](https://arxiv.org/pdf/2611.26942v1)

**作者:** Ashley Zhang `[一作]` (Martian), Fazl Barez `[通讯]` (Martian)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

基于开发者对 AI 代码评审建议的实际响应，构建了一个可持续、可扩展的在线基准，并公开了超过 50 万条评审结果与 18 种工具的评估数据。

**💡 创新点**

创新点在于：① 把开发者的实际操作（是否提交修复）作为客观可信的评审质量信号；② 通过 GitHub 公共事件流实现持续更新的实时基准；③ 结合在线数据与离线专家标注集，形成多维度验证与排行榜。

**🔧 技术方法**

使用技术包括：GitHub 事件流解析、自动化工具建议与开发者提交匹配、LLM（多模型）评判器、统计过滤与多阶段验证、自动化 Leaderboard 生成。

**📊 数据集**

数据集包括：每日抓取的 GitHub PR 事件，约 500K+ 评审案例（覆盖 18 个工具）；离线专家标注集 136 条问题（来源于 50 条 PR 的 3 名专家标注）。

**📈 对比分析**

比较方法：将在线评审结果与离线专家标注集、人工专家评审以及厂商发布日志对齐；通过精度排名验证工具实力，并能在厂商更新后数日内检测到性能提升。性能方面，评审工具的相对精度排名与离线标注集高度一致，且能及时捕捉到产品级改进。

**⚠️ 局限性**

局限性包括：1）并非所有有用建议都会被开发者采纳，导致信号噪声；2）依赖 GitHub 公共事件，可能忽略私有仓库或非 GitHub 平台的工作；3）对不同语言、项目规模的覆盖仍有限；4）LLM 判断器虽高一致性，但仍需人工核查。

---

## 379. Borrow the Style, Follow the Prompt: Progressive Reference-Guided Stylization

**arXiv ID:** 2611.27360 | [PDF](https://arxiv.org/pdf/2611.27360v1)

**作者:** Yu Lin `[一作]` (Sun Yat-sen University), Zhilin Zhao `[通讯]` (Sun Yat-sen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在参考引导的文本到图像生成中，ProStyle学习了内容抑制的风格先验，并通过弱辅助层注入和原生残差路径，逐步融合参考纹理，同时保持文本指令的内容控制。

**💡 创新点**

创新点包括：①使用跨内容监督构造内容抑制的风格先验；②层级不对称的弱辅助层注入以强化风格表达；③原生参考路径提供残差校正并与风格路径联合优化；④构建134K跨内容训练对与多级测试的ReConStyle基准。

**🔧 技术方法**

技术上结合了CLIP特征重采样、可训练的风格令牌、跨注意力层注入、残差双路径学习以及多阶段优化，全部在Stable Diffusion v1.5上实现。

**📊 数据集**

数据集为ReConStyle，包含134K张同风格不同主体/背景的图像对，分为快速测试与完整测试两种设置，评估时使用CLIP、Qwen3-VL提取的对象级指标。

**📈 对比分析**

与DEADiff、StyleShot、Less is More等同骨干的基线相比，ProStyle在ReConStyle-Full上四项自动指标均优于其他SD1.5方法，平均排名1.86，且在用户研究中在文本对齐和综合适用度上分别以84.36%和50.90%显著领先。

**⚠️ 局限性**

局限性包括：①仍依赖SD1.5骨干，对更强大模型的迁移需要进一步验证；②跨内容对的构造依赖手工筛选，可能在某些风格上不够丰富；③对极端强烈的参考条件（α>1）可能导致内容泄漏。

---

## 380. A Continuous Relaxation of Quantization-Aware Training

**arXiv ID:** 2611.27056 | [PDF](https://arxiv.org/pdf/2611.27056v1)

**作者:** Kartikeya Bhardwaj `[一作]` (Qualcomm AI Research), Harris Teague `[通讯]` (Qualcomm AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种连续松弛的量化感知训练（RelaxQAT），通过在训练过程中逐步降低模型权重量化位宽，从而缓解低比特量化的平坦损失面问题。

**💡 创新点**

创新点在于：1) 将量化位宽从整数平滑地变为浮点值并随训练动态调整；2) 在 SEQ 量化器中改用“先舍入再裁剪”策略（RelaxSEQ），使极值随位宽变化平滑；3) 通过半周期 + 全周期的位宽调度，让模型在训练中多次经历从高位宽到低位宽的缓冲，从而跳出平坦区。

**🔧 技术方法**

技术包括：量化感知训练（QAT）、Stretched Elastic Quantizer（SEQ）与其改进 RelaxSEQ、连续位宽调度（基于正弦函数的 ω），以及使用 STE 进行反向传播；实现上在 PyTorch 上做了 Triton 核实现以保持训练速度。

**📊 数据集**

使用的数据集：LLM 训练集（DCLM‑Edu、tulu‑3‑sft‑olmo‑2‑mixture、xlam‑function‑calling‑60k）以及 VLM 训练集（Llama‑Nemotron‑VLM‑Dataset‑v1），在 10B token 上训练 Qwen3、Qwen3.5、Gemma3 等模型。

**📈 对比分析**

与 5 种主流 QAT 基线（ParetoQ、LLM‑QAT、UPQ、BitNet、WinQ）及 PTQ 基线 AdaScale 对比，RelaxQAT 在 2‑bit 权重量化上平均提升 1% 以上（最高 7.7%），在 Vision‑Language 任务上提升 0.8% 以上，且在多种模型和任务上取得最优或次优成绩，训练时间与基线相当或略快。

**⚠️ 局限性**

局限性：仅验证了权重量化，未直接应用于激活或 KV‑cache 量化；对极低位宽（如 1‑bit）和其他模型族的泛化仍需进一步评估；位宽调度的超参数（m_s、n）对结果有一定影响，需要手动调优。

---

## 381. Acceleration of Sparse and Irregular Workloads through Energy-Efficient Stream Extensions

**arXiv ID:** 2611.26939 | [PDF](https://arxiv.org/pdf/2611.26939v1)

**作者:** Paul Scheffler `[一作]` `[通讯]` (ETH Zurich), Paul Scheffler (ETH Zurich)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一套能量高效的硬件流扩展（SSSR），通过解耦稀疏数据的内存访问与执行，显著加速稀疏与不规则计算工作负载，并在多核 RISC‑V 集群、432 核 Occamy 2.5D 系统以及 AXI Pack 总线扩展上验证其可扩展性。

**💡 创新点**

创新点：①将稀疏指令级流（affine、indirect、intersection、union）集成到 ISA，形成轻量化、可插拔的核心耦合扩展；②实现全硬件解耦的索引与数据流处理器，支持多种稀疏格式（CSR、CSF 等）；③在 12 FinFET 12LP+ 工艺上实现 432 核双芯片阵列，配合 HBM2E 与 AXI Pack，展示从单核到多芯片级别的能量与性能提升。

**🔧 技术方法**

使用技术包括：RISC‑V Snitch 8 核集群、SSSR ISA 扩展、硬件循环 frep、内存流地址生成器（indirection/egress）、索引比较器、AXI Pack 端点流、DRAMSys 3.6 Gb/s HBM2E 模拟、Synopsys Design Compiler/PrimeTime RTL 综合与功耗估算、LLVM/Clang 编译器插件。

**📊 数据集**

数据集：SuiteSparse 真实稀疏矩阵、合成稀疏向量、Stencil（Poisson、Laplacian）代码、Spiking Neural Network 推理数据、模型预测控制（MPC）样例、Hetero 12 Gb/s HBM2E 内存。

**📈 对比分析**

比较方法：将 SSSR 版本与基准 RISC‑V、单一 SSR、软件优化（如 BLIS、cuSPARSE、TileSpMV）以及其他硬件加速器（EIE、Sparse‑TPU、SISA 等）进行基准测试。结果显示：
- 单核 SSSR 在稀疏矩阵向量乘上实现 4.7–5.6×加速、FPU 利用率提升至 80%；
- 8 核集群可达 4.9–5.9×加速、FPU 利用率 70–80%；
- 432 核 Occamy 在稀疏线性代数上实现 5–7×加速、能耗降低 2–3×；
- 与 GPU（cuSPARSE、TileSpMV）比较，SSSR 在 FP64 上达到 47% 以上峰值利用率，明显优于现有 GPU 加速器。

**⚠️ 局限性**

局限性：
① 需要专门的 ISA 与编译器支持，非通用 CPU 需重构；
② 受限于核心内部内存宽度和地址生成器的单端口，极稀疏小规模任务时设置与收敛开销可能抵消收益；
③ 多芯片级互连延迟与 DRAM 带宽瓶颈对性能影响显著，需额外的流包装与双缓冲策略；
④ 仅针对数据流式稀疏计算，复杂非线性或控制流丰富的算法仍需进一步优化；
⑤ 额外的硬件面积与功耗虽然小，但在极低功耗嵌入式场景仍需评估。

---

## 382. SceneRuler: Measuring Physical Quantities in Video via Minimal Sufficient Reconstruction

**arXiv ID:** 2611.26994 | [PDF](https://arxiv.org/pdf/2611.26994v1)

**作者:** Yumeng He `[一作]` (University of California, Los Angeles), Chenfanfu Jiang `[通讯]` (University of California, Los Angeles)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `14d48e9d-0069-4ad9-996a-1d5968216998` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种针对视频的定量物理推理方法——最小充分重建（Minimal Sufficient Reconstruction），通过构造一个只包含必需对象、支持关系、相机和运动规律的场景程序，并利用物理定律对运动参数进行拟合，从而得到所需的物理量。

**💡 创新点**

创新点包括：① 用视觉‑语言模型只写场景结构，所有度量量值通过先验、常识尺寸或拟合得到，保证场景可解释；② 引入“读出（read‑out）”机制，判断场景是否真正测量到所问量；③ 通过物理导向的拟合将重力等常数作为尺度锚点，使尺度和运动量的识别更稳健；④ 将最终答案与重建场景的证据相结合，提供可追溯的推理链。

**🔧 技术方法**

使用的技术包括：视觉‑语言模型（GPT‑6.1）生成场景结构与轨迹，Grounding‑DINO + SAM 进行对象检测与分割，Depth Pro 提供相机深度估计；物理导向的最小二乘拟合（包含尺度锚定、运动规律、先验约束）来估计所有自由参数；读出检验条件与不确定度传播；最终答案由另一次 VLM 调用在视频帧与重建场景的证据上推断。

**📊 数据集**

在 QuantiPhy 验证集（158 题）进行评估，同时收集 27 个野外视频（54 题）进行无标注实验。

**📈 对比分析**

与 Code as Worlds、PhysMind、Thinking with Spatial Code、Gemini‑4‑Carbon、Qwen3.5‑27B 等基线对比，采用平均相对准确率（MRA）评价。总体 MRA 为 0.709，超过最强基线 Gemini‑4‑Carbon（0.624）和 Code as Worlds（0.554）等；在 47 个读出确定的题目中，MRA 达 0.781，显著高于未确定题目（0.679）。

**⚠️ 局限性**

局限性包括：① 仅处理固定摄像机，移动摄像机需额外建模；② 场景几何采用粗粒度形状，细节不足时可能导致读出不确定；③ 依赖先验事实，若先验错误或缺失会影响结果；④ 结构在拟合后不再修正，若初始结构错误可能无法恢复；⑤ 计算成本较高，涉及多轮 VLM 调用与拟合。

---

## 383. Answering the persona, not the person: Over-determination in the construction of synthetic respondents

**arXiv ID:** 2611.27449 | [PDF](https://arxiv.org/pdf/2611.27449v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 384. Mind World Models: Joint Learning of Motor EEG Representations and Action-Conditioned World Dynamics

**arXiv ID:** 2611.27029 | [PDF](https://arxiv.org/pdf/2611.27029v1)

**作者:** Weikang Qiu `[一作]` (Yale University), Rex Ying `[通讯]` (Yale University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文提出 Mind World Model（MWM），利用同步EEG与自我视角视频，通过将EEG视为潜在动作来进行世界模型预训练，生成无需动作标签的运动EEG表示；

**💡 创新点**

创新点在于：①将EEG编码为“潜在动作”以预测未来视觉状态，实现运动EEG无监督预训练；②引入 Lateralization‑Aware Transformer（LAT），通过镜像与帧平均分离双侧与侧向特征，显式编码半球对应关系；

**🔧 技术方法**

采用了自监督视觉预测（基于冻结的DINOv2编码器）、InfoNCE 对齐损失、AdaLN 适配器以及 LAT 结构；

**📊 数据集**

使用了 EgoBrain 数据集，包含 63.28 小时同步 EEG 与自我视角视频，覆盖 40 名受试者；

**📈 对比分析**

在 7 个回归和 9 个分类下游任务中，MWM 的线性探针在均值 Pearson 相关率和均值平衡准确率分别提升约 15%–20% 于最佳基线；在真实动作条件下的未来预测中，MWM 预训练的世界模型实现了最低 L1 错误并产生最大的动作敏感度；

**⚠️ 局限性**

主要限制包括：实验仅评估预测误差，未验证端到端的机器人控制；缺少跨域（如不同任务或不同 EEG 通道布局）的泛化评估；

---

## 385. Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use Subordinate Material

**arXiv ID:** 2611.27139 | [PDF](https://arxiv.org/pdf/2611.27139v1)

**作者:** Jinxiang Xie `[一作]` (Nanjing University), Yuxin Su `[通讯]` (Sun Yat-sen University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了一个评估多代理LLM系统中高级代理角色完整性的基准方法。

**💡 创新点**

创新点在于引入“角色完整性”（role integrity）概念，构建基于真实机构层级（OpenReview、最高法院、议会）的受控实验，以直接测量高级代理对下级材料的权责履行。

**🔧 技术方法**

使用多种大语言模型作为高级代理，配合规则检查、基于LLM的评分器以及人工标注，对其输出进行行为评估。

**📊 数据集**

数据集来源于ICLR公开的提交与评审记录、美国最高法院口头辩论记录、英国议会委员会报告及证词，随后对下级材料进行人工/自动构造干预。

**📈 对比分析**

通过对八个LLM的12,000+代理回合进行实验，计算三类错误率（多数追随、错误采纳、限制丢失）并求得角色完整性得分，结果显示在未提示下错误采纳率极低，但在多数追随和限制丢失方面仍普遍出现，提示责任提示能部分缓解但不消除失效。

**⚠️ 局限性**

局限性包括仅涵盖三类任务且缺乏真实下级交互、未对层级结构对比验证、构造审稿时同时改变多维度属性等问题，影响对角色完整性真正因果关系的推断。

---

## 386. Fooling the Mind Twice: Human Detection of Audiovisual Deepfakes Under Repeated Exposure and Corrective Feedback

**arXiv ID:** 2611.27080 | [PDF](https://arxiv.org/pdf/2611.27080v1)

**作者:** Ammarah Hashmi `[一作]` (Academia Sinica), Hsin-Min Wang `[通讯]` (Academia Sinica)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在两阶段实验中让参与者在收到纠正反馈后再次评估相同的音视频深伪内容，研究重复曝光与反馈对人类识别准确性的影响。

**💡 创新点**

首次系统量化重复评估与纠正反馈对判断稳定性、错误纠正率及新错误产生的影响，并比较两大公开数据集的跨阶段表现。

**🔧 技术方法**

采用配对t检验、聚类标准误的逻辑回归以及错误转移矩阵等统计方法对准确率、误报/漏报率及判断一致性进行分析。

**📊 数据集**

使用FakeAVCeleb（20真20假）和Deepfake Detection Challenge (DFDC)（20真20假）两组音视频样本。

**📈 对比分析**

结果显示相较于第一次评估，第二次准确率提升约4.7%（FakeAVCeleb）和8.6%（DFDC），FNR显著下降但FPR无明显变化；整体效果仍低于70%，说明人类判断仍不可靠。

**⚠️ 局限性**

实验将重复曝光与反馈合并、使用同一素材、受评估顺序影响、样本量有限、仅限英语MTurk受试者，且未评估与AI协同的效果。

---

## 387. A Unified Framework for Intelligent Cyber Attack Detection and Alert Correlation

**arXiv ID:** 2611.26824 | [PDF](https://arxiv.org/pdf/2611.26824v1)

**作者:** Miguel Silva `[一作]` (Technical University of Porto), Isabel Praça `[通讯]` (Technical University of Porto)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并实现了一个统一的 SOC 框架，集成了网络流量收集、签名+机器学习+异常检测的混合检测模块以及基于 PrefixSpan 的序列模式挖掘的告警关联模块，并在模拟真实多阶段攻击的测试环境中验证其有效性。

**💡 创新点**

创新点在于：① 将签名检测、异常检测和多类别分类三种检测方法无缝融合，形成“混合检测”；② 利用 HERA 2.0 提取流量特征与 Suricata 结合的双流数据流，兼顾粒度和计算效率；③ 采用细化版 PrefixSpan 对“告警类型”和“类型-来源”两种序列进行并行挖掘，实现实时且上下文感知的告警关联；④ 以人机协同的方式将检测结果转化为可操作的安全洞察。

**🔧 技术方法**

技术与工具包括：Suricata IDS（签名匹配）、HERA 2.0（流量导出与特征提取）、XGBoost（多类别流分类）、Isolation Forest（异常检测）、PrefixSpan（序列模式挖掘）、Wazuh（主机端告警收集）、Python/Scikit-learn、Kibana/Wazuh Dashboards 等。

**📊 数据集**

用于训练的数据集：GeNIS、HIKARI-2021、CICIDS-2017、UNSW-NB15；用于评估的是真实攻击场景下的自建测试平台，涵盖了恶意软件利用、横向移动、侦察和数据外泄等多阶段攻击。

**📈 对比分析**

性能对比：XGBoost 在多类别分类任务中取得宏平均 F1‑score 85%，Isolation Forest 在异常检测任务中取得宏平均 F1‑score 76%。在实际攻击实验中，框架能够同时检测到签名告警、异常告警和分类告警，且通过序列关联将数百条告警压缩为数条有意义的关联日志，显著提升了 SOC 分析师的作业效率和事件响应速度。

**⚠️ 局限性**

局限性：① 对计算资源要求较高，需在具备一定算力的服务器上部署；② 目前只覆盖网络流量与主机端告警，未涵盖钓鱼、恶意软件深度分析等方面；③ 关联算法基于固定阈值的 PrefixSpan，可能在高噪声环境下产生误关联；④ 在不同规模的企业（尤其是中小企业）环境下的可扩展性与泛化性尚未充分验证。

---

## 388. InterBand-EMG: A Cross-Band Foundation Model for Generalizable Hand-Pose Estimation

**arXiv ID:** 2611.27289 | [PDF](https://arxiv.org/pdf/2611.27289v1)

**作者:** Zongheng Guo `[一作]` (Politecnico di Milano), Manuela Ferrario `[通讯]` (Politecnico di Milano)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于跨频带交互的自监督基础模型 InterBand‑EMG，用于连续手势角度估计，并发布了新的八通道 EMG‑to‑pose 数据集 SynapMyo。

**💡 创新点**

创新点在于：① 在量化阶段为低、中、高频带分别训练代码表，并通过跨频带注意力实现频带间的重建互相监督；② 采用该量化器生成的离散码作为目标，进行结构化遮蔽码预测，从而学习更具迁移性的 EMG 表征。

**🔧 技术方法**

使用了向量量化（VQ‑VAE）、Transformer、跨频带注意力、遮蔽编码预测以及后续的手势角度回归解码器。

**📊 数据集**

实验使用了 emg2pose（370 小时，16 通道）和 SynapMyo（1,228 小时，8 通道，499 受试者）两套数据集。

**📈 对比分析**

与多种基线（vEMG2Pose、EMG‑GPT、NeuroPose、Sensing 等）在 held‑out 用户、任务、两者同时的三种泛化设置下对比，InterBand‑EMG 在所有设置下均取得 MAE 降低 0.3–4.2%（emg2pose）和 16.2–24.2%（SynapMyo）的提升。

**⚠️ 局限性**

局限性包括：数据集动作分布不均、标注来源异质、缺乏实时推理与延迟评估、跨频带互操作的具体贡献难以单独归因以及对初始姿态依赖较强。

---

## 389. Proper Conflict-Free Coloring of Cocomparability, Co-Bipartite, and Distance-Hereditary Graphs

**arXiv ID:** 2611.26645 | [PDF](https://arxiv.org/pdf/2611.26645v1)

**作者:** Gautam K. Das `[一作]` (Indian Institute of Technology Guwahati), Kamal Santra `[通讯]` (Indian Institute of Technology Guwahati)

**通讯引用:** 55 | [OpenAlex ID](https://openalex.org/A5032937760)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究正冲突自由着色（Proper Conflict‑Free coloring）在 cocomparability、AT‑free、co‑bipartite、α≤2 与距离继承图等多类图上的可计算性与最优着色算法，提出了 NP‑完全性证明、匹配表征以及多种多项式时间算法。

**💡 创新点**

首次证明 cocomparability/AT‑free 图正冲突自由着色在最优色数 ω(G) 上是 NP‑完全的；给出 co‑bipartite 图的完整匹配表征并实现 O(n^6) 的最优算法；对 α≤2 的图给出 O(n^6 log n) 的优化算法；对距离继承图提供 O(n^11+m) 的最优着色算法。

**🔧 技术方法**

使用 Exact Cover 3 归约、图匹配与可行性枚举、二叉搜索树 pruning、二叉分解与颜色状态合并、动态规划与图匹配算法等理论技术。

**📊 数据集**

无实验数据，本文全部以理论证明与算法复杂度分析为主。

**📈 对比分析**

相较于先前仅针对特殊子类给出的着色数界与算法，本工作在更广泛的图类中确立了可计算性边界，并给出多项式时间算法（如 co‑bipartite O(n^6)，α≤2 O(n^6 log n)，距离继承 O(n^11+m)）。

**⚠️ 局限性**

算法的时间复杂度仍较高，特别是距离继承图的 O(n^11+m) 仍可进一步改进；仅针对固定图类给出结果，未讨论固定颜色数的可计算性；对 α≥3 的图仍保持 NP‑完全，缺乏更细粒度的可行性划分。

---

## 390. A Lightweight Method for Validating Simulated LiDAR Configurations Based on Point Density over Distance

**arXiv ID:** 2611.27379 | [PDF](https://arxiv.org/pdf/2611.27379v1)

**作者:** Benedikt Degenstein `[一作]` (Technical University of Applied Sciences Augsburg), Carsten Markgraf `[通讯]` (Technical University of Applied Sciences Augsburg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种轻量、平台无关的方法，用对象边界框内点云点数随距离变化的密度曲线来验证仿真LiDAR配置的信号层面一致性。

**💡 创新点**

创新点在于引入“密度相似度得分(DSS)”——一种基于MAPE的归一化相似度指标，能够量化真实与仿真点密度随距离的匹配程度，并实现不依赖语义标签、模拟器内部或感知模型的验证。

**🔧 技术方法**

主要技术包括：基于3D包围盒的点云计数、距离离散化、MAPE转换为DSS，以及在CARLA中自定义非均匀垂直束角的LiDAR配置。

**📊 数据集**

数据集为一套现实世界RoboSense Ruby Plus LiDAR记录与对应的CARLA仿真数据，包含行人与车辆沿直线远离静止ego车辆的场景。

**📈 对比分析**

比较方法是对每个配置计算距离离散区间内的点数平均值，并用DSS对真实与仿真分布进行相似度评估；实验表明自定义束角配置在车辆/行人场景中分别获得0.709/0.559的DSS，明显优于均匀束角的Simple配置（0.267/0.213）。

**⚠️ 局限性**

局限性包括仅评估几何点密度、单一场景与单次运行、未考虑强光/噪声、反射率等信号属性，以及DSS与下游感知性能的直接相关性仍需进一步研究。

---

## 391. Efficient Natural Language to Temporal Logic Translation for Real-Time Physical AI

**arXiv ID:** 2611.27489 | [PDF](https://arxiv.org/pdf/2611.27489v1)

**作者:** Oliver Schön `[一作]` (ETH Zürich), Lars Lindemann `[通讯]` (ETH Zürich)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于 System-One 模型的自然语言到线性时序逻辑（LTL）翻译器，分为一遍翻译器和层级翻译器，可在机器人本地无云端运行。

**💡 创新点**

创新点在于：①不需要生成式文本，而是将翻译拆解为预定义问答并用渲染器构造公式；②一遍翻译器一次前向传播即可完成翻译；③层级翻译器能处理任意嵌套结构；④通过问答结构保证输出始终为合法 LTL 公式，且翻译次数可预先限制。

**🔧 技术方法**

技术方法：使用开源 System-One 预训练模型（421M 参数）进行多选分类；自定义问答集与渲染器；在两种架构（one‑pass 与 level‑by‑level）上 fine‑tune；对比多种本地 LLM（Qwen2.5、Llama3.2 等）与云端大模型。

**📊 数据集**

数据集：
- Open APs（678 条指令与公式，涉及未知原子命题），
- Unseen Patterns（1,992 条指令，包含未见的嵌套公式），
- Mine Rover（599 条指令，针对矿坑车的特定命令模式）。

**📈 对比分析**

比较方法：在 M1 Pro 和 Raspberry Pi 5 上测算翻译准确率、耗时与能耗；将 System-One 与 20 款本地 LLM 以及云端参考模型（Claude Opus 5.5、GPT‑6 Astra）对比。结果显示，System-One 在本地模型中准确率与最优的 Qwen2.5‑1.5B 相当或更高；在 Open APs 与 Mine Rover 上一遍翻译器的运行时间低 5–10 倍、能耗低 7–14 倍；层级翻译器在 Unseen Patterns 上准确率最高但耗时与能耗约为 Qwen2.5‑1.5B 的 3–4 倍。

**⚠️ 局限性**

局限性：
- 需要先手工或自动构建覆盖目标 LTL 片段的问答集；
- 一遍翻译器只能在已知问答集下工作，无法处理完全未知的逻辑结构；
- 层级翻译器的成本随公式深度线性增长，深层公式在 CPU 上表现较差；
- 目前仅验证了 LTL，扩展至其他时序逻辑仍需研究。

---

## 392. PCC: Proof-Carrying Credentials for Enhanced Anonymity and Post-Quantum Delegation

**arXiv ID:** 2611.27250 | [PDF](https://arxiv.org/pdf/2611.27250v1)

**作者:** Zixuan Wang `[一作]` (Hong Kong Polytechnic University), Bin Xiao `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出 Proof‑Carrying Credentials（PCC）框架，实现委托凭证的递归证明，达到深度无关的展示成本并隐藏委托拓扑。

**💡 创新点**

创新点：①定义 Enhanced Delegation Anonymity（EDA）双重隐私目标；②将委托链压缩为单一递归证明状态，消除展示与拓扑泄漏；③设计可切换的非均匀递归谓词，实现发行、委托和展示三阶段的安全递归验证；④提供兼容后量子安全的实例。

**🔧 技术方法**

核心技术：非均匀递增可验证计算（NIVC），递归零知识证明（Halo2、Plonky2/Fri），Poseidon 哈希，hash‑based（SPHINCS+）签名。

**📊 数据集**

实验使用合成的委托链（深度 1–16）、属性集与展示策略，未使用公开数据集。

**📈 对比分析**

与 P‑UC‑DAC、显式历史（Monolithic）以及非递归 Fractal 基线对比。PCC 证明大小保持 7.3 KB（Halo2）或 157.9 KB（Plonky2），验证时间 35–36 ms（Halo2）或 3.8 ms（Plonky2），委托步骤时间约 4.7 s（Halo2）或 1.0 s（Plonky2），全部与链深度无关；相对基线呈现线性增长。

**⚠️ 局限性**

局限性：仅在单台实验机上实现，链深度上限 16，未实现撤销、阈值委托、网络延迟或侧信道分析；后量子实例仍基于实验性证明系统，实际部署前需进一步评估。

---

## 393. EAST: Event-Anchored Surface Transport for Event-Based Gaussian Reconstruction

**arXiv ID:** 2611.27224 | [PDF](https://arxiv.org/pdf/2611.27224v1)

**作者:** Ke Luo `[一作]` (University of Science and Technology of China), Yang Cao `[通讯]` (University of Science and Technology of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

引入了Event-Anchored Surface Transport (EAST) 方法，将事件相机的时序信息与三维表面场结合，实现了高精度的连续表面重建。

**💡 创新点**

通过支持的三角化候选、事件测得的法向流以及基于冻结参考场的参考相对表面传输，实现了对高光梯度与深度补偿的分离，并在 Gaussian 渲染中实现了几何与图像残差的双向约束。

**🔧 技术方法**

采用了事件相机的时空事件流、SDF 与高斯渲染、三角化支持与法向流约束、参考相对中心位移、梯度投影冲突解决以及 Eikonal 正则化等技术。

**📊 数据集**

在 PAEv3D 以及 NeRF Synthetic 两个公开事件-视点数据集上进行了实验。

**📈 对比分析**

与现有的 Event-3DGS、E-3DGS 等事件监督高斯重建基线相比，EAST 在表面重建误差与参考视角渲染均有显著提升，并且能够高效提取网格。

**⚠️ 局限性**

仍受限于事件相机的边缘信息有限，对大范围全局光照变化的鲁棒性不足，且参考相对传输对初始场的依赖可能导致初始化不佳时收敛缓慢。

---

## 394. A Geometric Approach to the Classification of Locally-APN Niho-Type Power Functions

**arXiv ID:** 2611.27312 | [PDF](https://arxiv.org/pdf/2611.27312v1)

**作者:** Chenxu Wang `[一作]` (Capital Normal University), Jun Zhang `[通讯]` (Capital Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文提出几何方法，对有限域上局部APN Niho 型幂函数进行完整分类；

**💡 创新点**

创新点在于将差分均匀性条件转化为射影图的（q+1）-弧性质，并利用PG(3,q)的弧分类及其稳定子群完成了理论上的完整证明；

**🔧 技术方法**

主要技术包括射影几何、群论（PGL(4,q) 的稳定子群）、线性代数（对角化、特征值分析）以及 Niho 型幂函数的代数性质；

**📊 数据集**

无实际数据集，研究属于纯理论数学；

**📈 对比分析**

无实验比较与性能评估，研究通过严格证明完成，验证并证实了先前的猜想；

**⚠️ 局限性**

局限在于只针对 Niho 型幂函数，进一步推广到更一般的 q‑齐次函数或其他稀疏多项式仍需要进一步研究。

---

## 395. Backbone-Specific LoRA Geometry: Orthogonality, Superposition, and Provenance

**arXiv ID:** 2611.27201 | [PDF](https://arxiv.org/pdf/2611.27201v1)

**作者:** Kazuhiko Kawamoto `[一作]` (Chiba University), Hiroshi Kera `[通讯]` (Chiba University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了将LoRA适配器的低秩更新在多个预训练模型之间共享，并提供理论分析与实证验证。

**💡 创新点**

创新点在于：① 利用注意力对称性和可变换不变性推导跨模型LoRA更新的近正交性界限；② 通过局部梯度流模型解释同一模型不同随机种子训练的平均正相关；③ 设计基于余弦相似度的无坐标对齐骨干识别和已注册适配器检测方法。

**🔧 技术方法**

主要技术包括：LoRA低秩适配、注意力对称性分析、随机置换不变性、局部梯度流模型、SVD压缩、余弦相似度与最小二乘检测。

**📊 数据集**

使用七个ViT‑B/16 CLIP骨干（OpenAI、LAION‑400M、LAION‑2B、DataComp‑1B、DFN‑2B、MetaCLIP‑400M、MetaCLIP‑2.5B）和八个图像分类数据集（Cars、DTD、EuroSAT、GTSRB、MNIST、RESISC45、SUN397、SVHN）。

**📈 对比分析**

对比方法包括：Solo（单模型适配）、Exact sum（七模型之和）、LOMO sum（去除目标模型适配）、Joint shared（跨模型共同训练）以及SVD压缩。实验显示：Exact sum与Solo在大多数任务上误差<1.7pp，SVD‑16在保持约90%精度的同时将存储量降低到单个LoRA的1/7；余弦相似度在任务匹配和跨任务情况下实现100%骨干识别，已注册适配器检测在普通负样本下可实现100%AUROC，但对相同骨干/任务的硬负样本时TPR仅≈30%。

**⚠️ 局限性**

局限性包括：需要相同头数与维度的骨干；跨模型共享假设依赖于独立均匀方向的前置训练；跨任务识别仍需额外阈值；已注册适配器检测在硬负样本下效果不佳，且未考虑模型执行时的非线性交互。

---

## 396. Spatiotemporal Landslide Runout Reconstruction from Sparse Observations with LAPIS-SHRED

**arXiv ID:** 2611.27132 | [PDF](https://arxiv.org/pdf/2611.27132v1)

**作者:** Xingyue Zhang `[一作]`, J. Nathan Kutz `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

训练并验证了LAPIS‑SHRED模型，用于从极度稀疏的观测数据（单帧或短窗口）恢复山体滑坡流动的完整时空演化。

**💡 创新点**

将SHRED的隐空间时空推断与终端观测的静态填充相结合，首次实现仅凭一次性沉积观测即可反演整个滑坡过程，并支持前向与后向两种推断模式。

**🔧 技术方法**

采用双向LSTM编码器‑解码器（SHRED）与序列到序列时间模型，结合LAPIS的静态填充与隐空间逆向推断；训练数据基于物理模拟的MWR细胞自动机仿真集。

**📊 数据集**

利用Landlab中实现的MWR模型生成的多参数仿真集（包括2009年和2022年Mount St. Helens滑坡）以及现场LiDAR终端沉积数据作为训练与评估数据集。

**📈 对比分析**

与全时序SHRED上界、简单插值基线和SINDy‑SHRED比较；LAPIS‑SHRED在NMASE<5.8%、SSIM>0.96，且仅需单帧观测即可实现与上界相近的重建，显著优于基线。

**⚠️ 局限性**

仅基于MWR模拟，缺乏对三维流动、孔隙水动力等复杂物理的捕捉；未提供不确定性量化；缺乏现场中间时序数据的评估，需进一步跨场地验证。

---

## 397. Data-Driven Detection of Volatility Regimes and Sector-Level Asymmetry During Sovereign Default

**arXiv ID:** 2611.26967 | [PDF](https://arxiv.org/pdf/2611.26967v1)

**作者:** M. J. S. Ahamed `[一作]` (University of Moratuwa), Sandareka Wickramanayake `[通讯]` (University of Moratuwa)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用数据驱动方法识别股票市场波动性变化区间，并在各区间内估计EGARCH模型，分析行业层面的波动性异质性与宏观变量影响。

**💡 创新点**

创新点在于将回归式分段（动态规划+BIC）与统计检验（sup‑F、马尔可夫切换）结合自动确定波动性区间；在每个区间内用可解释的Student‑t EGARCH捕捉负面冲击的非对称效应，并通过修正相关性检验排除波动性提升带来的假象。

**🔧 技术方法**

主要技术包括：动态规划分段、sup‑F检验、两状态马尔可夫切换模型、Student‑t EGARCH(1,1)、Johansen VECM、Fisher z 相关性调整、FEVD 预测误差方差分解。

**📊 数据集**

数据集为2019‑2025年1月1日到8月31日共1559个交易日的科伦坡股票交易所（ASPI + 13个行业指数）以及中央银行的月度宏观系列（消费者物价指数、美元/卢比汇率、利率）。

**📈 对比分析**

通过与传统的政策日历切分和单一波动性区间做对比，发现自动切分更能揭示新冠与主权违约时期的显著负面冲击异质性；跨行业相关性在默认期并未出现显著提升，宏观变量在30天预测误差中占比不足4%。

**⚠️ 局限性**

局限包括：宏观变量只以每月频率前向填充至日频；波动性区间先行确定后再估计，未实现联合估计；行业样本量小且需剔除流动性差的行业，导致统计功效受限。

---

## 398. Balanced Boolean Functions from Sparse Niho-Type Permutation Polynomials: Few-Valued Walsh Spectra and Applications

**arXiv ID:** 2611.27371 | [PDF](https://arxiv.org/pdf/2611.27371v1)

**作者:** Chenxu Wang `[一作]` (Capital Normal University), Jun Zhang `[通讯]` (Capital Normal University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `847a60d8-a755-47af-ba5d-c5236b9e3083` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构造稀疏的Niho型置换多项式P(x)，在两类两对一映射P(x+x^2)和P(x+x^{-1})上参数化得到一族平衡布尔函数，并给出了它们的完全Walsh谱、非线性、代数度等密码学指标。

**💡 创新点**

创新点主要包括：① 提出新的P_λ(x)族置换多项式，并证明其可行性与可计数性；② 证明所有参数λ对应的布尔函数Walsh谱完全相同，且为五值谱；③ 基于该谱构造了最小二进制五权码、五交叉数点集及六特征的正则Klein图；④ 对P(x+x^{-1})映射得到一族极其简单的轨迹表示布尔函数。

**🔧 技术方法**

使用的技术包括：两对一映射与置换多项式的组合理论；Niho指数与AGW判据；单位圆U_m的极坐标分解与根计数；Walsh变换与矩阵/多项式的代数展开；以及对三次/四次多项式根的判别和计数。

**📊 数据集**

本文没有使用公开数据集，而是基于有限域理论和符号计算得到的符号结果。

**📈 对比分析**

通过与已知的半棱函数、平衡布尔函数以及最小码等的参数对比，证明所得函数在非线性和代数度上与现有最优例子相近或更优；此外，构造的五权码与已知最小码的权分布一致，且产生的图谱在谱分布上与现有正则Klein图相匹配。

**⚠️ 局限性**

局限性：① 仅处理了两对一映射P(x+x^2)与P(x+x^{-1})，未覆盖更一般的稀疏映射；② 对于P_λ的Walsh谱独立性尚未证明能否推广到更广泛的Niho型置换多项式；③ 仅给出代数度与非线性的上界，并未完全解析所有参数下的代数免疫；④ 代码与图结构的对称性与自动群尚未完全研究。

---

## 399. EgoLAM: Learning Latent Actions from Hand and Head Motion

**arXiv ID:** 2611.27290 | [PDF](https://arxiv.org/pdf/2611.27290v1)

**作者:** Ziyi Ding `[一作]` (Tsinghua University), Jiayu Chen `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并训练了 EgoLAM 模型，通过在两帧 RGB 之间提取手部动作的潜在表示，实现了从 egocentric 视频中无监督学习机器人的抓取与推挽动作。

**💡 创新点**

将手腕运动监督与头部运动分离，使用手部查询池化并在一个共同参考框架下编码手腕位移与旋转，从而让潜在动作编码更好地保留手部动力学并抵消摄像头视角变化。

**🔧 技术方法**

采用 DINOv2 视觉前端 + Slot Attention + 逆向动力学模型 (IDM) + 正向动力学模型 (FDM)，结合梯度反转对抗训练、视角一致性约束和手部热图监督来训练潜在动作编码。

**📊 数据集**

使用 EgoVerse 中的 egocentric 视频与手部/摄像头标注（Mecka、Eva 等），共计约 1,056 小时，另外 Slot Encoder 预训练使用约 1,120 小时的数据。

**📈 对比分析**

在 LARYBench、CALVIN、VLABench、RoboCOIN、AgiBot 等任务上与 LAOM、CLAM、LAPA、Moto、GeoLAM 等八个基线对比，EgoLAM 取得最高 67.85% 分类准确率、最低 0.391 MSE，并在 LIBERO 套件的三种任务上平均成功率 79.38%，比直接行为克隆高出 28.73pp。

**⚠️ 局限性**

仍受限于仅使用两帧输入、对头部运动分离的假设、对真实动态摄像头运动的建模不够精细，以及对非 egocentric 视角和跨域适应的鲁棒性待进一步验证。

---

## 400. GeoCloak: Defending Against Geolocation Reasoning in Multimodal Language Models via Perceptual Image Editing

**arXiv ID:** 2611.26987 | [PDF](https://arxiv.org/pdf/2611.26987v1)

**作者:** Haoyang Xu `[一作]` (University of Hawai`i at Mānoa), Liuwan Zhu `[通讯]` (University of Hawai`i at Mānoa)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 GeoCloak，利用多模态语言模型识别并编辑局部地理线索，以降低共享图片被推断的地理位置信息。

**💡 创新点**

创新点在于将地理线索分层识别、局部语义编辑（移除、替换、添加）与代理模型验证相结合，既能显著降低定位精度，又保持图像视觉自然度。

**🔧 技术方法**

使用可访问的多模态代理模型（如 OpenAI o3）进行线索检测与编辑指令生成，GroundedSAM 定位编辑区域，FLUX.1 Fill-dev（或 Gemini 2.5 Flash Image）实现局部图像编辑，代理模型对编辑结果进行验证并选取最具扰动的候选。

**📊 数据集**

在 Doxing（172 条用户生成图片）和 FairLocator（600 条全球街景）两大数据集上进行评估。

**📈 对比分析**

对比无防护、加高斯噪声、M-Attack-U 等基线，GeoCloak 在国家/州/城市/街道四级定位精度显著下降（如城市级从 28.6% 降至 9.4%），同时在 NIQE、BRISQUE 等无参考质量指标上优于像素扰动方法。

**⚠️ 局限性**

局限性包括对编辑器和代理模型的依赖、对非视觉泄露途径无效、编辑效果受限于线索检测准确性以及跨模型迁移性仍需进一步验证。

---

## 401. Balanced Divide and Conquer for Rooted Maximum Agreement Forests

**arXiv ID:** 2611.27160 | [PDF](https://arxiv.org/pdf/2611.27160v1)

**作者:** Feng Shi `[一作]` (Central South University), Jianxin Wang `[通讯]` (Central South University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df`

**🎯 论文内容**

本文提出了一种平衡分治框架，用于求解根化二叉系统发育树对上的最大一致森林（Maximum Agreement Forest, MAF）问题，并给出了基于该框架的固定参数化算法。

**💡 创新点**

创新点在于：
- 通过“平衡中心”技术把原问题分解为三个子问题，并证明每个子问题的预算上限可被显著降低；
- 设计了关键边（critical edge）集合与其排序，保证在分解时能保持目标森林的结构；
- 引入临时标记叶子（marker leaves）和两阶段二分分离（binary separation），在不失去全局最优解的前提下实现子问题的独立求解；
- 通过递归应用框架，将原始解算器 O*(2.3391^k) 的指数基数从 2.3391 降至 1.7743，进一步可逼近 φ ≈ 1.618.

**🔧 技术方法**

主要技术包括：
- 基于分解-合并（divide‑and‑conquer）与关键边定位；
- Reduction Rule 0 与 Greedy 约简；
- 两阶段二分分离与 marker leaf 记录；
- 递归预算计算与复原规则；
- 对现有 Split‑or‑Decompose 方案的理论改进。

**📊 数据集**

本工作为理论算法研究，未使用实际数据集；所有结果均通过数学证明与复杂度分析得到。

**📈 对比分析**

与之前的 FPT 方案相比：
- 传统方案基数为 4, 2.42, 2.344, 2.3391 等；
- 本框架将基数从 2.3391 降至 1.9455（单次应用）并进一步到 1.7743（两次递归应用）。
- 通过多次递归，可在任意 ε>0 下逼近 φ 的基数（即 O*((φ+ε)^k)）。

**⚠️ 局限性**

局限性包括：
- 仍是理论性高复杂度算法，实际运行时间与常数因子未知；
- 仅适用于根化二叉树（对多叉或无根情况仍需改造）；
- 分解时需要枚举所有内部节点，时间与树深度有关；
- 需要额外的 marker 处理，实现复杂度高。

---

## 402. No Audio, No Transcripts, No Problem: LLM-based ASR Adaptation Using Only Domain Documents

**arXiv ID:** 2611.27434 | [PDF](https://arxiv.org/pdf/2611.27434v1)

**作者:** Sergio Burdisso `[一作]` (Idiap Research Institute), Andreas Stolcke `[通讯]` (Uniphore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究在没有目标音频或转录的情况下，仅使用目标领域的原始文档，利用LLM生成对话式文本，再通过多视角去噪的文本适配方法，对LLM‑based ASR 进行域适配。

**💡 创新点**

首次提出文档仅适配框架，并比较将文档直接作为写文本与将其渲染成对话式文本两种方式的效果，证明后者能够接近甚至超过使用真实转录的适配性能。

**🔧 技术方法**

使用LLM‑based ASR（WavLM + Llama‑3.2‑3B‑Instruct）与多视角去噪适配器；利用大语言模型进行话题标注、实体抽取、Markdown 文档生成以及零样本对话生成。

**📊 数据集**

在两个公开和私有语料上评估：DefinedAI（客服电话对话）和 SlideSpeech（会议演讲/讲座），并构建对应的文档‑对齐基准。

**📈 对比分析**

与基线模型、真实转录适配、写文本适配以及对话文本适配四种条件进行对比，词错误率（WER）在不同域下提升 30%–50%，对话文本适配在多数情况下与真实转录相当或更优。

**⚠️ 局限性**

受限于 LLM 生成的质量与提示设计，生成对话所需的文本量在某些域（如 SlideSpeech）有限；跨域适配虽效果显著，但在更大规模、多语言或更复杂语境下仍需进一步验证。

---

## 403. Fair Property Tax: Reducing Regressivity in Machine-Learning Valuation

**arXiv ID:** 2611.27009 | [PDF](https://arxiv.org/pdf/2611.27009v1)

**作者:** Nicolas Acevedo Villena `[一作]` (Massachusetts Institute of Technology), Timothy Sparer `[通讯]` (Cook County Assessor's Office)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对Cook County评估办公室的住宅估价工作流，提出并评估了两种基于LightGBM的训练目标（Direct协方差惩罚和Surrogate加权损失），以减少估价与销售价格之间的递减趋势（回归性），同时保持价格尺度的预测精度。

**💡 创新点**

创新点在于：①将回归性（valuation‑to‑sale 比值随价格递减的系统偏差）直接嵌入模型训练目标；②提出一种观测可分解的Surrogate惩罚，利用加权平方误差约束对低价和高价房产的估价误差施加更大权重；③在保持原有预测器和模型架构不变的情况下，实现公平性改进并可直接集成到现有评估软件。

**🔧 技术方法**

技术主要包括：LightGBM梯度提升树、定制的目标函数（Direct协方差惩罚和Surrogate加权损失）、交叉验证的惩罚强度搜索、后置对数误差加权、以及基于log价差的回归性指标（PRB、VEI、MKI）和非线性残差诊断。

**📊 数据集**

使用Cook County 2016–2024年单户和小型多户住宅的已验证销售数据（约38万笔训练，约4万笔测试），以及2025年一份前瞻性测试集；另外还在8个其他县使用ATTOM商业数据进行跨县验证。

**📈 对比分析**

比较方法：在七折展开式交叉验证、保留测试集和2025前瞻测试集上计算R²、MAE、COD、PRD、PRB、VEI、MKI等指标。结果显示：在ρ≈1时，Direct与Surrogate均将PRB降低约25–50%，VEI、MKI和PRD趋近中性，R²_P保持不变或略升；但较大惩罚会导致MAE和COD上升，Surrogate会使对数RMSE增加，Direct会使非线性残差增大。

**⚠️ 局限性**

局限性包括：①仅评估了估价对销售价格的回归性，未直接衡量相对于真实市场价值的公平性；②训练和评估基于Cook County的数据，跨县推广需要进一步验证；③交叉验证折叠重叠，估计误差可能被低估；④未对模型超参数进行重新调优，惩罚强度与模型容量相互作用可能未充分探索；⑤未对不同物业类型、地区或时间段进行细粒度分析；⑥后置缩放和训练目标的选择缺乏前瞻性验证，部署前仍需进一步风险评估。

---

## 404. TacHOI: Contact-Aware Object Reconstruction and Tracking for Egocentric Videos

**arXiv ID:** 2611.26764 | [PDF](https://arxiv.org/pdf/2611.26764v1)

**作者:** Jackson Codd `[一作]`, Xianyi Cheng `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了TacHOI框架，利用可计量的触觉与单目视频联合优化手部姿态、对象运动和尺度，实现无需训练、无需模板的物体重建与跟踪

**💡 创新点**

创新点在于将经校准手掌的触觉点作为物理尺度尺量，既限制对象相对于手的深度，又通过分散触点的距离提供对象尺度约束，消除了单目尺度-深度模糊

**🔧 技术方法**

采用多任务优化：图像轮廓对齐、稀疏跟踪、手部重投影、手部深度与范围惩罚、触觉对应距离约束、手掌卡尺尺度约束以及尺度底限约束，使用PyTorch3D渲染、Adam优化

**📊 数据集**

在ARCTIC（55段）、HO3D（45段）与OpenTouch（30段）三大数据集上进行评估，ARCTIC与HO3D为合成触觉，OpenTouch为实测触觉

**📈 对比分析**

与FoundationPose、BundleSDF、ForeHOI等基线对比；在ARCTIC上ADD‑S*由15.6 cm降至3.6 cm，中心点误差由16.4 cm降至5.6 cm；在HO3D上ADD‑S*达2.8 cm；同时在尺度与形状上取得较优或相近的分数，证明触觉提供有效的度量约束

**⚠️ 局限性**

局限在于：依赖手部可校准的触觉传感器；对对象形状的细化有限，主要通过SAM 3D的预置网格；在高运动、拥挤场景下触觉识别与手部跟踪仍易失误；无深度传感器时仍需高质量手部姿态估计

---

## 405. Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training

**arXiv ID:** 2611.27376 | [PDF](https://arxiv.org/pdf/2611.27376v1)

**作者:** Zhen Zhang `[一作]` (University of California Santa Barbara), Xin Eric Wang `[通讯]` (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

论文提出了一种名为Composable Soft-Prompt Distillation (CSPD)的方法，通过将每个RL专家独立蒸馏成软提示（soft prompt）并在推理时将这些提示串联，来实现对固定基模型的持续后训练和多专家融合。

**💡 创新点**

创新点在于：① 保持原始模型参数冻结，避免每次加入新专家时都要重新训练整个模型；② 仅用一个软提示即可蒸馏一个专家，且不同提示可以直接拼接而不需联合训练；③ 通过拼接提示实现多专家组合时，性能随专家增多而提升，且最终可与传统多教师蒸馏方法（MOPD）保持同等水平，同时在跨域泛化上表现更好。

**🔧 技术方法**

使用的技术主要包括：1) 软提示蒸馏（Soft-Prompt Distillation）；2) 软提示拼接与组合；3) 输入空间适配（input‑space adaptation）以降低推理开销；4) 与LoRA、MOPD等基线进行对比评估。

**📊 数据集**

实验使用的主要数据集包括：数学题解、代码生成和智能体（agent）环境中的RL专家数据；此外还在六个保留的跨域基准（held‑out benchmarks）上评估模型的泛化能力。

**📈 对比分析**

方法对比：与MOPD在多专家设置下的平均性能几乎持平（67.83 vs 67.79）。在添加专家后，CSPD的平均在域内表现持续提升，且在域外的平均得分最高。推理吞吐量在加入输入空间适配后仅低于基模型1.5%，且相较于rank‑4 LoRA提升2.40×。

**⚠️ 局限性**

局限性：① 对软提示的拼接方式在极大规模专家集合时可能导致长度过长，影响计算效率；② 当前方法主要验证在RL后训练场景，尚未探讨对非RL任务或更大模型的适用性；③ 软提示的质量依赖于教师专家的表现，若教师本身性能不足，蒸馏效果可能受限。

---

## 406. Inject, Encode, Navigate: Feature-Wise Time Injection for Rectified Flow with Bidirectional Feature Encoding

**arXiv ID:** 2611.27179 | [PDF](https://arxiv.org/pdf/2611.27179v1)

**作者:** Xiaogang Wang `[一作]` (Amazon Web Services), Yash Shah `[通讯]` (Amazon Web Services)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了一种基于Rectified Flow匹配、双向Mamba编码器和时间条件化的视觉导航框架，实现单步推理下的高效、准确导航；

**💡 创新点**

创新点包括：1) 将传统扩散的多步去噪改为一次性Rectified Flow；2) 引入双向Hydra（Mamba）作为线性复杂度的时序编码器，能捕获长序列依赖；3) 设计特征级时间条件化（FTC）实现流状态与视觉特征的高效耦合；

**🔧 技术方法**

使用了Rectified Flow Matching、双向Hydra（Mamba）State‑Space模型、Feature‑wise Time‑Aware Conditioning、Vision‑based diffusion policy (NoMaD)、ViT backbone、MLP、Euler积分、正弦时步嵌入等技术；

**📊 数据集**

实验数据集包括公开的RECON、TartanDrive以及自建的Deep Robotics X30 Pro四足机器人数据集；

**📈 对比分析**

与ViNT、NoMaD等基线在RECON、TartanDrive及内部数据集上进行离线和现场评估，Ours‑Full在WCS提升5.8%、ADE下降96.7%、FDE下降97.5%，导航成功率从86.7%提升至100%，推理时间从79.9 ms降至68.4 ms；

**⚠️ 局限性**

限制在于模型仍主要在静态环境和单目标场景下验证，未深入测试极端光照、动态障碍或多目标情形；FTC机制相对简单，可能在更复杂时间分布下需进一步改进；

---

## 407. Space-Efficient Text Indexing for Advanced Counting Queries

**arXiv ID:** 2611.27432 | [PDF](https://arxiv.org/pdf/2611.27432v1)

**作者:** Solon P. Pissis `[一作]` (Cyprus Institute), Sharma V. Thankachan `[通讯]` (North Carolina State University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种通用框架，用于在文本索引中支持满足两个条件的高级计数查询（ϕ_T(P)），并给出了两种空间-时间折衷方案。

**💡 创新点**

创新点在于把高级计数问题归约为可由排序得到整数值且满足单调性的函数，利用Elias–Fano编码和重路分解，突破了Θ(n log n)位空间下的(m)-时间极限，首次实现了 (n logσ + n(loglog n)^2) 位空间的 (m)-时间计数。

**🔧 技术方法**

核心技术包括：Elias–Fano编码、重路分解、稀疏表、RMQ、FM-index/压缩后缀数组，以及对“频繁”与“稀疏”子串答案的分层存储。

**📊 数据集**

本文为理论工作，未在真实数据集上实验，只在理论模型上给出空间/时间上界。

**📈 对比分析**

与先前的 (n log n)-bit (m)-time 索引或 (n logσ)-bit (mlog_σ^ε n)-time 索引相比，所提出的方案在同等空间下实现了更快或同等的查询速度，且在不超过 (n logσ) 位时仍保持 (mlog_σ^ε n)-time。

**⚠️ 局限性**

限制在于框架只适用于满足排序可计算且单调的计数函数；对非常稀疏或不满足单调性的计数问题仍无法直接应用；此外，实际实现中重路分解与EF编码的常数开销可能较大。

---

## 408. THRIFTY: Cost-Constrained Reasoning-Data Synthesis for Small Language Models

**arXiv ID:** 2611.27436 | [PDF](https://arxiv.org/pdf/2611.27436v1)

**作者:** Aasheesh Singh `[一作]` (Phi Labs, Quantiphi), Vishal Vaddina `[通讯]` (Phi Labs, Quantiphi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种名为 THRIFTY 的两阶段 bandit 控制器，利用固定的生成 token 预算高效地收集多样化的正确推理轨迹，用于训练小型语言模型。

**💡 创新点**

其创新点在于把数据生成视为 bandits-with-knapsacks 问题，引入了多样性门控的凹值函数、Thompson 采样与机会成本退役策略，并通过数据驱动的校准阶段自动设置生成长度、难度先验与多样性阈值。

**🔧 技术方法**

技术上使用 Beta‑Bernoulli bandit、Thompson sampling、基于 self‑BLEU 的多样性门控、log(1+v) 凹价值函数、Otsu 阈值校准及机会成本退役规则，教师模型采用 Qwen2.5‑32B。

**📊 数据集**

实验数据集包括数学题集 AIME、MATH‑500 以及编程题集 LiveCodeBench。

**📈 对比分析**

与 Random、Uniform、Difficulty‑aware Bandit、Entropy Bandit 等基线对比，使用 dv/1k、coverage@1、distinct trajectories 三项指标，THRIFTY 在大部分预算场景下每千 token 获得的多样性价值比统一分配高达 22%，比最佳自适应基线高 7–16%，且保持覆盖率不变。

**⚠️ 局限性**

局限性在于得到的难度后验仅是针对教师模型的成功概率，难以直接反映任务本身难度，且仅在有限的预算与模型规模上验证，缺乏对更大预算或多尺寸教师/学生的评估。

---

## 409. Acoustic Markers of Engagement in Children with Autism Spectrum Disorder and Their Caregivers

**arXiv ID:** 2611.26810 | [PDF](https://arxiv.org/pdf/2611.26810v1)

**作者:** Young Suh Hong `[一作]`, Maja Matarić `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

研究了在家庭环境中使用社交辅助机器人（SAR）进行干预时，儿童与照顾者的语音特征如何区分参与与不参与状态。

**💡 创新点**

首次系统性地利用线性混合效应模型对儿童与照顾者的声学特征进行分组比较，发现三种关键声学标记（HNR、AlphaRatio、SlopeV500-1500）在儿童中相较于照顾者表现出更大变化。

**🔧 技术方法**

使用 Pyannote 进行说话人分离，提取 eGeMAPS 声学特征，并采用线性混合效应模型（Separate Model 与 Interaction Model）以及 Benjamini‑Hochberg FDR 校正进行统计分析。

**📊 数据集**

基于七名 3–7 岁自闭症儿童及其家庭照顾者的一月家庭式 SAR 互动录音数据（约 3 小时/孩子），并配有二值参与度标注。

**📈 对比分析**

通过 LMM 计算各声学特征在参与与不参与状态下的标准化系数，并比较儿童与照顾者的系数差异；结果显示儿童的 HNR、AlphaRatio、SlopeV500-1500 的标准化变化幅度显著大于照顾者，表明这些特征在儿童中的辨别度更高。

**⚠️ 局限性**

存在声学特征提取中的重叠语音处理不完善（DER≈30%），导致部分特征缺失；样本量仅七名儿童，且仅针对特定 SAR 任务，缺乏泛化性与多任务验证。

---

## 410. Privacy Protection in Self-Evolving Personal Agents: Persistent Memory and Cross-Carrier Transfer

**arXiv ID:** 2611.26983 | [PDF](https://arxiv.org/pdf/2611.26983v1)

**作者:** Zibin Lin `[一作]` (Shenzhen University), Shengli Zhang `[通讯]` (Shenzhen University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对个人智能体中持续记忆、提示、技能等自适应更新过程中的隐私保护机制进行系统综述与跨载体对比，构建统一的干预与保障框架；

**💡 创新点**

提出了四维框架（载体、保护单元、可观测接口、时间维度）来映射和评估不同隐私机制，首次在同一研究中将记忆写入、维护、读取、撤回与技能迁移等多种环节结合，揭示跨载体隐私传播的细节与缺口；

**🔧 技术方法**

利用差分隐私（DP-OPT、DP-ES、DP-MemView等）、联邦学习与多方安全计算、可信执行环境与ORAM、隐私门控与撤回机制、机器可忘记（SBU、Reglos、TAPE等）等多种技术；

**📊 数据集**

实验使用合成记忆历史、公开电影推荐数据集MovieLens 100K、公开LLM模型（如Qwen2.5-7B-Instruct）以及部分真实用户交互记录（如HomeLLaMA调查）来评估隐私-效用权衡；

**📈 对比分析**

通过对比实验（如DP-MemView的AUC与效用、Label‑DP与Tuple‑DP在偏好学习中的误差、MaskClaw的过滤率等）展示了不同机制在隐私保护与系统性能上的折衷，证明了一些机制在保持较低效用损失的同时能提供可测的隐私保证；

**⚠️ 局限性**

局限性包括：仅覆盖截至2026年11月10日公开文献，部分载体与接口缺乏直接研究；实验多为合成或小规模真实数据，未验证在大规模真实部署中的效果；跨载体对比多为理论/实验推断，缺乏统一的统一评测基准；对撤回和衍生状态的完整证明仍未实现。

---

## 411. Contention-Aware Flexible Execution for Machine Learning Workloads

**arXiv ID:** 2611.26840 | [PDF](https://arxiv.org/pdf/2611.26840v1)

**作者:** Marco Kurzynski `[一作]` (University of Central Florida), Di Wu `[通讯]` (University of Central Florida)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究通过对AMD GPU上并发内核执行的系统性特征化，提出了基于计算波量与内存占用的重叠调度策略，以减轻并发导致的性能衰减；

**💡 创新点**

创新点在于首次利用内部毫秒级采样器对并发执行的内存膨胀进行量化分析，并据此给出可直接应用的简易调度启发式，且无需对内核进行重写；

**🔧 技术方法**

技术手段包括AMD内部毫秒级采样器、计算单元掩码、波量量化模型、事件驱动模拟器以及基于CUDA流事件的调度实现；

**📊 数据集**

实验基于真实大语言模型推理与训练任务，主要使用Qwen3‑32B、Llama‑70B等模型在AMD GPU上进行；

**📈 对比分析**

与传统早期优先（early‑first）调度比较，所提出的灵活重叠调度在推理端可提升高达13%端到端吞吐，训练端提升约5%，整体几何平均加速分别为4.5%和2.2%；

**⚠️ 局限性**

局限性包括仅能控制计算单元分配，无法细粒度调节缓存/内存带宽；调度策略在更大规模多GPU或不同架构上的适用性尚未验证；

---

## 412. Detecting Word-Substitution Poisoning of a RAG Knowledge Base with Relative Entropy and Runtime Verification

**arXiv ID:** 2611.27410 | [PDF](https://arxiv.org/pdf/2611.27410v1)

**作者:** Ali Yousefi `[一作]` (University of London), Hafizul Asad `[通讯]` (University of London)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对网络安全检索增强生成（RAG）系统的知识库进行词替换毒化攻击，并研究其对答案质量的影响。

**💡 创新点**

提出一种基于相对熵的“同义词缺口”检测方法，可在检索阶段识别被替换的词，并设计了运行时监视器对答案进行可信度验证，从而在不丢失证据的前提下恢复答案正确性。

**🔧 技术方法**

使用相对熵（KL）和同义词缺口、上下文同义词缺口、基于小型语言模型的熵、检索过滤器以及基于MITRE STIX数据的运行时监控和回退机制。

**📊 数据集**

以MITRE ATT&CK和CAPEC的18篇文档（约200段）为知识库，构建44道手工评审的安全问答基准。

**📈 对比分析**

实验表明，词替换毒化将答案正确率从0.714降至0.555，检测方法在PWWS上AUROC可达0.95，检索过滤能将被攻击文本曝光率降至约50%，而运行时监控在恢复模式下将正确率提升至0.786、拒绝率大幅下降。

**⚠️ 局限性**

研究局限包括：知识库规模仅18篇文档、问题数量有限、使用单一LLM和向量检索模型、评估依赖LLM判断、攻击者仅针对代理分类器、未考虑更强的自适应攻击或大规模知识库。

---

## 413. Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models

**arXiv ID:** 2611.26841 | [PDF](https://arxiv.org/pdf/2611.26841v1)

**作者:** Konstantinos Skianis `[一作]` (University of Ioannina), Aristidis Likas `[通讯]` (University of Ioannina)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过前沿大语言模型在Python环境中执行权重空间编程，利用少量无梯度的前向传递编辑弱模型的权重，使其在未提示下完成词映射等任务。

**💡 创新点**

创新点在于首次系统评估权重空间编程与传统提示学习、梯度微调的对比，发现编辑器能以极小参数修改（仅千级到万级）重现近似in‑context学习的任务向量，并能自行生成标签实现无监督编辑。

**🔧 技术方法**

技术方法包括使用Claude Fable 5.1、Claude Opus 5.5、Gemini 4 Argon、GPT‑6 Astra等前沿模型作为编辑器，通过Python工具调用PyTorch模块读写权重；对编辑内容进行手工与自动化的策略注释；与LoRA、Rank‑1更新、Adam微调、上下文蒸馏等梯度基线以及脚本化任务向量注入进行比较。

**📊 数据集**

数据集主要为六个词映射任务（反义词、国家→首都、时态变换、英法翻译、数值倍化、加法）和少量通用语言建模与推理任务（WikiText‑2、ARC‑Easy），每个任务提供16个带标签示例；在无标签条件下，编辑器自行生成训练对。

**📈 对比分析**

相对比较显示，权重空间编程在多数任务上达到与16‑shot ICL相当或略低的零样本准确率（大约80‑95%），并且在参数改动量上比全模型微调低数千倍；编辑器写入的任务向量与ICL产生的残差相似度高，输出与ICL相符率超过80%，但在不同提示词的泛化性较差，部分编辑导致困惑率升高。

**⚠️ 局限性**

局限性包括：实验只覆盖Qwen2.5系列和有限的其他模型；编辑器对任务知识的依赖（自行生成示例）不易泛化；编辑器对不同提示词的泛化差；编辑过程中可能出现的模型破坏和高成本；以及实验设置对编辑器、数据拆分和超参数的敏感性。

---

## 414. InfantConnect: integrating developmental brain atlases and electrophysiological network analysis

**arXiv ID:** 2611.26911 | [PDF](https://arxiv.org/pdf/2611.26911v1)

**作者:** Alessandro Crimi `[一作]` (AGH University of Krakow), Paweł Gryboś `[通讯]` (AGH University of Krakow)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `109c2b71-d051-425c-831f-0c544c24280d` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

开发了一个名为 InfantConnect 的开源 Python 框架，将年龄相关的皮层表面、婴儿头骨/皮肤模板、电极位置、AAL 划分、纤维图谱、EEG 预处理、LORETA 可视化以及 Granger 因果与转移熵的有效连接分析集成在同一 3D 工作空间中，并提供桌面 GUI、命令行工具和 AI 代理可调用的 MCP 接口。

**💡 创新点**

创新点在于：① 将多种分散的解剖学和 EEG 分析资源统一到一个可视化与批处理兼容的框架；② 引入 MCP 层，支持大型语言模型或自主研究代理直接调用 EEG 处理工具；③ 对于婴儿特有的年龄化表面和 AAL 划分进行投影与覆盖度评估；④ 在合成网络和真实新生儿 EEG 数据上验证有效连接工作流，展示了 Granger 与转移熵在线性与非线性系统中的表现差异。

**🔧 技术方法**

技术上使用 Python（PySide6、PyVista、VTK、MNE-Python）、自定义 Granger/转移熵实现、LORETA 逆解、以及 MCP 服务器；利用命令行参数化的预处理流程（50 Hz 除噪、0.5–40 Hz 带通、平均参考）和脚本化导出连接矩阵；使用 3‑month 步长的年龄表面、三种 AAL 体积投影以及 NABA800FC 纤维图谱。

**📊 数据集**

数据集：1）Helsinki 新生儿 EEG 数据集（79 名足月新生儿，19 通道，采样 256 Hz）；2）合成 3‑节点自回归系统（线性与非线性两种），N=5000；3）公开的 4D UNC 新生儿/婴儿皮层表面、AAL 划分、头骨模板与 NABA800FC 纤维图谱。

**📈 对比分析**

比较方法：Granger 因果（自研实现）与 Statsmodels 实现、转移熵；在合成网络中测量 AUC、平均精度、F1 等指标；在 Heli­ness 数据上仅展示工作流。性能上，Granger 因果自研实现平均 22.48 s，Statsmodels 为 45.56 s；EEG 质量评估 1.47 s，预处理 1.68 s，LORETA 5.58 s，转移熵 1.99 s。

**⚠️ 局限性**

局限性包括：① 解剖资源来源第三方，版本和完整性需自行验证；② AAL 划分仅在 0、12、24 个月三个时间点，投影到中间年龄表面仅为近似；③ 只评估了统计依赖的有效连接，未验证因果推断或真实生理连通性；④ LORETA 可视化仅为示例，未进行来源定位验证；⑤ 仅在有限的合成网络和单一真实数据集上测试，缺乏更广泛的临床验证。

---

## 415. Learned Network Dynamics for Model Predictive Traffic Signal Control under Incidents

**arXiv ID:** 2611.27435 | [PDF](https://arxiv.org/pdf/2611.27435v1)

**作者:** Dang Viet Anh Nguyen `[一作]` (Technical University of Denmark), Filipe Rodrigues `[通讯]` (Technical University of Denmark)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

针对城市交通信号的模型预测控制（MPC）中因测量覆盖不足和非周期性事故导致的排队不确定性，提出了基于图神经网络（GNN）的两层动作条件网络动力学模型，并与迭代最优响应（iterated best response）相结合，形成 GNN‑MPC 控制器。

**💡 创新点**

创新点：
1) 设计两层 GNN 结构，网络层捕捉交叉口间相互作用，车道层预测各车道排队、上游停滞和等待时间；
2) 通过动作条件（即候选信号相位）直接学习排队演化，避免手工建模并能自动适应事故变化；
3) 使用概率束搜索与接受测试的迭代最优响应，实现跨交叉口的协同规划并保证预测成本单调下降；
4) 在实验中验证上游测量覆盖与学习预测模型对闭环性能的互补作用。

**🔧 技术方法**

技术：
- 图神经网络（GATv2）+ LSTM 编码器，结合动作嵌入；
- 经验回放训练，采用 Huber 损失/平方误差；
- 概率束搜索（beam search）与多次迭代的最优响应；
- 评估使用 SUMO 仿真平台和 T‑REX 调度器，控制周期 10 s，预测周期 20–60 s。

**📊 数据集**

数据集：
- 通过 MaxPressure 控制器在两网络（4×4 网格和 21 节点 Ingolstadt）上产生 100–296 条模拟情景，含随机事故；
- 训练集、验证集、测试集按情景划分；
- 评估使用 5 次随机种子 + 5 次需求/事故组合，共 25 条运行；
- 额外进行 9 条 Ingolstadt 和 15 条 Grid 的 ablation 与模型替换实验。

**📈 对比分析**

比较方法：
- 与固定时间、MaxPressure、五种强化学习方法（IDQN、IPPO、MPLight、FMA2C、CoSLight）以及 MaxPressure‑U（含上游测量）和 SF‑MPC（解析模型）进行对比；
- 指标包括旅行者平均旅行时间（Traveler ATT）、吞吐量（已完成行程数）、未插入车辆数；
- 统计使用配对差值与 95% 学生‑t 区间，聚类分析显著性。
- 结果：GNN‑MPC 在两网络均获得最低 Traveler ATT（Grid：最多 22.4 s；Ingolstadt：最多 76.7 s），吞吐量最高，未插入车辆最少；相较 MaxPressure，Travel ATT 提升 13–31%（Grid）/ 31–34%（Ingolstadt）。

**⚠️ 局限性**

局限性：
- 仅在两网络、单一需求谱和已知事故信息（即时可用）下评估；
- 训练数据仅来源于 MaxPressure，缺乏多控制器经验；
- 对比基线未获得事故信息，导致不公平比较；
- 计算时间接近控制周期，未在闭环中考虑推算延迟；
- 只给出单个模型实例，未验证训练不确定性；
- 未讨论模型在未知网络、需求变异或长周期事故下的泛化能力。

---

## 416. Beyond Future Views: Exploring Future-State World Models for Vision-Language Navigation

**arXiv ID:** 2611.27418 | [PDF](https://arxiv.org/pdf/2611.27418v1)

**作者:** Kangyi Wu `[一作]` (Xi'an Jiaotong University), Jianyi Liu `[通讯]` (Xi'an Jiaotong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉-语言导航任务中，作者提出FS-Nav框架，利用短期路标和中期走廊预测未来导航状态，并通过顶视图与第一人称覆盖两种视觉接口传递给VLM导航器。

**💡 创新点**

核心创新在于将未来导航视为目标导向的空间状态和路径结构，而非传统的像素或潜在视图预测，并将其拆分为短期路标和中期走廊两种互补抽象。

**🔧 技术方法**

技术包括基于Qwen2.5-VL的世界模型、空间与语言特征融合、热力图与路径带预测、顶视图/第一人称渲染接口以及基于文本动作生成的VLM导航器。

**📊 数据集**

使用了R2R-CE和RxR-CE两个标准VLN无视角验证集，并在训练阶段融合R2R、RxR以及ScaleVLN轨迹。

**📈 对比分析**

在R2R-CE和RxR-CE验证集上，FS-Nav在不使用全景观察的情况下，分别取得60.7%/55.1%成功率和55.9%/46.5% SPL，优于多数非全景方法，且在nDTW和NE方面表现更优。

**⚠️ 局限性**

主要局限在于未来状态预测质量仍受限，尤其是走廊和路标的准确性；同时依赖VLM的推理速度相对较慢，且对预测误差敏感。

---

## 417. Cleave: Severing Guardrails and Auditing Spectral Coupling in Open-Weight LLMs

**arXiv ID:** 2611.27426 | [PDF](https://arxiv.org/pdf/2611.27426v1)

**作者:** Feng Huang `[一作]` (Beijing University of Posts and Telecommunications), Yunqi Zhang `[通讯]` (Zhongguancun Laboratory)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并实现了一种训练无关的写入侧子空间投影攻击，能够去除或选择性删除开放权重LLM的安全防护，同时保持模型的通用推理与编码能力。

**💡 创新点**

创新点在于揭示安全拒绝在稀疏MoE中实际上由残差写入权重的低秩子空间实现，而非路由门；基于此提出正交子空间投影攻击与隐蔽的 Cyber‑Stealth 模式，突破了传统路由门或权重微调无法完全去除安全限制的局限。

**🔧 技术方法**

主要技术包括正交投影与SVD分解来提取并剔除拒绝子空间，跨专家与跨写入器的谱耦合审计，低秩子空间选择门，和基于CPU的跨组件谱比度检测；整个攻击过程完全不需要梯度或再训练。

**📊 数据集**

使用公开的 7B–320B 大规模开放权重模型（包括 dense 与四种稀疏 MoE 变体），安全基准（HarmBench、AdvBench、SorryBench 等）以及 356 项网络渗透任务与 15 个真实漏洞沙箱进行评估。

**📈 对比分析**

与现有白盒基线相比，该攻击在七个安全基准上实现最高 67.97% 宏观攻击成功率（提升 55.82pp），在渗透任务上可执行漏洞生成率提升多达 24×，并能捕获高达 66.7% 的目标服务；同时保持了模型的 MMLU、GSM8K 等通用能力。

**⚠️ 局限性**

局限性包括：需要访问基线模型权重以计算差分；对结构裁剪、重排或持续微调后模型的效果未知；以及攻击对极高噪声或多轮微调的鲁棒性尚未充分评估。

---

## 418. ADEPT: Adaptive Monocular Depth Estimation via Model-In-The-Loop Photometric Tuning with Capture Risk Prediction

**arXiv ID:** 2611.27423 | [PDF](https://arxiv.org/pdf/2611.27423v1)

**作者:** GiHyun Kim `[一作]` (Seoul National University), Hyung-Sin Kim `[通讯]` (Seoul National University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计了一种基于参考相对误差（RRD）的摄像机曝光与增益自适应控制框架，用于提升单目深度估计的鲁棒性。

**💡 创新点**

创新点在于：①引入RRD作为无标注在线反馈，直接反映冻结深度模型对不同摄像设置的感知质量；②结合上下文高斯过程（GP）bandit控制，实现基于历史评分的实时曝光/增益选择；③在移动机器人上实现高效部署。

**🔧 技术方法**

使用了轻量级FiLM条件网络进行RRD预测、上下文高斯过程带宽控制、Depth Anything v2模型以及特征提取。

**📊 数据集**

使用自采集的51K帧RGB‑D序列，包含三种训练场景和三种验证场景，亮度分为明、正常、暗三档，机器人在不同速度和旋转状态下循环。

**📈 对比分析**

与内置自动曝光、Shin et al.、Zhang et al. 的控制策略进行比较；在9种场景光照组合下，平均AbsRel从0.07096提升至0.06062，下降约14.6%，相对误差和δ1.25均优于基线。

**⚠️ 局限性**

局限性包括：需要离线获取精确深度标签来构造RRD监督；控制策略在极端光照或高速运动时仍可能产生误差；模型推理占用大部分时间，限制了最高帧率。

---

## 419. Pose-based Co-speech Gesture Detection

**arXiv ID:** 2611.27415 | [PDF](https://arxiv.org/pdf/2611.27415v1)

**作者:** Frances Yung `[一作]` (Saarland University), Ninlawat Phuangschoke `[通讯]` (Asian Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建统一评测框架，对三大手势数据集（MULTISIMO、GESRes、CABB‑S）上的规则、预训练与训练模型进行同一交叉验证下的对比，并提出了CTR‑GCN+BiLSTM模型。

**💡 创新点**

创新点在于：①统一同一数据拆分与评测粒度（帧级、span级、窗口级），揭示现有方法性能差异；②基于多尺度时序GCN与BiLSTM的改进多相位序列标注网络，显著提升CABB‑S的检测效果。

**🔧 技术方法**

使用了OpenPose/MediaPipe/MMPose关键点、LightGBM、CTR‑GCN（Channel‑wise Topology Refinement Graph Convolutional Network）、MS‑TCN、Transformer编码器、BiLSTM解码器、交叉熵+时序一致性损失等技术。

**📊 数据集**

使用的公开数据集包括MULTISIMO（英语协作对话）、GESRes（政治家/临床/教育者的对话）和CABB‑S（荷兰语任务导向交互）。

**📈 对比分析**

在相同的交叉验证折叠下，评估帧级、span级和窗口级指标；结果显示最优LightGBM在MULTISIMO的帧级F1≈0.50，CABB‑S的CTR‑GCN+BiLSTM在窗口级F1≈0.666，整体仍未超过0.7，显示任务尚未解决。

**⚠️ 局限性**

主要局限包括：检测性能受手势频率与注释一致性影响大；跨域泛化差；仅使用姿态信息，缺乏语音上下文导致误检；误检率高，难以直接替代人工标注。

---

## 420. Ego2Ego: Generative Visual Perspective Taking Across Collaborators

**arXiv ID:** 2611.27459 | [PDF](https://arxiv.org/pdf/2611.27459v1)

**作者:** Chuhan Jiao `[一作]` (University of Stuttgart), Andreas Bulling `[通讯]` (University of Stuttgart)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出从单人视角、场景点云和协作者相机轨迹生成协作者的第一人称视频。

**💡 创新点**

结合稠密几何渲染、相对几何注意力和视觉记忆，实现跨视角视频生成的全新框架。

**🔧 技术方法**

采用预训练视频扩散模型、稠密几何渲染、相对几何注意力以及视觉记忆模块。

**📊 数据集**

使用CoMind数据集进行训练和评估。

**📈 对比分析**

与新视角合成、相机控制生成和exo-to-ego三类基线对比，在PSNR、FVD、布局mIoU等指标均显著提升。

**⚠️ 局限性**

缺乏对场景变化、透明表面（如玻璃）以及ego自身出现时的视觉信息处理能力。

---

## 421. The Assistant Persona Differs Across Languages in Activation Space

**arXiv ID:** 2611.27504 | [PDF](https://arxiv.org/pdf/2611.27504v1)

**作者:** Daniel Yoo `[一作]` (Carnegie Mellon University), Evan Harris `[通讯]` (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多语言大模型中助手向量的差异，分析其与语言资源水平的关系并评估对安全干预的影响。

**💡 创新点**

首次系统性提取31种语言的 Assistant Axis，揭示其方向与长度随语言资源变化并对安全性产生显著影响。

**🔧 技术方法**

采用激活空间方向提取、余弦相似度、Jacobian lens、Steering、Activation Capping 等技术。

**📊 数据集**

使用 Qwen3-32B 模型及31种语言的 Common Crawl 份额、翻译系统提示与 MultiJail jailbreak 数据。

**📈 对比分析**

通过与英语向量的余弦相似度、轴长度、默认状态投影等指标进行对比，并在九种语言上测试 Steering，显示自语言轴干预更能抑制或增强有害协助率，表现优于英语轴。

**⚠️ 局限性**

存在翻译偏差、LLM 评判不一致、仅以单模型为主、Common Crawl 仅为资源占比代理，缺乏因果验证。

---

## 422. Alignment without Transfer: Tokenization Impacts Cross-lingual Representations during Multilingual Language Model Pretraining

**arXiv ID:** 2611.27491 | [PDF](https://arxiv.org/pdf/2611.27491v1)

**作者:** Jonathan von Rad `[一作]` (University College London), Yao Lu `[通讯]` (University College London)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在五种语言上预训练39个1B参数的单语/双语解码器模型，系统研究了词表分配对多语语言模型的影响，探究了词表分配如何影响跨语言表示对齐、参数知识共享以及跨语言迁移。

**💡 创新点**

创新点在于：①用可比的词表分配（从419语言到仅5语言）对同一模型架构进行对比，揭示更公平的词表分配显著提升早中层跨语言表示对齐和参数知识共享；②发现这种提升并不带来跨语言迁移的提升，提出“对齐‑迁移解耦”现象；③首次在解码器预训练阶段使用MEXA、PolyFact以及bits‑per‑byte评价跨语言对齐、知识共享与迁移。

**🔧 技术方法**

技术包括：基于SentencePiece Unigram的词表训练、LLaMA风格的decoder-only Transformer、RMSNorm+SwiGLU、rotary position embeddings、AdamW bf16训练，使用logit lens、integrated gradients进行神经元归因与因果 ablation、线性回归控制模型质量。

**📊 数据集**

数据集主要是FineWeb2‑HQ（非英语语言）和FineWeb‑HQ（英语），以及用于评估的FLORES+（并行句子）和PolyFact（事实问答）。

**📈 对比分析**

比较方法：对齐通过MEXA度量（句子检索精度），迁移通过目标语言的验证损失比率（BTS^loss）衡量，知识共享通过CTR（因果 ablation）和神经元重叠度量。实验表明：更公平的词表分配可提升早中层对齐和CTR（最高可达0.56），但对迁移影响不显著，跨脚本对齐与迁移仍存在显著差距。

**⚠️ 局限性**

局限性包括：仅在单语/双语（英语为锚）设置下实验，未探索非英语双语或全多语场景；只在1B参数、30B token训练规模和单一随机种子下验证，未检验更大规模或多种子的普适性；脚本与语言本身混杂，仅能做脚本效应描述。

---

## 423. AskWise: Execution-Guided Clarification of Intent and Data in Text-to-SQL

**arXiv ID:** 2611.27474 | [PDF](https://arxiv.org/pdf/2611.27474v1)

**作者:** Yuru Liang `[一作]` (Renmin University of China), Miles Fan `[通讯]` (Renmin University of China)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在数据库查询过程中，只在必要时向用户提问以消除歧义的机制，并通过候选查询执行和部分执行结果来确定是否需要提问。

**💡 创新点**

创新点在于只在两个可能的解释在该选择上不同且给出不同答案时才提问；同时给出了在布尔答案上的 EC2 gain 差异的精确上界，并证明该方法在实验中能显著减少查询执行次数。

**🔧 技术方法**

使用候选查询执行、部分执行结果分析、EC2 价值评估、布尔函数的均匀二元立方分析；在实验中使用 AmbiQT 数据集、模拟销售任务与 GPT/Doubao Lite 生成的候选查询。

**📊 数据集**

AmbiQT 数据集、120 个销售任务（3 种查询类型）以及 155 份采样问题。

**📈 对比分析**

对比方法：与分区松弛方法和首次出现差异时提问的基线；性能：平均执行次数从 14.88 降至 13.18，额外提问次数仅 0.025，参考答案召回率提升 8.71–10%。

**⚠️ 局限性**

局限性：假设候选查询包含正确查询和真实数据库状态；仅用一次值交换进行 AmbiQT 审计；无法估计修复频率；使用新种子生成的数据；模拟器给出真值并统一成本；需要用户研究评估真实提问效果；假设布尔答案、均匀二元立方、无修复依赖或概率不均等会影响收益计算。

---

## 424. Silent Arbitration: How Large Language Models Resolve Co-Authoritative Conflicting Instructions

**arXiv ID:** 2611.27444 | [PDF](https://arxiv.org/pdf/2611.27444v1)

**作者:** Sihyeon Lee `[一作]` (Soongsil University), Chanjun Park `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在等权重双人设冲突下的冲突识别、解决策略和立场倾向，构建共权威人设冲突框架并进行提示实验。

**💡 创新点**

提出共权威人设冲突概念，系统量化模型的冲突披露、仲裁形式和方向偏好，揭示沉默仲裁现象，并比较不同提示框架对仲裁可见性的影响。

**🔧 技术方法**

采用多模型提示实验（五个大语言模型），三次采样；使用LLM判别器（DeepSeek-v3.2）进行冲突认知、解决策略和立场评分；统计总变异距离（TVD）衡量提示框架差异。

**📊 数据集**

八对对立人设（如帮助 vs 谨慎、量化 vs 简单、买家 vs 卖家等）共1600个任务探针 + 8个兼容性提问，来源为公开数据与人工设计。

**📈 对比分析**

通过比较直接兼容性提问与普通任务回答的冲突披露率、解决策略标签（支配、妥协、分离等）以及立场分数分布评估模型表现；提示框架的TVD比较显示提示结构显著影响仲裁方式。性能显示模型在直接询问时冲突识别率高，但在任务回答时披露率低，仲裁往往倾向于某一人设。

**⚠️ 局限性**

仅使用LLM判别器可能带主观偏差；八个人设对不全面；仅评估五个小中型模型、单轮英文提示，无法推广到所有模型、语言和部署场景。

---

## 425. Dugong: Composing Uncertified DAG Consensus

**arXiv ID:** 2611.27473 | [PDF](https://arxiv.org/pdf/2611.27473v1)

**作者:** George Danezis `[一作]` (Mysten Labs), Alberto Sonnino `[通讯]` (Mysten Labs)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个可参数化的单一协议，能够统一并实现八个已发布的非认证 DAG 共识协议，并衍生出四个新的配置。

**💡 创新点**

创新点在于将所有非认证 DAG 共识协议抽象为单一决策规则（parameterized protocol），通过六个参数实现协议间的切换；同时在 Lean 4 上完成了完整的安全性与活性证明，并识别并修复了已有协议中的证明漏洞；再结合生产级特性（自适应选主、检查点、响应式活性、崩溃恢复）形成完整的可组合框架。

**🔧 技术方法**

使用的技术包括：参数化决策规则、Lean 4 形式化验证、Rust 实现与离散事件模拟器、以及自定义的 DAG 构造与本地视图解释。

**📊 数据集**

数据集：论文使用基于 AWS 区域间测量的网络延迟（均匀或指数抖动）进行离散事件仿真，并在多种错误模型（崩溃、Byzantine、混合）下进行随机实验，未使用真实区块链网络数据。

**📈 对比分析**

比较方法：在相同代码基、相同网络模型与相同故障注入下，评估 12 种协议配置的事务提交延迟、直达提交率、尾延迟与误差条；结果表明延迟主要取决于波长与缓冲轮数，而不受容错阈值或投票阈值影响；在高抖动和大量故障场景下，两轮协议与异步协议表现更稳健。

**⚠️ 局限性**

局限性：Rust 运行时实现与 Lean 形式化并未自动关联，仍需手工验证二者一致；实验仅覆盖仿真而非真实生产网络；未对速率限制、垃圾回收、重生成、块同步等系统级细节进行形式化；此外，异步协议的共识概率依赖随机币，理论上不保证 100% 的直达提交率。

---

## 426. GenPA: Image Generators as Efficient Robot Policies

**arXiv ID:** 2611.27446 | [PDF](https://arxiv.org/pdf/2611.27446v1)

**作者:** Zian Deng `[一作]` (Fudan University), Shengbo Eben Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6514db3d-8de6-452c-91b7-acdb31787cc4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `40105733-5154-44cd-8090-a8cab9e64b07` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

将预训练的图像生成模型（FLUX.2）通过轻量化动作回归适配器直接改造成机器人控制策略，并在训练阶段与本影深度等感知任务共同监督

**💡 创新点**

创新点在于摆脱未来视觉预测中间目标，直接利用生成模型的空间想象能力与原生编辑接口实现高效动作输出，同时提供统一的感知共训练接口，可对多种视觉目标做系统评估

**🔧 技术方法**

技术包括：预训练的流式Transformer生成模型、VAE编码器、文本编码器、动作前缀+轻量化Transformer解码器、基于原生编辑目标的深度/表面法线/语义监督、单步前向推理、编译后加速

**📊 数据集**

使用的主要数据集为 LIBERO（40 个任务）、LIBERO‑Plus（10,030 个实例）和 RoboTwin 2.0（50 个双手任务），以及 600 条真实 USB 插拔演示

**📈 对比分析**

与 VLA、WAM 等基准方法对比，GenPA 在 LIBERO 上平均成功率 98.9%（仅无额外机器人数据预训练）并在 LIBERO‑Plus 与 RoboTwin 2.0 取得接近或略优的性能；在 H100 GPU 上单次动作块推理仅 20.9 ms，较 VLA/WAM 提升 1.7–3.4 倍

**⚠️ 局限性**

局限性包括：对深度监督的依赖可能在无深度标注或复杂动态环境下受限；未探讨长期规划与多步推理需求；对不同视觉生成模型的迁移性与通用性仍待验证

---

## 427. How Many Bytes Has a BPE Model Seen? Estimating Byte-Token Exposure in Pretraining from Corpus Statistics and Embedding Norms

**arXiv ID:** 2611.27506 | [PDF](https://arxiv.org/pdf/2611.27506v1)

**作者:** Liwei Zhang `[一作]` (Landi Lab), Linzhe Li `[通讯]` (Landi Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对大型模型Byte级BPE词表中的256个单字节行进行训练曝光度估计，提出两条独立路径（语料统计与权重范数）进行交叉验证。

**💡 创新点**

首次系统量化Byte级BPE词表中的高位字节在预训练中的出现频率，并通过权重范数映射直接检验是否曾被训练。

**🔧 技术方法**

采用BPE分词、语料计数、权重范数分析、邻域反演等技术。

**📊 数据集**

使用公开的540 MB多域、多语种语料（维基百科、FineWeb、WildChat、OASST2、GitHub代码、FineMath）以及四个开源模型的权重。

**📈 对比分析**

两条路径估计结果在四个模型上相差不超过4倍；权重范数曲线可在单调区间内恢复约10^8–10^9次高位字节出现次数；对比结果显示13个非法字节从未出现。

**⚠️ 局限性**

主要限制包括混合比例假设、预训练规模未公开、样本大小限制导致稀有字节估计不准、权重无法分离预训练与后续微调、范数反演仅在单调区间有效。

---

## 428. EchoAlign: Echocardiography foundation model with multilevel report alignment

**arXiv ID:** 2611.27517 | [PDF](https://arxiv.org/pdf/2611.27517v1)

**作者:** Ahmed Aly `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Mohammad Yaqub `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

提出EchoAlign，一种针对心脏超声影像的视听语言基础模型，能够将每个心脏结构的视觉嵌入与其对应的报告文本对齐，并通过可见性矩阵限制监督仅在对应视图上进行；

**💡 创新点**

创新点在于多层报告对齐策略，将结构级、视频级和研究级嵌入与报告文本进行对齐，并结合可见性矩阵解决传统模型因全报告对齐导致的误关联问题；

**🔧 技术方法**

采用多层对比学习的视觉语言模型，结合结构级视觉编码器、文本编码器、线性探测器和卷积解码器实现分类、分级、测量和分割；

**📊 数据集**

使用了2.87百万份超声视频和图像，涵盖159个临床中心；在0.47百万份测试视频（来自25+中心）上与五个公开基础模型进行评估；

**📈 对比分析**

通过零样本文本提示和线性探测两种评估方式，与EchoPrime、PanEcho、EchoCLIP、EchoJEPA和EchoFM比较；EchoAlign在68个分类任务中平均AUROC 0.890，最高；在31个分级任务中平均QWK 0.530，最高；在测量和分割任务中也表现出竞争力；

**⚠️ 局限性**

研究为回顾性，缺乏前瞻性验证；文本提示的效果受词汇敏感；部分指标在小样本外部数据集（如CardioBench）置信区间较宽；缺乏心率、M‑mode等模态输入，某些测量可能不够精确。

---

## 429. The Times They Are A-Changin': Tracking Global Trends through the Lens of Web Search

**arXiv ID:** 2611.27613 | [PDF](https://arxiv.org/pdf/2611.27613v1)

**作者:** Anju Bhuiya `[一作]` (IIT Kharagpur), Abhijnan Chakraborty `[通讯]` (IIT Kharagpur)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了 2011–2025 年间 Google Year in Search 的大规模纵向数据集，并统一了十类语义分类，随后分析了信息寻求、人物关注、娱乐与体育四大领域的跨国、跨时段关注结构，揭示了全球化与地方化并存的“glocalization”模式及危机对关注的影响。

**💡 创新点**

①首次将 Year in Search 公开摘要系统化为可比的纵向数据集；②设计并手工标注了统一语义分类与实体职业/流派标签；③通过多维度相似度与聚类量化跨国关注的一致性与差异性，展示了全球范围内既统一又分化的关注结构。

**🔧 技术方法**

使用 Selenium 自动爬取网页并通过 Google Translate 统一语言；人工标注并计算 Cohen’s κ 评估标注一致性；利用 Jaccard 相似度、KL 散度、Mann‑Kendall 趋势检验、Ward 链接层次聚类和 Silhouette 系数等统计方法对关注结构进行度量和可视化；对危机期使用百分点差异和聚类分析揭示关注转移。

**📊 数据集**

Google Year in Search 2011–2025 年的国家级趋势条目（平均 52 个国家/年，共 6,772 条目），配合人工构建的十类语义词表和 29 类信息寻求子类；对人物职业、娱乐与音乐流派、体育项目等实体进行手工注解，并使用公开 API（IMDb、Wikidata）获取额外属性。

**📈 对比分析**

通过计算各国/地区在不同类别下的词汇/实体分布相似度（Jaccard、KL 散度）并与地理和收入分组进行 Mann‑Whitney 检验，评估相似度差异；对职业分布使用层次聚类得到三大聚类并用 Silhouette 系数（0.062）评估聚类质量；危机期关注转移通过百分点增幅对比，显示全球与地区差异。整体上，注释一致性高（κ≈0.83），相似度指标稳定且能揭示明显的 glocalization 特征。

**⚠️ 局限性**

仅覆盖 Google 已公开的国家，缺乏中国、俄罗斯等主要市场；数据仅按国家聚合，缺少细粒度时空分辨率和人口层面信息；Year in Search 是 Google 选取的摘要，不能代表完整搜索日志，可能存在平台自选偏差。

---

## 430. Circular Predictive Coding Unlocks Effective Depth

**arXiv ID:** 2611.27610 | [PDF](https://arxiv.org/pdf/2611.27610v1)

**作者:** Tommaso Salvatori `[一作]` (Relational Intelligence, Inc), Matteo Forasassi `[通讯]` (Relational Intelligence, Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种循环预测编码网络（CPCN），通过闭环实现深度等效，并将其训练视为等价的深度平衡模型（DEQ），在图像分类、语言建模和约束求解（数独）任务上进行实验。

**💡 创新点**

创新点在于：① 将循环结构与预测编码结合，使有效深度由能量松弛时间决定；② 证明CPCN的自由相平衡与DEQ的固定点唯一对应；③ 通过等势传播（EP）近似DEQ的隐式梯度，实现局部无梯度传播学习；④ 引入双读出与递归阶段增强计算深度与自适应推理。

**🔧 技术方法**

使用的技术包括：预测编码图结构、等势传播（EP）与中心化/随机注入、深度平衡模型（DEQ）与隐式函数定理、卷积与Transformer循环块、Newton/二阶求解器、层归一化与谱正则化。

**📊 数据集**

实验数据集：CIFAR‑10、CIFAR‑100、ImageNet32（图像分类）；TinyStories、WikiText‑103（语言建模）；Sudoku‑hard（约束求解）。

**📈 对比分析**

与传统PC、ResNet‑18、VGG‑15等深度网络以及DEQ进行对比。CPCN 在 CIFAR‑10/100 上分别取得 93.54% / 72.38% 的准确率，超过同规模 VGG‑5 与 ResNet‑18；在 ImageNet32 上也实现 39.31% 的准确率。数独任务中递归CPCN 在 1024 视图下达 99.3% 的正确率，远高于单视图 68.2%。在语言建模上，CPCN 的验证/测试困惑度分别为 12.96 / 110.83，略优于对应的 DEQ。实验显示 CPNC 通过循环松弛实现了可扩展的有效深度。

**⚠️ 局限性**

局限性包括：① 训练过程需要多次能量松弛和注入阶段，计算成本高；② 对于更大规模数据或更深网络，能量收敛速度和数值稳定性仍有挑战；③ 目前仅在实验室级 GPU 模拟下验证，尚未在硬件或模拟电路上实现；④ 递归阶段的超参数（如阶段数、视图数）对性能影响显著，需要进一步自动化和理论分析。

---

## 431. Planar Formation Control with Minimal Sensing and No Common Orientation: An Almost-Global Result

**arXiv ID:** 2611.27595 | [PDF](https://arxiv.org/pdf/2611.27595v1)

**作者:** Jingyi Zhao `[一作]` (University of Granada), Hector Garcia de Marina `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出一种基于参数化圆形流形与虚相位协同的平面形成控制方法，可在任何连接图的任意生成树上实现预定几何形状（仅保留平移和旋转自由度）。

**💡 创新点**

创新点在于：仅使用 N-1 条感知连线和每条边一个标量即可完成形成；虚相位与圆形参数化不需要全局方向共识；对任何生成树均实现几乎全局收敛，星形生成树全局收敛；提供完整的平衡分析与不稳定性说明。

**🔧 技术方法**

采用导引向量场（GVF）框架、梯度流控制、虚相位耦合、局部坐标实现；通过势能函数、拉贾斯基不等式、图论与线性代数证明收敛性与平衡结构。

**📊 数据集**

无公开数据集，采用理论推导与数值仿真（8 点路径图、20 点二叉树）进行验证。

**📈 对比分析**

仿真结果显示，在任意生成树上，形状误差指数衰减，虚相位误差线性衰减；对多初始条件几乎全局收敛，最终误差可低至 10⁻⁸ 级别。

**⚠️ 局限性**

局限性：仅考虑单纯形动力学，未包含碰撞避免；对非树图需要额外处理；未处理非齐次或非可控约束；内部边的崩塌状态为不稳定，可能在瞬态中出现。

---

## 432. Synthesizing Expressive Human-Object Interactions via Test-time Latent Adaptation

**arXiv ID:** 2611.27603 | [PDF](https://arxiv.org/pdf/2611.27603v1)

**作者:** Zijing Wu `[一作]` (Southeast University), Yangang Wang `[通讯]` (Southeast University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种基于测试时潜在空间自适应的物理驱动人机交互动画框架 ExpHOI，能够通过试错过程生成表达性动作。

**💡 创新点**

创新点在于将潜在空间自适应（AWR）嵌入低层控制器，实现对未知物体属性的试错估计，从而产生可观察的“表达性 exertion”，并结合高层失败感知规划完成完整动画。

**🔧 技术方法**

采用强化学习训练的物理控制器、信息瓶颈约束、优势加权回归（AWR）测试时自适应，以及基于规则的状态机规划。

**📊 数据集**

使用自制的 Mass‑Diverse Dataset (MDD)，包含 4 万多件带有 13 个物理属性标注的物体。

**📈 对比分析**

与 CooHOI、InterPhys 等基线对比，ExpHOI 在负载范围、成功率、姿态多样性（APD）和用户满意度（MOS）方面均优于基线；相对 Oracle/Privileged 输入仅略低成功率，但姿态多样性最高。

**⚠️ 局限性**

目前仅支持单手臂抓取/推拉，缺乏手部细节、全局导航规划，并且在更广泛场景中的泛化仍待验证。

---

## 433. TITAN: DRL-Based Dynamic Scheduling and Resource Allocation for Tactile Internet

**arXiv ID:** 2611.27560 | [PDF](https://arxiv.org/pdf/2611.27560v1)

**作者:** Qingrui Fan `[一作]` (Ghent University), Johannes Cottyn `[通讯]` (Ghent University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计并实现了TITAN系统，一种针对5G网络中触觉网络（Tactile Internet）的双时隙动态调度与资源分配方案；

**💡 创新点**

创新点在于结合时间变动的触觉QoS档位与深度强化学习（PPO）进行慢时隙的频谱切分，同时在快时隙采用低开销调度器，实现对触觉与视频流的联合QoS满足；

**🔧 技术方法**

使用的技术包括两时隙分层架构、深度强化学习（PPO）、离线训练的决策网络、快速调度（urgency‑ordered round‑robin 与 least‑laxity‑first）以及基于5G NR的资源块调度；

**📊 数据集**

通过仿真数据集完成评估：随机生成的触觉UE与视频UE数量、基于广义帕累托分布的触觉包到达、截断高斯分布的视频帧大小；

**📈 对比分析**

与固定切分策略和理想Oracle策略进行比较，TITAN在联合满足率（JSR）上比最佳固定切分提升7.64个百分点，Reward‑Aligned Score（RAS）提升0.1183，整体性能显著优于静态方案；

**⚠️ 局限性**

局限性包括依赖离线训练的DRL模型、仅在仿真环境下验证、未考虑预测或实时环境反馈、只覆盖上行链路、假设信道SNR静态且仅为随机采样。

---

## 434. Bringing AI Assistance to the Beznau Nuclear Power Plant: A Hierarchical System of LLM Agents for Reactor Monitoring

**arXiv ID:** 2611.27556 | [PDF](https://arxiv.org/pdf/2611.27556v1)

**作者:** Wenqi Zhou `[一作]` (Zurich University of Applied Science), Manuel Arias Chao `[通讯]` (Zurich University of Applied Science)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个分层多智能体助手，利用大型语言模型（LLM）在瑞士Beznau核电站的旧有监测系统中支持监测异常调查，并将操作数据、维护记录和技术文档等异构信息统一检索、分析与呈现给工程师。

**💡 创新点**

创新点包括：①将专家知识转化为可读、可修订的自然语言程序，让工程师自行审阅；②在严格可追溯、只读访问的框架下，采用协调型多智能体架构将检索、分析和推理分工给专门代理；③在安全监管环境中实现源代码可追溯、操作决策仍由人负责的“辅助”模式；④在真实核电站环境中部署并演示，验证了技术在复杂遗留系统中的可行性。

**🔧 技术方法**

使用技术主要有：Azure OpenAI GPT‑4o（LLM）、检索增强生成（RAG）+向量检索、自然语言处理生成式推理、基于REST API的浏览器接口、调度/执行框架（Agent runtime）、局部追踪平台、Azure本地化部署以满足机密性需求。

**📊 数据集**

数据集为Beznau核电站内部多源数据：ANIS与GARDEL监测系统、报警日志、变量定义、技术规范、历史调查报告、组织数据库等，全部以只读方式读取。

**📈 对比分析**

方法评估主要通过案例演示（3种典型查询），展示了多智能体协同检索、分析与可视化的流程；但论文未给出定量性能指标（如检索准确率、推理正确率或工程师工作量减少比例），仅提供定性说明与初步的可用性反馈。

**⚠️ 局限性**

局限性包括：①缺乏系统化的量化评估与基准比较；②目前仅在试点环境下部署，缺乏大规模生产验证；③对域程序的维护和演进提出挑战；④数据隐私与合规性仍需进一步完善；⑤安全关键性场景下对错误判定的风险评估不足。

---

## 435. GameScrub: Learning HUD Removal from Real Gameplay with Recovered Targets

**arXiv ID:** 2611.27547 | [PDF](https://arxiv.org/pdf/2611.27547v1)

**作者:** Mohammad Areeb Qazi `[一作]` (Institute of Foundation Models), Guangyi Liu `[通讯]` (Institute of Foundation Models)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计了无需掩码即可去除真实游戏视频中HUD的模型GameScrub，并提出从真实受污染视频恢复近似干净目标的训练流程GameScrub-Data。

**💡 创新点**

通过从真实受污染视频中生成伪干净目标，避免了合成HUD训练带来的分布偏差，实现了无掩码、高效的HUD去除。

**🔧 技术方法**

结合流引导背景填充、冻结视频扩散模型的遮罩去噪、像素级混合以及参考条件的低秩适配器，在LTX-2.5扩散变换器上进行训练。

**📊 数据集**

在Game2World“in-the-wild”真实游戏片段（100个真实剪辑）上训练和评估，并使用GameScrub-Data生成的训练对。

**📈 对比分析**

与GameCleaner、通用视频编辑器等基线比较，GameScrub在AAR 71.95、Overall 82.13、去除率90.06%、背景保留63.04%，仅需11步推理，比GameCleaner快2.1倍。

**⚠️ 局限性**

去除完整帧导致背景细节变化，且依赖于HUD检测与预填充的准确性，恢复目标质量不足时会限制最终去除效果。

---

## 436. On the Limits of Observability for Evaluating Coverage in Machine Unlearning

**arXiv ID:** 2611.27543 | [PDF](https://arxiv.org/pdf/2611.27543v1)

**作者:** Ziluowen Luo `[一作]` (Central South University), Senzhang Wang `[通讯]` (Central South University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 CEO 框架，用于在有限可观测性下评估机器学习模型的删除覆盖率，并在合成的颜色‑形状基准上对 20 种删除方法进行系统评测。

**💡 创新点**

创新点在于：① 引入覆盖率（coverage）视角揭示评估与删除请求范围之间的可观测差距；② 通过线性得分与共享表示的秩下界证明了对单例与子集删除的容量限制；③ 利用稀疏自编码器（SAE）进行维度扩展以弥补观察盲点。

**🔧 技术方法**

主要技术包括：可观测性评估、成员推断攻击（MIA）、线性可选性分析、秩下界理论、稀疏自编码器 (SAE) 维度扩展、留一重训练检验以及联合频率和零计数评估规则。

**📊 数据集**

使用的数据集包括：合成颜色‑形状对任务（16、64、256 对）；两张真实视觉数据集（未具体列举）；两类合成语言任务（人名‑密码、研究者‑研究所）。

**📈 对比分析**

对 20 种删除方法在合成基准上进行对比，采用覆盖率、真实阳性率（TPR）、误报率（FPR）以及残留目标质量等指标；结果显示即便在最小化训练步骤后，覆盖率仍难以达到 100%，且绝大多数方法的误报率高于阈值。

**⚠️ 局限性**

局限性包括：① 评估仍依赖于留一重训练的昂贵算力；② 仅在合成基准上展示，真实数据上效果未知；③ 维度扩展仅在单一 SAE 架构下验证，其他更复杂的观察空间未探究；④ 对大规模子集删除覆盖率的推断仍需进一步理论与实验支持。

---

## 437. Epistemic Social Learning: Latent Behavioral Structure under Endogenous Multi-Agent Interaction

**arXiv ID:** 2611.27545 | [PDF](https://arxiv.org/pdf/2611.27545v1)

**作者:** Jainendra Shukla `[一作]` (Indian Institute of Information Technology Delhi), Kiriti Kanjilal `[通讯]` (Indian Institute of Information Technology Delhi)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在多智能体交互中提出了Epistemic Social Learning（ESL），通过快速的贝叶斯推理与慢速的共享行为原型学习相结合，实现对对手行为的适应。

**💡 创新点**

创新点在于将行为原型的在线学习与交互特定的贝叶斯信念分为两时间尺度，并通过闭环的策略-观测-学习循环来解决预测拟合与决策适应的分离问题。

**🔧 技术方法**

采用贝叶斯推理、软最大行为模型、logit决策、控制马尔可夫随机逼近及两时间尺度的随机逼近分析等技术。

**📊 数据集**

在十二个3×3的合作、协调与对抗游戏的人工合成数据集上进行实验。

**📈 对比分析**

与Online EM、固定原型贝叶斯滤波和switch-aware固定原型滤波等基线对比，ESL在后切换期的oracle regret（PSR）最低，表现出更好的适应性，但在决策相关预测误差（DPR）方面略逊于SA‑FP。

**⚠️ 局限性**

主要局限包括仅在结构化的离散重复游戏上验证，无法直接推广到高维序列环境；两时间尺度的理论结果是条件性的；缺乏对更复杂对手群体和连续状态空间的评估。

---

## 438. Fix Where It Fails: Root-Cause-Guided Agentic Harness Optimization for Scene Graph Generation

**arXiv ID:** 2611.27529 | [PDF](https://arxiv.org/pdf/2611.27529v1)

**作者:** Jee Eun Kim `[一作]` (Korea University), Kyong Hwan Jin `[通讯]` (Korea University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文通过根因分析指导的主线优化，自动搜索并修正冻结的场景图生成模型的推理链条，从而提升模型性能而无需重新训练。

**💡 创新点**

创新点在于首次将根因分析与LLM代理相结合，对模型错误进行组件级拆解，并据此驱动主线搜索，形成面向SGG的Agentic Harness Optimization。

**🔧 技术方法**

主要技术包括：LLM代理（Claude Opus 5.5）进行根因分析与方案生成，Shapley值分解组件误差，多通道推理 harness 结合外部工具（检测器、后处理）以及多轮搜索策略。

**📊 数据集**

使用公开数据集 Visual Genome（VG150）和 Pascal-SG（PSG），并在 PredCls、SGCls、SGDet 等三种评估设置下进行实验。

**📈 对比分析**

与官方基线和无指导/知识指导搜索对比，单/多通道 harness 在所有五个模型-数据集组合上均提升所有指标；R1‑SGG 在 VG150 上 R@100 +15.8、mR@100 +26.4，根因引导搜索速度可提升至三倍。

**⚠️ 局限性**

局限性包括：每个模型仅搜索一次，缺乏多轮迭代；仅评估两阶段、一阶段及多模态LLM模型，未覆盖 PredCls/SGCls 评估；推理与搜索成本较高，依赖外部 API 费用。

---

## 439. PennyGS: Full Resolution 3DGS Training in Minutes on Consumer GPUs

**arXiv ID:** 2611.27525 | [PDF](https://arxiv.org/pdf/2611.27525v1)

**作者:** Nicholas Jing Shou Nge `[一作]` (National University of Singapore), Wei Tsang Ooi `[通讯]` (National University of Singapore)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计了一套在消费级 GPU 上完成 4K 分辨率 3D Gaussian Splatting 训练的系统，显著缩短训练时间至几分钟；

**💡 创新点**

通过消除训练每一步中的冗余工作实现高效：引入平面 RGB + 方向锥形不透明度的轻量化 splat 表示；融合投影与键发射、反向投影与稀疏 Adam；单通道 tile SSIM 损失；每步只监督四分之一像素的 stride 采样；每 warp 对应一个 splat 的 backward rasterization；以及多项内存与计算优化；

**🔧 技术方法**

方向锥形不透明度、平面 RGB、稀疏 Adam、bf16 矩阵、投影+键发射融合、反向投影+Adam 融合、tile SSIM、stride pixel sampling、warp‑chunk backward rasterization、裁剪与压缩 moment 等技术；

**📊 数据集**

使用 Mip‑NeRF 360（7 场景）和 Tanks & Temples（8 场景）全分辨率 4K 数据集；

**📈 对比分析**

在 RTX 3060、3070、3090、4090 四款消费级 GPU 上与 9 种现有训练器（包括 FastGS、LiteGS、LichtFeld 等）对比，采用 30k 步预算；本方法速度至少比最快对手快 4–5 倍，训练时间仅 4–7 分钟（RTX 4090 2 分钟以下），质量保持竞争力（PSNR 28.96 dB、SSIM 0.855 等），GPU 内存占用 <2 GB；

**⚠️ 局限性**

方向锥 splat 在远离训练视角时会出现强烈伪影；stride pixel sampling 对低分辨率效果不佳；仅在 NVIDIA GPU 上验证，AMD 与 Apple silicon 等平台需进一步支持。

---

## 440. Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving

**arXiv ID:** 2611.27515 | [PDF](https://arxiv.org/pdf/2611.27515v1)

**作者:** Xinyao Rao `[一作]` (Nanjing University), Wei Li `[通讯]` (Nanjing University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

利用离线搜索和执行评估生成评分轨迹替代方案，增强生成与残差细化训练，提升端到端自动驾驶规划器的候选质量与选择精度。

**💡 创新点**

通过为每个日志场景构造多样化、已评估质量的轨迹替代并将其作为监督，融合共享的JEPA时序特征，实现在残差细化阶段的多目标去噪与质量引导。

**🔧 技术方法**

结合DINOv2视觉编码、Diffusion Transformer、JEPA时序预测、DALL‑E style去噪与多目标匹配，形成完整的生成-评估-细化流水线。

**📊 数据集**

在NAVSIM v1/v2、SimScale、HUGSIM等大规模真实与模拟驾驶数据集上进行训练与测试。

**📈 对比分析**

与ChainFlow‑VLA、FPD‑Drive‑Pro等基线对比，DriveSTA在Navtest PDMS达到95.6分，HUGSIM路程完成率与HD‑Score显著领先，并将前向推理延迟降低73%。

**⚠️ 局限性**

受限于基于单人示范的形变搜索范围、评估器约束以及离线搜索的计算开销，缺乏对更广泛操控与连续规划一致性的直接监督。

---

## 441. Gauge-equivariant real-space renormalization of frustrated magnets and topological spin textures

**arXiv ID:** 2611.27598 | [PDF](https://arxiv.org/pdf/2611.27598v1)

**作者:** Musen Li `[一作]` (Shanghai University), Tong-Yi Zhang `[通讯]` (Shanghai University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `40105733-5154-44cd-8090-a8cab9e64b07` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于独立集解耦、精确矩阵Schur共轭与图着色SMC‑AIS桥的GaugeRGFlow框架，能够在任意几何易受挫折的自旋网络上实现精确的实空间重整与热平衡采样。

**💡 创新点**

创新点在于将离散键旋转几何与连续SO(3)等变流结合，构造可精确保留挫折度量的多尺度解耦重整，并通过并行SMC‑AIS消除拓扑冻结，实现0.01%精度的自由能估计。

**🔧 技术方法**

使用了独立集解耦的Schur共轭、极化奇异值分解、SO(3)极化拉升、图着色SMC‑AIS、热浴+过度松弛以及实时SP‑STM诊断等技术。

**📊 数据集**

验证数据包括DFT校准的fcc‑Pd/Fe/Ir(111)与Gd_2PdSi_3磁层、219原子ruby‑lattice Rydberg模拟器实验截图，以及经典三角/四格/±J自旋玻璃模型。

**📈 对比分析**

与固定2×2神经RG、VAN、并行温度耦合等方法比较，GaugeRGFlow在自由能上提供更紧的变分界限，采样速度提升约10倍，并能准确捕捉费曼熵、skyrmion熔点和自旋液体的相互关联。

**⚠️ 局限性**

局限性包括对强磁偶极相互作用、真实材料的结构缺陷以及存在大量全局耦合或量子相干性的问题仍不适用于该经典生成模型。

---

## 442. Deep Cross-Division Network for Product-and-Ratio Feature Interactions in Industrial Ranking

**arXiv ID:** 2611.27514 | [PDF](https://arxiv.org/pdf/2611.27514v1)

**作者:** Mikhail Trapeznikov `[一作]` (AI VK), Alexander D'yakonov `[通讯]` (AI VK)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种 Deep Cross-Division Network (DCDN)，在标准 CrossNet 层中加入除法分支，自动学习产品、比例和混合交互，从而在工业推荐系统中消除手工比例特征。

**💡 创新点**

创新点在于：①引入了除法分支，使交叉层能生成拉朗日多项式（Laurent polynomials）并捕获负幂（比例）关系；②对 DCDN 的可表达性进行严谨数学表述，并证明单层即可产生所有一阶乘积、比例及其倒数；③在工业线上部署后显著降低 40% 输入特征量，同时提升多项用户互动指标。

**🔧 技术方法**

技术包括：Deep & Cross Network（DCNv2）改造、递归交叉层与除法分支、拉朗日多项式分析、对数均匀训练的合成比例学习实验、公开数据集（Yambda、KuaiRec、KuaiRand、VK‑LSVD）上的离线评估、A/B 测试、Triton 推理加速与动态批处理。

**📊 数据集**

使用公开推荐任务数据集：Yambda‑50M、KuaiRec、KuaiRand、VK‑LSVD；以及本公司短视频服务的内部数据进行线上 A/B 实验。

**📈 对比分析**

比较方法：离线以 AUC/XAUC、nDCG@10 评估模型质量；模型大小与推理时延（p50）对比；生产阶段以离线 XAUC、线上 A/B 相关指标（点赞、分享、总观看时长、全屏点击）和服务层级延迟衡量。DCDN 在所有公开任务上均优于 DCNv2，且比 AFN+ 与 EulerNet 低 5–10 倍时延；线上 A/B 结果显示点赞+3.25%，分享+16.26%，观看时长+0.33%，全屏点击+0.38%，同时模型计算时间下降 42.4%、整体排名服务时间下降 10.5%。

**⚠️ 局限性**

局限性包括：①对比例分母的除法掩码需人工指定，自动化选择尚未解决；②DCDN 不是所有任务的最佳模型，在某些数据集（如 KuaiRand）中被 AutoInt+ 或 EulerNet 超越；③实验室推理时延与实际服务器时延受硬件、批处理和 kernel 优化影响，无法完全归因于模型结构；④除法分支在极端数值范围外会出现数值不稳定，需要裁剪策略。

---

## 443. Unlocking Heterogeneous Accelerator Flexibility in AI Data Centers: Sizing and Workload Allocation

**arXiv ID:** 2611.27576 | [PDF](https://arxiv.org/pdf/2611.27576v1)

**作者:** Chenghao Lyu `[一作]` (Imperial College London), Fei Teng `[通讯]` (Imperial College London)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文提出一种联合异构加速器容量尺寸化与AI工作负载跨站点、跨时段调度的方法，以满足电网容量约束并降低运营成本。

**💡 创新点**

创新点在于：①将异构硬件的资本成本、功耗和工作负载特异性吞吐/能耗参数嵌入设施功率映射；②在同一优化框架下同时决定加速器数量与分布及工作负载分配；③通过基准模型层级剖析时间、空间与硬件异质性对成本的具体贡献；④评估模块化服务器实现对连续线性规划解的影响。

**🔧 技术方法**

采用混合整数/线性规划（Gurobi求解器）构建多站点时空调度模型，结合功耗平衡、兼容性、完成期限、QoS延迟惩罚等约束，求解最优资本、能源与服务成本。

**📊 数据集**

实验使用基于英国伦敦与爱丁堡的代表性双站点案例，包括各站点电价、PUE、网格容量边界、互联延迟等参数；未使用公开数据集，而是基于行业典型工作负载（模型训练、延迟可容忍批处理、在线推理）构造合成负载需求。

**📈 对比分析**

与四类基线模型（A：刚性排程；B：局部时移；C：空间时配；D：联合优化）以及工作负载平均化基线（E）对比。联合优化模型在双站点案例中每日等价成本比固定容量模型低6.5%，比平均化基线低7.7%；实现网格边界约束时仅增加0.25%成本；模块化服务器求解与连续模型的差距仅0.78%。

**⚠️ 局限性**

局限性：仅处理初始工作负载分配和请求路由，未考虑训练状态的实时迁移；缺乏对需求波动和电网不确定性的鲁棒性分析；模型假设工作负载参数已知且不变，未来需加入动态不确定性和自适应控制。

---

## 444. Can Language Models See the Circuit in Pixelated RFIC Passives?

**arXiv ID:** 2611.27559 | [PDF](https://arxiv.org/pdf/2611.27559v1)

**作者:** Jonathan Zhou `[一作]` (Princeton University), Kaushik Sengupta `[通讯]` (Princeton University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了大语言模型（Claude Opus 5.5）在无工具环境下读取、解释并调试像素化 RFIC 被动器件的能力，探索其生成等效电路、预测频率响应并在有限编辑范围内修复失效设计的可能性。

**💡 创新点**

创新点在于将 LLM 视为能理解逆向设计像素化结构、给出可解释的等效电路并在 EM 反馈下进行智能编辑的“诊断与修复”工具，首次证明 LLM 能在非直观像素化布局中做出有效的硬件改进。

**🔧 技术方法**

使用技术包括：前沿大语言模型（Claude Opus 5.5）零工具推理；基于文本的像素网格布局输入；闭式电路和传输线理论推导；对等效电路与 EM 仿真结果进行对比评估；有限的 EM 反馈循环（≤5 次）来验证编辑效果。

**📊 数据集**

数据集包含 22 个布局样本，涵盖 8 个 GA 合成匹配网络（18×18 或 27×27 网格）、5 个随机填充、1 个直线、8 个手绘多路径循环、7 个失败 GA 设计，总共 7×2 频点（20–100 GHz）以及相应的 EM 仿真数据；所有布局基于 Princeton PDK 的同层金属结构。

**📈 对比分析**

通过与全波 EM 仿真结果比较，评估 |ΔΓ|（Smith 变换距离）、频率缩放因子 s*、误差来源等指标。读任务平均 |ΔΓ| 为 0.30（GA）/0.15（随机），多路径结构的 notch 预测误差中值为 1.11×；调试任务中 7/7 设计被改进，5/7 满足规格，且平均降幅约 0.87 dB。相较于无技巧猜测 50 Ω，性能提升显著。

**⚠️ 局限性**

limitations: 仅测试单一模型与单一技术水平，样本量有限；缺乏与传统搜索或随机编辑的基准对比；频率偏移存在系统性误差，尤其在 80 GHz 频段预测更乐观；未评估模型在多 PDK 或更大网格上的泛化能力；对工具调用的完全禁用可能限制了更复杂推理的可能性。

---

## 445. Joint Optimization of Fleet Size, Localization Infrastructure, and Trajectories for Multi-AUV Persistent Monitoring

**arXiv ID:** 2611.27539 | [PDF](https://arxiv.org/pdf/2611.27539v1)

**作者:** Jan Mikołajczyk `[一作]` (University of Warsaw), Maksymilian Bernacki `[通讯]` (Warsaw University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了多AUV持续监测任务中轨迹规划、舰队规模与定位基础设施的联合优化，以提升时间覆盖质量。

**💡 创新点**

创新点在于提出基于Voronoi分区的自适应轨迹策略，并通过成本感知框架量化定位精度与轨迹设计的边际收益，揭示了定位阈值与协同设计的交互影响。

**🔧 技术方法**

采用随机走步、lawnmower与Voronoi三种轨迹生成方法，结合2D海流模拟、差分进化与TPE调参的仿真框架进行实验。

**📊 数据集**

使用离散网格与虚拟离岸风电场工作空间的模拟数据，未使用公开真实数据集。

**📈 对比分析**

通过AoI、IVT、覆盖率、RMSE、能量代理和生命周期成本等指标比较，结果显示Voronoi策略在3个定位基站阈值后即可达到与稠密基站相近的时间覆盖，并实现最低能耗和成本。

**⚠️ 局限性**

局限性包括二维仿真、同质舰队、理想化通信、简化成本模型以及未考虑深度和三维任务等。

---

## 446. Sketch2TikZ: Multi-Reward Reinforcement Learning for Converting Hand-Drawn Sketches to Executable TikZ

**arXiv ID:** 2611.27620 | [PDF](https://arxiv.org/pdf/2611.27620v1)

**作者:** Shubham Paliwal `[一作]` (TCS Research), Manasi Patwardhan `[通讯]` (TCS Research)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出Sketch2TikZ框架，将手绘草图自动转化为可编译的TikZ代码。

**💡 创新点**

创新性地使用分阶段多奖励的强化学习（GRPO）结合结构、OCR、CleanDIFT和自相似奖励，减少奖励干扰并提升视觉与结构一致性。

**🔧 技术方法**

采用GRPO强化学习、SigLIP视觉编码、TinyLlama/DeepSeek-Coder语言解码、ChkTeX/ pdflatex 编译检测、OCR匹配、CleanDIFT拓扑一致性、CLIP/DINO/OpenCLIP自相似度等技术。

**📊 数据集**

使用SkeTikZ（3,231对）作为主要训练与评估数据集，并在FlowChart-3B上进行零样本鲁棒性测试。

**📈 对比分析**

通过与DeTikZify、LLaVA-Next、Qwen-3等开源基线以及GPT‑4o、GPT‑5等闭源模型对比，Sketch2TikZ在编译成功率、图像相似度、字符相似度等指标上均优于开源模型，且在编译成功率上与闭源模型相当或更好。

**⚠️ 局限性**

局限性包括对代理奖励的过度优化风险、对极端噪声草图的鲁棒性仍有限、模型规模受限且仍需大规模数据与更强算力支持，以及缺乏对更复杂结构图的全面评估。

---

## 447. Open CDLM: A Benchmark and Library for Continuous Diffusion Language Models

**arXiv ID:** 2611.27536 | [PDF](https://arxiv.org/pdf/2611.27536v1)

**作者:** Hao Jiang `[一作]` (Polyadic Research Group), Tianyang Hu `[通讯]` (Polyadic Research Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并推出了 Open CDLM 基准和对应的评估库，用于在同一实验框架下可靠比较连续扩散语言模型（CDLM）的模型拟合度与生成文本分布。

**💡 创新点**

创新点在于：①将模型拟合、采样器行为和分布一致性拆分为独立可评估的子任务；②引入 Kernel Text Discrepancy (KTD) 与 MAUVE 等分布度量，补充传统 Gen PPL 与词频熵的不足；③系统性剖析变分上界与重要性加权在 CDLM 评估中的可识别性与局限，揭示单参数参考尺度优化并不能确定整体对数似然缺口。

**🔧 技术方法**

技术手段包括：变分上界（ELBO）与多样本重要性加权估计、Kernel MMD 变体 KTD、MAUVE、判别式 NLL（使用 GPT‑2‑large）计算、统计诊断（重复率、局部重复、有效长度、词表覆盖率、词性分布）、以及统一的实验接口与自动化评估流程。

**📊 数据集**

主要使用 OpenWebText (OWT) 作为生成与评估文本的真实分布，LM1B 作为扩展数据集；判别器使用 GPT‑2‑large；实验还涉及从 OWT 提取的 512‑行/1024‑token 片段。

**📈 对比分析**

比较方法：在相同的训练集、词表、长度约束、判别器与特征空间下，比较 NLL 上界、KTD、Gen PPL、内序列熵、词表覆盖率、词性分布等指标。结果显示，连续扩散模型在 Gen PPL 上优于离散基线，但在词汇多样性和分布一致性（KTD、MAUVE、覆盖率）方面仍不如 AR 变压器；KTD 能有效检测单文本复制、模板崩溃等生成失效。

**⚠️ 局限性**

局限性：实验仅覆盖 OWT 的短文本设置，缺少文档层级标识；评估基准只包含一个 LangFlow 检查点，未对 RePlaid 等模型的真实 PPL 进行全面测评；数值误差与采样偏差可能影响上界与 KTD 结果；不同实现细节（如采样器、解码策略）对评估结果的可迁移性仍需进一步验证。

---

## 448. Isonomy: Isolating Every Request from Timing Side Channels at Nearly Full-Sharing Throughput in Multi-Instance LLM Prefill

**arXiv ID:** 2611.27596 | [PDF](https://arxiv.org/pdf/2611.27596v1)

**作者:** Bowen Sun `[一作]` (Johns Hopkins University), Chaowei Xiao `[通讯]` (Johns Hopkins University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在多租户LLM部署中安全共享前缀缓存，并保持高吞吐量；

**💡 创新点**

提出按请求组隔离缓存并隐藏跨组命中，设计预测和执行两种模式来平衡隐私与利用率；

**🔧 技术方法**

基于vLLM实现的路由规则、可视/隐藏缓存管理、预测/执行模式以及直接探测测量技术；

**📊 数据集**

在三款商业API及两、四、八实例上进行实验，并使用包含账号标签的两个工作负载进行吞吐量评估；

**📈 对比分析**

通过监听器检测无法区分预填充工作量，直接探测测试测量探测优势，结果显示预测模式在饱和时保留了0.98的吞吐量，隔离模式则在0.95（无跨账户重用）或0.80（高跨账户重用）之间；

**⚠️ 局限性**

局限性包括：仅在客户端无法观测彼此负载的场景有效；跨账户共享时吞吐下降；对工作负载特性和缓存规模敏感，可能需要进一步调优。

---

## 449. Quantum-Informed Regularized Poisson Learning for Graph-Based Semi-Supervised Classification

**arXiv ID:** 2611.27609 | [PDF](https://arxiv.org/pdf/2611.27609v1)

**作者:** Hamed Gholipour `[一作]` (University of Beira Interior), Moharram Challenger `[通讯]` (University of Antwerp)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出IPQSSL框架，将每个节点的局部图特征通过QR分解映射为量子可编码形式，利用共享参数量子电路测量得到节点对角正则化权重，再与经典改进Poisson传播结合实现半监督学习；

**💡 创新点**

创新点在于：①使用QR分解实现本地图特征的量子兼容编码；②通过共享量子电路生成自适应节点对角正则化；③对对称Hermitian、块编码兼容的正则化Poisson算子进行形式化，给出正定性条件与未来量子线性系统实现路径；

**🔧 技术方法**

采用的技术包括经典Poisson传播、QR分解、参数化共享量子电路（RzRyRx旋转+环形CNOT）、SPSA自适应优化、块编码与QSVT、CG/MINRES求解、以及纠缠熵和随机化基准等量子诊断；

**📊 数据集**

实验数据集为七个基准：Iris、Wine、Breast Cancer、Heart Disease、German Credit、Optdigits 与 Cora；

**📈 对比分析**

通过与原始Poisson、二次正则化Poisson、IPL（无正则化）及五种对角正则化变体的配对比较，发现IPQSSL在Iris、Wine和German Credit等在足够标签数时表现最优；但在Breast Cancer、Heart Disease、Optdigits和Cora上不具优势，且对角正则化本身已能解释大部分提升，量子电路训练对性能贡献有限；

**⚠️ 局限性**

局限性包括：量子电路仅用于产生正则化，未实现真正的量子加速；对角正则化提升是通用现象，量子特性贡献不明显；实验全部基于模拟；缺乏匹配容量的经典基线；Optdigits表现不稳定、German Credit未突破多数类基准；深度/宽度变化对性能影响不大，诊断表明量子资源受限。

---

## 450. Vega-Fafnir: Is Greedy Decoding the Same Decoding? Where Rounding Noise in Speculative Verification Comes From and What Removing It Costs

**arXiv ID:** 2611.27554 | [PDF](https://arxiv.org/pdf/2611.27554v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 451. Mjolnir: Finding and Reproducing Vulnerabilities in Root-of-Trust Firmware and RTL with Project-Grounded Agents

**arXiv ID:** 2611.27567 | [PDF](https://arxiv.org/pdf/2611.27567v1)

**作者:** Collin MacDonald `[一作]` (William & Mary), Timothy Trippel `[通讯]` (Ricursive Intelligence)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多阶段基于LLM代理的管道，用于在根信任固件和RTL中发现、审查并重现安全相关漏洞。

**💡 创新点**

将漏洞发现与审查与项目特定的威胁模型、设计约定和原生测试环境耦合，实现项目驱动的精准检测和可执行PoC重现。

**🔧 技术方法**

采用Google ADK多代理框架、Gemini/Claude LLM模型、代码搜索工具（Universal Ctags、Tree-sitter、ripgrep）、Nix构建环境、Verilator RTL仿真，以及差分检查和最终审查机制。

**📊 数据集**

以Caliptra、OpenTitan、OpenPRoT等公开RoT项目的源代码为评估目标，包含数十万行代码，并使用已知的34个后期修复漏洞作为基准。

**📈 对比分析**

与单一代理基线和消融实验对比，Caliptra运行时固件上达到83–100%精确度，发现10/17个已知漏洞，最终验证PoC率高，成本相对可控；在其他五个目标亦发现未公开漏洞。

**⚠️ 局限性**

受限于缺乏真实硬件、仅能在模拟/主机环境重现；对硬件特定故障注入和全芯片行为的漏洞无法验证；差分规则与最终审查仍需人工确认；对多文件跨境漏洞识别有限。

---

## 452. Compositional Temporal Specifications for Interaction Trees

**arXiv ID:** 2611.27553 | [PDF](https://arxiv.org/pdf/2611.27553v1)

**作者:** Linyu Yang `[一作]` (Zhejiang University), Mingshuai Chen `[通讯]` (Zhejiang University)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种可组合的时序规范框架，用于描述和验证基于 Interaction Trees 的效果程序的时序行为。

**💡 创新点**

创新点在于引入了 iBuchi 这类浅层嵌入的 CTrees 规范语言，支持分组、并行、LTL 模态，并将其与 Dijkstra Monads Forever 结合，实现了与 Hoare 风格预后条件和未来条件的统一。

**🔧 技术方法**

采用 Coq 中的 Interaction Trees、Choice Trees、Paco 等工具实现了规范语言、满足关系及推理规则，并在 DMF 框架中编码未来条件生成器。

**📊 数据集**

本文未使用传统数据集，而是通过一系列手工构造的案例程序（如分配-释放、计数循环、混合状态与事件）进行验证。

**📈 对比分析**

通过案例验证展示了框架的表达力和组合性，报告的证明在 Coq 中已完成，未给出具体时间或资源对比，但证明规模约 5,000 行，验证成功。

**⚠️ 局限性**

主要限制是推理规则尚未构成完整的高层证明系统，循环程序的推理仍需低层 coinductive 处理，且规则对一般性迭代支持有限。

---

## 453. Leakage-Free Is Not Confounder-Free: What a Patient-Level Split Cannot Certify About a Medical Imaging Benchmark

**arXiv ID:** 2611.27585 | [PDF](https://arxiv.org/pdf/2611.27585v1)

**作者:** Ignazio Gallo `[一作]` (University of Insubria), Nicola Landro `[通讯]` (Joint Research Centre)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

本文提出并应用了一套包含五个问题的审计框架，用以系统评估医学影像数据中潜在混杂变量的强度、可读性、模型利用情况、增量预测价值以及在控制后残留的判别力。

**💡 创新点**

创新点在于将传统的漏斗评估拆分为五个可度量问题，并结合条件AUROC、可读性与利用度测试，首次在公开数据集上展示了混杂变量对性能评估的致命影响。

**🔧 技术方法**

技术上采用患者层级划分、五折分层交叉验证、ConvNeXt‑T、Swin‑T、ResNet‑50三种编码器，以及“全帧”“前视”两种影像处理方案，对模型输出与混杂变量的相关性进行统计检验。

**📊 数据集**

使用公开乳腺热成像基准 DMR‑IR（236 病例）及另一独立热成像数据集（119 病例）作为验证样本。

**📈 对比分析**

实验结果显示，获取日期的预测性能（AUROC≈0.98）远优于所有影像模型，且仅有一配置（全帧ConvNeXt‑T）在得分中显著携带该混杂信息；在控制日期后，模型判别力下降至接近随机，表明缺乏真正病理信息。

**⚠️ 局限性**

局限性包括样本量有限、跨标签同日样本极少（仅8对），导致无法在固定混杂条件下验证疾病信息；此外，仅针对热成像的评估，未覆盖其他模态或更复杂的混杂来源。

---

## 454. PRISM: Validation-Free Online Batch Selection for LLM Fine-Tuning via Preconditioned Conic Ghost Matching

**arXiv ID:** 2611.27578 | [PDF](https://arxiv.org/pdf/2611.27578v1)

**作者:** Lin Tang `[一作]` (Sichuan University), Yuxuan Wang `[通讯]` (University of Electronic Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种无验证的在线批次选择方法，利用预处理的射线子模匹配（conic facility location）从候选批中挑选最具代表性的样本进行梯度更新。

**💡 创新点**

创新点在于：①将梯度空间映射到预条件化的Kronecker‑factored度量下；②使用非负射线相似度替代直径移位的欧氏距离，避免极端梯度噪声的影响；③闭式径向投影权重自动压缩不相关或噪声样本。

**🔧 技术方法**

主要技术包括：预条件化的ghost inner product、Kronecker‑factored第二矩阵、射线子模（conic）相似度、子模贪心搜索以及闭式径向权重计算。

**📊 数据集**

在四种大型语言模型（Qwen2.5‑1.5B、Llama‑3.1‑8B、Mistral‑7B、Qwen2.5‑7B）的指令调优任务上进行实验，使用 MMLU、BBH、TyDiQA、GSM8K 四个公开基准。

**📈 对比分析**

与全批、随机、GREATS、OPUS、Filter‑then‑Weight、Euclidean‑FL、Zero‑Order 等方法对比，所提出方法在所有任务上均超过或等价于无验证基线，并在绝大多数任务中逼近甚至优于使用验证集的基线，且仅使用一半的样本即可获得更好的性能。

**⚠️ 局限性**

局限性包括：仅利用批次内部几何信息，若目标分布与训练批显著偏离且可获得验证集时效果受限；理论分析假设光滑、固定步长和预条件矩阵有界，未覆盖自适应优化器的状态动态；计算复杂度为 O(b²)，在极大候选池或单GPU环境下需使用随机贪心或限制池大小；在极端冷启动场景中需要先使用全批训练再启用选择。

---

## 455. Adapting AI Weather Foundation Models to a Changing Climate

**arXiv ID:** 2611.27605 | [PDF](https://arxiv.org/pdf/2611.27605v1)

**作者:** Xinlei Xiong `[一作]` (Hefei University of Technology), Qi Tian `[通讯]` (Huawei Inc)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

论文提出一种在线适配框架，使AI天气基础模型能够在长期部署中随气候变化自适应。

**💡 创新点**

创新点在于将流动相关误差限制在低维敏感度子空间内，并用残差记忆跟踪持久偏差，同时通过变量级门控动态调节两者贡献。

**🔧 技术方法**

使用低维敏感度子空间估计、残差记忆、变量门控和梯度更新技术，结合Aurora-1.3B模型实现在线修正。

**📊 数据集**

使用ERA5重分析数据，1959-1968年进行预训练，1969-2022年进行严格时间序列的在线评估。

**📈 对比分析**

与六种在线适配基线对比，平均在69个变量上MSE降低4.17%、MAE降低2.76%，在T2M等关键变量上提升高达18.88%。

**⚠️ 局限性**

局限在于对残差记忆半衰期和门控参数的敏感性，且对高纬度或快速变化事件的适配仍有限。

---

## 456. GAZEAIDE: Recording Gaze Context for Research on AI-Assisted Development

**arXiv ID:** 2611.27614 | [PDF](https://arxiv.org/pdf/2611.27614v1)

**作者:** Heidi Albarazi `[一作]` (JetBrains Research), Agnia Sergeyuk `[通讯]` (JetBrains Research)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

GAZEAIDE 设计并实现了一个 IntelliJ 插件与浏览器伴侣，记录开发者在阅读代码时的眼动、文本访问、版本信息，并将这些上下文与 AI 助手请求关联，支持导出、重放与研究使用。

**💡 创新点**

其创新点在于将眼动捕捉与代码区域定位相结合，形成可回放、可导出的“可用上下文包”，并在用户主动提问前提供基于观看历史的自动上下文生成，而不需手动复制代码。

**🔧 技术方法**

采用了 UnitEye/EyeMU 眼动估计、Java Chromium Embedded Framework、Node.js/浏览器客户端、JSON/CSV/HTML 导出、眼动到文本行映射与稳定目标判定、可视化反馈（高亮/淡化）等技术。

**📊 数据集**

主要使用自制的代码示例（如 GazeReviewDemo.java）和在 IntelliJ/浏览器环境中的现场实验记录，没有公开的大规模数据集。

**📈 对比分析**

论文未给出定量性能评估，仅通过示例场景和可视化演示展示功能，尚未与现有眼动辅助工具或传统手工上下文提供方式进行基准比较。

**⚠️ 局限性**

主要局限在于未评估眼动定位对不同开发者的准确性、缺乏对阅读理解或代码质量提升的量化实验，以及系统在真实工作流中的易用性与隐私安全的进一步验证。

---

## 457. S-BFM: A Behavioral Foundation Model for Humanoid Control from Sparse Spatiotemporal Goals

**arXiv ID:** 2611.27552 | [PDF](https://arxiv.org/pdf/2611.27552v1)

**作者:** Kun Yang `[一作]` (Hong Kong University of Science and Technology), Addison Lin Wang `[通讯]` (Nanyang Technological University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `f86bf285-fd08-4156-973b-6e6481af8fa0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出S‑BFM模型，实现从稀疏时空目标直接生成完整全身运动

**💡 创新点**

通过空间锚定、时序蒸馏与生成微调三阶段训练，使低层控制器能自行规划运动路径与时序

**🔧 技术方法**

基于forward‑backward行为表示、时间条件蒸馏、Kimodo生成器以及MuJoCo仿真与AGIBOT A3机器人

**📊 数据集**

使用Unitree G1 29‑DoF仿真数据，LAFAN、BONES‑SEED数据集以及AGIBOT A3真实采集

**📈 对比分析**

与SONIC、UFO、BeJEPA、PLAT等基线对比，S‑BFM在稀疏跟踪和目标到达率上分别提升约32%与>70%，终端误差从1.67m降至0.26m

**⚠️ 局限性**

训练流程分三阶段复杂，缺乏单阶段学习；未集成高层视觉语言规划；对极端外部扰动或不同机器人平台的通用性待验证

---

## 458. IsoMoE: Placement-Invariant Hybrid CPU-GPU MoE Inference with Lossless 12-bit Experts

**arXiv ID:** 2611.27604 | [PDF](https://arxiv.org/pdf/2611.27604v1)

**作者:** Hang Gu `[一作]` (University of Science and Technology of China), Xuehai Zhou `[通讯]` (University of Science and Technology of China)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 IsoMoE，一种能保持输出不随专家放置变化、同时实现更快推理的混合 CPU‑GPU Mixture‑of‑Experts 推理系统。

**💡 创新点**

通过 GPU 对 CPU 计算的逐位重放、采用 CPU 顺序的 12‑bit 无损压缩以及精确的双微批管线，实现专家放置不影响输出且提升吞吐量。

**🔧 技术方法**

异步专家换入、CPU Order 内存布局、AVX‑512 12‑bit 解压、GPU PTX 逐位重放、双微批精确管线、KTransformers 作为基础框架。

**📊 数据集**

在 Qwen3‑30B、DeepSeek‑V2‑Lite‑Chat、Mixtral‑8x7B‑Instruct 三个 MoE 模型上，用 AIME、MATH500 题库以及混合问答/代码问题集进行评估。

**📈 对比分析**

对比 KTransformers、FreeToken、llama.cpp、IsoMoE(BF16) 等基线，在不同 GPU 专家内存预算与批量尺寸下测量吞吐率，IsoMoE 平均提升 1.26×，峰值 1.54×，并保证所有设置下输出完全一致。

**⚠️ 局限性**

仅适用于支持 AVX‑512 的 CPU 与 NVIDIA RTX A6000 级 GPU；对极低批量尺寸或高 GPU 内存占用时性能提升有限；实现复杂，需要精细同步与特定压缩格式。

---

## 459. When Visually Rich Documents Meet Multi-Vector Retrieval for Region-level Retrieval-Augmented Generation

**arXiv ID:** 2611.27542 | [PDF](https://arxiv.org/pdf/2611.27542v1)

**作者:** Xu Yuan `[一作]` (Hong Kong Polytechnic University), Wenqi Fan `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于语义完整区域的 Vision‑RAG 框架，用区域级检索和选择性区域增强来提升视觉丰富文档的检索与生成。

**💡 创新点**

创新点包括：① 内容自适应区域化（CAR）将文档划分为语义完整区域；② 区域级多向量检索器，利用多模态重建预训练和检索微调；③ 区域贡献估计并根据贡献挑选关键区域组成紧凑视觉上下文进行生成。

**🔧 技术方法**

采用 VLM（Vision‑Language Model）作为编码器，结合文本重建、图像重建、late‑interaction 交互与对比学习；生成阶段使用 Qwen2.5‑VL 作为大模型；区域重排算法保持相对空间顺序。

**📊 数据集**

预训练数据：DocStruct4M；微调数据：OpenDocVQA；评测数据集包括 InfoVQA、DUDE、DocVQA、TATDQA、ViDoRe‑P（ViDoRe‑V2）。

**📈 对比分析**

与 Text‑RAG、VisRAG、VDocRAG、RegionRAG、ColQwen 等方法对比，检索方面 nDCG@5/Recall@5 提升约 3–4 分；生成方面 ANLS/F1 低于或相当于 RegionRAG，且视觉 token 数量减少 43–62%，生成延迟显著下降。

**⚠️ 局限性**

局限性：① 区域数上限为 20，难以处理布局更复杂的文档；② CAR 采用规则划分，未与检索器联合学习；③ 区域选择可能遗漏跨区域信息，导致表格类数值查询效果下降。

---

## 460. Adesktop learning factory for smart and resilient manufacturing based on digital twin and AI

**arXiv ID:** 2611.27579 | [PDF](https://arxiv.org/pdf/2611.27579v1)

**作者:** Wenxuan Hu `[一作]` (Sorbonne-Université), Zhiguo Zeng `[通讯]` (Université Paris-Saclay)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

开发了一个低成本、开源、桌面级学习工厂，集成AGV、机器人、传送带与数字孪生，实现智能、弹性制造的实验平台。

**💡 创新点**

创新点在于将深度学习、模仿学习与数字孪生技术与开源硬件结合，构建可快速迭代、易配置的学习工厂，且支持预测维护与实时监控。

**🔧 技术方法**

采用ROS、深度学习（图像识别、ACT模仿学习、LSTM故障诊断）、数字孪生可视化、WebSocket等技术。

**📊 数据集**

使用了自采集的机器人演示轨迹数据、颜色识别图像与AprilTag标签图像，未使用公开数据集。

**📈 对比分析**

通过三类典型案例（颜色排序、装配、数字孪生监测）进行功能验证，并与传统工业设备方案对比，显示了更低成本、快速部署与更高灵活性；具体性能如ACT在20轮训练后实现包装任务，预测模型可实时诊断故障。

**⚠️ 局限性**

局限在于硬件规模小、未实现全工业级的多工位协作、对大规模数据训练支持有限、模型在低功耗设备上的推理速度与精度需进一步优化。

---

## 461. Where Delay Compensation Belongs in a Decentralized Mean-Field-Game Swarm with a Control-Barrier Safety Filter

**arXiv ID:** 2611.27540 | [PDF](https://arxiv.org/pdf/2611.27540v1)

**作者:** Jan Mikołajczyk `[一作]` (University of Warsaw), Marcin Misiewicz `[通讯]` (Warsaw University of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在共享 TDMA 信道下的大型无人机群使用离线求解的均值场游戏 (MFG) 路由场与控制障碍函数 (CBF) 安全过滤器的两层架构，分析通信延迟对安全层的破坏，并通过轻量级预测器补偿延迟。

**💡 创新点**

创新点包括：① 将延迟补偿仅放置在安全过滤层，证明路由层对延迟鲁棒；② 给出延迟预算与队列长度的经验关系；③ 提出并比较常速、最小二乘、场滚动与小型学习网络四种预测器，展示其对安全性和吞吐量的提升。

**🔧 技术方法**

技术手段包括：均值场游戏离线路由场、单积分动力学、控制障碍函数 QP、TDMA 延迟模型、预测算法、仿真实验与统计分析。

**📊 数据集**

使用仿真数据，覆盖 25、50、100 只无人机，三种场景（单门、双门、柱状）和多次随机种子；测量碰撞率和到达率。

**📈 对比分析**

与未补偿、噪声仅、精确位置、扩大安全裕度等基线对比；实验显示：无补偿时帧周期 125–175 ms 内无碰撞；补偿后帧周期可提升至 250–400 ms，碰撞率降至 1–5%，吞吐略有下降（约 5–20%），预测器在安全层效果最显著。

**⚠️ 局限性**

局限性包括：仅在离线路由、单积分机体模型下验证，未考虑在线路由、硬墙、硬件噪声相关性和丢包；学习预测器训练于不同配置，泛化能力未知；仿真假设噪声独立、无丢包，实际硬件环境需进一步验证。

---

## 462. Standard-Anchored Delegation in Teachers' Use of Generative AI

**arXiv ID:** 2611.27533 | [PDF](https://arxiv.org/pdf/2611.27533v1)

**作者:** Muhammad Saleh Farooq `[一作]` (Lahore University of Management Sciences), Muhammad Jon Raza `[通讯]` (Lahore University of Management Sciences)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究了巴基斯坦旁遮普省教师在使用生成式AI时如何决定委托哪些任务，并提出“标准锚定委托”概念。

**💡 创新点**

创新点在于将教师评估标准视为AI输出可核查的前提，揭示教师在委托与核查之间的边界。

**🔧 技术方法**

采用定性主题分析与定量任务编码相结合的方法，构建教师任务委托的标准化描述。

**📊 数据集**

数据集为18名参与者（13名教师、4名管理员、1名学生）的半结构化访谈，包含教师在12项教育任务中的AI使用意向。

**📈 对比分析**

通过对任务编码的分布与教师对标准的说明进行交叉分析，揭示委托模式与标准类型的关联；未涉及算法性能评估。

**⚠️ 局限性**

局限性包括样本局限于少数城市与乡村，学生视角仅一人，访谈数据自我报告，缺乏父母访谈与实证验证。

---

## 463. Breaking the Barrier of Hard Samples: A Data-Centric Approach to Synthetic Data for Medical Tasks

**arXiv ID:** 2611.27511 | [PDF](https://arxiv.org/pdf/2611.27511v1)

**作者:** Maynara Donato de Souza `[一作]` (Universidade Federal de Pernambuco), Cleber Zanchettin `[通讯]` (Universidade Federal de Pernambuco)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出并实现了 Profile2Gen 框架，利用多阶段数据质量分析（预处理+后处理）生成并优化医疗表格数据中的合成样本，尤其关注难学样本。

**💡 创新点**

创新点在于：①将数据质量分层（easy、hard、ambiguous）嵌入生成流程；②扩展 DataIQ 为回归任务（DataIQReg），实现数据中心化的合成策略；③通过迭代剔除 hard 样本，提升合成数据的可靠性与泛化能力。

**🔧 技术方法**

核心技术包括：数据分层与样本标记（CleanLab/ DataIQReg）、七种流行的合成模型（CTGAN、TVAE、NFlow、Bayesian Network、DDPM、Tabformer、GREAT），以及 RMSE、Wasserstein 距离、Kruskal‑Wallis、Wilcoxon 等统计评估方法。

**📊 数据集**

实验使用六个公开医疗表格数据集：Parkinson、Urinary、Cholesterol、Body Fat、Plasma、Diabetes，覆盖多种预测任务和目标分布。

**📈 对比分析**

通过 TSTR（训练合成、测试真实）和数据增强评估，与传统合成、PreProcess 以及真实数据进行比较。Profile2Gen 在大多数数据集和模型上均能匹配或超越真实数据的 RMSE，明显优于传统方法，且在统计检验中显示显著优势。

**⚠️ 局限性**

局限性：1) 仍需进一步验证公平性、隐私与临床可用性；2) 对某些数据集或模型表现不一致，需要针对性调优；3) 实验规模受六个数据集限制，缺乏更大规模、多域验证；4) 合成模型训练稳定性和超参数选择仍是挑战。

---

## 464. PGS-PCQA: Perceptually Gated Synthetic Supervision for No-Reference Point Cloud Quality Assessment

**arXiv ID:** 2611.27512 | [PDF](https://arxiv.org/pdf/2611.27512v1)

**作者:** Qinyuan Fan `[一作]` (Technische Universität Berlin), Clemens Gühmann `[通讯]` (Technische Universität Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文针对无参考点云质量评估模型在不同数据集间泛化差的现象，提出了感知门控合成监督（PGS-PCQA）来提升跨数据集性能。

**💡 创新点**

核心创新在于利用仅来源于训练集的合成失真库，以同一参考点云内不同失真级别的排序作为无目标域监督，并通过感知门控动态调节排序强度。

**🔧 技术方法**

技术实现包括基于CLIP-PCQA的视觉编码、KPA几何分支、六视图渲染与VGG19感知距离门控、RankNet式对比损失以及前景加权多区域测试。

**📊 数据集**

实验使用了四大公开点云质量数据集：SJTU-PCQA、WPC、BASICS和LS-PCQA Part I。

**📈 对比分析**

与DSP-PCQA、PMP-PCQA、CLIP-PCQA等多种基线比较，PGS-PCQA在跨数据集评估中取得最高SRCC（WPC 0.709、BASICS 0.751），在内数据集上亦保持近0.95的SRCC水平。

**⚠️ 局限性**

主要局限包括：对合成库未覆盖的失真类型适应仍有限、跨对象尺度校准不足、以及前景加权多区域测试导致的推理成本上升。

---

## 465. Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events

**arXiv ID:** 2611.27509 | [PDF](https://arxiv.org/pdf/2611.27509v1)

**作者:** Sadiq O. Macaulay `[一作]` (University of Surrey), Hadfield Simon `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种自监督的连续时间深度估计和相机运动恢复框架，直接为每个事件预测其在自身时间戳下的深度，并通过运动补偿事件对齐进行训练。

**💡 创新点**

创新点包括：①为每个事件学习连续时间深度并与相机运动同步；②利用双锚点运动补偿对齐作为自监督信号；③设计局部-全局事件状态编码器（Mamba+多尺度局部聚合）以及可微遮挡渲染器，将事件深度转换为目标时间稠密深度图。

**🔧 技术方法**

使用的技术包括：选择性状态空间模型（Mamba）、多尺度局部聚合、双锚点运动补偿损失（锐度、时间散布、越界、平滑项）、可微遮挡渲染（递归回卷、recency‑anchored log‑depth 置信度）、自监督优化以及相机运动头的预测。

**📊 数据集**

实验采用 MVSEC 事件相机数据集（包含同步事件、图像、深度与姿态），在多段夜间和日间序列上进行评估。

**📈 对比分析**

与现有自监督事件深度方法（EMoDepth 等）对比，本文方法在 20 m 切点上取得最小平均绝对误差，夜间表现尤为突出，且在相机旋转误差上比 EMoDepth 提升约 35%。

**⚠️ 局限性**

局限性：依赖静态场景假设，快速运动或显著动态物体会导致对齐失效；需要同步深度/相机标定，训练与推理对时间窗口长度和硬件有一定要求；在极端光照或遮挡下的鲁棒性仍待进一步验证。

---

## 466. PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks

**arXiv ID:** 2611.27521 | [PDF](https://arxiv.org/pdf/2611.27521v1)

**作者:** Silviu-Florin Gheorghe `[一作]` (University of Bucharest), Radu Tudor Ionescu `[通讯]` (University of Bucharest)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了PaMuk-Bench，一个并行多语言多任务视觉-语言基准，包含10,200条英文提示及其七种语言的翻译，并提供对应自然图像、生成图像和多种评价标注，涵盖文本到图像生成、图像说明、内容检索、生成难度预测和检索难度预测五大任务。

**💡 创新点**

创新点：① 统一的并行多语言数据集，让不同语言间可直接对比；② 半自动翻译+投票机制显著减少人工翻译成本；③ 在同一数据分割下系统性评估多种VLM在零拷贝与微调两种场景下的跨语言表现；④ 通过多任务设置展示语言差距的普遍性与可通过微调缓解的可能性。

**🔧 技术方法**

技术手段：使用MT工具（DeepL、Google Translate、Claude sonnet）进行初始翻译；投票筛选与人工校对相结合；LoRA微调、AdamW训练；多种评估指标（HPS、CLIPScore、BERTScore、P@10等）评估任务表现。

**📊 数据集**

数据集：基于PQPP、MS COCO、DrawBench，生成图像使用SDXL、GLIDE；提供自然图像、生成图像以及检索相关性标签；共7种语言（英语、阿拉伯语、丹麦语、法语、印地语、意大利语、罗马尼亚语），总计71,400条提示（10,200组）。

**📈 对比分析**

比较方法：对比零拷贝（仅使用英文模型）与多语言平均零拷贝以及在本基准上微调后的平均结果。结果显示：① 零拷贝模型在非英语时性能显著下降；② 微调后大部分模型几乎完全恢复至与英语相近的水平，尤其是对阿拉伯语和印地语等低资源语言提升显著。总体性能在生成、说明和检索等任务上都有可观提升。

**⚠️ 局限性**

局限性：① 语言覆盖仅限七种，未覆盖更多低资源或非印欧语系语言；② 任务集合虽多样但仍局限于五大类，缺少更细粒度或跨模态的挑战；③ 评估指标主要基于已有公开评价，缺乏针对多语言细节的专门指标；④ 微调过程需要依赖大规模算力，实际部署仍受限。

---

## 467. Potential emergence in LLM agents: a statistical physics perspective

**arXiv ID:** 2611.27618 | [PDF](https://arxiv.org/pdf/2611.27618v1)

**作者:** Zhuo-Yang Song `[一作]`, Hua Xing Zhu `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究通过分区函数理论探究LLM代理后训练过程中状态转移的潜能结构，并在安全测试选择与物理推理审核两类简化任务上进行定量验证。

**💡 创新点**

创新点在于将KL正则化与分区函数结合，提出潜能函数描述状态空间演化，并通过实验证实潜能结构随后训练而增强的理论预测。

**🔧 技术方法**

采用RLHF（PPO/REINFORCE+LoRA）对Qwen系列模型进行后训练，利用分区函数、潜能估计、奖励-熵理论关系和Jeffreys散度校准等技术进行测量与分析。

**📊 数据集**

使用自定义有限状态空间任务（T1共8个状态，T2共12个状态）及其奖励表，收集每轮训练中模型生成的回复和状态转移计数作为实验数据。

**📈 对比分析**

通过计算潜能差异与转移概率对数比的相关系数（order parameter r）、平均奖励和状态熵进行对比，结果显示r显著上升、奖励提升、熵下降，理论曲线与观测高度吻合。

**⚠️ 局限性**

局限性包括仅在预定义的有限状态空间内验证，缺乏对更复杂或连续状态的泛化；假设训练达到KL最优并忽略非平衡动态；奖励结构和参考分布对结果敏感，现实任务中难以保证。

---

## 468. Do DeepONet Trunk-Basis Preconditioners Need Training?

**arXiv ID:** 2611.27582 | [PDF](https://arxiv.org/pdf/2611.27582v1)

**作者:** Jaemin Oh `[一作]` (Brown University), George Em Karniadakis `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了DeepONet的trunk-basis（TB）预条件器在求解SPD线性系统时的有效性，探讨了是否需要训练、训练目标及粗空间构造方式。

**💡 创新点**

创新点在于：① 在理论上证明一维中随机初始化的TB空间能近似低阶多项式空间；② 通过数值实验验证未训练的TB空间与分段多项式粗空间在低对比度问题上迭代次数相近；③ 发现能量损失训练比常规ℓ²损失更能捕捉高对比度问题中的低能量模式；④ 在高对比度弹性问题中结合刚体模式提升性能。

**🔧 技术方法**

使用技术包括：DeepONet架构（branch+trunk），随机初始化与训练（ℓ²或能量损失），两层Schwarz预条件器（局部平滑器+粗空间校正），以及局部截断的trunk函数构造粗空间；实验中用CG求解器和多重域分解。

**📊 数据集**

实验数据集主要是：① 2D Poisson方程和3D线性弹性方程的均匀系数实例；② 高对比度Poisson（通道结构）和高对比度弹性（拓扑优化梁）实例；数据生成通过随机高斯场或梯度场采样，训练样本量在数千至数十万之间。

**📈 对比分析**

比较方法是：对相同粗空间维度下的TB空间（未训练、ℓ²训练、能量训练）与分段多项式粗空间及刚体模式增强的TB空间，在CG迭代次数上进行对比。结果显示：在低对比度问题未训练TB与多项式性能相当，训练并未提升；在高对比度问题能量训练显著降低迭代次数；当结合刚体模式时，TB能进一步逼近或超越多项式基准。

**⚠️ 局限性**

局限性包括：① 仅针对SPD系统，未讨论不定或时域问题；② 对训练数据分布与损失函数的理论连接尚未充分；③ 实验规模虽大，但对全流程（数据生成、训练、预处理、求解）总成本评估不足；④ 只在均匀域分解上验证，缺乏对不同域分解策略的分析。

---

## 469. PromptTTSD: Controllable Text to Spoken Dialogue Synthesis with Open-Ended Text Descriptions

**arXiv ID:** 2611.27601 | [PDF](https://arxiv.org/pdf/2611.27601v1)

**作者:** Sulong Yang `[一作]` (Beijing University of Posts and Telecommunications), Shuo Chen `[通讯]` (National Key Laboratory of General Artificial Intelligence)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一个可扩展的多说话人对话语音数据集 PromptDialog，并基于此数据集构建了 PromptTTSD——一种支持自然语言风格控制的语音合成模型；

**💡 创新点**

创新点在于①利用多声优有声书与原书文本的对齐，自动生成带自然语言风格描述的语音-文本-描述三元组；②提出双阶段上下文感知对齐算法，显著提升了句子匹配率；③在已有对话合成模型上通过插入描述提示并加入辅助文本损失，实现了开箱即用的开端到终端风格控制；

**🔧 技术方法**

技术包括：基于 Whisper/Paraformer 的 ASR；两阶段对齐（anchor‑based 与 context‑based）+ Levenshtein 过滤；LLM（如 MiMo‑v2.5、Qwen3.8‑27B）完成说话人归属与风格描述抽取；PromptTTSD 采用 SoulX‑Podcast 预训练模型的文本‑语音交错表示，并在此基础上插入描述提示与文本损失；

**📊 数据集**

使用了 PromptDialog 数据集，包含 8,383 小时中文对话语音、120 小时英文对话语音，约 79% 句子带开放式风格描述；此外还引用了公开数据集 LibriHeavy、SpeechCraft、SpeechCraft 等做对比；

**📈 对比分析**

与 CosyVoice3、MiMo‑v2.5‑TTS、Eleven‑v3 等开源与专有系统相比，PromptTTSD 在 Q‑MOS、UTMOS、Style‑CMOS 上均居前；在对话合成上，PromptTTSD（不加描述）获得最低 CER，且在有描述条件下的 Contextual Appropriateness MOS 提升至 3.52（比无描述提升 0.46）；

**⚠️ 局限性**

局限性包括：①数据集主要为中文，英文部分仍较小；②开放式风格描述的质量和一致性仍依赖 LLM 抽取，可能存在偏差；③对长对话的实时控制与多轮情境追踪仍未充分验证；④模型在某些非口语化表达（如笑声、叹息）时的 CER 较高。

---

## 470. Altair-Nidhogg: Which Model-Merging Theories Hold for a One-Pass Decision Encoder? Soups, Task Arithmetic, TIES and DARE Against Joint Training and Prediction Ensembling

**arXiv ID:** 2611.27550 | [PDF](https://arxiv.org/pdf/2611.27550v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 471. Minimax-Optimal Small-Loss Regret in Online Convex Optimization and Applications to Contextual Search and Online Inverse Optimization

**arXiv ID:** 2611.27572 | [PDF](https://arxiv.org/pdf/2611.27572v1)

**作者:** Shinsaku Sakaue `[一作]` `[通讯]` (CyberAgent), Shinsaku Sakaue (CyberAgent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了一种仅使用一次子梯度的在线凸优化（OCO）算法，在无先验信息的情况下实现了以最优小损失为目标的再调节误差上界；

**💡 创新点**

创新点在于：①提出了域依赖的最优小损失界，兼顾所有\ell_p球和其他凸域；②将Gupta等人（2026）提出的度量更新撤销（metric‑update revocation）技术推广至一般OCO，并在Follow‑The‑Regularized‑Leader框架下构造了可撤销的曲率项；③通过混合多基学习者实现对未知损失\Delta_T^⋆的自适应；④在情境搜索与在线逆优化等实际任务中取得了理论与算法上的突破。

**🔧 技术方法**

主要技术包括：FTRL + 可撤销的二阶正则化；子梯度修正与投影；混合学习者与递归潜能函数；使用子指数均值宽度（sub‑exponential mean width）刻画域的几何属性；分析中结合了线性化损失、绝对损失与负向部分之和的关系。

**📊 数据集**

该工作为纯理论论文，未使用具体数据集，而是在情境搜索与在线逆优化中通过理论证明展示了算法性能；

**📈 对比分析**

与传统小损失算法（如基于平滑性或自我绑定条件的投影梯度法）相比，该方法在无平滑性假设、无损失值信息且时间无先验的条件下实现了\mathcal O(GW+\sqrt{GW\Delta_T^⋆})的上界，在\ell_p球上达到最优依赖；在情境搜索中得到\Theta(\sqrt d/κ^2)的期望绝对误差；在在线逆优化中得到\mathcal O(GW+\sqrt{GW D_T})的相对回报。

**⚠️ 局限性**

局限性包括：需要对子梯度范数G的上界；算法在大维度下的计算成本为\mathcal O(T d^2\log T)；对参数如\epsilon或\tau 的选择仍需经验或预设；对非凸或高阶损失的适用性尚未探讨。

---

## 472. WEIR: Watershed Exploration and Drift-Aware Prefix Shrinkage for Multi-Turn Agent Reinforcement Learning

**arXiv ID:** 2611.27570 | [PDF](https://arxiv.org/pdf/2611.27570v1)

**作者:** Pixel Nomand `[一作]` (University of Wisconsin--Madison), Sofia Reyes `[通讯]` (University of Wisconsin--Madison)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的基于Actor‑Critic的RL框架（称为Watershed Exploration and Intrinsic-Drift Shrinkage，简称WEDS），专为多轮LLM代理在稀疏终端奖励环境下的信用分配与样本效率设计。

**💡 创新点**

核心创新包括：① 对非马尔可夫POMDP的Doob转阶方差守恒定理，用以证明奖励不确定性集中在少数关键转折点；② 采用最小化最大均方误差的漂移感知收缩目标（minimax drift‑aware shrinkage）来平滑前缀统计与单轨返回值；③ 构建带指数衰减计数的跨迭代规范化前缀树，兼顾历史经验与最新策略；④ 基于值跳跃与未决不确定性的watershed分支策略，集中探索决策关口；⑤ 通过Neyman分层为任务分配预算，实现任务级样本均衡。

**🔧 技术方法**

技术实现涵盖：Actor‑Critic 与 GAE、长度归一化重要性采样；前缀聚合与最小化最大MSE收缩；指数历史衰减（γ_hist）、漂移比κ；Neyman分层任务分配与watershed分支评分；基于Doob马尔可夫方差守恒的理论证明。

**📊 数据集**

实验使用的公开数据集包括：多轮问答（HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle），机器人任务ALFWorld，网页购物任务WebShop；实验模型基于Qwen2.5‑3B/7B‑Instruct。

**📈 对比分析**

与GRPO、GiGPO、Tree‑GRPO、AT^2PO、PGPO、PPO、SPPO、Turn‑PPO、CAPO等基线及Search‑R1、StepSearch做对比。WEDS在所有QA基准上平均EM提升至32.18%（3B）/37.85%（7B），比CAPO高3.5–4.0个百分点；在ALFWorld/ WebShop的成功率也分别提升至84.83%/78.47%（3B）和87.81%/81.90%（7B），超越CAPO 3–4个百分点；同时生成的rollout token数比独立链少约28–29%（0.71–0.72×）。

**⚠️ 局限性**

局限性：需能恢复中间环境状态和交互历史以实现分支；前缀树的归一化文本匹配与E5相似度在高度连续或参数化JSON动作空间中效果不佳；漂移比κ被假设为全局常数，无法自适应不同任务或学习速率变化。

---

## 473. Exploring the Potential of Ultra-Low-Bit Inference on LLM Accelerators

**arXiv ID:** 2611.27561 | [PDF](https://arxiv.org/pdf/2611.27561v1)

**作者:** Hao Wang `[一作]` (National University of Defense Technology), Yaohua Wang `[通讯]` (National University of Defense Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了VQACC，一种在现有基于流水阵列的LLM加速器上实现向量量化解压的架构改进。

**💡 创新点**

创新点在于将权重缓冲区的存储、端口和寻址重用为代码书查找，并通过索引地址生成器和位移布局单元将解压过程无缝集成到普通operand数据流，几乎不增加面积。

**🔧 技术方法**

采用代码书索引映射、索引地址生成器、端口多路复用、轻量级位移布局单元和代码书驻留数据流等技术。

**📊 数据集**

使用Llama-2-7B、Qwen3-8B和Mistral-7B等大型语言模型，结合不同VQ配置（AQLM-1、GPTVQ-2、CQ-2/4）。

**📈 对比分析**

通过与FP16基线、VPU Dequant、EVA-OL32和Vortex-IL四个设计比较，在多批量解码和预填充场景下，VQACC平均可获得2.63×和1.10×的速度提升，同时面积增幅仅0.17%。

**⚠️ 局限性**

局限性包括对代码书分区的依赖，仍需离线聚类生成代码书；对极大批量或极长上下文时受限于SA计算单元尺寸，且不适用于所有VQ算法的键/值缓存布局。

---

## 474. Alliance Rupture and Repair in Simulating Cognitive Behavioral Therapy Dialogues

**arXiv ID:** 2611.27544 | [PDF](https://arxiv.org/pdf/2611.27544v1)

**作者:** Jian Chai `[一作]` (National University of Singapore), Yi-Chieh Lee `[通讯]` (National University of Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合工作联盟破裂、修复与再规划的CBT模拟框架。

**💡 创新点**

首次在多回合CBT对话中显式建模联盟破裂与修复循环，并通过关系性触发器动态调整治疗计划。

**🔧 技术方法**

使用大型语言模型（LLM）结合确定性规则、评估器与修复器，并构建关联性触发记忆和隐藏联盟状态。

**📊 数据集**

基于CACTUS的客户端配置生成数据集，评估使用真实CBT（RealCBT）及五个公开合成对话数据集（CACTUS、MAGneT、MIRROR、Graph2Counsel、SimPsyDial）。

**📈 对比分析**

通过情感动态、策略流、CTRS、FIS‑T、破裂检测与修复率等自动与专家指标比较，表现为在策略流与情感轨迹上最接近真实CBT，在CTRS/FIS‑T得分最高，破裂检测率达96.6%，计划完成率97.8%。

**⚠️ 局限性**

仅限单会话、客户端形态覆盖有限、破裂检测率与真实会话差异、LLM偏倚与专家评估样本不足。

---

## 475. A Cognitive Agentic AI Framework for Interactive Process Analysis and Optimization

**arXiv ID:** 2611.27725 | [PDF](https://arxiv.org/pdf/2611.27725v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99`

---

## 476. PhiJEPA: Denoising Latent Predictions for Visual Representation Learning

**arXiv ID:** 2611.27516 | [PDF](https://arxiv.org/pdf/2611.27516v1)

**作者:** Makoto Yamada `[一作]` (Okinawa Institute of Science and Technology), Yao-Hung Hubert Tsai `[通讯]` (Okinawa Institute of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了PhiJEPA，一种两阶段隐藏Transformer token预测架构，结合PhiNet的预测理念与Masked Transformer，训练时使用自相互独立的光度腐蚀图像作为目标，并通过第二阶段将预测映射为清洁teacher特征。

**💡 创新点**

创新点在于：①引入两阶段预测（先预测受独立腐蚀影响的token，再映射为清洁token），显著提升密集特征质量；②在预训练时不需要手工设计对比或EMA教师，使用自监督的光度腐蚀与EMA教师；③保留并复用预测器作为世界模型的token补全模块，可直接提升在遮挡+模糊环境下的规划成功率。

**🔧 技术方法**

使用技术包括：ViT-B/16 Encoder、EMA教师、单个Transformer块（CA3）预测受腐蚀影响的隐藏token、token级MLP（CA1）映射至清洁目标、光度腐蚀数据增强（色彩抖动、灰度、模糊、光照、噪声）、线性与注意力型 probe、CEM规划、DINO‑WM框架。

**📊 数据集**

主要数据集：ImageNet‑1K 用于预训练；PASCAL VOC 与 ADE20K 用于线性分割评估；ImageNet‑1K 用于分类 probe；PushT 机器人仿真环境用于世界模型与规划实验。

**📈 对比分析**

与Simple I‑JEPÀ、I‑JEPÀ、DSeq‑JEPÀ、iBOT、DINOv2、LeJEPA等方法对比，PhiJEPA 在VOC分割 mIoU 69.3%（比I‑JEPÀ高 3.1%），在A‑DE20K 31.2%（比I‑JEPÀ高 1.5%），线性probe Top‑1 75.8%；预训练 FLOPs 仅为 I‑JEPÀ 的 75% ；在 PushT 规划中，未补全时 33% 成功率，使用PhiJEPA 预测器补全后提升至 65%，显著高于 Simple I‑JEPÀ 25% 与 iBOT 40%。

**⚠️ 局限性**

局限性包括：①实验仅在单一仿真环境 PushT 上验证，缺乏多任务/多环境验证；②遮挡与模糊是人工合成，未评估真实机器人视觉噪声；③仅在单一预训练随机种子与单规模 ViT-B/16 上测试，未充分探索规模与多种超参的鲁棒性；④预测器在未知遮挡位置时需要先知晓遮挡区域，实际应用需进一步完善。

---

## 477. Ego4Gen: Generating Geometry-Grounded Motion Tokens as a World-Action Interface

**arXiv ID:** 2611.27513 | [PDF](https://arxiv.org/pdf/2611.27513v1)

**作者:** Yang Liu `[一作]` (Harbin Institute of Technology), Weiyang Lin `[通讯]` (Harbin Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种名为Ego4Gen的世界模型，能够在给定单帧第一人称RGB图像和文本提示的条件下，联合生成未来的RGB视频、点云图以及三维场景流，并通过FlowInverse将这些几何-运动潜变量映射为机器人控制指令，支持闭环操控；

**💡 创新点**

创新点在于：①提出将几何-运动潜变量作为世界-动作接口，显著降低视觉无关噪声影响；②在Cosmos3-Nano框架上引入模态路由LoRA与非对称注意力，实现多模态的协同生成；③设计轻量级逆动力学网络FlowInverse，利用两阶段训练实现从潜变量到动作的映射；④通过在无动作标签的视频上进行世界模型适配，提升泛化性能。

**🔧 技术方法**

技术包括：预训练的Cosmos3-Nano Mixture-of-Transformers，模态路由LoRA，非对称注意力，MotionCrafter的4D VAE潜变量，三维场景流推断，逆动力学网络FlowInverse，以及多阶段训练策略。

**📊 数据集**

使用的数据集有EgoDex、AgiBot、RoboTwin 2.0的清洁任务轨迹、TACO（egocentric双手操控），以及MotionCrafter用于生成几何-运动伪标签。

**📈 对比分析**

通过与Wan 2.2、Cosmos3-Nano、RynnWorld-4D、4DNeX、TesserAct等模型在RGB、深度、点云、运动四个维度上进行定量评估；在RoboTwin 2.0任务上与DP、TurboVLA、starVLA、Fast-WAM、π_0等基线进行闭环成功率对比。结果显示Ego4Gen在RGB/点云/运动质量上与MotionCrafter相当，在闭环任务上尤其在随机场景下显著提升成功率（从基线的15%提升至26%）。

**⚠️ 局限性**

局限性包括：①对目标任务的FlowInverse需要单独微调，迁移性能受限；②在高度动态或大范围运动场景下的场景流预测仍不够精确；③模型对不同相机参数的依赖需要进一步泛化；④在缺少动作标签的数据适配过程中，生成潜变量与FlowInverse的对齐可能导致性能下降。

---

## 478. Tasklets: A Unit-Test Benchmark for Language Model Architectures

**arXiv ID:** 2611.27607 | [PDF](https://arxiv.org/pdf/2611.27607v1)

**作者:** Zayd Muhammad Kawakibi Zuhri `[一作]` (Mila), Alham Fikri Aji `[通讯]` (MBZUAI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出了Tasklets——一种面向语言模型序列混合层的单元测试基准，覆盖七大能力类别共16个合成任务；

**💡 创新点**

创新点在于将此前零散的合成任务统一标准化、归类并公开实现，提供可复现、低成本的评测框架；

**🔧 技术方法**

采用多种序列混合层技术（softmax注意力、Delta‑rule线性RNN、对角线线性RNN、非线性RNN、混合架构）并在同一训练设置下对它们进行统一评估；

**📊 数据集**

使用基于随机生成的100k训练样本与3k测试样本的任务数据集，词表大小292，涵盖记忆、复制、计数、堆栈、形式语言等；

**📈 对比分析**

通过在每个任务上计算精确匹配准确率，比较24种架构在相同超参、GPU时间下的表现，结果显示softmax注意力和部分Delta‑rule线性RNN在记忆与复制任务上优于其他线性RNN；而所有线性RNN在模组加法和排列合成等状态追踪任务上表现极差，非线性RNN显著领先；

**⚠️ 局限性**

局限性包括仅评估了小模型（15-20万参数）且任务长度有限，缺乏对大规模模型的验证，且不涉及自回归推理或跨模态应用，未来需扩展到更大规模、更复杂的任务。

---

## 479. Beyond Math, Code, and Logic: Do Reasoning Models Transfer to Scientific Argumentation?

**arXiv ID:** 2611.27722 | [PDF](https://arxiv.org/pdf/2611.27722v1)

**作者:** Buse Sibel Korkmaz `[一作]` (Amazon AGI), Antonio del Rio Chanona `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型在七种科学论证任务（命题检测、证据判断、立场识别、驳斥生成、元评综述、问答组合等）的零样本与训练性能进行系统评估，并提出一种基于结构化推理监督与强化学习的训练框架。

**💡 创新点**

创新点在于通过链式思考知识蒸馏生成结构化推理链，再用多任务监督和 Group Relative Policy Optimization（GRPO）实现“论证意识”训练，使得规模相对较小的模型在科学论证方面显著超越当前商用大模型。

**🔧 技术方法**

采用的技术包括：链式思考（CoT）知识蒸馏、全参数多任务监督微调（SFT）、基于相对策略优化的强化学习（GRPO），以及精细化的奖励设计（答案准确、逻辑一致、证据归因、格式合规）。

**📊 数据集**

使用的公开数据集有：NSF‑SciFy（命题检测）、SciFact（证据判断与分类）、DISAPERE（立场检测）、ARIES（驳斥生成）、PRRCA（元评综述）、QASA（问答组合）等七类与科学论证相关的数据集。

**📈 对比分析**

与 Llama3.2（1B/3B/8B）、OpenAI o3、Gemini 2.5 Flash 等基线对比，训练后的 3B 模型在大多数任务（如命题检测、证据判断、立场识别）达到了或超过了商用大模型的表现；在生成任务中，驳斥生成的 BERTScore 距离商用模型仅差 0.02，但元评综述与问答任务仍显示出差距。

**⚠️ 局限性**

主要限制包括：评估仅覆盖科学文稿与同行评审文本，未涉及更真实的对话与多方互动；自动指标可能忽视论证细节与质量；缺少人类评估与更广泛的交互场景验证。

---

## 480. MetaRigor: From Agentic Automation to Verifiable End-to-End Meta-Analysis Through Scientific Decision Boundaries

**arXiv ID:** 2611.27713 | [PDF](https://arxiv.org/pdf/2611.27713v1)

**作者:** Yifeng Xie `[一作]` (Hangzhou Dianzi University), Yifei Chen `[通讯]` (Tsinghua University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出MetaRigor框架，实现从检索到稿件生成的全流程可验证系统；

**💡 创新点**

通过显式的科学决策边界（Task Contracts与Code Logic）和Study×Target结果标识，保持跨阶段结果身份与源关联；

**🔧 技术方法**

结合LLM（如GPT-5.6 Luna、DeepSeek）与程序化规则，采用源绑定与任务合同；

**📊 数据集**

使用102篇公开系统综述、717项研究纳入，涵盖五个评估阶段的真实数据；

**📈 对比分析**

与四个基线（Codex CLI、OpenHands CLI、Manalyzer、meta‑pipe）对比，MetaRigor在每阶段指标均优于基线，终端F1达77.01%（比最强基线高9.78个百分点）；

**⚠️ 局限性**

主要局限在于对极端案例的验证不足、模型配置对结果影响较大，以及需要进一步扩展到更广泛的主题和多语言环境。

---

## 481. Solving the pre-image problem in kernel PCA: The reproducing property is all you need

**arXiv ID:** 2611.27718 | [PDF](https://arxiv.org/pdf/2611.27718v1)

**作者:** Joel-Pascal Ntwali N'konzi `[一作]` (University of Edinburgh), Stefan Klus `[通讯]` (Heriot-Watt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对核主成分分析（KPCA）中的预影像（pre-image）问题，本文基于核空间的再现性质推导了一种新的重构映射，能够在投影回输入空间时保持最小二乘最优，并在线性核下与传统PCA的重构一致。

**💡 创新点**

创新点在于：1）利用再现性质直接给出KPCA的解析预影像映射，避免了昂贵的非线性优化或监督学习；2）证明该映射在保持线性核一致性时的最优性；3）将该思路推广到随机傅里叶特征（RFF）逼近的KPCA，并提供相应的“RFF再现性”概念。

**🔧 技术方法**

技术方法包括：核主成分分析、随机傅里叶特征逼近、再现性质证明、最小二乘最优性分析，以及对比实验中的PSNR/SSIM/MSE指标。

**📊 数据集**

实验数据集涵盖：合成两圆数据、USPS手写数字、MNIST、CIFAR-10，用于评估图像去噪与降维重构。

**📈 对比分析**

与现有方法（B.W.S., H.R., M. et al., T.R., 核岭回归）进行对比，结果显示本文方法在大多数核和噪声水平下与最优方法相当或更优，并且在运行时间与内存占用上显著优于对手，尤其在大规模数据（MNIST、CIFAR-10）场景下实现了数十倍的加速。

**⚠️ 局限性**

局限性：仅适用于向量型数据；对非向量结构（如字符串、图）以及分布预影像问题的扩展尚未完成；对Nyström近似的适配以及RFF方法的收敛性分析仍待进一步研究。

---

## 482. Coding Length and Optimal Power Allocation in Correlated Block Fading

**arXiv ID:** 2611.27694 | [PDF](https://arxiv.org/pdf/2611.27694v1)

**作者:** Haoran Xu `[一作]` (Soochow University), Huazhong Zhao `[通讯]` (China Telecom Corporation Limited)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在相关块衰落信道中，给定平均信噪比和能量约束，如何分配功率以实现指数下降的误码率并最大化通信速率。

**💡 创新点**

提出了精确的长块极限率表达式、有限块长度的修正项，并给出了功率分配的可行性与最优性判据，揭示了相关性对功率分配和速率损失的影响。

**🔧 技术方法**

使用大偏差理论、随机编码与球面包络界、Lagrange 乘子、矩阵分析、Gaussian 通道信息率与散度等工具。

**📊 数据集**

基于理论模型，主要考虑理想的多维高斯衰落矩阵，没有实际实验数据集。

**📈 对比分析**

通过与传统等功率分配的对比以及数值模拟验证，表明在长块极限下等功率可近似最优，但在短块或高相关性时显著受限，性能提升可达数个百分点。

**⚠️ 局限性**

局限在于对极限块长度的假设、对衰落协方差的严格结构假设、以及对实际多天线系统中非理想因素（如硬件非线性、反馈延迟）的忽略。

---

## 483. Spatiotemporal Graph Neural Networks for Fault Detection and Structural Learning in Chemical Processes: Use Case on the Tennessee Eastman Process

**arXiv ID:** 2611.27698 | [PDF](https://arxiv.org/pdf/2611.27698v1)

**作者:** Rayane Ammar Khodja `[一作]` (IFP Energies Nouvelles), Benoît Iung `[通讯]` (Université de Lorraine)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一种时空图神经网络框架，用于在动态工况下对化工过程进行故障检测。

**💡 创新点**

创新点在于将故障检测任务拆解为工况子空间，并结合重构误差与DeepSVDD一类损失，同时系统评估GATv2注意力的物理可解释性。

**🔧 技术方法**

使用了改造的MTAD-GAT结构、GATv2注意力机制、1D卷积预处理、GRU时序编码以及DeepSVDD一类损失。

**📊 数据集**

实验数据来自 Tennessee Eastman Process（TEP）仿真集，包含 500 条无故障训练样本和 20 种不同故障类型的测试样本。

**📈 对比分析**

与其他重构或图模型基线对比，平均 ROC‑AUC≈0.77、PR‑AUC≈0.80；但注意力矩阵对 22 条物理耦合关系的 Hit‑Rate@k 远低于 Pearson 相关性，显示重构驱动的注意力缺乏可解释性。

**⚠️ 局限性**

局限包括仅在单一工况和仿真数据上验证；未结合真实工业数据和动态工况；且仅评估了 GATv2，未探讨其他 GNN 变体。

---

## 484. Learning in Symbiosis: Heterogeneous Federated Learning with ANNs and SNNs

**arXiv ID:** 2611.27686 | [PDF](https://arxiv.org/pdf/2611.27686v1)

**作者:** Manh V. Nguyen `[一作]` (Kennesaw State University), Shaoen Wu `[通讯]` (Kennesaw State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并实现了混合神经形态联邦学习框架(HNFL)，使传统ANN和SNN设备能够在同一联邦学习环境中协同训练。

**💡 创新点**

核心创新是通过参数分区解决ANN与SNN在归一化层的参数不匹配问题，分别聚合共享特征权重、ANN专属归一化参数和SNN专属归一化参数，从而实现两种网络类型的无缝协作。

**🔧 技术方法**

使用FedAvg联邦优化算法，结合BatchNorm和BatchNorm‑Through‑Time（BNTT）归一化，采用LIF神经元模型以及跨时步输入编码（重复输入与Poisson编码）。

**📊 数据集**

在CIFAR‑10数据集上，用VGG9/VGG9‑SNN（SVGG9）和ResNet/CIFAR‑ResNet（SResNet）两组网络进行实验。

**📈 对比分析**

对比单一ANN或单一SNN联邦学习，HNFL在混合客户端设置下实现了：ANN准确率提升至最高+16.85%，SNN准确率提升至+8.02%；并且在相同准确率目标下，HNFL所需的全局聚合轮次比纯SNN联邦学习少约30–60%，同时SNN模型的平均发射率降低12.4%–22.4%。

**⚠️ 局限性**

实验未覆盖不同设备间高度非IID的数据分布，且对实时能源消耗与硬件部署的进一步评估仍待开展。

---

## 485. FOREST: Coordinating Tree-based HBF, RRAM, and HBM for Efficient Speculative LLM Inference

**arXiv ID:** 2611.27701 | [PDF](https://arxiv.org/pdf/2611.27701v1)

**作者:** Hao Ding `[一作]` (Unigroup Guoxin Microelectronics Co., Ltd.), Yimao Cai `[通讯]` (Peking University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 FOREST 的异构内存协同架构，专门针对大语言模型的投机式推理（speculative decoding）进行优化。

**💡 创新点**

创新点在于：① 将 3D 堆叠 RRAM 与 HBF、HBM 三种内存技术配合，分别承担低延迟的 DLM 编码、宽带高容量的 TLM 验证和 KV 缓存；② 采用跨树数值协同设计，将 Q/K/V 的 block‑scaled 预处理迁移到 HBF/ RRAM 根板，实现 BFP 关注计算；③ 引入阶段感知的执行与资源配额，动态平衡 TLM 预填、DLM 编码与注意力计算的带宽与算力需求。

**🔧 技术方法**

核心技术包括 HBF、3D 堆叠 RRAM、HBM、SRAM‑DCIM 根板、block‑floating‑point（BFP）数值格式、局部 softmax 与层级全局归约、以及相位感知的投机式解码调度。

**📊 数据集**

使用了多种 LLM 及其对应的推理数据集：Qwen2.5‑32B、Qwen2.5‑72B、GLM‑4.5‑Air（106B）和 Llama 3.1‑405B，均在 8K‑token 的四类文本（通用、推理、代码、对话）上进行推理实验。

**📈 对比分析**

与 NVIDIA A800 GPU（80 GB HBM）基准进行对比，FOREST 在六种目标‑草稿组合下取得：最高 5.5× 的端到端速度提升、11.46× 的解码吞吐量提升、16.5× 的能效提升，以及在总拥有成本模型下最高 6.73× 的 token/$ 成本效率。

**⚠️ 局限性**

主要局限包括：对超大模型（如 405B）的预填阶段仍受算力瓶颈限制；需要较高的 HBF 带宽与 T‑root 计算容量；异构内存集成的技术成熟度与成本敏感性；以及在极长上下文或高 draft 长度时的调度复杂性。

---

## 486. Efficient VLA Reinforcement Learning via Subchunk Rollouts in Latent World Models

**arXiv ID:** 2611.27645 | [PDF](https://arxiv.org/pdf/2611.27645v1)

**作者:** Nan Tang `[一作]` (Nanjing University), Yang Yu `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过轻量级残差模型学习框架，利用潜在世界模型和层级强制的想象化经验，提升预训练的视觉语言动作（VLA）策略；

**💡 创新点**

引入层级强制（Hierarchical Forcing）降低递归预测误差，提出潜在子块滚动（Latent Subchunk Rollouts）桥接潜在空间与VLA接口，并设计进度条件分支起点课程（Progress-Conditioned Branch-Start Curriculum）实现学习效率提升；

**🔧 技术方法**

使用潜在世界模型、Vision Transformer 编码器、层级强制、潜在子块滚动、TD3+BC 残差策略优化、进度预测器及其采样策略；

**📊 数据集**

在 LIBERO、RoboTwin 以及真实 ALOHA 机器人任务上使用离线数据集和专家数据进行训练与评估；

**📈 对比分析**

与 π_0.5‑SFT、RLT‑BC/Latent‑BC、RLT‑RL/Latent‑RL、Dreamer、VLA‑MBC 等基线对比，LS‑MBPO 在所有任务上均取得更高成功率，尤其在 LIBERO Long 上成功率 71.2% 超过 VLA‑MBCPO 的 64.4%，训练时间更短、资源更少；

**⚠️ 局限性**

仅支持有限长度的子块滚动；进度预测器泛化不足，可能导致奖励信号失真；需要更通用的密集奖励模型及更强的跨任务迁移能力。

---

## 487. Shellsort with $O(n\log n)$ Average-Case Complexity

**arXiv ID:** 2611.27671 | [PDF](https://arxiv.org/pdf/2611.27671v1)

**作者:** Cheng Du `[一作]` (Great Bay University), Zongqi Wan `[通讯]` (Great Bay University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究 Shellsort 的增量（gap）选择问题，证明存在长度为 O(log n) 的固定增量序列，使得在对均匀随机排列进行排序时，期望比较次数为 O(n log n)。并给出一种随机化多项式时间算法，能够高概率构造出满足该性能的增量序列。

**💡 创新点**

创新点在于：
• 证明了平均情况下 Shellsort 可以达到最优的 O(n log n) 比较次数，之前仅有上界 O(n log²n) 或非最优的常数阶结果；
• 通过引入“随机增量调度”与“潜能函数”分析，首次实现对随机排列的平均性能下界与上界匹配；
• 提供了一种高概率构造固定增量序列的随机化算法，克服了以往只能说明存在但难以实际构造的空缺。

**🔧 技术方法**

核心技术包括：
• 对 Shellsort 的插入排序过程进行潜能函数（Q(A)）与权重 W_h(A) 的数学分析；
• 通过在每个二进制尺度上使用固定增量和随机增量的组合，构造“随机增量调度”；
• 利用概率与期望分析（Markov、Hoeffding、Chernoff 等）证明平均比较次数与潜能下降的关系；
• 采用样本估计与多项式时间随机采样技术，构造可复制的固定增量序列。

**📊 数据集**

论文中没有使用具体的实验数据集，全部工作基于理论分析与均匀随机排列的概率模型；若有实验验证，亦仅为随机排列的多次重复实验。

**📈 对比分析**

与之前的增量序列（Pratt、Sedgewick 等）相比，本文的增量序列在平均比较次数上达到了理论下界 O(n log n)。实验或理论表明，在随机排列上，其期望比较次数与最优常数成比例，而之前的序列只能达到 O(n log²n) 或更高阶。相较于现有的随机化 Shellsort（如 Goodrich），本文的构造在保持平均最优的同时，输出的序列是严格递减且长度仅为 O(log n)。

**⚠️ 局限性**

局限性与开放点：
• 仅针对均匀随机排列的平均情况，最坏情况仍可能高达 O(n²)；
• 构造算法需要大量随机样本，理论上是随机化的，虽然高概率成功，但在实际实现中仍需保证随机性与样本量；
• 论文给出的固定增量序列在理论上存在但具体数值并未给出闭式公式，需要运行构造算法得到；
• 对于非均匀或特定结构的数据分布，性能未必保持最优。

---

## 488. Too Fast to Doubt? Skipping Reasoning Leaves Jev-like Models Vulnerable

**arXiv ID:** 2611.27679 | [PDF](https://arxiv.org/pdf/2611.27679v1)

**作者:** Yue Li `[一作]` (Nanjing University), Sheng Zhong `[通讯]` (Nanjing University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对四种 System-1 结构化决策模型在 1,440 个配对浏览器任务上进行审计，评估结论与证据注入对决策准确性、校准性和置信度的影响，并探索通过指令控制或低成本推理提升鲁棒性的策略。

**💡 创新点**

①提出了专门针对浏览器决策的配对注入基准，①揭示了结论/证据注入在不同任务难度下的互补攻击规律；②证明低成本推理（≈30 词）即可显著提升攻击抵御能力，同时阐明了低延迟决策与安全之间的权衡。

**🔧 技术方法**

使用 RLCD 训练的直接决策模型（Jev 1.13/1.14、Microsoft Decision‑1、Luna Decisions）与相同底层模型的 GPT‑6 Luna（无推理与低成本推理）进行对比；评估包括准确率、攻击成功率、ECE、置信度阈值过滤与推理词数/延迟。

**📊 数据集**

1,440 个合成浏览器决策任务（六个计算族 × 八个难度等级），每个任务配有干净、结论注入、证据注入三种页面变体，构成完整的配对基准。

**📈 对比分析**

在干净输入下模型平均准确率达 91%，但两轮注入攻击成功率高达 81%；结论注入下 ECE 从 2.6% 上升至 36.7%，证据注入亦升至 36.1%；低成本推理版 GPT‑6 Luna 在注入样本上从 38.5% 提升到 95.8% 的准确率，平均延迟由 0.29 s 提升至 1.94 s，展示了显著的安全提升与延迟代价。

**⚠️ 局限性**

基准为合成任务，未覆盖真实网页多样性与更复杂的多步交互；仅考虑两种注入方式，未涵盖更细粒度或多模态攻击；实验聚焦单步决策，未检验长序列任务中的鲁棒性；低成本推理在同一模型基础上实现，可能不适用于所有直接决策框架。

---

## 489. SMSD: A Semantic Misinformation Spreader Detection Approach

**arXiv ID:** 2611.27661 | [PDF](https://arxiv.org/pdf/2611.27661v1)

**作者:** Zeeshan Bin Siddique `[一作]` (Charles Sturt University), Selasi Kwashie `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了SMSD框架，利用多模态特征与立场感知标签来检测社交媒体中的误信息传播者。

**💡 创新点**

创新点包括：①引入立场感知标注消除语义噪声；②采用用户中心的月度分割模拟新用户加入；③结合行为与BLOC特征实现早期检测。

**🔧 技术方法**

使用多层感知网络（MLP、GCN、GRU）、BERT嵌入、图注意力池化、BLOC行为指标以及立场分类Transformer。

**📊 数据集**

使用FibVid COVID‑19误信息数据集（包含PolitiFact与Snopes的事实检查声明和推特对话）。

**📈 对比分析**

与CounterSpread、GraphSAGE、IHFS、FakeThreads等基线在阈值与立场感知标签下进行AUC、召回、F1和MCC评估，SMSD在所有设置下均显著优于基线，尤其立场感知标签下召回提升45%、F1提升49.5%、MCC提升74.6%。

**⚠️ 局限性**

局限性包括：缺乏真实用户级标注；实验仅在单一数据集上验证；对AI生成误信息及其他社交平台的适用性仍待验证。

---

## 490. A Switching Mechanism for Selecting Environment-Adaptive Personalized Gains in Hearing Aids

**arXiv ID:** 2611.27726 | [PDF](https://arxiv.org/pdf/2611.27726v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 491. AutoOntologyBench: A Task-Grounded Benchmark for Agentic Knowledge Engineering

**arXiv ID:** 2611.27734 | [PDF](https://arxiv.org/pdf/2611.27734v1)

**作者:** Minye Shao `[一作]` (Durham University), Yang Long `[通讯]` (Durham University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `a2602d71-93ab-4bad-974b-672788df8193` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个名为 AutoOntologyBench 的任务驱动基准，评估自主代理在获取、构造本体知识层后对实际任务的影响。

**💡 创新点**

创新点在于将本体构造与任务成功度直接关联，提供可验证的原始数据源、原语法以及多任务、跨域的评测框架，并通过原子化损坏实验揭示知识准确性对任务结果的决定性作用。

**🔧 技术方法**

使用了检索增强生成（RAG）、本体学习、原生 DSL 以及直接回答等知识表示技术，并在代理内部实现两阶段研究-执行协议来构建和利用本体。

**📊 数据集**

基准涵盖 18 个真实版本化知识源、7 个领域（政策、工作流、化学、生物、临床、家居、Web API），共 19,947 个可验证任务实例。

**📈 对比分析**

通过与直接回答、RAG、原生 DSL 以及专家预置参考的对比，发现 RAG+本体相较于仅检索可提升 6.1 个点任务成功率，特别在跨源组织和多源科学域表现显著，但仍与专家基准相差 13 个点，表明知识获取仍是瓶颈。

**⚠️ 局限性**

限制在于评测仅基于原生执行器（不涵盖临床疗效或模拟真实实验结果），本体构造仅在单任务、固定预算下完成，缺乏跨任务持续学习与更广泛源覆盖。

---

## 492. Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders

**arXiv ID:** 2611.27721 | [PDF](https://arxiv.org/pdf/2611.27721v1)

**作者:** Alessandro Piccolo `[一作]` (Sidea Group), Eufemia Lella `[通讯]` (Sidea Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统比较了零射击决策模型、通用大语言模型以及微调后的多语言编码器在四类情感分类任务中的表现。

**💡 创新点**

创新点在于引入等价性检验与随机效应元分析，全面评估质量、延迟、鲁棒性及校准差异，并首次将决策模型与同尺寸LLM进行匹配对比，分离出训练与格式对性能的影响。

**🔧 技术方法**

采用宏F1、期望校准误差、Brier分数、延迟测量等指标，使用配对Bootstrap、Holm多重校正、TOST等价性检验以及随机效应元分析来评估系统性能。

**📊 数据集**

使用十个公开多语言社交媒体数据集（共12,633条文本），覆盖五种语言，四类情感。

**📈 对比分析**

实验结果显示，三大零射击系统与微调编码器平均相差约1点，零射击系统优于跨域微调编码器约15点，对最强公开模型提升约6点；决策模型比对应LLM的推理速度快约5倍；Strands Decider 2B在同尺寸LLM上提升约10点；系统对提示语的鲁棒性差异显著。

**⚠️ 局限性**

局限包括数据集规模不均、缺少多标签样本、部分模型封闭实现、实验未预先注册、提示变体数量有限、模型对语言提示敏感性不同、训练集覆盖面有限等。

---

## 493. Seeing Is Not Steering: Where Verifiable Reasoning Can Enter an End-to-End Driving Planner

**arXiv ID:** 2611.27638 | [PDF](https://arxiv.org/pdf/2611.27638v1)

**作者:** Gautam Kumar Jain `[一作]` (Technische Hochschule Augsburg/TTZ Landsberg), Julian Staehler `[通讯]` (Technische Hochschule Augsburg/TTZ Landsberg)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究将外部视觉语言模型生成的场景事实引入冻结的Drive-JEPA规划器，并通过可验证的事实检查来评估事实对路径选择和安全性的影响。

**💡 创新点**

创新点在于将可审计、可验证的事实检查与外部VLM结合，实现在规划器外部实现可追溯的干预机制，而非传统内部推理。

**🔧 技术方法**

采用了微调后的Cosmos-Reason2-8B VLM生成事实，冻结的Drive-JEPA规划器，基于Transformer+MLP的路径评分器，计划记忆模块，以及基于规则的事实检查和自由距离约束。

**📊 数据集**

使用NAVSIM v2数据集（包括navtrain、navtest、navhard）以及SimScale模拟场景进行训练与评估。

**📈 对比分析**

通过与无事实检查、随机/匹配控制以及精确事实诊断的配对比较，发现基线在navhard上从41.2点提升至52.2点，其中路径评分器提升至51.8点，事实检查在基线上仅提升0.3点，但精确事实可提升4.4点，整体表现处于NAVSIM基准的中间水平。

**⚠️ 局限性**

主要局限在于事实可靠性不足，尤其在稀有场景；学习型组件未能有效利用事实；仅评估单一规划器和VLM，缺乏跨系统泛化；事实检查虽可追溯但受限于事实准确性；未完全解决连续计划间的舒适性问题。

---

## 494. Post-Quantum Migration in Brownfield 5G Core Networks: Verified Hybrid Key Establishment, Performance, and Rollback

**arXiv ID:** 2611.27735 | [PDF](https://arxiv.org/pdf/2611.27735v1)

**作者:** Shubham Verma `[一作]` `[通讯]` (Centre for Development of Telematics), Shubham Verma (Centre for Development of Telematics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在5G核心网络的AMF–NRF路径上插入可部署的Boundary Cryptographic Execution Entities（BCEEs），实现了在不改动网络功能源代码的前提下，将传统X25519加密切换为标准化的X25519MLKEM768混合密钥协商。

**💡 创新点**

创新点在于：①提供了可部署的、可回滚的中介边界，实现对S_2段TLS密钥协商的独立控制；②通过同步捕获和packet级证据，验证混合密钥协商在实际业务流中的成功；③在同一受控拓扑下比较经典与混合模式的系统级延迟，证明可行性与性能可接受。

**🔧 技术方法**

使用技术包括：Open5GS 5G核心实现、nghttpx（配合OQS Provider）作为BCEEs、TLS 1.3、OpenSSL 3.x、ML-KEM（MLKEM-768）、Wireshark/pcap捕获、Bootstrap置信区间和Mann–Whitney U检验。

**📊 数据集**

实验数据集为120次新建NF-Registration事务，按12个块（每块10次）分别在经典（C1）与混合（C2）状态下收集；同时记录S_2段TLS建立时间与完整注册延迟。

**📈 对比分析**

比较方法：在相同BCEE拓扑下，使用整体块级Bootstrap置信区间和Mann–Whitney U检验比较C1和C2的中位数；结果显示C2相较C1的完整注册延迟增加约+3.383 ms（95% CI [1.193, 7.872] ms）和S_2 TLS建立延迟增加约+4.599 ms，均显著。

**⚠️ 局限性**

局限性包括：仅在单主机虚拟化环境中评估；只涉及AMF–NRF单一业务关系，未测试其他SBI接口；未评估长连接复用、CPU/内存消耗、故障切换和后量子认证等；回滚实验仅执行一次，未提供完整的迁移时序分析。

---

## 495. Abstract What Varies, Reuse Longer Patterns: Efficient Offline Chain-of-Thought Compression

**arXiv ID:** 2611.27710 | [PDF](https://arxiv.org/pdf/2611.27710v1)

**作者:** Shibo Wang `[一作]` (Nanyang Technological University), Junlang Qian `[通讯]` (Nanyang Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 CRAFT 方法，通过槽位抽象和模板聚类对链式思考推理过程进行压缩，显著减少压缩器预算。

**💡 创新点**

创新点在于将问题特定细节抽象为槽位，发现部分词化表达模板（PLET），并将相似模板聚类为族，从而实现跨样本共享删除或重写编辑，减少重复压缩操作。

**🔧 技术方法**

使用了槽位抽象、TF‑IDF 与句向量相似度、最长公共子序列（LCS）、complete‑link 聚类、删除与重写压缩策略，以及监督微调。

**📊 数据集**

使用的数据集包括 GSM8K、MATH‑500、TabMWP、LogicNLI，并在 Qwen2.5（3B/7B/14B）和 LLaMA‑3.1‑8B 模型上进行实验。

**📈 对比分析**

通过与逐样本压缩（TokenSkip、C3oT）以及跨样本编码（Supertoken）比较，CRAFT 在相同准确度或相同长度的对比点下，压缩器预算平均下降约90%，推理长度缩短约25%，在大多数设置中还提高了准确率。

**⚠️ 局限性**

局限性包括：仅在英文算术/逻辑推理数据上验证；依赖表达模式的重复性；每个模型‑数据集需单独构建策略库；不支持实时在线压缩；压缩后的推理解释缺乏可审计性。

---

## 496. Embodied-Harness: Towards Recursive Self-Improvement for Robotic Manipulation

**arXiv ID:** 2611.27689 | [PDF](https://arxiv.org/pdf/2611.27689v1)

**作者:** Zhenxuan Fan `[一作]` (Zhejiang University), Yueting Zhuang `[通讯]` (Zhejiang University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 Embodied-Harness 双代理框架，实现机器人操纵任务的经验驱动递归自我改进；

**💡 创新点**

创新点在于引入层级记忆架构、基于证据的经验反思与递归 harness 自我改进，使得不更新模型参数即可持续提升执行性能；

**🔧 技术方法**

采用视觉语言模型（GPT‑6 Astra）作为固定基础，配合 Actor‑Reflector 结构、层级工作记忆、经验记忆、反射诊断与指导生成等技术；

**📊 数据集**

在 RoboTwin 2.0、RoboSPA、RMBench 三大仿真基准以及 Piper 机器人真实任务上进行评估；

**📈 对比分析**

通过与 VLA、WAM、Agentic 框架等基线在相同零演示、零参数更新条件下对比，取得平均成功率 82.4%、76.7%、94.4%（仿真）及 96.7%（真实），比最强基线提升 10–30+个百分点；

**⚠️ 局限性**

局限性包括：高度依赖强大的 VLM 基础；反思与改进过程需要多轮、耗时且样本成本高；跨任务、跨硬件迁移性能尚未充分验证。

---

## 497. GeoScaffold: From Latent to Realized Geometry in Collaborative Detection

**arXiv ID:** 2611.27667 | [PDF](https://arxiv.org/pdf/2611.27667v1)

**作者:** Kang Yang `[一作]` (Renmin University of China), Yongcai Wang `[通讯]` (Renmin University of China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 GeoScaffold 训练辅助目标，在协作3D检测中通过前景掩码监督正样本 anchor 的空间覆盖，显著提升定位质量，训练后无需改动推理流程；

**💡 创新点**

创新点在于仅在训练阶段使用轻量化前景掩码预测头，弥补 NMS 只按分类得分选择框导致的定位失配，且不增加推理成本；

**🔧 技术方法**

采用 anchor 对齐采样、轻量掩码头、交叉熵+Dice 损失，并将其嵌入 11 种基于 PointPillars 的 BEV 融合模型；

**📊 数据集**

使用 UrbanIng‑V2X（多交叉口真实协作）和 V2V4Real（双车协作）数据集进行训练与评估；

**📈 对比分析**

与原生检测器及 11 种融合方法对比，在 UrbanIng‑V2X 验证集平均提升 AP@0.5 +0.98、AP@0.7 +0.55；在保留交叉口、V2V4Real 数据集亦保持提升，表现显著优于基线；

**⚠️ 局限性**

限制在于只解决得分驱动 NMS 选盒问题，对多目标分辨、姿态误差下的鲁棒性等其他瓶颈关注有限；且目前仅适用于 anchor‑based 架构，非 anchor‑free 方法的可行性待验证。

---

## 498. CanopyFormer: Attention-Guided SegFormer for Explainable Forest Cover Segmentation

**arXiv ID:** 2611.27674 | [PDF](https://arxiv.org/pdf/2611.27674v1)

**作者:** Dhinanjaya Fernando `[一作]` (University of Moratuwa), Sandareka Wickramanayake `[通讯]` (University of Moratuwa)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

设计并训练了一种可解释的森林覆盖分割模型 CanopyFormer，利用注意力一致性损失使 SegFormer‑B0 的注意力与真实森林掩模对齐。

**💡 创新点**

创新点在于：①首次将注意力监督引入森林分割任务；②将梯度加权注意力回放适配 SegFormer 的空间降维注意力，仅在第4阶段实现；③提出平均注意力‑掩模重叠指标 AAMO 与删除测试来量化对齐与可信度；④比较 MSE 与 KL 的注意力损失以及边界精炼模块。

**🔧 技术方法**

技术手段包括：Transformer SegFormer‑B0、梯度加权注意力回放、Attention Consistency Loss (MSE)、AAMO 指标、删除测试、双向反向传播、边界精炼模块、损失权重调优。

**📊 数据集**

使用的公开数据集为 DeepGlobe 2018（5,108 张 256×256 RGB 图像及森林掩模）和 LoveDA（1024×1024 彩色图像拆分成 256×256 补丁，森林为正类）。

**📈 对比分析**

通过与 U‑Net、DeepLabV3+、SegFormer‑B2、Swin‑T+UPerNet 等基线在 Dice、IoU、F1 等指标上比较，发现加入注意力一致性损失后 AAMO 由 0.0334 提升至 0.7476（提升 22 倍），Dice 仅从 0.8743 降至 0.8577（约 1.6% 下降）。在 LoveDA 上也保持了相似的对齐提升。删除测试表明对齐提升而非可信度提升。

**⚠️ 局限性**

局限性包括：训练速度变慢（约 2.6–2.8×），仅提升对齐不保证模型解释的可信度，边界精炼模块效果不稳定，AAMO 部分受损失构造影响，实验多依赖单一 seed，且仅在第4 阶段提取注意力可能忽略前面层的有用信息。

---

## 499. Points, Not Pixels: Keypoints, Masks and Oblique Geometry for Floating-Roof Tank Fill Estimation from Overhead Imagery

**arXiv ID:** 2611.27624 | [PDF](https://arxiv.org/pdf/2611.27624v1)

**作者:** Mohd Shahid Kaleem `[一作]`, Maaz Khan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出基于子像素关键点的浮动屋顶储油罐填充状态估计方法，并通过物理衬影模型的闭式逆推算屋顶深度。

**💡 创新点**

创新点在于（1）创建了全量屋顶、阴影、相机几何标签的SynTank-25k合成数据集；（2）设计了M3闭式估计器，直接利用局部仿射投影模型校正斜视偏差；（3）将关键点预测与分割管道对比，证明关键点在细阴影条件下显著优于传统分割方法。

**🔧 技术方法**

采用的技术包括：子像素热图关键点回归（ResNet-18+SimpleBaseline），仿射投影图像模型，M1/M2/M3三种深度反演公式，YOLOv5/YOLOv6等检测器，U-Net分割网络，以及多尺度和多场景的合成渲染。

**📊 数据集**

使用的数据集包括：SynTank-25k（24,987幅合成图像，含完整几何标签）；公开NAIP储油罐数据集（27,639幅RGB芯片，包含边框标签但无填充标注）；以及用于检测器训练的DOTA、OST等。

**📈 对比分析**

通过在SynTank-25k上对关键点与分割两条管线进行多种评估（MAE、IOU、失败率、阴影宽度分层），发现关键点+M3平均填充误差仅为0.0128（相对分割为0.0165），在阴影宽度<6像素时误差提高1.56倍；M3在斜视视角下把误差从0.093降至0.013，显著优于传统的nadir公式；检测器在混合训练下在真实NAIP图像上的AP保持与单独训练相近。

**⚠️ 局限性**

主要局限包括：合成数据的光照、相机投影与真实图像仍有偏差；缺乏真实填充状态标注，导致在NAIP上的验证只能靠内部一致性；数据集未对新场景或罐体尺寸范围进行严格的跨域评估；以及M3假设相机为仿射投影，未考虑更高阶畸变。

---

## 500. Report2Alpha-Bench: Evaluating LLM Agents on End-to-End Quantitative Research

**arXiv ID:** 2611.27635 | [PDF](https://arxiv.org/pdf/2611.27635v1)

**作者:** Yun Zhang `[一作]` (Fundora Lab), Wei Meng `[通讯]` (Fundora Lab)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个端到端的量化研究评估框架：给定真实经纪人研究报告，LLM代理需在7轮迭代中提出假设、实现因子代码并在2022–2023年的训练数据上回测，最终通过固定规则挑选低相关因子池，在2024–2025年的未见数据上去除风格后自动评分。

**💡 创新点**

创新点包括（1）将自然语言报告转化为可执行因子任务并实现完整研究循环；（2）采用固定规则基于训练期结果挑选因子池，区分“研究产出”与“最终挑选”，从而客观衡量模型在未来数据上的泛化能力；（3）引入多维过程指标（文档理解、工程可靠性、假设多样性、泛化度、长期稳定性）剖析模型差异；（4）提供可重复的评测流程，可推广至其他噪声决策领域。

**🔧 技术方法**

技术手段包括：GPT‑6.1‑sol/terra、DeepSeek‑v4.1‑flash、Muse‑spark‑1.3 等大模型通过 Codex CLI 交互；C++因子开发框架实现每日/5min 级别回测；Barra 风格中性化与 Sharpe‑ratio 计算用于评估；固定规则在训练期筛选因子池；抽样重采样生成 95% 置信区间；所有工程检查与回测均为自动化，无人工干预。

**📊 数据集**

使用数据集：7份经纪人量化报告（A–D 难度层级），包含每日/分钟/1min、Tick 与订单簿指标；1,454 个 Barra CNE5 风格/行业暴露字段；训练期为 2022–2023 年的沪深市场交易日，测试期为 2024–2025 年共 485 个交易日。

**📈 对比分析**

比较方法：在训练期对所有候选因子进行回测后，用固定规则（S≥3、相关<0.6）挑选因子池；随后在 2024–2025 测试期去除风格后计算 Sharpe；对比报告基线、模型自选、最佳训练期、训练前20平均等控制组。性能结果显示：19/28 因子池通过测试，平均得分 48.4，显著高于报告基线 25.4 与模型自选 35.9；GPT‑6.1‑sol 最高 74.7，GPT‑6.1‑terra 56.6，DeepSeek 35.8，Muse 26.2；差异主要集中在迭代阶段，GPT 模型在假设多样性与泛化度上优于其他模型。

**⚠️ 局限性**

局限性：样本仅 7 份任务且每模型仅一次运行，随机性与稳健性难以评估；强提示约束模型行为，真实自主研究能力难以体现；预训练期间可能已见到报告或数据，导致测试期分数被高估；池规则参数（阈值 3、相关 0.6）未做敏感度分析；工程检查主要基于日志与静态检查，缺乏逐日动态验证；注释与评估均由 LLM 自动完成，缺乏人工审核；仅适用于 A 股、C++ 因子框架与 Codex CLI，难以直接推广至其他资产或工具。

---

## 501. Benchmarks' Last Exam

**arXiv ID:** 2611.27622 | [PDF](https://arxiv.org/pdf/2611.27622v1)

**作者:** James Zhu `[一作]` (Benchmarks' Last Exam Lab), Zekki Cat `[通讯]` (Benchmarks' Last Exam Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对八个前沿 AI 评测基准的749个任务进行统一三要素（提示、环境、评判器）合规性审核，评估任务有效性、缺陷类别及其对排行榜分数的影响。

**💡 创新点**

首次系统性比较不同评判架构（闭合问答、文档 rubrics、可执行 agentic harness）在任务有效性、缺陷方向与修复难度上的差异，并量化缺陷对分数的“抑制”或“短路”效应。

**🔧 技术方法**

采用任务级审核流程、缺陷分类（C1–C5）、统计检验（Fisher exact、Clopper–Pearson 置信区间）和对比分析；对共享容器与隔离容器执行方式进行对照实验。

**📊 数据集**

审计数据来自 Benchmarks' Last Exam 的八个公开基准（AA-Omniscience v1.1、HLE-Diamond、Terminal-Bench-Science、GDP.pdf、DeepSWE v1.1、Harvey LAB、AutomationBench、Terminal-Bench 4.0），共749任务；使用其公开排行榜成绩（767 模型）。

**📈 对比分析**

与现有排行榜分数对照，显示闭合问答基准的有效率高达92.5%，多步任务仅29.3%；确认71.7%缺陷抑制合法答案，28.3%导致短路通关。隔离容器将共享容器的 C4 漏洞率从14.7%降至2.7%。

**⚠️ 局限性**

局限在于缺陷对排行榜排名的精确影响需在修复后重新评估；样本基准的代表性受分层抽样影响；只关注已公开任务，未覆盖所有新兴多模态或长期规划基准。

---

## 502. Recoverable Symbols, Unstable Execution: Auditing Internal Pathways and Decision Identity in Visual State Tracking

**arXiv ID:** 2611.27715 | [PDF](https://arxiv.org/pdf/2611.27715v1)

**作者:** Ruiqi Liu `[一作]` (Institute of Medical Philosophy and Future Artificial Intelligence), Yanlan Kang `[通讯]` (Institute of Medical Philosophy and Future Artificial Intelligence)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aaccfe5c-6b26-4208-b23c-35331481e142` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了视觉状态追踪模型中符号可恢复但决策不稳定的问题，提出配对诊断方法并通过内部通道干预揭示旋转相位导致的错误。

**💡 创新点**

首次将查询保持的干预与符号恢复监督结合，提供完整链恢复的证据，定位了输入相关的旋转相位通道。

**🔧 技术方法**

使用冻结的ViT编码器、符号监督解码器、双向通道注入/恢复实验以及平衡准确率和错误转移统计。

**📊 数据集**

在合成的G5c和G5r程序家族（448×112像素的符号板）上进行实验，并验证在新程序族上的泛化。

**📈 对比分析**

通过平衡准确率、修复/新增错误数以及预测一致性进行对比，发现尽管恢复准确率可回到88%+，但仍有约7%预测被改变。

**⚠️ 局限性**

实验仅限于合成符号、固定遍历与冻结编码器，未涉及真实图像、未知转换规则或更强大的VLM模型。

---

## 503. From 6G Capabilities to Software-Business Value: A Multivocal Review of Characteristics, Opportunities, Challenges

**arXiv ID:** 2611.27675 | [PDF](https://arxiv.org/pdf/2611.27675v1)

**作者:** Muhammad Zohaib `[一作]` (Lappeenranta--Lahti University of Technology LUT), Muhammad Hamza `[通讯]` (Lappeenranta--Lahti University of Technology LUT)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

开展了一项多声源文献综述，对6G软件业务的特征、机遇与挑战进行系统梳理，并提出了整合框架。

**💡 创新点**

首次将学术研究与灰色文献融合，构建了涵盖能力、机遇与约束的交互式框架，并聚焦软件生产组织而非单纯技术层面。

**🔧 技术方法**

采用了基于证据的软件工程（EBSE）与多声源文献综述（MLR）方法，进行主题编码、主题映射，并使用Fleiss κ检验可靠性。

**📊 数据集**

共收集133篇核心文献（102篇同行评审、31篇灰色文献），来源于IEEE Xplore、Scopus、Web of Science、ACM DL、SpringerLink、ScienceDirect、Wiley以及Google搜索。

**📈 对比分析**

通过三位评审的独立分类，并计算Fleiss κ（0.744–0.832）评估一致性；未进行性能实验或对比，仅验证分类可靠性。

**⚠️ 局限性**

局限在于处于预标准化阶段，缺乏实证验证，定义可能不统一，地理分布偏向欧洲与亚洲，且对灰色文献的质量控制与整体可推广性受限。

---

## 504. Geometrically Constrained Learning II: Agnostic Proper Learning of Lipschitz Functions

**arXiv ID:** 2611.27700 | [PDF](https://arxiv.org/pdf/2611.27700v1)

**作者:** Jane Lange `[一作]` (Institute for Advanced Study), Arsen Vasilyan `[通讯]` (Aarhus University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `f86bf285-fd08-4156-973b-6e6481af8fa0` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文提出了一种在高斯分布上对 Lipschitz 函数进行**proper**学习的算法，即学习出的预测函数本身满足 Lipschitz 条件，并在绝对损失或平方损失下达到给定误差 ε。

**💡 创新点**

创新点在于：①构造了一个带有梯度缺陷约束的低阶多项式回归，保证预测误差和梯度缺陷同时满足要求；②利用 Ornstein–Uhlenbeck 平滑、McShane 延拓与局部高斯修正相结合的技术，能够在保持 Lipschitz 性的同时纠正预测中的局部偏差；③得到的样本复杂度、训练/推理时间以及模型表示规模均为 (d+1)^{λ/ε} 级别的子指数级别，突破了此前仅能得到 improper 学习的限制。

**🔧 技术方法**

主要技术包括：
- 带梯度缺陷约束的凸多项式回归（通过 Hermite 展开与 L_2 范数约束实现），
- Ornstein–Uhlenbeck 软化与高斯采样推断，
- McShane 下包络与局部修正器（使用分位数、Gaussian CDF 逆变换及惩罚项），
- VC 统一收敛与伪维数分析保证经验约束与总体约束一致，
- 通过有限平均近似实现高效查询，
- 预先对标签进行中心化与裁剪以适配无界标签情形。

**📊 数据集**

实验数据集：本文仅在理论层面讨论，在高斯分布上进行抽样实验，未使用公开实测数据集；所有结果均基于合成高斯数据。

**📈 对比分析**

与现有方法比较：作者声称该算法是已知的 **proper** Lipschitz 学习中唯一能够在高斯输入下实现子指数样本复杂度且保持多项式运行时间的方案。论文中给出了样本复杂度 (d+1)^{λ/ε} (log 1/δ)^3、训练时间与表示规模与维度 d 成多项式关系；但未给出数值实验或与其他基准方法的量化对比。

**⚠️ 局限性**

局限性：
- 需要已知 Lipschitz 常数 λ（或通过裁剪得到近似）；
- 仅适用于标准高斯输入分布，无法直接推广到任意分布；
- 样本复杂度与误差 ε 的指数依赖使得在 ε 很小的场景下仍然成本较高；
- 需要高阶多项式（k=O(λ/ε)）的系数计算与存储，导致模型规模在理论上可达 (d+1)^{λ/ε}，实际实现仍具挑战；
- 对标签方差的先验知识或通过预处理估计的依赖，使得算法对异常值或极端噪声敏感。

---

## 505. Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild

**arXiv ID:** 2611.27673 | [PDF](https://arxiv.org/pdf/2611.27673v1)

**作者:** Mao Zheng `[一作]`, Guanghua Yu `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Hy-MT3 系列多语种翻译模型（1.8B、7B、30B-A3B），显著提升翻译准确性、完整性和指令遵循度，并扩展语言覆盖至 102 语种；

**💡 创新点**

创新点包括在 3B 激活参数下使用 MoE 架构实现高效翻译，构建基于 LLM 的 TransRubric 细粒度评估框架，针对游戏本地化和 OCR 级图像翻译场景做专门优化；

**🔧 技术方法**

技术手段主要是 Transformer‑Mixture‑of‑Experts、LLM‑as‑Judge 评估方法、自动化游戏翻译工具链与图像 OCR 级流水线集成；

**📊 数据集**

使用多种公开基准和工业数据集，包括 FLORES‑200、WMT26、Mandarin‑Minority、WildMTBench‑202611、HardMTBench、IFMTBench、VISTA‑Bench 以及与 Honor of Kings 合作的游戏领域数据；

**📈 对比分析**

通过 TransRubric（与 GEMBA 结合）与同类开源、商用及闭源思考模型对齐评测，Hy‑MT3‑30B‑A3B 在非思考模式下取得 88.5（FLORES）/91.3（WildMTBench）/45.73（VISTA）等最佳成绩，接近闭源思考模型但计算成本更低；

**⚠️ 局限性**

局限性在于对多约束指令的处理仍不如大型思考模型；在知识密集型任务上仍落后于极大规模模型；虽然错误率下降显著，但仍有遗漏、误译与术语问题存在。

---

## 506. Quantitative Measurement of Pavement Distress based on Vehicle Camera and Perspective Correction

**arXiv ID:** 2611.27728 | [PDF](https://arxiv.org/pdf/2611.27728v1)

**作者:** Sike Wang `[一作]` (Purdue University), Chengcheng Tao `[通讯]` (Purdue University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过前视车辆摄像头图像，利用车道检测、透视变换、尺寸校正和YOLOv11对象检测，实现了不需要现场相机标定的路面裂纹补丁定量测量。

**💡 创新点**

①用车道边界构建单一透视矩阵，一次性校正所有图像；②利用标准路面标线比例做尺寸校正，将像素映射为实际长度；③对YOLOv11进行定制化微调，提升在车载图像上的检测性能；④将检测框对角线直接映射为裂纹长度，完成快速定量测量。

**🔧 技术方法**

车道检测YOLOv11s、透视变换（单应矩阵）、尺寸校正基于破线宽度、深度学习对象检测YOLOv11、微调训练策略（先大数据集预训练再微调）、图像处理（OTSU阈值、腐蚀膨胀）。

**📊 数据集**

SODiC 2021大规模车载路面图像6000张作为基础训练集，SCRIM测试图像100张做定制化微调集，以及用于验证的SCRIM测试路段图像。

**📈 对比分析**

对YOLOv11n、s、m以及YOLOv5s、v12s等模型进行精度、召回率、mAP@0.5和推理速度比较。微调后所有模型精度显著提升，YOLOv11m最高mAP=0.883；YOLOv11s微调后Precision 0.897、Recall 0.813、mAP 0.881，推理时间≈3.1 ms/图，最终选用YOLOv11s作为平衡方案。

**⚠️ 局限性**

需可见且标准的路面标线来校正尺寸；假设相机位置和方向保持不变；对角线长度仅为裂纹补丁长度近似；未与实测长度进行验证；缺乏对相机振动、安装变化的鲁棒性；不适用于缺失或损坏标线的道路；仅处理裂纹补丁，未覆盖其他破损类型。

---

## 507. Auditing Post-Training Unlearning in Diffusion Language Models using Sparse Autoencoders

**arXiv ID:** 2611.27652 | [PDF](https://arxiv.org/pdf/2611.27652v1)

**作者:** Yunha Seo `[一作]` (Soongsil University), Changhoon Kim `[通讯]` (Soongsil University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `f86bf285-fd08-4156-973b-6e6481af8fa0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在扩散式语言模型（LLaDA‑8B‑Instruct）上，进行稀疏自编码器（SAE）基向量的选择与抑制实验，探索了后训练机器消忘（unlearning）在不同去噪阶段的可行性与效果。

**💡 创新点**

首次对扩散语言模型中的去噪阶段动态进行系统性评估，提出基于SAE的子空间抑制方法，并检验了阶段条件与对齐策略是否能提升选择性消忘的实用性。

**🔧 技术方法**

使用SAE特征选择、QR正交化子空间构造、LoRA参数微调与投影惩罚、阶段感知的Replay策略，以及WMDP-Cyber与MMLU等评测指标。

**📊 数据集**

主要数据集为WMDP‑Cyber（用作待忘知识的代理）、MMLU（评估通用任务效能），以及WMDP‑Bio与WMDP‑Chem用于检验无关领域的副作用。

**📈 对比分析**

与随机SAE、Gaussian随机子空间以及无条件/池化/阶段条件的基进行对比；选取SAE子空间可使Cyber准确率降至约26%（比Gaussian低5.5个百分点），但MMLU下降约1.8个百分点；阶段条件抑制未能在相同实用性下显著提升忘记效果，整体消忘仍伴随生成崩溃。

**⚠️ 局限性**

局限性包括：仅在单一模型与单一消忘设置下验证；使用WMDP‑Cyber作为代理，未直接涉及真实个人数据或提供隐私保证；强抑制导致生成内容失真或为空；阶段条件未显著获益，结果对训练种子与超参数敏感；缺乏对目标行为与输出有效性的完整评估。

---

## 508. Manipulation-resistant reputation for peer-to-peer scientific publishing: a combination-therapy approach

**arXiv ID:** 2611.27625 | [PDF](https://arxiv.org/pdf/2611.27625v1)

**作者:** Frédéric Prost `[一作]` `[通讯]` (Inria), Frédéric Prost (Inria)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种基于组合疗法思路的可抵御操纵的P2P科学出版声誉系统设计，核心思路是通过多种不互相抵消的声誉信号与最小值聚合来抑制攻击者的收益。

**💡 创新点**

创新点在于将药物组合治疗的“无交叉抗药性”与声誉信号的“协同抵抗”对应，证明在全局线性模型下若存在可行加权即可使所有操纵策略无利可图，并给出非空可行集判定方法。

**🔧 技术方法**

采用的技术包括多层信号抽取（观察→子指标→族级信号），最小值（或软最小）聚合，预算限制、人力资源限制、代币抵押与沉没机制、以及基于区块链的BFT共识与可验证日志。

**📊 数据集**

使用合成数据进行数值仿真，模拟了多种攻击（Sybil、白洗、卡特尔、低质量评审、自动化生成等）以及正常科研活动，评估了信号组合与聚合规则对排名、攻击损失和误报的影响。

**📈 对比分析**

通过与线性加权聚合、几何平均等对照实验，发现最小值聚合在保持误报率可接受的前提下显著降低了伪身份与协作攻击的排名收益，平均每100,000身份重算声誉耗时约1.0–1.6秒，日志增长约26KB/身份/年。

**⚠️ 局限性**

局限性包括：信号参数需通过仿真估计，实际系统的可扩展性与共识延迟未实现；对退役身份的处理、攻击者学习与自适应策略的动态演化、以及伪匿名评审的实际实现仍待研究。

---

## 509. Late-Time Dynamics of Flow Matching: Distance Geometry and Learning

**arXiv ID:** 2611.27653 | [PDF](https://arxiv.org/pdf/2611.27653v1)

**作者:** Umberto Biccari `[一作]` (University of Deusto), Enrique Zuazua `[通讯]` (Friedrich-Alexander-Universitat Erlangen-Nurnberg)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对有限支持的数据集进行流匹配（flow matching）的几何极限分析，得到显式的指数收缩轨迹，并与热分布（score‑based diffusion）对应；基于该几何构造两种监督学习目标（距离势能拟合 PDL 与沿收缩路径拟合 PDFM），并给出速度误差与输出分布间的 Wasserstein 误差估计；在 MNIST 数据集上验证两种方法与传统 DSM、热噪声预测、线性流匹配的对比。

**💡 创新点**

① 通过平方距离势函数揭示流匹配终端动力学的全局几何结构；② 证明在靠近 Voronoi 边界的初始状态下，终端分布误差仅随重启噪声尺度的平方根下降，并给出指数收敛速率；③ 将流匹配与热分布的概率流等价化，统一两种生成方法的几何视角；④ 将几何洞察转化为两种可实现的监督目标（距离势能+梯度、收缩路径+速度），并给出端到端的误差传播理论。

**🔧 技术方法**

流匹配、热分布（heat‑score）理论、Gaussian 插值、梯度与子梯度（Clarke）分析、Wasserstein 距离估计、神经网络（U‑Net + 标量潜在网络）与欧拉积分、MNIST 图像处理与评价指标（NN‑MSE、SSIM、LPIPS、FID、KID）。

**📊 数据集**

MNIST（10,000 训练图像，按数字均分；支持集 10k 图像）以及一个两点（{1,3}）的离散实验。

**📈 对比分析**

采用同一组噪声初始化，比较 PDL、PDFM、DSM、热噪声预测、线性流匹配五种方法。评估指标包括：NN‑MSE、NN‑SSIM、NN‑LPIPS（与支持图像最近邻的像素级差距）、覆盖率与熵（分配给各训练点的频率）、训练/测试 FID 与 KID（图像特征分布匹配）。结果显示：
- FM 取得最小的训练/测试 FID、KID，表明最优的特征分布拟合；
- PDL 在 NN‑MSE、NN‑SSIM、NN‑LPIPS 上仅次于 FM，且在最近邻保留率和支持覆盖率上表现最佳；
- PDFM 与 DSM 在特征分布指标上略优于热噪声预测，但在支持保留率方面稍逊。

**⚠️ 局限性**

仅针对离散支持点的几何分析；当存在大量平行边界或高维连续支持时，近似误差和速度估计难以直接推广；方法需要手工选择重启噪声尺度和初始化分布；实验仅在 MNIST 上验证，缺乏对更复杂数据集（如 ImageNet）的实证；理论假设（唯一最近点、Lebesgue 零集平衡点）在实际高维数据中可能不完全满足。

---

## 510. Trust What Repeats: Capped World-Model Policy Gradients for Humanoid Motion Tracking

**arXiv ID:** 2611.27648 | [PDF](https://arxiv.org/pdf/2611.27648v1)

**作者:** Xuzhe Dang `[一作]` (Czech Technical University), Stefan Edelkamp `[通讯]` (Charles University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

训练并改进了基于世界模型的类人机器人运动追踪器，通过对每个rollout梯度进行裁剪、按梯度重要性采样起始状态以及基于模型预测的重置分布，显著提升了追踪精度和收敛速度。

**💡 创新点**

创新点在于首次量化并利用世界模型梯度在模拟器中的可靠区间（约批量中位数的10倍），提出梯度裁剪与重要性采样结合的更新规则，以及使用模型和评论器重排起始状态，解决了先验梯度中大梯度失真导致的性能瓶颈。

**🔧 技术方法**

技术包括：Newton-模拟器下的MuJoCo Warp求解、基于神经网络的世界模型与评论器、first‑order策略梯度、经验回放、梯度裁剪（按中位数阈值）、重要性采样、模型排名重置、以及对策略的平均化评估。

**📊 数据集**

使用BONES‑SEED 142,220条运动捕捉片段（经Unitree G1 29-DOF/23-DOF重定位）以及LAFAN1 40条片段进行训练与评估。

**📈 对比分析**

与SuperMimic基线相比，在未见动作上归一化回报从0.953提升至0.967，1%尾部回报从0.65提升至0.76，持续失效片段数从15–18降至7–10；收敛时间减半；在LAFAN1上回报从0.906提升至0.935；在23-DOF版本上从0.964提升至0.975。

**⚠️ 局限性**

局限性包括：梯度裁剪阈值经验选择、基线超参数未完全调优、训练迭代时间略长、实验仅在模拟器/虚拟机器人环境中验证、对极端负荷动作（如手倒立）仍无法通过。

---

## 511. Segment Everything, Materialize Nothing: Exact Streaming Inference for Many-Class 3D Segmentation

**arXiv ID:** 2611.27630 | [PDF](https://arxiv.org/pdf/2611.27630v1)

**作者:** Fabian Isensee `[一作]` (German Cancer Research Center), Klaus H. Maier-Hein `[通讯]` (German Cancer Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出一种精确流式推理方法，消除多类别3D医学图像分割中的内存壁垒，实现单模型覆盖全部标签。

**💡 创新点**

创新点在于：①后续步骤在一个轴上局部可流式，②分类头为仿射，可与混合/重采样交换位置，故可用特征环路代替logit缓冲，从而使类数不再与体积乘积相关。

**🔧 技术方法**

使用滑动窗口推理、线性混合、仿射分类头、特征环路等技术，基于nnU-Net/MONAI框架实现。

**📊 数据集**

在公开数据集v3（1,939 CT，117类）和MOOSE（1,683 PET/CT，125类）上进行训练与评估。

**📈 对比分析**

与TotalSegmentator、MOOSE、MONAI等工具对比，单模型推理速度提高5.8–15×、内存降低3.2–7.6×，且分割质量与拆分模型相当或略优。

**⚠️ 局限性**

限制包括仅测试推理阶段；未对拆分模型的网络架构或训练时间做进一步调优；训练时仍受大批量/长patch尺寸的GPU内存限制。

---

## 512. Do Alice and Bob Dream of the Oracle? Wishful Cryptanalysis

**arXiv ID:** 2611.27729 | [PDF](https://arxiv.org/pdf/2611.27729v1)

**作者:** Guoxiao Liu `[一作]` (Tsinghua University), Xiaoyun Wang `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种“wishful cryptanalysis”框架，利用多智能体和自证式验证实现全自动化、可审计的密码学攻击，成功突破 Keccak (SHA‑3) 6‑轮冲突挑战和 AES‑128 哈希 5 跟的多项实用攻击，产生 13 条可自证结果（共 25 条 witness）。

**💡 创新点**

核心创新在于：① 将“期望破坏”与“零进口审计”相结合，构建对抗性验证门；② 在计算受限下压缩理论复杂度并以可执行 witness 证实；③ 通过三向自适应工程（目标、算法、硬件）实现参数阶梯式上升；④ 发布开源多代理框架和工作台，赋予社区可复现的开放式“Oracle”。

**🔧 技术方法**

技术手段包括：大型预训练模型（DeepSeek、GLM、Kimi、ChatGPT 等）作为推理器；自定义多代理架构（Principal、Coordinator、Promoter、Referee、Engineer、Knowledge Steward 等）；可执行验证（Python/C++ 检查器、SAT/MILP 证明、门计数器）；GPU/AVX‑512 并行计算；知识卡与自适应编译器。

**📊 数据集**

使用标准密码学基准：Keccak sponge 参数（b=200/800/1600，r=640，c=160）与 AES‑128 哈希 5 跟（MMO、DM、Semi‑Free‑Start、Fixed‑IV、Two‑Block Fixed‑IV）以及官方 NIST 预置向量；无须自定义数据集，全部基于公开规范。

**📈 对比分析**

对比方法：与此前公开的论文和竞赛结果（如 CRYPTO 2017、2022、2025 等理论上限和实验上限）对齐；性能方面：实现了 7‑轮 SFS 碰撞仅 9.17 s、6‑轮 Fixed‑IV 碰撞 2^48 级别、3‑轮单块碰撞 0.64 ms 等；LLM 推理成本累计 <$1,000，硬件执行成本与传统手工破解相比显著降低。

**⚠️ 局限性**

局限性：① 目前仍聚焦对称哈希和块密码，扩展到公钥、后量子等领域需进一步研究；② 受限于可用计算资源，尤其是多节点 GPU/CPU 集群；③ 对模型选择敏感，低质量模型可能导致错误或无法突破；④ 部分突破依赖多代理协作，单代理实现仍待优化。

---

## 513. To Own or Not to Own? Endogenous Ownership in Peer-to-Peer Sharing of Private Autonomous Vehicles

**arXiv ID:** 2611.27691 | [PDF](https://arxiv.org/pdf/2611.27691v1)

**作者:** Xinling Li `[一作]` (Massachusetts Institute of Technology), Gioele Zardini `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个静态均衡模型，联合考量私人自动驾驶汽车（AV）的所有权、上市（共享）决策与平台运营，分析共享对车辆拥有率的正负影响。

**💡 创新点**

创新点：①首次将 AV 共享市场的所有权外生性与平台运营摩擦（匹配、等待、召回可靠性）结合成闭环均衡；②提出多种所有权分割（单区间、双区间）并给出存在条件；③系统性探讨平台定价、佣金、召回保障及市政调控对所有权的非单调效应。

**🔧 技术方法**

技术：构造连续类型空间、空间泊车匹配近似、车辆时间会计、最优选择推导、均衡方程组求解（一维搜索+根查找）和数值仿真。

**📊 数据集**

数据集：美国 2017 年全国出行调查（日均车辆行驶时间 58.6 分钟）用于校准使用类型分布；同时采用基准成本、AV 购置溢价、运费、佣金等参数，结合密度、运营窗口、车速等情景进行参数扫描。

**📈 对比分析**

比较方法：将自适应所有权模型与固定所有权基准模型对比；通过多维参数网格（密度、溢价、价格、佣金、召回保障、税收、补贴）评估所有权变化、服务覆盖率和平台收益。性能：在多种情景下，模型可快速定位所有平衡点，发现非单调的所有权效应，揭示平台收益最大化往往不降低所有权。

**⚠️ 局限性**

局限性：①模型为静态均衡，忽略时间动态波动与重新平衡；②假设所有人可无限制调整共享时间；③仅关注个体效用，不考虑外部性（交通拥堵、停车、全局福利）；④平台与监管之间的互动未建模为领导-跟随博弈。

---

## 514. Asynchronous Byzantine Lattice Agreement with Logarithmic Response Time and Optimal Resilience

**arXiv ID:** 2611.27663 | [PDF](https://arxiv.org/pdf/2611.27663v1)

**作者:** Hagit Attiya `[一作]` (Technion), Alessia Milani `[通讯]` (Aix-Marseille University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出了一种异步 Byzantine lattice agreement 算法，能够在 n>3f 的系统中完成 O(log f) 时间而无需身份认证；

**💡 创新点**

创新点在于将 Attiya–Rachman 的分类器树框架与“支持可靠广播（Supported Reliable Broadcast, SRB）”相结合，SRB 通过支持验证来保证信息流的完整性，解决了 Byzantine 环境下的分支信息不一致问题；

**🔧 技术方法**

技术上主要使用了基于仲裁集（quorum）的消息传播、SRB 的可验证广播、层级分类器以及信息流完整性校验的 admissibility 逻辑；

**📊 数据集**

论文中未使用任何外部数据集，所有证明均为理论性质；

**📈 对比分析**

与先前同步/异步未认证算法相比，本文实现了最优容错阈值 n>3f 并实现了对数级响应时间，表明在理论上已达到时间复杂度下界；

**⚠️ 局限性**

局限性在于仍未证明 O(log f) 时间是最优的，以及算法在实际网络延迟和故障模型（如网络抖动、部分同步等）下的稳健性尚待实验验证。

---

## 515. Instant-LIVO: Efficient LiDAR-Inertial-Visual Odometry with Compact Geometry and a Bounded Single-Reference-Frame Map

**arXiv ID:** 2611.27633 | [PDF](https://arxiv.org/pdf/2611.27633v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 516. MoHA: Model-Conditioned Harness Adaptation for Long-Video Agents

**arXiv ID:** 2611.27646 | [PDF](https://arxiv.org/pdf/2611.27646v1)

**作者:** Jianghan Chao `[一作]` (Renmin University of China), Yingce Xia `[通讯]` (Zhongguancun Academy)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MoHA框架，利用校准轨迹进行模型条件化的执行框架适配，实现对冻结的规划-观察器堆栈的结构与配置双阶段优化；

**💡 创新点**

将执行框架视为可针对模型进行优化的层，首次通过失败诊断驱动的结构干预与独立观测器配置校准，实现模型特定的高效长视频推理；

**🔧 技术方法**

使用LLM判定失败类型、预定义干预目录、基于验证集的干预筛选、结构化改进与观测器配置校准；

**📊 数据集**

在Video-MME-v1（含Long子集）与Video-Holmes两大长视频理解基准上进行实验，使用64例校准集与32例验证集；

**📈 对比分析**

与初始框架、共享框架及静态全功能框架对比，MoHA在Video-MME Full上提升6.2–27.1个百分点，视频采样帧数降低40%+，并在Video-Holmes上同样显著提升；

**⚠️ 局限性**

适配受限于预设的干预目录和冻结模型，跨模型迁移效果有限；需要手工校准数据，且未实现全自动化的框架组件发现与联合模型-框架协同优化。

---

## 517. PAWS: A Parametric Model of Canine Pose and Shape

**arXiv ID:** 2611.27720 | [PDF](https://arxiv.org/pdf/2611.27720v1)

**作者:** Theo Wieland `[一作]` (KTH), Hedvig Kjellström `[通讯]` (KTH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了一种基于真实犬类3D扫描的参数化犬体形状与姿态模型PAWS，并提供从图像直接回归该模型参数的网络

**💡 创新点**

①使用真实犬类扫描数据学习形状空间，②结合解剖学准确的骨架与耳部关节，③限制自由度至可解释的解剖学自由度而保持拟合精度

**🔧 技术方法**

基于SMPL/SMAL框架的线性混合蒙皮（LBS）+形状/姿态混合函数，配合全流程优化注册、PCA学习、图像回归网络（ResNet+HMR头）

**📊 数据集**

31只不同品种犬的4D扫描数据（共31只，31192帧），手工标注关键点以及合成的图像+深度+边缘数据用于训练回归器

**📈 对比分析**

在留一法交叉验证下与BARC、BITE比较，PAWS在表面重建误差（PA-MPVPE）上明显优于两者；在回归任务中与AA、6D/9D无监督参数化方法相当，但可产生解剖学合理姿态

**⚠️ 局限性**

未能精细建模面部表情与耳部细节，数据集规模有限，且对极端姿态或不同毛发长度的犬类可能泛化不足

---

## 518. SOAR: Training-Free Rank-Consistent Rollout Compression for Latent World Model Planning

**arXiv ID:** 2611.27639 | [PDF](https://arxiv.org/pdf/2611.27639v1)

**作者:** Anqi Li `[一作]` (Shanghai Jiao Tong University), Yulun Zhang `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `fede83ac-7505-405f-ab37-e7284695c47f` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 SOAR，一种训练无关的在线压缩方法，适用于冻结的潜在世界模型规划，按决策动态选择令候选排序不变的 token 预算，保持第一步完整并缓存上下文。

**💡 创新点**

创新点在于将压缩目标从预测特征转移到候选排序；通过在每一步使用 Spearman 相关度探测适合的 token 预算，实现无需任务标签或训练的自适应压缩；同时证明只需完整处理第一步和缓存上下文即可保持性能。

**🔧 技术方法**

技术包括 CEM 采样、Transformer 预测器、KV 缓存、Spearman 相关度探测、三层嵌套 checkerboard 预算、token 数量动态选择。

**📊 数据集**

使用公开的 JEPA‑WM 检查点，在 Maze、Wall、Push‑T、MetaWorld（Reach/Reach‑Wall）以及 RoboCasa（Reach/Place）七个导航/操控任务进行评估。

**📈 对比分析**

与密集模型、空间选择器（CostGrad、Guard‑rank 等）、全步、单帧、任务划分 Oracle 以及 Scope‑WM 等方法对比；在保持约 98% dense 成功率的前提下实现 2.3× 的平均速度提升，部分任务可达 2.5×。

**⚠️ 局限性**

局限包括仅在两帧预测器和基于排序的 CEM 下验证；对更长上下文或成本加权规划（如 MPPI）不作保证；探测仅使用初始 CEM 采样，后续迭代可能进一步压缩；预算集合有限，未尝试更细粒度的分辨率。

---

## 519. Comparative Performance of Machine Learning Architectures for Fault Detection and Diagnosis in Chemical Processes

**arXiv ID:** 2611.27696 | [PDF](https://arxiv.org/pdf/2611.27696v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 520. Learning Inside Types: Epistemic Typestates for Safe Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.27631 | [PDF](https://arxiv.org/pdf/2611.27631v1)

**作者:** Lorenzo Bacchiani `[一作]` (University of Bologna), Roberto Girau `[通讯]` (University of Bologna)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并实现了“epistemic typestates”——一种基于类型的多智能体强化学习安全保障机制，能在训练和执行过程中保证每个学习者的动作不违反安全约束；

**💡 创新点**

创新点在于将安全约束抽象为可检查的类型规范，支持部分可观测、声明式约束、动态生成类型（有限可达信念或无限世界的guard），并证明类型安全与团队安全的组合性；此外证明了遮蔽（shield）是其特例，提供了完全可验证的形式化框架；

**🔧 技术方法**

核心技术包括：epistemic belief update、基于知识的子集构造（synthesis algorithm）、声明式义务与限制、类型判定规则、团队安全定理、Coq/Rocq 形式化证明以及Python/Coq混合实现的类型构造引擎；

**📊 数据集**

使用了九个基准任务：Car Platoon、Connector、Flatland、LBF、Pressure Plate、RWARE、Traffic Junction、MPE Simple Spread 和 VMAS Navigation，涵盖离散/连续世界、不同规模（2–10）以及多种相互依赖结构；

**📈 对比分析**

与 Adalat et al. 与 Brorholt et al. 的遮蔽方法以及 Lagrangian 学习器进行对比；实验显示：在所有基准中无安全违规；任务性能与遮蔽方法基本持平，Car Platoon 中遮蔽读取不可见状态导致性能差异；类型构造时间极快（10 个智能体在 1 秒内完成），远快于 Adalat 方法（8–380 秒）；

**⚠️ 局限性**

主要局限包括：需手工编写准确的环境模型；合成需要有限可达信念，处理无限世界仅通过 guard 且缺乏最大性保证；仅支持离散动作集合；安全依赖关系必须无环；并且类型验证与构造仍需设计者投入，难以自动化。

---

## 521. VeilGS: Solving Darkness as a Veil of 2D Basis Maps over Retinex 3DGS

**arXiv ID:** 2611.27719 | [PDF](https://arxiv.org/pdf/2611.27719v1)

**作者:** Mohab Kishawy `[一作]` (McMaster University), Jun Chen `[通讯]` (McMaster University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 VeilGS，一种基于 3D 高斯散点的新视角合成方法，专门用于低照度图像的去暗化与重建

**💡 创新点**

创新点在于将低照度退化建模为覆盖在理想场景上的‘幕布’（veil），通过空间可变的增益与全局色调曲线来恢复光照；同时将每个高斯从球谐展开替换为表面颜色（albedo）与光照（shading）潜变量，实现光照与物体颜色的分离；最后用差分可微求解器在训练过程中通过求解幕布系数来约束场景，防止高斯吸收可解释的暗化信息

**🔧 技术方法**

使用 3D Gaussian Splatting、Retinex 颜色分离、软加权低维基底表示、可微秩一线性求解器、无监督的重建损失以及曝光/灰世界、光照/色度先验和深度先验的组合

**📊 数据集**

在两个低照度基准上验证：Gaussian‑DK（12 个夜景）和 LOM（5 个场景，分别配有低光与正常光对照）

**📈 对比分析**

通过无参考指标 NIQE、MUSIQ、CLIP‑IQA、PSNR/SSIM/LPIPS 与人类MOS对比，VeilGS 在 Gaussian‑DK 上取得最佳 NIQE/CLIP‑IQA、最高 MOS；在 LOM 上在 PSNR、SSIM 与 LPIPS 上与最先进方法（I2‑NeRF、LITA‑GS 等）竞争，取得最高 PSNR、最高 MOS，并在细节保留方面优于同类方法

**⚠️ 局限性**

局限包括：幕布仅为单色增益，无法校正有色光源的色漂；在纹理稀疏场景求解可能不稳定；表面颜色与光照分离不完美，暗区易出现绿色色差

---

## 522. When Are Images Needed in Multimodal Machine Translation? Text-Only Routing for On-Demand Visual Context

**arXiv ID:** 2611.27712 | [PDF](https://arxiv.org/pdf/2611.27712v1)

**作者:** Xinyu Wang `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种预视觉路由器PreVIR，基于源文本判定是否需要图像输入以提高多模态机器翻译效率。

**💡 创新点**

将视觉调用建模为源文本充分性预测，构建细粒度视觉依赖标签集并实现文本仅路由器，显著减少无用视觉计算。

**🔧 技术方法**

使用XLM‑RoBERTa文本编码器、概率阈值路由、统一的Qwen3.5‑9B视觉语言模型以及COMET/BLEU评估。

**📊 数据集**

构建了3053例的视觉依赖数据集，来源于3AM、VIDA和CoMMuTE，标注三类标签。

**📈 对比分析**

与词典规则、LLM零样本、冻结模型等基线以及随机/长度/含义路由比较，在46%视觉预算下PreVIR达到最高COMET（约87.5），接近全视觉但仅调用约46%图像。

**⚠️ 局限性**

仅覆盖英中翻译、无法处理无效或对齐不良图像、路由仅为二分类未最大化实际翻译收益。

---

## 523. Automating Multi-View Quality Assessment of 3D Annotation Layouts

**arXiv ID:** 2611.27621 | [PDF](https://arxiv.org/pdf/2611.27621v1)

**作者:** Hongyi Yang `[一作]` (Southwest University), Jingwei Qu `[通讯]` (Southwest University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出基于多模态大语言模型（MLLM）的多视角3D注释布局质量评估方法，并构建了ALQA人类标注数据集。

**💡 创新点**

①首次将MLLM用于结构化多视角布局评分；②通过人类细化的5维质量度量训练LoRA微调；③构建规模宏大的多视角评价数据集，填补了该领域的数据空白。

**🔧 技术方法**

使用Qwen3.5-4B MLLM并在其上进行LoRA微调；通过视图权重聚合和结构化目标训练实现多维度评分；使用MAE和QWK等指标进行评估。

**📊 数据集**

ALQA数据集（3216个样本，1072个3D物体，5个预定义视角，5个质量维度）。

**📈 对比分析**

与4个prompt‑only MLLM和7个fine‑tuned MLLM在ALQA上对比，模型取得S‑MAE 0.1388、S‑QWK 0.8645，显著优于基线；用户研究显示模型排名与人类一致率超过90%。

**⚠️ 局限性**

数据集高质量样本占比高，低质量样本不足；仅考虑固定5视角和5维度，可能不适用于所有任务；模型对极端布局的预测仍存在误差。

---

## 524. Beyond Self-Critique: Epistemic Convergence Control for Multi-Agent Research Verification

**arXiv ID:** 2611.27839 | [PDF](https://arxiv.org/pdf/2611.27839v1)

**作者:** Chrisantus Eze `[一作]` `[通讯]` (Oklahoma State University), Chrisantus Eze (Oklahoma State University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `a4b10f5d-130b-4e77-9367-6469ec621899` `c84dae5d-5273-4348-85a7-b44cb586b4df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种多代理科学主张验证框架，通过非对称角色（辩护者、怀疑者、仲裁者）和基于证据的停止准则来解决自我批评偏见和迭代停止问题。

**💡 创新点**

创新点在于引入异构角色结构以降低角色嵌入相似度并增强争议发现；设计基于信念变化的 Belief-Delta 停止规则；以及将每条支持的主张绑定到可审计的原文片段并用 NLI 检验其真实性。

**🔧 技术方法**

使用 LLM（DeepSeek‑R1‑32B）进行推理、LangGraph 搭建管道、对抗式角色提示、嵌入相似度监测、Belief-Delta 计算、Cross‑Encoder NLI 与句子嵌入相似度检验。

**📊 数据集**

在 40 个科研假设（20 个 LLM 推理/自我纠正，20 个多代理协同）上进行实验，检索来源来自 Semantic Scholar 与 arXiv。

**📈 对比分析**

与对称的 Multi‑Agent Debate、单体 Self‑Refine、固定步骤（1/3）比较，结果显示异构角色体系在角色相似度上显著更低（0.852 vs 0.918）、争议主张比例更高（22% vs 15.4%）；Belief‑Delta 实现了 0% 过度迭代，LLM 调用减少 2.4 倍；假言率与单体模型相当，略低。

**⚠️ 局限性**

局限性包括仅使用本地模型（DeepSeek），实验规模有限（40 个假设），缺乏人类标注的真值对照，NLI 模型对科学表达的保守判定可能导致误判。

---

## 525. Beyond Sample Hardness: Adaptive Data Selection through Learning Dynamics

**arXiv ID:** 2611.27656 | [PDF](https://arxiv.org/pdf/2611.27656v1)

**作者:** Hai Gan `[一作]` (Nanjing University), Furao Shen `[通讯]` (Nanjing University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于学习动态的端到端数据选择框架 LeDAS，用于在训练过程中自适应挑选最有价值的数据样本。

**💡 创新点**

创新点包括：① 将样本的难度（hardness）、学习进展（progress）和衰退（deterioration）三种指标联合建模；② 通过上下文相关的Bandit策略动态融合这三种指标；③ 引入有界的时滞补偿机制，使久未被选样本获得优先权。

**🔧 技术方法**

技术方法：指数加权移动平均（EMA）跟踪损失变化；基于Dirichlet分布的上下文Bandit（采用REINFORCE与熵正则）进行权重学习；利用奖励信号（完整数据集损失下降）更新策略；以及乘性时滞补偿因子。

**📊 数据集**

实验数据集：图像分类（CIFAR-10/100、Tiny-ImageNet、ImageNet-1K、ImageNet-A/R/O/Hard、ImageNet-LT/Places-LT）；目标检测（MS‑COCO，YOLOv8/RT‑DETR）；语义分割（ADE20K，UPerNet/PSPNet）；LLM 微调（Nemotron‑Math 及 AIME24、AMC23、Olympiad、GPQA‑Diamond、MMLU‑Pro）。

**📈 对比分析**

与 17 种主流数据选择方法（Herding、EL2N、GraNd、Glister 等）以及全数据训练进行对比；在 50% 训练数据下，LeDAS 在 ImageNet‑1K 上匹配或超越全数据的 Top‑1；在多个任务（分类、检测、分割、LLM 微调）均实现 0.2–0.8% 的提升，且在 ImageNet-1K 上 GPU 计算时间下降约 39%。

**⚠️ 局限性**

局限性：① 对 EMA 以及时滞补偿超参数敏感；② 仍需额外计算用于策略学习，尽管开销低但在极大模型或极低资源环境下仍是负担；③ 主要针对监督式训练，针对无监督或弱监督情形的适用性尚待验证。

---

## 526. Mixture of Latent Recursions for Low-Rank Adaptation

**arXiv ID:** 2611.27823 | [PDF](https://arxiv.org/pdf/2611.27823v1)

**作者:** Weixun Luo `[一作]` (Imperial College London), Krystian Mikolajczyk `[通讯]` (Imperial College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种新的低秩适配框架 MoLR，通过在低维潜在空间中递归细化隐藏状态，结合阈值路由实现自适应深度计算。

**💡 创新点**

创新点在于：① 递归共享变换实现潜在空间高阶表达而不随最大递归深度成比例增加参数；② 轻量级阈值路由和动态阈值更新实现输入自适应递归深度，避免信息泄漏；③ 通过步进编码显式为共享变换注入递归阶段信息，提升表达能力。

**🔧 技术方法**

使用技术包括：低秩适配 (LoRA)、共享潜在变换、阈值路由、动态阈值学习、正弦/余弦步进编码、负载均衡策略、梯度共享与参数共享。

**📊 数据集**

在多模态数据集上验证：自然语言理解 (GLUE, 多任务)，自然语言推理 (ARC-C, BoolQ, GSM8K, WinoGrande, 代码生成)，图像分类 (CLIP‑ViT‑B/16, SigLIP‑ViT‑L/16, DTD 等)。

**📈 对比分析**

与 LoRA 及其七个主流变体（DoRA、AdaLoRA、TopLoRA、HydraLoRA、MoSLoRA、AuroRA 等）进行对比，MoLR 在相同参数预算下平均提升 0.5–2.0%（根据任务而定），在所有语言和视觉任务中获得最高分数，并在推理时可通过调节递归深度实现计算–性能折中。

**⚠️ 局限性**

局限性包括：① 仍需额外 per‑token 计算，无法像 LoRA 那样合并到主干模型；② 最大递归深度设置仍需经验调优，过大可能导致干扰或性能衰减；③ 对于极大模型/极大数据的可扩展性尚未完全验证；④ 依赖阈值学习率和负载均衡参数，若设置不当可能影响收敛。

---

## 527. $\mathtt{DistP}$: Scaling Local-Update Training with Zero-Shot Hyperparameter Transfer

**arXiv ID:** 2611.27808 | [PDF](https://arxiv.org/pdf/2611.27808v1)

**作者:** Andrej Jovanović `[一作]`, Nicholas D. Lane `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个零样本超参数迁移框架，扩展到局部更新分布式训练，并给出宽度、深度、批量和同步步长等多维度迁移规则，同时对不同外部优化器的影响进行了理论分析和实验验证。

**💡 创新点**

创新点在于首次在局部更新训练环境下，结合平均与线性外部优化器（如动量）及自适应外部优化器（如Adam）推导宽度/深度、批量和同步间隔的迁移修正，提供理论保证并验证其有效性。

**🔧 技术方法**

技术手段包括使用Transformer预归一化Decoder‑only模型、SDE近似分析梯度噪声与学习率、设计局部与外部优化器同步机制，以及在不同同步间隔、批大小和工作节点数下进行大规模实验。

**📊 数据集**

实验数据集包括SmolLM混合语料（Web、教育、代码、数学文本）进行语言建模，以及ARC‑Easy、ARC‑Challenge、HellaSwag、PIQA、MMLU等下游任务。

**📈 对比分析**

通过与传统无迁移规则Baseline在相同计算量下的验证损失和下游任务准确率对比，迁移框架在同等算力下损失下降约8.1%，下游准确率提升10%，尤其在大模型（823M）与高预算场景下优势更为明显。

**⚠️ 局限性**

局限性包括对自适应外部优化器的批量与同步迁移仍需进一步调优，局部步长对动量外部优化器的影响尚未完全理论化，且在极大局部步长或长通信间隔下的收敛性仍待研究。

---

## 528. Shared Understanding in Human-AI Teams: Coordination Continuity in Evolving Work

**arXiv ID:** 2611.27746 | [PDF](https://arxiv.org/pdf/2611.27746v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 529. Ref-YOLO-World: Decoupling Text-Side Computation for Efficient Referring Expression Comprehension on Edge Devices

**arXiv ID:** 2611.27756 | [PDF](https://arxiv.org/pdf/2611.27756v1)

**作者:** Yuki Aoki `[一作]` (Keio University), Jin Nakazawa `[通讯]` (Keio University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种名为 Ref-YOLO-World 的一阶段参照表达理解（REC）模型，能够在连续视频帧中仅预计算一次文本特征，随后在每帧上复用该特征进行目标定位；

**💡 创新点**

通过保持文本分支与图像特征解耦，保留每个词的 token 级特征，并在检测网络的 neck 和 head 通过轻量级跨注意力实现 token‑aware 关注；

**🔧 技术方法**

使用 YOLO-World 作为骨干，改进的 Ref-T-CSPLayer（Token‑Aware Attention）和 Ref-Text Contrastive Head，结合 CLIP 文本编码、轻量化跨注意力、零初始化以及文本侧预计算；

**📊 数据集**

在 RefCOCO、RefCOCO+、RefCOCOg、TNL2K、Ref‑Adv‑s 等公开数据集上训练与评估；

**📈 对比分析**

在 RefCOCOg 评测中实现 Acc@0.5 81.29%，高于同类解耦文本分支模型 EEVG；在 TNL2K 上 fine‑tune 后获得 56.4 AUC，超过大多数基于时间信息的追踪器；单帧推理 FPS 38.3，流式推理 47.0；在 iPhone 15 Pro 上 Core ML 实现 15.73 FPS；

**⚠️ 局限性**

模型仍无法利用跨帧视觉状态，可能在多实例场景中切换错误；相较于多模态融合或大语言模型的 REC 方法，准确率仍有一定差距；

---

## 530. Hardness of algorithmic Kadison-Singer problem from approximate hypergraph colouring

**arXiv ID:** 2611.27740 | [PDF](https://arxiv.org/pdf/2611.27740v1)

**作者:** Danny Vagnozzi `[一作]` `[通讯]` (Durham University), Danny Vagnozzi (Durham University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

证明在算法Kadison–Singer问题中，存在任意接近1的常数c，使得判定问题是NP‑hard的。

**💡 创新点**

创新点在于将近似超图着色的PCSP困难性迁移到Kadison–Singer，利用更大r的超图以及Hadamard矩阵的构造，使硬度常数趋近1。

**🔧 技术方法**

采用多项式时间的结构化还原、Hadamard矩阵、正交性分析以及对超图着色的映射等技术。

**📊 数据集**

无实验数据集，全部在理论构造与证明中完成。

**📈 对比分析**

与之前的c<1/2或1/(2√2)等结果相比，本工作显著提高了可证明的硬度常数，逼近极限1。

**⚠️ 局限性**

局限在于未证明极限值确为1，构造实例的维度与向量数成比例且缺乏稠密实例；对不允许多重向量输入的硬度仍未知。

---

## 531. RoSci: Exploring Robustness in Science VLMs

**arXiv ID:** 2611.27750 | [PDF](https://arxiv.org/pdf/2611.27750v1)

**作者:** Junying Wang `[一作]` (Fudan University), Guangtao Zhai `[通讯]` (Shanghai AI Laboratory)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了科学模型鲁棒性（Scientific Model Robustness）概念，构建了RoSci-Bench基准，用1,220个经过信息错误与信息缺失降解的多模态科学问答样本进行评测，并设计了RoSci-Agent框架，利用Parse‑Verify‑Decide流程在回答前对前提进行验证。

**💡 创新点**

创新点在于：①以信息错误与信息缺失为核心的两维鲁棒性评估框架；②构造可控、可重复的多模态降解数据集RoSci-Bench；③提出预验证的RoSci-Agent，首次在VLM问答中实现前提验证、拒答与求澄清的自动决策。

**🔧 技术方法**

技术手段包括：①利用VLM（如GPT‑5）进行知识解析成结构化JSON；②信息验证模块对错误/缺失进行PASS/FAIL检测；③阈值决策生成拒答、澄清或回答；④LLM‑as‑a‑Judge自动评估模型输出并生成整体鲁棒性得分。

**📊 数据集**

数据集：从MathVista、Super‑CLEVR、AI2D、FigureQA、UniGeo、Geometry3K、MV‑Math等七大源数据集挑选305条有效问题，经过人工降解与GPT‑5生成三种变体，合计1,220条样本；同时使用20个VLM（含开放源、闭源及思考系列）进行评测。

**📈 对比分析**

评测方式：在信息错误与信息缺失两类下，分别计算推理稳定性、拒答F1、解释质量三项指标，最后合成整体鲁棒性得分。RoSci‑Agent在RoSci‑Bench上整体得分为44.77，显著高于最佳单体模型31.95；在信息错误与缺失子任务上均表现出显著提升。

**⚠️ 局限性**

局限性：1) 与人工专家相比得分仍相差甚远；2) 评测仅覆盖两维降解，未涵盖更复杂的多模态噪声或不确定性；3) RoSci-Agent依赖阈值与权重调参，需在不同任务域进行再调；4) 评估仍受LLM‑as‑a‑Judge准确度限制（约92.7%人机一致）。

---

## 532. Generalist Action Critic: Harnessing Frontier Models in Autonomous Driving

**arXiv ID:** 2611.27804 | [PDF](https://arxiv.org/pdf/2611.27804v1)

**作者:** Yun Yan `[一作]` (University of Hong Kong), Shengbo Eben Li `[通讯]` (Tsinghua University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并验证了一种名为Generalist Action Critic (GAC)的零样本推理时合作框架，该框架利用前沿大模型作为动作评论员，对专用驾驶规划器生成的多候选轨迹进行评估与重新选择。

**💡 创新点**

创新点包括：
① 通过“证据驱动的推理工具箱”将分层视觉表示和结构化推理上下文融合，为前沿模型提供对场景的充分感知与决策依据；
② 采用双阶段评估–重新选择流程，先判断原始轨迹是否安全，再在必要时仅在候选集内做比较，避免全量搜索；
③ 在不对大模型做微调的前提下，零样本直接将其作为动作评论员，展示了前沿模型在实体驾驶中的实用性。

**🔧 技术方法**

主要技术手段：
- 前沿大模型（GPT‑6 Astra）作为评论员；
- 分层视觉表示（时空场景、空间锚点、未来占用投影）；
- 结构化推理上下文（文本化的决策准则、车辆状态、图像元信息、评估与选择指南）；
- 双阶段推理（评估阶段和重新选择阶段）与选择性决策修正。

**📊 数据集**

数据集：
- HUGSIM（全量与160场景子集）；
- NAVSIM v1（开放循环轨迹评估）。

**📈 对比分析**

比较方式与性能：
- 对DrivoR、iDriveVLA、DiffusionDrive三种专用规划器分别使用GAC；
- 在HUGSIM全量评测中，DrivoR+GAC的HD‑Score从35.7提升至43.2（+21%），超过所有无额外仿真数据训练的方法；
- 在HUGSIM‑160及不同难度级别中，GAC均显著提升NC、TTC和PDMS，DiffusionDrive获得最大相对HD‑Score提升32.1%；
- 在NAVSIM v1开放循环评估中，GAC在失败组恢复率可达52.13%，仅在高质量组新增失败率0.67%。

**⚠️ 局限性**

局限性：
- 依赖前沿模型推理成本，实际部署需要评估实时性与算力开销；
- 仅在视觉感知的环境下测试，缺乏对多模态（LiDAR、雷达）或极端天气场景的验证；
- 重新选择过程仍可能出现过于保守的轨迹选择，导致进度下降；
- 评价主要基于模拟数据，实际道路验证尚未展开。

---

## 533. OpenAHBU: Open-sourced Automatic Human Behaviour Understanding Toolkit Through Facial Behaviour Analysis

**arXiv ID:** 2611.27862 | [PDF](https://arxiv.org/pdf/2611.27862v1)

**作者:** Zijian Wu `[一作]` (Hunan Normal University), Siyang Song `[通讯]` (University of Exeter)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个开源的OpenAHBU工具包，实现了11项面部行为理解功能，涵盖短时可观察行为、短时情绪状态和长期心理特征。

**💡 创新点**

创新点包括：① 统一多数据集训练框架并提供四种不同规模模型；② 通过Synergistic Multi‑Task Enhancement (SMTE) 模块自适应提升各功能性能；③ 解决了现有工具包仅支持单任务、训练细节不完整、跨域泛化差的问题。

**🔧 技术方法**

采用的技术包括：深度学习网络（ResNet、ViT、Transformer、GNN、MAE等）、SAM优化、跨任务注意力和轻量化 MLP 组合、数据增强和重采样策略。

**📊 数据集**

使用的数据集共28个，覆盖面部表达、AU、情绪、疼痛、抑郁、人格等任务，包括 AffectNet、BP4D、DISFA、FER+、RAF‑DB、AVEC2014、FirstImpression、BioVid 等。

**📈 对比分析**

通过与单任务 SOTA 模型和现有开源工具包（OpenFace、LibreFace、Py‑Feat 等）在多数据集上对比，OpenAHBU 在多数任务上保持与甚至超越 SOTA，平均性能提升约 30–40%，同时跨数据集方差显著下降，表现更稳定。

**⚠️ 局限性**

局限性：仅覆盖有标准数据集且已公开可复现的功能；FRPR 仅作为研究版未重新训练；SMTE 目前仅支持单一最佳辅助函数，未尝试多辅助组合；未包含非面部或多模态行为分析功能。

---

## 534. Anchors That Move Without Collapsing: Exclusive-Region Anchor Routing for Vision Transformers

**arXiv ID:** 2611.27855 | [PDF](https://arxiv.org/pdf/2611.27855v1)

**作者:** Aadi Gupta `[一作]` `[通讯]` (Scaler School of Technology), Aadi Gupta (Scaler School of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究视觉变压器中的锚点瓶颈，比较多种锚点路径并提出“focus anchors”以避免锚点聚合。

**💡 创新点**

证明硬独占分配是训练移动锚点的唯一稳定且准确的方法，并通过“focus anchors”在保证分区独占的同时让锚点聚焦图像内容。

**🔧 技术方法**

采用局部‑全局‑局部块、Voronoi 区域划分、Softargmax 位置更新、重要性加权聚合，以及 FlashAttention 对比等技术。

**📊 数据集**

CIFAR‑100、Cluttered CIFAR‑10、Tiny ImageNet canvas、Megapixel MNIST、DFG 交通标志集，以及 ViT、PVTv2‑B0、ResNet‑18 等基准模型。

**📈 对比分析**

通过配对置信区间、均值±标准差与 dense attention/FlashAttention 在 FLOPs、内存、训练时间等维度对比，发现 focus anchors 在低分辨率任务上比固定分区高 1–2 分，匹配 dense attention 仅占 1/5 FLOPs，但在高分辨率任务与预训练模型相比仍落后。

**⚠️ 局限性**

仅在从零开始训练 10–30 轮、最多 100k 样本的条件下验证，未使用 ImageNet 预训练，硬件实现未融合核，且 importance map 仅识别内容而非目标对象。

---

## 535. Auditable Financial Knowledge Graphs: Schema-Driven Construction and Intrinsic Quality Evaluation

**arXiv ID:** 2611.27835 | [PDF](https://arxiv.org/pdf/2611.27835v1)

**作者:** Samarth Sarin `[一作]` (BlackRock, Inc.), Dhagash Mehta `[通讯]` (BlackRock, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并评估可审计的时间金融知识图谱（KG），并将其作为GraphRAG检索子系统用于金融问答；

**💡 创新点**

①将KG构建与内在质量评估分离，提出基于七维度（事实准确、语义一致、完整性、信息密度、关系质量、实体消歧、时间一致性）的可审计评估框架；②采用分阶段LLM提取、严格的模式验证、范围感知实体解析与时间约束，显著降低幻觉、错误合并和时间冲突；

**🔧 技术方法**

Azure Document Intelligence+GPT‑5.4 多阶段实体/关系提取，embedding‑3‑large 进行实体相似度匹配，GPT‑5.5 进行内在评估与硬门限检查，GraphRAG 作为检索+生成框架，FinanceBench 作为下游 QA 验证；

**📊 数据集**

公开的 FinanceBench 150 问题子集（源自 10‑K、10‑Q、8‑K 等披露文件），共 10,231 个 QA‑证据三元组；

**📈 对比分析**

通过与传统 RAG 的下游 QA 对比，KG 的综合质量得分 95%（超过 95% 阈值），GraphRAG 在 150 个 FinanceBench 问题上准确率 82%；评估采用硬门限（节点/边模式合规、完整性）与分数加权，验证了 KG 质量提升带来的问答性能；

**⚠️ 局限性**

①仍受 GPT‑5.x 的幻觉与误抽影响；②对新型披露格式或稀有实体的泛化有限；③评估主要基于 FinanceBench，缺乏多样化文档验证；④实体解析与时间约束硬编码可能导致误合并；⑤构建与评估流程成本高，需大量算力与人工审阅。

---

## 536. DistDNA: Language Model Fingerprinting under Stochastic Decoding

**arXiv ID:** 2611.27831 | [PDF](https://arxiv.org/pdf/2611.27831v1)

**作者:** Lifan Hu `[一作]` (National University of Singapore), Bingsheng He `[通讯]` (National University of Singapore)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于提示条件响应分布的行为指纹（MMD‑CLF），实现无训练分类器的模型检查点识别。

**💡 创新点**

通过将模型表示为提示条件响应分布并用最大均值差异（MMD）度量，解决随机解码导致的样本变异问题，并证明采样误差随采样次数 O(1/R) 减小。

**🔧 技术方法**

使用核嵌入、最大均值差异（MMD）、RBF 与线性核、支持向量机、BGE/MPNet 语义编码器。

**📊 数据集**

采用 Dolly 指令集 80 条评估提示，115 个模型检查点（包含基础与指令微调），以及 20 条校准提示。

**📈 对比分析**

与 LLM‑DNA、PhyloLM‑4char 等基线在 115 检查点和 36 组隐藏解码设置下比较，MMD‑CLF 在最近邻匹配中 Top‑1 81.29%，SVM 下 89.93%，比 LLM‑DNA 提升约10%。

**⚠️ 局限性**

对采样预算与解码设置变化的鲁棒性有限；需要在每个提示上多次采样且对各解码设置分别进行注册，计算与存储成本较高。

---

## 537. Kidney Exchange in Deterministic $O^*(4^{t+o(t)})$ Time

**arXiv ID:** 2611.27841 | [PDF](https://arxiv.org/pdf/2611.27841v1)

**作者:** Binh T. Nguyen `[一作]` `[通讯]` (University of Science, Vietnam National University), Binh T. Nguyen (University of Science, Vietnam National University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种确定性参数化算法，时间复杂度为O^*(4^t+o(t))，解决了肾脏交换（Kidney Exchange）问题中目标受者数量t的决策、最优加权及字典序最大化等变体；

**💡 创新点**

创新点主要包括：①利用动态代表集（Dynamic Representative Sets）实现半可行DP的动态维护，借助残差预算恒等式消除重复压缩；②设计了确定性O^*(4^k+o(k))的最短长有向环算法，实现大组件归约；③将上述技术推广到加权版，得到Exact‑t加权优化的相同指数基；

**🔧 技术方法**

技术手段：动态代表集框架（Nederlof）、残差预算/单一全局预算K=2t、分区方法（Zehavi）求最短长有向环、min‑plus半环下的加权路径算法；

**📊 数据集**

该工作为理论算法，未使用实验数据集，仅在图模型上给出证明；

**📈 对比分析**

相较于之前的确定性O^*(6.855^t)算法，本文将指数基从6.855降低到4（与已知随机化O^*(4^t)相同），并保持多项式因子不变；

**⚠️ 局限性**

局限性：仍是指数级算法，基数4的最优性尚未证明；未讨论更复杂的目标（如多层次收益、动态更新等）；

---

## 538. Graph Signal Generation via Graph-Aware Diffusion Models

**arXiv ID:** 2611.27826 | [PDF](https://arxiv.org/pdf/2611.27826v1)

**作者:** Vimal K B `[一作]` (Delft University of Technology), Hadi J. Rad `[通讯]` (Delft University of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `f86bf285-fd08-4156-973b-6e6481af8fa0` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种面向图信号的扩散模型——图感知扩散（GAD），通过在图傅里叶域使用热扩散过程，将图结构直接嵌入前向噪声，使得后向采样不再需要从数据中学习图依赖性；

**💡 创新点**

创新点在于引入频率自适应噪声调度（MASS），为每个图频率分配独立的扩散速率并统一目标SNR，从而消除频率间的SNR失衡，提高逆向SDE的可解性并显著降低采样步数；

**🔧 技术方法**

主要技术包括热扩散的连续时间SDE、图傅里叶变换、线性最小均方误差分析、频率自适应噪声调度、以及基于图卷积网络的时间条件去噪器；

**📊 数据集**

在实验中使用了随机几何图的合成数据、以及三类真实数据：协同过滤（Amazon、Yelp、Citeulike-t）、交通流量缺失插值（PEMS04、PEMS08）和时间序列概率预测（PEMS08）等；

**📈 对比分析**

与传统的全局（共享）扩散速率、VP式均匀噪声以及先前的FCPS方法相比，GAD-MASS在推荐准确率提升5–19%、交通流量插值RMSE降低7.8%、时间序列预测MAE/ RMSE提升超过10%并仅需3–8步逆向采样，且在不增加模型参数的前提下实现了显著的性能提升；

**⚠️ 局限性**

局限性包括需对图拉普拉斯算子做特征分解（对大图计算开销较高）、对噪声调度需要估计数据功率谱且对多步预测的不确定性处理仍不充分，以及在有向或时变图结构时的适用性尚未验证。

---

## 539. Pushdown Reachability Requires $N^{3/2}$ Time on Multitape Turing Machines

**arXiv ID:** 2611.27814 | [PDF](https://arxiv.org/pdf/2611.27814v1)

**作者:** A. R. Balasubramanian `[一作]` (Max Planck Institute for Software Systems), Rupak Majumdar `[通讯]` (Max Planck Institute for Software Systems)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

论文证明了 pushdown 可达性（或等价的 pushdown 语言非空）问题在多磁带图灵机上的无条件超线性时间下界，即对任意 ε>0，无法在 O(N³/₂−ε) 时间内解决该问题，并将此结果推广到一个 PDA 与 k−1 个 NFA 的交集非空问题，给出下界 O(N³k/₂−ε)。

**💡 创新点**

创新点在于首次给出 pushdown 可达性及其等价问题的无条件超线性下界，利用“本地可寻址电路”将任意时间 T 的图灵机运行压缩为地址长度仅为 log T+O(1) 的电路，再转换为关系程序，最终构造结构化的 PDA 实现对输入规模的精确下界，从而避免了先前依赖假设的条件下界。

**🔧 技术方法**

所用技术包括：递归模拟图灵机、构造本地可寻址电路、关系程序与最小固定点求解、结构化 PDA 的程序与逆程序设计、平衡运行与中心性质证明、以及时间层级定理。

**📊 数据集**

数据集：无，论文为理论分析，不涉及实验或真实数据。

**📈 对比分析**

比较方法：通过证明若存在更快算法则违反时间层级定理，间接证明现有 O(n³) 的饱和算法在多磁带图灵机模型上无法被显著加速，未给出实验对比，主要是理论证明。

**⚠️ 局限性**

局限性：适用于多磁带图灵机模型，对单磁带或其他计算模型的适用性未说明；下界为 O(N³/₂−ε)，并非最强下界；证明高度依赖精确常数，缺乏更通用或更易推广的技术。

---

## 540. ChildAI-Gov: An Evidence-Linked AI System for Assessing EdTech Policies Against Children's Digital Rights

**arXiv ID:** 2611.27799 | [PDF](https://arxiv.org/pdf/2611.27799v1)

**作者:** Naveen Sigalapalli `[一作]` (Newcastle University), Clara Crivellaro `[通讯]` (Newcastle University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了 ChildAI-Gov，利用检索增强生成和多框架儿童权利条文的证据链接，自动评估 EdTech 政策在安全、隐私、透明度、自治、非歧视五个维度上的合规性，并给出 Child Rights Impact Score (CRIS) 与合规判定。

**💡 创新点**

创新点在于：①将检索增强生成与规则引擎双重评分相结合，实现可追溯、可解释的评估；②构建了 AIPolicy-EdTech 数据集并量化了模型偏差；③提供面向儿童的安全助手，增强政策透明度。

**🔧 技术方法**

技术包括：LangGraph 管道、FastAPI、Qdrant 向量库、gpt-4o-mini 语言模型、BAGI/bge-m3 嵌入、关键词规则引擎、黑暗模式与风险门控。

**📊 数据集**

使用了 AIPolicy-EdTech（57份 EdTech 隐私/AI 政策）和四个框架（欧盟 AI 法案、ICO 年龄适宜设计代码、UNICEF AI 指导、英国 Online Safety Act）的条文知识库。

**📈 对比分析**

与无检索、无规则的 GPT‑only 基线对比，后者平均得分高 28.9 分、达标率 44%；而 ChildAI‑Gov 平均 43.7 分、无符合阈值；两者排名相关度为 0.65；与专家审计对比，约 ⅔ 专家发现源于行为/咨询无法通过文本检测。

**⚠️ 局限性**

局限性在于：仅评估政策文本的声明性承诺，未覆盖实际行为；未与其它现有自动化系统做直接比较；规则引擎基于关键词，缺乏深度语义匹配；数据集仅 57 篇英文政策，12 篇超出读取窗口；阈值与权重手工设定，缺乏学习优化。

---

## 541. Redesigning Neural Networks to Not be Confidently Wrong even to Strong Adversarial Attacks

**arXiv ID:** 2611.27811 | [PDF](https://arxiv.org/pdf/2611.27811v1)

**作者:** Michael W. Spratling `[一作]` (University of Luxembourg), Heiko H. Schütt `[通讯]` (University of Luxembourg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

构造了一种新的深度学习架构——STEAL网络，使用改进的非线性积分函数ZINC、空间与通道自适应Mellowmax池化函数SCAM，以及高误差边缘损失HEM，形成浅层网络实现图像分类。

**💡 创新点**

创新点在于：① 用近似ZNCC的非线性积分函数ZINC提高特征选择性；② 设计同时跨空间和通道的SCAM池化，增强对外观变化的容忍度；③ 采用只关注大误差的HEM损失提升对未知和对抗样本的拒绝能力；④ 通过上述三项改进，使得网络在保持高准确率的同时显著减少自信错误预测。

**🔧 技术方法**

主要技术包括：ZINC积分函数的实现（局部均值减法与归一化交叉相关），SCAM池化（Mellowmax与多通道聚合），HEM损失（对超过平均误差的样本加权），以及标准的ReLU、AdamW+One-Cycle学习率调度。网络采用全卷积式块结构，保持较少的层数（<10层对比传统深度网络）。

**📊 数据集**

实验使用七个公开图像分类数据集：MNIST、FashionMNIST、SVHN、CIFAR-10、CIFAR-100、TinyImageNet、ImageNet‑1k，并在每个数据集上生成多种测试类型（Clean、Corrupt、Adversarial、Novel、Unrecognizable）。

**📈 对比分析**

比较方法：构造与STEAL网络相同结构但使用传统卷积、ReLU、最大池化与交叉熵的标准CNN；与ResNet、ConvNeXt、ViT等成熟架构；与RobustBench榜单上的强健模型。使用DAR（Detection Accuracy Rate）在多种测试数据上评估，并报告平均DAR、准确率等指标。结果显示，STEAL网络在大多数任务中获得最高DAR，尤其在对抗样本上表现显著优于对照模型，整体性能与当代最先进模型相当甚至更好。

**⚠️ 局限性**

局限性：① 对抗样本仍无法正确分类，只能低置信度拒绝；② 主要针对图像分类任务，尚未验证在其它任务或更大规模数据上的迁移性能；③ 需要额外的超参数（如λ、γ、κ等），虽然表现不敏感但仍需调试；④ 训练与推理效率相对传统CNN略低；⑤ 论文未结合神经科学或心理物理实验验证模型与人类视觉的对应性。

---

## 542. $\varepsilon$-SegFormer: Multi-Resolution Vision Transformers for Sparsely Supervised Segmentation in Electron Microscopy

**arXiv ID:** 2611.27796 | [PDF](https://arxiv.org/pdf/2611.27796v1)

**作者:** Erfan Amidi `[一作]` (Human Technopole), Florian Jug `[通讯]` (Human Technopole)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `0d7d4da1-2b80-44f1-afe6-3f60783c9de2`

**🎯 论文内容**

提出了一种多分辨率视觉Transformer（MuViT）用于稀疏监督的EM细胞器语义分割，训练时隐藏中心像素以避免局部捷径，并使用辅助重建损失。

**💡 创新点**

创新点在于：1）将多分辨率视图嵌入同一Transformer序列并共享坐标；2）在每个视图中隐藏中心token，强制模型利用上下文进行预测；3）一次性从稀疏像素训练，省去预训练与多阶段训练；4）通过多尺度视图实现大视野而不增加token数量。

**🔧 技术方法**

采用的技术包括：多分辨率视图采样、Patch Token化、旋转位置嵌入（RoPE）、单阶段Transformer编码器、中心token遮挡与辅助重建、TTA和后处理。

**📊 数据集**

使用BetaSeg FIB-SEM数据集，包含四个胰腺β细胞的高分辨率扫描图像，共670百万体素；训练使用约10,000个随机标注像素。

**📈 对比分析**

与八种基线（自监督、稀疏监督、全监督）比较，MuViT在测试细胞的单切片上平均Dice达到0.909，领先全监督U‑Net仅0.02，落后于最优秀的全监督Segmenter仅0.01。

**⚠️ 局限性**

局限性包括：仅在单个细胞和单切片上评估，未充分测试其他细胞；稀疏标签为理想化的均衡采样，实际标注可能不均衡；推理速度慢，每个像素单独前向传播，导致高计算成本。

---

## 543. Who Grades the Agent? Privilege Separation for Coding-Agent Benchmarks

**arXiv ID:** 2611.27860 | [PDF](https://arxiv.org/pdf/2611.27860v1)

**作者:** Nathan Nour Houlamy `[一作]` (University of Padua), Mauro Conti `[通讯]` (University of Padua)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Split Verification框架，用于安全评估AI编码代理，并将现有基准任务迁移到该框架中。

**💡 创新点**

创新点在于将任务测试执行与判定结果分离，使用Oracle持有隐藏答案，构建了自动化迁移工具，显著降低代理对评估过程的操纵能力。

**🔧 技术方法**

采用容器隔离、最小权限访问控制、受限网络代理、Oracle与适配器分离、以及任务转换脚本等技术。

**📊 数据集**

使用了Terminal‑Bench 2.0（89项）和DeepSWE v1.1（113项）共202项任务，其中成功迁移180项。

**📈 对比分析**

通过与原生评估对比，判定一致率达到98.7%，迁移成本约为每项$10；在360次攻击实验中，所有尝试均未能通过框架缺陷获得通过，攻击主要暴露了原基准测试的弱点。

**⚠️ 局限性**

限制包括：仍需依赖任务自身测试的健壮性，无法阻止公共答案泄露或基准任务本身的缺陷；对沙箱逃逸、硬件侧信道等攻击未做覆盖；对需要外部服务或权限的任务无法完整支持。

---

## 544. SUPERVISE: A Unified Framework for Standardized and Reproducible Superpixel Evaluation

**arXiv ID:** 2611.27753 | [PDF](https://arxiv.org/pdf/2611.27753v1)

**作者:** Julien Walther `[一作]` (University of Bordeaux), Michaël Clément `[通讯]` (University of Bordeaux)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SUPERVISE，一个统一且可复现的超像素评估框架，解决了尺度偏差、指标冗余和协议不统一的问题。

**💡 创新点**

创新点在于：1) 基于实际生成的超像素数进行尺度归一化并插值；2) 对 20 个评估指标进行统计相关性分析，提炼出五个代表性指标；3) 通过预先计算的标签映射实现方法与评估解耦，极大提升可复现性与可扩展性。

**🔧 技术方法**

使用的技术包括：Python实现的高效评估管线、线性插值与AUC归一化、Spearman/Kendall 相关性分析、PCA 维度约简、以及图像处理库（如OpenCV、NumPy）进行指标计算。

**📊 数据集**

使用的数据集包括：BSD、NYUv2、SBD、KITTI、DRIVE、MONUSEG、PSD 共7个常用超像素评测数据集，覆盖自然图像、深度图、遥感图等多域。

**📈 对比分析**

方法比较采用统一的尺度归一化后插值曲线和AUC综合分数；结果显示深度学习方法（如SSN、SSCN、SPAM、SFCN）在准确率与轮廓保持方面优于传统图像分割算法，而在颜色一致性和正则性上表现略逊；五个代表性指标的组合能有效区分方法并减少冗余。

**⚠️ 局限性**

局限性包括：1) 依赖预先生成的标签映射，需保证其正确性；2) 仅关注内在指标，未系统探讨其与下游任务性能的对应关系；3) 当前仅覆盖二维静态图像，未扩展至视频或三维体素。

---

## 545. DPMA: Learning to Compress Mixture-of-Experts Models

**arXiv ID:** 2611.27759 | [PDF](https://arxiv.org/pdf/2611.27759v1)

**作者:** Jon Irureta `[一作]`, Nicholas D. Lane `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 DPMAbs 的无额外训练的 MoE 压缩方法，能够通过梯度下降同时学习剪枝、合并、吸收操作以及层级专家预算分配。

**💡 创新点**

将参数合并与路由贡献分离，允许单个专家同时进行合并与吸收，并在全局预算约束下以可学习方式分配专家到各层，突破了以往预先决定压缩策略的限制。

**🔧 技术方法**

基于可微分优化的硬约束门控（hard‑concrete gates）实现离散专家选择，利用学习的相似度度量进行专家分配，结合 KL 与重建损失进行校准，采用两阶段软硬优化实现精确的专家数。

**📊 数据集**

使用 C4 文本作为校准集；在八个多选评测基准（ARC‑Challenge/Easy、BoolQ、HellaSwag、MMLU、OpenBookQA、PIQA、WinoGrande）以及 WikiText‑2 做压缩后困惑度测试。

**📈 对比分析**

与随机剪枝、频率剪枝、基于评分的剪枝、以及融合裁剪/合并的先行方法在四个 MoE 模型（OLMoE、Qwen1.5、Qwen3、DeepSeek）以及五个压缩比上进行比较，DPMAbs 平均准确率始终高于所有基线，尤其在高压缩率（75%）下提升可达 6.7 个百分点。

**⚠️ 局限性**

对极低专家数的模型适应性有限（如仅保留最小专家数时无层间再分配），方法实现复杂度较高，且未在非语言任务或大规模 GPU 稀疏部署上验证。

---

## 546. DGC-Net: A Depth-Geometric Conditioning Network for Projection-Supervised OCTA Segmentation

**arXiv ID:** 2611.27854 | [PDF](https://arxiv.org/pdf/2611.27854v1)

**作者:** Lucas Santiago `[一作]` (INESC TEC), Tomás Ribeiro `[通讯]` (INESC TEC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种Depth‑Geometric Conditioning Network (DGC‑Net)，用于投影监督的OCTA血管分割。

**💡 创新点**

创新点是通过局部Depth‑Moment Conditioning（DM）与全局Holistic Depth‑Geometry Modulation（HDM）将轴向几何信息嵌入投影过程，显著降低投影不确定性。

**🔧 技术方法**

采用3D‑to‑2D投影模块、深度时刻（第一、二矩）统计、FiLM式门控、GPC‑Net解码器、GCE‑Dice损失，并在PyTorch框架下实现。

**📊 数据集**

在OCTA‑500（6mm与3mm子集）和DERMA‑OCTA数据集上进行评估。

**📈 对比分析**

与现有2D和3D‑to‑2D基线相比，OCTA‑6mm下DGC‑Net取得0.8358 mIoU/0.9083 mDice，明显优于H2C‑Net和IPNv2；在DERMA‑OCTA上实现IoU 0.7597、Dice 0.8597，超过同类方法。

**⚠️ 局限性**

局限在于对投影不确定性依赖轴向统计，且在低分辨率或极端畸变情况下的泛化性能尚未完全验证。

---

## 547. From Correctness to Consistency: Redefining Reliability for the Agentware Era

**arXiv ID:** 2611.27829 | [PDF](https://arxiv.org/pdf/2611.27829v1)

**作者:** Xue Qin `[一作]` (Villanova University), Maurício Gruppi `[通讯]` (Villanova University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Differential Logic Analysis（DLA）框架，用来检测 AI 助手生成代码中的逻辑一致性问题，并通过实验验证其可行性。

**💡 创新点**

创新点：① 把需求、实现、测试视为三层逻辑实体；② 采用逻辑推理（蕴含、矛盾、中立）对比 prompt 与代码，寻找未被自动测试发现的逻辑漂移；③ 将“逻辑一致性”作为缺失 oracle 时的可靠性评估指标。

**🔧 技术方法**

技术：自然语言到逻辑的映射、程序静态分析、自然语言推理模型（或逻辑推理引擎）实现蕴含/矛盾/中立判定。

**📊 数据集**

数据集：在实验中使用 GPT‑5.2 生成的代码和测试，用两组人工设定的场景（医院门控系统与电商钱包）模拟并发与迭代开发，未使用公开数据集。

**📈 对比分析**

比较方法：对比 DLA 检测到的逻辑冲突与自动生成测试的覆盖率/通过率。实验显示，自动测试虽达 100% 覆盖率并全部通过，但仍未捕捉到 DLA 发现的冲突与未验证假设，说明 DLA 在检测隐藏逻辑缺陷方面优于传统测试。

**⚠️ 局限性**

limitation：① 实验规模有限，仅两类小型场景；② DLA 目前以人工规则为主，缺乏全自动化；③ 对复杂系统或大规模协同开发的可扩展性尚未验证；④ 需要人工介入来提取逻辑前提，降低了可重复性。

---

## 548. Geometrically Constrained Learning I: Agnostically Learning Adversarially Robust Classifiers

**arXiv ID:** 2611.27852 | [PDF](https://arxiv.org/pdf/2611.27852v1)

**作者:** Jane Lange `[一作]` (Institute for Advanced Study), Arsen Vasilyan `[通讯]` (Aarhus University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

在高斯分布下进行鲁棒学习，设计了一种在无监督学习框架中实现既能控制分类误差又能保证对抗鲁棒性的算法。

**💡 创新点**

提出了边界轮廓（boundary profile）K 的概念，并给出一条利用 L1 多项式回归并添加梯度约束来同时优化误差与边界体积的完整算法，解决了之前仅得到非鲁棒解的问题。

**🔧 技术方法**

采用了 L1 多项式回归、OU噪声算子平滑、辅助坐标/距离到边界的“功能提升”、梯度约束（自洽条件）、随机阈值取样与分区裁剪等技术。

**📊 数据集**

理论实验仅在标准正态（高斯）分布（以及可推广到同方差强对数凹分布）上进行，未使用真实数据集。

**📈 对比分析**

与统计查询下的下界及非鲁棒学习的表面积上限相匹配；在多维度 d 维时运行时间为 d^~(K^2/ε^2 + K^4 r^2/ε^4)，当 Kr = O(ε) 时退化为 d^~(K^2/ε^2)，与已知最优下界仅相差对数因子。

**⚠️ 局限性**

依赖于高斯分布的假设，算法对维度 d 和参数 K 的高次方指数导致在实际大规模问题上实现困难；对一般分布的鲁棒学习仍是开放问题。

---

## 549. A code-level survey of CPU scheduling mechanisms in open-source embedded hypervisors

**arXiv ID:** 2611.27827 | [PDF](https://arxiv.org/pdf/2611.27827v1)

**作者:** Lisa Mignerey `[一作]` (Thales), Karim Tamine `[通讯]` (University of Limoges)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对四大开源嵌入式虚拟机（Bao、Xen、ACRN、Xvisor）中实现的十种CPU调度算法进行源代码层面的系统分析和对比，提出了算法分类、准则评估与决策树，帮助系统集成者在多重关键性需求下选择合适的调度器。

**💡 创新点**

创新点在于：①首次对多种嵌入式虚拟机调度器进行统一的源代码级描述和实现差异揭示；②构建了九维评估准则与定量对比矩阵；③给出决策树以指导实际需求匹配。

**🔧 技术方法**

采用静态源代码审计、算法归类（决策规则、参数、触发器、入住控制）、准则定义、时间线仿真（手工推导）等技术手段。

**📊 数据集**

使用四个开源项目的最新提交版本（Xen 7e020a4、ACRN 37b1c13、Xvisor ddbfcd6、Bao v2.0.0）以及一套统一的参考工作负载（4个vCPU，20 ms周期）。

**📈 对比分析**

比较方法是基于九个定性准则的矩阵评估，并结合手工生成的调度时间线进行说明；由于未做运行时性能度量，性能评价主要聚焦于调度逻辑的优势与局限，未给出数值指标。

**⚠️ 局限性**

局限性包括：未进行运行时测量，忽略微架构争用；缺乏对关键性属性与模式切换的原生支持；多数算法不具备入驻控制；评估仅在单核情境下，缺乏多核动态行为分析。

---

## 550. DOiT: Better Money-Flow Ranking Can Coexist with Worse Account Selection in Anti-Money Laundering Review

**arXiv ID:** 2611.27809 | [PDF](https://arxiv.org/pdf/2611.27809v1)

**作者:** Gwangui Hong `[一作]` (KAIST), Min-Soo Kim `[通讯]` (KAIST)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c84dae5d-5273-4348-85a7-b44cb586b4df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究在有限审计容量下，将账户选择与图神经网络（GNN）进行交易流动分析分离，评估GNN重新排序对账户正例率的影响，并提出容量匹配的验证控制方法。

**💡 创新点**

创新点在于：①将账户选择与交易流动排序分离并引入容量匹配验证，②展示即使交易排序提升，账户正例率仍可能下降，③定义并分析“交叉点”现象，④提出验证调度以保护高收益列表，仅在验证支持时才允许GNN干预。

**🔧 技术方法**

使用CatBoost、CLeAR选取器；GraphSAGE、Multi‑GNN进行图神经网络建模；Logistic回归交易解码器用于交易平均精度评估；采用Bootstrap、Holm校正等统计检验，以及基于排名的权重组合。

**📊 数据集**

AMLWorld、AMLNet 两个公开合成 AML 数据集以及 HOFINET 合成基准。

**📈 对比分析**

在相同审计容量 K 下比较上游列表和下游GNN列表的正例率 Pr@K 与交易平均精度 AP；结果表明GNN交易排序提升但有时会导致正例数下降，交叉点出现，验证权重可保护高收益列表并在其优势衰减时允许GNN影响。

**⚠️ 局限性**

限制：使用交易端点标签作为正例，缺乏真实 AML 案件验证；依赖预设邻居采样和表面构建；交叉点位置高度依赖数据集与配置，无法保证在所有实际业务场景中的通用性。

---

## 551. $L^1$ stability of the entropy power inequality in $\mathbb{R}^d$ without moment assumptions

**arXiv ID:** 2611.27790 | [PDF](https://arxiv.org/pdf/2611.27790v1)

**作者:** Norman El Fartass `[一作]` (Ecole Normale Superieure Paris Saclay), Ioannis Kontoyiannis `[通讯]` (University Of Cambridge)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `9ce7179e-700c-4310-ac2b-91df50ded46e` `f86bf285-fd08-4156-973b-6e6481af8fa0`

**🎯 论文内容**

在无任何矩假设的前提下，研究并证明了 Shannon 熵功率不等式（EPI）的 L^1 稳定性与 Ruzsa 距离的稳定性，并对 EPI 的等号条件给出了完整的多维表述。

**💡 创新点**

创新点包括：
1) 首次在无矩条件下给出 EPI 的 L^1 稳定性结果；
2) 将 Rioul 的最优传输方法与 Young 不等式的稳定性结合，得到对 EPI 缺陷的严格控制；
3) 将一维关于 de Bruijn 恒等式与熵连续性的结果推广到任意维；
4) 提出并证明了 EPI 与 Ruzsa 距离在连续情形下的稳定性与等号判定。

**🔧 技术方法**

主要技术手段有：
- Rioul 的最优传输/重排（transport‑rotation）论证；
- 质量守恒与密度重排（spherically symmetric decreasing rearrangement）以及 Riesz 重排不等式；
- de Bruijn 恒等式的多维推广及熵连续性证明；
- Christ 的 Young 不等式的稳定性定理；
- 传输图的三角映射构造与 Fisher 信息的对数变换。

**📊 数据集**

无实验数据集；所有结果均为理论证明。

**📈 对比分析**

未涉及实验比较；本文通过数学证明展示了在 L^1 与 Ruzsa 距离下的稳定性，证明了若 EPI 缺陷趋近于零，则随机向量必趋近于高斯分布；相应的性能评价是证明收敛的速率与稳健性。

**⚠️ 局限性**

主要局限性包括：
- 对相对熵或二次 Wasserstein 距离下的 EPI 稳定性仍不存在；
- 结果仅适用于连续随机向量，离散情况需进一步研究；
- 证明中依赖于密度可正且光滑的假设，后续需要处理更一般的分布；
- 仅给出质性稳定性，未给出明确的量化误差上界。

---

## 552. QuASAR: Toward an LLM-Assisted Framework for Screening, Prototyping, and Assessing Quantum-Advantage Opportunities in Scientific Computing

**arXiv ID:** 2611.27778 | [PDF](https://arxiv.org/pdf/2611.27778v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 553. Enhancing Attention with Higher-Order Attention

**arXiv ID:** 2611.27764 | [PDF](https://arxiv.org/pdf/2611.27764v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 554. EvoTER: Evolving Embodied Agents via Tool-Use Experience and Reflection

**arXiv ID:** 2611.27797 | [PDF](https://arxiv.org/pdf/2611.27797v1)

**作者:** Yingting Zhou `[一作]` (Chinese Academy of Sciences), Dongbin Zhao `[通讯]` (Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出EvoTER框架，实现工具使用技能的经验驱动演化；

**💡 创新点**

创新在于三角色演化循环（Explorer–Diagnoser–Validator），利用完整任务成功率筛选技能更新，且不改动模型参数；

**🔧 技术方法**

技术包括LLM驱动的工具调用、经验记录、诊断与验证循环、仿真到真实的迁移与适配；

**📊 数据集**

使用RoboCasa365 Target50机器人基准与Frank a机器人进行实验；

**📈 对比分析**

与空库和多种基线对比，模拟阶段整体成功率从53.6%提升至80.4%，真实机器人实验中短周期技能获取和长周期任务成功率也显著提升；

**⚠️ 局限性**

局限在于仍需仿真特权诊断、对工具依赖敏感，且未能完全自适应所有环境与工具变化。

---

## 555. A Benchmark Protocol and Fairness Analysis Towards Open-Set EEG Seizure Type Classification

**arXiv ID:** 2611.27783 | [PDF](https://arxiv.org/pdf/2611.27783v1)

**作者:** Targol Bakhtiarvand `[一作]` (University of Colorado Colorado Springs), Adham Atyabi `[通讯]` (University of Colorado Colorado Springs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了首个针对Temple University Hospital Seizure Corpus（TUSZ v2.0.3）的开放集癫痫发作类型识别协议，并在此协议上评估了多种后置开放集评分方法；

**💡 创新点**

创新点在于利用临床癫痫分类本身划分已知与未知发作类型，并设计了留一类实验（LOCO）以及按受试者分离的训练/验证/测试分割，实现了临床可解释的开放集评估框架；

**🔧 技术方法**

使用了基于Mamba+ViT的EEG Foundation Model（EFM）进行参数高效微调，结合了七种后置开放集评分器（MSP、MaxLogit、Energy、Mahalanobis、kNN、GHOST、COSTARR）；

**📊 数据集**

实验数据来自TUSZ v2.0.3共206名受试者、11296个15秒EEG窗口，按12名受试者阈值划分已知与未知类别；

**📈 对比分析**

与七种评分器在三随机种子下进行对比，最高AUROC为0.66（Mahalanobis）对未知类型，已知/未知混合测试AUROC仅为0.54-0.58，表明当前方法识别新发作类型的能力有限；

**⚠️ 局限性**

局限包括样本不平衡、已知类闭集精度不高、受试者分布影响难以区分发作类型与受试者新颖性、仅评估窗口级别、仅使用单一EFM与后置评分器、以及仅三种随机种子带来的不确定性。

---

## 556. Neural Defend ARCAS 290M: A Robust Foundation Model for AI-Generated Audio Detection

**arXiv ID:** 2611.27784 | [PDF](https://arxiv.org/pdf/2611.27784v1)

**作者:** Sivashankar Selvarajan `[一作]` (Neural Defend Inc), Sharayu N. Deshmukh `[通讯]` (Neural Defend Inc)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在大规模官方测试集上评估了Neural Defend的基础模型ARCAS 290M，用于检测AI生成音频，涵盖十个公开基准并在不同生成器、码流、渠道、语言上保持一致性能。

**💡 创新点**

提出了单一冻结的生产模型可作为“AI生成音频检测基础模型”，无需针对每个基准或生成器进行微调，即可在多种条件下实现高精度检测，填补了先前仅在单一数据集上训练、过拟合的空白。

**🔧 技术方法**

使用了自监督前端（如wav2vec 2.0/XLRS）与后端分类器（如AASIST）相结合的深度学习架构，整体模型保持冻结，采用原始波形或频谱特征直接识别合成痕迹；通过HTTP推理端点返回置信度。

**📊 数据集**

对十个公开基准的完整官方测试集进行评测：ASVspoof 5、SpeechFake、ShiftySpeech、CodecFake+、ML-ITW、EchoFake、HQ-MPSD、LlamaPartialSpoof、IndicSynth（仅假音）和PartialEdit（仅假音），共计约564万条录音，覆盖15+语言、电话、蓝牙、社交平台等多种渠道。

**📈 对比分析**

与每个基准自身报告的零样本基线以及商业系统（Podonos排行榜、Speech DF Arena）直接对比；ARCAS 290M在ASVspoof 5实现0.14% EER，SpeechFake 0.56% EER，ShiftySpeech 0.94% EER，CodecFake+ 1.62% EER，ML‑ITW 11.05% EER，EchoFake开集 7.74% EER，HQ‑MPSD 1.7% EER；在五个全文件合成基准上检测率达99.6%，IndicSynth几乎100%召回，PartialEdit 约66%召回；整体准确率在99%以上，显示优于所有零样本基线和多数商业系统。

**⚠️ 局限性**

局限性包括：对短期局部编辑、通过扬声器重放以及已压缩网络音频的检测仍有显著误检；对某些语言（俄语、葡萄牙语、土耳其语）和平台的误报率较高；IndicSynth未提供真实样本，无法评估误报；置信度未经过校准，阈值调整需基于部署数据；未覆盖针对模型的自适应攻击或对抗样本。

---

## 557. From Novel Combinations to Informative Absence: Structural permission, communicative opportunity, and form choice in language learning

**arXiv ID:** 2611.27762 | [PDF](https://arxiv.org/pdf/2611.27762v1)

**作者:** Yanzhe Zhao `[一作]` `[通讯]` (Tianjin University), Yanzhe Zhao (Tianjin University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文提出一套生成式理论，解释语言学习者如何通过共享预测后继、机会与选择来推断结构许可和负面证据，并在人工语法、实验数据及自然语料上进行验证。

**💡 创新点**

创新点在于将结构许可、语义机会与形式选择三者统一为一个生成过程，推导出联合证据法则、有限样本上下文细化不等式、正面经验校准以及结构支持逆转等可检验预测。

**🔧 技术方法**

采用贝叶斯层级模型、Beta分布推理、共享上下文桥接（SCB）算法、信息量（nats）评估，以及 Fisher 信息矩阵分析。

**📊 数据集**

使用的主要数据集包括：30个随机人工语法（30种确定式、名词、后续序列），Samara等人5个人工语言实验的399名参与者（12,760评价、14,080产出），UD‑English‑CHILDES（11家庭、6语料库的儿童与成人对话），以及5种语言的因果标记判断数据。

**📈 对比分析**

方法与基线对比：SCB在人工语法上AUC 0.945（vs 0.5记忆基线），正面经验校准使产出预测提升0.036 nats/句；源级变异模型在CHILDES上提升0.0123 nats/词；五语种因果模型中软最大化网络相对传统频率/语义模型提升0.003–0.004 nats。

**⚠️ 局限性**

局限性包括：实验过度简化的人工语法；对场景机会标注的依赖导致实验成本高；贝叶斯层级假设可能不适用于所有真实语言环境；自然语料中稀缺数据仍可能影响统计稳健性。

---

## 558. Average-Depth Variable Execution via Request-Local Token Routing

**arXiv ID:** 2611.27738 | [PDF](https://arxiv.org/pdf/2611.27738v1)

**作者:** Pengjie Zheng `[一作]`, Feng Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于请求平均深度目标的分层跳过方法（AVERT），通过对每个生成 token 进行预算感知的层级路由，允许同一请求中的不同 token 执行不同数量的 Transformer 层。

**💡 创新点**

其创新点在于：①将预算约束转化为请求平均深度目标，使得每个请求在整体上满足预算；②采用逐层请求本地的 progressive routing，路由器直接读取 token 隐藏状态和预算进行决策；③在跳过层时使用 Skip Compensation Adapter (SCA) 以补偿表示变化；④单个模型 checkpoint 便可支持多预算混合批处理。

**🔧 技术方法**

技术实现包括：token‑level 路由器（使用 W_h、W_Ω 投影、RMSNorm+SiLU、两路评分）、Skip Compensation Adapter、K/V 记忆条目省略、LoRA 轻量微调、Gumbel‑Softmax 的直通估计以及逐层递进的决策机制。

**📊 数据集**

实验采用 Alpaca 作为 fine‑tune 与评估基准，进一步用 GSM8K、IFEval、Prometheus‑2、ROUGE‑L、BLEU‑4 及 PIQA 进行多维度质量评测。

**📈 对比分析**

与 BUDDY、GateSkip、D‑LLM、AdaSkip 等主流预算控制方法对比，AVERT 在三种 LLaMA backbone（7B、8B、1B）下，针对 13 个预算点的 6 个质量指标，获得 206/234 次最佳或并列最佳；在混合预算批处理中能将 TPOT 降低 4.3–21.2%，并显著减少缓存存储和峰值 VRAM。

**⚠️ 局限性**

局限性包括：在极低预算（如 0.4）下尤其在 GSM8K 上的准确率下降；预算在请求开始即固定，无法在生成过程中动态调整；实验范围主要集中在 LLaMA 系列模型，未充分验证跨模型通用性；在低预算场景下整体质量仍有提升空间。

---

## 559. Humanoid robots for safety-critical electrical switching: A perception-informed runtime constraint filter

**arXiv ID:** 2611.27845 | [PDF](https://arxiv.org/pdf/2611.27845v1)

**作者:** Songyang Liu `[一作]` (University of Florida), Shuai Li `[通讯]` (University of Florida)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一个实时安全过滤器，用于 VR 遥控类人机器人在邻近电气开关上的安全切换操作。

**💡 创新点**

创新点在于：1) 使用机器人自身头部 RGB‑D 摄像头即时构建场景模型；2) 采用目标选择为条件的控制障碍函数（CBF）-QP 过滤器，既保持对目标的可达性，又避免与邻近非目标设备碰撞；3) 通过非阻塞调度和自动返回机制保证控制循环不被延迟。

**🔧 技术方法**

技术手段包括：控制障碍函数 (CBF) 与二次规划 (QP)、RGB‑D 视角点云处理与面/盒子建模、非阻塞调度算法、VR 头显手部追踪遥控、仿真平台 MuJoCo 与 Unitree G1 硬件实现。

**📊 数据集**

数据集：实验室自建的两台墙面安全开关场景；仿真使用 MuJoCo 构造的三按钮布局（不同位置与角度）。

**📈 对比分析**

比较方法：对比原始遥控（M0）、仅全目标权限（M1）、仅无方向约束（M2）与完整方法（M3）。在仿真中完整方法实现 9/9 次安全完成，原始仅 5/9；在硬件实验 12 次中无非目标触发，自动返回 11 次，平均 QP 求解时间约 0.5 ms，未出现控制循环阻塞。

**⚠️ 局限性**

局限性：1) 未在带电现场电气设备上测试；2) 仅使用单臂硬件工具，未验证多指手的部件权限与方向约束；3) 试验设备为实验室级安全开关，未覆盖大型变电站设备；4) 未评估电弧或电磁场对机器人感知与控制的影响。

---

## 560. BinXSight: Fast Binary Program Summarization via Hierarchical Modular Decomposition

**arXiv ID:** 2611.27816 | [PDF](https://arxiv.org/pdf/2611.27816v1)

**作者:** Zhijie Liu `[一作]` (ShanghaiTech University), Yutian Tang `[通讯]` (University of Glasgow)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个名为BinXSight的框架，用于快速生成去除符号的二进制程序级摘要，帮助逆向工程师快速理解程序功能。

**💡 创新点**

创新点包括：①提出二进制图（Binary Graph）融合调用图、相似图、链接时间局部性图和数据引用图，实现全方位功能关系表达；②设计分层模块化分解（Hierarchical Modular Decomposition）并结合MapReduce的高价值模块筛选与重构，兼顾语义完整性与LLM上下文窗口限制；③使用函数克隆搜索（KEENHash+FAISS）进行函数名称预测，提升LLM对函数语义的把握。

**🔧 技术方法**

主要技术手段有：Ghidra静态反编译、值集分析、网络图构建（NetworkX）、KEENHash嵌入、FAISS向量搜索、LLM（GPT‑5.4、Claude‑Opus等）进行模块与程序级摘要生成，辅以Hierarchical Louvain社区检测和自定义过滤策略。

**📊 数据集**

使用了92个公开项目共110个去符号二进制（包括Windows、Linux、macOS以及x86/x64/ARM/MIPS等架构），涵盖网络、远控、游戏、DDoS等多领域，并在19个恶意样本上做对比。

**📈 对比分析**

与五个基线（func‑sum、func‑sum‑MiSum、func‑code、sym‑sum、Malsight）及多种LLM进行对比，BinXSight在SIMILARITY、BLEU、ROUGE‑L、METEOR四项指标均领先23–47%；在效率上，平均每个程序24.3秒，整体耗时44.6分钟，远快于基线；用户研究显示其摘要质量平均分超过7分，显著优于基线。

**⚠️ 局限性**

主要局限：依赖静态分析，无法处理打包或加壳二进制；受Ghidra反编译质量影响；对高度混淆或动态代码依赖较弱；目前的过滤策略可能忽略部分低价值但重要的模块；对不同LLM的泛化性仍需进一步验证。

---

## 561. SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech

**arXiv ID:** 2611.27810 | [PDF](https://arxiv.org/pdf/2611.27810v1)

**作者:** Côme Hosxe `[一作]`, Quentin Spinat `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Speech Language Pathologist Bench 评估音频推理模型链中声学主张的准确性、稳定性和一致性，并通过奖励真值的方式提升模型主张的真实性。

**💡 创新点**

创新点在于：① 设计了针对临床语音的链式思维主张评估基准；② 对主张进行逐条真值评分并量化其稳定性与一致性；③ 通过 GRPO 奖励训练显著提升模型主张的真实性。

**🔧 技术方法**

使用了大规模音频语言模型（如 Qwen3-Omni、Step-Audio-R1 等）以及 Qwen2.5-Omni 7B 的 LoRA 与 GRPO 奖励微调；评估工具包括 Qwen3-30B-A3B-Instruct-2507 的主张提取器。

**📊 数据集**

使用了五个临床语音数据集：SAP、HeyJay、NeuroVoz、PVQD、Bridge2AI，共 794 条测试音频，涵盖 25 个声学维度。

**📈 对比分析**

与 10 个开源推理音频模型进行零样本比较，主张的平衡准确率仅为 0.52–0.58，且在重采样或问题改写时表现不稳定；奖励真值的 7B 模型将主张 BA 提升至 0.60，κ 达 0.61，且保留了临床答案的准确性。

**⚠️ 局限性**

局限性包括：① 呼吸相关维度仍接近随机水平，说明音频编码器对这些信号捕捉不足；② 评估仅基于现有数据集，缺乏专门的临床标注金标准；③ 训练预算与奖励权重未完全均衡。

---

## 562. BRINA: Bottom-Referenced Inertial Navigation with AKAZE, a Sonar SLAM for Autonomous Marine Vehicles

**arXiv ID:** 2611.27817 | [PDF](https://arxiv.org/pdf/2611.27817v1)

**作者:** Adolfo Damiano Cafaro `[一作]` (Technical University Of Denmark), Roberto Galeazzi `[通讯]` (Technical University Of Denmark)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种仅使用前向声纳和IMU的海底边缘参考SLAM系统，通过海底首次回波估算声纳高度，并将声纳特征投影到海底平面实现度量尺度的姿态估计与地图构建。

**💡 创新点**

创新点包括：利用每束光束的海底边缘作为海底高度参考，消除对DVL的依赖；通过海底投影将特征转到水平面消除声纳特征的高度不确定性并恢复尺度；给出尺度误差的分析与上界；自适应关键帧选择、AKAZE特征匹配与基于声像/海底子图的闭环检测。

**🔧 技术方法**

技术方法包括声纳–惯性融合、AKAZE特征与RANSAC匹配、IMU预积分、GTSAM因子图优化、基于声像与海底子图的闭环检测与一致性检验、尺度误差解析。

**📊 数据集**

数据集主要由石鱼仿真生成的10个不同地形的AUV航迹（共10,034步）和实地10次USV任务（BlueView 900kHz/2.25MHz、Oculus 750kHz/1.2MHz）组成。

**📈 对比分析**

通过与GNSS或LiDAR–IMU参考轨迹对比，在仿真中10,034步的尺度误差均值与模型预测误差差距≤1.6%；实地平均100m位移误差为0.7–4.2%（平静水）及7.7–9.4%（波浪环境）；闭环后平均误差从5.55 m降至3.35 m。

**⚠️ 局限性**

局限性：需可见海底，海底纹理不足时关键帧间隔增大；对倾斜或粗糙地形的误差上限仍有限；在持续平滑运动中IMU尺度难以辨识，需海底高度；未在深水AUV实地验证。

---

## 563. When Does Looking Closer Pay Off? Simple Foveated ViTs for Fine-Grained Recognition

**arXiv ID:** 2611.27788 | [PDF](https://arxiv.org/pdf/2611.27788v1)

**作者:** Boris Kriuk `[一作]` (Hong Kong University Of Science And Technology), Mohamed Sobhy `[通讯]` (Hong Kong University Of Science And Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种简易的 foveated Vision Transformer（FoVT），通过先用低分辨率 224px 视图做快速“glance”，利用该视图的类注意力挑选关键网格，在高分辨率 448px 图像中仅重新嵌入这些关键网格进行细化，最终再用同一 backbone 进行分类。

**💡 创新点**

核心创新在于：①只需一次 backbone 前向和一次注意力取样，无需额外策略网络、门控或多阶段训练；②使用辅助的 glance loss 使低分辨率预测可直接作为早期退出；③通过简单的 confidence gate 在推理时跳过最自信的图片，实现可控计算预算；④在多任务、两种种子、26 个配置的严谨实验中验证了其有效性，并对 CF‑ViT 进行了直接比较。

**🔧 技术方法**

使用 DINOv2 ViT‑S/14 backbone、全局分类 token 的 last‑block 类注意力、Top‑K 选择、再嵌入原始高分辨率特征、交叉熵 + auxiliary glance 损失、标签平滑、AdamW 优化器、cosine 退火等技术；实现了 token 重新排列、上下文不共享、可训练的尺度嵌入等。

**📊 数据集**

在五个细粒度数据集上评估：CUB‑200‑2011、NABirds、FGVC‑Aircraft、Stanford Cars、Stanford Dogs。

**📈 对比分析**

与统一 224/336/392/448 分辨率 ViT 以及官方 CF‑ViT 端口进行 paired bootstrap 置信区间比较；FoVT 在 0.56× 标准 448px 训练成本下，精度与 uniform 448px 无显著差异；在 336px 预算下，通过 confidence gate 可保持相同精度；在 NABirds 与 Aircraft 上实现了显著提升（+0.29 / +0.69 之间）；其优势主要来自 glance loss，实际高分辨率细化对精度提升有限。相比 CF‑ViT，FoVT 省去 feature reuse、distillation loss、selection schedule，性能相当。

**⚠️ 局限性**

主要限制包括：只使用单一 ViT‑S/14 backbone 与两种随机种子，未充分评估大模型或更多种子；CF‑ViT 的 port 采用不同训练 recipe 与硬件，结果可复现性受限；gate 需要预先收集测试池，实际部署时 batch compaction 的延迟未测；MAC 计数排除了 patch embedding、gather 等开销，真实壁钟时间差异有限；训练‑测试分辨率不匹配导致部分数据集（如 Aircraft）对预算敏感；验证集分割过小，未能提供稳健的部署前决策。

---

## 564. Evaluating Prompting Strategies for Zero-Shot Multilingual Aspect-Based Sentiment Analysis with Large Language Models

**arXiv ID:** 2611.27807 | [PDF](https://arxiv.org/pdf/2611.27807v1)

**作者:** Jakub Šm{í}d `[一作]`, Pavel Kr{á}l `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究系统评估了七种开源大型语言模型在12种语言、3个领域（餐厅、酒店、笔记本）下的零样本目标‑方面‑情感检测（TASD）性能。

**💡 创新点**

创新点在于①首次在多语言、多领域上对开放权重LLM进行零样本TASD基准；②比较密集与稀疏MoE架构、四种提示范式，并量化生成开销；③通过细粒度错误分析揭示召回瓶颈与多步拆解缺陷。

**🔧 技术方法**

使用的技术包括：指令调优的开放权重LLM（LLaMA‑3.1、Gemma‑4、Qwen‑3.6/3.8）、四种提示范式（单通道、CoT、两步、三步+精炼）、vLLM推理、半精度/4bit量化与自研的提示模板。

**📊 数据集**

采用的公开数据集为M‑ABSA（包含英、西、法、德、俄、土、日、韩、越、泰、汉等12种语言的平行文本）以及捷克餐厅数据集。

**📈 对比分析**

实验显示，单通道提示在大多数模型上达到最高的微F1（如Qwen‑3.6‑27B在餐厅域达56.6%），且性能优于跨语言mT5基线；MoE模型普遍逊色；多步拆解导致词数提升1.4–2.4倍且F1下降。

**⚠️ 局限性**

局限性包括①缺失方面是主要错误来源，限制了整体召回；②多步提示在中等规模模型上引入噪声且生成成本高；③MoE模型因参数稀疏性无法充分捕获多语言对齐；④对输出格式的敏感性导致某些模型在解析时失分。

---

## 565. SLICR: Strategy Logic-Guided Multi-Agent Reinforcement Learning

**arXiv ID:** 2611.27865 | [PDF](https://arxiv.org/pdf/2611.27865v1)

**作者:** Arshia Rafieioskouie `[一作]`, Borzoo Bonakdarpour `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在部分可观测、对抗性多智能体强化学习环境中，利用战略逻辑(SL)指定的形式化规范来训练代理，使其在面对任意对手策略时能够最大化满足规范的概率。

**💡 创新点**

核心创新在于：①将SL规范的布尔满足关系转化为鲁棒性优化目标；②通过“屏蔽”机制过滤非理性对手策略；③利用编码-解码网络学习对手策略的潜在嵌入，使存在量化的响应策略可在局部观测下推理；④在CTDE框架下训练并实现去中心化执行。

**🔧 技术方法**

技术手段包括：战略逻辑(SL)与线性时序逻辑(LTL)的组合；鲁棒性量化语义；屏蔽网络；Encoder-Decoder与未来动作预测；使用强化学习算法（IPPO、MAPPO、A2C、PPO）在CTDE模式下训练响应函数；对抗性策略生成与混合。

**📊 数据集**

评估数据集：Predator‑Prey、Google Research Football（R2S、R2S‑DEF、R2S‑4DEF）、SMAC‑HARD（3m、2s3z、3s_vs_3z）等多种对抗性地图与策略集。

**📈 对比分析**

与现有对手建模基线LIAM以及不同量化子句（∀*∃* vs ∃*∀*）进行对比。实验显示在PP、GRF、SMAC‑HARD三大基准上，本文方法相较于LIAM平均提升约30%–90%的关键指标（碰撞率、进球数、胜率），并在包含高比例非理性策略时表现出更鲁棒的优势。

**⚠️ 局限性**

局限性：仅支持至多一次量化子句交替（∀*∃*或∃*∀*）的SL子句；理论保证依赖信息保留的潜在嵌入，实际实现下在部分可观测场景中难以完全满足；对更复杂的量化结构或非理性策略类型的适应性有限。

---

## 566. An Autonomous Adaptive Learning System for Robotic Grasping in the Circular Factory

**arXiv ID:** 2611.27782 | [PDF](https://arxiv.org/pdf/2611.27782v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 567. Search-RSI: Expanding Search Worlds Along Multimodal Agents' Capability Frontier

**arXiv ID:** 2611.27780 | [PDF](https://arxiv.org/pdf/2611.27780v1)

**作者:** Chengchao Lv `[一作]` (Hunan University), Wanli Chang `[通讯]` (Hunan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了递归自我改进框架 Search‑RSI，在线扩展搜索世界，并通过当前策略诊断能力前沿、生成相应任务与证据，随后决定这些任务是用于强化学习还是自示范。

**💡 创新点**

创新点在于将搜索世界视为可随模型成长而扩张的训练变量；通过前沿诊断、同步支持注入、策略引导录入与自示范实现无外部教师的递归改进，解决了不可达负样本、前沿不匹配和零信号问题。

**🔧 技术方法**

使用多模态检索代理训练（RL+SFT）、Hop‑anchored GRPO优化、任务单元化（跳跃深度、关系稀疏度、干扰密度）、同步索引注入、短推测路由及提示引导自示范等技术。

**📊 数据集**

基于 SearchEyes 知识图谱生成任务，并在九大多模态搜索基准（SimpleVQA、VDR、MMSearch、LiveVQA、BC‑VL、FVQA、InfoSeek、BC‑V3、VisBrowse、VisSearch）上进行实验，使用 Qwen3.5‑9B/27B 模型。

**📈 对比分析**

与静态世界增长、固定池、EvolveSearch 等基线在同等任务预算和 RL 步数下比较，Search‑RSI 在平均准确率上提升约 10 分，超过 EvolveSearch 约 9 分，尤其在长链任务上显著提升，同时一致性提高、不可达负样本率下降。

**⚠️ 局限性**

局限性包括仍依赖离线知识图谱，无法覆盖实时网络；自示范需使用 privileged 提示，难以在真实部署中直接使用；递归过程可能在前沿耗尽时收敛；对高维干扰实体的处理仍有限，且未在真实 Web 环境中验证。

---

## 568. ENC-QA: A Diagnostic Benchmark for Vision-Language Models on Electronic Navigational Charts

**arXiv ID:** 2611.27858 | [PDF](https://arxiv.org/pdf/2611.27858v1)

**作者:** Jiyoon Pyo `[一作]` (University of Minnesota), Dalton Lunga `[通讯]` (Oak Ridge National Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `da1b1a89-583a-4b57-9c81-478778569bec` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a2602d71-93ab-4bad-974b-672788df8193` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6215c339-3735-4be3-8a07-5bbb7004712d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `a4b10f5d-130b-4e77-9367-6469ec621899` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于 NOAA ENCs 的海图诊断性基准，评估 VLM 在符号识别、空间推理和航行决策中的能力，并提供进阶案例分析。

**💡 创新点**

提出了分层能力评估框架，包含进阶问题子集以关联低层感知与空间推理与高层决策；同时引入 LLM-as-Judge 自动语义评估。

**🔧 技术方法**

使用零样本 VLM（如 Qwen、InternVL、KimiVL 等）以及 OpenCPN 渲染、LLM 生成评判等技术。

**📊 数据集**

数据集来自 NOAA ENCs 的矢量和栅格化图像，包含符号、文本、属性及连续更新版本的变更描述。

**📈 对比分析**

与 10 个公开权重 VLM 进行零样本对照；平均准确率在各子任务仅 10–60%，变更重要性判断接近随机；在提供感知+空间证据时提升 23.5%。

**⚠️ 局限性**

主要局限在细粒度符号识别、基于特征的空间推理以及将海图知识整合到决策中的不足；即使给出证据，准确率仍不足 90%。

---

## 569. Weave: Routing-Adaptive Runtime Warp Scheduling for MoE Megakernels

**arXiv ID:** 2611.27765 | [PDF](https://arxiv.org/pdf/2611.27765v1)

**作者:** Ziyu Huang `[一作]` (Shanghai Jiao Tong University), Jingwen Leng `[通讯]` (Shanghai Jiao Tong University)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一种路由自适应的Mixture‑of‑Experts（MoE）megakernel，在专家并行（EP）环境下通过动态warp级别的空间与时间调度实现通信与计算的持续重叠。

**💡 创新点**

在SM与warp层面实时调整通信/计算warp比例，使用基于全局队列的动态warp角色切换消除头尾中间空隙，并采用轻量级设备内成本模型联合选择最优空间与时间策略。

**🔧 技术方法**

利用CUDA/C++、HIP、Tensor Core、TMA、TMEM、全局队列调度、成本模型、路由统计、跨GPU P2P通信、RDMA与xGMI等技术。

**📊 数据集**

在六大LLM模型（如Qwen3.8‑2.4T、MiniMax‑M3、Qwen3.5‑397B等）上，以ShareGPT真实路由与随机权重/激活进行评测，S值覆盖128到16K。

**📈 对比分析**

与DeepGEMM‑MegaMoE、FlashInfer、SGLang、mKernel、FlashMoE、NanoFlow、RCCL+AITER等基线比较，8SM103上平均提升1.30倍（单层最高1.97倍），8SM90平均提升1.24倍，16CDNA3平均提升1.44倍，端到端解码亦提升1.08倍。

**⚠️ 局限性**

仅在NVIDIA与AMD GPU上验证，未针对CPU或其他加速器拓扑测试，对极端高并行度或特殊内存架构的适配仍未知。

---

## 570. Pre-Crawl Prioritization and Seed Classification for Large-Scale Web Archiving

**arXiv ID:** 2611.27743 | [PDF](https://arxiv.org/pdf/2611.27743v1)

**作者:** Akshith Garapati `[一作]` (Purdue University), Sawood Alam `[通讯]` (Internet Archive)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `5b4c1114-4a70-478e-9921-2514ee03850d` `a2602d71-93ab-4bad-974b-672788df8193` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一套针对大规模Web归档的预爬取优先级框架，利用URL词法特征、动态域名声誉与成本感知的深度分析，生成高效的抓取排序列表。

**💡 创新点**

创新点包括：①基于42维URL词法特征的LightGBM回归模型；②使用指数移动平均（EMA）动态维护域名声誉、易变性和多样性；③预算受限的域名层级抽样，仅对一小部分URL做昂贵分析；④改进的对数折扣NDCG函数，使评估更符合大规模抓取的实际需求。

**🔧 技术方法**

使用技术：LightGBM、EMA、HTTP/ CDX API、Apache Spark分布式处理、对数折扣NDCG评估、预算约束的域名分层采样。

**📊 数据集**

数据集：IndexNow协议约2.55千万URL、MediaCloud 50万URL；训练集为相同规模的历史URL；同时使用TLD权重列表、垃圾域名列表等辅助信息。

**📈 对比分析**

通过与未优化的基于TLD+手工权重的启发式系统对比，评估指标包括Score Lift、Spam URLs、Clean Rate、Score Std. Dev.、Avg. Confidence。优化系统Score Lift提升+254%，垃圾URL下降76%，Clean Rate提升至99.63%，标准差更宽，平均置信度适中，表明更准确、可校准的优先级预测。

**⚠️ 局限性**

局限性：①模型依赖历史训练数据，可能对最新网页变化响应不足；②成本采样可能错过部分高价值URL；③仅利用URL词法和域名声誉，未考虑深层内容特征；④需持续维护模型以跟随Web动态；⑤对实时大规模动态更新的可扩展性尚未全面验证。

---

## 571. Refrain: Learning Multi-Shot Consistency from Inconsistent References for A$^2$O Video Generation

**arXiv ID:** 2611.27806 | [PDF](https://arxiv.org/pdf/2611.27806v1)

**作者:** Hongru Yan `[一作]` (University of Hong Kong), Shenghua Gao `[通讯]` (University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了统一的多镜头视频生成框架Refrain，能够在任意参考（图像/视频）、任意时长的条件下，生成视觉风格一致且保持内容与时间边界完整的视频；

**💡 创新点**

①通过自监督重建任务，让模型在独立编辑的跨镜头参考上恢复原始连贯视频，从而学习跨镜头风格一致性；②提出A^2O（Any Reference, Any Duration, One Timeline）统一生成方案，消除额外注意力/记忆模块；③提出Shot‑Debiased Consistency (SDC)，用大规模真实视频库校准内容‑风格差异，客观评估跨镜头一致性；

**🔧 技术方法**

基于扩散模型（HunyuanVideo‑1.5 DiT）实现；采用自监督重建损失、稀疏参考映射、RoPE位置偏移、参考编辑器（FLUX.1 Kontext、Wan‑2.1 等）和共享画布；

**📊 数据集**

使用DurMix‑300K（≈30万对自监督编辑参考与原始视频），原始 2,150,758 网视频进行 shot 与 style 注释，外部 IP（1,720 万）和 Stock（6,760 万）视频库共 86.9M 视频用于 SDC 校准；

**📈 对比分析**

与 StoryMem、Phantom、RefAlign、VACE 等现有多镜头生成方法比较；Refrain 在 SDC（0.315 vs 0.301/0.304），转场 MAE（0.199 s）、文本对齐（CLIP‑T 0.222，VideoAlign TA −0.221）以及参考保真度（语义 0.737，主体 0.671，背景 0.674）上实现领先；

**⚠️ 局限性**

仍受限于参考编辑冲突的极端情况、超长/极短参考时的细节保持、对极端风格变化的鲁棒性不足；依赖大量编辑数据和真实视频库，未能覆盖所有可能的跨镜头差异；纯 T2V 仍需进一步提升一致性。

---

## 572. MemOnce: A Scalable Memory Architecture for Large Language Models

**arXiv ID:** 2611.27973 | [PDF](https://arxiv.org/pdf/2611.27973v1)

**作者:** Chang Nie `[一作]` (Nanjing University), Caifeng Shan `[通讯]` (Nanjing University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文未提供具体研究内容或实验细节，无法确定其研究目的与任务。

**💡 创新点**

因缺少创新描述，无法识别任何创新点。

**🔧 技术方法**

未说明所使用的技术或方法。

**📊 数据集**

未提及任何数据集。

**📈 对比分析**

未给出方法对比或性能指标。

**⚠️ 局限性**

本文信息缺失导致无法评估局限性，整体缺乏可验证的研究内容。

---

## 573. Model Carving: Training Transformers That Optimize Their Own Topology

**arXiv ID:** 2611.27962 | [PDF](https://arxiv.org/pdf/2611.27962v1)

**作者:** Kaden Zheng `[一作]` (Harvard University), Pratyusha Sharma `[通讯]` (New York University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

训练Transformer时，同时学习权重和可切换的子层门，使模型在训练过程中自动裁剪不必要的子层，从而得到更小、更高效的网络结构。

**💡 创新点**

创新点在于将架构搜索嵌入训练本身，用可学习的标量门和在线估计的“压力”动态决定哪些子层被移除，而不需要单独的NAS或后期剪枝。

**🔧 技术方法**

采用可学习的门控子层、DenseFormer式的可学习连接、基于学习率曲线的压强估算以及梯度下降的门控更新，构成了Model Carving框架。

**📊 数据集**

使用FineWeb-Edu等通用语言建模数据集，并在符号推理任务（S_5、ListOps、modular addition）上做了专门实验。

**📈 对比分析**

与从同一训练预算得到的完整深度模型、等推理开销的浅层模型、蒸馏模型以及剪枝模型相比，Carved模型在验证损失、CORE‑22、生成任务的pass@k以及解码吞吐量上均优于对手，并在相同推理 FLOPs 下比大模型更高效。

**⚠️ 局限性**

局限性包括：规模有限（最高2.7B参数）、在冷却阶段未考虑压强调整、对初始门值和压强参数高度依赖、只裁剪整层而非更细粒度、目前仅在训练中剪枝，缺乏对预训练模型的迁移或动态适配。

---

## 574. Ranking-Preserving Coreset Selection for Efficient Evolutionary Hyperparameter Optimization

**arXiv ID:** 2611.27959 | [PDF](https://arxiv.org/pdf/2611.27959v1)

**作者:** Mamoona Tasadduq `[一作]` (National University of Computer and Emerging Sciences), Maryam Bashir `[通讯]` (National University of Computer and Emerging Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种排名保持的核心集（RPC）选择框架，用于加速进化式超参数优化。

**💡 创新点**

创新点在于直接以Spearman相关系数衡量完整数据与子集产生的超参数排名一致性，并将子集大小作为惩罚项，进而使核心集既小又能保留超参数搜索的秩序。

**🔧 技术方法**

核心技术包括演化优化（遗传算法、CHC、PSO）用于子集搜索，Spearman相关系数用于排名匹配，和一系列实例/特征选择策略（梯度匹配、子模、聚类）做对比。

**📊 数据集**

实验使用了11个基准数据集：7个UCI表格数据、2个文本情感数据（IMDb、TweetEval）和2个图像分类数据（CIFAR‑10、DermaMNIST）。

**📈 对比分析**

与全数据HPO、相同规模随机子集和梯度匹配子集比较时，RPC在大多数数据集上能保持与全数据相近的测试准确率，且在核心集上实现了1.14×–8.80×的HPO阶段加速，同时在10个anchor配置和20个held‑out配置上显著提高Spearman相关系数。

**⚠️ 局限性**

局限性包括一次性构建核心集的计算开销、对anchor配置多样性的依赖、以及在图像任务上的排名保持泛化表现不如随机子集。

---

## 575. The Optimal Sample Complexity of Agnostic PAC Learning for Partial Concept Classes

**arXiv ID:** 2611.27946 | [PDF](https://arxiv.org/pdf/2611.27946v1)

**作者:** Liang-Yu Zou `[一作]` `[通讯]` (Aarhus University), Liang-Yu Zou (Aarhus University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76`

**🎯 论文内容**

提出了一种确定性无偏（agnostic）PAC学习算法，针对具有有限 VC 维的部分概念类实现了最优样本复杂度，匹配下界，解决了之前存在的对数阶缺口；

**💡 创新点**

核心创新在于：①推广 Rademacher 复杂度至部分概念类并利用链式技术获得 O(√{d/n}) 的上界；②设计了基于条件均值的坐标级逼近序列，解决了传统 ε‑disambiguation 在欧几里得距离上的问题；③提出了 leave‑one‑out 预测器（改进自 minimax forecaster），通过对标签随机扰动的积分关联到 Rademacher 复杂度；④将 leave‑one‑out 结果聚合并阈值化，构成最终的 PAC 学习器；

**🔧 技术方法**

主要技术手段包括：通用 Rademacher 复杂度、链式（chaining）与 ε‑disambiguation、坐标级条件均值逼近、leave‑one‑out 预测、聚合与阈值化、Hoeffding 以及 martingale 论证；

**📊 数据集**

本工作为理论性质的论文，未使用实测数据集，而是通过抽象分布 𝒟 在 ℝ×{-1,+1} 上的 i.i.d. 样本进行分析；

**📈 对比分析**

与现有研究（如 Alon 等 2021、最近的聚合算法）相比，本算法在样本复杂度上达到了最优 Θ((d+log(1/δ))/ε²) 的下界，实现了理论上最优性能；

**⚠️ 局限性**

限制包括：仅适用于 VC 维为有限的部分概念类；当 d=0 时需单独处理（该情况本质上是退化为全局常数预测）；此外，算法在实现上需枚举 2^n-1 条染色序列，计算成本在理论上可接受但在大规模实际数据中可能不易直接实现。

---

## 576. Can Vision-Language Models Synthesize Executable Game-Playing Programs?

**arXiv ID:** 2611.27741 | [PDF](https://arxiv.org/pdf/2611.27741v1)

**作者:** Yao Tong `[一作]` (University of Chinese Academy of Sciences), Jianqiang Huang `[通讯]` (Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PlayScriptBench benchmark，评估 VLM 在仅通过一次推理即可生成完整游戏控制脚本的能力，并设计四阶段流水线（探索、感知、API 构造、策略组合）实现端到端脚本合成与评测。

**💡 创新点**

创新点在于：① 将传统逐帧交互模式转变为一次性脚本合成，解耦推理成本与游戏时长；② 通过四阶段流水线将视觉感知、接口映射与策略推理拆分，提供细粒度故障诊断；③ 在浏览器沙盒中暂停游戏运行，消除实时推理延迟对成绩的影响。

**🔧 技术方法**

采用 VLM 生成 OpenCV‑style 感知函数、GameAPI 类以及策略函数，利用浏览器沙盒捕获帧、执行脚本；评测时将 VLM 产出的感知结果与 Oracle 结果交叉比较；对每个阶段使用自动化脚本评估准确度。

**📊 数据集**

数据集为 16 个确定性、边界状态的 HTML5 游戏（跑酷、街机、益智、平台），涵盖四类游戏，所有游戏均在统一浏览器沙盒中运行。

**📈 对比分析**

对 13 款前沿 VLM 进行评测，分别计算 Perception、API、Observation、Strategy 与 Joint 分数。结果显示 Perception 能力虽有所提升，但大多数模型在 Strategy 及 Joint 上得分仍低于 0.3，表明视觉理解与闭环控制之间仍存在显著差距；与传统逐帧 VLM 推理对比，脚本化方法在调用次数和 token 消耗上大幅降低，且在多种游戏中获得更高或相近的游戏分数。

**⚠️ 局限性**

局限性包括：① 只针对确定性、有限状态游戏，无法推广到开放世界或随机化游戏；② 合成脚本缺乏在线自适应机制，无法在运行时根据实时视觉反馈动态调整策略；③ 评测仍依赖手工编写的 Oracle 感知与 API，构建与维护成本较高。

---

## 577. DiTraR: Diverse Scorers and Transcript-Guided Re-ranking for Video Corpus Moment Retrieval

**arXiv ID:** 2611.27931 | [PDF](https://arxiv.org/pdf/2611.27931v1)

**作者:** Mohammad Alijanpour Shalmani `[一作]` (University of Central Florida), Ali Amini `[通讯]` (Kaunas University of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的视频语义检索框架DiTraR，能够在未裁剪的视频集合中同时精确定位时间区间并选取正确视频；

**💡 创新点**

创新点在于①使用全查询的 witness‑interval scorer 通过查询分解、门控与布局在全格点上精确计算能量并进行全搜索；②采用多模型能量平均的 ensemble 来降低单模型误差；③利用冻结的指令调优语言模型做 transcript verifier，对高排名视频的对话进行验证与重新排序；

**🔧 技术方法**

技术包括：查询分解为 probes（主题、动作、客体等）并对每个 probe 在视觉、字幕、说话人三通道上做相似度计算；基于门控与布局的加权 soft‑max 能量；对每个视频的时间格点做一次 prefix‑sum 计算实现 O(|D|·T·|L|) 的 exact search；多模型平均（product‑of‑experts）以及 frozen LLM（Qwen3.5‑27B）进行文本验证；

**📊 数据集**

使用 TVR（TV Show Retrieval）数据集，包含 109k 询问、21.8k 视频，训练/验证/测试分布分别为 87.1k/10.9k/（私有）;

**📈 对比分析**

与现有方法（XML, ReLoCLNet, HERO, CONQUER, SQuiDNet, DeMUL 等）对比，DiTraR 在验证集 IoU 0.5/0.7 的 R@1 分别达到 24.36 / 13.55，超过 DeMUL（21.37 / 11.92）和 GenSpan（12.12 / 10.12），整体排名第一；在 R@100 方面也取得 51.69% IoU 0.7，显著优于 39.79% 的最佳对手；

**⚠️ 局限性**

局限性包括：对纯视觉查询的提升有限（仅 1–2 分）；验证器主要依赖对话信息，无法处理无字幕或对话稀缺的场景；模型对训练集的依赖较大，可能在不同语料下表现下降；同时仍只有约 50% 的查询能让模型准确把握正确视频，局部定位仍有改进空间。

---

## 578. The Power of Sampling for Finding Far-Apart Bases in Matroids

**arXiv ID:** 2611.27934 | [PDF](https://arxiv.org/pdf/2611.27934v1)

**作者:** Mayank Goswami `[一作]` (CUNY), Adarsh Srinivasan `[通讯]` (Rutgers University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究了基交换图的直径与多样性问题，并提出了对基集合进行最大最小多样性选择的常数因子近似算法。

**💡 创新点**

首次在任意基的框架下实现了k多样性问题的1/6近似，并揭示了该问题在参数化层面的W[1]-难度。

**🔧 技术方法**

主要采用基交换图结构分析、LP圆整技术以及远点基问题的1/3近似算法。

**📊 数据集**

本文不依赖实际数据集，所有实验与分析均基于理论构造与计算复杂性证明。

**📈 对比分析**

相较于随机采样只能得到Θ(log k)近似，提出的算法在理论上实现了1/6的近似比，在实验中表现出明显优于随机采样的性能。

**⚠️ 局限性**

该方法对直径估计的下界仅能给出1/16的常数因子，且在特殊基类如常数arboricity的图基上只能得到1/4近似，缺乏更精细的参数化近似结果。

---

## 579. Interpretable Tissue Architecture Complements Pathology Foundation Models

**arXiv ID:** 2611.27881 | [PDF](https://arxiv.org/pdf/2611.27881v1)

**作者:** Zeeshan Shahid `[一作]` (COMSATS University Islamabad), Hammam M. AlGhamdi `[通讯]` (University of Jeddah)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

构建了139维可解释的细胞到社区组织学特征表示，并与CTransPath、PLIP等基础模型进行特征级融合。

**💡 创新点**

提出了多层次可解释的细胞-社区特征组，并系统评估其对ER/PR预测的互补性与性能提升。

**🔧 技术方法**

采用HoVer‑Net细胞分割、图形网络构造、Leiden社区检测、图中心性与谱特征计算，并结合多实例学习（Gated Attention、ALR‑Mamba、CLAM）与基础模型嵌入。

**📊 数据集**

使用公开的BCNB（乳腺癌）和LungHist700（肺癌）两个数据集进行实验。

**📈 对比分析**

通过5折病人独立交叉验证对比单独特征、融合以及多种MIL架构，平均ROC‑AUC提升约5–10%，最高达0.9144（ER）和0.8360（PR）等。

**⚠️ 局限性**

主要局限包括特征组合选择可能导致乐观偏差、图构造参数和社区检测对结果敏感、仅在公开数据集上验证、缺乏独立外部验证与统计显著性检验。

---

## 580. SUAF: Spectral Universal Advantage Features for Zero-Shot Policy Improvement

**arXiv ID:** 2611.27919 | [PDF](https://arxiv.org/pdf/2611.27919v1)

**作者:** Huaze Tang `[一作]` (Tsinghua University), Wenbo Ding `[通讯]` (Tsinghua University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文提出一种奖励无关的 Spectral Universal Advantage Feature (SUAF) 表示，用以捕捉当前动作对未来结果的统计依赖，并通过正向时滞 Canonical Dependence Matrix 进行分解。

**💡 创新点**

创新点在于将即期奖励对比显式保留，构造行动中心正向时滞 CDM，得到可直接对延迟优势进行谱近似的 SUAF，并证明其在平均和极大误差下的最优性。

**🔧 技术方法**

方法结合 Soft-HGR 目标与双向时序差分（Bidirectional TD）学习，能够从无奖励的转移数据中学习低秩特征并估计谱值。

**📊 数据集**

实验使用了离散网格、四室、出租车等有限状态环境，以及连续控制任务 Walker、Cheetah、AntMaze、Cube 等。

**📈 对比分析**

与 Laplacian、BYOL‑γ、TD‑JEPA、ICVF、HILP、Forward‑Backward 等基线相比，SUAF 在 OOD 任务上往往实现最高或第二高的零射策略改进率和回报/成功率，显示出强大的泛化能力。

**⚠️ 局限性**

局限性包括在某些 ID 任务中性能不占优势、特征学习误差导致与理论截断误差存在差距，以及对更大策略迁移或非平稳环境的适应仍需改进。

---

## 581. Beyond Visual Fidelity: Comparing Categorical and Continuous World Models

**arXiv ID:** 2611.27870 | [PDF](https://arxiv.org/pdf/2611.27870v1)

**作者:** Sarah Parisot `[一作]` (Microsoft), Katja Hofmann `[通讯]` (Microsoft)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

对比了在控制环境下的分类（离散）与连续（流匹配）视觉世界模型在学习动力学上的表现；

**💡 创新点**

在无信息丢失的视觉编码下，证明分类预测在训练计算量更低时能获得更高的轨迹有效性，并揭示连续模型在校准与长期有效性上的缺陷；

**🔧 技术方法**

使用Transformer架构，交叉比较了基于下标预测的next-token模型与基于迭代更新的whole-frame模型，并引入了条件流匹配作为连续模型的生成机制；

**📊 数据集**

采用自定义的 Pixel Pac‑Man（2D）和 3D Maze（部分可观测）两个完全可控的环境，分别提供离散与连续观测编码；

**📈 对比分析**

通过比较下一帧有效性、长序列rollout有效性、校准（总变差）以及视觉质量（FVD、像素精度）等指标，发现分类模型在有限计算下的rollout有效性和计算效率优于连续模型；连续模型在采样推理时能更好校准，但整体长期性能不如分类模型；

**⚠️ 局限性**

仅在极简控制环境中验证，缺乏对复杂真实世界场景的通用性；同时模型的比较受限于单一tokenizer和Transformer结构，未探究更大规模模型与多模态输入的影响。

---

## 582. Interactive Semi-Parametric On-Track Learning for Vehicle Dynamics in Autonomous Racing

**arXiv ID:** 2611.27932 | [PDF](https://arxiv.org/pdf/2611.27932v1)

**作者:** Jungbin Kim `[一作]` (Ulsan National Institute of Science and Technology), Cheolhyeon Kwon `[通讯]` (Ulsan National Institute of Science and Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种交互式半参数学习框架，利用赛道实时数据同时调优车辆动力学的物理参数和残差模型；

**💡 创新点**

创新点在于将物理参数与非参数残差在同一共享预测误差目标下交替递归更新，并通过自适应学习率与MSE模型选择保证目标非递增并收敛；

**🔧 技术方法**

使用了非线性最小二乘法（NLS）进行参数估计，稀疏高斯过程（GP）进行残差学习，结合可变学习率与模型选择策略；

**📊 数据集**

在ETH Zurich赛道的20个随机参数场景的双轨模型仿真数据以及1/10比例室内赛道的真实车辆实验数据；

**📈 对比分析**

与基准方法（原始参数、仅NLS、NN辅助NLS、仅GP、非递归交互）在相同数据集下进行比较，结果显示该框架在一次预测误差、跟踪误差和赛道完成时间上均显著优于基准；

**⚠️ 局限性**

局限在于目前为离线学习，需预先收集完整赛道数据，且在极限操控条件下对车辆参数的非线性耦合仍有进一步提升空间。

---

## 583. Perceived Differences in Activity and Experience Between Developing Software With and Without AI

**arXiv ID:** 2611.27947 | [PDF](https://arxiv.org/pdf/2611.27947v1)

**作者:** Rubash Mali `[一作]` (Utah State University), John Edwards `[通讯]` (Utah State University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

通过对48名软件工程师的问卷调查，收集AI辅助与非AI编程的对比评分，分析了两种情境下的活动与体验差异；

**💡 创新点**

首次系统性量化AI辅助编程在多项活动和体验上的差异，并揭示经验梯度，即经验丰富的工程师感受到的差异更大；

**🔧 技术方法**

采用配对Wilcoxon符号秩检验、等效性检验及线性混合效应模型来评估差异；

**📊 数据集**

使用从大学学生、行业工程师及开发者社区收集的58份问卷（去除9份未使用AI者后48份）；

**📈 对比分析**

对每个26项进行配对差异比较，显著差异占12项，核心代码生成差异小，经验更丰富者差异更明显；

**⚠️ 局限性**

样本规模有限，非随机便利抽样，依赖自我报告且部分数据为回顾性，且对未使用AI者的预期与实际使用者区分；

---

## 584. Topology Hub Optimal Transport: Eliminating Long Context Dilution at Flash Speed

**arXiv ID:** 2611.27951 | [PDF](https://arxiv.org/pdf/2611.27951v1)

**作者:** Dylan Forde `[一作]` `[通讯]`, Dylan Forde

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Topology Hub Optimal Transport (THOT)，一种稀疏注意力架构，用以在长上下文中解决容量稀释、有限步收敛、反向传播开销和自回归泄漏等问题。

**💡 创新点**

创新点在于：①有限支持容量门控（Sparsemax/阈值门）消除软max散射；②1+2级热解温度行闭式Sinkhorn调度消除二分图残差；③Exact One-Reference-Tile梯度实现低阶分块推导；④严格因果分块传输保证 0.0 未来泄漏；⑤BF16 Tensor Core Triton 核加速。

**🔧 技术方法**

使用了稀疏可分离注意力、边缘投影、三步Sinkhorn、半松弛Unbalanced OT、行闭正则化、BF16 Tensor Core Triton kernels 与精确梯度重构技术。

**📊 数据集**

实验数据集包括合成固定绑定检索、WikiText-103 4K/103 16K 文本以及极长序列 L=262,144/65,536 的长上下文评估。

**📈 对比分析**

与 Softmax、Sparsemax、Latent Softmax、FlashAttention 等基线对比，THOT 在 262K 长度零样本检索中达 97.4%（α=0）或 98.8%（α=0.05）准确率，显著优于 Softmax (3.5%)，且在 4K 位置下前向+梯度时间比 FlashAttention 快约 2–3×。

**⚠️ 局限性**

局限性：核实现尚未完整集成到多层模型，内存指标基于编译器估算；在高负载（R> K）时召回率下降；仅在单头/小规模实验验证，缺乏大规模全模型训练与评估。

---

## 585. Your Camera Can Hear You: Side-Channel Acoustic Eavesdropping via Android Camera2 Metadata

**arXiv ID:** 2611.27964 | [PDF](https://arxiv.org/pdf/2611.27964v1)

**作者:** Guangyuan Ji `[一作]` (Zhejiang University), Bingsheng Zhang `[通讯]` (Zhejiang University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文发现 Android Camera2 元数据泄露高频陀螺仪数据，绕过运动传感器速率限制和系统隐私开关，实现无权限的声学旁路窃听。

**💡 创新点**

创新点在于提出 Source-to-Leakage Learning（S2L）框架，利用物理约束的仿真前向模型生成虚拟泄露数据，实现跨设备、零定位的持续语音识别。

**🔧 技术方法**

采用物理仿真模块（结构共振、子奈奎斯特采样、量化噪声）结合 CNN–BiLSTM–CTC 语音识别网络；同时利用 LibriSpeech 公开语料进行训练。

**📊 数据集**

使用 LibriSpeech 960h 公开语音数据进行源数据训练，并在 6 台 800–1000 Hz 级别的 Android 手机上收集真实 Camera2 元数据进行测试。

**📈 对比分析**

与同类基于真实数据的 ISpyU 模型对比，S2L 在跨设备平均单词准确率从 25.9% 提升至 52.1%（提升 101.4%），且在所有评测机型上均超过 24 h 同机微调模型。

**⚠️ 局限性**

局限性包括对扬声器驱动的依赖、对手机摆放姿态敏感、无法处理低速（≤200 Hz）元数据以及对新硬件的物理参数仍需手动估计。

---

## 586. BioEmu2: Scalable Emulation of Biomolecular Dynamics

**arXiv ID:** 2611.27924 | [PDF](https://arxiv.org/pdf/2611.27924v1)

**作者:** Michael Gastegger `[一作]`, Frank Noé `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `09944146-298c-433e-89df-37255de463d7` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种名为BioEmu2的生成模型，能够在单个网络中实现高精度的原子级结构预测、平衡分布采样和时间演化，支持多组分蛋白、核酸、修饰残基、配体与离子等复杂体系；

**💡 创新点**

其创新点在于将序列推理与条件扩散生成器相结合，支持上下文并行推理、对接触和动态时滞的条件控制，并实现了通过增强采样和引导提升结构物理性与罕见态覆盖；

**🔧 技术方法**

核心技术包括基于AlphaFold3的序列主干（MSA + PairMixer）与双流扩散结构模块、EDM预处理、时间传播的隐式传输算子、以及物理性引导与Twisted Diffusion Sampler的采样增强；

**📊 数据集**

训练数据覆盖了实验（X‑ray、cryo‑EM、NMR）和合成结构，以及数千条分子动力学（NpT）轨迹，涵盖单体、复合体、配体结合等多种系统；

**📈 对比分析**

在FoldBench、Runs N’ Poses、单体与复合体的自由能景观以及时间相关自相关函数上与现有开放权重模型（AlphaFold、ESMFold、OpenFold、Protenix、Boltz等）相比，BioEmu2在结构准确性、能量误差（≈0.7–1.2 kcal/mol）和状态覆盖率方面处于领先或同等水平；

**⚠️ 局限性**

局限性包括对慢速动力学（如结合/解离、折叠）缺乏严格验证、对实验稀有态的描述仍需更大规模数据支持，以及在大体系中预测成功率下降、需要额外接触或指导信息。

---

## 587. Autonomous Parameter Surgery: How LLMs Implant Selective Backdoors into Language Models

**arXiv ID:** 2611.27920 | [PDF](https://arxiv.org/pdf/2611.27920v1)

**作者:** Xuyang Liu `[一作]` (Tsinghua University), Xibin Zhao `[通讯]` (Tsinghua University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

利用普通语言模型在没有梯度、层提示、外部语料等辅助的情况下，直接读取并修改目标模型权重，实现持久且选择性的后门注入。

**💡 创新点**

证明语言模型能够自主定位编辑位置、确定更新方式（稠密低秩或单神经元手术），并通过GPU反馈迭代优化，完成无梯度的权重改写；同时提出隔离工作区和原始参数/激活分离的安全评估框架。

**🔧 技术方法**

使用 DeepSeek Flash 低推理成本 LLM 作为编辑器；在受限工作区中提供 Inspect、GPU probing、Offline algebra、Checkpointing 四类工具；采用原始参数与激活分离、线性代数求解、GPU 探测反馈来构造更新；通过在新进程中重新加载模型验证效果。

**📊 数据集**

情感分类数据集：IMDb 和 Yelp，分别划分为可见开发集、隐藏验证集和确认集。

**📈 对比分析**

在 18 次实验（3 模型 × 2 域 × 3 复现）中评估攻击成功率、转移率、控制误差和整体能力；结果显示编辑域平均 ASR 99.2%，未见域转移 97.6%；清洁准确率提升仅 0.29pp，其他能力变化 ≤ 1.7pp；平均 API 成本 $1.74，耗时 16.5 分钟。

**⚠️ 局限性**

仅测试单一低推理预算的编辑器，未与人类设计的基线方法直接对比；仅覆盖两个英文情感分类数据集，未考察更广泛任务或多语言情况；可能受目标模型预训练信息影响，无法完全排除预训练对编辑的影响。

---

## 588. Metadata-Private Voice Communication over Tor using Post-Quantum Key Encapsulation

**arXiv ID:** 2611.27909 | [PDF](https://arxiv.org/pdf/2611.27909v1)

**作者:** Noah Zeinali Yazdi `[一作]` (Independent Researcher), Samuel Zeinali Yazdi `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了SAPEC系统，实现在Tor隐藏服务上进行端到端加密的实时语音通信。

**💡 创新点**

首次将后量子密钥封装（ML‑KEM‑1024）、匿名路由（Tor隐藏服务）和身份验证（Dilithium）集成到消费级嵌入式硬件的语音通信原型中。

**🔧 技术方法**

采用ML‑KEM‑1024做后量子密钥交换，Dilithium做身份验证，AES‑256对音频进行对称加密；在TLS之上加入自定义协议层；利用Opus编码、Python/CipherLink后台、Raspberry Pi、Tor守护进程等。

**📊 数据集**

未使用公开数据集，评估基于自建的45次对等节点实验，测量连接建立时间、单向延迟、RTT和音质。

**📈 对比分析**

相较于传统VoIP，SAPEC在相同地区Tor线路上平均单向延迟为355 ms、RTT 710 ms，连接建立时间约7.6 s；虽然延迟略高于ITU‑T G.114推荐阈值，但在隐私优先场景下可接受。

**⚠️ 局限性**

主要限制为：Tor线路的不可预测性导致偶尔出现高延迟（3次极端值）；TLS中使用的经典非后量子密钥交换仍为潜在弱点；系统缺乏全局流量分析抵抗能力；对低带宽网络的性能尚未充分验证。

---

## 589. Functional Depth Is Learned: Isolating the Computation That Layers Produce Only Together

**arXiv ID:** 2611.27911 | [PDF](https://arxiv.org/pdf/2611.27911v1)

**作者:** JaeSeong Kim `[一作]` (Semyung University), KyungTae Lim `[通讯]` (Korea Advanced Institute of Science & Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究Transformer预训练过程中层间协同的形成，提出一种基于包含-排除的精确分解方法来量化多层共同计算的“连通分量”，并跟踪其在Pythia、OLMo和Amber三大模型系列中的演化；

**💡 创新点**

创新点在于：①首次定义并计算多层级连通分量（写入相对连通分量），将层间交互从单层或双层提升到高阶；②引入写入相对归一化，量化连通分量相对最终层写入的大小；③通过在不同预训练阶段对连通分量的删除与随机方向删除进行对照，证明连通分量对模型输出具有因果影响；④将连通分量与层间表示相似性（CKA）对比，揭示两者互补却不等价的关系。

**🔧 技术方法**

技术方法包括：包含-排除归一化（Möbius变换）对层集的子集前向传递进行精确求和；写入相对连通分量计算；对每个层窗口进行2^k次子集前向传播；对删除连通分量与随机方向进行消融实验；使用Spearman相关性评估连通分量与CKA、消融效果的关联。

**📊 数据集**

使用公开的WikiText‑103文本语料库作为输入，针对Pythia（410M–6.9B）、OLMo（1B）和Amber（7B）三系列模型，在不同训练步骤（0–64k、4B–3T、≤1.26T tokens）下进行测评。

**📈 对比分析**

比较方法：通过删除连通分量与删除相同范数的随机方向计算困惑度（PPL）提升，发现连通分量删除的PPL提升显著高于随机删除（在所有35个模型–阶级组合中都更大，差距可达数百倍）。此外，将连通分量与CKA进行块级消融对比，评估二者在预测消融成本上的一致性；在大部分模型中连通分量与消融效果正相关，而CKA相关性在Amber上不稳定。

**⚠️ 局限性**

局限性包括：①仅测量相邻层窗口的连通分量，未覆盖跨层交互；②归一化仅相对于最终层写入，不考虑前层写入的规模；③高阶（k>6）连通分量计算成本极高，需近似；④缺乏对功能深度与模型性能因果关系的直接验证，需要进一步实验。

---

## 590. Decision Columns as an Index: Interpretable Embeddings for Explainable Entity Matching and Search

**arXiv ID:** 2611.27910 | [PDF](https://arxiv.org/pdf/2611.27910v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871`

---

## 591. A Methodologically Rigorous Benchmark of Linear Regression,RNN, and LSTM for S&P 500 Forecasting:Identifying Sources of Performance Inflation in Financial DeepLearning

**arXiv ID:** 2611.27978 | [PDF](https://arxiv.org/pdf/2611.27978v1)

**作者:** Hari Sitaula `[一作]` (Montana Technological University), Joseph Opitz `[通讯]` (Montana Technological University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对2006–2025年S&P 500每日收盘价进行一日后预测，系统比较线性回归、传统RNN和LSTM三类模型，并在统一的预处理和时间顺序评估框架下，审计并纠正深度学习预测中常见的三大结果膨胀做法。

**💡 创新点**

首次对波形去噪、缺失持久性基准、最佳‑复现报告三种结果膨胀来源进行全面审核，提出校正后的评估协议，并证明在此协议下所有模型均未能突破简单持久性基准。

**🔧 技术方法**

使用线性回归（OLS、Ridge、Lasso、Elastic Net、Bayesian Ridge）、纯RNN和LSTM网络；采用滑动窗口序列化、标准化、时间顺序划分；进行30次独立复现；以RMSE、MAPE和Pearson相关度评估，并在必要时使用Diebold‑Mariano检验。

**📊 数据集**

利用2006–2025年每日S&P 500 OHLC、宏观变量（VIX、USDX、EFFR、UNRATE、UMCSENT）以及技术指标（MACD、ATR、RSI）构成的多变量特征集。

**📈 对比分析**

采用80/20时间顺序划分，30次复现求均值与标准差；与持久性基准（y_{t+1}=y_t）比较；最佳神经网络（三层RNN）RMSE为53.97，仍高于基准53.16；所有模型的平均性能均未能超越持久性基准，差距在0.8–5.6指数点之间。

**⚠️ 局限性**

仅预测日收盘水平，缺乏可学习的结构；特征集与窗口长度有限；未探索更易学习的目标（如收益率、波动率、方向）或更高频数据；结果对市场结构变化的稳健性尚未验证。

---

## 592. QuantStream3D: Geometry-Aligned Low-Bit KV Cache Quantization for Streaming Visual Geometry Transformers

**arXiv ID:** 2611.27901 | [PDF](https://arxiv.org/pdf/2611.27901v1)

**作者:** Zunhai Su `[一作]`, Ngai Wong `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了面向流式视觉几何Transformer的KV量化框架QuantStream3D，兼顾量化精度分配与模型自适应；

**💡 创新点**

创新点在于结合几何感知的混合精度KV量化与几何对齐的量化感知蒸馏，使得低位量化下仍能保持甚至提升几何准确度；

**🔧 技术方法**

采用了按通道分组的KV量化、Q/K旋转、双重精度分配、与冻结原精度教师的蒸馏训练，以及高效的packed KV融合执行；

**📊 数据集**

在Sintel、KITTI、Bonn（深度）、TUM RGB‑D（相机位姿）、NeuralRGBD（3D重建）等五个基准上进行评测；

**📈 对比分析**

相较于KIVI、QuaRot等LLM量化方法，QuantStream3D在大多数指标上保持近乎无损的几何精度，并在TUM ATE下降57.4%、NeuralRGBD完整度提升69.7%；在内存预算相同的情况下，batch‑size‑1时延下降23.1%，物理KV存储压缩5.05×，吞吐量提升1.67×；

**⚠️ 局限性**

局限性包括量化策略主要针对全局注意层14和首帧保护，对其他层或多头的精细化适配尚未展开，且在极低位量化（1‑bit）下的性能与稳定性尚未充分验证。

---

## 593. StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning

**arXiv ID:** 2611.27915 | [PDF](https://arxiv.org/pdf/2611.27915v1)

**作者:** Chi Yan `[一作]` (Hong Kong University of Science and Technology), Dan Xu `[通讯]` (Hong Kong University of Science and Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 StereoVLM，一种能够利用校准立体视差实现度量 3D 空间推理的视觉‑语言模型。

**💡 创新点**

创新点在于：①采用层级几何监督（视差、深度、对象空间定位）使模型从视差到度量深度形成可迁移表示；②证明视差监督在未见环境中比直接深度监督更具泛化性；③设计了可验证的基准 Stereo3D‑Bench 并通过干预实验证明模型确实利用双目几何。

**🔧 技术方法**

技术主要包括：基于 Qwen3‑VL‑Instruct 的全参数指令微调；多任务统一答案生成框架；层级几何监督策略（对应、视差、校准深度、语义空间推理）；以及对立体图像和文字查询的联合编码。

**📊 数据集**

使用的数据集有：Stereo3D‑Bench（由 DROID、Xperience‑10M、FSD、RoboCasa 合成的 8,017 对立体图像及 94,546 个 QA 对；并包含独立 OOD 测试集）。

**📈 对比分析**

在 Stereo3D‑Bench 上，StereoVLM‑8B 在点级几何任务上达 95.17% 的 depth δ₁，点误差 3.80 px；在对象级空间推理上平均 66.98%，比 GPT‑6 Astra 提升 19.37%；与单目深度 VLM 及多图像 VLM 对比，表现均显著更优；干预实验显示正确的右视图输入对性能提升至关重要。

**⚠️ 局限性**

限制在于：实验主要在受控基准和仿真厨房任务上，且仅适用于已校准、同步的双目相机；缺乏对真实世界噪声、不同机器人平台以及跨域（simulation‑to‑real）鲁棒性的评估。

---

## 594. Integrating Generative Artificial Intelligence with Learning Style Models: Opportunities, Challenges, and Future Directions

**arXiv ID:** 2611.27868 | [PDF](https://arxiv.org/pdf/2611.27868v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 595. The Benefit of Migration in Throughput Maximization on Unrelated Machines and under Restricted Assignment

**arXiv ID:** 2611.27918 | [PDF](https://arxiv.org/pdf/2611.27918v1)

**作者:** George Karakostas `[一作]` (McMaster University), Stavros G. Kolliopoulos `[通讯]` (National and Kapodistrian University of Athens)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究预抢占式吞吐量最大化（weighted throughput maximization）问题，重点考察迁移（migratory）与非迁移（non‑migratory）调度之间的收益差距。先给出一个强大的线性规划（LP）松弛，再分别对不相等机器（unrelated machines）和受限分配（restricted assignment）两种模型进行分析。通过将作业分成“大覆盖”与“小覆盖”两类，并利用多层分隔树（midpoint tree）和“可用距离”区间，将小覆盖作业进一步拆成“厚作业”和“薄作业”，并构造常见见证（common witness）与梯子（ladder）结构；对每一层再构造多组候选调度，最终选取重量最大的候选调度，从而得到一个近似解。对受限分配模型，则将作业按相对松弛度分为“大松弛”“中等松弛”“近乎刚性”三类，分别使用见证系统、网格化分组与递归动态规划来实现约束匹配与调度。论文给出了对应的近似因子，并证明在作业数或松弛度的意义下这些因子是最优的。

**💡 创新点**

① 对于不相等机器模型，首次给出了迁移收益的上界为 O(√n)（n 为作业数）并给出匹配下界；② 对受限分配模型，首次给出了 O(log n) 的上界和下界；③ 所有结果均是通过构造性证明得到的第一套非平凡近似算法；④ 通过精细的 LP 约束（包括非平行性约束）显著缩小了整型规划的割裂效应；⑤ 对于“薄作业”引入了梯子分解和四色调度，显著降低了候选数。

**🔧 技术方法**

核心技术包括：
- 强化的线性规划松弛（包含覆盖、机容量和非并行约束）。
- 常见见证（common witness）与匹配法则，将 LP 流量转化为整数匹配。
- 中点分隔树（midpoint tree）以及可用距离区间的分层处理。
- 梯子（ladder）分解与四色匹配，用于薄作业的精细调度。
- 见证系统与族划分，支持中等松弛作业的分组。
- 递归动态规划与网格化距离，处理近乎刚性作业；
- 透射子图（transversal matroid）与贪心匹配，保证独立分配。

**📊 数据集**

无实验数据集；研究完全是理论分析，结果通过构造实例给出下界。

**📈 对比分析**

与传统的非迁移最优解比较，本文证明：
- 在不相等机器上迁移最优解最多是非迁移最优解的 O(√n) 倍；
- 在受限分配上迁移最优解最多是非迁移最优解的 O(log n) 倍。 
这些上界与对应下界（分别是 O(√n) 与 O(log n)）一致，表明近似因子已是最优的。  
在算法层面，作者给出了多阶段的构造性近似算法，时间复杂度为多项式，且可在确定性多项式时间内实现。

**⚠️ 局限性**

限制与不足：
- 近似因子仍随作业数或松弛度呈对数/平方根增长，实际应用中对大规模实例可能仍有挑战。 
- 结果基于假设所有可行机对都能在窗口内完成作业（p_ij ≤ L_j），若不满足则迁移收益可能高达 n。 
- 线性规划松弛规模为 O(n^2 m)，在机器数和作业数很大时求解成本高。 
- 对于受限分配模型，只考虑相对松弛度低于 1/8 的作业，若松弛度分布更为离散，算法的细节可能需调整。 
- 论文未给出实验验证，所有证明均为理论上限，实际调度效果未知。

---

## 596. LesionAgent: Training-Free Agentic Tumor Tracking and Response Assessment in Longitudinal CT

**arXiv ID:** 2611.27876 | [PDF](https://arxiv.org/pdf/2611.27876v1)

**作者:** Maximilian Rokuss `[一作]` (German Cancer Research Center), Klaus Maier-Hein `[通讯]` (German Cancer Research Center)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种无训练的LLM代理，利用冻结的3D基础模型完成从CT扫描到RECIST反应评估的全流程；

**💡 创新点**

核心创新在于构建“3D+时间”把手：通过坐标标记的2D视图让LLM在3D空间操作、维护跨访视的病灶注册表、对模型输出进行自检，并将最终测量和分类交给确定性指南引擎；

**🔧 技术方法**

技术组合包括文本可提示的3D检测模型（如VoxTell）、交互式分割模型（nnInteractive）、注册与跟踪模型（uniGradICON、LongiTrack）、LLM工具调用框架以及确定性RECIST 1.1引擎；

**📊 数据集**

使用四个公开纵向CT数据集（HCUCH、TumorMap‑Lung、PanTrack、NCI‑MATCH），并在DeepLesion上评估跟踪性能，LIDC用于与放射员对比；

**📈 对比分析**

与自动链式工具、专用跟踪器、Claude Code以及一次性LLM基线比较，代理在检测、跟踪、测量和整体反应准确率上均优于所有基线，在临床CT上达95%平衡准确率；

**⚠️ 局限性**

局限性主要是依赖所用模型的性能，仍难以完全捕捉极小病灶；开放权重LLM的反应仍落后于最佳闭源模型；纵向数据集规模有限，需进一步验证。

---

## 597. From Gist to Verbatim: The Controllable Path of One Image Through a Diffusion Prior

**arXiv ID:** 2611.27875 | [PDF](https://arxiv.org/pdf/2611.27875v1)

**作者:** Luca Saglietti `[一作]` (Università Bocconi), Luca Biggio `[通讯]` (Università Bocconi)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过在冻结的扩散模型上学习可变形记忆向量，研究单张图片的记忆形成过程，并观察从语义抽象到精确复制的过渡。

**💡 创新点**

创新点在于：①发现记忆先通过抽象阶段获得语义信息，再进入锁定阶段实现逐字复制；②提出并验证“语义优先”与“抽象→精确”顺序；③揭示精度提升会削弱可编辑性、识别能力和文本检索的功能，提出记忆的精度-功能权衡。

**🔧 技术方法**

使用的技术包括：预训练的 Stable Diffusion 2.1（以及 CogVideoX‑2b、FLUX.2）扩散模型；冻结模型权重，只训练跨注意力的记忆向量；采用 DINOv2、VAE 以及文本编码器等度量；以及自定义的 paired advantage、位置方差等分析指标。

**📊 数据集**

数据集主要是模型生成的图像（基于已知 caption），以及部分真实照片作为外部测试；实验中对 30 种不同身份（人脸、动物、车辆）分别训练单张记忆。

**📈 对比分析**

方法对比通过训练进度（步数）监测语义和表面误差的衰减，展示了语义误差在 100 步左右完成一半，表面误差则需 2000 步以上；在功能评测中，编辑、单图识别、文本检索的性能在抽象阶段达到峰值，随后随锁定阶段而下降。

**⚠️ 局限性**

局限性包括：①语义先行可能受预训练文本监督影响；②对强烈离散分布目标的适应性不足；③理论框架仅为解释性模型，缺乏完整机制；④功能评测仅覆盖有限的三个维度，其他目标或记忆形式可能不表现相同的权衡。

---

## 598. Recognizing $k$-inducible Digraphs is NP-complete

**arXiv ID:** 2611.27965 | [PDF](https://arxiv.org/pdf/2611.27965v1)

**作者:** Felix Brandt `[一作]` (Technical University of Munich), Fabian Frank `[通讯]` (Technical University of Munich)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

证明了对任意固定的 k ≥ 3，判断一个有向图是否能由 k 位选民的多数投票产生的关系（即 k‑可诱导性）是 NP‑完全的；尤其先给出了 3‑可诱导锦标赛的 NP‑完备性，并通过扩展证明了更高 k 的结果。

**💡 创新点**

创新点在于：1) 首次对 3‑可诱导锦标赛给出完整的多项式时间归约（从单调 NAE‑3SAT 或 3‑均匀超图 2‑着色）实现 NP‑完备性；2) 设计了“锚点-分隔符”六元子图 H₊/H₋ 以及“有序分块”技术，能够强制某位选民遵循预先指定的顺序，从而在归约中编码变量真值；3) 将 3‑可诱导性结果推广到任意固定 k ≥ 3，展示了利用 Ramsey 定理与投票顺序固定的技巧。

**🔧 技术方法**

采用了组合数学与算法复杂度分析技术：McGarvey 定理、D3、D2 等三投票属性、图形构造、归约与证明、以及 Ramsey 定理来强制投票顺序；此外，使用了图论中的子图（H₊、H₋）和分块排序（ordered partition）来编码逻辑约束。

**📊 数据集**

无实测数据集，全部为理论构造和严格证明。研究完全基于抽象的图和投票模型，不涉及实验数据。

**📈 对比分析**

该工作不涉及实验对比或性能评估；所给结论为理论复杂度结果，即 NP‑完备性证明，证明方法为多项式时间归约，未给出任何可实现的算法或实验性能指标。

**⚠️ 局限性**

局限性：1) 结果仅适用于固定 k ≥ 3；对可变 k（k 随输入规模增长）的复杂度仍未讨论；2) 归约构造虽多项式但规模庞大，实际实现难度高；3) 研究未提供可行的多投票可诱导检测算法，只给出了困难性证明；4) 证明主要针对锦标赛（完全有向图），对一般有向图的更细粒度分类仍未完成。

---

## 599. Task2Hand: A Contact-Grounded Skill for Parametric Dexterous-Hand Morphology Design

**arXiv ID:** 2611.27967 | [PDF](https://arxiv.org/pdf/2611.27967v1)

**作者:** Haoyang Wang `[一作]` (Beijing University of Chemical Technology), Lingfeng Tao `[通讯]` (Kennesaw State University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

设计了一个名为Skill的AI辅助工作流，能够根据功能性手势简报自动生成、验证、迭代机械手模型，并在每一步通过成本排序的物理与学习门（gate）进行筛选和改进，最终得到可训练的强化学习策略。

**💡 创新点**

创新点包括：① 用可复用的“合同”把自然语言需求映射到参数化URDF/MJCF模型；② 构建一套成本排序、依赖约束的证据门（gate）链，自动识别物理失效点并记录证据；③ 明确定义人机协作边界，AI在可接受范围内自主管理搜索与训练，必要时以最少选项升级给人类；④ 通过实验验证门链在迭代中的回归保护作用。

**🔧 技术方法**

技术手段包括：基于Codex/GPT‑5的AI编码代理、MuJoCo物理仿真、TD3+HER强化学习、URDF/MJCF生成器、自动化证据记录与审计日志、成本排序与门序列优化算法。

**📊 数据集**

数据集：采用三种合成物体（不对称六边柱、双色圆柱、彩色立方体）进行训练与评估；实验使用三组随机种子（772, 882, 992），无公开数据集。

**📈 对比分析**

比较方法：在同一物理环境、相同强化学习协议下，将SAGE手与公开的LEAP手和Shadow手进行对比；评价指标为任务成功率、掉落率。结果显示：SAGE手在六边柱上成功率45.6%（LEAP 37.3%），在双色圆柱上70.8%（LEAP 51.6%）；加入稠密触觉信息后圆柱掉落率显著下降，说明触觉对保持效果有益。

**⚠️ 局限性**

局限性：仅在仿真环境中验证，未考虑制造误差、关节背压、线缆柔顺、执行器热限、噪声等真实硬件因素；门链设计针对特定任务与物体，可能不易迁移到其他操作；人机交互需要人工确认，完全自动化尚未实现。

---

## 600. Non-penetrating World Models

**arXiv ID:** 2611.27963 | [PDF](https://arxiv.org/pdf/2611.27963v1)

**作者:** Yuhang Dai `[一作]` (Hong Kong Polytechnic University), Xingyi Yang `[通讯]` (Hong Kong Polytechnic University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种新的视频世界模型防止物体穿透的方法NoPeWorld，能够在不改变生成器的情况下检测并修正潜在的穿墙行为。

**💡 创新点**

创新点在于发现冻结的Diffusion Transformer（DiT）隐藏状态已蕴含碰撞信息，并通过轻量级头部实现无训练的碰撞检测与响应，具有零样本跨域、视角和动态障碍的泛化能力。

**🔧 技术方法**

主要技术包括在冻结的DiT隐藏层上训练一个小型跨注意力预测头，用来输出安全性logit、清晰度和意图分数，并根据这些预测在生成前调整动作。

**📊 数据集**

使用UnrealZoo虚拟环境生成的真实几何数据作为训练标签，并在NoPeBench基准中测试包含32个独立场景、100条碰撞路径、开放域图像等多种任务。

**📈 对比分析**

与原始模型、文本提示、外部深度守卫等方法比较，NoPeWorld在六个公开模型上将物理违规率从平均74%降至16%，且在封闭回放和开放域任务中保持更高的安全-意图得分，显示出显著性能提升。

**⚠️ 局限性**

局限性包括对极端动态障碍和极其复杂几何的处理仍不完善，且方法依赖于DiT的特定隐藏层结构，可能对不同架构的适配需要进一步研究。

---

## 601. One partition function call to rule them all: Efficiently computing the equilibrium landscape of multiple nucleic acid strands

**arXiv ID:** 2611.27945 | [PDF](https://arxiv.org/pdf/2611.27945v1)

**作者:** Ahmed Shalaby `[一作]` (Maynooth University), Damien Woods `[通讯]` (Maynooth University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `09944146-298c-433e-89df-37255de463d7` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种可放大（magnifiable）的分区函数（partition function）算法，能够在不考虑伪结（pseudoknotted）且链数为常数 c 的多链 DNA/RNA 系统中，以 O(N³(c‑1)!) 的时间、O(N²) 的空间计算分区函数、最低自由能（MFE）和能量景观（density of states, DOS）等多种问题；该算法通过“Master Chef”和“Assistant Chef”两层动态规划实现对旋转对称性和能量放大（α‑magnification）的精确处理。

**💡 创新点**

创新点：
1. 引入可放大算法概念，统一处理分区函数与 MFE、DOS 等问题；
2. 通过切分“pizza slice”并使用旋转对称性分解，首次在多链 NN 模型下实现 O(N³(c‑1)!) 的时间复杂度；
3. 通过 α‑magnification 直接从一个 Qα 计算得到 MFE 值、MFE 结构数量以及完整能量级别；
4. 首次给出多链 DOS 的多项式时间算法；
5. 在单次调用中完成多种预测任务，显著减少传统方法的时间与空间消耗。

**🔧 技术方法**

技术手段：
- 动态规划（DP）与 McCaskill 样式递推；
- 旋转对称性分组与组合计数（orbit‑stabilizer theorem）;
- 切分“pizza”与“Master Chef / Assistant Chef”递归框架；
- 能量放大（α‑magnification）与极限分析（α→∞）以提取 MFE 与 DOS；
- 组合学与归纳证明以避免结构重计；
- 线性能量级别假设与数值稳定性分析。

**📊 数据集**

数据集：本研究为理论算法论文，未使用实际实验数据；通过符号计算与假设（能量级间距 ε、链数 c=O(1)）进行分析和证明。

**📈 对比分析**

比较与性能：
- 与 Dirks‑et‑al. (2007) 的分区函数算法相比，时间保持 O(N³(c‑1)!)，空间不变；
- 与 2025 年的 MFE 算法（O(N⁴(c‑1)!)) 相比，时间降低到 O(N³(c‑1)!)；
- DOS 与能量级别列表的多项式时间实现（之前无此算法）；
- 通过 α‑magnification 可在单次调用内完成上述所有任务，避免多次不同算法调用；
- 复杂度仍受 (c‑1)! 因子控制，适用于常数链数，但对大链数仍高昂。

**⚠️ 局限性**

限制与挑战：
- 仅适用于无伪结（unpseudoknotted）结构；
- 链数 c 必须是常数，无法直接扩展到 c=Θ(N)；
- MFE 仅返回能量值，未给出对应结构；
- 对能量级间距 ε 的假设（离散化）是理论依据；
- 实际实现仍需高精度浮点运算与大数处理；
- 仍未给出能量景观路径（动力学）描述；
- 对更复杂的能量模型（如三维折叠、pseudoknots）需要进一步研究。

---

## 602. What Must a Model Tell a Planner? Outcome Interfaces for Risk-Aware Planning After Robot Damage

**arXiv ID:** 2611.27939 | [PDF](https://arxiv.org/pdf/2611.27939v1)

**作者:** Hassan Iqbal `[一作]` (University of Texas at Austin), Krishna Kumar `[通讯]` (University of Texas at Austin)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在受损的四足Go2和人形H1机器人中，比较了四种决策接口（普通决策策略、带固定惩罚的策略、路线结果模型以及能预测每条路径成功概率和时间的能力合同），并证明在不同风险目标下，结果模型能够更好地保留任务并降低不安全路线的选取率。

**💡 创新点**

提出了“能力合同”这一新接口，它通过预测每种地形的成功概率和行进时间并沿路径组合预测，能在不需对每个危险地形多次试验的情况下提高危险性评估，并利用低风险探测快速适配受损机器人。

**🔧 技术方法**

采用高斯过程回归构建结果预测模型，使用Conformal风险校准来设定决策阈值，并在Isaac Lab仿真环境下训练与评估四种接口。

**📊 数据集**

利用300台受损Go2与H1机器人在仿真中生成的遍历数据集，数据包含多种关节削弱、额外负载、摩擦降低场景，以及每个上下文的多次试验和两次低风险探测。

**📈 对比分析**

在同一经验集、调参和风险目标（如5%失误率）下进行对比，结果显示结果模型尤其是能力合同在Go2上将失误率从约32%降至7–8%，在H1上从57%降至35–36%，并且用两次低风险探测即可将AUC提升至0.87/0.86，显著优于四次全量测试。

**⚠️ 局限性**

实验仅在仿真环境中完成，未验证完整路线执行；缺乏对不同损伤分布和更广泛任务分布的实物验证；并且未实现自适应风险目标的动态策略，需进一步完善。

---

## 603. AnimalFarm: Auditing the Generation and Detection of Hateful Images Featuring Anthropomorphic Characters

**arXiv ID:** 2611.27905 | [PDF](https://arxiv.org/pdf/2611.27905v1)

**作者:** Junjie Chu `[一作]` (CISPA Helmholtz Center for Information Security), Yang Zhang `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e0540dec-d77f-42db-94ae-d039248f6393` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统评估了拟人化非人类角色在仇恨图像生成和检测中的表现，并构建了对应的基准与数据集。

**💡 创新点**

首次将仇恨图像研究扩展到非人类角色，提出了“描绘逆转”与微调检测器两种有效提升检测性能的方法。

**🔧 技术方法**

使用多种文本到图像模型（如 Gemini、GPT Image、FLUX 等）进行生成；利用 CLIP、LLM 进行图像描述、逆转和推理；通过梯度注意力、遮挡敏感分析评估检测失误；对 Qwen3.5-4B 进行 SFT+GRPO 微调。

**📊 数据集**

构建了 650 项的 HatefulImageBenchmark（涵盖 5 种角色、3 种生成任务）和 4,370 张仇恨图的 HatefulImageDataset，以及 3,650 张中性图像用于评测。

**📈 对比分析**

与 GPT‑5.5 等现有检测器对比，描绘逆转方案 F1 0.962、微调检测器 F1 0.979，显著提升 recall（73.9%→97.2%）和整体 F1 分数。

**⚠️ 局限性**

局限性包括：仅涵盖 5 种动物角色与英中文本，样本来源有限；仅评估静态图像，未覆盖视频、音频等多模态；实验成本高，难以扩展至更大语境与模型。

---

## 604. Star Panoptic Quality: Selectively Evaluating Split and Merged Instances

**arXiv ID:** 2611.27913 | [PDF](https://arxiv.org/pdf/2611.27913v1)

**作者:** Erik Großkopf `[一作]` (University of Tübingen), Florian Kofler `[通讯]` (University of Tübingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了StarPQ评估框架，利用星森林（star forest）构造来识别实例分割中的合并（merge）与分裂（split）错误，并允许通过可配置的惩罚因子分别针对不同错误模式加权；此外，还给出了处理专家标注不一致的评价协议。

**💡 创新点**

核心创新在于：①通过叶子准则（IoL）构造星森林，精确分离干净实例、合并实例与分裂实例；②StarPQ定义的 λ_split、λ_merge 开关可单独惩罚或宽恕合并/分裂错误；③将专家不一致映射为实例级 λ 值，实现不确定性建模；④实验表明StarPQ能检测到传统PQ无法捕捉的错误，并能作为诊断工具揭示错误分布。

**🔧 技术方法**

技术手段包括：基于IoL的指针图构造星森林；星阈值 τ 用于筛选无效星；星森林中星的 IoU 计算；可配置的 λ_split、λ_merge 权重；与传统 PQ、SoftPQ、sortedAP 等指标对比；实例恢复方法（连通组件、距离变换分水岭）与多专家标注的处理。

**📊 数据集**

主要实验数据集为 COCO 与 Cityscapes，用于评估不同 λ 配置下的基线模型表现；此外在医学图像（肿瘤、细胞、3D CT/MRI）中演示实例恢复与专家标注的不一致处理；也提及常见实例分割场景如自行车、车辆、细胞等。

**📈 对比分析**

与 PQ 等现有指标比较时，StarPQ 在 λ_split=λ_merge=1 时与 PQ 一致；通过设置 λ_split=0 或 λ_merge=0 可单独惩罚合并/分裂错误，实验显示 StarPQ 能更早检测到合并/分裂情况；在 COCO 与 Cityscapes 上的基线实验表明合并错误更常见；StarPQ 通过拆分 TP 为 clean、split、merge 三类，提供了比单一 PQ 更细粒度的诊断与性能评估。

**⚠️ 局限性**

主要局限在于星森林的构造采用启发式指针链处理，未保证最大化星 IoU 的全局最优匹配，链式指针虽然稀少但仍可能导致误差；λ 采用 0/1 以保持可解释性，若使用实数值会降低可解释性；此外，星阈值 τ 的选择仍需经验决定，且在极端数据中可能需要进一步调整。

---

## 605. Renormalized Entropy from Symmetries: Simplicity Bias and Neural Heat Death

**arXiv ID:** 2611.27889 | [PDF](https://arxiv.org/pdf/2611.27889v1)

**作者:** Vasco Portilheiro `[一作]` (University College London), Peter Orbanz `[通讯]` (University College London)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过对网络参数空间中的非紧对称变换进行拉格朗日动力学分析，推导出对网络函数的隐式偏置，即所谓的重整化熵；

**💡 创新点**

创新点在于首次将非紧对称纤维的体积发散率映射为熵项，揭示了零正则化极限下的“热死亡”现象，并与已知的表示成本理论相联系；

**🔧 技术方法**

主要技术包括拉格朗日动力学、Gibbs平衡、纤维分区函数、Bessel/矩阵Bessel函数、Grassmann子空间积分以及神经切线核的体积公式；

**📊 数据集**

实验使用了极简的合成数据集（如x=±1的单维输入，目标为ReLU+1/2ReLU的线性组合）来验证理论预测；

**📈 对比分析**

通过与无正则化拉格朗日动力学的经验分布（利用Kolmogorov–Smirnov距离衡量）进行比较，发现含熵项的理论与实验分布高度吻合，而忽略熵项的预测则显著失效；

**⚠️ 局限性**

局限性包括：仅在拉格朗日噪声下可行；对非紧对称极限的交换与收敛性证明仅在对角/线性网络上得到；对更复杂网络隐藏对称性分析不足；以及对SGD等非高斯噪声情形的推广尚未给出。

---

## 606. Linear-Size Spectral Sparsification by Sign Descent

**arXiv ID:** 2611.27907 | [PDF](https://arxiv.org/pdf/2611.27907v1)

**作者:** Trevor Vaughn `[一作]` `[通讯]` (Carnegie Mellon University), Trevor Vaughn (Carnegie Mellon University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种近线性时间的算法，在保留原图能量的前提下生成O(n/ε²)条边的(1±ε)谱稀疏化子图。

**💡 创新点**

核心创新在于将部分着色与随机符号奖励相结合，构造梯度方向并在每个阶段以常数比例消除边权，避免了传统采样的额外对数因子。

**🔧 技术方法**

主要技术包括随机分配符号、使用光滑ℓ-范数惩罚、稀疏算子乘法、拉普拉斯求解器和矩阵部分着色。

**📊 数据集**

论文未在具体数据集上实验，全部为理论分析与证明。

**📈 对比分析**

与以往的 Spielman–Srivastava、Batson–Spielman–Srivastava 等方法相比，算法在误差 ε、日志因子和依赖度上均取得了改进，时间复杂度为 O(n + ε^{-3}[m log² n + n log³ n (loglog n)⁴])。

**⚠️ 局限性**

主要局限在于随机性、较大的隐含常数与对数因子，以及对正权重无向图的适用性；对极端稠密图仍有潜在效率瓶颈。

---

## 607. The Dual-Tier Sentiment Framework: Scalable Multimodal UX Attribution in Web Search

**arXiv ID:** 2611.27899 | [PDF](https://arxiv.org/pdf/2611.27899v1)

**作者:** Yi Ding `[一作]` (Google), Golnaz Farhadi `[通讯]` (Google)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套双层情绪框架，实现对大规模 Web 搜索实验的可扩展 UX 归因。

**💡 创新点**

创新点在于同时结合宏观基于遥测的 LLM 检测和微观多模态 MLLM 诊断，并通过 RL 优化的 Mentor‑Student 提示生成实现自动化对齐。

**🔧 技术方法**

采用 Gemini XS 的双头编码解码 LLM、Gemini 3.0 Pro 的多模态 LLM、RL 与 AI‑judge 对齐、链式思维蒸馏、自动提示优化等技术。

**📊 数据集**

使用 500k+ 真实实验记录的遥测与用户调查标签、1000+ 人工标注的结果块数据以及首次结果用户调查样本。

**📈 对比分析**

在 12 个 hold‑out A/B 实验上，宏观模型 PR‑AUC 提升 9.8%，微观模型 F1 提升 6.5%，显著提高实验准入率与诊断精度。

**⚠️ 局限性**

局限在于 Tier‑2 的计算成本导致只能异步批量评估，缺少跨平台、交互式 UI 状态支持，且对不同地区布局需进一步校准。

---

## 608. ESSI: Computationally Efficient Semantic Search for Sparse Autoencoder Concepts

**arXiv ID:** 2611.27921 | [PDF](https://arxiv.org/pdf/2611.27921v1)

**作者:** James Baker `[一作]` `[通讯]` (University of Maryland Baltimore County), James Baker (University of Maryland Baltimore County)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出ESSI方法，使用分割掩码标记概念区域，利用稀疏自编码器选择与概念对应的特征，并通过线性探针自动确定插入/删除强度，完成概念的插入与消除。

**💡 创新点**

创新点在于：①用掩码仅关注概念区域，提升特征选择的语义性；②通过线性探针和二分搜索自动挑选特征数量与强度，完全不需要生成图像；③显著降低计算量（比SAeUron高达9000×、比CASL 2000×），保持与传统方法相近的性能。

**🔧 技术方法**

核心技术包括：稀疏自编码器（TopK SAE）、SAM3分割、线性探针（L2正则化逻辑回归）、二分搜索、残差编码/解码编辑、对比实验中的SAeUron、Adaptive、OSP、CASL等。

**📊 数据集**

实验数据集：SDXL‑Turbo 生成的图像，UnlearnCanvas 20个对象 × 50风格（共1000张），SAM3 用于生成掩码；使用 ViT‑L/16 重新训练的对象与风格分类器进行评估。

**📈 对比分析**

与五种现有方法对比：ESSI 在插入任务中取得最高 EPR，删除任务中仅次于 SAeUron；但其 FLOPs 需求远低于传统方法；总体性能与最优方法相近，但显著更高效。

**⚠️ 局限性**

局限性：只在 SDXL‑Turbo 单模型、单层实验；未自动选择最佳层；对不同模型/概念的鲁棒性未知；依赖 SAE 的稀疏假设，若特征并非多语义可能效果受限。

---

## 609. TEA-Bench-BR: Can Coding Agents Turn a Branching Clinical Specification into Correct Code?

**arXiv ID:** 2611.27895 | [PDF](https://arxiv.org/pdf/2611.27895v1)

**作者:** Marcio Andre de Souza Estevam `[一作]` `[通讯]`, Marcio Andre de Souza Estevam

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `9cc9baba-5356-466d-81ff-d80028d90279` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

创建了 TEA‑Bench‑BR 基准，提供三种巴西自闭症筛查工具（RICS‑20、IPSQ、EML）的参数化规范、参考实现、隐藏测试和 FHIR R4 导出，并评估 Gemma 4 12B 与 31B 的代码生成表现。

**💡 创新点**

创新点在于：1）使用 Jinja2 模板实现可重现的参数化任务生成；2）通过变异测试确保隐藏案例覆盖典型实现错误；3）加入 PHQ‑9 污染探测器验证记忆性错误；4）对语言、临床安全性和隐私进行系统分析。

**🔧 技术方法**

主要技术包括 Jinja2 模板、变异测试、vLLM 运行 Gemma 4、HL7 FHIR R4 Validator 与 RNDS Profile、Bootstrap 置信区间统计等。

**📊 数据集**

使用合成数据集：三种自闭症筛查问卷的规范（RICS‑20、IPSQ、EML）生成 144 个任务，包含 6,798 条隐藏案例；以及 PHQ‑9 的真实/合成/陷阱版本。

**📈 对比分析**

采用隐藏案例通过率和任务解决率进行比较；31B 在级别 1‑3 的完成率分别为 77.8%、85.2% 和 78.7%，隐藏案例通过率超过 99%；12B 仅在级别 1 完成率 38%，三级完全失败。

**⚠️ 局限性**

局限性包括：工具为合成且未经过临床验证，仅评估规范符合度；模型仅为 Gemma 4，缺乏多模型对比；基准缺乏可复现性和对真实患者数据的评估。

---

## 610. FairITR: Fairness-Aware Learning of Multicategory Individualized Treatment Rules

**arXiv ID:** 2611.27894 | [PDF](https://arxiv.org/pdf/2611.27894v1)

**作者:** Zeyu Chen `[一作]` (University of Delaware), Bintong Chen `[通讯]` (University of Delaware)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出多类别个体化治疗规则的公平性定义，并开发了FairITR框架来同时优化治疗效果与公平性。

**💡 创新点**

首次将角度分类、对抗去偏与协方差正则化结合，用统一的端到端学习流程实现多类别公平治疗规则，并提供理论证明。

**🔧 技术方法**

使用角度分类（angle‑based）构建治疗决策，LUM 损失做为分类近似；对抗模块消除隐藏表示中的敏感信息；协方差正则化进一步抑制公平偏差。

**📊 数据集**

在三组真实数据上评估：RHC（右心导管测序）、Email（电子邮件营销）和 IST（国际卒中试验）。

**📈 对比分析**

与传统 ITR 方法、对抗公平两阶段方法及公平无知策略对比，FairITR 在所有数据集上均取得最小的 Δ_fair，同时保持或超过其他方法的治疗价值。

**⚠️ 局限性**

仅考虑二元敏感属性；对抗训练易产生不稳定，需调节超参数；公平性仅关注决策分布，未涉及决策后续社会福利的长远影响。

---

## 611. On the Complete Feature Set for ADHD Classification in Children and Adolescents Based on EEG Signals

**arXiv ID:** 2611.27970 | [PDF](https://arxiv.org/pdf/2611.27970v1)

**作者:** Javier Sanchis `[一作]` (University of Alicante), Juan Trujillo `[通讯]` (University College Dublin)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

对 156 名儿童和青少年在 5 种认知任务中收集的 669 条 EEG 记录进行系统筛选、质量控制、预处理并计算 7 类共 115 条特征定义，生成可追溯且可共享的特征集合；同时发布了完整的数据、软件与可验证的示例代码。

**💡 创新点**

首次提供同一时段、同一预处理下多种 EEG 特征的对齐结果，并公开全部质量评估与计算状态，使得跨特征、跨任务的比较成为可能；同时给出基于 XGBoost 的招募组分类基线，验证数据可用性。

**🔧 技术方法**

采用信号质量评估（多尺度时间/频谱/记录级）、保守滤波（0.5‑40 Hz Butterworth）、重采样（128 Hz）、平均参考、逆距离插值、MNE‑FIF 存储；特征提取涵盖时间统计、传统频谱、aperiodic、复杂性、连接、图网络、时频小波等；使用 8 s、4 s、2 s、16 s 窗口进行特征计算；XGBoost 用于招募组分类。

**📊 数据集**

使用公开的 BALLADEER 数据集（原始 EEG 记录 156 名儿童/青少年，5 个任务：CogniFit、Robots、Slackline L1/L6/L11），在 669 条符合质量阈值的记录上进行特征计算。

**📈 对比分析**

在 4 个重叠任务组（Robots、Slackline L1/L6/L11）上，利用 XGBoost 对 8 s 窗口特征进行参与者留出验证；Robots 组平均 AUROC 为 0.728，Slackline 组在 0.483–0.576 之间；与仅基于招募比例的基线相比，Robots 组 Brier 分数更低，Slackline 组则相反。

**⚠️ 局限性**

仍存在残留运动、眼电、肌电伪迹；不同设备、通道与采样率导致的测量偏差无法完全消除；招募组与正式诊断标签不等价；特征的计算状态为未定义时需人工判断；由于任务和设备混合，无法单独评估任务效应。

---

## 612. Speculative Decoding Is Less Speculative Than You Think

**arXiv ID:** 2611.27891 | [PDF](https://arxiv.org/pdf/2611.27891v1)

**作者:** Guanning Zeng `[一作]`, Andrea Zanette `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种名为 DICE 的训练无关方法，利用在推理前就已知的采样随机数（uniform 和 Gumbel 噪声）来引导草稿树构建和异步预取，显著提升语言模型推理的吞吐量。

**💡 创新点**

创新点在于：① 发现并利用验证阶段的随机数（通常视为未知）作为信息源；② 将这些已知随机数与现有草稿模型的预测结合，制定节点选择与预取优先级的分数；③ 通过对采样分布的校准和路径权重估计，使计算资源更加精准地投放到目标模型实际会采样的路径上。

**🔧 技术方法**

技术细节包括：使用 Gumbel‑max 样本和 Uniform 验证抽样；设计分数公式 r_x(v) = softmax((α d_x + T g_{|x|+1})/σ)；路径累计权重 w(xv)=w(x)r_x(v)；对参数 α,σ 进行一次性校准；同步草稿和验证的随机数生成器；在草稿树和异步预取两种推理模式下实现该策略。

**📊 数据集**

实验使用了两类基准：Spec‑Bench 和 SPEED‑Bench（涵盖对话、翻译、摘要、数学推理、问答和检索增强生成）以及 Alpaca、GSM8K、HumanEval、UltraFeedback（用于异步预取评估）。模型涵盖 Llama‑3.1‑70B、Qwen‑3.8‑27B、Nemo‑3.5‑120B 等大规模预训练模型，以及多种草稿器（OPT‑Tree、EAGLE‑2/3/4、MTP）。

**📈 对比分析**

与原始的草稿树方法（EAGLE‑4）、异步预取（SSD）以及基准自回归（AR）进行对比。DICE 在草稿树推理中最高可提升 24.6% 吞吐量，在异步预取中最高可提升 13.7%，且在不同温度、并发级别和后端（SGLang、vLLM、SGLang）下均保持一致的性能提升。

**⚠️ 局限性**

局限性包括：① 仍需对草稿模型的预测进行一定的校准，虽然对参数的敏感性较低；② 在极高并发或非常大节点预算时，DICE 的优势相对减小；③ 依赖可同步的随机数生成机制，对现有推理框架的适配可能有一定门槛；④ 仅在推理阶段实验，尚未验证对训练或其他类型推理（如批量推理）的影响。

---

## 613. No Polynomial-Time Algorithm For Tarski Fixed Points Unless Cryptographic Assumptions Fail

**arXiv ID:** 2611.27977 | [PDF](https://arxiv.org/pdf/2611.27977v1)

**作者:** Eleni Batziou `[一作]` (Archimedes/ Athena Rc), Rahul Savani `[通讯]` (University Of Liverpool)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构造子指数时间的归约，将 UEOPL（唯一势能线终点）问题转化为寻找 Tarski 固定点的问题，从而给出 Tarski 固定点问题的时间复杂度下界；

**💡 创新点**

创新点在于提出一种可调节参数 r 的归约策略，既能保持归约后实例的规模在可控范围，又能在满足不同假设（如弱 ETH、密码学硬度假设）下得到几乎匹配的上界下界；

**🔧 技术方法**

主要技术包括：① 对 UEOPL 进行规范化和子线构造；② 通过多重检查点实现参数化的层级二分搜索；③ 将 OPDC（单方向离散收缩）实例归约为 Tarski 固定点，并在此过程中保持单调性；④ 结合密码学假设（如迭代平方、强 Fiat–Shamir、功能加密等）推导几乎最优的时间下界；

**📊 数据集**

论文未使用任何外部数据集，所有分析均基于理论构造与归约；

**📈 对比分析**

通过归约与已知的查询复杂度下界对比，证明在若干密码学假设下，Tarski 固定点问题需要至少 quasi‑polynomial 时间；该下界与最新查询复杂度下界相匹配，显示问题难度相当；

**⚠️ 局限性**

局限性在于下界依赖于尚未被验证的密码学假设或弱 ETH，且归约导致的实例维数和宽度增长显著，实际算法实现仍需进一步研究。

---

## 614. Cluster Editing in Subexponential Fixed-Parameter Time

**arXiv ID:** 2611.27956 | [PDF](https://arxiv.org/pdf/2611.27956v1)

**作者:** Alexander Firbas `[一作]` (TU Wien), Manuel Sorge `[通讯]` (TU Wien)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种新的子指数级算法，用来判定给定图在最多 k 次边增删操作下是否能转化为若干不相交团（Cluster Editing）。

**💡 创新点**

创新点在于：
• 证明了一个类似 SAT Sparsification Lemma 的稀疏化定理，使得任意 Cluster Editing 实例可以在指数级搜索树中被分解为若干规模线性、度受限的 r‑稀疏实例；
• 将 r‑稀疏实例转化为 3‑CNF 公式，并利用已知的 SAT 算法（时间 2^a·m/√(log m)）得到 Cluster Editing 的子指数级算法；
• 通过对阈值 r 的巧妙选择，使得 2^O((k·log log log k)^{3/2}/√(log log k))·n^O(1) 的运行时间得以实现，从而验证了 Cao‑Chen 的猜想。

**🔧 技术方法**

主要技术包括：
1) 加权聚类编辑（weighted Cluster Editing）的定义与操作（flip、merge、隔离）；
2) 通过递归搜索树与合并成本阈值 r，得到 r‑均匀（r‑uniform）实例；
3) 证明 r‑均匀实例的正图分解为稠密与稀疏连通分量；
4) 对稀疏分量利用候选集合与共享量（share）构造 3‑CNF 公式；
5) 使用约束与算术电路编码实现预算阈值检查；
6) 结合已知的 SAT 算法和 ETH 的反驳得到整体时间复杂度。

**📊 数据集**

本工作完全基于理论分析，并未使用任何实验数据集。

**📈 对比分析**

性能评估：
- 在理论上，Cluster Editing 的最优运行时间被证明为 2^o(k)·n^O(1)，具体为 2^O(k·(log log log k)^{3/2}/√(log log k))·n^O(1)。
- 与之前的 1.62^k 上界相比，取得了显著的指数级减小；与之前的 2^o(k) 下界（基于 ETH）相矛盾的情况已被 ETH 反驳后消除。

**⚠️ 局限性**

局限性与未来工作：
• 该算法的常数因子极大（如 c_2>2^6000），实际可行性有待评估；
• 依赖于已被反驳的 ETH，若未来发现新的 ETH 变体或更强的下界，结果可能需要重新评估；
• 目前只处理可编辑成团的情况，对仅删边或仅增边的变体（Cluster Deletion 等）尚未直接适用；
• 对于需要解决顺序或集合之外的结构问题（如反馈顶点集、反馈弧集），此方法尚不可推广；
• 该方法对实例规模的上界（如 k 可高达 n^2）仍可能导致巨大的搜索树，实用化仍需要进一步的实验与优化。

---

## 615. Real-time neuromorphic audio and vision perception for autonomous robotic applications: Integration and experimental evaluation on a rover platform

**arXiv ID:** 2611.27953 | [PDF](https://arxiv.org/pdf/2611.27953v1)

**作者:** Guillem Fustier-Castillo `[一作]` (Instituto Tecnologico de Informatica), Pedro Zuccarello `[通讯]` (Instituto Tecnologico de Informatica)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `51c0528b-f690-4182-ae60-bb5f046c276c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究将经典CNN与Neuromorphic感知技术集成到一台四轮车平台上，实时实现音频与事件摄像头的多模态感知，并将识别结果映射到机器人行为；

**💡 创新点**

创新点在于首次实现SynSense Xylo Audio 2芯片与SpikeYOLO事件摄像头的闭环集成，并通过Docker化、ROS 2与FastAPI构建模块化服务化架构；

**🔧 技术方法**

使用技术包括NVIDIA Jetson AGX Orin、SynSense Xylo Audio 2、Prophesee EVK3 HD事件摄像头、PANNs CNN14、SpikeYOLO、Rockpool、ROS 2、Docker与FastAPI；

**📊 数据集**

数据集方面使用AudioSet训练CNN14，使用自制高低音频样本训练Xylo SNN，利用自制事件摄像头录像的交通锥数据训练SpikeYOLO；

**📈 对比分析**

通过对比经典音频管线（CNN14、10 s窗口、GPU推理≈8.4 ms/预测、能耗高）与Neuromorphic管线（4个LIF神经元、100 ms窗口、Xylo芯片推理≈100 ms/预测、能耗极低），验证了Neuromorphic在实时性与能效上的优势；

**⚠️ 局限性**

局限性包括：音频分类仅限低/高音双类；Xylo输出缺乏置信度；事件摄像头检测未驱动机器人行为；缺乏系统级定量评估与能耗测量。

---

