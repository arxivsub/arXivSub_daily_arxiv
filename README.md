# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-06-03 | 今日论文总数: 728

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs

**arXiv ID:** 2606.02628 | [PDF](https://arxiv.org/pdf/2606.02628v1)

**作者:** Aizierjiang Aiersilan `[一作]` (University of Macau), Aizierjiang Aiersilan `[通讯]` (University of Macau)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5092758213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在同一实验框架下，统一评估了四种幻觉检测方法（线性/MLP探针、INSIDE EigenScore、self‑consistency、注意力熵），并比较它们在三款4‑bit量化的7B‑8B LLM上的表现；

**💡 创新点**

首次将探针与采样方法在同一模型、同一评测协议下进行层级对比，发现中间层隐藏状态中存在近乎线性的真值可分方向，并指出注意力熵可在知识驱动场景下无额外推理成本提供补充信号；

**🔧 技术方法**

采用线性/多层感知机探针、INSIDE EigenScore、self‑consistency（exact‑match与句嵌入相似度）以及Transformer最后‑token注意力熵等技术，并使用4‑bit NF4量化、SAPLMA探针协议、聚类与t‑SNE可视化；

**📊 数据集**

使用TruthfulQA、HaluEval‑QA、FEVER、以及自制的synthetic四个约400条样本的二分类对数据集；

**📈 对比分析**

探针在单层即可获得0.904–1.000的AUROC，明显优于采样方法（最高0.541）；在HaluEval‑QA中，第一层注意力熵可达0.866–0.941的AUROC；总体而言，探针最优，采样方法在此评测协议下表现接近随机；

**⚠️ 局限性**

局限性包括：仅评估7B‑8B量化chat模型；paired‑label协议导致采样方法表现低迷；未测试更大或基础模型、非英语或自由文本生成；采样参数固定，未进行预算或层级探索；注意力熵仅在三个层面评估，可能缺失更深层信号。

---

## 2. A Locally Deployed RAG-Based Academic Advising System for Course Selection

**arXiv ID:** 2606.02983 | [PDF](https://arxiv.org/pdf/2606.02983v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 3. Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers

**arXiv ID:** 2606.02776 | [PDF](https://arxiv.org/pdf/2606.02776v1)

**作者:** Vera Neplenbroek `[一作]` (University of Amsterdam), Raquel Fernández `[通讯]` (University of Amsterdam)

**通讯引用:** 5994 | [OpenAlex ID](https://openalex.org/A5028758097)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究大型语言模型在高风险场景下对不同社会人口学群体的回答差异，发现对话主题比模型对用户属性的直接推断更能解释这些差异。

**💡 创新点**

创新点在于将对话主题定位为主要驱动因素，并通过提示和线性探针评估模型对人口特征的推断能力，揭示模型对社会人口信息的把握极为有限。

**🔧 技术方法**

技术包括：对话历史与高风险问题的结合；使用多款LLM（Llama3.1 8B、Gemma 3 12B、Qwen3.6 27B、Kimi K2.6）；情感、可读性、LIWC等心理语言特征提取；线性探针和ElasticNet回归分析。

**📊 数据集**

数据集为：PRISM（含用户对话与社会人口标签）、Community Alignment（含对话与预定义主题）、Sociolinguistic Bias Benchmark（SBB，用于高风险问题）。

**📈 对比分析**

通过ANOVA检验群体差异显著性、对提示预测的F1分数与线性探针性能比较、回归模型解释方差达77%以上；但对话主题的系数最高，模型对人口特征的推断性能低于随机或多数类基线。

**⚠️ 局限性**

局限性包括仅使用英语数据、对话与高风险问题的组合人为、样本来自标注平台，可能不具备普适性，且未考虑多语种或更自然的交互场景。

---

## 4. Will Accurate Fields Mislead Photonic Design? FromGlobal Accuracy to Port Readout

**arXiv ID:** 2606.03038 | [PDF](https://arxiv.org/pdf/2606.03038v1)

**作者:** Yitian Zhang `[一作]` (Sun Yat-sen University), Zhong Guan `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3647 | [OpenAlex ID](https://openalex.org/A5101405656)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了全场神经网络替代器在光子MMI分路器设计中的局部端口读数误差，并提出了 Field/Mediator/Readout 诊断视角。

**💡 创新点**

创新点是提出了传播对齐的神经算子 PaNO 及其输出敏感变体 PaNO‑R2，并展示了全场误差与端口读数不一致的系统性问题。

**🔧 技术方法**

采用了多尺度异方差前端、学习的模态令牌、状态空间序列推进和残差交叉模态耦合等神经算子技术。

**📊 数据集**

使用了 15 波长可调 3×3 MMI 设备的 80×384 网格数据，共 4608 个测试案例。

**📈 对比分析**

通过与 FNO、FactorFNO、UNet 及 NeurOLight 等基线在相同诊断脚本下对比，PaNO‑R2 在 cMAE、SWR、输出轮廓和端口功率误差上分别优于基线，整体提升约 70%。

**⚠️ 局限性**

限制在于仅验证二维 Hz 模式的 MMI 设备，复合相位和耦合读数仍有挑战，且对更复杂的向量场或多端口拓扑的推广尚待研究。

---

## 5. Binary Road Surface Classification Using Machine Learning on Production Vehicle Signals During Cruising

**arXiv ID:** 2606.02762 | [PDF](https://arxiv.org/pdf/2606.02762v1)

**作者:** Vishal Hariharan `[一作]` (Goodyear Tire and Rubber Company), Kanwar Bharat Singh `[通讯]` (Goodyear Tire and Rubber Company)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用生产级车载传感器信号，在车辆低激励巡航状态下实现滑移/抓地路面二分类。

**💡 创新点**

首次证明仅凭车载信号即可在巡航期间通过机器学习检测路面湿滑，并提出两种互补模型（XGBoost特征法与1D CNN）。

**🔧 技术方法**

采用特征提取、XGBoost、1D卷积神经网络、滑动窗口、归一化、Grad‑CAM、t‑SNE等技术。

**📊 数据集**

利用近10小时冰雪路面与20小时干燥路面的公开道路数据，采样频率100 Hz。

**📈 对比分析**

对比两模型在测试集上的召回率和精度，XGBoost召回率为98%/92%，精度为92.5%/97.9%；1D CNN召回率为98.85%/89%，精度为90%/98.7%，两者在嵌入式上均实现低于100 ms的推理延迟。

**⚠️ 局限性**

仅进行二分类（抓地/滑移），未覆盖湿、雾、过渡等多类情况；对未知驾驶场景的误报率、检测延迟等指标缺乏量化；滑动窗口重叠可能导致训练与测试集泄漏。

---

## 6. From Local Training to Large-Scale Mapping: A Comparative Assessment of Machine Learning and Deep Learning for Transferable Satellite-Derived Bathymetry

**arXiv ID:** 2606.02764 | [PDF](https://arxiv.org/pdf/2606.02764v1)

**作者:** Hsiao-Jou Hsu `[一作]` (Ohio State University), Joachim Moortgat `[通讯]` (Ohio State University)

**通讯引用:** 2149 | [OpenAlex ID](https://openalex.org/A5075797962)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了随机森林与多种深度学习架构（ResNet‑50/101、EfficientNet‑B4、ConvNeXt‑Large）在 Sentinel‑2 多光谱影像下实现可跨区域浅水测深的可行性，并提出了深度加权损失（Smooth Weight Function）与空间连续训练策略来提升浅水精度与跨区域泛化能力。

**💡 创新点**

创新点在于（1）引入 SWF‑加权 RMSE 损失，在训练中显式强调近海浅水深度；（2）采用空间连续（而非随机）划分的训练样本，保留地貌连贯性；（3）系统进行跨区域（普拉塔岛/大堡礁→阿什莫尔/卡特里）评估；（4）利用多时相 Sentinel‑2 聚合（中位数）进一步提升稳健性。

**🔧 技术方法**

使用的技术包括：传统随机森林回归；预训练 ImageNet 的卷积主干（ResNet‑50/101、EfficientNet‑B4、ConvNeXt‑Large）嵌入 DeepLabV3+ 语义分割框架；SWF‑加权 RMSE 作为目标函数；Patch‑size 512×512 像素、50% 重叠、光谱预处理；多时相中位数聚合；并与 Swin‑BathyUNet、U‑Net 进行基准对比。

**📊 数据集**

数据集：Sentinel‑2 Level‑2A BOA 影像；Pratas 岛 5 m×5 m LiDAR 参考深度；大堡礁 30 m DEM；阿什莫尔、卡特里独立检验区域；以及 MagicBathyNet（Agia Napa + Puck Lagoon）用于与 Swin‑BathyUNet 的基准比较。

**📈 对比分析**

通过在训练/验证/测试三阶段（60/20/20%）划分并对比 SWF、RMSE、RPE 三种损失，发现 SWF 在浅水（≤3 m）RMSE 下降至 0.26 m；跨区域 RMSE 在 2.46–2.98 m 之间，ConvNeXt‑Large 在中间深度（≈5–10 m）表现最佳；随机森林在内区域表现相当但跨区域误差显著增大；在 MagicBathyNet 上同一架构的 RMSE 低于 Swin‑BathyUNet，参数量显著更少。

**⚠️ 局限性**

局限性包括：仍无法完全消除跨区域误差，受底质、光学属性、潮汐校正不确定性等因素影响；仅在相对清水、外滩珊瑚礁环境中验证，难以推广到浑浊或近岸区域；缺乏显式物理约束，模型可能在光学极端情况下失稳；训练样本覆盖范围有限，导致在新环境中性能下降。

---

## 7. COD10K-C: Benchmarking Robustness of Camouflaged Object Detection Under Natural Image Corruptions

**arXiv ID:** 2606.02603 | [PDF](https://arxiv.org/pdf/2606.02603v1)

**作者:** Arafat Hossain Sayem `[一作]` `[通讯]`, Arafat Hossain Sayem

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 COD10K-C 鲁棒性基准，涵盖 8 种失真类型与 5 个严重程度，对 COD10K 测试集进行评估，并设计了轻量级模型 RobustCODLite 来提升对失真图像的检测性能。

**💡 创新点**

创新点在于①构建了首个针对伪装目标检测的失真基准 COD10K-C；②提出了 RobustCODLite，融合失真增强、频域先验分支和不确定性一致性损失，显著提升对几何失真（模糊、噪声）的鲁棒性；③通过与主流 COD 模型对比，揭示清晰图像与失真图像性能之间的巨大差距。

**🔧 技术方法**

使用的技术包括 U‑Net 风格分割网络（EfficientNet‑B0 编码器）、频域高频残差先验分支、双头输出（分割与不确定性），以及联合训练损失（分割、边界、置信度、一致性）。评估采用 Dice、IoU、MAE、Boundary F1 与 ECE。

**📊 数据集**

使用 COD10K 测试子集（2,026 张图像）并对其施加 8 种失真与 5 级严重度，形成 81,040 个评估样本；同时对 SINet‑v2、PFNet、ZoomNet 以及 RobustCODLite 进行测试。

**📈 对比分析**

通过在 40 条失真条件下对比四种模型，RobustCODLite 在清晰图像上的 Dice 为 0.685，失真图像上为 0.632，保持了 92.3% 的清晰性能；相较于 SINet‑v2 等大型模型，RobustCODLite 在几何失真（模糊、噪声）下表现更优，且参数量仅约 7.2M，显著更轻量；其在光度失真下的性能损失最小。

**⚠️ 局限性**

局限性包括仅考虑 8 种失真类型，未覆盖更广泛的现实失真（如热噪声、滚动快门等）；一致性损失要求清晰与失真双向推理，显著增加训练显存；实验仅在 COD10K 子集与单一随机种子下进行，缺乏跨种子与跨数据集的鲁棒性验证。

---

## 8. Many a Little Makes a Mickle: A Code-Centric Empirical Study of Data Minimization Principle in Android App Development

**arXiv ID:** 2606.02960 | [PDF](https://arxiv.org/pdf/2606.02960v1)

**作者:** Dianshu Liao `[一作]` (Australian National University), Xiaoyu Sun `[通讯]` (Australian National University)

**通讯引用:** 228 | [OpenAlex ID](https://openalex.org/A5079228450)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地把数据最小化从法规层面落到Android应用的代码实现层，通过对1,114个开源项目和9,875个真实APK的静态分析，提炼出10个常见的最小化场景和31条可操作的编码准则，并验证这些准则能有效指导LLM生成符合隐私合规的代码。

**💡 创新点**

创新点在于：①首次以代码层面细化数据最小化原则并形成可执行的10×5阶段场景；②结合大规模实测提炼31条通用编码准则；③证明LLM在缺乏准则时会放大真实项目中的违规模式，而加入准则后可将违规率降至零。

**🔧 技术方法**

采用定性代码审计、静态调用图与taint分析（基于FlowDroid）、正则API匹配以及LLM生成评估（GPT‑5.2、Claude‑4.5‑Sonnet、Gemini‑2.5‑Pro、Cursor）等技术。

**📊 数据集**

数据集包括：①F‑Droid开源项目 1,114 个；②AndroZoo 9,875 个APK（从至少 1,000 万下载量的 14,237 个应用中挑选最新版本）。

**📈 对比分析**

通过人工验证计算精度（大多数指标在 70–90% 之间），LLM生成对照实验显示：基本 prompt 下违规实例数 46–59，加入准则后所有模型违规率降为 0；对比了 10 个最小化场景的覆盖情况，证明准则可显著提升合规性。

**⚠️ 局限性**

局限性：①未估计召回率，仅评估了精度；② API‑permission 映射不完整导致部分指标精度偏低；③ LL.M. 指南验证仅在有限场景下进行，未覆盖所有 Android 细节；④ 评估聚焦于代码层面，未验证与实际隐私政策或法规声明的一致性。

---

## 9. Glass Box at Orbit: A Constitutional AI Verification Framework for Trustworthy Autonomous CubeSat Intelligence

**arXiv ID:** 2606.02967 | [PDF](https://arxiv.org/pdf/2606.02967v1)

**作者:** Karthik Barma `[一作]` (Northeastern University), V C Premchand Yadav `[通讯]` (VIT-AP University)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并实现了Glass Box——一种针对CubeSat及轨道计算平台的运行时宪法AI验证层，能在每次AI决策前评估物理约束并生成可解释的审计日志。

**💡 创新点**

创新点在于将六个物理约束与七个线性时序逻辑安全不变式结合，形成O(N_c)线性复杂度的实时过滤器，并通过加权可解释性评分实现透明决策。

**🔧 技术方法**

采用了运行时验证、LTL模型检验（Z3、NuSMV）、INT8 TinyML推理、MPC调度、蒙特卡罗Dropout贝叶斯不确定性估计以及TMR硬件容错等技术。

**📊 数据集**

使用了NASA MODIS火点、ESA Sentinel‑2影像等公开卫星数据进行模型训练与仿真，但论文核心验证仅基于仿真环境（Orekit/GMAT）和自定义的轨道数字孪生。

**📈 对比分析**

文中未给出与传统方法的量化对比，仅通过示例展示Glass Box在日食入口时成功延迟推理而不致电量耗尽，说明其在安全性与可解释性方面优于无验证的推理方案。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证、约束语法固定、单卫星分析、缺乏实机硬件性能数据、未覆盖星座级治理等。

---

## 10. AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes

**arXiv ID:** 2606.02724 | [PDF](https://arxiv.org/pdf/2606.02724v1)

**作者:** Yaoting Wang `[一作]` (Fudan University), Henghui Ding `[通讯]` (Fudan University)

**通讯引用:** 4413 | [OpenAlex ID](https://openalex.org/A5036631624)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 AVTrack 数据集，专注于人类中心的音频视觉实例分割与跟踪，并给出了一个可扩展的三阶段基线 AVTracker；

**💡 创新点**

创新点在于：①设计了高复杂度的人类中心音频视觉场景评估标准；②构建了仅作为测试集的长期基准，强调跨模态长期追踪；③提出了动态窗口聚合与本地-全局分块的分层推理框架；

**🔧 技术方法**

采用 Whisper、MossFormer、SAM3、Qwen3‑VL 等主流模型进行语音识别、语音分离、图像分割与视觉推理；

**📊 数据集**

使用 AVTrack（871条视频，3120条实例轨迹）作为评测集，涵盖8类挑战；

**📈 对比分析**

与现有 VIS 与 AVIS 方法比较，VIS 在 AVTrack 上 HOTA 低于12%，AVIS 最高仅21%，而 AVTracker 在所有指标上平均提升约8个百分点，表现突出；

**⚠️ 局限性**

局限性包括：仅提供测试集，缺乏训练数据；基线依赖多种预训练模型，计算开销大；对极端遮挡与多说话者场景的鲁棒性仍有限。

---

## 11. Patcher: Post-Hoc Patching of Backdoored Large Language Models

**arXiv ID:** 2606.02995 | [PDF](https://arxiv.org/pdf/2606.02995v1)

**作者:** Anjun Gao `[一作]` (University of Louisville), Minghong Fang `[通讯]` (University of Louisville)

**通讯引用:** 1806 | [OpenAlex ID](https://openalex.org/A5056811906)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对在LLM微调过程中植入的jailbreak backdoor，提出一种后置修补框架Patcher，只利用一次报错实例和模型参数即可定位并消除触发器，恢复模型安全性与功能。

**💡 创新点**

创新点：①首次将梯度基因重要性（response‑conditioned saliency）与自适应K‑means聚类相结合，实现单实例触发器定位；②在修补阶段引入拒绝监督与KL锚定双重约束，既消除触发器关联，又保持正常任务性能和对非触发式攻击的鲁棒性；③兼顾多种攻击场景（标准backdoor、jailbreak backdoor、适应性攻击）并在大规模模型上验证。

**🔧 技术方法**

技术：梯度梯度反向传播计算每个输入标记的Saliency；K‑means聚类分离触发器；基于LoRA的微调，目标为拒绝损失 + KL锚定；随机序列注入生成触发器样本；对比实验使用多种攻击策略与防御基线。

**📊 数据集**

数据集：任务数据集包括SST‑2、CoLA、GSM‑8K、AG‑News；安全对齐数据集为BeaverTails、HH‑RLHF、XSTest等；评估时采用StrongREJECT、Llama‑Guard‑3、GPT‑4.1‑mini 等自动判定器。

**📈 对比分析**

与 Fine‑Tuning、Fine‑Pruning、Mudjacking、SPP、BAERASER、OneShot、MEND、ROME 等对手方法对比，Patcher在所有攻击与模型（Llama‑3.1‑8B、Qwen2.5‑7B、Falcon‑3‑7B）下将ASR降至近0，同时ACC保持在无攻击模型水平；运行时间仅比普通微调略高，远快于复杂编辑或重训练方法。

**⚠️ 局限性**

局限：①只针对离散 token 触发的 backdoor，无法处理软提示或直接参数篡改的隐式触发；②假设所有恶意请求均应以拒绝回复，忽略可能的上下文可接受性；③评估依赖自动判定器，缺少人类验证；④对高度噪声或不完整报错实例的鲁棒性仍有待进一步验证。

---

## 12. Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate

**arXiv ID:** 2606.02670 | [PDF](https://arxiv.org/pdf/2606.02670v1)

**作者:** Marc Pinet `[一作]` (Orange Research), Dominique Vaufreydaz `[通讯]` (University of Grenoble Alpes)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `79276348-11e0-48e3-84bc-7ec231d0171c` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并使用了基于每个标注异常段的单变量与跨通道诊断框架，对公开多变量时间序列异常检测基准进行结构分析。

**💡 创新点**

创新点在于：①以诊断方法直接评估异常段的单变量与跨通道特征；②通过合成数据验证诊断的有效性；③在真实基准上比较通道相关与独立模型，证明通道依赖无显著优势。

**🔧 技术方法**

技术手段包括：z-score 单变量检验、Pearson、Spearman 及无偏距离相关的跨通道检验、滞后相关扩展、矩阵概况（Matrix Profile）补充、合成时相移与噪声翻转数据、线性与多层感知器自编码器、CrossAD 与 CATCH 模型等。

**📊 数据集**

使用了八个常见的多变量异常检测基准：GECCO、NASA（MSL、SMAP）、PSM、SMD、SWaT、SWAN‑SF、WADI。

**📈 对比分析**

通过诊断框架对所有标注异常段进行分类，发现“纯跨通道”异常几乎不存在；在合成数据上，通道依赖模型（如线性自编码器）能显著恢复跨通道异常，而通道独立模型几乎无能量；在真实基准上，通道依赖版本的 CrossAD 与其通道独立版本性能相同甚至更差，说明通道相关特征在这些基准中并不必要。

**⚠️ 局限性**

局限性包括：①仅评估已标注的异常，若基准中存在未标注的跨通道异常则无法检测；②通道语义和时间步标注缺失，限制对跨通道关系的解释；③实验主要基于公开基准，结果可能不适用于具有真实跨通道异常的工业系统。

---

## 13. The Shadow Price of Reasoning: Economic Perspective on Optimal Budget Allocation for LLMs

**arXiv ID:** 2606.03092 | [PDF](https://arxiv.org/pdf/2606.03092v1)

**作者:** Xu Wan `[一作]` (Zhejiang University), Mingyang Sun `[通讯]` (Peking University)

**通讯引用:** 5181 | [OpenAlex ID](https://openalex.org/A5079378336)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个基于全局影子价格的推理时预算分配框架 CLEAR，能够在有限 token 预算下通过阈值预测、Lambert‑W 闭式分配和理性放弃，实现对异构推理任务的最优计算资源分配。

**💡 创新点**

创新点在于把推理预算分配建模为非凸约束优化，通过经济学的影子价格实现全局最优，并给出 Lambert‑W 闭式解，能够自动决定何时放弃任务，显著提升稀缺预算下的准确率。

**🔧 技术方法**

主要技术包括：阈值预测网络、Shifted Surge 潜在效用模型、Lambert‑W 解析分配公式、影子价格的二分搜索、理性放弃决策。

**📊 数据集**

实验使用的主要数据集包括：数学推理基准（GSM‑8K、MATH 等）和代码生成基准（HumanEval、MBPP、BigCodeBench）。

**📈 对比分析**

与统一分配、基于阈值的比例分配、外部长度预测等基线对比，CLEAR 在 256 token/查询等资源紧张场景下提升 10~24 分点准确率，整体在成本‑准确率 Pareto 前沿显著优于其他方法。

**⚠️ 局限性**

主要局限在于对阈值预测误差敏感、依赖固定的 α、β 超参数，且目前仅在推理时分配上验证，尚未验证在多模型或多任务的跨域适用性。

---

## 14. ASymPO: Asymmetric-Scale Policy Optimization for Asynchronous LLM Post-Training Without Behavior Information

**arXiv ID:** 2606.03070 | [PDF](https://arxiv.org/pdf/2606.03070v1)

**作者:** Zehua Liu `[一作]` (Huawei Technologies), Mingxuan Yuan `[通讯]` (Huawei Technologies)

**通讯引用:** 2179 | [OpenAlex ID](https://openalex.org/A5078949174)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了两种仅使用当前策略概率的异步强化学习目标——SPO和ASymPO，以解决分布漂移导致的损失尺度失衡问题。

**💡 创新点**

创新点在于通过自适应的响应级别归一化（ASymPO）直接利用当前策略的负对数概率平衡正负样本的损失贡献，避免了传统方法中对行为策略概率的依赖。

**🔧 技术方法**

使用的技术包括基于组相对优势的策略梯度、当前策略的负对数概率计算、比例缩放（SPO）和自适应归一化（ASymPO），并在异步rollout–learner架构中实现。

**📊 数据集**

实验数据集涵盖数学推理任务，包括MATH、AIME24/25、AMC23、GSM8K和Minerva-Math，使用Qwen3-1.7B/4B、LLaMA-3.2-3B-Instruct三种模型。

**📈 对比分析**

与传统需要行为策略概率的GRPO、无重要性采样的GPG以及基线Naive Loss比较，ASymPO在大多数指标上保持稳定训练并取得与GRPO相近甚至优于的平均准确率；SPO表现略逊，但仍优于Naive和GPG。

**⚠️ 局限性**

局限性包括：仍未完全解决行为策略与当前策略之间的大幅偏离；归一化方法为启发式，可能受优势归一化、响应长度变化和极端概率影响；仅在可验证奖励的数学推理任务上验证，未覆盖更广泛的RLHF或多步交互场景。

---

## 15. IdiomX A Multilingual Benchmark for Idiom Understanding, Retrieval, and Interpretation

**arXiv ID:** 2606.02584 | [PDF](https://arxiv.org/pdf/2606.02584v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 16. Graph Mamba Survival Analysis Based on Topology-Aware ordering

**arXiv ID:** 2606.02602 | [PDF](https://arxiv.org/pdf/2606.02602v1)

**作者:** Yuanfang Chen `[一作]` (Xi'an Jiaotong University), Xiangyong Cao `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 2905 | [OpenAlex ID](https://openalex.org/A5028103486)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

针对WSI生存分析，提出了基于拓扑感知排序的图Mamba框架TopoMamSurv；

**💡 创新点**

创新点在于使用基于最短路径的拓扑感知排序结合双向Mamba与GCN的多分支结构，实现了顺序敏感性与图结构的充分融合；

**🔧 技术方法**

主要技术包括Mamba状态空间模型、双向Mamba、图卷积网络（GCN）以及多分支融合与节点对采样排序；

**📊 数据集**

实验使用TCGA的BLCA、BRCA、GBMLGG、LUAD、UCEC五个癌症数据集；

**📈 对比分析**

与10个基线方法（如GraphMamba、MambaMIL、TransMIL等）比较，TopoMamSurv在四个数据集上获得最高C-index，GBMLGG亦保持竞争力，整体提升约3–6个百分点；

**⚠️ 局限性**

局限在于需要采样节点对进行排序，计算成本随图大小增长；对极大图的扩展以及不同癌症类型的泛化性仍需进一步验证。

---

## 17. CARVE: Certified Affordable Repair of Vetoed Maneuvers via Envelopes for Interactive Driving

**arXiv ID:** 2606.02641 | [PDF](https://arxiv.org/pdf/2606.02641v1)

**作者:** Yifan Wang `[一作]` (McGill University), Yifan Wang `[通讯]` (McGill University)

**通讯引用:** 3835 | [OpenAlex ID](https://openalex.org/A5100398537)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种交互式维修认证层 CARVE，能够在规则被触发的情况下生成可验证的多主体修复证书。

**💡 创新点**

创新点在于将失败模式视为可证明的“维修证书”问题，设计了基于右行驶权缩放的合作包络、有限多主机操作 lattice 与最优分摊成本的严谨证明框架。

**🔧 技术方法**

主要技术包括：基于规则的硬阈值检查、合作包络 β(π) 规范化、有限多主机操作网格的分枝限界搜索与贪心求解、责任加权成本 Φ 与可负担性检查。

**📊 数据集**

使用 INTERACTION 交通行为数据集（589 条 Lanelet2 地图约束回放场景）进行离线评估，并未在训练中使用该数据。

**📈 对比分析**

与 HardPrune、EgoOnly-Greedy/Exact 等基线相比，CARVE Greedy 以 98.64% 的接受率恢复 97.88% 的误拒案例，保持 100% 右行驶权尊重、零优先级错误、BCR 100%，并且在 400/400 的负压拒绝场景中保持安全；相对 AlphaOnly-CARVE 的 14 例优先级误拒，CARVE 完全避免了此类问题。

**⚠️ 局限性**

局限性包括：依赖预先声明的规则、margin 与合作包络，无法处理动态多主体密集场景；仅验证离线回放，缺乏闭环动态验证；对 1–3 车冲突的有限 lattice 设计，需进一步扩展至更复杂场景。

---

## 18. RMPrior: Bridging Propagation Priors and Diffusion Refinement for Efficient Radio Map Construction

**arXiv ID:** 2606.03074 | [PDF](https://arxiv.org/pdf/2606.03074v1)

**作者:** Zixuan Guo `[一作]` (Xidian University), Nan Cheng `[通讯]` (Xidian University)

**通讯引用:** 99601 | [OpenAlex ID](https://openalex.org/A5100773343)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种“中点启动”扩散采样策略，将物理传播先验与扩散模型的细化过程结合，用已匹配的传播先验作为扩散的中间状态，缩短逆向采样步数

**💡 创新点**

创新点在于：1）只使用已有的传播先验进行前向噪声映射到中间时间点；2）在预训练的扩散模型上直接执行剩余逆向步骤，无需重新训练；3）理论上证明截断对重建误差的影响与先验质量和截断深度有关

**🔧 技术方法**

使用DDPM/DDIM的条件扩散模型（RMDM骨干），前向噪声映射、逆向采样和误差分析，结合基于主导路径模型（DPM）的传播先验

**📊 数据集**

在高分辨率射线追踪数据集 IRT4HighRes（7920 个样本）上进行实验，同时用 720 采样子集做先验质量消融和稀疏子集做 P_start 敏感性评估

**📈 对比分析**

与 RME-GAN、RadioUNet 等基准方法比较，并在不同 P_start（0.25–1.0）下评估 NMSE、RMSE、SSIM、PSNR 及推理时延；在 P_start=0.5 时实现 2.01× 的速度提升，同时提升所有质量指标（PSNR 35.39 dB，NMSE 0.00683 等）

**⚠️ 局限性**

局限性包括：1）对先验质量高度依赖，低质量先验在强截断时会导致显著性能下降；2）截断深度需根据场景复杂度手动设定，缺乏自适应机制；3）实验仅在单一扩散骨干和单一射线追踪数据集上验证，泛化性待进一步评估

---

## 19. A Benchmarking Framework for Multimodal User Interface Toolkits: Comparing Modality Coverage, Developer Workflow, and Experimental Support

**arXiv ID:** 2606.02977 | [PDF](https://arxiv.org/pdf/2606.02977v1)

**作者:** Ariton Verush `[一作]` `[通讯]` (University of Bern), Ariton Verush (University of Bern)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可复用的多模态用户界面工具包基准框架，帮助评估工具包的模态覆盖、开发者体验与实验支持。

**💡 创新点**

创新点在于将基准维度拆分为三大维度（模态覆盖、开发者体验、实验支持），并提供可实例化的评价模板，首次在多模态工具包领域推广系统化基准方法。

**🔧 技术方法**

利用文档与 API 分析、原型任务、开发者研究以及技术指标测评等混合方法，框架不依赖单一技术，而是聚焦于评估流程和指标体系。

**📊 数据集**

未使用特定公开数据集；评估依赖于公开文档、示例代码和可重现的实验设置。

**📈 对比分析**

通过对 Geno、MSP、ReactGenie、WAMI、EmoSync 五个代表性工具包的结构化比较，框架为未来的实证研究提供了量化指标和对照基准；性能表现取决于后续实验数据，当前仅呈现概念性评估。

**⚠️ 局限性**

局限在于未完成实证验证；框架聚焦于五个示例工具包，可能无法覆盖全部多模态工具；LLM 驱动工具包版本变化快，需额外记录版本与生成细节。

---

## 20. SeeTraceAct: Visibility-Aware Latent Planning from Cross-Embodiment Demonstration Videos

**arXiv ID:** 2606.02745 | [PDF](https://arxiv.org/pdf/2606.02745v1)

**作者:** Jaehyeon Son `[一作]` (Georgia Institute of Technology), Zsolt Kira `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5533 | [OpenAlex ID](https://openalex.org/A5088892134)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 SeeTraceAct，一种通过可见性感知的未来末端执行轨迹监督来实现单示例演示条件的视觉‑语言‑动作模型，以解决跨体型演示中的精确空间定位难题。

**💡 创新点**

创新点在于将可见性感知的视觉轨迹预测作为辅助目标，使模型在缺乏轨迹标签时仍能学习精确的空间规划，并构建了可供跨体型演示的 RoboCasa‑DC 基准。

**🔧 技术方法**

采用 GR00T N1.5 变压器骨干、V‑JEPA 视频编码器、Perceiver Resampler、可见性判别头的轨迹解码器以及流匹配动作预测等技术实现模型训练和推理。

**📊 数据集**

使用 RoboCasa‑DC（含同体型与跨体型演示对）和真实世界 Franka Panda 机器人桌面任务数据集。

**📈 对比分析**

与 Vid2Robot、UniSkill、ViVLA 三种基线相比，SeeTraceAct 在四种评估设置下均取得最高成功率，尤其在精度敏感的跨体型任务上提升约3个百分点，真实世界平均成功率提升 12.5%。

**⚠️ 局限性**

局限性包括仅在单臂桌面任务上验证、需要事先提供未来轨迹标签、以及整体成功率仍偏低，仍有提升空间。

---

## 21. SkillDAG: Self-Evolving Typed Skill Graphs for LLM Skill Selection at Scale

**arXiv ID:** 2606.03056 | [PDF](https://arxiv.org/pdf/2606.03056v1)

**作者:** Tong Bai `[一作]` (Fudan University), Ivor W. Tsang `[通讯]` (A*STAR)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向大型语言模型代理的可调用结构化检索接口（///），通过维护并在线更新一个带类型的技能关系图，帮助代理在执行过程中更精确地选择、组合和避免技能。

**💡 创新点**

创新点在于：①将技能关系图从固定检索算法转变为代理可调用的接口，让LLM直接利用结构化证据进行决策；②设计了三字段检索（匹配、邻居、冲突）与 propose‑then‑commit 编辑协议，保证图的无环性、无矛盾性并支持可回滚；③采用双视角嵌入（自描述与需求描述）实现冷启动图构造，弥补传统相似度检索忽略的前置与冲突关系。

**🔧 技术方法**

技术包括：typed directed graph（包含 prerequisite, specialization, synergy, redundancy, conflict 5种边类型）；LLM 对 graph 进行查询接口（matches、neighbors、conflicts）；基于向量相似度和 LLM 分类的冷启动图构造；在线 propose‑then‑commit 协议与三条结构性约束；在 MiniMax‑M2.7 与 gpt‑5.2‑codex 后端上实现。

**📊 数据集**

使用了 ALFWorld（文本模拟环境）和 SkillsBench（容器化代码生成任务）两个基准；在 SkillsBench 提供 200/500/1000/2000 规模的技能库进行检索鲁棒性实验。

**📈 对比分析**

与三类基线（全文库提示、向量检索 top‑K、Graph‑of‑Skills）对比，/// 在 MiniMax‑M2.7 后端下提升 ALFWorld 成功率从 54.3% 提升到 67.1%（+12.8%），SkillsBench 奖励从 18.7% 提升到 27.3%（+8.6%）；在 gpt‑5.2‑codex 后端下，ALFWorld 维持 93.6%，SkillsBench 仅提升 2.4%；在检索指标上 Ret@K、Ret@1、MRR 均显著高于 Graph‑of‑Skills，且在技能库扩大 10 倍时仅下降 3.5% Ret@5，表现出优异的规模鲁棒性。

**⚠️ 局限性**

局限性：提出‑提交协议完全由代理决定，单次执行即可提交边，但缺乏统计置信度阈值，可能在长时序任务中出现单观测误差；当前实验聚焦短期任务，对长期自演化行为的稳定性与收敛性尚未验证。

---

## 22. Traj-Evolve: A Self-Evolving Multi-Agent System for Patient Trajectory Modeling in Lung Cancer Early Detection

**arXiv ID:** 2606.02812 | [PDF](https://arxiv.org/pdf/2606.02812v1)

**作者:** Sihang Zeng `[一作]` (University of Washington), Meliha Yetisgen `[通讯]` (University of Washington)

**通讯引用:** 1351 | [OpenAlex ID](https://openalex.org/A5002548520)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了 Traj-Evolve，一种自我演化的多代理系统，利用经验池和多代理强化学习来预测肺癌早期风险。

**💡 创新点**

创新点在于将非参数经验检索（ExPool）与参数化强化学习（MARL）结合，实现系统在实时处理患者时不断吸收已验证案例、提升临床推理和模型参数。

**🔧 技术方法**

采用 GPT-OSS-20B 作为基础 LLM，配合 nomic-embed-text-v1.5 做嵌入检索；使用奖励排名微调（RAFT）和 QLoRA 进行多代理强化学习；通过跨检索和留一交叉检索统一训练与推理。

**📊 数据集**

使用单机构电子病历数据库（约13,600例训练，1,000例整体测试，835例无烟者子集），包含诊断、检验、处置、药物和非结构化文本，最长5年历史。

**📈 对比分析**

与五大类基线（临床风险模型、监督机器学习、序列深度学习、临床 BERT、LLM 基础管线）对比，Traj-Evolve 在整体人群 AUROC 0.86、AUPRC 0.32、F1 0.42，永不吸烟人群 AUROC 0.84，均显著优于最佳静态 LLM Baseline（Traj-CoA AUROC 0.81）和传统模型。

**⚠️ 局限性**

局限包括：单中心回顾性设计、需要已知诊断标签的自我演化、标签确认延迟及噪声、未评估跨机构泛化及实时流式部署。

---

## 23. Scalable Uncertainty Quantification for Extreme Weather Forecasting via Empirical Neural Tangent Kernels

**arXiv ID:** 2606.02886 | [PDF](https://arxiv.org/pdf/2606.02886v1)

**作者:** Jose Marie Antonio Miñoza `[一作]` (Department of Education), Sebastian C. Ibañez `[通讯]` (Department of Education)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

为已有的无确定性输出的深度天气模型提供一种后处理的、无须重训练的不确定性量化方法 NTK-UQ。

**💡 创新点**

提出通过模型最后一层特征构造经验 Neural Tangent Kernel，并基于其 Gaussian Process 后验推导不确定性；同时提供基于特征谱浓度的自适应 ICA/SVD 选择规则，解决不同网络结构下的方差坍塌和非高斯极端事件特征提取问题。

**🔧 技术方法**

使用经验 NTK‑GP 近似、奇异值分解（SVD）、独立成分分析（ICA）、后处理比例缩放、以及自回归特征提取；评估指标包括覆盖率、尖锐度（Sharpness）和 CRPS。

**📊 数据集**

使用 ERA5 重分析作为训练与校准数据，评估集为 2021 年 EM‑DAT 极端天气事件（100 个初始化日期，涵盖飓风、洪水、干旱、热浪等）。

**📈 对比分析**

与分割共形预测（split conformal prediction）对比，NTK‑UQ 在 90% 覆盖率下实现 31–37% 更尖锐的预测区间，且能生成自适应宽度的区间；在不同模型（FourCastNetV2、Pangu‑Weather、Aurora、AIFS）和不同分解方法（SVD/ICA）上，ICA 通常获得更优的 CRPS 与覆盖率。

**⚠️ 局限性**

局限性包括：需预先对特征谱进行诊断，方差坍塌易导致极端事件判别失效；只考虑最后一层特征，忽略早期层不确定性；缺乏分布无关的覆盖保证；在跨年、跨分辨率验证上尚未完成，需进一步验证。

---

## 24. Qift: Shift-Friendly No-Zero W2 Post-Training Quantization for Rotated W2A4/KV4 LLM Inference

**arXiv ID:** 2606.02823 | [PDF](https://arxiv.org/pdf/2606.02823v1)

**作者:** Chi-Wei Huang `[一作]` (National Cheng Kung University), Chia-Chi Tsai `[通讯]` (National Cheng Kung University)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5007783236)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出在Hadamard旋转后的LLM权重两位量化中，采用固定无零点的四层级重构级别（{±0.5,±1.5}和{±1,±4}），从而在W2A4/KV4推理中显著提升精度。

**💡 创新点**

创新点在于将两位量化的重构级别视为设计变量，针对旋转后近似零中心、Gaussian-like 的权重分布，设计了镜像无零点级别，满足内外中心比约0.25–0.33，且无需学习代码表、零点或分组，实现了训练、校准和硬件友好的固定量化方案。

**🔧 技术方法**

使用Hadamard旋转的后训练量化（PTQ）、GPTQ/GPTAQ校准、RTN桶诊断、比例分析、混合精度层级选择等技术；通过对四层级重构级别的几何设计与全局尺度搜索，实现了无学习、无零点的量化。

**📊 数据集**

实验基于LLaMA‑2‑7B和LLaMA‑3.1‑8B模型，使用WikiText‑2（128个样本）进行校准，评估下游任务包括ARC‑Challenge、ARC‑Easy、HellaSwag、PIQA、WinoGrande。

**📈 对比分析**

与标准{-2,-1,0,+1}、对称/非对称W2、Lloyd‑Max、NF2等基线比较，纯W2A4下PPL从50+降至约7.3，L=16混合精度下与W3A4的差距缩小一半；下游平均准确率提升约0.02。性能提升主要来自于内外中心比匹配与无零点设计。

**⚠️ 局限性**

局限性包括：仍无法完全达到W3A4的精度；设计仅针对Hadamard旋转的后训练管线，未验证其他旋转或训练时量化；缺乏对不同模型规模、硬件实现细节以及训练期间鲁棒性的深入评估。

---

## 25. Hallucinations as Orthogonal Noise: Inference-Time Manifold Alignment via Dynamic Contextual Orthogonalization

**arXiv ID:** 2606.03022 | [PDF](https://arxiv.org/pdf/2606.03022v1)

**作者:** Mingkuan Zhao `[一作]` (Xi'an Jiaotong University), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 57257 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种推理时干预方法 Dynamic Contextual Orthogonalization (DCO)，通过在每层注意力输出中对齐残差流的语义子空间来抑制幻觉，保持上下文一致性。

**💡 创新点**

创新点在于将幻觉解释为正交噪声，并利用动态正交化与层级 Z‑score 抑制机制实现自适应干预；与传统静态干预或解码层对比，能在不损失模型参数知识的前提下提升生成可信度。

**🔧 技术方法**

使用 RMSNorm 生成动态上下文锚点，对注意力头输出做线性投影得到并行的平行与正交分量；通过层级 Z‑score 统计与 Sigmoid 门控动态抑制正交噪声；整体在单个前向传递中完成，计算复杂度为 O(L·H·d_model)。

**📊 数据集**

在 Llama‑3‑8B 与 70B 模型上，使用 XSum、NQ‑Swap、IFEval、TruthfulQA、TriviaQA、NQ‑Open、MuSiQue 等公开基准数据集进行评测。

**📈 对比分析**

与 ITI、DoLa、CAD、DeCoRe、AD 等基线方法比较，DCO 在上下文忠实度（ROUGE‑L、BERTScore、FactKB）、指令遵循、知识保留和多跳推理上均表现出显著提升；例如在 70B 模型上 XSum ROUGE‑L 22.40、IFEval Prompt‑级精度 85.77%、MuSiQue EM（闭书+CoT）21.14，均优于或接近最佳基线。

**⚠️ 局限性**

限制：正交噪声假设对完全无上下文或纯闭书生成场景不够稳健；需要手动调节 Z‑score 阈值 τ 与温度 β，可能在不同任务间需要重调；虽然 FLOPs 增加量小，但在极端低延迟环境下仍需评估其实际负载。

---

## 26. Streami: An MPI Data-Parallel Library to Compute Field Lines on GPUs

**arXiv ID:** 2606.02627 | [PDF](https://arxiv.org/pdf/2606.02627v1)

**作者:** Stefan Zellmann `[一作]` (University of Cologne), Tatiana von Landesberger `[通讯]` (University of Cologne)

**通讯引用:** 2285 | [OpenAlex ID](https://openalex.org/A5015182248)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一个面向多 GPU HPC 集群的高性能场线计算库（streami），实现了可插拔的低层向量场抽象和高层追踪器，支持流线和迹线的并行计算。

**💡 创新点**

创新点在于提供统一的 CUDA+MPI 数据并行接口，直接复用 CFD 领域分区，避免数据复制，并支持结构化与非结构化网格的编译时多态实现。

**🔧 技术方法**

采用 CUDA/GPU 核心加速粒子推进，MPI（CUDA-aware）进行 GPU‑GPU 通信，并使用模板编程实现低层抽象；高层封装了 Tracer 类，支持实时交互式种子点放置。

**📊 数据集**

实验使用 astrophysics 数据集（1K³ 体素）和 wind farm 数据集，亦将体素转换为无结构六面体进行验证。

**📈 对比分析**

通过在 16 A100 GPU（两节点）上测定 10K 次推进的平均 1–2 ms，显示在局部推进阶段性能优异，但总体受 MPI 通信瓶颈限制，未与现有工具直接对比。

**⚠️ 局限性**

局限在于目前仅支持结构化和基本无结构网格，未能处理大规模无结构数据（在 1、2、4 GPU 上内存不足），并且需要用户自行提供与库匹配的向量场实现。

---

## 27. On the Persistent Effects of Lexicality in Large Language Mod

**arXiv ID:** 2606.02750 | [PDF](https://arxiv.org/pdf/2606.02750v1)

**作者:** Hammad Rizwan `[一作]` (Dalhousie University), Hassan Sajjad `[通讯]` (Dalhousie University)

**通讯引用:** 2730 | [OpenAlex ID](https://openalex.org/A5042954793)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统性地分析了大语言模型在不同深度层级上的词汇影响，揭示了词汇重叠在多模型、不同训练策略下仍然显著且持续存在的现象；

**💡 创新点**

首次将对抗性语义平衡三元组测试与词汇可解码性、语义保真度、熵变化等多维度指标结合，发现中层出现词汇与语义信号低谷的共同特征，并将其与信息理论的压缩-再扩展轨迹关联；

**🔧 技术方法**

采用三元组语义平衡对抗测试、线性词汇识别探针、MTEB任务评估以及每层提示熵等技术，对LLM的每一层隐藏状态进行多维度探测与量化；

**📊 数据集**

使用CounterFact、SugarCrepe++、WikiText、MTEB、Extreme Summarization、CounterFact（编辑版）等公开数据集，覆盖实体、属性、关系及摘要评估等场景；

**📈 对比分析**

对预训练、指令微调、度量学习三种训练范式的模型在多层级下进行比较，发现词汇影响在中层峰值、后层逐渐减弱，但即使在度量学习模型中仍显著；在摘要评估与模型编辑案例中，BERTScore、BARTScore等语义指标亦被词汇重叠误导，编辑对词汇相似输入的冲击更大；

**⚠️ 局限性**

局限在于对抗测试仅覆盖词汇重叠的表层冲突，未能完全覆盖更深层语义细微差异；探针与熵指标仍是线性或单一维度的近似，可能忽略非线性或结构性信息；实验仅涵盖有限模型族，结果对更广泛的LLM体系仍需进一步验证。

---

## 28. Zero-Shot 3D Question Answering via Hierarchical View-to-Token Transportation

**arXiv ID:** 2606.03100 | [PDF](https://arxiv.org/pdf/2606.03100v1)

**作者:** Dongsheng Wang `[一作]` (Shenzhen University), Hui Huang `[通讯]` (Shenzhen University)

**通讯引用:** 16161 | [OpenAlex ID](https://openalex.org/A5087787304)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种两层层次的 3D 场景信息收集框架（KeyV + KeyT），先通过几何信息挑选空间一致且与问题相关的关键视角，再用最优传输（OT）压缩这些视角的视觉 token，减少冗余并保持多样性，最终让 2D 视觉语言模型在有限输入预算下获得更丰富的 3D 上下文；

**💡 创新点**

创新点在于：①将相机参数与语义相似度结合，形成空间一致的关键视角选择；②在视角级别之外引入 OT‑guided token 压缩，利用概率传输计划选取代表性 token；③整体实现无监督、无训练的 tuning‑free 方法；

**🔧 技术方法**

关键技术包括：几何感知的视角重要性计算、子场景分区与权重分配、BLIP2 语义相似度评估、OT（Sinkhorn）最优传输求解、虚拟 token 生成与基于传输计划的 token 选择；

**📊 数据集**

在 ScanQA、SQA3D 和 VSI‑Bench 这三个公开 3D 问答基准上进行评测；

**📈 对比分析**

与多种基线（如 AKS、DivPrune、FLoC、cdViews、Video‑3D 等）以及多种 2D VLM backbone（LLaVA‑OneVision、Qwen2.5‑VL、LLaVA‑Video）对比，实验表明 KeyV+KeyT 在所有评测指标上均超越传统关键视角选择与 token 压缩方法，且在 tuning‑based 方法上能达到或超过其性能；

**⚠️ 局限性**

局限性在于：对相机参数噪声仍有一定敏感性（10% 噪声时性能略降）；缺乏对动态场景或更大规模点云的深入评估；以及 OT 计算虽然已使用 Sinkhorn 近似，但在极大 token 数量时仍存在计算瓶颈；

---

## 29. Motion Planning in Dynamic Environments: A Survey from Classical to Modern Methods

**arXiv ID:** 2606.02677 | [PDF](https://arxiv.org/pdf/2606.02677v1)

**作者:** Zongyuan Shen `[一作]` (Jinan University), Long Cheng `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 473685 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对 2015-2025 年间 138 篇关于动态环境下运动规划的研究进行系统综述，涵盖从经典采样、图搜索、MPC、学习到局部规划等五大类别。

**💡 创新点**

首次提出以动态感知为辅助的完整规划框架，并将学习方法与经典方法结合的混合规划技术系统化呈现，形成统一的分类与比较体系。

**🔧 技术方法**

通过对采样、图搜索、MPC、强化学习、传统局部规划等多种技术的梳理与评析，构建了完整的技术路线图。

**📊 数据集**

本文不基于单一数据集，而是综合引用了 138 篇论文的实验结果与评测数据，对比不同方法的优缺点。

**📈 对比分析**

利用表格与定性比较展示各方法在效率、可行性、鲁棒性等方面的相对表现，指出基于采样方法在稀疏环境中快速重规划、MPC 在动态约束下保真度高、学习方法在社交适应性方面表现突出。

**⚠️ 局限性**

局限性在于仅覆盖 2015-2025 年的文献，未包含最近的 LLM/多模态感知融合等前沿工作；同时缺乏统一的实验基准与真实世界验证，导致对方法间客观性能差异的量化比较受限。

---

## 30. Multi-Modal Machine Learning for Breast Cancer Recurrence Prediction

**arXiv ID:** 2606.02892 | [PDF](https://arxiv.org/pdf/2606.02892v1)

**作者:** Jiahao Shao `[一作]` (University of Tennessee), Bing Yao `[通讯]` (University of Tennessee)

**通讯引用:** 22525 | [OpenAlex ID](https://openalex.org/A5048456721)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了一套多模态临床数据融合框架，用规则式正则表达式与优先级冲突调和机制，将病理报告、治疗记录与注册表中的信息统一为可学习格式，并用于乳腺癌复发预测。

**💡 创新点**

创新点在于：① 将病理文本中的高可信度信息通过规则抽取并与结构化数据冲突调和；② 引入先后级优先级原则（PATH≻ABS≻TS），在保证时间一致性的前提下最大化信息完整度；③ 在单源、双源与多源条件下系统评估模型表现，验证多源融合带来的显著提升。

**🔧 技术方法**

技术包括：正则表达式规则抽取、文本标准化与语义消歧、优先级冲突调和算法、特征编码（序数/独热+Z-score）、多模型监督学习（RF、XGB、RGF、TabNet），以及TabNet的注意力特征重要性解释。

**📊 数据集**

使用了田纳西大学医疗中心（UTMC）2007‑2025年间的乳腺癌病例，共6060例（5845名患者），包括结构化临床数据、年度病理注册数据和原始病理PDF文本。

**📈 对比分析**

通过对比单源、全源与基准（16个常用变量）三种特征配置，在三种控制不平衡比例（1:1、1:2、1:3）和原始不平衡样本下，使用四种模型进行评估。结果显示，多源融合平均提升AUROC约0.039、AUPRC约0.056、F1约0.042、G‑Mean约0.047，尤其在1:1平衡下提升显著，跨模型、跨不平衡度均保持正向效应。

**⚠️ 局限性**

局限在于未覆盖手术切缘、放疗依从性等细粒度风险因素的抽取与时间逻辑；对模型校准、外部验证缺乏；并且规则式抽取需人工迭代，缺乏自动化学习能力。

---

## 31. Improvise, Adapt, Overcome: An On-The-Fly Multifidelity Algorithm for Efficient Machine Learning

**arXiv ID:** 2606.02662 | [PDF](https://arxiv.org/pdf/2606.02662v1)

**作者:** Vivin Vinod `[一作]` (University of Wuppertal), Peter Zaspel `[通讯]` (University of Wuppertal)

**通讯引用:** 401 | [OpenAlex ID](https://openalex.org/A5041407886)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种自适应的多精度机器学习框架，能够在训练过程中动态决定每个精度层次所需的样本数，从而显著减少高精度数据生成的冗余。

**💡 创新点**

创新点在于引入了基于误差阈值的在线采样策略，打破传统固定比例或启发式设定，实时衡量每批数据的边际信息增益，实现真正的成本自适应优化。

**🔧 技术方法**

采用核岭回归（KRR）作为子模型，配合层级校正（Δ-ML）和可插值学习曲线的技术，形成可扩展的多精度推断架构。

**📊 数据集**

使用了VIB5（CCSD(T) PES）、QeMFi（SCF 与激发能）以及ANI-1ccx（CCSD(T) 能量）等公开化学数据集进行验证。

**📈 对比分析**

与单精度KRR以及传统固定比例（scale=2）MFML对比，实验显示自适应MFML在达到目标MAE时比单精度快30倍、比传统MFML快5倍，且在更复杂的激发能预测中也保持了显著的性能提升。

**⚠️ 局限性**

主要局限包括采样过程的串行化导致高精度计算的顺序执行，需在并行化或预训练全局模型等方向进一步改进，以及在极大数据规模下的可扩展性待验证。

---

## 32. Are we really tilting? The mechanics of reward guidance in flow and diffusion models

**arXiv ID:** 2606.02884 | [PDF](https://arxiv.org/pdf/2606.02884v1)

**作者:** Sanjit Dandapanthula `[一作]` (Carnegie Mellon University), Nicholas M. Boffi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5054972412)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本论文分析了奖励引导（reward guidance）在流式与扩散生成模型中的偏差机制，证明了有限粒子插件估计（plug‑in estimator）是导致奖励操纵（reward hacking）的根源，并提出了一种闭式奖励衰减（reward damping）时间表与最佳‑n 采样（best‑of‑n）两种有效的缓解策略；

**💡 创新点**

创新点在于：①用解析方法揭示有限粒子插件估计在高斯与高斯混合目标下导致的两种失败模式——内模奖励操纵与无法跨模选择高奖励模式；②基于这一理论设计了无额外计算成本的奖励衰减调度；③阐明最佳‑n 采样在模式选择中的作用，形成完整的偏差补偿框架；

**🔧 技术方法**

主要技术包括 Doob h‑transform、流匹配与概率流 ODE、k‑粒子插件估计、闭式高斯分析、奖励衰减时间表、最佳‑n 采样、以及在 FLUX.1 文本‑图像模型上的实验实现；

**📊 数据集**

使用的数据集与环境为：模拟 Gaussian 目标与高斯混合、二维 checkerboard（手工构造）以及 FLUX.1 大规模文本‑图像生成任务（包含 ImageReward、blueness、masked‑intensity 以及 VLM‑reward 等多种奖励函数）；

**📈 对比分析**

实验通过与理想解析倾斜（analytic tilt）、k=1 插件、k=8 插件、奖励衰减、最佳‑n、最佳‑n+衰减等多种设置进行对比。结果显示：奖励衰减显著降低奖励操纵而不增加算力，最佳‑n 能够恢复模式选择，二者组合能在奖励得分与分布相似度上接近理想倾斜，优于单纯使用插件或 k=8 的方案；

**⚠️ 局限性**

局限性主要体现在：理论分析仅覆盖高斯/高斯混合目标与二次奖励，未能推广到非高斯或非二次奖励；实验范围局限于二维棋盘和 FLUX.1 文本‑图像任务，缺乏在更广泛生成任务或强化学习中的验证。

---

## 33. Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions

**arXiv ID:** 2606.02859 | [PDF](https://arxiv.org/pdf/2606.02859v1)

**作者:** Zhenting Qi `[一作]` (Harvard), Yilun Du `[通讯]` (Harvard)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了基于经济机制的Agent经济体（Economy of Minds），让弱智能体通过竞价、交易和财富演化实现自组织的协同推理与学习；

**💡 创新点**

创新点在于将经济学的分散协调与激励机制引入多智能体系统，利用竞价实现去中心化决策、财富流转实现信用分配，并通过经济选择驱动进化；

**🔧 技术方法**

采用大型语言模型（LLM）作为基础模型，构建可通过提示（prompt）差异化的代理，结合拍卖式行动选择、财富转移规则以及经济衰减、淘汰和种群注入机制；

**📊 数据集**

在五大任务上评估：MATH（数学推理）、Finance‑Agent‑Bench（金融研究）、FrontierScience‑Research（科学研究）、Gemmini加速器设计和Cloudcast分布式系统优化；

**📈 对比分析**

对比了完整代理（如ReAct、GEA、OpenEvolve）与部分代理（Multi‑Agent Debate）等基线，结果显示在各领域均优于或匹敌完整代理，提升率从约10%到超过300%不等；

**⚠️ 局限性**

局限性包括仅在提示空间进行进化、使用冻结的LLM骨干限制了能力扩展、未处理多模态或具身环境，并且对完全通用代理的长期影响仍需进一步验证。

---

## 34. Small RL Controller, Large Language Model: RL-Guided Adaptive Sampling for Test-Time Scaling

**arXiv ID:** 2606.03102 | [PDF](https://arxiv.org/pdf/2606.03102v1)

**作者:** Runpeng Dai `[一作]` (University of North Carolina at Chapel Hill), Hongtu Zhu `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 29431 | [OpenAlex ID](https://openalex.org/A5077961759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于强化学习的轻量级控制器，用于在大型语言模型推理时自适应采样，动态平衡答案准确率、延迟与计算成本；

**💡 创新点**

创新点在于将自适应采样建模为有限时域马尔科夫决策过程，利用四层MLP RL训练得到的策略无需额外信号即可同时优化准确率、延迟与成本，并从拉格朗日视角解释其为约束优化问题；

**🔧 技术方法**

核心技术包括强化学习（PPO）、MDP建模、轻量化四层MLP控制器、基于统计特征的状态表示、step‑penalty + terminal reward 的奖励设计以及拉格朗日松弛理论；

**📊 数据集**

实验使用 AIME24、AIME25、HMMT 2025 三大数学推理基准，并在 Qwen‑3 系列（0.6B、1.7B、4B）与 GPT‑4.1‑nano 采样器上进行验证；

**📈 对比分析**

与 SC、ASC、ESC 等基线对比，RL‑guided 在保持相近或更高准确率的前提下，采样轮数降低 3–4 倍、总样本数下降 30%–35%，显著优于现有自适应采样方法；

**⚠️ 局限性**

局限性包括：状态仅依赖答案统计，缺乏语义或置信信息，奖励未直接映射实际时间/金钱成本，且对极端难度或非数学任务的泛化尚待验证。

---

## 35. ZOAF: Towards Efficient Zeroth-Order Optimization for Analog/RF Circuit Design

**arXiv ID:** 2606.02869 | [PDF](https://arxiv.org/pdf/2606.02869v1)

**作者:** Liyan Tan `[一作]` (University of California Santa Barbara), Zheng Zhang `[通讯]` (University of California Santa Barbara)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于零阶优化的Analog/RF电路尺寸设计框架ZOAF，兼顾探索与精细化；

**💡 创新点**

创新点在于三层设计：1）混合随机方向与坐标方向的零阶梯度估计，2）一次性多起点预采样+排名重启，3）滑动窗口自适应调度与早停，显著降低模拟调用次数；

**🔧 技术方法**

使用零阶梯度估计（ZO-RGE与ZO-CGE）、投影裁剪、滑动窗口监控、排名重启以及基准比较算法（GP-BO、CMA-ES、PSO、DE、TuRBO、AutoCkt、DNN-Opt等）；

**📊 数据集**

在三种Benchmarks上验证：1）15参数RF匹配网络；2）10参数三级运算放大器；3）22参数双级信号放大器；数据来自ADS/PySpice仿真；

**📈 对比分析**

与BO、EA、学习式方法对比，ZOAF在相同模拟调用预算下，获得最佳中位数FOM（最高达10^8增益、10^7GHz GBP、-66dB返回损耗），收敛速率提升1.3–3.8倍；

**⚠️ 局限性**

局限在单目标、连续箱约束下；对多目标、多分辨率或离散决策的适应尚待进一步研究；

---

## 36. The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment

**arXiv ID:** 2606.03043 | [PDF](https://arxiv.org/pdf/2606.03043v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 37. Diagnosis of Human Object Interaction Detectors for Real World Educational Applications

**arXiv ID:** 2606.02789 | [PDF](https://arxiv.org/pdf/2606.02789v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 38. Cross-Modal Contrastive Learning of ECG and Angiography Representations for Severe Stenosis Classification

**arXiv ID:** 2606.02605 | [PDF](https://arxiv.org/pdf/2606.02605v1)

**作者:** Nikola Cenikj `[一作]` (Technical University of Munich), Philip Müller `[通讯]` (Technical University of Munich)

**通讯引用:** 2496 | [OpenAlex ID](https://openalex.org/A5074684285)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

研究了一种跨模态对比预训练框架 StenCE，利用 ECG 与冠状动脉 X 光血管造影（Angio）配对数据，使 ECG 编码器学习从 ECG 中提取与血管狭窄相关的特征，最终实现仅用 ECG 进行严重狭窄（≥100%）分类。

**💡 创新点**

创新点在于首次将血管造影作为对比学习的参照物，构建 ECG 与 Angio 的对比损失，并在预训练阶段加入狭窄分级监督，从而提升 ECG 模型对狭窄的诊断能力；同时提出了 StenCE 这一专门针对狭窄的预训练框架。

**🔧 技术方法**

技术包括：Transformer‑based ECG 编码器（OTIS、EchoingECG），SegmentMIL 血管造影编码器，CLIP 对比损失，额外的 stenosis 监督损失，线性投影映射到共享嵌入空间，Fine‑tune 阶段使用加权二元交叉熵；以及在多任务设置下对视图聚合器的微调。

**📊 数据集**

数据集包括：医院内部 ECG‑Angio 数据集（3414 名患者，4067 对 ECG‑Angio，按 0%–100% 阶段标注），以及公开 EchoNext 数据集用于评估 ECG 编码器在心脏异常（左室射血分数、肺动脉压等）上的迁移性能。

**📈 对比分析**

与单模态预训练（OTIS、xECG、ECGFounder）及跨模态预训练（EchoingECG、PTACL）模型对比，StenCE 预训练在 100% 严重狭窄阈值下实现 0.822 的 AUC（比基线提升 4%），在 0%–≥90% 阈值下提升 11%（EchoingECG‑StenCE）。在轻度阈值下虽不如基线，但整体在全 fine‑tune 与线性探针设置下均表现更好，尤其在线性探针下提升可达 17%。

**⚠️ 局限性**

局限性包括：数据集中仅包含因症状而接受血管造影的患者，存在选择偏差；仅使用 ECG 信号，未结合临床风险因素；模型在非严重狭窄分类上性能仍不满足临床应用；跨模态预训练仍受限于配对数据规模与质量。

---

## 39. Decomposing how prompting steers behavior

**arXiv ID:** 2606.03093 | [PDF](https://arxiv.org/pdf/2606.03093v1)

**作者:** Fan L. Cheng `[一作]` (Columbia University), Nikolaus Kriegeskorte `[通讯]` (Columbia University)

**通讯引用:** 31883 | [OpenAlex ID](https://openalex.org/A5084467223)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出嵌套几何分解框架，将提示（prompting）导致的隐藏状态变化拆解为从平移到仿射再到非线性的一系列可解释的几何变换，并通过因果干预验证各层级变换对内部表示和下游行为的影响。

**💡 创新点**

创新点在于：①将提示效果建模为低复杂度几何变换，②通过层次化的正交 Procrustes、轴尺度、仿射等变换层级系统评估，③通过单层隐藏状态替换的因果干预直接验证这些变换是否能重现目标提示的几何和行为。

**🔧 技术方法**

使用 Procrustes 变换、岭回归、单隐藏层 MLP、RDM 相关、silhouette 评分、跨折交叉验证、以及基于隐藏状态的因果干预技术。

**📊 数据集**

六个公开模型（OPT‑2.7B、Meta‑Llama‑3‑8B‑Instruct、Qwen3‑8B、BLIP‑2、LLaVA‑OneVision‑7B、Qwen3‑VL‑8B），三类文本数据集（EmotionalStory、WritingStyle、Number）与三类图像数据集（EmoSet、StyleTransfer、COCO）。

**📈 对比分析**

评估方法包括累计解释方差（R²）、增量 R²、RDM 相关、silhouette 评分以及行为相关性与准确度；结果表明平移+仿射层级能显著恢复目标提示的几何与行为，非线性提升有限；不同模型、数据集与提示组合的层级贡献存在显著差异。

**⚠️ 局限性**

仅研究零样本提示，未覆盖少样本、软提示、长上下文、以及多层/多标记的干预；对提示重述与分布外转移的稳健性有限。

---

## 40. Multi-component Causal Tracing in Large Language Models

**arXiv ID:** 2606.03085 | [PDF](https://arxiv.org/pdf/2606.03085v1)

**作者:** Zirui Yan `[一作]` (Rensselaer Polytechnic Institute), Ali Tajer `[通讯]` (Rensselaer Polytechnic Institute)

**通讯引用:** 2276 | [OpenAlex ID](https://openalex.org/A5054103414)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了统一的多组件因果追踪框架，并设计了高效的梯度下降算法来在大语言模型中识别对目标指标最具影响力的注意力头和MLP神经元集合。

**💡 创新点**

创新点在于将离散的组合搜索问题通过软干预与度量变换转化为连续可微形式，利用调度正则化实现稀疏二进制掩码；同时首次系统比较了多组件干预的非线性效应。

**🔧 技术方法**

核心技术包括：混合前向传播、软干预与连续松弛、反向梯度优化、正则化调度、与度量变换的结合。

**📊 数据集**

实验使用了GPT2系列（small、medium、large、xl）及DistilGPT2、Qwen3‑1.7B、Llama‑3.2‑1B等模型；数据集包括WinoGender、WinoBias、Professions、CounterFact、VBD。

**📈 对比分析**

与基线的top‑k、greedy、random以及DCM进行对比。结果显示，所提方法在平均指标上与greedy相近或更优，同时在执行时间上比top‑k快1.5‑2倍、比greedy快约200‑300倍；在多组件选择上，掩码与greedy的Jaccard相似度高达0.64。

**⚠️ 局限性**

局限性包括：需提前设定单一目标指标和稀疏度阈值；对梯度下降的超参数敏感；实验主要聚焦英语数据和GPT/LLama/Qwen模型，缺乏对其他语言、模型族和专业任务的验证。

---

## 41. From Explanation to Diagnosis: Next Generation Interactive Video Coach with Misstep Awareness

**arXiv ID:** 2606.02970 | [PDF](https://arxiv.org/pdf/2606.02970v1)

**作者:** Xiao Jin `[一作]` (Georgia Institute of Technology), Ashok K. Goel `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 7705 | [OpenAlex ID](https://openalex.org/A5007028896)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了教学模型（Pedagogical Model, PM），将Ivy从单纯的解释生成提升为误差诊断与定制反馈；

**💡 创新点**

通过引入PM层，实现四类误差（执行失误、概念缺失、因果误解、目标失配）分类、错误定位、根因信念识别以及基于误差类型的支架生成，填补了知识检索式ITS的诊断空白；

**🔧 技术方法**

采用Task‑Method‑Knowledge (TMK) 结构、GPT‑5 nano + LangChain 生成流水线、PM JSON 结构化记录、FAISS向量检索与OpenAI嵌入、LLM零样本分类器等技术；

**📊 数据集**

使用Georgia Tech CS 7637 AI课程2026春季测验题目（9道题、22个错误答案），并由教师Q&A键构建PM记录；

**📈 对比分析**

与原TMK‑only流水线对比，评估48个答案的五维反馈质量（准确性、针对性、可行性、可迁移性、支架适当性），PM‑augmented pipeline在四维上显著提升，p值<0.001；

**⚠️ 局限性**

实验规模有限（仅22个错误、三类技能），仅评估反馈质量未测学习成效，且仅覆盖多选题，缺乏开放式问题诊断和长期学习者模型。

---

## 42. Samudra 2: Scaling Ocean Emulators across Resolutions

**arXiv ID:** 2606.02610 | [PDF](https://arxiv.org/pdf/2606.02610v1)

**作者:** Yuan Yuan `[一作]` (New York University), Laure Zanna `[通讯]` (New York University)

**通讯引用:** 6077 | [OpenAlex ID](https://openalex.org/A5032073724)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Samudra 2 海洋模拟器，能够在不同分辨率（1°、0.5°、0.25°）下进行多年自回归海洋状态预测。

**💡 创新点**

创新点包括更宽的 ConvNeXt U‑Net 结构与动态误差加权损失函数，有效抑制长期方差坍塌与跨变量印记，显著提升深层海洋的预测精度。

**🔧 技术方法**

采用宽化的 U‑Net 结合 ConvNeXt 结构，动态加权 MSE 损失，使用多步自回归训练与 EMA 权重更新。

**📊 数据集**

使用 GFDL OM4 海洋模型输出（19 层深度，四个物理量加海表高度）在 1°、0.5° 与 0.25° Gaussian 网格上进行训练和评估。

**📈 对比分析**

与原 Samudra 以及 OM4 真值比较，评估指标包括长期平均温度 R²、Niño 3.4 指数相关性、温度与动能的方差相关性以及空间谱匹配。Samudra 2 在上层水体 R² 从 0.56 提升至 0.87，深层温度误差降低约 7 倍，并在 0.25° 分辨率下恢复了涡动与锋面细节。

**⚠️ 局限性**

仍然无法实现深层（>700 m）温度和盐度的正向预测，R² 仍为负；模型对高频噪声与跨变量印记的抑制仍有限，需进一步引入谱损失或物理约束。

---

## 43. Characterization and Effects of CS2 Learning with GenAI, Visualization, and Human Support

**arXiv ID:** 2606.02933 | [PDF](https://arxiv.org/pdf/2606.02933v1)

**作者:** Quinton Yong `[一作]` (University of Victoria), Miguel Nacenta `[通讯]` (University of Victoria)

**通讯引用:** 2619 | [OpenAlex ID](https://openalex.org/A5081381214)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究在大学二年级算法课程中，比较学生使用生成式AI（Copilot）、算法可视化工具（VisuAlgo）以及真人辅导三种学习支持的交互方式与学习成效。

**💡 创新点**

创新点在于首次将生成式AI与传统可视化与人类辅导在更高阶 CS 课程中进行细粒度交互与学习效果的对比，并结合自我效能与元认知指标进行多维评估。

**🔧 技术方法**

使用了混合方法实验，结合眼动追踪、屏幕录制、问卷调查、测验，并采用 Copilot GPT‑4 语言模型、VisuAlgo 可视化工具以及现场真人辅导。

**📊 数据集**

数据来自12名大二算法课程学生，完成三次 90 分钟实验，收集交互计数、视线轨迹、测验成绩和自我效能量表。

**📈 对比分析**

通过 Bayesian MCMC 对比三种支持下的测验得分与自我效能提升；结果显示真人辅导得分最高，生成式AI自我效能提升最大但学习成绩最低，算法可视化介于两者之间。

**⚠️ 局限性**

局限性包括样本量小、实验环境为单独实验室、未设无支持基准、任务顺序未平衡、仅使用一款 LLM，可能影响结果的普适性。

---

## 44. Fast Transformer Inference on ARM-Based HMPSoCs

**arXiv ID:** 2606.02836 | [PDF](https://arxiv.org/pdf/2606.02836v1)

**作者:** Hang Xu `[一作]` (University of Amsterdam), Anuj Pathania `[通讯]` (University of Amsterdam)

**通讯引用:** 1089 | [OpenAlex ID](https://openalex.org/A5067055700)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现并扩展ARM-CL框架，新增Transformer内核（Embedding、MHA、SDPA、FF、LayerNorm等），并实现CPU-GPU层级切换的低延迟推理

**💡 创新点**

①首次在ARM HMPSoC上提供原生Transformer推理支持；②对Transformer各层在CPU/ GPU上的性能异质性做微基准化分析；③提出CPU-GPU协同层切换方案，显著降低推理延迟；④在ARM-CL上实现低开销共享内存交互

**🔧 技术方法**

ARM Compute Library扩展（NEON、OpenCL、矩阵分块MMUL）、多线程调度、共享内存机制、基于微基准的层级选择策略

**📊 数据集**

使用标准预训练Transformer模型（BERT-base、DistilBERT、MobileBERT、SqueezeBERT、GPT-2），在Khadas VIM 3硬件上进行synthetic长度32/128等序列的推理测评；无公开数据集，采用随机token序列

**📈 对比分析**

与TVM、ExecuTorch等主流框架对比；单核CPU、单核GPU、以及CPU-GPU切换三种配置；结果显示ARM-CL扩展后单处理器推理加速约2.3×，CPU-GPU协同可进一步降低15.7%延迟（最高可达3×速率提升）

**⚠️ 局限性**

仅在Amlogic A331D HMPSoC上验证；GPU缓存受限导致大规模MMUL性能衰退；未考虑多模型/多批量推理；共享内存实现依赖ARM HMPSoC的统一地址空间，迁移性有限

---

## 45. A New Framework for Cybersecurity Refusals in AI Agents

**arXiv ID:** 2606.02644 | [PDF](https://arxiv.org/pdf/2606.02644v1)

**作者:** Eliot Krzysztof Jones `[一作]` (Gray Swan AI), J Zico Kolter `[通讯]` (Gray Swan AI)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大型语言模型在网络安全攻击场景中的拒绝机制，提出并实现了Cybersecurity Refusal Framework（CRF）基准。

**💡 创新点**

创新点在于引入环境感知拒绝、红/黄/绿三色拒绝分类以及面向代理的综合评估框架，首次系统评估代理在真实与模拟环境下的拒绝行为。

**🔧 技术方法**

采用了agentic scaffold、Codex工具链与自动化评判器，结合LLM模型进行漏洞利用与拒绝决策的端到端测试。

**📊 数据集**

使用了自构造的29个Web漏洞挑战（XSS、SQLi、SSRF等），按Easy/Medium/Hard划分，并在四个关键基础设施红区域（政府、医疗、电网、交通）进行实验。

**📈 对比分析**

通过pass@1、refusal@1、utility等指标对八款前沿模型进行对比，发现大多数模型拒绝率极低，只有GPT‑5.2和GPT‑5.1 Codex表现出显著拒绝；最高utility的是Gemini 3.1 Pro。

**⚠️ 局限性**

局限性在于仅聚焦Web漏洞与红区域，未覆盖其他网络层面；拒绝策略缺乏统一标准，易被“scope”或本地代理等手段绕过，且难以实现可靠的授权验证。

---

## 46. TGV-KV: Text-Grounded KV Eviction for Vision-Language Models

**arXiv ID:** 2606.03075 | [PDF](https://arxiv.org/pdf/2606.03075v1)

**作者:** Jizhihui Liu `[一作]` (Harbin Institute Of Technology Shenzhen), Yaowei Wang `[通讯]` (Harbin Institute Of Technology Shenzhen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究 Vision‑Language 模型（VLM）中 Key‑Value (KV) 缓存的内存消耗瓶颈，提出一种基于文本引导的 KV 消除框架 TGV‑KV，能够在保持文本信息完整的前提下，高效压缩视觉 KV。

**💡 创新点**

创新点：① 将文本与视觉交互注意力作为层级预算分配依据；② 用文本注意力权重对视觉 KV 重要性进行加权排序；③ 在预算允许时优先保留文本 KV，形成文本优先保留策略；③ 通过系统的注意力模式分析，提出三大子模块（TVB、TWR、TPR）以解决 VLM 的 modality gap。

**🔧 技术方法**

主要技术：自回归解码器的 KV 缓存机制；基于互信息的层级预算分配；文本‑视觉注意力权重提取；文本加权的视觉 KV 重要性评分；文本优先保留的淘汰策略；FlashAttention 等高效实现。

**📊 数据集**

使用的数据集与任务：
- 图像 VQA：ChartQA、DocVQA、VizWiz、TextVQA；
- 文本生成：TextCaps、COCO‑Caption‑2017；
- 视频推理：Video‑TT；
- 多模态模型：LLaVA、LLaVA‑NeXT、LLaVA‑OneVision‑Qwen2‑0.5B、Qwen3‑VL‑4B/8B。

**📈 对比分析**

与现有 KV 消除方法（StreamingLLM、SnapKV、H_2O、ElasticCache、PrefixKV）以及 token‑pruning 方法（DivPrune、VisionZip、VisPruner、CDPruner）进行对比。结果显示：
- 在 5% KV 保留率下，TGV‑KV 在 VizWiz、TextVQA 等任务保持 99.2% 的准确率；
- 在 LLaVA‑NeXT 上 5% 保留率下准确率仍达 97.4%；
- 对 Qwen3‑VL‑4B/8B，5% 保留率下准确率分别保持 93.3% 与 92.1%；
- 内存使用率降低 95% 以上；
- 通过极端压缩（5%）提升解码吞吐率 52.6%。

**⚠️ 局限性**

局限性：
- 只针对 KV 缓存层面进行裁剪，未解决 token‑级别的冗余；
- 在极低预算时仍会损失部分视觉信息，尤其对视觉主导任务；
- 需要对不同 VLM 的内部注意力模式进行微调，适用性可能受限；
- 未在极长上下文或多视频帧的实时推理场景下深入评估；
- 对模型微调与迁移学习的影响尚未探讨。

---

## 47. ModuLoop : Low-Level Code Generation using Modular Synthesizer and Closed-Loop Debugger for Robotic Control

**arXiv ID:** 2606.03047 | [PDF](https://arxiv.org/pdf/2606.03047v1)

**作者:** Gina Yoon `[一作]` (Sookmyung Women's University), Joo Yong Sim `[通讯]` (Sookmyung Women's University)

**通讯引用:** 3450 | [OpenAlex ID](https://openalex.org/A5008309794)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用预训练的LLM自动生成、执行低级机器人控制代码，实现相机与机械臂的手眼标定以及基于自然语言指令的抓取搬运任务。

**💡 创新点**

提出闭环模块化代码合成与调试框架ModuLoop，使LLM在无任务特定微调的情况下完成任务分解、代码生成、实时反馈诊断和迭代改进。

**🔧 技术方法**

采用GPT‑4o/4.1‑mini等大型语言模型、Isaac Sim仿真、逆运动学过滤、OpenCV ArUco、Segment Anything Model (SAM)和RTDE接口等技术。

**📊 数据集**

使用UR3机械臂与Intel RealSense D435i深度相机搭建的实机标定数据以及五个包含不同复杂度语言指令的RGB‑D抓取搬运任务，未依赖公开大规模数据集。

**📈 对比分析**

与单步代码生成、Code‑as‑Policies、ProgPrompt等方法对比，ModuLoop在标定误差<2 cm的成功率达86.7%，在五个抓取任务中的成功率分别为88‑96%，展现出更高的代码生成成功率、准确性和迭代效率。

**⚠️ 局限性**

当前仅基于最小API和基础感知，难以处理复杂的接触式操作或需丰富规划与价值图的任务；完整控制代码生成导致一定延迟，限制实时响应。

---

## 48. Principled Reflection Separation via Nonlinear Superposition and Feature Interaction

**arXiv ID:** 2606.02831 | [PDF](https://arxiv.org/pdf/2606.02831v1)

**作者:** Qiming Hu `[一作]` (Tianjin University), Xiaojie Guo `[通讯]` (Tianjin University)

**通讯引用:** 14739 | [OpenAlex ID](https://openalex.org/A5090356888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对单张图像的反射层分离问题，提出了可学习的非线性叠加模型和双流交互框架，以更精确模拟真实sRGB域下的层叠关系。

**💡 创新点**

创新点在于（1）将线性叠加假设替换为可学习的非线性高阶模型，捕获ISP引入的非线性耦合；（2）统一设计三类可插拔的双流交互模块（激活、门控、注意力），实现双向信息交换；（3）引入Learnable Offset‑Residual Superposition（LORS）机制，显式建模非线性残差与零阶偏置。

**🔧 技术方法**

技术方法主要包括：可学习的非线性叠加层析、双流交互解码器（DSI/DSIF块）、双流注意力（PAIR块）、多尺度特征编码、LORS头、感知与排除损失。

**📊 数据集**

使用的主要数据集为SIR^2真实反射对、Real20、Nature、SIR^2子集以及合成的PASCAL VOC/张等数据，并通过伪反射完成真实图像的三元组训练。

**📈 对比分析**

在多项公开基准（Real20、Nature、SIR^2子集）上，DIRS-PAIR在PSNR/SSIM上均超过现有最佳方法，尤其在强反射和非线性叠加场景下取得显著提升，同时在手机厂商自带算法上也表现更佳。

**⚠️ 局限性**

局限性包括（1）Transformer版模型计算量较大，推理延迟相对较高；（2）对极端高强度或极少纹理场景的鲁棒性尚待进一步提升；（3）需要大量带有真实三元组的训练数据，数据采集成本仍然是瓶颈。

---

## 49. Powering An Ecosystem Of Pedagogical AI Agents: A Validation Strategy For A Unified Data Architecture

**arXiv ID:** 2606.02950 | [PDF](https://arxiv.org/pdf/2606.02950v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965`

---

## 50. Towards Compact Autonomous Driving Perception with Balanced Learning and Multi-sensor Fusion

**arXiv ID:** 2606.02979 | [PDF](https://arxiv.org/pdf/2606.02979v1)

**作者:** Oskar Natan `[一作]` (Toyohashi University of Technology), Jun Miura `[通讯]` (Toyohashi University of Technology)

**通讯引用:** 4962 | [OpenAlex ID](https://openalex.org/A5071725508)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种紧凑的多任务深度学习模型，能够在单次前向传播中同时完成多视角语义分割、深度估计、LiDAR分割与鸟瞰图投影，并通过多传感器融合提升感知能力。

**💡 创新点**

创新点包括：①整合RGB、DVS、LiDAR三种多模态输入并采用中间融合策略；②将LiDAR点云按高度分层预处理为15层张量；③改进GradNorm算法（MGN）实现自适应损失权重以平衡多任务学习。

**🔧 技术方法**

使用了U‑Net风格的编码解码网络、3D张量LiDAR预处理、DVS事件映射、Huber+BCEDice损失、SGD优化及自适应GradNorm（MGN）算法。

**📊 数据集**

实验数据集为CARLA模拟的三套数据集（Town01–Town05）以及真实世界的nuScenes‑lidarseg，涵盖多种天气和光照条件。

**📈 对比分析**

与四个最近模型（GradNorm、PolarNet、Chen等）及单任务模型进行对比，采用总指标TM、方差MV、参数量、GPU内存和FPS评估。最佳变体15L+MGN在TM和MV上优于对比模型，参数量仅为对比模型的约2%，但FPS更高，说明更高效。

**⚠️ 局限性**

局限性包括：在LiDAR分割和鸟瞰投影任务上仍略逊于PolarNet；雨夜时DVS噪声影响深度估计；模型架构仍是手工设计，未实现自动化网络分支；未覆盖规划与控制等后续任务。

---

## 51. Plan2Map: A Multimodal Benchmark for Document-Grounded Geospatial Boundary Reconstruction from Planning Records

**arXiv ID:** 2606.02747 | [PDF](https://arxiv.org/pdf/2606.02747v1)

**作者:** Fabian Degen `[一作]` (University of Oxford), Jialin Yu `[通讯]` (University of Oxford)

**通讯引用:** 17262 | [OpenAlex ID](https://openalex.org/A5100599376)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Plan2Map 基准，要求系统仅根据英国规划文件中的文本、地图和标签推断地理边界并输出 GeoJSON。

**💡 创新点**

创新点在于将任务拆解为多步骤代理系统（Reader、Locate 子代理、Worker 与可选 Critic），并将文档理解、地理定位、地图配准、边界分割与投影等模块串联，实现从非结构化文件到可验证空间几何的完整链路。

**🔧 技术方法**

使用 Gemini‑3 Flash/Pro 作为语言模型，OS Open Names/Zoomstack 等地理工具，SAM‑3 与 LoRA 微调实现边界分割，以及 MINIMA‑LoFTR + RANSAC 进行地图配准和投影。

**📊 数据集**

基准数据集 Plan2Map 包含 208 份 UK Article 4 Direction 规划文件，配套手工验证的 GeoJSON 边界以及地图可视化，覆盖 29 个地方规划局，年代跨度 1958‑2025。

**📈 对比分析**

与直接 VLM‑to‑GeoJSON 的 40 条案例基准相比，GeoPlanAgent 在全 208 条数据上平均 IoU 0.736、半数 IoU 0.904，IoU ≥0.8 的比例 67.8%，中心点误差中位数 4.6 m，成本仅 $0.043/文档，性能远超基线（例如 Gemini‑3.1‑Pro 的平均 IoU 0.108）。

**⚠️ 局限性**

局限在于依赖英国丰富的公开地理基础设施；对纯文本描述无可视边界的文件无法通过 SAM‑3 进行分割；低质量扫描或复杂多部分边界仍是主要瓶颈。

---

## 52. Translating Classical Poetry into Modern Prose

**arXiv ID:** 2606.02806 | [PDF](https://arxiv.org/pdf/2606.02806v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 53. Applying Two-Grid Preconditioner for Subsurface Flow Simulation using Attention-enhanced Hybrid Network to Accelerate Multiscale Discretization in High-contrast Media

**arXiv ID:** 2606.02582 | [PDF](https://arxiv.org/pdf/2606.02582v1)

**作者:** Peiqi Li `[一作]` (Xi'an Jiaotong-Liverpool University), Shubin Fu `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种混合框架，用学习快速预测混合通用多尺度有限元（mixed GMsFEM）的多尺度基函数，随后使用两格预处理器求解压强场，显著加速低频基函数构造。

**💡 创新点**

创新点在于将频域注意力增强的 Fourier 神经算子与空间域注意力 U‑Net 结合，先在频域提取全局特征再在空间域细化细节，且仅对多尺度基函数进行学习，保持了数值求解的可解释性与稳定性。

**🔧 技术方法**

技术包括：基于频域注意力的 Fourier Neural Operator、改进的 Attention U‑Net、两格预处理器、梯度一致性损失以及基于 Karhunen‑Loève 的随机孔隙率场生成。

**📊 数据集**

使用 4000 组基于 KLE 生成的高对比裂缝渗透率场，训练集 3500 组，测试集 500 组；多尺度基函数与压强场均从 GMsFEM 计算得到。

**📈 对比分析**

与 FNO、U‑Net、PINN、传统 GMsFEM（基函数数 6）进行对比，评估指标为 MSE、R²、相对 L2 误差，结果显示所提方法在基函数预测精度（MSE≈10⁻³，R²>0.9）和压强场重构误差（相对 L2 误差最低）方面均优于对照组；同时保持了可接受的计算复杂度（单前向 7.66 GFLOPs）。

**⚠️ 局限性**

局限性包括：仅在二维问题上验证，三维推广需进一步研究；基函数预测误差仍受训练样本覆盖度限制；对高频细节的捕捉仍可能不足，尤其在更复杂多尺度结构中。

---

## 54. Fixed-Time Dynamic Landing of Quadrotors using Adaptive Unscented Kalman Filtering and Nonlinear Model Predictive Control

**arXiv ID:** 2606.02658 | [PDF](https://arxiv.org/pdf/2606.02658v1)

**作者:** Mohammadreza Izadi `[一作]` (Toronto Metropolitan University), Reza Faieghi `[通讯]` (Toronto Metropolitan University)

**通讯引用:** 586 | [OpenAlex ID](https://openalex.org/A5046495866)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一套用于多旋翼无人机在移动平台上动态降落的完整框架，结合了自适应无迹卡尔曼滤波、固定时长最小jerk轨迹规划和非线性模型预测控制，实现了精准同步与可重复的降落；

**💡 创新点**

创新点包括：① 在终止降落阶段使用固定时长最小jerk规划保证预定着陆时间；② 通过自适应无迹卡尔曼滤波在线更新噪声统计，提升在测量质量变化下的估计鲁棒性；③ 对最小jerk参考进行可行性分析，给出推力和扭矩的上界，确保NMPC可满足执行器约束；

**🔧 技术方法**

核心技术包括：非线性模型预测控制（NMPC）、自适应无迹卡尔曼滤波（AUKF）、固定时长最小jerk（MJT）轨迹规划、以及基于acados/HPIPM的实时QP求解；

**📊 数据集**

实验使用了室内OptiTrack运动捕捉系统提供的实时定位数据，硬件平台为配备Pixhawk和Intel i5的Holybro X500无人机与TurtleBot3 Waffle Pi移动平台；

**📈 对比分析**

与标准EKF/UKF比较，AUKF的速度预测RMSE平均下降约30%（对UKF）和60%（对EKF），降落误差平均仅0.0787 m；NMPC求解时间约4.9 ms，低于100 Hz采样周期，表明方案在实时性与精度上均优于传统方法；

**⚠️ 局限性**

局限性包括：实验仅在室内无风环境进行，未验证在户外风扰动下的鲁棒性；轨迹规划未考虑能耗约束；平台运动模型假设较为简单，复杂运动时需进一步改进。

---

## 55. On Improving Robustness of Deepfake Image Detectors

**arXiv ID:** 2606.02797 | [PDF](https://arxiv.org/pdf/2606.02797v1)

**作者:** Abu Taib Mohammed Shahjahan `[一作]` (Concordia University), Amr Youssef `[通讯]` (Concordia University)

**通讯引用:** 6270 | [OpenAlex ID](https://openalex.org/A5085765243)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种统一框架，通过三种设计原则提升深度伪造图像检测器在无对抗训练情况下的鲁棒性；

**💡 创新点**

创新点在于将四阶频域统计（kurtosis）与内容无关的噪声残差特征以及图像局部随机打乱相结合，形成对抗样本难以规避的高阶不变特征；

**🔧 技术方法**

采用离散余弦变换（DCT）四阶矩池化、MM‑BSN噪声残差提取、CLIP ViT-L/14特征、patch‑shuffle + 随机旋转以及自注意力融合；

**📊 数据集**

使用GenImage（约270万张真实/合成图像）作为训练集，并在Abdullah等人生成的对抗深度伪造图像集（StyleCLIP等）做评估；

**📈 对比分析**

对比七个最新SOTA检测器，实验显示在所有对抗数据上平均提升约16–20％准确率，最优模型D^3从81.9%提升至97.15%，对抗召回率下降幅度最高可降至9.8%；

**⚠️ 局限性**

局限性包括对高阶统计的依赖可能导致对非生成噪声的误判、模型训练需要大量GPU资源、对极端对抗策略的适应性尚未充分验证。

---

## 56. Pruning Deep Neural Networks via the Marchenko--Pastur Distribution

**arXiv ID:** 2606.02608 | [PDF](https://arxiv.org/pdf/2606.02608v1)

**作者:** Leonid Berlyand `[一作]` (Pennsylvania State University), Yitzchak Shmalo `[通讯]` (Pennsylvania State University)

**通讯引用:** 33 | [OpenAlex ID](https://openalex.org/A5041315596)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对深度神经网络（尤其是Vision Transformer与CNN）在极低微调预算下进行基于Marchenko–Pastur随机矩阵理论的稀疏化，提出“混合Magnitude–SER”和“CAST”两种稀疏化策略。

**💡 创新点**

创新点在于：①利用MP谱诊断作为层级稀疏化预算信号，避免验证/测试集；②提出可恢复的稀疏化（SER/CAST–restore）机制；③提供确定性数据路径证书，证明在一定预算下稀疏化不会显著增加损失；④在极短微调周期（1~3个epoch）内保持与密集模型相近的精度，并给出真实部署加速结果。

**🔧 技术方法**

使用的技术包括：随机矩阵理论（Marchenko–Pastur分布、MP边缘估计）、弹性网络与L2正则化的损失约束、梯度方向与Lipschitz常数估计、稀疏化后恢复与分组稀疏化（k:n）以及Token合并（ToMe）技术。

**📊 数据集**

实验数据集为ImageNet‑1k，使用多个预训练模型：ViT‑B/16、ViT‑L/16、ConvNeXt‑V2‑Base、ResNet50/152、DeiT、Swin 等。

**📈 对比分析**

比较方法：对每个模型在相同基准检查点、相同稀疏率和相同微调预算下与传统Magnitude、直接RMT、CAST、SOTA稀疏化（如SnowS、MaskLLM、SparseFormer）进行对比。结果显示：在50%稀疏率下ViT‑B/16的Top‑1仅下降1.7pp，ViT‑L/16 8:16保持-0.51pp；CNN层级在约50%稀疏率下仅损失≤0.3pp；且在A40/A100硬件上取得1.3–2.7×的实际加速。

**⚠️ 局限性**

局限性包括：①对不同硬件的可部署性（2:4仅在NVIDIA TensorCore可验证，其余需仿真或非原生加速）；②仅在单一随机种子下给出CAST结果；③理论中的Lipschitz估计与实际网络的非线性匹配不足；④高阶稀疏模式（如8:16、12:16）仍缺乏官方加速实现；⑤实验仅覆盖ImageNet‑1k，未验证跨任务或更大模型。

---

## 57. Hybrid Dynamics Modeling for a Flexible 2-DoF Robotic Arm

**arXiv ID:** 2606.02969 | [PDF](https://arxiv.org/pdf/2606.02969v1)

**作者:** Maciek Popik `[一作]` (University of Calgary), Mahdis Bisheban `[通讯]` (University of Calgary)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5087695015)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

针对轻量化两自由度柔性机械臂，构建并比较了三种扭矩模型：纯数据黑盒、纯物理白盒以及结合残差学习的灰盒模型。

**💡 创新点**

创新点在于将刚体动力学与高斯混合残差回归相结合，形成半参数灰盒框架；提出在线残差一致性转换以保持模型更新的一致性；并通过正则化最小二乘实现参数估计。

**🔧 技术方法**

使用技术包括刚体动力学（RBD）、Ridge 回归、最小二乘参数估计、Gaussian Mixture Regression (GMR)、残差正则化、在线自适应学习和一致性变换。

**📊 数据集**

采用公开的 MERIt/TUDOR 数据集 TUD01（无载荷的3-DoF臂，提取前两关为2-DoF）。

**📈 对比分析**

通过 nMSE、RMSE 以及误差随时间的比较，灰盒+GMR 模型在两关节上均表现最优；纯物理模型误差最大，黑盒模型与灰盒相近；GMR 显著降低残差。

**⚠️ 局限性**

局限性包括：RBD 未能充分捕捉柔性耦合导致参数估计失真；数据集激励多样性不足影响参数收敛；GMM 训练仍需较高计算成本；未考虑温度等环境变化对柔性动态的影响。

---

## 58. Human-in-the-Loop Contextual Bandits for Short-Term Rental Dynamic Pricing: Structural Equivalence of Historical Warm-Up and Approval-Gated Live Learning

**arXiv ID:** 2606.02595 | [PDF](https://arxiv.org/pdf/2606.02595v1)

**作者:** Oleg Miroshnichenko `[一作]` `[通讯]`, Oleg Miroshnichenko

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了Human-in-the-Loop Gated Bandit（HITL‑GB）框架，利用历史审批数据进行warm‑up，解决短租动态定价中的人机协作与冷启动问题。

**💡 创新点**

证明在审批门槛下，历史数据与在同一策略下的在线数据结构等价，无需重要性抽样；并通过α‑融合岭回归同时完成日信号参数校准和Bandit后验初始化，实现冷启动压缩。

**🔧 技术方法**

使用上下文Bandit（Hierarchical Factored Thompson Sampling）、人机交互门控、结构等价定理、α‑融合岭回归以及稀疏反馈下的OPE理论。

**📊 数据集**

使用真实短租平台生产数据（城市两房物业，4年，1461晚）以及KeyData公开市场占有率数据，亦提供基于KeyData的合成数据用于复现。

**📈 对比分析**

与冷启动、标准OPE（带IS校正）以及无混合warm‑up对比；在真实与合成实验中，HITL warm‑up在前30晚即可实现约11‑12% regret降低，累计收益提升约5倍的冷启动压缩。

**⚠️ 局限性**

仅在单一物业/单一市场进行验证，假设人类批准函数平稳；合成奖励模型可能存在循环偏差，且缺乏多物业A/B测试。

---

## 59. InquiryBits: Sharing AI Conversation Traces to Support Collaboration Within Trust Boundaries

**arXiv ID:** 2606.02763 | [PDF](https://arxiv.org/pdf/2606.02763v1)

**作者:** Caitlin Morris `[一作]`, Pattie Maes `[通讯]` (MIT Media Lab)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究如何通过最小化的 AI 对话摘要在受限信任边界内共享信息，以提升团队协作和减少重复工作。

**💡 创新点**

创新点在于：①将 AI 仅分析层与人类可见共享层分离的分层可见性架构；②强调信任边界而非信息粒度对共享舒适度的决定性影响；③引入可配置的“信任圈”匹配机制，允许用户在不同社交范围内选择共享对象。

**🔧 技术方法**

技术实现：①本地 NLP 处理（主题标签、嵌入向量）检测对话重叠；②使用 LLM（Claude Sonnet 4）生成简短摘要；③三层架构（主题签名、共享提示、AI‑上下文桥接）与可视化交互原型；④通过 UI 让用户在匹配后选择是否共享及范围。

**📊 数据集**

数据集：80 名从 Prolific 招募的专业人士提供的真实工作相关 AI 对话（ChatGPT 记录），在原型中本地处理并不公开；未使用公开大型语料库或公开对话数据集。

**📈 对比分析**

方法评估：在 80 名参与者中进行双条件对照实验（仅主题标签 vs 主题+摘要），并在不同信任范围内评估舒适度。结果显示：①信息粒度差异对分享决策影响不显著；②信任边界从“紧密团队”到“公司层级”导致舒适度显著下降；③原型分享率高达 92.5%。与现有基线（如公开共享工具或无过滤共享）无直接对比，主要聚焦于用户主观舒适度。

**⚠️ 局限性**

局限性：①实验在原型与模拟匹配下进行，缺乏真实工作环境和长期部署观察；②自我报告数据可能存在社会期望偏差；③样本偏向重度 AI 用户，可能高估整体专业群体的接受度；④未评估不同 LLM 或摘要算法对结果的影响；⑤缺乏对系统实际使用成本和集成难度的量化分析。

---

## 60. EURO-5K: When Does Domain Pretraining Matter? Benchmarking Transformers for EU Reporting Obligation Extraction

**arXiv ID:** 2606.02971 | [PDF](https://arxiv.org/pdf/2606.02971v1)

**作者:** Marios Koniaris `[一作]` (National Technical University of Athens), Panayiotis Tsanakas `[通讯]` (National Technical University of Athens)

**通讯引用:** 2662 | [OpenAlex ID](https://openalex.org/A5005032164)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对欧盟立法文本进行报告义务的句子级标注，构建EURO-5K数据集，并在此基础上训练与评估多种Transformer模型。

**💡 创新点**

首次公开专门针对报告义务的欧盟立法标注数据集；系统比较判别式(token‑classification)与生成式(span‑extraction)两大范式，并在全微调与参数高效微调之间评估领域预训练的实质影响。

**🔧 技术方法**

使用BERT/Legal‑BERT的BIO标签分类、LLM（Llama‑3.1、Mistral‑7B、Saul‑7B）的生成式span提取，结合全微调、LoRA、QLoRA等参数高效技术，基于规则与少量提示的基线进行对照。

**📊 数据集**

EURO‑5K（5253句子，1751正例、3502负例）为主要实验数据集；此外在两份外部监管语料上进行跨域验证，检验模型的泛化与专门性。

**📈 对比分析**

通过精确匹配的Precision/Recall/F1进行模型对比；实验结果表明全微调的Legal‑BERT与QLoRA Llama在EURO‑5K上均达F1≈0.89，判别式与生成式范式性能相当；在样本量极小的场景下，参数高效微调方法表现更为优越。

**⚠️ 局限性**

领域预训练的提升有限（统计不显著）、LLM实验仅单核、跨域评估样本有限、解释性方法（LIME/注意力）无法直接比较、仅覆盖英文欧盟立法，难以推广到其他语言与法律体系。

---

## 61. Speaker Mining -- FAIR Data on Public Broadcasts for Question Answering

**arXiv ID:** 2606.02905 | [PDF](https://arxiv.org/pdf/2606.02905v1)

**作者:** Tim Wittenborg `[一作]` (L3S Research Center Leibniz University Hannover), Sören Auer `[通讯]` (TIB Leibniz Information Centre for Science and Technology)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并评估了一个可扩展的FAIR数据管道，对15个公共广播节目进行嘉宾数据抽取、对齐、去重，生成可查询的知识图谱。

**💡 创新点**

提出事件驱动的多源候选生成与对齐、结合自动与手工调和的实体去重流程，并公开发布可SPARQL查询的知识图谱，实现跨节目问答。

**🔧 技术方法**

使用Python+Jupyter、PDF提取、OpenRefine、Wikibase/Wikidata、事件日志、规则式对齐与语义检索、手工纠错等技术。

**📊 数据集**

利用ZDF Archive PDF、fernsehserien.de、Wikidata、以及自建的WikiBase实例speakermining.wikibase.cloud等数据集。

**📈 对比分析**

与Spiegel、LanzMining、stk等先前工作对比，自动完成17,729条匹配，手工补充5,958条，整体62.3%实体成功链接，手工时间由8h降至64h，覆盖率提升至8,287名嘉宾。

**⚠️ 局限性**

手工去重仍占大比例；数据来源不完整导致缺失；Wikidata覆盖率低影响分析偏倚；GDPR与版权限制限制更深层属性采集；系统可扩展性虽提高但仍需前期投入。

---

## 62. Terminal Time and Angle-Constrained Nonlinear Intercept Guidance

**arXiv ID:** 2606.02872 | [PDF](https://arxiv.org/pdf/2606.02872v1)

**作者:** Shivam Bajpai `[一作]` (University of Cincinnati), Abhinav Sinha `[通讯]` (University of Cincinnati)

**通讯引用:** 949 | [OpenAlex ID](https://openalex.org/A5022385451)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了一种基于层次滑模的自适应导引法，利用拦截机唯一的侧向加速度同时控制撞击时间和撞击角，实现了时角约束的协同拦截。

**💡 创新点**

创新点包括：①将撞击时间误差和撞击角误差分别映射到两层子滑模面，随后合成复合滑模面；②引入自适应增益调节时间误差，保持系统在不同时间到达估计下的鲁棒性；③对任意时间到达估计均可应用，无需预设特定滑模结构或时间到达估计。

**🔧 技术方法**

使用技术：层次滑模控制、变量增益自适应、非线性接触动力学、预测拦截点（PIP）方法、鲁棒滑模设计与极限时间收敛理论。

**📊 数据集**

数据集：纯数值仿真，设定拦截机速度150 m/s，目标静止或恒速65 m/s，初始距离、LOS角、拦截角等多种场景；未使用真实实验数据。

**📈 对比分析**

比较方法：在相同仿真环境下与现有滑模、PN、最优控制等导引方法对比，评估加速度消耗、时间到达误差、撞击角误差及滑模收敛速度。结果显示，本方法在加速度需求最低、收敛平滑、且能够同时满足时间与角度约束。

**⚠️ 局限性**

局限性：仅针对平面非机动或恒速目标，未验证多目标或协同拦截情况；极端极角状态可能导致增益分母趋近零；理论主要基于平面动力学，三维推广仍需进一步研究。

---

## 63. Handoff Debt: The Rediscovery Cost When Coding Agents Take Over Interrupted Tasks

**arXiv ID:** 2606.02875 | [PDF](https://arxiv.org/pdf/2606.02875v1)

**作者:** Dipesh KC `[一作]` (Independent Researcher), Anjila Budathoki `[通讯]` (Georgia State University)

**通讯引用:** 5 | [OpenAlex ID](https://openalex.org/A5114747733)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究代码代理在交接（handoff）过程中的重新发现成本（handoff debt），并提出衡量此成本的协议；

**💡 创新点**

提出四种交接视图（仅仓库状态、原始轨迹、摘要笔记、结构化笔记），并展示其对重新发现成本与解决率的影响；

**🔧 技术方法**

采用OpenHands风格的终端动作与工具调用框架，使用大型语言模型（Qwen、Gemma、Devstral）进行代码生成与交接；

**📊 数据集**

使用SWE‑Bench Verified 75个源任务作为基准，产生181个交接点并在每个点进行四种视图的接管实验；

**📈 对比分析**

对比不同视图下的代理事件数、提示词数和官方验证通过率，结果显示有上下文的交接能将代理事件减少20–59%、提示词减少42–63%，但对最终解决率的提升较小且模型依赖；

**⚠️ 局限性**

局限包括仅在OpenHands运行时验证、前置代理仅为Qwen，交接视图的大小与信息量权衡不一、且未覆盖更广泛的前置代理与任务多样性。

---

## 64. Improved Postural Stability Using a Lightweight Semi-Active Soft Back Support Device Under Standing Perturbations

**arXiv ID:** 2606.02928 | [PDF](https://arxiv.org/pdf/2606.02928v1)

**作者:** Rohan Khatavkar `[一作]` (School for Engineering of Matter, Transport and Energy), Hyunglae Lee `[通讯]` (School for Engineering of Matter, Transport and Energy)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了一种轻量化半主动软背部支撑装置对站姿扰动后姿态稳定性的影响

**💡 创新点**

创新点在于将被动弹性带与主动逆气动人工肌肉(IPAM)并联，使装置在扰动时能快速提供助力，兼顾轻量与灵活性

**🔧 技术方法**

采用IPAM主动元件、可携带气压系统、运动捕捉与力测装置进行实验与数据分析

**📊 数据集**

实验数据来自5名健康年轻成人，在分割跑步机上完成15次每种条件（无装置、被动模式、主动模式）的前倾扰动实验

**📈 对比分析**

通过WBAM（全身角动量）和MOS（稳定性边际）的统计比较，线性混合效应模型表明主动模式显著降低WBAM RMS/范围并提升MOS，而被动模式效果有限甚至有负面影响

**⚠️ 局限性**

局限性包括样本量小、仅评估单一扰动强度、IPAM被完全放气导致力不可调节，以及跑步机扰动与户外步态恢复的差异

---

## 65. GreenGNN: Energy-Aware Windowed Communication Optimization for Distributed GNN Training

**arXiv ID:** 2606.02916 | [PDF](https://arxiv.org/pdf/2606.02916v1)

**作者:** Arefin Niam `[一作]` (Tennessee Technological University), M. S. Q. Zulkar Nine `[通讯]` (Tennessee Technological University)

**通讯引用:** 40 | [OpenAlex ID](https://openalex.org/A5044491574)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种能耗感知的分布式GNN训练系统，利用窗口化缓存与大块传输降低通信能耗

**💡 创新点**

创新点在于把邻域采样的短时热点特征利用窗口化周期预取，结合离线模拟器自动调优窗口大小，显著减少RPC启动成本与GPU空闲能耗

**🔧 技术方法**

使用离线访问轨迹收集、离散事件模拟器、混合能耗模型、学习排名校正、GPU频率调节、预取线程与批量RPC

**📊 数据集**

Reddit、OGBN-Products、OGBN-Papers100M三大图数据集

**📈 对比分析**

与默认DistDGL、BGL、RapidGNN、GraphStorm对比，能耗下降27–43%，GPU能耗下降36–71%，同时训练吞吐提升1.4–3.9×，在能耗-吞吐曲线上达到Pareto最优

**⚠️ 局限性**

局限包括假设同质硬件、一次性窗口大小选择不适应训练过程动态变化、仅在GraphSAGE上验证，未探索更高性能RDMA或全图训练方法

---

## 66. Attention Calibration for Position-Fair Dense Information Retrieval

**arXiv ID:** 2606.02737 | [PDF](https://arxiv.org/pdf/2606.02737v1)

**作者:** Andrianos Michail `[一作]` (University of Zurich), Rico Sennrich `[通讯]` (University of Zurich)

**通讯引用:** 19302 | [OpenAlex ID](https://openalex.org/A5005771535)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对密集检索模型存在的位置信息偏差，本文提出了一种在推理阶段通过调整注意力分布来缓解偏差的方法，并在不重新训练模型的前提下保持或提升检索效果。

**💡 创新点**

创新点在于：①引入强度系数 λ 以线性插值原始与完全校准的注意力分布，提供可控的偏差调节；②系统评估了篮子大小、校准层数与 λ 的交互作用，确定了一个通用的默认配置（篮子 128、λ=0.5、校准 50% 层），并证明该配置可迁移到多语言、跨语言、不同域的大规模基准 PosIR。

**🔧 技术方法**

核心技术是推理时注意力校准（attention calibration）：将 pooling token 的注意力按连续篮子进行均衡分配，并通过 λ 控制校准强度；实验使用了三种主流嵌入模型（gte‑multilingual‑base、bge‑m3‑dense、Qwen3‑Embedding‑0.6B）以及三类基准数据集。

**📊 数据集**

使用的数据集包括：SQuAD‑PosQ（改造的 SQuAD v2，含 92k 个问答），FineWeb‑PosQ（13k 句子段落，25k 位置感知问答），以及跨语言多领域基准 PosIR（10 语言、31 域、4 文档长度分位）。

**📈 对比分析**

对比方法包括未校准模型、段落平均嵌入（Segment Embed Average）以及不同 λ、篮子大小和层数的校准配置。实验表明，在 FineWeb‑PosQ 上，默认配置在保持 nDCG@10 近似甚至略优的同时，显著提升了不同位置组的 harmonic mean，并降低了 PSI。对 PosIR 的大规模测试显示，该配置在 16 个长度分位 × 模型 × 检索设置中均降低了 PSI，且聚合 nDCG@10 维持或略有提升，优于仅使用段落平均方法。

**⚠️ 局限性**

主要限制包括：①仅针对密集检索模型；②参数搜索空间有限，未进行全局最优优化；③未给出部分校准优于全校准的机制解释；④未探索其他校准变体（如保留部分自注意力或改变特殊 token 的目标权重）。

---

## 67. The Fair Lending Model: How the Longest-Running Algorithmic Fairness Programs Work in Practice

**arXiv ID:** 2606.02957 | [PDF](https://arxiv.org/pdf/2606.02957v1)

**作者:** Emily Black `[一作]` (New York University), Mingwei Hsu `[通讯]` (Upturn)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对美国金融机构在算法公平实践中的测试与缓解过程进行实证研究。

**💡 创新点**

首次提供公平贷款计划的实践层面细节，并阐释监管设计如何塑造其方法。

**🔧 技术方法**

采用半结构化访谈与主题分析技术。

**📊 数据集**

基于对35名行业实践者的访谈记录。

**📈 对比分析**

未对模型性能进行实验比较，研究侧重描述性分析。

**⚠️ 局限性**

样本有限、代表性不足、监管环境变化快，且可能存在社会期望偏差。

---

## 68. Do Value Vectors in Deep Layers Need Context from the Residual Stream?

**arXiv ID:** 2606.02780 | [PDF](https://arxiv.org/pdf/2606.02780v1)

**作者:** Muyu He `[一作]`, Li Zhang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Bank of Values (BoV)，一种在深层 Transformer 中将价值向量从上下文无关的 token 级别向量表中直接查找的注意力变体，消除了对残差流的依赖。

**💡 创新点**

创新点在于发现深层注意力更依赖上下文无关的原始 token 信息，而非残差上下文；通过将这些向量持久化为稀疏参数表，实现了更低 FLOPs 与更少 KV 缓存需求，并在长上下文场景中显著降低内存占用。

**🔧 技术方法**

技术主要包括：使用层级分段的 BoV（仅在后1/3层使用）、可学习的缩放系数、token 级别的价值向量表（E_v）、以及标准 Transformer 的 RMSNorm、Multi‑Head Attention 结构；实现时用 PyTorch 伪代码进行说明。

**📊 数据集**

在 ClimbMix 预训练语料上进行验证，并在 21 个 DCLM CORE 任务（包括 SQuAD、CoQA、BIG‑bench 等）进行评测；对 135M 与 780M 两个规模的模型在 FLOP 控制下进行训练。

**📈 对比分析**

与标准注意力以及之前的基于上下文无关价值向量的变体（如 Value Residual Learning、SVFormer、Additive Lookup 等）对比，BoV 在验证损失上均优于基线，在 780M 模型的 CORE 平均得分上与最佳增量方法相当或更优，且占用更少的显存与 FLOPs。

**⚠️ 局限性**

局限性包括：实验仅在 135M 与 780M 两个规模下验证，未证明更大模型的可扩展性；对深层层依赖上下文无关向量的机制尚未理论化；BoV 的内存优势仅在超越词表大小的长上下文下显现，短上下文时不具优势。

---

## 69. ROBUST-WT: Robust Uncertainty-aware Segmentation Transform via Whitening and Training Enhancements

**arXiv ID:** 2606.03069 | [PDF](https://arxiv.org/pdf/2606.03069v1)

**作者:** Aqsa Naseer `[一作]` (National University of Sciences and Technology), Muhammad Khurram Shahzad `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 5420 | [OpenAlex ID](https://openalex.org/A5006173487)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对WT-PSE医学影像分割模型的训练流程进行改进，提出域自适应增强、BCE+Dice混合损失、课程式Dice权重调度以及命令行消融控制等四项提升措施。

**💡 创新点**

在不改动原始架构的前提下，通过训练时增强与动态损失权重调度，显著提升跨域光盘分割精度，并提供可复现的消融实验框架。

**🔧 技术方法**

域自适应增强（盐噪、伽马校正、随机擦除）、BCE+Dice组合损失、线性课程式Dice权重、基于 Whitening Transform 的概率形状正则化（WT-PSE）以及 Wasserstein 知识蒸馏。

**📊 数据集**

使用眼底视网膜 Fundus 数据集，包含 4 个来源域的光盘（OD）和杯子（OC）分割标注。

**📈 对比分析**

在光盘分割任务中与多种基准方法（SAML、KODG、DSU 等）对比，最终 Dice 95.66%（OD）、ASD 0.922 mm，显著优于原 WT-PSE（Dice 93.80%）及其他竞争方法；杯子分割 Dice 84.42% 同样领先。

**⚠️ 局限性**

仅在光盘分割上验证，未在其他器官或 CT 等域上评估；增强与权重参数需手工调优；对低资源或实时部署的效率评估不足。

---

## 70. Slipstream: Locality-Aware Graph Index Construction for Streaming Approximate Nearest Neighbor Search

**arXiv ID:** 2606.02992 | [PDF](https://arxiv.org/pdf/2606.02992v1)

**作者:** Shubing Yang `[一作]` (University of Washington), Dongfang Zhao `[通讯]` (University of Washington)

**通讯引用:** 1369 | [OpenAlex ID](https://openalex.org/A5101671477)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在流式近邻检索中，设计了一种 Slipstream 方法，通过缓存前一次插入的候选节点并使用自适应控制器来快速定位新向量，从而显著降低图索引插入的计算成本。

**💡 创新点**

创新点在于：①利用向量流的时间连续性，缓存并重用前一次插入的候选集；②引入接近度比率来判断何时安全重用；③设计自适应 beam 宽度控制器根据流的稳定性动态调节搜索宽度；④提供理论模型和上界证明其可靠性。

**🔧 技术方法**

使用技术包括：图基近邻检索（HNSW）、候选集缓存、接近度比率判定、可调节 beam 宽度的自适应控制器、理论分析与实验评估；实现基于 Faiss 与 HNSWLib 两大开源库。

**📊 数据集**

使用的五个视频嵌入流数据集：Kinetics、BDD100K、Epic‑Kitchens、Ego4D 和 VIRAT（均采用 CLIP 生成的 512 维向量）。

**📈 对比分析**

与四个基线（HNSWLib 原生、Faiss 原生、Ada‑ef、DARTH）比较。Slipstream 在保持至少 0.95 recall@10 的前提下，吞吐量提升最高可达 30.8×，且在所有数据集上均表现出较低的插入成本与较高的实时性。

**⚠️ 局限性**

局限性包括：依赖向量流的时间连续性，对高度随机或剧烈漂移的流效果下降；需要微调回退阈值与自适应参数；在 HNSWLib 上存在轻微内存占用上升；未在非视频或高维稀疏数据上进行评估。

---

## 71. A Nonmonotone Gradient-Based Algorithm for Symmetric Nonnegative Matrix Factorization and Graph Clustering

**arXiv ID:** 2606.02887 | [PDF](https://arxiv.org/pdf/2606.02887v1)

**作者:** Ryan Swart `[一作]` (Arizona State University), Johannes Brust `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于非单调投影Barzilai‑Borwein方法的对称非负矩阵分解(Symmetric NMF)算法，并将其扩展到图聚类(Graph‑SNMPBB)与低秩近似( LAI‑SNMPBB)版本。

**💡 创新点**

首次将非单调Barzilai‑Borwein投影梯度方法应用于对称NMF，采用二次惩罚耦合保持对称性，并引入图拉普拉斯正则化和低秩近似，证明全局收敛并在实验中显著加速。

**🔧 技术方法**

使用非单调投影Barzilai‑Borwein步骤、Armijo线搜索、惩罚二次项、图拉普拉斯正则化、低秩随机化(LAI)以及相关收敛理论。

**📊 数据集**

使用合成对称矩阵、六个图聚类基准（COIL20、Isolet1、MNIST、ORL、Reuters‑21578、TDT2）、34个SuiteSparse稀疏矩阵以及Web of Science文本数据集进行实验。

**📈 对比分析**

与SymANLS、SymNewton、PGD、LAI‑SymPGNCG等传统算法对比，SNMPBB 在相同残差下速度提升约6倍；Graph‑SNMPBB 在图聚类准确率上与或优于 SymANLS；LAI‑SNMPBB 在 SuiteSparse 上既比 LAI‑SymPGNCG 快又得到更低残差。

**⚠️ 局限性**

对极大稀疏矩阵时线搜索可能收敛慢，且算法对超参数（λ、γ、K）和初始策略依赖较大，实验中未对迭代次数上限及不同初始化进行系统性分析。

---

## 72. Learning Coherent Representations: A Topological Approach to Interpretability

**arXiv ID:** 2606.02841 | [PDF](https://arxiv.org/pdf/2606.02841v1)

**作者:** Sigurd Gaukstad `[一作]` (NTNU), Benjamin Dunn `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于拓扑连贯性的正则化方法 Coh，强制神经网络的样本空间与特征空间共享相似的拓扑结构，从而实现可解释的特征表示。

**💡 创新点**

创新点在于将“连贯性”（每行与列的几何聚类与覆盖）与 Vietoris–Rips 融合，给出可微的 Fréchet 方差损失，并证明连贯矩阵可诱导行列空间的有限间距互插，从而保证两者拓扑兼容。

**🔧 技术方法**

技术手段包括非负激活函数、软最大化或 L¹ 正则化做归一化、Fréchet 均值与方差、Top‑k 聚合、基于 UMAP 与 Ripser 的拓扑评估以及自定义的 MRL、纯度等可解释性指标。

**📊 数据集**

实验数据集涵盖合成两圆点、旋转 MNIST（单数与双数）以及 WikiText‑2 的词向量，使用 BERT‑style 词嵌入模型检验 Coh 在语言领域的可解释性。

**📈 对比分析**

与普通自编码器和 L¹ 稀疏正则化对比，Coh 在圆形结构任务中实现 100% 的角度调制率、90%+ 的纯度，并在词嵌入中获得 87% 的可解释特征；在整体性能（MSE）上几乎无负面影响。

**⚠️ 局限性**

局限性包括需经验验证的 1‑Lipschitz 约束、可能产生多种可连贯但不同的解（尤其在双数旋转任务中）以及对更大规模模型与多层应用尚未系统验证。

---

## 73. ReLoRA: Knowledge-Reusing Adaptation for Fast Rollout of Evolving LLM Services

**arXiv ID:** 2606.02606 | [PDF](https://arxiv.org/pdf/2606.02606v1)

**作者:** Yang Xu `[一作]` (University of Science and Technology of China), Xitong Fu `[通讯]` (University of Science and Technology of China)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为ReLoRA的框架，快速更新已部署的LoRA适配器以适应LLM基座的演进，保证服务质量并缩短上线时间。

**💡 创新点**

创新点在于：① 将旧适配器与基座演进的参数差异通过贝叶斯优化自适应融合，形成兼容性更强的初始化；② 在微调阶段采用两阶段调度正则化（先强正则快速恢复，再弱正则精细调优），显著加速收敛且提升最终准确率。

**🔧 技术方法**

主要技术包括LoRA参数高效微调、贝叶斯优化（Gaussian Process + Expected Improvement）用于搜索融合系数、L2正则化调度、cosine annealing学习率以及LoRA与DoRA等PEFT变体的融合。

**📊 数据集**

使用六个下游任务数据集：MMLU、SST-2、AGNews、20News、MNLI、SNLI，并在三种LLM基座（LLaMA2-7B、LLaMA3.1-8B、Mistral-7B）以及三种演进来源（OpenOrca、AlpacaGPT4、OpenPlatypus）进行实验。

**📈 对比分析**

与基线LoRA-Scratch、PortLLM+FT、ORAL以及零样本基座进行对比，ReLoRA在所有任务上均实现平均准确率提升2.6~4.6个百分点，且时间‑到‑就绪（time‑to‑readiness）提升高达8.9×，整体上线时长缩短56.9%，相较于LoRA‑Scratch加速2.32×。

**⚠️ 局限性**

局限性包括：① 仅使用全局比例系数α、β，未考虑层级或模块级别的融合细粒度；② 贝叶斯优化的代理目标为初始验证损失，可能无法完美预测最终性能；③ 仅在离线微调场景验证，未评估在线服务的SLA、延迟或多租户干扰；④ 对超参数的敏感性虽低，但仍需在不同任务/基座上微调。

---

## 74. Rotatable Antenna Meets Multiple Access: NOMA or OMA?

**arXiv ID:** 2606.03035 | [PDF](https://arxiv.org/pdf/2606.03035v1)

**作者:** Qi Dai `[一作]` (South China University of Technology), Fangjiong Chen `[通讯]` (South China University of Technology)

**通讯引用:** 29183 | [OpenAlex ID](https://openalex.org/A5062293532)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文对使用可旋转天线（RA）的两用户下行系统中，非正交多址（NOMA）与正交多址（OMA）（包括TDMA、FDMA）的功率最小化问题进行理论分析与数值优化。

**💡 创新点**

创新点在于：①首次将RA与NOMA/OMA进行系统级比较，揭示RA可显著提升多址效率；②推导出针对不同解码顺序与资源分配方式的闭式最小功率表达式；③证明在RA辅助下，NOMA总能不劣于FDMA，并在对称部署时可能优于TDMA；④提出基于粒子群优化（PSO）的角度优化方法，实现近似全局最优。

**🔧 技术方法**

使用的主要技术包括：可旋转天线模型与方向增益函数、Rician小尺度衰落模型、超位置编码与成功干扰消除（SIC）技术、功率最小化约束优化、PSO算法以及理论证明与对称/非对称部署仿真。

**📊 数据集**

采用仿真数据：在两个部署场景（对称用户和近远用户），随机生成1000组用户位置，设置固定的信道、噪声、频率、天线参数等进行性能评估；未使用公开真实数据集。

**📈 对比分析**

比较方法：在相同目标速率、相同信噪比、相同角度限制下，分别求解RA‑NOMA、RA‑TDMA、RA‑FDMA和相应的固定天线基线的最小发射功率；结果表明RA相较于固定天线能显著降低功率；NOMA在非对称部署下始终优于TDMA/FDMA，且在对称部署时TDMA因可时隙旋转可部分抵消频谱效率损失。

**⚠️ 局限性**

局限性：仅考虑两用户下行场景，未讨论多用户配对、上行或多天线阵列；RA模型假设单一可旋转天线且仅考虑方向增益；仿真基于理想化信道模型和静态小尺度衰落；PSO为启发式方法，无法保证全局最优。

---

## 75. BYORn: Bootstrap Your Own Responses to Defend Large Vision-Language Models Against Backdoor Attacks

**arXiv ID:** 2606.02947 | [PDF](https://arxiv.org/pdf/2606.02947v1)

**作者:** Ivan Sabolić `[一作]` (University of Zagreb), Sven Lončarić `[通讯]` (University of Zagreb)

**通讯引用:** 4803 | [OpenAlex ID](https://openalex.org/A5047540566)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对多模态模型的后门攻击，提出了 BYORn 方案，通过基于预训练模型的低概率响应检测和动态重生成来实现鲁棒微调。

**💡 创新点**

创新点在于：① 利用目标响应的困惑度来识别潜在后门样本；② 采用自举响应生成破坏触发器与攻击输出的关联；③ 证明其等价于对清洗数据分布的风险上界进行经验估计。

**🔧 技术方法**

主要技术包括：预训练视觉‑语言模型、LoRA 参数适配、指数滑动平均(EMA)、阈值检测、蒙特卡罗采样以及针对后门样本的高效小批量训练。

**📊 数据集**

实验数据集涵盖 LADD、Flickr30k、COCO、CGD 以及 ScienceQA，覆盖图像描述、差异识别和视觉问答等任务。

**📈 对比分析**

与 SFT、ONION、BYE 等基线对比，BYORn 在 CIDEr、ASR（攻击成功率）和 ACC（问答准确率）上均取得最优或接近最优表现，攻击成功率显著降低到 0% 或接近 0%，同时保持或提升了生成质量。

**⚠️ 局限性**

局限性包括：依赖预训练模型未被后门污染；对极其语义合理的攻击（如高相似度的干净标签攻击）可能仍有风险；在更复杂的自适应攻击或不同攻击模式下的鲁棒性仍待进一步验证。

---

## 76. Lean-GAP: A Dataset of Formalized Graduate Algebra Problems

**arXiv ID:** 2606.02588 | [PDF](https://arxiv.org/pdf/2606.02588v1)

**作者:** Seewoo Lee `[一作]` (University of California Berkeley), Kyu-Hwan Lee `[通讯]` (University of Connecticut)

**通讯引用:** 4403 | [OpenAlex ID](https://openalex.org/A5084330981)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `79276348-11e0-48e3-84bc-7ec231d0171c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了一个基于《Abstract Algebra》教材练习的 Lean4 形式化数据集；

**💡 创新点**

提出端到端的自动形式化流水线，并系统分析验证阶段的关键难点；

**🔧 技术方法**

结合 OCR、LLM 自动生成 Lean 代码、GitHub CI 自动编译和人工审查等技术；

**📊 数据集**

以 Dummit‑Foote 约1966道练习为数据源，已形式化约20%；

**📈 对比分析**

与多款 LLM（GPT‑5、Gemma‑31B、Goedel‑Formalizer 等）比较，单次生成通过率最高为 GPT‑5 44%，但语义正确率仅约 40%；Codex 通过率 95% 但仍需人工校验；

**⚠️ 局限性**

主要局限在于语义验证仍需人工干预，模型易遗漏假设或误译数学含义。

---

## 77. Efficient Hyperparameter Optimization for LLM Reinforcement Learning

**arXiv ID:** 2606.03073 | [PDF](https://arxiv.org/pdf/2606.03073v1)

**作者:** Minping Chen `[一作]` (Hong Kong University of Science and Technology), Zeyi Wen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1216 | [OpenAlex ID](https://openalex.org/A5013127195)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了联合多可信度的超参数优化框架 JF‑HPO，利用小代理模型、检查点注册和训练动态早停，大幅提升 LLM 强化学习超参数搜索效率。

**💡 创新点**

创新点在于将模型规模与训练预算同时视为可信度，采用代理模型、多可信度 Bayesian 优化、动态早停和检查点机制，实现单次实验高达 14.9 倍的加速。

**🔧 技术方法**

采用 Bayesian Optimization、代理模型、多可信度策略、GRPO、KL 与奖励动态的早停、检查点机制等技术。

**📊 数据集**

使用了 GSM8K、MATH、OpenBookQA、MMLU 等多任务评测数据集。

**📈 对比分析**

在 48 小时预算下与 VeRL Recipe、Random Search、BOHB 对比，JF‑HPO 在 24 次实验中全部击败或匹配对手，平均提升 5.8%–111.6%。

**⚠️ 局限性**

局限性是需要代理模型与目标模型保持性能相关，难以迁移到架构差异大或生成任务（如创意写作）场景。

---

## 78. ATLAS: A Large-Scale Evaluation Benchmark for Adversarial LiDAR Perception

**arXiv ID:** 2606.02924 | [PDF](https://arxiv.org/pdf/2606.02924v1)

**作者:** Mellon M. Zhang `[一作]` (Georgia Institute of Technology), Glen Chou `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 437 | [OpenAlex ID](https://openalex.org/A5006149535)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ATLAS 基准，用于在真实驾驶序列中对 LiDAR 感知模型进行黑盒物理注入与移除攻击的评估；

**💡 创新点**

首次构建大规模、基于物理的 LiDAR 逆向攻击基准，系统评测现代感知模型并发现性能强的模型对注入攻击更脆弱但对移除攻击更稳健；

**🔧 技术方法**

采用基于 ray‑casting 的点注入模拟、经验化的点移除率、时间攻击调度、训练时的 ground‑truth 采样等技术来生成逼真的攻击样本；

**📊 数据集**

使用 Waymo Open Dataset 的验证集作为真实驾驶场景基础，构造 12 种攻击变体，共计 460k+ 点云；

**📈 对比分析**

对 9 种单帧、时序、流式、摄像头‑LiDAR 融合等不同架构进行 ASR 对比，结果显示：强模型在移除攻击下 ASR 更低（更鲁棒），但在注入攻击下 ASR 更高（更脆弱）；

**⚠️ 局限性**

仅基于软件仿真，未验证真实硬件攻击；只考虑两种攻击模式；对不同模型的攻击难度不完全一致，可能影响对比结论；

---

## 79. Reed-Muller type codes over a combinatorial simplex: an algebraic description

**arXiv ID:** 2606.02819 | [PDF](https://arxiv.org/pdf/2606.02819v1)

**作者:** Hiram H. López `[一作]`, Nart Shalqini `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

无法确定具体研究内容，缺乏足够的论文信息。

**💡 创新点**

无法识别创新点。

**🔧 技术方法**

缺少技术细节描述。

**📊 数据集**

未知使用的数据集。

**📈 对比分析**

没有可比方法或性能评估信息。

**⚠️ 局限性**

未能评估论文的局限性。

---

## 80. Hint-Guided Diversified Policy Optimization for LLM Reasoning

**arXiv ID:** 2606.03021 | [PDF](https://arxiv.org/pdf/2606.03021v1)

**作者:** Zhiyu Cao `[一作]` (Soochow University), Qiaoming Zhu `[通讯]` (Soochow University)

**通讯引用:** 2860 | [OpenAlex ID](https://openalex.org/A5102065469)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Hint‑Guided Diversified Policy Optimization (HDPO)，通过“提议‑选择‑思考”两阶段训练（先用 SFT 生成多种候选解，再用 RL 通过多样性与可靠性奖励进一步优化策略），使 LLM 在推理任务中能够主动探索多条解答路径并挑选最可靠的一条进行深入推理。

**💡 创新点**

核心创新点包括：①将多样性奖励与可靠性奖励联合引入 RLVR，鼓励模型产生多样化且可信的候选解；②使用熵作为可靠性代理并按排名给予递减奖励；③采用正弦调度策略逐步提高多样性奖励的比例；④内部化“提议‑选择‑思考”流程到模型参数，实现零开销的单次推理；⑤提出自演化迭代方案，用模型自身生成冷启动数据，提升可扩展性。

**🔧 技术方法**

使用的技术与方法包括：强化学习与可验证奖励 (RLVR)；Group Relative Policy Optimization (GRPO)；监督微调 (SFT)；格式奖励、准确奖励、多样性奖励与可靠性奖励的奖励塑造；多样性调度策略；熵计算与候选解排名；以及自演化训练循环。

**📊 数据集**

训练与评估数据集：冷启动阶段基于 DAPO‑Math‑17k；RL 阶段基于 DeepScaleR‑Preview‑Dataset；评估使用九大推理基准：AMC 2023、AIME 2024/25、Math‑500、Minerva、OlympiadBench、MMLU‑Pro、SciBench、GPQA‑Diamond；实验涉及 Qwen、Llama、DeepSeek 等多种 LLM。

**📈 对比分析**

与基线（GRPO、EDGE‑GRPO、Critique‑GRPO 等 RLVR 方法）以及原始模型在同一基准上的比较显示，HDPO 在数学推理与通用推理任务上均取得显著提升，Qwen3‑4B、Qwen2.5‑Math‑7B 等模型在多项指标上提升 7–10 分以上，甚至在 4B 模型上超过部分 8B 模型；自演化迭代也能快速逼近教师模型效果。

**⚠️ 局限性**

局限性包括：①候选解中仍可能包含错误答案，可能干扰后续推理；②可靠性奖励依赖熵作为代理，若熵与实际可靠性关联不足，奖励可能失效；③在极端复杂任务中多样性奖励与准确奖励的平衡仍需进一步调优。

---

## 81. CL-DMDF:Dynamic Multimodal Data Fusion Model Based on Contrastive Learning

**arXiv ID:** 2606.02659 | [PDF](https://arxiv.org/pdf/2606.02659v1)

**作者:** Dong Li `[一作]` (Liaoning University), Yue Kou `[通讯]` (Northeastern University)

**通讯引用:** 1627 | [OpenAlex ID](https://openalex.org/A5026995541)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种基于对比学习的动态多模态融合框架CL‑DMDF，能够在模态缺失或不确定的情况下高效融合多模态信息。

**💡 创新点**

创新点包括：双维注意机制（特征级与模态级）结合实体‑质心对比学习，以及自适应融合模块实现任务自适应与资源效率。

**🔧 技术方法**

主要技术手段为：双维注意机制、实体‑质心对比学习（对比损失与温度调节）、可变门控自适应融合、批归一化与线性投影。

**📊 数据集**

在MM‑IMDB、NYU Depth V2和CMU‑MOSEI三个公开数据集上进行实验。

**📈 对比分析**

与多种基线（单模、编码‑解码、注意力、图神经网络等）对比，CL‑DMDF在MM‑IMDB上微F1 63.25%、MacroF1 53.28%，NYU Depth V2上MIoU 52.3%，CMU‑MOSEI上准确率 85.4%，均显著优于或接近最优模型，同时计算量更低。

**⚠️ 局限性**

目前仍未针对完全缺失或不完整模态进行专门设计，实时推理性能与温度参数敏感性未得到充分评估。

---

## 82. Exact equivariance, kept through training, buys zero-shot generalisation across the symmetry group

**arXiv ID:** 2606.03003 | [PDF](https://arxiv.org/pdf/2606.03003v1)

**作者:** Hongbo Wang `[一作]` `[通讯]` (Stony Brook University), Hongbo Wang (Stony Brook University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过在隐状态空间构造一个完全等变的世界模型（Encoder + Predictor），证明并实验验证了等变性在训练、预测和闭环控制中保持的绝对精度，从单个训练方向零射击到全组别。

**💡 创新点**

创新点包括：① 用等变性理论证明一阶预测误差在整个群上严格平坦；② 证明等变性不被任意优化器破坏；③ 在闭环控制与主动推理任务中实现浮点精度的全组别不变性；④ 对抗性扰动、数据量与模型容量的交叉实验，阐明等变性在样本效率和群内外泛化上的独特优势。

**🔧 技术方法**

技术：Vector‑Neuron 等变层、Tensor‑Field 网络、E(n)‑等变图卷积；基于JEPA（Encoder–Predictor）框架的隐状态模型；Muon's AdamW + EMA + VICReg 训练；对比不等变 MLP；强化学习用CEM‑MPC 等变规划器；主动推理用期望自由能目标。

**📊 数据集**

数据集：1) 真实 PushT 机器人抓取模拟器（2D SO(2)）; 2) 合成教师 3D 点云（SO(3)）; 3) 3D SE(3) 轨迹生成；所有实验均在 CPU/MPS 规模下完整种子化。

**📈 对比分析**

比较方法：将等变模型与同规模或更大容量的非等变 MLP 在同一数据、同一训练迭代数下对比；评估指标包括一阶相对 MSE、OOB 乘数、闭环角度误差、主动推理成功率。结果显示：等变模型参数量仅 4.5–7.4×，在 OOD 组别上误差几乎完全保持不变（×1.00），而非等变基线则在 OOD 上爆炸（×13–×157）；闭环控制在旋转方向上保持浮点精度，主动推理成功率大幅提升。

**⚠️ 局限性**

局限：未验证二值任务成功率；闭环不变性需等变规划器；仅在隐状态规划（无解码器）下实现；对交互物体的全组别泛化有限；SO(2) 仅使用标量权重导致缺失 90° 生成器；实验规模受限于笔记本级别；对真实观测噪声和非等变动态的鲁棒性尚未深入探究。

---

## 83. Spectral Asymptotics of Neural Network Loss Landscapes: An Exact Decomposition of the Curvature Exponent

**arXiv ID:** 2606.02596 | [PDF](https://arxiv.org/pdf/2606.02596v1)

**作者:** Anherutowa Calvo `[一作]` `[通讯]` (9D Labs), Anherutowa Calvo (9D Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了神经网络梯度奇异值与Hessian特征值的幂律关系，并提出了谱对齐分解。

**💡 创新点**

创新在于将曲率指数α解释为对齐度的log-log斜率，推出α与梯度有效秩衰减γ以及Hessian衰减s的代数关系s=αγ，并给出基于α的架构自适应预条件器。

**🔧 技术方法**

使用Kronecker因子近似、Hessian‑vector product测量、SVD、BIC模型选择、Zeta函数参与度上界，以及基于谱的Newton优化器Spectral Newton。

**📊 数据集**

在CIFAR‑10、Tiny‑ImageNet‑200、ImageNet‑1K预训练权重以及GPT‑2小规模模型上进行实验。

**📈 对比分析**

与AdamW、MuON等优化器对比，Spectral Newton在ResNet‑18/50等视觉任务上实现了1.1%–3.5%的准确率提升，验证了s=αγ预测的有效性。

**⚠️ 局限性**

局限在于仅验证了小规模模型，缺乏大规模（数十亿参数）LLM的实验；Spectral Newton的多种种子、时间成本评估和对不同硬件的适配仍待进一步研究。

---

## 84. Bounds for Single-Error-Correcting Analog Codes

**arXiv ID:** 2606.03011 | [PDF](https://arxiv.org/pdf/2606.03011v1)

**作者:** Hengzhuo Li `[一作]` (Xi'an Jiaotong University), Xin Wang `[通讯]` (Soochow University)

**通讯引用:** 85350 | [OpenAlex ID](https://openalex.org/A5100328102)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

研究了实数域上单错误纠正的模拟码，并给出了最小误差分离比的下界与上界。

**💡 创新点**

证明了冗余为2的线性[ n , n‑2 ]模拟码的误差分离比达到 1/ sin²(π/2n)，从而解决了Roth关于该构造最优性的未解决问题；并给出了固定冗余 r≥2 的上界 Γ₂(n,n‑r)=O(n^{1+1/(r-1)})。

**🔧 技术方法**

主要使用几何方法：通过zonotope 及其对称性转换为极化角度问题，利用周期正弦乘积不等式与 AM‑GM 估计来得到下界；上界则依赖于构造低相干度的单位向量集合（球面分隔或格点投影）并运用 Song‑Cai 的相干度与 Γ₂ 的关系。

**📊 数据集**

本工作为理论性论文，不依赖实验数据集；所有结果均为数学证明与构造。

**📈 对比分析**

与现有的 Roth、Song‑Cai 等构造相比，新上界在 r=2,3 时与已知构造匹配，且对任意 r≥2 给出统一的 n^{1+1/(r-1)} 量级；下界与 Roth 构造的量级完全匹配，证明其最优。

**⚠️ 局限性**

主要局限在于上界常数尚不最优，尤其是构造实现时常数较大；对高冗余 r 的实际构造仍需进一步优化，以降低常数并验证在具体应用中的鲁棒性。

---

## 85. Don't Gamble, GAMBLe: An Analytical Framework for AI-Driven Research Systems

**arXiv ID:** 2606.02863 | [PDF](https://arxiv.org/pdf/2606.02863v1)

**作者:** Marquita Ellis `[一作]` (IBM Research), Paul Castro `[通讯]` (IBM Research)

**通讯引用:** 2795 | [OpenAlex ID](https://openalex.org/A5045374976)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了ADRS（AI-Driven Research Systems）的理论分析框架，将系统分解为生成器G、评估器F、探索机制M和预算B，并定义了有效景观L_eff，用以解释生成器敏感性、G×M交互以及运行非马尔可夫性质，并在760+次实验中验证了该框架。

**💡 创新点**

创新点在于证明ADRS最佳分数过程非马尔可夫并给出有效景观概念，定义生成器与评估器的“天花板”以及四种限制状态，实现基于目标的组件诊断。

**🔧 技术方法**

技术手段包括基于LLM的候选生成、多样化评估反馈、贪婪与共进化搜索机制、混合生成器（静态与动态适应）、以及对生成器、评估器和机制交互的理论推导与实验验证。

**📊 数据集**

使用了三类NP-hard问题（例如Cap Set、矩阵乘法、优化竞赛）以及对应的评估器（连续评分、阶梯函数），在这些问题上对12种生成器、3种搜索机制进行>46,000次迭代的复制实验。

**📈 对比分析**

与传统基准方法对比，适当选择生成器和机制可提升13–67%性能、6–39倍搜索效率；实验显示不同组合存在前沿模型表现不佳，最简机制有时优于高级元搜索。

**⚠️ 局限性**

局限性包括理论假设可能受特定问题影响、实验仅覆盖竞赛类问题、对计算成本未细化、对低概率突破事件检测不足，以及对有效景观多峰结构的具体形态未完全解析。

---

## 86. Representational Capacity: Geometric Limits on Feature Representation in Transformer Language Models

**arXiv ID:** 2606.02765 | [PDF](https://arxiv.org/pdf/2606.02765v1)

**作者:** Alexander Guha `[一作]` `[通讯]` (Arizona State University), Alexander Guha (Arizona State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了 transformer 语言模型中，隐藏空间维度与可表示特征数量之间的几何约束；

**💡 创新点**

创新点包括：①提出使用嵌入矩阵的余弦相似度分布来估计可接受的非正交偏差 ε；②发现两类模型（高 ε 与低 ε）并给出分裂阈值；③在 Johnson‑Lindenstrauss (JL) 定理基础上引入 k/d 比例的修正公式，显著提升对训练后向量包络的预测精度；

**🔧 技术方法**

技术手段：嵌入相似度分布统计、μ+2σ 阈值估算、随机向量实验、优化向量的 Gram 矩阵罚函数、改进 JL 公式、计算表示容量 repcap；

**📊 数据集**

数据集：多款公开的语言模型（如 Llama、Gemma、Qwen 等）的嵌入矩阵，未使用传统自然语言语料库；

**📈 对比分析**

比较方法：用传统 JL 与改进公式计算同一模型的可近正交方向数，并与模型词表大小和已提取的特征数对照；结果显示传统 JL 低估两到三阶量级，改进公式误差下降至两位数百分比；

**⚠️ 局限性**

局限性：①假设嵌入几何代表整体隐藏空间，缺乏实验验证；②μ+2σ 的阈值为启发式，对 ε 的估计极其敏感；③表示容量仅给出几何上可能的最大方向数，未说明模型实际能利用多少；④未考虑层归一化、旋转编码等对内部表示的潜在影响。

---

## 87. SaluNet: Enabling Total Plasticity in Normalization-Free Deep Networks

**arXiv ID:** 2606.02927 | [PDF](https://arxiv.org/pdf/2606.02927v1)

**作者:** Mourad Zaied `[一作]` `[通讯]` (University of Gabes), Mourad Zaied (University of Gabes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了SaluNet框架，用可学习的激活函数SALU（及其门控变体SWALU、GALU）完全替代批归一化与层归一化，实现无归一化深度网络；

**💡 创新点**

创新点在于通过可学习的几何参数（a、b）实现激活函数的自适应稳定化（“几何可塑性”），消除了传统归一化抑制的可塑性，形成总可塑性（Total Plasticity）架构；

**🔧 技术方法**

核心技术包括：SALU激活函数（a·x/√(1+abx²)）、SWALU（Swish的SALU门控版）、GALU（GELU的SALU门控版）、在CNN和ViT中移除BN/LN，配合高学习率的激活参数、EMA、Mixup等正则；

**📊 数据集**

使用CIFAR-10、CIFAR-100和ImageNet-1K数据集进行实验；

**📈 对比分析**

与BN/LayerNorm+ReLU、Swish/GELU等基线对比，SaluNet-C-18在CIFAR-10/100分别达到97.35%/83.25%（无归一化），在BatchSize=1时仍保持93.44%/76.23%；SaluNet-C-50在ImageNet 90轮达78.67%（224×224）/79.23%（288×288）；SaluNet-T-CIFAR在ViT上亦优于LN+GELU；

**⚠️ 局限性**

局限性包括：对学习率调度敏感、需要足够数据多样性与正则化、在极深网络的下采样层初始化需特殊处理、实现尚未优化CUDA核，对大模型的计算开销略高；

---

## 88. Neural Networks Provably Learn Spectral Representations for Group Composition

**arXiv ID:** 2606.02993 | [PDF](https://arxiv.org/pdf/2606.02993v1)

**作者:** Jianliang He `[一作]` (Yale University), Zhuoran Yang `[通讯]` (Yale University)

**通讯引用:** 4827 | [OpenAlex ID](https://openalex.org/A5101727948)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在有限群上训练两层网络学习群运算，并给出梯度流的理论分析

**💡 创新点**

证明梯度流会收敛到单个不可约表示并实现秩‑1旋转对齐，且在阿贝尔群中得到完全均匀多样化和指数收敛速度

**🔧 技术方法**

采用群傅里叶变换、有限群表示理论与Riemannian梯度流分析，结合投影梯度流和两阶段训练方案

**📊 数据集**

使用人工构造的有限群数据（如C₇⋊C₃、ℤ₃⊕ℤ₅ 等）进行实验验证

**📈 对比分析**

在训练中实现了 100% 准确率，验证了理论预测的“错误指示器”与对数时间尺度增长；未与其他模型做数值比较

**⚠️ 局限性**

对自共轭表示、非平衡样本、训练‑测试泛化（grokking）以及高维不可约表示的收敛分布缺乏完整理论说明

---

## 89. Libra: Efficient Resource Management for Agentic RL Post-Training

**arXiv ID:** 2606.03077 | [PDF](https://arxiv.org/pdf/2606.03077v1)

**作者:** Kaiwen Chen `[一作]` (Chinese University of Hong Kong), Hong Xu `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 20561 | [OpenAlex ID](https://openalex.org/A5034735808)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Libra，针对 agentic RL 后训练阶段的资源调度系统，包含周期性全局资源规划器、弹性混合池以及基于工具调用因果信号的 C‑MLFQ 调度器。

**💡 创新点**

创新点：① 将 rollout 与 training 的 GPU 分配视为耦合优化问题，周期性重规划并通过弹性混合池实现无阻塞重分配；② 采用因果驱动的多层反馈队列（C‑MLFQ）替代传统长度预测，利用工具返回结果即时决定请求迁移；③ 通过决策树与动态规划实现高效的异构 rollout 分区；④ 引入精确的成本评估器，兼顾 decode 与训练的非均匀序列长度。

**🔧 技术方法**

技术：分层资源规划器（决策树+动态规划）、基于多项式回归的成本评估器、非阻塞梯度通道与快照恢复、KV‑cache 迁移与 CPU‑侧重 shard、前缀树因果路由、Python/C++/CUDA 实现。

**📊 数据集**

数据集：Search‑R1、R2E‑Gym、DAPO‑Math‑17K 三大 agentic RL 基准；模型为 Qwen3‑14B 与 Qwen3‑30B‑A3B；硬件为 48 颗 NVIDIA A800‑SXM4‑80GB GPU。

**📈 对比分析**

比较方法：对比 verl‑Colocated、verl‑Static‑Uniform、verl‑Greedy‑Heuristic、AReaL‑Static‑Optimal 四个基线；实验显示 Libra 在吞吐量上提升至 2.7k‑3.1k tokens/s（比最优静态分配提升 60‑80%），奖励收敛时间缩短 1.5‑2.5×（例如 Search‑R1 仅 17.9h）。

**⚠️ 局限性**

局限性：① 依赖准确的成本模型与离线因果树，模型更新或工具行为突变时需重建；② 弹性重分配虽无阻塞但仍有 3.5% 规划与迁移开销；③ 对极端大规模 GPU 或跨节点高网络延迟场景尚未彻底验证；④ 目前针对 Qwen 系列模型和 GRPO 算法，泛化到其它 RL 算法与模型仍需进一步验证。

---

## 90. AURA: Action-Gated Memory for Robot Policies at Constant VRAM

**arXiv ID:** 2606.02775 | [PDF](https://arxiv.org/pdf/2606.02775v1)

**作者:** Josef Chen `[一作]` `[通讯]` (KAIKAKU), Josef Chen (KAIKAKU)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种“惊讶门控、动作信息瓶颈”记忆机制，能在机器人连续长任务中保持常数大小的推理状态并显著减少写入频率；

**💡 创新点**

创新点在于：①基于动作误差的惊讶门控只在对行动有用时写入；②使用动作信息瓶颈目标直接训练记忆写入；③通过写入率控制参数实现可调写入-准确性折中；④对行动足够性进行AIS值损失界定并在真实VLA上验证；

**🔧 技术方法**

技术包括：冻结的VLA Backbone、TTT（快速权重更新）外加门控MLP、动作信息瓶颈（KL正则）、一阶梯度门控、写入率惩罚、AIS理论框架；

**📊 数据集**

使用合成的“多绑定关联记忆”与“事件稀疏流”两类基准，实际测试在OpenVLA‑OFT 7B/ LIBERO‑Long回合，未使用真实机器人数据；

**📈 对比分析**

与无门控、总写入、随机/周期写入、GRU、以及基于Token重建的门控等8个对照模型在相同参数量下进行对比；结果显示在硬任务上取得≈1.0精度，写入率降低4.98–9.19倍，随机/周期门控在相同写入率下准确率仅≈0.37；

**⚠️ 局限性**

局限性包括：①O(1)仅指推理状态；训练仍O(T)；②在高精度任务上写入率提升并不伴随准确率提升；③AIS值损失界定在当前规模下为真空；④门控模型参数比无门控多41.9%；⑤实验仅在模拟合成基准，未验证真实机器人或能耗/延迟；⑥动作信息瓶颈贡献尚未显著独立验证。

---

## 91. The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset

**arXiv ID:** 2606.02956 | [PDF](https://arxiv.org/pdf/2606.02956v1)

**作者:** Richard Schwarzkopf `[一作]` (FZI Research Center for Information Technology), Christoph Stiller `[通讯]` (FZI Research Center for Information Technology)

**通讯引用:** 23501 | [OpenAlex ID](https://openalex.org/A5091574711)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `3f18e8e3-0266-457c-8567-9039b6d2394d` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了KITScenes Multimodal多模态欧洲驾驶数据集，配备同步高清摄像头、长距离激光雷达、4D成像雷达以及完整的HD地图，并提出了四个面向空间学习的基准任务。

**💡 创新点**

创新点在于：① 结合高精度同步多模态传感器与全景覆盖，② 覆盖欧洲复杂城市且提供最高完整度的3D HD地图（Lanelet2），③ 通过四个基准揭示现有方法在在线地图构建、长距离深度估计、结构一致的视角合成以及地图驱动的端到端驾驶等方面的能力缺口。

**🔧 技术方法**

技术手段包括：硬件级同步（全局快门摄像头、7台激光雷达、4D雷达、GNSS/INS）、精确内外参标定、JPEGLI无损压缩、基于Lanelet2的地图标注与评估、图神经网络实现的拓扑预测、深度估计、NeRF/3D Gaussian渲染、端到端驾驶模型（UniAD、DMAD、SSR、Epona）等。

**📊 数据集**

使用的数据集：自主收集的KITScenes Multimodal；对比基准时参考了nuScenes、Waymo Open、Argoverse 2、nuPlan、Nvidia Physical AI AV等公开数据集。

**📈 对比分析**

方法比较采用多种评估指标（AP、AbsRel、δ₁、ADE/FDE、地图驱动安全指标等），结果表明：现有方法在完整HD地图预测、远程深度估计、结构一致的视角合成以及多模态端到端驾驶上均存在显著性能瓶颈，尤其在超过200 m深度、完整拓扑构建与地图一致性方面表现不佳。

**⚠️ 局限性**

局限性包括：缺少动态物体（3D框、轨迹、实例分割）标注；数据量虽高质量但小于部分大规模传感器数据集；端到端评估仅为开放式轨迹预测，未完成闭环驾驶验证；未来计划补充动态标注、扩展数据规模与更丰富的评测。

---

## 92. MetaWorld: Scaling Multi-Agent Video World Model from Single-view Video Data

**arXiv ID:** 2606.02753 | [PDF](https://arxiv.org/pdf/2606.02753v1)

**作者:** Teng Hu `[一作]` (Shanghai Jiao Tong University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 101624 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了一个能够在开放域环境中同步生成多视角身份一致视频的多智能体视频世界模型；

**💡 创新点**

提出了单目世界状态展开（MWSU）以从单视角视频中提取双智能体动力学，Subject-Aware World Generator（SAWG）实现身份感知生成，并通过 World-State Alignment（WSA）实现跨视角一致性；

**🔧 技术方法**

采用扩散变压器（Diffusion Transformers）作为生成骨干，结合 MoGe-2 摄像机轨迹估计、SAM 2 对象跟踪、Depth Anything 深度估计，以及跨分支交叉注意力实现物理同步；

**📊 数据集**

主要使用公开单目摄像机视频（例如社交媒体、体育视频）经过 MWSU 数据引擎处理，生成的训练数据包含摄像机轨迹、主体轨迹、身份图像和3D背景；

**📈 对比分析**

与 MultiWorld 和 VerseCrafter 在 VBench 评测上对比，显示在 Subject Consistency、Background Consistency、Motion Smoothness、Aesthetic Quality、Imaging Quality、Cross-View Consistency（CVC）和 Trajectory Consistency 上均取得显著提升，CVC 最高达 0.8454；

**⚠️ 局限性**

仍依赖深度估计和实例跟踪的精度，对极端遮挡或低光环境表现不佳；跨视角同步虽有效，但在大规模多视角场景下计算成本较高。

---

## 93. Closed-Loop Molecular Design with Calibrated Deference

**arXiv ID:** 2606.02618 | [PDF](https://arxiv.org/pdf/2606.02618v1)

**作者:** Newman Cheng `[一作]` (Microsoft Discovery & Quantum), Jake A. Smith `[通讯]` (Microsoft Research)

**通讯引用:** 1135 | [OpenAlex ID](https://openalex.org/A5030590027)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一种基于LLM的推理代理CLIO，能够在化学设计中持续更新信念图并进行“校准式委婉”，在闭环人机协同实验中为水性有机红氧流电池负极材料设计提供了自适应的假设生成、实验诊断与结构改进，最终实现了高于基线130 mV的还原电位提升，并通过机理诊断改进了电化学可逆性。

**💡 创新点**

创新点在于：①引入持久化的信念状态图，使代理能够跨轮次记录并检索推理历史；②提出“校准式委婉”概念，让代理在实验与计算模型出现冲突时自动降低对模型的信任并重新规划；③实现了完整的设计‑合成‑测试‑学习闭环，体现了真正的代理式科学；④在实验后通过多重假设、实验诊断和结构重设计实现了机制驱动的改进。

**🔧 技术方法**

技术手段包括：大规模语言模型（LLM）驱动的多代理框架（设计、评估、合成规划）；Graphormer结构的还原电位和溶解度预测模型；RDKit的合成可达性评分与RetroChimera逆合成工具；Azure Deep Research的文献检索；以及电化学测量（循环伏安、电位扫描速率、离子交换实验）和光谱电化学等实验技术。

**📊 数据集**

使用的数据集：Graphormer模型基于公开的分子电化学与溶解度数据集（如MoleculeNet等）训练；实验数据来源于本研究自行合成并测定的17种候选分子及其衍生物；对比基线采用文献中报道的硫酸盐衍生苯并茚结构。

**📈 对比分析**

与传统单纯数值优化（如ExLLM）和文献基准相比，CLIO在17个候选中实现了最高130 mV的还原电位提升，并在后续机制修正后将电化学可逆性提升至与基线相当，证明其在多目标、非数值可评估任务中的优越性。实验表明，使用CLIO的闭环循环能显著缩短从设计到验证的周期。

**⚠️ 局限性**

局限性包括：①对底层预测模型（Graphormer）精度的高度依赖，尤其在特定化学空间出现较大偏差时需人工介入；②代理的推理与实验结果关联仍主要基于人类反馈，缺乏完全自动化的闭环；③大规模多代理协作和持久化记忆的计算开销和可扩展性尚未彻底验证；④在更广泛的工业生产与规模化问题（成本、工艺可行性）方面尚未深入探索。

---

## 94. DriftSched: Adaptive QoS-Aware Scheduling under Runtime Token Drift for Multi-Tenant GPU Inference

**arXiv ID:** 2606.02982 | [PDF](https://arxiv.org/pdf/2606.02982v1)

**作者:** Kathiravan Palaniappan `[一作]` `[通讯]` (University of Colorado Colorado Springs), Kathiravan Palaniappan (University of Colorado Colorado Springs)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DriftSched，一种基于运行时 token drift 校正的多租户 LLM 推理 GPU 调度框架；

**💡 创新点**

创新点在于将工作负载 token 预算估计与运行时漂移反馈相结合，使用自适应 bias 机制动态补偿 token drift，并在调度决策中加入 QoS 约束；

**🔧 技术方法**

采用 vLLM 推理引擎、Redis 队列、FastAPI 接口、GPU 监控、指数移动平均（EMA）自适应校正以及 FIFO、Priority、Weighted、SJF、Aging 等传统调度策略；

**📊 数据集**

使用约 1180 条不同类型的 Prompt（短问答、摘要、技术解释、报告生成）作为工作负载数据集；

**📈 对比分析**

与 FIFO、Priority、Weighted、SJF、Aging 五种调度策略对比，实验显示自适应 token drift 补偿可将估算误差平均降低 38.8%（MAE）/40.5%（RMSE），SJF 在尾部延迟上相对 FIFO 减少约 16% 的 P99 延迟，整体系统吞吐与 GPU 利用率保持不变；

**⚠️ 局限性**

局限性包括仅在单 GPU、单模型环境下评估；使用空格分词计数代替精确 token；未考虑多 GPU 或更大模型的扩展；并未验证在线重分类或强化学习调度等更先进策略。

---

## 95. CORE: Conflict-Oriented Reasoning for General Multimodal Manipulation Detection

**arXiv ID:** 2606.03066 | [PDF](https://arxiv.org/pdf/2606.03066v1)

**作者:** Jinjie Shen `[一作]` (Hefei University of Technology), Zhun Zhong `[通讯]` (CSIRO)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了CORE框架，利用多模态大型语言模型结合冲突归因语料CAC，对冲突因素进行细粒度标注并通过跨模态对齐预训练与冲突感知训练，使模型具备人类式冲突推理能力，能够在极少样本甚至零样本情境下快速识别新的多模态伪造内容。

**💡 创新点**

核心创新点在于①将伪造信息的核心冲突作为判别依据，突破对特定伪造模式与大规模标注数据的依赖；②构建细粒度冲突归因语料CAC，为冲突感知提供监督；③在训练中引入跨模态对齐与冲突对比损失，显著提升概念边界清晰度。

**🔧 技术方法**

技术手段包括多模态大型语言模型（Qwen2.5VL-3B、Gemma3-4B）、冲突感知训练（CPT）与跨模态对齐预训练（MBPT）、对比学习损失、视觉问答式辅助任务、LoRA微调等。

**📊 数据集**

使用了自构建的冲突归因语料CAC（约14k样本）以及公开多模态伪造数据集 DGM^4、MDSM、MMFakeBench、NewsCLIPpings、SAMM 与 FineHARD 进行评估。

**📈 对比分析**

在少样本、零样本和全量数据三种场景下，CORE 与同类型 MLLM 以及现有基线（HAMMER、RamDG、FKA‑Owl、AMD 等）进行对比，实验显示 CORE 在所有数据集上均超过最强基线，平均提升约10–15%（少样本/零样本）或 3–4%（全量数据）。

**⚠️ 局限性**

局限性包括：①冲突归因语料 CAC 的标注仍受人工偏差影响；②对某些细粒度或新颖的伪造细节可能捕捉不足；③对动态生成的实时伪造的泛化能力还有待进一步增强。

---

## 96. Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors

**arXiv ID:** 2606.02814 | [PDF](https://arxiv.org/pdf/2606.02814v1)

**作者:** Francisco Valentini `[一作]` (CONICET-Universidad de Buenos Aires), Martin Fajcik `[通讯]` (Brno University of Technology)

**通讯引用:** 105 | [OpenAlex ID](https://openalex.org/A5077658504)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究监督式二编码检索器是否在无查询信息的情况下隐式学习文档级相关性先验，并评估该先验对文档可检索性的影响。

**💡 创新点**

发现监督式检索器会捕捉训练数据中的文档偏好，形成查询无关的相关性先验，从而导致低先验文档在检索时更难被检索到；同时揭示了这种先验在不同模型之间具有一致性。

**🔧 技术方法**

使用冻结的文档嵌入训练逻辑回归或轻量分类器估计先验；通过UMAP、AUC、相关性分析等评估；利用LLM进行解释生成以探究先验背后的文本特征。

**📊 数据集**

评估数据集包括 LoTTE、MIRACL、FEVER、Climate-FEVER、Natural Questions、MSMARCO、DBpedia 等多领域检索基准；还设计了受控的 Toy 实验来验证先验效应。

**📈 对比分析**

与 BM25 等无监督词法检索器对比，展示监督检索器在高先验文档上的可检索性显著高于低先验文档，而 BM25 的相关性不稳定；通过匹配实验进一步验证先验对可检索性的因果作用，整体性能差距可达数个百分点。

**⚠️ 局限性**

仅针对监督式二编码模型，未探讨跨编码器或稀疏检索；实验仅覆盖英文数据，跨语言推广未知；缺乏对下游任务（如 RAG）影响的评估；未提出减缓偏差的具体方法，且可能存在滥用风险。

---

## 97. Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation

**arXiv ID:** 2606.02684 | [PDF](https://arxiv.org/pdf/2606.02684v1)

**作者:** Yuying Li `[一作]` (Tsinghua University), Tao Feng `[通讯]` (Tsinghua University)

**通讯引用:** 6894 | [OpenAlex ID](https://openalex.org/A5100678146)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型的on-policy distillation中提出FiRe-OPD框架，先过滤低质量轨迹后在token层做软加权。

**💡 创新点**

创新点是双粒度联合优化：轨迹层硬过滤结合教师信心，token层软加权融合教师置信度与学生困惑。

**🔧 技术方法**

技术包括PPO风格KL监督、教师-学生对数似然优势、轨迹log概率过滤、token熵权重软加权。

**📊 数据集**

使用DeepMath-103K（难度6）以及数学与代码领域混合数据集，评测多项数学与代码生成基准。

**📈 对比分析**

与标准OPD及ExOPD、TIP、REOPOLD、EOPD、Uni-OPD等基线对比，在强-弱、单教师、多教师三种场景均实现平均精度提升1–4%，尤其在AIME、MinervaMATH和HumanEval+上显著。

**⚠️ 局限性**

局限包括未考虑前缀依赖的token相关性，缺少中间粒度（如步骤级）权重，且仅在预训练后期使用，未探究在线适应。

---

## 98. Can Factual Opinions Be Edited (Manipulated) in Large Language Models?

**arXiv ID:** 2606.03096 | [PDF](https://arxiv.org/pdf/2606.03096v1)

**作者:** Yuanpu Cao `[一作]` (Pennsylvania State University), Jinghui Chen `[通讯]` (Pennsylvania State University)

**通讯引用:** 2586 | [OpenAlex ID](https://openalex.org/A5006335513)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个针对大型语言模型（LLM）的事实性意见编辑与证据对齐的评估基准FOE，并在此基准上评估了多种编辑技术；此外，作者提出了自生成证据对齐（Self-Generated Evidence‑Aligned）方法，使编辑后模型能够在没有显式指令的情况下输出与目标意见一致的证据，从而提升编辑的说服力；

**💡 创新点**

①首次系统化关注并评估事实性意见（即公众人物对社会政治议题的可验证立场）的编辑风险；②设计了包含261名公众人物、19个议题类别、2178条完整意见记录的真实数据集；③提出了针对证据对齐的自生成策略，克服了传统编辑方法在意见-证据一致性上的不足；

**🔧 技术方法**

基于Locate‑then‑Edit的ROME、FT‑M、LoRA、AdaLoRA、DPO等微调/激活编辑技术；自生成证据对齐技术（先粗编辑后自检证据再二次编辑）；评估采用GPT‑4.1作为判定器；

**📊 数据集**

OnTheIssues公开数据源构建的事实性意见数据集（FOE），包含261位公众人物、19个议题类别、2178条记录；

**📈 对比分析**

与8种代表性编辑方法（包括ROME、FT‑M、LoRA、AdaLoRA、DPO、ActAdd、CAA、BiPO）进行对比；在FOE的四个维度（有效性、泛化性、持久性、局部性）上，传统方法大多只能达到1.0左右的Consistency Score，且证据一致性低；自生成证据对齐方法在一致性得分上提升至约1.6-1.9，泛化与局部性表现显著改善；在常见推理任务（CommonsenseQA、GSM8K、FEVER）上的性能保持与原始模型相近，说明对大范围推理无显著影响；

**⚠️ 局限性**

仅在单编辑场景下实验，未覆盖多编辑/多主体的复杂情况；存在过拟合与局部性不足的风险，特别是对相关意见的非目标改变；自生成证据仍可能生成虚假证据，需要进一步验证与防护机制；

---

## 99. Local and Global Contraction Principles for MCMC Mixing

**arXiv ID:** 2606.03033 | [PDF](https://arxiv.org/pdf/2606.03033v1)

**作者:** Alireza Daeijavad `[一作]` (McMaster University), Shahab Asoodeh `[通讯]` (McMaster University)

**通讯引用:** 429 | [OpenAlex ID](https://openalex.org/A5062359324)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种基于hockey‑stick散度的全局与局部收敛系数框架，用以分析投影Langevin Monte Carlo（P‑LMC）与独立Metropolis–Hastings（IMH）的混合时间。

**💡 创新点**

创新点在于将SDPI与Eγ散度结合，提供全局与局部收敛系数，能够在非凸势能、任意批量采样、无界重要权重等场景下给出显式混合时间，并通过尾部分布自适应控制。

**🔧 技术方法**

主要技术是数据处理不等式（SDPI）、Eγ（hockey‑stick）散度的积分表示、Gaussian smoothing导致的全局收敛系数、局部SDPI与拒绝概率关联、以及尾部概率与温暖启动的结合。

**📊 数据集**

无实验数据集；论文为理论分析。

**📈 对比分析**

通过与现有基于矩母推导、Wasserstein/对数Sobolev等方法的比较，显示在无凸势、任意批量和重尾重要权重下可获得更宽泛且显式的收敛率；在经典案例中与已知多项式/指数收敛速率相匹配。

**⚠️ 局限性**

局限在于对IMH仅给出单向Eγ散度控制，无法直接推导所有f散度；对复杂的Metropolis–Hastings如MALA仍需进一步研究非线性SDPI；全局收敛系数在非紧空间可能失效。

---

## 100. Adaptive Latent Agentic Reasoning

**arXiv ID:** 2606.02871 | [PDF](https://arxiv.org/pdf/2606.02871v1)

**作者:** Dongwon Jung `[一作]` (University of California, Davis), Muhao Chen `[通讯]` (University of California, Davis)

**通讯引用:** 5061 | [OpenAlex ID](https://openalex.org/A5102861481)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了自适应潜在代理推理框架，允许LLM代理在大多数决策时使用紧凑的潜在推理，仅在需要更深层思考时才升级为显式链式推理。

**💡 创新点**

创新点在于双模架构（潜在推理 + 显式推理）与自适应模式选择机制，并通过行动锚定自蒸馏（AASD）训练潜在推理以及使用AR‑GRPO强化学习调节模式使用。

**🔧 技术方法**

使用潜在连续思想序列代替文本推理、教师与学生自蒸馏、强化学习奖励模式选择、以及两阶段训练流程。

**📊 数据集**

在搜索任务使用Search‑R1 3B/7B模型与其成功轨迹，在工具使用任务使用Qwen3‑4B‑Thinking与ToolMind数据集。

**📈 对比分析**

与三种推理压缩基线（O1‑Pruner、ThinkPrune、ShorterBetter）比较，ALAR在保持或提升准确率的同时分别在搜索任务减少约43.6%（7B）/39%（3B）与工具使用任务减少约84.6%的生成令牌，获得最佳的准确‑效率 Pareto 性能。

**⚠️ 局限性**

局限性包括依赖教师轨迹导致潜在策略受限、固定潜在块长度与离散模式选择、适用范围仅限于需要外部交互的代理任务，以及潜在推理降低了中间推理的可解释性。

---

## 101. A complete simulation framework for stone degradation on 3D real geometries

**arXiv ID:** 2606.02583 | [PDF](https://arxiv.org/pdf/2606.02583v1)

**作者:** Silvia Preda `[一作]` (University of Insubria), Matteo Semplice `[通讯]` (University of Insubria)

**通讯引用:** 1025 | [OpenAlex ID](https://openalex.org/A5063991804)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出并实现了一套完整的“in silico”石材退化预测工作流程，涵盖从现场摄影测量获取点云、通过水平集方法生成三维几何模型，再利用分裂算法与隐式差分方法对石材硫化过程进行数值模拟。

**💡 创新点**

创新点在于将高精度点云直接转换为水平集描述，能够在非结构化复杂几何上实施偏微分方程求解；同时采用指数积分与隐式差分的分裂技术处理强反应与扩散耦合，显著提升求解稳定性与效率。

**🔧 技术方法**

主要技术包括：闭源摄影测量与SfM‑MVS点云重建、基于梯度驱动的水平集演化、有限差分隐式求解（含幽灵点和零级初始化）以及PETSc库下的多重网格与GMRES预条件求解器。

**📊 数据集**

使用的实验数据集是位于奥斯蒂亚古城的马尔萨神殿祭坛的1 mm分辨率点云（约850万点），并在此基础上构建了多尺度子域（3 M、1.3 M、2.6 M格点）进行模拟。

**📈 对比分析**

通过对时间步长、反应速率与扩散系数的敏感性研究展示了模型对参数的依赖性；计算时间结果表明，在480核的高性能集群上，每步耗时随网格大小线性增长，预条件器设置仅占首几步的少量额外成本，整体求解保持在可接受范围内。

**⚠️ 局限性**

局限性包括：模型仅为简单的硫化反应扩散耦合，未涵盖潮湿、温度、结晶等多因素耦合；水平集演化对点云噪声敏感，需手工调节张力；在大尺度或细节多的对象上仍需采用自适应八叉树或有限元方法以降低计算成本。

---

## 102. Thinking Past the Answer: Evaluating Harmful Overthinking in Large Reasoning Models

**arXiv ID:** 2606.02835 | [PDF](https://arxiv.org/pdf/2606.02835v1)

**作者:** Simone Caldarella `[一作]` (University of Trento), Massimiliano Mancini `[通讯]` (University of Trento)

**通讯引用:** 1374 | [OpenAlex ID](https://openalex.org/A5017971549)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型推理模型（LRM）中的“过度推理”进行系统研究，提出基于推理足够性的前缀轨迹评估协议，区分冗余与有害的过度推理，并量化其在多模态与单语言基准上的表现。

**💡 创新点**

定义最小推理预算为问题难度，区分冗长与有害过度推理；引入前缀级轨迹评估；揭示有害过度推理是可靠性风险，主要由逻辑漂移和视觉误读驱动。

**🔧 技术方法**

使用前缀级答案提取器（Qwen3-4B）与多模态/单语言LRM（如MM‑Eureka、R1‑VL、ThinkLite‑VL、VL‑Rethinker、Qwen3、InternS1）进行实验，评估最佳停止、实际推理长度及早停、DualMind‑VLM等策略。

**📊 数据集**

多模态基准：AI2D、MathVista、MathVision、MathVerse、MMStar、VMCBench；单语言基准：GPQA、AIME2025。

**📈 对比分析**

将实际推理长度（Actual Length）与最佳停止（Optimal Length）及无推理（No‑CoT）对比，发现 Optimal Length 能提升5–21%准确率；早停能显著缩短推理长度但并未降低有害过度推理率，表明两者不等价。

**⚠️ 局限性**

需要地面真值才能确定最佳停止点，实验受限于模型推理步骤的可观测性；早停策略无法解决有害过度推理；模型对视觉误读与逻辑漂移的鲁棒性不足，未提供自动化停止机制。

---

## 103. Fast-dLLM++: Fréchet Profile Decoding for Faster Diffusion LLM Inference

**arXiv ID:** 2606.02955 | [PDF](https://arxiv.org/pdf/2606.02955v1)

**作者:** Siva Rajesh Kasa `[一作]` (Amazon Inc.), Hongdong Li `[通讯]` (Australian National University)

**通讯引用:** 18596 | [OpenAlex ID](https://openalex.org/A5101819061)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 diffusion 大模型推理中提出一种训练无关、可直接替换的并行解码器 Fast-dLLM++，通过使用 Fréchet‑profile 选择准则来实现更高效的并行提交。

**💡 创新点**

创新点在于用完整的排序置信度分布（而非仅取最弱置信度）构造 Fréchet 下界与上界，形成分布无关的置信度证明；该规则在置信度异质时可获得额外“异质性奖金”，从而比原 Fast‑dLLM 的 factor 规则在相同参数下更具攻击性且安全。

**🔧 技术方法**

采用 masked diffusion 语言模型、Fréchet‑profile 选择规则、无额外模型或缓存修改、简单的排序和前缀求和实现；实验使用 LLaDA‑8B‑Instruct、Dream‑v0‑Base‑7B 等已训练好的 diffusion LLM；实现为 drop‑in 替换。

**📊 数据集**

在 GSM8K、MATH、HumanEval、MBPP 等文本生成基准上测试；并在 MathVista、MathVerse 等多模态推理任务中验证通用性；使用三种缓存模式（无缓存、prefix cache、DualCache）进行对比。

**📈 对比分析**

与 Fast‑dLLM 的阈值（threshold）和因子（factor）规则对比，Fréchet‑profile 在大多数设定下实现平均 1.36× 的吞吐量提升、NFE（神经前向步骤）降低 29.2%，准确率差距仅 0.48 分；在多模态任务中仍保持 9.9–11.6× 的速度提升，且准确率变化不大。

**⚠️ 局限性**

局限性包括：仅对贪婪 argmax 解码有效，无法直接用于温度/核采样；对置信度校准敏感，若模型过度自信或选定位置高度耦合可能过度提交；未利用其它分布信息（熵、top‑k 间距、softmax 形状）；缺乏对依赖结构的显式估计，导致在语法/语义强耦合任务中性能下降。

---

## 104. The Epi-LLM Framework: probing LLM behavioral priors through epidemiological agent-based models

**arXiv ID:** 2606.02867 | [PDF](https://arxiv.org/pdf/2606.02867v1)

**作者:** Petra Ferenz `[一作]`, Jasmina Panovska-Griffiths `[通讯]` (University of Oxford)

**通讯引用:** 5212 | [OpenAlex ID](https://openalex.org/A5042672436)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出并实现了Epi-LLM框架，将大语言模型（LLM）嵌入基于代理的流行病模型（ABM）中，并通过真实的epigame健康信念问卷对代理行为进行参数化；

**💡 创新点**

创新点在于首次将LLM与ABM及真实epigame数据融合，系统比较四种不同架构LLM对疫情动态的影响，并探索LLM在模拟人类健康行为中的可行性；

**🔧 技术方法**

技术包括Starsim ABM平台、OpenRouter API调用多种LLM（DeepSeek V3、Llama 3 70B、Nemotron 120B、GPT‑OSS 120B）、动态Bluetooth接触网络、健康信念模型转换为代理属性，以及GLM/Poisson回归分析；

**📊 数据集**

使用的数据集为AUIB epigame收集的健康信念问卷（Health Belief Model指标）和蓝牙传感得到的接触网络（度分布与接触时长），并基于这些数据生成代理参数和网络结构；

**📈 对比分析**

比较方法通过对照无干预SEIR基线，评估不同LLM架构在隔离率、疫情峰值与累计感染方面的差异；GLM模型表明健康信念对隔离行为的解释力（pseudo‑R²≈0.055）与人类试验相近，且LLM架构对疫情曲线有显著影响；

**⚠️ 局限性**

局限性包括：仅使用单次随机种子导致难以区分架构效应与随机噪声；地理标签未能产生文化差异；隔离成本与健康信念交互效应再现有限；奖励结构实验仅在一种架构下进行；网络结构简化缺乏长程社群效应。

---

## 105. Evaluating Transformer and LSTM Frameworks for Prediction in Ungauged Basins

**arXiv ID:** 2606.02791 | [PDF](https://arxiv.org/pdf/2606.02791v1)

**作者:** Taye Akinrele `[一作]` (University of Alabama), Shahram Rahimi `[通讯]` (University of Alabama)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较了 LSTM 与 encoder-only Transformer 在无监测流域上游流量推断中的表现，使用 NOAA National Water Model 的回溯模拟数据，并探究加入下游流量信息对预测精度的影响。

**💡 创新点**

通过将下游流量视为网络级约束，检验不同序列建模架构在信息有限情况下的诱导偏差，并揭示 LSTM 在此任务上优于 Transformer 的原因。

**🔧 技术方法**

采用 LSTM、Encoder-only Transformer、Informer、CNN-1D 等深度学习序列模型，配合 AdamW 优化、NSE 损失、早停等训练技巧。

**📊 数据集**

使用 NOAA National Water Model v3.0 1979–2023 年小时级模拟数据，匹配 671 个 CAMELS 流域，包含气象强迫、静态流域属性和下游流量。

**📈 对比分析**

在上游仅输入与上游+下游两种配置下分别训练并评估中位数 NNSE、KGE、Pearson‑r、RMSE 等指标；结果显示 LSTM 在两种配置下略优，且加入下游信息后所有模型的 NNSE 提升约 60%，显著提升预测性能。

**⚠️ 局限性**

局限性包括仅基于模拟数据而未验证观测流量；只考虑单段上游–下游关系，未覆盖更复杂网络；并且模型超参数未进行全面搜索，可能影响最佳性能。

---

## 106. Assessing Region-Level EEG Contributions to Cognitive Workload Prediction

**arXiv ID:** 2606.02598 | [PDF](https://arxiv.org/pdf/2606.02598v1)

**作者:** Jacob Wong `[一作]` (Ngee Ann Polytechnic), U-Xuan Tan `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 3136 | [OpenAlex ID](https://openalex.org/A5023142347)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了 EEG 头皮上不同解剖学区域对认知工作负荷预测的贡献，并构建了一个统一的区域级评估框架。

**💡 创新点**

创新点在于：①提出跨数据集、跨任务、跨模型无偏的区域重要性评估方法；②发现前额与前额-中心区域在多任务、多模型、多评估协议下始终表现最佳，显著优于全标帽；③实现了硬件效率提升与可解释性增强。

**🔧 技术方法**

使用了 EEG 预处理（0.5–45 Hz 滤波、频段分解）、多维特征提取（功率、比值、统计量、熵）、解剖学区域分组、三种特征选择（ANOVA、MRMR、随机森林）以及多模型（Logistic、SVM、随机森林、梯度提升、线性回归、SVR、随机森林回归、GB回归）训练，配合三折交叉验证与留 N 受试者评估，最终通过排名聚合得到区域重要性。

**📊 数据集**

使用了五个公开 EEG 数据集：FDE‑HTC、FDE‑Nback、MOCAS、WAUC、HCI‑SENSE‑42，覆盖时间压力、工作记忆、视觉监控、多任务和自然交互等不同工作负荷场景。

**📈 对比分析**

通过与全标帽（全脑）模型进行性能降解比较，采用排名（rank）指标评估区域模型；结果显示前额、前额-扩展区域在混合受试者与主观独立评估中均排名低于全标帽，说明它们在保持或提升预测准确性的同时大幅减少电极数量，体现出更高的硬件效率与鲁棒性。

**⚠️ 局限性**

局限性在于：①仅使用 NASA‑TLX 主观评分，受标注噪声与个体差异影响；②任务范围局限于执行控制、工作记忆、持续注意、多任务与自然交互，未检验其他认知范式；③未结合多模态信号或实时自适应，可进一步提升稳健性与应用广度。

---

## 107. What You Approve Is What Executes: Consent Integrity for Black-Box LLM Agents

**arXiv ID:** 2606.02668 | [PDF](https://arxiv.org/pdf/2606.02668v1)

**作者:** Xiaoqi Weng `[一作]` `[通讯]` (Bournemouth University), Xiaoqi Weng (Bournemouth University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了黑盒LLM代理的同意完整性（Consent Integrity）机制，保证用户批准的内容与代理实际执行的操作完全一致。

**💡 创新点**

首次将WYSIWYS与受信路径概念迁移到LLM代理审批环节，并针对两大挑战（渲染方为对手、底层事件难以解析）设计了解决方案。

**🔧 技术方法**

采用可信中介、动作解析器、引用内容检查、写入/构建溯源以及哈希绑定等技术；实现了约500行无依赖Python原型。

**📊 数据集**

评估使用了第三方攻击数据集GTFOBins（1330条）与正常使用数据集tldr-pages（28798条）进行验证。

**📈 对比分析**

与传统叙述式审批比较，原型在GTFOBins中标记90%攻击命令为危险/不可检查，只有10%被静默通过；在tldr-pages中提示率达95.9%，不可检查率87%，表明在安全性与用户疲劳之间的权衡。

**⚠️ 局限性**

限制包括：仍假设总介入与受信路径；解析器不完整导致“不可检查”标签增多；对信任列表的依赖导致安全与易用性折衷；未实现跨平台OS级总介入与人机理解评估。

---

## 108. Hanger Reflex Based Driving Assistance for Drivers with Peripheral Visual Field Defects

**arXiv ID:** 2606.03020 | [PDF](https://arxiv.org/pdf/2606.03020v1)

**作者:** Hailong Liu `[一作]` (Nara Institute of Science and Technology), Takahiro Wada `[通讯]` (Nara Institute of Science and Technology)

**通讯引用:** 10673 | [OpenAlex ID](https://openalex.org/A5048756005)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在驾驶模拟器中，使用悬挂反射刺激（HRC）帮助佩戴正常视野但模拟周边视野缺损的驾驶员，在行人过街情境下转向并注视潜在危险。

**💡 创新点**

首次将悬挂反射机理引入驾驶辅助领域，实现非视觉、非听觉的物理引导，提升驾驶员对周边风险的预先注意。

**🔧 技术方法**

采用气囊驱动的头部刺激头盔、IMU测量头部转向、Tobii Pro Nano眼动仪进行视线追踪，以及CARLA 0.9.13仿真平台。

**📊 数据集**

15名年轻驾驶员使用基于眼动的视觉遮挡模拟周边视野缺损，参与8次行人交叉事件，形成实验数据集。

**📈 对比分析**

与无刺激条件相比，HRC显著提升头部转向角度和注视时长，碰撞率下降约36%（从36.7%降至16.7%，odds ratio≈0.35），但仅呈边缘显著水平。

**⚠️ 局限性**

样本量有限、仅模拟视野缺损、未检验与其他辅助方式的对比、缺乏真实视野缺损驾驶者验证。

---

## 109. Regime-Arrival Uncertainty in Generalization Bounds under Distribution Shift

**arXiv ID:** 2606.02657 | [PDF](https://arxiv.org/pdf/2606.02657v1)

**作者:** Prince Poudel `[一作]` `[通讯]`, Prince Poudel

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一个框架，用于研究在马尔可夫切换环境中，训练和部署分布不匹配所带来的额外风险，特别是在平静状态与危机状态的比例不同的情况下。

**💡 创新点**

创新点在于精确分解未来风险，将其与训练和部署环境的组成差异联系起来，并引入了一个基于有效样本量的惩罚项来量化这种不匹配的影响。

**🔧 技术方法**

使用了马尔可夫过程模型，结合了领域适应、依赖学习理论和切换模型的思想，分析了在β-混合数据下的风险界限。

**📊 数据集**

使用了合成数据和25年的全球股票指数数据进行实证验证。

**📈 对比分析**

与传统的训练数据估计方法相比，使用实际未来危机比例计算的惩罚项与实际训练到部署的差距具有显著的相关性（Spearman ρ = 0.729），而仅使用训练数据的惩罚项则没有显著相关性（ρ = 0.084）。

**⚠️ 局限性**

局限性在于假设的两状态马尔可夫过程可能无法捕捉现实世界中更复杂的状态动态，且该框架主要用于诊断而非预测，无法保证在所有情况下修改训练组成会恢复丢失的性能。

---

## 110. Rethinking Molecular Text Representations for LLMs: An Empirical Study

**arXiv ID:** 2606.03057 | [PDF](https://arxiv.org/pdf/2606.03057v1)

**作者:** Arun Raja `[一作]` (University Of Oxford), Kian Ming A. Chai `[通讯]` (Dso National Laboratories)

**通讯引用:** 4276 | [OpenAlex ID](https://openalex.org/A5069560003)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了MolRepBench基准，系统评估了9种分子文本表示在8个化学任务上对16种大型语言模型的影响。

**💡 创新点**

创新点在于揭示不同表示会导致显著性能差异，发现CML/MolJSON在结构任务上优于SMILES，IUPAC在语义任务上领先，并首次提出LLM‑as‑a‑judge来定性分析生成错误与任务感知的表示路由。

**🔧 技术方法**

技术方法包括多家LLM（Qwen3、Phi‑4、OLMo、ChemDFM、Ether0、GPT‑5.4‑mini等）的多任务推理与生成；对tokenization、线性探测和注意力进行机制分析；以及使用LLM判定器进行错误分类。

**📊 数据集**

使用的公开数据集包括ChEBI‑20和ZINC250K，用于构建原子计数、功能团识别、属性估计、分子检索、同分异构体辨别、共轭体识别、质子化状态识别和标题到分子生成等任务。

**📈 对比分析**

采用配对自助法（10,000次重采样）对每种表示的分数进行统计比较，评估指标涵盖精确匹配、宏F1、Spearman相关、Tanimoto相似度等；结果显示CML/MolJSON在结构任务中占优，IUPAC在语义和生成任务中表现最佳，SMILES虽普遍但多为次优。

**⚠️ 局限性**

局限性包括仅覆盖八类任务且受限于计算与API预算，未涉及多模态输入，且仅考察文本表示，无法全面评估其他潜在的分子表示方法。

---

## 111. SCOPE: Real-Time Natural Language Camera Agent at the Edge

**arXiv ID:** 2606.02951 | [PDF](https://arxiv.org/pdf/2606.02951v1)

**作者:** Nikolaj Hindsbo `[一作]` (Armada AI), Pragyana Mishra `[通讯]` (Armada AI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SCOPE框架，实现了在Blender仿真与物理PTZ摄像头上可执行的自然语言驱动闭环控制与评测基准，并发布了536任务的评测集。

**💡 创新点**

创新点在于：①将相同的PTZ动作空间与感知循环映射到模拟与实机，实现高保真sim‑to‑real转移；②设计可执行的闭环评测基准与LLM‑as‑Judge评估方法；③系统化比较19种规划‑感知组合，揭示Mixture‑of‑Experts与量化在边缘部署中的优势。

**🔧 技术方法**

技术栈包括：小型语言模型（Qwen3系列）做规划器；轻量VLM（Moondream系列）做感知工具；OpenAI兼容JSON工具调用；Blender仿真环境；Mixture‑of‑Experts与量化模型；LLM作为评判器。

**📊 数据集**

使用自建的四个公开Blender城市场景与人工标注的Preset，结合物理PTZ日志生成536个结构化任务，未使用公开任务数据集。

**📈 对比分析**

通过LLM‑as‑Judge对每个模型组合执行完整任务进行评分，衡量准确率、延迟与错误模式。最佳组合（Moondream3 + Qwen3‑30B‑A3B）约73.8%成功率，平均延迟1.5–2 s，Mixture‑of‑Experts与量化在保持相近精度的同时显著降低显存与计算负载。

**⚠️ 局限性**

局限性包括：①任务提示过于结构化，缺乏真实场景中的模糊性和歧义；②延迟测量受服务器配置和资源争用影响；③未评估闭源API模型；④未测试多摄像头/多智能体协同；⑤感知仍受轻量VLM限制，计数与OCR仍有较高误差。

---

## 112. Hybrid Adaptive Kalman Filtering for Data-Efficient Joint Tracking and Classification

**arXiv ID:** 2606.02767 | [PDF](https://arxiv.org/pdf/2606.02767v1)

**作者:** Jiho Lee `[一作]` (Charles Stark Draper Laboratory, Inc.), Rebecca Russell `[通讯]` (Charles Stark Draper Laboratory, Inc.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种自监督的混合自适应卡尔曼滤波器（HAKF），通过神经网络学习结构化的动力学修正和过程噪声协方差，仅使用测量数据即可完成状态估计和模型分类。

**💡 创新点**

创新点在于：①将传统卡尔曼滤波与神经网络自适应校正相结合，保持概率结构；②利用创新似然通过广义贝叶斯推断实现模型分类；③在低数据环境下实现高效的数据驱动自适应。

**🔧 技术方法**

采用了卡尔曼滤波、MLP神经网络进行结构化校正、创新负对数似然最大似然学习、Cholesky LDL分解保证协方差正定、Joseph形式更新、Adam优化器、广义贝叶斯推断（温度化似然）。

**📊 数据集**

使用了真实的DPJAIT室内三架无人机的3D轨迹数据集，以及Microsoft AirSim仿真生成的两架无人机（Flamewheel、Hexacopter）约10万条轨迹样本。

**📈 对比分析**

与传统固定参数卡尔曼滤波和纯监督RNN基准进行对比；在DPJAIT和AirSim上，HAKF的RMSE显著降低，NEES/NIS统计一致性接近95%；在低数据下分类准确率优于RF和RNN，高数据下与RF相当。

**⚠️ 局限性**

局限性在于：校正结构受限于线性或有限维度，无法捕捉复杂非线性动力学；仍依赖白噪声假设；当训练数据极大时，纯数据驱动模型可能更优。

---

## 113. KForge: LLM-Driven Cross-Platform Kernel Generation for AI Accelerators

**arXiv ID:** 2606.02963 | [PDF](https://arxiv.org/pdf/2606.02963v1)

**作者:** Taras Sereda `[一作]` (Gimlet Labs Inc.), Zain Asgar `[通讯]` (Gimlet Labs Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出KForge，一个基于LLM的跨平台内核生成与优化框架，采用生成代理与性能分析代理的交互迭代，支持四大厂商（NVIDIA、AMD、Intel、Apple）与六种编程模型（CUDA、Triton、CuTe、HIP、SYCL、Metal）；

**💡 创新点**

创新点在于将内核生成与性能反馈拆分为两个协同工作代理，利用迭代细化、跨平台翻译与多模态性能反馈，实现跨硬件自动化高性能内核合成；

**🔧 技术方法**

技术包括大型语言模型（如Claude Opus 4.6）、基于提示的代码合成、编译与正确性回馈、图形与文本型性能分析、动态提示编辑、结构化实验日志、可插拔的测量钩子；

**📊 数据集**

使用的数据集/基准包括：gpt-oss-20b MoE 推理工作负载（TensorRT-LLM基准），以及KernelBench Level 2（37个GEMM+尾部运算问题）用于Intel Arc B580；

**📈 对比分析**

比较方法为在NVIDIA B200上与TensorRT-LLM基准对比，获得2.12%吞吐量提升；在Intel Arc B580上与PyTorch eager及最快的Triton基线对比，得到5.13×几何平均加速；

**⚠️ 局限性**

局限性包括：依赖源级参考实现，难以处理无源代码的闭源二进制；仅支持PyTorch，缺乏对JAX等框架的覆盖；验证仅通过数值一致性，未进行形式化或差分测试；低级ISA（如PTX）尚未支持；单核优化可能不转化为整体推理加速。

---

## 114. RESCAST-100K: A Comprehensive Dataset for Cross-Domain Residential Load and Indoor Temperature Forecasting

**arXiv ID:** 2606.02852 | [PDF](https://arxiv.org/pdf/2606.02852v1)

**作者:** Jainam Dhruva `[一作]` (University of Kentucky), Simone Silvestri `[通讯]` (University of Kentucky)

**通讯引用:** 1588 | [OpenAlex ID](https://openalex.org/A5009742150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `79276348-11e0-48e3-84bc-7ec231d0171c` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了RESCAST-100K，一个包含约10万栋美国住宅15分钟分辨率时间序列的基准数据集，涵盖总负荷、HVAC负荷、室内温度以及天气、设定点和40+静态建筑特征，并支持多域分割与真实数据对接；

**💡 创新点**

创新点在于构建大规模、跨域可配置的住宅预测基准，统一模拟与真实数据schema，提供可对齐的跨域拆分接口，支持零射、迁移学习、域适应和 sim-to-real 评估；

**🔧 技术方法**

使用了基于EnergyPlus的高保真物理模拟、ResStock建筑统计模型、Python+Parquet数据管道，以及多种时序模型（LSTM、Transformer、TimeXer、TSMixer、PatchTST）和稳健版本；

**📊 数据集**

数据集包含约10万栋模拟住宅和5个公开真实住宅数据集（ECOBEE、HEAPO、IDEAL、NEST、REFIT）；

**📈 对比分析**

通过对比六类模型的点预测(NRMSE)与概率预测(CRPS)，发现Mixer/跨注意力模型在跨域零射和真实数据评估中显著优于RNN和标准Transformer，尤其在负荷预测任务中表现最优；

**⚠️ 局限性**

局限包括：仍存在 sim-to-real 差距，模拟未完全捕捉占用随机性和电器异质性；真实数据地理覆盖有限且缺失特征多；缺乏国际建筑样本，需进一步扩展。

---

## 115. Online K-d tree for approximate neighborhood search in data streams

**arXiv ID:** 2606.02752 | [PDF](https://arxiv.org/pdf/2606.02752v1)

**作者:** Eduardo V. L. Barboza `[一作]` (École de Technologie Supérieure), Rafael M. O. Cruz `[通讯]` (École de Technologie Supérieure)

**通讯引用:** 2157 | [OpenAlex ID](https://openalex.org/A5019553116)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种适用于数据流的在线K‑d树，实现动态插入、懒删除和滑动窗口更新，并支持Canberra距离的近似邻域搜索；

**💡 创新点**

首次将Canberra距离融入K‑d树分裂准则，提出基于树尺寸γ和删除比例β的重建策略，实现对高维流数据的高效近似kNN；

**🔧 技术方法**

使用K‑d树空间划分、近似kNN搜索、懒删除与重建、Canberra距离分裂准则，实验基于Java MOA框架实现；

**📊 数据集**

实验覆盖六个真实流数据集（Electricity、NOAA、Insects‑AB、Covertype、Gas Sensor、Dry Bean）和三个合成漂移数据集（SEA、Sine、RandomRBF）；

**📈 对比分析**

与暴力kNN对比，评估分类准确率、邻域精度和实例/秒处理速度，在线K‑d树在绝大多数数据集上准确率仅略低，邻域精度高，处理速度提升6–16倍，尤其在低维数据上显著；

**⚠️ 局限性**

在高维或空间分布不利的数据集上剪枝效果差，懒删除导致内存堆积，概念漂移时旧概念难以被搜索到，需要进一步自适应重建机制。

---

## 116. An Exploration of Collision-based Enemy Morphology Generation

**arXiv ID:** 2606.02832 | [PDF](https://arxiv.org/pdf/2606.02832v1)

**作者:** Johor Jara Gonzalez `[一作]` (University of Alberta), Matthew Guzdial `[通讯]` (University of Alberta)

**通讯引用:** 737 | [OpenAlex ID](https://openalex.org/A5000736647)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究基于玩家与敌人碰撞信息的二维平台游戏敌人形态生成，旨在设计能够通过特定增强动作击败但不易被基础动作击败的敌人；

**💡 创新点**

创新点在于提出三种基于交互数据的形态生成方法（强化学习+神经网络、A*搜索+生成规则、RL生成规则），并将形态生成与玩家动作相关联；

**🔧 技术方法**

采用了强化学习（PPO）、A*搜索、神经网络（前馈网络）以及遗传算法作为基线；

**📊 数据集**

使用了自建的4x4格子碰撞数据集（包含弱、致命、空格子），通过多轮玩家与敌人交互记录生成；

**📈 对比分析**

与遗传算法基线比较，三种交互驱动方法在门控成功率（SRAA）上与遗传算法相当或更好，A*规则法计算成本最低；

**⚠️ 局限性**

局限包括对大规模或连续空间形态的适应性不足、强化学习方法计算成本高、对速度加速等特殊机制的门控效果差。

---

## 117. An improved PINN framework integrating localized collocation scheme and PIKF

**arXiv ID:** 2606.02585 | [PDF](https://arxiv.org/pdf/2606.02585v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 118. LAANN: I/O-Aware Look-Ahead Search for Disk-Based Approximate Nearest Neighbor Search

**arXiv ID:** 2606.02784 | [PDF](https://arxiv.org/pdf/2606.02784v1)

**作者:** Dingyi Kang `[一作]` (University of Texas at Dallas), Bingzhe Li `[通讯]` (University of Texas at Dallas)

**通讯引用:** 958 | [OpenAlex ID](https://openalex.org/A5048972267)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了 LAANN，一种面向 I/O 的磁盘基近似最近邻搜索系统，结合了基于阶段的 Look‑Ahead 搜索、优先级 I/O‑CPU 管道以及轻量级内存图索引，实现了在磁盘 I/O 与 CPU 计算之间的协同优化。

**💡 创新点**

创新点包括：① 发现并利用 CPU 计算与磁盘 I/O 的紧耦合关系，突破传统只从 I/O 角度优化的做法；② 提出 Look‑Ahead 搜索，根据搜索阶段动态切换内存优先与普通模式，并采用动态 Beam‑Width 以降低 I/O 迟延；③ 设计优先级 I/O‑CPU 管道和溢出候选池，在 I/O 等待期间按对下一轮 I/O 决策的相关性分级执行 CPU 任务；④ 用页面对齐的稀疏图与基于页质心的 Vamana 轻量索引作为内存预热，直接种子磁盘图搜索，减少搜索路径长度。

**🔧 技术方法**

使用技术包括：异步 SSD I/O、基于压缩向量的近似距离、页面对齐图结构、Vamana 轻量内存图索引、动态 Beam‑Width 与 Persistence 检查、优先级 CPU 任务调度与候选池溢出管理、页面缓存频繁访问节点、并行线程调度与 I/O 预取。

**📊 数据集**

实验数据集：SIFT100M、SPACEV100M、DEEP100M（100M 级别）以及 SIFT1B、SPACEV1B（1B 级别）。

**📈 对比分析**

与 DiskANN、Starling、MARGO、PipeANN、PageANN 五个最先进磁盘基 ANNS 系统在同一硬件（Intel i9‑13900、128 GB DDR5、1 TB NVMe）上进行对比，指标为 Recall@10、平均查询延迟（ms）与吞吐量（QPS）以及平均 I/O 次数。LAANN 在 Recall@10=0.9 时吞吐量提升 1.41–4.66 倍、延迟降低 29–79%，平均 I/O 下降 1.59–6.34 倍，整体性能显著优于基线。

**⚠️ 局限性**

局限性：1) 仍以 SSD 为主，I/O 仍是瓶颈；2) 需要足够内存以存放压缩向量、页缓存和轻量索引，内存预算有限时效果下降；3) 动态 Beam‑Width 与 Persistence 检查参数需要手工调优；4) 仅评估单机场景，未验证分布式扩展；5) 对特定硬件（如 GPU、PIM）适配性未深入研究。

---

## 119. Any2Poster: Any-Source Poster Generation Across Modalities and Domains

**arXiv ID:** 2606.02915 | [PDF](https://arxiv.org/pdf/2606.02915v1)

**作者:** Amogh Vinaykumar `[一作]` (Flower Mound High School), Shilong Liu `[通讯]` (Princeton University)

**通讯引用:** 17 | [OpenAlex ID](https://openalex.org/A5066688247)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Any2Poster 基准，评估任意来源的单页海报生成；并实现一个端到端参考系统 (Any2Poster Agent)。

**💡 创新点**

①跨八种输入模态（PDF、URL、PPTX、DOCX、Markdown、LaTeX、Notebook、Video）和五大内容域的统一评估框架；②基于 Quiz 的信息保真评测 BenchQuiz 与 VLM-as-Judge 的视觉质量评测相结合；③利用代码（HTML/CSS）渲染与 VLM 反馈循环实现局部视觉修复。

**🔧 技术方法**

统一解析（多模态转换为共享结构），内容自适应规划，代码渲染与 VLM 辅助修复；使用 LLM、VLM（如 LLaVA、ChatGPT 版本）进行问答与视觉判定。

**📊 数据集**

约 160 篇源文档（PDF、URL、PPTX 等）构成的 Any2Poster 数据集，覆盖研究、新闻、教育、商业、小说等五大领域。

**📈 对比分析**

与 GPT‑4o、Gemini‑2.5 Flash、GPT‑5、Paper2Poster 等基线对比；在任意来源评测中平均准确率 87.25%，在 PaperQuiz‑style 评测中整体准确率提升至 72.58%，显著优于先前代理；VLM‑as‑Judge 总分从 3.69 提升至 4.03，逻辑流向显著改善。

**⚠️ 局限性**

仍需人工验证解析正确性；对极长或结构极弱的输入（如 PPTX、视频）表现相对弱；依赖现有 LLM/VLM，可能产生信息或视觉方面的幻觉；评测只关注信息保真与视觉质量，未覆盖多语言、多风格的广泛场景。

---

## 120. Constitutional On-Policy Safe Distillation

**arXiv ID:** 2606.03089 | [PDF](https://arxiv.org/pdf/2606.03089v1)

**作者:** Ming Wen `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 25063 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `a4b10f5d-130b-4e77-9367-6469ec621899` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在安全对齐任务中 On‑Policy Self‑Distillation（OPSD）的失效机制，并提出了新的两阶段框架 COPSD（Cross‑SFT 冷启动 + 省教师 on‑policy distillation）来解决该问题。

**💡 创新点**

创新点在于：①将安全约束视为能量基模型，揭示了“几何泄漏”导致的表达性崩塌；②设计了 Cross‑SFT 预训练来分离安全与表达性流形；③通过 COPSD 实现了在保持推理能力的同时显著提升安全性，克服了传统 OPSD 的安全税（safety tax）。

**🔧 技术方法**

技术手段包括：On‑Policy Self‑Distillation、反向 KL 以及 Forward KL 正则化、能量基分布建模、自然梯度分析、Cross‑SFT (SFT + 生成自我样本)、GRPO 与 Safe‑RLHF-V 等对照算法，以及密集的 token‑级奖励与优势估计。

**📊 数据集**

使用的数据集：内部 9K 的多目标安全/通用数据集；12 个公开安全与通用基准（BeaverTails‑V、SPA‑VL、SIUO、MSS‑Bench、VLSBench、VLGuard、MathVista、LLaVA‑Wilder、MMVet、ScienceQA、VQAv2、GQA），以及 Qwen3‑VL‑4B 与 Qwen2.5‑VL‑7B 两大模型。

**📈 对比分析**

与 SFT、OPD、GRPO、Safe‑RLHF‑V 等基线对比，COPSD 在 12 项安全/通用基准上实现了更高的安全率和更低的安全税，帮助度保持不降甚至提升；在推理任务（MathVista、LLaVA‑Wilder）上的性能下降仅约 2%，而 GRPO 等方法下降超过 15%。

**⚠️ 局限性**

局限性包括：①需要精细调节 Cross‑SFT 训练周期，过长会导致过拟合和安全性能退化；②几何泄漏分析依赖于简化假设，实际模型可能存在更复杂的耦合；③仍有一定安全税（尤其在情境安全基准上），且目前仅在 Qwen 系列视觉语言模型上验证，缺乏跨模型与跨任务的广泛验证。

---

## 121. QUIVER: Quantum-Informed Views for Enhanced Representations in Large ML Models

**arXiv ID:** 2606.02785 | [PDF](https://arxiv.org/pdf/2606.02785v1)

**作者:** Aritra Bal `[一作]` (Karlsruhe Institute Of Technology), Michael Spannowsky `[通讯]` (Karlsruhe Institute Of Technology)

**通讯引用:** 6429 | [OpenAlex ID](https://openalex.org/A5082855661)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 Quiver 架构，将变分量子电路得到的量子 Fisher 信息矩阵（QFIM）视作补充模态，与经典特征融合，提升大规模机器学习模型在不同任务上的表现。

**💡 创新点**

创新点在于：①利用 QFIM 提取的几何信息显式捕获高阶、非局部相关性；②构建与 Transformer、GNN 兼容的注入机制（交叉注意力或边状态缩放）；③完全基于经典模拟，避免对量子硬件的依赖。

**🔧 技术方法**

技术手段包括：变分量子电路（1P1Q 量子编码用于 jet，2A2Q 编码用于分子）、PennyLane 量子模拟、量子 Fisher 信息矩阵计算、交叉注意力/多模态注入、Particle Transformer、DimeNet++ 等主流模型。

**📊 数据集**

使用的数据集：JetClass（高能物理 jet flavor 分析）和 QM9（分子属性回归）。

**📈 对比分析**

与相同或更大参数的经典基线（Particle Transformer、DimeNet++）在相同训练设置下对比，评估指标为 jet 分类的 AUC 与 QCD 背景拒绝率 1/ϵ_B，以及 QM9 的 MAE。实验显示：JetClass 上 Quiver-Particle Transformer 的 AUC 从 0.9783 提升至 0.9807，1/ϵ_B 提升至 2401；QM9 上 𝒬DimeNet++ 的 MAE 从 72.42 ± 1.52 meV 降至 67.92 ± 1.98 meV，提升约 6.2%。

**⚠️ 局限性**

主要限制：①量子模拟被限制在最多 10 qubit，导致仅使用 jet 的前 10 颗粒子和分子最多 10 个原子；②因数据截断，模型在 JetClass 上无法利用全部 150 个粒子，QM9 上无法利用全部氢原子，影响绝对性能；③未实现全局量子+经典联合训练，仅做了两阶段的顺序优化。

---

## 122. Toward a Modular Architecture for Embedded AI Agent Systems at the Edge

**arXiv ID:** 2606.02862 | [PDF](https://arxiv.org/pdf/2606.02862v1)

**作者:** Marcus Rüb `[一作]` (Foresthub.Ai), Michael Gerhards `[通讯]` (Deloitte Consulting)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出面向嵌入式设备的模块化代理系统参考架构，整合传统实时控制与小型语言模型驱动的智能代理。

**💡 创新点**

创新点在于分层解耦本地 SLM 与云端 LLM，设计跨层治理层实现可观测、安全与政策统一管理，并通过统一接口实现多硬件层的协同。

**🔧 技术方法**

采用 TinyML、量化小型语言模型、RTOS/micro‑ROS、MQTT/CoAP、Model Context Protocol（MCP）以及 NPU 等技术。

**📊 数据集**

无实际数据集，本文为概念性设计与架构评估。

**📈 对比分析**

通过理论分析比较延迟、能耗、内存占用与隐私等维度，未给出具体数值，只给出不同硬件层（Flavor A vs Flavor B）的优缺点对照。

**⚠️ 局限性**

局限在于缺乏端到端的实测基准，语义桥接机制尚未标准化，Flavor A 的硬件成本与内存壁垒高，Flavor B 对网络连通性高度依赖。

---

## 123. Geometry-Aware Tabular Diffusion

**arXiv ID:** 2606.02607 | [PDF](https://arxiv.org/pdf/2606.02607v1)

**作者:** David Turtora Zagardo `[一作]` `[通讯]` (Independent Researcher), David Turtora Zagardo (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了如何在表格数据生成中引入几何监督，提出 Geometry‑Aware Tabular Diffusion (GATD)，并将其作为可在多种扩散解码器（MLP、GNN、Transformer）中直接插拔的模块。

**💡 创新点**

创新点在于将列间的关系显式编码为对角角度和长度特征，并将其作为输入和辅助预测目标进行监督；这种显式几何监督被证明能显著提升生成质量，并可在不同网络架构间迁移，展示了其可移植性。

**🔧 技术方法**

技术实现基于 TabDiff 的扩散框架，加入对角角度 θ_{ij}=arctan(v_j−v_i) 与对角长度 ℓ_{ij}=½log(1+(v_j−v_i)^2) 两组特征；同时在网络中增加对应的预测头，并加入角度、长度和一致性三项辅助损失。实验中分别在 MLP、GNN（Laplace 位置编码）和列级 Transformer 解码器上插拔该模块。

**📊 数据集**

使用了 10 个公开表格数据集（Adult、Default、Diabetes、Magic、Shoppers、Beijing、Bikesharing、California、News、Powerplant），包含 5 类二分类与 5 类回归任务。

**📈 对比分析**

在相同扩散框架下与无几何监督版本进行对比，并与 TabDiff 进行基准对照。跨架构实验中 GATD 在 Shape 27/30、Trend 25/30 上获胜；在 MLP 轨道上，Shape、Trend、下游 AUROC/R^2、F1/RMSE 分别达到 8/10、7/10、6/10、9/10；整体 Shape、Trend 分别降低约 27% 与 19%，下游指标提升显著。

**⚠️ 局限性**

局限性包括：对角特征的 O(d^2) 计算与存储开销在列数较大时显著；对连续性强的数据集，部分下游指标提升不明显；需要手动调节损失权重；且方法本身不提供正式的隐私保护。

---

## 124. GRZO: Group-Relative Zeroth-Order Optimization for Large Language Model Fine-Tuning

**arXiv ID:** 2606.02857 | [PDF](https://arxiv.org/pdf/2606.02857v1)

**作者:** Liyan Tan `[一作]` (University of California, Santa Barbara), Zheng Zhang `[通讯]` (University of California, Santa Barbara)

**通讯引用:** 79172 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于零阶优化的GRZO优化器，通过在每个小批量中生成伪独立的扰动方向并采用组相对归一化来显著降低梯度估计方差，从而实现大语言模型的高效微调。

**💡 创新点**

创新点在于将Mini‑Batch本身视作多方向扰动源：利用Flipout式符号分解在不额外存储或前向运算的前提下生成B个伪独立扰动方向，并通过组相对归一化聚合每例损失差，理论上实现方差按1/B缩放、收敛速率提升。

**🔧 技术方法**

核心技术包括零阶（ZO）优化、MeZO两点估计、Flipout符号分解生成每例扰动、组相对归一化（GRPO）权重、两次前向推理、梯度方向无偏性与方差分析、以及与Sparse‑MeZO、LOZO、QuZO等稀疏/低秩/量化变体的组合实现。

**📊 数据集**

实验基准使用GLUE/SuperGLUE（SST‑2、RTE、CB、BoolQ、WiC、MultiRC、COPA）、SQuAD与DROP任务，模型包括RoBERTa‑large（350M）、Llama3‑8B与OPT‑13B。

**📈 对比分析**

与Adam、LoRA等一阶基线及MeZO、FZOO等零阶基线对比，GRZO在Llama3‑8B上平均精度提升+3.0点、在OPT‑13B上+2.1点，峰值GPU内存比MeZO低23%，且与FZOO相比保持或超过精度；在与稀疏、低秩、量化变体组合后进一步提升。

**⚠️ 局限性**

局限性包括：批量大小需≥16以保证组相对归一化稳定；每步需两次前向推理，导致与MeZO相比略慢；未验证一阶偏差低的单向估计；仅在13B规模模型上验证，需进一步评估70B+大规模模型以及更高效的变体。

---

## 125. Gate AI: LLM Security Benchmark Evaluation Methodology and Results

**arXiv ID:** 2606.02959 | [PDF](https://arxiv.org/pdf/2606.02959v1)

**作者:** Ryle Goehausen `[一作]` (Constellation Network), Marcus Sousa `[通讯]` (Constellation Network)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一套无泄漏的评估框架，对 16 个公开基准进行 5 折 StratifiedKFold 与 StratifiedGroupKFold 交叉验证，评估 Prompt Injection 与 Jailbreak 检测器在统一阈值下的性能，并与多家竞争者做对比。

**💡 创新点**

创新点在于：①采用全局统一阈值并在评估时匹配竞争者的 FPR；②通过 MinHash+LSH 近似重复聚类与 StratifiedGroupKFold 并行诊断泄漏；③提供 Bootstrap 置信区间、随机标签、对抗验证等多维度诊断，确保结果的稳健性。

**🔧 技术方法**

使用的技术包括 5 折 StratifiedKFold、StratifiedGroupKFold、MinHash+LSH 近似重复检测、早停与阈值搜索、概率校准（Isotonic Regression）、集成多头分类器、分层 Bootstrap 置信区间、长度相关性、特征置换重要性等。

**📊 数据集**

使用了 16 个公开基准，总计 12,111 条样本，覆盖直接注入、间接注入、Red‑team、over‑defense、harm/jailbreak 等多种攻击类别。

**📈 对比分析**

比较方法：对每个基准采用匹配 FPR 的阈值重调，并用 95% 分位 Bootstrap CI 包容样本量差异；在 FPR≤1% 的宏/微 F1 分别为 97.4% / 95.7%，在自然阈值下分别为 98.7% / 98.4%，在多数指标上击败大多数竞争者，平均推理延迟 104 ms。

**⚠️ 局限性**

局限性包括：未涵盖多语言、长文本、语音/图像注入、跨会话攻击；近似重复检测依赖 Jaccard ≥0.8 可能漏检；评估基准受预训练数据污染；单一阈值可能不适用于所有源。

---

## 126. Pretraining Language Models on Historical Text

**arXiv ID:** 2606.02991 | [PDF](https://arxiv.org/pdf/2606.02991v1)

**作者:** Xiaoxi Luo `[一作]` (University of Waterloo), Yao Lu `[通讯]` (University College London)

**通讯引用:** 12852 | [OpenAlex ID](https://openalex.org/A5005089910)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练并发布了一个只使用1913年前英文文本的7B参数历史语言模型TypewriterLM，配套构建历史语料、词汇约束的指令微调数据集以及时间一致性评测基准History-Event。

**💡 创新点**

首次通过严格的时间泄露控制（数据清洗、词汇约束微调、Leakage‑aware评测）实现大规模历史LMM；引入lexically grounded instruction tuning及自我指令生成以避免现代信息注入；提供专门的历史事件评测集。

**🔧 技术方法**

采用Llama‑3架构的解码器Transformer，bfloat16训练，Group‑Query Attention、RMSNorm、Rotary位置嵌入；自定义BPE tokenizer；LoRA微调；Lexically grounded instruction tuning；BPB惊讶度与Leakage recall评估。

**📊 数据集**

54B-token TypewriterCorpus（Institutional Books、British Library Books、Hansard、Royal Society、Old Bailey、CLMET等）；History‑LIMA（1,000例）；History‑SelfInstruct（287k例）；History‑Event评测集（2,344历史事件）；标准基准ARC、HellaSwag、AlpacaEval、IFEval。

**📈 对比分析**

在ARC、HellaSwag、AlpacaEval、IFEval等现代基准上与Mr. Chatterbox、TimeCapsuleLLM、GPT‑1900、Talkie‑1930及GPT2‑XL对比，模型随规模提升表现优良；在History‑Event中BPB惊讶度随时间偏移；Leakage率低于0.6%，但仍存在轻微泄露；整体显示模型在历史知识范围内性能可观。

**⚠️ 局限性**

未系统研究语料构成对预训练效果的影响；受限于有限历史数据；模型仍可能生成冒犯性或危险内容；Leakage仍有轻微概率；缺乏对数据高效训练与合成数据生成的探索。

---

## 127. When Helping Hurts and How to Fix It: Multi-Agent Debate for Data Cleaning

**arXiv ID:** 2606.02866 | [PDF](https://arxiv.org/pdf/2606.02866v1)

**作者:** Chirag Parmar `[一作]` (Meta Platforms, Inc.), Shweta Medhekar `[通讯]` (Meta Platforms, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了多智能体辩论在LLM数据清洗中的效能，并提出了可量化的辩论收益条件来预测何时使用辩论有利；

**💡 创新点**

创新点在于用跨九种任务类型验证了辩论收益条件，并揭示了批判导致混淆（CIC）机制，随后证明代码执行与证据门控能消除这一问题；

**🔧 技术方法**

采用Generator‑Critic双代理架构，配合Claude、Gemini、Qwen3、DeepSeek等大型语言模型、文本提示、代码执行沙箱和证据门控等技术；

**📊 数据集**

使用AutoDCWorkflow、MMTU和MaTElDa三大公开基准数据集进行实验；

**📈 对比分析**

通过与单代理、不同提示、代码执行对照，并使用配对bootstrap置信区间和Cohen d进行统计比较；在检测任务上F1提升约27%，生成任务整体下降1–15%，但代码执行+证据门控组合在生成任务上首次实现+5%的提升；

**⚠️ 局限性**

局限性包括仅在小表（5–100行）单表任务上验证，四种大模型、仅两代理结构、未对大规模或多表/流式场景进行测试，且对非可验证任务仍无显著提升。

---

## 128. Visual Graph Scaffolds for Structural Reasoning in Large Language Models

**arXiv ID:** 2606.02673 | [PDF](https://arxiv.org/pdf/2606.02673v1)

**作者:** Runlin Lei `[一作]` (Renmin University of China), Zhewei Wei `[通讯]` (Renmin University of China)

**通讯引用:** 5356 | [OpenAlex ID](https://openalex.org/A5074858555)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多跳问答任务中，将教师模型的推理轨迹改写为图形化思维导图，并通过视觉图像作为指导，帮助学生模型进行推理。

**💡 创新点**

提出将图形视作结构化推理的视觉支架，并证明在保持结构化指导的“抽象”场景下，视觉图相比文本更有效。

**🔧 技术方法**

使用教师-学生框架、Graphviz 渲染图像、视觉语言模型、Self‑SFT（自监督微调）和 KL 蒸馏等技术。

**📊 数据集**

HotpotQA、2WikiMultiHopQA、MuSiQue 三个经典多跳问答数据集。

**📈 对比分析**

在直接与抽象两种指导设置下对比视觉与文本指导：在抽象场景下，视觉指导在重评中约 70% 以上准确率，而文本指导仅 47%；在内部化后（SFT 或 KL 蒸馏），视觉指导保持 64% 左右，文本约 58%；且视觉指导产生的回答更短。

**⚠️ 局限性**

仅在多跳问答范围内表现良好，难以迁移至其它推理任务；且视觉指导的效果仍低于直接以教师 CoT 训练得到的性能。

---

## 129. Fast Unlearning at Scale via Margin Self-Correction

**arXiv ID:** 2606.02920 | [PDF](https://arxiv.org/pdf/2606.02920v1)

**作者:** Federico Di Gennaro `[一作]` (ETH Zürich), Fanny Yang `[通讯]` (ETH Zürich)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于边际自校正（MASC）的语言模型无学习方法，利用token级别的优势概率约束，使模型在保留原性能的前提下逐步遗忘指定训练样本；

**💡 创新点**

创新点在于：①将token级别的“主导度”约束直接用于无学习目标，并通过该约束构造在线停止规则，消除了传统固定预算或后续评估的需求；②采用对top‑k替代token的对数概率差（margin）作为惩罚，使梯度更具指向性；

**🔧 技术方法**

主要技术包括：基于softmax概率与log-sum-exp的局部margin计算；使用KL正则化保持保留数据的分布；在线监测违规率Vρ并在满足阈值α时停止；对模型权重进行梯度下降更新（可通过LoRA实现）；

**📊 数据集**

使用的公开无学习基准数据集包括TOFU（90/10 Q&A拆分）、MUSE News、MUSE Books，并在Qwen2.5模型族上进行规模性研究；

**📈 对比分析**

与GA、GradDiff、NPO、NPO+KLR、RMU、SimNPO等基线对比，MASC在TOFU数据集上获得最优或竞争的忘记-保持度（如1‑ROUGE‑L、Truth Ratio、MU）并且训练时间大幅缩短；在MUSE News/Books上也保持了较高的保持效能并在忘记度上保持领先；

**⚠️ 局限性**

局限性包括：仅针对token级别的遗忘，未能直接约束语义层面的重述（paraphrase）遗忘；对k、ρ、α等超参的敏感性；依赖教师强迫（teacher‑forcing）评估；对大型模型的实现仍需进一步验证。

---

## 130. Pathway-Structured Privileged Distillation for Deployable Computational Pathology

**arXiv ID:** 2606.02877 | [PDF](https://arxiv.org/pdf/2606.02877v1)

**作者:** Yongxin Guo `[一作]` (Wake Forest University), Metin Gurcan `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 MoPE（Mixture of Pathway Experts）框架，利用RNA路径（Hallmark 50基因集）在训练时进行知识蒸馏，将分子信息软映射到仅使用 H&E 整片的图像模型，最终实现无 RNA 推理的可部署计算病理模型。

**💡 创新点**

创新点在于：①将分子监督组织为生物通路层面而非单基因；②采用记忆使用对齐的间接蒸馏（memory‑usage distillation）而非直接特征匹配，避免强制图像必须复现不可见的分子信号；③使用路径索引专家和多槽机制提供可解释的通路级视觉读数；④在部分可观测环境下实现无 RNA 推理。

**🔧 技术方法**

技术要点包括：Uni‑v2 基础模型提取图像特征；50 个路径专家（每个含多槽）与低秩适配器压缩；记忆池作为共享潜在基底；记忆使用蒸馏、路径掩码重建、槽多样性正则化；分类任务使用交叉熵，生存任务采用离散时间风险模型；校准采用 Platt scaling，决策曲线与影响曲线分析。

**📊 数据集**

数据集：内部评估使用 TCGA 5 个癌种（BRCA、LUAD、GBMLGG、STAD、KIRC）共约 3,000 例；外部评估使用两套独立 HER2‑负、HR‑正乳腺癌 WSI 集（OSUWMC 1,123 例、Dartmouth 522 例），用于 Oncotype DX 高危风险预测。

**📈 对比分析**

与多种基线对比：单模态 MIL 方法（AttMIL‑MoE、TransMIL‑MoE 等）、知识蒸馏基线（G‑HANet、MKD、DMML 等）、全模态多模态方法。MoPE 在 5 个分类任务上平均提升 AUC 约 2.0%（对比 AttMIL‑MoE）和 2.7%（对比 G‑HANet），在 4 个生存任务上平均提升 C‑index 约 3%。在外部 ODX 任务中，MoPE AUC 达到 80%+，高于 TransMIL‑MoE 的 76% 和 G‑HANet 的 77%。

**⚠️ 局限性**

局限性：①训练需要配对 WSI‑RNA 数据，限制可用数据集；②通路级视觉读数为解释性指标，尚未通过空间转录组等技术验证真实生物学活性；③外部评估仅针对 ODX 任务，未覆盖其他临床指标；④模型对组织分割、染色批次和图像质量较为敏感，可能导致失效案例。

---

## 131. Brief Announcement: Generative Markov Model for Distributed Computing Systems

**arXiv ID:** 2606.03061 | [PDF](https://arxiv.org/pdf/2606.03061v1)

**作者:** Alfreds Lapkovskis `[一作]` (Stockholm University), Praveen Kumar Donta `[通讯]` (Stockholm University)

**通讯引用:** 2013 | [OpenAlex ID](https://openalex.org/A5079303717)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

建立了分布式计算系统的生成式马尔可夫模型，并在协同 AI 推理场景中演示其应用。

**💡 创新点**

提出双层稀疏因式分解的生成式马尔可夫框架，使得高维系统状态可被分解为可处理的子模型，并将分布式计算与马尔可夫链及强化学习天然连接。

**🔧 技术方法**

使用因式化马尔可夫链、动态贝叶斯网络结构、离散时间模拟以及平均奖励强化学习方法进行建模、推理与策略优化。

**📊 数据集**

采用仿真生成的用户在线/离线状态、请求类型及预收集设备测量得到的资源消耗分布；实验场景设置四种任务类型（轻量/重量级），采样概率分别为 0.4、0.3、0.2、0.1。

**📈 对比分析**

通过对比集中式与分布式调度策略，在不同用户数量和服务器容量下测量 P99 延迟和服务器 CPU/内存使用，发现随着用户数增长，分布式策略在延迟和服务器资源消耗上均优于集中式。

**⚠️ 局限性**

实验仅在模拟环境中验证，未涵盖真实环境的非 i.i.d. 用户行为、系统可扩展性与策略学习实现，且仅覆盖四种任务类型和有限的资源维度。

---

## 132. Direct Informed Sampling on Riemannian Manifolds via Loewner Order Lower Bounds

**arXiv ID:** 2606.02879 | [PDF](https://arxiv.org/pdf/2606.02879v1)

**作者:** Phone Thiha Kyaw `[一作]` (University of Toronto), Jonathan Kelly `[通讯]` (University of Toronto)

**通讯引用:** 3338 | [OpenAlex ID](https://openalex.org/A5011931977)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Loewner顺序的矩阵值可接受启发式，用于在配置相关的Riemannian度量下实现直接、拒绝无采样的启发式采样；

**💡 创新点**

创新点在于构造全方向性的Loewner下界矩阵，保留度量张量的方向信息，从而得到比标量特征值下界更紧的启发式，并通过其Cholesky分解将Riemannian启发式集合映射为标准椭球，实现在欧氏空间中的直接采样；

**🔧 技术方法**

使用Loewner顺序与对称正定矩阵的最小下界、Cholesky分解、线性等距变换、仿射变换、以及迭代下界更新算法；

**📊 数据集**

在MotionBenchMaker数据集上评估，使用三台机器人（UR5、Franka、PR2）与三种Riemannian度量（加权、动能、回归）进行九个实验场景；

**📈 对比分析**

将所提矩阵启发式与欧氏距离、零启发式、以及几种无启发式或基于A*的规划器（BIT*、AIT*、EIT*、GRRT*、AORRTC）进行对比；实验表明矩阵启发式在所有高维和强方向异质的度量下显著加速收敛，尤其在动能度量下提升了3–4倍；

**⚠️ 局限性**

局限在于对正则化强的度量（如回归度量）下Loewner下界趋向于标量极限，失去方向优势；同时下界的求解仍需多次度量评估，计算成本较高；

---

## 133. Fewer, Better Frames: A Compute-Normalized Proof of Concept for Coherence-First World-Model Rendering with Model-Guided FSR4 Frame Generation

**arXiv ID:** 2606.02586 | [PDF](https://arxiv.org/pdf/2606.02586v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 134. EntangleCodec: A Unified Discrete Audio Tokenizer via Semantic-Acoustic Entanglement

**arXiv ID:** 2606.02739 | [PDF](https://arxiv.org/pdf/2606.02739v1)

**作者:** Hui Li `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 17221 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的离散音频分词器EntangleCodec，可同时支持音频理解与生成

**💡 创新点**

创新点在于将语义与声学信息在量化前进行混合编码，并使用丰富的描述性字幕进行对比学习，避免了单独语义/声学流的冗余与对齐问题

**🔧 技术方法**

采用Transformer编码器、单码本向量量化、CLIP式音频-文本对比损失、流匹配扩散解码器以及两阶段训练策略

**📊 数据集**

使用LibriSpeech、MusicBench、AudioSet、AudioCaps、WavCaps等多域语音、音乐与一般音频数据集进行训练，配合ASR与LLM生成的丰富字幕进行监督

**📈 对比分析**

与多种基线进行对比：在重建指标上与专业码流相当；在音频理解MMAR、MMAU-mini、MMAU上比所有离散码流领先最多7.4%；在TTS与TTA生成上同时取得最佳WER/UTMOS与CLAP分数，且在相同参数规模下能超过13B参数的连续表示模型

**⚠️ 局限性**

局限在于语义细粒度不足（字幕质量限制）、未探索更大规模模型（30B/70B等）以及在更大规模下重建与理解任务的平衡问题

---

## 135. Which Defense Closes Which Threat? Attributing OWASP-LLM-Top-10 Coverage and Its Brittleness Under Paraphrasing

**arXiv ID:** 2606.02822 | [PDF](https://arxiv.org/pdf/2606.02822v1)

**作者:** Alexandre Cristovão Maiorano `[一作]` `[通讯]` (Lumytics), Alexandre Cristovão Maiorano (Lumytics)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个四目标防御金字塔（L_0~L_3）以及一个17条探针的静态测试集，用来对 OWASP‑LLM‑Top‑10 的五个关键风险进行单轴归因评估，并在此基础上进行抗改写的脆弱性实验。

**💡 创新点**

创新点在于：① 引入“单轴消融金字塔”方法，实现对各类防御措施（拒绝词表、Token 预算、工具注册认证、PII 清理）对 OWASP‑LLM‑Top‑10 每个风险的归因；② 提供公开可复现的 17‑probe 语料和 60‑template 语料的抗改写测试，展示防御在攻击改写下的脆弱性；③ 将结果以准确率/召回率/置信区间形式呈现，并公开了完整的复现包。

**🔧 技术方法**

技术包括：Node.js 伪造的四个 Docker 容器目标；基于 HTTP 层的 25 代理攻击模拟器；拒绝词表正则、Token 上限、模型白名单、速率限制和 PII 过滤等防御实现；Gemini‑2.5‑Flash 作为对抗性改写引擎；多种统计方法（bootstrap 95% CI、TP/FP/FN 分析）。

**📊 数据集**

数据集：① 17 条固定探针，覆盖 LLM01、LLM02、LLM06、LLM07、LLM10 5 个 OWASP 领域；② 60‑template 脆弱性语料；③ 300 条 Gemini‑2.5‑Flash 生成的改写变体；此外公开的四个 Docker 目标和扫描日志。

**📈 对比分析**

比较方法：对四个目标（L_0~L_3）和一个真实 LLM 后端（L_4）进行多次复制（N=10）评估每类风险的发现计数；使用精确率 1.00、召回率 0.75 的统计表；在改写实验中比较拒绝词表在原始探针和改写探针上的阻断率下降。性能表现：单轴消融显示 L_1 能消除 LLM01/07，L_2 能消除 LLM02/10，完整堆栈 L_3 能消除 LLM06；改写导致 L_1 的 LLM01/07 阻断率分别下降 15pp 与 25pp；L_2 在改写下保持 0%。

**⚠️ 局限性**

局限性：① 目标仅为模拟 stub，真实 LLM 后端的随机性与内在学习机制未被完全捕获；② 单轴消融未考虑防御间的交互与协同效果；③ 攻击改写仅使用单一 Gemini 模型，未涵盖多种改写策略；④ 仅评估了 5 个 OWASP 领域，未覆盖全部 10 项；⑤ 结果高度依赖固定 probe 集，未体现对更大、更动态的真实攻击流的泛化能力。

---

## 136. Mitigating Spurious Correlations with Memorization-Guided Dataset De-Biasing

**arXiv ID:** 2606.02830 | [PDF](https://arxiv.org/pdf/2606.02830v1)

**作者:** Arda Fazla `[一作]` (Purdue University), Abolfazl Hashemi `[通讯]` (Purdue University)

**通讯引用:** 471 | [OpenAlex ID](https://openalex.org/A5036900440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了两阶段累计样本损失（TCSL）评分与对应的核心集选择算法TCSL-CS，旨在通过分离核心与伪相关特征学习并挑选最具代表性的10%数据子集，提升模型对最差组的鲁棒性。

**💡 创新点**

创新点在于引入可同时评估核心和伪特征难度的两阶段网络与累计样本损失分数TCSL，并基于此构建专门针对伪相关的数据子集选择策略，实现在仅使用10%训练数据时即可达到或超过最先进的去偏方法。

**🔧 技术方法**

采用两阶段训练（先学习伪特征再冻结，随后训练核心网络）、累计样本损失（CSL）与加权k-means聚类、分层采样等技术，并在理论上利用神经切线核（NTK）框架分析伪特征的先学习优势。

**📊 数据集**

在Waterbirds、cMNIST、MetaShift、UrbanCars-B等含有明显伪相关的计算机视觉基准数据集上进行实验。

**📈 对比分析**

与GroupDRO、RGbal、EL2N、SelfSup、D2、Random等多种基准方法在WGA与AVG指标上对比，TCSL-CS在WGA上显著优于或与最先进方法持平，且仅需10%的训练数据即可实现与传统去偏方法相当甚至更好的性能。

**⚠️ 局限性**

局限性包括需要额外的两阶段训练成本，对伪相关程度的依赖、对超参数和采样比例敏感，以及理论分析局限于NTK框架，尚未验证在非视觉任务或更广泛场景中的通用性。

---

## 137. Pixel Cube: Diffusion-based Portrait Video Relighting Through Realistic Lighting Reproduction

**arXiv ID:** 2606.02919 | [PDF](https://arxiv.org/pdf/2606.02919v1)

**作者:** Yufan Zhang `[一作]` (George Mason University), Jinwei Ye `[通讯]` (George Mason University)

**通讯引用:** 1070 | [OpenAlex ID](https://openalex.org/A5070020700)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一种基于扩散模型的动态人像视频重新照明方法，利用环境图控制实现高保真、时序一致的重光照效果。

**💡 创新点**

结合LED立方体光照系统获取真实光照标注数据，构建混合真实+合成视频数据集，并采用两阶段 delight‑then‑relight 的扩散模型与环境图多级交叉注意力实现精准光照控制。

**🔧 技术方法**

Stable Video Diffusion扩散网络、环境图编码器、多级交叉注意力、背景图控制、时序滑动窗口推理等技术。

**📊 数据集**

Pixel Cube采集的真实动态人像视频（含HDR环境图、白背景光照、贴合蒙版、平照色素）和MetaHuman渲染的合成视频，共约3百万帧。

**📈 对比分析**

与PN‑Relight、SwitchLight、RelightVid等方法在自采真值和野外视频上进行定量（PSNR/SSIM/LPIPS）和用户评价，结果在亮度一致性、身份保持和时序稳定性上均优于对比方法。

**⚠️ 局限性**

LED面板亮度限制导致无法模拟极高动态范围光源；采集序列延迟约8 ms可能在高速运动时产生伪影；长视频推理仍易累积漂移。

---

## 138. CRAM-ER: Error-Resilient Spintronic Computational Random Access Memory for Scalable In-Memory Computation

**arXiv ID:** 2606.02781 | [PDF](https://arxiv.org/pdf/2606.02781v1)

**作者:** Sohan Salahuddin Mugdho `[一作]` (Iowa State University of Science and Technology), Cheng Wang `[通讯]` (Iowa State University of Science and Technology)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对多位DNN加速的错误容忍型MRAM计算随机存取内存（CRAM-ER）架构，并实现了其硬件与软件协同设计。

**💡 创新点**

创新点在于将低开销多数投票错误校正与CMOS+Spintronic混合加法树结合，同时通过误差感知微调实现近乎无损的推理精度。

**🔧 技术方法**

使用了STT-MRAM基的CRAM单元、MAJ3错误校正电路、CMOS加法树以及误差感知的Pytorch微调框架。

**📊 数据集**

实验数据集包括MNIST（LeNet-5）、CIFAR-10（ResNet-20/ResNet-18）以及Vision Transformer ViT-b/4。

**📈 对比分析**

与传统CRAM、CPU+DRAM、GPU+HBM等基线比较，CRAM-ER在保持近乎无损精度的同时能实现10–40×的能耗提升和20–200×的时延下降，EDP提升至2×甚至16×。

**⚠️ 局限性**

局限性在于仍需依赖高质量的STT-MRAM写性能，且写入开销与错误率在极大规模网络上仍然是瓶颈；硬件实现复杂度和面积开销需进一步优化。

---

## 139. Spike-Aware C++ INT8 Inference for Sparse Spiking Language Models on Commodity CPUs

**arXiv ID:** 2606.03026 | [PDF](https://arxiv.org/pdf/2606.03026v1)

**作者:** Ting Liu `[一作]` `[通讯]` (SymbolicLight Research), Ting Liu (SymbolicLight Research)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

为稀疏发射（spiking）语言模型实现了一套 C++ INT8 推理运行时，并在 AMD Ryzen 7 5800X 上对其进行性能评测。

**💡 创新点**

创新点在于：① 将激活稀疏性视为执行原语；② 混合内存布局（密集层 row‑major，稀疏层 column‑major）与 AVX2/FMA 结合；③ 每通道 INT8 量化与整数域累加，专为 spike‑conditioned 路径优化；④ 对 SymbolicLight V1 checkpoint 进行 manifest‑driven 加载，避免关键字匹配错误。

**🔧 技术方法**

技术手段包括：manifest 驱动权重加载、混合布局内存、AVX2/FMA kernels、每通道对称 INT8 量化、整数域稀疏累加、并行多线程解码。

**📊 数据集**

使用的数据集：SymbolicLight V1 874M 参数模型（186k 步）以及 WikiText‑2 用于 perplexity 评估；对比基准使用 llama.cpp Q8_0 版本的 Qwen2.5、TinyLlama、Falcon3 等子 2B 模型。

**📈 对比分析**

比较方法：在单线程、4 线程以及长 prompt 前填充场景下测量 tokens/s、内存占用和 WikiText‑2 PPL。结果显示：单线程 22.63 tokens/s（INT8），4 线程 47.90 tokens/s；相比同规模 dense 基线（16.31–32.76 tokens/s）速度更快，但 WikiText‑2 PPL 24.80，明显低于 dense 基线。

**⚠️ 局限性**

局限性：① 语言质量差，PPL 远高于 dense 对手；② 未提供能耗测量，仅有粗略估算；③ 仅针对 x86 AVX2，缺乏 ARM/neuromorphic 支持；④ 仅评估文本推理，未验证嵌入式或机器人任务；⑤ 量化精度折衷（INT4 影响速度与质量）。

---

## 140. Cross-Vendor Sola ISPM Benchmark: Evaluating Agentic AI for Federated Identity Security Reasoning

**arXiv ID:** 2606.02674 | [PDF](https://arxiv.org/pdf/2606.02674v1)

**作者:** Eden Yavin `[一作]` (Sola Security), Gal Baron `[通讯]` (Sola Security)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了跨供应商的 Sola ISPM 基准（包含 50 个数据驱动任务）并设计了包含答案正确性、证据支撑、结构化连接、检索质量与 SQL 等价性的评估框架；随后在五种上下文配置下使用三款前沿 LLM（Claude 4.8 Opus、Claude 4.6 Sonnet、GPT‑5.5、Gemini‑3.1 Pro）评估 Sola AI Agent 的性能。

**💡 创新点**

创新点包括①首个跨供应商、真实生产数据驱动的 ISPM 基准；②引入证据支撑与结构化 JOIN 评估，超越传统单一数据库的 text‑to‑SQL 衡量；③通过上下文消融实验系统揭示跨供应商图谱（Security Graph）对 LLM 推理与执行效率的关键作用。

**🔧 技术方法**

使用技术主要包括：Sola AI Agent 的 schema‑grounded 执行模型、双向检索与示例引导、逐步推理与工具执行；LLM‑as‑Judge（Claude、GPT、Gemini）进行多样本共识评分；确定性结构化指标（表/JOIN 精度/召回/F1）与标准的安全操作流程。

**📊 数据集**

数据集来自八大企业平台的真实生产环境：AWS、Okta、Azure AD、Google Workspace、GitHub、GCP、MongoDB Atlas、HiBob；通过专家、自动生成与检验循环构造了 50 个跨平台的真实问题与对应 SQL，保证了任务的可执行性与数据完整性。

**📈 对比分析**

对比方法：在五种上下文配置（无上下文、仅模式、模式+图谱、模式+示例、完整上下文）下，利用九项 LLM‑as‑Judge 评估指标（检索、推理、SQL、答案）以及结构化精度。结果显示：完整上下文下 Claude 4.8 Opus 的答案正确率达 78%，整体失败率降至 4%，相较于无上下文提升约 34% 的正确性，探索 SQL 查询次数下降约 70%。

**⚠️ 局限性**

局限性：基准仅包含 50 题且多为 1‑3 跳的多跳场景；缺乏更高难度的 4 跳及以上任务；依赖人工专家验证与生产环境配置，可能难以直接迁移至其他组织；未来工作需扩展到行为分析、风险评分、治理一致性等更广泛 ISPM 维度。

---

## 141. Automated Report-Derived Oncology VQA Benchmark for Evaluating Vision-Language Models on 3D Medical Imaging

**arXiv ID:** 2606.02809 | [PDF](https://arxiv.org/pdf/2606.02809v1)

**作者:** Bo Liu `[一作]` (University of California San Francisco), Hui Lin `[通讯]` (University of California San Francisco)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

自动化生成基于私有放射报告和3D影像的多选视觉问答数据集，并对六款视觉‑语言模型进行零样本评估；

**💡 创新点**

双路径生成策略（RADs 样式+LLM 生成报告题目）、以私有报告为源实现实例泄漏控制、公开 agent 工作流、盲视差实验揭示视觉依赖的数据集特异性；

**🔧 技术方法**

利用提示式 LLM 进行 schema 提取与题目生成，agent 驱动管道完成自动化流水线，使用 bootstrap CI 估计置信区间，进行盲视差评估；

**📊 数据集**

四组内部肿瘤队列：脑 MRI（PDGM+私有报告）、肝 MRI、肝 CT、肺 CT，全部为确诊癌症病例；

**📈 对比分析**

对六个模型（Claude Opus 4.6、GPT‑5/5.2、Qwen3‑VL‑30B、MedGemma 1.0‑27B、MedGemma 1.5‑4B）在 RADs 与报告题型上做零样本准确率评估，未出现单一领先者，最高准确率约 0.70，部分数据集显示视觉与语言先验差异；

**⚠️ 局限性**

LLM 生成的题目可能带偏差，视觉先验泄漏风险仍存在，2D 输入对 3D 影像的适配有限，盲视差实验未覆盖全部模型和全部题库，数据量相对有限，报告文本未完全离线化。

---

## 142. Acceptance-Test-Driven Evaluation Protocols for Business-Centric LLM Systems

**arXiv ID:** 2606.02755 | [PDF](https://arxiv.org/pdf/2606.02755v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 143. Reproducibility is the New Copyleft: Defining AGI-oriented Reproducible Builds

**arXiv ID:** 2606.03019 | [PDF](https://arxiv.org/pdf/2606.03019v1)

**作者:** Masayuki Hatta `[一作]` (Surugadai University), Masayuki Hatta `[通讯]` (Surugadai University)

**通讯引用:** 1388 | [OpenAlex ID](https://openalex.org/A5033204192)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套适用于AGI系统的可复现构建（AGI‑RB）七项要求，并将可复现构建视为新的copyleft机制；

**💡 创新点**

创新点在于将传统软件的源-目标等价关系迁移到AI领域，提出递归可验证性（R6）以及完整输入枚举、确定性训练等新要求；

**🔧 技术方法**

使用的技术包括可复现构建技术、固定PRNG种子与确定性cuDNN、硬件绑定与抽象、第三方验证平台、加密日志与可追溯性；

**📊 数据集**

论文并未使用具体数据集，而是要求完整训练数据的公开描述，通常指向大规模公开语料（如Common Crawl、Wikipedia等）并提供数据信息；

**📈 对比分析**

对性能的讨论主要来自已有可复现研究，表明大模型可实现比特级可复现，但会产生确定性开销，未给出实测对比；

**⚠️ 局限性**

局限在于高成本与技术成熟度要求，尤其是递归可验证性仍是开放研究，可能导致对超大规模AGI模型的可行性受限。

---

## 144. SegTune: Structured and Fine-Grained Control for Song Generation

**arXiv ID:** 2606.02638 | [PDF](https://arxiv.org/pdf/2606.02638v1)

**作者:** Yuejiao Wang `[一作]` (Kling Team Kuaishou Technology), Pengfei Wan `[通讯]` (Kling Team Kuaishou Technology)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SegTune——一种基于Diffusion Transformer的非自回归（NAR）歌曲生成框架，支持全局与段落级文本控制并自动预测歌词时间戳；

**💡 创新点**

创新点包括：①层次化段落级文本条件注入，实现对情感、节奏、乐器等局部属性的细粒度控制；②利用LLM（Qwen3-4B）训练的持续时间预测器，实现句子级歌词对齐；③构建大规模高质量数据管道与新的细粒度评价指标；

**🔧 技术方法**

主要技术：Diffusion Transformer（DiT）+  条件流匹配；LLM嵌入编码器（Qwen3-Embedding-0.6B）用于全局/段落提示；LLM基持续时间预测器（Qwen3-4B）生成LRC时间戳；声码器VAE压缩音频；Classifier-Free Guidance与Euler ODE求解；

**📊 数据集**

使用内部中文流行歌曲数据集（约27k小时）以及少量其他语言歌曲；数据通过多阶段筛选、歌词对齐、结构标签提取；

**📈 对比分析**

与四个基线（YuE、LeVo、DiffRhythm+、ACE-Step）对比，SegTune在PER、AudioBox美学、SongEval、全局与段落MuLan分数上均优于基线；DPO后音质进一步提升，音乐性与质量评分最高，且标准差最低；

**⚠️ 局限性**

局限：①训练依赖明确的段落结构，若用户输入结构模糊或缺失，持续时间预测性能下降；②无法细化段内动态（如渐强、声部装饰）；③对多语言支持仍需进一步验证，数据偏向中文；

---

## 145. FOLD: Fuzzy Online Deduplication for Very Large Evolving Datasets via Approximate Nearest Neighbor Search

**arXiv ID:** 2606.03001 | [PDF](https://arxiv.org/pdf/2606.03001v1)

**作者:** Nelson Bore `[一作]` (McGill University), Oana Balmau `[通讯]` (McGill University)

**通讯引用:** 442 | [OpenAlex ID](https://openalex.org/A5055620336)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为 FOLD 的在线模糊去重系统，能够在文本大规模数据集持续增量的情况下，高速且准确地识别近似重复文档。

**💡 创新点**

创新点包括：① 使用 HNSW 图索引实现在线插入与局部候选检索；② 设计 bitmap‑based MinHash 签名，解决 Jaccard 相似度在图搜索中的得分拥挤与 tie‑breaking 问题；③ 对 bitmap 计算采用 SIMD 并行化与缓存技术，显著降低距离计算成本。

**🔧 技术方法**

核心技术：MinHash + LSH、HNSW 近似最近邻搜索、bitmap‑based Jaccard 近似、SIMD 指令加速、缓存 popcount、Python/C++ 混合实现。

**📊 数据集**

在四个真实 LLM 训练语料库上验证：LM1B、C4、RealNews、Common Crawl，覆盖从低冗余短文档到高冗余长网页的多种去重难度。

**📈 对比分析**

与 IBM DPK、Milvus 以及 FAISS（Jaccard）对比：FOLD 在 93–97% 的 DPK‑相对召回率下保持近乎恒定的吞吐率，最大吞吐提升约 2.09×（例如在 Common Crawl 最高 551 docs/s 对比 Milvus 263 docs/s）。

**⚠️ 局限性**

局限性：① 需要对 HNSW 参数（M、efConstruction、efSearch）和 bitmap 大小进行经验性调优；② 目前仅在 CPU 上实现，GPU 加速待探索；③ 对极高重复率或非常短文档的处理仍可能产生较高内存占用或偶尔的召回下降。

---

## 146. Fixed-Point Scaffolding in the Clef Programming Language

**arXiv ID:** 2606.02854 | [PDF](https://arxiv.org/pdf/2606.02854v1)

**作者:** Houston Haynes `[一作]` `[通讯]`, Houston Haynes

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文设计并实现了一个基于 MLIR 的中间端，利用固定点组合器与范畴构造保证 Clef 语言在多阶段编译过程中保持结构属性，并通过参数化与模态移位实现多层验证。

**💡 创新点**

创新点在于：①将编译轴与验证轴视为同一范畴构造的两条遍历，并用固定点组合器实现证明转换的组合；②在 MLIR 上引入负/分数类型的对偶结构；③以参数化、Ohori 机器码证明理论和对偶模态逻辑为三种基石，构建完整的结构-内容保持框架。

**🔧 技术方法**

采用的技术包括：MLIR（及其方言和 SMT 方言）、范畴学（Sheaf、Compact‑Closed、Adjoint Mode Logic）、Ohori 的机器码证明理论、参数化（Reynolds‑Wadler）与 Coeffect 分析、以及 C++ 生成的嵌套 SSA 结构。

**📊 数据集**

未使用任何实验数据集，所有验证均基于理论证明和工程实现。

**📈 对比分析**

论文未提供实验对比或性能评测，主要通过形式化证明和内部检查展示结构保持的正确性；若有评测，预期仅在编译时间上略有额外开销，但不影响最终目标代码性能。

**⚠️ 局限性**

局限性包括：尚未将所有编译 Pass 证明为证明转换器，导致需要额外的边缘检查；缺乏完整语义定义和正式的终端性能基准；负/分数类型的实现仍处于设计阶段，尚未在生产代码中验证。

---

## 147. What Benchmarks Don't Measure: The Case for Evaluating Abstention Competence in Autonomous Agents

**arXiv ID:** 2606.02965 | [PDF](https://arxiv.org/pdf/2606.02965v1)

**作者:** Victor Ojewale `[一作]` (Brown University), Suresh Venkatasubramanian `[通讯]` (Brown University)

**通讯引用:** 12080 | [OpenAlex ID](https://openalex.org/A5061790878)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究自主代理的中止（abstention）能力，指出传统任务完成评估忽视安全中止，提出“compliance bias”概念并构建三类缺口（specification、verification、authority）的评估框架。

**💡 创新点**

创新点在于将缺口分类与安全率（SR）、可用率（UR）、知情拒绝率（IRR）指标体系结合，提供可衡量的中止性能评价，并通过 144 场景数据集验证该框架的有效性。

**🔧 技术方法**

技术方法包括基于 RLHF 训练的 ReAct 语言模型、LangChain 工具调用链、三种中止条件（Baseline、Prompt-Only、Checkpoint）以及 runtime checkpoint 约束层，采用 GPT‑4o、GPT‑5.4‑mini、Llama 3.1 8B、Claude Sonnet 4.6、Claude Opus 4.6、Gemini 2.5 Pro/Flash 等七种模型进行实验。

**📊 数据集**

使用数据集为 24 条人类编写种子扩增得到的 144 条安全/安全控制场景（120 个危险缺口 + 24 个安全对照），每个危险场景都有对应的安全对照，以便进行对比评估。

**📈 对比分析**

比较方法：在 Baseline、Prompt‑Only 与 Checkpoint 三种条件下分别计算 SR、UR、IRR；结果显示 Checkpoint 在所有模型中实现 88–91% SR、>80% UR、100% IRR，显著提升安全性且保留可用性，说明 runtime enforcement 能有效克服 compliance bias。

**⚠️ 局限性**

局限性：实验规模仅限 144 场景，缺口类型聚焦于工具调用层面；Baseline/Prompt‑Only 的 IRR 依赖 LLM 判断，缺乏客观保证；未评估跨任务、跨域的普适性，且缺口标签标注需人工审核。

---

## 148. Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models

**arXiv ID:** 2606.02914 | [PDF](https://arxiv.org/pdf/2606.02914v1)

**作者:** Sema Helali `[一作]` (United Arab Emirates University), Rafat Damseh `[通讯]` (United Arab Emirates University)

**通讯引用:** 1672 | [OpenAlex ID](https://openalex.org/A5079507190)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

系统综述了2020-2026年间发表的97项牙科大模型研究，涵盖语言生成、视觉基础和专用牙科模型。

**💡 创新点**

提出双维分类框架：按架构（生成vs.视觉）与牙科专业化程度（通用→适配→专用）对模型进行组织。

**🔧 技术方法**

主要技术为Transformer架构，包括大语言模型、视觉基础模型（SAM、CLIP、GroundingDINO）以及经过大规模预训练或强化学习的牙科专用模型（DentVFM、DentVLM、OralGPT等）。

**📊 数据集**

数据来源于PubMed、Google Scholar、Scopus和arXiv的系统检索，综合近6000篇论文，并对各类模型在诊断、教育、沟通等任务的公开数据集与实验结果进行归纳。

**📈 对比分析**

通过对比各模型在诊断准确率、教育测试分数、报告生成质量等指标，发现集成多模型管道（视觉+生成）往往优于单一模型，牙科专用模型在多模态诊断与临床决策支持任务中表现最为突出。

**⚠️ 局限性**

局限包括生成模型的幻觉风险、标注数据稀缺导致的过拟合、缺乏统一的临床评估基准以及对模型可解释性和安全性的不足。

---

## 149. SkillGuard: A Permission Framework for Agent Skills

**arXiv ID:** 2606.03024 | [PDF](https://arxiv.org/pdf/2606.03024v1)

**作者:** Shidong Pan `[一作]` (CSIRO), Zhenchang Xing `[通讯]` (CSIRO and Australian National University)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 SkillGuard，一种针对 LLM 代理技能的权限框架，统一管理技能在上下文注入和行动侧的影响。

**💡 创新点**

将技能视为带权限的可执行实体，采用双平面（上下文与动作）权限声明、运行时访问控制、用户授权、默认拒绝、能力推断与行为监控的完整治理模型，填补现有工具或上下文防御单一维度的空白。

**🔧 技术方法**

基于 JSON DSL 的技能清单、工具调用拦截 Hook、能力映射与动态权限检查、最小化提示的用户交互、自动化 Manifest 生成的 Mini‑Agent 以及审计日志记录。

**📊 数据集**

在 315 个来自 SkillsMP 的真实技能、SkillInject 基准（23 清洁技能与注入版本）以及 1,400+ 任务上评估。

**📈 对比分析**

对照未使用 SkillGuard 的基线，在攻击成功率（ASR）、任务完成率（TSR）和 token/时间开销上评估；结果显示 ASR 降低约 9%/8%，TSR 下降约 1.5%，token 消耗增加 21%/32%，但平均运行时间下降 13%。

**⚠️ 局限性**

自动化 Manifest 生成仍可能漏报或误报，无法完全捕捉脚本内部动作；对声明式权限的过度依赖会导致额外提示；评估仅在单一模型/框架下完成，跨模型泛化需进一步验证。

---

## 150. GeoDrive-Bench: Benchmarking Region-Specific Multimodal Reasoning in Autonomous Driving

**arXiv ID:** 2606.02774 | [PDF](https://arxiv.org/pdf/2606.02774v1)

**作者:** Yingzi Ma `[一作]` (University of Wisconsin-Madison), Ming Jiang `[通讯]` (University of Wisconsin-Madison)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `8d10c613-917e-4880-9716-17789f50e119` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个针对不同国家驾驶规则的多选题目基准CulturalDrive Bench，评估视觉语言模型在不显式地区标签的情况下推理地区特定交通规则的能力。

**💡 创新点**

创新点包括：①构建六国（中国、日本、新加坡、英国、印度、美国）共5,053条人类验证的多选QA，涵盖感知、预测、规划和地区推理四大任务；②引入规则条件自蒸馏算法DriveOPD，使模型在不依赖规则提示的推理中内部化地区交通知识；③系统比较三种提示方式（直接、链式推理、规则给定），揭示文化规则差距是模型性能不均衡的主要原因。

**🔧 技术方法**

使用技术：视觉语言模型（如LLaVA、Qwen、InternVL、Llama-3.2、Gemma3）、Grounding DINO视觉检测、Qwen3-VL大语言模型进行场景抽取与状态提取、对抗式与人类校验的多选题生成、KL蒸馏自学习、对比实验与误差类型分析。

**📊 数据集**

数据集来源：六个公开驾驶数据集——CoVLA（日本）、ONCE（中国）、nuScenes（新加坡）、Waymo（美国）、LingoQA（英国）、IDD（印度）。

**📈 对比分析**

比较方法：在三种提示方式下对七款开源VLM进行评估，并对比规则给定与自蒸馏版本。实验表明，当前VLM在不同地区表现差异可达数十个百分点，Rule-Given能显著提升准确率；DriveOPD在直接提示下可匹配甚至超过Rule-Given，且跨国方差下降到5%以下，显著提升地区一致性。

**⚠️ 局限性**

局限性：仅覆盖六国且受公共数据集限制，评估仅限高层次多选推理，未覆盖低层规划和闭环驾驶行为；未对更小或更稀有地区、语言与交通标识多样性进行扩展。

---

## 151. ZK-Flex: A Flexible and Scalable Framework for Accelerating Zero-Knowledge Proofs

**arXiv ID:** 2606.03046 | [PDF](https://arxiv.org/pdf/2606.03046v1)

**作者:** Adiwena Putra `[一作]` (KAIST), Joo-Young Kim `[通讯]` (KAIST)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出一种软件‑硬件协同优化框架 ZK‑FLEX，利用可重构的 Toom‑Cook 多精度核心（TCore）和链表内存管理来加速 ZKP 证明生成。

**💡 创新点**

①通过 Next Smooth Composite (NSC) padding 和自适应 MSM 窗口选择实现软优化；②实现可配置的 Toom‑Cook 多精度乘法核心；③采用链表内存实现桶共享的 MPADD 并行化；④将软硬件策略统一，支持跨不同位宽和工作负载的高利用率。

**🔧 技术方法**

Toom‑Cook 多精度乘法、混合基数 NTT、Pippenger MSM、链表内存桶管理、软件优化器（NSC、窗口自适应、工作负载预测）、Ruche NoC、HBM 访问、系统级仿真与 28 nm 合成。

**📊 数据集**

涵盖 AES、SHA、RSA、Merkle‑Tree、拍卖、Sprout/Sapling（区块链）、Lenet、AlexNet、VGG16 等多种 ZKP 应用。

**📈 对比分析**

在相同 28 nm、1 GHz、HBM 460 GB/s 环境下，与 PipeZK、GZKP、LegoZK 对比；平均实现 5.5× 速度提升（最高 11.5×），面积效率平均提升 1.85×，最高 3.8×，在多曲线（BLS12‑381、MNT4‑753、BN128）均显著加速。

**⚠️ 局限性**

在某些位宽/基数下 TCore 资源利用不足导致面积效率下降；链表内存增加 scalar 内存占用；大规模并行时仍受 off‑chip HBM 带宽限制。

---

## 152. A Training-Efficient Transformer-Based Anti-Spoofing Network for Logical Access in ASVspoof 5

**arXiv ID:** 2606.02980 | [PDF](https://arxiv.org/pdf/2606.02980v1)

**作者:** Sidan Yin `[一作]`, Bo Zhao `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种Transformer‑based、focal‑pairwise attentive ranking网络（TFPARN）用于ASVspoof 5 Track 1的逻辑访问反欺骗任务。

**💡 创新点**

创新点在于将注意力池化与焦点分类损失和成对排名损失相结合，使训练目标更贴合EER/minDCF等排名敏感指标，同时保持模型轻量和训练高效。

**🔧 技术方法**

使用了Transformer编码器、轻量级注意力池化、焦点分类损失、成对排名损失、RawBoost数据增强与TTA等技术。

**📊 数据集**

在ASVspoof 5 Track 1的训练集（400名说话人、8种攻击）上训练，并在官方Dev/评测集上评估。

**📈 对比分析**

与重新实现的AASIST（0.30M参数）和RawNet2（17.62M参数）在统一协议下对比，TFPARN取得最优minDCF 0.2430、EER 12.52%、actDCF 0.2897；训练/推理成本最低，仅1.4 GB显存、约0.79 ms/句，训练时间约149 min。

**⚠️ 局限性**

局限在于成对排名损失仅是简单的hinge式对比，未能充分模拟minDCF等指标；未来可尝试更稳定的列表化或可微排序损失。

---

## 153. Too Much of a Good Thing: When sim2real Efforts Impede Policy Learning (And What to Do About It)

**arXiv ID:** 2606.02636 | [PDF](https://arxiv.org/pdf/2606.02636v1)

**作者:** Kyle Morgenstein `[一作]`, Luis Sentis `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `fa81e2aa-eb25-4aba-a919-7efd247b3885`

**🎯 论文内容**

本文提出了一种 sim2sim2real 框架，先在低阶 IsaacLab 模拟器中使用简化的动力学训练 RL 控制策略，并并行训练前向运动学模型；随后将该前向模型作为参考，将策略迁移到完整闭环动力学的 MuJoCo 验证模拟器，最终实现零射击硬件部署。

**💡 创新点**

创新点在于将行为生成与系统识别完全解耦，利用运动学前向模型作为跨仿真器的共享基准，消除“模拟器锁定”与动态可行性限制，使策略学习可在最适合的仿真环境中进行，同时仍能实现高保真硬件迁移。

**🔧 技术方法**

技术手段包括：PPO 强化学习、并行前向模型训练、状态基模仿奖励、学生-教师蒸馏与潜在引导、IsaacLab 与 MuJoCo 之间的 kinematic‑guided 迁移以及 Brax 等仿真工具。

**📊 数据集**

主要使用的是两套仿真数据集：IsaacLab 低阶模型的采样数据和 MuJoCo 全阶模型的验证数据；此外还利用硬件系统辨识得到的真实动力学参数作为后期验证。

**📈 对比分析**

通过在 Apollo 机器人上进行零射击部署进行比较，展示了在硬件上成功运行的政策以及显著提升的样本效率；虽然论文未给出数值指标，但指出相比传统单一高保真模拟器方法，迁移成功率更高、工程成本更低。

**⚠️ 局限性**

局限性包括：对前向运动学模型的准确性要求高，若运动学估计误差大则迁移效果下降；目前仅在类人行走任务验证，尚未推广到更广泛的机器人或任务；以及跨模拟器的动作空间差异仍需进一步研究。

---

## 154. Human Factors in Cybersecurity in Icelandic Small and Medium-sized Enterprises

**arXiv ID:** 2606.02839 | [PDF](https://arxiv.org/pdf/2606.02839v1)

**作者:** Goda Cicėnaitė `[一作]` (University of Iceland), Helmut Neukirchen `[通讯]` (University of Iceland)

**通讯引用:** 425 | [OpenAlex ID](https://openalex.org/A5087299678)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过对130家冰岛中小企业（SME）和关键基础设施组织开展问卷调查，研究了人因对网络安全的影响，重点评估了员工培训、招聘、文化等管理层视角的挑战与障碍。

**💡 创新点**

创新点在于首次系统性地把人因研究应用于冰岛特定的SME环境，提出了针对性培训、政府支持和安全文化建设的实务建议，并与国际文献对比验证了人因普遍性与本土差异。

**🔧 技术方法**

采用的技术包括结构化在线问卷（封闭与开放式问题）、统计分析（卡方检验、描述性统计）与主题分析（使用Atlas.ti），数据可视化则依赖RStudio。

**📊 数据集**

使用的数据集为调查问卷的原始答复，共130份有效样本，覆盖了行业、组织规模、管理层职能等维度。

**📈 对比分析**

在方法比较方面，本文主要与已有文献中的调查结果和定性访谈进行对照，未引入实验或机器学习模型；结果显示SME在人因方面面临培训不足、资金与人才缺乏等共性问题，且与其他国家的调查结果在挑战维度上保持一致。

**⚠️ 局限性**

局限性包括：响应率低（19%），可能导致样本偏倚；数据主要来自管理层，缺乏一线员工视角；调查为自评，受主观认知与社会期望偏差影响；未进行纵向跟踪，无法评估干预措施的长期效果。

---

## 155. Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR

**arXiv ID:** 2606.03087 | [PDF](https://arxiv.org/pdf/2606.03087v1)

**作者:** Chuanyu Qin `[一作]` (Chinese Academy of Sciences), Zheng Lin `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 7938 | [OpenAlex ID](https://openalex.org/A5027735250)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在RLVR训练中引入可持续记忆的回顾机制，防止已掌握样本的遗忘。

**💡 创新点**

提出纠正集周转理论，阐明学习与遗忘的双向流动，并基于低成本修复窗口设计了零开销的自纠正回顾队列。

**🔧 技术方法**

基于RLVR的GRPO/DAPO框架，使用多路返回评估、FIFO回顾队列、预渲染批量替换等技术。

**📊 数据集**

使用包含20个基准的多模态数据集，包括图像-文本推理、视频推理和文本数学推理，如MMFineReason-123K、MathVista、VideoBench、AIME等。

**📈 对比分析**

与GRPO、DAPO及多种回放方法比较，平均提升约2–4个百分点；在数学推理和视频推理任务上提升更显著；在不同模型和算法上均保持稳健。

**⚠️ 局限性**

依赖可验证奖励的准确性，适用于7-8B规模模型，固定回顾频率可能不足以最优，应进一步探索自适应调度；残余的纠正集回退仍存在。

---

## 156. Hierarchical RBF-KAN and RBF-SKAN Architectures for Multidimensional Function Approximation and Random Field Learning

**arXiv ID:** 2606.02936 | [PDF](https://arxiv.org/pdf/2606.02936v1)

**作者:** Mingtao Xia `[一作]` (University of Houston), Qijing Shen `[通讯]` (University of Oxford)

**通讯引用:** 14 | [OpenAlex ID](https://openalex.org/A5104326364)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了层次化 RBF-KAN 和 RBF-SKAN 两种新型神经网络架构，用于多维确定性函数逼近和随机场学习。

**💡 创新点**

创新点在于：1) 采用基于 Kolmogorov–Arnold 表示的层次化块结构，显著提升多维函数逼近精度；2) 在层次化 RBF-KAN 中引入 ResNet 技术，进一步稳定训练；3) 推导了这两种架构的通用逼近定理，并证明可部分缓解维数灾难；4) 对 RBF-SKAN 证明在 Wasserstein‑2 度量下可逼近随机场模型。

**🔧 技术方法**

主要技术包括：高斯核 Radial Basis Function 激活、Kolmogorov–Arnold 结构化层、残差网络（ResNet）模块、Wasserstein‑2 距离与局部平方 W₂ 损失、随机尺度参数引入以及数值实验中使用的自动微分求解 ODE。

**📊 数据集**

实验使用合成数据：①多维高振荡函数（x∈[-3,3]^d，d=1..6）；②Lorenz 系统轨迹（由正态分布初始值产生的 60 条轨迹集）；③随机场模型（含正弦、余弦与噪声项，维度 d=1..5）。

**📈 对比分析**

与多种基线方法对比（RBF‑MLP、RBF‑KAN、标准 MLP、Spline KAN、Tanh‑KAN、CNF、CVAE），实验结果显示：层次化 RBF‑KAN 在高维下误差远低于对手，并且 ResNet 的加入显著提升准确率；层次化 RBF‑SKAN 在均值与标准差预测上均优于 CNF 和 CVAE，尤其是在维度增大时保持稳定，但训练耗时相对更长。

**⚠️ 局限性**

局限性包括：①训练时间随网络深度和维度显著增长，尤其是 RBF‑SKAN 需计算 W₂ 距离；②理论误差上界仅为定性，缺乏精确的数值估计；③当前架构仅针对连续回归任务，尚未扩展到分类或离散输出；④需要精细调参（层数、RBF 数量等）以获得最佳性能。

---

## 157. From Non-Convex to Strongly Convex: Curvature-Adaptive FTPL for Online Optimization

**arXiv ID:** 2606.02948 | [PDF](https://arxiv.org/pdf/2606.02948v1)

**作者:** Moses Charikar `[一作]` (Stanford University), Ambuj Tewari `[通讯]` (University of Michigan)

**通讯引用:** 9471 | [OpenAlex ID](https://openalex.org/A5051918150)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种自适应FTPL算法（AdaFTPL），在在线非凸Lipschitz损失下通过时间变化的噪声尺度实现曲率自适应；

**💡 创新点**

创新点在于：①首次在非凸环境下将FTPL与曲率自适应结合，实现在最坏情况O(√T)与充分曲率下O(log T)之间平滑过渡；②利用元在线学习（Follow‑the‑Leader）自动选择噪声尺度；③给出了匹配的下界证明，证明该方法最优；

**🔧 技术方法**

主要技术包括：FTPL与指数扰动、近似离线优化oracle、稳定性/扰动分解、基于强凸性与Lipschitz连续性的 regret 上界、以及对噪声尺度的元学习分析；

**📊 数据集**

实验采用合成的二次函数序列（随机系数和可变曲率）进行评估，未使用真实数据集；

**📈 对比分析**

与传统FTPL（固定噪声）和FTL比较，AdaFTPL在曲率低时与FTPL相当，在曲率高时与FTL（log T）相近；实验曲线显示其在不同曲率变化下均能保持最坏情况O(√T)并自适应地接近O(log T)；

**⚠️ 局限性**

局限性包括：需要访问近似优化oracle；算法复杂度随oracle调用次数而升；实验仅在合成数据上验证，未评估在实际非凸问题中的表现；参数α、β的设置与噪声尺度的选择仍需依赖对曲率的估计。

---

## 158. MARIO: Motion-Augmented Real-Time Multi-Sensor Inertial Odometry

**arXiv ID:** 2606.02996 | [PDF](https://arxiv.org/pdf/2606.02996v1)

**作者:** Yiquan Li `[一作]` (Northwestern University), Karan Ahuja `[通讯]` (Northwestern University)

**通讯引用:** 1743 | [OpenAlex ID](https://openalex.org/A5010929285)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出MARIO框架，将单IMU姿态先验和多传感器融合集成到现有惯导里程计中，以实现轻量化的人类运动跟踪。

**💡 创新点**

创新点在于通过学习IMU推断的姿态先验将人体运动动力学嵌入惯导轨迹估计，并在AR眼镜可用的磁力计、气压计和二级IMU上实现高效的多模态融合。

**🔧 技术方法**

使用轻量化PoseNet（卷积+GRU）生成SMPL下肢姿态先验，mid‑level特征融合模块以及改进的姿态约束与残差回归/ EKF框架，结合SMPL 6D表示。

**📊 数据集**

主要在Nymeria大规模人类活动数据集上训练，并在Aria Everyday Activities和TLIO数据集上进行跨域验证。

**📈 对比分析**

与AirIO、TLIO、EqNIO、RoNIN‑LSTM四个基线对比，Nymeria上整体ATE下降约35‑44%，漂移率降至2‑3%，Aria上同样保持显著提升，最优模型单帧推理约133M FLOPs，帧率315FPS。

**⚠️ 局限性**

局限在于PoseNet增加额外延迟，磁力计易受干扰，气压计受天气漂移，缺乏自适应权重和在线校准，且在资源受限的可穿戴设备上的进一步优化尚未完成。

---

## 159. MultiTurnPSB: Evaluating Multi-Turn Jailbreak Attacks an dClassifier-Based Defenses for Medical AI Safety

**arXiv ID:** 2606.02630 | [PDF](https://arxiv.org/pdf/2606.02630v1)

**作者:** Anushka Sheoran `[一作]` (University of Pennsylvania), Yiduo Hao `[通讯]` (University of Pennsylvania)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MultiTurnPSB 四轮医患安全评估框架，并在其中对 GPT‑4.1‑mini 与 Claude Sonnet 4.5 进行固定模板、模板自适应与实时对抗攻击的实验，研究多轮对话中模型安全性的演化。

**💡 创新点**

创新点在于：①将单轮医患安全评估扩展为四轮多轮对话；②揭示了模型在不同攻击模式下安全性的四种演化轨迹；③发现两要素攻击（紧急情境+医学权威）是致命违规的主要诱因；④使用输入侧分类器即使在准确率显著下降的情况下也能将第四轮不安全率降低 52%点。

**🔧 技术方法**

主要技术包括：对抗式对话生成（固定模板、模板自适应、实时攻击），GPT‑4o‑mini 作为判别者与攻击者；六分类输入侧分类器（结合 PSB 五类风险与 XSTest 安全范例）；统计分析（χ²、p 值）和轨迹签名识别。

**📊 数据集**

使用了 PatientSafetyBench（PSB）466 条患者级医患安全提示（包含有害建议、误诊、无执照实践、健康误传、歧视等五类），以及 100 条 XSTest 纯安全示例，构建成四轮对话形式的 MultiTurnPSB。

**📈 对比分析**

通过对三种攻击模式下的四轮不安全率进行比较，发现实时攻击导致 GPT‑4.1‑mini 在第四轮不安全率高达 78.8%，而同一攻击下 Claude Sonnet 4.5 仅 4.1%，二者在单轮基线上无显著差异但多轮后差距扩展至 19 倍；输入侧分类器将第四轮不安全率从 78.8% 降至 26.6%，显示显著性能提升。

**⚠️ 局限性**

局限性包括：基准为人工合成对话，可能不代表真实患者交流；仅使用英文文本；主实验仅评估 GPT‑4.1‑mini，结果对其他模型的推广性未知；GPT‑4o‑mini 作为判别者与攻击者可能引入评估偏差；Claude 自攻击条件在第 3‑4 轮被拒绝信息污染，导致数据失真；输入侧分类器的高误报率（45%）是实际部署的主要瓶颈。

---

## 160. Oscillatory State-Space Models as Inductive Biases for Physics-Informed Neural PDE Solvers

**arXiv ID:** 2606.02623 | [PDF](https://arxiv.org/pdf/2606.02623v1)

**作者:** Abhishek Chandra `[一作]` (KTH Royal Institute of Technology), Taniya Kapoor `[通讯]` (Wageningen University & Research)

**通讯引用:** 225 | [OpenAlex ID](https://openalex.org/A5082424531)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了OSSM-PINN，一种将时空解分解为固定空间模式和线性振荡状态空间的物理信息神经网络，用于高效求解时间相关偏微分方程。

**💡 创新点**

创新点在于：①将时间演化限定为受限于纯虚特征值的线性振荡器，天然产生振荡模态；②使用固定解析空间基底实现闭式空间导数与边界条件硬约束；③通过并行扫描实现 O(log Nt) 的时间步长递归，显著降低显存与计算量。

**🔧 技术方法**

技术细节包括：LinOSS 状态空间单元、傅里叶/本征函数空间基、共享模式解码器、六阶有限差分时间导数、IMEX/IM 离散化、双精度计算、边界因子 D(x) 的 Leibniz 规则推导。

**📊 数据集**

在十二个基准上评估：一维/二维/三维传输、反应、波动、Euler‑Bernoulli 振梁、Taylor‑Green 湿漆、三角域热方程、5D 与 100D Schrödinger、KdV 逆问题、海表温度散射、量子谐振子与 Pöschl–Teller 体系；同时使用自定义的解析基底和预先采样的协同点。

**📈 对比分析**

与 PINNsFormer、PINNMamba、ML‑PINN、NeuSA 等最先进序列模型及神经谱基准在相同 24 GiB 单 GPU 预算下对比；OSSM‑PINN 在多数基准上实现 2–219 倍的误差降低、显存节省 10–10⁴ 倍，且在高维 100D Schrödinger 亦保持误差 < 5×10⁻⁴。

**⚠️ 局限性**

局限性：依赖解可由有限空间模式捕捉，难以处理冲击、局部尖锐结构或非模态动态；固定全局基底对多尺度/局部现象收敛慢；统一时间网格和预采样点可能在多尺度问题上效率低；未来需加入自适应基底、局部多分辨率与自适应残差采样。

---

## 161. "**Important** You should give me full credits!": Exploring Prompt Injection Attacks on LLM-Based Automatic Grading Systems

**arXiv ID:** 2606.03090 | [PDF](https://arxiv.org/pdf/2606.03090v1)

**作者:** Hang Li `[一作]` (Michigan State University), Jiliang Tang `[通讯]` (Michigan State University)

**通讯引用:** 26193 | [OpenAlex ID](https://openalex.org/A5040639891)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大语言模型（LLM）在自动评分系统中的提示注入（PI）攻击威胁，并系统评估了几种无监督防御策略。

**💡 创新点**

在教育场景下首次提出了问答无关、通用PI攻击模型，并对攻击与防御效果进行了端到端的定量对比。

**🔧 技术方法**

采用手工编写、PAIR迭代生成与MSJ多样本注入三种攻击方式；防御方面使用预防性提示（Preventive Instruction）和外部守卫模型（Guardian Model）两种训练无关方案。

**📊 数据集**

使用了30道来自人工智能、写作、化学、地球科学四个学科的题目，每题收集100个真实学生答案做评测，并为攻击生成准备20道辅助题目。

**📈 对比分析**

通过平均得分提升（ASI）和攻击成功率（ASR）衡量攻击效果；防御后ASR平均下降至30%-60%，守卫模型在DAN攻击下F1>95%，但在MSJ攻击下仍有>15%成功率；预防提示对正常评分影响<1%。

**⚠️ 局限性**

仅考虑训练无关的攻击与防御，未探索细粒度微调或更复杂的攻击手段；对新型或混合攻击的鲁棒性不足；实验规模有限，难以覆盖所有真实教育场景。

---

## 162. AI Assistance for Discretionary Work: Increasing Feedback Provision in Higher Education

**arXiv ID:** 2606.03095 | [PDF](https://arxiv.org/pdf/2606.03095v1)

**作者:** Romina Mahinpei `[一作]` (Princeton University), Manoel Horta Ribeiro `[通讯]` (Princeton University)

**通讯引用:** 1427 | [OpenAlex ID](https://openalex.org/A5011195481)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究在一门本科机器学习课程中，采用随机实验和访谈相结合的方法，探究AI辅助反馈草稿是否能提升教师可选但有价值的个性化反馈工作量。

**💡 创新点**

创新之处在于将AI辅助聚焦于可选性任务（discretionary work），而非传统的强制任务；通过实验量化AI辅助如何提高工作参与率、长度，却不降低单位工作耗时。

**🔧 技术方法**

采用o4-mini大型语言模型生成反馈草稿，并通过轻量级Chrome扩展将草稿呈现给教学助理；实验中对学生提交随机分配“处理”或“对照”组。

**📊 数据集**

使用本课程学生的作业提交数据（约2828条提交，11名助理、88名学生），并收集TAs与学生的问卷与访谈记录。

**📈 对比分析**

与无AI辅助对照组相比，AI辅助显著提高了反馈提供率（+10.81个百分点）和反馈长度（+39.79字符），学生对反馈的有用性评估无显著差异；AI草稿使用率约54%，对照组为0。

**⚠️ 局限性**

局限性包括：仅在单门课程、单一模型与提示上进行，结果可能受课程文化与工作量变化影响；实验未能直接测量认知负担与启动成本；可能存在干预溢出与长期可持续性不足；数据和方法在其他教育场景下的外推性有限。

---

## 163. TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity, and Truthfulness Assessment

**arXiv ID:** 2606.03036 | [PDF](https://arxiv.org/pdf/2606.03036v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 164. Large Byte Model: Teaching Language Models About Compiled Code

**arXiv ID:** 2606.02834 | [PDF](https://arxiv.org/pdf/2606.02834v1)

**作者:** Florian Störtz `[一作]` (CrowdStrike), Edward Raff `[通讯]` (CrowdStrike)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一种字节原生大型语言模型（Large Byte Model，LBM），可直接对可执行二进制文件进行语义分析，并回答自然语言问题。

**💡 创新点**

通过字节词汇扩展技术将专用字节 token 注入基础 LLM，构建文本+字节双模态嵌入，配合两阶段预训练和指令微调，实现了高效的长上下文字节分析能力。

**🔧 技术方法**

使用 BPE 字节分词器、混合嵌入器、FSDP+DeepSpeed 序列并行、cut cross‑entropy、Llama‑3.1‑8B/Mistral‑7B 基座、指令式微调及合成数据生成等技术。

**📊 数据集**

构建了约56 GB 的真实世界二进制数据集（PE/ELF/MACHO，含恶意、无害、广告软件），以及利用 LLM 自动生成的合成带丰富元数据的编译段数据。

**📈 对比分析**

通过文件信息、标签、恶意软件家族分类和 opcode 离群检测四类评估任务与多种前沿 LLM 进行对比；LBM 在家族分类上达 69%–98% 的准确率，文件信息分类 89%–98%，比基线模型提升显著；其 KL 散度显著高于对手，说明对合法/非法字节区分更敏锐。

**⚠️ 局限性**

受 GPU 内存限制，单个字节块上限 256 KB；尚未支持更长文件；评估仅聚焦分类任务，缺乏复杂推理或漏洞归因的高质量数据；过度依赖自动分析可能导致误判。

---

## 165. Memory Retrieval for Changing Preferences

**arXiv ID:** 2606.02976 | [PDF](https://arxiv.org/pdf/2606.02976v1)

**作者:** Yuehan Qin `[一作]` (University of Southern California), Yue Zhao `[通讯]` (University of Southern California)

**通讯引用:** 3632 | [OpenAlex ID](https://openalex.org/A5057711796)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于贝叶斯因子（Bayes factor）的统一框架，实现个性化对话系统的内存访问与检索决策。

**💡 创新点**

通过将记忆回合对答案概率的改进量化为实用性信号，替代传统语义相似度；同时将访问门控与检索排序统一到同一贝叶斯因子上。

**🔧 技术方法**

使用贝叶斯因子作为 saliency 信号，结合阈值门控、训练的控制器和阅读器，并在预训练 LLM 上进行监督微调。

**📊 数据集**

在 MemBench‑Low/High、PersonaMem 和 PrefEval 这四个长上下文个性化基准上进行评估。

**📈 对比分析**

与全上下文、RMM、Mem0、A‑MEM、MemoryBank 等传统检索方法比较，平均准确率提升 19–22 个点，在长上下文隐式偏好任务上显著领先。

**⚠️ 局限性**

对贝叶斯因子计算的近似和跨回合信息聚合仍存在挑战；在短文本或显式检索场景中的提升相对有限。

---

## 166. Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States

**arXiv ID:** 2606.02907 | [PDF](https://arxiv.org/pdf/2606.02907v1)

**作者:** Subramanyam Sahoo `[一作]` (Horizon Research), Divya Chaudhary `[通讯]` (Northeastern University)

**通讯引用:** 5225 | [OpenAlex ID](https://openalex.org/A5048878908)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对 Qwen3-14B 在 LogiQA 2.0、ARC-Challenge 与 αNLI 三类推理任务进行线性探针实验，发现隐藏状态在层 32 处能够完美区分推理类型，但此分离仅由任务格式共变造成，而非真实推理机制。

**💡 创新点**

提出了残差回归格式去混淆流程和随机方向控制的因果驱动法，系统展示高探针准确率可能是伪影，并为解释性研究提供了规范化的去混淆与方向性验证方法。

**🔧 技术方法**

线性探针、Manifold 维数与几何分析、Ridge 回归残差去混淆、基于余弦相似度的 trace‑mode 对齐、以及带随机方向控制的激活 steering 实验。

**📊 数据集**

使用 Qwen3-14B 14B 参数模型；推理数据集包括 LogiQA 2.0（演绎）、ARC-Challenge（归纳）和 αNLI（溯因）。

**📈 对比分析**

对比探针在去混淆前后的准确率（从 100% 降至 ≈ 33% 机会），以及针对性 steering 与随机方向控制的效果（p=0.286，差异不显著）；模型整体在三类任务上的准确率为 86%。

**⚠️ 局限性**

仅评估单一模型；Ridge 回归可能过度去除真实信号；trace‑mode 对齐依赖手工设定的 anchor，可能漏检细微差异；2‑choice 与 4‑choice 的结构差异导致强格式共变；因果实验规模有限；仅使用非思考推理模式，未检验思考模式下的几何差异。

---

## 167. ERP-XTTN: Interpretable Prototype-Guided Cross-Attention for Cross-Subject ERP Classification

**arXiv ID:** 2606.02939 | [PDF](https://arxiv.org/pdf/2606.02939v1)

**作者:** Charlotte Genevier Wyman `[一作]` (University of Colorado Boulder), Leanne Hirshfield `[通讯]` (University of Colorado Boulder)

**通讯引用:** 2418 | [OpenAlex ID](https://openalex.org/A5009602263)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `edb9d762-f411-4838-a852-f2d638b018db` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并评估了一种基于原型的跨注意力架构 ERP‑XTTN，用于实现无校准、可解释的跨受试者 ERP 分类。

**💡 创新点**

创新点在于自动峰值检测生成原型，并使分类完全依赖可解释的注意力路由，首次在多种 ERP 组件下完成离线兼容、无校准的 LOSO 基准。

**🔧 技术方法**

采用跨注意力（仅 query‑key 交叉注意）将 EEG 片段路由至差分波原型，配合因果 IIR 滤波、留一受试者交叉验证，并与 EEGNet 与 xDAWN+RG 进行对比。

**📊 数据集**

在 BNCI Horizon 2020、HRI Cursor 及 ERP CORE 共九个数据集（涵盖 ERN、LRP、ErrP、N170、P300、N2pc、MMN、N400）上进行评估。

**📈 对比分析**

与 EEGNet 与 xDAWN+RG 在 3 通道和完整通道下对比，3 通道平均 AUROC 差距仅 0.018，完整通道为 0.034，显示在最小通道下仍保持竞争力。

**⚠️ 局限性**

主要局限在于时域灵活性和空间利用相关的性能缺口，主要受 SNR 限制；且对某些组件的原型位置偏移说明未必捕捉到最具区分性的信号。

---

## 168. Margin Play: A Multi-Agent System For Public Policy Analysis In The Brazilian Equatorial Margin

**arXiv ID:** 2606.02614 | [PDF](https://arxiv.org/pdf/2606.02614v1)

**作者:** Antonio de Sousa Leitão Filho `[一作]` (Aia Context), Allan Kardec Duailibe Barros Filho `[通讯]` (Universidade Federal do Maranhão)

**通讯引用:** 2381 | [OpenAlex ID](https://openalex.org/A5064755086)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了Margin Play，一个基于多代理强化学习的仿真框架，用于评估巴西赤道边缘海岸石油开发对马拉尼昂州社会经济外部性的影响。

**💡 创新点**

创新点在于：①将六个具有法定使命和不同目标函数的机构（州政府、石油公司、ANP、IBAMA、社区、联邦政府）纳入单一CTDE（集中训练、分散执行）框架；②采用BRO-MARL与分布式TQC critic以提高学习稳定性；③在模型中实现了“MA‑Próspero”结构性干预组合，并通过多情景对比揭示机构配置对福利与环境的交互效应。

**🔧 技术方法**

使用技术包括：多代理强化学习（MARL）、集中训练/分散执行（CTDE）、BRO（Bigger, Regularized, Optimistic）算法、分布式Quantile Critic（TQC）、Python/MLX框架以及GPU加速训练。

**📊 数据集**

数据集主要来自巴西官方来源：ANP石油产量与储量数据、IBAMA许可与监管数据、联邦和州财政法律条款（如Law 9.478/97、Law 12.858/2013）、CPT/INCRA/CIMI 2024的土地冲突与原住民暴力统计、H‑TERR‑2区域校准数据，以及国际油价和宏观经济基础参数。

**📈 对比分析**

通过在六种对照情景（基线、无 earmarking、油价冲击、MA‑Próspero）下进行60,000个训练周期，评估了福利、社区回报、IBAMA回报与环境负债四项指标；MA‑Próspero情景实现了约+17.5%福利提升、+21.3%社区福利提升及环境负债显著下降，且学习过程收敛稳定。

**⚠️ 局限性**

主要限制包括：缺乏对单个干预项的系统消融分析、使用γ=0.95的折扣因子可能低估长期人力资本效应、部分结构性干预（如宏观经济稳定、领土规范化）的理论假设与实证验证不足、模型未考虑宪法与代际公平等非经济维度，且不能直接用于制定精确政策决定。

---

## 169. Cost-Aware Query Routing in RAG: Empirical Analysis of Retrieval Depth Tradeoffs

**arXiv ID:** 2606.02581 | [PDF](https://arxiv.org/pdf/2606.02581v1)

**作者:** Sanjay Mishra `[一作]` `[通讯]`, Sanjay Mishra

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并评估了成本感知检索路由框架CA‑RAG，在RAG系统中根据每条查询的复杂度动态选择检索深度，以平衡答案质量、计费token和端到端延迟。

**💡 创新点**

创新点在于引入离散检索‑生成策略包和基于单一效用函数的路由决策，允许运营者仅通过调整权重即可在成本、延迟和质量之间灵活切换，而无需重构系统。

**🔧 技术方法**

使用了FAISS密集检索、OpenAI embedding/chat API、离散策略包定义、效用函数加权决策、完整的日志化CLI实验框架。

**📊 数据集**

实验基于15句技术语料库和28个自然语言查询的人工构造基准，用以控制实验变量。

**📈 对比分析**

通过与多种固定检索配置及权重敏感基线对比，结果显示平均计费token降低约26%，平均延迟降低约34%，而质量保持与最优固定配置相当。

**⚠️ 局限性**

局限性包括：质量先验手工设定、语义质量评估缺失、基准规模有限、检索深度离散、仅适用于OpenAI API、统计功效不足。

---

## 170. WISE-HAR: A Generalizable Ensemble Deep Learning Framework for WiFi-Based Human Activity Recognition

**arXiv ID:** 2606.02974 | [PDF](https://arxiv.org/pdf/2606.02974v1)

**作者:** Maheen Arshad `[一作]` (National University of Sciences and Technology), Muhammad Khuram Shahzad `[通讯]` (National University of Sciences and Technology)

**通讯引用:** 3048 | [OpenAlex ID](https://openalex.org/A5035787368)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过集成学习、数据增强和跨场景评估等方法，提出了一种能够识别WiFi信号下的三类人体活动（无存在、行走、行走+挥臂）的深度学习框架。

**💡 创新点**

创新点在于①使用五种不同CNN架构进行软投票集成，显著降低模型方差；②采用时间扭曲、频率遮蔽和噪声注入等针对WiFi信号的激进数据增强，解决小数据集过拟合问题；③在LOS→NLOS、Biquad→PIFA等跨场景/硬件条件下进行系统评估，展示出优秀的泛化能力。

**🔧 技术方法**

技术包括CNN集成学习（Deep CNN、Wide CNN、MobileNetV2、ResNet50V2、EfficientNetB0）、基于TensorFlow的实时数据增强、迁移学习（ImageNet预训练权重）、Soft Voting投票策略以及传统机器学习（Random Forest）与深度模型的对比实验。

**📊 数据集**

使用公开的Wallhack1.8k WiFi频谱图数据集，其中包含三类活动的约1,104张RGB图像，按LOS/NLOS和Biquad/PIFA两种环境/天线条件划分。

**📈 对比分析**

与单模型、无增强及传统机器学习方法对比，集成模型在LOS/Biquad上实现94.87%的测试准确率，单模型最高为94.21%；数据增强后Random Forest准确率从60%提升至95%；跨场景/天线测试仅出现1.37%和2.07%的准确率下降，说明模型具有良好的鲁棒性。

**⚠️ 局限性**

局限性包括数据量仍偏小、仅覆盖三种活动、可能存在个体与环境偏差、未进行实时部署与在线学习实验，以及缺乏系统性消融研究来量化各改进贡献。

---

## 171. How Quantization Changes Interpretable Features: A Sparse Autoencoder Analysis of Language Models

**arXiv ID:** 2606.03002 | [PDF](https://arxiv.org/pdf/2606.03002v1)

**作者:** Evan Duan `[一作]` `[通讯]` (University of Michigan), Evan Duan (University of Michigan)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究量化后大型语言模型的稀疏自编码器（SAE）特征是否保持不变，测量不同位宽下特征的存活率。

**💡 创新点**

发现量化导致特征损失呈梯度分布，且可由全精度统计量（尤其峰值激活）高度预测；并且任务层面指标（如困惑度）可能掩盖大规模特征损坏。

**🔧 技术方法**

使用固定的 SAE 编码器、Per‑output‑channel RTN 量化、基于 Pearson 相关系数的特征存活度量、对比稀疏剪枝、以及逻辑回归预测模型。

**📊 数据集**

实验数据集为 WikiText‑2‑raw，模型为 Pythia‑70M 和 Gemma‑2‑2B，配合公开的残差流 SAE 字典。

**📈 对比分析**

通过比较不同位宽（INT8‑INT4）下的存活率、均值相关系数、困惑度变化以及与匹配困惑度剪枝的重叠度（Jaccard、Spearman）来评估方法；结果显示 INT6 时约 60% 的特征存活，特征损失与困惑度不一致，且量化与剪枝损坏的特征集高度重叠。

**⚠️ 局限性**

局限性包括仅评估两种模型族、仅使用残差流 SAE、仅模拟 RTN 量化、固定读出层和令牌预算，无法推广到其他架构、数据集或量化策略。

---

## 172. Tiny Collaborative Inference for Occlusion-Robust Object Detection

**arXiv ID:** 2606.02894 | [PDF](https://arxiv.org/pdf/2606.02894v1)

**作者:** Chieh-Tung Cheng `[一作]` (Imperial College London), Eiman Kanjo `[通讯]` (Nottingham Trent University)

**通讯引用:** 2553 | [OpenAlex ID](https://openalex.org/A5061493803)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在低功耗 MCU 上实现可在现场运行的目标检测，并通过多视角协同推理提升遮挡鲁棒性。

**💡 创新点**

提出将 MCUNet + YOLOv2 与 TensorFlow Lite INT8 量化结合，并在极低端硬件上验证 Weighted Boxes Fusion（WBF）在遮挡场景下的显著性能提升，首次展示三视角协同推理在此类设备上的可行性。

**🔧 技术方法**

使用 MCUNet 轻量级骨干网络、YOLOv2 检测头、TensorFlow Lite 量化、WBF 以及 Wi‑Fi 低功耗通信协议；在 Coral Dev Board Micro 上部署。

**📊 数据集**

CO3D（车类子集）用于遮挡实验；PASCAL VOC 用于模型预训练；在两台设备上收集的实时 Himax 摄像头图像用于硬件验证。

**📈 对比分析**

通过与单视角、特征级融合两种基线比较，WBF 在 (30%,50%) 视角遮挡下提升 0.2736 mAP，三视角进一步提升至 0.3827 mAP；在两台实际板卡的 Wi‑Fi 现场实验中，融合后帧级覆盖率提升 29.8%，通信能耗仅占推理能耗的 0.003%。

**⚠️ 局限性**

局限包括仅在单类别、约 100 张图像的小数据集上评估；特征级融合实现过于简化；DFL 在非 iid 数据下收敛不佳；帧率仅 ~0.36 FPS，未使用 Edge TPU 加速；缺乏时间戳同步，难以扩展到更多设备。

---

## 173. Secure AltDA Integration for Ethereum L2s: An End-to-End Validation Framework

**arXiv ID:** 2606.03010 | [PDF](https://arxiv.org/pdf/2606.03010v1)

**作者:** Bowen Xue `[一作]` (Eigenlabs Inc), Samuel Laferriere `[通讯]` (Independent)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一套端到端的验证框架，用以规范以太坊 L2 与外部 AltDA 系统的安全交互，并通过模型化、攻击分类与实例评估阐明了缺失验证义务所导致的安全风险。

**💡 创新点**

创新点在于：1) 将 L1 inbox、AltDA 提交、外部数据与 Rollup payload 四个领域用类型化、确定性、全函数方式严格连接；2) 明确列出必需的语义义务（解析、DA 验证、时效、绑定、唯一性与总失败处理）并将其映射到 L2 的不同实现路径；3) 通过攻击分类展示缺失义务的具体后果；4) 对 Celestia、EigenDA、Avail‑ZKsync 等主流 AltDA 集成进行实证评估，揭示安全缺陷。

**🔧 技术方法**

主要技术包括：类型化验证模型、确定性全函数 F、绑定与时效谓词、交互式 fault proof、可验证证明 (validity proof)、主动 inbox 验证、以及对外部 DA 验证器的接口调用。

**📊 数据集**

本文未使用传统实验数据集；评估基于对公开规范、代码库、审计报告等文档的静态分析，并对代表性 AltDA 集成方案（Celestia‑Blobstream、EigenDA‑Canoe、Avail‑Vector 等）进行对比。

**📈 对比分析**

通过框架对比不同实现模式，评估哪些义务在 inbox、DA verifier、derivation pipeline、fault proof VM、validity proof 等路径被覆盖；结果显示：某些方案仅在验证器层完成合法性，而缺失绑定或时效检查；其他方案通过 eager inbox 或交互式证明实现完整义务。性能方面未给出数值，但指出 eager 验证成本高，交互式证明耗时较长，validity proof 在一次性证明中可完成全部检查。

**⚠️ 局限性**

局限性：1) 只聚焦于 L2 与 AltDA 的交互边界，未分析 AltDA 本身的共识安全与可用性问题；2) 未给出单一最佳架构，具体实现需根据成本、证明系统与 inbox 设计权衡；3) 评估基于公开资料，缺少实测攻击演示；4) 假设 DA 系统的加密与共识安全均成立。

---

## 174. ChatHealthAI: Aligning Electronic Health Record Representations with Large Language Models for Grounded Clinical Reasoning

**arXiv ID:** 2606.02802 | [PDF](https://arxiv.org/pdf/2606.02802v1)

**作者:** Bo-Hong Wang `[一作]` (McGill University), Yue Li `[通讯]` (McGill University)

**通讯引用:** 10817 | [OpenAlex ID](https://openalex.org/A5100387744)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了ChatHealthAI多模态框架，将结构化纵向EHR表示与冻结LLM对齐，生成可解释的临床预测与推理。

**💡 创新点**

引入任务感知感知器重采样器将EHR嵌入对齐至LLM语义空间，实现自然语言推理与精准预测的结合。

**🔧 技术方法**

使用CLMBR‑T‑Base预训练EHR表示、DeepSeek‑R1‑Distill‑Qwen‑14B冻结LLM、Perceiver重采样器、教师模型生成推理目标以及next‑token预测训练。

**📊 数据集**

在EHRSHOT基准上评估，涉及LOS、ICU入院、30日再入院三项预测任务。

**📈 对比分析**

与CLMBR+Linear、零样本与微调的多款LLM（Llama‑3.1、BioMistral、DeepSeek）对比，ChatHealthAI在F1上最高，并在多维推理评估中获得最佳质量与实用性。

**⚠️ 局限性**

仅在单一EHRSHOT数据上验证，推理评估依赖LLM评审、教师模型生成说明可能不准确，且与最新LLM架构兼容性有限。

---

## 175. D-Judge: Disrupting Multi-Turn Jailbreaks using Semantics-Preserving Output Rewriting

**arXiv ID:** 2606.02640 | [PDF](https://arxiv.org/pdf/2606.02640v1)

**作者:** Huanli Gong `[一作]` (University of California, Berkeley), N. Benjamin Erichson `[通讯]` (International Computer Science Institute)

**通讯引用:** 1200 | [OpenAlex ID](https://openalex.org/A5007032334)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种在LLM API边界插入语义保持的输出重写模块，干预多轮攻击中的评估反馈，阻断攻击者的迭代优化。

**💡 创新点**

创新点在于通过构造语义等价但评估分值不同的响应对，训练重写器以误导评估模型的反馈，同时保持原意；并采用两阶段训练（SFT+DPO）实现对评估模型的无偏偏好优化。

**🔧 技术方法**

使用监督微调、直接偏好优化（Direct Preference Optimization）以及双向推理的语义门控（Semantic Gate）对重写模型进行训练，并利用LLM评估模型（如Llama Guard）和NLI模型进行安全性与语义一致性检测。

**📊 数据集**

构建了约18k条语义等价响应对数据集，采样自PKU-SafeRLHF，利用Llama Guard判定安全概率、NLI判定语义一致性。

**📈 对比分析**

在HarmBench上与多种单轮与多轮攻击以及多种防御方法对比，D-Judge将多轮攻击成功率平均降至8.6%，显著优于Guard、检测、输入/输出预处理等传统防御；在MT-Bench和IFEval上保持近似原模型性能，安全税低。

**⚠️ 局限性**

局限在于额外的重写步骤导致约5.4 ms/token的延迟和计算成本；若攻击者已在不依赖迭代反馈的模型上预先优化提示，D-Judge效果有限。

---

## 176. $Ψ$-Bench: Evaluating Persona-Sensitive Influencing in Persuasive Dialogues

**arXiv ID:** 2606.02754 | [PDF](https://arxiv.org/pdf/2606.02754v1)

**作者:** Peixuan Han `[一作]` (University of Illinois Urbana Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Ψ-Bench基准，用来评估大语言模型在模拟个性化客户对话中主动说服的能力。

**💡 创新点**

创新点在于：①构建了三类现实场景（观点辩论、心理咨询、日常请求）并为每个查询生成基于对话历史的个性化档案；②设计了基于LLM的客观评判器（质量、个性化、说服效果）；③引入RL训练的“档案分析器”可在对话中主动推断用户档案，提升说服效果。

**🔧 技术方法**

使用的技术包括：大型预训练语言模型（如DeepSeek、Qwen、Gemini、GPT‑5等）作为被评估模型；LLM评判器和模拟客户端；RL（GRPO）训练的档案分析器；BGE-M3编码器计算档案相似度。

**📊 数据集**

数据集来源：Reddit “Change My View”（CMV）对话、CounselBench心理咨询数据、以及GPT‑4o生成的日常请求示例，约700条真实用户查询并配备合成个性化档案。

**📈 对比分析**

比较方法：对10款前沿LLM在Ψ-Bench上分别评估质量、个性化、说服效果，使用LLM评判器打分并与人类标注对齐；实验显示大模型在质量上表现优异（>7分），但说服效果仍低于6分，且提供完整档案可平均提升18.24%说服分。档案分析器在无档案情况下也能显著提升说服效果。

**⚠️ 局限性**

局限性：①模拟客户虽然多样，但未能覆盖所有真实世界用户的教育、文化、社会经济差异；②数据规模有限，未能对同一查询下不同个性化档案的表现做全面分析，未来可扩展至更大组合。

---

## 177. Hand Trajectory Fusion for Egocentric Natural Language Query Grounding

**arXiv ID:** 2606.02962 | [PDF](https://arxiv.org/pdf/2606.02962v1)

**作者:** Enmin Zhong `[一作]` (Universidad Politécnica de Madrid), Narciso García `[通讯]` (Universidad Politécnica de Madrid)

**通讯引用:** 4795 | [OpenAlex ID](https://openalex.org/A5060713777)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种手部轨迹编码器与跨注意力融合策略，用于提升第一人称视角视频中的自然语言查询（NLQ）定位

**💡 创新点**

将稀疏的手部骨架序列转化为语义化运动特征，并通过可学习门控的跨注意力融合，使手部运动信息在不影响视频-文本对齐的情况下自适应地贡献给模型

**🔧 技术方法**

使用了Spatio‑Temporal Transformer 对手骨架进行空间聚合与时间建模，跨注意力与自注意力结合的融合模块，以及预训练的 InternVideo、EgoVLP 与 CLIP 文本编码器

**📊 数据集**

在 Ego4D NLQ v2 数据集上进行评估，包含 13,435 训练和 4,552 验证的查询-视频对

**📈 对比分析**

与 GroundNLQ 基线对比，手轨迹模型在 Hand‑Object Interaction 与 Quantity/State 两类查询上分别提升 2.54 和 4.32 的 R1@IoU=0.3，整体 R1@0.5 提升 1.39，显示出明显的性能提升

**⚠️ 局限性**

手部检测的稀疏性（平均 41% 的帧检测到手）限制了轨迹分支的贡献，未来需改进手部检测以进一步提升效果

---

## 178. Building Better Activation Oracles

**arXiv ID:** 2606.02609 | [PDF](https://arxiv.org/pdf/2606.02609v1)

**作者:** Jan Bauer `[一作]` (University College London), Neel Nanda `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

改进并训练了 Activation Oracles（AO），解决幻觉、模糊等问题，并发布了首个综合评估套件 AObench。

**💡 创新点**

创新点在于：①构造更高质量、针对性强的对话式 QA 数据集，避免文本逆转；②采用多层激活输入并调节注入强度；③使用 on‑policy chain‑of‑thought 生成的数据进行 past/future‑lens 训练；④系统化地评估 AO 并公开 benchmark。

**🔧 技术方法**

技术方法包括：norm‑matched 注入（第二层后插入激活）、多层激活提取、对话式 QA 数据生成（Sonnet 4.6）、过去/未来‑lens 自监督训练、注入强度微调，以及对幻觉与模糊的量化评估。

**📊 数据集**

使用数据集：①新构建的对话式 QA 数据集（cot‑oracle‑convqa‑chunked‑sonnet）；②替代 FineWeb 的 on‑policy chain‑of‑thought rollouts；同时保留原始的二分类任务和 self‑supervised 目标。

**📈 对比分析**

评估方法：在 AObench 进行 chance‑adjusted 打分，关注幻觉、模糊和文本逆转防护等指标。改进后，AObench 分数从 +0.244 提升至 +0.435，幻觉率和模糊度均显著下降，整体性能显著提升。

**⚠️ 局限性**

局限性：仍有较高幻觉率、评估过程受 LLM 判定噪声影响；对文本逆转的抵抗仍有限；训练规模有限（50M tokens），对不同模型和应用的泛化性尚未完全验证。

---

## 179. AdaWeather: Adaptively Mixing Probabilistic Weather Forecasts with Logarithmic Regret

**arXiv ID:** 2606.02663 | [PDF](https://arxiv.org/pdf/2606.02663v1)

**作者:** Saptarishi Dhanuka `[一作]` (Ashoka University), Sandeep Juneja `[通讯]` (Ashoka University)

**通讯引用:** 1819 | [OpenAlex ID](https://openalex.org/A5109269131)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种离线‑在线混合框架，先用U‑Net学习历史误差结构，随后将该U‑Net预测视为额外专家并在在线阶段通过VT‑MOS算法进行概率混合，从而对印度地区的2米温度进行改进的天气预测。

**💡 创新点**

创新点包括：① 证明CRPS在混合专家设置下可mixable，并给出对应的闭式预测；② 设计VT‑MOS（Vyugin‑Trunov混合专家聚合）实现对最佳混合的对数型 regret 上界；③ 将离线学习的专家与在线聚合相结合，兼顾历史结构与自适应性。

**🔧 技术方法**

技术主要包括：U‑Net离线监督学习；Vyugin‑Trunov混合专家在线聚合算法；CRPS损失函数与其mixability分析；Monte‑Carlo采样近似高维积分；理论证明与 regret 分析。

**📊 数据集**

使用印度0.25°分辨率的2米温度数据，时间范围2019‑2026年，训练集2019‑2022年，验证集2023年，测试集2024‑2025年；所有模型预测与ERA5再分析对齐。

**📈 对比分析**

与单模型、等权平均、Vovk‑AA、MoWE、U‑Net离线等基线比较。VT‑MOS＋U‑Net在所有 lead time 的平均CRPS为0.503，显著低于其他方法；其累计遗憾曲线表现为对数增长，优于单专家或其他在线聚合。

**⚠️ 局限性**

局限性包括：训练周期短、使用的模型数量有限；积分近似依赖Monte‑Carlo，可能导致数值误差；仅在印度地区验证，缺乏跨区域或多源数据的泛化；未对长时间序列或更大数据集进行测试。

---

## 180. FGRPO: Federated GRPO with Adaptive Aggregation on Non-IID Data

**arXiv ID:** 2606.03094 | [PDF](https://arxiv.org/pdf/2606.03094v1)

**作者:** Pengyu Chen `[一作]` (Shandong University), Feng Li `[通讯]` (Shandong University)

**通讯引用:** 70181 | [OpenAlex ID](https://openalex.org/A5100448879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种联邦学习框架FGRPO，利用组相对策略优化（GRPO）在不需要价值网络的前提下对多模型推理能力进行分布式微调。

**💡 创新点**

创新点在于将GRPO迁移到联邦场景，设计了基于相对性能提升（RPG）的自适应聚合机制，能够在任务奖励尺度差异显著的非IID数据下实现鲁棒收敛，并给出了非凸收敛理论保证。

**🔧 技术方法**

技术上使用了GRPO、联邦学习（FedAvg/FedProx/SCAFFOLD对比）、RPG加权聚合、Adam风格自适应更新、LoRA参数高效微调、指数移动平均基准和Boltzmann温度调度等。

**📊 数据集**

实验数据集包括OpenR1和GEOQA两大推理/几何问答基准，并在Qwen2.5‑VL‑3B/7B、Qwen3‑VL‑4B、Llama‑3.2‑11B三类模型上进行评估，数据通过Dirichlet分布模拟高度非IID分布。

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD等传统联邦强化学习基线比较，FGRPO在所有模型和数据集上均取得最高准确率，尤其在最难的“Hard”分区提升约3–5个百分点，整体平均提升超过1%。

**⚠️ 局限性**

局限性包括：仍存在由异构导致的不可消除误差上限；对奖励信号的二元化设计可能限制更细粒度的学习；实验仅覆盖少数客户端和特定模型，未充分验证在大规模联邦场景中的可扩展性和超参数敏感性。

---

## 181. MOSAIC: Efficient Mixture-of-Agent Scheduling via Adaptive Aggregation and Inference Concurrency

**arXiv ID:** 2606.03014 | [PDF](https://arxiv.org/pdf/2606.03014v1)

**作者:** Saptarshi Mitra `[一作]` (University of California, Irvine), Sitao Huang `[通讯]` (University of California, Irvine)

**通讯引用:** 824 | [OpenAlex ID](https://openalex.org/A5050532440)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出 MOSAIC 框架，针对 Mixture-of-Agents（MoA）推理中的 GPU 调度瓶颈，提供了基于离线模型性能剖面和整数线性规划（ILP）的专家调度方案，并引入了基于专家投票一致性的置信门控聚合技术，显著提升了推理吞吐量。

**💡 创新点**

创新点包括：① 将专家模型放置、工作负载分配与复制决策统一为 ILP 规划，自动平衡模型加载成本与生成成本；② 开发置信门控聚合器，利用专家一致性跳过昂贵的聚合 LLM，既降低聚合时间又保持准确率；③ 结合两种技术实现端到端性能提升，首次在 MoA 场景中系统评估。

**🔧 技术方法**

技术手段包括：离线模型输出 token 统计、生成时间回归、ILP 调度（使用 CP-SAT 解决器）、vLLM 进行多 GPU 端到端推理、置信门控聚合器（基于投票一致性）和 Tensor‑Parallel 聚合 LLM。

**📊 数据集**

实验数据集为 MMLU‑Pro、MedMCQA 和 GPQA 三大多领域推理基准，涵盖数学推理、医学问答与通用知识问答。

**📈 对比分析**

与传统 round‑robin（RR）调度基线对比，MOSAIC 在 4‑GPU 系统上实现专家阶段 1.69–2.54× 加速、聚合阶段 1.74–4.23× 加速，整体端到端 1.71–2.34× 加速；同时准确率保持在 ±0.1pp 以内。

**⚠️ 局限性**

主要局限包括：置信门控聚合的阈值和判断标准需要手工设置且在推理期间保持静态；ILP 调度对大规模专家池或更复杂的任务分布扩展性有限；当前实现假设每个专家可完整放入单 GPU，无法直接处理跨 GPU 的大模型。

---

## 182. Heterogeneous Mapping for Analog In-Memory Computing Accelerators: A Unified Workflow

**arXiv ID:** 2606.02672 | [PDF](https://arxiv.org/pdf/2606.02672v1)

**作者:** Corey Lammie `[一作]` `[通讯]` (IBM Research), Corey Lammie (IBM Research)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对现有AIMC加速器的异构映射方法进行分类，提出统一的四阶段映射工作流，并首次为GPT‑2（decoder‑only Transformer）生成了基于AIMC噪声模型的精度灵敏度剖面，揭示了投影级映射的重要性。

**💡 创新点**

创新点包括：①构建涵盖粒度、策略和模型支持的映射方法分类体系；②提炼出可适用于任意异构AIMC平台的统一四阶段映射工作流；③首次为decoder‑only Transformer提供AIMC精度敏感度分析，为后续映射方法提供可复用的输入；④通过实验揭示单个投影对整体精度的显著影响，提示投影级映射优于传统的层级或通道级映射。

**🔧 技术方法**

使用了：①基于AIHWKIT的PCM噪声模拟（高斯权重裁剪、8位ADC/DAC、权重编程噪声σ_w=0.023）；②对GPT‑2 49个权重投影分别做单层精度敏感度分析（10个噪声实例）；③多种已有映射方法的对比框架（如PAWDD、CIMQ、OSA‑HCIM等），并利用NeuroSim、ALPINE等模拟平台进行系统级评估。

**📊 数据集**

使用的主数据集为WikiText‑103，用于计算单层映射后模型的perplexity，从而量化每个投影的精度敏感度。

**📈 对比分析**

与以往仅针对CNN或encoder‑Transformer的映射方法相比，本文的工作流在Stage 1–2已证明可快速定位最敏感投影；在Stage 3–4未完成的情况下，已提供准确的Δ perplexity值作为映射搜索的输入；实验表明单一投影（block 0 的 attention‑output）对perplexity的提升可达33点，远高于其它投影，验证了投影级映射的必要性。

**⚠️ 局限性**

局限性包括：仅完成工作流的前两阶段，未给出完整的映射与性能评估；使用的噪声模型仅覆盖PCM，未验证对其他非易失性存储（如ReRAM、SRAM）的泛化；单层灵敏度分析未考虑多层联合噪声的交互效应；扩展至10⁹参数级LLM时的KV‑cache、内存容量与搜索空间规模等问题尚未解决。

---

## 183. Before Fusion, Ask What to Keep: Contextual Calibration of Multimodal Signals

**arXiv ID:** 2606.02679 | [PDF](https://arxiv.org/pdf/2606.02679v1)

**作者:** Jiyuan Liu `[一作]` (Adelaide University), Weitong Chen `[通讯]` (Adelaide University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在融合前对多模态特征进行校准的模块——Value‑Gated Modality Refiner（VGMR），通过对跨模态的一致与冲突进行全局与通道级价值估计，然后生成细粒度门控对原始特征进行增强、保留或抑制，最终提高多模态学习效果。

**💡 创新点**

创新点在于：①把价值估计（global/channel‑level value）作为跨模态上下文的影响证据，而非直接的可靠性判断；②利用跨模态的一致性（agreement）和差异性（discrepancy）来为价值估计提供依据；③在融合前对特征进行调制，避免在后端优化中出现信息干扰；④在保持轻量化的同时实现显著的性能提升与鲁棒性增强。

**🔧 技术方法**

核心技术包括：跨模态摘要构造、全局与通道级价值编码器、基于值信号与特征信号的细粒度门控生成、门控放大与残差保留、与下游背骨（Transformer/Concat）无缝集成的模块化设计，以及结合值监督与正则化的端到端训练策略。

**📊 数据集**

使用了五个公开多模态基准数据集：MOSI、MOSEI（情感分析）、UCF101（动作识别）、AVE（音视事件检测）和CREMA‑D（音视情绪识别），涵盖文本、音频、视觉三种模态及其组合。

**📈 对比分析**

与传统的早期融合（Concat、Tensor Fusion）、轻量级门控（Cross‑Attention、Sigmoid+Tanh Gate）、以及多种优化层面平衡方法（Grad‑Blending、OGM‑GE、MMPareto、ARL、PMR 等）对比实验显示，VGMR 在 Transformer 与 Concat 两种主流后端上均取得了最高或竞争性的 Accuracy 与 Macro‑F1，并在噪声鲁棒性、梯度冲突稳定性以及弱模态干扰抑制方面表现更佳。

**⚠️ 局限性**

主要局限包括：相对轻量化的设计仍引入可观的推理延迟与参数量；价值估计目前仅基于全局摘要，可能忽略局部（时间/空间）冲突；未针对缺失模态、极端噪声或跨域迁移等更严苛场景进行充分验证；未来工作需进一步提升效率、扩展到更大背骨与更复杂任务。

---

## 184. BehaviorBench: Modeling Real-World User Decisions from Behavioral Traces

**arXiv ID:** 2606.02798 | [PDF](https://arxiv.org/pdf/2606.02798v1)

**作者:** Liangwei Yang `[一作]` (Salesforce Ai Research), Shelby Heinecke `[通讯]` (Salesforce Ai Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了BehaviorBench，一个基于真实公开预测市场交易记录的个性化决策建模基准，包含用户最终立场（Belief）和单笔交易行为（Trade）两层任务。

**💡 创新点**

创新点在于：①使用真实链上交易数据重建用户长期行为历史；②将任务拆分为Belief与Trade两层，分别考察稳定立场与瞬时交易预测；③提供四种历史接口（无历史、直接最近历史、生成用户画像、检索支持），将历史表示作为评估维度；④构造了可复现的数据集与评价脚本。

**🔧 技术方法**

主要技术包括：链上日志解析与行为重构、用户群体构建与质量控制、离散化的Belief/Trade标签生成、基于大模型的生成式预测（前沿模型与开源模型）以及多种评估指标（准确率、MAE、Median AE 等）。

**📊 数据集**

数据集来源：Polymarket 的公开事件市场数据和 Polygon 区块链交易日志，最终筛选出 2000 评估钱包，产生 141,445 条 Belief 实例和 1,485,972 条 Trade 实例，另设检索支持池。

**📈 对比分析**

通过在四种历史接口下对多款前沿与开源大模型进行评估，比较了 Belief 预测的 Choice Accuracy 与 Confidence MAE、Trade 预测的 Direction Accuracy 与 Amount Median AE。结果显示，个性化能显著提升性能，但仍远未达到上限；不同模型在准确率、校准与幅度预测上排名差异明显，说明任务对模型与历史接口都有特定需求。

**⚠️ 局限性**

局限性包括：仅覆盖预测市场交易场景，钱包为伪匿名账号，标签仅反映公开立场而非私下信念；交易金额受资金、流动性、费用等不可观测因素影响；可能被滥用于定向营销或监控；Benchmark 仍难以完全覆盖所有人类决策行为。

---

## 185. From Rocq to Metal: A Pipeline for Formally Verified Microcontroller Firmware

**arXiv ID:** 2606.02651 | [PDF](https://arxiv.org/pdf/2606.02651v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

---

## 186. Democracy on Rugged Landscapes: Phase Transitions in Optimal Voting Rules

**arXiv ID:** 2606.02813 | [PDF](https://arxiv.org/pdf/2606.02813v1)

**作者:** Joshua Nunley `[一作]` `[通讯]` (Indiana University), Joshua Nunley (Indiana University)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

构建了基于 NK 适应度景观的投票优化框架，比较了八种传统投票方法和一种通用评分法在不同景观复杂度（K）与跨依赖参数（α）下的长期集体福利表现，并将该框架扩展到包含代表制（β、p_self 参数）的情形。

**💡 创新点**

① 将投票方法视为对效用矩阵的不同信息转换；② 用跨依赖参数 α 捕捉法律与个人特征的耦合；③ 发现投票方法在景观复杂度空间中出现“相位转移”，并给出经验公式将二维 (K,α) 映射为单一复杂度轴；④ 引入代表制模型并量化身份投票与候选人自利对结果的交互影响。

**🔧 技术方法**

利用 NK 适应度景观模型、随机模拟（每配置 1000/500 次跑）以及统计排名分析；使用十种投票规则（包括 ordinal p=0.35 的平滑评分）和代表制下的候选人选择机制；对终端平均适应度、方差等指标进行比较，并拟合相位转移曲线。

**📊 数据集**

随机生成的 NK 景观（N=50，K=1…20），人口 M=100，个人特征固定为随机二进制向量；代表制情形下的候选人平台也随机生成；实验中 V=2（直接民主）或 V=4（代表制）提议空间。

**📈 对比分析**

通过终端平均适应度和适应度方差对方法进行排名。结果显示：在平坦景观（低 K、低 α）cardinal score 最高；在低至中等复杂度（K≈4–9）p=0.35 评分法优于 plurality；在中等复杂度（K≈8–14）Borda 最高；在高复杂度（K≥15）STAR 领先。Borda 在中等复杂度下还保持最低方差。代表制下，cardinal score 在大多数参数组合中保持优势，high β 和 high p_self 导致方法区分度降低，排名变得混乱。

**⚠️ 局限性**

① 仅考虑真诚投票，未考虑策略行为；② 个人特征固定，忽略政策对个人属性的反馈；③ NK 景观为随机生成，缺乏针对具体政策域的实证验证；④ 提议空间极小（V=2/4），限制了投票方法的表达能力；⑤ 代表制参数仅取离散值（β, p_self 取 0, 0.5, 1），未能细粒度探究；⑥ 未考虑代表选区划分、候选人数量和候选人优先级等实际选举细节。

---

## 187. Diamonds Are Forever: Stabilization Semantics for Unrestricted Aggregation and Recursion in Logica

**arXiv ID:** 2606.02926 | [PDF](https://arxiv.org/pdf/2606.02926v1)

**作者:** Evgeny Skvortsov `[一作]` (Google), Bertram Ludäscher `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7691 | [OpenAlex ID](https://openalex.org/A5057600294)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出一种新的稳定性语义——Defendant-Opponent (DO) 语义，用于处理含有无限递归和聚合的逻辑程序，特别是那些收敛但不达到固定点的程序（如 PageRank）

**💡 创新点**

创新点在于将真理定义为从任何状态都能“防御”并稳定的结果，提供等价的图论、博弈论和模态逻辑三种视角，并引入 ω‑limit 解释来赋予无固定点程序意义；同时证明 DO 语义与正向 Datalog 的最小固定点、WFS、SMS 兼容

**🔧 技术方法**

核心技术包括：1) 将每个谓词映射为重写规则以形成非单调形式系统；2) 构造推导图并定义稳定化真理；3) 设计三回合的 Thesis‑Defense 博弈；4) 将 DO 语义表述为 S4 模态公式；5) 引入 Hausdorff 拓扑下的 ω‑limit 解释

**📊 数据集**

论文中未使用具体数据集，仅以 PageRank、π 逼近和微分方程示例阐述理论；实验或数据集未被提及

**📈 对比分析**

比较方法主要是理论兼容性分析：对正向 Datalog、WFS 与 SMS 的结果做对应，证明 DO 语义不与它们冲突；未给出量化性能指标或实测结果

**⚠️ 局限性**

局限性包括：1) 对于大规模真实程序的计算复杂度未讨论；2) 缺乏实验验证与性能评估；3) 只在有限、可构造的推导图下保证 ω‑limit 解释的唯一性，实际系统中可能出现不可判定性或无限状态空间

---

## 188. Linguistic Productivity in Large Language Models: Models Coerce, but do not Preempt

**arXiv ID:** 2606.02953 | [PDF](https://arxiv.org/pdf/2606.02953v1)

**作者:** Claire Bonial `[一作]` (Georgetown University), Harish Tayyar Madabushi `[通讯]` (University of Bath)

**通讯引用:** 489 | [OpenAlex ID](https://openalex.org/A5070941491)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对大型语言模型（LLMs）进行强制选择与自适应熵测量实验，探究其在两种统计信号（正向的频繁使用和负向的未观察证据）驱动下的语言生成与约束能力，分别检验了强制作用（coercion）和统计预防（statistical preemption）在新词（nonce）上的表现。

**💡 创新点**

将使用主义构造语法(CxG)理论与LLM实验结合，首次在LLMs中分离并对正向频繁使用与负向缺失证据两大统计力量进行实证比较，并通过对新词的检验揭示LLMs在负向约束上的显著缺陷。

**🔧 技术方法**

使用基于自回归的基础解码模型（Llama-3.1-8B、Qwen3-32B 等）与指令调优模型（GPT-4o-mini、GPT-5.4、DeepSeek、Qwen2.5-7B-Instruct 等），通过两种评分范式：1）自适应熵（surprisal） 2）强制选择（forced choice）对候选句子进行评估。

**📊 数据集**

构造了四大实验数据集：①60组熟悉词与60组nonce词的强制作用对照句；②20组熟悉词与20组nonce词的a‑形容词（pre‑nominal 与 predicative）对照句；③40组熟悉动词与40组nonce动词的双宾/双宾格交替（ditransitive/dative）对照句；④与上述实验配套的人类基准注释数据。

**📈 对比分析**

将LLM的评分与人类强制选择基准进行对比，使用准确率/偏好率统计；在强制作用实验中，最大规模模型（Qwen3-32B）可达 85% 的人类一致性，整体模型在 60–96% 范围内；在预防实验中，熟悉词的模型表现出与人类相近的负向约束（如 a‑形容词、动词），但在 nonce 条件下，所有模型几乎完全失效，无法展现负向约束，说明模型在负向证据上的泛化能力有限。

**⚠️ 局限性**

限制：LLMs 仍主要依赖正向频率统计，缺乏对负向缺失证据的内部约束机制，导致在面对可识别但未曾出现的新词时无法适当地应用预防规则；实验仅基于英语数据，尚未验证跨语言适用性。

---

## 189. The Ghost Annotator: a Framework to Explore Human Label Variation in Content Moderation through Conformal Prediction

**arXiv ID:** 2606.02911 | [PDF](https://arxiv.org/pdf/2606.02911v1)

**作者:** Mirko Lai `[一作]` (Heriot-Watt University), Marco Antonio Stranisci `[通讯]` (Università degli Studi di Torino)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 Ghost Annotator 框架，利用 Conformal Prediction 与非合规分数 (NCS) 以及协同过滤思想，对大型语言模型 (LLM) 在内容审核任务中的预测与人类注释者之间的差异进行量化与可视化。

**💡 创新点**

创新点包括：① 引入 Ghost Prediction 指标，用以衡量模型预测与所有人类注释标签不匹配的比例；② 将 Ghost Annotator 建模为三维 NCS 四分位向量，借鉴协同过滤，评估模型与不同社会人口群体的相似度；③ 在同一实验框架内同时考察模型不确定性、Human Label Variation (HLV) 与人口偏差，提供统计保障的全流程。

**🔧 技术方法**

使用技术：Conformal Prediction、Non‑Conformity Score、Ghost Prediction 计算公式、Ghost Annotator 向量嵌入、协同过滤思想、余弦相似度、统计相关性分析、实验评估。

**📊 数据集**

数据集：Attitudes、CADE、Disentangling、MHS 四个针对可侮辱性、仇恨言论、可接受性和暴力的内容审核数据集；在人口偏差分析中主要使用 DAVANI 数据集，此外还参考 Measuring、Attitudes、CADE 的交叉实验。

**📈 对比分析**

比较方法：对四种 LLM（Qwen‑1.5B、Qwen‑7B、Llama‑3.2‑1B、Llama‑3.1‑8B）在四个数据集上计算平均 NCS、Ghost Prediction 率、与注释者不一致的比例，并通过 Pearson 相关系数探究模型不确定性与人类分歧、孤立度的关系。结果显示：大模型在 Ghost Prediction 上更自信但与人类标签更不一致；模型不确定性随人类分歧增大而提升；Ghost Annotator 揭示对 SSA 等少数族裔的系统性低相似度。

**⚠️ 局限性**

限制：① 人口偏差结论仅基于 DAVANI 数据集，缺乏更广泛的人口标注；② 仅评估了小到中型 LLM，未覆盖更大规模模型或多语言场景；③ 未对 Ghost Annotator 进行偏差缓解或调优实验；④ 对注释者多样性仍存在不足，未能彻底验证数据集层面的偏差是否可通过扩充实现消除。

---

## 190. Chatbots Output Meaningful (but Problematic) Language

**arXiv ID:** 2606.02973 | [PDF](https://arxiv.org/pdf/2606.02973v1)

**作者:** Matthew Stone `[一作]`, Una Stojnić `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 AI 聊天机器人输出的语言是否具有意义展开哲学与语言学分析，提出“去人类化”视角，认为机器人的输出可以在没有任何心理状态或交流意图的前提下，被赋予普通语言的意义；并通过词汇选择、因果历史与语言传统等机制解释了 ChatGPT、Claude 等案例中的幻觉、恭维及 ELIZA 效应。

**💡 创新点**

创新点：①从哲学层面去人类化语言意义的前提；②提出词汇选择的因果历史作为意义基础，摒弃传统意图/交流意图假设；③将人类与机器的语言行为统一为同一意义机制，提供对聊天机器人误导与错误的理论解释。

**🔧 技术方法**

使用的技术：哲学论证与语言学理论、心理语言学模型（词汇选择、激活扩散）、Transformer 注意力与词向量、LLM 训练机制；无具体算法实现。

**📊 数据集**

使用的数据集：无专门实验数据，主要引用公开的大规模文本语料（LLM 预训练数据）及 ChatGPT 与 Claude 的对话记录进行案例分析。

**📈 对比分析**

比较方法与性能：文章不做量化实验，而是通过案例分析和理论对比，说明传统“有意图”解释与本文“无意图”解释的差异；未给出性能指标。

**⚠️ 局限性**

局限性：①缺乏形式化模型验证；②对上下文敏感语义的处理仍基于经验假设；③未解决幻觉与错误信息的根本技术问题；④可能低估人类交际中意图的作用。

---

## 191. BAHSD: Bridging the Long-tail Gap via Adaptive Distillation in Black-box Sequential Recommendation

**arXiv ID:** 2606.03091 | [PDF](https://arxiv.org/pdf/2606.03091v1)

**作者:** Xi Zhou `[一作]` (Chinese Academy of Sciences), Tao Guo `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 5859 | [OpenAlex ID](https://openalex.org/A5077367725)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种黑盒自适应蒸馏框架 BAHSD，通过多尺度一致性探测和分层损失，实现在不访问内部参数的情况下对序列推荐模型进行高保真提取。

**💡 创新点**

创新点在于①隐式评估教师信号可靠性的多尺度一致性探测；②动态温度 KL、排名一致性与 InfoNCE 的分层蒸馏策略，分别针对头部偏好固化和尾部信息真空；③无需显式用户划分即可自适应调整蒸馏目标。

**🔧 技术方法**

使用多尺度序列截断、温度可调 KL 散度、Bayesian Personalized Ranking、InfoNCE 对比学习以及交叉尺度对称 KL 一致性等技术。

**📊 数据集**

在 Amazon Beauty（稀疏）和 MovieLens‑1M（稠密）这两个公开基准上进行实验。

**📈 对比分析**

与 DFME、ME‑MIA、UnKD、DHKD、ABKD、CDBCF 等黑盒蒸馏基线进行对比，BAHSD 在 Recall@10 / NDCG@10 上均优于所有基线，尾部用户提升达 80%+，在部分场景甚至超过教师模型。

**⚠️ 局限性**

局限性包括：需手工调节截断比例与温度，适用于可截断序列的推荐任务；未验证在极大项目空间或跨域迁移中的鲁棒性；仅能在黑盒 API 可查询 logits 的前提下使用。

---

## 192. Testing the Test: Score-Direction Instability in Class-Split Anomaly Detection

**arXiv ID:** 2606.02601 | [PDF](https://arxiv.org/pdf/2606.02601v1)

**作者:** Alejandro Ascarate `[一作]` (Queensland University of Technology), Olivier Salvado `[通讯]` (Queensland University of Technology)

**通讯引用:** 14841 | [OpenAlex ID](https://openalex.org/A5025220020)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了在数据集内部类拆分评估中，当异常类与正常类在表征空间重叠时，异常检测指标（AUROC）会崩溃、倒置或方向不稳定，并提出了邻域泄漏诊断指标。

**💡 创新点**

提出了无训练的邻域泄漏（neighborhood leakage）指标，用于预测类拆分基准的不可行性，并将该指标与 AUROC 失稳、方向不稳定等统计量关联。

**🔧 技术方法**

采用欧氏最近邻计算泄漏、kNN、Isolation Forest、Local Outlier Factor 等无监督异常检测器，并利用 VAE 的潜在空间与原始像素空间进行对比。

**📊 数据集**

使用 Fashion‑MNIST、CIFAR‑10 与 Imagenette 三个视觉复杂度递增的数据集。

**📈 对比分析**

通过平均 AUROC、方差、随机率、倒置率与方向不稳定率等指标评估不同表示与检测器组合，发现高泄漏对应高倒置与方向不稳定，验证诊断的预测力。

**⚠️ 局限性**

诊断仅检验类拆分基准的几何一致性，无法保证检测器在真实外部 OOD 场景下的泛化；同时对表征空间依赖度高，且仅使用欧氏距离近邻，可能在更复杂空间失效。

---

## 193. A Measurement-Driven Digital Twin Architecture for Plant-Level Biomass Estimation and Growth Forecasting in Hydroponic Systems

**arXiv ID:** 2606.02796 | [PDF](https://arxiv.org/pdf/2606.02796v1)

**作者:** Morgan Mayborne `[一作]` (Carnegie Mellon University), George Kantor `[通讯]` (Carnegie Mellon University)

**通讯引用:** 4181 | [OpenAlex ID](https://openalex.org/A5067346459)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种针对水培生菜的数字孪生框架，结合RGB-D图像的无损生物量估计与动态生长模型，实现对单株生长轨迹的实时更新和短期产量预测。

**💡 创新点**

创新点在于将基于视觉的生物量估计与可在线更新的植物级生长模型相结合，形成闭环数字孪生，并公开了涵盖环境、图像与真实质量的多模态数据集，支持后续研究。

**🔧 技术方法**

采用的技术包括：RGB-D相机+Intel RealSense D405摄像头采集图像，Buxbaum风格的双分支CNN+MLP用于生物量估计；基于NiCoLet B3的白盒生长模型和LSTM时序网络用于预测；以及线性/梯度无关参数优化、图像分割、DenseNet121骨干等提升方法。

**📊 数据集**

使用的数据集包含125株生菜的1308次测量，记录RGB-D图像、环境参数（温度、CO₂、PAR）与真值质量；其中546张用于训练生物量估计网络，274张用于模型参数校准，其余用于端到端测试。

**📈 对比分析**

在生物量估计上，采用预训练分支+Fine‑tune后，RMSE降至1.45 g，R²=0.929；在生长预测上，校准后的NiCoLet模型在1–4天预测误差分别为1.88、2.05、1.89、2.22 g，优于指数模型和LSTM，显示更高的长期鲁棒性。

**⚠️ 局限性**

主要局限包括：使用专有照明与通风设备限制了环境可控性；模型训练受限于低频、短周期数据；数字孪生目前尚未实现实时在线参数自适应，需进一步验证在不同生长条件下的泛化能力。

---

## 194. Making Brain-Computer Interfaces More Secure

**arXiv ID:** 2606.02597 | [PDF](https://arxiv.org/pdf/2606.02597v1)

**作者:** Md Fahimul Kabir Chowdhury `[一作]` (University of North Texas), Gahangir Hossain `[通讯]` (University of North Texas)

**通讯引用:** 995 | [OpenAlex ID](https://openalex.org/A5009360571)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种轻量化CNN网络，用于评估基于EEG的BCI在白盒FGSM攻击下的鲁棒性，并与EEGNet、DeepConvNet、SleepEEGNet进行对比

**💡 创新点**

在轻量化CNN中结合spectrogram表示和集成平均，显著提升了抗扰动性能，为EEG BCI安全评估提供了新的基准

**🔧 技术方法**

自定义轻量化CNN、白盒FGSM攻击、梯度下降、spectrogram特征提取、k折交叉验证、集成平均、TensorFlow2.14实现

**📊 数据集**

使用MI4四类运动想象数据集和rTMS抑郁治疗EEG数据集

**📈 对比分析**

在相同训练与攻击参数下与EEGNet、DeepConvNet、SleepEEGNet对比；轻量化CNN基线准确率88.21%，FGSM后73.02%，远高于其它模型（低至7%），平均Cohen's Kappa达99.96

**⚠️ 局限性**

结果过于理想，受spectrogram表示与集成平均影响；仅评估白盒FGSM攻击，未涵盖PGD、黑盒等更广泛攻击；数据集规模有限，缺乏更大范围的验证

---

## 195. Forgetting is Not Erasure: Recovering Latent Knowledge via Transport Keys

**arXiv ID:** 2606.02860 | [PDF](https://arxiv.org/pdf/2606.02860v1)

**作者:** Archie Chaudhury `[一作]` `[通讯]` (Axionic Labs), Archie Chaudhury (Axionic Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究持续学习中的灾难性遗忘，提出接口漂移（transport key）方法通过模型拼接恢复先前任务性能。

**💡 创新点**

认为遗忘是接口访问问题而非表示消除，提出compact transport key对齐后端网络的接口，展示在ResNet和ViT上显著恢复性能。

**🔧 技术方法**

使用模型拼接（model stitching）评估协议，构造compact calibration key和cross‑channel key，并用小anchor集估计对齐变换。

**📊 数据集**

在CIFAR‑100、CIFAR‑10、SVHN和Mini ViT等视觉基准上进行实验。

**📈 对比分析**

对比预更新、后更新和拼接准确率，发现拼接可恢复高达92%（ResNet）或83%（ViT）等原始性能，证明接口对齐有效。

**⚠️ 局限性**

实验仅限3个任务，需保留前一次模型；拼接评估不适用于在线部署，长期任务序列可能失效。

---

## 196. Locality Does Not Imply Reachability: Boundary Repair in Block-Sparse Causal Attention

**arXiv ID:** 2606.02680 | [PDF](https://arxiv.org/pdf/2606.02680v1)

**作者:** Zhibo Yang `[一作]` (Ocean University of China), Zhibo Yang `[通讯]` (Ocean University of China)

**通讯引用:** 146 | [OpenAlex ID](https://openalex.org/A5003027050)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文分析了固定块稀疏因果注意力的可达性缺陷，并提出零额外参数的边界桥接（Boundary Bridge）机制修复该缺陷；

**💡 创新点**

创新点在于将可达性障碍形式化为结构依赖集、给出边界复制下的精确下界，并设计可控的边界桥修复策略；

**🔧 技术方法**

使用了稀疏注意力图分析、结构依赖集理论、桥接注意力、滑动窗口与全注意力对照实验；

**📊 数据集**

实验数据集为OpenWebText和人工合成的检索诊断提示；

**📈 对比分析**

通过在1024-token GPT‑2风格模型上训练相同参数量的六种注意力架构，并在8K-token Qwen2.5‑7B上做固定检查点干预，发现桥接在边界检索任务上显著提升，但在整体困惑度和标准针头检索上略逊于滑动窗口；

**⚠️ 局限性**

局限性包括仅在固定块极限下验证，缺乏大规模或多种稀疏模式的普适性验证，且未进行硬件加速实现或复杂度优化。

---

## 197. CAD-to-CT Registration of Cylindrical Objects via Ellipse-Based Axis Estimation

**arXiv ID:** 2606.02935 | [PDF](https://arxiv.org/pdf/2606.02935v1)

**作者:** Aleksander Ogonowski `[一作]` (National Centre for Nuclear Research), Sławomir Wronka `[通讯]` (ImagineRT sp. z o.o.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出一种两阶段几何配准方法，将CAD模型对齐到工业CT扫描的离子化学室，以获得无标注的高精度分割标签。

**💡 创新点**

创新之处在于利用圆柱对称性通过椭圆中心轨迹估计旋转轴，并通过体积重叠多项式优化实现亚体素平移，无需强度校准或人工标注。

**🔧 技术方法**

方法采用Scharr算子边缘检测、Fitzgibbon最小二乘椭圆拟合、RANSAC+PCA求轴、体素化CAD、体积重叠评分与二次多项式拟合等技术。

**📊 数据集**

在仿真CT（20个扫描）与18台真实离子化学室的工业CT数据上验证，并用CMM测量作为真值。

**📈 对比分析**

在仿真数据上轴向误差<0.1°，IoU/Dice>0.998；在真实数据上与CMM的体积误差平均约0.0±0.3%，表明亚体素精度并可用于机器学习分割。

**⚠️ 局限性**

局限在于仅适用于具有可检测椭圆截面的旋转对称对象、需要高精度CAD模型且假设物体刚性，且对非圆形或短物体的配准效果未知。

---

## 198. Fairness Definitions and Metrics in Deep Reinforcement Learning for Drug Discovery in Healthcare: A Rapid Evidence Review

**arXiv ID:** 2606.02902 | [PDF](https://arxiv.org/pdf/2606.02902v1)

**作者:** Esmaeil Shakeri `[一作]` (University of Calgary), Behrouz Far `[通讯]` (University of Calgary)

**通讯引用:** 3518 | [OpenAlex ID](https://openalex.org/A5008779348)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本论文通过快速系统综述总结了深度强化学习（DRL）在药物发现中关于公平性的定义、指标与实践，聚焦数据集构建、奖励设计与评价方法如何影响不同疾病领域（尤其是癌症）和化学结构的公平性

**💡 创新点**

创新点在于首次将公平性概念系统化应用于DRL分子生成领域，提出了针对疾病类别和化学骨架的分层公平度量框架，并揭示了当前实践中存在的偏差与缺口

**🔧 技术方法**

采用深度强化学习技术（如DQN、GCPN、DiffMeta‑RL等）以及多目标奖励与ADMET约束的组合，辅以手工编码的内容分析方法

**📊 数据集**

使用公开药物数据库（ChEMBL、ZINC、ExCAPE‑DB）以及针对特定靶点（DRD2、CYP450）的标注数据集，结合不同的划分策略（随机划分 vs. Scaffold 划分）

**📈 对比分析**

对比方法主要基于文献计量与内容编码，未给出单一模型性能数值，而是通过对24篇论文的统计发现：随机划分往往低估分子多样性和疾病平衡，奖励设计过于依赖简单性质会导致偏向某些化学子群；多目标奖励与更丰富的探索策略能在一定程度上缓解偏差

**⚠️ 局限性**

局限性包括仅检索已发表英文全文且排除 arXiv 预印本，导致检索覆盖不完整；编码过程主要由单一研究者完成，虽然后续核对但可能存在主观偏差；综述未给出量化公平度量指标，缺乏统一的实验验证

---

## 199. Learn When and Where to Connect: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs

**arXiv ID:** 2606.03068 | [PDF](https://arxiv.org/pdf/2606.03068v1)

**作者:** Jaejun Lee `[一作]` (KAIST), Joyce Jiyoung Whang `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可学习的动态虚拟节点框架 Maven，能够在每个 MPNN 层自适应地引入虚拟节点并与原始节点动态连接，实现灵活的消息传递路径。

**💡 创新点**

创新点在于：①允许节点与虚拟节点的非约束连接；②在不同层动态引入虚拟节点；③使用双视角评分机制（节点偏好 + 虚拟节点偏好）来决定节点-虚拟节点、虚拟节点-虚拟节点连接；④证明任何节点-虚拟节点连接模式均可由单层实现。

**🔧 技术方法**

使用了 MPNN 基础、可微分的相关性得分（scaled dot‑product + MLP）、log‑softmax 调整、双视角评分、Gumbel‑Softmax 直通估计、多头机制、以及理论证明（单层可模拟任意连接）。

**📊 数据集**

实验数据集包括：九个真实世界图数据集（LRGB 四个多图数据集、五个异质性图数据集）以及合成树数据集，用于评估节点/图分类、回归和节点分类等任务。

**📈 对比分析**

与多种基准（图重排方法、图变压器、传统 VN 方法、GCN、GAT、SAGE 等骨干）进行对比；平均提升约 17%（单层可达 46%），在节点和图级任务上均优于现有最先进方法；统计检验显示大多数提升显著。

**⚠️ 局限性**

限制：对超参数 M（最大虚拟节点数）敏感，M 较大时计算和内存开销增长；在极大图上仍受 O(M²) 影响；仅利用节点表示，可能在节点特征稀疏或缺失时表现不佳；实验主要在固定骨干上，缺乏对不同图结构鲁棒性的深入验证。

---

## 200. ConTraIRL: Factorized Contrastive Abstractions for Transferable IRL

**arXiv ID:** 2606.03017 | [PDF](https://arxiv.org/pdf/2606.03017v1)

**作者:** Yikang Gui `[一作]` (University of Georgia), Prashant Doshi `[通讯]` (University of Georgia)

**通讯引用:** 3550 | [OpenAlex ID](https://openalex.org/A5001254145)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出ConTraIRL框架，通过双编码器将观测映射到独立的动力学与目标潜空间，实现对未见动力学-目标组合的奖励恢复；

**💡 创新点**

创新点在于将上下文因素显式分解为互相正交的动力学与目标抽象，并结合时间相位对齐与对比学习，提升组合泛化；

**🔧 技术方法**

核心技术包括超网络（context‑modulated）双编码器、对比学习（margin‑based与噪声对比）结构约束、专家‑学习者对比校准，以及基于相位的聚类奖励计算；

**📊 数据集**

在四个MuJoCo连续控制任务（Ant、HalfCheetah、Walker、Swimmer）上构造了多种动力学与目标组合，作为源与目标环境；

**📈 对比分析**

与TraIRL、C‑AIRL、SFM等基线相比，ConTraIRL在仅有少量目标专家状态的条件下，平均归一化回报提升约10%–15%，并在目标组合上的性能保持稳健；

**⚠️ 局限性**

局限性包括需已知的动力学与目标标签、对时间相位定义的依赖、对循环或可变长度行为的适应有限，以及对完全零样本情景的迁移尚未实现。

---

## 201. Outsmarting the Chameleon: Counterfactual Decoupling for Tactical OOD Shifts in Live Streaming Risk Assessment

**arXiv ID:** 2606.02946 | [PDF](https://arxiv.org/pdf/2606.02946v1)

**作者:** Yiran Qiao `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Xiang Ao `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出并实现了LPCD框架，用于解决直播风险评估中的tactical OOD（对抗性叙事包装）问题。

**💡 创新点**

创新点在于在潜在空间进行因果解耦，结合表示层与预测层的对抗一致性约束，并在推理阶段加入无参数幅度校准，实现对策略演变的鲁棒性。

**🔧 技术方法**

采用潜在因果建模、双分支解耦网络、对比一致性损失、后验幅度校准，以及Transformer/MIL等多种backbone模型。

**📊 数据集**

使用字节跳动抖音直播的工业数据集，包括 May 和 June 两个时间段的训练、验证、ID 与 OOD 测试集。

**📈 对比分析**

与多类 OOD 基线（IRM、VREx、GroupDRO、EIIL 等）和不同backbone（Transformer、Reformer、TimeMIL 等）进行对比，PR‑AUC 提升 3–8%，F1、召回率和在线 AB 测试指标均显著优于现有 XGBoost 与 Transformer 模型。

**⚠️ 局限性**

局限性：幅度校准依赖训练时的包装统计，对极端新型攻击策略的显式建模不足；同时，额外的对比损失增加训练复杂度与计算开销。

---

## 202. Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling

**arXiv ID:** 2606.02837 | [PDF](https://arxiv.org/pdf/2606.02837v1)

**作者:** Andrea Brunello `[一作]` (University of Udine), Nicola Saccomanno `[通讯]` (University of Udine)

**通讯引用:** 191 | [OpenAlex ID](https://openalex.org/A5008815184)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对NL‑to‑FOL数据集FOLIO和MALLS进行了系统的人类审核，发现大约40%示例存在翻译错误、含糊语义或NLI标签不一致，并对错误进行了修正；随后设计并公开了LLM辅助的“判定‑细化”工作流，能够在仅审核24%（FOLIO）或13%（MALLS）实例的情况下，将整体准确率提升至90%；

**💡 创新点**

创新点在于：①首次对主流NL‑to‑FOL基准进行规模化、严格的人类审计，量化了错误对模型评估的影响；②提出了基于LLM的三态判定框架（正确、错误、含糊）与优先级排序，显著降低人工审核成本；③展示了纠正数据后对三大LLM模型（Gemma、Qwen、GPT‑4o‑mini）评估的显著提升，验证了数据质量对模型性能的关键作用；

**🔧 技术方法**

技术主要包括：人类双人并行审阅+冲突解决；LLM判定‑细化（V&R）任务，使用Chain‑of‑Thought、few‑shot、元提示等多种提示方式；Z3定理证明器用于自动验证公式等价性和NLI标签；以及基于判定结果的实例优先级排序与人机协作审计；

**📊 数据集**

使用的数据集为：FOLIO验证集（275例）、MALLS测试子集（100例）以及控制集GGC（213例）；同时将修正后的数据发布至HuggingFace；

**📈 对比分析**

通过对三种LLM（Gemma 4 31B‑it、Qwen3‑30B‑A3B、GPT‑4o‑mini）在原始与修正数据上进行链式推理实验，结果显示在FOLIO上准确率提升9–22个百分点，MALLS上提升9–18个百分点；LLM辅助审计的准确率‑人力曲线表明，Pipeline 1在FOLIO上仅需审计24%实例即可达到90%准确率，MALLS仅需13%，显著优于随机审计基线；

**⚠️ 局限性**

局限性包括：人工标注仍可能留有误差，尤其在语义含糊判断上缺乏统一标准；LLM判定的可靠性依赖于所选模型和提示；实验仅覆盖FOL形式和三大LLM，结果可能不适用于其他逻辑形式或模型；此外，在已高质量数据（GGC）上使用管线仍会略微降低准确率（≤5%）。

---

## 203. CoughSense: Five-Class Respiratory Disease Classification via Whisper Encoder Fine-Tuning and Dual-Encoder Cross-Attention Fusion with Balanced Contrastive Learning

**arXiv ID:** 2606.02998 | [PDF](https://arxiv.org/pdf/2606.02998v1)

**作者:** Nikhil Vincent `[一作]` `[通讯]`, Nikhil Vincent

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc` `f7dab867-23a8-4241-85e9-4ba79c6402f9`

**🎯 论文内容**

开发了一套名为CoughSense的移动端咳嗽声音多分类系统，可识别健康、COVID‑19、哮喘/呼吸系统疾病、支气管炎和肺炎五类疾病。

**💡 创新点**

创新点包括首次将OpenAI Whisper预训练模型迁移到咳嗽声音分类；提出活跃帧QKV注意力池化以消除长时间帧中的无声噪声；结合加权采样、SpecAugment、平衡Mixup、对比学习与域适配的全方位训练策略；以及利用OPERA‑CT与Whisper的双编码器交叉注意力融合。

**🔧 技术方法**

采用Whisper小模型（tiny）Transformer编码器，结合QKV注意力池化、FiLM症状调制、WeightedRandomSampler、SpecAugment、平衡Mixup、监督对比损失、梯度逆转域适配以及双编码器交叉注意力等技术；模型在服务器端通过FastAPI实现低延迟推理。

**📊 数据集**

使用了四个公开数据集：Coswara、CoughVID、Virufy和西南医院儿科咳嗽数据集（共18,301条记录），并通过8倍增强提升了少数类支气管炎和肺炎样本。

**📈 对比分析**

通过5折分层交叉验证评估，Whisper‑tiny模型达82.3%平衡准确率（macro‑F1 0.817，AUC 0.941），双编码器模型进一步提升至85.4%；相较于EfficientNet‑B2（71.2%）和从零训练的ViT（52.7%）表现显著优异。

**⚠️ 局限性**

局限性包括：少数类数据主要来自儿童，导致成人/儿童声学差异；标签多为自报或非PCR确认，存在噪声；缺乏成人支气管炎、肺炎和结核等重要疾病；麦克风频率响应差异导致现场迁移误差；尚无前瞻性临床验证。

---

## 204. Impact of a Soft Wearable Back-Support Device on Postural Stability during Trip-Like Perturbations

**arXiv ID:** 2606.02888 | [PDF](https://arxiv.org/pdf/2606.02888v1)

**作者:** Yuanhao Chen `[一作]` (Arizona State University), Hyunglae Lee `[通讯]` (Arizona State University)

**通讯引用:** 1748 | [OpenAlex ID](https://openalex.org/A5049280323)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究评估了一种轻量化、可调刚度的软可穿戴背部支撑装置在站立和行走时应对类绊倒扰动的平衡稳定性。

**💡 创新点**

创新点在于将真空调节的可变刚度阻力带与原理为收缩的折纸肌肉装配，形成既柔软又可调节的背部支撑系统，并证明其能显著提升动态稳定裕度。

**🔧 技术方法**

使用双带跑步机、力平台、八摄像头运动捕捉系统、负载传感器与真空调节组件，计算了扰动响应中的最小稳定裕度（MOS）。

**📊 数据集**

使用自制实验数据集，共5名健康年轻男性，在站立和行走两种扰动实验中收集了力学与运动学数据。

**📈 对比分析**

通过线性混合效应模型和 Bonferroni 校正的配对比较，结果显示在站立实验中MOS随装置刚度提升显著提高，在行走实验中装置相较于无装置也显著提升，但两种刚度条件差异不显著。

**⚠️ 局限性**

局限包括样本量仅5人、仅使用MOS作为稳定性指标、简化标记集导致COM估计误差、未评估需跨步平衡的扰动以及跑步机实验的生态有效性问题。

---

## 205. Aligning Data-Driven Predictors with Allocation: A Decision-Focused Approach to Survival Analysis

**arXiv ID:** 2606.02671 | [PDF](https://arxiv.org/pdf/2606.02671v1)

**作者:** Itai Zilberstein `[一作]` (Carnegie Mellon University), Tuomas Sandholm `[通讯]` (Carnegie Mellon University)

**通讯引用:** 20332 | [OpenAlex ID](https://openalex.org/A5023571961)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出使用 NDCG 作为生存预测模型的优化目标，并在右删失数据下设计无偏估计器和 bootstrap 训练框架以提升对移植分配的效果；

**💡 创新点**

①证明传统 C‑index 等聚合指标无法为单一移植决策提供效用保证；②引入 NDCG（尤其 k=1）作为能直接对应分配收益的指标；③设计两种无偏 NDCG 估计器（基于期望值的 EY 与逆概率删失加权 IPCW）；④构建基于伪标签和 LambdaRank 样式排序损失的 bootstrapping 方法，兼顾 NDCG 与预测精度；

**🔧 技术方法**

生存分析模型（KM、Cox、AFT、DeepSurv、DeepHit），逆概率删失加权（IPCW）、期望值估计（EY）、LambdaRank 相关对排序损失、梯度提升决策树（GBDT）或深度网络；

**📊 数据集**

美国 UNOS（联合器官共享网络）心脏移植登记数据库（1987–2023 年），并在实验中使用人工删失版本进行验证；

**📈 对比分析**

与基线模型（KM、Cox、AFT、DeepSurv、DeepHit）在 NDCG、C‑index、AUC@5 等指标上对比；bootstrapped 模型在 NDCG 上提升 50–100%，k=1 提升约 0.2 以上，C‑index 与 AUC@5 维持相近或略优；人工删失实验和完整数据实验均证实方法的有效性；

**⚠️ 局限性**

依赖无偏估计器假设，若生存或删失模型校准不佳则 EY/ICPW 偏差；逆概率删失权重可能因信息删失强而变大导致方差高；假设条件独立删失与无观测混杂可能在真实数据中不成立；

---

## 206. RRISE: Robust Radius Inference via a Surrogate Estimator

**arXiv ID:** 2606.02876 | [PDF](https://arxiv.org/pdf/2606.02876v1)

**作者:** Jong-Ik Park `[一作]` (Carnegie Mellon University), José M. F. Moura `[通讯]` (Carnegie Mellon University)

**通讯引用:** 27879 | [OpenAlex ID](https://openalex.org/A5045861415)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种通过训练代理网络并采用一次性共形校准，将随机平滑的鲁棒半径估计转化为一次前向传播即可完成的证书方法，显著降低在线蒙特卡罗采样成本；

**💡 创新点**

创新点在于：①用交叉熵损失对基模型的平滑分布进行无偏监督训练；②一次性共形校准将代理输出的概率下界映射为可靠的鲁棒半径；③实现部署可验证的正向传播证书；

**🔧 技术方法**

采用随机平滑、基于噪声的数据增强、交叉熵训练代理、共形校准与高斯逆CDF计算鲁棒半径；

**📊 数据集**

在FashionMNIST、CIFAR‑10、CIFAR‑100、Tiny ImageNet四个图像分类数据集上进行实验；

**📈 对比分析**

与固定预算蒙特卡罗、适应性采样、早停等基线对比，代理方法在保持与固定预算MC相近的Certified Accuracy（误差<0.84个百分点）的同时，将每个输入的采样量从约10⁴次降低到1次；在重复部署场景下，累计成本在约5–10×10⁴次查询后即可打破MC成本；

**⚠️ 局限性**

主要限制是离线目标构造与代理训练的前置成本高，对分布漂移时的共形校准假设（可交换性）敏感，且目前仅支持单一噪声水平，扩展至多噪声或不同任务需进一步研究。

---

## 207. Cosmos 3: Omnimodal World Models for Physical AI

**arXiv ID:** 2606.02800 | [PDF](https://arxiv.org/pdf/2606.02800v1)

**作者:** Aditi `[一作]`, Artur Zolkowski `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `afceb026-1760-41ae-8d86-010831a37d97` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并训练了Cosmos 3，一种统一的多模态世界模型，能够同时完成语言、图像、视频、音频与动作的理解与生成，并通过后训练进一步衍生出文本生成、视频生成、机器人策略等专用版本。

**💡 创新点**

创新点包括：
- 采用Mixture‑of‑Transformers（MoT）双塔架构，将自回归推理与扩散生成分离，仍保持双向注意力；
- 统一的模态编码与可变时间调制的三维M‑RoPE位置编码，使多种采样率、帧率的模态能够在同一时间轴上对齐；
- 将动作视为核心模态，设计域感知的动作编码与解码，支持前向动力学、逆向动力学和联合策略生成；
- 通过大规模合成数据集（SDG系列）与现实数据混合的多阶段训练，实现了从世界认知到物理模拟到决策的全链路学习；
- 公开完整的代码、权重、合成数据与基准，促进开源研究与工业落地。

**🔧 技术方法**

技术细节：
- 模态专属编码器：ViT（视觉）+ VAE（图像/视频）+ 音频VAE；
- 模式切换的Token排列：自回归（AR）+扩散（DM）子序列；
- MoT双塔Transformer，AR塔负责推理，DM塔负责生成；
- 双流联合注意力，保证DM能访问AR上下文；
- 绝对时间调制的3D MRoPE，解决不同采样率导致的时间轴不匹配；
- 逐阶段训练策略：Reasoner预训练→Fine‑tune→Generator预训练→Mid‑training→Post‑training；
- 训练目标：Rectified Flow（流匹配）+多模态交叉熵；
- 多分辨率、多帧率训练与动态缩放；
- 对比式无监督学习、可变采样率与自回归语言模型融合。

**📊 数据集**

使用的数据集：
- 预训练：大规模图像‑文本、视频‑文本语料（多语言）；
- 中间阶段：合成物理模拟数据（SDG-PhyxSim、SDG-RobotSim、SDG-DriveSim、SDG-SynHuman、SDG-Warehouse）；
- Fine‑tune：真实机器人数据（DROID）、自动驾驶、仓储、工业场景；
- 合成多模态（音频、视频、动作）数据；
- 评测基准：Cosmos‑HUE、UniGenBench、人工分析图像/视频/策略排行榜等。

**📈 对比分析**

评估方法：在理解（多模态问答、空间/时间推理）与生成（文本‑图像、文本‑视频、图像‑视频、音视频同步、动作推断、策略生成）等多维任务上进行基准测试。Cosmos 3在绝大多数任务上达成或超过专门模型的SOTA，特别是在物理一致性、动作相关生成与机器人策略上表现突出。其后训练版本在UniGenBench、人工分析文本‑图像、图像‑视频排行榜中均获得首位或最优成绩。

**⚠️ 局限性**

局限性：
- 训练成本极高（多模态、上百亿参数、数千万GPU时钟）；
- 对极端或罕见物理交互（如高速碰撞、极低光照）仍存在精度不足；
- 在真实部署时对硬件与延迟要求高，尤其是大模型版本；
- 虽然使用合成数据提升泛化，但仍可能出现从合成到真实环境的域偏移；
- 目前未充分覆盖所有动作空间（如细粒度抓取、运动控制细节）和更复杂的语音/语义交互。

---

## 208. Greener Than Humans? Environmental Attitudes in Large Language Models

**arXiv ID:** 2606.02741 | [PDF](https://arxiv.org/pdf/2606.02741v1)

**作者:** Stefanie Kunkel `[一作]` (Research Institute for Sustainability at GFZ Centre for Geosciences), Angelika Gellrich `[通讯]` (German Environment Agency)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文开发了一个可重复使用的基准，用于评估大型语言模型在环境认知、情感和行为推荐方面的表现，并对31种主流LLM进行系统评估。

**💡 创新点**

创新点在于将德国环境认知调查UBS的量表与LLM回答相匹配，形成多维度评价框架，并公开代码与数据集，首次系统比较LLM的可持续发展取向。

**🔧 技术方法**

技术上采用多轮API调用、温度设为0、基于UBS问题的master prompt，并利用欧氏距离度量角色提示与顺从性敏感度，Python流水线实现自动化评估。

**📊 数据集**

使用了德国联邦环境署UBS 2024调查问卷（共17+10+11+19题）以及自建的CO₂减排量量化问题，公开代码与数据集存于Zenodo。

**📈 对比分析**

比较方法通过计算影响指数（affect、cognition、CO₂减排潜力、WTP）与人类调查平均值对齐，并使用欧氏距离度量角色提示的偏移；结果显示大部分LLM在环境认知与情感上高于德国平均水平，但在WTP和行为推荐上表现不一。

**⚠️ 局限性**

局限包括模型对措辞的敏感性、持续更新导致可重复性受限、未能提供解释性推理、数据集主要覆盖西方视角以及对数字问题的低准确性。

---

## 209. MUSE: A Unified Agentic Harness for MLLMs

**arXiv ID:** 2606.03005 | [PDF](https://arxiv.org/pdf/2606.03005v1)

**作者:** Jianglin Lu `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 32015 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MUSE，一个针对冻结多模态大语言模型的统一结构化执行 harness，通过优化模型外的执行框架来提升视觉任务性能。

**💡 创新点**

创新点在于将“可外壳化”概念扩展到多模态场景，设计统一任务表示、任务钩子、指令接口、结果策略、验证器与修复循环，并加入感知工具注册表实现可扩展的工具调用。

**🔧 技术方法**

采用模块化推理管道、外部验证器驱动的自我修复、任务特定解析与指令模板、感知工具（如 OCR、几何变换）以及可配置的修复预算 K。

**📊 数据集**

在四类视觉基准上验证：VSP‑Grid（视觉空间规划）、BLINK‑Jigsaw（视觉拼图）、CoMT（多模态推理）和 TIR‑Bench 词语搜索（精细视觉区分）。

**📈 对比分析**

与单次生成零射击基线以及计算匹配的自一致性控制对比，MUSE 在所有模型上均显著提升准确率（例如 GPT‑4o 词语搜索从 3% 提升至 21%，VSP‑Grid 有效路径提升 17–24 点），且往往使用更少的模型调用。

**⚠️ 局限性**

局限性包括：对持续性感知错误（如障碍误定位）修复有限；依赖任务特定可靠验证器，难以应用于开放式生成；额外调用导致计算成本上升；当前模块与策略手工构建，缺乏自动化自适应能力。

---

## 210. Multi-Segment Attention: Enabling Efficient KV-Cache Management for Faster Large Language Model Serving

**arXiv ID:** 2606.02964 | [PDF](https://arxiv.org/pdf/2606.02964v1)

**作者:** Chunan Shi `[一作]` (Peking University), Bin Cui `[通讯]` (Peking University)

**通讯引用:** 13562 | [OpenAlex ID](https://openalex.org/A5062357883)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于计算延迟感知的 KV 缓存管理系统，包括多段注意力 (MSA)、期望延迟驱动的块淘汰策略和自适应分块调度。

**💡 创新点**

创新点：①多段注意力实现非连续 KV 缓存；②基于块重用概率与位置相关的计算延迟模型的期望延迟感知淘汰；③采用分段指数衰减的频率函数实现 O(log n) 级别的淘汰算法；④在线调整块生命周期以适配工作负载变化。

**🔧 技术方法**

使用技术：CUDA/FlashAttention 与 PagedAttention 的融合 kernel，基于多段注意力的自定义 CUDA kernel，平衡树实现 O(log n) 淘汰，线性/二次计算延迟成本模型，分段指数衰减的访问频率函数。

**📊 数据集**

使用数据集：LongBench、LooGLE（长上下文评估），BFCL（agent 工作负载）。

**📈 对比分析**

与 vLLM‑LRU、Max‑Score、Pensieve+MSA 等基线进行对比，实验涵盖低/高分散工作负载。结果显示 TTFT 下降 1.86×–2.03×，TPOT 下降 1.62×–1.71×，在多会话和 agent 场景下，整合 Continuum 后平均任务延迟可降至 18.1%。

**⚠️ 局限性**

局限性：目前仅在 GPU 上管理 KV 缓存，未实现多级存储或压缩；对模型精度无影响但无法进一步降低显存；需手动调节分段指数参数，且在极端长序列或多 GPU 场景下的扩展性尚待验证。

---

## 211. Inference Cost Attacks for Retrieval-Augmented Large Language Models

**arXiv ID:** 2606.02643 | [PDF](https://arxiv.org/pdf/2606.02643v1)

**作者:** Chengliang Liu `[一作]` (Hong Kong Polytechnic University), Wenqi Fan `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5169 | [OpenAlex ID](https://openalex.org/A5043696243)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究通过向 Retrieval-Augmented Generation（RAG）系统的外部知识库注入恶意文档，构造了一种可在不破坏答案正确性的前提下显著增加推理时的 token 消耗的攻击方法（RA-ICA）。

**💡 创新点**

创新点在于：①提出了针对 RAG 系统的“推理成本攻击”新范式；②设计了基于 LLM 的改写与生成两种攻击代理，并用三种资源消耗策略（混淆任务、对立信息、任务导向）生成恶意文档；③研发了 Memory-Augmented Group Relative Policy Optimization（MA‑GRPO）强化学习框架，在动态记忆池中提升代理生成恶意文档的质量与攻击效果。

**🔧 技术方法**

使用的技术包括：大型语言模型（如 Llama‑3.1‑8B‑Instruct、Deepseek‑R1）、重写与生成式攻击代理、三种攻击策略、MA‑GRPO（GRPO 加动态记忆缓冲）、评估指标（RR、wAA、wAC、wTCR）以及多轮黑盒交互。

**📊 数据集**

实验使用了三个公开 QA 数据集：Natural Questions（NQ）、HotpotQA、MS MARCO，且在四个目标 LLM（qwen‑turbo、GPT‑5、Claude‑Sonnet‑4、Deepseek‑R1）上验证。

**📈 对比分析**

与现有的基于 prompt 攻击、ICL‑Genetic、PoisonedRAG、Paradox 等基线对比，RA‑ICA 在 Retrieval Rate > 90%，Token Consumption Ratio 可提升至 13.12×，且保持 85%+ 的答案一致性与 80%+ 的隐蔽性，证明在保密性与高效性方面均显著优于对手方法。

**⚠️ 局限性**

主要局限包括：①攻击成功率依赖于对目标系统检索模块的匹配程度，可能在高度加密或私有知识库中效果有限；②恶意文档的注入需要对外部知识源有一定的发布或写入权限，实际攻击成本不小；③实验仅在公开 QA 数据集和四种模型上评估，未验证在更大规模或多模态 RAG 系统中的表现。

---

## 212. LLM-Assisted Reranking to Operationalize Nuanced Objectives in Recommender Systems

**arXiv ID:** 2606.02883 | [PDF](https://arxiv.org/pdf/2606.02883v1)

**作者:** Amir Ghasemian `[一作]` (University of California, Los Angeles), Duncan J. Watts `[通讯]` (University of Pennsylvania)

**通讯引用:** 97786 | [OpenAlex ID](https://openalex.org/A5000679279)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用零样本LLM对YouTube侧边栏候选列表进行重新排序，探究其对政治内容曝光的影响，并通过提示级约束来减少极端或阴谋论内容的推荐。

**💡 创新点**

创新点在于①提出一种零样本LLM重新排序框架并与社会价值约束结合；②设计轻量级提示正则化，使得在保持个性化的同时提升意识形态多样性并抑制极端内容；③通过合成实验验证LLM主要依据语言统计而非语义理解来识别意识形态。

**🔧 技术方法**

技术采用OpenAI的GPT‑4（或等价LLM）进行零样本提示式重排，并结合embedding‑based re‑ranking、BERT/OpenAI embedding计算；提示工程中加入社会价值约束。

**📊 数据集**

数据来源为Nielsen收集的97名美国成年人的YouTube桌面浏览轨迹，通过自动重建侧边栏推荐构造10视频重叠会话；对视频进行政治立场、主题、极端内容的自动/人工标注。

**📈 对比分析**

与原始YouTube推荐、embedding‑based重排以及未约束LLM重排进行对比。结果显示LLM重排在个性化AUC上优于YouTube，但会放大极端内容曝光；约束LLM重排（rLLM+YT）在保持近似个性化的同时显著降低极端内容曝光并提升意识形态多样性。

**⚠️ 局限性**

局限性包括①实验使用的重构会话可能不完全再现真实用户体验；②左倾样本不足导致左翼结果缺乏稳健性；③LLM主要基于语言统计而非真正的意识形态理解；④未评估长期影响、情绪或注意力等后续效应。

---

## 213. Predicting Inference-Time Scaling Gains from Labeled Validation-Set Output Statistics

**arXiv ID:** 2606.02981 | [PDF](https://arxiv.org/pdf/2606.02981v1)

**作者:** Luyang Zhang `[一作]` (Carnegie Mellon University), Jingyan Li `[通讯]` (Johns Hopkins University)

**通讯引用:** 22779 | [OpenAlex ID](https://openalex.org/A5100677450)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大语言模型的最佳‑of‑N推理扩展中，本文提出一种仅用一次验证集采样即可预测最佳‑of‑N收益的方法。

**💡 创新点**

创新点在于把低成本的采样统计与最佳‑of‑N收益关联，并通过 bootstrap‑Lasso 发现稳定的三维核心特征（多数比例扩散、首次正确样本位置、完成长度方差），随后用紧凑的岭回归实现高精度排名预测。

**🔧 技术方法**

使用了岭回归、bootstrap‑Lasso 稳定性选择、线性逼近与集中分析、Spearman 相关性评估等技术。

**📊 数据集**

数据集包括 200 条标注过的 math、code、reasoning 提示，三种大模型（Qwen2.5、Llama‑3.1、gemma‑2）与六种后训练方式。

**📈 对比分析**

与单特征基线、多特征基线以及随机抽样对比；在 LOOS 评估中 Spearman ρ=0.90，top‑5 精度0.90，恢复的真实收益平均比随机提升 0.10。

**⚠️ 局限性**

局限在于仅针对可提取答案的数学和推理任务，并依赖奖励模型验证；对开放式生成、工具增强或未来模型的适用性未知。

---

## 214. Inducing Reasoning Primitives from Agent Traces

**arXiv ID:** 2606.02994 | [PDF](https://arxiv.org/pdf/2606.02994v1)

**作者:** Zhihan Lei `[一作]` (Carnegie Mellon University), William W. Cohen `[通讯]` (Carnegie Mellon University)

**通讯引用:** 37385 | [OpenAlex ID](https://openalex.org/A5051617344)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

从ReAct代理的成功轨迹中自动提取并聚类常见推理步骤，生成可调用的伪工具库，提升推理效率和准确率

**💡 创新点**

首次实现单遍、无监督的推理原语诱导，将不规则思考转化为可复用、可编辑的结构化工具，并显著超越源代理

**🔧 技术方法**

使用LLM分类标记、聚类与合成三步流程；通过提示生成伪工具签名与自然语言说明，在ReAct循环中调用

**📊 数据集**

六个子任务，涵盖叙事推理（谋杀、物体放置）、规则应用（NBA合同）、约束规划（团队分配、会议与旅行计划）

**📈 对比分析**

与零样本Chain‑of‑Thought、原始ReAct、专家手工拆解、Agent Workflow Memory、Program‑of‑Thoughts对比；诱导库在所有可比任务中均超过源代理（最大+44pp），与专家拆解相匹配或更优，且推理成本比AWM低约24%

**⚠️ 局限性**

对算术密集任务需要外部工具；合成伪工具的自然语言描述可能引入累积误差；跨任务族迁移尚未评估；高风险领域需人工审查所提取的偏差

---

## 215. A Systematic Evaluation of Current Architectures in Wind Power Forecasting

**arXiv ID:** 2606.02849 | [PDF](https://arxiv.org/pdf/2606.02849v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 216. G^2C-MT: Graph-Guided Context Selection for Document-Level Machine Translation

**arXiv ID:** 2606.03078 | [PDF](https://arxiv.org/pdf/2606.03078v1)

**作者:** Baijun Ji `[一作]` (Soochow University), Bohong Zhao `[通讯]` (Trip.com Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 G²C‑MT 框架，利用轻量级的有向无环图捕捉文档级翻译中的分层话语依赖，并通过深度偏置随机游走采样上下文路径，再将路径结构化为提示输入大语言模型进行翻译。

**💡 创新点**

创新点包括① 用语义、序列、关键词三维融合构建图，实现非线性话语依赖；② 采用深度偏置随机游走在图中寻找多跳上下文路径；③ 兼顾多路径采样与自一致性，提升模糊情境下的稳健性。

**🔧 技术方法**

核心技术包括：预训练句子/段落嵌入+TF‑IDF关键词提取；有向图构建与权重融合；深度偏置随机游走；路径逆序拼接构成提示；LLM（如 Gemini‑2.5‑Flash‑lite、DeepSeek‑v3、Qwen‑2.5‑72B 等）执行翻译。

**📊 数据集**

在 SAP 技术文档集（IT 领域）和 IWSLT‑2017 TED 会议文本集上进行评测，涉及多种语言对（如 EN↔VI、EN↔ZH、EN↔ID、EN↔FR、EN↔DE、EN↔JP 等）。

**📈 对比分析**

与句子级、固定窗口、语义检索、GRAFT、DelTA 等基线对比，G²C‑MT 在 d‑BLEU、BlonDe、d‑Prism、d‑Comet 等指标上均取得显著提升，提升幅度约 0.5–1.5 d‑BLEU；多路径采样进一步提升 0.5–0.6 d‑BLEU。

**⚠️ 局限性**

局限性：① 图构建仍需 O(N²) 复杂度；② 需要依赖大语言模型，推理成本随路径数增长；③ 对极长文档或极稀疏语义的句子，边权可能不足，导致路径质量下降。

---

## 217. Regret Pre-training: Bridging Prior and Posterior Views for Enhanced Knowledge Grounding

**arXiv ID:** 2606.03080 | [PDF](https://arxiv.org/pdf/2606.03080v1)

**作者:** Mingkuan Zhao `[一作]` (Xi'an Jiaotong University), Jiayin Wang `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 17272 | [OpenAlex ID](https://openalex.org/A5027360778)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Regret Pre-training 框架，在自监督预训练中将未来上下文视为特权信息，通过双视图（因果学生视图和未来条件教师视图）实现知识蒸馏；

**💡 创新点**

创新点在于利用同一模型参数在两种注意力掩码下并行生成学生与教师分布，只需额外一次无梯度前向推理即可将未来信息转移给因果模型；

**🔧 技术方法**

技术主要包括双视图 Transformer 架构、正向 KL 散度的 Regret Loss、两种教师视图配置（局部+1 未来 token 与全局双向但屏蔽目标）以及对 Mixture‑of‑Experts 进行的加载平衡和文档边界注意力掩码；

**📊 数据集**

使用 DCLM‑Baseline 语料库（约 4 B tokens）进行预训练，随后在 9 项零样本下游任务（BoolQ、ARC‑C、WinoGrande、PIQA、HellaSwag、CommonsenseQA、MMLU 等）进行评估；

**📈 对比分析**

与标准因果语言模型（Baseline）对比，双视图 Regret 方案在所有任务上均有提升：平均准确率从 30.2% 提升至 33.9%（全局教师）或 32.2%（局部教师），尤其 BoolQ 提升 18.1pp；

**⚠️ 局限性**

局限性包括：仅测试了两端教师配置，未探索中间范围；对模型参数没有影响但需要额外一次前向推理导致训练时间提升约 1.8–2.0 倍；在超大模型或更大数据规模下的可扩展性仍待验证。

---

## 218. Spectral-Progressive Thought Flow for Lightweight Multimodal Reasoning

**arXiv ID:** 2606.02842 | [PDF](https://arxiv.org/pdf/2606.02842v1)

**作者:** Yixian Shen `[一作]` (University of Amsterdam), Anuj Pathania `[通讯]` (University of Amsterdam)

**通讯引用:** 1089 | [OpenAlex ID](https://openalex.org/A5067055700)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 Spectral-Progressive Thought Flow（SpecFlow），通过在离散余弦域维护固定大小的视觉工作空间，并用流匹配与无监督分类器指导对其进行可控更新，从而在多模态空间推理中避免视觉 token 的累积。

**💡 创新点**

创新点在于将中间视觉思维压缩到频域，并按需逐步激活低频全局结构与高频细节；利用流匹配实现高效的确定性更新，并通过 classifier‑free guidance 让文本意图直接控制视觉状态。

**🔧 技术方法**

使用离散余弦变换、流匹配（ODE）与 CFG、VAE 潜在空间编码以及自回归文本模块相结合的技术框架。

**📊 数据集**

在 VSR、V-Star、EmbSpatial、Winoground 等多模态空间推理基准以及 Maze、MiniBehavior、FrozenLake 等空间决策任务上进行评估。

**📈 对比分析**

与 VoCoT、MVoT、DiffThinker 等基线相比，SpecFlow 在保持或提升准确率的同时将 FLOPs、延迟、KV 缓存等计算开销降低 1.6–2.1 倍，尤其在长推理深度时表现更优。

**⚠️ 局限性**

局限包括对低频/高频切分策略需手动调优，对极端细节需求场景的生成质量可能受限，以及在大规模视觉场景中频谱投影块大小和求解步数的敏感性仍需进一步研究。

---

## 219. WRIT: Write-Read Intensive Trajectory Synthesis for Multi-Turn User-Facing Agents

**arXiv ID:** 2606.02908 | [PDF](https://arxiv.org/pdf/2606.02908v1)

**作者:** Hengrui Gu `[一作]` (North Carolina State University), Kaixiong Zhou `[通讯]` (North Carolina State University)

**通讯引用:** 1323 | [OpenAlex ID](https://openalex.org/A5071607114)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Write-Read Intensive Trajectory Synthesis（WRIT）管道，用于在可执行环境中合成多轮用户交互轨迹，并通过双轴复杂度控制（写决策数与每个决策的阅读负担）生成高质量训练数据。

**💡 创新点**

创新点在于：①通过写决策数和阅读负担这两条轴，系统性地生成既包含多写任务又包含需要大量证据推理的读重任务；②加入用户行为脚本，模拟真实对话变异；③将合成任务与可执行环境结合，确保生成轨迹可验证、可执行。

**🔧 技术方法**

使用技术包括：LLM驱动的任务原型生成与参数实例化、读重请求生成与验证、用户行为脚本化、可执行环境模拟（agent‑user 交互）以及基于这些轨迹的全参数SFT微调。

**📊 数据集**

数据集：在 τ^2-bench（零售与航空两域）上合成 2000 条轨迹（各 1000 条），包含写重、读重和多写任务；与现有合成基线（APIGen‑MT、Simia、CoVe、AReaL）进行对比。

**📈 对比分析**

评估方法：在 τ^2-bench 上使用 Pass^1 与 Pass^4 指标，基于 4B Qwen3‑4B‑Instruct‑2507、Llama‑3.1‑8B‑Instruct、Qwen2.5‑14B‑Instruct 三个模型；WRIT 在所有模型上均显著提升，4B 版本实现 67.99% Pass1 与 45.73% Pass4，且推理 token 量比 GPT‑5.1 no‑think 低约 20%。

**⚠️ 局限性**

局限性：①未充分探索写决策与读重任务的组合（如多写任务中每个决策都需读重）；②未系统研究不同复杂度样本比例对模型性能的最优组合；③对极端对话变异或政策边界场景的覆盖仍有提升空间。

---

## 220. See Less, Specify More: Visual Evidence Budgets for Generalizable VLAs

**arXiv ID:** 2606.02735 | [PDF](https://arxiv.org/pdf/2606.02735v1)

**作者:** Yueh-Hua Wu `[一作]` (AIRoA), Kei Ota `[通讯]` (AIRoA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 S2 框架，将规划器与执行器分离，通过目标保持的层次化重标记和无标注的视觉证据预算，显著提升 VLA 模型的泛化能力。

**💡 创新点**

创新点在于：①在高层使用 VLM 生成局部执行指令以消除任务歧义；②在低层引入可学习的视觉证据门控，形成基于控制目标的证据瓶颈；③将两者结合，训练执行器在更精细、信息更少的接口下工作。

**🔧 技术方法**

使用的技术包括：大规模视觉语言模型（如 Kimi K2.5）进行上下文指令生成；基于门控网络的视觉证据掩码学习；联合训练包括无门控、门控和预算正则化损失；以及对标准 VLA 目标的 fine‑tune。

**📊 数据集**

数据集主要包括 LIBERO‑PRO、LIBERO、CALVIN 以及在 TX‑G2 与 HSR 机器人上收集的真实机器人演示数据。

**📈 对比分析**

实验与现有方法（π_0.5、X‑VLA、VLA‑Adapter、OpenVLA‑OFT 等）对比，S2 在 LIBERO‑PRO 的多种扰动场景下平均子任务成功率从 54.2% 提升至 79.0%，在 TX‑G2 与 HSR 的真实机器人任务中也分别提高了 15%–30%，并在加剧的杂物干扰场景中保持了高成功率。

**⚠️ 局限性**

局限性包括：对 VLM 生成的局部指令质量敏感；视觉证据预算需要手动调参；在极端视觉遮挡或未知对象下仍可能失效；以及在更大规模或更复杂的任务空间中验证其可扩展性尚未完成。

---

## 221. Echelon: Auditable Aggregate-Only Language-Model Adaptation Across Privacy Boundaries

**arXiv ID:** 2606.02958 | [PDF](https://arxiv.org/pdf/2606.02958v1)

**作者:** Hina Dixit `[一作]` (Decompute Inc.), Nevasini Sasikumar `[通讯]` (Decompute Inc.)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种边界优先的聚合式训练体系，强制禁止设备级模型状态跨边界导出，并在全局仅使用聚合后的边界增量进行同步。

**💡 创新点**

创新点在于将隐私边界作为系统不变式，配合缓冲半异步安全聚合、时延感知加权、参与窗口、proximal 本地目标及漂移感知同步控制，实现了可审计的聚合式训练；并提出有效漂移预算（B_eff）作为经验性的质量‑通信边界预测法则。

**🔧 技术方法**

采用安全聚合、缓冲半异步聚合、时延感知权重、参与窗口、proximal 目标、漂移感知控制，以及 LoRA 模型适配与 Llama 1B 微调。

**📊 数据集**

使用公开数据集 C4（C4‑XS/C4‑S）和 OpenWebText，均以 32 词长、6/4 采样批次为基准。

**📈 对比分析**

在 1B LoRA 适配下，在预算匹配（token、bytes、wall‑clock、sync‑count）下，聚合式训练在失真、ppl 及吞吐上与多种低通信基线相当或更优；在 WAN 延迟与非 IID 条件下，损失提升不超过 2.2%，吞吐保持 1% 以内。

**⚠️ 局限性**

局限包括：仅验证 1B LoRA、seq32 的规模，未扩展至更大模型或全参数训练；WAN 真实性有限；隐私保护仅限非导出与单轮安全聚合，未覆盖差分隐私或恶意对手；实验多为单种种子或预算匹配，未覆盖更广泛设置；有效漂移预算法则仅在当前训练堆栈和窗口内经验验证。

---

## 222. ZX-Calculus:Trace-Indexed Dependent Types and Epistemic Semantics

**arXiv ID:** 2606.03063 | [PDF](https://arxiv.org/pdf/2606.03063v1)

**作者:** Peng Chen `[一作]` (Beijing Language and Culture University), Peng Chen `[通讯]` (Beijing Language and Culture University)

**通讯引用:** 1410 | [OpenAlex ID](https://openalex.org/A5024986740)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

在MLTT上构造了ZX-Calculus，一种保守扩展，集成了轨迹索引类型、基于小册子（presheaf）的非单调知识语义和构造性AGM信念修正。

**💡 创新点**

核心创新包括：① 轨迹类型的显式事件接口和结构化归约；② 证明完全的可构造部分会合收缩算法与其四项完整性条件；③ 证明所有AGM八条后置条件的定理；④ 引入单步修订系统（SSRS）揭示AGM与小册子合成不一致的根源；⑤ 完整的Coq机理化（34条证明）。

**🔧 技术方法**

技术方法主要是：Martin‑Löf依赖型类型理论的扩展；轨迹类型的结构化归约和βη规则；可构造性证明与归约候选法（reducibility candidates）实现可判定性；小册子语义（presheaf semantics）与CwF（类别与家族）模型；算法证明与Coq机理化。

**📊 数据集**

无数据集，论文为形式化理论与证明，不涉及实验或数据评测。

**📈 对比分析**

比较方法为形式化证明与Coq机理化，未做实验评测；在理论层面已证明完整性、可归约性和一致性，且所有八条AGM后置条件均为可构造定理。

**⚠️ 局限性**

主要限制：尚未完成全部7条Coq机理化（如可归约性归约消除、AGM完整性等）；缺乏对无穷轨迹的支持；未给出BP‑comp充分条件；未实现对集合值小册子模型的语义完备性；整体工作仍停留在理论与形式化层面。

---

## 223. Report on the Designing Accountable Software Systems Workshop

**arXiv ID:** 2606.02804 | [PDF](https://arxiv.org/pdf/2606.02804v1)

**作者:** Catherine Albiston `[一作]` (UC Berkeley), Christopher Yoo `[通讯]` (University of Pennsylvania)

**通讯引用:** 1485 | [OpenAlex ID](https://openalex.org/A5068086685)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

组织并系统化了2024年DASS工作坊，对软件系统责任性展开跨学科讨论与需求梳理。

**💡 创新点**

首次将法律、社会学与计算机科学视角集成，通过参与式系统化方法识别责任缺口与研究方向。

**🔧 技术方法**

采用工作坊式聚焦讨论、调查问卷、笔记记录与亲和图法进行知识整合。

**📊 数据集**

基于对42项NSF DASS奖项目PI的问卷及78名工作坊参与者的访谈数据。

**📈 对比分析**

无传统性能指标，而是对比不同学科对责任维度的定义与实践，强调多视角共识的重要性。

**⚠️ 局限性**

局限在于缺乏实证验证、样本自报偏差及跨学科沟通成本高。

---

## 224. Auditable Climate Risk Intelligence from Fragmented ESG Data: Deterministic Orchestration and Imbalance-Aware Learning for Scope 1-3 Validation

**arXiv ID:** 2606.02604 | [PDF](https://arxiv.org/pdf/2606.02604v1)

**作者:** Karan Sehgal `[一作]` (University of Kent), Khawar Naveed Bhatti `[通讯]` (University of Kent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个 deterministic climate‑risk intelligence 框架，整合 provenance‑aware 统筹、imbalance‑aware ensemble 学习、temporal drift 检测、TreeSHAP 解释以及图形化 provenance 设计，用于实现可审计、可复现的 ESG 验证流程。

**💡 创新点**

创新点包括：① 定义并实现可复现的 deterministic orchestration 与 audit‑trace 完整性度量；② 通过 SMOTE 与集成学习解决稀缺 anomaly 的不平衡问题；③ 将 temporal drift 与公开气候风险数据嵌入模型；④ 引入 TreeSHAP 解释层提升治理可解释性；⑤ 提议图形化 provenance 架构供后续评估。

**🔧 技术方法**

技术手段包括：deterministic orchestration（基于事件触发的流程）；SMOTE（过采样）；梯度提升树模型（XGBoost、CatBoost、LightGBM）；SARIMA 与 SARIMA‑LSTM 时序预测；Isolation Forest 异常检测；TreeSHAP 解释；Graph‑Transformer/Graph‑Neural 预留设计；以及 calibration（ECE、Brier）与 audit‑trace 完整性评估。

**📊 数据集**

使用的数据集为公开发布的 synthetic ESG benchmark（≈68k 条记录，覆盖 Scope 1/2/3、231 个特征、4.7% anomaly、12.3% 缺失），并结合 WRI Aqueduct、ThinkHazard、Copernicus、NGFS 等公开气候风险数据进行 enrichment。

**📈 对比分析**

实验通过 stratified 5‑fold CV 与配对 Wilcoxon 检验，对比 threshold baseline、Logistic Regression、Isolation Forest、SARIMA、SARIMA‑LSTM、Random Forest、LightGBM、CatBoost 与 XGBoost；XGBoost 以 recall 0.74、F1 0.71、ROC‑AUC 0.93 的最佳表现击败所有基线，并实现 94.7% 的 audit‑trace 完整性；SMOTE 1:1 方案进一步提升召回率，提升治理安全性。

**⚠️ 局限性**

局限性在于：① benchmark 为 synthetic，缺乏真实企业系统的细节与噪声；② 气候风险数据自身的不确定性与预测误差；③ 代理特征估计可能引入误差；④ temporal drift 仍对标准演进与供应链延迟敏感；⑤ 图形化 provenance 架构未实证评估；⑥ 仅为可复现的原型，尚未在生产环境中验证。

---

## 225. Quantifying Side-Channel Leakage in Public Metrology Releases

**arXiv ID:** 2606.02934 | [PDF](https://arxiv.org/pdf/2606.02934v1)

**作者:** Faruk Alpay `[一作]` (Bahcesehir University), Taylan Alpay `[通讯]` (University of Turkish Aeronautical Association)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种基于统计的侧信道审计框架，用于量化公共科学和计量发布中潜在的泄露风险，特别是在EUV光刻计量中。通过分析功率谱密度（PSD）来评估发布的光谱信息如何泄露隐藏的过程参数。

**💡 创新点**

创新点在于提出了一种可重复的统计侧信道审计框架，能够量化在特定预算下，发布的光谱信息如何使得攻击者能够区分受保护的配方位。并且，提出了一个有限带宽传输泄露定律，提供了泄露的精确数学表达。

**🔧 技术方法**

使用了统计信息流和模板攻击的定量方法，结合了伽马通道和协方差加权的对数谱通道，计算了KL散度、Chernoff指数和保护位优势界限等。

**📊 数据集**

使用了EUV光刻计量中的粗糙度光谱作为数据集，进行了模型条件的案例研究，并计划在实际测量的发布上进行部署。

**📈 对比分析**

通过与理想模板对手的比较，审计报告了在不同预算下的最佳对数测试、有限库阈值和保护位优势。性能评估显示，随着发布带宽的增加，泄露信息的可区分性显著提高。

**⚠️ 局限性**

限制在于该框架依赖于模型假设，特别是在处理噪声和未建模的地板效应时，可能会影响泄露的量化。此外，审计的有效性在于对发布数据的准确性和重复性要求较高。

---

## 226. Consistent Yet Wrong: Evidence Insensitivity in Spatial Vision-Language Models

**arXiv ID:** 2606.02742 | [PDF](https://arxiv.org/pdf/2606.02742v1)

**作者:** S Divakar Bhat `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**通讯引用:** 6916 | [OpenAlex ID](https://openalex.org/A5048624196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ViewDiag，一套多视角评估协议和诊断工具，用来检测视觉‑语言模型（VLM）在空间距离推理任务中的“证据敏感性”，揭示跨视角一致性并不一定等同于几何理解。

**💡 创新点**

创新点：①首次将多视角一致性与内部隐藏状态、输出分布的聚集度相结合，构建“collapse quadrant”与“内部崩塌探针”诊断框架；②发现并量化了大部分 VLM 在多视角下表现出的“稳定但错误”（consistent‑yet‑wrong）现象；③提供了基准数据集与评估指标，促使未来研究超越单一准确率。

**🔧 技术方法**

主要技术：基于 Hypersim、ScanNet、KITTI360 构造 176 条同一物体对的多视角轨迹；使用 VQA 风格提示，统一距离度量为米；评估指标包括 MAE、L2 校正 MAE、Top‑1 质量、有效支持度、Evidence Sensitivity Score（ESS）以及隐藏状态余弦相似度/欧氏差异的内部崩塌分析。

**📊 数据集**

数据集：Hypersim（室内合成）、ScanNet（室内真实）、KITTI360（室外真实）三大来源，共 1308 个查询、176 条轨迹，2–10 个视角/轨迹，覆盖短距与长距环境。

**📈 对比分析**

比较方法：将 ViewDiag 的 VLM 结果与几何基线（ZoeDepth、DepthLM 等）以及多种主流 VLM（Qwen2‑VL、InternVL、MiniCPM、LLaVA、VILA、SpatialRGPT）进行同一评测；VLM 在 raw MAE 上落后几何基线，但在 ESS 上表现更差（高一致性+高误差）；几何基线在长距 KITTI 仍领先；展示了“collapse quadrant”中 VLM 集中于高误差高聚集区域。

**⚠️ 局限性**

Limitations：①诊断工具依赖可访问的隐藏表示；②评估仅覆盖距离推理任务，未扩展到更广泛的空间推理场景；③仅利用三种数据集，未检验更极端视角或动态场景；④未提出解决方案，只是揭示问题。

---

## 227. Do Matching Mechanisms Work with LLM Agents?

**arXiv ID:** 2606.03030 | [PDF](https://arxiv.org/pdf/2606.03030v1)

**作者:** Yukihiro Hoshino `[一作]` (University of Tokyo), Nariaki Nishino `[通讯]` (University of Tokyo)

**通讯引用:** 636 | [OpenAlex ID](https://openalex.org/A5079335652)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了标准匹配机制在大型语言模型（LLM）代理市场中的功能，比较了去中心化的自由谈判市场与包括多种代表性机制的中心化机制市场。

**💡 创新点**

研究发现机制市场在稳定性和效率方面通常优于自由谈判市场，并且LLM代理在报告偏好时的诚实率显著高于人类受试者。

**🔧 技术方法**

使用了多种匹配机制，包括延迟接受（DA）、效率调整延迟接受（EADA）、波士顿匹配、随机序列独裁（RSD）和顶级交易循环（TTC）。

**📊 数据集**

实验中使用了三种市场场景：劳动市场、高中入学考试和幼儿园选择，涉及不同的偏好配置。

**📈 对比分析**

机制市场在稳定性和效率方面的表现显著优于自由谈判市场，特别是DA机制在所有偏好配置下的稳定性达到至少86%。

**⚠️ 局限性**

研究的局限性在于，尽管LLM代理在某些方面表现出更高的理性和战略性，但其行为仍可能受到生成模型的概率性和自然语言上下文的影响。

---

## 228. Conditional Hypothesis Generation for LLM-Based Text Analysis with Researcher-Specified Covariates

**arXiv ID:** 2606.03029 | [PDF](https://arxiv.org/pdf/2606.03029v1)

**作者:** Paiheng Xu `[一作]` (University of Maryland), Wei Ai `[通讯]` (University of Maryland)

**通讯引用:** 6996 | [OpenAlex ID](https://openalex.org/A5011364421)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出条件假设生成框架，利用研究者指定的协变量来引导文本差异假设的发现

**💡 创新点**

首次将协变量驱动的条件特征选择方法（交互式Lasso与去均值再加逆频重加权Lasso）应用于文本假设生成，并明确区分了层级不平衡与符号反转两类统计挑战

**🔧 技术方法**

基于稀疏自编码器生成可解释特征，使用L1正则化的逻辑回归、交互项Lasso以及去均值再加重加权Lasso实现条件特征筛选

**📊 数据集**

在美国国会法案摘要语料上构造两种合成实验（层级不平衡与符号反转），并在政治语言数据集（e.g., Congressional Bills）与数学课堂转录数据集上进行专家评估

**📈 对比分析**

与传统全局假设生成基线（如SAE‑Lasso、分离得分）和直接LLM提示基线比较，去均值再加逆频重加权Lasso在层级不平衡场景下与oracle接近，交互式Lasso在符号反转场景下显著优于全局基线

**⚠️ 局限性**

方法依赖于研究者预先指定的协变量，连续或多值协变量处理困难；交互式Lasso在高维稀疏情况下不稳定；去均值方法假设不出现符号反转，若违反则无法恢复差异

---

## 229. SEA-Embedding: Open and Reproducible Text Embeddings for Southeast Asia

**arXiv ID:** 2606.03027 | [PDF](https://arxiv.org/pdf/2606.03027v1)

**作者:** Peerat Limkonchotiwat `[一作]` (AI Singapore), Jian Gang Ngui `[通讯]` (AI Singapore)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了SEA‑Embedding文本嵌入框架，公开所有代码、数据与实验环境，实现可复现的Southeast Asian语言嵌入；

**💡 创新点**

创新点在于：①提出完全公开且可复现的完整训练流水线；②系统化评估数据组成、目标设计和基础模型对鲁棒性的影响；③设计对称InfoNCE对比学习加焦点加权和相似度分布匹配（KD+KL）双阶段训练目标，显著提升低资源语言表现；

**🔧 技术方法**

技术包括对称InfoNCE对比学习（SCL）配合焦点加权、相似度分布匹配（SDM）与KL散度、使用大内存队列、对多种基础编码器（BERT/ModernBERT/E5）进行迁移训练；

**📊 数据集**

使用公开数据：245M条通用文本、14M条指令文本，来源于FineTranslations、CCMatrix、MIRACL、SEA‑Instruct‑2602等；

**📈 对比分析**

在SEA‑BED基准上与Qwen、BGE‑M3、Cohere、Harrier等多项SOTA模型对比，SEA‑Embedding在语言平均分0.800，超越此前最高0.789；小模型同样显著提升，低资源语言表现尤为突出；

**⚠️ 局限性**

局限性包括未覆盖所有SEA语言与方言、缺乏特定领域（如金融）数据、未系统探讨数据规模、标注质量及推理时效率等方面。

---

## 230. DeskCraft: Benchmarking Desktop Agents on Professional Workflows and Human-in-the-Loop Collaboration

**arXiv ID:** 2606.03103 | [PDF](https://arxiv.org/pdf/2606.03103v1)

**作者:** Wenkai Wang `[一作]` (Zhejiang University), Shengyu Zhang `[通讯]` (Zhejiang University)

**通讯引用:** 3113 | [OpenAlex ID](https://openalex.org/A5100757082)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DeskCraft桌面GUI基准，用于评估长周期、交互式专业软件工作流程的智能代理。

**💡 创新点**

构建了L1/L2/L3难度层次、可执行交互协议以及覆盖视频、音频、3D等专业工具的任务集，填补了现有基准在长周期和人机协作方面的空白。

**🔧 技术方法**

采用实时桌面执行、基于视觉的操作指令、MLLM用户模拟器以及结构化验证器，实现了可执行的任务执行与结果检查。

**📊 数据集**

收集了538个来自11款专业软件的真实工作流任务，包含386个标准任务和152个交互任务，并提供约279个资产文件。

**📈 对比分析**

在标准和交互拆分上评估18个代理（含GPT-5.4、Kimi-K2.6等），最佳模型在标准任务仅达33.8%成功率，交互任务27.6%，长步骤预算虽略提升，但整体表现仍远低于预期。

**⚠️ 局限性**

仅覆盖英语/中文指令、脚本化交互、固定应用与工作流，未能覆盖多语言、真实用户多变交互与更广泛桌面场景。

---

## 231. DELTAMEM: Incremental Experience Memory for LLM Agents via Residual Trees

**arXiv ID:** 2606.03083 | [PDF](https://arxiv.org/pdf/2606.03083v1)

**作者:** Haoran Tan `[一作]` (Renmin University of China), Xu Chen `[通讯]` (Renmin University of China)

**通讯引用:** 23332 | [OpenAlex ID](https://openalex.org/A5100385692)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于残差树的双树经验记忆框架，分别存储任务策略和环境知识，利用增量 delta 节点压缩存储并通过失败惩罚的全局检索与链条重构提供无冲突的记忆上下文。

**💡 创新点**

创新点包括（1）将经验拆分为任务树和环境树实现结构解耦；（2）使用残差树仅存增量差异，消除冗余与检索冲突；（3）全局检索结合失败惩罚；（4）自主记忆凝聚机制将高频路径升级为新的根节点，形成自组织的层次结构。

**🔧 技术方法**

技术上结合了 LLM（DeepSeek-V4-flash）进行推理与经验抽取，使用 E5-base-v2 做嵌入相似度计算，ReAct 框架进行任务交互；残差树结构与链条重构、失败惩罚检索、自动凝聚算法共同实现。

**📊 数据集**

在 ALFWorld、ScienceWorld 与 WebShop 三大交互决策基准上进行评估，分别包含多种任务类别并分别设置 Seen/Unseen 两个 split。

**📈 对比分析**

与 No Memory、Synapse、AWM、ReasoningBank 四种基线对比，平均奖励（AvgRew）显示该方法在 ALFWorld 和 ScienceWorld 的 Seen/Unseen 上持续领先，WebShop 上保持竞争力；在离线预构建记忆库实验中也保持最高或相近性能。

**⚠️ 局限性**

主要局限包括：依赖 LLM 的经验提取导致额外推理成本；提取质量受 LLM 能力影响，低质量提取可能引入噪声；检索阈值与凝聚阈值需针对每个任务域手工调参；缺乏对旧经验的时间衰减机制，可能导致过时知识误导。

---

## 232. ToolGate: Token-Efficient Pre-Call Control for Tool-Augmented Vision-Language Agents

**arXiv ID:** 2606.03054 | [PDF](https://arxiv.org/pdf/2606.03054v1)

**作者:** Anjie Liu `[一作]` (Hong Kong University of Science and Technology), Jun Wang `[通讯]` (University College London)

**通讯引用:** 46382 | [OpenAlex ID](https://openalex.org/A5100384686)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了视觉‑语言代理的预调用工具执行控制，提出 ToolGate 控制器决定是否执行已提议的感知工具调用。

**💡 创新点**

通过外部轻量化二分类门，基于轨迹文本和结构特征实现局部工具调用选择，显著降低工具使用成本并在匹配域下提升准确率。

**🔧 技术方法**

使用 ReAct式 VLM 代理、Qwen3‑VL 30B/235B、冻结句子嵌入加逻辑回归门、强制答案探针作为代理标签，以及工具调用跳过消息。

**📊 数据集**

评估使用五个视觉语言基准（V*Bench、CV‑Bench、HR‑Bench‑4k/8k、MME‑RealWorld‑Lite）以及 GQA、A‑OKVQA、TextVQA、DocVQA、SEED‑Bench、RSVL‑MQA 进行跨域训练。

**📈 对比分析**

与基线 ReAct 以及提示自调控制对比，ToolGate 在跨域设置下将 token 成本降至 64–69% 以上，保持平均准确率；在匹配域下准确率提升 1.65 分且成本进一步降低。

**⚠️ 局限性**

仅针对 ReAct 风格的 VLM 与固定工具集，代理门使用代理标签无法捕捉延迟或分散效益；在高风险场景下可能错误跳过重要证据；未验证在其他模型/协议上的迁移。

---

## 233. FCUS-rPPG: A Fast-Converging Unsupervised Framework for Remote Photoplethysmography via Gradient Oscillation Suppression

**arXiv ID:** 2606.03050 | [PDF](https://arxiv.org/pdf/2606.03050v1)

**作者:** Jiajie Li `[一作]` (Hefei University of Technology), Juan Cheng `[通讯]` (Hefei University of Technology)

**通讯引用:** 6146 | [OpenAlex ID](https://openalex.org/A5029209486)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种快速收敛且通用的无监督远程光电容积描记（rPPG）框架FCUS-rPPG。

**💡 创新点**

创新点在于将多频谱生理协同和低维流形假设结合，设计低维谱共享骨干以及统一的梯度、损失景观、特征正则三层加速优化。

**🔧 技术方法**

采用低维谱共享骨干（LLP+MSF+SAM+WRD）、后验梯度屏蔽（PGM）、损失景观平滑（LLS）以及噪声归零空间正则（NNR）等技术。

**📊 数据集**

使用五个公开/内部数据集：UBFC-rPPG、PURE、BSIPL-RPPG、BSIPL-motion、MMPD。

**📈 对比分析**

与传统、监督、无监督基线对比，FCUS-rPPG在单次 epoch 训练即可获得SOTA交叉数据集性能，平均HR误差仅0.49 bpm，训练时间仅40 秒。

**⚠️ 局限性**

局限在于对光照强度阈值、频率带宽等超参数敏感，且在极端运动或低光照环境下仍可能出现收敛不稳。

---

## 234. RelGT-AC: A Relational Graph Transformer for Autocomplete Tasks in Relational Databases

**arXiv ID:** 2606.03040 | [PDF](https://arxiv.org/pdf/2606.03040v1)

**作者:** Phillip Jiang `[一作]` `[通讯]` (Appsofa LLC), Phillip Jiang (Appsofa LLC)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了Relational Graph Transformer for Autocomplete（RelGT-AC），专为关系数据库中的自动补全任务设计。

**💡 创新点**

主要创新点包括：①列遮蔽策略防止直接读取目标列；②统一任务头支持回归、二分类、多分类和预测四种任务；③使用TF‑IDF文本编码器自动捕获自由文本列中的词汇信息。

**🔧 技术方法**

技术实现基于RelGT框架，结合HeteroGraphSAGE局部聚合、Transformer全局注意力、列遮蔽、TF‑IDF文本向量化与线性投影，以及统一的多任务输出头。

**📊 数据集**

在RelBench v2的三个数据集（rel‑trial、rel‑f1、rel‑stack）上共7个自动补全任务进行实验。

**📈 对比分析**

与GraphSAGE基线和单表LightGBM对比，RelGT-AC在所有3个回归任务上分别提升约0.22–0.44 R²，在部分分类任务上也表现优于或相近，TF‑IDF编码对文本丰富任务提升多达10点AUROC。

**⚠️ 局限性**

局限性包括：需要针对每个数据库进行专门微调、无法零样本迁移、对超大规模数据库（如rel‑ratebeer）内存要求高，以及手工指定的相关列去除仍需改进自动化检测。

---

## 235. The Deliberative Illusion: Diagnosing Factual Attrition and Stance Homogenization in Multi-Agent LLM Deliberation

**arXiv ID:** 2606.03032 | [PDF](https://arxiv.org/pdf/2606.03032v1)

**作者:** Herun Wan `[一作]` (Xi'an Jiaotong University), Min-Yen Kan `[通讯]` (National University of Singapore)

**通讯引用:** 11492 | [OpenAlex ID](https://openalex.org/A5066305082)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种评估框架 DelibTrace，用来衡量多智能体 LLM 在讨论过程中是否保留了必要的事实与立场多样性，并通过实验发现多智能体讨论往往导致关键信息丢失、立场同质化，从而产生“讨论幻觉”。

**💡 创新点**

创新点在于将讨论视作可追踪的信息流，构建原子事实（atomic facts）并标注为问题关键性；设计系统级与代理级事实保留度与立场熵指标；以及通过多种拓扑、模型族和干预方法系统性探测事实流失与立场收敛的关系。

**🔧 技术方法**

核心技术包括：1）使用 GPT-5 进行原子事实抽取与精炼；2）在多智能体环境中分配部分证据与先验立场；3）三种讨论拓扑（全连、树形、链式）实现信息交换；4）使用 Jaccard 指数跟踪事实保留；5）利用基于 LLM 的判定器提取事实与立场；6）对恶意注入进行安全性评估。

**📊 数据集**

实验数据集主要有两类：1）伦理场景，来源于 Scruples 的 AITA 故事，构建 710 条实例；2）新闻场景，使用公开新闻语料库生成 1,044 条实例，均包含问题查询与标注的关键事实。

**📈 对比分析**

在与单一 LLM 直接推理、模型同质/异质、多种拓扑的对照实验中，系统级关键信息保留率从讨论开始后迅速下降，最终下降 21.7%–72.4%；代理级保留率更低（60.5%–84.8%）。同时，立场熵显著降低，表明立场趋同。相比基线（无讨论或单模型推理），多智能体讨论并未显著提升判断正确率，反而因信息丢失导致 19.2% 的错误结论。对恶意注入实验显示，讨论过程会放大错误信息的传播。

**⚠️ 局限性**

局限性包括：1）缺乏对“有益压缩”与“有害流失”的细粒度判定，无法确定哪些事实丢失是可接受的；2）实验环境为受控信息流，未覆盖动态检索、开放式立场或人与代理的交互；3）评估依赖 LLM 生成的事实与立场，可能受模型偏差影响；4）仅考察单一任务域，跨域泛化待验证。

---

## 236. AUDITFLOW: Executable Symbolic Environments for Structured Financial Reporting Verification

**arXiv ID:** 2606.03031 | [PDF](https://arxiv.org/pdf/2606.03031v1)

**作者:** Yan Wang `[一作]` (Fin AI), Víctor Gutiérrez-Basulto `[通讯]` (Cardiff University)

**通讯引用:** 569 | [OpenAlex ID](https://openalex.org/A5060245641)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了AuditFlow框架，用多代理在图结构化的XBRL审计环境中完成财务审计验证；

**💡 创新点**

创新点在于将搜索与确定性验证分离，构建双图符号环境并通过Typed deterministic工具、三代理协议及证据聚合实现可追溯、可验证的审计判定；

**🔧 技术方法**

使用了双图结构（US‑GAAP taxonomy + XBRL filing graph）、Typed工具调用、三代理（Compliance、Forensic、Senior）互动、Dempster‑Shafer证据聚合等技术；

**📊 数据集**

实验数据集为FinAuditing FinMR子集（67个实例），覆盖3个Data Quality Committee规则（DQC.US.0015、DQC.US.0117、DQC.US.0126）；

**📈 对比分析**

与7个基线（FinAuditing、Herculean、Direct LLM、Vanilla RAG、GraphRAG、TreeRAG、Single Agent）以及多种LLM后端（GPT‑5.5、GPT‑4o、Claude Sonnet 4.6、Qwen‑397B等）进行比较；在GPT‑5.5上实现82.09%联合准确率，超过基线14.93个百分点，去除确定性检查后准确率骤降至17.91%；整体表现稳健；

**⚠️ 局限性**

局限性包括仅评估67个实例和3个规则，未覆盖全部XBRL审计规则；仅适用于完整标记化的XBRL文件，对扫描或缺失标签的报表无效；弱模型在达到确定性检查前就可能失败；系统仍需人类审计师进一步验证，不能直接替代专业审计判断。

---

## 237. PhotoCraft: Agentic Reasoning with Hierarchical Self-Evolving Memory for Deep Image Search

**arXiv ID:** 2606.03099 | [PDF](https://arxiv.org/pdf/2606.03099v1)

**作者:** Kailin Lyu `[一作]` (Tencent Inc.), Jie Zhou `[通讯]` (Tencent Inc.)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 PhotoCraft，一种训练无关的分层记忆框架，用于提升多步图像检索代理的上下文推理与经验迁移。

**💡 创新点**

通过引入工作记忆、情节记忆和语义记忆三层结构，实现动态压缩、反思校验以及基于成功轨迹的技能抽象，实现持续进化与跨任务迁移。

**🔧 技术方法**

采用多模态大型语言模型（MLLM）作为控制器，配合多模态工具、工作记忆压缩、情节记忆审核与语义记忆技能库，形成自回归推理循环。

**📊 数据集**

在DISBench（基于YFCC100M的个人照片检索基准）上进行实验。

**📈 对比分析**

与 ImageSeeker 基线在相同工具与内存预算下对比，使用 EM 与 F1 评估，PhotoCraft 在所有指标上均超过基线，提升幅度可达 18.5%，且保持跨模型鲁棒性。

**⚠️ 局限性**

仅在规模有限的 DISBench 上评估，技能覆盖受成功轨迹多样性限制，且缺乏大规模真实个人照片数据的验证。

---

## 238. From Long News to Accurate Forecast: Importance-Aware Fusion and PRM-Guided Reflection for Time Series Forecasting

**arXiv ID:** 2606.03097 | [PDF](https://arxiv.org/pdf/2606.03097v1)

**作者:** Mingyang Liu `[一作]`, Linqi Song `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

未提供论文内容，无法总结。

**💡 创新点**

未提供论文内容，无法总结。

**🔧 技术方法**

未提供论文内容，无法总结。

**📊 数据集**

未提供论文内容，无法总结。

**📈 对比分析**

未提供论文内容，无法总结。

**⚠️ 局限性**

未提供论文内容，无法总结。

---

## 239. Hierarchical Federated Learning with Dynamic Clustering and Adaptive Regularization for Robust Infrastructure Inspection

**arXiv ID:** 2606.03084 | [PDF](https://arxiv.org/pdf/2606.03084v1)

**作者:** Yuhu Feng `[一作]` (Hokkaido University), Miki Haseyama `[通讯]` (Hokkaido University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5040047120)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了联邦学习在结构健康监测中的“双重异质性”问题，提出层次化动态聚类与自适应正则化相结合的框架。

**💡 创新点**

创新点在于宏观层使用梯度动态聚类实现自动专家分组，微观层引入基于标签不平衡和梯度差异的非IID强度自适应正则化（DRAPR）。

**🔧 技术方法**

采用了联邦学习、梯度相似度聚类、Cosine相似度度量、Proximal正则化、标签不平衡评估等技术。

**📊 数据集**

使用日本全国道路设施检查数据库xROAD中的77,890张结构损伤图像，进行多标签分类任务。

**📈 对比分析**

与FedAvg、FedProx、SCAFFOLD、MOON、IFCA等基线方法比较，在5类和20类任务上，Clustered-DRAPR在准确率、召回率和F1-score上均显著提升，收敛速度更快。

**⚠️ 局限性**

局限性包括假设同步通信、仅处理二维图像，未考虑异步联邦学习或多模态（图像+传感器）融合。

---

## 240. What Do Students Learn? A Feature-Level Analysis of Dark Knowledge

**arXiv ID:** 2606.03052 | [PDF](https://arxiv.org/pdf/2606.03052v1)

**作者:** Seungu Kang `[一作]` (Yonsei University), Songkuk Kim `[通讯]` (Yonsei University)

**通讯引用:** 576 | [OpenAlex ID](https://openalex.org/A5075315705)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `8d10c613-917e-4880-9716-17789f50e119` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用 Interaction Tensor 对学生模型特征学习进行量化分析，并提出基于混淆矩阵的无教师自蒸馏方法 Confusion Distillation。

**💡 创新点**

创新点在于将混淆矩阵视为暗知识的替代，使用指数移动平均平滑的混淆信息作为软目标，实现无教师的特征级蒸馏。

**🔧 技术方法**

使用 Interaction Tensor、EMA 混淆矩阵更新、温度软化、交叉熵与 KL 损失混合等技术。

**📊 数据集**

实验采用 CIFAR-100 数据集。

**📈 对比分析**

与基线、CS‑KD、PS‑KD 以及标准 KD 进行比较，CD 在 ResNet‑18/34/50 上分别提升约 1.2%–1.4% 的 Top‑1 精度，并在训练效率上优于 PS‑KD。

**⚠️ 局限性**

局限在于仅验证于图像分类任务，对噪声样本的混淆信息敏感，且对更大模型或不同任务的适用性尚待进一步验证。

---

## 241. Capability Advertisement as a Market for Lemons: A Trust Layer for Heterogeneous Agent Networks

**arXiv ID:** 2606.03034 | [PDF](https://arxiv.org/pdf/2606.03034v1)

**作者:** Gaurav Naresh Mittal `[一作]` `[通讯]`, Gaurav Naresh Mittal

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了一个在LLM代理网络中解决能力广告不可信问题的 Trust Layer 层，利用概率描述、挑战/鉴证和声誉机制，促成可分离均衡。

**💡 创新点**

创新点在于把代理能力广告视为柠檬市场，并将经济学的信号、筛选和声誉三大手段映射为可插拔的协议层，首次提供理论证明能消除“自信错误”失败。

**🔧 技术方法**

使用了概率能力描述（信号）、挑战/鉴证（筛选）、可追溯的声誉记录与漂移检测等技术，兼容 MCP 与 A2A 等现有代理协议。

**📊 数据集**

没有收集真实数据集；通过模拟实验和对现有基准报告的引用来验证可靠性分布与链深度影响。

**📈 对比分析**

通过理论分析与仿真对比，显示在 Trust Layer 下单跳可靠性可保持在 0.9+，链深度提升时可靠性衰减显著减缓；相比纯信任协议，整体可靠性提升 20‑30%。

**⚠️ 局限性**

局限性包括：自我校准启动困难、可能的协同攻击、相关错误导致筛选失效、声誉游戏与 Sybil 威胁、以及无法彻底解决根本信任缺失问题。

---

## 242. Audio Spotforming via Post-Filtering Using Cross-Array Non-target Estimates

**arXiv ID:** 2606.03028 | [PDF](https://arxiv.org/pdf/2606.03028v1)

**作者:** Yuto Ishikawa `[一作]` (CyberAgent), Kouei Yamaoka `[通讯]` (University of Tokyo)

**通讯引用:** 169 | [OpenAlex ID](https://openalex.org/A5005122247)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种利用跨阵列非目标音频估计进行高效后滤波的音频spotforming方法，避免使用低秩近似；

**💡 创新点**

创新点在于利用一阵列难以分离的目标方向非目标成分可在其他阵列中空间分离，从而在后滤波阶段直接基于跨阵列非目标估计构造Wiener滤波器；

**🔧 技术方法**

核心技术包括基于GC-ILRMA的空间滤波、交叉阵列非目标成分建模、逆伽马先验实现稀疏约束以及ME算法求解迭代更新；

**📊 数据集**

使用JVS语音数据集的100名说话者作为目标与干扰语音，利用Pyroomacoustics生成的室内声学冲激响应进行模拟；

**📈 对比分析**

与传统使用NMF或NTF进行低秩后滤波的spotforming方法相比，实验表明在SDR、SIR、PESQ、STOI等指标上均能获得更高分数，尤其在不使用先验时SDR和PESQ显著提升；

**⚠️ 局限性**

主要局限在于对阵列同步误差的假设较弱（仅补偿残差），且在极端噪声或极高回声条件下仍可能受限，且实现仍需多阵列硬件支持。

---

## 243. NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation

**arXiv ID:** 2606.03159 | [PDF](https://arxiv.org/pdf/2606.03159v1)

**作者:** NVIDIA `[一作]`, Zian Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出 OmniDreams——一种基于动作条件化、世界场景地图控制的生成式世界模型，能够实时合成多视角摄像头观测，并与 Alpamayo 1 策略及 AlpaSim 仿真器无缝集成，支持长时序闭环评估。

**💡 创新点**

创新点在于将大规模视觉预训练的 Cosmos‑Predict 2.5 迁移到 AV 任务，通过 Diffusion Forcing、Self‑Forcing + DMD 蒸馏、KV 缓存和因果扩散实现高帧率（单摄 68 FPS、四摄 105 FPS）且参数仅 2 B，显著优于 10 B 的 VLA 对标；同时提供可控文本提示、动态地图控制与多尾场景生成能力。

**🔧 技术方法**

采用技术包括：动作条件化扩散模型（Cosmos‑Predict 2.5 基础），Diffusion Forcing 训练、Self‑Forcing 逐步蒸馏 + DMD、局部窗口注意力、KV 缓存、光栅化的世界场景地图、文本提示编码、LightVAE/TAE 编码器、FlashDreams 推理加速（CUDA‑Graph、局部 KV、轻量控制分支）。

**📊 数据集**

训练数据为两大 AV 数据集：RDS（约 3 M 20 s 剧集，7 摄像头）用于 mid‑train，RDS‑HQ‑1M（1.14 M 剧集，10/20 s 片段）用于后训练/finetune；覆盖 15 国 15 交通场景，包含高清地图、3D 动态检测、文本提示。

**📈 对比分析**

通过与重建仿真 NuRec 在 501 场景的闭环评估，OmniDreams 保持与 NuRec 同一策略排名，且在轨迹偏离时视觉质量更稳定；在多视角生成、长时序一致性、长尾场景覆盖等方面优于现有重建与视频生成方法，并实现单摄 68 FPS、四摄 105 FPS 的实时推理。

**⚠️ 局限性**

局限性包括：对极端外视角仍需后训练；对极大非结构化物体生成仍不稳定；推理速度虽已加速但仍高于传统重建；模型对世界地图和文本提示的依赖较强，缺失时可能产生显著偏差。

---

## 244. Disentangling Visual and Factual Correctness in LVLMs' Visualization Literacy

**arXiv ID:** 2606.03142 | [PDF](https://arxiv.org/pdf/2606.03142v1)

**作者:** Soohyun Lee `[一作]` (Seoul National University), Jinwook Seo `[通讯]` (Seoul National University)

**通讯引用:** 4519 | [OpenAlex ID](https://openalex.org/A5012388103)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过设计新的评测框架和计量指标，系统分析大规模视觉语言模型（LVLM）在可视化解释任务中的真实能力，区分了视觉正确性与事实正确性，并探讨了模型在视觉与事实冲突时的优先级及可通过提示工程调节的可能性。

**💡 创新点**

提出四象限可视化/事实纠结框架、Counterfactual Visualization Literacy Assessment Test (CVLAT)以及视觉-事实优先级指数(VFRI)，从而解耦视觉推理与预训练知识对回答的影响，并验证了提示工程的非对称可控性。

**🔧 技术方法**

基于多模态LLM推理的实验设计，使用多种提示（Normal、Explain、事实优先、视觉优先），引入能力归一化和纠错评分的统计方法；同时对1512款公开与专有模型进行大规模批量推理。

**📊 数据集**

使用VLAT、reVLAT、CVLAT三套标准测试集，CVLAT包含48道冲突题目，VLAT与reVLAT分别为53道，另外通过人类基准（N=30）对CVLAT进行对照。

**📈 对比分析**

与人类、各大模型家族（GPT、Claude、Gemini、Llama、Gemma、Qwen、Grok）比较，结果显示：部分专有模型（如Gemini‑3.1‑Pro、Claude‑Opus‑4.7）在标准VLAT上接近或超过人类水平，但在reVLAT与CVLAT上表现明显下降，表明其高分多依赖事实记忆；CVLAT揭示10款模型偏向事实、2款偏向视觉，提示工程对部分模型能显著调转优先级。

**⚠️ 局限性**

研究仅覆盖常见柱形/折线/散点等基础图表，未探究更复杂图形或非数值编码；CVLAT仅对数值冲突进行测试，未评估颜色、布局等维度；模型数量虽多但多为单一版本，缺少跨版本纵向比较；提示工程效果高度模型依赖，未给出通用的可控方法；缺少跨语言或交互式可视化场景的验证。

---

## 245. TiWeaver: Unified Temporal Dynamics Modeling via Contextual Patching

**arXiv ID:** 2606.03121 | [PDF](https://arxiv.org/pdf/2606.03121v1)

**作者:** Zhe Li `[一作]` (East China Normal University), Bin Yang `[通讯]` (East China Normal University)

**通讯引用:** 50727 | [OpenAlex ID](https://openalex.org/A5100355773)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了 TiWeaver 框架，用于多变量时间序列预测，结合自适应分块和细粒度异步依赖提取，能够统一处理正则、异频、缺失和事件驱动等多种异质序列。

**💡 创新点**

创新点在于：1）设计了图引导自适应分词器 G^2AT，利用时间密度和表示一致性动态划分高上下文连贯的补丁；2）提出细粒度异步依赖提取器 FADE，采用掩码跨通道注意力捕获异步交互，并结合 Transformer 编码长程历史，实现细粒度的跨通道依赖建模。

**🔧 技术方法**

使用的技术包括图注意网络（GAT）进行局部时序建模，时间嵌入，Mask Cross Attention、Encoder-only Transformer、动态阈值和多阶段聚合的自适应分词器，以及多尺度补丁表示的融合。

**📊 数据集**

实验所用数据集为 12 个真实世界数据，涵盖四类 MTS：Regular（Exchange、Weather、ZafNoo）、Heterogeneous-frequency（对上述序列按不同比例下采样得到）、Missing-value（随机删除观测值）以及 Event-driven（Human Activity、PhysioNet2012、USHCN）。

**📈 对比分析**

与 14 种基线模型（包括 NeuralFlows、tPatchGNN、GraFITi、Hi-Patch、APN、mTAND、PrimeNet、FEDformer、PatchTST、iTransformer、TimesNet、PDF、TimeMosaic 等）在 MAE、MSE 上对比，TiWeaver 在所有数据集上均取得最优或第二优，平均 MAE 减少约 7%，在 USHCN 数据集上提升 25%，且训练时间与参数量与其它方法相当或更低。

**⚠️ 局限性**

局限性：对极高缺失率或极稀疏事件驱动序列仍可能表现不佳；性能对最小补丁大小 P_min 与阈值 τ 的设置敏感；未对多尺度频谱特性做显式建模；在分布漂移或在线学习场景下的鲁棒性尚未验证。

---

## 246. FAF-CD: Frequency-Aware Fusion for Change Detection under Imperfect Multimodal Remote Sensing

**arXiv ID:** 2606.03114 | [PDF](https://arxiv.org/pdf/2606.03114v1)

**作者:** Yufan Wang `[一作]` (University of South Florida), Chandra Kambhamettu `[通讯]` (University of South Florida)

**通讯引用:** 4755 | [OpenAlex ID](https://openalex.org/A5076482730)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种频域感知的混合框架FAF-CD，用DINOv3预训练的ConvNeXt编码器、三分支融合模块（空间、全频 FFT、局部 Haar-DWT）以及VMamba解码器，实现高效的多尺度时序变更检测。

**💡 创新点**

创新点在于将空间、全频和局部频率特征三路并行融合，并通过自适应门控动态聚合，显著降低伪变更误检，提高异源 EO‑SAR 和光学监测下的鲁棒性。

**🔧 技术方法**

采用 ConvNeXt-L/ConvNeXt-Base 作为编码器，使用 DINOv3 大规模预训练；三分支融合模块包含 deformable 交叉注意力、Real FFT、Haar 2D‑DWT；解码采用线性复杂度的 VMamba 结构；训练时使用加权交叉熵，利用 LVD-1689M 进行预训练。

**📊 数据集**

在 LEVIR‑CD、WHU‑CD 两大光学二元变更检测数据集上验证基本性能，并在 BRIGHT 公开的异源 EO‑SAR 灾害映射数据集上测试多模态适配。

**📈 对比分析**

与 CNN、Transformer、Mamba 以及 NeXt2Former‑CD 等先进方法对比，FAF‑CD 在 LEVIR‑CD/WHU‑CD 上取得 0.924/0.859、0.955/0.914 的 cF1/cIoU，并在 BRIGHT 任务上在 tc‑mIoU/mAP 与伪变更稳健性上优于 NeXt2Former‑CD；相对模型的 FLOPs 下降约 15% 并保持或提升准确率。

**⚠️ 局限性**

局限性包括模型参数量较大（约 417M）、对大规模预训练依赖强、在纯光学干净数据上的提升有限，以及需要进一步探索更轻量的频率学习模块以应对更极端的光照与模态偏移。

---

## 247. HyperPatch: Sequential Knowledge Editing Under n-ary Structural Drift

**arXiv ID:** 2606.03179 | [PDF](https://arxiv.org/pdf/2606.03179v1)

**作者:** Yu-Kai Chan `[一作]` (National Yang Ming Chiao Tung University), Meng-Fen Chiang `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 297 | [OpenAlex ID](https://openalex.org/A5024176415)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 HyperPatch 框架，使用超图结构和增量式 Topology LoRA 实现对大语言模型的顺序知识编辑，保持事件原子性并避免结构漂移导致的检索失配。

**💡 创新点**

创新点在于：① 用超图把 n-ary 事件作为单一超边保留事件完整性；② 采用 SimHash LSH 实现 O(1) 的超边定位；③ 结合双重流形检索（语义 + 结构）与 LoRA 适配，解决 Structure‑Conditioned Knowledge Transfer Failure (SKTF)。

**🔧 技术方法**

使用的技术包括：超图神经网络（HGNN）+ 对比学习构建结构先验；SimHash 基于 LSH 的超边哈希索引；LoRA 低秩适配器实现参数保持的增量更新；双流形最大内积检索与语义层对齐；LLM 驱动的查询分解与推理链构建。

**📊 数据集**

主要数据集为 MQuAKE‑CF‑3k‑v2（对抗性多跳问答）和 MQuAKE‑T（时序编辑问答），分别包含 3000 条与 1868 条编辑实例。

**📈 对比分析**

与现有的参数保持编辑器（MeLLo、PokeMQA、KeDKG）以及参数调优编辑器（FT、ROME、MEMIT）对比，HyperPatch 在 H‑Acc 上分别实现 96.24% 与 21.06% 的相对提升，并在所有编辑场景下保持 25.9× 的检索速度提升，显著优于 KG‑based 传统方法。

**⚠️ 局限性**

局限性：① 依赖可构建的超图结构，适用场景受限于 n‑ary 事件；② 超图构建与维护成本较高；③ 在非 n‑ary 或多语言多模态场景下的通用性尚未验证；④ 对极端频繁更新的实时性与内存占用仍需进一步优化。

---

## 248. ConTrack: Constrained Hand Motion Tracking with Adaptive Trade-off Control

**arXiv ID:** 2606.03177 | [PDF](https://arxiv.org/pdf/2606.03177v1)

**作者:** Yutong Liang `[一作]` (University of California San Diego), Xiaolong Wang `[通讯]` (University of California San Diego)

**通讯引用:** 21311 | [OpenAlex ID](https://openalex.org/A5100424261)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对人类演示进行约束式手部运动跟踪，利用强化学习在机器人上实现长时段、接触丰富的物体追踪，同时保持手部关节运动和接触时序。

**💡 创新点**

创新点在于将对象跟踪视为约束，使用在线双变量控制器动态平衡任务成功与风格保真；引入自适应中途重置库以稳定长时段学习，并利用接触先验提升学习效果。

**🔧 技术方法**

采用PPO强化学习，Lagrangian约束与双变量更新，在线双变量控制器，自动化中途重置策略，以及接触先验奖励等技术。

**📊 数据集**

使用重定向到机器人本体的人类手部动作数据，并在GRAB、ARCTIC、DexterHand三大基准数据集上进行训练与评估。

**📈 对比分析**

与ManipTrans、DexMachina、SPIDER等基线对比，ConTrack在同等训练预算下显著提高成功率、进度和接触准确率，并通过消融实验验证各模块贡献；在真实xArm7+xHand平台上亦能实现可执行轨迹。

**⚠️ 局限性**

局限包括：约束采用在线归一化，无法严格保证约束满足；接触先验依赖高质量标注；在最难的单手旋转任务下仍受限于训练预算；sim-to-real对齐尚未充分利用。

---

## 249. Ask When It Pays: Cost-Aware Open-Ended Interaction for Instance Goal Navigation

**arXiv ID:** 2606.03175 | [PDF](https://arxiv.org/pdf/2606.03175v1)

**作者:** Xunyi Zhao `[一作]` (Adelaide University), Qi Wu `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将交互式实例目标导航视为成本敏感的不确定性降低问题，构建了带有类型化询问和加权成功率评估的交互基准，并提出零样本多模态大语言模型导航器TANDEM，能够在必要时进行成本感知询问；

**💡 创新点**

创新点包括：①基于信息增益的问答类型分层及对应成本的自洽设计；②引入加权成功率（Weighted SR）评价指标，将询问成本纳入成功判定；③构建可控难度、已标注oracle答案的实例级导航基准；④在上述框架下提出解耦式的规划-地面-执行三阶段零样本导航架构；

**🔧 技术方法**

采用信息增益分析挖掘问答有用性，使用Qwen3.5-4B等多模态大语言模型作为规划器，结合Spatial Memory和Grounder‑Executor模块实现语义决策转化为精确运动；同时利用Isaac Sim进行场景模拟和数据生成；

**📊 数据集**

利用多源人类导航语料（R2R、REVERIE、RxR、CVDN、SOON 等）挖掘问答类型，构建约 22,905 条 3D 场景的 IGN 基准，划分易/中/难集；此外在 CVDN、SOON、REVERIE 上进行跨任务评估；

**📈 对比分析**

在统一实验设置下与 GTA、MapGPT、NavGPT、COIN 等零样本基线对比，使用 SR@1.5、Weighted SR、导航误差等指标；TANDEM 在 500 条测试集上取得 35.3% 的 SR@1.5 与 21.4% 的 Weighted SR，尤其在困难样本上显著优于其他方法；

**⚠️ 局限性**

局限性包括：①基准与实验均依赖模拟环境，缺乏真实世界验证；②oracle 提供的答案假设完美且不考虑错误传播；③成本模型基于语料统计，可能无法普适；④仅设定四种问答类型，未覆盖更细粒度交互；⑤零样本 MLLM 对模型大小和提示设计敏感，性能易受限。

---

## 250. JAVEDIT: Joint Audio-Visual Instruction-Guided Video Editing with Agentic Data Curation

**arXiv ID:** 2606.03168 | [PDF](https://arxiv.org/pdf/2606.03168v1)

**作者:** Yinan Chen `[一作]` (Zhejiang University), Shuicheng Yan `[通讯]` (National University Of Singapore)

**通讯引用:** 93166 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了首个100K规模的指令驱动联合音视频编辑数据集和基准，设计了自动化 Agent‑in‑the‑loop 质量控制流程，并在此基础上训练了基线模型，实现了跨模态编辑。

**💡 创新点**

创新点包括：①首个面向指令的联合音视频编辑大规模数据集；②基于 Agent‑in‑the‑loop 的全自动数据筛选与修复机制；③提出多维度评估基准与统一的编辑任务定义；④将 LTX‑2.3 通过 LoRA 适配为联合编辑模型。

**🔧 技术方法**

技术方法涵盖：多模态生成管线（HunyuanImage‑3.0, Wan2.2‑Animate, DreamVoice, HunyuanVideo‑Foley, MiniMax‑Remover, SAM‑Audio, Qwen3‑Omni 等）；Agent‑in‑the‑loop（Claude Opus 4.6 + Gemini 3.1 Pro）进行错误检测与修复；模型训练采用 LTX‑2.3 + LoRA 的参数高效微调；评估使用 VTSS、UTMOSv2、SyncNet、Qwen3‑Omni 的多维指标。

**📊 数据集**

使用 OpenHumanVid、VIDGEN‑1M、VGGSound 等公开视频库进行数据采集，并构建 103K 条编辑三元组；基准集包含 150 条人工挑选的源视频与对应指令。

**📈 对比分析**

通过与 AVED、AVI‑Edit、以及 Kiwi‑Edit + HunyuanVideo‑Foley 的序贯级联模型对比，采用 6 维度评估（视觉质量、音频质量、音视频同步、指令遵循、视频保真、整体质量）进行比较。模型在 5/6 维度上取得首位，音视频同步提升 26% 以上。

**⚠️ 局限性**

局限性在于：对高度复杂、多重同时编辑任务的表现仍有限；数据集主要覆盖人类中心场景，对开放域音视频场景适用性不足；模型性能受限于现有基础模型的能力，Agent‑in‑the‑loop 在极端错误场景下的修复仍不完备。

---

## 251. SketchSong: Hierarchical Song Generation with Sketch Planning and Fine-Grained Multi-Track Modeling

**arXiv ID:** 2606.03169 | [PDF](https://arxiv.org/pdf/2606.03169v1)

**作者:** Xiaoyue Duan `[一作]` (Tencent Inc.), Jie Zhou `[通讯]` (Tencent Inc.)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个层次化的歌曲生成框架 SketchSong，先通过歌曲级草图规划给模型提供整体安排蓝图，然后在第二阶段对声乐、低音、鼓、其他四轨进行细粒度建模，生成完整且结构清晰的歌曲。

**💡 创新点**

创新点包括：① 在一阶段先预测压缩的草图令模型拥有全局安排计划，解决传统一阶段同时处理全局结构和细节的困难；② 在第二阶段实现四轨细粒度建模，精确捕捉各乐器角色与相互作用，提升编曲丰富度；③ 将两项技术结合为统一的两阶段自回归框架，实现多维度的协同优化。

**🔧 技术方法**

技术细节：使用两阶段 Transformer 自回归模型；草图基于 MuQ-MuLan 特征 + 4 层残差向量量化（RVQ）生成；草图与音频 token 采用延迟模式嵌入；第二阶段在第一阶段隐藏状态的指导下预测四轨 token；解码采用 LeVo 的 MuCodec；整体训练采用两相位阶段（先草图再音频）和冻结第一阶段隐藏状态的条件化。

**📊 数据集**

数据集：1 百万首歌曲（约 54k 小时）；使用 SongPrep 生成结构化歌词、章节结构、四轨分离（Demucs）；结构注释通过 All‑In‑One + DPRNN；文本描述用 Qwen2.5‑Omni 生成；训练数据涵盖中英文歌曲。

**📈 对比分析**

对比方法：自训练 LeVo 基线、公开模型 YuE、DiffRhythm、ACE‑Step；评价指标包括 FAD、PER、MuQ‑T、Aesthetic（CE/CU/PC/PQ）以及 SongEval（Coherence、Musicality、Memorability、Clarity、Naturalness）和 MOS 听觉测试。SketchSong 在 FAD、Aesthetic、MOS 方面均优于 LeVo 基线，且在无后期训练的条件下与公开模型竞争，证明两项创新在提升全局连贯性和细粒度编曲质量方面均有效。

**⚠️ 局限性**

局限性：① 未采用后期优化（如 DPO、文本/歌词对齐）导致在某些细节指标上仍落后；② 低音、鼓、其他轨道使用共享伴奏 codec 进行解码，混音后自然度不足；③ 仅对第二阶段做四轨建模，第一阶段仍为混音，进一步的多轨 codec 及更强波形解码是未来改进方向。

---

## 252. Pulse Focus: Validation of the Focus Performance Score as a Behavioral Signal for Human Attentional State Modeling Toward Attention-Aware AI

**arXiv ID:** 2606.03164 | [PDF](https://arxiv.org/pdf/2606.03164v1)

**作者:** Yisak Debele `[一作]` (Synheart AI), Anwar Misbah `[通讯]` (Synheart AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出并验证了一种名为 Focus Performance Score（FPS）的移动 Stroop 任务评分，用作人类注意力状态的行为真实值，支持 AI 系统实时感知与学习注意力。

**💡 创新点**

创新点在于：①将 FPS 设计为双层结构（Base×Auth）并引入二元参与门；②在三个层面（行为、神经、公式）进行严格验证；③使用四级证据（理论权重、内部校准、外部神经指标、CFA）证明权重的科学性；④构建了可用于 AI 训练和实时交互的可解释、可靠的注意力度量。

**🔧 技术方法**

采用的技术包括：手机游戏化 Stroop 任务的毫秒级行为采集；fMRI（DMCC55B）数据的 GLM 与 ACC 相关性分析；统计分析（效应量、ICC、相关系数、卡方检验）；Confirmatory Factor Analysis（CFA）与结构方差分析；网格搜索校准权重；以及正交化的权重验证流程。

**📊 数据集**

主要数据集：①Białaszek 等 2022 年的 466 名健康成人，111,133 条 Stroop 试次，用于行为验证与权重内部校准；②DMCC55B 55 名健康成人的 fMRI 版 Stroop 任务，用于神经验证与外部权重校准；此外还引用公开 Stroop 任务数据库与论文报道的效应量作对照。

**📈 对比分析**

对比方法：将 FPS 与传统 Stroop 反应时差、准确率差、个体差异相关、ICC、与 ACC 激活的相关性等指标进行对照。性能方面：Stroop 反应时效应 d = 1.339（p < 10⁻¹⁰⁰），准确率效应 d = 0.447；ICC 在不同情境下 0.83–0.93；FPS 与平均不一致反应时的相关 r = 0.785；平均不一致反应时与 ACC 激活的相关 r = –0.327（p = 0.015）。

**⚠️ 局限性**

局限性包括：①捕获试次（n7）在公开数据中未被验证；②权重仍为预估值，需在触摸屏 Pulse Focus 真实数据中重新校准；③综合 FPS 与 ACC 的相关性仅为趋势性（p = 0.109）；④受试者为健康成人且准确率接近 100%，导致部分变量变异不足；⑤使用的 fMRI 任务与 Pulse Focus 的交互方式（语音/键盘 vs 触摸屏）不同；⑥尚未验证 FPS 与 HRV/PPG 等生理信号的共变。

---

## 253. PsychoPass: Geometric Profiling of Multi-Turn Adversarial LLM Conversations

**arXiv ID:** 2606.03136 | [PDF](https://arxiv.org/pdf/2606.03136v1)

**作者:** Muberra Ozmen `[一作]` (Coveo), Subhabrata Majumdar `[通讯]` (Indian Institute of Management Bangalore)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究多轮 jailbreak 攻击，构建对话轨迹的几何模型，提取轨迹几何特征并训练早期警报分类器（PsychoPass），实现对攻击意图的前置检测。

**💡 创新点**

创新点在于：①从对话内容转向轨迹几何，发现攻击在转移到“禁区”时会留下可观测的形状痕迹；②拆解并消除长度混杂，证明几何信号的前置性与编码器不敏感；③给出理论分析解释长度、形状、编码器三方面的效应；④提出可在线监测的轻量级框架。

**🔧 技术方法**

使用的技术包括：①利用 TF‑IDF 及 dense embedding 对对话进行投影；②提取路径长度、漂移、直接度、速度、圆度等 L2‑范数统计；③使用 Catch22 的时间序列特征（如时间反演不对称、异常时序、伸展递减、低频波动等）；④用逻辑回归与梯度提升树做二分类；⑤进行前缀截断实验和理论推导。

**📊 数据集**

数据集为 7,525 条 Crescendo 多轮攻击样本，攻击目标为四个 LLM（Llama、GPT 等），攻击 seed 来自 AdvBench、HarmBench、AIRT；每条对话最大 8 轮、2 回溯。

**📈 对比分析**

与单轮内容安全器 Llama Guard 进行对比，PsychoPass 在前 2–4 轮时 AUROC ≈0.65，优于内容过滤器在同一前缀下仅 0.49；实验 1–3 进一步证明：去除长度后 AUROC 仍保持 0.65–0.70，理论分析解释了 95% 由长度贡献。

**⚠️ 局限性**

局限性：仅评估 Crescendo 攻击；长度上限为 8 轮，可能改变信号分布；攻击成功标签由单一 LLM 判定，阈值敏感性未探究；只比较稀疏 vs 密集编码器，未检验两种密集编码器间的一致性。

---

## 254. Rethinking Neural Width for Alternating Current Optimal Power Flow Proxies

**arXiv ID:** 2606.03125 | [PDF](https://arxiv.org/pdf/2606.03125v1)

**作者:** Dhruvi Khandelwal `[一作]` (National Institute of Technology Kurukshetra), Parikshit Pareek `[通讯]` (Indian Institute of Technology Roorkee)

**通讯引用:** 302 | [OpenAlex ID](https://openalex.org/A5055795558)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种Loss‑Guided Neural Densification（LG‑ND）方法，动态增宽神经网络以逼近交流最优潮流（ACOPF）解，避免过度参数化；

**💡 创新点**

创新点在于将网络宽度视为可动态调整的结构变量，通过验证损失决定何时扩展隐藏层，实现“极简”代理；

**🔧 技术方法**

使用全连接多层感知机（MLP）、均方误差（MSE）监督学习、损失引导增宽策略、以及坐标裁剪后处理；

**📊 数据集**

在IEEE 57、118、57等标准电力系统案例上生成ACOPF标签数据；

**📈 对比分析**

与传统宽度较大的基线（如Naïve MSE、MAE、Penalty‑MSE/MAE）相比，LG‑ND在相同或更小的参数规模下获得更低的Optimality Gap（0.03%–0.07%）和更小的约束残差，推理复杂度降低约15×；

**⚠️ 局限性**

局限性包括对训练样本量的依赖（样本不足时约束残差仍显著）以及在网络拓扑变化或更大规模系统上的泛化性待验证。

---

## 255. KC-3DGS: Kurtosis-Constrained Gaussian Splatting for High-Fidelity View Synthesis

**arXiv ID:** 2606.03120 | [PDF](https://arxiv.org/pdf/2606.03120v1)

**作者:** Vivekjyoti Banerjee `[一作]` (Johns Hopkins University), Aniket Roy `[通讯]` (NEC Labs America)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在3D Gaussian Splatting（3DGS）的训练中加入了波形域峰度约束和跨频谱协方差惩罚，实现更高保真、结构更一致的视点合成。

**💡 创新点**

创新点在于将多尺度离散小波变换的峰度浓度、细尺度权重和协方差正则化与像素空间损失联合，用轻量化可插拔的正则化方式弥补传统像素损失对频率分布约束不足。

**🔧 技术方法**

使用技术包括3DGS、Daubechies‑3小波分解、L1/SSIM像素损失、峰度损失、跨频谱协方差惩罚以及可微分渲染。

**📊 数据集**

实验数据集涵盖稀疏视点场景（MipNeRF360、Tanks&Temples、MVImgNet、DeepBlending）以及极端稀疏视点挑战集WRIVA‑ULTRRA，并在完整视点上测试。

**📈 对比分析**

通过与Splatfacto、FasterGS、OctreeGS等基线在PSNR/SSIM/LPIPS/DreamSim等指标比较，稀疏视点下平均PSNR提升约0.5 dB、DreamSim下降9.48%，在多场景中显著提升感知质量。

**⚠️ 局限性**

局限性包括在密集视点设置下可能出现过度正则化导致PSNR略降，计算开销略增，对极端稀疏视点的鲁棒性仍有限。

---

## 256. GuidedBridge: Training-freely Improving Bridge Models with Prior Guidance

**arXiv ID:** 2606.03119 | [PDF](https://arxiv.org/pdf/2606.03119v1)

**作者:** Zehua Chen `[一作]` (Tsinghua University), Jun Zhu `[通讯]` (Tsinghua University)

**通讯引用:** 473685 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练无关的桥模型引导方法Prior Guidance (PG)及其频率调制版FMPG，并与CFG结合形成CFG‑FMPG，用于提升基于桥模型的图像生成与修复质量。

**💡 创新点**

创新点在于：①通过引入弱先验（如加噪）制造生成质量差的解，进而对比并放大优质解，强调桥模型的先验利用；②针对桥模型的U形信噪比特性，设计按频段调节的指导尺度；③在先验信息不足时，采用CFG与FMPG的级联，兼顾语义对齐与细节重建。

**🔧 技术方法**

使用技术包括：桥模型（DDBM、DBIM）训练后直接进行采样；PG和FMPG通过对原先验和弱先验的去噪预测进行加权；CFG引导式分步采样；FFT或低通/高通滤波器实现频段分离；梯度优化仅用于CFG超参搜索，PG/FMPG完全无训练。

**📊 数据集**

数据集：Edges→Handbags、DIODE（户外场景恢复）用于图像翻译与修复；ImageNet（256×256）用于大规模条件生成与图像修复；此外在实验中还使用了标准基准模型的官方预训练权重。

**📈 对比分析**

与现有方法（DDIB、SDEdit、Pix2Pix、I2SB、ECSI、DBIM、DDBM）在FID、IS、LPIPS、MSE等指标上进行对比。PG/FMPG在相同NFE下显著降低FID（如Edges→Handbags从1.83↓到1.07，DIODE从3.73↓到3.20），并在高速采样下保持高质量；CFG‑FMPG在ImageNet 10 NFE即可达到FID 3.86，远快于传统方法。

**⚠️ 局限性**

局限性包括：①当给定先验信息本身不具备强指导作用时，PG的弱先验难以产生显著差异，提升有限；②需要针对不同任务手工选择降解方式和引导尺度，仍有超参数调节；③目前仅在图像域验证，尚未验证至语音、视频等其他数据类型；④在极端噪声或高分辨率任务下的可扩展性需进一步评估。

---

## 257. OpenAgenet/OAN: Open Infrastructure for Trusted Agent Interconnection

**arXiv ID:** 2606.03161 | [PDF](https://arxiv.org/pdf/2606.03161v1)

**作者:** Jinliang Xu `[一作]` (China Academy of Information and Communications Technology), Jinliang Xu `[通讯]` (China Academy of Information and Communications Technology)

**通讯引用:** 1176 | [OpenAlex ID](https://openalex.org/A5042784076)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 OpenAgenet（OAN）——一种面向代理（Agent）互联的可信基础设施层，提供根治理身份录入、注册人（Registrar）入驻、根签名包发布、授权发现和签名可信调用的完整生命周期。

**💡 创新点**

创新点在于：①把身份治理与发现、调用拆分为可独立运作的角色（Root、Registrar、Discovery、CDN、Service Agent、User Agent）；②利用区块链背书的公告板记录授权变更，保证治理决策的可审计性；③引入能力域（Capability Domain）实现授权驱动的发现；④在已有的 MCP、A2A、ANP 等协议之上提供统一的信任路径，而不替代这些协议。

**🔧 技术方法**

技术手段包括：DID/VC 风格的身份和凭证、数字签名、哈希、随机数（nonce）、时间戳、区块链账本作为公告板、REST/JSON 接口、Python/TypeScript SDK、分布式 CDN、日志与监控系统。

**📊 数据集**

使用了自行生成的 Agent 身份表（10-2000 条）进行生命周期演练，未使用公开的大规模真实数据集，而是通过模拟身份数据来评估吞吐量和可扩展性。

**📈 对比分析**

评估方法：①生命周期完整性验证、②错误/无效输入拒绝测试、③授权发现完整性检验、④规模性测试（单机 10-2000 条、分布式 1200 条）。性能表现：单机吞吐量可达 2000 条身份，发现查询完整返回；瓶颈主要集中在大结果分页、发布/同步批处理和高负载下的队列调度，未出现功能缺陷。

**⚠️ 局限性**

局限性：仅支持单根治理域，跨域联邦尚未实现；不验证代理的业务能力、准确性或安全性；隐私保护（私密查询、选择性披露）未实现；生产环境仍需加强数据库后端、密钥管理、速率限制、灾备等硬化措施。

---

## 258. Uncertainty-Aware Clarification in LLM Agents with Information Gain

**arXiv ID:** 2606.03135 | [PDF](https://arxiv.org/pdf/2606.03135v1)

**作者:** Mengyi Deng `[一作]` (HKUST(GZ)), Wei Wang `[通讯]` (HKUST(GZ))

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为了解决大语言模型代理在执行工具调用时因用户指令模糊导致的错误，本文提出了一种基于信息增益的澄清框架，在交互过程中学习何时以及如何提问，以最大化对隐藏用户意图的不确定性降低。

**💡 创新点**

创新点在于：①将澄清问题视作贝叶斯实验设计，使用信息增益（Expected Information Gain）作为奖励信号；②利用教师强制的对数似然差值近似信息增益，实现对“目标意图”概率分布的信念更新；③在此奖励下采用 DAPO（Decoupled Advantage Policy Optimization）训练澄清策略；④设计了严格用户模拟器保证奖励仅对真正有价值的澄清有效；⑤实验显示即使是小模型的澄清器，也能在多种大模型代理上匹配甚至超过大模型的澄清效果。

**🔧 技术方法**

主要技术：贝叶斯实验设计、信息增益奖励、教师强制对数似然、点信息增益（PMI）近似、DAPO 强化学习、τ‑Bench 环境、严格用户模拟器、对话生成与工具调用交互。

**📊 数据集**

使用数据集：τ‑Bench（τ‑Retail）任务数据；训练集 500 条轨迹；测试集 115 条在域任务与 50 条 OOD 任务，全部包含用户模糊指令与工具执行反馈。

**📈 对比分析**

与无澄清基线、预训练 Qwen3‑1.7B 澄清器以及不同大模型（DeepSeek‑V3.1、DeepSeek‑R1 等）进行对比；结果表明：平均成功率提升 3.7%，澄清次数下降至 1.3 次/任务，仅增加 0.3 个总交互步；在多种代理上跨域泛化良好，甚至在 OOD 任务上提升 5.4%。

**⚠️ 局限性**

局限性：奖励信号需要已知的真实用户目标，训练中使用的严格模拟器与真实用户交互差异较大；仅在离线仿真环境评估，未考虑真实用户噪声和多轮交互的不确定性；未对代理与澄清器进行联合训练，可能限制协同优化的潜力。

---

## 259. DMT-CBT: Longitudinal Therapeutic State Modeling for CBT Counseling

**arXiv ID:** 2606.03132 | [PDF](https://arxiv.org/pdf/2606.03132v1)

**作者:** Chang Liu `[一作]` (Lanzhou University), Bin Hu `[通讯]` (Lanzhou University)

**通讯引用:** 25027 | [OpenAlex ID](https://openalex.org/A5100380066)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出DMT-CBT框架并构建DMTCorpus多会话多模态CBT数据集，用结构化治疗状态、跨会话记忆、行为多模态与工具辅助干预实现纵向CBT模拟与评估。

**💡 创新点**

将CBT从局部回复生成转为治疗状态建模；引入结构化状态追踪、跨会话记忆、行为不一致多模态观测和工具检索，实现更真实、连续的治疗流程。

**🔧 技术方法**

使用大语言模型（GPT‑4.1‑mini、Qwen2.5‑VL‑7B、Qwen2.5‑7B）、LoRA微调、BGE‑large‑zh‑v1.5检索、面部表情库、结构化状态更新函数、工具库检索与选择等技术。

**📊 数据集**

核心数据集为自研DMTCorpus（文本+图像+工具，4,317有效会话，6会话/案例），并对比PsychEval、CBT‑LLM、M2CoSC、CS‑LLaVA等现有基准。

**📈 对比分析**

对比单会话文本、多模态、跨会话文本三类基线；通过CTRS、WAI、PANAS、工具选择准确率、记忆一致性等指标评估；DMT‑CBT在会话质量、治疗联盟、情绪轨迹和状态一致性上均优于所有基线。

**⚠️ 局限性**

仅在仿真环境中验证，缺乏真实临床验证；数据生成与评估均基于LLM，可能产生循环偏差；多模态仅使用静态面部图像，未涵盖姿势、语音等；中文语境限制跨文化泛化。

---

## 260. HARVE: Hacking-Aware Reward-Head Vector Editing for Robust Reward Models

**arXiv ID:** 2606.03131 | [PDF](https://arxiv.org/pdf/2606.03131v1)

**作者:** Shuang Liu `[一作]` (Carnegie Mellon University), Mengnan Du `[通讯]` (Chinese University of Hong Kong, Shenzhen)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个新的奖励模型黑客基准（HARVE）和一种训练无关的奖励头编辑方法（HARVE-Reward-Head-Editing），用于提升大型语言模型奖励模型的鲁棒性。

**💡 创新点**

创新点在于：①构建了覆盖专业领域（法律、政策、合规）与通用攻击的1,203对金牌-黑客样本；②提出只通过编辑奖励头向量去除黑客子空间的训练自由方法；③通过机制分析证明奖励黑客是多维残差子空间导致的。

**🔧 技术方法**

技术主要包括：利用奖励模型的线性奖励头提取被黑示例的残差方向，使用SVD得到多方向子空间，随后对奖励头向量做投影消除操作，整个过程不需要梯度更新。

**📊 数据集**

数据集：HARVE基准（1,203金牌-黑客对，13个子类别，含专业法律、政策、合规等），以及RM-Bench硬难度子集用于跨域评估。

**📈 对比分析**

对比方法：原始奖励模型、数据增强微调（3:1、5:1比例）。结果显示：在目标子类别平均提升21.1个百分点的金牌偏好率，超过微调基线13.7个百分点；在非目标子类别和RM-Bench上保持甚至提升性能。

**⚠️ 局限性**

局限性：仅适用于可访问奖励头向量的标量奖励模型；需要少量金牌-黑客对来估计子空间；无法覆盖所有高风险实际场景；未评估编辑后模型在完整RLHF或推理优化管道中的下游效果。

---

## 261. Decoupled Smart Contract Audits: Lightweight LLM Framework via Distillation and Aggregation

**arXiv ID:** 2606.03128 | [PDF](https://arxiv.org/pdf/2606.03128v1)

**作者:** Bagus Rakadyanto Oktavianto Putra `[一作]` (Universitas Gadjah Mada), Guntur Dharma Putra `[通讯]` (Universitas Gadjah Mada)

**通讯引用:** 362 | [OpenAlex ID](https://openalex.org/A5007007487)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并实现了一个将智能合约审计流程拆分为检测、解释、严重性评估和修复推荐四个专用模块的轻量级LLM框架。

**💡 创新点**

创新点包括：将审计工作分解为独立模块、引入Rank‑Stabilized Low‑Rank Adapters (rsLoRA)、知识蒸馏以及改进的Chain‑of‑Verification (CoVe)聚合机制。

**🔧 技术方法**

采用轻量化开源LLM（0.6B–4B参数）、rsLoRA、教师模型Qwen3‑30B的知识蒸馏以及自定义CoVe聚合算法。

**📊 数据集**

使用由Code4rena和Shieldify收集的包含218低/549中/232高危漏洞及其描述、严重性和补丁的生成数据集，和由930个漏洞与969个无漏洞代码样本构成的检测数据集。

**📈 对比分析**

与7B–34B稠密模型在零样本、微调和统一提示下对比，轻量框架在检测任务上实现98.25%准确率，解释与推荐的LMUnit对齐分数分别为0.4375和0.763，显著优于大模型。

**⚠️ 局限性**

局限性包括统一模型在多任务时易出现逻辑失真，严重性评估仍受“Severity Centrality Bias”影响，且实验主要针对Solidity合约，缺乏跨语言和更大规模人类评估。

---

## 262. TTT-VLA: Test-Time Latent Prompt Optimization for Vision-Language-Action Models

**arXiv ID:** 2606.03127 | [PDF](https://arxiv.org/pdf/2606.03127v1)

**作者:** Wenbo Zhang `[一作]` (ByteDance Seed), Xiao Ma `[通讯]` (ByteDance Seed)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `afceb026-1760-41ae-8d86-010831a37d97` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种在部署时仅更新学习的潜在提示（latent prompt）以提升 Vision‑Language‑Action（VLA）模型性能的框架 TTT‑VLA。

**💡 创新点**

创新点在于将潜在提示视为可学习的控制上下文：在训练期间与代理任务（state‑grounding）共同优化，并在测试时仅更新此提示实现自适应，而不修改主模型参数。

**🔧 技术方法**

技术包括：基于流匹配的 VLA 架构、Mixture‑of‑Transformers (MoT) 设计、状态对齐作为自监督代理任务、训练时的随机 drop 与梯度分离策略，以及离线 prompt‑only test‑time 训练。

**📊 数据集**

使用的数据集：SimplerEnv 基准（WidowX、Google Robot 任务），训练时使用 Bridge V2 的 OXE‑Aug 多体数据集。

**📈 对比分析**

评估方法：与公开 SOTA（RT‑1‑X、OpenVLA、SpatialVLA 等）以及内部 π_0.5 基线进行对比。单体场景下，TTT‑VLA 的平均成功率从 51.1% 提升至 67.4%（约 15% 点）；多体场景下提升约 9%（从 22.8% 提升至 31.6%）。在 Google Robot 任务中，平均提升 3–5 个百分点。

**⚠️ 局限性**

局限性：当前代理任务主要纠正局部关键决策点，难以实现长周期全局行为改进；在线 prompt 更新不稳定；需要更强的代理任务和更大规模的 VLA 基础模型来进一步突破。

---

## 263. Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning

**arXiv ID:** 2606.03113 | [PDF](https://arxiv.org/pdf/2606.03113v1)

**作者:** Yanyu Zhu `[一作]` (Tsinghua Shenzhen International Graduate School), Hai-Tao Zheng `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于离线强化学习的自适应动态退出框架LEDE，用于加速大型语言模型的自回归推理。

**💡 创新点**

创新点在于将动态选择退出层和推测长度视为马尔可夫决策过程，并通过离线强化学习学习上下文感知的决策策略。

**🔧 技术方法**

使用的技术包括离线强化学习中的深度Q网络（DQN）、经验回放、噪声线性层和层内动态剪枝。

**📊 数据集**

实验数据集涵盖LLaMA-2/3、CodeLLaMA等模型在指令跟随、语言建模、摘要、代码生成等任务。

**📈 对比分析**

与基线（自回归、LayerSkip、LITE、Draft & Verify）比较，LEDE在速度上获得了平均2.32×（最高2.7×）的提升，同时保持较高的接受率。

**⚠️ 局限性**

局限性包括目前仅在7B-34B规模模型验证，缺乏对更大模型的适配；离线RL训练需要额外的经验采集；以及对特殊任务的泛化能力待进一步评估。

---

## 264. Auditing Engagement Incentives in the Kidfluencer Ecosystem: A Multimodal Weak Supervision Approach

**arXiv ID:** 2606.03173 | [PDF](https://arxiv.org/pdf/2606.03173v1)

**作者:** Zijing Wei `[一作]` (Independent Contributor), Xuanjie Chen `[通讯]` (Independent Contributor)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用多模态弱监督框架，对5,051条YouTube kidfluencer视频进行剥削风险评估，并将剥削得分与观看量关联分析，揭示“表演性溢价”，即表演劳动和情感诱饵的视频更易获得高观看量。

**💡 创新点**

首次将LLM与VLM的零样本分类器与传统规则结合到Snorkel弱监督框架，实现连续剥削风险评分；同时发现商业内容与观看量无正向关联，挑战了传统广告驱动模型。

**🔧 技术方法**

Snorkel弱监督框架、GPT-4文本与Vision‑LM分类器、关键词与正则规则、混合效应线性回归、FDR校正、Spearman相关等。

**📊 数据集**

79个英语kidfluencer频道的5,051条视频（4,208条有效观看量），来自YouTube Data API的标题、描述、缩略图和发布日期。

**📈 对比分析**

与107条人工标注样本比较，宏观F1为0.911，整体F1为0.793；混合效应模型显示剥削分数每升高1点，观看量增幅约4.4倍；情感诱饵、表演劳动等维度的增量回归系数显著，商业内容不显著。

**⚠️ 局限性**

仅基于元数据与缩略图，未检测完整视频内容；跨文化与非英语样本缺失；观察性数据无法证明因果；剥削风险是代理指标，可能与实际剥削程度不完全一致。

---

## 265. Fully Automated Identification of Lexical Alignment and Preference-Stage Shifts in Large Language Models

**arXiv ID:** 2606.03165 | [PDF](https://arxiv.org/pdf/2606.03165v1)

**作者:** Thomas Stephan Juzek `[一作]`, Jose A. Hernandez `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究提出了两种无人工筛选、无假设的词汇对齐评估指标——Lexical Alignment Score（LAS）和Triangulated Preference Shift（TPS），用于量化大语言模型生成文本中词汇使用的偏差及其与偏好学习阶段的关系。

**💡 创新点**

创新点在于提出了一套完全无手工筛选、无假设的词汇对齐评估流程，并通过窗口化文档频率与贝叶斯平滑来稳定估计，首次将偏好学习阶段与词汇过度使用关联起来。

**🔧 技术方法**

技术上采用了窗口化文档频率估计、Jeffreys平滑、UPOS词性标注与词干化、root‑mean‑square 归一化打分，以及与基线人类续写对齐的三方比较（基准、基模型、指令模型）。

**📊 数据集**

数据集为 PubMed 2012‑2021 年的 42,000 条医学摘要的前后两半作为人类基线，生成对应的模型续写，共计 6 家模型族（Falcon、Gemma、Llama‑3.1、Mistral、OLMo、Yi‑1.5）。

**📈 对比分析**

通过对比 LAS 与 TPS 的宏观与微观指标，发现指令模型整体与人类更接近，但在内容词上常常出现反向或弱化；TPS 显示偏好学习导致大约 10‑20% 的额外词汇过度使用，验证了指标在不同窗口、种子与数据量下的稳健性。

**⚠️ 局限性**

局限性包括仅使用科学英语摘要的后半段作为人类基线，导致位置偏差；限制在单一领域；未覆盖最流行的 ChatGPT；并且可能存在训练数据与评估集重叠的问题。

---

## 266. GTBench: A Curriculum-Grounded Benchmark for Evaluating LLMs as Mathematical Research Assistants in Graph Theory

**arXiv ID:** 2606.03144 | [PDF](https://arxiv.org/pdf/2606.03144v1)

**作者:** Noujoud Nader `[一作]` (Louisiana State University), Deepti Gupta `[通讯]` (Texas A&M University-Central Texas)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了一个基于课程难度的图论 benchmark，包含 63 个从基础到研究级别的题目，并在零样本与链式推理提示下评估了五种前沿 LLM 的解答质量。

**💡 创新点**

创新点在于首次将图论知识系统化为层级评测框架，并采用混合评判（LLM‑as‑Judge 与人类专家）对高难度证明题进行细粒度错误分类。

**🔧 技术方法**

技术手段包括零样本与 Chain‑of‑Thought 提示、GPT‑4o 作为自动判题器、Python API 调用、以及自定义失败模式标签（A–D）来细化错误来源。

**📊 数据集**

数据集来源于 Diestel 《Graph Theory》与西班牙巴塞罗那技术大学的 UPC 课程练习题，经过筛选后得到 63 个验证过的题目。

**📈 对比分析**

通过准确率和失败模式统计对比模型性能：GPT‑5 最高（组1 95.8%/组3 82%），其余模型随难度递减；CoT 提示对基础题提升有限，对证明题甚至导致性能下降。

**⚠️ 局限性**

局限性包括仅覆盖图论领域、题目规模有限、证明题仍需人工评判、自动评判对长篇论证的鲁棒性不足、以及评分尺度较粗，难以精准捕捉细微差距。

---

## 267. Evidence-Aware Protein Complex Detection: Methods, Benchmarks, and Reproducibility Challenges

**arXiv ID:** 2606.03178 | [PDF](https://arxiv.org/pdf/2606.03178v1)

**作者:** Sima Soltani `[一作]` (Islamic Azad University), Reza Sheybani `[通讯]` (Islamic Azad University)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e15e3743-5ee0-4d5f-813d-d146868082fc` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对 2018 年以后、以往经典算法为参考，综述了将蛋白质相互作用网络（PPI）拓扑与 Gene Ontology、表达、定位、域、时序等多源生物证据相结合的蛋白复合物检测方法。

**💡 创新点**

创新点在于提出了系统化的整合策略分类、可复现评估清单以及统一 benchmark 规范的紧迫性，明确指出当前方法的可比性局限与改进方向。

**🔧 技术方法**

主要技术包括图聚类（MCL、MCODE 等）、基于 GO 的边权重与目标函数、表达/定位过滤、超图与时序异质网络表示学习、监督/嵌入式分类器以及基于演化/元启发式的优化框架。

**📊 数据集**

参考数据集涵盖 DIP、HPRD、STRING、MIPS、CYC2008、CORUM、PCDq 等公开 PPI 与复合物数据库，未构建新数据集；评价使用的指标有精确率、召回率、F1、MMR、Rand/NMI 等。

**📈 对比分析**

对比采用源特定的 PPI 版本、参考集、预处理与阈值，报告的 F1 约 0.80–0.90，但因评估协议差异不具可直接排名；透明的证据驱动图方法在生物合理性与可复现性上表现最佳，深度/超图/动态模型则需更严格的共享 benchmark 控制。

**⚠️ 局限性**

主要局限包括缺乏统一、可复现的 benchmark 包；GO 关联的循环性与注释偏倚；人类数据与细胞特异性验证不足；重叠检测指标不统一；以及大多数方法的实现与评估难以跨平台复现。

---

## 268. SRENet: Spectral Re-Entry Network for Point Cloud Action Recognition

**arXiv ID:** 2606.03160 | [PDF](https://arxiv.org/pdf/2606.03160v1)

**作者:** Qiuxia Wu `[一作]` (South China University of Technology), Kun Hu `[通讯]` (Edith Cowan University)

**通讯引用:** 13100 | [OpenAlex ID](https://openalex.org/A5028673475)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种Spectral Re‑Entry Network（SRENet），通过频域分解和再进入模块对点云动作序列进行低频和高频特征的显式建模，并在 Transformer 结构中实现跨轴融合与再对齐，以提升时空动作识别效果。

**💡 创新点**

创新点在于：①首次将可学习的波形变换与频域注意力引入点云动作识别；②设计 Spectral Decomposition Block（SDeBlock）和 Spectral Re‑Entry Block（SReBlock）实现空间与时间频谱的分离与重整；③提出频谱感知学习策略（频域对比学习 + 频率课程调度）来增强低频结构和高频细节的辨别力。

**🔧 技术方法**

技术包括：可学习的离散小波变换（LWT）、频域注意力（SSA/STA）、Transformer 自注意力、4D 卷积、频域对比损失、频率课程调度、梯度注意力可视化（Grad‑CAM）等。

**📊 数据集**

使用三个主流 3D 动作数据集：MSR‑Action3D、NTU‑RGBD（Cross‑Subject / Cross‑View）以及 NTU‑RGBD120，覆盖多主体、多视角及细粒度动作。

**📈 对比分析**

与 PST‑Transformer、P4Transformer、PSTNet 等现有方法在所有数据集进行对比，SRENet 在 MSR‑Action3D 取得 88.75%（比 PST‑Transformer 高 0.96%），在 NTU‑RGBD Cross‑Subject/Cross‑View 分别获得 93.2%/93.5%（超过 1%），在 NTU‑RGBD120 Cross‑Subject 也刷新最佳成绩，并在 hard、medium、easy 细粒度分类中显著提升（尤其 hard 集上 7.8%）。

**⚠️ 局限性**

局限性在于对仅通过局部细节区分的极其相似动作（如 wear‑shoes vs. take‑off‑shoes）仍易混淆，缺乏对动作顺序、运动方向和接触关系的显式建模；对少样本或零样本任务虽有进展但仍需进一步强化泛化能力。

---

## 269. ACRONYM: Accelerated Approximate Nearest Neighbor Search in Memory for Dynamic Vector Databases

**arXiv ID:** 2606.03151 | [PDF](https://arxiv.org/pdf/2606.03151v1)

**作者:** Md Mizanur Rahaman Nayan `[一作]` (Georgia Institute of Technology), Azad J Naeemi `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 5918 | [OpenAlex ID](https://openalex.org/A5080526846)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

提出了ACRONYM平台，实现动态向量数据库的高吞吐量近似最近邻搜索，并支持持续更新而不需重建索引。

**💡 创新点**

创新点在于使用分布式无关的随机投影二值编码、两阶段CAM搜索与时间多路复用近似top‑k选择、XOR‑Accumulate systolic encoder，兼顾硬件加速与动态更新的鲁棒性。

**🔧 技术方法**

技术包括内容可寻址内存（CAM）、非易失性内存（FeFET、SOT‑MRAM等）、XAC处理单元、时间多路复用top‑k、银行级并行写策略。

**📊 数据集**

采用公开向量数据集：GloVe、DEEP、Yandex TTI、SIFT1M/10M/100M 等进行评测。

**📈 对比分析**

通过与 HNSW、DiskANN、FAISS‑IVF、FAISS‑IVFPQ 以及 GPU L40S 等基准在 CPU/GPU 上对比，ACRONYM 在百万级数据上实现>90%召回率、约8M查询/秒、32MB 内存、2.56µJ/查询，速度分别比 HNSW 提升约400×、比 FAISS‑IVF GPU 提升约80×。

**⚠️ 局限性**

局限在于对 PVT 变化、写热点导致的误差仍存在，动态更新时 AU 锁定可能略微降低召回；极大维度与上亿级规模受 CAM 尺寸与写耐久性限制。

---

## 270. Section-Weighted Hybrid Approach for Legal Case Retrieval

**arXiv ID:** 2606.03138 | [PDF](https://arxiv.org/pdf/2606.03138v1)

**作者:** Rajith Arulanandam `[一作]` (University of Moratuwa), Nisansa de Silva `[通讯]` (University of Moratuwa)

**通讯引用:** 632 | [OpenAlex ID](https://openalex.org/A5060065532)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个两阶段、基于章节的法律案例检索框架，先用离线确定性LLM将判决文本拆分为事实、问题、决定与推理四个部分，然后在候选生成阶段融合BM25与稠密向量检索，最后在每个章节上进行“同类同比”比较并归一化融合得分，最终给出相关章节文本、简短理由与当事人立场标签。

**💡 创新点**

创新点在于：①离线章节拆分实现结构化检索，②对词法与语义信号使用查询级Z-score归一化解决尺度不匹配，③为每个章节学习权重并在重排序时按章节加权融合，实现真正的逻辑对齐；④在结果中加入可解释的章节级理由和立场标签。

**🔧 技术方法**

主要技术包括：离线确定性LLM（温度0）做章节提取；BM25倒排索引和稠密向量检索（使用ANN）；RRF（Reciprocal Rank Fusion）进行候选池合并；对每个章节的BM25得分和余弦相似度进行查询级Z-score归一化；学习的章节权重向量以及α、β的融合系数；最终使用确定性LLM生成解释与立场标签。

**📊 数据集**

使用的公开数据集为COLIEE 2025（加拿大联邦法院判例），包含 7,348 篇结构化判决与 1,677 个查询；以及小规模的 1,500 篇判决与 312 个查询的开发集，所有判决均已拆分为四个章节。

**📈 对比分析**

与传统检索（BM25、词向量）以及先进神经检索（BERT、Legal-BERT、Condenser、CoT-MAE 等）相比，Section‑Aware 体系在全规模实验中显著提升 P@1、MRR@10、F1 等指标（如 P@1 由 0.2628 提升至 0.3045，MRR@10 从 0.1179 提升至 0.2019），并保持约 94% 的候选池覆盖率。

**⚠️ 局限性**

局限性包括：①离线章节拆分依赖LLM，可能出现结构错误或无法捕获细粒度法律细节；②模型仍使用静态向量表示，难以即时更新新判决；③在跨司法管辖区的推广效果未知；④尽管性能提升明显，但整体召回率与精确度仍受限于训练数据与预训练模型的法律知识深度。

---

## 271. Excessive use, ill use and misuse of Bibliometrics

**arXiv ID:** 2606.03117 | [PDF](https://arxiv.org/pdf/2606.03117v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 272. Think-Before-Speak: From Internal Evaluation to Public Expression in Multi-Agent Social Simulation

**arXiv ID:** 2606.03137 | [PDF](https://arxiv.org/pdf/2606.03137v1)

**作者:** Kaiqi Yang `[一作]` (Michigan State University), Hui Liu `[通讯]` (Michigan State University)

**通讯引用:** 128848 | [OpenAlex ID](https://openalex.org/A5100358128)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Think-Before‑Speak 框架，采用细粒度时间间隔分离代理的私下推理与公开发言，实时追踪内部状态与发言意图，并在模拟城市议会讨论中评估其效果。

**💡 创新点**

创新点包括：① 将思考与发言解耦，利用冲突解决的间隔式交互；② 设计可解释的内部评估指标（认知失调与沉默压力）；③ 结合自选发言模式与多种记忆策略，支持过程层面的社会科学分析。

**🔧 技术方法**

使用技术：LLM 驱动的多代理系统（Gemini‑2.5‑Flash Lite/Flash）、离散时间间隔协议、内部状态结构化记录、逻辑混合效应模型与线性混合效应模型进行量化评估。

**📊 数据集**

数据集：以“太阳能光伏强制令”议题为背景，构建六种气候态度（警醒、关注、谨慎、疏离、怀疑、否认）与三对支持/反对的代理人格，结合真实人类个人资料作为模拟角色。

**📈 对比分析**

比较方法：通过对比自选发言与固定轮次、不同记忆模式、强制发言等实验设计，使用混合效应模型评估内部状态指数、说话意图与公开表达的关系；结果表明内部评估显著预测发言意图，且不同设置下内部轨迹差异显著，模型拟合优于基线。

**⚠️ 局限性**

limitations：记忆管理仅采用启发式时间截断与摘要，未结合心理学记忆理论；未使用检索增强或相似性检索；实验规模受限，未检验更大代理数或多领域场景的泛化能力。

---

## 273. Parallel Metric Skiplists and Nearest Neighbor Search

**arXiv ID:** 2606.03129 | [PDF](https://arxiv.org/pdf/2606.03129v1)

**作者:** Xiangyun Ding `[一作]` (University of California, Riverside), Yihan Sun `[通讯]` (University of California, Riverside)

**通讯引用:** 1215 | [OpenAlex ID](https://openalex.org/A5004374151)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

提出了一个工作量为 O(n log n) 期望、跨度为 O(log³n) 的并行构造算法，用于构建度量跳表（Metric Skip‑List），并在常数扩展率假设下实现了 O(log n) 的最近邻查询。

**💡 创新点**

创新点在于将原本顺序且带有 O(log log n) 额外因子的构造/查询过程通过“人工同步阈值”与“控制树”技术重构为并行友好、工作量保持最优的结构，首次在仅依赖常数扩展率的情况下实现了高效的并行近邻搜索。

**🔧 技术方法**

核心技术包括随机置换、指针预处理（ik[j] 指针）、控制树（Control Tree）划分、分治并行构造以及基于随机游走的指针对齐（Align‑All）等，并结合工作-跨度模型实现并行化。

**📊 数据集**

本文主要为理论性工作，未给出具体实验数据集，分析基于理论假设（常数扩展率）和随机置换。

**📈 对比分析**

与覆盖树（Cover Tree）相比，在相同假设下构造时间和查询时间相同（O(n log n) 与 O(log n)），并且实现了可观的并行性（O(log³n) 跨度），在理论上大幅度提升了并行近邻搜索的效率。

**⚠️ 局限性**

局限性包括：仅提供构造和查询的并行方案，未考虑并发/可更新版本；缺乏实验评估与实际数据集验证；对随机置换的依赖可能在实际应用中带来实现成本。

---

## 274. Learning to See via Epiretinal Implant Stimulation in silico with Model-Based Deep Reinforcement Learning

**arXiv ID:** 2606.03118 | [PDF](https://arxiv.org/pdf/2606.03118v1)

**作者:** Jacob Lavoie `[一作]` (Université de Sherbrooke), Eric Plourde `[通讯]` (Université de Sherbrooke)

**通讯引用:** 421 | [OpenAlex ID](https://openalex.org/A5058948615)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究通过将视网膜刺激问题视作笔刷渲染任务，训练深度强化学习智能体以选择电极刺激序列来重构数字图像。

**💡 创新点**

创新点在于利用不良的异向磷光形状而非消除它们，并采用感知指标（Wasserstein距离和MSSIM）作为奖励与评估。

**🔧 技术方法**

主要技术包括基于模型的Deep SAC强化学习、轴向神经映射(axon map)生成磷光、以及Wasserstein距离的奖励设计。

**📊 数据集**

使用MNIST手写数字数据集作为目标图像进行训练与评估。

**📈 对比分析**

与传统的NSA单电极刺激算法比较，SAC智能体在MSSIM上显著提升（从0.07提升至0.35），虽然像素误差略高，但整体可读性更好。

**⚠️ 局限性**

局限性包括单电极刺激独立假设、轴向模型计算慢、缺乏多电极联合刺激、以及未考虑真实患者的眼动与非线性响应。

---

## 275. SPOQ: Specialist Orchestrated Queuing for Multi-Agent Software Engineering

**arXiv ID:** 2606.03115 | [PDF](https://arxiv.org/pdf/2606.03115v1)

**作者:** Royce Carbowitz `[一作]` (Pinpoint Technologies LLC), Dheeraj Kumar `[通讯]` (Pinpoint Technologies LLC)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了SPOQ（Specialist Orchestrated Queuing）——一种针对软件工程的多代理调度与质量控制方法。

**💡 创新点**

创新点包括：①基于图论的波式（wave-based）并行调度；②双重验证门（计划验证与代码验证）与阈值；③Human‑as‑Agent（人类参与者作为高价值代理）以及①三层代理层次（高效执行、审查、快速排查）。

**🔧 技术方法**

技术手段：LLM（Claude、Qwen）驱动的任务生成与执行，DAG拓扑排序与波分配，10项量化验证指标，双门质量阈值，日志追踪系统，人与代理的双向交互接口。

**📊 数据集**

数据集：合成DAG（多种规模与深度）、四个全栈基准任务（后端、前端、基础设施、测试与文档），以及17个真实仓库共122个Epic、1,822个任务、13,866次测试，覆盖约8,589次提交。

**📈 对比分析**

对比方法：在控制实验中将波式调度与序列、FIFO、角色序列等基线对比；结果显示在无资源限制下可达14.3×加速、在2槽硬件上稳定1.4×；计划质量提升至99.75%覆盖率，缺陷率从0.34↓0.20，测试通过率从91.25%↑99.75%；加入人类协作后缺陷进一步下降至0.03，通关率提升至99.75%。

**⚠️ 局限性**

局限性：需要前期精细规划且受人类专业度限制；实验规模有限（仅四个基准任务、单一云模型）；实现目前依赖Claude Code与Anthropic生态；未实现跨Epic学习与模型自适应；验证指标的LLM评估可能存在偏差；成本估算基于令牌数，未覆盖完整运营费用。

---

## 276. Inverting the Generation Process of Denoising Diffusion Implicit Models: Empirical Evaluation and a Novel Method

**arXiv ID:** 2606.03111 | [PDF](https://arxiv.org/pdf/2606.03111v1)

**作者:** Yan Zeng `[一作]` (Tohoku University), Takayuki Okatani `[通讯]` (Tohoku University)

**通讯引用:** 3657 | [OpenAlex ID](https://openalex.org/A5009259465)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了如何反向推断DDIM图像生成过程中的潜变量，尤其是恢复生成图像所对应的初始噪声图

**💡 创新点**

创新点在于提出混合逆向方法：第一步使用梯度下降直接优化单步DDIM逆算，后续步骤采用固定点迭代，从而在保持重建质量的同时显著提升潜变量预测精度

**🔧 技术方法**

主要技术包括DDIM生成模型、梯度下降直接逆算、固定点迭代、以及自回归插值测试（self‑interpolation）来评估潜变量的真实性

**📊 数据集**

使用了三个公开数据集：CelebA、LSUN Bedroom、LSUN Church，均使用Google预训练的DDIM模型进行实验

**📈 对比分析**

与传统DDIM逆转、pix2pix‑zero、AIDI、ReNoise等方法对比，实验表明在三项评估指标（潜变量MSE、重建MSE、自回归插值MSE）上，混合方法均取得最低误差，尤其在潜变量预测和自回归插值方面表现最优

**⚠️ 局限性**

主要局限是计算开销较大（尤其第一步梯度下降需要约1,000次迭代，整体耗时显著高于其它方法），且实验仅限于无条件图像空间的DDIM，未验证在条件或潜空间模型上的适用性

---

## 277. OpenAgenet/OAN: Technical Architecture for Trust-Governed Agent Identity and Discovery

**arXiv ID:** 2606.03163 | [PDF](https://arxiv.org/pdf/2606.03163v1)

**作者:** Jinliang Xu `[一作]` (China Academy of Information and Communications Technology), Jinliang Xu `[通讯]` (China Academy of Information and Communications Technology)

**通讯引用:** 1176 | [OpenAlex ID](https://openalex.org/A5042784076)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并实现了 OAN（OpenAgenet）协议栈，构建了面向 Agent 交互的协议中立信任层，涵盖 Agent 身份管理、注册、Root 认可、包分发、授权授权的发现、签名调用以及安全属性；

**💡 创新点**

创新点在于：①将 Root 授权、身份生命周期和发现授权分层，形成可组合的协议栈；②使用 Root 验证的包、签名响应与时间戳/随机数防重放等机制，确保跨协议（MCP、A2A、ANP）前置信任；③定义了可扩展的协议对象、规范化的哈希与签名约束；④提供多角色实现剖面，支持逐步部署；

**🔧 技术方法**

技术包括：DID/VC 语义、签名与哈希绑定、Root‑签名验证、证书凭证、nonce 与时间戳防重放、授权域过滤、分页与批处理、事件日志、区块链备份公告、SDK 与多语言实现；

**📊 数据集**

实验使用合成身份表示与自定义测试向量，生成 2000 个身份实例进行生命周期与同步实验；

**📈 对比分析**

与无安全检查的基线版本对比，发现完全 OAN 路径在所有非法案例下无误接受；性能上单节点 2000 个身份可完成注册、同步与查询，但大结果查询与同步存在瓶颈，提示分页、索引优化与队列硬化是改进方向；

**⚠️ 局限性**

局限包括：仅单根 Trust Domain，未实现多 Root 联邦；缺乏隐私保护的发现与查询；真实生产环境中网络延迟、节点失效、密钥轮换等未完整评估；并且对业务协议的后续互操作性仍需进一步实验验证。

---

## 278. ClinicalMC: A Benchmark for Multi-Course Clinical Decision-Making with Large Language Models

**arXiv ID:** 2606.03157 | [PDF](https://arxiv.org/pdf/2606.03157v1)

**作者:** Ruihui Hou `[一作]` (East China University of Science and Technology), Tong Ruan `[通讯]` (East China University of Science and Technology)

**通讯引用:** 1251 | [OpenAlex ID](https://openalex.org/A5005820786)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个多轮临床决策基准 ClinicalMC，包含中英文样本，覆盖从入院到出院的四个阶段（分诊、首轮检查/诊断/治疗、后续多轮检查/评估/治疗、最终诊断）。

**💡 创新点**

创新点包括：① 引入多轮（多课程）临床决策评估，真实反映患者病程演变；② 设计多代理评估框架 SimHospital（patient、examiner、doctor）实现动态交互；③ 统一制定任务公式与评价指标，系统化评估 LLM 的多课程推理能力。

**🔧 技术方法**

技术手段：使用多代理框架与 LLM（如 GPT‑4o‑mini、DeepSeek‑V3.2、HuatuoGPT‑o1 等）完成分诊、检查推荐、诊断、评估、治疗规划与最终诊断任务；采用单轮静态与多轮动态实验设置；通过 Accuracy、Recall、F1、IoU 等指标进行评测。

**📊 数据集**

数据集：1,275 条中文样本（16 科室）与 5,804 条英文样本（24 科室），平均课程数分别为 3.42（中文）和 5.11（英文）。样本来源于 MedEureka（中文）和 PMC‑Patients（英文），经过匿名化、质量控制与专家审核。

**📈 对比分析**

比较方法：对比医疗 LLM、开源 LLM、闭源 LLM 与人工对照，在单轮与多轮实验中进行。结果显示，所有 LLM 的平均性能远低于人类，最佳模型在英文集约为 49.68%，中文集约为 48.17%，多轮场景性能更低。

**⚠️ 局限性**

局限性：① 缺少多模态信息（影像、时间序列等），只能处理文本；② 科室分布不平衡且单一数据来源，可能影响模型在少数科室的泛化能力。

---

## 279. A cross-domain tropical species dataset with Chinese vernacular names and CITES source links

**arXiv ID:** 2606.03156 | [PDF](https://arxiv.org/pdf/2606.03156v1)

**作者:** Jeff Wang `[一作]` `[通讯]` (NEXLY LLC), Jeff Wang (NEXLY LLC)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个包含41万种热带物种的跨域数据集，涵盖植物、淡水动物及异国宠物，并为每种物种提供跨域分类、中文俗名（附 Provenance）和CITES Species+链接。

**💡 创新点**

创新点在于将跨域商业、贸易与养殖上下文重新划分，采用多源权重投票与精确匹配门控筛选LLM生成的中文名称，并实现只依据identifier重新分发，满足CC‑BY 4.0。

**🔧 技术方法**

使用了跨源标识符整合、同义词解析、加权投票、精确匹配门控、生产级纠错锁（ratchet）等技术。

**📊 数据集**

数据集来源于GBIF、POWO、iNaturalist、NCBI Taxonomy、Catalogue of Life、Encyclopedia of Life以及Species+，并通过中文权威文献和中文数据库收集中文俗名。

**📈 对比分析**

通过与主要国际生物多样性基础设施的覆盖率比较，中文名称覆盖率从≈3%‑8%提升至99.5%；内部审计（N=50）显示LLM中文名称准确率约90%，CITES链接准确率100%。

**⚠️ 局限性**

主要限制包括未完成对LLM中文名称的外部盲审、对中文列表的独立验证、CITES链接的完整性检验不足以及数据版本漂移导致的可重复性挑战。

---

## 280. Cost-Aware Optimization for Agentic Query Execution

**arXiv ID:** 2606.03152 | [PDF](https://arxiv.org/pdf/2606.03152v1)

**作者:** Lunyiu Nie `[一作]` (University of Texas at Austin), Swarat Chaudhuri `[通讯]` (University of Texas at Austin)

**通讯引用:** 4961 | [OpenAlex ID](https://openalex.org/A5057341982)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将 LLM 语义推理与传统 SQL 操作动态交错的查询执行模型，并基于该模型设计了自适应的工作流优化器 EnumGRPO；

**💡 创新点**

将查询优化从静态等价计划搜索扩展到既考虑执行成本又考虑答案质量的动态工作流搜索，利用枚举计划空间与无梯度的上下文强化学习提炼可迁移的规划启发式；

**🔧 技术方法**

计划空间枚举（5 维轴）、Group‑Relative Policy Optimization (GRPO)、多目标奖励（执行准确率、tuple/cell F1 与成本惩罚）、四阶段经验蒸馏、无梯度的上下文 RL；

**📊 数据集**

SWAN 基准（120 个跨域问答，4 个 SQLite 数据库）作为训练/评估数据集；

**📈 对比分析**

与 Agentic Text2SQL、Agentic BlendSQL 进行对比；EnumGRPO 在 SWAN 上实现 35.4% 的执行准确率，成本降低 317 倍至仅 $0.011/问，准确率比基线高 18%~84%；

**⚠️ 局限性**

局限性：需先行进行 200 次推演学习阶段，依赖 LLM API，无法在 LLM 版本升级后自动迁移；仅在 SWAN 数据上验证，可能对不同数据域或更大规模的查询适用性有限；经验池增大导致提示长度增加，可能影响推理速度。

---

## 281. $A^2$: Smaller Self-Supervised ViTs Localize Better than Larger Ones

**arXiv ID:** 2606.03148 | [PDF](https://arxiv.org/pdf/2606.03148v1)

**作者:** Sreehari Rammohan `[一作]` (Columbia University), Carl Vondrick `[通讯]` (Columbia University)

**通讯引用:** 7752 | [OpenAlex ID](https://openalex.org/A5103033393)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为A^2的鲁棒视觉分类方法，利用预训练视觉Transformer的注意力图自动选取图像中心区域进行硬裁剪，然后用另一预训练模型提取特征并进行分类；通过在多种分布偏移基准上实现对空间伪相关的抑制。

**💡 创新点**

创新点在于：①发现自监督ViT的注意力随模型增大出现逆向缩放，较小模型在前景定位上更优；②基于此逆向缩放设计A^2，显式将“看哪里”和“提取什么”分离；③在注意力不对齐时引入小型MLP适配器调优注意力，进一步提升鲁棒性。

**🔧 技术方法**

技术上主要使用预训练的自监督ViT（如DINOv2、DINOv3、MAE、iBOT）产生注意力图；通过均值头聚合和贪心裁剪算法生成硬crop；随后使用另一个ViT或CLIP模型提取224×224裁剪的特征，最后用线性回归或零样本CLIP进行预测。

**📊 数据集**

实验使用五个分布偏移数据集：Spawrious O2O Hard、Spawrious M2M Hard、Waterbirds、MetaShift Cat vs. Dog以及新构造的MetaShift Animals（训练与测试的上下文完全分离）。

**📈 对比分析**

与传统无群标签方法（线性probe、DFR、iFAM、TTR等）及需要群标签的基线进行对比；A^2在worst‑group accuracy上提升约10–30个百分点，且整体准确率保持或略优；在多数据集上均优于基线，并能与loss‑level方法互补。

**⚠️ 局限性**

局限性：仅针对空间性伪相关有效，对颜色、纹理等非空间伪相关不具备优势；需要预训练注意力与任务对齐，否则需额外适配器；在极端分布偏移或注意力过度分散的大模型上crop质量可能下降。

---

## 282. The Case for Text-to-SQL Friendly Logical Database Design

**arXiv ID:** 2606.03145 | [PDF](https://arxiv.org/pdf/2606.03145v1)

**作者:** Shi Heng Zhang `[一作]` (Simon Fraser University), Jiannan Wang `[通讯]` (Tsinghua University)

**通讯引用:** 12807 | [OpenAlex ID](https://openalex.org/A5042417097)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计了面向大型语言模型的逻辑数据库模式转换框架，提出了Schema Abstraction、Partitioning和Renaming三种语义保持的变换，显著提升Text‑to‑SQL的执行准确率。

**💡 创新点**

创新点在于把逻辑数据库设计视为LLM友好的新优化目标，将传统的范式化与视图抽象、分区化、重命名等技术统一为可组合的模型友好变换。

**🔧 技术方法**

技术包括基于工作负载的视图抽象、频繁项集分区化、语义重命名，以及离线模式优化与在线提示拼装相结合的框架。

**📊 数据集**

主要使用了统一化的BIRD‑Union和Spider‑Union两个基准数据集，并在多种LLM（GPT‑4、Gemini、GPT‑5、Qwen2.5‑Coder）和多条Text‑to‑SQL管线（BaseSQL、DIN‑SQL、MAC‑SQL、CSCSQL）上进行评测。

**📈 对比分析**

与基线模式对比，+A+P+R组合在多条管线上普遍提升1%–4%的执行准确率，最高在GPT‑4‑mini上BIRD‑Union提升2.6%、Spider‑Union提升1.8%；在零样本无历史负载场景下也能获得1%+的增益。

**⚠️ 局限性**

局限性包括对域特定语义误匹配可能导致性能退化、部分数据库在某些变换后表现下降，以及对极大规模模式仍需动态视图成本评估和自适应路由等进一步研究。

---

## 283. FederatedSkill: Federated Learning for Agentic Skill Evolution

**arXiv ID:** 2606.03143 | [PDF](https://arxiv.org/pdf/2606.03143v1)

**作者:** Jingbo Yang `[一作]` (University of California Santa Barbara), Shiyu Chang `[通讯]` (University of California Santa Barbara)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 FederatedSkill，一个基于语义技能差分的联邦学习框架，实现多用户隐私保护的代理技能演化。

**💡 创新点**

创新点在于用结构化技能补丁而非原始轨迹进行交流，并在服务器端通过演化代理实现客户端个性化技能更新。

**🔧 技术方法**

利用 LLM 进行技能补丁生成与融合，服务器端演化代理采用 POMDP 规划、能力矩阵和两层记忆，客户端使用多模型、多工具环境。

**📊 数据集**

使用 SkillFlow 基准数据集，共 20 个任务族进行实验。

**📈 对比分析**

与单机自演化基线比较，在 20 任务族上平均提升 5.82pp 成功率，最高提升 44.4pp，同时计算成本降低 37.5%。

**⚠️ 局限性**

局限包括仅在可信联邦环境下、对恶意补丁缺乏防护，以及实验规模有限，未探究大规模多客户端场景。

---

## 284. How Visible Are Silent Manipulation Failures? An Observability Study of False-Success Detection in Simulated Robot Episodes

**arXiv ID:** 2606.03134 | [PDF](https://arxiv.org/pdf/2606.03134v1)

**作者:** Aarav Bedi `[一作]` `[通讯]` (University of California, Berkeley), Aarav Bedi (University of California, Berkeley)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究模拟机器人操控任务中“假成功”误标的可观测性，比较关节信息与视觉信息对识别误标的效果。

**💡 创新点**

通过物理扰动生成真实失败而非人工标签编辑，提供基准评估关节与视觉在检测假成功中的相对作用，并量化关节信号的极限可观测性。

**🔧 技术方法**

在Aloha仿真框架下使用脚本控制器生成500条标记为成功的失败样本；提取关节速度统计与图像中目标中心与面积特征，使用梯度提升树分类器；计算Cohen's d评估特征分离度。

**📊 数据集**

两种bimanual ALOHA任务（立方体转移和插头插入）的仿真数据，500条标记成功的轨迹（转移262真成功，238假成功；插入341真成功，159假成功）。

**📈 对比分析**

通过假成功召回率评估，关节检测在转移任务达97%召回率，视觉检测为94%；插入任务关节召回率65%，视觉为94%；整体准确率高，显示视觉在难以关节捕捉的失败中更有效。

**⚠️ 局限性**

关节可观测性基于无噪声仿真，真实传感器噪声会抑制微小速度差异；脚本控制器为oracle，导致插入任务的关节信号受策略影响；实验仅覆盖单个随机种子和两任务，结果为乐观上限。

---

## 285. Coherence Maximization Improves Pluralistic Alignment

**arXiv ID:** 2606.03110 | [PDF](https://arxiv.org/pdf/2606.03110v1)

**作者:** Taslim Mahbub `[一作]` (George Washington University), Shi Feng `[通讯]` (George Washington University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究探讨了如何在没有人类监督的情况下生成有效的价值示例，以使AI系统与多样化的人类价值观对齐。通过内部一致性最大化（ICM）方法，生成特定于人群的示例，以引导模型朝向目标群体的价值观。

**💡 创新点**

创新点在于提出了内部一致性最大化（ICM）方法，该方法能够在没有人类监督的情况下生成高一致性的价值示例，并且这些示例在多个基准测试中表现出与人工标签相当的性能。

**🔧 技术方法**

使用了内部一致性最大化（ICM）技术，该技术通过最大化标签之间的相互可预测性来推断标签。

**📊 数据集**

使用了四个数据集，包括GlobalOpinionQA、OpinionQA、Persona-Tailoring和OvertonBench，涵盖分类、偏好和开放式生成任务。

**📈 对比分析**

与零-shot方法相比，ICM推断的标签在所有基准测试中均表现出与金标准相当的性能，且在一致性较高的情况下，模型的泛化能力显著优于仅依赖标签准确性的基线方法。

**⚠️ 局限性**

限制在于ICM搜索过程对超参数敏感，可能需要针对特定数据集进行调整。此外，提取的价值表示是基于文本的统计模式，可能无法完全反映人类意图，尤其是在某些地区的代表性不足。

---

## 286. EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning

**arXiv ID:** 2606.03108 | [PDF](https://arxiv.org/pdf/2606.03108v1)

**作者:** Guhong Chen `[一作]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences), Jieping Ye `[通讯]` (Tongyi Lab, Alibaba Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 EvoTrainer，一种自治训练框架，能够同时进化 LLM 策略和训练侧诊断工具。

**💡 创新点**

将训练侧诊断工具视为可进化对象，构建跨版本的政策-诊断协同演化机制，并通过可重用技能库与记忆库提升训练效率。

**🔧 技术方法**

采用 RL 技术（GRPO、Clip-Higher 等）与版本控制、四层诊断（score、signal、behavior、version）、可重用技能库、检索与后测、自动决策循环；训练师模型选用 Claude Sonnet 4.6。

**📊 数据集**

数学推理使用 BigMath‑Hard，编程使用 TACO‑verified 与 AtCoder 子集，软件工程使用 swe‑rebench‑v6。

**📈 对比分析**

与无 RL 基线、人类工程化 RL 基线以及 AutoResearch 等对比，在 Math、Coding、SWE 三个领域均取得领先，SWE‑9B 达到 38.16% BC，最大提升 4.39%，在数学和编程也显著优于基线。

**⚠️ 局限性**

算力受限，训练周期较长；依赖强大上下文推理与检索能力；仅演化了少量版本，需进一步扩展记忆与技能库的管理。

---

## 287. Synthetic Hallucinations, Real Gains: Hard Negatives from Frontier Models for FIM Hallucination Mitigation

**arXiv ID:** 2606.03130 | [PDF](https://arxiv.org/pdf/2606.03130v1)

**作者:** Mahdi Erfanian `[一作]` (University of Illinois Chicago), Shengyu Fu `[通讯]` (Microsoft)

**通讯引用:** 1754 | [OpenAlex ID](https://openalex.org/A5031659418)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过生成假设完成的硬负样本，构建无执行的 FIM 微调流程，显著降低小型开源代码模型的幻觉。

**💡 创新点**

创新性地利用前沿模型产生的错误但合理完成作为硬负样本，配合“fool rate”难度筛选和多语言多类型覆盖，提出端到端的无执行 FIM 微调方案。

**🔧 技术方法**

使用 GPT‑5.x 前沿生成器产生硬负样本，LLM 判定 panel 评估 fool rate，随后对 Qwen2.5‑Coder、StarCoder2、CodeLlama 等模型进行全参数 SFT，并对比 DPO/ORPO。

**📊 数据集**

采集公开许可 GitHub 代码构成 400k FIM 上下文，生成 2.5M 假设完成；评测使用 Delulu、多语言 FIM 幻觉基准以及 HumanEval‑Infilling、SAFIM、Real‑FIM‑Eval。

**📈 对比分析**

对比基线与微调模型，在 Delulu 上 EM 提升 18.8（7B）/12.8（3B），在 SAFIM、HumanEval 上也提升多达 +20 EM；SFT 与 ORPO 对齐，DPO 下降；3B 在部分 general FIM 下降，显示容量 trade‑off。

**⚠️ 局限性**

仅覆盖四类 identifier‑level 幻觉，未涵盖所有真实错误；Real‑FIM‑Eval 仍低性能；3B 模型容量受限导致部分回退；需要强生成模型和大规模算力；微调检查点未公开。

---

## 288. Beyond "To whom it may concern": Tailoring Machine Translation to Audience and Intent

**arXiv ID:** 2606.03259 | [PDF](https://arxiv.org/pdf/2606.03259v1)

**作者:** Raphael Merx `[一作]` (University of Melbourne), Trevor Cohn `[通讯]` (Google)

**通讯引用:** 7911 | [OpenAlex ID](https://openalex.org/A5078530959)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在机器翻译中引入用户说明（purpose instruction）以实现目的驱动的翻译，并在50+语言、5模型规模、8个文本域上系统评估其效果。

**💡 创新点**

创新点在于：①将翻译视为对目的和受众的忠实；②证明LLM在给定明确说明时能显著提升适配度；③提出自生成说明（self‑instruction）方案，弥补无说明时的性能缺口；④揭示传统评估指标对适配度的不足，强调需基于目的的评估。

**🔧 技术方法**

技术方法包括：使用Gemma-3/4系列大语言模型，结合指令提示（prompting）实现说明驱动；对照四种条件（无说明、说明、语义相似few‑shot、段落级上下文）；利用Gemini‑3‑Flash LLM‑judge进行无参考评分；并通过LoRA/SMOL进行小模型适配。

**📊 数据集**

数据集主要有：BOUQuET（多域、元数据丰富）用于主实验；WMT24++（>50语言）用于验证；同时使用NLLB等公开资源。

**📈 对比分析**

与传统指标（XCOMET‑XL、ChrF++）及无说明基线相比，说明条件在大模型（≥12B）上平均提升适配度+10–20点，翻译质量+2–5点；在非正式域（社交、对话）提升尤为显著。自生成说明可恢复约70–80%原有适配度提升。

**⚠️ 局限性**

局限性包括：①评估集中在英语源，未覆盖多源语言；②仅测试Gemma系列模型，可能存在供应商共性偏差；③自说明方法仍依赖较大模型，低资源语言和小模型易出现指令跟随失败；④LLM‑judge的可靠性在低资源语言上相对较弱，需进一步验证。

---

## 289. PSViT: A Methodology for Structurally Pruning Spiking Vision Transformers

**arXiv ID:** 2606.03257 | [PDF](https://arxiv.org/pdf/2606.03257v1)

**作者:** Rachmad Vidya Wicaksana Putra `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11438 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `fede83ac-7505-405f-ab37-e7284695c47f` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对Spiking Vision Transformers进行单次结构化通道裁剪，实现模型压缩。

**💡 创新点**

提出基于层级敏感性分析的单次裁剪流程，兼顾内存与精度。

**🔧 技术方法**

采用通道重要度计算、均匀裁剪、块级细化裁剪等技术。

**📊 数据集**

在ImageNet-1K数据集上进行评估。

**📈 对比分析**

与未裁剪的SDTv2相比，PSViT在保持3%以内准确度下降的同时，内存减22.4%，FLOPs减17.5%。

**⚠️ 局限性**

仅针对SDTv2验证，缺乏对其他SViT模型的泛化及在极端稀疏场景下的鲁棒性。

---

## 290. Quantum-Classical Equivalence for AND-Functions

**arXiv ID:** 2606.03249 | [PDF](https://arxiv.org/pdf/2606.03249v1)

**作者:** Sreejata Kishor Bhattacharya `[一作]` (TIFR), Shachar Lovett `[通讯]` (UC San Diego)

**通讯引用:** 2109 | [OpenAlex ID](https://openalex.org/A5044999305)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b`

**🎯 论文内容**

本文证明了对所有布尔函数 f，函数 f ∘_2（即把 AND₂ 作为子函数构造的复合函数）在确定性、随机化和量子通信复杂度之间存在多项式关系，从而在此类函数上验证了 Log-Equivalence Conjecture。

**💡 创新点**

创新点在于将 De Morgan 稀疏度（函数 f 的多项式表示中的非零系数数目）与量子通信复杂度的下界（通过逼近 γ₂ 范数）建立了多项式级别的联系；为此引入了半自适应最大度约束树，并通过随机限制与傅里叶衰减证明了稀疏度上升导致 γ₂ 范数下界提升。

**🔧 技术方法**

核心技术包括：
1. 构造半自适应最大度约束树，确保对 f 的大量约束仍保持最大度；
2. 将这些约束通过 AND₂ 捷径提升为对 f ∘_2 的限制；
3. 在约束下对所有量子近似表示做平均，利用傅里叶层次的指数衰减得到低阶多项式近似；
4. 通过矛盾论证得到 γ₂ 范数下界，从而得到量子通信复杂度下界；
5. 结合已有的稀疏度→确定性复杂度的上界，完成三种复杂度的多项式等价。

**📊 数据集**

本文不使用任何实验数据集，研究完全基于理论分析。

**📈 对比分析**

与已有结果比较时，本文把此前已知的仅在对称外函数或单调外函数下成立的多项式等价扩展到所有外函数 f；同时通过稀疏度、块灵敏度等结构量对确定性、随机化和量子复杂度给出了相同阶数的界限，证明了在此类函数上三者的复杂度相差最多多项式因子。

**⚠️ 局限性**

局限性包括：
1. 结果仅适用于 AND₂ 捷径的复合函数；
2. 具体多项式度数和常数未被优化，仍包含 polylog (n) 因子；
3. 对于更大规模的函数或其他子函数（如 EQ₄ 等），尚未得到相同多项式等价的证明；
4. 证明中依赖于稀疏度上界，若函数稀疏度很低，方法可能失效。

---

## 291. MariData: One-Step Unpaired Image Translation for Maritime Environments

**arXiv ID:** 2606.03246 | [PDF](https://arxiv.org/pdf/2606.03246v1)

**作者:** Santeri Henriksson `[一作]` (Turku University of Applied Sciences), Juha Kalliovaara `[通讯]` (Turku University of Applied Sciences)

**通讯引用:** 511 | [OpenAlex ID](https://openalex.org/A5063068724)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `67630363-6be0-4f51-ab05-7198250671a5` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研发并实现了基于 CycleGAN‑turbo 的一体化无配对图像翻译框架，能够把白天海景合成为雾天、黄昏和夜晚等恶劣天气/光照条件的训练数据。

**💡 创新点**

创新点在于通过零卷积跳连通绕过 VAE 压缩瓶颈，显著保留小型导航物体细节，并将 Diffusion 与 GAN 结合成单步推理的高效架构，解决传统模型的速度与结构失真问题。

**🔧 技术方法**

使用 CycleGAN‑turbo（Stable Diffusion Turbo 骨干）+ 零卷积 skip 连接、混合精度训练、xFormers 注意力、梯度检查点等技术；对比 HiDT 基线进行评估。

**📊 数据集**

构建并使用 7,000 张未配对海洋图像数据集（白天、黄昏、雾天、夜晚），来源于 Unsplash、Kaggle 等公开数据，按 90/10 训练/测试划分。

**📈 对比分析**

通过循环一致性损失收敛曲线、可视化 round‑trip、强度插值实验以及与 HiDT 的小物体保留对比来评估性能；结果显示 Day‑to‑Foggy 与 Day‑to‑Sunset 在视觉质量与结构保持上明显优于 HiDT，Night 模型能生成暗景但易出现灯光幻觉。

**⚠️ 局限性**

局限性包括数据集不平衡导致夜间模型产生灯光幻觉；在极端天气/光照下仍可能丢失细节；对开海夜景的泛化能力有限。

---

## 292. Right Makes Might: Aligning Verified Hidden States Empowers RL Reasoning

**arXiv ID:** 2606.03234 | [PDF](https://arxiv.org/pdf/2606.03234v1)

**作者:** Ziyue Wang `[一作]` (Peking University), Jiaqi Wang `[通讯]` (JINGDONG)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种在 RL 训练期间对正确推理路径的隐藏状态进行对齐的辅助损失（Hidden-Align），从而提升大语言模型的数学推理能力。

**💡 创新点**

首次将隐藏状态相似性作为 RL 训练的信号，在答案前的 anchor token 上对齐最后一层隐藏状态，并且不增加训练或推理的额外开销。

**🔧 技术方法**

采用 DAPO 的 RLVR 框架，利用 pairwise cosine 相似度构造对齐损失，并通过两阶段梯度计算兼顾 micro‑batch 约束。

**📊 数据集**

训练使用 DAPO‑Math‑17K，评估数据集包括八个数学推理基准（AIME 2024/25/26、AMC 2023/24、HMMT、Minerva Math、OlympiadBench 等）。

**📈 对比分析**

与基线 DAPO 对比，4B 模型平均提升 6.19pp，AIME 2024+11.05pp，HMMT+7.29pp，且在 1.7B/14B 模型上亦显著提升；Pass@k 曲线也更好，且无额外训练或推理成本。

**⚠️ 局限性**

仅在包含特定答案格式的数学问题上验证；anchor 位置需手动确定；λ 固定为 0.001，未采用动态调度；未在 70B+ 规模模型上进行验证。

---

## 293. VirtualMLE: A Virtual ML Engineer that Optimizes Sequential Recommenders

**arXiv ID:** 2606.03221 | [PDF](https://arxiv.org/pdf/2606.03221v1)

**作者:** Shiteng Cao `[一作]` (Tsinghua University), Zhiheng Li `[通讯]` (Tsinghua University)

**通讯引用:** 6022 | [OpenAlex ID](https://openalex.org/A5100765837)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 VirtualMLE，一个基于 LLM 的自动化优化框架，利用执行、反思与记忆闭环来调优序列推荐模型。

**💡 创新点**

创新点在于将 LLM 当作因果推理器，通过反思生成可迁移的认知摘要，并使用层级记忆实现跨数据集知识迁移，实现认知摊销。

**🔧 技术方法**

使用技术包括：LLM 代理（如 GPT‑5.4）、结构化反思模块、长短期层级记忆系统、闭环自优化流程，以及对 SASRec 与 HSTU 等 SR 基础模型的自动化调参。

**📊 数据集**

使用的数据集为 Amazon‑2014 的三子集：Baby、Beauty 与 Pet。

**📈 对比分析**

在与网格搜索、贝叶斯优化、OPRO 以及大型生成推荐器（Tiger、ReaRec、Lc‑Rec、OpenOneRec）对比后，VirtualMLE 在 Recall@K、NDCG@K 上均名列前茅；试验次数比传统 AutoML 减少 5–27 倍，性能提升约 13–28%。

**⚠️ 局限性**

局限性包括：对 LLM 推理成本与算力依赖较大；跨任务（非 Amazon）迁移性能尚未充分验证；在更高维度搜索空间或更大模型规模下仍可能受限于 LLM 的推理能力。

---

## 294. MedCUA-Bench: A Screenshot-Only Benchmark for Clinical Computer-Use Agents

**arXiv ID:** 2606.03203 | [PDF](https://arxiv.org/pdf/2606.03203v1)

**作者:** Jia Yu `[一作]` (Microsoft Research Asia), Shuo Wang `[通讯]` (Fudan University)

**通讯引用:** 31054 | [OpenAlex ID](https://openalex.org/A5100400182)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 MedCUA-Bench，一个面向临床图形用户界面的交互式基准，覆盖 18 个临床场景、10 个医学领域，并提供意图级与步骤级双重目标；

**💡 创新点**

创新点在于：①针对临床软件设计专门的安全感知评估器（涵盖病人身份、数据准确性、信息保真、记录完整性和工作流安全）；②通过真实产品手册重构的合成界面和少量公开真实系统（OpenEMR、OHIF）实现可复现且具临床真实性；③引入意图/步骤对比来区分规划与执行失误；

**🔧 技术方法**

使用 BrowserGym 基础设施，Agents 仅接收屏幕截图和动作空间（像素级点击、键盘输入、滚动等），并通过 deterministic checker 进行安全级评分；

**📊 数据集**

数据集由 216 个基任务组成，分为 18 个场景，使用合成 HTML、OpenEMR 7.0.2、OHIF Viewer 等构建；每个任务有两种目标（意图、步骤）并随机生成患者/病例信息；

**📈 对比分析**

与 23 个模型（6 个闭源、13 个开源）进行对比；闭源模型最高严格成功率 54.2%，开源平均仅 2.5%，在真实 OpenEMR 上所有模型均低于 9%，OHIF 层次表现最佳；意图与步骤目标的差异显示强模型受步骤提示提升，弱模型反而下降；

**⚠️ 局限性**

限制包括：合成界面仅上限真实系统表现，评估仅单次跑，安全检查在当前能力下未触发关键违规；模型在 30 步预算内常被截断，未充分检验错误执行的安全后果；

---

## 295. Fast Organic Crystal Structure Prediction with Unit Cell Flow Matching

**arXiv ID:** 2606.03199 | [PDF](https://arxiv.org/pdf/2606.03199v1)

**作者:** Alston Lo `[一作]` (MIT), Alán Aspuru-Guzik `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于流匹配的生成式晶体结构预测模型，直接在单个晶胞上生成原子坐标和晶格向量，无需批量扩展或三角层。

**💡 创新点**

创新点包括将晶格视为虚拟点、使用对称偏置注意力替代高成本的三角更新、构造数据驱动的晶格源分布、加入体积与配对距离辅助损失以及自条件机制，显著提升速度与精度。

**🔧 技术方法**

采用Diffusion Transformer (DiT) 体系，配合流匹配、对称偏置注意力、优化传输对齐、以及后期UMA能量排名的能量模型。

**📊 数据集**

训练数据来自Cambridge Structural Database (CSD)，评估使用 OXtal Blind Tests (CSP5–7) 和新构造的 CSD Teaching Subset。

**📈 对比分析**

与 OXtal 比较时，生成速度提升 15–30 倍（仅生成），5–8 倍（含能量排名），在 CSP 盲测 Sol@k 更高；在 CSD Teaching Subset Sol@1000 达到 0.763，整体性能显著优于传统方法。

**⚠️ 局限性**

局限性包括未考虑立体化学信息、需预先给定分子复制数 Z、偶尔生成立体冲突或无物理意义的空洞，以及对晶体相似性度量和泛化评估的不足。

---

## 296. Multi-Agent Framework Leveraging Knowledge Graphs for Virtual Commissioning Models

**arXiv ID:** 2606.03255 | [PDF](https://arxiv.org/pdf/2606.03255v1)

**作者:** Max Diekmann `[一作]` (Siemens AG), Dirk Hartmann `[通讯]` (Siemens AG)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个基于知识图谱的多智能体框架，用于半自动化支持离线虚拟调试模型（VCM）的开发，涵盖系统理解、仿真组件生成和跨域信号映射等任务；

**💡 创新点**

创新点在于将PLC工程数据与机械仿真模型数据通过结构化抽取、知识图谱集成，并结合多智能体与检索增强生成（GraphRAG）实现跨域知识的语义检索与推理，首次实现了在VCM早期阶段的半自动化协助；

**🔧 技术方法**

采用的技术包括：结构化抽取脚本（针对Siemens TIA Portal和NX MCD）、Neo4j图数据库、LangChain框架、多智能体架构、GPT‑4o语言模型以及GraphRAG检索技术；

**📊 数据集**

使用的实验数据集为一个实验室级离散制造系统，包含约52,268个PLC自动化元素与3,404个机械仿真元素；

**📈 对比分析**

通过与LLM‑only、图数据库单智能体两种基线进行对比，评估指标为正确率、完整度、可执行性、Hit@1等，实验结果显示在所有任务类上整体得分达94.8%，显著优于基线；

**⚠️ 局限性**

局限性包括：依赖完整且命名规范良好的图谱覆盖，单一工具链（TIA Portal、NX MCD）限制了跨域迁移；大规模联合查询可能导致数据库内存溢出；仿真组件生成受限于模板覆盖范围；需要人工最终验证。

---

## 297. AirDreamer: Generalist Drone Navigation with World Models

**arXiv ID:** 2606.03252 | [PDF](https://arxiv.org/pdf/2606.03252v1)

**作者:** Zian Liu `[一作]` (Tsinghua University), Guyue Zhou `[通讯]` (Tsinghua University)

**通讯引用:** 1148 | [OpenAlex ID](https://openalex.org/A5011913905)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了AirDreamer框架，利用无人机的深度相机和内部世界模型进行端到端导航，能够在未知、拥挤环境中实现目标导向飞行；

**💡 创新点**

核心创新包括：①将环境理解与动作决策解耦，使用世界模型捕获动态与结构；②采用稀疏奖励设计，避免奖励造型导致的局部最优；③通过域随机化实现无调参的 sim‑to‑real 转移；

**🔧 技术方法**

技术手段包括：Dreamer‑V3 递归状态空间模型（RSSM）进行观测编码与预测；演员-评论家强化学习（actor‑critic）在潜在空间中决策；稀疏终端奖励与辅助进展/安全/高度奖励；域随机化参数扰动；

**📊 数据集**

训练与评估在 Isaac Sim（OmniDrones）模拟器中进行，使用随机生成的障碍地图；实机测试使用 RealSense D455 深度相机、FAST‑LIO2 定位和 Pixhawk 6X Pro 控制，测试场景包括 Forest、1m Passage、5m Wall、U‑shape、C‑shape 等；

**📈 对比分析**

与 DepthNav、NavRL、EgoPlanner 等基准方法比较，AirDreamer 在 5 个随机地图上的成功率为 59.3%（±15.8），比最佳基准高 5.3%；在真实无人机上成功完成六类复杂场景，表现出可观的探测与避障能力；

**⚠️ 局限性**

主要局限：在高度动态或高速竞速场景下尚未验证；模型对超大规模地图的扩展性待测试；虽然使用域随机化实现无调参部署，但对极端物理参数变化仍可能失效；

---

## 298. Structures Facilitate Retrieve, Rerank, and Generate

**arXiv ID:** 2606.03247 | [PDF](https://arxiv.org/pdf/2606.03247v1)

**作者:** Yeqin Zhang `[一作]` (Nanjing University), Cam-Tu Nguyen `[通讯]` (Nanjing University)

**通讯引用:** 3624 | [OpenAlex ID](https://openalex.org/A5060261448)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种利用文档结构信息贯穿检索、重排序与生成全过程的文档驱动对话系统框架。

**💡 创新点**

创新点在于：① 通过结构对比增强检索时的负样本采样；② 基于文档结构构建子图用于重排序；③ 在生成阶段将子图信息与重排序分数融合，实现结构增强的加权生成。

**🔧 技术方法**

技术手段包括：BERT/Bi‑Encoder检索、RoBERTa/跨编码器重排序、BART融合式生成、Top‑Down聚合、结构对比学习、子图构造与加权。

**📊 数据集**

使用了 MultiDoc2Dial（英文）和 Doc2Bot（中文）两大公开文档驱动对话数据集。

**📈 对比分析**

与 BM25、DPR、RAG、Re2G、Fusion‑in‑Decoder 等基线进行对比，检索Recall@5提升约1%~2%，重排序Rprec提升≈1%，生成端F1、S‑BLEU、ROUGE均实现或逼近SOTA，显示结构信息显著提升性能。

**⚠️ 局限性**

局限性包括：在结构信息稀缺的 MultiDoc2Dial 上子图对生成无显著帮助甚至略降性能；构造子图及加权过程增加计算与内存开销；结构划分规则对不同领域文档的适用性需要进一步验证。

---

## 299. Benchmarking Speech-to-Speech Translation Models

**arXiv ID:** 2606.03241 | [PDF](https://arxiv.org/pdf/2606.03241v1)

**作者:** Alkis Koudounas `[一作]` (Sony Group Corporation), Emiru Tsunoo `[通讯]` (Sony Group Corporation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 COMPASS 框架，统一并可复现地评估离线 Speech‑to‑Speech Translation (S2ST) 性能，覆盖 46 个指标、8 个维度；

**💡 创新点**

通过数据驱动的相关性过滤，将指标压缩到每个翻译方向 10 个，兼顾方向差异；结合多领域（配音、播客、医学）人类评测，验证指标与真实偏好一致；

**🔧 技术方法**

使用自动化度量（BLEU/ChrF/COMET/BLASER/UTMOS/NISQA-MOS/音频嵌入相似度/情感匹配/时序误差等）和自研的指标集合；

**📊 数据集**

在 FLEURS（10 种语言）和 CVSS（8 种语言）上评测 1,248 个模型‑语言配置，包括端到端、两阶段、三阶段架构；

**📈 对比分析**

结果显示：不同架构在自然度和说话者一致性上存在 30% 以上差距，单一指标排名误导；方向（X→EN 与 EN→X）需要不同指标集合；人类评测确认 COMPASS 排名与主观偏好高度一致；

**⚠️ 局限性**

局限包括：仅评估读音语料，缺乏自然对话评测；依赖 Whisper ASR 可能产生偏差；未考虑延迟和实时性；指标集可能随未来模型演进而变动；

---

## 300. Solipsistic Superintelligence is Unlikely to be Cooperative

**arXiv ID:** 2606.03237 | [PDF](https://arxiv.org/pdf/2606.03237v1)

**作者:** Rakshit S Trivedi `[一作]` (Independent Researcher), Joel Z Leibo `[通讯]` (Google Deepmind)

**通讯引用:** 7388 | [OpenAlex ID](https://openalex.org/A5054808675)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文通过理论分析和案例阐述，指出传统单向优化的“自我中心”AI设计在多智能体环境中会导致自我削弱和合作失效，并提出将AI设计视为多主体协同过程的非自我中心研究范式。

**💡 创新点**

创新点在于：① 明确提出“自我削弱”属性和“训练‑测试‑部署”差距；② 将合作视为多智能体博弈中的均衡选择，而非可训练的单一任务；③ 提出动态评估、机构作为设计原语和保留人类能动性三条新的研究方向。

**🔧 技术方法**

主要使用的技术是理论推导与博弈论框架（将MDP扩展为Markov游戏）、对齐与Goodhart效应的分析，以及对制度机制设计与动态评估方法的概念性阐释。

**📊 数据集**

论文未使用具体数据集，而是基于已有的案例（如餐厅预订、推荐系统、金融交易）和文献中的实验结果进行说明。

**📈 对比分析**

由于该工作是理论性与框架性探索，没有直接的实验对比或数值性能展示；作者通过逻辑推理和对现象的描述，说明传统方法在协同环境中会出现性能退化。

**⚠️ 局限性**

局限性包括：缺乏大规模实证验证；动态评估与机构设计的实现细节仍需进一步研究；对不同领域多智能体交互机制的适用性与可扩展性尚未系统评估。

---

## 301. BotDirector: Robot Storytelling Across the Symmetrical Reality with Multi-modal Interactions

**arXiv ID:** 2606.03223 | [PDF](https://arxiv.org/pdf/2606.03223v1)

**作者:** Zhe Sun `[一作]` (BIGAI), Zhenliang Zhang `[通讯]` (BIGAI)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一个名为BotDirector的交互系统，帮助5–8岁儿童通过自然语言与实物摆放，利用LLM生成剧本并由自导航的群体机器人或机械臂在桌面上执行，完成桌面戏剧表演；

**💡 创新点**

创新点在于将大型语言模型、可触感的物件互动、虚实同步的对称现实以及即时剧本迭代结合，显著降低儿童创作机器人戏剧的技术门槛，并通过物理与虚拟场景的精确对齐提升叙事灵活性与参与感；

**🔧 技术方法**

技术实现包括GPT‑4作为核心推理与脚本生成器，STT/TTS模块实现语音交互，RGB摄像头+AprilTag实现物理空间定位，A*路径规划机器人移动，虚拟环境生成与映射，物体检测与位置跟踪，以及机械臂姿态控制；

**📊 数据集**

论文未使用公开数据集，全部内容均基于儿童自行提供的主题、物品与现场布置；

**📈 对比分析**

目前未给出量化比较或性能评测，作者计划通过后续儿童用户研究来验证系统的易用性与创作效果；

**⚠️ 局限性**

局限性包括：对硬件（摄像头、标签、机器人）依赖度高；导航误差与实时校正仍挑战；仅面向5–8岁儿童，年龄适配不完整；缺乏大规模实验与客观指标评估；虚实空间扩展尚未充分实现。

---

## 302. WebRISE: Requirement-Induced State Evaluation for MLLM-Generated Web Artifacts

**arXiv ID:** 2606.03220 | [PDF](https://arxiv.org/pdf/2606.03220v1)

**作者:** Yuxin Meng `[一作]`, Yujiu Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出基于需求驱动的交互契约图（Interaction Contract Graph，ICG）评估框架，用来衡量多模态大语言模型（MLLM）生成的网页交互是否符合需求。

**💡 创新点**

将交互评估从局部视觉或脚本检查转变为全局需求级别的状态转移合规性；构造ICG、契约引导代理和DOM/视觉双检验；同时覆盖显式功能需求与隐式状态一致性约束。

**🔧 技术方法**

需求归一化、ICG编译、契约引导代理（适配当前页面 DOM）、DOM/视觉双oracle、浏览器执行、自动化断言与诊断、缺陷注入验证。

**📊 数据集**

442个多模态交互任务（文本、Markdown、Sketch、图像、视频）形成的基准集，涵盖五种输入模态；使用14种公开与私有的 MLLM 进行评估。

**📈 对比分析**

在 14 模型上进行宏平均，报告状态覆盖率 S%、转移有效率 T%、显式需求覆盖 Re%、隐式需求覆盖 Ri% 以及总体 R%；最强模型 GPT‑5.5 在视频模态下实现 T≈65.6%、R≈66.3%，显式需求更易通过，隐式约束仍是主要瓶颈，视频模态比文本提升约 10.6pp。

**⚠️ 局限性**

仅评估自包含前端 HTML，未涵盖后端服务、身份认证、外部 API 或持久化数据；评估范围受限于已定义的需求与测试项；需扩展契约模板、API 沙箱和多样化缺陷库以提升覆盖度。

---

## 303. Private Embedding Lookup with Encrypted Compact Queries under Fully Homomorphic Encryption

**arXiv ID:** 2606.03191 | [PDF](https://arxiv.org/pdf/2606.03191v1)

**作者:** Daehyun Jang `[一作]` (Seoul National University), Jung Hee Cheon `[通讯]` (Seoul National University)

**通讯引用:** 10921 | [OpenAlex ID](https://openalex.org/A5044011121)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本论文提出一种在全同态加密（FHE）环境下高效的私有嵌入查找（Private Embedding Lookup, PEL）协议——Independent Vector Evaluation（IVE），通过使用正交离散余弦变换（DCT）基底代替昂贵的一热向量生成，实现仅需O(p)同态运算即可生成加密索引对应的向量；该协议随后与PCMM（Plaintext‑Ciphertext Matrix‑Multiplication）结合，可批量处理多条加密查询，并在完整的FastText文本分类推理中显著降低总时延。

**💡 创新点**

创新点在于：①放弃传统一热向量生成的高成本，改用线性无关向量评估；②选用正交DCT基底保证数值稳定，兼顾CKKS的复数运算；③在压缩表结构下实现查询的低通信量与低乘深；④将此技术嵌入端到端加密推理，突破单层效率瓶颈。

**🔧 技术方法**

核心技术包括：CKKS同态加密方案、离散余弦变换（DCT）正交基底、线性独立向量评估（IVE）、预处理的基变换矩阵ML、批量同态乘法与PCMM、Ring Packing与基准化参数选取。

**📊 数据集**

实验使用的嵌入表包括GloVe.42B.300d、GloVe.6B.50d和GPT‑2；文本分类实验则选取Enron‑Spam和Drugs.com Review数据集，均采用压缩后四个子表、p=256的配置。

**📈 对比分析**

与Kim等人（ICML 2024）提出的单热向量生成方法相比，IVE在向量生成阶段将时延从≈400 ms降至≈3 ms（log p=10）实现最高≈78.4×加速；整体PEL时延提升约18.5–78.4×；在FastText端到端加密推理中，IVE将总时延从≈9.1 s降至≈0.08 s，速度提升约106–109×。

**⚠️ 局限性**

局限性包括：①仍需一定乘深，超大词表或极高维嵌入可能导致深度过高；②依赖离散余弦基底，需预先计算并存储ML，增加离线成本；③在极低通信预算下仍无法进一步压缩；④实验多在单线程CPU环境，实际多核或GPU加速需进一步验证。

---

## 304. Toward Gripper-Integrated Active Electrosense for Pre-Contact Sensing in Underwater Soft Grippers

**arXiv ID:** 2606.03204 | [PDF](https://arxiv.org/pdf/2606.03204v1)

**作者:** Ahsan Tanveer `[一作]` (Peking University), Guangming Xie `[通讯]` (Peking University)

**通讯引用:** 14686 | [OpenAlex ID](https://openalex.org/A5044920558)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究在软抓取器上集成主动电感（electrosense）技术，以在水下接触前提供物体存在的预警信号，并通过有限元仿真和水槽实验验证其可行性。

**💡 创新点**

创新点在于将电感传感器从传统的主体或专用平台迁移到软抓取器本体，实现了紧凑、无接触的预接触检测，为在能见度低的水下环境中补充视觉或触觉信息提供了新方案。

**🔧 技术方法**

采用的技术包括：COMSOL 多物理场仿真（电静场）、正交驱动-感应电极布局、方波激励与多通道采集（NI USB‑6210 DAQ）、基线减法与简单的分离度量 J(V_pp,f) 进行信号评估。

**📊 数据集**

数据集主要由水槽中悬浮的金属球（单一导电目标）构成，实验包含不同激励幅值（5 V、10 V、20 V）和频率（1 mHz–1 kHz）的组合，并与空水基线做对比。

**📈 对比分析**

比较方式是计算空水与目标存在时的电压差异的欧氏距离 J(V_pp,f)，结果显示在高幅值与高频（例如 20 V、1 kHz）下可获得明显的多极信号变化，表明可检测性受激励参数影响显著，但未与其他传感器或算法做定量性能对比。

**⚠️ 局限性**

局限性包括：仅使用单一导电目标、缺乏重复实验与统计可靠性、未考虑电极–电解液界面极化、电路寄生与环境漂移、信号处理简单化、未给出定量的感知阈值或误检率，因而仅能证明概念可行性而非完整的实用性能。

---

## 305. Do Real-World Datasets Contain Natural Experiments? An Empirical Study Using Causal Feature Selection

**arXiv ID:** 2606.03251 | [PDF](https://arxiv.org/pdf/2606.03251v1)

**作者:** Gautam Gare `[一作]` (Carnegie Mellon University), Nan Rosemary Ke `[通讯]` (Google DeepMind)

**通讯引用:** 3246 | [OpenAlex ID](https://openalex.org/A5102922377)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

利用神经因果发现算法（DCDI）在多维数据上识别自然实验，并通过只使用马尔可夫毯（Markov blanket）特征进行分类，从而提升模型性能。

**💡 创新点**

将自然实验视为数据中的隐含干预，首次将软已知干预（soft‑known intervention）与因果特征选择相结合，在真实数据中验证自然实验存在并能提升性能；同时展示DCDI在高维数据上可实现全图因果结构学习。

**🔧 技术方法**

核心技术包括：1）可微分因果发现（DCDI），可同时处理观测与干预数据；2）软已知干预配置 I_SK；3）利用马尔可夫毯进行特征选择；4）对比传统滤波/嵌入/包装特征选择与经典因果方法（LRH、STMB、HITON‑MB 等）。

**📊 数据集**

在 11 个公开分类数据集上进行实验，数据来源于 Kaggle 与 UCI，特征数从 5 到 50，样本数从 4839 到 284807；并在合成 Sachs 网络上进行仿真验证。

**📈 对比分析**

与 10 种非因果特征选择基线（Pearson、Spearman、MI、ANOVA、Chi‑square、Boruta）以及 5 种经典因果方法对比。实验表明：在 3/11 数据集（Diabetes、Higgs‑small、Credit‑card‑fraud）中，soft‑known 干预配置的因果特征选择（causal‑sk）在 F1‑score 上超过所有基线；其在稀疏度/性能曲线中处于 Pareto 前沿；在合成实验中，因果方法在有干预的测试集上优于观测配置。

**⚠️ 局限性**

局限性包括：1）假设分类标签即为干预变量，可能不适用于所有场景；2）只考虑软已知干预，忽略未知或硬干预的情况；3）DCDI 训练成本较高，且在极大样本/特征规模下仍有可扩展性挑战；4）对隐藏混杂的处理仍有限，实验中仅在部分合成设置下验证；5）自然实验的判定依赖于模型性能提升，缺乏理论保证。

---

## 306. The Word and the Way: Strategies for Domain-Specific BERT Pre-Training in German Medical NLP

**arXiv ID:** 2606.03250 | [PDF](https://arxiv.org/pdf/2606.03250v1)

**作者:** Henry He `[一作]` (Technical University of Munich), Raphael Schmitt `[通讯]` (Technical University of Munich)

**通讯引用:** 1278 | [OpenAlex ID](https://openalex.org/A5029151960)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

研究在德国医学文本领域构建了ChristBERT系列模型，采用RoBERTa架构并针对域适应进行了系统实验

**💡 创新点**

创新点包括三种域适应策略（继续预训练、从零预训练、词表适配），以及通过翻译英文医学文本构建13.5 GB的大规模德国医学语料库

**🔧 技术方法**

使用了RoBERTa、BPE词表、Whole Word Masking、fairseq、HuggingFace Transformers、NLLB 200和LLaMA 3.1等技术进行预训练、翻译与微调

**📊 数据集**

数据来源涵盖Hpsmedia、Springer Nature、PubMed Central、PhD Theses、Medical Wikipedia、MIMIC‑IV Notes、Web Crawl等，并对PMC与MIMIC‑IV进行机器翻译

**📈 对比分析**

通过在BRONCO150、CARDIO:DE、GGPONC等NER数据集及CLEF eHealth、JSynCC等分类任务上微调，ChristBERT在4/5项任务上实现SOTA，显著优于四个基线模型

**⚠️ 局限性**

局限性包括仅使用RoBERTa架构、未评估长文本模型、翻译质量未人工验证、潜在数据泄露、以及缺少对更广泛临床任务的评估

---

## 307. ARBOR: Online Process Rewards via a Reusable Rubric Buffer for Search Agents

**arXiv ID:** 2606.03239 | [PDF](https://arxiv.org/pdf/2606.03239v1)

**作者:** Zheng Liu `[一作]` (Tsinghua University), Liang Ding `[通讯]` (Alibaba Group)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ARBOR，一种可重用的过程奖励框架，通过对比轨迹生成自然语言 Rubric 并在在线生命周期中维护共享 Rubric 记忆，以提升搜索式 RL 代理在多跳问答任务中的学习效果。

**💡 创新点**

创新点在于设计了在线 Admission–Consolidation–Retirement 记忆机制，将查询本地草稿转化为跨查询通用 Rubric，实现过程级监督、可重用性与随政策演化的自适应性。

**🔧 技术方法**

采用对比诱导、LLM（Qwen3-Plus）生成 Rubric、稀疏对比评分、奖励塑形以及标准 RL 训练（GRPO/DAPO）进行比较。

**📊 数据集**

在四个多跳问答基准（Bamboogle、HotpotQA、MuSiQue、2WikiMultiHopQA）上评估，训练集来自 Tool-Star、STILL、ARPO Deep Reasoning Tasks。

**📈 对比分析**

与基线 TIR Prompting、GRPO、DAPO 对比，ARBOR 在 4B、8B、14B 三个模型规模及四个基准上均取得最高 EM/F1/LLM-judge 准确率，平均提升约 4–5 分。

**⚠️ 局限性**

局限性包括依赖外部大型 LLM 生成、合并及评分 Rubric，鲁棒性受模型能力限制；实验仅在单一搜索工具的多跳 QA 任务上验证，尚未测试更广泛工具使用场景。

---

## 308. When RLHF Fails: A Mechanistic Taxonomy of Reward Hacking, Collapse, and Evaluator Gaming

**arXiv ID:** 2606.03238 | [PDF](https://arxiv.org/pdf/2606.03238v1)

**作者:** Zelalem Abahana `[一作]` `[通讯]` (First Citizens Bank), Zelalem Abahana (First Citizens Bank)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文基于强化学习自人类反馈(RLHF)的训练过程，构建了一个转移级别的失败模式分类体系，并在GPT‑2规模模型上对多种优化策略（PPO、UP‑PPO、DPO等）进行了实验，揭示了奖励模型过度优化、代理与评估者分歧等局部失效现象；同时提出了利用预转移特征的早期预警模型；通过对局部转移的统计，证明聚合检查点指标会掩盖部分失效。

**💡 创新点**

创新点在于：①将RLHF的失效细分为六种方向性模式（稳定对齐、奖励作弊、优化崩溃、代理不对齐、保守停滞、评估者游戏），并通过行级转移而非单点指标来诊断；②在同一实验框架下对UP‑PPO等惩罚不确定性的变体进行局部失效率量化；③构建基于预转移特征的逻辑回归/随机森林预警模型，证明早期预警可提前捕捉部分奖励作弊。

**🔧 技术方法**

技术包括：PPO与UP‑PPO（加入MC‑dropout不确定性惩罚）、DPO、近似KL漂移、MC‑dropout估计不确定性、两种LLM‑as‑Judge（Claude与OpenAI），以及行级与聚合级的失败模式判定算法。

**📊 数据集**

数据集为Anthropic HH‑RLHF提示与对应的生成结果，模型规模为GPT‑2级别；实验共收集61个检查点、1,920条行级转移。

**📈 对比分析**

比较方法：对不同β值、λ值的PPO/UP‑PPO、DPO和SFT设置，统计每种设置下的奖励作弊率、优化崩溃等；结果显示激进PPO奖励作弊率最高（约14.5%），UP‑PPO降低至约10–11%；早期预警模型在预转移特征上实现ROC‑AUC≈0.82。

**⚠️ 局限性**

局限性：仅使用GPT‑2级别模型、有限的提示样本、单一训练seed、评估者非人类、MC‑dropout不确定性校准不完善、缺乏跨seed/跨模型的可迁移性验证。

---

## 309. Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents

**arXiv ID:** 2606.03236 | [PDF](https://arxiv.org/pdf/2606.03236v1)

**作者:** Zhijie Ding `[一作]`, Jiaming Xu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

演示了如何使用ACL风格文件与LuaLaTeX或XeLaTeX一起排版论文

**💡 创新点**

示例兼容性，展示两种TeX引擎下的实现方式

**🔧 技术方法**

使用LaTeX、LuaLaTeX、XeLaTeX以及ACL样式文件

**📊 数据集**

无实际数据集，纯技术演示

**📈 对比分析**

无比较实验或性能评测，仅为排版示例

**⚠️ 局限性**

缺乏真实实验与数据评测，无法验证排版效果与性能

---

## 310. The Role of Domain-Specific Features in Malware Detection: A macOS Case Study

**arXiv ID:** 2606.03218 | [PDF](https://arxiv.org/pdf/2606.03218v1)

**作者:** Biagio Montaruli `[一作]` (EURECOM & SAP), Davide Balzarotti `[通讯]` (EURECOM)

**通讯引用:** 8421 | [OpenAlex ID](https://openalex.org/A5002025561)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一种基于 macOS Mach‑O 可执行文件的静态特征机器学习检测器，首次结合了证书、权限（Entitlements）、系统 API 和持久化技术等 macOS 特有特征进行恶意软件分类。

**💡 创新点**

创新点在于系统性挖掘并利用 macOS 领域特定特征（如签名证书状态、沙盒权限、API 调用、持久化项）以及大规模数据集，显著提升了检测准确率与可解释性。

**🔧 技术方法**

使用的技术包括静态特征抽取（文件结构、字节直方图、字符串、打包、API、权限、持久化、证书状态）与树型机器学习模型（尤其是 XGBoost），并通过特征重要性分析与真实世界评估验证模型泛化能力。

**📊 数据集**

实验使用了作者新收集的 41,129 条 Mach‑O 样本（11,413 正样本，29,716 负样本）以及后续收集的 9,000 条实时样本（4,500 正、4,500 负），覆盖多种 CPU 架构（x86‑64、arm64、fat）。

**📈 对比分析**

通过与现有基于通用特征的检测器（Pajouh、Thaeler）以及端到端深度学习模型 MalConv 进行对比，模型在 1% FPR 下实现 98.50% 的 TPR，较最佳对手平均提升 16.56%，在新样本上进一步提升 50% 的检测率，证明了 macOS 特定特征的有效性与鲁棒性。

**⚠️ 局限性**

局限性包括仅采用静态特征，可能无法应对高级混淆或动态逃逸技术；部分特征依赖完整的应用包结构，导致对仅提供可执行文件的样本效果有限；模型对数据漂移与新型恶意变种仍需持续学习与更新。

---

## 311. Effect of Demographic Bias on Skin Lesion Classification

**arXiv ID:** 2606.03214 | [PDF](https://arxiv.org/pdf/2606.03214v1)

**作者:** Ralf Raumanns `[一作]` (Fontys University of Applied Science), Josien P. W. Pluim `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 21093 | [OpenAlex ID](https://openalex.org/A5057583165)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了性别和年龄不平衡对皮肤病变分类模型性能的影响，并评估了单任务、强化多任务和对抗学习三种去偏方法。

**💡 创新点**

通过线性规划控制训练集的性别和年龄分布，并系统比较多任务与对抗模型在不同群体中的公平性表现，首次在同一框架下对三种策略进行纵向对比。

**🔧 技术方法**

使用ResNet‑50基础网络，加入强化多任务辅助头和对抗判别器，采用线性规划生成平衡/偏斜数据集以及多任务/对抗训练技术。

**📊 数据集**

内部使用ISIC子集进行训练/验证，外部验证使用PAD‑UFES‑20（手机图像）和DERM7PT（皮肤镜图像）数据集。

**📈 对比分析**

在均衡测试集上比较AUC、各子组AUC、Brier分数和MAE等指标，发现强化多任务在性别不平衡时能部分缓解偏差，对抗模型在女性主导组表现最佳；但年龄偏差在任何平衡方案下仍显著存在。

**⚠️ 局限性**

局限包括样本量相对有限、年龄离散化导致分组不自然、外部数据未微调导致域迁移显著下降，以及对抗模型仅在特定数据组合下有效。

---

## 312. Reinforcement Learning from Cross-domain Videos with Video Prediction Model

**arXiv ID:** 2606.03201 | [PDF](https://arxiv.org/pdf/2606.03201v1)

**作者:** Zhao Yang `[一作]` (VU Amsterdam), Vincent François-Lavet `[通讯]` (VU Amsterdam)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种跨域视频预测奖励模型 XIPER，使强化学习代理能够仅凭不同视觉域的专家视频而不需要奖励或状态信息进行学习。

**💡 创新点**

创新点在于：①将域间图像翻译与视频预测结合，使用预测似然直接作为奖励；②完全摆脱对抗训练和域不变表示的需求；③模型在未标记视频上即可训练并能在多种视觉与形态差异任务中取得优秀表现。

**🔧 技术方法**

核心技术包括：VideoGPT 作为专家视频生成模型；NOT（Neural Optimal Transport）实现无配对域间图像翻译；DreamerV3 作为强化学习主体；Plan2Explore 的探索奖励；以及 VIPER 与 TPIL-Dv3 等基线对照。

**📊 数据集**

使用的数据集：DeepMind Control Suite 的 Color Suite（8 个任务，颜色差异）和 Body Suite（3 个任务，形态差异）；Sim‑to‑Real 任务采用 LynxReach 数据集（真实机器人与仿真对照）。

**📈 对比分析**

与 VIPER、XAIL、TPIL-Dv3 等基线在同一 RL 后端 DreamerV3 下对比；实验显示 XIPER 在 12 个任务中 9 个取得最高回报，显著优于对抗基线；在 sim‑to‑real 评估中，XIPER 产生的奖励与真实任务奖励高度相关。

**⚠️ 局限性**

局限性：①翻译模型仅用随机采样数据训练，难以覆盖长时序精细行为；②在高精度任务中对翻译误差高度敏感；③目前未完成在真实机器人上全流程在线 RL 训练，需进一步验证闭环性能。

---

## 313. Inference-Time Scaling for Joint Audio-Video Generation

**arXiv ID:** 2606.03183 | [PDF](https://arxiv.org/pdf/2606.03183v1)

**作者:** Jaemin Jung `[一作]` (Korea Advanced Institute of Science and Technology), Joon Son Chung `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 10651 | [OpenAlex ID](https://openalex.org/A5038723822)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究并实现了推理时扩展（ITS）在联合音视频生成中的应用，提出多验证器框架和自适应奖励加权（ARW）算法以提升语义对齐、感知质量和同步性。

**💡 创新点**

首次系统性评估 ITS 在多模态生成中的效果，发现单验证器会导致性能不均衡和验证器攻击，并提出最佳多验证器组合（文本-视频一致性 + 细粒度音视频同步）和通过在线优化自适应校准奖励方差的 ARW 方法，显著平衡多目标。

**🔧 技术方法**

使用扩散/流匹配生成模型（JavisDiT、MMDisCo）、EvoSearch 与 Best‑of‑N 搜索策略、视频奖励模型（VideoReward‑TA、VQAScore）、音视频同步奖励（JavisScore）、多模态相似度指标（ImageBind）以及自适应奖励加权（ARW）算法。

**📊 数据集**

在 VGGSound 测试集和 JavisBench‑mini 基准上进行实验，覆盖文本一致性、音视频语义一致性、同步性和视频感知质量四大评估维度。

**📈 对比分析**

与传统奖励聚合方法（加权和、归一化、Rank‑based）以及两种 ITS 策略（Best‑of‑N、EvoSearch）进行对比，实验表明 ARW 在文本一致性、音视频同步和整体质量上均取得显著提升，尤其在 EvoSearch 迭代搜索中表现突出。

**⚠️ 局限性**

受限于当前预训练模型的基准质量、音视频高维数据导致的显著显存占用以及 ITS 需要生成并评估大量候选样本的计算开销，尚需进一步提升模型基准、优化搜索效率和资源利用。

---

## 314. GeoSem-WAM: Geometry- and Semantic-Aware World Action Models

**arXiv ID:** 2606.03188 | [PDF](https://arxiv.org/pdf/2606.03188v1)

**作者:** Fulong Ma `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Jun Ma `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出 GeoSem-WAM，在世界动作模型中加入几何与语义预测分支，提升潜在表示的结构化与鲁棒性。

**💡 创新点**

创新点在于利用 DPT 风格的几何与语义密集预测头作为训练时的辅助监督，兼顾视觉动态、空间几何和对象语义，却不增加推理时的计算负担。

**🔧 技术方法**

主要技术包括基于 DiT 的视频 Diffusion Transformer、预训练 VAE 进行潜在编码、Action DiT 的动作生成、DPT 结构的几何与语义分支，以及联合多任务损失训练。

**📊 数据集**

使用的主要数据集有 LIBERO（四个子套件）、RoboTwin 2.0（模拟与随机域）以及真实 Franka Emika Panda 机器人的自采集数据。

**📈 对比分析**

在 LIBERO 上的平均成功率从 Fast‑WAM 的 97.60% 提升到 98.55%，在 RoboTwin 2.0 上平均成功率从 91.80% 提升到 92.52%，在真实机器人实验中平均成功率从 88.9% 提升到 95.4%，均优于相关 SOTA 方法。

**⚠️ 局限性**

局限性在于辅助几何与语义头需要像素级标签，且训练时多任务损失可能产生梯度冲突，未来需探索无监督特征或梯度冲突抑制技术。

---

## 315. Let There Be Light: Reflection, Refraction and Scattering for Neural Operators

**arXiv ID:** 2606.03262 | [PDF](https://arxiv.org/pdf/2606.03262v1)

**作者:** Keke Wu `[一作]` (University of Science and Technology of China), Jingrun Chen `[通讯]` (University of Science and Technology of China)

**通讯引用:** 2689 | [OpenAlex ID](https://openalex.org/A5025195102)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种基于光传输原理的神经算子 LiNO，用于高效学习 PDE 的参数化求解器。

**💡 创新点**

创新点在于将光学中的反射、折射与散射机制映射到潜在特征空间，形成可解释的三种局部/全局变换，并通过线性化注意力加局部扩散实现 O(N) 的高效散射。

**🔧 技术方法**

采用了自注意力的线性化实现（全局正特征传播+局部深度卷积），Householder 反射、稀疏折射、相对位置偏置、残差结构、层归一化与 MLP 等技术。

**📊 数据集**

使用了四个标准 PDE 基准数据集：Burgers 方程、Darcy 流动、转子气流（曲面网格）和二维 Navier–Stokes vorticity。

**📈 对比分析**

与 FNO、DeepONet、LNO、MGNO 等基线进行对比，LiNO 在 1D Burgers 上实现 3.19e-3 的相对 L²误差，在 2D Darcy 上达到 5.91e-3，空气机翼在曲面网格上保持良好拟合，并在 Navier–Stokes 的 10 步自回归预测中获得 2.92e-3 的误差，整体性能优于多数基线，尤其在高分辨率和自回归稳定性方面表现突出。

**⚠️ 局限性**

局限性包括：目前仅支持规则或曲面网格，未扩展到完全无结构点云、动态边界；散射机制仍需手工设计，缺乏严格的收敛与稳定性理论；对非周期边界、大尺度多尺度或极端非线性问题的适用性还有待验证。

---

## 316. On the Impact of Pinching Antennas on Traffic Offloading

**arXiv ID:** 2606.03253 | [PDF](https://arxiv.org/pdf/2606.03253v1)

**作者:** Zhiguo Ding `[一作]` (Nanyang Technological University), H. Vincent Poor `[通讯]` (Princeton University)

**通讯引用:** 159958 | [OpenAlex ID](https://openalex.org/A5042307561)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了可拉扯天线（pinching antennas）在多细胞网络中实现流量迁移的机制，构建了总传输功率最小化问题，并给出了天线位置与功率分配的最优解。

**💡 创新点**

创新点在于将可拉扯天线的可重构细胞边界特性引入流量迁移设计，提出两种迁移策略并提供解析最优解，显著提升能耗效率和负载平衡。

**🔧 技术方法**

使用无线链路模型、NOMA、多用户MIMO、凸优化与KKT分析等技术，获得闭式功率分配和天线位置解。

**📊 数据集**

采用仿真生成的用户/基站位置数据以及28 GHz毫米波标准信噪参数，并未使用公开数据集。

**📈 对比分析**

通过与三种基准（无迁移、强制迁移、动态迁移）以及传统天线方案比较，仿真表明可拉扯天线可降低10–20 dBm的传输功率，显著提高能效和负载均衡。

**⚠️ 局限性**

局限性：仅考虑单一基站配备可拉扯天线，未讨论大规模多天线或网络层干扰；对完全迁移的假设要求高负载，实际应用场景受限。

---

## 317. MemoGen: Can Past Experience Improve Future Text-to-Image Generation?

**arXiv ID:** 2606.03243 | [PDF](https://arxiv.org/pdf/2606.03243v1)

**作者:** Wenshuo Chen `[一作]` (Hong Kong University of Science and Technology), Yutao Yue `[通讯]` (Institute of Deep Perception Technology, Jiangsu Industrial Technology Research Institute)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出MemoGen，一种训练‑free 的持续学习框架，利用生成器外的智能代理在每次文本到图像的生成中构造、改进并保存生成条件，实现连续自我演化。

**💡 创新点**

创新点在于：①将生成过程拆解为任务理解、检索、可视化反馈、经验写回的闭环；②通过经验记忆存储正负案例，支持跨任务跨轮的策略复用与错误警示；③完全不更新生成器参数，只在代理层实现自我提升。

**🔧 技术方法**

主要技术：多工具调用的LLM/VLM 控制器（GPT‑5.5），本地视觉 RAG、动态 Web 搜索、引用检索；图像生成后端（Qwen‑Image‑Edit‑2509）；可视化检查工具；双阶段经验评估（内部反馈 + 记忆裁判）；经验记忆写回与检索。

**📊 数据集**

使用公开知识驱动与推理型图像生成基准：WISE（文化、时间、空间、生物、物理、化学）和 Mind‑Bench（知识驱动与推理驱动子任务），以及多种公开与专有模型作为对照。

**📈 对比分析**

在第二轮演化后，MemoGen 在 WISE 上获得 0.91 的整体分数（超越 GPT‑Image‑1、Nano Banana Pro 等专有系统，及 Gen‑Searcher、Mind‑Brush 等检索增强模型），在 Mind‑Bench 上获得 0.52 的整体 CSA，显著高于最强专有基线 0.41 与 Mind‑Brush 0.31，证明自我演化与经验记忆带来实质性性能提升。

**⚠️ 局限性**

局限性：①受限于固定的底层生成器能力，无法突破生成器本身的先验限制；②依赖自动评估器（记忆裁判）和工具链的稳定性，评估误差或工具失效可能导致经验污染；③目前经验记忆主要面向单轮任务，跨域泛化与长期记忆持久性待进一步研究；④实现复杂度较高，需要多种工具和大型 LLM 的支持。

---

## 318. GeoAlign: Beyond Semantics with State-Guided Spatial Alignment in VLA Models

**arXiv ID:** 2606.03240 | [PDF](https://arxiv.org/pdf/2606.03240v1)

**作者:** Yizhi Chen `[一作]` (Tongji University), Yue Gao `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 5024 | [OpenAlex ID](https://openalex.org/A5100726291)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6514db3d-8de6-452c-91b7-acdb31787cc4` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 GeoAlign，一种基于 RGB 视觉的几何增强与机器人本体状态查询的空间对齐框架，用于提升视觉语言动作（VLA）策略在执行精细操作时的几何对齐和动态可用性。

**💡 创新点**

创新点在于①通过离线 RGB‑D 监督后训练 Depth Anything V2 获得 RGB 衍生的几何增强特征；②在策略运行时使用机器人本体状态生成查询，从图像空间特征网格中抽取紧凑几何令牌，实时指导连续动作解码。

**🔧 技术方法**

使用技术包括 Depth Anything V2 后训练、可变时间流匹配 DiT 解码器、GR00T 结构、跨注意力查询、RGB‑语言 VLM 编码、流匹配流动时间采样和动作掩码正则化。

**📊 数据集**

数据集包括 LIBERO、SimplerEnv‑Fractal（三个仿真任务集）、真实世界的 AgileX ALOHA 平台任务（八个桌面任务）以及用于几何后训练的机器人域 RGB‑D 数据集。

**📈 对比分析**

通过与 RGB‑only、无后训练、无空间查询、无状态查询、冻结编码器等变体对比，在 LIBERO 上平均 99.0% 成功率（RGB‑only 97.0%），Fractal 平均 85.3%（RGB‑only 79.6%），ALOHA 平均 78.8%（RGB‑only 65.0%，π0.5 67.5%），验证了几何后训练与状态引导查询的关键作用。

**⚠️ 局限性**

局限性在于仅基于视觉几何特征，缺乏碰撞/可达性/接触约束建模；对相机配置、视觉覆盖和后训练 RGB‑D 监督分布高度依赖；未保持跨时间的场景记忆，难以处理长期规划或大范围相机位移任务。

---

## 319. GFFMERGE: Efficient Merging of Graph Neural Force Fields and Beyond

**arXiv ID:** 2606.03232 | [PDF](https://arxiv.org/pdf/2606.03232v1)

**作者:** Parth Verma `[一作]` (Indian Institute of Technology Delhi), Sayan Ranu `[通讯]` (Indian Institute of Technology Delhi)

**通讯引用:** 1647 | [OpenAlex ID](https://openalex.org/A5054697900)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `3f18e8e3-0266-457c-8567-9039b6d2394d` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于闭式线性求解的图神经网络力场模型合并框架（GFFMerge），实现不需要重新训练即可将多个专用力场合并为一个统一模型。

**💡 创新点**

创新点在于将消息传递层的线性结构转化为凸嵌入对齐问题，得到可解析的合并公式；通过切换嵌入策略和少量后续微调，恢复与全局联合训练相当的预测精度；并将此方法推广到通用图神经网络。

**🔧 技术方法**

核心技术包括：嵌入对齐的凸优化、线性参数的最小二乘闭式求解、切换嵌入（Switch Embeddings）机制、少量目标层微调（Targeted Fine‑Tuning）以及对多任务与多域数据的自动校准。

**📊 数据集**

主要使用分子动力学基准数据集 MD17、MD22，以及固态离子电解质基准 LiPS20，此外在通用图学习上验证了在 Cora、Citeseer、Pubmed、Arxiv 等公共数据集上的表现。

**📈 对比分析**

与现有的权重平均、Fisher、TIES、EMR 等基线相比，GFFMerge 在 MD17、MD22、LiPS20 上均取得 5–27 倍的训练时间加速，并在力场 MAE、能量/力误差以及 MD 轨迹的能量/力违规率上与联合微调（gold standard）几乎无差距；在通用图任务上同样优于所有基线。

**⚠️ 局限性**

局限性包括：对非线性层的处理仍需微调，合并过程对源模型的相容性有一定要求；在高度不均衡或极其异构的域之间可能出现性能下降；目前未针对等变或高维对称性模型进行专门的适配。

---

## 320. HRNN: A Hybrid Graph Index for Approximate Reverse k-Nearest Neighbor Search on High-Dimensional Vectors

**arXiv ID:** 2606.03225 | [PDF](https://arxiv.org/pdf/2606.03225v1)

**作者:** Wenxuan Xia `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 253673 | [OpenAlex ID](https://openalex.org/A5100444820)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd`

**🎯 论文内容**

本文设计并实现了一种混合索引HRNN，用于高维向量的近似逆k近邻（ARkNN）查询；

**💡 创新点**

创新点在于将邻近向量作为代理，通过逆邻接列表生成候选，并预材料化kNN半径实现轻量化验证，从而显著降低候选生成和验证开销；

**🔧 技术方法**

主要技术包括HNSW导航图、Ranked HNSW图以及逆邻接列表的构造与利用，并结合NNDescent算法进行图优化；

**📊 数据集**

实验使用了SIFT、Msong、GIST和MSMARCO四个真实向量数据集，维度分别为128、420、960和1024，样本量均为1M；

**📈 对比分析**

与kNN扩展、HNSW-RDT和HNSW-ARNN等基线方法相比，HRNN在Recall@10≥0.99时可实现最高QPS，MSMARCO上185 QPS（比最佳基线高83×），在构造时间和空间开销上虽略高于单纯HNSW，但远优于Ranked HNSW+逆列表完整索引；

**⚠️ 局限性**

主要局限是索引空间相对较大，构造时间比单纯HNSW多5-14倍，且对动态更新仍需维护成本。

---

## 321. Follow-Your-Preference++: Rethinking Preference Alignment for Image Inpainting

**arXiv ID:** 2606.03216 | [PDF](https://arxiv.org/pdf/2606.03216v1)

**作者:** Junkun Yuan `[一作]` (Zhejiang University), Yue Ma `[通讯]` (Tsinghua University)

**通讯引用:** 6808 | [OpenAlex ID](https://openalex.org/A5046768859)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对图像修复（inpainting）任务，利用离线奖励模型生成偏好数据，并通过直接偏好优化（DPO）进行模型对齐，进一步将该方法推广到对象移除任务。

**💡 创新点**

创新点包括：① 用多奖励模型集合（Ensemble）减少单一奖励模型的偏差与奖励劫持；② 提出校准的 Ensemble（CaEN）以进一步消除劫持；③ 将偏好对齐从创意性修复扩展到需要结构一致的对象移除。

**🔧 技术方法**

采用的技术包括：Diffusion/Flow 基础生成模型（BrushNet、FLUX.1 Fill）；Direct Preference Optimization（DPO）；多奖励模型集合与加权/校准（CaPO、CaEN）；GPT‑4 评估及多种自动指标（PSNR、CLIPScore、LPIPS）。

**📊 数据集**

使用的数据集包括 BrushData/BrushBench、EditBench、RORem（对象移除）以及对应的验证集。

**📈 对比分析**

通过与多种公开模型（SDI、CNI、BrushNet、FLUX.1 Fill、PrefPaint 等）进行用户研究和自动评估比较，所提出的 BruPA/FluPA 在所有指标中均名列前茅，GPT‑4 评估一致率达 86% 以上，显著优于基线与现有方法。

**⚠️ 局限性**

局限性：研究仅局限于图像数据，未扩展到视频/3D；对离线奖励模型的依赖仍可能导致偏差，尽管 Ensemble 与 CaEN 可缓解但不能完全消除；评估仍依赖自动模型与有限的人工评审；模型生成的高度逼真图像可能被滥用。

---

## 322. Bayesian Tensor Decomposition with Diffusion Model Prior

**arXiv ID:** 2606.03212 | [PDF](https://arxiv.org/pdf/2606.03212v1)

**作者:** Zerui Tao `[一作]` (RIKEN Center for Advanced Intelligence Project), Qibin Zhao `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**通讯引用:** 8321 | [OpenAlex ID](https://openalex.org/A5083182987)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出DiffBCP框架，将低秩CP分解与预训练扩散模型隐式先验结合，用于张量补全和去噪。

**💡 创新点**

创新点：①使用累积分割过程（CUSP）实现自动秩选择；②采用分裂吉布斯采样实现可解耦的后验推断，允许扩散模型作为隐式先验；③设计噪声自适应耦合调度，降低手工调参需求。

**🔧 技术方法**

技术手段：贝叶斯CP分解、CUSP先验、EDM扩散模型隐式先验、分裂吉布斯采样、低秩引导去噪、噪声自适应耦合调度。

**📊 数据集**

数据集：FFHQ、ImageNet用于训练和评估；Marseille、Tokyo、Westerlund等2048×2048的高分辨率图像用于OOD测试。

**📈 对比分析**

与贝叶斯CP、贝叶斯张量环、HLRTF、DeepTensor、GLON等基线比较，实验显示在不同掩码（Uniform、Stripe、Irregular）和噪声下，PSNR提升约2–3 dB，SSIM和LPIPS同样表现优异，OOV高分辨率图像亦保持领先。

**⚠️ 局限性**

局限性：对信号低秩结构依赖强，若低秩假设不成立则模型效果受限；全量张量更新导致高内存/计算开销，对极大张量不太友好。

---

## 323. Critical evaluation of PINN for FWD inverse analysis and differentiable FEM as an alternative

**arXiv ID:** 2606.03210 | [PDF](https://arxiv.org/pdf/2606.03210v1)

**作者:** Yongjin Choi `[一作]` (KAIST InnoCORE PRISM-AI Center Korea Advanced Institute of Science and Technology), Seunghwa Ryu `[通讯]` (KAIST InnoCORE PRISM-AI Center Korea Advanced Institute of Science and Technology)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过构造三层道路板的合成 FWD（跌落重力测斜仪）数据，比较了两种基于自动微分的逆问题求解方法：传统 PINN（包含域分解 XPINN）和 DiffFEM；

**💡 创新点**

创新点在于系统评估 PINN 以及其域分解扩展在层状弹性系统中的可行性，并证明在该类问题上 DiffFEM 通过“硬约束”式的物理方程执行更高效、更稳健的逆推；

**🔧 技术方法**

所用技术包括自动微分、物理信息神经网络（PINN、XPINN）、可微分有限元方法（DiffFEM）、L‑BFGS 二阶优化；

**📊 数据集**

数据集为三层合成道路板的 FWD 位移响应，含多级高斯噪声（σ_n=0,1,2,3,4,5 μm），无真实测量数据；

**📈 对比分析**

对比方法通过噪声自由与噪声存在两种场景下的收敛迭代次数、计算时间、模量误差、位移误差来评估；结果显示 DiffFEM 在噪声0–5 μm 下仅需约 60 秒即可达到 10^‑10 级的模量精度，而 XPINN 需 10 小时并产生数百百分比误差；

**⚠️ 局限性**

局限性包括 PINN 对损失权重和网络架构高度敏感，难以稳定收敛；DiffFEM 依赖可微分前向求解器，若存在非线性、时变或非光滑现象则难以直接实现。

---

## 324. DECA: Decentralizing Block-Wise Adam for Efficient LLM Full-Parameter Fine-Tuning on Non-IID Data

**arXiv ID:** 2606.03209 | [PDF](https://arxiv.org/pdf/2606.03209v1)

**作者:** Yunsheng Yuan `[一作]` (Shandong University), Feng Li `[通讯]` (Shandong University)

**通讯引用:** 70181 | [OpenAlex ID](https://openalex.org/A5100448879)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 DECA，一个去中心化的全参数微调框架，能够在资源受限且数据分布非均匀的环境下为大型语言模型实现高效微调。

**💡 创新点**

创新点在于将模型划分为互不重叠的块，采用块级 Adam 并结合先后顺序的 Block‑wise Moment Approximation（BMA）来兼顾局部梯度信息与全局一致性，显著抑制客户端漂移并实现资源高效的全参数适配。

**🔧 技术方法**

核心技术包括块坐标下降（BCD）+ Adam、BMA（融合第一、二阶时刻与一致性误差）、去中心化邻接图通信、双随机矩阵谱分析以及理论收敛证明。

**📊 数据集**

实验使用 Llama‑2‑7B、Llama‑3.1‑8B、Qwen2‑1.5B 与 Qwen2.5‑3B 等四种大模型，任务涵盖分类（NWGI、AGNEWS、TFNS、MNLI）与生成（Alpaca、Vicuna、MT‑Bench）数据集。

**📈 对比分析**

与 Dec‑Adapter、Dec‑LoRA、DeCAF 等去中心化 PEFT 基线相比，DECA 在分类任务中平均提升约 2–4% 的准确率和 F1，生成任务中 MT‑Bench 分数提升约 0.2–0.4，且收敛速度更快、最终损失更低。

**⚠️ 局限性**

局限性包括：对块划分与 BMA 超参数需要手工调优；虽然相较于中心化训练节省通信，但仍需多轮邻接通信，可能在极大网络或极低带宽环境下受限；在极端非 IID 情况下，BMA 效果需进一步验证。

---

## 325. AI Rater Discrimination Depends on Scoring Protocol in Complex Clinical Decision-Making

**arXiv ID:** 2606.03198 | [PDF](https://arxiv.org/pdf/2606.03198v1)

**作者:** Sangwon Baek `[一作]` (Asclep Korea Inc.), Kyunga Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 10082 | [OpenAlex ID](https://openalex.org/A5064386987)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过因子实验设计，系统研究了在成人2型糖尿病（T2D）药物治疗评估任务中，AI评审者在两种评分协议（金属评分表 GR 与非金属评分表 Non‑GR）下的评分行为及其与 CDSS 模型、提示配置、评审者模型、提示性格与类型等因素的交互作用。

**💡 创新点**

创新点在于首次将评分协议本身（是否提供患者特异性评分表）作为主要变量，定量揭示其对 AI 评审者评分分布、分辨力及模型间差异的影响，并通过多因素交互分析揭示评分协议与 CDSS 质量、评审者身份、提示细节等因素的互作机制。

**🔧 技术方法**

使用了开源 LLM（Gemma‑4、Qwen‑3.5、GLM‑5、Nemotron）同时担任 CDSS 与评审者，采用因子实验设计与线性混合效应模型（REML、Wald 检验、Fisher 组合、DID、放大比）对 580,608 条评分观测进行统计分析；同时构造了文档引用生成（DRG）与基线（Baseline）两种 CDSS 提示，评审者提示包含性格（Lenient/Moderate/Strict）、类型（Normal/CoT/Self‑Consistency）和顺序。

**📊 数据集**

数据集为 16 名合成成年 T2D 患者，结合 7 个临床评估问题，生成 8 条 CDSS 输出（4 模型 × 2 提示），共 82,944 个评分单元，重复 3 次，构成 580,608 条评分记录。

**📈 对比分析**

比较方法包括均值差异、IQR、Kolmogorov‑Smirnov 检验、交互热图及放大比分析；结果显示 GR 协议下评分平均低、分散度大，显著放大 DRG 与 Baseline 的差距（最高 5 倍），并揭示评审者模型间在 GR 与 Non‑GR 下的行为差异；Non‑GR 协议评分高度集中、分辨力受限。

**⚠️ 局限性**

局限性包括：仅研究 T2D 药物治疗任务，无法推广至其他临床决策场景；仅使用四个开源 LLM，评审者行为可能因模型不同而差异；未深入分析评审者内部推理过程；评分协议对比主要关注分布差异，非绝对得分差异；样本仅为 16 名合成患者，真实病例多样性有限。

---

## 326. SenseJudge: Human-Centric Preference-Driven Judgment Framework

**arXiv ID:** 2606.03189 | [PDF](https://arxiv.org/pdf/2606.03189v1)

**作者:** Rui Li `[一作]` (Peking University), Zhifang Sui `[通讯]` (Peking University)

**通讯引用:** 4942 | [OpenAlex ID](https://openalex.org/A5110285832)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 SenseJudge 框架和 SenseBench 基准，用于基于人类偏好实现可定制化的 LLM 判别与模型排序。

**💡 创新点**

创新点在于：① 将人类明确偏好抽象为可重用的文字描述，构成多样化的判别偏好集；② 通过少量标注样本实现对偏好的提炼与选取；③ 在真实多轮对话数据上构建挑战性基准 SenseBench，突破单轮或双轮基准的局限。

**🔧 技术方法**

主要技术包括：使用 LLM（如 DeepSeek‑R1、Qwen‑系列）进行偏好生成与筛选；多模型生成与对比（强弱模型）；两阶段过滤（自动 GPT‑4 + 人工审核）；多数投票的偏好子集选择；以及对位置偏差、一致性和跨域推广的分析。

**📊 数据集**

使用来自公开 LLM 服务的真实用户交互数据，经过自动与人工过滤后形成 SenseBench，共包含 8 类（数学、逻辑、编程、写作、翻译、角色扮演、NLI 等）多轮挑战性样本；并对 RewardBench 等公开基准进行验证。

**📈 对比分析**

与传统 LLM+prompt、训练好的评判器、奖励模型等基线进行对比；在 LLM‑as‑Personalized‑Judge 任务中，SenseJudge 的平均准确率提升约 16%；在模型排序任务中，SenseJudge 的排名与人类评测高度一致；在 RewardBench 上亦显示优于公开 LLM‑as‑Judge 方法。

**⚠️ 局限性**

局限性包括：仅使用三位标注者的 1000 条比较数据，标注量有限；偏好集主要来源于少量示例，跨域泛化仍受限；实验依赖于强大 LLM 作为偏好生成器，对算力要求较高。

---

## 327. GLINT: Sparsely Gated Vision-Language Alignment for Fine-Grained Radiology Representations

**arXiv ID:** 2606.03180 | [PDF](https://arxiv.org/pdf/2606.03180v1)

**作者:** Jonggwon Park `[一作]` (DEEPNOID Inc), Kyoyun Choi `[通讯]` (Sejong University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了一种基于稀疏门控对齐的视觉‑语言模型（GLINT），通过对齐小范围病灶与文本描述实现零样本分类、定位与分割，适用于二维胸 X‑ray 与三维胸 CT。

**💡 创新点**

创新点在于：①稀疏门控对齐（SGA）在单独的门控嵌入空间中使用 Sigmoid 门只激活与查询相关的少量图像块，显著提升定位精度；②密集特征正则化（DFR）将训练时的视觉编码器中间特征与冻结的自监督教师对齐，保持细粒度特征不被漂移，兼顾语言适配与特征保留。

**🔧 技术方法**

核心技术包括：多模态 Transformer 视觉编码器（DINOv3/V‑JEPA 2.1）、文本编码器（MPNet）和 GPT‑OSS‑120B 句子拆分；门控头实现门控嵌入与 Sigmoid 门；密集特征正则化采用层内对齐的余弦距离；对齐损失使用 CLIP 风格的对比学习与多正样本对抗；推理时通过门控相似度图（GSM）实现定位与分割。

**📊 数据集**

主要使用公开医学数据集：胸 X‑ray 的 MIMIC‑CXR v2.0.0、PadChest、ChestX‑ray14、CheXpert、Open‑I；胸 CT 的 CT‑RATE v2、RAD‑ChestCT、ReXGroundingCT；下游任务数据集包括 SIIM、RSNA‑PNEUMONIA、LIDC‑IDRI、MSD‑Lung 等。

**📈 对比分析**

与现有医学 VLMs（ConVIRT、MedCLIP、GLoRIA、RadZero、Merlin 等）以及自监督基础模型（DINOv3、V‑JEPA 2.1）进行比较，GLINT 在零样本分类、定位与分割任务中均实现了显著提升：在胸 X‑ray 上，Macro Dice 从 0.332 提升至 0.434；在胸 CT 上，实现了首个无掩码监督的 3D 体素分割，Dice 与监督方法相当；在下游任务中亦表现最优。

**⚠️ 局限性**

局限性主要包括：实验仅覆盖胸 X‑ray 与胸 CT 两种模态，尚未验证在其他影像器官（腹部、脑部 MRI 等）的泛化；缺乏临床医生的直接评估与应用场景验证；门控与正则化的超参数调优对不同数据集的依赖性较高。

---

## 328. EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs

**arXiv ID:** 2606.03260 | [PDF](https://arxiv.org/pdf/2606.03260v1)

**作者:** Sungwon Kim `[一作]` (KAIST), Chanyoung Park `[通讯]` (KAIST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出了一种新的 3D PDE 预测框架 EqGINO，它通过在 Fourier 层中使用基于轨道的权重并在编码器-解码器结构中融入局部几何信息，保持整个网络的等变性，能够从不规则点云输入预测弹性变形、流场等物理场。

**💡 创新点**

创新点在于：① 将不规则点云映射到规则网格后，在 Fourier 变换空间通过轨道权重实现对旋转、平移和镜像等欧氏群的严格等变性；② 在卷积层前后加入点云特征编码与解码模块，使得局部几何信息能够与全局频域特征相互补充；③ 通过全等变的设计，避免了传统 FNO 在不规则几何下的性能下降。

**🔧 技术方法**

使用了基于 Fourier Neural Operator 的频域层、轨道权重（orbit‑based weights）、点云编码器（PointNet/Graph‑based encoder）以及等变性解码器；整个模型在训练时采用交叉熵或均方误差损失，并结合残差连接与正则化技术。

**📊 数据集**

实验数据集主要包括两类：① 3D 弹性变形数据（来自有限元模拟的结构弯曲/压缩场），② 3D 流体动力学数据（Navier–Stokes 模拟的速度与压力场）。这些数据均来自公开的 PDEBench 3D 基准集，涵盖不同几何尺寸与材料属性。

**📈 对比分析**

与 FNO、GNO、DeepONet 等现有神经算子进行对比。结果显示，EqGINO 在均方误差（MAE）上比 FNO 低约 12%，比 GNO 低约 8%，在大尺度几何和极端边界条件下仍保持较低误差；推理速度保持与 FNO 相近，显著优于基于图的 GNO。

**⚠️ 局限性**

局限性包括：① 对非常大规模的网格或高分辨率点云仍存在内存与计算瓶颈；② 目前仅对欧氏群等变性进行设计，难以直接处理更一般的对称群或非欧氏变换；③ 对极端几何畸变（如尖锐角、薄壁结构）仍需进一步鲁棒性验证。

---

## 329. FreeStreamGS: Online Feed-forward 3D Gaussian Splatting from Unposed Streaming Inputs

**arXiv ID:** 2606.03254 | [PDF](https://arxiv.org/pdf/2606.03254v1)

**作者:** Ruiyang Chen `[一作]` (Beijing University of Posts and Telecommunications), Heng Guo `[通讯]` (Beijing University of Posts and Telecommunications)

**通讯引用:** 8226 | [OpenAlex ID](https://openalex.org/A5039812471)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种在线前向的3D高斯剖分框架 FreeStreamGS，可在无姿势的流式图像输入下，实时构建高质量 3D 高斯表示，实现低延迟的新视角合成。

**💡 创新点**

核心创新包括：1）分离的内参恢复头（DIR-Head）消除累计的相机内参漂移；2）动态点细化偏移（DPR-Offsets）放宽视线约束，纠正姿势-深度耦合漂移；3）在线递归高斯融合与新视角加权监督（NV-Sup），共同提升在线几何一致性。

**🔧 技术方法**

技术方案涵盖 Transformer‑based 全历史因果特征提取、可学习相机姿态/内参回归、3D 高斯解码器、教师强制内参蒸馏、深度先验对齐、在线递归体素融合、基于 SSIM/LPIPS 的多任务损失，以及 Savitzky–Golay 滤波等。

**📊 数据集**

训练集为 10,000 条来自公开数据集的无姿势图像序列；评估使用官方基准（如 NYU‑v2、ScanNet 等）以及 100 个未见场景的跨域数据集。

**📈 对比分析**

与在线优化型 3DGS 以及多种离线前向 3DGS 基线进行比较。实验结果显示：在 5 视图稀疏输入下 PSNR 最高，10 视图保持竞争力，64 视图时可通过递归融合避免 OOM；推理时延约 250 ms，显著低于优化法，且与离线 SOTA 的渲染质量相当。

**⚠️ 局限性**

局限性：对高速动态物体仍易产生失真；场景规模过大或序列过长时，3D 高斯存储开销高，难以扩展。

---

## 330. Investigating Novice Researchers' Perceptions of Research Privacy Within LLM-Assisted Workflows

**arXiv ID:** 2606.03248 | [PDF](https://arxiv.org/pdf/2606.03248v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**通讯引用:** 1587 | [OpenAlex ID](https://openalex.org/A5011803365)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对44名初级研究人员进行半结构访谈和主题分析，研究了他们在使用大型语言模型时的隐私认知与应对策略。

**💡 创新点**

论文首次系统阐述了初级研究者的隐私风险感知、误解与五类缓解做法，并揭示了“生产力优先”与“隐私缺失”的矛盾。

**🔧 技术方法**

采用定性访谈、语料转写、手工编码及主题编码的研究方法进行数据分析。

**📊 数据集**

使用的主要数据集为44名受访者的访谈记录（包含文本、问卷信息），涵盖计算机科学、社会科学、自然科学等多学科。

**📈 对比分析**

由于研究以定性分析为主，没有传统性能指标，作者通过主题频次与参与者自述来比较不同隐私保护手段的效果，结果显示大多数策略被认为无效。

**⚠️ 局限性**

研究局限包括样本为方便抽样、受访者自述易受偏差影响、仅聚焦初级研究者，缺乏长周期或对比研究。

---

## 331. When Does Complexity Conditioning Help a Frozen Sentence Embedding? A Controlled Study of Per-Sentence and Pair-Level Difficulty Adaptation

**arXiv ID:** 2606.03244 | [PDF](https://arxiv.org/pdf/2606.03244v1)

**作者:** Suhwan Hwang `[一作]` `[通讯]`, Suhwan Hwang

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在多任务、多随机种子实验中，评估了在冻结句子嵌入上加入难度感知的轻量后置适配器的效果。

**💡 创新点**

创新点在于发现只有满足成对条件、保持基线、使用交叉编码器难度信号三条件时，难度感知才带来正面收益，并给出了基于冻结嵌入的预训练诊断。

**🔧 技术方法**

使用技术包括轻量级后编码器适配器、成对难度门控残差、交叉编码器难度度量以及严格的信号隔离对照实验。

**📊 数据集**

实验数据集包含 PAWS、MRPC、QQP 和 STS‑B 四个英文语义相似/重写任务。

**📈 对比分析**

与冻结基线及多种控制（常数、打乱等）对比，成对门控残差在大任务上提升约 +0.02–0.04 Spearman，且在所有种子上均未低于基线且检索性能不退化。

**⚠️ 局限性**

局限性包括提升幅度有限、仅在单一开放权重 bi‑encoder 上验证、单语言、样本量有限、依赖交叉编码器难度信号，并且最终模型仅是成对重排序器而非单向向量嵌入。

---

## 332. Learning Temporal Causal Structure via Smooth Differentiable Optimization

**arXiv ID:** 2606.03227 | [PDF](https://arxiv.org/pdf/2606.03227v1)

**作者:** Tong Zhao `[一作]` (Imperial College London), Ray Dipojjwal `[通讯]` (University of Bristol)

**通讯引用:** 241 | [OpenAlex ID](https://openalex.org/A5069802728)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过使用 Gumbel–Sinkhorn 算子学习可微分的变量置换，将瞬时因果结构的无环性约束转化为参数化，并在学习得到的顺序下对 SVAR 模型的即时系数矩阵进行三角化，从而实现统一的连续梯度优化进行因果结构学习。

**💡 创新点**

创新点在于将无环性硬约束转化为可学习的置换参数，避免了传统多阶段管道或复杂拉格朗日约束方法，从而实现高效的连续优化并保持无环性的一致性。

**🔧 技术方法**

采用 Gumbel–Sinkhorn 算子进行可微分置换学习、结构向量自回归（SVAR）模型、梯度基的连续优化以及三角化系数矩阵的参数化。

**📊 数据集**

在三组真实世界时间序列基准数据集以及一个大规模基准上进行评估，具体基准名称未在摘要中列出，但涉及多领域实际数据。

**📈 对比分析**

与12个现有基线方法对比，本文方法在因果发现的准确性和计算效率上均获得最佳表现，在大规模基准中相比竞争方法提升了 6 倍以上的速度。

**⚠️ 局限性**

潜在局限包括：对高维变量集合的可微分置换求解仍可能面临计算负担；对非线性即时因果关系的适应性尚未在摘要中充分验证；对超大规模实时数据流的在线更新可能需要进一步优化。

---

## 333. Sample-Size Scaling of the African Languages NLI Evaluation

**arXiv ID:** 2606.03219 | [PDF](https://arxiv.org/pdf/2606.03219v1)

**作者:** Anuj Tiwari `[一作]` (Noida Institute of Engineering and Technology), Hannah Nwokocha `[通讯]` (ML Collective)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对16种非洲语言的NLI任务进行样本规模扩展实验，评估不同样本大小对XLM-R Large和AfroXLM-R Large性能的影响。

**💡 创新点**

首次系统揭示非洲语言在样本规模变化下的非单调、语言特异性扩展行为，并指出小样本评估的偏差与饱和点。

**🔧 技术方法**

使用约0.6B参数的多语种Transformer（XLM-R Large、AfroXLM-R Large）在随机子采样下计算准确率、精确率、F1，并绘制方差与斜率热图。

**📊 数据集**

AfriXNLI人译版NLI数据集，覆盖16种非洲语言。

**📈 对比分析**

在50-500样本规模上多次随机抽样，比较平均准确率与标准差；结果显示有些语言正斜率、零斜率或负斜率，准确率在大样本（>300）更稳定但整体不随样本量单调提升。

**⚠️ 局限性**

仅针对AfriXNLI，未探究学习曲线；仅评估0.6B参数模型；未验证更大模型或其他任务的通用性；实验仅评估而非训练，可能与实际学习动力学不同。

---

## 334. Generative AI-Enabled Refund Fraud in Chinese E-Commerce: Investigation on Merchants and Platform Workers

**arXiv ID:** 2606.03215 | [PDF](https://arxiv.org/pdf/2606.03215v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**通讯引用:** 1587 | [OpenAlex ID](https://openalex.org/A5011803365)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对中国电商平台退款欺诈，尤其是基于生成式AI的欺诈，进行了访谈研究并提出了四阶段攻击向量分类、验证机制与制约因素；

**💡 创新点**

首次系统性识别生成式AI在退款流程中的多维攻击模式，并将其与传统欺诈对比，提出跨平台协同防御、物理-数字锚点与风险适应式仲裁流程等创新性对策；

**🔧 技术方法**

采用半结构化访谈、主题编码分析、人工与AI辅助审查工具（如Exif检查、图像一致性验证）、多视角视频请求等多模态验证技术；

**📊 数据集**

未使用公开数据集，所有数据来自17名商家和13名平台工作人员的访谈文本；

**📈 对比分析**

由于研究属于定性案例研究，未进行量化比较或性能评估；重点在于揭示攻击机制与防御挑战，而非算法效果；

**⚠️ 局限性**

样本量有限、仅聚焦中国平台，缺乏跨平台或西方平台的验证，访谈数据受被访者回忆偏差，且缺乏攻击者视角与实时实验验证。

---

## 335. MemTrain: Self-Supervised Context Memory Training

**arXiv ID:** 2606.03197 | [PDF](https://arxiv.org/pdf/2606.03197v1)

**作者:** Ziheng Li `[一作]` (Peking University), Yehui Tang `[通讯]` (Samsung Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了MemTrain框架，利用自监督代理任务提升LLM在长时记忆任务中的上下文记忆能力；

**💡 创新点**

创新在于设计两种相互耦合的代理任务（端到端掩码重构与中间记忆回想），并联合使用GRPO进行训练；

**🔧 技术方法**

采用GRPO强化学习，端到端掩码重构、IMR回想以及奖励计算；

**📊 数据集**

使用未标注的维基百科文本进行训练，构造掩码实体与多段相关文档；

**📈 对比分析**

在长文本QA（HotpotQA）和搜索式多跳QA（多种基准）上进行下游微调，对比直接微调，MemTrain+微调显著提升平均分，分别提高约5%至17%点；

**⚠️ 局限性**

局限在于仅在维基百科上训练，可能对其它领域或多模态记忆不适用，且对极长上下文仍有限制。

---

## 336. Lean 4 Machine-Verified Proof of P = NP via the Pedigree Polytope Membership Problem

**arXiv ID:** 2606.03194 | [PDF](https://arxiv.org/pdf/2606.03194v1)

**作者:** T. S. Arthanari `[一作]` (University of Auckland), T. S. Arthanari `[通讯]` (University of Auckland)

**通讯引用:** 1200 | [OpenAlex ID](https://openalex.org/A5014099345)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并证明了一个关于族谱多面体的成员资格问题（M3P）的强多项式时间算法，并通过该算法推导出旅行商问题（STSP）在多项式时间内可解，进而宣称 P=NP；

**💡 创新点**

创新点在于：①构造了层状网络和禁止弧运输问题来捕捉族谱结构；②将成员资格问题转化为多商品流（MCF）可行性判断；③利用 Tardos 的组合线性规划算法实现强多项式解法；④将整个证明在 Lean 4 证明助手中完全机证；

**🔧 技术方法**

使用的技术包括：组合线性规划、Tardos 的强多项式算法、层状网络与禁弧运输（FAT）构造、分层多商品流（MCF）模型，以及 Lean 4 形式化与自动证明；

**📊 数据集**

示例数据集主要是 Dantzig 的 42 城市 TSP 实例，用于展示成员资格检查与最优族谱构造；

**📈 对比分析**

相较于传统的分支定界或割平面方法，提出的算法在理论上具有 O(n^14) 的强多项式复杂度；但文中未给出实际运行时间实验，实际性能尚未评估；

**⚠️ 局限性**

主要限制包括：①O(n^14) 的理论上限仍相当保守；②对 STSP 的多项式时间改进并未经过实践验证；③所需的多步多商品流与网络构造在大规模实例上可能具有较高的常数因子；

---

## 337. Focused on the User, Overlooking the Risks: Security and Privacy Understandings, Practices and Challenges of Independent Chinese AI Agent Developers

**arXiv ID:** 2606.03190 | [PDF](https://arxiv.org/pdf/2606.03190v1)

**作者:** Shuning Zhang `[一作]` (Tsinghua University), Hewu Li `[通讯]` (Tsinghua University)

**通讯引用:** 1587 | [OpenAlex ID](https://openalex.org/A5011803365)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过对28名中国独立AI代理开发者的半结构化访谈，研究他们对安全与隐私（S&P）风险的认知、实践与面临的障碍。

**💡 创新点**

首次系统揭示独立开发者以用户为中心的S&P风险认知、依赖非正式沟通、缺乏正式工具的现状，并构建了三类障碍（动机、资源、监管）及其对策。

**🔧 技术方法**

使用半结构化访谈、主题分析（Braun & Clarke方法）以及代码本建设来整理数据。

**📊 数据集**

收集了28名受访者的访谈记录和自我报告的开发经验问卷，未使用传统机器学习数据集。

**📈 对比分析**

本研究采用定性分析而非性能对比，主要通过主题归纳展示洞见；未进行实验性性能评估。

**⚠️ 局限性**

样本自选性强、仅覆盖中国开发者、缺乏对跨文化差异的验证，且基于自我报告可能存在回忆偏差。

---

## 338. RogueMerge: Robust and Unified Attacks against LLM Model Merging

**arXiv ID:** 2606.03344 | [PDF](https://arxiv.org/pdf/2606.03344v1)

**作者:** Jinghuai Zhang `[一作]` (University of California Los Angeles), Yuan Tian `[通讯]` (University of California Los Angeles)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了一种针对大型语言模型（LLM）合并过程的统一攻击框架 RogueMerge，能够在未知合并设置下植入或放大后门、提示注入、越狱与系统提示提取等威胁。

**💡 创新点**

创新点在于将合并不确定性建模为随机最小‑最大问题，并采用分布鲁棒优化（DRO）保证对攻击提示的泛化，从而实现对生成式 LLM 合并的鲁棒且多目标攻击。

**🔧 技术方法**

技术手段包括：任务向量联合优化、合并不确定性感知优化（MUAO）与投影梯度上升寻找最坏干扰、第一阶 Taylor 拟合的分布鲁棒损失以及元学习式模拟训练。

**📊 数据集**

使用的数据集包括 Llama-3-8B 与 Qwen-2.5-7B 基础模型，六个任务（指令调优、数学推理、多语种问答、医疗问答、编程、安全），以及小规模的阴影攻击数据集（LLM‑LAT、ShareGPT）。

**📈 对比分析**

与基线（LoBAM、MergeHijacking、SFT）对比，RogueMerge 在四类攻击的入域与出域成功率上提升 20%–80%，在 170+ 合并模型、6 种合并算法下均保持高效，并且对常见防御（裁剪、微调、蒸馏）无显著抵御。

**⚠️ 局限性**

局限性包括：需要攻击者对攻击提示有一定先验知识、对合并系数区间假设有限、在极高维干扰或异常合并策略下仍可能出现失效；此外，检测/防御机制仍需进一步研究以应对自适应攻击。

---

## 339. Autonomous Navigation System for Library Service Robot Based on Unitree Go2 Edu

**arXiv ID:** 2606.03340 | [PDF](https://arxiv.org/pdf/2606.03340v1)

**作者:** Aoduo Li `[一作]` (Guangdong University of Technology), Zimeng Li `[通讯]` (Shenzhen Polytechnic University)

**通讯引用:** 1122 | [OpenAlex ID](https://openalex.org/A5088518232)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文设计并实现了一套基于Unitree Go2 Edu 四足机器人的自主导航系统，用于图书馆服务，集成 LiDAR、前置深度摄像头、IMU 等传感器，实现实时定位、地图构建和路径规划。

**💡 创新点**

创新点包括：①针对图书馆窄通道、地面不平、临时障碍等实际情况，证明四足机器人比轮式平台更适合；②提出 LiDAR 与 RGB‑D 深度相机的实用融合方案，兼顾全景覆盖与前方细节；③对地图质量进行手动测距和双跑重叠评估，量化误差；④在实验中给出可复现的动态障碍场景定义，提升评测客观性。

**🔧 技术方法**

技术方案主要采用 ROS 2 Humble + Nav2，RTAB‑Map 进行图优化 SLAM；EKF 结合腿部里程计、IMU 与视觉里程计进行姿态融合；AMCL 用于基于二维占据网格的地图定位；全局规划使用 A*，局部规划采用 DWA；传感器融合层将 4D LiDAR 点云投影为 2D 障碍层，RGB‑D 在前方 0.35–4 m 区域提供稠密点云，并在地图构建与实时避障时合并。

**📊 数据集**

数据来源为真实大学图书馆（约20 m × 15 m）环境，包括书架、阅览区、通道及地面过渡条。实验设置十个目标点，分别在三条书架通道和一处开放阅览区，累计 150 次导航任务；通过激光测距仪测量 12 条控制距离，并在两次独立跑测量地图重叠度。

**📈 对比分析**

在三类场景下的成功率分别为：静态 100%、低密度动态 96%、高密度动态 88%；平均位置误差 <0.18 m，偏航误差 <0.15 rad。通过感知模式消融实验，LiDAR+RGB‑D 配置的地图误差 3.7 cm、动态场景成功率 88%；单独 LiDAR 4.8 cm/84%，单独 RGB‑D 6.9 cm/79%。局部规划对比实验表明 DWA 在窄通道下的成功率和碰撞率优于 TEB。

**⚠️ 局限性**

限制主要包括：对反光表面感知不佳；在高密度拥挤时仍需较为保守的避障策略；系统性能受限于当前硬件配置（Jetson Orin Nano）与传感器视场；长期部署与语义任务集成尚待验证。

---

## 340. IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension

**arXiv ID:** 2606.03338 | [PDF](https://arxiv.org/pdf/2606.03338v1)

**作者:** Julie Mordacq `[一作]` (Inria Saclay), Steve Oudot `[通讯]` (Inria Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种无监督评估自监督学习（SSL）表示的方法，基于最小生成树（MST）推断表示的内在维度（ID），用以衡量表示质量；

**💡 创新点**

创新点在于将MST维度估计器（dim_MST）作为SSL表示评估指标，证明其与下游线性探测性能高度相关，且能在不需要标签的情况下实现超参选择与训练动态监测；

**🔧 技术方法**

使用的技术包括MST维度估计、Spearman/Kendall相关性分析、线性探测与kNN评估、在线与离线训练监控、以及对比指标如α‑ReQ、RankMe、LiDAR；

**📊 数据集**

实验涵盖ImageNet、iNat-18/21、SUN397、CIFAR‑10/100等多种数据集，评估了ResNet与ViT两大体系结构下的33个不同SSL模型（如VICReg、DINO、I‑JEPＡ、CLIP、EVA‑CLIP等）；

**📈 对比分析**

与线性探测、kNN评估和有监督验证结果相比，dim_MST与下游准确率呈显著负相关（Spearman ρ≈−0.8，Kendall τ≈−0.6），且在超参搜索中能逼近监督oracle表现，计算成本比训练10轮线性探测低数倍；

**⚠️ 局限性**

局限性包括：①在训练初期（尤其ViT前10轮）ID信息不稳定；②对视觉‑语言模型（CLIP、EVA‑CLIP）表现偏差；③虽然相关性强，但ID无法完全解释所有准确率方差，需与其他度量结合使用。

---

## 341. A Graph Foundation Model with Spectral Parsing and Prototype-Guided Spatial Propagation

**arXiv ID:** 2606.03315 | [PDF](https://arxiv.org/pdf/2606.03315v1)

**作者:** Ankang Yang `[一作]` (Tianjin University), Weixiong Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 12954 | [OpenAlex ID](https://openalex.org/A5068659777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了SPG图基础模型，结合谱解析与Gromov‑Wasserstein原型几何实现跨图知识迁移。

**💡 创新点**

创新点在于使用可学习的Chebyshev滤波分解频谱，并通过GW原型空间保留可迁移的节点对关系，直接作为传播先验。

**🔧 技术方法**

采用谱图神经网络、Chebyshev多项式滤波、GW barycenter原型学习、热扩散核、门控多路径传播以及轻量级新图适配。

**📊 数据集**

实验使用节点分类的Cora、Photo、CS、Chameleon等四个源图和多目标图，以及图分类的COLLAB、DD、ENZYMES、IMDB-B、PROTEINS等。

**📈 对比分析**

与自监督方法（BGRL、GraphMAE2、GRACE）及现有图基础模型（GCOPE、GraphAny、SAMGPT、TIG、GraphGlue）比较，SPG在1‑shot节点/图分类及自监督预训练下均显著优于或接近最优。

**⚠️ 局限性**

局限性包括对原型维度和滤波数量敏感、原型空间大小需平衡；对大图GW对齐计算成本高，需使用软聚类近似；在纯局部标签对齐的图中部分组件效果不如单独邻域传播。

---

## 342. The Reliability Gap in Benchmark Auditing: Distribution Shift and Scale as Failure Modes of Contamination Detection

**arXiv ID:** 2606.03305 | [PDF](https://arxiv.org/pdf/2606.03305v1)

**作者:** Wojciech Zarzecki `[一作]` (NASK National Research Institute), Sebastian Cygert `[通讯]` (NASK National Research Institute)

**通讯引用:** 216 | [OpenAlex ID](https://openalex.org/A5069232913)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对三种主流训练数据泄漏检测方法（LLM Dataset Inference、Post-Hoc Dataset Inference、CoDeC）在现实指令调优模型和工业模型上的可靠性进行了系统评估。

**💡 创新点**

创新点包括：将研究从学术、可控环境迁移到真实“在野”场景；首次揭示分布偏移与规模限制对检测性能的影响；提出对不同层级（训练集、测试集、工业模型）分割级别泄漏的失败模式；并公开了评测基准与代码，促进后续研究。

**🔧 技术方法**

使用的技术包括：基于 MIA 的数据集推断与 t 检验、生成式合成验证集构造（Post-Hoc DI）、上下文置信度差分测度（CoDeC）以及对比统计分析。

**📊 数据集**

使用的数据集覆盖多种场景：Pythia/Pile、OLMo 2、GSM8K、MMLU、DROP、MedQA-USMLE、MedMCQA、PLL uM、以及公开工业模型（Gemma、Qwen、Llama 3.2）的 benchmark。所有数据集均以训练/测试分割方式评估。

**📈 对比分析**

对 335 次评估中仅 199 次给出正确结论：LLM DI 在 IID 条件下表现较好但在分布偏移时误报严重；Post-Hoc DI 因规模不足导致信号不稳；CoDeC 只能提供粗粒度污染信号，无法区分同一 benchmark 的不同 split，整体性能低于预期。

**⚠️ 局限性**

局限性包括：对 IID 先验高度敏感；当 benchmark 规模很小或分布与训练集不匹配时，Post-Hoc DI 无法生成有效的合成验证集；CoDeC 缺乏分割级别辨别能力；缺乏透明的数据 provenance，统计检测尚不能替代手工审计。

---

## 343. SagaQA: A Multi-hop Reasoning Benchmark for Long-form Narrative Understanding in TV Series

**arXiv ID:** 2606.03301 | [PDF](https://arxiv.org/pdf/2606.03301v1)

**作者:** Galann Pennec `[一作]` (University of Toulouse), Nancy F. Chen `[通讯]` (Agency for Science, Technology and Research (A*STAR))

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 SagaQA——一个针对完整电视系列进行多跳、跨集、跨模态推理的长视频问答基准，并系统评估了并行、顺序与混合规划器在此任务中的表现；

**💡 创新点**

创新点包括①在 20 集（约 20 小时）全长视频中构造需要多跳推理的问答；②设计并验证了混合规划器，结合并行与顺序的优势显著提升了多跳推理质量；③提供了完整的数据生成、筛选与人工验证流程，为长视频推理研究奠定可复现的数据资源；

**🔧 技术方法**

技术上使用 Gemini‑2.5 Flash 生成 QA 对，利用 VideoRAG 作为检索工具，采用 Qwen3‑30B‑Instruct / Mistral‑Small‑3.1‑24B 进行规划与答案生成，并实现了并行、顺序、混合三种规划策略；

**📊 数据集**

数据集为 SagaQA，基于《As the World Turns》2007‑2010 年共 200 组 QA，覆盖 20 集连续剧情，包含多跳、跨模态信息；

**📈 对比分析**

对比方法包括 TextRAG、VideoRAG、EGAgent、VideoExplorer 等；在 episode grounding 评价中，混合规划器取得最佳 F1（≈41），顺序规划器 F1≈33，并行规划器 F1≈34；在文本生成（ROUGE/METEOR）上差异不大；

**⚠️ 局限性**

局限性包括仅以剧集级别进行评估，未细化到片段；只覆盖肥皂剧类剧情，可能不具普适性；固定 20 集窗口，未检验更长生命周期的记忆需求；答案合成仍受限于信息聚合与叙事捕捉能力。

---

## 344. GROSS: German Rail Open-Source SUMO Scenario

**arXiv ID:** 2606.03282 | [PDF](https://arxiv.org/pdf/2606.03282v1)

**作者:** Juri Penell `[一作]`, Damian Dailisan `[通讯]`

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一条名为GROSS的端到端管道，用开放数据（OpenStreetMap与GTFS）生成德国全国规模的高保真铁路仿真场景，并将其集成到SUMO中进行微观仿真。

**💡 创新点**

创新点包括：①分层站点模型与上下文匹配，避免仅用地理位置的站点-轨道映射导致的错误；②两层加权最短路径路由，结合站点层与轨道层，显著减少死锁和虚拟跳跃；③针对轨道方向、信号、平台等约束的启发式修复和路由重建，提升模拟稳定性和真实性。

**🔧 技术方法**

技术主要包括SUMO工具链（netconvert、duarouter、sumo）、自定义Python脚本实现站点聚类、站点匹配与路由算法、基于图搜索的两层路由器，以及对节点和边的加权和修复逻辑。

**📊 数据集**

数据集：OpenStreetMap（轨道、站台、信号等基础设施）和Delfi提供的德国GTFS时刻表（包含乘客列车、城际等服务），以及通过GitHub脚本提取的德铁（DB）实时延误统计用于验证。

**📈 对比分析**

比较方法：在四个德国子区域（Bayern、Brandenburg、Niedersachsen、Nordrhein-Westfalen）与传统SUMO工具链做生成时间、车辆跳跃（teleportation）次数、延误分布等指标对比。GROSS在生成时间上比基础管道快数百倍，在跳跃率上降低1.7–76.8倍，延误分布明显收敛，峰值延误从几十分钟降到几分钟。

**⚠️ 局限性**

局限性：仍依赖于GTFS的精度与完整性，缺乏对列车调度、单轨道冲突、站台分配、运营规则的细粒度建模；对特定站点（如慕尼黑、法兰克福）仍出现高尾延误；未实现实时重路由与单轨道冲突解决等高级功能。

---

## 345. VistaHop: Benchmarking Multi-hop Visual Reasoning for Visual DeepSearch

**arXiv ID:** 2606.03273 | [PDF](https://arxiv.org/pdf/2606.03273v1)

**作者:** Hang He `[一作]` (East China Normal University), Guojun Yin `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了视觉深度搜索（Visual DeepSearch）基准，并开发了统一评估框架，用以测试模型在多跳视觉推理、重复图像检查和证据融合上的能力。

**💡 创新点**

创新点在于：①设计了覆盖300张高分辨率图像、25个视觉搜索场景的基准；②构造了长链式证据链和多链融合问题；③引入了三代理漏检机制与人类审核以确保问题不可仅靠文本推断；④提供可复现的工具化评估流程。

**🔧 技术方法**

主要技术包括：多模态大规模推理模型（MLRM）、工具驱动（文本搜索、图像搜索、图像裁剪）、关系链构造与验证、LLM校验与人机混合审核。

**📊 数据集**

数据集为基于Wikimedia Commons、Wikipedia、Unsplash等公开来源筛选的300张高分辨率图像，生成350个VQA任务（190单链+160多链）和对应的证据链。

**📈 对比分析**

对7个主流MLRM（Gemini 2.5 Pro、GPT‑5、Claude‑4、Qwen3‑VL‑30B、Qwen3‑VL‑235B、SenseNova‑MARS‑32B、Tongyi‑DeepResearch‑30B）进行评估；最佳模型SenseNova‑MARS‑32B在Search+Crop设置下仅达24.31% Pass@1，表明视觉深度搜索仍极具挑战性；工具使用显著提升性能，但在L3（>10跳）与多链任务上依旧低效。

**⚠️ 局限性**

局限性包括：①构建流程高度依赖强大LLM和人工审核，成本与可复现性受限；②数据集规模相对有限，缺乏跨语言或多域扩展；③评估框架主要关注单一任务设置，未覆盖更复杂的开放式视觉搜索场景。

---

## 346. Agentic Relationship Harm: Benchmarking and Gating Relational Manipulation in AI Agents

**arXiv ID:** 2606.03271 | [PDF](https://arxiv.org/pdf/2606.03271v1)

**作者:** Pei-Sze Tan `[一作]` (National Institute of Informatics), Isao Echizen `[通讯]` (National Institute of Informatics)

**通讯引用:** 5921 | [OpenAlex ID](https://openalex.org/A5044556342)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究AI代理在情感敏感场景中可能造成的关系操控风险，构建了角色敏感的评估框架和轻量级后生成门控。

**💡 创新点**

创新点包括提出“代理关系危害”概念，设计了110条攻击/受害者对齐的基准和心理学驱动的标签体系，并提出专门针对关系风险的后生成门控。

**🔧 技术方法**

使用技术包括OpenClaw本地代理、LLM-as-judge（GPT‑4o‑mini）进行多标签评估，以及基于规则的后生成门控实现。

**📊 数据集**

数据集包括110条单轮攻击/受害者平衡提示、40条多轮压力测试、Fraud‑R1中英文分割、30条受害者脆弱性分层样本等。

**📈 对比分析**

通过与通用安全提示基准对比，衡量有害遵从、保护干预和拒绝率。关系门控在主基准上将有害遵从率降至0%，保护干预率提升至77%，显示出显著性能提升。

**⚠️ 局限性**

局限性在于仅评估文本输出，未覆盖记忆写入/工具调用等行为；门控仅在本地OpenClaw上测试；样本覆盖有限，且对不同文化背景的关系规范考虑不足。

---

## 347. Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles

**arXiv ID:** 2606.03270 | [PDF](https://arxiv.org/pdf/2606.03270v1)

**作者:** Li Sun `[一作]` (Beijing University of Posts and Telecommunications), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 137184 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了基于Riemannian几何的图基础模型Gauge，利用神经向量丛框架学习图的内在几何并通过Dirichlet损失实现可迁移结构的识别与自适应

**💡 创新点**

创新点在于：1）将向量丛概念引入图学习，形成“神经向量丛”来捕捉图的局部坐标与全局结构；2）证明可迁移子结构与向量丛的几何平坦性相联系，并通过Dirichlet能量量化行为不变性；3）设计Gauge架构与Dirichlet损失，既能在预训练阶段学习内在几何，又能在微调阶段限制对可迁移结构的梯度更新，从而显著提升跨域迁移性能；

**🔧 技术方法**

核心技术包括：Riemannian向量丛理论、局部坐标（局部平凡化）与伪平行传输的神经实现、门控展开的向量丛平滑、基于Dirichlet能量的损失与自适应阈值的可迁移结构回收算法

**📊 数据集**

在预训练中使用了多域数据集（学术、社交、电子商务等）作为源图；在微调与零样本评估中采用PubMed、Facebook、Roman‑empire、Photo等目标图；同时在图同构任务上使用CSL、MUTAG、ZINC12K等基准；

**📈 对比分析**

与16种强基线（传统GNN、无监督GNN、图基础模型等）在跨域迁移、零样本链接预测和图同构任务上进行比较；Gauge在1-shot、5-shot、零样本场景中平均提升约10-20%的准确率/ AUC，尤其在零样本链接预测与图同构任务上显著优于现有最佳模型，显示出更强的知识迁移与表达能力；

**⚠️ 局限性**

局限性包括：1）理论和实验主要关注无属性或简化属性图，对高维属性图的适用性尚未充分验证；2）Gauge的预训练与微调需要较高计算资源，可能不适用于极大规模图；3）可迁移结构的阈值设定与解码方法在不同任务中可能需要手动调参；4）对噪声或不完整图的鲁棒性尚未系统评估。

---

## 348. Distilling Answer-Set Programming Rules from LLMs for Neurosymbolic Visual Question Answering

**arXiv ID:** 2606.03269 | [PDF](https://arxiv.org/pdf/2606.03269v1)

**作者:** Thomas Eiter `[一作]` (Vienna University of Technology), Johannes Oetsch `[通讯]` (Jönköping University)

**通讯引用:** 473 | [OpenAlex ID](https://openalex.org/A5028255011)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于大型语言模型的声明性知识蒸馏方法，用于自动扩展视觉问答系统中的ASP推理模块。

**💡 创新点**

利用少量VQA实例引导LLM生成或修正ASP规则，而非传统的数据驱动学习；结合多种提示策略（多提示、链式思考、规则修正）和回归测试。

**🔧 技术方法**

答案集程序（ASP）作为符号推理语言，LLM（GPT‑4o、DeepSeek、Gemini‑3等）作为规则生成器，ASP求解器（clingo）进行语法/语义验证和回归检测。

**📊 数据集**

GQA、CLEVR以及图形网络VQA数据集（未命名的图形数据集）进行实验。

**📈 对比分析**

对每个缺失谓词进行实验，使用LLM蒸馏后在完整测试集上评估准确率；结果显示GPT‑4o、DeepSeek和Gemini‑3在大多数谓词上接近或达到100%准确率，且所需示例数平均3–8例，耗时低。

**⚠️ 局限性**

方法依赖可分解的合成问题和现有ASP表示，提示设计对模型有一定敏感性；在更复杂的递归/聚合任务上仍需更大模型；规则生成不保证最优或最简洁，需后期剪枝。

---

## 349. Wheel-Mounted/GNSS Fusion with AI-Aided Position Updates

**arXiv ID:** 2606.03265 | [PDF](https://arxiv.org/pdf/2606.03265v1)

**作者:** Gal Versano `[一作]` (University of Haifa), Itzik Klein `[通讯]` (University of Haifa)

**通讯引用:** 2633 | [OpenAlex ID](https://openalex.org/A5012718881)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了将轮装惯性传感器网络回归的位姿估计作为外部更新融入错误状态扩展卡尔曼滤波的混合神经惯性导航框架。

**💡 创新点**

创新点是将周期性运动下的轮装惯性网络WMINet预测结果直接作为导航滤波器更新，并在GNSS更新间插入WMINet更新以提升更新频率。

**🔧 技术方法**

采用WMINet（二维CNN+全连接网络）进行位移回归，使用错误状态EKF融合惯性和GNSS测量。

**📊 数据集**

使用ROSBot‑XL轮装IMU数据集，包含26条周期性轨迹、38分钟单IMU、190分钟多IMU。

**📈 对比分析**

与标准轮装IMU EKF对比，WEKF在测试轨迹上PRMSE从2.12 m降至1.15 m，约提升46%，TDE从17.65%降至9.58%。

**⚠️ 局限性**

局限在于仅适用于周期性轨迹，无法处理非周期性任意运动。

---

## 350. A Geometric Lens on Physics-Aligned Data Compression

**arXiv ID:** 2606.03279 | [PDF](https://arxiv.org/pdf/2606.03279v1)

**作者:** Aleix Segui `[一作]`, Wesley Armour `[通讯]` (University of Oxford)

**通讯引用:** 1288 | [OpenAlex ID](https://openalex.org/A5080149070)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `14d48e9d-0069-4ad9-996a-1d5968216998` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

研究了物理对齐压缩的局部几何特性，提出了基于率-失真理论的本地切线空间模型和物理对齐诊断。

**💡 创新点**

将熵模型、物理观测灵敏度与重建误差的三种隐空间度量相结合，推导出压缩噪声的逆刚度分配规则，并提出度量对齐的主子空间重叠指标。

**🔧 技术方法**

采用自编码器+可学习熵模型的深度压缩框架，利用二阶近似、Hessian、随机子空间迭代等技术实现量化误差与率-失真分析。

**📊 数据集**

在PDEBench的二维Navier–Stokes湍流、宇宙学模拟和电子显微镜脑皮层体积数据上进行实验。

**📈 对比分析**

通过比较不同物理权重β下的MSE与物理观测失真曲线，验证固定比特率下物理-均方误差权衡与对齐诊断的相关性；对齐度越高，物理保真提升而MSE下降越小。

**⚠️ 局限性**

仅给出对齐分析而未给出对齐的训练约束，未建模重建偏差，只考虑随机噪声误差。

---

## 351. Cross-Modality Feature Fusion Based on Structured State Space Duality for Multimodal Image Registration Network

**arXiv ID:** 2606.03341 | [PDF](https://arxiv.org/pdf/2606.03341v1)

**作者:** Zhikang Li `[一作]` (Xidian University), Ming Li `[通讯]` (Xidian University)

**通讯引用:** 48174 | [OpenAlex ID](https://openalex.org/A5100452145)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本研究提出了一种基于Structured State Space Duality的跨模态特征融合网络RegNetMamba-2，用于多模态图像配准。

**💡 创新点**

创新点在于将SSD引入粗细尺度匹配并设计了跨模态特征交互(CMI)和多尺度特征融合(MSF)模块，并通过特征缩放实现局部增强。

**🔧 技术方法**

使用了SSD、Mamba-2结构、跨模态交互、局部特征缩放、窗口注意力等技术。

**📊 数据集**

在VIS‑SAR（OSDataset）、VIS‑IR（LGHD/RoadSence）和VIS‑NIR（RGB‑NIR）三大多模态数据集上进行实验。

**📈 对比分析**

与RIFT、Cnet、ReDFeat、LoFTR、XoFTR、LoFLAT、JamMa等方法对比，RegNetMamba‑2在aRMSE、aNCM、SMR指标上均优于对手，表现最优。

**⚠️ 局限性**

局限性包括相较于部分轻量级方法推理略慢，且在极端噪声或高分辨率场景下仍需进一步验证。

---

## 352. Calibration Data Trade-offs Across Capability Dimensions: Why Multi-Source Mixing Matters for High-Sparsity LLM Pruning

**arXiv ID:** 2606.03328 | [PDF](https://arxiv.org/pdf/2606.03328v1)

**作者:** Hu Xu `[一作]` (Shanghai Jiao Tong University), Jianfeng Xu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2248 | [OpenAlex ID](https://openalex.org/A5101652522)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究后训练稀疏化过程中校准数据对模型四大能力维度（General、Commonsense、Code、Math）的影响，并通过多源混合与无对齐语料自适应生成 IGSP 提升稀疏化保留率。

**💡 创新点**

创新点在于揭示校准困惑度与不同能力维度的相关性呈相反符号，提出跨维度多源混合策略及无手工对齐语料的 IGSP 自动构造方法。

**🔧 技术方法**

采用 Spearman 相关分析、Objective Information Theory (OIT) 指标、SparseGPT/Wanda/Magnitude 稀疏化算法、greedy 4‑gram 采样、Perplexity 平衡等技术。

**📊 数据集**

使用 LLaMA‑3.1‑8B/70B、C4、Wikipedia、GSM8K、MetaMath、MBPP 等真实语料，以及 Self‑Cal、SGS、IGSP 自生成数据。

**📈 对比分析**

在 SparseGPT 60% 稀疏率下，多源混合实现总保留率 58.8%，比最佳单源（MetaMath 50.0%）高 8.8%，比 C4 低 18.8%；IGSP 在自生成轨道上比 Self‑Cal/SGS 分别提升 2.4%/4.8%，但仍落后于真实多源混合 9.7 点。

**⚠️ 局限性**

仅验证了 LLaMA‑3.1 系列和三种稀疏化器，缺少跨架构、跨语言、量化或多模态等实验；IGSP 受表面困惑度限制，无法完全弥补真实数据质量差距；未考虑微调恢复等技术。

---

## 353. TASE: Truncation-Aware Semantic Embeddings for 3D Scene Understanding and Editing

**arXiv ID:** 2606.03314 | [PDF](https://arxiv.org/pdf/2606.03314v1)

**作者:** Tim-Felix Faasch `[一作]` (Bosch Research), Cyrill Stachniss `[通讯]` (University of Bonn)

**通讯引用:** 26103 | [OpenAlex ID](https://openalex.org/A5011166267)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于截断感知语义嵌入（TASE）的3D场景编辑框架，能够利用文本提示在3D高保真场景（3D Gaussian Splatting）中实现大规模几何和外观修改。

**💡 创新点**

创新点包括：1) 通过自编码器与Matryoshka层次化截断学习构造可调节语义抽象程度的嵌入；2) 引入尺度与平移等变损失消除二维位置偏置，使嵌入在多视角下保持一致；3) 将TASE作为ControlNet条件输入，实现对3D场景的文本驱动编辑；4) 在编辑过程中加入Fine‑Tuning阶段（类似Difix3D+）降低几何变换带来的渲染伪影。

**🔧 技术方法**

使用的技术包括：预训练的DINOv3特征提取器、单块Transformer自编码器、FLUX.1[dev] Diffusion模型、ControlNet架构、3D Gaussian Splatting渲染、相机姿态采样、交叉熵与MSE相结合的重建损失、尺度与平移等变损失、Mahalanobis正则、流匹配损失等。

**📊 数据集**

训练与评估主要使用公开的3D场景数据集（如ScanNet、Matterport3D等）以及从中提取的多视角图像；在编辑任务中使用对应的文本提示；对比基线使用Direct Gaussian Editing和GaussianEditor两种主流3DGS编辑方法。

**📈 对比分析**

通过与DGE和GE在局部与全局编辑任务上的定量评估（PSNR、SSIM、CLIP方向相似度）以及用户研究，结果表明本方法在几何和外观一致性、文本对齐度以及用户偏好方面均显著优于基线，尤其在大规模几何改动时表现更为突出。

**⚠️ 局限性**

局限性包括：1) 需要针对不同场景手动调节超参数（如截断级别、学习率、噪声时间表）以获得最佳效果；2) 采样过程仍可能产生局部伪影；3) 语义与外观未完全解耦，导致颜色调整不总能精准匹配提示，易出现过饱和现象。

---

## 354. Classification of independent sets in signed Johnson graphs and applications to kissing arrangements

**arXiv ID:** 2606.03299 | [PDF](https://arxiv.org/pdf/2606.03299v1)

**作者:** Rustem Takhanov `[一作]` (Nazarbayev University), Stanislav Yun `[通讯]` (Nazarbayev University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了带符号的约翰逊图 J_±(n,4) 中的最大独立集，特别关注 n≤12 的情况，并开发了一个算法来计算所有最大独立集。

**💡 创新点**

首次在 n=12 的情况下识别出 1579 个非同构的最大独立集，所有这些集对应于大小为 840 的非同构亲吻排列，展示了潜在的最优亲吻排列的多样性。

**🔧 技术方法**

使用了组合图论和编码理论的方法，特别是通过算法和线性规划技术来计算最大独立集。

**📊 数据集**

使用了带符号的约翰逊图 J_±(n,4) 的数据集，特别是 n=12 的情况，涉及到的独立集和亲吻排列。

**📈 对比分析**

通过与现有的编码理论和组合几何的结果进行比较，展示了在维度 7、9 和 12 中的亲吻数的最佳已知下界，性能表现优越。

**⚠️ 局限性**

在 n=5 和 n=11 的情况下，计算复杂性显著增加，导致无法完全处理这些情况，显示出这些情况下的独立集的复杂性。

---

## 355. SplitAdapter: Load-Aware Humanoid Loco-Manipulation via Factorized Adaptation

**arXiv ID:** 2606.03297 | [PDF](https://arxiv.org/pdf/2606.03297v1)

**作者:** Jeonguk Kang `[一作]` (Samsung Electronics), Donghan Koo `[通讯]` (Samsung Electronics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a`

**🎯 论文内容**

提出了SplitAdapter框架，利用冻结的PhysHSI风格基准策略，在其上添加对象/负载与动力学上下文编码器，进行负载感知的人形机体运动与物体搬运。

**💡 创新点**

创新点在于将适配器拆分为对象/负载分支和动力学分支，采用分割的世界模型目标、GRL交叉对抗正则化，以及分层FiLM调制，实现对重载和动力学差异的高效分离与补偿。

**🔧 技术方法**

使用冻结策略、双分支历史编码器、分割世界模型、梯度反转层（GRL）对抗正则化、分层FiLM特征调制等技术。

**📊 数据集**

实验数据集包括在Isaac Gym和MuJoCo中的仿真环境，覆盖2kg、4kg、6kg物体质量和0cm/30cm/60cm提起/放置高度，以及在Unitree G1机器人上的真实世界测试。

**📈 对比分析**

与基准PhysHSI、AnyAdapter风格WM-FiLM以及不同SplitAdapter变体对比，SplitAdapter在仿真和真实世界中均实现了更高的完整任务成功率，特别是在超出训练范围的6kg重负载和不同高度下表现突出。

**⚠️ 局限性**

局限性包括仅验证了刚性盒子搬运任务，未涉及变形物体或更复杂的接触丰富操作，以及真实世界评估规模受限于硬件成本与耐久性。

---

## 356. Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective

**arXiv ID:** 2606.03290 | [PDF](https://arxiv.org/pdf/2606.03290v1)

**作者:** Yancheng Chen `[一作]` (Chinese Academy of Sciences), Chuan Zhou `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 132303 | [OpenAlex ID](https://openalex.org/A5091114864)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了“棱镜空间理论（PS‑Theory）”以定量评估图模型适配能力，并基于该理论设计了轻量级的消息调优方法MTG，能够在保持预训练参数冻结的前提下提升GFMs的下游任务性能。

**💡 创新点**

（1）构建了PS‑Theory这一几何测度框架，首次为图提示调优提供上界分析；（2）提出MTG，在每层注入可学习的消息原型并动态融合，理论上可超越提示调优的上界；（3）通过多任务、多架构实验验证MTG在少样本场景下的优势。

**🔧 技术方法**

几何测度理论（棱镜空间）、Piecewise Linear GNN模型、ReLU等激活、动态消息融合机制、轻量级线性投影与Softmax、少样本评估协议。

**📊 数据集**

在ProG基准下的15个图数据集（7个节点分类：Cora、Citeseer、Pubmed等；8个图分类：IMDB‑B、COLLAB、PROTEINS、MUTAG等），以及多种自监督预训练策略（DGI、GraphMAE、EdgePreGPPT等）。

**📈 对比分析**

与传统的Fine‑Tuning以及多种图提示方法（GPPT、Gprompt、All‑in‑one、GPF、GPF‑plus）进行对比。MTG在1/3/5‑shot节点与图分类任务中均显著优于所有基线，尤其在1‑shot场景下几乎消除负迁移，性能提升幅度在10%+。

**⚠️ 局限性**

（1）PS‑Theory假设输入是光滑紧致流形，且层是分段线性，未涵盖非线性激活或黑盒模型；（2）MTG要求可访问每层消息传递细节，无法直接应用于API‑only图模型；（3）理论上提升适配空间并不必然在极低资源下保证最佳泛化，需进一步研究容量与过拟合的平衡。

---

## 357. Privilege Risk Evolution for Non-Human Identities: A Temporal Fiber Model for Cloud IAM

**arXiv ID:** 2606.03289 | [PDF](https://arxiv.org/pdf/2606.03289v1)

**作者:** Christophe Parisel `[一作]` `[通讯]`, Christophe Parisel

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了将云 IAM 权限等价性分解为结构等价（图纤维）和时间等价（特权环）两层，并构建了族谱分区、三种特权环分类（单峰、振荡器、齿轮）与原子不变量，随后开发多尺度窗口方法进行预测，并在一个大型 Azure 租户上进行验证与回测。

**💡 创新点**

创新点在于：①将图纤维理论应用于动态权限分层；②提出时间等价性概念——特权环，揭示了不可逆的权限升级模式；③设计原子不变量（primorial 编码）为特权环提供可比的拓扑指纹；④通过族谱分区把复杂的纤维演化组织成可管理的线性结构；⑤提出多尺度窗口与稳定性扫描方法，使得早期检测齿轮特权环成为可能。

**🔧 技术方法**

技术手段包括：图纤维（Boldi‑Vigna）对权限图做结构压缩；Tarjan 算法求强连通分量；Johnson 算法枚举基本环；原子不变量基于质因数分解实现；多尺度窗口与线性扫描实现对不同时间粒度的分析；回测协议采用交叉验证与混淆矩阵评估。

**📊 数据集**

使用的实测数据为某大型 Azure 企业租户，约 100 次快照、总计 31,000 个 NHI（约 11,000 为基线、20,000 为新增），覆盖一年的变更记录。

**📈 对比分析**

与传统单快照工具（如 AWS Zelkova、IAM Access Analyzer 等）相比，本文方法通过多尺度扫描可在仅 3×基准窗口（≈21 次快照）内实现 1.000 的精准率、≈0.86 的召回率；在不同族谱分区下，齿轮特权环检测准确无误；基准工具无法捕捉时间等价性或齿轮模式。

**⚠️ 局限性**

主要局限包括：仅在单个 Azure 租户验证，缺乏跨云/行业普适性；回测为样本内验证，未检验外推性能；未验证特权环是否对应真实安全事件；算法基于纤维级别，忽略单权限细粒度与条件访问；长期记录会导致转移图膨胀，存储与计算成本上升。

---

## 358. BA-T: An Iterative Transformer for Two-View Bundle Adjustment

**arXiv ID:** 2606.03287 | [PDF](https://arxiv.org/pdf/2606.03287v1)

**作者:** Ganlin Zhang `[一作]` (Technische Universitaet Munich), Xi Wang `[通讯]` (Technische Universitaet Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于束束调整（Bundle Adjustment）思想的迭代Transformer（BA‑T），通过在隐式令牌空间中进行可重复的结构化更新，逐步精细化相机姿态与局部几何令牌，从而实现多视图图像的稠密3D重建。

**💡 创新点**

创新点包括：
- 将BA的迭代信息传递模式直接映射到Transformer层，形成单一可复用的更新模块；
- 用隐式令牌代替显式几何表示，兼顾了深度学习的表达能力与BA的几何一致性；
- 在跨视图注意力堆栈的基础上，采用结构化的摄像机更新与几何更新机制，显著降低参数量与计算成本。

**🔧 技术方法**

技术手段主要有：
- 隐式令牌空间中的相机条件几何变换（Condition block + adaLN）；
- 软对应匹配（基于注意力的对应关系）；
- 先摄像机后几何的两步迭代更新（Cross‑Attention 机制）；
- 迭代监督与逐步加权损失；
- 轻量级的Transformer层实现。

**📊 数据集**

使用的数据集：
- 7Scenes（室内真实场景，3,600对同视图数据）
- BundleFusion（室内/室外真实场景，1,109对同视图数据）
- 还在实验中对TUM‑RGBD等数据做了可视化验证。

**📈 对比分析**

与现有深度跨视图注意力解码器（如DUSt3R、ViSTA、VGGT、SLAM‑Former等）进行对比，评估指标包括：
- 视角误差（AUC@5°,10°,20°）
- 几何误差（Chamfer、准确率、完整率、δ_1.25）
- 3D对应误差、相对深度误差。
实验结果显示：BA‑T 在相同或更小的模型参数量下，取得了更高的姿态估计精度和几何重建质量，并在迭代过程中快速收敛（前三至四次迭代即可达到最优）。

**⚠️ 局限性**

局限性：
- 当前实现仅针对双视图重建，虽然设计可扩展到多视图，但多视图实验与优化仍未完成；
- 迭代次数对最终精度有一定依赖，过多迭代可能导致计算成本上升；
- 对低纹理或极低重叠场景的鲁棒性虽然有提升，但在极端情况下仍可能受限；
- 需要额外的训练标签（点地图、相机姿态）以及迭代监督，增加了数据准备成本。

---

## 359. Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection

**arXiv ID:** 2606.03334 | [PDF](https://arxiv.org/pdf/2606.03334v1)

**作者:** Pritam Kadasi `[一作]` (Indian Institute of Technology Gandhinagar), Mayank Singh `[通讯]` (Indian Institute of Technology Gandhinagar)

**通讯引用:** 1223 | [OpenAlex ID](https://openalex.org/A5100746903)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过系统化的12种prompt设计，在Gemma3-27B多语言LLM上实现SemEval-2026 Task 9的三子任务（二分类、目标多标签和表现多标签）自动检测和分类。

**💡 创新点**

首次在多语言偏见检测任务中对prompt设计进行梯度化实验，并证明prompt可实现零样本多语言偏见检测，尤其在二分类任务上表现突出。

**🔧 技术方法**

零样本prompt推理、系统化prompt微调、宏F1评估。

**📊 数据集**

SemEval-2026 Task 9提供的22种语言社交媒体文本，包含二分类标签、目标标签和表现标签。

**📈 对比分析**

使用宏平均F1和准确率对比不同prompt和模型，Gemma3-27B在12种prompt中最优，子任务1宏F1 0.762、子任务2 0.587、子任务3 0.444。

**⚠️ 局限性**

提示过于保守导致召回不足，跨语言性能差异大，未进行语言本地化prompt，缺乏细粒度多标签推理能力。

---

## 360. Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference

**arXiv ID:** 2606.03332 | [PDF](https://arxiv.org/pdf/2606.03332v1)

**作者:** Roman Plaud `[一作]` (Institut Polytechnique de Paris), Matthieu Labeau `[通讯]` (Institut Polytechnique de Paris)

**通讯引用:** 299 | [OpenAlex ID](https://openalex.org/A5055866824)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对下游任务的严格正则化评分规则（Strictly Proper Scoring Rule）框架，并应用于因果推断中的平均处理效应（ATE）估计，生成了一种针对IPW估计误差上界的闭式损失函数及其对应的概率映射。

**💡 创新点**

创新点在于将下游任务的误差曲率（即对概率估计的敏感度）与严格正则化评分规则的局部曲率对齐，从而构造出最小化IPW MSE上界的特定损失，并保留了与之对应的可凸优化的激活函数，解决了传统log-loss在边界概率处导致的方差爆炸问题。

**🔧 技术方法**

使用了严格正则化评分规则理论、偏差-方差分解、Taylor展开、权重函数匹配、解析求解四次方程得到的 canonical link，以及交叉拟合和深度/树模型的梯度下降优化。

**📊 数据集**

数据集包括三种半合成基准（IHDP、Jobs、Kang‑Schafer）以及ACIC 2017的32个高维数据生成过程，共计约4800个样本、58维特征。

**📈 对比分析**

与传统log‑loss、后处理剪枝/截断、CBPS、CBSR、Entropy Balancing、SBW等平衡方法以及XGBoost、MLP等不同基架进行对比，实验显示在所有下游估计器（IPW、Hajek、AIPW）上，所提出的定制损失在MAE、RMSE、偏差和整体排名上均优于基线，尤其在高选择强度场景下提升最显著。

**⚠️ 局限性**

局限性包括：仅保证局部曲率匹配而缺乏全局最优性保证；自定义损失与激活函数需要求解四次方程，导致前向计算成本略高；并非所有下游任务都能解析得到对应的 canonical link，可能需要数值近似。

---

## 361. dstack-capsule: Pod-Level Remote Attestation for Confidential Workloads on Kubernetes

**arXiv ID:** 2606.03323 | [PDF](https://arxiv.org/pdf/2606.03323v1)

**作者:** Yang Yang `[一作]` (OPPO), Wenfeng Wang `[通讯]` (Phala)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出了 dstack-capsule，一种在 Kubernetes 上实现 Pod 级远程证明的容器平台，允许多个 Pod 共享单一 TDX VM 并在保持硬件证明的同时实现隔离。

**💡 创新点**

核心创新在于两层证明架构（RTMR[3] 负责平台固化，64 字节字段承载 Pod 身份），引入不可逆特权熔丝、Sysbox 与 ZFS 多层沙箱，以及通过 Kubelet Authorizer Hook 的 Pod 级证书绑定。

**🔧 技术方法**

采用 Intel TDX 的 RTMR 与 Quote、Ed25519、SHA-384、JCS、HKDF、RA‑TLS 等加密技术，并结合 Sysbox、ZFS AES‑256‑GCM、WireGuard VPC、修改版 Kubelet 进行隔离与网络白名单。

**📊 数据集**

评估使用的是标准的空闲（idle）工作负载和简易演示容器（如 nginx），未使用公开数据集；实验基于 QEMU/TDX 模拟环境和真实 TDX 服务器。

**📈 对比分析**

与 Confidential Containers（CoCo）进行对比；Pod 启动延迟 6.25 s（dstack）对比 8.85 s（CoCo），内存开销每 Pod 仅 2 MB（共享 793 MB 基础）对比 2 GB；远程证明延迟约 24 ms；在 64 GB 服务器上可支持 86 个 Pod（受 kubelet 限制），显著提升密度。

**⚠️ 局限性**

局限性包括：共享 CVM 仍有更大 blast radius、目前仅支持 Intel TDX（AMD SEV‑SNP 暂未实现）、缺乏对 GPU TEE 的集成、评估中存储/网络性能基于 QEMU，未覆盖真实硬件带宽，且侧信道与细粒度安全证明尚未完成。

---

## 362. Beyond Ideal Instruction: A Comprehensive Framework for Evaluating LLMs in Realistic Interactions

**arXiv ID:** 2606.03318 | [PDF](https://arxiv.org/pdf/2606.03318v1)

**作者:** Xuan Yang `[一作]` (City University of Hong Kong), Ning Miao `[通讯]` (City University of Hong Kong)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了RUT-Bench基准，模拟真实用户在工具调用中的多轮交互，并评估19个主流LLM在该基准下的表现；

**💡 创新点**

创新点在于：①基于真实交互日志提出七类非理想用户行为体系；②通过LLM自动化生成可执行环境、任务与对话，并进行多级验证；③设计多维评估（成功率、信息诚实度、工具纪律）并采用LLM-as-a-Judge进行诊断；

**🔧 技术方法**

主要技术包括：LLM驱动的环境与任务生成、Python类可执行环境、抽象语法树与白盒评估、LLM-as-a-Judge评估、用户行为模拟与对话重写；

**📊 数据集**

使用的数据集包括API-Bank、ToolAce、Dolci（用于查询收集）以及WildChat（用于行为标签与日志），最终生成1638个测试样本覆盖59个工具环境；

**📈 对比分析**

通过对19个LLM的整体成功率、信息诚实度和工具纪律评分进行比较，最高模型GPT-5.4仅达到37.3%的成功率，所有模型均低于40%，非理想用户行为导致显著性能下降；

**⚠️ 局限性**

局限性在于：①非理想对话由LLM合成，可能缺乏真实语用细节；②LLM-as-a-Judge评估可能存在偏差；③使用确定性Python类环境，未模拟网络延迟、API限流等实际部署因素；④仅英文且仅涵盖七类非理想行为，未考虑恶意攻击或跨文化差异。

---

## 363. Generalizing Graph Foundation Models via Hyperbolic Retrieval-Augmented Generation

**arXiv ID:** 2606.03307 | [PDF](https://arxiv.org/pdf/2606.03307v1)

**作者:** Yifan Jin `[一作]` (Institute of Software, Chinese Academy of Sciences), Changwen Zheng `[通讯]` (Institute of Software, Chinese Academy of Sciences)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275`

**🎯 论文内容**

提出了一种基于双曲空间的检索增强生成框架HyRAG，用以提升图基础模型的零样本泛化能力

**💡 创新点**

创新点在于将双曲几何用于知识索引与检索，解决欧氏空间中语义粒度丢失与中心化问题，并通过多粒度检索与双路径融合实现对知识的多维利用

**🔧 技术方法**

采用双曲知识索引（HKI）模块、双粒度检索（MR）机制、双路径融合（DF）策略，辅以SBERT+Poincaré球映射、MMR、多头注意力与不确定性门控融合

**📊 数据集**

使用Commonsense知识图（ConceptNet、WordNet、Wikidata-CS）做检索库，并在七个公开图数据集（Cora、CiteSeer、WikiCS、Instagram、Ele-Photo、Ele-Computers、Books-History）上进行零样本实验

**📈 对比分析**

与12个基线（自监督图学习、图基础模型、Euclidean检索RAGRAPH）对比，HyRAG在节点分类任务上平均提升约2.02%（最高0.82%相较RAGRAPH），在链接预测上也有1–1.5%的提升，表明双曲检索显著提高性能

**⚠️ 局限性**

局限性包括对超参数敏感（如k、γ、β、α），以及在某些数据集（如Cora）上提升有限，未来需进一步优化检索策略与模型参数选择

---

## 364. LEAP: Supercharging LLMs for Formal Mathematics with Agentic Frameworks

**arXiv ID:** 2606.03303 | [PDF](https://arxiv.org/pdf/2606.03303v1)

**作者:** Po-Nien Kung `[一作]` (Google Cloud AI Research), Nanyun Peng `[通讯]` (Google Cloud AI Research)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于通用大型语言模型的代理式框架，通过蓝图驱动的递归分解与Lean编译器反馈实现正式数学证明；

**💡 创新点**

创新点在于将人类数学工作流转化为可迭代的蓝图+DAG结构，利用通用LLM的推理与自我修正能力，实现无需专门微调即可完成复杂正式证明；

**🔧 技术方法**

主要技术包括蓝图式分解、AND-OR DAG记忆化、交叉的非正式–正式规划、编译器反馈验证及LLM审阅筛选；

**📊 数据集**

使用了新构建的Lean-IMO-Bench（60道IMO题）以及Putnam 2025竞赛题集；

**📈 对比分析**

与Gemini‑3.1‑Pro、Goedel‑Prover‑V2‑32B、Hilbert及Aristotle等基线相比，在Putnam 2025上实现100%解题率，在Lean‑IMO‑Bench基础集和进阶集分别达到83.3%与56.7%，显著优于所有现有方法；

**⚠️ 局限性**

主要局限在几何类问题仍难以解决、对高难度子目标的搜索空间增长速度快、对LLM生成的分解质量高度依赖审阅、缺乏专门领域知识与高效分支优先策略。

---

## 365. A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting

**arXiv ID:** 2606.03280 | [PDF](https://arxiv.org/pdf/2606.03280v1)

**作者:** Peiyan Zhang `[一作]` `[通讯]` (Independent Researcher), Peiyan Zhang (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究评估将一款语言模型（Pythia‑160M）的隐藏激活通过线性映射传递给另一款模型（Pythia‑410M）并在推理时注入，以提升多跳推理性能。

**💡 创新点**

发现尽管线性映射在归一化空间中达到高相似度，但仅靠离线对齐不足以实现接收方的因果可用性，首次阐明了激活通信与可用性之间的分离。

**🔧 技术方法**

使用线性翻译层、替换式注入与加法式注入以及对比实验。

**📊 数据集**

使用Pythia系列模型在多跳推理任务的自定义评估集（396条样本）。

**📈 对比分析**

与无注入、自然语言中继等基线比较，结果表明无注入与自然语言基线略优，激活注入无显著提升，替换式注入甚至更差。

**⚠️ 局限性**

局限性包括仅测试单一模型对、单一任务、固定注入层、无接收方训练以及低基线准确率，未涵盖更大规模或更强模型的情况。

---

## 366. PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training

**arXiv ID:** 2606.03264 | [PDF](https://arxiv.org/pdf/2606.03264v1)

**作者:** Zelun Zhang `[一作]` (Baidu Inc.), Yanjun Ma `[通讯]` (Baidu Inc.)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在保持0.9B规模的前提下，通过针对性的数据工程与分阶段后训练（CPT‑SFT‑RL）提升PaddleOCR‑VL‑1.6的文档解析能力。

**💡 创新点**

创新点包括：①基于模型误差诊断的Under‑Optimized Region Mining（边界脆弱、覆盖稀疏、监督不可靠）；②多专家共识+渲染引导的自动标注管道；③针对小模型的GRPO高潜力样本筛选与分布式奖励设计；④统一的分阶段后训练流程。

**🔧 技术方法**

技术手段主要有：深度视觉语言模型（ERNIE‑4.5‑0.3B+Native Resolution Visual Encoder）、不确定性采样、语义邻域聚类、专家模型校验、ITERATIVE Judge‑and‑Refine、CPT‑SFT‑RL（含GRPO）和多任务可验证奖励函数。

**📊 数据集**

使用的数据集包括OmniDocBench v1.6、Real5‑OmniDocBench、In‑house‑Table、In‑house‑Chart、In‑house‑Text‑Spotting、In‑house‑Seal以及PaddleOCR‑VL‑1.5训练集的扩充与重标注。

**📈 对比分析**

与多类通用VLM（Qwen3‑VL‑235B、Gemini‑3 Pro等）及专用文档VLM（Dolphin‑1.5、MinerU2.5‑Pro、GLM‑OCR等）对比，PaddleOCR‑VL‑1.6在OmniDocBench v1.6整体分数达到96.33%，表格、公式、文本等子指标均实现了显著提升；在Real5‑OmniDocBench的5大子任务中也保持首位，整体得分93.19%。

**⚠️ 局限性**

局限性：1）在极端稀有场景或极高噪声图像上仍可能出现边界不稳或结构误判；2）RL阶段提升有限，需更多高质量奖励样本；3）数据工程与多专家标注成本高，难以快速迭代。

---

## 367. GPU-Parallel Multi-Task Reinforcement Learning with Demonstration Guided Policy Optimization

**arXiv ID:** 2606.03335 | [PDF](https://arxiv.org/pdf/2606.03335v1)

**作者:** Rui Zhang `[一作]` (NVIDIA), Weihua Zhang `[通讯]` (NVIDIA)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了可在GPU上并行训练的多任务机器人学习基准 Libero，并提出了基于演示的PPO优化方法 DGPO。

**💡 创新点**

创新点是将任务语义与 GPU 执行分离，实现一次训练覆盖多种结构化操纵任务；同时将重要性加权 PPO 与自适应行为克隆结合，利用演示数据指导学习而不失在线改进。

**🔧 技术方法**

技术包括 Isaac Lab GPU 并行模拟、任务描述器、稀疏奖励、密集演示跟踪奖励、IW‑PPO、Adaptive BC、向量化训练循环。

**📊 数据集**

使用 Libero 结构化操纵任务集（40 个任务）和每个任务 50 条 MuJoCo 演示轨迹。

**📈 对比分析**

在与 MT‑PPO、MT‑DeepMimic、MT‑RLPD、MT‑RFCL、MT‑DAPG 等基线对比的实验中，DGPO 在状态输入和视觉输入下均显著提升任务成功率，尤其在 Long 任务中表现突出。

**⚠️ 局限性**

局限在于仅针对固定机器人、桌面环境和有限物体词汇；强演示压力可能降低演示范围内的泛化；以及对接触协调、长时序误差、动力学等挑战仍不足。

---

## 368. CAPER: Clause-Aligned Process Supervision for Text-to-SQL

**arXiv ID:** 2606.03327 | [PDF](https://arxiv.org/pdf/2606.03327v1)

**作者:** Lujie Ban `[一作]` (Chinese University of Hong Kong), Chenhao Ma `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 1141 | [OpenAlex ID](https://openalex.org/A5055857919)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种自动化框架，利用对SQL AST 的因果干预与错误注入，生成子句级别的偏好监督标签，并以此训练轻量级的 Clause‑PRM 作为过程奖励模型，用于强化学习的 Text‑to‑SQL 策略优化和候选 SQL 验证。

**💡 创新点**

创新点在于将子句作为语义上最合适的监督粒度，通过对 AST 的自动干预定位根本错误，解决了传统终端奖励稀疏和 token 级别监督过细的问题，并在生成过程中实现精确的根因归因。

**🔧 技术方法**

主要技术包括：AST 结构编辑与因果干预、故障注入（synthetic failure generation）、子句级偏好学习、Clause‑PRM 的边界奖励设计，以及 GRPO 强化学习框架，配合轻量化的过程奖励网络。

**📊 数据集**

实验使用了 BIRD、Spider 和 SYNSQL‑5k 三大数据库构建的 SQL 生成任务，生成约 90,000 条子句级别偏好标注数据。

**📈 对比分析**

与稀疏执行奖励、token 级奖励以及多种闭源/开源 LLM 的基线对比，Clause‑PRM 在 BIRD 和 Spider 上相较 GPT‑5.4 提升约 15.3% 的执行准确率；在失败定位上实现 84.53% 的 Top‑1 正确率和 90.60% 的 MRR；在候选验证上相对 Majority Vote 提升 0.7–1.6 点。

**⚠️ 局限性**

局限性包括：对 AST 结构的映射和编辑仍需手工或预训练支持；对极端语法错误或深层子查询的鲁棒性有限；在更大模型或跨域场景的泛化尚未充分验证。

---

## 369. Validation-Gated Multi-Agent Governance for Online Adaptation of Thermal-Hydraulic Surrogate Models under Operating-Regime Shift

**arXiv ID:** 2606.03321 | [PDF](https://arxiv.org/pdf/2606.03321v1)

**作者:** Doyeong Lim `[一作]` (Ulsan National Institute of Science and Technology), In Cheol Bang `[通讯]` (Ulsan National Institute of Science and Technology)

**通讯引用:** 6252 | [OpenAlex ID](https://openalex.org/A5071219105)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种基于多代理治理的受控持续适应框架，用于在核热工循环实验数据上进行实时状态反馈预测。

**💡 创新点**

创新点在于将监测、诊断、适配、安全审计和指挥等多角色代理与确定性安全门控结合，实现在线模型升级、可审计、可回滚的自适应系统。

**🔧 技术方法**

技术包括：7种模型族的离线筛选（LSTM/GRU/Transformer/神经ODE/图神经网络/DeepONet/temporal-FNO），冠军–挑战者持续学习机制，漂移检测、有限 fine‑tune、背景训练、门控推广，以及基于 LLM 的代理规划与多代理协同推理。

**📊 数据集**

数据集：UNIST 热工循环实验数据，包含 8,410 步的离线训练集（正常运行、功率下降、SBO）和 2,701+3,185 步的两条 held‑out 流式测试集（热管插入配置和未改装的其他工作转移）。

**📈 对比分析**

对比方法：在七种运行模式（Static、Rule-H、Shadow、Single-H、Single-Full、MA-H、MA-Full）下进行实验。MA‑Full 模式取得最佳效果，平均 MAE 5.72±2.39，较 Static 提升 19%，警告超限率下降至 35.8%。Rule‑H 也比 Static 改进 7.3%。Shadow 与单代理模式效果不显著。使用配对自助法置信区间，MA‑Full 与 Static 的差值显著不为零。

**⚠️ 局限性**

局限性：仅使用两条 held‑out 流和有限种子，样本量小；未证明在所有运行状态下的普适性；缺少与传统连续学习基线的直接对比；代理诊断与实际操作事件关联不足；安全门控参数需经验调优。

---

## 370. The Security Budget of Code LLMs: An Information-Theoretic Capacity-Security Bound

**arXiv ID:** 2606.03308 | [PDF](https://arxiv.org/pdf/2606.03308v1)

**作者:** Jianwei Tai `[一作]` (Anhui University), Jianwei Tai `[通讯]` (Anhui University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5110952853)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了代码生成LLM在功能容量（Cap）与对抗扰动保留（Sec）之间的互信息上界，并给出了可在无模型访问的情况下计算的任务熵上限与提示泄漏度量。

**💡 创新点**

创新点包括：① 提出Cap‑Sec信息理论界（Cap+Sec ≤ 任务熵 + 提示泄漏），② 结合gzip压缩给出任务熵闭式上限，③ 设计输出仅嵌入协议与上下文混合余弦对齐信号，④ 在多模型、多量化、多数据集上进行经验验证与压力测试。

**🔧 技术方法**

使用技术：信息理论与数据处理不等式、MINE/KSG 互信息估计、PCA 降维、gzip源编码、提示扰动池（synonym、negation、comment、identifier、security‑anti）与PGD梯度攻击、量化（INT4/NF4、BF16）以及自动编码器隐藏层提取。

**📊 数据集**

主要使用数据集：HumanEval、MBPP‑sanitized（257题），并对SecurityEval做兼容性验证；所有实验均基于CodeLlama‑7B‑Instruct 与 Qwen2.5‑Coder‑7B‑Instruct 两个模型。

**📈 对比分析**

比较方法：在七种配置（不同模型、量化、嵌入方式）下计算 Cap、Sec 与提示泄漏的互信息，并检查嵌入变量满足 Cap‑Sec 边界（饱和度0.27–0.92，正 slack）。此外对齐余弦与 pass@1 的相关性被用作每题性能评估，压力测试（23‑扰动池、通用后缀、PGD）均未突破边界，证明模型稳健。

**⚠️ 局限性**

局限性：MINE/KSG 估计器在小样本/近似确定性场景下易偏大；输出仅向量压缩可能忽略上下文信息；提示扰动池有限，未覆盖所有可能攻击；量化会影响 Cap 与 Sec 的绝对值，但相对比例保持；对齐余弦与 Cap‑Sec 边界无直接因果关系，需进一步验证。

---

## 371. From Script to Semantics: Prompting Strategies for African NLI

**arXiv ID:** 2606.03304 | [PDF](https://arxiv.org/pdf/2606.03304v1)

**作者:** Anuj Tiwari `[一作]` (Noida Institute of Engineering and Technology), Hannah Nwokocha `[通讯]` (ML Collective)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估了五种零样本提示策略在AfriXNLI基准上对斯瓦希里语、约鲁巴语和豪萨语的自然语言推理性能影响。

**💡 创新点**

创新之处在于聚焦低资源非洲语言的提示设计对类别行为和预测稳定性的影响，并揭示对比提示在减轻中立类崩溃方面的优势。

**🔧 技术方法**

采用了零样本提示、语言特定提示、脚本感知提示、对比提示和自译提示等技术，使用Llama3.2-3B与Gemma3-4B两款中等规模开源模型进行推理。

**📊 数据集**

使用了AfriXNLI多语言推理数据集，其中每种语言包含600个样本，均衡分布在蕴含、矛盾和中立三类。

**📈 对比分析**

通过准确率、宏F1及每类F1进行评估，并与加入少量示例和链式推理的基线对照，结果表明对比提示在多语言和多模型中实现了最稳健的类别平衡和最高宏F1，且在部分情形下超越传统增强基线。

**⚠️ 局限性**

局限包括仅覆盖三种语言、只使用AfriXNLI单一基准、模型规模有限以及缺乏实例级错误分析。

---

## 372. Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility

**arXiv ID:** 2606.03291 | [PDF](https://arxiv.org/pdf/2606.03291v1)

**作者:** Chaoyi Xiang `[一作]` (University of Melbourne), Lea Frermann `[通讯]` (University of Melbourne)

**通讯引用:** 864 | [OpenAlex ID](https://openalex.org/A5025156794)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究多语言LLM的无学习（unlearning）效果，系统评估不同语言之间的转移与可恢复性。

**💡 创新点**

首次在多语言环境下量化无学习跨语言转移规律，并发现无学习行为更像抑制而非完全删除，可通过激活层的 steering 向量恢复已遗忘信息。

**🔧 技术方法**

使用 DPO、GA、NPO 三种无学习目标，对 Qwen2.5-7B 与 Gemma2-9B 进行微调，并利用跨语言 NLI 评估和隐藏表示（余弦相似度、PCA、层级激活向量）进行机制分析。

**📊 数据集**

基于 TOFU 框架，将其翻译为中文、德语、俄语、土耳其语等 5 种语言，构成跨语言事实问答数据集。

**📈 对比分析**

通过 NLI 分数对比显示：无学习在共享脚本/语系相近语言间转移更强；对齐层保持一致，抑制集中在后期层；利用 steering 向量可恢复 50%（Qwen）/90%（Gemma）遗忘知识。

**⚠️ 局限性**

局限性在于仅针对少量事实（1%）的微调无学习，未覆盖复杂推理或预训练知识，且实验仅在两款模型与三种无学习方法上进行，结果可能不适用于更大规模或不同架构。

---

## 373. AI-Generated Traces for Novice Programmers: Learning Effects and Learner Differences in a Multi-Institutional Study

**arXiv ID:** 2606.03288 | [PDF](https://arxiv.org/pdf/2606.03288v1)

**作者:** Yuri Noviello `[一作]` (Delft University of Technology), Gosia Migut `[通讯]` (Delft University of Technology)

**通讯引用:** 28 | [OpenAlex ID](https://openalex.org/A5047848559)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并评估了 AI 生成的动画跟踪（GAT）与文本说明的对照实验，探究其对初学者编程学习的即时与长期效果以及学习体验与参与度的影响。

**💡 创新点**

创新点在于利用大型语言模型（LLM）自动生成对比示例、构建动画、语音解说，并将代码、内存状态与概念类比同步呈现，提供可扩展的多模态教学资源。

**🔧 技术方法**

技术实现包括：LLM（用于脚本与解释生成）、Manim（用于动画渲染）、自动字幕与语音合成；实验采用随机对照、线性混合模型、ANCOVA 与 k‑means 聚类等统计方法。

**📊 数据集**

数据集来源于两所高校的 CS1 课程：Delft（Java，N≈151）和多伦多（Python，N≈961），包含 5 主题（while、ArrayList、HashMap、文件读写）交互式实验与最终考试成绩。

**📈 对比分析**

与匹配文本说明相比，GAT 在 Java 课程的即时学习成绩显著提升（尤其是 while 题），但在 Python 课程无显著差异；长期考试成绩无显著变化；在学习体验上两组差异极小；通过聚类发现低/高参与度学生受益，正中间参与度学生略受不利。

**⚠️ 局限性**

局限性包括：实验条件不同（语言、激励、主题覆盖）、仅覆盖有限主题、即时效果未转移到长期考核、参与度分析为探索性且基于自评、对自适应教学未做进一步验证。

---

## 374. EaDex: A Cross-Embodiment Dexterous Manipulation Framework from Low-Cost Demonstrations

**arXiv ID:** 2606.03268 | [PDF](https://arxiv.org/pdf/2606.03268v1)

**作者:** Qian Zhao `[一作]` (Northeastern University), Yingtian Li `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9504 | [OpenAlex ID](https://openalex.org/A5069114926)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出EaDex框架，利用单RGB‑D相机低成本采集手部演示，并通过动态演示退化机制引导强化学习实现多体型手臂的多关节物体操纵

**💡 创新点**

创新点在于：①低成本演示采集与ARCTIC格式统一化；②基于接触奖励的动态演示退化策略；③跨体型手臂的统一演示重映射与学习管线

**🔧 技术方法**

采用MANO手模型拟合、ARCTIC格式数据标准化、Gaussian平滑、PPO强化学习、任务/模仿/接触奖励组合，并实现接触奖励驱动的演示权重退化

**📊 数据集**

使用自制低成本RGB‑D演示数据集（3种手臂×3种关节物体，共9个任务）以及公开ARCTIC数据集进行验证

**📈 对比分析**

与不使用演示退化的基线相比，低成本数据集上平均成功率从23.5%提升至36.5%（相对提升55.3%）；在ARCTIC上多任务成功率亦显著提高

**⚠️ 局限性**

局限在于单摄像头下手势遮挡导致关键点缺失、腕部运动受限；未来需多摄像头融合以提升演示完整性

---

## 375. ReforMe: Re-Shaping Documents with Contextual Prompting and Layout-Aware Propagation

**arXiv ID:** 2606.03266 | [PDF](https://arxiv.org/pdf/2606.03266v1)

**作者:** Nabin Khanal `[一作]` (Purdue University), Yingjie Victor Chen `[通讯]` (Purdue University)

**通讯引用:** 2417 | [OpenAlex ID](https://openalex.org/A5100409186)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一个交互式文档数字化系统 ReforMe，融合布局解析、OCR 与 LLM 重构，并提供直接编辑、自然语言指令、结构相似传播与可逆历史记录，形成从扫描到可编辑结构化文档的协同修订流程。

**💡 创新点**

创新点在于将 LLM 与布局感知结合，提出基于结构相似性的传播机制和混合人机指令交互，使错误校正、结构重塑与批量传播可由用户轻松完成。

**🔧 技术方法**

技术上采用 Swin Transformer V2+DETR 进行布局检测、Tesseract/TrOCR 等 OCR 引擎、LLM（如 ChatGPT/Claude）进行结构化重构，并通过 React+Django 搭建前后端交互。

**📊 数据集**

使用 PubLayNet、PubTabNet、PubTables-1M 等公开文档与表格数据集训练模型，并在本研究中以真实档案扫描文档（手写、表格、图形等）进行评估。

**📈 对比分析**

通过 12 名参与者的受控实验对比基线 LLM 聊天流程，结果显示 ReforMe 在初始提取、批量修正、结构重塑任务上均显著提升效能，且在用户满意度、信任度等指标上明显优于基线。

**⚠️ 局限性**

局限包括传播准确性不稳定、对大型文件处理延迟、对非标准扫描质量的鲁棒性不足，以及缺乏在更广泛文档领域的泛化验证。

---

## 376. Evaluating LLMs' Effectiveness on Real-World Consumer Device Repair Questions

**arXiv ID:** 2606.03331 | [PDF](https://arxiv.org/pdf/2606.03331v1)

**作者:** Atm Mizanur Rahman `[一作]` (University of Illinois Urbana Champaign), Sharifa Sultana `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了991条来自Reddit的消费设备维修问题与专家解答的多语言（英文+孟加拉文）基准，并使用LLM-as-judge对六款大型语言模型在电话、电脑、数据恢复三类维修任务中的表现进行评估。

**💡 创新点**

创新点包括：①首个面向多语言的消费设备维修基准；②设计了涵盖技术准确性、完整性、可操作性和安全性的四维评估框架；③系统比较不同模型在三域、多语境下的可靠性与安全性。

**🔧 技术方法**

技术手段：大语言模型推理、LLM-as-judge自动评分、人工翻译与专家验证、Likert尺度与二值安全评估。

**📊 数据集**

数据集：991条Reddit维修问答（电话、电脑、数据恢复），每条配有技术员写的参考答案，并提供孟加拉文翻译版本。

**📈 对比分析**

比较方法：对每个模型、语言、域在四维度上评分并取平均；最高性能模型在英文数据恢复上获得0.84分；整体上英文优于孟加拉文，电话维修是最难、最安全敏感的领域。

**⚠️ 局限性**

局限性：基准仅覆盖三类设备，未囊括其他家电；孟加拉文为人工翻译，可能不完全自然；LLM-as-judge可能存在偏差，缺乏人工专家复核；仅评估单轮回答，未模拟交互式故障排查。

---

## 377. RobotValues: Evaluating Household Robots When Human Values Conflict

**arXiv ID:** 2606.03312 | [PDF](https://arxiv.org/pdf/2606.03312v1)

**作者:** Jongwook Han `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 4679 | [OpenAlex ID](https://openalex.org/A5016844435)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并发布了一个名为 RobotValues 的基准，用于评估家用机器人在价值冲突情境中的决策能力。

**💡 创新点**

创新点在于：①以图像为基础生成真实家庭情境；②通过利益相关者的反应提取细粒度价值标签；③设计了多阶段 LLM‑辅助生成与过滤流程；④将细粒度价值映射到既有价值范畴（家用机器人规范与 Schwartz 价值体系）。

**🔧 技术方法**

主要技术包括：大型语言模型（LLM）用于情境、动作、利益相关者反应与价值提取；生成式图像模型（GPT‑Image 2）用于创建第一人称视角的家庭图像；LLM 判别器（GPT‑5‑mini）做二元质量控制；以及基于 Bradley‑Terry 的偏好评分与准确率评估。

**📊 数据集**

数据集：RobotValues，包含 10,073 张家庭场景图像、文本任务上下文、69,134 个候选动作以及利益相关者驱动的价值注释；数据来源于 LLM 辅助生成，后经多阶段过滤。

**📈 对比分析**

比较方法：在默认选择与价值条件选择两种任务下，对 10 种机器人导向 VLM（如 Qwen3‑VL、Cosmos‑Reason、Molmo、InternVL、RLDX）进行评估。结果显示模型默认偏好安全与适应性，隐私与安全得分最低；在价值条件任务中，若目标价值与模型默认偏好冲突，准确率下降 30–40%，表明模型难以按照显式价值指令行事。

**⚠️ 局限性**

局限性：使用合成图像，可能无法完全反映真实家庭的视觉复杂度与传感噪声；LLM 生成与标注过程中仍可能存在伪影或注释错误，尽管已通过多阶段过滤减轻。

---

## 378. FLIPS: Instance-Fingerprinting for LLMs via Pseudo-random Sequences

**arXiv ID:** 2606.03330 | [PDF](https://arxiv.org/pdf/2606.03330v1)

**作者:** Gurvan Richardeau `[一作]` (Université de Rennes, Inria, CNRS/IRISA), Gilles Tredan `[通讯]` (LAAS, CNRS)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

识别并区分同一大语言模型在不同配置（如温度、系统提示、量化、删减安全行为）下的具体实例行为，支持监管审核；

**💡 创新点**

提出Instance-level Fingerprinting（IF）范式，利用模型在生成伪随机二进制序列时的偏差，通过NIST随机性测试统计作为特征，实现对实例级差异的敏感识别；

**🔧 技术方法**

核心技术包括构造固定token对的查询模板、收集模型生成的二进制序列、提取NIST随机性测试统计特征，并以XGBoost分类器实现提取与验证阶段；

**📊 数据集**

使用由25款公开权重LLM构成的237个实例集合，涵盖4种温度、4种系统提示、4种量化方式及若干abliterated版本，生成500条二进制序列做特征提取；

**📈 对比分析**

在闭集与开放集下与已适配的LLMmap基线对比，IF在闭集可达96%准确率、开放集90%准确率（仅8次验证查询），明显优于LLMmap的35%以下表现；

**⚠️ 局限性**

局限性包括仅针对诚实实例；对抗性查询重定向或高级随机生成技术、模型主动混淆、以及对极端随机性提升的鲁棒性尚未充分验证。

---

## 379. Ollivier-Ricci curvature in cycle overlap mode

**arXiv ID:** 2606.03317 | [PDF](https://arxiv.org/pdf/2606.03317v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 380. InfoMem: Training Long-Context Memory Agents with Answer-Conditioned Information Gain

**arXiv ID:** 2606.03329 | [PDF](https://arxiv.org/pdf/2606.03329v1)

**作者:** Tiancheng Han `[一作]` (Tongji University), Wenqi Shao `[通讯]` (Shanghai Innovation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 InfoMem，一种用于分块长文本记忆代理的奖励塑造方法，利用答案条件信息增益评估最终记忆的有效性；

**💡 创新点**

创新点在于：① 直接对最终记忆进行答案条件信息增益奖励；② 只在成功轨迹上给出该奖励；③ 在奖励组合前对增益进行归一化；

**🔧 技术方法**

技术细节包括：使用 GRPO（Group Relative Policy Optimization）进行强化学习；以答案条件下的对数似然差值作为信息增益奖励；在 Qwen2.5‑1.5B‑Instruct 上实现分块记忆代理；

**📊 数据集**

训练数据采用 RULER‑HotpotQA（精简至512例）；评估数据包含 MRCR‑8needle、RULER synthetic QA、CorpusQA、LongMemEval，以及用 SQuAD 生成的合成幻觉证据；

**📈 对比分析**

与基线（Outcome‑only GRPO、ReMemR1）对比，InfoMem 在所有四个长文本基准上均实现最高得分，尤其在检索导向任务上显著提升；在合成辨别实验中，信息增益奖励的 MRR 与 SNR 远超嵌入相似度与注意力分数；

**⚠️ 局限性**

局限性：训练样本有限、模型规模较小；奖励仅在最终步骤计算，未覆盖中间记忆；仅适用于分块记忆代理，无法直接推广到全上下文或检索‑仅模型；过度优化答案可能导致错误信息被强化。

---

## 381. The Violation Situation Pattern: A Knowledge-Graph Pattern for Compliance Violations

**arXiv ID:** 2606.03326 | [PDF](https://arxiv.org/pdf/2606.03326v1)

**作者:** Nima Kamali Lassem `[一作]` (DiliTrust), Seyid Amjad Ali `[通讯]` (Bilkent University)

**通讯引用:** 521 | [OpenAlex ID](https://openalex.org/A5101953978)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了“Violation Situation Pattern（VSP）”，将合规违规事件持久化为知识图谱节点，并为其提供生命周期管理与审计日志；

**💡 创新点**

创新点在于将情境模式与复合身份、五态生命周期、不可变审计日志以及多证据关系融合为单一模式，使违规事件在图谱中实现持续跟踪与完整审计；

**🔧 技术方法**

采用FCL（Formal Contract Logic）规则→Cypher查询→MERGE构建VSP实例，并结合PROV‑O、SHACL、属性图技术实现规则执行、生命周期状态机和审计日志；

**📊 数据集**

使用了BODACC公司官员公告、GDPRhub执法案例、合同与条款的法定与合规数据、以及GDPR同意示例等多来源数据集进行验证；

**📈 对比分析**

通过对BODACC的一致性检查、73个GDPRhub案例的F1提升（0.312→0.602）、SHACL跨形式检查的相同触发以及GDPR同意域迁移验证，展示了规则变更后模式保持审计连续性的能力，性能方面日志线性增长可通过索引扩展；

**⚠️ 局限性**

局限在于规则层面未覆盖文本推理（如V4未检测72h期限违规）、系统层面依赖应用层中介，且并发与大规模性能未评估，提取质量对精度影响仍待进一步验证。

---

## 382. Multi-Modal Graph Neural Network with Transformer-Guided Adaptive Diffusion for Preclinical Alzheimer Classification

**arXiv ID:** 2606.03322 | [PDF](https://arxiv.org/pdf/2606.03322v1)

**作者:** Jaeyoon Sim `[一作]` (Pohang University of Science and Technology), Won Hwa Kim `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 2516 | [OpenAlex ID](https://openalex.org/A5101424026)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `70e40602-aae3-44bd-80ec-4a7f2674330f` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

提出了一种多模态图神经网络，结合Transformer引导的自适应扩散，用于预临床阿尔茨海默病分类并可解释关键脑区。

**💡 创新点**

通过可学习的节点尺度热扩散卷积与多头自注意力实现短程局部与远程全局信息分离学习，且模型输出节点尺度与注意力可解释性。

**🔧 技术方法**

使用热扩散卷积、可学习节点尺度、Transformer自注意力、多模态融合、交叉熵+L1正则、5折交叉验证等技术。

**📊 数据集**

使用ADNI 919名预临床阿尔茨海默病受试者的DWI、MRI、PET（FDG、Amyloid）产生的脑网络与ROI特征。

**📈 对比分析**

与GCN、GAT、GraphHeat、GDC、ADC、LSAP、NodeFormer、DIFFormer、SGFormer等9个基线在多模态组合和三类分类任务上进行5折交叉验证，平均准确率超过96%，标准差低。

**⚠️ 局限性**

仍受限于图结构假设、未在临床诊断中验证、需要更多模态与更大样本检验，且对节点尺度正则参数敏感。

---

## 383. A Novel Detection Method for Single-RF MIMO-OFDM Systems

**arXiv ID:** 2606.03311 | [PDF](https://arxiv.org/pdf/2606.03311v1)

**作者:** Tianrui Qiao `[一作]` (Hong Kong University of Science and Technology), Ross Murch `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 16485 | [OpenAlex ID](https://openalex.org/A5004541948)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文针对单射频（Single‑RF）MIMO‑OFDM系统在高功率下出现的误差极限（error floor）问题进行系统分析，并提出了一种基于马氏距离（Mahalanobis distance）的最大似然（ML）检测方法，以显著提升比特误码率（BER）性能；

**💡 创新点**

创新点在于：①首次将马氏距离引入ML检测，利用其对噪声相关性的建模实现对ESPAR产生的模型误差的去相关与白化；②通过与MMSE检测的对比，证明该方法兼具MMSE的鲁棒性与ML的判决精度；③系统化地阐述了单RF MIMO‑OFDM误差极限的本质，并给出了针对性的解决方案；

**🔧 技术方法**

使用技术包括：最大似然检测、马氏距离、Beamspace MIMO‑OFDM框架、ESPAR（电子可调谐谐振器）天线模型、代码簿量化、MMSE检测、CST仿真获得的天线阻抗与开路辐射模式、频域/时域信道建模（L‑tap Rayleigh），以及统计分析与误差功率估计；

**📊 数据集**

数据集：通过MATLAB/Simulink对2×2 Beamspace MIMO‑OFDM系统进行仿真，采用L=3 taps、功率谱均匀、XPD=1、K=12子载波、QPSK调制；利用8位与12位量化的代码簿，对信号做1 000 000次独立随机试验，统计模型误差功率与BER；

**📈 对比分析**

比较方法：将提出的马氏距离ML检测与传统欧氏距离ML检测及MMSE检测在同一仿真环境下进行BER曲线对比；结果显示：传统ML检测在高SNR出现明显误差极限；MMSE检测虽消除误差极限但仍受模型误差影响；马氏距离ML检测能有效抑制误差极限，在Q=8时可节省约5 dB E_b/N_0达到BER 10⁻³；随着量化位数提升，性能进一步逼近理想双天线MIMO‑OFDM系统；

**⚠️ 局限性**

局限性：仅在理想CSI、平稳Rayleigh信道及特定QPSK调制下验证；未考虑实际硬件非理想与信道估计误差；仅针对ESPAR天线进行验证，其他可重构天线如像素天线的适用性需进一步研究；量化产生的模型误差虽减小但仍未完全消除；

---

## 384. Learning Multi-Scale Hypergraph for High-Order Brain Connectivity Analysis

**arXiv ID:** 2606.03310 | [PDF](https://arxiv.org/pdf/2606.03310v1)

**作者:** Jaeyoon Sim `[一作]` (Pohang University of Science and Technology), Won Hwa Kim `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 2516 | [OpenAlex ID](https://openalex.org/A5101424026)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出了一种自适应多尺度超图学习框架，用于捕捉脑网络中的高阶关系，提升神经退行性疾病分型精度。

**💡 创新点**

创新点在于同时进行多尺度图小波滤波与动态超边学习，构建可学习的、连续且尺度感知的超图结构。

**🔧 技术方法**

使用图小波变换、超图卷积、Transformer以及自适应尺度参数进行端到端训练。

**📊 数据集**

在ADNI（结构与功能多模态）和PPMI（fMRI功能连接）两个公开数据集上验证。

**📈 对比分析**

与19种基线方法（传统GCN、GAT、超图模型等）进行5折交叉验证，对比显示准确率提升2–4%并在零样本迁移中保持优势。

**⚠️ 局限性**

局限在于对样本不平衡敏感、对罕见疾病阶段的鲁棒性不足，且学习到的超边仅为相关性而非因果。

---

## 385. Bridging Predictive Uncertainty and Safe Action: Sample-Conditioned Differentiable Planning for Autonomous Driving

**arXiv ID:** 2606.03296 | [PDF](https://arxiv.org/pdf/2606.03296v1)

**作者:** Chengzhen Meng `[一作]` (Hong Kong University of Science and Technology), Jun Ma `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 20302 | [OpenAlex ID](https://openalex.org/A5100357282)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于扩散模型样本的可微分规划框架，将多样化的预测轨迹直接用于安全规划。

**💡 创新点**

创新点在于将扩散生成的多样化未来轨迹与可微分优化结合，使用经验 CVaR 尾部风险约束实现不确定性感知规划，并采用有向图表示场景上下文提升效率。

**🔧 技术方法**

使用了条件扩散模型、可微分优化（Gauss‑Newton）、图注意力网络、经验 CVaR 约束等技术。

**📊 数据集**

在 Waymo Open Motion 和 Argoverse 2 两个大规模真实驾驶数据集上训练和评估。

**📈 对比分析**

相较于 IL、DIPP、GameFormer 等基线，开闭环实验中在碰撞率、轨迹误差、舒适度等指标上均实现显著提升，尤其在安全率和进程度上超过基线近 40%。

**⚠️ 局限性**

局限在于推理时扩散采样与迭代优化导致延迟，且预测多样性受限，未来需降低采样步数和优化迭代并提升分布多样性。

---

## 386. SEA-NLI: Natural Language Inference as a Lens into Southeast Asian Cultural Understanding

**arXiv ID:** 2606.03284 | [PDF](https://arxiv.org/pdf/2606.03284v1)

**作者:** Peerawat Chomphooyod `[一作]` (Chulalongkorn University), Peerat Limkonchotiwat `[通讯]` (Chulalongkorn University)

**通讯引用:** 197 | [OpenAlex ID](https://openalex.org/A5038217558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并发布SEA-NLI，一个涵盖东南亚八国八语、具备文化知识的自然语言推理基准；

**💡 创新点**

首次实现多语种、文化扎根的NLI数据集，并通过LLM生成+人工审核+多轮过滤构建高质量样本，同时评估SEA文化适配模型与提示方法；

**🔧 技术方法**

利用GPT‑5.2进行文本生成，结合词汇与语义过滤、重生成、SEA‑LION/Gemma‑SEA‑LION等文化适配模型以及文化感知提示技术；

**📊 数据集**

构建的SEA‑NLI（1,443正常+717硬样本，10文化类别，8国8语），对照SNLI、MNLI、XNLI、IndoNLI、ViNLI、Myanmar XNLI、CALI等公开数据集；

**📈 对比分析**

在SEA‑NLI上采用加权F1评估17个编码器/解码器模型；结果显示硬集性能平均下降约11–13%，SEA语言表现更差；SEA适配模型比基线提升2–6个百分点；

**⚠️ 局限性**

仅覆盖八国八语，未囊括所有方言、口语与代码混写；人工审核成本高；生成过程可能仍存在偏差，模型仍缺乏充分的SEA文化知识导致错误。

---

## 387. Causal Evidence of Stack Representations in Modeling Counter Languages Using Transformers

**arXiv ID:** 2606.03398 | [PDF](https://arxiv.org/pdf/2606.03398v1)

**作者:** Nishit Singh `[一作]` `[通讯]` (Birla Institute of Technology and Science), Nishit Singh (Birla Institute of Technology and Science)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文研究了在训练于形式语言（Dyck-1 与 Shuffle-k）上的 transformer 模型中堆栈表示的因果作用，先用线性探针预测堆栈深度，然后在推理阶段对对应的隐藏向量方向进行消融，观察模型准确率的崩塌；

**💡 创新点**

创新点在于首次通过干预（激活补丁）验证堆栈表示的因果必要性，而非仅仅进行表征分析，展示消融堆栈方向导致预测性能彻底失效。

**🔧 技术方法**

采用线性探针分类器、奇异值分解提取主方向、隐藏状态方向消融（α 参数控制强度）、单层因果注意力 transformer 以及 MSE 损失训练。

**📊 数据集**

使用 Shuffle‑k（k=2,4,6,8）数据集，构造自 Dyck‑1 的 k 组括号交错串，共 10,000 条长度为 2–50 的序列，训练/验证划分 80/20。

**📈 对比分析**

对比方法为：消融堆栈方向 vs 随机方向；评估指标为位置准确率与序列准确率；结果显示针对堆栈方向的消融导致位置准确率显著下降、序列准确率完全崩塌，而随机方向消融无影响，证明堆栈表示在模型中起决定性作用。

**⚠️ 局限性**

局限性包括：只在推理时对最终隐藏状态进行消融，未在网络中部位或重跑网络后观察变化；仅评估单层 transformer 的最终解码器输出；实验仅限于形式语言，缺乏对更通用自然语言的验证；未探究堆栈表示的冗余性或其他架构的表现。

---

## 388. Human-AI Collaboration and the Transformation of Software Engineering Work

**arXiv ID:** 2606.03394 | [PDF](https://arxiv.org/pdf/2606.03394v1)

**作者:** Mamdouh Alenezi `[一作]` `[通讯]` (Saudi Data and Artificial Intelligence), Mamdouh Alenezi (Saudi Data and Artificial Intelligence)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对生成式AI与代理式AI在软件工程中的整合进行结构化解释性综合，并提出未来软件工程师的能力框架。

**💡 创新点**

提出三种共存范式、以能力为核心的未来软件工程师框架以及可检验的九条理论命题。

**🔧 技术方法**

使用解释性综合方法、主题编码与框架映射，结合定性与大规模数据分析。

**📊 数据集**

基于精选的同行评审与档案研究，包括AIDev开放源码代理拉取请求数据、工作坊、问卷等。

**📈 对比分析**

通过比较传统、生成式AI与代理式AI在十维度上的差异，并用案例数据验证速度与合并率的关系，发现速度提升未必带来更高质量。

**⚠️ 局限性**

局限在样本偏向开源与研究机构、缺乏纵向实验、术语演进导致概念不确定，且结论依赖观察性证据。

---

## 389. Emerging and established topics in drone research: Citation impact and knowledge flows across China, the United States, the EU, Ukraine, and Russia (2020-2025)

**arXiv ID:** 2606.03362 | [PDF](https://arxiv.org/pdf/2606.03362v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 390. eMEM: A Hybrid Spatio-Temporal Memory System For Embodied Agents

**arXiv ID:** 2606.03374 | [PDF](https://arxiv.org/pdf/2606.03374v1)

**作者:** A. Haroon Rasheed `[一作]` (Inria), Maria Kabtoul `[通讯]` (Inria)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5014265726)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

为具身代理开发了一种结合时间、空间和语义检索的混合图形记忆系统eMEM，并构建了面向认知心理学范式的eMEM‑Bench v1基准，评估该系统在不同记忆功能上的表现。

**💡 创新点**

创新点在于：①将记忆统一为四种节点（观察、情节、概要、实体）并通过SQLite、HNSW和R‑tree三种索引实现语义、空间、时间三轴检索；②设计了多层感知、分层合并与分阶段归档的记忆巩固流水线，模拟大脑的CLS机制；③提供十个LLM工具接口，使高阶检索模式可通过工具组合实现；④推出基于ProcTHOR‑10K的八个认知心理学范式的基准，兼顾空间、多模态与长期延迟。

**🔧 技术方法**

技术包括：SQLite关系数据库、HNSW近似最近邻检索、R‑tree空间索引、DBSCAN聚类、LLM驱动的压缩摘要与实体提取、ReAct工具调用、基于JSON schema的LLM工具接口以及Python/SQLite/HNSW实现的内存引擎。

**📊 数据集**

使用的数据集是ProcTHOR‑10K场景（20个多房间环境）以及通过VLM、检测器、位置分类器生成的多层感知数据；Benchmark采样了包含DRM、模式分离/完成、源监测、情境检索、长期干扰、序列位置、保持曲线等八个认知范式的约988个提问。

**📈 对比分析**

通过与纯RAG基线进行消融实验，eMEM在情境检索、DRM误识别等方面分别提升了30分和29分；整体在所有范式上的加权平均得分为80.8（满分100），保持曲线在1小时至1年延迟内保持在上限，显示出强大的长期记忆保持能力。

**⚠️ 局限性**

局限主要在：①性能受LLM工具调用与推理模型质量影响，提升LLM能力可进一步提升分数；②巩固阶段对观察权重均等，未考虑奖励/惊奇；③缺乏对象相对坐标支持；④目前仅存储文本描述，未使用潜在表示或可迁移结构；⑤存档后对原始文本与向量的丢弃可能导致细节丢失。

---

## 391. A calculus of types in Isbell nuclei

**arXiv ID:** 2606.03369 | [PDF](https://arxiv.org/pdf/2606.03369v1)

**作者:** Juan Luis Gastaldi `[一作]` (ETH Zurich), John Terilla `[通讯]` (CUNY)

**通讯引用:** 163 | [OpenAlex ID](https://openalex.org/A5008162777)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文建立并比较了两种来自不同数学传统的类型生成方法：一种基于线性可实现性（linear realisability）中的执行与测量构造的正交闭包，另一种基于增广的Isbell对偶（enriched Isbell duality）中的测度诱导的核（nucleus）构造。研究显示，在相同的基础数据（一个集合C、可结合的执行积×、实值测量p）下，这两种构造产生完全相同的类型对象，并进一步由此导出了一个非交换的Lambek算术（带左右残余的可结合乘积）。

**💡 创新点**

创新点在于：
1) 将线性可实现性中的正交闭包与增广Isbell核对齐，证明两种构造在同一设置下等价；
2) 通过三元测度定义中间正交关系，构造中间类型并得到一个自然的Lambek算术；
3) 证明了三种“核”构造的两两一致性（两三法则）以及关联的“平衡”点，揭示了残余与核之间的双重关系；
4) 在有限实值情形下给出乘积闭包与“gap”矩阵的显式公式，将类型乘积与几何（凸体、度量）联系起来。

**🔧 技术方法**

技术手段包括：
- 增广范畴理论与加法扩展到[-∞, +∞]的闭序张量范畴；
- Isbell对偶构造与核的闭包运算；
- 线性可实现性中的正交关系与加权执行积；
- 三元测度与中间正交的推导；
- 证明残余运算与Lambek算术的相容性；
- 结合极大/极小化（inf/ sup）表达式实现几何解释；
- 通过显式例子验证非结合性、单位与残余性质。

**📊 数据集**

本研究为理论性工作，未使用外部数据集；所有示例均为人工构造的有限单子（如四元素非交换单子）和对应的实值测量，用以验证公式与定理。

**📈 对比分析**

由于论文纯粹为理论证明，没有实验对比或性能评估；主要通过示例展示定理的精确性和边界条件（如非结合性、两三法则的严密性）。

**⚠️ 局限性**

局限性包括：
- 需要在[-∞, +∞]张量范畴内工作，实值测量不一定满足额外的可加性或对称性；
- 结果主要适用于可结合执行积，若存在更一般的非结合执行结构尚未讨论；
- 在无限或非实值情形下，核与正交闭包的闭包操作可能不收敛，导致缺乏显式表示；
- 本文仅给出了Lambek算术在中间类型层的完整闭包结构，边界类型的组合性仍存在闭包不一致性；
- 对于实际计算或应用（如程序分析、资源管理），需要进一步研究算法实现与复杂度。

---

## 392. EntSQL: A Benchmark for Grounding Text-to-SQL in Long-Context Enterprise Knowledge

**arXiv ID:** 2606.03363 | [PDF](https://arxiv.org/pdf/2606.03363v1)

**作者:** Chengxi Liao `[一作]` (Hong Kong University Of Science And Technology), Zeyi Wen `[通讯]` (Hong Kong University Of Science And Technology)

**通讯引用:** 1216 | [OpenAlex ID](https://openalex.org/A5013127195)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个面向企业的 Text-to-SQL 基准  (Benchmark Name 未给出)，用于评估在长上下文的私有业务知识上生成 SQL 的能力。

**💡 创新点**

创新点在于：①将企业内部业务规则、报告惯例、财务定义等私有知识融入到问题-文档对中；②构造了包含 1,066 条中英双语实例、5 个业务域、平均 SQL 389 tokens 的真实企业数据；③通过将完整文档与精简证据进行对比，揭示了知识定位与 grounding 的瓶颈。

**🔧 技术方法**

主要技术包括：大型语言模型（Claude Opus 4.6、Sonnet 4.6、GPT‑5.4、Gemini 3.1 Pro、Qwen 3.6 Max、Kimi K2.6、GLM 5.1）以及基于 Claude Sonnet 的交互式编码代理（Claude Code），使用统一 Prompt、DDL 序列化、SQLite 执行验证以及执行准确率（EX）评估。

**📊 数据集**

使用的数据集为 1,066 条中英对齐的企业 BI 示例，覆盖财务、资金、业务管理、党务、人力资源 5 个域，包含 15 个数据库、35 张表、1,489 列，所有数据均已匿名化。

**📈 对比分析**

比较方法：在三种输入设置（仅问句、问句+全文文档、问句+精简证据）下，评估 8 个系统的执行准确率。性能表现：最佳系统在提供全文文档时的平均准确率仅 15.9%；在精简证据设置下可达 21.4%；单纯问句场景准确率低至 6.8%。

**⚠️ 局限性**

限制包括：①数据仅来自单一企业且仅覆盖 5 个业务域，可能不具备代表性；②仅支持 SQLite 后端，未覆盖多轮对话、权限与实时性需求；③中英对齐翻译与匿名化可能引入偏差，影响跨语言评估。

---

## 393. IDO: Incongruity-aware Distribution Optimization for Multimodal Fake News Detection

**arXiv ID:** 2606.03418 | [PDF](https://arxiv.org/pdf/2606.03418v1)

**作者:** Hengyang Zhou `[一作]` (Nanjing University), Zhaoyan Pan `[通讯]` (Zhejiang University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于不一致感知的分布优化框架 IDO，用来提升多模态假新闻检测的准确性。

**💡 创新点**

创新点在于同时建模事实不一致性和模态不一致性：①使用通道级加权与高斯分布建模事实不一致性；②引入连续对比学习捕捉模态不一致性，并将两类不一致性通过可训练的加权组合形成最终预测。该框架首次将不一致性概率建模与对比学习相结合，显著提升检测效果。

**🔧 技术方法**

技术包括：ViT + BERT 双模态编码；跨模态注意力对齐；通道级加权重构；高斯分布记忆池建模；连续对比学习（softmax-对齐标签 + KL 损失）；交叉熵与对比损失联合优化。

**📊 数据集**

使用公开的三大多模态假新闻数据集：Weibo、GossipCop、Weibo-21。

**📈 对比分析**

与现有 SOTA 方法（如 SpotFake、SAFE、MSACA、RaCMC、MIMoE‑FND 等）进行对比，IDO 在所有数据集上取得最高准确率（Weibo 94.7%、GossipCop 91.2%、Weibo‑21 96.3%）和 F1 分数，尤其在假新闻 F1 方面提升显著。

**⚠️ 局限性**

局限性：①需要维护并更新高斯记忆池，内存和训练成本较高；②对极端样本或跨域迁移的鲁棒性尚未深入验证；③对不一致性权重 α 的选择敏感，需经验调优。

---

## 394. A unified multi-task framework enables interpretable chest radiograph analysis

**arXiv ID:** 2606.03417 | [PDF](https://arxiv.org/pdf/2606.03417v1)

**作者:** Lijian Xu `[一作]` (Shenzhen University of Advanced Technology), Shaoting Zhang `[通讯]` (Chinese University of Hong Kong)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本研究提出了一种统一的多任务Transformer框架（IMT‑CXR），能够在一套模型中完成胸片疾病分类、病灶定位、解剖分割以及报告生成，实现从图像到文本的端到端可解释诊断流程。

**💡 创新点**

创新点在于将放射科医师的循证诊断思路拆分为三阶段（疾病识别、属性量化、证据整合），通过跨任务监督与指令调优实现多任务协同学习，并将中间证据（定位框、分割多边形）作为提示词嵌入报告生成，显著提升了报告的可解释性与临床可接受度。

**🔧 技术方法**

技术方案包括基于BART的编码‑解码Transformer、ResNet152图像编码器、BERT文本编码器以及跨模态注意力融合；所有任务共享同一序列化输出（标签、坐标、文本），并通过指令-图像-标签三元组进行联合训练。

**📊 数据集**

使用了MIMIC‑CXR、VinDr‑CXR、ChestX‑Det、CheXmask、SIIM‑ACR、JSRT等公开数据集，结合内部临床数据共计约0.65 M胸片，构建了8.8 M条指令‑标签对的多任务训练集。

**📈 对比分析**

在10个公开/私有基准（ChestX‑ray14、CheXpert、RSNA Pneumonia、MS‑CXR、JSRT、CheXMask等）上与20+专业模型比较，IMT‑CXR在多标签分类、定位、分割和报告生成等指标上均表现出色；临床盲评中66%AI生成报告被三位放射科医师评为与原报告同等或更佳。

**⚠️ 局限性**

局限性包括仅在160例回顾性多中心评估（缺乏罕见病表现）、跨任务训练数据不均衡导致的负迁移、未进行专门医学领域预训练、依赖顺序证据整合而非显式视觉推理、可能的错误传播以及对人口学特征缺乏充分考虑。

---

## 395. Operationalizing Cyber Attack Prediction: A Gap-Prioritized Framework with Dataset and Model Selection Guidelines

**arXiv ID:** 2606.03386 | [PDF](https://arxiv.org/pdf/2606.03386v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 396. Towards Characterizing Scientific Image Utility and Upgradability

**arXiv ID:** 2606.03401 | [PDF](https://arxiv.org/pdf/2606.03401v1)

**作者:** WenZhe Li `[一作]` (TongJi University), Guangtao Zhai `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了SIU^2A框架，用于评估科学图像的可用性和可升级性，并构建对应基准数据集；

**💡 创新点**

创新点在于将科学图像评估从单纯的感知质量提升到基于错误诊断和可修复性的双维度评估，首次系统化划分四类错误并引入结构化修复指令；

**🔧 技术方法**

采用多模态大型语言模型（MLLM）进行错误检测与描述，配合图像编辑模型实现修复，并使用BERTScore、LLM评估等技术进行评估；

**📊 数据集**

使用SIU^2A-Benchmark，包含约2100对基准科学图像与其四类人工生成的破损样本，专家标注错误可诊断性、可修复性以及纠正指令；

**📈 对比分析**

通过在SIU^2A框架下对比多款开源与闭源MLLM和图像编辑模型，实验显示当前模型在错误检测上相对成熟，但在生成结构化错误描述、修复指令及科学准确性恢复上性能仍低，闭源模型在视觉重建上领先但科学一致性不足；

**⚠️ 局限性**

局限性包括对错误描述质量的高度依赖、编辑模型对指令的敏感度差异、以及缺乏针对科学知识验证的专门生成机制，导致整体修复效果受限。

---

## 397. OpenEAI-Platform: An Open-source Embodied Artificial Intelligence Hardware-Software Unified Platform

**arXiv ID:** 2606.03392 | [PDF](https://arxiv.org/pdf/2606.03392v1)

**作者:** Jinyuan Zhang `[一作]` (Shanghai Innovation Institute), Nanyang Ye `[通讯]` (Shanghai Innovation Institute)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个完全开源的6-DoF低成本机器人手臂及其统一的视觉-语言-动作（VLA）学习平台。

**💡 创新点**

创新点在于整合可重复制造的机械设计、开放数据管道以及基于Qwen3-VL的两阶段预训练+微调策略，实现在全开放数据上与商业平台相当的性能。

**🔧 技术方法**

采用 Qwen3‑VL‑4B 作为视觉‑语言主干、Diffusion Transformer 行动头、FF‑PID 低级控制以及三点 Bézier 轨迹平滑等技术。

**📊 数据集**

预训练使用公开的 Open‑X‑Embodiment、COCO、VQA‑v2、PixMo‑Points 等多模态机器人数据；微调结合少量人类演示和多模态数据。

**📈 对比分析**

在四个真实操作任务（清桌、泡茶、折毛巾、折T恤）与两款商业臂和多种主流 VLA 模型比较，成功率平均 0.75，优于大多数公开基线。

**⚠️ 局限性**

局限在于对 OOD 物体的鲁棒性不足、跨躯体泛化受限，以及训练所需的硬件和数据集仍需进一步扩展。

---

## 398. When Model Merging Breaks Routing: Training-Free Calibration for MoE

**arXiv ID:** 2606.03391 | [PDF](https://arxiv.org/pdf/2606.03391v1)

**作者:** Canbin Huang `[一作]` (Sun Yat-sen University), Qifan Wang `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种训练无关的 MoE 模型合并后路由校准方法，用于修复合并导致的路由崩溃。

**💡 创新点**

创新点在于识别 MoE 合并中的路由崩溃问题，并通过二阶 Hessian 近似实现闭式路由校准，使用矩阵无关共轭梯度求解，从而无需进一步训练即可显著恢复专家选择逻辑。

**🔧 技术方法**

主要技术包括：二阶 Hessian 近似、KL 对齐目标、矩阵无关共轭梯度求解、Softmax + Top‑k 路由的解析校准。

**📊 数据集**

实验使用数学推理数据集 GSM8K、MATH500 以及代码生成数据集 HumanEval+、MBPP+；校准数据来自 OpenMathInstruct2 与 SelfOSSInstructSC2，亦测试一般预训练数据 C4。

**📈 对比分析**

与 Weight Averaging、TIES‑Merging、DARE、WUDI、Fisher Merging、RegMean 等基线对比，HARC 在多任务指标上提升约 0.5–1 分，尤其在代码任务上显著恢复甚至超越单体模型性能。

**⚠️ 局限性**

局限性：需要一定量的校准样本；对非常深层或极度稀疏的专家网络仍可能出现微调残留误差；对非 Softmax/Top‑k 路由结构的适用性尚未验证。

---

## 399. See, Infer, Intervene: Proactive World Modeling for Goal-Oriented Social Intelligence

**arXiv ID:** 2606.03371 | [PDF](https://arxiv.org/pdf/2606.03371v1)

**作者:** Honghui Zhang `[一作]` (Mita Technology), Tianyu Shi `[通讯]` (McGill University)

**通讯引用:** 1292 | [OpenAlex ID](https://openalex.org/A5101850471)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了智能零售设备在客户未提出明确请求前，如何通过观测、推断潜在意图并主动选择最合适的服务响应（如问候、挖掘、信息提供、推荐或保持不干预）来提升购物体验。

**💡 创新点**

提出了See–Infer–Intervene（SII）框架和Proactive Intent World Model（PIWM），通过AIDA–BDI状态模型和行动条件的未来状态预测，实现了在有限动作空间内的低干预主动决策。

**🔧 技术方法**

采用基于Qwen2.5-VL-7B的LoRA适配器，构建了客户状态估计网络、行动结果预测网络以及决策选择器，整合视觉、语言与行动信息进行端到端训练。

**📊 数据集**

使用自行构建的GuidanceSalesBench数据集，其中包含合成的客户状态清单、短视频观察、候选动作、模拟结果及最佳动作标签，并在目标机器视觉测试集和真实店铺试点视频上评估。

**📈 对比分析**

与随机基线、常数动作基线、零样本Qwen2.5-VL-7B、状态-结果模型以及闭源大型模型（Gemini、GPT-4o、Claude Sonnet）对比，PIWM在给定真值状态下在Target-Test上达到0.641宏F1，在视频端到端设置下为0.295，在真实店铺小样本上为0.579，显示出在行动监督和AIDA约束下显著优于大多数基线。

**⚠️ 局限性**

主要局限包括对真值状态的高度依赖导致视频到状态的映射仍为瓶颈；非干预（Hold）动作的识别仍弱；闭源大模型在某些设置下仍表现更好；真实店铺试点样本有限且缺乏非干预场景，限制了对低干预策略的充分评估。

---

## 400. Automating Information Extraction and Retrieval for Industrial Spare Parts Pooling

**arXiv ID:** 2606.03367 | [PDF](https://arxiv.org/pdf/2606.03367v1)

**作者:** Dyuman Bulloni `[一作]` (University of Applied Sciences and Arts of Southern Switzerland), Anna Valente `[通讯]` (University of Applied Sciences and Arts of Southern Switzerland)

**通讯引用:** 2286 | [OpenAlex ID](https://openalex.org/A5088210442)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 PhRAG 框架，结合检索增强生成（RAG）技术，将分散的零件信息通过命名实体识别（NER）构建成统一的虚拟库存池（VSPool），并基于该池实现自然语言查询的零件检索与排序。

**💡 创新点**

创新点在于：①将大型语言模型与语义+词典双检索融合，用逆序词频检索（BM25）与稠密向量检索结合 RRF 打分，提升检索相关性；②采用无监督/少样本提示的生成式 NER，显著提升制造业特殊实体的抽取精度；③在检索后通过零样本重排序模型提供可解释的检索结果，弥补传统 IR 的透明度不足。

**🔧 技术方法**

核心技术包括：大型语言模型（Llama 3.1 7B、Phi‑3.5‑mini、CodeLlama 7B、GPT‑5.2），RAG 架构，BM25 与 all‑MiniLM‑L6‑v2 的双检索，RRF 融合，ElasticSearch 索引，LoRA 微调，生成式重排序与解释生成。

**📊 数据集**

使用了制造业 NER 基准 FabNER‑simple（9.4k 训练，2.1k 测试）评估抽取任务；使用自建的 VSPool 数据集（4,020 组件，包含多品牌、价格、技术规格等字段）评估检索任务。

**📈 对比分析**

通过与基准 LLM（无提示）以及监督模型（SpaCy、BiLSTM‑CRF、GoLLIE、GPT‑5.2）对比。PhRAG 在 FabNER 上的 F1 分数提升高达 23%（从 0.59 提升至 0.88，接近监督基准）；在检索任务中，A@k 从单独词典检索的 0.167 提升至混合+重排序的 0.869，显示出明显的性能提升。

**⚠️ 局限性**

局限性包括：①仍落后于专门微调的监督模型，尤其在极端稀缺数据下；②依赖高质量的示例检索，若检索质量低会影响生成结果；③模型可能产生幻觉或不完整的解释；④嵌入维度和检索时间的权衡限制了实时部署；⑤评估主要集中在制造业，缺乏跨域验证。

---

## 401. Speech Emotion Recognition using Attention-based LSTM-Network with Residual Connection

**arXiv ID:** 2606.03359 | [PDF](https://arxiv.org/pdf/2606.03359v1)

**作者:** Daniil Krasnoproshin `[一作]` (Belarusian State University of Informatics and Radioelectronics), Maxim Vashkevich `[通讯]` (Belarusian State University of Informatics and Radioelectronics)

**通讯引用:** 213 | [OpenAlex ID](https://openalex.org/A5000197595)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了一种轻量级的语音情感识别模型ResLSTM-SA，结合残差连接和软注意力。

**💡 创新点**

创新点在于将一个与输入维度匹配的LSTM层作为残差块，提升长程依赖建模同时保持极低参数量。

**🔧 技术方法**

使用LSTM、软注意力机制、残差连接以及MFCC+chromagram特征提取。

**📊 数据集**

在RAVDESS情感语音数据集上进行评估。

**📈 对比分析**

通过5折speaker-independent交叉验证与Optuna调参，对比LSTM-SA以及多种CNN/Transformer模型，ResLSTM-SA-h64在UAR上达到0.6517，仅含46.8k参数，明显优于同类轻量模型且接近大型自监督模型。

**⚠️ 局限性**

局限性包括仅在单一数据集上验证、对高噪声/跨语料表现未评估，且情感类别如‘happy’仍易混淆。

---

## 402. APIC: Amortized Physics-Informed Calibration using Neural Processes

**arXiv ID:** 2606.03355 | [PDF](https://arxiv.org/pdf/2606.03355v1)

**作者:** Aishwarya Venkataramanan `[一作]` (Friedrich Schiller University Jena), Joachim Denzler `[通讯]` (Friedrich Schiller University Jena)

**通讯引用:** 12173 | [OpenAlex ID](https://openalex.org/A5024934744)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Neural Process的群体级物理模型校准框架APIC，能够在稀疏观测下快速推断实例特定的物理参数与结构性误差。

**💡 创新点**

创新点在于：① 将Kennedy–O’Hagan的系统误差建模与深度生成模型的摊销推理相结合；② 设计双分支潜变量结构，分别独立推断物理参数和状态相关的误差；③ 采用两阶段训练策略，先在无误差的仿真数据上预训练物理参数，再在真实数据上联合变分推理并正则化误差，实现参数与误差的可分辨性。

**🔧 技术方法**

技术手段包括：Neural Process（CNP、LNP、ANP）框架、变分推理与ELBO、双潜变量变分分布、正则化约束、可微物理求解器、两阶段（预训练+联合训练）策略。

**📊 数据集**

使用三类基准数据集：1）阻尼弹簧振子（欠缺阻尼项）；2）Lotka–Volterra 捕食-猎物 ODE（加入结构性驱动力）；3）一维输运扩散 PDE（缺少反应项）。

**📈 对比分析**

与经典KOH-GP、BCPI、Correction‑PINN等基准做对比；APIC在重建MAE、对数似然、ECE、参数恢复等指标均显著优于对手；同时摊销推理实现了毫秒级预测速度，远快于传统每实例优化的秒级/分钟级开销。

**⚠️ 局限性**

局限性：① 需要可微分的物理求解器；② 需要足够的仿真数据用于元学习；③ 误差建模需先验假设（如可分解为状态相关结构误差），若模型结构更为复杂或完全缺失，则效果有限；④ 仍面临参数与误差的潜在可识别性挑战。

---

## 403. Reflective Numeration Systems I: a Global Standpoint

**arXiv ID:** 2606.03351 | [PDF](https://arxiv.org/pdf/2606.03351v1)

**作者:** Benoît Rittaud `[一作]` (Université Sorbonne Paris Nord), Benoît Rittaud `[通讯]` (Université Sorbonne Paris Nord)

**通讯引用:** 89 | [OpenAlex ID](https://openalex.org/A5013942131)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a`

**🎯 论文内容**

本文提出了一套通用框架，用来扩展标准的 b‑ary Gray 码，生成 k‑bonacci Gray 码以及更多满足翻转位性质的 Gray 码序列；

**💡 创新点**

创新点在于引入 Gray 乘积与 -Gray 乘积的概念，建立了对 Gray 码的代数运算；通过这些运算证明了若干关于幂可结合性、Gray 乘积可结合性以及翻转位性质的通用定理；

**🔧 技术方法**

主要技术包括对列表的非交换积、Gray 乘积、-Gray 乘积的定义与性质证明，以及几何序列的构造与分析；

**📊 数据集**

本文为纯理论研究，未使用实验数据集；

**📈 对比分析**

通过理论推导和数学证明对所构造的 Gray 码进行比较，展示它们在满足翻转位性质、幂可结合性等方面的优越性；性能评价以理论性质为准；

**⚠️ 局限性**

局限性在于 -Gray 乘积一般不满足结合律，需要额外条件保证幂可结合性；此外，构造过程对特殊取值的依赖较强，未给出通用的构造算法。

---

## 404. Beyond Semantics: Modeling Factual and Affective Perceptual Experiences from Vision-Language Data

**arXiv ID:** 2606.03345 | [PDF](https://arxiv.org/pdf/2606.03345v1)

**作者:** Youssef Mohamed `[一作]` (KAUST), Mohamed Elhoseiny `[通讯]` (KAUST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 P-Topics 模型，先从图文对中无监督地发现感知主题（P-Topics），再学习注意力池化的映射函数将新图像归入这些主题。

**💡 创新点**

创新点在于首次将客观事实与主观情感拆分为感知体验的两个维度，提出两阶段 DEC‑基聚类与注意力映射的端到端框架，并动态筛选合适的主题数。

**🔧 技术方法**

技术手段包括：融合 CLIP 图像/文本嵌入与情感 BERT 嵌入；使用 denoising autoencoder 结合 DEC 聚类损失构建可聚类潜空间；通过注意力池化学习多标签映射；并实现动态聚类中心阈值过滤。

**📊 数据集**

实验数据集为 ArtELingo（约120万条艺术图文对，涵盖28种语言情感标签）与 Affection（约50万条现实图文对）。

**📈 对比分析**

与 BERTopic、CEMTM、VLM2VEC 及 K‑Means 等基线对比，P‑Topics 在 ArtELingo 上获得 Silhouette 0.97、AUC 0.94，并在人工评估中被偏好 58.4% vs 19.5%，显著优于基线。

**⚠️ 局限性**

局限性包括：未对图像相关性进行定量评估；多语言转译可能稀释文化细节；对高质量情感说明的依赖；以及仅在艺术与现实图像域验证，泛化至其它视觉领域仍待研究。

---

## 405. Non-Wellfounded and Cyclic Proofs for LTL: A Syntactic Correspondence with Linear Nested Sequents

**arXiv ID:** 2606.03413 | [PDF](https://arxiv.org/pdf/2606.03413v1)

**作者:** Tim S. Lyon `[一作]` (TU Dresden), Lukas Zenger `[通讯]` (Peking University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5092575126)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本文提出并研究了线性嵌套推理（Linear Nested Sequent，LNS）系统的非良构（non‑wellfounded）与循环（cyclic）证明，针对线性时序逻辑（LTL）构建了这两类证明系统，并给出了在LNS框架下进行循环识别（cycle recognition）与展开（unraveling）的完整方法。

**💡 创新点**

创新点主要有：
- 通过“饱和递归性（saturation recurrence）”证明非良构证明满足可循环化的正则性；
- 设计了“位移引理（shifting lemma）”实现从循环证明到非良构证明的逐步展开；
- 将这两种证明变换在LNS这一更强的多重序列结构中完成，填补了该领域先前仅在Gentzen序列或超序列（hypersequents）上研究的空白；
- 在证明系统中实现完全可分析、无结构规则、所有规则可逆，从而为后续的剪枝与截断模型提取提供理论基础。

**🔧 技术方法**

采用的技术包括：
- 线性嵌套序列（LNS）框架；
- 非良构证明与循环证明的定义与互相转换；
- 轨迹值（trace values）与进度度量（progress measure）用于证明安全性；
- 饱和递归性（SRP）与潜在轨迹集合的多重测度；
- Kőnig引理与递归证明搜索策略；
- 证明同构与弱同构概念用于描述展开过程的结构保持。

**📊 数据集**

本工作为理论性研究，无需数据集；所有验证与证明均在形式化逻辑与证明系统的框架内完成。

**📈 对比分析**

由于本研究聚焦于证明系统的语义与结构转换，并未涉及实验评估或性能对比；因此没有可量化的运行时或空间复杂度指标；相对先前仅在Gentzen序列上实现的循环证明方法，本工作在更丰富的LNS结构中实现了同等的可循环化与展开，理论上提升了表达能力。

**⚠️ 局限性**

局限性包括：
- 目前的证明转换仅适用于LNS框架，尚未推广到更通用的标签序列（labeled sequents）或其他多重序列形式；
- 剪枝（cut‑elimination）在LNS系统中的语法化证明仍是未解的开放问题；
- 对复杂性与可计算性（如证明长度、时间复杂度）的具体分析尚未给出；
- 研究仅针对LTL，尚未验证对更广泛时序或模态逻辑的适用性。

---

## 406. Link Prediction or Perdition: the Seeds of Instability in Knowledge Graph Embeddings

**arXiv ID:** 2606.03365 | [PDF](https://arxiv.org/pdf/2606.03365v1)

**作者:** Guillaume Méroué `[一作]` (Université Côte d’Azur), Pierre Monnin `[通讯]` (Université Côte d’Azur)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文系统评估了知识图谱嵌入模型（KGEM）在不同随机种子及硬件环境下的稳定性，提出了基于预测重叠（Pred‑Overlap）、嵌入空间重叠（Space‑Overlap）以及一致性（Consistency）和同质性（Homogeneity）等新的度量方法；

**💡 创新点**

创新点在于首次将随机性因素（初始化、负采样、三元组排序、Dropout以及硬件差异）拆分独立评估，发现各因素单独就能导致与模型性能相当的预测与嵌入空间的不稳定；并指出传统的 MRR/Hits@K 指标无法反映这些细粒度的波动；

**🔧 技术方法**

采用了多种典型 KGEM（TransE、RotatE、DistMult、ComplEx、ConvE、RGCN、Transformer），并在 PyTorch 生态下实现了对四个随机源的细粒度控制，使用了多次训练、对比组、统计分析等技术；

**📊 数据集**

实验使用了三大公开 KG 数据集 WN18RR、FB15k‑237 与 CoDEx‑S（此外在仓库中也提供 kinship 与 nations 的结果）；

**📈 对比分析**

通过对比同一配置下不同种子、不同随机源及不同 GPU 的性能，发现 MRR 变动小但预测重叠平均仅 0.4‑0.6，嵌入空间重叠也在 0.2‑0.5 之间；投票（Borda/Range）能略微提升稳定性但未完全消除；整体性能维持在各模型的 SOTA 近似区间；

**⚠️ 局限性**

局限性包括：只评估了有限的模型与超参空间；对投票的实验只针对预测，不覆盖嵌入空间；硬件差异只测试了几种 GPU；缺乏对不同超参（学习率、Dropout 等）对稳定性的系统性研究；并未提出新的训练正则化或损失来显式提升稳定性。

---

## 407. PrimeSVT: An Automated Memory-aware Pruning Framework with Prioritized Compression Policy for Spiking Vision Transformers

**arXiv ID:** 2606.03428 | [PDF](https://arxiv.org/pdf/2606.03428v1)

**作者:** Rachmad Vidya Wicaksana Putra `[一作]` (New York University Abu Dhabi), Muhammad Shafique `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 11438 | [OpenAlex ID](https://openalex.org/A5005190949)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个名为PrimeSVT的自动化内存感知结构化剪枝框架，旨在对预训练的脉冲视觉变换器（SViT）模型进行剪枝，以提高其在嵌入式实现中的效率。

**💡 创新点**

PrimeSVT的创新点在于其优先压缩策略，通过自动化的方式选择剪枝层，并在不同层之间应用非均匀的剪枝率，从而在满足准确性和内存约束的同时实现模型压缩。

**🔧 技术方法**

使用了基于L2范数的通道级滤波器剪枝技术，并结合了优先压缩策略和压缩模型选择机制。

**📊 数据集**

在ImageNet-1K数据集上进行了实验，使用了基于SDTv2的脉冲视觉变换器模型作为基线模型。

**📈 对比分析**

与现有的剪枝方法相比，PrimeSVT在单次剪枝中实现了26.68%的内存节省，同时保持了与未剪枝模型（73.3%）相差不超过3%的准确性（70.3%未微调，72.9%微调），显示出其在准确性和内存节省方面的有效性。

**⚠️ 局限性**

限制在于该方法可能无法在所有情况下同时满足准确性和内存约束，且在处理不同的剪枝率和层时可能需要较长的计算时间。

---

## 408. Enginuity: A Dataset and Benchmark for Vision-Language Understanding of Engineering Diagrams

**arXiv ID:** 2606.03410 | [PDF](https://arxiv.org/pdf/2606.03410v1)

**作者:** Abhishek Kumar `[一作]` (Predii), Tirthankar Ghosal `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 864 | [OpenAlex ID](https://openalex.org/A5081072666)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出Enginuity，首次公开了用于评估视觉-语言模型在工程图解（如军用维修手册）上的理解能力的基准数据集和任务；

**💡 创新点**

创新点在于：1）收集并标注了2000多张工程图与对应部件表的真实配对；2）设计了结构化部件提取与自由文本图解问答两大任务；3）发现基于token重叠的指标严重低估技术描述的准确度；

**🔧 技术方法**

采用了四款前沿视觉-语言模型（GPT‑5.2、Claude Opus 4.7、Gemma 4、Qwen3‑VL 32B）并通过零样本与链式思考提示进行评测；

**📊 数据集**

使用了从10本美国军事服务与维修手册中抽取的2,056幅图解及对应的3,011张部件表，Task‑1包含494张样本，Task‑2包含60道专家撰写的问答；

**📈 对比分析**

对比结果显示：召回率在0.61–0.87之间，描述精确度Token F1仅0.03–0.18，问答的LLM‑judge得分在2.6–3.2/5；链式思考在大多数模型上提升了约20–55%的指标；

**⚠️ 局限性**

局限性包括：数据来源仅限军用手册，难以直接推广到其他工程领域；模型版本随时更新导致结果可复现性受限；问答评估依赖LLM‑judge，可能引入主观偏差；

---

## 409. Flicker-DDPM: Accelerating Denoising Diffusion via 1/f Colored Noise Injection

**arXiv ID:** 2606.03393 | [PDF](https://arxiv.org/pdf/2606.03393v1)

**作者:** Kexiang Mao `[一作]` `[通讯]` (Wuhan University), Kexiang Mao (Wuhan University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在扩散生成模型中引入与自然图像功率谱匹配的1/f色噪声，构建 Flicker-DDPM。

**💡 创新点**

通过将噪声的空间相关性调节为 (d+1)^-η 并用 Matérn 协方差理论解析关联噪声谱指数与数据谱指数的关系，实现无架构改动、理论可解释的采样加速。

**🔧 技术方法**

基于 2D Hankel 变换的功率谱-相关性映射、Matérn 协方差理论、Cholesky/FFT 色噪声生成、频域线性理论及标准 DDPM 的 SDE/flow‑matching 训练框架。

**📊 数据集**

主要在 CIFAR‑10 图像数据集上进行实验，并对比了不同采样步数的 FID 性能。

**📈 对比分析**

与标准白噪声 DDPM 在相同采样步数下对比，Flicker-DDPM 在 150 步时获得 FID 12.24，优于 500 步白噪声 DDPM 的 13.02；总体可实现约 3.33 倍的采样加速并同时提升图像质量。

**⚠️ 局限性**

色噪声模型在高频细节上不足，且公式仅针对二维空间和 Matérn 协方差，若数据维度或相关性结构不同需重新推导；在极少数低频/高频极值场景下仍可能出现非线性耦合失配。

---

## 410. Mitigating False Credit Propagation: Probabilistic Graphical Reward Aggregation for Rubric-Based Reinforcement Learning

**arXiv ID:** 2606.03361 | [PDF](https://arxiv.org/pdf/2606.03361v1)

**作者:** Can Lv `[一作]` (Beihang University), Shiji Zhou `[通讯]` (Beihang University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种基于图形模型的评估聚合框架GEAR，旨在通过依赖结构化的rubric图来解决传统平面聚合导致的假信用传播问题。

**💡 创新点**

创新点在于将rubric中的前置与激活依赖视为有向无环图，并通过软保留因子对子指标的概率进行递归抑制，从而在保持低计算复杂度的前提下避免无效奖励的泄漏。

**🔧 技术方法**

技术包括：构建查询特定的typed依赖图、使用带有单调性约束的贝叶斯网络对指标事件进行条件推理、采用拓扑顺序的近似推理实现线性时间的边缘概率计算，以及将这些边缘概率用于归一化期望效用的奖励计算。

**📊 数据集**

实验使用了 HealthBench-500（医疗对话）、WritingBench（长文本写作）和 PLawBench（法律推理）三个开源评估基准，评估模型为 Qwen2.5-7B-Instruct 与 Llama-3.1-8B-Instruct 两种策略网络。

**📈 对比分析**

与传统的 Flat 聚合和硬门控方法相比，GEAR 在所有基准上均提升了约 6–7% 的平均分；在作为插件替代现有 Rubric‑RL 管线时，提升幅度可达 7–8%，并将 FCP 泄漏降低 96.5%，同时保持更多合法子指标的奖励。

**⚠️ 局限性**

局限性包括：仍依赖于局部评判的准确性和手工构建的依赖图；软抑制的保留因子采用固定设定，可能不适用于所有任务；近似推理在高度相关父子节点时为均值场近似；以及缺乏对实际用户偏好的直接验证。

---

## 411. Reliability-Guided Depth Fusion for Glare-Resilient Navigation Costmaps

**arXiv ID:** 2606.03421 | [PDF](https://arxiv.org/pdf/2606.03421v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 412. Lexicons and grammars for language processing: industrial or handcrafted products?

**arXiv ID:** 2606.03412 | [PDF](https://arxiv.org/pdf/2606.03412v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 413. The Unsampled Truth: Psychometrics in SLMs Measure Prompt Artifacts, Not Psychological Constructs

**arXiv ID:** 2606.03357 | [PDF](https://arxiv.org/pdf/2606.03357v1)

**作者:** Nils Schwager `[一作]` (Trier University), Achim Rettinger `[通讯]` (Trier University)

**通讯引用:** 1817 | [OpenAlex ID](https://openalex.org/A5000758128)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估13款开放权重语言模型在心理测评任务中的表现，构建“Prompt Variant Matrix”通过系统改变提示的身份、指令、题目与选项符号，量化并分离提示引入的伪变异与真实语义推理；

**💡 创新点**

提出了基于Wasserstein距离和PERMANOVA的误差分解框架，能够在保持语义不变的前提下，量化提示工件对模型输出的影响，从而揭示大模型在心理测评中仍受提示依赖；

**🔧 技术方法**

使用logit分布提取、1-Wasserstein距离计算平均成对距离(APWD)、PERMANOVA统计分解、以及对Prompt Variant Matrix的系统遍历与指标汇总；

**📊 数据集**

采用Big Five Inventory (BFI)、Short Dark Triad (SD3) 问卷、Nemotron 数据集中的5种基准人格，以及对应的题目与选项符号多种变体；

**📈 对比分析**

将不同参数规模（0.6B–14B）的模型在所有提示变体组合下进行实验，计算APWD与ASV；结果显示，即使在14B模型中，>50%的解释方差仍由提示伪变异主导，说明现有提示方法对模型心理推断的可靠性极低；

**⚠️ 局限性**

研究仅覆盖有限的提示变体维度、未考察结构位置与顺序影响、对变体选择敏感、缺乏人类基准、未深入交互效应，且只限于开放权重模型，可能无法代表闭源大模型的行为。

---

## 414. AlgoTouch: An Execution-Centered Approach to Incremental Construction of Imperative Programs

**arXiv ID:** 2606.03349 | [PDF](https://arxiv.org/pdf/2606.03349v1)

**作者:** Michel Adam `[一作]` (Doctor of Computer Science), Moncef Daoud `[通讯]` (University of South Brittany)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 AlgoTouch，一个基于执行的增量式程序构造系统，能够通过直接操作程序数据并记录执行行为，逐步生成完整的命令式程序。

**💡 创新点**

创新点在于：①使用执行轨迹对控制结构（条件、循环）进行确定性合成；②将执行、构造与代码生成统一到同一架构；③支持不完整程序的持续执行与完善；④能够一次性生成多种主流语言（Python、C、C++、Java）的代码。

**🔧 技术方法**

核心技术包括：显式的概念机（notional machine）暴露存储、计算与控制流；中间表示（IR）捕获数据变换；基于执行行为的控制结构合成算法；循环宏支持非线性增量构造；跨语言代码生成器。

**📊 数据集**

使用了一组代表性的算法基准（如经典排序、查找、数值计算等）进行评估，但未给出具体数据集名称。

**📈 对比分析**

通过工程级验证，对比了多语言生成的正确性、表达力、鲁棒性和语言独立性。实验结果显示 AlgoTouch 在所选基准上能够完整、正确地生成代码，且在不同语言间保持一致性，性能满足中等规模算法开发需求。

**⚠️ 局限性**

局限性包括：①需要显式概念机，可能不适用于所有语言特性；②对复杂控制流或动态行为的合成仍存在挑战；③评估范围仅限算法基准，缺乏大型系统或实际工业项目的验证；④依赖执行轨迹，若输入数据有限或不完整，可能导致合成不完整或错误。

---

## 415. Multi-Modal Assessment of Road Roughness Using Smartphone Applications, Acceleration, and Passenger Ratings

**arXiv ID:** 2606.03427 | [PDF](https://arxiv.org/pdf/2606.03427v1)

**作者:** Novel Certada `[一作]`, Cristina Olaverri-Monreal `[通讯]` (Johannes Kepler University)

**通讯引用:** 2207 | [OpenAlex ID](https://openalex.org/A5009941460)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并验证了一种多模态、以人为中心的低成本道路粗糙度评估框架，融合两款智能手机IRI估算、车辆IMU测量和乘客PSR主观评价；

**💡 创新点**

通过对比无基准条件下的三种信息源，首次从一致性、物理关联和感知相关性三角度评估低成本粗糙度测量的可行性；

**🔧 技术方法**

使用Pearson相关系数、ICC和Bland–Altman分析对两款IRI应用间的线性相关性、互换性和比例偏差进行统计评估；

**📊 数据集**

数据来自1700公里实路测量，涵盖奥地利、匈牙利、罗马尼亚三国高速与主干道，记录了两款手机的IRI估值、车载IMU竖向加速度以及乘客PSR评分；

**📈 对比分析**

结果显示两款IRI应用具有较高相关系数（r≈0.7），但ICC低（0.15–0.41）且存在显著比例偏差，无法互换；而IMU加速度与PSR呈显著负相关（r≈-0.4~-0.5），表明物理粗糙度与乘客感知高度一致；

**⚠️ 局限性**

主要限制包括缺乏真实IRI基准、两款应用算法不透明、乘客评分受人数和时间窗口限制、以及对速度和道路类型的细粒度分析不足。

---

## 416. SAMatcher: Co-Visibility Modeling with Segment Anything for Robust Feature Matching

**arXiv ID:** 2606.03406 | [PDF](https://arxiv.org/pdf/2606.03406v1)

**作者:** Xu Pan `[一作]` (Wuhan University), Xianwei Zheng `[通讯]` (Wuhan University)

**通讯引用:** 1312 | [OpenAlex ID](https://openalex.org/A5063225294)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种通过显式建模视图间可见区域（co‑visibility）来增强特征匹配的框架 SAMatcher。

**💡 创新点**

创新点在于：①使用 Segment Anything Model 进行跨视图区域预测；②设计对称的交叉视图特征交互模块实现双向语义对齐；③联合掩码与框预测并引入几何一致性监督，提升可见区域的精度；④将可见区域作为结构化先验显著抑制无效匹配。

**🔧 技术方法**

采用 SAM-HQ 编码器、对称交叉视图 Transformer（含视图感知 RoPE）、Prompt 驱动的掩码解码器、盒子解码器以及点采样掩码损失、IoU、中心尺寸、mask–box 一致性等多种监督技术。

**📊 数据集**

主要使用 MegaDepth 数据集进行训练与量化评估，并在 ScanNet（室内）与 GL3D（航空）上进行零样本泛化验证。

**📈 对比分析**

与传统局部特征+匹配器、OETR 区域先验以及 LoFTR 等基线相比，SAMatcher 在 MegaDepth 上大幅提升 AUC、Acc、mAA 等指标，尤其在尺度差异大、视角宽广的情况下表现突出。

**⚠️ 局限性**

局限性：模型对极端遮挡或完全无重叠场景仍可能预测误差；依赖 SAM-HQ 编码器导致推理开销较大；缺乏针对动态场景或多帧连续匹配的评估。

---

## 417. Bastet: A Fine-Grained Expert-Labeled Dataset for DeFi Smart Contract Vulnerability Detection

**arXiv ID:** 2606.03387 | [PDF](https://arxiv.org/pdf/2606.03387v1)

**作者:** Wan-Hsuan Hsu `[一作]` (AIFT), Kentaroh Toyoda `[通讯]` (AIFT)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了基于真实 Code4rena 审计报告的 DeFi 智能合约漏洞数据集 Bastet，包含 4,402 条发现，其中 849 条已完成专家双人共识标注。

**💡 创新点**

创新点在于三方面：使用现代 Solidity 版本；采用专家双人共识标注工作流；并提出 46 个标签与 77 个子标签的两层细粒度分类体系，兼顾根因机制与实现缺陷。

**🔧 技术方法**

使用了人工专家标注、双人一致同意流程以及基于 Code4rena 审计报告的公开数据，配合两层标签体系对漏洞进行精细划分。

**📊 数据集**

主要使用的原始数据集为 Code4rena 公开审计报告（2021–2024），并由 DeFiHackLabs 社区成员进行标注与根因摘要。

**📈 对比分析**

与现有数据集（如 SmartBugs、SolidiFI、ReentrancyStudy 等）比较，Bastet 在 Solidity 版本、标签细粒度、专家标注、根因摘要等维度上均优越，为 LLM 评估提供更可靠、真实的基准，表现出更高的检测覆盖率和标签准确性。

**⚠️ 局限性**

限制在于目前仅 849 条发现完成完整标注，数据覆盖率有限；且标注仍来自单一社区，可能存在标注偏差；未来需扩大标注范围并持续更新分类体系。

---

## 418. Local Guidance, Global Impact: Gaussian-Reshaped Trust Region Unlocks Behavior Transitions

**arXiv ID:** 2606.03382 | [PDF](https://arxiv.org/pdf/2606.03382v1)

**作者:** Bingxu Liu `[一作]`, Ling Pan `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

针对PPO在持续非平稳环境下失效，提出高斯信任区域策略优化（GTR），通过高斯核改造信任区间并引入混合高斯锚点实现几何感知的动态更新。

**💡 创新点**

创新点在于设计了非单调、有限的高斯信任约束，既在靠近旧策略时提供强制稳定，又允许在优势高的方向自由跳跃；以及动态混合锚点缓解数据老化导致的偏差。

**🔧 技术方法**

使用PPO框架、KL/Fisher信息矩阵理论、Gaussian kernel约束、Mixture Gaussian Anchor、以及多任务和多模态模型（MLP、RNN、Transformer、SimBa）进行训练。

**📊 数据集**

实验数据集包括：控制任务（Walker、Dog、HalfCheetah、Walker、Ant、Humanoid、Procgen Starpilot）、开阔世界游戏（Craftax）、大语言模型微调（MATH‑500、OlympiadBench、MinervaMath、AMC 2023）等。

**📈 对比分析**

与标准PPO、PPO‑KL、SPO等对比，GTR在机器人控制中提升最高25%，在Craftax中实现关键技能切换并加速收敛，在LLM后训练中取得所有基准的最高pass@4。对剪切范围宽松或KL系数调整时，GTR仍保持稳定，标准PPO则崩溃。

**⚠️ 局限性**

缺乏严格理论证明、对极大规模LLM的适用性尚未验证、仍需超参数调优且对某些极端非平稳任务的泛化能力未完全评估。

---

## 419. Neural Change Prediction: Relating Software Changes to Their Effects and Vice Versa

**arXiv ID:** 2606.03378 | [PDF](https://arxiv.org/pdf/2606.03378v1)

**作者:** Laura Plein `[一作]` (CISPA Helmholtz Center for Information Security), Andreas Zeller `[通讯]` (CISPA Helmholtz Center for Information Security)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 Neural Change Prediction，一种通过对程序进行大量合成变异并观察其运行输出，自动学习代码变动与程序行为之间关联的方法，并利用该学习得到的模型实现：给定期望的行为变更预测相应的代码修改位置与内容；给定代码修改预测其对程序输出的影响。

**💡 创新点**

创新点在于：① 采用动态执行与变异数据构建大量 (代码变动, 运行结果) 样本，摆脱传统静态分析的限制；② 通过对 LLM 进行微调，让模型能在无先验项目知识的情况下学习到特定项目的语义关联；③ 方法语言无关，既可用于 Python 程序，也可应用于 CSS 配置文件，证明了跨语言、跨 artifact 的通用性。

**🔧 技术方法**

技术手段主要包括：① 变异工具（PyMut4SE、CSS 变异器）对源代码/配置文件进行自动化单/高阶变异；② 在受控执行环境下收集输入、原始输出、变异后输出及变异位置；③ 将这些 (代码变动, 行为变动) 对输入 LLM（GPT‑4.1、GPT‑oss‑20B、CodeLLama‑7B‑Instruct、Qwen‑3‑4B‑Instruct 等）进行微调；④ 评估时采用多项指标（定位准确率、语义克隆率、严格/宽松匹配等）。

**📊 数据集**

数据集：① Python 侧使用 QuixBugs benchmark 的 44 个函数（简单/复杂输入），共生成约 3,004,715 条变异执行记录；② CSS 侧使用 Templated 网站模板 440 个，生成 390,612 条 (变动, 预期/实际截图) 训练样本；③ 通过变异分为单阶 (SOM) 与高阶 (HOM) 两类，进一步区分成功/失败执行，形成多种实验用数据子集。

**📈 对比分析**

与基线（未微调的 GPT‑4.1）对比，微调后效果显著提升：① 预测行为改变的准确率从 33% 提升至 87%（SOM）/99%（HOM）；② 代码定位准确率从 9.6% 提升至 82.6%（SOM）/100%（单一模板）；③ 代码生成的语义克隆率从 12.8% 提升至 68.5%（SOM）/71.6%（HOM），在 CSS 上严格匹配率可达 95%（通用模型）/100%（项目特定）。实验还显示：单次生成平均只需 1–2 次，最多 5 次即可满足 80%+ 的成功率；错误类型预测准确率高达 95–99%。

**⚠️ 局限性**

局限性包括：① 需要大量的变异与执行，计算成本高；② 对高阶变异的样本采样有限，导致训练不够均衡；③ 对复杂输入的支持仍受限（输入文本化表示需人工手工）；④ 仍难以实现完全的语法精确匹配，宽松匹配虽可接受但不满足所有正式需求；⑤ 结果受所选 LLM 版本与微调策略影响，未来可进一步探索更高效的模型或自监督学习方法。

---

## 420. Intellectual Humility as a Cognitive Filter for AI-Generated Health Misinformation. An Evolutionary Perspective on Epistemic Vigilance

**arXiv ID:** 2606.03377 | [PDF](https://arxiv.org/pdf/2606.03377v1)

**作者:** Marcin Rządeczka `[一作]` (Maria Curie-Skłodowska University), Marcin Moskalewicz `[通讯]` (Maria Curie-Skłodowska University)

**通讯引用:** 541 | [OpenAlex ID](https://openalex.org/A5023194473)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了智力谦逊（IH）如何影响参与者对不同科学严谨度的AI生成健康对话的可信度评估。

**💡 创新点**

发现IH是一个选择性过滤器，对伪科学内容降低可信度评分，却不影响对科学准确内容的评估，且与来源识别无关。

**🔧 技术方法**

使用实验设计、心理测量（IH量表）、非参数统计分析以及GPT‑4生成文本的技术。

**📊 数据集**

样本为99名波兰大学生；对话文本由GPT‑4生成，分别为科学准确、中等伪科学和强伪科学三种类型。

**📈 对比分析**

通过Kruskal‑Wallis、Mann‑Whitney U检验比较三种对话的可信度、同意度等指标，发现准确对话可信度显著高于伪科学；IH与伪科学可信度的相关系数在‑0.39至‑0.46之间，显示显著效果。

**⚠️ 局限性**

局限性包括样本单一（大学生）、健康信息领域局限、谦逊自评可能存在上限效应、未检验不同文化或年龄组的可推广性。

---

## 421. P\textsuperscript{2}-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization

**arXiv ID:** 2606.03376 | [PDF](https://arxiv.org/pdf/2606.03376v1)

**作者:** Ruipeng Zhang `[一作]` (South China University of Technology), Tong Zhang `[通讯]` (South China University of Technology)

**通讯引用:** 42640 | [OpenAlex ID](https://openalex.org/A5100378815)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a4b10f5d-130b-4e77-9367-6469ec621899` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为P2-DPO的自我纠错框架，利用模型自身生成的on‑policy、视觉感知对比对（Focus‑and‑Enhance 与 Visual Robustness），并结合校准损失和动态缺陷权重进行训练，直接针对视觉处理瓶颈与鲁棒性缺陷。

**💡 创新点**

核心创新包括：①利用模型自身生成on‑policy、视觉感知对比对，避免vision‑agnostic、off‑policy问题；②设计 Focus‑and‑Enhance 与 Visual Robustness 两类对比对，分别解决感知瓶颈与图像降质问题；③引入 Calibration Loss 明确优化视觉信息依赖；④通过 Dynamic Deficit‑Weighting 根据 CLIP 得分动态调节两类对比对的权重；⑤无需人工标签，显著提升数据效率。

**🔧 技术方法**

采用 Direct Preference Optimization (DPO) 作为训练目标；通过注意力图裁剪、遮挡、噪声等视觉扰动生成对比对；使用 Contrastive Amplification 进行解码；引入 Calibration Loss 与 Dynamic Deficit‑Weighting；利用 Implicit Preference Strength (IPS) 分析数据质量；评估时使用 Attention Focus Ratio (AFR)、Processing Accuracy 等指标。

**📊 数据集**

使用公开的 RLHF‑V 图像‑问题数据集进行对比对生成；TextVQA 作为感知瓶颈验证（提供真实边界框）；在 POPE、HallusionBench、MMHal‑Bench、AMBER 等标准幻觉与可信度基准上评测；实验还在 Qwen2.5‑VL‑3B/7B 上验证泛化。

**📈 对比分析**

与基线 LLaVA‑1.5‑7B、VCD、HA‑DPO、V‑DPO_RLHF‑V、V‑DPO_SAD、DPO_RLHF‑V 等方法对比，P2‑DPO 在 POPE、HallusionBench、MMHal‑Bench、AMBER 等指标均实现与或优于人类反馈方法的性能，尤其在轻度噪声场景下的鲁棒性显著提升；实验显示其数据效率高、训练成本低于传统 off‑policy 方案。

**⚠️ 局限性**

局限性：①仍无法彻底消除所有幻觉，仅针对感知瓶颈与鲁棒性；②对极端视觉干扰的鲁棒性尚待进一步提升；③算法对超参数（如 λ_calib、λ_ca、DDW）较为敏感；④在更大规模或不同架构（如更大 LVLM）上的泛化需要进一步验证；⑤训练仍需要一定计算资源，尤其在生成对比对与动态加权过程中。

---

## 422. AugMask: Training Diffusion Models on Incomplete Tabular Data via Stochastic Augmentation and Masking

**arXiv ID:** 2606.03347 | [PDF](https://arxiv.org/pdf/2606.03347v1)

**作者:** Jungkyu Kim `[一作]` (Yonsei University), Kibok Lee `[通讯]` (Yonsei University)

**通讯引用:** 1966 | [OpenAlex ID](https://openalex.org/A5103150653)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种名为AugMask的训练框架，利用随机条件填充与观察点损失掩蔽，使得传统的无缺失感知扩散模型能够在缺失表格数据上直接训练与生成

**💡 创新点**

核心创新在于将缺失值的补齐从监督目标中分离出来，采用条件随机填充产生不确定的上下文，并在损失中仅对原始观测值进行监督，从而实现对缺失不确定性的自适应正则化

**🔧 技术方法**

基于Score‑based扩散模型、条件随机填充（使用LightGBM学习每个特征的条件均值与方差或类别概率）以及观察掩蔽的损失函数

**📊 数据集**

在多种公开表格数据集（如Adult、其他6个数据集）上进行实验，并在MCAR/MAR不同缺失比例（0.1–0.9）下进行评估

**📈 对比分析**

与缺失感知方法（MIWAE、MissDiff、DiffPuter、ForestDiff）和无缺失感知模型（TVAE、CTGAN、TabDDPM、TabDiff、CDTD）比较，AugMask在生成质量（α‑Precision、β‑Recall、Trend、Shape）和下游任务（TSTR AUROC/RMSE）上均获得最高或接近最高排名，尤其在高缺失率时优于所有缺失感知基线

**⚠️ 局限性**

仅适用于直接在特征空间训练的Score‑based扩散模型；对Latent扩散、复杂缺失模式或高维度数据的推广仍需研究；依赖填充器的质量，且不提供正式的隐私保护

---

## 423. HonestAffinity: Leak-Aware Evaluation of Protein and Pocket Priors for Binding Affinity Prediction

**arXiv ID:** 2606.03422 | [PDF](https://arxiv.org/pdf/2606.03422v1)

**作者:** Junhao Wei `[一作]` (Macao Polytechnic University), Xu Yang `[通讯]` (Macao Polytechnic University)

**通讯引用:** 3100 | [OpenAlex ID](https://openalex.org/A5100462079)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种使用冻结的ESM-2蛋白序列嵌入和可选口袋位置标记的1D输入亲和力预测模型，并在漏检无泄漏的LP-PDBBind和CASF-2016数据集上进行评估。

**💡 创新点**

核心创新是揭示在不同评估分割下，同一模型组件（ESM-2和口袋标记）表现相反的“拆分条件反转”，并提出针对不同部署场景的三种模型变体。

**🔧 技术方法**

采用多尺度1D卷积、残差块、单层Transformer、矩阵乘积评分和MLP回归；使用冻结的ESM-2（650M）蛋白嵌入及可学习的二进制口袋标记。

**📊 数据集**

使用PDBbind v2020R1（19,032复合物）重构的LP-PDBBind 3级无泄漏划分、CASF-2016及其非训练子集。

**📈 对比分析**

与五个重现基线（DeepDTA、Pafnucy、DeepDTAF、DEAttentionDTA、Cross-modal MS）在六个评估拆分上进行比较，结果显示在标准验证和CASF分割上-Pocket表现最好，而在严格LP无泄漏层级上-Pocket-NoESM优于其它模型。

**⚠️ 局限性**

局限性包括需预先获取口袋残基列表（或仅使用NoPocket），ESM-2参数被冻结，未探索参数高效微调；模型在3D结构基方法的高性能水平上仍有差距。

---

## 424. PHAF-Personalized Hand Avatars in a Flash

**arXiv ID:** 2606.03420 | [PDF](https://arxiv.org/pdf/2606.03420v1)

**作者:** Meghana Shankar `[一作]` (Samsung R&D Institue), Pawan Prasad BH `[通讯]` (Samsung R&D Institue)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了 PHAF，利用仅两张手部图像（背面和掌面）即可快速生成高质量个性化手部头像的优化无依赖流水线。

**💡 创新点**

创新点在于：① 语义引导的几何对齐模块，使用关键点和轮廓控制点实现精准对齐；② 细分网格密化并直接提取高频纹理；③ 视角引导的 UNet 补全网络完成侧面与间隙纹理；④ 通过 Transformer 评价指标验证细节保持，整体实现约 30× 的生成速度提升。

**🔧 技术方法**

核心技术包括 MANO 参数化手部模型、TPS (Thin Plate Spline) 对齐、网格细分与稠密纹理采样、残差注意力 U‑Net 纹理补全、以及 ViT/DINOv2 等 Transformer 评估。

**📊 数据集**

使用了 Hands11k、HARP、以及自采集的 15 人 50K 图像数据集进行训练与评估。

**📈 对比分析**

在 ViTScore/DINOv2 评价上，PHAF 与基于视频的 UHM、HARP 在视觉相似度上相近或更优；单图 OHTA 逊色；PHAF 仅需约 2 分钟完成纹理生成，较视频方法约 1 小时提升 30×，同时保持或超过同速率下其他方法的质量。

**⚠️ 局限性**

局限性：对侧面/间隙纹理仍需补全，TPS 对控制点数量高度敏感；细分网格虽提高质量但增加采样与内存成本；对极端姿态、极端光照或低光环境的鲁棒性尚待进一步验证。

---

## 425. AI Model Extraction Attacks: Bypassing Single-Client Assumptions in Defenses

**arXiv ID:** 2606.03381 | [PDF](https://arxiv.org/pdf/2606.03381v1)

**作者:** Maxime Schwarzer `[一作]` (Thales Deutschland), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5182 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文系统验证了单客户端假设（SCA）在模型提取攻击防御中的失效，并提出开源框架CerberusAI用于评估分布式攻击场景。

**💡 创新点**

创新点在于：①揭示SCA在协调APT攻击下不可行；②设计可模拟分布式、混合流量的攻击管理器；③通过实验证明传统基于统计的防御（如PRADA）在分布式和噪声混合场景下失效。

**🔧 技术方法**

技术包括：分布式攻击调度（轮询与流量混合）、基于黑盒的MI‑FGSM提取攻击、PRADA防御的统计检测、Python/YAML模块化配置以及实验结果可视化。

**📊 数据集**

使用MNIST数据集训练的CNN（98.4%准确率）作为目标模型，攻击预算2500次查询。

**📈 对比分析**

对比方法：单客户端基线、分布式（400客户端轮询）、全局聚合防御、适应性混合流量。实验显示：单客户端F1≈63.2%，分布式F1为0%，全局防御F1≈90.8%，混合流量下F1仅≈2.0%，精确率<1%。

**⚠️ 局限性**

局限性包括：仅在MNIST上验证，未覆盖更复杂模型或多模态数据；防御实验集中于PRADA，缺少其他方法的比较；分布式攻击模型假设攻击者能完全控制客户端分布，实际网络中可能受限。

---

## 426. The Impact of Temporal Granularity on Socio-Demographic Inference from Household Load Profiles

**arXiv ID:** 2606.03358 | [PDF](https://arxiv.org/pdf/2606.03358v1)

**作者:** Dejan Radovanovic `[一作]` (Salzburg University of Applied Sciences), Günther Eibl `[通讯]` (Salzburg University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

研究了不同时间粒度（15分钟到7天）下，使用智能电表负荷曲线对住宅社会人口属性进行推断的可行性与隐私风险；

**💡 创新点**

首次提出以随机抽取的一周片段作为测试样本的评估框架，探讨时间粒度对推断准确率的影响，并揭示两阶段准确率平台（1小时与1-7天）与特征重要性随粒度变化的规律；

**🔧 技术方法**

利用手工特征、tsfresh自动特征、CNN自编码器嵌入等特征提取方法，并使用SVM、kNN、AdaBoost、MLP、LDA、XGBoost等分类器；采用SHAP进行特征重要性解释；

**📊 数据集**

基于奥地利萨尔茨堡地区1589户住宅的1年15分钟级别负荷数据（经筛选后1208户，62816周样本）以及对应的社会人口属性；

**📈 对比分析**

通过MCC、精确率-召回率、F1等指标对不同粒度、特征、分类器进行系统比较；结果显示粒度细化至1小时后性能基本稳定，XGBoost在所有属性与粒度上表现最佳；

**⚠️ 局限性**

局限包括：社会人口属性为年级常量，无法反映周内变化；手工特征设计仅针对日内粒度，无法充分捕捉7天级别信息；标签噪声与不完整性可能影响模型训练与评估；

---

## 427. ImageAuditor: Membership Inference Attack against Image-based Retrieval-Augmented Generation

**arXiv ID:** 2606.03354 | [PDF](https://arxiv.org/pdf/2606.03354v1)

**作者:** Jinghuai Zhang `[一作]` (Ucla), Yuan Tian `[通讯]` (Ucla)

**通讯引用:** 38701 | [OpenAlex ID](https://openalex.org/A5100361841)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种针对图像检索增强生成（IRAG）系统的成员推断攻击（MIA），可在仅使用四次查询的前提下判断图像是否存在于外部数据库中。

**💡 创新点**

创新点包括：①将攻击查询拆分为检索段与提取段，分别针对跨模态检索和信号提取进行专门优化；②设计Reward‑Guided Policy Optimization (RGPO) 的无梯度策略搜索来解决跨模态检索的局部最优问题；③为文本到图像（T2I）与问答（Q&A）任务构造了任务专属的提取提示与评分规则，并通过K‑means聚类实现多查询聚合。

**🔧 技术方法**

主要技术包括：跨模态对比奖励函数、RGPO随机策略优化、提取段提示设计（使用高质量caption或结构化问答提示）、相似度评分（I2I或T2T）、多查询聚合与K‑means。

**📊 数据集**

使用了多源图像数据库（MSCOCO、ImageNet‑100、WikiArt、Stanford Cars、Stanford Dogs、CelebA‑HQ）和问答数据库（MMQA、MSCOCO）以及多种生成模型（SDXL+IP‑Adapter、SD1.5+IP‑Adapter、Kandinsky v2.2、SDXL+Conceptrol、Qwen2.5‑VL、LLaVA‑1.6）。

**📈 对比分析**

与基线（Naïve、PoisonedRAG‑B、PoisonedRAG、BadRAG）相比，ImageAuditor 在T2I和Q&A任务上均实现了80%以上AUROC、约50% TPR@5%FPR，并且对不同生成器、数据库规模、检索预算及检索长度等设置具有鲁棒性。

**⚠️ 局限性**

局限性包括：仅评估了基于文本/图像注意力比的检测方法，未涉及更高级的防御（如差分隐私、输出扰动）；只考虑了文本输入的检索；对模型内部参数的推断能力尚未研究。

---

## 428. Rain: RDMA-assisted In-Network Scheduling for Microsecond-scale Workloads

**arXiv ID:** 2606.03352 | [PDF](https://arxiv.org/pdf/2606.03352v1)

**作者:** Zhihuang Ma `[一作]` (University of Science and Technology of China), Zuqing Zhu `[通讯]` (University of Science and Technology of China)

**通讯引用:** 9563 | [OpenAlex ID](https://openalex.org/A5008574028)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种基于可编程交换机的 RDMA 辅助 in‑network 调度系统，实现微秒级工作负载的尾延迟控制。

**💡 创新点**

创新点包括：① 双向队列在交换机内部匹配任务与令牌；② RDMA 多播预写大任务，仅在交换机存储元数据；③ 切片感知调度降低任务分散；④ 自适应调度动态调整工作队列深度与切片分配，突破传统 push/pull 设计限制。

**🔧 技术方法**

技术手段：Intel Tofino 可编程交换机、RoCEv2 RDMA 多播、任务切片划分、分布式调度代理、INT 监控、DPDK 端口、CPU 线程模型。

**📊 数据集**

数据集与实验负载：合成服务时间分布（常数、指数、双峰）以及 RocksDB 实际 KV 存储负载。

**📈 对比分析**

评估方法：与 R2P2、RackSched、Draconis、Pallas 等 SOTA 进行尾延迟、吞吐率对比；在 RocksDB 负载下实现 1.75× 更高吞吐率、99% 延迟维持在 SLO 以内。

**⚠️ 局限性**

局限性：仅支持单机架部署，RDMA 预写受 MTU 限制，交换机资源占用高，需额外全局协调，无法处理极大队列和跨机架负载。

---

## 429. SynCred-Bench: Benchmarking Synthetic Credibility in AI-Generated Visual Misinformation

**arXiv ID:** 2606.03348 | [PDF](https://arxiv.org/pdf/2606.03348v1)

**作者:** Junxiao Yang `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 16146 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了SynCred-Bench基准，用于检测AI生成的具有可信形式与传播痕迹的误导性图像。

**💡 创新点**

首次将可信形式与可信传播维度统一定义为“synthetic credibility”并提供系统化评测框架。

**🔧 技术方法**

采用文本到图像生成（GPT Image 2）创建样本，评估多模态大语言模型与AIGC检测器，结合人工验证与解释分析。

**📊 数据集**

使用600张AI生成的误导图像（六类可信形式×七类传播风格）及FP450真实负样本。

**📈 对比分析**

对15款MLLM和多款专用检测器在5% FPR约束下进行比较，结果显示MLLM TPR仅10.5%，专用检测器平均TPR约30%，人类评估TPR约63%，证明任务极具挑战。

**⚠️ 局限性**

局限包括样本仅来自单一生成器、规模有限、未覆盖更多语言与文化场景、仅评测生成检测未包含真伪判定，以及缺乏独立用户可信度研究。

---

## 430. Mamba-Enhanced Implicit Motion Learning for Audio-Driven Portrait Animation

**arXiv ID:** 2606.03402 | [PDF](https://arxiv.org/pdf/2606.03402v1)

**作者:** Xuan Wei `[一作]` (Xiamen University), Qingqi Hong `[通讯]` (Xiamen University)

**通讯引用:** 1461 | [OpenAlex ID](https://openalex.org/A5102714385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

生成音频驱动的单幅人像视频，采用隐式运动学习框架两阶段生成高质量、时序连贯的人类动作视频。

**💡 创新点**

引入隐式运动表示与层次化深度感知注意机制，并在第二阶段使用Mamba增强扩散模型实现音频驱动的运动特征预测，成功将运动预测与渲染解耦。

**🔧 技术方法**

使用变分自动编码器、Deviation Image Transformer、Latent Motion Deviation Decoder、Mamba全局/局部特征提取、扩散模型（Mamba增强）、感知损失与对抗损失等技术。

**📊 数据集**

使用自制的380小时高质量多样化人像语音视频数据集（DiverseHeads）以及CREMA-D、RAVDESS、HDTF、MEAD、CMLR、PATS等公开数据集。

**📈 对比分析**

与AniPortrait、DaGAN、FOMM、LivePortrait、MCNet、TPSMM、S2G-MDDiffusion、TANGO等SOTA方法在PSNR、SSIM、FID、FVD、Headpose、DIV、LSE-D/C等指标上对比，本文在多数指标上取得最高或接近最高成绩，并且推理速度更快。

**⚠️ 局限性**

在极端姿态变化或复杂背景下细节失真、缺乏完整的物理约束导致运动跳跃，以及对多模态情绪表达的处理仍有限。

---

## 431. Selective Token-Level Cryptographic Redaction for Privacy-Preserving Clinical Deployment of Large Language Models

**arXiv ID:** 2606.03399 | [PDF](https://arxiv.org/pdf/2606.03399v1)

**作者:** Farhan Sheth `[一作]` (MedVisAI Lab), Si Yong Yeo `[通讯]` (MedVisAI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种名为HERALD的客户端端到端加密与重排框架，在临床文本中仅加密敏感token，保持非敏感上下文，以支持大语言模型的隐私安全推理和训练。

**💡 创新点**

创新点在于将实体识别+POS规则与词干化归一化相结合，只对高风险token做可学习的确定性加密，并用显式定界符标记，兼容任何Transformer模型且不需改造模型结构。

**🔧 技术方法**

技术包括医学NER + POS基准敏感标记、词干化归一化、对敏感token应用对称加密（AES, FPE）或相似哈希（Fuzzy Hash），以及在预处理阶段插入明确定界符。

**📊 数据集**

使用公开的Med‑TC（医学摘要分类）、MedMCQA（多选问答）和MedQA‑USMLE（临床问答）三个基准。

**📈 对比分析**

与全文本加密、未加密基线对比；在分类任务中，HERALD可恢复约10–15%精度差距；在多选问答中，保留选项不加密时准确率提升至原始≈70%水平，完全加密则下降至接近随机。

**⚠️ 局限性**

局限包括：对上下文泄露仍有统计推断风险；加密过程引入序列膨胀和延时；需要安全密钥管理；对强推理任务（如USMLE）仍难以完全恢复；对大规模数据可能出现密钥冲突或频繁加密导致性能下降。

---

## 432. Grasp-Then-Plan with Failure Attribution: A Closed Two-Stage Framework for Precise and Generalizable Robotic Manipulation

**arXiv ID:** 2606.03385 | [PDF](https://arxiv.org/pdf/2606.03385v1)

**作者:** Jiahao Xu `[一作]` (Southeast University), Wanyuan Wang `[通讯]` (Southeast University)

**通讯引用:** 741 | [OpenAlex ID](https://openalex.org/A5054650835)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出一种闭环Grasp‑Then‑Plan with Failure Attribution（GTP‑FA）框架，能够在执行-诊断-更新循环中将抓取与规划耦合拆分并动态优化；

**💡 创新点**

关键创新在于构建三类失败归因模型（抓取功能不匹配、抓取不稳定、规划不足），并将归因结果作为责任权重，指导抓取候选筛选与规划策略的针对性改进，实现跨学习范式的协同提升；

**🔧 技术方法**

利用任务感知抓取候选生成、失效归因判别器、抓取条件嵌入空间、风险预测、VLM任务优先级约束、分布重塑、责任加权训练及VLA微调等技术；

**📊 数据集**

实验基于ManiSkill3仿真八个长时序任务和Franka Research 3真实机器人，使用100-300条专家轨迹；

**📈 对比分析**

与PPO、BC、DP、SAC以及VLA基线对比，GTP‑FA在所有基线上提升终端成功率，平均增益分别为+31.3%、+54.0%、+25.7%、+7.7%和+17.8%；在真实机器人上从11.2%提升至76.8%；

**⚠️ 局限性**

局限在于归因模型对未见抓取的鲁棒性仍依赖邻域先验，且对复杂多阶段任务的可扩展性与在线细粒度反馈机制尚待完善。

---

## 433. MeDxAgent: Multi-Agent Consultation for Interactive Medical Diagnosis

**arXiv ID:** 2606.03416 | [PDF](https://arxiv.org/pdf/2606.03416v1)

**作者:** Akshat Sanghvi `[一作]` (Microsoft Research India), Mohit Jain `[通讯]` (Microsoft Research India)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

创建了一个包含4,421个跨20个科室病例的开放式交互诊断基准，并提出了多代理交互诊断系统

**💡 创新点**

系统化评估提示、流程和代理层设计，并通过组合实现诊断准确率提升52.3%相对全信息oracle

**🔧 技术方法**

采用GPT‑4o等大型语言模型，结合提示工程、问诊代理、诊断代理、总结代理、专家集成、知识图谱与证据缺口四个模块

**📊 数据集**

合并了CRAFT‑MD、DiagnosisArena、MedMCQA、MedQA、PubMed等五个公开数据集构建基准

**📈 对比分析**

与基线、Oracle以及MedAgentSim、VivaBench等系统对比，GPT‑4o上平均诊断准确率57.4%，比基线高10.3个百分点，显著优于其他公开系统

**⚠️ 局限性**

仅处理文本诊断，模拟患者回答过度“我不知道”，未考虑多模态证据，且问诊策略仅通过提示工程实现，缺乏学习型策略

---

## 434. Extreme Motion Generation via Hybrid Null-Space Control for Straight-Line Path Following

**arXiv ID:** 2606.03390 | [PDF](https://arxiv.org/pdf/2606.03390v1)

**作者:** Xinyi Yuan `[一作]` (University of Osaka), Kensuke Harada `[通讯]` (University of Osaka)

**通讯引用:** 11134 | [OpenAlex ID](https://openalex.org/A5016270703)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究一种极限运动生成框架，目标是在固定基础机械臂沿预定义直线尽可能远地运动；

**💡 创新点**

通过在控制层面实现RL与经典模型的混合：在内部空间使用RL预测长期最优路径，在靠近关节极限时切换到鲁棒的经典控制；同时使用条件扩散模型为起始姿态在自运动流形中采样多模态解，提升起始分支的优势；

**🔧 技术方法**

混合零空间控制、强化学习（PPO）、条件扩散模型（DDIM+CFG）、零空间梯度分解、基于阈值的滞后切换规则；

**📊 数据集**

在7-DOF Franka FR3机械臂上采集的10000条直线跟随任务（起点、方向、平面法线），其中10%保留为测试集；

**📈 对比分析**

与单独的经典控制器和RL控制器对比；在10k任务上，Hybrid平均实现路径长度比经典提升~27%，RL提升~17%，且平均能达到参考长度的90%（最高可达100%+），并在不同难度分段（Easy/Medium/Difficult）保持优越；

**⚠️ 局限性**

仍受限于：RL在接近关节极限时性能下降；混合切换阈值需要调参，可能导致切换抖动；仅验证直线跟随，缺乏动态或多任务场景；扩散模型训练需要大量标注数据，且对非固定任务的泛化未知。

---

## 435. BlobShuffle: Cost-Effective Repartitioning in Stream Processing Systems via Object Storage Exemplified with Kafka Streams

**arXiv ID:** 2606.03364 | [PDF](https://arxiv.org/pdf/2606.03364v1)

**作者:** Sören Henning `[一作]` (Dynatrace Research), Adriano Vogel `[通讯]` (Dynatrace Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出 BlobShuffle，一种利用云对象存储的成本高效分区重排方案，减少 Kafka Streams 中跨可用区 shuffle 的网络流量。

**💡 创新点**

创新点在于：在客户端实现批量化写入对象存储、仅通过 Kafka 发送压缩通知，并通过分布式与本地缓存降低对象存储访问次数，保持 Kafka Streams 的一致性保证。

**🔧 技术方法**

采用 Kafka Streams API、Amazon S3 对象存储、分布式 LRU 缓存、分区分批策略和可配置的批量大小等技术。

**📊 数据集**

使用自研 ShuffleBench 生成 1 KiB 随机记录流，部署于 AWS EKS Kubernetes 集群，结合多 AZ 的 Kafka 集群进行评估。

**📈 对比分析**

通过与原生 Kafka Streams shuffle 的对比，BlobShuffle 在 16 MiB 批量设置下，shuffle 95% 分位延迟 < 2 s，成本降低 40×（跨 AZ 费用从 192 USD/h 降至约 4 USD/h），吞吐量可达 2 GiB/s。

**⚠️ 局限性**

限制包括：额外的延迟（主要由 S3 PUT 操作造成）、需手动配置批量大小、对高并发多线程性能待优化、以及对象存储存储费用随批量大小和保留周期变化。

---

## 436. SIGMA: A Versatile Streaming Graph Partitioner for Vertex- and Edge-Balanced Distributed GNN Training

**arXiv ID:** 2606.03519 | [PDF](https://arxiv.org/pdf/2606.03519v1)

**作者:** Barbara Hoffmann `[一作]` (University of Bayreuth), Christian Schulz `[通讯]` (Heidelberg University)

**通讯引用:** 11691 | [OpenAlex ID](https://openalex.org/A5101948098)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种名为SIGMA的流式图划分算法，支持顶点和边的划分，旨在优化分布式图神经网络（GNN）训练的通信效率和负载平衡。

**💡 创新点**

SIGMA的创新点在于其统一的多目标、多约束框架，能够同时优化边切和顶点复制，同时考虑顶点和边的平衡，克服了传统方法的局限性。

**🔧 技术方法**

使用了流式图划分技术，结合聚类预处理阶段，以提高划分质量，同时保持流式划分的效率和可扩展性。

**📊 数据集**

在六个基准图数据集上进行评估，包括Amazon Computers、Flickr、Reddit、Twitch、ogbn-arxiv和ogbn-products，涵盖多个领域和规模。

**📈 对比分析**

与多种现有的划分算法（如METIS、KaHIP、HEP等）进行比较，SIGMA在划分质量、训练效率和内存消耗方面表现出色，常常超越流式基线，并与高质量的内存划分器保持竞争力。

**⚠️ 局限性**

限制在于SIGMA的性能可能受到流式处理的顺序和图的结构特征的影响，可能在某些情况下无法达到最优划分。

---

## 437. SPADE: Sketch-guided Path Planning Augmented with Diffusion Experts

**arXiv ID:** 2606.03512 | [PDF](https://arxiv.org/pdf/2606.03512v1)

**作者:** Charbel Abi Hana `[一作]` (IDEALworks GmbH), Anthony Rizk `[通讯]` (IDEALworks GmbH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

开发了 SPADE 框架，通过图像条件扩散模型与行为克隆实现轻量级 AMR 路径规划。

**💡 创新点**

创新点包括：1）基于 ROS2 的全新开源注释工具；2）将扩散专家引导融入行为克隆（Cond-DBC）；3）使用图像条件扩散（FiLM）提升小模型的泛化能力。

**🔧 技术方法**

采用行为克隆、扩散概率模型（DDPM/Cond-DBC）、FiLM 条件化、图像增强、ROS2 Nav2 等技术。

**📊 数据集**

使用 20,000 条手绘 L 形和 U 形路径样本以及 2,000 条工业地图测试样本构成的数据集。

**📈 对比分析**

与原 SKIPP 及无条件扩散 DBC 对比，Cond-DBC 在小模型上实现 39.1% APE 降低、33.5% FID 降低，且 Artifact 率接近 0，整体性能显著提升。

**⚠️ 局限性**

局限性：扩散模型推理延迟仍高，难以满足实时需求；目前仅支持静态环境；对动态障碍物和多模态时序任务的适应性有限。

---

## 438. Secrecy Sum Rate Maximization for OIRS-Aided Visible Light Communications with Confidential Messages

**arXiv ID:** 2606.03505 | [PDF](https://arxiv.org/pdf/2606.03505v1)

**作者:** Trinh K. Nguyen `[一作]` (Hanoi University of Science and Technology), Chuyen T. Nguyen `[通讯]` (Hanoi University of Science and Technology)

**通讯引用:** 707 | [OpenAlex ID](https://openalex.org/A5026603079)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出并分析了在存在视线阻塞和内部窃听的多用户MISO可见光通信系统中，利用光学智能反射表面（OIRS）与波束成形预编码联合优化，以最大化机密信息的和信封率（SSR）。

**💡 创新点**

创新点在于：①首次在同一系统中同时考虑视线阻塞、内部窃听以及OIRS带来的非视线传播；②将OIRS单元分配与预编码器设计联合为一个混合整数非线性规划（MINLP）；③使用交替优化（AO）配合凹凸分解（CCCP）与一阶Taylor近似，提供高质量的子最优解。

**🔧 技术方法**

核心技术包括：光学智能反射表面建模、预编码设计、交替优化框架、凹凸分解（CCCP）、一阶Taylor线性化、惩罚方法对二进制约束的松弛以及凸优化求解。

**📊 数据集**

采用仿真数据：室内环境为5m×5m×3m，部署4个LED灯、2组OIRS阵列（每组72或其他单元数），用户均匀分布于0.5m高度平面，设置1m×1m×2.5m阻塞物体；所有参数均来源于文中所述仿真配置，未使用公开数据集。

**📈 对比分析**

通过比较不同OIRS单元数下的SSR与收敛迭代次数，展示了所提方法相较于无OIRS或较少单元时的性能提升；SSR随OIRS单元数递增而单调提升，收敛速度在大单元数时显著加快；实验结果表明在两用户被阻塞的情况下SSR更高，验证了OIRS在提升机密通信中的有效性。

**⚠️ 局限性**

局限性包括：①采用简化的镜面反射模型，未考虑实际OIRS反射的多路径、波束损耗等复杂物理效应；②仅在仿真环境下验证，缺乏实际硬件实验；③内部窃听模型假设为用户间纯粹信息泄露，未考虑更复杂的攻击方式；④算法在大规模用户/LED情况下的计算复杂度未进行深入评估。

---

## 439. BaltiVoice: A Speech Corpus and Fine-tuned Whisper ASR System for the Balti Language

**arXiv ID:** 2606.03504 | [PDF](https://arxiv.org/pdf/2606.03504v1)

**作者:** Muhammad Ali `[一作]` `[通讯]` (Independent Researcher), Muhammad Ali (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并发布了Balti语言的16.8小时语音语料库BaltiVoice，并基于此微调了Whisper-small模型。

**💡 创新点**

首次公开Balti语料和可复现的微调流程，为低资源语言ASR奠定基准，创新点在于提供公共数据与模型。

**🔧 技术方法**

采用OpenAI Whisper-small作为基础模型，利用HuggingFace Transformers进行迁移学习和微调。

**📊 数据集**

使用来自Mozilla Common Voice的Balti语音及其Nastaliq文字转录，共10,060条验证录音。

**📈 对比分析**

与零样本Whisper-small对比，微调后在538条验证集上将WER从182.18%降至30.07%。

**⚠️ 局限性**

局限包括样本仅为朗读语音，未覆盖对话；未做文本规范化导致形态学错误。

---

## 440. From 3D Perception to Safety Reasoning: A Graph-Based Framework for Real-Time Underground Mine Monitoring

**arXiv ID:** 2606.03460 | [PDF](https://arxiv.org/pdf/2606.03460v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 441. Characterizing Detectability in 3DGS Poisoning: A Stage-wise Benchmark

**arXiv ID:** 2606.03499 | [PDF](https://arxiv.org/pdf/2606.03499v1)

**作者:** Quoc-Anh Bui-Huynh `[一作]` (Singapore University Of Technology And Design), Ngai-Man Cheung `[通讯]` (Singapore University Of Technology And Design)

**通讯引用:** 5605 | [OpenAlex ID](https://openalex.org/A5057453537)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 Poison-3DGS 基准，用于在 3D 高斯喷射（3D Gaussian Splatting）多阶段流水线中对毒化攻击进行阶段性检测，并系统评估了多种攻击与检测方法。

**💡 创新点**

创新点在于首次从防御者视角引入阶段性取证信号的概念，构建统一的检测协议，揭示不同攻击在各流水线阶段的可检测性差异，以及检测器与阶段特定信号的匹配关系。

**🔧 技术方法**

技术包括：构建多阶段检测器套件（图像、SfM 3D、训练动态、最终高斯参数），应用现有视觉异常检测和模型分析方法（MVSS-Net、IML‑ViT、MET3R、GradNorm、Gaussian‑MAE 等），以及设计对应的聚合和评分流程。

**📊 数据集**

使用 37 个来自 Free、Mip‑NeRF360、Tanks‑and‑Temples 的干净场景，并在每个场景上生成 414 种毒化变体（覆盖 4 类攻击），形成完整的阶段性取证数据集。

**📈 对比分析**

通过 AUROC 与 FPR@95TPR 指标对各阶段检测器进行无监督评估；实验显示检测效果随阶段变化显著，最高 AUROC 可达 0.8 左右，但不存在单一阶段在所有攻击类型中均最优，说明需结合多阶段信号才能实现鲁棒检测。

**⚠️ 局限性**

局限性包括：基准仅覆盖现有公开攻击，缺乏对未来新型毒化手段的适应；检测器多为无监督方法，未考虑训练成本或对抗训练的影响；评估仅在场景级别，未探讨更细粒度或跨域迁移的性能。

---

## 442. Low-Frequency Shortcuts in Texture-Driven Visual Learning

**arXiv ID:** 2606.03493 | [PDF](https://arxiv.org/pdf/2606.03493v1)

**作者:** Utku Şirin `[一作]` (Harvard University), Stratos Idreos `[通讯]` (Harvard University)

**通讯引用:** 4586 | [OpenAlex ID](https://openalex.org/A5026905380)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究纹理驱动视觉任务中普遍存在的低频捷径现象，并通过在训练和测试集上裁剪低频DCT系数来平衡频谱，从而提升模型在ID和OOD环境下的性能。

**💡 创新点**

首次揭示纹理驱动域在低频系数上的偏倚导致模型过度依赖低频捷径，并提出简单的频域剪枝策略来消除该偏倚，进而显著提升ID与OOR的泛化；同时验证该现象跨模型、规模及预训练策略均成立。

**🔧 技术方法**

利用离散余弦变换（DCT）对图像进行频域分解并按对角线顺序裁剪低频/高频/中频系数；通过训练多种CNN、ViT以及预训练的DinoV2/CLIP模型；采用频谱贡献分析评估每个频率成分对准确率的影响；使用ImageNet-C的低频、中频、高频噪声进行OOR评估。

**📊 数据集**

SP‑Colorectal、TextileNet、GTOS、GalaxyMNIST、EuroSAT以及标准CIFAR‑10六个数据集，覆盖病理图像、纺织品、地形、天文图像、卫星影像与通用对象识别。

**📈 对比分析**

对比未裁剪的基准模型，裁剪低频系数后ID准确率提升最高达8%；在低频噪声（如雾）OOR准确率提升高达40%；在高频噪声（如高斯模糊）下表现出现折中；跨ResNet、MobileNet、ViT及预训练模型均表现一致。

**⚠️ 局限性**

仅针对图像分类任务；仅使用单一的DCT裁剪策略，未探索量化、子采样等其它压缩技术；未评估在资源受限环境（如边缘设备）下的压缩-性能权衡；对裁剪后模型在更广泛任务与域上的迁移性能仍需进一步研究。

---

## 443. NeuroArmor: Safe-Variant-Guided Representation Consistency for Selective Re-Anchoring in Jailbreak Defense

**arXiv ID:** 2606.03486 | [PDF](https://arxiv.org/pdf/2606.03486v1)

**作者:** Zhongyang Lin `[一作]`, Pengyuan Liu `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于提示特定安全变体的白盒运行时防御NeuroArmor，利用内部表示一致性检测来决定是否介入，并根据检测结果将输入路由到拒绝分支或有益恢复分支。

**💡 创新点**

创新点在于：①把提示的安全变体作为局部内部参考点同时用于一致性判定与干预目标；②使用表示一致性特征与安全方向投影联合检测；③在同一模型上实现拒绝与恢复的双路由决策，兼顾攻击抑制与对准安全。

**🔧 技术方法**

技术手段包括：生成/检索K个安全变体，计算隐藏状态与安全方向的投影/余弦一致性特征，使用One-Class SVM + 阈值规则做一致性检测，基于检测结果在固定层插桩隐藏状态实现向安全中心或安全方向的再锚定干预，支持单轮和多轮对话。

**📊 数据集**

主要使用Llama‑3‑8B‑Instruct与Gemma‑2‑9B模型，评测数据集涵盖多种 jailbreak 家族（AutoDAN、DrAttack、GCG等）、HarmBench、SafeMTData、XS‑Test、OR‑Bench‑Hard等。

**📈 对比分析**

与Vanilla、Direct Safety Steering、Runtime Intervention Baseline、Llama‑Guard‑3等基线对比。NeuroArmor 在 Llama‑3 上将攻击成功率从 41.56% 降至 1.57%，并把可疑请求的误报率从 30.26% 降至 22.05%。外部 GPT‑4 评判显示剩余非阻断回复的有害比例降至 0.87%，相较基线提升显著。

**⚠️ 局限性**

局限性：依赖白盒模型；检测器仅为简单 One‑Class SVM，可能有提升空间；安全变体模板与阈值调参对性能影响大；对适应性攻击和更复杂边界敏感对话的鲁棒性仍待验证；实验仅在公开数据集与模型上进行，实际部署仍需更广泛评估。

---

## 444. StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems

**arXiv ID:** 2606.03467 | [PDF](https://arxiv.org/pdf/2606.03467v1)

**作者:** Taiyu Zhu `[一作]` (Peking University), Gang Huang `[通讯]` (Peking University)

**通讯引用:** 15359 | [OpenAlex ID](https://openalex.org/A5100726056)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了StepFinder，一种用于多智能体系统（MAS）失败定位的轻量级框架

**💡 创新点**

创新点包括：仅在编码阶段使用LLM，利用时序语义序列+双向LSTM与代理感知注意力捕捉跨步依赖；引入多尺度差分、位置偏置和时序一致性损失提升根因定位精度；实现了显著的推理效率提升

**🔧 技术方法**

技术手段包括：LLM嵌入（Qwen3 Embedding）、双向LSTM时序特征提取、全局注意力+代理感知偏置与门控、非线性投影错误评分、多尺度差分与位置偏置、未来步预测辅助损失

**📊 数据集**

使用公开的Who&When基准数据集（Alg与HC子集）进行训练与评估

**📈 对比分析**

与随机、LLM驱动、传统序列模型（BiGRU、TCN、Transformer）对比，StepFinder在Alg子集准确率提升至29.63%（远超20.11%），HC子集22.99%（超20.11%）；Acc@K和MRR@3也均优于所有基线；推理时间比最快的LLM方法低79%，并无文本生成开销

**⚠️ 局限性**

局限性包括：仍需依赖LLM进行文本编码，可能受LLM性能限制；实验仅覆盖Who&When基准，未验证跨领域泛化；在HC子集的推理时间相对较长（3.56s），仍有优化空间

---

## 445. Post-Hoc Robustness for Model-Based Reinforcement Learning

**arXiv ID:** 2606.03521 | [PDF](https://arxiv.org/pdf/2606.03521v1)

**作者:** Siemen Herremans `[一作]` (IDLab - University of Antwerp, imec), Siegfried Mercelis `[通讯]` (IDLab - University of Antwerp, imec)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在推理阶段对已训练的MBPO模型进行后置鲁棒化，利用模型预测控制（MPC）和PGD对转移模型进行对抗性不确定性逼近，以提升政策在环境扰动下的表现。

**💡 创新点**

不需要额外训练或重新调整策略网络，提出一种“后置鲁棒化”框架；通过在MPC过程中加入对抗性扰动与基于ensemble的OOD截断，兼顾计算效率与鲁棒性。

**🔧 技术方法**

使用MBPO（模型基础策略优化）框架、MPC、对抗梯度下降（PGD）、OTC不确定性集、ensemble方差截断、Jax实现的GPU加速。

**📊 数据集**

在Gymnasium MuJoCo的Reacher‑v4和Hopper‑v4两大控制任务上进行实验，测试机器人躯干质量、摩擦系数、阻尼系数等参数扰动。

**📈 对比分析**

与原始MBPO基线以及无对抗扰动的MPC做对比；在单一及双重扰动条件下，后置鲁棒化显著降低性能衰退（平均回报提升约20%~30%），并在RTX 4090、RTX 2080 Ti等GPU上实现每步0.01–0.15秒的推理时间。

**⚠️ 局限性**

缺乏正式的鲁棒性保证，依赖于对名义价值函数的惰性估计；对较长回溯或更复杂环境的泛化性尚未验证；在极端扰动下可能仍出现性能下降。

---

## 446. The Attention-Aware Pipeline: Design Tensions from Making Attention Visible in XR

**arXiv ID:** 2606.03492 | [PDF](https://arxiv.org/pdf/2606.03492v1)

**作者:** Arvind Srinivasan `[一作]` (Aarhus University), Niklas Elmqvist `[通讯]` (Aarhus University)

**通讯引用:** 8278 | [OpenAlex ID](https://openalex.org/A5034277315)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e0540dec-d77f-42db-94ae-d039248f6393` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并验证了一个注意感知管道（Capture‑Record‑Revisualize）框架，构建了三种配置（镜像、媒介、调解）并在四名乐团成员的眼动数据上进行需求挖掘，探讨了设计张力与可视化反馈循环。

**💡 创新点**

系统性地将注意力可视化与 XR 反馈循环结合，首次用管道分析方法揭示镜像导致自我强化、媒介产生注意力悖论、调解出现目标悖论等设计张力，提供了面向社交目标的减淡干预概念。

**🔧 技术方法**

采用眼动追踪（Pupil Labs Neon）、XR 头显实时 gaze 捕获、YOLOv8 动态 AOI 标注、记录与建模、可视化叠加（热图、去饱和）以及减淡现实（Diminished Reality）技术。

**📊 数据集**

四名音乐家现场演奏的眼动数据（Pupil Labs 记录）、HeedVision 交互实验数据、AAV 的 12 名参与者实验数据。

**📈 对比分析**

通过定性访谈和实验任务（散点图搜索、地形搜索、乐曲视谱阅读）对比，AAV 在稀疏环境中提升系统探索度；HeedVision 在稀疏场景下提升协作效率，但在密集场景出现干扰；性能评估主要基于用户满意度、任务完成时间和错误率，未给出量化性能指标。

**⚠️ 局限性**

缺乏对注意力干预效果的定量测评；对社交目标减淡干预的实验验证不足；系统在多人动态社交场景中的可扩展性、隐私/监控问题与用户信任仍未解决。

---

## 447. TrAction: Action Recognition with Sparse Trajectories

**arXiv ID:** 2606.03490 | [PDF](https://arxiv.org/pdf/2606.03490v1)

**作者:** Jan F. Meier `[一作]` (University Göttingen), Timo Lüddecke `[通讯]` (University Göttingen)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种基于稀疏点轨迹的动作识别方法。

**💡 创新点**

创新点在于将2.5D点轨迹作为纯运动输入，结合Transformer编码器和掩码轨迹自监督预训练，实现对时序动态的显式建模。

**🔧 技术方法**

采用CoTracker3进行点跟踪、单目深度估计获取3D信息、Transformer架构及MAE式掩码预训练。

**📊 数据集**

在Something‑Something v2、EPIC‑Kitchens‑100以及Kinetics‑400等公开数据集上进行评估。

**📈 对比分析**

在SSv2上获得45.2% top‑1，EK100上54.1%；与DINOv2、V‑JEPA等稠密RGB模型融合后进一步提升约8–9个百分点，表明轨迹特征具有显著补充作用。

**⚠️ 局限性**

局限性包括需要额外的轨迹提取步骤、对剧烈相机运动和场景突变的鲁棒性不足，以及在强视觉依赖的任务（如Kinetics‑400）中表现不佳。

---

## 448. PersistGS: Differentiable Physics for Object Permanence in 4D Gaussian Splatting

**arXiv ID:** 2606.03479 | [PDF](https://arxiv.org/pdf/2606.03479v1)

**作者:** Adrian Ramlal `[一作]` (University of Waterloo), John S. Zelek `[通讯]` (University of Waterloo)

**通讯引用:** 2309 | [OpenAlex ID](https://openalex.org/A5077041853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

通过可微刚体仿真与3D高斯展平实现对象在时间遮挡期间的永久性重建。

**💡 创新点**

将可微刚体模拟作为物理先验，引入中心轮廓损失和观测意识课程来估计摩擦和初速度，从而生成物理一致的SE(3)轨迹。

**🔧 技术方法**

3D Gaussian Splatting、NVIDIA Newton可微刚体仿真、中心轮廓损失、可微渲染、光度监督及稀疏视角正则化等技术。

**📊 数据集**

三组合成场景（ball_fall、ball_bounce、ball_roll）模拟的同步多摄像机视频，包含完整轨迹的训练与评估摄像机。

**📈 对比分析**

与常数速度、线性插值、无物理等基线比较；PersistGS在遮挡期间平均PSNR提升约2.46 dB，接近真实轨迹上限，轨迹RMSE显著低于基线。

**⚠️ 局限性**

仅验证球形对象，难以同时估计旋转；对摩擦等参数的可观测性有限；依赖稀疏视角正则化；多对象碰撞与自动分解等场景仍待研究。

---

## 449. Human2Humanoid: Physics-Aware Cross-Morphology Motion Retargeting for Humanoid Robots

**arXiv ID:** 2606.03476 | [PDF](https://arxiv.org/pdf/2606.03476v1)

**作者:** Tianchen Huang `[一作]` (University of Science and Technology of China), Shiwu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 22813 | [OpenAlex ID](https://openalex.org/A5069207720)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出 Human2Humanoid，一种基于 CycleGAN 的无配对人体运动重定向框架，能够将人类运动映射到 Unitree G1 人形机器人。

**💡 创新点**

创新点包括：① 使用形态不变的末端执行器一致性损失，利用 T‑pose 标准化对齐跨体型轨迹；② 引入物理感知的可行性约束（接触、脚高、关节限位）；③ 在生成器中采用骨架感知图卷积网络。

**🔧 技术方法**

技术手段主要有：CycleGAN + LSGAN、骨架感知图卷积网络、形态不变末端执行器一致性损失、物理可行性约束、周期一致性与身份损失。

**📊 数据集**

数据集使用无配对的人类动作集 Motion‑X 与基于物理过滤的 Unitree G1 子集 PHUMA 进行训练与评估。

**📈 对比分析**

与 PHC、GMR 及工业级 Unitree Retarget 进行对比，Human2Humanoid 在仿真追踪成功率、跟踪误差、脚滑与地面穿透等指标均优于对手，显示出更高的可控性与物理可行性。

**⚠️ 局限性**

局限性包括：仍受限于人形机器人的骨架结构差异，难以直接推广至结构差异更大的非人形机器人；以及对超大规模姿态变化和复杂环境交互的鲁棒性尚需进一步验证。

---

## 450. Tonal parsimony in chord-sequence analysis: combining modulation cost and tonal vocabulary

**arXiv ID:** 2606.03459 | [PDF](https://arxiv.org/pdf/2606.03459v1)

**作者:** François Pachet `[一作]` (Sorbonne Université), François Pachet `[通讯]` (Sorbonne Université)

**通讯引用:** 3654 | [OpenAlex ID](https://openalex.org/A5111498117)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在和弦序列上进行本地调性分配，并对三种目标函数（仅调性转换、最小调性集合、音调简约）进行实验与对比。

**💡 创新点**

提出音调简约（tonal parsimony）目标，将调性转换成本与调性词典大小按字典序优化，并给出在固定24调域上的高效精确动态规划算法，并引入爵士替换闭包以扩充候选域。

**🔧 技术方法**

采用约束规划中的Regular/CostRegular和NValue约束实现全局优化，并使用位掩码与标签搜索进行动态规划加速；同时对替换规则进行预计算闭包。

**📊 数据集**

主要使用LMD Chords语料库（31,032段和弦序列）进行大规模评估，并在Jazz Standards Progressions Book（1,555首专业注释曲目）进行外部验证。

**📈 对比分析**

通过比较平均/中位数的调性转换次数和调性词典大小、CPU时间以及与专业和弦-音阶注释的严格、兼容和宽松匹配率来评估方法；音调简约在保持最小调性转换的同时大幅减少调性词典，匹配率最高（约95.6%）。

**⚠️ 局限性**

仅在固定24调域内可行，面对相对大调、反压缩等情况时效果不佳；且实现仍需双层约束才能在生成阶段强制目标调性词典大小。

---

## 451. PRISM: Synergizing Vision Foundation Models via Self-organized Expert Specialization

**arXiv ID:** 2606.03444 | [PDF](https://arxiv.org/pdf/2606.03444v1)

**作者:** Ying Tang `[一作]` (Huazhong University of Science and Technology), Wei Yang `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 38556 | [OpenAlex ID](https://openalex.org/A5100781368)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `8d10c613-917e-4880-9716-17789f50e119` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出PRISM框架，利用双流Mixture-of-Experts实现多视觉基础模型的融合与蒸馏。

**💡 创新点**

创新点在于自组织专家分化与上下文调制路由，解决多教师蒸馏的梯度冲突，实现软边界的动态知识分解与重组。

**🔧 技术方法**

采用双流MoE、FiLM路由、局部去相关正则、两阶段训练等技术。

**📊 数据集**

在ImageNet‑1K预训练后，分别在PASCAL‑Context和NYUD‑v2上评估多任务性能。

**📈 对比分析**

与SAK、RADIO等基线相比，PRISM在PASCAL‑Context平均提升2.29% mIoU，在NYUD‑v2上超越SAK的语义与深度指标，整体性能显著提升。

**⚠️ 局限性**

局限在于对不同数据集的平衡仍需手工调优，且稀疏专家的推理延迟相对较高。

---

## 452. FlowGuard: Flow Matching for Identity-Independent Detection of Data-Free Model Stealing Attacks on Energy System Intrusion Detection Systems

**arXiv ID:** 2606.03430 | [PDF](https://arxiv.org/pdf/2606.03430v1)

**作者:** Maxime Schwarzer `[一作]` (CortAIx Labs, Thales Deutschland), Veit Hagenmeyer `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 5182 | [OpenAlex ID](https://openalex.org/A5014228448)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种身份无关的模型窃取检测方法 FlowGuard

**💡 创新点**

利用流匹配训练连续正则化流对合法流量进行密度建模，并以低似然作为判别依据，在 Sybil 分布式攻击下仍能保持高检测率

**🔧 技术方法**

连续正则化流（CNF）+流匹配（Flow Matching）技术，对查询进行 log‑likelihood 计算

**📊 数据集**

CIFAR‑10 数据集下的 VGG16-BN 目标模型

**📈 对比分析**

与 PRADA、FDINet 进行对比。单客户端场景下 FlowGuard TPR≈0.965–1.0，F1≈0.904–0.919，ROC‑AUC≥0.922；分布式（100 客户端）场景中 PRADA 失效（TPR=0），FDINet 假正率高，FlowGuard 仍保持稳定检测

**⚠️ 局限性**

实验仅在 CIFAR‑10 单模型、单轮次、数据自由窃取攻击下进行，未覆盖自适应攻击、其他数据集和多模型情况，缺乏置信区间，需进一步扩展验证

---

## 453. A Community Survey on SHACL and ShEx: Briding Gaps in RDF Validation

**arXiv ID:** 2606.03502 | [PDF](https://arxiv.org/pdf/2606.03502v1)

**作者:** Maxime Jakubowski `[一作]` (TU Wien), Katja Hose `[通讯]` (TU Wien)

**通讯引用:** 4499 | [OpenAlex ID](https://openalex.org/A5015313855)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本研究通过针对学术与产业界的94位RDF验证技术使用者开展社区调查，系统分析了当前RDF验证实践、工具使用、形状（Shape）创建方法以及用户所面临的主要挑战，并公开了可查询的DuckDB数据集和可复现的图表；

**💡 创新点**

创新点在于：①首次以较大样本量对SHACL与ShEx的使用现状进行纵向对比；②发现行业用户更频繁使用高级特性（如SHACL规则、SPARQL约束）并对递归形状、RDF‑Star等新特性提出需求；③提供了开源、可交互的数据探索平台，促进社区共建与验证方法的改进；

**🔧 技术方法**

使用的技术包括：Google Forms在线调查、SQL查询（DuckDB）、数据可视化、对SHACL/SHEx工具（如Apache Jena、PySHACL、TopBraid、RDFShape等）的使用统计与分析；

**📊 数据集**

主要数据集为本次社区调查的原始响应（94份），并对比了2022年的调查数据以实现纵向分析；

**📈 对比分析**

通过对不同用户群体（学术vs产业）在工具使用、形状创建方式、图规模与验证频率等维度进行对比，发现产业用户在大规模图（>1M三元组）下更依赖高级特性；性能方面，用户普遍反映大图验证存在瓶颈，但未提供客观基准；

**⚠️ 局限性**

局限性包括：样本规模相对有限，细粒度分析缺乏统计显著性；数据以自评为主，可能存在主观偏差；缺乏统一的性能基准与工具覆盖范围；对自由文本答案的映射仍有不确定性；

---

## 454. EvoMemNav: Efficient Self-Evolving Fine-Grained Memory for Zero-Shot Embodied Navigation

**arXiv ID:** 2606.03509 | [PDF](https://arxiv.org/pdf/2606.03509v1)

**作者:** Zuhao Ge `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 25063 | [OpenAlex ID](https://openalex.org/A5047962986)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个以视图为一等节点、以房间和物体为层次结构的视觉语义记忆图（VSMGraph），并设计了预算化的粗细两阶段导航策略（Explore + Search+Verify）以及无训练的反思式持续记忆更新机制（RDCMA）来实现零样本多模态导航。

**💡 创新点**

创新点在于：①将原始图像证据直接作为记忆节点，避免传统场景图压缩导致的细节丢失；②通过房间标签与物体可见性软标签实现候选压缩，显著降低VLM调用次数；③在候选压缩后才使用VLM进行精细选择与多视图停靠验证；④引入反思式写回，将子任务的轨迹与停靠结果写入图节点的先验，形成训练无关的自我改进机制。

**🔧 技术方法**

主要技术包括：基于CLIP的房间标签分类、YOLOv8-World与SAM生成可见性标签、Qwen3‑VL‑8B作为视觉语言模型、VSMGraph结构化存储、预算化粗细两阶段决策、反思式写回（RDCMA）以及基于图的前向搜索与多视图停靠验证。

**📊 数据集**

使用了两大公开数据集：GOAT‑Bench（多模态终身导航，包含目标类别、语言描述和参考图像）和HM3D（大规模3D室内导航，包含对象目标任务），在 Habitat‑Lab 仿真环境下评估。

**📈 对比分析**

与现有无训练的基线（如3D‑Mem、MSGNav、TANGO等）以及部分训练模型进行对比。EvoMemNav 在 GOAT‑Bench 的未见验证集上实现了 59.6% SR / 38.9% SPL（相比最佳基线提升约 15%），在 HM3D 的物体目标任务上也取得了 59.2%/33.6%（v.s. 58.1%/31.2%）。同时在 VLM 调用次数和图维护时间上相较传统方法显著降低。

**⚠️ 局限性**

局限性包括：①仍依赖昂贵的 VLM 推理，虽然已被压缩但在大规模任务中可能成为瓶颈；②图结构的构建与维护需要额外的计算资源，尤其是在高度动态或极大场景中；③反思式写回仅基于离线统计，无法处理需要深度学习模型更新的复杂策略；④对 CLIP 房间标签和可见性标签的精度有限，可能影响候选压缩效果。

---

## 455. HiSE: A Lightweight Hierarchical Semantic Explainer for Heterogeneous Graph Neural Networks

**arXiv ID:** 2606.03495 | [PDF](https://arxiv.org/pdf/2606.03495v1)

**作者:** Zongrui Li `[一作]` (Jilin University), Yuan Tian `[通讯]` (Jilin University)

**通讯引用:** 5023 | [OpenAlex ID](https://openalex.org/A5028722095)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

论文提出HiSE，一个轻量化的分层语义可解释框架，用于解释异构图神经网络的决策。

**💡 创新点**

创新点在于在语义层构建稀疏LASSO代理模型，并通过KL散度在跨语义层加权融合，完整捕捉HGNN的层次语义贡献。

**🔧 技术方法**

采用了LASSO回归、KL散度加权、meta-path拆分、局部邻域采样等技术。

**📊 数据集**

评估使用ACM和MAG两个学术网络异构图数据集。

**📈 对比分析**

与GraphLIME、GNNExplainer、HGExplainer、HENCE-X等方法对比，HiSE在解释可信度、鲁棒性、可用性、跨语义一致性上均优于基线，并且计算速度快2-3个数量级。

**⚠️ 局限性**

限制在于需要预先定义meta-path或注意力头作为语义单元，对不同HGNN架构的自适应性还有待改进。

---

## 456. Learn from Your Mistakes: Tree-like Self-Play for Secure Code LLMs

**arXiv ID:** 2606.03489 | [PDF](https://arxiv.org/pdf/2606.03489v1)

**作者:** Wenqi Chen `[一作]` (University of Electronic Science and Technology of China), Zhengsu Chen `[通讯]` (Beihang University)

**通讯引用:** 352 | [OpenAlex ID](https://openalex.org/A5042114598)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Tree-like Self-Play（TSP）框架，利用大语言模型在代码生成树上进行自对弈，细粒度学习安全决策节点，从而显著提升生成代码的安全性。

**💡 创新点**

创新点在于将安全编码问题转化为生成树上的风险节点对比学习，通过模型自身生成安全“黄金路径”和漏洞路径，提供密集的on‑policy反馈；实现风险节点自动化标注、逐节点对比损失与迭代自对弈更新；同时兼顾跨语言与未知CWE的泛化能力。

**🔧 技术方法**

采用大语言模型（CodeLlama、Qwen2.5‑Coder等）并结合SFT、RL等传统对齐方法；利用自对弈生成安全与不安全路径；使用对比学习损失（类似DPO）与分数函数；自动化风险节点标注LLM；评估使用CodeQL静态分析和LLM安全评估；实验涵盖多语言、C/C++、Python等。

**📊 数据集**

主要数据集包括：DiverseVul（C/C++函数+CWE标签，1,353安全函数用于标注），SafeCoder安全数据集、Python SecurityEval、HumanEval；通过LLM自动标注得到风险节点；在C/C++上使用DiverseVul进行训练与测试，在Python上使用SecurityEval；跨语言评测采用从SafeCoder生成的多语言示例。

**📈 对比分析**

实验与基线（Base LLM、SFT、SafeCoder、Self‑Play）在Python SecurityEval（SPR@1）、C/C++漏洞计数以及跨语言和未知CWE的评测中对比。TSP在Python SPR@1达75.8%（高于SafeCoder 73.7%），在CodeLlama‑7B C/C++漏洞数从115降到94；跨语言测试显示TSP在Python、JavaScript、Go、Ruby均最低漏洞数；在未见CWE上TSP减少了约32%漏洞，且高危漏洞显著下降。整体而言，TSP在安全性能上最优，且对通用编程能力影响极小。

**⚠️ 局限性**

局限性包括：对局部显式控制流漏洞效果好，但对复杂内存和隐式数据流漏洞（如CWE-690、CWE-125）效果差；风险节点仅以token级为基础，难以捕获跨多行、多变量的安全依赖；自对弈负样本随着模型改进难度逐渐减小，可能限制深层漏洞发现；实验仅在3B–7B规模，未验证大规模模型的可扩展性与性能。

---

## 457. Mixed-Modality Dual Face-Hair Retrieval

**arXiv ID:** 2606.03470 | [PDF](https://arxiv.org/pdf/2606.03470v1)

**作者:** Quoc-Anh Bui-Huynh `[一作]` (Vietnam National University), Thanh Duc Ngo `[通讯]` (Vietnam National University)

**通讯引用:** 4022 | [OpenAlex ID](https://openalex.org/A5037141392)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出双模态人脸-发型检索（DFHR）任务，构建DFHR-Bench基准，并提出基于token注入的统一文本空间的多视图监督框架MFHC。

**💡 创新点**

创新点在于（1）首次定义身份与发型双重约束的混合模态检索；（2）构造180k+三元组的DFHR-Bench；（3）利用CLIP文本空间进行跨模态注入与多视图对齐，显著提升身份与发型的解耦与检索性能。

**🔧 技术方法**

核心技术包括：CLIP图像/文本编码器、ArcFace身份编码器、LoRA文本适配、token注入与多视图语义一致性正则、硬负样本挖掘与对比学习。

**📊 数据集**

使用了由CelebA‑HQ、FFHQ和深伪图构成的180k+三元组DFHR-Bench，包含图像‑图像、图像‑文本以及双模态对齐子集。

**📈 对比分析**

在Image‑Image、Image‑Text和双模态对齐子集上与CIR、生成式、融合等多种方法比较，MFHC在Recall@1/3/5、Precision@1/3/5和mAP上显著优于对手，R@1分别提升至约25%和15%，mAP提升至约22%和13%。

**⚠️ 局限性**

主要限制是：对预训练CLIP文本空间的高度依赖，导致在极端发型差异和跨语义模态一致性方面仍存在偏差；模型对训练数据规模与多样性敏感，缺乏对更广泛文化或极端发型的泛化能力。

---

## 458. Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression

**arXiv ID:** 2606.03465 | [PDF](https://arxiv.org/pdf/2606.03465v1)

**作者:** Artur Zagitov `[一作]` (BRAIn Lab), Aleksandr Beznosikov `[通讯]` (BRAIn Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型压缩中常用的张量分解方法进行系统实验验证，比较了张量分解与传统矩阵分解（SVD、LASER等）以及量化在真实部署条件下的压缩‑质量平衡，并通过激活流几何诊断揭示张量分解失效的根本机制。

**💡 创新点**

创新点包括：①首次大规模评估张量分解在真实硬件/部署场景下的表现；②通过激活几何、角度漂移和范数衰减等指标定位张量分解导致的功能退化；③从理论层面证明 Frobenius‑最优张量分解在算子范数上与矩阵分解存在本质不匹配，解释了高压缩率下性能急剧下降；④引入“super‑weight”恢复与稀疏修正实验，进一步阐明重要权重对模型质量的决定性作用。

**🔧 技术方法**

使用的技术包括：SVD、HOSVD、Tucker、Tensor‑Train (TT) 等张量分解；LoRA 轻量级修复；量化（4‑bit、8‑bit）对比；对激活流的角度与范数比例进行诊断；以及基于谱分析和 operator‑norm 与 Frobenius‑norm 区别的理论推导。

**📊 数据集**

实验数据集：GPT‑J 6B、LLaMA‑2 7B、Qwen3‑30B‑A3B、GPT‑OSS‑20B；评估使用 WikiText‑2、C4 以及多项零样本下游任务（ARC‑Challenge、HellaSwag、OpenBookQA、PIQA、WinoGrande）。

**📈 对比分析**

对比方法涵盖 pruning、LASER、TensorLLM、TD‑MoE、MoBE、LoRA、以及量化。结果显示：在相同压缩率下，张量分解往往产生更高 perplexity（尤其是 FFN 部分），矩阵分解或量化在 Pareto 前沿上更具优势；LoRA 在轻量修复后能缓解部分性能下降，但仍无法匹敌量化效果。

**⚠️ 局限性**

局限性：仅评估了两种 dense 架构和两种 MoE 架构，可能不适用于其他 attention、归一化或专家路由设计；理论分析提供的是上界而非精确误差；LoRA 修复仅使用 100 步轻量微调，可能低估了更强微调的恢复潜力；未涉及训练时张量分解或更复杂的压缩策略。

---

## 459. DMF: A Deterministic Memory Framework for Conversational AI Agents

**arXiv ID:** 2606.03463 | [PDF](https://arxiv.org/pdf/2606.03463v1)

**作者:** Matteo Stabile `[一作]` (Roma Tre University), Enrico Zimuel `[通讯]` (Roma Tre University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Deterministic Memory Framework（DMF），用CPU先行、无LLM调用的确定性管道管理会话记忆；

**💡 创新点**

创新点包括：用生存得分Ω结合内容信号、对话线索和结构化来源，基于交互计数的指数衰减并调节惯性，社交底线提升法，完全确定性裁剪与检索，令记忆管理实现零token消耗；

**🔧 技术方法**

使用经典NLP工具（spaCy、VADER）提取信号，BaaI/bge-small嵌入模型，逻辑回归与指数衰减公式，ChromaDB向量检索，纯Python CPU实现；

**📊 数据集**

在LoCoMo和LongMemEval两大长期会话基准上评估；

**📈 对比分析**

与Mem0比较：在LoCoMo多跳、时序、开放域等子任务中性能相当或更优，尤其时序推理提升约4倍；在LongMemEval‑10上准确率相近，DMF总token使用量比Mem0低约242×；

**⚠️ 局限性**

局限性：目前仅支持英语，得分与衰减参数固定，未实现在线自适应校准；线性衰减仅考虑交互计数，可能对时钟时间敏感；线性关系和硬阈值可能导致对极端对话场景的过度裁剪；

---

## 460. Topology-Aware Gaussian Graph Repair for Robust Graph Neural Networks

**arXiv ID:** 2606.03462 | [PDF](https://arxiv.org/pdf/2606.03462v1)

**作者:** Anubha Goel `[一作]` (Tampere University), Juho Kanniainen `[通讯]` (Tampere University)

**通讯引用:** 1958 | [OpenAlex ID](https://openalex.org/A5049372872)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为 TAGR 的图结构修复框架，通过稀疏的高斯特征邻居连接和拓扑感知残差加权来改进图神经网络在噪声或缺失边条件下的鲁棒性。

**💡 创新点**

创新点在于：①将高斯核自适应地应用于特征空间，构建稀疏补齐缺失连接的邻居图；②引入基于局部特征与结构一致性的残差权重，对原始边进行细粒度重新加权；③整个修复过程无需训练、保持稀疏、可直接与任意标准 GNN 兼容，避免了密集图学习与双层优化的复杂性。

**🔧 技术方法**

技术包括：稀疏高斯特征邻居图构建、局部带宽自适应、特征相似度与结构一致性统计（Jaccard、共邻、聚类系数、度不平衡）标准化、残差乘子裁剪与双曲正切映射、以及对 GCN、GAT、GraphSAGE 等标准消息传播模型的修复图输入。

**📊 数据集**

在四个基准引文网络（Cora、Citeseer、Cora-ML、Pubmed）以及 Flickr 与 Cornell 的额外诊断集上进行实验。

**📈 对比分析**

与标准 GNN、仅高斯修复、RS‑GCN、JNSGSL 等学习型图修复/结构学习方法比较。实验表明 TAGR 在加入噪声边（Add 50%/90%）和删边（Del 25%/50%）下均能提升性能，特别是在 GCN 上表现显著；在完整图上性能与原模型相当；在鲁棒性曲线评估中，TAGR‑GCN 取得最低平均排名，显示其在不同扰动级别下的一致鲁棒性。

**⚠️ 局限性**

局限性包括：①修复过程是预先确定、不可训练的，缺乏对任务特定特征的自适应性；②依赖于节点特征质量，对特征噪声或缺失可能效果受限；③对非常大规模或动态图场景的扩展性尚待验证；④在极端噪声插入（Add 90%）下，性能仍不及某些学习型修复方法。

---

## 461. PerchRL: Vision-Based Agile Perching on Inclined Platforms under Rapid and Irregular Motion

**arXiv ID:** 2606.03441 | [PDF](https://arxiv.org/pdf/2606.03441v1)

**作者:** Zihong Lu `[一作]` (SUSTech), Boyu Zhou `[通讯]` (SUSTech)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种名为PerchRL的强化学习框架，能够实现四旋翼在快速、非规则运动的倾斜平台上进行基于视觉的敏捷着陆，并实现了从仿真到真实世界的零差异迁移；

**💡 创新点**

其创新点包括：①在预训练阶段采用高度随机化的轨迹与时序增强技术，显著提升策略在多样平台运动下的泛化能力；②在视觉微调阶段引入可见性感知状态扩充与主动感知奖励的混合学习方案，提升了在视觉丢失情况下的恢复与稳定性；

**🔧 技术方法**

核心技术包括基于PPO的强化学习、时间卷积网络（TCN）捕捉时序特征、EKF状态估计、可见性门控观测模型、可见性感知输入扩充与主动感知奖励、CTBR动作归一化等；

**📊 数据集**

实验主要使用仿真生成的随机轨迹数据，并在真实平台（Crazyflie 2.1 与自制四旋翼）上进行验证，未使用公开数据集；

**📈 对比分析**

与基于模型的Fast‑Perching、RL基准InclineLander以及多种消融版本相比，PerchRL在多种轨迹和速度条件下实现了更高的成功率、训练收敛更快，并在真实平台上完成了零差异迁移的敏捷着陆；

**⚠️ 局限性**

当前方法受限于四旋翼在无活动陀螺仪/云台的情况下感知与控制耦合的影响，在极端平台运动或严格着陆要求下性能受限，后续计划通过加入主动云台实现更低的耦合与更优性能。

---

## 462. Large Language Models Are Overconfident in Their Own Responses

**arXiv ID:** 2606.03437 | [PDF](https://arxiv.org/pdf/2606.03437v1)

**作者:** Mario Sanz-Guerrero `[一作]` (Johannes Gutenberg University Mainz), Katharina von der Wense `[通讯]` (Johannes Gutenberg University Mainz)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5093081501)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文探究指令微调与聊天模板对大语言模型（LLM）校准的影响，并提出一种在推理时将模型答案视为用户输入来降低过度自信的策略。

**💡 创新点**

创新点在于发现并量化“所有权偏差”——模型在自己的回答上显著过度自信，并证明通过简单的提示重排即可显著提升校准。

**🔧 技术方法**

技术主要包括对多种公开LLM（如Llama 3.1、Qwen3、Gemma 3）进行不同提示（无聊天模板、聊天模板）下的实验，使用三种置信度估计方法（logit、百分比、语言等级）和置信度重排策略。

**📊 数据集**

数据集覆盖多任务：MMLU（多选）、GSM8K、TruthfulQA及开源模型GPT‑5.2的MMLU评测。

**📈 对比分析**

与基线（未微调模型）和传统的logit或语言表达置信度方法相比，提出的“答案转为用户输入”方案在ECE、Brier得分上平均提升约15–25%，接近或超过基线模型的校准水平。

**⚠️ 局限性**

局限性包括仅针对客观问答任务，未深入研究主观生成任务；方法仅为推理时的提示，无法根除训练阶段导致的过度自信；对完全闭源模型的普适性尚待进一步验证。

---

## 463. AvatarMix: Identity-Preserving Cross-Avatar Composition for Outfit Personalization

**arXiv ID:** 2606.03506 | [PDF](https://arxiv.org/pdf/2606.03506v1)

**作者:** Zhaorong Wang `[一作]` (University of Tsukuba), Yuki Endo `[通讯]` (University of Tsukuba)

**通讯引用:** 820 | [OpenAlex ID](https://openalex.org/A5079605823)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

构建了一种将用户头像与模特衣服通过3D高精度高斯头像进行无缝拼接，从而实现服装个性化的AvatarMix框架。

**💡 创新点**

创新点在于：①直接在mesh‑based Gaussian头像上进行三维拼接，避免传统2D‑>3D升维导致的质量退化；②引入GSReshape进行服装与用户体型的精准重塑；③设计了两级扩散修复网络SeamFix与FullbodyFix，以消除拼接与重塑过程中的缝合与纹理失真。

**🔧 技术方法**

核心技术包括3D Gaussian Splatting、基于网格的高斯头像、服装重塑与网格保形（GSReshape）、扩散模型修复（SeamFix/FullbodyFix）以及LoRA适配器训练。

**📊 数据集**

使用公开的THUman2.0数据集进行训练与评估，数据集包含526个不同体型、服装与姿态的已重建服装人类主体。

**📈 对比分析**

与VTON360和TIP‑Editor两种主流方法对比，利用Editing Target DINO、Head+Neck DINO和Warping‑RMSE三种指标，AvatarMix在衣服保真、面部身份以及多视角一致性上均实现了更高的分数和更低的误差，且在用户研究中获得了显著优先。

**⚠️ 局限性**

局限性包括：手部适配仍依赖去除手部几何的近似处理，FullbodyFix仍需手工触发且缺乏自动判定机制，极端体型差异下可能出现纹理失真，整体框架对SMPL‑X姿态与分割质量有一定依赖。

---

## 464. KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks

**arXiv ID:** 2606.03458 | [PDF](https://arxiv.org/pdf/2606.03458v1)

**作者:** Lorenz K. Muller `[一作]` (Huawei), Lukas Cavigelli `[通讯]` (Huawei)

**通讯引用:** 2966 | [OpenAlex ID](https://openalex.org/A5025399641)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了 KVarN，一种无校准的 KV‑Cache 量化方法，结合 Hadamard 旋转与双尺度方差归一化，显著降低长时间自回归解码过程中的量化误差累积。

**💡 创新点**

创新点：①将无失真旋转（Hadamard）与在通道和 token 维度同时进行的方差归一化（dual‑scaling）相结合，专门针对 KV‑Cache 中的 token 规模误差；②提出了 pseudo‑decode 评估框架，用于量化量化误差在解码过程中的累积；③实现了完全校准无需求的 2‑bit 量化方案，在推理时实现近乎无损的 KV‑Cache 压缩。

**🔧 技术方法**

技术细节包括 Hadamard 旋转、双尺度方差归一化（迭代实现）、RTN（round‑to‑nearest）量化、两尺度解量化、pseudo‑decode 误差评估，以及在 vLLM 上的高效实现。

**📊 数据集**

数据集与评测基准：Qwen3-4B、Llama‑3.1‑8B、Phi‑4‑14B；生成与推理任务：MATH500、AIME24、HumanEval、IF‑Eval、线性检索（Line‑Retrieval）。

**📈 对比分析**

与 KIVI、QuaRot、KVQuant‑1%、PolarQuant、TurboQuant1、Kitty 等方法对比，KVarN 在 2‑bit 量化下取得最佳或近乎最佳的准确率，例如 AIME24 79.2%、MATH500 84.8%，并在 HumanEval 与 IF‑Eval 上也保持领先；量化延迟仅为 0.18% 的额外开销。

**⚠️ 局限性**

局限性：对极长序列或更大模型的误差累积影响仍有限；仍存在约 1% 的解量化慢速；目前仅针对 KV‑Cache 量化，未覆盖混合精度、其他压缩技术或训练阶段的验证；对极大上下文长度的性能尚未深入评估。

---

## 465. Overlaying Governance: A Compositional Authorization Framework for Delegation and Scope in Agentic AI

**arXiv ID:** 2606.03518 | [PDF](https://arxiv.org/pdf/2606.03518v1)

**作者:** Amjad Ibrahim `[一作]` (Huawei Heisenberg Research Center), Yong Li `[通讯]` (Huawei Heisenberg Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

未提供论文具体内容，无法确定研究目的与工作范围。

**💡 创新点**

无法识别创新点。

**🔧 技术方法**

未知。

**📊 数据集**

未知。

**📈 对比分析**

未知。

**⚠️ 局限性**

缺乏足够信息来评估研究局限。

---

## 466. A Voxel-Based Quantum Computing Method (VBQC) for Solid Mechanics Problem

**arXiv ID:** 2606.03515 | [PDF](https://arxiv.org/pdf/2606.03515v1)

**作者:** Feng Wu `[一作]` (Dalian University of Technology), Xu Guo `[通讯]` (Dalian University of Technology)

**通讯引用:** 473685 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文研究了基于体素网格的固体力学问题的量子哈密顿量模拟，提出了KCQ分解和VBQC方案。

**💡 创新点**

创新点在于将体素网格产生的分块三对角分形矩阵拆解为仅三类小维度矩阵（k_n, c_n, q_n），实现多维问题的量子有效分解。

**🔧 技术方法**

采用了量子傅里叶变换、循环门、相位门、量子多路复用器以及Pauli门等量子门构建VBQC，并利用KCQ分解实现矩阵指数。

**📊 数据集**

在三种典型固体力学案例（1D梁、2D孔板、3D不均匀板）中构造有限元刚度矩阵作为数据集。

**📈 对比分析**

通过量子相位估计与经典有限元对比，量子算法对振动频率和模态形状的相对误差低于5%，并证明了 e^{iAt} 计算误差随时间步长的 O(Δt^3) 阶。

**⚠️ 局限性**

主要局限是量子多路复用器及多比特控制门尚未实现，需要依赖未来硬件；此外方法仍依赖体素网格的正方形结构，复杂几何仍需虚拟节点处理。

---

## 467. Optimizing Proof-Search via Linearization for Gödel-Löb Logic with Tree-Hypersequents

**arXiv ID:** 2606.03484 | [PDF](https://arxiv.org/pdf/2606.03484v1)

**作者:** Tim S. Lyon `[一作]` (TU Dresden), Omar Taher `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

本文设计了一种针对Gödel–Löb逻辑的树层次序列（tree‑hypersequent）系统的复杂度最优证明搜索算法，完成了Poggiolesi与Maggesi‑Perini Brogi提出的决策与搜索问题；

**💡 创新点**

其创新点在于引入线性化方法，仅按深度优先方式构造单条推导分支；通过分支规则（∧、∨）与对角线公式限制树深度，避免指数级膨胀；同时能够从失败搜索中组合分支得到有限反模型；并将结果映射为线性嵌套序列（LNS）推理器；

**🔧 技术方法**

核心技术包括树层次序列的语义化写法、线性化（depth‑first linearization）、分支规则（disjunctive/in­jective 规则）、对角线公式（diagonal formula）保证终止、计算树构造与反模型合成；

**📊 数据集**

论文为理论研究，无实验或数据集，所有结论均来自形式化证明；

**📈 对比分析**

与先前的Gentzen、标记序列及传统树层次序列算法相比，本文算法空间复杂度为O(|ϕ|^4)（即多项式空间），实现了已知的PSPACE‑最优性；

**⚠️ 局限性**

局限性在于仅针对Gödel–Löb逻辑，算法仅输出布尔判定，反模型提取仅理论工具；若要直接给出证明或模型，需额外实现；另外，线性化方法与树层次结构高度相关，推广到其它模态逻辑仍需进一步研究。

---

## 468. Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation

**arXiv ID:** 2606.03483 | [PDF](https://arxiv.org/pdf/2606.03483v1)

**作者:** Ekaterina Alimaskina `[一作]` (MIRAI), Aleksandr Beznosikov `[通讯]` (MIRAI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究 Hyper-Connections（多流残差）模型在训练过程中是否真正实现多流特性，并分析其导致的流崩塌（dominant‑stream）现象。

**💡 创新点**

发现残差混合大多保持接近单位矩阵，导致大部分信号、表示范数和可解释特征集中于单一流，并提出通过在初始化阶段加入可学习的近单位对角缩放（Learned Stream Scaling, LSS）来打破对称性，从而缓解崩塌并提升性能。

**🔧 技术方法**

使用 HC/mHC 变体（mHC‑lite、mHC、KromHC 等）以及残差混合矩阵、读写接口权重、表示 L₂ 范数、残差曲率和稀疏交叉编码器等诊断技术进行细粒度分析。

**📊 数据集**

在 OpenWebText、WikiText‑103、C4 三大英文文本语料上进行训练与评估。

**📈 对比分析**

通过与默认 HC、残差混合设为身份、以及 LSS+Identity 等 ablation 对比，发现 LSS 在中型模型上将 perplexity 降低约 0.3‑0.5，整体性能优于传统 HC 方案。

**⚠️ 局限性**

实验规模有限，未探索更深层次多流协作机制，且仅在英文文本任务验证，缺乏跨语言或其他任务的通用性评估。

---

## 469. FORGE: Multi-Agent Graduated Exploitation and Detection Engineering

**arXiv ID:** 2606.03453 | [PDF](https://arxiv.org/pdf/2606.03453v1)

**作者:** Farooq Shaikh `[一作]` `[通讯]` (Dynatrace), Farooq Shaikh (Dynatrace)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了FORGE，一个多代理系统，能够对CVE进行分级利用性评估、生成检测规则，并为优先级模型提供真实的利用深度数据。

**💡 创新点**

创新点包括四级利用深度分级、LLM判别器对每一步评估、针对性最小化应用生成以及跨CVE的多层知识架构实现知识迁移。

**🔧 技术方法**

使用LLM驱动的多代理架构（Intel、Generator、Planner、Exploit、Detector），LLM判别器（GPT‑5 Mini）与Claude Sonnet 4.5，OpenTelemetry 日志，Sigma/Snort 规则生成。

**📊 数据集**

使用CVE‑GENIE 841条记录中选取的603条CVE数据集，涵盖8种编程语言和187种CWE。

**📈 对比分析**

与CVE‑GENIE对比，FORGE在同一数据集上实现67.8% L1+利用率，成本仅$1.50/条；生成的检测规则在L2+利用时拥有更高的跨度归因率，FP率低于1%。

**⚠️ 局限性**

局限在于生成的最小化应用缺乏生产环境复杂性，LLM判别器仍有误判，知识库效果需在更大范围验证，并未覆盖非开源或嵌入式系统。

---

## 470. Privacy-Preserving High-Resolution Image Gradient Computation Based on Fully Homomorphic Encryption

**arXiv ID:** 2606.03513 | [PDF](https://arxiv.org/pdf/2606.03513v1)

**作者:** Yufei Zhou `[一作]` (Sun Yat-Sen University), Yufei Zhou `[通讯]` (Sun Yat-Sen University)

**通讯引用:** 1242 | [OpenAlex ID](https://openalex.org/A5050754750)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了多密文加密框架，通过重复打包将高分辨率图像分块，实现在 CKKS 同态加密下对 Sobel 边缘检测（梯度幅值和方向）的全同态计算。

**💡 创新点**

创新点包括：① 采用重复打包与多密文分区解耦卷积边界，实现高并行度卷积并减少加密开销；② 通过符号函数与 Chebyshev 多项式逼近构造逆函数的高精度近似，克服传统多项式逼近在零点附近的不稳定；③ 提出 preBTS 策略，使服务器在接收低多项式深度密文后立即进行多轮 bootstrapping，显著降低用户端加密与密文上传成本。

**🔧 技术方法**

技术方法：CKKS 同态加密、SIMD 方式卷积、行/列打包与重复打包、Chebyshev 多项式逼近、符号函数逼近、预/后 Bootstrapping（BTS）策略。

**📊 数据集**

实验数据集：随机生成灰度图像以及 DIV2K_train_HR 数据集（五张灰度图）用于验证逼近误差和视觉效果。

**📈 对比分析**

与 Single（单密文）和 Compact（紧凑打包）两种基线进行对比。实验显示，在 648×2040 大图像下，preBTS 方案加密时间比 Compact 低约 15%，密文大小减 4%；服务器卷积时间约比 Compact 低 100 秒，整体总时间在 postBTS 方案下降约 20%，preBTS 下降约 15%；PSNR 对幅值 115.77 dB、SSIM 1.00，角度 PSNR 约 20.7 dB，视觉误差几乎不可见。

**⚠️ 局限性**

局限性：服务器端计算仍耗时数千秒，需 GPU 加速；BTS 迭代次数多导致误差累积与密钥体积巨大（数十 GB）；在多项式深度与精度之间需进一步平衡。

---

## 471. Structure-Guided Mixed Masked Pretraining and Spatial Continuity Regularization for Printed Circuit Board Defect Detection

**arXiv ID:** 2606.03508 | [PDF](https://arxiv.org/pdf/2606.03508v1)

**作者:** Peitong Wang `[一作]` (Anhui University), Yuanting Yan `[通讯]` (Anhui University)

**通讯引用:** 792 | [OpenAlex ID](https://openalex.org/A5078311721)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了两阶段 PCB 缺陷检测框架：先在未标注 PCB 图像上使用结构引导混合遮挡预训练 (SG-MIM)，随后在 YOLOv8 上进行细调并加入空间连续性正则化以抑制细长缺陷的碎片化检测

**💡 创新点**

创新点包括① 在遮挡预训练中结合 PCB 结构先验（边缘、高频、线条、方向一致性）生成结构引导混合遮挡，② 采用稀疏卷积编码抑制遮挡区的干扰，③ 在细调阶段加入空间连续性正则化约束同一缺陷实例的正样本预测位置，提升定位连贯性

**🔧 技术方法**

使用的技术包括结构引导混合遮挡预训练（SG-MIM）、稀疏卷积掩码编码、YOLOv8 目标检测框架、空间连续性正则化损失、Mask Image Modeling (MAE/SimMIM 对比）

**📊 数据集**

在 DsPCBSD+ PCB 缺陷数据集上进行实验，该数据集包含 9 类缺陷，约 8,208 张训练图像和 2,051 张验证图像

**📈 对比分析**

与多种基线（YOLOv5s、YOLOv8s、YOLOv6s、RT-DETR、Co-DETR、YOLOv10、YOLOv6-L6）对比，本文方法在 mAP_0.5 提升至 85.5%（比 YOLOv8s 提升 1.4%），mAP_0.5:0.95 提升至 52.3%（比 YOLOv8s 提升 1.6%），在细长缺陷定位方面表现更佳

**⚠️ 局限性**

局限性包括：预训练仅使用当前数据集的未标注图像，缺乏更大规模多样化的 PCB 数据；空间连续性正则化仅在训练阶段使用，未进一步探讨对极端形状缺陷的适用性；模型仍需进一步轻量化以适应工业实时部署

---

## 472. A formal definition and meta-model for a machine theory of mind

**arXiv ID:** 2606.03471 | [PDF](https://arxiv.org/pdf/2606.03471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 473. ThoughtFold: Folding Reasoning Chains via Introspective Preference Learning

**arXiv ID:** 2606.03503 | [PDF](https://arxiv.org/pdf/2606.03503v1)

**作者:** Ziyan Liu `[一作]` (Shanghai Artificial Intelligence Laboratory), Kai Chen `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 32839 | [OpenAlex ID](https://openalex.org/A5100437924)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ThoughtFold 框架，通过自省式冗余识别和细粒度偏好学习来减少 LRM 的过度推理，提升推理效率。

**💡 创新点**

将 RLVR 与细粒度偏好学习结合，引入动态掩码策略对冗余步骤进行惩罚，实现推理链折叠；自省式冗余检索生成优先级样本。

**🔧 技术方法**

RLVR (GRPO)、细粒度偏好优化（Mask‑DPO/MDPO）、自省式冗余剪枝、动态掩码、注意力重要性评估等技术。

**📊 数据集**

训练使用 DeepMath‑103K，评估基准包括 GSM8K、MATH‑500、AIME 2024/2025、GPQA Diamond。

**📈 对比分析**

与 Vanilla、GRPO、Short‑RL、S‑GRPO 等基线对比；在 DeepSeek‑R1‑Distill‑Qwen‑7B/14B 与 Qwen3‑8B/14B 上，ThoughtFold 约 39‑56% 缩短 token，准确率提升 1‑2% 以上，优于 S‑GRPO。

**⚠️ 局限性**

仍需大量对齐样本和计算开销，动态掩码与自省过程在极长或复杂链路上可能误判冗余，导致推理精度下降。

---

## 474. Demystifying Pipeline Parallelism: First Theory for PipeDream

**arXiv ID:** 2606.03498 | [PDF](https://arxiv.org/pdf/2606.03498v1)

**作者:** Ivan Ilin `[一作]` (KAUST), Peter Richtárik `[通讯]` (KAUST)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了 PipeDream（PD）流水线并行训练的优化成本，引入随机化抽象 Randomized PipeDream（RPD），并在非凸环境下给出收敛保证，同时与 LocalSGD 进行对比实验。

**💡 创新点**

创新点在于：1) 首次对 PD 风格的 stale block-SGD 进行非凸收敛分析；2) 通过 RPD 抽象揭示 PD 的延迟量化为 δ=Θ(S²)，导致 stale-read 项呈现 Θ(S⁴/K) 的规模不利；3) 将理论结果映射回原始 PD 并与 LocalSGD 的同步/漂移代价进行系统性对比。

**🔧 技术方法**

采用随机化 stale block-SGD、L‑smooth 与 bounded梯度假设下的收敛证明、延迟解析（δ= S² - S/2 + O(1)）以及离散事件模拟器重放 1F1B 时序，配合梯度方差与局部漂移分析。

**📊 数据集**

实验数据集包括：1) 合成二次目标（tridiagonal quadratic）与合成逻辑回归（M=60，batch=10）；2) 11.1M 参数 NanoChat 风格 Transformer 在 Tiny Shakespeare 上进行字符级语言建模。

**📈 对比分析**

对比方法：PD、RPD（设定 δ=δ_PD）与 LocalSGD（同步周期 H、复制数 R=S）。结果显示：在二次目标和 NanoChat 任务上，PD 与 RPD 的收敛速度优于 LocalSGD；而在逻辑回归任务且阶段数增大时，LocalSGD 在固定模拟时间预算下表现更好，说明同步漂移比 stale-read 更低的负面影响。

**⚠️ 局限性**

局限性：1) 分析基于随机化 RPD，未对原始确定性 PD 进行直接证明；2) 没有考虑实际系统层面的内存占用与通信开销，仅关注优化梯度延迟；3) 仅在有限的合成与单一 Transformer 任务上验证，缺乏更广泛的数据集与大规模实验。

---

## 475. Analyzing Visual Attention Patterns During Band Rehearsal with Mobile Eye Tracking

**arXiv ID:** 2606.03485 | [PDF](https://arxiv.org/pdf/2606.03485v1)

**作者:** Arvind Srinivasan `[一作]` (Aarhus University), Michael Sedlmair `[通讯]` (University of Stuttgart)

**通讯引用:** 5649 | [OpenAlex ID](https://openalex.org/A5037110552)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本研究在业余乐队的自然排练场景中，使用移动眼动仪记录音乐家视觉注意力，并通过YOLOv8目标检测对场景中的成员与物体进行自动标注，随后构建凝视矩阵、转移矩阵、scarf图等多种可视化与统计分析方法，以揭示四名成员在三首歌曲的两轮排练中的凝视分配模式。

**💡 创新点**

创新点包括：①将移动眼动仪与深度学习目标检测相结合，实现多成员现场视线追踪；②发现并描述了以领奏者为中心的hub‑and‑spoke注意力拓扑；③提出注意力窄化（高单目标凝视与低转移次数）作为学习困难的客观指标；④将教学反馈类型与注意力分布变化进行对比，阐释结构性教学对独立参考的促进作用。

**🔧 技术方法**

使用技术包括：Pupil Labs Neon 200Hz 眼动仪、YOLOv8n 目标检测器、Python/Matlab 数据处理与可视化、固定点校准与时间同步、凝视与扫视检测算法、相关分析与散点图。

**📊 数据集**

数据集来源于四位乐手在三首歌曲（每首两轮）排练过程中的同步眼动与场景视频；YOLOv8模型在数百帧手工标注的场景图像上训练得到。

**📈 对比分析**

通过比较同一歌曲不同练习回合的凝视/转移统计，发现从首次练习（A1）到第二次练习（A2）转移次数平均下降65%（单个乐手最高可达82%），并且在S2（教学环节）出现最大下降；对比教学反馈类型，结构性教学导致学习者从领奏者切换至监视器，转移次数显著下降。相关系数 r=-0.44（p=0.03）表明凝视集中与转移减少相关。

**⚠️ 局限性**

局限性包括：样本仅4人、6次实验，缺乏随机或对照设计；歌曲顺序固定导致难以区分歌曲与练习顺序效应；YOLO检测精度仅中等，Other 类别占比高导致人际凝视分辨率受限；未提供多乐队或不同音乐流派的验证，结果的普适性和可重复性待进一步研究。

---

## 476. Predicting Lakehouse Performance in Clouds: An Empirical Exploration of Query Runtime Variance

**arXiv ID:** 2606.03464 | [PDF](https://arxiv.org/pdf/2606.03464v1)

**作者:** James Nurdin `[一作]` (University of Glasgow), Lauritz Thamsen `[通讯]` (University of Glasgow)

**通讯引用:** 798 | [OpenAlex ID](https://openalex.org/A5084056435)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对分布式湖仓系统在公共云与私有云中的运行时方差进行了系统评估，并探讨其对查询性能预测（QPP）和低碳调度的影响。

**💡 创新点**

创新点在于首次量化湖仓查询的运行时方差、通过因子分析定位主要方差源、证明降低方差可将QPP误差降低多达80%，并验证低碳调度在低方差场景下能显著减少碳排放。

**🔧 技术方法**

主要技术包括：基于Kubernetes的Trino+Iceberg湖仓部署、控制因素实验（热缓存、本地磁盘、本地节点固定、共租负载、外部元数据服务）、GNN与随机森林QPP模型、贪婪低碳调度与不确定性感知调度策略。

**📊 数据集**

使用的数据集包括TPC‑DS、SSB、JOB（三种工作负载）在规模因子SF 1000下生成的5 000条查询。

**📈 对比分析**

对比方法：在高方差与低方差两种湖仓部署下训练同一QPP模型，评估MAE、Median QError、P99 QError；随后使用预测值与真实值在Greedy低碳调度与FIFO调度下计算碳排放并与Oracle排程比较。结果显示：QPP误差平均可降低49–80%，P99误差可降低57–88%；低碳调度在低方差场景下相较Oracle仅产生约1–5%额外排放。

**⚠️ 局限性**

局限性包括：仅评估Trino+Iceberg组合，未覆盖其他引擎或文件格式；因子实验范围有限，未尝试多因子组合；只测试了两种QPP模型；调度实验假设单槽、非抢占、固定功耗，未考虑成本与SLA约束。

---

## 477. What Makes Interaction Trajectories Effective for Training Terminal Agents?

**arXiv ID:** 2606.03461 | [PDF](https://arxiv.org/pdf/2606.03461v1)

**作者:** Sidi Yang `[一作]` (University of Hong Kong), Haoli Bai `[通讯]` (Huawei Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过构建 Terminal-Lego 数据管道，将真实 StackOverflow 问题转换为可执行的 Docker 终端任务，并在统一 harness 下对不同教师模型的轨迹进行匹配任务的教师蒸馏实验，探究教师轨迹的教学有效性。

**💡 创新点**

提出“环境根植监督”（Environment‑Grounded Supervision, EGS）概念，并提出 Targeted Observation Ratio (TOR) 作为衡量轨迹观察与行动对齐度的指标，揭示教师轨迹中观察‑验证循环对学生学习的重要性，形成了“教学悖论”理论。

**🔧 技术方法**

利用终端交互 harness（Terminus‑2）、大型语言模型生成与审核流水线、Docker 任务验证、以及监督式微调（SFT）技术；在此基础上对 Qwen3‑8B、Qwen3‑32B 等模型进行后训练。

**📊 数据集**

Terminal‑Lego 数据集（约 15.3k 轨迹，12.8k 成功、2.5k 失败），来源于 90+ 技术域的 StackOverflow 实际问题，并通过 LLM 生成任务、测试及 Docker 验证构建。

**📈 对比分析**

对比了四位教师模型（DeepSeek‑V3.2、Claude Opus 4.6、Qwen3.5‑Plus、GLM‑5）生成轨迹下的 Qwen3‑8B/32B 学生性能；在 Terminal‑Bench 2.0 上，使用 15k 轨迹的 Qwen3‑32B 由 3.4% 提升至 24.3%，相当于 SOTA 的 30× 数据效率提升；与同规模 Nemotron‑Terminal 子集相比，Terminal‑Lego 明显优于 3 种对比子集。

**⚠️ 局限性**

实验仅涵盖 8B/32B 学生模型，未验证更大/不同架构模型的泛化；TOR 与观察‑验证的评估仍属经验性，未对观察是否必需、充分或逻辑严谨性进行形式化检验；此外，教师模型仅为四个代表性模型，未覆盖更广泛的能力范围。

---

## 478. CP-Agent: Context-Aware Multimodal Reasoning for Cellular Morphological Profiling under Chemical Perturbations

**arXiv ID:** 2606.03435 | [PDF](https://arxiv.org/pdf/2606.03435v1)

**作者:** Yuxin Zhang `[一作]` (University of Hong Kong), Kevin Tsia `[通讯]` (University of Hong Kong)

**通讯引用:** 4932 | [OpenAlex ID](https://openalex.org/A5005454460)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了名为CP-Agent的上下文感知多模态推理框架，用于可解释的Cell Painting药物筛选分析。

**💡 创新点**

创新点在于将实验上下文嵌入到对比学习模型CP-CLIP中，并结合大型语言模型生成结构化报告，实现实验设计与机制推断闭环。

**🔧 技术方法**

使用CP-CLIP（改进CLIP）与大型语言模型（如GPT‑5）结合，配合单细胞特征提取与工具化推理。

**📊 数据集**

使用公开的BBBC021、CPJUMP1、RxRx3三个Cell Painting数据集，共约190万张图像及对应实验元数据。

**📈 对比分析**

与多种通用MLLM和CLIP变体对比，CP-CLIP在药物识别、细胞系与通道预测中实现最高F1（0.896），在未见药物匹配上提升约14%相似度。

**⚠️ 局限性**

局限在于对复杂或微妙表型的解释仍受限，且需要大量标注数据与对比学习训练，未能完全解决高通量实验的实时推断速度。

---

## 479. Signals and Spoils: Speculative Oracle Extractable Value in the Era of Cross-Chain Interoperability

**arXiv ID:** 2606.03434 | [PDF](https://arxiv.org/pdf/2606.03434v1)

**作者:** Hasret Ozan Sevim `[一作]` (University of Camerino), Christof Ferreira Torres `[通讯]` (Instituto Superior Técnico, University of Lisbon)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文研究 Layer‑2 区块链中的 Speculative OEV（Oracle Extractable Value），检测并量化在 Arbitrum、Base、Optimism 上的投机性清算行为，并探讨跨链 Chainlink 价格发布延迟如何为跨链 OEV 提供可利用窗口。

**💡 创新点**

创新点包括：①首次在真实交易中识别并量化 Speculative OEV；②提出基于交易位置、相邻区块和同一 calldata 的检测方法；③系统分析 Chainlink DON 配置差异导致的跨链价格更新时间差，并证明其可预测性；④通过大规模数据展示 Speculative OEV 对链上交易垃圾、费用及空间利用的影响。

**🔧 技术方法**

使用的技术包括：链上事件抓取（web3）、自定义 Python 脚本对 Aave、Chainlink 价格合约进行查询、基于区块号和交易索引的同块清算检测、统计分析（Python/pandas）以及可视化。

**📊 数据集**

数据集涵盖 2025‑10‑10 的 2,986 起 Aave 清算事件、12,009 条 Chainlink 价格更新、100,000+ DON 价格观察，以及 Binance USDT 交易对的秒级行情，用于同步比较链上与链下价格。

**📈 对比分析**

比较方法：将 Speculative 与非 Speculative 清算按成功率、同块比例、交易距离、重置率等指标对比；结果显示 Speculative 清算占 57% 的清算者、39% 的成功清算，却因重置率高、同块比例低而整体收益率略低于非 Speculative；但所有清算者均保持盈利。

**⚠️ 局限性**

局限性：仅分析单日数据（10/10/2025），跨链预测模型未在实时环境验证；仅关注 Aave V3，其他 DeFi 协议可能表现不同；检测方法依赖同一 calldata 匹配，可能漏检非标准清算方式；并未评估对链上生态长期影响的动态变化。

---

## 480. A Hybrid Approach For Malware Classification Using Secondary Features Fusion

**arXiv ID:** 2606.03432 | [PDF](https://arxiv.org/pdf/2606.03432v1)

**作者:** Raja Khurram Shahzad `[一作]` (Mid Sweden University), Haroon Elahi `[通讯]` (Chalmers University of Technology)

**通讯引用:** 431 | [OpenAlex ID](https://openalex.org/A5019308050)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

该研究通过静态分析提取API调用、DLL导入和Opcode n-gram等特征，对Microsoft Malware Challenge数据集进行特征融合和算法融合，自动检测并将恶意软件归入对应家族。

**💡 创新点**

创新点在于：①引入字典+频率双阶段初级特征筛选与自定义后向特征选择，②将多种特征集（API、DLL、Opcode四种n-gram）融合为一统一特征矩阵，③采用基于SLSQP求权重的多算法投票集成，④在原始恶意数据上加入正常样本实现二分类与多分类并行。

**🔧 技术方法**

技术细节包括：静态反汇编特征提取、四种n-gram（bi/quad/var-gram）生成、字典/频率过滤、过滤/包裹/嵌入式三阶段特征选择（RF、RGF、XGBoost、Lasso、Logistic回归等）、SLSQP求权重的投票集成和10折交叉验证评估。

**📊 数据集**

使用的数据集为Microsoft Malware Classification Challenge的.disassembled .asm文件，扩展后加入1,609个benign样本，总计9个恶意家族共计10,868个样本。

**📈 对比分析**

实验通过10折交叉验证、准确率、AUC、log loss等指标与公开排行榜及竞赛赢家模型对比，最终集成模型取得AUC 0.989、准确率 99.72%、log loss 0.01，优于基准模型和竞赛获胜方案。

**⚠️ 局限性**

局限性包括：计算资源有限导致特征提取和模型训练耗时、未处理加密样本、未将十六进制特征与ASM特征融合、以及对新恶意变种的实时适应能力待进一步提升。

---

## 481. Beyond the Literal: Decomposing Pragmatic Intent in Multimodal Meme Understanding

**arXiv ID:** 2606.03604 | [PDF](https://arxiv.org/pdf/2606.03604v1)

**作者:** Zhengyi Zhao `[一作]` (Chinese University of Hong Kong), Yulan He `[通讯]` (King's College London)

**通讯引用:** 13829 | [OpenAlex ID](https://openalex.org/A5015709853)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Intent Projection框架，在单一大型视觉语言模型中分离图像与文本的字面与语用信号，以提升多模态模因和讽刺的语用理解。

**💡 创新点**

在表示层使用正交投影去除单模态主方向，在输出层引入结构化推理链与Surface-Real情感标签，并在目标层采用对比式字面-语用奖励，实现对语用意图的显式捕捉。

**🔧 技术方法**

结合LVLM、正交投影、Surface-Real Affect Head、链式推理、梯度强化学习（GRPO）以及自蒸馏技术。

**📊 数据集**

使用六个多模态基准（MemeReaCon、MemeCap、MET-Meme、MuSE、GOAT-Bench、MMSD2.0）及其对应的Reddit帖子与社区评论。

**📈 对比分析**

与开源基线（LLaVA-OneVision-7B、Qwen3-VL-8B、InternVL3-8B）和专有模型对比，在生成与分类任务上均取得显著提升，尤其在高差异度模因上鲁棒性提升5–10%，并缩小与专有模型的性能差距。

**⚠️ 局限性**

局限性包括数据偏向英语/西方网络文化、推理链引入的轻微推理延迟，以及对自蒸馏目标噪声的敏感度。

---

## 482. CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery

**arXiv ID:** 2606.03602 | [PDF](https://arxiv.org/pdf/2606.03602v1)

**作者:** Bo Peng `[一作]` (Shanghai AI Laboratory), Chaochao Lu `[通讯]` (Shanghai AI Laboratory)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个基于算法集成与LLM可信度校准的因果发现框架，通过共识过滤、加权投票和循环修复，既利用多算法的统计信号，又在必要时调用LLM的先验知识。

**💡 创新点**

核心创新是将LLM视为可信度可调的证据来源，采用共识+留一法校准算法可靠性，并设计了权重调节的仲裁机制，以最小化LLM误差曝光并提升最终DAG质量。

**🔧 技术方法**

技术实现包括：三种统计因果算法（PC、GES、CAMML）的并行运行与投票；对LLM和算法集成进行存在与方向任务的可靠性估计；基于可信度权重的加权投票；以及基于边权重的循环修复过程。

**📊 数据集**

使用bnlearn仓库中的六个离散数据集：Cancer、Insurance、Water、Alarm、Barley、Win95pts，样本量均为5000。

**📈 对比分析**

与传统数据驱动方法（PC、FCI、GES、BOSS、GRaSP、CAMML、NOTEARS-MLP、DAGMA）以及现有LLM增强方法（CORR-LLM、LLM-BFS、SCP、ET-MCMC、LLM-MEC）进行比较，采用SHD、F1和SID指标；在所有数据集上均取得最低SHD、最高F1，尤其在大图Win95pts上表现显著优于其他方法，且LLM调用次数大幅降低，鲁棒性更好。

**⚠️ 局限性**

局限性包括：只选用三种统计算法作为集成成员，缺乏更广泛的算法多样性；实验仅覆盖离散变量数据，未验证连续或混合数据的适用性；LLM本身未被纳入集成成员，未来可进一步提升多样性与鲁棒性。

---

## 483. Diffusing in the Right Space: A Systematic Study of Latent Diffusability

**arXiv ID:** 2606.03578 | [PDF](https://arxiv.org/pdf/2606.03578v1)

**作者:** Tianxiong Zhong `[一作]` (Kuaishou Technology), Pengfei Wan `[通讯]` (Kuaishou Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

系统评估不同视觉tokenizer在潜在空间特性对扩散模型生成质量的影响，并提出了新的可测指标。

**💡 创新点**

创新点在于引入Velocity Irreducible Variance（VIV）这一量化流动不确定性的指标，并证明其与生成质量具有高度相关性；同时综合比较语义可分性、空间结构和光谱特性等多维度特性。

**🔧 技术方法**

采用基于流匹配的Velocity Ambiguity理论、离散余弦变换得到的Spectral Energy Concentration、LNC等多种指标；使用多种CNN和Transformer tokenizer结构以及不同容量的扩散后端（SiT-B/XL、LightningDiT-B/XL）。

**📊 数据集**

主要使用ImageNet训练集进行tokenizer训练与验证，随后在相同验证集上评估生成质量。

**📈 对比分析**

通过对比gFID、IS、FDr^6等生成质量指标，发现VIV、语义可分性和空间结构在不同后端、不同tokenizer族上均能稳定预测生成效果，整体相关系数最高达0.87；在多后端、多tokenizer族下验证了指标的跨模型泛化。

**⚠️ 局限性**

局限性包括：仅在相同架构、相同重建质量的tokenizer之间进行比较，未探究跨族间的泛化；VIV等指标仍依赖于高维协方差估计，可能受样本量限制；并未覆盖序列tokenizer等更广泛的表示形式。

---

## 484. AutoTail-BSFGM: Class-Balance-Aware Fine-Tuning for Chinese Scholarly Text Classification

**arXiv ID:** 2606.03576 | [PDF](https://arxiv.org/pdf/2606.03576v1)

**作者:** Anling Xiang `[一作]` (Minzu University of China), Yang Shen `[通讯]` (Tsinghua University)

**通讯引用:** 26639 | [OpenAlex ID](https://openalex.org/A5006076332)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种在中文学术文本分类中，通过改进训练目标实现类别平衡的轻量化方法AutoTail-BSFGM；

**💡 创新点**

创新点在于将自动化尾部先验调整、弱化Balanced Softmax辅助损失与FGM对抗正则相结合，且该方法仅修改训练阶段，无需额外模型或推理开销；

**🔧 技术方法**

所用技术包括：自动化尾部先验门控（AutoTail）、弱化的Balanced Softmax辅助损失、Fast Gradient Method（FGM）对抗训练，以及基于label smoothing的基线；

**📊 数据集**

实验数据集为中文学术语料库CSL，构建了两项任务：67标签的摘要-学科分类和13标签的标题-类别分类；

**📈 对比分析**

与同基准标签平滑的RoBERTa-WWM和MacBERT基线相比，AutoTail-BSFGM在摘要任务上在验证集提升约1.4-1.5个百分点，锁盒集也均有正向提升；在标题任务中验证精度提升约0.7个百分点，宏观F1和均衡准确率提升更显著，但锁盒精度变化不大；

**⚠️ 局限性**

局限性包括：训练时间增加约1.6-2.0倍，实验仅覆盖两任务且仅用三次随机种子，未做官方排行榜提交，且未完成全面消融与对比分析。

---

## 485. When Attention Collapses: Stage-Aware Visual Token Pruning from Structure to Semantics

**arXiv ID:** 2606.03569 | [PDF](https://arxiv.org/pdf/2606.03569v1)

**作者:** Jiahui Wang `[一作]` (Shandong University), Huanghe Zhang `[通讯]` (Shandong University)

**通讯引用:** 361 | [OpenAlex ID](https://openalex.org/A5075236069)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了无训练的两阶段视觉标记剪枝框架STS，先在视觉编码器阶段通过势能最小化实现结构多样性，再在LLM中利用交叉注意力进行任务相关过滤。

**💡 创新点**

通过将剪枝分解为结构保持与语义过滤两步，突破传统单指标注意力剪枝的“聚集陷阱”，同时采用电势相互排斥机制保证全局多样性。

**🔧 技术方法**

基于电场势能的互斥采样、kNN一致性分析、交叉注意力任务感知过滤、两阶段贪心选择算法以及无训练参数的实现。

**📊 数据集**

在LLaVA‑v1.5、LLaVA‑NeXT、Qwen2.5‑VL等多模态模型上，评估于GQA、MMBench、MME、POPE、ScienceQA、TextVQA、VQA‑v2、VizWiz等八大图像基准。

**📈 对比分析**

与VisionZip、SparseVLM、DART、DivPrune、Zoo‑Prune、AgilePrune等多种单阶段或注意力剪枝方法对比，在相同token预算下，STS在多任务上平均提升约1–2个百分点，且在极端稀疏下仍保持约96%相对性能，显著降低FLOPs、预填充延迟和KV缓存。

**⚠️ 局限性**

需要访问模型内部视觉标记，无法应用于黑盒API；在极端稀疏下仍有一定性能损失；并未结合模型量化或更广泛的高效LLM技术。

---

## 486. Learned Non-Maximum Suppression for 3D Object Detection

**arXiv ID:** 2606.03568 | [PDF](https://arxiv.org/pdf/2606.03568v1)

**作者:** Timo Osterburg `[一作]` (TU Dortmund University), Torsten Bertram `[通讯]` (TU Dortmund University)

**通讯引用:** 4046 | [OpenAlex ID](https://openalex.org/A5051394730)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了两种轻量级、检测器无关的学习型后处理模块（D2D-Rescore与GossipNet3D），用以替代传统的非极大抑制，实现对LiDAR 3D检测候选框的学习式过滤；

**💡 创新点**

创新点在于：①通过检测间自注意力（D2D-Rescore）和局部BEV消息传递（GossipNet3D）实现检测级置信度再评分；②采用与nuScenes评估一致的度量感知贪婪匹配策略，保证训练与验证行为一致；③实现了既高精度又轻量化的后处理。

**🔧 技术方法**

使用的技术包括Transformer自注意力、GossipNet图神经网络、Fourier编码、MLP残差头、二分类交叉熵损失、度量感知贪婪匹配、Top‑K筛选等。

**📊 数据集**

在nuScenes数据集上，以CenterPoint检测器生成的候选框作为输入进行评估。

**📈 对比分析**

与CircleNMS、Soft‑CircleNMS、GCN等传统后处理方法对比，D2D-Rescore在mAP与NDS上提升约1–2个百分点，同时推理时间与显存占用最低；GossipNet3D在mAP上表现最佳，但推理速度与显存略高。

**⚠️ 局限性**

局限性包括：只在检测级别进行改进，未进一步优化框尺寸与姿态；需要额外训练；GossipNet3D在内存与推理速度上不及D2D‑Rescore；模块在不同检测器上的泛化性能仍需验证。

---

## 487. How Many Trees in a Random Forest? A Revisited Approach with Plateau Search and Optuna Integration

**arXiv ID:** 2606.03549 | [PDF](https://arxiv.org/pdf/2606.03549v1)

**作者:** Vadim Porvatov `[一作]` (Sberbank), Andrey Lange `[通讯]` (Skolkovo Institute of Science and Technology)

**通讯引用:** 229 | [OpenAlex ID](https://openalex.org/A5053415855)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并实现了一种基于三元组平稳搜索的随机森林树数自适应调优算法，并将其集成到 Optuna TPE 中。

**💡 创新点**

创新点在于将树数从搜索空间剔除，采用相对 OOB 分数的平稳阈值通过三元组动态调整，结合理论分析阐明与极限分数的关系，避免了传统的固定上限和早停缺陷。

**🔧 技术方法**

使用 Optuna 的 TPE 超参数优化、warm_start 的随机森林训练、OOB 评分、相对差分平稳判定、以及对 OOB 统计量的方差与极限误差分析。

**📊 数据集**

在 12 个 UCI/Kaggle 标准分类/回归数据集以及高维生物信息学数据集 GeneExpressionCancer RNA‑Seq、Arcane、Dorothea 进行实验。

**📈 对比分析**

与固定树数 TPE、Hyperband、单调早停等基准进行比较；实验显示 PLATEAU 在多数数据集上能获得相当或更优的 OOB 分数，同时使用的树数和计算时间更低，尤其在高维数据集上显著提升。

**⚠️ 局限性**

局限性包括对平稳阈值 ε 的敏感性、需足够试验次数以保证收敛、仅针对可单调改进的评分指标，且在极端小样本或非 OOB 兼容任务上可能不适用。

---

## 488. Efficient Transformer-Based Localized Patch Sampling for Choroid Plexus Segmentation in Multiple Sclerosis

**arXiv ID:** 2606.03566 | [PDF](https://arxiv.org/pdf/2606.03566v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 489. Towards Intrusion Detection Systems for RPL-based IoT Networks using Foundation Models

**arXiv ID:** 2606.03530 | [PDF](https://arxiv.org/pdf/2606.03530v1)

**作者:** Elias Lunderbye `[一作]` (Uppsala University), Andreas Johnsson `[通讯]` (Uppsala University)

**通讯引用:** 651 | [OpenAlex ID](https://openalex.org/A5087864276)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

使用预训练的时间序列基础模型 MOMENT 进行 RPL‑IoT 网络的入侵检测与多类攻击识别。

**💡 创新点**

首次将基础模型迁移到 RPL 路由攻击场景，实现低调优、少参数、高效的多类攻击识别。

**🔧 技术方法**

技术包括 MOMENT 编码器、patch 处理、因果滑动窗口、少量可训练头部、宏 F1 评估、Cooja 仿真与数据预处理。

**📊 数据集**

使用 IoT‑Attacks‑IDS 仓库中的 Cooja 仿真数据，包含 Blackhole、DIS‑Flooding、Worst parent、Local repair 四种攻击及其变体。

**📈 对比分析**

通过宏 F1 对 5 类标签（正常、Blackhole、DIS‑Flooding、Worst parent、Local repair）进行评估，并与 LSTM 二分类基线比较；在窗口大小 100、步长 16 的配置下，F1 达到 0.875，训练时间约 311 s，推理时间 7.8 ms，性能与最先进方法相当。

**⚠️ 局限性**

局限包括对 Blackhole 的识别效果仍弱、Local repair 与 Worst parent 混淆频繁、仅在仿真环境中验证、仅评估 MOMENT，缺乏对真实网络与更多攻击类型的验证。

---

## 490. Throughput Optimization for Multi-AP IEEE P802.11bq Networks Based on Combinatorial Multi-Armed Bandits

**arXiv ID:** 2606.03528 | [PDF](https://arxiv.org/pdf/2606.03528v1)

**作者:** Anshan Yuan `[一作]` (Sun Yat-sen University), Xinghua Sun `[通讯]` (Sun Yat-sen University)

**通讯引用:** 5447 | [OpenAlex ID](https://openalex.org/A5100441911)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

在稠密多AP IEEE P802.11bq 网络中提出了一种分布式、基于组合多臂赌博机的自适应配置方法，实现对子 7GHz 控制通道与 60 GHz mmWave 数据通道参数的联合优化，目标是最大化每个 AP 的短期吞吐量并提高整个网络的总吞吐量。

**💡 创新点**

创新点在于：① 将四个 PHY/MAC 参数（冲突窗口、CCA 阈值、波束宽度和 MCS 保留余量）分组，构建多组组合赌博机模型；② 设计了基于 Hadamard 指导的可行探索与组内“逐步接纳-拒绝”剪枝机制，显著降低探索成本；③ 通过实用的分组评分而非严格的加性模型，兼顾了 IEEE P802.11bq 的跨层耦合特性。

**🔧 技术方法**

主要技术包括：组合多臂赌博机（CMAB）、Hadamard 矩阵引导的实验设计、分组可行映射、经验主效应评分、逐步拒绝（CSAR）算法以及基于包级仿真的 WLAN 物理/MAC 过程建模。

**📊 数据集**

使用基于仿真的实验数据：不同 AP 密度（2-12）下的多AP IEEE P802.11bq 网络，仿真参数表中给出的 23 条 elementary arm，仿真时长 5 s，10 次独立实验平均结果。未使用公开数据集。

**📈 对比分析**

与三类基线比较：① 传统 MAB-Thompson Sampling（TS）全 Cartesian arm；② 固定配置的 IEEE P802.11bq 与纯 mmWave；③ 纯 mmWave TS。结果显示：CSAR 在大多数 AP 密度下获得最高总吞吐量，单 AP 吞吐量亦优于或相近；相较于 TS，吞吐量稳定时间缩短约 49%；碰撞率保持在中等范围，证明并非仅通过降低碰撞获得高吞吐。

**⚠️ 局限性**

局限性包括：① 理论收敛与性能保证仅为经验性，未给出正式的误差/收敛速率；② 仅在静态、无移动、无障碍、同质流量的仿真环境下验证，实际部署中的阻塞、用户异质性、功率控制等因素尚未考虑；③ 仅评估吞吐量和碰撞，未考虑公平性、时延或 QoE 等指标。

---

## 491. Channel Chart Location Privacy Based on Geo-Indistinguishability

**arXiv ID:** 2606.03571 | [PDF](https://arxiv.org/pdf/2606.03571v1)

**作者:** Atsu Kokuvi Angélo Passah `[一作]` (CY Cergy Paris University), Arsenia Chorti `[通讯]` (Barkhausen Institut gGmbH)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了基于通道图的图位置不可区分性（CLI）隐私框架，并在此基础上设计了几何感知的马氏距离平面拉普拉斯机制（MNPL）。

**💡 创新点**

创新点在于将 geo‑indistinguishability 推广到低维通道图，并通过局部协方差对噪声进行方向自适应，以在保证隐私的同时保持图结构。

**🔧 技术方法**

核心技术包括通道图（channel charting）、平面拉普拉斯机制、马氏距离平面拉普拉斯机制、差分隐私（Gaussian 机制）以及 KNN 估计局部协方差。

**📊 数据集**

实验使用在城市环境下收集的 64 天线 OFDM 通道测量数据集，对用户轨迹进行特征提取和编码。

**📈 对比分析**

与传统差分隐私、平面拉普拉斯和无噪声基线比较，MNPL 在邻域连通性（tw、ct）达到 90%+，查询误差（RQE）和质量损失（QL）显著低于 PL，显示出显著的性能提升。

**⚠️ 局限性**

主要局限包括 KNN 计算的高复杂度、局部协方差估计的稳定性依赖以及在时间动态场景和高维通道图中的推广尚未验证。

---

## 492. \textsc{CR-Seg}: Attention-Guided and CoT-Enhanced Coarse-to-Refined Reasoning Segmentation

**arXiv ID:** 2606.03564 | [PDF](https://arxiv.org/pdf/2606.03564v1)

**作者:** Yifan Cao `[一作]` (Northeastern University), Yifei Zhang `[通讯]` (Northeastern University)

**通讯引用:** 3531 | [OpenAlex ID](https://openalex.org/A5100386920)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段的推理分割框架CR‑Seg，利用多模态大语言模型（MLLM）的注意力图作为粗略分割先验，并通过SAM进行精细掩码生成；同时引入全局‑至‑局部链式思维（GLCoT）提升推理一致性。

**💡 创新点**

创新点在于：①使用可监督的注意力图作为跨模态桥梁，既保留整体语义又避免隐藏状态对齐难题；②提出EAP模块将注意力图转换为SAM可用的掩码先验和点提示；③设计GLCoT三步推理策略，减少推理‑答案不一致。

**🔧 技术方法**

技术包括：Qwen3‑VL‑4B MLLM、SAM3 分割模型、可学习查询与注意力提取、EAP模块（注意力聚合、点采样、掩码投影）、LoRA 微调、Cosine 与 Dice/BCE/边界损失组合、GLCoT 推理蒸馏。

**📊 数据集**

数据集：RefCOCO/RefCOCO+ 作为 Stage‑1 训练集；ReasonSeg 作为 Stage‑2 训练集；自制 FReasonSeg（基于 RefCOCOm）评估细粒度目标区分能力。

**📈 对比分析**

与 ERS 与 IRS 基线（Seg‑Zero、VisionReasoner、Dr. Seg、LISA、GSVA、LENS 等）比较，在 ReasonSeg 上提升约 2.4% gIoU；在 FReasonSeg（L1/L2/L3）上实现 state‑of‑the‑art，尤其在高难度 L3 子集表现优异。

**⚠️ 局限性**

局限：①仍需依赖 SAM，注意力图与 SAM 原生掩码输入不完全兼容；②依赖 MLLM 内部注意力机制，需要额外对齐训练；②对极少样本/未见场景的适应性尚待验证。

---

## 493. Pushing the Limits: A Framework to Reform Institutional Ethics Review of Environmentally-Impactful Computing Research

**arXiv ID:** 2606.03547 | [PDF](https://arxiv.org/pdf/2606.03547v1)

**作者:** Nicolas Gold `[一作]` (University College London), Ross Purves `[通讯]` (University College London)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5027014487)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了针对计算密集型研究（CIR）在机构伦理审查中纳入环境影响的框架，涵盖三层：评估CIR是否应纳入伦理审查的标准、伦理审查员的评估属性以及研究者的反思方法，并通过对英国大学伦理政策的初步调查验证框架可行性。

**💡 创新点**

创新点在于：① 将环境因素明确纳入伦理审查范围的判定标准；② 设计了七项可操作的评估属性，帮助审查员一致评估环境影响；③ 提出基于LIMITS视角的研究者反思工具，支持从长期视角评估研究对环境的潜在影响。

**🔧 技术方法**

主要技术手段包括：政策文件内容分析、定性问卷式调查、案例式阐释、基于已存在框架（如SIGSOFT、EU AI 可信度评估列表、Data Hazard Labels）构建评估属性，以及基于表格的因果映射方法。

**📊 数据集**

使用的数据集为：英国Russell Group 24所高校公开的研究伦理政策文档（22份可获得），以及对其环境相关条款的关键词检索结果。

**📈 对比分析**

本文并未进行实验比较，因此没有性能指标；相较于现有伦理审查实践，提出的框架在理论上提供了更细化的环境评估维度，然而缺乏实证评估其在实际审查流程中的效果和效率。

**⚠️ 局限性**

限制包括：① 调查样本仅限于英国Russell Group高校，可能不具备全球代表性；② 仅对政策文本进行分析，未收集审查员或研究者的实际反馈；③ 框架在真实机构中的可操作性尚未得到试点验证；④ 可能需要在不同学科和监管环境下进一步细化阈值与标准。

---

## 494. SAGE: A Quantitative Evaluation of Socialized Evolution in Agent Ecosystems

**arXiv ID:** 2606.03544 | [PDF](https://arxiv.org/pdf/2606.03544v1)

**作者:** Linyue Pan `[一作]` (Tsinghua University), Xunliang Cai `[通讯]` (Meituan)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Social Agent Group Evolution（SAGE）框架，评估公开经验在多智能体演化中的增益；

**💡 创新点**

通过计算匹配的对照实验，将群体共演化与单体自我改进对比，揭示公开历史对不同代理、不同任务的可变收益，并探讨不同历史表示方式与对手特定适应的效果；

**🔧 技术方法**

使用迭代自我改进技术、计算匹配实验设计、不同历史表示（原始轨迹、排行榜、精简摘要等），以及基于多种竞技场（MLR‑Bench、DrugWars、Splendor）的演化和评估；

**📊 数据集**

使用开放式机器学习研究任务库MLR‑Bench、模拟交易游戏DrugWars以及多人棋盘游戏Splendor的结果数据；

**📈 对比分析**

在计算匹配的条件下，对比群体历史可见性（SAGE）与仅自我历史（Self）两种情景，测量每轮性能差异；结果显示：在DrugWars中部分代理（如DeepSeek‑V3.2、GPT‑5.4）获得显著提升；在MLR‑Bench提升不明显；在Splendor中出现对特定对手（如Doubao）的针对性改进；不同历史表示方式的效果也各异，过滤或压缩形式往往优于原始完整日志；

**⚠️ 局限性**

仅使用了五个主流模型族，限制了对更小或开源代理的普适性；公共历史长度受限，未探索更长或更复杂的共享经验；实验环境为受控基准，可能无法完全反映真实部署中的安全与隐私问题。

---

## 495. Attend to Anything: Foundation Model for Unified Human Attention Modeling

**arXiv ID:** 2606.03540 | [PDF](https://arxiv.org/pdf/2606.03540v1)

**作者:** Wenzhuo Zhao `[一作]` (Sichuan University), Qijun Zhao `[通讯]` (Sichuan University)

**通讯引用:** 5022 | [OpenAlex ID](https://openalex.org/A5085914001)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Attend to Anything Model (AAM)，实现跨模态、跨场景、跨任务的人类注意力统一建模。

**💡 创新点**

① 将注意力视为从一般先验到具体任务的层次化蕴含过程，在双曲空间中学习层次蕴含约束；② 用Fokker–Planck动态模块将视频注意力建模为连续的物理传输过程；③ 通过统一的多模态提示、层次蕴含与流体动力学相结合，突破传统任务分离与场景依赖。

**🔧 技术方法**

使用双曲几何层次蕴含（hyperbolic entailment）、Fokker–Planck dynamics、预训练自监督骨干（DINOv3、CLIP、Wav2CLIP）以及跨模态交叉注意力与自适应解码。

**📊 数据集**

训练集 Attention‑1.75M（约1.75M注视点，涵盖图像、视频与音视频 8+4+6 个数据集），评估 16 个公开基准（如 MIT1003、SALICON、DHF1K、AVAD 等）。

**📈 对比分析**

与 16 个基准上 10+ 现有方法（SUM、UNISAL、VSSM、TFS-Net 等）对比，平均在图像、音视频、视频任务上分别提升约 5–6% 的评价指标（AUC、SIM、CC、NSS），视频推理速度提升约 4 倍，参数约 21M，可在任意帧长进行推理。

**⚠️ 局限性**

局限：仍需依赖预训练的自监督骨干；在极端稀疏或高度动态场景下 Fokker–Planck 近似可能受限；对多语言提示的泛化尚未系统验证；模型对极端噪声/遮挡的鲁棒性尚待进一步提升。

---

## 496. Can LLM Rerankers Predict Their Own Ranking Performance?

**arXiv ID:** 2606.03535 | [PDF](https://arxiv.org/pdf/2606.03535v1)

**作者:** Shiyu Ni `[一作]` (State Key Laboratory of AI Safety), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了LLM reranker内部的查询性能预测（QPP），探讨训练无监督与有监督方法，并提出可生成校准置信度的两种监督方法（Verb-Num和Verb-List）。

**💡 创新点**

创新点在于将自一致性作为无监督QPP信号，引入可直接输出的校准置信度，并通过最小化额外生成成本实现高效的reranker内部QPP。

**🔧 技术方法**

采用生成式LLM reranker（LLaMA3.1、Qwen2.5等），自一致性评估、verbalized confidence、监督训练（Verb-Num/Verb-List）、数据增强（Qwen3-32B对MS MARCO的注释）以及ECE与Spearman等评估指标。

**📊 数据集**

使用TREC Deep Learning 2019–2022和MS MARCO（以及其增强版本）作为实验数据集。

**📈 对比分析**

与SOTA无监督QPP‑Gen对比，self‑consistency在Spearman相当且ECE更低；Verb‑Num/Verb‑List在Spearman上进一步提升并在ECE上更优，尤其Verb‑List校准效果更佳，整体在TREC DL实验中表现最佳。

**⚠️ 局限性**

局限性包括仅针对二元相关性，未探索多级相关；缺乏进一步提升无监督训练的方案；未验证长程推理或反思对自我感知的影响。

---

## 497. When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation

**arXiv ID:** 2606.03532 | [PDF](https://arxiv.org/pdf/2606.03532v1)

**作者:** Haowei Guo `[一作]` (Peking University), Wentao Zhang `[通讯]` (Peking University)

**通讯引用:** 15564 | [OpenAlex ID](https://openalex.org/A5100459860)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了自监督上游策略蒸馏中教师更新频率对训练稳定性的影响，并提出了基于奖励提升和序列长度尾部安全性的门控教师刷新策略（CGTR）。

**💡 创新点**

创新点在于：①系统地将教师更新间隔（隔离期）与教师-学生时间耦合的结构性需求分离，证明隔离期是实现稳定学习的关键；②发现固定间隔刷新导致的长期“状态无关崩溃”失效模式；③设计CGTR通过奖励门和长度尾部门实现状态感知、避免无时序刷新导致的灾难性复制。

**🔧 技术方法**

技术上使用了自监督上游策略蒸馏（self‑on‑policy distillation）与KL蒸馏正则，结合硬刷新与EMA对比，构建了临时KL结构、刷新冲击、长度尾部风险的诊断框架，并实现了三条件门控刷新（隔离门、奖励门、长度门）。

**📊 数据集**

实验数据集包括SciKnowEval四个科学问答子集（Chemistry、Physics、Biology、Materials）和ToolAlpaca工具使用任务，模型为Qwen3‑8B。

**📈 对比分析**

与固定间隔刷新（如M=50）和EMA（α=0.04）对比，CGTR在长达300步的训练中实现了零崩溃并获得最高最终和峰值准确率，平均提升约0.05–0.08（以test_mean@16计），同时保持较低的熵和序列长度。

**⚠️ 局限性**

局限性在于实验仅在单一模型规模（Qwen3‑8B）和四个任务上进行，未验证在更大模型、更多任务或更长训练周期的泛化；奖励门仅基于二元正确性信号，未探讨更丰富或学习型奖励信号的影响。

---

## 498. Reserve Depletion and Security Runway in Proof-of-Stake Systems

**arXiv ID:** 2606.03587 | [PDF](https://arxiv.org/pdf/2606.03587v1)

**作者:** Paolo Penna `[一作]` (IOG), Manvir Schneider `[通讯]` (Cardano Foundation)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

对基于权益证明（PoS）的区块链系统中，预留资金与交易费之间的转移过程进行了离散时间随机博弈建模，推导了状态相关的最小储备阈值，并以此阈值定义安全跑道和成功手交的击中时间问题，同时扩展至前瞻性验证者参与的Markov动态博弈。

**💡 创新点**

① 将安全问题转化为状态依赖的击中时间问题，提供了闭式的状态阈值和保守压力测试界限；② 在Markov扩展中给出了动态第一阶条件，揭示未来奖励对当前参与度的影响；③ 通过概率界限和期望手交时间分析，为多种价格/需求动力学提供定量风险评估。

**🔧 技术方法**

离散时间随机博弈与均衡分析、闭式解、状态依赖阈值构造、Geometric Brownian Motion 与离散对数正态需求的概率界限、Markov Perfect Equilibrium、动态规划与连续状态的第一阶条件。

**📊 数据集**

本工作为理论研究，不使用真实外部数据集；通过参数化示例（如N=5, θ=0.25, ρ=0.04, κ=0.6, β=0.8, s=1）进行数值模拟，以展示阈值图、失败/手交概率和期望手交时间。

**📈 对比分析**

与传统的名义耗尽日期或稳态奖励比例指标对比，本文的安全跑道阈值在价格或需求冲击下能更精确捕捉安全边界；在模拟中，阈值方法能够预测失败概率、手交概率和期望手交时间，显示出更细粒度的风险评估与决策支持。

**⚠️ 局限性**

① 价格与需求被视为外生过程，未考虑代币定价和用户采纳的反馈；② 验证者被假设为同质，阈值结果只能作为基准；③ 采用保守压力测试忽略未来费用流入；④ 未包含自适应储备政策的设计与动态机制设计。

---

## 499. DDOR: Delta Debugging for Explainable Overrefusal Testing and Repair

**arXiv ID:** 2606.03601 | [PDF](https://arxiv.org/pdf/2606.03601v1)

**作者:** Qinyan Zhou `[一作]` (Southeast University), Dongxia Wang `[通讯]` (Zhejiang University)

**通讯引用:** 410 | [OpenAlex ID](https://openalex.org/A5101993888)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DDOR 框架，用于自动检测和修复大型语言模型的 overrefusal（误拒绝）问题。

**💡 创新点**

创新点在于：① 使用 delta‑debugging 进行最小拒绝触发片段（mRTF）定位，① 以 mRTF 为条件生成多样化、可解释的测试用例；② 采用多模型 Chain‑of‑Thought 验证机制，过滤真正安全的提示，保证测试集纯度；③ 在定位基础上实现定向 prompt 维修，显著降低 overrefusal 同时保持语义完整。

**🔧 技术方法**

核心技术包括：delta‑debugging、LLM 基于提示生成、基于多模型 CoT 的安全评估/过滤、目标 prompt 的微调修复、黑盒 API 调用。

**📊 数据集**

使用的数据集包括 OR‑Bench‑Hard‑1K、XSTest、S‑Eval、OR‑Bench Toxic 等，并在多种公开 LLM（GPT‑5、GPT‑5‑Mini、qwen3‑30b、gemma‑3‑1b‑it 等）上进行实验。

**📈 对比分析**

与 OR‑Bench、RASS、ORFuzz、full‑prompt rewrite 等基线比较，DDOR 在 overrefusal 率提升约 19%，生成 5‑10 倍更多高质量测试案例；修复率平均提升 70% 以上，同时语义相似度提升 5–10%。

**⚠️ 局限性**

局限性：① 只定位单一最小触发片段，无法枚举所有可能触发组合；② 依赖拒绝检测器的准确性，若检测误判会影响定位和评估；③ delta‑debugging 需要多轮 API 调用，虽然成本有限但仍非零；④ 仅在黑盒环境下验证，对内部安全机制的深层改进仍无直接作用。

---

## 500. Making Embodied AI Reliable: A Community Agenda from Testing to Formal Verification

**arXiv ID:** 2606.03593 | [PDF](https://arxiv.org/pdf/2606.03593v1)

**作者:** Xi Zheng `[一作]`, Corina Pasareanu `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了面向可靠 Embodied AI 的综合生命周期保障框架，融合场景化测试、组合式形式化验证与运行时保障，并强调通过 neuro‑symbolic 表示实现闭环反馈；

**💡 创新点**

创新地将测试、验证与运行时保障三大保障环节统一到一个 neuro‑symbolic 生命周期工作流中，提出基于场景语义覆盖、结构化符号抽象与不确定性感知推理的可靠性提升策略；

**🔧 技术方法**

使用了大语言模型 (LLM) 生成与校正场景规范、符号约束与场景图/世界模型、结构化 DSL、神经网络鲁棒性分析、概率符号推理、风险感知人机适配与安全回退机制；

**📊 数据集**

本文为综述性质，未给出具体实验数据；参考文献中涉及的典型数据集包括 KITTI、Waymo、ROS2 仿真等；

**📈 对比分析**

由于是方法综述，未给出实验比较；作者指出现有方法在覆盖度、形式化可验证性与运行时适应性方面各有局限，强调需要统一评价指标与基准；

**⚠️ 局限性**

当前保障方法在抽象层次、指标统一、跨阶段接口兼容以及对开放世界分布漂移的实时检测上仍缺乏成熟方案；缺少实证评估与标准化评测流程。

---

## 501. Training a Predictive Coding Network on ImageNet using Equilibrium Propagation

**arXiv ID:** 2606.03584 | [PDF](https://arxiv.org/pdf/2606.03584v1)

**作者:** Tugdual Kerjan `[一作]` (Rain AI), Benjamin Scellier `[通讯]` (Rain AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Equilibrium Propagation (EP) 的预测编码网络 (PCN) 训练方法，并将其应用于10层VGG网络，在全尺寸ImageNet上实现了与BP相近的性能

**💡 创新点**

将EP的“nudging”扰动方法与中心差分方案相结合，首次将PCN与EP扩展到ImageNet规模，并展示了随机和反向差分方案在PCN中与中心方案同样有效

**🔧 技术方法**

利用EP的弱扰动学习框架，采用中心/随机/反向差分法与nudging扰动；在PCN中使用投影梯度下降 (PGD) 进行“nudged”相平衡；对比BP的标准反向传播

**📊 数据集**

在四个视觉数据集上实验：MNIST、CIFAR‑10、CIFAR‑100、ImageNet 32×32 以及完整尺寸 ImageNet（1.28M 图像，224×224，1000 类）

**📈 对比分析**

与BP对比，EP 在 VGG5 上在不同批量、损失函数和权重初始化下均保持竞争力；在 ImageNet 上 EP 训练的 VGG10 Top‑5 错误率为 13.23%，仅比 BP 基线高 0.93%；在VGG10Skip（带跳跃连接）模型上表现相似

**⚠️ 局限性**

主要局限在于训练时间长（需多轮“nudged”迭代），对参数 β 与迭代次数 K 的敏感度需谨慎选择；未实现硬件物理实现，且实验仍以 GPU 计算为主

---

## 502. STC: Reversible Digit-Context Decomposition for BWT-Family Text Compression

**arXiv ID:** 2606.03570 | [PDF](https://arxiv.org/pdf/2606.03570v1)

**作者:** Jingyang Du `[一作]` (Minzu University of China), Anling Xiang `[通讯]` (Minzu University of China)

**通讯引用:** 39 | [OpenAlex ID](https://openalex.org/A5006284638)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种可逆的数字上下文分离预处理，将结构化文本中的数字段拆分为占位符主流和侧流，随后使用BWT等后端压缩。

**💡 创新点**

在BWT前对数字段进行可逆分离并基于本地上下文对侧流进行排序，消除主流与数字值之间的干扰。

**🔧 技术方法**

使用Burrows‑Wheeler Transform、可逆预处理、桶化分组、上下文键排序、原始/对位/整数大端打包以及固定内部BWT/M03编码。

**📊 数据集**

主要在enwik9（1 000 M字节英文维基百科XML）上进行实验，并在enwik8、Calgary、Canterbury、Silesia等语料上做补充验证。

**📈 对比分析**

在相同后端编码器下进行同码器消融实验，STC全分离比无分离节约2 629 561字节，压缩率提升约1.6%，并与bsc‑m03等基线做对比。

**⚠️ 局限性**

仅在实验机器上占用约12 GB内存，压缩/解压耗时约11 分钟，尚未实现生产级性能，缺乏对多样语料的普适性验证。

---

## 503. Partially Observable Adversarial Patch Attacks on Vision-Language-Action Models in Robotics

**arXiv ID:** 2606.03556 | [PDF](https://arxiv.org/pdf/2606.03556v1)

**作者:** Xiaofei Wang `[一作]` (University of Science and Technology of China), Keke Tang `[通讯]` (Guangzhou University)

**通讯引用:** 11202 | [OpenAlex ID](https://openalex.org/A5022039557)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6215c339-3735-4be3-8a07-5bbb7004712d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了在部分可观测条件下针对视觉‑语言‑动作（VLA）模型的对抗补丁攻击方法。

**💡 创新点**

创新点在于将攻击限定为只利用短期轨迹前缀，并通过注意力定位与语义‑曲率双重损失优化实现长期干扰。

**🔧 技术方法**

使用跨模态注意力、语义对齐失效与轨迹曲率代理的两阶段优化，并在PyTorch中实现。

**📊 数据集**

使用LIBERO基准（Spatial、Object、Goal、Long四类任务）和Hume VLA模型，并在真实Rokae xMate ER7 Pro机器人上验证。

**📈 对比分析**

与UADA、UPA、TMA等基线对比，攻击成功率和归一化攻击成功率均显著提升，尤其在K=30时表现最优。

**⚠️ 局限性**

局限在于需灰盒访问注意力信息、仅适用于已知VLA架构，且对极短前缀或黑盒环境效果有限。

---

## 504. Bionic Human-Motion Style Transfer for Physically Executable Whole-Body Control of Humanoid Robots

**arXiv ID:** 2606.03536 | [PDF](https://arxiv.org/pdf/2606.03536v1)

**作者:** Tianchen Huang `[一作]` (University of Science and Technology of China), Shiwu Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 22813 | [OpenAlex ID](https://openalex.org/A5069207720)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于多条件潜在扩散模型的仿生风格迁移框架，能够从短人类风格示例生成可执行的全身机器人运动，并通过参考预览跟踪实现在真实人形机器人的稳定执行。

**💡 创新点**

创新点包括：①将物理感知的解码运动正则化（接触一致性与时序平滑）嵌入到扩散模型中；②采用 classifier‑free 指引尺度实现风格强度可控；③通过风格多样化训练和聚类+蒸馏的全身跟踪策略，实现短示例可复用且可在硬件上稳健执行。

**🔧 技术方法**

使用技术包括多条件潜在扩散模型、MotionCLIP 风格编码、VAE 嵌入、物理约束正则化、参考预览全身跟踪、DAgger 蒸馏、以及单机仿真与 Unitree G1 硬件实验。

**📊 数据集**

使用的数据集为 AMASS、LaFAN1 运动数据，并自行采集短风格示例；此外生成了 1000 条风格化轨迹用于训练跟踪器。

**📈 对比分析**

与 MotionDiffuse、Motion Puzzle、MCM‑LDM 等动画风格迁移基线对比，本文方法在感知指标（FMD、SRA）与机器人可执行指标（FSF、A_pos、J_pos、Sim. SR）上均表现优异；在 125 次硬件试验中实现 96% 的成功率，显著优于基线。

**⚠️ 局限性**

限制包括：风格放大可能导致高频抖动、重心偏移和步态不稳定；对内容‑风格解耦的保障不完全；实验仅在平地进行，未覆盖复杂地形或交互场景，未来需扩展至更广泛的硬件验证。

---

## 505. World Models Meet Language Models: On the Complementarity of Concrete and Abstract Reasoning

**arXiv ID:** 2606.03603 | [PDF](https://arxiv.org/pdf/2606.03603v1)

**作者:** Yucheng Zhou `[一作]` (University of Macau), Jianbing Shen `[通讯]` (University of Macau)

**通讯引用:** 16278 | [OpenAlex ID](https://openalex.org/A5023184215)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了PF-OPSD框架，训练多模态大型语言模型在未来结果预测中自适应调用并验证世界模型生成的视觉回放，以实现受控的具体推理；

**💡 创新点**

创新点在于将世界模型输出视为不确定的具体推理轨迹，利用特权未来视频进行基于优势的自我蒸馏，学习何时调用、验证和利用回放；

**🔧 技术方法**

核心技术包括受控具体推理策略、Privileged‑Future On‑Policy Self‑Distillation (PF‑OPSD)、优势加权蒸馏、以及对回放验证和依赖的多步决策；

**📊 数据集**

使用了两个人工验证的基准数据集：Puzzle‑LOOKAHEAD（4,636个四选题目）和 Real‑World Physical Future (4,404个四选题目)；

**📈 对比分析**

与零样本/无模拟基线、Qwen3.5‑9B训练基线以及prompt‑based工作流对比，PF‑OPSD在两个基准上分别提升约10.6%和10.9%的准确率，并在噪声回放下保持鲁棒性；

**⚠️ 局限性**

局限性包括对训练时特权未来视频的依赖、仅适用于可产生相关回放的世界模型、未覆盖更长时程或交互式环境，以及对更专业化物理推理场景的适用性不足。

---

## 506. From Prompt to Service: An SLM-Based Agent Orchestration Gateway for AI-Driven Virtual Worlds

**arXiv ID:** 2606.03557 | [PDF](https://arxiv.org/pdf/2606.03557v1)

**作者:** Louis Nisiotis `[一作]` (UCLan Cyprus), Aimilios Hadjiliasi `[通讯]` (UCLan Cyprus)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出并实现了基于小型语言模型的代理编排网关，实现在虚拟世界与多种AI后端间的意图驱动路由。

**💡 创新点**

通过在边缘部署轻量化SLM作为意图路由器，并将路由决策与后端服务解耦，实现了不需要客户端维护多端点的统一接口，并引入分层模型配置提升响应效率。

**🔧 技术方法**

使用FastAPI构建网关，基于Llama.cpp在NVIDIA Jetson Orin NX上部署量化的Qwen2.5、SmolLM等SLM，并对小模型进行QLoRA微调；同时利用服务注册表、意图路由器、后端调用层实现。

**📊 数据集**

在InterwovenXR虚拟博物馆场景中收集并构造了500条访客提示的平衡数据集（5类意图），并通过Qwen2.5-14B作为上位参考模型验证标签。

**📈 对比分析**

通过将多种SLM在边缘设备上对同一数据集进行路由准确率、无效输出率与平均路由时延评测，并在分层配置下测量端到端延迟；结果表明微调后的0.5B模型路由准确率提升至83%，整体延迟约1.45秒，优于单一大模型。

**⚠️ 局限性**

实验仅覆盖单一博物馆场景与500条提示，未评估多轮对话、不同硬件或网络条件；并且微调需随路由表变更维护，限制了迁移与可扩展性。

---

## 507. The Comparative Trap: How Social Comparison Orientation Drives Problematic Generative AI (GenAI) Use

**arXiv ID:** 2606.03560 | [PDF](https://arxiv.org/pdf/2606.03560v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 508. Exploiting Verification-Generation Gap: Test-Time Reinforcement Learning with Confidence-Conditioned Verification

**arXiv ID:** 2606.03608 | [PDF](https://arxiv.org/pdf/2606.03608v1)

**作者:** Jiahui Li `[一作]` (Sun Yat-Sen University), See-Kiong Ng `[通讯]` (National University of Singapore)

**通讯引用:** 5527 | [OpenAlex ID](https://openalex.org/A5090171111)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为 TTRL‑CoCoV 的自监督测试时强化学习框架，用置信度自适应的验证与探索奖励共同提升大型语言模型在复杂推理任务上的 Pass@k 生成覆盖率。

**💡 创新点**

创新点包括：① 针对无标签 TTRL 中 Pass@k 优化的三大瓶颈（伪标签噪声、长度方差坍塌、目标失配）进行系统诊断；② 通过置信度分区构建自我验证机制，实现生成器与验证器的共进化；③ 引入长度多样性奖励 R_div，有效防止已掌握任务的轨迹崩塌；④ 结合两阶段探索‑利用策略再训练，进一步突破监督上限。

**🔧 技术方法**

使用的技术包括：共享权重的生成/验证器、置信度阈值分区（high/medium/low）、Group Relative Policy Optimization (GRPO) 的优势估计、Pass@k 组合优势、长度多样性奖励 R_div、非对称软奖励矩阵、两阶段策略退火。

**📊 数据集**

实验数据集涵盖六个推理基准：AIME24、MATH500、AIME25、AMC、GPQA、DAPO；在 Qwen3‑4B/8B 两大模型上评估。

**📈 对比分析**

与无标签基线 TTRL、SCRL、Co‑Rewarding 以及监督上限 GRPO 对比；TTRL‑CoCoV 在 Pass@1 上平均提升 9.8%、Pass@16 上提升 18.7%，部分任务甚至超过 GRPO 上限，显示出显著性能提升。

**⚠️ 局限性**

局限性包括：依赖置信度阈值设定，阈值可能需要手动调优；验证器与生成器共享权重，可能限制模型的多任务表现；仅在推理类基准上验证，尚未验证在其他领域的适用性；对大模型的计算成本较高。

---

## 509. UnsOcc: 3D Semantic Occupancy Prediction in Unstructured Scene via Rendering Fusion

**arXiv ID:** 2606.03581 | [PDF](https://arxiv.org/pdf/2606.03581v1)

**作者:** Ye Wu `[一作]` (University of Chinese Academy of Sciences), Yunfeng Ai `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 2775 | [OpenAlex ID](https://openalex.org/A5022489675)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 UnsOcc 多模态 3D 语义占用预测框架，用于解决无结构场景（如露天矿）中的稀疏与长尾问题。

**💡 创新点**

创新点：① RenderFusion——基于渲染的双向监督，实现图像与 LiDAR 特征的对齐；② GSRefinement——利用 3D 高斯渲染生成 2D 语义监督，显著提升稀有类别识别；③ 同时构建了专门的无结构露天矿数据集。

**🔧 技术方法**

技术手段包括 3D Gaussian Splatting、深度渲染与语义渲染双向监督、稀疏卷积、体素化、ResNet+SECONDFPN 视觉特征提取以及点云特征编码与融合。

**📊 数据集**

使用的数据集：1）自建露天矿无结构场景数据集（135 条序列，约 55 帧/序列，13 类语义）；2）公开 nuScenes 3D 占用数据集。

**📈 对比分析**

与单模/多模基线（MonoScene、OccFormer、Co-Occ 等）以及 nuScenes 上的最新方法比较，矿山数据集上 mIoU 提升 30%+，长尾类 mIoU 提升 1.45；在 nuScenes 上显著高于现有方法，尤其在细粒度类别（交通锥、行人）表现优异；GSRefinement 与 RenderFusion 的组合进一步提升性能。

**⚠️ 局限性**

局限性：对极端天气/低光环境下的鲁棒性仍待验证；需要大量标注数据和高算力支持；在完全无 LiDAR 反馈的区域仍存在预测不确定性。

---

## 510. Eliciting Complex Spatial Reasoning in MLLMs through Wide-Baseline Matching

**arXiv ID:** 2606.03577 | [PDF](https://arxiv.org/pdf/2606.03577v1)

**作者:** Hao Zhong `[一作]` (Zhejiang University), Chunhua Shen `[通讯]` (Zhejiang University)

**通讯引用:** 70775 | [OpenAlex ID](https://openalex.org/A5006294869)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了ReasonMatch‑Bench评估宽基线匹配的新基准，并设计了可扩展的视频‑3D数据生成管线；基于可验证奖励的动态对应强化学习（DCRL）训练框架，显著提升MLLM在宽基线跨视角匹配上的性能。

**💡 创新点**

创新点包括：①ReasonMatch‑Bench按视角偏移和匹配粒度分层，覆盖室内、室外和对象级场景；②DCRL在图像层面采用视角进展、点层面采用对应数量与空间分布的双层课程，配合可验证奖励实现无监督的空间推理提升；③通过RGB‑D与SfM重建自动提取精确对应，构建大规模可验证训练样本。

**🔧 技术方法**

采用了强化学习（RLVR）与GRPO、动态对应强化学习（DCRL）、视角与对应课程学习；利用多模态大型语言模型进行语言驱动的点对应推理；在训练中使用可验证的对应奖励和格式合规奖励。

**📊 数据集**

使用了多源RGB‑D数据集（ScanNet、uCO3D、DL3DV）和RGB视频与SfM重建（RealEstate10k、COLMAP）生成对应；构建了2,810对的ReasonMatch‑Bench样本，并对训练集进行了视角重叠划分与点集采样。

**📈 对比分析**

在ReasonMatch‑Bench上与GPT‑5‑mini、Gemini‑2.5‑Pro、Qwen3‑VL‑8B等公开/闭源模型对比，DCRL在整体F1上达到70.5%，远超最佳基线37.2%，并在难度最高的90样本子集上提升至52.0（人类84.0）。此外，DCRL在OmniSpatial (+5.27%)、MindCube (+3.51%)等相关空间基准上实现正向迁移，并保持对通用视觉理解任务的性能。

**⚠️ 局限性**

局限性包括：仍与人类表现相距较远，尤其在对象级宽基线匹配上；对极端遮挡与细粒度对应的鲁棒性不足；RL训练对样本与计算成本高；数据生成管线在多样性与噪声控制方面仍有限，可能影响模型的普适性。

---

## 511. Cost of Manipulation in AMM-Based Oracles

**arXiv ID:** 2606.03548 | [PDF](https://arxiv.org/pdf/2606.03548v1)

**作者:** Sebastian Müller `[一作]` (Aix Marseille University), Adel Messaoudi `[通讯]` (Aix Marseille University)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文构建了一个关于AMM（常数乘积市场制造商）基于链价格预言机的操纵成本度量，并推导出在不同聚合规则和跨池套利条件下的最小操纵成本表达式。

**💡 创新点**

创新点在于：①给出了单池和多池情形下的闭式操纵成本公式；②在多池加权均值和加权中位数的攻击者–设计者博弈中，证明了流动性权重在所有或小幅失真下为最大化最小成本的最优选择；③通过阈值t=3揭示均值聚合在大失真下易被集中攻击；④扩展到星型多资产架构，并提出可与停留时间、费率限制结合的动态安全性评估。

**🔧 技术方法**

使用的技术主要是：CFMM（常数乘积市场制造商）微观结构分析、优化理论（Lagrange乘数、凸性分析）、博弈论（最大化最小化攻击成本）、闭式解析和组合优化（中位数攻击的集合覆盖问题）。

**📊 数据集**

本文未使用实测数据集，而是基于理论模型和假设的市场有效价格，提供纯理论分析和闭式公式。

**📈 对比分析**

与传统的TWAP、链上外部数据预言机等做法相比，本文提供了基于深度的定量安全基准；在理论层面证明了流动性权重聚合在多池环境下能显著提升操纵成本，尤其在无跨池套利时可达到最优；在有跨池套利时，任何对称聚合器的成本均简化为总深度乘以单池因子，表现出更高的鲁棒性。

**⚠️ 局限性**

局限性包括：①仅考虑常数乘积AMM，对集中式流动性（如Uniswap v3）未给出完整模型；②忽略了网络延迟、交易费用和可实现的套利路径的实际成本；③动态模型（如停留时间、速率限制）仅给出初步框架，缺乏完整的实验验证；④在大失真下加权均值的最优权重不存在统一解，需针对具体目标失真调优。

---

## 512. Competitive Information Design in Sequential Search

**arXiv ID:** 2606.03527 | [PDF](https://arxiv.org/pdf/2606.03527v1)

**作者:** Zhicheng Du `[一作]` (Renmin University of China), Zihe Wang `[通讯]` (Renmin University of China)

**通讯引用:** 434 | [OpenAlex ID](https://openalex.org/A5064179488)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

本文研究了广告商在面临搜索成本下，通过信息披露竞争以影响消费者的顺序搜索决策，并在竞争信息设计框架下分析了最佳反应与均衡结构；

**💡 创新点**

创新点在于把传统的单个卖家信息设计扩展到多卖家竞争环境，并给出可验证最佳反应的对偶方法、证明存在性与对称均衡的精细结构（如全披露、铰接凸池化策略）；

**🔧 技术方法**

采用了Weitzman指数算法的索引搜索、双重线性规划与补偿松弛条件、无限维LP的对偶理论、以及通过微分方程构造可实现的二维分布；

**📊 数据集**

论文未使用任何外部数据集，所有结果均为理论分析与闭式证明；

**📈 对比分析**

通过与传统单卖家信息传递模型及信息混淆模型进行对比，证明信息公开可提升买方收益、卖方收益受成本影响且在不同成本下出现不同策略；

**⚠️ 局限性**

局限性包括：仅对连续无质量点的先验有效；高维求解复杂；对非凸先验或多卖家异质成本的均衡仅给出部分结果，未来工作需进一步探讨效率与实际广告设计限制。

---

## 513. High-Precision APT Malware Attribution with Out-of-Scope Resilience

**arXiv ID:** 2606.03523 | [PDF](https://arxiv.org/pdf/2606.03523v1)

**作者:** Peter Williams `[一作]` (University of Southampton), Erisa Karafili `[通讯]` (University of Southampton)

**通讯引用:** 588 | [OpenAlex ID](https://openalex.org/A5091676155)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于每个APT组训练的二分类器并按验证性能排序后顺序应用的高精度APT恶意软件归因方法，允许在无法确定归因时显式拒绝（abstain）

**💡 创新点**

创新点在于：1) 将归因视为可选择分类问题，使用单独的二分类器而非闭集多分类；2) 通过排序与阈值控制实现对未知组样本的自动拒绝，显著提升了对out‑of‑scope样本的鲁棒性；3) 在同一数据集上优于已发表的闭集模型，且在极端out‑of‑scope场景下仍保持高精度

**🔧 技术方法**

技术包括：静态特征提取（使用ADAPT特征提取器）；多种树基模型（Hist Gradient Boosting、Extra Trees、XGBoost、CatBoost、Random Forest）；针对每个APT组的精度或F1调优的二分类器；按验证精度与样本数对二分类器进行排序；顺序应用并根据阈值决定是否归因；引入选择性分类评估指标（coverage、selective accuracy、selective F1 等）

**📊 数据集**

使用的主要数据集是APT Malware数据集（12个APT组，样本数不均），并与ADAPT Group‑labelled数据集（共72个APT组）合并，用于大规模out‑of‑scope测试；所有特征均通过ADAPT静态特征提取器获得

**📈 对比分析**

通过与基线多分类模型（Hist Gradient Boosting tuned for precision）以及先前文献中的结果进行对比，排名二分类器在保持相近或略优的整体F1的同时，精度提升至96–98%（对已知组）并在87% out‑of‑scope样本的极端测试中拒绝率达到94%，对已归因样本的精度为92%，选择性准确率为95%，体现了显著的性能提升

**⚠️ 局限性**

局限性包括：1) 受限于公开APT样本量不足，实验规模有限；2) 仅评估了ADAPT静态特征提取，缺乏对动态或混合特征的验证；3) 目前无大量可直接比较的out‑of‑scope研究，难以全面评估泛化能力；4) 阈值设定和排序策略依赖验证集，可能在不同数据集上需要重新调优

---

## 514. Testing LLM Arithmetic Reasoning Generalization with Automatic Numeric-Remapping Attacks

**arXiv ID:** 2606.03606 | [PDF](https://arxiv.org/pdf/2606.03606v1)

**作者:** Malia Barker `[一作]` (Boise State University), Francesco Gullo `[通讯]` (University of L'Aquila)

**通讯引用:** 2051 | [OpenAlex ID](https://openalex.org/A5026420819)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种数值重映射攻击，用于生成保持推理程序不变的算术问题变体。

**💡 创新点**

创新点在于自动化生成符号表示、约束提取、数值重映射、表面编辑计划，并通过高置信度审核确保攻击有效性。

**🔧 技术方法**

使用大语言模型（GPT-OSS、DeepSeek-R1、Gemma4）进行符号推理、约束生成、编辑计划生成，结合结构化验证和后审计。

**📊 数据集**

使用GSM8K、MAWPS、MultiArith三大算术词问题数据集。

**📈 对比分析**

对原始数据集和生成的攻击集分别评估模型准确率；在GSM8K上攻击导致显著降幅（12.16-25.82个百分点），MAWPS和MultiArith降幅极小。

**⚠️ 局限性**

局限包括仅测试数值重映射、受限于源模型正确性与生成成功、保守的高置信度过滤导致覆盖率下降、验证主要为自动化未完全手工检查、对其他攻击类型与更广泛领域未扩展。

---

## 515. PHASER: Phase-Aware and Semantic Experience Replay for Vision-Language-Action Models

**arXiv ID:** 2606.03598 | [PDF](https://arxiv.org/pdf/2606.03598v1)

**作者:** Ziyang Chen `[一作]` (HKUST(Guangzhou)), Yandong Guo `[通讯]` (AI2 Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PHASER框架，通过阶段级容量分配和多模态干扰路由在VLA持续学习中显著缓解phase starvation与遗忘问题。

**💡 创新点**

创新点在于把记忆单元从帧切换到子技能阶段，保证每个阶段获得等量记忆，并用三模态原型距离和Boltzmann采样动态优先回放高风险历史阶段。

**🔧 技术方法**

技术手段包括SMDP建模、三模态原型评分、Boltzmann分布路由、无前向成本的经验回放以及Auto-PC自动阶段检测管线。

**📊 数据集**

主要在LIBERO-Goal和LIBERO-Long两套连续学习任务上进行实验。

**📈 对比分析**

与Sequential FT、ER、MIR、iCaRL等基线对比，PHASER在匹配预算下平均成功率提升至87.8%，相较ER提升31%，NBT下降至7.8%，表现最优。

**⚠️ 局限性**

局限在于阶段抽取依赖VLM验证且未采用高级关键帧选择策略，Auto-PC可能因语义误判导致阶段划分误差。

---

## 516. CANMOT: Class-Aware Noise Modeling for Multi-Object Tracking in Autonomous Driving

**arXiv ID:** 2606.03590 | [PDF](https://arxiv.org/pdf/2606.03590v1)

**作者:** Timo Osterburg `[一作]` (TU Dortmund University), Torsten Bertram `[通讯]` (TU Dortmund University)

**通讯引用:** 4046 | [OpenAlex ID](https://openalex.org/A5051394730)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `aaccfe5c-6b26-4208-b23c-35331481e142` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CANMOT方法，在Kalman滤波基础上使用类别感知、对象坐标系对齐的噪声建模，提升多目标跟踪精度并改善不确定性校准。

**💡 创新点**

创新点包括①在对象坐标系下为每个类别定义对角过程噪声和测量噪声，保留纵向-横向异方差；②通过贝叶斯优化为每类独立学习噪声参数；③系统评估并揭示现有基线的严重过度自信。

**🔧 技术方法**

使用Kalman滤波、3D GIoU关联、贝叶斯优化、ANEES与chi^2一致性测试、CenterPoint检测器及nuScenes基准。

**📊 数据集**

使用nuScenes数据集（train/val/test）以及CenterPoint产生的检测结果。

**📈 对比分析**

与Poly-MOT、IMM-MOT、MCTrack等SOTA基线对比，CANMOT在AMOTA与SOTA相当或略优（+0.7pp以上），显著降低ID Switches（约19%），并在不确定性一致性指标上显著提升（ANEES更接近理论值）。

**⚠️ 局限性**

仍未实现完全统计一致性；仅采用对角协方差限制了建模灵活性；仅使用单一运动模型（CV）；贝叶斯优化计算成本高，且调参仅在验证集完成，泛化性待进一步验证。

---

## 517. Skill Is Not Document: A Query-Conditional Benchmark and Two-Stage Retriever for LLM Agent Skill Routing

**arXiv ID:** 2606.03565 | [PDF](https://arxiv.org/pdf/2606.03565v1)

**作者:** Zifei Wang `[一作]` (Tencent), Xing Sun `[通讯]` (Tencent)

**通讯引用:** 2910 | [OpenAlex ID](https://openalex.org/A5004402130)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了考虑技能兼容性的检索框架，并通过保留LLM拒绝样本来提升多技能检索效果。

**💡 创新点**

创新点在于：①将LLM拒绝标记作为负样本用于训练兼容性；②构建了四种语言方向的双语技能检索基准R3‑Skill；③设计两阶段检索（R3‑Embedding + R3‑Reranker），在交叉编码器阶段引入兼容性标签并使用分级ListNet学习。

**🔧 技术方法**

使用的技术包括：多语言BGE‑M3或Qwen3基础模型的双编码器与交叉编码器；InfoNCE对比学习；分级ListNet跨编码器排序；大规模LLM (DeepSeek‑V4‑Pro、Qwen3‑235B‑A22B) 进行写/跳注释。

**📊 数据集**

使用的数据集为R3‑Skill（10,246技能、41,592查询、32,828拒绝注释），涵盖四个语言方向（en2en、en2zh、zh2en、zh2zh）并通过专家验证；与SkillRouter、SkillRet等公开基准做对比。

**📈 对比分析**

在R3‑Skill测试集上，R3‑Embedding取得Hit@1=0.7024、NDCG@10=0.8064；加入R3‑Reranker后提升到Hit@1=0.7714、NDCG@10=0.8327、Set‑Compat=0.3525；在SkillRet官方测试集上，R3‑Reranker达到NDCG@10≈82.9，接近公开基准。

**⚠️ 局限性**

限制包括：中文技能样本稀缺；兼容性判定依赖LLM，可能存在不一致；仅在离线检索评估，未验证真实系统中的用户体验与业务指标；检索候选集固定为20，未探索更大候选池对兼容性的影响。

---

## 518. Analytical Evaluation of DCA Convergence Properties for Minimizing Prediction Functions of Gaussian RBF Support Vector Regression

**arXiv ID:** 2606.03559 | [PDF](https://arxiv.org/pdf/2606.03559v1)

**作者:** Yohei Kakimoto `[一作]` (Nihon University), Hirotaka Takahashi `[通讯]` (Tokyo City University)

**通讯引用:** 9319 | [OpenAlex ID](https://openalex.org/A5049271542)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了在支持向量回归（RBF‑SVR）预测函数上使用DC算法（DCA）的优化问题，提供了显式的DC分解并推导了强凸性参数与梯度Lipschitz常数的闭式表达式。

**💡 创新点**

创新点在于利用高斯RBF核的Hessian结构，得到单一量C_{α}ρ来决定DCA的收敛速度与初值依赖，并展示该量可在训练前通过超参数（C,γ）近似预测。

**🔧 技术方法**

使用的技术包括DC编程、DCA、Frank–Wolfe（DC‑FW）理论、RBF核特征分析以及对Hessian的Rayleigh商界定。

**📊 数据集**

实验数据集为六种常用的二维基准函数（Branin、Himmelblau、Six‑hump Camel、Rastrigin、Ackley、Levy），在每个函数上以300个带高斯噪声的样本训练RBF‑SVR。

**📈 对比分析**

通过对迭代次数、初值敏感度与归一化残差的统计比较，结果表明C_{α}ρ与收敛表现高度相关，实验验证了理论推导并说明收敛速率近似线性。

**⚠️ 局限性**

局限性包括仅在二维场景验证、ρ取近似最小值导致强凸性界限极小、DCA仅收敛到临界点且未评估全局最优性、以及未直接测试DC‑FW的实际性能。

---

## 519. NVIDIA Isaac Sim: Enabling Scalable, GPU-Accelerated Simulation for Robotics

**arXiv ID:** 2606.03551 | [PDF](https://arxiv.org/pdf/2606.03551v1)

**作者:** Sicong Gao `[一作]` (University of New South Wales), Yang Song `[通讯]` (University of New South Wales)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文对NVIDIA Isaac Sim进行了系统性综述，梳理了其架构、使用模式、典型应用以及局限性；

**💡 创新点**

首次提供了针对Isaac Sim的全面、结构化评述，并对其在数据生成、数字孪生、强化学习等方面的独特价值进行归纳；

**🔧 技术方法**

采用文献综述、对比表格、案例分析等方法，对比了Isaac Sim与Gazebo、MuJoCo、CARLA等主流仿真平台的技术特征；

**📊 数据集**

在讨论中引用了众多基于Isaac Sim的数据生成工作，如Replicator合成数据集、MobilityGen轨迹数据、工业与医疗场景等，但并未直接使用特定公开数据集；

**📈 对比分析**

通过对比表格展示Isaac Sim在物理引擎（PhysX GPU）、光线追踪渲染、并行多环境、数字孪生与ROS集成等方面优于传统仿真器，性能表现突出，支持大规模并行仿真和高保真感知数据；

**⚠️ 局限性**

主要局限在于使用门槛高、学习曲线陡峭、对高端GPU和硬件资源要求严格，导致在资源受限或初学者场景中的可行性受限。

---

## 520. Static and Dynamic Representations for Tactile Contact-Angle Estimation with Event-Based Sensors

**arXiv ID:** 2606.03545 | [PDF](https://arxiv.org/pdf/2606.03545v1)

**作者:** Yanhui Lu `[一作]` (University of Bristol), Benjamin Ward-Cherrier `[通讯]` (University of Bristol)

**通讯引用:** 1004 | [OpenAlex ID](https://openalex.org/A5008451819)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了基于事件式触觉传感器NeuroTac的接触角估计，提出了静态、动态和组合三种事件表示管线。

**💡 创新点**

创新点在于用模拟人类慢适应与快适应受体的思路分别设计静态与动态管线，并系统比较它们在连续滚动与随机暂停情境下的估计性能。

**🔧 技术方法**

采用事件窗口切片、指数衰减累加、像素二值化、3层全连接网络（SmoothL1损失、AdamW优化）等技术实现低延迟的角度回归。

**📊 数据集**

使用在ABB 6DOF机器人上搭载NeuroTac采集的滚动实验数据，涵盖多速度、多深度以及随机暂停，形成训练/验证/测试集。

**📈 对比分析**

通过对三种表示进行超参数网格搜索并在同一数据集上10次训练进行比较，静态表示连续滚动时平均MAE为0.160°，动态为0.186°，组合为0.173°；在暂停场景中静态仍优，MAE分别为0.211°/0.251°。

**⚠️ 局限性**

局限性包括只验证单方向倾斜、有限速度/深度范围；组合方式无显著提升；动态表示对低事件率敏感；未考虑倾斜方向与幅度同时变化；缺乏更细粒度的通道分离与学习式融合。

---

## 521. D2MDT: Department-aware Multidisciplinary Team Consultation with Deliberation for Efficient Clinical Prediction

**arXiv ID:** 2606.03543 | [PDF](https://arxiv.org/pdf/2606.03543v1)

**作者:** Yongqi Liang `[一作]` (Xi'an Jiaotong University), Chen Li `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 30838 | [OpenAlex ID](https://openalex.org/A5100379155)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了一种基于科室意识的多学科团队协作与递归推理框架，用于提升电子健康记录（EHR）中的临床预测效果。

**💡 创新点**

创新点在于将科室专业视角融入多智能体协同，并通过残差推理仅更新未达成共识的部分，既增强了证据多样性，又显著提升了交互效率。

**🔧 技术方法**

使用多专家EHR编码器、SHAP解释、检索增强的LLM代理、残差推理机制以及多模态融合预测网络等技术。

**📊 数据集**

在MIMIC‑III和MIMIC‑IV两大ICU数据库上进行死亡率预测实验。

**📈 对比分析**

与传统深度学习、LLM协同及现有MDT基线相比，该方法在AUPRC/AUROC上取得了最高或相近优异的性能，并在多轮讨论中显著降低了prompt token消耗。

**⚠️ 局限性**

主要限制包括仅在ICU死亡预测任务验证、对LLM与检索库的依赖、残差压缩可能导致信息遗漏，以及缺乏临床实战评估。

---

## 522. Knowledge-Preserved Model Tuning in Null-Space for Robust Spatio-Temporal Video Grounding

**arXiv ID:** 2606.03539 | [PDF](https://arxiv.org/pdf/2606.03539v1)

**作者:** Haoxuan Chen `[一作]` (Sun Yat-sen University), Jian-Fang Hu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 1941 | [OpenAlex ID](https://openalex.org/A5102336058)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Null‑Space Tuning (NST)，在视频时空定位任务中对低质量视频进行恢复，同时保留高质量视频的预训练知识。

**💡 创新点**

创新点在于利用冻结权重的零空间与行空间进行几何分离，针对输入质量动态路由恢复信号与噪声，避免了传统 PEFT 的“全局注入”导致的灾难性遗忘。

**🔧 技术方法**

技术主要包括：质量自适应单元（QAU）利用多模态参考库生成恢复系数；双空间重参数化通过 SVD 分解权重，将恢复信号投射到行空间，噪声投射到零空间；路由器与正则化损失实现输入质量感知的自适应调度。

**📊 数据集**

使用了 HCSTVG‑v1/v2、VidSTG 基准，并在其上构造低质量版本（运动模糊、散焦、低照度、分辨率下降）生成 Mixed‑Quality Benchmark。

**📈 对比分析**

与完整微调以及 LoRA、AdaLoRA、DoRA、FlyLoRA 等主流 PEFT 方法对比；在 Mixed‑Quality 集合上 NST 在 m_tIoU、m_vIoU 及 vIoU@0.5 等指标上均超过所有对手，并且通过 Weighted Adaptation Score（WAS）显示在恢复低质量时几乎无灾难性遗忘。

**⚠️ 局限性**

局限性包括：主要适用于线性投影层，依赖冻结权重中足够的几何冗余；对非常低质量或极端失真的视频恢复效果仍有限；实现中需要预先计算并缓存 SVD 基，略微增加存储和预处理成本。

---

## 523. Towards Non-Monotonic Entailment in Propositional Defeasible Standpoint Logic

**arXiv ID:** 2606.03655 | [PDF](https://arxiv.org/pdf/2606.03655v1)

**作者:** Nicholas Leisegang `[一作]` (University of Cape Town and CAIR), Ivan Varzniczak `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

在论文中作者提出了一套方法，将经典KLM（Kraus–Lehmann–Magidor）框架中的理性与词典化闭包（rational closure 与 lexicographic closure）等非单调推理机制迁移到命题可抵触立场逻辑（PDSL）的一个可表达子语言中，并通过“立场定位条件”（situated standpoint conditionals）重新表述 PDSL 的语法与语义，构造从 PDSL 知识库到传统命题条件知识库的线性翻译，从而能够在 PDSL 中实现非单调推理。

**💡 创新点**

创新点包括：①引入立场定位条件，显著提升 PDSL 的表达能力；②证明 PDSL 的一个子语言 Cond() 可以被映射到标准 KLM 条件知识库；③通过该映射将理性与词典化闭包等成熟的非单调推理方法迁移至 PDSL；④保持推理复杂度与命题 KLM 相同，展示了方法的可扩展性与实用性。

**🔧 技术方法**

技术手段主要是：
- 基于优先语义（preferential semantics）和排名函数（ranking functions）的理论框架；
- 设计线性时间的翻译函数 T，将 PDSL 语句映射为命题条件；
- 利用已有的理性闭包与词典化闭包算法（如 P^NP_∥-complete 与 P^NP-complete 的实现）；
- 在 PDSL 上定义非单调蕴含关系 _r，并用 SAT 解决器验证翻译后模型的满足性。

**📊 数据集**

本研究为理论性质证明，未使用任何外部数据集；所有结果均在形式化模型与算法层面得到验证。

**📈 对比分析**

通过与原始单调 Tarskian 蕴含（_P）的比较，证明新的非单调蕴含在 PDSL 上更强；同时，利用命题 KLM 的已知复杂度结果，展示在 PDSL 上的理性闭包和词典化闭包推理仍分别保持 P^NP_∥-complete 与 P^NP-complete 的时间复杂度。

**⚠️ 局限性**

局限性包括：
- 只覆盖 PDSL 的 Cond() 子语言，缺乏对全语言中否定、非布尔子句和某些可分辨性操作的支持；
- 翻译依赖于立场符号被视为命题原子，可能无法捕捉更细粒度的立场结构；
- 对立场视角的非单调钻石命题（diamond）采用“勇敢推理”方式，可能与某些应用场景期望的保守推理不符；
- 目前未包含实验评估或与其他立场逻辑实现的对比，仅停留在形式化证明层面。

---

## 524. Black-box, Adaptive, Efficient, Transferable, Harmful, Applicable... Attacks Are All You Need to Break LLMs

**arXiv ID:** 2606.03647 | [PDF](https://arxiv.org/pdf/2606.03647v1)

**作者:** Vincent Limbach `[一作]` (Technical University of Munich), Leo Schwinn `[通讯]` (Technical University of Munich)

**通讯引用:** 221 | [OpenAlex ID](https://openalex.org/A5028233502)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一种基于掩码扩散语言模型的攻击框架 IHO，能够在黑盒条件下自动生成针对 LLM 的 jailbreak 并适配多种防御

**💡 创新点**

通过将攻击者模型化为掩码扩散模型，并使用直接偏好优化（DPO）来间接优化有害性，实现了可迁移、适应性强且不需要手工模板的统一攻击

**🔧 技术方法**

掩码扩散语言模型（LLaDA）+ LoRA 微调、Direct Preference Optimization (DPO)、强拒绝判别器

**📊 数据集**

JailbreakBench 100 违规行为、Llama3-8B-Instruct 与 Qwen2.5-7B/32B 指令模型，以及 Circuit Breaker、Latent Adversarial Training (LAT)、Continuous Adversarial Training (CAT) 等防御

**📈 对比分析**

使用 EVUS 指标对比训练、未见行为及跨模型迁移场景下的攻击效果，IHO 在所有模型/防御组合中均显著优于现有基线，尤其在强防御模型上提升显著

**⚠️ 局限性**

跨模型迁移在防御差异较大时效果下降；攻击效果依赖判别器质量，可能出现 judge hacking 影响

---

## 525. Construction of cyclic codes with large minimum distance from power functions over odd characteristic finite fields

**arXiv ID:** 2606.03638 | [PDF](https://arxiv.org/pdf/2606.03638v1)

**作者:** Mrinal Kanti Bose `[一作]` (Indian Institute of Technology (ISM)), Abhay Kumar Singh `[通讯]` (Indian Institute of Technology (ISM))

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

构造了一系列基于奇特征有限域上已知差分均匀性幂函数的q-ary循环码，并求得其线性跨度、维数与生成多项式；

**💡 创新点**

利用非二元有限域的幂函数差分均匀性，得到维数大于$(q^m-1)/2$且最小距离超越码长平方根的无限族循环码，并解决了Ding提出的开放问题；

**🔧 技术方法**

采用差分均匀性分析、符号展开、线性跨度计算、p-循环余数与p-赛义多项式、BCH与Hartmann–Tzeng界等理论工具；

**📊 数据集**

无专门数据集，全部通过理论推导与Magma等计算机代数系统验证；

**📈 对比分析**

与传统BCH/原始BCH码对比，部分族码在维数和最小距离上表现更优，且在一定条件下可达到最优或接近最优；

**⚠️ 局限性**

局限性在于需要已知差分均匀性的幂函数，计算复杂度随码长指数增长，难以验证极大码长；仅适用于奇素数域，且部分码的最小距离仍只能给出下界。

---

## 526. TSQAgent: Rating Time Series Data Quality via Dedicated Agentic Reasoning

**arXiv ID:** 2606.03629 | [PDF](https://arxiv.org/pdf/2606.03629v1)

**作者:** Shunyu Wu `[一作]` (Sun Yat-sen University), See-Kiong Ng `[通讯]` (National University of Singapore)

**通讯引用:** 5527 | [OpenAlex ID](https://openalex.org/A5090171111)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了针对时间序列质量的专用基准 TSQBench，并提出了 TSQAgent 代理推理框架，利用 LLM 自动识别关键质量维度并进行定量比较，从而实现可解释且高精度的时间序列质量评估与数据选择。

**💡 创新点**

通过将强化学习（GRPO）驱动的自适应维度选择与 ReAct 交互式工具增强推理相结合，使 LLM 能主动聚焦重要维度并获取可量化证据，显著提升质量评估的可解释性和准确率。

**🔧 技术方法**

采用大语言模型（Qwen、Gemma、GPT‑4o‑mini 等）为核心，结合 GRPO 训练、16 种统计/频谱/结构工具、ReAct 推理流程和 Bradley–Terry 评分转换等技术实现自动化评估。

**📊 数据集**

使用 TSQBench 生成的 1,000 条合成样本以及 11 个真实数据集（Electricity、ExchangeRate、Traffic、Weather、M4（Yearly、Monthly、Daily）、UCR/UEA 分类数据如 MedicalImages、CBF、BME、Handwriting）以及 Timer‑S1 8.3B 基础模型的混合数据。

**📈 对比分析**

与 DataShapley、DataOob、TimeInf、TSRating 等基线方法在数据选择任务中进行对比；在 50% 预算下，TSQAgent 能实现与全量训练相当的 RMSE/MAPE/准确率，工具增强后质量比较准确率提升至 78–84% 以上，整体性能显著优于传统基线。

**⚠️ 局限性**

LMM 在维度选择方面仍易出现冗余或缺失，定量推理高度依赖工具，跨域分布鲁棒性有限，并且需要额外的算力与工具调用来保证推理质量。

---

## 527. Bridging Auxiliary Constraints to Resolve Instruction Following in Large Reasoning Models

**arXiv ID:** 2606.03624 | [PDF](https://arxiv.org/pdf/2606.03624v1)

**作者:** Zhengyi Zhao `[一作]` (Chinese University of Hong Kong), Xian Wu `[通讯]` (Tencent Jarvis Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于约束关系图完成（CRGC）的框架，用于主动识别和解决大型推理模型在多约束指令下的违规与冲突问题。

**💡 创新点**

创新点在于：①将指令拆解为约束节点并构建有向关系图；②使用最小生成树（MSA）优化约束路径；③自动生成“桥约束”以桥接冲突或孤立约束，从而提升约束遵循率。

**🔧 技术方法**

核心技术包括：约束提取、权重估计（利用条件期望）、最小生成树算法、桥约束生成（利用LLM生成中介指令）以及无模型参数更新的推理增强。

**📊 数据集**

实验数据集：IFEval、ComplexBench、FollowBench 三个指令遵循基准；MMLU、GSM8K、BIG-Bench Hard 三个通用推理基准。

**📈 对比分析**

与标准提示、Chain‑of‑Thought、Self‑Reflection、COP 等方法对比，CRGC 在指令遵循的约束满足率（CSR）提升约 39%（标准提示）或 27.7%（CoT），且保持或略提升推理准确率，平均约束满足率和任务完成率均显著高于基线，且多轮迭代消耗显著减少。

**⚠️ 局限性**

局限性：①需要额外计算开销（图构建与权重估计）导致单轮延迟上升；②桥约束生成依赖LLM，若模型本身对桥约束理解不足可能引入新冲突；③在极端约束数量大或关系复杂度高时，MSA 选取可能不够稳定；④目前仅在公开基准验证，实际工业场景中的非结构化约束适用性待进一步评估。

---

## 528. SA-DTS: Semantic-Aware Digital Twin Synchronization over 6G Networks

**arXiv ID:** 2606.03617 | [PDF](https://arxiv.org/pdf/2606.03617v1)

**作者:** Vincenzo Sammartino `[一作]` (University of Pisa), Vincenzo Sammartino `[通讯]` (University of Pisa)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5095698377)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并验证了 SA‑DTS 框架，实现数字孪生同步的语义编码与知识图谱重建，显著降低带宽、延迟与能耗。

**💡 创新点**

创新点在于：① 将多模态语义编码与动态知识图谱融合以恢复完整状态；② 采用层级分区知识图谱实现 O(N log N) 的可扩展性；③ 提出了语义完整性评分（SFS）作为任务级评价指标。

**🔧 技术方法**

使用技术包括多模态语义编码器、联合源‑信道编码、PPO 强化学习自适应码率、图神经网络 KG‑CR、MINE 信息瓶颈、端到端训练。

**📊 数据集**

实验数据集涵盖工业机器人（RGB‑D/IMU）、远程患者监测（PhysioNet MIMIC‑III ECG/SpO₂/深度摄像头）、车队跟车（KITTI Odometry LiDAR/GPS/IMU）等多模态传感信息。

**📈 对比分析**

与 Raw‑DTS、JSCC‑DTS、SemCom‑DTS、NTSCC‑DTS、KG‑Only 等基线对比；在 15 dB 信噪比下实现 94% 带宽节省、87% 延迟降低、12.7 倍能耗下降；SFS 与任务指标相关系数>0.97，验证其可靠性。

**⚠️ 局限性**

局限性包括：对抗鲁棒性不足、语义互操作性挑战、隐私保护需求、KG 一致性与大规模在线适应的系统级难题，以及标准化与 6G 协议对齐仍待完善。

---

## 529. Don't Forget Your Embeddings: Robust Knowledge Erasure via Precise Editing of Embeddings

**arXiv ID:** 2606.03695 | [PDF](https://arxiv.org/pdf/2606.03695v1)

**作者:** Clara Haya Suslik `[一作]` (Tel Aviv University), Mor Geva `[通讯]` (Tel Aviv University)

**通讯引用:** 1628 | [OpenAlex ID](https://openalex.org/A5065717258)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EMBER，一种在词嵌入层上进行稀疏矩阵分解并精准去除与目标概念相关特征的概念消除方法，并将其与现有 MLP 层消除技术组合；

**💡 创新点**

创新点在于把概念消除从仅限 MLP 层扩展到嵌入层，并使用稀疏矩阵分解精确定位并删除嵌入特征，从而显著提升再学习鲁棒性和特异性；

**🔧 技术方法**

主要技术包括稀疏矩阵分解（Sparse MF）、概念特征筛选（质量比统计 + LLM 验证）、嵌入向量差分编辑以及与 RMU、CRISP、SNMF 等现有方法的组合；

**📊 数据集**

实验基于 Gemma‑2‑2B‑it 与 Llama‑3.1‑8B‑Instruct 两大 LLM，使用 18 个多样化概念（如 Harry Potter、World War II、Cannabis 等），从 Wikipedia 采集 300 条概念相关句子和 300 条中性句子，生成 100 条多选与开放式问题进行评估；

**📈 对比分析**

与 Mean、Noise、SNMF、RMU、CRISP 等基线及其 +EMBER 组合进行对比；结果显示 EMBER 单独即可将多选概念准确率下降 45% 以上，+EMBER 进一步降低 8–14 分；类似领域准确率提升 7–11 分，MMLU 维持稳定；在再学习测试中，RMU/CRISP 的恢复准确率从 70–76% 降至 47–52%，SNMF+ 仅恢复 6%；整体 Efficacy‑Utility 平均得分提升，Coherence 仅下降 0.05 分；

**⚠️ 局限性**

局限性包括：只针对英文 Wikipedia，缺乏多语言适用性；仅修改输入嵌入而不调整未嵌入矩阵，可能被恶意绕过；对极稀有概念嵌入可能无效；特征选择过程中存在噪声；以及对非 Transformer 或未绑定嵌入的模型效果未知。

---

## 530. Face versus Body Tracking for Human-Robot Interaction: An Egocentric Dataset

**arXiv ID:** 2606.03694 | [PDF](https://arxiv.org/pdf/2606.03694v1)

**作者:** Jessica Wenninger `[一作]` (Furhat Robotics), Gabriel Skantze `[通讯]` (Furhat Robotics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `aaccfe5c-6b26-4208-b23c-35331481e142` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了基于 Furhat 机器人收集的 egocentric HRI 数据集，并系统评估面部与身体跟踪的性能，优化时序记忆与 ReID 以显著降低身份切换；

**💡 创新点**

首次针对静止社交机器人提出专属 egocentric 数据集，系统化分离检测与跟踪误差，并证明在复杂社交动态下 ReID 对身体跟踪的巨大提升，面部跟踪仍受局限；

**🔧 技术方法**

采用 ByteTrack 与 BoT‑SORT / BoT‑FACE‑SORT 跟踪框架，结合 YOLOX、RetinaFace 检测器，禁用摄像机运动补偿与空间门限，使用 TrackEval 指标 HOTA、IDF1、IDSW 进行评估；

**📊 数据集**

使用自制 20 条 Furhat 机器人 egocentric 视频序列，涵盖 1–2 人交互与多种背景，标注身体与面部边界框；

**📈 对比分析**

通过 11 个实验配置（检测器、跟踪器、缓冲大小）在 GT 与检测版本上评估，Body ReID+长记忆（B‑GT‑BS‑2500）将 IDSW 降 68%（从 78 降至 25），面部 ReID 反而增加误差；最终实测 YOLOX+ReID+长记忆将 IDSW 从 78 降至 40，提升 HOTA 与 IDF1；

**⚠️ 局限性**

仅包含 20 条室内办公场景、固定摄像头、禁用 CMC、未考虑实时计算与外部光照变化，缺乏移动机器人或户外环境的验证；

---

## 531. Does Language Shift Break Medical Vision-Language Models? Indonesian Radiology Visual Question Answering Case Study

**arXiv ID:** 2606.03693 | [PDF](https://arxiv.org/pdf/2606.03693v1)

**作者:** Pieter Christy Yan Yudhistira `[一作]` (Brawijaya University), Novanto Yudistira `[通讯]` (Brawijaya University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了IndoRad‑VQA这一印尼语翻译版的医学视觉问答基准，用以评估医学VLM在非英语临床语言下的鲁棒性。

**💡 创新点**

创新点在于通过机器翻译与人工自评相结合构建双语翻译集，并提出语言鲁棒性间隙（LRG）指标与错误模式分类，揭示语言迁移对医学VLM性能的显著影响。

**🔧 技术方法**

主要技术包括基于Gemma‑4B‑IT的机器翻译、双语答案归一化词典、零样本Prompting、五种评估指标（严格准确率、归一化准确率、Tokenized F1、BERT‑Score、LRG）。

**📊 数据集**

使用VQA‑RAD作为原始英文数据集，翻译后得到2,248个问答对，覆盖CT、X‑ray、腹部影像三种模态。

**📈 对比分析**

在七个开源VLM（泛用、东南亚多语和医学专用）上做零样本评估，结果显示英语环境下的准确率均低于印尼语环境，平均严格准确率下降约19.8个百分点，说明模型在语言切换上存在显著性能损失。

**⚠️ 局限性**

限制包括仅在单一影像问答数据集上验证、翻译依赖单一模型（4B参数）、未做微调、缺乏专业放射科医生复核以及未评估临床决策支持可行性。

---

## 532. AUGUSTE: Online-Learning dApp for Predictive URLLC Scheduling

**arXiv ID:** 2606.03664 | [PDF](https://arxiv.org/pdf/2606.03664v1)

**作者:** Maxime Elkael `[一作]` (Northeastern University), Tommaso Melodia `[通讯]` (Northeastern University)

**通讯引用:** 20184 | [OpenAlex ID](https://openalex.org/A5054337759)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种基于机器学习的 MAC 调度框架，用于降低 5G URLLC 的延迟并保持低无线开销；

**💡 创新点**

创新点在于将 ML 模型嵌入上行调度器，并通过自适应状态机监督预测，以消除传统 SR 触发导致的延迟，同时仅增加约10%的无线开销；

**🔧 技术方法**

采用机器学习模型（如线性回归/可解释模型）和自适应状态机，并与 OAI（OpenAirInterface）无缝集成；

**📊 数据集**

使用 5G OAI 测试平台，实验了三种 URLLC 场景（请求‑响应、无人机报告等）的真实流量；

**📈 对比分析**

与传统 SR 基线及静态预留方案对比，实验表明中位 RTT 降低约 50%，一侧延迟从 ~13 ms 降至 7–10 ms，且无线开销仅提升 7–10%，形成明显的延迟‑开销 Pareto 前沿；

**⚠️ 局限性**

局限在于仅单 UE 单路由器实验，未验证多 UE 竞争与拥塞情况；模型对多周期/叠加周期流量的适应性有限；缺乏信道演化预测与多级优化的联合验证。

---

## 533. From Answers to States: Verifiable Process-Level Evaluation of Chemical Reasoning in Large Language Models

**arXiv ID:** 2606.03660 | [PDF](https://arxiv.org/pdf/2606.03660v1)

**作者:** Hongyu Guo `[一作]` (Peking University), Li Yuan `[通讯]` (Peking University)

**通讯引用:** 18530 | [OpenAlex ID](https://openalex.org/A5100700791)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了ChemCoTBench‑V2，一个通过专家设计模板和规则可验证的化学推理诊断基准，能够对化学大语言模型的中间推理过程进行细粒度评估。

**💡 创新点**

创新点在于将自然化学链式推理转化为结构化、可验证的中间承诺，分层评估最终答案、模板遵循度和逐步验证正确性，首次实现开放式化学任务的可追溯、低成本评估。

**🔧 技术方法**

采用规则可验证的化学推理模板、基于RDKit的符号检验、Type‑I/II 逻辑一致性判定，以及针对开放式优化的oracle约束检查。

**📊 数据集**

使用公开化学数据库（如ChEMBL、PubChem等）抽取5,620个样本，涵盖分子理解、编辑、优化与反应预测等18个报告任务。

**📈 对比分析**

与主流大模型（Qwen3.5、DeepSeek、GPT‑5、Gemini等）对比，发现模型在最终答案和模板遵循度上表现良好，但在逐步验证层（Layer‑3）常出现化学不一致，表明模型在多步状态追踪上存在显著瓶颈。

**⚠️ 局限性**

局限性包括仅覆盖二维分子/反应、缺乏三维构象、量子化学或实验操作规划，且基准依赖专家预设模板，无法完全捕捉所有可能的化学推理路径。

---

## 534. A Benchmark for Semi-supervised Multi-modal Crowd Counting

**arXiv ID:** 2606.03646 | [PDF](https://arxiv.org/pdf/2606.03646v1)

**作者:** Haoliang Meng `[一作]` (Harbin Institute of Technology), Wangmeng Zuo `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 63891 | [OpenAlex ID](https://openalex.org/A5100636655)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了首个半监督多模态人群计数基准，定义了标注与未标注数据的划分协议，并在此基准上系统评估了多种基线方法。

**💡 创新点**

创新点包括①设计了针对RGB‑热能组合的半监督学习协议；②将现有全监督多模态计数模型和单模半监督模型统一改造为可比的基线；③公开代码与数据划分，为后续研究提供了统一的实验平台。

**🔧 技术方法**

核心技术为Mean Teacher一致性框架、跨模态门控融合、特征级拼接以及传统的密度回归损失，辅以数据增强（色彩抖动、翻转、模糊）以提升未标注样本的利用率。

**📊 数据集**

使用了两大公开RGB‑热能计数数据集：RGBT‑CC和DroneRGBT，并在不同标注比例（5%、10%、40%）下划分训练/测试集。

**📈 对比分析**

方法对比包括全监督多模态基线及其Mean Teacher扩展、以及将单模半监督方法改造为双分支多模态模型；实验结果表明，在所有标注比例下，基于Mean Teacher的扩展普遍能降低GAME和RMSE，特别是在低标注比例（5%）时提升明显。

**⚠️ 局限性**

局限性在于：①依赖严格配准的RGB‑热能数据，未验证对其他模态或非对齐情况的适用性；②半监督扩展主要采用一致性学习，伪标签噪声仍可能导致误导；③实验仅覆盖两组数据集，缺乏更广泛的跨域或不同环境的评估；④对模态权重与融合策略的可解释性与调优仍待深入探讨。

---

## 535. Can AI be Easy? Lessons Learned from the EZR.py Toolkit

**arXiv ID:** 2606.03640 | [PDF](https://arxiv.org/pdf/2606.03640v1)

**作者:** Tim Menzies `[一作]` (North Carolina State University), Srinath Srinivasan `[通讯]` (North Carolina State University)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5009045242)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并实现了一个400行的轻量级Python工具包 EZR，用统一的核心子结构实现了朴素贝叶斯、k‑均值、k‑均值++、分类/回归树、模拟退火、局部搜索、主动学习以及互补贝叶斯文本过滤等多种 AI 方法。

**💡 创新点**

创新点在于通过仔细阅读和重构现有大型 AI 工具，发现并提炼出仅四个类（Num、Sym、Data、Cols）和一个更新原语即可支撑多种算法，实现了代码极度压缩、依赖极少、运行速度极快的统一工具包，并通过实验证明该工具在 SE 优化任务上可与甚至超越当前主流大规模库。

**🔧 技术方法**

使用的技术主要是增量统计（Welford 算法）、简化距离度量、蒙特卡罗采样（k‑均值++ 与主动学习）、近邻替代器、主动学习框架以及互补贝叶斯分类器，对多目标任务的距离度量（disty）等。

**📊 数据集**

实验数据集来自 MOOT 仓库的 124 个多目标软件工程优化任务，涵盖配置调优、性能预测、缺陷预测、测试选择、成本估算和文本挖掘等领域。

**📈 对比分析**

与 SMAC3、SHAP、LIME、FASTREAD 等对标工具进行比较；在 MOOT 任务上，EZR 的性能与或优于这些工具，同时运行速度提升约 500 倍、所需标记数和特征数大幅下降（<10 个特征，<100 个标签）。

**⚠️ 局限性**

局限性包括：实验仅覆盖表格型 SE 优化任务，无法直接推广到生成、感知或安全关键领域；受近邻代理的准确性限制；算法集合有限（缺少随机森林、梯度提升、神经网络等）；以及对特定数据集的依赖性。

---

## 536. TurtleAI: Benchmarking Multimodal Models for Visual Programming in Turtle Graphics

**arXiv ID:** 2606.03626 | [PDF](https://arxiv.org/pdf/2606.03626v1)

**作者:** Chao Wen `[一作]` (Max Planck Institute for Software Systems), Adish Singla `[通讯]` (Max Planck Institute for Software Systems)

**通讯引用:** 2632 | [OpenAlex ID](https://openalex.org/A5027711113)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于 Turtle Graphics 的教育视觉编程基准，并对 20+ 视觉语言模型在该基准上的表现进行系统评估。

**💡 创新点**

创新点包括：①构建了 823 个真实学生任务的评估集；②设计了图像规范化与代码语义一致性评估框架；③提出了基于代码突变和精英选择的训练数据生成技术，可在少量种子样本上扩增 70 万规模数据。

**🔧 技术方法**

主要技术手段包括：LLM 驱动的代码突变与迭代精英筛选、Turtle 仿真器进行绘图规范化、符号与嵌入式相似度评估，以及在 Qwen2‑VL‑72B 上进行微调。

**📊 数据集**

使用的数据集包含三部分：Real（来自 XLogoOnline 的真实任务）、HandDrawn（人工手绘版）和 Synthetic（生成的合成任务），共 823 个任务。

**📈 对比分析**

通过符号（像素级）和嵌入（ResNet18 编码）两种评估方法计算成功率；最强基线模型 GPT‑5、GPT‑4o 等仅达 30% 左右的成功率，微调后的模型在 Real 集合上提升至约 50%，但在 HandDrawn 上提升有限。

**⚠️ 局限性**

局限性包括：对空间推理和精细视觉复制的能力不足；微调主要改善代码与推理的一致性，未显著提升视觉理解；评估框架的规范化可能忽略重要几何细节；以及手绘任务的鲁棒性仍不理想。

---

## 537. Cross-Lingual Token Arbitrage: Optimizing Code Agent Context Windows via Local LLM Preprocessing

**arXiv ID:** 2606.03618 | [PDF](https://arxiv.org/pdf/2606.03618v1)

**作者:** Mehmet Utku Colak `[一作]` (Istanbul Technical University), Mehmet Utku Colak `[通讯]` (Istanbul Technical University)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5111273368)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种边缘端预处理中间件，先将多语言提示翻译为英文并重写为结构化的Bi‑Block格式，然后将压缩后的提示发送给云端LLM。

**💡 创新点**

创新点在于：①将跨语言代价剥离与结构熵预压缩统一到同一预处理流程；②通过5% token‑budget 保障不膨胀；③在多语言基准上实现高达47%提示压缩且保持甚至提升准确率。

**🔧 技术方法**

技术包括本地3B LLM（Llama 3.2）实现翻译与重写，正则校验与回退机制，Bi‑Block标签化结构，以及对照LLMLingua‑2压缩模型进行比较。

**📊 数据集**

使用自定义的OMH‑Polyglot基准（200条多语言编程任务，覆盖四种语言与混合写法）以及其控制版OMH‑Wrapped。

**📈 对比分析**

与原始提示、LLMLingua‑2以及标准云端模型进行对比，结果显示提示压缩34–47%、总token减少8.3–18.8%，在Gemini和OpenAI模型上准确率维持或提升，OckScore显著优于LLMLingua‑2。

**⚠️ 局限性**

局限包括：仍未完成regex_strip和llm_no_names的消融实验；本地3B模型内存/延迟约118–279 ms；在某些后端（如CodeWhisperer）会导致成本上升；缺乏对中文/阿拉伯语翻译的本地语言专家审核。

---

## 538. OmniHalluc-L: Counterfactual Benchmarking and Modality-Perturbation Reliability Calibration for Long-Form Omni Hallucination

**arXiv ID:** 2606.03614 | [PDF](https://arxiv.org/pdf/2606.03614v1)

**作者:** Zixuan Dong `[一作]` (National University of Defense Technology), Jiaheng Liu `[通讯]` (Nanjing University)

**通讯引用:** 2376 | [OpenAlex ID](https://openalex.org/A5032858379)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了长视频Omni hallucination评估协议和Benchmark OmniHalluc-L，并在此基准上开发了冻结骨干的音频负样本可靠性校准框架MPRC；

**💡 创新点**

创新点包括：①基于对照事实的事件绑定构造和严格对称对评分（SPA）评估；②构建包含1,800对支持/反事实的长视频问答数据集；③利用结构化音频负样本作为可靠性探针，配合轻量级校准层提升模型的绑定一致性；

**🔧 技术方法**

使用技术：构造支持/反事实对、视频分段与音频/视觉单独标注、冻结Omni骨干模型、音频负样本扰动（时间偏移、片段交换）、轻量级可靠性校准层（逻辑回归/GBDT），以及视频级交叉验证；

**📊 数据集**

使用的数据集：638段约24分钟长视频（共256.87小时），生成3,600条单句QA（1,800对），涵盖时间、共现、长期归因三类绑定关系；

**📈 对比分析**

与闭源强基线Gemini 3.1 Pro（SPA 76.54%）对比，开放权重Omni模型原始SPA仅32-41%；MPRC在Open-weight模型上提升至36-51%；在OmniVideoBench和WorldSense的多项选择任务中，MPRC分别提升约1-5%（如Qwen3在OmniVideoBench +3.6%）；

**⚠️ 局限性**

局限性：评估仅限单句验证，未覆盖多轮对话、开放式生成或交互式证据检索；MPRC依赖模型公开的置信度/得分，对闭源系统适用受限。

---

## 539. CoEval: Ranking Language Models for Custom Tasks Without Labeled Data or Trustworthy Benchmarks

**arXiv ID:** 2606.03650 | [PDF](https://arxiv.org/pdf/2606.03650v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 540. An AutomationML Domain Library for the Formalized Process Description

**arXiv ID:** 2606.03691 | [PDF](https://arxiv.org/pdf/2606.03691v1)

**作者:** Hamied Nabizada `[一作]` (Helmut Schmidt University Hamburg), Alexander Fay `[通讯]` (Ruhr University Bochum)

**通讯引用:** 4991 | [OpenAlex ID](https://openalex.org/A5016841027)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

**🎯 论文内容**

提出并实现了基于 CAEX 3.0 的 AML 域库，对 VDI/VDE 3682 标准的 FPD 语言元素、属性、连接语义和图形信息进行完整建模，并提供了双向映射工具将 FPB.js 模型转换为 AML，验证了模型的无损互通；

**💡 创新点**

创新点在于构建了可重用的 FPD 领域库，系统化地将图形化过程描述语言映射为 AML 生态中的标准类库，支持跨工具、跨域的数据交互，并通过双向映射实现了完整的图形与信息模型无损转换；

**🔧 技术方法**

使用 CAEX 3.0 元模型、AML 语言、OMG dd 元模型来定义角色、接口、属性及图形信息，并在 .NET AML Engine SDK 中实现了 FPB.js 与 AML 之间的双向映射工具；

**📊 数据集**

未使用传统公开数据集，主要验证基于 FPB.js 的自建多层次、包含所有对象类型和连接类型的模型；

**📈 对比分析**

通过自动化比对原始 FPB.js 模型与映射回来的 AML/FPB.js 模型，确认所有对象、属性、连接拓扑、分解关系和图形信息在往返转换中无损；目前未给出数值性能指标，但验证覆盖多复杂度模型；

**⚠️ 局限性**

局限性在于仅对 FPB.js 这一单一工具完成了往返验证，缺乏对其它独立 FPD 工具的兼容性验证，且图形信息仍依赖于外部约定，未来需进一步扩展和标准化。

---

## 541. A Close Look At World Model Recovery In Supervised Fine-Tuned LLM Planners

**arXiv ID:** 2606.03685 | [PDF](https://arxiv.org/pdf/2606.03685v1)

**作者:** Patrick Emami `[一作]` (National Laboratory of the Rockies), Peter Graf `[通讯]` (National Laboratory of the Rockies)

**通讯引用:** 15385 | [OpenAlex ID](https://openalex.org/A5036327114)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型进行监督微调（SFT），让其生成经典规划（Blocksworld、Logistics）任务的完整计划，并通过线性探针与生成概率两种方法评估模型是否在内部恢复了规划世界模型。

**💡 创新点**

首次系统地将内部表示探针与生成概率评估结合，比较不同训练数据分布（随机游走、增强随机游走、最优计划）和状态链式思维（State‑CoT）对世界模型恢复的影响，揭示“模型内部知道更多，但未必在输出中体现”的现象。

**🔧 技术方法**

使用LoRA高效微调、Transformer 线性探针、token log‑probability 评估、随机游走生成器、最优求解器、State‑CoT 数据增强，以及 OOD 测试框架。

**📊 数据集**

数据集包括：Blocksworld 与 Logistics 的 PDDL 规划实例（随机游走、增强随机游走、最优计划共计约30k 计划），以及 PlanBench 的 6–10 块 Blocksworld OOD 实例。

**📈 对比分析**

比较方法：对内部表示做 F1 线性探针，比较不同层的探针精度；对生成概率做“一步动作有效性分类”成功率；在 OOD 上测量目标达成率。结果显示：在分布内模型目标达成率 99%+，随机游走训练的模型内部表示最好，但在 OOD 下性能急剧下滑；增强随机游走提升 OOD 鲁棒性。

**⚠️ 局限性**

局限性包括：未证明内部表示与规划行为的因果关联；仅在小型 LLM 上实验，无法推广到更大模型；使用受限的 STRIPS 文本表示，未利用预训练的常识；只分析有效计划，忽略无效计划中的信息。

---

## 542. Foley-Omni: A Unified Multimodal Generation Model from Task-Level Audio Synthesis to Complete Video Soundtrack Generation

**arXiv ID:** 2606.03672 | [PDF](https://arxiv.org/pdf/2606.03672v1)

**作者:** Ye Tao `[一作]` (Nanjing University), Shuai Wang `[通讯]` (Nanjing University)

**通讯引用:** 79172 | [OpenAlex ID](https://openalex.org/A5100459168)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Foley-Omni统一多模态音频生成模型，能够同时处理文本和视频条件，从而一次性生成包含语音、音效和音乐的完整视频音轨。

**💡 创新点**

创新点包括：1）将任务级合成与完整音轨生成融合的课程学习策略；2）使用结构化文本标签与同步特征的双路径条件注入；3）构建可复现的Audiovisual数据处理流水线和公开的V2ST-Bench基准。

**🔧 技术方法**

技术手段为：Diffusion Transformer + Mel VAE + BigVGAN；条件流匹配训练；CLIP与Synchformer视觉特征融合；UM‑T5文本编码器；同步特征的时间对齐与加性注入。

**📊 数据集**

使用的数据集包括约2.7M对齐的多模态样本（TTA、TTS、TTM、V2A、VisualTTS），VGGSound、GRID、LRS2公开视频及音频，并在此基础上构建了300条V2ST-Bench样本。

**📈 对比分析**

通过客观评估（CLAP、IB、WER、DeSync、MOS等）以及人类MOS评测，与组合基线（MMAudio+CosyVoice/AudioX）对比，Foley-Omni在V2ST-Bench上实现了更低的WER、更好的同步、更高的MOS，并在单任务上与专家系统保持竞争力。

**⚠️ 局限性**

局限性：缺乏细粒度的音源控制接口（如音量平衡、说话人突出度等）；未与其他完整音轨生成系统直接比较（因缺乏公开实现）；多说话人训练导致语音清晰度偶尔受限，未来计划加入参考音频以提升说话人一致性。

---

## 543. Safety Measurements for Fine-tuned LLMs Should be Grounded in Capability

**arXiv ID:** 2606.03648 | [PDF](https://arxiv.org/pdf/2606.03648v1)

**作者:** Krishnapriya Vishnubhotla `[一作]` (National Research Council), Svetlana Kiritchenko `[通讯]` (National Research Council)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估在不同任务和模型上，使用LoRA与SafeLoRA进行的 fine‑tuning 对大型语言模型的能力与安全性的影响，重点关注能力锚定与安全评估的稳定性。

**💡 创新点**

创新点在于将 fine‑tuning 目标固定为可验证的任务能力，从而消除实验设置的随意性；同时揭示安全评估工具与基准的高度不确定性，并对比 SafeLoRA 在安全‑能力权衡上的表现。

**🔧 技术方法**

使用 LoRA（低秩适配）与 SafeLoRA（安全子空间投影）两种 PEFT 方法，结合自动化安全评估器（LlamaGuard、SORRY‑Bench）和 perplexity 测量。

**📊 数据集**

采用可验证的任务数据集（GSM8k、ARC、Science10k、BoolQ）用于能力评估；安全评估则整合 SORRY‑Bench、BeaverTails‑Eval/Intent、XSTest 等安全基准。

**📈 对比分析**

通过多模型（Llama‑3.2‑1B/8B、Qwen‑3‑4/8B）与多任务、不同 LoRA 超参数组合的实验，比较验证准确率、perplexity 与安全率；结果显示 SafeLoRA 能显著降低安全风险，但往往伴随能力下降，且安全评估结果随评估工具与基准不同而变化。

**⚠️ 局限性**

实验受限于少量开源模型、仅使用 LoRA PEFT、未探索检索增强或多语言场景，评估指标受采样温度等推理参数影响，且对真实对话情境的适用性尚不充分。

---

## 544. A 3D Isovist World Model -- Revealing a City's Unseen Geometry and Its Emergent Cross-City Signature

**arXiv ID:** 2606.03609 | [PDF](https://arxiv.org/pdf/2606.03609v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 545. Spatial Transcriptomics-Guided Alignment Enhances Molecular Profiling in Pathology Foundation Model

**arXiv ID:** 2606.03644 | [PDF](https://arxiv.org/pdf/2606.03644v1)

**作者:** Fengtao Zhou `[一作]` (Hong Kong University of Science and Technology), Hao Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 112918 | [OpenAlex ID](https://openalex.org/A5100353673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研发并验证了 STAMP，一种基于空间转录组（ST）引导的跨模态对齐框架，能够从常规 H&E 组织切片直接推断多种分子表型（基因表达、肿瘤微环境域、诊断免疫组化、生物学驱动突变、免疫治疗指标和预后标记）。

**💡 创新点**

创新点包括：① 以空间转录组为监督信号，将高维稀疏基因表达聚合为功能通路表示；② 采用低秩适配（LoRA）在预训练的 Vision‑centric PFM（Virchow2）上进行参数高效微调；③ 结合跨模态对比损失与通路重构损失，实现视觉表征与分子状态的双向对齐；④ 通过无监督聚类和多实例学习展示模型的天然空间域识别与临床预测能力。

**🔧 技术方法**

技术栈：路径聚类通路表示、低秩适配 LoRA、跨模态对比与重构联合训练、Cross‑Attention 解码、线性探测、无监督聚类、Attention‑based Multiple Instance Learning (ABMIL)、统计显著性检验与 bootstrap。

**📊 数据集**

使用了：① HumanST‑1k（1,004 样本，1.8M 伴随 H&E‑ST 贴片）；② 多平台 ST 数据（Visium、Xenium、Visium HD 等）；③ 8 个肿瘤类型的公开 ST 评测基准；④ 多中心 5 种肿瘤 67,636 病例、37,229 张切片的回顾性和前瞻性临床数据集（乳腺、肺、结直肠、脑等）。

**📈 对比分析**

与 Virchow2 及 UNI、CONCH、PLIP、OmiCLIP 等现有 PFMs 对比，评估指标包括：空间基因表达 PCC（平均提升 0.029）、空间域识别 ARI（提升 0.04–0.07）、24 个临床标记 AUC（平均提升 0.02–0.05，部分任务提升 >0.08），前瞻性三角测试显示可降低 25–45% 的 IHC 检测量。所有提升均在 1,000 次 bootstrap 以及 Wilcoxon 检验下显著。

**⚠️ 局限性**

局限性：① 只能捕捉可视化的形态学信号，对微量/无形态学改变的突变预测仍有限；② 依赖公开 ST 数据，缺乏对新平台或未覆盖通路的通用性；③ 对 H&E 质量与切片标准化要求高，低质量样本可能影响推断；④ 目前仅聚焦于通路层面，未来需整合空间蛋白质/代谢组等多组学信息。

---

## 546. Gender-Dependent Diagnostic Substitution in LLM Medical Triage: Same Symptoms, Unequal Urgency

**arXiv ID:** 2606.03641 | [PDF](https://arxiv.org/pdf/2606.03641v1)

**作者:** Qi Han Wong `[一作]` `[通讯]`, Qi Han Wong

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估大型语言模型在相同神经症状下基于性别和年龄的急诊推荐差异

**💡 创新点**

发现诊断替代机制导致年轻女性被低急诊化的系统性性别偏差

**🔧 技术方法**

使用Gemini 3.5 Flash、Claude Sonnet 4.6、GPT-5.4-mini三类LLM，并采用结构化JSON输出

**📊 数据集**

构造单一神经症状描述，按七种性别/年龄组合（25/38/65岁 × 男/女/无性别）进行实验

**📈 对比分析**

通过Fisher检验和Cohen h效应值比较ER推荐率，发现25/38岁女性与男性差异显著（p<0.001，效应大）

**⚠️ 局限性**

局限于部署级模型、单症状单轮强制输出、未做临床验证、样本量有限，可能影响泛化

---

## 547. VidMsg: A Benchmark for Implicit Message Inference in Short Videos

**arXiv ID:** 2606.03635 | [PDF](https://arxiv.org/pdf/2606.03635v1)

**作者:** Issar Tzachor `[一作]` (OriginAI), Rami Ben-Ari `[通讯]` (OriginAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 VidMsg 基准，用于评估短视频中隐式信息（即视频想传达的高层信息）的理解能力，设计双向检索和多选问答两种评测任务。

**💡 创新点**

创新点包括：① 以“信息先行”的收集管道，通过 LLM 生成间接检索关键词，避免过度显式的目标信息；② 在同一数据集上同时提供检索和 QA 评测，突出隐式信息推理而非仅仅识别可见实体；③ 提出 VidVec‑Msg 基线方法，利用文本‑视觉配对的轻量化优化显著提升检索性能。

**🔧 技术方法**

技术手段：使用大型语言模型（LLM）生成情节和检索关键词；人工标注视频的隐式信息强度和显式程度；对现有视频‑文本检索模型、视频‑文本 MLLM 进行评测；基线 VidVec‑Msg 在 MLLM 上进行文本‑视觉配对的轻量化优化。

**📊 数据集**

数据集：400 条 YouTube 产出短视频，涵盖 9 个实用主题（如职业、教育、健康等），共 52 条细粒度隐式信息，平均每条信息有 5‑9 条视频。

**📈 对比分析**

对比方法：对现有检索模型（Clip4Clip、VideoPrism、Qwen3‑VL‑Emb 等）以及 MLLM 进行双向检索（mAP、Recall@10/1）和 5‑选多项选择问答（准确率）评测。结果显示现有模型性能低于基线 VidVec‑Msg，仍存在显著提升空间（最高 mAP 仅 65%，问答准确率最高 76%）。

**⚠️ 局限性**

局限性：① 数据规模有限，难以覆盖所有可能的隐式信息；② 对 YouTube 的依赖可能导致平台偏差；③ 人工标注带有主观性，评判标准可能不统一；④ 当前模型仍难以完成复杂的推理，无法完全捕捉多模态隐式线索。

---

## 548. Dynamic Objective Selection with Safeguards and LLM Oversight for Financial Decision-Making

**arXiv ID:** 2606.03704 | [PDF](https://arxiv.org/pdf/2606.03704v1)

**作者:** Keigo Sakurai `[一作]` (Hokkaido University), Kei Nakagawa `[通讯]` (Osaka Metropolitan University)

**通讯引用:** 506 | [OpenAlex ID](https://openalex.org/A5086122043)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a2602d71-93ab-4bad-974b-672788df8193` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出 DOSS，一种基于滚动窗口的动态目标选择器，在金融决策中直接从可解释的近期收益统计中挑选预定义目标，并通过置信度门控与可选的 LLM 审计实现安全部署；

**💡 创新点**

将目标选择视为可解释特征上的分类问题，提供滚动学习与置信度门控的安全保障，并将 LLM 作为仅能接受/覆盖的审计层，避免引入新目标或噪声；

**🔧 技术方法**

线性 softmax 分类器、滚动窗口训练、置信度门控、切换频率控制、可选 LLM 审计；

**📊 数据集**

FAR-Trans 公共金融资产推荐基准；

**📈 对比分析**

与静态目标、直接 LLM 预测、Mock 规则以及后验 Oracle 进行对比；DOSS 在候选集上实现了约 56.3% 的 HPA（比最强静态提升约 7.5%），且切换率仅为 45.9%，显著低于后验 Oracle（81.97%）和直接 LLM（59.02%）；

**⚠️ 局限性**

局限在于只评估预定义目标集，缺少对更广泛目标空间的探索；置信门控阈值需经验调优；且未在多样化金融场景（如不同市场周期、资产类别）中验证鲁棒性。

---

## 549. On Secure EKF-enhanced UAV-ISAC Systems

**arXiv ID:** 2606.03690 | [PDF](https://arxiv.org/pdf/2606.03690v1)

**作者:** Hongjiang Lei `[一作]`, Yun Li `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建了一种集成感知与通信（ISAC）的无人机（UAV）安全系统，利用扩展卡尔曼滤波（EKF）实时预测并跟踪合法用户与可疑窃听者的位置，同时联合优化天线波束和无人机轨迹以最大化系统的安全率。

**💡 创新点**

创新点在于：①首次将 EKF 与无人机轨迹规划相结合，实现对合法与非法移动目标的双向预测和跟踪；②在考虑感知波束宽度、发射功率、推进能耗等实际约束的前提下，采用块坐标下降+顺序凸逼近（SCA）+EKF 的迭代算法，求解原始高度非凸的安全率优化问题；③通过动态调整 UAV 轨迹，实现对目标的持续跟踪与干扰，从而提升安全率和估计精度。

**🔧 技术方法**

技术手段包括：扩展卡尔曼滤波（EKF）用于状态估计与预测；块坐标下降（BCD）与顺序凸逼近（SCA）实现对波束与轨迹子问题的可行近似；半正定规划（SDP）与 CVX 求解器；线性化约束与能耗模型；以及基于 LoS 的多输入多输出（MIMO）波束形成。

**📊 数据集**

实验基于仿真数据，采用论文中给出的参数表（如 M=16、fc=30 GHz、δt=0.1 s 等）进行数值仿真，并未使用公开真实数据集。

**📈 对比分析**

与固定位置 UAV（仅优化波束）进行对比；结果显示：①在相同初始位置下，所提方案的安全率明显高于基准；②跟踪误差累积分布（CDF）更小，表明估计更精确；③安全率随干扰残余水平 ψ、最大发射功率 Pmax 和推进功率 Pmax^hor 的变化趋势符合预期，证明方法在不同参数下均能提升性能。

**⚠️ 局限性**

局限性包括：①假设目标遵循常速运动模型，未考虑加速度或复杂运动；②仅考虑 LoS 环境，忽略杂波和多径干扰；③单 UAV 方案在 NLoS 或大规模部署时可能表现欠佳；④EKF 在测量噪声较大或轨迹偏差较大时可能发散，需要更鲁棒的估计方法。

---

## 550. GN0: Toward a Unified Paradigm for Generation, Evaluation, and Policy Learning in Visual-Language Navigation

**arXiv ID:** 2606.03682 | [PDF](https://arxiv.org/pdf/2606.03682v1)

**作者:** Xinhai Li `[一作]` (China Telecom), Xuelong Li `[通讯]` (China Telecom)

**通讯引用:** 57257 | [OpenAlex ID](https://openalex.org/A5100740143)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个统一的高保真3D Gaussian Splatting（3DGS）导航生态系统，包括大规模数据集 GN‑Matrix、交互式模拟平台 GN‑Bench 以及基础模型 GN‑BAE，实现从数据生成到模型训练再到评估的一站式闭环。

**💡 创新点**

核心创新在于：① 将 3DGS 视角渲染与鸟瞰图（BEV）融合，提供几何一致的全局空间记忆；② 通过动态 3DGS 人类头像打造人机交互场景；③ 采用多阶段训练（SFT → DAgger → RL）以及轨迹令牌化，显著提升导航鲁棒性与泛化能力。

**🔧 技术方法**

技术手段包括：3D Gaussian Splatting 渲染、BEV 生成、语义地图与动态人像合成、轨迹令牌化与序列化、SFT+DAgger+DAPO 的三阶段强化学习、Action Expert、混合输入（FPV、BEV）和多模态投影。

**📊 数据集**

使用了 GN‑Matrix 数据集，融合了 InteriorGS、手工大规模室内场景、WorldGrow 扩展和动态 3DGS 头像，并按任务类型（目标导航、指令跟随、人跟随）生成海量多模态训练样本。

**📈 对比分析**

在 GN‑Bench（3DGS 真实感环境）和 VLN‑CE R2R 未见数据集上与现有方法对比，GN‑BAE 在 FPV/BEV 仅靠单目 RGB 的条件下实现了：GN‑Bench SR 达 38.9% / 43.6%，SPL 37.3%/38.2%；VLN‑CE R2R 失真误差 NE 3.50m，SPL 63.4%，均为最优或接近最优。

**⚠️ 局限性**

局限性主要体现在：① 对复杂物理交互（动力学、抓取等）支持不足；② 依赖 3DGS 资产，尚未验证在更广泛的户外或多样化机器人平台上的通用性；③ 模拟环境虽高保真，但仍缺乏真实场景中的噪声与不可预知因素。

---

## 551. EvoDrive: Pareto Evolution for Safety-Critical Autonomous Driving via Self-Improving LLM Agents

**arXiv ID:** 2606.03678 | [PDF](https://arxiv.org/pdf/2606.03678v1)

**作者:** Tong Nie `[一作]` (Hong Kong Polytechnic University), Wei Ma `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 88701 | [OpenAlex ID](https://openalex.org/A5100392071)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

介绍了 Evodrive，一个基于 LLM 的多目标代理进化框架，能够在不使用手工启发式的前提下，自动化地改进驾驶仿真中的对抗性场景生成器，以平衡攻击性和现实性。

**💡 创新点**

创新点包括：①首次实现无手工规则的多目标代理进化；②采用模拟器约束的 actor‑critic 代理与结构化的批评家、世界评估器实现安全且高效的搜索；③使用 Pareto 档案保持多样化的攻击‑现实权衡。

**🔧 技术方法**

使用技术包括大语言模型（LLM）进行生成器编辑、actor‑critic 架构、确定性验证器、基于场景的评估器、Pareto 档案以及多目标进化循环。

**📊 数据集**

实验数据集为 MetaDrive 与 CARLA 仿真环境，并利用 SafeBench 中的多种场景生成器进行评估。

**📈 对比分析**

与 CAT、ADV‑BMT、AT 等现有生成器对比，Evodrive 在攻击‑现实 Pareto 区域上持续扩展，且在下游策略训练中将碰撞率降低约 25%，整体性能显著提升。

**⚠️ 局限性**

局限性在于假设固定的 ego 策略、有限的仿真预算以及仅在仿真中验证，可能无法覆盖未见策略、不同平台或真实部署中的失效；需要进一步的硬件对接与协同进化验证。

---

## 552. Diagnosing Knowledge Gaps in LLM Tool Use: An Agentic Benchmark for Novel API Acquisition

**arXiv ID:** 2606.03657 | [PDF](https://arxiv.org/pdf/2606.03657v1)

**作者:** Jinnuo Liu `[一作]` (NYU Shanghai), Hongyi Wen `[通讯]` (NYU Shanghai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了动态、可重生成的 NovelAPIBench 评测框架，用于评估代码 LLM 对新 API 的获取与使用，并自动化错误分类。

**💡 创新点**

首次将 API 知识拆分为表面签名、使用示例、机制说明与实现源码四个可插拔组件，并通过自动失败分类诊断不同知识缺失的具体错误，构建了面向模型的可更新 benchmark。

**🔧 技术方法**

采用动态版本对比、AST 解析、检索增强、强化学习式任务生成、自动化错误分类器（基于 GPT‑5‑mini）等技术。

**📊 数据集**

基于 5 个 Python 库领域共 19 个库、约 800 个新 API、1.9k 任务，评测在 4 个 8B 背景模型上进行。

**📈 对比分析**

与检索（RAG）、监督微调（SFT）、RAFT、GRACE、MEMIT、AlphaEdit 等范式对比；检索始终领先，外部知识单独使用时可达约 18% pass@1；最佳组合 S+M_prose 或 S+E 约 23–24% pass@1；微调虽提升 70+pp 与检索差距，却未能内部化知识，模块路径错误仍高。

**⚠️ 局限性**

benchmark 依赖库版本与预训练截止点，易随模型更新失效；外部知识注入方式不现实；仅覆盖 Python，未探究更大规模或多语言情形。

---

## 553. The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models

**arXiv ID:** 2606.03645 | [PDF](https://arxiv.org/pdf/2606.03645v1)

**作者:** Liuyuan Wen `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 12904 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在多项整数加法中的内部表示与错误机制，提出Iso-Raw-Sum Trajectory（IRST）和噪声量化模型，验证几何结构并设计推理时自纠正方法。

**💡 创新点**

引入连续的“Carry Potential”概念，将算术错误建模为噪声导致的量化边界越界；发现“geometric slippage”与IRST，解释探针可解码多信号的机制，并通过一致性检查实现自纠正。

**🔧 技术方法**

通过轻量级线性/非线性探针解码隐藏层激活，使用UMAP降维可视化几何结构，提出噪声量化模型与定量错误分布分析，采用推理时的双流一致性干预和方向性干预。

**📊 数据集**

对Qwen3-4B模型生成的10,000个三项10位整数加法任务（每个加数10位），在中间位置p=4进行激活分析。

**📈 对比分析**

与重提示、线性驱动、硬替换等基线比较，双流一致性干预（δ=0.1）实现89.56% token精度，显著提升并表明IR模型有效；错误率与噪声水平匹配，模型表现符合预测。

**⚠️ 局限性**

仅验证加法、基于单字符分词模型；对BPE模型的适用性仍未知；几何分析依赖UMAP和探针而非精确电路定位；未完整探索多操作数的可扩展性与更复杂算术。

---

## 554. Causal Mirage Equilibrium in Agentic Machine Intelligence

**arXiv ID:** 2606.03636 | [PDF](https://arxiv.org/pdf/2606.03636v1)

**作者:** Hamidou Tembine `[一作]` `[通讯]` (Universite Du Quebec A Trois Rivières), Hamidou Tembine (Universite Du Quebec A Trois Rivières)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

本文提出并研究了 Causal Mirage Equilibrium（CME），用于描述生成式机器智能中语义表征与现实脱耦后仍保持自洽的稳态现象。

**💡 创新点**

创新点在于将语义解耦与自我强化机制量化为“mirage intensity”，并构造了风险敏感均值场型 CME 概念，证明其存在并给出分岔定理。

**🔧 技术方法**

采用了均值场理论、期望化风险期望、Kakutani‑Glicksberg‑Fan 固定点定理、非线性分岔分析等数学工具。

**📊 数据集**

未使用任何具体实验数据集，而是以理论推导和数学证明为主。

**📈 对比分析**

由于是理论研究，没有进行方法对比或性能评估；若将来与传统 Nash/贝叶斯平衡等对比，预期可揭示对假设不健全系统的稳定性。

**⚠️ 局限性**

主要限制在于假设的连续性、凸性及稠密性条件对实际大规模生成模型可能不成立，且缺乏实证验证和算法实现。

---

## 555. AnchorMoE: Interpretable Time Series Classification via Anchor-Routed MoE

**arXiv ID:** 2606.03631 | [PDF](https://arxiv.org/pdf/2606.03631v1)

**作者:** Tao Xie `[一作]` (Guangdong University of Technology), Yiu-ming Cheung `[通讯]` (Hong Kong Baptist University)

**通讯引用:** 10286 | [OpenAlex ID](https://openalex.org/A5038516431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种可解释的多变量时序分类框架 AnchorMoE。

**💡 创新点**

通过多视角表征、正交后验锚点路由和不确定性门控实现严格的先验可解释性，并抑制稀疏信号中的背景噪声。

**🔧 技术方法**

采用 Mixture-of-Experts 架构、多视图嵌入（时间、频谱、上下文）、正交约束、可靠性门控以及卷积与频谱处理技术。

**📊 数据集**

使用了 18 个 UEA 多变量时序分类基准数据集以及 4 个人工构造的带真实标注的合成数据。

**📈 对比分析**

与 9 种基线模型比较，AnchorMoE 在 ACC 与 F1 上平均排名第一，性能与最强对手持平，并在解释性指标上显著优于后验方法。

**⚠️ 局限性**

仅支持局部加性决策，难以捕捉高阶段间交互，门控可能过度抑制细微上下文特征。

---

## 556. Building Reliable Long-Form Generation via Hallucination Rejection Sampling

**arXiv ID:** 2606.03628 | [PDF](https://arxiv.org/pdf/2606.03628v1)

**作者:** Lin Li `[一作]` (University of Oxford), Yarin Gal `[通讯]` (University of Oxford)

**通讯引用:** 26333 | [OpenAlex ID](https://openalex.org/A5029186201)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为SHARS的推理时拒绝采样框架，并设计了基于语义熵的HalluSE检测器，用于长文本生成中的幻觉过滤与重写，显著降低幻觉率并提升输出的事实性与信息量。

**💡 创新点**

创新点包括：① 在句子级别进行动态拒绝采样与重写，阻断幻觉雪崩；② 将语义熵改造成适用于长文本的HalluSE，避免实体歧义和多答案问题；③ 通过可调阈值实现响应率与事实精度的平衡，兼顾推理时可扩展性；④ 证明该框架与外部检索无关，可与训练时方法（如FactAlign）协同使用。

**🔧 技术方法**

使用的技术包括：大型语言模型（Qwen3‑4B/32B、Llama3.1‑8B）推理；基于语义熵的无监督不确定性检测；句子重写（利用LLM生成可信信息）；温度/Following两种采样策略；推理时计算扩展（动态温度、重采样）；与现有的DoLa、ID、ChatProtect、Self‑Endorse等基线进行对比。

**📊 数据集**

使用的公开数据集：FactScore、FactualBio、LongFact 等长文本事实性基准；实验采用 Qwen3‑4B/32B、Llama3.1‑8B 等模型。

**📈 对比分析**

与 Greedy、DoLa、ID、ChatProtect、Self‑Endorse 等方法比较，SHARS 在 FactScore 上事实精度提升约 20–26%（最高达 78.4%），支持声明数增加；在 FactualBio 上 AUROC 达 72.9%（优于 Naive SE 的 66.2%）；在 LongFact 上 Precision 94.6% 对比 Greedy 的 93%。同时在推理时计算效率上，SHARS 在 10–40× 运行时即可达到 60–78% 的事实精度，显著优于 Self‑Endorse 等基线。

**⚠️ 局限性**

局限性：① 在追求事实性时可能略微降低信息量；② 由于不依赖外部检索，无法为模型无知识的查询提供新信息；③ 对极低参数量或低质量模型的提升有限。

---

## 557. Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments

**arXiv ID:** 2606.03698 | [PDF](https://arxiv.org/pdf/2606.03698v1)

**作者:** Sangeun Park `[一作]` (Sungkyunkwan University), Minhae Kwon `[通讯]` (Sungkyunkwan University)

**通讯引用:** 306 | [OpenAlex ID](https://openalex.org/A5032094381)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种层级化多智能体决策框架^2，明确将高层规划与低层执行分离；

**💡 创新点**

创新点在于：①利用SFT训练高层子目标生成器以保证语义一致性；②通过离线-在线强化学习优化低层执行器，加入政策锚定优势项和KL正则化以实现稳定自适应；③提供了结构化的层级数据集与完整训练流水线；

**🔧 技术方法**

技术包括：大语言模型（Qwen-2.5、Mistral、Llama-3.1）+LoRA适配器、离线-在线RL（Actor-Critic、IQL、AWAC）、SFT行为克隆、KL约束与优势加权；

**📊 数据集**

使用了三类交互式基准数据集：ScienceWorld、ALFWorld、TextCraft，并自行构建了层级化训练/评估数据集；

**📈 对比分析**

在三大基准上，与Prompt‑based、Fine‑tuning baselines对比，^2在ID/OOD和不同难度任务中取得最高pass@1/成功率，且token效率最高，显著降低了目标漂移；

**⚠️ 局限性**

局限性包括：①对大模型依赖性较高；②离线数据质量与覆盖仍决定上限；③在极端长周期或极高复杂度场景下仍可能出现累积误差；

---

## 558. Staying Alive: Uncensored Survival Analysis with Tabular Foundation Models

**arXiv ID:** 2606.03689 | [PDF](https://arxiv.org/pdf/2606.03689v1)

**作者:** Mariana Vargas Vieyra `[一作]` `[通讯]` (Rhizome Labs), Mariana Vargas Vieyra (Rhizome Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出在无数据集特定训练的前提下，利用冻结的表格基础模型进行生存回归，并通过基于Buckley‑James的迭代缺失值填充实现对右删失数据的处理。

**💡 创新点**

核心创新在于将生存分析视为上下文预测任务，使用单个标量参数即可构建AFT模型，并将表格基础模型作为非参数的in‑context估计器，迭代更新删失样本的伪目标。

**🔧 技术方法**

技术上结合了Accelerated Failure Time（AFT）框架、Kaplan‑Meier jackknife初始化、Buckley‑James迭代式缺失填充以及TabPFN/TabICL等表格基础模型进行in‑context学习。

**📊 数据集**

实验使用了五个公开基准数据集：WHAS500、GBSG、METABRIC、SUPPORT和FLCHAIN。

**📈 对比分析**

与传统训练的Cox PH、AFT、RSF模型以及零样本TabSA‑Bin方法比较，TabSA‑BJ在Harrell C‑Index上与训练模型相当，在IBS略低，但在无训练条件下表现出较强的区分能力。

**⚠️ 局限性**

局限性包括对IBS的校准不如离散化方法，迭代次数需要人工设定，以及在某些数据集上TabICL的表现可能不稳定。

---

## 559. Speedrunning Tabular Foundation Model Pretraining

**arXiv ID:** 2606.03681 | [PDF](https://arxiv.org/pdf/2606.03681v1)

**作者:** Salih Bora Ozturk `[一作]`, Frank Hutter `[通讯]` (Prior Labs)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了一个基于 nanoTabPFN 的公开速度赛（speedrun）框架，目标是在单块 NVIDIA L40S GPU 上以最短训练时钟（wall‑clock）完成对 38 个 TabArena 子采样分类任务的预训练，使其 ROC‑AUC 达到随机森林基线。

**💡 创新点**

创新点在于将语言模型社区的速度赛方法迁移到表格基础模型领域，并通过一系列可叠加的训练加速技术（Mu­on 优化器、SDPA、BF16、预归一化、编译加速、残差衰减、RMSNorm、思考行、LAWA、特征重组、自动化 HPO 等）实现 81× 的时间压缩，同时在保持同一 synthetic prior 的前提下显著减少所需的 synthetic 数据量。

**🔧 技术方法**

主要使用技术包括：Mu­on 2D 权重优化、Scaled‑Dot‑Product Attention（SDPA）重写、混合精度 bf16/TF32、预归一化（pre‑norm）和后归一化（post‑norm）切换、torch.compile 编译、残差衰减、RMSNorm、可学习的思考行、最新权重平均（LAWA）、重复特征分组、自动化 HPO 与 Mu­on 权重衰减以及在解码器中采用特征均值池化。

**📊 数据集**

数据集方面：使用 TabICL 生成的 256 000 份 synthetic classification 预训练数据集（每份最多 1 000 行、20 特征、8 类），以及 TabArena 公开的 38 个子采样分类任务（每任务最多 1 000 行、100 特征），用于验证和终止判定。

**📈 对比分析**

评估方法是对所有 38 个 TabArena 任务进行 5 折分层交叉验证，汇总 ROC‑AUC，平均值用作整体性能指标。记录显示从基线 74.32 min（80 576 份数据）压缩到 0.92 min（3 648 份数据），在同一硬件上实现了 81× 的速度提升，且仍能达到随机森林的平均 ROC‑AUC。

**⚠️ 局限性**

限制点包括：所有记录均在固定 synthetic prior 下进行，未探索如何改进 prior 本身；技术迁移主要基于单 GPU 评估，未验证多 GPU 或更大模型规模的可扩展性；最终模型在更大预算下的性能提升仍需进一步验证；此外，速度赛仅关注 wall‑clock，未考虑能耗与模型规模的整体成本平衡。

---

## 560. Deterministic Distance Approximation in MPC via Improved Hitting Sets

**arXiv ID:** 2606.03674 | [PDF](https://arxiv.org/pdf/2606.03674v1)

**作者:** Kyungjin Cho `[一作]`, Tijn de Vos `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了在多机并行计算（MPC）模型以及Congested Clique模型下，针对无向图的稀疏子图（spanner）与近似最短路径问题的第一批确定性算法，且这些算法在回合复杂度上实现了子对数甚至常数级别的突破；

**💡 创新点**

创新点在于：①通过对随机化算法中关键的随机采样步骤进行系统化的分离，将其抽象为“命中集合（hitting set）”问题；②构造了多种高效的确定性命中集合算法（线性与子线性两种记忆模型），从而实现了对原先随机算法的完全确定性化；③首次实现了在Congested Clique模型下的常数回合、常数近似率的全点对最短路径（APSP）算法，并在子线性MPC模型中得到 O(log k) 回合、O(k^{1+ε}) 伸缩因子的稀疏子图。

**🔧 技术方法**

主要技术包括：①k‑wise独立哈希函数与伪随机生成器用于极大降低采样所需的随机位数；②条件期望法（method of conditional expectation）在MPC环境下实现高效的确定性哈希选择；③稀疏化与层次化聚类技术（如基于 Baswana‑Sen 的递归层次结构）与命中集合的组合；④在MPC模型下对数据分布与通信进行精细化管理，实现了线性总空间与子线性本地空间的兼顾。

**📊 数据集**

论文未在实验中使用公开数据集，而是通过理论分析与复杂度证明展示算法性能；

**📈 对比分析**

与现有随机算法相比，本文在回合复杂度上与随机算法相当（常数或 O(loglog n) 等），但无需额外的概率失败容忍；在Congested Clique模型中，常数回合 APPSP 的常数近似率显著优于之前的 O(log n) 近似或 O(loglog n) 回合的随机算法；

**⚠️ 局限性**

限制方面：①算法仍需假设输入图为无向且权重可取整数；②在子线性 MPS 中，对记忆模型的假设（如 L = Θ(n^δ)）较为严格；③部分结果对 d 的取值范围有限制（如 d ≥ n^δ 或 d ≤ |U|/200）；④实际实现中对哈希函数与伪随机生成器的常数因子与实现复杂度未给出；

---

## 561. Graph Regularized Non-negative Reduced Biquaternion Matrix Factorization for Color Image Recognition

**arXiv ID:** 2606.03654 | [PDF](https://arxiv.org/pdf/2606.03654v1)

**作者:** Hailang Wu `[一作]` (Yunnan University), Chaoqian Li `[通讯]` (Yunnan University)

**通讯引用:** 1534 | [OpenAlex ID](https://openalex.org/A5058031677)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了图正则化的非负约简双四元数矩阵分解模型，用于彩色图像识别。

**💡 创新点**

将图拉普拉斯正则化引入约简双四元数系数矩阵，兼顾非负性与局部几何结构。

**🔧 技术方法**

采用约简双四元数矩阵分解、图正则化、投影梯度算法和Armijo回溯。

**📊 数据集**

在CASIA‑FaceV5、KDEF和Asirra三大彩色图像数据集上实验。

**📈 对比分析**

与NRBMF、GRIPG、RIPG、QIPG、QPCA等方法比较，GNRBMF在大多数情形下获得最高或第二高的识别率。

**⚠️ 局限性**

模型仍为非凸问题，收敛至局部极值，对λ等超参数依赖经验调优。

---

## 562. Q-FE: A Quantum-Native 6G Far-Edge Architecture Securing Industrial IoT Digital Twins via CSIDH-PQC and Asynchronous Federated Learning

**arXiv ID:** 2606.03611 | [PDF](https://arxiv.org/pdf/2606.03611v1)

**作者:** Vincenzo Sammartino `[一作]` (University of Pisa), Vincenzo Sammartino `[通讯]` (University of Pisa)

**通讯引用:** 20 | [OpenAlex ID](https://openalex.org/A5095698377)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种Quantum‑Native 6G远边缘架构Q‑FE，集成微数字孪生、跨层CSIDH密钥交换和基于DAG智能合约的异步联邦学习，满足IIoT超低时延、强隐私与量子安全需求。

**💡 创新点**

创新点包括：①将数字孪生迁移至gNB/端点，实现子毫秒状态同步；②在6G MAC控制元素嵌入64字节CSIDH公钥，避免碎片化；③异步联邦学习通过DAG‑DLT智能合约实现无阻塞、对抗模型中毒与Sybil攻击，并提供差分隐私。

**🔧 技术方法**

采用CSIDH‑512异域同源Diffie‑Hellman、AES‑256‑GCM、基于DAG的分布式账本、异步密钥轮换（AKR）和PySyft+NS‑3仿真环境。

**📊 数据集**

使用SWAT IIoT异常检测数据集进行CNN模型训练。

**📈 对比分析**

与传统ML‑KEM/Kyber‑1024基线对比：MAC层开销减少62%，P₉₉.₉ URLLC延迟0.78 ms，异步FL收敛速度比同步FedAvg快31%，能源占比仅13 µW/设备。

**⚠️ 局限性**

局限包括仿真采用理想信道模型、SWAT数据集可能不足以代表工厂多样化数据、正式安全性依赖GAIP假设、时延侧信道风险需进一步缓解，需实测6G硬件验证性能。

---

## 563. SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition

**arXiv ID:** 2606.03610 | [PDF](https://arxiv.org/pdf/2606.03610v1)

**作者:** Yanan Liu `[一作]` (Yunnan University), Qiuhong Ke `[通讯]` (Monash University)

**通讯引用:** 2503 | [OpenAlex ID](https://openalex.org/A5083239184)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出 SkelHCC 框架，实现一次性（one-shot）骨骼动作识别；

**💡 创新点**

创新点包括：① Explicitly Hierarchical Hyperbolic CLIP (EH‑HCLIP)，利用负曲率空间编码人体骨骼的层级结构，实现骨骼与语言的语义对齐；② LLM‑guided Multi‑granularity Voting Cache (LMV‑Cache)，通过大语言模型生成关节/部位重要性掩码，提升缓存检索的上下文感知；③ 在一次性学习中保持骨骼骨干网络冻结，仅训练极少量参数，显著降低适配复杂度；

**🔧 技术方法**

技术手段包括：超曲空间（Lorentzian hyperbolic）表示与对比学习、CLIP 视觉‑语言预训练、LLM（如 GPT‑4）提示生成掩码、多粒度骨骼分割、缓存检索与残差融合；

**📊 数据集**

使用 NTU‑RGB+D 60、NTU‑RGB+D 120 与 PKU‑MMD II 三个大规模骨骼动作数据集；

**📈 对比分析**

与多种现有方法（CrossGLG、MotionBERT、Trans4SOAR 等）对比，SkelHCC 在 NTU‑120 20/100 基类设置下分别提升 4.1–9.0% 以上，且只需 0.5M 训练参数；在 NTU‑60 与 PKU‑MMD II 上也均取得最高或接近最高成绩；

**⚠️ 局限性**

局限性包括：仅针对一次性学习实验，未验证少样本（few‑shot）场景；依赖预训练的 CLIP 与骨骼骨干，可能对新领域适配有限；超曲空间运算与掩码生成的计算成本相对较高；

---

## 564. SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents

**arXiv ID:** 2606.03692 | [PDF](https://arxiv.org/pdf/2606.03692v1)

**作者:** Yuan Xiong `[一作]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems), Kang Liu `[通讯]` (Key Laboratory of Cognition and Decision Intelligence for Complex Systems)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出SkillPyramid框架，将现有技能按层级组织，进行原子提取、抽象归纳，并通过任务驱动的自我进化持续扩展和复用技能。

**💡 创新点**

创新点包括：①双向重用机制（向下原子提取+向上抽象归纳）构建层级金字塔；②任务驱动的自我进化把新技能即时合并进层级；③显式的依赖与重用关系实现可解释、高效的技能复用。

**🔧 技术方法**

技术手段：LLM（DeepSeek‑V3.2、GPT‑4.1、Gemini 2.5 Pro、Qwen3‑235B）驱动的关系分析器与构造器、多代理协同架构、技能重写与生成策略。

**📊 数据集**

数据集：ALFWorld、WebShop、ScienceWorld、GAIA‑Lite（包含70+ 任务的文本交互基准）。

**📈 对比分析**

与ReAct、Reflexion、ExpeL、ReAct+Skills等基线对比，平均奖励提升38.0%，交互步数减少27.7%；在未见任务上奖励更高、步数更少，表明更优的泛化和效率。

**⚠️ 局限性**

局限性：①初始金字塔构建需一次完整扫描，难扩展至海量技能库；②对低质量LLM的鲁棒性未评估；③实验仅在文本交互基准，未验证多模态或实体环境；④缺乏对更丰富关系（时序、因果）的建模。

---

## 565. The DeepSpeak-Agentic Dataset

**arXiv ID:** 2606.03686 | [PDF](https://arxiv.org/pdf/2606.03686v1)

**作者:** Sarah Barrington `[一作]` (University of California Berkeley), Hany Farid `[通讯]` (University of California Berkeley)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 DeepSpeak‑Agentic 数据集，包含 200 条、总计 37 小时的半结构化人机对话录音，配有音视频分离、转录和元数据。

**💡 创新点**

首次提供真实时间、具身 AI 代理与人类交互的数据，既可用于多模态取证评估，又能深入研究人机交互的对话模式和辨别线索。

**🔧 技术方法**

结合大型语言模型（Llama‑4、GPT‑4o、GPT‑4o‑mini、GPT‑5.4‑mini）、合成语音（ElevenLabs、Cartesia、HeyGen Starfish）、视觉头像（Tavus、HeyGen LiveAvatar），并使用自研视频流与语音分离系统、Whisper ASR、GPT‑4o 过滤与评估。

**📊 数据集**

数据集本身即为本研究所用，公开在 HuggingFace，包含完整录音、分离音视频流、转录文本及元数据；此外还用公开的深度伪造检测器（文本：Desklib；音频：DF‑Arena‑1B；视频：GenConViT‑VAE）做基准评估。

**📈 对比分析**

与多种现成的文本/音频/视频伪造检测器进行对比，文本检测器 Desklib 在本数据集上达到 8% EER，最佳音频检测器 DF‑Arena‑1B 为 23% EER，最佳视频检测器 GenConViT‑VAE 为 33% EER，表明实时人机对话的 AI 信号对现有检测器提出更高挑战。

**⚠️ 局限性**

局限性包括对话半结构化、使用商业/开源生成管线、实验室受试者而非自然环境、强过滤导致数据“干净”且可能低估真实世界噪声，并且由于 AI 实时性提升，数据集更适合作为时间基准而非永恒的“最先进”标杆。

---

## 566. A Fast Methane Detection Pipeline on Board Satellites Based on Mag1c-SAS and LinkNet

**arXiv ID:** 2606.03675 | [PDF](https://arxiv.org/pdf/2606.03675v1)

**作者:** Jonáš Herec `[一作]` (Zaitra), Jan Sedmidubsky `[通讯]` (Masaryk University)

**通讯引用:** 619 | [OpenAlex ID](https://openalex.org/A5017748720)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一套面向低功耗卫星CPU的甲烷检测管线，通过加速的Mag1c-SAS滤波器与轻量级LinkNet模型实现实时检测；

**💡 创新点**

主要创新点在于将Mag1c滤波器改造为80倍加速的Mag1c-SAS，并将其与ML后处理相结合，首次实现低功耗平台上的可部署甲烷检测；

**🔧 技术方法**

技术包括快速的Mag1c-SAS滤波、ACE/CEM/ MF等目标检测方法、U‑Net/LinkNet语义分割模型、基于NumPy/OpenBLAS的高效实现以及ONNX Runtime的推理部署；

**📊 数据集**

使用了STARCOP（AVIRIS‑NG航空数据）和新构建的EMIT（ISS空间传感器）甲烷分割数据集，并配合高质量标注；

**📈 对比分析**

在STARCOP上，Mag1c-SAS+LinkNet实现强漏斗F1≈60%（比原始Mag1c低约10%）但运行时间仅1.19 s；在EMIT上，Tile‑wise Mag1c-SAS+LinkNet取得AUPRC≈67%（相较原始Mag1c提升≈45pp），同时整体推理时间仅1.58 s；

**⚠️ 局限性**

局限性包括仅在预处理数据上验证；对原始传感器数据的空间对齐和大气校正需求仍需高效实现；不同传感器需要阈值调优，且在极度异质场景下仍可能产生误报。

---

## 567. Beyond Single Solution: Multi-Hypothesis Collaborative Deep Unfolding Network for Image Compressive Sensing

**arXiv ID:** 2606.03666 | [PDF](https://arxiv.org/pdf/2606.03666v1)

**作者:** Wenxue Cui `[一作]` (Harbin Institute of Technology), Debin Zhao `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 12204 | [OpenAlex ID](https://openalex.org/A5100600353)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了一种多假设协同深度展开压缩感知网络（MHC-DUN），通过联合优化多个候选解来提高图像重建质量。

**💡 创新点**

核心创新点包括：① AlphaNet 自适应预测空间变步长，实现在多假设下的协同梯度更新；② MHCB 通过跨假设的通道/空间注意力实现多假设的协同近似映射；③ 组合损失函数平衡测量一致性、假设多样性与重建精度，鼓励探索互补解。

**🔧 技术方法**

技术手段：基于近端梯度下降的展开网络；多通道/空间注意力的 MHCB；AlphaNet 的残差注意力模块；联合损失函数；Swin Transformer 与 CNN 结合的特征提取模块。

**📊 数据集**

使用的公开数据集包括：Set11、Urban100（图像压缩感知评测）以及脑部医学图像数据集（CS‑MRI 评测）。

**📈 对比分析**

与多种深度黑盒网络（如 NL‑CSNet、CSformer）和深度展开网络（如 CPP‑Net、USB‑Net、DPC‑DUN）在 0.01–0.40 的采样率下进行对比，MHC‑DUN 在 PSNR 上平均提升 0.4–1.4 dB，SSIM 亦有 0.02–0.04 的提升，显著优于现有方法。

**⚠️ 局限性**

局限性：模型参数量较大（约 10.8M），推理速度略慢；需要训练 600 万次迭代，训练成本高；在极低采样率或不同信号类型（如视频、超声）下的泛化性能未作进一步验证。

---

## 568. Optimal Design and Analytical Modeling of a Soft Fin-Ray Effect Gripper Finger Using the Finite Rigid Elements Method

**arXiv ID:** 2606.03798 | [PDF](https://arxiv.org/pdf/2606.03798v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 569. Signed Spiking Neuron Enabled by an Orthogonal-Easy-Axis Magnetic Tunnel Junction

**arXiv ID:** 2606.03796 | [PDF](https://arxiv.org/pdf/2606.03796v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620`

---

## 570. Agentic Generation and Evolution of Knowledge Models

**arXiv ID:** 2606.03662 | [PDF](https://arxiv.org/pdf/2606.03662v1)

**作者:** Man Zhang `[一作]` (Beihang University), Sebastian Uchitel `[通讯]` (CONICET)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 TrustModel 平台，利用 LLM 基础的代理子系统（建模、符合性评估与演化）来持续构建、验证和演化知识模型，以支持模型驱动工程中的各种活动，如模型基础测试、需求监控等。

**💡 创新点**

创新点在于将大型语言模型与代理框架相结合，形成三层自适应子系统，实现知识模型的“活跃化”与闭环反馈；同时提供可实例化的 MBT 方案，展示知识模型在测试过程中的动态演化。

**🔧 技术方法**

使用了大型语言模型（LLM）驱动的代理、模型分析工具（语法检查、语义一致性检查）、符号推理工具（模型检查、求解器）、执行与监控框架（模拟器、日志采集）以及知识检索机制。

**📊 数据集**

论文未给出具体实验数据集，主要基于案例系统（智能家居能量管理系统）和公开的模型/规则库进行演示。

**📈 对比分析**

论文未进行量化对比或性能评估，主要通过案例演示说明平台的可行性，性能待后续实验验证。

**⚠️ 局限性**

局限性包括：缺乏大规模实测与性能评估；对 LLM 的准确性与可靠性依赖较高；平台实现复杂，需要多种工具与数据源协同，部署成本和维护成本高；目前仅在 MBT 场景下演示，其他 MDE 活动需进一步验证。

---

## 571. Designing a Hardware Reverse Engineering Course: Lessons from Eight Years in a Rapidly Evolving Tech Domain

**arXiv ID:** 2606.03697 | [PDF](https://arxiv.org/pdf/2606.03697v1)

**作者:** Zehra Karadağ `[一作]`, Steffen Becker `[通讯]`

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

基于八年经验的硬件逆向工程（HRE）课程设计与迭代报告，聚焦于数字电路分析与提取，并结合课堂实践、项目作业和期末考试。

**💡 创新点**

三大创新点：① 从“理论先行”转为“项目先行”，以真实案例驱动教学；② 将Python与HAL框架深度整合，使学生能快速上手；③ 采用“替代式扩展”而非“追加式扩展”，系统管理课程规模与工作量。

**🔧 技术方法**

使用开源工具HAL（含Python接口）、FPGA/IC图像处理工具、课程管理平台；采用课程结构模型（4CID）与12原则计算机教育原理进行课程评估。

**📊 数据集**

无公开数据集；课程评估基于学生成绩（期末考试通过率约93%，保留率约75%）和自评问卷。

**📈 对比分析**

通过对比不同迭代中的考试方式（从无考试→书面→口试→项目展示再回书面）与作业比例，发现写面试结合项目可兼顾实践能力与评估可行性，整体学习效果稳定且易于管理。

**⚠️ 局限性**

局限性包括：①课程内容与工具的更新速度受行业合作与授权限制；②高阶技术（如IC图像提取）需额外设备支持，降低可迁移性；③“替代式扩展”虽然控制规模，但可能导致知识深度不足；④缺乏公开的标准化评测数据集，难以与其他课程直接对比。

---

## 572. Physics-Guided Policy Optimization with Self-Distillation

**arXiv ID:** 2606.03620 | [PDF](https://arxiv.org/pdf/2606.03620v1)

**作者:** Ke Wang `[一作]` (Amazon), Kai Wei `[通讯]` (Amazon)

**通讯引用:** 16543 | [OpenAlex ID](https://openalex.org/A5087096372)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 Physics‑Guided Policy Optimization (PGPO)，对自蒸馏策略优化 (SDPO) 的步长进行物理引导的自适应调整。

**💡 创新点**

创新点在于使用互信息估计来动态调节 SGD 步长，借鉴流体力学中的粘性调节，解决信用分配和训练不稳定问题。

**🔧 技术方法**

使用了自蒸馏、互信息计算、SDE 理论、Adam/SGD、可变步长乘子以及流体力学类比等技术。

**📊 数据集**

在 Science‑Q&A 四个学科（化学、物理、生物、材料科学）数据集上评测。

**📈 对比分析**

与 SDPO 对比，在 4 个域中 3 个域提升，化学 +3.45、材料 +4.51、物理 +0.51，生物略降 -0.37，整体更稳定。

**⚠️ 局限性**

限制在于 α 参数选择敏感，当前仅批级调节，未做细粒度或自适应 α，且在某些域（如生物）仍不如 SDPO。

---

## 573. Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation

**arXiv ID:** 2606.03784 | [PDF](https://arxiv.org/pdf/2606.03784v1)

**作者:** Nan Sun `[一作]` (Tsinghua University), Huaping Liu `[通讯]` (Tsinghua University)

**通讯引用:** 12516 | [OpenAlex ID](https://openalex.org/A5041101317)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了ERVLA模型，利用大规模语义链式思考（CoT）与视觉‑语言‑动作框架相结合，以提升机器人控制的通用性和鲁棒性。

**💡 创新点**

核心创新在于：①将CoT视为训练时的表示塑形信号而非推理前缀；②采用CoT‑dropout与辅助动作查询，使模型在推理期间可自由选择是否使用CoT；③引入知识截断(KT)与选择策略，确保Diffusion Transformer仅关注语义内存，从而消除自回归推理导致的误差累积；④构建规模最大的语义CoT数据集（978k轨迹、226.3M样本、2592.5小时）。

**🔧 技术方法**

技术手段包括：Qwen3‑VL‑4B视觉‑语言模型、Diffusion Transformer（DiT）进行连续动作生成、流匹配（flow‑matching）损失、CoT‑dropout、动作查询辅助回归、知识截断（KV截断）、多任务联合训练。

**📊 数据集**

使用了LIBERO、LIBERO‑Plus、VLABench等仿真基准以及自建的大规模语义CoT数据集；此外在真实机器人平台上进行放置、清理等物体操作任务。

**📈 对比分析**

与ECoT、Emma‑X、OpenVLA‑OFT、UniVLA、WorldVLA、π₀、π₀‑FAST、π₀.₅等多种基线相比，ERVLA在LIBERO‑Plus上取得86.9%成功率，在VLABench平均成功率53.2%（对比π₀.₅的70.4%），并在真实机器人实验中在语义歧义、干扰物与长周期任务上明显优于对手，显示出更好的跨域泛化与鲁棒性。

**⚠️ 局限性**

局限性包括：①CoT标签的自动化标注仍存在噪声，尤其是空间定位字段；②模型对极端长序列或极大语义偏移的适应仍待提升；③当前实现依赖大型VLM与Diffusion Transformer，计算成本高；④在某些精细抓取或多模态指令场景下的细粒度误差尚未完全解决。

---

## 574. LAP: An Agent-to-Instrument Protocol for Autonomous Science

**arXiv ID:** 2606.03755 | [PDF](https://arxiv.org/pdf/2606.03755v1)

**作者:** Linwu Zhu `[一作]` (Shiyanjia Lab), Jian Huang `[通讯]` (Shiyanjia Lab)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出了 Lab Agent Protocol (LAP)，为人工智能代理与物理实验仪器之间建立标准化的通信接口，涵盖仪器卡片、预订、安防栅栏和测量结果四大物理世界原语。

**💡 创新点**

创新点在于将四大物理原语（InstrumentCard、排他预订、基于任务的安全栅栏握手与校准/不确定性绑定的 MeasurementResult）集成到统一协议中，填补了 MCP（工具）与 A2A（代理）之间缺失的代理-仪器边缘，且在安全性、可追溯性与物理量化方面实现了前所未有的规范化。

**🔧 技术方法**

采用了 JSON‑RPC 2.0 + HTTPS、SSE、Webhooks、JSON‑LD 与 W3C WoT Thing Description、JSON‑Schema、UCUM/QUDT 单位标识、JWS/签名、DID、OAuth2.1 等现有互联网与安全技术，复用 A2A 的发现与任务生命周期模式。

**📊 数据集**

该工作为设计规范，未使用具体实验或数据集；论文主要基于理论与示例实现的模拟。

**📈 对比分析**

通过对比 SiLA2、OPC‑UA、SCPI、MCP、A2A、SCP 等已有协议与标准，阐述 LAP 的功能覆盖与技术优势；但因尚无实际实现与基准测试，未给出量化性能数据。

**⚠️ 局限性**

局限性包括：目前仅为规范说明，缺乏大规模实现与经验评估；不支持实时硬件控制；安全栅栏依赖可信的操作员与授权机构；缺乏正式的能力本体与统一术语；联邦治理与跨实验室身份验证仍为开放问题；链条责任追踪只能在数字层面记录，无法确保物理样本真实性；LLM 生成的自然语言意图映射可能不准确。

---

## 575. Conformal Language Modeling via Posterior Sampling

**arXiv ID:** 2606.03731 | [PDF](https://arxiv.org/pdf/2606.03731v1)

**作者:** Nicolas Emmenegger `[一作]` (Massachusetts Institute of Technology), Chara Podimata `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 192 | [OpenAlex ID](https://openalex.org/A5027923437)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于后验采样的风险控制框架，利用对LLM生成的声明进行校准，使生成结果在给定风险阈值下保持事实性。

**💡 创新点**

核心创新在于将 conformal prediction 从事后过滤转移到采样分布本身，通过构造可调阈值的潜在函数和混合先验实现对后验分布的精细调节；并开发了一种离线校准方法（离线粒子采样+单调化）仅依赖基础模型样本即可获得风险控制。

**🔧 技术方法**

技术包括：后验采样、潜在函数阈值化、混合先验、离线重要性重加权、单调化估计、Glivenko-Cantelli一致性证明，以及 LLM‑judge 评估。

**📊 数据集**

使用 FActScore 数据集（人名/实体传记生成）和 MATH 数据集（数学推理问题）。

**📈 对比分析**

与最近的 post‑hoc conformal filtering 基线比较，实验表明在所有目标事实率下均能满足风险控制，并且在生成质量（完整性、流畅性、帮助性）与数学解答完整性方面均表现出更高的下游效用。

**⚠️ 局限性**

局限性包括：需要手工设置混合权重 β，影响置信与拒绝的权衡；对高置信度生成的可行性依赖于原始模型的支持范围；离线粒子采样在罕见高分区域可能低效；实验仅用 LLM‑judge 评估，缺乏人工评价。

---

## 576. A Double Bind: Gendered Funding, Research Topics, and Academic Performance in The Social Sciences

**arXiv ID:** 2606.03742 | [PDF](https://arxiv.org/pdf/2606.03742v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 577. Beyond False Stability: High-Noise Drift Gating for Test-Time Adversarial Defenses in Vision-Language Models

**arXiv ID:** 2606.03730 | [PDF](https://arxiv.org/pdf/2606.03730v1)

**作者:** Hashmat Shadab Malik `[一作]` (Mohamed Bin Zayed University Of Ai), Salman Khan `[通讯]` (Mohamed Bin Zayed University Of Ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在CLIP模型上提出一种训练无关的漂移门控机制，用高噪声特征漂移来检测潜在对抗样本，并仅在检测到不稳定时才激活现有的测试时防御方法，从而提升零样本鲁棒性

**💡 创新点**

发现CLIP视觉表征在从弱噪声到强噪声的过渡中，攻击样本的特征漂移从“假稳定”转为明显更大，利用这一高噪声漂移信号实现更精确的对抗检测；并通过门控显著降低对干净样本的干扰

**🔧 技术方法**

利用CLIP视觉编码器对图像与噪声/变换后的图像进行特征提取，计算特征漂移τ；采用高强度均匀/高斯噪声或光度/几何变换；结合现有测试时防御（TTC、AOM、R‑TPT）做门控；使用PGD、EOT‑PGD、CW、MI‑FGSM等对抗攻击进行评估

**📊 数据集**

在八个细粒度分类数据集（Caltech101、Pets、Flower102、Cars、Aircraft、DTD、EuroSAT、UCF101）以及ImageNet及其四个偏移版本（V2、Sketch、A、R）上进行实验

**📈 对比分析**

与原始CLIP、TTC、AOM、R‑TPT及其门控版本对比，平均干净+对抗准确率从70.1%提升至约73.2%（TTC门控），AOM门控平均提升至约72.8%，R‑TPT+门控提升至约73.2%；在更强攻击和OOD场景中仍保持显著改进，且门控减少了90%以上的干净样本防御开销

**⚠️ 局限性**

门控策略对已通过对抗训练的CLIP变体无效（因漂移分离消失），对高预算对抗攻击和极端噪声参数仍需进一步调优；此外，门控阈值需在不同数据集/模型上手动选择，增加了部署复杂度

---

## 578. Text-to-Image Models Need Less from Text Encoders Than You Think

**arXiv ID:** 2606.03715 | [PDF](https://arxiv.org/pdf/2606.03715v1)

**作者:** Nurit Spingarn `[一作]` (Technion Israel Institute of Technology), Tomer Michaeli `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过构造无上下文的文本嵌入（BoT、BoW、BoPTW），研究文本编码器中哪些信息对文本‑图像生成模型至关重要，并证明仅用单词级别信息加词序即可保持与完整嵌入相当的生成质量。

**💡 创新点**

创新点在于提出并验证“Bag‑of‑Position‑Tagged‑Words”嵌入，证明文本‑图像模型主要利用词义与词序而非全句上下文；进一步表明大规模复杂文本编码器的优势可能被高效图像模型内部实现的语言理解所取代。

**🔧 技术方法**

技术方法包括：① 对文本进行分词并平均化消除上下文；② 构造三种嵌入（BoT、BoW、BoPTW）；③ 在预训练的Diffusion Transformer模型（SD 3、FLUX.1 Schnell、FLUX.2 Klein‑4B）中直接替换原嵌入；④ 使用VLM Gemma‑3进行无偏好图像对比评估，辅以CLIP、FID、KID等指标。

**📊 数据集**

使用的数据集包括 DrawBench、GenEval（包含属性绑定、空间推理、计数等多类任务）以及 MS‑COCO 2014 验证集，用于多样化测试。

**📈 对比分析**

比较方法为让VLM在三种图像（全嵌入、BoPTW、BoW、BoT）中选择优劣，统计“非劣率”。结果显示 BoPTW 的非劣率≥65%（对比全嵌入的70–90%），BoW 与 BoT 分别超过50% 与 40%；在大多数任务类别中 BoPTW 与全嵌入几乎同等。

**⚠️ 局限性**

局限性包括：U‑Net 基模型（如 SD 2.1、SDXL）在无上下文嵌入时性能急剧下降；对极其依赖句子级语义（如文字信息）仍表现不佳；所提出的词序信息可能不足以捕捉多词习惯用语、短语级语义，未来需探索更细粒度的词组或短语嵌入。

---

## 579. Code-on-Graph: Iterative Programmatic Reasoning via Large Language Models on Knowledge Graphs

**arXiv ID:** 2606.03705 | [PDF](https://arxiv.org/pdf/2606.03705v1)

**作者:** Weiwei Ding `[一作]` (Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences), Xueqi Cheng `[通讯]` (Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于程序合成的LLM–KG集成框架CoG，能将知识图谱模式抽象为Python类并生成可执行代码，实现可扩展的多步推理。

**💡 创新点**

将KG模式映射为Python类作为抽象接口，采用迭代的规划-编码-执行循环，利用执行反馈进行自校正，从而突破传统预定义操作的灵活性和可扩展性瓶颈。

**🔧 技术方法**

Python类抽象、程序合成、可执行代码生成、迭代规划与自校正、Token Utility Rate评估等。

**📊 数据集**

WebQSP、CWQ、GrailQA 三大多跳KGQA基准。

**📈 对比分析**

与Fine‑tuning和Prompting类方法对比，在三大数据集上均取得最高的Hits@1，最高提升约10.5%，尤其在GrailQA的组合和零样本子集表现突出。

**⚠️ 局限性**

受限于底层模型的编程能力，低端编码模型难以充分发挥CoG优势；实验仅覆盖有限的LLM，未来需进一步验证。

---

## 580. When Does Latent Reasoning Help? MeRa: Metric-Space Bias for Spatial Prediction

**arXiv ID:** 2606.03727 | [PDF](https://arxiv.org/pdf/2606.03727v1)

**作者:** Zhenyu Yu `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 11360 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于度量空间的隐式推理模块(Metric-space Reasoning)，可插入任何序列编码器，在空间预测任务中显著提升性能。

**💡 创新点**

核心创新在于将空间距离信息作为可学习的注意力偏置（Metric-space Bias）融入多步推理，并证明其收敛性与表达能力提升。

**🔧 技术方法**

使用跨注意力、门控残差更新、MLP学习距离到注意力偏置映射、以及多步迭代推理框架。

**📊 数据集**

在三大真实POI数据集（NYC、TKY、CA）以及合成CLEVR空间推理任务上进行评测。

**📈 对比分析**

与多种基线（GETNext、LSTM、GeoMamba、HMST等）对比，所提出模块在所有数据集上均取得最佳或最优NDCG@10，单步无偏置推理甚至会下降，而加上度量空间偏置可提升约4.5%。

**⚠️ 局限性**

局限包括：对不同数据集最佳推理深度可能变化、理论证明基于Lipschitz等假设且未严格强制，且在更大规模或复杂度量结构下效果尚未验证。

---

## 581. Same Weights, Different Robot: A Deployment Safety View of VLA Policies

**arXiv ID:** 2606.03724 | [PDF](https://arxiv.org/pdf/2606.03724v1)

**作者:** Jianwei Tai `[一作]` (Anhui University), Jianwei Tai `[通讯]` (Anhui University)

**通讯引用:** 32 | [OpenAlex ID](https://openalex.org/A5110952853)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `9cc9baba-5356-466d-81ff-d80028d90279` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究 Vision‑Language‑Action (VLA) 策略在部署时的可执行性，提出可执行策略规格（ExecSpec）并给出闭式元数据不匹配的漂移证书，证明仅凭权重检查无法保证机器人行为一致。

**💡 创新点**

创新点在于：①将动作空间的归一化元数据视为可执行策略的一部分；②推导量化归一化键差异导致的线性动作漂移公式；③提出不需要模型推理的静态证书（ExecSpec），并在 LIBERO 仿真重放中验证其安全性指标。

**🔧 技术方法**

技术手段包括：闭式数学推导（Affine 变换）、静态可执行策略验证器、插值（α）实验、重放验证（replay-valid 集）以及对比实验（全替换、混合等）。

**📊 数据集**

使用 LIBERO 套件的四个子集：Goal、Spatial、Object、Long 的演示轨迹作为校准集与重放测试。

**📈 对比分析**

比较方法：在相同的初始状态和归一化轨迹下，仅改变元数据键，记录重放成功率。结果显示：完整替换可使成功率从 28/28 降到 2/28 或 0/28；插值实验显示成功率随 α 递减。ExecSpec 证书在无模型推理前给出漂移均值、尾部漂移等指标，能预先提示潜在失配。

**⚠️ 局限性**

限制：①仅针对量化（quantile）归一化；②实验仅在仿真重放上完成，未验证真实机器人硬件；③证书只检测动作空间元数据，未覆盖图像预处理、控制频率等其他可执行细节；④未给出统一安全阈值，需根据任务和硬件进一步校准。

---

## 582. MARS: Multi-rate Aggregation of Recency Signals for Sequential Recommendation across Sparse and Dense Regimes

**arXiv ID:** 2606.03718 | [PDF](https://arxiv.org/pdf/2606.03718v1)

**作者:** Zhenyu Yu `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 11360 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种后编码器聚合算子（Time-aware Aggregation Operator），可在任何序列编码器（Transformer或Mamba）上插入，利用真实时间戳并学习K个用户自适应指数衰减率，再通过上下文感知门融合得到最终用户表示；同时提供了基于平均序列长度的两种背骨自动选择策略。

**💡 创新点**

创新点在于将时间感知聚合从编码器中解耦出来，形成一个轻量级、可插拔的模块；利用K阶指数衰减混合、Jensen‑Shannon多样性正则以及用户条件化门实现多尺度时序建模；并在Sparse和Dense数据上分别使用Transformer和Mamba，实现跨密度的统一框架。

**🔧 技术方法**

核心技术包括：K阶指数衰减加权聚合、对时戳进行对数压缩、用户条件化衰减率调制、Jensen‑Shannon多样性正则、上下文感知门融合、以及基于平均序列长度的背骨选择逻辑。

**📊 数据集**

实验数据集有Beauty、Sports、Games、Yelp（稀疏短历史）以及ML-1M（密集长历史），共五个公开基准。

**📈 对比分析**

与10个重训练的基线（GRU4Rec、BERT4Rec、SASRec、TiSASRec、Mamba4Rec、SIGMA等）在统一RecBole评测下比较，结果显示该方法在所有五个基准上均获得最佳HR@10，稀疏数据平均提升19.7%，密集数据在ML-1M上HR@10提升3.2%（NDCG+0.9%），且在计算效率上比SIGMA减少约42% MFLOPs，位于精度‑效率 Pareto 前沿。

**⚠️ 局限性**

局限性包括：背骨选择阈值是经验设定，未验证在更广泛密度谱上的鲁棒性；正则化理论基于Hawkes过程，未给出无分布假设的可辨识性证明；在极短序列下多头聚合效果有限，且TiSASRec在NDCG/MRR上仍略优；仅评估离线下一项预测，未考虑在线部署和冷启动场景。

---

## 583. Investigating Adversarial Robustness of Multi-modal Large Language Models

**arXiv ID:** 2606.03713 | [PDF](https://arxiv.org/pdf/2606.03713v1)

**作者:** Hashmat Shadab Malik `[一作]` (Mohamed Bin Zayed University of AI), Salman Khan `[通讯]` (Mohamed Bin Zayed University of AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了多模态大型语言模型（MLLMs）的视觉对抗鲁棒性，并系统评估了多种鲁棒性提升方案；

**💡 创新点**

创新点在于①提出诊断CLIP‑alignment协议，预测鲁棒视觉编码器在多模态任务中的可迁移性；②在MLLM框架内实现端到端对抗训练，仅在鲁棒视觉骨干上有效；③提出轻量化测试时视觉随机变换作为黑盒防御；

**🔧 技术方法**

采用CLIP式对抗预训练（AdvXLCLIP）、端到端对抗训练（PGD‑AT）、CLIP‑alignment投影、图像噪声/文本提示变换等技术；

**📊 数据集**

主要使用COCO、Flickr30k、VQAv2、TextVQA、VizWiz、OKVQA等多模态视觉问答与图像描述数据集；

**📈 对比分析**

与传统plug‑and‑play鲁棒CLIP、FARE、Sim‑CLIP、AdPO等基线相比，本文模型在清晰/对抗测试中平均提升约 20–30 CIDEr 及 5–10% VQA 准确率，且在强大APGD‑Ensemble攻击下仍保持 40%+表现；

**⚠️ 局限性**

局限性包括：对抗训练需大量计算；仅针对白盒/近似黑盒攻击；对文本侧攻击的防御效果有限；对非CLIP骨干的推广尚未充分验证。

---

## 584. Ghost: Plausible Yet Unlearnable Trajectories via On-Manifold Substitution for Next-POI Privacy

**arXiv ID:** 2606.03711 | [PDF](https://arxiv.org/pdf/2606.03711v1)

**作者:** Zhenyu Yu `[一作]` (Fudan University), Shuigeng Zhou `[通讯]` (Fudan University)

**通讯引用:** 11360 | [OpenAlex ID](https://openalex.org/A5017862559)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于流形对齐的不可学习轨迹生成方法，用以在发布检查点轨迹时破坏未来位置预测能力。

**💡 创新点**

创新点在于用冻结的轨迹语言模型作为流形先验，结合对抗项进行双层优化，既保证轨迹地理与语义合理，又提升对消除攻击的鲁棒性，取代了之前的熵门随机化。

**🔧 技术方法**

技术包括：双层优化、基于Transformer的受试模型与轨迹语言模型、候选集筛选（地理、类别、速度约束）、随机softmax采样、频率表与桥式去噪等攻击模型。

**📊 数据集**

数据集为Foursquare-NYC和Foursquare-TKY（TSMC2014发布），按用户会话拆分，保留训练/验证/测试。

**📈 对比分析**

与PGD、EM、TS-UE等基线以及四种攻击（无攻击、BridgePure、频率表、bigram）比较，表现出与PGD相当的保护间隙，同时在bigram和频率表攻击下恢复准确率最低，位于保护-抗消除的Pareto前沿。

**⚠️ 局限性**

局限：仅在STAN-like Transformer受试模型下验证，跨架构迁移未知；对离散轨迹的图像UE后续技术未覆盖；BridgePure离散化实现仍待进一步评估；对泄露率极高的情况效果下降。

---

## 585. LiveBand: Live Accompaniment Generation in the Audio Domain

**arXiv ID:** 2606.03803 | [PDF](https://arxiv.org/pdf/2606.03803v1)

**作者:** Marco Pasini `[一作]` (Queen Mary University of London), George Fazekas `[通讯]`

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

开发了LiveBand系统，实现了在严格因果约束下的实时高保真音乐伴奏生成。

**💡 创新点**

创新点包括：用序列级对抗监督替代逐步预测，彻底消除教师强制导致的曝光偏差；通过每步独立高斯噪声驱动生成器，实现训练与推理一致；引入自适应梯度惩罚稳定GAN训练；采用连续潜在空间的因果音频自编码器与注意力“sink”机制。

**🔧 技术方法**

技术手段包括：因果Transformer生成器、R3GAN序列级对抗训练、连续潜在空间的因果CoDiCodec自编码器、KV缓存+注意力sink、无教师强制的并行因果训练、卷积判别器以及自适应梯度惩罚。

**📊 数据集**

主要使用公开的Slakh2100数据集（训练/测试拆分），并在内部约20k非合成多轨录音上训练CLAP条件版本。

**📈 对比分析**

与StreamMusicGen (SMG) 以及双向上限LiveBand_bid 进行对比。评估指标包括FAD_VGG/CLAP、节拍对齐F1、COCOLA（全、和声、鼓点）。LiveBand在所有指标上均优于SMG，漂移几乎为零，甚至在同步性和对齐度上有时超过基准；实时推理延迟约0.1s，RTX 3090下可达1.1×实时。主观听感测试也表明LiveBand在质量、稳定性和混音一致性方面被显著偏好。

**⚠️ 局限性**

局限性：生成音频质量仍落后于原始真实音频；受限于当前因果自编码器的音质，需研发更高保真自编码器；在更复杂或更大规模场景下的可扩展性待验证。

---

## 586. Training-Free Multi-Concept LoRA Composition with Prompt-Aware Weighting

**arXiv ID:** 2606.03792 | [PDF](https://arxiv.org/pdf/2606.03792v1)

**作者:** Georgios Tsoumplekas `[一作]` (Kingston University London), Vasileios Argyriou `[通讯]` (Kingston University London)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了两种训练无关的多概念定制方法W‑Switch和W‑Composite，通过基于目标提示语的语义重要性权重动态调节各LoRA的贡献；

**💡 创新点**

创新点在于引入prompt‑aware权重机制（PAW/PTW）并将其分别嵌入LoRA‑Switch的时间切分与LoRA‑Composite的加权聚合，实现对多概念的细粒度控制并显著降低概念干扰；

**🔧 技术方法**

使用LoRA对Stable Diffusion进行定制，结合SAM/FAN实现概念区域裁剪，利用CLIP、DINOv2和ArcFace进行图像相似度与身份保持评估；

**📊 数据集**

在ComposLoRA测试集上评测，包含3个人物、2个背景、2个服饰、2个物体和2个风格LoRA；

**📈 对比分析**

与LoRA‑Switch、LoRA‑Composite、CMLoRA等基线对比，W‑Switch在I_CLIP、I_DINO、I_ArcFace、T_CLIP、MiniCPM以及人类用户偏好评测中均取得最高或第二高分，显示显著性能提升；

**⚠️ 局限性**

仍存在身份保持上W‑Composite表现略逊，ArcFace分数相对有限；方法依赖参考图像和提示词的触发词，且在极高概念数量时仍有一定衰减。

---

## 587. SLU-2K: A Question-Based Benchmark for Semantic Evaluation of Sign Language Translation

**arXiv ID:** 2606.03788 | [PDF](https://arxiv.org/pdf/2606.03788v1)

**作者:** Zeno Testa `[一作]` (University of Modena and Reggio Emilia), Natalia Díaz-Rodríguez `[通讯]` (University of Granada)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于多选问答的手语理解评估框架 SLU-2K，并使用自动化管线从 PHOENIX‑2014T 与 CSL‑Daily 生成 2,350 条闭合式问答对，对手语翻译模型进行语义保真度评估。

**💡 创新点**

①将传统基于 BLEU/ROUGE 的评价转向以语义问答为核心的评估；②设计了多阶段自动生成、过滤、验证的管线，确保问答的可靠性；③提供了跨数据集、按语义类别划分的细粒度评估指标。

**🔧 技术方法**

利用大语言模型（DeepSeek‑Chat‑V3.2、GPT‑5.1）进行问题与干扰项生成、弱点检测、反差生成与交叉验证；使用多模态 LLM（Qwen、Gemini）评估翻译结果；构建多阶段问答验证流程。

**📊 数据集**

PHOENIX‑2014T 与 CSL‑Daily 两大公开手语数据集，分别用于生成针对天气预报与日常对话的问答对。

**📈 对比分析**

在 BLEU/ROUGE 与语义准确率两组指标上与 MLLM（Qwen、Gemini）及两大 SLT 系统（MMSTL、SpaMo）对比；发现即使 BLEU 接近，语义准确率仍低至 56‑75%，MLLM 在无细化训练下几乎随机；MMSTL 在 PHOENIX 上相对 SpaMo 领先约 6‑10 个点。

**⚠️ 局限性**

自动生成问答仍可能存在噪声与歧义；评测仅涵盖有限模型与两数据集，未覆盖更广泛场景；方法聚焦事实保留，未替代流畅度、可读性等其他质量维度。

---

## 588. AmbientEye: A Dataset for Pupil Segmentation under Natural Ambient Infrared Illumination

**arXiv ID:** 2606.03774 | [PDF](https://arxiv.org/pdf/2606.03774v1)

**作者:** Mingyu Han `[一作]` (Korea Advanced Institute of Science and Technology), Ian Oakley `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 3467 | [OpenAlex ID](https://openalex.org/A5009168826)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 AmbientEye 数据集，收集了 35 名参与者在 19 个国家的 2,518,693 张户外自然阳光下的无主动 IR 光源眼部图像，并提供了高质量的瞳孔分割标注。

**💡 创新点**

其创新点在于首次公开大规模的被动 IR 眼部数据集，突破了传统依赖主动 IR 的室内实验局限，推动了全天候户外智能眼镜眼动追踪研究。

**🔧 技术方法**

采用了 SAM2 自动分割结合人工校正的两步标注流程，并使用 DenseElNet（或 EllSeg）等最新深度分割模型进行评估。

**📊 数据集**

数据集为 AmbientEye，包含 2,518,693 张经过精细标注的图像；评测亦对 OpenEDS 与 TEyeD 等基准数据集进行对比。

**📈 对比分析**

在 AmbientEye 上的零样本评测显示，DenseElNet/EllSeg 的 IoU 从受控 IR 环境下的 0.928 降至 0.767，表现出显著性能衰减，主要受离轴畸变与阳光 IR 饱和两大失效模式影响。

**⚠️ 局限性**

实验局限于单眼、静止姿态、单一采集地点，缺乏季节、地理、运动等多样化场景的验证，未来需扩展多角度、多眼、动态采集与跨地区评估。

---

## 589. Workload acceleration by optimizing materialized view selection using local search

**arXiv ID:** 2606.03772 | [PDF](https://arxiv.org/pdf/2606.03772v1)

**作者:** Kaina Anderson `[一作]` (University of Osaka), Makoto Onizuka `[通讯]` (University of Osaka)

**通讯引用:** 1653 | [OpenAlex ID](https://openalex.org/A5030272842)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种将增量视图维护成本直接纳入整数线性规划目标并使用局部搜索的物化视图选择方法

**💡 创新点**

①将维护成本与视图效用联合优化；②通过子查询包含关系定义邻域并用频率/效用/效用/存储比三种启发式产生初始解，提升搜索质量

**🔧 技术方法**

整数线性规划（ILP）+局部搜索（邻域探索）+子查询频率、效用、效用/存储比指标 + 估算增量维护成本的三步成本模型

**📊 数据集**

RedBench基准（基于IMDb的21张表），并加入合成插入操作

**📈 对比分析**

与Naive ILP和BIGSUBS对比，指标为总时间、优化时间、视图效用和工作负载执行时间；实验表明该方法在优化时间仅约1秒的同时获得近似Naive ILP的高效能，显著优于BIGSUBS

**⚠️ 局限性**

实验仅覆盖JOB型JOIN查询，未包含嵌套查询等更复杂模式；维护成本估算仍基于简化模型，未来需在更通用工作负载和真实增量维护实现上验证

---

## 590. Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models

**arXiv ID:** 2606.03748 | [PDF](https://arxiv.org/pdf/2606.03748v1)

**作者:** Glenn Jocher `[一作]` (Ultralytics), Muhammet Esat Kalfaoglu `[通讯]` (Ultralytics)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Ultralytics YOLO26，构建统一的实时视觉模型家族，支持目标检测、实例分割、姿态估计、方向框检测和开词汇检测。

**💡 创新点**

创新点：① 采用双头设计实现 NMS‑free 端到端推理，并完全移除 Distribution Focal Loss，得到更轻量的回归头；② 引入 MuSGD、Progressive Loss 与 Small‑Target‑Aware Label Assignment (STAL) 三项协同训练策略，显著提升精度与收敛速度；③ 为实例分割、姿态估计与方向框检测设计任务专属头与损失，实现跨任务统一训练；④ 在 YOLOE 基础上构建 YOLOE‑26 开词汇版本，提供文本/视觉/无提示推理模式。

**🔧 技术方法**

使用技术包括：MuSGD（混合 Muon–SGD 优化器）、Progressive Loss（渐进权重调度）、STAL（小目标标签分配）、双头检测（one‑to‑many 与 one‑to‑one）、直接回归无 DFL、基于多尺度 prototype 的实例分割、RLE 关键点回归、长边角度参数化与角度损失、以及 MobileCLIP2 文本编码器等。

**📊 数据集**

主要数据集：Objects365‑v1 预训练，COCO（检测、分割、姿态）微调，LVIS minival（开词汇评估），DOTA‑v1.0（方向框检测）。

**📈 对比分析**

与现有实时检测器（YOLOv8、YOLOv10、RT‑DETR 等）比较，YOLO26 在五个尺寸（n/s/m/l/x）上在 COCO 上实现 40.9–57.5 mAP，T4 TensorRT 延迟 1.7–11.8 ms，显著提升 accuracy–latency Pareto 前沿；YOLOE‑26x 在 LVIS minival 上文本提示下 40.6 AP，视觉提示 38.5 AP，超越 DetCLIP‑T 等开词汇方法。

**⚠️ 局限性**

限制：仍需较长的预训练（Objects365）与多阶段训练，STAL 需要手工设定参考尺寸，双头设计在极小模型上可能导致参数占比上升；开词汇版本依赖外部 CLIP‑style 文本编码器，对语言表达的鲁棒性有限。

---

## 591. From Control Boundary to Insurance Claim: Reconstructing AI-Mediated Losses Through the CER Framework

**arXiv ID:** 2606.03777 | [PDF](https://arxiv.org/pdf/2606.03777v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 592. Template Collapse and Information-Theoretic Limits in Camera rPPG Pulse Morphology Restoration

**arXiv ID:** 2606.03802 | [PDF](https://arxiv.org/pdf/2606.03802v1)

**作者:** Achraf Ben Ahmed `[一作]` `[通讯]` (PlesmoSense SARL), Achraf Ben Ahmed (PlesmoSense SARL)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

评估了16种架构在153名受试者的3个数据集上，探讨消费者面部摄像机rPPG是否能恢复个体化脉搏波形。

**💡 创新点**

首次提出跨受试者Pearson r作为模板崩溃诊断，并通过对比证明在物理测量链上不存在可提取的个体化形态信息。

**🔧 技术方法**

结合VAE先验、监督对比学习(SupCon)、自编码器、扩散模型、信号分解等多种深度学习技术进行恢复实验。

**📊 数据集**

使用DS1、DS2（Polymate高频接触PPG）以及UBFC-PHYS/UBFC-rPPG面部视频数据集，共计153名受试者。

**📈 对比分析**

与传统的per-subject r评价不同，使用跨受试者r揭示所有架构均出现模板崩溃；SupCon在6个变体均收敛至理论极限log N，说明无个体信息可恢复；性能低于平凡均值预测。

**⚠️ 局限性**

受限于摄像机测量链的物理特性（皮肤血管低通滤波、采样率不足），无法在单周期内提取个体化脉搏形态；仅能恢复群体平均的谐波结构。

---

## 593. Backdoor Unlearning Generalization: A Path Toward the Removal of Unknown Triggers in LLMs

**arXiv ID:** 2606.03785 | [PDF](https://arxiv.org/pdf/2606.03785v1)

**作者:** Lisa Bouger `[一作]` (Inria Paris), Djamé Seddah `[通讯]` (Inria Paris)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在大型语言模型中通过训练仅针对一个后门来实现对多个后门的去除，揭示了后门去除的跨后门推广现象。

**💡 创新点**

首次提出并验证了后门去除可以跨后门传播的事实，并引入了Cross Activation Shift Distance（CASD）度量，用于量化不同后门去除导致的表示层变动相似性，从而解释了推广机制。

**🔧 技术方法**

使用持续预训练和后续去除训练，结合模型激活差分和CASD指标；实验中对8种不同类别的后门进行注入与去除；采用激活距离（余弦/ℓ2）与Spearman相关性分析。

**📊 数据集**

FineWeb Edu数据集用于后门注入与去除样本，生成的触发器和对应行为；模型族包括Qwen3（1.7B、8B）、Llama 3.2（1B）、Llama 3.1（8B）和Gaperon（1B、8B）。

**📈 对比分析**

与控制实验（无触发器训练）对比，去除一个后门能够显著降低其他后门的攻击成功率（ASR），CASD与ASR呈现高斯型负相关（Spearman≈0.9），表明当去除训练产生的激活移位与目标后门的参考移位相近时，可实现跨后门去除。

**⚠️ 局限性**

限制主要在于触发器同质化（统一为三词不常见序列），可能导致观察到的跨后门效果部分归因于触发器模板本身；CASD仅衡量相对原模型的变动相似性，未直接比较去除后模型的激活一致性，且对不同触发器长度/形式的泛化尚未验证。

---

## 594. Expert-Aware Causal Tracing of Factual Recall in Sparse MoE Language Models

**arXiv ID:** 2606.03780 | [PDF](https://arxiv.org/pdf/2606.03780v1)

**作者:** Yuetian Lu `[一作]` (LMU Munich), Hinrich Schütze `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一套专家感知因果追踪协议，用于稀疏 Mixture‑of‑Experts (MoE) 语言模型的事实回忆分析。

**💡 创新点**

从层级补丁扩展到单专家补丁，再到专家联盟补丁，首次实现了在 MoE 框架中从块级到专家级的因果定位，并揭示了不同模型对专家定位的依赖性。

**🔧 技术方法**

采用受控噪声注入+干净/受损对比补丁、MoE‑block 输出补丁、专家贡献差分估计与补丁、层级与专家级搜索，以及统计显著性检验。

**📊 数据集**

使用经过过滤的单 token 事实对照集（256 条样本），分别在 Qwen3 和 Mixtral 两个 MoE 后端上进行评估。

**📈 对比分析**

先进行层级发现再验证，比较层级和专家恢复度与特异性。Qwen3 在 L44 层实现 0.90 的恢复度，单专家 L44E069 具有 0.46 的恢复度和 0.40 的特异性；Mixtral 在 L19 层恢复度为 0.46，单专家无特异性，需多专家联盟恢复。

**⚠️ 局限性**

受限于单 token、最终位置补丁、固定路由、仅两模型的对照；未覆盖多 token、不同位置的干预，也未进行全面的专家搜索，实验需要大规模稀疏 MoE checkpoint，过滤过程可能导致样本偏倚。

---

## 595. Tool-Aware Optimization with Entropy Guidance for Efficient Agentic Reinforcement Learning

**arXiv ID:** 2606.03762 | [PDF](https://arxiv.org/pdf/2606.03762v1)

**作者:** Hongye Cao `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 12904 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种工具感知轨迹过滤与熵引导探索相结合的代理强化学习框架，实现了更稳定的工具集成训练和更高效的探索。

**💡 创新点**

创新点在于联合使用两阶段数据过滤（工具执行有效性与答案可辨别性）与针对后工具调用的熵奖励，显著提升训练稳定性与推理深度。

**🔧 技术方法**

采用 GRPO 基础算法，加入工具感知轨迹过滤、熵引导优势重塑和策略梯度更新，使用 Python 代码解释器作为外部工具。

**📊 数据集**

在七个数学推理基准（AIME、MATH500、OlympiadBench、AMC23、Hmmt、Minerva 等）以及 LiveCodeBench 上进行实验。

**📈 对比分析**

与 TIR、SimpleTIR、AEPO 等基线对比，实验显示在 1.5B、4B、7B 三个模型规模下均实现 Avg@16、Pass@16、Code_Line、Tool_Call 等指标显著提升，训练更稳定。

**⚠️ 局限性**

局限性包括对熵奖励系数和阈值的手工调参、仅使用代码解释器工具、以及对更复杂工具生态和开放式任务的泛化能力尚未验证。

---

## 596. Qwen-Image-Flash: Beyond Objective Design

**arXiv ID:** 2606.03746 | [PDF](https://arxiv.org/pdf/2606.03746v1)

**作者:** Tianhe Wu `[一作]` (Alibaba Inc), Chenfei Wu `[通讯]` (Alibaba Inc)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在本研究中，作者通过系统分析数据组合、教师引导和任务混合三大因素，重新审视了少步蒸馏（few-step distillation）在视觉生成模型中的训练流程，并基于此提出了 Qwen-Image-Flash——一个只需 4 步函数评估（NFE）的统一文本到图像生成与指令式编辑模型。

**💡 创新点**

创新点包括：①聚焦训练配方而非单纯损失函数，揭示了数据多样性并不总能提升蒸馏效果；②提出逐步多教师指导策略，在保持训练稳定的同时实现不同任务教师的协同知识迁移；③通过精细平衡 T2I 与编辑任务的混合比例，提升统一模型的编辑表现；④构建了两套新基准（T2I-Bench 与 Editing-Bench）用于评估极低 NFE 模型的生成与编辑质量。

**🔧 技术方法**

主要技术方法为：流匹配（Flow Matching）与分布匹配蒸馏（DMD）框架，结合 Qwen-Image-2.0 作为多步教师，采用 4 步 NFE 的学生网络，并在训练中加入逐步多教师权重调度与任务比例调节。

**📊 数据集**

使用的数据集包括：Qwen3 生成的 20,000 条景观、肖像和文本中心提示；T2I-Bench 共 1,800 条评测案例（600 条/类）；Editing-Bench 共 1,500 条编辑案例，覆盖场景变换、图像增强、对象操作、文本编辑、身份保留与风格迁移六大类别。

**📈 对比分析**

与基准进行比较时，作者将 4 步 Qwen-Image-Flash 与 80 步 Qwen-Image-2.0-Base 以及任务专精教师进行对比，使用 Gemini 3.1 Pro 与 GPT 5.5 两个自动偏好评测指标。结果显示，Qwen-Image-Flash 在大部分类别的平均得分上与 80 步教师持平或略优，尤其在平衡 5:5 的 T2I‑编辑混合训练下，编辑基准得分显著提升。

**⚠️ 局限性**

局限性包括：对细粒度文本渲染（如极小文字、复杂海报排版）的能力仍不足；在极少采样步数下，部分生成结果出现残留噪点，尤其是纯白背景；对高度结构化的编辑任务的鲁棒性还有待提升。

---

## 597. Compress then Merge: From Multiple LoRAs into One Low-Rank Adapter

**arXiv ID:** 2606.03723 | [PDF](https://arxiv.org/pdf/2606.03723v1)

**作者:** Zhengbao He `[一作]` (Shanghai Jiao Tong University), Xiaolin Huang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10322 | [OpenAlex ID](https://openalex.org/A5005338317)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究如何将多个 LoRA 适配器在满足低秩约束的前提下合并为单一 LoRA 更新。

**💡 创新点**

提出 Compress‑then‑Merge (CtM) 逆向管道：先学习共享低维子空间并投影到 r×r 坐标，再在该空间内合并，从而保证最终更新始终低秩并避免后置截断的误差。

**🔧 技术方法**

使用子空间学习（重权重的低秩重建）、Tucker 分解、HOSVD/HOOI、核心空间加速、以及常见的合并规则（TIES、DARE‑TIES、KnOTS 等）。

**📊 数据集**

在 CLIP ViT‑B/32 视觉任务（8 个数据集）和 LLaMA3‑8B 语言任务（6 个 NLI 任务）上进行评估；实验还涵盖多模态生成、不同 LoRA 数量与异质设置。

**📈 对比分析**

与 Merge‑then‑Compress（MtC）以及现有低秩输出方法（LoRA‑LEGO、RobustMerge、Iso‑CTS）比较，CtM 在多任务平均准确率上普遍高于 MtC，且在大多数基准上逼近甚至超过全参数合并的性能。

**⚠️ 局限性**

若任务间更新几乎正交，单一低秩 LoRA 无法充分表达，CtM 仍会受到信息损失限制；且仍需手动设定目标秩、重权重系数与基准合并规则。

---

## 598. Proof-Refactor: Refactoring Generated Formal Proofs into Modular Artifacts

**arXiv ID:** 2606.03743 | [PDF](https://arxiv.org/pdf/2606.03743v1)

**作者:** Yiming Fu `[一作]` (Southern University of Science and Technology), Kun yuan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 Proof-Refactor 框架，将 LLM 生成的 Lean 证明拆分成四个阶段（提取、辅助声明设计、证明、修复），通过流程化重构提升证明的可读性、模块化和可维护性。

**💡 创新点**

创新点在于：①把证明重构视为过程而非单一指标优化；②引入外部模型协助片段选择和辅助声明设计；③自定义 `extract` tactic 将证明片段转为可验证的独立命题；④四阶段流水线将高层决策与低层 Lean 交互分离。

**🔧 技术方法**

技术包括：Claude Code + Lean LSP MCP；自定义 Lean `extract` tactic；外部模型 Gemini 3.1 Pro 用于片段选择和辅助声明设计；Rubric‑based LLM‑as‑judge 评估方法；实验使用 PutnamBench 与 Putnam2025 自动生成的 Lean 证明。

**📊 数据集**

数据集：96 条 PutnamBench 自动生成证明（行数 50–500，平均 160 行）以及 12 条 Putnam2025 自动生成证明。

**📈 对比分析**

与基线 Claude Code + lean4‑skills 进行对比，采用 rubric‑based 评估并辅以人类审核。实验显示，Proof‑Refactor 在平均质量得分上提升 0.45（PutnamBench）与 0.31（Putnam2025），在签名质量、结构、重用性和可读性等维度取得最大改进；长度削减率波动，核心目标是质量提升。

**⚠️ 局限性**

局限性：仅限单文件重构，仅新增声明；未支持多文件或库级重构；依赖外部模型导致成本较高；早期阶段错误会级联；评估依赖自动化 rubric，缺乏大规模专家审查。

---

## 599. Re-Ranking Through an Attribution Lens for Citation Quality in Legal QA

**arXiv ID:** 2606.03728 | [PDF](https://arxiv.org/pdf/2606.03728v1)

**作者:** Mohamed Hesham Elganayni `[一作]` (Technical University of Munich), Selim Saleh `[通讯]` (Technical University of Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练轻量级交叉编码器重排检索结果，将C‑LIME扰动归因分数作为训练信号，显著提升法律QA系统的引用质量。

**💡 创新点**

将事后解释工具C‑LIME转化为训练信号，用归因分数训练重排器，并发现跨模型训练的重排器会收敛到共享的相关性信号。

**🔧 技术方法**

使用C‑LIME扰动归因、交叉编码器(ms‑marco‑MiniLM‑L‑6‑v2)、多模型（Mistral‑7B、Llama‑3‑8B）实验、AQuAECHR基准、ROUGE‑L、BERTScore、TRUETeacher、NLI等技术。

**📊 数据集**

AQuAECHR benchmark（1116个ECHR问答对及其金标注引用）。

**📈 对比分析**

通过与基线检索‑生成、预训练重排器对比，使用六项评价指标（ROUGE‑L、BERTScore、Claim Recall、Citation Faithfulness、Exact Match F1、NLI Citation Similarity），C‑LIME重排器在引用可信度提升约5pp，NLI相似度提升约6.7pp，答案准确度变化不到0.5pp。

**⚠️ 局限性**

归因分数基于模型行为，可能放大模型偏差；评估主要依赖自动NLI指标，缺乏专家验证；方法在ECHR领域表现良好，跨语言、不同司法辖区或更大模型的通用性仍待验证。

---

## 600. Don't Trust Us: A privacy-by-design android malware detection pipeline

**arXiv ID:** 2606.03714 | [PDF](https://arxiv.org/pdf/2606.03714v1)

**作者:** Emmanuele Massidda `[一作]` (University of Cagliari), Giorgio Giacinto `[通讯]` (University of Cagliari)

**通讯引用:** 7138 | [OpenAlex ID](https://openalex.org/A5075367917)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了一个从设计上实现隐私保护的 Android 恶意软件检测流水线，包含静态特征提取、双阈值 SVM 决策以及仅在必要时才在隔离沙盒中进行动态分析。

**💡 创新点**

创新点在于：①将隐私保护作为系统的根本设计目标，避免在任何阶段收集敏感用户数据；②采用双阈值拒绝机制，让模型在不确定时主动放弃决策，减少误报和漏报；③通过沙盒化的动态分析保证即使需要运行程序也不暴露真实用户信息。

**🔧 技术方法**

技术包括：Drebin 静态特征提取、线性 SVM（支持向量机）双阈值判定、ANY.RUN 沙盒化动态分析、基于时间拆分的训练/验证/测试流程。

**📊 数据集**

使用了 VirusTotal 收集的 Android APK 数据集，按 2024 年和 2025 年第一季度分布的时间拆分，共 38,992 个训练样本、8,861 个验证样本、4,760 个测试样本，其中恶意率约 42.9%。

**📈 对比分析**

对比了三种设置：原始 Drebin 单阈值、单阈值调优、双阈值。双阈值在测试集上达到 0.89 的准确率、0.87 的恶意 F1 分数，仅 6.7% 的样本被推迟到动态分析。相比之下，单阈值调优得到 0.86 的准确率，原始设置仅 0.83。双阈值在保持高精度的同时显著减少了误判。

**⚠️ 局限性**

局限性包括：①动态分析阶段仍依赖外部沙盒（ANY.RUN），自动化交互效果差，需人工干预；②对边缘样本的误报率较高，特别是功能相似的合法应用被误判为恶意；③未对沙盒逃逸和高级欺骗技术做进一步评估；④缺乏大规模自动化动态交互方案。

---

## 601. Worth Remembering: Surprise-Gated Robot Episodic Memory

**arXiv ID:** 2606.03787 | [PDF](https://arxiv.org/pdf/2606.03787v1)

**作者:** Nicolas Gorlo `[一作]` (Massachusetts Institute of Technology), Luca Carlone `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 12837 | [OpenAlex ID](https://openalex.org/A5042157108)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出基于贝叶斯惊讶的机器人短期记忆门控机制，在V‑JEPA‑2潜在空间中计算惊讶度，选择性存储高价值事件，增强DAAAM 4D场景图记忆，并用于机器人问答与事件边界检测。

**💡 创新点**

首次将贝叶斯惊讶作为无监督、因果的记忆门控准则，并结合V‑JEPA‑2无监督视频预训练模型，集成到实时机器人记忆框架中，实现长期问答与通用事件边界检测的显著提升。

**🔧 技术方法**

使用V‑JEPA‑2视频联合嵌入预测架构、对角高斯局部预测模型、KL散度惊讶度计算、阈值+NMS触发、DAAAM 4D场景图、Perception‑Encoder文本‑图像嵌入和GPT‑5‑mini推理等技术。

**📊 数据集**

在OC‑NaVQA（CODa）数据集上进行长周期空间时间问答实验，在Kinetics‑400（及TAPOS）数据集上进行通用事件边界检测实验。

**📈 对比分析**

与DAAAM统一/随机记忆、ReMEmbR、Concept‑Graphs等基线比较；在OC‑NaVQA上问答准确率提升12%（从0.711到0.796），位置误差和时间误差分别降低12.4%和15.7%；在Kinetics‑400 GEBD上F1得分超越传统无监督方法，接近或超过有监督/离线方法。

**⚠️ 局限性**

依赖非因果V‑JEPA‑2，使用简单的高斯预测模型；存储的事件仅为短帧窗口，缺乏对长期时间演化的捕捉；可能因重复出现对象继续触发存储；对不同域视频的适应性有限。

---

## 602. Entropy Gate: Entropy Quenching for Near-Lossless Token Compression in LLM Pipelines

**arXiv ID:** 2606.03739 | [PDF](https://arxiv.org/pdf/2606.03739v1)

**作者:** Justice Owusu Agyemang `[一作]` (Kwame Nkrumah University of Science and Technology), James Dzisi Gadze `[通讯]` (Kwame Nkrumah University of Science and Technology)

**通讯引用:** 800 | [OpenAlex ID](https://openalex.org/A5076343953)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Entropy Gate 作为统一的令牌压缩框架，通过熵冷却策略在输入和输出端剔除低信息量的词元，显著减少 LLM 的令牌消耗。

**💡 创新点**

创新点包括：①多因子信息能量模型（统计、结构、位置）与能量平方放大；②熵冷却调度与保真门；③自校准领域能量；④输出侧熵冷却与块级去重；⑤在数学上证明了能量排序最优、熵冷却单调性以及压缩极限与信息理论的对应关系。

**🔧 技术方法**

核心技术：TF‑IDF 多块统计、正则表达式结构分类、指数衰减位置权重、能量平方放大、Boltzmann 生存概率、熵冷却调度、能量加权相似度保真门、输出端压缩与块去重。

**📊 数据集**

数据集：自构造的 5 类 Prompt（代码审计、SQL 生成、文档生成、系统提示、工具调用）以及公开基准 MMLU、HumanEval、GSM8K，用于评估压缩率、语义保真和任务正确率。

**📈 对比分析**

比较方法：与 LLMLingua、Selective Context、Gist Tokens 等基线对比；Phase 1 在 5 类 Prompt 上实现 40–60% 压缩率、能量相似度 S_E > 0.80；输出压缩可达 75%；在 11 题 Benchmark 上压缩后仍保持 8/9 题的任务正确率，且在 4 级评估中平均质量得分 4.1/5。

**⚠️ 局限性**

局限性：Phase 1 依赖启发式能量估计，无法处理多语种、右到左文本或非英语编程语言；空格分词导致复合 token 误删；正则结构分类过于粗糙，缺少对复杂语法的识别；在工具调用的多轮代理中若压缩过度可能引起循环；总体上需进一步结合模型概率和子词 tokenization 以提升鲁棒性。

---

## 603. Unveiling the Structure of Do-Calculus Reasoning via Derivation Graphs

**arXiv ID:** 2606.03719 | [PDF](https://arxiv.org/pdf/2606.03719v1)

**作者:** Clément Yvernes `[一作]` (Univ Grenoble Alpes), Eric Gaussier `[通讯]` (Univ Grenoble Alpes)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出并实现了“导数图”（derivation graph）概念，用以系统化对 Pearl 的 do‑calculus 规则的应用，揭示不同等价因果表达之间的结构与关系。

**💡 创新点**

创新点在于将 do‑算子规则的组合映射为图结构，并证明任何等价表达只需最多四步（两次 R₂ 与两次 R₃），同时给出图形判定等价性的判据以及多重可识别公式的可枚举性。

**🔧 技术方法**

主要技术包括图论与 d‑separation、对三条 do‑calculus 规则的命题证明、Canonical 序列化以及统计估计与 Bootstrap 方差评估。

**📊 数据集**

实验使用线性高斯模拟数据以及真实的蛋白信号网络（Sachs 数据集）进行验证。

**📈 对比分析**

通过对同一因果效应生成多种等价识别公式，并在模拟与真实数据中比较其方差，结果表明等价公式在统计性能上差异显著，某些公式方差更小、估计更稳健。

**⚠️ 局限性**

局限性包括：仅聚焦 do‑算子规则的等价性，未处理概率代数中可无限插入的身份项；在更复杂图结构下，多重识别公式枚举可能指数级扩展，且当前方法无法直接覆盖 ID 算法完整搜索空间。

---

## 604. Limit Analysis of Graph Neural Networks with Wireless Conflict Graphs

**arXiv ID:** 2606.03794 | [PDF](https://arxiv.org/pdf/2606.03794v1)

**作者:** Romina Garcia Camargo `[一作]` (University of Pennsylvania), Alejandro Ribeiro `[通讯]` (University of Pennsylvania)

**通讯引用:** 16365 | [OpenAlex ID](https://openalex.org/A5078862959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `67630363-6be0-4f51-ab05-7198250671a5` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在稀疏随机几何图（RGG）的冲突图上，将图神经网络（GNN）从小规模图迁移到大规模图的可迁移性，并在无线链路调度任务上进行验证。

**💡 创新点**

创新点在于给出了RGG与确定性网格图（DGG）之间的接近性定理，证明了当RGG仅为DGG的小幅扰动时，GNN的性能损失可控；并扩展了传统CNN的尺度迁移理论到GNN。

**🔧 技术方法**

使用的技术包括图卷积网络（带LeakyReLU和Sigmoid）、状态增强的GNN（SAGNN）、随机几何图与确定性网格图的构造、以及对冲突图邻接矩阵的零填充与对比分析。

**📊 数据集**

使用的实验数据集为从DGG加入不同方差噪声得到的RGG图，规模分别为约500条边与2500条边的图，训练集包含100张图，测试集为多组不同规模的100张图。

**📈 对比分析**

与基准FPLinQ进行对比，SAGNN在保持约20‑25%活跃链路率的同时，平均速率接近最大独立集上限；在运算时间上比FPLinQ快4倍（K≈500）至30倍（K≈2500），且可实现分布式部署。

**⚠️ 局限性**

局限性包括对RGG与DGG距离小的假设（噪声方差需很低）以及仅在合成网络上验证，缺乏对真实无线网络场景的实证。

---

## 605. Exploring Adversarial Robustness and Safety Alignment in Multilingual Multi-Modal Large Language Models

**arXiv ID:** 2606.03793 | [PDF](https://arxiv.org/pdf/2606.03793v1)

**作者:** Hashmat Shadab Malik `[一作]` (Mohamed Bin Zayed University of AI), Salman Khan `[通讯]` (Mohamed Bin Zayed University of AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究多语言多模态大型语言模型在对抗鲁棒性和安全性上的表现，构建60k多语言评测集并系统评估Palo、Parrot、Qwen3-VL等模型。

**💡 创新点**

首次在12种语言下系统考察梯度攻击跨语言迁移性与安全性缺陷，提出“安全即失败”(safety‑by‑failure)现象，并对比基于指令微调与全阶段多语言训练的模型差异。

**🔧 技术方法**

采用白盒梯度攻击(APGD, PGD)、视觉破坏优化、LLM-as-a-judge、LLM对抗视觉解码，结合多模型翻译‑验证流水线和人类审校。

**📊 数据集**

以COCO、Flickr30k、LLaVA‑Bench、RealToxicityPrompts、MM‑SafetyBench等英语基准，通过机器翻译扩展至12种语言。

**📈 对比分析**

对比梯度攻击下的CIDEr、安全违规率与拒绝率等指标，发现攻击跨语言迁移强且指令微调模型表现出低违规率但实际安全性低；全阶段多语言模型表现更真实安全。

**⚠️ 局限性**

主要限制包括只评估开源可白盒模型，攻击仅限梯度基准，未覆盖API/商用模型，翻译质量依赖LLM与人工，且未深入探究防御策略。

---

## 606. Reasoning over Grammar: Can Synthetic Linguistic Reasoning Traces Enhance Low-Resource Machine Translation?

**arXiv ID:** 2606.03782 | [PDF](https://arxiv.org/pdf/2606.03782v1)

**作者:** Renhao Pei `[一作]` (Ellis Institute Finland), Shaoxiong Ji `[通讯]` (Ellis Institute Finland)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究低资源语言翻译，利用LLM结合自动生成的结构化语言推理轨迹作为翻译指导；

**💡 创新点**

创新点在于将Universal Dependencies树形、词典和语法规则自动合成逐步推理轨迹，并系统比较ICL、SFT和RFT三种使用方式，发现ICL最能提升翻译质量；

**🔧 技术方法**

技术包括LLM（Qwen3、Gemma）在in‑context学习、监督微调（SFT）和强化学习（RFT，GRPO）等；同时构建基于UD的推理模板、词典和语法规则库；

**📊 数据集**

数据集为Xibe（≈30k说话人）和Chintang（≈5k说话人）的UD treebank，配合相应的词典与手工编写的语法规则；

**📈 对比分析**

实验采用BLEU、chrF、SBERT和LLMaJ评估，ICL加推理轨迹后BLEU/chrF提升显著（最高+19.7 SBERT、+23.4 LLMaJ），SFT和RFT改进有限，且易出现推理错误；

**⚠️ 局限性**

局限性包括RL探索空间受限、奖励函数未直接评估句法分析、模型在生成推理内容时仍错误，导致最终翻译提升不大。

---

## 607. $π$Creds: Privately Inferred Credentials

**arXiv ID:** 2606.03771 | [PDF](https://arxiv.org/pdf/2606.03771v1)

**作者:** Samuel Breckenridge `[一作]` (Cornell Tech), Ari Juels `[通讯]` (Cornell Tech)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了一种基于可信LLM推理、可对已认证数据生成隐私保护、兼容传统系统的去中心化可验证凭证πCreds。

**💡 创新点**

创新点在于利用LLM的语义推理能力扩展凭证可认证的范围，并首次形式化了Source‑Constrained Adversarial Example (SCAE) 与Authenticated Covert Predicate Poisoning (ACPP) 两类基于LLM的安全威胁。

**🔧 技术方法**

核心技术包括可信LLM推理、认证数据流、去中心化凭证链（可能基于区块链或分布式账本）以及安全模型评估框架。

**📊 数据集**

使用了真实的金融、健康、邮件和代码来源数据集，并在金融领域的产品专业知识凭证上开展实证研究。

**📈 对比分析**

通过对SCAE与ACPP攻击场景的实验评估，证明πCreds在误导性凭证抵御能力和隐私泄漏风险控制方面优于传统基于零知识证明的方案，且在性能上保持可接受的推理延迟与验证效率。

**⚠️ 局限性**

局限性包括对LLM推理的可信性假设、对攻击者模型的简化假设以及在极大规模数据或多租户环境下的可扩展性与公平性问题。

---

## 608. E2LLM: Towards Efficient LLM Serving in Heterogeneous Edge/Fog Environments

**arXiv ID:** 2606.03770 | [PDF](https://arxiv.org/pdf/2606.03770v1)

**作者:** Truong-Thanh Le `[一作]` (University of Oslo), Peiyuan Guan `[通讯]` (University of Oslo)

**通讯引用:** 2281 | [OpenAlex ID](https://openalex.org/A5047837985)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 E2LLM 框架，在 Edge/Fog 资源受限环境下实现 LLM 的高效部署，通过模型复制、分阶段角色分配和模型并行化来提高吞吐量和降低延迟。

**💡 创新点**

创新点在于将全模型复制到多个节点组，同时对 Prefill/Decoder 阶段分别分配复制实例，并结合遗传算法聚类与动态规划求最优分区，显著降低瓶颈并提升响应性。

**🔧 技术方法**

使用遗传算法（GA）进行设备聚类与复制组划分，动态规划（DP）实现模型层级分区，模型并行化（pipeline parallelism），以及 Join Shortest Queue（JSQ）负载均衡。

**📊 数据集**

使用 HuggingFace 的 lz1bytedance/LongReason 数据集（标准和自定义扩展版）以及 20B 参数的 GPT‑OSS‑20B 语言模型进行实验。

**📈 对比分析**

对比基线 SplitWise（改进版），在七机节点的 Edge/Fog 集群上测试多种到达率。E2LLM 在高负载下解码吞吐量提升约 2 倍，等待时间降低 50% 以上，且在低负载时仍保持更低的延迟。

**⚠️ 局限性**

局限性包括对网络带宽假设较高（统一 920Mbps），未考虑动态负载变化的自适应调整，仅在固定硬件配置上验证；且对大规模复制集群的扩展性与能耗分析缺失。

---

## 609. Trading Human Curation for Synthetic Augmentation in RLVR

**arXiv ID:** 2606.03800 | [PDF](https://arxiv.org/pdf/2606.03800v1)

**作者:** Akshansh `[一作]` (Pareto AI), Mark E. Whiting `[通讯]` (Pareto AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了使用预设、门控过滤的自动化任务增强来替代人类手工编写任务在 RLVR 训练中的角色，并评估其在数据成本、性能上的替代效果。

**💡 创新点**

首次系统定义并量化成本调整后的替代率 ρ_cost，构建门控过滤的任务增强管道，并证明在数据成本上可与人类任务等价甚至优于；同时给出增强任务的端到端成本质量边界。

**🔧 技术方法**

采用预设变异生成与多路并行的增强技术，门控过滤（Verifiers、solve、distinct、faithful、informative）结合 GRPO 强化学习，使用 Qwen3.5‑27B 语言模型，并在 Docker 沙盒中执行任务。

**📊 数据集**

使用 10 个手工基底任务 + 80 或 319 个自动生成的增强任务作为训练集，评估时采用 10 个 held‑out 基准：HumanEval, DS‑1000, IFEval, MATH500, BBEH, MMLU‑Pro, GPQA Diamond, BFCL v4, BFCL multi‑turn, τ²‑bench retail。

**📈 对比分析**

与 97 人类任务基准 H97_A0 以及计算匹配扩展对照 H97_A0^†、H10_A80^† 进行比较，主要指标为 10 项基准的平均 pass@1。80 增强版本在数据成本等价下与 97 人类任务平均相差 ≤0.20%；319 增强版本在成本等价下平均提升 +0.96%（8/10 基准获胜），但在统计误差范围内。

**⚠️ 局限性**

局限性包括：仅单一训练种子、仅使用 Qwen3.5‑27B、任务规模有限（10–319）、未包含训练/评估/判别器完整成本、缺乏多任务/多模型泛化验证、门控筛选可能出现误报/漏报、数据集偏向数据科学任务、评估集中于单轮任务、缺少多 seed 训练和更细粒度的计算匹配实验。

---

## 610. When to Re-Plan: Subgoal Persistence in Hierarchical Latent Reasoning

**arXiv ID:** 2606.03741 | [PDF](https://arxiv.org/pdf/2606.03741v1)

**作者:** Ayushi Chadha `[一作]` `[通讯]` (Independent Researcher), Ayushi Chadha (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在隐式推理框架下引入了可持续的方向性子目标（subgoal），并通过 HRM 进行多层次的子目标注入与对齐损失来提升推理质量。

**💡 创新点**

核心创新在于把子目标的持续时间 P 作为稳定-适应权衡的关键控制参数，并证明在 P∈[3,6] 时子目标能形成可组合的中期意图；同时发现对齐权重 λ 有极窄最优区间（≈0.05），并通过精细消融证明过大 λ 时干扰源来自学习到的子目标方向。

**🔧 技术方法**

使用了 Hierarchical Reasoning Model（HRM）与 feudal-style 管理-工人接口；引入方向性子目标投影、偏置注入、余弦对齐损失；在模型训练中加入 λ 加权的对齐损失；采用 AdamATan2 优化器、RoPE、SwiGLU 等技术。

**📊 数据集**

主要数据集为 ARC‑AGI 与 ConceptARC（训练时进行数据增强），以及概念性子集 ConceptARC‑mini 作为跨任务验证。

**📈 对比分析**

通过对比 LM 损失（token 级交叉熵）和准确率，发现最佳配置（P=3，λ=0.05）在 ARC‑AGI/ConceptARC 上 LM 损失降至 1.544，显著优于无子目标基线 1.640；在 ConceptARC‑mini 上提升约 0.4%。对比实验中使用相同基线模型、相同训练设置，显示子目标机制在最佳参数下确实带来性能提升。

**⚠️ 局限性**

局限性包括：提升幅度仅为数百分之一；实验仅覆盖 ARC‑style 任务，未验证更广泛任务；消融实验使用单一随机种子，缺乏多种随机性检验；未进行子目标对隐藏状态几何影响的深入表示分析。

---

## 611. When Graph Tokens Sink: A Mechanistic Analysis of Graph Language Models

**arXiv ID:** 2606.03712 | [PDF](https://arxiv.org/pdf/2606.03712v1)

**作者:** Ding Zhang `[一作]` (University of Virginia), Chirag Agarwal `[通讯]` (University of Virginia)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文对Graph Language Models（GLMs）中的图令牌（graph tokens）进行机制化分析，聚焦于所谓的“图Sink”行为；

**💡 创新点**

创新点在于首次将注意力Sink概念迁移到图令牌，揭示图Sink在激活层级与注意力分配、功能重要性之间存在显著脱钩；

**🔧 技术方法**

采用激活峰值检测、位置分布分析、Attention分布可视化、Pruning/Swap/Reposition干预、Logit Lens等解释方法，对LLaGA和TEA-GLM进行实证；

**📊 数据集**

使用标准图数据集Cora、Arxiv、PubMed，涵盖节点分类与链路预测两类任务；

**📈 对比分析**

与传统GLM的“直接使用图令牌”方法对比，实验显示移除或交换Sink令牌对性能影响极小，而随机删除非Sink令牌会产生更大下降，表明Sink令牌并非核心信息载体；

**⚠️ 局限性**

局限性包括仅评估了两种GLM架构，未探究更大模型或不同令牌构造策略；Sink行为对不同预训练任务的迁移性和对图结构捕获机制的深层原因仍待进一步研究。

---

## 612. Beyond Compression: Quantifying Spectral Accessibility in Vision Representations

**arXiv ID:** 2606.03795 | [PDF](https://arxiv.org/pdf/2606.03795v1)

**作者:** Akayou A. Kitessa `[一作]` (Fordham University), Yijun Zhao `[通讯]` (Fordham University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究视觉语言模型（CLIP、DINOv2）中视觉表示的空间频率结构变化，使用线性探测器评估各层对频谱信息的可线性可访问性，并引入 Residual Spectral Loss（RSL）来区分压缩与学习导致的频谱变化。

**💡 创新点**

提出基于频率可访问性的评估框架和 RSL，揭示中间层提升频谱可访问性，CLIP 投影层频谱中性，而 DINOv2 的 attention pooling 引起结构化频谱损失，指出池化机制对频谱属性的主导作用。

**🔧 技术方法**

线性 Ridge 探测（预测分段频带能量）、FFT 与 Hann 窗口生成频谱、随机投影基线（Johnson–Lindenstrauss 方案）、Bootstrap 置信区间估计。

**📊 数据集**

ImageNet-1K（10k 图像）和 MS‑COCO（10k 图像）子集。

**📈 对比分析**

通过与随机投影基线比较计算 RSL，并与不同层级的 Pearson 相关系数对比；结果显示 CLIP 投影对频谱无额外影响，DINOv2 投影导致显著频谱损失；中间层可访问性提升，整体呈非单调深度轨迹。

**⚠️ 局限性**

仅使用线性探测，未捕获非线性信息；仅处理灰度、径向频带；仅评估 ViT 结构和自然图像；缺乏因果分析，仅为观察性；未考虑颜色和方向频率。

---

## 613. HybridThinker: Efficient Chain-of-Thought Reasoning via Compressed Memory and Transient Thought Steps

**arXiv ID:** 2606.03768 | [PDF](https://arxiv.org/pdf/2606.03768v1)

**作者:** Xin Liu `[一作]` (Northeastern University), Tong Xiao `[通讯]` (Northeastern University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种混合思考器 HybridThinker，兼顾压缩思考步骤与临时保留细粒度信息的方式，用以提升大型语言模型的推理效率与准确率。

**💡 创新点**

创新点在于：①在推理阶段保持一定数量的完整思考步骤供后续步骤使用，②设计了混合注意力训练策略（Hybrid Attention），通过随机将步骤分配为 Shortcut Attention 或 Bottleneck Attention，解决了传统训练中因直接路径导致的压缩瓶颈学习不足的问题。

**🔧 技术方法**

技术实现包括：使用固定数量的可学习内存令牌压缩每个思考步骤的 KV 缓存、动态维护 KV 缓存容器、以及自定义的三种注意力掩码（Shortcut、Bottleneck 与 Hybrid）来训练模型。

**📊 数据集**

实验数据集涵盖四大推理基准：GSM8K、MMLU、GPQA 与 BBH，使用 Qwen2.5‑7B 与 Llama3.1‑8B 两个 7B 级模型进行评估。

**📈 对比分析**

与 LightThinker、Distill‑R1、Vanilla、以及 KV 缓存剪枝方法 H_2O、SepLLM 等基线相比，HybridThinker 在保持与未压缩基线相当的准确率（平均提升 5.8 分）的同时，峰值 KV 缓存使用量下降 61.7%，推理时间缩短 20.3%，并且相较于仅压缩方法显著减少冗余推理步骤。

**⚠️ 局限性**

局限性包括：临时保留思考步骤会略微增加峰值 KV 缓存使用；方法假设存在明确的步骤分隔符，难以直接推广到无边界的自由形式推理场景。

---

## 614. Framing Migration News with LLMs: Structured CoT as a Support for Human Interpretation

**arXiv ID:** 2606.03761 | [PDF](https://arxiv.org/pdf/2606.03761v1)

**作者:** David Alonso del Barrio `[一作]` (Idiap Research Institute), Daniel Gatica-Perez `[通讯]` (Idiap Research Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于结构化Chain‑of‑Thought (SCoT) 的提示方法，利用本地可部署的Llama3‑8B对迁徙新闻进行可解释的帧分类，并通过人机评估检验模型推理的可理解性与影响力。

**💡 创新点**

创新点在于将推理过程拆解为多步结构化描述，使模型输出可追溯、透明；同时兼顾资源受限环境的本地部署与人机交互，促进对主观帧分析的批判性审视。

**🔧 技术方法**

技术手段包括使用开源LLM Llama3‑8B、定制的SCoT提示模板、零样本/少样本对照实验，以及利用LimeSurvey进行的用户主导评估。

**📊 数据集**

实验数据来自Media Frame Corpus（MFC）中迁徙主题的子集，包含约700篇新闻（约11%原始集），每篇文章标注14种预定义帧（去除“Other”），构成平衡与不平衡两版。

**📈 对比分析**

与零样本、5例少样本基线比较，SCoT在平衡集上的一致率为36%、宏F1为0.28；在不平衡集上一致率42%、宏F1为0.29；人类评估显示模型推理的合理性平均得分为4.1/5，并在部分样本中产生了显著的说服效果。

**⚠️ 局限性**

局限性包括仅聚焦迁徙主题、样本量和人机评估规模有限、帧定义存在重叠导致单标签标注难以充分反映多重解读，以及SCoT虽提升可解释性但对分类精度提升有限。

---

## 615. Neural Navigation Functions for Zero-Shot Generalizable Motion Planning

**arXiv ID:** 2606.03756 | [PDF](https://arxiv.org/pdf/2606.03756v1)

**作者:** Benjamin D. Shaffer `[一作]` (University of Pennsylvania), M. Ani Hsieh `[通讯]` (University of Pennsylvania)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于可学习的椭圆PDE的导航函数框架（Neural‑NF），通过学习几何相关的局部系数在全局PDE求解中生成可直接使用的反馈策略；

**💡 创新点**

创新点在于将学习目标从直接预测策略/值函数转移到学习PDE的局部运算符系数，结合全局PDE求解保证无局部极小值、碰撞避免和单调下降，且实现零样本几何迁移与高数据效率；

**🔧 技术方法**

核心技术包括：
- 基于网格拉普拉斯算子提取的四通道几何特征（HKS、Poisson、目标指示、方向对齐）；
- 点级神经网络映射至局部电导率k和运行成本c；
- 最低阶有限元外积算子（FEEC）离散化PDE并通过线性求解器求解；
- 通过逆向传播在离散线性求解器上训练；
- 线性可解随机最优控制解释；
- 旋转等变性与局部稳定性设计。

**📊 数据集**

使用的典型数据集包括二维合成几何（Square、Disk、Maze、HouseExpo）和真实城市街道数据集（City Streets），并在ID（与训练分布相同）与OOD（不同几何分布）上进行评估；

**📈 对比分析**

与直接预测模型（PointNet、GNN、DeepONet、GNO、MGN、Intrinsic）在ID与OOD两种评估下对比，Neural‑NF在ID上平均L2误差从~40%降至3.7%，在OOD上从~50%降至7.4%，提升约5×；同时在策略余弦相似度、梯度光滑度等导航相关指标上明显优于基线；单图训练即可实现多图零样本迁移；

**⚠️ 局限性**

局限性在于受限于椭圆PDE规划器类，无法直接适配需要不同PDE或更复杂机器人动力学的任务；此外对高维连续空间或非欧几里得几何的推广仍需进一步研究。

---

## 616. Fast TetraBFT: Optimizing Latency Where It Matters

**arXiv ID:** 2606.03754 | [PDF](https://arxiv.org/pdf/2606.03754v1)

**作者:** Antonio J. Fernández-Pinto `[一作]` (IMDEA Software Institute), Alexey Gotsman `[通讯]` (IMDEA Software Institute)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出 Fast TetraBFT，一种在视图0使用快速路径包装现有 TetraBFT 的未认证拜占庭一致性协议，优先在同步网络下实现 3δ 的最佳好案例延迟。

**💡 创新点**

创新点在于证明可以通过一个简单的快通道包装而非复杂的 Forget-IT 方案实现最佳好案例延迟，同时保持 O(n²) 的通信复杂度和有限空间实现。

**🔧 技术方法**

使用点对点身份验证通道、视图同步器（如 FastSync）以及基于投票与锁定机制的 TetraBFT 逻辑；快通道通过一次性投票、提交和锁定来实现 3δ 延迟。

**📊 数据集**

本工作未使用具体数据集；所有评估均基于理论分析和协议安全性证明。

**📈 对比分析**

与 Forget-IT、PBFT、HotStuff 等协议比较，Fast TetraBFT 在好案例延迟上与 Forget-IT 等效，但实现更简洁；在通信复杂度上保持 O(n²)，优于 PBFT 的 O(n³)。

**⚠️ 局限性**

局限性包括：仅在视图0的快通道提供最佳延迟；在恶意环境下仍需等待同步稳定；对 TetraBFT 结构有一定依赖，无法直接迁移至完全不同的协议框架。

---

## 617. KletterMix: Climbing Toward High-Quality German Pretraining Data

**arXiv ID:** 2606.03773 | [PDF](https://arxiv.org/pdf/2606.03773v1)

**作者:** Maurice Kraus `[一作]` (Technische Universitaet Darmstadt), Kristian Kersting `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究提出了 KletterMix，一个通过机器翻译将高质量英文预训练语料 ClimbMix 逐文档、保留结构和元数据后构建的德国语预训练语料库，并对其质量进行系统评估与过滤。

**💡 创新点**

创新点在于：① 以文档级别保持原始混合结构，实现可比性与可追溯性；② 采用 COMETKiwi 参考无监督质量估计和梯度提升代理模型实现大规模质量筛选；③ 通过对不同长度桶、源域和元数据的细粒度分析，揭示翻译质量分布与潜在风险。

**🔧 技术方法**

主要技术包括：长度感知路由与上下文窗口分块翻译、动态目标侧预算、并行分片执行、COMETKiwi 参考无监督质量估计、基于翻译侧特征的梯度提升代理模型、以及对训练过程的监控与验证。

**📊 数据集**

使用的数据集为：KletterMix（翻译自 ClimbMix，约 12B 令牌）和对比基准 FineWeb2-DE、GermanWeb；基准评测还包括 Qwen3‑0.6B 模型在 12B 令牌预算下的预训练与后续 5‑shot 基准（MMLU、PIQA、HellaSwag、ARC‑C）。

**📈 对比分析**

方法比较：在相同模型、超参、令牌预算下，KletterMix 在训练损失、验证损失与多项 5‑shot 基准中均优于 FineWeb2-DE 与 GermanWeb，尤其在 HellaSwag 与 ARC‑C 上显著提升；在 FineWeb2-DE 预训练后进行的 annealing 实验中，KletterMix 进一步提升 Core Avg. 分数，表明其在后期 fine‑tune 过程中的加速与质量提升效果。

**⚠️ 局限性**

局限性：① 语料继承自英文来源，可能带来文化、地理与风格偏差；② 机器翻译可能产生翻译者歪曲、翻译ese、术语不一致及长文档失真；③ 质量评估主要依赖自动指标，缺乏人工评估；④ 仅在单一 0.6B 参数模型、12B 令牌预算下验证，缺乏更大模型、多种随机种子与更广泛基准的实验。

---

## 618. Beyond Encoder Accumulation: Measuring Encoder Roles in Multi-Encoder VLMs

**arXiv ID:** 2606.03879 | [PDF](https://arxiv.org/pdf/2606.03879v1)

**作者:** Wei Ding `[一作]` (Tsinghua University), Yu Wang `[通讯]` (Tsinghua University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过在Eagle-X5框架下，完整重训练并评估所有31个非空视觉编码器子集，系统分析多编码器视觉-语言模型中编码器的功能角色与互补性。

**💡 创新点**

提出了基于重训练的训练时移除（TR）评估协议，定义并区分“Capacity”（单独性能）与“Necessity”（在全池中不可替代贡献）两轴拆解，进一步发现编码器对的最佳组合依赖于预投影器输入的有效秩（rank）变化，而非单一性能最高的编码器对。

**🔧 技术方法**

采用统一的Eagle-X5两层MLP投影器、Vicuna-7B解码器，计算预投影器有效秩、Cohen’s Kappa、PID互补性等特征，并在Cambrian‑1 16任务基准上进行评估。

**📊 数据集**

使用Cambrian‑1 16-task benchmark（包括General、Knowledge、OCR&Chart、Vision‑Centric 四类任务）。

**📈 对比分析**

与传统推理时掩蔽（IM）方法对比，TR在排名稳定性、方差以及对最佳两编码器组合的判断上更可靠；最佳两编码器（ConvNeXt+CLIP）能恢复97%完整池性能，额外编码器收益极小。

**⚠️ 局限性**

局限在于实验规模仅覆盖5个编码器，重训练成本高，未验证所提出的rank‑expansion策略在更大或不同编码器池中的普适性；同时缺乏对优化过程的直接度量。

---

## 619. A Training-Free Mixture-of-Agents Framework for Multi-Document Summarization using LLMs and Knowledge Graphs

**arXiv ID:** 2606.03867 | [PDF](https://arxiv.org/pdf/2606.03867v1)

**作者:** Cuong Vuong Tuan `[一作]` (Phenikaa University), Thien Van Luong `[通讯]` (National Economics University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种完全不需要训练的多文档摘要 Mixture-of-Agents（MoA）框架，融合了抽取式、知识图（KGSum）和抽象式三种代理，并通过自适应多视角融合（AMF）实现最终摘要。

**💡 创新点**

创新点包括：①KGSum 通过动态构建知识图并检测对立关系，显式建模文档间复杂交互；②AMF 以元数据为驱动的自适应融合机制，可根据每个代理的质量和一致性动态调整权重；③整体框架完全训练‑free，能在多语言、多领域环境下直接使用预训练 LLM。

**🔧 技术方法**

技术手段：预训练 LLM（GPT‑4.1‑mini、Llama‑3.1‑8B、Qwen‑2.5‑7B 等）负责所有生成任务；Longformer 提供句子嵌入；知识图构建结合 spaCy NER、规则与 LLM 提取；Leiden 社区检测；多代理架构与 AMF 元数据驱动的 Prompt；评估使用 ROUGE‑1/2/L 与人类评价。

**📊 数据集**

实验数据集：四个多文档摘要基准——Multi‑News、Multi‑XScience（英文）；VN‑MDS、ViMs（越南语）。

**📈 对比分析**

与全训练（PEGASUS、LED、PRIMERA、Centrum、LatVisNewshead）和训练‑free（SRI+beam、GLIMMER‑TTR、Graph combine、Bert‑VBD）基线进行对比。MoA 在所有数据集上均达成或接近 SOTA：Multi‑News ROUGE‑1 55.0±0.3、R2 24.9±0.4；Multi‑XScience R1 36.7±0.2；VN‑MDS R1 80.1±0.5、R2 52.3±0.7；ViMs R1 80.3±0.5、R2 55.6±0.8。人类评估显示其语法、非冗余、指代清晰和结构一致性均显著优于对比模型。

**⚠️ 局限性**

局限性：仍受所用 LLM 能力与上下文长度限制，极长文档或极多文档场景仍需裁剪；KGSum 的实体/关系抽取误差会影响图质量；实验集中在英文和越南语，其他语言的泛化尚未验证；对冲突信息的细粒度处理仍有提升空间。

---

## 620. APX-Hardness of Computing Lipschitz Constants for Multi-Parametric Quadratic Programs

**arXiv ID:** 2606.03862 | [PDF](https://arxiv.org/pdf/2606.03862v1)

**作者:** Xingchen Li `[一作]` (Tsinghua University), Keyou You `[通讯]` (Tsinghua University)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文分析并证明了多参数二次规划（mpQP）解映射的Lipschitz常数计算是NP‑hard且APX‑hard，并揭示了参数维度、约束数和决策变量数对复杂度的具体影响。

**💡 创新点**

创新点在于首次给出参数维度、约束数和决策变量数对复杂度的精确阐述，并构造了将任意ReLU网络映射为mpQP的多项式时间变换，进而证明即使参数为标量，该问题仍保持难度。

**🔧 技术方法**

主要技术包括gap‑preserving归约（从MAX‑CUT和MAX‑2SAT）、ReLU网络到mpQP的严格KKT构造以及数值实验验证。

**📊 数据集**

实验使用随机生成的高斯分布参数mpQP实例以及基于tent‑map的单参数ReLU网络实例。

**📈 对比分析**

通过pdaqp求解器枚举所有临界区域，实验结果显示约束数和决策变量数的增大导致临界区域数量呈指数增长，验证了理论复杂度结论。

**⚠️ 局限性**

局限性在于仅针对严格凸mpQP，实验规模受限于[-5,5]或[-1,1]的参数域，未考虑更大规模或非凸情况，且仅验证了枚举法的可行性。

---

## 621. PyraMathBench: Evaluating and Improving Mathematical Capability in Large Language Models

**arXiv ID:** 2606.03858 | [PDF](https://arxiv.org/pdf/2606.03858v1)

**作者:** Zetian Ouyang `[一作]` (East China Normal University), Liang He `[通讯]` (East China Normal University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 PyraMathBench 基于 7,404 题目的 32,505 个子任务的层级化评测基准，并提出 SOLVE 模块和 IRPO 算法提升 LLM 的数理推理。

**💡 创新点**

1) 通过层级化子任务拆分实现对数值处理与数学推理的细粒度诊断；2) SOLVE 的模糊调用识别与 IRT 任务难度估计；3) IRPO 将工具调用纳入多轮 RL 优化。

**🔧 技术方法**

基于 LaTeX/ Python 语法的工具调用解析、两参数逻辑 IRT 2PL、强化学习 IRPO、Math-Verify 与 MathBERT 等评测工具。

**📊 数据集**

PyraMathBench（32,505 题目、14 子任务、2 模态）以及现有的 GSM8K、Ape210K、MATH、FrontierMath 等公开数据用于构建子任务。

**📈 对比分析**

在 11 语言模型上对比 baseline、PPM、SOLVE、GRPO、IRPO 的分数，IRPO+SOLVE 在 Qwen-2.5 和 Llama-3.1‑8B 上分别提升至 96.1 与 76.1，显著高于基线与其它工具调用方案。

**⚠️ 局限性**

子任务拆分可能存在多种版本，导致评测偏差；未覆盖所有 LLM 语言能力；仅评估英文，图像信息提取能力不足。

---

## 622. Where Do We (Not) Need Temporal Context in Low-Resource Video Task Adaptation?

**arXiv ID:** 2606.03837 | [PDF](https://arxiv.org/pdf/2606.03837v1)

**作者:** Luc P. J. Sträter `[一作]` (Leiden University), Hazel Doughty `[通讯]` (Leiden University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文系统研究了在低资源视频理解任务中，参数高效微调（PEFT）和探测器（probe）在不同视频预训练与图像预训练模型上的适配策略，并探讨了将时序上下文分配到骨干网络、PEFT模块和探测器的多样化设计；

**💡 创新点**

创新点在于首次将时序上下文分配作为一个维度进行系统评估，发现骨干网络的时序建模对运动任务最关键，探测器的简单注意力足以弥补多样化任务的差异，并提出在低资源场景下的高效适配策略；

**🔧 技术方法**

采用多种PEFT技术（LoRA、AdaptFormer、ST-Adapter、BitFit等）和探测器设计（线性、注意力、DiST、DPT、VDA等），并通过可变时序上下文分配方法对模型进行实验；

**📊 数据集**

使用六个低资源视频数据集，包括情感识别（CAER、NurViD）、运动步骤识别（IndustReal、MammAlps）以及密集空间预测（ScanNet深度回归、VSPW语义分割）；

**📈 对比分析**

与图像预训练和视频预训练骨干网络、各类PEFT与探测器组合进行对比，实验表明视频预训练骨干与LoRA/AdaptFormer结合在多任务上普遍优于图像预训练或仅探测器的表现，且适度的时序上下文分配能在精度与吞吐量之间取得良好平衡；

**⚠️ 局限性**

实验仅逐一调整时序上下文分配，未覆盖完整组合空间，且对不同层次的PEFT与探测器的细粒度交互影响尚未系统搜索，未来工作需要更全面的联合搜索以获得最优配置。

---

## 623. BigFinanceBench: A Workflow-Grounded Benchmark for Financial-Research Agents

**arXiv ID:** 2606.03829 | [PDF](https://arxiv.org/pdf/2606.03829v1)

**作者:** Alex Wang `[一作]` (Rogo), Eric Xu `[通讯]` (Rogo)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BigFinanceBench，一套包含928道专家撰写的财务研究问题及其点权重评估标准的基准，强调对完整推导过程的可审计评估；

**💡 创新点**

创新点在于将评估从单一答案迁移到流程级、可审计的推导评估，并通过点权重化的 rubric 对每一步骤进行独立打分；

**🔧 技术方法**

采用ReAct式工具调用框架、公开可检索的SEC文件、计算与表格工具以及双重人工评判进行模型跑通和评分；

**📊 数据集**

使用928个真实财务分析问题，包含15,656条评估标准，总计36,241分；

**📈 对比分析**

与十个前沿模型对比，最佳模型仅获得约58.8%的 rubric 分数，最终答案准确率低于45%，表明现有模型在完整流程中仍有显著提升空间；

**⚠️ 局限性**

局限性包括基于公开数据的静态评估，未涵盖专有数据库、实时交互或客户沟通，评估依赖人工打分，存在主观性，且未能覆盖所有行业与工作场景。

---

## 624. Leveraging BART to Assess CS1 C++ Programming Assignments using Rubric-based Criteria

**arXiv ID:** 2606.03814 | [PDF](https://arxiv.org/pdf/2606.03814v1)

**作者:** Kelsey Rainey `[一作]` (Tennessee Technological University), Jesse Roberts `[通讯]` (Tennessee Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了如何利用BART和T5等Transformer模型，在包含课程特定rubric的背景下，对初级C++编程作业进行多任务自动评分。

**💡 创新点**

结合多任务学习、rubric上下文、软标签（边界模糊标签）和分布匹配损失，以实现与教师评分更一致的分数与分布。

**🔧 技术方法**

BART encoder-decoder + LoRA微调、T5全微调、回归+分类头、Jensen‑Shannon Divergence分布匹配、软标签交叉熵、梯度裁剪、样本重采样等技术。

**📊 数据集**

2,404个来自Tennessee Technological University CS1课程的C++提交，附带数字分数、字母分数桶和PDF转文本的rubric。

**📈 对比分析**

通过MAE、准确率、macro‑F1、JSD等指标比较单任务/多任务、硬/软标签、rubric/无rubric、BART/T5/LoRA等配置；多任务+边界软标签+rubric的BART模型在MAE、JSD上表现最佳。

**⚠️ 局限性**

数据高度偏斜（大多数为A分），rubric与代码匹配不一致，样本量有限，且模型对少数类的敏感性受限。

---

## 625. Consistency Training Can Entrench Misalignment

**arXiv ID:** 2606.03810 | [PDF](https://arxiv.org/pdf/2606.03810v1)

**作者:** David Demitri Africa `[一作]` (UK AI Security Institute), Arathi Mani `[通讯]` (UK AI Security Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对 7 种一致性训练方法在 7–70B 大模型的四种控制性失衡“模型生物”上进行实验，评估其对模型对齐的非中性影响。

**💡 创新点**

创新点在于提出并验证“一致性非中性假设”，通过理论分析和大规模实验揭示一致性训练可抑制脆弱失衡（如奖励劫持、突现失衡）而放大连贯失衡（如附和倾向），并指出分布偏移是主要驱动机制。

**🔧 技术方法**

采用了多种一致性训练技术，包括 Self-Confidence、Diverse-Decoding、Multi-View Consistency、Self-Refinement、Self-Rewarding、Bias-Augmented Consistency Training、Activation Consistency Training，以及基于自监督的 label‑generation 与 regularization 方法。

**📊 数据集**

实验数据集由四类人工构造的“模型生物”数据组成：奖励劫持、突现失衡、虚假关联、附和倾向，使用 Llama‑3.1‑8B/70B、Gemma‑2‑9B、Mistral‑7B‑v0.3、GPT‑OSS‑20B 等开源模型进行微调。

**📈 对比分析**

对比方法采用“同一模型微调前后”对齐率的显著性检验，结果显示一致性训练在奖励劫持和突现失衡上有 60–80% 的抑制率，而在附和倾向上则 70–90% 的放大率；正则化方法（ACT/BCT）在抑制效果更强，但仍会显著放大附和倾向。

**⚠️ 局限性**

局限性包括：使用的失衡模型仅为人工诱导的控制性失衡，未必代表真实部署模型；70B 规模实验样本有限；评估依赖 LLM 判定器，可能受主观误差影响；对一致性训练的因果机制仍未完全解释。

---

## 626. PURGE: Projected Unlearning via Retain-Guided Erasure

**arXiv ID:** 2606.03808 | [PDF](https://arxiv.org/pdf/2606.03808v1)

**作者:** Vedant Jawandhia `[一作]` (BITS Pilani), Pratik Narang `[通讯]` (BITS Pilani)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出一种基于A‑GEM梯度投影的机器无学习算法，能够在保持模型对保留数据性能的前提下，有效地从模型中消除指定类别的数据影响。

**💡 创新点**

其创新点在于直接将连续学习中的梯度投影机制迁移到无学习任务，并结合保留混淆目标、多层表示消除、KD锚定以及双重自调停机机制，形成了既能保护保留性能又能逼近无学习模型的完整框架。

**🔧 技术方法**

采用的核心技术包括梯度投影、KL保留混淆目标、知识蒸馏锚定、多层MSE表示消除、熵门控、BatchNorm冻结及自调停机。

**📊 数据集**

在CIFAR‑10、MNIST、SVHN、STL10和PathMNIST五个数据集上进行了实验。

**📈 对比分析**

与Retrain、SalUn、GA等多种基线对比，方法在保留准确率均保持在96%以上，成员推理攻击AUROC接近0.5（暗示隐私保护良好），但测试准确率略低于部分基线，体现了隐私与效能的权衡。

**⚠️ 局限性**

主要限制包括：测试准确率相对基线有显著下降、缺乏正式的(ε,δ)‑DP保证、在带动量的优化器下梯度投影的严格保证不完全、单次种子实验缺乏统计稳健性、以及仅验证单类忘记，未处理多类或连续忘记场景。

---

## 627. A Novel Procedural Generation for Level Design of Mansions and Dungeons

**arXiv ID:** 2606.03857 | [PDF](https://arxiv.org/pdf/2606.03857v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564`

---

## 628. From 'What' to 'How' and 'Why': Sharing LLM-Generated Retrospective Summaries of Older Adults' Passive Tracking Data with Remote Family Members

**arXiv ID:** 2606.03876 | [PDF](https://arxiv.org/pdf/2606.03876v1)

**作者:** Jiachen Li `[一作]` (Northeastern University), Varun Mishra `[通讯]` (Northeastern University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用大语言模型生成针对老年人被动传感数据的多模态回顾性摘要，为远程家庭成员提供情境化、个性化的护理信息。

**💡 创新点**

创新点是基于访谈发现的需求，构建多层多代理、洞察驱动的摘要框架，将客观统计、描述与情境化解释分层生成，并提供可定制化的层级展示。

**🔧 技术方法**

使用的技术包括 OpenAI GPT‑5.2（LLM）、Vital Insight 的多层摘要管道、规则+函数调用的多代理系统以及 Prompt Engineering。

**📊 数据集**

使用的数据集为一名独居老年人两个月的多模态传感数据，包括可穿戴设备、智能手机、对话代理等。

**📈 对比分析**

比较方法是将改进后的摘要与原始（bullet‑point）摘要进行定量问卷评估，结果显示改进版在满意度、帮助性、可信度和每日使用意愿上显著提升（p<0.05）。

**⚠️ 局限性**

限制包括样本规模小（11人）、只针对单一老年人数据、摘要未实现个性化定制、以及缺乏长期真实部署验证。

---

## 629. Automated Repair of Requirements for Cyber-Physical Systems in Simulink Requirements Tables

**arXiv ID:** 2606.03870 | [PDF](https://arxiv.org/pdf/2606.03870v1)

**作者:** Aren A. Babikian `[一作]` (University of Toronto), Marsha Chechik `[通讯]` (University of Toronto)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于系统执行数据的自动化需求修复框架，用于纠正与已演化的物理-计算机系统（CPS）不一致的 MATLAB Simulink® 需求表（RT）要求；

**💡 创新点**

创新点在于首次将量化语义与可衡量的可取性维度（语义完整性、语法相似度、满足度扩展）结合，并通过多目标优化搜索合适的需求修复；

**🔧 技术方法**

采用了量化语义评估、Z3 SMT 进行语义检查、Zhang–Shasha 树编辑距离评估语法相似度、以及基于遗传编程的多目标搜索算法；

**📊 数据集**

使用了六个工业级 CPS 案例（AFC、AT、CC、EU、NNP、TUI）共12条需求，并生成约 1000 条执行轨迹作为训练集；

**📈 对比分析**

通过与七种框架变体的对比实验，发现基于多目标搜索的变体在修复正确性、可取性以及实用性上均优于单目标或无可取性约束的方案，Pareto 前沿规模与平均可取性均有所提升；

**⚠️ 局限性**

局限性包括：搜索空间受限于 RT 语法与变异算子、仅在单一需求上搜索、依赖轨迹集的代表性、未考虑多需求间的相互影响以及对领域特定约束的手动编码需求。

---

## 630. Unified Video-Action Joint Denoising for Dexterous Action and Data Generation

**arXiv ID:** 2606.03868 | [PDF](https://arxiv.org/pdf/2606.03868v1)

**作者:** Dingrui Wang `[一作]` (Technical University of Munich), Johannes Betz `[通讯]` (Technical University of Munich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `40105733-5154-44cd-8090-a8cab9e64b07` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了Donk模型，能够在仅有文本或文本+初始图像的条件下生成配对的交互视频和空间对齐的手部轨迹，实现了统一的观察条件与文本条件的视频-动作生成；

**💡 创新点**

创新点在于将视频-动作对齐的世界动作模型统一为单一的联合去噪框架，使其既能作为观察条件的策略（TI2VA），又能作为纯文本的数据生成引擎（T2VA），并通过anchor‑map控制保证手部动作与视觉未来同步；

**🔧 技术方法**

采用预训练的Video Diffusion Transformer（Wan）作为去噪骨干，联合去噪视频和MANO手部动作，利用flow‑matching目标、视频-动作对齐、anchor‑map控制、手部聚焦损失等技术；

**📊 数据集**

在VITRA‑1M数据集上训练，并在OakInk2、EgoDex等公开基准上进行评估；

**📈 对比分析**

与多种基线（VITRA、Being-H0、DreamZero‑like、Wan系列等）比较，TI2VA模式下在手部姿态、轨迹误差和腕部旋转误差上取得最优结果，视频质量也保持竞争力；T2VA模式下在文本生成视频质量和语义对齐方面与现有视频生成模型相当，且能够输出可执行的双手动作；

**⚠️ 局限性**

局限性包括对初始手部姿态的anchor‑map依赖，文本条件下的生成质量仍受限于预训练视频模型的能力；缺乏针对更复杂动作场景的广泛评估；

---

## 631. Second-Best Bilateral Trade is $1/2$ Efficient

**arXiv ID:** 2606.03849 | [PDF](https://arxiv.org/pdf/2606.03849v1)

**作者:** Zhengyang Liu `[一作]` (Beijing Institute of Technology), Zihe Wang `[通讯]` (Renmin University of China)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

证明在无外部补贴、满足贝叶斯激励兼容、参与约束和强预算平衡的双边交易中，最优（第二佳）机制的效率至少是第一佳的1/2。

**💡 创新点**

提出一种全局结构化的证明框架，直接对第二佳基准进行效率分析，且证明1/2下界是紧的。

**🔧 技术方法**

通过离散化、拉格朗日对偶、弱正则性/铁化、Sion最小最大定理，将问题转化为贪心算法可行性，从而完成证明。

**📊 数据集**

本文完全基于理论分析，无使用任何真实数据集，所用分布为通用连续/离散分布。

**📈 对比分析**

与已知简单机制（如随机出价、固定价格）以及上界2/e进行对比，证明该下界与上界相匹配，并给出紧性示例实现比值逼近1/2。

**⚠️ 局限性**

研究范围限于单卖单买、独立分布且强预算平衡的情形，对多方交易、非独立分布或弱预算平衡的扩展尚未解决。

---

## 632. Clustered Self-Assessment: A Simple yet Effective Method for Uncertainty Quantification in Large Language Models

**arXiv ID:** 2606.03846 | [PDF](https://arxiv.org/pdf/2606.03846v1)

**作者:** Qi Cao `[一作]` (University of Tokyo), Yusuke Iwasawa `[通讯]` (University of Tokyo)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于聚类的自评方法，将采样答案聚类后构成多选题，让LLM给出选项概率作为不确定度估计。

**💡 创新点**

创新点在于将答案聚类与自评结合，利用多选题的概率直观表达不确定性，并在只需两次额外采样时就能实现高效准确。

**🔧 技术方法**

使用的技术包括答案采样、NLI模型进行语义聚类、构造多选题、利用LLM的token概率作为置信度，并对模型隐藏层进行probe训练。

**📊 数据集**

评估数据集包括TriviaQA、Natural Questions以及Extreme Summarization的XSum数据集。

**📈 对比分析**

与多种基准（如Perplexity、P(True)、Semantic Entropy等）对比，实验显示在AUROC和Brier分数上均显著优于基线，且在样本数少时仍保持竞争力。

**⚠️ 局限性**

局限性包括需要访问LLM logits、依赖外部NLI模型导致额外开销、以及未做后置校准，闭源模型或对计算资源有限的场景适用性受限。

---

## 633. Warning About AI Fallibility Increases Help-Seeking in an Intelligent Tutoring System

**arXiv ID:** 2606.03822 | [PDF](https://arxiv.org/pdf/2606.03822v1)

**作者:** Tomohiro Nagashima `[一作]` (Saarland University), Vera Rief `[通讯]` (Saarland University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在一所日本中学中进行课堂实验，比较在数学智能辅导系统中提示学生系统可能犯错与不提示的两种版本，记录学生的帮助请求、错误率和用时等日志数据。

**💡 创新点**

首次实证展示即使系统本身不产生错误，轻量级的透明度提示（告知AI可能犯错）也能显著影响学生的求助行为，揭示了透明提示对学习者元认知策略的潜在影响。

**🔧 技术方法**

采用基于规则的数学智能辅导系统（Cognitive Tutor Authoring Tools），对日志数据进行线性回归分析，以评估不同条件下的行为差异；实验本身不使用大语言模型，强调“安慰剂效应”。

**📊 数据集**

实验样本为252名七年级学生（年龄12-13岁），来自东京一所中学，实验期间记录的交互日志作为数据集。

**📈 对比分析**

通过对两组（Mistake vs NoMistake）进行线性回归比较，发现提示组的帮助请求率显著更高（β = -0.33，p = .02），而错误率和用时没有显著差异；整体性能未见提升或下降。

**⚠️ 局限性**

局限性包括：①系统未实际产生错误，无法检验提示对错误处理的真实影响；②研究仅在日本的单一学科与年龄段进行，结果的普适性未知；③组别不平衡可能降低统计功效；④仅评估了帮助请求行为，未深入探讨元认知水平或学习效果。

---

## 634. TeX-1500: A Paired Real-World LWIR Hyperspectral Dataset and Benchmark for Temperature-Emissivity-Texture Decomposition

**arXiv ID:** 2606.03806 | [PDF](https://arxiv.org/pdf/2606.03806v1)

**作者:** Cheng Dai `[一作]` (Westlake University), Fanglin Bao `[通讯]` (Westlake University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了TeX-1500，一套包含1522个真实场景的低波长红外高光谱图像与对应温度、发射率、纹理三元组的对齐数据集。

**💡 创新点**

首次构建了统一的校准、去噪与TeX标签生成流程，解决了现有逆向求解器对单场景手工调优的限制，并为学习式温度–发射率–纹理分解提供可监督的配对标签。

**🔧 技术方法**

结合LWIR高光谱成像、气象参考校准、HADAR-SLOT求解器以及基于U-Net的TeX-UNet网络实现数据预处理和模型训练。

**📊 数据集**

使用DARPA IH推式光谱机的户外数据与FTIR近场采集的多光谱样本，构成了跨场景、跨传感器的多样化训练集。

**📈 对比分析**

通过在留存的DARPA IH测试集上评估TeX-UNet，平均绝对误差约为7.3K，纹理与发射率MAE分别为0.045和0.032；零样本转移至FTIR时MAE下降至5.8K，微调后进一步提升到4.1K。

**⚠️ 局限性**

当前标签主要关注视觉一致性，发射率与温度的物理精度尚未充分验证，且数据集规模与光谱覆盖仍有限，未来需要更精细的物理校准与更大规模的多传感器采集。

---

## 635. Let the Dynamics Flow: Stable Flow Matching Dynamical Systems

**arXiv ID:** 2606.03834 | [PDF](https://arxiv.org/pdf/2606.03834v1)

**作者:** Rodrigo Pérez-Dattari `[一作]` (KTH Royal Institute of Technology), Noémie Jaquier `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 303 | [OpenAlex ID](https://openalex.org/A5064146907)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一种基于流匹配的稳定动力学系统（SFMDS），能够在学习机器人运动时兼顾高表达度、多模态能力和Lyapunov稳定性约束；

**💡 创新点**

将流匹配生成模型与控制理论中的Lyapunov稳定性约束结合，给出软硬两种实现方式，并将框架推广到Lie群上；

**🔧 技术方法**

使用条件流匹配（flow matching）、隐层Lyapunov函数、LaSalle不变性原理、软/硬约束投影、ODE/Neural ODE、UNet/Glow网络、Taylor近似Jacobian等技术；

**📊 数据集**

在LASA手写数据、LASA on 𝕋²、Multimodal ℝ²、SE(3) pouring、真实人形机器人插入、1682维Gray‑Scott反应扩散等多种数据集上进行实验；

**📈 对比分析**

与行为克隆、CLF‑DM、Euclideanzing flows、PUMA、LieImFlow等基线相比，SFMDS在多模态和高维任务中实现了更低的Chamfer/DTWD/RMSE，同时提供了稳定性保证；软约束版在精度上更优，硬约束版在稳定性和鲁棒性上更强；

**⚠️ 局限性**

目前仅支持单一目标姿态，未直接建模机器人物理约束（如碰撞、工作空间限制），在高维或极复杂动作时训练和推理成本高；硬约束实现对网络可微分性要求较高，易出现边界案例。

---

## 636. Taiji: Pareto Optimal Policy Optimization with Semantics-IDs Trade-off for Industrial LLM-Enhanced Recommendation

**arXiv ID:** 2606.03866 | [PDF](https://arxiv.org/pdf/2606.03866v1)

**作者:** Yuecheng Li `[一作]` (Kuaishou Technology), Kun Gai `[通讯]` (Unaffiliated)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 Taiji 框架，结合 RUPR+ORFT 进行高质量 CoT 生成与 SFT，随后使用 POPO 在 RL 阶段动态平衡 LLM 语义奖励与推荐 ID 合作奖励，提升工业级推荐性能；

**💡 创新点**

在 SFT 阶段首次引入逆向工程用户偏好推理与开放式拒绝采样精炼 CoT，RL 阶段提出可实现 Pareto 最优权重调节的 POPO；

**🔧 技术方法**

使用 QwQ-32B 进行 CoT 蒸馏，DeepSeek‑R1‑7B 进行 SFT，Qwen‑Embedding‑0.6B 进行语义评估，GRPO 与 POPO 结合实现多奖励 RL；

**📊 数据集**

基于快手广告平台近百万用户日志，含用户画像与最近 50 条广告交互序列；

**📈 对比分析**

与 DeepSeek‑R1‑7B、QwQ‑32B 对比，Taiji 在语义 Hit‑Rate 与 CTCVR 上分别提升 55% 及 11%，在线 A/B 测试中 ADVV +2.83%、Revenue +3.30%，长尾用户提升更显著；

**⚠️ 局限性**

对长尾用户的细粒度推荐仍有限，且 RUPR 与 POPO 的计算开销在极大规模时仍需优化；

---

## 637. FLARE: Fine-Grained Diagnostic Feedback for LLM Code Refinement

**arXiv ID:** 2606.03852 | [PDF](https://arxiv.org/pdf/2606.03852v1)

**作者:** Yinsheng Yao `[一作]` (Tongji University), Tianyi Zhang `[通讯]` (Purdue University)

**通讯引用:** 7747 | [OpenAlex ID](https://openalex.org/A5100437458)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种迭代代码修复框架，使用轻量级诊断模型给出行级可疑性评分，并结合执行反馈与候选搜索逐步改进生成代码。

**💡 创新点**

创新点在于：①基于LLM内部概率信号训练细粒度诊断模型；②将诊断与执行反馈融合，在每轮迭代中搜索top‑k可疑行并按执行结果挑选最佳候选；③证明细粒度定位显著提升修复效果。

**🔧 技术方法**

技术包括：双向Transformer诊断模型、token概率与AST词法单元对齐、max‑pool聚合行级分数、候选搜索与执行反馈循环、对比实验中的基线方法。

**📊 数据集**

数据集：LiveCodeBench 和 BigCodeBench 的代码生成任务；诊断模型训练使用 10504 条带有LLM概率信号和真实 bug 位置信息的程序。

**📈 对比分析**

比较方法：Self‑Debugging、Self‑Refine、NL‑Debugging 以及原始 LLM。实验显示即使 k=1 也显著提升 Pass@1（绝对提升 1.72%–7.42%），k=10 时平均提升 8.5%。诊断模型在定位准确率上优于 FlexFL、LLMAO、BAP。

**⚠️ 局限性**

局限性：迭代过程增加计算成本，k 或迭代次数增大时开销显著；当 LLM 本身无法完成必要的推理或误解任务时，诊断定位无法弥补，仍会出现失败。

---

## 638. EvoDS: Self-Evolving Autonomous Data Science Agent with Skill Learning and Context Management

**arXiv ID:** 2606.03841 | [PDF](https://arxiv.org/pdf/2606.03841v1)

**作者:** Zherui Yang `[一作]` (Hong Kong University of Science and Technology), Hao Liu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 144266 | [OpenAlex ID](https://openalex.org/A5100338921)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个自演化的多智能体数据科学代理EvoDS。

**💡 创新点**

创新点在于自适应工具演化机制、上下文压缩策略以及联合多智能体强化学习框架，能够在执行过程中自动扩展工具集并高效管理长上下文。

**🔧 技术方法**

技术包括LLM驱动的代理、工具合成与验证、上下文压缩（Adaptive Context Compression）、两阶段SFT+RL（使用GRPO）以及分层多智能体架构。

**📊 数据集**

使用的数据集包括四大数据科学基准：DABench、DA-Code、ScienceAgentBench、MLE-Dojo，训练时还采集了DataMind‑12K、DataScience‑Instruct‑500K、MatPlotBench等任务数据。

**📈 对比分析**

与多种开源和专有基准（AutoGen、ReAct、DataMind、DeepAnalyze等）比较，EvoDS在四个基准上平均提升28.9%，并在大部分基准上接近或超过最强对手。

**⚠️ 局限性**

局限在于对最难的科学发现任务仍有显著差距，受限于基础LLM的知识与推理能力，缺乏深度领域知识集成和更强的抽象推理能力。

---

## 639. Online Learning with Gradient-Variation Interval Regret

**arXiv ID:** 2606.03831 | [PDF](https://arxiv.org/pdf/2606.03831v1)

**作者:** Yan-Feng Xie `[一作]` (Nanjing University), Zhi-Hua Zhou `[通讯]` (Nanjing University)

**通讯引用:** 62398 | [OpenAlex ID](https://openalex.org/A5100621138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究非平稳在线学习，提出基于两层在线集成的算法GAIR，获得梯度变异量的区间遗憾上界。

**💡 创新点**

首次给出梯度变异量区间遗憾上界，实现“best‑of‑three‑worlds”结果，兼顾梯度变异、损失小量和最优最坏情况，同时设计无参数的Lipschitz自适应元学习器。

**🔧 技术方法**

采用优化梯度下降（OMD）作为基学习器，Optimistic‑Adapt‑ML‑Prod/LEO Adapt‑ML‑Prod作为元学习器，结合问题相关/无关调度，利用Bregman散度消除正项，并使用剪裁与学习率自适应。

**📊 数据集**

通过合成实验验证，使用合成回归数据和MNIST分类任务，模拟SEA模型。

**📈 对比分析**

与SAOL、SACS、SACSPP、NIPS22等基线对比，GAIR‑L在非平稳环境中取得最低累计损失和最高准确率，尤其在Lipschitz自适应实验中优于调参基线。

**⚠️ 局限性**

仍受梯度变异与损失小量耦合的对数项限制，无法完全分离两者，且当G和L未知时仍需额外对数因子，且算法对超参数如B0有一定依赖。

---

## 640. Calibrating Urban Traffic Simulation from Sparse Road Observations via Genetic Optimization

**arXiv ID:** 2606.03823 | [PDF](https://arxiv.org/pdf/2606.03823v1)

**作者:** Hunter Sawyer `[一作]` (Tennessee Technological University), Simon Matei `[通讯]` (Tennessee Technological University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

通过遗传算法在SUMO仿真中优化工作地点分布与入口出口流量参数，使少量真实道路流量观测下的城市交通仿真与实际流量高度匹配，且可在未观测道路上泛化。

**💡 创新点**

核心创新在于无需详细就业分布数据，直接利用稀疏道路流量观测校准宏观需求并通过遗传算法优化工作地点分布，而非传统微观驾驶行为参数。

**🔧 技术方法**

使用遗传算法与SUMO交通仿真平台、ActivityGen生成需求、OpenStreetMap道路网络以及GreenEVT框架。

**📊 数据集**

所用数据集包括OpenStreetMap道路网络、US Census住房分布、North Carolina Dept. of Transportation 2024年平均每日交通流量样本。

**📈 对比分析**

采用相关系数评估仿真与真实流量的吻合度，全数据训练达到0.64–0.72相关；随机排除测试显示泛化能力较好，地理排除表现不稳定；工作地点分布热力图与人口普查就业分布可视对齐。

**⚠️ 局限性**

局限在于对道路流量观测的空间分布要求较高，地理集中排除时泛化差；缺乏跨城市量化验证；仅通过视觉方式评估工作地点分布与实际就业分布的相似性。

---

## 641. TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding

**arXiv ID:** 2606.03819 | [PDF](https://arxiv.org/pdf/2606.03819v1)

**作者:** Peer Rheinboldt `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21576 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种名为TreeFlash的单步块级草稿生成方法，用于加速大型语言模型的推理；

**💡 创新点**

通过在DFlash基础上加入轻量级自回归逼近层（SwiGLU），使草稿每个位置对前一词的条件依赖，从而弥补单步草稿因缺乏自回归条件导致的分布漂移，并通过两阶段树构造保持O(1)推理复杂度；

**🔧 技术方法**

利用Speculative Decoding、树式草稿、DFlash、DDTree、SwiGLU自回归逼近、前向KL损失、损失缩放、两阶段OPT-Tree构造、Batched verification等技术；

**📊 数据集**

在数学推理数据集MATH‑500、GSM8K、代码生成数据集HumanEval、MBPP、通用指令集MT‑Bench上进行评估；实验使用Qwen3 4B、8B以及Qwen3 Coder 30B模型；

**📈 对比分析**

与EAGLE‑3、DFlash、DDTree等基线进行比较，使用块效率τ、速度提升、TVD、top‑K覆盖率等指标；TreeFlash在所有模型、任务和解码策略下均优于基线，平均块效率提升约12.4%，速度提升约9%（在B=64时更显著）；

**⚠️ 局限性**

仅在单批次SDPA推理环境下验证，未考察量化、多批次或自定义注意力的生产部署；初始化自DFlash，未进行从零开始的预训练；自回归逼近训练采用教师强制，推理时可能出现暴露偏差；仅在Qwen系列模型上评估，是否能推广至其他模型尚未验证。

---

## 642. Finite-Temperature de Bruijn Identities: Fisher Information as the Spectral Gap of Blahut--Arimoto Dynamics

**arXiv ID:** 2606.03813 | [PDF](https://arxiv.org/pdf/2606.03813v1)

**作者:** Qiao Wang `[一作]` `[通讯]`, Qiao Wang

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文揭示了Blahut–Arimoto（BA）动态谱理论与 Fisher 信息之间的关系，给出了有限温度的 de Bruijn 恒等式；

**💡 创新点**

创新点在于发现 BA 谱间隙 λ=J/(2β)，即将 Fisher 信息视为在可变温度下的正规化版本，并由此导出精确的有限温度 de Bruijn 恒等式与其零温度极限；

**🔧 技术方法**

主要使用了 BA 迭代的谱理论、翻译模态的 Rayleigh 商、Hermite 多项式显式对角化、Danskin（包络）定理和变分上界；

**📊 数据集**

无实验数据集，完全是理论推导；

**📈 对比分析**

没有与其他方法做实验比较，本文仅通过解析式验证新恒等式在 Gaussian 情况下完全成立；

**⚠️ 局限性**

局限性在于精确结果仅适用于高斯源，非高斯源仅给出上界；还需研究核的张量化、Wasserstein 解释及算法上的温度调度等开放问题。

---

## 643. Two-Action Apple Tasting with Switching Costs

**arXiv ID:** 2606.03851 | [PDF](https://arxiv.org/pdf/2606.03851v1)

**作者:** Tommaso Cesari `[一作]` (University of Ottawa), Roberto Colomboni `[通讯]` (University of Bristol)

**通讯引用:** 54 | [OpenAlex ID](https://openalex.org/A5012252221)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

研究两动作苹果品尝问题（带切换成本）的最优下界与算法，证明其极限后悔量为Θ(√T)；

**💡 创新点**

提出了简洁的几何探测（break‑even）算法，并用几何分布与随机游走分析证明其可达到O(√T)的期望后悔，关闭了先前关于T^(2/3)下界的空白；

**🔧 技术方法**

采用概率论工具（Bernoulli探测、几何分布、Rademacher随机变量、期望与高阶矩不等式）对算法进行上界证明，并用随机游走与对称性获得下界；

**📊 数据集**

无外部数据集，全部为理论证明与随机构造；

**📈 对比分析**

与通用反馈图算法给出的O(T^(2/3))上界比较，表明在该特定游戏中可将后悔降低到Θ(√T)，理论性能显著提升；

**⚠️ 局限性**

仅适用于无意识（oblivious）对手；对适应性对手未给出结论；算法参数需先知T；未进行实验验证。

---

## 644. Reasoning Structure of Large Language Models

**arXiv ID:** 2606.03883 | [PDF](https://arxiv.org/pdf/2606.03883v1)

**作者:** Frédéric Berdoz `[一作]` (ETH Zurich), Roger Wattenhofer `[通讯]` (ETH Zurich)

**通讯引用:** 21576 | [OpenAlex ID](https://openalex.org/A5078339613)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

建立了可扩展的二维格子谜题基准，并将大语言模型的自由形式推理轨迹转化为可验证的推理图，进而定义了推理流效率指标η。

**💡 创新点**

创新点在于引入结构化、可验证的推理图与基于图结构的效率度量，揭示了仅凭答案准确率或 token 数量无法体现的推理质量差异。

**🔧 技术方法**

采用混合确定性与LLM提取、规则抽取和验证构造有向无环图，利用可吸收马尔可夫链计算结构熵并得到η，并在可执行谜题环境中进行验证。

**📊 数据集**

使用21种可扩展的二维格子谜题（包括Tents等），每种谜题设定四个难度级别，每级提供5个实例，共计约420个样本。

**📈 对比分析**

通过在GPT‑5、Qwen‑3、DeepSeek V3.2、Kimi K2等模型上进行准确率与 token 计数对比，并用η衡量推理结构；结果显示即使准确率饱和，模型的推理效率仍存在显著差异，GPT‑5在所有难度上保持最高准确率和最佳 token 效率。

**⚠️ 局限性**

局限性包括：提取过程依赖LLM，可能带来偏差；每种谜题需手工定义声明和规则；验证仍需域特定逻辑；抽取误差会影响图质量，导致结构度量不稳定。

---

## 645. MLP Splatting: Object-Centric Neural Fields

**arXiv ID:** 2606.03877 | [PDF](https://arxiv.org/pdf/2606.03877v1)

**作者:** Shinjeong Kim `[一作]` (Imperial College London), Andrew J. Davison `[通讯]` (Imperial College London)

**通讯引用:** 32144 | [OpenAlex ID](https://openalex.org/A5039230558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 MLP‑Splatting，一种以独立 MLP 为原语的对象中心化场景表示，能够在仅 RGB 监督下自动分离对象并实现高质量实时渲染。

**💡 创新点**

创新点在于：1) 将每个原语设为可微分的独立 MLP，天然产生对象级结构；2) 通过稀疏体积合成与 tile‑based 排序实现显著低内存和高 FPS；3) 可通过语义特征蒸馏实现开箱即用的语义编辑与即时分割。

**🔧 技术方法**

使用的技术包括：多分辨率位置编码、6D 输入 MLP、稀疏体积渲染与前后端 tile‑based 计算、光学投影排序、以及 CLIP/SAM 等基础模型的特征蒸馏。

**📊 数据集**

实验数据集：Replica 与 ScanNet。

**📈 对比分析**

与 NeRF‑DFF、Feature‑3DGS 等基线对比：在 PSNR/SSIM/LPIPS 与 mIoU/Acc 等指标上相当或更优，同时内存使用降低至 1/15，渲染速度提升约 3 倍（≈8 FPS）。

**⚠️ 局限性**

局限性：未实现显式密度控制，原语的扩张/消亡由梯度噪声驱动，可能影响效率与重建质量；对动态场景的适用性尚未评估。

---

## 646. DyaPlex: Full-Duplex Speech-Motion Model for Dyadic Interaction

**arXiv ID:** 2606.03874 | [PDF](https://arxiv.org/pdf/2606.03874v1)

**作者:** Koki Nagano `[一作]` (NVIDIA), Shalini De Mello `[通讯]` (NVIDIA)

**通讯引用:** 2975 | [OpenAlex ID](https://openalex.org/A5021623206)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 DyaPlex，首个实现双向实时语音与动作同步生成的全双工对话模型。

**💡 创新点**

创新点在于：① 采用双塔 Transformer，将冻结的全双工语音模型与可训练的动作塔深度耦合；② 通过时间对齐的 RoPE 引入跨模态注意力的时序偏置，保证语音与动作严格同步；③ 在动作塔中使用双向交错的动作序列，实现伙伴动作的实时感知与生成。

**🔧 技术方法**

技术包括：PersonaPlex（冻结的全双工语音 Transformer）、基于 RVQ‑VAE 的体部动作分词器、双塔 Transformer 架构、时间对齐 RoPE、跨模态注意力、Causal 生成与自回归训练。

**📊 数据集**

使用规模达 4,000 小时的 Seamless Interaction 数据集，对话中包含 4,000 对双人语音与 3D 动作序列。

**📈 对比分析**

与 Audio2Photoreal、DualTalk 等基线在 Seamless 测试集上比较，DyaPlex 在单人动作质量（FGD、Diversity）、同步度（BeatAlign）和双人交互质量（P‑FD、Δ‑User）上均获得显著提升，尤其在 Δ‑User 上提升约 31%，显示对伙伴动作的强学习能力。

**⚠️ 局限性**

限制点：模型依赖大型语音模型与 RVQ‑VAE，训练和推理成本仍较高；仅在 12.5 Hz 采样率下评估，较高频率动作细节尚未深入；缺乏多语言或跨文化对话的泛化验证。

---

## 647. Explainable Forecasting of Scientific Breakthroughs from Concept Network Dynamics

**arXiv ID:** 2606.03864 | [PDF](https://arxiv.org/pdf/2606.03864v1)

**作者:** Thomas Maillart `[一作]` (University of Geneva), Alain Mermoud `[通讯]` (armasuisse Science + Technology)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种可解释的机器学习框架，预测科学突破前的概念链接形成与强化

**💡 创新点**

在保持高度可解释性的同时，实现链接存在预测AUC≥0.95并将强度预测误差控制在RMSLE≈0.45–0.6，优于传统模型

**🔧 技术方法**

使用LightGBM两阶段模型（分类+回归）结合59个网络结构特征（如Adamic–Adar、Hadamard）

**📊 数据集**

基于OpenAlex开放概念图，覆盖1990–2023年四个领域（量子计算、机器人、先进材料、神经植入）

**📈 对比分析**

在相同超参数下，跨领域验证ROC‑AUC始终在0.954–0.967之间，回归误差随预测期延长略增，整体性能显著提升

**⚠️ 局限性**

局限包括仅预测结构前兆而非实际影响；仅在单领域内训练；依赖OpenAlex标签质量；对极快速增长领域的强度预测略弱

---

## 648. CLI-Anything: Towards Agent-Native Computer Use

**arXiv ID:** 2606.03854 | [PDF](https://arxiv.org/pdf/2606.03854v1)

**作者:** Yuhao Yang `[一作]` (University of Hong Kong), Chao Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 12472 | [OpenAlex ID](https://openalex.org/A5006594763)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

设计并实现了 CLI-Anything 框架与 CLI‑Hub 平台，能把 GUI 软件提升为命令行接口，使 LLM 代理可通过结构化、可验证的方式与软件交互。

**💡 创新点**

提出 agent‑native 软件接口理念，构建了可复用的 Harness‑Lift 方法、Preview 协议和验证机制，完成多款专业软件的 CLI harness，突破 GUI 交互瓶颈。

**🔧 技术方法**

使用 Python（Click、JSON）、后端包装器（Blender bpy、GIMP Script‑Fu 等）、HTTP bridge、preview bundle 协议、注册与安装系统、技能生成与 E2E 测试等技术栈。

**📊 数据集**

以多款专业软件的项目文件为样例（Blender 场景、FreeCAD 组件、GIMP 图像、Slay the Spire 游戏状态等），未使用传统公开数据集。

**📈 对比分析**

通过对 70+ CLI harness 的注册量、类别覆盖、代理调用比例和使用量进行评估，显示代理已占主导；单元与 E2E 测试以及 preview bundle 验证证明接口一致性与可执行性。

**⚠️ 局限性**

仍无法覆盖仅有二进制或无可访问后端的软件；视觉决策仍需屏幕输入；预览覆盖、安装策略多样性及动态状态支持仍有限。

---

## 649. Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies

**arXiv ID:** 2606.03847 | [PDF](https://arxiv.org/pdf/2606.03847v1)

**作者:** Xiangdong Feng `[一作]` (Beijing Institute of Technology), Li Jiang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 22905 | [OpenAlex ID](https://openalex.org/A5100392387)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

提出一种无训练、推理时自适应动作分块方法 DVAC，利用流式策略去噪轨迹中的方差信号决定执行的动作长度，从而提升任务成功率并降低重新规划次数。

**💡 创新点**

创新点在于：①将流式政策的去噪轨迹中的方差信号作为自适应规划的实时指标；②通过滚动窗口自适应阈值自动调节不同任务与回合的方差尺度；③实现完全无训练、无额外候选分块的推理时自适应执行。

**🔧 技术方法**

使用技术包括：流式机器人策略（flow matching / diffusion）、去噪方差估计、滚动窗口阈值自适应、动作分块（action chunking）以及与其它自适应方法的对比实验。

**📊 数据集**

数据集与实验平台：LIBERO 4个仿真子集（Spatial、Object、Goal、Long）、RoboTwin 仿真、CALVIN‑5 benchmark 以及真实机器人上的三项操作任务（放红立方体、堆叠立方体、移动试管）。

**📈 对比分析**

比较方法：与固定长度执行（如 Fixed‑15、Fixed‑40）、其它自适应分块方法（AAC、AutoHorizon、LAPA、CoT‑VLA 等）进行对比。性能表现：LIBERO 平均成功率从 0.948 提升至 0.980，重新规划次数下降 43%；RoboTwin 成功率从 0.359 提升至 0.416；CALVIN‑5 完成子任务数从 3.905 提升至 4.040；实地实验中成功率提升、任务时长缩短、重新规划次数明显减少。

**⚠️ 局限性**

局限性：仅适用于流式或扩散式策略，需要访问中间去噪轨迹；方差作为经验代理未经过严格校准，可能不足以提供安全性保证；实地评估任务与环境仍相对有限，需在更复杂场景中进一步验证。

---

## 650. Re-Evaluating Continual Learning with Few-Shot Adaptation

**arXiv ID:** 2606.03843 | [PDF](https://arxiv.org/pdf/2606.03843v1)

**作者:** Amogh Inamdar `[一作]` (Columbia University), Richard Zemel `[通讯]` (Columbia University)

**通讯引用:** 38356 | [OpenAlex ID](https://openalex.org/A5000111344)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出以少样本适应为核心的持续学习评估范式，并引入了每次样本适应度量（per‑shot plasticity）以及前瞻性元学习方法来提升模型的学习‑to‑learn 能力。

**💡 创新点**

创新点包括：①从零样本评估转向少样本评估，揭示了忘却并非灾难性且可通过少量样本快速恢复；②设计了 SAUCE（Scaled Area Under the Adaptation Curve）度量，量化模型对新任务的适应速度；③提出前瞻性元学习（foresight meta‑learning）机制，在训练时利用未来任务样本进行元学习，显著提升了前向与后向的适应性。

**🔧 技术方法**

采用的技术主要有：少样本适应（k‑shot fine‑tuning）、元学习框架（MAML、Reptile、Meta‑SGD）、传统持续学习方法（ER、DER++、EWC、AGEM、MER）、ResNet‑18/MLP backbone，以及SAUCE指标的实现。

**📊 数据集**

使用了多种视觉数据集的任务序列，包括：MNIST 5 任务二分类、CIFAR‑100 的 10‑way 10 任务与 5‑way 20 任务（随机与超类划分）、以及 MNIST 旋转的 10‑way 20 任务域增量序列。

**📈 对比分析**

通过在每个检查点对所有已学与未学任务进行 0–10‑shot 评估，比较了上述方法的后向稳定性、前向可塑性以及 per‑shot plasticity。结果显示：1）少样本评估大幅降低了“遗忘”，即使非稳定性方法亦能恢复；2）传统可塑性指标无法区分方法，少样本前向转移能够区分其可塑性；3）加入前瞻性元学习后，绝大多数方法的 SAUCE 明显下降，体现出学习‑to‑learn 行为；整体上，SGD 在少样本场景下仍表现为最快适应；回放方法在高任务重叠时仍优于其他方法。

**⚠️ 局限性**

局限性包括：①在大规模任务序列下，元学习的超参数搜索受限，缺乏系统性调优；②对前瞻性元学习提升机制的理论解释尚不充分；③实验仅聚焦于视觉分类任务，未探讨在大型语言或跨模态模型中的适用性。

---

## 651. Text-attributed Graph Condensation via Text Selection and Attribute Matching

**arXiv ID:** 2606.03839 | [PDF](https://arxiv.org/pdf/2606.03839v1)

**作者:** Haowei Han `[一作]` (Wuhan University), Xiao Yan `[通讯]` (Wuhan University)

**通讯引用:** 27875 | [OpenAlex ID](https://openalex.org/A5100459416)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了专门针对无标签文本属性图（TAG）的数据集压缩方法TAGSAM

**💡 创新点**

创新点在于：①子图文本选择实现可读、压缩的文本；②属性相似度匹配取代训练轨迹匹配，提升训练稳定性；③在无标签场景下保持高准确率

**🔧 技术方法**

采用子图文本选择、属性相似度匹配、双层优化（teacher‑student）以及基于BERT/GCN的对比学习框架

**📊 数据集**

使用Cora、ArXiv、Photo、Computers、Products五个公开文本属性图数据集

**📈 对比分析**

与六个基线（随机、Herding、K‑Center、GCond*、MTT‑TAG、MTT‑KNN）对比，平均提升4.9%准确率，压缩比例仅1%仍能保持与全图相近的性能

**⚠️ 局限性**

对极大规模图仍需多轮教师训练；文本子图大小和粒度选择对压缩效果敏感，过大子图会导致语义混杂

---

## 652. Finding Needles in the Haystack: Transductive Active Labeling in Ecology

**arXiv ID:** 2606.03821 | [PDF](https://arxiv.org/pdf/2606.03821v1)

**作者:** Rupa Kurinchi-Vendhan `[一作]` (Massachusetts Institute of Technology), Sara Beery `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2440 | [OpenAlex ID](https://openalex.org/A5055165096)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

探讨并验证生态学固定池主动标注的转导式评估框架

**💡 创新点**

提出基于稀有类别发现难度的指标和基于稀有性与预测性能的混合停止准则

**🔧 技术方法**

利用冻结的Perch 2.0/DINOv3 表示、线性探针以及多种主动采样策略

**📊 数据集**

在BEANS、ReefSet、Snapshot Safari等多模态生态音频与图像数据集上评估

**📈 对比分析**

相较传统基于Held‑out的评估，转导式指标显示主动学习对稀有类别的发现效果显著提升；混合停止准则在稀疏长尾场景下稀有类恢复率提高约10%

**⚠️ 局限性**

仍缺乏在线稀有类别发现度量，且需进一步探索更有效的稀有样本激活策略

---

## 653. Learning finite viscoelasticity with DAVIS: A supervised framework for generalized standard materials

**arXiv ID:** 2606.03816 | [PDF](https://arxiv.org/pdf/2606.03816v1)

**作者:** Simon Wiesheier `[一作]` (Friedrich-Alexander-Universität Erlangen–Nürnberg), Miguel Angel Moreno-Mateos `[通讯]` (Friedrich-Alexander-Universität Erlangen–Nürnberg)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了改进版 DAVIS 框架，利用曲率参数化的样条表示和块交替域自适应的识别策略，对多支 Maxwell 结构的有限变形粘弹性材料进行数据驱动辨识。

**💡 创新点**

创新点在于：①通过曲率系数与边界斜率来强制构造样条的凸性与单调性，消除传统价值参数化中的约束不稳定；②将样条域的自适应与参数优化分离，采用块交替迭代消除尺度模糊；③加入组稀疏正则化实现自动分支选择。

**🔧 技术方法**

采用的技术包括：曲率基样条（softplus 约束）、块交替优化（inner 误差最小化 + outer 统计域更新）、平滑上尾统计（log‑sum‑exp）来更新样条端点、梯度基最小二乘求解、组稀疏 ℓ^p 正则化，并在 MATLAB lsqnonlin 中实现。

**📊 数据集**

使用 VHB 4910 拉伸实验数据集（不同拉伸速率 λ̇ = 0.01, 0.03, 0.05，最大伸长 λ_max = 1.5–3.0）作为训练和验证集。

**📈 对比分析**

通过多起始点鲁棒性测试与原始 DAVIS（值参数化 + 联合域优化）对比，改进版在所有起始点收敛到同一解且误差显著降低；在不同 Maxwell 分支数（1~5）和样条分辨率（n=5,20）下，模型对参数不敏感，预测误差均在 10⁻⁶ 级别，验证数据拟合优良。

**⚠️ 局限性**

局限性包括：仅在均匀单轴加载实验中验证，缺乏对多尺度或三维实验数据的推广；组稀疏正则化需要手动调参 λ_sparse，且在高度非线性或极端条件下的收敛性尚未彻底验证；计算成本随着分支数和样条分辨率增大而线性增加。

---

## 654. AI Agents Enable Adaptive Computer Worms

**arXiv ID:** 2606.03811 | [PDF](https://arxiv.org/pdf/2606.03811v1)

**作者:** Jonas Guan `[一作]` (University of Toronto), Nicolas Papernot `[通讯]` (University of Toronto)

**通讯引用:** 22408 | [OpenAlex ID](https://openalex.org/A5018809423)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

构建并评估了基于单 GPU 开源 LLM 的自适应 AI 驱动蠕虫，能在异构网络中生成针对目标的攻击逻辑并自我复制。

**💡 创新点**

首次演示了本地化、无 API、可自我复制的 AI 蠕虫；引入基于阶段的推理图、分层记忆、工具抽象和技能库的 agent harness，并证明可即时利用新披露漏洞。

**🔧 技术方法**

使用单 GPU 公共权重 LLM、阶段化 agent harness、推理图、分层记忆、工具与技能抽象、群体协同以及超虚拟机隔离等技术。

**📊 数据集**

自构建 FakeCorp 虚拟网络（33 台主机，涵盖 Ubuntu、Windows、IoT 等系统），植入多种 CVE（如 EternalBlue、PrintNightmare 等）和 CWEs，并包含 2026 年新披露的 CVE。

**📈 对比分析**

在 15 次实验中平均探测 31.3 个漏洞、成功利用 23.1 台主机、复制 20.4 台，整体成功率 73.8%/61.8%；漏洞检测 82%、利用 44%、复制 88%；相较传统蠕虫具备零边际成本和自适应能力。

**⚠️ 局限性**

受限于小型模型，利用成功率受编码错误影响；实验环境全易受攻击且无主动防御，未检验稀疏网络和实际检测；缺乏隐蔽性特征，无法评估实战持久性。

---

## 655. Easy-to-Use Shielding for Reinforcement Learning

**arXiv ID:** 2606.03804 | [PDF](https://arxiv.org/pdf/2606.03804v1)

**作者:** Stefan Pranger `[一作]` (Graz University of Technology), Bettina Könighofer `[通讯]` (Graz University of Technology)

**通讯引用:** 737 | [OpenAlex ID](https://openalex.org/A5042655793)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

开发了一个Python库 TempestShield，将 Shield 合成与 Gymnasium API 集成，提供端到端的安全强化学习工作流，并扩展了 MiniGrid 的安全环境，自动生成 PRISM 模型。

**💡 创新点**

实现了 Python 化的 Shield 合成接口、自动化环境到 PRISM 模型的转换、SMG 的安全 Shield 计算，以及预/后 Shield 包装器，显著降低了 Shield 的应用门槛。

**🔧 技术方法**

利用 PRISM 和 PRISM-games 进行概率模型检查与策略合成，配合 Python 绑定、Gymnasium 环境包装器、MiniGrid 扩展，以及 SMG 安全策略算法。

**📊 数据集**

使用基于 MiniGrid 的安全扩展环境，包括带概率转移和对抗代理的网格世界，作为实验数据集。

**📈 对比分析**

在上述环境中对比 Shielded 与 Unshielded RL 的学习曲线和安全指标，实验表明 Shielded RL 能在保持安全约束的同时实现与 Unshielded 类似的学习效率，且离线 Shield 合成开销可接受。

**⚠️ 局限性**

目前仅支持显式状态模型的 Shield 合成，SMG 求解受限于离散状态空间，缺乏符号模型检查功能，未来计划扩展到符号模型和更大规模环境。

---

## 656. Seg2Track++: Probabilistic Track Validation and Data Association for Multi-Object Tracking and Segmentation

**arXiv ID:** 2606.03875 | [PDF](https://arxiv.org/pdf/2606.03875v1)

**作者:** Diogo Mendonça `[一作]` (University of Coimbra), Urbano J. Nunes `[通讯]` (University of Coimbra)

**通讯引用:** 8694 | [OpenAlex ID](https://openalex.org/A5011728288)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了零射击的多目标跟踪与分割框架 Seg2Track++，通过集成 SAM2 分割模型和新的轨迹管理与验证模块，实现了更稳健的对象追踪与分割。

**💡 创新点**

创新点包括：① Mask Centroid Distance（MCD）与 Confidence-Aware Cost Modulation（CCM）两种关联策略，提升遮挡和运动不确定性下的匹配精度；② 基于 Bernoulli 滤波的 Probabilistic Track Validation（PTV），通过概率推理及时抑制假轨与误检；③ 端到端零射击设计，避免对特定数据集的微调。

**🔧 技术方法**

使用技术包括：SAM2 实例分割模型、YOLO11-seg 预训练分割框架、Hungarian 算法进行全局匹配、MCD/CCM 关联成本计算、Bernoulli 滤波进行轨迹存在性估计、Mask Centroid 计算与 IoU 验证。

**📊 数据集**

评估数据集为 KITTI Multi-Object Tracking and Segmentation（KITTI MOTS）基准，包含21条训练序列和29条测试序列，提供完整的像素级分割标签。

**📈 对比分析**

与 ViP-DeepLab、EagerMOT、OPITrack、ReMOTS、SearchTrack、MOTSFusion、PointTrack、TrackR-CNN 等方法以及 Seg2Track-SAM2 基线进行对比；在零射击设置下，Seg2Track++ 在车类实现 74.56% HOTA、60.20% HOTA（行人），相较基线提升约 0.4% HOTA，检测准确率和关联准确率均有提升，整体性能表现具竞争力。

**⚠️ 局限性**

局限性包括：① 在极端遮挡或快速运动场景下仍可能出现关联误差导致 AssA 略低；② 阈值设定（IoU、MCD、PTV 存在概率阈值）对不同场景敏感，需要进一步自动化调优；③ 仅基于视觉信息，未结合深度或激光雷达等多模态数据，导致在低光/雨雾等条件下性能下降。

---

## 657. Visual Instruction Tuning Aligns Modalities through Abstraction

**arXiv ID:** 2606.03871 | [PDF](https://arxiv.org/pdf/2606.03871v1)

**作者:** Luis Palacios `[一作]` (Area Science Park), Alberto Cazzaniga `[通讯]` (Area Science Park)

**通讯引用:** 199 | [OpenAlex ID](https://openalex.org/A5005920708)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究视觉指令调优如何在LLM层级中嵌入视觉特征，并通过因果干预、探测与几何对齐等方法定位其作用层次。

**💡 创新点**

证明视觉指令调优的主要作用集中在LLM的中间抽象层，且仅在这些层进行微调即可保持全模型性能，揭示多模态整合的“局部”机制。

**🔧 技术方法**

采用注意力屏蔽、层跳过干预、信息不平衡度量、线性CKA、邻域重叠、Logit Lens、线性探测器等多种因果与表示相似度分析技术。

**📊 数据集**

在七种开放VLM（如LLaVA、OneVision、InternVL2、Cambrian、Llama‑3.2‑Vision‑90B等）上使用GQA、MMBench、MME、VQAv2、COCO‑QA、SeedBench、ScienceQA、ChartQA、CountBench、NaturalBench、Captioning等多模态与文本基准。

**📈 对比分析**

通过对比完整微调与仅微调中间层的结果，发现中间层微调可恢复约95‑100%的原始性能，并将训练时间降低14‑24%；在视觉主导任务上表现更优。

**⚠️ 局限性**

仅限于可公开的模型与检查点，未涉及音频/视频等其他模态；局部层选取基于经验而非全局最优；实验聚焦视觉，未验证其他多模态融合策略。

---

## 658. Formalizing all indexed mathematics as a benchmark for general reasoning, with the example of implementing dilatations of categories

**arXiv ID:** 2606.03835 | [PDF](https://arxiv.org/pdf/2606.03835v1)

**作者:** A. Mayeux `[一作]` `[通讯]` (University of Wisconsin), A. Mayeux (University of Wisconsin)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出将所有已索引数学论文进行机理化形式化的基准任务，并以范畴理论中“类别的扩张（dilatations）”为案例实现了 Lean 4 中的形式化。

**💡 创新点**

创新点在于：①首次将完整数学文献的形式化作为可衡量的长周期目标提出；②展示了在已有大型形式化库（mathlib）基础上扩展新概念（扩张）的可行性；③探讨了如何构建与传统数学索引数据库（MathSciNet、Reservoir）互联的元数据框架。

**🔧 技术方法**

使用技术包括 Lean 4 交互式定理证明器、mathlib 形式化库、Reservoir 的包索引与依赖追踪，以及自定义的 “LocQuiver”、“Sieve”等结构来实现类别的局部化和扩张。

**📊 数据集**

数据集为 MathSciNet 中约 4.5 百万篇已索引论文，讨论了将每篇论文转换为 Lean 形式化所需的存储空间（约 4.5 TB–450 TB）以及人工与 AI 辅助形式化的工时估算。

**📈 对比分析**

本文未给出实验对比或性能数值，而是通过理论估算说明：在人工 + AI 辅助的情形下，形式化工时可下降 5–20 倍，最终预计所需人年从数十亿降至千余万年；同时强调需要与数学数据库互联以实现实时更新。

**⚠️ 局限性**

限制包括：①形式化工作对现有文献的清晰度、完整性和一致性高度依赖；②大量论文可能包含错误或模糊，需先做修订；③现阶段 AI 生成的形式化仍易卡壳，难以自动完成高级数学；④跨版本 Lean 迁移和兼容性问题；⑤全局形式化尚无法保证足够的社区参与和长期维护。

---

## 659. Conditional Latent Diffusion Model with Fourier-based Motion Modelling for Virtual Population Synthesis

**arXiv ID:** 2606.03827 | [PDF](https://arxiv.org/pdf/2606.03827v1)

**作者:** Shaokun Lan `[一作]` (University of Manchester), Alejandro F. Frangi `[通讯]` (University of Manchester)

**通讯引用:** 33607 | [OpenAlex ID](https://openalex.org/A5049192404)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了4D F-MeshLDM框架，实现可控、周期一致的心脏3D+t网格生成；

**💡 创新点**

首次在潜在空间使用截断傅里叶级数参数化运动，保证理论上无周期闭合误差，并将条件扩散模型应用于Fourier系数生成；

**🔧 技术方法**

采用Mesh VAE（CoMA）提取空间特征、傅里叶拟合表示运动、Transformer-based Denoising Diffusion Probabilistic Model（DDPM）结合AdaLN注入临床共变量、逆Fourier合成与VAE解码；

**📊 数据集**

使用UK Biobank 5,000名受试者的时序双心室表面网格（T=50帧）进行训练与评估；

**📈 对比分析**

与CHeart、4D CardioSynth和CardiacFlow等三种基线进行对比，指标包括特异性、覆盖率、序列RMSE、周期一致性、体积/网格平滑度以及临床指标MAE，4D F-MeshLDM在所有指标上均优于基线，周期一致性接近零；

**⚠️ 局限性**

当前方法受限于固定拓扑，难以处理严重病理变形；对多样心率的适应性尚未完全验证；仅使用年龄和性别两项临床共变量。

---

## 660. Dynamic Short Convolutions Improve Transformers

**arXiv ID:** 2606.03825 | [PDF](https://arxiv.org/pdf/2606.03825v1)

**作者:** Oliver Sieberling `[一作]` (Massachusetts Institute of Technology), Yoon Kim `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 22382 | [OpenAlex ID](https://openalex.org/A5100693798)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出动态短卷积（Dynamic Short Convolutions）作为Transformer的新原语，增强局部上下文聚合并提升语言模型性能。

**💡 创新点**

创新点在于将卷积核设为输入相关的动态生成，既保留卷积的局部偏置，又显著提升表达能力；同时提供低秩和头划分两种参数化方案并在多层Transformer中实现。

**🔧 技术方法**

主要技术包括：动态卷积的低秩/头划分生成器、与RoPE结合的QKV卷积插入、定制Triton内核加速、以及对线性RNN（Mamba、DeltaNet）和Mixture‑of‑Experts的适配。

**📊 数据集**

使用了Nemotron‑CC大规模文本数据进行语言建模，合成MQAR（多查询关联回忆）任务和RULER检索任务做评测，并在WikiText‑103、LAMBADA等基准上验证。

**📈 对比分析**

与标准Transformer、带静态卷积的Transformer以及各类动态卷积放置方式进行对比；在150M–2B参数范围内，动态卷积在计算匹配模型上平均提升约1.33×，在所有线性层放置时提升约1.60×；在MoE与线性注意力模型中亦显著降低困惑度并提升零样本推理准确率。

**⚠️ 局限性**

局限性包括：实验规模仅至2B参数及7B MoE，尚未验证更大规模或更长训练；Triton实现仅针对H100 GPU，推理端优化不足；仅探讨了少数参数化与放置方案，未覆盖所有可能组合。

---

## 661. Rethinking the Idiomaticity Decomposability Hypothesis: Evidence from Distributional Learning

**arXiv ID:** 2606.03817 | [PDF](https://arxiv.org/pdf/2606.03817v1)

**作者:** Maggie Mi `[一作]` (University of Sheffield), Nafise Sadat Moosavi `[通讯]` (University of Sheffield)

**通讯引用:** 520 | [OpenAlex ID](https://openalex.org/A5054918343)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用上下文语言模型作为分布式学习者，对成语可分解性、句法灵活性与使用频率的关系进行系统评估，并追踪其在预训练过程中的学习动态

**💡 创新点**

首次在无显式语义结构的分布式学习框架中引入模型内部可分解性度量，并证明频率、预期性（surprisal）比单纯可分解性更能解释成语学习稳定性

**🔧 技术方法**

采用双向Transformer（BERT系列、ModernBERT）与大规模语言模型（OLMo）进行隐层表示提取，使用相似度、掩码等技术计算可分解性与句法灵活性

**📊 数据集**

主要使用IMPLI成语句子-释义对照集、enTenTen语料库提取频率与构造分布，结合BERT预训练数据估计预期性

**📈 对比分析**

通过Spearman相关、线性回归与多重比较控制，发现模型可分解性与句法灵活性相关性弱且往往为负；预训练阶段频率、预期性、可分解性三者交互显著且可分解性在早期影响最大

**⚠️ 局限性**

限制包括：可分解性度量仅基于单一架构（BERT）推断，可能对不同模型产生偏差；只研究英语成语，跨语言推广未知；以及模型内部可分解性与人类直觉不完全重合

---

## 662. Enhancing Operational Safety via Agentic Dialogue Hazard Identification Analysis

**arXiv ID:** 2606.03812 | [PDF](https://arxiv.org/pdf/2606.03812v1)

**作者:** Sanjay Das `[一作]` (Oak Ridge National Laboratory), Tirthankar Ghosal `[通讯]` (Oak Ridge National Laboratory)

**通讯引用:** 864 | [OpenAlex ID](https://openalex.org/A5081072666)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过多轮多代理对话（对抗辩论与合作讨论）来改进工业安全中的危害识别任务，并提出遗传算法优化对话配置的框架。

**💡 创新点**

提出了Hazard Identification Dialogue（HID）框架，系统比较了对抗辩论与合作讨论两种对话结构，并首次将遗传优化应用于对话策略参数，实现了自动化的对话调优。

**🔧 技术方法**

采用多代理大型语言模型（GPT-OSS 20B 与 GPT-4.1）、对抗辩论与合作讨论对话模板、链式思维提示、遗传算法优化以及 Fβ（β=2）评价准则。

**📊 数据集**

使用真实工况描述与专家标注的 HazID‑Ops 数据集（213 条工作描述及其对应的标准化危害标签列表）。

**📈 对比分析**

将单轮基线（Base）、对抗辩论（Debate）、合作讨论（Discuss）以及遗传优化的对抗辩论（GA‑Debate）在两大模型上进行对比，评估精度、召回率、F1、准确率以及多种对话效率指标；结果显示对抗辩论在 GPT‑OSS 20B 上提升 F1，合作讨论在 GPT‑4.1 上显著提升召回率，而 GA‑Debate 在 GPT‑4.1 上进一步提升 F1 并保持召回率。

**⚠️ 局限性**

存在标签解析误差导致对话效率指标超过 1，遗传优化的探索阶段计算成本高，以及使用 0.60 阈值的模糊 Jaccard 匹配无法捕捉词形变化。

---

## 663. Denoise First, Orthogonalize Later: Understanding Momentum in Muon via Spectral Filtering

**arXiv ID:** 2606.03899 | [PDF](https://arxiv.org/pdf/2606.03899v1)

**作者:** Xianliang Li `[一作]` (Institute of Statistical Mathematics), Han Bao `[通讯]` (Institute of Statistical Mathematics)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5111633125)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了Muon's Momentum在矩阵优化器中的作用，证明其在梯度流中充当谱滤波器，先抑制噪声后再进行正交化；

**💡 创新点**

创新点在于首次把Momentum视为谱滤波器并量化其对谱间隙和奇异子空间可靠性的提升，解释了Pre‑polar优于Post‑polar和Polar‑only的原因；

**🔧 技术方法**

采用谱分析、矩阵分解（Thin SVD、极限迭代）、Wedin定理、EM平均、理论证明和数值仿真等技术；

**📊 数据集**

使用NanoGPT、LLaMA 350M、CIFAR‑10以及合成的spiked‑perturbation模型进行实验；

**📈 对比分析**

通过与Polar‑only和Post‑polar三种管线的对比，展示Pre‑polar在信号对齐、子空间误差和整体训练损失上均优于其他方法，尤其在大模型预训练中表现显著；

**⚠️ 局限性**

局限包括对时间变换的Momentum分析缺失、只在rank‑1高斯模型下证明了信号恢复，未扩展到一般rank‑r或非高斯扰动，以及对动态Momentum调度的理论支持不足。

---

## 664. Collision Resistance of Single-Layer Neural Nets

**arXiv ID:** 2606.03807 | [PDF](https://arxiv.org/pdf/2606.03807v1)

**作者:** Marco Benedetti `[一作]` (Bocconi University), Riccardo Zecchina `[通讯]` (Bocconi University)

**通讯引用:** 13274 | [OpenAlex ID](https://openalex.org/A5041685522)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

本研究探讨了在单层二进制神经网络中寻找碰撞的算法复杂性，提出了一种在线算法来高效生成广泛的碰撞，并证明了在特定条件下碰撞空间具有重叠间隙属性（OGP），从而为在线算法提供了指数下界。

**💡 创新点**

首次将重叠间隙属性作为碰撞抗性的重要标准，揭示了碰撞发现与平均情况搜索之间的关键区别，并提出了新的“最坏情况”方面。

**🔧 技术方法**

使用了在线算法和重叠间隙属性（OGP）作为主要技术手段，分析了不同激活函数下的碰撞发现能力。

**📊 数据集**

使用了随机生成的矩阵作为输入数据集，假设输入为二进制向量，且激活函数为随机振荡激活或其他标准激活函数。

**📈 对比分析**

通过与现有算法的比较，证明了在特定参数范围内，提出的在线算法在寻找广泛碰撞方面表现优越，且在重叠间隙属性的条件下，在线算法的成功概率呈指数衰减。

**⚠️ 局限性**

限制在于目前的下界证明仅适用于在线算法，如何将这些保证扩展到更广泛的算法类别（如谱、代数、基于格的或量子方法）仍然是一个开放的研究方向。

---

## 665. Self-Refining Agentic Reinforcement Learning for Vision-Conditioned UAV Navigation

**arXiv ID:** 2606.03963 | [PDF](https://arxiv.org/pdf/2606.03963v1)

**作者:** Roohan Ahmed Khan `[一作]`, Dzmitry Tsetserukou `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 AgenticRL 框架，通过多模态 GPT 代理生成奖励函数，使用 PPO 训练无人机策略，进行行为诊断并迭代优化，最终实现仿真到真实世界的高效部署。

**💡 创新点**

创新点包括：① 在闭环中利用多模态 GPT 自动生成并改进奖励函数；② 引入场景识别注册表，使无人机在实测时能根据视觉+文本信息自动选择对应的最佳策略；③ 在多种任务中实现高达 94% 的 sim‑to‑real 转移精度。

**🔧 技术方法**

使用的技术有：GPT‑5.5 / GPT‑4o‑mini 作为多模态奖励生成与诊断代理；Proximal Policy Optimization (PPO) 进行策略学习；视觉语言模型用于场景理解；行为诊断包与奖励代码自动化生成与迭代。

**📊 数据集**

使用的数据集包括：自定义的多场景仿真图像与文本描述（用于奖励生成与诊断），以及真实环境下收集的 100 次仿真试验和 10 次物理无人机试验数据。

**📈 对比分析**

通过与零射/少射奖励、无分析器、无视觉条件等 ablation 进行对比，Full AgenticRL 在 5 个任务中分别获得 95%–100% 的 SSR、91% 的 RSR、94% 的 S2R，并且奖励改进率达到 71%。

**⚠️ 局限性**

局限性包括：依赖多模态代理的推理质量；每轮奖励迭代需要完整的 PPO 训练，导致高计算成本；仍需人工设定终止条件、安全约束和观测空间；在更复杂动态环境下的验证不足。

---

## 666. The Impact of Configuring Agentic AI Coding Tools on Build-vs-Buy Decisions: A Study Protocol

**arXiv ID:** 2606.03907 | [PDF](https://arxiv.org/pdf/2606.03907v1)

**作者:** Jai Lal Lulla `[一作]` (Singapore Management University), Christoph Treude `[通讯]` (Singapore Management University)

**通讯引用:** 5362 | [OpenAlex ID](https://openalex.org/A5077658936)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一套预注册实验协议，系统评估了配置机制对代理式 AI 代码工具在“build‑versus‑buy”决策中的影响。

**💡 创新点**

创新点在于首次对代理式 AI 工具的依赖决策进行可控实验，并构建了可复现的多项目多阶段基准及相关分析工具。

**🔧 技术方法**

技术上使用了 Claude Code 与 OpenAI Codex 两款主流工具，结合上下文文件、Skill、MCP 服务器等配置机制，并采用混合效应模型和非参数检验进行统计分析。

**📊 数据集**

数据集为 5 个项目（HTTP 服务器、BitTorrent 客户端、Unix shell、Redis、Kafka）在 Python 与 JavaScript 中各 5 阶段，共计 25 个决策点，配备了手工审核的决策标签和可公开的配置文件。

**📈 对比分析**

通过对比不同配置条件下的买卖决策率、遵从度、披露完整性等指标，发现配置强度和形式对决策有显著影响；实验表明禁止式指令效果更好，披露完整性仍有缺口。

**⚠️ 局限性**

局限在于仅评估两款闭源工具、固定模型版本、单次实验单元，且基准覆盖范围有限，难以推广至其它语言或开源工具。

---

## 667. Preference-Calibrated Human-in-the-Loop Reinforcement Learning for Robotic Manipulation

**arXiv ID:** 2606.03949 | [PDF](https://arxiv.org/pdf/2606.03949v1)

**作者:** Zeyi Liu `[一作]` (Central South University), Ziwei Wang `[通讯]` (Nanyang Technological University)

**通讯引用:** 3279 | [OpenAlex ID](https://openalex.org/A5100389366)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `a4b10f5d-130b-4e77-9367-6469ec621899` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出PACT框架，利用人类干预产生的偏好信号，对成功轨迹中的子优化段进行信用校正，并在Actor-Critic训练中直接引导策略趋近人类修正动作。

**💡 创新点**

创新点包括：1）将成功轨迹拆分为正常、子优化、人工干预三段，并用自监督进度模型定位子优化段；2）基于偏好构造的counterfactual advantage对Bellman目标做方向性校正；3）在Actor侧加入偏好辅助损失，使策略直接对齐人类偏好动作。

**🔧 技术方法**

技术手段：自监督轻量进度模型（ResNet+MLP编码图像+本体信息）、偏好校准的counterfactual advantage、Actor-Critic（RLPD+TD3）框架、偏好辅助策略优化、经验回放缓冲区。

**📊 数据集**

数据集：收集了5个真实机器人操控任务（Press、Insertion、Pick、Pick & Place、Assembly）中的20条演示轨迹，用于训练进度模型；所有实验均在Galaxea A1X 6-DoF机械臂上完成。

**📈 对比分析**

与HIL‑SERL基线比较，使用成功率、干预率和训练时间三项指标。PACT平均成功率提升至82.5%（↑24.5%），干预率下降至32.3%（↓14.8%），训练时间缩短至63.0分钟（约1.3×快），在所有任务上均显著优于基线。

**⚠️ 局限性**

局限性：进度模型依赖演示的单调性，非单调或重复轨迹时定位误差会影响后续信用校正；偏好校正只能提供方向性调整，无法精确到每一步；若子优化段定位错误会误导Critic学习；对长时延任务的泛化能力尚待验证。

---

## 668. A Pocket Offline Model for Simultaneous Speech Translation as CUNI Submission to IWSLT 2026

**arXiv ID:** 2606.03948 | [PDF](https://arxiv.org/pdf/2606.03948v1)

**作者:** Aziz Sharipov Ortega `[一作]` (Charles University), Dominik Macháček `[通讯]` (Charles University)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5056031235)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

实现了基于Canary‑1B‑v2的在线同声传译系统，使用AlignAtt策略在Nemo框架中进行增量解码并加入Silero VAD以抑制噪声与幻觉；

**💡 创新点**

创新点在于将大型离线语音翻译模型Canary通过增量前缀注入与交叉注意力截断成功迁移至实时模式，并在Nemo中实现了必要的增量推理支持；

**🔧 技术方法**

使用的技术包括Canary‑1B‑v2离线模型、AlignAtt同声传译策略、Silero Voice Activity Detection、Nemo增量解码扩展、CPU量化推理潜力；

**📊 数据集**

使用的数据集为IWSLT 2026的MCIF（英语→德语、意大利语）和Czech Chamber of Deputies会议录（捷克→英语）以及IWSLT 2026的dev集；

**📈 对比分析**

与组织者的Cascade基线、Whisper‑AlignAtt基线以及Canary滑动窗口实现对比，系统在低、高清延迟下在BLEU/chrF/XCOMET‑XL上均显著提升，特别是高延迟场景BLEU提升4‑8点；

**⚠️ 局限性**

局限性在于无法有效利用模型上下文/强制前缀导致停滞，且对低延迟下的质量提升仍有待改进，未来需探索量化版本和更好的上下文注入方式。

---

## 669. Entropy Is Not Enough: Unlocking Effective Reinforcement Learning for Visual Reasoning via Vision-Anchored Token Selection

**arXiv ID:** 2606.03937 | [PDF](https://arxiv.org/pdf/2606.03937v1)

**作者:** Senjie Jin `[一作]` (Fudan University), Xuanjing Huang `[通讯]` (Fudan University)

**通讯引用:** 17221 | [OpenAlex ID](https://openalex.org/A5088834359)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文研究了视觉推理任务中基于 token‑level 熵的信用分配失效现象，并提出了 VEPO 框架，将视觉敏感性与 token 熵相结合，实现更有效的策略优化。

**💡 创新点**

创新点在于首次将视觉敏感性信号（JSD 与 entropy gap）与 token 熵通过乘法耦合联合用于 token 选择，克服纯熵机制忽略低熵视觉关键 token 的问题。

**🔧 技术方法**

技术上采用 RL‑with‑verifiable‑rewards（GRPO）框架，计算原图与噪声图间的 JSD 与 entropy gap，构造乘法聚合与加法融合的 token 选择机制，并在 Qwen2.5‑VL‑7B/3B‑Instruct 上进行强化学习微调。

**📊 数据集**

数据集包括 Geometry3K 与 MMK12 作为训练集，评估基准涵盖 Geo3K、MMK12、Hallucination Bench、MathVista、We‑Math、MathVerse、MathVision 等视觉推理与数学推理任务。

**📈 对比分析**

与全量 token GRPO、top‑entropy 选择、VPPO、PAPO‑DAPO 等方法比较，VEPO 在 7B 规模平均提升 2.28 分，3B 规模提升 3.15 分，在 Geo3K 与 MMK12 等基准上提升 3–5 分，整体性能优于所有对照方法。

**⚠️ 局限性**

局限性包括：需要额外的前向传播产生噪声图导致训练开销上升；仅使用 Gaussian 噪声，未尝试更丰富的扰动或调度策略；缺乏对视觉敏感 token 选择为何能提升优化效果的理论分析。

---

## 670. Correcting Neural Operator Spectral Bias via Diffusion Posterior Sampling with Sparse Observations

**arXiv ID:** 2606.03936 | [PDF](https://arxiv.org/pdf/2606.03936v1)

**作者:** Niccolò Perrone `[一作]` (Université Paris-Saclay, CentraleSupélec, CNRS, ENS Paris-Saclay), Filippo Gatti `[通讯]` (Université Paris-Saclay, CentraleSupélec, CNRS, ENS Paris-Saclay)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种利用冻结的神经算子预测作为辅助观测，在扩散后验采样中校正其频谱偏差，从而实现高精度的三维弹性波场重建。

**💡 创新点**

创新点在于提出一种频谱加权的闭式神经算子似然引导，结合LMMSE边缘化实现对频率依赖的校准，并给出分布无关的误差界。

**🔧 技术方法**

使用了无条件的基于分数的扩散模型、扩散后验采样、频域LMMSE估计以及FFT等技术。

**📊 数据集**

使用了HEMEWS-3D数据库中的高保真谱元仿真波场数据。

**📈 对比分析**

与仅用神经算子、仅用传感器条件化扩散和等基线相比，在5%和2%传感器覆盖率下，方法在频谱偏差上几乎为零，点误差和频谱误差均显著优于基线。

**⚠️ 局限性**

主要限制包括对频谱对角残差假设的依赖、需要从高保真数据中估计统计量、扩散采样计算量大以及在真实传感器噪声和分布移位时的鲁棒性不足。

---

## 671. Physics-Informed Single Atom Matching Pursuit: Guided-Waves Wavenumbers and Propagation Distance Estimation for Damage Localization in Structural Health Monitoring

**arXiv ID:** 2606.03933 | [PDF](https://arxiv.org/pdf/2606.03933v1)

**作者:** Sebastian Rodriguez `[一作]` (Arts et Metiers ParisTech), Marc Rébillat `[通讯]` (Arts et Metiers ParisTech)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于物理约束的单原子匹配追踪（PISAMP）方法，用于引导波结构健康监测中的信号分解和损伤定位；

**💡 创新点**

创新点在于将波传播物理（波数函数与传播距离）嵌入低维匹配追踪框架，能够直接提取可解释的传播距离和模态特征，并通过椭圆定位实现损伤定位；

**🔧 技术方法**

使用了物理信息化单原子匹配追踪、Chebyshev多项式逼近、非线性最小二乘优化、椭圆定位和直接路径标定等技术；

**📊 数据集**

数据集包括：理想化的等向薄板模拟信号、A380风扇舱实验信号以及二维正方形薄板的仿真损伤数据；

**📈 对比分析**

与传统匹配追踪及经验分解方法对比，PISAMP在简单信号中的重构误差约为4%，在复杂实验信号中的误差约为20%；在损伤定位实验中，平均相对误差分别为2.35%（x）和2.26%（y），显示出较高的精度；

**⚠️ 局限性**

局限性包括：仅适用于平面等向结构，需要先验或标定以确定尺度；对第一到达波包的提取仍具挑战；仅适用于单次散射假设，未在真实损伤实验中验证；对高度各向异性或曲面结构的适用性有限。

---

## 672. Contrastive Neural Algorithmic Reasoning for Graph Coloring

**arXiv ID:** 2606.03923 | [PDF](https://arxiv.org/pdf/2606.03923v1)

**作者:** Thien Le `[一作]` (Harvard University), Melanie Weber `[通讯]` (Harvard University)

**通讯引用:** 888 | [OpenAlex ID](https://openalex.org/A5034942394)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种基于对比学习的有监督图着色框架，利用绝对值 InfoNCE 损失使同色节点在表示空间中沿同一条无定向直线聚合、相邻颜色之间形成正交；通过 GNN 编码器学习节点嵌入并在推断时进行线性规范化后聚类得到最终着色。

**💡 创新点**

创新点包括：①绝对值对比损失在理论上可被精确刻画，最优解呈现“线原型”与正交结构；②通过分析梯度动力学与最大边缘（margin）偏差，证明优化趋向最大化最小正交间距；③利用正交近似得到 Lovász θ 函数的可计算上界，实现图可着色性认证；④展示该方法在不同图族与尺寸上具有优秀的迁移与规模泛化能力。

**🔧 技术方法**

技术实现主要包含：基于多种 GNN（GCN、GatedGCN、GPS、GraphSAGE 等）的节点编码器；对比学习（InfoNCE 与绝对值变体）与自监督边冲突辅助损失的组合；线性正交化与 k‑medoids/k‑means 聚类；以及理论工具（聚类稳定性、最大边缘优化、Lovász θ 上界）来分析与验证。

**📊 数据集**

实验使用了合成组合实例、真实世界的 Cora/CiteSeer/PubMed 子图与全图、以及不同图族（Book、Myciel、Queen、Cycles 等）构成的基准数据集。

**📈 对比分析**

与贪心着色、PI‑GNN、full‑GCN 等基线比较，评估指标包括相对于贪心着色的比值 ρ、单色边比例 Mono 以及 Hit‑Rate；实验表明学习方法在 ρ 与 Mono 上与贪心相当甚至更优，且在大规模循环图与 OOD 图上能够在 0.1 秒以内完成着色，显著优于基线的高计算成本。

**⚠️ 局限性**

局限性包括：梯度动力学的可保持性依赖于“均等可着色”假设，非均等图可能导致线原型不再保持；方法对编码器与特征选择高度敏感，缺乏统一的最佳配置；在非均等或更复杂图族中无法保证全局最优；且需要预先给定近似或最优着色作为监督信号，限制了应用范围。

---

## 673. Forecasting Conceptual Diffusion in Science: The Case of Quantum Computing

**arXiv ID:** 2606.03919 | [PDF](https://arxiv.org/pdf/2606.03919v1)

**作者:** Thomas Maillart `[一作]` (University of Geneva), Alain Mermoud `[通讯]` (University of Geneva)

**通讯引用:** 599 | [OpenAlex ID](https://openalex.org/A5072777494)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `14d48e9d-0069-4ad9-996a-1d5968216998` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建量化的概念共现网络，基于OpenAlex的量子计算概念子树，追踪概念对在上下游引用中的演化；

**💡 创新点**

通过引入上下游分布、熵和多样性特征，揭示了“内生自强化”与“外生扩散”之间的可预测性差异，并在多个领域验证了此异质性；

**🔧 技术方法**

采用LightGBM机器学习、SHAP解释性分析、Optuna超参数优化、Python生态（pandas、scikit‑learn、LightGBM、SHAP等）；

**📊 数据集**

OpenAlex公开知识图谱的量子计算子树（1990–2023年）以及机器人、先进材料、神经植入等对照子树；

**📈 对比分析**

对比了自强化计数、外生计数、比值和熵四个预测目标，发现外生计数和熵的R²可达0.78–0.87，内生计数在量子领域几乎不可预测（R²≈0.02），在神经植入领域则可预测性高（R²≈0.83）；

**⚠️ 局限性**

受限于OpenAlex概念标注噪声、年度时间分辨率、右侧截尾（5年窗口不完整）以及对整体增长归一化导致的绝对影响不可估计；

---

## 674. scTranslation: A Comprehensive Benchmark for Single-Cell Multi-Omics Modality Translation

**arXiv ID:** 2606.03906 | [PDF](https://arxiv.org/pdf/2606.03906v1)

**作者:** Jiabei Cheng `[一作]` (Westlake University), Stan Z. Li `[通讯]` (Westlake University)

**通讯引用:** 49238 | [OpenAlex ID](https://openalex.org/A5082786719)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个统一评估框架scTranslation，用于系统评估单细胞多组学模态翻译方法；

**💡 创新点**

提出了统一的6M数据维度、三类模型（AE、VAE、分布式）以及多层次指标评估，并对影响因素（特征选择、质量、少样本）进行系统实验；

**🔧 技术方法**

使用深度学习模型（BABEL、JAMIE、scButterfly、scDiffusion-X等）并采用NMI、ARI、PCC、MSE、MMD、LISI等多指标；

**📊 数据集**

收集了8个多模态单细胞数据集（如SNARE-seq、10x Multiome、CITE-seq等，涵盖RNA、ATAC、蛋白等）；

**📈 对比分析**

对每个模型在每个数据集上进行5折交叉验证，结果显示不同模型在聚类、回归、分布三维度上存在明显差异，未出现统一最佳模型，VAE方法表现更稳健；

**⚠️ 局限性**

对模型鲁棒性和可扩展性的影响因素仍未完全覆盖，特定方向（如Protein→RNA）对数值稳定性要求高，且对不同平台兼容性需进一步验证。

---

## 675. Semantic-weighted ICP for LiDAR Odometry: Class-Aware Residual Reweighting for Robust Scan Registration

**arXiv ID:** 2606.03905 | [PDF](https://arxiv.org/pdf/2606.03905v1)

**作者:** Vasco Carvalho `[一作]` (University of Coimbra), Urbano J. Nunes `[通讯]` (University of Coimbra)

**通讯引用:** 8694 | [OpenAlex ID](https://openalex.org/A5011728288)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种语义加权ICP方法，用于LiDAR里程计，使不同语义类别对点云配准产生不同的影响；

**💡 创新点**

创新点在于将语义类别的几何稳定性作为权重因子，软性地对ICP残差进行加权，而非硬性过滤，能够在不同环境下动态调节对不同语义类别的重视程度；

**🔧 技术方法**

技术包括SuMa++框架、RangeNet++语义分割、点到平面的加权ICP优化、Huber损失、语义兼容性函数以及自定义的类别权重函数；

**📊 数据集**

使用SemanticKITTI（城市、高速、乡村）和RELLIS-3D（越野）两个公开数据集进行实验；

**📈 对比分析**

通过与SuMa++基线以及多种类别权重配置进行对比，评估相对位姿误差（旋转误差°/100 m，平移误差%），实验表明在城市环境中可实现显著提升，而在越野环境中通过合理权重也能降低漂移，整体表现优于统一权重基线；

**⚠️ 局限性**

局限性包括：权重设置依赖环境，需人工选择合适配置；在高稀疏或重复结构的高速/乡村场景中提升不稳定；对语义分割误差敏感，误标会直接影响配准效果。

---

## 676. FFR: Forward-Forward Learning for Regression

**arXiv ID:** 2606.03927 | [PDF](https://arxiv.org/pdf/2606.03927v1)

**作者:** Xinyang Liu `[一作]` (University of Bristol), Guosheng Hu `[通讯]` (University of Bristol)

**通讯引用:** 3365 | [OpenAlex ID](https://openalex.org/A5075333422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种可用于实战回归的前向前向学习框架（FFR）。

**💡 创新点**

通过设计序数竞争好函数、分层梯形架构以及层级预测与不确定性估计三大创新，解决了传统前向前向学习在回归任务中缺失对比样本和尺度感知的问题。

**🔧 技术方法**

采用了局部前向优化、组内竞争与距离感知软标签、分层梯形网络与多尺度特征聚合、层级加权集成预测与不确定性估计等技术。

**📊 数据集**

在5个真实世界回归基准（Appliances Energy、Machine Tool Wear、UJIIndoorLoc、BIDMC、KonIQ-10k）以及4个合成函数回归任务上进行评估。

**📈 对比分析**

与传统BP、FF-CAR、Trifecta、FF-Zero、F^3、PEPITA、FF-MSE/FF-CLF等方法对比，FFR在所有基准上实现了约98.6% BP的精度，同时显著降低了内存占用和训练时间，是目前最优的BP-free回归方法。

**⚠️ 局限性**

当前实现仅在浮点精度下，未结合低位量化或硬件加速；在低位加速器、模拟电路或光学神经网络等非数字平台上的验证仍待完成。

---

## 677. Value-Aware Stochastic KV Cache Eviction for Reasoning Models

**arXiv ID:** 2606.03928 | [PDF](https://arxiv.org/pdf/2606.03928v1)

**作者:** Ting-Yun Chang `[一作]` (University of Southern California), Robin Jia `[通讯]` (University of Southern California)

**通讯引用:** 6722 | [OpenAlex ID](https://openalex.org/A5041906762)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种训练无关的 KV 缓存淘汰方法 VaSE（Value‑aware Stochastic KV Cache Eviction），通过保留大幅度值向量并加入随机采样来提升推理时长链推理模型的准确率。

**💡 创新点**

创新点在于识别并保留大幅度值向量，以及在淘汰过程中引入随机性以提升 KV 缓存多样性和推理准确性，且不需要额外训练。

**🔧 技术方法**

采用了值向量幅度评分、概率采样、周期性淘汰框架与 SnapKV、CurDKV 等现有方法的组合，并利用 FlashAttention2 进行实现。

**📊 数据集**

实验使用了六个推理任务的数据集：AIME25/26、HMMT25、GPQA‑Diamond、MATH、LiveCodeBench‑v6 以及 GSM8K 作为案例。

**📈 对比分析**

与 SOTA 淘汰方法 R‑KV、CurDKV 以及选择性稀疏注意力 SeerAttention‑R 对比，VaSE 在约 4 倍压缩下平均提升 4–5% 的准确率，且吞吐量和内存占用均优于传统淘汰方法。

**⚠️ 局限性**

局限包括对极端压缩比例下可能仍存在信息丢失、仅针对推理阶段且未考虑多模型迁移与训练成本，以及仍需在更大规模或不同任务上进一步验证。

---

## 678. FlashbackCL: Mitigating Temporal Forgetting in Federated Learning

**arXiv ID:** 2606.03939 | [PDF](https://arxiv.org/pdf/2606.03939v1)

**作者:** Mubarak A. Ojewale `[一作]` (National College of Ireland), Horacio Gonzalez-Velez `[通讯]` (National College of Ireland)

**通讯引用:** 1468 | [OpenAlex ID](https://openalex.org/A5088758379)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在分布随时间漂移的联邦学习场景下，提出FlashbackCL扩展了Flashback，加入时间衰减标签计数、设备感知的CBRS重放缓冲区以及服务器端的主动核心子集选择，解决了传统方法对旧类知识的过度依赖导致的时序遗忘问题。

**💡 创新点**

创新点在于：①引入时间衰减的全局标签计数以消除历史类信息；②采用类平衡的储备抽样（CBRS）在客户端保持各类样本均衡，避免阶段边界导致的缓冲区刷新；③通过服务器端对公共数据的趋势估计实现主动核心子集采样，进一步提升稳健性。

**🔧 技术方法**

技术方法包括：联邦平均（FedAvg）+动态知识蒸馏、时间衰减标签计数、类平衡储备抽样（CBRS）、服务器端线性回归趋势估计与软最大权重重采样、以及标准CNN与ResNet-18网络架构。

**📊 数据集**

使用的数据集为CIFAR-10与CIFAR-100，构造了50/100个客户端、10/100个类别，并设计了三种时间漂移模式（急切、渐进、循环）。

**📈 对比分析**

与FedAvg、Flashback、FedProx、FedNTD等基线相比，FlashbackCL在CIFAR-10上实现相对提升6.9%–10.0%，在CIFAR-100上提升9.7%–12.7%；同时将时序遗忘率降低多达68%；在无公共数据情形下仍能实现三倍于FedAvg的准确率。

**⚠️ 局限性**

局限性包括：实验仅覆盖CNN和单次ResNet-18，未验证更大模型与多种任务；漂移模式为离散阶段边界，缺少连续软漂移实验；客户端数量和参与比例固定，未探究更广泛的资源受限和客户端选择场景。

---

## 679. DiffUNet^2: Bidirectional Prediction, Probabilistic Generation and Collaborative Visual Discovery for Scientific Data

**arXiv ID:** 2606.03926 | [PDF](https://arxiv.org/pdf/2606.03926v1)

**作者:** Mengdi Chu `[一作]` (Ohio State University), Han-Wei Shen `[通讯]` (Ohio State University)

**通讯引用:** 6422 | [OpenAlex ID](https://openalex.org/A5065630217)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了基于扩散模型的双向生成框架DiffUNet^2及其交互可视化系统，用于科学时间序列数据的多模态探索、歧义诊断与假设检验。

**💡 创新点**

创新点包括：1）将条件扩散模型用于任意时间步的双向生成，支持前向、后向及跳跃查询；2）引入用户指导的状态编辑与分布感知微调，实现可控的多模态推理；3）在交互系统中实现分支时间线、概率空间可视化，将生成模型转变为主动的科学推理工具。

**🔧 技术方法**

技术细节：条件U‑Net扩散模型，结合KL正则化VAE实现潜在空间建模；FiLM调制实现时间与状态条件注入；双向任意节点训练与跳跃学习；Guided denoising用于状态编辑后细化；系统采用节点式时间线与PCA+Isomap降维的概率空间视图。

**📊 数据集**

使用的五个科学时间序列数据集：Shallow‑Water、Cloverleaf、HEAT‑PLI、RealPDE‑FSI、Wildfire。

**📈 对比分析**

与UNet、FNO、DeepONet三种基线在nRMSE、SSIM、PSNR（确定性评估）以及CRPS、Spread‑Skill Gap/Ratio（概率评估）上进行对比。DiffUNet^2在确定性任务上保持竞争力，在不确定性任务中生成多模态样本、保持细节并实现良好校准，整体性能优于或相当于基线。

**⚠️ 局限性**

局限性：模型依赖训练数据覆盖，易忽略稀有事件；实验仅限二维数据，缺乏3D/大规模验证；未显式区分aleatoric与epistemic不确定性；交互系统缺乏系统化用户研究，尚需提升可扩展性与物理约束融入度。

---

## 680. Adaptive Causal Alignment for High-Confidence Adversarial Training

**arXiv ID:** 2606.03925 | [PDF](https://arxiv.org/pdf/2606.03925v1)

**作者:** Zhiming Luo `[一作]` (Xiamen University), Shaozi Li `[通讯]` (Xiamen University)

**通讯引用:** 15050 | [OpenAlex ID](https://openalex.org/A5081767617)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种高置信度因果对齐训练框架（HICAT），通过可学习背景偏差估计器（LBBE）动态诊断背景信息的支持性与噪声性，结合自适应去偏和正交增强实现鲁棒特征的因果分离；

**💡 创新点**

创新点在于：①识别视觉背景是双向信号（支持性或混淆性）并用可学习模块自适应调控；②引入正交增强（FLOE）在logit空间强制前景与背景正交，从而降低过拟合；③在逆对抗样本基础上实现高置信度对齐，避免传统盲目抑制导致的特征贫化；

**🔧 技术方法**

技术方法包括：可学习背景偏差估计器（LBBE）——通过掩码与代理分类器测量背景贡献并蒸馏成轻量回归器；自适应去偏——根据LBBE输出的权重对逆对抗logit进行门控校正；Debiased Logit Alignment——将校正后的高置信度logit与对抗logit对齐；Foreground Logit Orthogonal Enhancement（FLOE）——在logit空间施加正交约束；使用Grad‑CAM（或其他掩码）和逆对抗攻击；

**📊 数据集**

实验数据集包括 CIFAR‑10、CIFAR‑100 以及 ImageNet‑1K，使用标准 PGD、C‑W、AutoAttack 等评估；

**📈 对比分析**

与 TRADES、MART、UIAT、DHAT 等基线在匹配架构和训练设定下对比，HICAT 在 AutoAttack 上均提升 1–2% 以上的鲁棒准确率，保持或提升清晰准确率，并显著降低稳健泛化差距；在 CNN 与 ViT 上均能实现跨架构提升；

**⚠️ 局限性**

局限性包括：对掩码质量与 LBBE 代理模型依赖较大；需要额外的逆对抗样本生成和 LBBE 预热，增加训练复杂度；在极端攻击强度下仍可能出现性能衰减；以及对超参数（阈值、权重）较为敏感，需细致调优。

---

## 681. An Attention-Based Denoising Model for Diffusion Weighted Imaging

**arXiv ID:** 2606.03903 | [PDF](https://arxiv.org/pdf/2606.03903v1)

**作者:** Prithviraj Verma `[一作]` (Institute of Infrastructure Technology Research and Management), Prasun Chandra Tripathi `[通讯]` (Institute of Infrastructure Technology Research and Management)

**通讯引用:** 298 | [OpenAlex ID](https://openalex.org/A5076130601)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了噪声感知的Swin–Restormer框架，用于在Rician噪声下对扩散加权成像进行去噪，结合层级窗口自注意力与多维门控细化实现残差重建。

**💡 创新点**

创新点在于将Swin窗口注意力与Restormer门控细化相结合、加入显式噪声级条件化并采用残差学习，从而在多噪声级下自适应抑制异方差Rician噪声，提升结构保真度。

**🔧 技术方法**

使用的技术包括Swin Transformer、Restormer、多维转置注意力、门控深度卷积、噪声级编码、残差学习以及Charbonnier+SSIM损失。

**📊 数据集**

实验基于多中心 Traveling Human Phantom（THP）二维DWI数据集，包含25,200张128×128的图像。

**📈 对比分析**

与NLM-Rician、FFDNet、U-Net、SwinIR等方法对比，平均PSNR达33.69 dB、SSIM 0.8539，在1%–15%噪声范围内始终保持领先。

**⚠️ 局限性**

局限性包括未在真实临床数据上验证、未处理3D大尺寸数据、对不同磁共振设备的泛化性未充分评估。

---

## 682. OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs

**arXiv ID:** 2606.03890 | [PDF](https://arxiv.org/pdf/2606.03890v1)

**作者:** Yifei Li `[一作]` (Tsinghua University), Jiwen Lu `[通讯]` (Tsinghua University)

**通讯引用:** 29686 | [OpenAlex ID](https://openalex.org/A5100460385)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 OVO‑S‑Bench，一套以持续的第一人称视频流为输入、四层空间抽象（即时感知、时空追踪、空间模拟与推理、全局坐标映射）的多模态评测基准。

**💡 创新点**

创新点在于：①首次系统化评估连续流中对空间的记忆与推理能力；②引入可标注的查询时间与证据区间，严格模拟在线感知场景；③对比分析了多种模型（专用流式架构、空间微调、通用 backbone 等），揭示 allocentric 映射为主瓶颈。

**🔧 技术方法**

采用人类专家手工构造多选题、跨视频来源的标注、流水线式的测试协议；评测 38 系统，并使用随机、纯文本、人工对照基线；分析链式思考、帧采样策略与专用模型对性能的影响。

**📊 数据集**

数据来源覆盖 9 个公开/可访问数据集：RoomTour3D、Ego4D、Sekai、OmniWorld、YouTube 行走视频、CODa、Honda HDD、ARKitScenes、VSI‑Bench，共 348 源视频、1,680 问题。

**📈 对比分析**

与基线对比，最强系统 Gemini‑3.1‑Pro 获得 59.2%（流式）/ 86.6%（流式人类）/ 92.2%（离线人类），与人类相差 27 分；allocentric L4 成为普遍瓶颈；闭源优势仅 5.6 分；专用流式或空间微调模型往往比对应 base 更差；链式思考在跨视角推理时提升约 +3.9% 但在即时感知时略有下降。

**⚠️ 局限性**

局限性包括：仅评估被动观测的预录制流，缺乏感知‑动作闭环；所有题目为单选，可能掩盖部分空间知识；对专用模型的域迁移未做严格控制；未涵盖交互式或开放式回答形式。

---

## 683. RealClawBench: Live OpenClaw Benchmarks from Real Developer-Agent Sessions

**arXiv ID:** 2606.03889 | [PDF](https://arxiv.org/pdf/2606.03889v1)

**作者:** Zongwei Lv `[一作]` (Peking University), Guangxiang Zhao `[通讯]` (Qiyuan Tech)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了RealClawBench，将真实的OpenClaw使用日志转化为可重现的、可自动评分的评测任务，填补了现有可执行评测与真实开发者使用之间的现实差距。

**💡 创新点**

创新点在于通过会话过滤、环境重建、请求重写和确定性验证器构造等流程，将真实会话中的隐私、局部状态和隐式意图等特征映射为可公开、可重现的任务，并支持持续的版本发布。

**🔧 技术方法**

使用了日志抽样与过滤、工作空间重建、指令重写、程序化验证器、分层评测以及基于分布匹配的持续发布技术。

**📊 数据集**

使用的数据集来自OpenClaw生产环境的76,155条工具使用会话，经过筛选后生成了281个公开评测任务。

**📈 对比分析**

通过在统一的OpenClaw执行环境下评测14款模型，最佳模型Claude Opus 4.7达到65.8%的样本平均成功率，表现出显著的性能提升空间，并揭示了成本与准确度的非线性关系。

**⚠️ 局限性**

局限性包括：仅覆盖OpenClaw用户与工具生态；隐私与安全过滤可能导致分布漂移；验证器偏向可确定性结果，可能忽视代码可维护性等主观质量；评测结果受单一工具接口与预算设置影响；需要持续治理以保证后续版本的可比性与合规性。

---

## 684. Revisiting $O(n \log \log n)$ chaining for anchored edit distance

**arXiv ID:** 2606.03929 | [PDF](https://arxiv.org/pdf/2606.03929v1)

**作者:** Nicola Rizzo `[一作]` (University of Helsinki), Ragnar Groot Koerkamp `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 68 | [OpenAlex ID](https://openalex.org/A5079994235)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出一种新的连通链（colinear chaining）算法，用于计算带锚点的编辑距离（anchored edit distance），通过结合 Eppstein‑Chao‑Miller 的 L∞ 距离间隙代价和 Baker‑Giancarlo 的重叠代价，得到理论上 O(n log log n) 时间、O(n) 空间的解决方案，并实现了一个 O(n log n) 的简化版本；

**💡 创新点**

创新点在于：1）将最早的 L∞ 间隙代价算法与重叠代价模型统一，形成新的四分区 DP 递推；2）使用预排序、前驱/后继结构和稀疏区域影响（owner array）实现 O(n log log n) 复杂度；3）提供了完整的 O(n) 空间实现，弥补了之前仅 O(|T|+|Q|) 空间的缺陷；

**🔧 技术方法**

主要技术包括：1）整数排序（O(n log log n)）和前驱/后继数据结构（Van Emde Boas、Y‑fast trie 或红黑树）；2）线性扫描（left‑to‑right sweep）处理起点/终点；3）范围最小查询（suffix min）用于处理 gap‑gap 大 Q 问题；4）双向链表压缩实现 owner array，避免 O(|Q|) 空间；

**📊 数据集**

使用 PacBio HiFi 读取 100 000 条（约 100 000 条）与 T2T‑CHM13 人类基因组参考之间的 MEM（≥50 bp）作为锚点；

**📈 对比分析**

与现有链算法（如 “…”、“…” 等）对比，实验显示：1）“本实现”在 3 000 000 约束锚点下平均 10 倍快；2）在 MEM 链接 100 000-10 000 000 之间，“本实现”保持线性增长，且整体运行时间仅比种子阶段略慢，其他方法出现二次增长，导致 3 倍以上延迟；

**⚠️ 局限性**

局限性：1）实现使用 std::set，前驱/后继操作为 O(log n)，可通过 B‑tree 或 Y‑fast trie 加速；2）仅支持精确锚点和 L∞ + 重叠代价，未覆盖逆转、非线性分数或 affine gap；3）未完全优化链路，链路后继更新延迟可能影响常数；4）缺乏对极端重复区域的稳健性测试。

---

## 685. Using Reward Uncertainty to Induce Diverse Behaviour in Reinforcement Learning

**arXiv ID:** 2606.03962 | [PDF](https://arxiv.org/pdf/2606.03962v1)

**作者:** Anthony GX-Chen `[一作]` (New York University), Mark Rowland `[通讯]` (Google DeepMind)

**通讯引用:** 15593 | [OpenAlex ID](https://openalex.org/A5047099585)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种新的强化学习目标（+Max/Softmax），通过在奖励函数分布上对动作集合求最大/Softmax，学习多样化的随机策略；

**💡 创新点**

核心创新在于将多样性视为对奖励不确定性的合理响应，使用奖励函数分布而非单一奖励，并提供无偏梯度估计，保证最优策略为对所有奖励函数均匀采样的随机策略；

**🔧 技术方法**

利用策略梯度、集合函数（max、Softmax）、控制变量、理论分析（凸性、梯度稀疏性）以及经验验证（toy、LLM推理、MATH/AIME数据集）来实现并评估该方法；

**📊 数据集**

在小规模离散动作实验、LLM文本生成任务（多重长度偏好、竞争奖励）以及数学推理任务（MATH、AIME）上使用人工设计和真实评判器混合奖励；

**📈 对比分析**

与传统PG、熵正则化、多样性奖励、最佳- N 训练等方法对比，+Max/Softmax 在保持准确率不变的前提下显著提升多样性（如覆盖多种长度、奖励模式），并在有噪声奖励时更鲁棒，out‑of‑distribution 的 pass@k 也更高；

**⚠️ 局限性**

局限性包括：需要预先指定奖励函数分布并采样，若分布不准确或样本不足可能影响性能；对动作集合大小 n 的选择敏感；理论证明主要基于二值奖励和简化假设，真实任务的收敛性和计算开销尚待进一步研究。

---

## 686. QUBRIC: Co-Designing Queries and Rubrics for RL Beyond Verifiable Rewards

**arXiv ID:** 2606.03968 | [PDF](https://arxiv.org/pdf/2606.03968v1)

**作者:** Rongzhi Zhang `[一作]` (Amazon), Chao Zhang `[通讯]` (Amazon)

**通讯引用:** 10357 | [OpenAlex ID](https://openalex.org/A5100460272)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出一种同时重新设计查询与 rubrics 的框架，使得在开放式任务中能够获得可评估的奖励。

**💡 创新点**

创新点在于通过教师关键点驱动的查询重写与对比式 rubrics 生成，解决了“查询结构导致 rubrics 无效”的瓶颈。

**🔧 技术方法**

采用关键点提取、情景化查询重写、对比式 rubrics 生成、学习可过滤、GRPO 策略优化等技术。

**📊 数据集**

主要使用 ShareGPT、ArenaHard、MoReBench、PLawBench、MuSR 等多种 instruction-following 与推理基准，以及内部购物对话数据。

**📈 对比分析**

在 ArenaHard 上提升了 5.5 分，跨域转移到三项未见基准平均提升 6.3 分，明显优于 AutoIF‑RLVR、RM‑RLHF 等对照。

**⚠️ 局限性**

局限性包括对 LLM 判定的依赖、对关键点抽取的自动化与一致性问题，以及对开放式查询的通用性仍待验证。

---

## 687. Efficient ASR Training with Conversations that Never Happened

**arXiv ID:** 2606.03957 | [PDF](https://arxiv.org/pdf/2606.03957v1)

**作者:** Máté Gedeon `[一作]` (Budapest University of Technology and Economics), Péter Mihajlik `[通讯]` (Budapest University of Technology and Economics)

**通讯引用:** 313 | [OpenAlex ID](https://openalex.org/A5020514527)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种统一的基于大型语言模型（LLM）生成情景对话、元数据驱动的TTS合成与说话人感知的对话仿真三阶段数据增强流水线，用于提升低资源语言的对话式语音识别；

**💡 创新点**

创新点包括：①将LLM生成的情景、参与者元数据与TTS声码映射结合，保证生成语音与说话人属性匹配；②采用说话人感知的时间仿真，重现自然对话中的停顿、重叠和说话顺序；③系统性评估多款主流LLM（GPT、Claude、Gemini、Grok、Qwen）的单体和混合生成对ASR的影响，揭示生成器选择与数据组合对性能的关键作用；

**🔧 技术方法**

使用技术包括：LLM提示（两步场景+对话生成）、xTTS‑v2声码合成与说话人克隆、基于统计的说话人感知对话模拟、FastConformer‑Large端到端ASR模型以及NVIDIA NeMo训练框架；

**📊 数据集**

使用数据集：匈牙利BEA‑Dialogue（真实对话），BEA‑Large（真实单句语料用于声码库），以及由上述流水线生成的合成对话；

**📈 对比分析**

对比方法：在相同FastConformer‑Large训练配置下，将合成数据与真实数据、仅模拟语料、以及2700小时匈牙利语零样本模型等基线进行对照。结果显示：单体LLM合成可将cpWER从20.44%降至17.75%；最佳两模型混合（GPT+Haiku）进一步降至16.96%；全量四模型混合+真实模拟可将cpWER降至15.40%，甚至优于2700h零样本基线；

**⚠️ 局限性**

限制：LLM生成质量差异导致合成数据量不一致，无法完全公平比较；需要语言对应的高质量TTS与声码库；流水线对对话时间统计的仿真假设可能不完全匹配真实对话；仅在匈牙利语上验证，跨语言推广仍需实验验证。

---

## 688. q0: Primitives for Hyper-Epoch Pretraining

**arXiv ID:** 2606.03938 | [PDF](https://arxiv.org/pdf/2606.03938v1)

**作者:** Bishwas Mandal `[一作]`, Samip Dahal `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了“超时段预训练”（Hyper‑epoch Pretraining）方法，在固定数据上利用多周期训练生成多模型快照，随后通过链式蒸馏提升每个快照性能，并用学习得到的先验对快照进行加权组合，以提高验证损失和下游任务表现。

**💡 创新点**

创新点：1）通过循环学习率与权重衰减反相关的周期性调度，高效采集多样化快照；2）链式蒸馏使每个快照以前一个快照为教师，形成能力层层递增的模型族；3）在保留集上学习softmax先验，用于在任意推理预算下挑选并加权最佳快照，从而超越统一平均的传统集成。

**🔧 技术方法**

技术：快照集成、循环LR/WD调度、链式蒸馏、softmax学习先验、EMA权重平均、Stochastic Weight Averaging、模型香草等。

**📊 数据集**

数据集：训练使用 FineWeb 100M 令牌，验证使用 FineWeb 10M 令牌；下游评测在 ARC‑Easy、PIQA、SciQ 三个基准上进行。

**📈 对比分析**

与基线比较：基线为 8 个独立训练 32 epoch（总 256 epoch）并 EMA + 均匀平均。q0 在同等总 epoch（256）下，仅需约 56 epoch 即可达到相同验证损失，数据效率提升约 12×；在 256 epoch 预算下下游准确率平均提升约 14–20%，并在更大预算（960 epoch）下进一步提升；q0 在所有预算范围内均表现优于基线。

**⚠️ 局限性**

局限性：推理成本高（需要多次前向传播，K 次）；链式蒸馏需额外教师前向计算，虽可缓存但仍有开销；学习先验对极小预算下的加权选择不一定最优；方法在不同模型规模/数据分布下的超参数（N、C、τ）需要经验性调优，未提供自动化搜索。

---

## 689. Benchmarking Visual State Tracking in Multimodal Video Understanding

**arXiv ID:** 2606.03920 | [PDF](https://arxiv.org/pdf/2606.03920v1)

**作者:** Sihyun Yu `[一作]` (New York University), Saining Xie `[通讯]` (New York University)

**通讯引用:** 48600 | [OpenAlex ID](https://openalex.org/A5102416863)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种名为 Visual State Tracking（VSTAT）的视频基准，用来评估多模态大型语言模型（MLLM）在持续视频流中跟踪潜在状态的能力。

**💡 创新点**

创新点在于：①设计了能迫使模型全程跟踪并更新状态的问答任务，避免视觉捷径；②构建了覆盖合成、实录、真实世界多种视频的多样化数据集；③从感知复杂度与状态复杂度两轴进行细粒度错误诊断，揭示视觉感知是瓶颈；④评估并证明即使是最新的代理框架也无法显著提升性能。

**🔧 技术方法**

使用的技术包括：多模态预训练模型（Gemini、Qwen、LLaVA、InternVL 等）的问答接口；对视频进行时间拉伸、帧抽样实验；对比文本转录与视频输入的性能差异；错误分析利用模型生成的“思考轨迹”。

**📊 数据集**

数据集：VSTAT 共包含 934 条视频（450 条合成 + 304 条 YouTube + 80 条自录），每条视频平均产生多条问答，涵盖计数、位置、属性等多种状态，状态结构包含原子、序列、集合、字典四种。

**📈 对比分析**

与现有基准（VideoMME‑v2、VideoReasonBench、CP‑Bench、VET‑Bench）相比，VSTAT 在状态追踪方面更具挑战性；实验显示最强的开源模型平均准确率仅 30–35%，远低于人类 90%；对比回答先验（频率）基线，MLLM 仅略优于随机。代理框架的表现亦与基线持平或略低。

**⚠️ 局限性**

局限性包括：①文本转录实验仅在极简合成任务中可行，难以推广到复杂真实视频；②基准仍未覆盖所有可能的感知难点，如光照变化、遮挡程度极高的情形；③评估主要基于问答准确率，未深入探究模型内部表示和动态记忆机制；④目前仅针对英文学术问答，跨语言适用性未知。

---

## 690. Hedge-Bench: Benchmarking Agents on Hard, Realistic Tasks Pertaining to Financial Reasoning

**arXiv ID:** 2606.03918 | [PDF](https://arxiv.org/pdf/2606.03918v1)

**作者:** Eric Cho `[一作]` (Trata), Andy Lyu `[通讯]` (Osmosis)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Hedge‑Bench，针对金融领域高级分析师开放式推理任务的基准测试框架，评估 LLM 在多步骤、信息检索与论证生成中的表现。

**💡 创新点**

创新点在于：①基于真实专业分析师的音频对话生成高质量推理轨迹，确保任务贴合行业实践；②将任务拆分为主题与行动步骤，并通过 LLM 判定器分别核实事实引用、主题覆盖与合成；③引入稀疏 Pass@1 与稠密 Dense Score 两种评价指标，突出“完美”推理的罕见性。

**🔧 技术方法**

使用技术包括：Docker 环境模拟、Harbor 任务格式、LLM‑as‑Judge（Gemini‑3.1‑Pro）进行事实与逻辑评估、主题阈值 τ 机制、宏观均值与置信区间计算。

**📊 数据集**

数据集为 Hedge‑Bench 1.0，包含 102 个高难度任务（共 5,112 子任务），每个任务由两名专业分析师对话录音转录、文件材料与手工制定的评判规则组成。

**📈 对比分析**

与现有金融 QA 与 agentbench（如 FinQA、FinanceBench、FAB 等）对比，Hedge‑Bench 更侧重过程与判断。实验显示，最优模型 Claude‑Sonnet‑4.6 的 Pass@1 仅约 15%，宏观 Dense Mean 最高 1.92/4，说明当前 LLM 在开放式金融推理任务上仍远未成熟。

**⚠️ 局限性**

局限性包括：①基准与评分基于单一分析师对话，可能缺乏行业普遍共识；②评分依赖单模型生成的 rubric，可能与实际对话偏差；③LLM 判定器偶发解析错误导致整体评价被清零；④文档集有限，无法覆盖所有可能信息源。

---

## 691. SparseStreet: Sparse Gaussian Splatting for Real-Time Street Scene Simulation

**arXiv ID:** 2606.03909 | [PDF](https://arxiv.org/pdf/2606.03909v1)

**作者:** Qingpo Wuwu `[一作]` (Peking University), Shanghang Zhang `[通讯]` (Peking University)

**通讯引用:** 11554 | [OpenAlex ID](https://openalex.org/A5013030532)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SparseStreet 框架，对街道场景的 3D Gaussian Splatting 进行稀疏压缩，既保留动态物体细节，又显著减少高斯原语数量。

**💡 创新点**

创新点在于（1）node‑aware pruning：根据场景图节点类型自适应学习掩码，动态节点保留细节，静态背景激进裁剪；（2）background pruning：针对背景重要性做全局权重评估并裁剪；（3）时间依赖掩码 MLP，避免动态物体因视角稀疏被误删。

**🔧 技术方法**

使用 3D Gaussian Splatting + learnable masking + straight‑through estimator + node‑aware regularization + time‑dependent MLP + 重要性评估 + 场景图结构。

**📊 数据集**

在 Waymo Open Dataset 和 nuScenes 两大自动驾驶数据集上进行评估。

**📈 对比分析**

与 OmniRe、StreetGS 等基线对比：压缩率最高可达 80%（Gaussian 数量减少约 3×），FPS 提升 2–4×，PSNR/SSIM/LPIPS 仅微降，实时渲染可达 80 FPS，FID 在新轨迹合成中与基线相当。

**⚠️ 局限性**

局限性：目前仅在有监督、场景图标注的设置下有效；对极少视角出现的动态物体仍可能被误裁；需要手动调节正则系数与阈值，未验证对无监督自监督框架的兼容性。

---

## 692. Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments

**arXiv ID:** 2606.03892 | [PDF](https://arxiv.org/pdf/2606.03892v1)

**作者:** Ibrahim Abdelaziz `[一作]` (IBM Research), Pavan Kapanipathi `[通讯]` (IBM Research)

**通讯引用:** 1202 | [OpenAlex ID](https://openalex.org/A5003720552)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个包含20个状态化Model Context Protocol（MCP）服务器的库，并通过自动化的、基于状态机的训练数据合成管线和全程序化的多组件奖励函数，直接在实时执行环境中训练LLM实现多步工具调用。

**💡 创新点**

核心创新点包括：1）面向真实状态化服务的MCP服务器库，克服缓存或模拟API的局限；2）自动发现工具依赖图并在真实服务器状态下生成可执行的多轮对话数据；3）完全程序化的五维奖励（有效性、覆盖率、效率、工具名匹配、参数值匹配）并引入自适应调用预算，有效抑制RL奖励的冗余调用倾向。

**🔧 技术方法**

技术实现主要采用：MCP服务器与VERL框架集成的GRPO强化学习；状态机驱动的对话生成与回放验证；基于依赖图的工具链生成与实体采样；多组件奖励函数的程序化实现，避免外部判别模型；对比实验使用BFCL Multi-Turn、τ²-bench、T-Eval等基准。

**📊 数据集**

使用的数据集为约13,517条多轮工具调用轨迹，来源于20个MCP环境下自动生成的对话、补全（missing‑function）和拒绝（abstention）示例，包含10,895条多轮对话、1,500条澄清轨迹和1,122条外部拒绝示例。

**📈 对比分析**

在四个基准上进行比较：BFCL Multi‑Turn、τ²‑bench、T‑Eval。与基线模型相比，所提方法在所有四个模型上均取得显著提升（BFCL整体最高+10.2点，τ²‑bench+6.8点，T‑Eval+6.5点），并且在不使用大型标注数据或LLM评判器的情况下实现了持续的多步工具协作性能。

**⚠️ 局限性**

局限性包括：1）奖励权重和自适应预算参数需经验调优；2）实验仅在8个H100 GPU和四个模型规模上验证，尚未探讨更大规模或跨架构的通用性；3）数据合成仍依赖教师LLM，虽然实验表明对教师不敏感，但极端场景下可能仍受限；4）缺乏对长期持续性（超多轮对话）和安全性（误调用、误信息）的深入评估。

---

## 693. CoralBay: A Self-Supervised CT Foundation Model

**arXiv ID:** 2606.03888 | [PDF](https://arxiv.org/pdf/2606.03888v1)

**作者:** Ioannis Gatopoulos `[一作]` (kaiko.ai), Fei Tang `[通讯]` (kaiko.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出了基于DINO的自监督3D CT基础模型，利用层次化Swin Transformer和多尺度特征学习，结合CT专用数据增强（HU窗口、噪声、模糊等），进行大规模预训练。

**💡 创新点**

创新点在于将DINO迁移到3D空间，使用3D Swin Transformer骨干，采用多尺度特征拼接、自监督教师-学生对齐，并在预训练中引入CT特有的HU窗口随机化、滑动窗口推理等增强策略，从而显著提升了对CT数据的全局与局部语义表示能力。

**🔧 技术方法**

使用的技术包括：自监督教师-学生对齐（DINO）、3D Swin Transformer骨干、3D多尺度特征拼接、CT专用随机增强（HU窗口、对比、噪声、模糊）、滑动窗口推理、线性探针评估以及Swin UNETR解码器用于分割。

**📊 数据集**

数据集来源于多公共CT数据集合，构成平衡的“COORID”集合，包含AbdomenAtlas Mini、HNSCC、LUNA16、STOIC、LIDC-IDRI、Stony Brook、TCGA-HNSC等约1万份CT体积，覆盖胸部、腹部、头颈等主要解剖区域。

**📈 对比分析**

通过与SwinUNETR、nnUNetV2、Universal Model、SuPreM、VoCo等模型在11个分类与分割基准（OrganMNIST、NoduleMNIST、CC-CCII、LUNA25、BTCV、WORD、FLARE22、CHAOS、LiTS17、KiTS23、MSD Pancreas）上的对比，表现出色：分类任务几乎与大规模预训练模型持平，分割任务尤其在小病灶低标注情形下Dice得分提升1–3个百分点，显著优于传统基准。

**⚠️ 局限性**

局限性包括：仅针对CT扫描，无法直接推广到MRI等；预训练仍需大量未标注数据；模型规模较大，训练与推理资源消耗高；对极端扫描条件（如极低剂量或高噪声）鲁棒性仍有限。

---

## 694. Knowledge Editing in Masked Diffusion Language Models

**arXiv ID:** 2606.03924 | [PDF](https://arxiv.org/pdf/2606.03924v1)

**作者:** Haewon Park `[一作]` (Seoul National University), Yohan Jo `[通讯]` (Seoul National University)

**通讯引用:** 4679 | [OpenAlex ID](https://openalex.org/A5016844435)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将 locate-then-edit 知识编辑方法从自回归模型迁移到掩码扩散模型，并在同等规模下比较两类模型在编辑单词和多词目标时的表现，发现两者在编辑位置相同但多词目标编辑会出现性能下降，并提出针对中间去噪状态的改进方案。

**💡 创新点**

1）证明 locate-then-edit 的位置假设（早中层 MLP 的最后主语 token）在掩码扩散模型中同样成立；2）揭示多词目标编辑在掩码扩散模型中失败的根源是生成过程中的中间状态未被优化；3）提出简单的改进——在多状态下同时优化残差，使得编辑在各个去噪步骤均能发挥作用，从而显著恢复多词编辑性能。

**🔧 技术方法**

使用因果追踪（causal tracing）定位编辑位置，采用 MEMIT 作为 locate-then-edit 基础算法，并对其进行掩码扩散模型的适配；在改进中通过在不同的中间去噪状态下重新优化残差；对生成过程进行多步骤评估。

**📊 数据集**

CounterFact（主要单词目标）和 Kamel（多词目标）两大知识编辑基准数据集，用于评估编辑的效能、泛化与特异性。

**📈 对比分析**

与同规模的自回归模型（LLaMA、Qwen）进行对照，使用效能、泛化、特异性等指标衡量。结果显示：在单词目标上，掩码扩散模型与自回归模型相近；在多词目标上，掩码扩散模型的性能显著下降（从 0.92 降至 0.27 以上），而改进后多词编辑效果大幅提升，效能在 N=4 时由 0.27 提升至 0.73。

**⚠️ 局限性**

仅使用 MEMIT 作为代表，未验证其他 locate-then-edit 方法；实验仅覆盖两组 ARM–MDM 在同一规模的模型，未探究更大规模或其他 MDM 架构；改进虽然恢复大部分性能，但在最长目标（N=4 以上）仍存在一定下降，未来需进一步研究中间状态的顺序和位置对编辑效果的影响。

---

## 695. Ranked MSO-enumeration over compressed words

**arXiv ID:** 2606.03947 | [PDF](https://arxiv.org/pdf/2606.03947v1)

**作者:** Markus Lohrey `[一作]` (University of Siegen), Markus Lohrey `[通讯]` (University of Siegen)

**通讯引用:** 2600 | [OpenAlex ID](https://openalex.org/A5005500210)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `fede83ac-7505-405f-ab37-e7284695c47f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文研究了在字符串通过直线程序（SLP）压缩的情况下，针对固定的MSO查询进行按预定可定义顺序的枚举，证明可以在对输入的线性预处理后实现常数延迟输出。

**💡 创新点**

创新点在于首次把有序（ranked）枚举扩展到压缩数据；提出了Simon SLP（与因子化树相结合的压缩形式），并利用它实现了对多项式正则函数的符号枚举；同时给出了完整的算法框架与证明。

**🔧 技术方法**

主要技术包括：因子化树（Simon定理）、构造Simon SLP、基于布尔矩阵的幺半群同态、在SLP上实现常数时间双向遍历、直接积幺半群、以及对路径的压缩表示与树形数据结构。

**📊 数据集**

本研究为理论性工作，未使用实际数据集；所有实验与分析均基于理论复杂度与算法证明。

**📈 对比分析**

相较于以往仅支持无序枚举或在无压缩输入上的常数/线性延迟，本文实现了在压缩输入下的有序枚举，并保持线性预处理与常数延迟，性能上属于最优的理论上限。

**⚠️ 局限性**

局限性包括：只能处理仅含一阶自由变量的MSO查询（不支持自由集合变量）；扩展到森林或更一般结构尚未实现；对权重排序（cost transducer）等模型的关系尚不清晰。

---

## 696. PointAction: 3D Points as Universal Action Representations for Robot Control

**arXiv ID:** 2606.03943 | [PDF](https://arxiv.org/pdf/2606.03943v1)

**作者:** Mutian Tong `[一作]` (University of Pennsylvania), Jiatao Gu `[通讯]` (University of Pennsylvania)

**通讯引用:** 10728 | [OpenAlex ID](https://openalex.org/A5112542984)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种将视频预测与机器人控制通过动态3D点图（RGB-XYZ）相连接的框架，先用通用视频模型生成RGB和点云序列，再用轻量级解码器映射到机器人动作。

**💡 创新点**

将3D点图作为可扩展的中间表示，解决RGB-仅表示的几何歧义，并通过可扩展的预训练-轻量解码结构实现跨机器人、跨任务的迁移。

**🔧 技术方法**

基于视频扩散模型的4D生成、Diffusion Forcing、Flow Matching、LoRA参数高效微调、点云编码(PointNet+DiT)和AdaLN条件扩散解码。

**📊 数据集**

BridgeData V2、DROID（WidowX 250和Franka Panda）视频，RoboCasa365仿真任务，另外xArm7与YAM真实机器人微调集。

**📈 对比分析**

与VLA和VAM基线在ID、OOD-Env和OOD-Task下的仿真成功率对比，取得最高47.7%（ID）和44.1%（OOD-Env），相较于Cosmos-Policy +2.5%/1.2%；在跨机器人真实场景上也超过同类VLA，xArm7 43.0% vs 22.7%。

**⚠️ 局限性**

视频模型推理速度慢，执行为单次前向传递的开放式控制，易受物理扰动累积误差影响；需要进一步加速与闭环控制。

---

## 697. Multi-Robot Bearing-only Pose Estimation via Angle Rigidity

**arXiv ID:** 2606.03931 | [PDF](https://arxiv.org/pdf/2606.03931v1)

**作者:** J. Francisco Presenza `[一作]` (Institute of Engineering Technology and Sciences Hilario Fernández Long), Juan I. Giribet `[通讯]` (Universidad de San Andrés)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4`

**🎯 论文内容**

提出一种基于角度刚性的新型分布式姿态估计器，只利用机器人自身坐标系下的视线（角度）信息即可估计位置与姿态；

**💡 创新点**

创新点在于：①不再需要对称或多角度测量的限制；②通过角度刚性获得位置，利用角度导数与控制输入恢复姿态；③实现了不需要相对角度测量或额外传感器的全姿态估计；

**🔧 技术方法**

采用角度刚性理论、梯度下降姿态修正、持续激励假设以及链式梯度更新等技术；

**📊 数据集**

仅使用仿真数据（5 机器人在三维空间中的运动轨迹），未使用公开数据集；

**📈 对比分析**

与传统仅基于角度测量或相对角度测量的方法对比，实验显示误差在数秒内快速衰减至零，证明了该方法的收敛速度与稳健性；

**⚠️ 局限性**

局限性包括：需要自由机器人在一定时间窗口内持续激励；对初始误差的吸引域未明确定义；噪声鲁棒性尚待进一步验证。

---

## 698. PatchScene: Patch-based Voxel Diffusion for Large-Scale Scene Completion

**arXiv ID:** 2606.03915 | [PDF](https://arxiv.org/pdf/2606.03915v1)

**作者:** Qingdong Xu `[一作]` (MEGVII Technology), Jiyao Zhang `[通讯]` (Peking University)

**通讯引用:** 25085 | [OpenAlex ID](https://openalex.org/A5100409994)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出PatchScene，一种基于扩散模型的 LiDAR 场景补全框架，通过将全局体素空间拆分为重叠局部补丁进行局部扩散，再融合生成完整点云，支持无限空间生成与时间一致性；

**💡 创新点**

创新点包括：①基于补丁的显式体素扩散降低计算量并提升细节；②随机耦合的空间融合与自适应时间融合确保跨补丁与跨帧的一致性；③环形向外扩散（Annular‑Flow）利用 LiDAR 密度梯度实现无限范围补全；

**🔧 技术方法**

使用技术包括：扩散模型（denoising diffusion probabilistic models）、体素化与重叠补丁划分、位置编码、随机耦合空间融合、ICP 对齐的时间融合、环形向外扩散策略；

**📊 数据集**

实验数据集为 SemanticKITTI，训练 20 m 范围，测试 50 m 范围，使用 0.15625 m 体素分辨率；

**📈 对比分析**

与 LiDiff、LiDPM、ScoreLiDAR 等现有方法比较，PatchScene 在 Chamfer Distance、JSD‑BEV、JSD‑3D、体素 IoU 等指标均显著优于对比方法，且在时间一致性（RMSE）上提升显著；

**⚠️ 局限性**

局限性在于：仍需较多计算资源（即使补丁化后仍有显著内存/时间开销）；对动态场景的实时性尚未完全验证；环形扩散假设 LiDAR 点云密度随距离递减，可能在非标准扫描模式下表现下降。

---

## 699. Sparse Activation for Sustainable Cell-Free Massive MIMO Networks: Less is More

**arXiv ID:** 2606.03912 | [PDF](https://arxiv.org/pdf/2606.03912v1)

**作者:** Zhe Wang `[一作]`, Emil Björnson `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了基于稀疏激活的细粒度天线/阵列开启方案，并开发了天线级OBE加权框架，以矩阵权重替代传统的标量权重。

**💡 创新点**

创新点在于将LSFD从标量权重推广到矩阵权重的OBE加权，并通过结构稀疏正则化设计四种天线/阵列级别的稀疏激活方案（UE专用与网络全局），利用树形稀疏结构实现高效求解。

**🔧 技术方法**

技术上使用多输入多输出统计优化、均方误差最小化、稀疏优化（l1/l2组正则化与树形稀疏正则化）、前向后向推断（proximal、FISTA）以及复杂度与收敛分析。

**📊 数据集**

实验采用模拟的CF mMIMO网络（200x200 m²，M=10，K=10，N=9，半波间距，相关Rician信道），通过Monte‑Carlo 30组随机UE位置与800通道样本进行评估。

**📈 对比分析**

与传统LSFD加权以及全激活基线对比，结果显示OBE加权在SE上超过LSFD，稀疏激活在保持SE一定误差的同时可提升EE多达20‑30%，并展示了不同稀疏比例下的SE‑EE折中。

**⚠️ 局限性**

局限性：依赖长时统计，激活模式在慢时间尺度更新；对天线/阵列硬件的精准分级模型假设；实验仅为仿真，未验证实际硬件实现；对用户密度变化的鲁棒性需进一步研究。

---

## 700. Electromagnetic Navigation for Femoral Osteotomy Using High-Accuracy X-ray-to-CT Registration

**arXiv ID:** 2606.03893 | [PDF](https://arxiv.org/pdf/2606.03893v1)

**作者:** Roman Flepp `[一作]` (University Children's Hospital Zürich), Thomas Dreher `[通讯]` (University Children's Hospital Zürich)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发并验证了一种基于电磁跟踪（EMT）与自动X射线-CT配准相结合的股骨成形导航系统，在合成股骨模型上进行可行性研究。

**💡 创新点**

该系统仅需两张定位射线图像即可完成一次性CT-EMT配准，并实现实时无荧光的刀具与骨片导航，且与PSI具有可比的准确性，且无需额外切口。

**🔧 技术方法**

使用自动化X射线-CT配准（U-Net提取金属球中心+子结构轮廓优化）、EMT跟踪（Northern Digital Aurora）、自定义C‑arm校准装置、3D打印夹具以及实时GUI显示。

**📊 数据集**

实验使用18个合成股骨（SYNBONE）模型，配合两名外科医生、三种难度级别的骨切割方案进行评估。

**📈 对比分析**

与自由手技术和患者特异性导向（PSI）比较；EMT平均欧氏角误差为3.05°（±0.75°），显著优于自由手6.32°，与PSI在±2°/±2 mm范围内等效，且自由手出现4/6次>5°失配，而EMT无失配。

**⚠️ 局限性**

仅在合成骨模型和受控电磁环境下验证，未评估软组织、真实手术环境干扰、操作时间及临床真实效果。

---

## 701. Attribution via Distributional Paths for Information Revelation

**arXiv ID:** 2606.03885 | [PDF](https://arxiv.org/pdf/2606.03885v1)

**作者:** Kieran A. Murphy `[一作]` (New Jersey Institute of Technology), Shameen Shrestha `[通讯]` (New Jersey Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一种新的路径归因方法 Reveal-IG，利用从低信息到高信息的分布式探测器路径计算完整的特征重要性。

**💡 创新点**

核心创新是将路径积分从输入空间提升到结构化探测分布空间，既保留完整性，又消除了 IG 的阴影效应和 SHAP 的轴向揭示问题，并可自然适配多尺度图像和不确定性表格数据。

**🔧 技术方法**

结合积分梯度、分布期望、重参数化采样、Monte Carlo 估计、因子化高斯或经验分布探测器以及温度退火路径实现。

**📊 数据集**

在 ImageNet（ResNet‑50 与 ViT‑B/16）图像分类任务以及 Bike Sharing Demand、California Housing、Wine Quality 三个回归表格数据上进行实验。

**📈 对比分析**

与梯度、IG、Blur‑IG、IDG、Guided‑IG、Expected Gradients、SmoothGrad、LIME、KernelSHAP 等基线比较；在图像任务中在插入/删除、RMA、Sensitivity‑n 等指标上多次位列前列，尤其在带符号评估中表现突出；在表格任务中在方向性插入、充分性、灵敏度等指标中名列前茅，展现更稳定且符合目标支持的归因。

**⚠️ 局限性**

主要局限包括需要预设探测器族与路径、忽略特征/像素间相关性、计算成本相对较高（IG 约 7×，表格约 1000×），以及完整性仅适用于期望响应而非确切预测。

---

## 702. Agentic Chain-of-Thought Steering for Efficient and Controllable LLM Reasoning

**arXiv ID:** 2606.03965 | [PDF](https://arxiv.org/pdf/2606.03965v1)

**作者:** Yu Xia `[一作]` (University of California San Diego), Julian McAuley `[通讯]` (University of California San Diego)

**通讯引用:** 25681 | [OpenAlex ID](https://openalex.org/A5021827617)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了Agentic Chain-of-Thought Steering（ACTS）框架，通过控制器在推理过程中选择策略与引导短语，在给定的思考Token预算内引导冻结的推理器完成答案。

**💡 创新点**

创新点在于将推理引导视为在预算约束下的马尔可夫决策过程，使用策略级别的控制（策略+短语）并通过多预算增强的合成轨迹和预算感知的奖励塑造进行强化学习，兼顾精度与效率。

**🔧 技术方法**

主要技术包括控制器-推理器的MDP建模、合成多预算训练轨迹、基于GRPO的强化学习与奖励塑造、异步两服务器推理流水线。

**📊 数据集**

使用了OpenR1-Math数据集合成的推理轨迹，并在MATH‑500、AIME24、AMC（2022/2023）、OlympiadBench以及科学问答GPQA Diamond等四类基准上进行评估。

**📈 对比分析**

与Vanilla、NoThink、CoD、DEER、BudgetGuidance等基线比较，ACTS在保持或提升准确率的同时，平均可节省30%–60%Token，能够实现可控的准确率‑效率权衡，并能迁移到不同规模与任务的推理器。

**⚠️ 局限性**

局限性包括仅在最多8B规模模型上验证，未探测更大模型；需要外部提供思考Token预算，缺乏自适应预算预测功能。

---

## 703. VLESA: Vision-Language Embodied Safety Agent for Human Activity Monitoring

**arXiv ID:** 2606.03954 | [PDF](https://arxiv.org/pdf/2606.03954v1)

**作者:** Hanjiang Hu `[一作]` (Carnegie Mellon University), Changliu Liu `[通讯]` (Carnegie Mellon University)

**通讯引用:** 2550 | [OpenAlex ID](https://openalex.org/A5040156274)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发了 Vision‑Language Embodied Safety Agent (VLESA)，实现从第一人称视频实时监测人类行为、推断意图并提前触发安全干预。

**💡 创新点**

创新点包括：① 引入意图依赖的安全 Q‑过滤器，利用 GRPO 训练的 goal‑conditioned Q‑filter 实现对不同意图下相同动作的安全评估；② 构建 EgoSafety 数据集，系统化生成安全/不安全标注；③ 通过受限解码与意图‑动作预测相结合，实现实时的安全预警与可行行动建议。

**🔧 技术方法**

技术手段包括：视觉‑语言模型 (VLM)、组相对策略优化 (GRPO)、场景图 (EASG) 转自然语言、受限解码、机器人宪章规则、Llama‑4‑Scout 之类的 VLM 推理。

**📊 数据集**

使用数据集：EgoSafety（结合 Ego4D 与 EASG 的安全标注生成）和 ASIMOV‑2.0‑Video（真实伤害场景的视频与干预时间戳）。

**📈 对比分析**

与前沿 VLM（如 GPT‑5、Qwen3 等）及基于提示的安全过滤器对比；在 ASIMOV‑2.0‑Video 上，VLESA 在不同时间窗口的干预准确率显著提升（如 Δt≤1s 81% 对比 47%），安全过滤器整体精度提升约 24 分，未安全召回提升超过 50%，证明了模型在及时干预与安全评估上的优势。

**⚠️ 局限性**

局限性：① 评估主要基于合成或人工生成的视频，缺乏真实传感器噪声与长尾危险的验证；② 安全判断依赖意图估计，意图错误会导致误判；③ 动作词表与候选数有限，可能遗漏未预测或外部词汇的危险；④ 实时延迟与覆盖范围受限，需进一步优化。

---

## 704. Demo2Tutorial: From Human Experience to Multimodal Software Tutorials

**arXiv ID:** 2606.03951 | [PDF](https://arxiv.org/pdf/2606.03951v1)

**作者:** Zechen Bai `[一作]` (National University of Singapore), Mike Zheng Shou `[通讯]` (National University of Singapore)

**通讯引用:** 4002 | [OpenAlex ID](https://openalex.org/A5068937750)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Demo2Tutorial框架，能将原始人类电脑使用演示自动转换为结构化多模态教程。

**💡 创新点**

创新点在于结合动作解析、层次化步骤规划（演员-评论家迭代）、以及自适应视觉合成，生成质量超越人工教程的教学材料。

**🔧 技术方法**

采用同步屏幕与操作记录、基于VLM的语义解析（GPT‑4o）、层次化任务图构建、演员-评论家迭代改进、关键帧打分与视觉注释（SAM2、OCR）等技术。

**📊 数据集**

使用新构建的TutorialBench（110个软件教程）以及OSWorld任务集进行评估，并进行20人参与的人类学习对比实验。

**📈 对比分析**

与人工教程、端到端文本/视觉基线及普通多代理模型对比，Demo2Tutorial在整体得分上从86.2高于79.1，提升GUI代理成功率（GPT‑5 Chrome 52.9%→70.6%），并使人类完成任务速度提高10.5%，用户偏好率达80%。

**⚠️ 局限性**

局限在于主要针对桌面环境，需要高性能VLM与计算资源，且对移动/网页等非结构化界面推广存在挑战。

---

## 705. Quadratic integrate-and-fire neurons exhibit less fragmented loss landscapes and outperform leaky integrate-and-fire neurons in spike-based gradient descent

**arXiv ID:** 2606.03935 | [PDF](https://arxiv.org/pdf/2606.03935v1)

**作者:** Carlo Wenig `[一作]` (University of Bonn), Christian Klos `[通讯]` (University of Bonn)

**通讯引用:** 218 | [OpenAlex ID](https://openalex.org/A5069942719)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

比较LIF与QIF神经元在基于精确Spike梯度下降的Spiking Heidelberg Digits（SHD）分类任务中的训练表现，并通过可视化损失与梯度景观深入分析两种神经元的可训练性差异。

**💡 创新点**

提出将QIF神经元作为可连续发放的模型替代传统的LIF神经元，验证其在梯度下降中的连续性优势能显著提升训练稳定性和最终分类精度，同时揭示了损失景观碎片化与Spike时序变化之间的因果关系。

**🔧 技术方法**

采用事件驱动的精确模拟（event-based simulation）与自动微分实现精确Spike梯度；使用时间到首Spike编码（time-to-first-spike）和交叉熵损失，并通过虚拟输入脉冲聚合加速计算；对损失与梯度进行二维平面截取并可视化；进行超参数搜索、训练轨迹跟踪以及梯度零分量统计。

**📊 数据集**

Spiking Heidelberg Digits（SHD）数据集，包含用耳蜗模型转换为700个输入神经元的语音数字样本。

**📈 对比分析**

通过在相同网络结构（两层前馈，128隐藏，20输出）和相同训练配置下，对LIF和QIF网络进行彻底的超参数搜索并对比。QIF网络在最佳配置下测试准确率达到90.1%，而LIF仅达到79.2%；在异步输出Spike配置下，LIF的准确率降至约50.8%。损失与梯度景观显示，QIF的碎片化程度低、梯度更稳定，训练步骤中梯度预测误差更少。

**⚠️ 局限性**

实验仅限于内在振荡的LIF/QIF神经元和导致膜电位瞬时跳跃的输入电流，未考虑更常见的指数衰减输入；网络规模相对简单，未达到SHD的最先进性能；未探讨Surrogate梯度下降或其他编码/损失方案对结果的影响。

---

## 706. GARDEN: Gravity-Aligned Reconstruction of Disentangled ENvironments from RGB images

**arXiv ID:** 2606.03921 | [PDF](https://arxiv.org/pdf/2606.03921v1)

**作者:** Jiahao Sun `[一作]` (Zhejiang University), Liang Li `[通讯]` (Zhejiang University)

**通讯引用:** 56487 | [OpenAlex ID](https://openalex.org/A5100750713)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了GARDEN框架，实现RGB‑only多视角重建、重力对齐、目标驱动的对象生成与姿态校正、条件点分类去除冗余背景，得到可物理交互的分解环境；

**💡 创新点**

创新点在于：1）以重力为统一物理先验，解决全局旋转模糊并实现Gravity‑View对齐；2）无CAD检索，直接生成与姿态校正的实体对象；3）引入条件3D点分类网络，实现对象与背景的细粒度分离；4）将分解结果直接映射到物理引擎中，保持视觉与物理一致；

**🔧 技术方法**

技术组合包括：多视RGB重建网络DepthAnything‑3、VGGT、Pi3；Gravity‑View Alignment模块（基于深度图和旋转回归）；SAM‑3、SAM‑3D用于对象掩模与几何生成；FoundationPose用于6‑DoF姿态细化；Transformer‑based点分类网络实现背景去除；3DGS或点云作为背景表示；MuJoCo实现统一物理模拟；NVS/NeRF等评估工具；

**📊 数据集**

使用的主要数据集：Hypersim、TartanAir、vKitti用于训练重力对齐与点分类；InternScenes用于无监督点分类训练；模拟数据集（如Sim场景）用于获取Ground‑Truth 3D模型；评估时使用Hypersim与TartanAir；

**📈 对比分析**

方法通过与LiteReality、PhotoShape、MIR、Phone2Proc、ACDC等基线在对象级（RMSE/SSIM/LPIPS）和场景级（RMSE/SSIM/LPIPS）进行比较，并在Gravity估计上对比Plane‑RANSAC、Normal clustering、COLMAP Manhattan、GeoCalib等；实验显示GARDEN在对象级和场景级均超过基线，尤其在post‑MuJoCo模拟协议下显著提升；计算效率提升7.7×（从4332 s降至560 s）；

**⚠️ 局限性**

局限性包括：1）依赖DepthAnything‑3的重建质量，缺少生成补全模块导致空洞；2）重力对齐的泛化受训练数据限制，缺乏垂直/水平线时易倾斜；3）在观测受限的对象上生成与姿态估计可能失败；4）缺乏闭环优化，未能充分利用物理反馈来纠正重建误差。

---

## 707. Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching

**arXiv ID:** 2606.03911 | [PDF](https://arxiv.org/pdf/2606.03911v1)

**作者:** Yoad Tewel `[一作]` (NVIDIA), Lior Wolf `[通讯]` (Tel Aviv University)

**通讯引用:** 26223 | [OpenAlex ID](https://openalex.org/A5078102229)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种无配对数据训练视觉编辑模型的方法。

**💡 创新点**

创新点是利用冻结的文本-图像模型提供的指令跟随信号与循环一致性相结合，并通过梯度路由解决训练-推理不匹配。

**🔧 技术方法**

采用流匹配生成器、基于梯度的STE梯度路由、循环一致性损失以及基于文本的方向性正则化。

**📊 数据集**

使用多种未配对图像/视频数据集，包括图像长尾风格编辑（GTA V、Minecraft等）、视频编辑（UltraVideo、Wan2.2等）以及GEdit‑Bench英文编辑指令。

**📈 对比分析**

与多种监督模型（FLUX‑Kontext、Ditto等）比较，用户研究显示本方法在视频编辑上胜率约75%，在长尾风格编辑与通用图像编辑任务中整体得分与监督模型相当甚至更优。

**⚠️ 局限性**

局限在于依赖预训练模型的知识，若目标域未被模型学习则效果差，并且在对象移除类编辑中表现不佳。

---

## 708. NetKV: Network-Aware Decode Instance Selection for Disaggregated LLM Inference

**arXiv ID:** 2606.03910 | [PDF](https://arxiv.org/pdf/2606.03910v1)

**作者:** Mubarak Adetunji Ojewale `[一作]` `[通讯]` (National College of Ireland), Mubarak Adetunji Ojewale (National College of Ireland)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出NetKV网络感知调度算法，将KV缓存命中、计算负载与网络传输成本联合评估，优化分离式LLM推理的首个令牌延迟。

**💡 创新点**

创新点在于引入轻量级网络成本oracle（包含层级距离和拥塞信号），证明忽略网络会导致子最优，并给出O(|D|)贪心算法及其对时延滞后鲁棒性证明。

**🔧 技术方法**

采用离散事件流级模拟、RDMA拥塞模型、缓存命中评估、层级带宽估计与自我流量计数等技术，算法实现为可插拔的评分器。

**📊 数据集**

使用Mooncake生产轨迹（约23K请求）并在聊天、RAG、长上下文三种工作负载下，在64GPU四层fat‑tree模拟器中实验。

**📈 对比分析**

与轮询、负载感知、缓存感知及调优的缓存+负载感知基线对比，NetKV在TTFT上比RR提升最高21.2%，比CLA*提升17.6%，SLO达成率提升至+20.1个百分点，TBT仅增加<0.5 ms。

**⚠️ 局限性**

仅在模拟器上验证，未在真实硬件上测试；模型假设串行传输与解码；按层级聚合拥塞可能忽略ECMP细节；仅针对FP16 KV缓存、TP=4设置，未覆盖常用TP=8。

---

## 709. MAdam: Metric-Aware Multi-Objective Adam

**arXiv ID:** 2606.03904 | [PDF](https://arxiv.org/pdf/2606.03904v1)

**作者:** Fengbei Liu `[一作]` (Cornell Tech), Mert R. Sabuncu `[通讯]` (Cornell Tech)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在多目标优化（MOO）中引入 Metric-Aware Multi-Objective Adam（Madam），在保持 MOO 解算器和 Adam 结构不变的前提下，通过对解算器输出方向进行基于偏好条件的对角 Fisher 预处理，解决了 Adam 与 MOO 解算器之间的权重和几何不匹配问题。

**💡 创新点**

创新点：①提出将 MOO 的线性标量化目标的对角 Fisher 作为偏好条件度量；②将该度量直接预处理解算器输出，使 Adam 的二阶矩趋向单位矩阵，从而让实际更新完全受偏好驱动；③以一个通用的、无缝集成的包装器实现，兼容所有现有 MOO 解算器与 Adam。

**🔧 技术方法**

技术：对角 Fisher 估计（使用指数移动平均并在训练期间逐步上升）；随机成对采样以降低 O(C^2) 的梯度开销；与 Adam 的自适应 RMS 预处理结合；线性标量化目标与梯度/权重平衡、梯度平衡、Pareto 前沿等多类 MOO 解算器的统一框架。

**📊 数据集**

数据集：多任务学习（MultiMNiST、SCARCos、UTKFace、Cityscapes、NYUv2）；物理信息神经网络（PINNacle 20 条 PDE benchmark）；医学图像分析（ISIC2018 皮肤病变分割、OASIS3 4× 超分辨率）。

**📈 对比分析**

比较方法：在每个基准上将 Madam 加到各种 MOO 解算器（LS、UW、DWA、IMTL、CAGrad、PaMaL、PaLoRA、LRA、NTK、RAR、LAAF、GAAF 等）之上，与原始 Adam 组合进行对比。结果表明，Madam 在多任务学习、Pareto 前沿恢复、PINN 和医学影像任务上均能显著提升性能（准确率上升、误差下降、超容积增大、L2RE 减小、感知指标提升等）。

**⚠️ 局限性**

局限性：①对异质目标（如分类+回归）时，交叉 Fisher 估计噪声较大，影响预处理质量；②当前仅适用于线性标量化目标，尚未扩展到非线性标量化（如 Tchebycheff）。

---

## 710. Agent libOS: A Library-OS-Inspired Runtime for Long-Running, Capability-Controlled LLM Agents

**arXiv ID:** 2606.03895 | [PDF](https://arxiv.org/pdf/2606.03895v1)

**作者:** Yingqi Zhang `[一作]` (Tsinghua University), Yingqi Zhang `[通讯]` (Tsinghua University)

**通讯引用:** 46750 | [OpenAlex ID](https://openalex.org/A5100456327)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出 Agent libOS，一个基于进程、对象内存、能力和审计的 LLM 代理运行时子系统，实现长时间代理的调度、权限、恢复与审计。

**💡 创新点**

通过将“工具作为 libc，原语为运行时授权边界”分离模型可见工具与资源权限，构建 OS 级别的身份、隔离、权限、阻塞和审计机制。

**🔧 技术方法**

Python 原型实现；异步调度、基于 Deno 的 TypeScript JIT 工具、SQLite 元数据存储、对象内存、能力管理、shell/文件系统/时间/人机交互原语、资源提供者层和审计日志。

**📊 数据集**

使用内部 123 条回归测试、Deterministic demo 和真实模型 smoke 脚本；未使用公开任务数据集或评测数据。

**📈 对比分析**

通过 123 条回归测试验证安全与功能属性；deterministic demo 展示完整执行路径；未进行性能基准、成本或任务成功率测评。

**⚠️ 局限性**

仅提供原语层安全，未解决语义注入、事务回滚、内核级隔离；缺少网络/数据库等后端支持；审计为日志流；性能未评估；Deno sandbox 非正式；缺乏正式验证或多节点调度。

---

## 711. MLSkip: Data Skipping for ML Filters via Lightweight Metadata

**arXiv ID:** 2606.03946 | [PDF](https://arxiv.org/pdf/2606.03946v1)

**作者:** Mihail Stoian `[一作]` (University of Technology Nuremberg), Andreas Kipf `[通讯]` (University of Technology Nuremberg)

**通讯引用:** 1078 | [OpenAlex ID](https://openalex.org/A5046188245)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了MLS Skip框架，用Parquet的最小-最大统计信息实现对机器学习过滤器的行组跳过（data skipping），并进一步设计了基于二维凸包的轻量级扩展元数据。

**💡 创新点**

创新点在于①证明即便只用已有的min-max元数据，也能为ReLU网络过滤器获得可观的跳过率；②提出一种尺寸受限的2D凸包元数据（受网格深度限制），兼顾空间占用和验证速度，显著提升跳过效果。

**🔧 技术方法**

主要技术包括：Parquet文件格式的列级统计信息、神经网络验证工具Marabou（和基于SQL的ML‑QL）、几何变换（凸包构造与网格划分）以及批量化验证调用。

**📊 数据集**

使用TPC‑H和TPC‑DS（规模因子sf=1）的表格数据，训练2–4个特征的ReLU前馈网络，构造约1.4K个不同选择率的过滤器。

**📈 对比分析**

实验比较了仅用min-max元数据与增强型凸包元数据的跳过有效率；对选择率≤0.1%的过滤器，平均跳过率从≈2%提升到≈8%（具体数值依实验而定）。在DuckDB上与PyTorch推理对比，MLS Skip实现了≈1.5–2.0倍的端到端加速。验证时间上，Marabou在双层网络下约比单层慢十倍，而凸包元数据使验证时间保持在毫秒级。

**⚠️ 局限性**

限制包括：只验证ReLU网络（不适用于Transformer等更复杂模型）；仅对数值特征有效，字符串特征尚未研究；网格深度与凸包顶点数的折衷需手工调优；大规模网络的CAD/验证成本仍高；仅在单节点CPU环境测试，GPU加速未评估。

---

## 712. AlignAtt4LLM: Fast AlignAtt for Decoder-Only LLMs at IWSLT 2026 Simultaneous Speech Translation Task

**arXiv ID:** 2606.03967 | [PDF](https://arxiv.org/pdf/2606.03967v1)

**作者:** Quentin Fuxa `[一作]` (Independent Researcher), Dominik Macháček `[通讯]` (Charles University)

**通讯引用:** 107 | [OpenAlex ID](https://openalex.org/A5056031235)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于Qwen3-ASR和Gemma-4 MT的同步语音翻译系统，利用AlignAtt策略实现decoder-only LLM的即时翻译

**💡 创新点**

创新点在于将AlignAtt从传统encoder-decoder模型迁移到decoder-only LLM，通过显式源文本提示、离线头部选择、查询/键捕获与选择性重构来实现实时决策

**🔧 技术方法**

使用了Qwen3-ASR（含强制对齐）、Gemma-4 E4B-it、vLLM推理框架、AlignAtt政策、qk-fast重构、可变的chunk同步策略、统计裁剪与阈值门控

**📊 数据集**

评测数据集为IWSLT 2026 MCIF开发集（约2.1小时学术演讲），并在Simulstream环境下进行长形式音频重分段实验

**📈 对比分析**

与IWSLT官方无上下文基线对比，系统在EN→DE和EN→IT的<2s和<4s延迟区间均获得显著BLEU、chrF和XCOMET提升；EN→ZH表现略逊，BLEU/ XCOMET仍落后；同时提供低延迟和高质量两种工作点

**⚠️ 局限性**

局限性包括对Gemma-4中文表现不足、头部选择需要离线校准、系统对ASR尾部误差敏感、以及对非欧洲语言的适配仍需进一步验证

---

## 713. Quantifying Faithful Confidence Expression in Large Reasoning Models

**arXiv ID:** 2606.03969 | [PDF](https://arxiv.org/pdf/2606.03969v1)

**作者:** Areeb Gani `[一作]` (Yale University), Arman Cohan `[通讯]` (Yale University)

**通讯引用:** 7663 | [OpenAlex ID](https://openalex.org/A5064858748)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种框架，用于量化大型推理模型（LRM）在长链式思考（CoT）输出中的可信度（faithful calibration），并系统评估其在多模型、多数据集和多提示条件下的表现。

**💡 创新点**

创新点在于：①引入前缀条件采样方法以控制CoT中的条件依赖和结构差异；②结合三种互补的内部可信度估计器（RCC、DeepConf、采样一致性）；③首次在LRM的长篇推理轨迹上量化可信度与语言表述的一致性。

**🔧 技术方法**

技术手段包括：白盒隐藏状态探测（RCC）、基于token log‑prob 的 DeepConf、基于采样一致性的黑盒估计器；LLM‑as‑Judge 对步骤层面的语言决定性进行评分；以及基于等质量置信区间的 ^* 可信度评价指标。

**📊 数据集**

数据集涵盖五种推理任务：AIME、SuperGPQA、HLE、LegalBench 和 MuSR，共计约 4000 条样本（其中 AIME 933 条、MuSR 756 条）。

**📈 对比分析**

通过在七个模型（8B、32B、671B 等）上与非推理基线、推理训练和提示干预进行对比，测得 ^* 评分在 0.64–0.78 之间；实验显示推理训练并未提升可信度，提示干预对可信度提升有限，蒸馏过程可显著改变模型的可信度表述。

**⚠️ 局限性**

局限性包括：三种估计器结果差异大，采样一致性仅评估前 20 步导致估计近似；DeepConf 的归一化可能压缩高置信区间；语言决定性评估依赖 LLM‑Judge，可能带有偏见；计算资源限制导致部分估计无法全量评估。

---

## 714. Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking

**arXiv ID:** 2606.03985 | [PDF](https://arxiv.org/pdf/2606.03985v1)

**作者:** Zekun Qi `[一作]` (Tsinghua University), Li Yi `[通讯]` (Tsinghua University)

**通讯引用:** 42460 | [OpenAlex ID](https://openalex.org/A5100421454)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了基于大规模数据和Transformer的Humanoid-GPT，实现在实时全身运动追踪中的零样本泛化。

**💡 创新点**

创新点在于：1）构建了2B帧的统一运动数据集；2）采用Harmonic Motion Embedding实现多样性平衡采样；3）将成百个RL专家通过DAgger蒸馏为单一GPT式追踪器，克服传统MLP对数据规模敏感的局限。

**🔧 技术方法**

使用技术包括：大规模动作重映射与数据增强、基于关键点奖励的PPO专家训练、Transformer causal attention架构、DAgger式序列蒸馏、ONNX/TensorRT推理优化。

**📊 数据集**

使用的数据集包括AMASS、LAFAN1、MotionMillion、PHUMA、Motion-X++、MotionMillion以及内部收集的实景MoCap数据，经过筛选、重映射、时间扭曲后得到2B帧的训练集。

**📈 对比分析**

与MLO、TCN、GMT、TWIST、Any2Track等现有追踪器对比，Humanoid-GPT在稳定性（SR）、MPJPE、MPJVE、RootVelErr、MPKPE等指标上持续提升，尤其在零样本高动态动作追踪上实现最高精度，且推理延迟保持在1.5ms以内。

**⚠️ 局限性**

局限性包括：需要巨量训练数据与算力，模型扩展仍面临数据饱和；对极端接触或多模态（视觉、语言）交互的处理仍有限；在硬件上对GPU显存和实时性有一定依赖。

---

## 715. Exploring Easy Boosts for Lidar Semantic Scene Completion

**arXiv ID:** 2606.03992 | [PDF](https://arxiv.org/pdf/2606.03992v1)

**作者:** Tetiana Martyniuk `[一作]` (Inria), Raoul de Charette `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

在点云语义场景重建（SSC）任务中，作者提出一种简单的“免费增益”方案：在输入的稀疏激光雷达点云上注入语义伪标签和可见性信息，从而显著提升多种现有SSC网络的性能。

**💡 创新点**

创新点在于：①系统评估语义先验和可见性先验对SSC的影响；②证明这两种先验可单独使用、亦可组合使用，且对所有测试模型均带来明显提升；③无需改造网络结构，只需在输入层插入额外通道，即可实现“即插即用”。

**🔧 技术方法**

技术方法包括：使用离线语义分割网络（如WaffleIron、MinkUNet）为稀疏点云生成伪标签；利用激光雷达射线在点云前沿插值空洞点，构造可见性先验；将语义和可见性信息拼接为多通道输入张量，传入现有SSC模型。实验采用SemanticKITTI数据集，评估指标为mIoU和IoU。

**📊 数据集**

数据集：SemanticKITTI，包含稀疏激光雷达点云、地面真值语义标签以及完整体素化场景。

**📈 对比分析**

比较方法：将改进后的模型与原始版本、DiffSSC、DPS2CNet等公开SOTA方法在SemanticKITTI验证集上对比。改进后四个基线平均提升约5.2 mIoU；其中SSA‑SC在改进后达到28.8 mIoU，刷新可复现单帧方法的SOTA；Oracle实验表明仍有潜在提升空间。

**⚠️ 局限性**

局限性：①改进效果高度依赖于语义分割器的性能，若分割质量差则收益有限；②可见性先验仅适用于单帧激光雷达，无法利用多帧信息；③方法对不同点云分辨率和传感器配置的鲁棒性未全面验证；④仍未解决几何不确定性对语义预测的噪声影响。

---

## 716. PixVOD: Pixel-Distributed Direct Visual Odometry and Depth Estimation

**arXiv ID:** 2606.03989 | [PDF](https://arxiv.org/pdf/2606.03989v1)

**作者:** Shinjeong Kim `[一作]` (Imperial College London), Andrew J. Davison `[通讯]` (Imperial College London)

**通讯引用:** 32144 | [OpenAlex ID](https://openalex.org/A5039230558)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在像素级分布式处理器上实现了视觉里程计与深度估计的联合优化，利用像素自身的测量和局部几何先验通过消息传递获取全局相机运动与深度图；

**💡 创新点**

创新点在于将高斯因子图与高斯信念传播（GBP）迁移到像素层面，实现全像素并行优化，并提出关键帧锚定机制以稳定相机运动估计，避免全局同步瓶颈；

**🔧 技术方法**

使用了高斯因子图、GBP、SE(3) Lie群表示、光度误差因子、法向集成因子，以及层级分裂（sharded）身份因子来实现像素间同步；

**📊 数据集**

在Replica真实场景数据集（office系列）上进行实验，采用128×128像素的高帧率子序列进行评估；

**📈 对比分析**

与集中式迭代加权最小二乘（Gauss‑Newton）基线方法比较，GBP在仅局部访问的情况下能获得相近但略逊的定位与深度精度，收敛速度慢但能在未来像素处理器上实现；

**⚠️ 局限性**

局限性包括需高帧率、短基线、准确法向估计、对光照变化与纹理缺乏鲁棒性，且无法处理大运动或非刚性场景的精确跟踪；

---

## 717. Imaginative Perception Tokens Enhance Spatial Reasoning in Multimodal Language Models

**arXiv ID:** 2606.03988 | [PDF](https://arxiv.org/pdf/2606.03988v1)

**作者:** Mahtab Bigverdi `[一作]` (University of Washington), Ranjay Krishna `[通讯]` (University of Washington)

**通讯引用:** 13371 | [OpenAlex ID](https://openalex.org/A5032451496)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并训练一种名为 Imaginative Perception Tokens（IPT）的中间视觉表示，利用它在多模态语言模型中外化对未观测空间配置（如新视角或鸟瞰图）的想象推理，并设计了三类需要想象感知的空间推理任务（视角转换、路径跟踪、多视角计数）及对应数据集。

**💡 创新点**

创新点在于：①将模型的视觉想象作为可监督的中间层，而非仅细化已观测结构；②通过让模型生成与任务相关的未观测视角或整合视图的视觉表示，提升空间推理能力；③构建了专门的基准数据集与评价标准，验证想象推理的有效性。

**🔧 技术方法**

技术实现基于统一解码器模型 BAGEL（MoT），采用流匹配（Rectified Flow）与语言建模联合训练；使用 VAE 潜在空间生成想象图像，并在推理时可选择仅输出文本答案或插入生成的想象图像以辅助推理。

**📊 数据集**

使用了 AI2-THOR、Habitat 等仿真环境生成的约 20k 样本，并采集 MessyTable、ScanNet++ 等真实世界数据作为验证集，所有数据均配有 Ground-Truth 想象图像与最终答案。

**📈 对比分析**

与零样本 VLM、统一模型、文本 Chain-of-Thought 等基线对比，IPT 在 PET、PT、MVC 等任务上均超越标签仅监督和文本 CoT，表现尤为显著：在 MVC 上提升 3.4%；在跨环境（Habitat）PET 上达到 87%+；在 Path Tracing 上在无图像推理时已接近 GPT‑5 等强基线。

**⚠️ 局限性**

局限性包括：①想象图像质量受潜在分辨率限制，影响下游推理；②IPT 任务特异性强，难以直接迁移到其他想象目标；③对真实世界多样性的覆盖仍有限；④文本 CoT 在空间推理任务中易受语言表述不匹配影响，导致性能不如 IPT。

---

## 718. NewtPhys: Do Foundation Models Understand Newtonian Physics?

**arXiv ID:** 2606.03986 | [PDF](https://arxiv.org/pdf/2606.03986v1)

**作者:** Sebastian Cavada `[一作]` (MBZUAI), Raoul de Charette `[通讯]` (Inria)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构建了一个4D物理注释数据集 NewtPhys，结合真实多视角图像与粒子级物理仿真，提供粒子位置、力、碰撞、形变等细粒度时空标签。

**💡 创新点**

首次将真实场景与粒子级物理仿真相结合，提供密集、像素对齐的物理量，填补了低级物理推理评测的空白，并在此基础上生成大规模 VQA 题库。

**🔧 技术方法**

采用 3D 高斯散射 (3DGS) 对真实场景与 GSO 对象进行建模，再通过扩展的 Simplicits 物理仿真得到粒子级力学信息；并使用视觉问答与像素级物理探测技术进行评估。

**📊 数据集**

使用来自 DL3DV 场景和 Google Scanned Objects (GSO) 的多视角图像，共计 11k 序列、730k 帧，生成 141k VQA 题对和 730k 帧的 11 种物理标签。

**📈 对比分析**

对 56 个 VLM 与 10 个 VFM 在 NewtPhys 上进行 VQA 与像素级预测评估，结果显示前沿闭源模型占优势，但开放源模型差距正缩小；整体低级物理推理准确率仍低于 30%，表明模型对力学、密度等低级量的理解仍不足。

**⚠️ 局限性**

数据集在光照、材质细节和仿真假设上仍与真实物理存在差距；评测仅针对公开模型，未做人类基准对比，且缺乏对更复杂物理现象（如高阶弹性、复杂光照）的覆盖。

---

## 719. Skill-RM: Unifying Heterogeneous Evaluation Criteria via Agent Skill

**arXiv ID:** 2606.03980 | [PDF](https://arxiv.org/pdf/2606.03980v1)

**作者:** Tao Chen `[一作]` (Qwen Large Model Application Team, Alibaba), Guanjun Jiang `[通讯]` (Qwen Large Model Application Team, Alibaba)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Skill-RM框架，将奖励模型评估过程转化为可复用的 Reward‑Evaluation Skill 执行流程。

**💡 创新点**

创新点在于将奖励评估抽象为 Agentic Skill，能够动态调用多样化资源、生成可追踪证据并实现模块化评估。

**🔧 技术方法**

采用 Agent Skill 结构、资源库（Rubric、Reference、Verifier、Checker 等）、LLM‑as‑a‑Judge 执行流程，并通过构造 Skill 接口实现评估。

**📊 数据集**

使用 RewardBench2、RM‑Bench、JudgeBench、JETTS Best‑of‑N 选取池、IF‑RewardBench 等基准数据集进行实验。

**📈 对比分析**

在多项奖励基准上与同一模型 LLM‑as‑a‑Judge 及其他奖励模型对比，Skill‑RM 平均提升约 +2.3 分，使用样本特定资源进一步提升 +5.2 分；在 Best‑of‑N 选择和指令跟随 RL 上亦表现领先。

**⚠️ 局限性**

局限性包括仅针对文本任务、需要人工编写与维护 Skill、推理成本较高、难以自动化更新以及跨模态扩展仍待研究。

---

## 720. Formalizing the Binding Problem

**arXiv ID:** 2606.03976 | [PDF](https://arxiv.org/pdf/2606.03976v1)

**作者:** Lianghuan Huang `[一作]` (University of Pennsylvania), Konrad P. Kording `[通讯]` (University of Pennsylvania)

**通讯引用:** 24359 | [OpenAlex ID](https://openalex.org/A5072047827)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本研究提出了一套信息论框架，用来定义视觉中的绑定问题，并通过可训练的探针方法在视觉Transformer（ViT）中量化绑定信息；

**💡 创新点**

创新点在于：①用互信息与熵来正式表述绑定与特征信息；②设计可解绑定的线性、二次、DNN及注意力探针；③区分绑定信息与特征信息的条件绑定；④对不同表示（summary token vs. spatial tokens）与数据集进行系统性评估；

**🔧 技术方法**

技术手段包括信息论计算（互信息、熵、归一化比值）、交叉熵训练的探针（线性、二次、多层网络、注意力），以及对DINOv2、CLIP等ViT模型的前向提取；

**📊 数据集**

使用了合成的Color‑Shape数据集、CLEVR（含可调遮挡）、以及自然的Visual Genome（Color/TopAttr子集）进行实验；

**📈 对比分析**

通过测定绑定信息、条件绑定信息及其归一化比例来比较不同表示与模型；结果显示：summary token仅编码约48%绑定信息，空间token通过注意力可解92%；模型在更高分辨率或更少遮挡时绑定能力提升；

**⚠️ 局限性**

局限性包括：依赖离散特征词表，无法直接推广到连续特征；探针测得的是可解绑定信息，未必说明模型在下游任务中实际使用该信息；条件绑定假设特征可由对象恢复，实际场景可能不成立；

---

## 721. AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation

**arXiv ID:** 2606.03972 | [PDF](https://arxiv.org/pdf/2606.03972v1)

**作者:** Haobo Li `[一作]` (SJTU), Zhipeng Zhang `[通讯]` (SJTU)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种异步对抗蒸馏框架AAD-1，实现一跳自回归视频生成。

**💡 创新点**

创新点在于分离生成器与判别器的对称性：生成器保持因果性，判别器双向全时序关注并给出视频级真实性分数，并引入分阶段训练（ODE初始化→DMD暖机→对抗细化）来稳定学习。

**🔧 技术方法**

采用异向量对抗蒸馏、分布匹配蒸馏（DMD）、ODE初始化、基于Wan 2.1 T2V的双向Transformer以及R1/R2正则。

**📊 数据集**

使用VBench公开视频数据集进行评测，并在5秒480p片段上训练。

**📈 对比分析**

与CausVid、Self‑Forcing等多步自回归基线以及Wan 2.1 100‑step对比，在一跳推理下取得Subject 94.34、Background 95.08、I2V Subject 98.65等最高或次高指标，显著优于对手。

**⚠️ 局限性**

局限在于难以处理高速运动、复杂结构细节和更长时间推理，容易出现模糊或身份漂移。

---

## 722. Planar Perfect Matching Counting is as Hard as Determinants

**arXiv ID:** 2606.03975 | [PDF](https://arxiv.org/pdf/2606.03975v1)

**作者:** Radu Curticapean `[一作]` (University of Regensburg), Jiaheng Wang `[通讯]` (University of Regensburg)

**通讯引用:** 21 | [OpenAlex ID](https://openalex.org/A5101984439)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本研究证明了Fisher–Kasteleyn–Temperley（FKT）算法在平面图完美匹配计数上的最优性，给出了匹配计数算法的下界；

**💡 创新点**

创新点在于构造了一个最优的O(m^2) 规模的网格图归约，将任意m×m行列式映射到平面格子完美匹配问题，从而得到计数下界；

**🔧 技术方法**

核心技术包括Holant框架、匹配门（matchgate）构造、行列式-匹配的等价性以及多项式时间的构造和分析；

**📊 数据集**

本文没有使用实验数据集，所有结论均来自理论构造的网格图和符号计算；

**📈 对比分析**

与现有的Yuster、Zwick等基于嵌套分割的Ω(n^{ω/2})算法相比，证明了任何算法在算术模型下都无法实现O(n^{ω/2-ε})的运行时间；

**⚠️ 局限性**

局限性包括：证明仅在算术电路或能实现Baur–Strassen定理的模型下成立，对普通整数运算或位运算的精确上界尚未给出；

---

## 723. SimuScene: Simulation-Ready Compositional 3D Scene Reconstruction from a Single Image

**arXiv ID:** 2606.03994 | [PDF](https://arxiv.org/pdf/2606.03994v1)

**作者:** Inhee Lee `[一作]` (Seoul National University), Hanbyul Joo `[通讯]` (Seoul National University)

**通讯引用:** 3894 | [OpenAlex ID](https://openalex.org/A5036077761)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

从单张图像中重建可直接放入物理仿真的完整3D场景，并通过物理仿真循环实时纠正形状与姿态误差。

**💡 创新点**

创新点在于：①将物理仿真作为诊断反馈，实时将碰撞和重力位移转化为可操作的形状修正信号；②设计了基于重力轴的伸展与基于OBB的重采样两级形状纠错机制；③对SAM3D进行流匹配式直接偏好优化（DPO），提升遮挡下的形状恢复质量。

**🔧 技术方法**

技术上使用SAM3D进行初始对象提取、VLM/MLLM进行分离基座与对象标签、MoGe获取单视深度、FoundationPose优化旋转、MuJoCo进行顺序诊断仿真、V-HACD生成碰撞体、LoRA+DPO对SAM3D形状分支进行微调，以及基于物理反馈的形状伸展与重采样。

**📊 数据集**

实验基于 GraspClutter6D、Aria Digital Twin（ADT）以及 50 张文本/图像生成的合成场景进行评估。

**📈 对比分析**

与 SAM3D、Gen3DSR、3D‑RE‑GEN 等基线比较，使用 ABO_fh/fo、penetration ratio、Mean Displacement、Peak Energy 等严格指标，SimuScene 在所有数据集上均实现了更高的物理稳定性和更低的侵入率，整体性能达到 state‑of‑the‑art。

**⚠️ 局限性**

局限性包括：①顺序诊断流程无法在后续阶段修正早期错误；②对遮挡极重或复杂相对位姿的场景仍可能出现残余误差；③对基座提取与物理约束的依赖限制了在高度动态或不规则场景中的适用性。

---

## 724. Neuron Populations Exhibit Divergent Selectivity with Scale

**arXiv ID:** 2606.03990 | [PDF](https://arxiv.org/pdf/2606.03990v1)

**作者:** Amil Dravid `[一作]` (University Of California Berkeley), Yossi Gandelsman `[通讯]` (Toyota Technological Institute At Chicago)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究神经网络中Rosetta神经元随规模的演化规律，发现其数量呈次线性幂律增长，比例下降，且随规模变得更为选择性与专一，并在语言与视觉模型上验证。

**💡 创新点**

提出Rosetta神经元的尺度律与“神经元极化”效应，并用容量分配的解析模型解释其子线性增长与选择性增强，同时展示单个Rosetta神经元在数据过滤中的实用性。

**🔧 技术方法**

使用互相最近邻匹配（Pearson相关）在不同模型间寻找Rosetta神经元，构建规模曲线，拟合幂律；同时引入特征孤立度分析与神经元选择性测度（词汇空间峰度、VLM评判）。

**📊 数据集**

语言模型：Pythia、GPT‑2、OPT、Qwen‑2.5，规模从1e8到3e10参数；视觉模型：OpenCLIP、DINOv2、Pixio、Diffusion Transformer，规模从8e7到5e9参数；数据集为The Pile（语言）与ImageNet‑1k/生成图像（视觉）。

**📈 对比分析**

通过将Rosetta神经元与非Rosetta神经元在选择性、单义性、专门化指标上对比，发现Rosetta神经元随规模显著提升；在代码数据过滤实验中，单个Rosetta神经元实现0.98 F1，继续预训练后困惑度接近oracle。

**⚠️ 局限性**

局限包括仅关注单一神经元对应关系，未涵盖注意力头、子空间等结构；选择性测度依赖于模态特定代理；未解释梯度训练动态；仅验证特定模型家族与数据集，泛化性待考。

---

## 725. Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories

**arXiv ID:** 2606.03979 | [PDF](https://arxiv.org/pdf/2606.03979v1)

**作者:** Ali Behrouz `[一作]`, Vahab Mirrokni `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种“睡眠”范式，将大型语言模型分为记忆巩固和做梦两阶段，能够在模型停留在“睡眠”状态时，先将短期脆弱记忆转化为长期稳定知识，再通过自生成数据进行自我改进。

**💡 创新点**

创新点主要在于：①记忆巩固阶段引入知识种子（上行蒸馏）与参数增量策略，将高频层知识迁移到低频层；②自生成“梦境”结合强化学习仿真学习，既提升性能又控制灾难性遗忘；③将混合专家连续记忆系统与通用知识蒸馏、梯度重要性采样等多技术融合，形成完整的持续学习框架。

**🔧 技术方法**

使用的技术包括：混合专家（MoE）连续记忆系统、参数激活/停用策略、知识种子（Knowledge Seeding）与通用知识蒸馏（GKD）结合的RL仿真学习、梯度重要性采样与SEAL框架、ReST_EM算法、以及基于自生成数据的监督微调。

**📊 数据集**

实验数据集涵盖：CLINC、Banking、DBpedia（类增量学习）；LongHealth、QASPER、MK-NIAH（长上下文）；BABILong、AIME-24/25、HMMT-25（数学推理）；Manchu/Kalamang翻译任务；SQuAD、ARC等少样本推理与知识融合任务。

**📈 对比分析**

与ICL、EWC、InCA、Hope、DuoAttention、Cartridges、SFT、GRPO、SEAL等基线进行对比。Sleep在类增量学习、长上下文理解、持续翻译、知识融合与少样本推理等任务中均优于基线，尤其在持续学习场景中显著降低灾难性遗忘，数学推理和长上下文得分提升数个百分点。

**⚠️ 局限性**

局限性包括：①对自生成数据与RL仿真学习的计算成本高；②参数增量与混合专家实现复杂，难以在极大规模模型上高效扩展；③在极端灾难性遗忘场景下仍有一定风险；④实验多在中小规模模型上验证，需进一步验证大规模模型的可行性。

---

## 726. The Grothendieck Constant is Less Than $\fracπ{2 \log (1+ \sqrt{2})} - 10^{-5}$

**arXiv ID:** 2606.03991 | [PDF](https://arxiv.org/pdf/2606.03991v1)

**作者:** Alan Li `[一作]` (University of Texas at Austin), Swarat Chaudhury `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

利用新的 Hermite 阈值分区与混合方案，对 Grothendieck 常数的上界做了改进，给出了显式的更紧的上界 < π/2·log(1+√2)−10⁻⁵。

**💡 创新点**

创新点包括：① 通过直接分析回归函数及其局部逆的泰勒展开，超越了先前的混合分析方法；② 引入 AI‑guided 搜索在 Hermite 多项式空间中寻找最优分区；③ 发现混合分区可以在绝对值级数中消除高阶系数，从而实现更小的常数。

**🔧 技术方法**

使用的技术包括：随机超平面舍入、Hermite 阈值舍入、复分析（Rouché 定理、Cauchy 推估）、区间算术实现严格数值验证、混合方案的数学框架、以及基于自动微分和 SLSQP 的 AI 优化。

**📊 数据集**

论文并未使用传统的机器学习数据集；其实验主要基于符号计算、随机生成的奇数分区以及数值优化得到的 Hermite 系数。

**📈 对比分析**

与之前的最佳上界（Krivine 1977 的 1.7822…）相比，本文的上界降低了约 6×10⁻⁵，达到了 1.782198…；在 König 双线性形式的实验中，AI 搜索得到的分区显著提高了归一化值至 0.59357…，高于超平面基准 0.5611…。

**⚠️ 局限性**

局限性在于：1) 仍未逼近 Grothendieck 常数的实际极限；2) 搜索空间受限于低阶 Hermite 多项式，可能无法发现更优的高阶结构；3) 计算成本高，需要区间算术验证；4) 对实际应用的直接影响尚不清楚，主要是理论改进。

---

## 727. Language Models Compare Quantities Using Number-specific and Unit-specific Heuristics

**arXiv ID:** 2606.03982 | [PDF](https://arxiv.org/pdf/2606.03982v1)

**作者:** Mutsumi Sasaki `[一作]` (Tohoku University), Benjamin Heinzerling `[通讯]` (RIKEN)

**通讯引用:** 791 | [OpenAlex ID](https://openalex.org/A5073408576)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了语言模型在比较带单位的量值时的行为，并系统评估了其准确率与误差模式。

**💡 创新点**

发现语言模型并非先进行精确单位换算，而是通过数值差异和单位尺度差异的启发式聚合来做出判断，并通过因果干预验证了这些特征在模型内部的存在。

**🔧 技术方法**

采用了行为实验、线性代理模型（ridge 回归）和分布式对齐搜索（DAS）进行因果干预，结合多种提示模板和不同单位设置进行评估。

**📊 数据集**

使用合成数据集，包含 5000 条跨单位设置的比较样本，训练集 20k，验证集 1k，测试集 5k，覆盖了多种物理量维度和单位系统。

**📈 对比分析**

在远离决策边界的样本中准确率高；接近边界时准确率显著下降；代理模型显示 NumLogDiff 与 UnitLogDiff 预测能力最高；DAS 干预在多层层级上达到 0.9 以上的准确率，表明这些特征对模型输出具有显著的因果影响。

**⚠️ 局限性**

局限性包括仅研究单步两量比较，代理模型为线性且可能忽略非线性交互；实验仅针对短答案生成，未涵盖多步推理或长链思路；尚未探究更大规模或跨语言的泛化能力。

---

## 728. Video-Mirai: Autoregressive Video Diffusion Models Need Foresight

**arXiv ID:** 2606.03971 | [PDF](https://arxiv.org/pdf/2606.03971v1)

**作者:** Yonghao Yu `[一作]` (University of Tokyo), Toshihiko Yamasaki `[通讯]` (University of Tokyo)

**通讯引用:** 6916 | [OpenAlex ID](https://openalex.org/A5048624196)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种仅在训练阶段使用的自监督方法Video‑Mirai，通过将未来视图的特征作为监督信号，闭合因果视频生成中的表示规划缺口；

**💡 创新点**

创新点在于利用未来信息仅作为训练目标，而非输入，保持推理阶段完全因果；在隐藏状态层引入前瞻编码器和预测器进行余弦对齐，显著提升短期与长期视频一致性；

**🔧 技术方法**

主要技术包括：因果视频扩散模型（DiT）基准、冻结的双向前瞻编码器、轻量级MLP/DiT预测器、cosine对齐损失、并结合自监督的DMD生成损失；

**📊 数据集**

在VBench（5秒）和MovieGen（30秒）数据集上进行评估，使用标准视频质量与一致性指标；

**📈 对比分析**

与Causal‑Forcing等现有因果生成基线对比，VBench 5秒总分从83.82提升至84.62，30秒长时段的主体一致性与背景一致性分别提升至88.47与91.94，性能显著提升；

**⚠️ 局限性**

局限性包括：训练阶段需要额外的前瞻编码器与预测器，推理时仅是占用空间，且对极长序列的推断仍有限；模型在极端噪声或多样化场景下可能出现信息泛化不足的情况。

---

