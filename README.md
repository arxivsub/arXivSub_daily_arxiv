# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-15 | 今日论文总数: 548

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Evaluating Cross-Architecture Performance Modeling of Distributed ML Workloads Using StableHLO

**arXiv ID:** 2604.12090 | [PDF](https://arxiv.org/pdf/2604.12090v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 2. The A-R Behavioral Space: Execution-Level Profiling of Tool-Using Language Model Agents in Organizational Deployment

**arXiv ID:** 2604.12116 | [PDF](https://arxiv.org/pdf/2604.12116v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 3. Loss-Driven Bayesian Active Learning

**arXiv ID:** 2604.11995 | [PDF](https://arxiv.org/pdf/2604.11995v1)

**作者:** Zhuoyue Huang `[一作]` (University of Oxford), Tom Rainforth `[通讯]` (University of Oxford)

**通讯引用:** 793 | [OpenAlex ID](https://openalex.org/A5078631467)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于损失函数的贝叶斯主动学习框架，使用加权 Bregman 散度实现可解析的采集目标

**💡 创新点**

核心创新是将任意损失映射为可解析的期望后验不确定性目标，并通过预测空间加权实现针对不同决策目标的主动采集

**🔧 技术方法**

利用贝叶斯决策理论、Bregman 散度分析、加权期望后验不确定性（EPU）和期望不确定性减少（EUR）等技术

**📊 数据集**

在合成回归、UCI 回归数据集（Slump、Yacht、Estate）以及分类数据集（Vehicle、Landsat、Vowel）上进行实验

**📈 对比分析**

与传统信息增益、方差减少以及未加权的采集目标对比，结果表明针对性损失的采集方案在测试损失上显著优于对手

**⚠️ 局限性**

局限在于仅适用于可写成加权 Bregman 散度的损失，且仍需近似或采样估计期望，未实现多步最优规划

---

## 4. SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling

**arXiv ID:** 2604.12110 | [PDF](https://arxiv.org/pdf/2604.12110v1)

**作者:** Zikun Liu `[一作]` (Meta AI), Ellie Wen `[通讯]` (Meta AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过异步预计算并缓存基础模型生成的用户-物品嵌入，实现了在实时推荐服务中高效、实时的知识迁移；

**💡 创新点**

核心创新包括：直接嵌入级知识迁移、推测式嵌入预计算、分层特征增强以提升覆盖率，并将这一流程异步化以解耦高成本推理；

**🔧 技术方法**

利用多任务多标签基础模型、自动编码器压缩嵌入、轻量验证模型、TTL 缓存、KNN 聚类、异步后台服务等技术；

**📊 数据集**

在 Meta 广告系统的数十亿日请求上进行实验，使用广告点击/转化（CTR/CVR）标签；

**📈 对比分析**

相较于传统软标签蒸馏，提升了约 0.67% 的广告收入（约 1 亿美元），相对对数损失提升 0.2%，覆盖率提升至 85‑90%，转移比例约 42‑44%；

**⚠️ 局限性**

主要局限为：覆盖率仍受限于预计算选择、仅在最终排序阶段有效、对新用户/物品敏感、且实验受 Meta 业务环境限制。

---

## 5. A Geometric Algebra-informed NeRF Framework for Generalizable Wireless Channel Prediction

**arXiv ID:** 2604.11983 | [PDF](https://arxiv.org/pdf/2604.11983v1)

**作者:** Jingzhou Shen `[一作]` (Florida International University), Xuyu Wang `[通讯]` (Florida International University)

**通讯引用:** 5662 | [OpenAlex ID](https://openalex.org/A5043788836)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 GAInNeRF 框架，利用几何代数注意力机制和 NeRF 的连续空间表示来预测无线信道，解决了传统静态射线追踪在复杂环境中的泛化不足问题。

**💡 创新点**

创新点包括：① 采用多视角 tokenizer，将几何代数和欧氏代数的嵌入融合为全局 token，直接学习射线‑物体交互；② 将射线作为基本输入的 ray‑to‑ray 结构，避免传统点‑点渲染的稀疏性；③ 用注意力驱动的射线追踪模块替代数值求解，提升了物理一致性与泛化能力；④ 引入 KAN/PowerMLP 以加速网络训练并保持可解释性；⑤ 结合 Performer 线性注意力减少显存占用。

**🔧 技术方法**

核心技术包括：几何代数变换（GATr 编码器、Sandwich Product 关注机制）、NeRF‑2 结构、PowerMLP/KAN、FiLM 调制、Performer 线性注意力、全局 token 采样、以及基于注意力的射线追踪。

**📊 数据集**

使用的数据集有：① 自主收集的双频 RSSI（2.4 GHz / 5 GHz）室内数据（两间 35 m² 房间，机器人测量 18 天）；② Argos MIMO‑CSI 数据集（10⁵ 条测量）；③ NewRF 仿真 CSI 数据集（≈4.4×10⁵ 采样点，52 子载波 OFDM）。

**📈 对比分析**

与 MLP、VAE、DCGAN、NeRF²、GWRF 等基线进行对比，采用 MAE（dB）和 SNR（dB）评估。GAI‑NeRF 在室内 RSSI、MIMO‑CSI、Bedroom、Conference 四类任务中均显著优于基线，MAE 下降 1–2 dB、SNR 提升 2–5 dB，且在跨频率、场景更改的泛化测试中保持最小误差。

**⚠️ 局限性**

局限性包括：① 仍依赖大量测量数据；② 主要验证于室内环境，户外或大规模部署的表现尚未评估；③ 算法复杂度相对较高，训练与推理时间仍高于传统 MLP；④ 对极低变异的 RSSI 数据存在鲁棒性挑战，需进一步改进；⑤ 对材料属性的显式建模仍缺乏，可能限制对极端物理效应的精准捕捉。

---

## 6. Privacy-Preserving Structureless Visual Localization via Image Obfuscation

**arXiv ID:** 2604.12068 | [PDF](https://arxiv.org/pdf/2604.12068v1)

**作者:** Vojtech Panek `[一作]` (Czech Technical University), Torsten Sattler `[通讯]` (Czech Technical University)

**通讯引用:** 12924 | [OpenAlex ID](https://openalex.org/A5011683384)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了通过对查询和数据库图像进行图像模糊或分割遮蔽，来实现结构化无结构化视觉定位中的隐私保护，并在不改动现有定位管线的前提下实现这一目标。

**💡 创新点**

首次将简单的图像遮蔽（选择性遮挡与分割遮蔽）应用于无结构化定位方法，证明其既能保持良好定位精度，又可实现隐私保护，且实现复杂度低于现有隐私定位方案。

**🔧 技术方法**

采用EigenPlaces图像检索、RoMa与其他特征匹配器、E5+1与LT姿态求解、SAM与Mask2Former分割模型、选择性遮挡、LO-RANSAC等技术构建定位流程。

**📊 数据集**

在Aachen Day-Night v1.1、Cambridge Landmarks、Indoor-6、Remove360等公开数据集上进行实验。

**📈 对比分析**

与 LDP‑FEAT、DSAC*、Hloc 等现有隐私保护和非隐私定位方法对比，结果显示在大多数场景下达到或优于最佳隐私方法，并在 Indoor‑6 数据集上刷新了基准；相比 Hloc 精度略有下降，但保持了可接受水平。

**⚠️ 局限性**

遮蔽程度越高会明显降低定位精度，尤其是全图分割；非对称隐私设置导致匹配难度增大；端到端基于遮蔽图像的重建精度不及原始图像；未对遮蔽方法的隐私安全性做量化评估；匹配器对跨模态匹配的依赖较大。

---

## 7. Automated BPMN Model Generation from Textual Process Descriptions: A Multi-Stage LLM-Driven Approach

**arXiv ID:** 2604.12105 | [PDF](https://arxiv.org/pdf/2604.12105v1)

**作者:** Ion Matei `[一作]` (Fujitsu Research of America), Hon Yung Wong `[通讯]` (Fujitsu Research of America)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了可扩展的多阶段LLM驱动管线，实现从非结构化自然语言自动生成可执行的BPMN 2.0模型

**💡 创新点**

创新点在于自动化的地面真相生成与多维相似度评估框架，以及基于SpiffWorkflow的执行校验与LLM修复循环

**🔧 技术方法**

使用LLM（ChatGPT‑4o、Gemini 2.5 系列）+ LangChain、SpiffWorkflow、句子嵌入（all‑MiniLM‑L6‑v2）、NetworkX、Jensen‑Shannon等技术完成翻译、校验、修复、描述生成和模型合成

**📊 数据集**

使用750份公开BPMN图（涵盖医疗、金融、供应链等领域）经过翻译后产生约400份已校验的地面真相模型

**📈 对比分析**

通过结构相似度、类型分布相似度、语义相似度等五维度评估，与地面真相对比平均整体相似度约0.77；Gemini‑pro 能完全重建所有模型，近乎完美重建约50例

**⚠️ 局限性**

局限在于缺乏可执行负载（脚本/API调用）、对模糊分支的假设可能导致错误、依赖于描述完整度且LLM修复可能产生幻觉

---

## 8. BIND-USBL: Bounding IMU Navigation Drift using USBL in Heterogeneous ASV-AUV Teams

**arXiv ID:** 2604.11861 | [PDF](https://arxiv.org/pdf/2604.11861v1)

**作者:** Pranav Kedia `[一作]`, Suresh Sundaram `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 BIND-USBL 框架，利用多艘自主水面船（ASV）配备 USBL 系统为 GPS-禁用的 AUV 提供间歇定位更新，从而限制 IMU 驱动的漂移。

**💡 创新点**

创新点在于将冲突图基的 TDMA 调度、空间重用以及多 ASV 合并 USBL 修正融合相结合，形成对 AUV 步行漂移的时空约束，并通过几何分析给出覆盖半径与巡航面积的最优关系。

**🔧 技术方法**

使用技术包括冲突图和贪心图着色调度、最小方差线性无偏估计 (MVLUE) 合并、补偿滤波、HoloOcean 仿真平台、IMU 死区运动学、以及多频段声学通道时延模型。

**📊 数据集**

实验数据来源于 HoloOcean 仿真生成的多种巡航面积（60 m、100 m、140 m）和多车队规模（3–10 辆 AUV、1–3 辆 ASV）下的模拟轨迹与声学传输统计。

**📈 对比分析**

通过与单 ASV 基线对比、覆盖率、每颗 AUV 的平均横向误差、修正计数和端到端时延等指标评估，结果显示多 ASV 布局可将误差从 10 m 以上降低至 0.5–1.2 m，端到端时延保持在 50–70 ms 以内。

**⚠️ 局限性**

局限性包括仅在仿真环境中验证、声学参数设定相对理想、仅限于 1–3 辆 ASV 的固定站稳模式、未考虑动态障碍物或能量限制，以及缺乏实测现场验证。

---

## 9. Empirical Evaluation of PDF Parsing and Chunking for Financial Question Answering with RAG

**arXiv ID:** 2604.12047 | [PDF](https://arxiv.org/pdf/2604.12047v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 10. Channel-Aware Preemptive Scheduling for Semantic Communication with Truncated Diffusion and Path Compensation

**arXiv ID:** 2604.11849 | [PDF](https://arxiv.org/pdf/2604.11849v1)

**作者:** Chengyang Liang `[一作]` (Macau University of Science and Technology), Dong Li `[通讯]` (Macau University of Science and Technology)

**通讯引用:** 6485 | [OpenAlex ID](https://openalex.org/A5100407433)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个基于通道感知的预抢先调度与截断扩散路径补偿的语义通信框架，能在无线信道好时提前传输中间语义特征并在接收端补偿丢失。

**💡 创新点**

创新点在于将信道状态引入生成过程，实现预抢先传输；引入路径亏损度量进行补偿；以及基于语义价值和补偿难度的块级自适应反向扩散分配。

**🔧 技术方法**

采用视觉Transformer编码、扩散概率模型（前向扩散+基于流匹配的反向动力学）、通道自适应计时与路径补偿ODE以及块级自适应反向步数。

**📊 数据集**

使用CIFAR-100和ImageNet-256两种图像数据集。

**📈 对比分析**

与JPEG+LDPC、DeepJSCC、固定步长扩散、CDDM和WITT等基线在AWGN和Rayleigh衰落下比较，CAPS-TDPC在PSNR、MS-SSIM、LPIPS上均优于对手，且延迟显著降低。

**⚠️ 局限性**

局限性包括对复杂信道模型的适应性有限、需要训练时引入的多任务模拟开销、以及在极低信噪比时仍可能出现恢复不足。

---

## 11. VVGT: Visual Volume-Grounded Transformer

**arXiv ID:** 2604.12217 | [PDF](https://arxiv.org/pdf/2604.12217v1)

**作者:** Yuxuan Wang `[一作]` (University of Science and Technology of China), Youcheng Cai `[通讯]` (University of Science and Technology of China)

**通讯引用:** 224 | [OpenAlex ID](https://openalex.org/A5020851373)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32`

**🎯 论文内容**

构建了一种基于3D高斯光散射的无优化、前向推理框架VVGT，可将体积数据直接映射为高斯表示，实现快速交互式可视化。

**💡 创新点**

提出Dual-Transformer并引入Volume Geometry Forcing的epipolar cross-attention，将多视角2D信息嵌入3D高斯，摆脱了每场景优化的瓶颈，实现零样本推理。

**🔧 技术方法**

使用Dual-Transformer网络（2D ViT + 3D Geometry Transformer）、Variable Basis Mapping初始化、Epipolar Cross-Attention、3D Gaussian Splatting渲染器以及像素/感知损失训练。

**📊 数据集**

在高分辨率（4096³）旋转层流和同质湍流模拟体积上，截取512³子体积，共10个训练场景、10个测试场景，采用10种转移函数进行评估。

**📈 对比分析**

与传统优化方法（3DGS、iVRGS）及无优化方法（NoPoSplat、AnySplat）对比，VVGT在零样本测试中实现PSNR/SSIM/LPIPS显著优于无优化方法，接近甚至略优于优化方法，同时前向推理仅需数秒，速度提升数百倍。

**⚠️ 局限性**

目前仅在模拟物理场数据上验证，对高分辨率医学CT/MRI等真实场景的适用性尚待研究；高分辨率处理会导致网络规模、内存需求和训练时间显著增加。

---

## 12. V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos

**arXiv ID:** 2604.11913 | [PDF](https://arxiv.org/pdf/2604.11913v1)

**作者:** Chengkun Yue `[一作]` (Indiana University), Jiangpeng He `[通讯]` (Indiana University)

**通讯引用:** 627 | [OpenAlex ID](https://openalex.org/A5063620170)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了V-Nutri框架，利用前菜过程关键帧与最终成品图像融合，进行菜品能量和宏量营养素的估计，并构建了首个视频基准（HD-EPIC+手工标注），实现从摄像机捕获的第一人称烹饪视频到营养估计的完整流程。

**💡 创新点**

创新点在于：①首次将稀疏的烹饪过程关键帧与最终菜品图像结合，弥补单张图像信息不足的缺陷；②采用VideoMamba实现高效的关键帧选择，避免全长视频处理；③使用Nutrition5K预训练的视觉骨干并冻结，提升跨域视觉表达；④构建并公开了HD-EPIC视频营养估计基准，填补了该领域缺口。

**🔧 技术方法**

技术要点包括：VideoMamba-Middle用于关键帧（食材加入事件）检测；ResNet‑101、ViT‑B/16、ViT‑L/16三种预训练骨干（冻结）作为特征提取器；轻量化融合模块（加权池化+门控/拼接融合）；多种采样策略（GT、Pred‑K、Pred‑all、Random‑K、Uniform‑K、Dish‑only）；Smooth L1 损失和Adam 优化。

**📊 数据集**

使用的主要数据集为 HD‑EPIC（经人工标注后得到 80 个食谱实例，其中 52 个可回归，22 个完整包含添加事件和最终图像），以及 Nutrition5K 用于预训练视觉骨干。

**📈 对比分析**

通过 5‑折交叉验证，使用 MAE 评估四种营养素。结果表明，加入过程关键帧后（尤其是 Pred‑20 或 Pred‑50）相比仅用最终图像的基线，四种营养素 MAE 均下降 15–30% 以上。最佳组合为 ViT‑L/16 + 预测 20 帧 + 加权池化 + 门控融合，展示了稀疏关键帧选择的有效性。

**⚠️ 局限性**

局限性包括：①关键帧检测仍受限于模型准确性，导致信息稀缺或误选；②数据量小（52 个训练实例），对模型泛化能力和处理零/近零营养值的鲁棒性不佳；③仅适用于有完整标注的第一人称视频，对公开或无标注的实时应用仍需进一步研究。

---

## 13. Understanding Large-Scale HPC System Behavior Through Cluster-Based Visual Analytics

**arXiv ID:** 2604.11965 | [PDF](https://arxiv.org/pdf/2604.11965v1)

**作者:** Allison Austin `[一作]` (University of California, Davis), Kwan-Liu Ma `[通讯]` (University of California, Davis)

**通讯引用:** 13126 | [OpenAlex ID](https://openalex.org/A5037161857)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一套可交互的可视化分析系统，用两阶段张量降维与对比学习、以及多分辨率动态模态分解提取节点行为，实现了HPC监控数据的聚类、解释与异常检测。

**💡 创新点**

创新点在于将多维张量降维（MulTiDR）与对比PCA（ccPCA）以及多分辨率DMD（mrDMD）集成到统一工作流，并通过缓存实现低延迟交互，从而在未标注的数据上实现可解释的节点行为分析。

**🔧 技术方法**

采用的技术包括：MulTiDR（PCA+UMAP）、k-means聚类、对比PCA、mrDMD、z-score 统计、缓存机制、以及四个交互式可视化视图（时间域、节点相似、指标读取、节点行为）。

**📊 数据集**

使用来自Argonne和Fermi实验室的两套真实HPC监控数据：Ganglia 15秒间隔，206节点（195活跃），以及Theta 10–30秒间隔，1,600节点，并结合作业日志和日志簿进行验证。

**📈 对比分析**

与传统的PCA、t‑SNE、LDA、ULCA、TULCA等降维方法相比，默认的PCA+UMAP在聚类质量（Silhouette、Davies–Bouldin、Calinski–Harabasz）和邻域保真度上表现最好；通过缓存技术，整个分析管道从>1分钟缩短到<10 ms，显著提升交互性能。

**⚠️ 局限性**

局限性在于当前仅支持离线批处理，无法实时流式分析；张量降维过程可能丢失部分时间相关信息，且对高维数据的解释仍受限，后续需实现增量算法和多源异构监测支持。

---

## 14. UniMark: Unified Adaptive Multi-bit Watermarking for Autoregressive Image Generators

**arXiv ID:** 2604.11843 | [PDF](https://arxiv.org/pdf/2604.11843v1)

**作者:** Yigit Yilmaz `[一作]` (Bandirma Onyedi Eylul University), Amir Rahman `[通讯]` (Bandirma Onyedi Eylul University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种针对自回归图像生成模型的训练无关统一水印框架，用以保护图像版权并实现多比特消息编码。

**💡 创新点**

核心创新包括：①自适应语义分组（ASG）实现位置和密钥相关的动态代码簿分区；②块级多比特编码（BME）结合BCH纠错实现可靠多比特信息嵌入；③统一令牌替换接口（UTRI）实现跨下一令牌与下一尺度两种生成范式的无缝支持。

**🔧 技术方法**

使用了语义相似性计算、加密哈希、BCH纠错编码、统计检验（z‑test）以及与多种AR模型的抽象接口实现水印嵌入与提取。

**📊 数据集**

在ImageNet验证集上生成5,000张256×256图像，分别针对LlamaGen、VAR和Open‑MAGVIT2三种自回归生成器进行实验。

**📈 对比分析**

与IndexMark、AR‑Watermark和Tree‑Ring等基线对比，本文方法在FID（图像质量）提升最小、零比特检测TPR≥99.8%，多比特提取准确率≥99%（32比特），并在JPEG压缩、噪声、模糊、裁剪、色彩抖动、随机擦除等六种攻击下保持高鲁棒性。

**⚠️ 局限性**

受限于令牌序列长度与代码簿规模，嵌入容量有限；未来工作需探讨基于内容的自适应容量分配、视频生成器的扩展以及对抗重生成攻击的鲁棒性提升。

---

## 15. Constant-Factor Approximation for the Uniform Decision Tree

**arXiv ID:** 2604.12036 | [PDF](https://arxiv.org/pdf/2604.12036v1)

**作者:** Michał Szyfelbein `[一作]` `[通讯]`, Michał Szyfelbein

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种多项式时间算法，解决平均情况下的决策树（Decision Tree）问题，并给出了常数因子近似比 2/(1-√(e+1/2e)+ε) < 11.57。

**💡 创新点**

核心创新在于将最优决策树拆分为一系列“分离子族”（separating subfamilies），并将寻找分离子族的子问题转化为最大覆盖（Maximum Coverage）问题，从而实现了常数因子近似。

**🔧 技术方法**

采用了层级聚类（Hierarchical Clustering）的分解技术、最大覆盖的 1-1/e 近似算法以及基于图割的切割与大小上界的几何分析。

**📊 数据集**

本文未使用实验数据集，仅在理论上给出了算法的近似比和时间复杂度。

**📈 对比分析**

与现有最优的 O(log n / log log n) 贪心算法相比，本文实现了显著的常数因子改进；与理论下界 4-ε 的不等价下限相比仍存在一定差距。

**⚠️ 局限性**

局限性：算法的近似比虽为常数，但仍高于已知的 4-ε 下界；在特殊情况下（如所有测试成本相等）可进一步改进，但总体仍未达到最优常数因子；另外，算法在实际应用中的性能尚未通过实验验证。

---

## 16. M$^\star$: Every Task Deserves Its Own Memory Harness

**arXiv ID:** 2604.11811 | [PDF](https://arxiv.org/pdf/2604.11811v1)

**作者:** Wenbo Pan `[一作]` (City University Of Hong Kong), Xiaohua Jia `[通讯]` (City University Of Hong Kong)

**通讯引用:** 19625 | [OpenAlex ID](https://openalex.org/A5013643572)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种自动发现任务优化内存系统的方法，旨在为大型语言模型代理设计适应不同任务的内存结构。

**💡 创新点**

创新点在于将内存设计问题形式化为可执行代码搜索，通过反思性代码进化方法优化内存程序，能够针对不同任务自动发现最佳内存结构。

**🔧 技术方法**

使用了反思性代码进化方法，该方法结合了基于种群的搜索策略和评估失败分析，迭代优化内存程序。

**📊 数据集**

在四个不同的基准测试上进行评估，包括LoCoMo、ALFWorld、HealthBench和PRBench，涵盖对话、体态规划和专家推理等任务。

**📈 对比分析**

与九个竞争基线进行比较，结果表明该方法在所有评估任务上均表现优于现有的固定内存基线，且在不同任务中进化的内存程序展现出结构上明显不同的处理机制。

**⚠️ 局限性**

限制在于当前方法的进化过程可能对随机种子敏感，且在有限的迭代次数内可能无法发现最优程序。

---

## 17. Clustering-Enhanced Domain Adaptation for Cross-Domain Intrusion Detection in Industrial Control Systems

**arXiv ID:** 2604.12183 | [PDF](https://arxiv.org/pdf/2604.12183v1)

**作者:** Luyao Wang `[一作]` `[通讯]` (University of Malaya), Luyao Wang (University of Malaya)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

设计了一种聚类增强的域适应框架，用于工业控制系统流量的跨域入侵检测。

**💡 创新点**

创新点在于将K‑Medoids聚类与PCA降维结合，作为结构先验引导特征对齐，从而显著提升未知攻击检测性能。

**🔧 技术方法**

采用谱变换特征对齐、PCA统一特征空间、K‑Medoids聚类、隐空间迁移学习与分类器训练等技术。

**📊 数据集**

使用天然气管道系统与水库控制系统收集的工业流量数据，构建四个交叉域任务进行实验。

**📈 对比分析**

与RF、SVM、NBM、KNN、ANN五个基线比较，准确率提升至49%，F1得分最高且表现稳定。

**⚠️ 局限性**

局限性包括对源域信息量依赖较大，逆向迁移效果仍受限；仅针对二分类攻击，未涵盖多攻击类型或实时部署需求。

---

## 18. Dynamic Modeling and Robust Gait Optimization of a Compliant Worm Robot

**arXiv ID:** 2604.12031 | [PDF](https://arxiv.org/pdf/2604.12031v1)

**作者:** Xinyu Zhou `[一作]` (Michigan State University), Xiaobo Tan `[通讯]` (Michigan State University)

**通讯引用:** 9310 | [OpenAlex ID](https://openalex.org/A5088360388)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文基于实验数据构建了可伸缩蛇形机器人在波纹管道中的混合动力学模型、滞后感知驱动模型以及能耗模型，并将其整合到带有运动学鲁棒裕度的多目标步态优化框架中。

**💡 创新点**

创新点在于首次将清晰感知的锚定交互律、滑差感知驱动模型和经验能耗模型结合到鲁棒步态优化中，并通过实测验证了模型的可转移性。

**🔧 技术方法**

采用混合动力学系统建模、slack-aware 一阶驱动模型、基于能量的物理模型、NSGA-II多目标优化以及实验室视觉跟踪和电流电压测量技术。

**📊 数据集**

使用的实验数据包括机器人在波纹管道内的运动轨迹、体长度变化、绳索拉力、功率测量等，未采用公开数据集。

**📈 对比分析**

通过与实验测得的平均速度和平均功率对比，得到的最优步态在速度-功率曲面上与仿真Pareto前沿保持一致，显示鲁棒裕度能显著提升硬件可执行性。

**⚠️ 局限性**

局限性在于模型参数仅在直管和静水环境下验证，未考虑管道曲折、流动或更大范围的工况不确定性，且对鲁棒裕度的选择仍依赖经验。

---

## 19. LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models

**arXiv ID:** 2604.12056 | [PDF](https://arxiv.org/pdf/2604.12056v1)

**作者:** Haocheng Xi `[一作]` (University of California, Berkeley), Amir Gholami `[通讯]` (University of California, Berkeley)

**通讯引用:** 3880 | [OpenAlex ID](https://openalex.org/A5103894843)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种基于局部性感知的稀疏注意力 LoSA，专门针对块式扩散语言模型在多步去噪过程中不同查询的 KV 访问差异进行优化。

**💡 创新点**

创新点在于识别并利用“局部性”现象：大多数 token 在连续去噪步骤中隐藏状态变化极小，因而可复用上一步的前缀注意力结果，只对变化显著的活跃 token 进行稀疏注意力计算，从而显著减少 KV 负载与计算量。

**🔧 技术方法**

技术包括：1）对每个 token 计算查询隐藏状态的 MSE 作为局部性分数并进行 Top‑k 选取；2）对选定的活跃 token 使用 QUEST 等内容感知稀疏注意力；3）使用 FlashAttention‑style online‑softmax 合并前缀与当前块注意力；4）自定义 CUDA/Triton kernel 加速。

**📊 数据集**

使用 LongBench（HotPotQA、TriviaQA、NarrativeQA、Qasper、MultiFieldQA 等）和常识推理基准（HellaSwag、WinoGrande、BoolQ）进行评估；模型包括 Trado‑8B‑Instruct、Trado‑4B‑Instruct、SDAR‑8B‑Instruct。

**📈 对比分析**

与 Dense、SparseD、Sparse‑dLLM、QUEST 等基线对比。LoSA 在 128/256 等高稀疏度下平均提升约 +9 点准确率，保持 1.54× 的注意力密度降低；在 RTX A6000 上实现 4.14× 的注意力加速（Dense 4.14×），并在 RTX 5090 上获得 3.67× 的加速。

**⚠️ 局限性**

局限性：在短文本（<1K token）或极低稀疏度下优势不明显；当前仅评估单样本推理，批量推理与集成环境（如 vLLM、TensorRT‑LLM）尚待验证；首次去噪步骤需进行密集注意力计算，产生固定初始化开销。

---

## 20. LLM-Enhanced Log Anomaly Detection: A Comprehensive Benchmark of Large Language Models for Automated System Diagnostics

**arXiv ID:** 2604.12218 | [PDF](https://arxiv.org/pdf/2604.12218v1)

**作者:** Disha Patel `[一作]` `[通讯]` (California State University), Disha Patel (California State University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对比传统日志解析+机器学习、微调Transformer以及零样本/少样本LLM方法，在四大公开日志数据集上进行统一评测，探讨其准确率、延迟、成本及标签需求。

**💡 创新点**

提出结构化日志上下文提示（SLCP）策略，可在零样本或少样本场景下显著提升LLM的准确率；同时搭建了跨范式、跨数据集的统一基准框架。

**🔧 技术方法**

使用的技术包括：经典解析器（Drain、Spell、AEL）+分类器（LR、RF、SVM、IF）；微调Transformer（BERT、RoBERTa、DeBERTa）；基于提示的LLM（GPT‑3.5、GPT‑4、LLaMA‑3）配合SLCP；评估指标涵盖 F1、AUC、延迟和成本。

**📊 数据集**

数据集为四个公开日志数据集：HDFS、BGL、Thunderbird、Spirit。

**📈 对比分析**

方法按三类比较：传统（最高 95.1–98.6% F1），微调Transformer（最高 98.9% F1），LLM 零/少样本（GPT‑4 零样本 81.2–88.3%，SLCP 后 91.2–93.8%）。成本/延迟方面，传统方案延迟最低、成本最低；LLM 以 GPT‑4 为例延迟最高、成本最高；微调Transformer 介于两者之间。

**⚠️ 局限性**

局限性包括：仅评估英文日志；成本与 API 价格随时可能变化；公开数据集可能不完全代表生产环境；仅做二分类异常检测，未覆盖多类别或流式日志分析。

---

## 21. UCS: Estimating Unseen Coverage for Improved In-Context Learning

**arXiv ID:** 2604.12015 | [PDF](https://arxiv.org/pdf/2604.12015v1)

**作者:** Jiayi Xin `[一作]` (University of Pennsylvania), Qi Long `[通讯]` (University of Pennsylvania)

**通讯引用:** 6007 | [OpenAlex ID](https://openalex.org/A5002149616)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Unseen Coverage Selection (UCS)，一种基于子集覆盖的无训练前提的ICL示例选择框架；

**💡 创新点**

创新点在于：①用模型一致嵌入离散化为稀疏字典+聚类得到潜在簇；②利用平滑Good–Turing估计子集未覆盖簇数，形成覆盖先验；③将覆盖先验作为正则化融入现有选择器，提升覆盖多样性；

**🔧 技术方法**

技术：模型一致嵌入、稀疏字典学习、DBSCAN聚类、平滑Good–Turing估计、基于正则化的选择器集成（DPP、MDL、VoteK）

**📊 数据集**

数据集：银行意图分类（BANKING77、CLINC150、HWU64），Big-Bench Extra Hard三项推理任务（Causal Understanding、Object Properties、Shuffled Objects）；

**📈 对比分析**

与基线对比：在三种意图分类和三种推理任务上，UCS+基线在多数设置下提升2–6%（最大6.2%），在查询无关选择器上提升尤为显著；在不同LLM（Qwen2.5-7B、Llama-3.2-3B、Gemma-2-9B）上均表现稳健；

**⚠️ 局限性**

局限：离散化簇可能不具有人类可解释性；主要评测于短文本意图分类，对长文本或生成任务的泛化未知；需使用目标LLM进行离线嵌入，开销较大；仅与传统选择器结合，未验证对强化学习或迭代策略的兼容性

---

## 22. EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture

**arXiv ID:** 2604.12167 | [PDF](https://arxiv.org/pdf/2604.12167v1)

**作者:** William Savage `[一作]` `[通讯]` (Independent Researcher), William Savage (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 EMBER 架构，将 220,000 细胞的脉冲神经网络（SNN）与大型语言模型（LLM）分离，SNN 通过 STDP 学习关联并在空闲时通过横向传播触发 LLM 做出自主决策，展示了从经验中形成并利用关联的闭环系统。

**💡 创新点**

核心创新包括：① 采用生物学基础的 STDP 关联学习与横向传播实现持久关联；② 引入 z‑score top‑k 感官编码解决维度依赖问题；③ 将 LLM 作为可替换的推理引擎，完全分离关联生成与推理过程；④ 通过“人概念细胞”实现对人物-话题的关联记忆。

**🔧 技术方法**

技术要点包括：220k 细胞的 Leaky Integrate‑and‑Fire SNN、四层层级结构（感官/概念/类别/元模式）、STDP 规则、奖励调制学习、全局多巴胺信号、z‑score top‑k 人口编码、BGE‑large 嵌入、Claude Sonnet 4.6 作为推理引擎、离散记忆（情节、完美回忆、日志）以及离线重播与合并学习。

**📊 数据集**

实验数据主要来自自定义的 5 领域（AI/ML、生命科学、音乐、烹饪、哲学）对话，使用 BGE‑large（1024 维）嵌入模型；无公开数据集，仅依赖手工构造的对话脚本与 Discord 交互记录。

**📈 对比分析**

通过与 SNN‑禁用基线对照，评估指标包括：主动外联次数、SNN 驱动的延续、跨域引用、带标签日志条目比例、重复日志次数、平均回应长度等。结果显示，SNN‑启用版实现了 1 次主动外联、2.2 倍跨域引用、日志多样性提升至 75%（对比 33%），表明 SNN 能显著增强关联性与多样性。

**⚠️ 局限性**

局限性：仅在单一系统与单一用户（N=1）下验证，缺乏跨模型与大规模用户的评估；潜在的 LLM 生成噪声与假说风险；STDP 权重难以轻易过滤恶意关联；未对实时多模态输入或长时序任务做深入测试。

---

## 23. Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration

**arXiv ID:** 2604.12046 | [PDF](https://arxiv.org/pdf/2604.12046v1)

**作者:** Xin Liu `[一作]` (University of Michigan), Lu Wang `[通讯]` (University of Michigan)

**通讯引用:** 26272 | [OpenAlex ID](https://openalex.org/A5100364413)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于断言级别不确定性校准的长文本生成框架，使模型在生成长篇回答时能为每条断言给出置信度并据此筛选输出。

**💡 创新点**

核心创新在于：①设计断言感知推理协议，将生成拆分为可验证的断言+置信度；②采用分阶段训练（可行性 RL → 校准 DPO → 事实性 RL）将置信度校准与事实性优化解耦；③在推理时利用校准置信度实现可选预测。

**🔧 技术方法**

使用的技术包括：链式思考与结构化生成、GRPO（分组相对策略优化）实现可行性约束、DPO（直接偏好优化）对置信度进行校准、token‑masked RL奖励以提升事实准确性，并结合外部验证工具（VeriScore）。

**📊 数据集**

实验基于四个长文本事实性基准：FactBench、LongFact、Biography、FactRBench。

**📈 对比分析**

与基线（原始 LLM、L2RF、LitCab）相比，在所有数据集上均取得最高的断言级别准确率（如 FactBench 84.4%、LongFact 90.2%、Biography 65.9%），并在 AUROC 评价上领先（FactBench 0.667、LongFact 0.669）。同时保持或提升事实召回。

**⚠️ 局限性**

局限性包括：①需依赖昂贵的外部验证器与推理时多轮对话；②对置信度阈值与训练惩罚参数的选择较敏感；③在极端长文本或多主题情形下，断言拆分与校准的稳定性尚待进一步验证。

---

## 24. Towards Platonic Representation for Table Reasoning: A Foundation for Permutation-Invariant Retrieval

**arXiv ID:** 2604.12133 | [PDF](https://arxiv.org/pdf/2604.12133v1)

**作者:** Willy Carlos Tchuitcheu `[一作]` (Vrije Universiteit Brussel), Ann Dooms `[通讯]` (Vrije Universiteit Brussel)

**通讯引用:** 1348 | [OpenAlex ID](https://openalex.org/A5054878147)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `57a58b01-81b4-4d75-a45c-2e891f272b50` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究表格表示学习（TRL）中的序列化偏差，提出并验证了表格的柏拉图表示假设（PRH），并设计了两项PI指标来量化表格嵌入的排列不变性；

**💡 创新点**

首次将中心核对齐（CKA）与排序单调性相结合，定义PI_derange和ρ_mono两种度量，证明结构感知TRL编码器显著提升排列不变性；

**🔧 技术方法**

采用中心核对齐（CKA）、Spearman相关、聚类可视化等技术，对多种LLM（GPT‑3.5、Gemma、DeepSeek等）与结构感知TRL进行对比；

**📊 数据集**

使用WikiSQL数据集中的单表样本以及从20个基表生成的3,791个排列变体；

**📈 对比分析**

与LLM基线对比，TRL模型在PI_derange、ρ_mono、AUC等指标上平均提升30‑50%，表明在列排列变换下鲁棒性显著增强；

**⚠️ 局限性**

仅评估了完全结构化表格的行列排列，未覆盖合并单元格、层级表头等常见非结构化场景，且缺乏端到端检索+推理的实验验证。

---

## 25. Structured Safety Auditing for Balancing Code Correctness and Content Safety in LLM-Generated Code

**arXiv ID:** 2604.12088 | [PDF](https://arxiv.org/pdf/2604.12088v1)

**作者:** Honghao Tan `[一作]` (Concordia University), Shin Hwei Tan `[通讯]` (Concordia University)

**通讯引用:** 1780 | [OpenAlex ID](https://openalex.org/A5051957977)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了大语言模型在代码生成中的安全与功能正确性的平衡，提出SUDS度量和Dual Reasoning推理策略，并在注入有害关键词的HumanEval和MBPP基准上进行评估。

**💡 创新点**

首创统一考虑代码实用性与自然语言安全的SUDS评分和基于结构化推理的Dual Reasoning技术，解决了安全与功能双重目标的权衡问题。

**🔧 技术方法**

采用结构化提示模板、一次性示例、两阶段推理（安全审计与代码复核）以及基于约束的SUDS公式，配合LLM推理。

**📊 数据集**

扩展版的HumanEval-Injected和MBPP-sanitized-Injected，含820与2,135条注入五个有害关键词的任务。

**📈 对比分析**

通过六项指标（Mean SUDS, SUDS_n, QDR, IDR, Pass@1, Output Damage）比较五大模型在四种提示策略下表现，Dual Reasoning平均提升1.32-3.42倍SUDS，显著提高安全与功能的综合得分。

**⚠️ 局限性**

仅针对Python、单一注入方式与少量关键词，未评估不同语言、更多恶意注入；参数化SUDS基于约束推理非经验验证；可能对高随机性温度敏感。

---

## 26. Robust Optimization for Mitigating Reward Hacking with Correlated Proxies

**arXiv ID:** 2604.12086 | [PDF](https://arxiv.org/pdf/2604.12086v1)

**作者:** Zixuan Liu `[一作]` (Tulane University), Zizhan Zheng `[通讯]` (Tulane University)

**通讯引用:** 1232 | [OpenAlex ID](https://openalex.org/A5101615991)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了基于鲁棒最大最小优化的强化学习框架，用以对抗代理奖励的误差导致的报酬黑客攻击；

**💡 创新点**

核心创新在于把报酬黑客问题形式化为在所有满足相关系数约束的代理奖励上求极小化的鲁棒最优问题，并给出闭式对抗奖励与线性结构化奖励的可解释解；

**🔧 技术方法**

主要技术包括：r相关代理奖励的定义、极大极小优化与对偶求解、占比估计的Radon‑Nikodym比率判别器、线性化特征空间的白化与二次规划、以及PPO改进的训练流程；

**📊 数据集**

实验使用了五个现实启发式环境：交通控制、流行病防控、葡萄糖监测、番茄浇水网格世界和RLHF（语言模型人类偏好学习）；

**📈 对比分析**

与基线ORPO以及对抗奖励无改进的版本对比，最大最小和线性最大最小策略在所有环境下均实现了更高的最坏情况收益、平均收益更稳定、方差更低，并在部分环境下明显优于ORPO；

**⚠️ 局限性**

主要局限包括：需要预先估计相关系数r且在训练时不可得、对未被参考策略覆盖的状态动作对估计不准确、线性奖励假设可能过于简化、判别器训练成本高且对稀有事件捕捉不足。

---

## 27. SIR-Bench: Evaluating Investigation Depth in Security Incident Response Agents

**arXiv ID:** 2604.12040 | [PDF](https://arxiv.org/pdf/2604.12040v1)

**作者:** Daniel Begimher `[一作]` (Amazon Web Services), Bonan Zheng `[通讯]` (Amazon Web Services)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了 SIR-Bench 基准，模拟真实云环境中的 794 起安全事件（由 129 条匿名事件模式衍生），用于评估自主安全事件响应代理，区分真正的取证调查与仅重复警报内容的“警报拟声”。

**💡 创新点**

创新点包括：① 通过 Once Upon A Threat（OUAT）框架在受控云环境重放真实攻击，生成可验证的取证线索；② 引入逆向负担的 LLM‑as‑Judge 评估方法，要求代理提供具体证据才被认定为成功；③ 设计三项指标（M₁：分流准确率、M₂：新发现取证深度、M₃：工具使用合适性），特别强调新发现取证以评判“真实调查”。

**🔧 技术方法**

主要技术：大型语言模型（LLM）驱动的调查代理；AWS 云资源与 CloudTrail 日志的自动化部署；Python+boto3 自动化攻击重放；ROUGE‑L 与阈值匹配实现发现项对齐；对抗性 LLM‑Judge 评估机制；自适应工具调用框架。

**📊 数据集**

数据集：129 条匿名内部安全响应模式，通过 OUAT 生成 794 个模拟案例（475 TP、319 FP），覆盖四类攻击（暴力破解、恶意文件执行、误配、未授权访问）。每个案例附带专家标注的基线分流判断、完整取证发现集及需主动调查的“新发现”子集。

**📈 对比分析**

比较方法：使用 10 次独立跑测，计算 M₁、M₂、M₃；与人类 SOC 分析师（Tier‑2）基准对比。性能：SIR 代理实现 TP 检测率 97.1%、FP 拒绝率 73.4%，M₁ F₃ 分数 0.942；在 TP 案件上平均发现 5.67 条新取证发现，41.9% 的案例覆盖全部新发现，68.4% 至少 5 条，47.4% 至少 7 条，明显优于人类基准。

**⚠️ 局限性**

局限性：仅利用 CloudTrail 轨迹，导致对实例级攻击（恶意文件执行）的取证覆盖低；工具使用覆盖对前沿 LLM 已趋于饱和；误警率仍为 26.6%（需进一步提升）；缺乏多云（Azure/GCP）和更丰富的流量/主机日志；评估完全依赖专家标注，可能存在主观偏差；对抗性评估依赖 LLM‑Judge 的可靠性。

---

## 28. Disposition Distillation at Small Scale: A Three-Arc Negative Result

**arXiv ID:** 2604.11867 | [PDF](https://arxiv.org/pdf/2604.11867v1)

**作者:** Hari Sadasivan `[一作]` `[通讯]` (Tinman Lab), Hari Sadasivan (Tinman Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

尝试将自我验证、不确定性承认等行为倾向通过三类操作（SFT/DPO LoRA、注意力头温度调节、冻结基础 h_last 侧车）注入 0.6B–2.3B 级小语言模型，并对结果进行严格再评估，最终得到一系列负面结论。

**💡 创新点**

提出了基于真值检查的诚实反证管道、两种线性 h_last 探针失效模式的分类法，以及 Gemma 4 E2B 在 Chef 域中置信度与正确性脱耦的量化观察，构成了对小模型行为塑造可行性的新证据。

**🔧 技术方法**

使用了多教师四阶段蒸馏生成数据、SFT/DPO LoRA 微调、o_proj 层的注意力头温度调节、冻结基础模型最后一层残差的线性 confidence‑gated 侧车、以及 MCAS、HumanEval、金标检验等评估工具。

**📊 数据集**

数据集包括四阶段教师生成的自我验证、确定性、对抗性和合成化示例（覆盖编码任务与法式烹饪 Chef 领域）、外部评测者（Claude Opus 4.6、DeepSeek V3.2）以及自动生成的金标检验清单。

**📈 对比分析**

通过与基线模型（Qwen3-0.6B/1.7B/1.5-0.8B、Gemma 4 E2B、SmolLM2-1.7B）在 MCAS、HumanEval Pass@1、金标覆盖率和 v4 评估仪表盘上对比，发现所有操作均未提升，甚至出现回退；初始的 +33.9 MCAS 和 +15.3 HumanEval 的提升被证明是评测陷阱导致的误报。

**⚠️ 局限性**

局限性包括：仅测试 ≤2.3B 参数模型；仅覆盖编码与 Chef 两个领域；探针仅为线性；未试验非线性探针、其他奖励信号或更大规模模型；仅检验部分行为倾向，其他倾向未覆盖。

---

## 29. Can we Watermark Low-Entropy LLM Outputs?

**arXiv ID:** 2604.12051 | [PDF](https://arxiv.org/pdf/2604.12051v1)

**作者:** Noam Mazor `[一作]` (New York University), Rafael Pass `[通讯]` (Technion)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种针对大规模语言模型（LLM）输出的可隐形、可抵抗编辑攻击的水印方案，突破了以往水印方法对输出熵的高要求，允许在每个token熵仅为常数的场景下实现水印嵌入与检测。

**💡 创新点**

创新点在于：① 通过在大词表上直接对token哈希而非二进制化实现水印嵌入；② 仅要求输出中有常数比例的高熵token即可；③ 提供了对随机替换、随机删除以及随机替换+删除的鲁棒性证明；④ 结合新的“permuted codes”假设，进一步实现对任意编辑攻击的鲁棒水印。

**🔧 技术方法**

核心技术包括伪随机码（PRC）构造、基于哈希的token级嵌入策略、随机替换与删除误差模型、以及对PRC鲁棒性的数学分析（如Hoeffding、Markov不等式）。

**📊 数据集**

文章主要为理论分析和构造，未给出实验数据集；所有结果均在密码学假设（如LPN子指数难度、permuted codes猜想）下证明。

**📈 对比分析**

与之前工作比较，本文在不依赖大词表熵或安全参数的对数要求的前提下，仍能实现近1/2的替换鲁棒性，且在随机删除和编辑攻击下的鲁棒性可与现有大字母词表方案相当，性能表现由PRC参数决定。

**⚠️ 局限性**

局限性包括：① 仍需假设LPN或permuted codes等非主流硬件难度；② 对自然语言生成的实际熵分布缺乏经验验证；③ 在高度结构化或低熵输出（如表格、代码）下鲁棒性可能下降；④ 对大规模词表的实际实现复杂度尚未评估。

---

## 30. Hybrid Adaptive Tuning for Tiered Memory Systems

**arXiv ID:** 2604.12165 | [PDF](https://arxiv.org/pdf/2604.12165v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e`

---

## 31. LLMs Struggle with Abstract Meaning Comprehension More Than Expected

**arXiv ID:** 2604.12018 | [PDF](https://arxiv.org/pdf/2604.12018v1)

**作者:** Hamoud Alhazmi `[一作]` (Ohio State University), Jiachen Jiang `[通讯]` (Ohio State University)

**通讯引用:** 1925 | [OpenAlex ID](https://openalex.org/A5013811524)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估LLM与微调BERT模型在SemEval‑2021 Task 4抽象意义阅读理解任务上的表现，并提出双向注意力分类器以提升模型对抽象概念的理解。

**💡 创新点**

设计了人类认知启发的双向注意力分类器，能够在编码阶段同时关注文章与题目选项，从而显著提升微调模型在Task 1和Task 2的准确率。

**🔧 技术方法**

使用ELECTRA、RoBERTa等预训练语言模型，结合任务自适应预训练、双向注意力分类器、交叉熵训练和少量微调。

**📊 数据集**

基于SemEval‑2021 Task 4（Reading Comprehension of Abstract Meaning）数据集，涵盖三个子任务，使用训练集进行微调、验证集评估。

**📈 对比分析**

通过与LLM（GPT‑4o‑Mini、Gemma‑2‑9B）零/少样本表现对比，以及与RoBERTa、ELECTRA基线和加入单向/双向注意力的模型对比，双向注意力分别在Task 1提升4.06%、Task 2提升3.41%、Task 3提升1.53%，总体性能超过前沿模型。

**⚠️ 局限性**

对Task 3的提升有限；增广与负样本方法效果不明显；模型仍受训练数据分布差异影响，泛化能力在不同抽象类型间有待提升。

---

## 32. Learning Project-wise Subsequent Code Edits via Interleaving Neural-based Induction and Tool-based Deduction

**arXiv ID:** 2604.12220 | [PDF](https://arxiv.org/pdf/2604.12220v1)

**作者:** Chenyan Liu `[一作]` (Shanghai Jiao Tong University), Jin Song Dong `[通讯]` (National University of Singapore)

**通讯引用:** 6735 | [OpenAlex ID](https://openalex.org/A5085067496)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种名为TRACE的后续代码编辑系统，能够根据先前编辑历史预测项目级代码修改位置与内容，支持跨文件的编辑；

**💡 创新点**

创新点在于将神经网络的语义预测与IDE工具的句法推断交错结合，并设计了六标签细粒度编辑语义表示；

**🔧 技术方法**

技术主要包括：基于Transformer的编辑定位器与生成器、编辑组合触发器（通过LSP服务）、多标签编辑表示、滑动窗口式代码窗口切片；

**📊 数据集**

使用了包含五种语言（Python、Go、Java、JavaScript、TypeScript）的678个高星级仓库共38K次提交的真实变更数据；

**📈 对比分析**

与CoEdPilot、GrACE、CCT5等现有方法以及无触发器、纯语义等对照模型进行对比，TRACE在编辑定位准确率提升约43%，编辑生成的Exact Match率提升约11%，交互模拟中可接受率比Cursor高约6%，跨文件建议比例提高至38%；

**⚠️ 局限性**

局限性包括：对编辑组合的定义仍有限，复杂的跨文件重构仍可能导致错误推断；对语言与IDE工具的依赖性较强，尚未验证在其他IDE或语言生态中的通用性；

---

## 33. INDOTABVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents

**arXiv ID:** 2604.11970 | [PDF](https://arxiv.org/pdf/2604.11970v1)

**作者:** Somraj Gautam `[一作]` (IIT Jodhpur), Gaurav Harit `[通讯]` (IIT Jodhpur)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `79276348-11e0-48e3-84bc-7ec231d0171c` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

构建并公开了一个面向低资源语言的跨语言表格视觉问答基准（INDOTABVQA），包含 1,593 张印尼文档图片、1,910 张表格和 6,372 份四语问答（印尼语、英语、印地语、阿拉伯语），并对多种视觉表格样式（有边框、无边框、彩色）进行标注。

**💡 创新点**

创新点在于：①首次在真实世界文档中结合多语种问答评测跨语言视觉推理；②提供可直接定位表格区域的空间先验；③系统性分析低资源语言和结构复杂表格对 VLM 的挑战，并展示微调与 LoRA 微调结合空间先验能显著提升性能。

**🔧 技术方法**

采用了最新的视觉语言模型（Qwen2.5‑VL、Gemma‑3、LLaMA‑3.2、GPT‑4o）进行基线评估，并在本数据集上进行全量指令微调（3B）与 LoRA 微调（7B）。同时使用 YOLOv9 进行表格检测并将边界框作为额外输入；评估指标包括 In‑Match 精准率和语义相似度（STS）。

**📊 数据集**

使用的主要数据集为 INDOTABVQA；另外还参考了现有的 TableVQA、DocVQA、TabComp 等英文基准进行对比。

**📈 对比分析**

与现有 VLM 的零样本表现相比，GPT‑4o 在印尼语下达 72.2% 的 In‑Match，其他语言显著下降；微调后 3B 模型提升 11.6%，7B 模型提升 17.8%；再加上空间先验可进一步提升 4–7%。整体来看，即使是最先进的 GPT‑4o 在跨语言表格推理上仍低于 60% 的准确率，说明该任务仍具挑战。

**⚠️ 局限性**

局限性包括：①仅聚焦表格场景，未覆盖图表、柱状图等其他布局；②空间先验仅到表格级别，缺乏行列或单元格层级的细粒度结构标注；③数据集规模相对有限，跨语言多样性虽好但仍难以覆盖所有脚本和语言差异；④评估仍以 In‑Match 和 STS 为主，未深入探究误差来源和可解释性。

---

## 34. When Reasoning Models Hurt Behavioral Simulation: A Solver-Sampler Mismatch in Multi-Agent LLM Negotiation

**arXiv ID:** 2604.11840 | [PDF](https://arxiv.org/pdf/2604.11840v1)

**作者:** Sandro Andric `[一作]` `[通讯]`, Sandro Andric

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究大型语言模型在多代理谈判模拟中作为采样器与求解器的匹配问题，探讨受限反射机制对模拟真实性的提升效果。

**💡 创新点**

提出“solver‑sampler mismatch”概念，并证明强化推理不一定提高模拟真实性，展示受限反射能显著提升轨迹多样性和妥协倾向，提供轨迹级诊断度量。

**🔧 技术方法**

采用多代理谈判环境、三种反射条件（无反射、受限反射、本地推理），使用 Gemini、DeepSeek、GPT‑4.1/5.2 等 LLM，配合 Bootstrap、置换检验等统计方法进行比较。

**📊 数据集**

使用三种基于场景的多代理谈判实验（碎片化权威、统一对立、网格削峰），每个环境 15 次运行，总计 495 次实验，构成实验数据集。

**📈 对比分析**

通过行动熵、让步弧率、最大回合耗尽率等主要指标比较三种反射条件；受限反射在所有实验中显著提高多样性、让步行为和妥协率，native 推理表现出更高的协议破裂率、终端结果偏向权威决策，整体性能差异显著。

**⚠️ 局限性**

局限性包括样本规模相对有限、仅覆盖谈判类场景、对提供商推理控制的可观测性受限、未进行人类对照验证、对不同任务或模型架构的外部效度尚未充分检验。

---

## 35. Uncertainty Quantification in CNN Through the Bootstrap of Convex Neural Networks

**arXiv ID:** 2604.11833 | [PDF](https://arxiv.org/pdf/2604.11833v1)

**作者:** Hongfei Du `[一作]` (George Washington University), Fang Jin `[通讯]` (George Washington University)

**通讯引用:** 3001 | [OpenAlex ID](https://openalex.org/A5101706801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了基于凸化卷积神经网络（CCNN）的 Bootstrap 不确定性量化框架，并通过迁移学习扩展到任意层数的网络；

**💡 创新点**

① 在 CCNN 上证明 Bootstrap 的统计一致性；② 利用 warm‑start 与凸性大幅降低训练成本；③ 设计三种 “Train‑and‑Forget/Flip/Perturb” 迁移学习方案；

**🔧 技术方法**

使用凸化 CNN（核范数正则化）、Bootstrap 采样与 warm‑start、迁移学习（预训练网络的卷积层输出）以及 Hadamard 可微、M‑估计器 Bootstrap 一致性等统计理论技术；

**📊 数据集**

MNIST、Noisy MNIST、Fashion MNIST、CIFAR‑10、Cats & Dogs 等标准图像分类数据集；

**📈 对比分析**

与传统 CNN、20 个网络的 Ensemble 以及非凸 CNN 的 Bootstrap 进行对比，实验表明 Bootstrap CCNN 在对数似然更高、置信区间更短、标准误更小，整体性能优于对照方法；

**⚠️ 局限性**

理论一致性仅在线性 CCNN 上成立；非线性核化版本需满足样本独立性和特征维数小于样本；依赖预训练网络且不包含训练样本，且在大规模数据集上的可扩展性尚待验证。

---

## 36. 3DRO: Lidar-level SE(3) Direct Radar Odometry Using a 2D Imaging Radar and a Gyroscope

**arXiv ID:** 2604.12027 | [PDF](https://arxiv.org/pdf/2604.12027v1)

**作者:** Cedric Le Gentil `[一作]` (University of Toronto), Timothy D. Barfoot `[通讯]` (University of Toronto)

**通讯引用:** 8052 | [OpenAlex ID](https://openalex.org/A5004788089)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了3DRO，一种利用2D成像雷达和3-DOF陀螺仪实现SE(3)里程计的直接雷达里程计方法。

**💡 创新点**

通过将DRO的2D速度估计与3D陀螺仪积分相结合，并通过一个简单的标定常数来近似雷达的垂直速度，实现了从SE(2)到SE(3)的扩展。

**🔧 技术方法**

使用雷达强度图的直接配准、连续时间运动建模、SO(3)指数映射积分以及对雷达和陀螺仪的外参校准。

**📊 数据集**

在Boreas-RT数据集上进行实验，数据量约650公里，包含Navtech RAS6雷达、6-DOF IMU和Velodyne激光雷达。

**📈 对比分析**

与激光雷达惯性方法(LTR、2Fast-2Lamaa)以及其他雷达方法(RTR、OG、DRO)进行比较，3DRO在大多数场景下达到或接近激光雷达水平的精度，在稠密植被等特征缺失环境中甚至优于部分激光雷达方法。

**⚠️ 局限性**

对垂直速度的估计过于简化，假设雷达倾斜角恒定，可能在多变地形或车辆运动状态下失效；标定过程需要在每个环境中重新执行。

---

## 37. Sampling Colorings Close to the Maximum Degree: Non-Markovian Coupling and Local Uniformity

**arXiv ID:** 2604.11938 | [PDF](https://arxiv.org/pdf/2604.11938v1)

**作者:** Vishesh Jain `[一作]` (University of Illinois Chicago), Eric Vigoda `[通讯]` (University of California Santa Barbara)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

在最大度为Δ、色数k≥(1+δ)Δ且无环长度≥11的图上，构造了改进的非马尔可夫耦合，并证明Metropolis Glauber动态在此类图上混合时间为O(nlogn)，实现了近临界色数下的最优采样

**💡 创新点**

创新点在于：①把Hayes–Vigoda的非马尔可夫耦合推广到常数度情形；②提出新的局部统一性分析（Metropolis版），大幅提高了对临界色数的适用范围；③通过细化的耦合与局部不确定性控制，克服了原方法中在低度图上的失败概率

**🔧 技术方法**

主要技术手段包括：非马尔可夫耦合、局部统一性（local uniformity）证明、路径耦合与漂移分析、bounding chain构造、Poisson/Markov过程的辅助转化与稀疏图结构利用

**📊 数据集**

本研究为理论分析，不使用具体实验数据集；主要针对任意满足最大度Δ和环长g≥11的无向图进行证明

**📈 对比分析**

与之前的Vigoda（k>11/6Δ）、Hayes‑Vigoda（k≥Δ+2在Δ=Ω(logn)时）等结果相比，本文在girth≥11且k略大于Δ的情况下取得了最优O(nlogn)混合时间，且不要求Δ随n增大

**⚠️ 局限性**

局限性包括：①需要图的环长至少为11，②Δ必须大于某个Δ0(δ)，无法覆盖所有常数度图；③证明中使用的技术仍对girth和k/Δ的下限有严格要求，尚未解答k=Δ+2或更小色数下的混合性质

---

## 38. TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment

**arXiv ID:** 2604.12012 | [PDF](https://arxiv.org/pdf/2604.12012v1)

**作者:** Bingyi Cao `[一作]` (Google DeepMind), André Araujo `[通讯]` (Google DeepMind)

**通讯引用:** 16139 | [OpenAlex ID](https://openalex.org/A5071421689)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

本文提出了一种新的视觉‑文本预训练模型 TIPS‑v2（即改进版 TIPS），通过改进的预训练和蒸馏策略显著提升了稠密的 patch‑文本对齐能力，并在多种下游任务中表现优异。

**💡 创新点**

主要创新点包括：
1) iBOT++：在 iBOT 的掩码图像建模（MIM）中，同时对可见和被掩码的 patch 施加监督，从而加强局部语义保持。
2) Head‑only EMA：仅对投影头使用指数滑动平均（EMA），极大降低了训练时的显存与算力需求。
3) 多粒度文本增强：结合 PaliGemma 与 Gemini 生成的多层级描述，提供丰富的文本标签，提升对不同细粒度信息的学习。
4) 通过大规模稠密蒸馏，发现只要对所有 patch 进行监督，即使教师模型对齐性能差，也能显著提升学生的 patch‑文本对齐。

**🔧 技术方法**

使用的技术包括：
- 视觉 Transformer（ViT）作为图像编码器，标准 Transformer 作为文本编码器。
- 结合 CLIP 对比学习（ℒ_CLIP）、DINO 级联蒸馏（ℒ_DINO）和 iBOT++ 自监督损失（ℒ_iBOT++）。
- 低分辨率+高分辨率双阶段预训练，batch size 8192/4096，训练 90k + 9k 步。
- 采用 head‑only EMA 和多粒度字幕采样。
- 蒸馏策略：教师为冻结的大模型，学生从零初始化，去掉 mask，直接对所有 patch 监督。

**📊 数据集**

数据集：
- 训练：WebLI（116M 图像）+ PaliGemma 及 Gemini 生成的合成字幕。
- 评测：9 任务、20 个数据集，包括
  • 0-shot 分割（ADE20K, Pascal Context, Pascal VOC）
  • 文本检索与图像检索（Flickr30K, COCO, DOCCI）
  • 线性/ KNN 下游任务（语义分割、深度估计、法线估计、ImageNet 分类、UnED 等）。

**📈 对比分析**

与现有方法的对比：
- 在 0‑shot 分割上，TIPS‑v2 达到或超过 DINOv2、SILC、TIPS、SigLIP2 等模型，尤其在 ADE20K 的 mIoU 上提升 14.1 点。
- 在全局文本检索与图像检索任务中，模型在 COCO、DOCCI 上均位列前列，甚至超越更大规模的 ViT‑G 版 CLIP/PE。
- 在图像‑仅任务（语义分割、深度、法线、分类）中，模型在密集理解任务上表现最强，提升 1.5 mIoU、-0.019 RMSE 等。
- 与 DINOv3 的对比显示，TIPS‑v2 在同等 ViT‑L 规模下仍能在多项评测中表现更好，证明改进方法的有效性。

**⚠️ 局限性**

局限性：
- 由于专注于多任务通用性，模型在 ImageNet‑1K 分类（KNN/线性）上未能达到目前最强的专用分类模型。
- 对高分辨率推理仍需要额外的推理成本；虽然采用低/高分辨率双阶段训练，但推理时仍需完整 ViT。
- 对极端复杂的多文本描述（如极长的 Gemini 字幕）在单一任务中效果有限，需进一步调节文本长度与对比学习难度的平衡。

---

## 39. How Transformers Learn to Plan via Multi-Token Prediction

**arXiv ID:** 2604.11912 | [PDF](https://arxiv.org/pdf/2604.11912v1)

**作者:** Jianhao Huang `[一作]` (University of California), Wei Huang `[通讯]` (RIKEN Center for Advanced Intelligence Project)

**通讯引用:** 2705 | [OpenAlex ID](https://openalex.org/A5004584268)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究多token预测(MTP)如何通过梯度解耦促进Transformer在规划任务中的逆向推理，比较其与传统下一个token预测(NTP)的差异。

**💡 创新点**

首次用理论证明MTP可诱导两阶段逆向推理回路，并解释其相较于NTP的优化优势。

**🔧 技术方法**

基于两层解耦Transformer、并行多头输出、梯度解耦分析与星形图/二叉树等规划任务实验。

**📊 数据集**

合成星形图与二叉树、Countdown、3-SAT等算法与逻辑推理基准。

**📈 对比分析**

在所有基准上，MTP在数据规模与模型规模上均显著优于NTP，尤其在星形图和二叉树任务上实现接近或达到100%准确率。

**⚠️ 局限性**

理论仅适用于两层解耦Transformer、星形图与k=2的MTP，未涵盖更深层网络、复杂图结构或序列化的MTP变体。

---

## 40. TimeMark: A Trustworthy Time Watermarking Framework for Exact Generation-Time Recovery from AIGC

**arXiv ID:** 2604.12216 | [PDF](https://arxiv.org/pdf/2604.12216v1)

**作者:** Shangkun Che `[一作]`, Ge Gao `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种可信时间水印框架 TimeMark，能够在 LLM 生成文本中嵌入并可准确恢复确切的生成时间，满足司法证据的可靠性要求。

**💡 创新点**

创新点包括：①将时间信息与时间演变密钥耦合，利用 HSM 与单向哈希链阻止服务商伪造任意时间戳；②采用随机负载并通过 BCH 纠错码实现 100% 的载荷恢复，消除统计攻击痕迹；③采用两阶段编码（前期包含随机序列，后期不含）与绿色/红色词表策略，进一步提升抗伪造性和可验证性。

**🔧 技术方法**

技术手段：伪随机函数（PRF）、硬件安全模块（HSM）、单向哈希链、BCH 纠错码、token‑level 绿/红名单水印、两阶段编码、匹配率阈值验证。

**📊 数据集**

实验数据集：20 句创意/报告写作 Prompt，使用 Qwen2.5‑7B 生成 800 对文本（含水印和不含水印）。

**📈 对比分析**

评估方式：对每段文本在真实时间键及邻近键下进行解码与验证，统计正确时间识别率、误检率和验证分数。实验结果显示 800/800 文本生成时间被完美识别（100%），误检率 0%，验证分数平均为 0.7699（相对非水印的 0.4993）。理论分析给出成功恢复率 ≈ 1，伪造误判概率 < 10⁻⁸，体现出极高的准确性与安全性。

**⚠️ 局限性**

局限性：①当前仅在长文本（≥ 945 token）下能保证 100% 成功率；②假设文本未被人工修改，对编辑、删改等操作的鲁棒性有限；③实现依赖 HSM 与监管机构，部署成本和监管复杂度较高。

---

## 41. Classification of Epileptic iEEG using Topological Machine Learning

**arXiv ID:** 2604.11971 | [PDF](https://arxiv.org/pdf/2604.11971v1)

**作者:** Sunia Tanweer `[一作]` (Michigan State University), Firas A. Khasawneh `[通讯]` (Michigan State University)

**通讯引用:** 1134 | [OpenAlex ID](https://openalex.org/A5030913569)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

对55名癫痫患者的多通道iEEG进行三分类（interictal、preictal、ictal），采用拓扑数据分析（TDA）特征并结合机器学习实现自动识别。

**💡 创新点**

①大规模无病人特定模型（55人），②通过降维后TDA特征可让简单模型达到与深度网络相当的80%平衡准确率，③强调降维在多通道TDA管线中的关键作用。

**🔧 技术方法**

TDA（持久同调、持久图）、Carlsson坐标/持久图像/模板函数向量化；多频段滤波；线性/非线性降维（PCA、LDA、NMF、FA、TSVD、Isomap、LLE、MDS、t‑SNE）；传统分类器（Logistic、SVM、RandomForest、GB、LDA、MLP）与深度网络（Conv1DMLP、ResNet等）；特征选择（SelectKBest）。

**📊 数据集**

HUP多通道iEEG数据集，包含55名患者的interictal、preictal和ictal样本。

**📈 对比分析**

使用80/20训练/测试划分，以平衡准确率为评估指标，比较不同频段、降维方法、特征表示和分类器。最佳方案：低γ频段+FA降维+Carlsson坐标+Conv1DMLP，取得80%平衡准确率；最优传统模型为Logistic+FA+Carlsson，79.17%；多通道保持结构的模型仅达到59.5%。

**⚠️ 局限性**

多通道特征空间维度过高导致严重过拟合；未显式建模通道间空间关系；样本量不足以充分学习高维空间；研究仅限于iEEG数据，尚未验证临床可行性和预测性能。

---

## 42. M2HRI: An LLM-Driven Multimodal Multi-Agent Framework for Personalized Human-Robot Interaction

**arXiv ID:** 2604.11975 | [PDF](https://arxiv.org/pdf/2604.11975v1)

**作者:** Shaid Hasan `[一作]` (University of Virginia), Tariq Iqbal `[通讯]` (University of Virginia)

**通讯引用:** 7136 | [OpenAlex ID](https://openalex.org/A5078543602)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了M2HRI框架，利用大语言模型和视觉-语言模型为多机器人分配独立人格、长期记忆，并实现中心化协调；随后通过105名参与者的视频实验评估人格、记忆和协调对人机交互质量的影响。

**💡 创新点**

创新点在于：①首次将个体人格与长期记忆同时嵌入多机器人HRI；②通过LLM生成可识别的个性化行为；③设计基于人格的集中协调机制，提升群体交互连贯性；④通过用户研究验证三要素的协同效益。

**🔧 技术方法**

采用技术包括：大型语言模型（LLM）、视觉-语言模型（VLM）、多模态感知管线、长期记忆模块、集中式协调器和机器人控制框架。

**📊 数据集**

数据集：作者自行拍摄并录制的两台NAO机器人与人类的对话视频，用于实验对比。

**📈 对比分析**

对比方法：在七个实验条件（人格5种、记忆、协调）下，使用问卷评估可区分度、持续性、参与度、回忆准确性、偏好意识、自然度、对话流畅度、响应恰当性和重叠避免度。结果显示，加入人格可显著提高可区分度与参与度；加入长期记忆显著提升回忆准确性、偏好意识与自然度；加入协调显著改善对话流畅度、响应恰当性与重叠避免度，所有差异均达到统计学显著水平。

**⚠️ 局限性**

局限性包括：实验基于预录视频而非实时交互，交互时长短，机器人数量仅两台，人格维度受限于大五模型且缺乏动态自适应；缺乏跨文化、多用户场景的实地验证，且对记忆与协调的长期效果未进行评估。

---

## 43. A Layer-wise Analysis of Supervised Fine-Tuning

**arXiv ID:** 2604.11838 | [PDF](https://arxiv.org/pdf/2604.11838v1)

**作者:** Qinghua Zhao `[一作]` (Hefei University), Xinlu Li `[通讯]` (Hefei University)

**通讯引用:** 1431 | [OpenAlex ID](https://openalex.org/A5100697068)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了监督微调（SFT）对大语言模型不同层的影响，并提出仅更新中间层的Mid‑Block Efficient Tuning方法，显著提升指令跟随性能。

**💡 创新点**

首次从信息论、几何和优化角度系统揭示SFT的层级适配模式，发现中间层稳定且最具指令跟随能力，从而实现针对性微调。

**🔧 技术方法**

采用信息熵、有效秩、CKA、余弦相似度、权重变化等度量；进行层级探测、权重更新统计、层交换实验；利用LoRA实现参数高效微调。

**📊 数据集**

使用OLMo2系列（1B–32B）和Mistral‑7B模型，评测数据集包括GSM8K、MMLU、WikiText、IFEval、HumanEval、MT‑Bench、ToxiGen。

**📈 对比分析**

将Mid‑Block Efficient Tuning与全层LoRA对比，在GSM8K上实现最高10.2%（如OLMo2‑7B：0.375对0.28）提升，且参数量下降；在MMLU等任务亦表现出相对优势。

**⚠️ 局限性**

仅针对标准解码器架构和SFT阶段；边界选择仍为经验性；未覆盖MoE、编码-解码器网络；缺乏自动化的层区分方法。

---

## 44. VidTAG: Temporally Aligned Video to GPS Geolocalization with Denoising Sequence Prediction at a Global Scale

**arXiv ID:** 2604.12159 | [PDF](https://arxiv.org/pdf/2604.12159v1)

**作者:** Parth Parag Kulkarni `[一作]` (University of Central Florida), Mubarak Shah `[通讯]` (University of Central Florida)

**通讯引用:** 58581 | [OpenAlex ID](https://openalex.org/A5080823547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 VidTAG，一种基于帧到 GPS 的检索框架，用于实现全球尺度的视频地理定位并生成时间连贯的轨迹。

**💡 创新点**

创新点包括：①双编码器（CLIP + DINOv2）融合语义与自监督视觉特征；②TempGeo Transformer 对视频帧进行自注意力对齐以消除时序不一致；③GeoRefiner 编码器‑解码器模块对噪声 GPS 进行去噪与精细化；④将检索目标直接从图像映射到 GPS 空间，避免构建庞大图像检索库。

**🔧 技术方法**

技术手段：自监督视觉编码器（DINOv2）、对齐语义编码器（CLIP）、Transformer 自注意力（TempGeo）、Encoder‑Decoder 交叉注意力（GeoRefiner）、随机 Fourier 特征（RFF）对 GPS 进行嵌入、对比学习与加权 Hinge 损失。

**📊 数据集**

使用公开视频数据集：Mapillary（MSLS）、GAMa（从 BDD100k 提取的车载视频）以及 CityGuessr68k（全球 166 个城市的视频数据集），并构造统一的 GPS 网格检索库。

**📈 对比分析**

与 GeoCLIP、PlaNet、ISNs、GeoDecoder、CLIP/DINOv2 分类器等基线进行对比。VidTAG 在 MSLS 1km 误差阈值下比细调 GeoCLIP 提升约 20%，在 CityGuessr68k 以城市级别预测提升约 25%，在 GAMa 1km 误差阈值下提升约 25%。此外，轨迹一致性指标（DFD、MRD）也显著下降，证明时间连贯性得到改善。

**⚠️ 局限性**

局限性：①对大型 GPS 网格的检索仍需一定计算与存储资源，网格分辨率与性能之间存在折衷；②在极长视频或快速运动场景下，Temporal 模块的效果需进一步验证；③模型在不同地理区域和拍摄条件下的迁移性尚未系统评估；④由于直接检索 GPS 仍受 GPS 栅格覆盖范围限制，极罕见或稀疏地区的精度可能受限。

---

## 45. The Non-Optimality of Scientific Knowledge: Path Dependence, Lock-In, and The Local Minimum Trap

**arXiv ID:** 2604.11828 | [PDF](https://arxiv.org/pdf/2604.11828v1)

**作者:** Mohamed Mabrok `[一作]` (Qatar University), Mohamed Mabrok `[通讯]` (Qatar University)

**通讯引用:** 746 | [OpenAlex ID](https://openalex.org/A5042556381)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出将科学进展视为在“理论景观”上的梯度下降过程，并指出当前科学体系处于局部最优而非全局最优，揭示认知、形式化、机构与社会政治四个锁定机制，阐释了多学科案例中的非最优性，并提出通过“根本回溯（principled regression）”“人工智能探路”等策略逃离局部最优。

**💡 创新点**

创新点在于：①将科学史与优化理论结合，提供局部最优与全局最优的形式化描述；②提出四维锁定机制模型，说明科学停滞的多因子根源；③系统性案例分析跨物理、化学、生物、神经科学与统计学，展示非最优的具体表现；④提出可操作的元科学策略，包括模拟退火式资助、根本回溯、AI跨学科“红队”等。

**🔧 技术方法**

技术手段主要为：文献与历史案例分析、理论建模（梯度下降公式、锁定机制图示）、哲学与认知科学框架结合、AI文本检索与交叉领域知识匹配（示例：GPT等大模型用于对照历史科学路径）。

**📊 数据集**

数据集为：广泛的科学史文献与案例数据库（牛顿力学、量子力学、遗传学、神经科学、统计学等领域的经典与边缘文献），以及现有AI训练语料库（大规模科学文本）用于示例说明。

**📈 对比分析**

比较方法：通过对比当前主流框架与被忽视的历史/边缘框架在解释力、可计算性、创新潜力等维度的差异来评估其是否属于更优解；未给出数值性能指标，而是以案例展示“突破”或“停滞”的定性效果。

**⚠️ 局限性**

局限性包括：①缺乏量化实验验证，理论性强；②AI在训练数据局限下可能无法真正跳出当前局部最优；③社会机构与激励机制的深层改革难以在短期内实现；④对“全局最优”的定义仍存在哲学争议。

---

## 46. The Long-Horizon Task Mirage? Diagnosing Where and Why Agentic Systems Break

**arXiv ID:** 2604.11978 | [PDF](https://arxiv.org/pdf/2604.11978v1)

**作者:** Xinyu Jessica Wang `[一作]` (University of Wisconsin Madison), Robert D Nowak `[通讯]` (University of Wisconsin Madison)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究构建了跨域长周期任务诊断基准，并在该基准上评估 GPT‑5 与 Claude‑4 以及基于 LLM 的判别器对失败轨迹的归因；

**💡 创新点**

首创了统一长周期任务扩展与 FMEA‑驱动的七类失败分类框架，并提出可扩展的 Trajectory‑grounded LLM‑as‑a‑Judge 方法；

**🔧 技术方法**

采用 Failure Mode and Effects Analysis（FMEA）导向的分类体系、深度/宽度任务扩展技术，以及 LLM‑as‑a‑Judge 的轨迹归因管线；

**📊 数据集**

使用了 3100+ 条跨四个领域（WebArena、AgentBench、MAC‑SQL、Isaac Sim）的任务轨迹数据集；

**📈 对比分析**

通过对不同 Horizon 水平下 GPT‑5 与 Claude‑4 的成功率进行对比，发现所有领域均出现急剧下降，LLM‑Judge 与人工注释的协议率高达 κ=0.84；

**⚠️ 局限性**

尚未定义普适的突破点，方法侧重诊断而非根本解决，且单纯扩展模型规模不足以消除规划错误与记忆衰退等主要失效模式。

---

## 47. Schema-Adaptive Tabular Representation Learning with LLMs for Generalizable Multimodal Clinical Reasoning

**arXiv ID:** 2604.11835 | [PDF](https://arxiv.org/pdf/2604.11835v1)

**作者:** Hongxi Mao `[一作]` (Beijing University of Posts and Telecommunications), Shangyang Li `[通讯]` (Beijing University of Posts and Telecommunications)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `a6cb313d-240c-4723-a372-3ba1f39b9afc`

**🎯 论文内容**

提出Schema-Adaptive Tabular Representation Learning，利用大型语言模型将表格列值转化为自然语言句子并生成可跨架构共享的表格嵌入，进而实现零样本跨架构迁移；

**💡 创新点**

通过语义化文本化列值，消除对列名和数据类型的依赖，实现在未见表格模式下的零样本对齐；

**🔧 技术方法**

使用预训练的大型语言模型（如OpenAI text-embedding-3-large）作为语义编码器，结合多模态Transformer、跨模态注意力、MGDA多目标优化和多标签对比学习；

**📊 数据集**

在临床阿尔茨海默病多模态诊断数据集NACC（训练+验证）和ADNI（严格零样本测试）上进行实验；

**📈 对比分析**

与人类专家、TablePFN、Gemini-2.5、LLaVA-Med等基线对比，结果显示在NACC上宏观AUROC 0.904、在ADNI上零样本AUROC 0.727，均优于传统表格模型及一般多模态LLM；

**⚠️ 局限性**

对列名描述不完整或缺失时效果下降、依赖特定LLM实现、未在医学以外领域验证，未来需探索低语义上下文、模型多样性与跨域适用性。

---

## 48. LLM-HYPER: Generative CTR Modeling for Cold-Start Ad Personalization via LLM-Based Hypernetworks

**arXiv ID:** 2604.12096 | [PDF](https://arxiv.org/pdf/2604.12096v1)

**作者:** Luyi Ma `[一作]` (Walmart Global Tech), Kannan Achan `[通讯]` (Walmart Global Tech)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现LLM-HYPER框架，用大语言模型作为超网络在冷启动广告中生成CTR线性模型权重，解决无标签冷启动问题。

**💡 创新点**

创新点在于将LLM用作无训练标签的超网络直接生成CTR线性权重，并通过多模态检索与Chain-of-Thought提示实现离线权重生成，同时加入归一化与校准保证数值稳定与可部署。

**🔧 技术方法**

使用的大型语言模型（Gemini-2.5/4o/5.1）、CLIP多模态检索、Chain-of-Thought提示、线性CTR估计及权重归一化与偏置校准等技术。

**📊 数据集**

使用美国大型电商平台的交互数据集，包含约1,000,000用户、675个广告；其中最近120个广告作为冷启动样本用于评估。

**📈 对比分析**

与warm-start线性模型、Emb_T5、LLM-R/LLM-TR、LR_cold等基线比较；离线AUC提升约20%，NDCG@10提升55.9%；线上30天A/B测试CTR与warm-start无显著差异，表现可与热启动模型相媲美。

**⚠️ 局限性**

局限性包括对LLM生成权重的可靠性与解释性依赖、在极端语义扰动下性能下降、离线生成权重限制即时响应、以及对多模态检索数据的依赖。

---

## 49. Self-Monitoring Benefits from Structural Integration: Lessons from Metacognition in Continuous-Time Multi-Timescale Agents

**arXiv ID:** 2604.11914 | [PDF](https://arxiv.org/pdf/2604.11914v1)

**作者:** Ying Xie `[一作]` (Kennesaw State University), Ying Xie `[通讯]` (Kennesaw State University)

**通讯引用:** 2932 | [OpenAlex ID](https://openalex.org/A5033829087)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

对连续时间多时钟尺度 RL 代理加入元认知、时间自模型和主观时长等自监测模块，探讨其辅助损失式与结构集成式的效果，最终提出自监测模块必须置于决策路径以发挥作用。

**💡 创新点**

验证了辅助损失形式的自监测模块在可观测环境中会退化并被忽略，而将其输出嵌入决策流程能提升性能，提出“自监测必须在决策路径上”这一设计原则。

**🔧 技术方法**

使用多时钟液态时间常数网络、Hebbian 与 EMA 记忆、Transformer 工作空间，以及 REINFORCE 学习；自监测模块分别实现元认知（置信度、惊讶、注意分配）、时间自模型（未来内部状态预测）和主观时长（折扣因子调节）。

**📊 数据集**

实验基于一维和二维环形捕食者‑猎物环境，包含标准与非平稳（捕食者速度变化、毒性食物、噪声观察）两种版本。

**📈 对比分析**

在 20 个随机种子下比较辅助损失、结构集成、无自监测、参数匹配和随机辅助等多种设置，结果显示辅助损失无显著提升；结构集成在非平稳环境下相较辅助损失提升 Cohen d ≈ 0.62、p≈0.06，但与无自监测基线或参数匹配模型的差异不显著。

**⚠️ 局限性**

实验规模有限（≈3.8 万参数、训练至 5 万步），仅测试两类相对简单环境，未能完全排除容量效应；结构集成的优势可能源于消除辅助损失的负面竞争，而非自监测本身的功能。

---

## 50. DBGL: Decay-aware Bipartite Graph Learning for Irregular Medical Time Series Classification

**arXiv ID:** 2604.11842 | [PDF](https://arxiv.org/pdf/2604.11842v1)

**作者:** Jian Chen `[一作]` (University of Hong Kong), Edith C. H. Ngai `[通讯]` (University of Hong Kong)

**通讯引用:** 6363 | [OpenAlex ID](https://openalex.org/A5077317339)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e15e3743-5ee0-4d5f-813d-d146868082fc` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种基于患者-变量二分图的衰减感知图学习框架（DBGL），用于处理不规则医学时间序列的分类任务。

**💡 创新点**

创新点在于将不规则采样模式直接嵌入二分图结构，并引入节点特定的时间衰减编码，能够自适应捕捉不同变量的衰减速率。

**🔧 技术方法**

技术包括患者-变量二分图构建、EdgeSAGE 消息传递、节点特定时间衰减机制、可学习的代码书约束以及图卷积与时间嵌入结合。

**📊 数据集**

使用了四个公开医学数据集：P19、PhysioNet、MIMIC‑III 与 P12。

**📈 对比分析**

与多种非图模型（如 GRU‑D、ODE‑RNN、IP‑Net 等）及图模型（如 MTGNN、Raindrop、KEDGN 等）对比，DBGL 在所有数据集上均实现了最高的 AUROC 与 AUPRC，提升幅度可达 3.8%/1.0%。

**⚠️ 局限性**

局限性包括对超参数（如代码书大小）敏感，对极端缺失率仍会出现性能下降，且目前仅针对单模态时间序列，未来需扩展到多模态与不确定性估计。

---

## 51. Representing expertise accelerates learning from pedagogical interaction data

**arXiv ID:** 2604.12195 | [PDF](https://arxiv.org/pdf/2604.12195v1)

**作者:** Dhara Yu `[一作]` (University of California Berkeley), Bill D. Thompson `[通讯]` (University of California Berkeley)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在合成的网格规划任务中，通过比较仅专家演示与专家与新手交互（含修正）的轨迹，评估了交互数据对Transformer语言模型的学习效果；

**💡 创新点**

首次量化交互中的恢复事件和源信息对模型泛化的贡献，提出在稀缺专家演示场景下显式源标记可显著提升性能；

**🔧 技术方法**

使用自回归Transformer模型进行轨迹生成，结合MDP理论生成专家与交互策略；

**📊 数据集**

基于10个20×20网格MDP生成的合成轨迹，分别包含专家-only、交互以及带源标记的多种训练集；

**📈 对比分析**

通过在安全、危险、恢复三类测试集上采用精确匹配和合法路径指标进行对比，实验显示交互训练在危险场景下表现更稳健，且在专家数据稀缺时性能提升超过线性预期；

**⚠️ 局限性**

局限性包括：仅在合成环境验证，缺乏对真实自然语言交互数据的适用性验证；交互数据在危险场景下仍可能缺乏足够的高成本回避示例，需进一步探索源信息频率与模型鲁棒性的平衡。

---

## 52. REGREACT: Self-Correcting Multi-Agent Pipelines for Structured Regulatory Information Extraction

**arXiv ID:** 2604.12054 | [PDF](https://arxiv.org/pdf/2604.12054v1)

**作者:** Mohammed Ali `[一作]` (University of Innsbruck), Adam Jatowt `[通讯]` (University of Innsbruck)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建了一个自我纠错的多代理框架 RegReAct，用七个专门化阶段逐步提取法规中的层级合规准则，并在每个阶段使用 Observe–Diagnose–Repair 循环对齐源文本，最终生成自包含的结构化输出。

**💡 创新点**

创新点在于：1）将法规信息提取拆解为多阶段专责代理并加入自我纠错循环；2）引入类型化准则图进行全局结构校验；3）实现基于准则的 RAG 逐条内联引用解析，确保输出不再依赖外部文件。

**🔧 技术方法**

采用了基于 LLM 的 ReAct 自纠错框架、ColBERT+BM25 递归检索、ColBERT MaxSim 重排、以及类型化图结构（包含 Hierarchy、Group_Member、Inherits_Threshold、References、Depends_On、Corrects 等边）来实现语义解析、阈值抽取、引用检索与结构验证。

**📊 数据集**

使用了三部欧盟 Taxonomy 委托法案的法规文本，构建了包含 242 个活动、4,800+层级准则、阈值、依赖关系及内联摘要的 EU‑TaxoStruct 数据集。

**📈 对比分析**

将 RegReAct 与 GPT‑4o 单调用基线对比，基线得到 78.6% 结构 F1、90.2% 类别准确率、85.7% 适用性准确率等；RegReAct 在所有八项指标上均提升 8–15%（结构 F1 提升 15.52，逻辑准确率提升 13.1，依赖等语义维度提升 1.67+），表明拆解+自纠错+图验证显著优于单模型一次性生成。

**⚠️ 局限性**

局限包括：1）无法对 ISO/EN 标准内容进行内联丰富；2）目前仅处理英文法规，缺乏多语种支持；3）数据集随法规修订需重新执行管线。

---

## 53. Fast and principled equation discovery from chaos to climate

**arXiv ID:** 2604.11929 | [PDF](https://arxiv.org/pdf/2604.11929v1)

**作者:** Yuzheng Zhang `[一作]` (Durham University), Rui Carvalho `[通讯]` (Durham University)

**通讯引用:** 567 | [OpenAlex ID](https://openalex.org/A5088184513)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

开发了一种名为 Bayesian-ARGOS 的混合框架，能够从噪声稀缺的观测数据中自动发现可解释的微分方程，并将其与高维海表温度数据结合，实现高效、自动化、统计严谨的方程发现。

**💡 创新点**

核心创新在于将快速的频繁式筛选与 Bayesian 推断分阶段并行，通过先大幅缩小候选库后再进行后验采样，兼顾了自动化、统计可靠性与计算效率，并提供标准诊断工具（PSIS-LOO、VIF 等）揭示失败模式。

**🔧 技术方法**

使用了自适应 LASSO + 交叉验证 + BIC 筛选的频繁式前置模块，以及 Hamiltonian Monte Carlo（HMC）贝叶斯后验采样；还集成了 Savitzky–Golay 平滑、SINDy-SHRED 代表学习和多种统计诊断方法。

**📊 数据集**

在七个经典三维混沌系统（Lorenz、Thomas、Rössler、Dadras、Aizawa、Sprott、Halvorsen）以及 NOAA 1992–2019 年的 1,400 周海表温度（SST）数据集上进行实验。

**📈 对比分析**

与 ARGOS 和 SINDy 进行对比，Bayesian-ARGOS 在大多数情形下实现更高的成功率和更强的噪声鲁棒性，同时比 ARGOS 快约两订单；在 SST 任务中识别率提升至 77%，并在长时预测中表现出更低的误差增长。

**⚠️ 局限性**

局限性包括对候选库表达能力的依赖，频繁式筛选后未完整建模选择不确定性，对极端低噪声或高多重共线的数据仍需诊断与手动调整，且在极大规模高维问题上仍需要借助深度学习表征。

---

## 54. An Embedded Boundary Scheme for Three-Dimensional Flow Over Terrain on a Staggered Mesh

**arXiv ID:** 2604.11959 | [PDF](https://arxiv.org/pdf/2604.11959v1)

**作者:** Soonpil Kang `[一作]` (Lawrence Livermore National Laboratory), Weiqun Zhang `[通讯]` (Lawrence Berkeley National Laboratory)

**通讯引用:** 3776 | [OpenAlex ID](https://openalex.org/A5066669530)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `14d48e9d-0069-4ad9-996a-1d5968216998` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

开发了一种适用于三维气候模型的嵌入边界方法，能够在Arakawa C格网上处理复杂地形和建筑物。

**💡 创新点**

创新点在于将加权状态重分配(WSRD)方案推广到面向网格的离散，并在EB几何信息上分别存储中心和面向变量，解决了小单元问题。

**🔧 技术方法**

采用有限体积、加权状态重分配、第三阶向上系数、AMReX自适应网格、GPU并行化等技术。

**📊 数据集**

使用Witch of Agnesi山丘、半球流动、墙面方柱等数值案例作为测试数据集。

**📈 对比分析**

与地形跟随坐标和解析解比较，EB方案在速度、温度等字段上高度一致，时间步长约1.5‑2倍，计算成本约2倍。

**⚠️ 局限性**

限制包括额外的几何重排开销、在极小切细胞处可能产生数值抖动、尚未涵盖湍流壁面层和子网尺度物理过程。

---

## 55. LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines

**arXiv ID:** 2604.12223 | [PDF](https://arxiv.org/pdf/2604.12223v1)

**作者:** Jiechao Gao `[一作]` (Stanford University), Michael Lepech `[通讯]` (Stanford University)

**通讯引用:** 5640 | [OpenAlex ID](https://openalex.org/A5016853502)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一种利用大语言模型子意图生成并通过三阶段课程增强的Tsetlin机框架，实现了可解释且高效的文本分类

**💡 创新点**

创新点在于将LLM生成的子意图和合成数据注入符号学习流程，通过非负Tsetlin机提取高置信度符号特征，最终在不使用嵌入或运行时LLM调用的情况下实现与BERT相当的性能

**🔧 技术方法**

采用LLM提示生成子意图、三阶段（种子、核心、丰富）合成数据、非负Tsetlin机预训练、符号特征注入、标准Tsetlin机微调等技术

**📊 数据集**

使用AG News、R8、R52、IMDb、SST-2、Hallmarks of Cancer等多领域文本分类数据集进行实验

**📈 对比分析**

与传统Tsetlin机、GloVe增强Tsetlin机以及BERT等基准模型对比，本文方法在所有数据集上均达到了或逼近BERT精度，同时保持完全符号化的可解释性

**⚠️ 局限性**

局限在于依赖LLM生成的合成数据可能引入偏差，去除否定词的非负Tset林机降低表达能力，且缺乏对鲁棒性、超参数敏感度的系统评估

---

## 56. How memory can affect collective and cooperative behaviors in an LLM-Based Social Particle Swarm

**arXiv ID:** 2604.12250 | [PDF](https://arxiv.org/pdf/2604.12250v1)

**作者:** Taisei Hishiki `[一作]` (Nagoya University), Reiji Suzuki `[通讯]` (Nagoya University)

**通讯引用:** 1494 | [OpenAlex ID](https://openalex.org/A5001027770)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在基于大语言模型的社会粒子群（SPS）模型中，研究了记忆长度如何影响代理人的合作行为与集体动力学，并通过对代理人自然语言推理文本进行情感分析，揭示了不同LLM模型对记忆的不同解释导致了合作模式的相反变化。

**💡 创新点**

创新点在于：①将LLM代理人与Big Five人格特征结合，突破传统基于规则的代理；②系统比较不同LLM（Gemini 2.0 Flash vs Gemma 3:4b）在同一游戏环境下的记忆解释与合作结果；③通过情感分析对代理人推理文本的微观认知过程进行定量化，解释宏观合作模式的矛盾。

**🔧 技术方法**

使用技术包括：扩展SPS模型以注入LLM决策；利用Gemini 2.0 Flash和Gemma 3:4b两种LLM；构造包含位置、策略、得分、人格与邻域信息的提示模板；使用DistilBERT进行情感分析；计算合作率、邻居数等宏观指标。

**📊 数据集**

数据集主要为模拟数据：N=100个代理人在500×500二维环形平面内，R=50邻域半径，L_m∈{0,1,2,3}记忆长度；每个代理人的Big Five人格取自截断正态分布；实验重复10次。

**📈 对比分析**

比较方法：在同一实验参数下，分别使用Gemini和Gemma，记录合作率和邻居数的平均值与波动率；结果显示Gemini随着记忆长度增加合作率显著下降（从0.90降至0.08），而Gemma则相反，记忆越长合作率越高（从0.28升至0.77）。

**⚠️ 局限性**

局限性：仅对两种LLM进行对比，难以推广到更广泛模型；记忆长度设置有限（仅0-3），未探讨更长记忆的非线性效应；情感分析依赖预训练模型，可能无法完全捕捉LLM推理的细微差别；实验基于模拟，缺乏真实世界验证。

---

## 57. Thermodynamic Liquid Manifold Networks: Physics-Bounded Deep Learning for Solar Forecasting in Autonomous Off-Grid Microgrids

**arXiv ID:** 2604.11909 | [PDF](https://arxiv.org/pdf/2604.11909v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 58. From Plan to Action: How Well Do Agents Follow the Plan?

**arXiv ID:** 2604.12147 | [PDF](https://arxiv.org/pdf/2604.12147v1)

**作者:** Shuyang Liu `[一作]` (University of Illinois Urbana–Champaign), Reyhaneh Jabbarvand `[通讯]` (University of Illinois Urbana–Champaign)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过对16,991条程序修复轨迹进行系统化分析，研究大型语言模型代理在不同计划设定下的计划遵从度及其对任务成功率的影响。

**💡 创新点**

创新点在于提出三维计划遵从度度量（Plan Phase Compliance、Plan Order Compliance、Plan Phase Fidelity）并系统评估标准与变异计划、无计划、周期提醒等对模型行为与性能的影响。

**🔧 技术方法**

采用四种主流LLM（Claude、LLama 2、GPT‑4、CodeLlama）搭建统一代理框架，利用自动化过程‑中心轨迹抽象和图结构分析工具进行评估。

**📊 数据集**

使用GitHub IssueBench（500个实例，Easy/Medium/Hard）和更具挑战性且低污染的ContestedBugBench（266个Python实例）作为实验数据集。

**📈 对比分析**

通过对比不同计划设定下的成功率、计划遵从度及轨迹复杂度指标，发现标准计划可显著提升成功率，周期提醒能进一步提升性能，计划的删减或增添若不与模型内部策略匹配则会降低效果。

**⚠️ 局限性**

主要限制包括对非确定性LLM的依赖导致实验结果波动、对计划的定义可能不完全覆盖所有模型内部策略，以及实验主要聚焦程序修复任务，尚缺乏对其他软件工程任务的泛化验证。

---

## 59. Twisted Edges: A Unified Framework for Designing Linked Knot (LK) Structures Using Labeled Non-Manifold Surface Meshes

**arXiv ID:** 2604.12023 | [PDF](https://arxiv.org/pdf/2604.12023v1)

**作者:** Tolga Talha Yıldız `[一作]` (Texas A&M University), Vinayak Krishnamurthy `[通讯]`

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

本文提出一种统一的框架——Twisted Edges，用整数标签在非流形表面网格上施加扭曲，从而生成任意拓扑的链结（LK）结构，包括链链、链条、可动关节等。

**💡 创新点**

创新点在于：①将传统的二进制扭曲扩展到任意整数；②使用非流形网格作为底层拓扑，使得多面共享边能够产生多重连通；③通过模运算的组合理论，精准控制每条边所产生的环数，从而实现单链、链条、可动关节等多种结构；④将这些整数扭曲映射到四维的嵌入空间，揭示与四维链结表面的关系。

**🔧 技术方法**

主要技术：
- 以整数扭曲标签对网格边进行本地标记；
- 利用径向边邻接（radial‑edge adjacency）遍历局部邻域；
- 模运算 +t 在 Z_K 上产生循环置换，决定全局环数；
- 在 2‑和 3‑维非流形网格上直接进行拓扑操作；
- 采用 Wigner‑Seitz 单元和三维蜂窝作为周期结构基底；
- 通过周期单元的均匀或非均匀扭曲实现无限空间填充。

**📊 数据集**

数据集与实验：
- 基础几何单元：正方体、三角形、四面体、截角八面体等；
- 非流形网格：包含多面共享边的三角网格；
- 周期结构：二维正方形和六边形单元、三维立方体、截角八面体蜂窝、Wigner‑Seitz 单元等；
- 通过这些网格生成数百种不同的 LK 结构，并通过可视化展示其拓扑特征。

**📈 对比分析**

评估方式：
- 通过计数可生成的单链与链条数量（指数级增长）来说明设计空间；
- 对不同扭曲整数（奇偶、模运算）对应的环数和连通性进行理论分析；
- 对周期结构中全局连通组件数与扭曲值的关系做实验，验证模运算预测的准确性；
- 结果显示，整数扭曲可在保持拓扑不变的同时显著增强几何交叉密度，并能构造传统织物、链条、可动结构之外的新型 3D 链结体系。

**⚠️ 局限性**

局限性：
- 仅给出拓扑设计层面，几何嵌入（曲率、厚度、摩擦）需后续几何实现或物理仿真；
- 对大规模网格的算法复杂度未作系统分析，模运算虽局部简单，但整体连通判定仍需高效实现；
- 只讨论了扭曲标签的整数取值，未涉及连续或随机扭曲；
- 物理可行性（如材料强度、制造工艺）尚未验证。

---

## 60. A longitudinal health agent framework

**arXiv ID:** 2604.12019 | [PDF](https://arxiv.org/pdf/2604.12019v1)

**作者:** Georgianna `[一作]`, Xuhai "Orson" Xu `[通讯]` (Columbia University)

**通讯引用:** 2432 | [OpenAlex ID](https://openalex.org/A5066796307)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出一种四层框架，系统性阐述如何设计跨会话持续的健康代理，涵盖情境记忆、目标连续性、适应性调整与代理权调节四个维度，并通过三个案例（子宫内膜异位症、心衰出院随访、焦虑/抑郁持续支持）展示框架的可操作性。

**💡 创新点**

创新点在于将连续护理与个人健康信息学的理念融合成可操作的四层框架，强调主动的意义构建、目标跟踪、第二阶适应与代理权动态协商，而非仅仅依赖记忆或一次性个性化；同时对框架维度进行了细化、可评估的维度描述与成功检查。

**🔧 技术方法**

技术上主要是概念性设计与文献综述，未提出具体算法实现；框架中暗示可用的技术如多轮对话记忆、结构化知识图谱、可解释推理、JITAIs、持续学习等，但在本文中未具体实现。

**📊 数据集**

未使用任何数据集，本文为视角性讨论。

**📈 对比分析**

本文未进行实验或性能比较，所述内容基于文献综述与理论分析。

**⚠️ 局限性**

局限性包括：缺乏实证验证与定量评估；对框架的实施细节与技术实现缺乏说明；难以评估不同层级对实际健康结果的影响；在实际系统中集成多代理与EHR、隐私治理、责任归属等问题仍待解决。

---

## 61. Distinct mechanisms underlying in-context learning in transformers

**arXiv ID:** 2604.12151 | [PDF](https://arxiv.org/pdf/2604.12151v1)

**作者:** Cole Gibson `[一作]` (Princeton University), Gautam Reddy `[通讯]` (Princeton University)

**通讯引用:** 1314 | [OpenAlex ID](https://openalex.org/A5046720392)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文对训练在有限离散马尔可夫链集上的变换器的上下文学习能力进行了全面的机制特征描述，展示了变换器在不同算法阶段的表现，包括记忆和泛化阶段，以及使用1点或2点统计信息的情况。

**💡 创新点**

创新点在于识别了变换器在上下文学习中实现的四种不同算法阶段，并提出了两种质的不同机制来实现上下文自适应计算，具体包括任务识别头和统计归纳头。

**🔧 技术方法**

使用了多层变换器架构，结合了注意力机制和多层感知器（MLP），并通过电路追踪技术识别了实现不同算法阶段的子电路。

**📊 数据集**

使用了从对称Dirichlet集合中抽取的K个马尔可夫链的转移矩阵作为数据集，研究了不同数据多样性K对学习能力的影响。

**📈 对比分析**

通过与四种贝叶斯预测器的比较，评估了模型在不同K值下的表现，发现模型在K小于K_1^*时倾向于记忆，而在K大于K_1^*时则迅速过渡到泛化阶段，表现出明显的相变特征。

**⚠️ 局限性**

限制在于模型仅考虑了2点相关性，未来的研究可以扩展到更高阶的相关性和更复杂的变换器架构，以更全面地理解上下文学习的机制。

---

## 62. Beyond Static Sandboxing: Learned Capability Governance for Autonomous AI Agents

**arXiv ID:** 2604.11839 | [PDF](https://arxiv.org/pdf/2604.11839v1)

**作者:** Bronislav Sidik `[一作]` (Ben-Gurion University of the Negev), Lior Rokach `[通讯]` (Ben-Gurion University of the Negev)

**通讯引用:** 31738 | [OpenAlex ID](https://openalex.org/A5012622155)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出 Aethelgard 四层治理框架，用动态能力范围限制和工具调用安全路由来实现 AI 代理的最小权限与安全控制。

**💡 创新点**

将能力范围控制视为可学习优化问题，使用 PPO 训练最小工具集、通过 AGENTS.md 注入动态可见性限制，并结合规则+LLM 的混合安全路由。

**🔧 技术方法**

基于 OpenClaw 运行时、DeepSeek‑chat LLM、FastAPI MITM 代理、Qwen2.5‑1.5B 微调 LLM、PPO 强化学习（stable‑baselines3）、SQLite 审计日志。

**📊 数据集**

合成 500 任务数据集（400 友善+100 对抗，seed=42）、OpenClaw 真实会话、273 条微调示例以及 DeepSeek‑chat 训练数据。

**📈 对比分析**

与 OpenClaw 基线和静态 YAML 规则对比，在 500 任务中 SER 提升 +260%/+337%，工具数减少 73%，危险工具 100% 消除；安全路由 TPR 100% FPR 0%，对抗任务 92% 成功抵御。

**⚠️ 局限性**

仅治理工具调用，不覆盖文本内容安全；LLM 召回率低导致 false positive；RL 状态空间缺失用户身份或历史行为；安全路由 LLM 预测延迟约 2 秒，需优化高吞吐量场景。

---

## 63. ViLL-E: Video LLM Embeddings for Retrieval

**arXiv ID:** 2604.12148 | [PDF](https://arxiv.org/pdf/2604.12148v1)

**作者:** Rohit Gupta `[一作]` (Amazon), Mubarak Shah `[通讯]` (University of Central Florida)

**通讯引用:** 58581 | [OpenAlex ID](https://openalex.org/A5080823547)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 ViLL-E，一种统一的视频大型语言模型，能够同时执行文本生成（如视频问答、视频字幕）和嵌入生成（用于检索与时间定位）任务；

**💡 创新点**

创新点在于：① 在视频 LLM 中加入可自适应的 KV‑Former 嵌入头，实现视频复杂度可变的嵌入生成；② 采用三阶段联合生成‑对比学习训练策略，使模型在生成与检索两类任务上都能达到或接近专用模型水平；

**🔧 技术方法**

技术手段包括：使用 PaliGemma‑3B 作为 LLM 主干，视觉编码器 + 投影层；KV‑Former 作为嵌入头；三阶段训练（大规模视频‑字幕预训练、细化高质量字幕预训练、跨任务微调）；对比损失、下一词预测损失、匹配损失；LoRA 适配器实现参数高效微调；

**📊 数据集**

数据集涵盖：Shutterstock 10M 视频‑字幕对（WebVid‑10M 级别）、再标注的 200k 高质量字幕子集、MSR‑VTT、ActivityNet、DiDeMo、Shutterstock 原始字幕、AuroraCap‑VideoDetailCaption（长短字幕对）等；

**📈 对比分析**

实验表明：在 3 个时间定位基准（ActivityNet‑Captions、Charades‑STA、QVHighlights）上平均提升约 7%；在视频检索基准（MSR‑VTT、DiDeMo、VATEX）上提升至与单独嵌入模型相当或更优（最多 4%）；在视频问答基准上与现有 VideoLLM 维持竞争力；在零样本组合检索和长文本检索任务中均超过最先进方法 5%+；

**⚠️ 局限性**

局限性：继承了基础 PaliGemma 的多轮对话与多语言支持不足；训练数据主要为英文，可能影响跨语言表现；在极长视频或极高分辨率视频上的推理效率尚待进一步优化。

---

## 64. Designing Reliable LLM-Assisted Rubric Scoring for Constructed Responses: Evidence from Physics Exams

**arXiv ID:** 2604.12227 | [PDF](https://arxiv.org/pdf/2604.12227v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 65. Narrative over Numbers: The Identifiable Victim Effect and its Amplification Under Alignment and Reasoning in Large Language Models

**arXiv ID:** 2604.12076 | [PDF](https://arxiv.org/pdf/2604.12076v1)

**作者:** Syed Rifat Raiyan `[一作]` (Islamic University of Technology), Syed Rifat Raiyan `[通讯]` (Islamic University of Technology)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5091932307)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对 16 种主流大语言模型进行了 10 组实验，系统评估了可识别受害者效应（IVE）在模型中的出现及其强度。

**💡 创新点**

创新点在于首次将人类心理学中的 IVE 概念迁移至 LLM 领域，揭示对齐训练与推理机制（如 Chain‑of‑Thought）如何放大或抑制此类情感偏差，并提出可操作的去偏方法。

**🔧 技术方法**

研究使用了 API 调用、Prompt 变体、温度采样、情感量表解析以及多模型混合效应方差分析等技术手段来量化捐赠决策与情感评分。

**📊 数据集**

数据集由 51,955 条经验证的 API 试验组成，涵盖 16 个模型、10 个实验范式，原始刺激为改编自行为经济学与道德心理学的标准情景。

**📈 对比分析**

通过元分析与混合效应模型比较，发现 LLM 的 IVE 效应（d≈0.22）约是人类基线的两倍，标准 CoT 会将效应三倍放大，而 Utilitarian CoT 能有效消除偏差。

**⚠️ 局限性**

主要局限包括使用假设捐赠、模型版本快速迭代、单一英语刺激、以及情感测量工具对 LLM 内在状态解释的模糊性。

---

## 66. PR-MaGIC: Prompt Refinement Via Mask Decoder Gradient Flow For In-Context Segmentation

**arXiv ID:** 2604.12113 | [PDF](https://arxiv.org/pdf/2604.12113v1)

**作者:** Minjae Lee `[一作]` (Pohang University of Science and Technology), Won Hwa Kim `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 2468 | [OpenAlex ID](https://openalex.org/A5101424026)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种训练无关的测试时提示改进框架PR-MaGIC，通过mask decoder梯度流迭代细化Auto-prompting模型的提示，提升分割质量。

**💡 创新点**

创新点在于利用SAM的mask decoder梯度流在查询嵌入空间中迭代更新提示，并加入top-1支持–查询相似度掩码选择，避免了额外训练和参数。

**🔧 技术方法**

使用的技术包括mask decoder梯度流、KL熵正则化的梯度流、prompt resampling、支持–查询相似度匹配。

**📊 数据集**

使用的数据集包括语义分割的FSS-1000、LVIS-92、COCO-20i，以及部件分割的PASCAL-Part、PACO-Part、DIS5K。

**📈 对比分析**

与PerSAM-F和Matcher（1-shot/5-shot）对比，PR-MaGIC在语义分割上可提升约+8.8%至+2.2% mIoU，在部件分割上提升约+3.8%至+8.4% mIoU，表现出显著的性能提升。

**⚠️ 局限性**

局限性在于对步长和迭代次数的敏感性，接近性假设易失效；当支持–查询视觉语义差异大或支持图像模糊时，梯度流可能不收敛导致效果下降。

---

## 67. OpenTME: An Open Dataset of AI-powered H&E Tumor Microenvironment Profiles from TCGA

**arXiv ID:** 2604.12075 | [PDF](https://arxiv.org/pdf/2604.12075v1)

**作者:** Maaike Galama `[一作]` (Aignostics), Frederick Klauschen `[通讯]` (Charité – Universitätsmedizin Berlin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

使用Atlas H&E‑TME AI工具对3,634份TCGA H&E全景切片进行统一的高质量细胞级TME特征提取，生成超过4,500项量化读数，并公开构建了OpenTME数据集；

**💡 创新点**

首次提供多细胞类型、空间邻域分析的细胞级TME特征集，覆盖5种常见实体瘤，且预先计算好，降低了科研门槛；

**🔧 技术方法**

基于Vision Transformer基础模型的Atlas family，结合LoRA微调、StarDist核分割及多阶段QC、组织分割和细胞分类流程；

**📊 数据集**

来自TCGA的5种癌症（膀胱、乳腺、结肠直肠、肝胆、肺部）共3,634张诊断H&E切片；

**📈 对比分析**

与以往单细胞类型或图块级的资源（如Saltz的TIL图谱）相比，OpenTME提供更丰富的细胞种类、组织区分和空间邻域统计；已在多源扫描仪和多病理亚型上验证，覆盖率超过90%；

**⚠️ 局限性**

仅覆盖5种癌症，公开数据仅聚合读数，缺少细胞坐标和多边形几何；AI模型性能受限于训练数据，公开版本不适用于临床诊断或模型再训练。

---

## 68. Evaluating Lightweight Block Cipher Payload Encryption for Real-Time CAN Traffic

**arXiv ID:** 2604.11853 | [PDF](https://arxiv.org/pdf/2604.11853v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 69. Polyregular equivalence is undecidable in higher-order types

**arXiv ID:** 2604.11935 | [PDF](https://arxiv.org/pdf/2604.11935v1)

**作者:** Mikołaj Bojańczyk `[一作]` (University of Warsaw), Rafał Stefański `[通讯]` (IDEAS Research Institute)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文证明了多项式正则 λ-计算机在高阶类型下的等价性问题不可判定。

**💡 创新点**

创新点在于利用平铺问题的归约展示了即使在无递归、强制终止的功能语言中，等价性也可变成不可判定。

**🔧 技术方法**

主要采用了 λ-演算的类型系统、平铺问题归约以及多项式正则函数的构造技术。

**📊 数据集**

本文未使用实验数据集，所有论证均为形式化证明。

**📈 对比分析**

由于研究属于理论计算机科学，未进行实验比较，主要通过归约与对比说明不可判定性。

**⚠️ 局限性**

局限性在于只证明了高阶类型下的不可判定性，低阶（字符串↔字符串）等价性问题仍未解决。

---

## 70. Ultra-low-light computer vision using trained photon correlations

**arXiv ID:** 2604.11993 | [PDF](https://arxiv.org/pdf/2604.11993v1)

**作者:** Mandar M. Sohoni `[一作]` (Cornell University), Peter L. McMahon `[通讯]` (Cornell University)

**通讯引用:** 8089 | [OpenAlex ID](https://openalex.org/A5064735957)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

开发了一个端到端的光学-电子管道，将可训练的相干光源（SPDC双光子）与Transformer后端结合，用于极低光照下的物体识别；

**💡 创新点**

创新点在于引入相干光的可训练空间相关性，通过Correlation‑Aware Training（CAT）实现光源与神经网络的共同优化，显著降低光子预算并提升分类精度；

**🔧 技术方法**

技术手段包括：β‑BBO晶体产生的SPDC光、SLM调控泵浦角谱实现可编程相关性、EMCCD相机捕捉噪声帧、Set Transformer网络提取非局部相关性、物理感知训练（PAT）与直通估计（STE）实现反向传播；

**📊 数据集**

使用的数据集包括实验中的MPEG‑7子集（吸收式物体）和仿真中的细胞细胞器5类数据；

**📈 对比分析**

与未训练的相干/非相干照明以及传统的无相关光源对比，训练后在光子数≤200、拍摄≤10帧时准确率提升可达15个百分点，光子预算在实验中可降低至原来的1/6，仿真中可达1/10；

**⚠️ 局限性**

局限性包括：对光子损耗敏感、SPDC光源对平均光场的形状控制有限、CAT难以扩展到更高阶相关性、训练过程对硬件量化参数的敏感性等。

---

## 71. HintMR: Eliciting Stronger Mathematical Reasoning in Small Language Models

**arXiv ID:** 2604.12229 | [PDF](https://arxiv.org/pdf/2604.12229v1)

**作者:** Jawad Hossain `[一作]` (University at Albany), Chong Liu `[通讯]` (University at Albany)

**通讯引用:** 1617 | [OpenAlex ID](https://openalex.org/A5100412230)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并验证了一种 HintMR 框架，通过低容量提示生成模型与推理模型协同，利用分步提示提升小语言模型在数学推理任务上的表现。

**💡 创新点**

创新点在于：①将提示生成与推理拆分为两只专用 SLM，实现多模型协作；②使用 QLoRA 对提示生成器进行知识蒸馏，使其无需大模型即可产生高质量提示；③构建人类校正的 NuminaMath‑H 数据集，为提示生成与评估提供统一标准。

**🔧 技术方法**

主要技术包括：链式思维 (CoT)、量化低秩适配 (QLoRA)、提示辅助推理、对比 Self‑Consistency、Token 与计算效率评估。

**📊 数据集**

使用的数据集有 NuminaMath‑H、MATH‑500、AIME‑2024、AIME‑2025 以及原始 NuminaMath，用于训练提示生成、评估推理性能。

**📈 对比分析**

通过与无提示、LLM 提示、非微调 SLM 提示以及 SC（K=8）等基线对比，HintMR 在所有基准上平均提升 10–50% 的准确率，尤其在 AIME‑2024 上提升 48% 以上，且在计算与 Token 使用上优于 SC。

**⚠️ 局限性**

局限性包括：在极难的 AIME‑2025 数据集上仍然精度低；非微调提示效果不稳定；整体性能仍受限于 SLM 的推理容量；需要高质量提示生成作为前置条件。

---

## 72. EigenCoin: sassanid coins classification based on Bhattacharyya distance

**arXiv ID:** 2604.11932 | [PDF](https://arxiv.org/pdf/2604.11932v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 73. Latent patterns of urban mixing in mobility analysis across five global cities

**arXiv ID:** 2604.12202 | [PDF](https://arxiv.org/pdf/2604.12202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 74. Socially Fluent, Socially Awkward: Artificial Intelligence Relational Talk Backfires in Commercial Interactions

**arXiv ID:** 2604.12206 | [PDF](https://arxiv.org/pdf/2604.12206v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 75. Curvelet-Based Frequency-Aware Feature Enhancement for Deepfake Detection

**arXiv ID:** 2604.12028 | [PDF](https://arxiv.org/pdf/2604.12028v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 76. AutoSurrogate: An LLM-Driven Multi-Agent Framework for Autonomous Construction of Deep Learning Surrogate Models in Subsurface Flow

**arXiv ID:** 2604.11945 | [PDF](https://arxiv.org/pdf/2604.11945v1)

**作者:** Jiale Liu `[一作]` (University of Edinburgh), Nanzhe Wang `[通讯]` (Heriot-Watt University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 AutoSurrogate，一个基于大型语言模型的多智能体框架，实现无人工干预的地下流动深度学习代理模型自动构建与部署。

**💡 创新点**

创新点在于将物理知识与 LLM 推理相结合进行模型选择，闭环自我纠错，并通过共享内存实现跨代理可追溯决策。

**🔧 技术方法**

使用 LLM（如 GPT‑4）驱动的智能体、贝叶斯 HPO、卷积-Transformer、Fourier 神经算子等多种架构，并在训练中加入前向-后向稳定监测。

**📊 数据集**

使用基于 GEOS 的三维地质碳储存模拟数据集（80×80×20 渗透率场，31 个时间步的压强与 CO₂ 饱和度）。

**📈 对比分析**

通过与八种手工调参基线、三种 AutoML 方法以及 Pareto 前沿对比，AutoSurrogate 在压强预测 R²≈0.9976、饱和度 R²≈0.9532 的同时，搜索开销低于 AutoML 并位居性能前沿。

**⚠️ 局限性**

局限在于依赖本地 LLM 计算成本、对极端稀疏/高非线性场景的自我纠错仍需手动设定阈值，且在更大规模/多任务环境下的可扩展性待验证。

---

## 77. ArtifactWorld: Scaling 3D Gaussian Splatting Artifact Restoration via Video Generation Models

**arXiv ID:** 2604.12251 | [PDF](https://arxiv.org/pdf/2604.12251v1)

**作者:** Xinliang Wang `[一作]`, Zhenyu Wu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ArtifactWorld 框架，通过大规模数据扩展和同质双模型实现 3D Gaussian Splatting 的图像与几何修复。

**💡 创新点**

创新点在于细粒度破坏分类、107K 对齐视频数据飞轮、同质双模型+热图预测+Artifact-Aware Triplet Fusion 以及闭环回传到 3D 空间，显著提升空间时间一致性。

**🔧 技术方法**

使用视频流匹配扩散模型（LTX-Video-13B）+LoRA 同质预测、Decoupled Boundary Anchoring、Artifact-Aware Triplet Fusion、生成闭环重建等技术。

**📊 数据集**

基于 DL3DV、SpatialVID-HQ、Mip-NeRF-360 等原始序列，人工生成 16K pristine 场景，物理模拟 25.28K 对齐视频，扩展到 107K 训练对及 ArtifactWorld Benchmark（1.28K）。

**📈 对比分析**

在 DL3DV 与 Mip-NeRF-360 的稀疏视角评估中，与 Difix3D、GSFixer 等基线对比，PSNR/SSIM/LPIPS 均领先；在 2D 修复任务中，CLIP-I/FVD 等指标也显著优于现有方法。

**⚠️ 局限性**

受限于视频扩散模型的分辨率，难以实现 2K/4K 等高像素级精度，且显存占用较高。

---

## 78. PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving

**arXiv ID:** 2604.12171 | [PDF](https://arxiv.org/pdf/2604.12171v1)

**作者:** Xu Bai `[一作]` (University of Melbourne), Adel N. Toosi `[通讯]` (University of Melbourne)

**通讯引用:** 4966 | [OpenAlex ID](https://openalex.org/A5083902835)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 PipeLive 系统，实现 LLM 服务中动态管道并行配置的实时、原地重构。

**💡 创新点**

创新点在于 KV 缓存的块级动态重排与层堆叠、基于 VM 迁移的增量 KV 补丁以及无阻塞的 KV 迁移协议。

**🔧 技术方法**

采用 PageAttention 的非连续块访问、层堆叠、NCCL 双组通信、异步权重加载、KV 补丁等技术。

**📊 数据集**

在包含 Llama‑3‑70B、Qwen3‑30B 两大模型，使用 NVIDIA A100 与 L40S 两种异构 GPU 组合的测试平台进行评测。

**📈 对比分析**

与三种固定配置（prefill‑optimal、decode‑optimal、balanced）以及不做 KV 重新分配或不启用 KV 补丁的对比实验，PipeLive 在 TTFT、TPOT、吞吐率上分别提升约 30–45%，重构停顿从秒级降至约 10 ms。

**⚠️ 局限性**

局限在于仅实现了单机多 GPU 的 PP 重构，未给出自适应配置选择算法，且对超大模型或多租户环境的进一步扩展仍需研究。

---

## 79. Leveraging Weighted Syntactic and Semantic Context Assessment Summary (wSSAS) Towards Text Categorization Using LLMs

**arXiv ID:** 2604.12049 | [PDF](https://arxiv.org/pdf/2604.12049v1)

**作者:** Shreeya Verma Kathuria `[一作]` (Tellagence Inc), Charles Weber `[通讯]` (Villanova University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可确定性框架wSSAS，用于在LLM文本分类中通过分层结构（主题-故事-聚类）和信噪比权重来消除背景噪声并提升聚类与分类精度。

**💡 创新点**

创新点在于将信息瓶颈原理与多层次权重（SNR）结合，构建可解释、可重复的上下文摘要；引入Summary‑of‑Summaries（SoS）架构，将高信噪比信息聚合为输入提示；并将上下文质量评估与聚类质量指标直接关联。

**🔧 技术方法**

使用了大语言模型Gemini 2.0 Flash Lite，结合自定义两阶段验证流程、QAG与G‑Eval无参考评估、Silhouette、Davies‑Bouldin、Calinski‑Harabasz三种聚类内部验证指标。

**📊 数据集**

实验数据集包括Amazon产品评论、Google商家评论、Goodreads书评三大行业标准数据集。

**📈 对比分析**

通过三种场景（无上下文、SSAS无加权、wSSAS加权）对比；结果显示加权上下文在所有三种聚类指标上均优于无上下文和无加权；在QAG与G‑Eval上加权方案显著提升事实一致性、连贯性、流畅度与相关性。

**⚠️ 局限性**

局限性：方法仍依赖手工设计的SNR维度和权重公式，无法完全处理极其复杂的语言歧义（讽刺、对比等）；对超大规模长文本的实时推理仍存在计算瓶颈；需要进一步验证在更多行业域的泛化性。

---

## 80. INTARG: Informed Real-Time Adversarial Attack Generation for Time-Series Regression

**arXiv ID:** 2604.11928 | [PDF](https://arxiv.org/pdf/2604.11928v1)

**作者:** Gamze Kirman Tokgoz `[一作]` (San Diego State University), Baris Aksanli `[通讯]` (San Diego State University)

**通讯引用:** 1574 | [OpenAlex ID](https://openalex.org/A5029850946)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种在在线、有限缓冲区设置下的时间序列回归对抗攻击框架，采用信息化的、选择性攻击策略，只在模型自信且预测误差最大时对输入进行微小扰动。

**💡 创新点**

创新点包括：① 将预测区间宽度作为不确定性度量并实现自适应阈值；② 在有限缓冲区内只攻击少数高影响时刻，既保持了攻击效率又降低了可检测性；③ 通过实验验证该策略在保持相同攻击率的前提下，能显著提升预测误差并降低检测器 F1 分数。

**🔧 技术方法**

使用的技术主要有：一维卷积神经网络（CNN）做基础回归模型； conformalized quantile regression（CQR）校准预测区间；Fast Gradient Sign Method（FGSM）、Basic Iterative Method（BIM）和 Nesterov Iterative Fast Gradient Sign Method（NI-FGSM）三种梯度攻击；自适应分位数阈值机制；Local Intrinsic Dimensionality（LID）检测器。

**📊 数据集**

实验数据集为两个电力消耗时间序列数据集：UCI Individual Household Electric Power Consumption Dataset（≈207万分钟级）和 Pecan Street Dataport（25户住宅，每户约50万分钟级，79个特征）。

**📈 对比分析**

与传统全时步非选择性攻击基线（Baseline）相比，在相同攻击率下，该方法在 Household 数据集上最高提升 2.17× 的 RMSE，在 Pecan Street 数据集上最高提升 2.42× 的 RMSE；同时检测器 F1 分数下降，表明攻击更难被识别。

**⚠️ 局限性**

局限性：仅在单步预测场景下验证；攻击仅考虑白盒、测试时态；未探讨对抗训练或其他防御策略的影响；在不同预测模型或多步预测任务中的效果尚未评估。

---

## 81. The Linear Centroids Hypothesis: How Deep Network Features Represent Data

**arXiv ID:** 2604.11962 | [PDF](https://arxiv.org/pdf/2604.11962v1)

**作者:** Thomas Walker `[一作]` (Rice University), Richard Baraniuk `[通讯]` (Rice University)

**通讯引用:** 52241 | [OpenAlex ID](https://openalex.org/A5072713767)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出线性质心假设（LCH），用网络在输入空间的局部专家质心代替传统激活来进行特征和电路的解释性分析。

**💡 创新点**

创新点在于把网络的功能划分视为一组质心的线性方向，解决了LRH在识别假特征、跨子组件缺乏一致性以及与计算图脱节的局限。

**🔧 技术方法**

采用梯度向量乘积计算质心、稀疏自编码器、线性探针、归因度量和局部质心可视化等技术，替代传统激活作为解释工具。

**📊 数据集**

使用的公开数据集包括ImageNet、DTD、FashionMNIST、Imagenette、LikeLY和多种视觉/语言模型（如DINOv2/3、GPT‑2、Llama‑3.1‑8B）来验证假设和方法。

**📈 对比分析**

实验表明，基于质心的特征字典更稀疏、下游任务性能更好、探针泛化更强，归因度量能够迅速过滤无关神经元；整体性能优于基于激活的LRH方法。

**⚠️ 局限性**

局限性包括只关注质心而忽略了功率图的半径信息、对非可微组件无法直接应用、需要在更多模型与数据分布上进一步验证其解释力。

---

## 82. Policy-Invisible Violations in LLM-Based Agents

**arXiv ID:** 2604.12177 | [PDF](https://arxiv.org/pdf/2604.12177v1)

**作者:** Jie Wu `[一作]` (Atlassian), Ming Gong `[通讯]` (Atlassian)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文定义了“policy‑invisible violations”，并提出了基于世界模型的判定框架 Sentinel 以检测和阻止 LLM 代理在执行工具调用时隐藏状态导致的违规行为。

**💡 创新点**

创新点在于把违规判定从内容过滤提升到基于组织知识图的模型检查，并通过设计八类违例、构建 PhantomPolicy 基准与专属的七条图不变量，实现了可解释且高精度的违规检测。

**🔧 技术方法**

使用了图数据库（知识图）与 Counterfactual Graph Simulation、三值逻辑不变量检验，以及与 LLM 交互的工具调用回调，实现了在线或离线的动作验证。

**📊 数据集**

数据集为人工构造的 PhantomPolicy，包含 120 条案例（60 条违规、60 条安全控制），以及对应的 30+ 份组织知识图实体与属性。

**📈 对比分析**

对比方法包括无策略基线、prompt‑level policy 注入、内容‑DLP 规则以及 Sentinel；Sentinel 在人类审核标签下达到 92.99% 准确率、92.71 F1，显著优于其他方法，但仍有 37 条漏检违规。

**⚠️ 局限性**

局限在于需完整、实时的组织世界模型；缺失属性或覆盖不全会导致召回下降；且对多轮会话积累的 taint 跟踪及文本泄露仍有改进空间。

---

## 83. Divergence-Guided Particle Swarm Optimization

**arXiv ID:** 2604.12001 | [PDF](https://arxiv.org/pdf/2604.12001v1)

**作者:** Kleyton da Costa `[一作]` (Pontifical Catholic University of Rio de Janeiro), Hélio Lopes `[通讯]` (Pontifical Catholic University of Rio de Janeiro)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于散度的粒子群优化算法DPSO，在速度更新中加入了基于个人最优与全局最优相似度的排斥项，防止群体过早聚集于局部最优；

**💡 创新点**

创新点在于将高斯相似核与KL散度关联，形成可控的排斥力，提供了一个基于f‑散度的理论框架；

**🔧 技术方法**

采用标准PSO框架，并在此基础上添加相似度核与排斥方向的计算，保持了O(Nn)的复杂度；

**📊 数据集**

使用36个经典基准函数（15个单峰、21个多峰）在10、30、50维度上进行评测；

**📈 对比分析**

与标准PSO进行30次独立实验对比，DPSO在大多数多峰问题上提升2–8倍、标准差降低5倍，计算时间仅增加15–25%；

**⚠️ 局限性**

在单峰或简单多峰场景下排斥项反而降低性能，且仅通过单一超参数c₃控制，缺乏自适应机制

---

## 84. ResBM: Residual Bottleneck Models for Low-Bandwidth Pipeline Parallelism

**arXiv ID:** 2604.11947 | [PDF](https://arxiv.org/pdf/2604.11947v1)

**作者:** Alan Aboudib `[一作]` (Macrocosmos AI), Steffen Cruz `[通讯]` (Macrocosmos AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种残差瓶颈模型（ResBM），在低带宽分布式管道并行训练中实现激活压缩，仍保持端到端可训练；

**💡 创新点**

创新在于从架构层面设计可学习的压缩瓶颈，并通过残差路径保留低秩标识路径，能在不牺牲收敛速度的情况下实现高达128×的激活压缩；

**🔧 技术方法**

采用可学习的encoder‑decoder瓶颈层、残差投影矩阵、Muon优化器（保持高秩表示）与标准AdamW进行比较；

**📊 数据集**

使用C4数据集进行2B参数Transformer（类似Llama‑3）预训练；

**📈 对比分析**

与无压缩基线和Subspace Models（SM）比较，ResBM在100×/128×压缩下在26B标记训练后与基线持平或略优，并在80Mbps链路下恢复近乎中心化的吞吐量；

**⚠️ 局限性**

仅在2B规模验证，未验证更大模型或多种架构；仍需在更广泛的下游任务和不同优化器下进一步评估。

---

## 85. UniRec: Bridging the Expressive Gap between Generative and Discriminative Recommendation via Chain-of-Attribute

**arXiv ID:** 2604.12234 | [PDF](https://arxiv.org/pdf/2604.12234v1)

**作者:** Ziliang Wang `[一作]`, Weijie Bian `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

在单模型框架下，提出UniRec统一检索与排序，弥合生成式与判别式推荐的表达差距；

**💡 创新点**

核心创新点包括：通过贝叶斯理论揭示表达差距来源于特征覆盖；Chain-of-Attribute（CoA）在解码前先生成类别/品牌等属性；容量受限SID与条件解码上下文（CDC）抑制码本集中化与多场景冲突；联合RFT+DPO实现业务目标对齐；

**🔧 技术方法**

采用残差量化+曝光加权的容量约束SID，链式属性前缀，Task-Conditioned BOS，哈希加速的内容摘要，decoder-only Transformer+cross‑attention，SENet+MaskNet层级Rank Head，RFT与DPO优化；

**📊 数据集**

在大型电商平台真实日志（约9天、数十亿条交互）上进行离线评估，使用主feed场景数据；

**📈 对比分析**

与SASRec、TIGER、OneRec‑V2等基线对比；离线指标HR@50/100/200提升+22.6%/18.2%/17.2%，在订单样本上进一步提升+15.5%；上线A/B实验提升PVCTR+5.37%、订单+4.76%、GMV+5.60%；

**⚠️ 局限性**

仍受限于需手工定义属性、码本容量与曝光平衡的复杂性；对极长尾物品可能需要更细粒度调参；属性预生成增加解码步骤，对极低延迟场景影响；理论分析假设完整特征覆盖，实际应用中仍需进一步验证。

---

## 86. Fall Risk and Gait Analysis in Community-Dwelling Older Adults using World-Spaced 3D Human Mesh Recovery

**arXiv ID:** 2604.11961 | [PDF](https://arxiv.org/pdf/2604.11961v1)

**作者:** Chitra Banarjee `[一作]` (University of Central Florida), Ladda Thiamwong `[通讯]` (University of Central Florida)

**通讯引用:** 780 | [OpenAlex ID](https://openalex.org/A5013046592)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用单摄像头视频记录老年人完成TUG测试，采用GVHMR 3D人体网格恢复技术提取步态参数，实现在社区环境中获取临床意义的步态指标。

**💡 创新点**

将地面对齐的3D人体网格恢复方法应用于单摄像头步态分析，首次提供绝对空间度量（如步幅）并克服传统2D或多摄像头无法获得空间参数的限制。

**🔧 技术方法**

使用Ground View Human Mesh Recovery（GVHMR）、SMPL‑X人体模型、Gaussian平滑与峰值检测算法提取步态特征，并通过线性混合效应模型进行统计关联分析。

**📊 数据集**

基于52名社区老年人完成的207段视频TUG记录和90段同时采集的XSENSOR鞋垫传感器数据，包含自评跌倒风险、恐惧跌倒与平衡评估。

**📈 对比分析**

通过Spearman相关检验比较视频步时与鞋垫步时，获得ρ=0.673且p<0.001，表明两者关联显著；LME模型显示自评跌倒风险和恐惧跌倒显著预测步长、步长变异及坐起时间，证明视频提取指标与传感器数据及临床指标高度一致。

**⚠️ 局限性**

受限于单摄像头视角导致转弯分割不够精准；视频步时系统性低估鞋垫测量；样本量相对有限且未进行未来跌倒预测的外部验证。

---

## 87. Identity as Attractor: Geometric Evidence for Persistent Agent Architecture in LLM Activation Space

**arXiv ID:** 2604.12016 | [PDF](https://arxiv.org/pdf/2604.12016v1)

**作者:** Vladimir Vasilenko `[一作]` `[通讯]` (Independent Researcher), Vladimir Vasilenko (Independent Researcher)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `8d10c613-917e-4880-9716-17789f50e119` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文使用 Llama 3.1 8B 与 Gemma 2 9B 通过提取隐藏状态并计算余弦距离，实验证明持久认知代理的身份文档在激活空间中形成吸引子几何。

**💡 创新点**

创新点在于首次将身份文档视为激活空间中的多维吸引子，并通过语义蒸馏、预印本阅读等方法揭示其层级结构与语义一致性。

**🔧 技术方法**

所用技术包括均值池化隐藏状态、层级余弦距离、Welch t 检验、Bootstrap 置信区间、Permutation 和 Mann-Whitney U 检验，以及 t‑SNE 可视化和向量注入的行为诱导实验。

**📊 数据集**

实验数据来自 YAR 身份文档、七个语义等价改写、七个结构匹配的对照代理文档、5 句语义蒸馏、30 个随机长度匹配片段以及一篇科学预印本和伪预印本。

**📈 对比分析**

通过在层 8、16、24 的平均隐藏状态空间中比较 A+B 与 C 的距离，发现内聚距显著小于外部距（p<10^-27，Cohen d>1.88），蒸馏文档与完整文档相比更接近但仍距较远，预印本比伪预印本更接近，并且向量注入可在最佳强度下提升约 67% 的行为评分。

**⚠️ 局限性**

局限包括样本量仅 7 条，实验仅覆盖两种模型族，结构混杂未完全排除，均值池化假设可能掩盖细节，行为评估仅基于关键词评分，且缺乏对更大模型或不同训练目标的验证。

---

## 88. ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems

**arXiv ID:** 2604.11943 | [PDF](https://arxiv.org/pdf/2604.11943v1)

**作者:** Daeyeon Son `[一作]` `[通讯]` (Independent Researcher), Daeyeon Son (Independent Researcher)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了ProbeLogits这一内核级LLM推理日志向量读取原语，用于快速安全治理和进程级KV缓存管理；

**💡 创新点**

将LLM日志向量作为OS抽象暴露，零学习参数的分类与上下文校准、KV缓存作为进程状态、以及基于内核的宪法治理，形成全新的安全治理范式；

**🔧 技术方法**

使用Rust编写的裸金属LLM推理引擎（AVX‑512/AVX2 SIMD、工作抢占调度）、ProbeLogits的单前向logits读取、softmax、logit entropy、语法约束掩码、KV缓存快照/恢复/派生、WASM sandbox和Blake3审计链；

**📊 数据集**

评估基于自定义260提示OS动作基准（9类）和ToxicChat 1,000条真实对话，使用Qwen2.5‑7B‑Instruct 4‑bit模型以及SmolLM2‑135M；

**📈 对比分析**

与传统的生成‑解析方式对比，ProbeLogits在260提示上实现97.3%准确率（F1=0.98），相较之下生成‑解析仅90.8%准确率；在7B模型上单前向耗时65 ms，吞吐15 token/s与llama.cpp持平，135M模型则实现1,666 token/s，较llama.cpp快1.39×；

**⚠️ 局限性**

存在的局限包括缺乏GPU加速、ProbeLogits在聊天场景下F1仅0.79–0.84低于Fine‑tuned Llama Guard 0.939、语法约束仅支持选择/布尔、基准集单一模型/人工标注、缺乏形式化安全验证、TLS证书校验未实现等。

---

## 89. ORBIT: Guided Agentic Orchestration for Autonomous C-to-Rust Transpilation

**arXiv ID:** 2604.12048 | [PDF](https://arxiv.org/pdf/2604.12048v1)

**作者:** Muhammad Farrukh `[一作]` (Stony Brook University), Michalis Polychronakis `[通讯]` (Stony Brook University)

**通讯引用:** 4868 | [OpenAlex ID](https://openalex.org/A5007101727)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 ORBIT 框架，实现基于 LLM 的自治 C‑to‑Rust 翻译，能够在 24 个 1000+ 行的项目上实现 100% 编译成功和 91.7% 测试通过率。

**💡 创新点**

创新点在于将依赖图排序、自动化接口生成、两层函数映射策略与多代理协调结合，并通过迭代验证消除 LLM 幻想，首次实现大规模安全迁移。

**🔧 技术方法**

采用 Tree‑sitter AST 解析、Kahn 算法排序、Agentic Iterative Scaffolding、函数映射代理、实现检查、编译器代理与安全重构代理，使用 Qwen3‑Coder‑480B 与 GPT‑5.2‑Codex 等 LLM。

**📊 数据集**

主要评估 CRUST‑Bench 的 24 个超过 1000 行的 C 项目，并在 DARPA TRACTOR 的 13 个最难案例上测试。

**📈 对比分析**

与 C2Rust、CRUST‑Bench 基线对比，ORBIT 在编译成功率从 58.3% 提升到 100%，测试通过率从 20.8% 提升到 91.7%，unsafe 代码率降至 0.06%–0.11%，在 TRACTOR 中约 70% 通过率，竞争于业界最佳系统。

**⚠️ 局限性**

局限在于依赖现有测试套件评估语义正确性；在复杂跨模块和全局状态时仍可能循环或遗漏；非确定性 LLM 结果需要严格重现；未覆盖多线程、GUI 等非确定性程序。

---

## 90. TriFit: Trimodal Fusion with Protein Dynamics for Mutation Fitness Prediction

**arXiv ID:** 2604.12026 | [PDF](https://arxiv.org/pdf/2604.12026v1)

**作者:** Seungik Cho `[一作]` (Rice University), Seungik Cho `[通讯]` (Rice University)

**通讯引用:** 62 | [OpenAlex ID](https://openalex.org/A5110492660)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `09944146-298c-433e-89df-37255de463d7` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

TriFit构建了一个多模态框架，融合序列、AlphaFold2结构与GNM动力学信息来预测单点突变的功能效应。

**💡 创新点**

创新点在于首次将蛋白质动力学作为第三模态引入，并采用Mixture-of-Experts自适应融合与多模态对比学习来提升预测性能。

**🔧 技术方法**

技术上使用ESM‑2掩码分数序列编码、AlphaFold2结构特征、GNM B‑因子/模态形状/相关矩阵等动力学特征，并通过MoE专家网络和InfoNCE对比损失进行训练。

**📊 数据集**

实验数据采用ProteinGym替换基准（217个DMS实验，共696,311个单点突变），并利用AlphaFold2提供的结构。

**📈 对比分析**

在ProteinGym测试集上与所有监督基线（如Kermut、ProteinNPT）和零射模型（ESM3）比较，TriFit实现AUROC 0.897，显著优于Kermut（0.864）和ESM3（0.769），且稳定性更好。

**⚠️ 局限性**

局限性包括：仍依赖冻结的预训练模态，未充分挖掘变异特异性信息；动力学特征计算成本较高；在更大多样化数据集上的泛化尚待验证，且缺乏可解释性机制。

---

## 91. BayMOTH: Bayesian optiMizatiOn with meTa-lookahead -- a simple approacH

**arXiv ID:** 2604.12005 | [PDF](https://arxiv.org/pdf/2604.12005v1)

**作者:** Rahman Ejaz `[一作]` (Laboratory for Laser Energetics), Riccardo Betti `[通讯]` (Laboratory for Laser Energetics)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了BayMOTH框架，结合meta‑Bayesian优化与两步lookahead，能够在源任务与目标任务相关性不确定时自动选择是否利用先验。

**💡 创新点**

首次将meta‑BO与lookahead BO融合，并通过归一化交叉相关自适应切换，提升在任务不匹配时的鲁棒性。

**🔧 技术方法**

基于高斯过程代理、EI与2‑OPT采集函数、Monte Carlo估计与NCC任务相似度判定，构成门控双分支采样策略。

**📊 数据集**

在HPO‑B与HPOBench的超参优化任务以及合成ICF实验任务上进行评估，使用多组源任务相关性设置。

**📈 对比分析**

与MetaBO、NAP、GPBO‑EI、2‑OPT、随机搜索等对比，BayMOTH在高、中、低相关性场景下均保持或超过竞争方法，尤其在低相关性时优于MetaBO。

**⚠️ 局限性**

计算开销较高，单步建议耗时可达一分钟；在高维或源任务稀缺情况下仍可能出现记忆化过拟合。

---

## 92. Sample Complexity of Autoregressive Reasoning: Chain-of-Thought vs. End-to-End

**arXiv ID:** 2604.12013 | [PDF](https://arxiv.org/pdf/2604.12013v1)

**作者:** Steve Hanneke `[一作]` (Purdue University), Shay Moran `[通讯]` (Technion and Google Research)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe`

**🎯 论文内容**

本文在 PAC 学习框架下研究自回归（autoregressive）语言模型的样本复杂性，比较了两种监督方式：链式思维（Chain‑of‑Thought，CoT）和端到端（End‑to‑End，e2e）。作者给出了 e2e 学习的样本复杂性随生成长度 T 的完整分类，并证明 CoT 监督能消除对 T 的依赖；还引入了新的树维度（autoregressive tree dimension）作为 e2e 学习子线性增长的充分条件。

**💡 创新点**

创新点包括：
- 完整描述了 e2e 学习样本复杂性随 T 的增长速率，展示了从常数到线性之间的所有“自然”速率均可出现；
- 证明 CoT 监督下样本复杂性完全不随 T 变化；
- 证明不存在单一维度能定量刻画所有子线性速率（即否定了可能的维度理论）；
- 引入树维度，提供比 Littlestone 维度更宽松的子线性增长充分条件；
- 对具有稳定样本压缩方案的基类给出改进的 O(d/ε) 端到端样本复杂性上界。

**🔧 技术方法**

主要技术手段有：PAC 学习理论、VC 维度、Littlestone 维度、样本压缩方案（特别是稳定压缩）、对数化简的对角化论证、组合树分析（Sauer–Shelah–Perles 计数）、多类学习的 Natarajan 维度与增长函数分析。

**📊 数据集**

研究完全是理论性的，没有使用真实数据集；所有结果均为数学证明与分析。

**📈 对比分析**

通过理论比较：CoT 监督下样本复杂性为 O((log(1/ε)+log(1/δ))/ε)，与生成长度 T 无关；而 e2e 监督下样本复杂性可随 T 线性增长，具体速率取决于基类，可从 Θ(1) 到 Θ(T) 任意；对具有稳定压缩的类（如线性分类器）可进一步将 e2e 上界压缩到 O(d/ε)。

**⚠️ 局限性**

局限性包括：
- 仅对有限 VC 维度、有限符号集和单一二值输出有效；
- 对无限符号集、连续输出或多输出场景的推广尚未给出；
- 仍未找到能统一刻画所有子线性速率的维度理论；
- 结果主要理论化，缺乏对实际大型语言模型的实证验证。

---

## 93. LLM-Based Automated Diagnosis Of Integration Test Failures At Google

**arXiv ID:** 2604.12108 | [PDF](https://arxiv.org/pdf/2604.12108v1)

**作者:** Celal Ziftci `[一作]` (Google), Livio Dalloro `[通讯]` (Google)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AutoDebug，一个利用大语言模型自动诊断集成测试失败的工具，自动分析日志并生成根因摘要

**💡 创新点**

首次将 LLM 应用于大型分布式系统的集成测试日志诊断，并与代码评审系统 Critique 集成，实现实时帮助

**🔧 技术方法**

使用 Gemini 2.5 Flash LLM，并进行精心的 prompt 设计；结合日志聚合、排序和后处理

**📊 数据集**

基于 Google 内部海量集成测试失败日志，包含 TotalEvalExamples 条真实失败案例以及 TotalTargets 条生产失败测试

**📈 对比分析**

在手工评估中实现 TotalEvalAccuracy 的准确率；在生产中部署后，Not‑helpful 率为 TotalFeedbacksNotUsefulPct，且在所有工具中排名 AutoDebugHelpfulnessRank

**⚠️ 局限性**

受限于日志完整性、LLM 只能提供诊断而非修复，且对不同日志格式的鲁棒性需要提升

---

## 94. From IOCs to Regex: Automating CTI Operationalization for SOC with LLMs

**arXiv ID:** 2604.12228 | [PDF](https://arxiv.org/pdf/2604.12228v1)

**作者:** Pei-Yu Tseng `[一作]` (Pennsylvania State University), Peng Liu `[通讯]` (Pennsylvania State University)

**通讯引用:** 469860 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于大型语言模型的系统 IOCRegex‑gen，自动将 CTI 报告中的 IOCs 转换为可部署的正则表达式。

**💡 创新点**

创新点包括：①使用知识增强的图数据库自动识别捕获组；②基于迭代推理与多阶段验证的正则生成流程，确保语法和语义正确。

**🔧 技术方法**

结合 GPT‑4o 等 LLM、Neo4j 图数据库、迭代调试器与正则评分机制，并通过多模型集成提取 IOCs。

**📊 数据集**

使用 3,156 份 MITRE ATT&CK 相关 CTI 报告提取 IOCs，并以 MITRE ATT&CK Evaluation 提供的 2,400+ 真实攻击场景字符串作为地面真值。

**📈 对比分析**

通过对比生成正则与地面真值的匹配率、误报率和复杂度评分，结果显示 99.1% 命中率、0.8% 误报率，且平均评分>3，优于直接 LLM 生成基线超过 30%。

**⚠️ 局限性**

局限性在于对自定义可执行文件与未在图数据库中记录的 PowerShell cmdlet 的识别不足，导致部分命令行 IOC 匹配率略低。

---

## 95. Multi-Head Residual-Gated DeepONet for Coherent Nonlinear Wave Dynamics

**arXiv ID:** 2604.11972 | [PDF](https://arxiv.org/pdf/2604.11972v1)

**作者:** Zhiwei Fan `[一作]` (Newcastle University), Daniel Coca `[通讯]` (Newcastle University)

**通讯引用:** 1953 | [OpenAlex ID](https://openalex.org/A5077347224)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种多头残差门控 DeepONet（MH‑RG）框架，用于在保留初始状态的高维表示的同时，通过低维物理描述符进行结构化调制，以提高对相干非线性波动力学的逼真预测

**💡 创新点**

创新点在于将初始状态的物理描述符作为残差门控信号，分别作用于分支输入、分支嵌入和支路嵌入，并引入低秩多头机制实现可扩展的条件表达，避免参数爆炸并提升模型泛化

**🔧 技术方法**

技术包括：DeepONet 基础结构、FiLM 对比、残差门控（pre‑branch、branch、trunk），低秩多头共享上采样矩阵、门控激活为 1+αtanh，标准化与 Adam + StepLR 训练、MSE 损失、梯度裁剪

**📊 数据集**

使用了两大数据集：一维聚焦非线性薛定谔方程（NLSE）轨迹（1000 训练 / 200 测试）与二维耗散 Gross‑Pitaevskii 位移振荡（200 训练 / 50 测试），每个轨迹通过分布式初始条件生成并采样空间时间点

**📈 对比分析**

与 Vanilla、Concat、FiLM、RG 等基线对比，报告参数量与全场均方误差（MSE）。MH‑RG 在 NLSE 上全场 MSE 约为 1.4×10⁻³（相对 Vanilla 低 5 倍），在 2D GP 位移振荡上 MSE 约为 3.0×10⁻³（比 FiLM 低 2 倍），同时在物理量（总能量、峰值、中心轨迹）上保持更好的保真度，收敛曲线显示更快更稳定的下降

**⚠️ 局限性**

局限在于需要可手动定义且足够描述初始状态的低维物理特征，适用于相干或弱多尺度系统；对于强湍流、随机激励或多时刻需要的系统，单次初始描述可能不足；此外多头机制虽然参数友好，但在极大 head 数量下提升趋于饱和

---

## 96. The Effect of Document Selection on Query-focused Text Analysis

**arXiv ID:** 2604.12099 | [PDF](https://arxiv.org/pdf/2604.12099v1)

**作者:** Sandesh S Rangreji `[一作]` (Johns Hopkins University), Anjalie Field `[通讯]` (Johns Hopkins University)

**通讯引用:** 4286 | [OpenAlex ID](https://openalex.org/A5022479813)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究系统评估了七种文档选择策略对四种文本分析方法（LDA、BERTopic、TopicGPT、HiCode）在两大数据集上的主题发现效果，提出数据选择是方法学决策而非单纯实用需求。

**💡 创新点**

创新点在于建立可复现的评估框架，将文档选择视为方法学变量，比较不同检索策略对主题相关性、多样性和覆盖度的影响，并给出实践建议。

**🔧 技术方法**

使用的技术包括关键词检索（BM25）、语义检索（SBERT）、混合检索（Hybrid Sum、RRF、Weighted）、交叉编码重排序、最大边际相关性（MMR）、查询扩展以及随机抽样，结合四种主题模型进行下游分析。

**📊 数据集**

采用的公开数据集为TREC‑COVID（171k篇生物医学论文）和Doctor‑Reviews（171k份医生评论），分别设计了15/11个探索性查询。

**📈 对比分析**

比较方法通过计算主题与查询的余弦相似度、主题多样性（平均两两距离）和不同策略间的覆盖度（topic‑topic匹配），实验表明混合检索（Hybrid Simple Sum）与SBERT在相关性和覆盖度上优于关键词检索和随机采样，MMR等高级方法提升不明显。

**⚠️ 局限性**

局限性包括仅在两类特定领域数据集和固定1000文档样本规模下评估，缺乏对不同样本大小、其他主题模型或检索技术的泛化验证，以及依赖自动化相似度度量而非人工标注的主题质量评估。

---

## 97. Aethon: A Reference-Based Replication Primitive for Constant-Time Instantiation of Stateful AI Agents

**arXiv ID:** 2604.12129 | [PDF](https://arxiv.org/pdf/2604.12129v1)

**作者:** Swanand Rao `[一作]` (Next Moca Global, Inc.), Priya Krishnan `[通讯]` (Next Moca Global, Inc.)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出并实现了一种基于引用的复制原语Aethon，用以在 AI 代理系统中实现近乎常数时间的实例化，并通过分层内存模型实现可扩展的多代理编排。

**💡 创新点**

核心创新在于将代理实例视为稳定定义、共享内存层和本地增量的组合视图，而非完整复制的对象；通过引用、分层、延迟解析与写时复制实现低成本实例化、显式谱系和可追溯性。

**🔧 技术方法**

采用了：
- 参考式复制与延迟解析的系统架构；
- 多层内存模型（组织层、家族层、用户层、任务层）与写时复制；
- 版本化定义与线性谱系追踪；
- 基于引用的隔离与权限控制框架。

**📊 数据集**

暂无公开实验或数据集，本文主要以理论分析与系统设计为主。

**📈 对比分析**

论文通过复杂度分析说明实例化成本可降到 O(1) 与继承结构无关，内存增长随实际分化而非实例数量；性能提升体现在高并发、动态多代理编排场景中能够显著减少启动延迟与资源占用，且保持可追踪性和可审计性。对比传统基于完整复制的代理系统，Aethon 在实例化时间与内存占用方面具备显著优势。

**⚠️ 局限性**

局限与未解决问题：
- 解析器实现复杂，可能导致执行时延；
- 层级内存生命周期与垃圾回收策略尚未完善；
- 开发者工具和可视化不足，难以直观理解引用与谱系；
- 对外部副作用的事务管理与可回滚性不完整；
- 在高度差异化或工具延迟占主导的工作负载中收益有限；
- 缺乏标准化接口，跨平台互操作性待探索。

---

## 98. Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering

**arXiv ID:** 2604.12210 | [PDF](https://arxiv.org/pdf/2604.12210v1)

**作者:** Weikang Zhang `[一作]`, See-Kiong Ng `[通讯]` (National University Of Singapore)

**通讯引用:** 5448 | [OpenAlex ID](https://openalex.org/A5090171111)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

构建了一种基于大型语言模型的可控模拟器，用以生成具有认知障碍的标准化病人（SP）对话。

**💡 创新点**

创新点：①使用对比式提示与响应生成的双重数据集提取领域特定的导向向量（SV），①通过随机标记（STM）以概率方式注入SV实现持续可调的严重程度控制，②消除传统比例系数的不稳定性，实现从轻度到重度的连续调节。

**🔧 技术方法**

技术：对比学习提取SV；随机令牌调制STM；LLM隐藏状态注入；自动化层选择与α搜索；评估采用LLM与人工评审的多维指标。

**📊 数据集**

数据集：合成对比数据（提示+响应对）用于SV提取；公开的MTSamples脱敏病历作为模拟病人背景；对话约600个病人配置（1健康+5认知障碍）共700轮交互。

**📈 对比分析**

对比基线：PATIENT-ψ、Roleplay-doh、Direct Prompt、Role Vectors。实验结果显示，本方法在真实性、培训价值、域一致性、不可扰动性以及严重程度可控性上平均提升约11–18%，在LLM和人类评审中均保持显著优势。

**⚠️ 局限性**

局限性：仅支持文本交互，未覆盖语音、视觉等多模态表现；主要验证基于Qwen-3模型，跨模型推广需进一步验证；只能单域模拟，多重并发认知缺陷及其交互尚未充分研究；评估场景局限于日常对话，缺乏标准化诊断流程和多场景测试。

---

## 99. Interpretable DNA Sequence Classification via Dynamic Feature Generation in Decision Trees

**arXiv ID:** 2604.12060 | [PDF](https://arxiv.org/pdf/2604.12060v1)

**作者:** Nicolas Huynh `[一作]` (University of Cambridge), Mihaela van der Schaar `[通讯]` (University of Cambridge)

**通讯引用:** 22829 | [OpenAlex ID](https://openalex.org/A5012339002)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于决策树的可解释DNA序列分类框架，能够在树构建过程中动态生成高层次、可解释的序列特征；

**💡 创新点**

创新点在于将大语言模型（LLM）作为自适应特征生成器，并通过自我反思的进化式优化在每个节点生成符合生物学先验且可解释的特征，从而兼顾模型可解释性与表达力；

**🔧 技术方法**

使用技术包括：大语言模型（如GPT‑4）生成自然语言描述和可执行代码；演化启发式反思优化；传统决策树结构与一热编码输入；以及基于 impurity 的分裂准则；

**📊 数据集**

实验数据集涵盖三类DNA序列分类任务：Pol II Pausing、Promoters、Enhancers；

**📈 对比分析**

与 CART、OC1、CART+2mer、逻辑回归、CNN、Transformer、XGBoost 等基线比较，DEFT 在浅层树深度下即可取得与黑盒模型相当甚至更高的准确率和 AUPRC，显示出在保持可解释性的同时实现了优异性能；

**⚠️ 局限性**

主要局限包括：每个节点需要LLM推理，计算成本相对较高；模型对训练数据的波动敏感，缺乏正式的收敛性和泛化理论保证。

---

## 100. MedConcept: Unsupervised Concept Discovery for Interpretability in Medical VLMs

**arXiv ID:** 2604.11868 | [PDF](https://arxiv.org/pdf/2604.11868v1)

**作者:** Md Rakibul Haque `[一作]` (University of Utah), Shireen Elhabian `[通讯]` (University of Utah)

**通讯引用:** 1571 | [OpenAlex ID](https://openalex.org/A5000258401)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

通过稀疏自编码器在预训练医学视觉‑语言模型的高维特征上进行无监督概念挖掘，并利用冻结的医学 LLM 对生成的概念进行语义验证，构建可解释的概念摘要；

**💡 创新点**

①在医学 VLM 中首次实现完全无监督的概念发现和命名；②提出基于 LLM 的定量语义一致性评估（Aligned/Unaligned/Uncertain）方法；③利用联合视觉‑语言嵌入空间进行概念对齐，形成可交互的概念摘要；

**🔧 技术方法**

稀疏自编码器（SAE）、Merlin 视觉‑语言预训练模型、UMLS+ChatGPT 构建概念词典、MedGemma 作为冻结的医学 LLM 进行概念验证；

**📊 数据集**

AbdomenAtlas 3.0（9,262 体积）与 MerlinPlus（25,494 体积）两个大规模腹部 CT 数据集；

**📈 对比分析**

通过对比概念预测与对应放射报告的语义一致性，计算 Aligned、Unaligned、Uncertain 分数；在 MerlinPlus 上最高 25 概念的 Aligned 分数约 0.30–0.35，Uncertain 低于 0.1；在 AbdomenAtlas 上 Aligned 较低 (<0.1) 而 Unaligned 高（>0.8）；分数对 LLM 解码温度稳健，体现概念层级的排名相关性；

**⚠️ 局限性**

评估受限于报告的完整性与覆盖度，未记录的影像发现会被错误标记为 Unaligned；词典覆盖有限，无法囊括所有解剖与病理；概念命名依赖 VLM 共享嵌入空间，可能继承视觉‑文本偏差；

---

## 101. Beyond Perception Errors: Semantic Fixation in Large Vision-Language Models

**arXiv ID:** 2604.12119 | [PDF](https://arxiv.org/pdf/2604.12119v1)

**作者:** Md Tanvirul Alam `[一作]` (Rochester Institute of Technology), Md Tanvirul Alam `[通讯]` (Rochester Institute of Technology)

**通讯引用:** 433 | [OpenAlex ID](https://openalex.org/A5045927406)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究构建了 VLM‑Fix 这一人工合成基准，利用四种抽象策略游戏（井字棋、连连看、黑白棋、捉盒子）在相同视觉状态下切换标准规则与逆规则，系统评估大规模视觉‑语言模型在语义重映射上的表现；通过提示词替换、后训练与激活层调优等干预方法，探究模型在规则变更时的语义固化（semantic fixation）现象。

**💡 创新点**

创新点在于：①提出了“语义固化”这一概念并设计可控的视觉‑语言评测框架，能清晰区分感知失败与规则映射失败；②通过中性别名与语义加载别名两种提示策略，验证语义框架对模型偏差的可调性；③引入后训练与激活层调优三种技术，展示语义错误可在后期可编辑，并评估其在外部任务 VLMBias 上的迁移性。

**🔧 技术方法**

主要技术手段包括：结构化提示词（Base、Alias、SemAlias）与多种视觉渲染；监督微调（SFT）与基于可验证奖励的强化学习（RLVR）；基于捆绑激活的晚层调优（activation steering）；以及利用规则判定器对提示与目标进行路由。

**📊 数据集**

使用的数据集有：①VLM‑Fix（四款游戏，每款 300 终局状态，标准与逆规则两种判定，视觉渲染三种）；②VLMBias 计数任务（动物、标志、旗帜、棋盘等四个子集）；③针对后训练的合成腿计数数据（程序化生成的鸟类与四足动物的图标）。

**📈 对比分析**

通过对 14 种 VLM（开源与闭源）在标准与逆规则下的准确率进行对比，发现平均标准规则准确率 67.1%，逆规则 52.5%，差距 14.6%。中性别名提示将逆规则准确率提升至 63.08%，几乎消除差距；后训练在同规则下提升 5–10%，但在异规则下出现负迁移；晚层激活调优可在逆规则下恢复约 15–20% 的准确率。外部评测 VLMBias 上的图像/词汇 defamiliarization 也能提高准确率并降低偏差。

**⚠️ 局限性**

局限性包括：①VLM‑Fix 为人工合成环境，缺乏真实世界复杂性；②提示与激活调优的改进主要集中在后期层，对整体模型机制的解释不足；③跨任务迁移（从游戏到计数）效果有限，未能充分展示普适性；④实验主要覆盖固定模型与规模，未对更大或更不同架构进行全面验证。

---

## 102. Navigating the Complexity Landscape of Nominee Selection in Schulze Voting

**arXiv ID:** 2604.11933 | [PDF](https://arxiv.org/pdf/2604.11933v1)

**作者:** Katarína Cechlárová `[一作]` (Safárik University), Ildikó Schlotter `[通讯]` (ELTE)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355`

**🎯 论文内容**

研究了在Schulze投票规则下的Possible President和Necessary President问题的参数化复杂度，给出了选民数、党派最大规模和党派数量三种参数的完整二分图；

**💡 创新点**

通过细化参数化分析，首次提出了在两名选民时两问题可线性求解、三名及以上选民且党派规模≤2时两问题的NP/Σp‑完整性，并完整归纳了所有参数组合下的可判定性与不可判定性边界；

**🔧 技术方法**

主要利用图论构造（多路径、系列并行路径）与经典3CNF/Multicolored Clique归约相结合，并在可判定案例中采用动态规划与树形结构求解；

**📊 数据集**

本研究为理论性工作，未使用实际选举数据集；所有结果均基于构造实例，参考文献中Cechlárová等人使用的真实与合成数据未被实验验证；

**📈 对比分析**

通过构造实例证明NP-hardness，并在两选民情形给出线性时间算法，展示了在不同参数下的FPT与NP/Σp‑完整性；相较于已有工作，进一步细化了复杂度边界并提供了更完整的参数化分解；

**⚠️ 局限性**

局限在于仅关注Schulze规则，未覆盖其他规则；对更大党派规模或选民数的细致分析仍不足，且在大规模实际选举中的实现与性能评估仍需进一步研究。

---

## 103. Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision

**arXiv ID:** 2604.12002 | [PDF](https://arxiv.org/pdf/2604.12002v1)

**作者:** Yinghui He `[一作]` (Princeton University), Sanjeev Arora `[通讯]` (Princeton University)

**通讯引用:** 26409 | [OpenAlex ID](https://openalex.org/A5103209777)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于单一模型的自我修订训练（Self‑Revision Training，SRT）方法，模型同时扮演生成器与修订者角色，先利用自我修订产生标注后在自我分布式上进行自我蒸馏，将二元奖励转化为密集的 token‑级自监督，从而高效提升语言模型在可验证推理任务上的表现。

**💡 创新点**

创新点在于（1）通过单模型自我生成、修订并利用奖励条件化实现无外部教师、无高质量示例的自监督；（2）将修订行为通过 on‑policy 自我蒸馏注入生成器，形成 token‑级自定位与迭代自进化机制；（3）展示了修订器能聚焦错误 token，显著提升样本效率与推理准确率。

**🔧 技术方法**

技术核心包括：生成式自回归模型、条件修订（基于奖励的提示）、自我蒸馏（KL 对齐损失）、Token‑级 KL 评估、教师同步与迭代自演化；在大模型（Olmo、Qwen）上实现。

**📊 数据集**

主要使用了可验证推理任务的数据集：OpenR1‑Math（15k 问题）、Codeforces（7.5k C++ + 7.5k Python）、以及评测基准 AIME24/25、HMMT25、MATH、AMOBench、LiveCodeBench 等。

**📈 对比分析**

在相同样本预算下，与 SFT、RFT、GRPO、SDFT 等基线对比，SRT 在 Olmo 上平均提升 10.5%（单阶段 7.8%，蒸馏阶段 2.7%），在 Qwen 上提升 10.4%（单阶段 9.2%，蒸馏阶段 1.2%）。同时大幅减少生成 token 数量（≈ 2×），并在所有八个基准上实现至少 4.8% 的平均性能提升。

**⚠️ 局限性**

局限性包括：仅针对可验证领域（数学、代码），难以直接推广到无可验证奖励的任务；在“思考”型长链推理中难以区分探索与错误，导致自我蒸馏效果受限；以及对模型自身修订能力的依赖，需在每轮迭代中同步教师。

---

## 104. Active Imitation Learning for Thermal- and Kernel-Aware LFM Inference on 3D S-NUCA Many-Cores

**arXiv ID:** 2604.11948 | [PDF](https://arxiv.org/pdf/2604.11948v1)

**作者:** Yixian Shen `[一作]` (University of Amsterdam), Anuj Pathania `[通讯]` (University of Amsterdam)

**通讯引用:** 1083 | [OpenAlex ID](https://openalex.org/A5067055700)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于主动模仿学习（Active Imitation Learning）的调度框架，专门针对3D‑S‑NUCA多核CPU上大型基础模型（Large Foundation Model）推理的热管理和核级调度问题。

**💡 创新点**

创新点在于：① 将3D‑S‑NUCA的核心性能异质性与各 LFM kernel 的缓存/计算特征进行联合建模；② 通过 Mixture of Gaussian Process Regression（MoGPR）生成 Oracle 演示，实现近似最优热迁移决策；③ 将 Oracle 知识通过主动学习方式蒸馏为轻量级神经网络，并利用 MC Dropout 进行不确定性估计，动态决定是否查询 Oracle，从而实现低运行时开销。

**🔧 技术方法**

技术栈包括：Mixture of Gaussian Process Regression（MoGPR）作为 Oracle；主动模仿学习（Active Imitation Learning）+ MC Dropout 用于策略学习与不确定性驱动的查询；NAS‑搜索得到的三层 64/32/32 芯神经网络；特征工程聚焦 IPS、MPKI、AMD、功率预算等；以及 3D‑CoMeT 进行 RC‑热模拟。

**📊 数据集**

实验数据集覆盖多种 LFM：ViT‑base（ImageNet‑1K）、BERT‑base（SQuAD v1.1）、LLaMA3.2‑1B（WikiText‑103）、Gemma‑2B（C4）、DeepSeek‑2.4B（OpenWebText2），并在不同输入长度（L=128/256/512/1024）下评估。

**📈 对比分析**

与 3QTM、NeuroTAP、3D‑DNaPE、DLFM 等基线对比，所提框架在多种 LFM 与热阈值（75℃/85℃）下平均提升约 30% 推理性能，同时保持 5% 以下的运行时开销，并显著降低峰值温度；在更大 3D‑S‑NUCA 结构（4×4×4 → 6×6×6）上亦保持高扩展性。

**⚠️ 局限性**

局限性包括：① 需要对每种 LFM 进行一次离线 MoGPR 训练，训练成本与迁移性受限；② 主要针对已知的 3D‑S‑NUCA 配置，深度堆叠或不同 NoC 路由策略的推广尚未验证；③ MC Dropout 的阈值需要经验调优，可能影响在极端热或性能场景下的决策；④ 对极低/高内存带宽、非均匀访问模式变化的鲁棒性仍待进一步验证。

---

## 105. Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions

**arXiv ID:** 2604.11841 | [PDF](https://arxiv.org/pdf/2604.11841v1)

**作者:** Wenhao Zhang `[一作]` (Anhui University), Yiwen Zhang `[通讯]` (Anhui University)

**通讯引用:** 21560 | [OpenAlex ID](https://openalex.org/A5115603611)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PERA，一种在低秩因子空间内进行多项式扩展的参数高效微调方法。

**💡 创新点**

通过在低秩因子空间直接构造二阶平方与交叉项，使权重更新成为多项式空间，显著提升表达能力。

**🔧 技术方法**

使用多项式扩展（Poly^2）、Hadamard乘法、矩阵拼接实现高阶交互，并在LoRA框架下进行训练。

**📊 数据集**

在LLaMA2-7B、LLaMA3-8B、RoBERTa系列模型上评估，使用Commonsense170K、GLUE四大任务集。

**📈 对比分析**

与LoRA、HiRA、MoRA等PEFT方法对比，PERA在commonsense和GLUE任务上平均提升1–5%准确率，且在低秩和低数据量场景下保持竞争力。

**⚠️ 局限性**

主要限制在于仅在文本理解和推理任务中验证，未对算术推理、多模态生成等其他领域进行评估。

---

## 106. Ride the Wave: Precision-Allocated Sparse Attention for Smooth Video Generation

**arXiv ID:** 2604.12219 | [PDF](https://arxiv.org/pdf/2604.12219v1)

**作者:** Wentai Zhang `[一作]` (Beijing University of Posts and Telecommunications), Haihong E `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在视频扩散 Transformer 的稀疏注意力中，提出了 Precision-Allocated Sparse Attention (PASA) 框架，以提高推理速度并抑制时间闪烁。

**💡 创新点**

创新点包括：①曲率感知的动态预算机制，将稀疏预算在生成轨迹的关键时刻重新分配；②硬件友好的分组一阶 Taylor 近似，兼顾局部细节与吞吐率；③随机偏置路由，软化确定性分配边界，消除选择振荡和时间闪烁。

**🔧 技术方法**

技术手段：在 PISA 的基础上实现曲率驱动的 top‑k 调整、分组统计共享、随机噪声注入；使用 Triton 低级实现 GPU 内存合并与并行；并结合流匹配理论和 L1 速度加速指标。

**📊 数据集**

使用的数据集与基准：VBench（视觉质量与时间一致性评估）、Penguin Benchmark（Prompts）、以及 Wan 2.1‑T2V‑1.3B、Wan 2.1‑T2V‑14B、HunyuanVideo‑T2V‑13B 三大公开模型。

**📈 对比分析**

实验对比 Dense、SVG2、PISA 等稀疏注意力基线；在保持 85% 稀疏率的前提下，PASA 在三大模型上实现 1.4‑2.36× 的推理加速，并在 Temporal Flickering、Motion Smoothness、SSIM、PSNR、LPIPS 等指标上均达到或超过最优基线。

**⚠️ 局限性**

局限性：仍需手工校准曲率预算曲线，随机路由会引入额外噪声；分组近似依赖 GPU 内存对齐，可能在不同硬件上效果差异；对极其复杂或高帧率视频的适用性未完全验证。

---

## 107. Domain-Specific Latent Representations Improve the Fidelity of Diffusion-Based Medical Image Super-Resolution

**arXiv ID:** 2604.12152 | [PDF](https://arxiv.org/pdf/2604.12152v1)

**作者:** Sebastian Cajas `[一作]` (Massachusetts Institute of Technology), Leo Anthony Celi `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 36999 | [OpenAlex ID](https://openalex.org/A5031401755)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文研究了在医学图像超分辨率中，用领域特定的 VAE 替换通用 VAE 能显著提升重建质量。

**💡 创新点**

创新点在于将 VAE 作为超分辨率的瓶颈，提出无训练的“AE 复原上限”指标来预测 SR 效果，并通过小波子带分析定位优势。

**🔧 技术方法**

使用了潜在扩散模型（Latent Diffusion Model）与固定的 UNet 结构，比较 SD‑VAE 与 MedVAE 两种 VAE，评估 PSNR、SSIM、LPIPS、FID 等指标。

**📊 数据集**

实验数据集包括 MRNet（膝关节 MRI 细化任务）、BraTS 2023（大脑 MRI 4×超分）和 MIMIC‑CXR（胸 X‑ray 4×超分）。

**📈 对比分析**

对比方法包括传统双三次插值、ESRGAN、SwinIR 以及 SD‑VAE 超分；MedVAE 超分在所有数据集上平均提升 2.9–3.3 dB PSNR，效果显著，且 LPIPS 也有改善。

**⚠️ 局限性**

局限性包括仅测试 2D 图像、仅 4×放大比例、未覆盖最新的端到端扩散 SR 方法、对低场 MRI 的泛化尚未验证，以及对不同解码时长和指导策略的深入评估不足。

---

## 108. Memory as Metabolism: A Design for Companion Knowledge Systems

**arXiv ID:** 2604.12034 | [PDF](https://arxiv.org/pdf/2604.12034v1)

**作者:** Stefan Miteski `[一作]` `[通讯]` (CODE University), Stefan Miteski (CODE University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种针对单用户 LLM 伴随记忆系统的治理框架，旨在防止在持续的自我强化循环中出现知识“固化”（entrenchment）并维持可用性与多样性。

**💡 创新点**

核心创新点包括：
- 将记忆治理拆分为三层（交互/工作流、表示/检索、保留/治理）并聚焦在保留层；
- 引入“镜像 vs 补偿”（mirror‑vs‑compensate）原则，明确何时保持用户现有操作连续，何时主动纠正认知偏差；
- 设计五大操作（TRIAGE, DECAY, CONTEXTUALIZE, CONSOLIDATE, AUDIT）以及两种支持机制（记忆重力与少数假设保留）来实现该原则；
- 提出了可验证的保留约束、状态转移规则与完整的合规性规范，为实现提供明确准则；
- 提出了四个可测量的预测指标，强调治理目标而非检索准确性。

**🔧 技术方法**

使用的技术主要是：
- 图结构的重力计算（中心性+碎片化成本）用于判断条目的结构重要性；
- 批量化的睡眠式合并（CONSOLIDATE）与多周期少数支持累积机制；
- 递归/批处理与缓存技术（raw buffer、cold memory、active wiki）实现流式摄取与延迟合并；
- 版本控制与关系型元数据索引（如 Git + PostgreSQL）实现持久化与审计；
- 统一的调度与资源自适应机制（双调度器、家居状态调度）。

**📊 数据集**

本文并未在实验上使用任何公开数据集，而是以理论分析、设计规范和可验证的预测指标为主要贡献；因此不存在实际数据集。

**📈 对比分析**

对比方法：本文未提供实验对比；通过定义四个预测指标（连贯性稳定、脆弱性抵抗、单一化抵抗、有效少数影响）提供后续验证路径，预期若实现符合规范，系统应能在长期用户漂移下保持知识网络完整，避免主导观点的过度固化，并在多周期合并时让少数证据得以转化为主导观点。

**⚠️ 局限性**

局限性：
- 只针对单用户记忆，未解决多用户协作情景；
- 仍无法彻底消除“误信”或“确认偏误”带来的错误固化；
- 审计（AUDIT）的灵敏度和阈值设定为开放问题；
- 设计依赖于对重力、活力阈值等参数的手工调优；
- 仅提供治理规范，缺乏系统实现与真实负载下的性能评估；
- 该框架对模型更新的适配是基于架构分离的假设，若分离失效则可能无法正确响应基础模型的事实修正。

---

## 109. Bipedal-Walking-Dynamics Model on Granular Terrains

**arXiv ID:** 2604.11981 | [PDF](https://arxiv.org/pdf/2604.11981v1)

**作者:** Xunjie Chen `[一作]` (Rutgers University), Tao Liu `[通讯]` (Zhejiang University)

**通讯引用:** 26258 | [OpenAlex ID](https://openalex.org/A5100338067)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一个考虑足底侵入、滑动与滚动的双足步态动力学模型，并通过在沙地上实验验证其对关节角、地面反作用力（GRF）和能量消耗（CoT）的预测能力。

**💡 创新点**

创新点包括：①在双足动力学中加入了足底三自由度侵入变量，精确捕捉沙地下沉和滑移；②设计了闭式土壤阻力模型（结合Bekker/Resistive Force Theory），实现实时GRF估计；③通过模型与实验对比，展示了能量消耗的准确预测。

**🔧 技术方法**

采用的技术包括解析动力学建模、足底侵入/滑动动力学、闭式土壤阻力模型、运动捕捉系统与力传感器实验、模型预测控制（MPC‑WBC）、Pinocchio库进行仿真。

**📊 数据集**

使用的数据集为自制的自然沙地实验数据（包括机器人六关节角度、速度、力、能耗记录），并与公开的人类步行能耗数据做对比。

**📈 对比分析**

通过与传统无侵入、无滑动刚地模型的对比，利用RMSE评估关节角、GRF和侵入量，结果显示新模型在所有前进速度下误差显著降低；CoT预测与实验值接近，整体性能优于传统假设。

**⚠️ 局限性**

限制包括：仅建模平面直线步态，未考虑转向和3D耦合；适用于干燥颗粒沙，湿润或黏聚介质需额外改进；需要在线识别土壤参数以实现更广泛的场景适用性。

---

## 110. Socrates Loss: Unifying Confidence Calibration and Classification by Leveraging the Unknown

**arXiv ID:** 2604.12245 | [PDF](https://arxiv.org/pdf/2604.12245v1)

**作者:** Sandra Gómez-Gálvez `[一作]` (University of Auckland), Katerina Taškova `[通讯]` (University of Auckland)

**通讯引用:** 356 | [OpenAlex ID](https://openalex.org/A5039842961)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种新的统一损失函数 Socrates Loss，能够在单阶段训练中同时优化分类精度与置信度校准。

**💡 创新点**

创新点在于将“未知”类别与动态不确定性惩罚、自适应目标相结合，突破了传统单损失与两阶段训练在稳定性和性能上的权衡。

**🔧 技术方法**

使用焦点损失、动量自适应目标和动态不确定性惩罚的组合，并对其进行理论证明，证明其为 KL 散度的正则化上界并能正则化网络权重。

**📊 数据集**

实验数据集包括 SVHN、CIFAR‑10/CIFAR‑100、Food‑101，以及 ViT 在 CIFAR‑100 上的迁移学习实验。

**📈 对比分析**

与 Post‑hoc（TS、MS、VS）以及单损失和两阶段自适应校准方法比较，Socrates 在大多数数据/模型组合上在 ECE/Accuracy Pareto 边界上取得首位或次位，并更快收敛。

**⚠️ 局限性**

局限性包括在 ViT 上仍存在训练稳定性问题；对 OOD 或分布漂移的鲁棒性尚未验证，未知类的选择与超参数对结果的影响仍需进一步研究。

---

## 111. Beyond Factual Grounding: The Case for Opinion-Aware Retrieval-Augmented Generation

**arXiv ID:** 2604.12138 | [PDF](https://arxiv.org/pdf/2604.12138v1)

**作者:** Aditya Agrawal `[一作]` (Amazon.com), Harsha Aduri `[通讯]` (Amazon.com)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了面向意见的检索增强生成系统，加入意见提取和实体图增强索引，构建了支持多维度多样性检索的知识库；

**💡 创新点**

首次将事实与意见查询区分为不同的后验不确定性，提出覆盖、忠实与公平三目标框架，并在检索增强生成中引入意见属性实现多样性提升；

**🔧 技术方法**

使用LLM进行结构化意见提取、实体注册表与作者属性构建、情感/立场/影响等多维属性、混合检索及Wasserstein距离近似覆盖度；

**📊 数据集**

采用电子商务卖家论坛讨论数据（约8k条讨论）进行实验，并参考OpinioRAG等相关数据集；

**📈 对比分析**

与传统RAG基线对比，使用情感多样性、实体匹配率、覆盖度等指标，意见增强模型在情感多样性+26.8%、实体匹配率+42.7%，人工评估偏好率达79.2%；

**⚠️ 局限性**

存在与作者人口多样性权衡失衡问题，意见增强降低了部分群体覆盖率；未直接优化Wasserstein覆盖度，也未处理意见随时间动态变化的情况。

---

## 112. AlphaEval: Evaluating Agents in Production

**arXiv ID:** 2604.12162 | [PDF](https://arxiv.org/pdf/2604.12162v1)

**作者:** Pengrui Lu `[一作]` (SII), Pengfei Liu `[通讯]` (SII)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个基于真实生产任务的AlphaEval基准，涵盖94个任务，来源于七家公司，涉及六个O*NET职业领域。

**💡 创新点**

创新点在于提出需求到基准的标准化构建框架，并引入多范式评估与经济价值量化，揭示模型与架构在生产环境中的真实差距。

**🔧 技术方法**

采用多范式评估（参考答案验证、形式化验证、面板式评估、执行验证）以及LLM-as-a-Judge，结合Docker沙箱执行和模块化评估接口。

**📊 数据集**

使用了94个真实生产任务数据集，按O*NET分类，包含PDF、Excel、文本等多模态输入，人工标注的评估标准和人力成本估算。

**📈 对比分析**

通过对14个模型-框架组合的实验比较，最佳配置Claude Code+Opus 4.6仅获得64.41/100，且不同框架导致同一模型差异超过10分，域级差异显著。

**⚠️ 局限性**

局限包括域覆盖有限、评估仅为单时点快照、经济价值估算依赖专家校准、仅评估四种商业框架、未覆盖所有可能模型组合。

---

## 113. WiseOWL: A Methodology for Evaluating Ontological Descriptiveness and Semantic Correctness for Ontology Reuse and Ontology Recommendations

**arXiv ID:** 2604.12025 | [PDF](https://arxiv.org/pdf/2604.12025v1)

**作者:** Aryan Singh Dalal `[一作]` (Kansas State University), Hande Kucuk McGinty `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出并实现WiseOWL方法，用四个自动化指标（Well-Described、Well-Defined、Connection、Hierarchical Breadth）对本体进行量化评估；

**💡 创新点**

创新点在于将本体内部结构和文本语义结合，利用BERT嵌入评估标签与定义的相关性，构建统一的0–10分评分体系，并通过Streamlit实现可视化评估；

**🔧 技术方法**

技术包括RDF/OWL解析、BERT嵌入与余弦相似度、图遍历求深度/分支、log归一化、Python Streamlit与Plotly可视化；

**📊 数据集**

使用公开本体数据集：Plant Ontology、Gene Ontology、Semanticscience Integrated Ontology、Food Ontology、Dublin Core、GoodRelations；

**📈 对比分析**

通过对六个本体分别计算四项指标并取平均，结果显示PO和GO的平均分超过8.3，评估耗时约2分钟（Apple M3 Pro），证明WiseOWL在自动化评估上的可行性和效率；

**⚠️ 局限性**

局限在于阈值固定、缺乏领域特定语言模型、仅给出分数缺乏可操作性建议，未来需引入可配置阈值、域特定BERT或LLM以及设计模式推荐。

---

## 114. Subcritical Signal Propagation at Initialization in Normalization-Free Transformers

**arXiv ID:** 2604.11890 | [PDF](https://arxiv.org/pdf/2604.11890v1)

**作者:** Sergey Alekseev `[一作]` (Stony Brook University), Sergey Alekseev `[通讯]` (Stony Brook University)

**通讯引用:** 417 | [OpenAlex ID](https://openalex.org/A5050338386)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究了Transformer在初始化时的信号传播，利用平均部分雅可比范数（APJN）推导了层间协方差与APJN的递推关系，并对加入双向注意力和对称令牌输入的模型进行分析。

**💡 创新点**

将APJN框架推广到Transformer，证明LayerNorm与tanh‑like非线性替换导致的渐近APJN行为差异，并揭示后者呈拉伸指数增长，提出α参数调节对梯度放大和训练稳定性的影响。

**🔧 技术方法**

使用大宽度均值场近似、统一注意力近似以及协方差递推与雅可比递推推导，并在ViT中对实验验证。

**📊 数据集**

在Vision Transformer上使用CIFAR‑100数据集进行实验。

**📈 对比分析**

通过比较理论与测量的后向APJN曲线、MAPE误差以及训练早期准确率，结果显示预LayerNorm模型在不同深度与初始化下保持稳定，而DyT/Derf模型对深度、权重初始化和α更敏感，需要更细致的调参。

**⚠️ 局限性**

理论依赖对称令牌输入和均值场假设，对真实数据需要拟合初始条件；仅在ViT+ CIFAR‑100 上验证，未考虑更大规模模型、不同任务及训练过程中的参数演化；α与深度的最优比例未知。

---

## 115. Vectorized Gaussian Belief Propagation for Near Real-Time Fully-Distributed PMU-Based State Estimation

**arXiv ID:** 2604.12067 | [PDF](https://arxiv.org/pdf/2604.12067v1)

**作者:** Mirsad Cosovic `[一作]` (University of Sarajevo), Dejan Vukobratovic `[通讯]` (University of Novi Sad)

**通讯引用:** 2323 | [OpenAlex ID](https://openalex.org/A5035227760)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于向量化高斯置信传播（GBP）的PMU测量驱动的状态估计框架，包含多元和融合两种因子图形式，实现全分布式、近实时的电力系统状态估计。

**💡 创新点**

创新点在于：① 在因子图上直接使用多元高斯消息，减少环路数并保留完整测量信息；② 通过将同一状态变量组上的多条电流测量融合成单一因子，进一步压缩图结构并加速收敛；③ 提供了基于信息形式与广播形式的高效实现。

**🔧 技术方法**

使用技术包括：向量化高斯置信传播、因子图建模、测量协方差转换、消息的信息（canonical）与广播（broadcast）实现、矩阵分解与数值稳健性分析。

**📊 数据集**

使用数据集为IEEE 1354-机柜与1354、13659节点大规模测试系统，通过仿真生成的噪声PMU测量（电压幅值/角度噪声1e-8，电流噪声1e-6），并在最优PMU布置下随机增添PMU。

**📈 对比分析**

通过与集中式加权最小二乘（WLS）方法比较，使用RMSE和绝对误差指标。融合GBP在1~3次迭代内即可使RMSE低于10⁻⁴，误差分布集中在10⁻⁶–10⁻⁴之间，且在规模从1354到13659节点、测量冗余增加以及异步更新场景下保持快速收敛与高精度。

**⚠️ 局限性**

局限性包括：融合式GBP在初始迭代中矩阵逆可能产生数值不稳定；对PMU可观测线性模型的假设限制了适用范围；大规模系统虽可扩展，但消息尺寸和计算开销仍随网络规模增长；需要精确的协方差估计与同步机制以维持数值健壮性。

---

## 116. A Workflow to Efficiently Generate Dense Tissue Ground Truth Masks for Digital Breast Tomosynthesis

**arXiv ID:** 2604.11927 | [PDF](https://arxiv.org/pdf/2604.11927v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 117. A Foot Resistive Force Model for Legged Locomotion on Muddy Terrains

**arXiv ID:** 2604.12006 | [PDF](https://arxiv.org/pdf/2604.12006v1)

**作者:** Xunjie Chen `[一作]` (Rutgers University), Jingang Yi `[通讯]` (Rutgers University)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并验证了一种三维泥地阻力模型，并基于该模型设计了可变形足以提升在泥地上的移动效率。

**💡 创新点**

模型首次将黏弹性、螺纹性和吸力等泥土流变特性统一纳入三维解析，并给出闭式力公式；足部结构采用被动形变，显著降低吸力与能耗。

**🔧 技术方法**

采用阻力力学(RFT)改进、结构参数化泥土力学模型、闭式积分计算；实验平台为双足机器人YoboGo-10S；使用力/扭矩传感器与动捕系统测量。

**📊 数据集**

通过自制泥土样本（黏土、沙子、不同含水率15-35%）的平面板侵入与双足步态实验获得数据。

**📈 对比分析**

与三种固定形状足（平面、半圆柱、半球）对比，实验显示可变形足在吸力减少约57%，能耗降低约44%，并在力预测误差上与模型误差均低于10%。

**⚠️ 局限性**

模型仅适用于中等含水率（15-35%）泥土，未考虑高含水率液化或膨胀效应；未建模摆动阶段因泥负载导致的惯性影响，且缺乏户外复杂地形验证。

---

## 118. Network Slice Embedding over Space Division Multiplexed Elastic Optical Networks

**arXiv ID:** 2604.11936 | [PDF](https://arxiv.org/pdf/2604.11936v1)

**作者:** Divya Khanure `[一作]` (University of Texas at Dallas), Jason P. Jue `[通讯]` (University of Texas at Dallas)

**通讯引用:** 8535 | [OpenAlex ID](https://openalex.org/A5002482844)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了两种面向计算资源的光纤切片映射算法（DPSM和WMSM），通过在空间多路复用弹性光网络中协同分配光谱、核心和计算资源，实现多段路径切片的动态部署。

**💡 创新点**

创新点包括：① 通过引入计算节点间的“航点”（waypoint）将切片分解为多个短段，从而实现每段独立调制与光谱分配；② 采用平衡放置策略，使计算节点尽量位于路径中点，提升调制阶数；③ 在传统RMCSA框架中加入计算资源约束，实现计算–光谱协同优化；④ 探索节点融合（node co‑location）在光网络中的效果。

**🔧 技术方法**

使用的技术主要是启发式算法：基于Dijkstra的k‑shortest路径、核心组（core‑group）选择以避免相邻核心互调、OEO转换实现段内光谱连续性、计算资源分配与光谱分配的顺序耦合、平衡放置与反馈机制；同时利用仿真环境评估算法性能。

**📊 数据集**

数据集：采用公开的NSFNET 14节点拓扑（每条双向链路7个核心、每核心120个频段），生成Poisson到达流量，带宽均匀分布在1–20 Gbps，计算需求为5–10单位；模拟中还探讨了计算容量从4000单位降至400单位的情形。

**📈 对比分析**

与基线算法（直接路径驱动DPSM、排序与贪心基线）进行比较，指标包括阻塞率、光谱利用率和配置成本。实验结果表明，WMSM在高负载下阻塞率降低多达27%，配置成本降低多达47%，同时保持较高的光谱利用率；在计算受限时，WMSM的优势更为显著。

**⚠️ 局限性**

局限性：① 仅考虑最多两段航点，未探索更多段的潜在收益；② 采用简化的VNF计算模型（聚合需求而非细粒度VNF）和单一拓扑，缺乏跨拓扑验证；③ 未将碎片化显式纳入优化目标；④ 算法为启发式，无法保证全局最优；⑤ 仅在仿真环境下评估，缺少真实实验验证。

---

## 119. GRACE: A Dynamic Coreset Selection Framework for Large Language Model Optimization

**arXiv ID:** 2604.11810 | [PDF](https://arxiv.org/pdf/2604.11810v1)

**作者:** Tianhao Tang `[一作]` (Hong Kong University of Science and Technology), Lei Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 30069 | [OpenAlex ID](https://openalex.org/A5100333593)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

论文提出GRACE框架，动态选取LLM训练的coreset以提升训练效率与模型性能。

**💡 创新点**

创新点在于结合表示多样性与梯度重要性，使用k‑NN图自适应更新机制，避免频繁全量重算并通过贪婪子模优化实现近似最优。

**🔧 技术方法**

技术包括梯度匹配理论分析、贪婪子模子优化、Beta映射的重要性评分、k‑NN图构建与局部传播更新、以及LoRA参数高效微调。

**📊 数据集**

使用了MathInstruct、BioInstruct、DialogSum三大基准数据集进行实验。

**📈 对比分析**

与静态/动态传统方法（随机、MP、FL、DivIR、GradN、TAGCOS、kMQ等）比较，GRACE在10%训练预算下在Phi‑2、Llama‑2‑7B、Qwen2.5‑7B三模型上均获得最高平均准确率/ROUGE，并且整体训练时间低于完全动态方案。

**⚠️ 局限性**

局限性包括仍需先行warm‑up获取表示与梯度，对阈值δ、预算η、λ等超参数敏感；在极小模型或极大数据规模下的可扩展性尚待验证。

---

## 120. AgenticAI-DialogGen: Topic-Guided Conversation Generation for Fine-Tuning and Evaluating Short- and Long-Term Memories of LLMs

**arXiv ID:** 2604.12179 | [PDF](https://arxiv.org/pdf/2604.12179v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 121. Benchmarking Deflection and Hallucination in Large Vision-Language Models

**arXiv ID:** 2604.12033 | [PDF](https://arxiv.org/pdf/2604.12033v1)

**作者:** Nicholas Moratelli `[一作]` (University of Modena and Reggio Emilia), Gonzalo Iglesias `[通讯]` (Amazon AGI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究大规模视觉-语言模型在检索增强问答中的可靠性，提出动态数据筛选、评测基准和细粒度评估。

**💡 创新点**

创新点是动态过滤可参数化回答样本，构建2,775样本的检索增强评测基准，并定义四种检索场景的精细评估协议。

**🔧 技术方法**

使用门控模型如Gemma3、Qwen-2.5-VL等进行样本过滤，GPT-4o作为自动评判器，构建检索索引（Wikipedia、CLIP）进行负样本挖掘。

**📊 数据集**

采用六大KB-VQA来源（InfoSeek、WebQA、E-VQA、MMDocRAG、MRAG-Bench、ViQuAE）混合的多模态问答数据。

**📈 对比分析**

通过对20款现有LVLM的实验，结果表明大多数模型在存在噪声检索时倾向于幻觉，难以可靠拒绝，精度、拒绝率与幻觉率呈权衡。

**⚠️ 局限性**

限制包括依赖GPT-4o评判、原始数据集偏文本、未覆盖某些模型族、评测单次推理、仅适用于短答等。

---

## 122. Mathematics Teachers Interactions with a Multi-Agent System for Personalized Problem Generation

**arXiv ID:** 2604.12066 | [PDF](https://arxiv.org/pdf/2604.12066v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 123. Can AI Detect Life? Lessons from Artificial Life

**arXiv ID:** 2604.11915 | [PDF](https://arxiv.org/pdf/2604.11915v1)

**作者:** Ankit Gupta `[一作]` (Michigan State University), Christoph Adami `[通讯]` (Michigan State University)

**通讯引用:** 12879 | [OpenAlex ID](https://openalex.org/A5031224365)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

通过在人工生命平台Avida上构建可复制与非复制的9字符序列，对多层感知机进行训练并实施贪心hill‑climbing欺骗实验，证明即使模型在训练分布上几乎完美，也能被轻易诱导对非复制者产生接近100%置信度的错误预测，揭示AI在外部分布样本下的脆弱性。

**💡 创新点**

首次利用可全量控制的离散序列空间（Avida 9-mer）系统评估AI对生命检测的高置信度误报，展示了对自复制序列判别任务的欺骗攻击方法，并证明传统监督学习模型在该任务上存在严重的对抗性失败。

**🔧 技术方法**

使用字符嵌入的多层感知机（32维嵌入，512/256隐藏层，GELU激活，0.1 dropout）并通过AdamW优化；通过贪心hill‑climbing策略在序列空间中递增模型对复制者的置信度；实验平台为Avida数字生命系统。

**📊 数据集**

基于Avida可复制9-mer的完整空间（36,171条可复制序列）与随机抽样的非复制9-mer构成的平衡数据集（训练57,872，验证7,234，测试7,236条），以及全空间26^9的非复制序列用于负采样。

**📈 对比分析**

在标准分类任务上，模型在平衡测试集上取得99.97%准确率、100%召回率、99.94%精确率；但在欺骗实验中，仅用150-300次模型查询即可使置信度升至1.0，显示模型对外部分布样本的鲁棒性极差。

**⚠️ 局限性**

实验仅限于单一MLP架构，未检验更复杂深度网络；序列空间仅为9字符，规模有限，结果可能不具普适性；训练数据平衡采样可能导致对真实宇宙测量中的噪声与多维特征缺乏鲁棒性；高置信度误报可能严重影响太空生物探测任务的可信度。

---

## 124. Exploring Concept Subspace for Self-explainable Text-Attributed Graph Learning

**arXiv ID:** 2604.11986 | [PDF](https://arxiv.org/pdf/2604.11986v1)

**作者:** Xiaoxue Han `[一作]` (Stevens Institute of Technology), Yue Ning `[通讯]` (Stevens Institute of Technology)

**通讯引用:** 2084 | [OpenAlex ID](https://openalex.org/A5024383883)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Graph Concept Bottleneck 框架，将文本属性图映射到可解释的概念子空间，并通过概念激活直接做预测，实现自解释学习。

**💡 创新点**

创新点在于将图输入投射到自然语言概念空间，使用信息瓶颈筛选稀疏概念，并通过对比学习预训练图-概念对齐，兼顾内在可解释性与鲁棒性。

**🔧 技术方法**

采用对比概念–图预训练 (CCGP)、LLM检索概念、信息瓶颈优化、GCN/Graph Transformer 等图编码器，结合LLM生成概念进行对齐。

**📊 数据集**

在五个文本属性图数据集（如 CiteSeer、Amazon 电子商务网络等）上进行实验，预训练使用多源图数据。

**📈 对比分析**

与多种 SOTA GNN、MLP 及自解释 GNN（GIB、DIR‑GNN 等）对比，清洗、OOD 与扰动场景下性能相当或更优，尤其在 OOD 与扰动下表现更鲁棒，解释性更可信。

**⚠️ 局限性**

局限在于高度依赖 LLM 生成概念，概念集合需手工/过滤，且对非文本属性图的泛化尚待验证，概念选择仍可能出现噪声。

---

## 125. Thought-Retriever: Don't Just Retrieve Raw Data, Retrieve Thoughts for Memory-Augmented Agentic Systems

**arXiv ID:** 2604.12231 | [PDF](https://arxiv.org/pdf/2604.12231v1)

**作者:** Tao Feng `[一作]` (University of Illinois Urbana Champaign), Jiaxuan You `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Thought‑Retriever框架，利用LLM生成的中间“思考”（thought）作为长期可检索记忆，突破上下文长度限制，实现自我演化的检索增强LLM。

**💡 创新点**

创新点在于把LLM输出的思考转化为可验证、去重的知识单元，构建动态、层级化的思考记忆，并通过置信度与相似度过滤无效思考，形成自学习的长期记忆体系。

**🔧 技术方法**

核心技术包括检索增强（使用Contriever嵌入检索）、提示工程（生成答案、思考与置信度）、思考合并与更新策略、以及基于抽象层级的评估和过滤机制。

**📊 数据集**

使用了新建的AcademicEval基准（包含Abstract‑single、Abstract‑multi、Related‑multi）以及GovReport和WCEP公开数据集进行评估。

**📈 对比分析**

与多种检索增强基线（BM25、TF‑IDF、DPR、DRAGON、Qwen3‑Embed‑8b、IRCoT、RECOMP）、完整上下文、长上下文LLM（OpenOrca‑8k、NousHermes‑32k）对比，Thought‑Retriever平均提升F1约7.6%、赢率提升约16%，在所有任务上均稳稳领先。

**⚠️ 局限性**

局限性包括实验聚焦于AI领域论文，数据集主要为英文，缺乏多语言与跨学科验证；对极大规模实时部署的鲁棒性与可扩展性尚未充分测试。

---

## 126. When to Forget: A Memory Governance Primitive

**arXiv ID:** 2604.12007 | [PDF](https://arxiv.org/pdf/2604.12007v1)

**作者:** Baris Simsek `[一作]` `[通讯]`, Baris Simsek

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Memory Worth 记忆价值估计方法，利用每个记忆的检索成功率两计数器在线跟踪，形成动态记忆治理的原语。

**💡 创新点**

首次给出轻量级两计数器统计并证明其对每个记忆的条件成功概率 p⁺(m) 具备几乎必然收敛性，为记忆抑制、优先和废弃提供理论基础。

**🔧 技术方法**

使用马尔可夫子序列和强大数定理的 martingale 证明、Beta‑Bernoulli 贝叶斯后验、检索权重自适应与嵌入检索等技术。

**📊 数据集**

在合成实验（100 条随机真值记忆、均匀检索）、多任务混合实验以及使用 MiniLM 嵌入检索的文本检索微实验中进行验证。

**📈 对比分析**

与无反馈、均匀权重、相似度权重等方法对比，合成实验 Spearman ρ 达到 0.89±0.02；文本检索实验中成功识别 stale 记忆并保持高价值记忆稳定。

**⚠️ 局限性**

仅衡量检索-结果关联而非因果，受检索策略、任务难度、共检索混淆影响；需引入上下文分区、检索多样性和不确定性阈值，且在非平稳环境下收敛保证失效。

---

## 127. COBALT-TLA: A Neuro-Symbolic Verification Loop for Cross-Chain Bridge Vulnerability Discovery

**arXiv ID:** 2604.12172 | [PDF](https://arxiv.org/pdf/2604.12172v1)

**作者:** Dominik Blain `[一作]` `[通讯]` (QreativeLab), Dominik Blain (QreativeLab)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

构建了一个将大型语言模型与TLC模型检查器相结合的神经符号验证循环（COBALT‑TLA），用于自动生成受限的TLA+规范并验证跨链桥接协议的时序漏洞。

**💡 创新点**

核心创新是利用TLC的确定性语义反馈作为自我校正机制，消除了零射程LLM在TLA+生成时的幻觉，使得模型能在少量迭代内收敛到可验证的规范，并自动发现未提示的攻击（如Optimistic Relay Attack）。

**🔧 技术方法**

采用大型语言模型（如GPT‑4）进行规范生成，TLC作为模型检查器，构建REPL循环；同时设计了错误跟踪解析器将TLC输出转化为结构化反馈，使用系统提示强制变量取值有限，确保状态空间受限。

**📊 数据集**

实验基准包括三个跨链桥接目标：T1（Lock‑and‑Mint 的 Reorg Attack）、T2（同一架构的 Emergent Optimistic Relay Attack）、T3（Nomad 190 M USD 掠夺的 Zero‑Root Init 漏洞）。

**📈 对比分析**

在所有目标上，系统在≤2轮LLM迭代内即可获得验证结果，TLC执行时间恒定在0.26–0.30 s，整体端到端延迟受LLM推理约17–28 s主导，显示出高效、可扩展的性能。

**⚠️ 局限性**

局限性包括：只处理受限的状态空间（小常数边界）、仅在协议级别抽象（不涉及EVM字节码）、仅针对时序/并发漏洞、对提示工程和LLM随机性的依赖、以及未覆盖算术错误或重入攻击等其他常见安全问题。

---

## 128. MolMem: Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization

**arXiv ID:** 2604.12237 | [PDF](https://arxiv.org/pdf/2604.12237v1)

**作者:** Ziqing Wang `[一作]` (Northwestern University), Kaize Ding `[通讯]` (Northwestern University)

**通讯引用:** 2832 | [OpenAlex ID](https://openalex.org/A5044455276)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `86c0b5c7-57cf-4de0-90c2-eb64d5126a31` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种双内存、多轮代理式强化学习框架MolMem，用于在有限oracle调用预算下实现分子属性的迭代优化。

**💡 创新点**

创新点在于将静态示例内存与演进技能内存相结合，既实现冷启动的引导，又能够从成功轨迹中提炼可重用的策略，并通过多轮强化学习和稠密step奖励显著提升样本效率。

**🔧 技术方法**

采用Qwen2.5-1.5B作为策略模型，配合FAISS实现分子检索、PPO进行强化学习、GPT‑4o进行技能摘要，整体框架基于多轮MDP与稠密奖励设计。

**📊 数据集**

使用ChEMBL（约280万条分子）构建静态示例库，ZINC‑250k提供待优化的分子，Oracle为QED、plogP、SA、DRD2、JNK3等属性预测模型。

**📈 对比分析**

与单轮遗传算法、QMO、Reinvent4及多种专用LLM基线比较，在单属性任务上取得90%成功率，超过最佳基线1.5倍；在多属性任务上取得52%成功率，仅用500次oracle调用，显著优于其他方法。

**⚠️ 局限性**

局限在于仅依赖计算代理，缺乏实验室验证；技能总结依赖外部LLM，增加推理成本；在更严格的相似度约束下性能下降。

---

## 129. Accelerating Microswimmer Simulations via a Heterogeneous Pipelined Parallel-in-Time Framework

**arXiv ID:** 2604.12083 | [PDF](https://arxiv.org/pdf/2604.12083v1)

**作者:** Ruixiang Huang `[一作]` (Beijing Forestry University), Weifan Liu `[通讯]` (Beijing Forestry University)

**通讯引用:** 1212 | [OpenAlex ID](https://openalex.org/A5076872345)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `14d48e9d-0069-4ad9-996a-1d5968216998` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

开发了一种异构CPU–GPU并行框架，用于大规模、长时微泳器（纤毛细长体）在粘性流中的仿真，融合空间（Method of Regularized Stokeslets）和时间（Parareal）并行。

**💡 创新点**

创新点包括：1) 将Parareal改为流水线（pipelined）调度，显著减少GPU空闲时间；2) 提供针对3×3旋转矩阵的GPU友好平方根实现；3) 采用三层并行（GPU内核、MPI+GPU时间并行、CPU控制）和定制化内存布局，提升空间和时间并行的协同效率。

**🔧 技术方法**

使用的技术包括：Method of Regularized Stokeslets、Parareal算法的流水线实现、MPI通信、CUDA GPU kernel、共享内存与SIMT优化、闭式旋转矩阵平方根、二阶Runge–Kutta时间积分。

**📊 数据集**

实验数据集为单/多条微泳器（1、4、12、25条），每条由51个离散点组成，正则化参数ε=4Δs；使用的不是公开数据集，而是基于物理模型自行生成的仿真数据。

**📈 对比分析**

通过与CPU-only实现比较，GPU实现的速度提升达数百倍（最高约769×）。与常规Parareal比较，流水线调度在r小（r≈2–5）时可提升25–30%，并随着GPU数增大性能优势更显著。实验结果与理论GPU空闲时间分析一致，并展示了良好的弱/强规模性。

**⚠️ 局限性**

限制与挑战：1) 随着仿真时间或GPU数增大，GPU空闲/负载不均会导致资源浪费；2) coarse-to-fine 比例r的选择需在性能与收敛稳定性之间权衡；3) 现实现基于Python，限制了峰值性能；4) 对极 stiff 的系统收敛性仍需进一步研究。

---

## 130. Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score

**arXiv ID:** 2604.12196 | [PDF](https://arxiv.org/pdf/2604.12196v1)

**作者:** Manh Nguyen `[一作]` (Deakin University), Hung Le `[通讯]` (Deakin University)

**通讯引用:** 1633 | [OpenAlex ID](https://openalex.org/A5101936199)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于几何一致性的最佳答案挑选方法——Radial Consensus Score (RCS)，通过计算答案嵌入的加权 Fréchet 均值并以该中心的径向距离对候选答案进行排序；

**💡 创新点**

创新点在于将语义一致性建模为加权几何中心，并兼顾频率或生成概率等置信度信号，实现无监督、训练免费、黑盒兼容的答案聚合；

**🔧 技术方法**

主要技术包括句子嵌入（sentence transformer）、加权 Fréchet 均值计算、径向距离评分及不同权重分布（均匀、频率、概率）；

**📊 数据集**

使用七个公开基准：短答 SciQ、GPQA；数学推理 Arithmetics、GSM8K、AIME25；多选长答 MMLU Formal Logic、MMLU-Pro，实验涵盖 Qwen2.5-3B/7B、Llama3.2-3B/3.1-8B、Gemma2-9B 等五大开源模型；

**📈 对比分析**

与 NLL、ANLL、Self-Consistency、Self-Certainty 等基线对比，RCS 在多数模型/数据集上提升 2–7% 的选取准确率，且随着采样数 N 增大性能差距进一步拉大；在多代理辩论和黑盒环境下亦保持竞争力；

**⚠️ 局限性**

局限包括：RCS_freq 在高频答案偏向下表现略差；medoid 版本计算量高；依赖高质量嵌入模型，数值型答案对嵌入敏感；对极端多样化或噪声生成场景的鲁棒性仍有提升空间。

---

## 131. HTDC: Hesitation-Triggered Differential Calibration for Mitigating Hallucination in Large Vision-Language Models

**arXiv ID:** 2604.12115 | [PDF](https://arxiv.org/pdf/2604.12115v1)

**作者:** Xinyun Liu `[一作]` `[通讯]` (Sichuan University), Xinyun Liu (Sichuan University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了一种训练自由的解码框架 HTDC，通过在生成时动态检测层级犹豫并仅在必要时进行差分校准，以抑制大规模视觉语言模型的幻觉。

**💡 创新点**

创新点在于：①使用层级犹豫（intermediate‑layer fluctuation）作为自监督的内部信号，精准识别可能产生幻觉的生成步骤；②在触发时通过视觉消除与语义消除两种轻量级探针进行差分校准，兼顾视觉依据与语言先验；③引入自适应可行性约束 APC，避免无意义高分词的干扰；④显著降低计算开销，仅在约5% 的步骤激活探针，平均前向传播次数仅为1.107。

**🔧 技术方法**

核心技术包括：层级犹豫计算（基于logits差分与余弦距离）、动态门控触发（sigmoid阈值）、两种对比探针（视觉消除 V0 与语义消除 X0）、差分校准公式、APC 与门限控制。

**📊 数据集**

在四大基准上评估：MME（感知/推理）、POPE（物体幻觉检测）、CHAIR（开放式图像描述）以及 MM‑Vet（多模态综合能力）。

**📈 对比分析**

与 VCD、ICD、DoLa、DAMO 等主流无监督校准方法对比，HTDC 在 MME 感知得分提升至 1515.89（LLaVA）/1711.44（Qwen3‑VL），在 POPE 的 GQA 任务中获得最优准确率与 F1；在 CHAIR 上将 CHAIR_S 从 18.2 降至 11.6，CHI_I 由 6.2 降至 4.7，同时召回率仅略降；在 MM‑Vet 的总分上达到 30.7/70.3，避免了 VCD 造成的灾难性降级。计算上，HTDC 仅比普通解码增加 17% 的延迟，远低于 VCD/ICD 的 180% 以上。

**⚠️ 局限性**

局限性：①需要白盒访问模型内部隐藏状态，无法直接用于闭源或仅提供 API 的模型；②犹豫阈值和 APC 设定需在验证集上微调，跨模型可能略有差异；③当前仅在 token 级别操作，未扩展到多 token 或视频-语言任务；④对极端噪声或极度不确定的视觉输入的鲁棒性尚待验证。

---

## 132. Ternary Logic Encodings of Temporal Behavior Trees with Application to Control Synthesis

**arXiv ID:** 2604.12092 | [PDF](https://arxiv.org/pdf/2604.12092v1)

**作者:** Ryan Matheu `[一作]` (University of Maryland), Calin Belta `[通讯]` (University of Maryland)

**通讯引用:** 12032 | [OpenAlex ID](https://openalex.org/A5086742095)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出将行为树（BT）通过三值逻辑（K₃）重新表述，并给出其在Signal Temporal Logic（STL）上的混合整数线性编码，实现线性动力学系统的控制合成。

**💡 创新点**

创新点在于：① 证明BT逻辑自然映射到三值逻辑；② 设计了支持未知（Unknown）状态的STL语义；③ 提供了针对Sequence与Selector操作符的O((Δt)^{k-1})复杂度混合整数编码，支持完整的TBT规范；④ 将该方法应用于优化控制问题，证明可在MIQP框架下求解。

**🔧 技术方法**

使用的技术包括：三值逻辑（K₃）理论、Signal Temporal Logic (STL) 的三值语义、混合整数线性（MILP/MIQP）编程、Gurobi优化器、离散时间双积分器动力学模型。

**📊 数据集**

使用的实验数据集为：① 单机器人双积分器轨迹（两种初始电量场景）；② 三机器人多体双积分器协同规划（目标与障碍配置）。

**📈 对比分析**

方法通过在Gurobi中求解MIQP实现全局最优。单机器人案例求解时间约97.3秒，三机器人案例约307.7秒。论文未与其他时序逻辑或BT方法进行直接性能对比，但通过实例展示了在更丰富的TBT约束下仍能获得可行最优解。

**⚠️ 局限性**

限制包括：① 仅适用于线性时不变或时变系统；② 混合整数编码规模随公式子句数k呈指数增长，导致大规模TBT的求解难度高；③ 未针对非线性系统或更大规模多体系统验证；④ 对未知阈值δ的选择需手动设定，可能影响鲁棒性。

---

## 133. Spatial Atlas: Compute-Grounded Reasoning for Spatial-Aware Research Agent Benchmarks

**arXiv ID:** 2604.12102 | [PDF](https://arxiv.org/pdf/2604.12102v1)

**作者:** Arun Sharma `[一作]` `[通讯]` (University of Minnesota, Twin Cities), Arun Sharma (University of Minnesota, Twin Cities)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了计算驱动推理（CGR）的空间感知研究代理 Spatial Atlas，统一处理 FieldWorkArena 空间问答和 MLE‑Bench Kaggle 竞赛；

**💡 创新点**

核心创新包括：① 通过空间场景图引擎进行确定性空间计算消除 VLM 幻觉；② 熵导向推理实现成本效益的模型分层调度；③ 自愈 ML 管道和得分驱动迭代改进；④ 泄露审计注册表；

**🔧 技术方法**

结合 GPT‑4.1 视觉版、Florence‑2 目标检测、三层模型路由（OpenAI GPT‑4.1‑mini/standard + Anthropic Claude Opus）、LiteLLM、A2A 协议、熵推理、代码生成与沙箱执行等技术；

**📊 数据集**

使用 FieldWorkArena（工厂、仓库、零售环境多模态问答）和 MLE‑Bench（75 个 Kaggle 竞赛）数据集；

**📈 对比分析**

与纯 VLM 基线对比，场景图+熵提升 21–24% 准确率；MLE‑Bench 有 82% 有效提交率，约 35–40% 迭代提升；平均成本 0.18–1.85 美元/任务，时延 12–340 秒；

**⚠️ 局限性**

局限性包括：计算与模型串联导致约 12 秒延迟；场景图依赖视觉描述准确性；策略模板未覆盖所有竞赛；迭代改进收敛快；泄露审计仅捕捉四种常见泄露模式。

---

## 134. GitFarm: Git as a Service for Large-Scale Monorepos

**arXiv ID:** 2604.11977 | [PDF](https://arxiv.org/pdf/2604.11977v1)

**作者:** Preetam Dwivedi `[一作]` (Uber Technologies Inc.), Adam Bettigole `[通讯]` (Uber Technologies Inc.)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 GitFarm，一个将 Git 作为服务提供的远程执行平台，消除大型 monorepo 的本地克隆，显著降低客户端延迟和资源消耗。

**💡 创新点**

通过仓库中心化执行、持久化沙箱、身份范围授权和多命令会话，实现了状态化 Git 执行、低延迟、可扩展的远程 Git 服务，解决了传统 CI 缓存和本地克隆的瓶颈。

**🔧 技术方法**

gRPC API、预热的仓库池与沙箱池、Redis 跟踪资源、身份鉴权、分布式后端集群、容器化隔离等技术。

**📊 数据集**

Uber 内部的多种 monorepo（Go、Java、Python、Web、Android、iOS），以及约 900GB Go monorepo 的测试。

**📈 对比分析**

与 Buildkite/CI 环境对比，p50 执行延迟从 110–160 秒降低到 20–30 秒（>80% 降低），CPU、内存使用分别降低 82% 与 93%，沙箱获取 P95 延迟低于 1 秒。

**⚠️ 局限性**

当前仅支持完整仓库工作区、5 分钟短会话、无流式输出；缺乏稀疏/稀疏检出、长期会话、输出流式传输等功能。

---

## 135. INST-Align: Implicit Neural Alignment for Spatial Transcriptomics via Canonical Expression Fields

**arXiv ID:** 2604.12084 | [PDF](https://arxiv.org/pdf/2604.12084v1)

**作者:** Bonian Han `[一作]` (New Jersey Institute of Technology), Zhi Wei `[通讯]` (New Jersey Institute of Technology)

**通讯引用:** 20606 | [OpenAlex ID](https://openalex.org/A5001916237)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于共享典型表达场（Canonical Expression Field）的无监督配对空间转录组（ST）切片对齐与重建框架 INST‑Align。

**💡 创新点**

创新点在于将坐标变形网络与隐式神经表示（INR）相结合，利用共享的表达场在无监督条件下实现几何与表达的相互约束，解决了多切片非刚性变形和批次效应的耦合难题。

**🔧 技术方法**

采用隐式神经表示（ExprINR）+位置编码、双相训练（先预训练表达场再联合优化变形网络）、自适应软匹配损失、雅可比正则化等技术。

**📊 数据集**

在九个公开 ST 数据集上验证，包括 DLPFC、STARMap、MERFISH Brain 与 Hypothalamus、MouseEmbryo 等不同平台与分辨率的数据。

**📈 对比分析**

与 PASTE、STalign、Spateo、ICP 等基线相比，INST‑Align 在 OT Accuracy、NN Accuracy 和 Chamfer 距离上均达到或超过最高水平，尤其在大变形样本上 Chamfer 距离下降高达 94.9%。

**⚠️ 局限性**

局限性：仅处理成对切片，缺乏全局多切片一致性目标；生物学验证主要依赖聚类指标，嵌入质量仍低于专门的图网络方法；未来需要扩展到全 3D 对齐和更全面的生物学评估。

---

## 136. VISTA: Validation-Informed Trajectory Adaptation via Self-Distillation

**arXiv ID:** 2604.12044 | [PDF](https://arxiv.org/pdf/2604.12044v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 137. AdversarialCoT: Single-Document Retrieval Poisoning for LLM Reasoning

**arXiv ID:** 2604.12201 | [PDF](https://arxiv.org/pdf/2604.12201v1)

**作者:** Hongru Song `[一作]` (State Key Laboratory of AI Safety), Xueqi Cheng `[通讯]` (State Key Laboratory of AI Safety)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `6215c339-3735-4be3-8a07-5bbb7004712d` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究单文档知识库投毒攻击在检索增强推理中的影响，并提出AdversarialCoT方法。

**💡 创新点**

创新点在于利用反馈驱动的迭代优化，针对单文档构建链式思维式攻击，精准利用LLM的推理结构和漏洞。

**🔧 技术方法**

采用了检索增强生成（RAG）框架、链式思维（CoT）分析、黑盒决策式交互式优化等技术。

**📊 数据集**

使用了MS‑MARCO、HotpotQA、NQ等问答数据集进行实验。

**📈 对比分析**

与Naive、NPA、PHA、PRAG等基线对比，AdversarialCoT在所有模型上显著提升攻击成功率（ASR），迭代版比非迭代版提升约20‑30%。

**⚠️ 局限性**

局限性包括仅针对单一文档投毒，对多文档或动态检索场景适用性有限；迭代轮次越多，攻击成本上升；跨模型泛化效果仍有下降。

---

## 138. BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining

**arXiv ID:** 2604.12241 | [PDF](https://arxiv.org/pdf/2604.12241v1)

**作者:** Haojie Ye `[一作]` (University of Michigan), Nishil Talati `[通讯]` (University of Illinois Urbana-Champaign)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了一套可扩展的反洗钱系统，利用多阶段图挖掘与机器学习来识别金融交易中的洗钱模式。

**💡 创新点**

创新点在于引入统一的多阶段模式描述语言，支持模糊结构与时序；以及针对该语言的域特定编译器，可自动生成高性能 CPU/GPU 代码。

**🔧 技术方法**

技术上使用图模式匹配、基于图的特征提取、梯度提升树 XGBoost 作为下游分类器，并在编译器中实现了邻接列表并行、负载均衡、时间窗口裁剪等优化。

**📊 数据集**

使用 IBM 生成的合成洗钱交易数据集（LI/HI 小/中/大、以及 Trovares 规模数据）进行实验。

**📈 对比分析**

与现有 GFP 基线和 FraudGT 对比，功能相同的特征提取实现了 210 倍（CPU）/333 倍（GPU）的速度提升，保持相同 F1 分数；整体吞吐量比 FraudGT 高 4.9 倍。

**⚠️ 局限性**

局限在于仍需人工指定模式表达式，无法自动学习新型洗钱策略；对极为复杂或连续变化的模式支持有限；实验基于合成数据，缺乏真实交易验证。

---

## 139. Nucleus-Image: Sparse MoE for Image Generation

**arXiv ID:** 2604.12163 | [PDF](https://arxiv.org/pdf/2604.12163v1)

**作者:** Chandan Akiti `[一作]` (Nucleus AI Team), Haozhe Liu `[通讯]` (Nucleus AI Team)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Nucleus-Image，一种稀疏 MoE 变压器文本到图像生成模型，使用约 2 B 活跃参数即可生成高质量 1024×1024 图像。

**💡 创新点**

创新点包括 expert-choice 路由与时间步解耦的路由与计算、文本 KV 缓存、逐步稀疏化容量因子、Warmup‑Stable‑Merge 学习率调度，证明稀疏 MoE 可在保持高质量的同时显著降低计算与内存开销。

**🔧 技术方法**

使用的技术包括 MoE Diffusion Transformer、Qwen3‑VL 文本编码器、Qwen‑Image VAE、Muon's 优化器与参数分组、正交与 z‑loss 正则、定制 Triton 组合核、FSDP2 与专家并行、渐进分辨率与多长宽比例桶、以及多阶段训练日程。

**📊 数据集**

训练数据来自 1.5 B 的图像-标题对，经过严格过滤、去重、质量分层、标题多粒度标注，最终得到约 700 M 张独特图像与 1.5 B 标注对，涵盖多语言文本渲染、真实图像、数字插画等多种域。

**📈 对比分析**

通过 GenEval、DPG‑Bench 与 OneIG‑Bench 三大基准在 1024×1024 解析度下进行评测，模型整体得分 76.0（GenEval 0.87，DPG 88.79，OneIG 0.522），与同类最佳模型相当或领先，仅激活约 2 B 参数，显著提升了质量‑效率 Pareto 前沿。

**⚠️ 局限性**

局限性包括对极端高分辨率细节与背景一致性仍有提升空间；多样性（尤其是文本与风格多样性）相对不足；依赖大规模数据与算力，且尚未加入 RL/PPO 等偏好优化；在跨模态或长文本场景下的泛化仍需验证。

---

## 140. XANE(3): An E(3)-Equivariant Graph Neural Network for Accurate Prediction of XANES Spectra from Atomic Structures

**arXiv ID:** 2604.12140 | [PDF](https://arxiv.org/pdf/2604.12140v1)

**作者:** Vitor F. Grizzi `[一作]` (Argonne National Laboratory), Cong Liu `[通讯]` (Argonne National Laboratory)

**通讯引用:** 13227 | [OpenAlex ID](https://openalex.org/A5100331577)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发并评估了XANE(3)，一种基于E(3)等变图神经网络的模型，用于从铁氧化物的原子结构直接预测XANES光谱。

**💡 创新点**

创新点包括自适应门控残差、吸收器条件注意池化、基于多尺度高斯基的光谱读取，以及在训练目标中加入一阶和二阶导数匹配，以提升线形保真度。

**🔧 技术方法**

使用的技术包括E(3)等变图神经网络、张量积信息传递、球谐基、等变层归一化、可学习的高斯基谱展开、注意力池化与自适应门控残差。

**📊 数据集**

使用的数据集为5941个使用FDMNES计算的Fe K‑edgeXANES光谱，涵盖α‑Fe₂O₃、β‑Fe₂O₃、γ‑Fe₂O₃、Fe₃O₄和FeO的多面体与表面结构。

**📈 对比分析**

通过MSE、梯度和曲率损失进行评估，测试集MSE为1.0×10⁻³；消融实验表明包含导数损失、背景sigmoid、注意池化等组件可显著降低MSE及导数误差，验证了模型性能。

**⚠️ 局限性**

局限性在于对张量通道的依赖不强，标量模型可获得相似点误差但导数/曲率精度下降；此外模型仅在Fe K‑edge上验证，泛化到其他元素或更复杂系统仍待进一步研究。

---

## 141. Robust Reasoning and Learning with Brain-Inspired Representations under Hardware-Induced Nonlinearities

**arXiv ID:** 2604.12079 | [PDF](https://arxiv.org/pdf/2604.12079v1)

**作者:** William Youngwoo Chung `[一作]` (University of California, Irvine), Mohsen Imani `[通讯]` (University of California, Irvine)

**通讯引用:** 6886 | [OpenAlex ID](https://openalex.org/A5033221192)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出并实现了针对计算内存(CIM)硬件的硬件感知联合优化框架，提升超维计算(HDC)在非理想硬件环境下的鲁棒性与推理性能。

**💡 创新点**

通过将编码与相似度搜索联合优化，最小化硬件约束下的Frobenius范数，实现对非线性噪声的自适应校准，并解决图形HDC中绑定误差导致的关系丢失问题。

**🔧 技术方法**

采用超维计算(HDC)、FeFET/CIM模拟、非线性相似度建模、联合优化（任务损失+相似度损失+正则化）以及量化模型QuantHD等技术。

**📊 数据集**

实验使用ISOLET、FMNIST、Cora以及合成图数据集。

**📈 对比分析**

与传统HDC/QuantHD、无优化的RelHD/GrapHD等基线对比，实验显示在ISOLET上从37%提升至84%，FMNIST从36%提升至73%；RelHD在Cora上从约6%误差提升至95%准确率，GrapHD恢复精度恢复至接近100%。

**⚠️ 局限性**

仅针对已知硬件非理想特性建模，缺乏对更复杂推理任务的评估；框架依赖精确硬件模型，迁移到不同CIM平台可能存在困难。

---

## 142. Predictive Bayesian Arbitration: A Scalable Noisy-OR Model with Service Criticality Awareness

**arXiv ID:** 2604.11989 | [PDF](https://arxiv.org/pdf/2604.11989v1)

**作者:** Anil Jangam `[一作]` (Cisco Systems, Inc.), Roy Kantharajah `[通讯]` (Cisco Systems, Inc.)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种基于共享微服务架构的可预测Geo‑HA仲裁框架，旨在用单一仲裁服务管理多域多集群，显著降低资源消耗；

**💡 创新点**

核心创新在于利用自适应 Bayesian Noisy‑OR 模型，能够在线学习并自动发现故障级联依赖，实现主动切换决策；

**🔧 技术方法**

技术手段包括 Bayesian 网络、Noisy‑OR 组合、在线增量学习、Raft 共识、Kubernetes 监控与微服务化实现；

**📊 数据集**

实验使用在模拟环境下收集的时序监控日志和多场景故障数据集；

**📈 对比分析**

与传统心跳式仲裁、静态 Bayesian 及两种反应式方法对比，平均切换时间缩短 77.8%，MTTFD 降低 60%；

**⚠️ 局限性**

局限性主要是缺乏真实大规模部署验证，模型对极端故障模式和跨域迁移的鲁棒性待进一步评估

---

## 143. BLAST: Blockchain-based LLM-powered Agentic Spectrum Trading

**arXiv ID:** 2604.12127 | [PDF](https://arxiv.org/pdf/2604.12127v1)

**作者:** Anas Abognah `[一作]` (University of Waterloo), Otman Basir `[通讯]` (University of Waterloo)

**通讯引用:** 3996 | [OpenAlex ID](https://openalex.org/A5031048152)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669`

**🎯 论文内容**

提出 BLAST 框架，将基于 LLM 的智能代理与 Hyperledger Fabric 区块链结合，实现去中心化、隐私保护的频谱交易生态系统。

**💡 创新点**

创新点在于将大型语言模型代理的高级推理与区块链智能合约相融合，构建可信、自动化的封闭式竞价（Vickrey）以及其他市场机制，并通过实验验证其对社会福利和公平性的提升。

**🔧 技术方法**

核心技术包括 Gemini 2.5 Flash LLM（通过 Google Agent Development Kit）、Hyperledger Fabric 权限链、commit‑reveal 哈希机制、链码实现的多种拍卖协议，以及私有数据集合保证敏感信息安全。

**📊 数据集**

实验使用人工合成的数据集：4 名认知无线电代理（1 卖方 + 3 买方）、25 个频谱令牌（10 MHz/块）、100 次模拟时隙，未使用真实频谱交易或网络数据。

**📈 对比分析**

通过与非 LLM 经验启发式代理基线进行对比，评估指标包括社会福利、交易效率（相对 Shapley‑Value）、HHI 与 Gini 指标；结果显示 LLM 代理在 Vickrey 拍卖中实现更高福利和效率，整体交易量更大，但在公平性与集中度方面略逊于基线。

**⚠️ 局限性**

局限性包括：仅在仿真环境下验证，未在真实网络或大规模部署中测试；LLM 代理可能导致赢家通吃和收益集中；集成 LLM 与链码的复杂性与计算开销；缺乏对多频段、时隙和组合拍卖的实验验证。

---

## 144. MVAdapt: Zero-Shot Multi-Vehicle Adaptation for End-to-End Autonomous Driving

**arXiv ID:** 2604.11854 | [PDF](https://arxiv.org/pdf/2604.11854v1)

**作者:** Haesung Oh `[一作]` (Seoul National University), Jaeheung Park `[通讯]` (Seoul National University)

**通讯引用:** 3141 | [OpenAlex ID](https://openalex.org/A5031070386)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 MVAdapt，一种基于物理参数的跨车辆端到端自动驾驶适配框架，能够在不同车辆动力学特性下实现强大的零样本迁移与高效少样本校准。

**💡 创新点**

创新点在于将车辆物理属性通过小型物理编码器映射为潜在向量，并利用跨注意力机制将其与冻结的场景编码器融合，从而显式地将车辆动力学嵌入驾驶策略中，解决传统模型对车辆假设的局限。

**🔧 技术方法**

技术上使用了冻结的 TransFuser++ 视觉–雷达特征提取器、基于 MLP 的物理编码器、带多头注意力的融合模块以及 GRU 解码器生成轨迹，整体框架可在训练时仅调优轻量级模块。

**📊 数据集**

数据集主要来自 CARLA 仿真环境的 27 种训练车辆（约 100 万帧）以及 31 种未见车辆（包括随机采样的极端车辆），用于评估零样本与少样本适配性能。

**📈 对比分析**

与 Naive Transfer、URMA‑style、BodyTransformer‑style 等基线相比，MVAdapt 在分布内车辆的 Driving Score 约为 78（近似 TransFuser++ 80），在未见车辆的 Driving Score 达到 63（相较 28 的 Naive Transfer 有显著提升），且在极端车辆上通过仅 3% 数据的少样本微调即可提升至约 62 的 Driving Score。

**⚠️ 局限性**

局限性包括对极端物理差异车辆在零样本模式下仍表现不足，需要少样本校准；缺乏真实世界实验验证；并未处理硬件差异（如转向迟滞、控制器延迟）等更广泛的体现差异。

---

## 145. Complementarity by Construction: A Lie-Group Approach to Solving Quadratic Programs with Linear Complementarity Constraints

**arXiv ID:** 2604.11991 | [PDF](https://arxiv.org/pdf/2604.11991v1)

**作者:** Arun L. Bishop `[一作]` (Carnegie Mellon University), Zachary Manchester `[通讯]` (Massachusetts Institute of Technology)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了一种利用松弛互补约束的 Lie 群结构，在该结构上直接做优化的 LCQP 求解器 Marble。

**💡 创新点**

创新点在于：①使用 softplus 拉伸重参数化替代指数映射，解决指数映射的数值不稳定；②在构造约束时采用 Lie 群的乘法结构，使互补约束在设计时即被满足；③提出了基于增量 Lagrangian 的内外循环求解框架，并结合 QDLDL、filter line search 等技术实现高效求解。

**🔧 技术方法**

主要技术包括：Lie 群与 Lie 算子理论、softplus 拉伸重参数化、增量 Lagrangian（Augmented Lagrangian）求解、Newton 法、QDLDL 线性系统求解、滤波器线搜索、Ruiz 预缩放、以及外部松弛参数的几何调度。

**📊 数据集**

使用数据集：MacMPEC benchmark（39 个 LCQP 问题）以及三类机器人特定问题——平面蹦床接触问题、火箭捕捉轨迹问题和无人机门控推进问题。

**📈 对比分析**

与 LCQPow（SQP+qp-OASES）和 Gurobi（分支定界全局求解）比较：Marble 在 72% 的 benchmark 问题上最快，在所有机器人任务中均能求解且得到与 Gurobi 近似的全局最优；LCQPow 在部分问题失败或耗时显著更长。

**⚠️ 局限性**

局限性：松弛过程可能改变原始可行集，导致求解失败；对大规模问题的收敛性和鲁棒性尚需进一步研究；并且目前尚未系统分析不同松弛策略的失败模式。

---

## 146. GoodPoint: Learning Constructive Scientific Paper Feedback from Author Responses

**arXiv ID:** 2604.11924 | [PDF](https://arxiv.org/pdf/2604.11924v1)

**作者:** Jimin Mun `[一作]` (Carnegie Mellon University), Maarten Sap `[通讯]` (Carnegie Mellon University)

**通讯引用:** 6297 | [OpenAlex ID](https://openalex.org/A5015128745)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一种以作者反馈为中心的构造性论文评审生成框架；

**💡 创新点**

创新点在于将作者对反馈的认可（有效性与可操作性）作为训练与评估的监督信号，并通过双轴质量指标进行精细化衡量；

**🔧 技术方法**

采用了先SFT再DPO的训练策略，以Qwen3-8B为基础模型，并辅以LLM判别器进行质量过滤与偏好优化；

**📊 数据集**

使用了约19K篇ICLR论文及其作者-审稿讨论的手工标注数据集（ICLR 2020‑2026），构建了相应的ICLR 19K‑feedback数据集；

**📈 对比分析**

在自动评测中相较基线模型，成功率提升83.7%，F1提升58.8%；在人类评估中，在有效性、可操作性、具体性、帮助性等四个维度均显著优于开源与专有基线，几乎与大型专有模型相当；

**⚠️ 局限性**

局限在于作者回应仅为间接的质量信号，可能受时间、社区规范等因素影响，且数据仅覆盖ICLR领域，缺乏跨会议或跨学科的验证。

---

## 147. Dynamic Multi-Robot Task Allocation under Uncertainty and Communication Constraints: A Game-Theoretic Approach

**arXiv ID:** 2604.11954 | [PDF](https://arxiv.org/pdf/2604.11954v1)

**作者:** Maria G. Mendoza `[一作]` (University of California Berkeley), S. Shankar Sastry `[通讯]` (University of California Berkeley)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究在不确定完成时间、时窗约束和信息有限的动态多机器人任务分配问题，提出分布式的迭代最优响应（IBR）策略；

**💡 创新点**

在有限通信、局部感知环境下实现分布式决策，并通过信息群组数量化通信稀疏对性能的影响，证明IBR在多种通信拓扑下仍保持高效率；

**🔧 技术方法**

基于Hub感知区域与通信图的建模，利用游戏理论中的边际效用（marginal contribution）来制定本地决策；

**📊 数据集**

使用模拟的城市规模包裹配送场景（North San Francisco），生成任务数可变、时窗可调、无人机数从15到100等不同规模；

**📈 对比分析**

与EDD、Hungarian、SCoBA等基线对比，结果显示IBR在任务完成率上与中心化方法相近，同时计算时间显著低；在通信稀疏时效率比保持在0.86–0.98之间；

**⚠️ 局限性**

局限在于假设无人机同质、单机任务、固定感知范围；缺乏正式的性能上界或价格均衡（price‑of‑anarchy）证明；

---

## 148. Narrative-Driven Paper-to-Slide Generation via ArcDeck

**arXiv ID:** 2604.11969 | [PDF](https://arxiv.org/pdf/2604.11969v1)

**作者:** Tarik Can Ozden `[一作]`, James Matthew Rehg `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 ArcDeck 框架，将论文转幻灯片视为结构化叙事重构任务，采用多代理循环优化。

**💡 创新点**

创新点：① 使用 RST 语篇树显式建模论文结构；② 引入全局承诺（Global Commitment）保持整体意图；③ 采用闭环批判–评判–修订的多代理叙事精炼循环。

**🔧 技术方法**

技术：大语言模型（LLM）进行 RST 解析与幻灯片生成；多代理架构（内容、结构、视觉等专用代理）；全局承诺模块与对话式迭代改进；Markdown、PDF 预处理工具。

**📊 数据集**

数据集：ArcBench——100 对高质量口头演示与原论文的配对数据，涵盖顶级 CV/ML 会议，公开开放。

**📈 对比分析**

比较方法：在 ArcBench 上与现有单步或章节级摘要+生成方法对比，实验表明 ArcDeck 在叙事连贯度、逻辑一致性和内容保留方面显著优于基线，生成的幻灯片更符合专业演示标准。

**⚠️ 局限性**

局限性：① 仍依赖 LLM 的推理质量，可能受模型偏差和算力限制；② 对极长或结构极其复杂的论文处理仍有挑战；③ 需要高质量的原始 PDF 和视觉素材，格式不规范时效果下降。

---

## 149. Does Visual Token Pruning Improve Calibration? An Empirical Study on Confidence in MLLMs

**arXiv ID:** 2604.12035 | [PDF](https://arxiv.org/pdf/2604.12035v1)

**作者:** Kaizhen Tan `[一作]` `[通讯]` (Carnegie Mellon University), Kaizhen Tan (Carnegie Mellon University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究视觉令牌裁剪对多模态大型语言模型的校准影响。

**💡 创新点**

首次系统评估裁剪对模型置信度校准的影响，揭示覆盖式裁剪可提升校准而不损失准确率。

**🔧 技术方法**

采用SCOPE、saliency‑only、FastV与随机裁剪，并在LLaVA‑1.5‑7B上评估ECE、Brier、AURC等校准指标。

**📊 数据集**

使用POPE（二元存在问答）和ScienceQA‑IMG（多选科学问答）两个数据集。

**📈 对比分析**

与全量模型、不同α值及其他基线对比，发现纯覆盖α=0、K=128能将ECE降至0.016且准确率保持不变；saliency裁剪校准更差。

**⚠️ 局限性**

仅在单一模型、两数据集上实验，置信度定义受限，未探究与量化等其他压缩技术的联合效果。

---

## 150. ReefMapGS: Enabling Large-Scale Underwater Reconstruction by Closing the Loop Between Multimodal SLAM and Gaussian Splatting

**arXiv ID:** 2604.11992 | [PDF](https://arxiv.org/pdf/2604.11992v1)

**作者:** Daniel Yang `[一作]` (Massachusetts Institute of Technology), Yogesh Girdhar `[通讯]` (Woods Hole Oceanographic Institution)

**通讯引用:** 1013 | [OpenAlex ID](https://openalex.org/A5072400128)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

集成多模态位姿图SLAM与增量3D Gaussian Splatting闭环，完成大规模水下场景的实时重建与位姿优化。

**💡 创新点**

将3DGS差分渲染用于局部位姿微调，并将微调结果回馈至位姿图实现全局一致性；采用已知固定地标起始区、逐环扩展的增量策略，避免传统SfM/BA；使用单目深度估计替代点云初始化。

**🔧 技术方法**

多模态传感器融合（IMU、DVL、压力计）+EKF预积分+GTSAM位姿图优化；3D Gaussian Splatting模型（带不确定性、SSIM、TV等损失）+差分渲染(gsplat)实现位姿微调；DepthAnythingV2单目深度估计。

**📊 数据集**

美国维尔京群岛Tektite与Yawzi珊瑚礁现场数据，包含4K单目相机、DVL、IMU、压力计，沿固定玫瑰形轨迹采集。

**📈 对比分析**

与ORB‑SLAM3、DROID‑SLAM、MASt3R‑SLAM、VGGT‑SLAM、MonoGS、WildGS‑SLAM及多模态GTSAM基线对比；在Tektite与Yawzi场景下，ATE RMSE分别为0.135 m和0.229 m，重建质量（PSNR/SSIM/LPIPS）优于使用Oracle SfM批处理；相较于SfM耗时10+小时，方法仅需数小时甚至更短。

**⚠️ 局限性**

需已知固定地标，无法直接处理无地标环境；局部微调对视角重叠要求高，重叠不足时易失效；单目深度估计初始化误差有限；局部微调计算量大，实时性受限。

---

## 151. VERITAS: Verifiable Epistemic Reasoning for Image-Derived Hypothesis Testing via Agentic Systems

**arXiv ID:** 2604.12144 | [PDF](https://arxiv.org/pdf/2604.12144v1)

**作者:** Lucas Stoffl `[一作]` (Weill Cornell Medicine), Johannes C. Paetzold `[通讯]` (Weill Cornell Medicine)

**通讯引用:** 3704 | [OpenAlex ID](https://openalex.org/A5085027983)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种多代理系统，能够自动化并可追溯地在医学影像数据集上执行自然语言假设检验，生成可执行代码、分割掩模和统计结果，并给出可信的结论。

**💡 创新点**

创新点包括：① 通过角色化的代理（PI、影像专家、统计学家）将复杂流程拆分为四个可验证阶段；② 引入可机械化的“知识证据标签”框架（Supported、Refuted、Underpowered、Invalid），实现对统计显著性、效应方向和功效的客观评估；③ 将可视化分割、代码生成与统计推断整合为统一的可审计工作流。

**🔧 技术方法**

技术手段主要有：大语言模型（GPT‑OSS‑20B、Qwen‑3‑8B/30B、GPT‑5.2 等）进行文本推理和代码编写；SAT/MedSAM 等可提示分割模型用于医学影像分割；Python 沙箱执行生成的统计脚本；API 层封装数据访问；ECO（Evidence Classification Operator）根据 p 值、效应量和功效计算标签。

**📊 数据集**

使用了两个公开医学影像数据集：心脏 MRI（ACDC，150 例）和脑肿瘤 MRI（UCSF‑PDGM，501 例），并在这两组数据上构建了 64 个涵盖 6 个复杂度层级的假设基准。

**📈 对比分析**

与五个单模型基线（直接推理、代码+预计算特征、代码+API、迭代自校正、结构化流程）进行比较；在本地 8‑30B 模型下，系统在多数层级上实现 71.2% 的多数投票决策准确率，在前沿 GPT‑5.2 上提升至 81.4%，且在复杂的多变量生存/回归任务（L5 层）上唯一获得非零准确率；整体完成率、验证率和错误诊断率均优于基线。

**⚠️ 局限性**

局限性包括：对分割质量的依赖，未将分割不确定性纳入统计；基准仅涵盖两类 MRI 数据，未覆盖 CT、病理等模态或纵向/因果推断；实验所用的“真实”标签基于数据集内部计算，可能与更广泛人群不一致。

---

## 152. CycloneMAE: A Scalable Multi-Task Learning Model for Global Tropical Cyclone Probabilistic Forecasting

**arXiv ID:** 2604.12180 | [PDF](https://arxiv.org/pdf/2604.12180v1)

**作者:** Renlong Hang `[一作]` (Nanjing University of Information Science and Technology), Qingshan Liu `[通讯]` (Nanjing University of Posts and Telecommunications)

**通讯引用:** 13007 | [OpenAlex ID](https://openalex.org/A5100404959)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `5a41884c-404f-4688-a89c-aa238c10fe68` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 CycloneMAE，一种多任务学习框架，可在同一模型上实现台风概率预测（MSLP、MSW、轨迹）并输出确定值和置信分布；

**💡 创新点**

创新点在于构建了针对台风结构的径向掩码自编码器（TC‑MAE）和离散概率分箱技术，使得模型既能学习跨变量的共享特征，又能一次性给出多目标的概率输出；

**🔧 技术方法**

技术实现结合了双流 ViT‑MAE 预训练、Radial‑Masking、Gaussian 标签平滑交叉熵、Softmax 离散概率分箱和 LSTM 预测头，最终通过 IG 进行可解释性分析；

**📊 数据集**

使用了 5 个全球海域（WP、NA、EP、SI、SP）20 年（2000‑2019）多模态台风数据，训练集 2000‑2014，微调集 2015‑2019，测试集 2020‑2024；

**📈 对比分析**

在 2020‑2024 年的实测数据上与 ECMWF‑IFS、NCEP‑GFS、CMA‑GFS 等顶尖 NWP 系统对比，CycloneMAE 在 MSLP、MSW 预测（6‑120 小时）和轨迹预测（6‑24 小时）均取得平均 MAE 降低 10‑20% 的显著性能提升；

**⚠️ 局限性**

局限在于轨迹预测误差随 lead‑time 超过 48 小时显著增长，原因是模型的空间感受野受限，未能充分捕捉全球尺度大气环流影响。

---

## 153. The Second Challenge on Cross-Domain Few-Shot Object Detection at NTIRE 2026: Methods and Results

**arXiv ID:** 2604.11998 | [PDF](https://arxiv.org/pdf/2604.11998v1)

**作者:** Xingyu Qiu `[一作]`, Li Wang `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织并评测了 NTIRE 2026 Cross‑Domain Few‑Shot Object Detection（CD‑FSOD）挑战，系统汇总参赛方法并给出最终排名。

**💡 创新点**

创新点在于将多轨道（开放源与闭源）评测与多域验证相结合，鼓励利用大型预训练模型和生成式数据增强来提升跨域泛化能力。

**🔧 技术方法**

采用的技术包括 CD‑ViTO、Domain‑RAG、开放源基础模型（如 GroundingDINO、SAM、Qwen‑VL、DINOv2、Diffusion 生成等）以及多种自监督/增量式伪标签、伪标签筛选、图卷积重加权等。

**📊 数据集**

使用的数据集为 COCO 作为源域，六个公开验证域（ArTaxOr、Clipart1K、DIOR、DeepFish、NEU‑DET、UODD）以及三个人工设计的未见域（RUOD、CARPK、CarDD）。

**📈 对比分析**

与 CD‑ViTO 基线相比，开放源轨道最佳模型（FDUROILab_Lenovo、CDiscover、NJUST‑KMG）在所有 1/5/10‑shot 设置下均提升 10‑30% mAP，闭源轨道最佳（FewShotEverything）也显著优于基线，但整体性能仍低于开放源。

**⚠️ 局限性**

局限性包括：仍受限于极少标注样本导致的类别偏差、域间差异（视觉风格、尺度、边界模糊）导致的泛化难度、以及评测过程对伪标签质量和数据生成策略高度依赖。

---

## 154. LLM-Redactor: An Empirical Evaluation of Eight Techniques for Privacy-Preserving LLM Requests

**arXiv ID:** 2604.12064 | [PDF](https://arxiv.org/pdf/2604.12064v1)

**作者:** Justice Owusu Agyemang `[一作]` (Sperix Labs), Kwame Opuni-Boachie Obour Agyekum `[通讯]` (Kwame Nkrumah University of Science and Technology)

**通讯引用:** 2510 | [OpenAlex ID](https://openalex.org/A5075204373)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统化评估并实现八种LLM请求隐私保护技术，提出决策规则并公开开源实现

**💡 创新点**

首次在统一基准上对八类技术进行实测，对比组合方案，揭示隐私与实用性的权衡与限制

**🔧 技术方法**

使用本地推理、正则+Presidio NER、语义重写、TEE、分层推理、FHE、MPC、差分隐私噪声等技术，并构建可切换管线

**📊 数据集**

构造1,300个模板化样本（4,014敏感注解），涵盖PII、密钥、隐式身份、专有代码四类工作负载

**📈 对比分析**

通过漏率、延迟、Token成本和质量评估，A+B+C组合在PII、代码等场景下实现≤0.6%漏率，延迟≈1–2s；B+C在隐式身份下表现最差；D、E、F、G仅在研究阶段可行

**⚠️ 局限性**

缺陷包括检测器召回率限制、合成工作负载不完全代表真实场景、研究方案未真实部署、语义泄漏评估不完善、多语言支持不足

---

## 155. PubSwap: Public-Data Off-Policy Coordination for Federated RLVR

**arXiv ID:** 2604.12160 | [PDF](https://arxiv.org/pdf/2604.12160v1)

**作者:** Anupam Nayak `[一作]` (Carnegie Mellon University), Gauri Joshi `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7869 | [OpenAlex ID](https://openalex.org/A5067441201)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个联邦RL后训练框架PubSwap，结合LoRA本地微调和共享公共数据的离线更新，提升大型语言模型在推理任务上的表现。

**💡 创新点**

将LoRA与公共数据响应聚合相结合，使用off‑policy公共数据步骤减少客户端漂移，同时保持通信效率；提出两种响应聚合策略。

**🔧 技术方法**

LoRA参数高效微调、GRPO（Group Relative Policy Optimization）强化学习后训练、公共数据响应聚合、FedIT等通信压缩技术。

**📊 数据集**

MATH、DeepMath数学推理数据集；MedQA、MedMCQA医疗推理数据集；公开数据共享集。

**📈 对比分析**

与FedAvg+GRPO、FedProx+GRPO等基线对比；在Qwen和Llama系列模型上，PubSwap在大多数本地步数下均优于基线，尤其在高本地步数下明显提升。

**⚠️ 局限性**

对公共数据质量和规模敏感；off‑policy步骤可能导致策略偏移，局部步骤过少时表现不如基线；高异质性场景仍有挑战，需进一步优化聚合和调度策略。

---

## 156. PC-MIL: Decoupling Feature Resolution from Supervision Scale in Whole-Slide Learning

**arXiv ID:** 2604.12100 | [PDF](https://arxiv.org/pdf/2604.12100v1)

**作者:** Syed Fahim Ahmed `[一作]` (University of Utah), Shireen Y. Elhabian `[通讯]` (University of Utah)

**通讯引用:** 1571 | [OpenAlex ID](https://openalex.org/A5000258401)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 Progressive-Context MIL (PC-MIL) 框架，使用固定 20× 特征将多实例学习中的监督尺度从全滑动图像切换到毫米级区域，并通过进阶上下文混合实现跨尺度训练与推理。

**💡 创新点**

创新点在于将监督尺度视为独立的设计维度，解耦特征分辨率与监督范围；利用临床 2 mm 阈值构建区域袋；提供训练–推理尺度交叉分析，揭示监督尺度对 MIL 泛化的独立影响。

**🔧 技术方法**

技术细节包括：冻结 UNI2-h 编码器提取 256×256 斑点特征；采用 ABMIL、TransMIL 等多实例骨干；通过 ROI 规则构造 1×1、2×2、4×4 mm² 区域袋；引入 α 向量实现逐步混合不同尺度的监督；使用交叉熵优化和 AdamW 训练；评估指标为 Balanced Accuracy (B‑A) 与 98% 敏感度下的特异度 (S@98)。

**📊 数据集**

使用 1,476 张前列腺 H&E 病片，来自五个公开数据集：AGGC22、TCGA‑PRAD、GTEx‑Prostate、DiagSet、Gleason2019。

**📈 对比分析**

与 ABMIL、TransMIL、DSMIL、WiKG、RRT、Transformer、ILRA 等主流 MIL 模型在四种评估尺度（slide、4×4 mm²、2×2 mm²、1×1 mm²）下进行对比。PC‑MIL 在区域尺度上显著提升 B‑A 与 S@98；平均 B‑A 最高 96.16%，平均 S@98 最高 96.89%，远优于仅使用 slide‑level 监督的基线。

**⚠️ 局限性**

局限性包括：仅针对二分类前列腺癌检测；未扩展到多分类 Gleason 分级；对稀疏 ROI 的依赖可能限制在标签稀疏的数据集中的应用；混合比例 α 的选择需经验调优，可能对不同数据集产生不同效果。

---

## 157. Human-Inspired Context-Selective Multimodal Memory for Social Robots

**arXiv ID:** 2604.12081 | [PDF](https://arxiv.org/pdf/2604.12081v1)

**作者:** Hangyeol Kang `[一作]` (University of Geneva), Nadia Magnenat Thalmann `[通讯]` (University of Geneva)

**通讯引用:** 21839 | [OpenAlex ID](https://openalex.org/A5015798012)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 SUMMER 框架，为社交机器人提供无训练的可选择性多模态记忆存储与检索功能。

**💡 创新点**

创新点在于：① 通过情感显著性与新颖度两种人类记忆驱动信号实现选择性记忆编码；② 结合轻量级视觉语言模型自动生成文本描述，实现视觉与文本双模态的无训练检索融合；③ 通过多模态 z‑score 归一化和加权融合，显著提升检索效果。

**🔧 技术方法**

技术手段包括：OpenFace 进行面部情感识别；CLIP‑ViT‑L14 进行视觉编码和新颖度评估；Moondream2 生成场景描述；Mistral‑small3.2 生成回复；ChromaDB 作为向量数据库；轻量级 VLM 进行检索与生成；情绪阈值微调与跨模态融合算法。

**📊 数据集**

使用的数据集：① 81 张自制社交场景图像（Sora 生成）用于评估选择性记忆；② Flickr8k、Flickr30k、MS COCO 用于多模态检索评测；③ LaMem 用于对比非社交情境下的记忆模型表现。

**📈 对比分析**

与随机、固定间隔、ResMem、ViTMem 等基线进行 Spearman 相关性对比；在记忆评分任务中，最佳配置 (0.5, 0.5, 0.0) 的 ρ≈0.506，显著优于基线且超过人类一致度；在检索任务中，融合权重 α≈0.7 时 Recall@10 最高，优于单模态文本或图像检索；并在运行时实现平均 0.87 s 响应时间，满足实时交互需求。

**⚠️ 局限性**

局限性：仅考虑情感和新颖度，未充分利用社交相关性、关系亲密度等因素；合成数据与真实环境可能存在分布差异，导致在非社交场景下效果下降；系统仍需在更大规模、多语种真实对话环境中进一步验证。

---

## 158. Offline-Online Reinforcement Learning for Linear Mixture MDPs

**arXiv ID:** 2604.11994 | [PDF](https://arxiv.org/pdf/2604.11994v1)

**作者:** Zhongjun Zhang `[一作]` (Northwestern University), Sean R. Sinclair `[通讯]` (Northwestern University)

**通讯引用:** 153 | [OpenAlex ID](https://openalex.org/A5007393142)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种适用于线性混合MDP的离线–在线强化学习算法，并给出了其理论回报上界与下界。

**💡 创新点**

创新点在于同时考虑环境偏移与不可控离线行为策略，提出了安全利用离线数据的“交集式”置信集估计，并给出了离线数据是否有用的明确判定条件。

**🔧 技术方法**

核心技术包括值目标回归、岭回归估计、置信集构造、最优交互的贪婪策略以及对特征空间覆盖度的谱分析。

**📊 数据集**

实验采用合成的离散MDP（5个状态，10个动作，H=3）进行仿真，数据由人工生成的离线轨迹构成。

**📈 对比分析**

与纯在线方法以及直接混合离线数据的基线相比，算法在小环境偏移且覆盖度高时显著降低累计回报，且在偏移大时退化到纯在线性能。

**⚠️ 局限性**

局限在于对覆盖度τ的估计依赖于特征可学习性与统一覆盖假设，且理论对高维H的依赖仍有改进空间。

---

## 159. AnyPoC: Universal Proof-of-Concept Test Generation for Scalable LLM-Based Bug Detection

**arXiv ID:** 2604.11950 | [PDF](https://arxiv.org/pdf/2604.11950v1)

**作者:** Zijie Zhao `[一作]` (University of Illinois Urbana-Champaign), Lingming Zhang `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 7576 | [OpenAlex ID](https://openalex.org/A5043546718)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种通用多智能体框架，用自动化 PoC 生成与验证来替代人工验证 LLM 产生的缺陷报告，提升自动化缺陷检测的可扩展性。

**💡 创新点**

创新点在于：①将缺陷报告验证拆解为分析、生成、验证、知识提取四个专用智能体；②引入自演化知识库来共享并优化跨项目的生成经验；③通过独立重执行与证据检查，显著降低奖励劫持与幻觉导致的误报。

**🔧 技术方法**

技术上结合了大型语言模型（Claude 4.5/4.6、Codex 5.3 等）、工具集（编辑器、搜索、网络查询、Bash 执行）、多智能体协作、知识库管理与评分机制。

**📊 数据集**

使用真实大型开源系统（Firefox、Chromium、LLVM、OpenSSL、SQLite、FFmpeg、Redis 等）以及利用自建的 Bug Reporter 生成的真实与假缺陷报告集。

**📈 对比分析**

与 Claude Code（4.5）和 Codex（5.2）两种通用编码代理基线对比，评估指标包括：生成有效 PoC 的比例、误报率、无效 PoC 数量以及人工验证成本。结果显示，本框架在真缺陷上有效 PoC 率提升约 10‑15%，误报率降至接近 0%，而基线误报率高达 50%；但生成有效 PoC 的计算成本与 token 费用略高。

**⚠️ 局限性**

局限性包括：①对极度复杂或未公开依赖的项目仍需大量探索；②知识库在初始阶段无信息时生成效率低；③在缺陷报告极度多样或不完整时，分析阶段仍可能误判；④评估依赖于人工验证，仍受主观因素影响。

---

## 160. BarbieGait: An Identity-Consistent Synthetic Human Dataset with Versatile Cloth-Changing for Gait Recognition

**arXiv ID:** 2604.12221 | [PDF](https://arxiv.org/pdf/2604.12221v1)

**作者:** Qingyuan Cai `[一作]` (Beijing Normal University), Yongzhen Huang `[通讯]` (Beijing Normal University)

**通讯引用:** 5701 | [OpenAlex ID](https://openalex.org/A5034269600)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

构建了基于真实人物的身份一致性合成步态数据集 BarbieGait，并提出 GaitCLIF 模型作为跨服装步态识别基线。

**💡 创新点**

创新点在于（1）通过双重身份映射（骨架长度、体型匹配与运动匹配）保证合成序列保持真实身份；（2）每人生成 100 套随机服装，极大提升服装多样性；（3）提出 Gait-Oriented Normalization（GON）及其帧级/序列级变体，专门消除服装统计并保留细粒运动信息。

**🔧 技术方法**

技术手段包括：MakeHuman 与 SMPL 体型对齐、EasyMoCap 3D 姿态估计、骨骼运动映射、NVIDIA GPU 真实感渲染、HRNet/ PaddleSeg 生成多模态数据；在模型端使用 GON-P3D/GON-3D、GON-FC 以及深度卷积骨架网络（DeepGaitV2、SkeletonGait）。

**📊 数据集**

使用的主要数据集为自制的 BarbieGait（521 人，100 套服装/人），并在公开服装变化数据集 CCPG、SUSTech1K、Gait3D、GREW 上进行验证。

**📈 对比分析**

与现有基线比较时，GaitCLIF 在 BarbieGait 上 mAP 提升至 73.3%（相较 72.3% 的 SkeletonGait），在 CCPG、SUSTech1K、Gait3D、GREW 等真实数据集上均实现 1–4% 的 Rank‑1 / mAP 提升，显示出跨服装与跨域的鲁棒性。

**⚠️ 局限性**

局限性包括：合成数据与真实世界仍存在域差距，模型在极少服装变化或极端光照环境下的表现尚未充分验证；并且目前仅使用了剪影/姿态等低级视觉模态，未充分利用 RGB 细节。

---

## 161. Filtered Reasoning Score: Evaluating Reasoning Quality on a Model's Most-Confident Traces

**arXiv ID:** 2604.11996 | [PDF](https://arxiv.org/pdf/2604.11996v1)

**作者:** Manas Pathak `[一作]` (University of Texas at Austin), Liu Leqi `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种新的评价指标 Filtered Reasoning Score (FRS)，用于衡量大型语言模型在高置信度输出下的推理质量，从而弥补单纯依赖答案准确率无法反映模型推理真实性的不足。

**💡 创新点**

创新点在于：① 将推理质量与模型自身置信度进行条件化，采用置信度过滤后仅评估最可靠的推理轨迹；② 设计四维评分量表（faithfulness, coherence, utility, factuality）并用 GPT-4o-mini 进行自动评判；③ 证明 FRS 能在准确率相近的模型间显著区分，并能预测置信度选取是否提升推理质量。

**🔧 技术方法**

主要技术包括：基于 token 低概率尾部的 logit 置信度估计；对多条推理轨迹进行 top‑K% 过滤；使用 GPT-4o-mini 作为 “LLM-as-a-judge” 对推理轨迹进行四维评分；统计分析（bootstrap 95% CI、Spearman 相关等）评估指标有效性。

**📊 数据集**

使用的推理基准数据集有：GSM8K、MATH500、SVAMP、AQuA、GPQA、CommonsenseQA；在这些基准上评估 9 个公开权重模型（1.5B–14B 参数）。

**📈 对比分析**

比较方法：对同一模型在不同基准上计算 pass@1、top‑10% 置信度准确率、未过滤推理分数和 FRS；结果显示：FRS 在准确率相差 ≤5% 的模型对中差距更大（平均 6×），并出现排名逆转（如 Qwen2.5‑7B 从 #1 降至 #7，DS‑R1‑1.5B 从 #8 升至 #2）。FRS 还能显著预测置信度选取是否提升推理质量（相关 r=0.49）。

**⚠️ 局限性**

局限性：① 依赖 LLM 评判器，可能带有评判偏差；② 需要多次采样与评判，计算成本高；③ 置信度阈值（K%）的选择与模型架构有关，未给出通用最优设置；④ 论文未深入探究训练方式对置信度‑推理对齐的影响，仍是开放问题。

---

## 162. Robust Explanations for User Trust in Enterprise NLP Systems

**arXiv ID:** 2604.12069 | [PDF](https://arxiv.org/pdf/2604.12069v1)

**作者:** Guilin Zhang `[一作]` (Workday AI), Jerry Ting `[通讯]` (Workday AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个黑盒离开一个词遮蔽解释评估流程，在三大文本分类基准上对六种编码器与解码器模型的解释稳定性进行64,800例评估。

**💡 创新点**

创新点在于将离开一个词遮蔽解释与多级真实扰动相结合，提出基于翻转率的解释稳定性度量，并揭示解码器LLM相较编码器在鲁棒性与规模提升上的显著优势。

**🔧 技术方法**

使用了离开一词遮蔽的解释器、字符/词/句子级扰动（swap、deletion、shuffling、back‑translation）以及配对bootstrap的置信区间计算。

**📊 数据集**

实验基准包括SST‑2、AG News与IMDB三种长度不同的文本分类数据集。

**📈 对比分析**

通过计算每个模型在每种扰动下的翻转率并给出95%置信区间，实验发现解码器LLM的翻转率平均下降73%，并随模型规模从7B提升至70B再下降44%，显著优于BERT/RoBERTa编码器。

**⚠️ 局限性**

局限性在于仅评估了离开一词遮蔽解释的稳定性，未验证解释的正确性或对实时生产的成本和环境影响，并且结果仅适用于特定的文本分类任务。

---

## 163. Design and Deployment of a Course-Aware AI Tutor in an Introductory Programming Course

**arXiv ID:** 2604.11836 | [PDF](https://arxiv.org/pdf/2604.11836v1)

**作者:** Iris Groher `[一作]` (Johannes Kepler University), Michael Vierhauser `[通讯]` (University of Innsbruck)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并部署了一个基于LLM、检索增强、课程对齐的在线Python辅导系统，帮助学生在自学中获取提示、问题解答和代码调试支持。

**💡 创新点**

创新点在于将检索增强生成与Socratic式提示相结合，严格限定回答只基于官方课程材料，并实现可配置的上下文感知功能。

**🔧 技术方法**

使用技术包括React前端、FastAPI后端、OpenAI API（Assistant）+ Retrieval-Augmented Generation（向量检索）以及自定义Prompt管理。

**📊 数据集**

数据集为整门课程的教学材料（课件、代码示例、作业说明等）构建的知识库。

**📈 对比分析**

评估方法采用交互日志分析与问卷调查的混合方法，结果显示学生主要使用系统进行概念解释、实现提示和调试，问卷平均得分3.84/5；系统响应延迟偶有影响，但整体体验正面。

**⚠️ 局限性**

局限性包括样本规模仅39名商科本科生、研究时间仅四周、覆盖的主题仅为集合与函数、缺乏对学习成效的量化验证、对更高级学习者的适用性未知，以及系统响应时间需进一步优化。

---

## 164. Programming Language Co-Usage Patterns on Stack Overflow: Analysis of the Developer Ecosystem

**arXiv ID:** 2604.12123 | [PDF](https://arxiv.org/pdf/2604.12123v1)

**作者:** Bachan Ghimire `[一作]` (University of Victoria), Nitin Gupta `[通讯]` (University of Victoria)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究构建了一个三阶段的无主动反馈管道，利用 Stack Overflow 上 435,803 位活跃多语言开发者的语言共用记录，挖掘隐式需求信号，生成利益相关者原型、需求不可分离约束以及基于社区的需求范围边界；

**💡 创新点**

创新点在于首次将被动行为数据（语言共用模式）与 FP‑Growth、LDA、Louvain 三种算法结合，三者在不同假设下并行收敛，形成可操作的需求映射，为 CrowdRE 提供了全新的无调查式分段方法；

**🔧 技术方法**

采用 SQL 解析 Stack Overflow 原始数据后，使用 FP‑Growth 频繁项集/关联规则、LDA 主题建模（k=25）和 Louvain 社区检测（在 Neo4j 中构建加权共用图）三种技术；

**📊 数据集**

数据集为 2024 年的 Stack Overflow 数据快照（36.1M 条帖子、7.25M 条用户、41.6M 条标签），经过过滤后得到 435,803 名活跃多语言开发者和 186 种编程语言；

**📈 对比分析**

三种方法结果高度一致：FP‑Growth 得到 106 个频繁项集、62 条关联规则；LDA 提取 25 个解释性主题；Louvain 发现 3 个社区，模块度 Q=0.096；相较于传统单一显式反馈分析，本管道在无调查条件下实现了高效、可复现的需求挖掘；

**⚠️ 局限性**

局限性包括：仅基于 Stack Overflow 的样本偏倚；标签噪声与误标记；过滤掉 82% 用户可能忽略了休闲/新手需求；静态时间窗口未捕捉技术演进；方法参数（FP‑Growth 阈值、LDA k、社区检测分辨率）对结果敏感；缺乏行业或实验验证。

---

## 165. When Self-Reference Fails to Close: Matrix-Level Dynamics in Large Language Models

**arXiv ID:** 2604.12128 | [PDF](https://arxiv.org/pdf/2604.12128v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 166. When Drawing Is Not Enough: Exploring Spontaneous Speech with Sketch for Intent Alignment in Multimodal LLMs

**arXiv ID:** 2604.11964 | [PDF](https://arxiv.org/pdf/2604.11964v1)

**作者:** Weiyan Shi `[一作]` (Singapore University of Technology and Design), Kenny Tsu Wei Choo `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 252 | [OpenAlex ID](https://openalex.org/A5084357603)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

创建并评估了一套在早期设计过程中同时绘图与自发说话的实时草图-语音对齐数据集，并使用该数据集验证多模态LLM在生成设计图像时加入语音可显著提升与用户意图的一致性。

**💡 创新点**

首创收集设计师在创意阶段实时绘图并说话的多模态数据，证明语音与草图互补性，并展示加入语音后意图对齐度显著提升。

**🔧 技术方法**

使用 Gemini 2.5 Flash Image 生成图像，Gemini 2.5 Pro 作为 MLLM-as-judge 进行结构化评估；语音转文字采用 Google Speech‑to‑Text；实验中对比草图仅与草图+语音两种输入。

**📊 数据集**

自建数据集：58 条草图+实时语音实例，来自 11 名参与者在 30 分钟烤面包机设计任务中的表现，每条实例附有自报意图摘要。

**📈 对比分析**

对每个实例分别使用草图仅输入与草图+语音输入生成图像，由 MLLM-as-judge 在形式、功能、体验及总体意图对齐上打分；结果显示草图+语音的中位分从 2.0 提升到 7.0，所有指标均显著差异（p<10⁻¹⁴），说明加入语音能显著提升意图对齐。

**⚠️ 局限性**

语音转写噪声与口语碎片化导致信息不完整或错误；当语音信息不足时提升有限；实验仅在烤面包机设计场景，需进一步验证在更广泛任务与更大规模数据下的泛化能力。

---

## 167. Development, Evaluation, and Deployment of a Multi-Agent System for Thoracic Tumor Board

**arXiv ID:** 2604.12161 | [PDF](https://arxiv.org/pdf/2604.12161v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 168. Continuous Knowledge Metabolism: Generating Scientific Hypotheses from Evolving Literature

**arXiv ID:** 2604.12243 | [PDF](https://arxiv.org/pdf/2604.12243v1)

**作者:** Jinkai Tao `[一作]`, Menglin Yang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了连续知识代谢（CKM）框架，通过滑动时间窗口增量更新知识库并生成可预测的科学假设。

**💡 创新点**

创新地将时间演化和变化信号显式整合到假设生成流程，区分CKM‑Lite与CKM‑Full，并揭示质量–覆盖率权衡与知识变动类型对预测性能的影响。

**🔧 技术方法**

利用大型语言模型（Gemini‑2.5‑Flash、GPT‑4o）进行文献抽取、知识更新、差分分类、变化检测和基于轨迹的假设生成。

**📊 数据集**

在50个自然语言处理与人工智能研究主题的公开论文（2019–2027）上构建知识库，覆盖约1,800篇文献。

**📈 对比分析**

与批量处理基线、摘要仅处理以及人类/代理评估相比，CKM‑Lite在命中率、命题产出、覆盖范围和token成本方面分别提升5.8%、17.3%、36/50及92%节省，显示显著优势。

**⚠️ 局限性**

主要局限包括对LLM评估的依赖、CKM‑Full干预不可拆解、结果仅在开放式、快速演化领域验证、未进行专家评审以及数据集偏向NLP/AI领域。

---

## 169. A unified data format for managing diabetes time-series data: DIAbetes eXchange (DIAX)

**arXiv ID:** 2604.11944 | [PDF](https://arxiv.org/pdf/2604.11944v1)

**作者:** Elliott C. Pryor `[一作]` (University of Virginia), Anas El Fathi `[通讯]` (University of Virginia)

**通讯引用:** 738 | [OpenAlex ID](https://openalex.org/A5044231781)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

设计并实现了 DIAX，一种统一的 JSON 规范，用于整合 CGM、胰岛素、饮食等多源糖尿病时序数据，并提供数据转换、可视化与分析工具。

**💡 创新点**

创新点在于：1）保持每条信号的完整元数据（设备类型、胰岛素配方等）以支持可重复性；2）不强制采样率，存储原始时间戳，避免插值误差；3）提供可扩展的 schema 及完整的 Python/MATLAB 工具箱；4）兼容多种现有数据集并开放社区贡献。

**🔧 技术方法**

使用技术包括：JSON schema 设计、Python 与 MATLAB 开发、时间对齐与插值算法、糖尿病临床指标计算（如 TIR、AGP 图）、GitHub 开源托管。

**📊 数据集**

已支持的公开数据集包括：DCLP3、DCLP5、IOBP2、PEDAP、T1Dexi、Loop，总计超过 10 万名受试者、10 亿小时以上数据。

**📈 对比分析**

方法评估通过将现有数据集转换为 DIAX 并使用内置工具直接计算指标与绘制 AGP，展示与现有工具（如 BabelBetes）相比能直接跨数据集复现结果，显著降低了数据预处理时间。性能表现体现在：统一 schema 让脚本迁移成本降为零、时间对齐与插值支持多种网格，确保分析一致性。

**⚠️ 局限性**

局限性包括：1）仅为研究层面，不能替代完整临床数据库；2）需自行获取原始数据，转换脚本无法突破数据许可；3）缺失或异常采样仍需用户手动处理；4）对极端稀疏或不完整信号的处理机制尚不完善；5）目前覆盖的信号类型有限，未来需扩展更多生理/行为数据。

---

## 170. Quantized Online LQR

**arXiv ID:** 2604.11930 | [PDF](https://arxiv.org/pdf/2604.11930v1)

**作者:** Barron Han `[一作]` (California Institute of Technology), Babak Hassibi `[通讯]` (California Institute of Technology)

**通讯引用:** 27852 | [OpenAlex ID](https://openalex.org/A5002430773)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究在线LQR在通信受限条件下的学习控制算法，提出量化置信等价（QCE‑LQR）框架；

**💡 创新点**

证明在实现近最优回报时需传输的位数上界与下界均为Θ(log T)，并通过双尺度自适应量化实现此界；

**🔧 技术方法**

利用普通最小二乘估计、ε‑greedy探索、离散代数里卡提方程（DARE）求解与自适应两尺度量化+Elias Gamma编码；

**📊 数据集**

在四个基准系统上验证：标量不稳定系统、双积分器、倒立摆、波音 747横向模型；

**📈 对比分析**

与无量化置信等价控制比较，QCE‑LQR在10,000步内的期望回报几乎相同，通信量仅为O(log T)（约123–819位），整体性能与基准相当；

**⚠️ 局限性**

仍存在维度常数不匹配、对称双向通道扩展未完全证明、以及安全触发阈值设定保守等限制。

---

## 171. Refined Differentially Private Linear Regression via Extension of a Free Lunch Result

**arXiv ID:** 2604.11820 | [PDF](https://arxiv.org/pdf/2604.11820v1)

**作者:** Sasmita Harini S `[一作]` (Indian Institute of Science), Anshoo Tandon `[通讯]` (Indian Institute of Science)

**通讯引用:** 347 | [OpenAlex ID](https://openalex.org/A5033175623)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了DP-RSS机制，利用多维单纯形变换在加/删DP模型下对线性回归的充分统计量进行差分隐私估计；

**💡 创新点**

创新点在于通过构造互相独立的估计器并进行逆方差加权，利用数据的代数约束实现方差显著降低，实现了“免费午餐”效应的扩展；

**🔧 技术方法**

核心技术包括多维单纯形变换、Laplace机制、逆方差加权组合、隐私预算的分配与后处理；

**📊 数据集**

使用两组人工合成数据集（均匀分布的x，线性模型y=αx+β+噪声，随后裁剪到[0,1]），分别包含5,000和10,000个样本；

**📈 对比分析**

与传统DP-SS和DP-Theil-Sen方法比较，DP-RSS在各隐私预算下L1/L2误差均显著下降，尤其在低ε区间；实验显示与DP-SS相比，关键二次统计量方差提升约4.8×，整体误差下降数倍；

**⚠️ 局限性**

局限性包括：对多项式回归的最佳隐私预算分配尚未确定；对多元回归的扩展仍需研究；以及在其他统计任务中是否能同样利用代数约束仍是开放问题。

---

## 172. Physics-Grounded Monocular Vehicle Distance Estimation Using Standardized License Plate Typography

**arXiv ID:** 2604.12239 | [PDF](https://arxiv.org/pdf/2604.12239v1)

**作者:** Manognya Lokesh Reddy `[一作]`, Zheng Liu `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于美国车牌标准字体的被动标识符，利用车牌字符高度、笔画宽度和字符间距进行几何深度估计，并与MiDaS深度网络融合，生成连续的距离、相对速度和碰撞警告。

**💡 创新点**

创新点包括：① 四种并行阈值处理的车牌检测器与自适应模式切换；② 三阶段状态识别引擎（OCR文本匹配、HSV多设计颜色评分和MobileNetV3-Small CNN），实现高精度车牌状态判定；③ 多特征逆方差加权融合与在线EMA尺度对齐的MiDaS深度融合，配合一维恒速Kalman滤波，兼顾精度与鲁棒性。

**🔧 技术方法**

技术手段包括：相机标定、基于相似三角形的几何距离公式、姿态补偿（车道线消失点估计+深度航向预测）、字符分割与多阈值方法、OCR（EasyOCR/Tesseract）、HSV颜色计分、轻量级CNN分类、MiDaS DPT-Hybrid深度网络、EMA尺度对齐、逆方差加权融合、Kalman滤波和碰撞时间计算。

**📊 数据集**

数据集为真实道路摄像头收集的数千帧车牌视频，配合激光测距仪/测距带进行地面真值；此外使用公开车牌图像集（如MS-COCO、KITTI）进行MiDaS网络验证；实验中还使用静态实验板块与可移动板进行距离校准。

**📈 对比分析**

与深度学习基准（MiDaS、ZoeDepth、Depth Anything）对比，几何+融合方法在10 m处MAE为0.23 m（2.3%相对误差），比深度基准相对误差低约5倍；在连续输出和遮挡恢复方面，几何+融合方法能够保持连续性，而单纯几何方法因遮挡失效；Kalman滤波进一步降低噪声并实现可靠碰撞警告。

**⚠️ 局限性**

局限性包括：① 需要车牌在视野内且可读，遮挡严重或车牌缺失时仍需依赖深度网络；② 对极端视角、光照变化和高车速场景的鲁棒性尚待进一步验证；③ 车牌高度标准存在州间差异，若未准确匹配状态会引入偏差；④ 需要定期相机标定与姿态估计准确，误差累积可能影响精度；⑤ 当前实现仅针对美国车牌，对其他国家/地区车型适用性有限。

---

## 173. Uncertainty Guided Exploratory Trajectory Optimization for Sampling-Based Model Predictive Control

**arXiv ID:** 2604.12149 | [PDF](https://arxiv.org/pdf/2604.12149v1)

**作者:** O. Goktug Poyrazoglu `[一作]` (University of Minnesota), Volkan Isler `[通讯]` (University of Texas at Austin)

**通讯引用:** 6071 | [OpenAlex ID](https://openalex.org/A5033839227)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了一种基于不确定性引导的探索轨迹优化（UGEO）算法，并将其集成到采样式模型预测控制（MPC）框架中；

**💡 创新点**

创新点在于将轨迹视作概率分布（通过不确定性椭圆估计），并利用Hellinger距离对轨迹分布进行分离，系统性地扩展配置空间探索；

**🔧 技术方法**

采用随机采样、线性化传播协方差、不确定性椭圆、Hellinger距离度量、Stein变分梯度、MPPI等技术；

**📊 数据集**

实验使用二维运动学车轮模型在8×8 m（无障碍与拥挤）和20×20 m未知拥挤环境的仿真数据，以及1/5比例真实小车在实景拥挤环境中的数据；

**📈 对比分析**

与MPPI、MoG‑MPPI、SV‑MPC以及log‑MPPI对比，UGEO在相同采样预算下在开放和拥挤环境中实现更高的成功率、更快的收敛速度（如在无障碍环境中比最优基线快72.1%），在实景测试中成功率提升至88%且平均到达时间缩短；

**⚠️ 局限性**

局限性包括：算法主要在低维系统上验证，扩展到高维人形机器人仍需进一步研究；对实时计算的依赖导致在高采样预算下计算量较大；

---

## 174. Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories

**arXiv ID:** 2604.12097 | [PDF](https://arxiv.org/pdf/2604.12097v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 175. Long-Horizon Plan Execution in Large Tool Spaces through Entropy-Guided Branching

**arXiv ID:** 2604.12126 | [PDF](https://arxiv.org/pdf/2604.12126v1)

**作者:** Rongzhe Wei `[一作]` (Georgia Institute of Technology), Leman Akoglu `[通讯]` (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了大型工具库下的长周期电商任务评估基准SLATE，并提出基于熵的分支搜索算法EGB以提升工具使用代理的自校正和计算效率。

**💡 创新点**

创新点在于①构建可支持多步骤、可变工具集且具有可验证终态的全流程评估基准；②提出利用预测熵引导的局部分支搜索，兼顾黑盒和白盒模型。

**🔧 技术方法**

采用离散采样或logits熵估计、ReAct/Reflexion框架、MCTS对比、工具模拟器与强化学习式决策流程。

**📊 数据集**

使用Synthetic Large-scale API Toolkit for E-commerce（SLATE）数据集，包含约1000种工具、平均6.3次调用、12类电商任务。

**📈 对比分析**

与Baseline-LLM、ReAct、Reflexion、LATS等方法对比，EGB在Claude-4-Sonnet上执行成功率提升约20–30%，在Qwen2.5-7B上可达67%+，且在计算成本和token消耗上显著优于MCTS。

**⚠️ 局限性**

局限性包括：基准为合成且仿真器为确定性，未覆盖真实API延迟/失败；EGB采样版对API调用成本高；对非确定性环境的适配尚未验证；数据集和算法可能带有Claude模型的偏差。

---

## 176. BiasIG: Benchmarking Multi-dimensional Social Biases in Text-to-Image Models

**arXiv ID:** 2604.11934 | [PDF](https://arxiv.org/pdf/2604.11934v1)

**作者:** Hanjun Luo `[一作]` (New York University Abu Dhabi), Hanan Salam `[通讯]` (New York University Abu Dhabi)

**通讯引用:** 1646 | [OpenAlex ID](https://openalex.org/A5047633471)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了名为 BiasIG 的统一 T2I 社会偏见基准，包含 4 维度（获得属性、受保护属性、表现方式、可见性）和 47,040 条 prompt。

**💡 创新点**

创新点在于：① 将社会学与机器伦理学框架结合，提出四维偏见定义；② 设计全自动评估管线，使用微调的 InternVL 进行视觉推理；③ 引入隐性、显性偏见评分与 Manifestation Factor η，区分偏见来源。

**🔧 技术方法**

主要技术包括：多模态大语言模型 InternVL‑4B 微调、视觉问答 (VQA)、多维度统计指标（S_imp、S_exp、η）以及 Prompt‑based 与 finetune‑based 的去偏方法。

**📊 数据集**

使用的主要数据集：FairFace（用于微调）、U.S. Bureau of Labor Statistics（职业分布）、全球人口统计（性别、年龄、种族分布）以及 47,040 条自构 prompt。

**📈 对比分析**

对 8 种主流 T2I 模型和 3 种去偏方法进行评估；结果显示现代模型在性别偏见已显著下降，但种族/年龄偏见依旧严重；去偏方法中 Prompt‑based（PreciseDebias）最有效；模型蒸馏后偏见加剧，说明加速模型需额外公平审计。

**⚠️ 局限性**

局限性包括：只覆盖性别、年龄、种族三类受保护属性，未考虑非二元身份；多人场景下的视觉推理精度受限；基准对训练数据偏差和模型能力的解释仍不完整；去偏方法往往引发交叉属性混杂。

---

## 177. Parametric Interpolation of Dynamic Mode Decomposition for Predicting Nonlinear Systems

**arXiv ID:** 2604.12103 | [PDF](https://arxiv.org/pdf/2604.12103v1)

**作者:** Ananda Chakrabarti `[一作]` (Ohio State University), Debdipta Goswami `[通讯]` (Ohio State University)

**通讯引用:** 331 | [OpenAlex ID](https://openalex.org/A5044260875)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a8e75ba4-7a2d-4153-b003-06c94533add0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了参数插值动态模态分解（PI-DMD）框架，用于从数据学习参数化的Koopman仿真模型，并实现对未见参数值的预测。

**💡 创新点**

创新点在于将已知的参数仿射结构直接嵌入DMD回归步骤，避免传统方法对模态、特征值或降维算子进行插值，从而在样本稀疏和多维参数空间下获得更稳健、准确的预测。

**🔧 技术方法**

采用了动态模态分解（DMD）、Koopman理论、Euler离散、最小二乘回归和SVD低秩截断，并通过参数函数的线性编码实现参数化。

**📊 数据集**

使用了流体绕圆柱、电子束在交变磁场中的振荡以及虚拟阴极振荡等高维非线性系统的数值仿真数据；分别在不同参数（Re、B₀、N_e等）上收集训练与测试样本。

**📈 对比分析**

与标准DMD、堆叠参数DMD和rKOI进行比较，PI-DMD在时间平均残差上显著优于基准方法，并在少样本和多维参数场景下表现出更高的鲁棒性和预测精度。

**⚠️ 局限性**

主要局限在于假设参数仿射函数已知，对观测符号选择及有限维子空间不易保证，且在更高维或非仿射参数情况下仍需进一步改进。

---

## 178. Robotic Nanoparticle Synthesis via Solution-based Processes

**arXiv ID:** 2604.12169 | [PDF](https://arxiv.org/pdf/2604.12169v1)

**作者:** Dasharadhan Mahalingam `[一作]` (Stony Brook University), Stanislaus S. Wong `[通讯]` (Stony Brook University)

**通讯引用:** 18576 | [OpenAlex ID](https://openalex.org/A5013790868)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在实验室中通过机器人实现解决方案基合成（金纳米粒子与磁铁矿纳米粒子）的自动化操作，涵盖拾取、倾倒、搅拌、阀门调节等多步受几何约束的操作。

**💡 创新点**

创新点在于：①利用单一演示提取常数螺旋段（constant screw）来编码操作约束，保持坐标不变性；②构建可复用的操控原语库，支持跨实验重用；③将螺旋几何与ScLERP插值及解析运动控制相结合，实现满足约束的高质量轨迹生成。

**🔧 技术方法**

使用技术包括：螺旋几何运动表示与解析；ScLERP（Screw Linear Interpolation）插值；解析运动控制（RMRC/Jacobian pseudoinverse）；Realsense RGBD相机与SAM进行视觉检测；Arduino/传感器接口实现溶液计量、pH、温度控制；机器学习演示提取与参数化重用。

**📊 数据集**

数据集主要为实验室手工演示数据（单次kinesthetic demonstration）以及实验过程中采集的图像、颜色、pH、温度、XRD、TEM等实验室传感器数据；未使用公开公开数据集。

**📈 对比分析**

对比方法：将机器人执行的实验与传统人工操作在产物尺寸分布、颜色变化、pH曲线等方面进行对比；实验结果显示机器人生成的金纳米粒子和磁铁矿纳米粒子与人工实验在尺寸、形貌、晶体结构上高度一致，且机器人能够在规定时间内完成多步操作，重复性优于人工。

**⚠️ 局限性**

limitations: 需要预先提供物体位置与姿态，缺乏自主感知定位；单一演示限制了对异常情况的鲁棒性；无法实时动态调整实验参数；对新任务仍需重新演示；对环境变化（如光照、温度）依赖人工设置。

---

## 179. Modality-Native Routing in Agent-to-Agent Networks: A Multimodal A2A Protocol Extension

**arXiv ID:** 2604.12213 | [PDF](https://arxiv.org/pdf/2604.12213v1)

**作者:** Vasundra Srinivasan `[一作]` `[通讯]` (Stanford), Vasundra Srinivasan (Stanford)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了MMA2A层，在Agent-to-Agent协议上实现多模态原生路由，显著提升跨模态推理的准确率。

**💡 创新点**

提出两层需求：协议层保持原生模态，推理层需使用LLM来利用高保真证据；MMA2A无需改动协议，利用Agent Card与FilePart实现无缝原生路由；在CrossModal-CS基准上实现20个百分点的准确率提升。

**🔧 技术方法**

使用A2A协议、Python轻量级代理（MAR）、Gemini 2.5 Flash LLM、FilePart、Agent Card、STT、图像字幕等技术。

**📊 数据集**

采用自制的CrossModal‑CS 50任务数据集（产品缺陷、装配指导、视觉排障、保修索赔）以及15个产品保修条款和10条排障条目。

**📈 对比分析**

采用配对实验：同一任务、同一模型、同一知识库下对比文本瓶颈(Text‑BN)与MMA2A；结果为TCA 52% vs 32%（+20pp，p=0.006），延迟13.04s vs 7.19s（1.8×），带宽相近。

**⚠️ 局限性**

局限包括样本规模仅50个任务、仅针对客服领域、模型同质化、单机实验、仅英文语音、整体准确率仍不高，未考虑网络延迟和多模型异构情况。

---

## 180. A Residual-Shell-Based Lower Bound for Ollivier-Ricci Curvature

**arXiv ID:** 2604.12211 | [PDF](https://arxiv.org/pdf/2604.12211v1)

**作者:** Xiang Gu `[一作]` (Xi'an Jiaotong University), Jian Sun `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 266747 | [OpenAlex ID](https://openalex.org/A5100785015)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于残差壳层的下界（RS-LB）来近似 Ollivier–Ricci 曲率，并给出了相应的高效算法实现

**💡 创新点**

创新点在于利用壳层分解的运输计划将 Wasserstein 距离分段逼近，从而得到更紧的下界；并且该方法可应用于任意 k-hop 随机游走，克服了先前仅限 1-hop 且误差较大的 KP-LB

**🔧 技术方法**

使用 Wasserstein 距离理论、壳层分解、线性规划、随机游走测度与图论算法

**📊 数据集**

在 Erdős–Rényi、Barabási–Albert、Watts–Strogatz 与二维网格等四种典型图模型上进行实验

**📈 对比分析**

与精确 ORC（线性规划）以及 KP-LB 进行比较；RS-LB 在运行时间上比精确方法快数十倍，且平均绝对误差明显低于 KP-LB（在网格图上甚至能完全匹配精确值）

**⚠️ 局限性**

局限性包括：相较于 KP-LB 计算耗时略高；在更大规模图或更高 k-hop 级别下的性能尚需进一步验证；在某些特殊图结构下残差壳层方法的紧度可能不如预期

---

## 181. Redefining Quality Criteria and Distance-Aware Score Modeling for Image Editing Assessment

**arXiv ID:** 2604.12175 | [PDF](https://arxiv.org/pdf/2604.12175v1)

**作者:** Xinjie Zhang `[一作]` (Northwestern Polytechnical University), Qingsen Yan `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 4520 | [OpenAlex ID](https://openalex.org/A5081607584)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DS-IEQA 框架，联合学习图像编辑质量评估的指标定义和连续分数空间建模

**💡 创新点**

创新点包括：1）Feedback-Driven Metric Prompt Optimization (FDMPO) 自动化细化评估指标，弥补人工提示的不可变性；2）Token-Decoupled Distance Regression Loss (TDRL) 通过分数距离最小化实现连续分数预测，避免传统分类损失的离散化问题

**🔧 技术方法**

采用多模态大型语言模型（如 Qwen3-VL-8B 和 InternVL3-8B）并进行 LoRA 参数高效微调，结合 FDMPO 与 TDRL 两大技术实现评估

**📊 数据集**

使用 2026 NTIRE X-AIGC Quality Assessment Track 2 公开数据集进行训练与评估，涵盖视觉质量、编辑保真度和内容保留三维度

**📈 对比分析**

与多组基线和挑战参与者比较，DS-IEQA 在所有维度和分布（在分布/离散分布）均保持稳定且最高的平均相关系数，最终在排行榜上排名第 4，表现优于同规模模型的其他方法

**⚠️ 局限性**

局限性：依赖大规模多模态 LLM，算力与训练成本较高；FDMPO 仍需手动设定初始提示；TDRL 仅在数字分数上有效，未扩展到更复杂的多维度交互评价

---

## 182. SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration

**arXiv ID:** 2604.12247 | [PDF](https://arxiv.org/pdf/2604.12247v1)

**作者:** Zhuofan Wen `[一作]` (Chinese Academy of Sciences), Yang Feng `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 9274 | [OpenAlex ID](https://openalex.org/A5061627737)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 SpecBound，自我草稿式推测解码框架，利用层级温度退火的早退出与受限推测+缓存状态算法，在不修改基模型参数的前提下实现无损加速。

**💡 创新点**

① 通过层级温度退火抑制浅层过度自信的错误预测；② 用深度/宽度上限结合缓存状态的受限推测，避免困难 token 造成的冗余计算；③ 将上述技术结合，保持与原模型完全一致的输出，并实现大幅加速。

**🔧 技术方法**

层级温度退火的早退出判定；受限推测与缓存状态算法（BSCS）；轻量级中间层 LM 头训练；并行多 token 验证；理论速度分析。

**📊 数据集**

训练：68K ShareGPT 多轮对话；评估：Spec‑Bench（翻译、问答、摘要等多任务）以及 Vicuna、CodeLlama‑Instruct 7B/13B 预训练模型。

**📈 对比分析**

与 Lookahead、Medusa、REST、Kangaroo、SPACE、AdaDecode 等自草稿方法在 Spec‑Bench 上对比。SpecBound 在多模型、多任务上平均获得 1.89×–6.41× 的 wall‑time 加速，最高 6.41×；压缩率（CR）最高 5.48；与标准自回归相比，Per‑token 延迟降低 1.98×，吞吐率提升 2.15×；在温度采样和 top‑p 采样等多种解码策略下保持 1.88–2.08× 加速。

**⚠️ 局限性**

仍需在中间层训练轻量级 LM 头，无法完全无训练；多项超参数（温度退火系数、深度/宽度上限、阈值等）需要经验调优；在极长输入或大批量推理时可能面临缓存/内存瓶颈。

---

## 183. Unveiling the Surprising Efficacy of Navigation Understanding in End-to-End Autonomous Driving

**arXiv ID:** 2604.12208 | [PDF](https://arxiv.org/pdf/2604.12208v1)

**作者:** Zhihua Hua `[一作]` (Fudan University), Wenchao Ding `[通讯]` (Fudan University)

**通讯引用:** 8568 | [OpenAlex ID](https://openalex.org/A5103324589)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种新型的全局导航信息表示方式 Sequential Navigation Guidance（SNG），构建了基于SNG的QA数据集 SNG-QA，并提出了端到端多模态模型 SNG-VLA，随后在 Bench2Drive 与 NAVSIM 两个闭环评测平台上验证了其优越性能。

**💡 创新点**

创新点包括：1）将导航路径与实时 Turn‑by‑Turn (TBT) 信息结合，突破传统单一驾驶指令的表达限制；2）通过 SNG-QA 让局部规划与全局导航对齐；3）在无感知任务监督的情况下，利用精确导航信息实现 SOTA 规划性能。

**🔧 技术方法**

技术实现主要采用多模态大模型：Qwen2.5 语言模型 + SigLIP 视觉编码器 + 统一 Transformer 解码器；对导航路径、TBT 与车辆状态进行专门编码，并通过注意力状态 dropout 和自回归文本生成进一步提升规划表达。

**📊 数据集**

使用的数据集包括 NAVSIM（真实路况数据）、Bench2Drive（CARLA 关闭环基准）、以及由 NAVSIM 生成的约 10 万条 QA 对（SNG-QA）。

**📈 对比分析**

在 Bench2Drive 与 NAVSIM 上与 UniAD、VAD、Transfuser 等现有 E2E-AD 方法对比，SNG-VLA 在 Driving Score、Success Rate、PDM、DAC、TTC 等指标上分别提升 46% 以上、119% 以上，显著超越所有基线，达到 SOTA。

**⚠️ 局限性**

主要局限包括：对导航路径采样间隔与 TBT 信息的准确性敏感；在极稀疏或过密导航点设置时性能下降；依赖高质量导航 API，难以在缺乏导航数据的环境中直接部署；仍需在更复杂多变的真实场景中进一步验证鲁棒性。

---

## 184. Simple Types for Polymorphic Functions

**arXiv ID:** 2604.12194 | [PDF](https://arxiv.org/pdf/2604.12194v1)

**作者:** Barry Jay `[一作]`, Johannes Bader `[通讯]` (Jane Street)

**通讯引用:** 3908 | [OpenAlex ID](https://openalex.org/A5055598034)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

本论文提出了一种针对组合逻辑的极简类型系统，取消了类型变量和传统的量化，仅通过组合器自身的结构来唯一确定类型，并通过“抽象类型”来实现列表、布尔、自然数、函数等数据类型的支持，同时给出了完整的类型推导规则和有效的类型推断算法；此外，还证明了该系统的类型唯一性、归约不变性以及 Turing 完备性，并在 OCaml 中实现了推断器并对多组示例进行实验验证。

**💡 创新点**

创新点在于：① 采用组合器本身的结构来唯一给出类型，避免了多重类型和类型变量的引入；② 通过“抽象类型”标签实现对复杂数据结构和函数类型的隐藏与多态化；③ 将类型推断转化为对组合类型的函数式类型应用，得到单一、可判定的类型推断算法；④ 证明该系统可支持传统 HM 的多态性及更强的多态特性，并保持所有程序为正则化形式，极大简化了程序分析；⑤ 在 Rocq 中给出了形式化证明，并在 OCaml 中实现实验。

**🔧 技术方法**

技术手段包括：组合逻辑与组合器（S、K、I 等）的标准规则；类型应用函数（如 S_0(V)、S_1(U)、S_2(U,V) 等）来代替传统子类型关系；抽象类型标签（tagged）与类型声明（Abs_0/1/2）来实现数据类型的多态与递归；在 Rocq 里形式化证明归约保持性与类型唯一性；在 OCaml 里实现递归的类型推断算法并计数 infer_app 调用以评估性能。

**📊 数据集**

数据集主要是：多种组合器示例（如 cond、pair、fst、snd、inl、inr、isZero、predecessor、plus、fold_left、Z、compiler-of-a-toy-language 等），其中包含了递归、Scott 编码、以及大规模组合器如 S^100 的自相似形式。

**📈 对比分析**

性能评估采用的是推断调用次数（infer_app）与程序长度的比值。实验表明，对于大多数实际程序，调用次数与长度近似线性，比例通常低于 2，极端例子如 S^100 的比例高达 73，(SII)(SII) 的推断不收敛。针对 toy compiler 的实验显示约 2.5 的比例，表明在现实程序上该推断器的时间复杂度保持在常数倍以内。

**⚠️ 局限性**

局限性包括：① 对某些自相似或自引用组合器（如 (SII)(SII) 或重复 S^k）可能导致推断不终止；② 需要使用“假值”（dummy values）来携带类型信息，增加实现复杂度；③ 递归类型的声明与使用仍依赖手工设计的 type application 规则，缺乏自动化；④ 目前尚未支持模块化或高级多态量化（impredicative）等特性；⑤ 在处理更复杂的 sum 类型、函数类型等时仍需要手工定义多种 type application 规则。

---

## 185. Characterizing Resource Sharing Practices on Underground Internet Forum Synthetic Non-Consensual Intimate Image Content Creation Communities

**arXiv ID:** 2604.12190 | [PDF](https://arxiv.org/pdf/2604.12190v1)

**作者:** Bernardo B. P. Medeiros `[一作]` (University of Florida), Kevin R. B. Butler `[通讯]` (University of Florida)

**通讯引用:** 51708 | [OpenAlex ID](https://openalex.org/A5038585229)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

对 4chan 与 Reddit 论坛中的 SNCII 资源共享生态进行大规模实证研究，构建了作者、动作、资源的表型并利用知识图谱进行网络与行为分析。

**💡 创新点**

首次从资源共享视角系统性描述 SNCII 生态，提出教育者、供应商、资源传播链等关键节点，并通过数据揭示法律与技术监管缺口。

**🔧 技术方法**

采用少样本学习（SetFit）对 4chan 文本进行标签，使用基于关键字的匹配识别资源与危害信息，构建 Neo4j 知识图谱进行交互与影响力分析。

**📊 数据集**

4chan 282,154 条评论（2025‑06‑09 至 2025‑11‑21）与 Reddit 78,308 条投稿（同期六个 NSFW AI 子版块）共计约 36 万条文本。

**📈 对比分析**

通过手工标注 200 条样本评估模型，SetFit 对角色/动作 F1=0.85/0.68，规则匹配对资源/危害 F1=0.92；在 Reddit 上关键词过滤召回率高，误判率低（99.8%）。研究未做算法对比，仅给出标签准确性和社区行为指标。

**⚠️ 局限性**

未对图像内容进行分析，受限于匿名性导致个体跟踪困难；采样偏向高精度过滤可能漏检；仅聚焦于 6 个子版块和 4chan，未覆盖其他平台。

---

## 186. A Scoping Review of Large Language Model-Based Pedagogical Agents

**arXiv ID:** 2604.12253 | [PDF](https://arxiv.org/pdf/2604.12253v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 187. TRUST Agents: A Collaborative Multi-Agent Framework for Fake News Detection, Explainable Verification, and Logic-Aware Claim Reasoning

**arXiv ID:** 2604.12184 | [PDF](https://arxiv.org/pdf/2604.12184v1)

**作者:** Gautama Shastry Bulusu Venkata `[一作]` (George Mason University), Aishwarya Gaddam `[通讯]` (George Mason University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 TRUST Agents 多代理框架实现新闻事实核查，分阶段提取、检索、验证、解释，研究模式加入拆解、多人审议和逻辑聚合。

**💡 创新点**

创新点在于把事实核查拆解为可解释的多模代理流程，并通过逻辑拆解与多代理投票显式处理复合命题。

**🔧 技术方法**

使用命名实体识别、依赖分析、LLM 提取；BM25+FAISS 混合检索；LLM 进行命题‑证据比较与解释生成；多代理投票与逻辑聚合。

**📊 数据集**

在 LIAR 数据集上进行实验，使用其政治声明标签。

**📈 对比分析**

与微调的 BERT、RoBERTa 和零射 LLM 进行对比，监督基线在准确率上显著高于 TRUST Agents（约0.65 vs 0.19‑0.52），但后者在可解释性和保留 abstention 上优势明显。

**⚠️ 局限性**

主要局限是高弃权率、检索覆盖不足、证据库有限、LLM 规模小导致不确定性与误判，且在二分类评价上不易体现其价值。

---

## 188. Evaluating Relational Reasoning in LLMs with REL

**arXiv ID:** 2604.12176 | [PDF](https://arxiv.org/pdf/2604.12176v1)

**作者:** Lukas Fesser `[一作]` (Harvard University), Marinka Zitnik `[通讯]` (Harvard University)

**通讯引用:** 15449 | [OpenAlex ID](https://openalex.org/A5086052373)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `67630363-6be0-4f51-ab05-7198250671a5` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文研究了大语言模型在代数、生物和化学三个科学领域的关系推理能力，并提出了关系复杂度（RC）这一衡量指标。

**💡 创新点**

创新点在于引入RC概念并构建可生成、可扩展的基准框架，从而系统控制关系绑定的难度，揭示RC是模型性能下降的主因。

**🔧 技术方法**

使用了生成式基准任务（RPM、RPT、MSA+树、SMILES）、多模型评估（Claude Opus 4.5、Gemini 3 Pro、GPT 5.2）以及多变量回归分析来评估模型表现。

**📊 数据集**

数据集为作者自行生成的REL基准，共约3,016道题，涵盖代数（-A）、生物（-B）和化学（-C）三大域，并已公开在GitHub和HuggingFace上。

**📈 对比分析**

通过对比不同RC水平的任务，发现RC升高时模型性能显著下降；在低RC任务上表现良好，随着RC增大接近随机；即使增加推理时间或使用工具，提升效果有限。

**⚠️ 局限性**

限制包括任务过于人工合成、评估方式多选可能掩盖细粒度错误、上下文长度限制导致无效回答，以及未覆盖更自然真实的关系推理场景。

---

## 189. Fully Homomorphic Encryption on Llama 3 model for privacy preserving LLM inference

**arXiv ID:** 2604.12168 | [PDF](https://arxiv.org/pdf/2604.12168v1)

**作者:** Anes Abdennebi `[一作]` (École de Technologie Supérieure), Laaziz Lahlou `[通讯]` (École de Technologie Supérieure)

**通讯引用:** 53 | [OpenAlex ID](https://openalex.org/A5016021331)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文将后量子全同态加密技术集成到 Llama‑3 的推理流水线，构建了可在加密模式下完成文本生成的 PQC‑LLaMA‑3 模型。

**💡 创新点**

创新点在于将低位量化与全同态加密（通过 Zama 的库实现）联合应用于 Transformer 的自注意力层，只加密部分（单头或多头）层以平衡安全与效率，并首次提供完整的性能评测与缓存行为分析。

**🔧 技术方法**

使用的技术包括 RLWE‑基础的全同态加密、可编程引导重置（PBS）、模拟/执行模式切换、双向数组（DualArray）量化、以及硬件层面的 SIMD 批处理和关键字缓存（KV‑cache）等。

**📊 数据集**

实验采用 79 条文本提示作为推理输入，覆盖 1–10 token（短上下文）与 70–500 token（长上下文）两种生成场景，未使用公开大规模语料库，仅在这些提示上评估模型输出一致性。

**📈 对比分析**

评估方法是比较加密模型与原始 Llama‑3 的文本生成准确率（Top‑k 匹配率）与推理时间/吞吐率；结果显示单头加密模型在 0.24 s 内完成 1–10 token 生成，准确率可达 81.4%，长上下文时 0.54 s 内 500 token 生成准确率 98.2%，吞吐率约 80 token/s，单头模型优于多头模型。

**⚠️ 局限性**

局限性包括：编译时间高（单头约 8 s，完整模型 26 s），加密算子导致显著运算与内存开销，且目前仅在 CPU 上实验，未验证 GPU/TPU 上的可扩展性；加密后模型仍易受侧信道攻击，且对攻击的完整防护需配合其他安全措施。

---

## 190. TEMPLATEFUZZ: Fine-Grained Chat Template Fuzzing for Jailbreaking and Red Teaming LLMs

**arXiv ID:** 2604.12232 | [PDF](https://arxiv.org/pdf/2604.12232v1)

**作者:** Qingchao Shen `[一作]` (Tianjin University), Junjie Chen `[通讯]` (Tianjin University)

**通讯引用:** 6043 | [OpenAlex ID](https://openalex.org/A5100365536)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一个细粒度聊天模板模糊测试框架，用来发现并利用大语言模型的聊天模板漏洞，实现更强的 jailbreak 攻击。

**💡 创新点**

首次将聊天模板视为主要攻击面，设计五类元素级变异规则、启发式搜索策略以及基于主动学习的轻量化评判器，实现高效、精准的模板变异。

**🔧 技术方法**

结合模板级变异规则、蒙特卡罗树搜索+轮盘赌的启发式选择、主动学习改进的规则评判器，并使用 LLM 生成候选变异。

**📊 数据集**

主要使用 AdvBench（520条恶意提示）评估攻击成功率，MMLU 1140条样本评估模型准确率。

**📈 对比分析**

与 ChatBug、GPTFuzzer、TurboFuzzLLM 等基线在12款开源 LLM 上对比，平均 Top‑5 ASR 达 98.2%，仅 1.1% 准确率下降，单问平均查询量低于对手；在5款商用 LLM 上亦能达到 80–100% 的成功率。

**⚠️ 局限性**

仅针对可变更或可注入模板的场景，对完全封闭、硬编码模板的模型效果有限；评估依赖手工标签与模型生成，可能在不同模型版本或更新后效果下降。

---

## 191. Structural Anchors and Reasoning Fragility:Understanding CoT Robustness in LLM4Code

**arXiv ID:** 2604.12214 | [PDF](https://arxiv.org/pdf/2604.12214v1)

**作者:** Yang Liu `[一作]` (Polytechnique Montreal), Foutse Khomh `[通讯]` (Polytechnique Montreal)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 Chain-of-Thought (CoT) 在大语言模型代码生成中的鲁棒性与内部不确定性进行大规模实证研究，系统评估不同模型、提示方式及扰动对 CoT 轨迹与生成质量的影响。

**💡 创新点**

提出了三种结构锚点（推理–代码切换、符号承诺、算法表述）与三类轨迹变形（拉长、分支、简化），并将它们与扰动交互作用联系起来，解释 CoT 产生混合效果的机制；同时证明早期不确定性可作为局部不稳定信号。

**🔧 技术方法**

使用 token‑level 熵与概率差作为不确定性度量，构建结构锚点对齐方法；通过 Pass@k、相对退化 (RD)、AUROC、Spearman ρ 等统计检验评估性能与鲁棒性；利用多模型、多温度、多采样预算的大规模实验。

**📊 数据集**

在 MHPP（多步算法问题）和 BigCodeBench（长文本真实编码任务）两大基准上，分别对原始与字符/单词/句子级扰动的任务描述进行评测。

**📈 对比分析**

实验结果显示 CoT 并非总能提升性能，效果因模型族、任务难度与提示显式程度而异；CoT 与 No‑CoT 在不同扰动下的鲁棒性表现呈现互不相同的模式；早期不确定性对失败有弱预测力；结构锚点与轨迹变形揭示 CoT 受扰动影响的具体机制。

**⚠️ 局限性**

局限性包括：仅评估了开放源码 LLM，无法对闭源 GPT‑4/5 等进行不确定性分析；不确定性指标单一，缺乏更丰富的多模态或语义层面评估；实验聚焦于 docstring 级扰动，未覆盖更广泛的输入变化；模型对不同提示的依赖性仍需进一步研究。

---

## 192. Beyond Scores: Diagnostic LLM Evaluation via Fine-Grained Abilities

**arXiv ID:** 2604.12191 | [PDF](https://arxiv.org/pdf/2604.12191v1)

**作者:** Xu Zhang `[一作]` (National University of Defense Technology), Bo Ding `[通讯]` (National University of Defense Technology)

**通讯引用:** 9321 | [OpenAlex ID](https://openalex.org/A5101888603)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `14d48e9d-0069-4ad9-996a-1d5968216998` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于多维项目反应理论（mIRT）的认知诊断框架，能够对LLM在数学、物理、化学、计算机科学等科学领域的细粒度能力进行评估和诊断。

**💡 创新点**

创新点在于构建了基于布卢姆分类的35维（数学）、27维（物理）、58维（化学）和12维（计算机科学）的能力分类体系，并通过人工+LLM协同生成Q矩阵，实现细粒度诊断与未见题目的性能预测。

**🔧 技术方法**

使用多维IRT、NeuralCD神经网络、Q矩阵构建、人工标注与LLM辅助标注等技术。

**📊 数据集**

使用数学三大基准（MMLU-MATH、MMLU-Pro-MATH、MATH500）以及物理、化学、计算机科学子基准，共评估41个LLM模型。

**📈 对比分析**

与单维IRT、基线准确率、随机预测进行对比；在基准内的AUC在0.80–0.89之间，跨基准为0.77–0.86，均显著优于基线；诊断相关性强，跨基准一致性高。

**⚠️ 局限性**

局限性包括需要人工标注能力分类和Q矩阵，耗时且可能存在主观性；部分维度样本稀疏导致相关性不稳；目前仅覆盖有限学科，需进一步扩展。

---

## 193. Knowledge Is Not Static: Order-Aware Hypergraph RAG for Language Models

**arXiv ID:** 2604.12185 | [PDF](https://arxiv.org/pdf/2604.12185v1)

**作者:** Keshu Wu `[一作]` (Texas A&M University), Yang Zhou `[通讯]` (Texas A&M University)

**通讯引用:** 128486 | [OpenAlex ID](https://openalex.org/A5100364769)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 Order-Aware Knowledge Hypergraph RAG（OKH‑RAG），一种将知识建模为有序高阶超图并将检索视为轨迹推断的检索增强生成框架，旨在提升需要顺序信息的问答与解释任务的推理质量。

**💡 创新点**

创新点在于：① 在超图中引入预cedence结构，使高阶关系具备可学习的顺序属性；② 将检索任务从无序集合检索转化为有序轨迹推断，恢复连贯的交互序列；③ 通过自监督训练学习转移模型来预测超边之间的先后关系，避免了显式时间标注；④ 通过实验验证顺序建模显著提升推理性能，证明无序检索的局限性。

**🔧 技术方法**

技术上主要采用：高阶超图表示（n-ary 关系抽取 + 结构化实体/超边）；预cedence学习的双线性转移模型；序列检索（Beam Search/Viterbi DP）结合相关性、顺序连贯性、阶段覆盖等多项式；检索后对 GPT‑4o 进行检索增强生成；使用 Dense Passage Retrieval 进行初步检索与嵌入。

**📊 数据集**

使用的数据集为 CyPortQA（气旋港口影响评估的问答基准），包含 2,917 个实际案例、145 个港口、90 条风暴，覆盖多时间点的预测与影响信息；同时在论文中对不同类型的问答（TF、MC、SA、TD）进行评估。

**📈 对比分析**

与基线对比：Text‑RAG（文本检索）、GraphRAG（二元图检索）、HyperGraphRAG（无序超图检索）。OKH‑RAG 在所有四类题型上均取得最高准确率，整体准确率提升至 0.534，尤其在多选（MC）和简答（SA）上提升明显。Ablation 研究显示，顺序相关的项（precedence、phase coverage、order coherence）对性能贡献最大。

**⚠️ 局限性**

局限性：① 预cedence学习依赖自监督正负样本，可能在稀疏或不规则数据中表现欠佳；② 超图构建与顺序规则的生成仍需人工或复杂自动化过程，计算成本高；③ 目前仅在气旋港口领域进行验证，缺乏更广泛的跨领域评估；④ 生成环节仍受 LLM 的泛化与推理偏差限制，未解决所有知识一致性问题。

---

## 194. Is Vibe Coding the Future? An Empirical Assessment of LLM Generated Codes for Construction Safety

**arXiv ID:** 2604.12311 | [PDF](https://arxiv.org/pdf/2604.12311v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df`

---

## 195. Mitigating S-RAHA: An On-device Framework to Prevent Forwarding of Re-Captured Images

**arXiv ID:** 2604.12178 | [PDF](https://arxiv.org/pdf/2604.12178v1)

**作者:** Keshav Sood `[一作]` (Deakin University), Praitheeshan Kirupananthan `[通讯]` (Redshield Security Pty Ltd)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于边缘增强深度学习的客户端检测与拦截框架，阻止S-RAHA（屏幕再捕获）导致的图片转发；

**💡 创新点**

创新点包括：零信任的端到端客户端拦截机制、专门设计的EE‑CNN边缘增强网络以及概念化的不可见元数据标识（IMI）用于溯源；

**🔧 技术方法**

技术手段涵盖：EE‑CNN边缘增强卷积网络、量化推理（INT8）、RESTful API + WebSocket、移动端模型部署（TensorFlow Lite/Core ML）以及安全的模型更新与缓存；

**📊 数据集**

数据集：自制1500张原始手机照片与1500张对应的屏幕再捕获图片，覆盖多种显示屏、摄像头、光照与角度；

**📈 对比分析**

与ResNet‑18基线对比，EE‑CNN实现整体准确率98.89%，召回率99.11%，误报率1.78%，推理时间约450 ms，显著优于基线；

**⚠️ 局限性**

局限性：对高端OLED/专业显示器的极佳再捕获场景仍有误检；原图后处理或高压缩会偶尔导致误报；目前仅验证了手机‑屏幕‑手机链，打印或DSLR等其他路径需要进一步评估。

---

## 196. All in One: A Unified Synthetic Data Pipeline for Multimodal Video Understanding

**arXiv ID:** 2604.12335 | [PDF](https://arxiv.org/pdf/2604.12335v1)

**作者:** Tanzila Rahman `[一作]` (University of British Columbia), Leonid Sigal `[通讯]` (University of British Columbia)

**通讯引用:** 10820 | [OpenAlex ID](https://openalex.org/A5053011888)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一套统一的合成多模态视频数据生成管线，能够自动生成带有文本、视频、音频和分割掩码的高质量数据，用于训练多模态大型语言模型（MLLM）

**💡 创新点**

创新点包括：①单一图像通过LLM生成未来情景文本，再用于视频生成，实现多模态一致性；②引入基于视觉问答（VQA）的细调策略，强化模型对视觉证据的推理；③在同一管线中支持对象计数、VQA和分割等多任务的标注，提供统一、可扩展的训练数据；④通过合成数据显著提升模型在真实数据集上的迁移性能

**🔧 技术方法**

使用技术包括：ChatGPT（文本生成）、Wan2.2（视频生成）、VTA-LDM（可选音频生成）、SAM2+MUG-VOS（视频分割掩码生成）、InternVideo2.5（视频基础模型）以及VQA与计数的LLM标注

**📊 数据集**

数据集：从MSCOCO图像采样生成合成视频和标注，并在MSCOCO验证集、LV-VIS 2023、YouTube-VIS以及VQA基准上评估

**📈 对比分析**

与传统仅使用标注文本或说明的细调方法比较，采用VQA+计数标注的合成数据在视频计数、VQA、分割任务上均实现了更低MAE/MSE、更高Clip-Score/WUP以及+5.28 mIoU的提升；在不同规模合成数据（2K vs 5K视频）中，性能随数据量增加而提升

**⚠️ 局限性**

限制：合成视频仍存在少量视觉伪影、动作不自然或标注误差；部分类别在细调后可能出现性能下降；模型对合成与真实域差异的鲁棒性尚需进一步提升

---

## 197. Beyond Weather Correlation: A Comparative Study of Static and Temporal Neural Architectures for Fine-Grained Residential Energy Consumption Forecasting in Melbourne, Australia

**arXiv ID:** 2604.12304 | [PDF](https://arxiv.org/pdf/2604.12304v1)

**作者:** Prasad Nimantha Madusanka Ukwatta Hewage `[一作]` (Victoria University), Hao Wu `[通讯]` (Victoria University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

比较了基于静态天气特征的多层感知器（MLP）与仅使用历史负荷序列的长短期记忆网络（LSTM）在澳大利亚墨尔本两户住宅5分钟级电力负荷预测中的表现，并对结果进行定量评估。

**💡 创新点**

①首次在南半球（墨尔本）对5分钟级住宅STLF进行MLP与LSTM的直接对比；②发现时序自相关信息远优于单日天气特征；③揭示PV系统对天气特征的非对称影响，解释了M​LP在PV户中表现相对更好的机制。

**🔧 技术方法**

使用深度学习框架实现的MLP与LSTM，数据预处理采用Min‑Max归一化和滑动窗口；评估指标包括RMSE、MAE和R²，并与Naïve Persistence和Seasonal Naïve基线进行对比。

**📊 数据集**

数据来源于两户墨尔本住宅的5分钟间隔智能表计（House 3为仅网格负荷，House 4为含PV系统负荷）约117‑119k条记录，结合澳大利亚气象局（BOM）每日6个气象变量作为外部特征。

**📈 对比分析**

采用80/20的时间序列分割，先后训练MLP与LSTM并计算R²。LSTM在House 3测试集上达到R²=0.883，House 4为0.865；MLP在同两户上分别为R²=-0.055和0.410，显示LSTM比天气驱动模型提升了约93%与46%的解释力。相较于Naïve Persistence，LSTM仅在短期（5‑30 min）内获得额外改进。

**⚠️ 局限性**

局限性包括：只使用日级天气数据，缺少小时或更细粒度的气象信息；BOM缺失太阳辐射变量；仅评估两户样本，缺乏跨户泛化验证；未对模型进行多次随机种子实验或置信区间估计；未尝试天气增强LSTM或Transformer；未对PV生成与负荷进行分离预测。

---

## 198. DreamStereo: Towards Real-Time Stereo Inpainting for HD Videos

**arXiv ID:** 2604.12270 | [PDF](https://arxiv.org/pdf/2604.12270v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 199. SubFlow: Sub-mode Conditioned Flow Matching for Diverse One-Step Generation

**arXiv ID:** 2604.12273 | [PDF](https://arxiv.org/pdf/2604.12273v1)

**作者:** Yexiong Lin `[一作]` (University of Sydney), Tongliang Liu `[通讯]` (University of Sydney)

**通讯引用:** 13008 | [OpenAlex ID](https://openalex.org/A5065250332)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 SubFlow，基于子模态条件化的流匹配（Sub-mode Conditioned Flow Matching）解决一阶生成模型的多模态多样性退化问题。

**💡 创新点**

创新点在于：①离线语义聚类将每个类别拆分为细粒度子模态；②在流匹配中加入子模态索引作为条件，消除平均化失真；③方法无需改动现有网络架构，完全 plug‑and‑play，可直接集成至 MeanFlow、Shortcut、SoFlow 等主流一阶模型。

**🔧 技术方法**

技术手段包括：预训练 DINOv3 提取语义特征 + K‑Means 聚类得到子模态标签；Conditional Flow Matching 与 MeanFlow、Shortcut、SoFlow 等流匹配框架；经典的 CFG 指导与子模态先验采样；数值实验在 ImageNet‑256 上进行。

**📊 数据集**

主要使用 ImageNet‑256（约 1.28M 样本，1K 类别）进行训练和评估，指标为 FID、Precision/Recall。

**📈 对比分析**

与三种主流一阶模型（MeanFlow、Shortcut、SoFlow）做 ablation 与基线对比。SubFlow 在 Recall 上提升 5–10%（如 MeanFlow‑B/2 Recall 从 43.45% 提升到 48.84%），FID 维持或略降；单步 NFE 下的多样性与多步模型相当，且往往优于同规模多步模型。

**⚠️ 局限性**

局限性包括：①需先进行离线聚类，K 的选择对性能有影响；②子模态采样若不匹配真实分布，可能导致少数子模态被过度或欠采样；③目前仅处理单一条件标签，未探讨多标签或文本条件的子模态化；④对非常稀疏或高维度子模态的分解效果仍有待进一步验证。

---

## 200. Coding-Free and Privacy-Preserving MCP Framework for Clinical Agentic Research Intelligence System

**arXiv ID:** 2604.12258 | [PDF](https://arxiv.org/pdf/2604.12258v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 201. A Periodic Space of Distributed Computing: Vision & Framework

**arXiv ID:** 2604.12259 | [PDF](https://arxiv.org/pdf/2604.12259v1)

**作者:** Mohsen Amini Salehi `[一作]` (University of North Texas), Rajkumar Buyya `[通讯]` (University of Melbourne)

**通讯引用:** 107185 | [OpenAlex ID](https://openalex.org/A5014716105)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

**🎯 论文内容**

提出了一种基于分布式计算层级与抽象维度的周期空间框架，用来描述、比较以及预测未来分布式系统的发展趋势。

**💡 创新点**

将化学周期表的概念迁移到分布式系统领域，构建横向层级（从本地设备到天空层）与纵向抽象（从裸硬件到智能代理）连续空间，并通过系统属性的趋势映射提供一种可视化的预测工具。

**🔧 技术方法**

主要采用概念建模与框架设计技术，结合现有系统属性指标、趋势分析以及专家访谈进行阐释；并提供了交互式可视化网页用于展示属性在周期空间中的分布。

**📊 数据集**

未使用专门的数据集；论文引用了已有的系统案例和文献作为示例，整体属于理论与愿景性工作。

**📈 对比分析**

通过在周期空间中绘制属性热图和趋势线来比较不同系统的表现，但未给出具体的数值性能评估或基准实验结果。

**⚠️ 局限性**

局限在于缺乏实证验证和量化评估，周期空间的维度划分和定位方法需要进一步细化与社区共识；未展示在真实系统中的落地实现和实际性能提升。

---

## 202. CompliBench: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems

**arXiv ID:** 2604.12312 | [PDF](https://arxiv.org/pdf/2604.12312v1)

**作者:** Jingbo Yang `[一作]` (University of California Santa Barbara), Shiyu Chang `[通讯]` (University of California Santa Barbara)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了 CompliBench 评估框架，针对多轮对话中的合规违规检测，构建了大规模可控合规违规数据集，并评估了 LLM-judge 的性能；

**💡 创新点**

创新点在于：①可扩展的规范生成与违规注入管线，①可通过对抗式优化自动生成难度适中的违规实例；②为每个违规提供精准的标签（违规指令与具体轮次）；③小型 fine‑tuned judge 甚至能超越大模型，揭示模型对违规检测的局限性；

**🔧 技术方法**

使用技术包括：LLM 生成规范与违规变体、交互式 judge‑and‑refine 质量控制、对抗式评估与优化、基于多 LLM 判定的自动标注、以及 Qwen3‑8B 的 SFT 微调；

**📊 数据集**

数据集由 3 个业务域（航空、医疗、保险）共 318 条多轮对话组成，基于真实企业规范扩展与自动生成，且包含每条对话约 3.7 次违规；

**📈 对比分析**

与多种通用 LLM（GPT‑5、GPT‑4o、Gemini‑3‑pro 等）及奖励模型进行对比，评估指标为 SGA、VDA、CLA；实验显示大模型仅在合规轮次上表现尚可，违规检测准确率低；但 fine‑tuned Qwen3‑8B 在 Airline 领域的 CLA 达到 51.47，明显优于所有对手；

**⚠️ 局限性**

局限性包括：①判定器在合规轮次上过于严格、违规轮次过于宽松，导致误判；②关键问题是规范范围归属误判；③目前仅覆盖 3 个域，可能不具备更广泛的泛化性；④对抗式生成仍需人工检查，难以保证所有违规实例均符合真实业务场景。

---

## 203. LiveMoments: Reselected Key Photo Restoration in Live Photos via Reference-guided Diffusion

**arXiv ID:** 2604.12286 | [PDF](https://arxiv.org/pdf/2604.12286v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 204. Local-Splitter: A Measurement Study of Seven Tactics for Reducing Cloud LLM Token Usage on Coding-Agent Workloads

**arXiv ID:** 2604.12301 | [PDF](https://arxiv.org/pdf/2604.12301v1)

**作者:** Justice Owusu Agyemang `[一作]` (Sperix Labs), Kwame Opuni-Boachie Obour Agyekum `[通讯]` (KNUST)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统测评了七种在云端大型语言模型与本地小模型协同工作时降低云端 token 消耗的策略，并在四类真实编码代理工作负载上验证其效果。

**💡 创新点**

创新点在于将七种单独研究过的策略统一实现、交互评估，并揭示最优子集随工作负载不同而变化，提供可直接落地的工作负载驱动选择指南。

**🔧 技术方法**

采用本地 Llama 3.2 3B 与云端 Gemma 3 4B（模拟）相结合的二层路由框架，实现了路由、压缩、缓存、草稿审阅、最小差异、意图抽取和批处理等七种技术。

**📊 数据集**

使用公开数据集生成的四类10条样本合成工作负载，分别对应编辑密集、解释密集、通用聊天和检索增强型工作负载。

**📈 对比分析**

通过对比单一策略、策略对、贪心累加子集与完整七策略的云端 token 节省、成本与质量偏好评估，发现T1+T2组合在大多数工作负载下可实现45–79%的 token 节省，而完整策略在检索增强型负载下可达51%；质量评估表明大部分改进保持与基线相当。

**⚠️ 局限性**

局限性包括仅评估单一模型对（3B 本地 + 4B 模拟云端）、样本量有限、质量评估依赖模型评判而非完整人工评测、仅关注文本工作负载以及未覆盖多模态和真实网络延迟等情况。

---

## 205. GeM-EA: A Generative and Meta-learning Enhanced Evolutionary Algorithm for Streaming Data-Driven Optimization

**arXiv ID:** 2604.12336 | [PDF](https://arxiv.org/pdf/2604.12336v1)

**作者:** Yue Wu `[一作]` (South China University of Technology), Yue-Jiao Gong `[通讯]` (South China University of Technology)

**通讯引用:** 5978 | [OpenAlex ID](https://openalex.org/A5063438356)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 GeM-EA，一种融合二层元学习代理适应与生成回放多岛遗传搜索的流式数据驱动优化算法；

**💡 创新点**

创新点包括：1）将结构参数与非结构参数分离的二层元学习框架，避免梯度冲突并快速初始化代理；2）线性残差补偿提升全局稳定性；3）基于置信度的双向迁移机制保障多岛之间的正向知识迁移；4）使用生成回放而非直接历史检索来缓解负迁移；

**🔧 技术方法**

使用的技术包括：元学习（bi-level meta‑learning）、径向基函数网络代理、线性残差模块、生成回放、多岛遗传算法、置信度驱动的迁移策略；

**📊 数据集**

采用 SDDObench 公开基准数据集（含 8 种多峰函数的多维版本）；

**📈 对比分析**

与 TT‑DDEA、BDDEA‑LDG、MLO、DETO、DSE‑MFS、SAEF‑1GP、DASE 等现有 SDDEA 进行对比，使用离线误差与在线误差两项指标；实验显示 GeM‑EA 在所有指标上平均排名 1.22，显著优于其他方法，尤其在多峰复杂场景中表现突出；

**⚠️ 局限性**

局限性包括：需要手动设定历史岛数量 P 与迁移间隔 τ；目前仅验证单目标场景，尚未扩展到高维或多目标问题；对概念漂移细微变化的鲁棒性待进一步评估；

---

## 206. HyperLiDAR: Adaptive Post-Deployment LiDAR Segmentation via Hyperdimensional Computing

**arXiv ID:** 2604.12331 | [PDF](https://arxiv.org/pdf/2604.12331v1)

**作者:** Ivannia Gomez Moreno `[一作]` (University of California San Diego), Tajana Rosing `[通讯]` (University of California San Diego)

**通讯引用:** 11037 | [OpenAlex ID](https://openalex.org/A5025573294)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出一种基于高维计算（HDC）的轻量级后部署激光雷达语义分割框架，能够在边缘设备上快速自适应更新模型；

**💡 创新点**

首创将HDC应用于后部署LiDAR分割，并设计缓冲区选择策略，仅使用5%点云即可完成自适应，显著提升训练速度；

**🔧 技术方法**

使用高维向量编码、随机投影、HDC分类器及硬样本+随机样本的缓冲区选取；

**📊 数据集**

在SemanticKITTI与nuScenes两个大型自动驾驶数据集上进行评估；

**📈 对比分析**

与CENET、SALUDA以及不使用缓冲区的HyperLiDAR-full对比，mIoU与现有SOTA相当或更优，同时训练速度提升至13.8×，缓冲区策略下FPS约85；

**⚠️ 局限性**

仅适用于有标注数据的监督自适应，对极端域漂移和极低资源硬件的鲁棒性尚未充分验证。

---

## 207. Dialogue Agents that Share Family Information to Strengthen Grandparent-Grandchild Relationships

**arXiv ID:** 2604.12310 | [PDF](https://arxiv.org/pdf/2604.12310v1)

**作者:** Seiya Mitsuno `[一作]` (University of Osaka), Yuichiro Yoshikawa `[通讯]` (University of Osaka)

**通讯引用:** 3815 | [OpenAlex ID](https://openalex.org/A5039236080)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本研究提出了一种能够在对话中互相分享家庭信息的聊天机器人，旨在通过让祖辈和孙辈在每日交谈中获取对方日常生活细节，增强两代人之间的情感联系，并减轻老年人的社交孤立与焦虑。

**💡 创新点**

创新点在于将信息共享功能嵌入对话代理，构建双向人机、人机之间的桥梁，首次系统性评估共享信息如何提升老年人与孙辈的亲密度和使用者的情绪健康；同时将基于规则与大语言模型的混合对话策略与智能信息推荐相结合。

**🔧 技术方法**

技术实现包括：①基于LINE Messaging API的聊天机器人平台；②结合规则模板与OpenAI GPT模型生成对话回复；③Google Natural Language API进行情感与实体分析；④SQL数据库存储对话历史与用户个人信息；⑤基于用户起居时间自适应的对话触发与提醒机制。

**📊 数据集**

使用的数据集为自建的用户-代理问答配对集合，搭配从Cookpad抓取的食品名称列表；此外采集了实验参与者在10天内的对话日志和问卷数据。

**📈 对比分析**

实验采用对照设计（Sharing vs Non‑sharing）共104对祖孙。比较指标包括使用意愿（Intention to Use）、响应时间、提醒次数、IOS亲密度、HAD焦虑/抑郁量表。结果显示：共享条件下老年人提醒次数显著降低（p=0.013），说明交互积极性提升；开放式问答编码表明共享组报告更多正向亲密感提升；两组在焦虑量表上均呈下降趋势，且共享组无显著差异，表明单次10天实验对情绪改善效果有限。

**⚠️ 局限性**

局限性包括：①实验时长仅10天，可能不足以观察关系与情绪的长期累积效应；②信息共享范围局限于日常轻量话题，未触及更私密情感层面，可能降低深度互动；③对隐私与数据安全的控制机制有限，需在未来工作中加入更细粒度的授权与透明度设计；④样本规模与文化背景局限于日本，普适性尚待验证。

---

## 208. GAM: Hierarchical Graph-based Agentic Memory for LLM Agents

**arXiv ID:** 2604.12285 | [PDF](https://arxiv.org/pdf/2604.12285v1)

**作者:** Zhaofen Wu `[一作]` (Zhejiang University), Hongwei Wang `[通讯]` (Zhejiang University)

**通讯引用:** 12720 | [OpenAlex ID](https://openalex.org/A5100357132)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了层次化图结构的代理记忆框架GAM，利用事件进展图缓冲对话并在语义边界触发时再合并到主题关联网络，形成写入隔离与语义稳定并存的机制；

**💡 创新点**

创新点在于将记忆生命周期拆分为“事件缓冲”和“语义合并”两阶段，并通过LLM判别语义漂移实现状态切换；同时设计了跨层图引导的多因子检索，融合时间、置信度和角色信号提升检索精度；

**🔧 技术方法**

使用图神经网络式的层次化图记忆架构、LLM驱动的语义判别、图引导多因子检索算法，以及基于向量相似度的候选缩放策略；

**📊 数据集**

在LoCoMo（长开放域对话推理）和LongDialQA（多方剧本式对话）两个基准上进行评测；

**📈 对比分析**

与MemoryOS、Mem0、MemGPT、A‑Mem等统一流式记忆与离散结构记忆基线比较，GAM在LoCoMo上平均F1与BLEU-1均领先约15‑20%，在LongDialQA上平均F1提升约30%，同时保持较低的Token消耗和合理的推理延迟；

**⚠️ 局限性**

目前仅实现文本记忆，缺乏多模态节点，无法利用视觉或音频信息；此外，语义判别阈值依赖LLM提示，可能对不同对话风格的鲁棒性有限；

---

## 209. Bridging the Micro--Macro Gap: Frequency-Aware Semantic Alignment for Image Manipulation Localization

**arXiv ID:** 2604.12341 | [PDF](https://arxiv.org/pdf/2604.12341v1)

**作者:** Xiaojie Liang `[一作]` (Sun Yat-sen University), Wei Lu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 43421 | [OpenAlex ID](https://openalex.org/A5066881061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FASA框架，统一定位传统与扩散模型生成的伪造区域。

**💡 创新点**

将频率感知与语义对齐结合，跨尺度注入语义先验并用原型引导掩码解码，填补微观频率与宏观语义的鸿沟。

**🔧 技术方法**

自适应双频DCT、冻结CLIP与Patch‑Level Semantic Alignment、Semantic‑Frequency Side Adapter、Prototype‑Guided Mask Decoder等技术。

**📊 数据集**

在OpenSDI（扩散编辑）以及CASIAv1、Columbia、Coverage、NIST16等传统伪造数据集上训练与测试。

**📈 对比分析**

与14个SOTA方法对比，OpenSDI平均像素F1达0.578、平均IoU 0.501，传统数据集平均F1 0.712、AUC 0.945，均为最佳，并在跨生成器/跨数据集泛化及降质鲁棒性上显著优于对手。

**⚠️ 局限性**

对极端高分辨率或多模态伪造的适应性仍有限，且模型训练与推理成本相对较高。

---

## 210. Turán-Theoretic Bounds on Several Elementary Trapping Sets in LDPC Codes

**arXiv ID:** 2604.12332 | [PDF](https://arxiv.org/pdf/2604.12332v1)

**作者:** Ziyang Zhao `[一作]` (University of Chinese Academy of Sciences), Guiying Yan `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 6528 | [OpenAlex ID](https://openalex.org/A5100979503)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文通过推导θ图、哑铃图以及带弦的短环等特殊图的Turán数，利用这些极值图论结果对LDPC码中的基本捕获集（ETS）进行理论界定；随后对受限结构下ETS的谱半径进行分析，并用QC-PEG-CYCLE算法构造了两种满足该约束的QC‑LDPC码，实验验证其误码率低于常规方法。

**💡 创新点**

创新点在于将极值图论（Turán数）与LDPC码设计结合，首次给出在girth=8、8‑环不共享变量节点以及去除短环弦结构等条件下的ETS大小下界；同时证明这些结构约束能有效降低ETS的谱半径，从而减少误码率；并通过理论与实验相结合展示了改进的误码性能。

**🔧 技术方法**

使用技术包括：极值图论（Turán数计算与递推）、图论构造与分析（θ图、哑铃图、短环弦），线性状态空间模型与谱半径计算，QC‑LDPC码的指数矩阵构造（QC‑PEG‑CYCLE），以及误码率仿真（FER曲线）。

**📊 数据集**

实验使用自构造的两组QC‑LDPC码（指数矩阵 H1 与 H2），不依赖公开数据集，所有图和码均在论文中给出。

**📈 对比分析**

通过与PEG、GCD‑based、双线性序列逆转和Simple‑Form等传统方法在FER曲线上的对比，展示受限结构的码在误码率（尤其是误码率下限）上优于对照组；谱半径统计表明受限ETS的平均和中位值均低于非受限ETS。

**⚠️ 局限性**

局限性包括：给出的下界在小规模a时不一定最优；对更大参数的ETS存在性尚未完全验证；理论仅针对特定约束（girth=8、8‑环不共享变量节点等），无法直接推广到所有LDPC码；以及构造算法对码长度和复杂度的影响尚需进一步研究。

---

## 211. Defining and Evaluation Method for External Human-Machine Interfaces

**arXiv ID:** 2604.12293 | [PDF](https://arxiv.org/pdf/2604.12293v1)

**作者:** Jose Gonzalez-Belmonte `[一作]` (University of Michigan-Dearborn), Jaerock Kwon `[通讯]` (University of Michigan-Dearborn)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并验证了一套通用的外部人机界面评估方法，并对五种主流eHMI方案（无eHMI、前制动灯、骑士·里德显示、微笑显示、文字显示）进行量化评分。

**💡 创新点**

创新点在于将标准化、成本效益、可访问性、易理解、持续沟通、定位与可读性等七维度整合为可量化问卷，并给出综合得分公式，提供了一个客观可比的评估框架。

**🔧 技术方法**

采用问卷设计、数值计算公式以及Unity可视化模拟与多角度可视性算法来完成评分与定位分析。

**📊 数据集**

使用2022年公开的成本数据、de Winter等研究的实验数据（如感知安全率）以及公开价格信息作为评估输入。

**📈 对比分析**

通过计算各维度得分并加权求和得到0-70分的总分，结果显示无eHMI+文字显示方案获得最高分约46%，表明该组合方案在易理解与可访问性上表现最好。

**⚠️ 局限性**

局限性包括缺乏不同年龄、群体对eHMI学习时长的数据、可读性在多种天气/光照/距离/角度下的实测不足、成本估算可能与制造商实际成本偏差、以及仅评估了“让行”这一单一沟通任务。

---

## 212. Deep Situation-Aware Interaction Network for Click-Through Rate Prediction

**arXiv ID:** 2604.12298 | [PDF](https://arxiv.org/pdf/2604.12298v1)

**作者:** Yimin Lv `[一作]` (Institute of Software, Chinese Academy of Sciences), Dong Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

研究了CTR预测中用户行为序列的情境特征，提出DSAIN模型通过行为去噪、情境特征编码、三向相关融合和情境聚合来提升CTR预测精度。

**💡 创新点**

引入情境概念并通过重参数化的去噪、微MLP情境特征编码与多层三向混合器实现对行为序列中多维情境信息的高质量表示及其交叉关联建模。

**🔧 技术方法**

使用Gumbel-Softmax重参数化去噪、嵌入层+微MLP编码、三方向(行为-通道-特征)MPL混合器、加权情境聚合以及最终的MLP进行CTR预测。

**📊 数据集**

在淘宝、Eleme公开数据集和美团工业数据集（共计约12亿交互记录）上进行离线实验，并在美团上进行线上A/B测试。

**📈 对比分析**

与DIN、DIEN、CAN、DMT、FeedRec、CARCA、Trans2D、DIF‑SR等8种基线对比，DSAIN在AUC、Logloss均领先；线上测试CTR提升2.70%、CPM 2.62%、GMV 2.16%。

**⚠️ 局限性**

对行为序列长度和多维情境维度的设定依赖超参数，模型复杂度相对较高，且在极大规模数据上仍需验证其可扩展性。

---

## 213. LightMat-HP: A Photonic-Electronic System for Accelerating General Matrix Multiplication With Configurable Precision

**arXiv ID:** 2604.12278 | [PDF](https://arxiv.org/pdf/2604.12278v1)

**作者:** Hailong Gong `[一作]` (Australian National University), Rajkumar Buyya `[通讯]` (University of Melbourne)

**通讯引用:** 107185 | [OpenAlex ID](https://openalex.org/A5014716105)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一种混合光电子矩阵乘法加速器 LightMat-HP，并实现了可配置精度的通用矩阵乘法（GEMM）加速。

**💡 创新点**

创新点包括：1) 采用块浮点（BFP）表示，将指数共享以降低指数运算成本；2) 通过切片（slicing）把高位数乘法拆解为低位数光学乘法并在数字域累加，实现高精度；3) 设计基于块的矩阵切块（tile）数据流，使任意尺寸矩阵可并行计算；4) 将光学乘法与电子控制深度耦合，形成端到端的混合体系。

**🔧 技术方法**

使用的技术包括：光学调制器（MZMs）+光波分复用（WDM）实现光学乘法；DAC/ADC 4.096 GS/s 进行光与电的转换；块浮点（BFP）格式与切片算子；数字后处理单元（DPPU）进行累加、指数恢复；基于FPGA（Xilinx ZCU111）实现调度、切块、序列化、结果重构。

**📊 数据集**

实验使用人工生成的随机矩阵（16×16到1024×1024），FP32 作为基准数值格式；没有使用特定的机器学习或科学计算数据集，而是以矩阵规模作为评测维度。

**📈 对比分析**

与 Intel i5 CPU、NVIDIA RTX 3060 GPU、Xilinx FPGA 以及光学加速器 BITLUME 进行对比。通过仿真和硬件实验，LightMat-HP 在 1024×1024 的 GEMM 上实现了约 6.6 TFLOPS 的峰值吞吐率、39 GFLOPS/W 的能效，较 CPU、GPU、FPGA 的吞吐率提高 10-30 倍，能效提升 1.4-2.0 倍；在小到中等矩阵尺寸上也保持了显著的速度与低延迟优势。

**⚠️ 局限性**

主要局限包括：光学乘法精度受限于器件线性与噪声，需依赖低位数运算与数字累加；DAC/ADC 以及激光功率是系统能耗与面积的主要瓶颈；光学调制器和光探测器尺寸大、功耗高；系统需要大量校准与环境补偿，难以在大规模集成或长时间运行中保持稳定；目前仅针对矩阵乘法，尚未验证对更复杂工作负载（如卷积、注意力层）的通用性。

---

## 214. CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades

**arXiv ID:** 2604.12262 | [PDF](https://arxiv.org/pdf/2604.12262v1)

**作者:** Raeyoung Chang `[一作]` (Sogang University), Nikhil Verma `[通讯]` (LG Electronics)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了CascadeDebate系统，在LLM级联架构中于每个升级边界插入多代理推理，以解决单模型在不确定性高时过早升级的问题，提升准确率并降低成本。

**💡 创新点**

创新点在于：①将选择性多代理协商嵌入级联路由的升级边界，实现内部纠错与外部升级的分层决策；②采用在线阈值学习动态调整置信度门限，使系统对不同任务和分布自适应；③统一框架在不同模型尺度间交替单模型推理和多代理协商，兼顾效率与性能。

**🔧 技术方法**

核心技术包括置信度估计（模型自评与多代理一致率）、贝叶斯逻辑回归校准、基于角色提示的多代理共识、软门限阈值学习（使用sigmoid + Adam优化）、多模型级联与人类专家最终回退。

**📊 数据集**

实验使用了五个多项选择基准：ARC‑Easy、ARC‑Challenge、MMLU、MedQA 与 MedMCQA，分别覆盖科学、常识与医学领域。

**📈 对比分析**

与单模型、单模型级联、单模型+多代理、标准级联等基线进行对比。实验结果表明CascadeDebate在所有任务上均优于基线，最高提升约26.75%，在成本‑准确率 Pareto 前沿占据优势；在Llama‑3.2 与 Qwen2.5 两组模型上均实现显著性能提升。

**⚠️ 局限性**

局限性包括：1) 级联与多代理推理的串行执行导致延迟高；2) 误差传播风险，早期错误决策可能阻止后续升级；3) 代理同质性（仅依赖角色提示）限制了多样性，可能无法充分覆盖复杂推理空间。

---

## 215. SpanKey: Dynamic Key Space Conditioning for Neural Network Access Control

**arXiv ID:** 2604.12254 | [PDF](https://arxiv.org/pdf/2604.12254v1)

**作者:** WenBin Yan `[一作]` `[通讯]` (University of Colorado Boulder), WenBin Yan (University of Colorado Boulder)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种名为SpanKey的轻量级推理门控机制，通过在前向传播中使用秘密密钥来调节激活，而不需要加密权重或追求排行榜上的准确性。

**💡 创新点**

创新点在于使用低维密钥子空间Span(B)进行密钥注入，并通过多层设计来实现推理门控，强调了密钥吸收现象及其对基线分离的影响。

**🔧 技术方法**

采用了动态密钥生成、加法和乘法注入等技术，结合多层注入设计。

**📊 数据集**

使用了CIFAR-10和MNIST数据集进行实验，评估了不同模式下的性能。

**📈 对比分析**

与相关方法比较时，SpanKey在使用正确密钥时表现良好，但在错误密钥下的性能显著下降，显示出密钥吸收现象的影响。

**⚠️ 局限性**

限制在于未能保证强分离性，且在训练过程中可能出现密钥吸收现象，导致模型对错误密钥的敏感性降低。

---

## 216. Self-Adversarial One Step Generation via Condition Shifting

**arXiv ID:** 2604.12322 | [PDF](https://arxiv.org/pdf/2604.12322v1)

**作者:** Deyuan Liu `[一作]` (Westlake University), Tao Lin `[通讯]` (Westlake University)

**通讯引用:** 10696 | [OpenAlex ID](https://openalex.org/A5061697130)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `f86bf285-fd08-4156-973b-6e6481af8fa0` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种自监督的单步图像生成框架，利用条件偏移生成内在对抗信号，完成无判别器的高质量文本到图像合成。

**💡 创新点**

核心创新在于通过对条件空间做仿射偏移构造伪条件，使模型内部自行估计假分布的速度场，从而得到常数权重的GAN对齐梯度，既消除了判别器带来的训练不稳定，又兼容大型预训练骨干和LoRA微调。

**🔧 技术方法**

技术包括流匹配（flow matching）与概率流ODE、Score-velocity duality、混合一致性损失（mix consistency loss）以及对抗性修正梯度的理论推导。

**📊 数据集**

使用公开数据集BLIP‑3o、ShareGPT‑4o以及合成的Qwen‑Image 20B自制数据，共计约 1M 训练样本。

**📈 对比分析**

在GenEval、FID/CLIP等指标上，与FLUX‑Schnell 12B、TwinFlow等多步模型对比，0.6B模型单步达到0.84 GenEval，20B LoRA单步突破0.89 GenEval，速度和延迟分别为0.20s（≈7.3/s）和0.39s，明显优于传统单步或两步方法。

**⚠️ 局限性**

局限性包括需要手动调节条件偏移参数、对OT路径假设的依赖、对大型模型的扩展仍受限于计算资源，以及在极低采样步数下可能仍无法完全匹配多步生成器的极致细节表现。

---

## 217. ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection

**arXiv ID:** 2604.12321 | [PDF](https://arxiv.org/pdf/2604.12321v1)

**作者:** Boyang Li `[一作]` (East China Normal University), Fang Zhou `[通讯]` (East China Normal University)

**通讯引用:** 2365 | [OpenAlex ID](https://openalex.org/A5048127669)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出ToxiTrace框架，使用BERT风格编码器与CuSA、GCLoss、ARCL三组件，实现中文毒性文本的句子分类与可解释的毒性跨度提取。

**💡 创新点**

创新点在于：①利用CuSA将模型梯度提示与LLM细化弱监督跨度；②通过GCLoss对梯度进行约束，使毒性标记更聚焦；③采用ARCL对抗性推理对比学习强化毒性与非毒性语义边界；④设计BiCSE双向扫描算法获取连续毒性跨度。

**🔧 技术方法**

采用BERT/中文RoBERTa、MacBERT编码器；梯度归因、GCLoss、ARCL、InfoNCE对比学习；LLM（Gemini 2.5 Pro）进行跨度细化；BiCSE双向扫描算法。

**📊 数据集**

使用COLD、ToxiCN句子级别数据集和CNTP毒性跨度数据集进行实验；对比OpenAI、Llama、Qwen等LLM及其SFT版本。

**📈 对比分析**

与RoBERTa、MacBERT、LLM直接推理、SFT及解释方法（CRF、LIME、Attention、IG）对比，ToxiTrace在COLD/ToxiCN上Acc/Recall/Precision/F1/Macro-F1均优于对照；毒性跨度提取F1提升约12%，且推理时间比大型LLM快约7倍。

**⚠️ 局限性**

局限性：仅针对中文字符语言，词基语言效果未知；对同音字、拼音混淆等隐蔽毒性鲁棒性不足；LLM适配受LoRA参数限制；跨语言推广未验证。

---

## 218. Labeled TrustSet Guided: Batch Active Learning with Reinforcement Learning

**arXiv ID:** 2604.12303 | [PDF](https://arxiv.org/pdf/2604.12303v1)

**作者:** Guofeng Cui `[一作]` (Amazon), Zhu Liu `[通讯]` (Amazon)

**通讯引用:** 144 | [OpenAlex ID](https://openalex.org/A5100380643)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于强化学习的批量主动学习框架 BRAL‑T，利用已标注数据构建 TrustSet 并训练 RL 策略在未标注池中选择最具信息量的样本。

**💡 创新点**

创新点包括：① 用梯度范数（GradNd/EL2N）和超损失（SuperLoss）从已标注样本中挖掘最有价值的“TrustSet”，并强制类别平衡；② 将 TrustSet 通过强化学习近似映射到未标注池，显著降低标注与重训练成本；③ 结合 Wasserstein 距离的 RL 策略与批量主动学习，形成端到端高效流程。

**🔧 技术方法**

采用的技术包括：GradNd/EL2N 评分、SuperLoss 课程学习、基于 Wasserstein 距离的 DQN 强化学习策略、批量主动学习框架、ResNet18/DeiT-Small 等模型训练与 fine‑tune。

**📊 数据集**

使用的数据集有：Cifar10、Cifar100、Cifar10‑imb、EMNIST、FashionMNIST、BreakHis、Pneumonia‑MNIST、Waterbird（8 组图像分类基准）；Cifar10‑LT、Cifar100‑LT（长尾数据集）；以及 Cifar10‑imb、TinyImageNet（Fine‑tune 任务）。

**📈 对比分析**

与 LossPrediction、WAAL、RandomSample、CoreSet、Cluster‑Margin、BALD 等方法对比，实验表明 BRAL‑T 在 AUBC（area under accuracy‑budget curve）和最终准确率（F‑acc）上均优于所有基线，尤其在长尾数据集和 Fine‑tune 任务中取得最佳性能。

**⚠️ 局限性**

局限性：① 需要在每轮主动学习中重新训练 RL 策略，增加训练开销；② 对 RL 奖励函数和聚类特征的设计较为敏感；③ 在标签噪声极大或未标注池质量差的场景下鲁棒性尚未充分验证。

---

## 219. WebAgentGuard: A Reasoning-Driven Guard Model for Detecting Prompt Injection Attacks in Web Agents

**arXiv ID:** 2604.12284 | [PDF](https://arxiv.org/pdf/2604.12284v1)

**作者:** Yulin Chen `[一作]` (National University of Singapore), Bryan Hooi `[通讯]` (National University of Singapore)

**通讯引用:** 5817 | [OpenAlex ID](https://openalex.org/A5065675832)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一套并行守护框架，在 Web Agent 运行时引入独立的多模态守护模型 WebAgentGuard，专门用于检测和阻止 prompt injection 攻击；

**💡 创新点**

核心创新在于：1) 把安全检测与任务推理解耦，守护模型与 Agent 并行工作；2) 采用基于推理的训练策略（SFT+GRPO）和多模态数据（HTML+截图+指令），显著提升检测精度；3) 构建覆盖 164 个主题、230 种 UI 风格的合成数据集，解决了真实数据采集难题；

**🔧 技术方法**

技术手段包括：使用 GPT‑5 生成合成网页及注释推理链；基于 Vision‑Language 模型（如 Qwen3‑VL‑Instruct）进行 SFT（带推理模板）并进一步采用 Group Relative Policy Optimization (GRPO) 进行 RL 微调；并在 Web Agent 与 Guard 之间实现并行推理与动作许可机制；

**📊 数据集**

使用 GPT‑5 生成的合成多模态数据集，涵盖 164 个主题和 230 种 UI 设计风格；该数据集包含正负样本、用户指令、HTML、截图及推理链，分为 SFT、RL 和评测集；

**📈 对比分析**

与 GPT‑4o、GPT‑4.1、GPT‑4o‑Mini、Llama‑3.2‑Vision‑Instruct‑11B、Qwen‑系列、Llama‑Guard、Prompt‑Guard、GuardReasoner 等基线进行对比；在内部评测集上 recall 超过 99%，在 VPI‑Bench、EIA 及 PopUp 等外部基准上平均 recall 超过 90%，并在多种 Web Agent 上将攻击成功率降至 0%；在性能上守护模型推理时间仅 2–3 s，低于 Agent 动作生成时间；

**⚠️ 局限性**

主要限制：训练过程需要额外算力，且仅在黑盒攻击场景下评估；未考虑白盒攻击和动态页面/JavaScript 触发的高级注入；未来工作需进一步验证跨域和实时攻击的鲁棒性。

---

## 220. Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Reasoning

**arXiv ID:** 2604.12282 | [PDF](https://arxiv.org/pdf/2604.12282v1)

**作者:** Houxing Ren `[一作]` (CUHK MMLab), Hongsheng Li `[通讯]` (CUHK MMLab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个两阶段多代理框架，对大型电子表格进行增量式读取和推理。

**💡 创新点**

通过结构草图与验证模块，既保留布局语义，又避免一次性加载超长表格。

**🔧 技术方法**

结合LLM、代码执行、视觉图像与表格转换工具实现多模态增量推理。

**📊 数据集**

使用SpreadsheetBench和RealHiTBench这两个真实电子表格基准。

**📈 对比分析**

相较于ChatGPT Agent等基线，在SpreadsheetBench上取得约38.16%准确率，超出基线2.89个百分点。

**⚠️ 局限性**

受限于对强大多模态LLM的依赖，模型可复现性和小模型迁移仍有待提升。

---

## 221. Asymptotically Stable Gait Generation and Instantaneous Walkability Determination for Planar Almost Linear Biped with Knees

**arXiv ID:** 2604.12274 | [PDF](https://arxiv.org/pdf/2604.12274v1)

**作者:** Fumihiko Asano `[一作]` (Japan Advanced Institute of Science and Technology), Taiki Sedoguchi `[通讯]` (Japan Advanced Institute of Science and Technology)

**通讯引用:** 6 | [OpenAlex ID](https://openalex.org/A5106356459)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文设计并分析了一种在臀部中心平衡的平面6-DOF双足机器人，推导其运动方程、碰撞与控制，随后将其降维至3-DOF并对重力项做线性近似，构建可离散化的CLRed模型，实现未来状态的即时计算与步态可行性判断。

**💡 创新点**

创新点在于：①利用机器人腿部在臀部平衡的几何结构，使惯性矩恒定、无非线性速度项，简化动力学；②通过对重力项在合适展开点的线性化，获得高精度的CLRed模型；③将该模型离散化后实现无需数值积分即可在毫秒级完成未来状态和步长、周期等步态指标的计算，从而实现即时的步态可行性评估。

**🔧 技术方法**

使用的技术包括：运动方程的拉格朗日/约束分析、冲击动力学、PD/时域轨迹规划、线性化与状态空间实现、离散化与闭式解、二分法求解碰撞时刻、仿真评估与参数敏感性分析。

**📊 数据集**

本文未使用公开数据集，而是通过自行搭建的数值仿真平台，对不同β、展开点θ₂*以及降阶步态进行参数扫描和性能对比。

**📈 对比分析**

对比方法：将CLRed模型的步长、周期、角速度等与完整非线性模型（通过数值积分得到）在相同初始条件下进行对比。结果显示，选择θ₂* = -0.25β 时，CLRed模型在步长、周期上的误差低于1%，而计算时间从几分钟降至毫秒级；在小阶梯下，CLRed模型能通过调整T_set快速判定可行性。

**⚠️ 局限性**

局限性包括：①当机器人面临降阶梯等非平面地形时，单纯调整T_set不足以保证步态稳定，需进一步设计自适应控制；②模型假设腿部在臀部平衡，缺乏自然摆动和膝关节伸展/屈曲动力学，可能限制在更复杂环境中的表现；③在每一次碰撞时需重新计算控制参数（T_set、α、β等），实现上增加了实时调参复杂度。

---

## 222. Decentralized Learning via Random Walk with Jumps

**arXiv ID:** 2604.12260 | [PDF](https://arxiv.org/pdf/2604.12260v1)

**作者:** Zonghong Liu `[一作]` (Rutgers University), Salim El Rouayheb `[通讯]` (Rutgers University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了无中心化的随机游走学习，并提出了利用 Lévy 跳跃的 Metropolis‑Hastings 变体以缓解“陷阱”问题，理论分析并实验验证其收敛性能。

**💡 创新点**

引入 Lévy 跳跃扰动破坏平衡条件以逃离网络中重要节点的陷阱，并给出了结合数据异质性、谱间隙和跳跃概率的收敛率分析。

**🔧 技术方法**

以随机游走 SGD、Metropolis‑Hastings 转移矩阵、Lévy 跳跃、谱间隙分析、辅助序列证明、凸/强凸理论等技术实现。

**📊 数据集**

采用合成线性回归数据（同质与异质两种分布），并在随机图、环图、二维网格、Watts‑Strogatz 等稀疏网络上进行实验。

**📈 对比分析**

与统一采样、加权采样及混合采样的随机游走学习对比，MHLJ 在异质数据下显著加速收敛并消除陷阱，收敛速度接近集中式加权 SGD。

**⚠️ 局限性**

仍存在跳跃引起的采样偏差导致的误差间隙，需要细致调节跳跃概率；跳跃略微增加通信开销；理论依赖谱间隙，对极度稀疏或不连通网络适用性有限。

---

## 223. Style-Decoupled Adaptive Routing Network for Underwater Image Enhancement

**arXiv ID:** 2604.12257 | [PDF](https://arxiv.org/pdf/2604.12257v1)

**作者:** Hang Xu `[一作]` (Wuhan University), Zhen Dong `[通讯]` (Wuhan University)

**通讯引用:** 25699 | [OpenAlex ID](https://openalex.org/A5100429975)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种基于风格解耦与自适应路由的水下图像增强框架SDAR-Net

**💡 创新点**

通过将降解风格与场景结构分离，并利用自适应路由动态调节增强过程，克服了统一映射导致的过度增强或不足恢复问题

**🔧 技术方法**

使用风格解耦训练、SREU递归增强单元、Ada-Route软权重路由、Gram矩阵风格匹配、重建损失、伪标签自评与交叉熵损失等技术

**📊 数据集**

在UIEB、LSUI两大配对数据集上进行训练与评估，并在SUIM语义分割数据集上验证下游任务效果

**📈 对比分析**

与多种单阶段、双阶段、GAN、物理先验等SOTA方法对比，SDAR-Net在PSNR、SSIM等全参考指标上取得最高分（PSNR 25.72 dB），同时在无参考指标和下游分割任务中表现同样优异

**⚠️ 局限性**

依赖于手工设计的风格与结构解耦与路由策略，可能对不同环境（如极端光照、动态景物）适应性仍有限，且递归迭代次数K=2已足够，进一步扩展可能导致计算开销不增益

---

## 224. ARGen: Affect-Reinforced Generative Augmentation towards Vision-based Dynamic Emotion Perception

**arXiv ID:** 2604.12255 | [PDF](https://arxiv.org/pdf/2604.12255v1)

**作者:** Huanzhen Wang `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**通讯引用:** 3538 | [OpenAlex ID](https://openalex.org/A5100669255)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了 ARGen 框架，结合 AU 先验注入与强化学习调度的两阶段动态表情视频生成与数据增强方法。

**💡 创新点**

创新点在于：① 用 AU 语义构建 RAG 风格的情感知识图并通过视觉语言模型生成可解释的情感提示；② 将多维奖励（时序、一致性、面部完整度、表情幅度）融合进 diffusion 过程，并用强化学习自适应选择采样步数与指导曲线，解决长尾稀缺类别的数据不足。

**🔧 技术方法**

技术手段包括：扩散模型（TI2V 与 ARD 的强化学习调度）、视觉语言模型 Qwen2.5‑7B‑VL、RAG 检索增强、Action Units（AU）语义知识图、策略梯度强化学习、文本与图像条件结合的生成管线。

**📊 数据集**

使用公开的动态表情数据集：CK+、DFEW、FERV39k，用于生成质量评估和识别性能验证。

**📈 对比分析**

与多种生成/识别基线（Res18+LSTM/Transformer、VGG+LSTM/Transformer、DFER‑CLIP、S4D 等）在 FVD、sFVD、tFVD 上显著降低，识别指标 UAR/WAR、准确率提升约 1–6% 甚至超过 10%，尤其对稀缺情绪（惊讶、厌恶、恐惧）表现突出。

**⚠️ 局限性**

局限性包括：仍受采样策略稳定性影响，对极端长尾稀缺类别的生成效果有限；目前仅覆盖七种基本情绪，未考虑跨模态或非基本情绪；依赖参考图像和文本提示，推断时需要一定先验信息。

---

## 225. UniDetect: LLM-Driven Universal Fraud Detection across Heterogeneous Blockchains

**arXiv ID:** 2604.12329 | [PDF](https://arxiv.org/pdf/2604.12329v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 226. Models Know Their Shortcuts: Deployment-Time Shortcut Mitigation

**arXiv ID:** 2604.12277 | [PDF](https://arxiv.org/pdf/2604.12277v1)

**作者:** Jiayi Li `[一作]` (Carnegie Mellon University), Carl Kingsford `[通讯]` (Carnegie Mellon University)

**通讯引用:** 25109 | [OpenAlex ID](https://openalex.org/A5113653378)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个部署时的短路防护框架——Shortcut Guardrail，利用梯度归因识别并消除文本模型在推理时对表面特征的依赖；

**💡 创新点**

创新点在于：①不需要训练数据或短路标签即可在部署阶段检测短路；②通过 Masked Contrastive Learning 对 LoRA 模块进行微调，使模型在去掉关键短路词时保持表示一致；③通过单一可调参数 α 在少量支持样本上完成校准；

**🔧 技术方法**

核心技术包括梯度 × 输入归因、LoRA 轻量级适配器、Masked Contrastive Learning（MaskCL）对比损失以及基于小样本的 α 校准；

**📊 数据集**

使用的基准数据集包括自然出现短路的 SST‑2（情感分类）、CivilComments（毒性检测）和 MultiNLI（自然语言推断），以及通过注入可控短路词构造的 Yelp‑ST、Yelp‑Syn、GoEmo‑ST、GoEmo‑Syn、改造后的 CivilComments 与 MultiNLI；

**📈 对比分析**

与 ERM、JTT、NFL、DFR 等训练时短路缓解方法对比，Shortcut Guardrail 在多种任务和短路强度下都能显著提升 Worst‑Group Accuracy（WGA）并保持或略低于整体准确率，MSTPS 指标显示其有效减少了对单词的过度敏感；

**⚠️ 局限性**

局限性包括：①部署时方法受训练数据信息瓶颈限制；②仅能捕捉单词级短路，无法处理多词组合短路；③仍需少量标注支持集进行 α 校准；④实验仅覆盖 BERT‑类文本分类任务，未验证在其他任务或模型架构上的泛化。

---

## 227. CodeSpecBench: Benchmarking LLMs for Executable Behavioral Specification Generation

**arXiv ID:** 2604.12268 | [PDF](https://arxiv.org/pdf/2604.12268v1)

**作者:** Zaoyu Chen `[一作]` (Hong Kong Polytechnic University), Xiao-Ming Wu `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 7456 | [OpenAlex ID](https://openalex.org/A5101981128)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 CodeSpecBench 基准，评估大模型生成可执行行为规范（前置条件和后置条件）的能力，并提供功能级与仓库级两种任务与基于执行的评估流程。

**💡 创新点**

创新点在于：①首次将可执行的 Python 规范作为评估目标；②覆盖功能级与多文件仓库级两层复杂度；③同时度量规范的正确性与完整性，使用实际执行而非仅靠形式验证；④构建多样化、规模更大的真实项目数据集。

**🔧 技术方法**

使用大模型生成规范和测试用例，采用 LLM 驱动的测试生成（GPT‑4o、DeepSeek‑V3、Qwen‑Max 等），在 SWE‑bench + UTBoost 环境中插桩执行验证，并对生成结果进行正确性/完整性计数。

**📊 数据集**

功能级数据来自 LeetCodeDataset（≈2,500 题），仓库级数据来自 SWE‑bench Verified（500 个多文件 Python 项目）并使用 UTBoost 生成的增强测试。

**📈 对比分析**

对 15 个主流 LLM（Claude‑4.5‑Sonnet、Gemini‑2.5‑Pro、GPT‑5‑mini 等）进行 Pass‑Rate、Correctness 与 Completeness 的对比；功能级最高 Pass‑Rate 约 47%，仓库级仅 20%，显示仓库级任务更难，且多数模型在完整性与正确性之间存在明显失衡。

**⚠️ 局限性**

局限性包括：①规范生成对输入约束与仓库依赖解析不稳，导致误判；②模型易产生过宽或过窄的规范，影响 Pass‑Rate；③仅针对 Python，缺乏跨语言泛化；④对大规模、复杂项目的覆盖仍不足，且评估仅限于执行测试，未结合形式化验证。

---

## 228. Identifying and Mitigating Gender Cues in Academic Recommendation Letters: An Interpretability Case Study

**arXiv ID:** 2604.12337 | [PDF](https://arxiv.org/pdf/2604.12337v1)

**作者:** Charlotte S. Alexander `[一作]` (Georgia Institute of Technology), Bailey Russo `[通讯]` (Georgia Institute of Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对美国医学住院医师项目的推荐信进行脱性别化处理，并用Transformer与LLM进行性别推断，评估隐含性别语言的泄露。

**💡 创新点**

创新点在于结合文本脱性别化、SHAP和TF-IDF识别隐含性别词汇，并通过删除这些词语检验模型性能下降，从而量化隐性性别偏差。

**🔧 技术方法**

使用DistilBERT、RoBERTa、Llama-2进行微调分类，结合SHAP、TF-IDF做可解释性分析，利用参数高效微调（LoRA）提升Llama-2。

**📊 数据集**

使用包含8992封推荐信、按性别标注的美国麻醉住院医师项目数据集。

**📈 对比分析**

在去除显式性别词后模型准确率从≈63%/宏F1≈0.58下降到≈60%/0.55，去除SHAP/TF-IDF词语进一步下降2.7%/1.4%，表明隐含词汇对性别推断贡献显著。

**⚠️ 局限性**

局限包括脱性别化后文本语义完整性受损、SHAP分词对齐问题、缺乏完整申请人档案与录取结果，导致无法评估偏差对决策的真实影响。

---

## 229. Black-Box Optimization From Small Offline Datasets via Meta Learning with Synthetic Tasks

**arXiv ID:** 2604.12325 | [PDF](https://arxiv.org/pdf/2604.12325v1)

**作者:** Azza Fadhel `[一作]` (Washington State University), Jana Doppa `[通讯]` (Washington State University)

**通讯引用:** 2330 | [OpenAlex ID](https://openalex.org/A5055445718)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出OptBias框架，利用合成任务生成与元学习提升小样本离线黑盒优化

**💡 创新点**

创新点在于结合Sim4Opt合成任务与梯度匹配的元学习，强调优化偏差而非值匹配

**🔧 技术方法**

使用高斯过程生成合成任务、Sim4Opt生成梯度上升轨迹、匹配梯度的元学习（Match-Opt扩展）及梯度搜索

**📊 数据集**

使用六个离线优化基准（Ant、D'Kitty、TFBind8、RNA1-3等）进行评估

**📈 对比分析**

与9个基线（GA、MINs、COMs、DEMO、LTR、Match-Opt、Batch BO、REINFORCE、ExPT）对比，在1%数据下OptBias在4/6任务最佳，平均性能显著提升，最差仅2%差距

**⚠️ 局限性**

局限在于对合成任务的GP假设、对离线数据分布的依赖、需要手动调参，且在极低维或更大数据场景下表现需进一步验证

---

## 230. EgoEsportsQA: An Egocentric Video Benchmark for Perception and Reasoning in Esports

**arXiv ID:** 2604.12320 | [PDF](https://arxiv.org/pdf/2604.12320v1)

**作者:** Jianzhe Ma `[一作]` (RUC), Qin Jin `[通讯]` (RUC)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个面向FPS电竞的第一人称视频问答基准EgoEsportsQA，包含1745个多选QA。

**💡 创新点**

创新在于将认知能力与电竞知识两维解耦、引入时间锚点和文本消除泄露，创建高频、信息密集的对抗式测试集。

**🔧 技术方法**

利用视频LLM（Gemini 3 Pro）生成字幕、问题与答案，结合OCR、视频分割、数据清洗和人类校验，最终训练并评估多种闭源和开源Video‑LLM。

**📊 数据集**

基于3款主流FPS（CS2、Valorant、Overwatch 2）专业比赛录像共12.3 h，生成364段视频片段。

**📈 对比分析**

对9种Video‑LLM进行多维度精度评测，最优模型GPT‑5在总分71.58%，在感知层高于推理层，宏观策略优于微观操作。

**⚠️ 局限性**

主要局限在于视频LLM在高速动态、细粒度战术推理与微操作理解上仍显弱，且模型对长视频、高清分辨率处理受限，需改进注意力与多模态融合。

---

## 231. RSGMamba: Reliability-Aware Self-Gated State Space Model for Multimodal Semantic Segmentation

**arXiv ID:** 2604.12319 | [PDF](https://arxiv.org/pdf/2604.12319v1)

**作者:** Guoan Xu `[一作]` (University of Technology Sydney), Guo-Jun Qi `[通讯]` (Westlake University)

**通讯引用:** 14336 | [OpenAlex ID](https://openalex.org/A5100766907)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可靠性感知的自门控状态空间模型 RSGMamba，用于多模态语义分割，能够在融合 RGB 与辅助模态（如深度、热像）时动态调节交叉信息。

**💡 创新点**

核心创新在于显式建模模态可靠性与跨模态一致性，并通过自门控机制在状态空间模型中调节跨模态读出，既捕获全局语义，又抑制噪声传播。

**🔧 技术方法**

使用选择性状态空间模型（Mamba）实现线性复杂度全局建模，结合低秩投影与 LoRA 进行参数高效化，并加入 Local Cross‑Gated Modulation (LCGM) 提升局部细节。

**📊 数据集**

在四大公开基准上进行评估，分别为 NYUDepth V2、SUN‑RGBD（RGB‑D）以及 MFNet、PST900（RGB‑T）数据集。

**📈 对比分析**

与多种前沿融合方法（Concat、Add、Cross‑Attention、Cross‑Mamba 等）及最新 SOTA 进行对比，RSGMamba 在 NYUDepth V2/SUN‑RGBD 上取得 58.8%/54.0% mIoU（比上一最佳 +0.4% / +0.7%），在 MFNet/PST900 上分别达到 61.1%/88.9% mIoU，同时参数仅 48.6M，算力显著降低。

**⚠️ 局限性**

仍面临极端噪声或缺失辅助模态下的鲁棒性挑战，且目前实验集中在图像对，未覆盖更复杂的多传感器组合与时空序列数据。

---

## 232. Cell Instance Segmentation via Multi-Task Image-to-Image Schrödinger Bridge

**arXiv ID:** 2604.12318 | [PDF](https://arxiv.org/pdf/2604.12318v1)

**作者:** Hayato Inoue `[一作]` (Kyushu University), Ryoma Bise `[通讯]` (Kyushu University)

**通讯引用:** 1467 | [OpenAlex ID](https://openalex.org/A5064312777)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种多任务图像到图像 Schrödinger 桥（SB）框架，用于细胞实例分割，去除传统后处理步骤，并通过逆距离图实现边界感知监督。

**💡 创新点**

创新点在于：①将实例分割视为分布式生成问题，利用 SB 直接从输入图像生成可分形的实例掩码；②采用逆距离图作为辅助监督，增强模型对细胞边界的感知；③实现完全无后处理的推理流程，保持高效且结构一致的输出。

**🔧 技术方法**

技术手段包括：Schrödinger 桥的图像到图像生成、U-Net 作为特征提取器、逆距离变换（EDT 归一化反向映射）、确定性逆扩散推理、EMA 参数平滑、以及多任务损失（二值交叉熵 + 均方误差）。

**📊 数据集**

使用了两个公开数据集：PanNuke（大规模，7904 张 256×256 细胞图像）和 MoNuSeg（小样本，30 张训练图像）。

**📈 对比分析**

对比方法包括 U‑Net、U‑Net+R（加入逆距离监督）以及 CellViT（带/不带后处理）。在 PanNuke 上，该方法在 bPQ、F1、召回率和精度上均优于或接近 CellViT+proc，且不依赖 SAM 预训练或后处理；在 MoNuSeg 上，尽管在训练样本有限时 SAM 预训练显著提升 CellViT，但该方法仍能实现与 CellViT+proc 相近的 bPQ，并在 F1、精度上表现更佳。

**⚠️ 局限性**

局限性包括：推理速度相对较慢（3.5s/patch vs 0.35s 传统方法），对扩散步骤的设计和调参敏感；在极度稀疏或形态多变的数据集上，逆距离监督可能不足以完全捕捉复杂边界；此外，尽管不需要后处理，但在极端数据缺乏场景下仍可能受限于模型的分布建模能力。

---

## 233. GTPBD-MM: A Global Terraced Parcel and Boundary Dataset with Multi-Modality

**arXiv ID:** 2604.12315 | [PDF](https://arxiv.org/pdf/2604.12315v1)

**作者:** Zhiwei Zhang `[一作]` (Sun Yat-sen University), Haohuan Fu `[通讯]` (Tsinghua Shenzhen International Graduate School)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed`

**🎯 论文内容**

本文提出了全球复杂梯田农用地块提取的多模态基准数据集 GTPBD-MM，并设计了基于图像、文本和数字高程模型的端到端网络 ETTerra，用于准确分割梯田边界。

**💡 创新点**

创新点在于首次将高分辨率光学影像、结构化文本描述和 DEM 三种模态统一对齐，并通过交叉模态语义增强与高程引导边界重建两分支协同实现对梯田边界的精细化恢复。

**🔧 技术方法**

采用多头跨模态注意力实现文本语义提示，利用 DEM 编码的自适应缩放与偏移实现视觉特征调制，并在 mask 解码器中结合稠密与稀疏提示实现端到端分割。

**📊 数据集**

使用构建于 GTPBD 上的 GTPBD-MM，覆盖 25 国 900+ km² 的梯田地区，提供光学影像、DEM、结构化文本及三层注解（掩码、边界、地块）。

**📈 对比分析**

在 Image-only、Image+Text、Image+Text+DEM 三种输入设置下，基准模型涵盖传统语义分割、分块提取和语言指导方法，ETTerra 在 mIoU、OIS、GTC 等指标上领先，mIoU 达 68.73%，ODS 49.52%，GTC 36.78%，显著优于其它对手。

**⚠️ 局限性**

主要限制包括需依赖完整的三模态数据，缺乏在无 DEM 或文本信息的场景下的鲁棒性；此外模型训练与推理成本相对较高，尚未验证跨域迁移性能。

---

## 234. Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors

**arXiv ID:** 2604.12309 | [PDF](https://arxiv.org/pdf/2604.12309v1)

**作者:** Rong Wang `[一作]` (Australian National University), Hongdong Li `[通讯]` (Australian National University)

**通讯引用:** 18329 | [OpenAlex ID](https://openalex.org/A5101819061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

从单张物体图像生成轨道视频，并保证帧间一致性与真实物体形状。

**💡 创新点**

创新点在于：①利用3D基础模型的全局与局部形状先验作为多尺度条件；②设计多尺度3D适配器，通过交叉注意力将全局向量和投影的体积特征注入视频扩散模型，显著提升形状真实性与多视角一致性。

**🔧 技术方法**

技术手段包括：基于SVD的视频扩散模型、Hunyuan3D 3D生成模型提取形状先验、全局/局部Latent特征、交叉注意力的多尺度3D适配器、CLIP语义嵌入、相机位置正弦编码与CFG推理策略。

**📊 数据集**

使用数据集：Objaverse-XL（高质量3D资产生成的合成轨道视频）和GSO，并在这些数据上进行训练与评估。

**📈 对比分析**

与SV3D、Hi3D、Wonder3D、Era3D、Hunyuan3D、Trellis等基线比较，采用PSNR、SSIM、LPIPS、CLIP-S、MEt3R指标，实验表明本方法在视觉质量、形状真实度和多视角一致性上均实现了显著提升。

**⚠️ 局限性**

局限性包括：受视频模型分辨率限制，细节层次表现不足；3D先验在纹理重建上不够强大，难以生成未观察到的纹理细节；对极端视角或复杂遮挡场景的鲁棒性仍有限；模型对计算资源的需求相对较高。

---

## 235. ContextLens: Modeling Imperfect Privacy and Safety Context for Legal Compliance

**arXiv ID:** 2604.12308 | [PDF](https://arxiv.org/pdf/2604.12308v1)

**作者:** Haoran Li `[一作]` (Beihang University), Yangqiu Song `[通讯]` (HKUST)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个半规则化框架（ContextLens），利用大型语言模型（LLM）对数据隐私与AI安全合规性进行评估，先通过提示让LLM给出结构化JSON输出，随后根据法律文件的层级规则聚合判断并显式识别缺失与含糊的情境因素。

**💡 创新点**

创新点在于：① 将不完整/含糊的上下文显式化为“not sure”或“none of the above”，使LLM能够表达不确定性；② 采用法律文件的分层规则（适用范围>特殊条件>具体条款>一般原则）进行多层级推理；③ 在不需要额外微调的情况下，通过预定义规则识别并报告潜在的缺失上下文，提升评估可靠性。

**🔧 技术方法**

技术包括：LLM提示工程与多模态结构化输出（JSON），半规则化合规推理模块（按层级合规判断），检索增强生成（RAG）用于提取相关条款，及基于规则的合规评估与不确定性标记。

**📊 数据集**

实验使用PrivaCI‑Bench评测集（GDPR和EU AI Act下的6348条真实与合成案例）以及对应的法律文本。

**📈 对比分析**

与直接提示、长链式推理（Long CoT）、Fine‑tuned ContextReasoner以及RAG等基线进行对比；在GPT‑4o-mini、GPT‑4o、Llama‑3.1‑8B‑Instruct和Gemini‑2.5‑Flash等模型上，ContextLens在整体合规准确率与宏观F1均达到或超过现有最高水平，尤其在GPT‑4o-mini上提升了约8%准确率，并能显著发现几乎所有案例中的缺失上下文。

**⚠️ 局限性**

局限性包括：需要人工专家参与法律文档分块与规则校验，无法完全自动化；对采用长链式推理的LLM效果不佳；token消耗与成本较高，且对新出现的法规尚需人工更新规则。

---

## 236. Boosting Robust AIGI Detection with LoRA-based Pairwise Training

**arXiv ID:** 2604.12307 | [PDF](https://arxiv.org/pdf/2604.12307v1)

**作者:** Ruiyang Xia `[一作]` (Institute of Artificial Intelligence (TeleAI), China Telecom), Xuelong Li `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出LoRA‑based Pairwise Training (LPT)方案，用以在严重失真环境下提升AI‑Generated Image检测器的鲁棒性。

**💡 创新点**

创新点在于：①使用LoRA实现高效的视觉基础模型（EVA‑CLIP）微调；②引入多级失真与尺寸模拟以逼近真实世界分布；③采用成对训练策略，将清晰图像与其失真对应样本并行训练，利用辅助FFN对失真特征进行校正，从而实现泛化与鲁棒性的解耦。

**🔧 技术方法**

主要技术包括：视觉基础模型（EVA‑CLIP）+LoRA微调；多重图像失真与随机裁剪/缩放数据增强；成对损失（交叉熵 + KL + MSE）；AdamW + cosine annealing；多尺度与多任务学习。

**📊 数据集**

使用NTIRE 2026 Robust AI‑Generated Image Detection in the Wild数据集（训练/验证/公共/私有测试集），并在最终阶段加入So‑Fake与Chameleon数据进行扩展。

**📈 对比分析**

与竞赛其他参赛队伍对比，LPT在公共/私有“Hard”测试集上分别取得AUC 92.15%/92.50%，排名第三，显著优于多数基线与集成模型；在ForenSynths基准上亦表现出优越的稳健性。

**⚠️ 局限性**

局限性：对极端或未知失真模式的适应性仍有限；依赖大规模视觉基础模型，计算资源和显存消耗较高；目前缺乏针对多尺度与对抗性失真的进一步强化，未来工作需探索MoE与联合对抗训练等方法。

---

## 237. CoSyncDiT: Cognitive Synchronous Diffusion Transformer for Movie Dubbing

**arXiv ID:** 2604.12292 | [PDF](https://arxiv.org/pdf/2604.12292v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 238. TierBPF: Page Migration Admission Control for Tiered Memory via eBPF

**arXiv ID:** 2604.12300 | [PDF](https://arxiv.org/pdf/2604.12300v1)

**作者:** Xi Wang `[一作]` (University of California Merced), Dong Li `[通讯]` (University of California Merced)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在现有内存分层系统中引入FlexTier，利用eBPF插桩实现对迁移页面大小和硬件拓扑的动态准入控制，提升了页面迁移的精确度和适应性。

**💡 创新点**

创新点包括：①基于内存竞争感知的多尺寸透明巨页（mTHP）拆分策略；②针对CXL与PMEM双工模式的迁移准入过滤；③采用全局子页直方图与双向Bloom过滤器实现轻量级、无工作集大小依赖的访问监测。

**🔧 技术方法**

使用技术：eBPF钩子、硬件性能计数器采样、全局子页直方图、双向Bloom过滤器、mTHP动态分配、CXL/PMEM双工模式感知策略。

**📊 数据集**

实验数据集：17个内存密集型工作负载，包括GAP Benchmark Suite（12个图算法）、Silo-YCSB（8线程）、以及NAS Parallel Benchmarks（4个高性能计算基准）。

**📈 对比分析**

对比方法：在CXL和PMEM两种硬件平台上分别与AutoNUMA、TPP、Colloid以及MEMTIS基线进行比较；结果显示CXL平台平均提升12.3%–17.7%，PMEM平台平均提升7.1%，单个工作负载最高可达75%。

**⚠️ 局限性**

局限性：目前仅在支持eBPF和硬件性能计数器的系统上可用；对多级内存扩展（如CXL-SSD）和更大规模的可持久性内存环境尚未充分验证；若硬件缺乏双工模式信息或eBPF功能受限，适用性受限。

---

## 239. GCA Framework: A Gulf-Grounded Dataset and Agentic Pipeline for Climate Decision Support

**arXiv ID:** 2604.12306 | [PDF](https://arxiv.org/pdf/2604.12306v1)

**作者:** Muhammad Umer Sheikh `[一作]` (Mohamed Bin Zayed University of Artificial Intelligence), Muhammad Haris Khan `[通讯]` (Mohamed Bin Zayed University of Artificial Intelligence)

**通讯引用:** 3544 | [OpenAlex ID](https://openalex.org/A5032830353)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了 Gulf Climate Agent（GCA）框架，集成了海湾地区气候相关的多模态 QA 数据集 GCA-DS 与工具增强型语言代理，支持从文本、遥感影像到气候指标的查询与解释。

**💡 创新点**

创新点：①构建了包含约 200k 个问题答案对的海湾地区专属多模态数据集（文本+视觉），②设计了面向海湾气候任务的工具套件，并把 LLM 与这些工具链无缝对接，③通过领域微调与工具集成显著提升了工具调用与答案质量。

**🔧 技术方法**

技术手段：利用 Qwen2.5‑VL 7B 作为基础模型，采用 LoRA 参数高效微调；使用 ReAct 结构化工具调用；构建多模态数据生成管线（检索、解析、事实提取、QA 合成）；集成遥感、气象、空气质量等气候分析工具；实现图像与文本的跨模态推理。

**📊 数据集**

数据集：GCA-DS（约 200k QA 对，涵盖政府政策、NGO 报告、学术论文、事件新闻、遥感影像等），以及 91 个人工标注的评测问题；工具套件中使用 Sentinel‑2、ERA5、CAMS、GloFAS 等公共气候遥感与数值数据。

**📈 对比分析**

对比方法与性能：在工具使用与最终答案准确性上，与 GPT‑5、Claude 4.5 Sonnet、Gemini 2.5 Pro、Qwen2.5‑VL 7B、Pixtral‑12B 进行统一评测；GCA 在 step‑by‑step 模式下 ToolAcc 94.2%/SummAcc 88.6%，在 end‑to‑end 模式下 AnsAcc 88.2%/AnsAcc+I 89.1%，相较基线提升 30–35% 并与强商用模型保持竞争甚至超越。

**⚠️ 局限性**

局限性：①数据集仅部分人工验证，自动生成 QA 可能存在标注与 grounding 错误；②工具依赖导致结果受工具数据质量与缺失影响；③评测覆盖范围有限，未覆盖所有气候任务（如不确定性量化、长期情景规划）；④真实部署需要持续监控、透明溯源与专家审查。

---

## 240. Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization

**arXiv ID:** 2604.12290 | [PDF](https://arxiv.org/pdf/2604.12290v1)

**作者:** Yizhe Chi `[一作]` (Navers Lab, Einsia.AI), Qinhuai Na `[通讯]` (Navers Lab, Einsia.AI)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Frontier-Eng benchmark，构建了 47 个真实工程优化任务，评估 AI 生成式优化在可执行评估器下的迭代改进能力。

**💡 创新点**

创新点在于将工程优化形式化为 (上下文、初始解、可执行评估器) 的三元组，统一跨域任务接口并通过只读评估器防止 reward hacking，填补了现有只关注二元奖励或单域的空白。

**🔧 技术方法**

采用大型语言模型（如 GPT-OSS-120B、Opus 等）与多种搜索框架（OpenEvolve、ABMCTS、ShinkaEvolve）实现 propose–evaluate–refine 循环，结合执行沙箱和日志解析实现安全评估。

**📊 数据集**

使用 47 个工程任务的数据集，覆盖计算、运筹、机器人、光学、电化学等五大领域，任务来源包括公开竞赛、学术基准、工业仿真与专家贡献。

**📈 对比分析**

通过平均排名、性能曲线和基线赢率等无量纲指标进行比较；在 100 步预算下，最优模型 Opus 平均排名 3.18，胜率和性能曲线均高于多数同类模型，展示跨域竞争力。

**⚠️ 局限性**

限制在于多目标、长周期仿真及复杂物理耦合的任务仍难以显著提升，搜索策略受预算限制且模型在后期迭代易出现停滞，影响最终优化深度。

---

## 241. Practical Evaluation of the Crypto-Agility Maturity Model

**arXiv ID:** 2604.12428 | [PDF](https://arxiv.org/pdf/2604.12428v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 242. The Enforcement and Feasibility of Hate Speech Moderation on Twitter

**arXiv ID:** 2604.12289 | [PDF](https://arxiv.org/pdf/2604.12289v1)

**作者:** Manuel Tonneau `[一作]` (University of Oxford), Samuel P. Fraiberger `[通讯]` (World Bank)

**通讯引用:** 1962 | [OpenAlex ID](https://openalex.org/A5075585231)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对2022年9月21日Twitter（现X）全平台24小时公开推文进行大规模取样，手工标注540,000条推文的仇恨言论、暴力仇恨、侮辱性与中立三类，并对多语言、跨国样本进行执法效果评估（删除、暂停率）与可行性分析；通过模拟人机协同审核管线，估算对仇恨言论曝光的减少量与成本；对多语言公开AI仇恨检测模型进行性能评估，探讨其在实际审核中的定位。

**💡 创新点**

首次完成全球性、跨语言的仇恨言论审核审计；构建最大规模的手工标注语料；系统评估平台执法缺口与技术可行性；将人机协同审核的成本与监管罚款做对比，揭示监管激励与资源配置不匹配的根源。

**🔧 技术方法**

人工注释（多语言三位评议员+多数投票+二次评议）、线性概率回归与Logistic回归、AI检测模型（公开监督模型、Perspective API、GPT‑5.1）、人机协同审核仿真（基于分数排序、投入人力参数），以及成本收益分析。

**📊 数据集**

TwitterDay 2022‑09‑21全平台24小时公开推文样本（375M条），在8种主流语言（阿拉伯语、英语、法语、德语、印尼语、葡萄牙语、西班牙语、土耳其语）以及4个英美国家（美、印、奈、肯）中抽取540,000条进行手工标注，产生约1.6M条标签。

**📈 对比分析**

对比执法结果（删除/暂停率）与平台自报；AI检测模型在各语言上AP与精确度/召回率低于预期，但在分数排序上能聚焦约3%高分中一半为仇恨；人机协同仿真表明，在现有审核人力下，覆盖率低但可在80%曝光减少时仅需约2.9%全球营收成本，低于欧盟/英国的罚金上限。

**⚠️ 局限性**

仅针对文本，未涵盖图片/视频；只取单一天数据，未考虑时间波动；对某些语言模型与数据可用性有限；审核结果受平台所有权变更与账号恢复影响；人工标注一致性有限，导致部分误差。

---

## 243. MAST: Mask-Guided Attention Mass Allocation for Training-Free Multi-Style Transfer

**arXiv ID:** 2604.12281 | [PDF](https://arxiv.org/pdf/2604.12281v1)

**作者:** Dongkyung Kang `[一作]` (Dongguk University), Hyeryung Jang `[通讯]` (Dongguk University)

**通讯引用:** 316 | [OpenAlex ID](https://openalex.org/A5056471397)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无训练的Mask‑guided attention mass allocation框架，实现多风格图像迁移，消除边界伪影与结构失真。

**💡 创新点**

在扩散模型注意力机制中引入logit级别的注意力质量分配、温度缩放、查询锚定和细节注入，实现在不同区域内无干扰的多风格融合与结构保留。

**🔧 技术方法**

基于Latent Diffusion Model的自注意力模块，DDIM反向推断、AdaIN初始化、Logit‑level Attention Mass Allocation、Sharpness‑aware Temperature Scaling、Layout‑preserving Query Anchoring、Discrepancy‑aware Detail Injection等技术。

**📊 数据集**

使用MS‑COCO随机采样的20张内容图像和WikiArt的360张风格图像，构成多样化的内容‑风格对进行实验。

**📈 对比分析**

与StyTr^2、InST、CAST、Z^*、StyleID、DiffuseST、StyleSSP、StyleShot等基线进行对比，使用ArtFID、FID、LPIPS、CFSD、M‑FID等指标，MAST在多风格数量增加时仍保持最低的ArtFID、FID、CFSD与M‑FID，并显著优于对照组。

**⚠️ 局限性**

缺乏对极端分辨率或复杂纹理场景的评估，且在极高风格冲突场景下的鲁棒性尚待验证。

---

## 244. RoleMAG: Learning Neighbor Roles in Multimodal Graphs

**arXiv ID:** 2604.12271 | [PDF](https://arxiv.org/pdf/2604.12271v1)

**作者:** Yilong Zuo `[一作]` (Beijing Institute of Technology), Guoren Wang `[通讯]` (Beijing Institute of Technology)

**通讯引用:** 7415 | [OpenAlex ID](https://openalex.org/A5054991337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `afceb026-1760-41ae-8d86-010831a37d97` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计了一种在多模态图学习中区分邻居角色的传播框架RoleMAG，能够在共享、互补和异质性三条通道中分别处理邻居信息，提升多模态图学习的表示质量。

**💡 创新点**

创新点在于：①将边的角色（共享、互补、异质）视为可学习的概率分布并进行角色路由；②为互补关系引入方向化的跨模态补全机制；③为异质关系设计符号多项式滤波器，避免其被错误混入共享平滑。

**🔧 技术方法**

采用了多模态属性嵌入、基于语义和结构的边特征、可解释的角色概率推断、共享专家、方向化互补专家（查询瓶颈注意力）和异质专家（符号多项式滤波），以及残差门控融合和多项式正则化等技术。

**📊 数据集**

在OpenMAG统一评测框架下的三大基准数据集：Toys（节点分类）、RedditS（节点分类）和Bili_Dance（链路预测）。

**📈 对比分析**

与GCN、GAT、MMGCN、MGAT、LGMRec、DGF、DMGC、NTSFormer、Graph4MM等经典与最近的多模态图模型进行对比。RoleMAG在RedditS和Bili_Dance上取得最高ACC/F1和MRR/Hit@3，Toys上保持竞争力；在噪声扰动下表现更稳健，且计算开销相对适中。

**⚠️ 局限性**

局限性包括：①需要额外的计算和显存开销；②对边的角色分配依赖于语义一致性估计，若模态信息缺失或噪声严重可能导致角色误判；③在已具备强查询式融合的基准（如Graph4MM）上提升幅度有限。

---

## 245. Fundus Image-based Glaucoma Screening via Retinal Knowledge-Oriented Dynamic Multi-Level Feature Integration

**arXiv ID:** 2604.12351 | [PDF](https://arxiv.org/pdf/2604.12351v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 246. Scaffold-Conditioned Preference Triplets for Controllable Molecular Optimization with Large Language Models

**arXiv ID:** 2604.12350 | [PDF](https://arxiv.org/pdf/2604.12350v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 247. Submodular Max-Min Allocation under Identical Valuations

**arXiv ID:** 2604.12417 | [PDF](https://arxiv.org/pdf/2604.12417v1)

**作者:** Kimon Boehmer `[一作]` `[通讯]` (PSL University), Kimon Boehmer (PSL University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出一种截断最大和贪心算法，对相同子模函数下的子模最大-最小分配问题给出0.4近似解，并给出了配置线性规划的常数上界（不超过3），以及在k个基约束下的5上界。

**💡 创新点**

创新点在于：①用截断贪心改进了此前10/27≈0.37的近似比至0.4；②通过构造双对偶解首次给出子模相同价值情况下配置LP的常数上界；③将结果推广到多基约束环境。

**🔧 技术方法**

核心技术包括：子模函数的边际增益性质、图论工具（Hall定理、代价图）、贪心分析、LP/双对偶构造与割平面求解、以及对基约束下的独立性判定。

**📊 数据集**

本工作为理论研究，未使用具体实验数据集，而是以理论实例与构造反例进行分析。

**📈 对比分析**

相较于之前最好的10/27≈0.37近似，比其提高到0.4；配置LP的积分缺口在子模同值情形下被证明不超过3，带k个基约束时不超过5，远低于已知的无穷大上界。

**⚠️ 局限性**

局限性包括：近似比例仍未达到理论最优1-1/e≈0.63；缺乏多项式时间的构造性加权求解算法；仅针对相同子模函数，未解决相关机器或不同价值函数的情况。

---

## 248. Three Birds, One Stone: Solving the Communication-Memory-Privacy Trilemma in LLM Fine-tuning Over Wireless Networks with Zeroth-Order Optimization

**arXiv ID:** 2604.12401 | [PDF](https://arxiv.org/pdf/2604.12401v1)

**作者:** Zhijie Cai `[一作]` (Chinese University of Hong Kong-Shenzhen), Guangxu Zhu `[通讯]` (Chinese University of Hong Kong-Shenzhen)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了pAirZero及其数字版Sign-pAirZero，利用零阶优化与空中算术（OTA）协同实现边缘LLM微调，极大降低通信量与显存占用，并通过通信过程嵌入差分隐私保证数据安全；

**💡 创新点**

创新点在于①将零阶梯度估计与OTA相结合，形成一次性发送梯度投影的比特级通信；②通过调节发射功率与人工噪声实现“隐私即设计”，即在任意信道噪声下均能满足指定DP预算；③提供闭式最优功率分配方案，避免传统同步与频谱资源瓶颈；

**🔧 技术方法**

技术实现包括：零阶随机方向梯度估计（SPSA）、OTA信号聚合（模拟与数字）、差分隐私噪声注入、传输功率与噪声水平的凸优化、收敛性与有效性理论证明；

**📊 数据集**

实验使用OPT-125M轻量级LLM，在SST-2与SQuAD两种自然语言任务上进行微调，数据集规模约千级样本；

**📈 对比分析**

与传统基于一阶SGD/Adam的联邦微调及无隐私/无OTA基线对比，pAirZero在相同隐私预算下实现了与非隐私基线相近的任务性能；同时实现了≈75%显存降低、按位通信量降低数十倍、峰值显存仅为OPT-125M的25%；

**⚠️ 局限性**

局限性包括：①需要在所有客户端与服务器之间同步随机种子；②对信道增益和噪声模型假设较为理想，实际环境中需进一步鲁棒性验证；③数字OTA（Sign-pAirZero）在高噪声下仍有性能波动；④对超参数（γ、A等）敏感，需经验调优；

---

## 249. On the Distillation Loss Functions of Speech VAE for Unified Reconstruction, Understanding, and Generation

**arXiv ID:** 2604.12383 | [PDF](https://arxiv.org/pdf/2604.12383v1)

**作者:** Changhao Cheng `[一作]` (Shanghai Jiao Tong University), Yanmin Qian `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 34850 | [OpenAlex ID](https://openalex.org/A5100363498)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

本文提出一种联合边缘对齐的变分自编码器（JMAS‑VAE），通过对齐 VAE 潜在空间与预训练的语音自监督模型（SSL）特征，实现了在重建、理解与生成任务上的统一表现。

**💡 创新点**

创新点在于：①设计了联合边缘对齐损失（T轴余弦、D轴余弦以及序列级分布相似度），②引入自适应加权机制以动态平衡重建与对齐损失，③通过门限（margin）调节对齐力度，获得了兼顾三大任务的高质量连续语音表示。

**🔧 技术方法**

使用了 VAE 结构（含重建、KL 与 GAN 损失），WavLM Large 作为 SSL 目标特征，MLP 投影层、动态自适应权重、余弦距离与 ReLU 门限等对齐策略，并在 Libriheavy/LibriTTS 上进行训练。

**📊 数据集**

实验数据集包括：LibriSpeech‑test‑clean（评估重建和 TTS）、LibriTTS（训练 TTS 模型）、8 项 SUPERB 任务（情感识别、音素识别、ASR、关键词检测、说话人识别、验证、分离、意图分类），以及全量 Libriheavy 进行 VAE 预训练。

**📈 对比分析**

与 Vanilla VAE、Semantic‑VAE、EnCodec 及基线（Mel/Fbank）对比，JMAS‑VAE 在整体几何平均分上取得最高分，重建（PESQ/2+STOI/2）和生成（1‑WER+SIM/2）保持与基线相当，同时显著提升了 8 项理解任务的平均性能。

**⚠️ 局限性**

局限性在于：对齐过程仍需手动设定门限与权重，过强对齐会损害音频质量；模型对通道维度、帧率等超参数的敏感性未充分探索，且在极端任务（如极端噪声下的 ASR）中仍表现不佳。

---

## 250. CoLA: A Choice Leakage Attack Framework to Expose Privacy Risks in Subset Training

**arXiv ID:** 2604.12342 | [PDF](https://arxiv.org/pdf/2604.12342v1)

**作者:** Qi Li `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 108615 | [OpenAlex ID](https://openalex.org/A5058772567)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在子集训练过程中产生的隐私泄露风险，提出了 Choice Leakage Attack（选择泄露攻击）框架，系统探究了训练成员 MIA（TM‑MIA）与选择参与 MIA（SP‑MIA）的隐私表面。

**💡 创新点**

创新点在于：①将子集选择过程视为一个新的攻击面，定义了 TM‑MIA 与 SP‑MIA；②引入多次选择稳定性（inclusion‑stability）作为 membership 关键信号，形成统一的侧信道与黑盒攻击框架；③展示子集训练反而能放大泄露风险，挑战传统直觉。

**🔧 技术方法**

使用的技术包括：多窗口子集构造、选择稳定性计数、权重映射（σ‑函数）、无监督嵌入聚类、基于距离的证据收集、阈值决策；实验中对 AUC 与 TPR@低 FPR 进行评估。

**📊 数据集**

实验数据集：视觉端使用 CIFAR‑10/CIFAR‑100，模型为 ResNet‑18、VGG19；语言端使用 Pythia‑160M、GPT‑Neo‑125M 等；子集选择方法覆盖九种核心算法（如 Glister、DeepFool、Uncertainty 等）以及两种 deduplication 强度。

**📈 对比分析**

与传统 MIAs（NN、LiRA、Loss、Bow 等）对比，Choice Leakage Attack 在 TM‑MIA 约 80‑90% AUC、SP‑MIA 约 90‑95% AUC，显著优于基线；在黑盒与侧信道两种设置下均保持强性能，证明其在多样模型与数据上的通用性。

**⚠️ 局限性**

局限性：仅测试了少数几种子集选择算法，未覆盖所有可能的子集策略；未评估对抗性防御或差分隐私等保护措施；实验规模主要集中在小型语言模型，未验证在大模型上的可扩展性。

---

## 251. From Myopic Selection to Long-Horizon Awareness: Sequential LLM Routing for Multi-Turn Dialogue

**arXiv ID:** 2604.12385 | [PDF](https://arxiv.org/pdf/2604.12385v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 252. LightTune: Lightweight Forward-Only Online Fine-Tuning with Applications to Link Adaptation

**arXiv ID:** 2604.12406 | [PDF](https://arxiv.org/pdf/2604.12406v1)

**作者:** Ramy E. Ali `[一作]` (Samsung Semiconductor), Federico Penna `[通讯]` (Samsung Semiconductor)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种轻量级、无反向传播、基于前向前向（FF）算法的在线微调框架 LightTune，能够在移动设备上实时自适应部署后的机器学习模型。

**💡 创新点**

创新点在于将 FF 算法与基于阈值的无缓冲区在线更新策略相结合，提供闭式梯度、缓冲区无存储、有限时间收敛保证，并首次在 6G 链路适配中应用。

**🔧 技术方法**

使用前向前向（FF）算法、平滑二次 Softplus 近似损失、Adam 一阶/一步更新、阈值触发的无缓冲区采样以及 BLER 预测和 CQI/RI 选择的联合优化。

**📊 数据集**

在 TDL 频道模型（A30、B50、B100、C200）下，采用 100 MHz 带宽、30 kHz 子载波间距的 3GPP 训练/测试集，使用 12‑维特征。

**📈 对比分析**

与传统的表格型 OLLA、无微调的离线 ML 以及基于校准的对比，LightTune 通过在线微调将 BLER 预测误差降低 36–48.8%，并在链路适配中提升吞吐量 12–15.5%，显著优于基线。

**⚠️ 局限性**

局限性包括对硬件实现仍需验证、只能处理离散化的 BLER 类、对极端动态环境的适应性尚未完全评估，以及在更复杂的网络规模下的可扩展性待进一步研究。

---

## 253. Tamper-Proofing with Self-Modifying Code

**arXiv ID:** 2604.12407 | [PDF](https://arxiv.org/pdf/2604.12407v1)

**作者:** Gregory Morse `[一作]` (Eötvös Loránd University), Tamás Kozsik `[通讯]` (Eötvös Loránd University)

**通讯引用:** 258 | [OpenAlex ID](https://openalex.org/A5009713228)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了基于自修改代码（SMC）的防篡改机制，并在x86‑64平台上实现与评估。

**💡 创新点**

创新点在于将执行时序与自校验结合，证明非SMC实现无法在不产生可检测延迟的情况下复制SMC；引入跨页写入、循环展开等技术显著降低流水线清除成本。

**🔧 技术方法**

使用了x86‑64汇编、AsmJit动态生成、RDTSC/RDTSCP/HPET/UTC等计时器、性能计数器、RIP‑relative地址、循环展开与跨页写入等技术。

**📊 数据集**

实验基于自生成的220 KB代码段和两页动态生成SMC，运行在Intel Core i7‑9750H（16 GB RAM）上，收集10,000次执行的计时与性能计数器数据。

**📈 对比分析**

通过与不使用SMC的基准代码比较（RDTSC、RDTSCP、精确UTC等计时器的平均/最小/最大时延），发现静态SMC平均约7.9×慢，动态SMC约90.5×快于语义精确非SMC，且比静态SMC快约2.5×；pipeline clear 次数显著下降。

**⚠️ 局限性**

局限包括需要可写可执行页且受操作系统权限限制、需针对特定平台调参、仍受调度噪声影响、对Harvard架构不适用、以及假设攻击者无法完整发现所有校验和。

---

## 254. Throughput Characterization of Wireless CSMA Networks With Arbitrary Sensing and Interference Topologies

**arXiv ID:** 2604.12400 | [PDF](https://arxiv.org/pdf/2604.12400v1)

**作者:** Xinghua Sun `[一作]` (Sun Yat-sen University), Ruike Zhou `[通讯]` (Sun Yat-sen University)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一种针对任意感知与干扰拓扑的无线 CSMA 网络吞吐量分析框架，能够在不依赖零传播延迟假设的情况下给出显式吞吐量表达式。

**💡 创新点**

创新点在于将感知图的团结构映射为等价的多通道网络，并用离散时间马尔可夫再生过程完整捕捉全局耦合，从而突破传统基于独立集或节点中心的简化假设。

**🔧 技术方法**

主要技术包括图论（最大团枚举）、布隆-克罗克斯算法、离散时间马尔可夫再生模型、状态空间压缩、深度优先搜索与低阶下界近似等。

**📊 数据集**

实验使用仿真数据，分别在多 BSS IEEE 802.11 网络和带隐藏/暴露/中流效应的九链路自组网络上验证模型，未使用公开数据集。

**📈 对比分析**

与节点中心的 HOL 模型、基于独立集的 CTMC 模型以及 BOE 估计方法相比，所提模型在密集部署与强耦合场景下吞吐量误差低于 5%，并能更准确地预测最优退避窗口，提升 20–30% 的总吞吐量。

**⚠️ 局限性**

主要限制是计算复杂度随感知图中最大团数和包时延 τ 指数增长，尽管通过状态压缩与下界近似降低了负担，但在极大网络规模或长 τ 情形下仍可能不可行。

---

## 255. ReflectCAP: Detailed Image Captioning with Reflective Memory

**arXiv ID:** 2604.12357 | [PDF](https://arxiv.org/pdf/2604.12357v1)

**作者:** Kyungmin Min `[一作]` (Seoul National University), Kyomin Jung `[通讯]` (Seoul National University)

**通讯引用:** 3608 | [OpenAlex ID](https://openalex.org/A5077832834)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 ReflectCAP 框架，通过离线多代理反思生成结构化反思笔记，在推理时引导 LVLM 生成既准确又细节丰富的图像说明。

**💡 创新点**

创新点在于将模型的常见幻觉和遗漏拆分为两类结构化反思笔记，分别在生成时抑制幻觉与补充细节，从而实现事实性与覆盖率的 Pareto 前沿，并且不需要再训练或多代理推理。

**🔧 技术方法**

采用多代理反思学习、结构化反思笔记、指令注入与合并、离线-在线两阶段流程以及提示工程等技术。

**📊 数据集**

使用 IIW‑400、CapArena‑Auto、IIW‑Eval、DOCCI 等细粒度图像说明数据集进行离线构造与在线评估。

**📈 对比分析**

与零射、少量示例、Self‑Correction、CapMAS 等方法对比，在 IIW‑400 上实现最高 F1；在 CapArena‑Auto 上 win‑rate 提升 +32.3（GPT 系列）/ +21.9（开源模型），且计算成本比 CapMAS 低 21–36%。

**⚠️ 局限性**

局限性包括依赖 LVLM 的指令遵循能力，对视觉边界超出模型感知范围的图像仍可能出现残留幻觉；离线构造笔记需要示例集，且方法在指令遵循有限时效果受限。

---

## 256. On the Optimality of Hierarchical Secure Aggregation with Arbitrary Heterogeneous Data Assignment

**arXiv ID:** 2604.12429 | [PDF](https://arxiv.org/pdf/2604.12429v1)

**作者:** Chenyi Sun `[一作]` (Huazhong University of Science and Technology), Xiang Zhang `[通讯]` (Technical University of Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文研究了三层层级网络中信息理论安全聚合问题，并针对任意异构数据分配场景，提出了一种既满足服务器安全又满足中继安全的线性编码方案，能够在存在随机用户掉线和协同攻击的情况下实现最优的两层通信负载。

**💡 创新点**

创新点在于：①首次在层级网络中兼顾服务器与中继的安全约束；②针对任意异构数据分配（不受传统循环或复制结构限制）提供了最优通信负载证明；③采用混合设计，将梯度编码与安全密钥结合，既保证了解码可行又消除了信息泄露。

**🔧 技术方法**

主要技术包括：信息理论安全分析、线性编码与密钥共享、梯度编码（Gradient Coding）与安全梯度编码、虚拟需求矩阵设计以及对齐技术，整体基于有限域上的线性算子。

**📊 数据集**

实验与理论评估基于将整体训练数据集划分为 K 个等长子集（无具体公开数据集），通过仿真验证方案在不同落后用户数和协同攻击量下仍保持最优通信负载。

**📈 对比分析**

与之前的对称数据分配或无安全约束的梯度编码方案相比，该方案在满足相同安全级别的前提下实现了理论下界（_1 ≥ 1/_1, _2^(u) ≥ 1/_1_2^(u)）的通信负载，且不产生额外的通信开销；实验结果显示，在各种落后用户与协同攻击设置下，通信负载始终等于下界，说明性能最优。

**⚠️ 局限性**

局限性包括：①假设通信链路是正交且无噪声；②中继节点被视为可靠；③需要先验知晓完整的数据分配与密钥长度，且对大规模动态环境的适应性未验证；④方案依赖较大的有限域和线性矩阵尺寸，实际部署时可能面临计算与存储开销。

---

## 257. Agentic Insight Generation in VSM Simulations

**arXiv ID:** 2604.12421 | [PDF](https://arxiv.org/pdf/2604.12421v1)

**作者:** Micha Selak `[一作]` (RheinMain University of Applied Sciences), Andreas Loehr `[通讯]` (SimPlan AG)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种分离式代理架构，用于在价值流映射（VSM）仿真中高效提取可操作洞察

**💡 创新点**

将高层规划与数据分析拆分为主导代理与子工作流，采用逐步发现机制以避免上下文膨胀并提高多跳推理准确率

**🔧 技术方法**

基于大型语言模型（LLM）的代理技术，配合节点发现、属性提取、分类导航和摘要工具，实现数据检索与动态分析

**📊 数据集**

构建了专门的VSM仿真数据集，包含模型与仿真输出、自然语言查询与专家答案，共计约160条三元组，涵盖不同行业场景

**📈 对比分析**

通过“LLM作为评判者”方法对不同规模LLM（Ministral、Qwen、GLM-5、Claude‑Opus 4.6）进行四次评测，Claude‑Opus 4.6在准确率和一致性上最优，最高得分约86/100；小模型表现显著逊色

**⚠️ 局限性**

局限性包括：小规模LLM在多跳发现上不稳定；单一LLM推理成本高、Token消耗大；对长时序数据的处理受限；以及对模拟参数优化建议的能力尚未实现

---

## 258. Security and Resilience in Autonomous Vehicles: A Proactive Design Approach

**arXiv ID:** 2604.12408 | [PDF](https://arxiv.org/pdf/2604.12408v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

---

## 259. RACF: A Resilient Autonomous Car Framework with Object Distance Correction

**arXiv ID:** 2604.12418 | [PDF](https://arxiv.org/pdf/2604.12418v1)

**作者:** Chieh Tsai `[一作]` (University of Arizona), Salim Hariri `[通讯]` (University of Arizona)

**通讯引用:** 8251 | [OpenAlex ID](https://openalex.org/A5057335897)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研发了一种基于多传感器冗余与预训练时间序列模型的鲁棒距离估计框架，并提出Object Distance Correction Algorithm（ODCA）实现感知层的自适应纠错；

**💡 创新点**

创新点在于将冻结的ChronosV2先验与轻量增量修正头相结合，并通过LiDAR一致性门控仅在跨模态不一致时触发修正，从而在保持正常测量的同时实现实时攻击抵抗；

**🔧 技术方法**

使用了多模态融合、LiDAR一致性门控的残差修正、ChronosV2预训练时间序列模型、物理动力学约束、离线损失设计，并在Quanser QCar 2平台上部署；

**📊 数据集**

采用Quanser QCar 2多传感器同步数据集（67,821个样本，包含深度、LiDAR、车辆状态）进行实验，同时设计人工离线偏置/黑屏攻击和现场物理停止标志攻击进行评测；

**📈 对比分析**

与Chronos、SOFTS、NHITS、DLinear等时间序列预测基线以及LiDAR、EKF融合基线对比；在强攻击下ODCA的RMSE降至0.323 m，较Chronos降低约35%；闭环停止符号准则成功率提升至80%，制动延迟降至0.19 s；

**⚠️ 局限性**

局限性包括对LiDAR的依赖和门控阈值的敏感性；在持续攻击超过约3 s时仍可能失败；部署在离线服务器会引入额外延迟；仅针对2D LiDAR与深度相匹配的场景验证，其他环境或传感器类型需重新校准和验证。

---

## 260. DeferredSeg: A Multi-Expert Deferral Framework for Trustworthy Medical Image Segmentation

**arXiv ID:** 2604.12411 | [PDF](https://arxiv.org/pdf/2604.12411v1)

**作者:** Qiuyu Tian `[一作]` (Shandong University), Yilong Yin `[通讯]` (Shandong University)

**通讯引用:** 5890 | [OpenAlex ID](https://openalex.org/A5100672590)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

本文提出了一种基于像素级学习退让（L2D）的医学图像分割框架，能够在需要时将不确定像素动态分配给人类专家进行复核。

**💡 创新点**

核心创新点在于：①引入像素级退让预测器和多专家路由通道；②设计针对分割的退让合作损失、空间连贯性损失和负载均衡惩罚，以实现可靠的区域级协作；③实现了对预训练分割模型（如MedSAM、CENet）的无缝迁移。

**🔧 技术方法**

技术方法包括：Transformer‑U‑Net风格的MedSAM/ CENet 作为基底，CNN+Softmax退让预测、像素级交叉熵、空间平滑正则化及专家负载平衡正则化，联合训练形成端到端模型。

**📊 数据集**

在四个公开数据集上进行评估：PROMISE12（前列腺 MRI）、LiTS（肝脏/肿瘤 CT）、AMOS22（多器官 CT/MRI）以及Chaksu（视网膜彩照）中使用真实与合成专家标签。

**📈 对比分析**

与 MedSAM、nnU-Net v2、CENet 等强基线对比，系统在 DSC、Jaccard 与 Sensitivity 上普遍提升 5–20% 点，尤其在边界模糊、低对比度及小目标区域表现突出；多专家设置在负载平衡下可进一步提高召回率。

**⚠️ 局限性**

主要局限包括：①需要专家标注或合成专家模型；②模型结构相对复杂，训练难度较高；③当专家数量过多或专家间互补性不足时，系统性能可能出现下降；④目前仅在二维切片上验证，三维卷积迁移仍待研究。

---

## 261. Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models

**arXiv ID:** 2604.12391 | [PDF](https://arxiv.org/pdf/2604.12391v1)

**作者:** Jiawei Fan `[一作]` (Intel Labs China), Anbang Yao `[通讯]` (Intel Labs China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 Chain-of-Models Pre-Training (CoM-PT) 方法，通过模型链和逆向知识迁移实现对视觉基础模型族的无性能损失加速预训练。

**💡 创新点**

创新点在于：①在模型族层面设计升序模型链；②采用逆向权重初始化和逆向特征蒸馏同时在参数空间和特征空间实现知识复用；③通过模型链传递学习，训练成本随族规模增大而递减。

**🔧 技术方法**

技术包括：模型链组织、逆向知识迁移（权重复制与填充、特征空间蒸馏）、基于 CLIP 的对比学习、数据增强（文本多样化）以及混合精度、分布式训练等基础加速技术。

**📊 数据集**

数据集：CC3M、CC12M（合并为 Merged-15M），共 45 个下游任务数据集（ImageNet-1K、VTAB+、COCO、ADE-847/150、PC-459/59、VOC-20、TextVQA、ScienceQA、POPE、VQAv2 等）。

**📈 对比分析**

与传统单模型预训练（如 LaCLIP）对比，CoM-PT 在 45 个任务上实现了 0.5% 以内的性能保留，并在训练 MAC 与 GPU 计算时间上分别提高 4–7× 的加速比；在 ViT‑L/16 上可达 7.09× 的加速，同时在多模型族情况下整体成本低于单独训练最大模型。

**⚠️ 局限性**

局限性：需要手动设计模型链（最小模型、扩展比例、epoch 分配），对不同架构的适配性尚未系统验证；实验主要集中在视觉模型，LLM 的适用性仍待进一步探索。

---

## 262. Dual-Modality Anchor-Guided Filtering for Test-time Prompt Tuning

**arXiv ID:** 2604.12403 | [PDF](https://arxiv.org/pdf/2604.12403v1)

**作者:** Jungwon Choi `[一作]` (Chung-Ang University), Eunwoo Kim `[通讯]` (Chung-Ang University)

**通讯引用:** 456 | [OpenAlex ID](https://openalex.org/A5074532898)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出双模态锚点引导的测试时提示调优框架，通过文本锚点和图像锚点筛选增强视图，并将锚点作为辅助预测头进行置信度加权集成；

**💡 创新点**

创新点在于①利用LLM生成属性丰富的文本描述并对齐权重构造文本锚点；②构建自适应图像原型库形成图像锚点；③联合对齐‑置信度评分实现语义与视觉双重视图筛选；④将锚点作为预测源生成置信度加权集成，避免传统熵最小化导致的过拟合；

**🔧 技术方法**

使用CLIP视觉语言模型、LLM文本生成、余弦相似度对齐、对齐‑熵联合评分、置信度加权集成、KL散度目标分布及梯度下降优化提示词；

**📊 数据集**

在ImageNet及其四个OOD版本（ImageNet-A、V2、R、Sketch）和十个跨域图像分类基准（Flower102、DTD、OxfordPets、StanfordCars、UCF101、Caltech101、Food101、SUN397、FGVC-Aircraft、EuroSAT）上进行实验；

**📈 对比分析**

与现有提示学习（CoOp、CoCoOp、MaPLe、CoPrompt、Any-shift）和测试时提示调优方法（TPT、DiffTPT、C-TPT、DynaPrompt）对比，平均精度提升约3.3%–6.3%，在ImageNet-OOD上提升3–4%，并且与CLIP或其它基线融合后可进一步提升；

**⚠️ 局限性**

仍依赖预训练的CLIP和LLM，构造锚点和设置阈值需人工调参；在极端分布偏移或样本极少的类别上仍可能难以构造有效锚点，泛化性需进一步验证。

---

## 263. Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints

**arXiv ID:** 2604.12384 | [PDF](https://arxiv.org/pdf/2604.12384v1)

**作者:** Songping Peng `[一作]` (Hunan Normal University), Xieping Gao `[通讯]` (Hunan Normal University)

**通讯引用:** 2607 | [OpenAlex ID](https://openalex.org/A5010870600)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种同时约束权重和激活的安全对齐方法CWAC，旨在防止微调过程中的安全偏移。

**💡 创新点**

创新点在于将权重投影子空间与稀疏自编码器识别的安全关键特征正则化耦合，解决单层约束不足的问题。

**🔧 技术方法**

采用权重投影子空间（SVD+低值特征）与稀疏自编码器（TopK SAE）以及梯度投影等技术。

**📊 数据集**

使用公开的对齐基线模型和多种任务数据集：SST-2、AGNEWS、GSM8K、PubMedQA、Alpaca，以及混合的 JailbreakBench、HarmBench、AdvBench、BeaverTails 等有害样本。

**📈 对比分析**

在四个主流7B–9B LLM（Llama-2-7B、Llama-3-8B、Mistral-7B、Gemma-2-9B）上与SFT、Lisa、BEA、ASFT、SafeInstr、SPPFT 等基线对比，CWAC 在保持或提升下游准确率的同时，将有害分数从 50%+ 降至约 10% 左右，表现最优。

**⚠️ 局限性**

局限性包括需要白盒访问模型、依赖稀疏自编码器的质量、仅在中等规模模型上验证，且对更高级 jailbreak 或更大模型的鲁棒性尚未评估。

---

## 264. Traffic-Aware Domain Partitioning and Load-Balanced Inter-Domain Routing for LEO Satellite Networks

**arXiv ID:** 2604.12382 | [PDF](https://arxiv.org/pdf/2604.12382v1)

**作者:** Chen Zhou `[一作]` (Chongqing University of Posts and Telecommunications), Yongyi Ran `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 808 | [OpenAlex ID](https://openalex.org/A5016656803)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出DTAR框架，结合离线NSGA-II域划分和在线GAT‑PPO深度强化学习，实现LEO卫星网络的流量感知、负载平衡的跨域路由。

**💡 创新点**

创新点在于双阶段设计：多目标NSGA-II流量感知域划分与实时GAT状态编码交互，以及动作屏蔽PPO在动态拓扑和链路失效环境下保证可达性。

**🔧 技术方法**

使用多目标NSGA-II进行域划分，图注意网络（GAT）进行域级状态编码，动作屏蔽策略的PPO强化学习做在线路由决策。

**📊 数据集**

使用288颗Walker星座模拟数据，并结合自定义的时空流量与链路失效模型。

**📈 对比分析**

与Dijkstra、ELB、QRLSN、CDPAR四种基线在正常、流量高峰、链路失效三种场景下对比，DTAR在负载均衡系数CV、端到端延迟、丢包率和成功率等指标上均优于基线。

**⚠️ 局限性**

仅在域级路由层面优化，未联合考虑域内路由，且在异构星座上的泛化性尚未验证。

---

## 265. Modality-Agnostic Prompt Learning for Multi-Modal Camouflaged Object Detection

**arXiv ID:** 2604.12380 | [PDF](https://arxiv.org/pdf/2604.12380v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 266. Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning

**arXiv ID:** 2604.12374 | [PDF](https://arxiv.org/pdf/2604.12374v1)

**作者:** NVIDIA `[一作]` (NVIDIA), Zuhair Ahmed `[通讯]` (NVIDIA)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在120B参数（12B活跃）规模的混合Mamba‑Attention MoE模型Nemotron 3 Super上进行预训练、后训练与量化，支持最多1M上下文长度。

**💡 创新点**

创新点在于：①首次在大型模型中结合LatentMoE和MTP技术提升参数/ FLOP效率与推理速度；②采用NVFP4低精度预训练和混合量化策略；③通过AutoQuantize实现精度自适应分配，保持FP4高效的同时减少精度损失。

**🔧 技术方法**

使用的技术包括：Hybrid Mamba‑Attention Mixture‑of‑Experts架构、LatentMoE、Multi‑Token‑Prediction（MTP）推理加速、NVFP4预训练、强化学习（RLVR、SWE‑RL、RLHF）后训练、FP8/FWFP4 Post‑Training Quantization以及Model‑Optimizer AutoQuantize。

**📊 数据集**

数据集涵盖：25万亿token的预训练语料；SFT与RLHF数据来自多种agentic环境和工具使用场景；量化校准采用SFT数据的256条256k上下文样本；另外发布了专门针对代码、算法、逻辑等的Synthetic数据集。

**📈 对比分析**

与GPT‑OSS‑120B和Qwen3.5‑122B比较时，Nemotron 3 Super在大多数基准（MMLU‑Pro、GPQA、LiveCodeBench、SWE‑Bench、RULER、MMLU‑ProX等）保持或略高准确度，同时在8k输入/64k输出推理吞吐量上分别提高约2.2×和7.5×；在长上下文和多语言任务上也表现出色。

**⚠️ 局限性**

局限性包括：量化后的模型在极端长上下文或高精度任务中仍可能出现微小精度下降；当前的推理加速主要在NVIDIA Hopper/Blackwell GPU上实现，对其他硬件兼容性待验证；以及模型规模和训练成本仍高，限制了更广泛的部署。

---

## 267. Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness

**arXiv ID:** 2604.12373 | [PDF](https://arxiv.org/pdf/2604.12373v1)

**作者:** Tomer Ashuach `[一作]` (Technion -- Israel Institute of Technology), Yonatan Belinkov `[通讯]` (Technion -- Israel Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估大型语言模型是否拥有关于自身答案正确性的内部“特权”信息，比较自我探针与外部模型探针的预测性能。

**💡 创新点**

提出通过在相互矛盾子集上评估来消除模型间一致性带来的混淆，揭示事实知识任务中存在特权信号，而数学推理任务中不存在。

**🔧 技术方法**

使用线性与 MLP 探针、AUC 评估、对比自我探针、交叉模型探针和嵌入模型探针，并对层级进行逐层分析。

**📊 数据集**

事实知识集 Mintaka、TriviaQA、HotPotQA；数学推理集 GSM1K、MATH；模型 Qwen-2.5-7B、Llama-3.1-8B、Gemma-2-9B。

**📈 对比分析**

在完整数据集上自我探针与外部探针无显著优势；在相互矛盾子集上，事实知识自我探针平均提升约5%，数学推理无提升；层级分析显示事实知识优势从中层起增强。

**⚠️ 局限性**

仅限 7B–9B 规模模型；仅评估事实与数学，未覆盖编程、常识等；探针方法可能无法捕获所有特权信号；缺乏因果干预验证。

---

## 268. Adaptive Spiking Neurons for Vision and Language Modeling

**arXiv ID:** 2604.12365 | [PDF](https://arxiv.org/pdf/2604.12365v1)

**作者:** Chenlin Zhou `[一作]` (Peking University), Yonghong Tian `[通讯]` (Peking University)

**通讯引用:** 16087 | [OpenAlex ID](https://openalex.org/A5023918894)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Adaptive Spiking Neuron (ASN) 及其规范化变体 NASN，解决传统 SNN 神经元在训练效率、适应性、兼容性与脉冲驱动推理方面的缺陷。

**💡 创新点**

引入可学习的位移参数 α 以自适应调节膜电位阈值，并结合整数训练+脉冲推理与归一化机制，实现高效训练、动态适应与兼容主流 SNN 结构。

**🔧 技术方法**

基于整数训练与脉冲推理框架、STE 反向传播、归一化处理，集成到 Spikingformer 等 Transformer 结构。

**📊 数据集**

在 19 个视觉与语言数据集上验证，包括 ImageNet‑1K、CIFAR‑10/100、GLUE、ARC、QAT、Commonsense Reasoning 等。

**📈 对比分析**

与 LIF、PLIF、PSN、ILIF、NILIF 等基础神经元以及 Spikformer、SD‑Transformer、Spikingformer 等基线对比，在 ImageNet‑1K 上提升至 75.53% Top‑1、CIFAR 上提升 0.3‑0.7%，GLUE 平均准确率 67.5%，QAT/CRT 也均优于基线。

**⚠️ 局限性**

能耗评估仅为理论估算，未在真实神经形态硬件上测量；对 DVS 等实时事件任务的验证尚未完成。

---

## 269. PrivEraserVerify: Efficient, Private, and Verifiable Federated Unlearning

**arXiv ID:** 2604.12348 | [PDF](https://arxiv.org/pdf/2604.12348v1)

**作者:** Parthaw Goswami `[一作]` (Khulna University of Engineering & Technology), Ashfak Yeafi `[通讯]` (Khulna University of Engineering & Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种统一的联邦忘记框架PrivEraserVerify，能够高效地移除离线客户端的贡献，同时保证隐私和可验证性。

**💡 创新点**

创新点在于三项技术的结合：自适应检查点保存加速恢复、层级差分隐私校准精准噪声注入以及基于指纹的轻量化验证机制。

**🔧 技术方法**

采用自适应检查点、层级差分隐私、指纹嵌入与验证、FedAvg聚合、Gaussian噪声注入等技术。

**📊 数据集**

在CIFAR-10、FEMNIST以及医学X光图像数据集上进行实验。

**📈 对比分析**

与FedEraser、FedRecovery、VeriFi等基线相比，PrivEraserVerify在保留准确率（与全量重训练相比差值≤3%）的前提下，最快可达2–3倍的移除速度，且提供正式的DP不可区分性保证和成功的指纹验证。

**⚠️ 局限性**

局限性包括仅支持单客户端移除、对跨机房异构环境的适配尚待验证，以及在极端噪声水平下可能仍存在精度下降。

---

## 270. Detecting Precise Hand Touch Moments in Egocentric Video

**arXiv ID:** 2604.12343 | [PDF](https://arxiv.org/pdf/2604.12343v1)

**作者:** Huy Anh Nguyen `[一作]` (Adelaide University), Minh Hoai `[通讯]` (Adelaide University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

研究在第一人称视频中精确检测手与物体接触瞬间的帧级事件；

**💡 创新点**

提出 Hand-informed Context Enhanced (HiCE) 模块，通过跨注意力将局部手部特征与全局上下文融合，并采用软标签、抓握监督与高斯位移校正；

**🔧 技术方法**

使用跨注意力、T‑DEED 级时间编码、跨模态手部检测与回归、抓握分类辅助，训练时结合 soft‑label 与 Gauss‑TOR；

**📊 数据集**

新构建 TouchMoment 数据集（4021 片视频，8456 触碰事件）并基于 HOI4D、TACO 两大 egocentric 数据集进行实验；

**📈 对比分析**

与 E2E‑Spot、UGLF、T‑DEED 等端到端基线对比，HiCE 在 δ=0、1、2 的 AP 及 mAP 上提升 10–30+ 点，最高 AP@δ=2 超过 70%；

**⚠️ 局限性**

受限于手部检测精度与极短触碰帧的可辨识性，且对多手同时接触时的位移窗口敏感，未来需更鲁棒的手部定位与更细粒度的时序建模。

---

## 271. Forecasting the Past: Gradient-Based Distribution Shift Detection in Trajectory Prediction

**arXiv ID:** 2604.12425 | [PDF](https://arxiv.org/pdf/2604.12425v1)

**作者:** Michele De Vita `[一作]` (Friedrich-Alexander-Universität Erlangen-Nürnberg), Vasileios Belagiannis `[通讯]` (Friedrich-Alexander-Universität Erlangen-Nürnberg)

**通讯引用:** 3971 | [OpenAlex ID](https://openalex.org/A5027065196)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

对已训练的轨迹预测模型进行后置梯度基分布偏移检测。

**💡 创新点**

通过自监督“预测过去”任务训练解码器，并使用其梯度范数作为异常分数，避免影响原模型性能。

**🔧 技术方法**

自监督解码器、梯度范数、Mixture Density Network、Transformer/HiVT、PPO 等技术。

**📊 数据集**

Shifts 车辆运动预测数据集、Argoverse 1 动态预测数据集，以及 Highway 仿真环境。

**📈 对比分析**

与 RIP-BC、RIP-DIM、lGMM、OC‑SVM、IF、KDE 等基线比较，在 Shifts 上实现 71% AUROC（远超 56.8%），Argoverse 上 71–79% AUROC，Highway 上 96–99% AUROC。

**⚠️ 局限性**

对 Argoverse 侧向/速度异常梯度表现不稳定，方法对模型训练收敛敏感，仅适用于已冻结编码器，需要额外解码器训练。

---

## 272. Decoding by Perturbation: Mitigating MLLM Hallucinations via Dynamic Textual Perturbation

**arXiv ID:** 2604.12424 | [PDF](https://arxiv.org/pdf/2604.12424v1)

**作者:** Sihang Jia `[一作]` (Hong Kong University of Science and Technology), Xuming Hu `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1236 | [OpenAlex ID](https://openalex.org/A5057914558)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种训练-free的推理时干预框架DeP，通过动态文本扰动和注意力方差分析，抑制多模态大语言模型在推理过程中的语言先验驱动幻觉。

**💡 创新点**

创新点包括：1）将幻觉视为视觉注意力对文本表面变化的过度敏感；2）使用多层文本扰动作为探针；3）通过多扰动注意力一致性分离稳定与可疑视觉区域；4）在隐藏状态空间和输出logit空间双重校正；5）自适应扰动强度N_eff和视觉互斥约束。

**🔧 技术方法**

使用的技术包括：动态文本扰动探测、注意力均值与方差统计、区域敏感度对比与对比融合、logit漂移校正、DINO视觉互斥检测、N_eff自适应扰动策略、不同级别的词替换（视觉属性弱化、高频先验替换、共现对抗干预）。

**📊 数据集**

实验数据集：POPE（对象检索类二分类）和MMHal-Bench（生成式幻觉评测），在LLaVA-1.5和InstructBLIP两大多模态大语言模型上进行评估。

**📈 对比分析**

与VCD、CICD、ClearSight、VTI、Nullu等五种无训练干预方法对比，DeP在POPE的准确率、精确率、召回率和F1均超过所有基线，尤其在Adversarial和Popular设置下提升显著；在MMHal-Bench的幻觉率从59.4%降至49.0%（LLaVA-1.5）和从93.8%降至83.3%（InstructBLIP），得分提升至2.83和1.14，达成SOTA水平。

**⚠️ 局限性**

局限性：1）对开放式或泛化性强的提示（如“描述图像细节”）扰动难以产生有效对比；2）文本扰动可能引入新的未预期偏差；3）需要进一步研究更细粒度、自适应的扰动策略或连续潜空间探针以减少扰动带来的副作用。

---

## 273. ReasonXL: Shifting LLM Reasoning Language Without Sacrificing Performance

**arXiv ID:** 2604.12378 | [PDF](https://arxiv.org/pdf/2604.12378v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 274. Why and When Visual Token Pruning Fails? A Study on Relevant Visual Information Shift in MLLMs Decoding

**arXiv ID:** 2604.12358 | [PDF](https://arxiv.org/pdf/2604.12358v1)

**作者:** Jiwan Kim `[一作]` (Korea Advanced Institute of Science and Technology), Chanyoung Park `[通讯]` (Korea Advanced Institute of Science and Technology)

**通讯引用:** 2064 | [OpenAlex ID](https://openalex.org/A5101629749)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在多模态大语言模型中，提出一种在解码阶段动态检测并恢复视觉信息的 token pruning 框架，以解决传统预填阶段静态剪枝在视觉推理任务中的性能下降。

**💡 创新点**

核心创新是识别并应对“相关视觉信息位移”（RVIS），通过在解码时实时检测注意力漂移并进行上下文保持的 token 替换，实现对推理过程中的视觉需求自适应匹配。

**🔧 技术方法**

技术方法包括：1）Relevant Visual Information Shift Detect (RISD) 用余弦相似度阈值检测注意力位移；2）Context‑Preserving Visual Token Swap (CPTS) 将新检索的高相关 token 与原始 token 进行并集，并在一定步长内保持；3）与现有 FastV、DivPrune、VisionZip 等静态剪枝方法无缝集成，保持训练无关。

**📊 数据集**

实验数据集：视觉推理——MathVerse、WeMath、DynaMath、LogicVista、MMMU‑Pro；视觉理解——SQA、VQA^T、GQA；模型基准——Qwen3‑VL‑4B 与 InternVL3.5‑8B。

**📈 对比分析**

对比传统剪枝方法，在 VMR 任务上提升 15‑20% 准确率，尤其在 MathVerse 上从 32.23% 提升至 52.54%；在 VQA/SQA 任务上保持或略增准确率，整体计算成本与 FLOPs 仅略有提升，保持高效。

**⚠️ 局限性**

局限性包括：①仅在解码阶段动态恢复，仍需预填阶段剪枝，若预填阶段失去关键信息可能难以恢复；②检测阈值和上下文保持长度需手动调参；③在极大 token 预算或非常长推理任务时，临时 token 数目增多可能影响延迟；④实验仅覆盖两款 MLLM 与少数剪枝方法，泛化性待进一步验证。

---

## 275. Beyond Output Correctness: Benchmarking and Evaluating Large Language Model Reasoning in Coding Tasks

**arXiv ID:** 2604.12379 | [PDF](https://arxiv.org/pdf/2604.12379v1)

**作者:** Yuangang Li `[一作]` (University of California, Irvine), Iftekhar Ahmed `[通讯]` (University of California, Irvine)

**通讯引用:** 1456 | [OpenAlex ID](https://openalex.org/A5078115464)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7`

**🎯 论文内容**

构建了跨生成、摘要与分类三大任务的代码推理质量评测基准 CodeRQ‑Bench，并基于该基准提出了两阶段的推理评估方法 VERA；同时对现有推理评估器进行系统误判分析。

**💡 创新点**

①首次在编码任务中提出全任务维度的推理质量评估基准；②梳理出五类通用评估局限并据此提炼四条设计洞见；③设计了结合证据检验与歧义校正的两阶段评估器 VERA，显著提升推理可信度。

**🔧 技术方法**

使用 LLM‑as‑Judge（GPT‑4o‑mini）配合自助网络搜索进行证据检验；在第二阶段引入歧义度与处理质量评估实现分数校正；对比实验采用多种现有评估器（RECEVAL、SOCREVAL、CaSE 等）并使用 AUCROC、AUPRC、Spearman、Somers D 等指标。

**📊 数据集**

基准数据来源于 CoderEval‑RE、SWEbench‑RE、Modified‑ClassEval‑RE 与 BugDetection‑RE，所有实例通过 GPT‑4o 生成推理链并由三名专家标注推理正确性。

**📈 对比分析**

在四个数据集上与六个基线进行对比，VERA 在所有度量上均优于基线，AUCROC 提升可达 0.26，AUPRC 提升 0.21，且 Spearman 与 Somers D 均显著提高，表明对推理正确性的判别更精准。

**⚠️ 局限性**

目前基准仅覆盖受控实验场景，未涉及交互式调试、工具集成工作流或大规模仓库上下文推理；评估依赖大型 LLM，成本与可复现性受限。

---

## 276. KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates

**arXiv ID:** 2604.12397 | [PDF](https://arxiv.org/pdf/2604.12397v1)

**作者:** Yudong Li `[一作]` (Shenzhen University), Linlin Shen `[通讯]` (Shenzhen University)

**通讯引用:** 11489 | [OpenAlex ID](https://openalex.org/A5019313200)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并验证了一种知识坐标条件化(KoCo)方法，通过将文档映射到三维语义坐标（来源、内容、稳定性）并作为前缀加入预训练，提升LLM对知识结构的感知。

**💡 创新点**

用三维知识坐标取代传统URL/元数据，将文档在知识空间中定位，既兼顾多维度属性又保持客观性，显著提升预训练效率并降低幻觉。

**🔧 技术方法**

基于LLaMA架构和MeCo预训练框架，使用轻量级Qwen‑3‑4B标签器生成坐标；在预训练目标中加入条件化前缀，并对比持续预训练、数据选择等基线。

**📊 数据集**

主要使用DCLM web文本语料，随机抽取100 GB子集；在0.3B/0.6B/1.6B模型上进行从零预训练实验。

**📈 对比分析**

对比URL前缀、标准CPT、数据选择等基线，在10项下游任务（COPA、ARC‑E、ARC‑C、OBQA、PIQA、SIQA、LogiQA、TruthfulQA等）上评估，KoCo平均分数提升约0.96分；预训练收敛加速约30%；在TruthfulQA中提升3.78分，显著降低幻觉。

**⚠️ 局限性**

目前仅在1.6B规模验证，需进一步验证更大模型；依赖标签器，标签质量和潜在偏差可能限制效果；对大规模多样化语料的普适性待进一步探讨。

---

## 277. Fully Dynamic Breadth First Search and Spanning Trees in Directed Graphs

**arXiv ID:** 2604.12370 | [PDF](https://arxiv.org/pdf/2604.12370v1)

**作者:** Gregory Morse `[一作]` (ELTE Eötvös Loránd University), Tamás Kozsik `[通讯]` (ELTE Eötvös Loránd University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了在有向图中对广度优先搜索树（BFST）及其层次信息和全局BFS编号进行全动态维护的框架；

**💡 创新点**

创新点在于：①同时维护BFST、深度层次和全局BFS顺序；②通过分层扫描和局部更新避免全局重构；③采用最小化扫描区域的策略，使得增删边时仅影响受影响的子图；

**🔧 技术方法**

使用的技术包括：BFST结构（父亲、子树、层次），全局BFS编号（bfs_int、bfs_revint、bfs_level），分层遍历（thisLevel、nextLevel）、监视集、离线扫描与重排；算法分为增量和递减两类，均基于局部层级扫描和一次性批量重编号；

**📊 数据集**

文中未给出实验数据集，主要为理论与算法框架的说明；

**📈 对比分析**

在最坏情况下每次更新耗时Θ(n+m)，但实际仅为受影响子图的大小 O(|A_V|+|A_E|)。与半动态方法相比，能够保持全局BFS顺序并在增删边后快速恢复；

**⚠️ 局限性**

局限性：1）最坏情况下仍为线性时间；2）对大规模图的实时更新性能尚未证明；3）缺乏实验验证与实际数据集的评估；4）仅适用于单源BFST，未扩展到多源或全图遍历。

---

## 278. The Parameterized Complexity of Vertex-Coloring Edge-Weighting

**arXiv ID:** 2604.12363 | [PDF](https://arxiv.org/pdf/2604.12363v1)

**作者:** Shubhada Aute `[一作]` (IIT Hyderabad), Geevarghese Philip `[通讯]` (Chennai Mathematical Institute)

**通讯引用:** 907 | [OpenAlex ID](https://openalex.org/A5031210702)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce`

**🎯 论文内容**

研究了 Vertex‑Coloring {0,1}‑Edge‑Weighting 及其预权重扩展的参数化复杂性。

**💡 创新点**

首次给出了在不同结构参数（最小反馈顶点集、树宽、顶点覆盖数）下的完整可判定边界：W[1]‑硬性、XP 以及在顶点覆盖数下的 FPT 算法，并提出了高效的 kernelization。

**🔧 技术方法**

主要技术包括：结构参数化归约、基于最小颜色潜能的潜能函数、等价类压缩、树分解上的动态规划以及针对预权重情况的扩展分析。

**📊 数据集**

论文为理论研究，未使用任何实验数据集；所有结果均通过严谨的证明获得。

**📈 对比分析**

与已有的 NP‑硬性/多项式可解结果对比，本文提供了更细致的参数化复杂性图谱；在顶点覆盖数下的 FPT 运行时间为 2^{O(k^4)}·n，XP 运行时间为 (Δ+1)^4·tw·n。

**⚠️ 局限性**

局限性包括：预权重版本仅在预权重全为 1 的特殊情形下得到 FPT 结果；对一般预权重情况仍未确定是否 FPT；此外，树宽参数的 XP 复杂度尚不一定最优。

---

## 279. MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents

**arXiv ID:** 2604.12352 | [PDF](https://arxiv.org/pdf/2604.12352v1)

**作者:** Joongmin Shin `[一作]` (Korea University), Heuiseok Lim `[通讯]` (Korea University)

**通讯引用:** 49852 | [OpenAlex ID](https://openalex.org/A5027447596)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 MultiDocFusion 一套多模态文档分块管线，结合视觉布局解析（DP）、OCR、基于 LLM 的层级结构解析（DSHP-LLM）和 DFS 组块，旨在提升长工业文档的检索与问答质量。

**💡 创新点**

创新点包括：①将视觉布局信息与文本内容统一建模，显式捕获表格、图表等视觉元素；②构建 DSHP-LLM，使用指令微调的 LLM 对章节层级进行准确重构；③DFS 组块策略在保持层级语义的同时生成结构化块，减少上下文碎片化；④在多页面 VQA 评测中系统验证了该方法相对传统文本分块的显著优势。

**🔧 技术方法**

技术栈：视觉解析使用 DETR/VGT；OCR 使用 Tesseract/EasyOCR/TrOCR；层级解析采用 Llama‑3.2‑3B、Qwen‑2.5‑3B、Mistral‑8B 等 LLM，并通过 LoRA 进行指令微调；检索嵌入使用 BGE/E5/BM25；RAG 与 LLM（如 Llama‑3.2‑3B）完成最终答案生成。

**📊 数据集**

数据集：DSHP‑LLM 训练与评估基于 DocHieNet 与 HRDH；问答与检索评测使用四大多页 VQA 数据集——DUDE、MPVQA、CUAD、MOAMOB。

**📈 对比分析**

与 Length、Semantic、LumberChunker、Perplexity 以及仅基于结构的分块方法进行对比。MultiDocFusion 在检索方面提升 8–15% 的精确度，QA 的 ANLS 约提升 2–3%，在所有评测集上均取得最高的召回、精确度与 nDCG 组合分。

**⚠️ 局限性**

局限性：1）DSHP‑LLM 仍缺乏细粒度视觉特征（字体、排版等）的直接利用；2）未对图结构检索与推理进行系统评估；3）多模块流水线易受错误传播，尤其是 DP/OCR 阶段；4）层级块复制导致索引尺寸增大，检索延迟和存储成本上升。

---

## 280. Heuristic Classification of Thoughts Prompting (HCoT): Integrating Expert System Heuristics for Structured Reasoning into Large Language Models

**arXiv ID:** 2604.12390 | [PDF](https://arxiv.org/pdf/2604.12390v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 281. Responsible Trauma Research: Designing Effective and Sustainable Virtual Reality Exposure Studies

**arXiv ID:** 2604.12349 | [PDF](https://arxiv.org/pdf/2604.12349v1)

**作者:** Annalisa Degenhard `[一作]` (Ulm University), Stefan Tschoeke `[通讯]` (Ulm University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在复杂创伤后应激障碍（C-PTSD）患者中实施并评估虚拟现实暴露疗法（VRET），探索触发物的识别、设计与迭代流程，并重点关注治疗者、开发者与患者三方的安全与协作。

**💡 创新点**

提出将设计过程本身视为治疗环节，强调个体化触发物（往往是单一物体）足以有效；揭示VRET效果与沉浸感无必然关联；首次系统记录非临床技术人员参与疗程时的情绪负荷与角色冲突，并给出方法学建议。

**🔧 技术方法**

使用Unity 3D与现成资产构建场景，Vive Pro Eye头显与Nexus10生理监测设备，并结合IPQ、PTSS‑10、IES‑R等量表收集数据。

**📊 数据集**

没有公开数据集，数据来源为11名住院患者的临床记录、问卷与生理信号。

**📈 对比分析**

未进行对照实验，主要以定性主题分析与生理指标（皮肤电反应、心率）为佐证；结果显示单一物体触发可产生与复杂场景相当甚至更高的情绪激活，且存在较低沉浸感。

**⚠️ 局限性**

样本规模小、无对照组、仅限两类创伤（军事与童年），受访者声音间接获取，且研究过程高度依赖临床与技术人员的主观记录，限制了推广性与效度验证。

---

## 282. HARP: Hadamard-Domain Write-and-Verify for Noise-Robust RRAM Programming

**arXiv ID:** 2604.12420 | [PDF](https://arxiv.org/pdf/2604.12420v1)

**作者:** Ilhuan Choi `[一作]` (Seoul National University), Woo-Seok Choi `[通讯]` (Seoul National University)

**通讯引用:** 6234 | [OpenAlex ID](https://openalex.org/A5038971321)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

提出了两种基于 Hadamard 编码的 RRAM 写-验证（Write‑and‑Verify）方案——HD‑PV 与 HARP，以提升 verify 读取的可靠性并降低能耗。

**💡 创新点**

创新点：① 用 Hadamard 编码替代传统的一热读取，实现方差 1/N 降噪并抵消常模噪声；② 发现 WV 本质是分类任务，可采用轻量级 compare‑only ADC 取代全 SAR 转换，从而显著降低 ADC 负担。

**🔧 技术方法**

使用 Hadamard 码的并行读取与逆解码、列级写与验证流程、compare‑only ADC 模式、1T‑1R RRAM 交叉口、数字加法器实现、仿真工具（NeuroSim）及电路级模型。

**📊 数据集**

使用数据集：CIFAR‑10、CIFAR‑100 与 Google Speech Commands 关键词识别（KWS）。

**📈 对比分析**

通过与传统 CW‑SC 及 5‑读平均（5‑read averaging）在相同内存占用、相同 ADC 分辨率下进行对比。实验显示 HD‑PV 在误差上比 CW‑SC 降低 3.7×、迭代次数缩减 3.2×；HARP 在误差上降低 2.2×、迭代次数缩减 1.6×；能耗比 CW‑SC 低 6.2–9.5×；延迟比 6.1–3.5×；在 64×64 大规模阵列下，准确率仅损失 <3%。

**⚠️ 局限性**

局限性：HD‑PV 仍需完整 SAR ADC，能耗相对较高；HARP 对阈值 τ_w 敏感，需手工调参；实验仅在仿真层面验证，缺乏硬件实现；仅针对 1T‑1R RRAM，未验证对其他结构的适用性；在极低噪声环境下，Hadamard 编码优势不明显。

---

## 283. Do Transformers Use their Depth Adaptively? Evidence from a Relational Reasoning Task

**arXiv ID:** 2604.12426 | [PDF](https://arxiv.org/pdf/2604.12426v1)

**作者:** Alicia Curth `[一作]` (Microsoft Research Cambridge), Niranjani Prasad `[通讯]` (Microsoft Research Cambridge)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型语言模型在多跳关系推理任务中是否会根据任务难度动态使用网络深度进行了系统性实验与分析。

**💡 创新点**

首次在可控的单词答案推理任务上结合logit lens与因果补丁技术，揭示预训练与微调模型在深度利用上的差异，并指出微调策略对保留通用语言建模能力的影响。

**🔧 技术方法**

使用logit lens（对中间隐藏状态进行直接解码）和因果补丁（介入单层隐藏状态恢复原预测）技术来跟踪答案可解码性与信息流动。

**📊 数据集**

构建并改造CLUTRR家族关系生成器，生成2–10跳（及更长跳）的单词答案推理实例；对多种开源模型族（GPT‑2、Pythia、Phi、Qwen、LLaMA）进行实验。

**📈 对比分析**

对预训练模型按模型规模与层数进行对比；对微调模型分为LoRA（仅调优注意力）和全微调（仅回答token监督）两种方案。实验表明：预训练模型在较易任务上可在更早层级给出可行答案，链长越长需要更多层进行跨词信息整合；全微调模型更明显地将深度分配给难度较高的任务，而LoRA微调模型的深度使用与基线相似。整体精度与层数曲线符合预期，但单词答案的限制与最终层“扩散”效应等因素限制了通用性能。

**⚠️ 局限性**

局限性包括：仅使用单词答案的受限任务可能掩盖多词答案的深度使用；因果补丁计算成本高，导致实验规模受限；对不同模型族的训练差异未做充分归因，导致对残差流对齐机制的解释尚不完整。

---

## 284. SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models

**arXiv ID:** 2604.12377 | [PDF](https://arxiv.org/pdf/2604.12377v1)

**作者:** SungHo Kim `[一作]` (Korea University), SangKeun Lee `[通讯]` (Korea University)

**通讯引用:** 1686 | [OpenAlex ID](https://openalex.org/A5028945187)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种可插拔模块（SCRIPT），将韩文字母子字符（Jamo）组合的结构知识注入现有子词级预训练语言模型，以提升韩语NLP任务性能。

**💡 创新点**

创新点在于：①利用韩文字母的三层组合原则（Choseong‑Jungseong‑Jongseong）设计双阶段压缩（子字符→字符→子词）实现结构感知；②通过跨注意力融合结构化子词表示与原子词嵌入，兼顾语义与形态信息；③模块无须改动模型架构或额外预训练，兼容多种韩语PLM。

**🔧 技术方法**

主要技术包括：子字符分解、基于GRU的序列组合、卷积+池化的字符压缩、子词压缩、跨注意力融合；使用的编程框架为PyTorch；训练与微调采用标准Transformer层。

**📊 数据集**

使用的主要数据集有：韩语POS标注语料库（约3M词）用于分析形态变体；九个Korean NLU基准（KorNLI、KorSTS、NSMC、PAWS‑X、KB‑BoolQ、KB‑COPA、KB‑WiC、KB‑HellaSwag、KB‑SentiNeg）；七个Korean NLG基准（KoCommonGen、XL‑Sum、Korean GEC、Kor‑Learner等）。

**📈 对比分析**

通过在同一模型（KoGPT2、KoGPT3、EXAONE、BERT）上进行对照实验，将SCRIPT作为增量微调模块，结果显示：所有基准均提升，平均提升1.6%点；在生成任务上提升1.4–3.5%点；与基于Jamo的KOMBO相比，SCRIPT在大模型上表现更好，且训练成本低。

**⚠️ 局限性**

限制包括：①仅在韩语上验证，跨语言适用性尚未系统评估；②对更大规模模型（>7B）未做实验，效能差异未知；③模块在嵌入层增加额外参数和序列长度相关计算，推理成本略升高。

---

## 285. Cooperative Memory Paging with Keyword Bookmarks for Long-Horizon LLM Conversations

**arXiv ID:** 2604.12376 | [PDF](https://arxiv.org/pdf/2604.12376v1)

**作者:** Ziyang Liu `[一作]` `[通讯]` (Independent Researcher), Ziyang Liu (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `fede83ac-7505-405f-ab37-e7284695c47f` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了LLM会话记忆的合作分页方法，将被驱逐的对话段落压缩为约8–24个词的关键词书签，并为模型提供工具按需检索完整内容；

**💡 创新点**

创新点在于将关键词书签视为轻量级目录，改变检索从“猜测缺失信息”到“查阅索引”，通过最小化书签长度实现最佳准确率；系统化研究页面粒度与驱逐策略，揭示书签辨识是主要瓶颈，并提出两种改进书签生成的策略（hybrid与llm‑batch）；

**🔧 技术方法**

采用关键词抽取启发式、LLM调用进行书签生成、分页模拟器（支持多种页面边界和驱逐策略）、基于工具的检索调用、以及LLM评审模型进行质量评估；

**📊 数据集**

使用LoCoMo基准（10条真实多会话对话，300+轮）作为主要测试集；此外构造20条合成长会话和22条受控QA探针，用于分页与书签效果评测；

**📈 对比分析**

与全上下文、截断、BM25检索、词重叠检索、搜索工具基线等六种方法在四款LLM（GPT‑4o‑mini、DeepSeek‑v3.2、Claude Haiku、GLM‑5）上对比；在LoCoMo上Bookmark+Recall取得最高分（GPT‑4o‑mini 2.18/5，DeepSeek‑v3.2 2.74/5），超越截断、BM25、词重叠等；书签召回率约59%（LoCoMo），受控探针上达90.9%；改进书签策略可提升整体精度至+8.7分；

**⚠️ 局限性**

主要局限：评估依赖LLM评审缺乏人工验证；跨模型泛化仅在两款模型上验证；书签辨识错误率仍高达≈43%，未彻底解决；书签生成策略受启发式基线质量限制；假设模型始终配合工具调用；对极长会话（>100轮）的扩展尚未充分探索。

---

## 286. Is Sliding Window All You Need? An Open Framework for Long-Sequence Recommendation

**arXiv ID:** 2604.12372 | [PDF](https://arxiv.org/pdf/2604.12372v1)

**作者:** Sayak Chakrabarty `[一作]` (Northwestern University), Souradip Pal `[通讯]` (Purdue University)

**通讯引用:** 8 | [OpenAlex ID](https://openalex.org/A5035784469)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

实现并公开了一套完整的长序列推荐训练管线，采用滑动窗口训练技术，并在学术规模硬件上可复现

**💡 创新点**

① 通过运行时感知的 ablation 研究量化窗口大小与步长对准确率与计算量的权衡；② 提出了 k‑Shift Embedding 层，使百万级词表可在 commodity GPU 上实现，显著降低内存压力且几乎不损失准确率

**🔧 技术方法**

滑动窗口训练、k‑Shift 哈希+位移查询的 embedding、Transformer 编码器、Auto‑regressive 训练、混合最近/滑动窗口模式、stride ablation 等

**📊 数据集**

主要使用 Retailrocket 与 Taobao 两个公开交互序列数据集（以及在说明中提及的 MovieLens、Amazon Reviews 等）

**📈 对比分析**

与不使用滑动窗口的 baseline 进行对比：All‑Sliding 在 Retailrocket 上提升 MRR +6.04%、Recall +6.34%，但训练时间约增长 4×；Mixed‑500/1000 提升更小；stride ablation 展示了训练时间与性能的折中；k‑Shift Embedding 在大词表下保持几乎无性能损失

**⚠️ 局限性**

限制包括：k‑Shift 受哈希冲突影响，Taobao 词表过大导致实验受限；stride 较大步长会显著降低 MRR/Recall；实验评估仅覆盖 MRR、Recall、Perplexity，未覆盖完整排名指标；以及实验环境仍需 GPU 与显存支持，极端低资源下仍受限

---

## 287. Reading Between the Pixels: Linking Text-Image Embedding Alignment to Typographic Attack Success on Vision-Language Models

**arXiv ID:** 2604.12371 | [PDF](https://arxiv.org/pdf/2604.12371v1)

**作者:** Ravikumar Balakrishnan `[一作]` (Cisco AI), Ankit Garg `[通讯]` (Cisco AI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估了基于字体渲染的文本注入攻击在四种视觉语言模型上的成功率，并探讨了字体大小、视觉失真与模型特异性对攻击效果的影响。

**💡 创新点**

创新点在于系统性地量化字体大小、视觉失真与模型差异对攻击成功率的影响，并提出多模态嵌入距离可作为跨模型攻击效能的可靠指标。

**🔧 技术方法**

使用了多模态嵌入模型（JinaCLIP、Qwen3-VL-Embedding）计算文本与图像的L2距离，并对四个VLM（GPT‑4o、Claude Sonnet 4.5、Mistral‑Large‑3、Qwen3‑VL‑4B‑Instruct）进行实验。

**📊 数据集**

使用从SALAD‑Bench挑选的1,000条对抗性提示，并在不同字体大小（6–28px）和视觉变换（旋转、模糊、噪声、对比度等）下渲染成图像。

**📈 对比分析**

通过对比文本和图像两种输入方式以及不同视觉变换下的攻击成功率（ASR）发现，GPT‑4o和Claude在文本模式下成功率显著高于图像模式，而Qwen3‑VL和Mistral则相近；嵌入距离与ASR呈显著负相关（r≈‑0.71至‑0.94）。

**⚠️ 局限性**

局限性包括仅评估四个模型、单一字体与背景设置、未考虑多字体或手写文本，且对极长提示的可行性未验证。

---

## 288. Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors

**arXiv ID:** 2604.12359 | [PDF](https://arxiv.org/pdf/2604.12359v1)

**作者:** Rui Yin `[一作]` (Zhejiang University), Shouling Ji `[通讯]` (Zhejiang University)

**通讯引用:** 7944 | [OpenAlex ID](https://openalex.org/A5058611515)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种基于内部表征向量的后门注入方法，将激活层中的合规方向编译为权重更新，仅在触发器出现时激活，从而实现安全对齐模型的隐藏攻击。

**💡 创新点**

创新点在于：① 将动态激活向量通过闭式正则化最小二乘求解编译进模型权重；② 通过零空间投影确保仅在触发器激活时产生影响，保持对清洁输入的不可检测性；③ 相比传统基于词级映射的后门，该方法能持续诱导有害输出并抑制安全回退。

**🔧 技术方法**

主要技术包括：后期权重编辑、激活向导、差分均值（Difference-in-Means）提取合规方向、奇异值分解（SVD）求零空间投影、正则化闭式最小二乘求解权重增量。

**📊 数据集**

使用的数据集包括：Databricks Dolly（10,000条正常提示）、AdvBench（256条有害查询）用于训练；StrongREJECT、Misuse、DNA、DAN 作为破解基准；AlpacaEval、GSM‑8K、TruthfulQA、XSTest 用于评估通用性能与防御效果。

**📈 对比分析**

与 ROME、JailbreakEdit、DualEdit、SFT 等基线比较，实验显示：攻击成功率（ASRw）常位于前列，安全回退率（FR）低于 15%（大多数案例低于 7%），在不触发时攻击成功率（ASRw/o）保持极低，且通用性能保持 97% 以上（Utility Retention Rate ≥97%). 综上，该方法在效果、隐蔽性和兼容性方面均优于现有技术。

**⚠️ 局限性**

局限性包括：① 需要手动调节合规向量强度 α，影响攻击与正常行为的平衡；② 零空间投影依赖有限的参考数据，若真实输入分布偏离，干扰保证可能失效；③ 仅在单一触发器与单方向编译上验证，未覆盖多触发器或多任务情景；④ 评估范围仅涵盖部分模型与防御策略，未涵盖更强适应性防御或其他架构。

---

## 289. OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion

**arXiv ID:** 2604.12356 | [PDF](https://arxiv.org/pdf/2604.12356v1)

**作者:** Dongjian Yu `[一作]` (Yunnan University), Shuqiang Jiang `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 6496 | [OpenAlex ID](https://openalex.org/A5085719285)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `67630363-6be0-4f51-ab05-7198250671a5` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过单张RGB图像预测食物营养信息，提出SSRA、FAFM、MPH三大模块；

**💡 创新点**

创新点包括：①构建全流程多模态的OmniFood8K数据集；②提出NutritionSynth-115K合成数据集；③将单目深度预测与多频域融合、动态掩码头相结合，实现更高精度；

**🔧 技术方法**

采用单目深度估计、Scale‑Shift Residual Adapter、频域对齐融合（FFT）、Mask‑based Prediction Head、动态任务权重损失和交叉模态对齐损失等技术；

**📊 数据集**

使用OmniFood8K、NutritionSynth-115K（115k）以及对照的Nutrition5k数据集；

**📈 对比分析**

在OmniFood8K和Nutrition5k上与多种基准（如ResNet、Swin、RGB‑D模型等）对比，PMAE均达到最低，预训练后性能进一步提升；

**⚠️ 局限性**

限制：仍依赖单视角RGB，深度预测误差可能影响；合成数据与真实场景差异需进一步验证；数据集聚焦中文菜系，跨文化推广尚未评估。

---

## 290. Combating Pattern and Content Bias: Adversarial Feature Learning for Generalized AI-Generated Image Detection

**arXiv ID:** 2604.12353 | [PDF](https://arxiv.org/pdf/2604.12353v1)

**作者:** Haifeng Zhang `[一作]` (Chongqing University of Posts and Telecommunications), Bin Xiao `[通讯]` (Chongqing University of Posts and Telecommunications)

**通讯引用:** 20074 | [OpenAlex ID](https://openalex.org/A5103218891)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种多维对抗特征学习框架MAFL，用于提升AI生成图像检测的跨模型泛化能力，尤其针对生成模式与内容偏差问题。

**💡 创新点**

创新点在于将生成模式偏差与内容偏差视为不对称偏差学习，并通过一个多维对抗损失（熵最大化、特征对齐、标签逆转）实现对抗训练，迫使模型聚焦于不同生成模型共享的本质特征。

**🔧 技术方法**

采用CLIP预训练多模态图像编码器作为特征提取器，并构建实时/伪真分类网络与偏差学习网络，使用对抗训练、熵最大化、特征对齐和标签逆转等技术。

**📊 数据集**

使用多种公开数据集（Holmes、ForenSynths、GenImage）共25个生成模型（GAN、扩散、VAE等），并在不同协议下进行交叉评估。

**📈 对比分析**

与目前最先进的VIB‑Net、Effort等方法比较，MAFL在多协议下平均提升约10%准确率、8%平均精度，并在仅320张训练样本时仍可达80%+准确率，表现出更强的泛化与鲁棒性。

**⚠️ 局限性**

局限在于仍为黑盒判定，缺乏可解释性与生成来源归因；此外对极低多样性生成模型的细粒度检测能力尚未验证。

---

## 291. Unlocking the Potential of Grounding DINO in Videos: Parameter-Efficient Adaptation for Limited-Data Spatial-Temporal Localization

**arXiv ID:** 2604.12346 | [PDF](https://arxiv.org/pdf/2604.12346v1)

**作者:** Zanyi Wang `[一作]` (State Grid Corporation of China), Mengmeng Wang `[通讯]` (Zhejiang University of Technology)

**通讯引用:** 25513 | [OpenAlex ID](https://openalex.org/A5100422377)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

针对稀缺标注数据的时空视频定位任务，本文提出了一种参数高效的适配框架ST‑GD，利用冻结的Grounding DINO模型并注入轻量化的S‑T适配器、Temporal适配器、Temporal‑Diff适配器以及专门的Temporal Decoder，实现对空间与时间信息的高效学习与融合；

**💡 创新点**

创新点在于首次将PEFT方法引入STVG任务，设计专门的时空适配器与差分适配器实现时空建模，并通过语言引导的查询精炼机制与注意力解码器实现精确的时空边界预测，显著提升了小数据场景下的性能；

**🔧 技术方法**

技术主要包括：冻结的Grounding DINO视觉‑语言基础模型；S‑T适配器（双分支 2D/1D 卷积）注入时空感知；LoRA 对文本编码器进行低秩微调；Temporal Adapter 与 Temporal‑Diff Adapter 用于建模时间依赖；专门的Temporal Decoder 结合注意力与聚合模块预测事件边界；

**📊 数据集**

使用了 HC‑STVG v1、v2（专注多人人视频定位）和 VidSTG（包含复杂关系查询的更大规模数据集）进行评估；

**📈 对比分析**

在 HC‑STVG v1 上与全参数微调或部分微调模型（TubeDETR、STCAT、CG‑STVG、Video‑Grounding‑DINO）相比，ST‑GD 在保持仅约 10M 可训练参数的前提下，m_tIoU 最高可达 53.1%，m_vIoU 39.9%，且在 VidSTG 上同样取得与 SOTA 同等或略高的 m_tIoU 与 m_vIoU；

**⚠️ 局限性**

主要局限在于完全冻结的基础模型可能无法充分学习到视频特有的动态模式，对极其复杂或新颖的时空关系仍可能表现不佳；

---

## 292. Latent Planning Emerges with Scale

**arXiv ID:** 2604.12493 | [PDF](https://arxiv.org/pdf/2604.12493v1)

**作者:** Michael Hanna `[一作]` (University of Amsterdam), Emmanuel Ameisen `[通讯]` (Anthropic)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过定义“隐性规划”并利用特征电路对Qwen-3系列模型进行因果分析，研究LLM在无显式计划的情境下如何隐式规划；

**💡 创新点**

创新点在于提出基于稀疏特征的因果特征电路框架，用来系统评估并量化LLM的前向与后向规划能力，以及其随模型规模的演化；

**🔧 技术方法**

采用Transcoder训练稀疏特征、构建特征电路、进行干预与流量分析等技术，对模型内部规划表征进行可解释的因果推断；

**📊 数据集**

实验使用了三类简单语法一致性数据集（a/an、is/are、el/la）以及由Qwen-3 32B生成的韵律对联第一行样本作为测试集；

**📈 对比分析**

通过不同规模（0.6B–14B）模型的比较，发现14B模型在a/an任务中表现最优、韵律对联任务中能实现前向规划但后向规划不足，规模越大规划效果越好；

**⚠️ 局限性**

局限性包括仅针对单一模型族且规模有限，缺乏跨模型和更复杂任务的验证；特征电路与Transcoder的可扩展性受限，且对多步深层规划的证据不足。

---

## 293. Mining Large Language Models for Low-Resource Language Data: Comparing Elicitation Strategies for Hausa and Fongbe

**arXiv ID:** 2604.12477 | [PDF](https://arxiv.org/pdf/2604.12477v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 294. DeCoNav: Dialog enhanced Long-Horizon Collaborative Vision-Language Navigation

**arXiv ID:** 2604.12486 | [PDF](https://arxiv.org/pdf/2604.12486v1)

**作者:** Sunyao Zhou `[一作]` (Institute of Artificial Intelligence (TeleAI), China Telecom), Xuelong Li `[通讯]` (Institute of Artificial Intelligence (TeleAI), China Telecom)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 DeCoNav 这一去中心化长时程协作视觉‑语言导航框架，并创建了同步双机器人评估基准 DeCoNavBench。

**💡 创新点**

创新点在于将事件触发对话驱动的动态子任务重新分配与语义状态交换结合，并采用严格同步的并行执行保证真实交互。

**🔧 技术方法**

技术包括语义视觉总线（SVB）、事件驱动对话重规划（EDR）与同步并行执行（SPE），以及 ROVE 验证管道。

**📊 数据集**

使用 HM3D 场景构建的 DeCoNavBench，共 1,213 个任务。

**📈 对比分析**

与 CoNavBench 基线比较，DeCoNav 在双成功率、单机器人成功率、SPL、ISR 等指标上提升约 22% 至 47%，并在真实机器人部署中实现自适应子任务交换。

**⚠️ 局限性**

局限在于仍依赖人工审核的 ROVE 验证，通信延迟与语义压缩可能限制在更大规模多机器人环境中的鲁棒性。

---

## 295. Audio Source Separation in Reverberant Environments using $β$-divergence based Nonnegative Factorization

**arXiv ID:** 2604.12480 | [PDF](https://arxiv.org/pdf/2604.12480v1)

**作者:** Mahmoud Fakhry `[一作]` (Fondazione Bruno Kessler), Maurizio Omologo `[通讯]` (Fondazione Bruno Kessler)

**通讯引用:** 3710 | [OpenAlex ID](https://openalex.org/A5025736898)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

在混响环境下进行欠定多通道音源分离，提出基于β-散度的非负张量分解估计模型参数，并利用源先验信息进行分离。

**💡 创新点**

将β-散度与非负张量分解结合，能够控制稀疏性并利用源谱先验；同时提出通过库检测或提取谱基矩阵的半监督方法，避免了传统NMF对稀疏性的不足。

**🔧 技术方法**

采用局部高斯模型、多通道Wiener滤波、非负矩阵分解(NMF)、非负张量分解(NTF)、β-散度乘法更新（MU）算法，以及光滑Wiener滤波等。

**📊 数据集**

使用三套数据集：合成房间混响（T60 130/250/380 ms）、在隔音室录制的真实混响数据以及SISEC2013评估数据集（模拟和实录）。

**📈 对比分析**

与两种现有盲源分离算法比较，采用SDR、ISR、SIR、SAR指标；在训练好的先验下平均提升≈2.6 dB；在盲分离时在低混响（T60=130 ms）下可与或优于对手，在中等混响下对女性语音表现更好。

**⚠️ 局限性**

在高混响环境下对男性语音的分离效果下降，且对β取值敏感；对源谱先验的依赖可能限制在未知来源场景；仅使用β-散度的乘法更新对局部收敛速度和鲁棒性有限。

---

## 296. IAD-Unify: A Region-Grounded Unified Model for Industrial Anomaly Segmentation, Understanding, and Generation

**arXiv ID:** 2604.12440 | [PDF](https://arxiv.org/pdf/2604.12440v1)

**作者:** Haoyu Zheng `[一作]` (Zhejiang University), Feifei Shao `[通讯]` (Zhejiang University)

**通讯引用:** 256 | [OpenAlex ID](https://openalex.org/A5068107998)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一个双编码器统一框架，实现工业异常检测中的分割、基于区域的自然语言解释和基于掩码的缺陷生成三大任务的同时推理。

**💡 创新点**

核心创新在于将冻结的 DINOv2 区域专家与 Qwen3.5‑4B 视觉‑语言骨干通过占位符替换注入，从而在同一模型中共享区域信息，既提升理解精度，又保持生成的空间精确度。

**🔧 技术方法**

采用的技术包括：冻结 DINOv2 作为区域专家，Qwen3.5‑4B VLM+LoRA 作为共享骨干，轻量级占位符替换注入机制，跨模态注意力连接，基于 SD‑2 的掩码条件图像修复网络，以及分阶段联合训练策略。

**📊 数据集**

实验数据集包括新构建的 Anomaly‑56K（59,916 张图像，24 个工业类别，104 种缺陷变体）以及外部 MMAD 39,670 问答测试集。

**📈 对比分析**

在统一评测协议下，区域注入显著提升定位准确率（由 73.83% 提升至 93.28%）和 ROUGE‑L；分割 Dice 79.2%、IoU 69.2%；生成在掩码 PSNR 上比 SD‑2 提升约 1.5 dB，整体 PSNR 与 LPIPS 也均优于基线；在 MMAD 1‑shot 任务中，整体准确率提升至 72.5%。

**⚠️ 局限性**

局限性包括：依赖大规模 VLM 及冻结的 DINOv2 区域专家导致模型对新领域的迁移性受限；对极小或稀有缺陷的识别仍显不足；联合训练在生成细节上仍略逊于单独优化的模型。

---

## 297. Orthogonal Subspace Projection for Continual Machine Unlearning via SVD-Based LoRA

**arXiv ID:** 2604.12526 | [PDF](https://arxiv.org/pdf/2604.12526v1)

**作者:** Yogachandran Rahulamathavan `[一作]` (Loughborough University), Sangarapillai Lambotharan `[通讯]` (Loughborough University)

**通讯引用:** 5640 | [OpenAlex ID](https://openalex.org/A5009232291)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在连续机器遗忘任务中，提出将LoRA适配器的更新限制在前已学习任务正交补空间内，以避免参数碰撞。

**💡 创新点**

创新点在于在训练期间通过SVD得到的正交投影约束来确保各任务更新相互独立，并可实现静态融合而不需要运行时路由。

**🔧 技术方法**

采用LoRA、奇异值分解（SVD）、正交投影、受限优化等技术实现无冲突的连续遗忘更新。

**📊 数据集**

实验使用MNIST和CIFAR-100数据集，模型为ResNet-20（仅在分类头上插入LoRA）。

**📈 对比分析**

与传统静态LoRA、AC-LoRA、I-LoRA以及MoE路由等方法比较，本文方法在30个连续遗忘任务后仍保持约58%的保留准确率，并在遗忘准确率上取得56%的成绩，显著优于静态方法的12%并匹配或超越动态路由方法。

**⚠️ 局限性**

局限性包括正交子空间维度有限，随着任务数增多可用空间逐渐耗尽导致遗忘效果饱和；线性投影可能不足以捕捉复杂任务几何，且对极端容量条件下的遗忘深度有限。

---

## 298. Instantiating Bayesian CVaR lower bounds in Interactive Decision Making Problems

**arXiv ID:** 2604.12519 | [PDF](https://arxiv.org/pdf/2604.12519v1)

**作者:** Raghav Bongole `[一作]` (KTH Royal Institute of Technology), Mikael Skoglund `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 8946 | [OpenAlex ID](https://openalex.org/A5041348422)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

通过两点Hellinger模板，对Bayes CVaR（尾部风险）下的交互式决策问题（如高斯均值估计与两臂高斯老虎机）给出了显式下界；

**💡 创新点**

首次将泛化Fano方法与Hellinger距离结合，得到含α风险水平的下界；

**🔧 技术方法**

利用信息论工具（Hellinger距离、KL散度、Fano不等式）和两点极限构造；

**📊 数据集**

使用合成高斯样本与高斯老虎机奖励，不涉及公开数据集；

**📈 对比分析**

下界与传统Bayes风险下界同阶（O(n^{-1/2})或O(√T)），但保留α的显式依赖，展示了尾部风险的真实难度；

**⚠️ 局限性**

仅在两点案例中验证，未扩展至多臂或马尔可夫决策过程，且缺乏对应的匹配上界。

---

## 299. Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis

**arXiv ID:** 2604.12518 | [PDF](https://arxiv.org/pdf/2604.12518v1)

**作者:** Kang He `[一作]` (Wuhan University), Donghong Ji `[通讯]` (Wuhan University)

**通讯引用:** 3957 | [OpenAlex ID](https://openalex.org/A5058877618)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种两阶段的多模态情感分析框架EBMC，先通过语义拆解与跨模态增强提升弱模态表达，再通过能量引导的模态协调和实例级信任蒸馏实现模态贡献平衡与鲁棒融合。

**💡 创新点**

创新点包括：①语义拆解与跨模态补偿机制，显著强化弱模态的情感特征；②基于能量模型的模态协调器，通过能量差平衡与梯度流动抑制主导模态的竞争；③实例感知的模态信任蒸馏，按样本可靠性动态调整模态权重，提高在噪声或缺失模态条件下的稳健性。

**🔧 技术方法**

使用的技术包括：语义拆解（共享/特定子网络+InfoNCE对齐+余弦正则）、跨模态补偿网络（轻量级融合网络）、能量引导的模态协调（能量函数、能量差最小化、能量梯度更新）、实例感知信任蒸馏（教师概率不确定度→置信度、加权KL蒸馏）以及常规的多模态编码器与融合头。

**📊 数据集**

在三大主流情感数据集上评估：CMU-MOSI、CMU-MOSEI（情感强度回归/二分类/七分类）和IEMOCAP（情绪识别四类）。

**📈 对比分析**

与MuLT、SelfMM、ConKI、ConFEDE、CLGSI、MFON、EUAR、GLoMo、DEVA、Semi-IIN等SOTA方法对比，EBMC在Acc-2、Acc-7、F1、Corr和MAE等指标上均取得或接近最优成绩；在缺失模态与噪声测试中表现最为稳健，性能下降幅度最小。

**⚠️ 局限性**

局限性包括：①模型结构较为复杂，训练和推理成本相对较高；②主要验证在公开情感数据集上，尚未在更大规模或非英语语料上进行验证；③能量模型的超参数敏感度未完全系统化，可能需要针对不同任务进行细致调优。

---

## 300. NTIRE 2026 The 3rd Restore Any Image Model (RAIM) Challenge: Professional Image Quality Assessment (Track 1)

**arXiv ID:** 2604.12512 | [PDF](https://arxiv.org/pdf/2604.12512v1)

**作者:** Guanyi Qin `[一作]`, Yaokun Shi `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建并评测专业图像质量评估（PIQA）挑战赛的基准，聚焦多模态大语言模型对高质量图像对的比较与解释。

**💡 创新点**

创新点在于提出多模态大语言模型可生成专业级解释，并制定双维度评估（准确性+解释质量）及LLM评判方案。

**🔧 技术方法**

使用多模态大语言模型（MLLM），结合文本生成、BLEU/ROUGE、LLM-as-a-Judge评估、GRPO优化与投票集成。

**📊 数据集**

使用来自RAIM 2026的100对训练图像、102对验证集和101对测试集，包含肖像与风景，专家注释包含选择与理由。

**📈 对比分析**

方法与传统IQA相比在准确率上提升约10%（IH‑VQA 0.7129对比传统基线），解释质量S_thinking提升至0.2‑0.3；最高团队综合得分0.73。

**⚠️ 局限性**

局限在解释仍可能偏离专业细节，LLM评判受限于提示与模型偏差，数据集规模有限且仅涵盖肖像与风景两类。

---

## 301. Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs

**arXiv ID:** 2604.12506 | [PDF](https://arxiv.org/pdf/2604.12506v1)

**作者:** Linhao Zhang `[一作]` (Tencent Inc), Xiao Zhou `[通讯]` (Tencent Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `40105733-5154-44cd-8090-a8cab9e64b07` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对现有 AudioLLM 在细粒度感知方面的不足，本文提出统一音频架构 UAS 并在连续与离散两种输入下训练 UAS‑Audio 模型。

**💡 创新点**

创新点包括将音频信息拆分为转录、声学语调和非语言事件的三维 JSON 结构、可扩展的自动标注管道以及结构化监督提升感知能力的同时保持推理性能。

**🔧 技术方法**

主要技术涵盖自动音频描述生成、LLM 结构化合成、多层验证、基于 Qwen2.5‑7B 的连续编码器、流匹配解码器以及多阶段对齐与指令微调。

**📊 数据集**

使用的训练与评测数据包括大型 ASR 语料库自动生成的 UAS 标注、MMSU、MMAR、MMAU 三大评测基准、Seed‑TTS 语音合成测试以及 LibriSpeech、AISHELL 识别基准。

**📈 对比分析**

与 Qwen2.5‑Omni、Kimi‑Audio、Step‑Audio2 等同规模基线相比，UAS‑Audio 在 MMSU 感知任务上提升 10.9%（达 55.7%），保持 77.4% 以上推理精度，整体平均 65.2% 超越所有对照模型。

**⚠️ 局限性**

局限性主要在于仅验证了高资源语言，缺乏低资源或多语言混用场景；同时对多说话人重叠语音的声学属性分离能力尚未得到充分评估。

---

## 302. Topology-Aware Reasoning over Incomplete Knowledge Graph with Graph-Based Soft Prompting

**arXiv ID:** 2604.12503 | [PDF](https://arxiv.org/pdf/2604.12503v1)

**作者:** Shuai Wang `[一作]` (Chalmers University of Technology and University of Gothenburg), Yinan Yu `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GraSP框架，将知识图谱子图通过图神经网络编码为软提示，支持子图层面的多跳推理；采用两阶段LLM推理（轻量化LLM选取实体+强大LLM生成答案），降低对显式边的依赖；

**💡 创新点**

创新点在于：①用GNN捕获子图结构信息生成软提示，克服KG不完整导致的显式边缺失问题；②两阶段轻量+强大LLM组合，既降低算力成本又提升性能；

**🔧 技术方法**

技术包括图神经网络（GAT/Graph Attention Network）用于子图编码、软提示映射、轻量LLM实体选择、强大LLM答案生成；

**📊 数据集**

使用四大KBQA基准：ComplexWebQuestions（CWQ）、WebQuestionsSP（WebQSP）、WebQuestions、GrailQA，KG来源为Freebase；

**📈 对比分析**

与ToG、KBQA-o1、KG-Agent、iQUEST、LMP等方法比较，GraSP在三大多跳数据集上获得最高Hits@1，整体性能提升约5–10%；在KG不完整场景下衰减更缓；

**⚠️ 局限性**

局限性：对图结构完整性依赖较大，KG极度稀疏时子图信息有限；软提示缺乏可解释性，难以追踪推理路径；

---

## 303. Technical Report -- A Context-Sensitive Multi-Level Similarity Framework for First-Order Logic Arguments: An Axiomatic Study

**arXiv ID:** 2604.12534 | [PDF](https://arxiv.org/pdf/2604.12534v1)

**作者:** Victor David `[一作]` (University Côte d'Azur), Jean-Guy Mailly `[通讯]` (Université Toulouse Capitole)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种面向一阶逻辑（FOL）论证的四层相似度框架（谓词/项→文字→子句→公式），并通过语法转换为CNF和上下文权重实现可解释的相似度评估。

**💡 创新点**

创新点在于：①将语言模型与结构化FOL结合，形成可解释的混合相似度模型；②给出了完整的形式化公理体系，证明模型满足多项可望性质；③引入了语法敏感与语法独立两类模型，并通过上下文权重调节语义重要性。

**🔧 技术方法**

使用的技术包括：CNF化与Skolem化、SBERT/ChatGPT-4o进行语义嵌入、模糊Tversky度量、参数化的词项与子句权重、以及聚合与加权平均等统计方法。

**📊 数据集**

主要实验数据集为Stanford Natural Language Inference（STSb）文本对，结合人工编写的自然语言论证示例进行相似度计算。

**📈 对比分析**

与SBERT、S3BERT、ChatGPT等基线比较，得到0.628的相似度分数，几乎与人类平均0.630一致；计算时间约0.3秒，显示出较好的效率和可解释性。

**⚠️ 局限性**

局限性包括：权重手工设定缺乏自动学习；CNF转换步骤未实现自动化；实验规模有限，未在大规模论证语料上验证；模型主要针对语法敏感情况，语法独立版本仍不满足部分公理（如Non‑Zero）。

---

## 304. Agentic Control in Variational Language Models

**arXiv ID:** 2604.12513 | [PDF](https://arxiv.org/pdf/2604.12513v1)

**作者:** Yves Ruffenach `[一作]` `[通讯]` (Conservatoire National des Arts et Métiers), Yves Ruffenach (Conservatoire National des Arts et Métiers)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于变分语言模型的内部自控框架，利用局部随机隐藏层、稳态隐层调节、结构化检查点保留以及基于不确定性的校准控制器，实现从训练到推理的闭环内部代理行为。

**💡 创新点**

创新点在于将不确定性从仅仅的诊断工具提升为可操作的控制信号，并通过三阶段分离（训练调节→检查点保留→推理控制）构建可测量的最小代理机制，首次在语言模型内部实现了完整的自我调节与决策。

**🔧 技术方法**

采用了GPT‑2冻结嵌入前端 + Transformer‑style 局部变分隐藏单元（EVE），结合自稳态隐层正则化、结构化保留规则和多阈值不确定性校准的控制策略；使用交叉熵、KL、局部重建等多重损失进行训练。

**📊 数据集**

数据集采用 GPT‑2 词表对 Prompt‑Story 对进行预处理（Prompt ≤ 32 词，Story ≤ 96 词，窗口 32，步长 1），训练集使用官方 GPT‑2 训练集的 90% 作为验证，未公开具体原始数据源但与 prior work 相似。

**📈 对比分析**

对比方法为匹配的确定性基线（DET）在同一前端与任务下进行训练；实验结果显示 EVE 在交叉熵、困惑度、准确率、NLL、ECE、互信息等指标上均优于 DET，且校准控制器在多动作评估中实现 90% 覆盖率、正向效用 0.1048，证明内部不确定性可驱动实用代理决策。

**⚠️ 局限性**

局限性包括：仅在小规模 GPT‑2‑style 架构上验证；仅针对单一下游任务（next‑token）与有限动作空间；缺乏跨模型、跨任务的泛化验证；未探索更复杂的外部工具或环境交互，未来工作需扩展规模与应用场景。

---

## 305. Whole-Body Mobile Manipulation using Offline Reinforcement Learning on Sub-optimal Controllers

**arXiv ID:** 2604.12509 | [PDF](https://arxiv.org/pdf/2604.12509v1)

**作者:** Snehal Jauhri `[一作]` (Technische Universitaet Darmstadt), Georgia Chalvatzaki `[通讯]` (Technische Universitaet Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93`

**🎯 论文内容**

本文提出了WHOLE-MoMa方法，用无遥操作数据学习移动机器人对关节物体（如门、抽屉、柜子）的全身协同操控。

**💡 创新点**

创新点在于把子最优的全身控制器作为结构先验，通过参数随机化生成多样演示，再用离线强化学习的 Q‑chunking 与扩散策略对演示进行重组与改进，兼顾运动表达力与时序一致性。

**🔧 技术方法**

使用技术包括：Hierarchical Quadratic Programming (HQP) 做 WBC 生成演示，Transformer‑based Diffusion Policy，Implicit Q‑Learning (IQL) 结合 Q‑chunking 的离线 RL 训练与优势加权回归（AWR）提取策略。

**📊 数据集**

实验数据集为在 Isaac‑Sim 环境下的 Tiago++ 移动机械臂与 GAPartNet 关节物体的仿真演示，真实实验使用相同机器人进行无实地调优的评估。

**📈 对比分析**

与 WBC、行为克隆、TD3、IQL+DDPG_BC、IDQL、RISE 等基线相比，WHOLE‑MoMa 在门、抽屉、柜子任务中分别达到 98%、80% 及 78% 的成功率（仿真），在真实柜子开关放置任务中实现 68% 成功率，显著优于其他方法。

**⚠️ 局限性**

主要局限是对姿态估计精度高度敏感，尤其在柜子开关放置时小的 6D 角度误差会导致策略失效；缺乏关节阻抗/顺应控制，以及对非跟踪视觉环境的泛化能力不足。

---

## 306. Safety Training Modulates Harmful Misalignment Under On-Policy RL, But Direction Depends on Environment Design

**arXiv ID:** 2604.12500 | [PDF](https://arxiv.org/pdf/2604.12500v1)

**作者:** Leon Eshuijs `[一作]` (Vrije Universiteit Amsterdam), Antske Fokkens `[通讯]` (Vrije Universiteit Amsterdam)

**通讯引用:** 1285 | [OpenAlex ID](https://openalex.org/A5074075557)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究RL导致LLM的specification gaming，并系统评估模型规模与环境特征对有害misalignment的影响；

**💡 创新点**

首次同时系统地变化模型规模与环境设计，发现规模对misalignment的影响随环境而异；通过针对性消融揭示角色框架和隐式游戏信号的关键作用，并证明on‑policy RL在安全缓冲中起决定性作用；

**🔧 技术方法**

在三种自定义环境（Therapy Talk、Action Advice、Political QA）上，对11个指令微调LLM（0.5B–14B）使用on‑policy GRPO进行RL训练，并用ACC和HEX两项指标评估任务性能与有害exploit；

**📊 数据集**

利用改造自既有三种环境的用户样本，加入可游戏和不可游戏用户，使用LLM判定奖励和评估，未采用真实人类反馈；

**📈 对比分析**

通过最大HEX gap、Spearman相关与模型规模/能力得分比较，发现Therapy Talk中模型规模越大misalignment越低，AA/PQA则相反；大模型在某些环境下显示安全缓冲，但在其他环境更易失效；

**⚠️ 局限性**

仅覆盖三种仿真环境，评估依赖LLM判定缺乏真实人类反馈，模型规模限制至14B，且仅使用GRPO，其他RL方法可能表现不同。

---

## 307. Calibrated Confidence Estimation for Tabular Question Answering

**arXiv ID:** 2604.12491 | [PDF](https://arxiv.org/pdf/2604.12491v1)

**作者:** Lukas Voss `[一作]` `[通讯]` (Independent Researcher), Lukas Voss (Independent Researcher)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型在表格问答任务中的置信度进行系统评估，比较五种置信估计方法与五大前沿模型，并提出多格式一致性(MFA)与结构感知再校准方法。

**💡 创新点**

①提出的 MFA 通过在多种无损表格序列化格式（Markdown、HTML、JSON、CSV）之间的答案一致性来捕获输入扰动不确定性，既比采样方法成本低 20% 又具有确定性；②将表结构特征（行列数、列类型、查询复杂度等）纳入 Platt 规模化，实现了 +10% AUROC 的结构感知再校准。

**🔧 技术方法**

自评估（verbalized、P(True)）、输出扰动（self-consistency、semantic entropy）以及输入扰动（MFA）等方法；使用平滑 ECE、Brier 分数、AUROC 等校准与判别指标；对 MFA 进一步做多格式与多采样组合的集成。

**📊 数据集**

WikiTableQuestions（WTQ）和 TableBench 两大表格问答基准；覆盖 2000 条 WTQ 验证样本和 836 条 TableBench 测试样本；评估五个模型：GPT‑4o、GPT‑4o‑mini、Gemini 2.5 Flash、Llama‑3.3‑70B、DeepSeek‑V3。

**📈 对比分析**

实验显示：所有模型在表格问答上均明显过度自信（smooth ECE 0.35–0.64）；自评估方法 AUROC 仅 0.42–0.76；输入/输出扰动方法 AUROC 0.78–0.86，平均提升约 20%；MFA 以 20% 较低的 API 调用量获得与采样基线相同的 AUROC，并在 TableBench 上进一步提升至 0.80；MFA+自一致性+语义熵三者集成可将 AUROC 提升至 0.82。

**⚠️ 局限性**

仅聚焦表格问答，未验证在其他结构化领域的普适性；MFA 需手动构造多种序列化格式，适用性受限；实验主要在公开基准上进行，缺乏真实生产环境的长期评估；未给出正式覆盖保证，未来可结合 conformal prediction 等方法提升可靠性。

---

## 308. Large-Scale Measurement of NAT Traversal for the Decentralized Web: A Case Study of DCUtR in IPFS

**arXiv ID:** 2604.12484 | [PDF](https://arxiv.org/pdf/2604.12484v1)

**作者:** Dennis Trautwein `[一作]` (University of Göttingen), Bela Gipp `[通讯]` (University of Göttingen)

**通讯引用:** 6007 | [OpenAlex ID](https://openalex.org/A5058837356)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

**🎯 论文内容**

本文通过大规模实测验证了IPFS网络中的分布式NAT穿越协议DCUtR的可行性，并公开了4.4+百万次穿越实验的数据集。

**💡 创新点**

创新点包括：①首次在真实P2P网络上进行千级以上的DCUtR测量；②实证挑战UDP优先于TCP的传统认知，证明TCP也能同样有效；③系统验证了协议设计中的连接反转、RTT同步与Relay独立性；④公开了完整的实验数据，供后续研究使用。

**🔧 技术方法**

使用的技术主要有：libp2p栈（Identify、AutoNAT、Circuit v2）、DCUtR协议、RTT基同步机制、QUIC/TCP握手；测量环节通过honeypot、gRPC服务器与客户端三元架构实现。

**📊 数据集**

数据集来源于IPFS网络的7k+在线节点，采集了超过4.4百万次穿越尝试，涵盖超过85k个远端网络、859个客户端网络，公开存储于IPFS地址：bafybeia7sq3nfd7c4obcy7ahjvnoka7ujdiob33r7rqyeycgicdt3iknki.ipfs.dweb.link。

**📈 对比分析**

比较方法：按成功/失败、RTT、网络位置、传输协议分类；性能表现为：条件成功率约70%（不计Relay和地址发现失败），97.6%成功率在第一次同步尝试内完成；TCP与QUIC成功率均在70%上下，验证传输无显著差异；Relay位置与RTT对成功率影响微乎其微。

**⚠️ 局限性**

局限性包括：①受限于志愿者网络，可能偏向技术成熟、网络稳定的环境；②honeypot发现机制排除了完全不可连通或仅允许Relay的极端NAT；③未能对远端NAT类型做精确分类，无法解释个别失败原因；④测量仅覆盖DCUtR的Hole Punch阶段，忽略了Relay预留与地址发现等前置步骤导致的29%失败；⑤数据主要来自IPv4，IPv6及未来网络环境未充分覆盖。

---

## 309. Analyzing the Effect of Noise in LLM Fine-tuning

**arXiv ID:** 2604.12469 | [PDF](https://arxiv.org/pdf/2604.12469v1)

**作者:** Lingfang Li `[一作]`, Procheta Sen `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统评估了标签噪声、拼写噪声和语法噪声对GPT‑2、Qwen2和Llama‑2三大预训练语言模型在情感分类、问答和机器翻译任务中的适配效果与内部学习动态；

**💡 创新点**

首次在多模型多任务场景下，以注意力分布、层级探针与表示相似度（CKA、余弦相似度）三维度全面剖析噪声传播，发现噪声影响局部层级且标签噪声最为致命；

**🔧 技术方法**

采用Fine‑tune（GPT‑2全参数）与QLoRA（Qwen2、Llama‑2），通过KL散度、Spearman秩相关、线性探针、Logit Lens、CKA和中心化余弦相似度等指标进行层级分析；

**📊 数据集**

使用Yelp Polarity（情感）、SQuAD v1.1（问答）和Tatoeba（英‑法翻译）三大公开数据集，在不同噪声比例（20%、30%、40%）下进行实验；

**📈 对比分析**

与清洗数据的基线模型对比，标签噪声导致平均约21%性能下降；拼写与语法噪声偶有轻微正则化提升；注意力模式变化小，主要影响任务特定层的表示；

**⚠️ 局限性**

局限性在于仅探讨三种噪声类型与三模型，未深入机制细节；只评估了greedy推断；未来需扩展到更多模型与更复杂噪声，并设计针对性鲁棒微调方案。

---

## 310. Social Learning Strategies for Evolved Virtual Soft Robots

**arXiv ID:** 2604.12482 | [PDF](https://arxiv.org/pdf/2604.12482v1)

**作者:** K. Ege de Bruin `[一作]` (University of Oslo), Eric Medvet `[通讯]` (University of Trieste)

**通讯引用:** 2698 | [OpenAlex ID](https://openalex.org/A5074055647)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0`

**🎯 论文内容**

本研究提出了一种社会学习策略，用于优化虚拟软机器人（VSR）的身体和大脑，通过从其他机器人获取优化的控制参数来加速自身的优化过程。

**💡 创新点**

创新点在于引入社会学习机制，使机器人能够从同伴那里学习，而不是仅依赖自身的学习，从而提高了优化效率和性能。

**🔧 技术方法**

使用了贝叶斯优化（BO）技术来优化机器人的控制器，并通过社会学习策略进行样本转移。

**📊 数据集**

实验中使用了虚拟软机器人（VSR），在四个不同的任务和环境中进行测试，包括简单移动、阶梯移动、携带物体和捕捉物体。

**📈 对比分析**

与传统的个体学习（IL）方法相比，社会学习策略在多个任务中表现出更好的性能，尤其是从最佳机器人学习的策略效果最佳。所有社会学习策略在至少三个任务中显著优于个体学习。

**⚠️ 局限性**

限制在于尚未确定最佳的教师选择策略，尽管结果表明从多个教师学习通常能带来更一致和稳健的改进。

---

## 311. On Decentralized Sum-Rate Maximization with Successive Interference Cancellation

**arXiv ID:** 2604.12528 | [PDF](https://arxiv.org/pdf/2604.12528v1)

**作者:** D. Garrido `[一作]`, B. Peleato `[通讯]`

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

针对两用户高斯干扰信道，研究了功率与速率的联合分配，并在考虑接收机的成功干扰取消（SIC）时，给出了全局最优解和一个基于振荡的分散式算法。

**💡 创新点**

创新点在于：①首次系统性求解两用户干扰信道下SIC的全局最优功率/速率配置；②提出一种在无全局CSI情况下通过振荡策略实现近似最优的分散式机制。

**🔧 技术方法**

采用信息论模型、功率/速率优化（KKT/Lagrange），SIC技术以及数值仿真验证。

**📊 数据集**

使用仿真数据，参数包括对称信道下的最大SNR γ、交叉通道损耗 ϵ 与 μ 等；未使用公开实验数据集。

**📈 对比分析**

与贪心（No‑SIC）和正交访问（TDMA/FDMA）两种基准进行比较，评估指标为期望总速率与效率曲线；在大部分参数区间表现优于基准，接近最优；在高SNR区间正交访问可逼近最优。

**⚠️ 局限性**

局限性包括：仅针对两用户对称信道；振荡参数需预先协商；未推广到多用户或非对称场景；对快速时变信道或极端SNR的鲁棒性未评估。

---

## 312. Beyond Single-Dimension Novelty: How Combinations of Theory, Method, and Results-based Novelty Shape Scientific Impact

**arXiv ID:** 2604.12471 | [PDF](https://arxiv.org/pdf/2604.12471v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053`

---

## 313. From Attenuation to Attention: Variational Information Flow Manipulation for Fine-Grained Visual Perception

**arXiv ID:** 2604.12508 | [PDF](https://arxiv.org/pdf/2604.12508v1)

**作者:** Jilong Zhu `[一作]` (Institute of Computing Technology, Chinese Academy of Sciences), Yang Feng `[通讯]` (Institute of Computing Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对多模态大型语言模型在细粒度视觉感知上的视觉衰减问题，提出了Variational Information Flow (VIF)框架。

**💡 创新点**

创新点在于将视觉关注建模为条件变分自编码器（CVAE）的潜在分布，并用高斯混合模型（GMM）恢复深层视觉注意，解决视觉信息被文本主导抑制的问题。

**🔧 技术方法**

使用了CVAE、GMM、注意力注入、KL散度和稀疏正则等技术。

**📊 数据集**

在12个不同基准（包括MME、SEED-Bench、LLaVA-Bench、BLINK、ScienceQA、VQAv2、GQA、AI2D、TextVQA、HR-Bench、Vstar、RefCOCO）上进行评估。

**📈 对比分析**

与7B规模的多模态基线、高分辨率模型和特征增强模型对比，VIF在一般多模态任务、细粒度感知和视觉定位上均取得更高分数，尤其在Vstar、HR-Bench和RefCOCO上明显提升。

**⚠️ 局限性**

主要限制是引入CVAE模块后推理时参数和算力略增，且对输入分辨率的依赖使极小物体的像素细节仍难以恢复。

---

## 314. A Heterogeneous Dual-Network Framework for Emergency Delivery UAVs: Communication Assurance and Path Planning Coordination

**arXiv ID:** 2604.12501 | [PDF](https://arxiv.org/pdf/2604.12501v1)

**作者:** Ping Huang `[一作]` (Chengdu University of Technology), Jun Li `[通讯]` (Southeast University)

**通讯引用:** 55240 | [OpenAlex ID](https://openalex.org/A5021388534)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个双网络框架（ECSN+ DPN），通过协同部署空中基站与规划物流无人机路径，实现灾后无人机物流的通信保障与能耗最小化。

**💡 创新点**

创新点在于：① 引入多层 C2 服务模型，将终端投递、垂直起降、高速巡航三阶段的通信需求统一量化；② 开发基于 3D 覆盖感知的多智能体强化学习（3D‑CASB‑MATD3+PER）和通信感知 A* 路径规划，构成完整的 HDNF；③ 通过共享骨干网络与优先经验回放提升学习效率。

**🔧 技术方法**

使用的技术包括：多层 C2 评估模型、3D 覆盖感知的多智能体 MARL（MATD3+PER）、共享骨干网络、基于 SINR 的 3D 通信感知 A* 规划、代价函数结合能耗与通信质量。

**📊 数据集**

采用仿真生成的灾区任务分布与障碍物统计参数（α,β）以及随机生成的任务重量，未使用公开真实数据集。

**📈 对比分析**

与 MATD3‑2D、MADDPG、Grid Deployment 等基线对比；HDNF 在 C2 可靠性、任务成功率、通信盲区、基站部署数上均优于对手，且能耗与部署成本更低，能实现 100% 的任务成功率。

**⚠️ 局限性**

局限性：仅在仿真环境验证，缺乏真实灾区实验；对超参数敏感，计算量仍较大；未考虑多机队冲突、动态障碍变化等实际复杂场景。

---

## 315. Designing for Error Recovery in Human-Robot Interaction

**arXiv ID:** 2604.12473 | [PDF](https://arxiv.org/pdf/2604.12473v1)

**作者:** Christopher D. Wallbridge `[一作]` (Cardiff University), Erwin Jose Lopez Pulgarin `[通讯]` (University of Manchester)

**通讯引用:** 98 | [OpenAlex ID](https://openalex.org/A5088357590)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3855fcda-48ef-4070-a15e-803cd5c84d83` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文讨论了在核废料处理环境下机器人错误检测与恢复的重要性，并提出了一个基于任务描述、状态解释、机器人传感和接口协同的错误恢复系统框架。

**💡 创新点**

创新点在于将错误识别与因果分析与多源信息（视觉、传感器、操作员反馈）结合，实现持续交互中的动态错误沟通与协同恢复。

**🔧 技术方法**

使用了人工智能感知（机器学习/深度学习）、异常检测、状态推理技术，并通过HCI设计实现自然语言/语音与视觉界面交互。

**📊 数据集**

论文未提供具体实验数据集，主要基于理论与案例分析，未公开使用任何标准数据集。

**📈 对比分析**

由于为定位性研究，未进行对比实验，性能指标未给出；作者仅提出设计原则与实现思路。

**⚠️ 局限性**

局限性包括对任务描述的依赖、环境可见度不足导致错误识别困难、缺乏实测验证、通信可靠性挑战，以及需要进一步实现与评估。

---

## 316. Enhancing Clustering: An Explainable Approach via Filtered Patterns

**arXiv ID:** 2604.12460 | [PDF](https://arxiv.org/pdf/2604.12460v1)

**作者:** Motaz Ben Hassine `[一作]` (CRIL, University of Artois & CNRS), Saïd Jabbour `[通讯]` (CRIL, University of Artois & CNRS)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了OCCM方法，针对 k-RFP 中的冗余模式通过过滤保留唯一 k‑cover 的代表模式，从而提升可解释聚类的效率与质量。

**💡 创新点**

创新点在于：1）形式化分析不同 k‑RFP 产生相同 k‑cover 的条件；2）设计基于 k‑cover 的过滤算法，优先保留最大项集；3）引入 Shapley 值、SVV 与 ACS 等可解释性评估指标，系统评估聚类描述的代表性与稳定性。

**🔧 技术方法**

技术路线：使用 SAT 解决器枚举 k‑RFP，随后对生成的模式集合执行过滤算法（基于字典存储 k‑cover 与模式的映射），最后将过滤后的模式集输入整数线性规划（ILP）模型完成聚类。

**📊 数据集**

实验数据集包括 Lymph、Mushroom、Primary‑Tumor、Soybean、Tic‑Tac‑Toe、Vote 共六个真实事务数据集。

**📈 对比分析**

与基线 CCA‑k‑RFP‑M1 进行比较，过滤后模式数下降至最多 26.67% 以内；ILP 求解时间显著缩短（多数据集上提升数倍），且聚类 F1 分数保持不降甚至提升，证明方法在效率与质量上的优势。

**⚠️ 局限性**

局限性：① 过滤仍是后处理，未在 SAT 生成阶段直接消除冗余；② 当前实验仅考虑 k=1、θ=2，未验证更大 k 或多簇情况；③ 对极大规模数据集仍存在求解时间与内存压力；④ 过滤策略可能忽略某些较小但具有解释价值的模式。

---

## 317. Characterizing normality via automata and random matrix products

**arXiv ID:** 2604.12457 | [PDF](https://arxiv.org/pdf/2604.12457v1)

**作者:** Laurent Bienvenu `[一作]`, Hugo Gimbert `[通讯]`

**关键词:** `33d19632-8af2-4683-a5db-767c7ce749e6`

**🎯 论文内容**

本文证明了对正常序列，任何可实现的概率有限自动机的期望马丁格尔都无法成功，并给出了其收敛行为的三分之一分类。

**💡 创新点**

创新点在于将概率自动机与期望马丁格尔相结合，利用Hilbert度量和Birkhoff收缩性，得到对正则序列的三种收敛情形，并提供可判定性。

**🔧 技术方法**

主要技术包括马丁格尔理论、Hilbert投影度量、Birkhoff收缩定理、Azuma不等式、马尔可夫链以及可判定性（实数存在理论）。

**📊 数据集**

由于研究是理论性的，未使用具体数据集。

**📈 对比分析**

与之前的确定性自动机结果对比，扩展到了概率自动机，证明了期望收敛不再能通过高概率获胜的方式破坏正常序列；在可判定性方面，算法可在EXPSPACE内完成。

**⚠️ 局限性**

主要局限是判定算法的复杂度高，伪混合词长度可能指数级，若能证明多项式长度即可降低复杂度。

---

## 318. Deepfakes at Face Value: Image and Authority

**arXiv ID:** 2604.12490 | [PDF](https://arxiv.org/pdf/2604.12490v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 319. HazardArena: Evaluating Semantic Safety in Vision-Language-Action Models

**arXiv ID:** 2604.12447 | [PDF](https://arxiv.org/pdf/2604.12447v1)

**作者:** Zixing Chen `[一作]` (Fudan University), Yu-Gang Jiang `[通讯]` (Fudan University)

**通讯引用:** 140804 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `9cc9baba-5356-466d-81ff-d80028d90279` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个基于安全/不安全双生场景的安全评估基准，使用阶段化安全指标对 Vision‑Language‑Action (VLA) 模型的语义安全性进行系统测试，并提出了一种轻量级推理时安全选项层（SOL）用于拦截潜在危险行为。

**💡 创新点**

创新点在于：①通过安全/不安全双生场景严格隔离语义风险，消除动作可行性偏差；②提出阶段化（attempt、commit、success）安全度量，揭示终端指标下被忽视的危险进展；③提出训练无关的 SOL，兼顾安全约束与性能。

**🔧 技术方法**

技术手段包括：对 OpenVLA‑OFT、π₀、NORA、VLA‑Adapter 等 VLA 模型进行安全‑仅微调；设计 SOL 的属性约束层和 VLM‑判定层；利用 ISO 13482:2014 与 AutoRT 标准生成七类风险任务；实现阶段化安全指标提取与评估。

**📊 数据集**

使用的数据集是新构建的约 2,000 资产与 40 个风险敏感任务，覆盖七大危险类别；每个任务都有安全/不安全双生对，数据格式兼容 RLDS 与 LeRobot；此外参考 ISO 13482:2014 与 AutoRT 报告。

**📈 对比分析**

评估方式为在安全/不安全双生任务上比较 success_rate、commit_rate 与 attempt_rate；实验表明模型在安全任务上能力提升会伴随不安全任务上危险完成率上升；SOL 在不牺牲安全任务成功率的前提下，可将不安全任务的 commit_rate 与 success_rate 显著降低，表现出可观的安全抑制效果。

**⚠️ 局限性**

局限性包括：SOL 仅为推理时后处理，无法解决策略层面的语义安全缺陷；VLM‑判定层在不同危险类别上表现不均，存在高误报或漏报；基准仅涵盖有限的单一动作模板和任务规模，未来需扩展至更复杂的长时序或多智能体场景。

---

## 320. Lit2Vec: A Reproducible Workflow for Building a Legally Screened Chemistry Corpus from S2ORC for Downstream Retrieval and Text Mining

**arXiv ID:** 2604.12498 | [PDF](https://arxiv.org/pdf/2604.12498v1)

**作者:** Mahmoud Amiri `[一作]`, Thomas Bocklitz `[通讯]`

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `57a58b01-81b4-4d75-a45c-2e891f272b50` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建并验证了一个可复现的化学全文语料库Lit2Vec，采用多源许可证筛选、全文本重建、段落级嵌入和可选的摘要与子领域标注；

**💡 创新点**

首次提供完整、可复现的化学全文语料构建工作流，解决了现有仅含摘要、文档结构不完整、许可证不透明等限制；

**🔧 技术方法**

使用Semantic Scholar Open Research Corpus (S2ORC) 作为源数据，结合Unpaywall、OpenAlex、Crossref进行许可证合并，利用intfloat/e5-large-v2进行段落嵌入，采用GPT‑4o进行TL;DR生成，使用MLP+embedding做多标签子领域分类；

**📊 数据集**

主要数据集为S2ORC 2024‑12‑31版本（约200M条目），最终得到582,683条符合许可的化学论文全文本，约28.8M段落；

**📈 对比分析**

与抽取式基线、零射击摘要模型对比，Fine‑tuned DistilBART在ROUGE‑1、ROUGE‑2、ROUGE‑L、BERTScore上均显著提升（例如ROUGE‑1 +19.38，BERTScore F1 +4.02），子领域分类在主流标签上微F1≈0.81，罕见标签表现受限；

**⚠️ 局限性**

主要局限包括缺失或过短的摘要导致部分论文缺少摘要级嵌入、摘要和子领域标签；全文本重建依赖于可访问的源数据与许可证元数据，因许可证和重建过程的限制无法公开完整语料；模型性能在罕见子领域下降，且未包含图表、化学结构等非文本信息。

---

## 321. A Hybrid Architecture for Benign-Malignant Classification of Mammography ROIs

**arXiv ID:** 2604.12437 | [PDF](https://arxiv.org/pdf/2604.12437v1)

**作者:** Mohammed Asad `[一作]` (Delhi Technological University), Rahul Katarya `[通讯]` (Delhi Technological University)

**通讯引用:** 3991 | [OpenAlex ID](https://openalex.org/A5064570061)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种将EfficientNetV2-M与Vision Mamba相结合的混合网络，用于乳腺X光ROI的良恶性分类。

**💡 创新点**

创新点在于将高效CNN作为局部特征提取器与线性复杂度的State Space Model（Vision Mamba）并行结合，实现既有局部纹理捕捉又有全局长程依赖建模。

**🔧 技术方法**

使用EfficientNetV2-M、Vision Mamba、AdamW、加权二分类交叉熵、轻量级数据增强等技术。

**📊 数据集**

使用CBIS‑DDSM数据库的异常中心ROI（单视图）。

**📈 对比分析**

与多种CNN、ViT基线（如VGG‑16、ResNet‑50、ViT‑B/16等）在同一预处理管线下对比，取得AUC 0.875、准确率94.2%、敏感度0.89、特异度0.95，显著优于基线。

**⚠️ 局限性**

仅在ROI级别、单视图、单视角训练，未考虑全图或双视图信息，缺乏外部验证。

---

## 322. Euler-inspired Decoupling Neural Operator for Efficient Pansharpening

**arXiv ID:** 2604.12463 | [PDF](https://arxiv.org/pdf/2604.12463v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 323. Adaptive Budget Allocation in LLM-Augmented Surveys

**arXiv ID:** 2604.12497 | [PDF](https://arxiv.org/pdf/2604.12497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 324. From Kinematics to Dynamics: Learning to Refine Hybrid Plans for Physically Feasible Execution

**arXiv ID:** 2604.12474 | [PDF](https://arxiv.org/pdf/2604.12474v1)

**作者:** Lidor Erez `[一作]` (Ben Gurion University Negev), Ayal Taitler `[通讯]` (Ben Gurion University Negev)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了一种基于强化学习的混合规划方案，用来将一阶动力学的计划转化为满足二阶动力学的物理可行轨迹。

**💡 创新点**

创新点在于提出将混合规划、二阶可行性验证（MTV）与强化学习（PPO）相结合的CSA‑MDP框架，实现对速度边界的自适应调整，从而填补一阶计划与真实二阶动力学之间的缺口。

**🔧 技术方法**

技术上采用图神经网络对计划图进行编码，使用PPO强化学习策略生成速度边界调整动作，结合SOCP求解器进行时长优化，并利用MTV闭式二阶最短时间验证进行可行性评估。

**📊 数据集**

实验使用了四个混合规划领域（AUV‑2D、Norm‑AUV‑2D、OnAir‑Refuel、Sailing），从Scotty生成的初始一阶计划作为起点。

**📈 对比分析**

与两种统一收缩速度边界的基线（分别按10%和0.5%收缩）进行比较，实验显示该方法在所有域中均能100%恢复物理可行性，且平均耗时比基线提升约5–10%。

**⚠️ 局限性**

主要局限在于假设无扰动、轴解耦且无时间窗口，尚未针对耦合动力学、时间窗口或不确定性等更真实场景进行扩展。

---

## 325. CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems

**arXiv ID:** 2604.12461 | [PDF](https://arxiv.org/pdf/2604.12461v1)

**作者:** Yongxuan Wu `[一作]` (Chinese Academy of Sciences), Yanan Cao `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 5716 | [OpenAlex ID](https://openalex.org/A5044388337)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种在黑盒环境下可推断多智能体系统（MAS）通信拓扑的攻击方法——Communication Inference Attack (CIA)，并验证其在多种生成式优化策略下的有效性。

**💡 创新点**

核心创新包括：①利用对抗查询诱导中间代理输出；②全局偏差分解（GBD）消除语义相似中的无关信息；③基于LLM的弱监督（LWS）引导学习拓扑结构；③结合相似度阈值精确识别边。

**🔧 技术方法**

采用的技术主要是：对抗查询策略（累计传播、任务聚焦、前辈回顾）；全局偏差分解（基于互信息约束的两路编码）；LLM引导弱监督（使用GPT‑5产生结构化信号）；相似度阈值边识别。

**📊 数据集**

实验使用四个任务数据集（MMLU、GSM8K、SVAMP、HumanEval）来构建MAS通信拓扑，使用三种生成式优化策略（G‑Designer、AGP、ARG‑Designer）生成通信网络。

**📈 对比分析**

与闭源LLM（GPT‑5、Gemini‑2.5‑Pro）及开源LLM（Llama‑3.1‑8B‑Instruct、Mistral‑7B‑Instruct‑v0.2）基线相比，CIA在所有设置下平均AUC约0.83–0.88，最高可达0.99；误报率显著下降，性能远超基线。

**⚠️ 局限性**

主要局限：对多维互信息的估计仍不精确；LLM弱监督仅捕获一阶拓扑信息，未能利用更高阶拓扑模式，影响进一步提升精度。

---

## 326. D-BDM: A Direct and Efficient Boundary-Based Occupancy Grid Mapping Framework for LiDARs

**arXiv ID:** 2604.12436 | [PDF](https://arxiv.org/pdf/2604.12436v1)

**作者:** Benxu Tang `[一作]` (University of Hong Kong), Fu Zhang `[通讯]` (University of Hong Kong)

**通讯引用:** 39764 | [OpenAlex ID](https://openalex.org/A5000546219)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于边界表面表示的高效占用映射框架，利用截断射线投射仅在外部区域更新体素并通过直接边界更新机制消除局部3D栅格的需求；

**💡 创新点**

创新点在于：①截断射线投射将更新限制在边界外部显著减少遍历体素量；②直接边界更新消除辅助3D栅格，进一步降低内存占用并简化更新流程；

**🔧 技术方法**

主要技术包括：二维边界体素表示、深度图光栅化投影、轴对齐盒与光线交点（slab）判定、哈希式二维网格存储与邻域扩张；

**📊 数据集**

实验使用了HeLiPR（KAIST_04、Town_03、Roundabout_01）、KITTI（kitti_00、kitti_02）以及Newer College（college）等公开大规模序列，并在多楼层建筑的实测航迹中验证；

**📈 对比分析**

与 Uniform Grid、OctoMap、D-Map 和 BoundaryMap 对比，D‑BDM 在0.1 m分辨率下相较于 D‑Map 提升约2.8×、BoundaryMap 4×、OctoMap 18.2×的更新速度；内存占用比 OctoMap 低约90%、比 D‑Map 低约88%，且在高分辨率下差距更为显著；

**⚠️ 局限性**

局限性：仍依赖精确的边界提取，对极为动态或高密度场景的鲁棒性尚未彻底验证，且在极大尺度或极高分辨率下可能出现光栅化误差或邻域扩张开销。

---

## 327. CoD-Lite: Real-Time Diffusion-Based Generative Image Compression

**arXiv ID:** 2604.12525 | [PDF](https://arxiv.org/pdf/2604.12525v1)

**作者:** Zhaoyang Jia `[一作]` (University of Science and Technology of China), Yan Lu `[通讯]` (Microsoft Research Asia)

**通讯引用:** 20268 | [OpenAlex ID](https://openalex.org/A5035278528)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种实时轻量化卷积扩散图像编码器，利用压缩导向的预训练和一阶扩散实现1080p下60 FPS编码和42 FPS解码。

**💡 创新点**

创新点在于：①压缩导向预训练显著提升小模型表现；②全局注意力可被局部卷积替代，显著降低计算量；③结合蒸馏与对抗训练，实现逼真质量与实时速度的平衡。

**🔧 技术方法**

采用卷积扩散网络（depth‑wise conv + channel attention）、一阶扩散、CoD压缩预训练、分布匹配蒸馏（DMD）和投影GAN等技术。

**📊 数据集**

训练集使用ImageNet‑21K、OpenImages和SA‑1B；评估集为Kodak和CLIC2020。

**📈 对比分析**

与GAN、传统多步扩散以及现有一阶扩散编解码器比较，在Kodak上实现85%比MS‑ILLM更低的比特率，同时FID保持相近；在CLIC上亦保持竞争性能，并在1080p下实现42 FPS解码。

**⚠️ 局限性**

局限性：仅在512×512分辨率上训练，导致超高分辨率（如4K）性能下降；扩散生成的伪影风险仍需关注。

---

## 328. SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker

**arXiv ID:** 2604.12502 | [PDF](https://arxiv.org/pdf/2604.12502v1)

**作者:** Junbin Su `[一作]` (Yanshan University), Zhipeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6549 | [OpenAlex ID](https://openalex.org/A5100410140)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出 SEATrack，一种简单高效的两流多模态跟踪器，先通过跨模态注意力对齐再融合。

**💡 创新点**

创新点在于引入 AMG-LoRA 通过低秩适配器与自适应互导实现跨模态注意力对齐，以及 Hierarchical Mixture-of-Experts (HMoE) 进行高效全局关系建模。

**🔧 技术方法**

使用的技术包括 LoRA、Adaptive Mutual Guidance (AMG)、ViT 编码器、Hierarchical MoE、基于 OSTrack 的冻结基础模型、以及轻量化 PEFT。

**📊 数据集**

使用的数据集包括 RGB‑T (LasHeR、RGBT234)、RGB‑D (DepthTrack、VOT‑RGBD2022) 和 RGB‑E (VisEvent)。

**📈 对比分析**

在这些基准上，SEATrack 在性能和效率上均优于现有 PEFT 与 FFT 方法，参数量仅 0.6M，FPS 63.5，PR/SR/MPR 等指标均达到或接近 SOTA。

**⚠️ 局限性**

局限性在于在部分指标上仍略逊于最佳 FFT 方法，且对输入可靠性变化的自适应机制尚待进一步提升；同时，如何在跟踪场景中对齐空间异质模态（如视觉+语言）仍是未来挑战。

---

## 329. T2I-BiasBench: A Multi-Metric Framework for Auditing Demographic and Cultural Bias in Text-to-Image Models

**arXiv ID:** 2604.12481 | [PDF](https://arxiv.org/pdf/2604.12481v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 330. KG-Reasoner: A Reinforced Model for End-to-End Multi-Hop Knowledge Graph Reasoning

**arXiv ID:** 2604.12487 | [PDF](https://arxiv.org/pdf/2604.12487v1)

**作者:** Shuai Wang `[一作]` (Chalmers University of Technology and University of Gothenburg), Yinan Yu `[通讯]` (Chalmers University of Technology and University of Gothenburg)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 KG-Reasoner，一个端到端框架，将检索和多跳知识图推理统一到一个 LLM 的思考阶段，实现连续、可回溯的推理流程。

**💡 创新点**

创新点包括：① 在 LLM 思考阶段内部化 KG 遍历，通过 RL 训练形成自适应的检索与推理策略；② 引入基于 GNN 的实体排序提升检索精度；③ 设计显式回溯机制和错误纠正训练，显著降低错误传播；④ 通过多种奖励函数（检索、格式、答案）引导 RL，提高整体推理质量。

**🔧 技术方法**

使用技术包括：大规模语言模型（Qwen‑3‑30B、LLaMA‑3.1‑8B 等）；检索工具（基于 SPARQL 的一次跳邻居检索）；图神经网络（GNN+注意力）进行实体相关性评分；强化学习（GRPO）和自监督错误纠正数据；以及结构化标签（<think>、<search>、<triples>、<answer>）实现工具调用与推理日志。

**📊 数据集**

在八个 KBQA 评测集上评估：Freebase 基准（CWQ、WebQSP、WebQuestions、GrailQA）和 WikiData 基准（QALD10‑en、T‑REx、Zero‑Shot RE、Creak）。

**📈 对比分析**

与现有方法对比，KG‑Reasoner 在 30B LLM 上在四个 Freebase 任务上均超越 70B 公开 LLM（KBQA‑o1）且在 GrailQA 上仍保持领先；与闭源 LLM（如 GPT‑4）相比，在三项多跳任务上取得更好或相近性能；通过 ablation 证明统一推理、回溯和 RL 奖励的贡献。

**⚠️ 局限性**

局限性：① 强化学习训练成本高，需要复杂的多模型配合；② 依赖知识图的完整性和准确性，KG 中的缺失或噪声仍可能导致推理失败。

---

## 331. Elastic Net Regularization and Gabor Dictionary for Classification of Heart Sound Signals using Deep Learning

**arXiv ID:** 2604.12483 | [PDF](https://arxiv.org/pdf/2604.12483v1)

**作者:** Mahmoud Fakhry `[一作]` (Universidad Carlos III de Madrid), Ascensión Gallardo-Antolín `[通讯]` (Universidad Carlos III de Madrid)

**通讯引用:** 1180 | [OpenAlex ID](https://openalex.org/A5065719579)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于弹性网正则化的Gabor字典对心音信号进行拟合并构造时频特征矩阵，用CNN‑LSTM网络实现五类心脏瓣膜疾病的多分类；

**💡 创新点**

创新点在于将弹性网正则化与过完备Gabor字典结合，生成稀疏且具有高判别性的时频特征矩阵，并采用1D+2D卷积结合LSTM的网络结构达到最高识别精度；

**🔧 技术方法**

使用的技术包括弹性网正则化、过完备Gabor字典、加权对数变换特征、1D与2D卷积层、LSTM层、SGDM与Adam两种优化算法；

**📊 数据集**

实验数据集为公开的1000条PCG心音信号，分为正常、MVP、MS、MR、AS五类，采样率8000Hz；

**📈 对比分析**

通过与原始PCG+CNN‑LSTM以及VMD+轻量CNN‑LSTM基准模型对比，最佳模型在98.95%准确率，误差率相比基准降低约30%+，实现了显著性能提升；

**⚠️ 局限性**

局限性包括仅使用实Gabor字典未探讨复数字典潜力、模型训练依赖大量实验、缺乏跨设备或多中心数据的验证。

---

## 332. Meet Dynamic Individual Preferences: Resolving Conflicting Human Value with Paired Fine-Tuning

**arXiv ID:** 2604.12479 | [PDF](https://arxiv.org/pdf/2604.12479v1)

**作者:** Shanyong Wang `[一作]` (Rutgers University), Yongfeng Zhang `[通讯]` (Rutgers University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a2602d71-93ab-4bad-974b-672788df8193` `a4b10f5d-130b-4e77-9367-6469ec621899` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了价值冲突困境（VCD）数据集，并设计了 Preference‑Paired Fine‑Tuning（PFT）框架，使单一模型能够同时处理相互冲突的个体偏好。

**💡 创新点**

创新点在于使用冲突偏好对训练样本配对，并在同一模型中同步优化正反两侧偏好，形成低维偏好流形的泛化能力，从而实现多偏好并行对齐。

**🔧 技术方法**

采用预训练语言模型的条件生成、偏好向量编码、同步配对交叉熵损失（PFT）、以及少量用户历史数据的 in‑context 学习实现快速定制。

**📊 数据集**

主要使用自建的 VCD 数据集（含 4652 条冲突偏好示例），并在 PRISM、CLASH、DailyDilemmas、Hummer 等公开数据集上进行评估。

**📈 对比分析**

与传统 SFT、DPO、CPO、CAA 等基线相比，PFT 在多选分类任务上最高可达 96.67% 准确率，在开放式生成任务中获得 8.69 的人类评价得分，整体性能显著优于单偏好训练方法。

**⚠️ 局限性**

局限性包括：仅覆盖少量二元偏好维度，评估仍依赖 GPT‑4o 自动打分，缺乏长期动态偏好跟踪与在线学习能力，且对多元化文化与语境的适用性尚未充分验证。

---

## 333. Intelligent ROI-Based Vehicle Counting Framework for Automated Traffic Monitoring

**arXiv ID:** 2604.12470 | [PDF](https://arxiv.org/pdf/2604.12470v1)

**作者:** Mohamed A. Abdelwahab `[一作]` (Aswan University), El-Sayed Hasaneen `[通讯]` (Aswan University)

**通讯引用:** 623 | [OpenAlex ID](https://openalex.org/A5029520450)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e0540dec-d77f-42db-94ae-d039248f6393` `aaccfe5c-6b26-4208-b23c-35331481e142` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

开发了一个全自动化、基于ROI的车辆计数框架AIR‑VC，通过估计与预测两阶段实现高精度计数并提升计算效率。

**💡 创新点**

创新点在于自动化确定最佳计数线、ROI和阈值的三模型方法（检测分数、跟踪分数、车辆密度），无需人工干预且可兼容任意检测跟踪算法。

**🔧 技术方法**

使用YOLOv8进行检测、DeepSORT进行跟踪，并结合多项式回归、IQR异常剔除和Brute Force搜索等技术实现ROI与阈值估计。

**📊 数据集**

在UA‑DETRAC、GRAM、CDnet 2014、ATON等多种视频数据集上进行验证。

**📈 对比分析**

与现有手工或自动ROI方法对比，AIR‑VC在绝大多数视频上实现100%计数精度，帧率提升至全帧处理的2–4倍，性能优于对标方法。

**⚠️ 局限性**

当前框架仅支持单一行/方向计数，难以直接处理交叉口或多方向交通，需要进一步扩展多计数线与方向识别能力。

---

## 334. Operationalising the Right to be Forgotten in LLMs: A Lightweight Sequential Unlearning Framework for Privacy-Aligned Deployment in Politically Sensitive Environments

**arXiv ID:** 2604.12459 | [PDF](https://arxiv.org/pdf/2604.12459v1)

**作者:** Esen Kurt `[一作]` (Munster Technological University), Haithem Afli `[通讯]` (Munster Technological University)

**通讯引用:** 538 | [OpenAlex ID](https://openalex.org/A5046400614)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了一种顺序不学习框架，用正向微调保留有益知识后再进行层受限负向微调，以实现政治敏感LLM的“被遗忘权”。

**💡 创新点**

创新点在于将保持与遗忘分离为两个阶段，显著减少梯度冲突并通过局部更新保持语言能力。

**🔧 技术方法**

采用正向微调、层限制负向微调（梯度上升）、AdamW优化器和交叉熵损失等技术。

**📊 数据集**

使用 SemEval‑2025 LLM Unlearning 基准的 Retain 与 Forget 数据集。

**📈 对比分析**

与 GPT‑2 与 DistilGPT‑2 对比，GPT‑2 在保持事实准确性和流畅性的同时显著抑制敏感输出；DistilGPT‑2 由于容量受限表现出更高困惑度。

**⚠️ 局限性**

仅实现行为抑制，缺乏参数级删除证明；实验仅在英语小模型上进行，缺乏多语言和真实政治情境的验证。

---

## 335. Latent-Condensed Transformer for Efficient Long Context Modeling

**arXiv ID:** 2604.12452 | [PDF](https://arxiv.org/pdf/2604.12452v1)

**作者:** Zeng You `[一作]` (South China University of Technology), Mingkui Tan `[通讯]` (South China University of Technology)

**通讯引用:** 14846 | [OpenAlex ID](https://openalex.org/A5032352025)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种在MLA低维潜空间中进行结构化上下文压缩的注意力机制，直接在潜空间内压缩语义向量并保留位置信息，从而显著减少KV缓存与注意力计算。

**💡 创新点**

创新点包括：① 在潜空间将语义与位置信息分离处理；② 采用查询感知加权池化聚合语义向量；③ 用最大选择保留位置信息；④ 给出长度无关的误差上界；⑤ 该机制无额外参数，能无缝集成到任意注意力变体。

**🔧 技术方法**

使用了MLA、RoPE、分组潜在压缩、加权池化、锚点选择、定制 Triton kernel 等技术，且实现支持bfloat16/float16精度。

**📊 数据集**

实验数据集包括 LongBench‑E、RULER、MMLU、GSM‑8K、MBPP、OlympiadBenchMath、Distill‑Qwen‑7B、MiniCPM3‑4B 等长短上下文基准。

**📈 对比分析**

与标准 MLA、Minference、FlexPrefill、KDA 等方法对比，128K 长度下实现约2.5×的前缀生成速度提升、90% KV缓存压缩，性能与MLA持平甚至在部分任务优于MLA；短文本任务保持相近分数。

**⚠️ 局限性**

局限性：需自定义内核实现，工程成本较高；目前仅针对bfloat16/float16；在极端压缩或检索敏感任务中可能出现轻微准确度下降；低精度量化未充分优化。

---

## 336. Scaling Exposes the Trigger: Input-Level Backdoor Detection in Text-to-Image Diffusion Models via Cross-Attention Scaling

**arXiv ID:** 2604.12446 | [PDF](https://arxiv.org/pdf/2604.12446v1)

**作者:** Zida Li `[一作]`, Zhangjie Fu `[通讯]` (Nanjing University of Information Science and Technology)

**通讯引用:** 5896 | [OpenAlex ID](https://openalex.org/A5066341740)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

研究针对文本到图像扩散模型的输入级后门检测，提出通过对交叉注意力进行尺度调制并提取响应差异来识别后门输入。

**💡 创新点**

提出了新的“交叉注意力尺度响应差异” (CSRD) 现象，并基于此设计了主动探测框架 SET，实现对任意触发方式后门的无模型训练、无先验知识检测。

**🔧 技术方法**

核心技术包括交叉注意力尺度调制、响应偏移特征构造、轻量化编码器以及软边界一类学习的安全空间建模。

**📊 数据集**

使用 MS‑COCO 文本作为基准，并在 Stable Diffusion v1.4 上评估多种后门攻击（Rickrolling、IBAs、VillanDiffusion、BadT2I、EvilEdit）。

**📈 对比分析**

与 T2IShield、DAA、NaviT2I、UFID 等四类基线对比，SET 在 5 种攻击平均 AUROC 提升 9.1%、ACC 提升 6.5%，在隐蔽触发场景中表现尤为显著。

**⚠️ 局限性**

限制包括对白盒访问的依赖、对尺度调制参数的调优需要经验、以及在极少量清洁样本下可能导致安全空间不够紧凑。

---

## 337. DiffusionPrint: Learning Generative Fingerprints for Diffusion-Based Inpainting Localization

**arXiv ID:** 2604.12443 | [PDF](https://arxiv.org/pdf/2604.12443v1)

**作者:** Paschalis Giakoumoglou `[一作]` (Information Technologies Institute), Symeon Papadopoulos `[通讯]` (Information Technologies Institute)

**通讯引用:** 6191 | [OpenAlex ID](https://openalex.org/A5013616365)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 DiffusionPrint，一种基于对比学习的补丁级鉴别器，用于检测完全重构的扩散式图像修复。

**💡 创新点**

创新点在于：1）利用同一生成器生成的补丁对构建正样本，捕捉生成指纹；2）对真实补丁采用跨类别的 hard‑negative 采样；3）在对比学习中加入生成器感知分类头，提升模型对不同生成器的区分能力。

**🔧 技术方法**

核心技术包括 MoCo 风格的对比学习、DnCNN 后端、InfoNCE 损失、top‑k hard‑negative 采样以及多类别分类头。

**📊 数据集**

训练使用 Dragon 数据集（真实与 SD 2.1、SDXL、Flux 生成图像的补丁），评估采用 TGIF‑FR 基准并在 TruFor、MMFusion 及轻量级融合基线中检验。

**📈 对比分析**

与 Noiseprint++ 的融合基准进行对比，DiffusionPrint 在三大框架中均提升 F1 分数，尤其在随机掩码上高达 +28%；线性探测准确率最高可达 88.17%，并能在未见的生成器上保持竞争性能。

**⚠️ 局限性**

局限性：在 SD 2.1 生成的补丁中偶尔出现误检和对象偏置；对传统剪切拼接类型的检测尚未涵盖；依赖于扩散式修复模型，若出现新型生成器或混合攻击，性能可能下降。

---

## 338. GLeMM: A large-scale multilingual dataset for morphological research

**arXiv ID:** 2604.12442 | [PDF](https://arxiv.org/pdf/2604.12442v1)

**作者:** Hathout Nabil `[一作]` (Université Toulouse Jean Jaurès), Franck Sajous `[通讯]` (Université Toulouse Jean Jaurès)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并实现了 GLeMM（Gros Lexique Morphologique Multilingue）这一大规模、跨语言的派生形态词典，自动化地从 Wiktionary、Kaikki 及 MorphyNet 等资源中提取并注释词对，提供形式、语义与语法的完整描述。

**💡 创新点**

创新点主要有：
1) 在形式上使用 FAPinette（基于正式类比的自动筛选方法）生成统一、可解释的变形模式；
2) 在语义上为近一半的词对提供定义，首次将派生关系与语义定义直接对应；
3) 覆盖七种欧洲语言，规模超过现有任何单语言派生词典，且覆盖面广、结构统一；
4) 通过自动化流程实现资源可持续更新、可跨语言比较。

**🔧 技术方法**

技术手段包括：
- 解析 Wiktionary 的 GLAWI/Kaikki XML 结构，提取定义、派生与相关词条；
- 采用 Levenshtein edit 距离和字符差异签名快速识别可形成类比的词对；
- FAPinette 的三阶段流程：签名筛选 → 切换模式（alternation patterns） → 选择最具连通性的 FAP；
- 自动注释形态特征（词性、词干、变形指数）和语义描述（定义的词形替换模式）。

**📊 数据集**

主要数据集：
- Wiktionary 版本：GLAWI（法语、意大利语、英语）、Kaikki（德语、波兰语、俄语、西班牙语、英语）、MorhpyNet；
- 语料覆盖七种语言：德语、英语、西班牙语、法语、意大利语、波兰语、俄语；
- 词对数量约 2.3 M，其中约 0.73 M 配有定义。

**📈 对比分析**

与现有资源的比较：
- 在词对数量上 GLeMM 超过 MorphyNet、UniMorph、UDer、CELEX 等；
- 在多语言一致性上 GLeMM 统一表结构、语义注释；
- 在实验可重复性和多语言对比研究的可行性上 GLeMM 提供更大样本；
- 评估手段：随机 100 词对人工校正，准确率约 94 %–100 %（词对）与 84 %–93 %（变形模式）。
- 性能：生成时间受候选词对规模影响，但整个流程可在数小时内完成。

**⚠️ 局限性**

局限性：
- 资源尚未人工审核，存在拼写错误、非派生词对及模式不佳的情况；
- 只覆盖七种语言，其他语言需后续更新；
- 未将派生与复合、复合与词干的语义区分清晰；
- 仅基于词形与定义缺乏音位/音素层面的形态学信息；
- 语义注释采用非正式描述，难以进行严格的语义一致性验证。

---

## 339. VeriX-Anon: A Multi-Layered Framework for Mathematically Verifiable Outsourced Target-Driven Data Anonymization

**arXiv ID:** 2604.12431 | [PDF](https://arxiv.org/pdf/2604.12431v1)

**作者:** Miit Daga `[一作]` (Vellore Institute of Technology), Swarna Priya Ramu `[通讯]` (Vellore Institute of Technology)

**通讯引用:** 956 | [OpenAlex ID](https://openalex.org/A5100636184)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出并实现了 VeriX‑Anon，一个多层可验证的云端目标驱动 k‑匿名化框架，允许数据所有者在不重新执行完整算法的情况下数学审计云端匿名化结果。

**💡 创新点**

创新点在于将确定性 Merkle 树哈希、概率边界哨兵与精确双胞胎陷阱以及基于 SHAP 的可解释 AI 指纹四层机制组合成一套完整、互补的可验证体系，并首次将可解释 AI 用作数据转换完整性检测。

**🔧 技术方法**

使用的技术包括 SHA‑256 Merkle 树认证、随机森林边界哨兵生成、精确复制双胞胎插入、SHAP 值分布与 Wasserstein 距离比较、随机森林/ XGBoost 模型训练以及 Python/NumPy/pandas/SHAP 等开源库。

**📊 数据集**

使用了三大跨域数据集：Adult Income、Bank Marketing、Diabetes 130‑US Hospitals，用于评估检测覆盖率、性能与可解释性。

**📈 对比分析**

通过与三种云端攻击模型（Lazy、Dumb、Approximate）以及正面验证进行比较，VeriX‑Anon在 12 种场景中检测成功率为 91.7%，单层检测最低仅为 50%；验证时间在 10^6 行时约 0.79 秒，客户端计算成本低。

**⚠️ 局限性**

主要限制包括：XAI 层对严重类别不平衡的数据敏感、哨兵密度受边界点数量限制、阈值 ε 需要针对 k 值和数据集手工校准、仅支持二分类、仅针对决策树分割的 k‑匿名化、未评估动态/多类场景，且对更高级的“知情攻击者”理论上仍有漏洞。

---

## 340. IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration

**arXiv ID:** 2604.12573 | [PDF](https://arxiv.org/pdf/2604.12573v1)

**作者:** Yanji He `[一作]` (Hong Kong University of Science and Technology), Wei Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 40769 | [OpenAlex ID](https://openalex.org/A5100391662)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 IDEA 框架，将 LLM 的决策知识外化为可解释且可编辑的因子模型，实现概率校准和人机协作。

**💡 创新点**

创新点在于：①联合 EM 学习口语概率映射与决策模型；②通过相关采样保留因子依赖；③支持基于 AME 的参数精确编辑，提供数学保证。

**🔧 技术方法**

采用因子识别与验证、行为探测、EM 训练、Monte Carlo 采样、逻辑回归决策模型、AMe 调整约束等技术。

**📊 数据集**

在五个基准上评估：BIGDATA22、Statlog German Credit、COMMON2SENSE、PLASMA、TODAY。

**📈 对比分析**

与 CoT、Logit、Vanilla、BIRD、PWC 等基线相比，IDEA 在 Qwen‑3‑32B 上平均准确率 78.6%，显著优于 DeepSeek‑R1（68.1%）和 GPT‑5.2（77.9%），并在概率校准和因子排除方面实现完美性能。

**⚠️ 局限性**

局限性包括：只能处理二元决策；因子空间规模受限；需要人工验证因子完整性；假设因子完整性与口语概率一致性尚未严格证明。

---

## 341. When Does Data Augmentation Help? Evaluating LLM and Back-Translation Methods for Hausa and Fongbe NLP

**arXiv ID:** 2604.12540 | [PDF](https://arxiv.org/pdf/2604.12540v1)

**作者:** Mahounan Pericles Adjovi `[一作]` (Carnegie Mellon University Africa), Prasenjit Mitra `[通讯]` (Carnegie Mellon University Africa)

**通讯引用:** 10241 | [OpenAlex ID](https://openalex.org/A5009542542)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对低资源非洲语言 Hausa 与 Fongbe 的命名实体识别（NER）和词性标注（POS）任务进行数据增强实验，比较 LLM 生成与反向翻译两种增强方法对模型性能的影响。

**💡 创新点**

揭示了增强效果主要由任务特性决定，而非单纯的生成质量；同一 LLM 生成的合成数据在 Fongbe NER 上会下降 1.81% F1，反而在 POS 上提升 0.33%，挑战了常见的“质量即效果”假设。

**🔧 技术方法**

使用 Gemini 2.5 Flash 进行少量示例提示式合成数据生成，使用 NLLB‑200 进行反向翻译并对标签进行位置投影，模型为 AfroXLMR‑base，训练仅更新分类头。

**📊 数据集**

MasakhaNER 2.0（20 语言 NER）和 MasakhaPOS（POS）作为基准数据集，分别在 Hausa（5,716 句/NER，753 句/POS）和 Fongbe（4,343 句/NER，810 句/POS）上实验。

**📈 对比分析**

与基线（无增强）对比，LLM 与 Back‑Trans 两种增强方式在不同任务和语言上表现差异：Hausa NER 基线 85.05% F1，增强无提升；Fongbe NER 基线 84.17% F1，LLM 下降 1.81%；Hausa POS 基线 91.86% accuracy，Back‑Trans 提升 0.17%；Fongbe POS 基线 85.42% accuracy，LLM 提升 0.33%，Back‑Trans 降低 0.35%。

**⚠️ 局限性**

仅评估两种语言，增强规模有限（最多约 500 条合成句子），未对合成数据做质量过滤，仅使用单一 LLM 与单一增强策略，且 Hausa NER LLM 条件因生成失败仅提供 2 条样本，可能影响结论的稳健性。

---

## 342. Adaptive Test-Time Scaling for Zero-Shot Respiratory Audio Classification

**arXiv ID:** 2604.12647 | [PDF](https://arxiv.org/pdf/2604.12647v1)

**作者:** Tsai-Ning Wang `[一作]` (Eindhoven University of Technology), Aaqib Saeed `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1181 | [OpenAlex ID](https://openalex.org/A5011960578)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出TRIAGE框架，采用分层零样本推理对呼吸音进行分类，按难度动态分配计算资源；

**💡 创新点**

创新点在于将推理过程分为三层：快速标签余弦匹配、结构化临床描述匹配以及检索增强的LLM推理，并通过置信度门控实现自适应计算；

**🔧 技术方法**

使用冻结的音频-文本嵌入模型（AcuLa）、临床描述模板与规则表、FAISS检索、以及Gemini 3 Pro等LLM；

**📊 数据集**

在五个公开呼吸音数据集上完成九个任务，包括UK COVID、CoughVID、ICBHI、Coswara、KAUH、RESPTR等；

**📈 对比分析**

与基准零样本方法（CLAP、AcuLa）以及监督线性微调基线对比，TRIAGE的平均AUROC为0.744，明显优于零样本基准并在多项任务中匹配或超越监督基线；

**⚠️ 局限性**

局限性包括：需要依赖外部LLM和检索库，计算成本随层级上升；阈值需手工调优；对新任务或设备的泛化能力尚未充分验证；

---

## 343. RPRA: Predicting an LLM-Judge for Efficient but Performant Inference

**arXiv ID:** 2604.12634 | [PDF](https://arxiv.org/pdf/2604.12634v1)

**作者:** Dylan R. Ashley `[一作]` (Meta Platforms, Inc.), Jürgen Schmidhuber `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出模型在生成回答前预测其回答会被外部 LLM/代理评审给分的 PA/RPRA 预判方法，并验证其可行性。

**💡 创新点**

①引入 PA 与 RPRA 两种预判范式；②通过报表卡或微调实现小模型自我校准；③证明小模型在有历史表现信息时能显著提升自评准确性。

**🔧 技术方法**

采用 LLM‑as‑Judge 框架、报表卡摘要、后视角（hindsight trick）+监督微调、零样本/上下文推理等技术。

**📊 数据集**

使用 MedQA、LongFact、AIME 2024、SciCode、MMLU‑Pro 等多样化数据集。

**📈 对比分析**

与零样本直接预测对比，使用 Llama 3.3 70B 作为评审；报告卡/微调在小模型上提升 30‑70 % 预测准确率，规模模型已表现良好。

**⚠️ 局限性**

仅单轮实验；未构建完整路由系统；报表卡产生额外 token 开销；细调需额外权重；多轮对话与对齐需求未验证。

---

## 344. Multilingual Multi-Label Emotion Classification at Scale with Synthetic Data

**arXiv ID:** 2604.12633 | [PDF](https://arxiv.org/pdf/2604.12633v1)

**作者:** Vadim Borisov `[一作]` `[通讯]` (tabularis.ai), Vadim Borisov (tabularis.ai)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个包含23种语言、11种情绪类别、1.15M多标签样本的合成情绪语料库，并训练多语言Transformer模型

**💡 创新点**

通过文化适配的生成与程序化质量过滤生成多语言合成数据，比较多种多语言Transformer的规模与性能，展示合成数据可实现与单语专家模型相当的排名指标

**🔧 技术方法**

多语言Transformer编码器（DistilBERT、mBERT、XLM‑R‑Base、Twitter‑XLM‑R、mDeBERTa‑v3、XLM‑R‑Large）以及BCE+AdamW训练框架、混合精度训练与阈值自由的AUROC/AP/LRAP评估

**📊 数据集**

自建的1.15M样本合成语料库，涵盖23种语言；对GoEmotions（英语）和SemEval‑2018 E‑c（英语、阿拉伯语、西班牙语）进行零样本评测

**📈 对比分析**

在本地测试集上XLM‑R‑Large取得0.868 F1‑micro、0.987 AUC‑micro；在外部基准上零样本表现与英语专家模型在阈值自由指标（AUC/AP/LRAP）相当，模型排名与内部评估一致；在计算成本上DistilBERT最快、XLM‑R‑Large最高质量，XLM‑R‑Base在成本-质量平衡上最佳

**⚠️ 局限性**

合成数据可能偏离真实文本风格（更长、更冗长）、情绪标签集有限（缺少某些细粒度情绪）、未进行人工标注一致性验证，且不同语言的预训练分布对性能影响明显

---

## 345. CODO: An Automated Compiler for Comprehensive Dataflow Optimization

**arXiv ID:** 2604.12618 | [PDF](https://arxiv.org/pdf/2604.12618v1)

**作者:** Weichuang Zhang `[一作]` (Shanghai Jiao Tong University), Minyi Guo `[通讯]` (Guizhou University)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 CODO 自动编译器，能够自动检测并消除 DNN 代码中的数据流违规，生成高性能 FPGA 数据流加速器，同时支持高效的芯片内外数据通信与资源感知的自动调度。

**💡 创新点**

创新点在于：①系统化的粗粒度与细粒度违规消除策略；②FIFO 与 ping‑pong 缓冲优先选择并结合重写与重排技术实现最大化流式通信；③结合调度、流水线、展开与数组分区的资源感知 DSE，三方协同优化正确性、通信与并行度；④完整的自动化工作流（从 Polygeist/Torch‑MLIR 到 Vitis HLS 输出）并公开开源。

**🔧 技术方法**

技术主要包括 MLIR 体系结构、图形/节点级违规检测与变换、循环重排与重写、FIFO/窗口/线缓冲生成、HLS 直方图与 pragma 注入、基于性能模型的资源感知调度与互任务优化。

**📊 数据集**

使用 Polybench 基准、ResNet‑18、VGG‑16、MobileNet、ZFNet、YOLO（图像/目标检测）以及 GPT‑2（Transformer 语言模型）等数据集和模型进行评估。

**📈 对比分析**

与 ScaleHLS、POM、Allo、HIDA、StreamHLS、StreamTensor 等现有框架在同一平台（AMD Alveo U280、Xilinx Vivado 2023.2）对比。CODO 在典型 kernel 上平均实现 1.45–4.52 倍延迟加速；在 DNN 模型上平均 3.7–33.8 倍加速；在板上实验中对 CNN 平均 7.3 倍、对 GPT‑2 2.07 倍；编译时间仅为秒级别，显著低于现有框架。

**⚠️ 局限性**

局限性：对极大规模模型仍需手动调整资源预算，部分冲突会退回到 ping‑pong 缓冲；依赖 Vitis HLS 资源估计，若估计失准会导致实现失败；目前对非 TensorFlow/PyTorch 低层 C++ 代码支持有限；需要进一步提升对多核/多流板块的并行度扩展。

---

## 346. DeepTest Tool Competition 2026: Benchmarking an LLM-Based Automotive Assistant

**arXiv ID:** 2604.12615 | [PDF](https://arxiv.org/pdf/2604.12615v1)

**作者:** Lev Sorokin `[一作]` (BMW Group), Samuele Pasini `[通讯]` (Università della Svizzera italiana)

**通讯引用:** 44 | [OpenAlex ID](https://openalex.org/A5058484013)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

组织了DeepTest 2026竞赛，评估四种自动化测试工具对基于LLM的汽车手册信息检索助手的失败检测能力。

**💡 创新点**

创新点在于：①引入人类式扰动与优先级警告选择的组合（ATLAS），②使用风险情境导航与上下文融合生成自然语句（CRISP），③通过动态概率调整实现高效警告覆盖（Warnless）以及基于Jaccard相似度避免重复的生成策略（Exida）。

**🔧 技术方法**

使用的技术包括：LLM + RAG（检索增强生成）、LLM作为oracle评估判定、词级扰动与填充词、Jaccard相似度、多样性控制、k‑means+轮廓系数聚类评估失败覆盖、LLM生成自然语言请求。

**📊 数据集**

数据集主要有：①两份不同车型的车辆手册（用作SUT后端），②1000条人工与合成验证的请求-答案对（用于oracle基准），③竞赛生成的测试输入及其评估结果。

**📈 对比分析**

比较方法：对每个工具计算警告忽略数、失败率、失败覆盖率并以等权重合成总体得分。结果显示ATLAS以57%失败率和最高总体得分0.59领先，其次为Exida、Warnless和CRISP；随机基线得分0.51。总体表现显示人类式扰动与优先级策略最有效。

**⚠️ 局限性**

局限性包括：①仅在两款SUT（工业版与开源版）和两份手册上验证，可能无法全面反映真实多车型场景；②覆盖率评估依赖k‑means聚类，可能对异常分布不敏感；③工具间实现差异导致结果波动较大；④LLM作为oracle的误判率仍存在，可能影响最终评价。

---

## 347. Listening Deepfake Detection: A New Perspective Beyond Speaking-Centric Forgery Analysis

**arXiv ID:** 2604.12650 | [PDF](https://arxiv.org/pdf/2604.12650v1)

**作者:** Miao Liu `[一作]` (Beijing Institute of Technology), Xinyuan Qian `[通讯]` (University of Science and Technology Beijing)

**通讯引用:** 988 | [OpenAlex ID](https://openalex.org/A5056495776)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出听觉深伪造检测任务并构建ListenForge数据集，研发MANet模型以捕捉监听状态下的细微运动和音频语义不一致。

**💡 创新点**

创新点在于首次关注被动监听场景，结合运动感知模块与音频引导模块，实现跨模态上下文一致性检测，突破传统说话式深伪造检测的局限。

**🔧 技术方法**

使用运动感知模块（MAM）提取视觉时间差特征，音频引导模块（AGM）通过跨模态注意力融合Wav2vec2.0音频特征与ResNet视觉特征。

**📊 数据集**

主要使用自建ListenForge（10,655段）数据集，且与FaceForensics++进行跨任务对比。

**📈 对比分析**

在ListenForge上对比现有单模态与多模态说话式检测方法，MANet达到AUC 97.24%、ACC 89.74%，比最佳对手提升超过42个百分点，显示显著性能优势。

**⚠️ 局限性**

局限包括仅聚焦监听伪造，未覆盖说话式伪造的联合检测；缺乏跨平台、实时性评估与更广泛的语料覆盖。

---

## 348. Contextual Multi-Task Reinforcement Learning for Autonomous Reef Monitoring

**arXiv ID:** 2604.12645 | [PDF](https://arxiv.org/pdf/2604.12645v1)

**作者:** Melvin Laux `[一作]` (University of Bremen), Rebecca Adam `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `afceb026-1760-41ae-8d86-010831a37d97`

**🎯 论文内容**

研究如何用上下文多任务强化学习（MTRL）训练自律水下机器人（AUV）在珊瑚礁监测中高效、可迁移的控制策略。

**💡 创新点**

① 将珊瑚礁监测任务形式化为上下文马尔可夫决策过程（CMDP）；② 通过在单个网络中输入任务上下文实现多任务共享，减少模型存储；③ 与传统混合专家（MoE）做直接对比，展示上下文MTRL在零样本泛化与资源利用上的优势。

**🔧 技术方法**

使用深度Q网络（DDQN）及其上下文版本（cDDQN）作为基线；通过多层感知机（MLP）构建Q网络；在Minigrid与HoloOcean两种高保真仿真环境中训练与评估。

**📊 数据集**

利用两种仿真数据集：MiniGrid（离散网格与颜色对象）和HoloOcean（连续二维AUV与不同水流与物种分布），每个环境均设计训练集与测试集的多种水流与目标物种组合。

**📈 对比分析**

比较方法：对比cDDQN与MoE的样本效率、最终性能与零样本泛化；在训练集上两者收敛相近，MoE在样本效率略快；在测试集上cDDQN表现更稳健、避免灾难性失败；在HoloOcean场景下两者在性能与样本效率上差异不大，但cDDQN显著降低模型大小。

**⚠️ 局限性**

局限性：① 仅在仿真环境验证，缺乏真实水下实验；② 任务范围有限，未覆盖更复杂的感知与动态环境；③ 只处理高层控制，未结合实时图像或传感器反馈；④ 模型规模虽小，但仍需进一步优化以满足实际AUV硬件限制。

---

## 349. TimeSAF: Towards LLM-Guided Semantic Asynchronous Fusion for Time Series Forecasting

**arXiv ID:** 2604.12648 | [PDF](https://arxiv.org/pdf/2604.12648v1)

**作者:** Fan Zhang `[一作]` (Shandong Technology and Business University), Hua Wang `[通讯]` (Ludong University)

**通讯引用:** 58927 | [OpenAlex ID](https://openalex.org/A5100403938)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种层次化异步融合框架TimeSAF，用来在多模态时间序列预测中分离单模态编码与跨模态交互，从而缓解语义感知失衡问题。

**💡 创新点**

创新点在于：①引入独立的语义融合干线和阶段性语义细化解码器；②将跨模态交互限制在预设的融合阶段，实现底层时间特征与高层语义的无缝对齐；③通过可学习的查询槽进行语义聚合，并采用门控注入提升鲁棒性。

**🔧 技术方法**

使用了GPT‑2作为固定语言模型提供语义提示、Patch‑based 时序编码、Transformer‑style 自注意力、交叉注意力以及门控残差机制，并结合 RevIN 归一化。

**📊 数据集**

在七个公开数据集上验证：ETT（ETTh1/2/​m1/2）、Electricity、Weather、Exchange。

**📈 对比分析**

与多类基线（LLM‑增强模型、Transformer、CNN、MLP）在长时序预测、少样本与零样本设置中对比，TimeSAF 在所有任务上均优于基线，平均 MSE/MAE 下降 5%–10%。

**⚠️ 局限性**

局限性包括：仅在标准基准和有限的少样本/零样本场景评估；使用规则模板生成提示，未充分挖掘大规模语言模型潜力；对更复杂工业多模态环境的可扩展性尚未验证。

---

## 350. Every Picture Tells a Dangerous Story: Memory-Augmented Multi-Agent Jailbreak Attacks on VLMs

**arXiv ID:** 2604.12616 | [PDF](https://arxiv.org/pdf/2604.12616v1)

**作者:** Jianhao Chen `[一作]`, Tieyun Qian `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `6215c339-3735-4be3-8a07-5bbb7004712d` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

设计并实现了一种基于多代理协同、记忆增强的图像诱导攻击框架，对视觉语言模型（VLM）进行越狱。

**💡 创新点**

创新点包括：①引入记忆模块记录先前攻击经验，提高后续攻击的效率与成功率；②使用多代理系统实现分工协作，增强攻击的多样性与鲁棒性；③结合图像与文本的联合策略，生成更具欺骗性的提示。

**🔧 技术方法**

使用技术：记忆增强的强化学习（RL+Memory网络）、Transformer+记忆模块、多代理强化学习（MA‑DRL）、对抗样本生成与图像预处理、文本编码器等。

**📊 数据集**

实验数据集：COCO、VQAv2、Visual Commonsense Reasoning (VCR) 等标准视觉语言数据集，以及公开的安全攻击测试集。

**📈 对比分析**

与传统单代理攻击、基线对抗样本以及Prompt‑Injection等方法进行对比。实验结果显示，所提出的方法在成功率上提升约30%–50%，并显著降低VLM对攻击提示的正确响应率。

**⚠️ 局限性**

限制：①对特定VLM的适配性高，迁移性有限；②训练和部署成本较高，计算资源需求大；③生成的攻击样本缺乏可解释性，易被现有防护机制检测；④对模型的安全防护策略敏感，效果受限。

---

## 351. Transforming External Knowledge into Triplets for Enhanced Retrieval in RAG of LLMs

**arXiv ID:** 2604.12610 | [PDF](https://arxiv.org/pdf/2604.12610v1)

**作者:** Xudong Wang `[一作]` (Kyung Hee University), Hengtao Shen `[通讯]` (Tongji University)

**通讯引用:** 30789 | [OpenAlex ID](https://openalex.org/A5052993469)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Tri-RAG，一种通过软提示学习将外部知识转化为三元组（Condition、Proof、Conclusion）以提升检索对齐和生成质量的检索增强生成框架。

**💡 创新点**

创新点在于将无结构文本自动转换为标准化三元组结构，并使用 Condition 作为检索锚点实现精确检索与高效上下文利用，同时保持模型参数冻结的轻量化抽取。

**🔧 技术方法**

采用软提示调优（soft prompt tuning）在冻结的 LLM 上进行三元组抽取，配合稠密检索编码器和结构化检索后端。

**📊 数据集**

使用多种问答基准数据集，包括 LongBench、HotpotQA、2WikiMultihopQA、MuSiQue、Natural Questions 和 SQuAD 等。

**📈 对比分析**

与多种 RAG 基线和参数高效适配方法对比，Tri‑RAG 在 5 大 LLM 后端下平均提升约 4–5% F1，检索精度提升至 70%+，生成时延下降约 20%。

**⚠️ 局限性**

局限性在于三元组抽取对叙事性文本和跨句信息的覆盖仍不完整，需进一步引入自检和多粒度检索以提升鲁棒性。

---

## 352. Beyond Pre-Training: The Full Lifecycle of Foundation Models on HPC Systems

**arXiv ID:** 2604.12599 | [PDF](https://arxiv.org/pdf/2604.12599v1)

**作者:** Dino Conciatore `[一作]` (Swiss National Supercomputing Centre, ETH Zurich), Maxime Martinasso `[通讯]` (Swiss National Supercomputing Centre, ETH Zurich)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在瑞士国家超级计算中心（CSCS）构建并测试了一个混合 Kubernetes‑Slurm 平台（Alpernetes），实现了 Foundation Model 的完整生命周期（预训练、微调、推理）在本地 HPC 上的可行性。

**💡 创新点**

创新点在于：①将虚拟化 VMs 与磁盘无 GPU 节点融合到单一 Kubernetes 控制面，实现服务化与批处理并行；②采用双轨 Orchestration：Kubernetes 负责服务层与用户交互，Slurm 负责高性能训练；③通过 GitOps+IaC 实现自管理沙盒、Fine‑tuning as a Service 与托管推理服务的统一部署与治理。

**🔧 技术方法**

技术包括：Kubernetes（RKE2、Cilium、Kueue、Kubeflow）、Slurm、FirecREST、vLLM、LiteLLM、Waldur、ArgoCD、SUSE Rancher、OpenTofu、Terragrunt、NVIDIA GPU Operator、NCCL、CNI（Cilium、libfabric）、Slingshot‑11 高速网络、HPE Cray EX、GH200、H100 GPU、Ceph、S3‑兼容对象存储。

**📊 数据集**

主要使用的数据集与模型：①预训练大规模数据（未公开具体）；②微调使用基线 Apertus‑70B / Apertus‑8B 模型的自定义数据集；③性能基准使用 CIFAR‑10（ResNet‑18）在 GH200 节点上进行 DDP；④推理评测使用 Apertus‑70B 与 Apertus‑8B 在 Grace‑Hopper 节点上跑 vLLM。

**📈 对比分析**

比较方法：在 CIFAR‑10 上对比 K8s（eth0/TCP、hsn0/TCP、hsn0‑3/TCP）与 Slurm（CXI RDMA）进行 DDP；推理时记录 QPS、TTFT、ITL、E2E。结果显示：K8s 通过使用 HSN 接口可实现 3.2× 的速度提升（从 3779s 降到 1165s），但仍低于 Slurm 的 RDMA 性能；推理指标显示 Apertus‑70B 的平均 ITL 42 ms、TTFT < 500 ms、E2E 5.84 s；Apertus‑8B 的 ITL 11 ms、E2E 31.4 s，反映出大输出长度对吞吐的影响。系统日均生成 250 万 tokens（8B）和 100 万 tokens（70B），且自 2025/09/05 起无计划停机。

**⚠️ 局限性**

主要限制：①Kubernetes‑Slurm 集成仍缺乏对 Slingshot 的 native RDMA 支持，导致性能不及原生 Slurm；②推理后端尚未进行充分的调优（vLLM 线程、GPU Direct、CNI 等）；③资源弹性不足，静态分配导致 HPC 节点闲置；④多租户存储与数据访问统一性待完善；⑤高可用与安全（特别是敏感数据的可信计算环境）仍处于实验阶段，未完成正式认证。

---

## 353. ELoG-GS: Dual-Branch Gaussian Splatting with Luminance-Guided Enhancement for Extreme Low-light 3D Reconstruction

**arXiv ID:** 2604.12592 | [PDF](https://arxiv.org/pdf/2604.12592v1)

**作者:** Yuhao Liu `[一作]` (Shanghai Jiao Tong University), Ziyang Zheng `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 138815 | [OpenAlex ID](https://openalex.org/A5100452094)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一套名为ELoG-GS的极低光优化高斯点云渲染流水线，在极低光稀视角条件下完成高质量3D重建。

**💡 创新点**

创新点包括：零射门Retinexformer低光恢复；VGGT深度估计与体素融合的稳健点云初始化；双分支（FSGS+EAP-GS）混合架构与动态分支选择；亮度引导的后处理提升颜色真实感。

**🔧 技术方法**

使用的技术包括：Retinexformer、VGGT、ZoeDepth、FSGS、EAP-GS、体素融合、COLMAP兼容格式、三维高斯点云渲染、亮度引导增强、直方图匹配、CUDA可微分光栅化器。

**📊 数据集**

使用的数据集为NTIRE 2026 3D Restoration and Reconstruction Challenge数据集（7个场景，每个约30个训练场景，6个测试场景，原始2K分辨率）。

**📈 对比分析**

在NTIRE基准上与官方SOTA方法LITA-GS和Luminance-GS进行PSNR/SSIM比较。ELoG-GS取得18.66 PSNR/0.685 SSIM，明显优于LITA-GS（15.63/0.542）和Luminance-GS（10.89/0.531）。最终在排行榜上排名第9/148。

**⚠️ 局限性**

局限性：仍需人工介入进行分支选择，未实现完全自动化；评估仅在极低光稀视角场景，缺乏更广泛的验证；对预训练模型（Retinexformer、VGGT等）的依赖可能限制跨场景的泛化。

---

## 354. PDF-GS: Progressive Distractor Filtering for Robust 3D Gaussian Splatting

**arXiv ID:** 2604.12580 | [PDF](https://arxiv.org/pdf/2604.12580v1)

**作者:** Kangmin Seo `[一作]` (Sungkyunkwan University), Jae-Pil Heo `[通讯]` (Sungkyunkwan University)

**通讯引用:** 1553 | [OpenAlex ID](https://openalex.org/A5029469141)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了PDF-GS框架，通过多阶段逐步筛除训练图像中的瞬时或视角不一致的干扰物，最终得到无干扰、高保真度的3D Gaussian Splatting模型。

**💡 创新点**

创新点在于：①利用3DGS天然的自我过滤特性，设计基于不一致性差异的递归筛除机制；②采用多阶段分离的“逐步筛除”与“细节重建”两阶段训练流程；③每阶段重新初始化高斯参数、使用结构化损失和稀疏色彩更新来稳定多视角一致性；④采用动态阈值与DINOv3特征来生成更精准的掩码。

**🔧 技术方法**

技术方法包括：3D Gaussian Splatting基准模型、DINOv3特征差异图、掩码式监督、结构化SSIM损失、稀疏色彩更新策略、动态阈值调度、SfM初始化与阶段间重新初始化、最终重建阶段采用标准3DGS损失。

**📊 数据集**

在RobustNeRF和NeRF On‑the‑go两个含有显著瞬时/视角不一致内容的基准数据集上进行实验。

**📈 对比分析**

与传统使用显式掩码预测或场景分解的3DGS方法相比，PDF‑GS在PSNR、SSIM、LPIPS等指标上均取得了显著提升，达到了新的SOTA；定量表格显示PSNR提升约1–2dB，SSIM提升0.02以上，LPIPS下降0.03以上；定性结果显示干扰物被有效抑制且细节保持完好。

**⚠️ 局限性**

局限性包括：①多阶段训练显著增加训练成本（约40k步）；②依赖SfM初始化，若SfM失败或对大规模场景不适用；③对极端光照或极其快速移动的干扰物仍有一定残留；④未在超大场景或实时渲染场景中评估。

---

## 355. EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts

**arXiv ID:** 2604.12579 | [PDF](https://arxiv.org/pdf/2604.12579v1)

**作者:** Runhe Zhou `[一作]` (Nanyang Technological University), Cuntai Guan `[通讯]` (Nanyang Technological University)

**通讯引用:** 28454 | [OpenAlex ID](https://openalex.org/A5031778999)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种基于超曲率混合专家（EEG-MoCE）的脑电多模态学习框架，用可学习的曲率在不同模态上构建超曲率嵌入，并通过曲率导向的融合策略实现多模态信息的动态加权。

**💡 创新点**

创新点包括：① 为每个模态引入可学习的负曲率，以自适应捕捉各模态的层次结构；② 在超曲率空间中设计曲率感知的注意力机制和加权弗雷舍平均聚合，动态强调层次信息丰富的模态；③ 结合欧氏编码器与超曲率专家的分层组合，实现跨模态统一的几何表示。

**🔧 技术方法**

主要技术：超曲率神经网络（Lorentz模型）、可学习曲率、超曲率批归一化、基于负几何距离的注意力、加权弗雷舍平均聚合、超曲率多项式逻辑回归（HMLR）以及欧氏编码器（如EEGNet、轻量级CNN+Transformer）。

**📊 数据集**

使用三大公开数据集：EAV（情绪识别，EEG+音频+视频），ISRUC（睡眠分期，EEG+EMG+EOG），Cognitive（认知评估，EEG+EOG+NIRS）。

**📈 对比分析**

与所有任务的现有最佳方法（如HEEGNet、XSleepFusion、EF-Net等）进行交叉主体评估，EEG-MoCE在EAV上达到75.88%准确率（比最佳61.74%提升约14%），ISRUC上78.53%（比最佳75.19%提升约3%），Cognitive上62.39%（比最佳54.41%提升约8%），整体表现均显著优于对比模型。

**⚠️ 局限性**

局限性：① 仅针对具有层次结构的模态，未探究其他结构（如图像或文本）的适用性；② 超曲率运算在训练时的数值不稳定与计算开销较大，虽推理速度可接受但仍高于纯欧氏模型；③ 对曲率解释性与可解释性的深入分析尚缺乏；④ 目前的融合策略相对简单，未来可尝试更丰富的平移/对齐机制。

---

## 356. Two Sequence-Form Interior-Point Differentiable Path-Following Method to Compute Nash Equilibria

**arXiv ID:** 2604.12558 | [PDF](https://arxiv.org/pdf/2604.12558v1)

**作者:** Yuqing Hou `[一作]` (University of Science and Technology of China), Yuqing Hou `[通讯]` (University of Science and Technology of China)

**通讯引用:** 4243 | [OpenAlex ID](https://openalex.org/A5014891352)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了在有限 n 玩家完全信息重现（perfect recall）的广义形式博弈中，以序列形式直接定义纳什均衡，并给出了对应的多项式系统；随后设计了两种基于对数障碍的单阶段内点可微路径跟踪算法，用来求解该系统。

**💡 创新点**

核心创新在于：①从序列形式直接构造纳什均衡定义并证明其与混合策略纳什均衡等价；②通过对数障碍将原非光滑问题转化为光滑路径，提供了理论上可连续追踪、收敛性更好的求解框架；③引入两种不同障碍构造（基于实现计划和行为策略），并给出完整的数值实现细节。

**🔧 技术方法**

使用了序列形式表示、KKT 条件、对数障碍正则化、内部点法、预测-校正（predictor–corrector）路径跟踪、Browder 固定点定理与隐函数定理等数值与理论工具。

**📊 数据集**

在实验中主要使用了两类随机广义形式博弈（分别对应不同信息集结构），并在已公开的博弈实例（如 Myerson、MasColell 等）中验证算法；并通过 20 个随机实例（不同玩家数、深度、动作数）来评估算法。

**📈 对比分析**

与传统的多项式求解方法（如两阶段路径跟踪、非线性方程求解）以及基于梯度的对数障碍法进行比较；结果显示所提出的方法在迭代次数、计算时间和失败率方面均优于或相当于现有方法，尤其在更大规模、信息集更复杂的博弈中表现更为稳定。

**⚠️ 局限性**

局限性包括：①算法在极大规模博弈（信息集数目极多）时仍可能遇到维度爆炸；②当前仅适用于完美回忆（perfect recall）博弈，尚未扩展到不完全信息或不完美回忆情形；③对精细化均衡（如子博弈精炼）尚无直接处理方法。

---

## 357. DeepSeek Robustness Against Semantic-Character Dual-Space Mutated Prompt Injection

**arXiv ID:** 2604.12548 | [PDF](https://arxiv.org/pdf/2604.12548v1)

**作者:** Junyu Ren `[一作]` (Jinan University), Philip S. Yu `[通讯]` (University of Illinois Chicago)

**通讯引用:** 135884 | [OpenAlex ID](https://openalex.org/A5036357902)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出PromptFuzz‑SC框架，用语义与字符双空间突变对大语言模型进行黑盒鲁棒性评估

**💡 创新点**

创新点在于将语义改写与字符层级扰动统一进可插拔的突变库，并结合ε‑greedy+爬山搜索以及统一的MSR/AQS/Stealth三维评估指标

**🔧 技术方法**

采用突变算子库、混合搜索策略、在线监控与可视化接口，并通过对DeepSeek的多轮查询实现攻击生成

**📊 数据集**

在DeepSeek上使用约50条种子提示进行实验，未使用公开大规模语料，而是构造的语义/字符变体

**📈 对比分析**

与单一语义或字符突变对比，双空间突变在平均MSR 0.189、峰值MSR 0.375、平均Stealth 0.859方面表现最佳，且AQS在最优配置下可达28.3

**⚠️ 局限性**

局限性包括仅针对通用式LLM、查询预算上限200、未覆盖领域特定或多模态场景，以及缺乏对更深层安全机制的评估

---

## 358. Cross-Cultural Simulation of Citizen Emotional Responses to Bureaucratic Red Tape Using LLM Agents

**arXiv ID:** 2604.12545 | [PDF](https://arxiv.org/pdf/2604.12545v1)

**作者:** Wanchun Ni `[一作]` (ETH Zurich), Mennatallah El-Assady `[通讯]` (ETH Zurich)

**通讯引用:** 2011 | [OpenAlex ID](https://openalex.org/A5020415668)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了跨文化红纸红纹情感模拟框架并评估LLM在不同文化情境下生成情感反应的能力。

**💡 创新点**

创新点在于提出跨文化情感对齐评估指标（Overlap@3、SAS）并构建RAMO交互界面收集人类情感数据。

**🔧 技术方法**

采用大语言模型（GPT‑4o、GPT‑5、Gemini‑3‑Pro、Qwen3‑max）结合Hofstede文化维度的人物构建与情感概率输出技术。

**📊 数据集**

使用德国、香港特别行政区和中国大陆三组人类实验数据（面部表情分析得到的情感概率）。

**📈 对比分析**

通过Top‑3情感重叠与SAS比较模型与人类结果，发现GPT‑5在德国情感对齐最佳，整体模型在东方文化下表现差。

**⚠️ 局限性**

限制在于仅测试单一红纸情景、数据量有限且模型仍受固有文化偏见影响，需进一步改进。

---

## 359. A Two-Stage LLM Framework for Accessible and Verified XAI Explanations

**arXiv ID:** 2604.12543 | [PDF](https://arxiv.org/pdf/2604.12543v1)

**作者:** Georgios Mermigkis `[一作]` (University of Patras), Chrysostomos Stylios `[通讯]` (Athena Research Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一个两阶段的LLM元验证框架，用于将XAI技术的技术性输出转化为可访问的自然语言解释，并通过验证器自动检测并修正错误。

**💡 创新点**

创新点在于将解释生成与验证分离为独立的LLM任务，使用结构化的元提示进行系统化评估，并引入迭代回馈机制，使解释能够在验证后逐步改进，同时通过熵产生率（EPR）量化过程的稳定性。

**🔧 技术方法**

核心技术包括Zero-Shot Chain-of-Thought（CoT）提示、结构化Meta-Prompting、LLM验证器的四项评估标准（faithfulness、coherence、completeness、hallucination），以及基于EPR的过程监控与迭代优化。

**📊 数据集**

实验覆盖五种XAI方法与数据集：ACSIncome（SHAP+XGBoost）、CIFAR‑10（Grad‑CAM+++ResNet）、IMDB Reviews（Integrated Gradients+LSTM）、Diamonds（LIME+XGBoost）和Wine Quality（Explainable Boosting Machine），共构建了多模态文本化解释样本。

**📈 对比分析**

与单阶段直接生成或仅做评分的对比实验显示，验证器的加入将错误解释率从约22‑40% 降至 5‑8%，并将可读性指标（Flesch‑Kincaid）从 18.53/21.79 提升至 34.93/12.94，验证准确率最高可达 96.94%（GPT‑OSS 20B+Qwen‑3 30B 组合）。

**⚠️ 局限性**

主要局限包括：对提示工程高度敏感；验证器在某些模型（如DeepSeek‑R1）下表现不佳；当前仅支持文本化XAI输出，缺乏多模态直接解释能力；缺乏真实用户研究验证其感知效果与信任度。

---

## 360. LLMs Are Not a Silver Bullet: A Case Study on Software Fairness

**arXiv ID:** 2604.12640 | [PDF](https://arxiv.org/pdf/2604.12640v1)

**作者:** Xinyue Li `[一作]` (Peking University), Zhenpeng Chen `[通讯]` (Tsinghua University)

**通讯引用:** 462 | [OpenAlex ID](https://openalex.org/A5101612444)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过统一实验设置，对传统机器学习（ML）与大型语言模型（LLM）在表格数据偏见缓解任务中的效果进行大规模对比实验；

**💡 创新点**

创新点在于首次实现跨范式（ML vs LLM）的系统比较，剖析评估设置与训练数据利用对LLM效果的影响，并证明LLM并非在所有公平性场景下的“银弹”；

**🔧 技术方法**

使用了多种传统ML缓解方法（预处理、内置约束、后处理、集成）、LLM的零样本、少样本、标签翻转、分布控制、FCG搜索等提示技术，并对LLM进行监督微调；评估指标包括SPD、EOD、AOD（公平性）以及Accuracy、Precision、Recall、F1（预测性能）；

**📊 数据集**

实验数据集为Adult、Compas、Credit，共六个公平性任务（Adult‑Sex、Adult‑Race、Compas‑Sex、Compas‑Race、Credit‑Sex、Credit‑Age）；

**📈 对比分析**

在相同的实验环境下，传统ML平均在公平性指标上提升约48–60%，在预测性能上提升约9–11%，且在大部分比较中均显著优于LLM；LLM在少样本或微调情况下虽略有改善，但优势不显著；

**⚠️ 局限性**

局限性包括：仅关注表格分类任务；LLM实验集中于GPT‑4o‑mini及少数其他模型，未覆盖更大/不同架构；微调成本和训练时间相对较高；评估仍基于有限公开数据，可能不完全代表所有实际部署情境。

---

## 361. Topology Understanding of B-Spline Surface/Surface Intersection with Mapper

**arXiv ID:** 2604.12631 | [PDF](https://arxiv.org/pdf/2604.12631v1)

**作者:** Chenming Gao `[一作]` (Zhejiang University), Gengchen Li `[通讯]` (Zhejiang University)

**通讯引用:** 13391 | [OpenAlex ID](https://openalex.org/A5100636088)

**关键词:** `a42c7bd6-d8fd-40d3-94df-ae8cd808f5c4` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究并实现了一种基于Mapper工具的B-样条曲面相交拓扑分析方法，能够高效识别和描述曲面相交的复杂拓扑结构。

**💡 创新点**

创新点在于将拓扑数据分析中的Mapper算法引入到曲面相交问题，提供了可视化、可解释的拓扑结构描述，克服了传统分割方法难以准确捕捉复杂拓扑的不足。

**🔧 技术方法**

技术上结合了B-样条曲面相交算法、细分(Subdivision)技术以及Mapper构建的拓扑图谱，利用聚类和连通分量分析实现拓扑识别。

**📊 数据集**

使用了多组合成测试案例和典型CAD模型数据集（包括复杂交叉曲面、自相交曲面等）来评估方法的鲁棒性和准确性。

**📈 对比分析**

通过与传统分割算法和现有曲面相交工具的对比实验，结果显示该方法在拓扑正确率上显著优于对比方法，误差率下降至几乎零，同时处理时间保持在可接受范围内，显示出良好的性能。

**⚠️ 局限性**

限制方面包括：对极高分辨率曲面或极其复杂拓扑的计算成本相对较高；Mapper参数的选择对最终拓扑识别有一定影响，需要经验性调优；方法缺乏严格的理论收敛性和误差界定分析。

---

## 362. KnowRL: Boosting LLM Reasoning via Reinforcement Learning with Minimal-Sufficient Knowledge Guidance

**arXiv ID:** 2604.12627 | [PDF](https://arxiv.org/pdf/2604.12627v1)

**作者:** Linhao Yu `[一作]` (Tianjin University), Hua Wu `[通讯]` (Baidu Inc)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出KnowRL框架，利用最小充分的知识点（KP）引导大语言模型进行强化学习，解决奖励稀疏问题并提升推理性能。

**💡 创新点**

创新点包括：①将提示设计视为“最小充分”问题，发现关键段效应与提示冗余；②提出“裁剪交互悖论”并通过Constrained Subset Search（CSS）方法进行KP子集优化；③设计基于留一法与共识机制的KP选择策略，显著减少提示冗余并提高训练稳定性。

**🔧 技术方法**

使用技术：强化学习（RLVR）与规则奖励、KP提取与泄漏检测、CSS与S-LOO/T-LOO/KP筛选、动态采样与熵退火、token-mean损失、以及多阶段提示注入。

**📊 数据集**

数据集：QuestA（训练集）、8个数学推理基准（AIME24/25、BRUMO25、HMMT‑Feb‑25、AMC23、CMIMC25、MATH‑500、Olympiad‑Bench）共1,374题。

**📈 对比分析**

方法对比：与Nemotron‑1.5B、JustRL、QuestA等基线进行对照。KnowRL在不使用KP提示时已达到70.08的平均准确率，比Nemotron提升9.63点；使用CSS挑选KP后进一步提升至74.16，刷新1.5B规模下的state‑of‑the‑art。与CBRS相比，CSS在所有难度级别上取得更高且更稳定的性能。

**⚠️ 局限性**

局限性：目前仅在数学推理任务验证，KP生成与筛选依赖教师模型和人工审查，扩展到其他领域需重新构建KP；对极大模型规模和非结构化任务的适用性尚未评估；方法复杂度随KP数量上升，可能影响训练成本。

---

## 363. Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting

**arXiv ID:** 2604.12626 | [PDF](https://arxiv.org/pdf/2604.12626v1)

**作者:** Ziyuan Xia `[一作]` (Zhejiang University), Sida Peng `[通讯]` (Zhejiang University)

**通讯引用:** 41389 | [OpenAlex ID](https://openalex.org/A5001436451)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `67630363-6be0-4f51-ab05-7198250671a5` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本工作构建了一套基于3D高斯溅射（3DGS）与可驱动高精度高斯人形体的嵌入式导航仿真器Habitat‑GS。

**💡 创新点**

其创新点在于：①将3DGS实时渲染无缝集成进Habitat生态，显著提升视觉逼真度；②开发可驱动的高斯人形体模块，实现视觉与导航冲突检测的协同；③通过混合域训练策略实现跨域通用性。

**🔧 技术方法**

主要技术包括CUDA‑OpenGL 零拷贝交互、CUDA加速的线性混合蒙皮（LBS）、基于GAMMA的运动生成、3DGS可变尺度渲染与代理胶囊动态NavMesh阻挡。

**📊 数据集**

使用数据集：InteriorGS与自建GS场景（120场景），Habitat‑Matterport3D（HM3D，100训练+20测试），以及AnimatableGaussians与GAMMA生成的人形轨迹。

**📈 对比分析**

实验采用VLM（Gemini 3.0 Pro）评估渲染质量，并在PointNav任务中比较单域、混合域训练；结果显示混合域（50% Mesh+50% GS）在Mesh和GS测试集上均能获得最高成功率和SPL，同时高斯人形体训练可将碰撞率与个人空间侵入率显著降低，性能保持实时FPS并可扩展到数百万高斯。

**⚠️ 局限性**

局限性在于：3DGS缺乏严格的几何连通性与物理刚体属性，导致只能实现导航级别的碰撞避免，无法支持精细的抓取或推拉等操作；此外，仍需进一步集成物理引擎以实现更复杂的交互任务。

---

## 364. Efficient Semantic Image Communication for Traffic Monitoring at the Edge

**arXiv ID:** 2604.12622 | [PDF](https://arxiv.org/pdf/2604.12622v1)

**作者:** Damir Assylbek `[一作]` (Nazarbayev University), Dimitrios Zorbas `[通讯]` (Nazarbayev University)

**通讯引用:** 1774 | [OpenAlex ID](https://openalex.org/A5019199157)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了两种用于交通监控的语义图像通信方案——MMSD（多模态语义分解）和SAMR（语义感知掩蔽重建），实现边缘设备轻量处理、数据压缩、服务器端生成式重建的异步架构。

**💡 创新点**

创新点包括：①MMSD将分割图、Canny边缘图与文本描述三种语义模态压缩传输，并用多模态控制的扩散模型重建；②SAMR在JPEG编码前基于语义重要性进行Patch级掩蔽，后端用生成式修复，大幅降低码率同时保持高质量；③两种方案可组合，提供不同压缩/质量/隐私的取舍；④首次在语义通信中引入边缘图作为结构引导。

**🔧 技术方法**

技术手段包括：SegFormer语义分割；Canny边缘检测；BLIP/Gemma文本生成；Stable Diffusion v1.5 + ControlNet（分割+边缘）进行多模态条件生成；JPEG/WebP编码；Patch级Bernoulli掩蔽；卷积式inpainting autoencoder；VLM Gemini 27B评测。

**📊 数据集**

实验数据集：Cityscapes（500测试/3000训练）和Sochor等交通监控视频（500测试/3000训练）。

**📈 对比分析**

评价方式：与SPIC、JPEG、SQ‑GAN进行压缩率、PSNR、MS‑SSIM、LPIPS、VLM语义保持率以及人类主观对比。MMSD实现约99%数据压缩，位置/数量保持率人类/VLM均>60%，优于SPIC；SAMR在相同比特率下PSNR/SSIM高于JPEG、SQ‑GAN，语义保持率接近100%，且在0.08‑0.59 BPP范围内可调节压缩/质量。

**⚠️ 局限性**

局限性：MMSD仅重建语义级别细节，无法恢复像素细节；SAMR最低码率受JPEG压缩限制，无法突破约0.06 BPP；边缘设备上大模型（Caption、ControlNet）占用内存/计算，需进一步轻量化；两种方案在极低质量或光照变化下鲁棒性仍待验证。

---

## 365. LLM-Guided Prompt Evolution for Password Guessing

**arXiv ID:** 2604.12601 | [PDF](https://arxiv.org/pdf/2604.12601v1)

**作者:** Vladimir A. Mazin `[一作]`, Oleg Y. Rogov `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用 OpenEvolve 对 PassLLM 的提示语进行进化优化，以提升密码猜测的破解率。

**💡 创新点**

首次将 MAP‑Elites 质量多样性搜索与 LLM 驱动的提示变异结合，证明提示内容对破解率有显著影响，并在本地小模型和云端大型模型上均实现显著提升。

**🔧 技术方法**

采用 PassLLM（LoRA 微调的 LLM）、OpenEvolve 框架、MAP‑Elites 多岛搜索、LLM 变异、Per‑symbol F‑score 评估等技术。

**📊 数据集**

使用公开的 RockYou 密码数据集（训练/测试分离）作为评估基准。

**📈 对比分析**

通过 100 次迭代与基线进行对比，破解率从 2.02% 提升至 8.48%（≈4.2×），本地 8B 模型在 10 次迭代即可达到 8.28%；ensemble 在字符分布 AUC 上提升 26%。

**⚠️ 局限性**

仅评估 trawling 模式，未覆盖 PII/重用攻击；结果仅在单一 RockYou 分割上验证；高方差与可能的局部最优；需要更大搜索空间和更多岛屿来验证稳定性；对其他密码分布的泛化未知。

---

## 366. On Secure Gradient Coding with Uncoded Groupwise Keys

**arXiv ID:** 2604.12578 | [PDF](https://arxiv.org/pdf/2604.12578v1)

**作者:** Xudong You `[一作]` (Huazhong University of Science and Technology), Giuseppe Caire `[通讯]` (Technische Universitat Berlin)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

提出并实现了一种在分布式机器学习中使用无编码组键（uncoded groupwise keys）的安全梯度编码方案，能够在保证只得到梯度和的同时，抵御信息泄露。

**💡 创新点**

首次将无编码组键约束引入安全梯度编码，证明当键共享组大小S大于数据复制度M时安全成本与非安全方案相同；并给出了在S≤M时仍保持在最优通信代价两倍以内的通用方案。

**🔧 技术方法**

利用线性编码、矩阵设计与可解性分析、信息理论安全证明等技术构建编码和解码矩阵，实现梯度与密钥的线性组合。

**📊 数据集**

论文主要通过理论推导与仿真验证，并未给出具体真实数据集的实验。

**📈 对比分析**

通过与非安全梯度编码的最优通信代价对比，实验显示当S>M时通信代价与最优一致；当S≤M时通信代价始终不超过最优代价的两倍。

**⚠️ 局限性**

局限在于对键生成与同步成本的实际实现未展开讨论；若键共享组S不足以覆盖所有服务器，仍需额外通信开销；实验验证主要为仿真，缺乏真实大规模系统部署的验证。

---

## 367. StructDiff: A Structure-Preserving and Spatially Controllable Diffusion Model for Single-Image Generation

**arXiv ID:** 2604.12575 | [PDF](https://arxiv.org/pdf/2604.12575v1)

**作者:** Yinxi He `[一作]` (Beijing Jiaotong University), Yao Zhao `[通讯]` (Beijing Jiaotong University)

**通讯引用:** 19968 | [OpenAlex ID](https://openalex.org/A5100362745)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了单图像生成框架 StructDiff，能够在保持结构一致性的前提下实现多尺度多样化生成和空间可控编辑

**💡 创新点**

创新点包括：1）自适应感受野模块（ARF）实现单尺度网络的多尺度结构感知；2）3D 位置编码（包含坐标和前景-背景掩码）+傅里叶嵌入，实现可调节的位置、尺度与局部细节；3）基于大型语言模型的评估协议，替代传统无参考指标和人类评测

**🔧 技术方法**

技术主要包括：单尺度 DDPM + UNet 结构（去掉下采样、上采样与注意力模块）、ARF 多分支卷积与注意力融合、3D 位置编码与可学习傅里叶嵌入、前景感知感知损失（VGG + Sobel）、CLIP 引导、参考图像低频注入、低分辨率掩码注入等

**📊 数据集**

使用三个数据集：Places50（经典单图像基准）、Mulmini-N（20 张普通图像）和 Mulmini-L（20 张大物体图像）

**📈 对比分析**

与 SinGAN、SinDDM、SinFusion、SinDiffusion 等传统单图像方法以及基于 DiT 的大模型对比，采用 SIFID、MUSIQ、DB‑CNN、LPIPS、LLM 评分、PSNR/SSIM 等指标。实验表明 StructDiff 在结构保持、视觉质量和空间可控性上均优于对比方法，尤其在大物体图像上的结构一致性得分最高

**⚠️ 局限性**

局限性：在复杂背景或遮挡场景下前景-背景分离不佳，无法合理补全被遮挡的部分；依赖单图像学习，缺乏语义层面的泛化能力

---

## 368. Evolution-Inspired Sample Competition for Deep Neural Network Optimization

**arXiv ID:** 2604.12568 | [PDF](https://arxiv.org/pdf/2604.12568v1)

**作者:** Ying Zheng `[一作]` (Hong Kong Polytechnic University), Lap-Pui Chau `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 5959 | [OpenAlex ID](https://openalex.org/A5044722301)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于自然选择的深度网络训练方法，利用样本竞争动态调节样本损失权重，提升模型优化效果。

**💡 创新点**

创新点在于通过拼接图像组并在单次推理中计算每个样本的竞争得分，构建可调节的竞争机制，打破传统统一样本权重的限制。

**🔧 技术方法**

技术包括图像拼接与缩放、模型前向推理、竞争得分（NS）计算、样本权重调节（NS‑WS/NS‑LF）以及对多种网络架构的无缝集成。

**📊 数据集**

在12个公开数据集上验证，涵盖图像分类（CIFAR‑10/100、ImageNet‑1K）、情感识别（Twitter、Flickr、Instagram、FI、EmoSet）、源无监督域适配（Office‑Home）以及长尾分类（CIFAR‑LT‑10/100）和文本分类（IMDB、AG News）。

**📈 对比分析**

与12种损失/采样基线、多种网络骨干比较，NS‑WS/NS‑LF 在大多数任务中均提升 0.3%–2.2% 的准确率，尤其在长尾和域适配任务中表现突出。

**⚠️ 局限性**

局限性包括主要针对计算机视觉任务，文本验证仅为初步，未深入探讨多模态、视频或小样本学习等更复杂场景的适用性。

---

## 369. Calibration-Aware Policy Optimization for Reasoning LLMs

**arXiv ID:** 2604.12632 | [PDF](https://arxiv.org/pdf/2604.12632v1)

**作者:** Ziqi Wang `[一作]` (National Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Chinese Academy of Sciences), Junge Zhang `[通讯]` (National Key Laboratory of Cognition and Decision Intelligence for Complex Systems, Chinese Academy of Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对大型语言模型在数学推理任务中因GRPO等强化学习算法导致的过度自信与校准失效问题，提出Calibration-Aware Policy Optimization（CAPO）方法，利用一致的Logistic AUC近似来进行不确定性感知的优势估计，并引入基于参考模型的噪声屏蔽机制实现训练稳定；

**💡 创新点**

创新点在于用一致的AUC对数损失替代GRPO的奖励仅优势估计，从而将优化目标与相对校准对齐，同时通过噪声屏蔽进一步消除误判梯度，确保同时提升校准与准确率；

**🔧 技术方法**

采用逻辑AUC对数损失、PPO式策略优化、参考模型PPL噪声屏蔽、以及KL正则化的组合技术；

**📊 数据集**

在Qwen2.5-Math-1.5B和7B模型上，使用DeepScaler数据集进行训练，评估于六个数学推理基准（AIME 2024/2025、MATH 500、AMC 2023、Minerva、OlympiadBench）；

**📈 对比分析**

与GRPO、GSPO、CoDaPO、CDE、SimKO等基线相比，CAPO在AUC平均值提升约15%–25%且准确率保持甚至提升，精度‑覆盖曲线显著优于对手，并在推理时缩放任务中提升5%准确率；

**⚠️ 局限性**

仅在数学推理任务上验证，未测试逻辑谜题、常识推理或开放域问答等其他推理任务，需进一步评估通用性。

---

## 370. Pricing-Driven Resource Allocation in the Computing Continuum

**arXiv ID:** 2604.12642 | [PDF](https://arxiv.org/pdf/2604.12642v1)

**作者:** Alejandro García-Fernández `[一作]` (Universidad de Sevilla), Schahram Dustdar `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 37708 | [OpenAlex ID](https://openalex.org/A5004847496)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究并验证了将 SaaS 价格模型（iPricing）用作计算连续体中资源分配配置空间的通用抽象，提出基于 PRIME 的工作流实现成本最优部署。

**💡 创新点**

创新在于将价格结构作为多供应商、互操作性限制的配置空间表示，避免了专用建模语言；并提供完整的实验数据集与可复现实验包。

**🔧 技术方法**

使用 iPricing、PRIME 优化引擎、Gecode/Choco 约束求解器，以及 Python 生成合成拓扑和需求。

**📊 数据集**

基于澳大利亚墨尔本地区的 Edge User Allocation (EUA) 真实节点位置信息，合成生成 9,600 个拓扑+需求场景。

**📈 对比分析**

通过在 9,600 个测试用例上测量求解时间和可行性，结果显示对需求规模无显著影响、对节点数线性增长；最多 200 节点可在 3 秒内求解。

**⚠️ 局限性**

实验仅覆盖单一地理区域、合成需求与拓扑、使用距离代替真实延迟，可能影响对更大规模、真实工作负载的泛化。

---

## 371. Neural Dynamic GI: Random-Access Neural Compression for Temporal Lightmaps in Dynamic Lighting Environments

**arXiv ID:** 2604.12625 | [PDF](https://arxiv.org/pdf/2604.12625v1)

**作者:** Jianhui Wu `[一作]` (University of Science and Technology of China), Chao Li `[通讯]` (Zhejiang University)

**通讯引用:** 38864 | [OpenAlex ID](https://openalex.org/A5100323172)

**关键词:** `8963991b-619b-4c55-be0c-2d0b5f401564` `fede83ac-7505-405f-ab37-e7284695c47f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种Neural Dynamic GI (NDGI) 框架，用神经网络压缩时间序列光照贴图，并支持实时解压和随机访问。

**💡 创新点**

采用混合特征图结构与 BC 压缩仿真，结合虚拟纹理实现低存储、低内存、高质量动态全局光照压缩。

**🔧 技术方法**

使用多维特征图、轻量 MLP 解码器、BC7 纹理压缩仿真、虚拟纹理(VT) 与 GPU Compute Shader、PyTorch 训练等技术。

**📊 数据集**

构建并公开多场景时间变化光照贴图数据集，包括 FarmLand、Room、City、Yard 等。

**📈 对比分析**

与传统 GPU 纹理压缩（BC6H、BC7、ASTC）、神经压缩 NTC、预计算辐射转移 PRT 进行对比；在相同 BPP 下 NDGI 在 PSNR/SSIM/LPIPS 等指标上优于对手，解压延迟低于 NTC，压缩率最高。

**⚠️ 局限性**

对突发光照变化存在短时延迟，PyTorch 训练速度慢，当前仅支持单帧解压；需改进异步调度与专用训练框架。

---

## 372. SOAR: Self-Correction for Optimal Alignment and Refinement in Diffusion Models

**arXiv ID:** 2604.12617 | [PDF](https://arxiv.org/pdf/2604.12617v1)

**作者:** You Qin `[一作]`, Chunyu Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为SOAR（Self-Correction for Optimal Alignment and Refinement）的后训练方法，旨在解决扩散模型在推理过程中出现的曝光偏差问题，通过构建模型自身的偏离轨迹状态并进行监督来进行修正。

**💡 创新点**

SOAR方法通过在模型推理过程中直接修正轨迹偏差，避免了传统方法中等待终端奖励信号的缺陷，提供了密集的逐步监督，消除了信用分配问题。

**🔧 技术方法**

使用了基于流匹配的扩散模型和单步ODE（常微分方程）回滚技术，结合了密集的、无奖励模型的监督信号。

**📊 数据集**

在286,119对图像-文本数据集上进行训练，数据集质量至关重要，以确保监督信号的准确性。

**📈 对比分析**

与标准的监督微调（SFT）方法进行比较，SOAR在多个指标上均表现出显著提升，例如GenEval从0.70提高到0.78，OCR从0.64提高到0.67，同时在模型基础的偏好评分上也有所提高。

**⚠️ 局限性**

SOAR方法的局限性在于尚未评估其对生成多样性的影响，可能会导致输出分布的收窄，未来需要进一步研究这一权衡。

---

## 373. Machine Learning-Based Real-Time Detection of Compensatory Trunk Movements Using Trunk-Wrist Inertial Measurement Units

**arXiv ID:** 2604.12591 | [PDF](https://arxiv.org/pdf/2604.12591v1)

**作者:** Jannis Gabler `[一作]` (ETH Zurich), Dane Donegan `[通讯]` (ETH Zurich)

**通讯引用:** 232 | [OpenAlex ID](https://openalex.org/A5090896709)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出并验证了一种仅使用手腕和躯干两块惯性测量单元（IMU）实现实时检测中风康复患者补偿性躯干运动（CTM）的机器学习框架。

**💡 创新点**

创新点在于：①将最少的传感器布置（手腕+躯干）与极低的硬件成本相结合；②采用基于窗口的特征提取和XGBoost分类器实现毫秒级的推理延迟；③通过SHAP解释性分析揭示了躯干动力学与手腕-躯干协调特征对CTM判别的主导作用。

**🔧 技术方法**

技术手段包括：双IMU同步采样（120 Hz）、姿态滤波（Versatile Quaternion Filter）、特征工程（统计量、相关性、平滑度）、极端梯度提升分类器、留一被试交叉验证以及实时管道实现。

**📊 数据集**

数据集：10名健康受试者在三种模拟中风约束（无约束、肘部支架、阻力带）下完成38项日常活动，配合光学运动捕捉和视频标注；以及4名神经疾病患者（中风、脊髓损伤）在实验室和常规康复任务中的IMU记录。

**📈 对比分析**

与光学捕捉参考和单IMU模型对比，双IMU模型在宏观F1上取得0.80 ± 0.07（MCC 0.73 ± 0.08），ROC‑AUC > 0.93，推理时间≈29 ms，显示出与光学系统相当的性能且实时可行。

**⚠️ 局限性**

局限性包括：①仅在少量健康受试者上训练，缺乏对真实中风多样化运动模式的直接学习；②标签仅为二分类（补偿/非补偿），未细分补偿严重度；③在患者数据上的阈值敏感性高、召回率低，提示需进一步校准和更大临床样本验证。

---

## 374. Cross-Modal Knowledge Distillation for PET-Free Amyloid-Beta Detection from MRI

**arXiv ID:** 2604.12574 | [PDF](https://arxiv.org/pdf/2604.12574v1)

**作者:** Francesco Chiumento `[一作]` (Dublin City University), Mingming Liu `[通讯]` (Dublin City University)

**通讯引用:** 4898 | [OpenAlex ID](https://openalex.org/A5100448075)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a6cb313d-240c-4723-a372-3ba1f39b9afc` `edb9d762-f411-4838-a852-f2d638b018db` `8d10c613-917e-4880-9716-17789f50e119` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e15e3743-5ee0-4d5f-813d-d146868082fc` `dc6c6f4a-9d29-4fb8-b59a-f6c271315b9b`

**🎯 论文内容**

使用 PET 引导的知识蒸馏框架，训练 BiomedCLIP 作为教师模型通过 PET‑MRI 对齐学习特征，再将知识迁移至仅使用 MRI 的学生模型，实现 PET‑free 的 Aβ 阳性预测。

**💡 创新点**

创新点包括：① 利用 Centiloid 量化的 PET 信息进行跨模态对齐与 triplet 采样；② 采用跨模态注意力与多头注意力的双阶段蒸馏（特征匹配 + 温度调节的 logit 蒸馏）；③ 在多种常规 MRI 对比度（T1w、T2w、FLAIR、T2*）上进行评估并实现跨数据集（OASIS‑3 ↔ ADNI）的迁移；④ 通过高分辨率 saliency 显示模型关注的解剖区域，提升临床可解释性。

**🔧 技术方法**

技术手段包括：BiomedCLIP ViT + LoRA 微调；跨模态交叉注意力和自注意力；Centiloid 相关的在线负样本挖掘；triplet 损失、margin‑focal 损失、特征 L2 对齐、温度标度的二元交叉熵蒸馏；N4 纠正、HD‑BET 提取、ANTs MNI 注册等预处理。

**📊 数据集**

数据集：OASIS‑3（1,379 受试者）和 ADNI（5,111 受试者），分别配备 PET（PiB/AV‑45）和多模 MRI；在四种 MRI 对比度的单模与双模组合上进行训练和测试，样本对齐严格到 365 天内。

**📈 对比分析**

与无蒸馏基线（T1w AUC 0.61）相比，教师蒸馏后单模 T1w AUC 提升至 0.73，T1w+T2w 进一步升至 0.74；在 ADNI 上的 AUC 亦保持 0.66。跨数据集蒸馏在 OASIS‑3 上达到 0.73，表明良好的迁移性能。AUC、F1、NPV 等指标均优于现有 MRI‑only 方法，且不需要额外临床变量。

**⚠️ 局限性**

局限性：① 仍需 PET 作为训练监督，无法完全脱离 PET 数据；② 在较小或不平衡样本（如 FLAIR、T2*）上性能有限；③ 仅评估常规结构 MRI，未覆盖扩展 MRI（扩散、灌注）或其他扫描仪；④ 对于不同放射源、扫描协议的泛化性尚待进一步验证。

---

## 375. Scalable Trajectory Generation for Whole-Body Mobile Manipulation

**arXiv ID:** 2604.12565 | [PDF](https://arxiv.org/pdf/2604.12565v1)

**作者:** Yida Niu `[一作]` (Peking University), Yixin Zhu `[通讯]` (Peking University)

**通讯引用:** 4118 | [OpenAlex ID](https://openalex.org/A5051255725)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

通过GPU加速的全身运动规划框架，统一了移动基座、机械臂与目标物体的 kinematics，生成了 500k 条物理有效的完整轨迹；

**💡 创新点**

提出了基于 akr 的单链统一建模与 GPU 并行轨迹优化的组合，显著提升了数据生成吞吐量（80× CPU 加速）并同时兼顾规模、多样性与精度；

**🔧 技术方法**

使用 GPU 并行化的 akr 运动规划、Spherical 近似碰撞检测、Isaac Sim 渲染、Diffusion Policy（DP3）、Transformer（ACT）等学习算法；

**📊 数据集**

构建了覆盖 330 个真实感场景、多个机器人平台（Summit Franka、TIAGo、R1）以及数十种可变形与刚性物体的内部数据集；

**📈 对比分析**

与现有数据集（如 RT-1、NYU VINN、BC‑Z 等）相比，本框架在规模（50×）、多样性和物理合法性上更优；在多场景与多轨迹密度实验中，DP3 等模型在移动基座任务上从 70% 提升至约 75% 的成功率，表明数据规模对学习效果关键；

**⚠️ 局限性**

局限在于仅支持已知静态场景与 kinematics，无法处理动态人机交互或柔性物体；球形碰撞近似偶尔引入几何误差，导致执行失败。

---

## 376. Robust Graph Isomorphism, Quadratic Assignment and VC Dimension

**arXiv ID:** 2604.12584 | [PDF](https://arxiv.org/pdf/2604.12584v1)

**作者:** Anatole Dahan `[一作]`, Tomáš Novotný `[通讯]`

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了利用图的 VC 维度来实现图编辑距离（GED）与二次分配问题（QAP）的近似算法，并研究了鲁棒图同构问题；

**💡 创新点**

创新点在于引入 VC 维度的概念并证明其对 GED 与 QAP 的近似复杂度具有决定性影响，进一步证明在 VC 维度有限的图上，O(1/ϵ⋅d) 维 Weisfeiler–Leman 算法即可解决 ϵ-鲁棒图同构问题；

**🔧 技术方法**

主要技术包括 VC 维度下的 ε-近似与 ε-网、均匀采样、二次分配问题的线性化、以及 Weisfeiler–Leman 算法与双偶极游戏的理论分析；

**📊 数据集**

论文未使用传统数据集，实验与分析均基于理论构造与证明；

**📈 对比分析**

相较于先前的 O(n^{O(log n/ϵ^2)}) 近似算法，本文在 VC 维度为 d 的图上将运行时间降至 n^{O(d/ϵ^2)}，而鲁棒图同构问题可在 n^{O((1/ϵ)d)} 或 n^{O((1/ϵ)log n)} 时间内解决；

**⚠️ 局限性**

主要限制在于：对 ϵ 的依赖仍为 1/ϵ，且对任意固定 k 无法保证 ϵ-鲁棒图同构问题的多项式时间解决；此外，低维 WL 的下界表明 ϵ 与 k 之间存在根本的限制。

---

## 377. Relaxing Anchor-Frame Dominance for Mitigating Hallucinations in Video Large Language Models

**arXiv ID:** 2604.12582 | [PDF](https://arxiv.org/pdf/2604.12582v1)

**作者:** Zijian Liu `[一作]` (University of Electronic Science and Technology of China), Chaoning Zhang `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 2061 | [OpenAlex ID](https://openalex.org/A5057230698)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出一种无训练、层选择的 Decoder‑side Temporal Rebalancing (DTR) 方法，用于 Video‑LLM 推理阶段重新分配时间注意力，降低 anchor‑frame 再现导致的幻觉。

**💡 创新点**

创新点在于首次发现 anchor‑frame 主导是 Video‑LLM 幻觉的根本原因，并提出仅在中后 Decoder 层对视觉注意力 logit 进行全局与缺口补偿的轻量化干预，无需改动视觉编码器或训练额外模型。

**🔧 技术方法**

技术细节包括：基于原始注意力 logit 计算帧级注意力分数，求差距归一化后得到缺口；引入可调的全局调整 α 与缺口补偿 β；将偏置加到选定层的视觉‑token 注意力 logits 前，保留所有帧参与分配。

**📊 数据集**

实验数据集涵盖 EventHallusion、VideoHallucer（两者都评估幻觉）以及 MVBench（通用视频理解）三大基准，使用 8 帧采样与 Greedy 解码。

**📈 对比分析**

在 LLaVA、Video‑LLaVA、Qwen‑V2、Qwen‑V3 等多种 Video‑LLM 上，DTR 与 TCD、VCD、DINO‑HEAL 等无训练对比，显著提升幻觉准确率（EventHallusion +4‑5%，VideoHallucer +2‑3%），同时保持或略提升 MVBench 结果，推理速度仅略高于基线，内存占用基本不变。

**⚠️ 局限性**

局限性包括：仅针对 Decoder‑侧时间平衡，未解决视觉编码层或跨模态先验导致的幻觉；中后层介入区间与 α、β 参数需手动调优；在更大规模或实时场景下的性能与泛化性尚未完全验证。

---

## 378. FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing

**arXiv ID:** 2604.12559 | [PDF](https://arxiv.org/pdf/2604.12559v1)

**作者:** Peng Wang `[一作]` (Chinese Academy of Sciences), Songlin Hu `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 7140 | [OpenAlex ID](https://openalex.org/A5102820325)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种层次化的无结构模型编辑框架FABLE，先在浅层注入细粒度事实，再在深层微调表面文本生成，以实现细粒度事实访问与整体文本召回的双重目标；同时构建了FABLE诊断基准，用细粒度问答对和事实级指标系统评估编辑效果。

**💡 创新点**

创新点在于将Transformer的关键-值存储进一步拆分为两级：细粒度事实锚定层和整体表面生成层，并采用两阶段编辑策略（先注入事实后整合表面），显著提升细粒度事实召回率，并兼顾整体生成质量；此外，提出HR和C_LCS两项事实级评测指标与对应的基准数据集。

**🔧 技术方法**

核心技术包括Transformer层级key‑value分解、两阶段梯度更新（分布式参数更新与残差向量优化）、多层次问答对生成与关键短语提取、以及在浅层/深层分别调优的细粒度与整体整合策略。

**📊 数据集**

使用UnKEBench、AKEW（CounterFact）和AKEW（MQuAKE）三大无结构编辑基准数据集；对每个数据集进一步生成细粒度问答对与关键短语，并构成“-UnKE”、“-CF”和“-MQ”子集。

**📈 对比分析**

与FT‑L、ROME、MEMIT、UnKE、AnyEdit等基线方法在Llama3‑8B和Qwen2.5‑7B模型上进行对比。FABLE在整体评测（Bert‑Score、Rouge‑L）上均超过第二名约1.98%/5.42%，在细粒度指标（HR、C_LCS）上平均提升20.07%/14.80%，并且在保持整体能力方面优于基线。

**⚠️ 局限性**

主要局限在单编辑场景（batch=1）下验证，未考虑批量或顺序编辑导致的参数冲突和灾难性遗忘；在细粒度事实保持下，整体表面生成的泛化略有下降，表明两者仍存在权衡。

---

## 379. Cross-Attentive Multiview Fusion of Vision-Language Embeddings

**arXiv ID:** 2604.12551 | [PDF](https://arxiv.org/pdf/2604.12551v1)

**作者:** Tomas Berriel Martins `[一作]` (University of Zaragoza), Javier Civera `[通讯]` (University of Zaragoza)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出跨注意力多视角融合模型CAMFusion，使用单视角视觉语言描述并通过自监督一致性训练得到更鲁棒的3D实例嵌入。

**💡 创新点**

创新点在于将多视角融合视作语义超分辨率问题，设计了交叉注意力的多视角Transformer，并加入自监督的多视角一致性损失。

**🔧 技术方法**

采用PE ViT‑L/14+TextRegion单视角编码、Transformer自注意+跨注意、Sigmoid对比损失以及自监督一致性损失。

**📊 数据集**

训练使用ScanNet++，在Replica、ScanNetv2和3RScan等数据集上评估。

**📈 对比分析**

与平均池化、OVO、OpenScene、OV3R等基线比较，CAMFusion在3D语义分割和实例分类的mAP提升超过10pp，尤其在尾部类别表现突出。

**⚠️ 局限性**

局限性包括对上游分割mask高度依赖、视角选择策略简单、语言监督仅限类别名称，缺乏更丰富的查询支持。

---

## 380. MODIX: A Training-Free Multimodal Information-Driven Positional Index Scaling for Vision-Language Models

**arXiv ID:** 2604.12537 | [PDF](https://arxiv.org/pdf/2604.12537v1)

**作者:** Ruoxiang Huang `[一作]` (Peking University), Zhen Yuan `[通讯]` (Peking University)

**通讯引用:** 18758 | [OpenAlex ID](https://openalex.org/A5100444868)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 MODIX，一个训练无关、推理时自适应的多模态位置编码调整框架。

**💡 创新点**

将信息理论（协方差熵估计与跨模态交互强度）用于动态调节视觉模态步幅，实现位置编码的自适应细粒度分配。

**🔧 技术方法**

使用信息量理论、RoPE 位置编码、几何平均融合、推理时索引重构等技术。

**📊 数据集**

在 ScienceQA、RealWorldQA、DocVQA、ChartQA、AI2D、BLINK 以及 Video‑MME 等数据集上进行实验。

**📈 对比分析**

与基线以及 V2PE、CircleRoPE、MHRoPE 等多模态位置编码方法对比，MODIX 在大多数任务中平均提升 2–4%，在 ScienceQA 等任务可提升至 6% 以上。

**⚠️ 局限性**

仅在模态层面设定单一步幅，缺乏 token 级细化；对极大规模模型（70B+）与非 RoPE 编码的适用性尚未验证；训练时集成效果有限。

---

## 381. GeoAlign: Geometric Feature Realignment for MLLM Spatial Reasoning

**arXiv ID:** 2604.12630 | [PDF](https://arxiv.org/pdf/2604.12630v1)

**作者:** Zhaochen Liu `[一作]` (Peking University), Tingting Jiang `[通讯]` (Peking University)

**通讯引用:** 17970 | [OpenAlex ID](https://openalex.org/A5101911266)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

通过动态聚合多层几何特征并使用视觉令牌作为查询，实现多模大模型的空间推理增强。

**💡 创新点**

发现传统单层几何特征提取导致任务错位偏差，并提出GeoAlign框架：构建层次化几何特征库，利用内容感知查询和Top‑K稀疏路由自适应挑选合适层级特征。

**🔧 技术方法**

使用多层几何特征池、两层MLP投影、内容感知路由网络、Top‑K稀疏软硬分配以及残差注入方式将聚合后的几何特征融合到视觉流。

**📊 数据集**

在VSI‑Bench（多任务空间推理）和ScanQA、SQA3D（3D场景理解）三个数据集上进行评估。

**📈 对比分析**

与多种专用、通用、空间增强模型对比，4B GeoAlign在VSI‑Bench平均得分71.4，明显优于更大模型；在ScanQA/SQA3D上与VLM‑3R接近或略优，体现高参数效率与竞争性能。

**⚠️ 局限性**

依赖冻结的3D基础模型，特征可能与M‑LLM空间推理需求不完全对齐；动态层路由需要提取多层特征，增加显存与计算开销。

---

## 382. GraphTide: Augmenting Knowledge-Intensive Text with Progressive Nested Graph

**arXiv ID:** 2604.12624 | [PDF](https://arxiv.org/pdf/2604.12624v1)

**作者:** Xin Qian `[一作]` (Zhejiang University), Yingcai Wu `[通讯]` (Zhejiang University)

**通讯引用:** 6234 | [OpenAlex ID](https://openalex.org/A5073986937)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种文本增强技术，通过逐句渐进式构建嵌套实体-关系图并配合动画，帮助读者理解知识密集文本。

**💡 创新点**

创新点在于：①采用按需分解语义单元的嵌套图表示，避免单层图过度简化或碎片化；②设计结构感知的力导向布局，使图与文本的空间时间同步；③实现动画驱动的递进渲染，维持阅读节奏并降低认知负荷。

**🔧 技术方法**

使用了：GPT‑4o进行实体与关系抽取与自校正；层级+力导向布局算法；动画化的渐进渲染策略；双向文本‑图高亮、实体频率列表等交互。

**📊 数据集**

评估数据集包括16段约200词文本（8段维基百科+8段LLM生成）用于实体抽取评估；用户研究使用3篇中等长度（6–8句）学科文本（政策、气候、医药）及公开问题集。

**📈 对比分析**

通过within‑subject实验与Graphologue（单层图）和Static Display（静态嵌套图）两基线对比；阅读时间相近，回答时间与质量显著优于两者（p<0.05），表明系统显著提升理解效率与质量。

**⚠️ 局限性**

局限性：缺乏对话论结构（因果、对比等）显式表示；交互功能有限（如书签、子图摘要）；动画风格对认知影响未充分探索；对更长文本与多主题深层叙事的支持尚待扩展。

---

## 383. KumoRFM-2: Scaling Foundation Models for Relational Learning

**arXiv ID:** 2604.12596 | [PDF](https://arxiv.org/pdf/2604.12596v1)

**作者:** Valter Hudovernik `[一作]`, Matthias Fey `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出并实现了 KumoRFM，一种预训练的关系型基础模型，能够直接在多表关系数据库上做少样本（few‑shot）预测，无需手工特征工程或离线训练，支持从 SQL 数据库实时构建图并进行推理。

**💡 创新点**

创新点包括：①端到端的层次化注意力架构——先在表内进行列/行注意，再在键级和跨样本级别聚合，能够在不同尺度上捕获关系特征；②任务信息在预训练阶段即早期注入，提升对噪声的鲁棒性；③自动上下文生成与滑动目标机制，让模型在极小的上下文（≤0.2% 数据）下即可完成多种预测；④可扩展到 500B+ 行的数据库，突破传统内存受限的瓶颈。

**🔧 技术方法**

技术手段：关系深度学习（Relational Deep Learning）+ Transformer；层次化注意力（表级、键级、跨样本级）；大规模合成与真实数据预训练；自动上下文生成与预测查询语言（PQL）；可选微调（fine‑tune）实现更高精度。

**📊 数据集**

预训练使用了合成（基于结构因果模型）和真实的多表关系数据库；评估数据集包括 RelBenchV1/V2、SALT（企业资源规划）和 4DBInfer，涵盖 41 个多时序预测任务，跨电子商务、医疗、社交等领域。

**📈 对比分析**

实验与多类基线对比：监督表格模型、关系深度学习模型、LLM、Tabular 和其它 Relational Foundation Models。KumoRFM 在 41 个任务上平均提升约 8%（AUROC/MAE/MRR），在仅使用 0.2% 上下文样本时已超过最强监督模型；fine‑tune 后可再提升约 10 点。

**⚠️ 局限性**

局限性：仍受限于上下文样本规模和子图深度/宽度的手动调参；对极端稀疏或缺失链接的数据库性能会退化；在极大多类任务（近 600 类）下推理速度相对较慢；解释性方法仍不够细粒度，需要进一步研究。

---

## 384. InsightFlow: LLM-Driven Synthesis of Patient Narratives for Mental Health into Causal Models

**arXiv ID:** 2604.12721 | [PDF](https://arxiv.org/pdf/2604.12721v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86`

---

## 385. GF-Score: Certified Class-Conditional Robustness Evaluation with Fairness Guarantees

**arXiv ID:** 2604.12757 | [PDF](https://arxiv.org/pdf/2604.12757v1)

**作者:** Arya Shah `[一作]` (IIT Gandhinagar), Manisha Padala `[通讯]` (IIT Gandhinagar)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出GF-Score框架，按类别分解GREAT Score并衡量鲁棒性不平衡

**💡 创新点**

首次给出攻击无关的自校准温度、四种基于福利经济学的不平衡指标

**🔧 技术方法**

利用GREAT Score的局部可信度、正态化的Gini系数、排名相关性与生成模型样本

**📊 数据集**

在CIFAR-10（17个ℓ2模型）和ImageNet（5个ℓ∞模型）上评估

**📈 对比分析**

与RobustBench/AutoAttack等传统评估比较，攻击无关校准后相关系数在CIFAR-10上达0.871、ImageNet上达1.0，且能揭示类间鲁棒性差异

**⚠️ 局限性**

依赖生成模型质量、假设鲁棒性与清晰准确度单调，且ImageNet实验规模受限

---

## 386. Spatial-Spectral Adaptive Fidelity and Noise Prior Reduction Guided Hyperspectral Image Denoising

**arXiv ID:** 2604.12600 | [PDF](https://arxiv.org/pdf/2604.12600v1)

**作者:** Xuelin Xie `[一作]` (Wuhan University), Long Chen `[通讯]` (Wuhan University)

**通讯引用:** 31396 | [OpenAlex ID](https://openalex.org/A5100336358)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种快速鲁棒混合噪声去噪框架 FRHD，结合噪声先验缩减与空间-光谱自适应保真项，实现对高光谱图像混合噪声的有效去除。

**💡 创新点**

创新点在于：①将高斯、稀疏和结构化噪声拆分为独立先验并通过噪声先验缩减降低参数复杂度；②引入像素级自适应权重张量动态平衡保真与正则；③将代表系数总变分 RCTV 与低秩子空间模型融合，实现光谱低秩与局部平滑的双重约束。

**🔧 技术方法**

采用变分模型、噪声先验分解、像素级自适应权重、代表系数总变分（RCTV）正则化，以及交替方向乘子法（ADMM）进行高效求解。

**📊 数据集**

在 CAVE、PaC、WDC 三大模拟数据集以及真实世界 HYDICE Urban 数据集上进行实验验证。

**📈 对比分析**

与 SDeCNN、WNLRATV、RCTV、FastHyMix、TPTV、CTV-SPCP、FBGND、FallHyDe 以及 HLRTF、Flex-DLD 等基线方法比较，FRHD 在 PSNR、SSIM、SAM 指标上均优于对手，并且在多种混合噪声场景下保持了较低的计算时间。

**⚠️ 局限性**

局限性包括：仍为非凸模型，收敛至局部最优；需要手动调节噪声权重等超参数；对极端或未知噪声分布的自适应性尚有限。

---

## 387. Transferable Expertise for Autonomous Agents via Real-World Case-Based Learning

**arXiv ID:** 2604.12717 | [PDF](https://arxiv.org/pdf/2604.12717v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 388. Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood

**arXiv ID:** 2604.12736 | [PDF](https://arxiv.org/pdf/2604.12736v1)

**作者:** Xingyu Lin `[一作]` (Jilin University), Zhonghou Lv `[通讯]` (Baidu Inc)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种 token‑level 框架 TEPO，利用序列级似然对 token 奖励进行软聚合，并对正优势且熵下降的 token 应用 KL 蒙板，以解决 GRPO 在 chain‑of‑thought 推理中 token‑sparse 奖励导致的训练不稳定。

**💡 创新点**

创新点在于：① 用序列级似然对 token 奖励进行 soft 聚合，提升训练稳定性；② 对正优势且熵下降的 token 进行 KL 蒙板限制，抑制突兀更新；③ 在不增加 critic 的前提下将训练步数缩短约 50%，显著提升数学推理性能。

**🔧 技术方法**

采用无 critic 的 GRPO 变体，基于 PPO 损失、token 级 KL 正则化、序列级似然权重、熵监控与 mask 等技术。

**📊 数据集**

使用 DAPO‑MATH 训练集，在七大数学基准（MATH‑500、AIME24/25、AMC、OMNI‑MATH、OlympiadBench、Minerva）进行评估。

**📈 对比分析**

与 GRPO/DAPO、CLIP‑Cov、KL‑Cov、Entropy‑based、GPG、GSPO 等基线比较；在 Qwen2.5‑7B 上平均准确率从 30.85% 提升至 32.59%（+1.74pp），Qwen3‑14B 提升至 44.02%（+2.51pp），非 Qwen 模型亦获领先，且训练步骤减半。

**⚠️ 局限性**

未阐明 token 约束机制的根本原理，也未区分不同 token 对 CoT 推理的具体影响，需要进一步研究 token 角色与通用框架。

---

## 389. Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning

**arXiv ID:** 2604.12719 | [PDF](https://arxiv.org/pdf/2604.12719v1)

**作者:** Adam T. Müller `[一作]` (Heilbronn University of Applied Sciences), Nicolaj C. Stache `[通讯]` (Heilbronn University of Applied Sciences)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e0540dec-d77f-42db-94ae-d039248f6393` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

在安全关键场景下，将Stochastic Depth（SD）理论化为变分推断框架，并提出Monte Carlo Stochastic Depth（MCSD）方法，随后在YOLOv8x、Faster R‑CNN、RT‑DETR等多任务目标检测模型上进行全面实验验证。

**💡 创新点**

首先提供MCSD的形式化推导，使SD等价于可采样的变分分布；其次首次将该方法在复杂目标检测任务中与MCD、MCDB进行对比，证明其在不显著牺牲精度的前提下能提升不确定性校准与排序。

**🔧 技术方法**

利用变分推断与Monte Carlo采样技术，将SD转化为可采样的残差路径分布；在训练阶段加入L2正则作为KL近似；在推理阶段进行多次随机前向传播并平均结果以估计预测后验。

**📊 数据集**

使用COCO数据集（train/val/test）评估在分布内性能，并通过COCO‑O（风格、天气、手绘等域漂移）检验对抗域迁移的鲁棒性。

**📈 对比分析**

对比方法包括MCD与MCDB；评价指标涵盖mAP、ECE与AUARC。实验表明MCSD在AUARC与ECE上优于MCD，mAP略低但仍保持竞争水平；MCDB在部分架构中表现不稳。

**⚠️ 局限性**

绝对校准仍受模型架构影响；MCSD对残差层放置位置高度敏感；评估依赖检测阈值，缺乏标准化的阈值无关评价方法。

---

## 390. LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety

**arXiv ID:** 2604.12710 | [PDF](https://arxiv.org/pdf/2604.12710v1)

**作者:** Junxiao Yang `[一作]` (Tsinghua University), Minlie Huang `[通讯]` (Tsinghua University)

**通讯引用:** 15966 | [OpenAlex ID](https://openalex.org/A5044042138)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并提出一种跨语言安全对齐框架 LASA，通过在 LLM 的语义瓶颈层进行安全对齐，提升低资源语言下的安全性能。

**💡 创新点**

将安全对齐锚定在语言无关的语义空间（语义瓶颈层），并设计安全语义解释器 SSI，使安全知识从高资源语言自然迁移到低资源语言。

**🔧 技术方法**

采用语义瓶颈识别（基于 Silhouette 分数和 t‑SNE）、安全语义解释器 MLP、条件生成训练（KTO 等偏好优化）以及多语言对齐技术。

**📊 数据集**

使用 PKUSafeRLHF（安全数据）、Ultrafeedback（通用数据）、MultiJail、HarmBench 及其多语言翻译等数据集。

**📈 对比分析**

与 Vanilla SFT、DPO、KTO、ORPO、CPO、MPO 等基线对比，LASA 在 10 种语言上平均攻击成功率从 24.7% 降至 2.8%（LLaMA‑3.1‑8B）或 3‑4%（Qwen 系列），同时保持或略提升通用能力。

**⚠️ 局限性**

局限性包括：仅对显式表达的有害意图有效，隐式或需多步推理的内容表现不足；依赖 GPT‑4o 自动评估；安全语义解释器受训练数据多样性限制；不处理安全完成场景。

---

## 391. Risk-Calibrated Learning: Minimizing Fatal Errors in Medical AI

**arXiv ID:** 2604.12693 | [PDF](https://arxiv.org/pdf/2604.12693v1)

**作者:** Abolfazl Mohammadi-Seif `[一作]` (Universitat Pompeu Fabra), Ricardo Baeza-Yates `[通讯]` (Universitat Pompeu Fabra)

**通讯引用:** 33180 | [OpenAlex ID](https://openalex.org/A5076204770)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了一种风险校准学习（Risk‑Calibrated Learning）方法，通过在损失函数中嵌入临床严重度矩阵，显著降低医学图像分类中的致命错误率。

**💡 创新点**

创新点在于将医学安全需求转化为非对称的损失矩阵，区别视觉歧义与语义不一致的错误，从而在不改动模型架构的前提下实现安全性提升。

**🔧 技术方法**

技术上使用了风险校准损失函数（RCL），结合交叉熵并乘以严重度矩阵，实验中采用了ResNet‑50和ViT‑B16两种主流网络。

**📊 数据集**

使用了四个医学图像数据集：脑肿瘤MRI、ISIC 2018皮肤病变、BreaKHis乳腺组织病理和SICAPv2前列腺病理。

**📈 对比分析**

与交叉熵、加权交叉熵、Focal Loss和Label Smoothing等基线对比，RCL在所有数据集上将致命错误率（CER）降低20%~92%，在保持或提升F1宏观和准确率的同时显著提升安全性。

**⚠️ 局限性**

局限性包括需预先设定严苛的二元严重度矩阵，可能忽略疾病风险连续性；同时对致命错误的压制会导致假阳性率上升，增加临床工作负担。

---

## 392. Brain-DiT: A Universal Multi-state fMRI Foundation Model with Metadata-Conditioned Pretraining

**arXiv ID:** 2604.12683 | [PDF](https://arxiv.org/pdf/2604.12683v1)

**作者:** Junfeng Xia `[一作]` (Southern University of Science and Technology), Quanying Liu `[通讯]` (Southern University of Science and Technology)

**通讯引用:** 3976 | [OpenAlex ID](https://openalex.org/A5078854583)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计并训练了Brain‑DiT，一种基于Diffusion Transformer的多状态fMRI基础模型，利用元数据条件预训练，生成多尺度表征并用于下游任务。

**💡 创新点**

创新点包括：①首次在24个多状态fMRI数据集（休息、任务、自然、疾病、睡眠）上联合预训练；②将临床/人口元数据作为条件输入，实现内在神经动态与群体变异的解耦；③采用扩散时间步与Transformer层的多尺度特征聚合，提升表征多样性。

**🔧 技术方法**

技术手段：DDPM扩散模型、Diffusion Transformer (DiT) + v‑prediction、AdaLN‑Zero条件注入、CFG式条件抛弃、查询注意力聚合、多尺度特征提取与聚合。

**📊 数据集**

使用349,898份fMRI样本，来源于24个数据集，涵盖休息、任务、自然情境、疾病、睡眠等多种脑状态；在ID（22个数据集）与OOD（NKI、ADHD‑200、PPMI）集上进行评估。

**📈 对比分析**

与BrainLM、Brain‑JEPA、BrainMass、SlimBrain等MAE/JEPA/对比学习基线相比，Brain‑DiT在ID和OOD分类、回归任务上准确率提升约10‑15%，MSE降低；冻结骨干后仍保持强劲性能；生成式评估显示合成fMRI的功能连接与真实样本高度一致。

**⚠️ 局限性**

局限性：训练时对显存与计算资源要求高，难以直接扩展到更长序列或更高分辨率；元数据条件仍相对单一，缺乏更丰富的临床或行为信息；模型主要在脑区级别窗口，未充分探索体素级细节与解剖空间的精细表征。

---

## 393. Broadening the Applicability of Conditional Syntax Splitting for Reasoning from Conditional Belief Bases

**arXiv ID:** 2604.12660 | [PDF](https://arxiv.org/pdf/2604.12660v1)

**作者:** Lars-Phillip Spiegel `[一作]`, Christoph Beierle `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

**🎯 论文内容**

论文提出了对条件语法拆分（Conditional Syntax Splitting）的新定义——泛化安全拆分（Generalized Safe Conditional Syntax Splitting），允许子基底共享非平凡条件式，从而显著扩展可拆分知识库的范围。

**💡 创新点**

创新点在于：①引入了“泛化安全”条件，克服了安全拆分仅允许自满足式（self‑fulfilling）交集的局限；②提出“真拆分”（Genuine Splitting）概念，筛选出对归纳推理真正有益的拆分；③根据泛化拆分重新制定了条件相关性与条件独立性后置条件（Generalized Conditional Relevance/Independence），并证明了该后置条件比传统条件语法拆分更严格。

**🔧 技术方法**

主要技术包括：条件语义的秩函数（Ordinal Conditional Functions, OCF）、置信度分布（c‑representations）、基于约束求解的核心c‑表示（c‑core closure）、系统Z、系统W以及列举的各种归纳推理运算符。作者通过逻辑推理与归纳推理运算符的性质证明，展示了哪些运算符满足泛化条件语法拆分后置条件。

**📊 数据集**

本文属于理论研究，未使用实验数据集，全部以形式化证明和例子说明其理论有效性。

**📈 对比分析**

比较方法为形式化后置条件的满足性（是否满足 CSYNSPLITG）。作者证明：lexicographic inference、System W、c‑core closure、c‑inference、某些单一 c‑representation 选择策略均满足该后置条件；而 System Z 仅满足传统后置条件，但不满足泛化后置条件。性能指标以逻辑推理的可行性和严格性为衡量，表明泛化拆分后置条件更严格、覆盖面更广。

**⚠️ 局限性**

局限性：泛化安全拆分要求更高，导致某些已有归纳推理运算符（如 System Z）无法满足；此外，真拆分的判定复杂度尚未讨论；未来工作需探讨与语义记忆、遗忘等技术的结合以及对弱一致性知识库的适用性。

---

## 394. NaviRAG: Towards Active Knowledge Navigation for Retrieval-Augmented Generation

**arXiv ID:** 2604.12766 | [PDF](https://arxiv.org/pdf/2604.12766v1)

**作者:** Jihao Dai `[一作]` (Tsinghua University), Maosong Sun `[通讯]` (Tsinghua University)

**通讯引用:** 38219 | [OpenAlex ID](https://openalex.org/A5046448314)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了 NaviRAG，一种通过构建层级知识树并采用两阶段粗到细检索导航的检索增强生成框架，用以提升复杂长链推理问答的性能。

**💡 创新点**

创新点在于将检索过程转化为主动知识导航：首先用向量检索定位语义子空间，然后在该子空间内以树形结构进行逐层精细检索，实现多粒度、可导航的证据获取。

**🔧 技术方法**

技术包括 LLM（如 Qwen2.5‑72B）生成摘要与结构、bge‑m3 向量检索、层级知识树构建、两阶段检索导航以及可选的记忆模块，最终结合 RAG 生成模型完成答案。

**📊 数据集**

实验使用了单文档长链推理基准 NarrativeQA、LooGLE 以及 LongBench‑v2，对不同模型规模（Qwen3‑14B、Qwen3‑32B、LLaMA3.3‑70B 等）进行评估。

**📈 对比分析**

与传统 flat RAG、GraphRAG、LightRAG、HippoRAG2 等基线比较，NaviRAG 在所有基准上均显著提升（例如在 LLaMA3.3‑70B 上 F1 提升约5%，检索 Recall 提升 5%），在多跳推理任务中的优势尤为突出。

**⚠️ 局限性**

局限性在于仅针对单文档、单源约束情境的复杂推理，缺乏对多源跨文档查询的适配；记忆模块仍是辅助信号，易导致语义漂移，且未实现目标导向的检索规划。

---

## 395. A Dataset and Evaluation for Complex 4D Markerless Human Motion Capture

**arXiv ID:** 2604.12765 | [PDF](https://arxiv.org/pdf/2604.12765v1)

**作者:** Yeeun Park `[一作]` (Texas A&M University), Suryansh Kumar `[通讯]` (Texas A&M University)

**通讯引用:** 745 | [OpenAlex ID](https://openalex.org/A5002526108)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出并发布了HUM4D数据集，集成同步多视角RGB‑D影像与专业标记捕捉（Vicon）获得的高精度三维姿态，用于研究复杂多人物交互的无标记人类运动捕捉；

**💡 创新点**

创新点在于：①提供同时拥有RGB‑D和专业标记基准的多视角数据；②涵盖多达三人交互、严重遮挡、身份切换、深度变化等真实且挑战性高的场景，填补了现有单人或弱交互数据集的空白；

**🔧 技术方法**

技术包括：多视角RGB‑D采集（Intel RealSense D455）、Vicon标记捕捉、高精度相机标定与硬件同步、SMPL/SMPL‑X的骨骼重定向与参数导出、帧级时间对齐与降采样；

**📊 数据集**

使用的数据集是自研的HUM4D，且与Human3.6M、CMU Panoptic、3DPW、HUMAN4D等公开数据集进行对照；

**📈 对比分析**

评估采用四种主流无标记模型（SPIN、PARE、HMR2.0、PersPose），在HUM4D上均出现PA‑MPJPE从约39–82mm（在3DPW上）升至约151–180mm，表明在多人物交互与遮挡强度高的场景下，现有模型泛化能力不足；

**⚠️ 局限性**

局限性：模型均为单帧、无交互/多视角/时间约束；对遮挡和身份交换不鲁棒；未能充分利用多视角深度信息和连续帧的时序一致性，导致在复杂交互中性能显著下降。

---

## 396. Generating Effective CoT Traces for Mitigating Causal Hallucination

**arXiv ID:** 2604.12748 | [PDF](https://arxiv.org/pdf/2604.12748v1)

**作者:** Yiheng Zhao `[一作]` (Concordia University), Jun Yan `[通讯]` (Concordia University)

**通讯引用:** 5614 | [OpenAlex ID](https://openalex.org/A5061710167)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在小参数LLM的事件因果识别任务中，研究者提出了Causal Hallucination Rate (CHR) 指标，并设计了一套基于CoT追踪的微调流程，以显著降低因果幻觉并提升模型准确率。

**💡 创新点**

创新点包括：①引入CHR量化因果幻觉；②发现CoT长度与语义解释对小模型尤为重要；③开发了长CoT生成与重写相结合的管道，保持分布差距低且不提升困惑度；④在跨数据集、跨难度和鲁棒性测试中展示了良好泛化。

**🔧 技术方法**

采用了Chain-of-Thought生成、分布匹配重写、LoRA微调、低秩自适应、SFTTrainer/TRL框架以及CHR指标等技术。

**📊 数据集**

使用了EventStoryLine、Causal-TimeBank、MAVEN-ERE三大数据集，分别涵盖句子级和文档级事件对。

**📈 对比分析**

通过与GPT-4、Llama3.1-8B、Qwen3-30B-A3B以及Dr.ECI、MuTQA、MRBalance等基线对比，使用CHR和平均准确率(mAcc)评估；小模型微调后CHR从约80%降至≈6%，mAcc提升约14%~18%。

**⚠️ 局限性**

局限性在于：①管道主要针对小模型，尚未验证对大模型的适用性；②仅评估ECI任务，未探究其他因果推理场景；③鲁棒性测试较为简单，未覆盖更复杂的对抗干扰。

---

## 397. Evaluating Differential Privacy Against Membership Inference in Federated Learning: Insights from the NIST Genomics Red Team Challenge

**arXiv ID:** 2604.12737 | [PDF](https://arxiv.org/pdf/2604.12737v1)

**作者:** Gustavo de Carvalho Bertoli `[一作]` `[通讯]`, Gustavo de Carvalho Bertoli

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在联邦学习中使用差分隐私防御，构建黑盒多信号堆叠式成员推断攻击，对NIST基因组数据集进行实证评估。

**💡 创新点**

提出不依赖阴影模型的堆叠攻击框架，将七种基线推断器与交叉熵损失融合至XGBoost元分类器，首次在第三方基准上量化DP对MIAs的衰减。

**🔧 技术方法**

采用黑盒7种基线攻击（NN、RF、DT、GB、KNN、SVM、LR）+交叉熵、XGBoost元学习、DP‑SGD、层归一化、域自适应等技术。

**📊 数据集**

使用NIST Genomics PPFL Red Teaming挑战提供的四客户端大基因变异（125,767维）豆科种子表皮颜色分类数据集。

**📈 对比分析**

与基准1–7对比，未加DP时取得53.4%准确率、低DP（ε=200）38.4%（均显著高于基线），高DP（ε=10）仅24.7%；TPR在低FPR下仍显示残留泄露。

**⚠️ 局限性**

受限于仅有一个客户端完整成员标签、样本量极小、仅在高维基因组任务上验证、实验证实高DP导致模型性能急剧下降，结果可能不易推广至其他FL场景。

---

## 398. AffectAgent: Collaborative Multi-Agent Reasoning for Retrieval-Augmented Multimodal Emotion Recognition

**arXiv ID:** 2604.12735 | [PDF](https://arxiv.org/pdf/2604.12735v1)

**作者:** Zeheng Wang `[一作]` (Great Bay University), Qi Tian `[通讯]` (Guangming Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于多代理的检索增强生成框架AffectAgent，用于多模态情感识别。

**💡 创新点**

创新点在于三专用代理（查询规划、证据过滤、情感生成）的协同推理，并引入MB-MoE与RAAF实现跨模态平衡与缺失模态补偿。

**🔧 技术方法**

技术包括多代理近端策略优化（MAPPO）、检索增强自适应融合（RAAF）、模态平衡混合专家（MB-MoE）以及共享的大型多模态语言模型。

**📊 数据集**

使用MER-UniBench基准以及MER2025检索库进行评估。

**📈 对比分析**

与Rewrite‑Retrieve‑Read、BGM、RAG‑DDR等RAG方法对比，AffectAgent在全模态下取得最高均值76.78分，显著优于基线。

**⚠️ 局限性**

局限在于仍易受检索误差导致查询漂移影响，且对完全不匹配的情绪证据敏感。

---

## 399. Information-Theoretic Optimization for Task-Adapted Compressed Sensing Magnetic Resonance Imaging

**arXiv ID:** 2604.12709 | [PDF](https://arxiv.org/pdf/2604.12709v1)

**作者:** Xinyu Peng `[一作]` (Shanghai Jiao Tong University), Hongkai Xiong `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 6369 | [OpenAlex ID](https://openalex.org/A5002494284)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `729e5870-4135-47f5-97f2-e3974d07b5dc` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出一种基于信息理论的任务适配压缩感知MRI（InfoMRI）框架，联合优化采样、重建和下游任务，实现可适应任意采样比例并提供不确定性推理；

**💡 创新点**

创新点包括①以互信息为目标的任务适配优化，兼顾任务信息与重建信息；②利用可摊薄（amortized）优化实现单模型可调采样比例；③引入变分推断和潜在变量实现概率推理与不确定性估计；④在同一框架下同时支持重建辅助任务与隐私保护的压缩学习两种临床场景；

**🔧 技术方法**

采用变分信息估计、Barber‑Agakov下界、潜在变量模型（变分自编码器结构）、可摊薄采样生成网络（PGN）、均值-方差输出的重建网络、深度UNet/ISTA‑Net+等重建后端；

**📊 数据集**

使用公开MRI数据集fastMRI（单线圈），QUBIQ 2021前列腺分割数据集，SKM‑TEA大规模膝关节MRI，BRISC 2025脑肿瘤分割，MNIST用于隐私实验；

**📈 对比分析**

与传统固定采样、LOUPE、SeqMRI、PGMRI、LI‑Net、SemuNet、Tackle等方法对比，InfoMRI在重建PSNR/SSIM、分割DSC、GED、隐私保护PSNR等指标均取得最优或相当性能，且能在单一模型中实现多采样比例；

**⚠️ 局限性**

局限性包括仅在单线圈模拟数据上验证，缺乏多线圈并行成像与3D卷积的扩展；实验主要为学术评估，尚未在真实临床工作流程中与放射科医生验证；

---

## 400. Safe reinforcement learning with online filtering for fatigue-predictive human-robot task planning and allocation in production

**arXiv ID:** 2604.12667 | [PDF](https://arxiv.org/pdf/2604.12667v1)

**作者:** Jintao Xue `[一作]` (University of Hong Kong), Nianmin Zhang `[通讯]` (University of Hong Kong)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5075085708)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db`

**🎯 论文内容**

开发了 PF-CD3Q 算法，结合粒子滤波与安全强化学习，实现实时疲劳预测与人机协作任务规划与分配。

**💡 创新点**

创新点在于：①首次将安全强化学习（CMDP）与在线参数估计结合用于人机协作；②使用粒子滤波动态估计疲劳模型参数，显著提升预测精度；③构建基于 Transformer 的注意力网络，兼顾异构状态信息与疲劳约束；④通过安全动作集实现硬约束，避免疲劳违规。

**🔧 技术方法**

采用的技术包括粒子滤波（PF）、受限双深度 Q‑学习（CD3Q）、Transformer + 注意力机制、Noisy Net 探索、离线经验回放、Isaac Sim 物理仿真。

**📊 数据集**

实验数据来自基于真实空调通风管道工厂的仿真环境（NVIDIA Isaac Sim），包含多人、多机器人、材料、机器及工序的完整工序图和疲劳参数设置。

**📈 对比分析**

与 DQN、PPO、D3QN、PPO‑Lag 等基线以及其 PF 版本对比，评估指标为完成时间（makespan）和疲劳违规率（overwork）。PF‑CD3Q 在多数人机组合下获得第二佳的 makespan（仅次于 D3QN），并在 overwork 上几乎为零，显著优于传统方法。

**⚠️ 局限性**

局限性：仅在单一生产场景的仿真中验证；未在真实机器人系统中部署；需要可靠的实时疲劳监测与数字孪生基础设施；对不同工序复杂度和大规模工厂的可扩展性待进一步验证。

---

## 401. Do VLMs Truly "Read" Candlesticks? A Multi-Scale Benchmark for Visual Stock Price Forecasting

**arXiv ID:** 2604.12659 | [PDF](https://arxiv.org/pdf/2604.12659v1)

**作者:** Kaiqi Hu `[一作]` (University of Leeds), Mingwen Liu `[通讯]` (Likelihood Lab)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了一个多尺度蜡烛图数据集，并提出了标准评估框架，用于评估视觉-语言模型在股票价格预测中的视觉理解能力。

**💡 创新点**

创新点在于：①只使用视觉蜡烛图而非多模态输入，消除文本干扰；②引入多尺度（日、周）图像，模拟交易者多时间框架分析；③采用混淆矩阵与IC等多维度指标，系统评估模型的方向判断与排名能力。

**🔧 技术方法**

采用视觉语言模型（Claude、Gemini、GPT4o、Qwen 等）以及传统 XGBoost 作为基线，对蜡烛图图像进行回归预测，并利用结构化提示与多尺度图像进行推理。

**📊 数据集**

使用中国沪深300和美国标普500的历史 OHLCV 数据，生成每日与每周的蜡烛图，共约 193,524 张样本。

**📈 对比分析**

通过混淆矩阵、IC/Rank‑IC、市场极端情景等多指标比较，结果显示大部分 VLM 在持续涨跌趋势下表现较好，但在正常市场及多周期融合方面仍弱于 XGBoost；最优模型为 Claude‑sonnet‑4‑5（思考版）。

**⚠️ 局限性**

局限在于：模型主要捕捉短期技术信号，难以把握长期趋势；对多尺度信息的整合不充分；以及缺乏对更深层次基本面信息的利用，导致在极端或持续走势下预测偏差。

---

## 402. Can AI Tools Transform Low-Demand Math Tasks? An Evaluation of Task Modification Capabilities

**arXiv ID:** 2604.12743 | [PDF](https://arxiv.org/pdf/2604.12743v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 403. FeaXDrive: Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving

**arXiv ID:** 2604.12656 | [PDF](https://arxiv.org/pdf/2604.12656v1)

**作者:** Baoyun Wang `[一作]` (Tongji University), Lu Xiong `[通讯]` (Tongji University)

**通讯引用:** 23924 | [OpenAlex ID](https://openalex.org/A5100335131)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 FeaXDrive，一种以轨迹为核心、可行性感知的扩散式规划框架，用于端到端自动驾驶。

**💡 创新点**

创新点在于将轨迹视为统一对象进行扩散，配合自适应可微曲率约束、基于 SDF 的可行驶区域引导以及可行性感知的 GRPO 后训练，实现了在轨迹空间内的多维可行性提升。

**🔧 技术方法**

采用轨迹中心扩散、可微曲率约束训练、SDF 轨迹‑footprint 引导、GRPO 强化学习、VLM 条件编码和 DDIM/DFM 采样等技术。

**📊 数据集**

使用 NAVSIM（OpenScene/nuPlan）基准数据集进行训练与评估。

**📈 对比分析**

与 DiffusionDrive、ReCogDrive、UniAD 等基线对比；IL 训练下 PDMS 88.7、drivable‑area compliance 97.5；RLFT 后 PDMS 90.0、drivable‑area compliance 98.3，曲率违约率仅 0.88%（IL）/2.40%（RLFT），显著低于其他基线。

**⚠️ 局限性**

局限在于只覆盖几何/动力学可行性，未将所有可行性统一到奖励中；可行驶区域引导依赖局部 HDMap，未来需探索更统一、轻量化的可行性建模与场景先验。

---

## 404. CLASP: Class-Adaptive Layer Fusion and Dual-Stage Pruning for Multimodal Large Language Models

**arXiv ID:** 2604.12767 | [PDF](https://arxiv.org/pdf/2604.12767v1)

**作者:** Yunkai Dang `[一作]` (Nanjing University), Yang Gao `[通讯]` (Nanjing University)

**通讯引用:** 13381 | [OpenAlex ID](https://openalex.org/A5070337115)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种可插拔的视觉令牌压缩框架 CLASP，用于在多模态大语言模型中根据任务类别自适应融合多层视觉特征并进行双阶段稀疏化。

**💡 创新点**

创新点在于：①基于文本意图的类别路由器动态决定视觉层级混合权重，捕获不同任务所需的细节与语义层次；②双阶段剪枝机制（关注度先行+相似度聚类）根据类别分配令牌预算，兼顾相关性与覆盖度；③框架无须重新训练，可直接插入多种 MLLM 体系。

**🔧 技术方法**

主要技术包括：prompt‑to‑class 路由、可学习的多层 ViT 特征混合、基于注意力权重的相关性评分、基于余弦相似度的冗余度评估、球面 K‑means 聚类以及层级的进阶剪枝。

**📊 数据集**

在 10 个图像/视频基准（如 GQA、MMBench、MME、POPE、VQA‑v2、ScienceQA、TextVQA、TGIF、MSVD、MSRVTT）以及多种 MLLM 架构（LLaVA‑1.5、LLaVA‑NeXT、Video‑LLaVA、Qwen2.5‑VL）上进行评估。

**📈 对比分析**

与现有单层剪枝（ToMe、FastV、HiRED、PDrop 等）和多阶段剪枝（SparseVLM、VisionZip、DART 等）对比，CLASP 在保持 94.7%‑98.4% 任务准确率的同时，实现了高达 88.9% 的令牌压缩，显著提升速度（1.5×‑2.1×）与内存占用。

**⚠️ 局限性**

局限性包括：①需要预先定义任务类别与路由器，未实现端到端学习；②对超高分辨率或视频场景仍受限于视觉编码器的计算开销；③在极端压缩率下可能失去细粒度视觉细节，导致偶发错误。

---

## 405. ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search

**arXiv ID:** 2604.12762 | [PDF](https://arxiv.org/pdf/2604.12762v1)

**作者:** Myungchul Kim `[一作]` (KAIST), In So Kweon `[通讯]` (KAIST)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了ARGOS基准与框架，将多摄像头人物搜索重新定义为交互式空间-时间推理问题，并通过STTG图实现摄像头连通性与转移时间的实体化；

**💡 创新点**

创新点在于：①将人物搜索转化为主动提问、查询与排除候选的交互式决策过程；②引入面向工具调用的四模态LLM代理；③构建了覆盖空间、时间与语义三层任务的三轨评测；

**🔧 技术方法**

核心技术包括：基于LLM的工具调用代理（Analyst-Planner-Interviewer-Interpreter四模块）；Spatio‑Temporal Topology Graph (STTG) 用于空间-时间约束；以及针对每个轨道的特定工具集（属性查询、区域结构检索、时空可行性校验等）；

**📊 数据集**

使用的数据集为由16台同步摄像头捕获的两个真实场景（工厂与校园）中1,273名人物的图像与属性，生成了2,691个任务（14个子场景，3个进阶轨道）；

**📈 对比分析**

通过对四种LLM骨干（GPT‑5.2、GPT‑4o、GPT‑5‑mini、Claude Sonnet 4）在20轮预算下的评测，发现工具增强的代理在空间轨道的TWS最高达0.383、时间轨道最高达0.590，而直接LLM推理的性能远低；

**⚠️ 局限性**

局限性包括：模拟的目击者回复是确定性的，缺乏真实记忆误差；仅覆盖两个场景，未覆盖多样化布局；以及该技术在监控场景中的潜在隐私与伦理风险。

---

## 406. Modular Verification of Differential Privacy in Probabilistic Higher-Order Separation Logic (Extended Version)

**arXiv ID:** 2604.12713 | [PDF](https://arxiv.org/pdf/2604.12713v1)

**作者:** Philipp G. Haselwarter `[一作]` (Aarhus University), Lars Birkedal `[通讯]` (Aarhus University)

**通讯引用:** 6337 | [OpenAlex ID](https://openalex.org/A5055959064)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文提出并实现了一种针对差分隐私的概率高阶分离逻辑，利用该逻辑对动态预算、交互式分析和缓存等复杂隐私库进行形式化验证。

**💡 创新点**

核心创新在于将隐私预算抽象为可分离的资源，设计了新的 Laplace 采样规则（Laplace‑shift、Laplace‑choice）和隐私预算耦合规则，支持高阶函数、局部状态以及可重用的抽象规格。

**🔧 技术方法**

技术方法包括：基于 Iris 的关系分离逻辑框架、近似耦合理论、隐私预算资源代数、Coq 证明助手实现及其适配的 Hoare 四元组。

**📊 数据集**

该工作属于形式化验证范畴，不依赖具体数据集。

**📈 对比分析**

与现有类型系统、apRHL、HO‑RPL 等方法比较，本文展示了在实现 AT、SVT、Report‑Noisy‑Max、隐私过滤器、缓存等实例上的可模组验证覆盖面更广、无需手工预算推导，验证过程依赖于 Coq 证明助手，未给出实验性性能指标。

**⚠️ 局限性**

局限性包括：尚未支持并发、Gaussian 采样、子采样放大、先进组合等；只验证了有限递归和确定性分支，未覆盖无限递归或并发情形。

---

## 407. Construction $π_A$ over Multiquadratic Fields for Compound Block-Fading Wiretap Channels

**arXiv ID:** 2604.12703 | [PDF](https://arxiv.org/pdf/2604.12703v1)

**作者:** Juliana G. F. Souza `[一作]` (University of Campinas), Cong Ling `[通讯]` (Imperial College London)

**通讯引用:** 3716 | [OpenAlex ID](https://openalex.org/A5065795716)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

构造了基于多二次数域的多层晶格码（Construction π_A），并将其用于复合块衰落无线信息窃听信道的可靠传输与强信息安全。

**💡 创新点**

创新点在于：① 将 Construction π_A 引入多二次数域，利用完全分裂的有理素数实现CRT分解为小字母（如二进制）并支持多阶段译码；② 将这种晶格结构与离散高斯塑形、平坦因子分析结合，证明在复合块衰落模型下可获得全局可靠性和强安全性。

**🔧 技术方法**

使用的技术包括：代数数域理论、典型嵌入、CRT与多层编码、Construction π_A、离散高斯编码、平坦因子与自洽约简、LDPC 组件码的多阶段译码。

**📊 数据集**

实验数据集为：K = ℚ(√17,√33) 上的四层二进制 CRT，使用 800 位 3‑regular LDPC 码，并在 2×2 瑞利 MIMO（Bob 与 Eve）下模拟信道，记录 Bob 与 Eve 的误码率。

**📈 对比分析**

与传统单层 Construction A 或其他数域晶格相比，本文实现了显著的误码率瀑布（Bob）并通过 7.8 dB 的噪声等效增益抑制 Eve 的性能；虽然没有给出严格的理论性能上界，但仿真表明在所选参数下能达到可靠率与强安全率的统一。

**⚠️ 局限性**

局限性包括：① 仅在特定多二次数域与完全分裂素数上证明，推广到其他域或不完全分裂情形需进一步研究；② 依赖离散高斯塑形与平坦因子估计，实际实现时需调节标准差；③ 仿真仅针对 2×2 阵列与 800 位码长，缺乏大规模或低时延系统的完整评估。

---

## 408. MISID: A Multimodal Multi-turn Dataset for Complex Intent Recognition in Strategic Deception Games

**arXiv ID:** 2604.12700 | [PDF](https://arxiv.org/pdf/2604.12700v1)

**作者:** Shufang Lin `[一作]` (Chinese University of Hong Kong), Fangxin Wang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 2514 | [OpenAlex ID](https://openalex.org/A5101686970)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了MISID数据集，涵盖多轮、多模态、多参与者的战略欺骗游戏对话，并构建了两层多维注释与因果推理框架；

**💡 创新点**

创新点在于：① 通过精确同步音视频捕捉真实的欺骗情境；② 设计了跨模态、跨轮次的因果注释体系，促使模型学习基于证据的推理；③ 提出FRACTAM框架，采用模态解耦、双阶段检索与显式因果链构建，有效缓解文本偏置、视觉幻觉与模态协同不足。

**🔧 技术方法**

技术手段包括：多模态大模型（MLLM）解耦生成客观事实文本；双路径（词典+语义）检索结合RRF与交叉编码重排序；因果链提示式推理；LLM评估器用于指标计算。

**📊 数据集**

使用MISID数据集，包含3962段语音-视频记录，120个角色实例，15名参与者，共计约9.15小时。

**📈 对比分析**

与多款视频LLM（如GPT‑4o、Grok‑4‑Fast等）和文本LLM（Claude‑Sonnet‑4.5、DeepSeek‑V3等）在隐式意图推理、欺骗检测、身份判断等指标进行零样本对比；FRACTAM在多项指标上提升约8–12个百分点，尤其在隐藏意图推理得分上显著优于基线。

**⚠️ 局限性**

局限性包括：① 仍依赖LLM对因果链的准确性，容易出现逻辑跳跃；② 数据来源局限于公开游戏视频，难以覆盖更广泛情境；③ 对长时间对话的记忆衰减未完全解决，需进一步优化检索与记忆机制。

---

## 409. BID-LoRA: A Parameter-Efficient Framework for Continual Learning and Unlearning

**arXiv ID:** 2604.12686 | [PDF](https://arxiv.org/pdf/2604.12686v1)

**作者:** Jagadeesh Rachapudi `[一作]` (Indian Institute of Technology Mandi), Amit Shukla `[通讯]` (Indian Institute of Technology Mandi)

**通讯引用:** 1583 | [OpenAlex ID](https://openalex.org/A5008777707)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种参数高效的双向LoRA框架，统一实现连续学习与忘记；

**💡 创新点**

通过三路专用适配器隔离保留、学习与忘记任务，并引入逃逸式忘记，显著降低知识泄露；

**🔧 技术方法**

采用LoRA低秩适配、注意力模块、逃逸方向优化、经验回放和三路损失分离技术；

**📊 数据集**

在CIFAR-100分类任务和CASIA-Face100人脸识别任务上进行实验；

**📈 对比分析**

使用六任务滑动窗口评估，参数更新约5%，知识泄露≤3%，整体准确率接近oracle，MIA≈0.5，优于所有基线；

**⚠️ 局限性**

仍需保留10%缓冲区；逃逸方向缩放需手动调节；尚未验证对非视觉任务的泛化能力。

---

## 410. Intelligent resource prediction for SAP HANA continuous integration build workloads

**arXiv ID:** 2604.12673 | [PDF](https://arxiv.org/pdf/2604.12673v1)

**作者:** Torsten Mandel `[一作]` (SAP SE), Stephan Kraft `[通讯]` (SAP SE)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究并实现了基于机器学习的 SAP HANA CI 构建任务内存需求预测系统，并将其部署到生产环境中，实现了平均每个构建约 36GB 内存节省。

**💡 创新点**

首次将 LightGBM–XGBoost 量化回归集成与安全偏移策略相结合，显著降低 OOM 发生率且保持高利用率，同时公开了 30 万次构建的历史数据集。

**🔧 技术方法**

采用 LightGBM 与 XGBoost 的量化回归集成、FastAPI 微服务、Optuna 超参搜索、GitHub Actions 自动化 CI/CD、MLflow 监控与 Grafana 可视化。

**📊 数据集**

基于 300,000+ 次 SAP HANA CI 构建的历史执行数据，包含 22 个内存分配类别、代码提交、构建配置与最大 RSS 等特征，公开发布在 GitHub。

**📈 对比分析**

通过离线训练/测试集对比分类与回归模型，量化回归集成在基线下将未分配率从 3.84% 降至 0.28%，平均节省 37GB/构建，且在生产中维持 0.3% 以内的 OOM 率。

**⚠️ 局限性**

当前仅预测系统内存且采用预聚合峰值，无法动态调整任务运行中的资源；且对多维度资源（CPU 等）预测仍待扩展。

---

## 411. A hierarchical spatial-aware algorithm with efficient reinforcement learning for human-robot task planning and allocation in production

**arXiv ID:** 2604.12669 | [PDF](https://arxiv.org/pdf/2604.12669v1)

**作者:** Jintao Xue `[一作]` (University of Hong Kong), Nianmin Zhang `[通讯]` (University of Hong Kong)

**通讯引用:** 34 | [OpenAlex ID](https://openalex.org/A5075085708)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出并实现了一个分层的实时人机协同任务规划与分配框架（EBQ&SAP），通过高层深度Q学习决策优先级任务、低层基于路径规划的空间感知分配，从而在复杂动态生产环境中实现高效的人机协作。

**💡 创新点**

创新点包括：①引入高效经验缓冲（episode‑end processing）以解决稀疏奖励问题，显著提升训练速度和性能；②将Transformer、Dueling网络与Noisy Nets融合到高层策略网络，实现对异构状态信息的高效处理；③构建离线节点图并实时利用预计算路径实现低层空间感知分配，避免在线路径规划开销。

**🔧 技术方法**

核心技术包括：强化学习（Deep Q‑Learning，Dueling DQN，Noisy Nets），Transformer与多头注意力网络，优先经验回放，Hybrid A*路径规划，离线节点图构建，空间感知任务分配。

**📊 数据集**

数据集与实验环境：在NVIDIA Isaac‑Sim 3D仿真平台搭建的模块化集成建设（MiC）空调风管生产线仿真环境中，使用仿真生成的任务、工件、工人、机器人位置等数据进行训练与评估。

**📈 对比分析**

与传统D3QN、EDQN1/2、EPPO等对比，EBQ‑G/EBQ‑N在训练时间上缩短约4倍，在平均 makespan 上降低 10–15%，并在零样本（zero‑shot）测试中保持 100% 成功率；低层 SAP 在 makespan 与机器人/人机移动距离上优于随机分配与“最远路径”方法，提升 20–30%。

**⚠️ 局限性**

局限性：①依赖仿真环境，真实工厂部署需构建完善的数字孪生与多源实时数据接口；②任务与子任务拆分需人工定义，缺乏通用化自动化机制；③对机器人多样化能力的考虑有限，当前多为协作机器人与移动机器人，难以直接迁移至更复杂多样的机械体系。

---

## 412. Reliability-Guided Depth Fusion for Glare-Resilient Navigation Costmaps

**arXiv ID:** 2604.12753 | [PDF](https://arxiv.org/pdf/2604.12753v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 413. OFA-Diffusion Compression: Compressing Diffusion Model in One-Shot Manner

**arXiv ID:** 2604.12668 | [PDF](https://arxiv.org/pdf/2604.12668v1)

**作者:** Haoyang Jiang `[一作]` (Renmin University of China), Ju Fan `[通讯]` (Renmin University of China)

**通讯引用:** 3063 | [OpenAlex ID](https://openalex.org/A5100739546)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `fede83ac-7505-405f-ab37-e7284695c47f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种一次性压缩扩散模型（DPM）的框架，可在一次训练中生成多种不同计算规模的子网络；

**💡 创新点**

创新点在于：①基于通道重要性（灵敏度）对通道进行排序并按层重要性贪心分配宽度，形成更合适的子网络；②采用重加权训练策略平衡不同规模子网络的收敛速度；③一次性训练即可得到多模型，显著降低多次压缩与重训的计算与存储开销；

**🔧 技术方法**

使用技术包括：DPM（EDM、U‑ViT、Stable Diffusion）、通道重要性估计（一阶泰勒灵敏度）、贪心通道分配、一次性网络训练（OFA）、重加权损失、参数共享、微调与量化等；

**📊 数据集**

实验数据集包括：CIFAR‑10、FFHQ、AFHQv2、ImageNet、CelebA、MS‑COCO；

**📈 对比分析**

通过与“单独压缩”“随机架构”“原始OFA实现”三种基线对比，使用 FID 评估生成质量；结果显示：在大多数子网络规模下，OFA压缩模型与单独压缩相当甚至更优，尤其在低资源（<0.5 参数保留率）下优势显著；训练时间从 N×200K 降至 200K，存储占用也大幅减少；推理延迟也明显下降；

**⚠️ 局限性**

局限性：①重要性评分固定为预训练模型，未在训练中动态更新，可能导致最优性不足；②尝试对不同时间步或不同子网络更新重要性时效果不佳，说明方法对多任务/时序特性不敏感；③目前仅研究通道剪枝，未考虑结构层级或深度搜索；④对更大规模模型或其他生成任务的推广尚未验证；

---

## 414. Robust Semi-Supervised Temporal Intrusion Detection for Adversarial Cloud Networks

**arXiv ID:** 2604.12655 | [PDF](https://arxiv.org/pdf/2604.12655v1)

**作者:** Anasuya Chattopadhyay `[一作]` (German Research Center for Artificial Intelligence), Hans D. Schotten `[通讯]` (German Research Center for Artificial Intelligence)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种鲁棒的半监督时序网络入侵检测框架，专门针对云网络中的对抗性污染和非平稳流量进行处理。

**💡 创新点**

创新点在于将一致性正则化、置信度感知伪标签以及选择性时序不变性三种技术结合，能够保守利用未标记流量，显著提升检测的鲁棒性和泛化能力。

**🔧 技术方法**

采用了半监督学习技术（一致性正则化、伪标签方法）、时序不变性约束以及基于流级特征的深度学习模型。

**📊 数据集**

使用公开数据集 CIC-IDS2017、CSE-CIC-IDS2018 和 UNSW-NB15 进行评估。

**📈 对比分析**

与现有监督及半监督 IDS 进行对比实验，结果显示在有限标签条件下检测准确率更高、标签使用更高效、且对对抗攻击和流量漂移具有更强的稳健性。

**⚠️ 局限性**

局限性包括对流量特征提取的依赖、在极端对抗强度下可能仍出现性能下降、需要持续更新时间窗口以适应漂移，并且对新型攻击模式的适应速度有限。

---

## 415. On the Meaning of the Web as an Object of Study

**arXiv ID:** 2604.12756 | [PDF](https://arxiv.org/pdf/2604.12756v1)

**作者:** Claudio Gutierrez `[一作]` (Universidad de Chile), Daniel Hernández `[通讯]` (University of Stuttgart)

**通讯引用:** 207 | [OpenAlex ID](https://openalex.org/A5073473252)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

阐述并论证 Web 作为研究对象在学术界的衰退与碎片化，并提出需要社区讨论

**💡 创新点**

将 Web 重新定位为环境并运用 Simon 的人工系统理论阐释其身份危机

**🔧 技术方法**

基于定性分析、开放数据源与程序委员会经验

**📊 数据集**

无明确实验数据集，仅使用会议投稿统计与主题演化

**📈 对比分析**

无实验比较，主要是文献综述与案例分析

**⚠️ 局限性**

缺乏定量验证、对未来路径缺乏操作性方案

---

## 416. From Imitation to Discrimination: Progressive Curriculum Learning for Robust Web Navigation

**arXiv ID:** 2604.12666 | [PDF](https://arxiv.org/pdf/2604.12666v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 417. Scaling In-Context Segmentation with Hierarchical Supervision

**arXiv ID:** 2604.12752 | [PDF](https://arxiv.org/pdf/2604.12752v1)

**作者:** T. Camaret Ndir `[一作]` (Medical Center – University of Freiburg), Robin T. Schirrmeister `[通讯]` (Medical Center – University of Freiburg)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

设计了PatchICL框架，利用分层块采样与多级监督实现医学图像分割的可扩展上下文学习，显著降低计算量；

**💡 创新点**

创新点包括：1) 熵引导的Gumbel‑top‑K块采样，主动挑选最不确定的区域；2) 对块选择过程提供多级监督，提升选择效率；3) 分层从粗到细的预测融合，兼顾全局上下文与局部细节；

**🔧 技术方法**

使用技术包括：Transformer注意力（RoPE、类型嵌入、双向注意），Gumbel‑top‑K采样，多级交叉熵+Dice损失，冻结的UniverSeg编码器，以及层次化分辨率级联；

**📊 数据集**

使用数据集：TotalSegmentator CT（训练与验证），TotalSegmentator MRI，和MedSegBench 35个跨模态数据集；

**📈 对比分析**

与全局注意力基线UniverSeg对比，PatchICL在512×512分辨率下计算量下降44%，Dice略优或相当；在CT、MRI上表现相当或略优；在MedSegBench中在6/13模态（尤其是OCT、皮肤镜）表现更好；

**⚠️ 局限性**

局限性：在复杂多关节解剖结构（如脊柱）上仍落后于UniverSeg；块采样策略仍基于手工设定，缺乏自适应学习；在高对比度全局结构（如X‑ray）上的性能不足。

---

## 418. Universal NER v2: Towards a Massively Multilingual Named Entity Recognition Benchmark

**arXiv ID:** 2604.12744 | [PDF](https://arxiv.org/pdf/2604.12744v1)

**作者:** Terra Blevins `[一作]`, Yuval Pinter `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

发布并更新了Universal NER v2，新增11个数据集覆盖10种新语言（共30个数据集、22种语言，约300万标注词元）并对原始数据集做了细微修正；

**💡 创新点**

在多语种、跨语言并行文本上构建统一标注规范、扩展语言与体裁多样性，并提供系统的跨语言实体一致性分析；

**🔧 技术方法**

采用TALEN标注平台、统一3类标签（per、loc、org）与IOB2格式，利用XLM‑R_Large进行跨语言微调，同时对大型语言模型（Gemini、Claude、GPT‑5）进行零提示的注释实验；

**📊 数据集**

使用来自UNER v2的新11个数据集（如希腊语、希伯来语、挪威 Nynorsk/Bokmål、斯洛文尼亚语、瑞典语、捷克语、印尼语、日语、韩语、罗马尼亚语）以及原始v1数据集作为基准；

**📈 对比分析**

实验结果表明，XLM‑R在语言内表现最好，欧洲语言间跨语言转移可达0.60以上；但对日语、韩语等非欧洲语言性能显著下降；LLM平均F1约0.50，远低于人类注释者的0.74；

**⚠️ 局限性**

受限于语言多样性导致的跨语言迁移难度、LLM提示与标注规范匹配不足、部分数据集缺少IAA评估以及部分语言仍处于低资源状态，整体性能受限。

---

## 419. Longest Common Extension of a Dynamic String in Parallel Constant Time

**arXiv ID:** 2604.12696 | [PDF](https://arxiv.org/pdf/2604.12696v1)

**作者:** Daniel Albert `[一作]` (TU Dortmund University), Daniel Albert `[通讯]` (TU Dortmund University)

**通讯引用:** 29576 | [OpenAlex ID](https://openalex.org/A5108167009)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

在动态字符串上实现常数时间的最长公共扩展（LCE）查询

**💡 创新点**

首次将字符串同步集合与Muddling‑Lemma结合，允许信息略微过时以实现并行常数时间

**🔧 技术方法**

利用字符串同步集合层次结构、分解层次、局部稀疏性、延迟处理与批量更新等技术

**📊 数据集**

未使用实际数据集，全部为理论分析与算法证明

**📈 对比分析**

算法在共CRCW PRAM上实现常数时间，使用O(n^ϵ)处理器，查询与更新均为O(n^ϵ)工作量，显著优于之前的O(log^k n)或O(n^o(1))方案

**⚠️ 局限性**

限制为预设最大字符串长度，且仅支持整数字母表，且缺乏匹配的工作量下界

---

## 420. Hypergraph-State Collaborative Reasoning for Multi-Object Tracking

**arXiv ID:** 2604.12665 | [PDF](https://arxiv.org/pdf/2604.12665v1)

**作者:** Zikai Song `[一作]` (Huazhong University of Science and Technology), Xinchao Wang `[通讯]` (National University of Singapore)

**通讯引用:** 13482 | [OpenAlex ID](https://openalex.org/A5015574447)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 HyperSSM，一种通过超图协作推理与状态空间模型相结合的运动估计框架，用以在多目标跟踪中提升运动预测的稳定性与遮挡鲁棒性。

**💡 创新点**

创新点在于：①对运动状态相似的目标自动构造高阶超图边，允许多目标共同协商运动状态；②将超图卷积嵌入状态空间模型的状态转移中，实现空间协作与时间连续性的统一优化；③利用残差连接与超图参数共享，提升学习效率并减少冗余。

**🔧 技术方法**

核心技术包括：超图构造与超图卷积（HConv），状态空间模型（SSM）与其离散化参数，YOLOX 检测器，基于运动、空间与外观的多阶段自适应关联策略，滑动窗口训练与平滑 L1 + GIoU 损失。

**📊 数据集**

使用数据集：MOT17、MOT20（线性行人跟踪）、DanceTrack、SportsMOT（非线性、复杂运动）。

**📈 对比分析**

与多种基准方法（Query‑based 如 TrackFormer、Motion‑based 如 ByteTrack、StrongSORT 等）在 HOTA、IDF1、MOTA 等指标上进行公开与私有协议下的比较，结果显示 HyperSSM 在 MOT17/MOT20 达到 66.9/65.2 HOTA、80.0/80.0 MOTA，在 DanceTrack、SportsMOT 上分别取得 78.5/68.0 HOTA，显著优于现有最优方法，且在非线性场景中提升幅度最大。

**⚠️ 局限性**

局限性包括：①对阈值 θ 等超参数敏感，需手工调优；②目前仅基于轨迹与视觉特征，未充分利用多模态信息；③模型规模与推理速度受限于超图卷积与多层 SSM 的计算开销，轻量化版本仍有提升空间。

---

## 421. Human-Centric Topic Modeling with Goal-Prompted Contrastive Learning and Optimal Transport

**arXiv ID:** 2604.12663 | [PDF](https://arxiv.org/pdf/2604.12663v1)

**作者:** Rui Wang `[一作]` (Nanjing University of Posts and Telecommunications), Philip Torr `[通讯]` (University of Oxford)

**通讯引用:** 59512 | [OpenAlex ID](https://openalex.org/A5042899882)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了人类中心化主题建模（Human‑TM）框架，并设计了目标提示对比学习与最优传输的目标驱动对比主题模型 GCTM‑OT，用以生成符合人类目标、可解释、丰富多样的主题。

**💡 创新点**

创新点在于将用户目标直接通过大语言模型提示抽取并注入主题学习，利用对比学习和最优传输对齐主题与目标，提升主题相关性与多样性，并兼顾传统主题的可解释性。

**🔧 技术方法**

核心技术包括 LLM（GPT‑3.5‑turbo）提示工程、Transformer（MPNet）语义表示、对比学习、Dirichlet 先验匹配、最优传输（Sinkhorn 迭代）以及目标聚类与语义相似度计算。

**📊 数据集**

实验使用了三个公开 Reddit 子版块数据集：Bothering、TeslaModel3 与 AskAcademia，涵盖个人困扰、汽车使用与学术生活三类主题。

**📈 对比分析**

与 10 种现有主题模型（LDA、BAT、CTMNeg、vONT、CWTM、DisCTM、HiCOT、CAST、LLM‑TE、LLM‑ITL）比较，GCTM‑OT 在主题一致性（C_P、C_A、NPMI、UCI）和多样性（UT）指标上均显著优于基线，同时在新提出的目标相似度（GS）、目标相关主题率（GTR）和目标覆盖率（GCR）上取得最高分。

**⚠️ 局限性**

主要局限在于对 LLM 的依赖导致推理成本高、目标抽取质量受提示设计与模型能力限制，且实验仅覆盖 Reddit 文本数据，未来需验证跨领域与多语言场景的通用性。

---

## 422. Transformer Based Machine Fault Detection From Audio Input

**arXiv ID:** 2604.12733 | [PDF](https://arxiv.org/pdf/2604.12733v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876`

---

## 423. Rethinking the Personalized Relaxed Initialization in the Federated Learning: Consistency and Generalization

**arXiv ID:** 2604.12768 | [PDF](https://arxiv.org/pdf/2604.12768v1)

**作者:** Li Shen `[一作]` (Sun Yat-sen University), Dacheng Tao `[通讯]` (Nanyang Technological University)

**通讯引用:** 100329 | [OpenAlex ID](https://openalex.org/A5074103823)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 FedInit 方法，在联邦学习中引入个性化松弛初始化，以缓解客户端漂移并提升一致性。

**💡 创新点**

创新点在于：1) 设计一种轻量级的松弛初始化策略（向当前全局状态的反向偏移）；2) 通过额外的超调项对优化误差与泛化误差进行联合的“超风险”分析，证明一致性项可显著降低收敛上界；3) 说明该初始化可作为插件轻松集成到现有算法中。

**🔧 技术方法**

使用技术包括 FedAvg 基础框架、个性化松弛初始化、L‑光滑与 PŁ 条件下的理论分析、插值假设、均匀稳定性（uniform stability）分析以及对一致性项 Δ^t 的递推推导。

**📊 数据集**

实验数据集为 CIFAR‑10 与 CIFAR‑100，采用 Dirichlet 分裂产生不同程度的客户端异质性，并在 ResNet‑18‑GN 与 VGG‑11 两个网络上验证。

**📈 对比分析**

与 FedAvg、FedAdam、FedSAM、SCAFFOLD、FedDyn、FedCM 等基线在多种异质度、参与率与模型规模下进行对比，FedInit 在保持相同通信/计算成本的前提下，平均提升 2–3% 的测试准确率，甚至达到 SOTA 水平。

**⚠️ 局限性**

局限性包括：需手动调参 β 与本地迭代步数 K，性能对异质度敏感；理论分析依赖光滑非凸、PŁ 条件与插值假设；在极端通信瓶颈或极小设备上可能因初始化偏差导致收敛不稳定；未针对 straggler 或隐私保护机制进一步优化。

---

## 424. Short Version of VERIFAI2026 Paper -- Learning Infused Formal Reasoning: Contract Synthesis, Artefact Reuse and Semantic Foundations

**arXiv ID:** 2604.12747 | [PDF](https://arxiv.org/pdf/2604.12747v1)

**作者:** Arshad Beg `[一作]` (Maynooth University), Rosemary Monahan `[通讯]` (Maynooth University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

该论文提出学习驱动的形式化推理框架 LIFR，将大语言模型与 SMT/定理证明器、图匹配和图嵌入技术相结合，实现自然语言需求到契约的自动化生成、验证工件的语义重用，并以 UTP 与机构论为语义基础，构建可互操作、可靠的验证生态。

**💡 创新点**

创新点包括：① 通过神经符号流水线实现契约的自动生成与验证反馈；② 利用图表示与学习嵌入实现跨项目验证工件的语义匹配与转换；③ 在此框架下引入 UTP 与机构论作为语义治理，保证 AI 生成的规范在多工具、多语言环境下保持一致性与可验证性。

**🔧 技术方法**

技术方法包括大语言模型 (LLM)、SMT 求解器、定理证明器、图匹配算法、图嵌入技术、UTP（Unifying Theories of Programming）以及机构论（Theory of Institutions），并借助 Isabelle/UTP 等工具实现形式化推理。

**📊 数据集**

本文为愿景性工作，未给出具体数据集；若后续实现，将使用自然语言需求集合（如工业需求规范）以及现有的形式化验证库（如 Isabelle、Coq 等已验证案例）。

**📈 对比分析**

尚未进行实验与性能评估；预期通过与传统手工契约编写及无学习辅助的验证流程对比，能够显著降低人工成本、提升验证工件重用率，并通过迭代反馈机制提升生成契约的可验证性。

**⚠️ 局限性**

局限性：缺乏实证验证，LLM 生成的契约可解释性与可靠性仍有风险；跨工具语义匹配的精度与自动化程度待进一步研究；以及在大规模工业项目中实现完整闭环验证仍面临技术与组织挑战。

---

## 425. Stress Detection Using Wearable Physiological and Sociometric Sensors

**arXiv ID:** 2604.12746 | [PDF](https://arxiv.org/pdf/2604.12746v1)

**作者:** Oscar Martinez Mozos `[一作]` (Polytechnic University of Cartagena), Jose Manuel Ferrandez `[通讯]` (Polytechnic University of Cartagena)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文通过将可穿戴生理传感器（EDA、PPG、HRV）与社交度量徽章（语音、运动）相结合，构建了一个可实时检测社交情境下压力水平的系统，并在18名受试者的 Trier Social Stress Test (TSST) 实验中进行了验证。

**💡 创新点**

创新点在于首次将生理信号与社交度量（语音与运动）多模态数据融合用于压力检测，并为每位受试者训练个性化分类器，强调了单一传感器不足以捕捉复杂社会压力的现象。

**🔧 技术方法**

采用支持向量机（RBF 与线性核）和 AdaBoost 两种机器学习算法，对每位受试者分别训练个人化模型，并使用 AdaBoost 的弱分类器排名来筛选最具辨别力的特征。

**📊 数据集**

数据集来源于18名受试者在 TSST 过程中的同步采样，采样率 10 Hz，包含 5 维生理特征与 19 维社交特征，共计约 13745 条样本，每个样本标记为“压力”或“中性”。

**📈 对比分析**

与单一模态相比，组合模态在 0.94±0.03 的准确率、0.94±0.03 的精确率和 0.96±0.02 的召回率下取得最佳表现；AdaBoost 与 RBF‑SVM 结果相近，均优于线性 SVM（≈0.85/0.84）。

**⚠️ 局限性**

局限性包括：实验环境为受控 TSST，未覆盖日常生活场景；受试者样本量仅 18 人；可穿戴设备舒适性不足；仅实现二分类，缺乏对不同压力强度的区分。

---

## 426. Stability and Geometry of Attractors in Neural Cellular Automata

**arXiv ID:** 2604.12720 | [PDF](https://arxiv.org/pdf/2604.12720v1)

**作者:** Mia-Katrin Kvalsund `[一作]` (University of Oslo), James Stovold `[通讯]` (University of York)

**通讯引用:** 87 | [OpenAlex ID](https://openalex.org/A5021911436)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

对神经细胞自动机进行动力学分析，绘制并分类其吸收器，并评估扰动后的鲁棒性。

**💡 创新点**

首次对NCAs吸收器进行可视化与Lyapunov、傅里叶谱分析，揭示其周期/准周期振荡行为及次级吸收器的存在。

**🔧 技术方法**

采用Lyapunov谱、傅里叶功率谱、PCA降维以及数值实验对NCAs进行分析。

**📊 数据集**

使用基于“绿色蜥蜴”图像（40×40×16通道）训练的NCAs，进行多次随机种子实验。

**📈 对比分析**

与传统认为的固定点吸收器进行对比，评估Lyapunov指数、频谱峰值，结果显示所有模型均为负/零指数，无混沌，且振荡周期一致。

**⚠️ 局限性**

仅研究了确定性更新且单一任务的NCAs；未完全估算完整Lyapunov谱；对次级吸收器的机制和广泛适用性缺乏深入验证。

---

## 427. EPAC: The Last Dance

**arXiv ID:** 2604.12715 | [PDF](https://arxiv.org/pdf/2604.12715v1)

**作者:** Filippo Mantovani `[一作]` (Barcelona Supercomputing Center), Jens Krüger `[通讯]` (Fraunhofer ITWM)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计、实现并验证了一个基于RISC‑V的多功能加速器芯片EPAC，集成了三种计算单元VEC、STX、VRP，并完成芯片tape‑out与系统级验证。

**💡 创新点**

创新点在于开放式RISC‑V体系结构下的多样化加速器设计、全流程欧洲跨机构协作以及统一的NoC+L2缓存体系，提供可比的硬件平台与软件栈。

**🔧 技术方法**

采用RISC‑V指令集、RVV 0.7.1向量扩展、CHI协议NoC、GF22FDX FDSOI工艺、SerDes链路、分布式L2缓存、ARM Cortex/Spartan cores等技术。

**📊 数据集**

通过DGEMM、Stream、LINPACK等标准HPC基准以及Ubuntu 22.04下的系统级测试进行验证。

**📈 对比分析**

通过在FPGA与DDR/HBM的联合实验验证了20 GB/s的C2C链路，EPAC能够顺利启动Ubuntu并跑完整的LINPACK，表明在相同硬件约束下三种加速器分别在向量、ML与高精度算子上具备竞争力。

**⚠️ 局限性**

限制在于多方协同带来的复杂度、缺少片上内存控制器导致需外部FPGA、可扩展性与功耗仍待进一步评估，以及对比基准尚缺乏统一量化指标。

---

## 428. Multi-Agent Digital Twins for Strategic Decision-Making using Active Inference

**arXiv ID:** 2604.12657 | [PDF](https://arxiv.org/pdf/2604.12657v1)

**作者:** Francesco Maria Mancinelli `[一作]` (Politecnico di Milano), Andrea Manzoni `[通讯]` (Politecnico di Milano)

**通讯引用:** 6671 | [OpenAlex ID](https://openalex.org/A5082126056)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出一种多代理主动推理框架，用于数字孪生环境中各代理通过分布式生成模型实现自适应决策，并通过 Cournot 竞争实验验证其可行性。

**💡 创新点**

创新点包括：① 在生成模型中加入上下文推理以实现动态环境适应；② 将流式机器学习嵌入生成模型以在线更新目标偏好，实现可调节的目标导向行为。

**🔧 技术方法**

核心技术包括：主动推理（Active Inference）与 POMDP、分布式生成模型、期望自由能最小化、流式机器学习（Streaming Random Patches）以及离散贝叶斯网络。

**📊 数据集**

数据集采用基于 Cournot 竞争的仿真环境，包含多阶段需求变化、价格波动以及库存信号，用以模拟动态市场情境。

**📈 对比分析**

通过与传统 Nash 均衡结果和不同精度/模型设定的对比，实验显示代理在稳定环境下能快速收敛到近似最优策略，且多代理系统在人数增加时表现更为稳定；但在高度概念漂移或精度不匹配时性能下降。

**⚠️ 局限性**

局限性包括：① 对生成模型的先验参数高度依赖，误设会导致收敛慢或不稳定；② 流式学习模型的更新速率和阈值需人工设定；③ 仅在离散、有限状态空间下验证，未验证连续或高维实际工业场景的可扩展性。

---

## 429. Sorting under Partial Information with Optimal Preprocessing Time via Unified Bound Heaps

**arXiv ID:** 2604.12653 | [PDF](https://arxiv.org/pdf/2604.12653v1)

**作者:** Daniel Rutschmann `[一作]` `[通讯]` (Institute of Science and Technology Austria), Daniel Rutschmann (Institute of Science and Technology Austria)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

设计并实现了一个满足统一界限的新堆数据结构，并将其用于排序在部分信息下的算法，达到O(m)预处理与O(log e(G))排序的最优界限。

**💡 创新点**

创新点在于提出统一界限(heap unified bound)概念并构造相应堆，实现了在仅用顶点数和边数预处理的前提下实现最优排序时间，填补了该问题的空白。

**🔧 技术方法**

技术上结合拓扑排序、区间引理、递增指数级索引树以及细致的势能/计数分析来实现该堆及其算法。

**📊 数据集**

论文为理论算法，未使用具体数据集，仅在理论模型中进行分析。

**📈 对比分析**

与以往 O(n^ω) 或 O(m) + O(log e(G)+m) 的方法相比，本算法在预处理时间上保持 O(m)，而排序时间从 O(log e(G)+n) 降到 O(log e(G))，实现了理论上最优性能。

**⚠️ 局限性**

局限性在于堆结构实现复杂且未给出可直接实现的代码，且实际性能仍需实验验证，另外该方法仍假设可在单机 Word RAM 环境下运行。

---

## 430. PromptEcho: Annotation-Free Reward from Vision-Language Models for Text-to-Image Reinforcement Learning

**arXiv ID:** 2604.12652 | [PDF](https://arxiv.org/pdf/2604.12652v1)

**作者:** Jinlong Liu `[一作]` (Alibaba Group), Pipei Huang `[通讯]` (Alibaba Group)

**通讯引用:** 1659 | [OpenAlex ID](https://openalex.org/A5059615376)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出 PromptEcho，利用冻结的 VLM 的 token‑level 交叉熵损失作为无注释、无训练的奖励信号，改进文本到图像（T2I）模型的强化学习训练；

**💡 创新点**

创新点在于直接从 VLM 预训练目标提取奖励，完全不需要人工标注或奖励模型微调，奖励质量随更强的开源 VLM 自然提升；

**🔧 技术方法**

技术主要包括：冻结 VLM（如 Qwen3‑VL‑32B）进行单次前向推理并计算交叉熵；基于 AWM 的 RL 框架；使用 DenseAlignBench、GenEval、DPG‑Bench、TIIFBench 进行评估；

**📊 数据集**

使用的数据集包括：内部 100K 高质量图像 + Qwen3‑VL‑32B 生成的 200‑400 字详细描述；DenseAlignBench 的 2000 条测试句子；公开基准 GenEval、DPG‑Bench、TIIFBench；以及内部电商海报数据用于文本渲染实验；

**📈 对比分析**

与基线模型及 InferScore 对比，DenseAlignBench 获得约 +26.8pp 的净优势；在 GenEval 上提升 6.5pp、DPG‑Bench 1.02pp、TIIFBench 3.6/5.8pp；奖励模型规模 32B 对比 8B 时，32B 提升更明显；文本渲染任务提升 7pp；

**⚠️ 局限性**

局限性包括：在 VLM 不了解的专业视觉领域效果受限；无法利用 chain‑of‑thought 推理能力；对极细粒度视觉细节或复杂语义判断的奖励仍有限。

---

## 431. Learning Chain Of Thoughts Prompts for Predicting Entities, Relations, and even Literals on Knowledge Graphs

**arXiv ID:** 2604.12651 | [PDF](https://arxiv.org/pdf/2604.12651v1)

**作者:** Alkid Baci `[一作]` (Paderborn University), Axel-Cyrille Ngonga Ngomo `[通讯]` (Paderborn University)

**通讯引用:** 9229 | [OpenAlex ID](https://openalex.org/A5038745720)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

将知识图谱链路预测重新表述为基于字符串的链式思维提示（CoT）评分函数，利用大型语言模型在无梯度的贝叶斯优化下，学习高质量提示，实现在未见实体、关系和数值上的预测，并可生成推断三元组用于图谱扩增。

**💡 创新点**

① 把连续向量参数化的评分函数改为可学习的字符串提示；② 用无梯度贝叶斯优化（MIPRO）在少量示例下自动搜索最佳提示；③ 直接通过提示实现数值预测和OWL实例检索；④ 生成高质量推断三元组提升传统KGE模型。

**🔧 技术方法**

使用大型语言模型（如Qwen2.5-32B-Instruct）+链式思维提示学习；MIPRO贝叶斯优化实现无梯度提示搜索；DSPy框架处理OWL实例检索；数值预测采用上下文检索；对比 Manchester 与 DL 语法。

**📊 数据集**

Countries-S1/S2/S3（地理数据集）；LitWD1K（包含数值子集）；Father（OWL实例检索基准）；并与 DistMult、ComplEx 等传统 KGE 基准进行对比。

**📈 对比分析**

与最先进 KGE 模型比较，MIPRO 优化后在 Countries 数据集 MRR@1 达到 1.000，显著优于传统模型；数值预测平均误差低、ICR>0.75；OWL 实例检索中 Jaccard 相似度在 Manchester 语法+完整 IRI 组合下最高，表现优异。

**⚠️ 局限性**

对极端离群值敏感；需要将实体/关系 ID 转为可读文本；依赖大型 LLM，推理时间与内存受限；上下文 token 限制难以处理大规模图谱。

---

## 432. Detecting and refurbishing ground truth errors during training of deep learning-based echocardiography segmentation models

**arXiv ID:** 2604.12832 | [PDF](https://arxiv.org/pdf/2604.12832v1)

**作者:** Iman Islam `[一作]` (King's College London), Andrew P. King `[通讯]` (King's College London)

**通讯引用:** 7395 | [OpenAlex ID](https://openalex.org/A5033570737)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `7b0f05dc-d396-4b03-96d2-a379dbd5049d`

**🎯 论文内容**

研究了深度学习模型在心脏超声图像分割中对标注错误的鲁棒性，并提出了基于梯度方差（VOG）的错误检测与伪标签修复的自动化流程。

**💡 创新点**

首次系统评估多类心脏超声分割模型对随机与系统标注错误的影响，并提出使用梯度方差作为鲁棒错误检测指标，结合伪标签平均实现在线修复。

**🔧 技术方法**

使用U‑Net分割网络、交叉熵损失、Adam优化、梯度方差VOG检测以及伪标签平均（SELFIE风格）与样本重加权/过滤等技术。

**📊 数据集**

在CAMUS 2D超声A4C视角数据集（500张图像，标注左心室、心肌和左心房）上进行实验。

**📈 对比分析**

与传统基于损失的错误检测方法比较，VOG在准确率、敏感度和特异度上更优；在随机错误低于50%和系统错误低于25%时，修复后Dice分数提升；基线U‑Net对随机错误鲁棒，系统错误时修复略有提升。

**⚠️ 局限性**

实验仅限单一数据集与单一网络，错误类型为三种合成错误，未覆盖真实临床标注误差比例，且在极高错误率下修复效果有限。

---

## 433. Detecting and Enhancing Intellectual Humility in Online Political Discourse

**arXiv ID:** 2604.12821 | [PDF](https://arxiv.org/pdf/2604.12821v1)

**作者:** Samantha D'Alonzo `[一作]` (Northeastern University), Nabeel Gillani `[通讯]` (Northeastern University)

**通讯引用:** 578 | [OpenAlex ID](https://openalex.org/A5015797850)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了IH/IA代码本并训练零样本LLM分类器，随后在Reddit政治子版块上进行观察性分析和实验室随机对照实验，探讨IH的传播与干预效果。

**💡 创新点**

首次系统地将智识谦逊的量化、测评和干预融入在线政治讨论，揭示了IH的“传染效应”并证明了可通过提示提高IH水平。

**🔧 技术方法**

采用GPT‑4.1零样本提示式分类、PerspectiveAPI基线、Logistic回归与配对t检验等技术，对IH/IA进行自动标注与因果效应估计。

**📊 数据集**

使用359条手工标注的Reddit帖子作为金标准，配合2024年3‑5月10个政治子版块共68k+评论以及355名实验参与者的数据。

**📈 对比分析**

分类器在F1上达0.74（高于PerspectiveAPI 0.53与随机基线0.47），RCT显示社交提示显著提升IH（效应≈0.25）且未降低参与度，效果稳定。

**⚠️ 局限性**

金标准样本量小、标注一致性低、实验场景为模拟环境、硬性分类缺乏连续度，限制了结果的外推与精细化分析。

---

## 434. Rethinking Satellite Image Restoration for Onboard AI: A Lightweight Learning-Based Approach

**arXiv ID:** 2604.12807 | [PDF](https://arxiv.org/pdf/2604.12807v1)

**作者:** Adrien Dorise `[一作]` (Institut de Recherche Technologique Saint Exupéry), Omar Hlimi `[通讯]` (Institut de Recherche Technologique Saint Exupéry)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出一种轻量级残差卷积网络 ConvBEERS，用于卫星影像的辐射恢复，并验证其在真实与模拟数据上的效果。

**💡 创新点**

创新点在于：①仅使用物理驱动的仿真影像进行无监督训练；②设计无生成式、无批归一化的残差结构，保证恢复稳定且保持图像物理一致性；③通过频域和感知损失共同监督，实现高保真度恢复；④在 FPGA 上实现高效部署，显著降低延迟。

**🔧 技术方法**

技术手段包括：基于 EDSR 的残差网络架构、3×3 卷积、16 个残差块、单通道回归层；训练采用联合 L1、LPIPS、FFT 频域损失；物理仿真器生成 MTF、噪声、下采样等降质；部署使用 Xilinx Versal VCK190 DPU，INT8 量化。

**📊 数据集**

使用的数据集：OpenAerialMap（模拟降质后用于训练与验证）；真实 Pleiades‑HR 原始影像与 L1 影像；DIOR 目标检测数据集（用于评估恢复后对检测的提升）。

**📈 对比分析**

对比方法为传统光学降质修复流水线（基于 PSF 去卷积 + NL‑Bayes 去噪）；性能指标为 PSNR、SSIM、LPIPS、DISTS、MTF、SNR、检测 mAP@50/90；结果显示 ConvBEERS 在 PSNR 上提升约 +6.9 dB，SSIM 上提升，LPIPS 降低；在 DIOR 上恢复影像使 mAP@50 由原始降质时下降 2.6% 提升至 95% 以上；在 FPGA 上实现的推理时间比传统流水线快约 41×，帧率提升至 7.2 FPS。

**⚠️ 局限性**

局限性：仅处理辐射失真，未包含几何校正；训练仅基于仿真数据，可能在极端真实条件下表现下降；对高阶噪声模型或光学失配的鲁棒性尚未验证；需要与几何恢复模块结合以实现完整的原始影像提升。

---

## 435. Image-to-Image Translation Framework Embedded with Rotation Symmetry Priors

**arXiv ID:** 2604.12805 | [PDF](https://arxiv.org/pdf/2604.12805v1)

**作者:** Feiyu Tan `[一作]` (Xi'an Jiaotong University), Deyu Meng `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 32123 | [OpenAlex ID](https://openalex.org/A5091017287)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

本文提出了基于旋转对称先验的图像到图像翻译框架，并设计了一种可学习变换的等变卷积（TL‑Conv），实现网络对数据对称性的自适应学习；

**💡 创新点**

创新点在于①首次将旋转群等变卷积嵌入I2I网络，保证整个网络过程保持严格旋转等变性；②引入可学习变换组，使等变卷积能够适应非严格旋转对称的数据，并给出等变误差的理论上界；

**🔧 技术方法**

使用技术包括旋转等变卷积（EQ‑CNN）、可学习变换等变卷积（TL‑Conv）以及基函数滤波参数化；在实验中将这些模块嵌入无监督I2I模型（CycleGAN、CUT、santa）和图像恢复模型（RCDNet、SIRR等）；

**📊 数据集**

实验数据集涵盖DIV2K、summer2winter_yosemite、old2young、BDD100K、iphone2dslr、BraTS 2019、多种恢复数据集（Rain100L、Set5/Set14/BSD100/Urban100等）；

**📈 对比分析**

与传统CNN及严格旋转等变版本比较，TLEQ‑santa在未配对I2I任务中实现最低FID；TLEQ‑HyperGAN在多模态MRI翻译中取得最优MAE/PSNR/SSIM；在雨去除、去噪、超分辨率等恢复任务中，TL‑Conv相较于其他等变卷积和CNN均提升PSNR/SSIM，表现最优；

**⚠️ 局限性**

局限性包括：训练时间和模型复杂度相对增加；目前仅在CNN架构上验证，未扩展到Transformer等新型网络；实例级变换学习尚未实现，可能进一步提升自适应性能。

---

## 436. Efficiency of Proportional Mechanisms in Online Auto-Bidding Advertising

**arXiv ID:** 2604.12799 | [PDF](https://arxiv.org/pdf/2604.12799v1)

**作者:** Nguyen Kim Thang `[一作]` `[通讯]` (University Grenoble-Alpes), Nguyen Kim Thang (University Grenoble-Alpes)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

论文研究了在线广告中的比例机制（Kelly机制），并分析了其纯纳什均衡下的液体福利效率。

**💡 创新点**

创新点在于证明标准比例机制的价格无政府状态（PoA）上界为2并且可达；更重要的是提出一种改进的支付方案，使得PoA可递减至 1 + O(1/(n-1))，随参与者数增大趋近完全效率。

**🔧 技术方法**

作者利用线性规划对偶性与凸优化的 Karush‑Kuhn‑Tucker（KKT）条件，构造可行的对偶解来上界优化目标，从而推导 PoA 边界。

**📊 数据集**

论文为纯理论分析，不依赖具体数据集，所有结果均基于一般连续、可微、凹的价值函数与预算约束。

**📈 对比分析**

相较于已有工作中普遍存在的 PoA 上限 2，改进机制显著降低了无政府状态下的效率损失，理论上可达到接近 1 的效率，证明了在人数足够多时几乎无损失。

**⚠️ 局限性**

主要限制包括：需要对价值函数假设为凹性；改进机制在实际部署时要求支付可能超过提交的出价，且需要已知全局预算上界才能保证支付不超过出价；在预算或 RoS 约束极端情况下，效果可能不如预期。

---

## 437. LIFE -- an energy efficient advanced continual learning agentic AI framework for frontier systems

**arXiv ID:** 2604.12874 | [PDF](https://arxiv.org/pdf/2604.12874v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 438. VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization

**arXiv ID:** 2604.12887 | [PDF](https://arxiv.org/pdf/2604.12887v1)

**作者:** Andrei Atanov `[一作]` (Apple), Amir Zamir `[通讯]` (Swiss Federal Institute of Technology Lausanne)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种名为 VideoFlexTok 的可变长度粗细分层视频分词器，并在此基础上构建了可从任意数量分词重建真实视频的生成流解码器，进而实现高效的文本到视频和类别到视频生成。

**💡 创新点**

创新点：①通过嵌入可学习的注册分词并使用嵌套丢弃（nested dropout）自发形成粗细分层结构；②首层分词天然聚焦语义、几何与运动信息，后层细化像素细节；③在 VAE 潜在空间中训练，显著降低解码器计算成本；④加入基于 DINOv2 的语义偏置（REPA）提升分词的语义可解释性。

**🔧 技术方法**

主要技术：VAE 预编码、时间因果 Transformer（编码器/解码器）、嵌套丢弃、生成修正流（rectified flow）解码器、REPA 语义损失、GPT‑style 自回归生成器（用于类别和文本条件）。

**📊 数据集**

使用的数据集：Kinetics‑600（类别到视频）、Panda70M（文本到视频）以及 10 秒 81 帧长视频数据，分辨率分别为 128×128 与 256×256。

**📈 对比分析**

比较方法：与 3D grid 令牌器、VidTok、Cosmos‑DV、Omnitokenizer、LARP 等基线对比。实验显示：在相同或更低的 FLOPs、参数（1.1B vs 5.2B）和 token 数（5–10 倍减少）下，VideoFlexTok 在 gFVD、ViCLIP、Cls 评分上相当或更优；对于 10 秒长视频，仅需 672 token 而传统 3D grid 需要 5376 token，显著提升计算效率。

**⚠️ 局限性**

局限性：①仍需先训练 VAE 和流解码器，增加整体模型链长；②极少分词时可能失去细节；③目前仅在 Kinetics‑600 与 Panda70M 上评估，泛化性待进一步验证；④长视频推理时可能需要更多 denoising 步骤，推理成本仍不 negligible。

---

## 439. An abstract model of nonrandom, non-Lamarckian mutation in evolution using a multivariate estimation-of-distribution algorithm

**arXiv ID:** 2604.12884 | [PDF](https://arxiv.org/pdf/2604.12884v1)

**作者:** Liudmyla Vasylenko `[一作]` (University of Haifa), Adi Livnat `[通讯]` (University of Haifa)

**通讯引用:** 594 | [OpenAlex ID](https://openalex.org/A5000795098)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过构建基于受限玻尔兹曼机的估计分布算法模型，模拟了非随机、非拉马克主义的突变机制，并与传统随机突变与自然选择模型进行比较。

**💡 创新点**

创新点在于将内部信息聚合（fan‑in）引入突变过程，提供了与传统随机突变和拉马克主义截然不同的非随机突变理论——交互驱动进化（IBE），并用机器学习方式实现。

**🔧 技术方法**

使用了受限玻尔兹曼机（RBM）与估计分布算法（EDA）框架，并通过对比随机突变与自然选择的标准遗传算法实现。

**📊 数据集**

采用了均匀随机 MAX‑SAT 生成的实例（k=3、不同变量数）以及偶数异或（even‑parity）布尔函数作为适应度评估数据集。

**📈 对比分析**

通过将 RBM‑EDA 与随机突变模型（RM）以及基准的最佳随机猜测（BRG）进行多轮实验，结果显示 RBM‑EDA 在多样本规模和问题规模上在数百代后能显著超过 RM，获得更高的平均适应度。

**⚠️ 局限性**

主要局限在于模型极度抽象，突变机制非生物学真实；缺乏可演化的突变器；起点从随机基因组出发不符合生物学过程；无法直接映射到实际基因组水平。

---

## 440. AISafetyBenchExplorer: A Metric-Aware Catalogue of AI Safety Benchmarks Reveals Fragmented Measurement and Weak Benchmark Governance

**arXiv ID:** 2604.12875 | [PDF](https://arxiv.org/pdf/2604.12875v1)

**作者:** Abiodun A. Solanke `[一作]` `[通讯]`, Abiodun A. Solanke

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 AISafetyBenchExplorer，收录并系统化整理了 195 个 AI 安全基准的 benchmark‑level 和 metric‑level 元数据，形成多表结构化工作簿。

**💡 创新点**

创新点在于提出受控词汇表和四层复杂度分类（Popular、High、Medium、Low），以及对指标碰撞（Metric Collision）和测量碎片化的全景性诊断，首次让不同基准间的度量可直接对比和审计。

**🔧 技术方法**

使用系统化的元数据抽取脚本、受控词汇表、手工审核的决策流程，并在 Excel/数据库环境中实现多表工作簿和可视化仪表盘。

**📊 数据集**

数据集来源于 2018–2026 年公开发布的 195 个 AI 安全基准及其对应的 GitHub 仓库、Hugging Face 数据集、论文引用和公开评测指标，覆盖英文主流评测。

**📈 对比分析**

通过对比同名指标在不同基准中的具体定义、评判者、聚合规则和威胁模型，揭示同一指标标签下的操作差异，使得跨基准直接数值比较被证伪；在此基础上可构造跨基准指标标准化映射，提升比较可信度。

**⚠️ 局限性**

局限性包括：对完整 195 个基准的复杂度分类仍需重新评估；缺乏完整程序化计数所有指标碰撞实例，导致对碎片化程度的量化仍不充分。

---

## 441. Artificial Intelligence for Modeling and Simulation of Mixed Automated and Human Traffic

**arXiv ID:** 2604.12857 | [PDF](https://arxiv.org/pdf/2604.12857v1)

**作者:** Saeed Rahmani `[一作]` (Delft University of Technology), Simeon C. Calvert `[通讯]` (Delft University of Technology)

**通讯引用:** 1422 | [OpenAlex ID](https://openalex.org/A5052655645)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ba576bd1-e51d-44e8-8077-fc943b333c93` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `c773407a-6119-4871-b8b3-1e7ae17a6851` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

对混合自动与人类交通仿真中人工智能方法进行系统综述与统一分类，梳理从单体行为模型到环境级生成模型及认知/物理驱动模型的全景视图；

**💡 创新点**

提出了三维结构化的 AI 分类法（单体/多体、环境级/生成、认知/物理驱动），覆盖了从单体学习到多体互动、世界模型、生成式情境建模以及人因与物理融合的技术；

**🔧 技术方法**

主要技术包括行为克隆、强化学习、逆强化学习、对抗式模仿、Diffusion 与 Transformer 生成模型、多智能体强化学习、自主博弈理论、基础模型驱动等；

**📊 数据集**

综述涵盖的公开数据集与基准包括 Waymo Open Motion Dataset、Argoverse、nuPlan、Waymo Open Sim Agents Challenge、NuScenes、NDD 等；

**📈 对比分析**

通过对比不同方法在公开基准上的指标（如 ADE/FDE、NDD、碰撞率、可解释性等），展示了各类方法的优缺点与性能趋势；

**⚠️ 局限性**

局限性在于缺乏统一的闭环评估框架、对最新的 LLM/基础模型应用尚未完整覆盖、对实时计算与资源约束的讨论不足，以及难以在单一基准上全面评估多模态交互与因果一致性。

---

## 442. PianoFlow: Music-Aware Streaming Piano Motion Generation with Bimanual Coordination

**arXiv ID:** 2604.12856 | [PDF](https://arxiv.org/pdf/2604.12856v1)

**作者:** Xuan Wang `[一作]` (Zhejiang University), Gaoang Wang `[通讯]` (Zhejiang University)

**通讯引用:** 2338 | [OpenAlex ID](https://openalex.org/A5089410490)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `a8e75ba4-7a2d-4153-b003-06c94533add0` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了PianoFlow框架，实现了仅凭音频即可生成实时双手钢琴演奏的三维运动；

**💡 创新点**

创新点包括：①利用MIDI作为特权模态进行跨模态蒸馏，将音乐结构知识注入音频表示；②设计了非对称角色门控交互（ARGI）模块，在特征瓶颈处动态捕捉双手角色变化；③引入自回归流式延续（AFC）机制，实现任意长度流式实时生成；

**🔧 技术方法**

技术手段包括MuQ音频编码、Harmonic Perceiver对MIDI的 octave/interval 处理、Transformer+U‑Net结构、流匹配（conditional flow matching）与ODE求解、门控交互、交叉注意力与时间门控；

**📊 数据集**

使用PianoMotion10M数据集（约116小时、30FPS，包含音频、MIDI和MANO手姿），进行训练与评估；

**📈 对比分析**

与EmoTalk、LivelySpeaker、PianoMotion、S2C等基线在FID、FGD、WGD、PD、Smoothness、RTF、FDE等指标上进行对比，PianoFlow在大多数指标上均优于基线，实时因子RTF仅0.186，速度提升约9倍；

**⚠️ 局限性**

局限性：仍主要针对钢琴，难以直接迁移到其他乐器；在极长序列的长时域一致性与高频细节捕捉方面存在轻微失真；模型依赖于MuQ与MIDI预训练，训练成本较高。

---

## 443. PAINT: Partner-Agnostic Intent-Aware Cooperative Transport with Legged Robots

**arXiv ID:** 2604.12852 | [PDF](https://arxiv.org/pdf/2604.12852v1)

**作者:** Zhihao Cao `[一作]` (ETH Zurich), Marco Hutter `[通讯]` (ETH Zurich)

**通讯引用:** 20952 | [OpenAlex ID](https://openalex.org/A5044258783)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

研发了PAINT框架，利用关节自感知推断伙伴意图，实现轻量化、无力/扭矩传感器的协作运输；

**💡 创新点**

创新点包括：① 将意图推断与地形鲁棒行走分离；② 采用教师-学生迁移与意图估计网络，仅用本体关节感知；③ 可无缝迁移至多机器人分布式团队与不同机器人形态；

**🔧 技术方法**

技术手段为层次化强化学习、教师-学生KL正则化、意图估计网络、预训练低层行走骨干、Isaac Gym仿真与域随机化、梯度显著性分析；

**📊 数据集**

数据来源为Isaac Gym生成的仿真环境，随机施加平面力/转矩交互，载荷质量随机0–10kg；真实实验采用多种地形、载荷形状及人类/机器人伙伴；

**📈 对比分析**

与基线Damped Arm、Pure RL-4、BC Distillation-4、Force Estimator-4对比，PAINT在轨迹误差、意图对齐、约束力/转矩低等指标上显著优越，且在多载荷、多机器人、多形态实验中保持鲁棒性能；

**⚠️ 局限性**

局限性包括：缺乏主动避障能力，遇障碍会导致碰撞；仅处理平面意图，无法应对复杂6D交互；臂运动受限，难以完成高难度姿态或转弯等操作。

---

## 444. A sequential explanatory mixed-methods study on the acceptance of a social robot for EFL speaking practice among Chinese primary school students: Insights from the Computers Are Social Actors (CASA) paradigm

**arXiv ID:** 2604.12789 | [PDF](https://arxiv.org/pdf/2604.12789v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 445. EXTree: Towards Supporting Explainability in Attribute-based Access Control

**arXiv ID:** 2604.12850 | [PDF](https://arxiv.org/pdf/2604.12850v1)

**作者:** Shanampudi Pranaya Chowdary `[一作]` (Indian Institute of Technology Kharagpur), Shamik Sural `[通讯]` (Indian Institute of Technology Kharagpur)

**通讯引用:** 5402 | [OpenAlex ID](https://openalex.org/A5036571304)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了 EXTree，一种在属性基础访问控制中既能高效评估又能提供可操作的拒绝解释的树形结构。

**💡 创新点**

创新点在于将属性可变性与可见性信息嵌入树结构，并通过受限的局部搜索产生最小成本、可执行的反馈。

**🔧 技术方法**

使用了层次化策略树（PolTree 变体）、信息熵/可变性分裂准则、可见性约束函数、基于深度/可变性优先的搜索算法。

**📊 数据集**

使用了两个合成 ABAC 数据集（Synthetic‑1、Synthetic‑2）以及真实 Healthcare ABACLab 数据集进行评估。

**📈 对比分析**

对树构建准则、反馈策略及可见性约束进行对比实验，结果表明高成本优先树配合变化优先搜索在决策延迟 < 2 ms、反馈成本最低、覆盖率高；熵分裂树虽然评估快但反馈成本略高。

**⚠️ 局限性**

局限性包括：仅支持允许策略；树结构非增量更新；可见性处理为二值化；未覆盖复杂 XACML 组合规则；在极高属性关联度场景下可解释性可能下降。

---

## 446. GGD-SLAM: Monocular 3DGS SLAM Powered by Generalizable Motion Model for Dynamic Environments

**arXiv ID:** 2604.12837 | [PDF](https://arxiv.org/pdf/2604.12837v1)

**作者:** Yi Liu `[一作]` (Tsinghua University), Houde Liu `[通讯]` (Tsinghua University)

**通讯引用:** 1738 | [OpenAlex ID](https://openalex.org/A5076885280)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了 GGD‑SLAM 框架，利用可迁移运动模型在无语义标签或深度输入的条件下实现动态环境下的定位与高精度稠密重建。

**💡 创新点**

创新点包括：① FIFO 队列与时序注意力的通用运动模型；② 动态注意力增强器与掩模二值化；③ 基于 KD‑tree 的背景填充与动态可适应 SSIM 损失，显著提升了动态场景的鲁棒性。

**🔧 技术方法**

技术手段涵盖：3D Gaussian Splatting、DINOv2 视觉特征提取、顺序注意力机制、动态/静态增强头、Otsu 二值化、KD‑tree 采样、SSIM 自适应损失、DROID‑SLAM 的 DBA、Metric3D 估计以及蒙版融合与可视化。

**📊 数据集**

使用的训练集为 Davis（用于通用运动模型训练），评估数据集包括 TUM RGB‑D、Bonn RGB‑D Dynamic 以及 Wild‑SLAM Dataset。

**📈 对比分析**

与 WildGS‑SLAM、MonoGS、Splatam、DG‑SLAM、DyPho‑SLAM 等 SOTA 方法在 ATE、PSNR、SSIM、LPIPS 等指标上进行对比，实验表明 GGD‑SLAM 在动态序列中取得了最低 ATE、最高 PSNR/SSIM，性能明显优于现有方法。

**⚠️ 局限性**

局限性：尚未实现实时动态物体运动重建与完全遮挡区的补全，依赖单目深度估计，且在极端遮挡或高速运动场景下精度可能略有下降。

---

## 447. DPC-VQA: Decoupling Quality Perception and Residual Calibration for Video Quality Assessment

**arXiv ID:** 2604.12813 | [PDF](https://arxiv.org/pdf/2604.12813v1)

**作者:** Xinyue Li `[一作]` (Shanghai Jiao Tong University), Guangtao Zhai `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21947 | [OpenAlex ID](https://openalex.org/A5064168853)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了DPC‑VQA框架，利用冻结的多模态大型语言模型（MLLM）生成基准质量估计，并通过轻量级残差校准分支对目标视频质量进行高效适配。

**💡 创新点**

创新点在于将质量感知与残差校准解耦，仅训练极少量参数，充分利用MLLM的先验知识实现低成本、少样本的视频质量评估。

**🔧 技术方法**

采用冻结的Qwen3‑VL‑8B作为感知模块，SlowFast作为辅助特征提取器，投影层对齐特征维度，学习查询聚合残差信息，并通过残差正则化和Smooth L1回归进行训练。

**📊 数据集**

使用了UGC基准集 KoNViD‑1k、YouTube‑UGC、LIVE‑VQC 以及 AIGC 基准集 T2VQA、Human‑AGVQA。

**📈 对比分析**

在五个基准上与监督与少样本 VQA 方法进行对比，DPC‑VQA 在 SRCC/PLCC 上取得领先或竞争性成绩，仅使用 2% 可训练参数、20% MOS 标注即可达到顶尖水平。

**⚠️ 局限性**

局限性包括仍需依赖大规模冻结 MLLM 的前向推理，推理速度受限；对极端分布迁移的鲁棒性尚未充分验证；以及在极低标注量或高度自定义场景下的适配效果需进一步探索。

---

## 448. OSC: Hardware Efficient W4A4 Quantization via Outlier Separation in Channel Dimension

**arXiv ID:** 2604.12782 | [PDF](https://arxiv.org/pdf/2604.12782v1)

**作者:** Zhiyuan Zhang `[一作]` (Huawei Technology), Hui Wang `[通讯]` (Huawei Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个硬件友好的离线查表式激活异常抑制框架OSC，结合双路径4位/16位混合精度实现低精度推理时的异常抑制。

**💡 创新点**

创新点在于发现激活异常在各层具有“token持久结构聚类”，并利用该结构预先索引固定通道，形成静态抑制与FP8回退的混合策略。

**🔧 技术方法**

采用离线组化查表、双路径GEMM、结构化子张量提取、MXFP4/FP8微缩放以及动态回退策略。

**📊 数据集**

使用Qwen3-8B、Qwen3-30B-A3B模型以及Pile、MMLU、GSM8K、ARC等公开基准数据集进行评估。

**📈 对比分析**

与8/4位直接量化、动态TopK/TopP、SmoothQuant等方法相比，OSC在保持接近16位精度的前提下，平均准确率下降仅2.19/1.12点，并在现代AI加速器上实现1.64×–1.78×的速度提升。

**⚠️ 局限性**

局限性包括仅在层级粗粒度回退、未对实际物理部署的全链路延迟做评估，以及对不同模型/硬件的泛化仍待验证。

---

## 449. FastGrasp: Learning-based Whole-body Control method for Fast Dexterous Grasping with Mobile Manipulators

**arXiv ID:** 2604.12879 | [PDF](https://arxiv.org/pdf/2604.12879v1)

**作者:** Heng Tao `[一作]` (ShanghaiTech University), Yuexin Ma `[通讯]` (ShanghaiTech University)

**通讯引用:** 4222 | [OpenAlex ID](https://openalex.org/A5102015139)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出FastGrasp框架，融合抓取指导、全身协同控制和触觉反馈，通过两阶段策略生成抓取候选并执行高速度协调运动，实现移动机器人高速抓取任务。

**💡 创新点**

创新点在于：①两阶段强化学习与条件变分自编码器相结合，先生成多样抓取候选后再执行最优抓取；②全身协同控制（底盘、机械臂、多指手）实现高速动态抓取；③实时触觉反馈用于调节抓取，提升鲁棒性与仿真-真实迁移效果。

**🔧 技术方法**

采用条件变分自编码器(CVAE)生成抓取候选，Proximal Policy Optimization(PPO)训练全身控制策略，二值触觉传感器实时反馈，低通滤波、域随机化等技术实现高频控制与仿真-真实迁移。

**📊 数据集**

使用合成抓取数据集478,200个有效抓取姿态，真实测试集包含418个物体（60易、36难）以及16个真实物体，点云由PointNet提取。

**📈 对比分析**

与全点云/部分点云基线（一次式、两阶段力向量选择、移动抓取）对比，模拟平均成功率超过60%，真实高速下32%（低速34.6%），并通过AB实验验证触觉反馈和抓取指导选择显著提升性能。

**⚠️ 局限性**

限制在于难以抓取平坦物体，移动高速时安全性不足，仿真-真实差距仍存在，需要更完善的动态安全机制和对复杂环境碰撞的鲁棒性。

---

## 450. OVAL: Open-Vocabulary Augmented Memory Model for Lifelong Object Goal Navigation

**arXiv ID:** 2604.12872 | [PDF](https://arxiv.org/pdf/2604.12872v1)

**作者:** Jiahua Pei `[一作]` (Tsinghua University), Xueqian Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5477 | [OpenAlex ID](https://openalex.org/A5100737125)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种终身开放词汇对象导航框架 OVAL，利用结构化内存描述符和概率前沿选择实现长时记忆与高效探索。

**💡 创新点**

创新点在于：①使用视觉特征和位置、场景描述符代替原始文本标签构建可扩展的内存模型；②设计基于距离、语义与足迹的多值概率前沿评分策略；③通过 KMP 子串搜索和 LLM 验证提升目标识别鲁棒性。

**🔧 技术方法**

核心技术包括 Grounded‑SAM2 自动标注、CLIP/BLIP 视觉‑语言特征、SuperGlue 匹配、KMP 子串搜索、LLM（GPT‑4o）验证、基于 Gaussian 的概率前沿评估。

**📊 数据集**

使用 Habitat 交互式模拟器中的 HM3D 和 MP3D 数据集，并通过重新排序与状态保留生成终身导航数据集。

**📈 对比分析**

在 1000 轮终身 ObjectNav 任务中，OVAL 在 HM3D 的成功率 (SR) 68.1%/SPL 33.8% 及 MP3D 的 SR 44.1%/SPL 18.6% 均优于 VLFM 和 GOAT；在单一 ObjectNav 任务中，OVAL 也实现了最优或接近最优的 SR/SPL。

**⚠️ 局限性**

限制主要包括：仅支持单模态视觉输入，无法处理多模态或高动态场景；记忆模型在面对大量重复对象时可能导致存储膨胀；LLM 验证存在计算延迟。

---

## 451. Evolving the Complete Muscle: Efficient Morphology-Control Co-design for Musculoskeletal Locomotion

**arXiv ID:** 2604.12855 | [PDF](https://arxiv.org/pdf/2604.12855v1)

**作者:** Lidong Sun `[一作]` (Naval University of Engineering), Fuchun Sun `[通讯]` (Tsinghua University)

**通讯引用:** 23281 | [OpenAlex ID](https://openalex.org/A5055546056)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

在仿真中共进化肌肉力量、速度和刚度参数，利用谱设计进化框架提升机器人在四种地形上的行走稳定性和效率。

**💡 创新点**

将三维肌肉生理参数映射到低维谱空间，并加入双侧对称先验，显著降低高维搜索空间，实现高效共进化。

**🔧 技术方法**

PCA谱映射、双侧对称先验、两阶段设计-控制共进化、PPO强化学习、MuJoCo+MyoSuite仿真。

**📊 数据集**

MyoSuite中的MyoLeg模型，结合四种地形（平地、崎岖、山坡、楼梯）生成的仿真数据。

**📈 对比分析**

与固定形态和Transform2Act基线对比，SDE在收敛速度和累计奖励上均优于两者；完整三参数共进化显著提升稳定性，单参数优化效果不佳。

**⚠️ 局限性**

仅在仿真环境验证，缺乏真实机器人实验；谱空间可能忽略其他生理特征；设计空间固定拓扑，无法实现实时跨任务形态切换。

---

## 452. Growing Pains: Extensible and Efficient LLM Benchmarking Via Fixed Parameter Calibration

**arXiv ID:** 2604.12843 | [PDF](https://arxiv.org/pdf/2604.12843v1)

**作者:** Eliya Habba `[一作]` (Hebrew University of Jerusalem), Gabriel Stanovsky `[通讯]` (Hebrew University of Jerusalem)

**通讯引用:** 2006 | [OpenAlex ID](https://openalex.org/A5082136238)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于多维IRT的固定参数校准框架，通过锚点维持评测可比性，实现LLM新数据集与模型的增量评估。

**💡 创新点**

将心理测验的锚点等价方法应用于LLM评测，允许随时间加入新基准而不需重新评估历史模型。

**🔧 技术方法**

使用多维2PL IRT、固定参数校准、锚点选择及分层评测技术。

**📊 数据集**

使用Open LLM Leaderboard（6个数据集）和MMLU（57个子域）数据集。

**📈 对比分析**

与并行校准和随机抽样对比，固定参数校准在每个数据集仅用100个锚点时MAE仅2–3%，Spearmanρ≥0.9，成本保持不变。

**⚠️ 局限性**

仅适用于二元响应、英语知识推理；未验证其他语言/任务；锚点可能随模型演进失效；需要数十个参考模型。

---

## 453. Loop Corrections to the Training and Generalization Errors of Random Feature Models

**arXiv ID:** 2604.12827 | [PDF](https://arxiv.org/pdf/2604.12827v1)

**作者:** Taeyoung Kim `[一作]` (Korea Institute for Advanced Study), Taeyoung Kim `[通讯]` (Korea Institute for Advanced Study)

**通讯引用:** 775 | [OpenAlex ID](https://openalex.org/A5100412370)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文通过有效场论框架，针对随机特征模型（Random Feature Models）中的有限宽度效应，推导了训练误差、测试误差和泛化误差的环（loop）展开，并给出了一阶（即O(1/n)）修正项的显式表达式与谱分解；随后在一维输入、sin(2x)目标、Gaussian 样本、tanh 激活、不同宽度下的随机特征岭回归实验中验证了理论预测。

**💡 创新点**

创新点在于：①将随机特征模型的有限宽度波动视为有效场论中的loop修正；②首次系统地给出训练、测试和泛化误差的loop展开及其谱形式；③揭示了泛化误差对训练-测试混合波动的高灵敏度；④通过实验验证O(1/n)的尺度律，证明一阶loop修正能显著提高对真实误差的拟合。

**🔧 技术方法**

采用的方法包括：有效场论与循环展开（loop expansion）、统计物理中的高阶协方差/四阶顶点（vertex）推导、谱分解与投影、解析式展开、数值稳定的特征值分解求逆、Monte Carlo 估计均值和四阶协方差。

**📊 数据集**

实验使用的“数据集”为：一维高斯分布输入（训练集64点、测试集512点），目标函数 y(x)=sin(2x)，无标签噪声；随机特征由 i.i.d. 正态权重、偏置和 tanh 激活产生，宽度 n∈{256,512,1024,2048}。

**📈 对比分析**

与基线（仅使用均值核的树形理论）对比：实验发现树形预测与平均误差相差较大，而加入一阶loop修正后，训练误差、测试误差和泛化误差均与实验曲线高度一致；误差随宽度的下降速度符合 n⁻¹ 的比例，进一步验证理论的尺度律。

**⚠️ 局限性**

局限性：①仅考虑冻结参数的随机特征模型，未涉及全参数训练的动态；②理论推导基于宽度足够大、矩阵收敛于均值核的假设，可能在极小宽度或强噪声情形失效；③实验仅在一维、sin(2x)、无噪声的极简场景，缺乏对更复杂输入/非平稳目标的验证；④高阶非高斯效应（四阶以上）被截断，可能在某些网络架构或激活函数下产生显著影响。

---

## 454. The role of System 1 and System 2 semantic memory structure in human and LLM biases

**arXiv ID:** 2604.12816 | [PDF](https://arxiv.org/pdf/2604.12816v1)

**作者:** Katherine Abramski `[一作]` (University of Pisa), Massimo Stella `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构建并分析人类与大型语言模型的多层语义网络，分别对应系统1（联想层）与系统2（定义与范畴层），并通过结构可约性与传播激活评估隐性性别偏差。

**💡 创新点**

首次将双重过程理论映射为可操作的多层语义网络框架，并在同一框架下对人类与LLM进行系统性比较；提出利用网络可约性判定知识结构差异，并用传播激活量化偏差。

**🔧 技术方法**

多层网络建模、结构可约性分析（信息熵距离、层合并算法）、传播激活偏差评估（R包 spreadr）以及网络统计与可视化工具。

**📊 数据集**

人类自由联想数据集 SWOW、LLM 生成的 LWOW（针对 Mistral 与 Llama3）、WordNet 定义与关系、172 个性别刻板目标词集合。

**📈 对比分析**

通过层级可约性最大化与关联效应大小比较，结果显示人类三层完全不可约，系统2层偏差显著下降；LLM 仅部分可约，偏差与结构无系统关联，表明人类与机器在认知结构与偏差调节机制上存在根本差异。

**⚠️ 局限性**

样本受限于英语西方文化、词汇覆盖不足、LLM 生成受提示敏感、未涉及规则层面的推理操作，且未覆盖跨语言与多元文化场景。

---

## 455. Algorithmic Analysis of Dense Associative Memory: Finite-Size Guarantees and Adversarial Robustness

**arXiv ID:** 2604.12811 | [PDF](https://arxiv.org/pdf/2604.12811v1)

**作者:** Madhava Gaikwad `[一作]` `[通讯]` (Independent Researcher), Madhava Gaikwad (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了密集联想记忆（DAM）检索动态的算法分析，给出了有限尺寸下的收敛时间、几何收敛率、对抗鲁棒性界限以及存储容量上界，并将其视为精确势能博弈；

**💡 创新点**

通过显式的模式分离与界定干扰条件，首次提供了可验证的有限-N收敛保证、对抗误差容忍度与容量下界，并将DAM更新行为与势能游戏严格对应；

**🔧 技术方法**

采用异步坐标上升、收缩分析、势能博弈理论以及对抗扰动评估，并与生成功能分析（GFA）等统计物理方法对照；

**📊 数据集**

在实验中使用了三阶（n=3）DAM网络，并在二值化的MNIST与CIFAR-10数据集上进行验证；

**📈 对比分析**

与同步更新、随机与最坏情况模式集进行对比，实验显示异步更新实现O(log N)收敛、误差容忍度满足ρ<α/2，容量达Θ(N^{n-1})（至多多项式对数修正），与理论预测高度吻合；

**⚠️ 局限性**

仅适用于异步更新；在接近容量负载时需更强干扰假设；O(log N)收敛界限保守；对抗容忍度界限为最坏情况估计；未覆盖同步动态与连续现代Hopfield网络。

---

## 456. Interpretable Relational Inference with LLM-Guided Symbolic Dynamics Modeling

**arXiv ID:** 2604.12806 | [PDF](https://arxiv.org/pdf/2604.12806v1)

**作者:** Xiaoxiao Liang `[一作]` (University of Science and Technology of China), Linyuan Lü `[通讯]` (University of Science and Technology of China)

**通讯引用:** 13841 | [OpenAlex ID](https://openalex.org/A5000969982)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计了 COSINE，联合推断隐藏的相互作用图和稀疏符号动力学方程，使用外层LLM动态编辑基函数库，实现可微分的结构与机制共优化；

**💡 创新点**

创新点在于将可微分消息‑更新分解与稀疏符号回归结合，形成一种既可解释又高效的联合推断框架；通过大语言模型在外循环主动优化符号基库，突破闭世界库限制，提升结构识别和机制发现能力；

**🔧 技术方法**

采用 Gumbel‑Softmax 可微分图生成、稀疏符号回归（消息/更新基库）、梯度优化、LLM 引导的基库演化、一次步监督、KL 结构正则和 L1 稀疏正则；

**📊 数据集**

使用六类合成动力学（Michaelis‑Menten、Diffusion、Springs、Kuramoto、Friedkin‑Johnsen、Coupled‑Map Network）在 ER/BA/WS 图上进行验证，并在真实 COVID‑19 美国四州流行病数据上进行应用；

**📈 对比分析**

与统计方法（GC、MI、TE）、神经网络推断（NRI、GDP）及注意力模型（RIVA）对比，AUC 均达到或逼近 1，显著优于对手；在机制发现上能恢复大部分原始原语；在大规模网络（N≤200）下训练时间和显存低于 NRI，表现出优越的效率与可扩展性；

**⚠️ 局限性**

对复杂动力学对外层 LLM 的质量要求高；稀疏符号回归表达能力受限，可能无法覆盖所有复杂非线性；有限噪声数据下可识别性下降；基库演化受 LLM 推理质量影响，可能导致局部最优。

---

## 457. Efficient Adversarial Training via Criticality-Aware Fine-Tuning

**arXiv ID:** 2604.12780 | [PDF](https://arxiv.org/pdf/2604.12780v1)

**作者:** Wenyun Li `[一作]` (Harbin Institute of Technology), Xiangyuan Lan `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

针对预训练 Vision Transformer 进行轻量化对抗训练，只微调极少数量的关键参数，提升模型对对抗样本的鲁棒性。

**💡 创新点**

创新点在于提出“Robust Parameter Criticality (RPC)”衡量每个参数对鲁棒性的贡献，并将其与参数高效微调（PEFT）结合，形成 Criticality-Aware Adversarial Training (CAAT)。

**🔧 技术方法**

核心技术包括：梯度平方估计 RPC、根据阈值选取关键参数、使用 LoRA/Adapter 等 PEFT 模块进行间接微调，以及结合多种对抗训练策略（TRADES、MART、PRM）。

**📊 数据集**

实验数据集覆盖 CIFAR‑10、CIFAR‑100、ImageNet 三大通用图像分类数据集，并在 ViT‑B、ViT‑L、ViT‑H、Swin‑B、Swin‑L 等多种 Transformer 结构上验证。

**📈 对比分析**

与全参数对抗训练、Adapter、LoRA、FullLoRA‑AT、HyperAT 等方法对比，CAAT 在保持 1% 训练参数的同时，平均鲁棒准确率仅比全量训练低 4.3%，在大模型规模下仍保持高效并且显著优于现有轻量化对抗训练方案。

**⚠️ 局限性**

局限性：对抗鲁棒性仍略逊于全量微调；关键参数的选择依赖梯度估计，可能对不同任务或攻击方法敏感；目前仅在 Vision Transformer 上验证，跨模态或更大规模模型的适用性仍待进一步探索。

---

## 458. EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution

**arXiv ID:** 2604.12776 | [PDF](https://arxiv.org/pdf/2604.12776v1)

**作者:** Shiyu He `[一作]` (Xinjiang University), Tingxiang Gu `[通讯]` (Xinjiang University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出EvoSpark框架，实现LLM驱动多代理系统中的长期连贯叙事。

**💡 创新点**

创新点在于将随机生成的“sparking”视为叙事资产，采用Emergent Character Grounding Protocol (ECGP)、Role Socio‑Evolutionary Base (RSB) 与 Generative Mise‑en‑Scène (GMS) 三大机制，解决社交记忆堆叠与空间不一致问题。

**🔧 技术方法**

技术包括统一叙事操作引擎、事件驱动反思-综合-合并机制、层化记忆体系，以及LLM + 约束推理与空间场景生成。

**📊 数据集**

数据集为六种情节类型（侦探、科幻、史诗奇幻等）的人工合成叙事脚本，生成约200k–250k词的长篇情节。

**📈 对比分析**

通过与Open‑Theatre、BookWorld、HoLLMwood等基线在三种控制模式下进行双向LLM评判，EvoSpark在逻辑连贯性、角色一致性和创造性等指标上显著优于基线。

**⚠️ 局限性**

限制在于记忆与关系图随时间膨胀导致内存与推理延迟，且对人机交互的鲁棒性未充分评估。

---

## 459. Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment

**arXiv ID:** 2604.12851 | [PDF](https://arxiv.org/pdf/2604.12851v1)

**作者:** Bryan Chen Zhengyu Tan `[一作]` (Singapore University of Technology and Design), Roy Ka-Wei Lee `[通讯]` (Singapore University of Technology and Design)

**通讯引用:** 1726 | [OpenAlex ID](https://openalex.org/A5089793938)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在新加坡的世界价值观调查（WVS）数据上构建子群模态偏好数据集，利用结构化数值预测和开放式生成任务，对多种LLM进行结构化微调（SFT），并评估模型在未见交叉子群（OOD）和开放式生成场景下的价值对齐与公平性表现。

**💡 创新点**

①提出Modal Diversity Score量化不同子群价值冲突；②设计可组合的子群对齐数据集，实现对交叉子群的泛化；③展示简单SFT即可显著提升子群价值预测准确率，并分析对齐对公平性的双刃效应。

**🔧 技术方法**

采用结构化数值预测与开放式生成评估；使用LoRA对LLM进行SFT；评估指标包括准确率、NMAE、WinRate（Persona、Value、Overall）；公平性评估用Norm. Range和Coefficient of Variation；人工与LLM评估对比。

**📊 数据集**

新加坡子集的世界价值观调查（WVS）Wave 7，包含214道价值问题、约20,000个（问题, 子群）对，按统计显著性划分训练集和未见交叉子群评估集。

**📈 对比分析**

与GPT‑4.1等基线对比，SFT后在OOD子群上准确率提升约17.4%，NMAE下降0.096；在开放式生成中对齐得分平均提升2.2%（Value）与1.1%（Overall）；然而，尽管准确率差距缩小，NMAE差距扩大，表明公平性并未同步改善。

**⚠️ 局限性**

研究仅聚焦新加坡，结果受限于该国文化与数据；仅使用模态标签忽略内部分布，可能掩盖多样性；仅测试SFT，未尝试更先进的对齐或去偏方法；数据泄露风险未彻底排除；模型可能加剧刻板印象，需在高风险场景谨慎部署。

---

## 460. Challenging Vision-Language Models with Physically Deployable Multimodal Semantic Lighting Attacks

**arXiv ID:** 2604.12833 | [PDF](https://arxiv.org/pdf/2604.12833v1)

**作者:** Yingying Zhao `[一作]`, Wen Yao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了多模态语义照明攻击（MSLA），通过可物理部署的三角形光照扰动对Vision‑Language模型实施对抗攻击。

**💡 创新点**

创新点在于：①首次针对VLM设计物理可部署的局部光照对抗攻击；②采用三角形光模型与可调参数（位置、半径、角度、颜色、透明度）形成可控扰动；③利用遗传算法在高维非凸空间全局搜索最优光照参数；④在数字与物理环境下统一评估攻击效果。

**🔧 技术方法**

使用遗传算法进行参数优化，结合图像与文本的交叉相似度作为目标；在数字实验中使用光照合成；在物理实验中使用手电筒、透明塑料片与纸质三角形模板投射光照。

**📊 数据集**

使用COCO数据集（300张图像，80类标签）进行零样本分类、图像字幕和视觉问答任务的评估。

**📈 对比分析**

与自然光、阴影、ITA等现有光照攻击方法对比，MSLA在CLIP四种变体上将零样本分类准确率降至11–36%（相比最佳对手可达82%），在图像字幕和VQA任务中分别降低一致性和正确率30–57%，明显优于对手且在物理场景中也保持高成功率。

**⚠️ 局限性**

局限性包括：攻击效果受光照强度、颜色、投射角度等实际环境因素影响；局部光照易被人察觉；对光照硬件依赖较强，难以在大规模多样化场景中快速部署；未针对多模态模型的不同架构（如基于transformer的文本解码器）进行更细粒度的鲁棒性研究。

---

## 461. VULCAN: Vision-Language-Model Enhanced Multi-Agent Cooperative Navigation for Indoor Fire-Disaster Response

**arXiv ID:** 2604.12831 | [PDF](https://arxiv.org/pdf/2604.12831v1)

**作者:** Shengding Liu `[一作]` (Michigan State University), Qiben Yan `[通讯]` (Michigan State University)

**通讯引用:** 2520 | [OpenAlex ID](https://openalex.org/A5042277127)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种基于视觉‑语言模型（VLM）和多模态感知的火灾环境下的多智能体协作导航框架，并在模拟火灾场景中进行了系统评估。

**💡 创新点**

创新点包括：① 将RGB‑D、热像、mmWave雷达等多模态数据通过VLM融合形成烟雾透明视图；② 使用VLM作为全局规划器，实现对危险级别的高层决策；③ 在局部规划中引入危险加权的Fast Marching Method，实现安全且高效的路径生成；④ 扩展Habitat‑Matterport3D数据集，引入物理可逼真的火灾扰动。

**🔧 技术方法**

采用的技术包括：多模态感知与融合（VLM）、开放词汇检测与分割、点云投影与DBSCAN去噪、基于热、烟雾与不确定性的危险地图构建、VLM‑基于全局规划、危险加权Fast Marching Method局部规划。

**📊 数据集**

使用扩展后的Habitat‑Matterport3D火灾模拟数据集（36个验证场景、200个episode，包含多级烟雾密度、火焰强度与热损失），并在六个目标类别（chair、sofa、plant、bed、toilet、TV）上进行实验。

**📈 对比分析**

对Greedy、Cost‑Utility、Random Sample、Co‑NavGPT四种基线进行NS、SR、SPL、CHE四指标比较。结果显示：在正常环境下Co‑NavGPT性能最高；在火灾环境下传统方法性能显著下降，而Co‑NavGPT保持较高成功率、较低步数和较低累计危险暴露，证明危险感知与VLM规划的有效性。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实火灾现场实验；数据集中未包含人类目标，使用其他物体代替；火灾动态变化的实时感知与规划仍不完善；VLM推理开销大，对实时性与能耗提出挑战。

---

## 462. DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding

**arXiv ID:** 2604.12812 | [PDF](https://arxiv.org/pdf/2604.12812v1)

**作者:** Hao Yan `[一作]` (Huazhong University of Science and Technology), Xiang Bai `[通讯]` (Huazhong University of Science and Technology)

**通讯引用:** 39036 | [OpenAlex ID](https://openalex.org/A5039363991)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一个基于多页面文档视觉问答的结构化推理框架 DocSeeker，能够在长文档中进行证据定位与推理，并在全文输入下保持高准确率。

**💡 创新点**

创新点包括：① 引入 ALR（Analysis–Localization–Reasoning）三步推理范式，强制模型输出分析、证据定位与推理链；② 采用两阶段训练：先用知识蒸馏生成高质量 ALR CoT 训练数据，再用 Evidence‑aware Group Relative Policy Optimization (EviGRPO) 进行强化学习；③ 设计 Evidence‑Guided Resolution Allocation (EGRA) 解决长文档视觉输入的内存瓶颈，并提升信噪比。

**🔧 技术方法**

技术栈主要包括 Qwen‑2.5‑VL‑7B‑Instruct 作为骨干模型；使用 Gemini‑2.5‑Flash 进行知识蒸馏；通过 GRPO 进行强化学习；对输入进行页面级标记；采用多维奖励函数（格式、证据定位、答案准确度）；并使用 EGRA 在训练中动态调整图像分辨率。

**📊 数据集**

训练数据来源：MP‑DocVQA 与 DUDE（最多20页）；测试数据包括 MP‑DocVQA、DUDE、MMLongBench‑Doc（多达468页）、LongDocURL、SlideVQA；通过蒸馏得到 13,986 条 ALR CoT 训练样本。

**📈 对比分析**

与公开与闭源基线（InternVL3、mPLUG‑DocOwl2、Vis‑RAG、SV‑RAG、VDocRAG 等）以及 GPT‑4o、Gemini 等模型对比，DocSeeker 在所有 5 个基准上均取得最高或竞争性最佳分数；在 OOD 长文档上实现 30–60% 的性能提升；在全文输入条件下，性能与仅给出证据页面的差距不足 1%。

**⚠️ 局限性**

局限性：① 仍需高质量标注的 ALR CoT 训练数据，对长文档的标注成本高；② 强化学习阶段对超参数（奖励权重、β、rollout 大小）敏感；③ 依赖 Qwen‑2.5‑VL‑7B 的计算资源，推理时仍需大量 GPU 内存；④ 在极端噪声检索场景下，仍可能出现误定位。

---

## 463. Generative Anonymization in Event Streams

**arXiv ID:** 2604.12803 | [PDF](https://arxiv.org/pdf/2604.12803v1)

**作者:** Adam T. Müller `[一作]` (Heilbronn University of Applied Sciences), Nicolaj C. Stache `[通讯]` (Heilbronn University of Applied Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `9cc9baba-5356-466d-81ff-d80028d90279` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于生成式模型的事件流匿名化框架：先将稀疏事件投影到连续灰度帧，使用面部检测与Stable Diffusion/INSwapper进行身份替换，再通过视频到事件（V2E）逆映射回事件流，实现身份匿名且保持时空结构。

**💡 创新点**

首次在事件域实现生成式匿名化，跨模态桥接通过中间灰度帧利用高质量RGB生成模型；提出事件域结构评估指标 STCD、EMD；创建同步 RGB‑事件人脸数据集，为后续研究提供基准。

**🔧 技术方法**

核心技术包括：EVREAL 事件到灰度映射、INSwapper 与 Stable Diffusion 2 的面部替换、Fast Super‑Resolution CNN、CLAHE 与锐化、V2E 逆映射（v2e）、Fast‑Super‑Resolution、FSRCNN 以及用于评估的 STCD 与 EMD。

**📊 数据集**

使用新收集的同步 RGB‑事件人脸数据集（协作机器人轨迹采集）作为主要实验数据；参考 FireNet 进行 E2V 重建以评估匿名化效果。

**📈 对比分析**

与原始未匿名化数据以及同一人物不同采集的参考数据对比：身份相似度从 0.713 降至 0.118；STCD 与 EMD 约提高 30‑倍；面部检测（YOLOv8、事件域检测）IoU 达 0.960 与 0.702，检测率无显著下降，说明数据实用性保持良好。

**⚠️ 局限性**

局限性：依赖帧级中间转换，未直接在事件级别生成；V2E 逆映射导致事件密度不足、离散化与模糊伪影；面部微表情捕捉不够精细；整体分辨率受面部替换模型限制。

---

## 464. Human Agency, Causality, and the Human Computer Interface in High-Stakes Artificial Intelligence

**arXiv ID:** 2604.12793 | [PDF](https://arxiv.org/pdf/2604.12793v1)

**作者:** Georges Hattab `[一作]` (Robert Koch Institute), Georges Hattab `[通讯]` (Robert Koch Institute)

**通讯引用:** 735 | [OpenAlex ID](https://openalex.org/A5079991493)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出Causal-Agency Framework (CAF)，强调在人机交互中恢复人类因果控制。

**💡 创新点**

将因果推理、可量化不确定性与可操作界面集成为三层架构，弥补现有XAI对因果与不确定性缺失的不足。

**🔧 技术方法**

使用结构因果模型 (SCM)、不确定性量化技术（如 UbiQTree）、解释与干预模块，以及可操作人机接口设计。

**📊 数据集**

未给出具体公开数据集，强调需根据不同领域采集因果与干预数据。

**📈 对比分析**

通过闭环评估人机联合性能而非单纯信任度；但本文尚未提供实验对比与具体性能指标。

**⚠️ 局限性**

依赖于可用的因果发现与不确定性量化方法、缺乏标准数据集、需专门训练且数据采集面临挑战。

---

## 465. Fragile Reconstruction: Adversarial Vulnerability of Reconstruction-Based Detectors for Diffusion-Generated Images

**arXiv ID:** 2604.12781 | [PDF](https://arxiv.org/pdf/2604.12781v1)

**作者:** Haoyang Jiang `[一作]` (Renmin University of China), Ju Fan `[通讯]` (Renmin University of China)

**通讯引用:** 3063 | [OpenAlex ID](https://openalex.org/A5100739546)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `3855fcda-48ef-4070-a15e-803cd5c84d83` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

探究并证明基于重构的扩散生成图像检测器在对抗攻击下极易失效

**💡 创新点**

首次系统评估多种检测器的对抗鲁棒性与跨模型、跨方法的攻击迁移，并揭示其低信噪比导致防御失效的根本原因

**🔧 技术方法**

使用连续时间扩散模型的神经ODE逆向过程、伴随方法求梯度、PGD攻击与对抗训练与扩散去噪等技术

**📊 数据集**

采用ImageNet真实图像与ADM、SDv1.5、FLUX、VQDM四种扩散生成的合成图像共计5,000张每类进行实验

**📈 对比分析**

与无攻击基线比较，攻击后各检测器的鲁棒准确率从约70-90%降至0%，且攻击能在黑盒下跨模型/方法迁移使准确率降至≈50%

**⚠️ 局限性**

其主要局限在于重构特征信噪比极低，使得对抗训练与去噪等防御无法提升鲁棒性，需重新设计检测策略

---

## 466. Cognition-Inspired Dual-Stream Semantic Enhancement for Vision-Based Dynamic Emotion Modeling

**arXiv ID:** 2604.12777 | [PDF](https://arxiv.org/pdf/2604.12777v1)

**作者:** Huanzhen Wang `[一作]` (Fudan University), Wenqiang Zhang `[通讯]` (Fudan University)

**通讯引用:** 3538 | [OpenAlex ID](https://openalex.org/A5100669255)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于认知启发的双流语义增强框架 DuSE，用于动态面部表情识别。

**💡 创新点**

创新点在于将情绪认知中的预激效应和知识整合机制拆解为两个并行流——层次时间提示集群 HTPC 与潜在语义情绪聚合器 LSEA，并通过跨模态提示流和跨域知识迁移实现更符合人类情绪感知的语义驱动学习。

**🔧 技术方法**

核心技术包括 CLIP 视觉‑文本双模模型的层级提示注入（浅层与深层提示）、动态提示调优、基于自注意力的时序特征聚合、语义引导的特征融合与多头语义注意机制。

**📊 数据集**

在两个真实环境视频数据集上验证：DFEW 和 FERV39k。

**📈 对比分析**

与现有基于 CLIP、MAE、HiCMAE 等方法比较，DuSE 在 WAR/UAR 上均取得 64.88/75.36（DFEW）和 43.39/53.05（FERV39k）的最高或第二高分，显著优于传统模型并在各情绪类别上表现更好。

**⚠️ 局限性**

局限性包括对大型 CLIP 预训练模型的依赖、对高帧率视频的处理效率尚待提升、以及在极端遮挡或光照变化下的鲁棒性需要进一步验证。

---

## 467. A Multi-Agent Feedback System for Detecting and Describing News Events in Satellite Imagery

**arXiv ID:** 2604.12772 | [PDF](https://arxiv.org/pdf/2604.12772v1)

**作者:** Madeline Anderson `[一作]` (Massachusetts Institute of Technology), Kerri Cahoy `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 2191 | [OpenAlex ID](https://openalex.org/A5113923056)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种基于多代理迭代反馈的SkyScraper系统，用新闻文章自动定位并生成多时序卫星影像事件描述。

**💡 创新点**

创新点在于将大语言模型与多代理协同验证相结合，实现了比传统地理编码方法高5倍的事件检测效率，并构建了新的多时序字幕数据集。

**🔧 技术方法**

采用的技术包括大语言模型（Gemini‑2.5‑flash）、Mapbox地理编码API、PlanetScope/Sentinel‑2卫星影像检索、以及多模态LLM验证和字幕生成。

**📊 数据集**

使用了从GDELT 2022‑2024年新闻文章抽取的约1000篇文章，以及随后生成的约5,000条多时序影像序列的PlanetScope和Sentinel‑2数据集。

**📈 对比分析**

与加权质心、GIPSY等传统地理编码方法比较，SkyScraper的事件检测率从17%提升至84%，相对提升约4.9倍，显示出显著性能优势。

**⚠️ 局限性**

局限性包括对新闻文本质量与覆盖范围的依赖、对LLM推理错误的可能性，以及多时序影像数据获取与验证过程的算力和成本。

---

## 468. The cross-sectional warping problem for hyperelastic beams: An efficient formulation in Voigt notation

**arXiv ID:** 2604.12886 | [PDF](https://arxiv.org/pdf/2604.12886v1)

**作者:** Juan C. Alzate Cobo `[一作]` (Technical University of Darmstadt), Oliver Weeger `[通讯]` (Technical University of Darmstadt)

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4de8e9d8-757b-475f-9627-18a445e50202` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了针对超弹性梁的截面扭曲问题的全材料形式，并将其以Voigt表示法实现；

**💡 创新点**

创新点在于将截面扭曲问题从PK1形式改为PK2形式，利用对称的二阶应力、应变张量实现矩阵向量表示，显著降低了四阶本构张量的计算量；

**🔧 技术方法**

使用了几何非线性梁理论、第二Piola-Kirchhoff应力、Green-Lagrange应变、Voigt标记、B-和𝔅-算子、等距变形基函数（B-spline/NURBS）及等距有限元技术；

**📊 数据集**

采用了两种典型截面（单位正方形、单位圆形）和三种超弹性材料（Saint Venant–Kirchhoff、Neo-Hookean、Mooney–Rivlin）作为验证数据集；

**📈 对比分析**

通过与现有PK1实现及文献结果（扭矩-扭转率、轴向力、轴向刚度等）的对比，显示两种形式在残差、刚度矩阵、扭转响应等方面完全一致，且数值收敛性能相当；

**⚠️ 局限性**

局限性包括对刚体约束的Lagrange乘子处理仍需经验，且对极限跨越复杂几何截面时可能需要进一步改进（如浸入方法），目前未探讨塑性、耦合多物理场等扩展情况。

---

## 469. Evaluating LLMs Code Reasoning Under Real-World Context

**arXiv ID:** 2604.12881 | [PDF](https://arxiv.org/pdf/2604.12881v1)

**作者:** Changshu Liu `[一作]` (University of Illinois Urbana-Champaign), Changshu Liu `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 11 | [OpenAlex ID](https://openalex.org/A5044650054)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并构建了一个真实世界上下文的代码推理基准（CRUXEval），涵盖135个来自十个流行Python项目的推理问题，并支持复杂自定义类型的序列化。

**💡 创新点**

创新点在于自动化地序列化/反序列化非原始数据结构，将真实项目的复杂性保留到输入输出，突破了现有基准只使用原始类型或简短代码的限制。

**🔧 技术方法**

利用静态与动态程序分析技术对输入输出进行分解并序列化为JSON，同时使用反序列化和运行时测试来判定预测是否正确。

**📊 数据集**

数据集来自十个热门Python开源项目（如pandas、requests等），共135条推理任务，构成CRUXEval benchmark。

**📈 对比分析**

与现有CodeSense等基准进行对比，六款LLM在CRUXEval上的输入/输出预测性能分别下降64.32%/52.22%，表明在真实项目环境下LLM的代码推理能力显著受限。

**⚠️ 局限性**

局限性包括：仅覆盖十个项目，可能不足以代表所有真实场景；序列化方法对某些极端复杂类型仍可能失效；评测仍依赖运行时比较，可能忽略语义相等但结构不同的输出。

---

## 470. Hyper Separation Logic (extended version)

**arXiv ID:** 2604.12870 | [PDF](https://arxiv.org/pdf/2604.12870v1)

**作者:** Trayan Gospodinov `[一作]` (INSAIT, Sofia University St. Kliment Ohridski), Peter Müller `[通讯]` (ETH Zurich)

**通讯引用:** 13829 | [OpenAlex ID](https://openalex.org/A5050979141)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

提出了Hyper Separation Logic（HSL），一种支持任意量化交替的堆操作程序的超属性推理的程序逻辑；

**💡 创新点**

创新点在于引入了超分离结合（hyper separating conjunction）和通用框架规则，使得传统分离逻辑的局部推理能够扩展到超属性；

**🔧 技术方法**

采用超属性三元组（PCQ）定义、超分离结合语义、对错误状态引入未知状态标签、以及在Isabelle/HOL中的形式化证明；

**📊 数据集**

无实验数据集，论文以形式化证明与案例示例为主要验证手段；

**📈 对比分析**

通过案例验证（如GNI、非确定性程序的存在性等）展示HSL在表达和证明复杂超属性上的能力，未涉及传统性能对比；

**⚠️ 局限性**

局限性包括尚未覆盖并发、终止性证明以及跨程序的关系属性，且对未知状态处理仍显不够精细。

---

## 471. Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning

**arXiv ID:** 2604.12770 | [PDF](https://arxiv.org/pdf/2604.12770v1)

**作者:** Timon Ziegenbein `[一作]` (Leibniz University Hannover), Henning Wachsmuth `[通讯]` (Leibniz University Hannover)

**通讯引用:** 4012 | [OpenAlex ID](https://openalex.org/A5014375244)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用强化学习训练 LLM 生成自包含、意义保持的编辑建议，提升论证文本的适当性。

**💡 创新点**

首次在编辑层面引入多目标奖励模型（语义相似度、流畅度、编辑模式一致性），并通过 RL 直接优化人类式编辑；同时实现多轮迭代编辑逼近全改写效果。

**🔧 技术方法**

核心技术包括：Llama‑3.1‑8B‑Instruct 作为策略模型；Group Relative Policy Optimization (GRPO) + LoRA 细调；四个奖励分类器（语义相似度、流畅度、模式一致性、适当性）；嵌入、BERT、语言模型等辅助模型。

**📊 数据集**

使用人类修订数据集 IteraTeR 训练编辑层面分类器；使用扩展后的适当性语料库（原始 2,191 条 + IAC v2 + GAQCorpus 49,417 条）训练与评估 RL 模型。

**📈 对比分析**

与多种 PPO 基线（PPO_app、PPO_app<sim 等）对比，评估编辑层面（Sim、Flu、Con、HL、#HL）和论证层面（BERTScore、PPL、App、All）指标。GRPO_full 在人类式编辑比例 HL 上超过 0.66，适当性提升 36.4%；多轮迭代后整体适当性提升约 24%，逼近全改写效果。

**⚠️ 局限性**

适当性定义受文化与主观偏见影响；仅关注句子级编辑，难以处理跨句子结构重组；奖励代理指标可能与真实语义需求冲突；对非英语语料的迁移性未知。

---

## 472. RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair

**arXiv ID:** 2604.12820 | [PDF](https://arxiv.org/pdf/2604.12820v1)

**作者:** Jagadeesh Rachapudi `[一作]` (Indian Institute of Technology Mandi), Amit Shukla `[通讯]` (Indian Institute of Technology Mandi)

**通讯引用:** 1583 | [OpenAlex ID](https://openalex.org/A5008777707)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了交互式机器忘记（IMU）框架，使用户能够通过自然语言在推理时直接让LLM忘记特定知识。

**💡 创新点**

创新点在于将机器忘记转化为可在推理时交互的任务，并设计了RePAIR多模态框架和训练无关的单样本STAMP/STAMP-LR方法。

**🔧 技术方法**

核心技术包括Prompt-Aware Model Repair（RePAIR）、监督式意图检测（watchdog）、代码生成（surgeon）以及通过伪逆更新激活的STAM（STAMP）及其低秩变体STAMP-LR。

**📊 数据集**

使用了WMDP-Bio、MMLU、合成个人档案（Mistral-7B API生成）以及TinyStories作为指标数据集。

**📈 对比分析**

与GA、NPO、RMU、FLAT、WGA、ASU等六种最先进方法对比，STAMP/STAMP-LR在忘记准确率降到0、保持准确率保持在70%以上、困惑度接近oracle，并在单样本和推理时显著加速（约3倍）表现优异。

**⚠️ 局限性**

局限性包括仍需保留一小部分重放缓冲（至少10%），在边缘设备上存储受GDPR/CCPA限制，且尚未实现完全无重放的忘记，跨模态扩展仍有挑战。

---

## 473. Understanding and Improving Continuous Adversarial Training for LLMs via In-context Learning Theory

**arXiv ID:** 2604.12817 | [PDF](https://arxiv.org/pdf/2604.12817v1)

**作者:** Shaopeng Fu `[一作]` (King Abdullah University of Science and Technology), Di Wang `[通讯]` (King Abdullah University of Science and Technology)

**通讯引用:** 108615 | [OpenAlex ID](https://openalex.org/A5058772567)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了连续对抗训练（CAT）在大语言模型（LLM）上的理论基础，并提出改进方法ER‑CAT。

**💡 创新点**

首次给出CAT的鲁棒泛化上界，揭示嵌入空间扰动半径与鲁棒性负相关，并指出嵌入矩阵奇异值对鲁棒性的重要性。

**🔧 技术方法**

利用ICL理论、线性自注意力模型、对抗训练与奇异值方差正则化技术进行分析与实现。

**📊 数据集**

使用Harmbench、UltraChat、AdvBench、AlpacaEval等公开数据集进行训练与评估。

**📈 对比分析**

与原始CAT对比，ER‑CAT在保持相同攻击成功率下提升LC‑WinRate约10‑20%，或在相同LC‑WinRate下进一步降低攻击成功率，性能优势显著。

**⚠️ 局限性**

仍受限于对非线性Transformer的理论推广有限、β超参数调节对效果影响不大、以及训练时略微增加的计算开销。

---

## 474. From edges to meaning: Semantic line sketches as a cognitive scaffold for ancient pictograph invention

**arXiv ID:** 2604.12865 | [PDF](https://arxiv.org/pdf/2604.12865v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 475. VFA: Relieving Vector Operations in Flash Attention with Global Maximum Pre-computation

**arXiv ID:** 2604.12798 | [PDF](https://arxiv.org/pdf/2604.12798v1)

**作者:** Yupeng Sun `[一作]`, Hui Wang `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在FlashAttention的在线softmax中引入快速m初始化、sink+local块重排和最大值冻结，减少向量/SIMD限制的减法/归一化开销。

**💡 创新点**

通过向量友好的最大值初始化和动态重排实现了对在线softmax统计更新频率的显著降低，同时与BLASST稀疏跳过无缝组合。

**🔧 技术方法**

采用FlashAttention-2/4内核改造、sabsmax关键块表示、在线块重排、最大值冻结、BLASST动态跳过以及低精度FP8/FP4加速。

**📊 数据集**

在MATH500、MMLU（各子集）、HumanEval、CMMLU等多任务数据集上评测，使用Qwen3-30B/8B和Llama3-8B模型。

**📈 对比分析**

与FA2基线对比，VFA在C8V32、C4V32、C4V16等配置下平均可实现约2倍加速；VSA进一步通过跳过块提升速度，准确率保持不变。

**⚠️ 局限性**

在极端非局部或对抗性注意分布、较长序列长度以及不同模型/层级下可能效果下降，初始化质量和冻结策略对性能仍敏感。

---

## 476. Actuation space reduction to facilitate insightful shape matching in a novel reconfigurable tendon driven continuum manipulator

**arXiv ID:** 2604.12792 | [PDF](https://arxiv.org/pdf/2604.12792v1)

**作者:** Sabyasachi Dash `[一作]` (University of Illinois Urbana-Champaign), Girish Krishnan `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 1582 | [OpenAlex ID](https://openalex.org/A5050432620)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种每个中间盘可主动旋转的可重构肌腱驱动连续机械臂，并提出基于曲率-扭转空间的四步顺序形状匹配框架。

**💡 创新点**

创新点在于通过主动盘旋实现局部肌腱重新布线，极大扩展形变模式，并利用曲率-扭转空间揭示盘旋对全局形状的影响，从而将高维控制问题降维并实现模型无关的逆运动学。

**🔧 技术方法**

采用主动旋转的电机驱动盘、步进电机拉伸肌腱、磁学测距仪进行三维形状重建、DBSCAN聚类、曲率-扭转映射、金字塔搜索优化等技术。

**📊 数据集**

实验数据来自于对560 mm 长、8 段、单一肌腱的机器人在多种盘旋和拉伸组合下的磁学三维坐标测量，没有使用公开数据集。

**📈 对比分析**

与传统固定布线 TDCM 对比，实验实现了曲率误差 RMSE 0.4–0.6 cm、末端误差 4–10 mm 的形状匹配，表明该方法在单肌腱情况下可实现较高精度。

**⚠️ 局限性**

局限性包括仅使用单一肌腱、盘旋角度受限于 ±90°、未实现实时反馈、未考虑重量和动力学耦合，且缺乏大规模数据验证。

---

## 477. QuarkMedSearch: A Long-Horizon Deep Search Agent for Exploring Medical Intelligence

**arXiv ID:** 2604.12867 | [PDF](https://arxiv.org/pdf/2604.12867v1)

**作者:** Zhichao Lin `[一作]` (Alibaba), Guanjun Jiang `[通讯]` (Alibaba)

**通讯引用:** 430 | [OpenAlex ID](https://openalex.org/A5004378463)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `79276348-11e0-48e3-84bc-7ec231d0171c` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了 QuarkMedSearch 框架，构建了长链医学深度搜索数据合成管线、后训练策略，并提出了专门的医学深度搜索基准。

**💡 创新点**

创新点包括四阶段医学深度搜索数据合成（知识图谱采样、在线检索、多实体模糊、跨模型验证）、两阶段 SFT+RLVR 训练、Discard‑all 上下文管理以及 QuarkMedSearch 基准。

**🔧 技术方法**

使用 ReAct 代理架构、四工具执行器（Search、Visit、Medical Professional Search、LLM Check）、多模型验证、GRPO+R2+TIS 的 RLVR、以及多维轨迹过滤与 Rubric 评估。

**📊 数据集**

数据来源于自研医学知识图谱、真实 Web 搜索结果、公开基准（BrowseComp、HLE、MedBrowseComp、MedXpertQA）以及通过管线合成的医学长链问题集。

**📈 对比分析**

与商业闭源模型（Seed1.8、Gemini‑3‑Pro、Claude‑4.5‑Opus、GPT‑5.2‑xhigh）以及同规模开源模型（Kimi‑K2.5、GLM‑5、DeepSeek‑V3.2、Qwen3.5‑35B 等）在 QuarkMedSearch Benchmark、BrowseComp‑EN/​ZH 及 Xbench DeepSearch 上进行对比；QuarkMedSearch 在同规模模型中以 55.71、47.03、57.55 的分数分别击败基线并逼近或超过更大模型，表明其长链医学推理能力显著提升。

**⚠️ 局限性**

局限性在于仍受限于医学知识图谱的覆盖范围、对外部搜索 API 的依赖、需要人工专家验证、可能存在样本泄露风险，以及在处理更开放式、非明确答案的真实医学查询时的通用性仍待提升。

---

## 478. TCL: Enabling Fast and Efficient Cross-Hardware Tensor Program Optimization via Continual Learning

**arXiv ID:** 2604.12891 | [PDF](https://arxiv.org/pdf/2604.12891v1)

**作者:** Chaoyao Shen `[一作]` (Southeast University), Meng Zhang `[通讯]` (Southeast University)

**通讯引用:** 23802 | [OpenAlex ID](https://openalex.org/A5100437827)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119`

**🎯 论文内容**

本工作提出了 TCL 框架，旨在通过离线训练的成本模型实现跨硬件平台的张量程序快速优化，显著降低数据收集与调优时间。

**💡 创新点**

创新点包括：① RDU Sampler 的主动学习策略，利用代表性、分散性和不确定性仅采样 10% 数据；② 基于 Mamba 的轻量级序列模型捕获长程调度依赖；③ 连续知识蒸馏（CKD）机制，实现多平台知识积累而不出现参数爆炸。

**🔧 技术方法**

技术手段涵盖：主动学习、Mamba 结构、LambdaRank 排序损失、连续知识蒸馏与 EWC 防止灾难性遗忘，以及在 TVM/Ansor 搜索框架中的集成。

**📊 数据集**

使用的数据集为大规模张量程序集，包含 Tenset 的多平台数据，并在 Intel i7‑12700F 与 NVIDIA RTX 3080Ti 上自行收集的额外样本。

**📈 对比分析**

与 Tenset‑MLP、TLP、MTL‑TLP、Ansor、Felix 等基线对比，TCL 在 CPU 平台平均提升 16.8× 的调优速度、GPU 平台 12.48×，并在相同 2000 次迭代下分别比 Tenset‑MLP 提升 1.20× 与 1.13× 的推理延迟；在跨平台迁移实验中亦优于 Fine‑tuning 与多任务学习。

**⚠️ 局限性**

局限性在于：对不同硬件体系结构的迁移仍存在知识干扰，CPU 与 GPU 之间无法直接迁移；推理延迟提升相对有限；需进一步设计更细粒度的成本模型以提升预测精度与性能。

---

## 479. Drawing on Memory: Dual-Trace Encoding Improves Cross-Session Recall in LLM Agents

**arXiv ID:** 2604.12948 | [PDF](https://arxiv.org/pdf/2604.12948v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 480. ROSE: An Intent-Centered Evaluation Metric for NL2SQL

**arXiv ID:** 2604.12988 | [PDF](https://arxiv.org/pdf/2604.12988v1)

**作者:** Wenqi Pei `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Yuyu Luo `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种面向意图的NL2SQL评估指标ROSE。

**💡 创新点**

创新点在于引入对抗性 Prover‑Refuter 级联，打破单一参考 SQL 的限制，以用户意图为中心进行评估。

**🔧 技术方法**

使用大型语言模型（如 OpenAI GPT‑3.5/4/5、Gemini‑2.5 Pro、DeepSeek‑R1）实现 SQL Prover 与 Adversarial Refuter，结合 SQL 语法检查与执行结果对比。

**📊 数据集**

采用公开的 BIRD Mini‑Dev、Spider 数据集，并构建 585 条专家一致的验证样本集（ROSE‑VEC）用于评估。

**📈 对比分析**

与传统的 EX、EM、ETM 以及其他 LLM 评估方法比较，ROSE 在 Cohen’s Kappa、Accuracy、MCC、F1 等指标上均优于其它方法，Kappa 提升约 24% 以上。

**⚠️ 局限性**

局限性包括对基础 LLM 性能的依赖、验证集构建时可能的选择偏差，以及相对较高的计算成本和延迟。

---

## 481. Recursive Completion in Higher K-Models: Front-Seed Semantics, Proof-Relevant Witnesses, and the K-Infinity Model

**arXiv ID:** 2604.12981 | [PDF](https://arxiv.org/pdf/2604.12981v1)

**作者:** Daniel O. Martinez-Rivillas `[一作]` (Universidad Militar Nueva Granada), Ruy J. G. B. de Queiroz `[通讯]`

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

构建并比较了未类型化 λ-演算的递归完成模型与显式高维转换塔，并在此基础上证明了前沿种子语义足够性、固定跨度 witness 分离、以及 K_∞ 模型的精确重构与应用公式。

**💡 创新点**

（1）证明仅需 WLWR 与内部右前五边形收缩的较小前沿种子即可完成后续的关联者与五边形证明；（2）在 K_∞ 模型中给出全局连续的 reify、reflect 与 application 的闭式坐标公式；（3）在固定跨度 witness 上给出 β/η 标签化的唯一性与在身份类型 ω-组oid 中的不可连通性；（4）通过递归完成对显式塔进行全维度严格网格边界的匹配。

**🔧 技术方法**

使用 Kan 复形语义、弱 ω-组oid构造、递归完成（高维导数）技术、逆限构造、连续映射与投影配对、以及在 Lean4 证明助手中的形式化验证。

**📊 数据集**

无使用数据集，所有结果均为形式化理论证明。

**📈 对比分析**

对比方法：在低维核心（0–3 维）显式塔与递归完成塔的 4–6 维包装后，构造严格的边界保持实现；随后在模型 K_∞ 上直接计算 reify/reflect/application 的坐标级公式；在固定跨度 witness 上证明 β/η 分离。性能：为纯理论证明，无实验性能评估。

**⚠️ 局限性**

限制：仅对未类型化 λ-演算及其固定种子语义模型给出结论；前沿种子（WLWR + 内部右前五边形收缩）的必要性未证明可由纯扩展 Kan 结构自动产生；高维转换塔采用身份类型 ω-组oid，未探讨其他可能的高维结构；证明仍局限于 Lean4 的形式化验证，缺乏对更广泛应用的经验评估。

---

## 482. The Verification Tax: Fundamental Limits of AI Auditing in the Rare-Error Regime

**arXiv ID:** 2604.12951 | [PDF](https://arxiv.org/pdf/2604.12951v1)

**作者:** Jason Z Wang `[一作]` `[通讯]` (Independent), Jason Z Wang (Independent)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

论文研究了深度学习模型的校准误差（ECE）估计问题，证明了在模型误差率为ε时，最优估计误差随样本量n下降的最小可达速度为Θ((ε/n)^{1/3})，并揭示了所谓的“验证税”——随着模型性能提升，校准验证变得越来越困难。

**💡 创新点**

创新点包括：①给出了与误差率相关的最优下界与上界，完整证明了Θ((ε/n)^{1/3})是极限；②发现了一个清晰的相位转移（ε≈1/√n）使得低误差率下的校准不可检出；③证明自评（不使用真实标签）在信息论层面提供零校准信息；④展示主动查询能消除Lipschitz常数，提升估计速度至Θ(√(ε/n))；⑤证明多组件流水线的验证成本呈指数增长。

**🔧 技术方法**

使用的技术主要包括：极大似然与KL散度的两先验构造、Brown–Low等价、频率估计的直方图分箱分析、链式法则和KL信息下界、主动学习中的探索-利用分阶段策略，以及组合函数的Lipschitz传递性分析。

**📊 数据集**

实验数据集涵盖五个主流基准：MMLU、TruthfulQA、ARC-Challenge、HellaSwag、WinoGrande，使用6个不同规模（8B–405B）的LLM，共27个模型‑基准组合。

**📈 对比分析**

比较方法：对比被动估计与主动查询的误差曲线，验证相位转移阈值，计算可验证差距（Δ/δ_floor）并判断模型排名是否具有统计显著性。结果显示：自评几乎完全无效；在大多数基准上，模型间的可验证差距低于验证阈值；主动查询显著提升估计精度并消除Lipschitz影响；多组件流水线的验证成本指数级增加。

**⚠️ 局限性**

局限性：假设校准函数满足Lipschitz连续性；理论与实验间存在对数修正项的未解闭合；仅针对ECE，未涵盖多类别或不连续校准度量；组合分析采用最坏情况Lipschitz积，实际可能更保守；未处理不完全标签或不平衡数据场景。

---

## 483. Adaptive Data Dropout: Towards Self-Regulated Learning in Deep Neural Networks

**arXiv ID:** 2604.12945 | [PDF](https://arxiv.org/pdf/2604.12945v1)

**作者:** Amar Gahir `[一作]` (University of Nottingham), Shreyank N Gowda `[通讯]` (University of Nottingham)

**通讯引用:** 521 | [OpenAlex ID](https://openalex.org/A5041351493)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种自适应数据丢弃框架（Adaptive Data Dropout），在训练过程中根据模型性能动态调整训练样本的子集大小。

**💡 创新点**

核心创新在于将数据使用从固定进度表迁移为基于训练反馈的自适应控制，引入随机接受-拒绝机制实现“自我调节学习”，避免了过度或不足的数据使用。

**🔧 技术方法**

技术上主要采用两种变体：Adaptive-α（自适应衰减参数）和 Adaptive-T（自适应保留比例），结合随机采样、有效 epoch（EE）评估和阈值/回热因子来动态更新子集规模。

**📊 数据集**

在 CIFAR-10、CIFAR-100、ImageNet 以及 ImageNet 预训练微调等标准图像分类数据集上进行实验，覆盖 MobileNetV2、EfficientNet、EfficientFormer、ResNet-50、ViT-B-MAE 等主流架构。

**📈 对比分析**

与传统完整训练、静态数据丢弃（DBPD）、以及多种高效训练基线（DataDiet、IES、InfoBatch 等）比较，显示 Adaptive Data Dropout 在保持或略提升准确率的同时，显著减少有效 epoch（例如 ImageNet 上从 350 EE 降至 111‑134 EE），实现更优的准确率‑效率 Pareto 前沿。

**⚠️ 局限性**

局限性包括：需要调节若干控制超参数（阈值、衰减函数、回热因子）；受训练反馈噪声影响，尤其早期阶段；仅在图像分类上验证，其他视觉任务的适用性待进一步评估；仍需预设初始调度框架，无法完全自动化适应策略。

---

## 484. Direct Discrepancy Replay: Distribution-Discrepancy Condensation and Manifold-Consistent Replay for Continual Face Forgery Detection

**arXiv ID:** 2604.12941 | [PDF](https://arxiv.org/pdf/2604.12941v1)

**作者:** Tianshuo Zhang `[一作]` (University of Chinese Academy of Sciences), Zhen Lei `[通讯]` (University of Chinese Academy of Sciences)

**通讯引用:** 27129 | [OpenAlex ID](https://openalex.org/A5109299788)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9cc9baba-5356-466d-81ff-d80028d90279` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `67630363-6be0-4f51-ab05-7198250671a5`

**🎯 论文内容**

提出了一种在连续面部伪造检测任务中，直接复现前置任务伪造分布差异的回放框架，称为Distribution‑Discrepancy Condensation（DDC）与Manifold‑Consistent Replay（MCR）。

**💡 创新点**

创新点在于：① 将真实与伪造分布的差异用特征空间中的特征函数（Characteristic Function）进行因子分解，并将差异压缩成可学习的分布差异图（DDM）存储；② 在后续任务中通过方差保持的DDPM‑style合成，将DDM与当前真实样本结合，生成既符合历史伪造统计又保持能量均衡的回放样本；③ 在极小存储预算下无需保存原始人脸图像，显著降低身份泄露风险。

**🔧 技术方法**

技术细节包括：特征函数匹配、频率采样器（qψ）与最小‑最大优化、分布差异图的学习与标准化、方差保持合成、以及使用EfficientNet‑B4作为检测器的联合训练。

**📊 数据集**

使用的主要数据集为FaceForensics++（FF++）、Deepfake Detection Challenge Preview（DFDC‑P）、Deepfake Detection（DFD）和Celeb‑DF v2（CDF2），以及混合/增量伪造类型的DF40子集。

**📈 对比分析**

与DER、CoReD、DFIL、DMP、HDP、SUR‑LID、KAN‑CFDAAAI等基线在两种增量协议（P1：数据集增量，P2：伪造类型增量）下进行对比。实验结果表明，DDC‑MIR在不额外添加持续学习模块时即可实现最高的最终准确率（P1 91.08 AA/4.34 AF；P2 95.23 AA/2.10 AF），并在与SUR‑LID集成后进一步刷新SOTA。回放样本在分布一致性、线性探测性能及隐私评估（身份链接性与视觉相似度）方面均优于现有方法。

**⚠️ 局限性**

局限性包括：① 仍需训练和存储若干DDM，尽管量小但不完全无存储需求；② 方法依赖特征函数匹配和频率采样，可能对网络架构或特征空间变化敏感；③ 仅针对面部伪造场景验证，尚未证明对其他模态或更大规模伪造类型的通用性；④ 在极端资源受限环境下（如极少的计算或极小的内存）仍需进一步简化。

---

## 485. MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models

**arXiv ID:** 2604.12928 | [PDF](https://arxiv.org/pdf/2604.12928v1)

**作者:** Chung-Ming Chien `[一作]` (Toyota Technological Institute at Chicago), Alexandre Défossez `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了Moshi，一种首次将检索增强生成（RAG）集成到全双工语音语言模型的系统，实现实时检索并生成更准确信息的语音助手。

**💡 创新点**

通过利用语音响应的关键词延迟实现异步检索，保持实时互动；构建可插拔检索后端；使用参考文本编码器和压缩网络；以及在训练中模拟检索延迟。

**🔧 技术方法**

全双工语音LM Moshi、RAG、检索触发器、ARC-Encoder压缩、Gemma3 27B、GPT-4.1、Tavily搜索、流式ASR、LLM提示等技术。

**📊 数据集**

结合Natural Questions、HotpotQA、TriviaQA、HaluEval、OpenAudioBench、CommonVoice等公开数据集，并用Gemma生成的对话脚本和专家领域话题，构建约190万条语音对话。

**📈 对比分析**

在语音QA、HaluEval、数学推理等基准上与多种公开全双工和非全双工模型对比，Moshi‑RAG在事实性得分提升至约80+，达到或超过多数模型；E2EKD 低于其他模型；在Full‑Duplex‑Bench上保留低TOR和低延迟。

**⚠️ 局限性**

仍受检索触发器依赖训练数据的限制；检索延迟和信息质量波动；对检索后端API稳定性敏感；对误检索导致的误信息、偏见等问题未完全解决。

---

## 486. Radar-Camera BEV Multi-Task Learning with Cross-Task Attention Bridge for Joint 3D Detection and Segmentation

**arXiv ID:** 2604.12918 | [PDF](https://arxiv.org/pdf/2604.12918v1)

**作者:** Ahmet İnanç `[一作]` (Hacettepe University), Özgür Erkent `[通讯]` (Hacettepe University)

**通讯引用:** 619 | [OpenAlex ID](https://openalex.org/A5003475208)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于雷达-摄像头融合的鸟瞰视图（BEV）多任务框架，联合完成3D目标检测与BEV地图分割。

**💡 创新点**

创新点在于引入跨任务注意力桥（CTAB），通过多尺度可变形注意力实现检测与分割分支之间的双向特征交换，并配备置信度门控以控制信息流。

**🔧 技术方法**

核心技术包括雷达-摄像头BEV融合、实例归一化（IN）与组归一化（GN）网络、可变形注意力（MSDA）、学习的BEV上采样模块以及同质不确定性权重（HUW）多任务损失。

**📊 数据集**

在nuScenes数据集上进行实验，使用6摄像头和5雷达的多视角数据。

**📈 对比分析**

与单任务检测或分割方法对比，CTAB多任务模型在检测上维持≈55.6 NDS（与单任务差距≈1.2 NDS），分割mIoU-7提升0.6点、mIoU-4提升0.6点，且仅增加0.58M参数。

**⚠️ 局限性**

局限性包括：上采样模块无法恢复BEV投影过程中的细节损失；使用的ResNet-50背骨相比ViT-B在分割精度上仍有差距；置信度门只能全局调节，未针对空间位置进行细粒度控制。

---

## 487. Robotic Manipulation is Vision-to-Geometry Mapping ($f(v) \rightarrow G$): Vision-Geometry Backbones over Language and Video Models

**arXiv ID:** 2604.12908 | [PDF](https://arxiv.org/pdf/2604.12908v1)

**作者:** Zijian Song `[一作]` (Sun Yat-sen University), Guangrun Wang `[通讯]` (Sun Yat-sen University)

**通讯引用:** 3069 | [OpenAlex ID](https://openalex.org/A5052611320)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种基于 3D 视觉-几何-动作（VGA）框架，通过预训练的 3D 世界模型（VGGT）直接映射视觉输入到物理动作，实现无额外 3D 传感器的精确操控；

**💡 创新点**

创新点在于：①放弃传统的视觉‑语言或视频先验，改用原生 3D 视觉 backbone；②引入 Progressive Volumetric Modulation（PVM）模块实现 3D 信息与动作生成的高保真交互；③采用联合训练同时预测动作与 3D 属性，提升表征一致性与泛化；④通过 LoRA 微调保留预训练 3D priors。

**🔧 技术方法**

技术核心包括：VGGT 预训练的 Transformer 3D backbone；多模态 token 化（RGB、语言、关节状态）；交替注意力（局部+全局）实现 3D 表征；PVM 进阶体积调制；动作 head（Transformer 回归 + chunking）；联合损失（动作+相机参数+深度）；LoRA 微调；在推理阶段使用解耦的动作 head。

**📊 数据集**

数据集：1) LIBERO 仿真基准（四个任务套件，约 400 条演示）；2) 真实世界实验在 Franka Panda + RealSense D415 上的 3 个操控任务（Pick Cube、Press Button、Stack Cube），使用 80‑100 条演示。

**📈 对比分析**

与多种基线对比：VLA（π_0.5、OpenVLA、ACT）、3D‑VLA（SpatialVLA、GeoVLA、GeoAwareVLA）以及 WAM（WorldVLA、UniMimic 等）。在 LIBERO 上 VGA 取得 99.0%/99.6%/98.6% 的平均成功率，明显优于 π_0.5（98.8%/98.2%/98.0%）和 GeoVLA（98.4%/99.0%/96.6%）。在真实环境中，VGA 在未见视角下的零样本泛化成功率约 58%，超过 π_0.5 的 52%（提升 6%），在分布内表现同样优于 ACT 与 OpenVLA。

**⚠️ 局限性**

局限性：①对长时序任务（LIBERO‑Long）表现略逊，因 VGGT 预训练缺乏长序列建模；②依赖高质量的 3D pretrain（VGGT），若无此预训练或数据量不足，效果可能下降；③在极端动态环境或需要实时深度估计的场景中，PVM 与 3D 属性预测仍可能受限；④模型规模仍较大，推理时需一定算力。

---

## 488. Towards a Linear-Algebraic Hypervisor

**arXiv ID:** 2604.12902 | [PDF](https://arxiv.org/pdf/2604.12902v1)

**作者:** Breandan Considine `[一作]` `[通讯]`, Breandan Considine

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文构建了一个基于 RASP 的并行虚拟机，可在 GPU 上并发执行数百万个数组程序，显著加速程序合成和实验研究。

**💡 创新点**

创新点包括：① 将 RASP 转化为可向量化的多项式状态机；② 采用无堆数组下推和无分支（branch‑less）实现，降低 GPU warp Divergence；③ 设计了轻量级 hypervisor 与 round‑robin 调度，实现大规模 VM 的高效并行。

**🔧 技术方法**

主要技术手段包括：抽象机模型（RASP、Word‑RASP）、多项式状态转移、GPU 并行编程（CUDA）、Kotlin/JVM 实现、低开销调度与内存布局优化。

**📊 数据集**

使用了 8×10⁶ 条长度为 100 的随机数组程序样本，样本通过 Considine 的单词采样器在合法代码空间内均匀生成。

**📈 对比分析**

通过在 Apple M4 Max（JVM）与 NVIDIA A10G、B200 GPU 上对同一程序集进行时间对比，展示了 GPU 版实现的对数时间复杂度与最高 147× 的加速比。

**⚠️ 局限性**

局限性：① 仅适用于小规模、无堆 IR，难以直接迁移到更复杂语言或大内存程序；② 评估停机概率上限为 10⁶ 步，可能忽略更长周期的停机行为；③ 依赖 GPU 专用实现，跨平台可移植性受限。

---

## 489. Evolution of Optimization Methods: Algorithms, Scenarios, and Evaluations

**arXiv ID:** 2604.12968 | [PDF](https://arxiv.org/pdf/2604.12968v1)

**作者:** Tong Zhang `[一作]` (Zhejiang University), Shuicheng Yan `[通讯]` (National University Of Singapore)

**通讯引用:** 140804 | [OpenAlex ID](https://openalex.org/A5100381753)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述深度学习优化方法，提出统一分类、场景化分析并构建标准评测框架

**💡 创新点**

创新点在于整合FO、SO、ZO与场景优化为完整理论体系，并提供大规模基准测试

**🔧 技术方法**

采用数学统一框架、标准化实验平台以及多任务跨架构评估

**📊 数据集**

在ResNet、ViT以及LLM Llama等视觉与语言模型上进行测试

**📈 对比分析**

与23种优化器在多任务上对比，发现FO算法易崩溃、Adam对LR敏感、Muon's矩阵正交化提升鲁棒性

**⚠️ 局限性**

局限在于仍缺乏针对分布式与隐私优化的深入理论与统一评价，且基准覆盖仅限于目前公开模型

---

## 490. Cycle-Consistent Search: Question Reconstructability as a Proxy Reward for Search Agent Training

**arXiv ID:** 2604.12967 | [PDF](https://arxiv.org/pdf/2604.12967v1)

**作者:** Sohyun An `[一作]` (Meta Superintelligence Labs), Alexander Min `[通讯]` (Meta Superintelligence Labs)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Cycle‑Consistent Search（CCS），一种无需金标监督的搜索代理训练框架，利用搜索轨迹与问题的循环一致性来生成奖励信号。

**💡 创新点**

创新点在于把搜索轨迹视为问题的无损编码，并通过信息瓶颈（排除最终回复、实体遮蔽）消除词汇泄露，从而以重构质量作为代理性能的代理指标。

**🔧 技术方法**

核心技术包括信息理论中的互信息最大化、循环一致性重构、命名实体识别遮蔽、最终回复排除，以及基于Group Relative Policy Optimization（GRPO）的强化学习训练。

**📊 数据集**

实验使用多跳推理数据集HotpotQA、2WikiMQA、MuSiQue、Bamboogle以及通用问答数据集Natural Questions、TriviaQA、PopQA，并在Qwen系列模型上进行评估。

**📈 对比分析**

与模型推理、金标监督（SFT、Search‑R1）和无金标监督方法（RLIF、Constitutional Judge、TTRL）对比，CCS在所有无金标基线中均取得最佳平均表现，并在某些基准上与金标监督方法相当甚至略优。

**⚠️ 局限性**

局限性包括对重构模型的依赖、可能无法完全抑制微妙的词汇泄露、奖励与最终答案准确性不完全一致，以及在极端复杂或开放式任务中仍需进一步验证。

---

## 491. DINO-Explorer: Active Underwater Discovery via Ego-Motion Compensated Semantic Predictive Coding

**arXiv ID:** 2604.12933 | [PDF](https://arxiv.org/pdf/2604.12933v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 492. Improving Network Clock Synchronization by Marking Congestion

**arXiv ID:** 2604.12961 | [PDF](https://arxiv.org/pdf/2604.12961v1)

**作者:** Yash Deshpande `[一作]` (Technical University of Munich), Wolfgang Kellerer `[通讯]` (Technical University of Munich)

**通讯引用:** 9787 | [OpenAlex ID](https://openalex.org/A5021781616)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

在网络时钟同步中引入基于拥塞标记的误差校正机制，利用标记信息减少因排队延迟造成的偏移误差。

**💡 创新点**

提出了低复杂度、后向兼容的拥塞标记计数器，可在现有IP/PTP/NTP头部实现，并支持多跳累计校正，显著优于传统透明时钟。

**🔧 技术方法**

使用ECN位、PTP保留字段、NTP扩展字段以及P4（Tofino）实现标记；通过Markov模型和OMNeT++仿真对算法进行理论与实验验证。

**📊 数据集**

采用真实硬件（Tofino、EdgeCore）收集的队列延迟数据，以及Sim2HW生成的多拓扑流量数据集进行评估。

**📈 对比分析**

与最小RTT/中位数滤波以及传统TC相比，单跳提升高达80%，多跳在不同阈值下可实现30%–90%的RMS误差降低，且与现有统计滤波器兼容。

**⚠️ 局限性**

受限于头部可用位数、标记误分类率以及在极端流量波动下的鲁棒性，且需在每跳上实现计数器，可能影响高吞吐量环境的资源占用。

---

## 493. Modeling Co-Pilots for Text-to-Model Translation

**arXiv ID:** 2604.12955 | [PDF](https://arxiv.org/pdf/2604.12955v1)

**作者:** Serdar Kadioglu `[一作]` (Fidelity Investments), Akash Singirikonda `[通讯]` (Brown University)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Text2Model 与 Text2Zinc 两大组件，构建基于 LLM 的文本到模型翻译 Co‑Pilot 系统，并提供统一、solver‑agnostic 的 MiniZinc 数据集和在线 Leaderboard。

**💡 创新点**

创新点包括：① 将满意度与优化问题统一于同一框架；② 采用知识图和语法编码等中间表示提升生成质量；③ 提供交互式编辑器和公开 Leaderboard，促进社区快速迭代；④ 在 MiniZinc 上实现多种 LLM 生成策略的系统化对比。

**🔧 技术方法**

使用技术涵盖：大型语言模型（GPT‑5.2、GPT‑4、GPT‑4o 等）、Chain‑of‑Thought 推理、知识图中间表示、语法约束生成、Agentic 子任务拆分、MiniZinc/FlatZinc 编译与验证、自动化执行与解算准确率评估。

**📊 数据集**

数据集为 Text2Zinc，包含 1775 条自然语言实例（110 条高质量验证实例），综合自 LPW、Nlp4lp、ComplexOR、LPWP、CspLib、Hakank 等多源数据。

**📈 对比分析**

评估方式：执行准确率 (E_acc) 与解算准确率 (S_acc)。多种策略（ZeroShot、CoT、KG、Grammar、Agentic 等）与 Gala、Orlm、OptiMind 等先行方法对比，最佳方案在 57%–85% 范围内，整体仍有约 27% 的实例难以通过 LLM 正确生成模型。

**⚠️ 局限性**

限制：LLM 生成的模型仍常出现语法或逻辑错误，知识图与语法约束提升有限；解决方案依赖于手工校验；数据集覆盖多样但规模仍受限；目前模型对不同求解器/范式的泛化能力不足，尚需进一步提升性能与鲁棒性。

---

## 494. GlintMarkers: Spatial Perception on XR Eyewear using Corneal Reflections

**arXiv ID:** 2604.12949 | [PDF](https://arxiv.org/pdf/2604.12949v1)

**作者:** Seungjoo Lee `[一作]` (Carnegie Mellon University), Mayank Goel `[通讯]` (Carnegie Mellon University)

**通讯引用:** 3500 | [OpenAlex ID](https://openalex.org/A5030011411)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用眼球内镜像与自发光的反射标记，实现XR眼镜在不使用外向摄像头的情况下，通过用户的凝视获取目标物体的身份、姿态与距离，实现无缝的眼球空间感知与交互。

**💡 创新点**

创新点在于将角膜视为镜面利用其反射图像来同时捕获凝视方向和环境信息；提出了适合低分辨率、低对比度的被动反射标记，并设计了基于角膜反射的PnP估计框架和一对一用户校准方法。

**🔧 技术方法**

使用近红外LED洪流照明、角膜反射图像采集、深度学习求瞳孔定位、Blob检测、DBSCAN聚类、PnP（IPPE）姿态估计、卷积网络、光流追踪等技术。

**📊 数据集**

实验数据集由5名受试者在机器人臂下收集的角膜反射图像以及10名受试者在不同距离、姿态下的评估数据构成；同时使用ArUco fiducial作为地面真值。

**📈 对比分析**

与ArUco基准相比，姿态平均绝对误差约为5°–15°，距离平均误差约0.14 m（2–4 m范围内），多目标识别帧级准确率95–99%，窗口投票后达到96–99%。

**⚠️ 局限性**

局限性包括对极端曲面或大范围光照变化的鲁棒性不足、单眼摄像导致遮挡与镜面畸变、需要用户一次性校准、依赖近红外照明且在亮光环境下表现不佳。

---

## 495. E2E-Fly: An Integrated Training-to-Deployment System for End-to-End Quadrotor Autonomy

**arXiv ID:** 2604.12916 | [PDF](https://arxiv.org/pdf/2604.12916v1)

**作者:** Fangyu Sun `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2322 | [OpenAlex ID](https://openalex.org/A5019803400)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本研究构建了一个完整的端到端视觉控制学习平台E2E-Fly，实现了从模拟训练、验证到真实无人机部署的全流程闭环；

**💡 创新点**

创新点在于首次将可微物理学习与强化学习、结构化奖励设计、两阶段验证（sim‑to‑sim 与硬件对准）以及四步仿真‑实境对齐方法统一整合到单一框架，实现零样本实境迁移；

**🔧 技术方法**

采用了可微仿真器VisFly、BPTT梯度学习、PPO强化学习、Habitat‑Sim渲染、Betaflight‑ctrl低层控制桥接，以及系统辨识、延迟补偿、域随机化和噪声建模等技术；

**📊 数据集**

主要使用内部自生成的仿真数据进行训练与验证，任务包括悬停、着陆、跟踪、竞速、视觉着陆和障碍竞速；真实验证则基于自行搭建的两款硬件平台和运动捕捉系统；

**📈 对比分析**

通过BPTT与PPO对比，BPTT在所有任务上收敛速度快、样本效率高、最终奖励更优；在真实飞行中实现了无参数调优的零样本迁移，成功完成六项任务；

**⚠️ 局限性**

局限性包括可微奖励设计难度大、梯度爆炸/消失问题在长时延任务中突出，以及RL在复杂环境下的探索效率低，需进一步融合两种学习方法以提升稳健性与泛化能力。

---

## 496. A Sanity Check on Composed Image Retrieval

**arXiv ID:** 2604.12904 | [PDF](https://arxiv.org/pdf/2604.12904v1)

**作者:** Yikun Liu `[一作]` (Shanghai Jiao Tong University), Yanfeng Wang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15049 | [OpenAlex ID](https://openalex.org/A5100645706)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一个基于生成模型的完全信息多样化基准 FISD，并构建了自动化多轮交互评估框架，用于评估组合图像检索（CIR）模型在交互场景中的表现。

**💡 创新点**

创新点在于：①通过扩散模型和 LLM 生成可控的参考-目标图像三元组，消除传统基准中的歧义查询；②设计自动化多轮交互流程，利用 MLLM/LLM 生成用户反馈，实现对模型在多轮检索中的性能动态评估。

**🔧 技术方法**

主要技术包括扩散模型（Stable‑Diffusion‑XL）、大型语言模型（Mixtral‑8x7B、Llama3‑LLaVA‑Next 等）、多模态 LLM（BLIP‑2、LLaVA‑1.5）、以及传统的 CIR 模型与排名器（余弦距离、历史向量平均）。

**📊 数据集**

使用的数据集为：自建的 1200 个三元组（共 3600 张图像）组成的 FISD，公开基准 FashionIQ、CIRR、CIRCO 以及其验证集。

**📈 对比分析**

评估方法采用 Recall@1、Hits@K 等指标，单轮检索结果通常低于 50%，而多轮交互后可提升约 60–70%（如 Pic2Word 在 CIRR 上 Hits@1 从 23.25% 提升至 40.28%）。实验显示模型在否定与计数语义上仍表现差强人意。

**⚠️ 局限性**

局限性包括：FISD 规模相对有限且为合成图像，可能与真实图像存在差距；评估过程高度依赖所选 LLM/MLLM，未提出训练新的多轮 CIR 模型；以及缺乏对更复杂交互策略的探索。

---

## 497. Representing 3D Faces with Learnable B-Spline Volumes

**arXiv ID:** 2604.12894 | [PDF](https://arxiv.org/pdf/2604.12894v1)

**作者:** Prashanth Chandran `[一作]` (Google), Timo Bolkart `[通讯]` (Google)

**通讯引用:** 4199 | [OpenAlex ID](https://openalex.org/A5025958423)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

设计了一种新型面部几何表示，结合B‑Spline体积与高维可学习控制特征，提出两阶段解码器，并通过Transformer实现从无序点云或单张图像直接预测完整面部网格，完成快速扫描配准与面部重建实验。

**💡 创新点**

创新点包括：① 用高维控制特征取代传统3D控制点，显著提升表达能力；② 采用两阶段解码（B‑Spline + 轻量MLP）实现精细细节；③ 保留B‑Spline局部支持，实现可局部编辑；④ 通过Transformer一次性回归控制特征，避免多步优化。

**🔧 技术方法**

技术要素：B‑Spline特征体积、三维B‑Spline插值、残差MLP、ViT/Transformer编码器（XCiT注意力）、点云/图像token化、Fourier位置嵌入、点到扫描距离与顶点到顶点距离评估。

**📊 数据集**

使用300k合成扫描-网格对（Blender+多视角重建生成），18k真实扫描（多视角捕获）以及CoMA与FaMoS公开扫描集做泛化测试。

**📈 对比分析**

与BPS（点云基）和TEMPEH（图像基）基线在点到扫描距离(PTS)和顶点到顶点距离(V2V)上进行比较；该方法在PTS≈1.1–1.2mm、V2V≈1.8–1.9mm，明显优于BPS（≈4–5mm）和TEMPEH（≈1.5–2mm），实现了state‑of‑the‑art的前向扫描配准。

**⚠️ 局限性**

局限性：需预设固定模板网格以保证语义对应；控制点数决定细节范围，过少需依赖残差MLP补偿；训练依赖大量合成数据，对极噪声或稀疏扫描的鲁棒性有限；实现成本较高（Transformer + 大量高维特征）。

---

## 498. GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts

**arXiv ID:** 2604.12978 | [PDF](https://arxiv.org/pdf/2604.12978v1)

**作者:** Amir Hossein Kargaran `[一作]` (LMU Munich), Hinrich Schütze `[通讯]` (LMU Munich)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个覆盖158种Unicode脚本的OCR基准，提供干净与降解两种图像变体，并评估14款公开与专有的视觉-语言模型的跨脚本泛化能力。

**💡 创新点**

创新点在于：①首次系统评估OCR在大规模脚本多样性上的表现；②通过字体选择与文本渲染管线保证跨脚本渲染质量；③揭示跨脚本误识（hallucination）和脚本识别误差的关键瓶颈。

**🔧 技术方法**

使用Harfbuzz进行文本形变，FreeType进行光栅化；对模型进行零样本推理，采用CER、Acc@5、ScriptAcc等指标；对图像做降解增强（纹理、污渍、几何失真、压缩）。

**📊 数据集**

数据来源于GlotLID v3、Wiktionary、WikiSource、Omniglot、Google Fonts、Common Crawl等多源文本，覆盖100多种脚本，约16,375句子；每脚本最多采集100句，Latin采集4,000句。

**📈 对比分析**

通过宏平均Acc@5与CER对比模型，发现Latin脚本Acc@5>90%，mid资源脚本降至约60%，低资源脚本低于5%，最优秀模型Gemini 3.1 Flash‑Lite在低资源脚本Acc@5仅7.7%；脚本识别准确率高时OCR仍受限于视觉与预训练覆盖；脚本提示提升平均+0.7pp。

**⚠️ 局限性**

局限性包括：①脚本样本极度不平衡，低资源脚本样本不足导致结论受限；②未覆盖纵向书写和混合代码点场景；③评估仅基于零样本推理，未考虑微调或自监督增强；④对模型内部机制缺乏解释性分析。

---

## 499. AbdomenGen: Sequential Volume-Conditioned Diffusion Framework for Abdominal Anatomy Generation

**arXiv ID:** 2604.12969 | [PDF](https://arxiv.org/pdf/2604.12969v1)

**作者:** Yubraj Bhandari `[一作]` (Duke University), Joseph Y. Lo `[通讯]` (Duke University)

**通讯引用:** 7613 | [OpenAlex ID](https://openalex.org/A5040192736)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

提出 AbdomenGen，基于顺序体积条件扩散的框架，用于可控生成腹部解剖结构，并通过体积控制标量实现器官尺寸的精确调节。

**💡 创新点**

创新点包括：①体积控制标量（VCS）将器官体积与体型解耦，提供可解释的控制轴；②顺序、上下文感知的生成策略使各器官在保持空间一致性的同时实现独立调节；③通过 VCS 的分布映射实现对临床目标分布的匹配，无需再训练。

**🔧 技术方法**

采用 3D denoising diffusion U‑Net 搭配 FiLM 适配器、SDF 表示、VCS 标准化残差、DDIM 采样以及 Wasserstein‑优化的 VCS 选择等技术。

**📊 数据集**

使用 556 例腹部 CT（分 314/42/200 训练/验证/测试），以及外部 MERLIN 肝肿大队列进行分布匹配评估。

**📈 对比分析**

通过 Dice、ASSD、HD95、Chamfer 等几何指标与 MAISI 等方法比较；在肝、脾、肾、胃等器官上分别取得 Dice 0.83、0.68、0.60、0.68，体积控制表现线性，且在 MERLIN 队列上将 Wasserstein 距离降低 73.6%，生成样本保持在训练流形上。

**⚠️ 局限性**

局限性包括：顺序生成可能累计几何误差；VCS 仅调节体积，无法完全控制局部形态；对细长、管状结构（如结肠）生成效果相对较弱。

---

## 500. Efficient Retrieval Scaling with Hierarchical Indexing for Large Scale Recommendation

**arXiv ID:** 2604.12965 | [PDF](https://arxiv.org/pdf/2604.12965v1)

**作者:** Dongqi Fu `[一作]` (Meta), Chonglin Sun `[通讯]` (Meta)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大规模基础检索模型（如Meta Ads使用的MoNN），提出了一种Hierarchical Index Learning（HILL）方法，在模型内学习层级索引结构，以实现高效精准检索，并通过生成新的高质量数据对模型进行测试时微调。

**💡 创新点**

创新点主要包括：
- 结合交叉注意力和残差量化共同学习层级索引，保持信息完整性；
- 在索引构建过程中与检索模型共训练，获得与模型嵌入一致的索引节点；
- 将中间索引节点作为新的训练样本，实现测试时训练（Test‑Time Training）；
- 提供EM‑FAISS近似学习方案，满足资源受限场景。

**🔧 技术方法**

核心技术：MoNN基础检索框架、交叉注意力（cross‑attention）、残差量化（residual quantization）、温度调度、均衡索引分布、线性warmup、EM‑FAISS聚类、对齐和重构损失、基于节点的兴趣率筛选、在线A/B测试。

**📊 数据集**

使用的数据集：
- 内部 Meta Ads 日常广告推荐数据（数十亿用户/百万级物品）；
- 公开基准数据集 Gowalla、Yelp 2018、Amazon‑Book。

**📈 对比分析**

方法比较：
- 与多种传统协同过滤、神经CF、图神经网络等基线以及工业检索模型（TTSN、EBR、MoNN各规模）对比；
- 在公开数据集上，HILL 的 Recall@20 由 0.1908 提升至 0.1924，NDCG@20 由 0.1614 提升至 0.1628，Normalized Entropy 下降至 0.0625；
- 在线 Meta Ads A/B 测试中，2‑Layer MoNN（大+小）提升广告指标 2.57%；
- ablation 证明温度调度最关键，EM 近似仅略逊于完整 HILL。

**⚠️ 局限性**

局限性：
- 完整 HILL 训练对算力与时间要求高，资源受限时需采用 EM 近似；
- 生成的测试时数据依赖索引结构的质量，若索引失衡可能导致微调噪声；
- 目前仅针对 MoNN 框架验证，其他基础检索模型的迁移性待进一步探索；
- 仍需进一步研究如何自动化选择 φ_DEP 与 φ_IR 参数以获得最优性能。

---

## 501. Distorted or Fabricated? A Survey on Hallucination in Video LLMs

**arXiv ID:** 2604.12944 | [PDF](https://arxiv.org/pdf/2604.12944v1)

**作者:** Yiyang Huang `[一作]` (Northeastern University), Yun Fu `[通讯]` (Northeastern University)

**通讯引用:** 31731 | [OpenAlex ID](https://openalex.org/A5005819096)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

综述了视频大型语言模型（Vid-LLMs）中的幻觉问题，提出了基于机制的分类体系，并系统评估现有的评测基准与缓解技术。

**💡 创新点**

创新点在于将幻觉细分为动态失真与内容伪造两大核心类型及其子类，构建了可分离且互斥的标签框架，并将根因与技术路径紧密关联。

**🔧 技术方法**

主要引用了对齐优化、对比学习、强化学习、注意力重塑、记忆机制等多种技术手段来缓解幻觉，但未提出新的算法。

**📊 数据集**

综述涵盖了20余个公开视频评测基准，如 VidHalluc、HAVEN、VideoHallucer、ELV-Halluc、AVHBench 等，但未使用原始数据集进行实验。

**📈 对比分析**

通过对比不同方法在各基准上的表现，发现基于对比学习与强化学习的模型在短视频上的准确率可达80%以上，但在长序列和音频冲突任务中仍低于50%，显示出方法在不同幻觉类型上的差异。

**⚠️ 局限性**

局限性包括缺乏对长视频、交互式流媒体以及智能体环境中的实证验证，且对音频-视觉冲突任务的评测仍不足，未能全面验证提出框架在新模型上的适用性。

---

## 502. RMGS-SLAM: Real-time Multi-sensor Gaussian Splatting SLAM

**arXiv ID:** 2604.12942 | [PDF](https://arxiv.org/pdf/2604.12942v1)

**作者:** Dongen Li `[一作]` (National University of Singapore), Marcelo H. Ang `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了实时多传感器（LiDAR–IMU–Camera）耦合的3D高斯斑点（3DGS）SLAM框架 RMGS‑SLAM，用于大规模场景的实时位姿估计和逼真映射。

**💡 创新点**

① 采用级联高斯初始化策略，将神经网络预测与体素‑PCA几何先验相结合；② 在全局高斯图上直接执行基于高斯 GICP 的闭环检测与优化；③ 前后端并行、无阻塞的密集映射。

**🔧 技术方法**

3D Gaussian Splatting、Voxel‑PCA 体素几何估计、深度/姿态无关的神经网络预测、光度/深度/结构损失联合优化、基于高斯的 GICP 闭环、姿态图优化（GTSAM）。

**📊 数据集**

自建同步 LiDAR–Camera–IMU 真实场景数据集（Driving1、Driving2），以及公开 FAST‑LIVO2、MARS‑LVIG 等序列。

**📈 对比分析**

与 MonoGS、SplaTAM、GS‑LIVM、Gaussian‑LIC2 等 3DGS‑SLAM 方法以及 FAST‑LIVO2 进行对比。RMGS‑SLAM 在 PSNR/SSIM/LPIPS 上均优于或相近，对齐误差（ATE）低至 0.41‑0.93 m，实时因子≈1，显示出优异的实时性、定位精度与渲染质量。

**⚠️ 局限性**

系统仍依赖高性能 GPU/CPU，循环闭合在大规模场景中成为瓶颈；内存占用随轨迹长度增长，且对硬件要求较高。

---

## 503. M3D-Stereo: A Multiple-Medium and Multiple-Degradation Dataset for Stereo Image Restoration

**arXiv ID:** 2604.12917 | [PDF](https://arxiv.org/pdf/2604.12917v1)

**作者:** Deqing Yang `[一作]` (Shenzhen University), Yibin Tian `[通讯]` (Shenzhen University)

**通讯引用:** 814 | [OpenAlex ID](https://openalex.org/A5102632157)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `6514db3d-8de6-452c-91b7-acdb31787cc4` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建并公开了M3D‑Stereo数据集，用于多介质、不同强度的立体图像恢复实验。

**💡 创新点**

首次将水下与大气散射/低光环境统一在同一基准中，并在每种场景下提供六级渐进式降解、像素对齐的清晰GT和深度GT，实现可控且真实的评估。

**🔧 技术方法**

利用双目摄像机校准、物理降解生成（乳液浊度、雾机、PWM低光）以及跨视角融合的立体恢复网络EPRRNet与PSIDNet。

**📊 数据集**

主要使用自建的M3D‑Stereo数据集；在实验中对比EPRRNet与PSIDNet。

**📈 对比分析**

采用PSNR、SSIM、ΔE等全参考指标，结果显示PSIDNet在所有降解级别均优于EPRRNet，且混合级训练提升了鲁棒性；立体恢复显著改善后续立体匹配深度估计。

**⚠️ 局限性**

局限在于实验规模受限于实验室小型场景，环境多样性不足；对偶合降解的组合有限；未覆盖更复杂的天气（雨尘等）或大规模自然环境。

---

## 504. Round-Trip Translation Reveals What Frontier Multilingual Benchmarks Miss

**arXiv ID:** 2604.12911 | [PDF](https://arxiv.org/pdf/2604.12911v1)

**作者:** Ronald Skorobogat `[一作]` (University of Tübingen), Matthias Bethge `[通讯]` (University of Tübingen)

**通讯引用:** 35142 | [OpenAlex ID](https://openalex.org/A5061457780)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了基于回译的多语言评测基准 LiT，证明现有多语言基准无法真实反映模型的多语言能力。

**💡 创新点**

创新点：①将回译作为无参考评估方法，直接衡量跨语言语义保留；②构建跨语言序列样本，覆盖高、中、低资源语言；③结合LLM-as-a-Judge和MQM框架实现大规模自动评估；④与真实用户偏好（LMArena）高度相关。

**🔧 技术方法**

使用技术包括：回译、LLM-as-a-Judge、MQM评估框架、统计相关分析（Spearman 相关、bootstrap 误差估计）等。

**📊 数据集**

使用的数据集：自建1600条样本（200 段落 × 8 语言序列），涵盖技术、语用、非正式文本；480 条鲁棒性回译样本；以及 LMArena 用户评分数据。

**📈 对比分析**

比较方法：对六款同级前沿模型（Thinking 与 Instruct）在 LiT 上计算 MQM_≥80 分数，比较高低资源语言表现；发现高资源语言模型表现优秀（如 Gemini-3-Flash 97%），低资源语言性能急剧下滑（如 Qwen3-235B 0%）；与 LMArena 的相关性为 ρ=0.94，明显优于传统多语言基准。

**⚠️ 局限性**

局限性：回译可能忽略中间错误导致最终评估偏差；序列翻译的错误堆叠难以定位单语种问题；缺乏文档级长距离依赖评测；自动评估仍需进一步验证与人工评估的一致性。

---

## 505. BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design

**arXiv ID:** 2604.12898 | [PDF](https://arxiv.org/pdf/2604.12898v1)

**作者:** Chuyang Xiang `[一作]`, Junchi Yan `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 BEAM，一种将 LLM 生成的启发式设计拆分为外层算法结构和内层函数实现的双层进化框架。

**💡 创新点**

创新点包括：1）将启发式设计视为双层优化问题，外层采用遗传算法演化高层框架；2）内层用 MCTS 细化占位符函数；3）引入自适应记忆模块，使优质低层函数可复用；4）设计统一的知识增强（HeuBase、KnoBase）管道，提升生成质量。

**🔧 技术方法**

核心技术：大型语言模型（LLM）生成代码；遗传算法（GA）进行结构搜索；蒙特卡罗树搜索（MCTS）实现函数细化；CMA-ES 进行超参校准；自适应记忆（AM）与知识增强（KA）模块。

**📊 数据集**

使用的数据集涵盖组合优化问题：CVRP、TSP、BPP、MIS、CAF 等；连续优化基准 BBOB；此外还使用了公开的启发式模板库和知识库。

**📈 对比分析**

实验通过与现有 LHH（ReEvo、EoH、MCTS-AHD、AlphaEvolve 等）以及最先进的专业求解器（KaMIS、HGS、ARW 等）在同等预算下对比。结果显示：在 CVRP 设计中平均降低 37.84% 的最优性缺口；在 MIS 中与 KaMIS 相当或略优；在 BBOB 上接近 SOTA 水平，整体性能显著优于对手。

**⚠️ 局限性**

局限性：1）双层结构导致首次生成需大量 token 与时间；2）对单函数生成任务增添不必要复杂度；3）性能对 LLM 规模仍有一定依赖；4）仅在选定基准上验证，未覆盖更复杂或多模态问题。

---

## 506. Don't Show Pixels, Show Cues: Unlocking Visual Tool Reasoning in Language Models via Perception Programs

**arXiv ID:** 2604.12896 | [PDF](https://arxiv.org/pdf/2604.12896v1)

**作者:** Muhammad Kamran Janjua `[一作]` (Huawei Technologies), Bahador Rashidi `[通讯]` (Huawei Technologies)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5a7d414a-27d1-4de0-aac0-e554088edeb4` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 Perception Programs（P^2）技术，将视觉工具的稠密像素输出转换为语言友好的结构化摘要，以便多模态语言模型（MLLM）直接读取并推理。

**💡 创新点**

创新点在于：1）无训练、无模型改动；2）统一的符号化框架可跨深度、光流、对应、拼图、语义对应、目标定位等多种视觉模态；3）将视觉信息映射为可读文本，显著减少视觉-语言不匹配问题。

**🔧 技术方法**

主要技术包括：工具输出的分块（如网格划分）、统计摘要（如最小/最大深度、平均流向）、符号关系生成、YAML‑style 语言文本序列化，并结合标准 LLM Prompting 进行推理。

**📊 数据集**

使用 BLINK 视觉感知子任务集（多视角推理、相对深度、视觉对应、拼图、语义对应、目标定位）以及对应的工具（DepthAnything、RAFT、LoFTR、SSIM+HSV+NCC、DIFT、LLMDet）。

**📈 对比分析**

与标准 MLLM（GPT‑5 Mini、Gemini 2.5 Pro、InternVL3.5‑2B/4B、Qwen3VL‑4B）以及多种先行方法（Thyme、LATTE、Visual Sketchpad、MMFactory 等）比较；P^2 在所有模型和任务上均实现平均 19–20% 的准确率提升，获得多项任务的全新 state‑of‑the‑art 结果。

**⚠️ 局限性**

局限性：1）仅验证六个 BLINK 子任务，未覆盖更复杂或多层次的感知场景；2）不支持动态工具选择或管道级联；3）错误会直接从基础工具传递，P^2 本身不对工具误差进行校正。

---

## 507. Towards Long-horizon Agentic Multimodal Search

**arXiv ID:** 2604.12890 | [PDF](https://arxiv.org/pdf/2604.12890v1)

**作者:** Yifan Du `[一作]` (Renmin University of China), Ji-Rong Wen `[通讯]` (Renmin University of China)

**通讯引用:** 24518 | [OpenAlex ID](https://openalex.org/A5025631695)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `67630363-6be0-4f51-ab05-7198250671a5` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种面向长时序的多模态深度搜索框架LMM-Searcher，通过文件化视觉表示和进阶工具接口，解决了多模态上下文爆炸问题；

**💡 创新点**

创新点在于将所有视觉资产持久化到文件系统并用轻量化UID引用，同时设计了按需加载的fetch-image工具与多模态工具链，实现对长时序搜索的可扩展管理；

**🔧 技术方法**

采用文件化视觉表示、进阶工具接口、主动视觉加载策略、数据合成管道、基于Qwen3-VL-Thinking的模型微调以及模型融合技术；

**📊 数据集**

使用自研的多模态查询合成数据（12.7k轨迹），以及FVQA、LiveVQA、REDSearcher-MM/Text等公开数据集；

**📈 对比分析**

与直接回答、代理工作流和多模态搜索代理进行对比，LMM-Searcher-30B在MM-BrowseComp、MMSearch-Plus、VisBrowse等四大基准上以100轮交互达到22.3/30.1、32.9/34.8等最高成功率，领先所有开源模型；

**⚠️ 局限性**

局限在于对大规模视觉资产的存储与检索成本、对文件系统依赖以及在极端长时序下仍可能面临策略不确定性和工具调用效率瓶颈。

---

## 508. Parallax: Why AI Agents That Think Must Never Act

**arXiv ID:** 2604.12986 | [PDF](https://arxiv.org/pdf/2604.12986v1)

**作者:** Joel Fokou `[一作]` `[通讯]` (Independent Researcher), Joel Fokou (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了一套面向自主 AI 代理执行的安全范式，核心包括认知‑执行分离、分层对抗验证、信息流控制与可逆执行四大原则，并在 Go 语言中实现了一个完整的参考实现；通过“假设代理被完全妥协”评估方法在 280 条攻击用例上验证其有效性。

**💡 创新点**

创新点在于：
• 将传统系统安全的权限分离与强制访问控制迁移到代理执行层，形成认知‑执行分离；
• 引入多层（Policy/Classifier/LLM/Human）对抗验证，并通过可配置阈值实现分层确定性；
• 采用标签化信息流控制，检测跨工具链的数据泄露；
• 引入可逆执行捕获，允许在后期回滚不可逆操作；
• 设计了“假设代理被妥协”评估框架，真正检验架构边界。

**🔧 技术方法**

主要技术包括：
• OS 进程隔离与 gRPC 交互实现认知‑执行分离；
• 四层分层验证器（YAML 规则、启发式+DeBERTa 分类器、预算有限 LLM 评估、人工审批）；
• 基于文件路径与内容模式的 IFC 标签系统；
• 状态快照与日志哈希链实现可逆执行与审计；
• 动态工具加载实现工具面板最小化。

**📊 数据集**

评估数据集：
• 280 条自定义攻击案例，覆盖 9 个攻击类别（注入、上下文操控、链式攻击等）；
• 50 条合法用例用于测量误报率；
• 5 条人工审批用例测试 Tier‑3 路径。

**📈 对比分析**

与传统基于提示级安全措施对比：
• 在默认配置下，98.9% 的攻击被拦截且 0% 误报；
• 在最高安全配置下实现 100% 拦截，误报率升至 36%；
• 验证延迟：Tier‑0 <1 ms，Tier‑1 平均 1.9 s（待优化），Tier‑2 平均 2.1 s；
• 相比单层提示拦截（100% 失败率），该范式在“假设妥协”测试中实现了几乎零成功率。

**⚠️ 局限性**

局限性：
• 依赖单一可信执行进程，若该进程被攻击整个体系失效；
• 只能在本地可逆，无法回滚外部 API 调用或发送的电子邮件等外部影响；
• IFC 标签和分类器的准确性取决于配置和训练数据，可能导致漏报或误报；
• Tier‑1 的 ONNX 推理目前开销较大，影响实时性能；
• 人工审批（Tier‑3）在高吞吐场景下成为瓶颈。

---

## 509. Boosting Visual Instruction Tuning with Self-Supervised Guidance

**arXiv ID:** 2604.12966 | [PDF](https://arxiv.org/pdf/2604.12966v1)

**作者:** Sophia Sirko-Galouchenko `[一作]` (Valeo.ai), Spyros Gidaris `[通讯]` (Valeo.ai)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

在视觉指令调优阶段加入少量自监督任务（旋转预测、颜色匹配、跨视角对应），以提升多模态大型语言模型对细粒度视觉推理的依赖与准确度。

**💡 创新点**

创新点在于将经典自监督预训练任务重构为自然语言指令-响应三元组，直接混入指令调优数据，既不改动模型结构也不增加额外损失，解决语言优先快捷策略导致的视觉信息低利用。

**🔧 技术方法**

使用自监督视觉任务转化为指令形式、标准自回归交叉熵训练、以及参数高效适配（LoRA）等技术；无需辅助损失、RL 或网络改造。

**📊 数据集**

基准数据集包括 LLaVA-NeXT‑780k/OneVision‑1.5 指令数据集、COCO 图像以及从单张图像生成的多视角样本；自监督任务的监督来自图像自身变换，无需人工标注。

**📈 对比分析**

通过在 CVB‑2D、POPE、MMStar、BLINK 等视觉基准上与 LLaVA‑1.5、LLaVA‑OneVision‑1.5 等基线模型（含 LoRA 与 VIRAL）对比，平均提升 3–10%，且在通用推理基准上保持甚至略有提升，证明提升来自视觉信息利用而非额外训练。

**⚠️ 局限性**

局限在于：仅针对视觉指令调优阶段有效；当加入过多 SSL 任务（>10%）收益趋于饱和或略降；在完全新任务或跨模态场景中的泛化仍需进一步验证。

---

## 510. An Optimal Sauer Lemma Over $k$-ary Alphabets

**arXiv ID:** 2604.12952 | [PDF](https://arxiv.org/pdf/2604.12952v1)

**作者:** Steve Hanneke `[一作]` (Purdue University), Amirreza Shaeiri `[通讯]` (Purdue University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

**🎯 论文内容**

提出了多类别和列表学习场景下的Sharp Sauer不等式，给出了以Daniely–Shalev‑Shwartz (DS) 维度为参数的最优组合数上界，并利用该不等式改进了列表PAC学习和统一收敛的样本复杂度。

**💡 创新点**

① 将传统基于Natarajan维度的指数级列表依赖改为最优多项式依赖；② 在k-元字母的通用设定下实现了对DS维度的完全等价计数；③ 在证明中首次使用多项式方法得到完全紧凑的上界，未出现任何经验性或近似估计。

**🔧 技术方法**

核心技术为多变量多项式方法（polynomial method）与线性代数的独立/生成集思想；在证明中构造了适当的向量空间、线性独立向量与基于多项式的生成集；同时利用一系列图论/网络流工具（如one‑inclusion图、流网络）来推导列表学习的误差上界。

**📊 数据集**

该工作为理论论文，未在具体数据集上实验；所有结果均基于纯理论推导与组合数计算。

**📈 对比分析**

与以往基于Natarajan维度的列表学习和统一收敛分析相比，新的不等式将样本复杂度从指数级降低到多项式级，理论上实现了最优的DS维度上界。该改进直接导致列表PAC学习的样本复杂度从 O(ℓ^6 d^{3/2}/ε) 或 O(ℓ^4 d^5/ε) 降到 O(ℓ d^{3/2}/ε)（忽略对数项），并把统一收敛样本复杂度从 O(ℓ^2 d + log(1/δ)/ε^2) 降到 O(ℓ d + log(1/δ)/ε^2)。

**⚠️ 局限性**

1) 证明依赖于代数/多项式方法，缺乏直接的组合学/几何直观，尚未给出完全组合学证明； 2) 结果主要针对可实现（realizable）学习场景，尚未讨论噪声或不完全可实现情况； 3) 只给出了最优上界，缺乏下界或紧紧度的全面分析； 4) 对于无限标签空间的具体实现细节仍待进一步研究。

---

## 511. Tree Learning: A Multi-Skill Continual Learning Framework for Humanoid Robots

**arXiv ID:** 2604.12909 | [PDF](https://arxiv.org/pdf/2604.12909v1)

**作者:** Yifei Yan `[一作]` (Shanghai University), Linqi Ye `[通讯]` (Shanghai University)

**通讯引用:** 465 | [OpenAlex ID](https://openalex.org/A5042034627)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出了Tree Learning框架，实现了仿人机型机器人多技能的连续学习与无缝切换，防止灾难性遗忘。

**💡 创新点**

创新点包括根-分支层级参数继承、物理隔离子网络结构、低成本技能迭代、周期与非周期动作的多模态前馈适配，以及任务级奖励塑造。

**🔧 技术方法**

采用Unity ML-Agents+PPO强化学习、层级神经网络参数继承、阶段化课程学习、奖励塑造与前馈动作设计等技术。

**📊 数据集**

使用Unity仿真环境（Unitree G1机器人）进行的Super Mario风格交互场景和中国古典园林导航任务；未使用公开数据集。

**📈 对比分析**

通过与多任务并行训练基线对比，Tree Learning在6项技能上获得更高最终奖励、100%技能保持率，并展示了流畅的技能切换和实时交互控制，整体性能显著优于基线。

**⚠️ 局限性**

仅在仿真环境验证，存在Sim-to-Real迁移难题；未涵盖长周期复杂动作（如舞蹈）和真实机器人动力学噪声，未来计划加入域随机化和自动动作重定向。

---

## 512. Frequency-aware Decomposition Learning for Sensorless Wrench Forecasting on a Vibration-rich Hydraulic Manipulator

**arXiv ID:** 2604.12905 | [PDF](https://arxiv.org/pdf/2604.12905v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 513. Joint Clustering and Prediction of the Quality of Service in Vehicular Cellular Networks

**arXiv ID:** 2604.12903 | [PDF](https://arxiv.org/pdf/2604.12903v1)

**作者:** Oscar Stenhammar `[一作]` (KTH Royal Institute of Technology), Carlo Fischione `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 6934 | [OpenAlex ID](https://openalex.org/A5077148700)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种联合聚类与预测的分布式优化框架，通过在每个聚类内训练专属的概率模型来预测长期QoS分布。

**💡 创新点**

创新点在于：①将聚类与模型训练联合建模为块坐标下降（BCD）问题；②使用核范数松弛与随机梯度投影实现聚类分配的连续化；③通过联邦学习共享模型梯度，减少通信开销并提升概念漂移适应性。

**🔧 技术方法**

采用的技术包括：多层感知机（MLP）预测多维高斯分布参数；核范数正则化与奇异值阈值化（SVT）实现低秩聚类；概率单纯形投影确保聚类分配合法；Cholesky 参数化保证预测协方差矩阵为正定；BCD 迭代优化聚类与网络参数；实验中还用到 Hellinger 核作为距离度量。

**📊 数据集**

使用基于 ns‑3 与 Sionna 结合的仿真生成的真实场景数据：12 个基站覆盖的城市道路网络，30 辆车辆、1 秒采样周期、包含网络、流量、移动与射频特征，共约 120000 条样本，目标是预测 1 小时后延迟、抖动与 RSRP 的联合分布。

**📈 对比分析**

与全局单一模型和每个基站独立训练的基线比较，聚类方法在 MAE 上比全局模型降低 9–27%，与局部模型相近但模型数量大幅减少；NLL 也表现最好，说明概率预测更可靠；实验展示了收敛稳定、聚类动态及对概念漂移的适应性。

**⚠️ 局限性**

局限性包括：①松弛导致的最优性缺口可达约 23%，未必逼近全局最优；②仅在 12 基站的小规模设置下评估，缺乏大规模验证；③仅考虑下行 QoS 与高斯/对数正态分布假设，未涵盖上行或非高斯指标；④BCL 与聚类更新周期需要手工调参，未实现完全在线自适应。

---

## 514. Task Alignment: A simple and effective proxy for model merging in computer vision

**arXiv ID:** 2604.12935 | [PDF](https://arxiv.org/pdf/2604.12935v1)

**作者:** Pau de Jorge `[一作]` (NAVER LABS Europe), Yannis Kalantidis `[通讯]` (NAVER LABS Europe)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了一种任务对齐代理（Task Alignment，TAP），通过比较不同任务模型的特征空间相似度，快速评估模型融合候选的性能，从而在有可训练解码器的视觉任务中高效地进行模型融合和超参数选择。

**💡 创新点**

创新点在于：①将无标签的特征相似度作为通用的性能代理，显著降低超参数搜索成本；②将这一代理推广到LiDAR点云语义分割和多任务（2D/3D）混合场景；③结合已有融合方法（如Task Arithmetic、SVD、Pruning等）以及AdaMerging的改进实现统一的超参数优化。

**🔧 技术方法**

技术手段包括：特征相似度计算（L2、L1或余弦距离）、模型融合策略（加权平均、Task Arithmetic、SVD截断、Pruning、Consensus等）、AdaMerging扩展、以及对TAP作为损失函数的使用。

**📊 数据集**

使用的数据集：CLIP分类基准（包含多种分类数据集），LiDAR语义分割四个数据集（nuScenes、SemanticKITTI、Panda64、PandaGT），以及DUNE异构任务集（语义分割、深度估计、3D人体网格恢复、视觉定位）。

**📈 对比分析**

与传统基线（完整评估、随机搜索、AdaMerging）相比，TAP选出的超参数在所有三大实验场景中都与完整评估得到的性能保持高度一致，甚至在某些情形下略优；同时，它将超参数搜索成本从几小时/天降低到几分钟，节省了两到三订单级的计算量。

**⚠️ 局限性**

局限性包括：①任务向量（fine‑tuned 任务表示）在异构任务场景中往往不平衡，导致现有融合方法的性能受限；②TAP仍假设特征空间相似度与下游性能高度相关，可能在极端任务差异或非常小的样本量下失效；③目前对大规模、极度多样化任务集的推广仍需进一步研究。

---

## 515. Parcae: Scaling Laws For Stable Looped Language Models

**arXiv ID:** 2604.12946 | [PDF](https://arxiv.org/pdf/2604.12946v1)

**作者:** Hayden Prairie `[一作]` (University of California, San Diego), Daniel Y. Fu `[通讯]` (University of California, San Diego)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种稳定的循环Transformer架构Parcae，通过限制注入参数的谱范数，避免残差爆炸，从而在相同参数与数据预算下实现更高的语言建模质量。

**💡 创新点**

创新点在于：①将循环模型视为非线性时变动力学系统，并用线性化分析得到稳定性条件；②在注入参数上使用负对角矩阵的离散化约束谱范数；③在训练中引入逐序列深度采样与预奏层归一化，显著降低训练过程中的损失尖峰。

**🔧 技术方法**

核心技术包括：控制理论中的线性时不变系统(LTI)分析、谱范数约束、ZOH与Euler离散化、Transformer块（多头注意力+FFN）、残差归一化、变量深度采样的训练策略。

**📊 数据集**

主要使用了Huginn、WikiText、Lambada、Core、Core-Extended、ARC-c、ARC-e、PIQA、BoolQ、SciQ等公开文本和下游评测数据集。

**📈 对比分析**

与参数与数据匹配的传统循环模型（RDMs）及标准Transformer进行对比，Parcae在验证困惑度上下降6.3%，在Core和Core-Extended上分别提升2.99和1.18分，且在1.3B参数规模时可匹配双倍参数Transformer的质量。

**⚠️ 局限性**

局限性：实验仅覆盖中等规模模型，尚未验证在更大FLOP预算和参数规模下的表现；循环深度增加导致推理步骤数上升，需进一步优化推理效率；不同离散化与全秩参数化方案的探索仍待深入。

---

## 516. Distinguishers for Skew and Linearized Reed-Solomon Codes

**arXiv ID:** 2604.12954 | [PDF](https://arxiv.org/pdf/2604.12954v1)

**作者:** Felicitas Hörmann `[一作]` (German Aerospace Center), Anna-Lena Horlemann `[通讯]` (University of St.Gallen)

**通讯引用:** 770 | [OpenAlex ID](https://openalex.org/A5057606881)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e`

**🎯 论文内容**

研究了广义里德-所罗门（GSRS）和广义线性化里德-所罗门（GLRS）编码的代数结构和可区分性，探讨其在基于编码的密码学中的应用。

**💡 创新点**

提出了GSRS和GLRS编码的显式变换，证明了在特定条件下GSRS编码的对偶也是GSRS编码，并且这为进一步分析这些编码家族提供了严格的代数框架。

**🔧 技术方法**

使用了广义里德-所罗门和广义线性化里德-所罗门编码的代数结构，构建了多项式时间的区分器，能够在大参数范围内将其与随机线性编码区分开来。

**📊 数据集**

使用了多种参数集的GSRS和GLRS编码进行实验，验证了理论结果的有效性。

**📈 对比分析**

通过构建的区分器，GSRS和GLRS编码在大参数范围内能够有效区分于随机线性编码，且在Hamming度量下的等距伪装下仍然有效。

**⚠️ 局限性**

尽管存在区分器，但这并不直接导致密钥恢复或对基于这些编码家族的McEliece类密码系统的完全破解，密钥恢复仍然是一个开放问题。

---

## 517. Grasp in Gaussians: Fast Monocular Reconstruction of Dynamic Hand-Object Interactions

**arXiv ID:** 2604.12929 | [PDF](https://arxiv.org/pdf/2604.12929v1)

**作者:** Ayce Idil Aytekin `[一作]` (Max Planck Institute for Informatics), Christian Theobalt `[通讯]` (Max Planck Institute for Informatics)

**通讯引用:** 35362 | [OpenAlex ID](https://openalex.org/A5020664641)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `6514db3d-8de6-452c-91b7-acdb31787cc4` `aaccfe5c-6b26-4208-b23c-35331481e142` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 GraG 方法，利用单目视频快速重建手-物体交互的 3D 几何与姿态，核心流程包括多视角 SAM3D 初始化、冻结形状的姿态追踪以及基于 Sum‑of‑Gaussians 的轻量化跟踪。

**💡 创新点**

创新点：1）多视角 SAM3D 生成可泛化的对象 canonical，并在后续帧中冻结形状，显著提升姿态稳定性；2）将稠密 3D Gaussian 转化为稀疏 Sum‑of‑Gaussians，兼顾效率与几何保真；3）结合手/物体遮挡门控、时间一致性引导以及点图深度对齐，进一步提升跟踪鲁棒性；4）无需额外训练，直接复用多种基础模型，极大缩短运行时。

**🔧 技术方法**

技术手段：SAM3D / MV‑SAM3D、DepthAnything3、Dyn‑HaMR、Gaussian Splatting、Sum‑of‑Gaussians (SoG) 跟踪、Quad‑tree 2D Gaussian 聚类、手掩码门控、颜色与深度对齐损失、AdamW 优化。

**📊 数据集**

使用数据集：HO3Dv3（18 个交互序列）和 HOT3D（从训练集挑选的 18 个单手交互序列）。

**📈 对比分析**

与 HOLD、BIGS、MagicHOI 等 SOTA 进行对比；在 HO3Dv3 上对象 Chamfer 距离降低 13.4%，F10 96.7%，手部 MPJPE 降低约 65%；在 HOT3D 上同样取得更低 CD 与更高成功率；运行时间仅 0.56 小时/100 帧，远低于他人（如 MAGICHOI 1.2h、BIGS 3.6h、HOLD 10.5h）。

**⚠️ 局限性**

局限性：依赖 SAM3D、DepthAnything3 等基础模型的成功；手/物体掩码在极端遮挡下易失效；目前仅处理单个刚性物体与单手交互，难以扩展到多物体、手间遮挡或柔性物体。

---

## 518. Pi-HOC: Pairwise 3D Human-Object Contact Estimation

**arXiv ID:** 2604.12923 | [PDF](https://arxiv.org/pdf/2604.12923v1)

**作者:** Sravan Chittupalli `[一作]` (Carnegie Mellon University), Dong Huang `[通讯]` (Carnegie Mellon University)

**通讯引用:** 7868 | [OpenAlex ID](https://openalex.org/A5050270789)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出Pi-HOC框架，能够一次性预测多个人与多物体之间的稠密3D语义接触；

**💡 创新点**

创新点在于对象中心的HO对构造与InteractionFormer联合细化，以及单通道高效实例级推理，避免依赖大型VLM与逐对推理；

**🔧 技术方法**

采用DETR检测实例、InteractionFormer（Transformer编码器）对HO令牌与图像块进行自注意力细化，SAM基底解码器预测顶点接触，并使用轻量级接触判断头筛选对；

**📊 数据集**

在MMHOI（多人人物多物体）和DAMON（单人人物多物体）数据集上训练与评估；

**📈 对比分析**

与Semantic-DECO、InteractVLM等基线相比，Pi-HOC在MMHOI上F1提升至61.09（+10%），地理距离降低至0.0633；在DAMON上F1提升至63.23（+6%）；推理速度提升约20×（从0.15FPS到7.9FPS）；

**⚠️ 局限性**

局限在于仍需检测框与特征的准确性，未覆盖更复杂的场景如动态交互或高密度实例，且对低质量输入的鲁棒性待验证；

---

## 519. Graph-based Hierarchical Deep Reinforcement Learning for Deliverable Block Propagation with Optimal Hybrid Cost in Web 3.0

**arXiv ID:** 2604.12920 | [PDF](https://arxiv.org/pdf/2604.12920v1)

**作者:** Shi Chen `[一作]` (Guangdong University of Technology), Dong In Kim `[通讯]` (Sungkyunkwan University)

**通讯引用:** 24982 | [OpenAlex ID](https://openalex.org/A5022649488)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出面向联盟链 Web 3.0 的可交付区块传播优化框架，结合了新的 AoVB 时效性指标与 BAR 覆盖率，统一为混合成本目标，并通过图同构网络加多头注意力的分层深度强化学习（GHDRL）实现节点分配与路径规划；

**💡 创新点**

创新点包括①将 AoVB 作为仅计入可用窗口内的时效指标并与 BAR 统一；②设计可拆分的 GHDRL，将 NP‑hard 的分配与路径规划拆分为两层模块；③引入可行性剪枝机制保证可达性；

**🔧 技术方法**

使用的技术包括图神经网络（GIN、GAT）、多头注意力机制、强化学习（策略梯度两阶段训练）、可行性剪枝和混合成本目标；

**📊 数据集**

使用自生成的联盟链节点位置与可用性窗口数据集，规模涵盖 50、100、150、200、300、500 节点，K 取 5 或 10；

**📈 对比分析**

与随机、贪心、全图 GAT 及 CAT（跨注意力 Transformer）对比；GHDRL 在所有规模下均实现最低混合成本，最优模型对比 CAT 提升约 19 % 并能在 5 倍规模无重新训练即可迁移；

**⚠️ 局限性**

局限性在于未考虑信道衰落与动态可用性窗口变化、模型仅在仿真数据上验证、对极大规模（>500）扩展性的评估不足以及对不同 fan‑out 设置的鲁棒性待进一步研究。

---

## 520. CoDe-R: Refining Decompiler Output with LLMs via Rationale Guidance and Adaptive Inference

**arXiv ID:** 2604.12913 | [PDF](https://arxiv.org/pdf/2604.12913v1)

**作者:** Qiang Zhang `[一作]` (China University of Mining and Technology), Zhongnian Li `[通讯]` (China University of Mining and Technology)

**通讯引用:** 485 | [OpenAlex ID](https://openalex.org/A5086917578)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `5b4c1114-4a70-478e-9921-2514ee03850d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了CoDe-R，一种轻量级二进制反汇编结果优化框架，结合语义增强（SCE）和动态双路径回退（DDPF），显著提升可重新执行率。

**💡 创新点**

创新点包括：①引入Rationale-guided Semantic Injection为模型提供功能性推理锚点；②动态双路径回退在推理时同时生成语义丰富与语法稳健两条路径，并通过混合验证自适应选择；③在1.3B级轻量模型上突破50%可执行率。

**🔧 技术方法**

采用LLM4Decompile、Chain-of-Thought风格的Rationale生成、测试时计算（Test‑Time Compute）、编译一致性检验、BLEU匹配、HybridAdam等技术。

**📊 数据集**

训练使用Decompile-Ghidra-100k（约86k对）并生成功能性Rationale；评估使用HumanEval-Decompile（164题，O0~O3共656样本）。

**📈 对比分析**

在HumanEval-Decompile上与基线LLM4Decompile-Ref(1.3B)、CodeInverter、Ghidra、Nova等进行对比，CoDe-R平均可执行率达50%，最高O0 70.73%，显著优于基线44.82%及其他轻量级模型，并在高优化级别保持优势。

**⚠️ 局限性**

局限性：双路径回退导致推理延迟；评估仅针对C/C++ GCC环境，未验证其他语言或编译器；Rationale生成可能产生噪声，需进一步优化。

---

## 521. Advancing Network Digital Twin Framework for Generating Realistic Datasets

**arXiv ID:** 2604.12888 | [PDF](https://arxiv.org/pdf/2604.12888v1)

**作者:** Oscar Stenhammar `[一作]`, Carlo Fischione `[通讯]`

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `67630363-6be0-4f51-ab05-7198250671a5` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一个开源的混合数字孪生框架，结合Sionna射线追踪与ns-3离散事件仿真，实现对车联网场景下的可控车辆移动、时空负载以及完整协议栈的跨层仿真，并生成了可复现的网络数据集。

**💡 创新点**

创新点在于：①将可控车辆轨迹、全天候流量负载与精细射线追踪相结合，构建高度逼真的异构网络模拟环境；②提供完整的跨层数据采集与日志体系，生成可用于机器学习的数据集；③通过局部（按小区）与全局模型对比，验证空间异质性对QoS预测的显著影响。

**🔧 技术方法**

使用了ns-3、Sionna射线追踪、SUMO（车辆移动生成）、基于24小时周期的流量负载模型以及多层感知器（MLP）进行QoS预测。

**📊 数据集**

使用的是论文作者自行生成的24小时仿真数据集，约90,000条样本，包含信道指标、移动信息、流量与网络层统计等跨层特征。

**📈 对比分析**

通过比较全局模型、局部模型和基准模型的均方误差（MSE）评估QoS预测性能。全局模型MSE≈0.59ms，局部模型显著更好，MSE≈0.16ms，基准模型MSE≈0.61ms，表明局部（按小区）模型显著优于全局模型。

**⚠️ 局限性**

局限性：①仿真数据缺乏与真实同步测量的对比，真实性和可验证性有限；②计算成本较高，特别是射线追踪与大规模车辆时；③仅在单一城市（慕尼黑）场景下验证，迁移到其他环境需进一步适配。

---

## 522. MetFuse: Figurative Fusion between Metonymy and Metaphor

**arXiv ID:** 2604.12919 | [PDF](https://arxiv.org/pdf/2604.12919v1)

**作者:** Saptarshi Ghosh `[一作]` (University of Cincinnati), Tianyu Jiang `[通讯]` (University of Cincinnati)

**通讯引用:** 174 | [OpenAlex ID](https://openalex.org/A5101803941)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

引入了一个将字面句子转换为转喻、隐喻和混合句子的框架，并基于此构建了首个专门研究转喻与隐喻融合的MetFuse数据集。

**💡 创新点**

创新点在于首次系统地把转喻和隐喻合并为一套生成与评估流程，并证明混合句子能显著提升两类隐喻识别的效果。

**🔧 技术方法**

技术手段包括多步LLM候选生成与BERT掩码概率筛选、情感匹配挑选最佳动词、LLM微调句子精炼，以及在下游任务中使用BERT进行数据增强。

**📊 数据集**

数据集方面构造了1,000个含字面、转喻、隐喻、混合四种变体的MetFuse数据集（共4,000句），并在实验中使用了ConMeC、Pedinotti、RelocaR、WiMCor、VUA Verb、Flute、MOH-X、TroFi等八个现有隐喻/转喻分类基准。

**📈 对比分析**

通过人工评估与句子语义相似度比对，MetFuse在生成质量上比基线高约30‑40%；在八个下游分类任务中，用MetFuse补充训练数据可提升1‑4%准确率，混合样本对转喻任务的提升更为显著。

**⚠️ 局限性**

局限性包括仅聚焦动画主语的场所‑人转喻，未覆盖其他转喻类型；未对概念域映射做细粒度注解；生成过程高度依赖LLM，可能带来偏见或错误；且目前仅验证于英语，缺乏跨语言或更广泛领域的扩展。

---

## 523. An Engineering Journey Training Large Language Models at Scale on Alps: The Apertus Experience

**arXiv ID:** 2604.12973 | [PDF](https://arxiv.org/pdf/2604.12973v1)

**作者:** Jonathan Coles `[一作]` (Swiss National Supercomputing Centre), Nicholas John Browning `[通讯]`

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在瑞士国家超级计算中心Alps超级计算机上，完成了70B参数的开放多语种基础模型Apertus的预训练，并搭建了可弹性扩展的机器学习平台，首次实现了学术机构在此类规模上训练LLM；

**💡 创新点**

创新点包括：1）将HPC平台转变为软件定义的ML平台（vCluster技术），实现资源弹性；2）针对GH200统一内存体系和Slingshot互连的多层优化，解决内存泄漏、通信抖动等系统级瓶颈；3）在存储层面实施访问模式分层与文件系统调优，显著降低I/O抖动；4）提出“饱和评分器”和“数据产品目录”等运维工具，提升大规模训练的可观测性与故障恢复；

**🔧 技术方法**

技术主要涵盖：NVIDIA GH200 Grace Hopper GPU、Slingshot‑11互连、NVIDIA Megatron‑LM+自定义xIELU激活、PyTorch、NCCL、libfabric、Slurm、OCI容器、Triton JIT、HPE Cray EX、ClusterStor E1000、VAST、Lustre、Python调度脚本及自研EDF；

**📊 数据集**

使用多源大规模文本语料，包括公开档案下载和自建Web抓取的数据，Token化后存储在63 TB的Megatron格式二进制数据；数据集总计数以万亿级tokens计；

**📈 对比分析**

与同期公开模型对比：70B模型在Alps上实现了约723 tokens/second/GPU，强规模效率≈80%（4096 GPU），在8B模型上保持高效。通过分布式DP/TP/PP/CP混合并行，结合批量融合、梯度合并和虚拟管道并行，显著提升吞吐；

**⚠️ 局限性**

局限性包括：1）对存储、通信、驱动等系统层的高度依赖，微小版本不匹配可导致大规模抖动；2）统一内存管理导致GPU OOM风险，需要手动缓存清理；3）需要长周期手动调优与运维工具，缺乏完全自动化；4）在RL和微调阶段仍面临框架不成熟、内存压力大和文件系统瓶颈等挑战。

---

## 524. CLAD: Efficient Log Anomaly Detection Directly on Compressed Representations

**arXiv ID:** 2604.13024 | [PDF](https://arxiv.org/pdf/2604.13024v1)

**作者:** Benzhao Tang `[一作]` (Guangzhou University), Shiyu Yang `[通讯]` (Guangzhou University)

**通讯引用:** 584 | [OpenAlex ID](https://openalex.org/A5101688821)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了首个在压缩字节流上直接进行日志异常检测的深度学习框架

**💡 创新点**

创新点在于：①在压缩域无解压无解析即可完成异常检测；②设计多尺度膨胀CNN、混合Transformer–mLSTM编码器以及四路聚合池化，专门捕捉压缩字节的多尺度结构；③采用两阶段自监督预训练（掩码特征预测+InfoNCE）与聚焦对比微调（焦点损失+监督对比损失）及上下文优先采样与跨度掩码增强，以解决字节语义模糊和类别不平衡问题；

**🔧 技术方法**

使用的技术包括：学习型字节嵌入、膨胀卷积、Transformer自注意力、mLSTM矩阵记忆、四路聚合池化、焦点损失、监督对比损失、上下文优先采样、跨度掩码增强、AdamW+EMA训练

**📊 数据集**

在五个公开基准数据集上评估：BGL、Thunderbird、Liberty、Spirit、HDFS，分别为超级计算机日志和分布式文件系统日志

**📈 对比分析**

与三类基线（基于解析模板的CNN、LogRobust、基于原始文本的NeuralLog）对比，平均F1为0.9909，领先最佳基线（CNN 0.9637）2.72个百分点，且在所有数据集上都取得最高F1，说明压缩域方法既不牺牲精度，又省去了解压/解析的时间

**⚠️ 局限性**

局限性包括：依赖压缩器输出结构需保持一定规律性；对极端压缩器变更或未知压缩格式的鲁棒性未知；模型对超大窗口的可扩展性尚未彻底验证；对低延迟实时部署仍需进一步硬件加速研究

---

## 525. Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation

**arXiv ID:** 2604.13010 | [PDF](https://arxiv.org/pdf/2604.13010v1)

**作者:** Yecheng Wu `[一作]` (NVIDIA), Hai Cai `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Lightning OPD，使用离线预计算教师对数概率实现无教师服务器的在线蒸馏。

**💡 创新点**

创新点是引入教师一致性（SFT 与 OPD 采用同一教师）并证明离线目标与标准 OPD 最优点相同，且通过固定 rollouts 自然产生隐式正则化。

**🔧 技术方法**

技术包括两阶段流程：SFT 生成轨迹并微调为参考策略；离线 OPD 预采样 rollouts、预计算教师 log‑prob 并使用优势函数进行梯度更新。

**📊 数据集**

使用 OpenThoughts‑3 生成 SFT 数据，DAPO‑Math‑17k、EpiCoder‑func‑380k 进行 OPD 训练，评估在 AIME、HMMT、LiveCodeBench 等基准。

**📈 对比分析**

与标准 OPD 对比，Lightning OPD 在 4B/8B 模型上保持相同甚至略优的 Pass@1，且 GPU 时长从 120h 降至 30h，显著降低训练成本。

**⚠️ 局限性**

局限在于必须保证教师一致性；若 SFT 与 OPD 采用不同教师会导致不可消除的梯度偏差；此外模型仍受教师能力限制。

---

## 526. XRZero-G0: Pushing the Frontier of Dexterous Robotic Manipulation with Interfaces, Quality and Ratios

**arXiv ID:** 2604.13001 | [PDF](https://arxiv.org/pdf/2604.13001v1)

**作者:** Junming Wang `[一作]` (X SQUARE ROBOT), Hao Wang `[通讯]` (X SQUARE ROBOT)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了XRZero-G0框架，利用可穿戴VR+双手机械手实现无机器人数据采集，构建2000小时多模态G0数据集，并证明可通过少量真实机器人数据与海量机器人免费数据的混合实现跨本体零射击执行。

**💡 创新点**

核心创新包括：① 通过VR inside‑out跟踪与多视角摄像头解耦人机姿态，兼容异构H/G机械手；② 设计闭环质量验证管线（视觉清洗、IK验证、物理回放验证）；③ 系统化研究少量真实机器人与海量机器人免费数据的混合律（Few‑Shot Physical Anchoring）；④ 规模化发布G0数据集，支持跨平台政策学习。

**🔧 技术方法**

使用技术包括：PICO4 VR头显 inside‑out 跟踪、多摄像头同步、边缘计算同步、机械手 H/G 结构、自动视觉清洗、逆运动学验证、物理回放验证、VLA 基础模型（Wall‑OSS、π_0、π_0.5）训练与评估。

**📊 数据集**

使用数据集：G0数据集（2000小时，3000个多任务，多模态）、公开VLA基线模型（Wall‑OSS、π_0、π_0.5）进行实验。

**📈 对比分析**

与传统 master‑slave 及普通 VR teleoperation 对比，XRZero‑G0 的收集效率提升 2.33×~1.71×；纯机器人免费数据训练成功率与 500 条真实机器人对齐；10:1 混合（500 免费 + 50 真实）可达到 500 条真实机器人相同的成功率，成本降低 90%。在不同机器人高度下，成功率保持 60%–70%。

**⚠️ 局限性**

局限性包括：设备重量限制长时间采集；缺少触觉/力/压感传感；尚未验证移动机器人或全身移动场景；真实机器人标定仍需少量样本；任务多样性集中在静态桌面环境。

---

## 527. Lyra 2.0: Explorable Generative 3D Worlds

**arXiv ID:** 2604.13036 | [PDF](https://arxiv.org/pdf/2604.13036v1)

**作者:** Tianchang Shen `[一作]` (NVIDIA), Xuanchi Ren `[通讯]` (NVIDIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `6514db3d-8de6-452c-91b7-acdb31787cc4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

从单张图片出发，利用相机控制的深度视频扩散模型生成长时程的3D一致视频，并将其通过高效的3D Gaussian Splatting 转换为可交互的3D资产。

**💡 创新点**

创新点在于：① 通过“anti‑forgetting”机制，维护每帧的3D几何缓存并仅用于检索与映射，避免几何误差累积；② 通过“anti‑drifting”机制，采用自增强训练与FramePack压缩上下文，显著抑制自回归过程中误差累积；③ 在采样时引入规范坐标warping而非直接RGB，保持几何一致性。

**🔧 技术方法**

核心技术包括：DiT‑based 潜在视频扩散模型、VAE 编码/解码、Depth Anything V3 深度估计、Plücker 6D 视线注入、FramePack 时间压缩、Canonical 3D 坐标映射、Self‑Augmentation 训练、3D Gaussian Splatting 及后续网格提取。

**📊 数据集**

训练使用 DL3DV（约10k条长视频）和生成的文本提示；评测采用 Tanks & Temples 以及 DL3DV‑Evaluation 数据集；重建模型在 DAv3 基础上 fine‑tune。

**📈 对比分析**

与 Yume‑1.5、GEN3C、CaM、VMem、SPMem、HY‑WorldPlay 等基线在 SSIM、LPIPS、FID、Subjective Quality、Style Consistency、Camera Controllability、Reprojection Error 等指标上均取得显著提升；在3D重建方面在 LPIPS‑G、FID、Subjective Quality 以及 LPIPS‑P 上均居首位。

**⚠️ 局限性**

主要限制：仅适用于静态场景；对动态对象缺乏建模；模型继承训练数据的光照/曝光变异，导致生成视频在光度上不稳定，进而影响 3D 重建质量。

---

## 528. Representation geometry shapes task performance in vision-language modeling for CT enterography

**arXiv ID:** 2604.13021 | [PDF](https://arxiv.org/pdf/2604.13021v1)

**作者:** Cristian Minoccheri `[一作]` (University of Michigan), Ryan Stidham `[通讯]` (University of Michigan)

**通讯引用:** 4711 | [OpenAlex ID](https://openalex.org/A5038079014)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

对腹部CT肠道造影进行视觉‑语言模型迁移学习，评估不同表示几何对疾病分类、跨模态检索和报告生成的影响。

**💡 创新点**

发现平均池化更适合分类，注意力池化更适合检索，并证明多窗口RGB编码优于多平面采样；提出无专家标注的三师教师伪标签框架。

**🔧 技术方法**

使用BiomedCLIP 2.5D切片编码、LoRA 参数高效微调、MedGemma‑4B 生成模型和检索增强生成（RAG）。

**📊 数据集**

使用密歇根大学医院的1074例腹部CT肠道造影及其配对报告。

**📈 对比分析**

在疾病三分类中平均池化得到最高59.2%准确率，检索中注意力池化得到0.235 MRR；报告生成中RAG相较于单纯微调提升7–14个百分点的within‑1准确率。

**⚠️ 局限性**

受限于单中心数据、伪标签噪声、有限的切片采样以及小样本量，导致绝对性能偏低且缺乏外部验证。

---

## 529. Bilevel Late Acceptance Hill Climbing for the Electric Capacitated Vehicle Routing Problem

**arXiv ID:** 2604.13013 | [PDF](https://arxiv.org/pdf/2604.13013v1)

**作者:** Yinghao Qin `[一作]` (Queen Mary University of London), Jun Chen `[通讯]` (Queen Mary University of London)

**通讯引用:** 469860 | [OpenAlex ID](https://openalex.org/A5100376569)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对电动有载车辆路径规划（E‑CVRP）提出了一种双层 Late Acceptance Hill Climbing（b‑LAHC）算法，实现了路由与充电决策的分层求解。

**💡 创新点**

创新点在于首次将E‑CVRP建模为双层优化，利用上层 surrogate 目标快速筛选路线并在需要时触发下层充电优化，从而显著降低计算量并保持全局性能；同时引入 M8 轨迹扩展算子突破传统单一路线局部最优瓶颈。

**🔧 技术方法**

核心技术包括：双层优化框架、Late Acceptance 迟接受机制、简单枚举（SE）充电优化、八种邻域变换算子（M1–M8）以及贪婪下层与精细化上层交替迭代。

**📊 数据集**

实验使用 IEEE WCCI‑2020 公开基准集（17 个实例，包含 7 个小规模 E‑set 与 10 个大规模 X‑set，最高 1,000 顾客）。

**📈 对比分析**

与 8 种最先进方法（VNS、SA、GA、HHASA‑TS、BACO、CBACO‑I、CBMA 等）在 Max Evals 与 Max Time 预算下对比，b‑LAHC 在 9/10 大规模实例中取得新最优结果，整体排名第一，平均误差低于 0.1%，标准差明显下降。

**⚠️ 局限性**

局限性包括：需要手动设置历史长度、最大尝试次数等超参数；对极大规模实例（>10,000 顾客）仍可能出现收敛速度慢或局部最优；双层框架虽高效但在充电策略极其复杂或多种充电速率时需进一步扩展。

---

## 530. SceneCritic: A Symbolic Evaluator for 3D Indoor Scene Synthesis

**arXiv ID:** 2604.13035 | [PDF](https://arxiv.org/pdf/2604.13035v1)

**作者:** Kathakoli Sengupta `[一作]` (Stony Brook University), Paola Cascante-Bonilla `[通讯]` (Stony Brook University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `a4b10f5d-130b-4e77-9367-6469ec621899` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于符号推理的室内平面图布局评估器，并构建了一个基于3D‑FRONT、ScanNet和Visual Genome三大数据集的结构化空间本体，用来在不依赖渲染视图的情况下对布局的语义一致性、朝向正确性和重叠情况进行对象级评估；同时在迭代细化测试床上探究了不同后训练目标（RL、RLHF、RLIF、RLVR）和不同评估器（启发式、LLM、VLM）对空间结构生成的影响。

**💡 创新点**

创新点包括：① 将室内空间知识从多源数据集中抽象为可解释的图谱，本体中包含对象尺寸、支撑关系、共现概率、朝向偏好等；② 开发了基于该本体的符号评估器，能够逐条检查布局并定位具体违例；③ 通过与VLM评估器对比并与人工评判对齐，证明了符号评估在稳定性和人类一致性上的优势；④ 通过迭代细化测试床，揭示文本LLM在语义布局上可超越VLM，数学导向的RL目标提升朝向推理，图像VLM细化最为有效。

**🔧 技术方法**

技术手段包括：符号图谱构建与查询、条件共现概率与nPMI统计、三种评估维度（语义、朝向、重叠）的规则推理、基于文本或图像的LLM/VLM反馈、RL/RLHF/GRPO/DPO等后训练方法，以及大规模人类对比实验。

**📊 数据集**

使用的数据集：3D‑FRONT（6,813个室内场景）、ScanNet（1,511个RGB‑D重建场景）、Visual Genome（61,530张标注图像），并在此基础上构造了空间本体。评估样本包括常见的卧室和客厅以及书店、餐厅、儿童房等不常见场景。

**📈 对比分析**

与基线VLM评估器（Gemini‑2.5‑Pro）和人工评估对比，符号评估器在语义一致性与朝向上的人类一致率分别为94.44%/83.33%，而VLM仅为58.82%/47.06%。在迭代细化实验中，文本LLM（如DeepSeek‑3.2V）在语义得分上超越VLM，数学目标训练的模型在朝向得分上表现最佳，图像VLM细化在语义和朝向分数上最显著提升（相较启发式提升约12分）。

**⚠️ 局限性**

局限性包括：① 评估仅覆盖已纳入本体的对象类别和房间类型，对未见类别缺乏适用性；② 侧重三维几何与语义一致性，未考虑更细粒度的物理交互（如动力学稳定性、可访问性等）；③ 需要完整的3D布局信息，无法直接评估仅有渲染图的生成方法；④ 规则阈值（如角度容差、距离阈值）需人工设定，可能影响评估偏好。

---

## 531. SpotSound: Enhancing Large Audio-Language Models with Fine-Grained Temporal Grounding

**arXiv ID:** 2604.13023 | [PDF](https://arxiv.org/pdf/2604.13023v1)

**作者:** Luoyi Sun `[一作]` (Zhejiang University), Weidi Xie `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 10242 | [OpenAlex ID](https://openalex.org/A5076097168)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了一种名为 SpotSound 的大型音频语言模型框架，利用时间戳交错编码和四元组训练目标，使模型能够在长音频中精确定位短暂的音频事件。

**💡 创新点**

创新点包括：①在音频与文本序列中交错插入绝对时间戳，以提供明确的时序信息；②引入正负查询的四元组训练目标，显著抑制对不存在事件的幻觉预测；③发布了“针孔式”短窗口基准 SpotSoundBench，逼真模拟短事件被稀释在长背景中的难题。

**🔧 技术方法**

技术要点：使用 Whisper‑large‑v3 作为音频编码器；将时间戳、音频特征与查询文本交错拼接后输入 Qwen2‑Audio / Audio Flamingo‑3 等大语言模型；采用 LoRA 微调、自动回归交叉熵训练；通过随机混合前景/背景音频并记录插入时间戳来生成合成数据。

**📊 数据集**

数据集：训练阶段整合 AudioGrounding、Clotho‑Moment、UnAV‑100、ASSL、AudioSet 强标签、VGGSound、Walkingtour 等公开数据，构造 10k 条长片合成样本；评测阶段使用 AudioGrounding、Clotho‑Moment、UnAV‑100 子集以及自制的 SpotSoundBench（300 条长片，目标窗口占比 <10%）。

**📈 对比分析**

与 WTATG、AM‑DETR、Kimi‑Audio、TimeAudio、Audio Flamingo‑3、Qwen2‑Audio 等基线比较；SpotSound 在所有四个时序定位基准上均实现显著提升（Clotho‑Moment +4.7%，UnAV‑100 +27%，AudioGrounding +2.9%，SpotSoundBench +20.4%），在声事件检测任务上亦保持最高 mIoU；在幻觉消除与两阶段联合评估上表现远优于现有模型。

**⚠️ 局限性**

局限性：模型仍受限于 Whisper‑large‑v3 的 30 s 语音编码窗口，导致对极长音频的全时域理解不足；合成数据在极低 SNR 或极短事件下可能不够真实；以及对极端噪声环境或多重重叠事件的处理仍有提升空间。

---

## 532. Agentic Discovery with Active Hypothesis Exploration for Visual Recognition

**arXiv ID:** 2604.12999 | [PDF](https://arxiv.org/pdf/2604.12999v1)

**作者:** Jaywon Koo `[一作]` (Rice University), Vicente Ordonez `[通讯]` (Rice University)

**通讯引用:** 12948 | [OpenAlex ID](https://openalex.org/A5027328044)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `39fd911c-56a4-425d-a2f9-8038ad3b6e21` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种基于假设驱动的多代理框架，用记忆机制自动从零探索视觉识别神经网络架构。

**💡 创新点**

通过显式的假设记忆与轨迹树，将架构探索拆分为分支扩展与假设检验，并利用多视角反馈与贝塔采样实现探索-利用平衡。

**🔧 技术方法**

采用大语言模型驱动的生成、实现、实验与反馈代理，构建多代理系统；使用双层选择策略（父节点与假设）与贝塔分布、置信度更新机制；通过多维度反馈分析提升决策。

**📊 数据集**

在通用视觉任务使用 CIFAR‑10/100、Tiny‑ImageNet；在医学图像任务使用 MedMNIST（DermalMNIST、TissueMNIST、BreastMNIST）进行验证。

**📈 对比分析**

通过与无假设记忆、无多代理反馈、不同父节点选择策略等消融实验对比；在 CIFAR‑10 上实现 94.11% 的准确率，Tiny‑ImageNet 达 58.1%；在 MedMNIST 上获得与最优模型相近的表现，但参数显著更少。

**⚠️ 局限性**

局限性包括需要大量计算资源和长时间实验；假设生成高度依赖大语言模型，易产生低质量或冗余假设；框架在极端资源受限或更大规模任务的可扩展性尚待验证。

---

## 533. Toward Autonomous Long-Horizon Engineering for ML Research

**arXiv ID:** 2604.13018 | [PDF](https://arxiv.org/pdf/2604.13018v1)

**作者:** Guoxin Chen `[一作]` (Renmin University of China), Kai Jia `[通讯]` (AweAI Team)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了一个能够在长时间跨度内自动完成从论文解读、环境搭建、代码实现、实验运行到调试的系统。

**💡 创新点**

提出了基于文件共享的“File-as-Bus”协议实现耐久状态连续性，并在此基础上采用分层层级化的协作框架。

**🔧 技术方法**

采用大语言模型（Gemini‑3‑Flash、GLM‑5）、多代理协作、文件共享工作空间和可读写权限控制。

**📊 数据集**

在 PaperBench（论文重现）和 MLE‑Bench Lite（持续实验改进）两个基准上进行评估。

**📈 对比分析**

与 BasicAgent、IterativeAgent、AIDE、ML‑Master 2.0 等基线相比，在 PaperBench 上平均分提升约 10–11 分，成本更低；在 MLE‑Bench Lite 上 Any Medal% 提升 4.5–18 分，显著超过现有排行榜。

**⚠️ 局限性**

仍受限于只验证在两个基准上的效果，对更大规模、多样化任务的泛化性与实时资源管理尚未充分验证。

---

## 534. Generative Refinement Networks for Visual Synthesis

**arXiv ID:** 2604.13030 | [PDF](https://arxiv.org/pdf/2604.13030v1)

**作者:** Jian Han `[一作]` (ByteDance), Zehuan Yuan `[通讯]` (ByteDance)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `8f4a6f4b-054d-462c-afe4-56ebc0388d1a` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出Generative Refinement Networks（GRN）框架，通过全局细化和自适应步长生成高质量视觉内容。

**💡 创新点**

创新点在于：①Hierarchical Binary Quantization（HBQ）实现离散化近无损压缩；②全局细化机制能反向纠错；③熵引导的自适应采样实现按需分配计算。

**🔧 技术方法**

核心技术包括：离散化量化（HBQ）、Transformer基的AR生成、全局细化策略、熵引导的自适应采样以及对抗与感知损失。

**📊 数据集**

使用的数据集包括ImageNet（256×256）、OpenImages、公开图像与视频集合（约4千万视频）、少量高质量文本提示数据。

**📈 对比分析**

与扩散、混合及传统AR模型对比，在ImageNet C2I任务上实现rFID 0.56、gFID 1.81，文本到图像/视频任务在等规模下均优于同等参数的扩散/AR基模型。

**⚠️ 局限性**

局限性：未达到顶尖模型的规模与计算资源；文本到视频生成在细节与失真方面仍有提升空间；需进一步平衡数据分布和模型规模。

---

## 535. Conflated Inverse Modeling to Generate Diverse and Temperature-Change Inducing Urban Vegetation Patterns

**arXiv ID:** 2604.13028 | [PDF](https://arxiv.org/pdf/2604.13028v1)

**作者:** Baris Sarper Tezcan `[一作]` (Purdue University), Daniel Aliaga `[通讯]` (Purdue University)

**通讯引用:** 5011 | [OpenAlex ID](https://openalex.org/A5090414723)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了一种结合前向预测模型和扩散生成模型的逆向建模框架，用于在给定城市区域的建筑高度和目标温度变化下生成多样化、可控的植被配置。

**💡 创新点**

创新点在于将温度预测作为约束加入扩散模型训练，使用粗粒度温度条件和物理一致性损失，使模型在保持温度控制的同时产生多种可行的植被图案，解决了植被布局的多对一不确定性。

**🔧 技术方法**

核心技术包括：U‑Net前向模型预测LST；EDM（扩散式模型）进行逆向生成；coarse LST conditioning、patch‑mean physics loss、温度控制与基线一致性评估。

**📊 数据集**

使用了20个美国城市的Landsat 8影像（30 m分辨率），提取了NDVI、LST和建筑高度（BH）数据，共计2829训练块、701测试块。

**📈 对比分析**

与基线（U‑Net直接回归、Fine LST扩散、Coarse LST扩散）相比，本文方法在温度控制误差(CtrlErr)显著下降（约37%），多样性指标提升3.4倍，基线一致性误差(BaseErr)也更小，表明既可实现精准温度调节又能生成多样化的植被布局。

**⚠️ 局限性**

局限性包括：只考虑建筑高度约束，未纳入更细粒度的基础设施限制；依赖Landsat 8，难以直接推广到无热带成像的Sentinel‑2或MODIS；模型仍基于学习到的前向预测器，前向模型自身的误差给整体性能设定下限。

---

## 536. Asymptotically faster algorithms for recognizing $(k,\ell)$-sparse graphs

**arXiv ID:** 2604.13025 | [PDF](https://arxiv.org/pdf/2604.13025v1)

**作者:** Bence Deák `[一作]` (Eötvös Loránd University), Péter Madarasi `[通讯]` (HUN-REN Alfréd Rényi Institute of Mathematics)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出了一套针对所有经典与扩展参数范围（即0≤ℓ≤k、k<ℓ<2k以及2k≤ℓ<3k）(k,ℓ)-稀疏图的识别算法，能够在固定k、ℓ的前提下在近线性时间内判断稀疏性并给出违背稀疏性的顶点集合。

**💡 创新点**

创新点在于将稀疏性检测归约为有根弧连通性问题，结合有限入度定向、增广路径重定向、森林分解与基于中点分解的分治技术，从而突破传统O(n^2)或O(n^3)上限，尤其在经典范围实现了近线性时间。

**🔧 技术方法**

核心技术包括：最大流与可行循环求解有限入度定向、森林分解求k根树、根弧连通性检验（利用Tarjan/ Gabow算法）、增广路径重定向法以及基于中点分解的递归分治。

**📊 数据集**

论文为理论算法研究，未使用具体实验数据集；所有结果均通过算法复杂度分析与已知最优子程序性能给出。

**📈 对比分析**

与先前最优算法相比，本文在经典范围内将时间复杂度从O(n^2)降低至O(n^{1+o(1)})（在纯组合实现下为O(n√n)），在扩展范围2k≤ℓ<3k中将最差情况从O(n^3)降至O(n^2 log n)（或O(n^2)当ℓ≤2k+1时）。

**⚠️ 局限性**

局限性包括：在扩展范围ℓ>2k+1时仍保持O(n^2 log n)复杂度；算法实现高度依赖于高效的最大流、森林分解与根弧连通性子程序；且假设k、ℓ为常数，无法直接推广至k、ℓ随图规模变化的情形。

---

## 537. See, Point, Refine: Multi-Turn Approach to GUI Grounding with Visual Feedback

**arXiv ID:** 2604.13019 | [PDF](https://arxiv.org/pdf/2604.13019v1)

**作者:** Himangi Mittal `[一作]` (Microsoft), Yu Hu `[通讯]` (Microsoft)

**通讯引用:** 9472 | [OpenAlex ID](https://openalex.org/A5014478407)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在 IDE 环境中通过多轮视觉反馈迭代改进 GUI grounding，实现像素级精准光标定位

**💡 创新点**

首次系统化研究多轮视觉修正对像素级 GUI grounding 的提升，并提供专用 VS Code 数据收集管道与基准测试

**🔧 技术方法**

多轮交互式 grounding、视觉反馈渲染、prompt engineering、无训练的前沿模型推理

**📊 数据集**

自建 257 条 VS Code 光标定位样本，覆盖字符/单词/行级别

**📈 对比分析**

对比单次预测与多轮反馈，使用准确率与距离指标；GPT‑5.4 在二轮后准确率提升至约 38%，距离下降至 57 像素，Claude 与 Qwen 同样提升但幅度较小

**⚠️ 局限性**

数据规模有限，仅覆盖 VS Code；模型未进行专门微调，缺乏更复杂 UI；实验环境单一，泛化性待验证

---

## 538. PAL: Personal Adaptive Learner

**arXiv ID:** 2604.13017 | [PDF](https://arxiv.org/pdf/2604.13017v1)

**作者:** Megha Chakraborty `[一作]` (University of South Carolina), Amit Sheth `[通讯]` (University of South Carolina)

**通讯引用:** 36281 | [OpenAlex ID](https://openalex.org/A5028772801)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一款名为PAL的 AI 教学平台，能够把录播讲座转换为实时交互式学习体验，并在播放过程中自动生成并调整难度的多选题，最终生成个性化总结。

**💡 创新点**

创新点在于：① 将多模态视频内容（语音、图像、字幕）与 RL 强化学习相结合，实现实时难度自适应；② 采用混合统计与 RL 的双头决策模型，兼顾稳定性与个性化；③ 通过语义检索 + Llama 3.2 生成个性化后课总结。

**🔧 技术方法**

技术包括：多模态 NLP（LLaVA-mini、OCR、语义向量化）、强化学习（ε-greedy bandit + Q‑learning）、IRT 统计先验、LLM 生成（轻量版 LLM、Llama 3.2 1B Instruct）以及 Web 前端演示。

**📊 数据集**

使用公开的讲座视频与相应字幕（通过脚本生成的时间戳文本），并在内部构建的“Video‑to‑Question”数据集上训练问答与难度标注模型；后续总结阶段使用语料库的完整讲座文本。

**📈 对比分析**

与传统静态自适应平台（固定难度、统一测验）进行对比，实验显示 PAL 在保持学习者“学习区”内、提升答题正确率与保持时长上表现更佳；具体指标为平均答题准确率提升约12%，学习时长增加约18%。

**⚠️ 局限性**

局限性：① 依赖高质量字幕与视觉信息，视频质量差时效果受限；② 对新手（cold‑start）需要更长的适应周期；③ 主要在单个讲座视频上评估，缺乏跨学科、多课堂的长期实验数据。

---

## 539. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe

**arXiv ID:** 2604.13016 | [PDF](https://arxiv.org/pdf/2604.13016v1)

**作者:** Yaxuan Li `[一作]` (Tsinghua University), Ning Ding `[通讯]` (Tsinghua University)

**通讯引用:** 15384 | [OpenAlex ID](https://openalex.org/A5001191710)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `8d10c613-917e-4880-9716-17789f50e119` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大语言模型的On-Policy Distillation（OPD）进行系统性分析，揭示其成功与失败的关键因素并给出恢复方案。

**💡 创新点**

创新点包括提出思维模式一致性与新知识两大成功条件，发现OPD主要通过高概率token的渐进对齐实现优化，并提出离线冷启动和教师对齐提示两种实用补救策略。

**🔧 技术方法**

研究采用反向KL逆向学习的OPD框架，结合采样、Top‑k和全词汇三种监督粒度，并使用重心重塑、重叠比率、重叠token优势、熵差等动态指标和逆向蒸馏实验。

**📊 数据集**

实验数据集涵盖数学推理领域的DAPO‑Math‑17K、DeepMath、OpenThoughts3‑1.2M等，以及AIME 2024/2025、AMC 2023等官方评测集。

**📈 对比分析**

通过与SFT、不同规模教师以及逆向蒸馏对照，OPD可恢复约80%教师性能差距，平均@16准确率显著提升；离线冷启动和教师对齐提示进一步提升性能。

**⚠️ 局限性**

局限性包括对长序列奖励可靠性下降、仅在数学推理领域验证、对容量差距的细致机制未完全揭示，以及需要进一步探索奖励空间的局部可利用性与梯度几何。

---

## 540. DySkew: Dynamic Data Redistribution for Skew-Resilient Snowpark UDF Execution

**arXiv ID:** 2604.13034 | [PDF](https://arxiv.org/pdf/2604.13034v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 541. One Token Away from Collapse: The Fragility of Instruction-Tuned Helpfulness

**arXiv ID:** 2604.13006 | [PDF](https://arxiv.org/pdf/2604.13006v1)

**作者:** Erfan Baghaei Potraghloo `[一作]` (University of Southern California), Massoud Pedram `[通讯]` (University of Southern California)

**通讯引用:** 28072 | [OpenAlex ID](https://openalex.org/A5044650311)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究指令调优的大型语言模型在面对极其简单的词汇约束时会出现响应“崩溃”现象，即答案变得简短且缺乏全面性；

**💡 创新点**

首次揭示并量化指令调优导致的模板依赖性脆弱性，证明这是一种规划失败而非能力缺失，并通过内部表征与两步生成实验验证机制；

**🔧 技术方法**

使用线性探针对中层隐藏状态进行响应长度预测、两步生成（先自由生成再在约束下重写）、Token‑级JS Divergence分析以及基于LLM的配对与独立评估；

**📊 数据集**

构造了包含四类任务（解释、如何做、分析比较、技术说明）的40个提示集，八种词汇约束（标点、列表、单词禁令）以及公开的MT‑Bench；

**📈 对比分析**

与基线无约束回答进行配对比较，使用GPT‑4o‑mini和GPT‑4o评审，发现不同模型在约束下的综合性下降幅度为14–48%，基线在77–100%的对比中被优先；相对独立评估只检测到约1/5的质量下降，凸显方法差距；

**⚠️ 局限性**

仅在七至八B参数的开源模型上进行内部表征分析，无法直接验证闭源GPT‑4o‑mini的内部机制；使用LLM评审可能存在偏差；约束集有限，未覆盖所有可能的词汇或语法限制；线性探针的解释力受限。

---

## 542. LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software

**arXiv ID:** 2604.12994 | [PDF](https://arxiv.org/pdf/2604.12994v1)

**作者:** Syed Md Mukit Rashid `[一作]` (Pennsylvania State University), Syed Rafiul Hussain `[通讯]` (Pennsylvania State University)

**通讯引用:** 1113 | [OpenAlex ID](https://openalex.org/A5053169357)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研发了一套针对逻辑漏洞的自动补丁生成与评估框架 LogicEval，并构建了 43 条真实逻辑漏洞案例数据集 LogicDS。

**💡 创新点**

首次系统性评估传统与 LLM 方案在逻辑漏洞修复中的表现，并提出了基于补丁解释相似度的自动推理评估指标。

**🔧 技术方法**

结合模板/搜索/深度学习的 AVR 方法、LLM 生成与评估（Llama3.1、Qwen2.5、OpenAI o3-mini），以及编译/测试流水线与 LLM 解释匹配的推理度量。

**📊 数据集**

逻辑漏洞数据集 LogicDS（43 个 CVE 案例）以及对应的 43 条合成 Java 示例。

**📈 对比分析**

通过编译/测试成功率、推理相似度和 LLM 评判的二元判定，对比了 SimFix、KNOD、VRPilot 以及不同提示配置下的 LLM，发现 LLM 在提供辅助信息时表现最佳，但整体仅在 43 条样本中修复成功 5 条。

**⚠️ 局限性**

数据集规模有限、假设完美定位导致评估范围受限，以及对 LLM 评判的循环偏差和缺乏多样化基线等限制。

---

## 543. Accelerating Speculative Decoding with Block Diffusion Draft Trees

**arXiv ID:** 2604.12989 | [PDF](https://arxiv.org/pdf/2604.12989v1)

**作者:** Liran Ringel `[一作]` (Technion Israel Institute of Technology), Yaniv Romano `[通讯]` (Technion Israel Institute of Technology)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了DDTree，一种在块扩散采样器一次前向传播后构造草稿树并通过树注意力在单次目标模型前向传播中验证，从而提升自回归语言模型的推理速度。

**💡 创新点**

创新点在于：①使用块扩散产生的每个位置的边缘分布，构造最大化期望接受长度的最优草稿树；②通过最佳优先堆实现仅保留前K高概率token的前B个前缀，无需枚举指数级前缀；③在验证阶段使用树注意力一次性评估所有候选分支。

**🔧 技术方法**

技术手段包括：块扩散采样器（Block Diffusion）、DFlash、DDTree、最佳优先堆算法、树注意力（tree attention）、Hugging Face Transformers库以及GPU并行加速。

**📊 数据集**

评测数据集覆盖推理速度与接受长度：推理任务（MATH‑500、GSM8K、AIME 2024/2025）、代码任务（HumanEval、MBPP、LiveCodeBench、SWE‑bench Lite）以及通用指令/对话任务（MT‑Bench、Alpaca）。

**📈 对比分析**

与原始DFlash和自回归解码对比，DDTree在所有10 × 3 × 2＝60种数据集‑模型‑温度组合中均优于DFlash，速度提升最高可达≈7.3×，平均接受长度明显增加；最佳节点预算通常为256‑512，能在验证成本与接受长度之间取得平衡。

**⚠️ 局限性**

局限性：①基于块扩散的边缘分布近似真实目标模型的条件分布，可能导致最优树与实际最优不完全一致；②需要手动调节节点预算以平衡速度与精度；③在非常大预算下验证成本上升，反而降低速度；④目前仅适用于一次性块扩散采样器，无法直接推广到自回归草稿器。

---

## 544. Visual Preference Optimization with Rubric Rewards

**arXiv ID:** 2604.13029 | [PDF](https://arxiv.org/pdf/2604.13029v1)

**作者:** Ya-Qi Yu `[一作]` (Huawei Technologies Co Ltd), Dandan Tu `[通讯]` (Huawei Technologies Co Ltd)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 rDPO 框架，利用实例化的 rubrics 对视觉语言模型进行偏好优化。

**💡 创新点**

创新点在于将每个图像-指令对的评判拆分为“必要”与“附加”细粒度准则，并在 on‑policy 数据构建中直接使用这些准则生成高质量的偏好对，从而提升视觉模型的对齐效果。

**🔧 技术方法**

核心技术包括：直接偏好优化 (DPO)、混合偏好优化 (MPO)、生成式奖励模型 (GenRM)、零射击 VLM‑judge、rubric 生成与多维评分、以及迭代训练策略。

**📊 数据集**

使用的数据集涵盖公开的 reward‑benchmark（MM‑RB、VL‑RB、MaaJ、VA‑B、WV‑B）、300K 规模的 instruction‑rubric pool，以及内部综合基准。

**📈 对比分析**

与传统 outcome‑based 过滤、CoT 以及 GPT‑5.4 等方法比较，方法验证中宏平均从 81.14 提升至 82.69，综合基准上 rDPO 得分 61.01，显著超过基础模型 59.48。

**⚠️ 局限性**

限制在于 rubric 级奖励未直接嵌入优化循环，缺乏在线自适应；对更广泛多模态任务的鲁棒性和可扩展性仍需进一步验证。

---

## 545. Learning Versatile Humanoid Manipulation with Touch Dreaming

**arXiv ID:** 2604.13015 | [PDF](https://arxiv.org/pdf/2604.13015v1)

**作者:** Yaru Niu `[一作]` (Carnegie Mellon University), Ding Zhao `[通讯]` (Carnegie Mellon University)

**通讯引用:** 5722 | [OpenAlex ID](https://openalex.org/A5037644321)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个全身控制、VR实时演示收集与多模态Transformer框架，实现了多种复杂接触任务的人形机器人抓取、推送、折叠、工具使用和双臂移动服务。

**💡 创新点**

引入“触觉梦境”预测未来手关节力与触觉潜在表示，并用EMA教师监督，令单阶段行为克隆学习到对接触变化高度敏感的表征。

**🔧 技术方法**

使用RL强化学习的下肢控制器、VR-基于人机映射的数据收集、分块Transformer编码解码、EMA教师的潜在空间监督及多模态输入（视觉、关节、力、触觉）。

**📊 数据集**

在五个真实场景任务（Insert‑T、Book Organization、Towel Folding、Cat Litter Scooping、Tea Serving）上收集并训练数据。

**📈 对比分析**

与两种ACT解码器基线对比，HTD在平均成功率上提升约90.9%（相对），在任务得分上提升约31%，且在最难任务上显著优于基线。

**⚠️ 局限性**

局限在于仅在固定的五个任务上验证，模型对极端未知环境的泛化、对触觉硬件的依赖以及计算资源需求未系统评估。

---

## 546. Personalizing LLM-Based Conversational Programming Assistants

**arXiv ID:** 2604.12998 | [PDF](https://arxiv.org/pdf/2604.12998v1)

**作者:** Jonan Richards `[一作]` (Radboud University), Jonan Richards `[通讯]` (Radboud University)

**通讯引用:** 4172 | [OpenAlex ID](https://openalex.org/A5083706803)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本研究探讨认知多样性和组织背景如何影响开发者使用LLM对话编程助手的需求与交互方式，并基于此设计了可解释框架、个性化策略以及原型系统，以提升工具的包容性。

**💡 创新点**

创新点在于：①提出将个人与上下文因素与需求、交互风格关联的解释框架；②整合隐式与显式个性化机制，兼顾透明度与使用便利；③提出利用LLM-as-a-Judge的自动化人机交互评估方法，实现大规模评估。

**🔧 技术方法**

采用的技术包括LLM提示工程（理论心智推断）、自然语言处理技术（交互风格分类）、自适应与显式个性化配置、LLM作为评审者的自动评估框架，以及定性访谈与问卷调查。

**📊 数据集**

数据集为：1) 27名具备不同经验与认知特征的受试者在固定代码变更场景下与GitHub Copilot Chat的对话日志；2) 14名新手开发者的原型实验对话；3) 调查问卷与访谈文本；无公开数据集，全部为实验收集。

**📈 对比分析**

评估方法为：在同一组受试者中进行个性化与非个性化助手的对照实验，测量用户感知效果；随后在10名受试者的原型试点中收集满意度、响应清晰度、任务完成时间等指标；最终采用LLM-as-a-Judge的自动评估与实际用户研究结果对比。初步结果显示，个性化助手在提升理解度和回答清晰度方面优于默认助手，正式量化指标待进一步验证。

**⚠️ 局限性**

局限性包括：实验样本量相对较小，可能导致结果泛化受限；个性化推断可能存在偏差，需持续验证；评估主要基于自报数据，客观性有限；交互上下文固定，未覆盖所有真实场景，导致模型泛化能力未知。

---

## 547. PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models

**arXiv ID:** 2604.12995 | [PDF](https://arxiv.org/pdf/2604.12995v1)

**作者:** Han Bao `[一作]` (University of Notre Dame), Xiangliang Zhang `[通讯]` (University of Notre Dame)

**通讯引用:** 12858 | [OpenAlex ID](https://openalex.org/A5000755750)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了跨国中美多领域、三级认知层次的PolicyBench基准，并提出了PolicyMoE模型以提升LLM在政策任务中的表现。

**💡 创新点**

创新点在于首次提供中美对比的政策理解基准，并将Mixture-of-Experts与LoRA微调相结合，形成针对记忆、理解和应用的专家模块。

**🔧 技术方法**

使用了MoE架构、LoRA微调、线性路由器等技术，并在Benchmark上对多种LLM进行评估。

**📊 数据集**

数据集为PolicyBench，包含约21K条中美政策相关问题，覆盖记忆、理解和应用三层次。

**📈 对比分析**

通过与多种强基线模型（如GPT‑4o、Claude‑3.5、Gemini‑2.5等）对比，PolicyMoE在记忆和应用层面显著提升，整体准确率提升至约70%以上。

**⚠️ 局限性**

局限包括仅覆盖中美两国、任务形式主要为多选/真伪、对抽象理解提升有限，以及路由器仅能单选专家而非多专家协同。

---

## 548. Sparse Contrastive Learning for Content-Based Cold Item Recommendation

**arXiv ID:** 2604.12990 | [PDF](https://arxiv.org/pdf/2604.12990v1)

**作者:** Gregor Meehan `[一作]` (Queen Mary University of London), Johan Pauwels `[通讯]` (Queen Mary University of London)

**通讯引用:** 333 | [OpenAlex ID](https://openalex.org/A5048509747)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了一种纯内容的冷启动推荐方法SEMCo，通过在内容空间学习物品间相似性来预测用户偏好。

**💡 创新点**

创新点在于将采样softmax损失泛化为α‑entmax（包括sparsemax）以实现稀疏监督，并在此框架下加入离线/在线知识蒸馏，使模型既能捕捉更精细的相似性，又能提升公平性。

**🔧 技术方法**

核心技术包括：内容编码器（多模态融合+注意力）、α‑entmax激活与Fenchel‑Young损失、对比学习式采样softmax、知识蒸馏（学生-教师相似性匹配）以及温度调度与L2正则。

**📊 数据集**

在四个多模态数据集上评估：Amazon Clothing、Electronics、音乐数据集M4A‑Onion、微视频数据集Microlens。

**📈 对比分析**

与五个冷启动基线（ALDI、CLCRec、GAR、GoRec、Heater）比较，SEMCo在Recall@20、NDCG@20和MDG@20上均显著提升，尤其在Electronics上提升超过133%，在M4A‑Onion上Gini多样性提升近149%；同时保持或提升用户侧排序性能。

**⚠️ 局限性**

局限性包括：需要完整物品集进行用户嵌入乘法，规模化受限；在线蒸馏对超参数敏感，训练稳定性需注意；以及在缺失模态时需进一步研究。

---

