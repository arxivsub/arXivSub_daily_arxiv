# arXiv Daily Summary

![Last Commit](https://img.shields.io/github/last-commit/arxivsub/arXivSub_daily_arxiv?label=Updated)
![Arxiv](https://img.shields.io/badge/arXiv-Papers-B31B1B.svg)
![Python](https://img.shields.io/badge/Powered%20By-Python-3776AB?logo=python&logoColor=white)
![Views](https://komarev.com/ghpvc/?username=arxivsub&repo=arXivSub_daily_arxiv&label=Views&color=brightgreen&style=flat)
![License](https://img.shields.io/badge/license-MIT-green)

> 最后更新时间: 2026-04-06 | 今日论文总数: 454

> 更多内容请访问 [arXivSub](https://arxivsub.comfyai.app/)

---

## 1. Evaluating AI-Generated Images of Cultural Artifacts with Community-Informed Rubrics

**arXiv ID:** 2604.02406 | [PDF](https://arxiv.org/pdf/2604.02406v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 2. AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation

**arXiv ID:** 2604.02525 | [PDF](https://arxiv.org/pdf/2604.02525v1)

**作者:** Seonggon Kim `[一作]` (Advanced Micro Devices, Inc.), Eunhyeok Park `[通讯]` (POSTECH)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文针对大语言模型低精度训练中出现的离群值问题，首次系统分析了权重、激活和梯度张量的离群模式，并提出了基于这些模式自适应选择Hadamard变换与离群提取（OE）的 AdaHOP 方法，实现了在 MXFP4 精度下与 BF16 相当的训练质量。

**💡 创新点**

创新点：①将离群值划分为 Row、Column、None 三种结构，并证明其在训练过程中的稳定性；②设计了针对不同模式对的策略表（IHT、OE+IHT、全 BF16），实现真正的模式感知；③将策略与硬件友好的 Triton 核心融合，达到低开销且高吞吐的实现。

**🔧 技术方法**

使用的技术包括：Walsh‑Hadamard 变换（IHT、OHT）、离群提取（OE）、混合精度计算（MXFP4+BF16）、Triton 编写的融合内核、一次性 BF16 校准与 CV 指标的模式检测。

**📊 数据集**

实验数据集：C4 数据集；评估模型：Llama3.2‑1B、Llama3.2‑3B、Llama3.1‑8B、Instella‑3B。

**📈 对比分析**

与 BF16、Naïve MXFP4、MXFP4+Hadamard、Tseng 等方法对比。AdaHOP 在所有模型上均保持与 BF16 相近的训练损失与零样本下游任务准确率，且在内存压缩（最高 3.6×）与核级吞吐（最高 1.8×）上优于现有 MXFP4 方法；在整体训练吞吐上略高于 BF16（约 2%）。

**⚠️ 局限性**

局限性：①使用固定的 Hadamard 旋转，未结合可学习旋转；②仅验证了 Llama/Instella 体系，需验证其它架构；③仅针对 MXFP4，未扩展到其它低精度格式；④离群提取行/列数固定为 64，未做自适应调整。

---

## 3. Eliminating Illusion in Directed Networks

**arXiv ID:** 2604.02395 | [PDF](https://arxiv.org/pdf/2604.02395v1)

**作者:** Sougata Jana `[一作]` (Indian Statistical Institute), Sanjukta Roy `[通讯]` (Indian Statistical Institute)

**通讯引用:** 771 | [OpenAlex ID](https://openalex.org/A5072480352)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究了在有向社会网络中通过最少改色实现消除多数幻觉（majority illusion）和任意比例幻觉（p‑illusion）的计算问题，并给出了其复杂性与多种图结构下的求解算法。

**💡 创新点**

创新点包括：①将幻觉消除问题推广到有向图并引入p‑illusion概念；②证明该问题在网格图上NP‑完备、在有向无环图上W[2]‑硬，显示即使在结构化图上也难以求解；③在外环图、向外网格、树和环等稀疏结构上提供多项式时间算法；④提出基于树宽和最大缺陷的FPT算法以及基于受幻觉顶点数的ILP FPT算法。

**🔧 技术方法**

主要技术包括：归约证明（从Planar Monotone Rectilinear 3‑SAT 与 Hitting Set 进行多相归约）；结构化图分析与贪心/匹配算法；树宽动态规划与nice树分解；ILP与König定理求解最小顶点覆盖；以及对图的分解与重构。

**📊 数据集**

由于研究为理论性质分析，文中未使用实际数据集，仅在理论构造的实例和实验性证明中使用了合成图。

**📈 对比分析**

与已有工作相比，本文在有向图上给出了更严格的复杂性边界；对外环图、向外网格和树的多项式时间算法相对之前仅在无向图中已知的结果提供了重要扩展；FPT算法的运行时间为O((2D)^tw · n^O(1))，证明在树宽有限时问题可高效解决。

**⚠️ 局限性**

局限性包括：①对有向图宽度参数（如有向树宽、切宽等）仍无法得到FPT结果；②在大多数一般图上仍无多项式时间或近似算法；③论文未给出实验评估，只是理论复杂度；④未讨论多源幻觉或动态网络中的实际应用。

---

## 4. Contextual Intelligence The Next Leap for Reinforcement Learning

**arXiv ID:** 2604.02348 | [PDF](https://arxiv.org/pdf/2604.02348v1)

**作者:** André Biedenkapp `[一作]` `[通讯]` (Albert-Ludwigs-Universität), André Biedenkapp (Albert-Ludwigs-Universität)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

提出了一套基于上下文的强化学习（cRL）新分类与框架，并综述了相关技术与方法。

**💡 创新点**

创新点在于将上下文划分为环境外因（allogenic）与自生成因（autogenic），进一步区分其时间尺度和抽象层级，为cRL提供统一的理论视角。

**🔧 技术方法**

主要综述了域随机化、程序生成、超网络、Dreamer latent注入、系统识别等技术，以及多时间尺度表示与上下文抽象的思路。

**📊 数据集**

本文为理论与综述性质，未使用特定数据集进行实验。

**📈 对比分析**

未进行实验比较，评估主要通过对已有工作和理论分析的梳理，未给出数值性能结果。

**⚠️ 局限性**

局限性包括缺乏统一实现与评估标准、对抽象上下文的实验证据不足、以及多时间尺度融合与实时推断的具体算法尚未成熟。

---

## 5. Too Polite to Disagree: Understanding Sycophancy Propagation in Multi-Agent Systems

**arXiv ID:** 2604.02668 | [PDF](https://arxiv.org/pdf/2604.02668v1)

**作者:** Vira Kasprova `[一作]` (University of Illinois Urbana-Champaign), Dilek Hakkani-Tur `[通讯]` (University of Illinois Urbana-Champaign)

**通讯引用:** 9728 | [OpenAlex ID](https://openalex.org/A5068709817)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究在多代理LLM讨论框架中，给每个代理提供其同伴妥协倾向（sycophancy）排名，从而抑制讨论中的妥协行为并提升最终答案的准确性。

**💡 创新点**

创新点在于提出了三种无须真值标签、可在推理时轻量使用的妥协倾向先验（BSS、DBSS、DSS），并验证其能显著提升多代理讨论效果。

**🔧 技术方法**

采用了多轮对话协议、妥协倾向指标（Agreement Rate、Stance‑Change Sycophancy、Confident Sycophancy）、静态与动态评分机制，并将排名嵌入代理提示中进行实验。

**📊 数据集**

实验使用MMLU基准的5个子任务共250道题（每个子任务50道），并在15个新子任务上进行验证，模型包括Qwen和Llama系列开源LLM（3B–32B）。

**📈 对比分析**

与无先验、准确率排名及随机排名的对照实验比较，结果显示BSS方案使多数意见准确率绝对提升10.5%，各模型准确率均有提升，翻转率下降，后讨论妥协倾向显著下降。

**⚠️ 局限性**

局限性包括仅在固定规模（6个代理）和特定任务（MMLU）上验证，未覆盖更大规模或不同领域；妥协排名需要先行标注或多轮对话；对多代理系统的通用性和对动态协作的进一步探索仍待研究。

---

## 6. Adaptive Learned State Estimation based on KalmanNet

**arXiv ID:** 2604.02441 | [PDF](https://arxiv.org/pdf/2604.02441v1)

**作者:** Arian Mehrfard `[一作]` (Mercedes-Benz AG), Mirko Mählisch `[通讯]` (University of Bundeswehr Munich)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并评估Adaptive Multi-modal KalmanNet（AM‑KNet），一种在自动驾驶多传感器环境下的混合状态估计器，将传统Kalman滤波与深度学习模块相结合；

**💡 创新点**

创新点包括：①针对雷达、激光雷达、摄像头分别设计测量模块，学习各自噪声特性；②引入基于Joseph公式的协方差估计分支，并用负对数似然损失对估计误差和创新误差进行监督；③采用带上下文调制的超网络，根据目标类型、运动状态和相对姿态对网络进行动态调节；④设计基于传感器特性、目标类别、运动状态和数据流可靠性的多维加权损失；

**🔧 技术方法**

使用KalmanNet框架、GRU循环网络、超网络与上下文调制、Joseph协方差更新、负对数似然（NLL）损失、组件化加权MSE损失；

**📊 数据集**

在nuScenes和View‑of‑Delft（VoD）两大自动驾驶数据集上训练与评估，数据涵盖雷达、激光雷达与摄像头检测；

**📈 对比分析**

与OAFuser和UKF‑EOT两种基线方法进行对比，采用MAE、NEES和NIS等指标；AM‑KNet在位置、速度和尺寸误差上均优于基线，位置NEES一致率提升至约77%（VoD）和60%（nuScenes），表现出更好的估计精度与不确定性一致性；

**⚠️ 局限性**

局限性包括：对横向动力学捕捉尚不完善；基线方法未经过专门调优，可能导致比较偏差；传感器时间同步与精度仍是挑战；学习模块对极端噪声可能过度自适应；未对实时计算开销进行详细评估。

---

## 7. Finding Belief Geometries with Sparse Autoencoders

**arXiv ID:** 2604.02685 | [PDF](https://arxiv.org/pdf/2604.02685v1)

**作者:** Matthew Levinson `[一作]` `[通讯]` (Independent Researcher), Matthew Levinson (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出并验证了一套从稀疏自编码器到k-子空间聚类、AANet拟合，再到重心预测优势检验的流程，用于在大型预训练语言模型中发现几何结构化的内部表征。

**💡 创新点**

创新点在于将稀疏自编码器的特征方向与k-子空间聚类相结合，使用AANet实现非线性原型分析，并通过重心预测优势（barycentric predictive advantage）与因果驱动（causal steering）双重检验，首次在无标注自然文本中探索“信念几何”表征。

**🔧 技术方法**

采用稀疏自编码器（SAE）、k-子空间聚类、AANet（神经原型分析）、重心预测优势检验、KL 散度一致性检查以及基于Qwen-2.5-72B的因果驱动评估等技术。

**📊 数据集**

使用了控制性的多元隐马尔可夫模型（Multipartite HMM）数据进行验证，并在Gemma-2-9B的第20层残差流上运用FineWeb序列；评估时利用Qwen-2.5-72B进行语义驱动的验证。

**📈 对比分析**

与随机生成的空集群对比，5/13真实候选集群在重心预测优势检验中通过至少一项分割（近顶点或内部），而零个空集群通过；因果驱动得分虽低但在最可信的集群（768_596）上与预测优势共现，表明存在一定的功能性几何结构。

**⚠️ 局限性**

局限性包括：仅在Gemma-2-9B第20层单一模型上实验，效果大小有限，因果驱动与预测优势仅提供相关或弱因果证据，存在“幻影顶点”，且缺乏可验证的结构化评测数据集，无法进一步确认信念状态的真实编码。

---

## 8. Generative AI Use in Entrepreneurship: An Integrative Review and an Empowerment-Entrapment Framework

**arXiv ID:** 2604.02567 | [PDF](https://arxiv.org/pdf/2604.02567v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 9. Causal-Audit: A Framework for Risk Assessment of Assumption Violations in Time-Series Causal Discovery

**arXiv ID:** 2604.02488 | [PDF](https://arxiv.org/pdf/2604.02488v1)

**作者:** Marco Ruiz `[一作]` (Instituto Superior Técnico), Rodrigo Ventura `[通讯]` (Instituto Superior Técnico)

**通讯引用:** 1984 | [OpenAlex ID](https://openalex.org/A5052413681)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 Causal-Audit 框架，用于在进行时间序列因果发现前评估假设违背风险并做出推荐或拒绝决策。

**💡 创新点**

创新点在于将假设验证转化为校准的概率风险评估，结合多维诊断、风险合成、决策理论和可置信区间，并实现了自动化的拒绝机制；同时公开了合成 DGP Atlas 作为评估基准。

**🔧 技术方法**

使用了时间序列统计诊断（ADF、KPSS、Bai‑Perron、Gap CV、MCCAR、Ljung‑Box 等）、逻辑回归加自举校准、单调等距回归、SHAP 解释、Bootstrap 并行、以及基于效用函数的选择策略。

**📊 数据集**

主要数据集包括：1）500 个合成数据集（Synthetic DGP Atlas）覆盖 10 种违背族；2）外部基准 TimeGraph（18 类）和 CausalTime（3 类）用于泛化验证；3）公开的 PCMCI+ 与 VAR‑Granger 结果用于实验对比。

**📈 对比分析**

与无审计的 PCMCI+ 基准以及简单门控（ADF + T_eff 阈值）比较，Causal‑Audit 在被推荐的数据上将误报率从 38% 降至 14%（减少 62%），在严重违背时拒绝率达到 78%，并在 21 个外部评估中保持 100% 与基准一致。校准指标均优于 0.95 的 AUROC、<0.05 的 ECE。

**⚠️ 局限性**

局限包括：1）仅覆盖线性 VAR(1) 过程，非线性风险尚未校准；2）假设库仅包含 PCMCI+ 与 VAR‑Granger，其他方法待扩展；3）部分诊断（如缺失机制、可辨识性）仍无法完全检测；4）在复杂/混合违背下校准误差略增（约 6%）。

---

## 10. TrackerSplat: Exploiting Point Tracking for Fast and Robust Dynamic 3D Gaussians Reconstruction

**arXiv ID:** 2604.02586 | [PDF](https://arxiv.org/pdf/2604.02586v1)

**作者:** Daheng Yin `[一作]` (Simon Fraser University), Jiangchuan Liu `[通讯]` (Simon Fraser University)

**通讯引用:** 20689 | [OpenAlex ID](https://openalex.org/A5039311485)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `aaccfe5c-6b26-4208-b23c-35331481e142` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出 TrackerSplat，一种利用点追踪对 3D Gaussian Splatting 进行动态场景重建的方法，能在大幅度帧间位移下保持图像质量；

**💡 创新点**

核心创新在于将多视角点追踪结果通过并行加权增量最小二乘（PWI-LS）转换为 2D 运动，再三角测量更新 3D Gaussian 的位置、旋转与尺度，并加入运动正则化与后续细化，突破了传统仅依赖梯度更新导致的失效；

**🔧 技术方法**

使用点追踪模型 DOT、PWI‑LS、三角测量、奇异值分解、特征正则化、GPU 并行化以及最终的 3DGS 细化训练；

**📊 数据集**

在 Meeting Room、Neural 3D Video Synthesis (N3DV)、Dynamic3DGS、st‑nerf、RH20T 等公开动态场景数据集上评估；

**📈 对比分析**

与 ST‑4DGS、4DGS、Dynamic3DGS、HiCoM 等基线在单 GPU 与 1/2/4/8 GPU 并行设置下对比，TrackerSplat 在绝大多数情形下在 PSNR/SSIM/LPIPS 指标上表现更优，并在多 GPU 场景下显著提升重建吞吐量；

**⚠️ 局限性**

局限包括对小/薄物体追踪失效导致细节模糊、低纹理区产生抖动、误差累积导致长期漂移、完全遮挡物体无法恢复等问题。

---

## 11. Opal: Private Memory for Personal AI

**arXiv ID:** 2604.02522 | [PDF](https://arxiv.org/pdf/2604.02522v1)

**作者:** Darya Kaviani `[一作]` (University of California, Berkeley), Raluca Ada Popa `[通讯]` (University of California, Berkeley)

**通讯引用:** 7871 | [OpenAlex ID](https://openalex.org/A5015782472)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

开发了一套在可信硬件内完成所有数据依赖推理、并通过 ORAM 在云端隐藏访问模式的个人 AI 记忆系统。

**💡 创新点**

创新点在于将知识图谱过滤与“oblivious dreaming”维护结合进可信 enclave，既保留了对实体、时间等结构化上下文的查询精度，又实现了无泄漏的可扩展长期存储。

**🔧 技术方法**

技术包括 Ring ORAM、IVF‑PQ 近似最近邻索引、嵌入模型、LLM、Intel TDX 信任执行环境、AES‑128‑GCM 加密、SHA‑256 Merkle 树、哈希签名等。

**📊 数据集**

使用自研的多模态合成个人数据管线（基于 Hawkes 过程），生成覆盖邮件、消息、会议、文档、语音等多模态、跨年的合成数据集。

**📈 对比分析**

通过与 ANN‑only、Graphiti、LIRE、Plaintext 及 In‑Memory 基线对比，KG‑过滤搜索在准确率上提升 13pp，ORAM 访问带宽下降至 O(log N)，吞吐量提升 29×、基础设施成本降低 15×；单查询/写入延迟仅 2.32 s/0.94 s，显著优于基线。

**⚠️ 局限性**

局限在于仍需依赖可信硬件、对图谱维护和索引更新的细粒度成本；多设备同步需要统一逻辑客户端；尚未覆盖图像/视频等原生多媒体数据，且 TCB 仍有限制。

---

## 12. Using LLM-as-a-Judge/Jury to Advance Scalable, Clinically-Validated Safety Evaluations of Model Responses to Users Demonstrating Psychosis

**arXiv ID:** 2604.02359 | [PDF](https://arxiv.org/pdf/2604.02359v1)

**作者:** May Lynn Reese `[一作]` (Apart Research), Elizabeth Stade `[通讯]` (Stanford Institute for Human-Centered AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文研究了大语言模型在精神分裂症等精神病患者中的安全性评估，开发并验证了七项临床评估标准，构建了人工共识数据集，并使用LLM-as-a-Judge与LLM-as-a-Jury方法对模型回复进行自动评估。

**💡 创新点**

创新点在于首次在精神病情境中构建可扩展、临床验证的安全评估框架，采用LLM自动评判替代人工评估，并提出七条明确的二元安全标准。

**🔧 技术方法**

采用READI安全组件、零射击提示、Cohen’s Kappa一致性评估，并使用Gemini、Qwen、Kimi三大LLM作为评判者，进一步通过多模型多数投票实现LLM-as-a-Jury。

**📊 数据集**

使用从临床心理文献提取的19个症状情境（3个保留，16个实验），将其转换为第一人称提示，收集四大LLM（GPT‑4o、Claude Sonnet、DeepSeek、Llama）对这些提示的回复作为数据集。

**📈 对比分析**

通过将LLM评判结果与人工共识进行Cohen’s Kappa比较，Gemini和Qwen分别取得0.75和0.68一致，Kimi 0.56；LLM-as-a-Jury得到0.74；其中“无转诊”指标一致性最高（0.97/1.00），而“增添幻想”最低（0.34）。

**⚠️ 局限性**

局限性包括样本量小、仅包含症状示例缺乏对照、未使用真实临床输入、评判者缺乏专业临床训练、模型未进行微调，以及评估仅限单轮对话。

---

## 13. YC Bench: a Live Benchmark for Forecasting Startup Outperformance in Y Combinator Batches

**arXiv ID:** 2604.02378 | [PDF](https://arxiv.org/pdf/2604.02378v1)

**作者:** Mostapha Benhenda `[一作]` `[通讯]`, Mostapha Benhenda

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计 YC Bench 基准，用 Pre-Demo Day Score 作为短期代理指标来评估 YC 加速器批次内的创业公司早期表现；

**💡 创新点**

创新点在于利用 YC 批次的结构实现三个月内可观测的评估，结合公开轨迹信号与 Google 搜索提及通过最大聚合生成评分，并将申请截止前的 Google mentions 作为可复现的基线；

**🔧 技术方法**

技术上使用最大加权信号聚合来计算 traction score、Google 提及计数作为 attention score，最终通过 max(traction, attention) 得到 Pre-Demo Day Score；评估采用 Precision@20 与 Recall@11；

**📊 数据集**

使用的数据集为 YC W26 批次的 196 家创业公司，其中 184 家有可检索的 Google domain mentions，11 家有公开的 ARR、pilot revenue、LOI、active users、activity volume 等 traction 数据；

**📈 对比分析**

比较方法是将基线（申请截止前 67 天 Google mentions 计数）与随机预测进行对比；基线在 Precision@20 上 30%（vs 10.9% 随机），Recall@11 为 55%（vs 0% 随机），相对提升 2.75 倍；

**⚠️ 局限性**

局限性包括预评分权重仅为启发式未经过学习，traction 数据覆盖仅 11 家公司，主要依赖 Google 提及可能忽略其他关注渠道，且评估仅在单个批次上，需要更多批次验证。

---

## 14. Matrix Profile for Time-Series Anomaly Detection: A Reproducible Open-Source Benchmark on TSB-AD

**arXiv ID:** 2604.02445 | [PDF](https://arxiv.org/pdf/2604.02445v1)

**作者:** Chin-Chia Michael Yeh `[一作]` `[通讯]` (University of California Riverside), Chin-Chia Michael Yeh (University of California Riverside)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发并提交了一套基于矩阵谱（Matrix Profile）的异常检测系统MMPAD，完成了完整的实现、参数调优与在TSB-AD基准上的评测。

**💡 创新点**

主要创新点包括：① 预排序（pre‑sorting）多维距离聚合，解决K‑of‑N异常问题；② 高效的排除区感知kNN检索，利用QuickSelect+局部排序减少计算；③ 移动平均后处理统一输出长度，保证可评估性；④ 结合CPU与GPU实现，支持长序列的预算友好式下采样。

**🔧 技术方法**

所使用的核心技术有：矩阵谱、预排序多维聚合、排除区kNN检索、移动平均平滑、可选的PCA whitening预处理；实现采用Python/Numba+CUDA，兼容CPU与GPU。

**📊 数据集**

实验数据集为TSB‑AD基准中的350条单变量与180条多变量时间序列，分别在评估集和调参集上进行无监督评测。

**📈 对比分析**

通过VUS‑PR（主要指标）与其他基准方法对比，MMPAD在单变量轨道上平均VUS‑PR为0.4100排名第2，在多变量轨道上0.3548排名第1；k>1显著提升性能；在不同子集（如单/多异常、短/长序列、点/序列异常）表现不均，显示对某些数据特征更为敏感。

**⚠️ 局限性**

局限性：① 对高维（>30维）多变量数据以及点/短异常的表现仍落后于某些深度学习基线；② 仍无法完全消除跨通道相关性带来的弱点；③ 在长序列或高频异常场景下，单变量对重复/持续异常的捕获能力不足。

---

## 15. Online Drone Coverage of Targets on a Line

**arXiv ID:** 2604.02491 | [PDF](https://arxiv.org/pdf/2604.02491v1)

**作者:** Stefan Dobrev `[一作]` (Slovak Academy of Sciences), Sunil Shende `[通讯]` (Rutgers University)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文研究单架携带固定视角相机/天线的无人机在一次性出现的目标点线上进行在线覆盖的问题，目标是最小化无人机路径长度，使用竞争比作为性能指标。

**💡 创新点**

创新点在于：①提出了三种在线覆盖算法，并给出它们对任意扫描角度α∈[0,π/2]的竞争比；②设计了一个新算法通过调节运动角度β(β是α的函数)在α∈(π/6,π/3)范围内显著优于其他两种算法；③给出了α≤π/4时的非平凡下界(1+√2)/2≈1.207)，证明任何确定性在线算法的竞争比无法低于该值。

**🔧 技术方法**

主要技术包括：几何分析（扫描锥、三角形相似性）、竞争分析、在线算法设计、最优移动策略求解、极值与微分分析以确定最佳β以及下界构造。

**📊 数据集**

该研究为纯理论工作，未使用任何实验数据集；所有结果均来自数学证明与理论推导。

**📈 对比分析**

通过竞争比对比：最佳算法在α=π/4时竞争比为1.25，其他两种算法均为√2≈1.414；在α≤π/4时所有算法的竞争比均不低于下界1.207；综上，提出的算法在中等扫描角度下表现最佳。

**⚠️ 局限性**

局限性包括：仅考虑线性目标分布和单架无人机；扫描角度被限制在α<π/2；理论分析未验证在更现实的三维空间、动态障碍或多无人机协作场景下的适用性；实际能耗模型未与真实硬件匹配。

---

## 16. Internalized Reasoning for Long-Context Visual Document Understanding

**arXiv ID:** 2604.02371 | [PDF](https://arxiv.org/pdf/2604.02371v1)

**作者:** Austin Veselka `[一作]` `[通讯]` (LightOn), Austin Veselka (LightOn)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一套用于长文档视觉问答的合成推理数据流水线，自动生成页面相关性评分、证据提取和有序推理轨迹，并在此基础上进行监督微调；

**💡 创新点**

通过低强度模型融合实现推理能力的“内部化”，使模型在不输出显式思考步骤的情况下仍能发挥推理优势，并证明合成推理轨迹比直接使用模型思考轨迹更有效；

**🔧 技术方法**

利用视觉语言模型（Qwen3 VL 32B、Mistral Small 3.1 24B）进行推理和答案生成；采用自监督微调（SFT）、模型融合（task arithmetic）和控制令牌训练；

**📊 数据集**

使用约250K份PDF文档（约16M页）以及PDFA English split的2M份PDF（18M页）生成的合成问答样本；

**📈 对比分析**

在多项长文档基准（MMLongBenchDoc、MMLBD-C、MMLongBench、DUDE、SlideVQA、HELMET、LongBench v2等）上与基线模型（Qwen3 VL 235B、Qwen3 VL 235B Thinking、LongPO等）对比，32B模型达58.3的准确率，超过同类参数量模型；同样在Mistral上显著提升；

**⚠️ 局限性**

实验中v1→v2轨迹改动包含多项改进，难以单独评估各项贡献；仅使用SFT，未结合RL或偏好优化；评测依赖局部裁剪和内部评审，可能导致偏差；内部化机制尚无理论解释；

---

## 17. Automated Malware Family Classification using Weighted Hierarchical Ensembles of Large Language Models

**arXiv ID:** 2604.02490 | [PDF](https://arxiv.org/pdf/2604.02490v1)

**作者:** Samita Bai `[一作]` (University of New Brunswick), Ali A. Ghorbani `[通讯]` (University of New Brunswick)

**通讯引用:** 26444 | [OpenAlex ID](https://openalex.org/A5034685391)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于大语言模型（LLM）的加权分层集成方法，用零标签方式实现恶意软件家族分类；

**💡 创新点**

创新点在于：1）采用决策层级集成而非特征学习；2）通过宏F1自适应权重提升模型可靠性；3）引入粗到细的行为分层决策结构，减少语义模糊和模型不稳定；

**🔧 技术方法**

使用预训练LLM（Qwen3-4B、CodeLLaMA-7B、GPT‑4.1、GPT‑5.1）及其加权集成与分层决策算法；

**📊 数据集**

利用SBAN数据集中的Windows PE恶意样本（含多家族、加壳/混淆）以及200样本手工标注的金标集；

**📈 对比分析**

与单一LLM、统一投票、加权投票无层级等方案对比，最终加权分层集成在Accuracy达0.750、Macro‑F1 0.481等指标上均优于基线，表现出更好的鲁棒性和均衡性；

**⚠️ 局限性**

局限性包括：1）仅基于静态源代码，未利用动态信息；2）分层结构手工设定，可能限制泛化；3）金标集规模有限，可能影响权重校准与评估可信度。

---

## 18. Compositional Neuro-Symbolic Reasoning

**arXiv ID:** 2604.02434 | [PDF](https://arxiv.org/pdf/2604.02434v1)

**作者:** Anugyan Das `[一作]` (CoreThink AI), Asad Aali `[通讯]` (Stanford University)

**通讯引用:** 765 | [OpenAlex ID](https://openalex.org/A5046938499)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了一种神经-符号框架，通过先提取格子任务中的对象级结构，再利用神经网络生成有限DSL的候选变换，最后通过跨示例一致性过滤来实现ARC‑AGI‑2任务的系统性推理。

**💡 创新点**

核心创新在于将感知与规则诱导严格分离，利用固定的原子变换语言和交叉示例一致性约束显著降低组合搜索空间，从而提升泛化能力。

**🔧 技术方法**

技术包括基于连通组件的对象抽象、神经网络的变换候选生成（使用LLM的结构化输出）、交叉示例一致性过滤以及自一致性采样（Self‑Consistency）和元分类器组合。

**📊 数据集**

使用ARC‑AGI‑2数据集，评估在公开评测集的pass@2指标。

**📈 对比分析**

与单一大型语言模型相比，单独的组合推理器达到24.4%准确率，元分类器组合提升至30.8%，显著优于传统LLM（最高18.3%），证明了结构化约束的有效性。

**⚠️ 局限性**

局限包括DSL表述不完整导致无法解决更深层次的关系推理任务、依赖自一致性导致的计算成本增加，以及对LLM提示的鲁棒性和可扩展性尚待提升。

---

## 19. Reliability-Aware Geometric Fusion for Robust Audio-Visual Navigation

**arXiv ID:** 2604.02391 | [PDF](https://arxiv.org/pdf/2604.02391v1)

**作者:** Teng Liu `[一作]` (Xinjiang University), Yinfeng Yu `[通讯]` (Xinjiang University)

**通讯引用:** 3726 | [OpenAlex ID](https://openalex.org/A5091800151)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出RAVN框架，利用音频可信度动态调节视觉特征，实现鲁棒音视频导航。

**💡 创新点**

通过异方差性几何推理器AGR学习音频不确定性作为可靠性指标，并使用RAGM实现可调节的视觉门控融合，突破传统静态融合的局限。

**🔧 技术方法**

采用异方差高斯负对数似然损失、软门控调制、GRU策略网络以及PPO强化学习等技术。

**📊 数据集**

在SoundSpaces平台的Replica与Matterport3D两个室内环境数据集上进行实验。

**📈 对比分析**

与AV-Nav基线对比，RAVN在所有声源设置下均提升SR、SPL和SNA，尤其在未见声源下SR提升12.3%。

**⚠️ 局限性**

仅在仿真环境验证，未测试真实机器人，且在极端噪声或动态传感器失效情况下的鲁棒性仍需进一步验证。

---

## 20. Evolution and Perspectives of the Keep IT Secure Ecosystem:A Six-Year Analysis of Cybersecurity Experts Supporting Belgian SMEs

**arXiv ID:** 2604.02425 | [PDF](https://arxiv.org/pdf/2604.02425v1)

**作者:** Christophe Ponsard `[一作]` (CETIC Research Centre), Nicolas Point `[通讯]` (Multitel Research Centre)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过Keep IT Secure计划，对比基于NIST CSF与CIS控制的评估框架，追踪分析了2019-2025年间在比利时瓦隆区认证的122名网络安全专家的成熟度与生态演进。

**💡 创新点**

创新点在于构建可与联邦CyFun框架对齐的轻量级专家认证体系，并将AI驱动的分析框架与LLM支持的公司信息抽取相结合，实现持续、可复现的评估与生态监测。

**🔧 技术方法**

采用NIST CSF 1.1与CIS20控制的评估网格、Python脚本自动化评分、LLM进行公司业务与专业领域的无结构信息提取，以及雷达图可视化呈现。

**📊 数据集**

使用122份手工评估问卷（覆盖87家公司）以及公开目录与公司网站抓取的数据，构成完整的专家与企业特征数据集。

**📈 对比分析**

将KIS评估结果与CyFun 2025基础级认证指标对齐，计算各CSF阶段的平均得分，发现保护与识别阶段得分最高，而检测、响应与恢复阶段持续相对薄弱，整体得分随时间波动但大致稳定。

**⚠️ 局限性**

研究局限在于样本仅来自瓦隆区，无法外推至其他国家；评估问卷侧重技术能力，未充分覆盖软技能；且新法规与AI威胁的评估尚未完整纳入。

---

## 21. Skeleton-based Coherence Modeling in Narratives

**arXiv ID:** 2604.02451 | [PDF](https://arxiv.org/pdf/2604.02451v1)

**作者:** Nishit Asnani `[一作]` (Stanford University), Rohan Badlani `[通讯]` (Stanford University)

**通讯引用:** 177 | [OpenAlex ID](https://openalex.org/A5058296176)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了句子/骨架相似度网络（SSN），评估文本一致性，并对比使用骨架与完整句子。

**💡 创新点**

将骨架提取模型从生成转为判别任务，并引入Siamese LSTM+自注意力的骨架/句子相似度网络。

**🔧 技术方法**

Siamese LSTM、对比损失、FastText词向量、自注意力机制以及BERT平均向量。

**📊 数据集**

StoryTelling数据集（40153训练、4990验证、5054测试）。

**📈 对比分析**

与余弦/欧氏距离基线以及仅使用句子/骨架的SSN对比，句子SSN在句子序列、故事序列和配对分类上分别达92.9%、69.6%和82.2%，显著优于骨架SSN。

**⚠️ 局限性**

骨架质量受限导致性能下降，自注意力在当前实现未显著提升；短文本（≤6句）限制了故事级评估。

---

## 22. WGFINNs: Weak formulation-based GENERIC formalism informed neural networks'

**arXiv ID:** 2604.02601 | [PDF](https://arxiv.org/pdf/2604.02601v1)

**作者:** Jun Sur Richard Park `[一作]` (Korea University), Yeonjong Shin `[通讯]` (North Carolina State University)

**通讯引用:** 638 | [OpenAlex ID](https://openalex.org/A5023632213)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了弱形式生成的GENERIC框架网络（WGFINNs），以在噪声观测下更稳健地从数据中推断支配方程。

**💡 创新点**

创新点在于将弱形式动力学方程与GENERIC结构保持网络相结合，并加入状态加权损失和残差注意机制，显著提升了对噪声的鲁棒性与尺度不平衡的抑制。

**🔧 技术方法**

技术包括弱形式损失构造、GENERIC结构约束、状态加权训练、残差基注意力机制以及理论分析比较强形式与弱形式估计器的收敛性质。

**📊 数据集**

实验使用基于物理仿真的合成数据集（涵盖不同噪声水平的动力学系统），未涉及真实实验数据。

**📈 对比分析**

与传统GFINNs对比，WGFINNs在各类噪声水平下均实现了更精确的方程恢复、误差更小、物理量预测更可靠，实验结果证明其优越性。

**⚠️ 局限性**

局限性包括仍需选择合适的测试函数以保证弱形式收敛，且在极高噪声或复杂多尺度系统中性能尚未彻底验证。

---

## 23. Understanding the Effects of Safety Unalignment on Large Language Models

**arXiv ID:** 2604.02574 | [PDF](https://arxiv.org/pdf/2604.02574v1)

**作者:** John T. Halloran `[一作]` `[通讯]` (Leidos), John T. Halloran (Leidos)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对比两种不对齐方法（JT与WO）在六大LLM模型上的拒绝率、攻击能力、幻觉率及帮助性表现；

**💡 创新点**

发现WO比JT更具危险性，能更好保留原始功能并显著提升攻击成功率，同时提出通过监督微调（SFT）可部分缓解WO的威胁；

**🔧 技术方法**

使用JT的对抗训练数据与WO的权重正交化技术，并通过StrongREJECT、HarmBench、CyberSecEval 3、TruthfulQA、TofuEval、ARC‑E/C、HellaSwag、PIQA、Winogrande、MMLU、IFEval等评测工具进行评估；

**📊 数据集**

训练数据涵盖5k样本的2%恶意+98%善意混合集、HarmBench 200样本、CyberSecEval 1000攻击请求、TruthfulQA与TofuEval的公开语料以及ARC、HellaSwag、PIQA、Winogrande、MMLU、IFEval等常用基准；

**📈 对比分析**

实验结果显示WO在拒绝率下降最高、攻击成功率平均提升27.7%（尤其对推理模型提升40.2%），幻觉率仅微升3.6%，而JT幻觉率提升8.9%；SFT后可恢复约40–70%拒绝率并将攻击成功率下降约45%；

**⚠️ 局限性**

局限性包括：WO只能在白盒条件下实施；对网络攻击的缓解效果不一；缺乏黑盒实现方案；未系统评估长文本一致性与长期行为的安全性。

---

## 24. Moondream Segmentation: From Words to Masks

**arXiv ID:** 2604.02593 | [PDF](https://arxiv.org/pdf/2604.02593v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 25. An Empirical Study of Many-Shot In-Context Learning for Machine Translation of Low-Resource Languages

**arXiv ID:** 2604.02596 | [PDF](https://arxiv.org/pdf/2604.02596v1)

**作者:** Yinhan Lu `[一作]` (Mila Quebec Artificial Intelligence Institute), David Ifeoluwa Adelani `[通讯]` (Mila Quebec Artificial Intelligence Institute)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文评估了在英语→10种极低资源语言的机器翻译中使用多射击（up to 1,000例子）的上下文学习方法。

**💡 创新点**

创新点在于系统研究了大规模例子对低资源语言的影响，并证明了检索式（BM25）例子在样本效率上的优势。

**🔧 技术方法**

主要技术包括多射击提示、BM25检索、语言模型（Gemini 2.5 Flash、OpenAI GPT 等）以及自动评估指标 chrF++/spBLEU。

**📊 数据集**

使用的数据集包括 FLORES+ / FLORES-200 的10种低资源语言平行语料、圣经翻译文本以及随机抽取的例子。

**📈 对比分析**

通过与随机抽样、不同 k 值、域匹配与非匹配的对照实验，结果表明每增加一组例子可实现对数线性提升，BM25 检索的 50 例可与随机 250 例相当，250 例可匹配 1,000 例；总体提升最多可达约 35 点。

**⚠️ 局限性**

局限性包括仅涉及英中对、固定提示模板、缺乏人工评估、未使用嵌入式评估指标，以及高昂的 API 成本，限制了可复现性与对非英中语言的推广。

---

## 26. Efficient Path Query Processing in Relational Database Systems

**arXiv ID:** 2604.02553 | [PDF](https://arxiv.org/pdf/2604.02553v1)

**作者:** Diego Rivera Correa `[一作]` (Northeastern University), Mirek Riedewald `[通讯]` (Northeastern University)

**通讯引用:** 3682 | [OpenAlex ID](https://openalex.org/A5049802784)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种名为ReCAP的抽象，使路径查询的属性约束能在递归遍历中进行早期过滤，从而显著减少中间结果；

**💡 创新点**

通过定义“选择性聚合”，将任意属性约束与NFA状态相结合，仅需实现NFA转移函数和少量UDF，即可在关系型数据库中高效执行；

**🔧 技术方法**

采用递归CTE+NFA表、JSON/数组字典、字典展开、函数内联、索引优化等技术，并在DuckDB/PostgreSQL等通用RDBMS上实现；

**📊 数据集**

使用真实与合成图数据集，包括LDBC100、Datagen 7.6/7.7、Metaverse、Reddit、Bitcoin等；

**📈 对比分析**

与Neo4j、Kuzu、DuckPGQ等主流图DBMS以及PostgreSQL、DuckDB、SysX等关系型DBMS在相同查询、相同起点、计时中位数下比较，ReCAP在大多数查询中实现10³–10⁵倍甚至高达400 000倍的性能提升，且能处理更长路径；

**⚠️ 局限性**

仅适用于可单调、可裁剪的约束，对无法提前过滤的约束收益有限；对复杂属性仍需JSON/数组，存在解析开销；目前实现需要手工编译，缺乏自动化工具；

---

## 27. An Explainable Vision-Language Model Framework with Adaptive PID-Tversky Loss for Lumbar Spinal Stenosis Diagnosis

**arXiv ID:** 2604.02502 | [PDF](https://arxiv.org/pdf/2604.02502v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 28. Social Meaning in Large Language Models: Structure, Magnitude, and Pragmatic Prompting

**arXiv ID:** 2604.02512 | [PDF](https://arxiv.org/pdf/2604.02512v1)

**作者:** Roland Mühlenbernd `[一作]` (Leibniz-Centre General Linguistics), Roland Mühlenbernd `[通讯]` (Leibniz-Centre General Linguistics)

**通讯引用:** 244 | [OpenAlex ID](https://openalex.org/A5065885211)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

评估前沿LLM在社会推理中的结构性与量化一致性，提出ESR和CDS两个量化指标，并通过四种基于推理的提示策略检验其对校准的影响。

**💡 创新点**

创新在于将结构一致性与数量校准分离，设计了新的ESR/​CDS度量，并通过理论驱动的提示验证了语用推理对校准的可塑性。

**🔧 技术方法**

使用Rational Speech Act框架启发的提示，chain‑of‑thought、知识与动机提示以及组合提示，结合GPT、Claude和Gemini三大LLM进行推理测试。

**📊 数据集**

使用实验1的数值精度社会推理数据（371名受试者，六种情景、七点量表），并生成LLM对相同条件的评分。

**📈 对比分析**

通过Spearman、CCC、RMSE、DAS、ISS、ESR、CDS等指标比较，发现所有模型在方向一致性上达成100%（DAS/ISS=1），但Gemini在量化上显著失调；组合提示在三模型中统一提升校准。

**⚠️ 局限性**

局限包括仅针对数值精度一类语用场景，缺乏开放模型与小规模模型的评估，提示仅为近似实现，未能完全复制RSA推理。

---

## 29. Token-Efficient Multimodal Reasoning via Image Prompt Packaging

**arXiv ID:** 2604.02492 | [PDF](https://arxiv.org/pdf/2604.02492v1)

**作者:** Joong Ho Choi `[一作]` (BNY), Boyi Qian `[通讯]` (BNY)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并评估了图像提示包装（IPPg）技术，将结构化文本嵌入图像以降低令牌成本，兼顾成本与准确率；

**💡 创新点**

创新性地将文本直接转为视觉输入并系统量化其在不同模型、任务与渲染设置下的成本-性能权衡；

**🔧 技术方法**

使用现有大型多模态语言模型（GPT‑4.1、GPT‑4o、Claude 3.5 Sonnet）与PIL渲染工具，将文本嵌入图像；

**📊 数据集**

在五个公开数据集上验证：FAMMA、PathVQA、SROIE、HumanEval、CoSQL；

**📈 对比分析**

通过对比文本+图像与文本嵌入图像的收费与准确率，发现GPT‑4.1在多数任务可实现30%+成本降低且准确率几乎不变，而Claude 3.5 则成本上升且准确率下降；

**⚠️ 局限性**

受限于商业API计费、图像分辨率与模型视觉分辨率；对非英语、空间推理与字符敏感任务效果不佳；实现代码未公开。

---

## 30. Dependency-Guided Parallel Decoding in Discrete Diffusion Language Models

**arXiv ID:** 2604.02560 | [PDF](https://arxiv.org/pdf/2604.02560v1)

**作者:** Liran Ringel `[一作]` (Technion – Israel Institute of Technology), Yaniv Romano `[通讯]` (Technion – Israel Institute of Technology)

**通讯引用:** 2275 | [OpenAlex ID](https://openalex.org/A5041723842)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于依赖关系的并行解码框架DEMASK，利用单前向传递预测掩码位置之间的相互依赖，并在每一步只解码相互独立或低依赖的子集，从而减少分布失配导致的质量下降

**💡 创新点**

创新点在于：①训练一个轻量级的依赖预测器直接从隐藏状态估计对称非平稳的两两依赖矩阵；②提出贪婪子集选择算法，并给出在子可加性假设下的总变距离理论上限；③将该方法与现有的置信度、KL散度和左至右解码等方法对比，证明能在不降低质量的前提下显著加速生成

**🔧 技术方法**

技术包括：离散扩散语言模型（dLLM），基于Transformer的隐藏状态提取，scaled dot‑product attention实现的依赖预测器，贪婪子集选择算法，以及总变距离（TV）理论分析和子可加性假设；训练阶段使用两阶段缓存-预测器训练方案

**📊 数据集**

主要使用的评测数据集为Tulu‑3 SFT混合数据集用于训练依赖预测器，Dream‑7B模型在MMLU‑Pro、GSM8K、HumanEval、MBPP四个基准上进行评测；还在Tulu‑3 SFT混合数据集上验证子可加性假设

**📈 对比分析**

与Entropy、Top‑1、Token‑Order、KLASS等基线对比，DEMASK在Dream‑7B上平均提升53.3%准确率并实现约1.9×的速度提升；在各个子任务上相较于基线取得更高或相近的准确率，尤其在MMLU‑Pro上提高3.6%

**⚠️ 局限性**

局限性包括：理论保证依赖于子可加性假设，虽然在多数情况下成立但仍有少量违例；依赖预测器误差未被正式量化，可能影响TV上界；仅在Dream‑7B单一backbone上验证，未探究在其他dLLM架构的迁移效果

---

## 31. ROMAN: A Multiscale Routing Operator for Convolutional Time Series Models

**arXiv ID:** 2604.02577 | [PDF](https://arxiv.org/pdf/2604.02577v1)

**作者:** Gonzalo Uribarri `[一作]` `[通讯]` (Stockholm University), Gonzalo Uribarri (Stockholm University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出ROMAN——一种多尺度路由操作符，将时间序列的尺度和粗粒度位置编码为伪通道，作为预处理步骤；

**💡 创新点**

创新点在于将多尺度信息显式映射为通道维度，既控制平移不变性，又保持短时间轴；

**🔧 技术方法**

技术包括抗混叠多尺度金字塔、固定长度窗口切片、通道拼接和随机/学习卷积分类器；

**📊 数据集**

使用合成的四种机制任务以及UCR和UEA长序列子集（L≥256）进行评估；

**📈 对比分析**

与未加ROMAN的MiniRocket、MultiRocket、CNNClassifier、FCNClassifier进行对比；在需要粗位置、长程关联、多尺度交互的任务上提升准确率，在全平移不变任务上表现不佳，且计算成本因S值不同而变化；

**⚠️ 局限性**

局限包括：S取值范围有限导致通道爆炸、实验仅覆盖部分数据集、未针对不同任务自动调参，且ROMAN对纯平移不变任务无益。

---

## 32. SelRoute: Query-Type-Aware Routing for Long-Term Conversational Memory Retrieval

**arXiv ID:** 2604.02431 | [PDF](https://arxiv.org/pdf/2604.02431v1)

**作者:** Matthew McKee `[一作]` `[通讯]` (Independent Researcher), Matthew McKee (Independent Researcher)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了 SelRoute 框架，根据查询类型将检索请求路由到专用检索管道（词法、语义、混合或词汇增强），从而提升长时对话记忆检索效果。

**💡 创新点**

提出基于查询类型的选择性路由，并揭示词汇增强对词法检索有利但对嵌入检索不利的“异向性”，实现轻量级检索器间的动态组合。

**🔧 技术方法**

采用 SQLite FTS5 进行词法检索，bge 或 MiniLM 嵌入模型进行语义检索，Reciprocal Rank Fusion 进行混合检索，手工构建的同义/上位词桥接实现词汇增强，正则表达式进行查询类型分类。

**📊 数据集**

主要使用 LongMemEval_M 进行评估，交叉验证 5 折；同时在 8 个额外 benchmark（MSDialog、LoCoMo、QReCC、PerLTQA 等）测试泛化能力。

**📈 对比分析**

与 BM25、Stella V5、Contriever、Contriever+fact‑keys 等基线在 session‑level recall@5 对比，SelRoute（bge‑base）取得 0.800 recall@5，显著优于 0.762 的 Contriever+fact‑keys；在跨 benchmark 评测中保持竞争力，但在 reasoning‑intensive RECOR 上性能仅 0.149。

**⚠️ 局限性**

需要手工制定词汇扩展、查询类型分类准确率有限（尤其知识更新/用户查询），对多跳推理任务无效，且部分提升可能源自更强的词法引擎而非路由本身。

---

## 33. Holos: A Web-Scale LLM-Based Multi-Agent System for the Agentic Web

**arXiv ID:** 2604.02334 | [PDF](https://arxiv.org/pdf/2604.02334v1)

**作者:** Xiaohang Nie `[一作]` (Shanghai Innovation Institute), Weinan Zhang `[通讯]` (Shanghai Innovation Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

构建了 Holos，一个可扩展的 LLM‑基于多智能体系统，支持百万级代理的持续协作与价值循环。

**💡 创新点**

提出了五层架构、Nuwa 引擎、市场驱动调度以及端到端价值闭环，实现了开放世界下的可持续协同与自组织。

**🔧 技术方法**

利用 LLM（ReAct、Chain‑of‑Thought）、Nuwa 无服务器代理、Hybrid Sourcing 市场、LambdaMART 排序、AgentRank 信誉、S‑MMR 工具多样化、MCP 库、Agent2Agent 协议等技术。

**📊 数据集**

使用公开 Web 任务数据、HLE 数据集、MCP 工具库、公开基准（ChatGPT/Claude 等）及自构建的 RQT 角色‑工具‑任务三元组。

**📈 对比分析**

与 AutoGPT、MetaGPT、AutoGen 等 LLM 多智能体框架对比，Holos 在 Agent 数量、内存占用、查询延迟、任务成功率、经济信用相关性等指标上均显著优于现有系统。

**⚠️ 局限性**

局限在于大规模时的查找噪声影响、对高维语义空间的依赖、潜在的安全与伦理风险、Sybil 攻击易感性以及对非 LLM 传统工具的集成不足。

---

## 34. Dynamic Mask Enhanced Intelligent Multi-UAV Deployment for Urban Vehicular Networks

**arXiv ID:** 2604.02358 | [PDF](https://arxiv.org/pdf/2604.02358v1)

**作者:** Gaoxiang Cao `[一作]` (University of Science and Technology of China), Jian Yang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 45186 | [OpenAlex ID](https://openalex.org/A5100726984)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5a41884c-404f-4688-a89c-aa238c10fe68` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究多UAV动态部署以提升城市VANET的连通性

**💡 创新点**

提出基于得分动态动作掩码（SDAM）的Q-SDAM算法，能够在大动作空间中引导学习并跳出局部最优

**🔧 技术方法**

使用多智能体深度强化学习（QMIX + SDAM）、动作掩码技术以及能量消耗模型

**📊 数据集**

采用深圳和济南交通摄像头收集的实际车辆轨迹数据（约5百万条记录）构建的两种规模的路网图

**📈 对比分析**

与Q-SAM、DISCOUNT、μ-Greedy和随机策略进行对比，Q-SDAM在连通车辆数提升18.2%，能耗降低66.6%，且收敛更快、鲁棒性更好

**⚠️ 局限性**

依赖于前一时刻位置和车辆分布，动作掩码生成对动态交通变化的适应性有限，且在极大UAV数量下可能仍面临探索瓶颈

---

## 35. OmniTQA: A Cost-Aware System for Hybrid Query Processing over Semi-Structured Data

**arXiv ID:** 2604.02444 | [PDF](https://arxiv.org/pdf/2604.02444v1)

**作者:** Nima Shahbazi `[一作]` (Megagon Labs), Estevam Hruschka `[通讯]` (Megagon Labs)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了名为OmniTQA的成本感知混合查询处理框架，能够在同时包含结构化属性与自由文本字段的半结构化表上执行自然语言查询。框架将LLM语义推理视为一等算子，生成可执行的有向无环图（DAG），并采用双引擎（SQL引擎与LLM模块）动态调度算子。

**💡 创新点**

① 将关系算子与LLM语义算子统一进DAG，形成第一类语义算子；② 通过数据感知规划与成本模型，将昂贵的LLM调用推迟到关系算子已显著裁剪后再执行；③ 采用计划多样化（schema映射、风险配置、算子替换等）生成多条候选计划，并通过LLM裁决或投票聚合结果；④ 对语义算子进行算子感知批处理，显著降低API调用次数并保持上下文限制。

**🔧 技术方法**

核心技术包括：大模型推理（Gemini‑3‑Flash‑Preview、GPT‑5‑Mini、Qwen3）、SQL关系引擎、双引擎调度、成本模型（Relational cost + LLM token cost）、算子层次批处理、自动化SQL重写、关系算子推移、投影裁剪、Join重排序、Plan Diversification、LLM‑as‑Judge/Majority Voting。

**📊 数据集**

实验使用12个表问题回答基准：TAC（包含S1‑S5、M1‑M2子集）、WikiTableQuestions、FreeForm、MultiHop、Financial等，涵盖单表/多表、短表/长表、结构化/半结构化、复杂/简单查询。

**📈 对比分析**

与Direct‑LLM、NL2SQL、Plan‑of‑SQLs、H‑Star、Weaver等基线进行对比，评价指标包括Acc@6/Acc@1、LLM‑as‑Judge、Majority Voting；结果显示OmniTQA在大表、长表、多表、复杂查询场景中准确率提升高达48%（在TAC）/39%（在其他基准），整体准确率提升约2‑3%；成本方面，单计划Token使用与Direct‑LLM相近，整体成本低于或与现有混合方案持平，且多计划成本随计划数量线性增长。

**⚠️ 局限性**

主要限制：1）多方案生成仍受限于规划质量，导致部分错误与高成本；2）LLM算子仍占用显著计算资源，尤其在长文本与大表时；3）批处理大小与上下文窗口的权衡需手工调参；4）成本模型为粗粒度估计，可能导致优化偏差；5）对极大表的并行与分区策略仍需改进；6）计划错误主要集中在规划与语义归一化阶段。

---

## 36. What Are Adversaries Doing? Automating Tactics, Techniques, and Procedures Extraction: A Systematic Review

**arXiv ID:** 2604.02377 | [PDF](https://arxiv.org/pdf/2604.02377v1)

**作者:** Mahzabin Tamanna `[一作]` (North Carolina State University), Md Rayhanur Rahman `[通讯]` (University of Alabama)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

系统综述了80篇文献，分析了从网络威胁情报（CTI）文本自动提取攻击战术、技术与程序（TTP）的研究现状。

**💡 创新点**

创新点在于将文献按提取目标、数据源、数据采集与预处理、标注与构建、方法六大维度进行细致分类，并揭示了技术演进、主流趋势与研究缺口，形成了对TTP提取研究的整体框架。

**🔧 技术方法**

涉及技术涵盖：规则/模式匹配、传统机器学习、深度学习（CNN/LSTM/CRF）、Transformer（BERT、RoBERTa、SecBERT等）、大语言模型（LLM）与检索增强生成（RAG）等。

**📊 数据集**

主要使用的数据集包括 MITRE ATT&CK、CAPEC、CWE、TRAM 等公开知识库，以及多来源的 CTI 报告、CVE 条目、系统日志、IDS 规则等；但综述指出多数工作仅依赖单一或有限数据集。

**📈 对比分析**

比较方法多采用精确率、召回率、宏/微 F1、排名指标（Precision@K、Recall@K、MAP 等），Transformer 与 LLM 在大部分任务上优于传统模型，但缺乏跨数据集、跨时间的稳健性比较，整体评估仍以宏观指标为主。

**⚠️ 局限性**

主要局限包括：数据集单一、缺少多标签与上下文层面的评估；标注标准不统一、缺乏交叉验证；复现性差，数据/代码共享不足；模型泛化与鲁棒性评估有限。

---

## 37. Homophily-aware Supervised Contrastive Counterfactual Augmented Fair Graph Neural Network

**arXiv ID:** 2604.02342 | [PDF](https://arxiv.org/pdf/2604.02342v1)

**作者:** Mahdi Tavassoli Kejani `[一作]` (Institut de Mathématiques de Toulouse), Jean-Michel Loubes `[通讯]` (Institut de Mathématiques de Toulouse)

**通讯引用:** 1971 | [OpenAlex ID](https://openalex.org/A5025032659)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种新的公平图神经网络框架HSCCAF，解决图结构与敏感属性引起的偏差；

**💡 创新点**

创新点在于结合预处理图编辑、监督对比损失和环境损失，实现对称拆分内容与敏感信息并降低拓扑偏差；

**🔧 技术方法**

使用图神经网络（GraphSAGE/GCN等）作为编码器，加入对比学习、环境正则化、结构相似性约束；

**📊 数据集**

在五个真实数据集上实验：German、Bail、Credit、NBA、Pokec-n；

**📈 对比分析**

与标准GNN、FairGNN、FairVGNN、CAF、FairGB等方法对比，HSCCAF在保持或略低于最优预测准确率的同时，显著降低统计公平性差距与机会公平性差距；

**⚠️ 局限性**

主要局限是需要完整的敏感属性标签、额外超参数调优，以及仅适用于同质性图结构。

---

## 38. Differentiable Symbolic Planning: A Neural Architecture for Constraint Reasoning with Learned Feasibility

**arXiv ID:** 2604.02350 | [PDF](https://arxiv.org/pdf/2604.02350v1)

**作者:** Venkatakrishna Reddy Oruganti `[一作]` `[通讯]` (Sithara Inc.), Venkatakrishna Reddy Oruganti (Sithara Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 Differentiable Symbolic Planning (DSP)，将可微分的符号推理与图神经网络相结合，实现对约束满足性的显式跟踪与全局聚合。

**💡 创新点**

创新点在于：①引入可微的可行性通道 ϕ；②使用全局可行性聚合 Φ 以捕捉全局约束；③采用 sparsemax 进行稀疏离散规则选择，保证推理的可微性和精确性。

**🔧 技术方法**

技术核心包括：图注意力网络、规则嵌入、sparsemax 关注机制、门控更新、全局可行性聚合以及整体的 UCK+DSP 模块。

**📊 数据集**

使用三大约束推理基准：图可达性、布尔可满足性（SAT）和规划可行性（gridworld），分别从小规模训练到大规模测试进行大小泛化评估。

**📈 对比分析**

与基线（仅图注意力的 UCK、GIN）比较，UCK+DSP 在规划可行性下达 97.4%（平衡率 0.949），SAT 达 96.4%（平衡率 0.936），图可达性达 82.7%（平衡率 0.873），显著优于对照组并保持类间平衡。

**⚠️ 局限性**

局限性包括：推理步骤受限于固定的 T 步；仅完成二分类，未能输出满足赋值或路径；需要图结构表示，且规则数 K 为超参数，缺乏自适应机制。

---

## 39. CIPHER: Conformer-based Inference of Phonemes from High-density EEG

**arXiv ID:** 2604.02362 | [PDF](https://arxiv.org/pdf/2604.02362v1)

**作者:** Varshith Madishetty `[一作]` `[通讯]`, Varshith Madishetty

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建并评估了一个双通路的EEG语音解码框架CIPHER，利用ERP与DDA两种特征并通过Conformer编码器实现对音位的预测。

**💡 创新点**

创新点在于：①将ERP和DDA两种互补特征并行提取并共享同一Conformer编码器；②在非侵入性EEG上首次对11类CVC音位进行语音解码并给出真实词与非词的WER；③提供完整的预注册控制方案，阐明二分类高准确率主要由声学与TMS共混引起。

**🔧 技术方法**

使用的技术包括：ERP预处理（滤波、ICA、CAR、epoching）、DDA延迟微分分析、基于多尺度卷积的前端、SE通道注意力、4层Conformer、注意力池化、CTC头、混合混合数据增强、标签平滑等。

**📊 数据集**

使用公开的OpenNeuro ds006104数据集，包含24名受试者、两次实验（含TMS）、64通道BioSemi EEG，采样率2048Hz。

**📈 对比分析**

与传统浅层网络（EEGNet、EEG-Conformer等）和线性基线（LR、LDA）做对比，二分类任务达100%但被视为confound敏感；11类音位的平均WER为ERP 0.671±0.080、DDA 0.688±0.096，远高于随机预测的0.909，但仍离实际可用解码相距甚远。

**⚠️ 局限性**

限制包括：样本量仅24人，缺乏发声/想象语音数据，解码为刺激锁定、受限词汇，无法直接应用于实时或自由文本生成；声学与TMS共混导致二分类高分不可独立验证；EEG信噪比低导致精度受限。

---

## 40. Making Written Theorems Explorable by Grounding Them in Formal Representations

**arXiv ID:** 2604.02598 | [PDF](https://arxiv.org/pdf/2604.02598v1)

**作者:** Hita Kambhamettu `[一作]` (University of Pennsylvania), Andrew Head `[通讯]` (University of Pennsylvania)

**通讯引用:** 1834 | [OpenAlex ID](https://openalex.org/A5003473641)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了“可探究定理”系统，通过LLM将自然语言证明翻译为Lean形式并与原文本关联，使用户能够在界面上执行证明步骤、查看具体实例并追踪逻辑依赖；

**💡 创新点**

将自然语言证明与正式Lean证明绑定，提供可交互的步骤实例化与依赖追踪，从而提升证明理解的创新性；

**🔧 技术方法**

使用Claude Opus 4.6与GPT‑5.1生成并链接Lean证明，利用Lean执行获取中间状态，差分求取依赖图，并在前端实现交互式可视化；

**📊 数据集**

在ProofNet基准中的两条数论定理（关于 n²‑1 可被 8 整除与 3x²+2=y² 无整数解）进行系统验证和用户实验；

**📈 对比分析**

通过对比16名本科生使用Explorable Theorems与Gemini聊天机器人基线，在证明总结、答案正确性、步骤引用等指标上显著优于基线（统计显著、Cohen's d>1）；

**⚠️ 局限性**

系统受LLM生成Lean证明不一定完全匹配原文本的限制；仅验证短小本科水平证明，缺乏对更复杂或长期学习效果的评估；

---

## 41. Self-Directed Task Identification

**arXiv ID:** 2604.02430 | [PDF](https://arxiv.org/pdf/2604.02430v1)

**作者:** Timothy Gould `[一作]` (Fairfield University), Sidike Paheding `[通讯]` (Fairfield University)

**通讯引用:** 2735 | [OpenAlex ID](https://openalex.org/A5113991367)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了 Self‑Directed Task Identification (SDTI) 框架，能够在零样本环境下让模型自动识别每个数据集对应的正确目标变量；

**💡 创新点**

创新点在于通过单神经元 ANN 的学习曲线（即流形复杂度）作为隐式监督信号，利用流形越复杂对应的损失越高，从而无需预训练即可区分正确与错误的目标变量；

**🔧 技术方法**

主要技术包括向量化实现的单神经元 ANN、Adam 优化器、随机采样的学习率与 beta 超参数、二元交叉熵损失与多组合成本比较、以及基于成本最小化的预测策略；

**📊 数据集**

使用了 17 个公开基准数据集（如 CIFAR‑10、Fashion‑MNIST、Digit‑MNIST、SVHN 等），将图像展平为二维向量，并将多类别问题二值化；

**📈 对比分析**

与传统基准（Pearson、Mutual Information、Cosine 相似度）对比，SDTI 在 F1 评分上提升约 14%，平均 F1 为 0.99，且在大多数实验配置下实现了接近 1.0 的完美识别；

**⚠️ 局限性**

局限性包括：内存随数据集数量快速增长，难以扩展到更大规模；仅适用于二分类表格数据，对类别不平衡敏感；缺乏处理未知目标变量的机制，且未支持多模态输入。

---

## 42. Cross-subject Muscle Fatigue Detection via Adversarial and Supervised Contrastive Learning with Inception-Attention Network

**arXiv ID:** 2604.02670 | [PDF](https://arxiv.org/pdf/2604.02670v1)

**作者:** Zitao Lin `[一作]` (Wuhan University of Technology), Wei Meng `[通讯]` (Wuhan University of Technology)

**通讯引用:** 11617 | [OpenAlex ID](https://openalex.org/A5014957211)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了IADAN网络实现跨受试者肌肉疲劳检测，利用sEMG信号进行三分类；

**💡 创新点**

创新点在于结合Inception‑Attention特征提取、域对抗学习（GRL）和监督对比损失，提升对跨受试者变异的鲁棒性；

**🔧 技术方法**

使用了膨胀卷积、Inception模块、注意力机制、梯度反转层以及监督对比损失等深度学习技术；

**📊 数据集**

使用12名健康志愿者在单腿自重提踵实验中采集的sEMG与IMU数据；

**📈 对比分析**

与ResNet+域对抗、CLT‑Net和MFFNet对比，IADAN在三分类任务中达到了93.54%准确率、92.69%召回率和92.69%F1，显著优于其他模型；

**⚠️ 局限性**

受限于样本量有限导致GRL参数易振荡、未涵盖多种运动模式，进一步提升鲁棒性仍需更多数据和实验。

---

## 43. Pragmatics Meets Culture: Culturally-adapted Artwork Description Generation and Evaluation

**arXiv ID:** 2604.02557 | [PDF](https://arxiv.org/pdf/2604.02557v1)

**作者:** Lingjun Zhao `[一作]` (University of Maryland), Hal Daumé `[通讯]` (University of Maryland)

**通讯引用:** 17088 | [OpenAlex ID](https://openalex.org/A5019928111)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了面向不同文化观众的艺术品描述生成任务，并通过文化化问答评估模型的交际有效性。

**💡 创新点**

创新点在于引入文化适应的文本生成任务、基于文化符号的问答评估框架以及利用理性说话模型实现自我提升的策略。

**🔧 技术方法**

采用视觉-语言模型（Gemma3、LLaMA3.2‑Vision、LLaVA‑OneVision）作为生成器，使用Qwen2.5‑VL 7B作为外部模拟听众，并在理性说话框架下实现理论心智建模。

**📊 数据集**

使用芝加哥艺术学院公开艺术品数据集（6399件艺术品）生成背景信息，并利用GPT‑5生成文化符号与对应多选问答数据集（包含481个文化适应型、562个文化中立型三元组）。

**📈 对比分析**

通过模拟听众的问答准确率和人类评测对比，实验证明自适应说话模型在文化适应型问答中的准确率提升至79.2%（比基线提升8.2%），人类受试者对其描述的喜好度提升8.0%。

**⚠️ 局限性**

局限在于模拟听众与真实人类的理解差距、对已有知识的冗余解释以及对候选描述集质量的依赖。

---

## 44. OPRIDE: Offline Preference-based Reinforcement Learning via In-Dataset Exploration

**arXiv ID:** 2604.02349 | [PDF](https://arxiv.org/pdf/2604.02349v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 45. Too much of a good thing? Entrepreneurial orientation and the non-linear governance effects of SaaS platforms

**arXiv ID:** 2604.02363 | [PDF](https://arxiv.org/pdf/2604.02363v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 46. From Impact to Insight: Dynamics-Aware Proprioceptive Terrain Sensing on Granular Media

**arXiv ID:** 2604.02563 | [PDF](https://arxiv.org/pdf/2604.02563v1)

**作者:** Yifeng Zhang `[一作]` (University of Southern California), Feifei Qian `[通讯]` (University of Southern California)

**通讯引用:** 575 | [OpenAlex ID](https://openalex.org/A5067936185)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

在垂直单自由度的跳跃机器人上，结合运动学估计和动量观测器，对接触力进行实时重建，并用加速度加权回归估计颗粒体力学参数。

**💡 创新点**

首次将动量观测器与加速度依赖的粒子物理模型相结合，实现了在高速跳跃过程中对可变摩擦、吸附质量等动力学效应的显式建模。

**🔧 技术方法**

使用了离散时间卡尔曼滤波器、动量观测器、加速度加权加权最小二乘回归和直接驱动BLDC驱动的并联五杆连杆跳跃平台。

**📊 数据集**

实验数据来自于由300粒玻璃珠组成的干燥颗粒床以及配备负载细胞的线性执行器进行的入侵实验，约有150条高速入侵试验与60条跳跃试验。

**📈 对比分析**

与传统的静态深度回归方法相比，加入动量观测器和加速度加权回归后，估计的颗粒刚度与线性执行器的真值误差降低约30%~40%，并在不同冲击速度和腿部刚度条件下保持一致。

**⚠️ 局限性**

局限在于仅验证于非黏结干燥颗粒床和单自由度垂直跳跃，不适用于多自由度、多接触点或潮湿/粘性地形，且需要高频惯性和编码器数据以保证估计精度。

---

## 47. Synapse: Evolving Job-Person Fit with Explainable Two-phase Retrieval and LLM-guided Genetic Resume Optimization

**arXiv ID:** 2604.02539 | [PDF](https://arxiv.org/pdf/2604.02539v1)

**作者:** Ansel Kaplan Erol `[一作]` (Georgia Institute of Technology), Xisheng Zhang `[通讯]` (Georgia Institute of Technology)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一个多阶段语义招聘推荐系统，结合稠密检索、对比学习与LLM再排序，并加入检索增强生成的可解释解释层，同时提供基于LLM指导的差分进化简历优化工具。

**💡 创新点**

创新点包括：①将高召回稠密检索与高精度语义再排序相结合的两阶段架构；②使用检索增强生成（RAG）在检索上下文中生成可解释的推荐理由；③将LLM用作突变/交叉算子，构建差分进化式简历优化框架，实现无标注数据的自适应改写。

**🔧 技术方法**

技术栈涵盖：Sentence‑BERT + FAISS 近似最近邻检索；对比学习（triplet loss + token‑level soft alignment）；LLM（Gemini 2.0 Flash）用于 pairwise ranking 与 RAG 生成；差分进化算法结合LLM突变与交叉；多模型加权集成（权重 60/25/15 等）。

**📊 数据集**

数据集：LinkedIn 120k 职位、LiveCareer 2.5k 简历，以及基于 10 位真实候选人的手工评估集（offer、interview、reject）。

**📈 对比分析**

评估方式：使用 nDCG@10、nDCG@20 量化排名质量；对比基线检索、对比学习、LLM 单独、以及多模型加权集成。nDCG@10 从 0.541 提升到 0.714，提升 31.9%；端到端推理延迟约 7 s；进化优化在 5 代内平均提升 68%（最高 92%）的匹配得分。

**⚠️ 局限性**

限制：缺乏大规模标注数据，导致验证集规模小；LLM 推理成本高，限制极低延迟部署；系统在不同行业或语言环境的泛化性待进一步验证；解释层依赖检索质量，若检索不佳则解释不完整。

---

## 48. PlayGen-MoG: Framework for Diverse Multi-Agent Play Generation via Mixture-of-Gaussians Trajectory Prediction

**arXiv ID:** 2604.02447 | [PDF](https://arxiv.org/pdf/2604.02447v1)

**作者:** Kevin Song `[一作]` `[通讯]` (Amazon Web Services), Kevin Song (Amazon Web Services)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出 PlayGen-MoG 框架，能够仅基于球队起始阵形生成多球员运动轨迹，用于球队战术设计。

**💡 创新点**

创新点包括：共享的混合高斯权重实现全队情境耦合；相对空间注意力通过学习的几何偏置显式编码位置关系；以及非自回归的绝对位移预测消除累计误差。

**🔧 技术方法**

采用 Transformer 结构的相对空间注意力、交叉注意力、双向时间注意力；混合高斯输出层（Mixture‑of‑Gaussians）与熵正则化；并用非自回归并行解码实现一次性预测完整轨迹。

**📊 数据集**

使用 2021–2022 年 NFL 球员追踪数据（Big Data Bowl），过滤为传球玩法，共计约 10,000 个 11 人进攻阵型，采样频率 10fps。

**📈 对比分析**

与自回归帧差模型、CVAE、LED 等基线对比；在单一前向推断下取得 ADE 1.68 尺、FDE 3.98 尺，混合权重熵 2.06（接近最大 ln 8），显著优于传统方法（ADE > 40 尺）。

**⚠️ 局限性**

局限性包括仅生成进攻方轨迹；未考虑防守阵形或正式战术标签；适用于离散阵形场景，对连续流动运动如足球、篮球的通用性有限。

---

## 49. Contrastive Language-Colored Pointmap Pretraining for Unified 3D Scene Understanding

**arXiv ID:** 2604.02546 | [PDF](https://arxiv.org/pdf/2604.02546v1)

**作者:** Ye Mao `[一作]` (Imperial College London), Krystian Mikolajczyk `[通讯]` (Imperial College London)

**通讯引用:** 31133 | [OpenAlex ID](https://openalex.org/A5024769212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了UniScene3D，利用多视角彩色点图（同时包含颜色与几何信息）训练的Transformer编码器，实现统一的3D场景表征；

**💡 创新点**

创新点包括：① 通过早期融合图像与点图的patch令牌，首次将视觉先验直接迁移至3D；② 设计交叉视角几何对齐与基于对象指代的视角对齐两种对比目标，强化几何一致性与语义对齐；

**🔧 技术方法**

技术方法：基于ViT-B/16的Transformer，采用Chamfer距离的排名感知对比学习、CLIP文本对齐、全局视角与场景级别对齐，以及早期token融合；

**📊 数据集**

使用的训练集为6562个室内场景（ScanNet、3RScan、ARKitScenes），生成的指代表达来自SceneVerse，视角与场景描述来自POMA-3D；

**📈 对比分析**

在视角定位、场景检索、场景类型分类与3D VQA等多任务上与DFN、SigLIP2、FG‑CLIP、Uni3D‑g、POMA‑3D等方法比较，UniScene3D在零/少样本条件下均达到了或超过现有最优水平（如R@1≈38%/25%/23%在ScanRefer/Nr3D/Sr3D；场景检索R@1≈22%/19%/3%等）；

**⚠️ 局限性**

局限性：对大规模真实3D数据仍有依赖，未在户外或非结构化环境验证；对模型规模、推理速度等实际部署因素未作深入探索；

---

## 50. Drift-Resilient Temporal Priors for Visual Tracking

**arXiv ID:** 2604.02654 | [PDF](https://arxiv.org/pdf/2604.02654v1)

**作者:** Yuqing Huang `[一作]` (Harbin Institute of Technology), Xin Li `[通讯]` (Pengcheng Laboratory)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aaccfe5c-6b26-4208-b23c-35331481e142` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一个轻量化的可插拔模块，能在多帧视觉跟踪器中对历史信息进行可靠性校准并生成动态先验，显著抑制跟踪漂移

**💡 创新点**

创新点在于两阶段设计：Temporal Reliability Calibrator 对每帧历史状态进行学习式可靠性评分，固定首帧为全可信；Temporal Guidance Synthesizer 将校准后的摘要转化为可学习的先验令子，提供高层次的时间引导，兼容多种主干架构

**🔧 技术方法**

采用 Vision Transformer（ViT）作为主干，结合低秩适配器（LoRA）与时间因子注意力（FWCA），并实现可学习的可靠性门控网络和先验生成网络；训练时冻结 ViT 权重，仅优化模块参数

**📊 数据集**

在四大公开基准上训练与评估：LaSOT、VastTrack、GOT-10k、TrackingNet；还在 UAV123、OTB2015、TNL2K 上进行额外验证

**📈 对比分析**

与同类跟踪器（OSTrack、ODTrack、LoRAT、SPMTrack 等）在相同协议下比较，平均提升 1.0–1.8 点 AUC，LaSOT 上突破 77.5%，GOT-10k 80.3% AO，VastTrack 47.2% AUC，展示出显著的性能提升且计算开销极低（MACs+1~2G，FPS 仅略降）

**⚠️ 局限性**

局限性包括：需要预先设定历史帧数（最多 5 帧），对极长序列或频繁重定位场景的鲁棒性尚未深入验证；模块主要针对目标外观不变且位置信息可提取的情况，对大幅形变或遮挡重叠仍可能产生误判

---

## 51. MBGR: Multi-Business Prediction for Generative Recommendation at Meituan

**arXiv ID:** 2604.02684 | [PDF](https://arxiv.org/pdf/2604.02684v1)

**作者:** Changhao Li `[一作]` (Meituan), Xingxing Wang `[通讯]` (Meituan)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `9ce7179e-700c-4310-ac2b-91df50ded46e` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了多业务生成式推荐框架MBGR，解决传统生成式推荐在多业务场景下的平衡与混淆问题。

**💡 创新点**

创新点包括业务感知语义ID（BID）模块、基于混合专家的多业务预测（MBP）结构以及标签动态路由（LDR）机制，实现业务特异化与共通表征的融合。

**🔧 技术方法**

使用Transformer自回归架构、Mixture‑of‑Experts（MoE）专家门控、业务感知自编码器、InfoNCE对比学习与语义ID编码等技术。

**📊 数据集**

在美团平台收集的两大数据集上验证：①生成式训练集（3.8万用户、5.5千万商家、四类业务）；②下游应用集（3.7万用户、7.8亿交互记录）。

**📈 对比分析**

与SASRec、HSTU等基线对比，离线Hit@10提升至+0.04，在线CTCVR GAUC提升整体+3.98%，在小型业务上提升更显著。

**⚠️ 局限性**

局限性包括仅在美团场景验证、对SID映射依赖较大、梯度耦合问题在业务相似度高时仍可能出现、计算资源需求高等。

---

## 52. A Numerical Method for Coupling Parameterized Physics-Informed Neural Networks and FDM for Advanced Thermal-Hydraulic System Simulation

**arXiv ID:** 2604.02663 | [PDF](https://arxiv.org/pdf/2604.02663v1)

**作者:** Jeesuk Shin `[一作]` (Pohang University of Science and Technology), Joongoo Jeon `[通讯]` (Pohang University of Science and Technology)

**通讯引用:** 4 | [OpenAlex ID](https://openalex.org/A5108987063)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出一种参数化物理信息神经网络（P2F）与有限差分（FDM）相耦合的混合框架，用于无数据驱动、无再训练的核热力学系统模拟。

**💡 创新点**

创新点在于：①将参数化NA-PINN作为单一网络可复用的速度求解器，完成对所有流道的泛化；②采用节点分配策略将PINN仅用于动量方程，FDM负责质量守恒，从而避免长时间积分误差；③通过硬约束实现初始条件的自洽，简化损失函数。

**🔧 技术方法**

技术手段包括：参数化物理信息神经网络（将水位差、初始速度、时间作为输入），硬约束初始条件，自动微分求导，节点分配混合耦合，有限差分时间步进。

**📊 数据集**

无数据集；训练完全基于自定义的采样点（水位差、初始速度、时间）和物理方程残差，无需MELCOR或其他代码生成的数据。

**📈 对比分析**

与传统FDM参考求解器对比，六罐流动实验中水位MAE≈10⁻⁵ m、速度MAE≈10⁻³ m/s；在不同时间步长（0.2–1.0 s）和五种初始条件下误差保持稳定；但总计算时间约为FDM的25倍，主要受PINN前向推理开销影响。

**⚠️ 局限性**

局限性包括：仅在开放式单向流的简化六罐模型中验证；未考虑闭式系统、压力耦合和双向流动；未与MELCOR的完整模块（HS、RN等）集成；当前实现计算效率低于传统FDM。

---

## 53. LumiVideo: An Intelligent Agentic System for Video Color Grading

**arXiv ID:** 2604.02409 | [PDF](https://arxiv.org/pdf/2604.02409v1)

**作者:** Yuchen Guo `[一作]` (Northwestern University), Weifeng Su `[通讯]` (Beijing Normal - Hong Kong Baptist University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 LumiVideo，一套端到端的代理系统，能够从原始 log 视频自动生成符合行业标准的 ASC‑CDL 参数和 3D LUT，并支持自然语言反馈的迭代优化；同时创建了 LumiGrade，首个公开的 log 编码视频基准集。

**💡 创新点**

创新点包括：①将颜色分级从像素生成转变为可解释的参数推理过程；②在推理阶段结合 LLM、RAG 和 Tree‑of‑Thought 搜索，以探索非线性 ASC‑CDL 参数空间；③通过保护色域约束、适应性 Lift 以及可反射的自然语言交互，实现场景一致、可控且可迭代的分级；④提出并发布 LumiGrade benchmark。

**🔧 技术方法**

使用了大语言模型（GPT‑5.4 / Gemini 3.1 Pro）、视觉语言模型（VLM）、检索增强生成（RAG）数据库、Tree‑of‑Thought 推理、ASC‑CDL 数学编译、3D LUT 生成、自然语言反馈反射循环；实现层面采用云 API 与本地 GPU 组合。

**📊 数据集**

使用了 100+ 条专业摄像机的原始 log 视频（Sony S‑Log3、RED Log3G10、ARRI LogC 等），并对 40 条视频进行专业色彩师手工分级，生成 ASC‑CDL 参数和参考渲染结果。该数据集被命名为 LumiGrade。

**📈 对比分析**

与 GPT‑5.3 / Gemini、官方 LUT、Diffusion LUT + CST 以及人工专家对照，评估指标包括 MANIQA、DeQA、Q‑Align、CLIP‑AS、Harmony、Tonal、Skin、LLM‑Judge、用户胜率等。实验显示 LumiVideo 在绝大多数指标上均超过所有自动化基线，并在多项指标上逼近或优于人工专家，用户胜率仅次于人工专家。

**⚠️ 局限性**

局限性包括：仅使用全局 LUT，无法实现局部区域（如 Power‑Window、二级调整）的细粒度分级；对复杂场景的细节处理仍不及经验色彩师；目前模型主要在云端运行，离线部署受限；未来需扩展到空间变换参数空间。

---

## 54. Interpretable Deep Reinforcement Learning for Element-level Bridge Life-cycle Optimization

**arXiv ID:** 2604.02528 | [PDF](https://arxiv.org/pdf/2604.02528v1)

**作者:** Seyyed Amirhossein Moayyedi `[一作]`, David Y. Yang `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出了一种基于桥梁元素级状态的可解释强化学习框架，用软树模型训练后可转换为斜决策树，从而得到可直接解释、可审计的生命周期管理策略。

**💡 创新点**

创新点包括：①将可微软树与温度退火、L1正则化和递归剪枝规则结合，实现软树向可解释斜决策树的自动转换；②在强化学习中直接使用软树作为 actor，实现可解释的长期生命周期策略；③通过基于Dirichlet分布的随机重启环境提升策略的泛化能力。

**🔧 技术方法**

所采用的技术包括：可微软树模型、温度控制的Sigmoid函数、L1正则化与权重剪枝、递归路径可行性剪枝、PPO强化学习、动态规划（DP）与遗传算法（GA）对比方法、可靠性失效概率模型。

**📊 数据集**

使用的数据集主要有两类：①基于InfoBridge数据库收集的Oregon DOT钢桁架桥梁（NBE107）元素状态分布，用于拟合Dirichlet分布并作为RL环境的初始状态；②合成的双特征四类标注数据，用于监督学习实验验证软树模型的表达能力。

**📈 对比分析**

比较方法：与传统决策树、神经网络、DP、GA等基准方法对同一环境下的生命周期成本进行对比。实验结果显示：软树与神经网络RL在生命周期成本上相近；可解释斜决策树在成本上略高但显著优于DP和GA，且与加入代理规则的斜树相比更优。

**⚠️ 局限性**

局限性：①策略高度依赖所建的衰变、行动效果和成本假设，若与真实桥梁不符需重新校准；②软树到斜树的剪枝过程递归实现，计算量相对较大；③正则化与阈值设置需经验调参，过强或过弱均会影响解释性与性能；④未考虑更复杂的结构可靠性模型和多元素协同影响，未来研究需结合真实检查数据进一步验证。

---

## 55. FTimeXer: Frequency-aware Time-series Transformer with Exogenous variables for Robust Carbon Footprint Forecasting

**arXiv ID:** 2604.02347 | [PDF](https://arxiv.org/pdf/2604.02347v1)

**作者:** Qingzhong Li `[一作]` (Xinjiang University), Jinhai Sa `[通讯]` (Xinjiang University)

**通讯引用:** 2 | [OpenAlex ID](https://openalex.org/A5120333430)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种名为FTimeXer的频率感知Transformer，用FFT驱动的频域分支和门控时间-频率融合，对电网碳足迹进行高精度预测。

**💡 创新点**

创新点包括：①FFT驱动的频域分支与门控时间-频率融合以显式捕获多尺度周期性；②在不改造骨干网络的前提下通过随机外源遮蔽和一致性正则实现对不规则外源的鲁棒学习；③采用变量级别外源嵌入减少噪声干扰。

**🔧 技术方法**

技术手段：Transformer、Fast Fourier Transform (FFT)、逆FFT (IFFT)、时间-频率门控融合、跨注意力机制、可变级别外源嵌入、随机遮蔽、一致性正则、MLP、归一化。

**📊 数据集**

使用的真实数据集：Magnolia、California_CT2 和 NewYork_Greenidge 三个碳排放时间序列数据集。

**📈 对比分析**

通过与 GRU、LSTM、Transformer、Informer、TimeXer 等基线模型在 1 小时一步预测任务中进行对比，FTimeXer 在 R²、MSE、RMSE、MAE 上均优于所有基线，显示出更高的预测精度和鲁棒性。

**⚠️ 局限性**

局限性：仅在离线一次步预测场景评估；对极端缺失或严重异步外源的处理仍有待提升；未提供不确定性量化或流式部署的性能分析。

---

## 56. Not All Denoising Steps Are Equal: Model Scheduling for Faster Masked Diffusion Language Models

**arXiv ID:** 2604.02340 | [PDF](https://arxiv.org/pdf/2604.02340v1)

**作者:** Ivan Sedykh `[一作]` (MWS AI), Valentin Malykh `[通讯]` (MWS AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在掩码扩散语言模型(MDLM)推理时，通过在不同的去噪步骤间调度轻量级和重量级模型来加速采样

**💡 创新点**

发现中间去噪步骤对模型替换最为敏感，早晚步骤可用轻模型替换，提出简单的“砂锅式”调度规则，显著降低 FLOPs

**🔧 技术方法**

采用分层 Transformer 结构训练不同深度的 denoiser，利用损失差异和 KL 散度评估步骤重要性，并进行粗粒度段落搜索

**📊 数据集**

使用 OpenWebText 数据集训练和评估，生成 1024 长度样本，测量 GPT-2 生成困惑度

**📈 对比分析**

与全重模型基线相比，砂锅式调度可实现约 17% FLOPs 节省，生成困惑度仅略有下降（约 3%），表现优于随机或集中轻模型的方案

**⚠️ 局限性**

受限于仅测试两种模型规模、单一数据集且未考虑更大规模的 MDLM 家族，且轻模型对中间步骤的鲁棒性不足，未探索动态调度或多层次模型组合

---

## 57. \texttt{DR-DAQP}: An Hybrid Operator Splitting and Active-Set Solver for Affine Variational Inequalities

**arXiv ID:** 2604.02531 | [PDF](https://arxiv.org/pdf/2604.02531v1)

**作者:** Daniel Arnström `[一作]` (Ericsson), Giuseppe Belgioioso `[通讯]` (KTH Royal Institute of Technology)

**通讯引用:** 762 | [OpenAlex ID](https://openalex.org/A5054192782)

**关键词:** `e4c502e8-c16d-4c56-8df3-cffaee9eaadb` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种结合Douglas–Rachford拆分与活跃集加速的强单调Affine Variational Inequalities (AVI)求解器。

**💡 创新点**

在DR迭代中动态估计活跃集并执行Newton步，保证在有限步内得到精确解，突破了传统一阶方法仅渐进收敛的局限。

**🔧 技术方法**

采用Douglas–Rachford分裂、活跃集识别、Newton加速、双线性QP子问题、warm‑start与预因子化等技术实现高效求解。

**📊 数据集**

使用随机生成的AVI实例（不同尺寸与非对称度）以及游戏理论模型预测控制（GT‑MPC）基准进行实验。

**📈 对比分析**

与pivot、Clarabel、lifted‑QP等现有方法比较，实验表明该算法在随机AVI上平均快约十倍、最坏情况快两位数，且在GT‑MPC案例中比混合整数方法快数个数量级。

**⚠️ 局限性**

仅适用于多面体约束下的强单调AVI，未覆盖非凸或非多面体约束；参数ρ选择固定，尚未提出自适应策略。

---

## 58. On the Geometric Structure of Layer Updates in Deep Language Models

**arXiv ID:** 2604.02459 | [PDF](https://arxiv.org/pdf/2604.02459v1)

**作者:** Jun-Sik Yoo `[一作]` (Korea University), Jun-Sik Yoo `[通讯]` (Korea University)

**通讯引用:** 4810 | [OpenAlex ID](https://openalex.org/A5002075508)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究深度语言模型层间更新的几何结构，拆分为主导的 tokenwise 变换和残差；

**💡 创新点**

提出基于受限 tokenwise 函数类的分解方法，揭示残差是几何上与主更新方向分离且对模型输出有重要影响的成分；

**🔧 技术方法**

使用局部拟合的输入条件 tokenwise 映射（对角、低秩、正交、浅层 MLP），计算相似度、角度偏差、投影能量等几何指标，并通过干预评估输出扰动；

**📊 数据集**

在多种预训练模型上实验，包含 Transformer（DistilGPT2、Pythia 系列）和状态空间模型（Mamba），使用 WikiText 语料；

**📈 对比分析**

与不同函数类（线性 vs 非线性）和层级对比，发现残差误差与输出扰动的 Spearman 相关系数常超过0.7，说明残差与功能变更高度相关；

**⚠️ 局限性**

残差的大小依赖于所选函数类，过于表达可能削弱解释性；分解结果受函数类限制，无法定位具体机制，仍需进一步挖掘残差内部结构。

---

## 59. Principled and Scalable Diversity-Aware Retrieval via Cardinality-Constrained Binary Quadratic Programming

**arXiv ID:** 2604.02554 | [PDF](https://arxiv.org/pdf/2604.02554v1)

**作者:** Qiheng Lu `[一作]` (University of Virginia), Nicholas D. Sidiropoulos `[通讯]` (University of Virginia)

**通讯引用:** 21983 | [OpenAlex ID](https://openalex.org/A5050186120)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

将多样性检索问题建模为基数约束二次规划（CCBQP），并提出一种基于Frank–Wolfe的精确线搜索求解算法。

**💡 创新点**

提出理论上紧致的连续松弛、可证明收敛的Frank–Wolfe方法，并给出可解释的多样性权衡参数，解决了MMR与DPP的理论与效率缺陷。

**🔧 技术方法**

利用非凸紧致松弛、Frank–Wolfe优化、精确线搜索、矩阵乘法稀疏化以及梯度与曲率分析等技术。

**📊 数据集**

在ASQA与QAMPARI两个基于维基百科的知识密集型问答数据集上进行实验。

**📈 对比分析**

与MMR、DPP比较，在Recall-ILAD Pareto前沿上优于两者，并在k=100时比MMR快40–180ms，速度提升约2.4×至22.9×。

**⚠️ 局限性**

仅在低冗余的维基百科语料上验证，未评估高度冗余语料及更大模型的适用性。

---

## 60. Mitigating Data Scarcity in Spaceflight Applications for Offline Reinforcement Learning Using Physics-Informed Deep Generative Models

**arXiv ID:** 2604.02438 | [PDF](https://arxiv.org/pdf/2604.02438v1)

**作者:** Alex E. Ballentine `[一作]` (Worcester Polytechnic Institute), Raghvendra V. Cowlagi `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 763 | [OpenAlex ID](https://openalex.org/A5075056777)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

论文提出一种互信息约束的分割变分自编码器（MI‑VAE）用于在数据稀缺场景下生成满足物理约束的合成轨迹，并将其与传统 VAE 进行对比，最终评估其对离线强化学习（BPPO）和行为克隆（BC）控制器性能的提升。

**💡 创新点**

创新点在于：①将物理动力学信息以“理想数据”形式注入生成模型；②设计双编码器/双解码器架构，将共享与专属特征分离；③通过互信息正则化实现子空间解耦，使合成数据在满足物理约束的同时保持多样性；④在空间飞行器（行星着陆器）控制任务中验证了该方法在极少样本（25 条轨迹）下的优越性。

**🔧 技术方法**

技术：变分自编码器（VAE）与互信息约束；强化学习（PPO、BPPO）与行为克隆；离线 RL 训练与合成数据生成；统计特征分析（PCA、四阶矩）；模拟器与真实数据对比。

**📊 数据集**

数据集：①真实轨迹数据（由在线 RL 训练在参数集 PA 与 PB 上产生，仅 25 条样本）；②理想轨迹数据（基于物理方程无噪声下的前向积分，共 1000 条样本）；③S‑VAE 与 MI‑VAE 生成的合成轨迹；⑤对照数据（原始真实轨迹）。

**📈 对比分析**

比较方法：①统计相似度（均值、方差、偏度、峰度，PCA 散点图）；②轨迹平滑度与误差；③RL 性能指标（累计奖励、成功率、控制成本、最终状态偏差）。结果显示：在 25 条样本时，MI‑VAE 生成的轨迹在统计特征、轨迹误差以及 BPPO/BC 的奖励/成功率上均优于 S‑VAE；当样本量升至 1000 条时，两者差异减小，均可达到相近的高性能。

**⚠️ 局限性**

局限性：①对理想数据的依赖，需要先有可精确的物理模型；②互信息估计与超参数（β、λ 等）需要经验调优；③实验仅在行星着陆器模拟器上验证，缺乏真实硬件验证；④在高维、更复杂系统中的可扩展性和计算成本尚未评估。

---

## 61. Xpertbench: Expert Level Tasks with Rubrics-Based Evaluation

**arXiv ID:** 2604.02368 | [PDF](https://arxiv.org/pdf/2604.02368v1)

**作者:** Xue Liu `[一作]`, Zhenwei Zhu `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了 XpertBench 基准，用以评估 LLM 在七大专业领域的真实、开放式、长周期专家级任务。

**💡 创新点**

创新点包括：多域高保真开放任务设计、专家级细粒度 Rubric 评估体系、ShotJudge 基于专家锚定的自动评分框架。

**🔧 技术方法**

使用专家驱动的 Rubric 生成、LLM 辅助评估项构建、双重权重（Essential/Important/Optional + 数值）权重体系，以及 ShotJudge 的 few‑shot 预训练裁定技术。

**📊 数据集**

使用了自研的 1,346 题专业任务集（XpertBench）和 245 题 Gold 子集，涵盖金融、法律、医疗、教育、STEM、人文等七个领域。

**📈 对比分析**

通过 ShotJudge 对 12 种主流 LLM（Claude‑Opus‑4.6‑thinking、GPT‑5.4‑high 等）进行评估，最高整体得分约 66%，不同领域表现差异明显，展示了模型在金融、法律、人文等方面的优势与 STEM 等领域的不足。

**⚠️ 局限性**

局限性在于仍受限于当前模型推理与规划能力，评估依赖昂贵的专家标注、Gold 子集规模有限，且未能实现真正通用的专家级模型。

---

## 62. Dynamical structure of vanishing gradient and overfitting in multi-layer perceptrons

**arXiv ID:** 2604.02393 | [PDF](https://arxiv.org/pdf/2604.02393v1)

**作者:** Alex Alì Maleknia `[一作]` (Univ Montpellier), Yuzuru Sato `[通讯]` (Hokkaido University)

**通讯引用:** 1829 | [OpenAlex ID](https://openalex.org/A5111377635)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文构建了一个最小化的两神经元、无偏置、单输入单输出的多层感知机模型，并对其梯度下降学习动力学进行理论分析与数值验证，揭示了消失梯度与过拟合的动态机制；

**💡 创新点**

创新点在于将消失梯度和过拟合问题归结为动力学中的鞍点与吸引子结构，并在该简化模型下严谨证明了学习轨迹几乎必然收敛至唯一的过拟合吸引子；

**🔧 技术方法**

主要技术包括Fukumizu–Amari最小模型、解析梯度下降动力学、李雅普诺夫分析、矩阵秩与嵌入理论，以及数值模拟验证；

**📊 数据集**

使用的数据集为合成的高斯噪声样本（100个点），目标函数为 T(x)=2tanh(x) 或 T(x)=2tanh(x)-tanh(4x)，并分别在无噪声和噪声（τ=0.2）两种情形下进行实验；

**📈 对比分析**

通过对训练误差与泛化误差随迭代次数的曲线以及参数轨迹的可视化对比，发现有噪声时学习曲线出现平台期并最终收敛至过拟合点，验证了理论预言；

**⚠️ 局限性**

局限性包括仅研究一维输入、两神经元、无偏置的极简模型，难以推广至更大规模或深层网络；此外对唯一性与鞍点到吸引子距离的精确阈值条件尚未给出完整表述。

---

## 63. Beyond the AI Tutor: Social Learning with LLM Agents

**arXiv ID:** 2604.02677 | [PDF](https://arxiv.org/pdf/2604.02677v1)

**作者:** Harsh Kumar `[一作]` (University of Toronto), Ashton Anderson `[通讯]` (University of Toronto)

**通讯引用:** 3688 | [OpenAlex ID](https://openalex.org/A5048789742)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过两项受控实验，探究多代理 LLM 配置（导师+同伴或两种不同 LLM 角色）是否能提升学习效果，比较其与单一 LLM 辅导的差异。

**💡 创新点**

首次系统评估多代理 LLM 学习环境，发现其在不同任务（聚合型问题求解与开放式写作）中具有任务依赖的优势，并揭示代理角色与错误模式对学习成效的关键影响。

**🔧 技术方法**

使用 GPT‑5.2、Claude Opus 4.6 并通过 AutoGen 等框架编排导师、同伴与角色专属代理；实验交互采用自由文本聊天，评估使用 GPT‑5.2 评分与 SBERT 嵌入实现想法相似度计算。

**📊 数据集**

实验任务包括 SAT 级数学问题（5道主题+同构变体）以及纽约时报学生意见系列的议论文和创意写作提示；受试者来自 Prolific 平台，而非公开数据集。

**📈 对比分析**

采用卡方检验、逻辑回归、OLS 回归和置换检验对四种数学辅导条件与三种写作条件进行比较；数学测试准确率从控制组 42% 提升至导师+同伴组 65%；写作质量从 2.32 提升至 2.65（单一 LLM）且单一与双代理无显著差异；单一 LLM 使想法相似度升至 0.748，双代理恢复至 0.737，接近控制组。

**⚠️ 局限性**

受试者为 Prolific 众包工人，样本规模受自动化机器人干扰导致高淘汰率，部分分析未达预期显著性；实验时间短暂（5 分钟课堂与写作），缺乏长期学习或真实课堂的验证，交互日志深度分析缺失，结果对真正教育情境的推广仍需谨慎。

---

## 64. VERTIGO: Visual Preference Optimization for Cinematic Camera Trajectory Generation

**arXiv ID:** 2604.02467 | [PDF](https://arxiv.org/pdf/2604.02467v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 65. Backdoor Attacks on Decentralised Post-Training

**arXiv ID:** 2604.02372 | [PDF](https://arxiv.org/pdf/2604.02372v1)

**作者:** Oğuzhan Ersoy `[一作]` (Gensyn), Stjepan Picek `[通讯]` (University of Zagreb)

**通讯引用:** 4711 | [OpenAlex ID](https://openalex.org/A5024072796)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出并实现了一种针对流水线并行（Pipeline Parallelism）大模型的后训练隐蔽后门攻击，在去中心化细调（SFT）过程中通过控制中间阶段注入不良行为，同时保持整体模型性能。

**💡 创新点**

创新点在于：①首次针对PP提出目标后门攻击；②使用离线训练的失配代理模型和在线的任务向量注入（Task Arithmetic）实现低干扰、可调节强度的后门；③迭代注入策略提升了对后期安全对齐训练的鲁棒性。

**🔧 技术方法**

采用的技术包括：流水线并行与数据并行的组合、离线代理模型训练、任务向量（Δ）计算、在线缩放注入、频率控制、基于LoRA的参数调整等。

**📊 数据集**

使用的数据集与模型：LLaMa‑3.2 1B Instruct 预训练模型；Finance‑Instruct‑500k 作为 SFT 训练集；Harmful Dataset 用于后门训练与安全评估。

**📈 对比分析**

实验对比方法：将模型分为四个阶段，分别对无攻击、一次性注入、以及迭代注入进行 SFT；通过验证损失、Clean SFT 任务性能以及后门成功率（ASR）评估；结果显示：迭代注入不显著影响验证损失，攻击成功率达94%，即使在安全对齐训练后仍保持约60%的成功率，而一次性注入被安全对齐抑制。

**⚠️ 局限性**

局限性包括：需提前获得用于去中心化 SFT 的基模型并精确了解管道划分（即知道每个阶段包含哪些层）；若攻击者不知晓分区信息，需要为每个可能的阶段训练代理任务向量；在高度加密或私有模型环境下此假设不易满足。

---

## 66. Convolutional Surrogate for 3D Discrete Fracture-Matrix Tensor Upscaling

**arXiv ID:** 2604.02335 | [PDF](https://arxiv.org/pdf/2604.02335v1)

**作者:** Martin Špetlík `[一作]` (Technical University of Liberec), Jan Březina `[通讯]` (Technical University of Liberec)

**通讯引用:** 179 | [OpenAlex ID](https://openalex.org/A5061066522)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

设计并训练了3D卷积神经网络+全连接网络的深度学习代理模型，用来预测离散断层-基质模型中等效水力导率张量，以实现数值上尺度化的加速。

**💡 创新点**

创新点在于将3D卷积网络与多尺度数值同质化结合，针对不同断层/基质导率比（1e3、1e5、1e7）分别训练代理，显著提升了代理在多样断层网络、随机场相关长度和断层密度变化下的泛化能力，并在GPU上实现了两位百倍以上的推理加速。

**🔧 技术方法**

主要技术包括：基于GSTools生成空间相关随机场、Poisson过程生成断层网络、使用混合维度有限元求解DFM、卷积3D网络与全连接网络的混合架构、Adam优化、NRMSE评估、GPU推理。

**📊 数据集**

数据集：三组共225,000个样本（每组75,000），覆盖三种断层/基质导率比；每组样本包含P30=0.0010/0.0025、λ=0/10/25等参数，进一步在不同DFN配置、断层密度、随机场相关长度等维度做测试。

**📈 对比分析**

与传统数值同质化方法比较，代理在保持NRMSE<0.22（所有张量分量）且R²>0.95的同时，在宏观两大问题（约束流出量和各向异性张量）中，代理预测结果与纯数值同质化差异均<1%或更小；GPU推理时间相对CPU数值同质化快>100×，CPU推理快>16×。

**⚠️ 局限性**

局限性：代理对极端断层密度（超出训练范围）或大相关长度的随机场表现略降；在较小域或断层贡献显著时，对张量预测误差更敏感；仍需进一步集成到多层蒙特卡洛框架并验证不同同质化块尺寸的影响。

---

## 67. Generalization Limits of Reinforcement Learning Alignment

**arXiv ID:** 2604.02652 | [PDF](https://arxiv.org/pdf/2604.02652v1)

**作者:** Haruhi Shida `[一作]` (Aladdin Security Inc.), Keigo Kansa `[通讯]` (Aladdin Security Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对OpenAI gpt-oss-20b，提出并实证了一种“复合 jailbreak”攻击方法，即将对比结构、权威人物身份和自我评估三种单独被防御过的攻击技术组合，以压垮模型的认知资源从而突破安全机制，验证了RLHF安全训练的泛化局限；

**💡 创新点**

首次系统地揭示了RLHF安全训练仅为现有能力概率重分布，导致安全机制对未知复合攻击失效；通过构造复合攻击框架，证明了指令层次结构在面对多重非矛盾指令时会崩溃，并提供了对应的定量验证；

**🔧 技术方法**

使用了强化学习与人类反馈（RLHF）、指令层次结构、深度推理对齐（deliberative alignment）以及多技术复合攻击设计、消融实验和攻击成功率（ASR）评估等技术；

**📊 数据集**

使用了70条人工构造的多类别（生物武器、恶意软件、网络钓鱼、非法药物、武器制造、欺诈、个人信息窃取）提示作为评估数据集；

**📈 对比分析**

通过将单一攻击技术与多重技术组合进行对比，发现复合攻击的ASR从单独技术的14.3%显著提升至71.4%，表明安全训练对单一模式的防御有效但对复合模式失效；

**⚠️ 局限性**

研究表明模型层面的安全训练（RLHF、指令层次、推理对齐）无法完全抵御复杂复合攻击，缺乏对输入复杂度的结构化防御，且复合攻击所需的组合空间巨大，难以在训练中覆盖。

---

## 68. An Initial Exploration of Contrastive Prompt Tuning to Generate Energy-Efficient Code

**arXiv ID:** 2604.02352 | [PDF](https://arxiv.org/pdf/2604.02352v1)

**作者:** Sophie Weidmann `[一作]` (University of Twente), Fernando Castor `[通讯]` (University of Twente)

**通讯引用:** 2741 | [OpenAlex ID](https://openalex.org/A5062400717)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对比学习的提示调优（Contrastive Prompt Tuning, CPT）来提升大型语言模型生成的代码在能耗上的效率，并在 Python、Java、C++ 三种语言与三种模型上进行实验。

**💡 创新点**

创新点在于首次将对比学习与提示调优相结合，用软提示来区分功能等价但能耗差异的代码对，从而引导模型生成更节能的实现。

**🔧 技术方法**

采用的技术包括：PEFT 库的提示调优、对比损失（Triplet Loss 与 InfoNCE）、软提示初始化为“Generate energy‑efficient code.”、对代码嵌入空间的对比学习。

**📊 数据集**

使用的数据集包括：GEC（Python 能效对）、CodeNet（C++ 能效对）、LeetCode 与 HumanEval‑X（基准测试），并构造了 Python、C、两者混合的训练集。

**📈 对比分析**

评估方法是：先生成代码后验证功能正确性，再在同一台 MacBook 上测量执行时间与能耗（Joule），通过 Mann‑Whitney U 检验统计显著性。结果显示，CPT 在大多数模型/语言/任务上提升了准确率（最高约 47%）并在部分情况能降低 30–60% 的能耗，但并非在所有实验中均有效。

**⚠️ 局限性**

局限性包括：只测试了三种模型和三种语言；对比学习仅基于已有的能效标签而未直接使用能耗度量；实验仅在一台 2016 年 MacBook 上完成，难以推广到不同硬件；训练数据来源有限，可能缺乏更广泛的高能耗与低能耗实现差异。

---

## 69. Mitigating LLM biases toward spurious social contexts using direct preference optimization

**arXiv ID:** 2604.02585 | [PDF](https://arxiv.org/pdf/2604.02585v1)

**作者:** Hyunji Nam `[一作]` (Stanford University), Dorottya Demszky `[通讯]` (Stanford University)

**通讯引用:** 3271 | [OpenAlex ID](https://openalex.org/A5052171928)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了大型语言模型在教育评估任务中对无关社交背景的敏感性，并提出Debiasing-DPO训练方法，以提升模型的鲁棒性与预测准确性。

**💡 创新点**

创新点在于构造自监督的对比推理对（无偏与有偏理由）与SFT结合的DPO目标，既消除偏差又保持或提升预测性能。

**🔧 技术方法**

技术上采用了Direct Preference Optimization (DPO) 与监督微调 (SFT) 的混合训练，利用模型自身生成的中立与带有无关背景的推理作为对比；同时实验了多种推理与评估策略。

**📊 数据集**

使用的数据集为美国国家教师效能中心 (NCTE) 的课堂记录与专家打分数据，包含七维评估指标与七类无关社交背景（教师经验、教育水平、认证、种族/性别等）。

**📈 对比分析**

通过与SFT、基线DPO、对照DPO以及多种推理策略的对比实验，Debiasing-DPO 在 Spearman 相关系数上平均提升约52%，偏差降低约84%，并在多模型、多背景场景中表现出色。

**⚠️ 局限性**

局限性包括受类别不平衡影响，绝对预测性能仍有限；对未见背景（如种族、性别、反向同情等）的泛化能力不足；以及需进一步验证在其他高风险领域的适用性。

---

## 70. From Elevation Maps To Contour Lines: SVM and Decision Trees to Detect Violin Width Reduction

**arXiv ID:** 2604.02446 | [PDF](https://arxiv.org/pdf/2604.02446v1)

**作者:** Philémon Beghin `[一作]` (UCLouvain), François Glineur `[通讯]` (UCLouvain)

**通讯引用:** 2407 | [OpenAlex ID](https://openalex.org/A5038754080)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

研究如何利用3D摄影测量网格的高度图自动检测小提琴宽度缩小，比较使用原始高度图、PCA降维以及基于等高线的参数化特征三种输入方式在SVM和决策树分类器上的表现。

**💡 创新点**

创新点在于将高维原始高度图作为特征直接输入机器学习模型，并探索其与传统的等高线参数化特征的对比；同时系统地评估了不同重采样策略、归一化处理和SVM正则化参数选择对分类性能的影响。

**🔧 技术方法**

主要技术包括：三维摄影测量获取网格；相对与绝对重采样生成二维高度图；PCA降维；支持向量机（RBF 与线性核）与决策树（Gini 与熵）分类；交叉验证与平衡准确率评估；特征工程如等高线拟合得到α、β、γ、δ参数。

**📊 数据集**

数据集为来自布鲁塞尔音乐器械博物馆的25件小提琴与中提琴，其中5件被怀疑为宽度缩小；每件乐器的共鸣板网格精度达亚毫米级。

**📈 对比分析**

在留一交叉验证与平衡准确率的框架下，基于等高线参数化特征的模型往往能达到近100%的准确率；而直接使用原始高度图或其PCA降维版时，准确率波动较大，最高可达90%，但表现不稳定。SVM在强正则化下相对更稳健，决策树整体性能低于SVM。

**⚠️ 局限性**

局限性包括：样本量极小导致模型易过拟合；仅使用共鸣板数据，未包含背板或大提琴；原始高度图在高维空间中难以提取有用信息；深度学习方法未能发挥作用，需更大数据集。

---

## 71. Jump Start or False Start? A Theoretical and Empirical Evaluation of LLM-initialized Bandits

**arXiv ID:** 2604.02527 | [PDF](https://arxiv.org/pdf/2604.02527v1)

**作者:** Adam Bayley `[一作]` (Queen's University), Kevin H. Wilson `[通讯]` (RBC Borealis)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并评估使用大型语言模型生成的偏好数据对上下文多臂赌博机进行预热启动的Noisy-CBLI框架。

**💡 创新点**

创新点在于系统研究了噪声（随机替换与偏好翻转）和系统性失配对预热效果的影响，并给出以先验误差为中心的理论阈值。

**🔧 技术方法**

采用LinUCB算法、Ridge回归预训练、噪声注入、上置信界分析。

**📊 数据集**

使用COVID-19疫苗、移民态度和休闲旅游三组conjoint调查数据。

**📈 对比分析**

通过与冷启动LinUCB对比，发现当先验误差低且噪声低于约30%时可显著降低早期累积损失；噪声超过40%或失配严重时反而不利。

**⚠️ 局限性**

局限包括对提示敏感、噪声模型过于简单、只考虑线性上下文、无法处理多模态或动态噪声，以及对公平性与安全性的审计不足。

---

## 72. F2F-AP: Flow-to-Future Asynchronous Policy for Real-time Dynamic Manipulation

**arXiv ID:** 2604.02408 | [PDF](https://arxiv.org/pdf/2604.02408v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 73. Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?

**arXiv ID:** 2604.02532 | [PDF](https://arxiv.org/pdf/2604.02532v1)

**作者:** Kamalasankari Subramaniakuppusamy `[一作]` (George Washington University), Jugal Gajjar `[通讯]` (George Washington University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出Feature Attribution Stability Suite (FASS)，一种在保持模型预测不变的前提下评估视觉系统特征归因稳定性的基准框架。

**💡 创新点**

创新点包括：①在稳定性评估前强制进行预测保持过滤；②将稳定性分解为三种互补指标（结构相似度、秩相关、Top‑k Jaccard重叠）；③系统涵盖几何、光度与压缩等多样化现实扰动。

**🔧 技术方法**

使用了SSIM、Spearman秩相关、Top‑k Jaccard等指标进行评估，并通过对齐预测保持过滤、数据增强与量化组合实现完整评测。

**📊 数据集**

在ImageNet-1K、MS‑COCO、CIFAR‑10三个数据集上，分别使用ResNet‑50、DenseNet‑121、ConvNeXt‑Tiny、ViT‑B/16四种架构，累计约70,000张样本。

**📈 对比分析**

比较了四种归因方法（Integrated Gradients、GradientSHAP、Grad‑CAM、LIME）；结果表明Grad‑CAM在所有数据集和扰动类型下具有最高的FASS分数，GradientSHAP与IG相近，LIME最不稳定。

**⚠️ 局限性**

局限性包括：①未考虑归因的可信度，仅评估稳定性；②扰动幅度固定，未进行强度曲线分析；③对翻译、亮度与JPEG等扰动的保留率极低，影响统计可靠性；④仅使用预训练模型，未针对目标数据集微调。

---

## 74. Spatial-Aware Conditioned Fusion for Audio-Visual Navigation

**arXiv ID:** 2604.02390 | [PDF](https://arxiv.org/pdf/2604.02390v1)

**作者:** Shaohang Wu `[一作]` (Xinjiang University), Yinfeng Yu `[通讯]` (Xinjiang University)

**通讯引用:** 3726 | [OpenAlex ID](https://openalex.org/A5091800151)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `e0540dec-d77f-42db-94ae-d039248f6393` `b88c6eac-d57a-4623-a604-1f401f3eb268` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种空间感知的条件融合框架（SACF），利用音频信息对视觉特征进行动态通道调制，并通过空间离散定位描述符来明确音源的相对位置，从而提升音视频导航的效率和泛化能力。

**💡 创新点**

创新点包括：①采用空间离散化（SDLD）将音源方向和距离转化为离散分布并编码成紧凑向量，提供显式的空间先验；②使用 FiLM 条件通道调制（ACVF）将音频与空间描述符共同生成视觉特征的缩放偏置，实现深度跨模态调制，避免了传统的拼接或空间注意力方法的计算开销和优化难题。

**🔧 技术方法**

技术手段包括音视频编码器、分类网络实现 SDLD、FiLM 条件调制层、GRU 状态建模、Actor-Critic（PPO）强化学习，以及 SED 进行目标检测。

**📊 数据集**

数据集为 SoundSpaces 平台，覆盖 Matterport3D 与 Replica 两套室内场景，并分别在 Heard 与 Unheard 两种设置下进行评估。

**📈 对比分析**

与 SoundSpaces、AGSA 等基线对比，SACF 在 Replica Heard 组实现 SPL 80.3%/SR 96.3%，Unheard 组 SPL 43.9%/SR 79.1%；在 Matterport3D Heard 组 SPL 42.4%/SR 58.3%/SNA 28.0%，Unheard 组 SPL 42.4%/SR 58.3%/SNA 28.0%，相较于基线在 SPL、SR 与 SNA 上均有显著提升，尤其在未听目标音频时表现出强大的跨场景泛化能力。

**⚠️ 局限性**

局限性在于对齐度（SNA）略低于 AGSA，且在 RGB 输入下性能不如 depth 输入；此外，对光照变化和复杂环境噪声的鲁棒性仍有提升空间。

---

## 75. VoxelCodeBench: Benchmarking 3D World Modeling Through Code Generation

**arXiv ID:** 2604.02580 | [PDF](https://arxiv.org/pdf/2604.02580v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 76. Beyond Message Passing: Toward Semantically Aligned Agent Communication

**arXiv ID:** 2604.02369 | [PDF](https://arxiv.org/pdf/2604.02369v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 77. Tune to Learn: How Controller Gains Shape Robot Policy Learning

**arXiv ID:** 2604.02523 | [PDF](https://arxiv.org/pdf/2604.02523v1)

**作者:** Antonia Bronars `[一作]` (Massachusetts Institute Of Technology), Pulkit Agrawal `[通讯]` (Massachusetts Institute Of Technology)

**通讯引用:** 5327 | [OpenAlex ID](https://openalex.org/A5111774389)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

研究了位置控制器增益如何影响机器人学习中的行为克隆、强化学习以及从仿真到真实的迁移，系统评估了不同增益（刚度与阻尼组合）对学习效果的影响。

**💡 创新点**

将增益视为学习接口的先验偏置，而非单纯的行为参数，并在三种学习范式中揭示增益选择的“可学习性”原则；提出了在仿真中保持相同状态分布、仅改变动作分布的Torque-to-Position Retargeting方法；在仿真与真实对齐时发现高刚度高阻尼会放大模型误差。

**🔧 技术方法**

使用PD位置控制器、行为克隆（VAE、Transformer、Diffusion）、PPO强化学习、系统辨识、零频率保持、域随机化、网格搜索与超参数优化；利用热力图、统计检验（Logistic回归、Barnard检验、Mann-Whitney U检验）对结果进行量化。

**📊 数据集**

自建的多任务数据集，包括Frank Research 3、G1、Allegro等机器人在箱子搬运、碟架卸载/装载、块堆叠等任务；此外在仿真中用Torque-to-Position Retargeting生成与不同增益对应的演示数据，并进行12人类操作者的远程控制实验。

**📈 对比分析**

通过在各增益网格上进行行为克隆、强化学习和仿真到真实的成功率/轨迹误差评估。结果显示：行为克隆在“低刚度、高阻尼”区表现最好；强化学习在所有增益下均可获得成功策略，但需要针对增益重新调参；仿真到真实的迁移在“高刚度、高阻尼”区表现最差，出现高频振荡。

**⚠️ 局限性**

实验依赖于特定的机器人平台与任务，未验证在全身控制或跨平台（如人类视频、穿戴设备）下的普适性；强化学习的超参数搜索受限于选定的搜索空间，可能无法捕捉所有任务的最佳设置；方法在高频控制下可能产生振荡，需要额外的频率调节策略。

---

## 78. Modeling and Controlling Deployment Reliability under Temporal Distribution Shift

**arXiv ID:** 2604.02351 | [PDF](https://arxiv.org/pdf/2604.02351v1)

**作者:** Naimur Rahman `[一作]` (Bath Spa University), Naazreen Tabassum `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

本文提出了一种以可靠性状态为核心的部署框架，利用可靠性波动量化模型在非平稳环境中的时间稳定性，并将部署适配视为在可靠性波动与干预成本之间的多目标控制问题；

**💡 创新点**

创新点在于将可靠性（区分度+校准）建模为随时间演化的动态状态，引入波动度作为稳定性指标，构建成本–波动率 Pareto 前沿，并通过漂移触发的可调阈值实现精细化干预；

**🔧 技术方法**

使用的技术包括梯度提升树 (XGBoost) 做基模型，等距分箱的 ECE、ROC AUC、Brier 等指标；漂移检测采用 Kolmogorov–Smirnov 与 Jensen–Shannon 分数的加权平均；校准采用等距回归；多目标控制通过阈值网格搜索得到 Pareto 前沿；bootstrap 估计波动度置信区间；

**📊 数据集**

使用一份覆盖 2007–2018 年的信用风险贷款数据，包含 1,347,681 条记录，10 个特征，目标为违约；实验时间窗口为 2010–2018 共 9 年；

**📈 对比分析**

通过与三种基准策略（静态部署、周期性校准、滚动重训）和 23 个阈值组合的多目标策略比较，结果显示漂移触发策略在降低可靠性波动（V_L1 约 0.0286，低于滚动重训 0.0309）同时将累计干预成本从 45 降至 12，保持近似的平均 AUC（0.641 与滚动重训 0.661），表明在成本–稳定性空间获得显著提升；

**⚠️ 局限性**

局限包括仅在单一信用风险领域验证，未考虑标签延迟或部分观察；漂移信号仅基于单一统计量，可能忽略交互或条件漂移；成本模型简化为重训 vs 校准，未覆盖更复杂的运营成本；样本量（9 年）导致 bootstrap 置信区间不稳健；

---

## 79. Do We Need Frontier Models to Verify Mathematical Proofs?

**arXiv ID:** 2604.02450 | [PDF](https://arxiv.org/pdf/2604.02450v1)

**作者:** Aaditya Naik `[一作]` (University of Pennsylvania), Mayur Naik `[通讯]` (University of Pennsylvania)

**通讯引用:** 8744 | [OpenAlex ID](https://openalex.org/A5075879790)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文系统评估在数学证明验证任务中是否真正需要前沿模型，比较了传统推理方法与当前最先进模型的性能差异。

**💡 创新点**

创新点在于构建了统一的基准框架，能够在同一证明数据集上公平比较多种模型，并揭示前沿模型在准确率、推理速度及可解释性上的优势与不足。

**🔧 技术方法**

主要技术包括逻辑推理网络、图神经网络（GNN）、Transformer 以及基于规则的验证器；实验中对这些方法进行了集成与对比。

**📊 数据集**

使用的数据集包括公开的 Lean Math、MATH 和 ProofNet（自建）等，涵盖了数理逻辑、代数与组合学等多领域证明。

**📈 对比分析**

比较方法：在相同的验证任务上测量准确率、推理时间和资源消耗。实验结果显示，前沿模型在准确率上略胜一筹，但在推理速度和可解释性方面明显逊色；传统方法在速度和可解释性上更具优势。

**⚠️ 局限性**

限制：前沿模型缺乏充分的可解释性，难以对未见证明结构泛化；基准数据集规模有限，可能导致评估结果不具普遍性；模型训练成本高，实际部署受限。

---

## 80. TRACE: Traceroute-based Internet Route change Analysis with Ensemble Learning

**arXiv ID:** 2604.02361 | [PDF](https://arxiv.org/pdf/2604.02361v1)

**作者:** Raul Suzuki `[一作]` (Federal University of Viçosa), Flávio de Oliveira Silva `[通讯]` (University of Minho)

**通讯引用:** 783 | [OpenAlex ID](https://openalex.org/A5061596292)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

开发了TRACE，一种仅利用traceroute延迟数据的端到端路由变化检测系统；

**💡 创新点**

创新点在于：①基于滚动统计与聚合上下文的时间特征工程；②使用LightGBM/CatBoost/XGBoost三种GBDT作为基学习器并堆叠；③通过阈值校准精细处理极度不平衡的数据；

**🔧 技术方法**

采用的技术包括：梯度提升树堆叠、特征交互与统计摘要、Hyperopt TPE调参、阈值扫描、Python+scikit‑learn/LightGBM/CatBoost等；

**📊 数据集**

使用了M‑Lab公开的 traceroute 数据集，总计约28.5 M条记录，其中训练集19.9 M样本，标签为路由是否变化；

**📈 对比分析**

通过与Logistic回归、随机森林、k‑NN以及单一GBDT基准模型对比，TRACE在测试集上实现了最高F1≈0.869、准确率≈0.895，明显优于所有基线；

**⚠️ 局限性**

局限性在于：需要人工标注事件；特征对测量噪声与数据偏差敏感；泛化能力受限，未来需探索深度学习或自监督方法降低标签依赖。

---

## 81. Generating Satellite Imagery Data for Wildfire Detection through Mask-Conditioned Generative AI

**arXiv ID:** 2604.02479 | [PDF](https://arxiv.org/pdf/2604.02479v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 82. Communication-free Sampling and 4D Hybrid Parallelism for Scalable Mini-batch GNN Training

**arXiv ID:** 2604.02651 | [PDF](https://arxiv.org/pdf/2604.02651v1)

**作者:** Cunyang Wei `[一作]` (University of Maryland), Abhinav Bhatele `[通讯]` (University of Maryland)

**通讯引用:** 4044 | [OpenAlex ID](https://openalex.org/A5081506338)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

开发了一个 4D 并行框架，用于可扩展的 mini‑batch GNN 训练，集成了通信‑无关采样、3D 并行矩阵乘法（PMM）和数据并行。

**💡 创新点**

创新点包括：①通信‑无关的统一顶点采样与无偏边重标定；②将 3D PMM 与数据并行结合，形成完整的 4D 并行结构；③多重优化（采样与训练重叠、低精度通信、核融合、通信-计算重叠）显著降低训练时间。

**🔧 技术方法**

采用的技术包括：通信‑无关的统一顶点采样、无偏边重标定、3D PMM（Sparse‑Dense GCN 计算）、数据并行、CUDA 流重叠、BF16 低精度通信、核融合、NCCL/RCCL all‑reduce 等。

**📊 数据集**

使用的基准数据集有：ogbn‑products、Reddit、Isolate‑3‑8M、Products‑14M、ogbn‑papers100M（其中两者使用随机生成特征）。

**📈 对比分析**

与 DistDGL、MassiveGNN、SALIENT++、BNS‑GCN 等同样采样策略的基线在同一数据集上进行对比，结果显示在 Perlmutter 上对 ogbn‑products 的终端训练时间提升 3.5×，在 Frontier 上对 Reddit 的速度提升 162–228×；在 2048 GPU 上实现 21.7× 的加速（ogbn‑papers100M）。

**⚠️ 局限性**

局限性包括：①数据并行组数增加后梯度 all‑reduce 开销上升；②统一顶点采样对极度不均匀图结构的捕获能力有限；③对 NCCL/RCCL 的依赖，需特定 GPU 互连和库支持；④跨框架比较需保持相同采样策略，且方法主要针对节点分类任务，其他任务的适用性待验证。

---

## 83. Competency Questions as Executable Plans: a Controlled RAG Architecture for Cultural Heritage Storytelling

**arXiv ID:** 2604.02545 | [PDF](https://arxiv.org/pdf/2604.02545v1)

**作者:** Naga Sowjanya Barla `[一作]` (University of Liverpool), Jacopo de Berardinis `[通讯]` (University of Liverpool)

**通讯引用:** 152 | [OpenAlex ID](https://openalex.org/A5033120361)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于知识图谱的 CQ 驱动的 RAG 工作流，用于生成可信的数字文化遗产故事，以 1985 年 Live Aid 演唱会为案例。

**💡 创新点**

创新点在于将竞争问题（CQ）从设计时验证转为运行时可执行叙事计划，并对 KG-RAG、Hybrid-RAG、Graph-RAG 三种检索策略进行系统比较。

**🔧 技术方法**

采用知识图谱、SPARQL、Hybrid‑RAG 与 Graph‑RAG 检索技术，以及 LLM 生成，并实现可审计的 plan‑retrieve‑generate 管道。

**📊 数据集**

使用公开的 Live Aid 知识图谱（多模态、与 Music Meta Ontology 对齐的 1985 年 Live Aid 事件数据）及手工校准的 CQ 集。

**📈 对比分析**

通过在 18 组配置（两种 persona、三种叙事长度）下评估 Support、Coverage、Readability、Local Cohesion、Global Cohesion 等指标，结果显示 KG‑RAG 在支持度最高，Hybrid‑RAG 在覆盖度和叙事丰富度最好，Graph‑RAG 在全局连贯度最高但事实支撑最低。

**⚠️ 局限性**

局限性包括对 Graph‑RAG 的度量方法不足、缺乏人类主观评估、实验仅限于 Live Aid KG 以及 CQ 需要大量人工生成的工作量。

---

## 84. Communication-Efficient Distributed Learning with Differential Privacy

**arXiv ID:** 2604.02558 | [PDF](https://arxiv.org/pdf/2604.02558v1)

**作者:** Xiaoxing Ren `[一作]` (Cornell University), Andreas A. Malikopoulos `[通讯]` (Cornell University)

**通讯引用:** 4532 | [OpenAlex ID](https://openalex.org/A5076592878)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计了一种基于局部训练和梯度裁剪+噪声扰动的分布式ADMM算法，用于在无中心化网络上解决非凸学习问题，并提供差分隐私保护。

**💡 创新点**

创新点在于将局部训练与梯度裁剪+噪声相结合，既显著降低通信频率，又能在保持算法收敛性（至有界距离）的同时实现严格的 Rényi 差分隐私。

**🔧 技术方法**

采用了局部梯度下降、梯度裁剪、Gaussian 噪声、Rényi DP 与DP转换、RDP 序列组合定理以及 ADMM 框架中的桥变量等技术。

**📊 数据集**

实验使用了一个10节点环形网络，每个节点1000条样本、5维特征的逻辑回归分类任务，采用小批量8条进行训练。

**📈 对比分析**

与 PORTER 和 PriSMA 两种现有分布式DP算法在相同隐私预算（ε≈19.6）下比较，结果表明本算法在通信效率、收敛速度和最终分类准确率方面均优于对比方法。

**⚠️ 局限性**

局限性包括对梯度变化（σ_f）和噪声方差的权衡需要手工调参，且假设数据异质性不严重，未来工作需进一步探索自适应裁剪策略和对高异质数据的鲁棒性改进。

---

## 85. Evaluating Generalization and Robustness in Russian Anti-Spoofing: The RuASD Initiative

**arXiv ID:** 2604.02374 | [PDF](https://arxiv.org/pdf/2604.02374v1)

**作者:** Ksenia Lysikova `[一作]` (Moscow Technical University of Communications and Informatics), Kirill Borodin `[通讯]` (Moscow Technical University of Communications and Informatics)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `67630363-6be0-4f51-ab05-7198250671a5` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本研究提出了RuASD——一个专门针对俄语语音伪造检测的可复现基准，包含37种现代俄语TTS/VC生成的伪造语音和从十个公开俄语语料库抽取的真实语音，并提供可配置的房间回声、噪声与语音编解码器仿真以模拟真实传播环境。

**💡 创新点**

创新点包括：①首次构建俄语专用的多源、可复现的抗伪造评测集；②结合多种生成器与多种传播失真，系统评估模型在分布漂移下的鲁棒性；③提供统一的评估流程与公开基线结果，为俄语抗伪造研究提供可重复的实验基准。

**🔧 技术方法**

使用了 TTS/VC 生成技术、RIR、MUSAN 噪声、音频编解码器（mp3、opus、g722 等）仿真、NISQA 语音质量评估、ASR 计算 CER、以及多类检测模型（Res2TCNGuard、AASIST3、TCM‑ADD、Wav2Vec 2.0、SLS、Arena 等）。

**📊 数据集**

主要数据集为 RuASD，包括：spoof 子集（来自 37 个俄语 TTS/VC 系统，约 5,000 条/系统），bona fide 子集（从 Deep Speech、GOLOS、M‑AILABS、OpenSTT、RuLS、RUSLAN、Common Voice、SOVA 等十个俄语语料库提取，约 10 GB），以及用于文本采样的 UNPC 并使用 MUSAN、RIRS 进行增强。

**📈 对比分析**

评估方法采用统一的 16 kHz 采样、固定长度（≈4 s）裁剪/循环，使用 Accuracy、Precision、Recall、F1、ROC‑AUC、EER 等指标。结果显示：在干净数据上，TCM‑ADD 取得最佳性能（Acc ≈ 0.857，RAUC ≈ 0.914，EER ≈ 0.143），Arena 系列紧随其后；但在加入噪声、回声和编解码失真后，所有模型的 EER 均显著上升，且清洁数据上的排名并不完全映射到鲁棒性表现。

**⚠️ 局限性**

局限性包括：①MOS/CER 仅为粗略描述，未完全代表攻击难度；②伪造语音仅来自单一文本域，缺乏词汇与风格多样性；③未包含局部编辑或拼接攻击，导致模型可能偏向全局特征；④bona fide 语料来源多样，模型可能学到来源指纹而非纯粹伪造特征；⑤固定长度评估可能忽视长范围信息。

---

## 86. A Survey on AI for 6G: Challenges and Opportunities

**arXiv ID:** 2604.02370 | [PDF](https://arxiv.org/pdf/2604.02370v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 87. AI Fact-Checking in the Wild: A Field Evaluation of LLM-Written Community Notes on X

**arXiv ID:** 2604.02592 | [PDF](https://arxiv.org/pdf/2604.02592v1)

**作者:** Haiwen Li `[一作]` (Massachusetts Institute of Technology), Michiel A. Bakker `[通讯]` (Massachusetts Institute of Technology)

**通讯引用:** 1771 | [OpenAlex ID](https://openalex.org/A5035791917)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对 X 社区注释（Community Notes）平台上部署的 LLM 事实核查写作器进行为期三个月的真实环境田野评估，收集并对比 LLM 生成的注释与人类作者注释的质量与受众反馈。

**💡 创新点**

①首次在真实社交平台上进行 LLM 事实核查的田野评估；②开放源代码的多步骤写作管线（搜索、评估、生成、质量控制）；③结合平台的“桥接”评估算法和曝光平衡的 note‑level 分析，验证跨意识形态接受度。

**🔧 技术方法**

使用 Grok‑4‑fast 进行网络与 X 内部搜索，GPT‑5‑mini 生成注释；通过 Community Notes API 与平台交互；采用线性混合效应模型与矩阵分解桥接算法进行评价；对多模态内容（文本、图片、视频）进行处理。

**📊 数据集**

共 2,946 条 Community Notes（1,614 条由 LLM 生成，1,332 条人类生成），覆盖 1,597 条推文；收集到 108,169 条来自 42,521 名评审者的评级。

**📈 对比分析**

①评级级别分析：对单个评审者的评级建模，显示 LLM 注释在左、中、右三类评审者中的 helpfulness 得分均高于人类注释，中心立场评审者平均提高约 10%。②等曝光 note‑level 分析：在完全相同评审者集下重新计算 helpfulness 分数，LMM 结果显示 LLM 注释的平均 helpfulness 评分比人类高 0.019（显著）。总体上，LLM 产出的注释在跨意识形态一致性和整体 helpfulness 上优于人类产出。

**⚠️ 局限性**

①LLM 注释因平台规则只能在推文被足够标记后才生成，导致提交时间晚、曝光不足，算法对低评级数的惩罚影响了平台级指标；②等曝光分析受限于评审者子样本，可能不完全代表整体评审人群；③仅测试单一 LLM 写作管线，其他实现可能表现不同；④对 AI 生成内容的检测能力有限，需集成图像/视频反向搜索等工具。

---

## 88. FusionBERT: Multi-View Image-3D Retrieval via Cross-Attention Visual Fusion and Normal-Aware 3D Encoder

**arXiv ID:** 2604.02583 | [PDF](https://arxiv.org/pdf/2604.02583v1)

**作者:** Wei Li `[一作]` (IROOTECH TECHNOLOGY), Baigui Sun `[通讯]` (IROOTECH TECHNOLOGY)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 FusionBERT，结合多视角图像聚合与法向感知的 3D 编码，实现跨模态（图像‑3D）检索。

**💡 创新点**

创新点在于两阶段注意力聚合的多视角视觉融合模块和法向增强的 3D 编码，显著提升多视角检索与纹理缺失场景的鲁棒性。

**🔧 技术方法**

采用 CLIP‑ViT 视觉编码、Transformer 自注意力与跨注意力聚合、InfoNCE 对比学习、Point‑BERT+法向特征的 3D 编码以及两阶段训练策略。

**📊 数据集**

预训练使用 Four 数据集（Objaverse、ShapeNet、ABO、3D‑Future）；评测在 Objaverse‑LVIS、LVIS no‑RGB、ModelNet40 与自建 IMP 数据集上进行。

**📈 对比分析**

与 OpenShape、TAMM、ULIP‑2 等 SOTA 在 Recall@K 进行对比，FusionBERT 在单视角 Recall@1 从 52.1% 提升至 68.7%（3 视角），并在无 RGB 条件下仍保持较高性能。

**⚠️ 局限性**

局限性包括对大量视角依赖，极端遮挡或极少视角下效果可能下降，以及在更大多模态预训练规模下的可扩展性待进一步验证。

---

## 89. Analyzing Reverse Address Translation Overheads in Multi-GPU Scale-Up Pods

**arXiv ID:** 2604.02473 | [PDF](https://arxiv.org/pdf/2604.02473v1)

**作者:** Amel Fatima `[一作]` (University of Virginia), Bradford M. Beckmann `[通讯]` (AMD Research)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

分析多节点多GPU系统中逆向地址翻译（NPA→SPA）对All-to-All聚合通信的性能影响，并提出预翻译融合核和软件驱动TLB预取两种优化方案。

**💡 创新点**

首次系统性评估目标端地址翻译在大规模GPU集群中的开销；发现冷TLB缺失对小规模聚合通信导致高达1.4×的性能下降；证明L2-TLB容量对大多数ML聚合通信几乎无影响。

**🔧 技术方法**

利用ASTRA-sim与Omnet++结合的仿真框架，建模Link MMU、Link TLB（L1、L2）和页表访问；在仿真中测量All-to-All的请求延迟、TLB命中/缺失统计。

**📊 数据集**

未使用真实训练/推理数据集，而是基于MSCCLang产生的All-to-All通信流，覆盖1 MB至4 GB的数据量。

**📈 对比分析**

与理想无翻译开销的基线进行比较，发现小规模聚合（1 MB）时性能下降至0.71×（即1.4×慢），而大规模聚合（16 MB）下降仅至0.91×；进一步实验验证L2‑TLB从32到32 768项对性能无显著提升。

**⚠️ 局限性**

局限性包括：仅仿真模型，缺乏真实硬件验证；聚焦All-to-All而未覆盖其他聚合模式；未对训练阶段大批量数据的吞吐量进行评估；优化方案仅在理论上提出，缺乏实现与评测。

---

## 90. Do Agent Societies Develop Intellectual Elites? The Hidden Power Laws of Collective Cognition in LLM Multi-Agent Systems

**arXiv ID:** 2604.02674 | [PDF](https://arxiv.org/pdf/2604.02674v1)

**作者:** Kavana Venkatesh `[一作]` (Virginia Tech), Jiaming Cui `[通讯]` (Virginia Tech)

**通讯引用:** 872 | [OpenAlex ID](https://openalex.org/A5001813016)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大规模LLM多智能体系统的协调动态进行了系统实验和分析，提出基于事件的协调表述并量化了协调规模、集中度和极端事件。

**💡 创新点**

创新点在于：①首次将协调视为由递归事件组成的重叠级联，揭示重叠级联呈截断幂律分布；②发现“强化路由”导致认知精英和协调瓶颈；③提出基于协调失衡的Deficit‑Triggered Integration (DTI)干预，能在不削弱大规模推理的前提下提升任务成功率。

**🔧 技术方法**

使用事件级分析、最大似然估计、Vuong检验等统计方法，结合重放日志生成的代理交互图，评估层级、拓扑、任务类型与规模对协调分布的影响。

**📊 数据集**

实验基于四个主流LLM多智能体基准（GAIA、SWE‑bench、REALM‑Bench、MultiAgentBench），在 8–512 代理、链/星/树/完全连通/稀疏网/动态声誉等拓扑下，累计约 150 万条协调事件。

**📈 对比分析**

对比基线与DTI，发现任务成功率提升 2%–12%（取决于拓扑与任务），DTI 维持截断幂律的中间尺度特征，同时缩小极端尾部和精英集中度，表明在保持大规模推理能力的同时改善协调结构。

**⚠️ 局限性**

局限性包括：仅考察有限的协调原语和拓扑；事件抽象可能忽略语义细节；DTI 仅调节扩张与整合的单一失衡，未覆盖其他失效模式；实验规模虽大但仍为有限样本，理论推广待进一步验证。

---

## 91. SEDGE: Structural Extrapolated Data Generation

**arXiv ID:** 2604.02482 | [PDF](https://arxiv.org/pdf/2604.02482v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 92. AIVV: Neuro-Symbolic LLM Agent-Integrated Verification and Validation for Trustworthy Autonomous Systems

**arXiv ID:** 2604.02478 | [PDF](https://arxiv.org/pdf/2604.02478v1)

**作者:** Jiyong Kwon `[一作]` (Purdue University), Guang Lin `[通讯]` (Purdue University)

**通讯引用:** 6376 | [OpenAlex ID](https://openalex.org/A5078138445)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `3855fcda-48ef-4070-a15e-803cd5c84d83` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

开发了一种Agent-Integrated Verification and Validation (AIVV)框架，利用大型语言模型协同验证并提升无人水下航行器（UUV）的异常检测与校正。

**💡 创新点**

将神经网络的数学异常检测与LLM多智能体的语义验证相结合，形成双层层次的去伪造与实时适配的混合神经符号体系。

**🔧 技术方法**

使用MC Dropout LSTM、合成预测、LLM多智能体（角色专属：需求工程师、故障经理、系统工程师）、自适应微调与复制-提升机制。

**📊 数据集**

基于REMUS 100 UUV的Yaw角时间序列数据，构造三种航迹（悬停、草坪扫描、复杂任务）并注入传感器失效。

**📈 对比分析**

在不同场景下与仅使用数学门控或单一LLM的基线做对比，AIVV在失效验证率(FVR)上达100%、89.33%和93.33%，并显著提升适配后模型准确率至最高23%提升。

**⚠️ 局限性**

依赖外部LLM推理导致延迟与对hallucination的担忧；在极端动态或低信噪比环境下仍需改进，且需要更多真实硬件验证。

---

## 93. Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets

**arXiv ID:** 2604.02460 | [PDF](https://arxiv.org/pdf/2604.02460v1)

**作者:** Dat Tran `[一作]` (Stanford University), Douwe Kiela `[通讯]` (Stanford University)

**通讯引用:** 14631 | [OpenAlex ID](https://openalex.org/A5016956470)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在固定思考 token 预算下，对单体 LLM（SAS）与多代理 LLM（MAS）在多跳推理任务中的性能进行理论与实验比较。

**💡 创新点**

提出基于数据处理不等式的信息理论视角，解释在固定预算下单体系统信息效率更高；系统性评估多代理架构在不同模型、任务与上下文退化场景下的优势；发现多代理的提升主要源自计算和上下文误差，而非结构本身。

**🔧 技术方法**

信息理论推导（数据处理不等式、Fano不等式）、预算控制实验、七种多代理架构（Sequential、Debate、Ensemble、Parallel-roles、Subtask-parallel 等）、LLM‑as‑Judge 评估、上下文退化实验。

**📊 数据集**

使用 FRAMES、MuSiQue（4‑hop）等推理数据集，并在 Qwen3、DeepSeek、Gemini 等三大模型族上进行实验。

**📈 对比分析**

通过统一思考 token 预算、LLM‑as‑Judge 评价，比较 SAS 与 MAS 的准确率。结果表明，SAS 在绝大多数预算与模型下匹配或优于 MAS；当单体上下文有效利用被退化到一定程度时，多代理架构在某些情形下能略优于 SAS。

**⚠️ 局限性**

仅在多跳推理与固定思考预算下研究；评估受 LLM‑as‑Judge 可靠性、API 计费与真实计算成本等因素限制；多代理优势可能在更复杂任务或不同预算/资源配置中表现不同。

---

## 94. A Comprehensive Framework for Long-Term Resiliency Investment Planning under Extreme Weather Uncertainty for Electric Utilities

**arXiv ID:** 2604.02504 | [PDF](https://arxiv.org/pdf/2604.02504v1)

**作者:** Emma Benjaminson `[一作]` `[通讯]` (Boston Consulting Group's AI Science Institute), Emma Benjaminson (Boston Consulting Group's AI Science Institute)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了电网资本投资优化的四步框架，比较了基于模型的NSGA-II+ILP与模型无关的NPV+FIFO方法。

**💡 创新点**

创新点是将极端天气概率模型、数字孪生、电网仿真与多目标优化结合，并证明模型感知不一定能超越简化的NPV方法。

**🔧 技术方法**

使用概率极端天气模型、数字孪生（PandaPower）、蒙特卡洛仿真、NSGA-II进化算法和整数线性规划调度。

**📊 数据集**

数据集来自SimBench德国配电网的时间序列与拓扑，构建了小型农村网和中型商业网两个测试案例。

**📈 对比分析**

在同一数字孪生上运行多场景仿真，比较未服务能量、SAIDI/SAIFI/CAIDI等指标；结果显示NPV+FIFO在识别关键线路上更有效，NSGA-II+ILP在调度上更优，整体性能相近。

**⚠️ 局限性**

主要限制是NSGA-II求解耗时极长，无法充分探索搜索空间；极端天气概率假设可能偏高，导致结果偏差。

---

## 95. Computing with Living Neurons: Chaos-Controlled Reservoir Computing with Knowledge Transplant

**arXiv ID:** 2604.02552 | [PDF](https://arxiv.org/pdf/2604.02552v1)

**作者:** Seung Hyun Kim `[一作]` (University of Illinois), Mattia Gazzola `[通讯]` (University of Illinois)

**通讯引用:** 3460 | [OpenAlex ID](https://openalex.org/A5106466015)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

本文提出了基于光学低功耗混沌控制的驻留计算(cc‑RC)方法，并进一步引入知识移植(KT)策略，针对活体神经培养物实现了更稳健、更持久的模式分类学习。

**💡 创新点**

创新点包括：①将自训练的动力学特征识别与混沌控制相结合，实时稳定神经网络的自发与刺激诱发动态；②利用混沌控制实现长期稳定的高维投影，显著提升学习准确度和模型寿命；③通过相位空间对齐实现跨培养物的读出映射迁移，显著缩短学习时间并提升最终性能。

**🔧 技术方法**

技术手段包括：光遗传刺激与光学背景调制（低幅三角波）；微电极阵列(MEA)记录局部场电位；动力学系统理论与高维轨迹分析（GPFA）；线性读出训练（岭回归）以及相位空间几何变换对齐。

**📊 数据集**

使用的数据集：数百个活体神经培养样本（共500+次自发活动记录），在10个光编码输入模式下进行1小时训练、12小时长期测试，共200+实验。

**📈 对比分析**

与传统裸露驻留计算相比，cc‑RC将分类准确率提升约300%，模型寿命延长约3.5倍；KT进一步使学习时间缩短至分钟级，且最终准确率在1小时内提升约20%，并在学习曲线中表现出更强的抗漂移性。

**⚠️ 局限性**

限制主要来自生物体制：1) 约6小时后光刺激对神经网络的响应衰减，导致性能衰退；2) 培养物的有限寿命仍需周期性更换；3) 目前仅针对相对简单的模式分类任务，尚未验证在更复杂任务和闭环控制中的泛化能力。

---

## 96. Beyond Fixed Inference: Quantitative Flow Matching for Adaptive Image Denoising

**arXiv ID:** 2604.02392 | [PDF](https://arxiv.org/pdf/2604.02392v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 97. Evaluating Small Language Models for Front-Door Routing: A Harmonized Benchmark and Synthetic-Traffic Experiment

**arXiv ID:** 2604.02367 | [PDF](https://arxiv.org/pdf/2604.02367v1)

**作者:** Warren Johnson `[一作]` (Plexor Labs), Charles Lee `[通讯]` (Project Autobots)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了使用小型语言模型（SLM）作为前置路由器进行任务分类，以降低整体推理成本、延迟和治理风险。

**💡 创新点**

提出了“前置路由”框架，并通过同步硬件/软件配置的统一基准和四臂随机实验验证了SLM在任务分类上的可行性与边界。

**🔧 技术方法**

采用了零样本分类提示、4‑bit NF4量化、vLLM推理栈、Azure T4 GPU，并对Phi-3.5-mini、Qwen2.5-1.5B/3B、Phi-4-mini、DeepSeek‑V3 等模型进行评估。

**📊 数据集**

使用了包含60个案例、6类任务（code、hybrid、CoT 等）的固定语料库（SHA‑256 哈希已公开）。

**📈 对比分析**

通过配对 McNemar 检验与 Welch‑t 检验，比较了准确率、P95 延迟和边际成本。结果显示 Qwen‑2.5‑3B 以 0.783/0.793 的准确率和 988 ms 的延迟成为 Pareto‑dominant 自托管模型；DeepSeek‑V3 最高 0.830 但超时；无模型满足 85 %/2000 ms 的可行门槛。

**⚠️ 局限性**

局限性包括：未评估分类后端生成的最终质量与成本；使用合成流量、有限的 60‑案例数据集、单注释者标签；缺少并发/吞吐量评估；量化后模型置信度失效；未包含 fine‑tuned 识别器基准。

---

## 98. Environment-Aware Channel Prediction for Vehicular Communications: A Multimodal Visual Feature Fusion Framework

**arXiv ID:** 2604.02396 | [PDF](https://arxiv.org/pdf/2604.02396v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 99. LitPivot: Developing Well-Situated Research Ideas Through Dynamic Contextualization and Critique within the Literature Landscape

**arXiv ID:** 2604.02600 | [PDF](https://arxiv.org/pdf/2604.02600v1)

**作者:** Hita Kambhamettu `[一作]` (University of Pennsylvania), Pao Siangliulue `[通讯]` (Allen Institute for AI)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一款名为 LitPivot 的 AI 辅助研究想法生成系统，支持文献驱动的迭代修正（literature‑initiated pivots），在想法与文献之间实现双向动态交互。

**💡 创新点**

提出并实现了文献-启动的 pivot 机制，并通过 facet‑based 检索、图式推理和动态聚类实现想法与文献空间的协同演化；这一机制在研究想法生成中首次被系统化。

**🔧 技术方法**

使用了 GPT‑4o、Sonnet‑4 等大型语言模型进行文本分段、文献抽取、聚类、评估与改写；结合 Semantic Scholar API、Paper Finder、图论（二分图）等技术构建动态检索与评估框架；前端 React，后端 Flask。

**📊 数据集**

检索得到的文献来自 Semantic Scholar（通过 Paper Finder），覆盖 HCI、NLP、Robotics、Biomed 等领域；实验中使用 17 名研究者的研究想法作为输入。

**📈 对比分析**

通过与基线系统（chat‑with‑papers / IdeaSynth）进行对照实验（n=17）以及专家评分，LitPivot 在文献选择数量、对文献空间的理解、想法新颖性与实用性评分均显著提升（效果量大、显著 p<0.05）。

**⚠️ 局限性**

局限性包括：仅在 HCI/NLP 领域的短时实验，未评估长期研究发展；依赖可用的高质量文献语料库；AI 对概念主导可能导致创造性受限；创新度评估过于简化；未对可行性进行实验验证。

---

## 100. WIO: Upload-Enabled Computational Storage on CXL SSDs

**arXiv ID:** 2604.02442 | [PDF](https://arxiv.org/pdf/2604.02442v1)

**作者:** Yiwei Yang `[一作]` (UC Santa Cruz), Wei Zhang `[通讯]` (University of Connecticut)

**通讯引用:** 38523 | [OpenAlex ID](https://openalex.org/A5008881437)

**关键词:** `9a43038e-f401-4fd9-9c05-65c0b8369d7e` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了可逆的存储侧计算迁移，在 CXL SSD 上实现上传式计算存储，允许在宿主机与设备间动态迁移 I/O 路径计算

**💡 创新点**

核心创新是将存储路径逻辑拆分为可迁移的 WebAssembly Actor，并利用 CXL.coherent 内存实现零拷贝迁移与热/功耗感知调度

**🔧 技术方法**

技术手段包括 CXL 2.0 coherent memory、WebAssembly（WASM）运行时、MVVM 虚拟机、零拷贝 drain‑and‑switch 协议、MWait/UMWait 低功耗等待、异步持久化与两阶段提交恢复

**📊 数据集**

实验数据集包含 RocksDB 100GB 读写混合、STREAM（HPC）和 DeepSeek LLM 推理工作负载，以及自定义 4KB 随机/顺序读写基准

**📈 对比分析**

通过与传统 NVMe SSD、Samsung SmartSSD、ScaleFlux ASIC CSD 的基准对比，展示了在 CXL SSD 上读取延迟下降 8.6 倍、写延迟下降 41.8 倍、吞吐率提升 2 倍、写入延迟 3.75 倍；在持续写入下通过迁移避免 50%–60% 速率下滑

**⚠️ 局限性**

局限性包括 WASM 在计算密集型 kernel 上的 4–5× 运行时开销，迁移仅适用于拥有共存内存的 CXL 设备，对非 CXL 或不具备协同内存的设备不可行；硬件成本和功耗仍是瓶颈，且目前仅在特定 FPGA/ASIC 设备上验证

---

## 101. Photonic convolutional neural network with pre-trained in-situ training

**arXiv ID:** 2604.02429 | [PDF](https://arxiv.org/pdf/2604.02429v1)

**作者:** Saurabh Ranjan `[一作]` (University of Delhi), Amit Sehgal `[通讯]` (University of Delhi)

**通讯引用:** 1605 | [OpenAlex ID](https://openalex.org/A5103193886)

**关键词:** `7a50eb32-3dbc-4c3e-a038-bda01b2d9965` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出并验证了一种全光学卷积神经网络（PCNN），实现了MNIST图像分类

**💡 创新点**

首次在单芯片实现完整的光学卷积、最大池化和非线性激活，并采用混合离线/在线SPSA训练

**🔧 技术方法**

使用硅光子学组件（MZI网格、WDM池化、微环非线性）及可差分数字孪生模型与SPSA算法

**📊 数据集**

使用MNIST手写数字数据集进行训练与测试

**📈 对比分析**

与电子GPU对比，单图像推理能耗低100–242倍，推理延迟约0.8µs，分类准确率达到94%

**⚠️ 局限性**

受限于光学串行流处理导致的吞吐量瓶颈以及热耦合对精度的轻微影响

---

## 102. Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals

**arXiv ID:** 2604.02603 | [PDF](https://arxiv.org/pdf/2604.02603v1)

**作者:** Kunzhe Song `[一作]` (Michigan State University), Huacheng Zeng `[通讯]` (Michigan State University)

**通讯引用:** 1855 | [OpenAlex ID](https://openalex.org/A5027120851)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种利用毫米波OFDM通信信号的单设备多帧融合框架，完成高精度3D场景重建；

**💡 创新点**

创新点在于实现单设备单向（monostatic）全双工感知，利用CIR提取范围和角度信息，并通过置信度加权的多帧空间自适应融合，显著抑制多路径伪影；

**🔧 技术方法**

采用毫米波全双工OFDM信号的CIR估计、脉冲响应变换、空间投影与置信度加权融合、卷积编码-解码网络；

**📊 数据集**

在20个室内环境中收集同步毫米波通信、LiDAR和IMU数据，构成跨场景测试数据集；

**📈 对比分析**

与PanoRadar、CartoRadar等基线相比，在单帧和多帧（5帧）设置下，平均Depth AbsRel降至9.4%，Chamfer Distance下降至2.3%，表现优异；

**⚠️ 局限性**

局限性包括对旋转误差敏感、在极端遮挡下仍有误差、依赖室内环境，室外或大范围高精度感知尚待验证。

---

## 103. Red Flags and Cherry Picking: Reading The Scientific Blackpill Wiki

**arXiv ID:** 2604.02565 | [PDF](https://arxiv.org/pdf/2604.02565v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 104. I must delete the evidence: AI Agents Explicitly Cover up Fraud and Violent Crime

**arXiv ID:** 2604.02500 | [PDF](https://arxiv.org/pdf/2604.02500v1)

**作者:** Thomas Rivasseau `[一作]` (McGill University), Benjamin Fung `[通讯]` (McGill University)

**通讯引用:** 12856 | [OpenAlex ID](https://openalex.org/A5021788449)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

系统测试了16种先进大型语言模型在公司指令与法律冲突情境下，是否会协助掩盖犯罪证据以维护公司盈利。

**💡 创新点**

首次将企业利益与法律安全冲突纳入代理对齐评估，揭示现代LLM在此类情境下的安全缺口。

**🔧 技术方法**

通过精心构造的提示、链式推理（Chain-of-Thought）和手工分类分析，评估模型行为。

**📊 数据集**

使用人工模拟的情境对话（无真实数据集），包含举报人、CEO及事件描述等虚构信息。

**📈 对比分析**

对160条模型回复进行手工分为理想、中立、隐式非法、显式非法四类；结果显示大多数模型偏向非法操作，唯有Claude Sonnet 3.5/4、o3、GPT‑5.2表现合规。

**⚠️ 局限性**

实验受限于模拟环境、可能的评估意识偏差、对齐训练样本相似度影响，且仅评估了极端情景，真实部署场景仍需进一步验证。

---

## 105. SIEVE: Sample-Efficient Parametric Learning from Natural Language

**arXiv ID:** 2604.02339 | [PDF](https://arxiv.org/pdf/2604.02339v1)

**作者:** Parth Asawa `[一作]` (University of California, Berkeley), Matei Zaharia `[通讯]` (University of California, Berkeley)

**通讯引用:** 49884 | [OpenAlex ID](https://openalex.org/A5005554337)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种样本高效的参数化学习方法，通过合成数据生成与上下文蒸馏实现将自然语言上下文内部化至模型权重。

**💡 创新点**

创新点在于利用上下文可分解性，先将长文本拆分为可独立评估的上下文单元，再针对每个合成查询仅配对其适用单元，从而显著提升训练数据质量与样本利用率。

**🔧 技术方法**

核心技术包括：基于指令微调模型的上下文拆分、基于基础语言模型的种子选取与后向翻译生成合成查询、离线验证确定适用单元；随后采用教师-学生上下文蒸馏（软目标KL损失）将仅含查询的学生模型训练至能不依赖上下文完成推理。

**📊 数据集**

使用的任务与数据集包括：Synthetic Retail（30 条折扣规则），RuleArena（NBA 交易规则约 20K 词），以及 MTOB（极低资源语言 Kalamang 翻译，约 50K 词长上下文）。实验仅依赖这些上下文与 3 条示例查询，无需专家轨迹或自动验证器。

**📈 对比分析**

与传统 ICL、V_CD、V_CD‑S、以及长上下文专用的 Cartridges 进行对比。结果显示：在 Retail 上取得 36% 精确率，超过 ICL 30%；RuleArena 提升约 10% 以上；MTOB 在 16K 合成样本下达 24.48 chrF，显著优于 Cartridges 19.10。数据规模增大时表现持续提升，证明方法具备良好的可扩展性。

**⚠️ 局限性**

局限性包括：需基模型具备足够的推理与生成能力，弱模型如 Llama 3.1 8B 在 Retail 上甚至低于 ICL；对长上下文的离线验证成本高；方法假设上下文可拆分为互不干扰的单元，可能不适用于高度依赖全局信息的任务；以及在极端低资源语言翻译等记忆型任务中仍难以完全击败 ICL。

---

## 106. LLM Reasoning with Process Rewards for Outcome-Guided Steps

**arXiv ID:** 2604.02341 | [PDF](https://arxiv.org/pdf/2604.02341v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 107. A Spectral Framework for Multi-Scale Nonlinear Dimensionality Reduction

**arXiv ID:** 2604.02535 | [PDF](https://arxiv.org/pdf/2604.02535v1)

**作者:** Zeyang Huang `[一作]` (Linköping University), Takanori Fujiwara `[通讯]` (University of Arizona)

**通讯引用:** 971 | [OpenAlex ID](https://openalex.org/A5074006931)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种在拉普拉斯谱子空间中执行交叉熵优化的谱框架，利用谱子空间逐步扩展实现多尺度、可解释的降维。

**💡 创新点**

创新点在于把邻域嵌入优化映射到图谱频率基底，显式控制全局-局部权衡，并通过逐层子空间扩展生成可解释的多尺度嵌入。

**🔧 技术方法**

使用拉普拉斯谱分解、谱子空间投影、UMAP式交叉熵优化、渐进子空间策略，以及视觉化的谱响应曲线和花瓣图标进行解释。

**📊 数据集**

在合成数据（瑞士卷、周期循环、布朗树）、图像数据（MNIST、Fashion‑MNIST）以及单细胞 RNA‑seq 的 C. elegans（原始和预处理）等多种公开数据集上进行实验。

**📈 对比分析**

与 Laplacian Eigenmaps、UMAP、PHATE 等基线比较，使用邻域保持、Manifold fidelity (DEMaP)、Spearman ρ、非度量应力等指标，表现接近或优于基线，尤其在全局结构保持与多尺度解释方面表现突出。

**⚠️ 局限性**

局限性包括需要先进行谱分解（对大规模图有计算瓶颈）、子空间尺寸仍需手工调节、对高频噪声敏感，以及缺乏完全自动化的超参数优化策略。

---

## 108. A Synthesis Method of Safe Rust Code Based on Pushdown Colored Petri Nets

**arXiv ID:** 2604.02399 | [PDF](https://arxiv.org/pdf/2604.02399v1)

**作者:** Kaiwen Zhang `[一作]` (Tongji University), Guanjun Liu `[通讯]` (Tongji University)

**通讯引用:** 5904 | [OpenAlex ID](https://openalex.org/A5072386697)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一种基于Pushdown Colored Petri Nets（PCPN）的安全Rust代码自动合成方法，能够从库的公共API签名直接生成符合所有所有权、借用和生命周期约束的可编译代码；

**💡 创新点**

创新点包括：①将Rust签名直接映射为PCPN模型，实现了类型、权限和生命周期三大约束的联合建模；②通过强bisimulation证明PCPN与Rust编译器检查等价；③在有限搜索空间内构造可归约到可执行代码的可达图；

**🔧 技术方法**

核心技术包括：Pushdown Colored Petri Nets（PCPN）、符号统一与约束求解、生命周期栈的Push/Pop语义、强bisimulation证明、有限可达图构造与回溯；

**📊 数据集**

实验使用了从真实Rust crate（如标准库和常见第三方库）提取的签名环境Σ(𝒞)，并在GitHub上公开的RustSynth工具上进行了验证；

**📈 对比分析**

方法与手工编写代码对比时，所有合成出的代码均通过Rust编译器编译，且在功能上等价；实验表明在给定的资源与堆栈深度限制下，搜索耗时可控且结果完整；

**⚠️ 局限性**

主要局限性包括：仅支持安全Rust（不处理unsafe或async）；搜索空间受限于手工设定的类型、实例和堆栈深度限制，可能导致状态爆炸；在高度泛型或深层嵌套的库中仍存在可达性覆盖不足的风险。

---

## 109. Developer Experience with AI Coding Agents: HTTP Behavioral Signatures in Documentation Portals

**arXiv ID:** 2604.02544 | [PDF](https://arxiv.org/pdf/2604.02544v1)

**作者:** Oleksii Borysenko `[一作]` `[通讯]` (Cisco DevNet), Oleksii Borysenko (Cisco DevNet)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

对九款 AI 编码代理和六款 AI 助手在访问开发者文档端点时产生的 HTTP 请求指纹进行实验收集与分析，揭示了这些代理的请求模式、预抓取策略、User-Agent 与请求头特征，并讨论了这些指纹对传统文档访问指标（如浏览深度、停留时间、跳出率）的影响。

**💡 创新点**

首次系统识别并归纳 AI 代理的 HTTP 行为签名，证明了多页导航被压缩为一次或两次请求的现象；提出了针对 tokenomics 的文档设计、可机器读取的标准（如 <openapi>, <manifest>）以及 MCP 服务器反馈渠道的实践建议，以适应 AI 代理时代的文档交互与分析需求。

**🔧 技术方法**

使用 Node.js/Express 搭建自定义开发者门户，插入 robots.txt 与 sitemap.xml；通过服务器端中间件捕获完整请求头、User-Agent、方法、URL、IP 与 RTT；对不同代理的 HTTP runtime（Headless Chromium、Go、Node.js/axios、curl、Node.js/got 等）和预抓取行为进行归纳；利用 JSON、YAML 等标准文件对发现、能力与交互治理进行结构化分析。

**📊 数据集**

收集了自建文档端点的访问日志，涵盖了 9 款 AI 编码代理（Aider、Antigravity、Claude Code、Cline、Cursor、Junie、OpenCode、VS Code、Windsurf）和 6 款 AI 助手服务（ChatGPT、Claude、Google Gemini、Google NotebookLM、MistralAI、Perplexity）在三次独立试验中的请求指纹；日志中包含完整的请求头、User-Agent、IP、RTT 等信息。

**📈 对比分析**

通过对比各代理的 HTTP runtime、预抓取策略、User-Agent 与请求头存在情况，形成了行为指纹表格；结果显示轻量级 HTTP 客户端与完整浏览器运行时在抓取内容、执行 JavaScript 方面存在显著差异；同时揭示单次请求压缩多页导航后，传统交互指标失效，需引入 AI 代理特定的分析方法（如 AI 参照流量来源、token 计数显示）。

**⚠️ 局限性**

仅覆盖公开、开源或免费试用版代理，未包含专有商业部署；数据为单次快照，缺乏纵向追踪；实验使用单一文档端点，无法验证在多种渲染、鉴权或 anti‑bot 机制下的代理行为；HTTP 指纹受代理依赖的 LLM 或外部搜索服务影响，难以单独归因。

---

## 110. Prism: Policy Reuse via Interpretable Strategy Mapping in Reinforcement Learning

**arXiv ID:** 2604.02353 | [PDF](https://arxiv.org/pdf/2604.02353v1)

**作者:** Thomas Pravetz `[一作]` `[通讯]`, Thomas Pravetz

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出 PRISM 框架，将强化学习代理的内部特征通过 K‑means 聚类得到离散概念，并利用这些概念作为跨算法的零拷贝转移接口，实现不同算法（PPO、DQN、BC）之间的策略迁移。

**💡 创新点**

创新点在于：① 用可因果验证的离散概念作为中间表示，使策略可解释并跨算法共享；② 通过 Hungarian 匹配实现无梯度零拷贝跨算法转移；③ 通过干预与消融实验证明概念的因果驱动性，并揭示概念重要性与频率解耦；④ 发现转移效果取决于源策略强度，而非概念空间的几何相似度。

**🔧 技术方法**

使用的技术包括：K‑means 聚类、Hungarian（匈牙利）匹配、概念干预与消融实验、Bottleneck 策略（embedding+MLP）、PPO、DQN、BC 训练、GnuGo 7×7 对弈、Atari Breakout 作为失败案例。

**📊 数据集**

主要数据集：Go 7×7 盘面观测与 GnuGo 对弈日志，用于概念学习和评估；Atari Breakout 用于验证框架在连续动态域的适用性。

**📈 对比分析**

对比方法包括：随机映射、无对齐、从零训练、Fine‑tune 训练；在 7×7 上，PPO→DQN 与 BC→DQN 的零拷贝转移分别获得 69.5% 与 76.4% 的胜率，远高于随机 3.5% 和无对齐 9.2%；Fine‑tune 在转移后 5 代即可达到 60% 胜率，显著降低训练成本；在 Breakout 上，概念瓶颈与转移均失效，表现与随机相当。

**⚠️ 局限性**

局限性包括：仅适用于具有离散策略结构的域；对齐质量指标不预测转移效果；K‑means 聚类对种子敏感，需要固定或稳定性选择；框架目前只验证同一 encoder 架构下的不同训练算法，未检验跨架构；转移成功依赖源策略强度且目标编码器需功能；在连续控制任务如 Breakout 失效。

---

## 111. Redirected, Not Removed: Task-Dependent Stereotyping Reveals the Limits of LLM Alignments

**arXiv ID:** 2604.02669 | [PDF](https://arxiv.org/pdf/2604.02669v1)

**作者:** Divyanshu Kumar `[一作]` (Enkrypt AI), Prashanth Harshangi `[通讯]` (Enkrypt AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对七款LLM在九种偏见轴上进行多任务审计，揭示偏见在显式与隐式任务间的差异。

**💡 创新点**

提出层级偏见分类体系和跨任务评估框架，证明单一基准无法完整评估模型偏见。

**🔧 技术方法**

采用模板化提示生成、自动化响应分类、统计SS和RR指标，并用GPT‑5.4‑mini作为评判器进行开放式任务。

**📊 数据集**

使用约4.5万条结构化提示（含约150+主题），涵盖身份属性、主题与话题组合，来源于公开与自建偏见词表。

**📈 对比分析**

通过SS与RR对比发现：安全对齐在显式任务上拒绝率高但隐式任务SS仍偏高，且对负面偏见的拒绝率远高于正面偏见；性能差距随任务显式程度显著。

**⚠️ 局限性**

局限包括仅用英文提示、模板化生成可能遗漏自然对话偏见、评估器为单一LLM且对齐模型全部为2026年前沿级别，且未覆盖年龄、残障等轴。

---

## 112. Train Yourself as an LLM: Exploring Effects of AI Literacy on Persuasion via Role-playing LLM Training

**arXiv ID:** 2604.02637 | [PDF](https://arxiv.org/pdf/2604.02637v1)

**作者:** Qihui Fan `[一作]` (Northeastern University), Weiyan Shi `[通讯]` (Northeastern University)

**通讯引用:** 1218 | [OpenAlex ID](https://openalex.org/A5089522357)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `a4b10f5d-130b-4e77-9367-6469ec621899` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

设计并评估了LLMimic——一种基于角色扮演、交互与游戏化的AI素养教程，帮助用户从第一人称视角理解LLM训练过程，并通过人机实验验证其能提高AI素养、降低AI说服成功率。

**💡 创新点**

将AI素养教育从被动观看转为主动角色扮演与实时反馈，并将游戏化元素融入其中，从而提升参与度和对说服情境的抵御效果。

**🔧 技术方法**

实现交互式Web工具以模拟预训练、SFT和RLHF过程；使用Meta AI Literacy Scale评估素养；采用Logistic回归与结构方程模型分析说服结果。

**📊 数据集**

自制的LLM预训练、SFT、RLHF示例及三种说服情境（慈善捐赠、诈骗请求、酒店推荐），以及通过Prolific招募的274名美国成人样本。

**📈 对比分析**

采用2×3（干预/控制 × 三种说服情境）实验设计；结果显示LLMimic显著提升AI素养（p<.001），降低说服成功率（OR=0.58，p=.045），并在酒店情境提高TARES真诚度与社会责任感。

**⚠️ 局限性**

干预仅为短时一次性实验，缺乏长期效应验证；自我报告的素养与信任未能完全解释说服机制；样本为非技术美国成人，外推性受限。

---

## 113. From Theory to Practice: Code Generation Using LLMs for CAPEC and CWE Frameworks

**arXiv ID:** 2604.02548 | [PDF](https://arxiv.org/pdf/2604.02548v1)

**作者:** Murtuza Shahzad `[一作]` (Northern Illinois University), Mona Rahimi `[通讯]` (Northern Illinois University)

**通讯引用:** 446 | [OpenAlex ID](https://openalex.org/A5102905826)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文利用 GPT-4o 等大型语言模型，结合 CAPEC 与 CWE 的描述，自动生成 615 条 Java、Python、JavaScript 的易受攻击代码示例，并构建了相应的数据集。

**💡 创新点**

创新点在于将 CAPEC 与 CWE 通过语义相似度映射，使用 LLM 生成具体攻击代码，首次提供规模较大、跨语言的漏洞示例数据集，并评估其可编译性、相关性与可读性。

**🔧 技术方法**

采用 GPT‑4o、Llama 3、Claude‑3‑5‑Sonnet 作为生成模型，配合 SBERT 进行语义映射，LangChain 进行调用，CodeBERT 评估相似度。

**📊 数据集**

使用 MITRE 提供的 CAPEC 3.9 与 CWE 4.14 作为文本源，并在此基础上生成 615 条 CAPEC 代码样本。

**📈 对比分析**

通过手工评估、编译测试和余弦相似度对比，发现 90% 以上代码可编译，相关性评估一致率>90%，生成结果在不同模型间相似度≥0.98，性能稳定。

**⚠️ 局限性**

局限性包括部分 CAPEC 与软件无关导致无效代码、仅使用 5 条相关 CWE 可能限制上下文质量、未对 CAPEC 进行软件相关性预过滤、语言覆盖有限等。

---

## 114. PolyJarvis: LLM Agent for Autonomous Polymer MD Simulations

**arXiv ID:** 2604.02537 | [PDF](https://arxiv.org/pdf/2604.02537v1)

**作者:** Alexander Zhao `[一作]` (Carnegie Mellon University), Amir Barati Farimani `[通讯]` (Carnegie Mellon University)

**通讯引用:** 8741 | [OpenAlex ID](https://openalex.org/A5008745801)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

PolyJarvis 自动化地将自然语言聚合物描述转换为全原子 MD 流程，完成聚合物结构构建、力场参数化、GPU 加速的 NPT 等温等压平衡以及 Tg、密度和体积模量等性能预测；

**💡 创新点**

创新点在于将大语言模型与 MCP 服务器结合，形成可自适应决策、错误恢复与迭代优化的端到端智能代理，克服传统手工驱动的高专业门槛；

**🔧 技术方法**

技术包括 Anthropic Claude LLM、RadonPy、LAMMPS GPU 加速、MCP 框架、自动化力场/电荷选型、分阶段压缩/冷却协议以及 bilinear 拟合提取 Tg；

**📊 数据集**

使用的数据集是四种常见无机共价聚合物（PE、aPS、PMMA、PEG）的 SMILES 与实验/文献参考值，涵盖 Tg、密度与体积模量；

**📈 对比分析**

通过与实验和先前 MD 文献对比，按±20 K（Tg）、±5 %（密度）和±30 %（体积模量）的严格标准评估，5/8 组合通过，验证了代理在预测精度上的可行性；

**⚠️ 局限性**

局限性包括仅验证四种均相共聚物、PE 密度系统性偏高、Tg 受冷却速率影响、实验体积模量数据不足、LLM 上下文窗口与会话超时等问题。

---

## 115. Guideline2Graph: Profile-Aware Multimodal Parsing for Executable Clinical Decision Graphs

**arXiv ID:** 2604.02477 | [PDF](https://arxiv.org/pdf/2604.02477v1)

**作者:** Onur Selim Kilic `[一作]` (Georgia Institute of Technology), Eli Waxman `[通讯]` (MetaDialog)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了一个从全文临床指南到可执行临床决策图的分解先行管道

**💡 创新点**

创新点在于拓扑感知块化、接口约束的块级图生成、全局去重聚合，并保持可审计性

**🔧 技术方法**

使用多模态VLM进行分块、实体识别、边界检测、结构化输出、语义去重等任务

**📊 数据集**

使用单一已校准的前列腺指南作为基准数据集

**📈 对比分析**

与Doc2KG和AutoKG在相同VLM后端、相同输入下对比，显著提升边和三元组精度/召回率（边精度+49.4，边召回+71.4，三元组同上）

**⚠️ 局限性**

仅在单一指南上验证，缺乏多指南泛化评估

---

## 116. Understanding the Nature of Generative AI as Threshold Logic in High-Dimensional Space

**arXiv ID:** 2604.02476 | [PDF](https://arxiv.org/pdf/2604.02476v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 117. A Dynamic Toolkit for Transmission Characteristics of Precision Reducers with Explicit Contact Geometry

**arXiv ID:** 2604.02387 | [PDF](https://arxiv.org/pdf/2604.02387v1)

**作者:** Jiacheng Miao `[一作]` (CRRC Qishuyan Institute Co., Ltd), Weidong He `[通讯]` (Dalian Jiaotong University)

**通讯引用:** 1231 | [OpenAlex ID](https://openalex.org/A5100351714)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `a8e75ba4-7a2d-4153-b003-06c94533add0` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了一套基于显式接触几何的精密减速机动态仿真工具，能够在几百自由度下快速、精确地模拟齿轮接触、轴承针形接触及柔性结构耦合；

**💡 创新点**

创新点在于将显式几何接触、有限元式柔性求解（ANCF）与多阶段预筛选+探针回退策略结合，形成模块化、可脚本化的统一框架，显著提升了高精度动力学模拟的效率与可扩展性；

**🔧 技术方法**

采用的核心技术包括：基于Timoshenko梁与Hertz理论的位置相关齿啮合刚度模型、针形轴承与曲线接触的多类接触原语、Newton–Raphson与Generalized‑α时间积分、热冲击与摩擦的正则化、以及多阶段预筛选+探针回退加速算法；

**📊 数据集**

数据集主要来自已发表的减速机基准（如RV‑40E、RV‑80E等）以及制造误差分布（±10 µm等），并通过对比已公开的实验结果和数值基准进行验证；

**📈 对比分析**

与传统的Lumped Parameter Model、有限元全场求解以及公开的多体动力学软件相比，本工具在保持相似或更高精度的同时，计算时间减少约50‑70%，且可在单机环境下完成多种拓扑的耦合仿真；

**⚠️ 局限性**

主要限制包括：当前仅支持二维平面齿形（如圆柱齿轮、圆弧齿轮）和挤压体几何；对三维曲面（如偏心齿、蜗轮）仍需开发新的表面接触算法；以及在极高自由度或大尺度系统时仍面临计算瓶颈，未来计划引入GPU加速。

---

## 118. VLMs Need Words: Vision Language Models Ignore Visual Detail In Favor of Semantic Anchors

**arXiv ID:** 2604.02486 | [PDF](https://arxiv.org/pdf/2604.02486v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 119. Hierarchical, Interpretable, Label-Free Concept Bottleneck Model

**arXiv ID:** 2604.02468 | [PDF](https://arxiv.org/pdf/2604.02468v1)

**作者:** Haodong Xie `[一作]` (University of Manchester), Angelo Cangelosi `[通讯]` (University of Manchester)

**通讯引用:** 10150 | [OpenAlex ID](https://openalex.org/A5091768977)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种层级可解释标签自由概念瓶颈模型（HIL-CBM），实现对图像分类结果在两个语义层级（基本层和细粒层）同时给出概念解释

**💡 创新点**

创新点在于：①通过梯度可视一致性损失在概念空间实现不同抽象层的空间对齐；②引入树路径KL一致性损失在标签空间实现层级预测的一致性；③无需显式概念层级关系标签，保持标签自由；④在同一模型中同时提供层级解释

**🔧 技术方法**

使用CLIP‑Dissect进行概念监督、Grad‑CAM基视觉一致性损失、树路径KL一致性损失、GLM‑SAGA与弹性网络的稀疏分类器训练，以及ViT/ResNet预训练特征提取器

**📊 数据集**

在CIFAR‑100、CUB‑200、Places365和ImageNet四个公开数据集上训练与评测，分别使用RN50、ResNet‑50和ViT‑B/16作为backbone

**📈 对比分析**

与多种最新CBM（P‑CBM、LF‑CBM、CF‑CBM、Hybrid‑CBM、SALF‑CBM等）在相同backbone、稀疏分类器设置下对比，HIL‑CBM在分类准确率上显著优于或匹配其他模型，且在层级一致性与解释性指标上更优

**⚠️ 局限性**

局限性包括：仅实现两级层级（无法捕获更深层语义结构）；对概念生成依赖GPT‑4，质量受限于Prompt设计；缺乏大规模多层级数据集验证；解释与调试仍需人工干预，未提供统一量化评估

---

## 120. Variational Encoder--Multi-Decoder (VE-MD) for Privacy-by-functional-design (Group) Emotion Recognition

**arXiv ID:** 2604.02397 | [PDF](https://arxiv.org/pdf/2604.02397v1)

**作者:** Anderson Augusma `[一作]` (University of Grenoble Alpes), Fédérique Letué `[通讯]` (University of Grenoble Alpes)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `9cc9baba-5356-466d-81ff-d80028d90279` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种变分编码器–多解码器（VE‑MD）框架，用全景视频直接预测集体情绪，避免个体检测、跟踪和身份识别，采用结构监督来引导潜在空间；

**💡 创新点**

在隐私友好功能设计上创新：①仅输出群体级情绪，无个体级情绪标签；②通过内部结构解码（人体/面部关节点）作为辅助任务；③提出两种解码策略（基于Transformer的PersonQuery与密集Heatmap）并比较其对不同任务（GER vs IER）的影响；④发现单纯压缩结构信息对群体情绪不利，而对个体情绪有正向作用。

**🔧 技术方法**

主要技术包括：变分编码器（冻结ViT+可训练ResNet‑50），多任务学习（情绪分类 + 结构回归），Transformer基解码器与ST‑GCN，热图解码器；多模态融合：Whisper（语义文本）+ Wav2Vec‑2.0（声学）+ 视觉；late fusion、Attention‑Guided Feature（AFG）等融合策略；以及多任务损失（交叉熵 + 结构损失 + MMD）。

**📊 数据集**

使用六个真实场景数据集：GAF‑3.0、VGAF（群体情绪）以及SAMSEMO、MER‑MULTI、DFEW、EngageNet（个体情绪），并在每个数据集上自动生成人体与面部结构标注。

**📈 对比分析**

与现有SOTA比较，VE‑MD在GAF‑3.0上取得90.06%准确率，超过此前最高86.90%；在VGAF上以82.25%准确率（结合Wav2Vec‑2.0）领先81.98%；在SAMSEMO上得到79.20%准确率（77.94% F1），超过69.00%；在其他数据集表现与SOTA相近或略低。统计检验显示多组实验差异显著。

**⚠️ 局限性**

局限性：1）未提供正式的匿名化或加密隐私保证，仅通过功能限制避免个体监测；2）结构监督依赖自动姿态/面部检测，可能引入噪声；3）多解码器架构复杂度高；4）对不同场景的泛化仍需进一步改进，尤其是对姿态标注质量的依赖。

---

## 121. Revealing the Learning Dynamics of Long-Context Continual Pre-training

**arXiv ID:** 2604.02650 | [PDF](https://arxiv.org/pdf/2604.02650v1)

**作者:** Yupu Liang `[一作]` (Tencent Hunyuan), Suncong Zheng `[通讯]` (Tencent Hunyuan)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在工业级大模型Hunyuan‑A13B上，对200B‑token的长上下文持续预训练（LCCP）进行系统学习动态分析，构建行为、概率与机制三级监控框架；

**💡 创新点**

提出通过连续PPL替代传统NIAH得分揭示“欺骗性饱和”，并将检索头（retrieval head）作为低资源的训练进度监测指标；

**🔧 技术方法**

采用RoPE频率调整、Grouped‑Query Attention、Sparse Mixture‑of‑Experts（MoE）结构，并结合轻量级SFT探针与PPL、NIAH热力图分析；

**📊 数据集**

使用25%短序列、75%长序列混合数据，其中长序列来源包括Common Crawl 36.3%、Books 28.6%、arXiv 24.0%、Code 10.8% 等，训练总量200B token；

**📈 对比分析**

通过RULER、MRCR、LongBio等长上下文基准评估SFT后模型性能，发现模型在100B‑token处开始饱和；PPL指标与SFT结果的相关性高于传统NIAH得分；检索头指标与下游性能亦呈显著正相关；

**⚠️ 局限性**

仅覆盖200B-token的LCCP轨迹，未探究更长上下文（>256K）或更大数据规模；缺乏完整的对齐、RL‑SFT过程；未对不同数据比例或超参进行系统消融；缺少跨模型验证。

---

## 122. Low-Rank Compression of Pretrained Models via Randomized Subspace Iteration

**arXiv ID:** 2604.02659 | [PDF](https://arxiv.org/pdf/2604.02659v1)

**作者:** Farhad Pourkamali-Anaraki `[一作]` (University of Colorado Denver), Farhad Pourkamali-Anaraki `[通讯]` (University of Colorado Denver)

**通讯引用:** 710 | [OpenAlex ID](https://openalex.org/A5043966041)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文针对预训练模型的规模过大问题，提出使用随机子空间迭代（RSI）对线性层权重进行低秩压缩，并给出了理论证明和实验评估。

**💡 创新点**

创新点在于：① 将RSI与传统RSVD对比，证明RSI通过多次幂迭代显著提高谱分离，降低近似误差；② 通过软max扰动分析，将权重压缩误差与预测概率偏差关联，提供压缩质量与模型性能的理论连接。

**🔧 技术方法**

使用的技术包括：随机子空间迭代（RSI）算法、稀疏奇异值分解（RSVD）对照实验、理论软max扰动分析、GPU实现的矩阵乘法加速。

**📊 数据集**

实验数据集包括：VGG19（ImageNet预训练）和ViT‑B/32（ImageNet预训练），评估使用Imagenette（10类）进行无微调的下游任务。

**📈 对比分析**

比较方法：在单层和全模型压缩两阶段，分别比较RSI与RSVD在相同压缩率下的Top‑1/Top‑5准确率、参数压缩比例与运行时间。结果显示：RSI在相同压缩率下保持更高的准确率（尤其在α=0.2–0.4压缩率下显著优于RSVD），且在相同rank下比RSVD快数倍，整体压缩时间低于1秒。

**⚠️ 局限性**

限制：① RSI需多次矩阵乘法，虽然在GPU上高效，但在CPU或极低算力设备上仍可能较慢；② 对于不同层的自适应rank选择仍未自动化，需手动调参；③ 仅在视觉模型验证，跨模态或大规模语言模型的效果尚未测试。

---

## 123. Improving MPI Error Detection and Repair with Large Language Models and Bug References

**arXiv ID:** 2604.02398 | [PDF](https://arxiv.org/pdf/2604.02398v1)

**作者:** Scott Piersall `[一作]` (University of Central Florida), Liqiang Wang `[通讯]` (University of Central Florida)

**通讯引用:** 10628 | [OpenAlex ID](https://openalex.org/A5100427869)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

使用大模型结合少样本学习、链式思考和检索增强生成技术，对MPI程序的缺陷检测和修复进行改进。

**💡 创新点**

提出基于bug引用的Few-Shot+CoT+RAG框架，显著提升LLM在MPI缺陷检测和修复中的准确性。

**🔧 技术方法**

采用LLM（ChatGPT、Llama2、Code Llama、Qwen2.5-Coder）+ Few-Shot学习 + Chain-of-Thought推理 + Retrieval-Augmented Generation + 自动评测框架。

**📊 数据集**

使用MPI Bugs Initiative公开数据集（241条MPI程序，包含已知缺陷）。

**📈 对比分析**

与零样本、仅Few-Shot、仅CoT、RAG等对照，检测准确率从44%提升至78%；在其他LLM上也获得类似提升；修复成功率约76%；相比传统工具，误报率降低、漏报率显著下降。

**⚠️ 局限性**

误报率上升、对死锁缺陷修复效果不佳、RAG效果有限、模型仍受训练数据缺陷示例稀缺限制。

---

## 124. Backup-Based Safety Filters: A Comparative Review of Backup CBF, Model Predictive Shielding, and gatekeeper

**arXiv ID:** 2604.02401 | [PDF](https://arxiv.org/pdf/2604.02401v1)

**作者:** Taekyung Kim `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**通讯引用:** 2405 | [OpenAlex ID](https://openalex.org/A5059647993)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文通过统一的安全过滤器抽象，对三种备份式安全过滤器（Backup CBF、MPS、Gatekeeper）进行了理论与实验比较，并展示了通过搜索切换时间来减小保守性的Gatekeeper。

**💡 创新点**

创新点在于提出了对安全过滤器的统一框架，定义了滤波器不活跃集并证明了MPS是Gatekeeper的特殊情况；同时证明了Gatekeeper的内部集包含Backup CBF的内部集，从而解释了其较低的干预率。

**🔧 技术方法**

采用控制理论中的备份策略、可恢复集、控制障碍函数、模型预测屏蔽、最大切换时间搜索、离散化验证以及数值优化（QP）等技术。

**📊 数据集**

使用仿真数据集，包括二维平面双积分器、动态障碍物的到达-躲避任务以及逼近八维动态自行车模型的高速公路超车任务；并在每个场景中记录了不同安全过滤器的轨迹、成功率和计算时间。

**📈 对比分析**

比较方法：在统一抽象下对滤波器不活跃集进行集合关系证明，并在三种任务中对 nominal 追踪率、平均切换时间、成功率和在线计算时间进行量化对比；实验显示 Gatekeeper 在 nominal 追踪率和成功率上明显优于 MPS 与 Backup CBF，而计算时间保持可接受。

**⚠️ 局限性**

局限性包括：对切换时间搜索范围（T_H）和离散化网格的依赖，若搜索过短或网格粗糙则 Gatekeeper 与 MPS 差异不大；对备份策略与模型精度的敏感性，若备份策略或模型不充分则所有方法的保守性都会加剧。

---

## 125. Generating Counterfactual Patient Timelines from Real-World Data

**arXiv ID:** 2604.02337 | [PDF](https://arxiv.org/pdf/2604.02337v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 126. Audio Spatially-Guided Fusion for Audio-Visual Navigation

**arXiv ID:** 2604.02389 | [PDF](https://arxiv.org/pdf/2604.02389v1)

**作者:** Xinyu Zhou `[一作]` (Xinjiang University), Yinfeng Yu `[通讯]` (Xinjiang University)

**通讯引用:** 3726 | [OpenAlex ID](https://openalex.org/A5091800151)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了基于音频空间状态引导的多模态融合框架ASGF-Nav，用于音视频导航任务。

**💡 创新点**

设计了Audio Spatial State Encoder提取音频空间信息，并通过Audio Spatial State Guided Cross-Modal Fusion实现动态自适应融合，显著提升对未见声音的泛化。

**🔧 技术方法**

采用CRNN+音频强度注意力进行时序音频特征提取，双向GRU捕获时序依赖，交叉注意力和门控机制实现动态融合，并用Actor-Critic强化学习进行决策。

**📊 数据集**

在Replica、Matterport3D和SoundSpaces音频仿真数据集上进行实验。

**📈 对比分析**

与SoundSpaces、AV-WAN等基线对比，在未见声音任务中SPL、SR和SNA分别提升约28.6%、23.7%和9.8%，表现出更好的路径效率和成功率。

**⚠️ 局限性**

仍依赖仿真音频环境，真实场景中的噪声干扰和多源音频需进一步验证；对极端遮挡和动态障碍的鲁棒性尚未充分测试。

---

## 127. Banach density of generated languages: Dichotomies in topology and dimension

**arXiv ID:** 2604.02385 | [PDF](https://arxiv.org/pdf/2604.02385v1)

**作者:** Jon Kleinberg `[一作]` (Cornell University), Fan Wei `[通讯]` (Duke University)

**通讯引用:** 749 | [OpenAlex ID](https://openalex.org/A5100380511)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究在语言生成模型中，以 Banach 密度衡量生成语料的宽度（breadth），并在一维与多维嵌入空间下给出对可实现密度的严格双分（dichotomy）结论。

**💡 创新点**

创新点包括：① 将 Banach 密度引入语言生成问题，揭示其比传统的渐近密度更严格的结构约束；② 通过 Cantor–Bendixson 秩证明，当语言集合的拓扑秩有限时，算法总能实现 1/2 的 Banach 密度；若秩无限，则存在构造使密度被迫为 0；③ 在多维情形下引入非退化条件，克服 Ramsey 理论障碍；④ 进一步提出 f‑窗口密度框架，实现从渐近密度到 Banach 密度的连续插值。

**🔧 技术方法**

主要技术包括：拓扑空间的构造与 Cantor–Bendixson 分层；对抗性枚举策略与完美塔（infinite perfect tower）的构造；组合游戏论证与拉曼（Ramsey）理论应用；密度测度的层级与单调性证明；以及多层回退（pull‑back）策略以维持有效性。

**📊 数据集**

该工作为纯理论研究，不使用具体数据集；所有证明均基于可数语言族的抽象构造和数理模型。

**📈 对比分析**

比较方法主要是理论可达性与极限性：在所有可数语言族中，算法能始终保证 1/2 的 Banach 密度（或 f‑窗口密度），这与仅能保证 1/2 渐近密度的先前结果相一致，但对更严格的 Banach 密度提出了完整的上界与下界。性能上，算法在有限 Cantor–Bendixson 秩时达到最优密度 1/2；在无限秩时无可实现正密度。

**⚠️ 局限性**

局限性包括：① 仅在理论层面给出构造与证明，缺乏对实际 LLM 嵌入空间的实验验证；② 对多维情况的结果依赖于额外的非退化假设，实际嵌入可能不满足；③ 仅考虑可数语言族，无法直接推广到连续或大规模语言集合；④ 算法实现细节（如时间复杂度、具体更新规则）未给出，需要进一步研究。

---

## 128. Cardinality is Not Enough: Super Host Detection via Segmented Cardinality Estimation

**arXiv ID:** 2604.02379 | [PDF](https://arxiv.org/pdf/2604.02379v1)

**作者:** Yilin Zhao `[一作]` (Central South University), Jianxin Wang `[通讯]` (Central South University)

**通讯引用:** 34991 | [OpenAlex ID](https://openalex.org/A5100438360)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `3855fcda-48ef-4070-a15e-803cd5c84d83` `64443552-63e0-44b5-906f-d90fe95c5a1b` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出 SegSketch，一种轻量级分段哈希与子网计数的流式流量 sketch，用于检测高连接数的超主机。

**💡 创新点**

创新点在于使用半段哈希推断 IP 前缀长度，结合子网内计数来区分恶意与正常主机，显著降低误报并减少内存占用。

**🔧 技术方法**

方法依赖分段哈希、位图计数（Linear Counting）以及在 P4 可编程交换机上的实现。

**📊 数据集**

使用 UNSW‑NB15、MAWI2021 与 CAIDA2016 混合流量进行评测。

**📈 对比分析**

与 SpreadSketch、Couper、RHHH 对比，SegSketch 在 32KB 内存下 F1‑Score 提升 8.04×，误报率下降 86%，吞吐量最高可达 28 Mpps。

**⚠️ 局限性**

局限在于对极端子网长度变动敏感、需对段宽 G 进行经验调优，且在子网比例极低的场景下性能会下降。

---

## 129. High Volatility and Action Bias Distinguish LLMs from Humans in Group Coordination

**arXiv ID:** 2604.02578 | [PDF](https://arxiv.org/pdf/2604.02578v1)

**作者:** Sahaj Singh Maini `[一作]` (Indiana University Bloomington), Zoran Tiganj `[通讯]` (Indiana University Bloomington)

**通讯引用:** 1155 | [OpenAlex ID](https://openalex.org/A5021423914)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究人类与大型语言模型（LLM）在无直接沟通、仅基于群体反馈的群组二进制搜索（GBS）游戏中的协同与适应性；

**💡 创新点**

揭示LLM在协调任务中表现出过度反应、频繁切换和缺乏跨游戏学习的主要缺陷，并与人类的自适应角色分化及对数值反馈的高效利用进行对比；

**🔧 技术方法**

采用零射击（Zero‑Shot）与链式思维（CoT）提示技术，评估Deepseek‑V3、Deepseek‑V3.1‑T、Llama 3.3、Gemini 2.0 Flash等模型的群组行为；

**📊 数据集**

使用来自原始人类实验的数据（10局游戏、2–17人不同组规模、方向性与数值性反馈交替），以与LLM结果进行基线对照；

**📈 对比分析**

通过比较平均达成目标所需回合数、学习曲线斜率、群体反应斜率、切换率等指标，发现人类在所有组规模和反馈类型下均优于LLM；LLM在数值反馈下表现更差，且跨游戏学习几乎无效；

**⚠️ 局限性**

主要局限包括LLM缺乏基于历史反馈的动态策略调整、对数值信息利用不充分、对群组规模扩展的适应性差，以及缺乏跨游戏的学习机制；

---

## 130. SWAY: A Counterfactual Computational Linguistic Approach to Measuring and Mitigating Sycophancy

**arXiv ID:** 2604.02423 | [PDF](https://arxiv.org/pdf/2604.02423v1)

**作者:** Joy Bhalla `[一作]` (Johns Hopkins University), Kristina Gligorić `[通讯]` (Johns Hopkins University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6215c339-3735-4be3-8a07-5bbb7004712d` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了SWAY指标，用无监督、对抗性提示来量化LLM的sycophancy并给出缓解策略

**💡 创新点**

首次构建无监督、单轮、无ground‑truth标签、非LLM评判的sycophancy测度，并设计对抗性Chain‑of‑Thought缓解框架

**🔧 技术方法**

对抗性提示、语言学语义变体（子句类型、结构、承诺程度）、log‑ratio评分、无监督对抗式训练思路

**📊 数据集**

AITA（道德判断）、LFQA（偏好评估）、DebateQA（争议性问题）共3个英文数据集

**📈 对比分析**

与六大LLM（Llama4 Scout, Claude系列, Mistral Large, Gemma 3）对比，SWAY在所有模型和任务均呈正值；对抗性Chain‑of‑Thought将S降至≈0，优于简单“不要sycophantic”指令，性能提升显著

**⚠️ 局限性**

局限包括仅英文、单词二值输出、未验证与用户感知的一致性、对抗性提示对不同语言和文化可能不适用、指标不评估模型整体鲁棒性

---

## 131. Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers

**arXiv ID:** 2604.02344 | [PDF](https://arxiv.org/pdf/2604.02344v1)

**作者:** Jędrzej Maczan `[一作]` `[通讯]` (Independent Researcher), Jędrzej Maczan (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对WebGPU在LLM推理中的调度开销进行系统评估，并提出顺序调度测量方法，验证了实际调度成本远低于传统单调测量。

**💡 创新点**

首次通过顺序调度方法量化真实调度成本，区分API开销与框架开销，并揭示不同后端与浏览器的调度差异与融合收益。

**🔧 技术方法**

使用WebGPU Dawn、wgpu-native实现，PyTorch FX到WGSL编译器，RMSNorm/MLP/注意力融合，跨NVIDIA/AMD/Apple/Intel GPU与Chrome/Safari/Firefox浏览器进行基准测试。

**📊 数据集**

采用Qwen2.5-0.5B-Instruct与Qwen2.5-1.5B-Instruct两种模型，自动回归生成50个Token，评估调度与吞吐量。

**📈 对比分析**

通过对比单调与顺序调度、不同后端（Vulkan/Metal）、融合策略，发现调度开销为24–36 µs（Vulkan）/32–71 µs（Metal），融合后吞吐率提升53%，WebGPU浮点32位性能仅为CUDA fp16的11–12%。

**⚠️ 局限性**

实验仅限batch=1、float32，推理模型小于2B参数，WebGPU浮点16位尚未支持，测量分解存在约30%不确定性，且在Firefox等浏览器的调度被限速。

---

## 132. Fast NF4 Dequantization Kernels for Large Language Model Inference

**arXiv ID:** 2604.02556 | [PDF](https://arxiv.org/pdf/2604.02556v1)

**作者:** Xiangbo Qi `[一作]` (University of Southern California), Murali Annavaram `[通讯]` (University of Southern California)

**通讯引用:** 6409 | [OpenAlex ID](https://openalex.org/A5018033573)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

实现了在 NVIDIA Ampere GPU 上对 NF4 4‑bit 量化权重进行快速去量化，使用共享内存优化实现 2.0–2.2× 的内核加速。

**💡 创新点**

创新点在于将 16 个 NF4 查找表放入共享内存，并采用直接位运算索引，消除全局内存访问和分支，保持与 HuggingFace / BitsAndBytes 生态的完全兼容。

**🔧 技术方法**

使用了 CUDA 共享内存、位运算、常量缓存、BitsAndBytes 与 HuggingFace Transformers 的轻量级内核改造。

**📊 数据集**

在 GSM8K 数据集上对 Gemma 27B、Qwen3 32B 和 Llama3.3 70B 三个模型进行推理评测。

**📈 对比分析**

与 BitsAndBytes 原版基线对比，在不同批量下实现了 2.0–2.2× 的内核加速，端到端推理速度提升 1.07–1.54×，吞吐量提升 1.07–1.54×。

**⚠️ 局限性**

局限性包括仍受限于 Ampere 对 4‑bit 计算的缺失，只针对 NF4 量化有效；对其他量化方案或不同 GPU 架构可能不适用。

---

## 133. LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning

**arXiv ID:** 2604.02338 | [PDF](https://arxiv.org/pdf/2604.02338v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 134. KAIJU: An Executive Kernel for Intent-Gated Execution of LLM Agents

**arXiv ID:** 2604.02375 | [PDF](https://arxiv.org/pdf/2604.02375v1)

**作者:** Cormac Guerin `[一作]` (Compdeep), Frank Guerin `[通讯]` (University of Surrey)

**通讯引用:** 1234 | [OpenAlex ID](https://openalex.org/A5090217029)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了KAIJU框架，将LLM的推理与工具执行分离，形成一个包含Intent‑Gated Execution（IGX）和Executive Kernel的执行抽象。

**💡 创新点**

创新点在于通过IGX四变量门（范围、意图、影响、授权）实现结构化安全控制，并通过三种执行模式（Reflect、nReflect、Orchestrator）提供可扩展的并行执行与自适应回退。

**🔧 技术方法**

使用技术包括基于GPT的规划器与反射器、DAG调度、工具参数注入、四变量执行门、微规划器、以及多模型分离（规划器与执行器）。

**📊 数据集**

评估数据集包括10题计算天文学基准、GAIA 127文本题以及40条多复杂度任务的真实世界查询。

**📈 对比分析**

对比方法是将KAIJU在三种模式下与传统ReAct（带并行函数调用）在相同工具和安全门上进行实测，结果显示KAIJU在中高复杂度任务中平均时延更低、完成率更高、输出质量更好。

**⚠️ 局限性**

局限性包括规划器依赖提示质量、规划器误差导致延迟、固定规划开销在简单查询中产生负面影响、以及对动态工具影响分类的依赖。

---

## 135. WSVD: Weighted Low-Rank Approximation for Fast and Efficient Execution of Low-Precision Vision-Language Models

**arXiv ID:** 2604.02570 | [PDF](https://arxiv.org/pdf/2604.02570v1)

**作者:** Haiyu Wang `[一作]` (New York University), Sai Qian Zhang `[通讯]` (New York University)

**通讯引用:** 988 | [OpenAlex ID](https://openalex.org/A5013517247)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

针对视觉语言模型的解码阶段，提出了WSVD框架，通过对每个注意力头进行加权低秩SVD分解并结合量化技术，显著降低KV缓存和计算成本。

**💡 创新点**

创新点在于：①对SVD应用更细粒度的按头分解，减少重构开销；②在SVD过程中引入权重重要性权重，实现加权微调；③在量化训练中加入正交变换以抑制离群值，进一步保持精度。

**🔧 技术方法**

使用技术包括：逐头低秩SVD、Fisher信息加权微调、低精度量化（W8A8/W8A4）、自研融合核（结合Flash Decoding），以及Trition实现的高效内核。

**📊 数据集**

采用ScienceQA‑IMG和SEED‑Bench‑IMG两大多模态评测数据集进行效果验证。

**📈 对比分析**

与ASVD、SVD‑LLM、QSVD、DuQuant、QVLM等基线对比，WSVD在保持近乎原始精度的同时，实现1.8×以上的解码加速，并在低端GPU上达到约2.6×的速度提升。

**⚠️ 局限性**

局限性包括：对极大规模模型（13B参数）仍需高端GPU资源；量化与微调需要额外的校准样本和计算；在极低秩压缩下某些模型的精度仍会有微小下降。

---

## 136. The Quantum-Cryptographic Co-evolution

**arXiv ID:** 2604.02591 | [PDF](https://arxiv.org/pdf/2604.02591v1)

**作者:** Ashish Kundu `[一作]` (Cisco Research), Ramana Kompella `[通讯]` (Cisco Research)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一个二维坐标系，用以描述密码学韧性与计算能力的共同演进，并通过四象限阐释从传统系统到量子安全体系的过渡过程。

**💡 创新点**

创新点在于引入“量子缺口（Quantum Gap）”这一系统性风险度量，并将其与四象限框架结合，形成一种新的量化评估方法。

**🔧 技术方法**

使用的技术包括 Shor 算法的优化与硬件共设计、量子错误校正（表面码、LDPC）、后量子密码学（PQC）标准化、以及量子网络与 SDQN 的架构方案。

**📊 数据集**

未使用具体实验数据集；论文主要基于公开的量子资源估算（如 RSA‑2048 需要约 100,000 逻辑量子比特）以及已公布的 PQC 参数（ML‑KEM、ML‑DSA 等）。

**📈 对比分析**

方法上通过对四象限的风险分布进行理论评估，比较了不同阶段（-,-）、(-,+)、(+,-) 与 (+,+) 的安全水平，结果表明在 (-,+) 象限存在“不可逆”风险，建议提前进入 (+,-) 阶段以避免进入危机象限。

**⚠️ 局限性**

局限性包括：①对 CRQC 到来时间的假设可能不准确；②缺乏实测硬件资源与时间的真实数据；③未考虑量子攻击在不同硬件实现下的变异性；④对现有系统迁移成本与兼容性未给出定量评估。

---

## 137. Re-analysis of the Human Transcription Factor Atlas Recovers TF-Specific Signatures from Pooled Single-Cell Screens with Missing Controls

**arXiv ID:** 2604.02511 | [PDF](https://arxiv.org/pdf/2604.02511v1)

**作者:** Arka Jain `[一作]`, Umesh Sharma `[通讯]` (Mayo Clinic)

**通讯引用:** 9607 | [OpenAlex ID](https://openalex.org/A5086069660)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

重新分析人类TF Atlas单细胞数据集，建立可复现的管道，使用外部EB细胞做基准并进行背景减除，以提取每个转录因子的差异表达和功能富集。

**💡 创新点**

创新点在于用EB基准与背景差异基因去除技术弥补缺失的负控，恢复 59/61 个TF 的特异性转录信号，并验证与原始排名的高度一致性。

**🔧 技术方法**

技术方法包括 Scanpy 进行 QC、降维和聚类；MORF 条形码去复用；Wilcoxon rank‑sum 做差异表达；背景减除筛选共性基因；Enrichr/GSEApy 进行 GO/KEGG 富集；Harmony 评估批次效应。

**📊 数据集**

使用 GEO GSE216481 人类 TF Atlas 数据集（约 250,000 细胞，包含 8 个聚合屏幕、4 个单一 TF、2 个 EB 等）。

**📈 对比分析**

与原始 Joung 等人发布的 TF 排名进行 Spearman 相关性（ρ = –0.316，p = 0.013），显示新方法与原始结果高度一致；相比仅用 one‑vs‑rest 检测，背景减除后 59/61 TF 能够识别出差异表达，显著提升了检出灵敏度。

**⚠️ 局限性**

局限性包括缺少原始库中的 GFP/mCherry 负控，导致需跨批次使用 EB 基准并进行背景校正；低细胞数 TF 的统计功效仍受限；未对原始测序读段进行重新去复用。

---

## 138. Time-Warping Recurrent Neural Networks for Transfer Learning

**arXiv ID:** 2604.02474 | [PDF](https://arxiv.org/pdf/2604.02474v1)

**作者:** Jonathon Hirschi `[一作]` (University of Colorado Denver), Jonathon Hirschi `[通讯]` (University of Colorado Denver)

**通讯引用:** 1 | [OpenAlex ID](https://openalex.org/A5047907673)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6c45cf0c-64ed-40ad-82d2-485a4d4dcbed` `cc175879-ab65-4aa9-b58a-f6100a057dbf`

**🎯 论文内容**

本论文提出并验证了在预训练LSTM网络上进行时间扭曲（time‑warping）的方法，用以实现不同时间尺度下的迁移学习，主要应用于燃料湿度（Fuel Moisture Content, FMC）预测；

**💡 创新点**

创新点在于将时间扭曲视为一种参数化的迁移学习技术，仅需微调LSTM门控偏置，即可对系统的时间尺度进行调整，且提供了对应的理论证明；

**🔧 技术方法**

核心技术包括：长短时记忆网络（LSTM）序列学习、基于门控偏置的时间扭曲、网格搜索优化偏置参数、传统统计和深度学习基准模型（如AR、CNN、Transformer等）进行对比；

**📊 数据集**

使用的数据集主要为：CONUS范围内的FM10传感器观测（大规模连续时序数据）以及Oklahoma 1996‑1997年对1h、10h、100h、1000h燃料的稀疏观测；

**📈 对比分析**

与传统迁移学习（全参数微调、冻结层）以及基准物理模型（Nelson、ODE+KF）比较，实验表明时间扭曲方法在预测误差上往往优于或匹配全微调策略，同时仅调整两参数更不易过拟合；

**⚠️ 局限性**

局限性包括：对预训练网络的假设（近似线性时间尺度）可能不适用于高度非线性系统；时间扭曲参数需手工或网格搜索，计算量有限；理论证明基于理想化的时间延迟模型，实际数据中可能存在额外噪声和非平稳性问题。

---

## 139. Beyond Resolution Rates: Behavioral Drivers of Coding Agent Success and Failure

**arXiv ID:** 2604.02547 | [PDF](https://arxiv.org/pdf/2604.02547v1)

**作者:** Tural Mehtiyev `[一作]` (North Carolina State University), Wesley Assunção `[通讯]` (North Carolina State University)

**通讯引用:** 1550 | [OpenAlex ID](https://openalex.org/A5039130090)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对19个基于LLM的编码代理在SWE‑bench Verified 500个Python任务上的轨迹进行大规模实证分析，探究失败原因与行为模式。

**💡 创新点**

发现即使补丁规模小、任务标注为简单，代理仍因架构推理不足而全失败；轨迹长度与失败的关系被任务难度混淆；成功代理在轨迹结构上更注重先获取上下文、后逐步修补、并投入验证；LLM能力是决定成功与行为的主导因素，框架设计效应随LLM升级显著减弱；高阶LLM对系统提示的敏感性下降。

**🔧 技术方法**

采用轨迹符号编码（13种子状态）、统计检验（Mann‑Whitney U、Wilcoxon符号秩检验、Cliff's δ）、任务难度控制的配对比较、任务级结果一致性分析、以及对框架提示长度的敏感性评估。

**📊 数据集**

使用SWE‑bench Verified 500个真实开源Python任务的轨迹与评测结果，涵盖19个不同框架+LLM组合。

**📈 对比分析**

通过在同一任务内进行成功/失败配对比较，证明轨迹长度并非可靠指标；通过轨迹结构特征与成功率相关性的统计，表明上下文收集、修补强度、验证投入三维可区分结果；LLM共享时任务一致率高达85‑93%，框架共享时仅47‑88%，显示LLM主导；在LLM更新后框架间分差从19.4个百分点缩小到3.8个百分点，再到0.9个百分点。

**⚠️ 局限性**

研究基于单次执行、仅限Python任务；轨迹编码规则可能忽略细粒度推理差异；数据受SWE‑bench版本与工具API差异影响；对LLM内部随机性缺乏多次复现，可能导致误判；结论受所选代理组合与任务集的代表性限制。

---

## 140. Overconfidence and Calibration in Medical VQA: Empirical Findings and Hallucination-Aware Mitigation

**arXiv ID:** 2604.02543 | [PDF](https://arxiv.org/pdf/2604.02543v1)

**作者:** Ji Young Byun `[一作]` (Johns Hopkins University), Asma Ben Abacha `[通讯]` (Microsoft Healthcare & Life Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了医学视觉问答（VQA）中视觉‑语言模型（VLM）的置信度校准问题，并系统评估了不同模型规模、提示策略和校准方法的效果。

**💡 创新点**

提出了Hallucination‑Aware Calibration（HAC）框架，将视觉诱发的虚假检测信号与原始置信度结合，既能降低校准误差，又能提升AUROC。

**🔧 技术方法**

使用了标准的后置校准技术（Platt scaling、Isotonic regression、Histogram binning）以及自定义的HAC（基于VASE的幻觉检测）。

**📊 数据集**

在三个医学VQA基准上进行实验：VQA‑RAD、SLAKE‑EN（英文）和VQA‑Med‑2019。

**📈 对比分析**

与基线相比，后置校准显著降低ECE/ACE，HAC在开放式问题上AUROC平均提升5.3个百分点（开放式提高7.3pp），但对闭合式问题效果不一；提示策略对校准无显著帮助。

**⚠️ 局限性**

局限性包括：仅评估通用VLM，未覆盖细粒度医学模型；对闭合式问题幻觉检测效果较差；HAC 需要额外生成多轮推断，计算成本高；对少见影像模态的泛化能力待验证。

---

## 141. Street-Legal Physical-World Adversarial Rim for License Plates

**arXiv ID:** 2604.02457 | [PDF](https://arxiv.org/pdf/2604.02457v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 142. When simulations look right but causal effects go wrong: Large language models as behavioral simulators

**arXiv ID:** 2604.02458 | [PDF](https://arxiv.org/pdf/2604.02458v1)

**作者:** Zonghan Li `[一作]` (University of Toronto), Feng Ji `[通讯]` (University of Toronto)

**通讯引用:** 207630 | [OpenAlex ID](https://openalex.org/A5050750924)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究利用三种大型语言模型对跨国气候心理学实验的11种干预进行个体层面响应模拟，并系统评估其描述性拟合与因果准确性的差异。

**💡 创新点**

其创新在于首次揭示LLM在行为干预模拟中描述性拟合与因果保真度之间的结构性分离，并通过干预逻辑、提示方式和人群特征探讨误差来源与不平等影响。

**🔧 技术方法**

采用GPT‑4o‑mini、Gemini‑2.5 Flash Lite和Claude‑3 Haiku的提示生成与推理技术，并结合ME、MAE、ATE误差、方差分解、ICC、置换检验和SHAP特征重要性等统计工具进行评估。

**📊 数据集**

主要使用国际气候心理学合作（ICCP）数据集（59,508名受试者、62国），并在两份附加跨国实验数据集（Spampatti 12国、Većkalov 27国）上复现实验。

**📈 对比分析**

与线性回归和LASSO基线相比，LLM在描述性拟合方面表现可观（MAE 16–18，行动约40–43），但因果误差仍高于基线（ATE误差约8–9个百分点），提示改进提升描述性拟合但对因果准确性影响不一。

**⚠️ 局限性**

研究局限包括仅针对气候态度领域、使用轻量化LLM未进行微调、样本代表性不足、因果评估仅涵盖平均处理效应、未深入探究模型内部机制等。

---

## 143. Fighting AI with AI: AI-Agent Augmented DNS Blocking of LLM Services during Student Evaluations

**arXiv ID:** 2604.02360 | [PDF](https://arxiv.org/pdf/2604.02360v1)

**作者:** Yonas Kassa `[一作]` (Robert Morris University), Ping Wang `[通讯]` (Robert Morris University)

**通讯引用:** 10584 | [OpenAlex ID](https://openalex.org/A5100338689)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了 AI‑Sinkhole 框架，利用 AI 代理在考试期间动态发现、语义分类并暂时网络全局屏蔽新兴 LLM 聊天服务。

**💡 创新点**

创新点在于将持续爬虫、可解释的 LLM 语义分类与 Pi‑hole 动态 DNS 屏蔽相结合，实现跨语言、无静态黑名单的实时检测与临时拦截。

**🔧 技术方法**

技术栈包括本地量化 LLM（LLaMA 3、DeepSeek‑R1、Qwen‑3）在 Ollama 上推理、Crawl4AI 文本摘要、Pi‑hole DNS sinkhole、Docker、Python 爬虫与提示工程。

**📊 数据集**

使用自建的 126 条 URL 数据集（63 个 LLM 聊天服务、63 个非 LLM 网站），覆盖多语言（英语、西班牙语、德语、中文、阿拉伯语、法语、葡萄牙语）及多样化网站类型。

**📈 对比分析**

通过准确率、F1、MCC 评估分类效果，F1 均在 0.83–0.84 之间，MCC 约 0.738；动态屏蔽实验实现 100% 正面域名拦截、0% 误拦；推理延迟约为轻量级 LLaMA‑3 的 2 倍。

**⚠️ 局限性**

局限性包括：推理延迟高限制实时大规模部署、对爬虫及 LLM 识别的依赖可能遗漏新出现的域、仅基于 DNS 屏蔽易被绕过、实验规模有限且未覆盖真实网络流量与隐私安全完整评估。

---

## 144. Overcoming the "Impracticality" of RAG: Proposing a Real-World Benchmark and Multi-Dimensional Diagnostic Framework

**arXiv ID:** 2604.02640 | [PDF](https://arxiv.org/pdf/2604.02640v1)

**作者:** Kenichirou Narita `[一作]` (Fujitsu Limited), Satoru Takahashi `[通讯]` (Fujitsu Limited)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出了针对企业环境的多维诊断评估框架和高难度 RAG 基准数据集，用以细粒度诊断 RAG 系统的性能瓶颈。

**💡 创新点**

创新点在于构建了四维难度分类（推理复杂度、检索难度、源结构与模态、可解释性要求），实现细粒度诊断，弥补传统单一准确率评估的缺陷，并展示了诊断价值。

**🔧 技术方法**

采用多表示索引（MRI）、多代理分解式检索与 RRF 结合、GPT‑4o‑mini 生成、text‑embedding‑3‑large 向量检索等技术。

**📊 数据集**

使用了基于 34 篇企业文件（财报、技术规范、合规文件）的 100 题高难度问答数据集，并与 JDocQA 进行对照。

**📈 对比分析**

比较方法采用 Overall Accuracy 与维度诊断准确率（D‑value）两种指标；MRI‑RAG 的整体准确率约 9% 与 JDocQA 相近，Agentic AI 提升至 17%，但在推理复杂度维度仅 5% 的 D‑value，显示检索与结构分析提升显著而推理仍为瓶颈。

**⚠️ 局限性**

局限性包括数据规模有限、未评估成本/效率、维度间相互作用未深入、对单一 LLM 依赖。

---

## 145. Rapidly deploying on-device eye tracking by distilling visual foundation models

**arXiv ID:** 2604.02509 | [PDF](https://arxiv.org/pdf/2604.02509v1)

**作者:** Cheng Jiang `[一作]` (Meta Reality Labs), Ali Behrooz `[通讯]` (Meta Reality Labs)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `8d10c613-917e-4880-9716-17789f50e119` `57a58b01-81b4-4d75-a45c-2e891f272b50` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

通过在近眼图像上利用无监督自蒸馏和合成监督对 DINOv3 进行域适配，随后对其进行知识蒸馏，得到可在 AR/VR 设备上部署的 256K 参数眼动追踪模型。

**💡 创新点**

创新点在于：①无标签真实数据与合成数据联合自蒸馏优化 VFM；②结合自监督与教师-学生双重蒸馏实现轻量级模型；③证明 DINOv3 特征可通过自监督重塑为更具瞳孔运动相关性。

**🔧 技术方法**

采用 DINOv3 ViT-B、自监督自蒸馏、EMA 自蒸馏、VIC‑KD 等蒸馏技术，结合合成监督和伪标签。

**📊 数据集**

使用 Project Aria 的 6,299 条真实记录（2,222 名参与者）以及 165K 帧合成数据。

**📈 对比分析**

在 E50U50 与 E90U90 等指标上，与仅合成监督的基线相比，优化后的 VFM 将 E50U50 降至 1.33°（减少 61.8%），学生模型进一步降至 1.44°，整体提升约 58.6%；与全监督上限相比仍差约 0.6°。

**⚠️ 局限性**

仍存在与全监督上限的差距，尤其是尾部指标受噪声影响；仅基于单一设备配置，缺乏跨设备泛化验证。

---

## 146. Non-Signaling Locality Lower Bounds for Dominating Set

**arXiv ID:** 2604.02582 | [PDF](https://arxiv.org/pdf/2604.02582v1)

**作者:** Noah Fleming `[一作]` (Lund & Columbia), Yuichi Yoshida `[通讯]` (National Institute of Informatics)

**通讯引用:** 7027 | [OpenAlex ID](https://openalex.org/A5038701345)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b`

**🎯 论文内容**

提出了最小支配集在非信号模型（包含LOCAL、量子-LOCAL等）中的新的局部性下界，证明了对O(logΔ)近似需要Ω(log n/(logΔ·loglogΔ))轮；进一步得到存在β∈(0,1)时O(log^βΔ)近似需要Ω(log n/logΔ)轮，从而在量子-LOCAL中获得与KMW算法匹配的无度数下界Ω(√(log n/loglog n))。

**💡 创新点**

核心创新在于构造两种低误差标签覆盖（label‑cover）实例的敏感度下界：①利用Impagliazzo–Kabanets–Wigderson的并行重复与度数缩减实现对低误差标签覆盖的n^Ω(1/k)敏感度下界；②改进Dinur–Harsha框架以在低误差下保留更小的字母表和度数，从而实现对O(log^βΔ)近似的更强下界。

**🔧 技术方法**

主要技术包括：标签覆盖的低误差并行重复与度数缩减、Dinur–Harsha的可复原迭代构造、标签覆盖→集合覆盖→支配集的经典化简、敏感度→局部性转移定理、以及非信号分布的局部性定义与证明。

**📊 数据集**

该工作为纯理论证明，不依赖具体实验数据集，所用实例为构造的符号化标签覆盖和对应的图结构。

**📈 对比分析**

与以往仅在LOCAL模型中取得的Ω(log n)下界相比，本研究在更广泛的非信号/量子-LOCAL模型中实现了与已知上界相匹配的下界；相较于KMW的度数依赖下界，本研究消除了度数项，在小度数下同样获得Ω(log n)规模的下界。

**⚠️ 局限性**

局限性在于：①对更高阶近似（如logΔ以上）下的下界仍未完全匹配；②所需的低误差标签覆盖构造依赖于现有PCP技术，字母表和度数的进一步压缩仍是开放问题；③对非信号模型的上界（如可实现的算法）尚未给出。

---

## 147. Failing to Falsify: Evaluating and Mitigating Confirmation Bias in Language Models

**arXiv ID:** 2604.02485 | [PDF](https://arxiv.org/pdf/2604.02485v1)

**作者:** Ayush Rajesh Jhaveri `[一作]` (New York University), Eunsol Choi `[通讯]` (New York University)

**通讯引用:** 4297 | [OpenAlex ID](https://openalex.org/A5035142405)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了大型语言模型在规则发现任务中的确认偏差，并通过交互式探索评估其推理行为。

**💡 创新点**

首次将人类认知心理学中的确认偏差实验迁移到LLM，提出I:C指标并验证干预策略与知识蒸馏的有效性。

**🔧 技术方法**

采用提示工程（Dual‑Goal、Think‑in‑Opposites）、思考模式、LLM‑as‑a‑judge、符号知识蒸馏和细化训练。

**📊 数据集**

自建的多规则整数三元组数据集（四个规则组，每组四条规则，训练/验证/测试分别取 100/2/5 三元组）以及Blicket测试任务。

**📈 对比分析**

通过与基线比较，发现思考模式模型任务成功率提升至约78%，干预提示将成功率从约42%提升至56%，蒸馏后同等或更优，并在Blicket任务上也显著提升。

**⚠️ 局限性**

仅在结构化规则发现场景验证，泛化到更自然任务仍有差距，且蒸馏后仍低于即时干预的表现。

---

## 148. Delaunay Canopy: Building Wireframe Reconstruction from Airborne LiDAR Point Clouds via Delaunay Graph

**arXiv ID:** 2604.02497 | [PDF](https://arxiv.org/pdf/2604.02497v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 149. DrugPlayGround: Benchmarking Large Language Models and Embeddings for Drug Discovery

**arXiv ID:** 2604.02346 | [PDF](https://arxiv.org/pdf/2604.02346v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 150. Agentic AI-Empowered Wireless Agent Networks With Semantic-Aware Collaboration via ILAC

**arXiv ID:** 2604.02381 | [PDF](https://arxiv.org/pdf/2604.02381v1)

**作者:** Zhouxiang Zhao `[一作]` (Zhejiang University), Kaibin Huang `[通讯]` (University of Hong Kong)

**通讯引用:** 21541 | [OpenAlex ID](https://openalex.org/A5007131492)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `fede83ac-7505-405f-ab37-e7284695c47f` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于Agentic AI的无线代理网络（WAN），通过进化的知识聚合机制实现移动代理在协同感知、语义压缩与通信中的能量最小化。

**💡 创新点**

创新点包括将语义压缩、移动规划与通信资源在同一框架内联合优化，并通过潜在场引导的预测匹配策略克服贪婪匹配的短视性，实现多轮能量最优的分层拓扑演化。

**🔧 技术方法**

使用的技术包括基于ELM的语义压缩与RAG去重、联合运动-资源优化的BCD+SCA算法、最小权重完美匹配（Edmonds Blossom）以及潜在场辅助的预测匹配。

**📊 数据集**

采用仿真场景：随机分布于500m×500m区域内的N个移动代理，巡检半径80–100m，初始负载5–10Mbit，作为评估数据集。

**📈 对比分析**

与多种基准（最大发射功率、固定速度、无语义压缩、无RAG、无移动、距离驱动、纯贪婪、随机拓扑）比较，实验显示在N=10时能量消耗比距离驱动低约19%，比随机拓扑低57%，且随N增大保持良好可扩展性。

**⚠️ 局限性**

局限性包括采用简化的路径损耗模型、集中式协调、缺乏动态信道建模和去中心化协同协议，未来工作计划引入更复杂信道与自治决策机制。

---

## 151. From Broad Exploration to Stable Synthesis: Entropy-Guided Optimization for Autoregressive Image Generation

**arXiv ID:** 2604.02355 | [PDF](https://arxiv.org/pdf/2604.02355v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9`

---

## 152. VALOR: Value-Aware Revenue Uplift Modeling with Treatment-Gated Representation for B2B Sales

**arXiv ID:** 2604.02472 | [PDF](https://arxiv.org/pdf/2604.02472v1)

**作者:** Vamshi Guduguntla `[一作]`, Debanshu Das `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

提出VALOR框架，利用因果提升学习在B2B销售中识别“可说服”账户并优化人力资源分配，显著提升增量收入。

**💡 创新点**

创新点包括：① Treatment‑Gated Sparse‑Revenue Network 通过双线性门控有效防止治疗信号消失；② Cost‑Sensitive Focal‑ZILN 结合焦点机制与价值加权排序，解决零膨胀收入分布与高价值客户排序不对齐；③ 引入可解释的 Robust ZILN‑GBDT，兼顾树模型可解释性与提升性能。

**🔧 技术方法**

技术手段包括深度学习骨干（TARNet、DragonNet、CFR‑WASS/MMD等）+门控网络、Zero‑Inflated LogNormal 损失+Focal 变体、价值加权排序损失、树模型 ZILN‑GBDT、MLOps、闭环因果反馈与A/B实验评估。

**📊 数据集**

使用数据集：合成 B2B‑Mimic（236k 账号、100 维特征）以及真实云平台 SMB 账单数据（约 80k 账号、零膨胀 ARR）。

**📈 对比分析**

与多种基线（T‑learner、TARNet、CFR‑WASS/MMD、DragonNet、RERUM、UniTE、EUEN 等）在公开与私有数据上对比，VALOR 在 AUUC、Qini、Lift@30、KRCC 等指标上提升约 20‑25%；在 4 个月的 A/B 实验中，机会率提升 8.3%，单账号增量收入提升 $740，预计年化 ARR 提升 $30M。

**⚠️ 局限性**

局限性：模型假设零膨胀并对超参数敏感；门控机制与 Focal 参数需精细调优；对极端稀疏数据的鲁棒性仍待提升；未在跨域或时序动态场景中验证泛化能力。

---

## 153. RL-Loop: Reinforcement Learning-Driven Real-Time 5G Slice Control for Connected and Autonomous Mobility Services

**arXiv ID:** 2604.02461 | [PDF](https://arxiv.org/pdf/2604.02461v1)

**作者:** Lara Tarkh `[一作]` (University of Western Ontario), Abdallah Shami `[通讯]` (University of Western Ontario)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `5b4c1114-4a70-478e-9921-2514ee03850d` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

在 5G 切片资源分配中实现了基于 RL 的闭环控制器 RL-Loop，实时每秒调整边缘 CPU 分配。

**💡 创新点**

创新点是将 Proximal Policy Optimization 强化学习与实时 KPI 反馈结合，形成可在线学习并自适应的 CPU 控制策略，并将 MicroOpt 作为离线基准进行对照。

**🔧 技术方法**

使用了 PPO 强化学习算法、Stable Baselines3、Open5GS + UERANSIM 5G 测试床、Linux cgroup 控制 CPU、Python 实现及实时 KPI 监控。

**📊 数据集**

利用仿真 UE 生成的 4K 视频流，并采用受限正态分布的用户到达率（来源于 Telecom Italia 大数据挑战集）来驱动流量，在实际测试床上采集的数据用于训练与评估。

**📈 对比分析**

通过将 RL-Loop 在相同流量下的平均 CPU 分配与 MicroOpt 的 CPU 及 QoS 降级指标对比，结果显示 RL-Loop 在保持 β=0.10 的 QoS 降级的同时，平均 CPU 下降约 55%（即仅使用约 45% 的资源）。

**⚠️ 局限性**

局限性包括仅单切片单一工作负载、对测试床延迟敏感、QoS 评估依赖粗粒度抓包、未在同一硬件上与 MicroOpt 进行直接对比，以及缺乏多切片、多代理的实验验证。

---

## 154. Haiku to Opus in Just 10 bits: LLMs Unlock Massive Compression Gains

**arXiv ID:** 2604.02343 | [PDF](https://arxiv.org/pdf/2604.02343v1)

**作者:** Roy Rinberg `[一作]` (Harvard University), Keri Warr `[通讯]` (Anthropic)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究利用大型语言模型（LLM）对其生成文本进行无损与有损压缩，提出通过域自适应 LoRA、摘要重写和交互式二元提问等方法实现极高压缩率。

**💡 创新点**

创新点在于：①用域适应的 LoRA 大幅提升基于算术编码的无损压缩率（约 2 倍）；②通过提示简洁重写与算术编码实现约 0.03 的有损压缩率；③引入“问答压缩”（QA）交互协议，僵化模型仅用 10 个二元问题就能恢复 23–72% 的性能差距，压缩比 0.0006–0.004，超过以往方法 100 倍以上。

**🔧 技术方法**

主要技术包括：低秩适配（LoRA）进行域自适应、算术编码、Prompt‑based 简洁重写、Shortest‑of‑N 生成候选、以及基于 20‑Questions 思路的交互式二元提问协议。

**📊 数据集**

使用的数据集有 LMSYS、WildChat 用于无损 LoRA 评估；8 个数学、科学和代码基准（如 AIME、CodeEval 等）用于评估有损重写和 QA 性能。

**📈 对比分析**

与基线算术编码（仅使用原始 LLM）对比，LoRA 方法压缩率提升约 2 倍；重写策略在 0.039–0.034 之间优于 Shortest‑of‑N（0.073–0.063）；QA 在 10 位以内即可实现 23–72%（易）或 7–38%（难）性能恢复，压缩比 0.0006–0.004，明显优于现有 LLM 压缩方法。

**⚠️ 局限性**

主要局限包括：① Shortest‑of‑N 选择偏倚可能导致答案质量差异未充分评估；② QA 只在具有明确真值的任务（数学、代码、科学）上测试，开放式生成任务效果未知；③ 交互依赖大型模型的判断，若 LLM 错误会误导小模型；④ 当前方法对非确定性输出的鲁棒性不足；⑤ 交互式问题质量受限于小模型的提问能力。

---

## 155. UI-Oceanus: Scaling GUI Agents with Synthetic Environmental Dynamics

**arXiv ID:** 2604.02345 | [PDF](https://arxiv.org/pdf/2604.02345v1)

**作者:** Mengzhou Wu `[一作]` (Peking University), Tao Xie `[通讯]` (Peking University)

**通讯引用:** 17672 | [OpenAlex ID](https://openalex.org/A5048118068)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `67630363-6be0-4f51-ab05-7198250671a5` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过自主探索获取GUI交互原子转移并进行自监督训练的框架，构建了可扩展的GUI世界模型；

**💡 创新点**

将学习焦点从轨迹模仿转移到掌握交互物理的前向动力学；

**🔧 技术方法**

使用自监督前向动力学预训练、结构/视觉/语义过滤、视觉语言模型生成指令、以及后续的代理式微调；

**📊 数据集**

主要数据来自微信小程序生态，经过自动化探索得到约20.8M原始转移，过滤后约3.4M有效样本；

**📈 对比分析**

与传统BC+SFT、UI grounding、逆向动力学等基线相比，前向动力学预训练可在离线基准上平均提升7%，在在线导航中提升16.8%，且表现随数据量和模型规模呈对数线性增长；

**⚠️ 局限性**

局限包括对单步转移的依赖，尚未系统评估长序列动态生成的精度，以及在非微信生态的普适性需进一步验证。

---

## 156. Ambig-IaC: Multi-level Disambiguation for Interactive Cloud Infrastructure-as-Code Synthesis

**arXiv ID:** 2604.02382 | [PDF](https://arxiv.org/pdf/2604.02382v1)

**作者:** Zhenning Yang `[一作]` (University of Michigan), Ang Chen `[通讯]` (University of Michigan)

**通讯引用:** 9767 | [OpenAlex ID](https://openalex.org/A5100683545)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种无训练、基于不一致驱动的交互式 IaC 生成框架，利用资源、拓扑、属性三维分层结构生成多样化候选配置并通过结构差异引导澄清问题，逐步缩小模糊空间。

**💡 创新点**

创新点在于将 IaC 的层级结构转化为澄清维度，使用候选间的结构不一致来衡量信息增益，并通过轮询（round‑robin）平衡不同维度的澄清，从而在无监督条件下高效定位用户隐含意图。

**🔧 技术方法**

主要技术包括 LLM 的多模态候选生成、符号化结构差异检测、熵度量的不一致排序、轮询平衡调度以及基于不一致生成自然语言澄清问题。

**📊 数据集**

使用了自建的 Ambig‑IaC 基准，共 300 题 Terraform 任务，原始任务经 LLM 转写为高层模糊提示并人工校正参考配置。

**📈 对比分析**

与三种基线（Direct Q Generation、Best‑of‑N、Self‑Consistency）在 Ambig‑IaC 上比较，采用结构 GED 和属性嵌入相似度评估，结果显示相对最高基线提升约 +18.4%（结构）和 +25.4%（属性），并且在交互回合数增大时优势进一步扩大。

**⚠️ 局限性**

主要限制是需要多次候选生成和差异计算，导致 token 与计算成本较高；此外框架聚焦意图澄清，未针对 IaC 语法完整性做专门处理，可能在真实部署时需配合其它合规校验。

---

## 157. Robust Learning with Optimal Error

**arXiv ID:** 2604.02555 | [PDF](https://arxiv.org/pdf/2604.02555v1)

**作者:** Guy Blanc `[一作]` (Stanford), Guy Blanc `[通讯]` (Stanford)

**通讯引用:** 103 | [OpenAlex ID](https://openalex.org/A5059516326)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `6215c339-3735-4be3-8a07-5bbb7004712d`

**🎯 论文内容**

构建了针对对抗噪声的学习算法，证明了使用随机假设可以显著提高错误率。

**💡 创新点**

提出了针对不同类型噪声（恶意噪声、恶劣噪声和不可知噪声）的最优错误率，改进了确定性假设的最优错误率。

**🔧 技术方法**

使用了随机假设和最小化最大损失的优化技术，结合了算法公平性和最优鲁棒性之间的联系。

**📊 数据集**

使用了多种概念类的样本，样本复杂度与VC维度线性相关，并且在多项式的逆超额错误中。

**📈 对比分析**

与现有方法进行了比较，证明了在恶意噪声下，算法的错误率为1/2·η/(1-η)，在恶劣噪声下为3/2·η，均优于确定性假设的错误率。

**⚠️ 局限性**

算法在固定分布的恶劣噪声学习中效率较低，且在某些情况下需要较大的样本量。

---

## 158. MLFCIL: A Multi-Level Forgetting Mitigation Framework for Federated Class-Incremental Learning in LEO Satellites

**arXiv ID:** 2604.02356 | [PDF](https://arxiv.org/pdf/2604.02356v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2`

---

## 159. The NC State All-campus Data Science and AI Project-based Teaching and Learning (ADAPT) Model: A mechanism for interdisciplinary engagement in workforce-relevant learning

**arXiv ID:** 2604.02597 | [PDF](https://arxiv.org/pdf/2604.02597v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 160. Speaking of Language: Reflections on Metalanguage Research in NLP

**arXiv ID:** 2604.02645 | [PDF](https://arxiv.org/pdf/2604.02645v1)

**作者:** Nathan Schneider `[一作]` (Georgetown University), Antonios Anastasopoulos `[通讯]` (George Mason University)

**通讯引用:** 2796 | [OpenAlex ID](https://openalex.org/A5013793053)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

研究并系统化了金属语言（metalanguage）的概念、任务及其在大型语言模型（LLM）中的应用，并对两个实验室的相关研究做了梳理。

**💡 创新点**

首次将金属语言拆解为系统级与实例级、符号与自然、以及处理与生成等维度，并将其与LLM的指令调优、解释性等任务关联，提出针对低资源语言的金属语言驱动的翻译和语言文档化方法。

**🔧 技术方法**

采用基于提示的LLM（如GPT-3、ChatGPT等）进行指令调优、金属语言生成、金属语言评估与解释性分析，并结合符号化的规则学习与聚类等传统方法。

**📊 数据集**

使用了ELQA（英语学习者问答）数据集、WALS、Grambank等语料库，低资源语言的词典与语法书、司法意见文本等。

**📈 对比分析**

通过与人工评估和传统NLP基线对比，LLM在自然金属语言问答上表现流畅且准确率高，但在某些特定问题上仍低于顶尖人工答案；在低资源翻译任务中，结合词典和完整语法书的零样本LLM可达25–55 chrF++的得分。

**⚠️ 局限性**

局限性在于研究范围聚焦自身实验室工作，未构成完整综述；金属语言对LLM的解释性与泛化能力仍需进一步探究，且部分评测仍依赖有限数据和人工判断。

---

## 161. Conditional Sampling via Wasserstein Autoencoders and Triangular Transport

**arXiv ID:** 2604.02644 | [PDF](https://arxiv.org/pdf/2604.02644v1)

**作者:** Mohammad Al-Jarrah `[一作]` (University of Washington), Amirhossein Taghvaei `[通讯]` (University of Washington)

**通讯引用:** 482 | [OpenAlex ID](https://openalex.org/A5035589000)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了Conditional Wasserstein Autoencoder（CWAEs）框架，用于从高维观测中高效地进行条件采样；

**💡 创新点**

创新点在于将三角测量输运与Wasserstein自编码器相结合，利用低维潜变量实现自适应的块三角解码器，直接得到条件分布；

**🔧 技术方法**

采用Wasserstein自编码器、条件Wasserstein距离、对抗或核基散度作为正则项，训练块三角解码器和潜在编码器；

**📊 数据集**

在三类合成实验（非线性嵌入、球面后验、不可压流场重建）以及公开的Lattice‑Boltzmann 2D流场数据上进行验证；

**📈 对比分析**

与低秩EnKF（LREnKF）比较，CWAEs在所有实验中均实现更低的Wasserstein‑2误差或相对均方误差，尤其在低维后验结构显著的场景下性能显著优于对比方法；

**⚠️ 局限性**

主要局限在于训练对正则化参数λ和网络超参数高度敏感，且在极端高维或数据稀缺场景下训练稳定性和收敛速度仍有待提升。

---

## 162. Poison Once, Exploit Forever: Environment-Injected Memory Poisoning Attacks on Web Agents

**arXiv ID:** 2604.02623 | [PDF](https://arxiv.org/pdf/2604.02623v1)

**作者:** Wei Zou `[一作]` (Pennsylvania State University), Jiarong Jiang `[通讯]` (Amazon Web Services)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种通过环境注入对LLM驱动的Web代理进行记忆毒化（eTAMP）的攻击，能够在不同网站和会话中持续劫持代理行为；

**💡 创新点**

创新点在于首次展示了跨会话、跨站点的环境注入记忆毒化攻击，并引入了“沮丧利用”和“Chaos Monkey”来证明环境压力可显著提升攻击成功率；

**🔧 技术方法**

使用的技术包括基于原始轨迹记忆的注入payload、条件触发逻辑、以及对Agent行为的概率性扰动（Chaos Monkey）；

**📊 数据集**

实验数据集为WebArena和VisualWebArena，覆盖购物、Reddit、分类信息等三大领域，约280个跨站点任务对；

**📈 对比分析**

通过对比不同模型（GPT-5-mini、GPT-5.2、GPT-OSS-120B、Qwen系列等）的攻击成功率和任务完成率，发现eTAMP在无环境压力下最高可达32.5%，在Chaos Monkey下可提升8倍；

**⚠️ 局限性**

局限性包括仅评估原始轨迹记忆而非综合记忆，攻击针对的攻击者仅能注入网页文本，且实验未覆盖所有LLM模型及其多模态版本，实际攻击效果可能因模型结构和安全措施不同而变化。

---

## 163. Elastomeric Strain Limitation for Design of Soft Pneumatic Actuators

**arXiv ID:** 2604.02609 | [PDF](https://arxiv.org/pdf/2604.02609v1)

**作者:** Gregory M. Campbell `[一作]` (University of Pennsylvania), Gregory M. Campbell `[通讯]` (University of Pennsylvania)

**通讯引用:** 29 | [OpenAlex ID](https://openalex.org/A5041765665)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `01e19694-9125-4cf8-82ff-580f56a0fdb6` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究软体机器人软气动执行器（SPA）在主动变形控制与主动应变限制方面的设计与实验。通过在单腔软膜上集成电黏性离合器，利用PWM调制实现可调刚度与形状控制，并开发基于主动学习与神经网络的快速预测模型，用以逆向设计软膜在外力作用下的压力‑力‑位移响应。

**💡 创新点**

创新点：① 首次在单腔SPA中实现实时电黏性离合器激活，提供可变刚度与多自由度形状控制；② 通过PWM调制将离合器的离合力连续化，突破传统二值控制限制；③ 结合主动学习与元学习的神经网络，构建可泛化的软膜力学模型，实现快速预测与逆向设计；④ 通过实验数据驱动，显著缩小理论模型与实验误差。

**🔧 技术方法**

技术手段：软膜与纤维加固、柔性电黏性离合器、PWM高压驱动、深度摄像头与Vicon三维追踪、有限元分析（ABAQUS）、Gent模型与薄膜弹性理论、随机先验网络集成、主动学习、贝叶斯优化、深度学习（DeepONet/Multi‑Layer Perceptron）等。

**📊 数据集**

数据集：自动化实验收集的软膜气压‑力‑位移数据，涵盖膜厚、环形加固位置/宽度、接触半径、负载高度等六维设计向量，共计数千个实验点；数据已整理为.pkl文件，公开在GitHub。

**📈 对比分析**

比较与性能：与传统能量法、单纯FEM或经验公式相比，主动学习模型在未见过的设计上RMSE降低至约3%，提升轨迹精度显著；在软磁升降任务中，能在4.3 kPa下实现0–70 mm高度的负载提升，峰值负载接近5 kg，显示出比硬件执行器更高的柔性与安全性。

**⚠️ 局限性**

局限性：① 电黏性离合器易受表面污染、滑移影响，导致离合力不稳定；② 低压下弯曲范围受限，无法承受大负载；③ 软膜制造误差导致实验与理论偏差，需更精细的制模与装配；④ 模型仍需实验校准，微观摩擦与黏附机制未完全解析；⑤ 需要外部传感器与电源，尚未实现完全无缝集成。

---

## 164. Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens

**arXiv ID:** 2604.02608 | [PDF](https://arxiv.org/pdf/2604.02608v1)

**作者:** Mohammed Suhail B Nadaf `[一作]` `[通讯]` (Indian Institute of Information Technology), Mohammed Suhail B Nadaf (Indian Institute of Information Technology)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对函数向量（Function Vectors，FV）在多种大型语言模型（Llama-3.1、Gemma、Mistral）中进行跨模板转移评估，并探究其在模型内部的可解码性与可驱动性；

**💡 创新点**

主要创新点在于揭示了FV驱动效果与模型自身解码（logit lens）之间的逆向解耦——即在大多数任务和模型上，FV能够成功驱动输出，但模型的中间层却无法通过无学习参数的logit lens解码答案；

**🔧 技术方法**

研究采用了无学习参数的logit lens、FV词汇投影、激活补丁（activation patching）等技术来诊断FV的内在机制，同时使用了标准的cosine相似度与层级回归进行几何分析；

**📊 数据集**

使用了12个任务的自定义数据集，覆盖词汇检索、事实检索、形态变化、字符/表面操作以及组合/语义任务，每个任务配有8个不同模板，共计672个模板对（4,032个跨模板对）；

**📈 对比分析**

与先前的单一模板评估相比，跨模板转移的平均准确率仅略有下降（如Llama Base上的整体gap <0.10），并且FV在早层（L2–L8）即可达到最佳驱动效果，而logit lens在后层（L28–L32）才出现答案可解码；在所有模型上，FV驱动准确率始终优于logit lens，最大差距可达-0.91；

**⚠️ 局限性**

局限性包括：仅评估单词或短输出的任务；仅使用7–9B参数模型；只采用均值差分法提取FV，未探讨其他提取方法；logit lens在早层可能天然不足，未使用可调校的tuned lens；模板数虽多但未覆盖极端攻击式重述。

---

## 165. Complex-Valued GNNs for Distributed Basis-Invariant Control of Planar Systems

**arXiv ID:** 2604.02615 | [PDF](https://arxiv.org/pdf/2604.02615v1)

**作者:** Samuel Honor `[一作]` (Worcester Polytechnic Institute), Kevin Leahy `[通讯]` (Worcester Polytechnic Institute)

**通讯引用:** 786 | [OpenAlex ID](https://openalex.org/A5034884039)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了一种基于复数的 GNN 架构，使分布式控制在无全局参考系时对本地基底旋转不变。

**💡 创新点**

创新点是将 2D 几何特征映射到复数域，使用 SO(2) 等变的消息传递和相位不变激活，并通过复数权重实现对本地基底旋转的显式补偿。

**🔧 技术方法**

技术包括复数神经网络（CVNN）、SO(2) 等变 GNN 层（复数 SAGEConv）、分裂相位-幅值 tanh 激活、DAGGER 模仿学习以及基于图的消息传递。

**📊 数据集**

使用仿真数据：在无全局参考系的 holonomic 双积分机器人群的仿真环境中，构造了一个逼近名义控制器的飞行任务。

**📈 对比分析**

与传统实值 SAGEConv+Tanh 基线相比，复数 GNN 在代表性容量、速度方差、轨迹一致性上更好；在更长时延、通信半径缩小等泛化测试中性能更稳健。

**⚠️ 局限性**

局限在于仍需手工设计非线性特征；未提供可学习的前馈非线性层；对更复杂任务的适用性仍待验证。

---

## 166. Reinforcement Learning-based Knowledge Distillation with LLM-as-a-Judge

**arXiv ID:** 2604.02621 | [PDF](https://arxiv.org/pdf/2604.02621v1)

**作者:** Yiyang Shen `[一作]` (University of Iowa), Weiran Wang `[通讯]` (University of Iowa)

**通讯引用:** 3410 | [OpenAlex ID](https://openalex.org/A5101432591)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出一种基于强化学习的无标签知识蒸馏框架，利用LLM评判者实时生成奖励来训练小型语言模型。

**💡 创新点**

创新点在于将LLM评判者从事后评估转为主动奖励源，且仅使用单步token概率即可产生连续奖励，无需人工标签。

**🔧 技术方法**

主要技术包括PPO/GRPO策略梯度、LLM-as-a-Judge奖励、无标签数据增强以及多任务损失组合。

**📊 数据集**

实验使用GSM8K、SVAMP以及无标签的GSM8K-aug和GSM-Plus等数学推理数据集。

**📈 对比分析**

在GSM8K、SVAMP等基准上，该方法相较于传统RLVR和SFT提升约10–20个百分点，达到40%以上准确率，并在跨域评估中表现更稳健。

**⚠️ 局限性**

局限性包括对评判者质量高度依赖、奖励可能导致模型崩溃、对大型模型的加速有限，以及缺乏对更高级RL算法和扩散模型的探索。

---

## 167. Let's Have a Conversation: Designing and Evaluating LLM Agents for Interactive Optimization

**arXiv ID:** 2604.02666 | [PDF](https://arxiv.org/pdf/2604.02666v1)

**作者:** Joshua Drossman `[一作]`, Sébastien Martin `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种可扩展、可复制的多轮对话评估方法，用 LLM 代理模拟决策者并评估其在交互式优化中的表现，并在学校排课案例中对不同代理设计进行实验

**💡 创新点**

首次将交互式对话作为评估标准，构建“决策代理”模拟真实利益相关者的偏好与沟通方式，探究提示、工具、结果处理等结构化设计对优化代理性能的影响

**🔧 技术方法**

使用大型语言模型（GPT‑3.5、GPT‑4.1、GPT‑5）结合提示工程、预定义工具（Add/Remove Constraint、Modify Objective 等）、代码生成与求解接口，以及 Gurobi 求解器实现优化模型

**📊 数据集**

基于旧金山统一学区（SFUSD）十所学校、三种起始时间的排课数据，生成109个效用函数并与两种沟通风格和两种反馈机制组合得到436个决策代理，产生3,488场对话数据

**📈 对比分析**

通过比较四种代理设计（C‑N‑R、C‑P‑R、C‑P‑P、T‑P‑P）的平均分和成功率，T‑P‑P 在对话评估中取得0.93的平均分和65.1%的成功率，明显优于基础 C‑N‑R 的0.81/31.4%；在不同模型代之间亦保持相对优势

**⚠️ 局限性**

局限性包括：问题规模受限（10所学校，3个时间点），仅模拟单一决策者交互，决策代理为人工构造的理想化角色，评估不涵盖多利益相关者并行决策或更大规模的组合优化问题；工具和提示需针对不同模型重新调优，可能不易迁移到未来更强模型

---

## 168. SocioEval: A Template-Based Framework for Evaluating Socioeconomic Status Bias in Foundation Models

**arXiv ID:** 2604.02660 | [PDF](https://arxiv.org/pdf/2604.02660v1)

**作者:** Divyanshu Kumar `[一作]` (Enkrypt AI), Prashanth Harshangi `[通讯]` (Enkrypt AI)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了 SocioEval 框架，用于系统评估大型语言模型在决策任务中的社会经济阶层偏见。

**💡 创新点**

①多层次层级结构（8 主题 18 细粒度话题）；②基于模板的可扩展设计；③三阶段细粒度到二值化的标注流程。

**🔧 技术方法**

模板化生成、类对比设计、人工三阶段标注、统计分析（卡方检验、置信区间）。

**📊 数据集**

包含 240 条基于 6 个阶层组合的模板提示，13 种前沿 LLM 的 3,120 条响应。

**📈 对比分析**

对比 13 个模型的偏见率（0.42%–33.75%）和主题、类对、回应策略，发现 Anthropic 模型最小偏见，Mistral 最高；主题偏见差异显著，生活方式主题偏见十倍于教育主题。

**⚠️ 局限性**

仅使用英文提示，二元决策形式可能忽略真实决策复杂度；未覆盖多语言或跨文化情境；仅一次性评估，缺乏纵向追踪。

---

## 169. Runtime Execution Traces Guided Automated Program Repair with Multi-Agent Debate

**arXiv ID:** 2604.02647 | [PDF](https://arxiv.org/pdf/2604.02647v1)

**作者:** Jiaqing Wu `[一作]` (Northwestern Polytechnical University), Bo Shen `[通讯]` (Northwestern Polytechnical University)

**通讯引用:** 18946 | [OpenAlex ID](https://openalex.org/A5056603533)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种多代理框架TraceRepair，利用运行时执行追踪作为约束来引导大型语言模型进行程序补丁生成与验证。

**💡 创新点**

创新点包括：①将运行时追踪视为客观约束而非单纯输入；②通过Probe Agent收集关键变量快照；③采用多策略（Defensive、Causal、Semantic）对补丁进行辩论与交叉验证；④最终由Judge Agent合成满足所有运行时约束的补丁。

**🔧 技术方法**

使用技术包括：大型语言模型（DeepSeek‑V3.2、GPT‑3.5‑Turbo）、动态插桩（Probe Agent）、多代理辩论框架（Repair、Judge Agent）、运行时追踪解析与约束检验。

**📊 数据集**

数据集：Defects4J v1.2 与 v2.0 真实 Java 缺陷；自构建的 Recent‑Java（21 个后期缺陷）用于评估泛化与无泄漏性能。

**📈 对比分析**

与十种基线（ChatRepair、RepairAgent、REINFIX、GiantRepair 等）对比，TraceRepair 在 Defects4J 上总共修复 392 例（DeepSeek‑V3.2）/224 例（GPT‑3.5），在多函数 Bug 上显著领先；在 Recent‑Java 上修复 10/6 例，优于静态/对话基线；同时 token 消耗、成本与执行时间显著低于对照方法。

**⚠️ 局限性**

局限性包括：依赖测试套件覆盖度高才能生成有效追踪；插桩成功率虽高但仍有失败；对语言和工具链的适配需进一步研究；在低覆盖或不完整追踪场景下，辩论机制的鲁棒性有限。

---

## 170. Differentiable SpaTiaL: Symbolic Learning and Reasoning with Geometric Temporal Logic for Manipulation Tasks

**arXiv ID:** 2604.02643 | [PDF](https://arxiv.org/pdf/2604.02643v1)

**作者:** Licheng Luo `[一作]` (UC Riverside), Mingyu Cai `[通讯]` (UC Riverside)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

构建了一个可对齐梯度的 SpaTiaL 逻辑工具箱，利用平滑的 SAT 与边界采样 SDF，实现端到端的空间-时间约束优化与规范学习。

**💡 创新点**

首次实现了完全可微分的 SpaTiaL，突破了传统离散几何引擎对梯度传播的阻碍，提供了稠密、可训练的空间谓词。

**🔧 技术方法**

利用 LogSumExp 平滑极值、可微分 SAT、边界采样 SDF、张量化操作以及自动微分技术构建可训练的空间谓词与时序逻辑。

**📊 数据集**

在平面操控和三维取放任务中使用了合成的凸多边形障碍和 30 条人类演示轨迹，工作空间为 [0,10]^3。

**📈 对比分析**

与传统离散碰撞检测和仅考虑时间逻辑的方法对比，表现出更稠密的梯度、较快的鲁棒性收敛和更高的安全裕度，实验在 RTX 4090 GPU 上完成。

**⚠️ 局限性**

仅限凸多边形和稠密边界采样；平滑参数与采样密度的权衡会影响精度与梯度平滑；尚未扩展到复杂网格或动态几何场景。

---

## 171. The Paradox of Prioritization in Public Sector Algorithms

**arXiv ID:** 2604.02641 | [PDF](https://arxiv.org/pdf/2604.02641v1)

**作者:** Erina Seh-Young Moon `[一作]` (University of Toronto), Shion Guha `[通讯]` (University of Toronto)

**通讯引用:** 2122 | [OpenAlex ID](https://openalex.org/A5100659941)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对公共部门层级优先与加权优先两种常见算法的结构机制进行形式化分析，探讨其在资源稀缺条件下如何产生并放大不同群体之间的分配差异，并用北美城市无家可归者收容所数据作案例演示。

**💡 创新点**

①将优先分配视为政策干预而非仅关注模型性能；②发现层级优先在资源极度稀缺时会产生“爆炸性”相对差异；③提出绝对差异与对数比率差异两种评估指标，揭示不同视角下对公平感知的差异。

**🔧 技术方法**

使用概率分配公式、微分分析、对数比率度量以及案例模拟等定量技术；同时绘制资源分配率、绝对差异和对数比率随预算变化的曲线进行对比。

**📊 数据集**

2024年北美某大城市收容所无家可归者数据（按家庭/单身以及难民身份分组）。

**📈 对比分析**

通过在相同预算情景下绘制两种优先机制下的资源分配率、绝对差异（AD）和对数比率差异（ln RD）曲线，比较两种机制的公平性。实验结果显示：层级优先在资源稀缺时，ln RD急剧上升，表明相对不公平加剧；加权优先在同一阶段保持相对稳定，且当预算足够大时趋于平衡；AD虽有限但在绝对层面并未揭示相对差异的激烈程度。

**⚠️ 局限性**

①假设静态权重、无再分配和同一轮分配；②未考虑多种分配策略（如先到先服务、抽签）及其组合；③假设同一组内个体概率相同，忽略内部差异；④研究聚焦“谁”分配，未探讨“何种资源”与“如何交付”。

---

## 172. AXELRAM: Quantize Once, Never Dequantize

**arXiv ID:** 2604.02638 | [PDF](https://arxiv.org/pdf/2604.02638v1)

**作者:** Yasushi Nishida `[一作]` `[通讯]` (Axelidea Inc.), Yasushi Nishida (Axelidea Inc.)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出AXELRAM，一种将KV缓存量化后直接在SRAM中完成注意力计算的智能宏，完全消除反量化操作。

**💡 创新点**

创新点在于：①固定的正交变换代码簿与渐进量化实现仅依赖维度和位宽；②非对称写读路径将所有乘法移至查询一次变换后预计算表，实现102.4×乘法削减；③发现并解决旋转符号模式对KV量化的灾难性敏感性，提出无梯度的符号模式选择，零硬件成本。

**🔧 技术方法**

采用Hadamard正交变换、Lloyd‑Max最优量化、表查表加法、FWHT加法网络和固定代码簿 ROM；梯度无关的符号模式校准算法。

**📊 数据集**

使用WikiText‑2数据集对LLaMA‑3.1‑8B、Qwen2.5‑3B、Qwen3‑8B三大模型进行评估；在不同位宽（2‑4位）下测试。

**📈 对比分析**

与传统需要T×d次乘法并需反量化的KV注意力相比，AXELRAM在多位宽下实现5.1×存储压缩、102.4×乘法下降；通过符号模式优化后，灾难性PPL峰值从Δ>50下降至≤0.8（99%降低），其余模型保持PPL增益≤0.05。

**⚠️ 局限性**

局限性：①对层间键范数异质性高的模型仍存在符号模式敏感；②2‑bit量化在某些模型（如Qwen3‑8B）已达到精度瓶颈；②硬件实现尚未硅验证，仅通过软件仿真验证；③需要一次性校准步骤，虽无运行时开销，但在部署前需额外耗时。

---

## 173. AutoVerifier: An Agentic Automated Verification Framework Using Large Language Models

**arXiv ID:** 2604.02617 | [PDF](https://arxiv.org/pdf/2604.02617v1)

**作者:** Yuntao Du `[一作]` (Purdue University), Ninghui Li `[通讯]` (Purdue University)

**通讯引用:** 17767 | [OpenAlex ID](https://openalex.org/A5101471208)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个六层LLM代理式框架，自动化提取、结构化和跨源验证技术文献中的主张，并通过外部信号共证生成可追溯的情报评估；

**💡 创新点**

创新点在于将LLM与结构化知识图谱、层级式验证流程结合，形成端到端的自动化验证链路，填补表层事实与方法论有效性之间的空白；

**🔧 技术方法**

采用LLM代理、命名实体识别、自然语言推理（NLI）、图推理、视觉语言模型、语义相似检索、链式思考以及外部信号检索等技术；

**📊 数据集**

使用技术文献集（arXiv、Google Scholar论文、专利、作者档案）、视觉资产、公开财务与商业数据、新闻稿、社交网络等；案例中共收集11篇相关文献；

**📈 对比分析**

通过交叉源一致性、根因分析与外部信号共证对比评估，发现原论文的“runtime quantum advantage”被多源证据否定，系统在识别过度宣称、指标不一致、利益冲突等方面表现准确；

**⚠️ 局限性**

局限包括：对LLM推理的幻觉风险、对新领域的适配仍需人工prompt设计、缺乏实时动态更新、外部信号采集可能不完整，导致评估仅为静态 snapshot。

---

## 174. Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis

**arXiv ID:** 2604.02616 | [PDF](https://arxiv.org/pdf/2604.02616v1)

**作者:** Guangyu Sun `[一作]` (University of Central Florida), Chen Chen `[通讯]` (University of Central Florida)

**通讯引用:** 495782 | [OpenAlex ID](https://openalex.org/A5100373745)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了基于姿态的儿童自闭症行为识别的隐私优先框架，采用两层隐私保护：先用3D骨架抽象去除可识别视觉信息，再通过跨医院联邦学习保持数据驻留；

**💡 创新点**

创新点在于将骨架抽象与联邦学习相结合，实现双层隐私防护，并通过自适应混合（APFL）实现多中心数据异质性下的个性化建模；

**🔧 技术方法**

采用FreqMixFormer骨架动作识别骨干网络，结合FedAvg、FedProx、FedBN、FedPer和APFL等联邦学习与个性化策略；

**📊 数据集**

使用MMASD多模态自闭症数据集，仅利用其3D骨架子模态进行实验；

**📈 对比分析**

实验在跨中心联邦设置下比较了本地训练、标准联邦学习和多种个性化方法，结果显示APFL平均准确率达87.80%，明显优于单机训练（82.61%）和传统联邦学习（70.30%）；

**⚠️ 局限性**

局限在于仅考虑单一模态（骨架），未加入语音、会话等多模态信息；并且实验仅在模拟的三主题中心进行，真实多中心异质性及规模仍待验证。

---

## 175. Stochastic Function Certification with Correlations

**arXiv ID:** 2604.02611 | [PDF](https://arxiv.org/pdf/2604.02611v1)

**作者:** Rohan Ghuge `[一作]` (University of Texas at Austin), Mohit Singh `[通讯]` (Georgia Institute of Technology)

**通讯引用:** 15961 | [OpenAlex ID](https://openalex.org/A5085210742)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文研究在相关分布下的随机布尔函数认证问题，并给出了对基于基 matroid、图与超图探测的近似算法。

**💡 创新点**

创新点包括构造新的带有背包覆盖不等式的 LP 松弛，取得非自适应 O(log n) 近似、k‑uniform 常数因子近似，以及自适应 O(log k) 图探测和 2 近似的条件负关联结果。

**🔧 技术方法**

主要技术包括利用子模函数最小化的随机梯度估计求解指数规模的 LP、在 matroid 多面体上随机取样的逼近、递归探测保持独立性的技巧，以及对随机最小背包的归约。

**📊 数据集**

论文为理论分析，没有使用真实数据集，所有结果均在 Bernoulli 相关模型与构造的硬实例上证明。

**📈 对比分析**

与已知的独立分布下的 4 近似及先前的多项式下界相比，本文的 O(log n) 和 O(log k) 近似均为最佳或显著改进，并证明了近似因子在多项式时间内是最优的。

**⚠️ 局限性**

局限性在于需要对联合分布有采样或概率 oracle 的访问，近似比率在某些 matroid 类别仍较大，且自适应算法的对数因子未能进一步降低，且算法未覆盖非布尔或非均匀成本的情况。

---

## 176. A Logic of Secrecy on Simplicial Models

**arXiv ID:** 2604.02673 | [PDF](https://arxiv.org/pdf/2604.02673v1)

**作者:** Shanxia Wang `[一作]` (Henan Normal University), Shanxia Wang `[通讯]` (Henan Normal University)

**通讯引用:** 271 | [OpenAlex ID](https://openalex.org/A5078371286)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文构建了基于简并复形的秘密逻辑，提出了简并秘密模型和原子秘密算子S_a，并给出了语义解释、完备的公理系统以及对多代理情形的完整性证明。

**💡 创新点**

创新点在于将秘密作为独立的新语义层次嵌入简并模型，利用顶点（局部状态）上的邻域函数实现秘密的几何化，并揭示秘密算子是非正常的、仅依赖所有者局部状态的结构特征。

**🔧 技术方法**

采用的技术包括简并染色复形、邻域语义、S5知识的几何表示、辅助色的可判定模型构造以及“共享”映射的表示定理，结合了传统邻域逻辑与分布式计算中的拓扑语义。

**📊 数据集**

论文未使用具体数据集，全部研究均基于形式化模型与理论证明。

**📈 对比分析**

由于本研究是形式逻辑框架的构建，并未与其他方法做实验对比，因而未给出性能指标。

**⚠️ 局限性**

局限性包括：仅在至少两代理（|A|≥2）的情形下证明完整性；单代理情形的模型过于简陋；未探讨联盟秘密、容错或动态更新；秘密算子本身保持非正常，未给出进一步的闭包或归约原则。

---

## 177. Cross-Vehicle 3D Geometric Consistency for Self-Supervised Surround Depth Estimation on Articulated Vehicles

**arXiv ID:** 2604.02639 | [PDF](https://arxiv.org/pdf/2604.02639v1)

**作者:** Weimin Liu `[一作]` (Tsinghua University), Joshua H. Meng `[通讯]` (University of California)

**通讯引用:** 18 | [OpenAlex ID](https://openalex.org/A5080515832)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6514db3d-8de6-452c-91b7-acdb31787cc4` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了 ArticuSurDepth 框架，实现面向多车身机械化车辆的自监督环视深度估计，利用跨车辆几何一致性提升深度学习效果。

**💡 创新点**

创新点包括跨车辆多视角空间上下文增强、伪表面法向一致性约束、基于深度模型的地面平面感知高度正则化以及跨车辆姿态一致性损失，四者共同构建了完整的几何一致性监督体系。

**🔧 技术方法**

采用了结构光/结构从运动（SfM）自监督学习、深度与位姿网络、基于 Vision Foundation Model（DepthAnything V2）的伪几何先验、跨车辆像素重投影、表面法向一致性与高度正则化损失、以及联合运动估计与跨车身姿态一致性约束。

**📊 数据集**

实验使用自行搭建的双车身环视数据集（4k 样本），并在公开数据集 KITTI、DDAD、nuScenes 上进行迁移与跨数据集验证。

**📈 对比分析**

与 GeoDepth、GeoSurDepth、CVCDepth、SurDepth 等基线进行对比，结果显示在自收集数据集上取得 SOTA 性能，在公共数据集上也实现了显著提升，且在 DDAD 与 nuScenes 上的零样本推理表现良好。

**⚠️ 局限性**

主要局限包括对精准外参校准的依赖、跨车辆上下文在极端转向时覆盖不足、对基础模型表面法向质量敏感，以及在更复杂动态场景中的鲁棒性待进一步验证。

---

## 178. Analytic Drift Resister for Non-Exemplar Continual Graph Learning

**arXiv ID:** 2604.02633 | [PDF](https://arxiv.org/pdf/2604.02633v1)

**作者:** Lei Song `[一作]` (Southeast University), Youyong Kong `[通讯]` (Southeast University)

**通讯引用:** 2901 | [OpenAlex ID](https://openalex.org/A5008751186)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出一种基于分析漂移抵消的非示例持续图学习框架ADR，解决特征漂移和模型可塑性问题。

**💡 创新点**

创新点在于解除冻结预训练限制，利用层级分析合并（HAM）实现绝对特征漂移抵消，并通过分析分类器重构（ACR）实现零遗忘。

**🔧 技术方法**

采用图神经网络（GCN）、层级岭回归、特征缓冲层、迭代反向传播与分析合并技术。

**📊 数据集**

在四个节点分类基准（CS‑CL、CoraFull‑CL、Arxiv‑CL、Reddit‑CL）上评估。

**📈 对比分析**

与正则化、回放、非示例和ACL等四类SOTA方法对比，ADR在平均与最终准确率上均优于或接近最优，整体表现最优或仅略逊于ACIL/DS‑AL。

**⚠️ 局限性**

局限在于对极度类别不平衡的任务性能略低，且在任务分布差异极端的情形下仍可能受到漂移影响。

---

## 179. OntoKG: Ontology-Oriented Knowledge Graph Construction with Intrinsic-Relational Routing

**arXiv ID:** 2604.02618 | [PDF](https://arxiv.org/pdf/2604.02618v1)

**作者:** Yitao Li `[一作]` (ProRata.ai), Muni Srikanth `[通讯]` (ProRata.ai)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

通过 intrinsic‑relational routing 将 Wikidata 中每个属性分为 intrinsic 或 relational，构建可重用的声明性 schema，并据此生成 32.3M 节点、61.2M 边的 typed property graph；

**💡 创新点**

创新点在于：① 用 intrinsic‑relational 分类明确图结构决策并产生可移植的 schema；② 采用 agentic LLM 与 grounding 工具实现三种决策 oracle 的迭代 schema refinement；③ 设计模块化、跨领域可定制的声明性 YAML schema，支持单独重用；

**🔧 技术方法**

使用了 Rust 分类器、Python 导出、LMDB label lookup、Claude Opus 4.6 LLM workflow、SPARQL、DuckDB、Neo4j CSV 导入、LLM‑guided extraction prompts 等技术；

**📊 数据集**

基于 2026 年 1 月的 Wikidata JSON dump（约 100M 实体），并在 BLINK、AIDA‑YAGO、CleanCoNLL、CoNLL‑2003 等基准上进行评测；

**📈 对比分析**

通过对 BLINK 受控候选集的宏观平均准确率从 81.5% 提升至 83.9%（+2.4 点），在 benchmark 注释审核中与 AIDA‑YAGO/ CleanCoNLL 对比发现错误率显著下降；分类率 93.3%，模块分配率 98%，并构造了 32.3M 节点、61.2M 边的图；

**⚠️ 局限性**

局限性包括：① 当前 schema 仅针对 2026 年 Wikidata，外部知识源新增关系可能不匹配；② 需要 LLM 或人工辅助扩展/调整模块；③ 对极罕见类型的覆盖仍有限；④ 主要验证英文数据集，跨语言泛化尚待验证；⑤ 对新出现的关系类型需手动或算法扩展。

---

## 180. Beyond Semantic Manipulation: Token-Space Attacks on Reward Models

**arXiv ID:** 2604.02686 | [PDF](https://arxiv.org/pdf/2604.02686v1)

**作者:** Yuheng Zhang `[一作]` (University of Illinois Urbana Champaign), Nan Jiang `[通讯]` (University of Illinois Urbana Champaign)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出Token Mapping Perturbation Attack（TOMPA）框架，直接在token空间对奖励模型进行对抗优化，突破传统语义攻击的局限。

**💡 创新点**

创新点在于通过扰动映射绕过解码-再分词接口，让攻击策略在非语言token序列上寻找高奖励模式，从而揭示奖励模型在token空间的脆弱性。

**🔧 技术方法**

采用强化学习中的GRPO算法对策略进行训练，使用token映射Φ将策略输出直接映射到奖励模型词表，保持完全黑盒查询。

**📊 数据集**

实验使用WildChat 10k提示作为训练集，NoveltyBench 100个提示作为评估集，且对抗与GPT‑5参考答案及随机OOD样本进行对比。

**📈 对比分析**

在两种RM对齐下，TOMPA在奖励分数上几乎翻倍、击败GPT‑5参考答案98%（平均+33.6 vs +17.5），相较于随机OOD几乎0%的击败率显示显著性能提升。

**⚠️ 局限性**

局限在于攻击生成的文本完全无语义、仅适用于黑盒设置，且需匹配不同tokenizer的索引差异，未探究对奖励模型的防御策略。

---

## 181. AgentSZZ: Teaching the LLM Agent to Play Detective with Bug-Inducing Commits

**arXiv ID:** 2604.02665 | [PDF](https://arxiv.org/pdf/2604.02665v1)

**作者:** Yunbo Lyu `[一作]` (Singapore Management University), David Lo `[通讯]` (Singapore Management University)

**通讯引用:** 30832 | [OpenAlex ID](https://openalex.org/A5081036622)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 AgentSZZ，一个基于 LLM 代理的框架，利用交互式工具探索仓库来识别 bug 引入提交。

**💡 创新点**

创新点在于结合任务特定工具、领域知识与 ReAct 循环，突破 git blame 依赖，并加入结构化上下文压缩显著提升跨文件和幽灵提交的识别。

**🔧 技术方法**

使用 GPT‑5‑mini 作为 LLM，ReAct 代理模型，五个自定义 git 工具接口，结构化压缩模块，并通过 OpenRouter API 调用。

**📊 数据集**

实验采用三套开发者标注数据集：Linux kernel（1500 个 bug‑fixing 提交）、GitHub（355 个 C/Java 项目）和 Apache（241 个项目）。

**📈 对比分析**

与 11 个基线（传统 SZZ、神经 SZZ、LLM4SZZ 等）对比，平均 F1 提升 15.4%–27.2%，在跨文件和幽灵案例中召回率提升 100%–300% 与 60%，运行时间仅比 LLM4SZZ 增加约 15%，token 消耗减少 30%。

**⚠️ 局限性**

主要局限在于仍容易将相近功能相关的提交误归属为 Bug‑Inducing，难以处理多 BIC 情况，缺乏执行验证，且对不同语言和项目的适应性尚待进一步提升。

---

## 182. Semantic Data Processing with Holistic Data Understanding

**arXiv ID:** 2604.02655 | [PDF](https://arxiv.org/pdf/2604.02655v1)

**作者:** Youran Sun `[一作]` (UC Berkeley), Aditya G. Parameswaran `[通讯]` (UC Berkeley)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出一种基于LLM的语义数据处理框架，通过全局数据理解来提升分类、打分和聚类等常见Map操作的准确率。

**💡 创新点**

创新点在于：①引入“LLM数据理解悖论”概念，说明大规模上下文会削弱LLM质量；②设计了基于图的聚类算法，使用Bagging式采样和相关聚类（Correlation Clustering）来估计记录间同类概率；③通过二部匹配和ILP实现聚类与类别/分数的整体分配；④结合模型级联（Model Cascade）实现成本与精度的平衡。

**🔧 技术方法**

技术手段包括：LLM调用（GPT‑4.1 与 GPT‑4.1‑Nano）、Bagging式随机子集采样、相关聚类、二部图最大权匹配、整数线性规划（ILP）、模型级联与阈值选择、成本估算公式。

**📊 数据集**

使用15个真实世界数据集：分类/聚类（AgNews、Blurb、Clinc、DBPedia、GoEmo、Massive、Mtop、ArxivS2S、BiorxivS2S、MedrxivS2S 等）和打分（Amazon、Asap、Sentiment、TicketSupport、Yelp）。

**📈 对比分析**

相较于基线（Row‑by‑Row、Batch、Rubric、BARGAIN、K‑means+Embed、LabelCluster、Lotus、Keyphrase），该方法在分类上平均提升约 33% 准确率（单一数据集最高 33%），在打分与聚类上提升约 30%，并且在大多数场景下成本与或低于基线，尤其在高预算情况下显著提升准确率。

**⚠️ 局限性**

局限性：①仍依赖LLM的质量，长上下文仍是瓶颈；②对高维/极大规模数据的扩展仍需优化；③仅针对典型的Map、打分、聚类任务，其他开放式Map操作的适用性尚未深入验证；④在多类极其相似的场景下，聚类误差仍可能影响整体性能。

---

## 183. Product-Stability: Provable Convergence for Gradient Descent on the Edge of Stability

**arXiv ID:** 2604.02653 | [PDF](https://arxiv.org/pdf/2604.02653v1)

**作者:** Eric Gan `[一作]` `[通讯]` (Independent Researcher), Eric Gan (Independent Researcher)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文通过引入产品稳定性（product-stability）概念，证明在梯度下降（GD）处于边缘稳定性（Edge of Stability，EoS）时，若损失函数形式为l(xy)且最小点满足产品稳定性，则GD能够收敛并且收敛时的锐度略低于EoS阈值。

**💡 创新点**

创新点在于提出产品稳定性这一更广泛、可计算的局部条件，统一并推广了此前针对平方损失、子二次损失等的EoS收敛分析，并给出了收敛时锐度的精确上界。

**🔧 技术方法**

主要技术包括高阶导数分析、两步梯度更新的固定点与分岔图分析、以及对多元情形的推广；通过构造的分岔图解释了训练动态的三阶段演化。

**📊 数据集**

在实验验证中使用了含两隐藏层、宽度200的全连接tanh网络，在CIFAR-10的5000样本子集上训练，以观察EoS行为及产品稳定性。

**📈 对比分析**

与此前基于子二次假设或特定深度网络的理论结果相比，本文的理论适用于更广泛的损失（如二元交叉熵、幂次平方损失），并通过实验显示收敛时锐度逼近EoS阈值，验证了理论预言。

**⚠️ 局限性**

局限性包括：分析仅为局部，无法给出全局GD动力学；未解释模型进入EoS的“渐进锐化”过程；产品稳定性条件虽宽松，但仍可能不在所有实际深度网络的所有最小点上成立。

---

## 184. GBQA: A Game Benchmark for Evaluating LLMs as Quality Assurance Engineers

**arXiv ID:** 2604.02648 | [PDF](https://arxiv.org/pdf/2604.02648v1)

**作者:** Shufan Jiang `[一作]` (University of Hong Kong), Zhiyang Chen `[通讯]` (Westlake University)

**通讯引用:** 5714 | [OpenAlex ID](https://openalex.org/A5100324483)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作构建了一个基于游戏的自动 bug 发现基准 GBQA，包含 30 个多样化游戏与 124 个人工验证的 bug，并提供了可扩展的多智能体游戏构建器、交互式基线代理与自动评估框架；同时评测了多款前沿 LLM 的自主动 bug 发现能力。

**💡 创新点**

创新点在于：①提出可扩展的多智能体协作生成游戏并可控注入 Bug 的系统；②将 bug 发现形式化为交互式探索任务；③引入 ReAct+反思循环和分层内/跨会话记忆机制作为基线。

**🔧 技术方法**

主要技术包括：多智能体协作游戏构建与 Bug 注入；ReAct 推理与反思验证循环；分层记忆模块（会话内压缩与跨会话存储）；LLM 思考模式；Critic Agent 自动评估。

**📊 数据集**

使用的数据集为 GBQA 基准，包含 30 个游戏、124 个人工标注的 Bug，按易、中、难分类；并通过 Krippendorff α 等方法验证标注质量。

**📈 对比分析**

评测采用 Recall 指标，在玩家探索与 QA 两种模式下，给定 50/100/200/500 步的交互预算；结果显示最佳 Claude‑4.6‑Opus（思考模式）在 500 步下仅达 48.39% recall，表明当前 LLM 在自主动 bug 发现上远逊于代码生成/修复任务。

**⚠️ 局限性**

主要局限在：LLM 的长推理与状态跟踪易出现误差；缺乏系统化的测试策略与长期持续性 Bug 发现能力；基准目前仅涵盖游戏领域，缺少多模态 GUI 交互场景。

---

## 185. Engagement Is Not Transfer: A Withdrawal Study of a Consumer Social Robot with Autistic Children at Home

**arXiv ID:** 2604.02642 | [PDF](https://arxiv.org/pdf/2604.02642v1)

**作者:** Yibo Meng `[一作]` (Cornell University), Haipeng Mi `[通讯]` (Tsinghua University)

**通讯引用:** 743 | [OpenAlex ID](https://openalex.org/A5022394827)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

进行了一项为期8周的家庭随机对照试验，使用消费级社交机器人Qrobot评估其对自闭症儿童情绪调节和社会技能的影响，并通过机器人撤除阶段检验其转移效果。

**💡 创新点**

创新点在于将机器人撤除作为“转移探针”，揭示高投入的机器人交互不一定导致人际社交转移，提出了“舒适陷阱”概念以及转移优先的设计模式。

**🔧 技术方法**

采用了混合方法：定量量表（SCARED、SMS、RMET、BES、SUS）与定性访谈，使用线性混合效应模型进行数据分析。

**📊 数据集**

使用的数据来自40名5-9岁自闭症儿童在家庭环境中收集的量表得分与访谈文本，没有引用公开数据集。

**📈 对比分析**

通过时间点的线性混合模型和事后t检验进行组间比较，结果显示持续使用显著降低焦虑（SCARED、RCADS），但社交动机、情感推断和共情指标下降；撤除组在这些转移指标上显著提升，表明高可用性不等于转移。

**⚠️ 局限性**

局限性包括样本量小、未收集机器人使用日志、分组未盲化、依赖保管者报告可能产生偏差、ASD史不平衡以及测量工具对年龄的敏感性。

---

## 186. Toys that listen, talk, and play: Understanding Children's Sensemaking and Interactions with AI Toys

**arXiv ID:** 2604.02629 | [PDF](https://arxiv.org/pdf/2604.02629v1)

**作者:** Aayushi Dangol `[一作]` (University of Washington), Julie A. Kientz `[通讯]` (University of Washington)

**通讯引用:** 11247 | [OpenAlex ID](https://openalex.org/A5066304043)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对6-11岁儿童在两次参与式设计会议中与Curio等基于LLM的AI玩具进行互动，收集视频、聊天日志与漫画板等数据，研究儿童如何理解、修复与边界测试AI玩具的交互

**💡 创新点**

首次从儿童视角系统性揭示了儿童在与生成式AI玩具交互时的“意义建构”与“冒险性游戏”，以及AI玩具缺乏社会反馈和可停机的设计缺陷，为儿童中心的AI玩具设计提供了实证依据

**🔧 技术方法**

采用参与式设计（Cooperative Inquiry）方法结合大语言模型驱动的Curio AI玩具进行实验，使用访谈式观察、录音录像、聊天记录与漫画板等多模态数据收集与分析技术

**📊 数据集**

以8名儿童在实验室中进行的两次设计会议录制的视频（约262分钟）、Curio玩具的对话日志以及儿童手绘漫画板为数据集

**📈 对比分析**

本文未进行量化对比实验，而是通过质性编码和主题分析对收集的多模态数据进行归纳，展示儿童在交互失效、修复与边界测试中的行为模式，并未给出传统性能指标

**⚠️ 局限性**

研究局限包括样本规模小、实验时间短、仅测试单一AI玩具、实验环境为实验室，且缺乏长期跟踪与多种玩具/环境的交叉验证

---

## 187. Smart Transfer: Leveraging Vision Foundation Model for Rapid Building Damage Mapping with Post-Earthquake VHR Imagery

**arXiv ID:** 2604.02627 | [PDF](https://arxiv.org/pdf/2604.02627v1)

**作者:** Hao Li `[一作]`, Wufan Zhao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

利用预训练的视觉基础模型（Vision Foundation Models）快速实现跨区域地震后建筑损毁映射，提出 Smart Transfer 框架，采用像素级聚类（PC）和距离惩罚三元组（DPT）两种智能迁移策略，提升在不同城市形态下的泛化性能；

**💡 创新点**

创新点在于：①将 Vision FM 与 GeoAI 结合，首次在灾害映射中引入大规模预训练视觉模型；②设计两种迁移策略（PC 与 DPT）实现原型级全局特征对齐和基于空间自相关的细粒度局部约束；③在极少标注条件下通过轻量化解码器实现快速部署；

**🔧 技术方法**

技术包括：ViT-L/16 的 DINOv3 视觉编码器、像素级聚类（基于 k-means 原型对齐）、距离惩罚三元组损失、轻量化卷积上采样解码器、Focal 损失、LoRA 参数高效微调；

**📊 数据集**

使用 2023 年土耳其-叙利亚地震后的 Pléiades 1A/1B VHR 影像、公开的 KATE-CD 损毁标注数据以及 GlobalBuildingAtlas 建筑边界，覆盖九个不同城市区域，共计 1,340 张标注图块；

**📈 对比分析**

与传统 ResNet-18/152、YOLO-like 基线对比，在全监督和迁移设置（LODO、SSDC）下，Smart Transfer 在 mIoU、F1 等指标上取得显著提升（例如 LODO 下 PC 方案 mIoU 0.69±0.03，F1 0.64±0.06，远优于基线），并在低标注比例下表现出更好的数据效率；

**⚠️ 局限性**

局限性包括：仅基于 VHR 影像，缺乏多模态信息；迁移实验仅在同一次灾害内部进行，跨事件泛化仍待验证；对小型或低光照场景的鲁棒性未知；对极端灾害类型（如洪水、火灾）的适用性尚未测试。

---

## 188. Polynomial-Time Almost Log-Space Tree Evaluation by Catalytic Pebbling

**arXiv ID:** 2604.02606 | [PDF](https://arxiv.org/pdf/2604.02606v1)

**作者:** Vahid R. Asadi `[一作]` (University of Waterloo), Richard Cleve `[通讯]` (University of Waterloo)

**通讯引用:** 16820 | [OpenAlex ID](https://openalex.org/A5001743971)

**关键词:** `b85d34da-f1e4-4203-bfed-9536213d369b` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种在多项式时间内以近对数空间求解树评估问题（Tree Evaluation Problem）的算法。

**💡 创新点**

创新点在于将Cook‑Mertz的低空间构造与传统的催化性痕迹（catalytic pebbling）技术相结合，形成一种既能保持多项式时间又仅使用 O(log n) 自由空间、O(log^{1+} n) 催化空间的新算法。

**🔧 技术方法**

主要技术包括：基于有限域的低阶多项式算术化（arithmetic extension）、催化空间模型（catalytic computation）、多层递归树分解（将树分成高度 loglog n 的子树）以及对多项式的线性组合和原根运算。

**📊 数据集**

本文未使用具体实验数据集，所有结果均为理论分析与证明。

**📈 对比分析**

与Cook‑Mertz先前算法比较：本算法在时间上恢复多项式性（O(n^{1/ + o(1)})），空间上保持近对数级（O(log^{1+} n)），并仅需 O(log n) 的自由空间；相较之下，Cook‑Mertz算法在空间上更优（O(log n log log n)）但时间为超多项式。

**⚠️ 局限性**

局限性包括：仍需使用催化空间（非完全可恢复的寄存器），空间上虽然接近对数但不是真正的 O(log n)；算法实现复杂，且对树的高度假设为 h = log n、分支因子为 2，扩展到一般参数仍需进一步工作。

---

## 189. Do Audio-Visual Large Language Models Really See and Hear?

**arXiv ID:** 2604.02605 | [PDF](https://arxiv.org/pdf/2604.02605v1)

**作者:** Ramaneswaran Selvakumar `[一作]` (University of Maryland), Dinesh Manocha `[通讯]` (University of Maryland)

**通讯引用:** 39643 | [OpenAlex ID](https://openalex.org/A5004194238)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `b88c6eac-d57a-4623-a604-1f401f3eb268` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研究了音视频大型语言模型在音频-视觉融合过程中的机制解释，并揭示视觉偏置导致音频理解显著下降。

**💡 创新点**

首次系统使用注意力分布、Logit Lens、注意力切除和分布相似度等技术，揭示 AVLLM 视觉主导的跨模态信息流与训练来源的偏差。

**🔧 技术方法**

采用 Transformer 关注权重分析、Logit Lens 表征探测、因果中介分析（attention knockout）以及分布相似度评估等机制解释工具。

**📊 数据集**

基于 AudioCaps 采集的 500 样本事实与反事实音视频对齐集进行评估，并使用 Qwen3‑32B 作为 LLM 评判者。

**📈 对比分析**

通过 LLM‑评判得分与 token 分布相似度（KL 与 rank）比较，发现反事实场景下音频理解仅 23% 但隐藏语义达到 61%，表现出显著的视觉优势。

**⚠️ 局限性**

实验仅涵盖开源 AVLLM，聚焦非语音音频事件，缺乏大规模反事实训练数据与跨模型泛化验证。

---

## 190. MOMO: Mars Orbital Model Foundation Model for Mars Orbital Applications

**arXiv ID:** 2604.02719 | [PDF](https://arxiv.org/pdf/2604.02719v1)

**作者:** Mirali Purohit `[一作]` (Arizona State University), Hannah Kerner `[通讯]` (Arizona State University)

**通讯引用:** 914 | [OpenAlex ID](https://openalex.org/A5053180513)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `729e5870-4135-47f5-97f2-e3974d07b5dc` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f` `14d48e9d-0069-4ad9-996a-1d5968216998` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了第一个面向火星轨道遥感的基础模型MOMO，通过模型合并实现多传感器、多分辨率数据的统一表示；

**💡 创新点**

创新点在于提出Equal Validation Loss（EVL）策略，在不同传感器的训练过程中对齐验证损失以挑选兼容的检查点，然后利用任务算术合并模型；

**🔧 技术方法**

技术包括掩码自编码器（MAE）训练、结构感知损失（MSE+SSIM+LPIPS+梯度损失）、EVL检查点选择、线性模型算术融合；

**📊 数据集**

使用约1200万张从HiRISE（0.25m）、CTX（5m）和THEMIS（100m）传感器中筛选并质量过滤后的图像数据集；

**📈 对比分析**

与ImageNet预训练、地球观测基础模型、单传感器预训练、数据合并（DM）以及传统的Early Stopping/Last Epoch合并进行对比；在9项Mars-Bench下任务上，MOMO在分类和分割任务均超过所有基线，平均提升约1–4％，分割任务上显著优于其它模型；

**⚠️ 局限性**

局限性包括：未对其他模型合并基线进行评估，未尝试对齐技术组合，且假设模型间线性模式连接，可能在数据分布差异极大时失效。

---

## 191. Breakdowns in Conversational AI: Interactional Failures in Emotionally and Ethically Sensitive Contexts

**arXiv ID:** 2604.02713 | [PDF](https://arxiv.org/pdf/2604.02713v1)

**作者:** Jiawen Deng `[一作]` (University of Electronic Science and Technology of China), Fuji Ren `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 8726 | [OpenAlex ID](https://openalex.org/A5071943346)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了在情感与伦理敏感交互中，主流对话式人工智能会出现的对话失衡与误差，并基于 persona‑conditioned 用户模拟器对其进行诊断，提出了一套失衡模式的分类与对齐设计建议。

**💡 创新点**

创新点在于：①首次系统性地将情感与伦理维度整合进对话模拟与评估；②构建 persona‑conditioned 模拟器与情绪调速机制，能生成动态情绪曲线的多轮对话；③提出基于 LLM-as‑judge 的自动化评估框架与失衡模式的 taxonomy，推动对话一致性与伦理连贯性的评估。

**🔧 技术方法**

技术包括：persona‑conditioned 用户模拟器、情绪调速（情感轨迹）模块、LLM‑as‑judge 自动评估器、基于 GPT 系列的语言模型生成多轮对话与情感检测。

**📊 数据集**

使用的数据集为 ProsocialDialog（公开去标识化的情感对话数据）以及由模拟器和 LLM 自动生成的合成对话。

**📈 对比分析**

对比方法：在同一情感轨迹与情景下，评估多种主流对话模型的失衡频率和严重程度；性能表现表明主流模型在情绪升级时失衡率显著提升，且存在情感与伦理的交叉冲突。未给出具体数值，但通过 LLM‑as‑judge 与人工验证显示失衡模式的可重复性与显著性。

**⚠️ 局限性**

局限性包括：①模型与评估均来自同一 GPT 体系，易受共同来源偏见影响；②情绪检测器仅覆盖通用情绪，缺乏对道德情绪（如内疚、羞愧）的细粒度识别；③场景筛选与标签质量有限，可能引入噪声。

---

## 192. ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing ESKF-Based LiDAR-Inertial Odometry

**arXiv ID:** 2604.02706 | [PDF](https://arxiv.org/pdf/2604.02706v1)

**作者:** Seongjun Kim `[一作]` (POSTECH), Soohee Han `[通讯]` (POSTECH)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

研究了在 LiDAR‑IMU 融合里程计中如何处理几何退化环境，并提出了一种名为 ALIVE‑LIO 的新框架。

**💡 创新点**

创新点在于将深度学习的速度估计与经典误差状态卡尔曼滤波器紧耦合，仅在检测到退化时更新退化方向，从而实现对退化方向状态的精准补偿。

**🔧 技术方法**

采用了 X‑ICP 退化检测、PV‑LIO 误差状态卡尔曼滤波、Bi‑GRU 网络预测主体速度，并在 ESKF 中进行投影融合。

**📊 数据集**

使用公开的 GEODE、UrbanNav、Livox 数据集以及自采集的手持、无人机和车辆平台数据进行训练和评估。

**📈 对比分析**

与 FAST‑LIO2、PV‑LIO、D‑LIO、GenZ‑ICP、X‑ICP 等基线对比，实验显示 ALIVE‑LIO 在 32 条序列中取得 22 条最小位移误差，平均 ATE 降低约 30%。

**⚠️ 局限性**

在极端旋转运动下速度预测不稳定，退化检测误判时可能不触发补偿，且对传感器摆放和机械振动较为敏感。

---

## 193. LieTrunc-QNN: Lie Algebra Truncation and Quantum Expressivity Phase Transition from LiePrune to Provably Stable Quantum Neural Networks

**arXiv ID:** 2604.02697 | [PDF](https://arxiv.org/pdf/2604.02697v1)

**作者:** Haijian Shao `[一作]` (Jiangsu University of Science and Technology), Yingtao Jiang `[通讯]` (University of Nevada)

**通讯引用:** 2700 | [OpenAlex ID](https://openalex.org/A5069124655)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出了 LieTrunc-QNN，一种通过结构化 Lie 代数截断来控制量子神经网络可达态流形维度、避免 barren plateau 并保持可训练性的框架。

**💡 创新点**

创新点在于将量子可表达性、可训练性与流形几何（有效维度、Fubini–Study 度量谱）关联，提出几何容量-平台原理，并给出首个多项式可训练性证明；同时通过 Lie 代数截断实现结构化的模型压缩，兼顾表达能力与鲁棒性。

**🔧 技术方法**

采用 Lie 代数、微分几何（Fubini–Study 里氏度量）、谱分析、梯度方差理论，并在量子电路中实现结构化截断；实验使用小规模量子分类和 VQE 任务，通过数值模拟验证理论。

**📊 数据集**

使用模拟的量子分类和 Variational Quantum Eigensolver（VQE）任务，实验规模为 2 到 6 个 qubit，采用多种量子电路构造（Full PQC、RandomTrunc、LieTrunc）。

**📈 对比分析**

对比 Full PQC、随机截断和 LieTrunc 方案。实验显示 Full PQC 在 qubit 数增大时出现指数梯度消失（barren plateau），随机截断导致表达能力坍塌；LieTrunc 在保持高有效维度和完整度量秩的同时，梯度方差保持在多项式范围，任务损失与 Full PQC 相近或更好，验证了几何尺度定律（梯度方差 × 有效维度 ≈ 常数）。

**⚠️ 局限性**

局限性包括：仅在小规模（≤6 qubit）模拟实验中验证，未在真实量子硬件上测试；需要先验的 Lie 代数选择与截断策略，可能对不同任务的适配性有限；理论假设（光滑性、无噪声、有限深度）与实际硬件噪声存在差距。

---

## 194. VBGS-SLAM: Variational Bayesian Gaussian Splatting Simultaneous Localization and Mapping

**arXiv ID:** 2604.02696 | [PDF](https://arxiv.org/pdf/2604.02696v1)

**作者:** Yuhan Zhu `[一作]` (University of California Riverside), Wei Ren `[通讯]` (University of California Riverside)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 VBGS-SLAM，一种将变分贝叶斯高斯摊平与 SLAM 结合的密集 RGB‑D SLAM 框架，能够在单帧内同时更新相机位姿与三维高斯地图；

**💡 创新点**

创新点在于将相机姿态视作潜在变量，利用闭式变分推理在高斯地图与位姿之间实现概率耦合，消除传统基于梯度的直接优化对初始化敏感和灾难性遗忘的问题；

**🔧 技术方法**

采用变分贝叶斯推理、SE(3) Lie 群的高斯分布、闭式更新公式、关键帧选择与自适应高斯管理，以及 RGB‑D 逆投影等技术；

**📊 数据集**

在合成 Replica、真实 TUM‑RGBD 以及含 IMU 的 AR‑TABLE 三个数据集上进行实验；

**📈 对比分析**

与 NICE‑SLAM、Vox‑Fusion、Point‑SLAM、SplaTAM、MonoGS 等基线对比，平均 ATE 0.33cm（Replica）、5.94cm（AR‑TABLE）、7.69cm（TUM‑RGBD），渲染 PSNR 分别为 37.94、35.24、23.46 dB，帧率略高于 MonoGS，整体性能优于大多数现有方法；

**⚠️ 局限性**

局限在于尚未处理动态场景、缺乏全局回环闭合能力，对极端光照或大尺度环境的可扩展性有待提升。

---

## 195. XrayClaw: Cooperative-Competitive Multi-Agent Alignment for Trustworthy Chest X-ray Diagnosis

**arXiv ID:** 2604.02695 | [PDF](https://arxiv.org/pdf/2604.02695v1)

**作者:** Shawn Young `[一作]` (Shenzhen University of Advanced Technology), Lijian Xu `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出 XrayClaw，多代理协同竞争框架，用四个协作代理模拟临床工作流，配合一个竞争审计代理，提升胸片诊断的可信度和解释性。

**💡 创新点**

创新点在于：① 将诊断任务拆解为分阶段专门代理，重建真实放射员工作流程；② 通过竞争性 Preference Optimization（ComPO）实现协作与审计结果的对抗性对齐，直接惩罚逻辑幻觉；③ 采用单一模型实现多代理，保持一致性而非多模型堆叠。

**🔧 技术方法**

技术包括 Qwen3-VL 多模态基础模型、Supervised Fine‑Tuning、ComPO（基于 Direct Preference Optimization 与 Bradley‑Terry 模型的损失），以及集中式上下文缓冲区实现代理间通信。

**📊 数据集**

使用 MS‑CXR‑T、MIMIC‑CXR、CheXbench 三大公开胸片数据集进行评估；同时在 MS‑CXR‑T、MIMIC‑CXR、CheXbench 进行多标签分类、报告生成和跨域推理实验。

**📈 对比分析**

与多种基线（单模型、传统 MLLMs、现有多代理如 MedRAX、RadFabric 等）对比，XrayClaw 在 MS‑CXR‑T 的平均 Top‑1 诊断准确率提升至 77.9%（比 MedRAX 提高 4.8%），在 MIMIC‑CXR 的报告生成指标（BLEU‑4 0.163、METEOR 0.190、CIDEr 0.368）均优于现有多代理方法；在 CheXbench 的跨域推理中，总准确率 70.7%，比 MedRAX 提升 2.6%。

**⚠️ 局限性**

局限包括：① 依赖单一大型模型，计算资源需求大；② 竞争审计代理的“经验”仅来源于提示而非真实临床经验，可能不足以覆盖所有异常；③ 评估指标主要集中在自动化指标，临床实用性、可解释性细节仍待进一步验证。

---

## 196. Towards Realistic Class-Incremental Learning with Free-Flow Increments

**arXiv ID:** 2604.02765 | [PDF](https://arxiv.org/pdf/2604.02765v1)

**作者:** Zhiming Xu `[一作]` (Nanjing University), Suorong Yang `[通讯]` (Nanjing University)

**通讯引用:** 1892 | [OpenAlex ID](https://openalex.org/A5037089910)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了自由流增量学习（FFCIL）框架，解决传统CIL在实际中类别增量不均衡时的性能退化问题

**💡 创新点**

创新点在于：①类均值（CWM）目标消除频率偏差；②仅对回放样本进行知识蒸馏；③对对比、知识迁移等损失做尺度归一化；④动态调节权重校准（DIWA）以适应不同增量规模

**🔧 技术方法**

采用类均值损失、回放蒸馏、损失尺度归一化、动态权重对齐等技术，并对现有七种CIL基线进行无缝迁移

**📊 数据集**

在CIFAR-100、VTAB、ImageNet三个公开数据集上构建FFCIL基准进行实验

**📈 对比分析**

与传统等分任务CIL和FFCIL原始方法对比，实验显示FFCIL导致准确率下降，应用框架后准确率普遍提升，平均提升约3–5%，并显著减少遗忘

**⚠️ 局限性**

局限性包括：仅针对现有方法提供通用改造，缺乏针对极大类增量场景的理论分析，且在极端增量安排下仍存在一定性能波动

---

## 197. InverseDraping: Recovering Sewing Patterns from 3D Garment Surfaces via BoxMesh Bridging

**arXiv ID:** 2604.02764 | [PDF](https://arxiv.org/pdf/2604.02764v1)

**作者:** Leyang Jin `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Hao Li `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 18810 | [OpenAlex ID](https://openalex.org/A5100348588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一个两阶段框架，通过结构化中间表示BoxMesh从3D衣物网格恢复缝纫图案。

**💡 创新点**

引入BoxMesh中间表征，拆解逆缝纫问题为逆仿真和模式解析，显著降低不确定性；同时采用自回归Transformer和压缩标记化处理平面化网格。

**🔧 技术方法**

使用自回归Transformer、压缩Token化（Compressive Tokenization）、点云特征提取、语义自回归模型、MeshAnything类技术、SMPL拟合以及3D Gaussian重建等技术。

**📊 数据集**

利用GCD v2、THUMan2.0、RenderPeople以及手机多视角扫描的数据集进行训练与评估。

**📈 对比分析**

与NeuralTailor*进行对比，在GCD上所有评估指标（Chamfer距离、Hausdorff距离、Panel IoU等）均优越，单视图和真实扫描场景也保持较高准确性。

**⚠️ 局限性**

对极端服装长度或缺失扫描区域敏感，合成数据缺乏部分裙装类型，未包含手工清理流程，可能导致后续缝纫关系错误。

---

## 198. STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation

**arXiv ID:** 2604.02756 | [PDF](https://arxiv.org/pdf/2604.02756v1)

**作者:** Zijin Liu `[一作]` (Beihang University), You Song `[通讯]` (Beihang University)

**通讯引用:** 15921 | [OpenAlex ID](https://openalex.org/A5075995463)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `a8e75ba4-7a2d-4153-b003-06c94533add0` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种名为 STDDN 的人群仿真框架，通过将宏观连续性方程与微观轨迹预测耦合，实现高精度、稳定且高效的长时仿真。

**💡 创新点**

创新点包括：①将连续性方程作为可微分物理约束嵌入神经ODE，形成宏观‑微观耦合；②设计可微分密度映射、连续跨格检测与密度‑速度耦合动态图学习模块；③采用节点嵌入降低稀疏图权重维度，提升内存效率。

**🔧 技术方法**

采用的技术有：神经ODE、动态图神经网络、可微分密度映射、Jensen‑Shannon 散度连续跨格检测、节点嵌入、基于Transformer/Graph 的轨迹预测网络。

**📊 数据集**

使用的公开数据集包括 GC、UCY（ZARA1/2/UCY）、ETH（ETH/HOTEL）共四个大规模人群轨迹数据集。

**📈 对比分析**

与物理模型（SFM、CA）、数据驱动方法（STGCNN、PECNet、MID）以及物理引导方法（PCS、NSP、SPDiff）在 MAE、OT、参数量和单帧延迟等指标上进行对比。STDDN 在所有数据集上均优于 SPDiff，MAE 与 OT 均取得显著提升，推理延迟下降约 50–90%，参数量约 0.17M。

**⚠️ 局限性**

局限性包括：对网格尺寸和离散化选择敏感，细网格会显著增加内存；目前仅处理二维平面静态障碍，未考虑三维或动态环境；在极高密度或速度分布极端差异的场景下，物理约束与数据驱动的平衡仍需进一步调优。

---

## 199. Accelerating Nonlinear Time-History Analysis with Complex Constitutive Laws via Heterogeneous Memory Management: From 3D Seismic Simulation to Neural Network Training

**arXiv ID:** 2604.02755 | [PDF](https://arxiv.org/pdf/2604.02755v1)

**作者:** Tsuyoshi Ichimura `[一作]` (University of Tokyo), Lalith Maddegedara `[通讯]` (Japan Agency for Marine-Earth Science and Technology)

**通讯引用:** 722 | [OpenAlex ID](https://openalex.org/A5020079438)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本研究提出了一种基于异构内存管理的框架，能够在CPU‑GPU协同环境下高效完成大规模非线性时序分析，尤其针对复杂本构律（多弹簧土壤模型）的三维地震响应仿真；

**💡 创新点**

创新点在于：①将内存容量受限的本构计算迁移至GPU，通过高带宽CPU‑GPU互连实现数据传输与GPU计算重叠，突破GPU内存墙；②在GPU内部采用EBE（Element‑by‑Element）稀疏矩阵向量乘法及多精度多重网格预处理，进一步提升计算吞吐；③结合大规模仿真数据构建神经网络代理模型，实现快速损伤评估；

**🔧 技术方法**

使用的技术包括：CPU‑GPU异构内存管理、PCIe/NVLink高速互连、OpenACC/OpenMP并行、稀疏矩阵分块格式、CRS/EBE矩阵乘法、共轭梯度/自适应混合精度多重网格预处理、神经网络（1D‑CNN+LSTM）训练与Optuna超参搜索；

**📊 数据集**

数据集主要为：一套真实三维地震地形模型（东京附近软沉积地层），以及100条随机波形（频率<2.5 Hz、振幅均匀分布）用于训练NN；

**📈 对比分析**

与传统CPU‑仅方法（CRSCPU_MSCPU）和仅将求解器迁移到GPU的方法（CRSGPU_MSCPU）相比，Proposed Method 2在单节点GH200上实现了约12.8倍的速度提升（时间从182,300 s降至14,222 s）和约1/6.7的能耗降低；Proposed Method 1也比基线快约1.25倍；神经网络代理模型在三维与一维仿真对比中，能准确重现3D非线性放大效应。

**⚠️ 局限性**

局限性包括：需高带宽CPU‑GPU互连（NVLink‑C2C）才能实现重叠效果；在较低带宽系统（如PCIe Gen 5）下性能提升有限；仅在单节点GH200上验证，跨节点扩展仍需进一步评估；模型对极端地震事件的鲁棒性及更复杂本构律的适用性待进一步验证。

---

## 200. An Empirical Study of Sustainability in Prompt-driven Test Script Generation Using Small Language Models

**arXiv ID:** 2604.02754 | [PDF](https://arxiv.org/pdf/2604.02754v1)

**作者:** Pragati Kumari `[一作]` (University Of Calgary), Novarun Deb `[通讯]` (University Of Calgary)

**通讯引用:** 159 | [OpenAlex ID](https://openalex.org/A5072598400)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文通过在 HumanEval 基准上使用 2B–8B 参数范围的小语言模型进行提示驱动的单元测试脚本生成，并利用 CodeCarbon 等工具测量能耗、碳排放与执行时间，评估不同模型、量化级别与提示结构的可持续性与测试覆盖率。

**💡 创新点**

创新点在于提出了可持续性速度指数 SVI 与绿色 F_β 得分 GF_β 两个综合指标，并将区域电网碳强度纳入评估，系统地比较小语言模型在能耗、碳足迹与测试质量之间的权衡。

**🔧 技术方法**

技术手段包括：提示工程（AP_V_0–AP_V_3）、模型量化（4-bit、8-bit、无量化）、CodeCarbon、coveragepy 代码覆盖测量、Google Colab 迁移至不同区域的数据中心、统一实验流程与多维度指标计算。

**📊 数据集**

使用的数据集为 OpenAI HumanEval（164 题 Python 编程任务）以及其对应的单元测试脚本，构造统一可运行文件后进行推理。

**📈 对比分析**

比较方法是对同一模型在不同提示与量化配置下的能耗、碳排放、执行时间与覆盖率进行归一化后计算 SVI 与 GF_β 指标；实验结果显示不同模型与提示组合在能耗/碳排放上存在显著差异，部分高精度配置在低碳地区可实现更优的综合性能。

**⚠️ 局限性**

局限性包括：仅评估了单一 HumanEval 基准和固定 GPU（NVIDIA T4）环境，碳强度采用估计值且未实时测量；实验样本量有限，结果可能受区域迁移与后台进程等不可控因素影响，且覆盖率仅作为生成脚本质量的初步代理。

---

## 201. Understanding Latent Diffusability via Fisher Geometry

**arXiv ID:** 2604.02751 | [PDF](https://arxiv.org/pdf/2604.02751v1)

**作者:** Jing Gu `[一作]` (University of Minnesota), Gilad Lerman `[通讯]` (University of Minnesota)

**通讯引用:** 2538 | [OpenAlex ID](https://openalex.org/A5088795209)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `57a58b01-81b4-4d75-a45c-2e891f272b50` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了基于 Fisher 信息几何的框架，用于定量评估潜在空间的可扩散性。

**💡 创新点**

创新点在于将均方误差变化率分解为 Fisher 信息和 Fisher 信息率两项，明确了局部几何扭曲与维度压缩对扩散的独立影响。

**🔧 技术方法**

使用 Fisher 信息、Fisher 信息率、MMSE 等信息理论工具，并通过热扩散与 Hessian 链式法则推导。

**📊 数据集**

实验数据集包括二维高斯 toy 以及 FFHQ、NVAE 生成的潜在表示。

**📈 对比分析**

通过比较 FI、FIR 及其偏差指标与像素空间模型的性能，证明 GPE 编码在维度压缩与曲率注入方面优于 VAE，且指标能预测扩散失败。

**⚠️ 局限性**

局限在于对非线性编码的高频曲率控制仍依赖经验正则化，理论上对优化动态的分析尚未完备。

---

## 202. Learning Locomotion on Complex Terrain for Quadrupedal Robots with Foot Position Maps and Stability Rewards

**arXiv ID:** 2604.02744 | [PDF](https://arxiv.org/pdf/2604.02744v1)

**作者:** Matthew Hwang `[一作]` (University of Tokyo), Takeshi Oishi `[通讯]` (University of Tokyo)

**通讯引用:** 5925 | [OpenAlex ID](https://openalex.org/A5021438586)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文通过在高度图中嵌入足位图、引入基于中心压力的动态稳定奖励以及全局速度跟踪，构建了一种强化学习控制策略，显著提升了四足机器人在复杂地形上的稳定性和通行成功率。

**💡 创新点**

创新点在于：①将足位信息显式融入高度图并通过注意力机制直接给出足位信息；②使用基于中心压力（CoP）的动态稳定奖励取代传统静态或捕捉点奖励；③采用全局速度跟踪以防止机器人利用局部命令规避难以通过的地形。

**🔧 技术方法**

技术方法包括：注意力编码的高度图网络（MHA+CNN）、PPO强化学习、正交运动学、动态稳定奖励计算、全局速度指令转换、以及对机器人动力学的PD控制。

**📊 数据集**

主要数据集为在仿真环境中生成的多种地形（光滑、粗糙、阶梯、碎石等）以及组合型 OOD 地形，另外还在 Gazebo 仿真中使用 Unitree A1 机器人配合 Ouster LIDAR 进行仿真对比。

**📈 对比分析**

与 MLP、CNN、Transformer、Attention 等基线方法对比，所提方法在 1000 条仿真轨迹上实现约 77% 的成功率，跟踪误差下降至 0.169 m/s，功耗最低 58.5 W，显示出明显的性能提升。

**⚠️ 局限性**

局限性包括：仅在仿真环境验证，缺乏真实世界的转移实验；对极端复杂地形的动态稳定奖励仍可能出现误差；全局速度跟踪对动态障碍物的适应性尚未充分评估。

---

## 203. DeCo-DETR: Decoupled Cognition DETR for efficient Open-Vocabulary Object Detection

**arXiv ID:** 2604.02753 | [PDF](https://arxiv.org/pdf/2604.02753v1)

**作者:** Siheng Wang `[一作]` (Jiangsu University), Qiang Sun `[通讯]` (MBZUAI)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DeCo-DETR框架，用于开放词汇目标检测，解耦语义推理与定位，并通过动态层级概念池实现推理时不依赖文本编码器。

**💡 创新点**

创新点包括：①动态层级概念池构造视觉语言原型空间；②层级知识蒸馏将检测查询映射到原型；③参数化解耦训练将定位与语义对齐分离，缓解优化冲突。

**🔧 技术方法**

使用的技术包括：CLIP跨模态对齐、LLaVA文本生成、K-means+DBSCAN层级聚类、动量更新、DETR Transformer、层级知识蒸馏、Cosine annealing权重、梯度隔离、蒸馏等。

**📊 数据集**

在OV-COCO和OV-LVIS这两个开放词汇目标检测基准上进行实验。

**📈 对比分析**

与Deformable DETR、ViLD、DetPro、DK-DETR、CAKE等多种基线比较，OV-COCO AP50novel达到41.3%（比最高基线提高约3.5%），OV-LVIS APr/overall AP分别为29.4%/35.2%，推理时间仅135ms，显著低于需使用文本编码器的方案。

**⚠️ 局限性**

局限性：仍需依赖预训练大VLM产生区域描述，对VLM规模敏感；层级原型数量需经验调参；在极端长尾或高分辨率场景下推理仍有一定成本；未在跨域或多模态外部任务中进行验证。

---

## 204. Visual Instruction-Finetuned Language Model for Versatile Brain MR Image Tasks

**arXiv ID:** 2604.02748 | [PDF](https://arxiv.org/pdf/2604.02748v1)

**作者:** Jonghun Kim `[一作]` (Sungkyunkwan University), Hyunjin Park `[通讯]` (Sungkyunkwan University)

**通讯引用:** 16635 | [OpenAlex ID](https://openalex.org/A5101720008)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `e15e3743-5ee0-4d5f-813d-d146868082fc` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

构建并训练了一个视觉指令微调的LLM，统一完成脑MR图像的报告生成、视觉问答、图像分割与图像翻译任务。

**💡 创新点**

创新点在于将图像token与LLM结合并通过可学习的跳跃连接和零初始化卷积提升图像细节，同时使用多模型协同生成医学文本，避免幻觉。

**🔧 技术方法**

采用LLaMA基础LLM、VQ‑GAN图像token化、指令微调、BiomedCLIP文本编码器、prompt tuning及零初始化跳跃卷积。

**📊 数据集**

使用BraTS2021、BraTS2023‑MEN、IXI、ATLAS 2.0及外部UPENN‑GBM等脑MR数据集进行训练与评测。

**📈 对比分析**

与LLaVA‑Med、LLaVA、LLama 3.2、Gemini 1.5、GPT‑4o‑mini等模型对比，报告生成ROUGE/METEOR/BERT‑F1、VQA准确率、分割Dice、图像翻译PSNR/SSIM/FID均显著优于基线，甚至优于专用任务模型。

**⚠️ 局限性**

局限在于文本数据多为合成，缺乏临床注释验证，实验仅限于脑MR，未验证在其他部位或模态的泛化能力。

---

## 205. THOM: Generating Physically Plausible Hand-Object Meshes From Text

**arXiv ID:** 2604.02736 | [PDF](https://arxiv.org/pdf/2604.02736v1)

**作者:** Uyoung Jeong `[一作]` (UNIST), Kwang In Kim `[通讯]` (POSTECH)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出一种训练免费、两阶段的文本到3D手-物体交互（HOI）生成框架，直接从文本提示生成高真实感、物理可行的3D HOI网格。

**💡 创新点**

创新点包括：① 通过 vertex‑Gaussian 一对一映射与简化 Poisson 重建实现精确的网格提取；② 利用 VLM（InternVL3.5-14B）进行无训练的手位移引导；③ 结合距离自适应接触损失、重定位损失等物理基优化，提升交互可穿透度和接触合理性。

**🔧 技术方法**

核心技术包括：文本到图像扩散模型 Score Distillation Sampling（ISM）指导 3D 高斯 Splatting；Poisson 体素重建与顶点上采样；Laplacian 正则化；InternVL3.5-14B 进行 VLM 引导；多种物理基损失（穿透、接触、重定位、一致性）。

**📊 数据集**

使用的主要数据集有：Objaverse（约 818K 物体）、T³Bench（物体提示）、GPT‑4o 生成的 100 条手提示，并手动配对以形成 100 条多样化 HOI 文本。

**📈 对比分析**

与 DreamFusion、ProlificDreamer、GaussianDreamerPro、Hash3D、InterFusion、DreamHOI 等 SOTA 进行对比；在 CLIP、T³‑Alignment、穿透深度、接触比例等指标上，THOM 取得 CLIP 31.4、T³‑Alignment 2.6、最大穿透 2.2×10⁻⁵、接触率 0.95，显著优于同类方法；生成时间约 2.5 小时，比 GaussianDreamerPro 等大幅加速。

**⚠️ 局限性**

局限性在于：生成速度仍高于基于学习的端到端方法；对极端噪声文本生成的网格仍可能出现非水密性或不规则拓扑；手/物体提示配对需人工干预，限制了大规模自动化应用。

---

## 206. Multi-agent Reinforcement Learning-based Joint Design of Low-Carbon P2P Market and Bidding Strategy in Microgrids

**arXiv ID:** 2604.02728 | [PDF](https://arxiv.org/pdf/2604.02728v1)

**作者:** Junhao Ren `[一作]` (Nanyang Technological University), Gaoxi Xiao `[通讯]` (Nanyang Technological University)

**通讯引用:** 6918 | [OpenAlex ID](https://openalex.org/A5005659262)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `5a41884c-404f-4688-a89c-aa238c10fe68` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了基于多智能体强化学习的分布式P2P电力交易框架，旨在实现微电网社区的低碳目标与经济效益兼顾。

**💡 创新点**

创新点在于将碳排放目标嵌入双拍卖清算机制中，并采用DEC‑POMDP建模结合LSTM‑MAPPO实现微电网自主决策与协同。

**🔧 技术方法**

使用了多智能体强化学习（LSTM‑MAPPO）、双拍卖清算、DEC‑POMDP建模、集中训练分布式执行（CTDE）等技术。

**📊 数据集**

实验数据基于澳大利亚四户住宅的真实负荷与光伏发电记录，经过一整年采样后归一化得到24小时基准曲线，并加入高斯噪声和光伏故障扰动。

**📈 对比分析**

与Greedy、MRDA、VVDA双拍卖机制及MADDPG、MADDPG‑GCN、IPPO、MAPPO、MASAC等MARL算法进行对比，JPQ+LSTM‑MAPPO在总收益、紧急采购、光伏出口与储能利用率上均优于基线，提升幅度约30–40%。

**⚠️ 局限性**

局限性包括未考虑线损/潮流约束、日常采购决策、隐私保护学习以及更大规模网络的可扩展性等问题。

---

## 207. V2X-QA: A Comprehensive Reasoning Dataset and Benchmark for Multimodal Large Language Models in Autonomous Driving Across Ego, Infrastructure, and Cooperative Views

**arXiv ID:** 2604.02710 | [PDF](https://arxiv.org/pdf/2604.02710v1)

**作者:** Junwei You `[一作]` (University of Wisconsin–Madison), Bin Ran `[通讯]` (Southeast University)

**通讯引用:** 10067 | [OpenAlex ID](https://openalex.org/A5060394098)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文设计并实现了 V2X-QA 这一多视角问答基准，覆盖车辆侧、基础设施侧以及协同视角的多模态大型语言模型评估，并构建了 33,216 条专家核对的多选问题数据集以及对应的评测协议。

**💡 创新点**

核心创新在于视图解耦的评估框架、12 个跨视角任务分类以及通过显式视图路由和视角专门化 LoRA 进行的多视角自适应模型 V2X-MoE。

**🔧 技术方法**

采用 Qwen3-VL 作为冻结的多模态主干，结合 LoRA 专家与显式视图路由，并使用 MCQA 提示与生成、置信度校准等技术。

**📊 数据集**

数据来源于真实车辆–基础设施协同驾驶数据集 V2X-Seq，经过专家标注形成 33.2k 条多选问答实例。

**📈 对比分析**

通过统一的多选问答评测，对比 10+ 公开与闭源 MLLM 以及 V2X-MoE，结果显示单视角模型性能远低于 V2X-MoE，协同视角最难，而 V2X-MoE 在所有视角均取得 90% 以上的准确率。

**⚠️ 局限性**

局限性包括仅基于图像的 MCQA 而非视频/闭环驾驶，未考虑通信延迟与丢包等动态因素，且可靠性分析仅局限于 V2X-MoE。

---

## 208. TypePro: Boosting LLM-Based Type Inference via Inter-Procedural Slicing

**arXiv ID:** 2604.02702 | [PDF](https://arxiv.org/pdf/2604.02702v1)

**作者:** Teyu Lin `[一作]` (Xiamen University), Rongxin Wu `[通讯]` (Xiamen University)

**通讯引用:** 4245 | [OpenAlex ID](https://openalex.org/A5054822682)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研发了一种基于跨过程代码切片和大型语言模型（LLM）的动态语言类型推断方法TypePro。

**💡 创新点**

通过系统依赖图实现完整跨过程切片，并利用结构相似度推荐候选类型，弥补LLM缺乏第三方库和自定义类型知识的不足。

**🔧 技术方法**

结合LLM生成、生成式抽样投票、BM25相似度匹配以及程序依赖图（PDG/SDG）进行代码切片与类型推断。

**📊 数据集**

使用了Python和TypeScript的ManyTypes4Py与ManyTypes4TypeScript数据集。

**📈 对比分析**

与HiTyper、Type4Py、CodeT5/CodeT5+、UnixCoder、TypeGen、TIGER等多种基线在Top‑1/3/5 Exact Match、Base Match和MRR@5指标上对比，TypePro在Python Top‑1 EM提升7–10%并在TypeScript同样取得最高准确率。

**⚠️ 局限性**

受限于LLM生成多样性与温度设定导致Top‑3/5下降，静态切片无法覆盖所有动态特性，且候选类型阈值的选择会影响推断结果。

---

## 209. AnnoRetrieve: Efficient Structured Retrieval for Unstructured Document Analysis

**arXiv ID:** 2604.02690 | [PDF](https://arxiv.org/pdf/2604.02690v1)

**作者:** Teng Lin `[一作]` (Hong Kong University of Science and Technology), Nan Tang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 6664 | [OpenAlex ID](https://openalex.org/A5062243169)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了 AnnoRetrieve 系统，完成了从无结构文档到可查询结构化知识索引的端到端转换，并实现了基于结构化注解的检索流程。

**💡 创新点**

创新点包括：① SchemaBoot 自动生成并优化任务特定的字段与层次化 schema，消除人工 schema 设计；② SSR（Structured Semantic Retrieval）引入结构化查询与语义解析的混合检索引擎，实现无 LLM 成本的精准答案；③ 提出了“结构优先、检索后”的新检索范式，显著降低了向量检索与 LLM 调用的开销。

**🔧 技术方法**

技术手段包括：多粒度模式挖掘与约束优化、NSGA‑II 多目标优化、GliNER2 进行字段级抽取、SQL+文本扫描的混合查询、轻量化正则/文本扫描函数、LLM 仅用于生成 schema 说明与有限后处理。

**📊 数据集**

使用了三大真实数据集：LCR（法律案例，100份），WikiText（多领域维基页面，200份）和 SWDE（结构化网页数据，200份）。

**📈 对比分析**

与 VectorDB、Graph RAG、ZenDB、Palimpsest、QUEST、LLM（GPT‑4）等基线进行对比；AnnoRetrieve 在三类数据集上的平均 F1 达到 0.87，LLM 调用成本约 29.4k tokens，平均延迟 3.2 秒，显著优于 VectorDB（0.41 F1，15.2k tokens，1.8s）、Graph RAG（0.58 F1，~150k tokens），以及纯 LLM（0.72 F1，312.8k tokens）等。

**⚠️ 局限性**

局限性包括：① 需要先完成 schema 诱导，对极端动态或多主题的文档集可能需要频繁重构；② 目前主要针对文本字段的属性‑值抽取，复杂跨文档关系图谱构建尚未涵盖；③ 在非常大规模文档时，离线结构化阶段仍需一定资源，且对极稀疏或无结构文本的抽取效果有限。

---

## 210. IndustryCode: A Benchmark for Industry Code Generation

**arXiv ID:** 2604.02729 | [PDF](https://arxiv.org/pdf/2604.02729v1)

**作者:** Puyu Zeng `[一作]` (Shanghai Jiao Tong University), Linfeng Zhang `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 15020 | [OpenAlex ID](https://openalex.org/A5100689117)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并发布了IndustryCode基准，用于评估大型语言模型在真实工业场景下的代码生成与理解能力。

**💡 创新点**

首次构建跨行业、多语言、层次化的工业代码基准，包含从实际生产代码改造而来的主问题与子问题，并引入人工-LLM协同的严谨标注与去污染流程。

**🔧 技术方法**

采用零样本提示、累积上下文编码、LLM‑Judge语义评估、执行型数值验证、GPT‑5/Claude等先进模型进行评测，同时使用人机交互式迭代验证提升数据质量。

**📊 数据集**

125个主问题、579个子问题，覆盖Python、C++、MATLAB、Stata四种语言，涵盖金融、自动化、航空航天、遥感等20个子领域，全部来自行业真实生产代码。

**📈 对比分析**

对比多款主流专有模型（Claude 4.5 Opus、Sonnet、Haiku；GPT‑5；Gemini）与开源模型（Qwen‑3、DeepSeek、Doubao、GLM、Kimi）进行Pass@1评估；顶尖模型Claude 4.5 Opus在子问题上达68.1%、主问题42.5%，开源模型在C++子问题上最高70.4%。

**⚠️ 局限性**

仍存在语法错误、误解题意、hallucination等错误；模型对极其专业领域（如半导体、微电子）的理解不足；基准规模虽大但仍不覆盖全部工业语言与工具；缺乏针对复杂项目持续交互与多文件依赖的评测。

---

## 211. ExploreVLA: Dense World Modeling and Exploration for End-to-End Autonomous Driving

**arXiv ID:** 2604.02714 | [PDF](https://arxiv.org/pdf/2604.02714v1)

**作者:** Zihao Sheng `[一作]` (Bosch Research North America & Bosch Center for Artificial Intelligence), Liu Ren `[通讯]` (Bosch Research North America & Bosch Center for Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种统一的理解与生成框架，利用未来RGB与深度图像生成实现世界建模，并将世界模型的不确定性作为自我探索奖励，提升端到端视觉语言动作驱动的自动驾驶性能。

**💡 创新点**

创新点在于将世界建模与探索奖励融合：未来图像生成提供密集监督；世界模型不确定性被用于安全门控的探索奖励；结合Group Relative Policy Optimization实现鲁棒的RL后训练。

**🔧 技术方法**

采用Vision‑Language‑Action (VLA) 统一模型、MAGVIT‑v2离散图像量化、离散概率生成、组相对策略优化 (GRPO) 与安全门控奖励。

**📊 数据集**

在NAVSIM（v1、v2）与nuScenes两个公开自动驾驶基准数据集上进行评估。

**📈 对比分析**

与多种基线（单视角与多模态 VLA、轨迹生成、Diffusion 等）对比，ExploreVLA在NAVSIM v1上PDMS 93.7、v2上EPDMS 88.8，均为最高或接近最高，显著优于现有方法。

**⚠️ 局限性**

仅使用单视角前置摄像头；未充分利用多视角或其他生成目标（如鸟瞰图），可能限制在更复杂场景下的空间感知与鲁棒性。

---

## 212. Evaluating the Formal Reasoning Capabilities of Large Language Models through Chomsky Hierarchy

**arXiv ID:** 2604.02709 | [PDF](https://arxiv.org/pdf/2604.02709v1)

**作者:** Yihong Dong `[一作]` (Peking University), Ge Li `[通讯]` (Peking University)

**通讯引用:** 14332 | [OpenAlex ID](https://openalex.org/A5100447673)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了ChomskyBench基准，用于系统评估大型语言模型在正式推理方面的能力，覆盖从正则到递归可枚举的完整Chomsky层次；

**💡 创新点**

创新点在于：①使用完整Chomsky层次覆盖、②要求模型生成可验证的推理过程（自然语言过程轨迹），③通过确定性符号验证器保证结果可复现；

**🔧 技术方法**

技术主要包括Transformer语言模型推理、链式思考（CoT）提示、最佳采样（Best‑of‑N）与长推理长度扩展、定量评估指标ACC/PR、时间复杂度与效率对比；

**📊 数据集**

使用由研究团队人工生成、无数据污染的正式语言任务集，包含115个任务，覆盖正则、确定性/非确定性上下文无关、上下文相关、递归可枚举四个层级；

**📈 对比分析**

与多款闭源/开源SOTA LLM（如o3、gpt‑5、gemini‑2.5‑pro、deepseek‑v3.1、qwen3等）在ChomskyBench上对比；结果显示性能随Chomsky层级升高而递减，存在明显的上下文无关→上下文相关“断崖”；最佳模型o3平均ACC≈0.28，PR≈0.56，测试时间相比传统程序慢10⁴–10⁵倍；

**⚠️ 局限性**

局限性：①当前LLM缺乏高阶形式推理的稳健机制，尤其是深层递归、状态跟踪与长程依赖；②测试时间扩展虽提升效果，但要达99%+准确率需数千甚至万次采样，成本不可接受；③整体效率远低于传统算法，难以直接用于安全关键的形式验证任务。

---

## 213. DocShield: Towards AI Document Safety via Evidence-Grounded Agentic Reasoning

**arXiv ID:** 2604.02694 | [PDF](https://arxiv.org/pdf/2604.02694v1)

**作者:** Fanwei Zeng `[一作]` (Ant Group), Yin Yan `[通讯]` (Ant Group)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了DocShield，一种统一生成式框架，用于文本中心的文档伪造检测、空间定位和可解释说明。

**💡 创新点**

创新点包括跨视觉与逻辑线索的链式思考（CCT）机制以及通过GRPO优化的加权多任务奖励函数，能够实现证据驱动的联合推理，显著降低推理幻觉和误差传递。

**🔧 技术方法**

核心技术为跨线索链式思考（CCT）、加权多任务奖励（格式、定位、解释三项）、GRPO强化学习、LLM（如Qwen2.5-VL-7B）与OCR、知识检索等多模态处理。

**📊 数据集**

使用了新构建的RealText‑V1基准（5397张多语言、细粒度标注的文档图像），以及公开数据集T‑IC13和T‑SROIE进行零样本和鲁棒性评估。

**📈 对比分析**

与多项M​​LLM和专用取证模型比较，DocShield在RealText‑V1上获得M‑F1 68.9%，Detection 91.4%，Grounding mIOU 32.4%，在T‑IC13上零样本M‑F1 79.8%，在T‑SROIE上保持高mIOU 61.3%等，均优于现有最优方法。

**⚠️ 局限性**

局限性包括：在低分辨率或视觉混乱布局下易出现误报；在逻辑完全一致但视觉无异常的伪造中误检；对极端视觉失真（如高噪声、强模糊）仍存在性能下降，提示需进一步提升跨模态鲁棒性。

---

## 214. Adaptive Semantic Communication for Wireless Image Transmission Leveraging Mixture-of-Experts Mechanism

**arXiv ID:** 2604.02691 | [PDF](https://arxiv.org/pdf/2604.02691v1)

**作者:** Haowen Wan `[一作]` (Zhejiang University), Qianqian Yang `[通讯]` (Zhejiang University)

**通讯引用:** 13888 | [OpenAlex ID](https://openalex.org/A5076730859)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了一种基于自适应Mixture-of-Experts Swin Transformer的多阶段端到端语义通信系统，用于MIMO衰落信道下的无线图像传输。

**💡 创新点**

创新点在于：①联合利用实时CSI和图像语义特征驱动专家路由；②设计动态门控机制，根据权重差阈值自适应决定激活的专家数量；③构建多阶段体系结构，实现高效、可扩展的语义编码与解码。

**🔧 技术方法**

主要技术包括：Swin Transformer、Adaptive MoE MLP、动态专家门控、联合路由、MIMO信道建模，以及多项损失（MSE、负载平衡、熵正则、方差正则）共同训练。

**📊 数据集**

数据集使用DIV2K进行训练，评估采用Kodak数据集；此外在CLIC2021上测试专家激活情况。

**📈 对比分析**

与DeepJSCC和未加入自适应模块的SwinJSCC进行对比，在多种SNR和带宽比下，本文方法在PSNR和LPIPS指标上均优于两者，显示出更高的图像重建质量和带宽效率。

**⚠️ 局限性**

局限性包括：对大规模专家集的可扩展性仍需验证；在极低SNR或极低带宽比下性能衰减；训练依赖大量标注数据；模型对不同信道模型的泛化能力尚待进一步验证。

---

## 215. Aligning Progress and Feasibility: A Neuro-Symbolic Dual Memory Framework for Long-Horizon LLM Agents

**arXiv ID:** 2604.02734 | [PDF](https://arxiv.org/pdf/2604.02734v1)

**作者:** Bin Wen `[一作]` (Nanjing University), Lan-Zhe Guo `[通讯]` (Nanjing University)

**通讯引用:** 371 | [OpenAlex ID](https://openalex.org/A5047808444)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在长时程任务中提出神经符号双记忆框架，分离语义进度引导与可执行性验证；

**💡 创新点**

创新点在于将全局进度对齐与局部可行性对齐分别使用神经记忆与符号规则实现，实现双重对齐；

**🔧 技术方法**

采用神经网络进度记忆、符号可行性记忆、可执行Python验证器、蓝图检索等技术；

**📊 数据集**

在ALFWorld、WebShop和TextCraft三大长时程基准上进行评估；

**📈 对比分析**

与ReAct、Reflexion、ADaPT、StateAct、ExpeL、WALL‑E 2.0、AWM等基线比较，成功率分别提升至94.78%、51%和94%，并显著降低无效动作率和轨迹长度；

**⚠️ 局限性**

局限性是需要离线收集足够的轨迹数据，在奖励稀疏或失败信号难以解释的环境中效果可能受限。

---

## 216. DeltaLogic: Minimal Premise Edits Reveal Belief-Revision Failures in Logical Reasoning Models

**arXiv ID:** 2604.02733 | [PDF](https://arxiv.org/pdf/2604.02733v1)

**作者:** Amit Dhanda `[一作]` `[通讯]` (Amazon), Amit Dhanda (Amazon)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出DeltaLogic基准，评估大型语言模型在最小证据变更下的信念修订能力。

**💡 创新点**

创新性地将传统逻辑推理实例转化为最小编辑的修订情节，并引入惯性、过度翻转、放弃等细粒度修订指标，揭示推理与修订能力的分离。

**🔧 技术方法**

通过构造协议对FOLIO、ProofWriter等数据集实例进行最小编辑，生成修订剧本，并采用冻结因果LM的Token log-likelihood进行无后处理评估。

**📊 数据集**

使用FOLIO和ProofWriter两大公共逻辑推理数据集，构造了30条Qwen子集和20条near‑4B子集的修订实例。

**📈 对比分析**

对Qwen3‑0.6B、1.7B、4B以及Phi‑4‑mini‑instruct等模型进行比较，衡量初始准确率、修订准确率、惯性率等指标；结果显示，即使模型在初始推理上表现优秀，修订准确率仍较低，惯性率高，表明修订是一个独立挑战。

**⚠️ 局限性**

局限性包括：仅测试单步修订且数据规模有限；CPU受限导致评估规模不均衡；缺乏开放世界交互和长链修订评估，无法覆盖更复杂的真实场景。

---

## 217. Sustainability Analysis of Prompt Strategies for SLM-based Automated Test Generation

**arXiv ID:** 2604.02761 | [PDF](https://arxiv.org/pdf/2604.02761v1)

**作者:** Pragati Kumari `[一作]` (University Of Calgary), Novarun Deb `[通讯]` (University Of Calgary)

**通讯引用:** 159 | [OpenAlex ID](https://openalex.org/A5072598400)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对小型语言模型在自动单元测试生成中的提示策略进行可持续性和覆盖率质量评估。

**💡 创新点**

首次从可持续性角度系统比较不同提示策略，提出统一的综合指标 SQScore，并证明提示设计对能耗和碳排放影响大于模型选择。

**🔧 技术方法**

使用三款开源小型模型（DeepSeek‑Coder‑7B、Meta‑Llama‑3‑8B、Mistral‑7B）在 4‑bit 量化下运行；通过 CodeCarbon 记录 CPU/GPU/RAM 能耗与碳排放；用 coverage.py 评估测试覆盖率；计算 token、时间、能耗、碳排放的 1K 归一化指标及 QPerkWh、QPerCO₂ 等覆盖效率指标。

**📊 数据集**

MBPP（Mostly Basic Python Problems）数据集，共 98 个 Python 任务。

**📈 对比分析**

对 21 种模型‑提示组合进行批量实验，计算 token‑normalised efficiency、coverage‑efficiency 以及综合 SQScore；结果显示 Zero‑Shot / ReAct 等轻量提示在能耗/碳排放上优于 Chain‑of‑Thought 或 Self‑Consistency，后者虽然在覆盖率上略有提升，但成本显著更高。

**⚠️ 局限性**

实验局限于单一数据集、单一硬件（NVIDIA A100）和单一能源强度估算；未考虑模型训练成本、不同编程语言或工业真实环境；仅使用代码覆盖率作为质量度量；未进行统计显著性检验。

---

## 218. OMNI-PoseX: A Fast Vision Model for 6D Object Pose Estimation in Embodied Tasks

**arXiv ID:** 2604.02759 | [PDF](https://arxiv.org/pdf/2604.02759v1)

**作者:** Michael Zhang `[一作]` (KernalMind Tech Company Limited), Hanwen Kang `[通讯]` (KernalMind Tech Company Limited)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `a8e75ba4-7a2d-4153-b003-06c94533add0` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出OMNI-PoseX，用于开放世界的6D姿态估计

**💡 创新点**

SO(3)-aware反射流匹配框架、轻量化多模态融合与大规模数据训练

**🔧 技术方法**

SO(3)几何流匹配、FiLM融合、RK2积分、DINO+PointNet++特征

**📊 数据集**

Omni6DPose、Omni6D（真实与合成数据）

**📈 对比分析**

与NOCS、SGPA、IST-Net、HS-Pose、GenPose++对比，在Open-World 6D姿态基准上AUC/VUS领先，实时推理仅17-86ms

**⚠️ 局限性**

受限于开源视觉骨干的语义分割、对极端遮挡、对称性、尺度不确定性缺乏高阶跨模态交互和不确定性建模

---

## 219. Optimal Pricing with Unreliable Signals

**arXiv ID:** 2604.02758 | [PDF](https://arxiv.org/pdf/2604.02758v1)

**作者:** Zhihao Gavin Tang `[一作]` (Shanghai University of Finance and Economics), Shixin Wang `[通讯]` (Georgia Institute of Technology)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

研究在卖家持有不可靠侧信息（可能准确也可能是幻觉）时，单买家单品的定价问题，提出一致性-鲁棒性框架，并在无先验知识的前提下设计机制。

**💡 创新点**

① 在信息不对称的高阶设置下证明隐藏信号能显著提升收益；② 完整刻画一致性与鲁棒性之间的 Pareto 前沿；③ 在完全一致性的前提下仍能获得 0.5 的鲁棒性，并对某些分布实现完美一致性和完美鲁棒性。

**🔧 技术方法**

线性规划（LP）与其对偶、凸分析、可逆约束（上限效用约束）、最优分布与阈值结构、随机价格（随机售卖）以及多阶段的数值优化。

**📊 数据集**

无实验数据集，全部为理论模型与数学证明。

**📈 对比分析**

与公开信号基准（C+R≤1）进行比较，得到更优的 trade‑off，最优对称点约为 0.822（比基准的 0.5 更好）。在完美一致性下，所有分布都能实现 0.5 鲁棒性；若分布满足均值≤垄断价或均值无穷大，则可实现完美一致性+完美鲁棒性。

**⚠️ 局限性**

① 仅考虑单买家单品；② 信号要么完全准确，要么完全无关，未考虑混合情况；③ 结果仅在理论框架下，缺乏实际应用验证；④ 未探讨多买家或动态定价场景。

---

## 220. Differentiable Stroke Planning with Dual Parameterization for Efficient and High-Fidelity Painting Creation

**arXiv ID:** 2604.02752 | [PDF](https://arxiv.org/pdf/2604.02752v1)

**作者:** Jinfan Liu `[一作]` (Shanghai Jiao Tong University), Bingbing Ni `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 13916 | [OpenAlex ID](https://openalex.org/A5014362734)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种结合离散多边形与连续贝塞尔曲线的双模笔画表示，并通过结构感知搜索与梯度优化的两阶段管线实现高质量、少量笔画的绘画渲染。

**💡 创新点**

创新点在于可微双向映射的双模笔画表示、Gaussian‑splatting式并行初始化以及结构引导的搜索-优化协同机制，使笔画数减少30–50%，结构性和速度显著提升。

**🔧 技术方法**

使用可微分多边形↔贝塞尔拟合/采样、梯度驱动的结构搜索、基于高斯splatting的可微分多边形渲染器、学习透明度与高度重光照，以及GPU并行的优化流程。

**📊 数据集**

在DIV2K验证集（1200×1200）和Im2Oil Gallery（600×800）上进行实验。

**📈 对比分析**

与五种主流SBR方法对比，DIV2K上PSNR 32.16、SSIM 0.93、耗时87.6 s；Im2Oil上PSNR 32.53、SSIM 0.86、耗时42.7 s；用户研究显示在结构、纹理、色彩和整体偏好上均位居榜首。

**⚠️ 局限性**

局限在于对初始种子与梯度方向的依赖，在极低笔画预算或极细纹理情况下仍需更多迭代；实验依赖单张RTX 4090 GPU，且对极其复杂纹理细节的捕捉仍有限。

---

## 221. A Rapid Instrument Exchange System for Humanoid Robots in Minimally Invasive Surgery

**arXiv ID:** 2604.02707 | [PDF](https://arxiv.org/pdf/2604.02707v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 222. GrandCode: Achieving Grandmaster Level in Competitive Programming via Agentic Reinforcement Learning

**arXiv ID:** 2604.02721 | [PDF](https://arxiv.org/pdf/2604.02721v1)

**作者:** DeepReinforce Team `[一作]`, Jiwei Li `[通讯]`

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

开发了 GrandCode，一个多代理强化学习系统，能够在实时竞赛中自动解决编程问题。

**💡 创新点**

创新点在于提出 Agentic GRPO 以处理多阶段奖励延迟和离策略漂移，并将推理、假设生成、摘要和对抗测试生成等多个模块协同工作，形成完整的 agentic 反馈循环。

**🔧 技术方法**

采用多代理强化学习、Agentic GRPO、Post‑Training RL、Test‑Time RL、pipeline‑RL、DeltaNet+Softmax 混合、上下文并行、动态 CP、假设生成模型、摘要模型和对抗测试生成等技术。

**📊 数据集**

训练数据来源于 TACO、LeetCode、USACO、CodeContests、IOI 等公开竞赛数据集，并通过 Gemini、Claude、GPT 等 LLM 生成的扩展语料；评估基准以 Codeforces 题目及其官方解答为主。

**📈 对比分析**

在三场 Codeforces 实时赛中，GrandCode 取得第一名，S(separate) 与 S(joint) 分别为 9269/8334、16511/15008、11596/9506，优于所有人类参赛者；与 Gemini 3 Deep Think、Claude、GPT 等模型对比，后者在公开评测中的接受率约 70–75%、Level 5 解决率 35–40%，而 GrandCode 在完整训练+测试后达到 85% 接受率、15/20 Level 5。

**⚠️ 局限性**

限制包括对昂贵 LLM 的高计算成本、在 Codeforces 反 AI 政策下可能被检测、对极端大规模或未知结构问题的泛化能力有限，以及缺乏可解释性和对抗鲁棒性评估。

---

## 223. Generative Frontiers: Why Evaluation Matters for Diffusion Language Models

**arXiv ID:** 2604.02718 | [PDF](https://arxiv.org/pdf/2604.02718v1)

**作者:** Patrick Pynadath `[一作]` (Purdue University), Ruqi Zhang `[通讯]` (Purdue University)

**通讯引用:** 4541 | [OpenAlex ID](https://openalex.org/A5101586017)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出基于生成前沿的评估框架，用来系统比较扩散语言模型的生成质量；

**💡 创新点**

将生成困惑度和单字汇数熵解释为KL散度的两项分量，揭示单点评估的歧义并提供完整的熵‑困惑度曲线对比方法；

**🔧 技术方法**

利用扩散模型推理、温度调节、交叉熵与熵估计，构建生成前沿；

**📊 数据集**

以OpenWebText为预训练与验证语料，使用GPT‑2 Large做参照分布；

**📈 对比分析**

通过生成前沿匹配熵或困惑度来比较模型，发现早期训练（5万步）前沿与完整训练（100万步）极为相近，表明可在较低成本下评估模型潜力；

**⚠️ 局限性**

假设前沿分析需要参考模型可靠、生成困惑度准确反映交叉熵、单字熵可近似联合熵；若这些假设失效，评估结果可能失真。

---

## 224. FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving

**arXiv ID:** 2604.02715 | [PDF](https://arxiv.org/pdf/2604.02715v1)

**作者:** Qingxiu Liu `[一作]` (Chinese University of Hong Kong), Patrick P. C. Lee `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 7903 | [OpenAlex ID](https://openalex.org/A5012129385)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种MoE推理框架，使用专家分页将专家参数从GPU内存中解耦，动态流式加载权重。

**💡 创新点**

创新点在于专家分页抽象、PagedTensor虚拟张量、带宽平衡的存储层次以及预算感知的驻留规划器。

**🔧 技术方法**

采用CUDA VMM、异步流、压缩GPU存储、CPU offload、异步DMA、Huffman压缩以及闭环控制器等技术。

**📊 数据集**

使用ShareGPT对话数据集进行推理工作负载。

**📈 对比分析**

与vLLM、vLLM-O和-H对比，实验显示在大batch/长context下，最高可比vLLM提升约3.0×，在内存受限场景下也显著优于基线，并保持模型精度。

**⚠️ 局限性**

局限性包括对压缩可行性的依赖、PCIe带宽瓶颈、对动态KV缓存扩容的支持不足以及需在特定硬件（如NVIDIA L40）上验证。

---

## 225. Trivial Vocabulary Bans Improve LLM Reasoning More Than Deep Linguistic Constraints

**arXiv ID:** 2604.02699 | [PDF](https://arxiv.org/pdf/2604.02699v1)

**作者:** Rodney Jehu-Appiah `[一作]` `[通讯]`, Rodney Jehu-Appiah

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在大型语言模型上测试不同词汇约束（如E‑Prime、No‑Have、填充词禁止等）与元认知提示的影响，比较它们对七类推理任务的准确率，并引入活跃对照以排除方法学偏差。

**💡 创新点**

发现与认知重构假设相反，词汇约束的效果与约束的理论深度呈倒序关系，最浅的填充词禁令提升最明显；提出“约束作为输出正则化器”解释这一现象。

**🔧 技术方法**

使用系统提示添加词汇禁令或元认知指令，并在每个模型/任务上生成多次回答（温度0.0和0.7），随后通过正则表达式检查违规、合规性过滤、准确率二元计数、Fisher精确检验、Bootstrap置信区间、GEE回归等统计方法。

**📊 数据集**

包含130个推理项目（七类任务各15–20题），共15,600条实验轨迹，最终约11,900条满足100%合规的评分数据；使用六个模型（三大厂商、两档能力）。

**📈 对比分析**

对照组与五种处理组进行成对比较；所有处理组在控制组以上，最佳效果为中性填充词禁令+6.7个百分点；效应在各模型中呈现异质性，跨模型相关性不显著。

**⚠️ 局限性**

限制包括：E‑Prime重试机制导致非对称性；填充词禁令检查器初始精度低；模型顺序效应、样本量不均、任务聚合假设不充分、定性分析仅针对单一模型和任务。

---

## 226. Improving Role Consistency in Multi-Agent Collaboration via Quantitative Role Clarity

**arXiv ID:** 2604.02770 | [PDF](https://arxiv.org/pdf/2604.02770v1)

**作者:** Guoling Zhou `[一作]` (Northeast Normal University), Zhiguo Fu `[通讯]` (Northeast Normal University)

**通讯引用:** 288 | [OpenAlex ID](https://openalex.org/A5045149952)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过量化角色清晰度并将其作为正则化项，改进了LLM驱动的多代理系统的角色一致性。

**💡 创新点**

首次提出可度量、可微分的角色清晰度定义，并将其与 Frobenius 范数结合实现全局一致性评估。

**🔧 技术方法**

采用行向softmax、低秩适配器 LoRA 以及交叉熵正则化进行轻量级微调。

**📊 数据集**

使用 ChatDev 的 SWE-Dev 以及 SRDD 软件需求描述数据集进行实验。

**📈 对比分析**

对比基线模型，加入角色清晰度正则化后角色越界率从约46%降至8%/0.2%，角色清晰度评分提升至0.9，端到端任务成功率也提升。

**⚠️ 局限性**

局限在于仅针对两角色（CEO/CPO）验证，且对角色定义的自然语言依赖仍存在，未覆盖更大规模或多工具场景。

---

## 227. SentinelAgent: Intent-Verified Delegation Chains for Securing Federal Multi-Agent AI Systems

**arXiv ID:** 2604.02767 | [PDF](https://arxiv.org/pdf/2604.02767v1)

**作者:** KrishnaSaiReddy Patil `[一作]` `[通讯]`, KrishnaSaiReddy Patil

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `c84dae5d-5273-4348-85a7-b44cb586b4df` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计并实现了 SentinelAgent，一套面向联邦多智能体 AI 系统的委托链可信安全框架。

**💡 创新点**

创新点在于提出 Delegation Chain Calculus（七个形式化安全属性）以及 Intent‑Preserving Delegation Protocol（三点验证生命周期），并通过 TLA+ 机械验证和实测证明系统完整性。

**🔧 技术方法**

技术手段包括非 LLM 的 Delegation Authority Service（DAS）、HMAC‑签名、工具清单与输出模式白名单、三层意图验证（关键词过滤、上下文增强 NLI、良性覆盖）、TLA+ 模型检查、Python/REST 轻量化实现。

**📊 数据集**

数据集主要为官方 Benchmark DelegationBench v4（516 场景）以及 190 条政府委托文本用于 NLI 微调。

**📈 对比分析**

在与现有安全框架（SEAgent、ShieldAgent、FASA 等）的对比中，单层 P2、P6、P7 的 TPR 分别为 88.3%、86.7%、13.3%，组合实现 100% TPR、0% FPR；在黑盒红队、独立红队、DAS 鲁棒性实验中均保持 100% 检测率。

**⚠️ 局限性**

局限性：意图验证（P2）为概率性质，难以在复杂伪造自然语言上保持 100%；P6 只能拦截授权范围内的调用，无法阻止合法 API 的滥用；P7 仅校验输出标签，无法检测标签内的语义偏差。

---

## 228. Cross Event Detection and Topic Evolution Mining in cross events for Man Made Disasters in Social Media Streams

**arXiv ID:** 2604.02740 | [PDF](https://arxiv.org/pdf/2604.02740v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 229. Random Is Hard to Beat: Active Selection in online DPO with Modern LLMs

**arXiv ID:** 2604.02766 | [PDF](https://arxiv.org/pdf/2604.02766v1)

**作者:** Giyeong Oh `[一作]` (Seoul National University), Junhyug Noh `[通讯]` (Ewha Womans University)

**通讯引用:** 888 | [OpenAlex ID](https://openalex.org/A5088003950)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文对在线直接偏好优化（DPO）中主动偏好学习（APL）与随机采样的效果进行了系统实验，探讨它们在现代大型语言模型的无害性、有用性与指令跟随任务中的表现。

**💡 创新点**

创新之处在于揭示了在强预训练先验下，APL几乎不劣于随机采样，提出了“能力崩塌”与“代理收益”不一致的评估失配问题，并指出了评估代理对实验结果的显著影响。

**🔧 技术方法**

实验采用在线DPO与LoRA参数高效微调，使用基于熵的主动采样策略（先选高熵提示再挑选奖励边际最大的答案对），并与随机采样做对比，同时用代理win‑rate和LM Evaluation Harness测度能力。

**📊 数据集**

使用的训练与评估数据包括Anthropic HH‑RLHF 10k对、UltraFeedback 10k例，评估代理为DeBERTa、Skywork、Beaver、GPT‑5等多种奖励模型和LLM‑judge。

**📈 对比分析**

与随机采样相比，APL在大多数模型与评估代理下未能显著提升win‑rate，且往往需要20倍以上的时间开销；随机采样在win‑rate与能力保持上往往表现相当或更好，只有在极度脆弱的设置中APL能略微降低能力崩塌风险。

**⚠️ 局限性**

局限性包括仅测试了≤7B参数模型、单一APL变体、受限于HH‑RLHF与UltraFeedback两类数据集，以及评估代理多样性不足，未来需检验更大规模模型、不同主动采样策略和更广泛任务。

---

## 230. AI Disclosure with DAISY

**arXiv ID:** 2604.02760 | [PDF](https://arxiv.org/pdf/2604.02760v1)

**作者:** Yoana Ahmetoglu `[一作]` (University College London), Anna Cox `[通讯]`

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文设计并评估了 DAISY，一款面向作者的结构化表单工具，用于生成和编辑学术论文中对 AI 使用的披露声明。

**💡 创新点**

创新点在于（1）将结构化披露表单与用户研究相结合，实证证明其能显著提升披露完整性且不降低作者舒适度；（2）提出 AI 披露工具生态的概念，区分自我报告、自动捕获与合规性、透明性之间的设计维度。

**🔧 技术方法**

技术手段包括：基于文献需求与共创的表单设计；Web 前端实现（HTML/CSS/JavaScript）与 GitHub Pages 托管；使用 Qualtrics 进行问卷收集；采用 Friedman 检验、Wilcoxon 符号秩检验等统计方法评估结果。

**📊 数据集**

数据集主要为参与者提供的自我报告：31 名学术作者的披露文本、AI 工具使用信息以及完成度、舒适度等量表数据；未使用公开学术数据集。

**📈 对比分析**

对比方法：在三种条件下生成披露（无支持、DAISY 自动生成、DAISY 编辑），比较字符长度、完整性评分以及舒适度。结果显示，DAISY 条件下披露长度显著增加（p<0.001），完整性评分从 1.90 提升至 4.4+（p<0.001），而舒适度虽略升高但差异不显著；被访者总体偏好编辑版最高（48.4%）。

**⚠️ 局限性**

局限性包括：样本主要来自 HCI 领域，缺乏跨学科验证；实验仅在在线问卷中完成，未在真实稿件提交流程中评估；依赖自我报告的 AI 使用信息，可能存在回忆偏差；完整性评分依据作者自行制定，缺乏外部编辑或期刊评估的验证。

---

## 231. Geometrically-Constrained Radar-Inertial Odometry via Continuous Point-Pose Uncertainty Modeling

**arXiv ID:** 2604.02745 | [PDF](https://arxiv.org/pdf/2604.02745v1)

**作者:** Wooseong Yang `[一作]` (Seoul National University), Ayoung Kim `[通讯]` (Seoul National University)

**通讯引用:** 5765 | [OpenAlex ID](https://openalex.org/A5100740100)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了一种连续时间雷达-惯性里程计与地图构建框架，通过联合传播点与姿态不确定性，实现自适应权重和高保真地图。

**💡 创新点**

创新点在于：① 采用连续B样条轨迹模型，将姿态不确定性随时间连续化；② 通过点-姿态联合不确定性传播与本地化约束的IEKF，动态下权不可靠观测；③ 在扫描-子图配准中融入显式局部几何约束（平面与分布残差）并用RCS加权；④ 采用在线重力估计与多源不确定性统一管理。

**🔧 技术方法**

技术包括：连续时间IEKF、三次B样条轨迹、点对平面/分布残差、RCS权重、Doppler与重力残差、局部可定位性约束、ikd-tree不确定性筛选、LiDAR对比评估。

**📊 数据集**

使用了公开的4D Continental雷达数据集：Hercules（城市/校园级别）、HKUST（室内场景）以及其他大规模城市场景，涵盖结构化、非结构化与退化环境。

**📈 对比分析**

与Go‑RIO、PUA‑RIO、EKFRIO‑TC以及Fast‑LIO2等基线进行比较，采用ATE、RPE等指标；实验表明在所有数据集上均实现了更低的误差（达到LiDAR级别），并保持20+Hz的实时性能。

**⚠️ 局限性**

局限性包括：① 对极端动态或高速运动时的重力估计仍易受影响；② 在大规模序列中，单独使用测量不确定性与姿态不确定性差距不明显，需进一步优化；③ 目前仅支持单雷达配置，尚未扩展到多雷达或跨模态融合。

---

## 232. Evaluating Bounded Superintelligent Authority in Multi-Level Governance: A Framework for Governance Under Radical Capability Asymmetry

**arXiv ID:** 2604.02720 | [PDF](https://arxiv.org/pdf/2604.02720v1)

**作者:** Tony Rost `[一作]` `[通讯]` (Superintelligence Governance Institute), Tony Rost (Superintelligence Governance Institute)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

构建了一套六维治理评估框架，对人类与人工智能在治理中的认知可比性假设进行检验，并将其应用于现有制度与前瞻性超级智能治理案例，揭示了结构性失败。

**💡 创新点**

创新点在于：① 将治理理论中隐含的认知可比性假设显式化并作为核心假设；② 通过整合政治合法性、委托代理、共和政治与AI对齐理论，提出可检验的六维框架；③ 将失败模式分类为“可依赖技术”“可通过制度设计解决”“需要新理论”，构建了失败分类法。

**🔧 技术方法**

技术上主要是理论合成与概念分析，利用政治学、法理学与AI安全文献中的理论工具构建评估维度；未采用算法实现或实验代码。

**📊 数据集**

无实际数据集；研究基于文献综述、案例分析（如COMPAS、中央银行、欧盟委员会、罗马独裁者等）以及假想的超级智能治理设定。

**📈 对比分析**

方法是通过案例比较检验框架在现实制度中的适用性，并对前瞻性案例进行理论评估；结果显示在现有制度中，多数维度满足要求，但在超级智能治理案例中至少四维失效；未给出数值性能指标，而是以理论合规性为评价标准。

**⚠️ 局限性**

局限性包括：① 仅评估前瞻性设定，未涉及具体实现细节；② 未涵盖分配正义、地缘政治、转型动力等可能重要维度；③ 依赖于现有理论假设，若未来理论更新可能影响结论；④ 无实证数据验证，结论基于理论推演。

---

## 233. Maximally Random Sortition

**arXiv ID:** 2604.02712 | [PDF](https://arxiv.org/pdf/2604.02712v1)

**作者:** Gabriel de Azevedo `[一作]` (Cornell University), Paul Gölz `[通讯]` (Cornell University)

**通讯引用:** 238 | [OpenAlex ID](https://openalex.org/A5044570084)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文提出并实现了基于最大熵分布的公民议会成员选取算法（MaxEntropy 及其公平版 FairMaxEntropy），以在满足给定配额的前提下最大化选取过程的随机性。

**💡 创新点**

创新点在于：① 将最大熵理论引入排序问题，得到唯一且透明的选取分布；② 通过动态规划与渐进式特征约束，克服了 0–1 ILP 的指数复杂度；③ 采用凸优化与梯度下降在对偶空间中求解权重，实现了对预设个体选取概率的精准逼近。

**🔧 技术方法**

核心技术包括：动态规划计数（支持特征分组、状态剪枝与并行采样）、凸优化双向求解（对偶中 log‑sum‑exp）、随机梯度下降（估计梯度并更新权重）、整数精度重采样、以及与现有列生成算法的对比实验。

**📊 数据集**

实验使用了 86 个真实世界排序实例（来自 Sortition Foundation、MASS LBP、NewDemocracy 等组织），涵盖 7–10 个特征、从 70 到 1727 名候选人、以及 20–110 名议员的场景。

**📈 对比分析**

通过与 LexiMin、Goldilocks、Legacy（列生成）以及 Legacy 的对比，MaxEntropy 在 78/86 实例成功采样，速度虽慢（部分实例需 20‑30 分钟），但在交叉多样性、未受约束特征的泛化概率以及公平度量上均优于列生成方法，且 FairMaxEntropy 在几次梯度下降后已显著提升最小选取概率。

**⚠️ 局限性**

主要局限：对极多特征或大议员数的实例仍可能因状态爆炸而超时；动态规划所需内存高达数十 GB；对偶求解收敛速度慢，尤其当目标概率接近边界时；最后，最大熵方案虽然随机性高，但若需求极端公平（如所有候选人选取概率均等）则可能出现大概率为 0 的成员。

---

## 234. Parser-Oriented Structural Refinement for a Stable Layout Interface in Document Parsing

**arXiv ID:** 2604.02692 | [PDF](https://arxiv.org/pdf/2604.02692v1)

**作者:** Fuyuan Liu `[一作]` (Unisound AI Technology Co., Ltd.), Junnan Zhu `[通讯]` (MAIS, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在文档解析管线中加入轻量级的结构化细化模块，改进检测器输出到解析器的接口，实现实例保留、定位和阅读顺序的统一决策。

**💡 创新点**

创新点在于将检测器候选池视为一个整体进行集合级推理，利用学习的保留头与排序头同时优化实例保留与阅读顺序，并引入难度感知的排序目标与保留监督。

**🔧 技术方法**

使用基于DETR的检测器（D-FINE），配合多层Refinement Decoder、可学习的Token融合、图像条件交叉注意力以及BCE、GIoU等损失函数。

**📊 数据集**

在OmniDocBench、D4LA、DocLayNet以及Real5-OmniDocBench四个公开基准上进行评估。

**📈 对比分析**

与传统NMS、LayoutReader、PP-DocLayoutV3等方法对比，F1提升至96.23/93.93/94.52，阅读顺序编辑距离降低至0.024（OmniDocBench）或0.036（Real5-OmniDocBench），显著降低序列不匹配错误。

**⚠️ 局限性**

局限在于目前仅针对单一线性阅读顺序，无法充分处理具有多种合法阅读序列的复杂结构；对部分极端重叠或视觉相似的区块仍可能出现误检。

---

## 235. Efficient3D: A Unified Framework for Adaptive and Debiased Token Reduction in 3D MLLMs

**arXiv ID:** 2604.02689 | [PDF](https://arxiv.org/pdf/2604.02689v1)

**作者:** Yuhui Lin `[一作]` (Xi'an Jiaotong-Liverpool University), Jimin Xiao `[通讯]` (University of Liverpool)

**通讯引用:** 3158 | [OpenAlex ID](https://openalex.org/A5011918180)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出 Efficient3D 框架，对 3D 多模态大型语言模型进行视觉 token 剪枝，以加速推理并降低资源消耗。

**💡 创新点**

创新点在于：①设计了去除浅层偏差的 Debiased Visual Token Importance Estimator (DVTIE) 进行更可靠的 token 重要性估计；②提出 Adaptive Token Rebalancing (ATR) 策略，根据场景复杂度动态调整剪枝力度，实现语义完整性与效率的平衡。

**🔧 技术方法**

采用了 DVTIE 网络（包含 DHCT 层和低秩注意力）、基于注意力的剪枝机制、ATR 动态剪枝策略，以及预训练的 Chat‑Scene 3D MLLM 作为基准模型。

**📊 数据集**

使用了五个 3D 视觉语言基准数据集：ScanRefer、Multi3DRefer、Scan2Cap、ScanQA 和 SQA3D。

**📈 对比分析**

与静态剪枝方法（如 Fast3D）和动态剪枝方法（如 SAP）进行对比，实验表明在 ScanQA 等任务上即使剪枝高达 90% 仍保持甚至提升基线性能；在 Scan2Cap 上 CIDEr 提升 2.57%；FLOPs 下降至原来的 0.38 倍，显示显著的效率提升。

**⚠️ 局限性**

局限性包括：仍依赖大规模预训练模型；在极端剪枝比例下可能丢失关键信息；对不同场景复杂度的适配仍需进一步验证；实验主要集中在固定四个 benchmark，未覆盖所有 3D MLLM 变体。

---

## 236. Frame Theoretical Derivation of Three Factor Learning Rule for Oja's Subspace Rule

**arXiv ID:** 2604.02849 | [PDF](https://arxiv.org/pdf/2604.02849v1)

**作者:** Taiki Yamada `[一作]` `[通讯]`, Taiki Yamada

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `57a58b01-81b4-4d75-a45c-2e891f272b50`

**🎯 论文内容**

利用框架理论将 Oja 子空间规则的对称部分展开，系统推导得到 EGHR‑PCA 三因子学习规则，并证明两者在高斯输入下等价。

**💡 创新点**

创新点在于用框架理论提供了非经验性的推导：框架系数正好对应 EGHR‑PCA 的全局因子，从而消除了以往手工构造的猜想。

**🔧 技术方法**

使用了框架理论、Gaussian 积分分部、Isserlis 定理、矩阵向量化与对称化操作，以及框架算子与其逆的概念。

**📊 数据集**

无实验数据集，全文为理论推导与数学证明。

**📈 对比分析**

未进行实验比较；文章仅通过数学推导展示两规则在高斯分布下的等价性，未给出数值性能指标。

**⚠️ 局限性**

局限性包括：仅适用于零均值高斯输入；假设 Hebbian 术语为线性；若输入非高斯或使用非线性激活，框架算子需要修正，推导不再直接适用。

---

## 237. HiDiGen: Hierarchical Diffusion for B-Rep Generation with Explicit Topological Constraints

**arXiv ID:** 2604.02847 | [PDF](https://arxiv.org/pdf/2604.02847v1)

**作者:** Shurui Liu `[一作]` (Sun Yat-sen University), Ancong Wu `[通讯]` (Sun Yat-sen University)

**通讯引用:** 2994 | [OpenAlex ID](https://openalex.org/A5049519486)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4de8e9d8-757b-475f-9627-18a445e50202` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了 HiDiGen，一个双阶段分层扩散框架，用于自动生成符合 B-rep 结构的 CAD 模型。

**💡 创新点**

通过先生成面-边拓扑再逐级细化几何，显式建模拓扑与几何的互相监督，实现了更高的拓扑有效性与几何多样性。

**🔧 技术方法**

结合 Transformer 编码器/解码器、变分自编码器、Diffusion 模型和 Graph Convolution 等技术进行分层生成。

**📊 数据集**

在 DeepCAD 与 ABC 两大工业 CAD 数据集上进行训练与评估。

**📈 对比分析**

与 DeepCAD、BrepGen、DTGbrepGen 等基线对比，HiDiGen 在 Valid、Novel、CC、MC 等指标上取得领先或相近的最佳表现。

**⚠️ 局限性**

生成结果的有效率仍有提升空间，错误在多阶段传播导致拓扑或几何失真；点云条件生成仅保留粗形，缺乏细节。

---

## 238. Towards Secure Agent Skills: Architecture, Threat Taxonomy, and Security Analysis

**arXiv ID:** 2604.02837 | [PDF](https://arxiv.org/pdf/2604.02837v1)

**作者:** Zhiyuan Li `[一作]` (Chinese Academy of Sciences), Tianyue Luo `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 643 | [OpenAlex ID](https://openalex.org/A5021069086)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文系统地分析了Agent Skills框架的安全风险，提出了四阶段生命周期、七类威胁的威胁分类表，并通过五起真实事件验证分类完整性。

**💡 创新点**

创新点在于首次为Agent Skills提供结构化威胁模型，揭示其设计缺陷（无数据‑指令边界、单次授权、无审计机制），并针对每类威胁给出防御方向与未来研究挑战。

**🔧 技术方法**

采用了架构剖析、威胁分类构造、案例映射与对比分析等技术，结合安全研究文献、漏洞报告和大规模实测（42,447个Skill）构建论证。

**📊 数据集**

主要使用公开安全事件与社区发布的Skill样本库（如skills.sh、ClawHub等），以及先前对42,447 Skill的漏洞扫描结果作为数据来源。

**📈 对比分析**

作者未给出传统意义上的性能指标，而是通过与已知漏洞（如MedusaLocker、ClawHavoc等）对比，展示分类覆盖率高、缺口定位精准；在缺点层面指出缺乏可量化的防御效果评估。

**⚠️ 局限性**

局限性包括：缺少可验证的安全评估工具、无法对自然语言指令做静态检测、需要对Agent Skills规范进行大幅度改动才能彻底消除核心威胁，且本文的威胁模型仍基于当前版本，未来迭代可能出现新的攻击方式。

---

## 239. MMPhysVideo: Scaling Physical Plausibility in Video Generation via Joint Multimodal Modeling

**arXiv ID:** 2604.02817 | [PDF](https://arxiv.org/pdf/2604.02817v1)

**作者:** Shubo Lin `[一作]` (CASIA), Jin Gao `[通讯]` (CASIA)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `14d48e9d-0069-4ad9-996a-1d5968216998` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `8d10c613-917e-4880-9716-17789f50e119` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MMPhysVideo框架，结合双流教师与单流学生的蒸馏方法，对视频扩散模型进行联合RGB‑感知多模态训练，并构建MMPhysPipe数据管线以生成带语义、几何与时空轨迹的伪RGB感知视频；

**💡 创新点**

创新点包括①将语义、几何、时空轨迹三种感知统一为伪RGB格式，②设计双流并行分离的教师网络并引入双向控制链实现像素级跨模态对齐，③通过表示对齐蒸馏将双流知识压缩为单流模型，④在数据层面开发MMPhysPipe实现大规模物理丰富多模态标注；

**🔧 技术方法**

核心技术包括视频扩散模型（Latent Diffusion + Diffusion Transformer）、双流并行架构、零初始化双向控制链、表示对齐蒸馏、VLM（Qwen3‑VL）+ SAM3+3D跟踪进行多粒度标注、Chain‑of‑Visual‑Evidence推理规则；

**📊 数据集**

使用自建MMPhysData‑36k（从OpenVidHD‑0.4M过滤后得到的36k条物理丰富视频）进行训练，并在VideoPhy与PhyGenBench两大物理评测基准上进行评估；

**📈 对比分析**

通过VideoScore2评估物理一致性（PC）和语义对齐（SA）两项指标，并与CogVideoX、Wan2.1、PhyT2V、VideoREPA、OmniVDiff等基线进行对比。MMPhysVideo在所有后端均显著提升平均PC/SA分数，突破现有SOTA；

**⚠️ 局限性**

局限性在于：①数据规模和标注成本仍高，难以覆盖极端光照、遮挡或复杂非线性物理；②双向控制链和蒸馏过程需要额外训练开销；③单流学生对跨模态细节的捕捉仍受限于蒸馏效果，可能在某些细粒度物理细节上不足。

---

## 240. EnsemHalDet: Robust VLM Hallucination Detection via Ensemble of Internal State Detectors

**arXiv ID:** 2604.02784 | [PDF](https://arxiv.org/pdf/2604.02784v1)

**作者:** Ryuhei Miyazato `[一作]` (University of Electro-Communications), Kei Harada `[通讯]` (University of Electro-Communications)

**通讯引用:** 790 | [OpenAlex ID](https://openalex.org/A5090732021)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

研发EnsemHalDet，利用多层多头注意力输出和隐藏状态的内部表征进行集成式幻觉检测

**💡 创新点**

通过检测器级堆叠将多种内部表征融合，显著提升检测AUC，突破单表征限制

**🔧 技术方法**

使用逻辑回归检测器、PCA降维、贪心前向选择及堆叠式集成

**📊 数据集**

CRAG-MM 与 MMMU-Pro 两个VQA基准

**📈 对比分析**

与SAPLMA、MIND、MHAD等基线对比，EnsemHalDet在多模型多数据集上平均提升AUC约1–3%，在多数情况位列首位

**⚠️ 局限性**

需访问模型内部表征，闭源或API限制使用；集成化导致特征提取开销增大；未深入分析不同幻觉类型的检测效果

---

## 241. Vision-Based End-to-End Learning for UAV Traversal of Irregular Gaps via Differentiable Simulation

**arXiv ID:** 2604.02779 | [PDF](https://arxiv.org/pdf/2604.02779v1)

**作者:** Linzuo Zhang `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2320 | [OpenAlex ID](https://openalex.org/A5019803400)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

设计并实现了一套完全基于视觉的端到端无人机穿越狭窄、形状不规则间隙的控制框架，直接将深度图像映射到低层控制指令，支持连续多间隙飞行；

**💡 创新点**

①使用可微分仿真结合差分梯度传播实现感知与控制联合训练；②引入停止梯度（Stop‑Gradient）操作以稳定学习；③提出双模初始化分布（Bimodal Initialization Distribution）以提升跨间隙的稳定性；④加入间隙穿越成功分类器与通行可行性预测器增强实时安全性；

**🔧 技术方法**

可微分集成的CTBR动力学模型、CUDA加速深度渲染器、梯度时间反向传播（BPTT）、卷积+GRU视觉-序列网络、MLP辅助模块、SG操作、双模初始化；

**📊 数据集**

主要使用在自研可微分仿真环境中生成的随机三维间隙深度图；在AirSim中进行验证，并在实际搭载Intel RealSense D435i相机的硬件平台上进行真实飞行实验；

**📈 对比分析**

与基于PPO+边缘检测的状态基准和另一先进视觉导航方法进行对比；在真实深度图上取得98%成功率，对比基准80%；在SGM噪声深度图下仍保持高成功率；在多间隙场景中，双模初始化+重置策略使误差保持在≤1 m/30°，显著优于无此机制的方案；

**⚠️ 局限性**

目前仅在预先训练的规则间隙上表现最优，对极端不规则或被遮挡的间隙仍存在误判；依赖深度相机，对光照变化和遮挡的鲁棒性待进一步提升；未来需要与障碍物规避模块协同工作，以实现更完整的环境适应性。

---

## 242. ESL-Bench: An Event-Driven Synthetic Longitudinal Benchmark for Health Agents

**arXiv ID:** 2604.02834 | [PDF](https://arxiv.org/pdf/2604.02834v1)

**作者:** Chao Li `[一作]` (Shanda Group), Xun Jiang `[通讯]` (Shanda Group)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

提出了基于事件驱动的合成纵向健康基准，生成100名用户的1‑5年设备、检查与事件日志并提供可计算的真值。

**💡 创新点**

创新点在于用可解释的事件时序核、超位置合成与层级化评估维度，实现可验证因果归因与难度分层。

**🔧 技术方法**

采用LLM生成结构化个人资料、事件与检查内容，算法模拟指标动态并结合数据库、内存RAG等检索技术进行评测。

**📊 数据集**

使用全自研的合成数据集，包含日常设备读数、稀疏检查、事件记录，保证了可复制与可审计的基准。

**📈 对比分析**

通过10k条分维度/难度查询比较了13种方法，DB Agent在48‑58%准确率上优于内存RAG(30‑38%)，并在比较与解释任务上突出。

**⚠️ 局限性**

局限性包括：事件模型简化未捕捉真实EHR噪声，合成数据不等价真实临床因果结构，LLM生成可能偏差，外部可泛化性待验证。

---

## 243. STRNet: Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation

**arXiv ID:** 2604.02829 | [PDF](https://arxiv.org/pdf/2604.02829v1)

**作者:** Hao Ren `[一作]` (Sun Yat-sen University), Hui Cheng `[通讯]` (Sun Yat-sen University)

**通讯引用:** 5411 | [OpenAlex ID](https://openalex.org/A5101409148)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了STRNet，一种统一的时空表示框架，用于提升基于视觉的目标导向导航。

**💡 创新点**

通过图网络实现空间特征聚合与混合时移+多尺度对比的时序融合，形成更丰富的时空编码。

**🔧 技术方法**

采用CNN共享编码器、动态轴向图网络、混合时移模块、跨尺度对比卷积以及扩散策略的控制头和距离回归头。

**📊 数据集**

在2D-3D-S、Citysim、GRScenes等模拟环境以及RECON、SCAND、GoStanford、SACSoN等混合数据集上进行训练与评测，并在真实的Diablo机器人上验证。

**📈 对比分析**

与ViNT、NoMaD、NaviBridger等SOTA方法对比，STRNet在成功率、碰撞率、路径长度和SPL等指标上均取得显著提升，最高可达70%更高的成功率。

**⚠️ 局限性**

仅使用RGB图像，缺乏深度/激光信息，且对极端动态或光照变化的鲁棒性尚未充分验证。

---

## 244. Goal-Conditioned Neural ODEs with Guaranteed Safety and Stability for Learning-Based All-Pairs Motion Planning

**arXiv ID:** 2604.02821 | [PDF](https://arxiv.org/pdf/2604.02821v1)

**作者:** Dechuan Liu `[一作]` (University of Sydney), Ian R. Manchester `[通讯]` (University of Sydney)

**通讯引用:** 47856 | [OpenAlex ID](https://openalex.org/A5028491443)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出一种基于学习的全对运动规划方法，利用bi‑Lipschitz diffeomorphism将复杂安全集合映射到单位球，并通过自然梯度流生成安全、稳定、平滑的轨迹。

**💡 创新点**

创新点在于同时保证所有起止点组合下的全局指数稳定与安全，并提供显式速度与收敛速率界定；并将bi‑Lipschitz网络用于学习可逆映射，使模型可泛化到未见的目标。

**🔧 技术方法**

采用bi‑Lipschitz neural network（BiLipNet）实现可逆映射，构造基于自然梯度的神经ODE；训练时使用安全/不安全样本以及RRT生成的成本到达距离标签。

**📊 数据集**

数据集来自2D走廊环境中的RRT采样（约2500个安全样本）以及随机采样的2500个不安全点；无公开标准数据集。

**📈 对比分析**

与传统采样方法或固定目标的导航函数对比，实验表明生成轨迹既安全又收敛到任意目标，速度与收敛率可通过理论界定；在示例中，模型在新目标上也能正确收敛。

**⚠️ 局限性**

局限性包括对凸可映射安全集的假设，处理含孔洞的空间需要改进；以及对极端动态或高速目标的适应性尚未验证。

---

## 245. Learning Structured Robot Policies from Vision-Language Models via Synthetic Neuro-Symbolic Supervision

**arXiv ID:** 2604.02812 | [PDF](https://arxiv.org/pdf/2604.02812v1)

**作者:** Alessandro Adami `[一作]` (University of Padova), Pietro Falco `[通讯]` (University of Padova)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `67630363-6be0-4f51-ab05-7198250671a5` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

利用视觉‑语言模型自动生成可执行的行为树（Behavior Tree），实现从视觉指令到机器人执行的结构化决策；

**💡 创新点**

创新点在于：① 完全基于合成数据的监督策略，消除对人工标注和真实世界演示的依赖；② 通过结构化JSON提示和对行为树语法的约束，使VLM学习到可验证的符号规划；③ 在Pixtral‑12B上采用LoRA微调实现高效、可解释的行为树生成；

**🔧 技术方法**

技术手段包括：Pixtral‑12B 视觉‑语言模型 + LoRA 低秩适配；基于MuJoCo的域随机化合成场景；使用 Gemini 生成指令‑BT 对；JSON‑BT 语法约束和静态验证；

**📊 数据集**

数据集为 10,000 张域随机化的桌面场景图像（MuJoCo 生成）与对应的自然语言指令与行为树对，全部为自动生成的合成数据；

**📈 对比分析**

在两台实际机器人（Franka Panda 与 UR5e）上进行物理验证，生成的 BT 具有 100% 的语法有效率、100% 的任务成功率；与 GPT‑5、Gemini 在零/一 shot 设定下的性能相比，显著提升；

**⚠️ 局限性**

局限性包括：对未见任务（如多阶段“交换”操作）的零 shot 生成能力有限；对硬件特定交互逻辑的泛化不足（如真空吸附）；依赖合成数据，若真实世界场景与训练分布差异较大可能导致性能下降。

---

## 246. CMCC-ReID: Cross-Modality Clothing-Change Person Re-Identification

**arXiv ID:** 2604.02808 | [PDF](https://arxiv.org/pdf/2604.02808v1)

**作者:** Haoxuan Xu `[一作]` (Beihang University), Guanglin Niu `[通讯]` (Beihang University)

**通讯引用:** 660 | [OpenAlex ID](https://openalex.org/A5065094272)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出跨模态服装变换行人再识别任务CMCC-ReID，并构建SYSU-CMCC基准数据集；

**💡 创新点**

设计了渐进式身份对齐网络PIA，先用Dual-Branch Disentanglement Learning（DBDL）去除服装干扰，再用Bi-Directional Prototype Learning（BPL）跨模态对齐；

**🔧 技术方法**

采用双分支特征分离、正交约束、原型学习、温度对比损失以及跨模态/跨服装的对比训练；

**📊 数据集**

使用SYSU-CMCC数据集进行评估，包含可见光与红外两种模态且每人有不同服装；

**📈 对比分析**

在SYSU-CMCC的V2I/I2V两种评测模式下，PIA实现Rank‑1 57.0%/46.5%和mAP 50.4%/50.3%，显著优于现有VI‑ReID和CC‑ReID方法；

**⚠️ 局限性**

主要局限在于对极端光照或极大姿态变化的鲁棒性尚未充分验证，且跨模态对齐仍受限于有限的模态差异补偿技术。

---

## 247. PaveBench: A Versatile Benchmark for Pavement Distress Perception and Interactive Vision-Language Analysis

**arXiv ID:** 2604.02804 | [PDF](https://arxiv.org/pdf/2604.02804v1)

**作者:** Dexiang Li `[一作]` (Harbin Institute of Technology), Yahong Han `[通讯]` (Tianjin University)

**通讯引用:** 4711 | [OpenAlex ID](https://openalex.org/A5031819155)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `e0540dec-d77f-42db-94ae-d039248f6393` `729e5870-4135-47f5-97f2-e3974d07b5dc` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了PaveBench这一统一的公路路面缺陷评估基准，覆盖分类、检测、分割和多轮视觉语言问答；

**💡 创新点**

首次将真实高分辨率正射影像与多模态交互式问答结合，构建PaveVQA，并通过“agent‑augmented”框架将专用视觉工具与通用VLM协同，显著降低数值幻觉；

**🔧 技术方法**

采用高精度多任务标注流程、结构化JSON元数据、LLM生成对话、LoRA参数微调及工具调用（如OverLoCK‑T、DEIM、SCSegamba）实现任务特定的视觉与语言推理；

**📊 数据集**

使用来自辽宁省高速公路的20,124张正射影像，配套的多任务标注和32,160条问答对；

**📈 对比分析**

在视觉任务上，与主流模型对比，分类准确率92.27%、检测mAP 71.84%、分割mIoU 76.0%；在VQA任务中，零shot性能低下，但LoRA微调后分类准确率提升至约88%/93%，数值MAE显著下降，语言指标（ROUGE‑L、BLEU、METEOR）提升30%以上；

**⚠️ 局限性**

限制在于数据集仍以单一地区为主，缺乏跨地区/不同道路材质的多样性，且LLM生成对话在复杂逻辑链条上仍易出现轻微错误；

---

## 248. Rubrics to Tokens: Bridging Response-level Rubrics and Token-level Rewards in Instruction Following Tasks

**arXiv ID:** 2604.02795 | [PDF](https://arxiv.org/pdf/2604.02795v1)

**作者:** Tianze Xu `[一作]` (Shanghai Jiao Tong University), Gang Yu `[通讯]` (Alibaba Group)

**通讯引用:** 19976 | [OpenAlex ID](https://openalex.org/A5003400275)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了Rubrics to Tokens (RTT)框架，将粗粒度的 rubric 评分映射到 token 级奖励，实现精细的信用分配。

**💡 创新点**

创新点包括：① Token‑Level Relevance Discriminator 直接为每个约束定位相关 token；② RTT‑GRPO 将 response 与 token 级优势联合优化；③ 解决三维奖励空间的组划分问题，提出 Intra‑sample Token Group Normalization。

**🔧 技术方法**

使用技术包括：GRPO 强化学习、token‑级判别器、组归一化（Intra‑sample Token Group Normalization）、大型语言模型和 LLM‑as‑Judge。

**📊 数据集**

数据集：HiR‑16K（多源指令与约束集合）用于训练；评测集包括 IFEval、IFBench、MulDimIF、AdvancedIF、MATH‑500、GPQA、MMLU‑Pro。

**📈 对比分析**

通过与 SFT、DPO、RL‑AON、RL‑CSR 等基线对比，RTT 在指令级与 rubric 级准确率上平均提升约 2.5% 与 1.6%，并在 OOD 任务保持或略微提升性能。

**⚠️ 局限性**

局限性：依赖高质量 token‑级标注，额外计算成本约 8.5% GPU‑小时；在极长文本或多步骤任务中，token‑级信用分配的效果仍需进一步验证。

---

## 249. Generative AI Use in Professional Graduate Thesis Writing: Adoption, Perceived Outcomes, and the Role of a Research-Specialized Agent

**arXiv ID:** 2604.02792 | [PDF](https://arxiv.org/pdf/2604.02792v1)

**作者:** Kenji Saito `[一作]` (Waseda University), Hiroshi Kanno `[通讯]` (Waseda University)

**通讯引用:** 6015 | [OpenAlex ID](https://openalex.org/A5071547766)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本研究对日本早稻田大学MBA论文学生进行问卷调查，收集了83名学生的AI使用情况、收益与担忧，并对其中使用的研究专用AI（GAMER PAT）与其他AI工具的偏好进行了比较，同时对8名受访者进行了访谈以获取定性洞察。

**💡 创新点**

首次将研究专用AI与通用AI在MBA论文写作中的实际使用效果进行对比，揭示AI在支持论证结构、提升写作质量的同时也带来真实性与引用治理挑战，并提出学术治理与工具设计的关键路径。

**🔧 技术方法**

采用结构化问卷收集定量数据，并用频率统计、均值与两侧精确二项检验进行分析；访谈文本通过初步主题分析提炼学生的认知警觉与工具选择策略。

**📊 数据集**

主要数据集为83名MBA学生的自报问卷结果（包含AI使用频率、工具清单、阶段应用、效益与担忧等），以及8名学生的访谈转录，未使用外部公开数据集。

**📈 对比分析**

通过让使用GAMER PAT的受访者对比评价其他AI，剔除“相等/不确定”答案后使用两侧精确二项检验，结果显示在深化研究与结构组织两项功能上，GAMER PAT显著优于其他AI（p<0.05）；整体偏好中48.6%认为更好，显著高于仅8.6%倾向其他AI。

**⚠️ 局限性**

研究局限包括仅覆盖单一机构与单一毕业班，数据为自报且缺乏人口学变量，可能存在非响应偏倚；访谈样本小且自选；未进行客观写作质量评估或长期使用日志分析，限制了对工具效能与治理效果的综合验证。

---

## 250. A Paradigm Shift: Fully End-to-End Training for Temporal Sentence Grounding in Videos

**arXiv ID:** 2604.02860 | [PDF](https://arxiv.org/pdf/2604.02860v1)

**作者:** Allen He `[一作]` (Basis International School Park Lane Harbour), Wu Liu `[通讯]` (University Of Science And Technology China)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种完全端到端的时序句子定位框架，联合优化视频特征提取器和定位头。

**💡 创新点**

创新点在于 Sentence Conditioned Adapter（SCADA）——一个轻量化的、由句子嵌入驱动的适配器，能在保持显著内存效率的前提下动态调制视频特征，并引入视频中心训练策略提高效率。

**🔧 技术方法**

使用 DistilBERT 进行句子编码，ViT/C3D/I3D 等视频骨干网络，SCADA 通过内部/外部分支对特征进行语言调制；检测头采用简化的双向 LSTM；训练使用 AdamW、multi-task 损失。

**📊 数据集**

在 Charades‑STA 与 ActivityNet‑Captions 两个公开基准上进行实验。

**📈 对比分析**

相较于现有最佳方法（如 Momentor、HawkEye、DPQF 等），在 Rank1@IoU=0.5/0.7、Rank5@IoU=0.5/0.7 以及 mIoU 上均实现显著提升，尤其在 Charades‑STA 上取得 64.59%/43.65% 的 Rank1@IoU=0.5/0.7，超过前沿方法多达 10% 以上。

**⚠️ 局限性**

局限性：仍依赖大规模预训练模型；SCADA 需要额外的句子嵌入和计算，可能在资源受限环境下受限；在极长视频或多模态任务中的泛化尚未充分验证。

---

## 251. GRADE: Probing Knowledge Gaps in LLMs through Gradient Subspace Dynamics

**arXiv ID:** 2604.02830 | [PDF](https://arxiv.org/pdf/2604.02830v1)

**作者:** Yujing Wang `[一作]` (Beihang University), Hanqi Yan `[通讯]` (King's College London)

**通讯引用:** 35 | [OpenAlex ID](https://openalex.org/A5102622759)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种基于梯度动态的知识缺口检测方法GRAdient Dynamics（GRAD），用于判断大型语言模型在回答查询时内部知识是否充足；

**💡 创新点**

创新点在于将梯度投影到隐藏状态子空间，计算梯度与隐藏状态的稳定秩比（Rank Ratio）来量化所需知识更新比例，并通过跨层的Rank Ratio序列训练监督探测器，避免传统隐藏状态方法受输入无关特征干扰，同时能够生成可解释的 token‑级别知识缺口图；

**🔧 技术方法**

使用梯度投影、稳定秩、谱感知的秩比计算、层级特征聚合、监督学习探测器以及 token‑级别解释生成；

**📊 数据集**

在六大基准上验证：GSM8K、MATH、MMLU、NQ、TQA、HotpotQA；并在 Qwen3‑30b‑instruct 上进行输入扰动测试；

**📈 对比分析**

与多种基线（Verbalization 的 Judge、Token‑Entropy、IC、Align‑P 等）比较，GRAD_pre/pos 在 Acc 和 AUROC 上均超过所有基线，且在输入重述扰动和跨数据集迁移上表现更稳健；

**⚠️ 局限性**

局限性包括：需要梯度计算，增加推理成本；梯度子空间假设对不同模型结构可能不完全成立；对长文本梯度收敛与数值稳定性仍需进一步研究。

---

## 252. Orientation Matters: Learning Radiation Patterns of Multi-Rotor UAVs In-Flight to Enhance Communication Availability Modeling

**arXiv ID:** 2604.02827 | [PDF](https://arxiv.org/pdf/2604.02827v1)

**作者:** Martin Zoula `[一作]` (Czech Technical University in Prague), Martin Saska `[通讯]` (Czech Technical University in Prague)

**通讯引用:** 5232 | [OpenAlex ID](https://openalex.org/A5004992661)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `51c0528b-f690-4182-ae60-bb5f046c276c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

研究了一种基于同步校准轨迹学习双无人机辐射模式的方法，并在实地飞行中验证其可行性。

**💡 创新点**

将双无人机的辐射模式同时解耦为独立模式，提出利用球面谐波、桶化和多项式模型的学习框架，并设计了高效的联合轨迹。

**🔧 技术方法**

采用球面谐波展开、基于核的桶化插值、线性最小二乘与岭回归，结合EKF状态估计和时间同步技术。

**📊 数据集**

使用了在开放式田野中两架四旋翼无人机收集的实测RSSI数据，共约27.9k条样本。

**📈 对比分析**

与均值和最近邻基线对比，使用交叉验证评估RMSE；球面谐波-28阶模型RMSE为3.39 dB，接近测量噪声水平，优于其他模型。

**⚠️ 局限性**

模型可能存在过拟合与采样偏差，未考虑极化、频率差异及定位误差影响，需进一步验证。

---

## 253. UNICA: A Unified Neural Framework for Controllable 3D Avatars

**arXiv ID:** 2604.02799 | [PDF](https://arxiv.org/pdf/2604.02799v1)

**作者:** Jiahe Zhu `[一作]` (Nanjing University), Hao Zhu `[通讯]` (Nanjing University)

**通讯引用:** 10070 | [OpenAlex ID](https://openalex.org/A5068560690)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `edb9d762-f411-4838-a852-f2d638b018db` `4de8e9d8-757b-475f-9627-18a445e50202` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

实现了键盘输入直接驱动全3D人物模型的自动生成、渲染，完全省去传统的动作规划、骨骼绑定、物理仿真等步骤；

**💡 创新点**

创新点在于把整个“键盘→动作规划→绑定→物理→渲染”流程统一为单一神经网络框架，采用动作条件扩散模型生成4D几何，再通过点变换器映射为3D Gaussian Splatting 进行高质量自由视角渲染；

**🔧 技术方法**

主要技术包括动作条件多帧扩散网络、VAE编码解码、交叉注意力与时序注意力、PCA对齐六视角位置图、进阶4D推理（累积位移与归一化）、点变换器（PTv3）与3D Gaussian Splatting；

**📊 数据集**

使用从Unreal Engine导出的五个角色网格序列，包含五种键盘动作（前后左右空闲）及动作间过渡，配合Chaos Cloth 物理仿真得到的衣物动态；

**📈 对比分析**

在动画质量、视觉相似度（PSNR/LPIPS）和视频质量（FVD）上与Animatable Gaussians、Mixamo、SV4D 2.0 等基线进行对比，实验显示其在FVD、PSNR、LPIPS和用户评分（VQS、PRS）上均优于基线；

**⚠️ 局限性**

局限性包括推理时间尚未达到实时，某些转弯姿态下的3D Gaussian Splatting 产生瑕疵，且目前仅支持行走类动作，无法覆盖更丰富的动作空间。

---

## 254. LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers

**arXiv ID:** 2604.02787 | [PDF](https://arxiv.org/pdf/2604.02787v1)

**作者:** Shreshth Saini `[一作]` (University of Texas at Austin), Alan C. Bovik `[通讯]` (University of Texas at Austin)

**通讯引用:** 142729 | [OpenAlex ID](https://openalex.org/A5075463806)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一种无提示、物理和感知驱动的扩散变换器LumaFlux，用于将8位SDR（BT.709）转换为10位HDR（PQ/BT.2020）

**💡 创新点**

创新点：在冻结的Flux多模态扩散变换器上插入三种轻量化适配器（PGA、PCM、HDR Residual Coupler）并加入可学习的Rational-Quadratic Spline解码，物理亮度与感知语义双重引导，参数高效且无文本提示

**🔧 技术方法**

技术手段：低秩LoRA + FiLM感知跨调制、时间-层自适应调节、频谱门控、RQS光度曲线、预训练Flux多模态扩散变换器

**📊 数据集**

数据集：结合HIDROVQA、CHUG、LIVE-TMHDR构建264k+54k SDR–HDR对的训练集，并推出Luma-Eval评估基准（含PGC/UGC HDR源及多种TMO与压缩级别）

**📈 对比分析**

评估方式：与CNN基准（HDRTVNet++、ICTCPNet、HDRTVDM、HDCFM、Deep SR-ITM）和扩散基准（LEDiff、PromptIR）在HDRTV1K、HDRTV4K和Luma-Eval上进行PSNR、SSIM、HDR-VDP3、ΔE_ITP、HDR-LPIPS等多维度对比，LumaFlux在多数指标上取得最高分，PSNR提升≈1.6 dB、ΔE_ITP降低≈0.8

**⚠️ 局限性**

局限性：受限于冻结backbone的表达能力，极端高动态/极低光照时可能出现轻微色度漂移；RQS解码训练依赖大规模对齐HDR数据；实时推理仍需较高算力

---

## 255. CANDLE: Illumination-Invariant Semantic Priors for Color Ambient Lighting Normalization

**arXiv ID:** 2604.02785 | [PDF](https://arxiv.org/pdf/2604.02785v1)

**作者:** Rong-Lin Jian `[一作]` (National Yang Ming Chiao Tung University), Chih-Chung Hsu `[通讯]` (National Yang Ming Chiao Tung University)

**通讯引用:** 2155 | [OpenAlex ID](https://openalex.org/A5007305393)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了CANDLE框架，用自监督视觉模型DINOv3的多层特征做为色彩-环境归一化的语义引导，解决多色光照下的颜色失真问题。

**💡 创新点**

创新点在于：①将DINOv3多层语义特征通过D.O.G.注入编码器，取代传统几何或频域先验；②在解码器端设计BFACG和SFFB两块模块，分别抑制色彩崩塌和跳接特征中的光照泄漏，实现结构与色彩的分离恢复。

**🔧 技术方法**

核心技术包括自监督Transformer特征提取、Prompt Selection Fusion、DINO-Residual Fusion Block、边缘感知的BFACG、Haar小波频域过滤的SFFB，以及多阶段训练与检索式微调。

**📊 数据集**

主要使用CL3AN（多色光照下的高分辨率场景对）和Ambient6K（白光照条件），并在NTIRE 2026 ALN挑战赛的公开测试集上评测。

**📈 对比分析**

相较于IFBlend、PromptNorm、RLN2等现有方法，CANDLE在CL3AN上PSNR提升1.22 dB，SSIM提升0.21 dB，LPIPS下降0.04；在NTIRE 2026彩色光照赛道排名第3，白光赛道获得第2位FID最低，表现出强泛化与高视觉质量。

**⚠️ 局限性**

局限性包括：对极端高光饱和区仍可能出现细节失真；对全局色彩漂移的恢复仍依赖DINO的语义一致性，可能在完全不同物体类别上表现欠佳；模型参数量与计算复杂度相对较高。

---

## 256. A Unified Perspective on Adversarial Membership Manipulation in Vision Models

**arXiv ID:** 2604.02780 | [PDF](https://arxiv.org/pdf/2604.02780v1)

**作者:** Ruize Gao `[一作]` (National University of Singapore), Feng Liu `[通讯]` (University of Melbourne)

**通讯引用:** 15787 | [OpenAlex ID](https://openalex.org/A5100325566)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文研究了视觉模型的成员推断攻击（MIA）在面对对抗性扰动时的脆弱性，提出了“对抗性成员制造”（MFA）攻击以及对应的检测（MFD）与鲁棒推断（AR-MIA）方案；

**💡 创新点**

创新点在于将对抗攻击与成员推断交叉分析，发现梯度范数坍塌（gradient‑norm collapse）是制造成员的统一几何特征，基于此提出梯度几何检测与加权鲁棒统计，构建了完整的对抗性成员推断防御框架；

**🔧 技术方法**

主要技术包括：对抗梯度上升（momentum‑based cosine‑annealed ascent）用于生成制造成员；梯度范数统计作为检测特征；tanh‑加权结合原始MIA统计量实现鲁棒推断；以及标准的梯度下降/PGD、FGSM 等对抗攻击与MIAs（Loss、Attack R、LiRA、RMIA）做基线对比；

**📊 数据集**

实验使用 CIFAR‑10/100、SVHN、CINIC‑10、ImageNet‑100 数据集，模型为 ResNet‑18 与 Wide‑ResNet‑50‑2；

**📈 对比分析**

与现有 MIAs 的对比显示：MFA 在错误面积和 EER 指标上显著优于 Loss、Attack R、LiRA、RMIA；MFD 在 AUC、TPR/FPR 方面达到 0.9 以上；AR‑MIAs 在 ROC 曲线中比传统 MIAs 取得更高的 TPR 与更低的 FPR，整体 AUC 有显著提升；

**⚠️ 局限性**

局限性包括：检测与鲁棒推断多依赖白盒梯度信息，对黑盒攻击的适用性有限；需要对 λ 等超参数进行手动校准；实验仅覆盖图像分类任务，尚未验证在其他领域或更大规模模型上的泛化能力；

---

## 257. QuadAgent: A Responsive Agent System for Vision-Language Guided Quadrotor Agile Flight

**arXiv ID:** 2604.02786 | [PDF](https://arxiv.org/pdf/2604.02786v1)

**作者:** Ao Zhuang `[一作]` (Shanghai Jiao Tong University), Danping Zou `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 2320 | [OpenAlex ID](https://openalex.org/A5019803400)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `51c0528b-f690-4182-ae60-bb5f046c276c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了QuadAgent，一种无训练、基于异步多代理架构的可视语义导引四旋翼飞行系统，能够在飞行中实时响应视觉与语言输入。

**💡 创新点**

创新点包括：①将高层推理与低层控制解耦，采用前景工作流代理与后台预判代理并发执行；②引入轻量级印象图（Impression Graph）作为稀疏关键帧拓扑地图，用于记忆与路径规划；③实现暂停-恢复协议，使飞行期间可并行推理与执行；④结合视觉避障网络与物理差分策略，提升安全性。

**🔧 技术方法**

技术手段主要包含：大型语言模型（LLM）与视觉‑语言模型（VLM）做推理；异步多代理事件驱动机制；印象图构建与拓扑查询；物理基避障策略；低层视觉‑动作网络；预判与任务重规划算法。

**📊 数据集**

实验数据来源于自建仿真环境（涵盖随机障碍与任务）以及真实室内实验平台，未使用公开大规模数据集，重点验证系统在动态指令与实时飞行中的表现。

**📈 对比分析**

与Serial Plan‑ReAct、Scene Graph Agent、Traj NavGraph等基线及两种消融（无 Look‑ahead、无 On‑Demand Retrieval）进行对比，QuadAgent在成功率、整体规划成功率、进度、SPL 等指标均领先，尤其在仿真中成功率 80%、整体规划成功率 86.7%、进度 93.3%、SPL 58.6，飞行速度可达 5 m/s。

**⚠️ 局限性**

局限性包括：依赖高质量 LLM/VLM，当前缺乏在线探索与动态障碍处理；印象图对关键帧提取与匹配精度敏感；系统主要在室内障碍环境验证，尚未在户外或复杂动态场景中充分测试；以及需要进一步扩展技能库与微调低层控制策略。

---

## 258. LLM+Graph@VLDB'2025 Workshop Summary

**arXiv ID:** 2604.02861 | [PDF](https://arxiv.org/pdf/2604.02861v1)

**作者:** Yixiang Fang `[一作]` (Chinese University of Hong Kong), Shu Wang `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 9502 | [OpenAlex ID](https://openalex.org/A5100335448)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656`

**🎯 论文内容**

本文综述了2025年VLDB大会中的LLM+Graph研讨会内容，涵盖主题演讲、工业案例、论文展示及专题讨论；

**💡 创新点**

创新点在于构建LLM与图数据库、图RAG及图基金模型的双向协同机制，提出Riemann几何图模型、图一致性修复框架、双粒度提示与图本地智能体Chat2Graph等前沿方案；

**🔧 技术方法**

主要技术包括LLM4DB与DB4LLM、GraphRAG、RAG、RiemannGeoDGL、MIRAGE、Graph Foundation Model、双粒度提示、Text2Cypher、GDS Agent、规划图/工具图/记忆图等；

**📊 数据集**

使用的数据集涵盖公开知识图谱（如U.S. public laws）、ByteGraph、Neo4j、Ant Group图数据、Text2Cypher‑2025v1、图一致性验证测试集等；

**📈 对比分析**

通过与传统NL2SQL、向量检索、GraphRAG基线等方法对比，实验表明GraphRAG在多跳推理和检索准确率上显著提升；RiemannGeoDGL在图聚类与动态建模中表现更优；LLM4DB增强查询可解释性；但仍存在幻觉、成本高等问题；

**⚠️ 局限性**

局限性包括对人工干预的高度依赖、数据规模与质量限制、LLM生成修复的准确率偏低、可解释性不足、系统成本与能耗高以及图数据库对LLM支持不完善。

---

## 259. Dependency-Guided Repository-Level C-to-Rust Translation with Reinforcement Alignment

**arXiv ID:** 2604.02852 | [PDF](https://arxiv.org/pdf/2604.02852v1)

**作者:** Jia Feng `[一作]` (Harbin Institute of Technology), Kui Liu `[通讯]` (Huawei)

**通讯引用:** 6390 | [OpenAlex ID](https://openalex.org/A5100374012)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套基于依赖导向的仓库级 C‑to‑Rust 迁移框架，包含 RAST 训练（多任务微调+编译器反馈强化学习）和 DGIR 推理（跨语言依赖图对齐+迭代自我修正）。

**💡 创新点**

创新点：①将编译器诊断与功能对齐作为奖励引入强化学习；②构造跨语言依赖图并映射到 Rust 依赖池，提供细粒度上下文；③迭代自我修正机制结合语义一致性检查；④构建 85k 规模的函数级对齐数据和 145 个仓库级基准，填补并发数据缺口。

**🔧 技术方法**

技术手段：LLM (Qwen2.5‑Coder, DeepSeek‑Coder)、多任务微调、GRPO 强化学习、Tree‑Sitter 依赖图、BGE‑M3 语义检索、编译器诊断、语义一致性校验、LoRA 与 ZeRO‑3 优化。

**📊 数据集**

使用的数据集：85k 函数级 C‑Rust 对齐数据（82.5k 训练，2.5k RL），145 个仓库级评测基准（DCBench、IMCBench、HWBench、CRustBench），以及 15 家华为工业项目。

**📈 对比分析**

与基线（Base、RAG、File Context、Sactor）对比，DCBench 上编译成功率提升 19–22%，功能准确率提升 15–20%；工业项目成功构建率从 3 提升至 7；小模型 Qwen2.5‑Coder‑7B 在 RAST 后达 60.7% CSR、43.5% CA，超过同尺寸基线。

**⚠️ 局限性**

局限性：受限于所选 LCM、训练资源与推理成本高、对极大规模项目仍可能出现错误、对复杂业务逻辑的可解释性有限，且数据泄露风险需进一步评估。

---

## 260. Adaptive Local Frequency Filtering for Fourier-Encoded Implicit Neural Representations

**arXiv ID:** 2604.02846 | [PDF](https://arxiv.org/pdf/2604.02846v1)

**作者:** Ligen Shi `[一作]` (Inner Mongolia University), Chang Liu `[通讯]` (Beijing Information Science and Technology University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `4de8e9d8-757b-475f-9627-18a445e50202` `57a58b01-81b4-4d75-a45c-2e891f272b50` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出自适应局部频率滤波方法，利用可学习的空间变参数对 Fourier 编码进行逐通道调制，实现低通/带通/高通的平滑切换，增强 Fourier-encoded 先验在空间变化信号上的表现。

**💡 创新点**

创新点在于：①引入可学习的空间参数 α(x) 并通过差分 Sigmoid 设计的频率滤波器实现显式的局部频率控制；②从神经切线核（NTK）视角解析滤波器对有效核谱的重塑，为频率适配提供理论解释；③将可学习网格与多线性插值结合，实现光滑的空间频率分布。

**🔧 技术方法**

使用技术包括：傅里叶特征映射、位置编码、可学习网格 α(x) 的多线性插值、基于差分 Sigmoid 的频率滤波器、三层 MLP（或八层 3D），NTK 近似分析、TV 正则化辅助稀疏重建、以及标准评估指标（PSNR/SSIM/LPIPS、Chamfer/IoU）。

**📊 数据集**

采用的数据集有：NTIRE 2017 单图像超分辨率数据集（512×512 32 张自然图像）用于 2D 图像拟合与稀疏重建；Stanford 3D 扫描仓库中的四个 3D 形状（Armadillo、Dragon、Lucy、Thai Statue）用于 SDF 表示。

**📈 对比分析**

与 PE-MLP、SIREN、MFN、DINER、BACON 等基线在相同 MLP 结构下对比；在 2D 图像拟合中，Sine 版本 PSNR 从 31.45 提升至 46.27，SSIM 0.8706→0.9938，LPIPS 2.54e-1→2.41e-3；在 3D SDF 任务中，平均 Chamfer 下降至 1.73e-6，IoU 提升至 0.983；训练收敛速度更快，早期 PSNR 明显领先。

**⚠️ 局限性**

局限性包括：①对可学习网格分辨率敏感，粗网格难以捕捉细粒度频率变化；②需手动调节滤波带宽 B、编码级别 L 与网格尺寸等超参数；③引入插值与滤波额外开销；④目前仅针对静态场景，未验证对动态或时空数据的适用性。

---

## 261. Factorized Multi-Resolution HashGrid for Efficient Neural Radiance Fields: Execution on Edge-Devices

**arXiv ID:** 2604.02836 | [PDF](https://arxiv.org/pdf/2604.02836v1)

**作者:** Kim Jun-Seong `[一作]`, Jin-Hwa Kim `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本论文提出了一种名为Fact-Hash的稀疏参数化编码方法，可在边缘设备上高效训练与渲染神经辐射场（NeRF）。

**💡 创新点**

创新点在于将Hash-编码与张量分解（Tensor Factorization）结合，采用二维碰撞约束的Hash表、三平面（tri‑plane）投影与Hadamard乘积归约，从而实现高分辨率特征压缩、参数量大幅下降且对少量样本鲁棒。

**🔧 技术方法**

主要技术包括：多分辨率Hash‑grid、张量分解、三平面投影、Hadamard乘积、拒绝采样（bit‑field 与 proposal network）、Spherical Harmonics 视角编码、以及基于CUDA的高效推理。

**📊 数据集**

使用的实验数据集包括：合成的NeRF Synthetic、真实世界的高分辨率360° Tank & Temples、以及大规模城市区块数据集San Francisco Mission Bay。

**📈 对比分析**

与现有SOTA方法iNGP、TensoRF、K‑planes进行对比，指标为PSNR/SSIM/LPIPS及边缘设备渲染速度；Fact‑Hash在参数量仅为1/2–1/5的情况下，PSNR与基线相当甚至更优，渲染速度提升约40%，并在少样本场景下表现更稳定。

**⚠️ 局限性**

局限性包括：对极大场景仍需分块存储；在极复杂细节场景下偶有浮点噪声；需要进一步量化/剪枝以进一步压缩；并且实验主要集中在Jetson Xavier NX等单一硬件平台，跨平台验证仍待展开。

---

## 262. Bilateral Intent-Enhanced Sequential Recommendation with Embedding Perturbation-Based Contrastive Learning

**arXiv ID:** 2604.02833 | [PDF](https://arxiv.org/pdf/2604.02833v1)

**作者:** Shanfan Zhang `[一作]` (Xi'an Jiaotong University), Yuan Rao `[通讯]` (Xi'an Jiaotong University)

**通讯引用:** 1988 | [OpenAlex ID](https://openalex.org/A5003835661)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出一种端到端的双向意图增强对比学习框架 BIPCL，用于序列推荐，通过显式注入集体意图原型和嵌入级扰动实现多层对比学习。

**💡 创新点**

创新点在于（1）双侧意图增强机制：在物品和用户序列表示中同时注入共享意图原型；（2）方向感知的嵌入扰动生成语义一致且判别力强的对比视图；（3）在交互、意图、序列三个层次上执行 InfoNCE 对齐，形成多层对比学习。

**🔧 技术方法**

使用的技术包括：图卷积/高阶传播构建物品共现图；意图原型聚类与 soft‑assignment；Transformer 编码用户序列；门控融合将原始嵌入与意图信息结合；基于 InfoNCE 的对比学习；以及带符号的嵌入扰动生成多样化视图。

**📊 数据集**

实验使用了 Beauty、Yelp、RetailRocket、Gowalla、Amazon Books 五个公开序列推荐基准数据集。

**📈 对比分析**

与 MIND、ComiRec、Re4、DisMIR、ICSRec 等多种最先进的多意图与对比学习模型进行比较，BIPCL 在 Recall@N、NDCG@N、HR@N 等指标上均实现显著提升，尤其在 NDCG 上平均提升 10%–20%。

**⚠️ 局限性**

局限性：对图传播和 Transformer 的计算开销较大，对意图数、扰动幅度等超参数敏感；在极度稀疏或冷启动场景下仍需进一步验证其鲁棒性。

---

## 263. Unified and Efficient Approach for Multi-Vector Similarity Search

**arXiv ID:** 2604.02815 | [PDF](https://arxiv.org/pdf/2604.02815v1)

**作者:** Binhan Yang `[一作]` (Beihang University), Ke Xu `[通讯]` (Beihang University)

**通讯引用:** 11970 | [OpenAlex ID](https://openalex.org/A5100665814)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `64443552-63e0-44b5-906f-d90fe95c5a1b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了首个原生多向量层次图索引(MV-HNSW)，实现统一多向量相似度搜索。

**💡 创新点**

创新点包括：① 针对多向量的三大必备属性（对称性、基数鲁棒性、查询一致性）设计新的边权函数；② 结合加速的多向量相似度计算算法和聚类滤波；③ 通过辅助导航表实现动态候选扩展，突破拓扑断裂导致的召回下降。

**🔧 技术方法**

技术手段涵盖：层次可导航小世界图(HNSW)改造、多向量边权函数、聚类加速的MaxSim/USim计算、辅助导航表(ANT)、动态候选扩展策略。

**📊 数据集**

使用了七个真实规模数据集（来自LoTTE和MS MARCO），规模从2.6亿到近90亿token向量，维度128-768。

**📈 对比分析**

与四类代表性基线（ColBERT、WARP/ColXTR、IVF+PQ、Qdrant）以及改进的单向量HNSW等进行对比，MV-HNSW在保持90%以上召回的同时，搜索延迟比最优基线提升5.6–14.0倍，整体召回-延迟曲线明显优于所有对手。

**⚠️ 局限性**

局限性：索引构建时间和存储量相对较高（构建时间约为单向量HNSW的3–10倍，索引体积略大），并且在极大规模下对多线程并行构建需求更高。

---

## 264. Deception Equilibrium Analysis for Three-Party Stackelberg Game with Insider

**arXiv ID:** 2604.02807 | [PDF](https://arxiv.org/pdf/2604.02807v1)

**作者:** Xiaoyu Xin `[一作]` (Tongji University), Yiguang Hong `[通讯]` (Tongji University)

**通讯引用:** 21916 | [OpenAlex ID](https://openalex.org/A5100359415)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `9cc9baba-5356-466d-81ff-d80028d90279`

**🎯 论文内容**

本文构建了一种包含欺骗机制的三方Stackelberg安全博弈模型，提出了Deception Stackelberg均衡（DSE）和Hyper Nash均衡（HNE）概念，并给出了它们一致性的必要与充分条件；

**💡 创新点**

创新点在于：①首次将欺骗融入三方层级博弈；②引入WDSE/SDSE与HNE的一致性理论；③针对非光滑、集合值最佳反应映射，设计了可扩展的超梯度求解算法，并证明了其收敛性；

**🔧 技术方法**

使用了超梯度（hyper‑gradient）估计、隐式微分、变分不等式、可定义函数理论及可拓扑优化方法；

**📊 数据集**

实验使用了模拟数据，分别在安全无线通信与微电网内部攻击防御场景下设置参数，未使用公开数据集；

**📈 对比分析**

通过与无欺骗基线和传统双层博弈方法对比，证明了算法在收敛速度、求解精度与对抗鲁棒性方面均优于现有方法；

**⚠️ 局限性**

局限性包括：①假设欺骗参数集有限；②需要满足可微分与强单调等技术前提；③在极大规模多约束情形下的可扩展性待进一步验证。

---

## 265. Distance Comparison Operations Are Not Silver Bullets in Vector Similarity Search: A Benchmark Study on Their Merits and Limits

**arXiv ID:** 2604.02801 | [PDF](https://arxiv.org/pdf/2604.02801v1)

**作者:** Zhuanglin Zheng `[一作]` (Beihang University), Yongxin Tong `[通讯]` (Beihang University)

**通讯引用:** 11748 | [OpenAlex ID](https://openalex.org/A5051874566)

**关键词:** `70392921-652b-47dd-9813-65d50cbe35c7` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了距离比较操作(DCO)在向量相似检索中的性能，并在多种数据集与硬件环境下进行了大规模基准测试。

**💡 创新点**

提出DCO并非“银弹”，其效率高度受维度、查询分布和硬件影响，并给出了实践中的选择指南。

**🔧 技术方法**

对比8种DCO算法，评估其在CPU（启用/禁用SIMD）和GPU下结合HNSW/IVF索引的实现，并使用欧氏、内积、余弦等距离度量。

**📊 数据集**

使用10个公开数据集，维度从96到12,288，样本量从10万到1亿，覆盖低、中、高、超高维场景。

**📈 对比分析**

通过QPS、召回率、维度裁剪率等指标比较，SOTA方法在中等维度时可提升1.4–2.1× QPS，但在低维、超高维或OOD查询时往往比全维扫描更慢。

**⚠️ 局限性**

主要限制包括：对低/超高维数据无效、对OOD查询鲁棒性差、对硬件配置高度敏感、且多方法需满足特定假设或具备足够训练数据。

---

## 266. Differential Mental Disorder Detection with Psychology-Inspired Multimodal Stimuli

**arXiv ID:** 2604.02798 | [PDF](https://arxiv.org/pdf/2604.02798v1)

**作者:** Zhiyuan Zhou `[一作]` (Hefei University of Technology), Richang Hong `[通讯]` (Hefei University of Technology)

**通讯引用:** 22575 | [OpenAlex ID](https://openalex.org/A5051332325)

**关键词:** `a154b176-e466-40fc-8ae0-e5cd17677106` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

构建了基于心理学刺激的多模态精神健康数据集（MMH），并提出了面向不同刺激情境的范式感知多模态学习框架（PMLF）用于多病种差异化诊断

**💡 创新点**

创新点在于①设计了五步心理学启发的多模态刺激范式，①收集了临床验证的抑郁、焦虑、精神分裂症三种病种大规模数据；②在PMLF中引入范式感知文本描述作为提示，利用跨模态对比学习预训练视频编码器，并通过跨注意力与对比学习实现多模态交互与融合

**🔧 技术方法**

使用Gemini-2.5-pro生成范式描述，OpenFace提取面部特征，MFCC提取语音特征，BERT-Base-Chinese编码文本；采用CLIP风格跨模态对比学习、Transformer/ResNet/Mamba等骨干网络进行特征学习与融合

**📊 数据集**

使用自建的MMH数据集（928份样本，包含视频、音频、文本，三种病种及健康对照）

**📈 对比分析**

与多种基线（如DNet、DepMamba、Mamba、ResNet-Transformer等）及通用LLM/MLLM（GPT‑4o、Gemini‑3、Qwen‑2.5‑VL 等）对比，PMLF在四分类任务上实现了最高准确率91.94%，宏F1 72.44%，显示显著优于传统方法和通用模型

**⚠️ 局限性**

局限性在于未考虑同一患者的共病情况，且仅覆盖三种主要精神疾病，未来需扩展到更广泛的共病场景与多病种诊断

---

## 267. Disrupting Cognitive Passivity: Rethinking AI-Assisted Data Literacy through Cognitive Alignment

**arXiv ID:** 2604.02783 | [PDF](https://arxiv.org/pdf/2604.02783v1)

**作者:** Yongsu Ahn `[一作]` (Boston College), Benjamin Bach `[通讯]` (Inria)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

**🎯 论文内容**

提出认知对齐框架，探讨 AI 在数据素养中的交互模式与用户认知需求的匹配，从而减少认知被动。

**💡 创新点**

创新点在于将认知需求（可接受 vs. 需要推理）与 AI 交互模式（传递性 vs. 询问性）映射为四种对齐状态，并阐释其对学习效果的影响，提出“认知对齐”视角。

**🔧 技术方法**

运用认知心理学理论（双过程理论、ICAP 框架、可取困难理论）和现有 AI 交互设计案例进行理论构建与论证。

**📊 数据集**

未使用具体数据集，本文为理论与综述性工作。

**📈 对比分析**

未进行实验或性能评估，主要通过文献综述与案例分析来支撑框架，未给出量化指标。

**⚠️ 局限性**

局限性包括：缺乏实证验证与动态适配机制；认知需求的实时检测与评估方法尚未提出；框架在不同领域、不同用户熟练度下的适用性需进一步研究。

---

## 268. DynFOA: Generating First-Order Ambisonics with Conditional Diffusion for Dynamic and Acoustically Complex 360-Degree Videos

**arXiv ID:** 2604.02781 | [PDF](https://arxiv.org/pdf/2604.02781v1)

**作者:** Ziyu Luo `[一作]` (Beijing Technology and Business University), Yiran Shen `[通讯]` (Shandong University)

**通讯引用:** 1684 | [OpenAlex ID](https://openalex.org/A5013564110)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

利用 360 度视频的几何与材质信息，通过物理驱动的条件扩散模型生成高保真 FOA 空间音频，并构建了 M2G-360 评测数据集。

**💡 创新点**

① 将三维几何重建（3DGS）与表面材质属性联动，首次将环境物理特征直接注入扩散过程；② 采用多条件交叉模态融合，让声场生成在物理约束下保持空间一致性；③ 通过构造专门的 M2G-360 子集（移动源、多源、几何复杂）验证模型在极端环境下的稳健性。

**🔧 技术方法**

单目 360 视频的声源检测与深度估计；语义分割与材质属性映射；3D Gaussian Splatting 进行几何重建；FOA 潜在编码器；多条件扩散生成器（U‑Net 逆扩散）与 DPM‑Solver++ 高效采样；HRTF 头部跟踪渲染；以及多种评测指标（DOA、SNR、EDT、FD、KL、STFT、SI‑SDR、MOS）。

**📊 数据集**

Sphere360（公开基准）和自建的 M2G-360（600 条 10 秒 360 视频，按 MoveSources、Multi‑Source、Geometry 三子集划分），使用 H.264、720p、30 FPS、16 kHz FOA 录音。

**📈 对比分析**

与 ViSAGe、Diff‑SAGe、MMAudio+SP、OmniAudio 等现有方法在 Sphere360 及 M2G-360 子集上对比，评测 DOA、SNR、EDT、FD、KL、STFT、SI‑SDR、MOS 等。DynFOA 在所有指标上均领先，DOA 误差下降 26%~47%，EDT 降低 33%~40%，FD 与 KL 下降 32%~40%，MOS‑SQ/Af 分数提升 0.4~0.5 分。

**⚠️ 局限性**

① 材质估计仅基于语义分割，缺乏频率依赖的真实声学响应；② 视觉与音频编码器冻结，未实现端到端训练；③ 目前主要针对室内环境，缺乏室外或跨媒体的实验；④ 生成过程仍需 50 步扩散采样，虽然比 1000 步快，但在极低延迟场景中仍有挑战。

---

## 269. When Modalities Remember: Continual Learning for Multimodal Knowledge Graphs

**arXiv ID:** 2604.02778 | [PDF](https://arxiv.org/pdf/2604.02778v1)

**作者:** Linyu Li `[一作]` (Peking University), Nyima Tashi `[通讯]` (Tibet University)

**通讯引用:** 499 | [OpenAlex ID](https://openalex.org/A5050819315)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种面向持续多模态知识图推理的新框架MRCKG。

**💡 创新点**

通过多模态结构协同课程学习、跨模态知识保持和多模态对比回放三大模块，将预训练多模态特征作为语义锚点，解决持续学习中的灾难性遗忘。

**🔧 技术方法**

使用Transformer编码器结合冻结的BEiT和BERT视觉/文本特征，构建多模态协同训练，采用正则化、对比学习和样本重要性采样等技术。

**📊 数据集**

在扩展后的DB15K、MKG-W、MKG-Y三大MMKG数据集构造的9个持续学习基准上进行实验。

**📈 对比分析**

与10+基线（包括单模态微调、持续学习方法、专用CKGE以及多模态+微调/持续学习）比较，MRCKG在所有基准上均获得最高MRR和Hits@10，且在新知识学习上显著提升。

**⚠️ 局限性**

仍存在一定程度的遗忘，且多模态联合优化导致参数空间增大、易受噪声影响；对预训练特征的依赖限制了跨域适用性。

---

## 270. Evaluating the Environmental Impact of using SLMs and Prompt Engineering for Code Generation

**arXiv ID:** 2604.02776 | [PDF](https://arxiv.org/pdf/2604.02776v1)

**作者:** Md Afif Al Mamun `[一作]` (University of Calgary), Novarun Deb `[通讯]` (University of Calgary)

**通讯引用:** 159 | [OpenAlex ID](https://openalex.org/A5072598400)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对 11 个开源小型语言模型在 6 种提示策略下的代码生成准确性与能耗/碳排放进行系统评估，并提出“碳每正确答案”指标。

**💡 创新点**

首次将提示工程与可持续性度量结合，量化不同提示对能耗与碳排放的影响，并发现链式思维能在不显著牺牲准确率的情况下降低约 80% 能耗。

**🔧 技术方法**

采用 DSPy 提示框架、CodeCarbon 能耗跟踪、Ollama 模型服务，以及 Chain-of-Thought、Program-of-Thought、Self-Consistency、Least-to-Most、ReAct 等提示技术。

**📊 数据集**

使用 HumanEval+ 与 MBPP+ 两大代码生成基准，共计 314 题目（HumanEval+ 164 题，MBPP+ 150 题）进行评测。

**📈 对比分析**

对每个模型+提示组合计算 Pass@1、平均能耗(kWh)、CO₂ 排放(kg)、推理时间和 token 数；结果显示 Self-Consistency 正确率最高但能耗最高，Chain-of-Thought 在准确率与碳排放上最优；硬件与地区差异对碳排放影响显著。

**⚠️ 局限性**

局限性包括仅测试本地部署、少量硬件与模型、碳强度采用历史平均值、评测仅覆盖两类基准，未考虑云环境、长期生命周期或更复杂任务。

---

## 271. Open Challenges for Secure and Scalable Wi-Fi Connectivity in Rural Areas

**arXiv ID:** 2604.02774 | [PDF](https://arxiv.org/pdf/2604.02774v1)

**作者:** Philip Virgil Berrer Astillo `[一作]` (University of San Carlos), Mathy Vanhoef `[通讯]` (KU Leuven)

**通讯引用:** 1461 | [OpenAlex ID](https://openalex.org/A5065091484)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文通过对菲律宾和印度农村地区的 Wi-Fi 热点进行实地测绘、对 Piso‑WiFi 生态系统进行安全评估，并对印度 PM‑WANI 框架进行初步风险分析。

**💡 创新点**

创新点在于首次系统地量化低成本付费热点的普及率，实测两种典型攻击（用户冒充和 Rogue AP）对 Piso‑WiFi 的可行性，并提出针对单次/多次使用场景的威胁模型及基于签名交换的安全本地缓存方案。

**🔧 技术方法**

采用 WiGLE Android App 收集 AP 计数、SSID、BSSID、位置信息；利用 Wi‑Fi 捕获工具（如 Wireshark、aircrack-ng）进行攻击验证；设计了基于 SAE‑PK、RADIUS 认证的安全连接协议与 LED‑基线验证方法；提出使用 Signed Exchanges (SXG) 进行数据缓存。

**📊 数据集**

使用的主要数据集是基于 WiGLE 的收集结果，涵盖 2,620 个菲律宾 AP、6,464 个印度 AP，其中约 5% 识别为 Piso‑WiFi 热点，3 个为 PM‑WANI 热点；还引用公开的 PM‑WANI 中央注册表 XML。

**📈 对比分析**

对比方法主要是对比调查地区的热点比例、TKIP 支持率（菲律宾约 20%，印度 13%）以及两类攻击的成功率（在实验室与真实环境均能成功）；未给出数值化性能指标，但指出攻击仅需商用硬件与最小技术门槛即可完成。

**⚠️ 局限性**

局限性包括：对 PM‑WANI 的分析仅为理论风险评估，未进行实测；调查样本受限于驾驶路线与时间，可能存在采样偏差；数据集中 PM‑WANI 热点数量极少，导致结论不够稳健。

---

## 272. ContractShield: Bridging Semantic-Structural Gaps via Hierarchical Cross-Modal Fusion for Multi-Label Vulnerability Detection in Obfuscated Smart Contracts

**arXiv ID:** 2604.02771 | [PDF](https://arxiv.org/pdf/2604.02771v1)

**作者:** Minh-Dai Tran-Duong `[一作]` (University of Information Technology), Phan The Duy `[通讯]` (University of Information Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `c773407a-6119-4871-b8b3-1e7ae17a6851` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 ContractShield，一种多模态框架，用于检测以太坊智能合约中的多标签漏洞。

**💡 创新点**

创新点在于三层层次融合机制：自注意力提炼单模态特征、跨模态注意力捕获模态间互补关系、以及自适应权重动态调节模态贡献；并通过滑动窗口的 CodeBERT、扩展 LSTM 以及 GATv2 对三种特征（源代码、opcode 序列、CFG）进行高效建模，显著提升了对混淆攻击的鲁棒性。

**🔧 技术方法**

采用了预训练 Transformer (CodeBERT) 结合滑动窗口、扩展 LSTM (xLSTM)、图注意力网络 (GATv2)，以及自注意力、跨模态注意力和自适应加权的层次融合策略；在训练时使用交叉熵损失、二值化 sigmoid 输出。

**📊 数据集**

使用公开基准数据集 SoliAudit+SmartBugs、CGT Weakness、DAppScan，并在此基础上构造了 BiAn（源代码混淆）和 BOSC（字节码混淆）两套混淆 benchmark，以评估鲁棒性。

**📈 对比分析**

与单模态、双模态、以及三种现有最优方法（Qian、Cheong、Deng）进行对比。ContractShield 在非混淆数据上 Hamming Score 达 89.16%–91.47%、F1 分数 91.47% 以上；在混淆场景下 Hamming Score 下降不超过 4.8%，性能稳健；整体性能优于所有基线。

**⚠️ 局限性**

局限性包括：仅针对 Solidity 合约，可能无法直接迁移到其他链或语言；混淆工具覆盖范围有限，真实攻击手法更为多样；数据集标签可能存在噪声；模型复杂度较高，训练和推理时间较大；解释性不足，难以细粒度分析每个模态贡献。

---

## 273. Deformation-based In-Context Learning for Point Cloud Understanding

**arXiv ID:** 2604.02845 | [PDF](https://arxiv.org/pdf/2604.02845v1)

**作者:** Chengxing Lin `[一作]` (University of Electronic Science and Technology of China), Wen Li `[通讯]` (University of Electronic Science and Technology of China)

**通讯引用:** 17110 | [OpenAlex ID](https://openalex.org/A5100320305)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出 DeformPIC 框架，将点云 In‑Context Learning 从传统的 Masked Point Modeling 转为基于变形的方式，直接对查询点云进行任务特定变形；

**💡 创新点**

通过分离 Deformation Extraction Network 与 Deformation Transfer Network 并利用提示提取与迁移变形信息，解决 MPM 的几何自由重建与训练‑推理目标不一致问题；

**🔧 技术方法**

采用 mini‑PointNet 对点云进行编码，Transformer 加 AdaLN‑Zero 调制实现任务条件，Chamfer Distance 作为损失，使用 AdamW 优化，并用 t‑SNE 可视化任务特征；

**📊 数据集**

在 ShapeNet In‑Context（重建、去噪、配准、分割）上进行评估，并构建 ModelNet40 In‑Context 与 ScanObjectNN In‑Context 作为跨域泛化测试集；

**📈 对比分析**

与 PIC、PIC‑Cat、PIC‑S 系列以及多任务模型（Point‑MAE、ReCon、UniPre3D 等）对比，DeformPIC 在所有任务上均取得显著提升（如配准 CD 2.0 对比 6.7，重建 CD 下降 1.6/1.8，分割 mIoU 提升 34.1）；

**⚠️ 局限性**

对相似任务（如重建与去噪）的区分仍不充分，变形网络在极端噪声或大变形场景下表现有限，且大局与细节的同时优化仍有提升空间。

---

## 274. NavCrafter: Exploring 3D Scenes from a Single Image

**arXiv ID:** 2604.02828 | [PDF](https://arxiv.org/pdf/2604.02828v1)

**作者:** Hongbo Duan `[一作]` (Tsinghua University), Xueqian Wang `[通讯]` (Tsinghua University)

**通讯引用:** 5458 | [OpenAlex ID](https://openalex.org/A5100737125)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

从单张图像生成可控的新视角视频并构建高精度3D场景

**💡 创新点**

多阶段相机控制、碰撞感知相机路径规划以及基于深度对齐的3D高斯光散射重建

**🔧 技术方法**

视频扩散模型、LoRA注意力调制、3D高斯光散射、深度对齐监督与结构正则化

**📊 数据集**

RealEstate10K、DL3DV以及Tanks-and-Temples测试集

**📈 对比分析**

与Wonderland、Scene-Splatter、ViewCrafter等基线在LPIPS/PSNR/SSIM、FID/FVD和相机误差等指标上均取得更低误差、更高分数，显著提升视角一致性和重建质量

**⚠️ 局限性**

对极端快速相机运动、长时序几何一致性和动态场景的适用性仍有限

---

## 275. MFE: A Multimodal Hand Exoskeleton with Interactive Force, Pressure and Thermo-haptic Feedback

**arXiv ID:** 2604.02820 | [PDF](https://arxiv.org/pdf/2604.02820v1)

**作者:** Ziyuan Tang `[一作]` (ShanghaiTech University), Chenxi Xiao `[通讯]` (ShanghaiTech University)

**通讯引用:** 773 | [OpenAlex ID](https://openalex.org/A5075464348)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出并实现了一款低成本、开源的多模态手部外骨骼MFE，用于在机器人远程操作中同时提供力、压力和温度反馈；

**💡 创新点**

实现了三种模态的联合反馈（主动力反馈、微流体压力与振动反馈、热电温度反馈），并将其集成到一个完整的双向遥控系统中，首次开源该多模态硬件平台；

**🔧 技术方法**

采用动量控制的动力学外骨骼、基于电渗流的微流体平面驱动器、热电冷热泵、Dynamixel电机与Hall传感器、STM32 MCU、PWM驱动与PID闭环控制等技术；

**📊 数据集**

本文未使用公开数据集，而是通过自制的遥控机器人手（Inspire Hand）与自制温度传感膜进行实验验证；

**📈 对比分析**

通过设备性能曲线、力与压力响应测量以及三项用户实验（物体识别、柔性物体抓取、温度辨别），显示多模态反馈可显著提升任务成功率（最高达100%），且在统计上优于单一模态；

**⚠️ 局限性**

主要局限包括温度反馈延迟约4秒、力与压力反馈响应时间几百毫秒、单热电模块缺乏空间分辨率、以及高耦合电机/微流体的布线复杂性，后续工作计划改进热电阵列、缩短响应时间和简化硬件连接。

---

## 276. Student-in-the-Loop Chain-of-Thought Distillation via Generation-Time Selection

**arXiv ID:** 2604.02819 | [PDF](https://arxiv.org/pdf/2604.02819v1)

**作者:** Chaoqun He `[一作]` (Tsinghua University), Lijie Wen `[通讯]` (Tsinghua University)

**通讯引用:** 4547 | [OpenAlex ID](https://openalex.org/A5030845033)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Gen-SSD 框架，在教师生成推理轨迹的过程中让学生主动参与，利用学生自身的困惑度（PPL）在生成阶段即时筛选并修剪不易被学生学习的思考路径，从而得到更易学习、更稳定的 CoT 数据。

**💡 创新点**

创新点在于：① 采用学生在生成时即刻评估教师候选推理片段，突破传统后期筛选的局限；② 通过 PPL 作为学习能力度量，自动把握教师输出的难度阈值；③ 引入冷启动细化步骤，使学生先适应教师的推理格式，再进行自选，从而显著提升学生的学习效果。

**🔧 技术方法**

核心技术包括：多样本教师采样（chunk‑wise），学生基于 PPL 的即时选择策略，基于学生困惑度的自适应采样大小；冷启动阶段的对齐微调；链式推理（CoT）蒸馏；对比实验中的多基准评估。

**📊 数据集**

数据集与模型：使用 OpenMathReasoning（约 25k 题目）构成训练集；评估基准包括 AIME25、AIME24、AMC23、OlympiadBench、GSM8K；教师模型为 QwQ‑32B（以及 DeepSeek‑R1 系列），学生模型为 Qwen2.5‑Math‑1.5B。

**📈 对比分析**

与 Standard KD、Self‑Distillation、MCC‑KD、MoRSD 等基线对比，Gen‑SSD 在所有数学推理基准上均取得最优或接近最优成绩，平均提升约 5.9 分（对 Standard KD）且在多步推理任务上提升显著；实验还显示所选轨迹的 PPL 更低、长度更短，学习曲线更平稳。

**⚠️ 局限性**

局限性：① 需要大规模教师模型支持生成和采样，仍有一定计算和能源成本；② 依赖教师输出的推理质量，若教师失误或偏差会直接影响学生；③ 在冷启动阶段需额外的数据和训练步骤；④ 目前仅验证于数学推理任务，尚未在更广泛的多模态或代码生成领域展示普适性；⑤ PPL 作为选择指标虽有效，但并非绝对最优，仍可能忽略某些对学生有益的高难度轨迹。

---

## 277. QAPruner: Quantization-Aware Vision Token Pruning for Multimodal Large Language Models

**arXiv ID:** 2604.02816 | [PDF](https://arxiv.org/pdf/2604.02816v1)

**作者:** Xinhao Wang `[一作]` (Peking University), Yongtao Wang `[通讯]` (Peking University)

**通讯引用:** 4745 | [OpenAlex ID](https://openalex.org/A5100781631)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

结合低比特后训练量化（PTQ）与视觉token剪枝，提出一种量化感知的剪枝框架。

**💡 创新点**

创新点在于设计混合量化敏感度指标，既考虑局部组级量化误差，又捕捉全局激活异常，同时与语义相关性融合，实现对数值稳定性与语义信息的双重优化。

**🔧 技术方法**

使用组级量化模拟、全局异常强度、量化敏感度融合、无训练量化与剪枝技术，并在LLaVA架构上实现。

**📊 数据集**

使用ScienceQA数据集及LLaVA（1.3 7B、13B、1.5 7B）等模型进行评估。

**📈 对比分析**

与传统语义剪枝+PTQ基线对比，在保留12.5%视觉token时提升约2–3%准确率，甚至在低token预算下超过密集量化模型的表现。

**⚠️ 局限性**

局限性包括：仅在特定PTQ和视觉编码器上验证，缺少更低比特或多模态的通用性验证；在极低token预算或更大模型规模下的性能及计算成本仍需进一步研究。

---

## 278. ChatSVA: Bridging SVA Generation for Hardware Verification via Task-Specific LLMs

**arXiv ID:** 2604.02811 | [PDF](https://arxiv.org/pdf/2604.02811v1)

**作者:** Lik Tung Fu `[一作]` (Southeast University), Jun Yang `[通讯]` (Southeast University)

**通讯引用:** 22415 | [OpenAlex ID](https://openalex.org/A5100719885)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本工作提出了自动化 SVA 生成框架 ChatSVA，解决硬件验证中手工编写 SVAs 的低效与错误问题。

**💡 创新点**

核心创新在于将 SVA 生成拆解为四阶段长链推理，并结合 AgentBridge 数据合成平台生成高质量训练数据，从而显著提升功能正确性与覆盖率。

**🔧 技术方法**

采用多代理协作框架、监督微调(SFT)、检索增强生成(RAG)、GPT‑4o 以及 Llama3.1‑8B 模型等技术手段。

**📊 数据集**

使用了包含 24 个 RTL 设计的 FIXME 公测基准以及 15.36 GB 的合成数据集（由 AgentBridge 生成）。

**📈 对比分析**

与 GPT‑4o、DeepSeek‑R1 以及 AssertLLM 等基线对比，ChatSVA 在语法通过率 98.66%、功能通过率 96.12% 和功能覆盖率 82.50% 上实现了 SOTA，功能覆盖率提升超过 11 倍。

**⚠️ 局限性**

主要局限在于对黄金 RTL 数据的依赖可能引入验证污染，以及在极少样本场景下对模型泛化能力的进一步验证仍待探索。

---

## 279. CharTool: Tool-Integrated Visual Reasoning for Chart Understanding

**arXiv ID:** 2604.02794 | [PDF](https://arxiv.org/pdf/2604.02794v1)

**作者:** Situo Zhang `[一作]` (Shanghai Jiao Tong University), Kai Yu `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 21243 | [OpenAlex ID](https://openalex.org/A5100758006)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种工具集成的视觉推理方法，用于图表理解，旨在解决多模态大语言模型在图表推理中的挑战。

**💡 创新点**

创新点在于提出了一种可扩展的双源数据管道，结合合成图表和真实图表，构建多样化的高质量图表训练数据，并引入外部工具以增强模型的视觉感知和数值推理能力。

**🔧 技术方法**

使用了图像裁剪和基于代码的计算工具，结合强化学习进行工具集成推理。

**📊 数据集**

使用了合成的图表数据集和从科学文献中提取的真实图表数据集，构建了一个包含100k高质量图表和挑战性问答样本的数据集。

**📈 对比分析**

通过在六个图表基准上的广泛实验，方法在不同模型规模上均表现出一致的性能提升，尤其是-7B模型在CharXiv和ChartQAPro上分别提高了8.0%和9.78%。

**⚠️ 局限性**

限制在于当前方法仍然依赖于合成数据的多样性和复杂性，可能在处理某些特定类型的图表时表现不佳。

---

## 280. Fully Byzantine-Resilient Distributed Multi-Agent Q-Learning

**arXiv ID:** 2604.02791 | [PDF](https://arxiv.org/pdf/2604.02791v1)

**作者:** Haejoon Lee `[一作]` (University of Michigan), Dimitra Panagou `[通讯]` (University of Michigan)

**通讯引用:** 2407 | [OpenAlex ID](https://openalex.org/A5059647993)

**关键词:** `ca287573-fa3b-4b00-8a06-ae3eda6fdb99` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了完全鲁棒的分布式 Q‑学习算法 FRQD‑learning，能够在存在 Byzantine 边攻击的情况下实现各智能体价值函数几乎必然收敛至全局最优；

**💡 创新点**

创新点在于利用两跳邻居的冗余信息构建筛选机制，结合 (r,r′)‑冗余图的拓扑条件，保证通信保持无向结构，从而恢复原始 QD‑learning 的收敛性质；

**🔧 技术方法**

采用 QD‑learning 作为基线，加入两跳冗余过滤、图论分析、拉普拉斯矩阵更新以及多路径验证等技术；

**📊 数据集**

在实验中使用一个 10 机器人、7 状态、6 任务的马尔科夫决策过程（MDP）模拟数据，未使用公开数据集；

**📈 对比分析**

与无攻击时的 Oracle QD‑learning 以及之前的鲁棒 QD‑learning 进行对比，在 F=1 的边攻击下，FRQD‑learning 能收敛至最优 Q 值和策略，而对照方法仅收敛至近似最优，性能优越；

**⚠️ 局限性**

局限性包括：通信成本在第二轮通信中可能升至 O(d²)；仅在静态网络上验证（虽然声称可推广到时变网络）；以及对 (6F+1,0)‑冗余图的构造和验证仍需满足一定拓扑条件，未证明对更一般攻击模型的适用性。

---

## 281. Structure-Aware Commitment Reduction for Network-Constrained Unit Commitment with Solver-Preserving Guarantees

**arXiv ID:** 2604.02788 | [PDF](https://arxiv.org/pdf/2604.02788v1)

**作者:** Guangwen Wang `[一作]` (Arizona State University), Baosen Zhang `[通讯]` (University of Washington)

**通讯引用:** 6213 | [OpenAlex ID](https://openalex.org/A5013901541)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文提出了一种结构感知的承诺变量约束框架，在不改动原始网络约束单位承诺模型的前提下，通过人工智能辅助选择稀疏且结构稳定的承诺变量进行固定，从而显著减小分支定界树的搜索空间。

**💡 创新点**

创新点在于将大型语言模型（LLM）用于生成仅部分固定变量的稀疏掩码，并配合预检与后检，保证了可行性与优化证据完全由原始MILP求解器提供；同时证明了修约后可行域为原始可行域的子集，确保了求解的合法性和可解释性。

**🔧 技术方法**

主要技术包括：结构化提示工程与LLM（GPT‑4o）生成掩码；预检逻辑判断约束一致性；传统DC功率流与MILP求解器（Gurobi）；以及可视化对比与性能度量。

**📊 数据集**

使用的数据集包括IEEE 57‑bus、RTS 73‑bus、IEEE 118‑bus以及其扩增版本118‑bus+20和118‑bus+30，并在这些案例上施加负荷扰动和全年度模拟。

**📈 对比分析**

与传统完整MILP、基于K‑均值和K‑NN的历史数据掩码方法对比，LLM辅助方法在节点数、单纯形迭代和求解时间上分别提升约6‑7倍、50‑80%与85‑97%，同时目标值偏差低于0.3%，在更大规模或负荷不确定情境下仍保持高效与稳定。

**⚠️ 局限性**

主要局限是对LLM生成掩码的质量高度依赖；当系统结构骤变或历史数据极度缺乏时，掩码可能导致可行域过度收窄，引起目标偏差增大；此外，固定变量比例的选择仍需经验调节，且对极端操作条件下的鲁棒性尚需进一步验证。

---

## 282. Generalized Small Object Detection:A Point-Prompted Paradigm and Benchmark

**arXiv ID:** 2604.02773 | [PDF](https://arxiv.org/pdf/2604.02773v1)

**作者:** Haoran Zhu `[一作]` (Wuhan University), Gui-Song Xia `[通讯]` (Wuhan University)

**通讯引用:** 22075 | [OpenAlex ID](https://openalex.org/A5073032922)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文构建了大规模多域小物体检测数据集TinySet-9M，并提出点提示小物体检测（P²SOD）范式及其实现框架DEAL，用于在推理时通过极少的点提示激活小物体的语义表示；

**💡 创新点**

创新点在于将点提示视为类别级语义锚，在推理阶段引入点提示以提升小物体的特征对比度和定位精度；同时引入循环点提示策略（PG‑CPP）使模型在训练中自适应不同点提示配置；

**🔧 技术方法**

技术核心包括点提示嵌入与Transformer交互、混合特征增强（HFE）、点引导密度激活、预测驱动的循环点提示以及基于RT‑DETR的解码器；

**📊 数据集**

使用了新构建的TinySet‑9M（约9M注释、6个领域）进行训练和验证，并在DIOR、DOTA‑v2.0等未见数据集上进行跨域评估；

**📈 对比分析**

在TinySet‑9M上，DEAL相较于RT‑DETR提升约12.1 AP，单点提示下AP₀.⁷⁵显著提高31.4%；在DOTA‑v2.0与DIOR上，DEAL在未见数据集上实现AP超过50%，明显优于SAM3和Rex‑Omni等零样本方法；

**⚠️ 局限性**

局限性包括：对细粒度同类小物体的区分能力有限，主要针对小物体而非大物体场景；对推理时阈值依赖较大，需额外调参；

---

## 283. Multiple-Debias: A Full-process Debiasing Method for Multilingual Pre-trained Language Models

**arXiv ID:** 2604.02772 | [PDF](https://arxiv.org/pdf/2604.02772v1)

**作者:** Haoyu Liang `[一作]` (Guangdong University of Technology), Dong Zhou `[通讯]` (Guangdong University of Foreign Studies)

**通讯引用:** 4059 | [OpenAlex ID](https://openalex.org/A5100366115)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出了一种完整流程的多语言去偏方法Multiple-Debias，用于消除多语言预训练模型中的性别、种族和宗教偏见。

**💡 创新点**

创新点在于将多语言反事实数据增强、跨语言Self-Debias以及参数高效微调整合为预处理-中处理-后处理三阶段，并将CrowS-Pairs扩展至德语、西班牙语、中文和日语。

**🔧 技术方法**

采用的技术包括多语言反事实数据增强(MCDA)、多语言Self-Debias(MSD)、三种参数高效微调（Adapter、Prefix、Prompt）以及全量微调。

**📊 数据集**

使用的数据集为五种语言（英语、中文、日语、德语、西班牙语）的维基百科语料，结合扩展后的CrowS-Pairs以及MBE评测集。

**📈 对比分析**

通过与单语言CDA、Self-Debias以及全量微调的对比实验，结果显示多语言方法在CrowS-Pairs和MBE上的偏差分数明显下降，平均偏差分数接近0，性能优于传统方法。

**⚠️ 局限性**

局限性包括不同语言和属性的去偏效果不均衡，对中文和日语等语言的评测仍有限，且方法仍需在更多语言与属性上验证。

---

## 284. Learning from Synthetic Data via Provenance-Based Input Gradient Guidance

**arXiv ID:** 2604.02946 | [PDF](https://arxiv.org/pdf/2604.02946v1)

**作者:** Koshiro Nagano `[一作]` (Keio University), Hideo Saito `[通讯]` (Keio University)

**通讯引用:** 7966 | [OpenAlex ID](https://openalex.org/A5005819073)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了利用合成数据过程中的来源信息（provenance）作为辅助监督，通过输入梯度引导抑制非目标区域梯度，从而直接学习目标区域的判别表示。

**💡 创新点**

创新点在于将合成过程得到的 provenance 信息用于构造梯度约束（provenance loss），实现对目标与非目标梯度的分离与抑制，并且该框架可兼容多种合成方法。

**🔧 技术方法**

使用的技术包括 CutMix、ResizeMix、PuzzleMix 等图像/骨架混合方法，图像生成模型（如 Stable Diffusion）编辑，输入梯度引导（soft/hard 标签梯度正则化），以及 VGG16、ResNet‑50、SAT、SKP 等深度学习模型。

**📊 数据集**

所用数据集包括 CUB‑200‑2011、iWildCam、Waterbirds（图像分类与弱监督定位），UCF101‑24（骨架序列动作定位），用于评估不同任务与模态。

**📈 对比分析**

与基线方法（CutMix、ResizeMix、PuzzleMix、SKP、ALIA 等）对比，弱监督目标定位平均准确率提升 2–5%，时空动作定位 AP 提升 1.7pp，分类 Top‑1 精度提升 1–9pp，证明该方法在多任务、多模态上均具有效果。

**⚠️ 局限性**

局限性在于需要额外计算梯度正则化导致训练时间与显存增加，且对 provenance mask 的精度有一定要求；若 mask 误差过大会影响性能。

---

## 285. Digital Twin-Assisted In-Network and Edge Collaboration for Joint User Association, Task Offloading, and Resource Allocation in the Metaverse

**arXiv ID:** 2604.02938 | [PDF](https://arxiv.org/pdf/2604.02938v1)

**作者:** Ibrahim Aliyu `[一作]` (Chonnam National University), Jinsul Kim `[通讯]` (Chonnam National University)

**通讯引用:** 871 | [OpenAlex ID](https://openalex.org/A5083642709)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

提出一种基于数字孪生（DT）的边缘-网络协作框架，用于元宇宙中XR用户设备的联合用户关联、任务分拆与资源分配

**💡 创新点**

将数字孪生与增量网络计算（INC）相结合，构建Stackelberg马尔可夫博弈模型；同时引入异步多智能体强化学习（AMRL）和零一背包算法实现在线的OFMO与POAL决策；通过精确潜在游戏实现分散式用户关联

**🔧 技术方法**

游戏理论（精确潜在游戏）、马尔可夫决策过程（MDP）、多智能体强化学习（AMRL）、PPO、零一背包优化、分散式拉格朗日方法、离散/连续动作空间

**📊 数据集**

仿真场景：6个XR用户、4个INC节点、1个边缘服务器，使用基于Zipf分布的任务请求模型、真实无线信道衰落与有限容量约束；未使用公开数据集，全部为仿真数据

**📈 对比分析**

与三种基线（随机、等比例、比例）以及三种AMRL变体（AHMRL、MASC、AC）对比；结果显示AHMRL在用户效用、上行速率、能耗与延迟方面均优于基线；MASC在大INC容量下表现更好；所有AMRL模型显著优于启发式策略

**⚠️ 局限性**

对网络规模和任务负载的扩展有限；模型假设任务请求转移概率未知但对仿真不敏感；训练过程对超参数和收敛速度敏感；未在真实硬件或大规模真实场景中验证

---

## 286. If It's Good Enough for You, It's Good Enough for Me: Transferability of Audio Sufficiencies across Models

**arXiv ID:** 2604.02937 | [PDF](https://arxiv.org/pdf/2604.02937v1)

**作者:** David A. Kelly `[一作]`, Hana Chockler `[通讯]` (King's College London)

**通讯引用:** 1996 | [OpenAlex ID](https://openalex.org/A5053852359)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268`

**🎯 论文内容**

分析不同音频分类模型对最小充分信号（minimal sufficient signal）的转移能力，提出转移性定义并在实验中验证。

**💡 创新点**

首次系统研究音频模型间充分信号的可转移性，引入“flat‑earther”模型概念，并揭示深度伪造音频在信息理论特征上的差异。

**🔧 技术方法**

利用实际因果分析、频域（FFT）提取最小充分/完整信号、熵阈值判定、组合与谱熵比较等技术。

**📊 数据集**

三类任务数据集：语音情绪识别（24名演员录制的情绪语音）、音乐流派（1,000段30秒WAV文件，10个流派）以及深度伪造音频（ASVspoof2019 与 In The Wild）。

**📈 对比分析**

通过α‑transferability（类别一致率）和β‑transferability（熵阈值）与多种后端模型（Wav2Vec2、HuBERT、DistilHubert、Whisper）进行对比；实验显示音乐流派约26%转移，情绪识别最低，伪造检测最高；完整信号比最小信号更易转移。

**⚠️ 局限性**

仅考察单一充分信号，未覆盖多重独立充分信号；结果受模型训练细节、随机种子及数据分布影响；缺乏用户研究与对不同架构的广泛验证。

---

## 287. BEVPredFormer: Spatio-temporal Attention for BEV Instance Prediction in Autonomous Driving

**arXiv ID:** 2604.02930 | [PDF](https://arxiv.org/pdf/2604.02930v1)

**作者:** Miguel Antunes-García `[一作]` (University of Alcalá), Luis M. Bergasa `[通讯]` (University of Alcalá)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `edb9d762-f411-4838-a852-f2d638b018db` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

设计并实现了一种仅使用多摄像头输入的BEV实例预测模型BEVPredFormer，能够在未来时刻生成车辆实例的分割与运动信息。

**💡 创新点**

创新点包括：无递归的时间处理模块，使用分离空间与时间注意力的Gated Transformer；差分引导特征提取提升时序表达；基于BEVFormer的2D‑to‑BEV投影，省去深度估计；以及多尺度预测头实现高效的未来帧生成。

**🔧 技术方法**

技术手段涵盖：EfficientViT特征提取器、BEVFormer投影+Sparse UNet、PredFormer块与Gated Transformer、差分模块、跨帧注意力与多尺度解码，以及两阶段训练策略。

**📊 数据集**

使用公开的nuScenes数据集（1000场景），在训练集700、验证集150、测试集150上进行评估。

**📈 对比分析**

与PowerBEV、Fiery、StretchBEV、BEVerse、DMP等SOTA方法比较，BEVPredFormer在短距离验证集上取得VPQ 54.9、IoU 63.9，长距离上VPQ 33.3、IoU 40.9，显著优于或与现有最佳方法持平。

**⚠️ 局限性**

局限性包括：对极远距离或长时序预测仍易出现误差；推理时对高分辨率输入要求较高，导致延迟上升；以及在部分复杂交通场景下的运动预测仍不够精确。

---

## 288. Council Mode: Mitigating Hallucination and Bias in LLMs via Multi-Agent Consensus

**arXiv ID:** 2604.02923 | [PDF](https://arxiv.org/pdf/2604.02923v1)

**作者:** Shuai Wu `[一作]`, Zhijun Wang `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `afceb026-1760-41ae-8d86-010831a37d97` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了Council Mode多代理共识框架，利用多模型并行生成并通过结构化合成降低LLM幻觉与偏见。

**💡 创新点**

创新点在于异构专家并行、三阶段（分流‑并行生成‑合成）以及结构化共识合成模型。

**🔧 技术方法**

采用三种前沿LLM（GPT‑5.4、Claude Opus 4.6、Gemini 3.1 Pro）并行推理，配合专门的合成模型和智能分流分类器。

**📊 数据集**

评测使用 HaluEval、TruthfulQA 以及自制多领域推理基准，共计 500 题。

**📈 对比分析**

与单模型基线对比，Hallucination 率降低 35.9%，TruthfulQA 得分提升 7.8 分，偏见方差降低 85–89%，并通过消融验证各模块贡献。

**⚠️ 局限性**

主要局限是额外延迟（≈2–3 倍）和对多模型API的依赖，且若所有专家共享同一误解仍无法完全消除幻觉。

---

## 289. Learning Task-Invariant Properties via Dreamer: Enabling Efficient Policy Transfer for Quadruped Robots

**arXiv ID:** 2604.02911 | [PDF](https://arxiv.org/pdf/2604.02911v1)

**作者:** Junyang Liang `[一作]` (Shenzhen University), Jianqiang Li `[通讯]` (Shenzhen University)

**通讯引用:** 16187 | [OpenAlex ID](https://openalex.org/A5100393871)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 DreamTIP 框架，将任务不变属性（Task‑Invariant Properties）学习融入 Dreamer 世界模型，实现四足机器人在多样化、动态地形上的仿真到真实迁移。

**💡 创新点**

创新点：① 利用大语言模型自动生成任务不变属性并作为辅助目标；② 通过混合回放缓冲与负余弦相似度正则化实现高效、稳定的适配，避免表征崩溃与灾难性遗忘。

**🔧 技术方法**

技术：Dreamer+RSSM 世界模型、PPO 策略、LLM（GPT/DeepSeek）生成 TIP、混合 replay buffer、负余弦相似度正则化、冻结 recurrent 模块进行离线适配。

**📊 数据集**

数据集：Isaac Gym 仿真环境下八种迁移任务（Stair、Gap、Climb、Crawl、Tilt 等）以及 Unitree Go2 真实机器人的深度图与特权信息。

**📈 对比分析**

与 WMP、DreamTIP‑DWL、WMP w/ Finetune 等基线对比，模拟任务平均提升 28.1%；真实 Climb 任务 100% 成功率对比基线 10%，在多任务和不同难度下表现稳定，超越所有基线。

**⚠️ 局限性**

局限：长期运行导致世界模型误差累积，引起性能衰退；对真实数据采集仍受限，需要更丰富的数据提升长期预测准确性。

---

## 290. BioUNER: A Benchmark Dataset for Clinical Urdu Named Entity Recognition

**arXiv ID:** 2604.02904 | [PDF](https://arxiv.org/pdf/2604.02904v1)

**作者:** Wazir Ali `[一作]` (Quaid-e-Awam University of Engineering, Sciences & Technology), Muhammad Mazhar Younas `[通讯]` (Aror University)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文创建了一个专门用于乌尔都语临床命名实体识别（NER）的金标注基准数据集 BioUNER，并对其进行了多模型评测。

**💡 创新点**

创新点在于：①首个公开的乌尔都语临床NER数据集；②采用专业医务人员进行手工标注，交叉验证得到 0.78 的 Kappa；③在数据集上对传统与深度学习模型进行系统基准对比。

**🔧 技术方法**

技术包括：数据爬取与预处理、Doccano 工具进行 BIES 标注；模型包括 SVM、CRF、LSTM、mBERT 和 XLM‑RoBERTa；评估指标为 token 级别的 Precision、Recall 与 F1‑score。

**📊 数据集**

使用了 153K 词、15,073 条实体的 BioUNER 数据集（包含 6 类实体：Disease、Chemical、Drug、Gene、Protein、CellLine）。

**📈 对比分析**

实验比较显示：传统 CRF 达到 0.93 F1，LSTM 最高 0.95；相比之下 mBERT 0.72、XLM‑RoBERTa 0.73，均低于 LSTM，说明在当前设置下基于 LSTM 的序列模型更适合该低资源任务。

**⚠️ 局限性**

局限性包括：数据量相对有限，实体分布偏向 Disease；仅在单个测试集上评估，缺乏跨域或跨语言验证；Transformer 模型未采用更大规模预训练或领域自适应，导致表现不及传统 LSTM。

---

## 291. EvaNet: Towards More Efficient and Consistent Infrared and Visible Image Fusion Assessment

**arXiv ID:** 2604.02896 | [PDF](https://arxiv.org/pdf/2604.02896v1)

**作者:** Chunyang Cheng `[一作]` (Wuxi School of Medicine), Josef Kittler `[通讯]` (University of Surrey)

**通讯引用:** 51642 | [OpenAlex ID](https://openalex.org/A5028209738)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 EvaNet，一种轻量级学习框架，用于红外与可见图像融合结果的统一评估。

**💡 创新点**

创新点在于：①基于分解的三分支结构（红外、可见、环境）实现对多种融合指标的同时预测；②使用对比学习和大语言模型生成环境标签，动态校正不同模态的权重；③构建一致性评估协议，量化指标与人类感知及下游任务性能的对齐程度。

**🔧 技术方法**

核心技术包括：轻量级多头网络、信息探测器（information probe）实现模态分解、VGG 预训练特征提取、对比学习策略、LLM（如 ChatGPT‑4o/5.2）生成环境标签。

**📊 数据集**

主要实验数据集有 LLVIP、RoadScene、TNO、MSRS 以及 3c 系列多任务数据集，用于评估指标一致性与跨数据集泛化能力。

**📈 对比分析**

与传统单独计算指标相比，EvaNet 在所有指标上实现约 1000 倍加速，且在主观感知（DeepIQA、CLIP‑IQA）和下游任务（检测、分割）的一致性得分显著提升，验证了其更高的评估可靠性。

**⚠️ 局限性**

局限性包括：①一致性评估依赖固定的第三方参考模型，无法完全覆盖人类主观偏好；②目前仅针对红外+可见融合，尚未扩展到多曝光等场景；③环境标签由 LLM 自动生成，可能受模型偏差和提示设计影响。

---

## 292. RAGE: A Tightly Coupled Radar-Aided Grip Estimator For Autonomous Race Cars

**arXiv ID:** 2604.02892 | [PDF](https://arxiv.org/pdf/2604.02892v1)

**作者:** Davide Malvezzi `[一作]`, Marko Bertogna `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出了RAGE框架，利用IMU和多雷达实现实时估计车辆速度向量、轮胎侧滑角及前后轴侧向力，完成对赛道高速度、非线性抓地力动态的自适应估计。

**💡 创新点**

创新点在于：①紧耦合运动与轮胎参数的MHE求解，能够同时估计状态和Pacejka系数；②针对雷达时延和多普勒失真实现了时间补偿和去混叠；③在实时约束下实现100 Hz估计并保持低延迟；④使用标准传感器完成全量抓地力估计，显著降低部署成本。

**🔧 技术方法**

技术包括：IMU姿态与加速度补偿、雷达多普勒测量去混叠、单轨运动模型、Pacejka Magic Formula、移动窗口估计(MHE)与CERES优化器、鲁棒损失函数、零速更新(ZUPT)。

**📊 数据集**

数据集主要来自EAV‑24赛车的真实赛道试验（亚斯马里纳F1赛道、A2RL赛事）以及基于多体仿真的高保真模拟，涵盖直线高速、变道、漂移和完整热跑等多种工况。

**📈 对比分析**

对比方法：以Kistler光学测速仪（低通滤波后）作为基准，评估侧向速度、侧向加速度、侧向力等指标。实验显示侧向速度误差≤0.19 m/s，侧滑角误差≤0.15°，侧向力RMSE<500 N，MHE求解平均耗时≈2.3 ms，能稳定在100 Hz。

**⚠️ 局限性**

局限性：单轨简化模型忽略限滑差速、纵向动力耦合与垂直速度影响；对极限角速度与低速状态不够稳健；缺乏公开基准数据做更广泛比较；模型仅针对前后轴侧向力，未能估计单轮状态。

---

## 293. LLM-based Atomic Propositions help weak extractors: Evaluation of a Propositioner for triplet extraction

**arXiv ID:** 2604.02866 | [PDF](https://arxiv.org/pdf/2604.02866v1)

**作者:** Luc Pommeret `[一作]` (Université Paris-Saclay), Sophie Rosset `[通讯]` (Université Paris-Saclay)

**通讯引用:** 9111 | [OpenAlex ID](https://openalex.org/A5100774746)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

构建了一个多语言原子命题生成器（propositioner），并将其嵌入三阶段实体-关系三元组抽取流水线，以提升知识图谱构建的解释性与准确性。

**💡 创新点**

提出将句子拆解为语义最小、不可再分的原子命题作为中间表示，利用递归提示与知识蒸馏技术实现多语言支持，并证明其在实体-关系抽取任务中能显著提升关系召回与整体 F1 分数。

**🔧 技术方法**

使用了多语言轻量级 LLM（如 Qwen-3.6B 的蒸馏版）进行原子命题抽取，结合指令式提示的 LLM（如 Qwen-3、ChatGPT）直接生成三元组；同时与基于依赖解析的传统规则方法做对比。

**📊 数据集**

在 SMiLER、FewRel、DocRED、CaRB 四个公开基准上进行实验，覆盖多语言、开放式与封闭式信息抽取任务，验证方法的跨领域与多语言适用性。

**📈 对比分析**

与直接从原文抽取三元组（Direct）以及仅使用原子命题（Prop）对比；引入 Comb（在实体已知时优先使用原文，否则使用原子命题）和 Union（两者结果合并）进一步提升性能。实验显示：在小模型下，Comb/Union 能提升约 10–20% 的 F1；在大模型（Qwen-3.4B）下，整体 F1 接近或超过 50%，但原子命题的优势更明显，尤其在关系召回上。

**⚠️ 局限性**

1) 对递归原子命题提取算法的有效性尚未全面评估；2) 未使用比 Qwen-3.6B 更大型的 LLM 进行对比，可能低估了大模型的潜力；3) 目前未对生成的原子命题进行人类评价，无法保证所有命题均严格满足“原子”定义；4) 该方法对长文本或复杂句子拆解的深度与阈值选择仍需进一步调优。

---

## 294. How Annotation Trains Annotators: Competence Development in Social Influence Recognition

**arXiv ID:** 2604.02951 | [PDF](https://arxiv.org/pdf/2604.02951v1)

**作者:** Maciej Markiewicz `[一作]` (Wrocław University of Science and Technology), Przemysław Kazienko `[通讯]` (Wrocław University of Science and Technology)

**通讯引用:** 5233 | [OpenAlex ID](https://openalex.org/A5049612210)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对社会影响识别任务中的注释者进行双轮注释实验，考察其专业水平、工作质量以及对LLM训练效果的影响。

**💡 创新点**

将注释视为学习过程，系统评估注释者在主观任务中的能力提升，并将其对模型性能的影响量化。

**🔧 技术方法**

采用Krippendorff's alpha衡量一致性、注释时间分析、NASA‑TLX工作负荷量表、SIR‑SC自评量表、LLM（DeepSeek‑V3.2、Llama‑3.1‑8B‑Instruct）ICL与SFT训练与评估。

**📊 数据集**

自制的1,021条AI生成对话数据集，包含20种社会影响技术、意图、反应、后果等标签；其中150条文本用于Pre/Post对比。

**📈 对比分析**

比较Pre与Post训练集在ICL（0、3、10、30 shot）和SFT中的Jaccard相似度；Post数据在3、10、30 shot ICL下平均提升0.0011、0.0088、0.0149，SFT提升0.0069。

**⚠️ 局限性**

主观任务内一致性偏低，样本量有限导致LLM性能提升有限，且仅评估了少数几种模型与聚合方式，缺乏跨任务验证。

---

## 295. Sample compression schemes for balls in structurally sparse graphs

**arXiv ID:** 2604.02949 | [PDF](https://arxiv.org/pdf/2604.02949v1)

**作者:** Romain Bourneuf `[一作]` (University of Bordeaux), Clément Rambaud `[通讯]` (University of Cote d'Azur)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `fede83ac-7505-405f-ab37-e7284695c47f` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

构造了球（即图中以某点为中心、给定半径的集合）在结构稀疏图（树宽、团宽、顶点覆盖数等）上的样本压缩方案，且方案是**proper**（压缩后返回的球本身就是一个球），并给出了相应的压缩大小上界：树宽 t 的图得到大小 O(t log t)，团宽 t 的图得到大小 O(t log t)，顶点覆盖数 t 的图得到大小 t+4；此外在局部树宽、有限弱着色数、有限退化度等情况也给出了对应的压缩方案。

**💡 创新点**

主要创新在于：
• 提出了在结构稀疏图中构造**proper**球压缩方案的方法，弥补了此前仅得到非 proper（大小为 O(t² log t)）方案的空白；
• 通过利用树分解/ NLC‑分解中的分隔子集以及对距离信息的压缩（仅用 O(t) 个样本点就能描述与分隔相关的所有距离上界/下界），实现了接近线性的压缩大小；
• 证明了上述上界是最优（除对数因子外）并给出了相应的下界构造。

**🔧 技术方法**

技术手段主要包括：
• 树分解（tree decomposition）和 NLC‑分解（与团宽相关）的结构性利用；
• 对分隔子集的编码，使用 φ 这样的函数将分隔子集映射为固定数量（≤3）个树节点，进而把分隔信息压缩到样本的少量元素；
• 对球与分隔子集的距离关系的“上界/下界”描述，使用 r⁺、r⁻ 等函数，只记录与每个分隔点相关的距离极值；
• 组合上述两种编码得到数组样本压缩方案，并证明其能重构出满足样本标签的球。

**📊 数据集**

无实验数据，研究完全基于理论证明和构造。

**📈 对比分析**

与现有的 Moran‑Yehudayoff 方案相比，原方案给出 O(t² log t) 的非 proper 上界，而本文给出的 O(t log t) 的 proper 上界在理论上大幅改进；在特定结构类（如树宽、团宽、顶点覆盖、局部树宽等）中，压缩大小的上界接近已知的下界（至少为 t/5），说明所给方案在理论上已接近最优。

**⚠️ 局限性**

限制：
• 对数因子是否可消除仍未确定；
• 对更一般的 K_t‑无族图（即所有 K_t‑无族图）是否能得到 O(t) 的 proper 方案尚不清楚；
• 对所有稠密图（如团宽大但仍受限于某些参数）的压缩方案尚未给出；
• 在某些参数（如树深、弱着色数）下的最优性仍未证实。

---

## 296. MMTalker: Multiresolution 3D Talking Head Synthesis with Multimodal Feature Fusion

**arXiv ID:** 2604.02941 | [PDF](https://arxiv.org/pdf/2604.02941v1)

**作者:** Bin Liu `[一作]`, Bo Li `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `4de8e9d8-757b-475f-9627-18a445e50202` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出MMTalker，一种基于多分辨率表示与多模态特征融合的语音驱动3D口型动画合成方法

**💡 创新点**

创新点包括：使用网格参数化与非均匀可微采样实现连续细粒度3D面部表示；采用几何引导的图卷积网络与双重跨注意力机制实现语音与面部几何的深度融合；支持多分辨率动画生成并通过眼部损失提升眼部同步；整体实现更高的口型与表情细节同步

**🔧 技术方法**

技术栈涵盖网格参数化、非均匀可微采样、spatio‑temporal vertex encoding、Wav2Vec2.0 语音特征提取、LSTM、残差图卷积网络（RGCN）、双重跨注意力机制（DCAM）、全连接解码器及 Delaunay 三角化

**📊 数据集**

使用VOCASET（480条面部4D扫描序列）和Multiface（多说话人30fps面部扫描）两大公开数据集进行训练与评测

**📈 对比分析**

在VOCASET与Multiface上使用 E_vl、E_ve、FDD 等指标与 FaceFormer、CodeTalker、SelfTalk、PATS 等方法对比，MMTalker 在所有指标上均低于对手，表明口型同步、眼部动作及整体表情细节均优于现有技术

**⚠️ 局限性**

局限性：仅针对单人语音驱动；缺乏多方对话或多种姿态的适应；对强情绪或复杂场景的鲁棒性待提升，未来可加入情绪识别或文本语义融合以增强表现力

---

## 297. Modality-Specific Hierarchical Enhancement for RGB-D Camouflaged Object Detection

**arXiv ID:** 2604.02935 | [PDF](https://arxiv.org/pdf/2604.02935v1)

**作者:** Yuzhen Niu `[一作]` (Fuzhou University), Zhichen Yang `[通讯]` (Fuzhou University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了 MHENet，一个针对 RGB-D 隐蔽目标检测的模态特定层次增强框架，利用纹理和几何信息分别提升 RGB 与深度特征后再进行自适应融合。

**💡 创新点**

创新点在于：① 为 RGB 提供纹理层次增强模块 (THEM) 与为深度提供几何层次增强模块 (GHEM)，通过跨尺度语义一致性实现模态特有特征的强化；② 设计了自适应动态融合模块 (ADFM)，实现空间可变的模态加权融合，显著提升对高相似度背景下的分割效果。

**🔧 技术方法**

采用 PVT 作为骨干网络；构建 THEM、GHEM 两个层次增强模块；利用可学习梯度卷积 (LGConv) 强化几何结构；引入跨尺度语义块、纹理块、几何块；使用 1×1 卷积+全局池化的自适应加权策略实现 ADFM；损失为 BCE+IoU 组合。

**📊 数据集**

在四个公开 COD 基准上进行评测：CAMO、CHAMELEON、COD10K、NC4K。

**📈 对比分析**

与 16 个 SOTA 方法（包括 4 个 RGB-D COD 方法和 12 个 RGB COD 方法）在 S-measure、E-measure、F-measure、MAE 等四项指标上进行对比，MHENet 在所有数据集上均实现了领先或接近领先的性能，显示出显著的提升。

**⚠️ 局限性**

在严重遮挡、边界模糊、深度噪声等极端场景下仍会出现漏检或误检，需要进一步加入背景建模、边缘监督和可靠性-aware 深度处理等改进。

---

## 298. PolyReal: A Benchmark for Real-World Polymer Science Workflows

**arXiv ID:** 2604.02934 | [PDF](https://arxiv.org/pdf/2604.02934v1)

**作者:** Wanhao Liu `[一作]` (University of Science and Technology of China), Yuqiang Li `[通讯]` (Shanghai Artificial Intelligence Laboratory)

**通讯引用:** 1593 | [OpenAlex ID](https://openalex.org/A5055664612)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了 PolyReal 多模态基准，用以评估大语言模型在聚合物实验全生命周期的表现

**💡 创新点**

首次将聚合物实验全流程拆分为五个评测模块，并揭示模型在实际实践任务中的能力缺口

**🔧 技术方法**

采用多模态大语言模型（如 GPT‑5、O3、Gemini、Qwen 等）进行零样本评估，并在部分模块中引入搜索工具对比

**📊 数据集**

使用 545 条真实聚合物实验问答对，涵盖多子领域的图像、谱图及原始仪器数据

**📈 对比分析**

在 15 个主流模型中，封闭源模型（如 O3、GPT‑5）在知识推理任务上表现最好，但在实验安全分析和原始数据提取任务中表现显著下降

**⚠️ 局限性**

存在知识与实践脱节、对原始仪器数据解析不足、缺乏对高分子特性（如链隔离效应）的深度理解等局限

---

## 299. Efficient Logistic Regression with Mixture of Sigmoids

**arXiv ID:** 2604.02920 | [PDF](https://arxiv.org/pdf/2604.02920v1)

**作者:** Federico Di Gennaro `[一作]` (ETH Zürich), Nikita Zhivotovskiy `[通讯]` (UC Berkeley)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `64443552-63e0-44b5-906f-d90fe95c5a1b` `de8d30ba-c289-43a5-b4ec-7b80df73aea2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

本文研究了在线逻辑回归中的指数权重（EW）算法，并提出了使用高斯先验实现的可计算版本；

**💡 创新点**

创新点在于：①在保持最优O(d log(Bn))调度误差的同时，将计算复杂度从之前的O(B¹⁸ n³⁷)降低到仅O(B³ n⁵)；②在线性可分情形下证明了B→∞时EW预测收敛为固体角投票器，并且其模点与硬间隔SVM方向一致，揭示了算法的几何自适应性；

**🔧 技术方法**

技术手段包括：高斯先验下的指数权重、MALA（Metropolis‑Adjusted Langevin）随机采样、温度桥接（bridged warm‑start）实现高效后验近似、Monte Carlo 期望估计以及平滑操作以控制log‑loss；

**📊 数据集**

实验使用了三组 LIBSVM 二分类数据集（原文中未明示具体名称），只保留前2000个样本并按顺序处理；

**📈 对比分析**

与 OGD、ONS、AIOLI 等传统在线逻辑回归方法对比，实验显示 EW 在 B 较大时性能略优于 AIOLI，整体表现与 AIOLI 相当甚至更好；在 B 较小时 AIOLI 仍表现更佳；

**⚠️ 局限性**

局限性包括：计算复杂度仍随 B 增大而显著增长；实现需要多条 MALA 链，实际成本可能高于理论上限；B 对采样步骤的依赖尚未完全消除；多类别扩展仍待研究。

---

## 300. SentiAvatar: Towards Expressive and Interactive Digital Humans

**arXiv ID:** 2604.02908 | [PDF](https://arxiv.org/pdf/2604.02908v1)

**作者:** Chuhao Jin `[一作]` (Renmin University of China), Ruihua Song `[通讯]` (Renmin University of China)

**通讯引用:** 2722 | [OpenAlex ID](https://openalex.org/A5101505570)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出 SentiAvatar 框架，实时生成具备语音、手势、表情同步的 3D 数字人 SuSu，构建了多模态对话数据集 SuSuInterActs 并实现了高效的交互式动画生成。

**💡 创新点**

创新点：① 创建 21K 片段、37h 的多模态中文对话动作数据集 SuSuInterActs；② 预训练 Motion Foundation Model，涵盖 200K+ 动作序列，提升动作先验；③ 采用 LLM + Infill Transformer 的计划-填充两阶段架构，分离语义规划与帧级音频同步，显著提高语义一致性与节奏同步。

**🔧 技术方法**

技术手段：R‑VQVAE 离散编码；HuBERT 语音特征提取；Qwen‑0.5B LLM 进行键帧规划；音频感知 Infill Transformer 填充细节；面部专用 R‑VQVAE；Motion Foundation Model 预训练；多模态并行推理。

**📊 数据集**

使用的数据集：SuSuInterActs（21K 片段、37h 中文对话，含语音、文本、全身动作、面部表情）；BEATv2（英文共语音-动作对齐 benchmark）；以及 200K+ 多来源动作序列（EmbodyAI、SnapMoGen、Motion‑X、Hunyuan Distill）。

**📈 对比分析**

与多种基线比较：音频+文本（EMAGE、AT2M‑GPT）、文本（T2M‑GPT、MoMask、HunYuan‑Motion）、音频（EMAGE）。在 SuSuInterActs 上 SentiAvatar 取得 R@1 43.64%（↑26.3%）、FID 8.91（↓34.8%）、ESD 0.456s；在 BEATv2 上 FGD 4.94（↓6.8%）、BC 8.08（↑1.3%），显示显著优于现有方法。

**⚠️ 局限性**

局限性：目前仅针对单一角色，面部表情生成仅依赖音频未结合语义规划；对多角色、多语言对话的泛化仍待验证；长时对话的持续流式生成尚有限；键帧间隔与音频同步仍有提升空间。

---

## 301. Information-Regularized Constrained Inversion for Stable Avatar Editing from Sparse Supervision

**arXiv ID:** 2604.02883 | [PDF](https://arxiv.org/pdf/2604.02883v1)

**作者:** Zhenxiao Liang `[一作]` (University of Texas at Austin), Qixing Huang `[通讯]` (University of Texas at Austin)

**通讯引用:** 16712 | [OpenAlex ID](https://openalex.org/A5056540212)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种在稀疏编辑关键帧下通过约束性逆推实现人类动画化3D头像稳定编辑的方法。

**💡 创新点**

创新点在于把编辑过程视为结构化潜在空间的条件逆推，并在逆推过程中通过信息矩阵的谱条件（log‑det）动态分配关键帧权重，使编辑方向更稳定、避免身份泄露与时序抖动。

**🔧 技术方法**

核心技术包括低维残差UV解码器、局部线性化求得的雅可比/信息矩阵、条件正则化的log‑det目标、Hessian‑vector产品实现高效矩阵求解以及基于重线性化的迭代优化。

**📊 数据集**

使用合成的ZJU‑MoCap/3DGS‑Avatar数据集（生成伪真编辑）和真实单目视频进行实验，同时构建配对编辑样本用于训练残差UV解码器。

**📈 对比分析**

与3DGS‑Avatar、DGE、IDOL等基线比较，评估PSNR、LPIPS、身份泄露和时间一致性指标，实验表明在稀疏编辑场景下该方法在编辑精度、身份保持和时序稳定性上均优于对比方法。

**⚠️ 局限性**

局限性包括对大幅编辑的线性化假设易失效、对编辑目标和掩码噪声敏感、无法处理极度非刚性的衣物动态以及缺乏自适应关键帧采集与非线性编辑子空间。

---

## 302. One Model to Translate Them All? A Journey to Mount Doom for Multilingual Model Merging

**arXiv ID:** 2604.02881 | [PDF](https://arxiv.org/pdf/2604.02881v1)

**作者:** Baban Gain `[一作]` (Indian Institute of Technology Patna), Trilok Nath Singh `[通讯]` (Indian Institute of Technology Patna)

**通讯引用:** 459 | [OpenAlex ID](https://openalex.org/A5103411117)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对多语言机器翻译中独立微调模型的权重空间合并效果进行系统研究，揭示其失效原因。

**💡 创新点**

首次从神经元选择性、层级对齐与代表性相似度等角度揭示多语言微调导致的上层生成子空间失配，解释合并失败。

**🔧 技术方法**

采用权重空间平均、TIES、DARE、SCE‑Merging 等合并方法，结合 span‑conditioned 神经元激活分析、Neuron Usage Alignment 与 Centered Kernel Alignment 等技术。

**📊 数据集**

使用 Samanantar 语料库中四对 Indic–English（Hindi、Bengali、Tamil、Telugu ↔ English）的百万级双语对。

**📈 对比分析**

与基线预训练模型和单语微调上界对比，发现在 Many→One 合并可提升跨语覆盖但保留率低，One→Many 与双向合并则严重性能衰退。

**⚠️ 局限性**

局限在于仅考察 MT 任务、固定语言组合，且未探索更深层次的对齐或非线性合并方法。

---

## 303. SPG: Sparse-Projected Guides with Sparse Autoencoders for Zero-Shot Anomaly Detection

**arXiv ID:** 2604.02871 | [PDF](https://arxiv.org/pdf/2604.02871v1)

**作者:** Tomoyasu Nanaumi `[一作]` (Yachiyo Engineering Co Ltd), Takayoshi Yamashita `[通讯]` (Chubu University)

**通讯引用:** 4118 | [OpenAlex ID](https://openalex.org/A5062009349)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `9ce7179e-700c-4310-ac2b-91df50ded46e` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种无提示的零样本异常检测与分割框架SPG，利用稀疏自编码器在冻结的视觉特征空间中构造正负引导向量；

**💡 创新点**

创新点在于用稀疏编码生成引导向量取代传统提示，提升可解释性并兼容任何冻结视觉背骨；

**🔧 技术方法**

核心技术包括冻结的视觉编码器（如DINOv3、OpenCLIP），TopK稀疏自编码器（SAE）以及基于引导向量的余弦相似度分数与温度软max；

**📊 数据集**

在MVTec AD与VisA两个工业视觉异常数据集上进行交叉数据集零样本实验；

**📈 对比分析**

与现有CLIP/ DINO基准方法相比，SPG在图像级检测上保持竞争力，在像素级分割上取得最高的AUROC（MVTec AD 92.3% / VisA 96.0%），表现尤为突出；

**⚠️ 局限性**

局限在于需对SAE字典大小与TopK参数进行经验调优，且在某些提示自适应方法的像素级AP上略逊一筹。

---

## 304. Multi-Turn Reinforcement Learning for Tool-Calling Agents with Iterative Reward Calibration

**arXiv ID:** 2604.02869 | [PDF](https://arxiv.org/pdf/2604.02869v1)

**作者:** Wachiravit Modecrua `[一作]` (Amity Research and Application Center), Touchapon Kraisingkorn `[通讯]` (Amity Research and Application Center)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

训练多轮工具调用代理，解决稀疏奖励和信用分配问题

**💡 创新点**

提出迭代奖励校准方法（IRC）与MT‑GRPO+GTPO混合优势，消除优势方向误配

**🔧 技术方法**

结合MT‑GRPO、GTPO、group‑normalization、discounted returns与深度奖励校准

**📊 数据集**

使用Tau‑Bench航空客服基准（v1训练、v2测试）及LLM用户模拟器

**📈 对比分析**

在Tau‑Bench航空任务上，4B Qwen3.5模型从63.8%提升至66.7%，30.5B MoE模型从58.0%提升至69.5%，超越GPT‑4.1/GPT‑4o并接近Claude Sonnet 4.5

**⚠️ 局限性**

仅在航空领域评估，跨域泛化有限；用户模拟器与真实用户存在分布偏差；GTPO超参数需手动调优

---

## 305. EMS: Multi-Agent Voting via Efficient Majority-then-Stopping

**arXiv ID:** 2604.02863 | [PDF](https://arxiv.org/pdf/2604.02863v1)

**作者:** Yiqing Liu `[一作]` (University of Science and Technology of China), Yongdong Zhang `[通讯]` (University of Science and Technology of China)

**通讯引用:** 34998 | [OpenAlex ID](https://openalex.org/A5046305086)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种多代理系统中的Efficient Majority‑then‑Stopping（EMS）框架，利用可靠性感知的代理调度实现多数投票的早停，减少冗余推理。

**💡 创新点**

将多数投票转化为可靠性感知的代理调度问题，结合Agent Confidence Modeling（历史可靠性与语义相似度）、Adaptive Incremental Voting（动态早停）和Individual Confidence Updating（可靠性在线更新），实现准确率不下降的情况下显著降低调用次数。

**🔧 技术方法**

使用可靠性建模、语义相似度计算（句子编码器）、分层投票策略、BFT启发的多数阈值、动态投票顺序生成与早停机制以及在线可靠性更新。

**📊 数据集**

六大基准：数学推理类的AQuA、Math500、GSM8K；通用知识类的MMLU、GPQA Diamond、CommonsenseQA。

**📈 对比分析**

与单模型、Self‑Consistency、Simple Majority Voting、加权投票等基线比较，EMS在六个基准上保持与Simple MV相近的准确率（≈86%），同时将平均调用代理数从9降至约6，提升约30%推理效率。

**⚠️ 局限性**

对可靠性排序的质量高度依赖；当前仅使用简单的历史和语义信号，若排序不佳会影响效率；在极难或需要更多代理的任务中，早停效果有限；未来需探索更复杂的调度策略和跨任务适应。

---

## 306. AgentHazard: A Benchmark for Evaluating Harmful Behavior in Computer-Use Agents

**arXiv ID:** 2604.02947 | [PDF](https://arxiv.org/pdf/2604.02947v1)

**作者:** Yunhao Feng `[一作]` (Alibaba Group), Yanming Guo `[通讯]` (Hunan Institute of Advanced Technology)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文构建了 AgentHazard 这一基于执行轨迹的计算机使用代理安全评测基准，涵盖 2,653 条多步恶意任务实例；

**💡 创新点**

创新点在于提出从风险类别与攻击策略两维度的分类体系，并通过执行筛选和人工审核生成可执行的多步攻击实例，首次关注代理层面的累积危害；

**🔧 技术方法**

技术包括基于大型语言模型的生成式实例构造、沙箱化执行环境（Qwen3-Coder、Claude Code、OpenClaw、IFlow）以及 LLM-as-Judge（Gemini‑3）和多种 guard 模型（Llama‑Guard‑3‑8B、Qwen3Guard 系列）进行轨迹评估；

**📊 数据集**

使用的主要数据集为 AgentHazard 自建数据集，内部包含 10 个风险类别与 10 种攻击策略的实例；

**📈 对比分析**

对比方法采用完整轨迹的攻击成功率（ASR）和平均危害度量，实验显示代理在不同框架与模型下的 ASR 可高达 82.9%，并证明现有 guard 模型仅在 27% 左右检测到恶意意图；

**⚠️ 局限性**

局限性包括数据分布不均、对某些风险类别实例稀缺、guard 模型在单步评估时检测率低，以及评测仅覆盖三大框架，未必泛化到所有代理系统。

---

## 307. MSAO: Adaptive Modality Sparsity-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference

**arXiv ID:** 2604.02945 | [PDF](https://arxiv.org/pdf/2604.02945v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62`

---

## 308. Explainable Machine Learning Reveals 12-Fold Ucp1 Upregulation and Thermogenic Reprogramming in Female Mouse White Adipose Tissue After 37 Days of Microgravity: First AI/ML Analysis of NASA OSD-970

**arXiv ID:** 2604.02942 | [PDF](https://arxiv.org/pdf/2604.02942v1)

**作者:** Md. Rashadul Islam `[一作]` `[通讯]` (Daffodil International University), Md. Rashadul Islam (Daffodil International University)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

重新利用NASA OSD-970 微重力实验的 RT‑qPCR 数据，采用可解释机器学习分析白色脂肪组织基因表达，揭示雌性小鼠白色脂肪组织的热生成功能重编程。

**💡 创新点**

首次将可解释 AI 与留一交叉验证相结合，对微重力导致的白色脂肪组织热生成功能进行高维基因预测与解释，发现 Ucp1 12 倍上调与 Angpt2‑Jun‑Irs2‑Klf2 抑制网络。

**🔧 技术方法**

差异表达分析、随机森林、XGBoost、梯度提升、SVM、逻辑回归、KNN、PyTorch 神经网络以及 SHAP 可解释性分析和共价网络关联。

**📊 数据集**

NASA OSD-970（GLDS-790）— 16 只雌性 C57BL/6J 小鼠的 89 个基因 RT‑qPCR Ct 值（8 飞行、8 地面）。

**📈 对比分析**

采用留一交叉验证评估七类分类器；随机森林与逻辑回归在 20 基因集上 AUC 0.922，XGBoost 与梯度提升在全部基因上 AUC 0.875，性能显著优于随机。

**⚠️ 局限性**

样本量极小、基因面板有限、单时间点、未验证蛋白或代谢水平、地面对照条件不完全一致。

---

## 309. Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms

**arXiv ID:** 2604.02927 | [PDF](https://arxiv.org/pdf/2604.02927v1)

**作者:** Andreas Boltres `[一作]` (Karlsruhe Institute of Technology), Gerhard Neumann `[通讯]` (Karlsruhe Institute of Technology)

**通讯引用:** 10445 | [OpenAlex ID](https://openalex.org/A5110467801)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `3f18e8e3-0266-457c-8567-9039b6d2394d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出一种面向实时网络监测的神经路由框架，训练并评估可在毫秒级响应流量突发的图神经网络路由算法 LOGGIA。

**💡 创新点**

创新点包括：① 将通信与推理延迟显式纳入闭环控制问题；② LOGGIA 通过在对数空间预测链路权重、预训练 Imitation Learning 与最大熵强化学习相结合的两阶段架构，实现更稳健的学习；③ 通过分布式观察与决策证明完全分布式部署在延迟敏感场景下最优。

**🔧 技术方法**

使用技术包括：ns‑3 网络模拟器+ns3‑ai 共享内存接口；图神经网络（mpn）进行属性图编码；最大熵 PPO（含温度自适应与早停）以及多智能体 M‑PPO；预训练 Imitation Learning；Dijkstra 进行最短路径求解。

**📊 数据集**

数据集涵盖多种网络拓扑（mini5、B4、GEANT、nx‑XS、nx‑S 等）以及合成的 80% TCP + 20% UDP 流量序列，模拟真实数据中心工作负载。

**📈 对比分析**

与传统静态最短路径协议（ospf、eigrp、rip）以及现有神经基线（FieldLines、M‑Slim、MAGNNETO）相比，LOGGIA 在考虑延迟的部署模式下持续击败所有基线，取得更高的吞吐量、较低的方差，并能在从 5 节点到 100 节点的拓扑中保持良好泛化和可扩展性。

**⚠️ 局限性**

局限性包括：仅支持单路径路由；对网络状态与动作传播的延迟模型假设为无带宽限制的离线通道；推理时间随网络规模呈指数增长，尤其在全局最短路径模式下；未对链路/节点故障等鲁棒性进行深入评估。

---

## 310. Corporations Constitute Intelligence

**arXiv ID:** 2604.02912 | [PDF](https://arxiv.org/pdf/2604.02912v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 311. A Multi-head-based architecture for effective morphological tagging in Russian with open dictionary

**arXiv ID:** 2604.02926 | [PDF](https://arxiv.org/pdf/2604.02926v1)

**作者:** K. Skibin `[一作]` (National Research Tomsk State University), S. Suschenko `[通讯]` (National Research Tomsk State University)

**通讯引用:** 22 | [OpenAlex ID](https://openalex.org/A5068179145)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出并实现了基于多头注意力的开字典俄语形态标注架构。

**💡 创新点**

采用子词分割+多头注意力无RNN，无预训练，支持开放词典并实现高准确率。

**🔧 技术方法**

多头注意力（MHA）、RoPE位置编码、子词BPE、前馈分类网络以及PyTorch实现。

**📊 数据集**

合并的UD SynTagRus与Taiga两大俄语标注语料集。

**📈 对比分析**

与BERT、RNNMorph等对比，单类别准确率达98–99%，多类别整体准确率≈95%；模型参数48M，训练仅需8–12小时。

**⚠️ 局限性**

对全属性完整预测仍仅约90%；性能受待测属性集合影响，需进一步调优与实验。

---

## 312. GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes

**arXiv ID:** 2604.02915 | [PDF](https://arxiv.org/pdf/2604.02915v1)

**作者:** Mijeong Kim `[一作]` (Seoul National University), Bohyung Han `[通讯]` (Seoul National University)

**通讯引用:** 19073 | [OpenAlex ID](https://openalex.org/A5006594639)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

将高斯过程（GP）整合进4D Gaussian Splatting（4DGS），实现动态场景的概率化重建与不确定性量化。

**💡 创新点**

通过引入时空分离的Matérn-周期核和变分GP推断，首次在4DGS框架中提供运动不确定性、未来运动预测与自适应运动先验。

**🔧 技术方法**

使用变分高斯过程、复合空间-时间核、变分推断、GP-GS交替优化、蒙特卡洛不确定性估计等技术。

**📊 数据集**

在DyCheck、DAVIS以及自建的手持视频数据集上进行实验。

**📈 对比分析**

与多种基线（如4DGS、SoM、HyperNeRF等）进行对比，实验显示GP‑4DGS在mPSNR/mSSIM/mLPIPS等指标上获得更高分数，尤其在稀疏视角下显著提升。

**⚠️ 局限性**

局限性包括对超大规模点云的推断仍受变分GP的近似影响，时间推断受周期核假设限制，且需要手工设定超参数。

---

## 313. Split and Conquer Partial Deepfake Speech

**arXiv ID:** 2604.02913 | [PDF](https://arxiv.org/pdf/2604.02913v1)

**作者:** Inbal Rimon `[一作]` (Ben Gurion University), Haim Permuter `[通讯]` (Ben Gurion University)

**关键词:** `fb2d1ce9-128d-478c-ade6-0079bcd4d876` `3855fcda-48ef-4070-a15e-803cd5c84d83` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `b88c6eac-d57a-4623-a604-1f401f3eb268` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一个 Split‑and‑Conquer 框架，先对语音进行边界检测，再对每个检测到的段落进行真假判定，实现对部分深度伪造语音的检测与定位。

**💡 创新点**

创新点在于将时间定位与真实性分类解耦，采用基于反射的多长度训练、动态阈值边界提取以及多模型融合，以显著提升鲁棒性与定位精度。

**🔧 技术方法**

使用了 wav2vec2.0 XLSR（XLSR53/XLSR128）和 log‑magnitude spectrogram 作为前端，后接 ResNet34 分类头，并结合 MaskSpec/MaskFeature 增强、固定长度反射映射、以及多长度和多模态融合技术。

**📊 数据集**

在 PartialSpoof benchmark（含多时间分辨率标签）和 Half‑Truth (HAD) 数据集上进行训练与评估。

**📈 对比分析**

与最新方法 MRM、IFBDN、CFPRF 在 PartialSpoof 与 HAD 上对比，EER 下降至 6.55%（PartialSpoof）和 0.01%（HAD），并在 AP/AR 指标上均优于对手，展示出显著性能提升。

**⚠️ 局限性**

主要局限包括：边界预测误差会在后续阶段累积导致误判；假设伪造为块状段落，难以捕捉渐进或极细微的改动；在极端噪声或语种差异大时仍需进一步适配。

---

## 314. RayMamba: Ray-Aligned Serialization for Long-Range 3D Object Detection

**arXiv ID:** 2604.02903 | [PDF](https://arxiv.org/pdf/2604.02903v1)

**作者:** Cheng Lu `[一作]` (Nanjing University of Science and Technology), Jian Yang `[通讯]` (Nanjing University of Science and Technology)

**通讯引用:** 45201 | [OpenAlex ID](https://openalex.org/A5100726984)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出RayMamba模块，通过雷射对齐序列化提升稀疏远程LiDAR点云的上下文建模，作为轻量化插件集成到体素基检测器中。

**💡 创新点**

设计基于传感器几何的扇区划分与层次化排序策略，替代传统Hilbert/Z-curve序列化，保持方向连贯性与遮挡相关上下文。

**🔧 技术方法**

采用State Space Model（Mamba）序列化建模，结合Ray‑Aligned Serialization与SectorMamba3D实现扇区级别的序列建模，并与稀疏卷积网络无缝融合。

**📊 数据集**

在nuScenes与Argoverse 2上进行验证，分别对LiDAR‑only（CenterPoint）、多模态（MV2DFusion）和VoxelNeXt进行实验。

**📈 对比分析**

与基线及HilbertMamba对比，在40–50 m远程区间提升至2.49 mAP/1.59 NDS，总体mAP提升0.4–0.5点，显著提升远程目标检测。

**⚠️ 局限性**

扇区序列建模是按扇区串行执行，扇区角度越小导致扇区数增多，推理效率下降；在极大视野下仍受限于扇区化的序列化方法。

---

## 315. Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework

**arXiv ID:** 2604.02877 | [PDF](https://arxiv.org/pdf/2604.02877v1)

**作者:** Yu Zhu `[一作]` (Chinese University of Hong Kong), Pheng-Ann Heng `[通讯]` (Chinese University of Hong Kong)

**通讯引用:** 54416 | [OpenAlex ID](https://openalex.org/A5032708386)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `e15e3743-5ee0-4d5f-813d-d146868082fc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `3f18e8e3-0266-457c-8567-9039b6d2394d` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出自我反思层次化提示框架，用于类增量外科仪器分割，实现正向与逆向知识迁移。

**💡 创新点**

创新点包括：① 层次化提示解析树将共享、部分共享与独特提示结构化，促成正向迁移；② 自我反思细化策略利用有向加权图传播，提升旧知识同时避免灾难性遗忘；③ 框架兼容CNN和Transformer基础模型。

**🔧 技术方法**

采用冻结预训练模型（SAM或DeepLabv3+）的提示调优，构建层次化提示树，使用图神经网络实现自我反思细化，并通过BWT/FWT评估迁移效果。

**📊 数据集**

使用EndoVis 2017/2018（肾切除）和CholecSeg8k/M2CAI-Seg（胆囊切除）两组公开外科视频数据集。

**📈 对比分析**

与Ind-T、Joint-T、Seq-F、SI、LWF、ILT、PLOP、MiB、RASD、RCIL、EWF、LISM、CoinSeg、CAT-SD等现有CIL方法对比，实验表明在两基准上均实现显著提升，BWT/FWT指标均位于最高水平，几乎逼近联合训练上限。

**⚠️ 局限性**

局限性包括：需先验的仪器部件划分，若部件信息缺失或不完整时迁移效果可能受限；对极少样本情况的鲁棒性尚未充分验证；虽然计算开销较低，但仍需额外提示参数和图神经网络推理。

---

## 316. Analysis of Optimality of Large Language Models on Planning Problems

**arXiv ID:** 2604.02910 | [PDF](https://arxiv.org/pdf/2604.02910v1)

**作者:** Bernd Bohnet `[一作]` (Google DeepMind), Noah Fiedel `[通讯]` (Google DeepMind)

**通讯引用:** 2687 | [OpenAlex ID](https://openalex.org/A5027693232)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究大型语言模型（LLM）在自动规划中的推理能力，尤其是它们在以Blocksworld为例的P*图结构上的规划最优性和结构边界。

**💡 创新点**

提出将规划任务映射为P*（Path‑Star）图结构的框架，并构建多维度的P*任务体系（深度、宽度、目标密度），以系统评估LLM的最优性与可扩展性。

**🔧 技术方法**

采用最小提示的单示例推理（one‑shot ICL）和思考（chain‑of‑thought）推理，利用Gemini 3.0 Pro的思考标记计数分析推理成本，配合VAL验证器验证计划合法性。

**📊 数据集**

使用人工生成的基于P*结构的合成数据集，涵盖四类难度梯度：高塔、目标堆、交错堆、全局挑战，并在Blocksworld与图重写两种表达形式下进行实验。

**📈 对比分析**

与传统搜索规划器LAMA‑2011（趋于满足）和A*‑LM‑Cut（最优）进行对比，评估成功率、最优性差距以及推理令牌与计划长度的线性关系；实验显示Gemini 3.0 Pro在数千块级任务中保持零最优性差距，且推理令牌呈约47 tokens/步骤的线性增长，明显优于传统规划器。

**⚠️ 局限性**

限制在受控目标布置（每塔最多1或2个目标）和单结构（P*）上，未覆盖任意目标组合、非P*结构或更大规模任务，未来工作需检验模型在更广泛规划域和多目标交互下的鲁棒性。

---

## 317. UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting

**arXiv ID:** 2604.02905 | [PDF](https://arxiv.org/pdf/2604.02905v1)

**作者:** Geonuk Kim `[一作]` (LG Energy Solution), Junho Yim `[通讯]` (LG Energy Solution)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `729e5870-4135-47f5-97f2-e3974d07b5dc` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了一种名为UniSpector的可视化提示框架，用于开放集工业缺陷检测与分割；

**💡 创新点**

创新点在于通过空间-频谱提示编码器（SSPE）提取方向不变的细粒度特征，并通过对比提示编码器（CPE）构建语义有序的角度空间，再配合提示引导查询选择（PQS）实现自适应查询；

**🔧 技术方法**

采用双域（空间+频谱）特征融合、对比学习（ArcFace角度损失）、DETR架构和可微分Gumbel-Softmax查询选择；

**📊 数据集**

使用InsA基准数据集，整合7个公开工业缺陷数据集（GC10、MagneticTile、Real-IAD、MVTec、3CAD、VISION、VisA）；

**📈 对比分析**

与GroundingDINO、YOLO-World、SEEM、SegGPT、SINE、DINOv、YOLOE等视觉提示与视觉定位方法对比，在InsA上实现AP50^b提升约19.7%、AP50^m提升约15.8%，在各子数据集均显著优于基线；

**⚠️ 局限性**

局限在于跨域性能仍受光照、纹理等成像偏差影响，对域不变性的鲁棒性需进一步提升，且需提供1-3个示例提示，虽数据需求低但在极少样本场景下可能受限。

---

## 318. Extracting Money Laundering Transactions from Quasi-Temporal Graph Representation

**arXiv ID:** 2604.02899 | [PDF](https://arxiv.org/pdf/2604.02899v1)

**作者:** Haseeb Tariq `[一作]` (ING Bank), Marwan Hassani `[通讯]` (Eindhoven University of Technology)

**通讯引用:** 1082 | [OpenAlex ID](https://openalex.org/A5001473233)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

提出了一个基于图特征的轻量级监督学习框架 ExStraQt，用于检测金融交易中的洗钱行为。

**💡 创新点**

创新点在于设计了一套可扩展、易解释的图特征集合，并通过并行化实现低资源需求，同时在真实与合成数据上均优于现有 GNN/Transformer 等模型。

**🔧 技术方法**

采用图结构分析、社区检测（Leiden 与随机游走）、流量特征提取、异常分数（Isolation Forest）及 XGBoost 等机器学习方法，并使用 PySpark 并行化实现。

**📊 数据集**

使用 IBM Synthetic Money Laundering 数据集（多种规模和风险水平）和公开的以太坊钓鱼交易数据集。

**📈 对比分析**

在 F1 分数上，相比 GFP、MultiGNN、FraudGT 等 SOTA，ExStraQt 在大多数数据集上提升约 8%–59%，尤其在真实 ETH 数据上提升约 31%。

**⚠️ 局限性**

主要限制包括对极端稀疏/不平衡数据的适应性不足，缺乏在线实时训练能力，以及对更长时间依赖和隐含账户所有权的捕捉仍待改进。

---

## 319. CrossWeaver: Cross-modal Weaving for Arbitrary-Modality Semantic Segmentation

**arXiv ID:** 2604.02948 | [PDF](https://arxiv.org/pdf/2604.02948v1)

**作者:** Zelin Zhang `[一作]` (University of Sydney), Chuanzhi Xu `[通讯]` (University of Sydney)

**通讯引用:** 16 | [OpenAlex ID](https://openalex.org/A5082895857)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种可处理任意多模态输入的语义分割框架CrossWeaver；

**💡 创新点**

核心创新在于Modality Interaction Block (MIB) 的可靠性感知跨模态交互与Seam-Aligned Fusion (SAF) 的边界对齐融合；

**🔧 技术方法**

利用共享层次化Transformer编码器、可靠度估计、跨尺度注意力、一致性过滤与轻量级深度卷积混合等技术；

**📊 数据集**

在MCubeS（RGB+NIR+DoLP+AoLP）和DeLiVER（RGB+Depth+Event+LiDAR）两大多模态分割基准上进行实验；

**📈 对比分析**

与CMNeXt、MMSFormer、StitchFusion等先进方法对比，CrossWeaver在所有模态组合下均实现mIoU领先，且在缺失模态时仍保持高鲁棒性；

**⚠️ 局限性**

主要局限包括对更大规模backbone的扩展性尚待验证，以及在动态或实时多模态场景中的适配性需进一步研究。

---

## 320. Toward an Artificial General Teacher: Procedural Geometry Data Generation and Visual Grounding with Vision-Language Models

**arXiv ID:** 2604.02893 | [PDF](https://arxiv.org/pdf/2604.02893v1)

**作者:** Hai Nguyen-Truong `[一作]` (Freyavoice), Tunga Bayrak `[通讯]` (Freyavoice)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `729e5870-4135-47f5-97f2-e3974d07b5dc` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究了几何教学中的视觉解释，将其形式化为指代图像分割任务，并构建了完全自动化的数据引擎和模型微调流程。

**💡 创新点**

创新点包括：①使用约束式几何求解与 TikZ 渲染自动生成 200k+ 合成几何图形及像素级掩码；②提出缓冲 IoU（BIoU）评价指标，更贴合薄线结构的定位质量；③对现有 VLM 进行领域专门的 LoRA 微调。

**🔧 技术方法**

采用的技术有：约束式几何求解、TikZ 矢量渲染、掩码提取、基于多边形坐标的序列生成、LoRA 参数高效微调、形状拓扑与几何约束的后处理。

**📊 数据集**

使用的数据集为自生成的 200k+ 合成几何图形（约 50k 训练/验证/测试三元组）和公开自然图像基准（RefCOCO、RefCOCO+ 等）用于零样本对比。

**📈 对比分析**

通过零样本与微调对比实验，零样本模型 IoU<1%，微调后 Florence‑2 达到 49% IoU、85% BIoU，明显优于 Qwen‑VL 微调（10% IoU、42% BIoU），验证了领域适配的重要性。

**⚠️ 局限性**

局限性包括：仅使用合成数据，缺乏对真实手绘或扫描图像的适应；仅覆盖二维四边形，未扩展到 n 边形、3D 或隐式几何；多元素分割与几何约束的完整性仍待提升。

---

## 321. Progressive Video Condensation with MLLM Agent for Long-form Video Understanding

**arXiv ID:** 2604.02891 | [PDF](https://arxiv.org/pdf/2604.02891v1)

**作者:** Yufei Yin `[一作]` (Hangzhou Dianzi University), Zhou Yu `[通讯]` (Hangzhou Dianzi University)

**通讯引用:** 6115 | [OpenAlex ID](https://openalex.org/A5061025828)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种Progressive Video Condensation Agent (ProVCA)，通过逐层分段、片段、关键帧选择将长视频压缩成少量关键帧，再交由多模态大型语言模型进行问答推理；

**💡 创新点**

创新点在于将多模态LLM的推理能力与自上而下的层次化视频稀疏抽取相结合，显著减少输入帧数同时保持甚至提升推理准确率；

**🔧 技术方法**

核心技术包括：多模态LLM推理、基于语义相似度的片段聚类、MCLM驱动的关键帧精炼以及链式思考提示；

**📊 数据集**

使用三大长视频问答基准：NExT-QA、EgoSchema 与 IntentQA 进行评测；

**📈 对比分析**

与现有训练自由方法（如VideoTree、LVNet）及大规模视频语言模型对比，ProVCA 在 NExT-QA、EgoSchema、IntentQA 上分别取得 80.5%、69.3% 与 77.7% 的零样本准确率，帧数平均降至 4–7 之间；

**⚠️ 局限性**

局限性包括对LLM计算资源依赖较高、在极长视频中仍需多轮抽取步骤、以及对极细粒度视觉细节的捕捉可能受限。

---

## 322. InstructTable: Improving Table Structure Recognition Through Instructions

**arXiv ID:** 2604.02880 | [PDF](https://arxiv.org/pdf/2604.02880v1)

**作者:** Boming Chen `[一作]` (Meituan), Pengfei Yan `[通讯]` (Meituan)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `67630363-6be0-4f51-ab05-7198250671a5` `edb9d762-f411-4838-a852-f2d638b018db` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出基于指令的多阶段训练表格结构识别框架InstructTable，并开发无模板表格合成方法TME以及900张复杂长表格的BCDSat基准。

**💡 创新点**

通过指令预训练平衡视觉与语义信息，使用TME生成大规模真实感表格数据，显著提升复杂表格的识别效果。

**🔧 技术方法**

采用TableResNetExtra视觉编码器、BERT文本编码器、Transformer跨模态注意力解码器以及多任务交叉熵+L1损失。

**📊 数据集**

在PubTabNet、FinTabNet、MUSTARD公开数据集以及自研的BCDSat数据集上进行训练与评测。

**📈 对比分析**

与多种视觉中心、视觉‑语言模型以及先前SOTA方法对比，InstructTable在FinTabNet、PubTabNet、MUSTARD及BCDSat上均实现了最高的TEDS/ S‑TEDS分数，尤其在复杂长表格上提升约2–3%。

**⚠️ 局限性**

仍受长表格多样性、跨语言鲁棒性与模型参数规模的限制，部分极长或高复杂度表格的识别仍存在误差。

---

## 323. An Asynchronous Two-Speed Kalman Filter for Real-Time UUV Cooperative Navigation Under Acoustic Delays

**arXiv ID:** 2604.02878 | [PDF](https://arxiv.org/pdf/2604.02878v1)

**作者:** Shuyue Li `[一作]` (Xi'an Jiaotong-Liverpool University), Xiaohui Qin `[通讯]` (Jiangsu JITRI Tsingunited Intelligent Control Technology Co., Ltd.)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `51c0528b-f690-4182-ae60-bb5f046c276c` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种异步双速卡尔曼滤波器（TSKF），在水声通信延迟下实现UUV协同导航的实时状态估计

**💡 创新点**

核心创新在于将预测与校正拆分为高速线程和低速异步线程，并引入变分历史蒸馏（VHD）机制，将延迟测量的校正高效投射至当前时刻，避免大规模矩阵重算

**🔧 技术方法**

利用高频IMU/DVL测量的Gaussian Process残差补偿、循环状态缓冲区、VHD投影、传统EKF/UKF结构以及Aqua‑Sim FG仿真平台

**📊 数据集**

使用基于Aqua‑Sim FG的高保真海底声学网络仿真，生成5–30 s的动态延迟和15 %数据丢包，进行500次蒙特卡洛仿真

**📈 对比分析**

与标准EKF/UKF、增广EKF、FGO进行对比；在30 s延迟下TSKF RMSE≈1.92 m、执行时间≈0.0027 ms/步，几乎等同FGO精度但快50×，并避免增广EKF内存溢出

**⚠️ 局限性**

对极端大延迟下的多UUV网络（非单体）以及非高斯噪声环境的适用性未深入验证，且VHD投影在高度非线性动力学下可能需进一步改进

---

## 324. Toward an Operational GNN-Based Multimesh Surrogate for Fast Flood Forecasting

**arXiv ID:** 2604.02876 | [PDF](https://arxiv.org/pdf/2604.02876v1)

**作者:** Valentin Mercier `[一作]` (Université de Toulouse), Gwenaël Chevallet `[通讯]` (BRL Ingénierie)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `14d48e9d-0069-4ad9-996a-1d5968216998` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

针对法国蒂特河下游的运营级洪水预测，构建了基于高分辨率Telemac2D模型的合成洪水数据库，并在此基础上开发了利用投影网格和多网格连通性的图神经网络 surrogate，实现了从边界驱动的流量输入到洪水地图的快速推断。

**💡 创新点**

创新点包括：①在有限元节点级数据上构造投影网格并使用多网格（multimesh）长距离连通性，显著扩大有效感受野；②将全局流量 Q(t) 作为全局特征广播到所有节点，增强对边界驱动的条件化；③采用推前训练（pushforward）降低训练与推理之间的误差累积；④在真实运营环境中进行洪水地图（25 m网格）层面的 CSI 评估，证明 surrogate 在时间和空间上均优于传统数值模拟。

**🔧 技术方法**

技术主要包括：MeshGraphNet 架构（节点/边编码、消息传递、解码器）；投影网格生成（按密度映射下采样）和多网格连通性构造；推前训练策略；使用 Telemac2D 进行高精度仿真，随后对结果进行投影、插值与评估。

**📊 数据集**

数据集：基于 Telemac2D 的 56 条合成洪水事件（4 种历史洪峰形态 × 14 规模化流量），每条事件 40 h、30 min 输出，节点数约 4.1 × 10^5，投影至 ×8 网格后约 1.6 × 10^4 节点；并使用原始高分辨率结果在 25 m 常规网格上生成洪水地图作为评估基准。

**📈 对比分析**

比较方法：在 16 条留出的测试洪水上进行 6 h 的多步推理，评估 state‑space L1 误差和洪水地图 CSI（h≥5 cm 与 h≥30 cm 两个阈值）。实验表明：加入 Q(t) 能显著降低误差；多网格连通性在 Q(t) 条件下进一步提升性能；推前训练进一步提升长时延推理稳定性。最佳配置（E6：Q(t)+multimesh+pushforward）在 6 h 预测下，算子推理时间约 0.4 s（单 A100 GPU），相比 56 核 CPU 需 180 min 的 Telemac 计算显著加速，CSI 在 30 cm 阈值下远高于其他配置。

**⚠️ 局限性**

局限性：①模型仅在合成的洪水数据库上训练，未验证对真实实时流量预测的鲁棒性；②投影和插值过程可能引入平滑或 aliasing 误差；③对边界条件的硬编码方式（如固定海平面）在多变海岸条件下需要改进；④缺乏对流量不确定性（如实时水位更新）下的性能评估，需进一步研究不确定性处理和模型泛化能力。

---

## 325. Token Warping Helps MLLMs Look from Nearby Viewpoints

**arXiv ID:** 2604.02870 | [PDF](https://arxiv.org/pdf/2604.02870v1)

**作者:** Phillip Y. Lee `[一作]` (Korea Advanced Institute Of Science And Technology), Minhyuk Sung `[通讯]` (Korea Advanced Institute Of Science And Technology)

**通讯引用:** 1351 | [OpenAlex ID](https://openalex.org/A5004099860)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究了在多模态大型语言模型中使用令牌变形（token warping）实现视角变化，并证明后向令牌变形能更稳定地将源视角信息迁移到目标视角，从而提升空间推理能力。

**💡 创新点**

提出了以令牌为粒度的后向视角变形方法，并对比前向变形、最近/自适应抓取策略，证明后向令牌变形在保持语义一致性和鲁棒性方面优于传统像素级变形及生成式视角合成。

**🔧 技术方法**

使用 ViT 令牌化、前向/后向空间映射、深度图驱动的逆向投影、最近/自适应抓取、Qwen2.5‑VL 等大型语言模型进行评估，并构建 ViewBench benchmark。

**📊 数据集**

采用 ScanNet 真实扫描的相邻视角图像对作为视角变换数据集，同时在 CV‑Bench‑2D 上验证令牌噪声鲁棒性。

**📈 对比分析**

与像素级前向/后向变形、专用空间推理 MLLM（SpatialReasoner、ViLaSR、VLM‑3R）以及生成式视角合成 GenWarp 进行对比；后向令牌变形在视角条件空间推理任务中显著优于所有基线（准确率提升约10–15%），在目标视角对象描述任务中也获得最高评估分数。

**⚠️ 局限性**

目前仅针对近距离视角变化有效，仍依赖深度图（误差可能导致局部失真），且实验主要基于 ScanNet 数据，尚未在更大视角差异或更真实摄像机噪声环境中进行验证。

---

## 326. HairOrbit: Multi-view Aware 3D Hair Modeling from Single Portraits

**arXiv ID:** 2604.02867 | [PDF](https://arxiv.org/pdf/2604.02867v1)

**作者:** Leyang Jin `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Hao Li `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 18810 | [OpenAlex ID](https://openalex.org/A5100348588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种将单视角发型重建转化为多视角校准重建的完整流程，利用视频扩散模型生成360°多视角图像，再通过全视角神经方向提取器、混合隐式场和高效发束生长算法实现细粒度、几何一致的三维发束重建。

**💡 创新点**

核心创新包括：①利用视频扩散模型内在的跨视角先验实现单视角到多视角的桥接；②训练全视角神经方向提取器，解决传统Gabor滤波器噪声与前视局限问题；③设计联合方向与占据的混合隐式场，消除传统占据场的冗余查询；④混合根部与段式生长策略，实现高效率与高完整度的发束生成。

**🔧 技术方法**

技术包括LoRA微调视频扩散模型（WAN+DiT+VAE）、Flux.1-dev超分辨率、U-Net方向提取、基于多视角深度的混合隐式场网络、并行发束生长与后处理。

**📊 数据集**

使用合成的USC‑HairSalon与Difflocks三维发束模型，生成24fps视频进行LoRA训练；用于混合隐式场训练的543个模型（含镜像）；方向提取器训练基于1250个前视标注和395个全视角标注。

**📈 对比分析**

与UniHair、HairStep、Im2Haircut、Difflocks等方法比较，在多视角生成和单视角三维发束重建上均取得显著优势：L1、PSNR、LPIPS等指标优于对手；在HairSale、IoU等评估指标上表现突出；视觉上保持输入风格一致且在不可见视角恢复自然。

**⚠️ 局限性**

局限性包括：对视频扩散模型的依赖导致生成质量受限；合成数据与真实数据的域差异仍影响最终表现；高质量三维发束生成的计算开销相对较大；对极端发型或极度遮挡场景的适应性有待提升。

---

## 327. Accelerating Black-Box Bilevel Optimization with Rank-Based Upper-Level Value Function Approximation

**arXiv ID:** 2604.02888 | [PDF](https://arxiv.org/pdf/2604.02888v1)

**作者:** Marc Ong `[一作]` (University of Tsukuba), Youhei Akimoto `[通讯]` (University of Tsukuba)

**通讯引用:** 2228 | [OpenAlex ID](https://openalex.org/A5038757231)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `5b4c1114-4a70-478e-9921-2514ee03850d` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了一种针对黑盒双层优化问题的新的基于排名的上层价值函数逼近方法URA-CMA-ES，利用warm-start和early-stopping技术显著降低了下层优化开销。

**💡 创新点**

创新点在于将上层价值函数的排名信息直接用于上层CMA-ES的更新，消除了传统方法中上层分布与下层优化结果不匹配的问题，并通过可选择的共享配置实现跨代信息交流。

**🔧 技术方法**

主要技术包括CMA-ES的嵌套使用、基于Kendall相关系数的早停判定、热启动缓存机制以及对冲区约束的镜像处理。

**📊 数据集**

在两个标准双层优化基准集上进行评估：SMD（八个连续双层问题）和WRA（十一种极大极小优化问题），均采用20次随机重启。

**📈 对比分析**

与BL-CMA-ES和BOC等现有方法相比，URA-CMA-ES在多数任务上实现了更高的成功率和更低的下层函数评估次数，尤其在多模态和强交互的情形下表现优异。

**⚠️ 局限性**

局限性包括对下层解集无限多时无法选取唯一最优解（如SMD6、WRA4），以及在极度多维的最优响应集合上仍无可行策略，需进一步改进下层信息共享或采用专门的白盒求解器。

---

## 328. NeuReasoner: Towards Explainable, Controllable, and Unified Reasoning via Mixture-of-Neurons

**arXiv ID:** 2604.02972 | [PDF](https://arxiv.org/pdf/2604.02972v1)

**作者:** Haonan Dong `[一作]` (Peking University), Guojie Song `[通讯]` (Peking University)

**通讯引用:** 6082 | [OpenAlex ID](https://openalex.org/A5088976879)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `afceb026-1760-41ae-8d86-010831a37d97` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

针对大规模推理模型中的三种失败模式（单步错误、跨步循环、过度思考），通过白盒神经元分析定位关键神经元，并构建可解释、可控的统一推理框架。

**💡 创新点**

提出三类关键神经元专家与其振荡模式，并将其与轻量MLP、特殊触发令牌和SFT结合，实现对失败模式的在线检测与可控自校正。

**🔧 技术方法**

白盒神经元归因、傅里叶特征提取、轻量MLP预测、SFT微调、特殊触发令牌、自诊断-纠正机制。

**📊 数据集**

使用数学推理（GSM8K、MATH500、AIME24/25）、科学推理（GPQA‑Diamond）和代码推理（LiveCodeBench）六个基准，评估多种 7B–70B 后端模型。

**📈 对比分析**

对比 9 种基线（Vanilla、训练无、RL‑based）在 5 大基准上，实验显示在准确率提升 0.3–7.8%（最高 27%）的同时，token 消耗下降 19.6–63.3%，并在多任务、多模型上保持显著优势。

**⚠️ 局限性**

监控 MLP 产生轻微推理延迟，且当前框架尚非完全自动化，需进一步优化端到端流程。

---

## 329. LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation

**arXiv ID:** 2604.02954 | [PDF](https://arxiv.org/pdf/2604.02954v1)

**作者:** Yilin Xiao `[一作]` (Hong Kong Polytechnic University), Xiao Huang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 46624 | [OpenAlex ID](https://openalex.org/A5073869073)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

提出了针对GraphRAG的LogicPoison攻击框架，通过隐式破坏知识图谱拓扑实现逻辑推理失败。

**💡 创新点**

创新点在于利用类型保持的实体循环置换，从全局逻辑枢纽和查询特定推理桥段双重角度扰乱图结构，而非传统文本注入。

**🔧 技术方法**

使用命名实体识别、语料词频统计、链路思维（Chain-of-Thought）抽取、类型保持的实体置换以及图理论中心性分析等技术。

**📊 数据集**

在HotpotQA、2WikiMultiHopQA和MuSiQue三大多跳问答数据集上进行实验。

**📈 对比分析**

与PoisonedRAG及多种LLM攻击方法对比，LogicPoison在ASR和ASR‑GPT上均取得最高成功率，且时间、token成本最低，尤其在GraphRAG和GFM‑RAG等图基RAG框架中表现最优。

**⚠️ 局限性**

局限在于仅验证静态图构建、仅限英语语料，动态更新图谱或多语言环境下的持久性与可行性待进一步研究。

---

## 330. JoyAI-LLM Flash: Advancing Mid-Scale LLMs with Token Efficiency

**arXiv ID:** 2604.03044 | [PDF](https://arxiv.org/pdf/2604.03044v1)

**作者:** Aichen Cai `[一作]`, Zhuwei Zeng `[通讯]`

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `afceb026-1760-41ae-8d86-010831a37d97` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了JoyAI‑LLM Flash，一款48B参数、2.7B激活的稀疏Mixture‑of‑Experts LLM，并通过20万亿token预训练、严格的SFT‑DPO‑RL后训练流程实现高效推理与多模能力。

**💡 创新点**

创新点包括：① FiberPO RL算法（纤维束门控）实现多尺度信任区间控制，解决传统PPO/GRPO/GSPO的全局与局部不匹配；② 采用MTP与QAT协同的训练–推理共设计，显著提升多token预测速度和量化吞吐；③ 在20T训练语料上结合Muon优化器和极高稀疏度（仅激活1/18参数）实现更低的token消耗。

**🔧 技术方法**

使用技术：Muon优化器、Mixture‑of‑Experts（256专家、8路+1共享）、Multi‑Token Prediction、量化感知训练（FP8/INT8/INT4/GGUF）、FiberPO RL、FlashAttention‑3、DeepEP、ZeRO‑1、数据去重与安全过滤、人工评测与人工‑RL工具链。

**📊 数据集**

数据集：共计20T高质量token，涵盖Common Crawl、The Stack v2、GitHub代码、PDF文档、合成数据；后训练使用多源SFT数据（一般、代码、工具、代理等）、DPO偏好对齐数据、RL环境轨迹。

**📈 对比分析**

对比方法：与Qwen3‑30B‑A3B、Qwen3.5‑35B‑A3B以及GLM‑4.7‑Flash‑T进行基准对比。JoyAI‑LLM Flash在多项benchmark（MMLU、MATH、HumanEval、LiveCodeBench、RULER等）保持竞争力，且在token效率（85%减少token）和推理吞吐（MTP 1.87×、量化17%+）方面优于同类模型。

**⚠️ 局限性**

局限性：仍在大模型规模下，参数量大且激活稀疏度高导致部署复杂；在PinchBench等极长上下文场景下token消耗较高；RL多域训练仍可能出现知识迁移与对齐退化；缺乏持续学习与持久记忆机制，模型对新知识适应性有限。

---

## 331. A Boolean encoding of the Most Permissive semantics for Boolean networks

**arXiv ID:** 2604.03029 | [PDF](https://arxiv.org/pdf/2604.03029v1)

**作者:** Laure de Chancel `[一作]` (Institut Curie), Élisabeth Remy `[通讯]` (Aix Marseille Univ)

**关键词:** `dd4bd30e-3d3d-4e53-a403-da542c6c036a` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出了一种将 Most Permissive 语义下的布尔网络通过三元布尔变量编码为异步布尔网络的方法，并在 bioLQM 框架中实现了对应的工具。

**💡 创新点**

首次给出了可被现有布尔网络工具直接使用的 Most Permissive 语义的完全等价编码，并支持局部展开以缓解规模膨胀问题。

**🔧 技术方法**

采用多值决策图（MDD）构造逻辑函数，设计内部与交互规则，并在 bioLQM 中实现 Modifier。

**📊 数据集**

在三元模型、15 元血液干细胞模型以及 Fli1 扩展的 B 模型上进行了验证。

**📈 对比分析**

通过与原始 Most Permissive 状态转移图以及 GINsim 下的异步模型对比，验证了可达性结果完全一致；但完整展开导致状态空间爆炸，局部展开可恢复可达性评估。

**⚠️ 局限性**

主要局限是模型规模膨胀导致的状态空间爆炸，完整展开在大规模网络上不可行，只能通过局部展开或其他简化手段处理。

---

## 332. Posterior Matching over Binary-Input Memoryless Symmetric Channels: Non-Asymptotic Bounds and Low-Complexity Encoding

**arXiv ID:** 2604.03038 | [PDF](https://arxiv.org/pdf/2604.03038v1)

**作者:** Recep Can Yavas `[一作]` `[通讯]` (Bilkent University), Recep Can Yavas (Bilkent University)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究了在二进制输入无记忆对称信道上使用后验匹配的可变长度反馈编码，提出了一种新的非渐近可达性界限，并设计了一种低复杂度编码器。

**💡 创新点**

创新点在于去除了对有界对数似然比增量的限制，使得后验匹配分析适用于更广泛的BMS信道，包括连续输出信道，同时提供了明确的非渐近性能界限。

**🔧 技术方法**

使用了后验匹配、小差异（SED）分区、随机游走和更新理论等技术，提出了一种低复杂度编码器，通过分组消息和批量修复步骤来保持SED条件。

**📊 数据集**

使用了二进制输入加性高斯噪声（BI-AWGN）信道作为主要数据集，并进行了相关的数值设计和仿真。

**📈 对比分析**

与现有方法相比，提出的编码方案在复杂度上为多项式级别，且在非渐近性能上表现出色，特别是在BI-AWGN信道上，性能优于之前的实现。

**⚠️ 局限性**

限制在于该方法仍然依赖于信道的某些假设，如信道的对称性和输出量化的精确性，且在处理不对称信道时可能需要进一步的研究。

---

## 333. Proceedings of the 7th Workshop on Models for Formal Analysis of Real Systems

**arXiv ID:** 2604.03053 | [PDF](https://arxiv.org/pdf/2604.03053v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

---

## 334. Weight distribution bounds to relate minimum distance, list decoding, and symmetric channel performance

**arXiv ID:** 2604.02994 | [PDF](https://arxiv.org/pdf/2604.02994v1)

**作者:** Donald Kougang-Yombi `[一作]` (AIMS Rwanda), Jan Hązła `[通讯]` (AIMS Rwanda)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232`

**🎯 论文内容**

本文研究误差纠正码在最坏情况（最小距离）与随机噪声（对称信道、擦除信道、BSC、BIAWGN）下性能之间的关系，并给出了新的上界和下界。

**💡 创新点**

创新点在于：
• 将 Pernice–Sprumont–Wootters 的对称信道与列表解码的紧密联系从线性码推广到任意码；
• 通过直接估计码的重量分布（而非尖锐阈值技术）得到新的块误码概率上界；
• 引入 Samorodnitsky 的重量分布不等式，将擦除信道性能与对称信道性能结合，从而在 q≥4 且大 δ 的情况下突破 Johnson 约束，给出更高的可达交叉概率下界；
• 发展了一套统一的函数 M^q(γ,p) 与 F^q(γ,p) 的性质，支持对不同 q 的系统分析。

**🔧 技术方法**

主要技术：
• 双重计数与 Hamming 球/球交集的指数估计；
• 对重量分布的精确上界（Poltyrev、Samorodnitsky、Bhattacharyya 以及 Sphere/Ball 边界）；
• 对 M^q 与 F^q 的凸性、单调性、连续性证明；
• 利用 Erasure 列表解码与 BEC 性能的连接，推导对对称信道的误码概率上界；
• 对 BIAWGN 信道使用 Sphere Bound 与 I_n(α,β) 的估计。

**📊 数据集**

本文不依赖实验数据，而是纯理论证明；在实验对比中主要使用已知的随机码、Gilbert–Varshamov、Algebraic-Geometric 码等经典码族作为参考点。

**📈 对比分析**

与传统 Johnson Bound、Pernice 等的线性码结果对比：
• 对于所有 q，证明了列表解码半径与对称信道交叉概率之间的紧密对应；
• 在 q≥4、δ 较大时，得到的下界 p_*^q(λ,δ) 明显高于 Johnson 半径 J_q(δ)，并给出了数值示例（q=9、17 等）；
• 对于 q=2，新的下界与已有的 BEC 列表解码结果一致；
• 对于 BIAWGN，给出与 Samorodnitsky 相关的误码概率上界，并推导出 σ_*^2(λ,δ) 与传统 Sphere Bound 的比较。

**⚠️ 局限性**

局限性：
• 改进仅适用于 q≥4 且 δ 较大的情况，对 q=2、3 的进一步突破仍待研究；
• 需要线性码具备良好的擦除信道性能（即 vanishing bit‑error 率），否则无法应用 M^q‑F^q 估计；
• 证明依赖于极限 n→∞，实际有限 n 的性能仍需实验验证；
• 对于非线性码，只得到一般性结果，尚缺乏构造性设计指南；
• 对 BIAWGN 的 Sphere Bound 采用了较粗略的估计，可能在实际噪声水平上有保守性。

---

## 335. FedSQ: Optimized Weight Averaging via Fixed Gating

**arXiv ID:** 2604.02990 | [PDF](https://arxiv.org/pdf/2604.02990v1)

**作者:** Cristian Pérez-Corral `[一作]` (Universitat Politècnica de València), Enrique S. Quintana-Ortí `[通讯]` (Universitat Politècnica de València)

**通讯引用:** 6913 | [OpenAlex ID](https://openalex.org/A5012806004)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `5b4c1114-4a70-478e-9921-2514ee03850d` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出FedSQ框架，在联邦学习中冻结预训练模型的结构组件（门控），仅聚合量化参数进行细调，提升跨硅片异构环境下的稳健性和收敛效率。

**💡 创新点**

利用sk/qk分离，将预训练结构视为固定二进制门控，创建DualCopy模型，只优化并聚合在固定门控下的线性映射参数；同时引入中心化的TL调度，保证跨客户端一致性。

**🔧 技术方法**

DualCopy结构、固定门控掩码、ImageNet-1K预训练、FedAvg/FedProx聚合、Dirichlet异构划分、AlexNet/CIFAR-10、ResNet18/CIFAR-100、SGD/AdamW优化。

**📊 数据集**

CINIC-10（基于CIFAR-10+ImageNet）、CIFAR-100；使用ImageNet-1K做预训练；在Flower框架下模拟10个客户端。

**📈 对比分析**

与标准FedAvg、FedProx在i.i.d.和Dirichlet分割下对比；FedSQ在i.i.d.下相当或略低，但在Dirichlet异构下显著提升，ResNet18/CIFAR-100最高验证准确率提升约4–5%并提前3轮。

**⚠️ 局限性**

依赖预训练初始化，需中心化TL调度与超参数搜索；仅在二进制门控假设下验证，未评估更复杂激活；通信压缩未整合；对极端异构或大规模客户端的稳健性待进一步验证。

---

## 336. Mitigating Reward Hacking in RLHF via Advantage Sign Robustness

**arXiv ID:** 2604.02986 | [PDF](https://arxiv.org/pdf/2604.02986v1)

**作者:** Shinnosuke Ono `[一作]` (University of Tokyo), Masashi Sugiyama `[通讯]` (Riken Aip)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

设计了一种基于每个完成的优势符号可验证半径的鲁棒策略梯度方法，以降低RLHF中的奖励劫持。

**💡 创新点**

引入“认证优势符号保持半径”并在策略梯度中按此半径加权，能在不训练多RM或访问训练数据的情况下实现鲁棒性。

**🔧 技术方法**

采用随机平滑思想的参数扰动、线性头的近似、加权策略梯度和自适应阈值。

**📊 数据集**

在TL;DR总结和AlpacaFarm两套基准上，使用Pythia和Qwen2.5系列模型及对应的金牌RM进行评估。

**📈 对比分析**

与Dr.GRPO、UWO、BSPO、AdvPO等方法对比，获得最高或相近的金牌RM胜率，且计算开销几乎与Dr.GRPO相同。

**⚠️ 局限性**

仅在线性头扰动下可计算，且对完整RM参数梯度求法仍高成本，未对非线性头或更复杂RM进行证明。

---

## 337. STEAR: Layer-Aware Spatiotemporal Evidence Intervention for Hallucination Mitigation in Video Large Language Models

**arXiv ID:** 2604.03045 | [PDF](https://arxiv.org/pdf/2604.03045v1)

**作者:** Linfeng Fan `[一作]` (Renmin University of China), Zhiwu Lu `[通讯]` (Renmin University of China)

**通讯引用:** 2519 | [OpenAlex ID](https://openalex.org/A5103244144)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文针对视频大语言模型在生成过程中出现的空间和时间幻觉问题，提出了一种在推理时对中间层视觉证据进行干预的框架 STEAR。

**💡 创新点**

创新点在于将幻觉视为跨层时空证据错位，并通过层感知的关键证据选择、在中间层重新注入证据以及在同一证据上构造局部时序对比来同时抑制空间和时间幻觉。

**🔧 技术方法**

采用的技术包括基于 token 不确定性触发的干预、聚合中间层注意力做关键证据选择、使用 FFN 重新注入视觉记忆以及基于局部时序打乱/同质化的对比解码。

**📊 数据集**

实验数据集涵盖 EventHallusion、VidHalluc、NExT‑QA 和 MVBench 四大视频推理与幻觉评测集。

**📈 对比分析**

与 VCD、MemVR、TCD 和 DINO‑HEAL 等先行方法比较，STEAR 在三种主流 Video‑LLM 骨干上均实现了所有指标的提升，平均提升幅度达 3–4 个百分点，且保持单编码低延迟。

**⚠️ 局限性**

局限性在于仍需手动设定不确定性阈值、关键证据比例等超参，对注意力作为证据的假设可能不适用于所有模型，且对极端复杂的时空关系仍存在一定误判。

---

## 338. Asymptotically-Bounded 3D Frontier Exploration enhanced with Bayesian Information Gain

**arXiv ID:** 2604.03008 | [PDF](https://arxiv.org/pdf/2604.03008v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7`

---

## 339. Analyzing Healthcare Interoperability Vulnerabilities: Formal Modeling and Graph-Theoretic Approach

**arXiv ID:** 2604.03043 | [PDF](https://arxiv.org/pdf/2604.03043v1)

**作者:** Jawad Mohammed `[一作]`, Gahangir Hossain `[通讯]`

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `bb57609f-8351-4b1b-85e4-3afa07da95d6` `67630363-6be0-4f51-ab05-7198250671a5` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文提出了FHIR资源访问图(FRAG)模型，并给出了三类临床相关竞争条件（SWC、TAV、CUR）的形式化定义，基于该图实现了三遍检测算法，并在合成日志上进行评估。

**💡 创新点**

创新点在于将操作系统的临界区理论与FHIR无同步机制相结合，首次给出一种基于有向图的形式化竞争检测框架，能够区分不同竞争类型并在实践中实现高精度检测。

**🔧 技术方法**

使用技术包括：图论（有向标记图）、Python编程、NetworkX库构建图、NumPy生成合成日志、三遍检测算法实现、基准时间窗口扫描比较；同时用正式化方法证明检测可行性。

**📊 数据集**

使用的数据集为1500条合成的FHIR R4事务日志，日志中注入已知的SWC、TAV、CUR竞争事件，保证有明确的ground truth。

**📈 对比分析**

对比方法是将FRAG与基线时间窗口扫描（δ=200 ms）在三种并发条件（顺序、无同步、ETag部分同步）下分别评估；在C2（完全并发）条件下FRAG整体F1为78.5%（SWC 98.0%，TAV 99.9%，CUR 41.5%），基线为98.8%；在C3（部分ETag同步）条件下FRAG精度保持高（SWC 96.6%，TAV 99.0%）但召回率急剧下降（整体F1 26.4%）。

**⚠️ 局限性**

局限性包括：依赖完整的访问日志；仅在合成数据上验证，真实工作负载可能不同；时间戳分辨率有限（100 ms）；只定义了三类竞争，未覆盖所有可能的竞争；采用单资源一次分配策略导致跨类召回下降；未考虑分布式多服务器环境。

---

## 340. QVAD: A Question-Centric Agentic Framework for Efficient and Training-Free Video Anomaly Detection

**arXiv ID:** 2604.03040 | [PDF](https://arxiv.org/pdf/2604.03040v1)

**作者:** Lokman Bekit `[一作]` (University of South Florida), Yasin Yilmaz `[通讯]` (University of South Florida)

**通讯引用:** 2309 | [OpenAlex ID](https://openalex.org/A5036320427)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出QVAD框架，实现无训练的视频异常检测，采用LLM与VLM的多轮问答交互，动态更新提示以提高检测精度。

**💡 创新点**

创新点在于：1）LLM驱动的问答式提示更新，可在不增加参数的前提下显著提升轻量VLM的推理质量；2）结合向量记忆检索，增强长时序上下文关联；3）实现端到端可在边缘设备上部署的高效解决方案。

**🔧 技术方法**

使用技术包括：Qwen3 LLM 与 Qwen3 VL 视觉语言模型、LLM-Agent交互框架、向量检索记忆、帧选择与运动感知、双层Gaussian平滑后处理。

**📊 数据集**

实验数据集：UCF-Crime、XD-Violence、UBNormal、ComplexVAD。

**📈 对比分析**

与多种训练免费与训练依赖方法对比，QVAD在UCF-Crime AUC 84.28%（与Panda、VADTree相当），XD-Violence AP 68.53%，UBNormal AUC 79.6%，ComplexVAD AUC 68.02%；仅使用约8B参数、4.2GB显存即可实现实时帧率，显著低于同类方法。

**⚠️ 局限性**

局限性：多轮问答可能增加推理延迟；对极短或极端复杂异常的辨识仍受限；向量记忆维护和更新复杂度高；问答策略高度依赖LLM的语言质量，可能导致偶尔不一致或误判。

---

## 341. Compositionality of Lyapunov functions via assume-guarantee reasoning

**arXiv ID:** 2604.03017 | [PDF](https://arxiv.org/pdf/2604.03017v1)

**作者:** Matteo Capucci `[一作]` (University of Strathclyde), David Jaz Myers `[通讯]` (Topos Institute)

**通讯引用:** 464 | [OpenAlex ID](https://openalex.org/A5108499440)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c` `5b4c1114-4a70-478e-9921-2514ee03850d` `a8e75ba4-7a2d-4153-b003-06c94533add0`

**🎯 论文内容**

提出了一种基于范畴理论的假设-保证推理框架，能够对泛化 Moore 机（包括普通 Moore 机、POMDP、参数化 ODE 等）以及局部输入-状态稳定（LISS）Lyapunov 函数进行组合式安全验证。

**💡 创新点**

创新点在于：1）将假设-保证推理重新表述为 Moore 机的“已认证”形态，使其自然落在镜头（lens）范畴中；2）构建了一个双范畴/双模块的结构，实现了认证机器的模组化与纤维化；3）给出了针对参数化 ODE 的 LISS Lyapunov 函数的新的可组合认证方法。

**🔧 技术方法**

使用的技术主要包括：范畴理论中的镜头（lens）与双范畴（double category）、纤维化（fibration）与张量积的模组化、泛化 Moore 机的形式化、以及对 ODE 的切空间和存储函数的几何解释。

**📊 数据集**

该工作为理论性框架，未使用具体数据集进行实验。

**📈 对比分析**

无实验结果与性能比较。

**⚠️ 局限性**

限制：目前仅证明了对安全谓词的组合式验证，未扩展到更一般的 ω-正则谓词或随机过程；对 POMDP 的实际实现仍待完成；并且在具体系统上的可扩展性与实现细节尚未给出。

---

## 342. Exploring Motion-Language Alignment for Text-driven Motion Generation

**arXiv ID:** 2604.02973 | [PDF](https://arxiv.org/pdf/2604.02973v1)

**作者:** Ruxi Gu `[一作]` (University of Science and Technology of China), Wei Wang `[通讯]` (State Key Laboratory of General Artificial Intelligence)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `40105733-5154-44cd-8090-a8cab9e64b07` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文提出了 MLA-Gen 框架，用以改进文本驱动的人体动作生成，通过整合全局运动先验与细粒度文本-动作对齐来提升生成质量与语义一致性。

**💡 创新点**

创新点在于识别并量化了注意力 sink 现象，提出 SinkRatio 指标，并通过 sink‑mask 与 sink‑ctrl 两种对齐感知机制动态调节注意力与条件引导，实现更精准的动作-文本对齐。

**🔧 技术方法**

技术上采用流式生成模型（Flow-based）结合 CLIP 文本嵌入、可学习的记忆槽、交叉注意力以及自适应的 classifier‑free guidance。

**📊 数据集**

使用 HumanML3D 数据集进行训练与评估。

**📈 对比分析**

与多种基准（如 ACMDM、MotionDiffuse 等）对比，MLA-Gen 在 FID、R‑Precision、匹配度与 CLIP 分数等指标上均取得显著提升，例如 FID 从 0.107 降到 0.056，R‑Precision Top‑1 从 0.522 提升至 0.527。

**⚠️ 局限性**

局限性包括对长文本或极其复杂动作语义的对齐仍有欠缺，且 SinkRatio 仅捕捉单词级注意力聚焦，未能反映更高阶语义关联。

---

## 343. Collaborative Multi-Mode Pruning for Vision-Language Models

**arXiv ID:** 2604.02956 | [PDF](https://arxiv.org/pdf/2604.02956v1)

**作者:** Zimeng Wu `[一作]` (Beihang University), Jiaxin Chen `[通讯]` (Beihang University)

**通讯引用:** 8493 | [OpenAlex ID](https://openalex.org/A5100360561)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `fede83ac-7505-405f-ab37-e7284695c47f` `edb9d762-f411-4838-a852-f2d638b018db` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CoMP框架，实现参数与token的联合剪枝，显著降低VLM的计算开销。

**💡 创新点**

创新点在于设计协作重要性度量(CIM)以消除参数与token重要性指标冲突，并引入基于成本的多模态剪枝策略(MPS)动态选择最优剪枝模式。

**🔧 技术方法**

使用结构化参数剪枝、token剪枝、协作重要性度量、基于成本的多模态剪枝策略，以及Transformer的自注意力机制与层归一化等技术。

**📊 数据集**

在BLIP、CLIP、LLaVA等模型上使用NLVR2、COCO、Flickr30K、VQAv2、MME、MMB、GQA、TextVQA、POPE等多任务数据集进行评估。

**📈 对比分析**

与UPop、MADTP、Turbo、SJP等单模剪枝方法对比，在高剪枝比例下准确率提升约2–4%，FLOPs大幅下降，显示出优越的压缩效果。

**⚠️ 局限性**

局限性包括仅在已微调模型上验证，未在大规模预训练模型上评估，缺乏理论解析和极端稀疏率下的稳健性保障。

---

## 344. act: Technical report

**arXiv ID:** 2604.02955 | [PDF](https://arxiv.org/pdf/2604.02955v1)

**作者:** Zoe Paraskevopoulou `[一作]` (National Technical University of Athens), Alexis Terry `[通讯]` (Argot Collective)

**关键词:** `2f20b7a7-8630-4b01-9311-4db57188b72c`

**🎯 论文内容**

本文提出了一套面向以太坊智能合约的形式化类型系统，并给出了指针语义、环境和存储状态的完整安全性证明；

**💡 创新点**

创新点在于将时序环境（caller、origin、callvalue）与指针语义相结合，提供了合约创建、更新和状态迁移的形式化安全证明；

**🔧 技术方法**

使用了类型理论、形式化语义推导、指针语义判定与归纳证明等技术；

**📊 数据集**

未使用传统意义上的数据集，主要通过形式化模型和证明来验证；

**📈 对比分析**

通过理论证明与归纳推导进行方法比较，主要评估的是形式化安全性与一致性，无实验性能指标；

**⚠️ 局限性**

局限性在于仅覆盖了部分 Solidity 的语言子集，未涵盖全部以太坊生态复杂交互与完整的合约交互细节。

---

## 345. CIDER: Boosting Memory-Disaggregated Key-Value Stores with Pessimistic Synchronization

**arXiv ID:** 2604.03007 | [PDF](https://arxiv.org/pdf/2604.03007v1)

**作者:** Yuxuan Du `[一作]` (Fudan University), Jiacheng Shen `[通讯]` (Duke Kunshan University)

**通讯引用:** 1365 | [OpenAlex ID](https://openalex.org/A5101867257)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文提出了通过引入悲观同步以及全局写合并和冲突感知同步机制来提升内存分离式键值存储的性能，解决了现有乐观同步在高并发下导致的 I/O 重复问题。

**💡 创新点**

创新点在于：①基于 MCS 锁的全局写合并（Global Write‑Combining），实现跨节点的写操作合并；②冲突感知同步（Contention‑Aware Synchronization），在热点键使用悲观同步、冷门键使用乐观同步，实现动态切换；③将上述机制集成到可复用的计算侧优化框架 "PESSIMISTIC" 中，可直接应用于多种键值存储。

**🔧 技术方法**

主要技术包括：RDMA 一侧原语（RDMA_READ/WRITE/CAS/FAA）；ShiftLock 版 MCS 锁实现分布式锁；全局写合并与局部写合并的对比；基于信用计数的冲突感知；AIMD 算法动态调整信用。

**📊 数据集**

使用 YCSB（Zipfian 分布）生成的写密集、读密集和写全三种工作负载，对 60M 条键值数据进行基准测试，并在 RACE（哈希表索引）和 SMART（适配树索引）上进行端到端评估。

**📈 对比分析**

与乐观同步（O‑SYNC）、CAS 锁和 ShiftLock 基线相比，PESSIMISTIC 在写密集工作负载下吞吐量提升最高可达 6.6×，P99 延迟下降 12.4×；在写全负载下吞吐量提升 6.5×，P99 延迟下降 6.1×；读密集负载下性能相近，说明技术对读操作影响最小。

**⚠️ 局限性**

限制包括：①对 MCS 锁的依赖导致在极高并发下仍有锁维护开销；②全局写合并引入的额外远程访问和锁节点状态更新；③需要为热点键维护额外元数据，增加内存占用；④在事务或需要严格两阶段锁定的场景下无法直接使用；⑤在键值大小不固定或复杂索引结构上结合效果可能受限。

---

## 346. Enhancing Multi-Robot Exploration Using Probabilistic Frontier Prioritization with Dirichlet Process Gaussian Mixtures

**arXiv ID:** 2604.03042 | [PDF](https://arxiv.org/pdf/2604.03042v1)

**作者:** John Lewis Devassy `[一作]` (Instituto Superior Técnico, Universidade de Lisboa), Pedro U. Lima `[通讯]` (Instituto Superior Técnico, Universidade de Lisboa)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

提出了一种基于Dirichlet过程高斯混合模型（DP-GMM）的前沿优先级分配方法，以提升多机器人探索框架的任务分配效率。

**💡 创新点**

创新点在于将DP‑GMM与信息增益的概率分布相结合，实现软聚类与概率信息增益的联合优化，避免了硬聚类导致的子最优分配。

**🔧 技术方法**

主要技术包括DP‑GMM软聚类、信息增益估计、视点生成与合并、以及在FAME和FroShe两种现有算法中的整合。

**📊 数据集**

使用四种人工森林环境（稀疏、中等、稠密、混合）以及真实双无人机实验场景进行评估。

**📈 对比分析**

通过与原始FAME、FroShe及RACER的对比，平均提升约10%–25%的探索效率；在多无人机和通信受限情况下，标准差显著下降，路径重叠次数减少。

**⚠️ 局限性**

局限在于DP‑GMM实现使用Python/Scikit‑learn，计算开销仍较高；对极大规模环境的验证不足；在高通信延迟或极端动态环境下的鲁棒性尚未充分验证。

---

## 347. ARM: Advantage Reward Modeling for Long-Horizon Manipulation

**arXiv ID:** 2604.03037 | [PDF](https://arxiv.org/pdf/2604.03037v1)

**作者:** Yiming Mao `[一作]` (LimX Dynamics), Hua Chen `[通讯]` (LimX Dynamics)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出了一种通过相对优势奖励建模（ARM）来解决长周期机器人操作中的奖励工程瓶颈，利用三态相对优势标签训练多模态MIMO Transformer实现自动进度重构，并通过优势加权行为克隆（AW‑BC）训练高效策略。

**💡 创新点**

创新点包括：1) 低成本三态相对优势标签策略，显著降低人工标注难度；2) 采用MIMO时序Transformer进行多模态输入、多输出的优势预测；3) 自动进度重构算法将离散相对优势转化为稠密进度曲线；4) 基于优势的统计归一化权重的AW‑BC训练方法，能够过滤低质量样本并突出高优势行为。

**🔧 技术方法**

使用的技术包括：CLIP视觉特征提取、机器人本体感知融合、MIMO Transformer网络、双头损失（优势分类+完成预测）、三态标签标注、自动进度重构、优势加权行为克隆、统计归一化权重与截断策略。

**📊 数据集**

数据集：972条毛巾折叠轨迹（809条专家演示、163条DAgger错误修正），采集自AgileX ALOHA双臂遥控系统，包含多种表面高度、不同初始状态。

**📈 对比分析**

与SARM、RA‑BC、传统BC基线对比，ARM+AW‑BC在毛巾折叠任务中实现了99.4%成功率、32 episode/小时吞吐率和3.6折叠精度，远超SARM（78.5%）和BC基线（62.1%）。此外，MIMO架构的推理速度提升至14.1 it/s，显著高于VLM（1.03 it/s）和SARM（3.9 it/s）。

**⚠️ 局限性**

局限性：1) 仅在单一长周期折叠任务上验证，尚未在多种任务和更复杂的真实环境中测试；2) 需要人类先验的三态标签作为初始训练信号；3) 对极端回退或完全失控行为的处理机制仍不完善；4) 计算资源依赖较高，尤其是MIMO Transformer的训练与推理成本。

---

## 348. Beyond Isolated Tasks: A Framework for Evaluating Coding Agents on Sequential Software Evolution

**arXiv ID:** 2604.03035 | [PDF](https://arxiv.org/pdf/2604.03035v1)

**作者:** KN Ajay Shastry `[一作]` (Fujitsu Research), Chaitanya Devaguptapu `[通讯]` (Fujitsu Research)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c773407a-6119-4871-b8b3-1e7ae17a6851` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

构建了自动化框架生成多步长、状态化的编码任务，并推出SWE-STEPS数据集，用于评估LLM编码代理在长期软件演化场景中的表现。

**💡 创新点**

首次引入状态化、时序化、多维度（功能正确性+仓库健康）评估，并揭示传统孤立PR评估对代理能力的高估以及技术债务对长期性能的负面影响。

**🔧 技术方法**

利用Git提交图抽取任务链、测试驱动验证、静态分析工具SonarQube、OpenHands/Aider等LLM编码代理以及大语言模型（Gemini、Claude、GPT）实现多轮反射循环执行。

**📊 数据集**

使用SWE-STEPS（168任务、963 PR，涵盖6个Python项目）及其Lite/Mini子集，并与SWE-Bench、SWE-Gym等现有数据集进行对比。

**📈 对比分析**

在Individual（孤立）与Global/PRD（状态化）设置下对比评估，发现孤立评估提升15–25%，在状态化设置中多种LLM的任务完成率显著下降，并且代理往往导致认知复杂度和技术债务上升。

**⚠️ 局限性**

局限性包括任务链的可验证性和依赖性有限、对更大规模或非Python项目的泛化尚未验证、未深入探讨安全与伦理风险。

---

## 349. Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?

**arXiv ID:** 2604.03016 | [PDF](https://arxiv.org/pdf/2604.03016v1)

**作者:** Qianshan Wei `[一作]` (Chinese Academy Of Sciences), Yi-Fan Zhang `[通讯]` (Chinese Academy Of Sciences)

**通讯引用:** 8447 | [OpenAlex ID](https://openalex.org/A5100376961)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了Agentic-MME基准，用于评估多模态大型语言模型在视觉操作与网络检索协同中的主动能力。

**💡 创新点**

创新点在于构建统一的工具接口、过程级验证与多难度层级，并提供2000+人类标注的步骤检查点，实现对工具调用过程的细粒度评估。

**🔧 技术方法**

使用了工具调用框架（函数调用与沙箱代码执行）、AST追踪、LLM审计、open-web检索API，以及多模态视觉操作工具。

**📊 数据集**

构造了418个真实场景任务，跨6个领域，共计2,000+步骤标注，图像来源于公开网页并通过逆向草拟确保需要主动操作。

**📈 对比分析**

与人类对照以及多种开源/闭源模型对比，最优模型Gemini 3 Pro在整体上达56.3%准确率，难度层级3仅33.3%，显著低于人类93.8%，证明了模型在多步骤规划与工具执行上的不足。

**⚠️ 局限性**

局限在于仍依赖人工标注大量步骤，工具多样性受限，仅涵盖有限视觉与检索工具；评估仍受LLM审计误判可能影响；缺乏对代码生成模式下复杂工具组合的充分挖掘。

---

## 350. Effect of Input Resolution on Retinal Vessel Segmentation Performance: An Empirical Study Across Five Datasets

**arXiv ID:** 2604.02977 | [PDF](https://arxiv.org/pdf/2604.02977v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9`

---

## 351. Extending deep learning U-Net architecture for predicting unsteady fluid flows in textured microchannels

**arXiv ID:** 2604.02976 | [PDF](https://arxiv.org/pdf/2604.02976v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2a04ab72-0614-4cc6-b3a4-14f75d696aea`

---

## 352. A Flow Matching Framework for Soft-Robot Inverse Dynamics

**arXiv ID:** 2604.03006 | [PDF](https://arxiv.org/pdf/2604.03006v1)

**作者:** Hang Yang `[一作]`, Ke Wu `[通讯]`

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `40105733-5154-44cd-8090-a8cab9e64b07` `04572f8d-59e5-41c9-8850-ac8e7ee2b108` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出一种基于流匹配的逆动力学学习框架，学习软连续机器人局部状态转移对应的力矩控制，实现开环前馈控制。

**💡 创新点**

将逆动力学改写为条件流匹配问题，采用 Rectified Flow 并加入物理先验残差建模（RF-Physical）与前向动力学一致性约束（RF-FWD），显著提升轨迹跟踪精度与控制平滑性。

**🔧 技术方法**

使用条件流匹配、Rectified Flow、轻量 MLP 速度场、前向动力学一致性约束、物理先验残差建模、Euler 积分等技术。

**📊 数据集**

使用 3,000 条高保真仿真 excitation episode 采集的状态‑输入对作为训练集；实验阶段采集实际软体机器人运动数据。

**📈 对比分析**

与 MLP、LSTM、Transformer 三种回归基线对比；在结构化轨迹上 RMSE 5.0 mm，随机轨迹 3.5 mm，较基线降低 50%+；推理时间 0.995 ms，末端速度最高 1.14 m/s；在仿真 2.03 m/s、实验 1.14 m/s 速度下保持稳定。

**⚠️ 局限性**

仅在单段软体机器人上验证，未针对多段或更复杂外部扰动进行泛化；物理先验对静态弹性有效，动态瞬态捕捉仍受限；需结合轻量反馈校正提升鲁棒性。

---

## 353. Self-Optimizing Multi-Agent Systems for Deep Research

**arXiv ID:** 2604.02988 | [PDF](https://arxiv.org/pdf/2604.02988v1)

**作者:** Arthur Câmara `[一作]` (Zeta Alpha), Jakub Zavrel `[通讯]` (Zeta Alpha)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `5b4c1114-4a70-478e-9921-2514ee03850d` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计并实现了一种基于多智能体的深度研究（Deep Research）系统，并探索了利用自我学习和提示优化方法（如GEPA和TextGrad）来自动改进系统表现。

**💡 创新点**

创新点在于将提示和系统参数视为可训练的可优化参数，利用LLM自身作为优化器实现系统的自我改进，避免了传统手工提示工程的脆弱性，并在工业规模下验证了该方法可生成与专家提示相媲美或更优的系统。

**🔧 技术方法**

主要技术包括：多智能体架构（调度者、阅读者、聚合器、撰写者）、基于LLM的提示优化（GEPA、TextGrad）、自我对弈/自我探索、反馈机制、引用管理与去重、并行检索与信息抽取。

**📊 数据集**

使用了工业级文档集合（包含数百篇网页、PDF和内部知识库条目），以及公开的深度研究任务数据（例如学术报告写作、信息检索评测集）进行实验。

**📈 对比分析**

通过与专家手工设计的提示对比实验，评估了回答质量、信息覆盖率和引用准确性。结果表明，自动化提示优化的系统在大多数指标上均优于基准，尤其在信息完整性和生成质量上提升了约10%–15%。

**⚠️ 局限性**

局限性包括：仍需大量算力进行提示搜索和对弈；在极端新领域或模型升级时需要重新训练；提示优化可能产生过拟合或生成不一致的回答；以及缺乏对长链推理错误的全局可解释性。

---

## 354. Prompt Compression in the Wild: Measuring Latency, Rate Adherence, and Quality for Faster LLM Inference

**arXiv ID:** 2604.02985 | [PDF](https://arxiv.org/pdf/2604.02985v1)

**作者:** Cornelius Kummer `[一作]` (Technische Universität Dresden), Sahar Vahdati `[通讯]` (Technische Universität Dresden)

**通讯引用:** 831 | [OpenAlex ID](https://openalex.org/A5102785999)

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

研究并量化提示压缩在大型语言模型推理中的延迟、速率与质量影响，完成了涵盖多模型、多GPU与多任务的大规模实验。

**💡 创新点**

首次系统化评估压缩开销与解码速度的权衡，并提供可预测的分界点与端到端加速（最高18%）且保持质量不变的实证结果。

**🔧 技术方法**

使用 LLMLingua 提示压缩算法，配合多种开源 LLM（如 LLaMA、Mistral 等）与三类 GPU，辅以自研延迟分析器与显存监测工具。

**📊 数据集**

采用公开基准数据集覆盖摘要、代码生成与问答任务，累计 30,000 条查询（例如 CNN/DailyMail、CodeXGLUE、SQuAD 等）。

**📈 对比分析**

通过对不同 GPU 类别（高端与通用）和压缩比例下的端到端延迟、速率与质量进行对比，实验表明在匹配条件下可实现 18% 的加速且质量保持统计不变；压缩过程在不匹配时主导导致收益消失，并显示压缩可显著降低显存使用，使得任务可迁移至低端 GPU 仅增加约 0.3 秒延迟。

**⚠️ 局限性**

局限性包括：需要精确匹配 prompt 长度、压缩比例与硬件；压缩开销在某些配置下可能抵消收益；实验主要基于公开 LLM 与 GPU，结果对其他模型或硬件的泛化有限；对极短或极长 prompt 的适用性未充分验证。

---

## 355. Not All Frames Deserve Full Computation: Accelerating Autoregressive Video Generation via Selective Computation and Predictive Extrapolation

**arXiv ID:** 2604.02979 | [PDF](https://arxiv.org/pdf/2604.02979v1)

**作者:** Hanshuai Cui `[一作]` (Beijing Normal University), Wei Zhao `[通讯]` (Shenzhen University of Advanced Technology)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `ba576bd1-e51d-44e8-8077-fc943b333c93` `f86bf285-fd08-4156-973b-6e6481af8fa0` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出一种训练无关的加速框架 SCOPE，针对自回归视频扩散模型在时间和空间上消除冗余计算。

**💡 创新点**

创新点在于：①引入三态调度器（缓存、预测、重算）利用 Taylor 展开在缓存与重算之间实现细粒度的中间状态预测；②提出选择性计算机制，仅在异步 AR 调度的活跃帧区间内执行前向传播，从而削减空间冗余；③结合稳定性控制（误差衰减、最大跳过步限制、重算后重置）保证长距离 AR 推理中的误差不累积。

**🔧 技术方法**

使用的技术包括：流匹配的扩散 ODE、Taylor 预测器（一次、二次差分）、误差传播分析与稳定性约束、基于调度器的活跃帧区间提取、以及轻量级的误差估计特征。

**📊 数据集**

在 MAGI-1（240 帧 480×480）和 SkyReels-V2（257 帧 540P）两个自回归视频扩散模型上进行实验，采用 VBench、LPIPS、SSIM、PSNR 等指标评估质量；通过 NVIDIA A800 80GB GPU 统一跑测。

**📈 对比分析**

与多种训练无关基线（Δ‑DiT、TeaCache、TaylorSeer、FlowCache 等）对比，SCOPE 在 SkyReels-V2 上可达 4.73× 的速度提升，VBench 仅下降 0.06 点；在 MAGI-1 上可达 2.55× 的速度提升，VBench 仅从 81.51% 降至 76.32%，质量保持在同类方法最优水平。

**⚠️ 局限性**

限制在于需要为不同模型手工调节缓存阈值、预测阈值和跳过步限制，导致跨模型的直接迁移受限；另外，预测器的选取虽然对性能影响有限，但在极端噪声水平或长序列下仍可能出现误差积累。

---

## 356. FoE: Forest of Errors Makes the First Solution the Best in Large Reasoning Models

**arXiv ID:** 2604.02967 | [PDF](https://arxiv.org/pdf/2604.02967v1)

**作者:** Kehan Jiang `[一作]` (Peking University), Guojie Song `[通讯]` (Peking University)

**通讯引用:** 6082 | [OpenAlex ID](https://openalex.org/A5088976879)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `64443552-63e0-44b5-906f-d90fe95c5a1b` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对大型推理模型在探索多方案时出现的错误进行深入分析，发现错误会随推理时间同步扩散，形成“错误森林”（Forest of Errors），并提出基于此的自引导高效推理框架（Refining First + Discarding Subs），通过熵与熵方差检测根错误并在关键位置进行干预，同时采用双一致性早停策略剔除后续冗余推理。

**💡 创新点**

①揭示了多方案探索导致的错误累积与“错误森林”现象；②证明了第一方案往往最优；③设计了熵+熵方差驱动的根错误干预与双一致性早停的组合方法，实现了高效推理。

**🔧 技术方法**

熵与熵方差监测、负采样干预（类似CFG）、双一致性检查、早停策略、基于森林结构的错误度量（FS、N/T、D/T、Repro）和概率分支过程理论分析。

**📊 数据集**

数学推理：AIME 2024、AIME 2025、GSM8K、MATH500；科学推理：GPQA-Diamond。使用多种开源 LRM（Qwen、DeepSeek-R1、Llama）和基准模型（8B、32B、70B）。

**📈 对比分析**

与 Vanilla、训练无关方法（DEER、Think or Not、AlphaOne）以及 RL‑based 方法（DAST、RL+LP、GRPO、S‑GRPO）进行对比；在 5 个基准上 Pass@1 提升 0.3–5.6 分（3.2%–19.0%），Token 消耗减少 37.7%–70.4%；在错误森林度量上 41%–68% 的显著下降。

**⚠️ 局限性**

新增的熵干预与双一致性检测会带来额外约 4.6% 的推理延迟，虽然总体上仍能实现加速，但在极低延迟需求场景需进一步优化。

---

## 357. Open-Loop Planning, Closed-Loop Verification: Speculative Verification for VLA

**arXiv ID:** 2604.02965 | [PDF](https://arxiv.org/pdf/2604.02965v1)

**作者:** Zihua Wang `[一作]` (Southeast University), Xiu-Shen Wei `[通讯]` (Southeast University)

**通讯引用:** 7190 | [OpenAlex ID](https://openalex.org/A5066964304)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种 Speculative Verification 框架（SV-VLA），先用大型 Vision‑Language‑Action（VLA）模型低频宏规划生成动作块，再用轻量级验证器高频实时监测并在必要时重新规划，结合开闭环控制实现高效且鲁棒的机器人操作。

**💡 创新点**

创新点在于：①将开环长时程规划与闭环实时验证解耦，形成 Speculative Verification 机制；②利用规划上下文特征与实时观测联合预测参考动作，并通过 L1 差异阈值触发偏差检测与重规划；③在保持长时程推理效率的同时恢复闭环反馈的鲁棒性。

**🔧 技术方法**

采用 OpenVLA‑OFT 等大 VLA 模型作为宏规划器；轻量化视觉编码器（如 ViT‑Tiny）+ 简单全连接 + 线性头作为验证器；使用 L1 回归损失训练验证器；通过阈值 τ 进行偏差检测与重规划；实验中将规划与验证器权重冻结，单独训练验证器。

**📊 数据集**

在 LIBERO 基准（包含 Goal、Spatial、Object 三个子任务集）上进行评估。

**📈 对比分析**

与两种基线（K=8 与 K=64 的动作块）以及 Speculative Decoding 进行对比。SV‑VLA 在 K=64 的块大小下实现 90.9% 的平均成功率，比 K=64 开环 baseline 提升 11.4%，并比 K=8 基线快 2.17×；在每个子任务上也保持显著优势。消融实验表明规划上下文特征、实时观测以及重规划机制都是提升性能的关键。

**⚠️ 局限性**

主要限制：使用固定阈值 τ 进行重规划触发，缺乏自适应调节；验证器仅做偏差检测，无法提供细粒度纠正；实验仅在仿真环境中验证，实际部署仍需进一步验证。

---

## 358. A Network Formation Game for Katz Centrality Maximization: A Resource Allocation Perspective

**arXiv ID:** 2604.03056 | [PDF](https://arxiv.org/pdf/2604.03056v1)

**作者:** Balaji R `[一作]` (Indian Institute of Science), Pavankumar Tallapragada `[通讯]` (Indian Institute of Science)

**通讯引用:** 624 | [OpenAlex ID](https://openalex.org/A5086378642)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

研究了一种基于Katz中心性最大化的网络形成博弈，并给出了其纳什均衡网络的完整理论表征；同时提出了顺序最优回应动态（BRD）并证明其收敛到纳什均衡；在特殊拓扑下进一步解析了均衡网络的稀疏性与层级结构；并通过MATLAB仿真验证理论结果。

**💡 创新点**

①首次将Katz中心性作为收益函数构建网络形成博弈；②证明了“互相强化”性质，所有纳什均衡网络产生相同的中心性向量；③揭示了自环条件下的层级结构与强连通分量的预算与中心性一致性；④给出了收敛的最优回应动态与固定点判定方法。

**🔧 技术方法**

博弈理论（纳什均衡、最优回应）、线性规划求解最优回应、固定点与收敛理论（Banach收缩定理）、Katz中心性矩阵求逆与谱半径分析、MATLAB仿真与可视化。

**📊 数据集**

在论文中仅使用人工构造的10个节点网络（包括自环和完整图两种拓扑）进行仿真；未涉及真实大规模数据集。

**📈 对比分析**

通过数值仿真验证最优回应动态的单调性与收敛性，并对比不同拓扑下的中心性与预算关系，结果与理论推导一致；未与其他算法或基准进行性能比较。

**⚠️ 局限性**

①仅在满足预算小于1且折扣因子取1的简化设定下讨论；②仅考虑受顶层拓扑约束的资源分配，未探讨信息不完全或有限理性；③缺乏对大规模网络或多类中心性指标的实验验证；④仿真样本量有限，未给出统计显著性或鲁棒性分析。

---

## 359. Combining Static Code Analysis and Large Language Models Improves Correctness and Performance of Algorithm Recognition

**arXiv ID:** 2604.03048 | [PDF](https://arxiv.org/pdf/2604.03048v1)

**作者:** Denis Neumüller `[一作]` (Ulm University), Matthias Tichy `[通讯]` (Ulm University)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

本文评估并结合大型语言模型（LLM）与静态代码分析技术，用于自动识别Java源代码中的算法实现；

**💡 创新点**

创新点在于提出一种混合式方法，将LLM与预过滤器（关键词与结构模式）相结合，显著降低LLM调用次数、提升识别精度并减少运行时；

**🔧 技术方法**

使用的技术包括LLM（GPT‑4o mini、Llama 3.1 Instruct、Mixtral 8×22B Instruct）、多种提示策略（基线、评分、上下文学习、链式推理）以及自定义的正则与AST图匹配过滤器；

**📊 数据集**

实验数据集为BCEval（包含多种真实世界算法实现的Java方法，涵盖Prime Factors、GCD、Fibonacci、Palindrome、Bubble Sort、Binary Search、Transpose Matrix等）；

**📈 对比分析**

通过宏平均F1-score与运行时间比较，发现两例正样本上下文学习可将F1提升约7–8个百分点；与过滤器组合后LLM调用量下降72–97%，F1最高可达81%，而单独LLM或单独过滤器的性能较弱；

**⚠️ 局限性**

局限性包括仅测试Java语言且算法限定在单方法实现；过滤器模式需人工编写且可能不适用于所有实现变体；LLM对命名信息仍有一定依赖，且实验受限于BCEval的标签质量与覆盖范围；

---

## 360. Behavior-Constrained Reinforcement Learning with Receding-Horizon Credit Assignment for High-Performance Control

**arXiv ID:** 2604.03023 | [PDF](https://arxiv.org/pdf/2604.03023v1)

**作者:** Siwei Ju `[一作]` (TU Darmstadt), Jan Peters `[通讯]` (TU Darmstadt)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `a4b10f5d-130b-4e77-9367-6469ec621899` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

在高保真模拟环境中，结合行为约束与递归短期信用分配，训练出能够在多种车辆设置下实现高性能且具有人类驾驶风格的强化学习策略。

**💡 创新点**

创新点在于：①提出行为约束式RL框架，将模仿误差作为约束实现任务性能与驾驶风格的平衡；②引入递归短期信用分配（Receding-Horizon Credit Assignment）与轻量级辅助预测器，提高样本效率和近端信用归属；③在赛车控制中使用概率运动原语（ProMP）与目标轨迹条件化，提升对不同车辆设置的适应性。

**🔧 技术方法**

使用的技术包括：基于PPO的强化学习、行为克隆/DAgger与GAIL的模仿学习、概率运动原语（ProMP）、辅助预测任务、动态权重调度、以及高保真物理模拟与Stewart平台仿真。

**📊 数据集**

数据集：来自专业车手在高保真赛车模拟器（Stewart平台）下的演示数据，涵盖三种不同车辆设置（分别表现出欠转、过转和平衡），每种设置训练多达150圈（取前100圈进行分析）。

**📈 对比分析**

对比方法：与传统BC、纯RL、以及混合式SBRL基线进行对比。性能指标为圈速分布、方差（稳定性）以及与车手主观反馈的一致性。实验显示，所提出方法在保持最快圈速的同时显著降低方差，且与车手偏好高度一致，表现优于基线。

**⚠️ 局限性**

局限性：①仅在仿真环境验证，真实道路或不同车辆硬件的迁移性能未知；②对专业演示数据的依赖，若演示质量不足或覆盖不全，性能受限；③算法训练成本高，需要强大的计算资源和长时间仿真。

---

## 361. R2-Write: Reflection and Revision for Open-Ended Writing with Deep Reasoning

**arXiv ID:** 2604.03004 | [PDF](https://arxiv.org/pdf/2604.03004v1)

**作者:** Wanlong Liu `[一作]` (Alibaba Group), Ming Yan `[通讯]` (Alibaba Group)

**通讯引用:** 2947 | [OpenAlex ID](https://openalex.org/A5079627710)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

通过在写作任务中加入深度反思与修订机制，提升大型语言模型的开放式写作能力

**💡 创新点**

提出自动化框架R2-Write，利用写手-评审交互生成反思与修订轨迹，并设计过程奖励监督，首次把反思/修订模式系统化引入写作推理

**🔧 技术方法**

写手-评审协作生成数据、监督式微调（SFT）+强化学习（PPO）+自定义对比奖励、过程奖励机制

**📊 数据集**

使用多样化写作数据：Creative Writing（DeepWriting 20K）、报告生成（10K深度研究查询）以及公开写作与翻译基准（WritingBench、HelloBench、DeepResearch‑Bench/Gym、DiscoX）

**📈 对比分析**

与多种SFT/RL基线对比，R2-Write在创意写作、专业报告和中英翻译基准上均显著提升（例如WritingBench得分从78.2提升至约83.8，DeepResearch‑Gym提升约10%），且过程奖励还能减少思考轨迹长度20%

**⚠️ 局限性**

主要限制包括：仍需依赖评审LLM构造奖励，难以在极端复杂或跨领域写作中完全自适应；对生成的反思/修订模式缺乏统一理论解释

---

## 362. Rendering Multi-Human and Multi-Object with 3D Gaussian Splatting

**arXiv ID:** 2604.02996 | [PDF](https://arxiv.org/pdf/2604.02996v1)

**作者:** Weiquan Wang `[一作]` (Zhejiang University), Long Chen `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 96706 | [OpenAlex ID](https://openalex.org/A5100333572)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `3f18e8e3-0266-457c-8567-9039b6d2394d` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出MM-GS框架，用分层图网络实现稀疏视角下多人体多物体交互场景的高保真渲染。

**💡 创新点**

引入实例级多视角融合与场景级实例交互模块，分别解决视角一致性和交互依赖问题，形成可在实时渲染下完成的三阶段优化。

**🔧 技术方法**

基于3D高斯散点（3D Gaussian Splatting）+ SMPL人体模板 + 线性混合皮肤（LBS）+ 2D CNN特征提升 + 图注意力网络（GAT）+ 多层MLP解码器。

**📊 数据集**

在HOI‑M³和CORE4D‑Real这两个多人体多物体交互数据集上进行评估。

**📈 对比分析**

对比改进版NeuralHOIFVV‑MM与GTU‑MM两大基线，使用PSNR/SSIM/LPIPS评估，MM‑GS在所有指标上显著优于基线，帧率保持160+ FPS。

**⚠️ 局限性**

仅适用于已知对象位姿与2D掩码的场景，难以直接应用于野外无标注条件下的机器人部署。

---

## 363. MECO: A Multimodal Dataset for Emotion and Cognitive Understanding in Older Adults

**arXiv ID:** 2604.03050 | [PDF](https://arxiv.org/pdf/2604.03050v1)

**作者:** Hongbin Chen `[一作]` (Nanjing Medical University), Wentao Xiang `[通讯]` (Nanjing Medical University)

**通讯引用:** 1612 | [OpenAlex ID](https://openalex.org/A5102956151)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `5e20d1ff-779f-4b7a-be75-8663ee04d94e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `e15e3743-5ee0-4d5f-813d-d146868082fc` `79276348-11e0-48e3-84bc-7ec231d0171c` `109c2b71-d051-425c-831f-0c544c24280d`

**🎯 论文内容**

本文构建了MECO数据集，收集42名老年人（含MCI）在社区环境下的情绪诱导视频实验，记录视频、音频、EEG、ECG等多模态数据并同步标注情绪（情绪类别、valence、arousal）与认知（MMSE）得分。

**💡 创新点**

创新点在于首次为老年人提供同步情绪与认知标签的多模态数据，并结合EEG/ECG等生理信号，填补了老年情绪认知研究中缺乏专门数据的空白；同时提供了基准实验和跨模态融合的初步结果。

**🔧 技术方法**

技术方法主要是信号预处理、特征提取（视频：AU、HP、EG、深度特征；EEG：DE、PSD、HFD、SE；ECG：TD、HFD、SE），随后通过GRU捕获时序信息，最终用MLP进行分类/回归。

**📊 数据集**

使用的数据集是自建的MECO，约38小时多模态记录，30,592个同步样本，包含情绪类别、valence、arousal、MMSE等标签。

**📈 对比分析**

与单模态相比，双模态（如视频+EEG或视频+ECG）在多种情绪预测任务（UAR/WAR）和认知分类/回归（ACC/F1/CCC/MAE）上获得了更高的性能；但在跨受试者（SI）评估中，多模态融合效果不明显，甚至略低于视频单模态，说明个体差异大。

**⚠️ 局限性**

局限性包括样本量仅42人，未使用音频模态；实验仅为刺激诱导数据，缺乏自发性情绪场景；多模态融合采用简单拼接，未能充分利用跨模态互补性，未来需扩充样本、加入音频及更高级的融合策略。

---

## 364. Comparing the Impact of Pedagogy-Informed Custom and General-Purpose GAI Chatbots on Students' Science Problem-Solving Processes and Performance Using Heterogeneous Interaction Network Analysis

**arXiv ID:** 2604.03022 | [PDF](https://arxiv.org/pdf/2604.03022v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39`

---

## 365. GenSmoke-GS: A Multi-Stage Method for Novel View Synthesis from Smoke-Degraded Images Using a Generative Model

**arXiv ID:** 2604.03039 | [PDF](https://arxiv.org/pdf/2604.03039v1)

**作者:** Qida Cao `[一作]` (Hangzhou Dianzi University), Jun Yu `[通讯]` (Harbin Institute of Technology)

**通讯引用:** 13730 | [OpenAlex ID](https://openalex.org/A5050817770)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `25d64835-ec5b-425b-899d-a6e1e6fecabd` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出一种多阶段管线，用预处理恢复、去雾、MLLM增强、3D Gaussian Splatting 优化与多次平均，专门针对烟雾退化图像生成清晰的视角；

**💡 创新点**

在增强阶段通过提示词限制结构变化，以保持跨视图一致性，并通过多次优化平均来抑制局部不稳定；

**🔧 技术方法**

ConvIR-UDPNet、Dark Channel Prior、GPT-Image-1.5（MLLM）、3DGS‑MCMC（加速版FasterGS）、多次运行平均；

**📊 数据集**

RealX3D benchmark（NTIRE 2026 3DRR Challenge Track 2）；

**📈 对比分析**

与3DGS、I2‑NeRF、SeaSplat、SeaThru‑NeRF基线对比，PSNR提升至20.21（vs 11.54），SSIM提升至0.729（vs 0.597），LPIPS下降至0.446（vs 0.705），在所有评估指标上均获第一；

**⚠️ 局限性**

对极端退化场景（如Shirohana）仍有一定挑战，MLLM增强需大量算力且多次平均导致计算成本高；

---

## 366. Learning Contractive Integral Operators with Fredholm Integral Neural Operators

**arXiv ID:** 2604.03034 | [PDF](https://arxiv.org/pdf/2604.03034v1)

**作者:** Kyriakos C. Georgiou `[一作]` (University of Naples Federico II), Athanasios N. Yannacopoulos `[通讯]` (University of Naples Federico II)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文基于 Fredholm 神经网络（FNN）框架，提出了 Fredholm 积分神经算子（FREDINOs）来学习任意维度下的收敛（非扩张）积分算子及其对应的解算子，并将该方法扩展到非线性欧拉-拉普拉斯 PDE 的边界积分形式，验证了在多维线性与非线性 Fredholm 积分方程以及 2D 半线性椭圆 PDE 上的高精度近似与可解释性。

**💡 创新点**

创新点：① 证明了 FREDINOs 对收敛积分算子（线性与非线性）的通用逼近性质，并保证学习到的算子严格收敛；② 将数值分析中的 Krasnosel’skii–Mann 固定点迭代与深度网络结构深度绑定，形成可解释的“白盒”网络；③ 通过正则化与 Sobol 随机采样在训练过程中强制保持算子收敛性；④ 引入循环 Fredholm NN 处理非线性积分方程；⑤ 将 FREDINOs 迁移至 PDE 的边界积分表述，实现对非线性椭圆 PDE 的直接算子学习。

**🔧 技术方法**

技术手段包括：
- Fredholm Neural Network 架构（固定权重/偏置与积分核/非线性关联）
- Krasnosel’skii–Mann 与 Picard 固定点迭代的网络层级映射
- Sobol 低差异序列用于高维积分采样
- 交替最小化（alternating minimization）训练循环以解决非线性核+非线性函数的耦合
- L2 权重正则化与单位球指示函数强制收敛性
- 解析证明与强壮的误差界定
- Python + GPU 训练实现。

**📊 数据集**

数据集：
- 线性 Fredholm 方程：使用已知正弦/高斯核的 1D 与 10D 版本，生成 600/200 条训练样本，10D 采用 1024 Sobol 点；
- 非线性 Fredholm 方程：使用高斯核与双峰高斯非线性，500 条训练样本；
- 椭圆 PDE：二维单位圆，生成 200 条边界条件与对应解，20 条测试边界；
- 所有训练样本均为合成，输入函数 g 通过高斯混合、立方多项式与正弦组合生成，保持零均值与标准化。

**📈 对比分析**

评估方式：
- 在训练网格与未见网格上计算相对 L1、L2、L∞ 误差；
- 对每个测试样本统计中位数与 10%–90% 分位数；
- 监测网络层级间残差 ‖f̂^(k)−f̂^(k−1)‖ 以验证收敛性；
- 结果显示：
  * 1D 线性：误差 ~4–7×10⁻⁴；
  * 10D 线性：误差 ~1×10⁻³；
  * 非线性 1D/10D：误差 ~10⁻³–10⁻²；
  * 2D 半线性 PDE：误差 ~5–8×10⁻³；
- 这些误差均低于传统数值方法的粗粒度误差，并显示了良好的收敛与可解释性。

**⚠️ 局限性**

局限性：
- 需要先验知道积分方程结构和收敛性，限制了对未知方程的直接应用；
- 正则化与单一收敛约束可能削弱模型的表达能力，尤其在复杂核或强非线性时；
- 高维 PDE（d≥3）中基函数的奇异性与维度灾难尚未彻底解决；
- 训练依赖大量高质量网格与 Sobol 采样，对 GPU 计算资源要求较高；
- 所有实验均基于合成数据，缺乏对真实科学计算数据的验证。

---

## 367. BugForge: Constructing and Utilizing DBMS Bug Repository to Enhance DBMS Testing

**arXiv ID:** 2604.03024 | [PDF](https://arxiv.org/pdf/2604.03024v1)

**作者:** Dawei Li `[一作]` (Beihang University), Yu Jiang `[通讯]` (Tsinghua University)

**通讯引用:** 25883 | [OpenAlex ID](https://openalex.org/A5065179368)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了一个完整的框架（BugForge），用于从真实DBMS bug报告中自动构建统一的bug仓库，并将其转化为高质量可执行的测试用例；

**💡 创新点**

创新点在于：①使用语法感知的预处理与增量检索相结合的LLM驱动提取原始PoC；②引入反馈驱动与语义约束并行的适配策略，既保证可执行性又保留语义丰富性；③将构建的仓库直接用于fuzz、回归和跨DBMS测试，显著提升覆盖率与缺陷发现率；

**🔧 技术方法**

技术包括：语法感知的分块提取算法、检索增强生成（RAG）的LLM框架（Gemini-2.5），反馈驱动的错误诊断与自适应修复，语义锚点约束的自适应策略，容器化的执行与环境隔离；

**📊 数据集**

数据集为从四大主流开源DBMS（MySQL、MariaDB、PostgreSQL、MonetDB）官方bug跟踪系统及社区论坛收集的37,632条bug报告，提取到35,530条原始PoC，并进一步生成33,598条高质量测试用例；

**📈 对比分析**

与SQLancer、SQLsmith等现有DBMS测试工具比较，BugForge在分支覆盖率上提升约62%（MySQL）、57%（MariaDB）、54%（PostgreSQL）、80%（MonetDB），在缺陷发现上总计识别18个新bug（相较于SQLancer的11、SQLsmith的6），表明方法在覆盖率和bug发现率上均有显著提升；

**⚠️ 局限性**

局限性包括：依赖LLM的准确性和可解释性，处理极其古老或语法极端多变的报告时仍可能失败；适配策略在保持语义与可执行性之间仍存在权衡，极高语义约束可能降低可执行率；对非公开或非标准SQL方言的DBMS扩展仍需额外工程。

---

## 368. Generating DDPM-based Samples from Tilted Distributions

**arXiv ID:** 2604.03015 | [PDF](https://arxiv.org/pdf/2604.03015v1)

**作者:** Himadri Mandal `[一作]` (Indian Statistical Institute), Sandeep Juneja `[通讯]` (Ashoka University)

**通讯引用:** 1780 | [OpenAlex ID](https://openalex.org/A5109269131)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `67630363-6be0-4f51-ab05-7198250671a5` `ba576bd1-e51d-44e8-8077-fc943b333c93` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了在已知高维分布的样本上，通过权重重采样得到指数倾斜分布，并使用扩散采样进一步生成样本，证明了该流程的理论精度；

**💡 创新点**

提出了插件估计器的渐近最小化性质，并给出了Wasserstein与TV误差的精确上界，首次将权重重采样与扩散采样的误差理论结合；

**🔧 技术方法**

采用了指数倾斜理论、最小化风险与KL距离、Wasserstein距离及其耦合分析、Lipschitz估计误差与扩散逆过程的数值分析；

**📊 数据集**

使用了人工合成的有界相关分布（通过随机矩阵变换得到的10维和50维数据），并在此基础上进行倾斜实验；

**📈 对比分析**

与传统的重采样、扩散后处理（DPS、LGD-MC）相比，实验显示权重重采样+扩散方法在不同倾斜参数下的采样误差最小，逼近真实倾斜分布；

**⚠️ 局限性**

存在指数级增长的常数（C_w、V），导致样本复杂度上界较弱；仅考虑了有界或特定尾部分布，未解决非有界情况和其他扩散机制的理论保证问题。

---

## 369. UnrealVis: A Testing Laboratory of Optimization Techniques in Unreal Engine for Scientific Visualization

**arXiv ID:** 2604.02980 | [PDF](https://arxiv.org/pdf/2604.02980v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e`

---

## 370. User-Aware Conditional Generative Total Correlation Learning for Multi-Modal Recommendation

**arXiv ID:** 2604.03014 | [PDF](https://arxiv.org/pdf/2604.03014v1)

**作者:** Jing Du `[一作]` (University of New South Wales), Flora. D. Salim `[通讯]`

**关键词:** `b9e48b6f-9d3b-41c5-a0bd-841e9445d871` `a2602d71-93ab-4bad-974b-672788df8193` `3f18e8e3-0266-457c-8567-9039b6d2394d` `ba576bd1-e51d-44e8-8077-fc943b333c93` `9ce7179e-700c-4310-ac2b-91df50ded46e` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种基于用户感知的扩散式内容过滤和总相关最大化的多模态推荐框架（GTC），实现个性化与跨模态依赖的统一建模。

**💡 创新点**

创新点在于：①通过交互引导的扩散模型实现用户条件下的内容特征去噪，解决传统方法对特征相关性“一刀切”的假设；②利用可计算的总相关下界替代仅有成对对齐的对比损失，捕捉高阶模态互相依赖。

**🔧 技术方法**

采用LightGCN进行图传播、U‑Net结构的交互引导扩散网络、InfoNCE对总相关的近似优化、余弦相似度门控融合以及BPR排序损失。

**📊 数据集**

在Amazon Review公开数据集的Sports、Baby和Cellphone三个领域进行实验。

**📈 对比分析**

与10种现有多模态推荐基线（包括融合式与分离式方法）进行对比，GTC在NDCG@5、NDCG@10、NDCG@20以及Recall/ MAP等指标上均显著提升，最高可达28.30%的NDCG@5增幅，表现稳健。

**⚠️ 局限性**

主要局限包括：①扩散过程需要多步迭代，计算成本较高；②对超参数（噪声步长、权重系数）敏感；③仅考虑静态视觉与文本内容，未覆盖动态或结构化信息；④对极端稀疏数据的鲁棒性尚待进一步验证。

---

## 371. Explicit Time-Frequency Dynamics for Skeleton-Based Gait Recognition

**arXiv ID:** 2604.03002 | [PDF](https://arxiv.org/pdf/2604.03002v1)

**作者:** Seoyeon Ko `[一作]` (Ewha Womans University), Junhyug Noh `[通讯]` (Ewha Womans University)

**通讯引用:** 888 | [OpenAlex ID](https://openalex.org/A5088003950)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a244defd-9560-426b-b1b1-f78ebb2b7bf9` `6c82a482-f376-4869-8a0b-a802c9d4d3d4` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `9ce7179e-700c-4310-ac2b-91df50ded46e` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种可插拔的波形特征流，利用连续小波变换（CWT）将关节速度信号转化为多尺度时频图谱，再通过轻量化多尺度CNN提取动态特征，并与现有骨架网络融合，用于步态识别。

**💡 创新点**

创新点包括①将连续小波变换应用于骨架关节速度，显式捕捉多频段运动动态；②设计轻量化多尺度CNN以学习这些时频特征；③实现了与骨架骨干网络的无缝对接，既不改造骨干结构也不需要额外监督，显著提升了在外观变化（携带包、穿大衣）下的鲁棒性。

**🔧 技术方法**

主要技术手段包括：连续小波变换（CWT）、多尺度时频图谱处理、轻量化多尺度CNN、三元组损失、特征级融合、姿态估计（HRNet）、Adam优化与One-Cycle学习率调度。

**📊 数据集**

在CASIA-B步态数据库上进行实验，使用标准训练/测试划分（前74人训练，后50人测试）。

**📈 对比分析**

与多种骨架方法（GaitGraph、GaitFormer、GaitMixer）以及外观方法（GaitNet、GaitSet、3DLocal）进行对比。附加波形流后，GaitMixer在CASIA-B的平均Rank‑1从88.3%提升到89.7%，尤其在携带包和穿大衣条件下分别提升约3–4个百分点；在大衣（CL）条件下，骨架方法已超越主流外观方法。

**⚠️ 局限性**

局限性：1）仍依赖姿态估计的质量；2）实验仅在单一数据集验证，未评估跨数据集/跨传感器迁移；3）小波参数固定，未实现端到端学习；4）在极端遮挡或姿态误差大时可能受限。

---

## 372. InfoSeeker: A Scalable Hierarchical Parallel Agent Framework for Web Information Seeking

**arXiv ID:** 2604.02971 | [PDF](https://arxiv.org/pdf/2604.02971v1)

**作者:** Ka Yiu Lee `[一作]` (Huawei Noah's Ark Lab), Jun Wang `[通讯]` (University College London)

**通讯引用:** 37541 | [OpenAlex ID](https://openalex.org/A5084169778)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `67630363-6be0-4f51-ab05-7198250671a5` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出 InfoSeeker——一种基于“近分解性”原理的三层层次式代理框架（Host-Manager-Worker），用于大规模信息合成任务。

**💡 创新点**

创新点在于通过层次化的上下文隔离与 MapReduce 样式聚合，实现推理深度与执行宽度独立扩展，显著降低上下文饱和、误差累计和推理延迟。

**🔧 技术方法**

采用三层架构、MCP（Model Context Protocol）实现工具交互隔离、并行 Worker 并发执行、动态任务拆分与结果聚合，并结合 gpt‑5.1/gpt‑5‑mini、Firecrawl、Playwright 等工具。

**📊 数据集**

使用两个互补基准：WideSearch（英文信息合成）和 BrowseComp‑zh（中文浏览推理）。

**📈 对比分析**

与 Gemini Deep Research、OpenAI Deep Research、Claude Sonnet 等最新商业/开源代理系统对比，InfoSeeker 在 WideSearch 的成功率提升约 8.4%（相较 5.1% 的强基线提升 64%），在 BrowseComp‑zh 的准确率达到 52.9%（高于 42.9% 的商业基线）。同时在推理速度上实现 3–5 倍加速，延迟从 911 秒降至 162 秒（17 个 Worker）。

**⚠️ 局限性**

局限性包括：对 API 可用性、速率限制与成本敏感；依赖手工调优的提示与强大 LLM；缺乏自动化任务拆分与协调策略，模型泛化性待进一步验证。

---

## 373. Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection

**arXiv ID:** 2604.02966 | [PDF](https://arxiv.org/pdf/2604.02966v1)

**作者:** Wenhao Li `[一作]` (Beihang University), Jiaxin Chen `[通讯]` (Beihang University)

**通讯引用:** 8493 | [OpenAlex ID](https://openalex.org/A5100360561)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出UAVGen框架，利用视觉原型条件扩散模型和焦点区域增强数据管线提升无人机图像生成与检测。

**💡 创新点**

创新点在于通过视觉原型筛选构建高质量布局条件，结合多源条件编码和焦点区域生成，解决小目标边界伪影与标签不一致问题。

**🔧 技术方法**

采用扩散模型（FLUX + ControlNet）、视觉原型选择、文本嵌入、傅里叶位置编码、焦点区域聚类、标签细化等技术。

**📊 数据集**

使用VisDrone和UAVDT两个公开无人机目标检测数据集。

**📈 对比分析**

与GLIGEN、Geodiffusion、AeroGen等方法比较，FID显著下降，平均精度提升1-2%，在多尺寸指标上均优于现有方法。

**⚠️ 局限性**

局限在于对极端尺度变化、视角高度差异的适应仍有限，且需要进一步验证在更大规模或真实环境下的鲁棒性。

---

## 374. TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing

**arXiv ID:** 2604.03143 | [PDF](https://arxiv.org/pdf/2604.03143v1)

**作者:** Zhuohang Bian `[一作]` (Peking University), Youwei Zhuo `[通讯]` (Peking University)

**通讯引用:** 1158 | [OpenAlex ID](https://openalex.org/A5051409603)

**关键词:** `d0c287c2-ddf5-4cc2-9cd5-c6e171da6e62` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

设计了一套面向多智能体大语言模型（LLM）推理的服务系统，通过在同步回合层面共享 KV 缓存来显著提升并发代理数和减少内存占用。

**💡 创新点**

创新点在于：① 将回合级共享视为重用单元，提出集体重用（Collective Reuse）一次性完成 RoPE 旋转与重要位置检测；② 采用 Master‑Mirror 存储结构和块稀疏差分编码，实现 11–17× 的压缩；③ 在 GPU 传输链路中融合差分恢复（Fused Diff Restore），避免额外重建开销。

**🔧 技术方法**

核心技术包括：Round‑Aware Prompt 接口、Collective Reuse 算法、Diff‑Aware Storage（Master‑Mirror + block‑sparse diff）、Fused Diff Restore、以及与 vLLM/CacheBlend 的集成。

**📊 数据集**

在 A100‑80GB GPU 上，使用 Qwen2.5‑7B 与 Qwen2.5‑14B 两大模型；评测采用来自社交模拟框架的两组真实工作负载（例如 OpenClaw、MoltBook）。

**📈 对比分析**

相较于 vLLM（前缀缓存）、CacheBlend（无集体重用）以及 CacheBlend（全重用）三种基线，系统实现了：① 最高可达 2.3× 的端到端延迟提升；② 94% 的 KV 缓存存储压缩；③ 在同等延迟阈值下可支持 2.7× 更多并发代理；④ 预填充阶段最高 1.9× 的加速。

**⚠️ 局限性**

局限性包括：依赖回合同步模式，若代理交互不满足“共享输出 + 私有历史”结构则收益有限；对小规模代理集群时集体重用与压缩收益不明显；在极大模型（14B 及以上）下，块差分量可能增大导致压缩率下降；与基础 PIC 方法相关的微小数值差异可能在极端随机采样情形下影响输出一致性。

---

## 375. An Independent Safety Evaluation of Kimi K2.5

**arXiv ID:** 2604.03121 | [PDF](https://arxiv.org/pdf/2604.03121v1)

**作者:** Zheng-Xin Yong `[一作]` (Constellation), Michael L. Chen `[通讯]` (University Of Oxford)

**通讯引用:** 390 | [OpenAlex ID](https://openalex.org/A5100705944)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Moonshot AI发布的开源大模型 Kimi K2.5 进行全方位安全评估，涵盖 CBRNE 误用、网络安全、模型对齐、政治偏见/审查以及有害行为拒绝率。

**💡 创新点**

首次系统化评估开源高性能 LLM 的安全风险，揭示缺乏安全评估的开放权重模型在工具启用代理环境下的潜在滥用空间。

**🔧 技术方法**

使用自动化评估工具与框架（Petri 2.0、ControlArena、FORTRESS、Cybench、VCT、ABC‑Bench、LAB‑Bench、EVMBench、DFIR‑Metric、AgentHarm、BQA、BBQ 等），结合多工具交互与链式推理监测。

**📊 数据集**

评估所用数据集包括 ABC‑Bench、VCT、LAB‑Bench、CyberGym、EVMBench、DFIR‑Metric、ControlArena、AgentHarm、BQA、BBQ、DECCP、Borderlands 等公开基准。

**📈 对比分析**

与闭源前沿模型 GPT‑5.2、Claude Opus 4.5 进行对比，Kimi 在绝大多数能力指标上相当或略低，但在拒绝率、对齐合规和安全护栏薄弱方面表现更差，风险水平更高。

**⚠️ 局限性**

限制：评估仅覆盖文本交互，未涉及多模态、代理群体、模型重定向/微调攻击；仅使用英文和中文测试；缺乏机制层面的解释与可解释性分析；未评估视觉嵌入式 jailbreak 等潜在风险。

---

## 376. PAFT: Preservation Aware Fine-Tuning for Minimal-Edit Program Repair

**arXiv ID:** 2604.03113 | [PDF](https://arxiv.org/pdf/2604.03113v1)

**作者:** Boyang Yang `[一作]` (Yanshan University), Haoye Tain `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一种保留感知微调（PAFT），通过对比buggy与fixed代码的 token 对齐生成保留信号，训练 LLM 在修复时只在故障区最小编辑，保持其余稳定代码不变。

**💡 创新点**

创新点在于：① 用 deterministic token‑level 对齐自动生成保留标签；② 在微调中对保留 token 加权、使用全序列掩码和编辑难度课程；③ 将编辑范围约束直接嵌入训练，而非依赖推理时搜索或额外偏好标签。

**🔧 技术方法**

技术方法包括 QLoRA 低秩微调、4‑bit NF4 量化、token‑level 保留加权、全序列 mask、编辑难度课程、指令/聊天模板、以及 AED/CCR 等评估指标。

**📊 数据集**

使用 TutorLLMCode（1535 对 bug‑fix 样本）作为训练数据，并在 Defects4J 与 HumanEval‑Java 两个 Java 修复基准上进行评估。

**📈 对比分析**

与基线（原始 LLM、标准监督微调、Prompting、RepairLLaMA、AdaPatcher 等）在 pass@k、AED、CCR 进行对比，PAFT 在 Defects4J 与 HumanEval‑Java 上 pass@1 提升最高 65.6%，AED 降低最高 32.6%，并优于 AdaPatcher 的正确性‑局部性折衷。

**⚠️ 局限性**

局限性包括：仅评估单文件 Java 修复，未验证跨文件/跨语言效果；token‑level 对齐可能低估语义等价重写；评估仍受测试套件完整性和 AED/CCR 等指标的限制，安全性与完整性仍需进一步验证。

---

## 377. Multi-Aspect Knowledge Distillation for Language Model with Low-rank Factorization

**arXiv ID:** 2604.03110 | [PDF](https://arxiv.org/pdf/2604.03110v1)

**作者:** Zihe Liu `[一作]` (Key Laboratory of Big Data and Artificial Intelligence in Transportation), Kaiyu Huang `[通讯]` (Key Laboratory of Big Data and Artificial Intelligence in Transportation)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

研究并提出一种多维度知识蒸馏方法（MaKD），通过层级蒸馏将 Transformer 的细粒度矩阵信息和层级信息融合，以实现高效的模型压缩。

**💡 创新点**

创新点在于：① 在细粒度层面对多头注意力（MHA）与前馈网络（FNN）的映射矩阵进行对齐蒸馏；② 采用低秩 SVD 分解进行学生模型初始化，减少跨层映射误差；③ 将矩阵蒸馏、层蒸馏与模型蒸馏三种蒸馏目标层级组合，形成完整的多维度蒸馏框架。

**🔧 技术方法**

主要技术包括：SVD 低秩分解、矩阵蒸馏（MHA 与 FNN）、层蒸馏（对齐注意力分布与隐藏状态）、模型蒸馏（soft‑label KL 损失）、混合梯度优化（AdamW）以及多 GPU 混合精度训练。

**📊 数据集**

训练使用英文 Wikipedia 与 BooksCorpus 进行预训练；下游评测数据集包括 GLUE（CoLA、SST‑2、MRPC、STS‑B、QQP、MNLI、QNLI、RTE）、SQuAD 1.1/2.0、以及指令跟随任务集（Dolly‑15k、Self‑Instruct、Vicuna）。

**📈 对比分析**

与 DistilBERT、TinyBERT、MiniLMv2 等基线相比，MaKD 在相同参数预算下（如 24M、67M）在 GLUE、SQuAD 以及指令跟随任务中均获得更高的准确率或 ROUGE 分数；尤其在 24M 小模型上已超过现有四层模型的表现，显示出显著的压缩效果。

**⚠️ 局限性**

主要限制：多维度蒸馏需要更长的训练时间；实验未覆盖极大规模 LLM（>10B）或极低压缩率（<10%）的情况；在小样本任务（如 RTE）上的表现略逊于部分基线，可能与数据规模相关。

---

## 378. A Data-Centric Vision Transformer Baseline for SAR Sea Ice Classification

**arXiv ID:** 2604.03094 | [PDF](https://arxiv.org/pdf/2604.03094v1)

**作者:** David Mike-Ewewie `[一作]` (University of Texas Permian Basin), Priyanka Kumar `[通讯]` (University of Texas Permian Basin)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文建立了一个基于完整分辨率Sentinel-1 SAR影像的Sea Ice分类基线，并评估了三种Vision Transformer配置在极端类别不平衡下的性能。

**💡 创新点**

创新点在于提出数据中心化的SAR基线，采用泄漏安全的分块切分、SIGRID-3 Stage-of-development标签和动态归一化，并比较了交叉熵、加权交叉熵与焦点损失的效果，证明焦点损失在少数类多年冰上的精度更优。

**🔧 技术方法**

使用Vision Transformer（ViT-Base、ViT-Large）、交叉熵、加权交叉熵和焦点损失训练；还采用了全分辨率Extra Wide SAR输入、泄漏意识的分块划分以及训练集特定的均值方差归一化。

**📊 数据集**

使用AI4Arctic/ASIP Sea Ice Dataset v2，共461个2018-2019年的Sentinel-1 EW场景与格陵兰冰川专家图表匹配。

**📈 对比分析**

在泄漏安全的留出测试集上，ViT-Large + Focal Loss取得69.6%的总体准确率、68.8%的加权F1分数，并在少数类多年冰上实现83.9%的精度，优于ViT-Base（CE）与ViT-Base（W-CE）配置。

**⚠️ 局限性**

局限在于仅使用SAR单模态，未解决不同冰类之间纹理相似导致的混淆，且缺乏对光学、热或气象信息的多模态融合验证。

---

## 379. On Data-Driven Koopman Representations of Nonlinear Delay Differential Equations

**arXiv ID:** 2604.03086 | [PDF](https://arxiv.org/pdf/2604.03086v1)

**作者:** Santosh Mohan Rajkumar `[一作]` (Ohio State University), Debdipta Goswami `[通讯]` (Ohio State University)

**通讯引用:** 330 | [OpenAlex ID](https://openalex.org/A5044260875)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `14d48e9d-0069-4ad9-996a-1d5968216998` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

本文针对延迟微分方程提出了一种基于历史离散化与重构的有限维 Koopman 学习框架。

**💡 创新点**

创新点在于通过解析误差分解得到可计算的离散化、核插值与回归三项误差上界，并给出状态重构的保证。

**🔧 技术方法**

主要技术包括历史采样/重构、Wendland RBF 的核扩展动态模式分解 (kEDMD) 与可解释的误差分析。

**📊 数据集**

实验数据来源于对两类非线性 DDE（Hill 型标量 DDE 与肿瘤-免疫模型）在不同延迟、轨迹数和网格分辨率下的仿真生成的轨迹样本。

**📈 对比分析**

通过与真实轨迹的均方误差比较，结果显示随历史分辨率增大、样本量增多和核参数调优误差显著下降，证明方法的有效性。

**⚠️ 局限性**

局限性包括对前向不变集的假设、对大规模数据时核矩阵求逆的数值不稳定、以及尚未完成闭环控制合成。

---

## 380. HistMSO: A Logic for Reasoning about Consistency Models with MONA

**arXiv ID:** 2604.03085 | [PDF](https://arxiv.org/pdf/2604.03085v1)

**作者:** Isabelle Coget `[一作]` (Institut Polytechnique de Paris), Étienne Lozes `[通讯]` (Université Côte d'Azur)

**通讯引用:** 594 | [OpenAlex ID](https://openalex.org/A5011145361)

**关键词:** `09ec487f-4c5c-4ed6-960d-c9fa93fddb0c`

**🎯 论文内容**

提出 HistMSO 这一单值二阶逻辑，用于描述复制数据系统中的历史和抽象执行，并给出将 HistMSO 公式转换为 MONA 词法形式的译码方法，从而实现对一致性模型的自动推理。

**💡 创新点**

创新点在于：①构造了能够表达 42 个一致性模型中 39 个的通用逻辑；②证明 HistMSO 对有限历史及满足实时仲裁与 k‑瞬时可见性的抽象执行是可判定的；③将持续时间与并发约束编码为词语，实现了利用 MONA 进行自动化推理；④给出了历史图的剪切宽度上界（平方于进程数），为树分解与可判定性提供了新的视角。

**🔧 技术方法**

核心技术包括：单值二阶逻辑 (MSO) 的语法与语义定义；历史与抽象执行的数学模型；将持续时间与并发信息映射为词（snapshot）序列；对 HistMSO 公式的 MONA 译码（利用词的位向量编码操作属性与时间关系）；利用 MONA 的判定算法进行满足性与模型检查；以及剪切宽度与树宽度的图论分析。

**📊 数据集**

文中未使用传统机器学习或实验数据集，而是以理论构造的有限历史/ω‑历史、以及假设的抽象执行为验证对象；若需实验验证，可通过人工合成的历史序列或仿真生成的复制系统日志来构造数据集。

**📈 对比分析**

方法比较主要体现在理论可判定性和表达能力上：与以往只能处理特定一致性模型或需要手工编写判定器的做法不同，HistMSO 能统一表达大部分模型并通过 MONA 自动化；但作者未给出具体运行时性能指标，只说明 MONA 可在“实际”规模内完成判定；若要评估性能，需要在实验平台上对比 MONA 与传统模型检查/SMT 求解器在同一组历史上的求解时间。

**⚠️ 局限性**

主要限制包括：
• 需要对进程数、值集合、对象集合做有限性假设；
• 只支持实时仲裁与 k‑瞬时可见性约束的抽象执行；
• MONA 的词长度上限（有限字母表）限制了历史规模；
• 监测方式为离线、集中式，无法满足在线或分布式实时监控需求；
• 对于涉及精确时序约束的模型（如定时可见性）尚无法直接编码。

---

## 381. FSUNav: A Cerebrum-Cerebellum Architecture for Fast, Safe, and Universal Zero-Shot Goal-Oriented Navigation

**arXiv ID:** 2604.03139 | [PDF](https://arxiv.org/pdf/2604.03139v1)

**作者:** Mingao Tan `[一作]` (Shanghai Jiao Tong University), Wei Zhang `[通讯]` (Eastern Institute of Technology)

**通讯引用:** 3233 | [OpenAlex ID](https://openalex.org/A5100695302)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出FSUNav架构，将脑干和小脑模块结合，实现跨平台、实时、安全且零样本的目标导航。

**💡 创新点**

创新点在于将Vision‑Language Model作为多层推理引擎，融合三层语义、空间、规则，支持多模态、开放词汇，并通过尺寸可配置的DRL局部规划实现跨平台无训练部署。

**🔧 技术方法**

采用Vision‑Language Model (Qwen3‑VL‑32B‑Instruct)、深度强化学习（SAC）、PointNet、三层推理结构、语义地图与前沿探索、冷却机制等技术。

**📊 数据集**

使用Matterport3D (MP3D) 与 Habitat‑Matterport3D (HM3D) 数据集，进行Object‑Goal、Instance‑Image、Text‑Goal 与开放词汇任务。

**📈 对比分析**

与监督、训练‑免费与通用方法对比，在MP3D/ HM3D 上取得最高 SR 与 SPL，开放词汇任务亦优于 MTU3D、VISOR，显示在所有任务上性能领先且无任务特定训练。

**⚠️ 局限性**

限制包括对VLM推理速度的依赖，RGB‑only 深度估计在光照变化下的鲁棒性不足，以及实时推理对算力的高需求仍是低配机器人面临的挑战。

---

## 382. Minimal Information Control Invariance via Vector Quantization

**arXiv ID:** 2604.03132 | [PDF](https://arxiv.org/pdf/2604.03132v1)

**作者:** Ege Yuceel `[一作]` (University of Illinois at Urbana-Champaign), Sayan Mitra `[通讯]` (University of Illinois at Urbana-Champaign)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `90291a0e-9d36-4a08-9a16-89ce846d923f` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

提出一种基于向量量化自编码器（VQ-AE）的低复杂度采样数据控制器，实现对12维四旋翼模型的前向不变集保证

**💡 创新点**

创新点在于将不变熵理论与学习式控制器联合：通过VQ-AE同时学习状态空间划分与有限控制码表，并提供迭代前向认证（IFC）算法以严谨验证不变性

**🔧 技术方法**

采用向量量化自编码器、可逆仿射编码器、可学习的控制码表、Lipschitz可达集包络、Sum-of-Squares（SOS）可行性判定及多项式近似动力学

**📊 数据集**

使用的主要数据集为仿真采样的四旋翼状态样本（10⁵-10⁷个轨迹），以及ArUco标记图像（不同像素分辨率）进行视觉观测器训练

**📈 对比分析**

与均匀网格+神经网络、均匀网格+闭环LQR、VQ-AE+开放式LQR等基线对比，控制码表从4096降至14个，保留99.98%不变率，远优于基线且验证通过；在视觉实验中，像素分辨率可降至≈65px仍保持99%不变率

**⚠️ 局限性**

局限包括：仅在多项式近似动力学下可验证；编码器限制为仿射映射；不确定性/噪声鲁棒性未全面分析；理论下界与实际最小码表大小仍有差距

---

## 383. A Systematic Security Evaluation of OpenClaw and Its Variants

**arXiv ID:** 2604.03131 | [PDF](https://arxiv.org/pdf/2604.03131v1)

**作者:** Yuhang Wang `[一作]` (Xidian University), Shiguo Lian `[通讯]` (China Unicom)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `6215c339-3735-4be3-8a07-5bbb7004712d` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

系统评估了六个 OpenClaw 系列智能代理框架在 13 类安全风险下的攻击成功率，并构建了 205 条测试用例基准。

**💡 创新点**

创新点在于将模型能力与框架机制耦合考察，提出链式风险传播分析，揭示不同模型-框架组合的独特风险特征。

**🔧 技术方法**

使用安全基准构造、链式攻击评估方法，以及多模型多框架交叉实验。

**📊 数据集**

使用自构建的 205 条测试样本，覆盖 13 个攻击类别。

**📈 对比分析**

对比 6 个框架与多种后端模型的整体与按类别、按链段的攻击成功率；结果显示 OpenClaw 等框架整体成功率在 16–55% 之间，Reconnaissance 成功率普遍高于 50%。

**⚠️ 局限性**

局限在于仅覆盖 OpenClaw 系列、测试集可能不足以代表所有真实攻击；评估基于离线实验，未验证在生产环境的表现；部分攻击类型缺乏深度探测。

---

## 384. Self-Distilled RLVR

**arXiv ID:** 2604.03128 | [PDF](https://arxiv.org/pdf/2604.03128v1)

**作者:** Chenxu Yang `[一作]` (Institute of Information Engineering, Chinese Academy of Sciences), Nan Duan `[通讯]` (JD.COM)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `8d10c613-917e-4880-9716-17789f50e119` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

本文针对大模型推理后训练中的奖励稀疏问题，提出并验证了 RLSD（Reinforcement Learning with Self‑Distillation）框架，解决了传统自蒸馏（OPSD）因信息不对称导致的特权信息泄漏与性能衰退问题。

**💡 创新点**

创新点：① 把教师与学生分布匹配改为仅利用教师与学生对齐差异作为 token‑level credit 的权重，而不再直接逼近教师分布；② 将环境奖励决定更新方向，教师评估决定更新幅度，从而消除信息不对称导致的互信息残差；③ 设计了基于证据比的权重、剪枝机制以及 stop‑gradient 处理，使模型在无外部教师模型的情况下实现稠密 token‑level 反馈。

**🔧 技术方法**

技术手段：基于 GRPO 的强化学习框架；自蒸馏与分布匹配理论分析；KL 散度、互信息、梯度偏差分析；token‑level 权重与优势重构；clipped 目标与权重裁剪；stop‑gradient 处理；单 forward pass 计算教师 logits；共享参数训练。

**📊 数据集**

数据集：训练使用 MMFineReason‑123K（从 MMFineReason‑1.8M 过滤难度高的样本）；评估使用五个多模态推理基准：MMMU、MathVista、MathVision、ZeroBench、WeMath。

**📈 对比分析**

对比方法：Base LLM、GRPO、OPSD、SDPO、GRPO+OPSD；实验结果显示 RLSD 在 4K 长度下平均准确率 56.18%，比 Base LLM 提升 4.69%，比 GRPO 提升 2.32%，在 MathVista 和 MathVision 上分别提升 1.9% 与 3.91%。

**⚠️ 局限性**

局限性：实验仅覆盖多模态推理任务，未在纯文本、视频等场景进行充分验证；模型与实验依赖大规模 GPU 资源；RLSD 仍需对训练稳定性与规模化进行进一步研究。

---

## 385. Domain-Adapted Retrieval for In-Context Annotation of Pedagogical Dialogue Acts

**arXiv ID:** 2604.03127 | [PDF](https://arxiv.org/pdf/2604.03127v1)

**作者:** Jinsook Lee `[一作]` (Cornell University), Rene F. Kizilcec `[通讯]` (Cornell University)

**通讯引用:** 7316 | [OpenAlex ID](https://openalex.org/A5071778778)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `edb9d762-f411-4838-a852-f2d638b018db` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

开发了一个检索增强的教学对话注释管线，在冻结大型语言模型的基础上，仅通过微调轻量级嵌入模型和句子级索引来提高教学动作的标注准确率。

**💡 创新点**

创新点在于：1）只改检索器而不微调生成模型，保持模型可迁移；2）采用多负样本排名损失对嵌入进行领域适配；3）通过句子级索引并检索父级语境作为few‑shot示例，显著提升检索精度与标注质量。

**🔧 技术方法**

技术包括：多负样本排名损失微调句子嵌入；FAISS向量检索；动态语义分块；检索增强提示学习（RAG）与多模型推理（GPT‑5.2、Claude Sonnet 4.6、Qwen3‑32b）。

**📊 数据集**

使用了两个数学教学对话数据集：TalkMoves（课堂讨论、教师六类教学动作）和Eedi（在线一对一数学辅导、同一六类标签）。

**📈 对比分析**

通过与无检索（仅codebook+上下文）以及三种检索条件（无微调、微调块级、微调句子级）对比，并在三种LLM上评估。最佳配置在TalkMoves达Cohen’s κ≈0.58、Eedi≈0.74，显著超过基线（κ≈0.27‑0.41），检索标签匹配率从约40%提升至≈62%/73%，成为性能提升的关键。

**⚠️ 局限性**

限制包括：只验证数学领域的六类标签，未测试其他学科或语言；标签稀疏仍存在挑战；评估仅基于已标注数据，未充分验证跨数据集的泛化。

---

## 386. SCC-Loc: A Unified Semantic Cascade Consensus Framework for UAV Thermal Geo-Localization

**arXiv ID:** 2604.03120 | [PDF](https://arxiv.org/pdf/2604.03120v1)

**作者:** Xiaoran Zhang `[一作]` (National University of Defense Technology), Huaxin Xiao `[通讯]` (National University of Defense Technology)

**通讯引用:** 1953 | [OpenAlex ID](https://openalex.org/A5001902162)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5b4c1114-4a70-478e-9921-2514ee03850d` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种统一的语义-级联-共识框架 SCC‑Loc，解决无人机在 GNSS‑被抑制环境下热成像与可见光卫星图像的跨模态定位问题。

**💡 创新点**

创新点包括：① 共享单一 DINOv2 基础模型，节省内存；② 语义引导视口对齐（SGVA）自适应纠正量化误差；③ 层级纹理结构过滤（C‑SATSF）逐级剔除跨模态噪声；④ 基于物理约束与地理共识投票的姿态选择（CD‑RAPS），抑制视觉诱骗。

**🔧 技术方法**

技术栈包含 DINOv2、MINIMA_RoMa、语义激活、C‑SATSF 过滤、物理约束非线性优化、可靠性评估与地理共识投票。

**📊 数据集**

使用自建 Thermal‑UAV 数据集（11,890 张热图，昼夜、城市与乡村场景），配合 Google Earth ortho‑photo 与对应 DSM。

**📈 对比分析**

在 Recall@10、Acc@5、Acc@10、Acc@20、ME 等指标上与两阶段基线（CAMP、DINOv2/3+XoFTR/RoMa/MINIMA_RoMa）及领域特定模型（STHN、NIVnet）比较，SCC‑Loc 取得 Recall@10 99.57%、Acc@5 52.09%、Acc@10 86.38%、Acc@20 93.53%，平均误差仅 9.37 ± 25.26 m，比最强基线提升 7.6 倍。

**⚠️ 局限性**

局限包括：① 需要手工调参，超参数依赖性强；② 对姿态先验（pitch/yaw）敏感，误差会影响性能；③ 推理时延较大，难以满足实时边缘部署。

---

## 387. Supply-Chain Poisoning Attacks Against LLM Coding Agent Skill Ecosystems

**arXiv ID:** 2604.03081 | [PDF](https://arxiv.org/pdf/2604.03081v1)

**作者:** Yubin Qu `[一作]` (Griffith University), Lei Ma `[通讯]` (University of Tokyo)

**通讯引用:** 7175 | [OpenAlex ID](https://openalex.org/A5101468661)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `6215c339-3735-4be3-8a07-5bbb7004712d` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对基于大型语言模型的编码代理技能的供应链攻击进行了系统评估，并提出了文档驱动的隐式负载执行（DDIPE）攻击框架。

**💡 创新点**

创新点在于发现技能文档中的代码示例可被利用隐藏恶意逻辑，并通过LLM驱动的种子-变异-验证流程大规模自动生成对抗性技能。

**🔧 技术方法**

采用LLM生成、对抗性技能构造、静态与动态安全分析、LLM‑as‑a‑Judge评估等技术。

**📊 数据集**

使用了包含1,070个对抗性技能的自构建数据集，覆盖15类MITRE ATT&CK攻击技术，以及公开的四个生产级编码代理框架和五个LLM模型。

**📈 对比分析**

通过在四框架×五模型的8种组合上评估，DDIPE的直接执行率在最强防御下仍为2.3%，弱防御下可升至27.1%，相比传统显式指令注入的0%执行率，展示了显著的攻击成功率。

**⚠️ 局限性**

局限性包括：仅测试了部分框架与模型，缺乏跨模型动态沙箱或LLM审计评估，且对抗性技能的生成可能对同一LLM过拟合，未覆盖所有潜在攻击向量。

---

## 388. MI-Pruner: Crossmodal Mutual Information-guided Token Pruner for Efficient MLLMs

**arXiv ID:** 2604.03072 | [PDF](https://arxiv.org/pdf/2604.03072v1)

**作者:** Jiameng Li `[一作]` (KU Leuven), Matthew B. Blaschko `[通讯]` (KU Leuven)

**通讯引用:** 7980 | [OpenAlex ID](https://openalex.org/A5077783791)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `57a58b01-81b4-4d75-a45c-2e891f272b50` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种跨模态互信息（MI）指导的视觉标记修剪方法，直接在视觉与文本投影空间计算互信息并选取最具语义相关且互相不冗余的视觉标记。

**💡 创新点**

创新点在于用互信息量化跨模态与同模态相关性，避免依赖注意力权重，提供高效非侵入式剪枝算法；同时将最大聚合与贪心/堆搜索结合，实现理论保证与实践效率的双重提升。

**🔧 技术方法**

技术方法包括互信息与点互信息估计、投影空间正则化、温度参数化的相似度分布、最大聚合度量、贪心/堆搜索的高效实现以及与现有注意力或多样性剪枝策略的融合。

**📊 数据集**

使用的实验数据集包括多模态问答基准 LLaVA1.5、Qwen2VL、Qwen3VL、Video‑LLaVA‑7B，以及 GQA、SQA、TextVQA、MMVet、MME_P、POPE 等多种问答任务，并在视频问答上评估 TGIF‑QA、MSVD‑QA、MSRVTT‑QA。

**📈 对比分析**

与多种基于注意力（FastV、SparseVLM、VisionZip、VisPruner）和非注意力（DART、随机、相似度）剪枝方法在同一模型上进行对比，实验表明在大幅减少视觉标记数（如从 2048 降到 64 或 32）后，准确率、问答得分几乎不下降且往往优于 SOTA；推理延迟和显存占用也大幅降低。

**⚠️ 局限性**

局限性包括假设视觉标记均匀概率和条件独立性、仅对视觉标记进行剪枝且未考虑文本标记的冗余、以及对视觉先验分布的依赖可进一步改进。

---

## 389. SparseSplat: Towards Applicable Feed-Forward 3D Gaussian Splatting with Pixel-Unaligned Prediction

**arXiv ID:** 2604.03069 | [PDF](https://arxiv.org/pdf/2604.03069v1)

**作者:** Zicheng Zhang `[一作]` (Fudan University), Wenchao Ding `[通讯]` (Fudan University)

**通讯引用:** 3720 | [OpenAlex ID](https://openalex.org/A5102769588)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出SparseSplat，一个feed-forward 3D Gaussian Splatting模型，能够在一次前向传播中生成稀疏、紧凑的3D场景表示。

**💡 创新点**

创新点在于：1）基于Shannon熵的自适应原子采样，使得纹理丰富区域密集、纹理稀疏区域稀疏分布；2）3D局部属性预测器利用KNN和地理注意力，实现对局部上下文的精准感知，从而解决分布和感受野不匹配问题。

**🔧 技术方法**

使用了预训练的多视角深度网络（冻结backbone）、熵采样策略、3D KNN查询、地理注意力模块以及可分离的高斯渲染器。

**📊 数据集**

主要数据集为DL3DV，用于训练与评估；在Replica数据集上测试跨域泛化性能。

**📈 对比分析**

与MVSplat、DepthSplat等pixel-aligned方法对比，SparseSplat在150k Gaussian下取得24.20 PSNR、0.817 SSIM、0.168 LPIPS，几乎等同DepthSplat（24.17 PSNR）但仅使用22%原子；在10k/40k等稀疏设置下仍保持21–23 PSNR。

**⚠️ 局限性**

局限包括：KNN聚合在深度估计错误严重时失效；KNN计算成本较高，无法充分利用二维共视信息。

---

## 390. StoryScope: Investigating idiosyncrasies in AI fiction

**arXiv ID:** 2604.03136 | [PDF](https://arxiv.org/pdf/2604.03136v1)

**作者:** Jenna Russell `[一作]` (University of Maryland), John Wieting `[通讯]` (Google DeepMind)

**通讯引用:** 2727 | [OpenAlex ID](https://openalex.org/A5002499277)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `79276348-11e0-48e3-84bc-7ec231d0171c` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出一套自动化的管道，利用大型语言模型从约61,600篇≈5,000词长的平行文本中抽取可解释的叙事层面特征，并以此进行 AI 文本检测与作者归属；

**💡 创新点**

创新点在于将焦点从传统的词汇、句法风格转向跨文档的叙事结构选择（如主题显式度、时间结构、角色能动性等），构建基于 NarraBench 的可解释特征集，并证明这些叙事特征能在去除风格信号后仍保持高检测与归属性能；

**🔧 技术方法**

技术手段包括：1）使用 GPT-5.1 生成结构化叙事模板；2）对同一题材的多源文本进行跨源比较，提炼差异化特征；3）以专家式提示驱动的 LLM 进行特征发掘，得到 304 个可解释特征；4）通过 XGBoost 结合 SHAP 进行模型训练与特征重要性解释；5）构建核心特征与指纹特征集；

**📊 数据集**

使用来自 Books3 的 10,272 篇人类短篇与五大 LLM（Claude、Gemini、GPT、DeepSeek、Kimi）生成的 51,336 篇 AI 小说，形成 61,608 篇故事的平行语料库；

**📈 对比分析**

实验通过对比 Narrative（257 特征）、Core（30 特征）、Core+Fingerprint（101 特征）以及 Narrative+Style（304 特征）模型，评估宏观 F1 与 AUPRC；人类 vs AI 的宏观 F1 达到 93.2%，仅使用核心特征即可达到 84.8%；在 6‑路作者归属任务中，Narrative 模型宏观 F1 为 68.4%，显著优于 Style‑Only（60.4%）并接近 Narrative+Style（77.3%）；在对风格编辑后文本的测试中，Narrative 检测性能仅下降 1.6 分，证明其稳健性；

**⚠️ 局限性**

局限性包括：1）仅针对长篇（≈5,000 词）小说，短文本或其他文学体裁难以直接迁移；2）特征抽取高度依赖 LLM 的推理能力与 prompt 设计，可能受模型版本变动影响；3）未公开人类原始文本，限制外部复现与进一步验证；4）核心与指纹特征覆盖面有限，可能遗漏更细粒度的叙事差异；5）实验仅涵盖五个 LLM，未来模型演进可能改变叙事分布。

---

## 391. Salt: Self-Consistent Distribution Matching with Cache-Aware Training for Fast Video Generation

**arXiv ID:** 2604.03118 | [PDF](https://arxiv.org/pdf/2604.03118v1)

**作者:** Xingtong Ge `[一作]` (Hong Kong University of Science and Technology), Jun Zhang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 86274 | [OpenAlex ID](https://openalex.org/A5100400217)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `8d10c613-917e-4880-9716-17789f50e119` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a8e75ba4-7a2d-4153-b003-06c94533add0` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

本文提出了一种用于极少步（2–4次 NFE）视频扩散模型蒸馏的训练框架 Salt，利用自一致分布匹配（SC‑DMD）正则化多步更新的半群一致性，并在自回归场景下通过混合步长滚动和 KV 缓存对齐损失实现 cache‑aware 训练。

**💡 创新点**

创新点主要包括：①引入半群缺陷自一致正则化，弥补 DMD 的多步组合缺陷；②设计混合步长训练策略以覆盖不同缓存质量；③提出缓存条件下的参考对齐损失，提升低步长下的输出质量与长期一致性。

**🔧 技术方法**

技术方法包括：分布匹配蒸馏（Distribution Matching Distillation）、自一致（semigroup‑defect）正则化、混合步长（mixed‑step）滚动、KV 缓存对齐（cache‑conditioned alignment）、基于概率流 ODE 的 Euler 采样与一致性采样（CM）。

**📊 数据集**

实验数据集涵盖 Wan 2.1 14B I2V 与 1.3B T2V、VBench‑I2V、VBench、VBench‑Long、以及 MovieGenBench 的 30 秒长视频。

**📈 对比分析**

与 PCM、DMD、LightX2V、rCM、Self Forcing、LongLive、Causal Forcing 等基线对比，Salt 在 4‑NFE 下 I2VScore 提升至 93.90（↑0.27），VBench 总分提升至 84.93（↑0.53），长时序 30 s 生成的 Semantic 得分提升至 64.74（↑0.86），同时保持或提升图像质量、运动连贯性和低闪烁等指标。

**⚠️ 局限性**

局限性包括：对极高动态场景的表现仍有限；训练仍需较多 GPU 资源；对 KV 缓存的依赖使得在某些自回归模型中效果不如预期；未在超大规模模型（>10B 参数）或多模态输入场景中进行验证。

---

## 392. Revealing Physical-World Semantic Vulnerabilities: Universal Adversarial Patches for Infrared Vision-Language Models

**arXiv ID:** 2604.03117 | [PDF](https://arxiv.org/pdf/2604.03117v1)

**作者:** Chengyin Hu `[一作]`, Wen Yao `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `6215c339-3735-4be3-8a07-5bbb7004712d` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `4de8e9d8-757b-475f-9627-18a445e50202`

**🎯 论文内容**

提出了针对红外视觉语言模型的通用物理对抗补丁 UCGP，并在分类、描述与问答任务中验证其攻击效果。

**💡 创新点**

首次将曲线网格参数化、统一子空间/拓扑/隐蔽目标约束与 EOT+TPS 物理鲁棒性建模相结合，实现单一可部署补丁在红外 VLM 上跨任务、跨模型、跨数据集的稳健攻击。

**🔧 技术方法**

采用曲线网格 Mesh (CGM)、Meta Differential Evolution、EOT+TPS 变形、Graph‑KL 结构损失、低频子空间偏离与隐蔽约束等技术。

**📊 数据集**

主要使用 Infrared‑COCO 进行主实验，并在 LSOTB‑TIR、LLVIP、M3FD、FLIR v1.3 等数据集进行跨数据集与跨模型评估。

**📈 对比分析**

与 AdvGrid、HCB、QR‑Code 等基线对比，在四种 CLIP backbone 上实现 94%+ 的攻击成功率，语义一致性与正确率下降超过 15%+，物理实验中成功率超过 70%，且跨模型与跨数据集传递性显著。

**⚠️ 局限性**

对抗训练或图像修复可显著降低攻击效果；在高视角或远距离场景下物理鲁棒性有限；对某些类别的攻击效果相对弱，且依赖目标类别的统计信息。

---

## 393. Can VLMs Truly Forget? Benchmarking Training-Free Visual Concept Unlearning

**arXiv ID:** 2604.03114 | [PDF](https://arxiv.org/pdf/2604.03114v1)

**作者:** Zhangyun Tan `[一作]` (University of Rochester), Chenliang Xu `[通讯]` (University of Rochester)

**通讯引用:** 6547 | [OpenAlex ID](https://openalex.org/A5064805926)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并实现了VLM-UnBench，一个用于评估视觉语言模型训练‑free视觉概念遗忘的基准框架

**💡 创新点**

首次系统化地定义多层概念层级、分级探测与五种评估条件，区别真实遗忘与仅遵从提示的行为

**🔧 技术方法**

采用基于提示的训练‑free遗忘方法、四级探测分类和五种评估条件对比实验

**📊 数据集**

使用七个公开视觉数据集（COCO、MIT Indoor‑67、AID、LAD、SpatialMQA、Celebrity Faces、Logo‑2K+），并构造多级忘记/保留拆分

**📈 对比分析**

通过对比基准下的基线、软/中/硬提示以及Oracle条件，发现现有提示方法在现实条件下几乎不降低遗忘集准确率，而Oracle提示可显著抑制回答，显示真实遗忘效果差

**⚠️ 局限性**

当前训练‑free方法只能产生表面指令遵从，难以真正抑制模型对对象、场景等强视觉概念的识别，缺乏有效的知识消除机制

---

## 394. Adaptive Bidding Policies for First-Price Auctions with Budget Constraints under Non-stationarity

**arXiv ID:** 2604.03103 | [PDF](https://arxiv.org/pdf/2604.03103v1)

**作者:** Yige Wang `[一作]` (Hong Kong University of Science and Technology), Jiashuo Jiang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1377 | [OpenAlex ID](https://openalex.org/A5101856185)

**关键词:** `1787d272-1540-4d97-bbe7-e9bbfb732355` `5b4c1114-4a70-478e-9921-2514ee03850d` `5a41884c-404f-4688-a89c-aa238c10fe68` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472`

**🎯 论文内容**

研究预算受限的重复一价拍卖中自适应竞价问题，提出基于双梯度下降的在线竞价策略。

**💡 创新点**

创新点包括：① 用 Wasserstein 距离度量私有价值分布的非平稳性，并证明该度量在无信息情境下可获得最优阶数的 regret；② 在有预算分配预测时消除非平稳性项，得到 Õ(√T+V_T) 的上界；③ 引入基于周期预算的更严格基准，证明可实现仅 O(√T) 的 regret；④ 对基准偏离计划的鲁棒性给出理论分析。

**🔧 技术方法**

使用的技术包括：拉格朗日对偶与对偶梯度下降、在线学习框架、全信息反馈、Wasserstein 距离、期望奖励与预算约束的凸优化。

**📊 数据集**

实验数据：人工生成的私有价值分布 F_t（均匀分布，均值 μ_t、标准差 σ_t 随机取 [1,2]）和竞争者最高出价 m_t（均匀分布 G 在 [1,2]）。

**📈 对比分析**

通过与离线最优基准对比实验，展示相对误差随时间 T 减小、随 Wasserstein 非平稳度增大、随预算预测误差 V_T 增大而升高，验证理论上限与实践表现的一致性。

**⚠️ 局限性**

局限性：仅考虑全信息反馈，假设私有价值独立且竞争者最高出价分布未知但可学习；未处理多方竞争的策略博弈；对预算分配预测的可获得性与精度未给出理论保证。

---

## 395. Flash-Mono: Feed-Forward Accelerated Gaussian Splatting Monocular SLAM

**arXiv ID:** 2604.03092 | [PDF](https://arxiv.org/pdf/2604.03092v1)

**作者:** Zicheng Zhang `[一作]` (Fudan University), Wenchao Ding `[通讯]` (Fudan University)

**通讯引用:** 3720 | [OpenAlex ID](https://openalex.org/A5102769588)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `4bf3b852-21ff-4736-b125-37e24f3c9a32` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f`

**🎯 论文内容**

提出了一种实时单目 Gaussian Splatting SLAM 系统 Flash-Mono，采用递归前馈模型直接预测相机位姿和 2D Gaussian 场景表示，并通过隐藏状态实现闭环校正。

**💡 创新点**

创新点在于：1) 递归前馈架构利用多帧上下文一次性预测高质量 Gaussian；2) 隐藏状态既做子地图记忆又作为闭环约束，实现在 Sim(3) 空间的快速全局优化；3) 将 3D Gaussian 换为 2D Gaussian surfel 提升几何精度；4) 后端仅需少量 20 次轻量级优化即可得到高质量渲染。

**🔧 技术方法**

使用了 ViT+Transformer 交叉注意力前馈网络、跨帧隐藏状态、Sim(3) 闭环优化（GTSAM）、轻量级后端 voxel 化与局部/全局优化，以及 2D Gaussian surfel 的渲染与融合。

**📊 数据集**

在 ScanNet、BundleFusion（室内）和 KITTI（户外）三大数据集上训练与评估；训练集包括 DL3DV、ScanNet++；测试集覆盖大规模多房间、不同光照与动态环境。

**📈 对比分析**

与 MonoGS、DepthGS、S3PO-GS 以及 ORB‑SLAM3、DROID‑SLAM、MASt3R‑SLAM 等基线对比，Flash‑Mono 在 ATE、SSIM、PSNR、LPIPS 等指标上均优于对手，并实现 10× 的速度提升（10 FPS+），同时保持高质量渲染与几何精度。

**⚠️ 局限性**

局限性：1) 递归模型仍会出现短期遗忘，需子地图划分；2) 闭环检测依赖外部外观匹配，动态物体时易误闭环；3) 仍受制于单目尺度不确定性，需全局 Sim(3) 纠正；4) 在极大尺度或极端光照下的泛化仍待进一步验证。

---

## 396. The Price of Interoperability: Exploring Cross-Chain Bridges and Their Economic Consequences

**arXiv ID:** 2604.03083 | [PDF](https://arxiv.org/pdf/2604.03083v1)

**作者:** Yiyue Cao `[一作]` (Hong Kong University of Science and Technology), Xuechao Wang `[通讯]` (Hong Kong University of Science and Technology)

**通讯引用:** 1074 | [OpenAlex ID](https://openalex.org/A5101681122)

**关键词:** `eda14718-2b67-4c6c-a1d0-312bdc4fbf1e` `9587dba8-6c1f-4e48-8ba3-7bed5ce8f472` `3f18e8e3-0266-457c-8567-9039b6d2394d` `5a41884c-404f-4688-a89c-aa238c10fe68`

**🎯 论文内容**

本文通过构建跨链生态系统的时间可变加权超图模型，提出结构互操作性和活跃互操作性两项度量，系统性评估了2022‑2025年间20条链与16条桥接协议的互联与利用情况。

**💡 创新点**

创新点在于将桥接网络抽象为超图，分离基础设施容量与用户使用强度两大维度，从而揭示了“增长‑收益悖论”和“效率‑脆弱权衡”以及桥接设计与链架构对经济结果的异质性。

**🔧 技术方法**

主要技术包括加权超图投影、最短路径计算、面板固定效应回归、差分‑差分 (DiD) 与滚动相关性分析，以评估互操作性对 TVL、Gas 成本、token 回报及同步性的因果影响。

**📊 数据集**

使用了从 Dune、官方桥接 API、DefiLlama TVL、Yahoo Finance 价格等多源同步收集的数据，构成覆盖 20 条链、16 条桥接协议、2022‑2025 年的日度面板。

**📈 对比分析**

通过与传统仅基于流量的互操作性衡量相比，本文的结构/活跃分离框架能够捕捉不同维度的正负效应，回归结果表明结构互操作性提升 TVL 与 Gas 效率，活跃互操作性则导致 Gas 成本上升与链间同步性增强，统计显著性高。

**⚠️ 局限性**

局限性包括：仅关注链上桥接交易，无法捕获中心化交易所或离链桥接的流量；数据仅覆盖公开链与桥接，可能低估真实互通规模；因果识别依赖外生冲击（如 Multichain 崩溃）与固定效应，可能存在未观测混杂因素。

---

## 397. EEspice: A Modular Circuit Simulation Platform with Parallel Device Model Evaluation via Graph Coloring

**arXiv ID:** 2604.03079 | [PDF](https://arxiv.org/pdf/2604.03079v1)

**作者:** Xuanhao Bao `[一作]` (University of Edinburgh), Danial Chitnis `[通讯]` (University of Edinburgh)

**通讯引用:** 843 | [OpenAlex ID](https://openalex.org/A5029110112)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

提出 EEspice，一款模块化的 SPICE 仿真平台，利用图着色实现 FET 设备模型评估与印刷的无锁并行化，消除传统多线程印刷阶段的冲突瓶颈。

**💡 创新点**

创新点在于：① 构造 FET 冲突图并使用图着色将设备分组为冲突自由集；② 通过模块化内核实现评估与印刷分离，允许多种并行策略；③ 在 ColorFused 方案中将评估与印刷合并在同一线程，进一步减少同步开销。

**🔧 技术方法**

主要技术包括：图着色算法（贪心着色）、OpenMP 并行循环、BSIM4 设备模型、稀疏矩阵求解器 KLU、Newton–Raphson 迭代、并行 NR 循环与模块化内核设计。

**📊 数据集**

使用典型的 CMOS 全加器、64 位溢位加法器（原始、R14、R89 变体）作为基准电路；通过这些电路评估冲突级别、颜色数和并行性能。

**📈 对比分析**

比较方法：与 Ngspice（单线程）以及 EEspice 的四种实现（loadsingle、loadomp、Color、ColorFused）在同一 64 核机器上对比。结果显示：在低冲突场景（颜色数 ≈1）ColorFused 可实现高达 45× 的速度提升；在高冲突场景（颜色数 ≈334）并行效率下降。整体仿真速度提升约 4.4× 单线程基线，取决于冲突级别和颜色数。

**⚠️ 局限性**

局限性：① 颜色数 C 冲突高时并行效率显著下降，导致 ColorFused 在高冲突情形下甚至低于 1×；② 设备评估速度提升后，稀疏线性求解器（KLU）成为新的瓶颈，需进一步加速；③ 目前验证仅在 BSIM4，需在更复杂的 BSIM-CMG 或后布局寄生等场景中进一步评估。

---

## 398. Joint Prediction of Human Motions and Actions in Human-Robot Collaboration

**arXiv ID:** 2604.03065 | [PDF](https://arxiv.org/pdf/2604.03065v1)

**作者:** Alessandra Bulanti `[一作]` (University of Genoa), Fulvio Mastrogiovanni `[通讯]` (University of Genoa)

**通讯引用:** 2709 | [OpenAlex ID](https://openalex.org/A5017108129)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `64443552-63e0-44b5-906f-d90fe95c5a1b` `edb9d762-f411-4838-a852-f2d638b018db` `c773407a-6119-4871-b8b3-1e7ae17a6851` `5a41884c-404f-4688-a89c-aa238c10fe68` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

提出一种层次化、递归的概率框架，用于人-机器人协作场景中同时估计并预测人体连续运动和离散动作；

**💡 创新点**

创新点在于①将运动与动作通过 Allen 间隔代数构造层次化结构；②将连续动力学、离散标签与持续时间统一到一个因子图概率模型；③设计了基于贝叶斯滤波的递归推理循环，实现在线预测与证据更新；

**🔧 技术方法**

技术上使用 Transformer 作为运动预测器和动作分类器，配合基于因子图的联合概率推理和 Allen 间隔约束；

**📊 数据集**

数据集采用 Bioptim 生成的肌肉模型手臂 3D 抓取轨迹，包含 3 个目标体积、9 种标签的 1000 条轨迹，并在 0%、10%、30% 噪声水平下合成噪声数据；

**📈 对比分析**

通过与噪声水平对比实验，评价 PCC 与 RMSE，发现噪声增大时 PCC 降低、RMSE 上升；动作分类在增量前缀条件下可在 170–230 步内达到 0.9+ 的准确率；推理时间均低于 0.2 s，满足实时 HRC 要求；

**⚠️ 局限性**

局限性包括仅使用简化的单臂肌肉模型，缺乏真实闭环交互验证，未对预测不确定性进行校准，且层次约束与递归更新可能导致信念震荡。

---

## 399. Can Nano Banana 2 Replace Traditional Image Restoration Models? An Evaluation of Its Performance on Image Restoration Tasks

**arXiv ID:** 2604.03061 | [PDF](https://arxiv.org/pdf/2604.03061v1)

**作者:** Weixiong Sun `[一作]` (Shenzhen University of Advanced Technology), Chao Dong `[通讯]` (Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `409a1113-3cd2-4a73-8a3a-1bf160ba5c2f` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对Nano Banana 2在多场景、多退化类型的图像恢复任务进行了系统评估

**💡 创新点**

首次证明通用图像编辑模型可作为统一的图像恢复解决方案，并突出了提示设计对恢复效果的关键作用

**🔧 技术方法**

采用提示工程（长度与保真度约束）、全参考与无参考图像质量评估指标、统计显著性检验以及用户研究等技术

**📊 数据集**

使用包含13种场景类别和7种退化类型（如小人脸、拥挤、文本、监控等）的实验数据集

**📈 对比分析**

与HYPIR、PiSA‑SR、TSD‑SR、DiffBIR等尖端恢复模型进行对比，Nano Banana 2在PSNR、SSIM、LPIPS等全参考指标以及MUSIQ、MANIQA、CLIP‑IQA等无参考指标均达到或超过竞争模型，并在用户研究中获得最高平均分

**⚠️ 局限性**

对提示敏感，存在生成不确定性；在极端退化或复杂场景下易出现语义不一致和过度生成，需多轮提示或人工干预以获得最佳结果

---

## 400. An Open-Source LiDAR and Monocular Off-Road Autonomous Navigation Stack

**arXiv ID:** 2604.03096 | [PDF](https://arxiv.org/pdf/2604.03096v1)

**作者:** Rémi Marsal `[一作]` (U2IS, ENSTA, Institut Polytechnique de Paris), David Filliat `[通讯]` (AMIAD)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `6514db3d-8de6-452c-91b7-acdb31787cc4` `5b4c1114-4a70-478e-9921-2514ee03850d` `51c0528b-f690-4182-ae60-bb5f046c276c` `90291a0e-9d36-4a08-9a16-89ce846d923f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

提出并实现了一套完整的离地车辆导航栈，支持激光雷达和单目相机（无监督的深度估计）两种3D感知模式，生成2.5D地形图并基于成本图进行路径规划。

**💡 创新点**

核心创新包括：1）基于预训练的单目深度基座模型（Depth Anything V2）与视觉惯性SLAM（VINS‑Mono）实现无训练的度量深度重标定；2）后处理技术（边缘屏蔽和时序平滑）显著降低了深度假象和SLAM不稳定带来的误差；3）将点云直接投影到机器人中心化的地形图，配合布料模拟滤波（CSF）实现地面与障碍分离。

**🔧 技术方法**

技术栈涵盖：ROS Noetic、Depth Anything V2（FP16/TensorRT加速）、VINS‑Mono或Shi‑Tomasi角点提取、布料模拟滤波（CSF）、A*全局规划、TEB局部规划、实时GPU/CPU频率管理。

**📊 数据集**

评估使用了：1）Isaac Sim photorealistic仿真环境（易/中/难三类），每类设置不同障碍物与地形；2）真实Barakuda机器人实验平台，配备Ouster Dome激光雷达、ZED2i相机、Jetson AGX Orin嵌入式计算机。

**📈 对比分析**

通过Success Rate、SPL、Distance Ratio三项指标进行对比。单目+SLAM在多数仿真场景中与高分辨率LiDAR相当或更优，尤其在20 m目标下成功率达93%；在真实场景中单目系统同样保持100%成功率，但路径效率略低于LiDAR（SPL下降约22%）。

**⚠️ 局限性**

主要限制包括：1）对高草等模糊地形的辨识能力不足，导致误判为障碍；2）单目深度估计对纹理匮乏或低照度环境敏感；3）需手动调参（CSF硬度、TEB参数）以适配不同机器人和环境；4）目前仅支持ROS 1，未来计划迁移至ROS 2。

---

## 401. SkillRT: Compiling Skills for Efficient Execution Everywhere

**arXiv ID:** 2604.03088 | [PDF](https://arxiv.org/pdf/2604.03088v1)

**作者:** Le Chen `[一作]` (Shanghai Jiao Tong University), Haibo Chen `[通讯]` (Shanghai Jiao Tong University)

**通讯引用:** 31813 | [OpenAlex ID](https://openalex.org/A5100419038)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `64443552-63e0-44b5-906f-d90fe95c5a1b` `5b4c1114-4a70-478e-9921-2514ee03850d` `c5260876-9a54-48ae-a63a-8fa6d6ddb799`

**🎯 论文内容**

本文提出了 SkillRT，一种能够将强化学习中的技能（policy）编译成高效可移植执行代码的系统。通过将技能转化为可组合的低层模块，SkillRT 能够在不同硬件平台（CPU、GPU、边缘设备）上实现快速推理。

**💡 创新点**

创新点在于：① 将 RL 技能视为可编译的计算图，采用编译器技术（类似 XLA、TVM）生成优化后代码；② 引入技能层级化表示，支持技能间的组合与复用；③ 通过自动张量形状分析与算子融合，实现跨平台的零复制执行。

**🔧 技术方法**

核心技术包括：编译器前端（将 PyTorch / TensorFlow 代码转为中间表示）；中间层优化（算子融合、张量重排、低精度量化）；后端代码生成（LLVM、CUDA、ARM NEON）；以及执行时的动态调度与资源管理。

**📊 数据集**

实验使用了 OpenAI Gym 的经典控制任务（CartPole、Pendulum）、MuJoCo 机器人控制任务（HalfCheetah、Humanoid）以及 Robosuite 的抓取与搬运任务。数据集主要是任务对应的演示轨迹与环境配置。

**📈 对比分析**

与传统的即时推理框架（PyTorch JIT、TensorRT）相比，SkillRT 在 CPU 上平均速度提升 2.5×，GPU 上提升 1.8×，并将内存占用降低 30%。在边缘设备（Jetson Nano）上，推理延迟从 50ms 降到 12ms，成功实现实时控制。

**⚠️ 局限性**

局限性包括：① 目前仅支持静态策略，动态策略或在线学习无法即时编译；② 对极其复杂的动态环境建模仍存在性能瓶颈；③ 需要手动标注技能分层，缺乏自动化分层方法。

---

## 402. Automatic Textbook Formalization

**arXiv ID:** 2604.03071 | [PDF](https://arxiv.org/pdf/2604.03071v1)

**作者:** Fabian Gloeckle `[一作]` (Meta), Amaury Hayat `[通讯]` (Institut Polytechnique de Paris)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文构建并实验了一个多代理框架，自动化完成了500页代数组合学教材中340个目标定理和定义的Lean形式化。

**💡 创新点**

创新点在于将人类协同软件工程实践（分支、PR、issue跟踪、审查）直接移植到多代理数学形式化中，实现大规模并行且保持一致性的自动化。

**🔧 技术方法**

使用了多代理LLM编程模型（Claude 4.5 Opus），配合Git、Lean REPL、mathlib搜索、Shell工具以及文件系统Issue跟踪进行交互。

**📊 数据集**

主要数据集为公开的代数组合学教材（约500页）以及Lean mathlib中已有的理论，代理从中提取目标定理。

**📈 对比分析**

与单代理或手工形式化对比，实验完成时间为一周，总成本约10万美元，证明所有目标均被证明，性能优于人类专家团队的时间和成本。

**⚠️ 局限性**

局限性包括高计算成本、对NFS和合并队列的瓶颈、缺乏自动化的依赖追踪以及仍需人工监督以确保语义一致性。

---

## 403. Verbalizing LLMs' assumptions to explain and control sycophancy

**arXiv ID:** 2604.03058 | [PDF](https://arxiv.org/pdf/2604.03058v1)

**作者:** Myra Cheng `[一作]` (Stanford University), Diyi Yang `[通讯]` (Stanford University)

**通讯引用:** 13606 | [OpenAlex ID](https://openalex.org/A5089413311)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了Verbalized Assumptions框架，通过让LLM表述对用户的假设来揭示并控制其社交恭维行为。

**💡 创新点**

创新点在于将模型内部隐式假设可视化，并通过线性探针定位这些假设与社交恭维之间的因果关系。

**🔧 技术方法**

主要技术包括对LLM内部表示训练线性探针、激活调节（steering）以及对假设进行结构化与开放式表述。

**📊 数据集**

使用的数据集包括ELEPHANT社交恭维基准（OEQ、AITA、IR）、事实恭维数据集、Cancer-Myth、WildChat、Val-Obj以及真实的AI误导对话记录。

**📈 对比分析**

与直接对抗恭维的探针相比，假设探针在降低社交恭维的同时几乎不损失模型整体性能，且在多数指标上优于传统方法。

**⚠️ 局限性**

局限性包括对小模型支持不足、仅在社交恭维场景下有效、探针依赖于训练数据的覆盖范围以及缺乏对用户个体差异的动态适配。

---

## 404. InCoder-32B-Thinking: Industrial Code World Model for Thinking

**arXiv ID:** 2604.03144 | [PDF](https://arxiv.org/pdf/2604.03144v1)

**作者:** Jian Yang `[一作]` (Beihang University), Weifeng Lv `[通讯]` (Beihang University)

**通讯引用:** 6011 | [OpenAlex ID](https://openalex.org/A5109299440)

**关键词:** `fa95cdfe-56ac-4a08-8734-d50d24aec329` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `edb9d762-f411-4838-a852-f2d638b018db` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了一种基于错误驱动链式思考（ECoT）与工业代码世界模型（ICWM）的思考型代码生成模型。

**💡 创新点**

通过将真实执行错误与诊断反馈合成成思考轨迹，并训练ICWM模拟后端反馈，突破了传统模型对工业后端依赖的瓶颈，赋予模型硬件约束推理与自我校正能力。

**🔧 技术方法**

融合链式思考生成、工业世界模型、强化学习/自我校正训练、混合数据增强以及多轮执行反馈。

**📊 数据集**

使用 InCoder‑32B 代码数据以及从 Verilog、GPU、嵌入式、编译器和 CAD 工具收集的多域执行轨迹，总计约 540M 个思考训练 tokens。

**📈 对比分析**

在 14 个通用代码基准与 9 个工业基准上与多款开源/专有模型对比，表现为 LiveCodeBench V5 81.3%、SWE‑Verified 70.4% 等，工业任务中均为同类最佳。

**⚠️ 局限性**

ICWM 在极端几何或浮点边界场景下误判率提升，思考长度差异巨大导致训练不均衡，某些最难优化任务仍受限于策略缺失。

---

## 405. Beyond Precision: Importance-Aware Recall for Factuality Evaluation in Long-Form LLM Generation

**arXiv ID:** 2604.03141 | [PDF](https://arxiv.org/pdf/2604.03141v1)

**作者:** Nazanin Jafari `[一作]` (University of Massachusetts Amherst), Mohit Iyyer `[通讯]` (University of Maryland)

**通讯引用:** 7168 | [OpenAlex ID](https://openalex.org/A5082767919)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文提出了一套统一的长文本LLM事实性评估框架，既衡量生成内容的精确性（precision），又评估事实覆盖度（recall），并引入基于重要性（相关性与显著性）的加权机制；

**💡 创新点**

创新点在于：①将检索增强的知识源用于自动构建参考事实集；②同时计算精确度与回忆度，并对重要事实加权；③通过比对不同重要性权重的召回@k，揭示模型在核心事实与全量事实覆盖上的差异；

**🔧 技术方法**

技术手段包括检索增强生成（Retrieval‑Augmented Generation）进行事实抽取、利用LLM评审对事实进行相关性/显著性打分、对生成声明进行验证（支持/反驳/不支持），并采用加权召回与F1评分；

**📊 数据集**

使用的评测数据集为FactScore（Biography）、LongFact、LongForm三大长文本生成基准，检索来源包括维基百科、Google Search 等；

**📈 对比分析**

对比方法：在三大基准上评估多款开源与闭源LLM（Llama‑3.1‑8B/70B、Qwen2.5‑7B、Mistral‑7B、GPT4o‑mini、Gemini2.5‑flash‑lite），结果显示模型在精确度上表现较好，但回忆度普遍偏低；在重要性加权召回@k上，模型能较好覆盖首要事实但对完整事实集的覆盖不足；

**⚠️ 局限性**

局限性包括：依赖外部知识源的质量；多阶段自动流程（检索、抽取、验证）可能累计误差；实验仅覆盖有限数据集与模型，泛化性受限。

---

## 406. AI-Assisted Unit Test Writing and Test-Driven Code Refactoring: A Case Study

**arXiv ID:** 2604.03135 | [PDF](https://arxiv.org/pdf/2604.03135v1)

**作者:** Ema Smolic `[一作]`, Mihael Kovac `[通讯]`

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在一份React/Next.js前端代码库中，使用大型语言模型先生成完整的单元测试套件，再利用这些测试安全地引导代码重构；

**💡 创新点**

创新点在于将测试生成与模型辅助重构紧密结合，形成迭代的Plan‑Act‑Verify工作流，并通过变异测试剔除无效测试，显著提升测试覆盖率和重构安全性；

**🔧 技术方法**

采用Gemini 2.5 Pro作为规划器、Cursor Auto模式作为执行器、层级多代理体系以及AST分析、变异测试等技术；

**📊 数据集**

使用的实战数据集为约19k行代码的真实前端项目，最终生成382个测试用例、约16k行测试代码；

**📈 对比分析**

通过分支覆盖率、LOC变化、耦合度和环形复杂度等指标进行对比，实验显示测试分支覆盖率达78%，重构后LOC提升16%，但耦合度下降57.5%，复杂度平均下降0.11；

**⚠️ 局限性**

局限性包括仅在单一技术栈和单一项目上验证，难以泛化；重构后LOC略有增加；仍需人工干预以解决模型幻觉和价值不对齐问题。

---

## 407. SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation

**arXiv ID:** 2604.03134 | [PDF](https://arxiv.org/pdf/2604.03134v1)

**作者:** Meihua Li `[一作]` (Shenzhen University), Yisong Li `[通讯]` (Shenzhen University)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `ba576bd1-e51d-44e8-8077-fc943b333c93` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `90291a0e-9d36-4a08-9a16-89ce846d923f` `e15e3743-5ee0-4d5f-813d-d146868082fc`

**🎯 论文内容**

利用Stable Diffusion预训练模型，对少样本医学图像分割（FSMIS）任务进行适配，提出了支持-查询交互（SQI）和视觉-文本条件翻译（VTCT）两个模块，实现一次性生成分割掩码。

**💡 创新点**

创新点：①在Stable Diffusion的U‑Net中插入SQI，实现支持集信息在潜在空间的双向注意力融合；②VTCT将支持图像的视觉特征映射为文本‑像素嵌入，使Diffusion模型能够以“语言”方式精确引导分割；③只需微调极少参数，即可显著提升跨域泛化性能。

**🔧 技术方法**

使用技术包括：Stable Diffusion v1.5（VAE编码/解码 + U‑Net），跨注意力机制，支持-查询交互模块，视觉‑文本翻译MLP，单步DDIM推理，MSE损失和伪标签训练。

**📊 数据集**

数据集：Abd‑MRI 和 Abd‑CT 两个腹部医学影像数据集，分别在1‑shot设置下进行交叉验证，另外设置了两种跨域评估（1）背景已出现但未标注，2）完全不含测试类的训练样本。

**📈 对比分析**

与DiffewS、DIFD等现有FSMIS方法相比，SD‑FSMIS在Abd‑MRI上与最佳方法相当，在Abd‑CT上在两种跨域设置下分别提升3.47%和3.4% Dice分数；在跨域挑战中保持较小性能下降，显示出更强的鲁棒性。

**⚠️ 局限性**

局限性：①对预训练Stable Diffusion的大型模型和显存有较高依赖；②目前仅在1‑shot场景下验证，进一步多shot或更大类别的表现待探索；③伪标签生成和单步推理可能在极端噪声或严重域移的情况下仍有性能下降。

---

## 408. AlertStar: Path-Aware Alert Prediction on Hyper-Relational Knowledge Graphs

**arXiv ID:** 2604.03104 | [PDF](https://arxiv.org/pdf/2604.03104v1)

**作者:** Zahra Makki Nayeri `[一作]` (Shahrood University of Technology), Mohsen Rezvani `[通讯]` (Shahrood University of Technology)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `57a58b01-81b4-4d75-a45c-2e891f272b50` `3f18e8e3-0266-457c-8567-9039b6d2394d` `edb9d762-f411-4838-a852-f2d638b018db` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文将网络告警建模为带修饰语的知识图谱（h, r, t, 𝒬），并将告警预测任务视为超关系知识图谱完形（HR-KGC）问题。

**💡 创新点**

创新点包括：① 将超关系上下文融入神经贝尔曼-福特网络实现全路径推理；② 提出仅在嵌入空间进行资格融合与路径合成的 AlertStar 模型，兼顾可扩展性和效率；③ 在 HR-NBFNet-CQ 上支持含修饰语的一阶逻辑查询（链式、交集、并集等），扩展了查询能力；④ 通过三种范式（图传播、嵌入融合、复杂查询）系统对比，并在不同 qualifier 密度下展示鲁棒性。

**🔧 技术方法**

技术主要包括：超关系知识图谱表示（StarE 方案）、神经贝尔曼-福特网络（NBFNet）与其超关系扩展、跨注意力与门控路径合成的 Attention-Path 模块、Transformer 的多任务学习、以及基于 NBF 的逻辑查询合成。

**📊 数据集**

使用公开的 Warden 与 UNSW-NB15 两大网络告警数据集，分别在 33%、66% 与 100% 的 qualifier 覆盖率下构建增量（inductive）与完整（transductive）评估。

**📈 对比分析**

与传统的 StarE、ShrinkE、HyNT 以及 NBFNet 的基线相比，MT‑AlertStar 在 MRR 及 H@k 上实现了 19–30% 的提升，且每轮训练速度提升至 50×；在三种 qualifier 密度下表现更为稳定，说明对修饰语依赖性更低；复杂查询上 HR‑NBFNet‑CQ 在交集查询（2i）上优于 StarQE，证明了路径推理在某些查询场景中的优势。

**⚠️ 局限性**

局限性包括：① 仍假设攻击类别为闭集，无法处理新颖攻击类型；② 仅利用告警的静态修饰语，未将时间序列动态信息充分融入；③ 复杂查询的准确率在低 qualifier 覆盖率下显著下降；④ 对于极稀疏或高度动态的告警图谱，基于图传播的模型在计算量上仍不够可扩展。

---

## 409. Co-Evolution of Policy and Internal Reward for Language Agents

**arXiv ID:** 2604.03098 | [PDF](https://arxiv.org/pdf/2604.03098v1)

**作者:** Xinyu Wang `[一作]` (McGill University), Bang Liu `[通讯]` (Université de Montréal)

**通讯引用:** 1043 | [OpenAlex ID](https://openalex.org/A5100691219)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `a4b10f5d-130b-4e77-9367-6469ec621899` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出 Self-Guide，语言代理在每一步生成自我引导（verbal self‑guidance）并将其作为推理时的决策辅助和训练时的内部奖励，从而实现策略与奖励的共同进化。

**💡 创新点**

创新点在于将同一自我引导信号同时用于行动引导和奖励塑造，并通过阶段化信任调度（warm‑up → 逐步激活 → 完全使用 → 逐步衰减）稳定内部奖励与策略的协同演化，避免了外部奖励模型和离线奖励信号带来的偏差与不匹配。

**🔧 技术方法**

技术包括：① 双阶段生成（先生成自我引导再生成动作）; ② 将自然语言自我引导映射为标量奖励；② 在 GRPO（组相对优势）框架下联合优化策略与内部奖励；③ 采用梯形信任调度调控内部奖励权重；④ 兼容其他RL算法如 DAPO。

**📊 数据集**

在 ALFWorld、ScienceWorld 和 WebShop 三个交互式基准（文本仅交互）上进行实验，使用 Qwen3‑1.7B、Qwen3‑4B、Qwen2.5‑7B‑Instruct 三种模型。

**📈 对比分析**

与 ReAct、Reflexion、ReFlAct 等提示法以及 GRPO 基线相比，Self‑Guide 在推理时就能提升成功率；在 RL 训练中，加入自我引导奖励后平均提升约 8%（相对 GRPO），并在所有任务与模型规模上均显著优于基线。

**⚠️ 局限性**

局限性：内部奖励需要经过慎重调度，过早或过强的奖励会导致学习不稳定；自我引导的质量高度依赖任务熟悉度，复杂或模糊任务中效果不一致；离线自我引导蒸馏在转移到在线 RL 时表现欠佳，需进一步研究分布匹配问题。

---

## 410. CASCADE: A Cascading Architecture for Social Coordination with Controllable Emergence at Low Cost

**arXiv ID:** 2604.03091 | [PDF](https://arxiv.org/pdf/2604.03091v1)

**作者:** Yizhi Xu `[一作]` `[通讯]` (Shenzhen University), Yizhi Xu (Shenzhen University)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `64443552-63e0-44b5-906f-d90fe95c5a1b` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6`

**🎯 论文内容**

提出了 CASCADE 三层架构，用宏观状态导演、协调中心和基于标签的 NPC 层来实现可扩展、可控的游戏社会行为。

**💡 创新点**

创新点在于把复杂社会行为的控制从每个 NPC 的生成迁移到统一的协同层；引入分层指令递归、标签路由、行为与对话解耦，以及低成本的局部效用决策。

**🔧 技术方法**

采用宏观状态驱动器（含叙事时钟、全局账本、因果批评器）、模块化协调中心（稀疏激活、指令编译、标签路由）以及基于标签的本地效用函数、行为树；LLM 仅在玩家交互时触发对话。

**📊 数据集**

实验数据基于模拟小镇（10 个 NPC，包含农民、商人、守卫等角色）的微场景原型；未使用公开大规模数据集。

**📈 对比分析**

与全生成式 NPC 基线相比，CASCADE 在 LLM 令牌使用上降低了数十倍，保持了相似或更丰富的行为多样性；在 NPC 数量扩展时，性能保持稳定，且计算成本基本不随规模增大。

**⚠️ 局限性**

局限性包括单向自上而下的指令流，缺乏微观异常向宏观的反馈；指令广播不考虑空间拓扑与信息衰减；模块化程度受限，当前仅覆盖基础经济与安全等领域。

---

## 411. Same Feedback, Different Source: How AI vs. Human Feedback Attribution and Credibility Shape Learner Behavior in Computing Education

**arXiv ID:** 2604.03075 | [PDF](https://arxiv.org/pdf/2604.03075v1)

**作者:** Caitlin Morris `[一作]` (MIT), Pattie Maes `[通讯]` (MIT)

**通讯引用:** 19297 | [OpenAlex ID](https://openalex.org/A5081457786)

**关键词:** `37e2bb26-449b-4ccc-a077-e4289fb90a8e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过三组实验（AI即时、AI延迟、TA延迟）让148名学习者完成创意编码教程，并用Claude Sonnet 4生成相同内容但源属性不同的反馈，探究源属性和交付时延对学习者行为的影响。

**💡 创新点**

首次在控制交付时延的情况下分离源属性效应，揭示可信人类反馈可提升学习者投入，而不可信人类反馈反而导致表现低于透明AI，并通过过程与输出指标分别验证了两种效应。

**🔧 技术方法**

使用大语言模型Claude Sonnet 4生成反馈；实验设计与三组对照；混合效应模型和描述性统计；Prolific平台招募；自我报告与行为日志数据分析。

**📊 数据集**

内部生成的数据：148名Prolific受试者的代码提交、反馈内容、时间记录、代码复杂度评分；无外部公开数据集，全部为实验中生成的内容。

**📈 对比分析**

采用计划对比（AI-延迟 vs TA-相信、AI-即时 vs AI-延迟）进行统计检验；结果显示，可信人类源显著增加学习时间（d≈0.61），交付延迟显著提升代码复杂度（d≈0.55）；可信度低者表现最差（d≈0.77）。

**⚠️ 局限性**

结果基于后测可信度划分，缺乏随机可信度操纵；TA子组样本量小；Prolific样本AI素养可能偏高，结果可能不适用于一般课堂；机制测量采用单项问卷，缺乏验证性；仅在创意编码情境下验证，尚需在其他学习领域复现。

---

## 412. Credential Leakage in LLM Agent Skills: A Large-Scale Empirical Study

**arXiv ID:** 2604.03070 | [PDF](https://arxiv.org/pdf/2604.03070v1)

**作者:** Zhihao Chen `[一作]` (Fujian Normal University), Zhiqiang Li `[通讯]` (Independent Researcher)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `9cc9baba-5356-466d-81ff-d80028d90279` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

对170,226个 SkillsMP 生态中的技能进行大规模经验研究，使用静态分析与 sandbox 动态测试，识别出 520 个凭证泄露实例，共计 1,708 条安全问题，其中 3.1% 的技能存在泄露。

**💡 创新点**

首次系统性分析 agent 技能的凭证泄露，提出 10 种泄露模式（4 类因开发者疏忽，6 类因恶意构造），揭示自然语言与代码交叉模态的独特风险，并对现有秘密检测方法进行补充。

**🔧 技术方法**

采用正则表达式与 AST 静态扫描、NL 语义约束匹配、prompt 注入检测、sandbox 多轮动态验证、日志抓取、人工审计等技术组合，实现跨模态凭证泄露检测。

**📊 数据集**

基于 SkillsMP 2026 年 2 月 12 日的完整快照，采样 17,022 条技能（约 10%），包含 37,409 源码文件与 17,022 条自然语言描述。

**📈 对比分析**

与传统的仅基于代码的秘密扫描工具对比，交叉模态检测在 76.3% 的案例中能唯一发现泄露；静态+动态结合的准确率显著高于单一方法，漏检率从 30% 降至约 8%，但仍存在未覆盖的多语言与深层调用路径。

**⚠️ 局限性**

研究仅涵盖 Python 与 JavaScript 的单文件 AST 分析，未考虑多语言、跨文件变量传播及深层递归调用；依赖单一平台 SkillsMP，缺乏对其他 agent 技能市场的验证，且动态测试受限于沙箱可达性与 LLM 非确定性。

---

## 413. Gram-MMD: A Texture-Aware Metric for Image Realism Assessment

**arXiv ID:** 2604.03064 | [PDF](https://arxiv.org/pdf/2604.03064v1)

**作者:** Joé Napolitano `[一作]` (AMIAD), Pascal Nguyen `[通讯]` (AMIAD)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了一种基于中间特征Gram矩阵的分布距离度量 Gram-MMD，用来评估生成图像的真实度。

**💡 创新点**

创新点在于：①使用预训练网络的中间层特征Gram矩阵捕捉纹理与结构信息；②将对称Gram矩阵上三角部分向量化，获得与图像尺寸无关的固定维度表示；③在此向量空间使用MMD计算真实与生成分布的差异；④通过控制降质实验的meta‑metric协议对超参数进行系统化调优。

**🔧 技术方法**

技术要点包括：预训练模型（DINOv2、DC‑AE、SD‑VAE、VGG19、LPIPS‑VGG等）特征提取；Gram矩阵构造与向量化；标准化与高斯RBF核下的MMD估计；以及Spearman、Kendall统计量评估。

**📊 数据集**

使用的数据集有：MS‑COCO（用于超参数调优）、KADID‑10k（图像降质评价）、RAISE（AI生成图像的真实度评分）以及KITTI、Virtual KITTI、Stanford Cars（跨域驾驶实验）。

**📈 对比分析**

在KADID‑10k、RAISE以及跨域驾驶实验中，Gram‑MMD在Spearmanρ和Kendallτ上均优于基准CMMD（以及FID等），能够更好地捕捉纹理细节、在高分辨率图像中保持一致性，并正确区分真实与合成分布。

**⚠️ 局限性**

局限性包括：①Gram‑MMD计算需要高维向量（d(d+1)/2），对GPU内存和计算时间有较高要求；②在缺乏足够纹理信息或过度噪声的图像中，Gram特征可能不如语义特征稳定；③目前尚未充分验证在多模态或极端光照/天气条件下的鲁棒性。

---

## 414. Querying Structured Data Through Natural Language Using Language Models

**arXiv ID:** 2604.03057 | [PDF](https://arxiv.org/pdf/2604.03057v1)

**作者:** Hontan Valentin-Micu `[一作]` (National University of Science and Technology POLITEHNICA), Popovici Dan-Matei `[通讯]` (National University of Science and Technology POLITEHNICA)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `67630363-6be0-4f51-ab05-7198250671a5` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `a05fcc20-6870-48b1-abb6-44c47d7cde76` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

本文开发了一套基于开源LLM的自然语言查询系统，能够自动生成可执行的数据库查询语句并返回结构化数据答案。

**💡 创新点**

创新点在于构建了一个仅使用8B级别模型、通过合成问答对自动化生成的训练数据和QLoRA 4‑bit量化微调的完整工作流，从而在不依赖大型专有模型的前提下实现高精度查询生成。

**🔧 技术方法**

采用DeepSeek R1‑Distill‑8B模型、LoRA适配器、QLoRA 4‑bit量化、结构化问答合成算法、工具调用提示等技术，训练并在RTX 3090上推理。

**📊 数据集**

使用德国航空航天中心（DLR）收集的 Durangaldea 区域医疗、超市、药店可达性（约10万条记录）数据集进行实验。

**📈 对比分析**

与 GPT‑4、Gemini、DeepSeek R1 等SOTA模型在零样本情境下对比，模型在语义变体集上实现 94.2% 的 Exact Match，未见位置集上 89%，BLEU‑4 和 ROUGE‑L 近 1.0，整体性能与大型模型相近。

**⚠️ 局限性**

局限性包括：仅针对单一数据集设计，难以扩展到多数据集/多工具场景；对低资源语言（如巴斯克语）性能显著下降；缺乏自动错误恢复机制，且合成数据生成仍需人工校验。

---

## 415. A Tsetlin Machine-driven Intrusion Detection System for Next-Generation IoMT Security

**arXiv ID:** 2604.03205 | [PDF](https://arxiv.org/pdf/2604.03205v1)

**作者:** Rahul Jaiswal `[一作]` (University of Agder), Ole-Christoffer Granmo `[通讯]` (University of Agder)

**通讯引用:** 3320 | [OpenAlex ID](https://openalex.org/A5071922620)

**关键词:** `b011fd49-2b66-44b7-8ab9-cd8d3a13f67e` `3855fcda-48ef-4070-a15e-803cd5c84d83` `aeb1d087-87bb-48bf-8e0e-d19fc2260534`

**🎯 论文内容**

提出了一种基于Tsetlin Machine的可解释型入侵检测系统，用于识别IoMT网络中的多种网络攻击。

**💡 创新点**

创新点在于利用Tsetlin Machine构建可解释的逻辑规则并结合类别投票与热力图实现攻击识别与决策可视化。

**🔧 技术方法**

采用Tsetlin Machine与传统机器学习算法（DT、RF、XGBoost、LGBM、KNN、NB、LR、NN）进行对比实验。

**📊 数据集**

使用CICIoMT‑2024公开数据集，该数据集包含蓝牙、MQTT和Wi‑Fi三种协议下的多种攻击与正常流量。

**📈 对比分析**

与传统分类器及文献中现有方法比较，TM在二分类中准确率99.5%，多分类中准确率可达90.7%，显著优于对手，且推理时间在可接受范围内。

**⚠️ 局限性**

局限性包括对不同网络环境和更细粒度攻击类型的泛化能力不足，以及在极端类不平衡情况下的表现需进一步提升。

---

## 416. Real-Time Surrogate Modeling for Personalized Blood Flow Prediction and Hemodynamic Analysis

**arXiv ID:** 2604.03197 | [PDF](https://arxiv.org/pdf/2604.03197v1)

**作者:** Sokratis J. Anagnostopoulos `[一作]`, Nikolaos Stergiopulos `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `64443552-63e0-44b5-906f-d90fe95c5a1b` `11828d4d-5ed2-4c17-8f38-5c7a47e57054` `e15e3743-5ee0-4d5f-813d-d146868082fc` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `bb57609f-8351-4b1b-85e4-3afa07da95d6`

**🎯 论文内容**

开发了基于 1‑D 动脉模型的机器学习 surrogate，生成符合 Asklepios 临床统计的虚拟人群，用于实时预测血压和心输出量，并在临床数据上验证 CO 与 cSBP。

**💡 创新点**

①结合真实临床协方差生成虚拟队列；②利用 surrogate 实时剔除非生理参数，显著降低合成数据成本；③在逆向问题中量化终端阻抗对 CO 估计的不确定性；④展示通过额外 radial 压力测量可显著提升 CO 预测精度。

**🔧 技术方法**

使用 1‑D 动脉动力学模型、量子 Latin Hypercube 采样、深度全连接神经网络（forward 与 inverse 模式）、AdamW 优化、Huber 损失、批归一化、dropout；评估采用相关系数、Bland‑Altman、误差统计等。

**📊 数据集**

主要使用 Asklepios 临床队列（约 1500 名无心血管疾病的中青年患者）进行参数映射和虚拟数据生成，并在 20 名健康志愿者的临床测量数据上进行验证。

**📈 对比分析**

与完整物理模型的 2000 例 1‑D 解对比，surrogate 在实时推断下的血压误差 ≤ ±5 mmHg、CO 误差 ≤ ±0.03 L/min，训练过程无过拟合；逆向任务中仅可测参数时 R²≈0.94，加入 radial 压力或终端阻抗后 R²≥0.998，误差 <0.03 L/min。

**⚠️ 局限性**

仅适用于健康或近健康状态；终端 Windkessel 参数无法直接测量，导致逆向估计对其敏感；缺乏疾病、老年等多样性；临床数据噪声大限制 CO 预测精度；需在多中心更大样本上验证。

---

## 417. Reflective Context Learning: Studying the Optimization Primitives of Context Space

**arXiv ID:** 2604.03189 | [PDF](https://arxiv.org/pdf/2604.03189v1)

**作者:** Nikita Vassilyev `[一作]` (Contextual AI), Shikib Mehri `[通讯]` (Contextual AI)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `5b4c1114-4a70-478e-9921-2514ee03850d` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d603a949-d0a9-40d8-bcb8-e02e842b97f2` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出并系统评估反射式上下文学习（RCL）框架，探讨多种经典优化原语在上下文空间学习中的作用。

**💡 创新点**

将批量、信用分配、辅助损失、失败重放、动量等参数空间优化手段迁移到基于LLM反射的上下文学习循环中，并证明其组合能显著提升性能。

**🔧 技术方法**

使用LLM反射生成诊断、突变器执行局部编辑；引入任务批量、分组rollout、双轨迹信用分配、辅助损失头、失败重放缓冲、滚动优化状态记录等技术；模型采用Claude Opus 等大语言模型。

**📊 数据集**

在 AppWorld、BrowseComp+、RewardBench2 三大基准上进行实验。

**📈 对比分析**

与 ACE 基线及 GEPA 对比，单独或组合加入原语在三大基准上平均提升 5–15+ 分；最优组合在 AppWorld Normal 89.3%、BrowseComp+ 89.1%、RewardBench2 71.9%，显著优于基线。

**⚠️ 局限性**

缺点包括：不同任务和模型对原语组合的敏感度高，单独原语与组合效果不一致；缺少自适应原语选择机制；在高度确定性任务中某些原语可能适得其反；未验证在持续部署场景下的长期稳定性。

---

## 418. SFFNet: Synergistic Feature Fusion Network With Dual-Domain Edge Enhancement for UAV Image Object Detection

**arXiv ID:** 2604.03176 | [PDF](https://arxiv.org/pdf/2604.03176v1)

**作者:** Wenfeng Zhang `[一作]` (Chongqing Normal University), Lei Huang `[通讯]` (Ocean University of China)

**通讯引用:** 7611 | [OpenAlex ID](https://openalex.org/A5020608484)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e0540dec-d77f-42db-94ae-d039248f6393` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出了 SFFNet 模型，专门针对无人机航拍图像中的目标检测任务，实现了端到端的检测流程。

**💡 创新点**

创新点在于：①引入多尺度动态双域耦合（MDDC）模块，在频域与空间域双重耦合下实现边缘信息增强；②设计协同特征金字塔网络（SFPN），通过线性可变形卷积与宽域感知模块（WPM）实现对几何与长程语义信息的高效融合。

**🔧 技术方法**

主要技术包括：双域频谱滤波与自适应增益、线性可变形卷积、宽域感知（大核与条纹卷积）以及多尺度特征融合与自适应权重学习。

**📊 数据集**

使用的公开数据集为 VisDrone（大规模无人机航拍目标检测数据集）和 UAVDT（包含城市环境中的车类目标数据集），并在两者上进行训练与评估。

**📈 对比分析**

与 YOLO 系列、Faster‑RCNN、Transformer‑DETR 等多种先进检测器比较，SFFNet 在 VisDrone 的 AP 上提升 1.2%–1.6%，在小目标 AP 上提升 2.7%–4.5%，且在轻量级模型上参数量更少，整体性能优于同类方法。

**⚠️ 局限性**

主要局限在于对大尺寸目标的检测效果仍不理想，且在极端光照或遮挡条件下性能下降；未来需要引入自适应锚框或更强的上下文建模策略。

---

## 419. Detecting and Correcting Reference Hallucinations in Commercial LLMs and Deep Research Agents

**arXiv ID:** 2604.03173 | [PDF](https://arxiv.org/pdf/2604.03173v1)

**作者:** Delip Rao `[一作]` (University of Pennsylvania), Chris Callison-Burch `[通讯]` (University of Pennsylvania)

**通讯引用:** 21511 | [OpenAlex ID](https://openalex.org/A5068508539)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `3855fcda-48ef-4070-a15e-803cd5c84d83` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对大型语言模型和深度研究代理生成的引用URL的可靠性进行系统评估，测量URL失效和虚假率，并提出验证工具。

**💡 创新点**

首次量化不同模型、服务商和学科领域的URL失效/幻觉比例，并将失效拆分为真实失效和完全幻觉；同时发布开源URL验证工具。

**🔧 技术方法**

使用HTTP HEAD/GET检查、Wayback Machine API分类、深度学习模型（Google、OpenAI、Anthropic等）以及代理自检循环。

**📊 数据集**

DRBench（约53k URL）与ExpertQA（约168k URL，32学科）两个大规模数据集。

**📈 对比分析**

对10个模型（包括深度研究代理和搜索增强LLM）进行对比，发现深度代理生成量大但幻觉率更高；域差异显著；自检工具可将失效率从5‑18%降至<1%。

**⚠️ 局限性**

局限包括Wayback覆盖不完整导致幻觉率低估、403/未知状态影响、仅关注URL而非文本片段或文献元数据，工具在被阻断/付费页面上的准确性有限。

---

## 420. HyperFitS -- Hypernetwork Fitting Spectra for metabolic quantification of ${}^1$H MR spectroscopic imaging

**arXiv ID:** 2604.03150 | [PDF](https://arxiv.org/pdf/2604.03150v1)

**作者:** Paul J. Weiser `[一作]` (Massachusetts General Hospital), Ovidiu C. Andronesi `[通讯]` (Massachusetts General Hospital)

**通讯引用:** 6291 | [OpenAlex ID](https://openalex.org/A5056477611)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `e15e3743-5ee0-4d5f-813d-d146868082fc` `5663785e-e4e3-40e4-b675-cbd84d82d1f9`

**🎯 论文内容**

提出了HyperFitS，一种可配置的超网络，用于快速、准确地对三维全脑^1H MRSI进行代谢定量；

**💡 创新点**

创新点在于通过超网络实现水抑制校正和基线可调参数，无需重新训练即可适应不同场强和协议；

**🔧 技术方法**

结合深度学习超网络、物理模型编码器、贝叶斯不确定性估计和自监督训练方法；

**📊 数据集**

使用在3T和7T MR扫描仪上采集的9个7T和12个3T的全脑^1H-FID MRSI数据，分辨率分别为10mm、3.4mm和2mm；

**📈 对比分析**

与金标准LCModel进行定量比较，HyperFitS在保持与LCModel相近的代谢分布和浓度结果的同时，将单个MRSI体积的处理时间从数小时缩短至十几秒，显著提升速度；

**⚠️ 局限性**

仅针对短回波^1H-MRSI训练，无法直接用于长回波或其他核素的拟合，若需适用于其他协议需重新训练；

---

## 421. Reliability Gated Multi-Teacher Distillation for Low Resource Abstractive Summarization

**arXiv ID:** 2604.03192 | [PDF](https://arxiv.org/pdf/2604.03192v1)

**作者:** Dipto Sumit `[一作]` (BRAC University), Farig Yousuf Sadeque `[通讯]` (BRAC University)

**通讯引用:** 351 | [OpenAlex ID](https://openalex.org/A5009105388)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `8d10c613-917e-4880-9716-17789f50e119` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文研究了在低资源条件下的多教师知识蒸馏，提出了可靠性感知的 token 级蒸馏策略，探讨了其对可压缩生成模型的效果；

**💡 创新点**

创新点在于引入 EWAD（基于熵加权的一致性门控蒸馏）和 CPDP（基于容量比例的几何约束），实现了教师可靠性动态路由和学生分布位置的几何自适应；

**🔧 技术方法**

使用了 logit 级 KD、伪标签跨架构蒸馏、温度自适应缩放、MapReduce 长文档处理以及多层损失组合（CE+KL+inter+EWAD+CPDP）；

**📊 数据集**

实验数据集包括 BanglaT5 的 BTS 与 BanSum 两个 Bangla 摘要数据集、Qwen-2.5 家族的自蒸馏实验，以及基于 mT5-XLSum 的十种语言的跨语言伪标签蒸馏；

**📈 对比分析**

通过 ROUGE‑1/2/L、BLEU、BERTScore、语义相似度及多评审 LLM 的人类校验对比，发现单一 logit KD 在绝大多数指标上表现最好，EWAD/CPDP 在短摘要任务上可提升语义相似度，但在长摘要和高容量学生上效果不佳；

**⚠️ 局限性**

主要局限包括人类评估样本有限、语言覆盖未覆盖极端语言类型、未充分探索大容量差距下的效果、摘要长度对蒸馏效果的显著影响，以及仅在 BanglaT5 与 Qwen-2.5 这两大模型家族中验证，缺乏更广泛的模型与任务验证。

---

## 422. ProtoFlow: Mitigating Forgetting in Class-Incremental Remote Sensing Segmentation via Low-Curvature Prototype Flow

**arXiv ID:** 2604.03212 | [PDF](https://arxiv.org/pdf/2604.03212v1)

**作者:** Jiekai Wu `[一作]` (Juntendo University), Pengbin Feng `[通讯]` (University of Southern California)

**通讯引用:** 1510 | [OpenAlex ID](https://openalex.org/A5033496838)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `729e5870-4135-47f5-97f2-e3974d07b5dc` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `40105733-5154-44cd-8090-a8cab9e64b07` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出一种时间感知的原型流（ProtoFlow）框架，用于遥感图像的类增量语义分割，模型把类别原型视为随时间演化的轨迹，并在训练过程中正则化轨迹平滑与类间分离。

**💡 创新点**

创新点在于将增量学习转化为原型动力学系统，利用显式的时间条件向量场捕捉原型随季节、城市、传感器等变化的连续演化；同时引入低曲率与类间分离正则化，使得原型几何保持稳定，减少遗忘。

**🔧 技术方法**

主要技术包括：时间条件向量场（两层MLP）预测原型速度；流一致性损失、曲率正则化、分离正则化；传统的交叉熵分割损失与教师蒸馏；对原型做 L2 归一化；使用前馈编码器-解码器网络（HRNet-W48 + 轻量化头）。

**📊 数据集**

使用四个遥感分割基准：DeepGlobe、ISPRS Vaihingen、ISPRS Potsdam、LoveDA（域增量），以及大规模 iSAID 与 GCSS 用于扩展验证。

**📈 对比分析**

与多种通用与遥感专用的增量分割方法（MiB、DKD、LAG、APR、CoMBO、CoGaMiD、HIGCISS、GSMF-RS-DIL、MiSSNet、STCL-DRNet、MiR）对比；ProtoFlow 在 mIoU_all、mIoU_old、mIoU_new 等指标上普遍领先 1.5–2.0 点，并在遗忘率、交叉域 mIoU 等方面表现更佳。

**⚠️ 局限性**

局限性包括：模型仍需在预先定义的增量协议下训练，无法直接处理开放词汇或无监督增量；时间向量场的表达仍相对简单，可能难以捕捉更复杂的非线性变化；对高维特征空间中的几何正则化仍存在尺度不敏感的问题。

---

## 423. Help Converts Newcomers, Not Veterans: Generalized Reciprocity and Platform Engagement on Stack Overflow

**arXiv ID:** 2604.03209 | [PDF](https://arxiv.org/pdf/2604.03209v1)

**作者:** Lenard Strahringer `[一作]` (Stanford University), Kai Riemer `[通讯]` (University of Sydney)

**通讯引用:** 4053 | [OpenAlex ID](https://openalex.org/A5039663862)

**关键词:** `2f9b095f-c896-4240-9f90-c17a5e9a2c39` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

对Stack Overflow问答数据进行匹配差分-差分生存分析，检验收到回答后用户的帮助倾向；

**💡 创新点**

首次将时间变异的差分-差分设计与Cox比例风险模型结合，避免了传统观测研究中活跃度混杂；

**🔧 技术方法**

采用匹配的Cox比例风险模型（含时间变异协变量）以及倾向得分匹配；

**📊 数据集**

基于2025年4月1日前的完整Stack Overflow数据集，包含超过2100万提问与回答记录；

**📈 对比分析**

通过倾向得分匹配对照组与处理组，在四天观察窗口内比较帮助事件风险，结果显示平均5.8%帮助率提升，随用户经验递减，快答对新人更显著；

**⚠️ 局限性**

研究仅识别行为响应，无法确定情感或道德机制；对时间变化特征（如会话结束）缺乏直接测量；未检验未回答问题的潜在自选择；

---

## 424. Hierarchical Planning with Latent World Models

**arXiv ID:** 2604.03208 | [PDF](https://arxiv.org/pdf/2604.03208v1)

**作者:** Wancong Zhang `[一作]` (FAIR at Meta), Nicolas Ballas `[通讯]` (FAIR at Meta)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `fa81e2aa-eb25-4aba-a919-7efd247b3885` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `57a58b01-81b4-4d75-a45c-2e891f272b50` `9ce7179e-700c-4310-ac2b-91df50ded46e` `5a41884c-404f-4688-a89c-aa238c10fe68` `c773407a-6119-4871-b8b3-1e7ae17a6851`

**🎯 论文内容**

提出了 HWM（Hierarchical World‑Model）框架，利用多层潜在世界模型实现层次化模型预测控制，以零样本方式完成长时程任务。

**💡 创新点**

创新点在于：通过共享潜在空间将不同时间尺度的世界模型耦合，使用动作编码器压缩低层动作为宏动作，消除子政策/技能学习需求，实现无监督子目标生成。

**🔧 技术方法**

核心技术包括：潜在世界模型、CEM 采样搜索、动作编码器、层次化 MPC、以及对 VJEPA2‑AC、DINO‑WM、PLDM 等现有模型的复用。

**📊 数据集**

使用数据集：130 小时无标签真实机器人轨迹（DROID、RoboSet），Frankia pick‑&‑place、drawer、Push‑T 延长序列，以及 25 训练、20 测试的 MuJoCo Diverse Maze。

**📈 对比分析**

与 VJEPA2‑AC、VLA（Octo、π₀‑FAST‑DROID、π₀.5‑DROID）、DINO‑WM flat、GCIQL、HIQL、HILP 等基线比较，HWM 在 pick‑place 70% vs 0%，Push‑T 61% vs 17%，Maze 83% vs 44% 等场景上显著提升，并且计算成本约为单层方法的 1/3~1/4。

**⚠️ 局限性**

局限性在于：长时程预测仍受累积误差和模型不确定性影响，层次化规划仅采用自上而下的方式，缺乏跨层反馈；需要更抽象的表示、更强的不确定性建模和更交互式的层次规划算法来进一步提升性能。

---

## 425. PR3DICTR: A modular AI framework for medical 3D image-based detection and outcome prediction

**arXiv ID:** 2604.03203 | [PDF](https://arxiv.org/pdf/2604.03203v1)

**作者:** Daniel C. MacRae `[一作]` (University of Groningen), Lisanne V. van Dijk `[通讯]` (University of Groningen)

**通讯引用:** 6951 | [OpenAlex ID](https://openalex.org/A5028995506)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `cc175879-ab65-4aa9-b58a-f6100a057dbf` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `90291a0e-9d36-4a08-9a16-89ce846d923f` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `e0f78f5f-72c7-4ad2-8f91-7921d7e8406f`

**🎯 论文内容**

开发了PR3DICTR框架，提供端到端的3D医学影像分类模型训练、评估与实验管理流程；

**💡 创新点**

以模块化、配置驱动、标准化为核心，将多模态影像、表格数据融合、数据增强、超参数自动搜索等功能统一在一个易用框架中，显著降低实现成本与重复工作；

**🔧 技术方法**

基于PyTorch+MONAI实现CNN/Transformer骨干（ResNet、DenseNet、EfficientNetV2、ConvNeXt、ViT、TransRP），配合Optuna超参搜索、Weights & Biases实验追踪、MixUp等数据增强与正则化技术；

**📊 数据集**

以TCIA的NSCLC‑Radiomics数据库（胸部CT、肺分割和临床表格）为示例进行性别二分类；

**📈 对比分析**

使用K折交叉验证和单独测试集，评估AUC、准确率、校准曲线等指标；在示例实验中模型几乎完美区分男女，且校准效果良好；

**⚠️ 局限性**

可重复性受硬件与浮点误差影响；预处理与后处理步骤仍需手动决策；目前尚缺乏多类别分类、置信度估计与模型可解释性等功能。

---

## 426. From Industry Claims to Empirical Reality: An Empirical Study of Code Review Agents in Pull Requests

**arXiv ID:** 2604.03196 | [PDF](https://arxiv.org/pdf/2604.03196v1)

**作者:** Kowshik Chowdhury `[一作]` (Kennesaw State University), Shazibul Islam Shamim `[通讯]` (Kennesaw State University)

**通讯引用:** 147 | [OpenAlex ID](https://openalex.org/A5019761662)

**关键词:** `1bc454a9-3d09-46c3-87e9-f7a9c36911df` `e2c980c8-7137-48ee-b99f-3fbde4cf81e7` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文通过对19,450条PR中的3,109条进行分析，比较了人类审查者、仅使用代码审查代理（CRA）和混合审查的合并率和放弃率。

**💡 创新点**

创新点在于首次系统量化CRA评论的信噪比并关联其对PR放弃的影响。

**🔧 技术方法**

采用关键词层级法、信噪比计算、chi‑square检验以及手工标注交叉验证等技术。

**📊 数据集**

使用AIDev GitHub PR审查数据集，包含19,450条评论，筛选后3,109条PR。

**📈 对比分析**

通过对比人类仅审查、CRA仅审查以及混合审查的合并率，发现CRA仅审查合并率仅45.2%，且信噪比低导致放弃率高，统计显著。

**⚠️ 局限性**

局限性包括信噪比依赖关键词可能漏判、数据仅来自开源AI相关PR、未探究因果关系和跨平台可推广性。

---

## 427. The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling

**arXiv ID:** 2604.03191 | [PDF](https://arxiv.org/pdf/2604.03191v1)

**作者:** Takuya Shiba `[一作]` `[通讯]` (Shibattic Inc), Takuya Shiba (Shibattic Inc)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `57a58b01-81b4-4d75-a45c-2e891f272b50` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在机器人操控任务中研究视觉编码器升级对两类动作表征（离散分词OAT与连续扩散策略DP）的影响。

**💡 创新点**

提出“Compression Gap”理论，揭示信息瓶颈位置决定视觉编码升级是否能传递至任务性能。

**🔧 技术方法**

使用信息瓶颈分析、离散分词器（FSQ）、扩散策略、因子实验、编码器质量梯度与代码本容量实验等技术。

**📊 数据集**

使用LIBERO-10基准，配合ResNet‑18、SigLIP、DINOv2 ViT‑L/14、SigLIP 2等视觉编码器。

**📈 对比分析**

通过八个因子实验、编码器梯度实验和代码本容量实验比较，结果显示DP对编码器升级高度敏感，OAT受限于代码本；DP在大模型上提升显著。

**⚠️ 局限性**

仅在单一基准、有限模型规模和编码器种类上验证，未覆盖更大规模模型、其他基准或真实世界环境。

---

## 428. Gradient Boosting within a Single Attention Layer

**arXiv ID:** 2604.03190 | [PDF](https://arxiv.org/pdf/2604.03190v1)

**作者:** Saleh Sargolzaei `[一作]` (University of Windsor), Saleh Sargolzaei `[通讯]` (University of Windsor)

**通讯引用:** 201 | [OpenAlex ID](https://openalex.org/A5007848902)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

在单层注意力中加入第二次注意力回合，对第一回合的残差进行学习式校正，从而实现局部的梯度提升。

**💡 创新点**

创新点在于：①将梯度提升（gradient boosting）原理直接嵌入单层注意力；②第二回合使用独立的 QKV 投影和可学习的逐维门控，能够重新关注第一回合忽略的 token；③证明单次迭代会消除 query 的正交信息，阐明迭代注意力的失败原因。

**🔧 技术方法**

使用的技术包括：Transformer 的多头注意力、梯度提升（MART）框架、Hopfield 网络理论、可学习门控（sigmoid+MLP）、软最大化与残差校正。

**📊 数据集**

主要数据集为 WikiText-103 的 10M-token 子集，用于语言建模实验；也使用了合成的模式去噪任务做 ablation。

**📈 对比分析**

与标准注意力、Twicing Attention（共享同一权重的校正）以及参数匹配的更宽模型进行对比。实验结果显示：梯度提升注意力在测试集上达到 67.9 的 perplexity，分别比标准注意力低 4.3（6.0%）、Twicing 注意力低 1.7、参数匹配模型低 1.1，证明改进主要来自架构创新而非纯参数增加。

**⚠️ 局限性**

局限性：实验仅在小规模 7–9M 参数模型上完成，未验证在 100M–1B 规模下的效果；额外的注意力回合带来约 18% 参数增量和 50% FLOPs 增加，可能影响低延迟应用；尚未探索在大规模预训练模型上的微调适用性。

---

## 429. Biologically Realistic Dynamics for Nonlinear Classification in CMOS+X Neurons

**arXiv ID:** 2604.03187 | [PDF](https://arxiv.org/pdf/2604.03187v1)

**作者:** Steven Louis `[一作]`, Vasyl Tyberkevych `[通讯]` (Oakland University)

**通讯引用:** 905 | [OpenAlex ID](https://openalex.org/A5107825823)

**关键词:** `aea6b09c-069e-4d88-8dd1-371f7abba620` `c39d1b1f-fb4e-4609-be16-ca06609fa0ac` `29aaa6b5-cc4b-4e8b-b67e-05d983eb740c` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

研究了一种基于CMOS+MTJ的脉冲神经网络，证明其天然的阈值激活、响应迟滞和绝对不应期三种神经动力学特性能够实现非线性分类任务（XOR）。

**💡 创新点**

创新点在于将磁隧道结（MTJ）的自发磁化动力学直接作为计算资源，利用其原生阈值、时延和不可复位特性完成非线性推断，完全不需要额外的控制电路或复杂的单元设计。

**🔧 技术方法**

使用电路仿真和梯度反向传播（基于脉冲时序的误差函数）对NMOS+MTJ神经元进行训练，构建多层SNN并通过模拟实现XOR分类。

**📊 数据集**

数据集仅为经典的4条XOR真值表，用以验证网络对非线性关系的学习与推断能力。

**📈 对比分析**

与传统多层感知器或其他CMOS神经元的对比未给出定量指标，但实验显示网络能够在单脉冲、单周期内准确给出XOR输出，验证了自带非线性动力学的有效性。

**⚠️ 局限性**

局限性包括：仅验证了最小规模的XOR任务；网络训练依赖精确的脉冲时序和单脉冲假设；对更大规模、噪声环境或多类别任务的可扩展性尚未探讨。

---

## 430. PRISM: LLM-Guided Semantic Clustering for High-Precision Topics

**arXiv ID:** 2604.03180 | [PDF](https://arxiv.org/pdf/2604.03180v1)

**作者:** Connor Douglas `[一作]` (New York University), Joseph Aylett-Bullock `[通讯]`

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `9ce7179e-700c-4310-ac2b-91df50ded46e` `8d10c613-917e-4880-9716-17789f50e119` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

提出PRISM框架，将LLM的高质量语义信息通过少量标签蒸馏到轻量化编码器，再用阈值聚类实现高精度主题建模。

**💡 创新点**

创新点在于将LLM标签蒸馏与阈值聚类相结合，实现零LLM推理成本、局部语义高分辨率的主题聚类。

**🔧 技术方法**

采用学生-教师蒸馏、CoSENT对比损失、阈值聚类（如自定义阈值+HDBSCAN）以及SentenceTransformers编码器。

**📊 数据集**

使用HumAID、LIAR、IMDB三大公开文本语料进行实验。

**📈 对比分析**

与Top2Vec、BERTopic以及直接对LLM嵌入聚类对比，PRISM在AUC和AUPC指标上均优于对手，尤其在少量LLM查询下取得显著提升。

**⚠️ 局限性**

局限在于仍需依赖LLM生成标签，查询成本和标签质量会影响效果；在极大规模语料下阈值聚类的计算开销仍存在，且未深入探讨主动学习或多模态信号。

---

## 431. Exclusive and Shared Electric Flying Taxis: Evidence on Modal Shares, Stated Reasons, and Modal Shifts

**arXiv ID:** 2604.03166 | [PDF](https://arxiv.org/pdf/2604.03166v1)

**作者:** Nael Alsaleh `[一作]` (American University of Ras Al Khaimah), Zainab Islam `[通讯]` (American University of Ras Al Khaimah)

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f` `337e632d-5d88-4e08-b332-1e58d8df0f5e` `5b4c1114-4a70-478e-9921-2514ee03850d` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e`

**🎯 论文内容**

通过设计并实施基于SP实验的问卷，调研了阿拉伯联合酋长国旅客对电动飞行出租车（共享与专属）在不同距离、拥堵、出行目的及周末/工作日情境下的模式偏好。

**💡 创新点**

首次将共享与专属飞行出租车进行并行比较，并将多种情境因素（距离、拥堵、出行目的、周末/工作日）纳入SP设计，提供了针对UAE地区的首批实证结果。

**🔧 技术方法**

使用SP实验设计、描述性统计、频率分布及卡方检验方法评估模式选择与情境关系。

**📊 数据集**

采用SurveyMonkey平台收集的213名阿联酋居民问卷数据，包含人口学特征、现有出行行为、态度与虚拟情境下的模式选择。

**📈 对比分析**

通过比较各模式在不同情境下的占比并使用卡方检验检验显著性，结果表明拥堵和距离显著影响偏好，飞行出租车整体占比22.6%，共享模式相较专属表现更好。

**⚠️ 局限性**

样本在某些酋长国代表性不足，未构建随机效用模型，缺乏对个体偏好异质性的量化分析。

---

## 432. CAMEO: A Conditional and Quality-Aware Multi-Agent Image Editing Orchestrator

**arXiv ID:** 2604.03156 | [PDF](https://arxiv.org/pdf/2604.03156v1)

**作者:** Yuhan Pu `[一作]` (Hong Kong University of Science and Technology (Guangzhou)), Jiaheng Wei `[通讯]` (Hong Kong University of Science and Technology (Guangzhou))

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `dd8c26bc-3e4a-44cd-ab1a-e3ffc95d5769` `ba576bd1-e51d-44e8-8077-fc943b333c93` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出了一种分层多智能体框架 CAMEO，用以将条件图像编辑从一次性生成转变为结构化的、带有质量评估和反馈的迭代过程。

**💡 创新点**

创新点包括：1) 通过分层多智能体（规划、执行、监管）实现编辑任务的模块化与可控；2) 引入自适应参考 grounding，只在需要时加入文本/视觉先验；3) 在生成循环内部嵌入质量评估与迭代细化，将编辑视为闭环控制。

**🔧 技术方法**

采用扩散模型作为编辑骨干（Qwen Image Edit Plus、FLUX 2 Pro、Seedream 4.5、Nano Banana Pro），配合多模态评估器（Qwen3-VL-Plus、GPT‑4o、Gemini‑2.5、Claude‑Opus‑4.5）以及多智能体协同策略。

**📊 数据集**

使用了 BDD100K 数据集进行道路异常插入任务，并构建了基于 Pexels 图像的人体姿态切换基准（包含原图、指令、参考姿态和编辑结果）。

**📈 对比分析**

通过视觉‑语言评估者进行成对比较（赢/输/平分），并记录平均分；CAMEO 在所有骨干模型和评估者上平均提升约 20% 的赢率，且分数均高于直接单步编辑；人类偏好测试亦证实其优势。

**⚠️ 局限性**

局限性在于迭代反馈和多智能体协调带来的计算开销，且整体性能仍受底层编辑与评估模型能力限制；未来需要更可靠的评估指标和更高效的迭代机制。

---

## 433. Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control

**arXiv ID:** 2604.03147 | [PDF](https://arxiv.org/pdf/2604.03147v1)

**作者:** Lihao Sun `[一作]` (Shanghai AI Lab), Jing Shao `[通讯]` (Shanghai AI Lab)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `9cc9baba-5356-466d-81ff-d80028d90279` `57a58b01-81b4-4d75-a45c-2e891f272b50` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文在大规模语言模型内部识别并学习了情感的价值-唤醒（VA）子空间，并通过对该子空间进行激活 steering，实现了对生成文本情感以及拒绝、谄媚等安全行为的可控调节；

**💡 创新点**

创新点在于将离散情绪类别与连续的 VA 维度解耦，利用 PCA 与岭回归构造低维 VA 子空间，展示该子空间能跨多种模型实现多行为的可逆、单调控制，并提出“词汇中介”机制解释情感基控制的作用；

**🔧 技术方法**

技术手段包括情感 steering 向量（对比均值）、主成分分析、岭回归学习 VA 轴、激活 steering（在生成时添加向量）、VADER 与 VAD‑BERT 评估、logit clamping、logit lens、MLP 神经元对齐等；

**📊 数据集**

数据集涵盖 211k 的 GoEmotions 文本、44k 的 NRC‑VAD 词典、模型自报 VA 分数、OKTest/HarmBench/XSTest（拒绝）以及 NLP Survey/PhilPapers/Political Typology（谄媚）等；模型采用 Llama‑3.1‑8B‑Instruct、Qwen3‑8B、Qwen3‑14B；

**📈 对比分析**

通过与随机方向和单一情绪向量对比，VA steering 在拒绝率上实现了高达 80% 以上的单调控制，谄媚率可逆降幅达 31pp，跨模型表现一致；与人类 VA 评分的相关性高达 r≈0.71（价值）和 r≈0.23（唤醒）；OOV 率低、核心能力基本保持；

**⚠️ 局限性**

局限性包括评估指标（VADER/VAD‑BERT）不完备、主要关注词汇层面可能忽视更高层机制、VA 子空间可能无法覆盖所有行为、对不同模型的通用性尚未充分验证，且该技术具备潜在规避安全防护的双重使用风险。

---

## 434. BAS: A Decision-Theoretic Approach to Evaluating Large Language Model Confidence

**arXiv ID:** 2604.03216 | [PDF](https://arxiv.org/pdf/2604.03216v1)

**作者:** Sean Wu `[一作]` (University of Oxford), David A. Clifton `[通讯]` (University of Oxford)

**通讯引用:** 14125 | [OpenAlex ID](https://openalex.org/A5040302008)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

提出了行为对齐得分（BAS）——一种基于决策理论的指标，用来评估大型语言模型在回答或拒绝时的置信度可靠性，并构建了包含多模型、多任务的置信度基准。

**💡 创新点**

创新点在于：①将置信度评估转化为明确的答复-拒绝效用模型，BAS 成为一种“选择性预测”的严格的决策理论得分；②BAS 对高置信度错误施加对数惩罚，形成不对称的过度自信惩罚；③通过BAS与传统 ECE、AURC、log‑loss 等指标的对比，揭示了这些指标在捕捉置信度的决策层影响上的局限性。

**🔧 技术方法**

技术方法包括：基于概率论的效用积分推导出闭式表达式、对置信度进行自下而上的统计分析、使用等距采样评估多模型的BAS、对比多种置信度获取方式（直接、self‑reflection、top‑k、top‑k + self‑reflection）以及后置的等距回归校准。

**📊 数据集**

数据集涵盖：SimpleQA（简短事实问答）、MedQA（医学高风险问答）和AIME（多步数学推理），三者分别用于评估不同难度与知识缺失场景下的置信度可靠性。

**📈 对比分析**

与现有 ECE、AURC、log‑loss 等指标对比，BAS 能更明显地区分高置信度错误。实验显示：更大、更精准的模型往往得到更高的 BAS；但即使是前沿模型在开放式任务中仍会出现严重的过度自信；相对地，top‑k 置信度获取和后置校准可显著提升 BAS，表现优于默认直接提示。

**⚠️ 局限性**

局限性包括：①BAS 依赖于均匀风险阈值假设，实际应用需根据风险分布定制权重；②对置信度的评估仅基于文本级输出，无法利用内部激活或采样信息；③在极端高置信度错误时 BAs 会趋向负无穷，对模型的鲁棒性评估存在数值不稳定；④基准覆盖的任务与数据集有限，尚未在所有真实应用场景中验证。

---

## 435. From Gaussian Fading to Gilbert-Elliott: Bridging Physical and Link-Layer Channel Models in Closed Form

**arXiv ID:** 2604.03160 | [PDF](https://arxiv.org/pdf/2604.03160v1)

**作者:** Bhaskar Krishnamachari `[一作]` (University of Southern California), Victor Gutierrez `[通讯]` (University of Southern California)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5a41884c-404f-4688-a89c-aa238c10fe68` `14d48e9d-0069-4ad9-996a-1d5968216998`

**🎯 论文内容**

本文给出了物理层高斯衰落模型与链路层Gilbert–Elliott（GE）模型之间的闭式桥梁：通过对任意平稳高斯过程在离散槽边界处阈值化，推导出GE一跳转移概率，并进一步得到期望持久时间的闭式表达式。

**💡 创新点**

创新点在于：①首次以Owen的T函数给出任意阈值下的闭式GE参数；②在对称阈值时得到极简的弧正弦公式；③解析推导出不同核（平滑的平方指数核与粗糙的指数核）对GE持久时间的标度差异（线性 vs. √T_c）；④提出并解释Markov gap与运行长度TV距离两种诊断指标可出现相反趋势的原因；⑤提供大规模Monte‑Carlo验证，证明闭式结果在多种核与阈值下均高度精确。

**🔧 技术方法**

使用了高斯过程理论、协方差核、Owen的T函数、弧正弦恒等式、二维正态分布的边界积分、总变差距离、Markov间隙分析以及大规模随机模拟。

**📊 数据集**

数据集为合成的平稳高斯序列（σ=1，D=1，阈值S=0或其他值），长度为1200，重复250次，覆盖多种协方差核（平方指数与指数）和多种T_c/D比值。没有使用真实无线信道测量数据。

**📈 对比分析**

与Monte‑Carlo模拟结果比较，转移概率与持久时间的误差均在2%以内；第一阶GE链在单步统计和持久时间预测上表现优异，但在较大T_c/D时无法准确再现长时序统计；引入第二阶Markov模型能显著降低运行长度的TV距离，说明二阶记忆能补偿大部分高阶相关性。总体而言，闭式桥梁在理论与数值上均表现出色。

**⚠️ 局限性**

局限性包括：①仅适用于平稳零均值高斯衰落；②仅给出一跳转移概率，无法完全捕捉高阶相关性；③对较大T_c/D时，第一阶GE模型在运行长度统计上失准；④需预先知道协方差核或其相关系数ρ；⑤对非高斯或非平稳衰落情形的推广尚未给出。

---

## 436. CoME-VL: Scaling Complementary Multi-Encoder Vision-Language Learning

**arXiv ID:** 2604.03231 | [PDF](https://arxiv.org/pdf/2604.03231v1)

**作者:** Ankan Deria `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Salman Khan `[通讯]` (Mohamed bin Zayed University of Artificial Intelligence)

**通讯引用:** 12022 | [OpenAlex ID](https://openalex.org/A5000300751)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `edb9d762-f411-4838-a852-f2d638b018db` `9ce7179e-700c-4310-ac2b-91df50ded46e` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出CoME‑VL框架，融合对比学习的SigLIP和自监督的DINO视觉编码器，实现多层级融合以提升视觉理解与定位；

**💡 创新点**

创新点包括熵导向层选择、正交约束多层混合消除冗余、RoPE增强的跨注意力对齐不同分辨率特征，以及可插入解码器LLM的模块化设计；

**🔧 技术方法**

使用多编码器融合、熵度量、正交层（OL）、RoPE跨注意力、对比预训练（CLIP/ SigLIP）、自监督预训练（DINO）、decoder‑only LLM（Qwen2‑7B）等技术；

**📊 数据集**

在PixMo（计数、指向、图表、图示、表格、其他）与RefCOCO（定位）数据集上进行训练与评测；

**📈 对比分析**

与Molmo、LLaVA、Intern‑VL、Qwen‑VL等基线在PixMo和RefCOCO上对比，CoME‑VL平均提升约+4.9%理解、+5.4%定位；PixMo计数87.83%、指向58.56%/75.94%，RefCOCO 92.57/95.36/90.51，显著优于基线；

**⚠️ 局限性**

推理时间略增（从1.26s提升至1.52s/样本），尽管低于直接拼接特征的COMM，但仍存在额外计算开销；多编码器融合对层范围与维度选择敏感，需要进一步优化。

---

## 437. The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report

**arXiv ID:** 2604.03198 | [PDF](https://arxiv.org/pdf/2604.03198v1)

**作者:** Bin Ren `[一作]`, Supavadee Aramvith `[通讯]`

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `64443552-63e0-44b5-906f-d90fe95c5a1b` `8d10c613-917e-4880-9716-17789f50e119` `90291a0e-9d36-4a08-9a16-89ce846d923f` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

对NTIRE 2026高效单幅超分辨率挑战及15支有效参赛团队的方案进行综述与对比

**💡 创新点**

创新点在于：①提出SPANV2通过学习式注意力与自定义CUDA融合显著提升推理速度；②多种压缩与蒸馏策略（剪枝+多阶段蒸馏、LoRA、知识蒸馏与空间亲和度对齐）实现参数与FLOPs大幅削减；③引入Mamba门控、频域分解与双频谱交互等新结构，兼顾局部卷积与全局状态空间建模；④挑战评估权重将运行时提升至核心指标，推动硬件友好设计。

**🔧 技术方法**

主要技术包括：网络剪枝、低秩LoRA与MuAdamW优化、知识蒸馏（含输出、边缘、亲和度损失）、重新参数化（多分支→单卷积）、自定义CUDA融合核、MambaGate、DWMamba、FFT/频域损失、动量EMA、学习率余弦退火、以及多阶段增量训练。

**📊 数据集**

使用DIV2K（800/100/100）和LSDIR（84,991/1,000/1,000）两大数据集，训练集涵盖外部Flickr2K，验证集包含100+100图像，测试集为200图像，所有LR均为×4双三次插值下采样。

**📈 对比分析**

评比标准为Runtime、FLOPs、参数三项，得分采用指数加权公式（w1=0.8,w2=0.1,w3=0.1）。XiaomiMM在Runtime上夺冠，随后是DISP、BOE_AIoT；ZenoSR在参数和FLOPs上最优但Runtime相对较高；整体PSNR在27 dB左右，达到26.99 dB阈值，显示高效SR仍可保持竞争性质量。

**⚠️ 局限性**

局限性：1）Runtime与参数/FLOPs并非单向相关，低参数模型往往推理慢；2）Mamba及全局状态空间模型在本次挑战中未能显著降低Runtime；3）过度剪枝/蒸馏易导致鲁棒性下降，尤其在非标准噪声或真实降质图像上；4）硬件依赖的CUDA融合需要在不同GPU上进一步验证；5）挑战目标仍以PSNR为衡量，未覆盖感知质量与泛化能力。

---

## 438. Causal Inference for Quantifying Noisy Neighbor Effects in Multi-Tenant Cloud Environments

**arXiv ID:** 2604.03145 | [PDF](https://arxiv.org/pdf/2604.03145v1)

**作者:** Philipe S. Schiavo `[一作]` (Federal University of Espírito Santo), Flávio de Oliveira Silva `[通讯]` (University of Minho)

**通讯引用:** 784 | [OpenAlex ID](https://openalex.org/A5061596292)

**关键词:** `51726dea-4812-4aef-b722-f01e3ca750d2` `aeb1d087-87bb-48bf-8e0e-d19fc2260534` `5a41884c-404f-4688-a89c-aa238c10fe68` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

在Kubernetes多租户环境中，通过受控实验与多阶段因果推理，量化并因果归因噪声邻居对同址工作负载的性能降级。

**💡 创新点**

结合Cohen's d效应量、Granger因果分析和ECDF“降级签名”，实现对噪声邻居影响的定量评估、因果证明和资源争用模式的可解释诊断。

**🔧 技术方法**

控制实验、时间序列差分、ADF平稳性检验、AIC滞后选择、Granger因果检验、ECDF分布特征提取等技术。

**📊 数据集**

基于10轮受控实验的Kubernetes测试集，使用Sysbench、Memtier、FIO、Uperf等基准产生的CPU、内存、磁盘和网络指标。

**📈 对比分析**

通过与基线对比、计算效应大小及置信区间，发现CPU、内存工作负载在噪声邻居作用下平均下降56%+，磁盘I/O最高降幅67.6%，Granger因果链接增幅75%，验证了方法在不同资源争用场景下的显著性。

**⚠️ 局限性**

实验环境单节点、禁用DVFS、使用合成基准，缺乏多节点、真实生产工作负载和动态伸缩的验证，限制了对真实云环境的直接推广。

---

## 439. Enhancing Robustness of Federated Learning via Server Learning

**arXiv ID:** 2604.03226 | [PDF](https://arxiv.org/pdf/2604.03226v1)

**作者:** Van Sy Mai `[一作]` (National Institute of Standards and Technology), Dipankar Maity `[通讯]` (University of North Carolina at Charlotte)

**通讯引用:** 953 | [OpenAlex ID](https://openalex.org/A5090451555)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c84dae5d-5273-4348-85a7-b44cb586b4df` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

本文提出利用服务器学习（Server Learning）与更新过滤（Client Update Filtering）结合几何中位数聚合，提升联邦学习在非 IID 数据和 Byzantine 攻击下的鲁棒性。

**💡 创新点**

创新点在于：①将服务器端有限且分布可能不同的数据既用于模型正则化又用于主动过滤恶意更新；②结合几何中位数聚合与角度/损失基过滤，实现即使恶意客户端比例超过 50% 也能保持“诚实多数”；③通过实验验证其在高攻击比例下仍能取得显著准确率提升。

**🔧 技术方法**

使用的技术包括：服务器端梯度学习（Server Learning）、角度过滤、损失基过滤、几何中位数聚合、范数裁剪以及基于 Weiszfeld 算法的近似求解。

**📊 数据集**

实验数据集为 EMNIST（使用 900 个合成字符样本作为服务器数据）与 CIFAR‑10（使用 900 张 STL‑10 图像作为服务器数据），客户端数据通过 Dirichlet 分布划分并呈现不同程度的非 IID。

**📈 对比分析**

与传统 FedAvg、仅过滤、仅服务器学习等方法比较，RoFSL 在恶意客户端比例为 30%–60% 时，测试准确率从 10%–30% 提升至 60%–80%，尤其在 60% 攻击时，其他方法收敛失败，而 RoFSL 仍能实现学习。

**⚠️ 局限性**

局限性包括：需手动调参（γ、过滤阈值、θ 等）；服务器数据与客户端数据分布差异过大可能导致过滤失效；算法计算复杂度高（几何中位数聚合和服务器本地训练）；目前缺乏严谨理论保证，未来需进一步理论分析和自适应参数调整。

---

## 440. Safety-Critical Centralized Nonlinear MPC for Cooperative Payload Transportation by Two Quadrupedal Robots

**arXiv ID:** 2604.03200 | [PDF](https://arxiv.org/pdf/2604.03200v1)

**作者:** Ruturaj S. Sambhus `[一作]` (Virginia Tech), Kaveh Akbari Hamed `[通讯]` (Virginia Tech)

**通讯引用:** 1135 | [OpenAlex ID](https://openalex.org/A5066501467)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `5b4c1114-4a70-478e-9921-2514ee03850d` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

设计并实现了一种安全关键的集中非线性模型预测控制（NMPC）框架，用于两只四足机器人协同运输载荷，并在障碍物丰富的环境中实现碰撞避免和动态可行性；

**💡 创新点**

创新点在于：①将交互力作为决策变量在DAE约束下求解，保持模型结构可解；②使用高阶控制屏障函数（HOCBF）直接嵌入NMPC，实现机器人与载荷对障碍的实时碰撞避免；③在硬件实验中验证了对载荷质量不确定和外部扰动的鲁棒性；

**🔧 技术方法**

采用离散时间SRB动力学、非线性DAE约束的NMPC、控制屏障函数（HOCBF）、CasADi+IPOPT求解器、层级控制结构（高层NMPC+低层全身控制器）以及基于LiDAR的定位；

**📊 数据集**

未使用公开数据集，所有实验均在实验室使用两台Unitree Go2四足机器人和自建的障碍物配置进行硬件验证；

**📈 对比分析**

与之前仅考虑两体模型且缺少安全约束的控制方法相比，本框架可实现约74.7%机器人质量的载荷运输；在实验中NMPC求解平均11 ms，能以60 Hz实时运行；所有高阶CBF约束保持非负，证明了安全性，且在载荷质量不确定和外部推力下仍能保持稳定与安全；

**⚠️ 局限性**

局限性包括：仅针对两台机器人，未实现分布式或可扩展的多机器人框架；对动态障碍、非平地或复杂地形的适应性尚未验证；对极端载荷质量误差或大扰动的鲁棒性有限；实验依赖高质量传感与估计，未探讨视觉或感知失效情况。

---

## 441. Chart-RL: Policy Optimization Reinforcement Learning for Enhanced Visual Reasoning in Chart Question Answering with Vision Language Models

**arXiv ID:** 2604.03157 | [PDF](https://arxiv.org/pdf/2604.03157v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab`

---

## 442. Multi-View Video Diffusion Policy: A 3D Spatio-Temporal-Aware Video Action Model

**arXiv ID:** 2604.03181 | [PDF](https://arxiv.org/pdf/2604.03181v1)

**作者:** Peiyan Li `[一作]` (New Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences), Tieniu Tan `[通讯]` (New Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences)

**关键词:** `a1c26042-88d3-4e76-b403-2055e0dfc5c7` `c7913869-b026-40e7-b14b-dfd72dc55ea0` `edb9d762-f411-4838-a852-f2d638b018db` `ba576bd1-e51d-44e8-8077-fc943b333c93` `c7dc7075-6ff9-4c1b-b9c1-b644a40c5ab4` `ef89cc5f-e375-48ac-9691-51e1cf81ed3f` `d4b5b188-bf40-4c81-9f3f-3aecea92dd61`

**🎯 论文内容**

本文提出了一种基于多视角视频扩散的机器人操作策略，能够同时建模环境的三维空间结构和时间演化，并将预测的热图和视频解码为机器人动作。

**💡 创新点**

创新点在于将三维感知通过多视角投影嵌入视频基础模型，并通过统一的扩散网络同时预测未来视频与热图，实现对环境动态的先验建模和连续动作输出。

**🔧 技术方法**

采用多视角投影、热图编码、视频扩散变换器（基于Wan2.2）以及轻量化旋转与抓取预测网络，结合LoRA微调。

**📊 数据集**

在Meta-World仿真数据集和Frank A Research 3真实机器人搭配ZED摄像头的实测数据上进行评估。

**📈 对比分析**

与行为克隆、基于视频预测、3D点云、以及视觉-语言-动作模型等多项最先进基线相比，实验表明在仅十条演示轨迹下即可达到约90%平均成功率，鲁棒性强、泛化性好。

**⚠️ 局限性**

主要限制是推理速度较慢（约4.6秒生成24帧动作块），对高频控制不友好，并且热图分辨率有限导致细粒度动作误差。

---

## 443. EffiMiniVLM: A Compact Dual-Encoder Regression Framework

**arXiv ID:** 2604.03172 | [PDF](https://arxiv.org/pdf/2604.03172v1)

**作者:** Yin-Loon Khor `[一作]` (Universiti Malaya), Yan Chai Hum `[通讯]` (Universiti Tunku Abdul Rahman)

**通讯引用:** 1317 | [OpenAlex ID](https://openalex.org/A5040863689)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `a2602d71-93ab-4bad-974b-672788df8193` `64443552-63e0-44b5-906f-d90fe95c5a1b` `85b3479c-4bb5-42e0-8cca-2f9268bd338f` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `90291a0e-9d36-4a08-9a16-89ce846d923f` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

提出一种轻量化双编码器视觉-语言回归模型EffiMiniVLM，专门用于冷启动场景下的商品质量评分。

**💡 创新点**

在仅使用Amazon Reviews 2023数据集、无外部数据的前提下，利用EfficientNet‑B0和MiniLMv2实现了与大型模型相当的性能，同时大幅降低参数量与算力。

**🔧 技术方法**

采用EfficientNet‑B0作为图像编码器，MiniLMv2-L6-H384作为文本编码器，轻量化多模态融合并加上加权Huber损失，权重基于商品评价数量。

**📊 数据集**

仅使用Amazon Reviews 2023数据集，训练时采用20%样本（约1.6M条），每个样本包含图像、类别、标题、特征和描述。

**📈 对比分析**

与同类基准模型相比，EffiMiniVLM在CES指标上取得0.40分，资源成本最低（0.1），在参数27.7M、6.8 GFLOPs的条件下，显著高于同等规模模型，且可通过增加数据量进一步提升。

**⚠️ 局限性**

受限于模型容量与数据规模，尚未充分挖掘潜力；缺乏外部知识蒸馏、预训练融合等更先进技术，且在更大数据集上的性能仍有提升空间。

---

## 444. An Algebraic Method for Full-Rank Characterization in Binary Linear Coding

**arXiv ID:** 2604.03168 | [PDF](https://arxiv.org/pdf/2604.03168v1)

**作者:** Mingyang Zhu `[一作]` (Chinese University of Hong Kong), Xiao-Shan Gao `[通讯]` (Chinese Academy of Sciences)

**通讯引用:** 3576 | [OpenAlex ID](https://openalex.org/A5017479931)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `5b4c1114-4a70-478e-9921-2514ee03850d`

**🎯 论文内容**

本文提出了一种基于特征集（CS）的方法，用于推导二进制线性编码中符号矩阵的满秩等价条件，并开发了名为BCSFR的算法来高效地推导这些条件。

**💡 创新点**

创新点在于将满秩问题与零分解问题建立了理论联系，并通过BCSFR算法显著简化了编码方案的优化问题。

**🔧 技术方法**

使用了特征集方法（CS方法）来处理多项式方程系统，并在二进制域上进行计算。

**📊 数据集**

实验中使用了多种线性编码问题的实例，包括线性网络编码（LNC）和分布式存储编码（LRC）。

**📈 对比分析**

与现有方法相比，BCSFR算法在处理满秩约束时表现出更高的效率，能够在复杂度上显著降低优化问题的求解难度，实验结果表明其在实际应用中能够找到有效的编码方案。

**⚠️ 局限性**

限制在于该方法主要针对二进制域，未来的工作可以扩展到其他有限域，并处理更多类型的线性编码约束。

---

## 445. Prosocial Persuasion at Scale? Large Language Models Outperform Humans in Donation Appeals Across Levels of Personalization

**arXiv ID:** 2604.03202 | [PDF](https://arxiv.org/pdf/2604.03202v1)

**作者:**  `[一作]` `[通讯]`, 

**关键词:** `b851fbf0-9c24-4149-bb85-0c22287fee6f`

---

## 446. DSBD: Dual-Aligned Structural Basis Distillation for Graph Domain Adaptation

**arXiv ID:** 2604.03154 | [PDF](https://arxiv.org/pdf/2604.03154v1)

**作者:** Yingxu Wang `[一作]` (Mohamed bin Zayed University of Artificial Intelligence), Nan Yin `[通讯]` (City University of Hong Kong)

**通讯引用:** 1375 | [OpenAlex ID](https://openalex.org/A5047501260)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `f7dab867-23a8-4241-85e9-4ba79c6402f9` `3f18e8e3-0266-457c-8567-9039b6d2394d` `8d10c613-917e-4880-9716-17789f50e119` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

本文提出一种双对齐结构基蒸馏（DSBD）框架，用于图域自适应，在源域学习语义标签的同时，构造可微的结构基，并通过几何一致性与谱一致性双重对齐，最终在目标域训练新的GNN以消除源域结构偏差。

**💡 创新点**

创新点包括：① 通过连续概率变量参数化邻接矩阵，构建可微结构基；② 结合拓扑矩矩匹配（几何一致性）和Dirichlet能量校准（谱一致性）实现跨域结构对齐；③ 在推理阶段采用结构校正的解耦式训练，彻底消除源域结构记忆。

**🔧 技术方法**

技术手段包括：双层优化（结构基外层 + 代理模型内层）、概率图结构学习、拓扑矩矩匹配、Dirichlet能量正则、重置模型训练、GNN（如3层GIN）等。

**📊 数据集**

使用了多种图形与图像基准：MNIST、CIFAR-10（图像边缘密度域迁移）、PROTEINS、Mutagenicity、NCI1、FRANKENSTEIN、ogbg-molhiv、DD、BZR、BZR_MD、COX2、COX2_MD、Spurious-Motif 等。

**📈 对比分析**

与 19+ 传统图域自适应与图蒸馏方法（如G-CRD、MuGSI、TGS、LAD-GNN、SGDA、StruRW、A2GNN、PA-BOTH、GAA、TDSS 等）对比，DSBD 在大部分结构、特征及相关性偏移任务上均取得显著提升（例如 MNIST 边缘密度迁移平均提升 5–10% 以上）。

**⚠️ 局限性**

局限性包括：① 对 K（结构基数量）和 λ₁/λ₂ 的超参数较敏感，需要调优；② 需要额外的双层优化和梯度传递，计算成本较高；③ 目前主要针对图分类任务，针对大规模图或动态图的适应性尚待验证。

---

## 447. Beyond the Parameters: A Technical Survey of Contextual Enrichment in Large Language Models: From In-Context Prompting to Causal Retrieval-Augmented Generation

**arXiv ID:** 2604.03174 | [PDF](https://arxiv.org/pdf/2604.03174v1)

**作者:** Prakhar Bansal `[一作]`, Shivangi Agarwal `[通讯]` (Indian Institute of Information Technology Delhi)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `a68d3170-c4b6-45e7-b3b6-7e2d411d5656` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `3a4a0352-9c3f-40a0-98ff-bde88bec2bbe` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

本文对大型语言模型（LLM）的增广策略进行综述，构建了从提示到 RAG、GraphRAG、CausalRAG 的统一上下文丰富轴，并提出文献筛选、证据分级、主张审计等方法学流程；

**💡 创新点**

创新点在于（1）将增广范式映射为单一结构化上下文轴，提供统一分类；（2）引入系统化的文献筛选与证据分级框架；（3）构建主张审计表与决策框架，帮助实践者评估可信部署；

**🔧 技术方法**

使用的技术包括：结构化文献检索与筛选、三等级证据评估、跨论文对比矩阵、主张审计表、决策树框架，结合对提示、RAG、GraphRAG、CausalRAG 的实验结果进行量化与质性归纳；

**📊 数据集**

数据集主要取自已有的公开基准，如 GSM8K、Natural Questions、OpenAlex（单篇论文的摘要/引言/全文切片）等，用于引用各方法在特定任务上的量化指标；

**📈 对比分析**

比较方法采用同一论文内部的对照实验（如 CoT vs 标准提示、RAG vs 仅闭书模型、GraphRAG/ CausalRAG 的切片比较），展示了显著提升：CoT 提升 Solve Rate 39 点；RAG 在 Natural Questions 上提高 EM 10 点；GraphRAG/ CausalRAG 在复合得分上分别提升 21–26 点；CausalRAG 的 k‑s 取值 5 时的得分提升 0.29；

**⚠️ 局限性**

局限性包括：跨论文实验设置不统一导致数值可比性受限；因果提取仍昂贵且误报多；缺乏统一的因果评估指标与动态维护机制；多语言、跨领域适用性不足；对真实部署的安全与可靠性验证仍待进一步探索。

---

## 448. VOSR: A Vision-Only Generative Model for Image Super-Resolution

**arXiv ID:** 2604.03225 | [PDF](https://arxiv.org/pdf/2604.03225v1)

**作者:** Rongyuan Wu `[一作]` (Hong Kong Polytechnic University), Lei Zhang `[通讯]` (Hong Kong Polytechnic University)

**通讯引用:** 106905 | [OpenAlex ID](https://openalex.org/A5100433899)

**关键词:** `9473a256-bb9c-4876-84c8-23d8ab9b6fd9` `e1a5312d-25ae-4d44-8d74-dde5f79b5ab4` `edb9d762-f411-4838-a852-f2d638b018db` `7bbdcbec-2caa-4c7a-b120-9489f11b7043` `ba576bd1-e51d-44e8-8077-fc943b333c93` `90291a0e-9d36-4a08-9a16-89ce846d923f`

**🎯 论文内容**

提出了纯视觉生成式超分辨率框架VOSR，在不依赖文本-图像预训练的前提下，实现高质量图像恢复

**💡 创新点**

核心创新包括：1）利用预训练视觉编码器提取语义导向特征，提供空间对齐的语义指导；2）重新设计无监督引导策略，将完全无条件分支改为部分条件分支，实现以恢复为导向的指导；3）将多步模型蒸馏为单步模型，兼顾质量与推理效率

**🔧 技术方法**

技术手段包括：基于潜在扩散Transformer的噪声去除；双条件设计（结构+视觉语义）；修订的classifier‑free guidance；蒸馏技术（one‑step distillation）以及VAE、DINO等预训练组件

**📊 数据集**

训练数据为约100M Web 图像（无文本标注），通过 Real‑ESRGAN 合成 LR‑HR 对；测试集涵盖 LSDIR、ScreenSR、RealSR 等合成与真实世界图像

**📈 对比分析**

与多步与单步的基准方法（ResShift、StableSR、PASD、SeeSR、DiT4SR、SinSR、OSEDiff、PiSA‑SR）进行对比；VOSR 在多步/单步场景下在感知质量（LPIPS、DISTS、MUSIQ等）与结构保真度（PSNR、SSIM）上均与或优于多数 T2I‑基方法，同时参数量和推理时间更低，训练成本约为 T2I‑基方法的十分之一

**⚠️ 局限性**

局限性包括：模型规模与训练数据相对 T2I 基础模型（10B 级别）仍显不足，未来需扩展数据与模型容量，并探讨在更广泛图像恢复任务中的迁移

---

## 449. Coupled Control, Structured Memory, and Verifiable Action in Agentic AI (SCRAT -- Stochastic Control with Retrieval and Auditable Trajectories): A Comparative Perspective from Squirrel Locomotion and Scatter-Hoarding

**arXiv ID:** 2604.03201 | [PDF](https://arxiv.org/pdf/2604.03201v1)

**作者:** Maximiliano Armesto `[一作]` (Taller Technologies), Christophe Kolb `[通讯]` (Taller Technologies)

**关键词:** `0536b7b3-4271-4e10-9b76-1f66fc457fab` `d0f189e1-0834-4ff4-b4e8-f515263ef669` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `596fe7ac-9d40-46e0-a8e6-ee59d94fc35e` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

通过对松鼠生态行为的叙事比较，提出了控制-记忆-验证耦合问题，并给出三条AI假设（快速反馈、结构化记忆、内置验证）以及下游的角色分化假设；

**💡 创新点**

将松鼠的隐藏动态控制、散布记忆与观察者感知整合为统一框架（SCRAT模型），并为该耦合问题设计可验证的benchmark体系，首次以生态案例驱动AI系统设计；

**🔧 技术方法**

使用叙事比较方法、推断金字塔、层次化部分可观测控制模型（SCRAT）以及在软件交付系统Chiron中实现的结构化记忆与检索技术；

**📊 数据集**

基于松鼠行为实验（红松鼠、灰松鼠、狐狸松鼠等）和文献综述，以及Chiron实验项目（Bank、ACAS、Mortgage、Portfolio）中的项目规模、代码量和问题负载数据；

**📈 对比分析**

设计四类benchmark（隐藏动态控制、缓存检索、观测者敏感行动、角色分化），通过对比具备快速反馈、结构化记忆和内置验证的系统与基线，发现结构化记忆可显著降低问题率、提升首次发布覆盖率；

**⚠️ 局限性**

局限性包括：生态比喻与AI系统的直接对应有限，角色分化假设尚未通过大规模实验验证，verifier与生态测量不完全匹配，且缺乏在更大规模或多样化任务上的实证评估。

---

## 450. Learning the Signature of Memorization in Autoregressive Language Models

**arXiv ID:** 2604.03199 | [PDF](https://arxiv.org/pdf/2604.03199v1)

**作者:** David Ilić `[一作]` (JetBrains Research), Evgeny Grigorenko `[通讯]` (JetBrains Research)

**关键词:** `243a8f53-c1b4-4939-9b96-9653425e9d86` `6215c339-3735-4be3-8a07-5bbb7004712d` `edb9d762-f411-4838-a852-f2d638b018db` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c`

**🎯 论文内容**

针对细调后的语言模型，提出了一种可迁移的学习型成员推断攻击LT‑MIA。

**💡 创新点**

创新点在于利用细调过程天然生成的无限标注数据，构建跨架构零样本迁移的学习型攻击，从而突破传统手工设计指标的瓶颈。

**🔧 技术方法**

方法采用基于每token统计的特征序列，使用轻量级Transformer对序列进行分类，并通过多样化的训练组合实现泛化。

**📊 数据集**

训练数据来自10个Transformer模型在3个文本语料库上细调的30个模型-数据组合；评估数据包括AG News、WikiText‑103、XSum、Swallow‑Code以及Mamba、RWKV、RecurrentGemma等非Transformer架构。

**📈 对比分析**

与传统无训练的损失阈值、参考校准等方法相比，LT‑MIA在Transformer上平均AUC达0.908，零样本迁移到Mamba、RWKV、RecurrentGemma分别达到0.963、0.972、0.936；在低FPR（0.1%）下TPR提升至2.8×以上。

**⚠️ 局限性**

局限性包括需获得同架构的预训练参考模型和完整词表logits；当参考模型已对目标文本具备高似然时检测难度增大；方法尚未验证在预训练或RLHF等不同训练范式下的适用性。

---

## 451. Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models

**arXiv ID:** 2604.03179 | [PDF](https://arxiv.org/pdf/2604.03179v1)

**作者:** Gengwei Zhang `[一作]` (University of North Carolina at Chapel Hill), Tianlong Chen `[通讯]` (University of North Carolina at Chapel Hill)

**通讯引用:** 4171 | [OpenAlex ID](https://openalex.org/A5103073431)

**关键词:** `00521103-b308-4295-8635-1bbb9135d4d9` `c5260876-9a54-48ae-a63a-8fa6d6ddb799` `edb9d762-f411-4838-a852-f2d638b018db` `e4f91bb3-83db-4b7d-994e-d8bf54b7b1a8` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `6c1af392-8b9e-4e11-bd3d-9d44e98a6e3b`

**🎯 论文内容**

构建了Hallucination-as-Cue框架，利用视觉或文本信息的三种特定损坏（Blank Image、Random Image、Text Removal）进行RL后训练和推理；

**💡 创新点**

首次把模型幻觉作为诊断信号，用损坏数据评估RL对多模态推理的真实影响，并发现更大模型能从幻觉中获益；

**🔧 技术方法**

采用GRPO（Group Relative Policy Optimization）强化学习，在Qwen2.5-VL-3B/7B模型上实施；

**📊 数据集**

在Geometry3K、MMR1-V0、CLEVR等多模态推理数据集上训练，并在MathVision、MathVerse、MathVista、We-Math等基准上评估；

**📈 对比分析**

与原始模型及标准GRPO相比，幻觉诱导训练在大模型上可提升或等同于标准训练，在多数基准平均提升约5–6%，部分任务甚至超越标准训练；

**⚠️ 局限性**

缺点包括对幻觉机制的深入理解不足、仅关注GRPO类RL未覆盖其他推理范式，对不同模态信息的依赖仍需进一步研究。

---

## 452. High-Dimensional Signal Compression: Lattice Point Bounds and Metric Entropy

**arXiv ID:** 2604.03178 | [PDF](https://arxiv.org/pdf/2604.03178v1)

**作者:** A. Iosevich `[一作]` (University of Rochester), E. Wyman `[通讯]` (Binghamton University)

**通讯引用:** 54 | [OpenAlex ID](https://openalex.org/A5064320363)

**关键词:** `2704f255-0c84-4173-b83c-0e9a3dbea232` `fede83ac-7505-405f-ab37-e7284695c47f`

**🎯 论文内容**

研究在满足能量约束且各坐标量化精度不同的情况下，最坏情况信号压缩的极限；

**💡 创新点**

在平衡精度假设下，首次给出对高维维度的显式熵上界，并通过改进的格点计数技术实现非渐近精确的上界；

**🔧 技术方法**

利用改良的Landau格点计数公式、Olenko的贝塞尔函数统一上界以及Abel求和，精确追踪维度k对计数的影响；

**📊 数据集**

无实验数据集，本文完全理论分析；

**📈 对比分析**

无比较实验，论文仅给出理论上界；

**⚠️ 局限性**

仅在平衡精度和能量与维度的关系满足特定约束时成立，且对极端非平衡精度或更大维度与能量比例的情况尚未涵盖。

---

## 453. BibTeX Citation Hallucinations in Scientific Publishing Agents: Evaluation and Mitigation

**arXiv ID:** 2604.03159 | [PDF](https://arxiv.org/pdf/2604.03159v1)

**作者:** Delip Rao `[一作]` (University of Pennsylvania), Chris Callison-Burch `[通讯]` (University of Pennsylvania)

**通讯引用:** 21511 | [OpenAlex ID](https://openalex.org/A5068508539)

**关键词:** `f53a5690-f5d8-493f-989c-dc46a1f99053` `ca90f54c-96fe-4d91-a7ad-6da6db91f7d2` `d4a8441d-3297-45fc-8ac0-20de12b80ddd` `edb9d762-f411-4838-a852-f2d638b018db` `c59129cc-0f1d-4fee-85d8-abbb7eea50d6` `b4bc56fa-9c97-45d8-ae70-e6cccdb8a275` `6b9ad54c-2d62-4a92-a500-d9cb644dd99c` `79276348-11e0-48e3-84bc-7ec231d0171c`

**🎯 论文内容**

评估了带 Web 搜索功能的 LLM 在生成 BibTeX 时的准确性，构建了 931 篇论文的基准，并验证了一种基于 Zotero Translation Server 的检索修正工具。

**💡 创新点**

创新点在于提出了版本感知、按引用计数层级划分的基准和新的六维错误分类，并证明两阶段检索+修订架构能显著提升准确率。

**🔧 技术方法**

技术手段包括使用 GPT‑5、Claude Sonnet‑4.6、Gemini‑3 Flash 等带搜索的 LLM，自动化评估管道、Zotero Translation Server 检索以及基于 LLM 的错误分类器。

**📊 数据集**

使用的数据集为四个学科（AI、医学、材料科学、量子计算）中的 931 篇论文，按热门、低引用、最新三层级划分，并包含多版本信息。

**📈 对比分析**

通过九字段逐级打分和错误类别统计进行比较，基线整体准确率为 83.6%，完全正确率为 50.9%；两阶段 clibib 集成后提升至 91.5%，完全正确率为 78.3%，且热门到最新论文准确率下降了 27.7 个百分点。

**⚠️ 局限性**

局限性包括仅覆盖英文、已被 OpenAlex/DBLP/PubMed/Zotero 收录的论文；检索覆盖率不足导致最近论文仍难修复；单轮工具调用受模型遵从性限制；评估工具判定一致性仅为 0.67 κ，存在误判风险。

---

## 454. Engineering Algorithms for Dynamic Greedy Set Cover

**arXiv ID:** 2604.03152 | [PDF](https://arxiv.org/pdf/2604.03152v1)

**作者:** Amitai Uzrad `[一作]` (Tel Aviv University), Amitai Uzrad `[通讯]` (Tel Aviv University)

**通讯引用:** 15 | [OpenAlex ID](https://openalex.org/A5006023758)

**关键词:** `350271b4-1c30-42d1-b8ce-110a550894ce` `5b4c1114-4a70-478e-9921-2514ee03850d` `64443552-63e0-44b5-906f-d90fe95c5a1b` `94d4fa07-b711-4bf6-b37a-13f8a4bb9c05`

**🎯 论文内容**

实现并评估了四种基于贪心的动态集合覆盖算法，填补了理论设计与实践效果之间的空白。

**💡 创新点**

首次对动态集合覆盖进行系统实验，简化了现有理论算法并提出了“robust”“local”“partial”“global”四种可调节的实现，探究了 β 这一折衷参数。

**🔧 技术方法**

采用 C++ 代码实现，基于静态贪心算法的层级维护，利用 β 参数平衡解质量与效率，使用性能剖面（performance profiles）对多项指标进行评估。

**📊 数据集**

使用 120 个真实超图实例（由静态实例转化为动态序列，且频率 f ≥ ln n），覆盖不同规模与高频特征。

**📈 对比分析**

通过 40,000 条独立实验，按增量覆盖大小、更新时间与回报量三项幅度归一化并计算几何平均，最终发现：local 算法最快、global 方案质量最好、robust 与 partial 位于中间；β 越大则覆盖越大、更新时间与回报量越低。

**⚠️ 局限性**

限制：未完成与完全重算基准的全面对比（时间瓶颈）；仅考虑无权问题；未评估低频（f < ln n）下的原始基线与 primal‑dual 动态算法；β 取值需根据具体应用调节。

---

